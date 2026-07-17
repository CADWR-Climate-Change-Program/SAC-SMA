"""The daily forcing MASTER (Livneh-unsplit, non-detrended) — build / verify / cut.

The local WGEN NonDetrend-Unsplit statewide store
(``C:\\Users\\warnold_la\\Local\\WGEN_NonDetrend_Unsplit_Statewide``, one
``data_<lat>_<lon>`` ASCII per cell: year month day prcp tmax tmin, daily
1915-01-01..2018-12-31) is the PROVEN source of every committed historical
forcing store (verified 2026-07-16: ``prcp`` matches to float32 rounding and
``tavg`` == (tmax+tmin)/2 exactly for ``cdec15_grid``; the calsim stores match
to their 3-decimal write precision, 5e-3).  ``tminmax_livneh_percell.nc`` is
the same store, float32, with the ~0.05% of days carrying an inverted
(tmin > tmax) ASCII pair SORTED — the master applies the same fix (the sum,
and hence tavg, is unchanged).

**The x10 precipitation artifact**: the raw lineage carries misplaced-decimal
precipitation spikes — 197 (cell, day) pairs over 168 region cells, isolated
summer days (1916-07-01, 1954-08-28, 1974-07-08..10, 1976-08-15, 1980-07-02),
each EXACTLY 10x too large.  The CalSim-domain ingest corrected them (/10,
consistent across 9unimp/11obs/12rim); the cdec15 lineage kept them raw (146
of the cells are shared with the committed ``cdec15_grid`` store, which the
GA/dPL calibrations trained on — a pre-existing upstream inconsistency).  The
master stays bit-faithful to the RAW source; ``--scan-x10`` derives the
auditable correction table ``data/region/prcp_x10_artifacts.csv`` from the
committed calsim stores, ``--verify`` expects the calsim stores to differ by
exactly that table, and ``--cut`` applies the correction by default
(``--no-fix-x10`` opts out, reproducing the raw cdec15 convention).

This tool packs the region's cells (``data/region/grid_cells.csv``) into ONE
compressed NetCDF master kept on local disk (NOT the repo — repo policy is
compact processed layers only; the master is ~1 GB):

  python dataprep/wgen_forcing.py --build-master     # ASCII -> master nc
  python dataprep/wgen_forcing.py --scan-x10         # derive the artifact table
  python dataprep/wgen_forcing.py --verify           # master vs committed stores
  python dataprep/wgen_forcing.py --cut <name> --cells <csv> --out-dir <dir>
      # new-basin setup: emits historical_livneh_unsplit_<name>.nc (prcp+tavg)
      # + tminmax_livneh_percell_<name>.nc for the cells listed in <csv> (col
      # ``key``), in the committed cdec15_grid schema (coords key/time)

Default master path: ``D:\\sacsma-data\\forcing\\livneh_unsplit_nondetrend_daily_region.nc``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

WGEN_DIR = r"C:\Users\warnold_la\Local\WGEN_NonDetrend_Unsplit_Statewide"
GRID_CSV = "data/region/grid_cells.csv"
MASTER = r"D:\sacsma-data\forcing\livneh_unsplit_nondetrend_daily_region.nc"
X10_CSV = "data/region/prcp_x10_artifacts.csv"
DAYS = pd.date_range("1915-01-01", "2018-12-31", freq="D")
CALSIM_DOMAINS = ("9unimp", "11obs", "12rim")


def norm_key(k: str) -> str:
    lat, lon = str(k).split("_")
    return f"{round(float(lat), 5)}_{round(float(lon), 5)}"


def build_master(wgen_dir: str, grid_csv: str, out: str) -> None:
    grid = pd.read_csv(grid_csv)
    keys = grid["key"].astype(str).to_list()
    n = len(keys)
    arr = {v: np.empty((n, len(DAYS)), dtype=np.float32)
           for v in ("prcp", "tmin", "tmax")}
    for i, k in enumerate(keys):
        w = pd.read_csv(Path(wgen_dir) / f"data_{k}", sep=r"\s+", header=None,
                        names=["y", "m", "d", "pr", "tmax", "tmin"])
        if len(w) != len(DAYS):
            raise SystemExit(f"data_{k}: {len(w)} rows != {len(DAYS)}")
        if i == 0:
            t0 = pd.Timestamp(int(w.y.iloc[0]), int(w.m.iloc[0]), int(w.d.iloc[0]))
            t1 = pd.Timestamp(int(w.y.iloc[-1]), int(w.m.iloc[-1]), int(w.d.iloc[-1]))
            if (t0, t1) != (DAYS[0], DAYS[-1]):
                raise SystemExit(f"WGEN span {t0}..{t1} != {DAYS[0]}..{DAYS[-1]}")
        arr["prcp"][i] = w["pr"].to_numpy(np.float32)
        # ~0.05% of WGEN days carry an inverted (tmin > tmax) pair; the
        # committed tminmax sidecar sorts them (tavg unchanged) — same fix here
        tmn = w["tmin"].to_numpy(np.float32)
        tmx = w["tmax"].to_numpy(np.float32)
        arr["tmin"][i] = np.minimum(tmn, tmx)
        arr["tmax"][i] = np.maximum(tmn, tmx)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} cells read", flush=True)
    ds = xr.Dataset(
        {v: (("key", "time"), a) for v, a in arr.items()},
        coords={"key": keys, "time": DAYS},
        attrs={"title": "Livneh-unsplit non-detrended daily forcing master "
                        "(region: 15cdec_grid + 9unimp + 11obs + 12rim)",
               "source": wgen_dir,
               "note": "committed domain stores use tavg = (tmax+tmin)/2"},
    )
    enc = {v: dict(zlib=True, complevel=4, chunksizes=(1, len(DAYS)))
           for v in arr}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out, encoding=enc)
    print(f"wrote {out}: {n} cells x {len(DAYS)} days "
          f"({Path(out).stat().st_size / 1e9:.2f} GB)")


def _x10_table(x10_csv: str) -> pd.DataFrame | None:
    p = Path(x10_csv)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    df["key"] = df["key"].map(norm_key)
    return df


def _apply_x10(prcp: np.ndarray, keys: list[str], t: np.ndarray,
               x10: pd.DataFrame | None) -> int:
    """Divide the documented x10 artifact (cell, day) pairs by 10 in place."""
    if x10 is None:
        return 0
    ki = {k: i for i, k in enumerate(keys)}
    ti = {pd.Timestamp(v): j for j, v in enumerate(t)}
    n = 0
    for _, row in x10.iterrows():
        i, j = ki.get(row["key"]), ti.get(row["date"])
        if i is not None and j is not None:
            prcp[i, j] /= 10.0
            n += 1
    return n


def scan_x10(master_path: str, out_csv: str) -> None:
    """Derive the x10 precip-artifact table from the committed calsim stores.

    Records every (cell, day) where a committed calsim store equals master/10
    within write precision — the correction ``--cut`` applies and ``--verify``
    expects.  Any calsim mismatch that is NOT an exact x10 aborts."""
    master = xr.open_dataset(master_path)
    pairs: dict[tuple[str, pd.Timestamp], float] = {}
    for dom in CALSIM_DOMAINS:
        ref = xr.open_dataset(Path("data") / "calsim" / "forcing"
                              / f"historical_livneh_unsplit_{dom}.nc")
        mk = [norm_key(k) for k in np.asarray(ref["key"]).astype(str)]
        t = np.asarray(ref["time"])
        mp = master["prcp"].sel(key=mk, time=t).to_numpy().astype(np.float64)
        rp = ref["prcp"].to_numpy().astype(np.float64)
        ci, dj = np.where(np.abs(mp - rp) > 6e-3)
        ratio = mp[ci, dj] / np.maximum(rp[ci, dj], 1e-9)
        if not np.all(np.abs(ratio - 10.0) < 0.02):
            raise SystemExit(f"{dom}: non-x10 prcp mismatch found "
                             f"(ratio range {ratio.min()}..{ratio.max()})")
        for i, j in zip(ci, dj, strict=True):
            pairs[(mk[i], pd.Timestamp(t[j]))] = float(mp[i, j])
        ref.close()
    master.close()
    df = pd.DataFrame([{"key": k, "date": d.date(), "prcp_raw_mm": v}
                       for (k, d), v in sorted(pairs.items())])
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(df)} x10 artifact (cell, day) pairs over "
          f"{df['key'].nunique()} cells, days {sorted(set(df['date']))}")


def _compare(master: xr.Dataset, ref_path: Path, pairs: list[tuple[str, str]],
             tol: float, x10: pd.DataFrame | None = None) -> None:
    ref = xr.open_dataset(ref_path)
    rk = np.asarray(ref["key"]).astype(str)
    mk = [norm_key(k) for k in rk]
    sub = master.sel(key=mk)
    t = np.asarray(ref["time"])
    sub = sub.sel(time=t)
    worst = 0.0
    for mv, rv in pairs:
        if mv == "tavg":
            m = (sub["tmin"].to_numpy().astype(np.float64)
                 + sub["tmax"].to_numpy().astype(np.float64)) / 2.0
        else:
            m = sub[mv].to_numpy().astype(np.float64)
        nfix = _apply_x10(m, mk, t, x10) if mv == "prcp" else 0
        r = ref[rv].to_numpy().astype(np.float64)
        d = float(np.nanmax(np.abs(m - r)))
        worst = max(worst, d)
        print(f"  {ref_path.name} {rv}: max|d| = {d:.2e} "
              f"({'PASS' if d <= tol else 'FAIL'} @ tol {tol:g})"
              + (f" [{nfix} documented x10 corrections applied]" if nfix else ""))
    ref.close()
    if worst > tol:
        raise SystemExit(f"VERIFY FAILED for {ref_path}")


def verify(master_path: str, x10_csv: str) -> None:
    master = xr.open_dataset(master_path)
    root = Path("data")
    x10 = _x10_table(x10_csv)
    if x10 is None:
        print(f"  ({x10_csv} absent — run --scan-x10 first for the calsim gates)")
    # the cdec15 lineage is RAW — the master must match it bit-for-bit, no table
    _compare(master, root / "cdec15_grid" / "forcing" / "historical_livneh_unsplit.nc",
             [("prcp", "prcp"), ("tavg", "tavg")], tol=1e-4)
    _compare(master, root / "cdec15_grid" / "tminmax_livneh_percell.nc",
             [("tmin", "tmin"), ("tmax", "tmax")], tol=1e-5)
    # the calsim stores carry the /10 artifact correction — expect the table
    for d in CALSIM_DOMAINS:
        _compare(master, root / "calsim" / "forcing"
                 / f"historical_livneh_unsplit_{d}.nc",
                 [("prcp", "prcp"), ("tavg", "tavg")], tol=6e-3, x10=x10)
    master.close()
    print("VERIFY: all committed stores reproduced from the master")


def _warn_x10_suspects(sub: xr.Dataset, keys: list[str],
                       x10: pd.DataFrame | None) -> None:
    """Warn about POSSIBLE x10 artifacts the reference table cannot cover.

    The table is exact only where a committed calsim store exists; the
    upstream correction was station-informed (no value threshold reproduces
    it — calibration against the 197 known pairs found no clean separator),
    so cells outside the modeling domains may carry uncorrected spikes on the
    same known days.  Flag cut cell-days on those days whose value is >= 30 mm
    AND > 2x the cell's own max summer daily precip over the rest of the
    record — a HUMAN decision, never an automatic edit."""
    if x10 is None or not len(x10):
        return
    t = pd.to_datetime(np.asarray(sub["time"]))
    days = np.array(sorted(set(pd.to_datetime(x10["date"]))),
                    dtype="datetime64[ns]")
    art = np.isin(t.values, days)
    summer = (t.month >= 6) & (t.month <= 9) & ~art
    prcp = sub["prcp"].to_numpy()
    base = prcp[:, summer].max(axis=1)
    covered = {(r["key"], pd.Timestamp(r["date"])) for _, r in x10.iterrows()}
    hits = []
    for j in np.where(art)[0]:
        v = prcp[:, j]
        for i in np.where((v >= 30.0) & (v > 2.0 * base))[0]:
            if (keys[i], t[j]) not in covered:
                hits.append(f"{keys[i]} {t[j].date()} {v[i]:.1f} mm")
    if hits:
        print(f"WARNING: {len(hits)} cell-days look like UNCORRECTED x10 "
              "artifacts (outside the reference table's calsim coverage) — "
              "inspect before trusting summer extremes:")
        for h in hits[:20]:
            print("   ", h)
        if len(hits) > 20:
            print(f"    ... and {len(hits) - 20} more")


def cut(master_path: str, name: str, cells_csv: str, out_dir: str,
        x10_csv: str, fix_x10: bool) -> None:
    cells = pd.read_csv(cells_csv)
    keys = [norm_key(k) for k in cells["key"].astype(str)]
    master = xr.open_dataset(master_path)
    missing = sorted(set(keys) - set(np.asarray(master["key"]).astype(str)))
    if missing:
        raise SystemExit(f"{len(missing)} cells not in the master "
                         f"(first: {missing[:3]}) — outside the region?")
    sub = master.sel(key=keys).load()
    nfix = 0
    x10 = _x10_table(x10_csv)
    if fix_x10:
        if x10 is None:
            raise SystemExit(f"{x10_csv} absent — run --scan-x10 first, or "
                             "pass --no-fix-x10 for the raw cdec15 convention")
        nfix = _apply_x10(sub["prcp"].values, keys, np.asarray(sub["time"]), x10)
    _warn_x10_suspects(sub, keys, x10)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    tavg = ((sub["tmin"].astype(np.float64) + sub["tmax"].astype(np.float64))
            / 2.0).astype(np.float32)
    enc = dict(zlib=True, complevel=4)
    fc = xr.Dataset({"prcp": sub["prcp"], "tavg": tavg})
    fc.to_netcdf(od / f"historical_livneh_unsplit_{name}.nc",
                 encoding={v: enc for v in fc.data_vars})
    tm = xr.Dataset({"tmin": sub["tmin"], "tmax": sub["tmax"]})
    tm.to_netcdf(od / f"tminmax_livneh_percell_{name}.nc",
                 encoding={v: enc for v in tm.data_vars})
    master.close()
    print(f"cut {name}: {len(keys)} cells -> {od}\\historical_livneh_unsplit_{name}.nc "
          f"+ tminmax_livneh_percell_{name}.nc"
          + (f" ({nfix} x10 artifact days corrected)" if fix_x10
             else " (RAW — x10 artifacts NOT corrected)"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-master", action="store_true")
    ap.add_argument("--scan-x10", action="store_true",
                    help="derive the x10 precip-artifact table from the "
                         "committed calsim stores")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--cut", metavar="NAME")
    ap.add_argument("--cells", help="CSV with a `key` column (for --cut)")
    ap.add_argument("--out-dir", help="output directory (for --cut)")
    ap.add_argument("--no-fix-x10", action="store_true",
                    help="cut RAW (the cdec15 convention) instead of applying "
                         "the documented x10 corrections")
    ap.add_argument("--wgen-dir", default=WGEN_DIR)
    ap.add_argument("--grid", default=GRID_CSV)
    ap.add_argument("--master", default=MASTER)
    ap.add_argument("--x10-table", default=X10_CSV)
    args = ap.parse_args()
    if args.build_master:
        build_master(args.wgen_dir, args.grid, args.master)
    if args.scan_x10:
        scan_x10(args.master, args.x10_table)
    if args.verify:
        verify(args.master, args.x10_table)
    if args.cut:
        if not (args.cells and args.out_dir):
            raise SystemExit("--cut requires --cells and --out-dir")
        cut(args.master, args.cut, args.cells, args.out_dir,
            args.x10_table, fix_x10=not args.no_fix_x10)
    if not (args.build_master or args.scan_x10 or args.verify or args.cut):
        raise SystemExit("nothing to do: pass --build-master / --scan-x10 / "
                         "--verify / --cut")


if __name__ == "__main__":
    main()
