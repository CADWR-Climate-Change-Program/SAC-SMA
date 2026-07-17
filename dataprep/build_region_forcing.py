"""Build the UNIFIED region forcing stores -> data/region/forcing/<product>.nc.

One file per product at the region grid (``data/region/grid_cells.csv``),
3 variables each (``prcp``/``tmin``/``tmax`` float32; ``tavg`` is derived at
load as ``(tmax+tmin)/2`` — the committed stores' exact convention), coords
``key`` (5-decimal normalized) x ``time``.  These REPLACE the per-domain
``data/calsim/forcing/*.nc`` + ``data/cdec15_grid/forcing/*.nc`` +
``data/cdec15_grid/tminmax_livneh_percell.nc`` stores (decision 2026-07-16:
one forcing source for everything 1/16-deg-grid-based — calsim SAC-SMA, dPL,
hybrids; the same cells were previously stored up to 4x across domain files,
and tmin/tmax lived in a separate sidecar).  ``data/cdec15/forcing`` (the
dense off-grid fine-HRU product) is deliberately untouched.

Products / sources / conventions:

* ``historical_livneh_unsplit`` — from the local WGEN NonDetrend-Unsplit
  master (``wgen_forcing.py --build-master``; raw lineage, 1915-2018), with
  the documented x10 misplaced-decimal spikes CORRECTED (/10) at the 197
  ``data/region/prcp_x10_artifacts.csv`` pairs (user decision 2026-07-16:
  the unified store carries the corrected convention everywhere — the
  committed calsim stores already did; the raw cdec15_grid convention is
  retired with its store, and its dPL consumers are retrained).
* ``wgen_product_a`` — verbatim from the OneDrive release
  (``BASE/WGEN/Product_A/1/meteo_<lat>_<lon>``, 1915-2018): all 4410 region
  cells present, and the release is ALREADY x10-corrected (verified at the
  artifact pairs).
* ``historical_lto`` — verbatim from the OneDrive LTO release
  (``BASE/Historical_Climate_LTO/1_Historical/data_<lat>_<lon>``, columns
  prcp tmax tmin wind [wind dropped], implicit daily 1915-2021): its own
  precipitation realization (split lineage) with its own upstream
  corrections.  Only 4057/4410 region cells exist in the release (the
  missing 353 are footprint-sweep cells outside the LTO study domain,
  Kern/Tulare area) — the store carries the available cells and lists the
  missing ones in its ``missing_cells`` attribute; basins touching them
  cannot run the LTO product (unchanged from today).

Each build verifies against the committed per-domain stores it replaces
before those are deleted (prcp/tavg within their 3-decimal write precision;
the unsplit-vs-cdec15_grid diff is EXPECTED to be exactly the x10 table).

Usage:
    python dataprep/build_region_forcing.py \
        --product all|historical_livneh_unsplit|wgen_product_a|historical_lto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(".")
GRID_CSV = "data/region/grid_cells.csv"
X10_CSV = "data/region/prcp_x10_artifacts.csv"
MASTER = r"D:\sacsma-data\forcing\livneh_unsplit_nondetrend_daily_region.nc"
ONEDRIVE = Path(r"C:\Users\warnold_la\OneDrive - California Department of Water"
                r" Resources\CalSim Synthetic Hydroclimate"
                r"\calsim3-stochastic-input-generation\data\BASE")
LTO_DIR = ONEDRIVE / "Historical_Climate_LTO" / "1_Historical"
PA_DIR = ONEDRIVE / "WGEN" / "Product_A" / "1"
OUT_DIR = Path("data/region/forcing")
CALSIM_DOMAINS = ("9unimp", "11obs", "12rim")


def norm_key(k: str) -> str:
    lat, lon = str(k).split("_")
    return f"{round(float(lat), 5)}_{round(float(lon), 5)}"


def _write(name: str, keys: list[str], time: pd.DatetimeIndex,
           arr: dict[str, np.ndarray], attrs: dict) -> Path:
    ds = xr.Dataset({v: (("key", "time"), a) for v, a in arr.items()},
                    coords={"key": keys, "time": time}, attrs=attrs)
    enc = {v: dict(zlib=True, complevel=4, chunksizes=(1, len(time)))
           for v in arr}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.nc"
    ds.to_netcdf(out, encoding=enc)
    print(f"wrote {out}: {len(keys)} cells x {len(time)} days "
          f"({out.stat().st_size / 1e9:.2f} GB)")
    return out


def _verify_vs_committed(out: Path, product: str, tol: float = 6e-3,
                         expect_x10_vs_grid: bool = False) -> None:
    """New region store vs every committed per-domain store it replaces."""
    reg = xr.open_dataset(out)
    refs = [(ROOT / "data" / "calsim" / "forcing" / f"{product}_{d}.nc", None)
            for d in CALSIM_DOMAINS]
    if product == "historical_livneh_unsplit":
        refs.append((ROOT / "data" / "cdec15_grid" / "forcing"
                     / f"{product}.nc", "x10" if expect_x10_vs_grid else None))
        refs.append((ROOT / "data" / "cdec15_grid"
                     / "tminmax_livneh_percell.nc", "tmm"))
    x10 = pd.read_csv(X10_CSV, parse_dates=["date"])
    x10_pairs = {(r["key"], pd.Timestamp(r["date"])) for _, r in x10.iterrows()}
    for ref_path, mode in refs:
        if not ref_path.exists():
            print(f"  (skip {ref_path.name}: already removed)")
            continue
        ref = xr.open_dataset(ref_path)
        rk = np.asarray(ref["key"]).astype(str)
        mk = [norm_key(k) for k in rk]
        have = set(np.asarray(reg["key"]).astype(str))
        keep = [i for i, k in enumerate(mk) if k in have]
        if len(keep) < len(mk):
            print(f"  {ref_path.name}: {len(mk) - len(keep)} cells absent "
                  "from the region store (LTO coverage gap) — comparing rest")
        sub = reg.sel(key=[mk[i] for i in keep],
                      time=np.asarray(ref["time"]))
        t = pd.to_datetime(np.asarray(ref["time"]))
        if mode == "tmm":
            pairs = [("tmin", "tmin"), ("tmax", "tmax")]
        else:
            pairs = [("prcp", "prcp"), ("tavg", "tavg")]
        for mv, rv in pairs:
            if mv == "tavg":
                m = (sub["tmin"].to_numpy().astype(np.float64)
                     + sub["tmax"].to_numpy().astype(np.float64)) / 2.0
            else:
                m = sub[mv].to_numpy().astype(np.float64)
            r = ref[rv].to_numpy().astype(np.float64)[keep]
            d = np.abs(m - r)
            if mode == "x10" and mv == "prcp":
                ci, dj = np.where(d > tol)
                bad = [(mk[keep[i]], t[j]) for i, j in
                       zip(ci, dj, strict=True)]
                extra = [p for p in bad if p not in x10_pairs]
                print(f"  {ref_path.name} {rv}: {len(bad)} diffs > {tol:g} "
                      f"(expected = the x10 table; unexplained: {len(extra)})"
                      + ("  PASS" if not extra else "  FAIL"))
                if extra:
                    raise SystemExit(f"unexplained diffs: {extra[:5]}")
            else:
                mx = float(np.nanmax(d))
                print(f"  {ref_path.name} {rv}: max|d| = {mx:.2e} "
                      f"({'PASS' if mx <= tol else 'FAIL'} @ {tol:g})")
                if mx > tol:
                    raise SystemExit(f"VERIFY FAILED for {ref_path}")
        ref.close()
    reg.close()


def build_unsplit(grid: pd.DataFrame) -> None:
    master = xr.open_dataset(MASTER)
    keys = grid["key"].astype(str).tolist()
    sub = master.sel(key=keys).load()
    master.close()
    t = pd.to_datetime(np.asarray(sub["time"]))
    x10 = pd.read_csv(X10_CSV, parse_dates=["date"])
    ki = {k: i for i, k in enumerate(keys)}
    ti = {pd.Timestamp(v): j for j, v in enumerate(t)}
    prcp = sub["prcp"].to_numpy()
    n = 0
    for _, r in x10.iterrows():
        i, j = ki.get(norm_key(r["key"])), ti.get(r["date"])
        if i is not None and j is not None:
            prcp[i, j] /= 10.0
            n += 1
    out = _write("historical_livneh_unsplit", keys, t,
                 {"prcp": prcp, "tmin": sub["tmin"].to_numpy(),
                  "tmax": sub["tmax"].to_numpy()},
                 {"title": "Livneh-unsplit daily forcing, unified region store",
                  "source": "WGEN NonDetrend-Unsplit statewide ASCII (local "
                            "master); tavg convention = (tmax+tmin)/2",
                  "corrections": f"{n} x10 misplaced-decimal precip spikes "
                                 f"corrected /10 per {X10_CSV}"})
    print(f"  ({n} x10 corrections applied)")
    _verify_vs_committed(out, "historical_livneh_unsplit",
                         expect_x10_vs_grid=True)


def _build_from_ascii(grid: pd.DataFrame, name: str, src_dir: Path,
                      prefix: str, dated: bool, time: pd.DatetimeIndex,
                      attrs: dict, fills: dict[str, str] | None = None) -> None:
    """``fills``: {missing_cell: source_cell} — the documented neighbor-fills
    the committed stores carry (LTO Mt Shasta); applied ONLY when the target
    is absent from the release and the source is present."""
    keys, missing = [], []
    for k in grid["key"].astype(str):
        (keys if (src_dir / f"{prefix}{k}").exists() else missing).append(k)
    fill_of = {}
    for tgt, srcc in (fills or {}).items():
        if tgt in missing and (src_dir / f"{prefix}{srcc}").exists():
            missing.remove(tgt)
            keys.append(tgt)
            fill_of[tgt] = srcc
    keys = sorted(keys, key=lambda k: tuple(float(x) for x in k.split("_")))
    arr = {v: np.empty((len(keys), len(time)), dtype=np.float32)
           for v in ("prcp", "tmin", "tmax")}
    for i, k in enumerate(keys):
        src = src_dir / f"{prefix}{fill_of.get(k, k)}"
        if dated:   # year month day prcp tmax tmin
            w = pd.read_csv(src, sep=r"\s+", header=None,
                            names=["y", "m", "d", "pr", "tmax", "tmin"])
        else:       # prcp tmax tmin wind, implicit daily calendar
            w = pd.read_csv(src, sep=r"\s+", header=None,
                            names=["pr", "tmax", "tmin", "wind"])
        if len(w) != len(time):
            raise SystemExit(f"{prefix}{k}: {len(w)} rows != {len(time)}")
        arr["prcp"][i] = w["pr"].to_numpy(np.float32)
        tmn = w["tmin"].to_numpy(np.float32)
        tmx = w["tmax"].to_numpy(np.float32)
        arr["tmin"][i] = np.minimum(tmn, tmx)   # same inverted-pair sort as
        arr["tmax"][i] = np.maximum(tmn, tmx)   # the unsplit lineage
        if (i + 1) % 500 == 0:
            print(f"  {name}: {i + 1}/{len(keys)} cells read", flush=True)
    if fill_of:
        attrs["filled_cells"] = "; ".join(f"{t} <- {s}"
                                          for t, s in fill_of.items())
        print(f"  {name}: neighbor-filled {fill_of}")
    if missing:
        attrs["missing_cells"] = " ".join(missing)
        dom_cols = [c for c in grid.columns if c.startswith("in_")
                    and c != "in_calsim3_fp"]
        gd = grid.set_index("key")
        n_dom = int(sum(gd.loc[k, dom_cols].sum() > 0 for k in missing))
        print(f"  {name}: {len(missing)} region cells absent from the release "
              f"(stored {len(keys)}; {n_dom} of the missing are modeling-"
              "domain cells)")
    out = _write(name, keys, time, arr, attrs)
    _verify_vs_committed(out, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="all")
    args = ap.parse_args()
    grid = pd.read_csv(GRID_CSV)
    todo = (("historical_livneh_unsplit", "wgen_product_a", "historical_lto")
            if args.product == "all" else (args.product,))
    for p in todo:
        if p == "historical_livneh_unsplit":
            build_unsplit(grid)
        elif p == "wgen_product_a":
            _build_from_ascii(
                grid, p, PA_DIR, "meteo_", dated=True,
                time=pd.date_range("1915-01-01", "2018-12-31", freq="D"),
                attrs={"title": "WGEN Product A scenario 1, unified region store",
                       "source": str(PA_DIR),
                       "note": "release is already x10-corrected; temperature "
                               "detrended to 1991-2020; precip = unsplit basis"})
        elif p == "historical_lto":
            _build_from_ascii(
                grid, p, LTO_DIR, "data_", dated=False,
                time=pd.date_range("1915-01-01", "2021-12-31", freq="D"),
                attrs={"title": "Historical LTO (split-lineage Livneh), "
                                "unified region store",
                       "source": str(LTO_DIR),
                       "note": "own precip realization + upstream corrections; "
                               "wind column dropped"},
                # the committed stores' documented Mt Shasta fill
                fills={"41.46875_-122.15625": "41.40625_-122.15625"})
        else:
            raise SystemExit(f"unknown product {p}")


if __name__ == "__main__":
    main()
