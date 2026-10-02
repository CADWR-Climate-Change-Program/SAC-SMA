"""WGEN Product A climate scenarios -> compact, exact region stores ``data/inputs/forcing/wgen_product_a_sNN.nc``.

Source (READ ONLY): DWR's WGEN Product A release, 30 thermodynamic scenarios of daily 1/16-deg weather,
one ASCII file per cell (``<box>/<s>/meteo_<lat>_<lon>``: year month day prcp tmax tmin, 1915-2018,
exact hundredths), and the scenario key ``CC.thermodynamic.change_list.xlsx`` one folder up. Scenario 1
is the committed ``data/inputs/forcing/wgen_product_a.nc``. The format and its decoder live in
``sacsma/wgen_scenarios.py``; ``io.load_forcing(product="wgen_product_a_sNN")`` reads them.

Stages (sacsma env; run from the repo root):
  --key                 parse the xlsx (stdlib zip/xml; no openpyxl) -> data/inputs/forcing/wgen_product_a_scenarios.csv
  --pull S [S ...]      read s1 + each S for the 4410 region cells (3 threads, resumable blocks) into
                        int32-hundredths checkpoints <ckpt>/sNN_prcp_h.npz, with the source gates:
                          G1 Box s1 == wgen_product_a.nc (prcp float32-identical; tmin/tmax after the
                             store's min/max sort);
                          G2 tmax_S - tmax_1 == tmin_S - tmin_1 == 100*dT hundredths on EVERY day
                             (what licenses storing only dT);
                          G3 prcp_S > 0 only where s1 > 0, apart from counted added-wet days.
  --build S [S ...]     table-encode each S (see sacsma/wgen_scenarios.py) -> <out>/wgen_product_a_sNN.nc
  --verify S [S ...]    reopen each file and decode EVERY cell with sacsma.wgen_scenarios (all cells, and a
                        random subset in random order) against the checkpoints: 0 mismatches, every CRC.
Options: --box DIR (the release folder; default ``wgen_scenarios`` in data/local_paths.toml),
         --checkpoint-dir DIR (default tmp/wgen_product_a_scen/pull; local only, not committed),
         --out DIR (default data/inputs/forcing).
Timing (2026-09-29): pull about 80 s per 147-cell block for s1 + one scenario (about 40 min); build and
verify a few minutes per scenario.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import layout, local_path  # noqa: E402

from sacsma import wgen_scenarios as W  # noqa: E402

NT = 37986
BLOCK = 147
# paths relative to the repository root, where this script is run from
KEY_CSV = layout.wgen_scenario_key()


# ---- key ------------------------------------------------------------------------------------------
def read_key_xlsx(path: Path) -> pd.DataFrame:
    """The first sheet of CC.thermodynamic.change_list.xlsx as a DataFrame (stdlib only)."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as z:
        ss = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{ns}si"):
                ss.append("".join(t.text or "" for t in si.iter(f"{ns}t")))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for r in sheet.iter(f"{ns}row"):
        row = {}
        for c in r.iter(f"{ns}c"):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find(f"{ns}v")
            if v is None:
                continue
            row[col] = ss[int(v.text)] if c.get("t") == "s" else float(v.text)
        rows.append(row)
    head = rows[0]
    return pd.DataFrame([{head[k]: r.get(k) for k in head} for r in rows[1:] if r])


def stage_key(box: Path) -> pd.DataFrame:
    raw = read_key_xlsx(box.parent / "CC.thermodynamic.change_list.xlsx")
    cols = {c.lower(): c for c in raw.columns}
    pick = lambda *w: raw[next(c for k, c in cols.items() if all(x in k for x in w))]  # noqa: E731
    key = pd.DataFrame({"scenario": pick("scenario").astype(int),
                        "dT_C": pick("temperature").astype(float),
                        "cc_pct_per_C": pick("quantile").astype(float).round(6),
                        "dmean_pct": pick("mean").astype(float).round(6)})
    key["F_extreme"] = ((1 + key.cc_pct_per_C / 100) ** key.dT_C).round(6)
    assert list(key.scenario) == list(range(1, 31)), key.scenario.tolist()
    assert (key.dT_C == key.dT_C.round()).all()
    KEY_CSV.parent.mkdir(parents=True, exist_ok=True)
    key.to_csv(KEY_CSV, index=False)
    print(key.to_string(index=False))
    return key


def load_key() -> pd.DataFrame:
    return pd.read_csv(KEY_CSV).set_index("scenario")


# ---- pull -----------------------------------------------------------------------------------------
def stage_pull(box: Path, ckpt: Path, scen: list[int]) -> dict:
    import netCDF4 as nc

    DT = {int(s): int(round(r.dT_C)) for s, r in load_key().iterrows()}
    parts = ckpt / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    keys = pd.read_csv(layout.grid_cells(), dtype={"key": str}).key.tolist()
    assert len(keys) == 4410 and len(set(keys)) == 4410
    days = pd.date_range("1915-01-01", "2018-12-31", freq="D")
    YMD = np.stack([days.year, days.month, days.day], 1).astype(np.int64)
    R = nc.Dataset(layout.forcing_dir() / f"{W.BASE_PRODUCT}.nc")
    rk = {str(k): i for i, k in enumerate(R["key"][:])}

    def read_one(arg):
        k, s = arg
        t = time.time()
        b = open(box / str(s) / f"meteo_{k}", "rb").read()
        tr = time.time() - t
        a = np.loadtxt(io.BytesIO(b))
        assert a.shape == (NT, 6), (k, s, a.shape)
        assert np.array_equal(a[:, :3].astype(np.int64), YMD), (k, s, "dates")
        h = np.rint(a[:, 3:] * 100)
        assert np.abs(a[:, 3:] * 100 - h).max() < 1e-6, (k, s, "not hundredths")
        return k, s, h.astype(np.int64), len(b), tr          # columns prcp, tmax, tmin

    exe = ThreadPoolExecutor(3)
    log = open(ckpt / "pull_log.txt", "a")
    T0 = time.time()
    nb = (len(keys) + BLOCK - 1) // BLOCK
    for b in range(nb):
        fn = parts / f"block{b:02d}.npz"
        if fn.exists():
            continue
        kb = keys[b * BLOCK:(b + 1) * BLOCK]
        res = {(k, s): (h, n, tr) for k, s, h, n, tr in exe.map(read_one, [(k, s) for k in kb for s in [1] + scen])}
        idx = [rk[k] for k in kb]
        rep = {v: np.stack([np.asarray(R[v][i, :], np.float32) for i in idx]) for v in ("prcp", "tmin", "tmax")}
        P = {s: np.stack([res[(k, s)][0][:, 0] for k in kb]).astype(np.int32) for s in [1] + scen}
        g = dict(n=len(kb), bytes=int(sum(v[1] for v in res.values())), read_s=float(sum(v[2] for v in res.values())))
        X1 = np.stack([res[(k, 1)][0][:, 1] for k in kb])
        N1 = np.stack([res[(k, 1)][0][:, 2] for k in kb])
        g["G1_prcp_bad_cells"] = int(sum(not np.array_equal(rep["prcp"][j], (P[1][j] / 100.0).astype(np.float32))
                                         for j in range(len(kb))))
        lo, hi = np.minimum(X1, N1), np.maximum(X1, N1)
        g["G1_temp_bad_cells"] = int(sum(not (np.array_equal(rep["tmin"][j], (lo[j] / 100.0).astype(np.float32))
                                              and np.array_equal(rep["tmax"][j], (hi[j] / 100.0).astype(np.float32)))
                                         for j in range(len(kb))))
        g["inverted_pairs"] = int((X1 < N1).sum())
        for s in scen:
            Xs = np.stack([res[(k, s)][0][:, 1] for k in kb])
            Ns = np.stack([res[(k, s)][0][:, 2] for k in kb])
            g[f"G2_s{s:02d}_bad_days"] = int(((Xs - X1) != 100 * DT[s]).sum() + ((Ns - N1) != 100 * DT[s]).sum())
            g[f"G3_s{s:02d}_added_wet"] = int(((P[s] > 0) & (P[1] == 0)).sum())
            g[f"s{s:02d}_removed_wet"] = int(((P[s] == 0) & (P[1] > 0)).sum())
            g[f"s{s:02d}_sum_h"] = int(P[s].sum(dtype=np.int64))
        g["s01_wet"] = int((P[1] > 0).sum())
        g["s01_sum_h"] = int(P[1].sum(dtype=np.int64))
        np.savez(str(fn) + ".tmp.npz", keys=np.array(kb), gates=json.dumps(g), **{f"s{s:02d}": P[s] for s in P})
        os.replace(str(fn) + ".tmp.npz", fn)
        log.write(f"block {b:02d}/{nb} {time.time() - T0:.0f}s {json.dumps(g)}\n")
        log.flush()
        print(f"block {b:02d}/{nb} {time.time() - T0:.0f}s", flush=True)
    R.close()
    G, arrs, kk = [], {s: [] for s in [1] + scen}, []
    for b in range(nb):
        z = np.load(parts / f"block{b:02d}.npz")
        kk += [str(k) for k in z["keys"]]
        G.append(json.loads(str(z["gates"])))
        for s in arrs:
            arrs[s].append(z[f"s{s:02d}"])
    assert kk == keys
    for s in arrs:
        A = np.concatenate(arrs[s])
        assert A.shape == (4410, NT) and A.min() >= 0
        np.savez_compressed(ckpt / f"s{s:02d}_prcp_h.npz", keys=np.array(keys), prcp_h=A, ymd=YMD)
    tot = {k: sum(g[k] for g in G) for k in G[0]}
    tot["G1_pass"] = tot["G1_prcp_bad_cells"] == 0 and tot["G1_temp_bad_cells"] == 0
    for s in scen:
        tot[f"G2_s{s:02d}_pass"] = tot[f"G2_s{s:02d}_bad_days"] == 0
        tot[f"s{s:02d}_volume_ratio"] = tot[f"s{s:02d}_sum_h"] / tot["s01_sum_h"]
        tot[f"s{s:02d}_removed_wet_frac"] = tot[f"s{s:02d}_removed_wet"] / tot["s01_wet"]
    tot.update(scenarios=[1] + scen, dT={s: DT[s] for s in [1] + scen}, wall_s=round(time.time() - T0, 1))
    json.dump(tot, open(ckpt / "pull_gates.json", "w"), indent=1)
    print(json.dumps(tot, indent=1))
    if not tot["G1_pass"] or not all(tot[f"G2_s{s:02d}_pass"] for s in scen):
        raise SystemExit("pull gates FAILED (see pull_gates.json)")
    return tot


# ---- build ----------------------------------------------------------------------------------------
def encode_cell(h1: np.ndarray, hs: np.ndarray, mon: np.ndarray):
    """One cell -> (delta table int32, int8 residual per s1-wet day, overrides [(day, value)])."""
    wet = h1 > 0
    ux, inv = np.unique(mon[wet] * W.SHIFT + h1[wet], return_inverse=True)
    y = hs[wet]
    order = np.lexsort((y, inv))
    cnt = np.bincount(inv, minlength=len(ux))
    st = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    yk = y[order][st + (cnt - 1) // 2]                       # lower median of each bin
    first = np.ones(len(ux), bool)
    first[1:] = (ux[1:] // W.SHIFT) != (ux[:-1] // W.SHIFT)
    dtab = np.where(first, yk, yk - np.concatenate([[0], yk[:-1]]))
    res = y - yk[inv]
    big = np.abs(res) > 127                                  # q99-straddle bins: exact override
    wd = np.nonzero(wet)[0]
    ovr = [(int(d), int(v)) for d, v in zip(wd[big], y[big])]
    res[big] = 0
    ad = np.nonzero(~wet & (hs > 0))[0]                     # dry in s1, wet here
    ovr += [(int(d), int(hs[d])) for d in ad]
    assert np.abs(dtab).max() < 2 ** 31
    return dtab.astype(np.int32), res.astype(np.int8), sorted(ovr)


def stage_build(ckpt: Path, out: Path, s: int, box: Path) -> Path:
    import netCDF4 as nc

    key = load_key().loc[s]
    z1, zs = np.load(ckpt / "s01_prcp_h.npz"), np.load(ckpt / f"s{s:02d}_prcp_h.npz")
    keys = [str(k) for k in z1["keys"]]
    assert keys == [str(k) for k in zs["keys"]]
    H1, HS = z1["prcp_h"].astype(np.int64), zs["prcp_h"].astype(np.int64)
    mon = z1["ymd"][:, 1].astype(np.int64)
    base = nc.Dataset(layout.forcing_dir() / f"{W.BASE_PRODUCT}.nc")    # the input the decoder will see
    rk = {str(k): i for i, k in enumerate(base["key"][:])}
    T0 = time.time()
    tabs, ress, tc, wc, c1, cs, ok_, od_, ov_ = [], [], [], [], [], [], [], [], []
    for n, k in enumerate(keys):
        h1_store = W.hundredths(np.asarray(base["prcp"][rk[k], :], np.float32))
        assert np.array_equal(h1_store, H1[n]), f"checkpoint s1 != {W.BASE_PRODUCT}.nc at {k}"
        d, r, o = encode_cell(H1[n], HS[n], mon)
        chk = W.decode_cell(H1[n], mon, d, r, o)
        assert np.array_equal(chk, HS[n]), f"encode/decode round trip failed at {k}"
        tabs.append(d); ress.append(r); tc.append(len(d)); wc.append(len(r))
        c1.append(W.crc(H1[n])); cs.append(W.crc(HS[n]))
        for dd, vv in o:
            ok_.append(n); od_.append(dd); ov_.append(vv)
    base.close()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{W.BASE_PRODUCT}_s{s:02d}.nc"
    tmp = path.with_suffix(".nc.tmp")
    ds = nc.Dataset(tmp, "w", format="NETCDF4")
    ds.title = f"WGEN Product A scenario {s}: precipitation as an exact table codec over scenario 1"
    ds.codec = W.CODEC
    ds.scenario = np.int32(s)
    ds.dT_C = float(key.dT_C)
    ds.dT_hundredths = np.int32(round(100 * key.dT_C))
    ds.cc_pct_per_C = float(key.cc_pct_per_C)
    ds.dmean_pct = float(key.dmean_pct)
    ds.F_extreme = float(key.F_extreme)
    ds.base_product = W.BASE_PRODUCT
    ds.source = (f"DWR WGEN Product A release, {box}/{s}/meteo_<key> (daily 1915-01-01..2018-12-31, "
                 "0.01 mm / 0.01 degC); scenario key CC.thermodynamic.change_list.xlsx")
    ds.table_order = ("per cell (key order): sorted distinct month*2**20 + h1 over the cell's scenario-1 "
                      "wet days; dprcp_table = the first value of each (cell, month) segment, then "
                      "differences; see sacsma/wgen_scenarios.py")
    ds.temperature = "not stored: tmin/tmax = wgen_product_a.nc + dT_hundredths/100 exactly (integer route)"
    ds.built_by = "data/inputs/forcing/wgen_product_a_scenarios.py"
    n_tab, n_wet, n_ovr = int(sum(tc)), int(sum(wc)), len(ok_)
    for name, n in (("key", len(keys)), ("table", n_tab), ("wet", n_wet), ("override", n_ovr)):
        ds.createDimension(name, n)
    v = ds.createVariable("key", str, ("key",))
    v[:] = np.array(keys, dtype=object)
    for name, arr, typ in (("table_count", tc, "i4"), ("wet_count", wc, "i4"),
                           ("s1_crc32", c1, "u4"), ("prcp_crc32", cs, "u4")):
        v = ds.createVariable(name, typ, ("key",))
        v[:] = np.asarray(arr, np.uint32 if typ == "u4" else np.int32)
    v = ds.createVariable("dprcp_table", "i4", ("table",), zlib=True, complevel=4, shuffle=True,
                          chunksizes=(min(n_tab, 2 ** 18),))
    v[:] = np.concatenate(tabs)
    v.units = "0.01 mm/day (delta-coded within each cell-month segment)"
    v = ds.createVariable("prcp_resid", "i1", ("wet",), zlib=True, complevel=4,
                          chunksizes=(min(n_wet, 2 ** 20),))
    v[:] = np.concatenate(ress)
    v.units = "0.01 mm/day"
    for name, arr in (("override_key", ok_), ("override_day", od_), ("override_value", ov_)):
        v = ds.createVariable(name, "i4", ("override",))
        v[:] = np.asarray(arr, np.int32)
    ds.close()
    os.replace(tmp, path)
    print(f"s{s:02d}: {path} {path.stat().st_size / 1e6:.1f} MB; table {n_tab}, wet {n_wet}, "
          f"overrides {n_ovr}, |resid| max {max(int(np.abs(r).max()) for r in ress)} "
          f"({time.time() - T0:.0f}s)", flush=True)
    return path


# ---- verify ---------------------------------------------------------------------------------------
def stage_verify(ckpt: Path, out: Path, s: int) -> dict:
    zs, z1 = np.load(ckpt / f"s{s:02d}_prcp_h.npz"), np.load(ckpt / "s01_prcp_h.npz")
    keys = [str(k) for k in zs["keys"]]
    HS, H1 = zs["prcp_h"], z1["prcp_h"]
    dT_h = int(round(100 * load_key().loc[s].dT_C))
    base = layout.forcing_dir() / f"{W.BASE_PRODUCT}.nc"
    path = out / f"{W.BASE_PRODUCT}_s{s:02d}.nc"
    rng = np.random.default_rng(s)
    sub = [keys[i] for i in rng.choice(len(keys), 40, replace=False)]
    rep = {}
    for tag, want in (("all", keys), ("random40", sub)):
        t = time.time()
        ds = W.decode_region(base, path, want)
        pos = [keys.index(k) for k in want]
        p = ds["prcp"].values
        bad_p = int((p != (HS[pos] / 100.0).astype(np.float32)).sum())
        import xarray as xr
        b = xr.open_dataset(base)[["tmin", "tmax"]].sel(key=want).load()
        bad_t = sum(int((W.hundredths(ds[v].values) - W.hundredths(b[v].values) != dT_h).sum())
                    for v in ("tmin", "tmax"))
        rep[tag] = dict(cells=len(want), prcp_mismatch=bad_p, temp_mismatch=bad_t,
                        wet_s1=int((H1[pos] > 0).sum()), s=round(time.time() - t, 1))
    rep["pass"] = all(r["prcp_mismatch"] == 0 and r["temp_mismatch"] == 0 for r in rep.values())
    rep["file_mb"] = round(path.stat().st_size / 1e6, 2)
    json.dump(rep, open(ckpt / f"verify_s{s:02d}.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    if not rep["pass"]:
        raise SystemExit(f"verify FAILED for s{s:02d}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key", action="store_true")
    ap.add_argument("--pull", nargs="+", type=int)
    ap.add_argument("--build", nargs="+", type=int)
    ap.add_argument("--verify", nargs="+", type=int)
    ap.add_argument("--box", default=str(local_path("wgen_scenarios")))
    ap.add_argument("--checkpoint-dir", default="tmp/wgen_product_a_scen/pull")
    ap.add_argument("--out", default=str(layout.forcing_dir()))
    a = ap.parse_args()
    box, ckpt, out = Path(a.box), Path(a.checkpoint_dir), Path(a.out)
    for grp in (a.pull, a.build, a.verify):
        assert grp is None or all(2 <= s <= 30 for s in grp), grp
    if a.key:
        stage_key(box)
    if a.pull:
        stage_pull(box, ckpt, sorted(set(a.pull)))
    for s in a.build or []:
        stage_build(ckpt, out, s, box)
    for s in a.verify or []:
        stage_verify(ckpt, out, s)


if __name__ == "__main__":
    main()
