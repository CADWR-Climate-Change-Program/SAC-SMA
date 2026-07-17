"""STATEWIDE ET/SWE observation ingest from Google Earth Engine.

Re-exports the 7 GEE-derived obs products (3 ET + 4 SWE) over the full
statewide 1/16-deg grid (``data/statewide/grid_cells.csv``, 13,786 cells) in
the exact per-cell monthly form the dPL obs losses consume
(``sacsma.dpl.data.ET_FILES``/``SWE_FILES`` npz: keys/dates/<var>/lat/lon).
The two non-GEE products (GLEAM, FLUXCOM) have local raw sources and their own
ingest (see dataprep/README.md).

Monthly reductions mirror the original 2074-cell ingest (its parameters are
recorded in ``D:\\sacsma-data\\et_processed\\_ingest_*.log``): per-cell mean
over the 1/16-deg cell rectangle at 11132 m scale, months 1988-01..2018-12,
units converted to mm/month (ET) or mm mean monthly state (SWE; TerraClimate
stays an end-of-month SNAPSHOT — the loss loader applies the adjacent-mean
phase fix, so do NOT convert it here).

RUN ORDER (needs an authenticated earthengine-api: ``earthengine authenticate``):
  1. ``python dataprep/gee_obs_statewide.py --verify``
     re-ingests ONLY the 2074 15cdec_grid cells for every product and diffs
     against the existing D:\\sacsma-data npz — every product must match
     (rel RMS < 1e-3) before any statewide run.  Catches band/unit drift in
     GEE assets since the original ingest.
  2. ``python dataprep/gee_obs_statewide.py --products all``
     the statewide burn (13,786 cells x 372 months; hours — one-time).
     Writes data/statewide/et_obs/<p>_cell_monthly.npz and
     data/statewide/swe_obs/<p>_swe_cell_monthly.npz.

The dPL loaders then point at the statewide store via SACSMA_ET_DIR /
SACSMA_SWE_DIR (or the in-repo defaults once data.py is repointed).
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GRID_CSV = "data/statewide/grid_cells.csv"
SCALE_M = 11132                    # reduction scale of the original ingest
DATES = pd.date_range("1988-01-01", "2018-12-01", freq="MS")
CELL_DEG = 1.0 / 16.0

#: product -> (collection, band, kind, to_mm)
#: kind: "monthly" = one image per month; "daily_mean" = daily collection,
#: monthly mean of the daily band.  to_mm(value, days_in_month) -> mm.
PRODUCTS: dict[str, dict] = {
    # --- ET (mm/month totals) -------------------------------------------------
    "terraclimate": dict(
        coll="IDAHO_EPSCOR/TERRACLIMATE", band="aet", kind="monthly",
        var="et", out="et_obs/terraclimate_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * 0.1),                 # 0.1 mm scale factor
    "fldas": dict(
        coll="NASA/FLDAS/NOAH01/C/GL/M/V001", band="Evap_tavg", kind="monthly",
        var="et", out="et_obs/fldas_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * 86400.0 * nd),        # kg m-2 s-1 -> mm/month
    "era5land": dict(
        coll="ECMWF/ERA5_LAND/MONTHLY_AGGR", band="total_evaporation_sum",
        kind="monthly", var="et", out="et_obs/era5land_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * -1000.0),             # m (negative) -> mm/month
    # --- SWE (mm monthly-mean state; terraclimate = end-of-month snapshot) ---
    "daymet_swe": dict(
        coll="NASA/ORNL/DAYMET_V4", band="swe", kind="daily_mean",
        var="swe", out="swe_obs/daymet_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # kg m-2 == mm
    "terraclimate_swe": dict(
        coll="IDAHO_EPSCOR/TERRACLIMATE", band="swe", kind="monthly",
        var="swe", out="swe_obs/terraclimate_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # mm, EOM snapshot (keep)
    "fldas_swe": dict(
        coll="NASA/FLDAS/NOAH01/C/GL/M/V001", band="SWE_inst", kind="monthly",
        var="swe", out="swe_obs/fldas_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # kg m-2 == mm
    "era5land_swe": dict(
        coll="ECMWF/ERA5_LAND/MONTHLY_AGGR", band="snow_depth_water_equivalent",
        kind="monthly", var="swe", out="swe_obs/era5land_swe_cell_monthly.npz",
        to_mm=lambda v, nd: v * 1000.0),              # m of water -> mm
}
#: existing 2074-cell stores for --verify (product -> path)
VERIFY_AGAINST = {
    "terraclimate": r"D:\sacsma-data\et_processed\terraclimate_gee_cell_monthly.npz",
    "fldas": r"D:\sacsma-data\et_processed\fldas_gee_cell_monthly.npz",
    "era5land": r"D:\sacsma-data\et_processed\era5land_gee_cell_monthly.npz",
    "daymet_swe": r"D:\sacsma-data\swe_processed\daymet_swe_gee_cell_monthly.npz",
    "terraclimate_swe": r"D:\sacsma-data\swe_processed\terraclimate_swe_gee_cell_monthly.npz",
    "fldas_swe": r"D:\sacsma-data\swe_processed\fldas_swe_gee_cell_monthly.npz",
    "era5land_swe": r"D:\sacsma-data\swe_processed\era5land_swe_gee_cell_monthly.npz",
}


def _cells(verify: bool) -> pd.DataFrame:
    g = pd.read_csv(GRID_CSV)
    if verify:  # restrict to the 2074 cells the existing stores cover
        z = np.load(VERIFY_AGAINST["terraclimate"], allow_pickle=True)
        keep = set(str(k) for k in z["keys"])
        g = g[g["key"].astype(str).isin(keep)].reset_index(drop=True)
    return g


def _fc(ee, g: pd.DataFrame):
    """The cell rectangles as an ee.FeatureCollection (index -> row order)."""
    feats = []
    h = CELL_DEG / 2.0
    for i, r in g.iterrows():
        rect = ee.Geometry.Rectangle([r.lon - h, r.lat - h, r.lon + h, r.lat + h],
                                     proj="EPSG:4326", geodesic=False)
        feats.append(ee.Feature(rect, {"i": int(i)}))
    return ee.FeatureCollection(feats)


def _month_image(ee, spec: dict, d: pd.Timestamp):
    d0 = d.strftime("%Y-%m-%d")
    d1 = (d + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    coll = ee.ImageCollection(spec["coll"]).filterDate(d0, d1).select(spec["band"])
    if spec["kind"] == "daily_mean":
        return coll.mean()
    return coll.first()


def _reduce_month(ee, spec: dict, d: pd.Timestamp, fc, n: int,
                  chunk: int = 4500) -> np.ndarray:
    """Per-cell mean of one month's image -> (n,) float array (row order)."""
    img = _month_image(ee, spec, d)
    out = np.full(n, np.nan)
    lst = fc.toList(fc.size())
    for lo in range(0, n, chunk):
        sub = ee.FeatureCollection(lst.slice(lo, min(lo + chunk, n)))
        red = img.reduceRegions(collection=sub, reducer=ee.Reducer.mean(),
                                scale=SCALE_M).getInfo()
        for f in red["features"]:
            v = f["properties"].get("mean")
            if v is not None:
                out[int(f["properties"]["i"])] = float(v)
    return out


def run_product(ee, name: str, g: pd.DataFrame, out_root: Path,
                verify: bool) -> None:
    spec = PRODUCTS[name]
    n = len(g)
    fc = _fc(ee, g)
    val = np.full((n, len(DATES)), np.nan, dtype=np.float32)
    print(f"{name}: {spec['coll']}  {len(DATES)} months  {n} cells @ {SCALE_M} m",
          flush=True)
    for j, d in enumerate(DATES):
        nd = calendar.monthrange(d.year, d.month)[1]
        raw = _reduce_month(ee, spec, d, fc, n)
        val[:, j] = spec["to_mm"](raw, nd)
        if d.month == 12:
            print(f"  {name}: {j + 1}/{len(DATES)} months ({d.year}); "
                  f"cell-mean {np.nanmean(val[:, max(0, j - 11):j + 1]):.1f} mm",
                  flush=True)
    if verify:
        ref = np.load(VERIFY_AGAINST[name], allow_pickle=True)
        order = {str(k): i for i, k in enumerate(ref["keys"])}
        idx = np.array([order[str(k)] for k in g["key"]])
        rd = pd.to_datetime(ref["dates"])
        cols = [list(rd).index(d) for d in DATES if d in set(rd)]
        new = val[:, :len(cols)]
        old = ref[spec["var"]].astype(np.float64)[idx][:, cols]
        m = np.isfinite(new) & np.isfinite(old)
        rel = float(np.sqrt(np.mean((new[m] - old[m]) ** 2))
                    / (np.std(old[m]) + 1e-9))
        status = "PASS" if rel < 1e-3 else "FAIL"
        print(f"  VERIFY {name}: rel RMS {rel:.2e} vs {VERIFY_AGAINST[name]} "
              f"-> {status}", flush=True)
        return
    out = out_root / spec["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=g["key"].astype(str).to_numpy(),
                        dates=DATES.to_numpy(), lat=g["lat"].to_numpy(),
                        lon=g["lon"].to_numpy(), **{spec["var"]: val})
    nan_frac = float(np.isnan(val).mean())
    print(f"wrote {out}  (domain-mean {np.nanmean(val) * 12:.0f} mm/yr, "
          f"NaN frac {nan_frac:.4f})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["all"],
                    help=f"subset of {sorted(PRODUCTS)} or 'all'")
    ap.add_argument("--verify", action="store_true",
                    help="re-ingest ONLY the 2074 15cdec_grid cells and diff "
                         "against the existing npz (run FIRST, must PASS)")
    ap.add_argument("--out-root", default="data/statewide")
    args = ap.parse_args()
    try:
        import ee
        ee.Initialize()
    except Exception as e:                                    # noqa: BLE001
        sys.exit(f"earthengine-api not ready ({e}); run `pip install "
                 "earthengine-api` + `earthengine authenticate` first")
    names = sorted(PRODUCTS) if args.products == ["all"] else args.products
    g = _cells(args.verify)
    for name in names:
        run_product(ee, name, g, Path(args.out_root), args.verify)


if __name__ == "__main__":
    main()
