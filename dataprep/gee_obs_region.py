"""REGION ET/SWE observation ingest from Google Earth Engine.

Re-exports the 7 GEE-derived obs products (3 ET + 4 SWE) over the region
1/16-deg grid (``data/region/grid_cells.csv``, 4410 cells = the modeling
domains ∪ the full CalSim3 gpkg footprint) in the exact per-cell monthly form the dPL obs losses
consume (``sacsma.dpl.data.ET_FILES``/``SWE_FILES`` npz:
keys/dates/<var>/lat/lon).  The two non-GEE products (GLEAM, FLUXCOM) have
local raw sources and their own ingest (dataprep/local_obs_region.py).

**The region store is its own spec** (decision 2026-07-16, replacing the
"reproduce the legacy 2074-cell snapshot" gate): per-cell mean over the
1/16-deg cell rectangle at each asset's NATIVE scale, computed on the asset
versions current at export time (recorded in the npz ``meta`` field), months
1988-01..2018-12, units converted to mm/month (ET) or mm mean monthly state
(SWE; TerraClimate stays an end-of-month SNAPSHOT — the loss loader applies
the adjacent-mean phase fix, so do NOT convert it here).  The legacy
``D:\\sacsma-data`` npz are the frozen record of what the pre-region
``noah_ft`` trained on: GEE assets drift (ERA5-Land was reprocessed — rel RMS
~0.2 vs the snapshot under every reduction we tried, and the snapshot's exact
pipeline is lost), so the snapshot is irreproducible and everything that
consumed it (noah_ft -> the hybrids) is RETRAINED on this store instead.

RUN ORDER (needs an authenticated earthengine-api with a REGISTERED cloud
project: ``earthengine authenticate`` + pass ``--project <your-ee-project>``):
  1. ``python dataprep/gee_obs_region.py --products all --project <id>``
     the region burn (4410 cells x 372 months; a few hours — one-time).
     Writes data/region/et_obs/<p>_cell_monthly.npz and
     data/region/swe_obs/<p>_swe_cell_monthly.npz.
  2. ``python dataprep/gee_obs_region.py --verify --project <id>`` (optional)
     re-ingests the legacy stores' 2074 cells and REPORTS the delta vs the
     D:\\sacsma-data snapshot per product — documentation of asset drift,
     not a gate.

The dPL loaders then point at the region store via SACSMA_ET_DIR /
SACSMA_SWE_DIR (or the in-repo defaults once data.py is repointed).
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GRID_CSV = "data/region/grid_cells.csv"
DATES = pd.date_range("1988-01-01", "2018-12-01", freq="MS")
CELL_DEG = 1.0 / 16.0

#: product -> (collection, band, kind, scale, to_mm [, span, chunk, referee])
#: kind: "monthly" = one image per month; "monthly_mosaic" = several spatial
#: tiles per month (mosaic them); "daily_mean" = daily collection, monthly
#: mean of the daily band.  scale = the asset's NATIVE resolution (m) — the
#: reduction samples the cell rectangle at this density (verified: a coarser
#: scale aliases the mean; daymet@11132 vs its 1 km native gave a 0.33 rel
#: RMS error).  span = product-specific month range (default DATES).
#: referee=True products are benchmark-only (post-2000 coverage, excluded
#: from the training losses and from ``--products all``; run explicitly).
#: to_mm(value, days_in_month) -> mm.
PRODUCTS: dict[str, dict] = {
    # --- ET (mm/month totals) -------------------------------------------------
    "terraclimate": dict(
        coll="IDAHO_EPSCOR/TERRACLIMATE", band="aet", kind="monthly",
        scale=4638, var="et", out="et_obs/terraclimate_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * 0.1),                 # 0.1 mm scale factor
    "fldas": dict(
        coll="NASA/FLDAS/NOAH01/C/GL/M/V001", band="Evap_tavg", kind="monthly",
        scale=11132, var="et", out="et_obs/fldas_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * 86400.0 * nd),        # kg m-2 s-1 -> mm/month
    "era5land": dict(
        coll="ECMWF/ERA5_LAND/MONTHLY_AGGR", band="total_evaporation_sum",
        kind="monthly", scale=11132, var="et",
        out="et_obs/era5land_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * -1000.0),             # m (negative) -> mm/month
    # --- ET referee (benchmark-only: no calibration-window coverage) ----------
    "openet": dict(
        coll="projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0",
        band="et_ensemble_mad", kind="monthly_mosaic", scale=30,
        span=("1999-10-01", "2024-12-01"), chunk=1500, referee=True,
        var="et", out="et_obs/openet_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # mm/month; 25 tiles/month
    # --- SWE (mm monthly-mean state; terraclimate = end-of-month snapshot) ---
    "daymet_swe": dict(
        coll="NASA/ORNL/DAYMET_V4", band="swe", kind="daily_mean",
        scale=1000, var="swe", out="swe_obs/daymet_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # kg m-2 == mm
    "terraclimate_swe": dict(
        coll="IDAHO_EPSCOR/TERRACLIMATE", band="swe", kind="monthly",
        scale=4638, var="swe", out="swe_obs/terraclimate_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # mm, EOM snapshot (keep)
    "fldas_swe": dict(
        coll="NASA/FLDAS/NOAH01/C/GL/M/V001", band="SWE_inst", kind="monthly",
        scale=11132, var="swe", out="swe_obs/fldas_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # kg m-2 == mm
    "era5land_swe": dict(
        coll="ECMWF/ERA5_LAND/MONTHLY_AGGR", band="snow_depth_water_equivalent",
        kind="monthly", scale=11132, var="swe",
        out="swe_obs/era5land_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * 1000.0),              # m of water -> mm
}
#: existing 2074-cell stores for --verify (product -> path)
VERIFY_AGAINST = {
    "openet": r"D:\sacsma-data\et_processed\openet_gee_cell_monthly.npz",
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
    if spec["kind"] == "monthly_mosaic":
        return coll.mosaic()
    return coll.first()


def _reduce_month(ee, spec: dict, d: pd.Timestamp, fc, n: int) -> np.ndarray:
    """Per-cell mean of one month's image -> (n,) float array (row order)."""
    img = _month_image(ee, spec, d)
    out = np.full(n, np.nan)
    lst = fc.toList(fc.size())
    chunk = spec.get("chunk", 4500)
    for lo in range(0, n, chunk):
        sub = ee.FeatureCollection(lst.slice(lo, min(lo + chunk, n)))
        red = img.reduceRegions(collection=sub, reducer=ee.Reducer.mean(),
                                scale=spec["scale"]).getInfo()
        for f in red["features"]:
            v = f["properties"].get("mean")
            if v is not None:
                out[int(f["properties"]["i"])] = float(v)
    return out


def run_product(ee, name: str, g: pd.DataFrame, out_root: Path,
                verify: bool) -> None:
    spec = PRODUCTS[name]
    dates = (pd.date_range(*spec["span"], freq="MS") if "span" in spec
             else DATES)
    n = len(g)
    fc = _fc(ee, g)
    val = np.full((n, len(dates)), np.nan, dtype=np.float32)
    print(f"{name}: {spec['coll']}  {len(dates)} months  {n} cells @ "
          f"{spec['scale']} m", flush=True)
    for j, d in enumerate(dates):
        nd = calendar.monthrange(d.year, d.month)[1]
        raw = _reduce_month(ee, spec, d, fc, n)
        val[:, j] = spec["to_mm"](raw, nd)
        if d.month == 12:
            print(f"  {name}: {j + 1}/{len(dates)} months ({d.year}); "
                  f"cell-mean {np.nanmean(val[:, max(0, j - 11):j + 1]):.1f} mm",
                  flush=True)
    if verify:
        ref = np.load(VERIFY_AGAINST[name], allow_pickle=True)
        order = {str(k): i for i, k in enumerate(ref["keys"])}
        idx = np.array([order[str(k)] for k in g["key"]])
        rd = list(pd.to_datetime(ref["dates"]))
        keep = [j for j, d in enumerate(dates) if d in set(rd)]
        cols = [rd.index(dates[j]) for j in keep]
        new = val[:, keep]
        old = ref[spec["var"]].astype(np.float64)[idx][:, cols]
        m = np.isfinite(new) & np.isfinite(old)
        rel = float(np.sqrt(np.mean((new[m] - old[m]) ** 2))
                    / (np.std(old[m]) + 1e-9))
        print(f"  DELTA {name}: rel RMS {rel:.2e} vs the legacy snapshot "
              f"{VERIFY_AGAINST[name]} (asset-drift report, not a gate)",
              flush=True)
        return
    out = out_root / spec["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = (f"exported {pd.Timestamp.today().date()} | {spec['coll']}:"
            f"{spec['band']} | cell-rectangle mean @ {spec['scale']} m")
    np.savez_compressed(out, keys=g["key"].astype(str).to_numpy(),
                        dates=dates.to_numpy(), lat=g["lat"].to_numpy(),
                        lon=g["lon"].to_numpy(), meta=np.array(meta),
                        **{spec["var"]: val})
    nan_frac = float(np.isnan(val).mean())
    print(f"wrote {out}  (domain-mean {np.nanmean(val) * 12:.0f} mm/yr, "
          f"NaN frac {nan_frac:.4f})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["all"],
                    help=f"subset of {sorted(PRODUCTS)} or 'all'")
    ap.add_argument("--verify", action="store_true",
                    help="re-ingest ONLY the legacy stores' 2074 cells and "
                         "REPORT the delta vs the D:\\ snapshot (asset-drift "
                         "documentation, not a gate)")
    ap.add_argument("--out-root", default="data/region")
    ap.add_argument("--project", default=None,
                    help="Earth-Engine-registered cloud project id (required "
                         "unless your default credentials carry one)")
    args = ap.parse_args()
    try:
        import ee
        ee.Initialize(project=args.project)
    except Exception as e:                                    # noqa: BLE001
        sys.exit(f"earthengine-api not ready ({e}); run `pip install "
                 "earthengine-api` + `earthengine authenticate`, and pass "
                 "--project <an-EE-registered-cloud-project>")
    if args.products == ["all"]:   # referees (openet) run explicitly only
        names = sorted(p for p in PRODUCTS if not PRODUCTS[p].get("referee"))
    else:
        names = args.products
    g = _cells(args.verify)
    for name in names:
        run_product(ee, name, g, Path(args.out_root), args.verify)


if __name__ == "__main__":
    main()
