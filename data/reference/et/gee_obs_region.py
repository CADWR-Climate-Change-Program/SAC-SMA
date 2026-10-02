"""REGION ET/SWE observation ingest from Google Earth Engine.

Re-exports the 7 GEE-derived obs products (3 ET + 4 SWE) over the region
1/16-deg grid (``data/inputs/grid/grid_cells.csv``, 4410 cells = the modeling
domains ∪ the full CalSim3 gpkg footprint) as per-cell monthly npz files
(keys/dates/<var>/lat/lon), kept as reference observations (no code reads them).  The two non-GEE products (GLEAM, FLUXCOM) have
local raw sources and their own ingest (data/reference/et/local_obs_region.py).

**The region store is its own spec** (decision 2026-07-16, replacing the
"reproduce the legacy 2074-cell snapshot" gate): per-cell mean over the
1/16-deg cell rectangle at each asset's NATIVE scale, computed on the asset
versions current at export time (recorded in the npz ``meta`` field), months
1988-01..2018-12, units converted to mm/month (ET) or mm mean monthly state
(SWE; TerraClimate stays an end-of-month SNAPSHOT, not converted here).  The earlier
per-cell npz under the local staging root cannot be reproduced: GEE assets drift
(ERA5-Land was reprocessed — rel RMS ~0.2 vs the snapshot under every reduction we
tried, and the snapshot's exact pipeline is lost).  This store replaces them.

RUN ORDER (the ``sacsma-gis`` env, ``environment-gis.yml``; needs an authenticated
earthengine-api with a REGISTERED cloud project: ``earthengine authenticate`` + pass
``--project <your-ee-project>``):
  1. ``python data/reference/et/gee_obs_region.py --products all --project <id>``
     the region burn (4410 cells x 372 months; a few hours — one-time).
     Writes data/reference/et/<p>_cell_monthly.npz and
     data/reference/swe/<p>_swe_cell_monthly.npz.
  2. ``python data/reference/et/gee_obs_region.py --verify --project <id>`` (optional)
     re-ingests the legacy stores' 2074 cells and REPORTS the delta vs the
     staged snapshot per product — documentation of asset drift,
     not a gate.

"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import DATA, layout, local_path, local_value  # noqa: E402

GRID_CSV = layout.grid_cells(DATA)
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
        scale=4638, var="et", out="terraclimate_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * 0.1),                 # 0.1 mm scale factor
    "fldas": dict(
        coll="NASA/FLDAS/NOAH01/C/GL/M/V001", band="Evap_tavg", kind="monthly",
        scale=11132, var="et", out="fldas_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * 86400.0 * nd),        # kg m-2 s-1 -> mm/month
    "era5land": dict(
        coll="ECMWF/ERA5_LAND/MONTHLY_AGGR", band="total_evaporation_sum",
        kind="monthly", scale=11132, var="et",
        out="era5land_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * -1000.0),             # m (negative) -> mm/month
    # --- ET referees (benchmark-only: no calibration-window coverage) ---------
    "openet": dict(
        coll="projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0",
        band="et_ensemble_mad", kind="monthly_mosaic", scale=30,
        span=("1999-10-01", "2024-12-01"), chunk=1500, referee=True,
        var="et", out="openet_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # mm/month; 25 tiles/month
    "modis": dict(
        coll="MODIS/061/MOD16A2GF", band="ET", kind="monthly_sum", scale=500,
        span=("2000-01-01", "2025-12-01"), referee=True,
        var="et", out="modis_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * 0.1),   # sum of 8-day kg m-2 composites x 0.1
    # --- SWE (mm monthly-mean state; terraclimate = end-of-month snapshot) ---
    "daymet_swe": dict(
        coll="NASA/ORNL/DAYMET_V4", band="swe", kind="daily_mean",
        scale=1000, var="swe", out="daymet_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # kg m-2 == mm
    "terraclimate_swe": dict(
        coll="IDAHO_EPSCOR/TERRACLIMATE", band="swe", kind="monthly",
        scale=4638, var="swe", out="terraclimate_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # mm, EOM snapshot (keep)
    "fldas_swe": dict(
        coll="NASA/FLDAS/NOAH01/C/GL/M/V001", band="SWE_inst", kind="monthly",
        scale=11132, var="swe", out="fldas_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v),                       # kg m-2 == mm
    "era5land_swe": dict(
        coll="ECMWF/ERA5_LAND/MONTHLY_AGGR", band="snow_depth_water_equivalent",
        kind="monthly", scale=11132, var="swe",
        out="era5land_swe_gee_cell_monthly.npz",
        to_mm=lambda v, nd: v * 1000.0),              # m of water -> mm
}
#: existing 2074-cell stores for --verify (product -> path)
_STAGING = local_path("staging")       # ``staging`` in data/local_paths.toml
VERIFY_AGAINST = {
    **{p: _STAGING / "et_processed" / f"{p}_gee_cell_monthly.npz"
       for p in ("openet", "modis", "terraclimate", "fldas", "era5land")},
    **{f"{p}_swe": _STAGING / "swe_processed" / f"{p}_swe_gee_cell_monthly.npz"
       for p in ("daymet", "terraclimate", "fldas", "era5land")},
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
    if spec["kind"] == "monthly_sum":   # n-day composites starting in month
        return coll.sum()
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


def run_product(ee, name: str, g: pd.DataFrame, data_dir: Path,
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
    store = layout.et_obs if spec["var"] == "et" else layout.swe_obs
    out = store(data_dir, spec["out"])
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
                         "REPORT the delta vs the staged snapshot (asset-drift "
                         "documentation, not a gate)")
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--project", default=local_value("gee_project"),
                    help="Earth-Engine-registered cloud project id (default: gee_project in "
                         "data/local_paths.toml, else the one your credentials carry)")
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
        run_product(ee, name, g, Path(args.data_dir), args.verify)


if __name__ == "__main__":
    main()
