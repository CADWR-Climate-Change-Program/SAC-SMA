"""REGION ET-obs ingest for the two locally-sourced products: GLEAM + FLUXCOM.

Re-ingests the two non-GEE ET observation products over the region grid
(``data/inputs/grid/grid_cells.csv``) as per-cell monthly npz files
(keys/dates/et/lat/lon).  They are kept as reference observations; no code reads them.

Methods reverse-engineered from the existing 2074-cell stores (the original
ingest scripts were session scratch; the stores define correctness — verified
2026-07-16, rel RMS ~1e-7):

* **GLEAM v4.3a** (``gleam/v4.3a_monthly_E`` under the local staging root, 0.1-deg global
  monthly ``E`` in mm/month): nearest-neighbor sample at the cell center,
  1988-01..2018-12.
* **FLUXCOM RS_METEO CRUNCEP v8** (``fluxcom/..._LE_monthly`` under the staging root,
  0.5-deg global monthly ``LE`` in MJ m-2 d-1): nearest-neighbor sample,
  ET mm/month = LE / 2.45 * days_in_month, 1988-01..2016-12 (product ends 2016).

RUN ORDER:
  1. ``python data/reference/et/local_obs_region.py --verify``
     re-ingests at the existing stores' 2074 cells and diffs (rel RMS < 1e-3
     required — catches method/unit drift).
  2. ``python data/reference/et/local_obs_region.py``
     the region ingest -> data/reference/et/{gleam,fluxcom}_cell_monthly.npz.
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import DATA, layout, local_path  # noqa: E402

GRID_CSV = layout.grid_cells(DATA)
#: the local data staging root (``staging`` in data/local_paths.toml)
STAGING = local_path("staging")
PRODUCTS: dict[str, dict] = {
    "gleam": dict(
        raw=STAGING / "gleam" / "v4.3a_monthly_E",
        pattern="E_{year}_GLEAM_v4.3a_MO.nc", band="E",
        span=("1988-01-01", "2018-12-01"),
        to_mm=lambda v, nd: v,                      # already mm/month
        out="gleam_cell_monthly.npz",
        verify=STAGING / "et_processed" / "gleam_cell_monthly.npz"),
    "fluxcom": dict(
        raw=STAGING / "fluxcom" / "RS_METEO_CRUNCEP_v8_LE_monthly",
        pattern="LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-CRUNCEP_v8.720_360.monthly.{year}.nc",
        band="LE",
        span=("1988-01-01", "2016-12-01"),
        to_mm=lambda v, nd: v / 2.45 * nd,          # MJ m-2 d-1 -> mm/month
        out="fluxcom_cell_monthly.npz",
        verify=STAGING / "et_processed" / "fluxcom_cell_monthly.npz"),
}


def ingest(name: str, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, pd.DatetimeIndex]:
    spec = PRODUCTS[name]
    dates = pd.date_range(*spec["span"], freq="MS")
    val = np.full((len(lats), len(dates)), np.nan, dtype=np.float32)
    la = xr.DataArray(lats, dims="cell")
    lo = xr.DataArray(lons, dims="cell")
    for year in sorted({d.year for d in dates}):
        with xr.open_dataset(Path(spec["raw"]) / spec["pattern"].format(year=year)) as ds:
            arr = ds[spec["band"]].sel(lat=la, lon=lo, method="nearest")
            for t in pd.to_datetime(np.asarray(arr["time"])):
                d = pd.Timestamp(year=t.year, month=t.month, day=1)
                if d not in dates:
                    continue
                j = dates.get_loc(d)
                nd = calendar.monthrange(d.year, d.month)[1]
                raw = arr.sel(time=t).to_numpy().astype(np.float64)
                val[:, j] = spec["to_mm"](raw, nd)
    return val, dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=sorted(PRODUCTS))
    ap.add_argument("--verify", action="store_true",
                    help="re-ingest at the existing stores' cells and diff "
                         "(run FIRST, must PASS)")
    ap.add_argument("--data-dir", default=str(DATA))
    args = ap.parse_args()

    grid = pd.read_csv(GRID_CSV)
    for name in args.products:
        spec = PRODUCTS[name]
        if args.verify:
            ref = np.load(spec["verify"], allow_pickle=True)
            lats, lons = ref["lat"].astype(float), ref["lon"].astype(float)
            val, dates = ingest(name, lats, lons)
            rd = list(pd.to_datetime(ref["dates"]))
            cols = [rd.index(d) for d in dates]
            old = ref["et"].astype(np.float64)[:, cols]
            m = np.isfinite(val) & np.isfinite(old)
            rel = float(np.sqrt(np.mean((val[m] - old[m]) ** 2))
                        / (np.std(old[m]) + 1e-9))
            status = "PASS" if rel < 1e-3 else "FAIL"
            print(f"VERIFY {name}: rel RMS {rel:.2e} over {len(lats)} cells x "
                  f"{len(dates)} months vs {spec['verify']} -> {status}")
            continue
        val, dates = ingest(name, grid["lat"].to_numpy(), grid["lon"].to_numpy())
        out = layout.et_obs(args.data_dir, spec["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, keys=grid["key"].astype(str).to_numpy(),
                            dates=dates.to_numpy(), lat=grid["lat"].to_numpy(),
                            lon=grid["lon"].to_numpy(), et=val)
        print(f"wrote {out}: {val.shape[0]} cells x {val.shape[1]} months "
              f"(domain-mean {np.nanmean(val) * 12:.0f} mm/yr, "
              f"NaN frac {float(np.isnan(val).mean()):.4f})")


if __name__ == "__main__":
    main()
