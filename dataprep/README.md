# dataprep/ — region auxiliary-data program

Goal: every processed auxiliary layer the **noah_ft fine-tune** needs, for any
basin **within the cdec15 + CalSim areas**, **in-repo** in the most compact
processed form (decision 2026-07-16; supersedes the short-lived statewide
scope).  Raw sources and the heavy daily forcing master stay on local disk
(`D:\sacsma-data`, `C:\Users\warnold_la\Local`) — the repo carries the compact
per-cell stores plus these living tools to (re)build and extend them.

## The region grid

`build_region_grid.py` → `data/region/grid_cells.csv` (**done**): **2480
cells** of the 1/16° Livneh grid — the union of the four modeling domains
(15cdec_grid 2074 ∪ 9unimp 414 ∪ 11obs 1770 ∪ 12rim 1594; the CalSim
footprints overlap 15cdec heavily — only 406 cells are calsim-only).  Keys are
normalized 5-decimal `<lat>_<lon>` (`round(x, 5)`; the calsim stores carry
6-decimal fixed-format keys — `sacsma.dpl.data._norm_obs_key` bridges).
Membership flags `in_<domain>` per cell.  Every ingest below targets this list.

## Layer roadmap

| layer | script | source | in-repo store | status |
|---|---|---|---|---|
| grid definition | `build_region_grid.py` | domain hruinfo/forcing keys | `data/region/grid_cells.csv` (0.1 MB) | **done** |
| statics: soilveg + LAI climatology | `build_region_statics.py` | the 4 committed per-domain sidecars | `data/region/{soilveg_continuous,lai_climatology}.csv` (~4 MB) | **done** |
| ET obs: gleam, fluxcom | `local_obs_region.py` | `D:\sacsma-data\{gleam,fluxcom}` raw | `data/region/et_obs/*.npz` | **done** (verified 1e-7) |
| ET obs: terraclimate/fldas/era5land | `gee_obs_region.py` | GEE (user-run export) | `data/region/et_obs/*.npz` | script ready — needs `--project` |
| SWE obs: daymet/terraclimate/fldas/era5land | `gee_obs_region.py` | GEE (user-run export) | `data/region/swe_obs/*.npz` | script ready (same run) |
| daily forcing MASTER | `wgen_forcing.py` | WGEN NonDetrend-Unsplit statewide ASCII (local) | **local only**: `D:\sacsma-data\forcing\livneh_unsplit_nondetrend_daily_region.nc` | **done** (+ `--cut` for new basins) |
| ×10 precip-artifact table | `wgen_forcing.py --scan-x10` | committed calsim stores vs the master | `data/region/prcp_x10_artifacts.csv` (197 pairs) | **done** |

**Verification rule for every ingest**: reproduce the existing committed/legacy
store first (the original ingest scripts were session scratch and are lost —
the stores define correctness).  `local_obs_region.py --verify` and
`gee_obs_region.py --verify` diff against the 2074-cell `D:\sacsma-data` npz
(rel RMS < 1e-3 required); `wgen_forcing.py --verify` reproduces the committed
forcing + tminmax stores from the master.

## Provenance notes (established 2026-07-16)

- **The WGEN NonDetrend-Unsplit statewide store IS the historical forcing
  lineage**: `prcp` matches every committed `historical_livneh_unsplit*` store
  to float32 rounding, and the committed `tavg` is exactly `(tmax+tmin)/2` of
  its ASCII (calsim stores match to their 3-decimal write precision, ≤5e-3).
  No OneDrive mount is needed for historical forcing.
- **Obs-product methods**: GLEAM = nearest-neighbor sample of the 0.1°
  monthly `E` (mm/month), 1988–2018.  FLUXCOM RS_METEO = nearest-neighbor
  sample of the 0.5° monthly `LE` (MJ m⁻² d⁻¹), ET = LE/2.45 × days,
  1988–2016.  GEE products: per-cell rectangle means at 11132 m
  (parameters in `D:\sacsma-data\et_processed\_ingest_*.log`).
- **Statics conventions differ by lineage**: the cdec15_grid sidecars are
  cell-footprint means (dPL-training convention); the calsim sidecars are
  cell-center point samples (median |Δdem_elev| ≈ 93 m over shared cells).
  `data/region/soilveg_continuous.csv` keeps cdec15_grid rows where they
  exist and fills the 406 calsim-only cells from the calsim sidecars — the
  `src` column marks the seam.
- **The ×10 precipitation artifact**: the raw lineage carries misplaced-decimal
  precip spikes — 197 (cell, day) pairs over 168 region cells on 7 isolated
  summer days (1916-07-01, 1954-08-28, 1974-07-08..10, 1976-08-15,
  1980-07-02), each exactly 10× too large (up to 569 mm/day).  The
  CalSim-domain ingest corrected them (÷10, consistent across all three
  stores); the **cdec15 lineage kept them raw** — 146 of the cells are shared
  with the committed `cdec15_grid` store, which the GA/dPL calibrations
  trained on (a pre-existing upstream inconsistency between the two
  applications).  The master stays bit-faithful to the raw source;
  `data/region/prcp_x10_artifacts.csv` (from `--scan-x10`) is the auditable
  correction, applied by `--cut` by default (`--no-fix-x10` reproduces the
  raw cdec15 convention) and expected by `--verify` at the calsim stores.

## GEE export runbook (user-run)

Needs an EE-registered cloud project (the stored credentials carry none):

```
earthengine authenticate                              # if stale
python dataprep/gee_obs_region.py --verify --project <your-ee-project>   # must PASS all 7
python dataprep/gee_obs_region.py --products all --project <your-ee-project>  # ~1-2 h
```

Outputs land in `data/region/{et_obs,swe_obs}/*.npz` (LFS via `data/**/*.npz`).
After all 9 obs products exist region-wide, flip `sacsma/dpl/data.py`
`ET_DIR`/`SWE_DIR` defaults to the in-repo store (env overrides kept) and
retire the `D:\sacsma-data\{et,swe}_processed` dependency.

## New-basin setup (the end state)

A basin inside the region needs only a delineation + a gage/FNF target:

1. Cells: select from `data/region/grid_cells.csv` (or intersect the
   delineation with the 1/16° grid).
2. Forcing: `python dataprep/wgen_forcing.py --cut <name> --cells <csv>
   --out-dir <dir>` → `historical_livneh_unsplit_<name>.nc` (prcp+tavg) +
   `tminmax_livneh_percell_<name>.nc`, in the committed cdec15_grid schema
   (×10 artifact days corrected by default; `--no-fix-x10` for the raw
   cdec15 convention).
3. Statics: rows from `data/region/{soilveg_continuous,lai_climatology}.csv`.
4. Obs losses: the region npz stores cover the cells (SACSMA_ET_DIR /
   SACSMA_SWE_DIR until the data.py defaults flip).
