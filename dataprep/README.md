# dataprep/ — region auxiliary-data program

Goal: every processed auxiliary layer the **noah_ft fine-tune** needs, for any
basin **within the cdec15 + CalSim areas**, **in-repo** in the most compact
processed form (decision 2026-07-16; supersedes the short-lived statewide
scope).  Raw sources and the heavy daily forcing master stay on local disk
(`D:\sacsma-data`, `C:\Users\warnold_la\Local`) — the repo carries the compact
per-cell stores plus these living tools to (re)build and extend them.

## The region grid

`build_region_grid.py` → `data/region/grid_cells.csv` (**done**): **4410
cells** of the 1/16° Livneh grid — the union of

* the four modeling domains (15cdec_grid 2074 ∪ 9unimp 414 ∪ 11obs 1770 ∪
  12rim 1594 = 2480 cells; the CalSim domains overlap 15cdec heavily), and
* the **full CalSim3 footprint** — every cell whose rectangle intersects any
  polygon of `data/calsim/gis/calsim3.gpkg` (both layers: all 215 Rim
  watersheds incl. Goose Lake + the 170 Valley polygons), adding 1930 cells
  beyond the domains so every CalSim3 rim location is coverable, not just the
  ones the modeling domains carry.  14 sweep cells are dropped as absent from
  the land-only WGEN store (Delta/open-water).

Keys are normalized 5-decimal `<lat>_<lon>` (`round(x, 5)`; the calsim stores
carry 6-decimal fixed-format keys — `sacsma.dpl.data._norm_obs_key` bridges).
Membership flags `in_<domain>` + `in_calsim3_fp` per cell.  Every ingest below
targets this list.

## Layer roadmap

| layer | script | source | in-repo store | status |
|---|---|---|---|---|
| grid definition | `build_region_grid.py` | domain hruinfo/forcing keys | `data/region/grid_cells.csv` (0.1 MB) | **done** |
| statics: soilveg + LAI climatology | `build_region_statics.py` | the 4 committed per-domain sidecars | `data/region/{soilveg_continuous,lai_climatology}.csv` (~4 MB) | **partial: 2480/4410 cells** — the 1930 footprint-only cells have no committed sidecar; fill path = a raster ingest (POLARIS/LANDFIRE/3DEP/MODIS-LAI on `D:\sacsma-data\raw_gis`) gated on reproducing the committed calsim point-sample rows |
| ET obs: gleam, fluxcom | `local_obs_region.py` | `D:\sacsma-data\{gleam,fluxcom}` raw | `data/region/et_obs/*.npz` | **done** (verified 1e-7) |
| ET obs: terraclimate/fldas/era5land | `gee_obs_region.py` | GEE (`--project ee-warnold`) | `data/region/et_obs/*.npz` | **done** (exported 2026-07-16, spec v2, see below) |
| SWE obs: daymet/terraclimate/fldas/era5land | `gee_obs_region.py` | GEE (same run) | `data/region/swe_obs/*.npz` | **done** (exported 2026-07-16) |
| ET referees: openet (1999-10..2024-12), modis (2000-01..2025-12) | `gee_obs_region.py --products openet modis` | GEE (explicit run; excluded from `all` and from the training losses) | `data/region/et_obs/{openet,modis}_gee_cell_monthly.npz` | **done** (exported 2026-07-17) |
| daily forcing MASTER (raw) | `wgen_forcing.py` | WGEN NonDetrend-Unsplit statewide ASCII (local) | **local only**: `D:\sacsma-data\forcing\livneh_unsplit_nondetrend_daily_region.nc` | **done** (`--verify` proves region store == master + table) |
| ×10 precip-artifact table | `wgen_forcing.py --scan-x10` | the retired per-domain calsim stores (git history) | `data/region/prcp_x10_artifacts.csv` (197 pairs, FROZEN) | **done** |
| **UNIFIED region forcing** | `build_region_forcing.py` | master (+ table) / OneDrive Product_A / OneDrive LTO | `data/region/forcing/{historical_livneh_unsplit,wgen_product_a,historical_lto}.nc` (~3.1 GB LFS; prcp/tmin/tmax, tavg derived at load) | **done** — replaced the per-domain calsim + cdec15_grid stores (2026-07-16); parity gate re-passed |

**Verification rules**: `local_obs_region.py --verify` reproduces the legacy
2074-cell `D:\sacsma-data` npz (rel RMS < 1e-3 — PASSED at 1e-7: the raw
sources are local and pinned); `wgen_forcing.py --verify` reproduces the
committed forcing + tminmax stores from the master (PASSED).  **The GEE
products are the exception** (decision 2026-07-16): the reproduce-the-snapshot
gate FAILED — daymet was a fixable scale error (1 km native, not 11132 m),
but ERA5-Land shows genuine asset drift (rel RMS ~0.2 vs the snapshot under
every reduction tried; GEE reprocesses assets and the original pipeline is
lost), so the snapshot is irreproducible in principle.  The region store is
therefore **its own spec** — cell-rectangle mean at each asset's native
scale, asset versions as of the export date (recorded in each npz's ``meta``)
— and everything that consumed the old snapshot is **retrained on it**:
`noah_ft` (warm-start fine-tune of `noah`, which is flow-only and unaffected)
→ the +2 °C teacher → the `hybrid`/`hybrid_pet_dt` ensembles.
`gee_obs_region.py --verify` remains as an asset-drift REPORT vs the legacy
snapshot, not a gate.  The legacy `D:\` npz stay frozen as the record of what
the pre-region canonical trained on.

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
  **The table is exact only where a committed calsim store exists.**  The
  OneDrive `BASE/WGEN/Historical_Unsplit` copy was checked and is identical
  to the local store (raw) — the corrections live only in the original
  study's per-domain meteo files, so no corrected reference exists for the
  footprint-only cells, and no value threshold reproduces the known table
  (the upstream fix was station-informed).  `--cut` therefore WARNS about
  suspect cell-days there (≥30 mm on a known artifact day and >2× the cell's
  other-summer max) instead of editing them — a human decision.

## GEE export runbook (user-run)

Project = `ee-warnold` (EE-registered; the stored credentials carry none):

```
python dataprep/gee_obs_region.py --products all --project ee-warnold  # region burn, hours
python dataprep/gee_obs_region.py --verify --project ee-warnold        # optional drift report
```

Outputs land in `data/region/{et_obs,swe_obs}/*.npz` (LFS via `data/**/*.npz`).
The `sacsma/dpl/data.py` `ET_DIR`/`SWE_DIR` defaults were flipped to the
in-repo store 2026-07-16 (env overrides kept; the frozen
`D:\sacsma-data\{et,swe}_processed` snapshot is reachable via
`SACSMA_ET_DIR`/`SACSMA_SWE_DIR`).  Drift at the level training consumes
(15cdec basin-aggregated monthly climatology): ET rel RMS 1.1%, SWE 4.1%,
snowy-basin mask unchanged — the retrain chain (noah_ft fine-tune → +2 °C
teacher → hybrid ensembles; recipes in `artifacts/dpl/RUNS.md`) reruns on
this basis.

## New-basin setup (the end state)

A basin inside the region needs only a delineation + a gage/FNF target:

1. Cells: select from `data/region/grid_cells.csv` (or intersect the
   delineation with the 1/16° grid).
2. Forcing: `python dataprep/wgen_forcing.py --cut <name> --cells <csv>
   --out-dir <dir>` → `historical_livneh_unsplit_<name>.nc` (prcp+tavg) +
   `tminmax_livneh_percell_<name>.nc`, in the committed cdec15_grid schema
   (×10 artifact days corrected by default; `--no-fix-x10` for the raw
   cdec15 convention).
3. Statics: rows from `data/region/{soilveg_continuous,lai_climatology}.csv`
   (currently the 2480 modeling-domain cells; footprint-only cells await the
   raster ingest — see the roadmap).
4. Obs losses: the region npz stores cover the cells (the `data.py`
   defaults; SACSMA_ET_DIR / SACSMA_SWE_DIR override).
