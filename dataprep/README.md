# `dataprep/` — region auxiliary-data build tools

Living tools that build and extend the **region auxiliary-data store**
(`data/region/`): the compact, processed per-cell layers the dPL models draw on
for training, fine-tuning, and new-basin setup, for any basin within the
cdec15 + CalSim areas. Raw sources and the heavy daily forcing master stay on
local disk; the repo carries only the compact processed stores plus these scripts
to rebuild and extend them. The stores and their full provenance are catalogued in
[`../data/INVENTORY.md`](../data/INVENTORY.md) (§`data/region`).

## The region grid

`build_region_grid.py` → `data/region/grid_cells.csv`: **4410 cells** of the 1/16°
Livneh grid, the union of

- the four modeling domains (15cdec_grid ∪ 9unimp ∪ 11obs ∪ 12rim = 2480 cells), and
- the **full CalSim3 footprint** — every cell intersecting any polygon of
  `data/calsim/gis/calsim3.gpkg` (all 215 Rim watersheds including Goose Lake, plus
  the 170 Valley polygons), adding 1930 cells so every CalSim3 location is coverable.

Keys are normalized 5-decimal `<lat>_<lon>`; per-cell flags `in_<domain>` and
`in_calsim3_fp` mark membership. Every ingest below targets this list.

## Layers

| layer | script | in-repo store | status |
|---|---|---|---|
| grid definition | `build_region_grid.py` | `grid_cells.csv` (0.1 MB) | done |
| statics: soil/veg + LAI climatology | `build_region_statics.py` | `{soilveg_continuous,lai_climatology}.csv` (~4 MB) | **partial: 2480/4410 cells** — footprint-only cells need a raster ingest (see below) |
| ET obs: gleam, fluxcom | `local_obs_region.py` | `et_obs/*.npz` | done (verified to 1e-7) |
| ET/SWE obs: terraclimate/fldas/era5land/daymet | `gee_obs_region.py` | `et_obs/*.npz`, `swe_obs/*.npz` | done (GEE spec v2, 2026-07-16) |
| ET referees: openet, modis | `gee_obs_region.py --products openet modis` | `et_obs/{openet,modis}_*.npz` | done (benchmark-only, 2026-07-17) |
| daily forcing master (raw) | `wgen_forcing.py` | local only (not in repo) | done |
| ×10 precip-artifact table | `wgen_forcing.py --scan-x10` | `prcp_x10_artifacts.csv` (frozen) | done |
| **unified region forcing** | `build_region_forcing.py` | `forcing/{historical_livneh_unsplit,wgen_product_a,historical_lto}.nc` (~3.1 GB LFS) | done — replaced the per-domain stores (2026-07-16) |

## Verification

Every local ingest reproduces its committed or legacy predecessor before it lands.
`local_obs_region.py --verify` reproduces the legacy 2074-cell npz (rel RMS < 1e-3,
achieved 1e-7); `wgen_forcing.py --verify` reproduces the committed forcing stores
from the master; `build_region_forcing.py` re-passes the SAC-SMA parity gate for
every domain with a simflow reference (KGE > 0.9999).

**The GEE products are the exception.** The reproduce-the-snapshot gate failed:
ERA5-Land shows genuine asset drift (GEE reprocesses its assets and the original
pipeline is lost), so the snapshot is irreproducible in principle. The region GEE
store is therefore **its own spec** — a cell-rectangle mean at each asset's native
scale, with the asset versions recorded in each npz's `meta`. `gee_obs_region.py
--verify` stays on as a drift *report* against the legacy snapshot, not a gate. At
the level training consumes — 15-basin monthly climatologies — the drift is small
(ET rel RMS 1.1%, SWE 4.1%, snowy-basin mask unchanged), so anything built on the
old snapshot re-runs cleanly on this basis. The frozen legacy npz stay as the
record of what the pre-region models trained on.

## GEE export runbook (user-run)

Project = `ee-warnold` (EE-registered):

```bash
python dataprep/gee_obs_region.py --products all --project ee-warnold   # region burn, hours
python dataprep/gee_obs_region.py --verify --project ee-warnold         # optional drift report
```

Outputs land in `data/region/{et_obs,swe_obs}/*.npz` (LFS). The `dpl/data.py`
`ET_DIR`/`SWE_DIR` defaults point at the in-repo store; `SACSMA_ET_DIR` /
`SACSMA_SWE_DIR` override them to a frozen local snapshot.

## New-basin setup

A basin inside the region needs only a delineation and a gage/FNF target:

1. **Cells** — select from `grid_cells.csv`, or intersect the delineation with the
   1/16° grid.
2. **Forcing** — `python dataprep/wgen_forcing.py --cut <name> --cells <csv>
   --out-dir <dir>` writes `historical_livneh_unsplit_<name>.nc` (prcp + tavg) plus
   a per-cell tmin/tmax sidecar, with the ×10 artifact days corrected by default.
3. **Statics** — rows from `{soilveg_continuous,lai_climatology}.csv` (currently the
   2480 modeling-domain cells; footprint-only cells await the raster ingest).
4. **Obs losses** — the region npz cover the cells via the `data.py` defaults.

## Known gap

The statics stores cover only the 2480 modeling-domain cells. The 1930
footprint-only cells need a raster ingest (POLARIS / LANDFIRE / 3DEP / MODIS-LAI
from `data/raw_gis/`, see [`../data/raw_gis/SOURCES.md`](../data/raw_gis/SOURCES.md)),
gated on reproducing the committed point-sample rows, before any full-region
training.
