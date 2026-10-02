# data/: what the models read, what they are fitted to, what they are compared with

The store has three parts, by the role the data plays.

| Part | Role | Where it comes from |
|---|---|---|
| [`inputs/`](inputs/) | What a run reads to produce flow: forcing, the grid and its static attributes, the modeling domains, the CalSim3 geometry and mappings. | Partly delivered, partly built in this repository. |
| [`targets/`](targets/) | Flow records a model is fitted to. | All from outside this repository. |
| [`reference/`](reference/) | Series that results are only compared with. | All from outside this repository. |

Two rules go with the layout.

1. **Nothing under `sacsma/` writes into `targets/` or `reference/`.** Those folders change only
   through the ingest script that sits in them. The package writes two generated input tables
   (the canonical areas and the screened footprints in `inputs/calsim3/`) and nothing else.
2. **A series that any model here is fitted to is a target**, even if another model only scores
   against it. The CalSim3 inflows are the case: dPL-CalSim trains on them, the calibrated
   SAC-SMA comparison only scores against them.

Each dataset folder holds its data, a short `README.md`, and the script that builds or ingests
it. The README is the provenance record of that folder.

## The folders

| Folder | What | Source | Built by | Size |
|---|---|---|---|---|
| [`inputs/forcing/`](inputs/forcing/README.md) | Daily precipitation and temperature: five products on the 1/16° grid, and the dense store of the 15cdec HRU points. The table of corrected ×10 precipitation spikes. The WGEN scenario key. | Livneh-unsplit, WGEN Product A, Historical LTO, NOAA AORC | `build_region_forcing.py`, `wgen_forcing.py`, `wgen_product_a_scenarios.py`, `aorc_region.py` | 5.7 GB |
| [`inputs/grid/`](inputs/grid/README.md) | The 4,410-cell grid and its per-cell attributes: soil, vegetation, terrain, leaf-area climatology, AlphaEarth embeddings. | POLARIS, LANDFIRE, 3DEP, MODIS, Earth Engine | `build_region_grid.py`, `build_region_statics.py`, `sample_gis.py`, `download_gis.py`, `gee_aef_region.py` | 8 MB |
| [`inputs/domains/`](inputs/domains/README.md) | The five calibration domains (`15cdec`, `15cdec_grid`, `9unimp`, `11obs`, `12rim`): HRU tables, calibrated parameters, areas. | Wi and Steinschneider study archive | delivered | 39 MB |
| [`inputs/domains/multifamily/`](inputs/domains/multifamily/README.md) | The training-entity registry of the learned-parameter models: entities, their cells and weights, flow lengths. | built here | `build_entities.py`, `build_entity_cells.py`, `build_flowlens.py` | 1 MB |
| [`inputs/calsim3/`](inputs/calsim3/README.md) | CalSim3 catchment polygons, the crosswalk between watersheds and CalSim3 arcs, the canonical areas, the screened footprints, the tier-1 location sets. | DWR polygons; the rest built here or kept by hand | `sacsma.calsim.catchments` (the two generated tables) | 2 MB |
| [`targets/cdec/`](targets/cdec/README.md) | CDEC daily full natural flow: the 15 watersheds of the original study, and the later pulls. | CDEC | `cdec_fnf.py`, `build_fnf_depth.py` | 7 MB |
| [`targets/callite/`](targets/callite/README.md) | Monthly full-natural-flow targets of the `9unimp`, `11obs` and `12rim` sets. | study archive | delivered | 3 MB |
| [`targets/dwr_unimpaired/`](targets/dwr_unimpaired/README.md) | DWR's published unimpaired flows, 24 Central Valley subbasins, water years 1922 to 2014. | DWR (2016) report | `dwr_unimpaired.py`, `build_uf_depth.py`, `check_uf_locations.py` | 1 MB |
| [`targets/usgs/`](targets/usgs/README.md) | Daily flow at 69 USGS gauges inside the CalSim3 domain. | USGS, through the neuralhyd-ca cleaned set | `usgs_flows.py` | 6 MB |
| [`targets/calsim3/`](targets/calsim3/README.md) | CalSim3 monthly inflows on the rim arcs, the unimpaired series of the rim systems, and the tables that say which arc-months are an arc's own record. | CalSim3; DWR hydrology report | `build_calsim_arcs.py` | 20 MB |
| [`reference/matlab/`](reference/matlab/README.md) | The archived MATLAB simulations: the parity baseline. | study archive | delivered | 49 MB |
| [`reference/vic/`](reference/vic/README.md) | VIC routed monthly flows under three climates, and the VIC routing grids. | CalSim3 stochastic-input pipeline | delivered | 31 MB |
| [`reference/bcm/`](reference/bcm/README.md) | Basin Characterization Model v8, two weather-generator scenarios. | USGS | `bcm_region.py` | 202 MB |
| [`reference/et/`](reference/et/README.md) | Nine evapotranspiration products on the grid. | Earth Engine, GLEAM, FLUXCOM, Reitz et al. (2023) | `gee_obs_region.py`, `local_obs_region.py`, `reitz_et.py` | 39 MB |
| [`reference/swe/`](reference/swe/README.md) | Four snow-water-equivalent products on the grid. | Earth Engine | `../et/gee_obs_region.py` | 8 MB |
| [`reference/dwr_swat/`](reference/dwr_swat/README.md) | DWR's SWAT simulation of the rim watersheds and the valley and Delta totals. | DWR (2016) report | `../../targets/dwr_unimpaired/dwr_unimpaired.py` | 1 MB |

## Conventions

- **Join key.** `key = "<lat>_<lon>"` links an HRU to its parameters and to its forcing cell.
  On the 1/16° grid the key is the cell centre to five decimals.
- **Forcing is by grid cell, not by watershed.** A store has dimensions `(key, time)`. One cell
  can feed several HRUs; what belongs to an HRU (elevation, flow length, area weight, watershed)
  is in the domain's `hruinfo.csv`.
- **Units.** Flow is mm/day over the watershed area; the `basin_area.csv` tables convert to
  cfs. Monthly CalSim3 and VIC series are TAF per month. The USGS store keeps cfs as delivered
  and a derived mm/day.
- **Watershed codes** differ by domain: CDEC codes for `15cdec` and `11obs` (SHA, BND, and so
  on); names for `9unimp` (CacheCreek, StonyCreek, and so on); codes of its own for `12rim`
  (SHAST, OROVI, and so on). The same river can carry different codes in different domains.
- **Formats.** Tables are plain CSV. Gridded stores are NetCDF or npz, tracked with git-LFS
  (`.gitattributes`); so are four large CSV tables.

## Files kept by hand

These are sources of truth. No script overwrites them.

| File | What |
|---|---|
| `inputs/calsim3/calsim_crosswalk.csv` | Which CalSim3 arc belongs to which watershed and rim system. |
| `inputs/calsim3/tier1_sets.csv` | The 20 tier-1 locations. Only its two window columns are written by a tool (`sacsma dpl calsim windows --write`). |
| `targets/calsim3/calsim3_arc_derivation.csv` | How CalSim3 built each rim inflow series, transcribed from the DWR hydrology report. |
| `targets/dwr_unimpaired/uf_gauges.csv` | Pour point and report area of each unimpaired-flow subbasin. |
| `targets/cdec/fnf_daily_mask.csv` | Daily target values confirmed to be wrong, masked in training. |

Three more files need care.

- `targets/dwr_unimpaired/uf_locations.csv` is written by the ingest script and then corrected
  by hand in a few cells. A rerun of the ingest loses the corrections (see that README).
- `inputs/forcing/prcp_x10_artifacts.csv` is a frozen result. The scan that produced most of it
  can no longer run, and would drop the rows that were found later. Do not regenerate it.
- `inputs/forcing/historical_livneh_unsplit.nc` was patched in place for the last 92 rows of
  that table. It is the store every tracked result was made with. Do not overwrite it.

## The build scripts

A script is run by its path from the repository root, for example
`python data/targets/cdec/cdec_fnf.py --help`. Each one starts from
[`_paths.py`](_paths.py), which gives it the repository root, the data layout, and the places
where this machine keeps inputs that are not in the repository.

- **The layout is one module**, [`sacsma/paths.py`](../sacsma/paths.py). The package and the
  scripts ask it for every path under `data/`. To move a folder, change it there.
- **Inputs outside the repository** (the raw raster stage, the forcing master, the weather
  generator releases, the observation downloads) are named by key in `local_paths.toml`, which
  is not tracked. Copy [`local_paths.example.toml`](local_paths.example.toml) and fill in the
  keys you need. A key can also be set as the environment variable `SACSMA_<KEY>`.
- **Environments.** Most scripts run in the `sacsma` environment. The raster and Earth Engine
  scripts (`sample_gis.py`, `reitz_et.py`, `build_flowlens.py`, `gee_obs_region.py`,
  `gee_aef_region.py`) run in `sacsma-gis` (`environment-gis.yml`). `usgs_flows.py` needs the
  environment of the neuralhyd-ca repository.

The order in which the tables depend on each other, and what a clone cannot rebuild, are in
[`docs/reproduce.md`](../docs/reproduce.md).
