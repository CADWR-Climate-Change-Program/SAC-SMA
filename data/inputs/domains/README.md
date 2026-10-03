# data/inputs/domains: the five calibration domains

A domain is a group of watersheds with their modeling units: the HRU table and the drainage areas. Its
calibrated parameters, the archived GA optimum, are in
[`artifacts/models/`](../../../artifacts/models/README.md) (`15cdec/`, `15cdec_grid/`, `callite/<domain>/`).
Role: input. The tables come from the archived MATLAB-era study materials of Wi &
Steinschneider (Cornell / UMass Amherst; California DWR watershed studies). One folder per domain. The
sixth domain, `multifamily`, has [its own README](multifamily/README.md).

| Domain | HRU rows | Cells (distinct `key`) | Watershed codes | What it is |
|---|---|---|---|---|
| `15cdec` | 7,891 | 6,033 | CDEC codes (SHA, BND, ...) | The 15 CDEC reservoir watersheds. HRU points are study-specific centroids off the 1/16° grid. One pooled calibration against daily CDEC full natural flow. |
| `15cdec_grid` | 2,802 | 2,074 | the same 15 codes | The coarse, grid-aligned parallel of `15cdec`: one unit per native 1/16° Livneh cell, in place of the about 3.8 times denser HRU cloud. |
| `9unimp` | 466 | 414 | names (CacheCreek, StonyCreek, ...) | 9 CalLite watersheds, calibrated one by one against monthly full natural flow. |
| `11obs` | 2,448 | 1,770 | CDEC codes (SHA, BND, ...) | 11 CalLite watersheds, calibrated likewise. |
| `12rim` | 1,756 | 1,594 | five-character codes (SHAST, OROVI, FOL_I, ...) | 12 CalLite watersheds, calibrated likewise. Not in the CalSim3 cross-compare. |

Counts are from the `hruinfo.csv` files. `9unimp`, `11obs`, `12rim` and `15cdec_grid` sit on the region
grid (see [the grid README](../grid/README.md)).

## Files

| File | What | Source |
|---|---|---|
| `<domain>/hruinfo.csv` (all five) | One row per HRU: `basin, lat, lon, area_weight, elev, flowlen, soil_class, veg_class, key` (plus `basin_id`, `basin_id2` except in `15cdec_grid`). `area_weight` is the HRU's percentage of its watershed. A cell shared by two watersheds appears once per watershed. | The study's HRUinfo tables |
| `<domain>/basin_area.csv` (`15cdec`, `9unimp`, `11obs`, `12rim`) | `[basin, area_mi2]`. `15cdec`: the published drainage areas of the 15 watersheds. `9unimp`, `11obs`: the authoritative drainage areas. `12rim`: the areas the original study's CalLite wrapper converted with (its m² over 1609.344²; its `11obs` areas agree with the table here to 0.01 mi²), used only to write the CalLite product. Converts mm/day to cfs and back. | Study archive; `12rim`: the study's MATLAB CalLite wrapper |
| `<domain>/soilveg_continuous.csv` (all five) | Continuous soil, vegetation and terrain attributes: POLARIS `sand`, `clay`, `ksat`, `theta_s` at six depths, LANDFIRE `EVC`, `EVH` and the decoded cover % and height, 3DEP elevation, slope, aspect sin and cos, curvature and relief, LAI mean, min, max, amplitude and peak day, with the `polaris_gapfill` and `lai_gapfill` flags (`15cdec_grid` has only the first). One row per HRU in `hruinfo` order, keyed by `key` (not unique). In `15cdec_grid`: one row per cell (2,074). | `data/inputs/grid/sample_gis.py <domain>`; `15cdec_grid`: no script in the repository |
| `<domain>/lai_climatology.csv` (all five) | The MODIS leaf-area day-of-year climatology, 46 8-day values per row (`lai_doy001` to `lai_doy361`). Same rows as the table above. The `15cdec_grid` file names its key column `cellkey`. | the same |
| `15cdec/basin_tminmax_livneh.csv` | Watershed-mean daily Tmin and Tmax of the 15 watersheds: `date, tmin_<basin>, tmax_<basin>`, °C, 1915-01-01 to 2018-12-31. Ingested once from the WGEN 1/16° grid. | No script in the repository |
| `15cdec/SACSMA_15CDEC.geojson` (3.6 MB) | The original SAC-SMA delineations of the 15 watersheds (polygons, `Name` = watershed code). | Study archive |

## How it is built

`hruinfo.csv`, `basin_area.csv`, `basin_tminmax_livneh.csv` and the geojson: delivered; cannot be rebuilt
from a clone. The one-time ingest scripts were retired and are in git history (commit `ad89558` and
earlier). The `12rim` area table was transcribed from the study's MATLAB CalLite wrapper.

The attribute sidecars of `15cdec`, `9unimp`, `11obs` and `12rim` are rebuilt by the raster sampler
(environment `sacsma-gis`; raw rasters in the folder named by `raw_gis` in `data/local_paths.toml`):

    python data/inputs/grid/sample_gis.py 11obs [--layers polaris,landfire,terrain,lai]

The sampler has no `15cdec_grid` mode, and no script in the repository rebuilds that domain's two
sidecars. Products, units, encodings and the sampling conventions are in
[the grid README](../grid/README.md).

## Checks

- Parity. `hruinfo.csv` and the GA optimum with the default forcing must reproduce the archived MATLAB
  simulation: `sacsma verify parity` requires KGE > 0.9999 and a largest daily difference < 0.1 mm/day on
  one watershed per domain (results in [the MATLAB reference README](../../reference/matlab/README.md)).
  `15cdec_grid` has no MATLAB reference.
- Sidecars. `15cdec`: 7,891 of 7,891 rows finite; 35 HRUs are POLARIS gap-filled and 239 LAI gap-filled.
- `basin_tminmax_livneh.csv`: the watershed mean of Tmin and Tmax reproduces the stored `tavg` forcing to
  0.37 °C.
- `15cdec_grid`: the learned-parameter run on it (`hamon`) scores validation KGE 0.836, against 0.840 for
  the same model on the original HRUs, so the coarse grid keeps almost all of the skill
  ([Runs](../../../docs/runs.md)).

## Know before using

- Join key. `key = f"{lat:.6f}_{lon:.6f}"` links an HRU to its parameters and its forcing cell. The grid
  stores use 5-decimal keys; `sacsma.io.norm_grid_key` converts.
- Forcing is per grid cell, not per watershed. One cell can feed many HRUs; what belongs to an HRU (`elev`,
  `flowlen`, `area_weight`, soil and vegetation class, `basin`) is in `hruinfo.csv`.
- Units. Flow is mm/day over the watershed area. The `basin_area.csv` tables convert to cfs
  (`sacsma.io.cfs_to_mmday`, `mmday_to_cfs`).
- Watershed codes are specific to a domain. The same river can carry different codes (the Feather at
  Oroville is `ORO` in `15cdec` and `FTO` in `11obs`).
- caution: in the CalLite domains, filter the GA optimum by `basin` before indexing by `key`. Shared
  cells repeat with different parameters.
- `12rim` has no column in the CalSim3 crosswalk, and its parity is in mm/day. Its `basin_area.csv` serves
  only the CalLite product; no score reads it.
- `15cdec` reads its own dense forcing store and has no per-cell Tmin and Tmax, so the Priestley-Taylor
  and Noah ET options need a grid domain. `15cdec_grid` is on the grid: it reads the shared stores, and
  its sidecars supply the observed canopy (cover fraction, LAI) of the Noah ET path.
- caution: the `15cdec_grid` footprints over-reach the true catchments by 9 to 66 % in 11 of the 15
  watersheds. The dPL runs re-foot those to the CalSim3 catchments. What the coarse grid loses against
  `15cdec` is the orographic downscaling of HRU meteorology, an upstream CADWR product that is not in the
  repository.
- The `15cdec_grid` sidecars are means over the cell footprint (the convention the dPL parameter network
  was trained on). All other sidecars are point samples at the HRU point.
- The sidecars replace the one-hot `soil_class` and `veg_class` in the `physical` feature variant of
  `sacsma.dpl.features`.
- `SACSMA_15CDEC.geojson`: the four Tulare watersheds (ISB, PNF, SCC, TRM) have no CalSim3 polygons, so
  their footprints in the entity registry come from this file.
- caution: the HRU footprints of SHA and BND in `11obs` carry the endorheic Goose Lake block, and SNS and
  ChowchillaRiver over-reach their catchments. The CalSim3 comparison screens those four (see
  [the CalSim3 inputs README](../calsim3/README.md)); the calibration basis is never changed.

## Read by

- `sacsma.io`: `load_hru_table`, `load_basin_area`, `soilveg_path`, `lai_climatology_path`,
  `load_canopy_obs` (and `load_params`, which reads the GA optimum from `artifacts/models/`). Through them
  every model run (`sacsma run`, `sacsma plots`, `sacsma calsim`, `sacsma product`, `sacsma dpl ...`).
- `12rim/basin_area.csv`: `sacsma.product` (the CalLite file of the 12 rim inflows).
- `basin_tminmax_livneh.csv`: `sacsma.dpl.hybrid.data` (inputs of the hybrid and LSTM models, which add the
  diurnal range that `tavg` alone discards) and `sacsma.dpl.studies.forcing_sensitivity`.
- Build scripts: `data/inputs/grid/build_region_grid.py` and `build_region_statics.py` (cells and
  sidecars of the grid domains), `data/inputs/domains/multifamily/build_entities.py` (`15cdec` areas),
  `build_entity_cells.py` (the geojson), `build_flowlens.py` (the `15cdec_grid` flow lengths, as a check).
