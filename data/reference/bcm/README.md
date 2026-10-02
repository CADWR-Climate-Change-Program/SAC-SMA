# data/reference/bcm: BCM v8 monthly hydrology, two Weather Generator scenarios

The USGS Basin Characterization Model v8 run on the CalSim3 Weather Generator scenarios (USGS
ScienceBase, *California 270-m BCM using Weather Generator Scenarios, Part I*, parent item
`68029987d4be0210cdcc98d1`): 270 m, monthly, WY1916–2018, aggregated here to the 1/16° grid and
to the CalSim3 catchments. Role: reference. It is an independent model of the same watersheds
over the same period as the SAC-SMA calibration targets. It is not a forcing product.

Two of the release's 24 scenarios are kept, the pair that isolates warming at unchanged
precipitation. Both are the hydrology child of the release.

| Key | ScienceBase item | Scenario |
|---|---|---|
| `s01` | `69e7d5c9b66b0164d0f72e91` | Scenario 1, Baseline |
| `s13` | `69e91d03b66b0183fe17a443` | Scenario 13, 0 % ave ppt, +3 °C |

## Files

| File | What | Source |
|---|---|---|
| `bcm_s01_monthly.nc`, `bcm_s13_monthly.nc` | 4410 grid cells × 1236 months (1915-10 to 2018-09) × 6 variables, dims `(key, time)`, float32 zlib, plus `n_bcm_cells` per cell. Mean of the valid 270-m cells whose centre falls in each 1/16° cell (median 519 cells; 2,289,972 in total). 77 + 75 MB, LFS | `bcm_region.py` |
| `bcm_s01_catchments_monthly.csv`, `bcm_s13_catchments_monthly.csv` | The same aggregation onto the 386 polygons of layer `CalSim3_And_GooseLake` of `data/inputs/calsim3/calsim3.gpkg` (median 1506 cells each; 2,111,973 in total), long by `cid` and `month` (`yyyymm`). 25 MB each, LFS | `bcm_region.py` |
| `bcm_catchments.csv` | The static catchment identity `[cid, node, ct_name, type, sq_mi, n_bcm_cells]`; `cid` is the row order of the layer | `bcm_region.py` |
| `bcm_region.py` | The build script | |

The six variables are `aet` (actual evapotranspiration), `cwd` (climatic water deficit), `pck`
(snowpack, as snow water equivalent), `rch` (recharge), `run` (runoff) and `str` (soil moisture
storage). All are in mm, per the release's FGDC metadata.

## How it is built

`bcm_region.py`, in the `sacsma` environment. It needs no local input: it reads
`data/inputs/grid/grid_cells.csv` and `data/inputs/calsim3/calsim3.gpkg` and streams the source
from the public ScienceBase S3 bucket.

```bash
python data/reference/bcm/bcm_region.py --list        # the zip inventory
python data/reference/bcm/bcm_region.py --scenarios s01 --vars run --zips WY1916_19
python data/reference/bcm/bcm_region.py --all         # everything, resumable
python data/reference/bcm/bcm_region.py --status      # progress from disk
python data/reference/bcm/bcm_region.py --assemble    # partials -> the files above
```

- Source: one zip per variable and decade, one ESRI ASCII grid per month (`run1915oct.asc`).
  The two scenarios are 112 GB zipped (about 2.3 TB inflated). Each member is fetched, inflated
  in memory, reduced to 4410 + 386 numbers and discarded. Nothing raw is written to disk.
  Each grid is fixed-width text (10-character fields, 2 decimals); the reader asserts this per
  file.
- Grid, read from the files and confirmed against the release metadata: 3486 columns × 4477
  rows, cell size 270 m, NODATA −9999, lower-left corner (−374495.84, −616153.31), California
  Teale Albers (NAD83) = EPSG:3310. The corner is written to varying precision across variables
  (7 mm apart), so the script tests the origin to a 1 m tolerance and everything else exactly.
- Aggregation: each BCM cell centre is projected to longitude and latitude and assigned to the
  grid cell containing it and to the catchment polygon containing it. Each target is the mean
  over valid cells, so the NODATA mask never biases a total.
- Resume: partials are banked per month and per zip under `tmp/bcm_parts/` (not tracked), each
  written atomically. A rerun skips what is banked.

## Checks

BCM is an independent model, so the checks test agreement, not identity.

Coverage: 98.6 % of the mapped cells carry data.

Snowpack `pck` against the four products of [`data/reference/swe`](../swe/README.md)
(1988–2018, regional monthly series). The month convention was scanned, not assumed. Three of
the four peak at lag 0, so `pck` needs no shift.

| Product | r at lag −1 | r at lag 0 | r at lag +1 | Product mean SWE |
|---|---|---|---|---|
| `fldas` | 0.735 | 0.923 | 0.799 | 12.3 mm |
| `era5land` | 0.861 | 0.912 | 0.681 | 26.9 mm |
| `terraclimate` | 0.641 | 0.896 | 0.805 | 18.6 mm |
| `daymet` | 0.673 | 0.561 | 0.302 | 19.0 mm |

BCM's regional mean is 37.8 mm, above every product, as expected: 270 m resolves high-elevation
snow that an 11-km product smooths away. Daymet is not a usable referee here: its climatology
peaks in April and never melts out.

Total discharge (`run` + `rch`) against the `11obs` full-natural-flow targets, area-weighted
onto each watershed's HRU footprint, WY1922–2018:

| | Median | Range |
|---|---|---|
| Monthly r | 0.882 | 0.82 (TNL) to 0.94 (BLB) |
| Annual r | 0.959 | 0.84 (TNL) to 0.97 (AMF, TLG) |
| Percent bias | +6.4 % | −18.7 % (SHA) to +20.9 % (TNL) |

The two most negative watersheds are SHA and BND. Their HRU footprints carry the endorheic
Goose Lake block (about 1000 mi² that never reaches the gauge), which dilutes a footprint-mean
depth. Renormalising over the valid cells moves the bias by under a point.

Warming signature, Scenario 13 minus Scenario 1, regional (fluxes in mm/yr, the two state
variables as mean storage). Precipitation is unchanged, so this is a pure temperature response.

| Variable | Scenario 1 (baseline) | Scenario 13 (+3 °C) | Change |
|---|---|---|---|
| `pck` (mean SWE) | 38.7 mm | 14.0 mm | −63.9 % |
| `cwd` | 855 mm/yr | 917 mm/yr | +7.4 % |
| `run` | 129 mm/yr | 138 mm/yr | +7.1 % |
| `aet` | 359 mm/yr | 367 mm/yr | +2.1 % |
| `rch` | 165 mm/yr | 163 mm/yr | −1.1 % |
| `str` (mean storage) | 394 mm | 386 mm | −2.0 % |

Snowpack collapses, and deficit and soil drying rise. Runoff rises slightly because rain that
once fell as snow runs off in winter instead of infiltrating as spring melt. Total discharge
(`run` + `rch`) rises 2.5 %.

## Know before using

- BCM is a comparison series only. It is scored beside SAC-SMA and VIC against CalSim3 by
  `sacsma calsim --sacsma-vic-bcm`. No model here is trained on it, selected with it, or
  corrected toward it.
- Caution: Goose Lake is a hole. BCM masks open water, so five grid cells (41.84–42.03 N,
  −120.41 to −120.47 W) are NaN in every month, ringed by partially masked neighbours. 3673 of
  4410 cells are fully valid; the rest clip a lake or reservoir. Weight by `n_bcm_cells`, or
  renormalise over valid cells, when averaging a footprint. A plain `nansum` of weighted cells
  under-counts any watershed that touches the mask (SHA 0.95 % of weight, BND 0.71 %). The
  five cells sit inside the endorheic Goose Lake block that the anchors already exclude.
- Caution: two catchment polygons are smaller than one 270-m cell (`EMD007` 0.015 mi²,
  `EBP030` 0.038 mi²) and contain no cell centre. Each is assigned its nearest cell, visible as
  `n_bcm_cells = 1` in `bcm_catchments.csv`.
- Scenario 1 is the same climate sequence as the `wgen_product_a` forcing. Scenario 13 pairs
  with WGEN scenario 13 (see the [forcing stores](../../inputs/forcing/README.md)).
- BCM is a water-balance model with no channel routing. `run` + `rch` is water generated in a
  month, not water arriving at an outlet, so read its month-to-month timing loosely.
- The catchment tables are on layer `CalSim3_And_GooseLake` (386 polygons), not on
  `CalSim3_Merged`. The two layers do not join on the node name; `sacsma.calsim.sacsma_vic_bcm`
  assigns each polygon by its representative point.

## Read by

`sacsma.calsim.sacsma_vic_bcm` (`sacsma calsim --sacsma-vic-bcm`) reads
`bcm_s01_catchments_monthly.csv` (`run` + `rch`; path from `sacsma.paths.bcm`). No module of
the package reads the two `.nc` files, the `s13` tables or `bcm_catchments.csv`.
