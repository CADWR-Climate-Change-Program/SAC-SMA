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
| `CalBasins_v8_DWR_FNF_PRISM19.xlsx` | The USGS routing workbook: sheet `Cal_outfile` (BCM v8 forced with PRISM, 24 basins, WY2000–2010) and one calibration sheet per routed basin. Local only, not tracked | Delivered with the BCM release |
| `bcm_routing_params.csv` | One row per calibration sheet (11, `source` `workbook`) and per refit (1, `source` `refit`): its basin, the CDEC record its measured flow is (`measured_station`), the site it is applied at (`cdec_site`), the fitted area, the seven parameters, the antecedent storage, the two lags, the windows, and the r2, NSE and PBIAS of the fit. 2 KB | `bcm_routing_params.py` |
| `bcm_cal_outfile.csv` | The BCM input of every basin of the workbook's `Cal_outfile`: `basin_no, month, rch_mm, run_mm, area_m2`, 24 basins × 132 months (WY2000–2010, BCM v8 forced with PRISM). The refit reads it. 99 KB | `bcm_routing_params.py` |
| `bcm_routing_check.csv` | Each sheet's 132 months: `rch_mm`, `run_mm`, `area_m2`, the measured flow and the sheet's cached `P` and `Q` (m3). The port of the sheets is checked against it. 118 KB | `bcm_routing_params.py` |
| `bcm_routing_params.py` | Extracts the three tables from the workbook, makes the refit and checks the port | |

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

## Monthly routing

BCM has no channel routing: `run` and `rch` are water generated in a month. The USGS turns them
into a monthly flow at a gauge with three reservoirs fitted per basin, delivered as the workbook
`CalBasins_v8_DWR_FNF_PRISM19.xlsx` (kept here, not tracked). Its sheet `Cal_outfile` holds BCM
v8 forced with PRISM, WY2000–2010, for 24 basins numbered 2–24 and 26 (26 is the whole region).
Eleven basins have a calibration sheet, each fitted to CDEC full natural flow over the same
eleven years. `sacsma.benchmark.bcm_routing` is an exact port of the sheets' formulas. The
parameters are used as delivered, except at the Kings, whose sheet is mis-wired and is replaced
by a refit (below).

Each month, in m3, with H and I the recharge and runoff times the area and A the antecedent
storage:

| Term | Formula |
|---|---|
| Surface store J | J[t−1] + I[t] − K[t], or 0 when J[t−1] + I[t] ≤ 0 |
| Surface release K | SurfaceScale · J[t−1]^SurfaceExp, 0 when J[t−1] ≤ 0 |
| Shallow store L | L[t−1] + H[t] − M[t] − O[t] |
| Shallow release M | ShallowScale · L[t−1]^ShallowExp, 0 when L[t−1] < 0 |
| Deep store N | A + min(L[t], 0) |
| Deep flow O | DeepScale · N[t−1]^DeepExp |
| Flow Q | AquiferRch · (K[t + peak lag] + M[t + recession lag] + O[t]) |

The first month reads A as the previous surface and shallow store and has no deep flow. Every
scale parameter is 1; the exponents and `AquiferRch` carry the fit.

| Sheet | Fitted to | Site | Area (mi²) | Site area (mi²) | SurfaceExp | ShallowExp | DeepExp | AquiferRch | Antecedent (10⁶ m3) | Lags (peak, recession) | r2 | NSE | PBIAS (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `6-pit` | SIS | SHA | 6,749 | 6,665 | 0.99 | 0.85 | 0.9 | 1.1 | 2,000 | 1, 2 | 0.748 | 0.746 | +0.49 |
| `8-Feather` | FTO | ORO | 3,496 | 3,607 | 0.99 | 0.95 | 0.91 | 1 | 400 | 1, 2 | 0.841 | 0.807 | −0.27 |
| `9-Yuba` | YRS | YRS | 1,226 | 1,108 | 0.999 | 0.97 | 0.89 | 0.84 | 150 | 1, 1 | 0.866 | 0.866 | −0.33 |
| `11-American` | AMF | FOL | 1,824 | 1,885 | 0.999 | 0.98 | 0.6 | 0.94 | 50 | 1, 1 | 0.856 | 0.857 | +0.56 |
| `13-Cosumnes` | CSN | CSN | 681 | 539 | 0.98 | 0.97 | 0.95 | 0.6 | 3 | 1, 1 | 0.826 | 0.824 | −0.61 |
| `14-Mokelumne` | MKM | MKM | 526 | 544 | 0.99 | 0.97 | 0.9 | 1.07 | 4 | 1, 1 | 0.826 | 0.809 | +0.86 |
| `16-Stanislaus` | SNS | NML | 953 | 900 | 0.999 | 0.999 | 0.88 | 0.955 | 150 | 1, 1 | 0.870 | 0.870 | +0.45 |
| `17-Tuolumne` | TLG | TLG | 1,562 | 1,538 | 0.999 | 0.99 | 0.5 | 1.06 | 10 | 1, 1 | 0.866 | 0.867 | −0.06 |
| `18-Merced` | MRC | MRC | 997 | 1,061 | 0.99 | 0.99 | 0.9 | 1.04 | 15 | 1, 1 | 0.859 | 0.860 | −0.36 |
| `21-SanJoaquin` | SBF | MIL | 1,613 | 1,675 | 0.9999 | 0.99 | 0.91 | 1.17 | 40 | 1, 1 | 0.891 | 0.889 | −1.13 |
| `23-Kings` | KGF | PNF | 816 | 1,545 | 0.9999 | 0.96 | 0.9 | 4.65 | 5 | 0, 1 | 0.622 | 0.625 | −0.89 |
| `22-Kings (refit)` | KGF | PNF | 1,523 | 1,545 | 0.9795 | 0.9999 | 0.517 | 1.18 | 78.7 | 1, 1 | 0.859 | 0.859 | +1.00 |

The refit. Sheet 23 routes basin 23 against the Kings flow (below). Basin 22 is the Kings at
Pine Flat. `sacsma.benchmark.bcm_routing.fit` fits the sheet's equations to basin 22's
`Cal_outfile` input against sheet 23's measured column, over the sheet's window (1999-10 to
2010-09), the way the sheets were fitted: it maximises the sheet's NSE with |PBIAS| held within
1 % (the sheets reach 1.13 %), the scales at 1, and the boundary rules of the workbook. The
bounds sit around the range of the ten well-wired sheets (SurfaceExp 0.97–0.99999, ShallowExp
0.80–0.9999, DeepExp 0.5–0.95, AquiferRch 0.5–1.5, antecedent storage 10⁶ to 10^9.5 m3), and
only the causal lags are tried (peak 0 or 1, recession 1). Differential evolution with a fixed
seed; a rerun gives the same row. ShallowExp ends on its upper bound and DeepExp near its lower
one: the shallow store empties within the month and the deep flow is about zero, as on the
American and Tuolumne sheets. Raising the ShallowExp bound to 0.99999 changes the NSE by 0.0002.
The fit's NSE (0.859) is within the range of the ten well-wired sheets (0.746–0.889). The
`params_for` prefers a refit over a sheet. The benchmark does not cover the Kings.

How the three tables are built (the workbook and `openpyxl`, in the `sacsma` environment):

```bash
python data/reference/bcm/bcm_routing_params.py           # extract, refit, write, check (needs the workbook)
python data/reference/bcm/bcm_routing_params.py --refit   # refit only, from the tracked tables (about 15 s)
python data/reference/bcm/bcm_routing_params.py --check   # check only, from the tracked tables
```

The script refuses to write unless:

- every formula of columns F to T, rows 14 to 145, of every sheet matches one template in
  relative form;
- each sheet reads the `Cal_outfile` block of its own basin number, 132 months from 1999-10,
  with one area;
- no formula reads a `Cal_outfile` column other than `rch_mm`, `run_mm` and `Basin_area_m^2`
  (`rchrunscaler` enters nothing).

Checks:

- The port reproduces the cached flow of all 11 sheets, 132 months each, within 4.3e-14
  relative, and each sheet's r2, NSE and PBIAS within 4e-15. `sacsma verify bcm` runs this
  check.
- Each sheet's measured column is a CDEC full-natural-flow record: monthly r 0.995 to 1.000
  against the record in "Fitted to".
- The mm columns of `Cal_outfile` are its acre-foot columns rounded to 0.1 mm. The rounding
  changes no sheet's input volume by more than 0.03 %.
- A rerun writes byte-identical tables, and `--refit` from the tracked tables gives the same
  parameter table as the full extraction.

Know before using the routing:

- Caution: the Kings sheet is mis-wired. It routes basin 23, 816 mi², against Kings at Pine
  Flat flow (r 1.0000 with KGF), and `AquiferRch` 4.65 makes up the volume: basin 23 yields 0.22
  of it. Basin 23 is warm and dry (12.9 °C, 688 mm/yr, about 900 m mean elevation), between the
  Kaweah and the Tule. Basin 22 (1,523 mi², 7.0 °C) is the Kings and has no sheet. Applied to
  Scenario 1 at PNF, the sheet gives 2.4 times the observed volume, a month late. The refit
  (above) scores KGE 0.82 and +14 % there.
- `6-pit` is the whole Shasta Lake inflow (6,749 mi²; r 0.9999 with SIS), not the Pit alone.
- `16-Stanislaus` was fitted to the Stanislaus at Goodwin (SNS). New Melones (NML) full natural
  flow is about 12 % lower over WY2000–2010.
- Sheets 6 and 8 have a recession lag of 2, so their shallow release reads the next month's
  recharge: a one-month lead. Sheet 23's peak lag of 0 delays its surface release by a month.
- The workbook's last month (two for sheets 6 and 8) lacks its surface and shallow terms,
  because `OFFSET` reads blank rows as 0. The port computes every release the final state
  determines and leaves NaN only where an input past the end is needed: the last month of
  sheets 6 and 8 (2018-09 on Scenario 1).
- The parameters act on volumes. Route the depth times the workbook area, the volume the fit
  used; `AquiferRch` already maps it to the gauge. The result scales with the area: a 5 %
  larger area gives 4.4 to 5.0 % more flow and changes the depth by under 0.6 %. With the
  workbook area the Cosumnes runs 23 % high on Scenario 1 (basin 13 is 681 mi² against CSN's
  539).
- The deep store never fills. While the shallow store is not negative, the deep flow is a
  constant `A^DeepExp` drawn from it, the same volume whatever the area. On Scenario 1,
  WY1991–2018, it is 39 % of the Pit sheet's flow, 14 % of Feather's, 11 % of Stanislaus's and
  0 to 7 % elsewhere. Over WY1916–2018 the shallow store is overdrawn in up to 537 of 1,236
  months (Stanislaus), and the deep flow then declines.
- `AquiferRch` above 1 adds water (six sheets). The `impairment` term (1 − AquiferRch)·P is then
  negative.
- The sheets' NSE divides by the sample variance. Their PBIAS is in percent, positive when the
  model is low. Sheet 6 scores from 1999-11, the others from 1999-10.
- The parameters were fitted on PRISM-forced BCM. Scenario 1 yields a different volume on the
  same basins: over WY2000–2010 its `run + rch` is 19 % higher at the Kings (basin 22), 16 % at
  the San Joaquin, 14 % at the Tuolumne and 8 % at the Pit sheet's basin, and 3 % lower at the
  Feather and the American. `AquiferRch` carries the fit's volume scale over unchanged, so the
  routed Scenario 1 runs high where Scenario 1 is wetter. The benchmark therefore fits its own
  routing on Scenario 1.
- 10 of the 12 sites of the benchmark have a sheet; BND and CLE have none. The benchmark's
  own fit covers all 12.
- The `Cal_outfile` basin numbers follow the DWR unimpaired-flow list from 2 to 16 (UF 2 Putah
  to UF 16 Stanislaus). From 17 they run one lower, the valley floor (UF 17) having no basin:
  17 Tuolumne, 18 Merced, 19 Chowchilla, 20 Fresno, 21 San Joaquin.

## Know before using

- BCM is a comparison series only. It is one of the three models of `sacsma benchmark`, routed
  by the equations above with parameters the benchmark fits itself on Scenario 1, at 12 sites
  ([benchmark](../../../artifacts/results/benchmark/README.md)); the sheets as delivered are
  kept beside it for comparison. No model here is trained on it, selected with it, or
  corrected toward it. Every fit here fits only the routing, to CDEC flow.
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
  month, not water arriving at an outlet. Use the routing above for a flow at a gauge.
- The catchment tables are on layer `CalSim3_And_GooseLake` (386 polygons), not on
  `CalSim3_Merged`, its dissolve. The two layers do not join on the node name: the merged layer
  renames each dissolved catchment for its INFLOW arc. Assign a polygon by its representative
  point; the largest overlap goes through boundary slivers.

## Read by

`sacsma.benchmark.gridded` (`sacsma benchmark`) reads `run` and `rch` of `bcm_s01_monthly.nc`
on the footprints of the CDEC sites, fits the routing on them (`bcm_fit`) and routes them by
the sheets for comparison (`bcm_workbook`); `sacsma.benchmark.bcm_routing` reads
`bcm_routing_params.csv` and `bcm_routing_check.csv` (`sacsma verify bcm`). Only
`bcm_routing_params.py` reads `bcm_cal_outfile.csv`. Paths come from
`sacsma.paths.bcm`. No module of the package reads the catchment tables, the `s13` store or
`bcm_catchments.csv`.
