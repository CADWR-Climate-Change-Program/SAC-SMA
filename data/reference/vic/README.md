# data/reference/vic: VIC routed monthly flow and routing grids

Monthly routed flow of the VIC model at the CalSim3 inflow nodes, under three climates, and the
routing grid tables of six nodes. Role: reference. VIC is the benchmark model of the CalSim3
comparison; no model in this repository is fitted to it. The series come from the VIC runs of
the CalSim3 stochastic-input-generation pipeline.

## Files

| File | What | Source |
|---|---|---|
| `vic_routed_monthly.csv` | `[date, vic_name, flow_taf]`, TAF/month, month-end dates, 1915–2018. VIC routed historical flow from the `Historical_Unsplit` run (10 MB) | pipeline output `vic/output/routed/Historical_Unsplit` |
| `vic_routed_monthly_historical_lto.csv` | The same routing under the split-precipitation `Historical` run, 1915–2021 (10–11 MB) | `vic/output/routed/Historical` |
| `vic_routed_monthly_wgen_product_a.csv` | The same routing under the detrended-temperature `Product_A` validation run, 1915–2018 (10–11 MB) | `vic/output/routed/Product_A/1` |
| `vic_gridinfo_I_SHSTA.csv`, `vic_gridinfo_I_SHSTA_no_gooselake.csv` | The routing's GridInfo (station to cell weight) table for Shasta, with and without the Goose Lake cells: `[id, lat, lon, cell_km2, basin_km2]` | pipeline file `mod_forcing/vic/reference/GridInfo/CS3_<node>_GridInfo.txt`, headers added |
| `vic_gridinfo_8RI_N_MEL.csv`, `vic_gridinfo_I_ESTMN.csv`, `vic_gridinfo_I_TRNTY.csv`, `vic_gridinfo_I_HNSLY.csv` | GridInfo of Stanislaus (102 cells), Chowchilla River (30), Trinity (71) and Fresno River (30); single grids used as they are | the same |

A GridInfo table has one row per 1/16° VIC cell fragment. `cell_km2` is the cell area and
`basin_km2` the cell area inside the node's basin. Two `I_SHSTA` boundary cells appear as two
fragments.

## How it is built

Delivered; cannot be rebuilt from a clone. The two alternate-climate tables were ingested in
2026-07 with the recipe of the baseline table, including the `_no_gooselake` substitution
below. The GridInfo tables are verbatim copies with headers added.

## Checks

- The ingest of the alternate-climate tables reproduces `vic_routed_monthly.csv` exactly from
  `Historical_Unsplit`.
- GridInfo round trip (2026-07): the rows Shasta keeps are identical between its two files, and
  all cells are on the 1/16° grid.

## Know before using

- In all three routed tables the keys `I_SHSTA` and `8RI_SRBB` hold the `_no_gooselake` variant
  of the VIC routing. Goose Lake is endorheic and gives no real downstream inflow.
- `vic_gridinfo_I_SHSTA_no_gooselake.csv` is the spatial basis of that variant. It drops the
  94 cells (about 1000 mi²) of the Goose Lake over-reach (640 to 546 cells). It was built in
  the pipeline by `build_no_gooselake_gridinfo.py`.
- Forcing basis: `vic_routed_monthly.csv` shares the unsplit-precipitation Livneh basis of the
  SAC-SMA historical forcing, so the SAC-SMA and VIC comparison is on the same forcing. Each
  alternate table pairs with the forcing product of the same name (see the
  [forcing stores](../../inputs/forcing/README.md)).
- `historical_lto` runs three years longer (to 2021) than the other two tables.
- Flow is a monthly volume (TAF/month), not a depth.

## Read by

- `sacsma.calsim.load_vic_monthly` (path from `sacsma.paths.vic_routed`): `sacsma calsim`
  (`sacsma.calsim.compare`, the baseline table), `sacsma calsim --forcing-compare`
  (`sacsma.calsim.forcing_compare`, the baseline against each alternate table) and
  `sacsma benchmark` (`sacsma.benchmark.flows`, the `wgen_product_a` table: the VIC-CalSim3
  column of the CDEC benchmark, at the nodes of the 15-CDEC track of `sacsma calsim`, except
  YRS, which sums the 16 arc series above Smartsville (`8RI_SMART` includes Deer Creek below
  the gauge), plus `I_TRNTY` for CLE and the four Cosumnes arcs for CSN).
- `sacsma.calsim.load_vic_gridinfo` (path from `sacsma.paths.vic_gridinfo`): the footprint maps
  of `sacsma.calsim.compare` (`make_shasta_footprint_maps`, `make_basin_footprint_maps`).
