# data/inputs/forcing: daily forcing stores

The daily precipitation and temperature every run reads. Role: input. Five stores sit on the 4,410-cell
1/16° region grid (see [the grid README](../grid/README.md)), one NetCDF file per product, and one dense
store holds the off-grid HRU points of the `15cdec` domain. The products were made outside this repository
(the Livneh gridded meteorology, DWR's weather-generator releases, the CalSim3 LTO study climate, NOAA
AORC); the scripts here cut them to the grid.

## Files

| File | What | Source |
|---|---|---|
| `historical_livneh_unsplit.nc` (1.06 GB, LFS) | The default product. 4,410 cells, daily 1915-01-01 to 2018-12-31. Livneh 1/16° meteorology on the unsplit-precipitation basis: precipitation carries the Pierce et al. (2021) storm-splitting correction, so storm totals are not artificially split across days; temperature is Livneh, PRISM-adjusted and bias-corrected. The x10 precipitation artifacts are divided by 10 at the 289 pairs of `prcp_x10_artifacts.csv`. | The WGEN NonDetrend-Unsplit statewide ASCII store, through a local forcing master. `wgen_forcing.py`, `build_region_forcing.py` |
| `wgen_product_a.nc` (1.03 GB, LFS) | WGEN Product A scenario 1: the historical-parallel sequence of the CalSim3 stochastic-input pipeline (it forced the pipeline's VIC Product A validation run). 4,410 cells, 1915 to 2018. Precipitation is identical to the Livneh-unsplit product (same basis, released rounded to 0.01 mm, x10 artifacts already corrected upstream). Temperature is detrended to a 1991-2020 baseline, which warms the early record: over the grid the daily mean is about 1.0 °C higher in 1915 to 1919 (minimum +1.8 °C, maximum +0.2 °C), tapering linearly to about 0 by the 2010s. Over the Observed11 watersheds the shift is about 1.3 °C (minimum +2.1 °C, maximum +0.4 °C). | DWR gridded weather-generator release ("Gridded Weather Generator Perturbations…", data.ca.gov), files `WGEN/Product_A/1/meteo_<lat>_<lon>`, verbatim. `build_region_forcing.py` |
| `wgen_product_a_s12.nc` (16.3 MB, LFS) | WGEN Product A climate scenario 12 (+2 °C, extreme-tail rate 7 %/°C, wet-day mean unchanged), 4,410 cells, 1915 to 2018. Not a gridded store: an exact table codec (`wgen_table_v1`) over `wgen_product_a.nc`, decoded at load by `sacsma/wgen_scenarios.py`. | DWR's Product A 100-year scenario release, files `12/meteo_<key>` (2-decimal ASCII). `wgen_product_a_scenarios.py` |
| `historical_lto.nc` (1.00 GB, LFS) | Historical LTO: the observed-climate VIC forcing of the CalSim3 LTO (Long-Term Operations) study, on the pre-Pierce-2021 "split" Livneh precipitation lineage. 4,058 cells, daily 1915-01-01 to 2021-12-31, three years past the other Livneh stores. | Files `Historical_Climate_LTO/1_Historical/data_<lat>_<lon>` of the same release (`prcp tmax tmin wind`, no date columns; wind is dropped). `build_region_forcing.py` |
| `aorc.nc` (1.79 GB, LFS) | NOAA Analysis of Record for Calibration (AORC) v1.1: 1-km hourly, box-averaged to the grid on a UTC day. 4,410 cells, 1979 to 2025, nine variables: `prcp`, `tmin`, `tmax`, `dlwrf`, `dswrf`, `pres`, `spfh`, `ugrd`, `vgrd`. An independent product, not a Livneh lineage. | Public Zarr store `s3://noaa-nws-aorc-v1-1-1km/{year}.zarr`. `aorc_region.py` |
| `historical_livneh_unsplit_15cdec_hru.nc` (774 MB, LFS) | Livneh-unsplit daily forcing at the 6,033 HRU points of `15cdec`, 1915 to 2018, variables `prcp` and `tavg` only. A dense off-grid product with its own upstream interpolation. The x10 artifacts are corrected at every pair it covers. | Delivered with the study materials of Wi & Steinschneider |
| `prcp_x10_artifacts.csv` | `[key, date, prcp_raw_mm]`: the misplaced-decimal precipitation spikes of the raw Livneh lineage. 289 (cell, day) pairs over 244 cells on 15 isolated summer days (1916 to 1992), each exactly 10 times too large. `prcp_raw_mm` is the raw value; rows are sorted by (key, date). | 197 pairs from `wgen_forcing.py --scan-x10`, 92 from a comparison with `wgen_product_a.nc`. Frozen |
| `wgen_product_a_scenarios.csv` | The WGEN Product A scenario key, 30 rows: `scenario`, `dT_C` (0 to 5 °C), `cc_pct_per_C` (extreme-tail rate: 0, 7 or 14 %/°C), `dmean_pct` (wet-day mean change, -25 to +25 %), `F_extreme` = (1 + cc)^dT. Scenario 1 is `wgen_product_a.nc`; 12 is +2 °C; 13 is +3 °C (the pairing of the BCM `s13` run in `data/reference/bcm/`); 26 is +3 °C with precipitation identical to scenario 1. | The release's `CC.thermodynamic.change_list.xlsx`, parsed by `wgen_product_a_scenarios.py --key` |

A grid store has dimensions `(key, time)`, where `key` is the 5-decimal `<lat>_<lon>` cell key, and three
float32 variables, chunked per cell and zlib-compressed: `prcp` (mm/day), `tmin` and `tmax` (°C). `tavg` is
not stored: it is derived at load as `(tmax + tmin) / 2`. Forcing is per grid cell, not per watershed.

## How it is built

Environment `sacsma`, from the repository root. Inputs from outside the repository, by key of
`data/local_paths.toml`:

- `wgen_ascii`: the WGEN NonDetrend-Unsplit statewide ASCII store, one `data_<lat>_<lon>` file per cell
  (`year month day prcp tmax tmin`, daily 1915-01-01 to 2018-12-31, 13,786 cells).
- `staging`: holds the forcing master `forcing/livneh_unsplit_nondetrend_daily_region.nc` (about 1 GB, the
  raw lineage of the region cells, not in the repository).
- `climate_release`: the `BASE` folder of the CalSim3 stochastic-input release.
- `wgen_scenarios`: the Product A 100-year scenario release (folders 1 to 30, read-only; the key workbook
  sits one folder up).

```bash
python data/inputs/forcing/wgen_forcing.py --build-master          # ASCII -> the local master
python data/inputs/forcing/build_region_forcing.py --product all   # or one of the three product names
python data/inputs/forcing/wgen_forcing.py --verify                # Livneh store = master + x10 table
python data/inputs/forcing/wgen_product_a_scenarios.py --key       # -> wgen_product_a_scenarios.csv
python data/inputs/forcing/wgen_product_a_scenarios.py --pull 12 --build 12 --verify 12
python data/inputs/forcing/aorc_region.py --all                    # 8 source variables, 1979-2025
python data/inputs/forcing/aorc_region.py --status                 # progress, read from the partials
python data/inputs/forcing/aorc_region.py --assemble               # partials -> aorc.nc
```

- The master stays bit-faithful to the raw ASCII; the Livneh build applies the x10 table. The tracked
  Livneh store was built with 197 corrections and patched in place for the 92 pairs added on 2026-09-29
  (the same float32 division at those cell-days; attribute `corrections` says 289). A full rebuild applies
  the whole table and gives the same values. The store before the patch stays in LFS history (oid
  `5c71d6c7…`).
- `prcp_x10_artifacts.csv` is frozen. `--scan-x10` derived 197 of its pairs from per-domain stores that
  exist in git history only, and cannot derive the other 92: never let it overwrite the table.
- `historical_livneh_unsplit_15cdec_hru.nc`: delivered; cannot be rebuilt from a clone.
- Scenarios 2 to 30 build like 12: about 40 minutes of pull each, minutes to build and verify. Pull
  checkpoints go to `tmp/wgen_product_a_scen/pull`, local only.
- AORC needs only the network. No raw data is written: each chunk is decompressed in memory, box-averaged
  onto the cells and discarded, at a local cost of about 2.4 GB of partials (`tmp/aorc_parts`) plus the
  store. On the wire, all eight variables are about 1.1 TB and `APCP` + `TMP` alone about 175 GB (measured
  on summer chunks, the conservative end); the full pull took 10.8 hours at about 26 MB/s. Run in
  `us-east-1`, the same job transfers about 3 GB. Work is banked per time chunk and per variable-year, each
  written atomically, so a rerun resumes. `--assemble` stitches the variables banked so far over the years
  they share, so `prcp`, `tmin` and `tmax` can land before the rest.

## Checks

- Parity. With the grid stores in place, `run_basin` against the MATLAB simulations
  (`data/reference/matlab/`) gives KGE 0.99999658 (BND), 0.99998883 (CacheCreek), 0.99995206 (SHA),
  0.99990781 (SHAST), all with max |Δ| ≤ 0.063 mm/day. Rerun with `sacsma verify parity`.
- Livneh store. The WGEN ASCII store is the verified source of the lineage (2026-07-16: `prcp` to float32
  rounding, `tavg` exactly `(tmax + tmin) / 2`). `wgen_forcing.py --verify` requires the store to equal the
  master with exactly the x10 table applied, `tmin` and `tmax` included.
- The x10 table. 197 pairs (168 cells, 7 days) came from the per-domain CalLite stores, which held the
  corrected values. The other 92 (76 cells, 11 days, 8 of them new) are the only cell-days where the grid
  store differed from `wgen_product_a.nc` beyond rounding: each raw value over 10 is within that store's
  2-decimal rounding (ratio 9.997 to 10.003) and the raw value equals the WGEN ASCII. With them corrected
  the two stores agree to 0.0050 mm everywhere, so the table is complete against scenario 1. The ASCII
  value over 10 reproduces the stored value at all 289 pairs.
- The grid stores replaced per-domain stores, now in git history. The Livneh store matched them to their
  3-decimal write precision, artifact days included, and differed from the one raw store at exactly 175
  table pairs; `wgen_product_a.nc` and `historical_lto.nc` matched bit for bit. The build still runs this
  comparison and skips each store that is absent.
- Scenario 12. Pull gates on all 4,410 cells: scenario 1 of the release equals `wgen_product_a.nc`;
  scenario temperature minus scenario 1 equals dT on every day; scenario precipitation is positive only
  where scenario 1 is, apart from counted added wet days. `--verify`: every cell decodes bit-exact to the
  release. 40 random cells re-read from the release through `io.load_forcing` are bit-exact. The parsed
  scenario key matches the key decoded independently from the data.
- AORC against the Livneh store, 2015 (independent products, so agreement is the gate). Mean, bias, RMSE,
  r per cell-day, r of the regional series: `prcp` 1.305 against 1.290 mm/d, +0.015, 3.396, 0.8234, 0.9565;
  `tmin` 6.950 against 7.342 °C, -0.392, 1.536, 0.9795, 0.9963; `tmax` 21.557 against 21.201 °C, +0.356,
  1.621, 0.9865, 0.9957. Per-cell annual precipitation totals correlate at 0.9755 (bias +1.13 %). Monthly
  `tmax` differences stay within +0.02 to +0.64 °C. Wet-day fraction (> 1 mm) is 0.131 against 0.143: AORC
  puts precipitation into fewer, more intense days, with a slightly wider diurnal range.
- AORC day and bounds. Over all 24 whole-hour offsets for 2015, agreement with the Livneh store falls off
  monotonically away from UTC (regional daily r 0.9565 at offset 0, 0.7624 at -8, 0.5616 at -16). Physical
  bounds: zero violations in all nine variables (`prcp` mean 1.9582 mm/day, range 0.0 to 393.9; `pres`
  61.9 to 103.7 kPa; `dlwrf` 111 to 450 W/m²; `tmin`/`tmax` -34.4/+47.1 °C). Each grid cell receives 49 to
  64 AORC cells (mean 56.2).

## Know before using

- `15cdec` runs on the default product only. Its HRU points are study-specific centroids off the 1/16°
  grid, and the other releases do not cover them without a nearest-cell mapping that the original study
  never defined. Its store has no `tmin`/`tmax`.
- `historical_lto.nc` lacks 352 region cells (Kern/Tule and Goose Lake), listed in its `missing_cells`
  attribute; a watershed touching them cannot run this product and the load raises. One cell absent from
  the release (`41.46875_-122.15625`, Mt Shasta flank, ≤ 0.12 % of the BND/SHA/SHAST areas) is filled from
  its southern neighbour `41.40625_-122.15625`.
- LTO precipitation is a different realization: daily correlation with the unsplit product is 0.83 to 0.93
  per cell and annual totals are within ±7 %. Storm mass is preserved, but daily values differ well beyond
  the storm-splitting signature. Its temperature is the same product (about +0.02 °C, uniform).
- About 0.05 % of days in the ASCII sources carry an inverted pair (tmin > tmax). The stores hold them
  sorted; `tavg` is unchanged. Scenario 12 has 182,716 such pairs, sorted likewise.
- caution: every corrected x10 value is ≥ 150.19 mm raw, so the upstream correction looks floored at
  150 mm. 419 cell-days of 50 to 150 mm on the same 15 days are 3 to 9.7 times their largest corrected
  neighbour. They are identical in Livneh and scenario 1, so they are in the parity baseline and in every
  WGEN scenario. They are flagged, not corrected (no referee in the repository). `--cut` warns about them.
- Every tracked learned-parameter run was trained on stores with every pair of
  `prcp_x10_artifacts.csv` corrected.
- A scenario file stores only the change against `wgen_product_a.nc`. `tmin` and `tmax` are scenario 1 plus
  `dT_hundredths`, exactly. Precipitation is, per cell and calendar month, a table of the scenario value at
  each distinct scenario-1 wet value, a residual per scenario-1 wet day and exact overrides (none for
  scenario 12). Each cell is checked by CRC32 against scenario 1 and against the release, so the load
  raises if `wgen_product_a.nc` ever changes. Scenario 12 has 1.0025 times the region-wide precipitation
  volume of scenario 1, 0.86 % of scenario-1 wet days dry, and no added wet days.
- caution: run AORC from WY1982 (1981-10-01). The 1979-81 head is artifact-prone and all of December 1979
  is missing at the source; 1979-01-01 to 1981-09-30 is spin-up.
- caution: `aorc.nc` is the only store that contains NaN: 186,551 cell-days (0.2464 %) on 214 days. 32 days
  are whole-domain (December 1979 and 2024-06-18); the rest are runs of 1 to 6 days over 15 to 575 cells,
  mostly before 2010. For `prcp`/`tmin`/`tmax` the share is 0.1996 % on 79 days over the full record, and
  0.0196 % on 44 days from WY1982 (one whole-domain day, 43 partial days of 15 to 429 cells). 1983, 2002,
  2010 to 2023 and 2025 are clean.
- AORC gaps are filled at load by `io.fill_missing_days`: a missing day takes the previous day's value of
  the same cell (a leading gap takes the first observed day), and `tavg` is derived after the fill. The
  file keeps its NaN; the count is in the loaded dataset's `nan_filled_cell_days` attribute. Gap-free
  products are returned untouched.
- AORC source fill (`missing_value` -32767) is dropped before averaging: each hour is the mean of the 1-km
  cells that reported. A day is reduced over its valid hours only, so a partly observed day under-reports
  precipitation rather than being scaled up; a day is NaN only if no hour reported.
- caution: screening on physical bounds cannot certify an AORC store. A first pull that scaled fill as data
  was replaced by a full re-pull; 42 cell-days of `dlwrf`, `pres` and `dswrf` had changed inside physical
  bounds. Any change to the aggregation needs a full re-pull, not a patch. The partials hold temperature
  in °C, and a `TMP_*.npz` holds two arrays (`tmin`, `tmax`).
- The VIC benchmark (`data/reference/vic/vic_routed_monthly.csv`) is routed from the VIC
  `Historical_Unsplit` run, so it shares the default product's precipitation basis.

## Read by

- `sacsma.io.load_forcing`, behind every model run. For a grid domain it selects the domain's cells by the
  keys of its HRU table, derives `tavg` and fills AORC gaps. A product is a file stem:
  `sacsma run SHA --domain 11obs --forcing wgen_product_a_s12`, `--forcing aorc`.
- `sacsma.wgen_scenarios` (scenario decode); `sacsma.dpl.data` (per-cell `tmin`/`tmax`).
- `prcp_x10_artifacts.csv`: `build_region_forcing.py` (at build), `wgen_forcing.py --cut` and `--verify`.
  `wgen_product_a_scenarios.csv`: `wgen_product_a_scenarios.py` (the dT of a build).

## New-basin setup

A watershed inside the region needs only a delineation and a flow target. Cells: select from
`data/inputs/grid/grid_cells.csv`, or intersect the delineation with the 1/16° grid. Forcing: runs in this
repository read the grid stores by cell key. For a cut outside the repository,
`python data/inputs/forcing/wgen_forcing.py --cut <name> --cells <csv> --out-dir <dir>` writes
`historical_livneh_unsplit_<name>.nc` (`prcp`, `tavg`) and `tminmax_livneh_percell_<name>.nc` from the
master, x10 days corrected unless `--no-fix-x10`. Static attributes: rows of `soilveg_continuous.csv` and
`lai_climatology.csv` in `data/inputs/grid/`; every region cell is covered.
