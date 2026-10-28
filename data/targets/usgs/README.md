# data/targets/usgs: cleaned USGS daily gauge flows inside the CalSim3 domain

Daily discharge at the 69 USGS gauges whose watersheds lie inside the CalSim3 domain. Role:
target. The `usgs_daily` family of the learned-parameter models (dPL-CalSim) is
fitted to them: all 69 in its basis, 60 in its fine-tune (nine whole records masked;
`usgs_11218500` trains on `usgs_11218501_mm.csv`). For the calibrated SAC-SMA they are an
observational set independent of the full-natural-flow and CDEC series it was fitted to. The
flows come from the training store of the sibling neuralhyd-ca repository, whose QA/QC pipeline
retrieves NWIS parameter `00060` and screens it; nothing is re-cleaned here.

## Files

| File | What | Source |
|------|------|--------|
| `flow_daily.nc` | `flow_cfs` and `flow_mm`, dims `(gauge, time)` = 69 x 25202, float32 zlib, daily 1950-01-01 to 2018-12-31. 3.2 MB (LFS) | `usgs_flows.py` from the neuralhyd-ca store |
| `gauges.csv` | One row per gauge: `gid`, `tier`, `tier_label`, `area_km2_delineated`, `frac_in_calsim`, `n_obs`, `coverage_frac`, `first_obs`, `last_obs`, `synthetic`, and from NWIS `station_name`, `lat`, `lon`, `area_km2_usgs`, with `area_ratio_delin_over_usgs`. 12 KB | `usgs_flows.py`; NWIS site service |
| `usgs_watersheds.gpkg` | The 69 delineations, layer `usgs_watersheds`, EPSG:4326 to match `data/inputs/calsim3/calsim3.gpkg`. 3.2 MB | `usgs_flows.py` from the neuralhyd-ca delineations |
| `usgs_11218501_mm.csv` | `date, value_mm`: USGS 11218501 (the Kings River below North Fork plus the Kings River powerhouse near Trimmer), the record dPL-CalSim's fine-tune trains `usgs_11218500` on (the basis trained the river-only record), in depth over that gauge's area. 1962-01-01 to 1993-09-30, 11,596 days. 0.24 MB | `usgs_11218501.py` from NWIS |
| `location_mask.csv` | `entity_id, start, end, reason`: the days of the USGS records that dPL-CalSim's fine-tune does not train on (the basis trains them) ([below](#the-location-mask)). 949 ranges | Kept by hand, never generated |

## How it is built

`usgs_flows.py` runs in the `neuralhyd` conda environment (it needs `zarr`, which `sacsma`
lacks). Everything downstream only reads the files.

    conda run -n neuralhyd python data/targets/usgs/usgs_flows.py
    conda run -n neuralhyd python data/targets/usgs/usgs_flows.py --verify

Local input: the folder named by `neuralhyd_dir` in `data/local_paths.toml` (the
`data/training` folder of the neuralhyd-ca repository; see `data/local_paths.example.toml`),
which holds `flow.zarr` (224 basins x 25202 days, cfs) and `watersheds/` (delineations and
area table). Station names, coordinates and USGS drainage areas are fetched from the NWIS
site service; `--no-nwis` skips that and leaves those columns blank.

Selection: a gauge is kept when at least 90 % (`--min-frac`) of its delineated watershed
area lies inside the union of layer `CalSim3_And_GooseLake` of
`data/inputs/calsim3/calsim3.gpkg`, measured in EPSG:3310. The threshold decides the set: 42
gauges are fully inside, 69 clear 90 % and 95 touch the union. Nothing falls between 50 % and 90 %, so
those two thresholds pick the same set. The source's 14 synthetic `99xxxxxxx` ids are SAC-SMA / CDEC
watershed footprints injected into that training set; they are excluded
(`--include-synthetic` keeps them), since those basins are already in this repository under
their own names.

`usgs_11218501.py` fetches the Kings River record from NWIS daily values in the `sacsma`
environment (network access, no local input) and converts it with the factor `flow_daily.nc`
holds for `usgs_11218500`. `--verify` fetches again and compares with the tracked file:

    python data/targets/usgs/usgs_11218501.py
    python data/targets/usgs/usgs_11218501.py --verify

The river-only gauge 11218500 misses what the powerhouse carries past it: the annual
river-only volume is 0.72 to 0.88 of the combined one. The combined record is below the river
on 2 of 11,537 common days. A pull on 2026-10-19 matched the file exactly.

## The location mask

An audit of the non-CDEC training records in October 2026 read each USGS gauge, CalSim3 arc and
unimpaired-flow subbasin against upstream storage, diversions and its own record, annually and
for July–September. `location_mask.csv` in this folder, in `../calsim3/` and in
`../dwr_unimpaired/` holds what dPL-CalSim's fine-tune does not train on as a result, one file per
family.
The files are kept by hand: the audit's working files are not tracked. The reasons:

| Reason | USGS | Arcs | UF | Drops |
|---|---|---|---|---|
| `Jul-Sep (location audit)` | 847 | 2,419 | 283 | July–September of the years the summer is impaired (regulation or diversion upstream), every year where the cause is structural |
| `train once (usgs_<id>)` | | 4,084 | 273 | The months of an arc or subbasin that a USGS gauge at the same place already trains |
| `whole record` | 9 | 2 | 4 | The whole record: impaired, misidentified, or a copy of another target |
| water years (`WY<a>-<b> (location audit)`) | 3 | 1 | | A span of years the record is impaired |
| `Jul-Nov (11218501 regulated)` | 69 | | | July–November at `usgs_11218500`, whose replacement record (`usgs_11218501_mm.csv`) is regulated in those months |
| `October (Box Canyon)` | 18 | | | October at `usgs_11341400` after Box Canyon Dam (December 1968) |
| `WY<a>-<b> forcing (R4, station index)` | 3 | | | Early eras of three gauges where the Livneh forcing drifts from PRISM's and CDEC's station index confirms it (the same test as `../cdec/forcing_era_mask.csv`) |

USGS ranges are days, arc and subbasin ranges months (`YYYY-MM`). Training reads all three files
together with the CDEC daily screen:

    sacsma dpl train ... --target-mask data/targets/cdec/fnf_daily_screen.csv,data/targets/usgs/location_mask.csv,data/targets/calsim3/location_mask.csv,data/targets/dwr_unimpaired/location_mask.csv

The checkpoint keeps the masked entries, so a trained run is scored on the same targets without
these files.

## Checks

`usgs_flows.py --verify` has no predecessor to reproduce, so it gates on internal consistency
and physics:

- `flow_mm` back to cfs through the published area: maximum relative error 1.65e-07 (float32).
- No negative discharge.
- Mean annual runoff depth inside the California band: observed 10 to 1386 mm/yr. This is
  the check that catches a wrong unit or area: a wrong factor throws it orders of magnitude out.
  The driest gauges are Coast Range and Tehachapi creeks (Caliente, Cantua, Avenal), the
  wettest are high-Sierra snow basins (S Yuba near Cisco, Duncan Canyon near French Meadows).
  The snow-regime median (806 mm/yr) is about twice that of rain and mixed (about 425), which
  independently corroborates the inherited `tier` labels.

## Know before using

- caution: `tier` is a hydrologic regime (1 rain, 2 mixed, 3 snow), not a data-quality grade.
  Do not filter on it as if it ranked record quality.
- caution: the source flows are cfs, despite a `flow_mm` name used downstream in neuralhyd-ca.
  Read as cfs, the largest basins give 143 to 965 mm/yr; read as mm/day they give millions.
- `flow_cfs` is canonical (verbatim, reproducible against USGS). `flow_mm` is derived as
  `cfs x 2.4465755 / area_km2` on the delineated polygon area, the basin the record is
  attributed to here, not on the USGS-reported area. Both areas are in `gauges.csv` and agree
  within 5 % (ratio 0.952 to 1.016), so a reader can re-derive the depth on either.
- `area_km2_delineated` is published at 6 decimals on purpose: the smallest basin is 4.9 km2,
  where 3 decimals already cost 1e-4 relative and make the published mm/day irreproducible.
- caution: the records are sparse. Median coverage is 44 % of the 1950 to 2018 window
  (minimum 9 %, maximum 95 %). They are intermittent series, not a continuous panel: compute
  any skill score on each gauge's own observed days.

## Read by

- `flow_daily.nc` (`flow_mm`): `sacsma.dpl.multi_timescale` and
  `sacsma.dpl.evaluate_multi_timescale` (the `usgs_daily` entities), `sacsma.dpl.calsim.atlas`.
- `usgs_watersheds.gpkg`: `sacsma.dpl.calsim.atlas` and
  `data/inputs/domains/multifamily/build_entity_cells.py` (the entities' cell sets).
- `gauges.csv`, `flow_daily.nc`: `data/inputs/domains/multifamily/build_entities.py`.
- `usgs_11218501_mm.csv`: `sacsma dpl train --target-series`, and through the checkpoint
  `sacsma.dpl.multi_timescale` (every scoring of dPL-CalSim reads it again).
- `location_mask.csv`: `sacsma dpl train --target-mask`.
