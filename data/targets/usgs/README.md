# data/targets/usgs: cleaned USGS daily gauge flows inside the CalSim3 domain

Daily discharge at the 69 USGS gauges whose watersheds lie inside the CalSim3 domain. Role:
target. The `usgs_daily` family of the learned-parameter models (dPL-CalSim) is
fitted to them. For the calibrated SAC-SMA they are an observational set independent of the
full-natural-flow and CDEC series it was fitted to. The flows come from the training store
of the sibling neuralhyd-ca repository, whose QA/QC pipeline retrieves NWIS parameter `00060`
and screens it; nothing is re-cleaned here.

## Files

| File | What | Source |
|------|------|--------|
| `flow_daily.nc` | `flow_cfs` and `flow_mm`, dims `(gauge, time)` = 69 x 25202, float32 zlib, daily 1950-01-01 to 2018-12-31. 3.2 MB (LFS) | `usgs_flows.py` from the neuralhyd-ca store |
| `gauges.csv` | One row per gauge: `gid`, `tier`, `tier_label`, `area_km2_delineated`, `frac_in_calsim`, `n_obs`, `coverage_frac`, `first_obs`, `last_obs`, `synthetic`, and from NWIS `station_name`, `lat`, `lon`, `area_km2_usgs`, with `area_ratio_delin_over_usgs`. 12 KB | `usgs_flows.py`; NWIS site service |
| `usgs_watersheds.gpkg` | The 69 delineations, layer `usgs_watersheds`, EPSG:4326 to match `data/inputs/calsim3/calsim3.gpkg`. 3.2 MB | `usgs_flows.py` from the neuralhyd-ca delineations |

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
`data/inputs/calsim3/calsim3.gpkg`, measured in EPSG:3310. The rule matters: 42 gauges are
fully inside, 69 clear 90 %, 95 merely touch. Nothing falls between 50 % and 90 %, so those
two thresholds pick the same set. The source's 14 synthetic `99xxxxxxx` ids are SAC-SMA / CDEC
watershed footprints injected into that training set; they are excluded
(`--include-synthetic` keeps them), since those basins are already in this repository under
their own names.

## Checks

`usgs_flows.py --verify` has no predecessor to reproduce, so it gates on internal consistency
and physics:

- `flow_mm` back to cfs through the published area: maximum relative error 1.65e-07 (float32).
- No negative discharge.
- Mean annual runoff depth inside the California band: observed 10 to 1386 mm/yr. This is
  the real check on units and areas, since a wrong factor throws it orders of magnitude out.
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
