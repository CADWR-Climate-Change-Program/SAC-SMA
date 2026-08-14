# data/dpl_entities — multi-timescale training-entity registry

`entities.csv`: 106 training entities — 18 `uf_monthly` + 69 `usgs_daily` +
17 `cdec_daily` (15 + CLE + CSN) + 2 `obs11_monthly` (SHA, TNL). One row per
(site × timescale × family); every target trains as an independent entity.

Generated file — do not edit. Regenerate (sacsma conda env):
`python dataprep/build_entities.py`. UF pour points are hand-maintained in
`data/dwr_unimpaired/uf_gauges.csv`.

## Columns

| column | meaning |
|---|---|
| `entity_id` | `<family>_<site>`, unique (`uf_06`, `usgs_11258000`, `cdec_SHA`, `obs11_TNL`) |
| `family`, `timescale` | loss family and its native timescale |
| `site_id`, `name` | site identity (read `site_id` with `dtype=str`) |
| `delineation` | `arcs` (CalSim3_Merged union), `usgs_gpkg`, or `cdec15_grid_footprint` (Tulare 4 — no CalSim polygons exist) |
| `arcs` | semicolon list of `I_*` nodes |
| `area_mi2` | the observing site's own drainage area (divides volume → depth) |
| `area_mi2_swat` | Appendix A SWAT model area, decision-5 alternate — from `uf_gauges.csv` (16 UFs) |
| `outlet_lat/lon/source` | pour point (flowlen anchor); `uf_07` null — no gauge exists |
| `train_start/end` | the entity's own window, from its actual record |
| `n_obs` | valid observations in-window, counted from the raw store |
| `obs_store` | `<path>:<column>` of the target series |
| `flags` | caveats, semicolon list — see below |

## Flags

`train_only` + `inherited_cadwr_footprint` (Tulare 4) ·
`outlet_below_delineation` (gauge/dam 5–13 km below the delineation) ·
`obs_routed_through_lakes` (Cache — obs is the outflow below Clear
Lake/Indian Valley) · `obs_includes_valley_floor` + `no_gauge_composite`
(UF 7) · `calsim_ref_wetter_summers` (Bear) ·
`daily_runs_6pct_below_monthly` (CSN) · `fnf_computed_at_trinity_dam` (CLE) ·
`footprint_overlaps_MIL_2.5pct` (PNF) ·
`obs_already_mm`, `obs_ends_2014-01` (obs11).

Note for consumers of `arcs`: `I_RUB002` (UF 11 / FOL lists) has no
`CalSim3_Merged` polygon — its terrain was dissolved into `MFA025`, so
coverage is complete; tolerate the unresolvable token, don't KeyError.


