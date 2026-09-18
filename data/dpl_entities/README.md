# data/dpl_entities — multi-timescale training-entity registry

`entities.csv`: 95 training entities — 9 `uf_monthly` + 69 `usgs_daily` +
17 `cdec_daily` (15 + CLE + CSN). One row per (site × timescale × family);
every target trains as an independent entity.

Generated file — do not edit. Regenerate (sacsma conda env):
`python dataprep/build_entities.py`. UF pour points are hand-maintained in
`data/dwr_unimpaired/uf_gauges.csv`.

A de-duplication rule is applied at build time: a watershed does not
train at both daily and monthly timescales unless its daily record is short
(starts ~2000 or later). Eleven monthly twins of long-record CDEC dailies
are built and then dropped — UF 8, 9, 11, 14, 15, 16, 18, 19, 22 and both
`obs11_monthly` rows (SHA ≡ SIS daily, TNL ≡ CLE daily at 96% shared area).
The short-daily pairs stay: uf_13/cdec_CSN (CSN 1999-04) and uf_06/cdec_BND
(SBB 1999-05). Per-drop twin and record start: `DEDUP_DROPS` in the builder;
record completeness verified against `cdec15/gage.csv` and
`cdec_fnf/fnf_daily.csv`.

## Columns

| column | meaning |
|---|---|
| `entity_id` | `<family>_<site>`, unique (`uf_06`, `usgs_11258000`, `cdec_SHA`) |
| `family`, `timescale` | loss family and its native timescale |
| `site_id`, `name` | site identity (read `site_id` with `dtype=str`) |
| `delineation` | `arcs` (CalSim3_Merged union), `usgs_gpkg`, or `sacsma_15cdec_gis` (Tulare 4 — the original SAC-SMA boundary polygons; no CalSim polygons exist) |
| `arcs` | semicolon list of `I_*` nodes |
| `area_mi2` | the observing site's own drainage area (divides volume → depth) |
| `area_mi2_swat` | Appendix A SWAT model area, from `uf_gauges.csv` (7 of the 9 UFs; UF 6/7 have no usable single model) |
| `outlet_lat/lon/source` | pour point; `uf_07` null — no gauge exists |
| `record_start` | true data-availability start of the source record: UF = first published month (WY1922), USGS = `first_obs`, CDEC = earlier of the advertised CDEC start and the committed store's first day |
| `train_start/end` | the training window (for USGS it equals the record window) |
| `n_obs` | valid observations in-window, counted from the raw store |
| `obs_store` | `<path>:<column>` of the target series |
| `flags` | caveats, semicolon list — see below |

## Flags

`train_only` (Tulare 4) · `polygon_2.6pct_above_published_area` (TRM —
`area_mi2` keeps the published 561 as the depth basis) ·
`outlet_below_delineation` (gauge/dam 5–13 km below the delineation) ·
`obs_routed_through_lakes` (Cache — obs is the outflow below Clear
Lake/Indian Valley) · `obs_includes_valley_floor` + `no_gauge_composite`
(UF 7) · `calsim_ref_wetter_summers` (Bear) ·
`daily_runs_6pct_below_monthly` (CSN) · `fnf_computed_at_trinity_dam` (CLE).

Note for consumers of `arcs`: `I_RUB002` (FOL's list) has no
`CalSim3_Merged` polygon — its terrain was dissolved into `MFA025`, so
coverage is complete.

## entity_cells.csv — cell sets and weights

One row per (entity, region grid cell): 5,818 rows, 95 entities, 2,654
distinct cells of `data/region/grid_cells.csv`. Replaces the per-domain
`hruinfo` tables as the aggregation basis for entity training.

Flow lengths live in `flowlens.csv` (below), keyed identically; outlet
coordinates live in the registry only.

Generated file — do not edit. Regenerate (sacsma conda env):
`python dataprep/build_entity_cells.py`.

| column | meaning |
|---|---|
| `entity_id` | registry key |
| `key`, `lat`, `lon` | region grid cell (key 5-decimal-normalized) |
| `overlap_mi2` | cell-square ∩ delineation overlap area — the aggregation weight (normalize per entity; sums are NOT the registry `area_mi2`, which is the observing site's published area) |

Weights come from square-overlap mapping of `CalSim3_Merged` polygons
(`arcs` entities), the delineated USGS watersheds (`usgs_gpkg`), or the
original SAC-SMA boundary polygons (`sacsma_15cdec_gis`, Tulare 4 —
`data/cdec15/gis/SACSMA_15CDEC.geojson`; supersedes the inherited
`cdec15_grid` weights, which the new mapping reproduces at r ≥ 0.99 on
the common cells). Per-entity Σoverlap reproduces each footprint's
reference area to +0.14% worst-case (uf_10; the build asserts <0.2%). The
mapped sums equal the polygons' true geometric areas — the small positive
residual is mostly the gpkg `SQ_MI` attributes running ~0.05–0.12% below
true geometric area, plus a once-per-arc count of arc-overlap slivers
(largest pairwise overlap 1.4% of the smaller arc; ≤0.04% at entity
level).

Cell-basis note: the training basis is 2,654 cells. The de-dup drops left
the union unchanged (every dropped monthly twin's cells stay via its daily
twin; TNL's extra `I_LWSTN` cells via `usgs_11525500`); the Tulare remap
onto the real polygons then added 8 edge cells (the inherited cell sets
were a strict subset of the new). The full-rim basis — every cell touching
any rim polygon + USGS + Tulare — is 2,847 (an earlier tally of 2,853 was
that basis, less a 6-cell bookkeeping difference). Statics coverage is
**complete**: the full-grid ingest on main (`a77e4a8`) extended
`data/region/soilveg_continuous.csv` and `lai_climatology.csv` to all
4,410 region cells, closing what was a 256-cell gap in the training basis
(447 full-rim).

## Where the observation series live

This store holds no observation series — the registry's `obs_store`
column points at each entity's target in its source store: `usgs_daily`
reads `usgs/flow_daily.nc:flow_mm` and the 15 committed `cdec_daily`
basins read `cdec15/gage.csv:flow` (both already depth). The two stores
that are not depth-native carry derived companions beside their raw
tables, built by `dataprep/build_obs_depth.py`:
`dwr_unimpaired/uf_monthly_mm.csv` (TAF → mm/month at the CalSim
arc-sum areas, all 18 arc-mapped UFs, full WY1922–2014 record) and
`cdec_fnf/fnf_daily_mm.csv` (CLE + CSN cfs → mm/day, negative days
dropped). Training windows are never baked into a series —
`record_start`/`train_start`/`train_end` own the windowing at load
time.

## flowlens.csv — per-entity traced flow lengths

One row per (entity, region grid cell), covering exactly the
`entity_cells.csv` pairs (5,818 rows). `flowlen_m` is the along-network
distance (m) from the cell to the entity outlet, traced on the
HydroSHEDS v2 1-arcsec flow-direction grid (TanDEM-X basis,
hydrosheds.org; see `references.bib`).

Generated file — do not edit. Regenerate (sacsma conda env + `pip
install rasterio`): `python dataprep/build_flowlens.py` — auto-downloads
the four DIR + ACC tiles to `tmp/hydrosheds/` (~6 GB, size-validated,
not in git).

| column | meaning |
|---|---|
| `entity_id`, `key` | as in `entity_cells.csv` |
| `flowlen_m` | traced channel distance to the entity outlet (0 at the outlet cell ⇒ identity UH) |
| `method` | `channel` / `center` / `fallback` — see below |

Conventions. The start pixel per cell is its **main-channel pixel**: the
highest-accumulation pixel in the cell square (capped at 1.3× the entity
area — a pixel carrying more water than the basin cannot drain to its
outlet) whose path reaches the outlet; `center` marks cell-center starts
(157 rows). The outlet is snapped to the nearest pixel (≤ ~2 km) whose
implied upstream area falls within [0.2×, 5×] of the registry
`area_mi2` (snapped-ACC/area landed at 0.75–1.11, median ≈ 1.00).
`uf_07` (multi-outlet composite) traces each cell to where its path
exits the entity footprint. `fallback` (1,362 rows, **6.0% of total area
weight**) = haversine × the entity's median traced sinuosity, for cells
none of whose candidates drain through the outlet — below-outlet valley
cells, square-overlap edge slivers, and sub-cell basins; per-entity
shares are printed by the builder (worst: a few 2–10-cell USGS basins,
and uf_21 at 29% with its outlet at 0.75× area).

Method precedents (`docs/references.bib`): the per-cell channel-pixel
convention follows the coarse-grid upscaling tradition — COTAT's
outlet-pixel-with-area-threshold (`reed2003`) and Dominant River
Tracing's accumulation-based channel selection (`wu2011drt`) — and the
outlet snap uses the upstream-area agreement test the HydroSHEDS authors
use to link GRDC gauges to the grid.

Validation: the archived CADWR flowlens (`cdec15_grid`) reproduce at
r = 0.977–0.992 with median ratio 0.97–1.06 across all 15 basins —
except MKM (ratio 1.18), whose archive was measured to a legacy
reference point ~13 km short of the dam. Per-basin median sinuosity
(traced / straight line) runs 1.0–1.6 in small basins up to 2.0–2.3 for
the large dendritic ones (SHA, BND, uf_06) — the basin-size dependence
a flat factor cannot capture.


