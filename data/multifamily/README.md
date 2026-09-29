# data/multifamily — multi-timescale training-entity registry

`entities.csv`: 291 entities — the 95 base training entities (9
`uf_monthly` + 69 `usgs_daily` + 17 `cdec_daily`: 15 + CLE + CSN),
followed by the opt-in `calsim_monthly` family of 196 CalSim3 rim arcs (see
"The CalSim3 arc family" below). One row per (site × timescale × family); every target trains
as an independent entity. Which entities a run trains on is chosen at launch
(`dpl train --basins`; a run without it takes the 95 base entities); the
registry is the superset.

Generated files — do not edit. The committed `entities.csv`,
`entity_cells.csv` and `flowlens.csv` are the `--calsim-arcs` build.
Regenerate them in this order, from the repo root:

    python dataprep/build_calsim_arcs.py                 # sacsma env; the three arc tables under calsim/
    python dataprep/build_entities.py --calsim-arcs      # sacsma env
    python dataprep/build_entity_cells.py --calsim-arcs  # sacsma env
    python dataprep/build_flowlens.py --calsim-arcs      # needs rasterio; appends the arc rows

Without `--calsim-arcs`, `build_entities.py` and `build_entity_cells.py`
write the 95-entity base tables over the committed ones.
`build_calsim_arcs.py` reads the registry's base rows, so starting without a
registry, run `build_entities.py` once without the flag first. UF pour
points are hand-maintained in `data/dwr_unimpaired/uf_gauges.csv`.

A de-duplication rule is applied at build time: a watershed does not
train at both daily and monthly timescales unless its daily record is short
(starts ~2000 or later). Eleven monthly twins of long-record CDEC dailies
are built and then dropped — UF 8, 9, 11, 14, 15, 16, 18, 19, 22 and both
`obs11_monthly` rows (SHA ≡ SIS daily, TNL ≡ CLE daily at 96% shared area).
The short-daily pairs stay: uf_13/cdec_CSN (CSN 1999-04) and uf_06/cdec_BND
(SBB 1999-05). Per-drop twin and record start: `DEDUP_DROPS` in the builder;
record completeness verified against `cdec15/gage.csv` and
`cdec_fnf/fnf_daily.csv`.

A second build-time drop list exists for target validity (`TARGET_DROPS`,
empty by default). `uf_03` (Cache Creek above Rumsey) is the documented case:
the observation is the routed outflow below Clear Lake and Indian Valley,
while its four arcs are the inflows CalSim routes through those lakes
itself (+16% volume, r 0.877 against the obs), so a lake-free cell
parameterization can only fit it by learning the lakes' storage and
evaporation into the runoff parameters the inflow arcs then reapply. The
row is kept (flag `obs_routed_through_lakes`) so a run without the USGS
family can still supervise the Cache Creek cells; a run with the in-basin
USGS gauges can leave it out via `--basins`.

## The CalSim3 arc family (`--calsim-arcs`, opt-in)

The committed tables carry this family. Built with `--calsim-arcs` (build
order above), the three builders append it after the 95 base entities, whose
rows stay byte-identical (the builders gate it). Its inputs come from
`python dataprep/build_calsim_arcs.py`: `calsim/arc_hierarchy.csv`,
`calsim/arc_obs_mask.csv` and `calsim/calsim3_inflow_monthly_mm.csv`.
Without the flag, `build_entities.py` writes only the 95 base rows and
`build_entity_cells.py` ignores any `calsim_monthly` rows in the registry,
so both rebuild the base tables byte for byte from either registry.
`build_flowlens.py --calsim-arcs` traces only the arc entities and appends
them to the existing store, whose base rows it copies byte for byte; a
`build_flowlens.py` run without the flag re-traces every entity the
registry holds.

`calsim_monthly` — 196 entities, one per CalSim3 rim `INFLOW` arc with a
series and a `CalSim3_Merged` polygon (the below-rim `EXCLUDE_ARCS` and the
polygon-less `I_RUB002`, `I_JBP006`, `I_FSL012` have none):

- `entity_id` `cs_<ARC>` (e.g. `cs_I_ALMNR`), `site_id` = the arc,
  `delineation` `arcs`, `arcs` = the arc itself, `area_mi2` = its `SQ_MI`
  (unrounded: the depth basis of the store), no outlet coordinate;
- `obs_store` `calsim/calsim3_inflow_monthly_mm.csv:depth_mm`, kept only on the
  months of `calsim/arc_obs_mask.csv` (the arc's own gauge record: listed
  period minus the report's correlation-extension years, WY1950-75 +
  WY1986-2015 only, so the WY1976-85 holdout never enters);
  `train_start/end` 1949-10-01 / 2015-09-30; `n_obs` = the arc's mask months
  (0 for donor-split, proportioned, reservoir-without-record and no-record
  arcs — the loader refuses those if selected);
- `flags`: `tier_<A-F|X>` (from `arc_hierarchy.csv`), `train_default` (tier
  A and not an entity duplicate: 64 arcs, 25,523 arc-months),
  `duplicates_<entity>` (the 7 arcs that are a single-arc training entity:
  I_SHSTA, I_TRNTY, I_MLRTN, I_MCLRE, I_NHGAN, I_PTH070, I_ESTMN),
  `training_usgs_<gid>` (built from a registry USGS gauge; 9 of the tier-A
  arcs).

Cells: the `arcs` path of the base entities (square-overlap of the arc's
polygon; the 7 duplicates reproduce their entity's cells exactly). The
overlap sum equals the polygon's true area to 1e-9 but departs from the
`SQ_MI` attribute by -3.1% .. +1.5% for single arcs (55 of 196 beyond
0.2%; the attribute error averages out in multi-arc entities). 3,842 rows,
2,393 cells, 197 of them outside the 2,652-cell base basis.

Flow lengths: no outlet coordinate, so every cell is traced to where its path
leaves the arc's cell-square footprint — uf_07's mode (the builder re-traces
uf_07 and asserts its stored rows are reproduced exactly) and tier 2's
convention for arcs. Fallback weight 0.8%. These lengths run shorter than an
outlet-snapped trace (the 7 duplicate arcs: mean 0.49-0.87 of their
entity's length, r 0.06-0.71): paths along a footprint edge exit early.

Loading: `load_domain_tensors(basins=None)` (a run without `--basins`) keeps
the 95 base entities — the arc family loads only when named. Training on it
(the fourth family's loss weight, anchor rescaling) is not wired yet: with
`--mt-family-weight none` the monthly loss term would pool named `cs_*`
entities with the `uf_monthly` ones, and a family-share run refuses them
(`calsim_monthly` has no share key in `config.FAMILY_KEYS`).

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

Flags are annotations for readers: each marks a caveat of its row, and no code
branches on them. Footprint and target choices live in the builder
(`EXTRA_ARCS`, `TRIM_ARCS`, `TARGET_DROPS`), and which entities a run trains on
is set by `--basins`.

`train_only` (Tulare 4) · `polygon_2.6pct_above_published_area` (TRM —
`area_mi2` keeps the published 561 as the depth basis) ·
`outlet_below_delineation` (gauge/dam 5–13 km below the delineation) ·
`obs_includes_valley_floor` + `no_gauge_composite`
(UF 7) · `calsim_ref_wetter_summers` (Bear) ·
`daily_runs_6pct_below_monthly` (CSN) · `fnf_computed_at_trinity_dam` (CLE) ·
`footprint_includes_valley_node` (BND — the cell set includes the series-less
`I_SRBB_VAL` valley node because the Bend Bridge FNF drainage covers the
valley floor between the rim margin and the gauge; `area_mi2` keeps the
published 8,900 as the depth basis) ·
`footprint_excludes_below_gauge_arcs` (YRS — the two Deer Creek arcs
`I_DER001`/`I_DER004` (64 mi²) join the Yuba below the Smartville gauge,
so their water never passes the observing station; they are trimmed from
the cell set and `area_mi2` keeps the published 1,108).

Note for consumers of `arcs`: `I_RUB002` (FOL's list) has no
`CalSim3_Merged` polygon — its terrain was dissolved into `MFA025`, so
coverage is complete.

## entity_cells.csv — cell sets and weights

One row per (entity, region grid cell): 9,691 rows on 2,849 distinct cells
of `data/region/grid_cells.csv`. The base set's 5,849 rows (95 entities,
2,652 cells) come first, then the arc family's 3,842 (196 entities, 2,393
cells, 197 of them outside the base set). Replaces the per-domain `hruinfo`
tables as the aggregation basis for entity training.

Flow lengths live in `flowlens.csv` (below), keyed identically; outlet
coordinates live in the registry only.

Generated file — do not edit. Regenerate (sacsma conda env):
`python dataprep/build_entity_cells.py --calsim-arcs` (without the flag it
writes the 5,849-row base table).

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
the common cells). Per-entity Σoverlap reproduces each base footprint's
reference area to +0.14% worst-case (uf_10; the build asserts <0.2%). The
mapped sums equal the polygons' true geometric areas — the small positive
residual is mostly the gpkg `SQ_MI` attributes running ~0.05–0.12% below
true geometric area, plus a once-per-arc count of arc-overlap slivers
(largest pairwise overlap 1.4% of the smaller arc; ≤0.04% at entity
level).

Cell basis: the base set's cell union is 2,652 distinct cells. The dropped
monthly twins contribute none of their own (their cells are their daily
twins'; TNL's extra `I_LWSTN` cells are those of `usgs_11525500`); the
Tulare polygons reach 8 edge cells beyond the `cdec15_grid` cell sets;
cdec_BND's `I_SRBB_VAL` valley cells all serve uf_06 as well; uf_03 holds
93 cells, 49 of which no other entity uses (its other 44 are shared with
the three in-basin USGS gauges and uf_02/uf_04 edge overlaps); the two
Deer Creek cells below the YRS gauge are not in the store. A run's basis
is the union over the entities it selects (2,603 without uf_03). The
full-rim basis — every cell touching any rim polygon + USGS + Tulare — is
2,847. Statics coverage is **complete**: `data/region/soilveg_continuous.csv`
and `lai_climatology.csv` cover all 4,410 region cells (full-grid ingest,
`a77e4a8`).

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
`entity_cells.csv` pairs (9,691 rows: 5,849 base + 3,842 arc-family
rows, the latter traced as "The CalSim3 arc family" describes).
`flowlen_m` is the along-network distance (m) from the cell to the entity
outlet, traced on the
HydroSHEDS v2 1-arcsec flow-direction grid (TanDEM-X basis,
hydrosheds.org; see `references.bib`).

Generated file — do not edit. Regenerate (sacsma conda env + `pip
install rasterio`): `python dataprep/build_flowlens.py --calsim-arcs`
traces the arc family and appends it after the stored base rows, which it
copies byte for byte (it reads the tiles already in `tmp/hydrosheds/`).
The full re-trace, `python dataprep/build_flowlens.py` without the flag,
traces every registry entity and auto-downloads the four DIR + ACC tiles
to `tmp/hydrosheds/` (~6 GB, size-validated, not in git).

| column | meaning |
|---|---|
| `entity_id`, `key` | as in `entity_cells.csv` |
| `flowlen_m` | traced channel distance to the entity outlet (0 at the outlet cell ⇒ identity UH) |
| `method` | `channel` / `center` / `fallback` — see below |

Conventions. The start pixel per cell is its **main-channel pixel**: the
highest-accumulation pixel in the cell square (capped at 1.3× the entity
area — a pixel carrying more water than the basin cannot drain to its
outlet) whose path reaches the outlet; `center` marks cell-center starts
(159 base rows). The outlet is snapped to the nearest pixel (≤ ~2 km) whose
implied upstream area falls within [0.2×, 5×] of the registry
`area_mi2` (snapped-ACC/area landed at 0.75–1.11, median ≈ 1.00).
`uf_07` (multi-outlet composite) traces each cell to where its path
exits the entity footprint. `fallback` (1,362 base rows, **5.9% of the
base set's area weight**) = haversine × the entity's median traced
sinuosity, for cells none of whose candidates drain through the outlet — below-outlet valley
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


