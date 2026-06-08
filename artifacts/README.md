# `artifacts/` — simulated outputs & diagnostic figures

Generated, **not** version-controlled (everything here except this README is
gitignored). Outputs are organized per **run** under `artifacts/<run>/`
(default run name `15cdec`). Regenerate with:

```bash
python -m sacsma.plots                 # -> artifacts/15cdec/
python -m sacsma.plots --run myrun     # -> artifacts/myrun/
```

## Contents of `artifacts/<run>/`

| Path | What |
|------|------|
| `figures/<BASIN>_diagnostics.png` | Per-basin 4-panel: sim-vs-gage time series with **separate calibration & validation skill** (dashed split + shaded validation), daily scatter (1:1), mean-monthly regime, flow-duration curve. |
| `figures/skill_summary.png`       | KGE/NSE (calibration vs validation) and percent bias across all 15 basins. |
| `figures/parity_vs_matlab.png`    | Python port vs. MATLAB reference `simflow` — exact-match proof (worst-case overlay, pooled 1:1 scatter, max daily \|Δ\| per basin). |
| `metrics_15cdec.csv`              | Per-basin calibration/validation KGE, NSE, pbias, r, and mean flow (mm/day + cfs). |

"Simulated" = `sacsma.model.run_basin` (the faithful port from the GA optimum).
"Observed/calibrated" = the gage full-natural-flow target in
`data/reference/gage_15cdec.csv`. "Reference" (parity) = the MATLAB
`simflow` in `data/reference/simflow_15cdec.csv`. Flows are mm/day; areas in
`data/reference/basin_area_15cdec.csv` convert to cfs.

## CalSim cross-compare (`artifacts/calsim/`)

`python -m sacsma.compare` (or `sacsma calsim`) cross-compares each SAC-SMA
calibration set (`15cdec`, `9unimp`, `11obs` — kept **separate**) and **VIC** against
the **CalSim3 historical inflow (the actual)** at the CalSim inflow nodes (the
calsim-view `watersheds.geojson` → `data/gis/calsim3.gpkg`; `Inflow_arc` = the CalSim
node names). Two views are produced:

**1. Per-catchment** — each CalSim3 inflow node scored individually (the set's local
runoff aggregated to that catchment vs CalSim3), drawn on the **merged** whole-basin layer
(`CalSim3_Merged`) so the cumulative single-node basins (Merced `I_MCLRE`, San Joaquin
`I_MLRTN`, Shasta, Trinity) appear as **whole catchments** rather than grey holes.  Below-rim
valley reaches (`I_SJR258/265`, `I_TUO054/105`) and zero-series arcs (`I_RUB002`) are excluded.

| Path | What |
|------|------|
| `monthly_calsets.csv` | Long `[date, arc, node, source, flow_taf]` for source ∈ {each set, `calsim3`, `vic`}. |
| `calset_metrics.csv`      | Per (`node`, `set`) KGE/NSE/pbias/r vs CalSim3, plus `cov_frac`/`n_hru` (the set's honest HRU coverage of that catchment — low = extrapolated, less trustworthy). |
| `best_of_set.csv`         | Best-scoring calibration set per node (vic excluded). |
| `coverage_by_set.csv`     | Per (`set`, `cid`) assignment: `node`, `arc`, `basin`, `kind`, `cov_frac`, `n_hru`. `cov_frac` is the fraction of the catchment the set's HRUs actually sample (gap-free Voronoi, capped per HRU at its grid-cell footprint, so it reads ~1.0 where well-sampled and stays low only at genuine domain-edge gaps like `I_DER001`). Node **inclusion is crosswalk-driven**, not gated by coverage. |
| `vic_full_metrics.csv`    | VIC vs CalSim3 for **every** VIC-covered CalSim3 arc (~206, incl. `I_SLUIS`). |
| `figures/<set>_coverage_map.png` | Scored catchments as a **NSE-vs-CalSim3 choropleth** (0–1) with a coloured **outline of each basin's HRU footprint** (the watershed it represents) + HRU cells. |
| `figures/<set>_skill.png`        | Per-node monthly KGE vs CalSim3, VIC alongside. |
| `figures/vic_coverage_map.png`   | VIC's NSE vs CalSim3 over **all ~200 VIC-covered Rim catchments** (independent of any SAC set). |
| `figures/calsets_bestof.png`     | Best-of-set counts + best achievable per-node skill. |
| `figures/calsets_bestof_map.png` | **Choropleth of the best achievable NSE** (max over the sets) per scored catchment — the spatial companion to `best_of_set.csv`. |
| `figures/diff_bestof_vic_full.png` | **Diverging choropleth** of (best-of − VIC) monthly NSE, **full period** (green = SAC best-of beats VIC). |
| `figures/diff_bestof_vic_validation.png` | Same difference map for the **post-sub-arc-adjustment validation** (test period). |
| `figures/diff_pbias_improvement.png` | Diverging choropleth of the **change in \|percent bias\| from the sub-arc adjustment** (best-of, validation), `\|pbias_raw\| − \|pbias_corr\|`; **green/positive = bias moved toward CalSim3**, red/negative = worse (±20% scale). |
| `figures/cdf_vic_bestof.png` | **CDFs** of NSE/KGE/pbias over the arcs common to VIC and the SAC best-of: best-of (black) vs VIC (purple), **dashed = full period, solid = validation (post sub-arc adjustment)**. |
| `figures/cdf_taf_bias.png` | **CDF of the signed mean-monthly volume bias (actual TAF)** (negative = under-, positive = over-prediction) over the same common arcs: best-of vs VIC, dashed = full / solid = validation. |
| `subarc_validation_metrics.csv`  | Per-(`set`,`arc`) **per-sub-arc bias-correction validation** (train/test split) raw-vs-corrected KGE/NSE/pbias on the held-out test period (see below). |
| `figures/subarc_validation_map.png` | Choropleth of the best-of **corrected** sub-arc NSE (test period). |

All maps share **one fixed extent**; **every figure is 6.5 in wide and rendered at 300 dpi**,
and uses **one consistent colour palette** (CalSim3 black, 15cdec blue, 9unimp green, 11obs
orange, VIC purple). Long titles wrap automatically so they never run off the page.

**2. Basin-level "anchor"** — uses **only the gauge-calibrated sets** (`compare.ANCHOR_SETS` =
**11obs** for the rim systems, **9unimp** for the creeks); **15cdec is excluded** from the
anchor (it is reservoir-calibrated with the ~−23% rim bias and contributes only per-catchment
sub-arcs). Each anchor-set basin (native `run_basin` total) is compared vs the **faithful
CalSim3 reference for that basin** (`ref_kind` column):

- **`unimp`** — if the basin maps to a CalSim **rim system** (`BASIN_RIM_SYSTEM`), the
  reference is that system's single **`FLOW-UNIMPAIRED`** whole-watershed series. This is
  the only correct target for systems like **Sac @ Bend Bridge** (`unimp_SRBB`), whose flow
  includes valley-floor/local accretion (Cottonwood, Battle, Cow, Mill, Thomes…) that the sum
  of individual INFLOW sub-arcs **misses** (~12% low: 598 vs 679 TAF/mo mean for SRBB).
- **`inflow_sum`** — otherwise (creeks, secondary basins with no aggregate series), the
  reference is the **sum of the basin's CalSim3 INFLOW sub-arcs** (`calsim.derive_basin_nodes`,
  a pure projection of the hand-edited crosswalk `data/reference/calsim_crosswalk.csv`:
  `basin_<set>` column + `BASIN_NESTS` for Bend Bridge ⊇ Shasta).

Basin-level **VIC** uses each rim basin's one 8-River major-basin series (the crosswalk
`vic_basin`), avoiding double-counting nested inflows. This anchor now subsumes the
standalone unimpaired-rim comparison for the SAC sets (same FLOW-UNIMPAIRED reference).

**Anchor area nudge.** The SAC-SMA series in *all* anchor outputs (skill metrics +
hydrographs) use a small per-basin **area adjustment** (`data/reference/anchor_area_scale.csv`,
hand-editable; ±10% cap) chosen to **minimise |pbias| without degrading KGE/NSE** — a pure
area rescale is multiplicative on the series, so it leaves correlation untouched and only
moves the bias/variability terms. It is **anchor-only** (the gage cal/val and per-catchment
views keep the true published areas). The cap deliberately leaves the genuine *depth* biases
partly uncorrected (15cdec SHA ≈ −21%, BND ≈ −8% — real precip/calibration bias, not area).

| Path | What |
|------|------|
| `anchor_monthly.csv` | Long `[date, set, basin, source, flow_taf, ref_kind]` (SAC-SMA series include the area nudge). |
| `anchor_metrics.csv`     | Per (`set`, `basin`, `source`) KGE/NSE/pbias vs CalSim3, with `ref_kind` (`unimp` \| `inflow_sum`). |
| `anchor_area_scale.csv`  | The per-basin **area nudge** (copy of `data/reference/anchor_area_scale.csv`): `area_before_mi2` → `area_after_mi2`, `adj_pct`, and pbias/KGE/NSE before vs after. |
| `figures/anchor_skill_{kge,nse,pbias}.png` | **Vertical** dumbbell per (set, basin): SAC-SMA vs VIC. |
| `figures/anchor_scatter.png` | Pooled basin inflow vs CalSim3 anchor-node sum, per set. |
| `figures/anchor_hydrographs.png` | Monthly hydrographs for the **8 main river indices** (CA 8-River Index: Sac@Bend Bridge, Feather, Yuba, American, Stanislaus, Tuolumne, Merced, San Joaquin — Shasta/Trinity/Whiskeytown excluded): each set's SAC-SMA run + VIC vs the CalSim3 FLOW-UNIMPAIRED reference (bold). |
| `figures/main_river_climatology.png` | **Mean-monthly (water-year O–S) climatology** of the 8 main river indices over the full period: SAC-SMA **11obs** vs **VIC** vs **CalSim** (FLOW-UNIMPAIRED), TAF/mo. |
| `vic_full_metrics.csv`   | VIC vs CalSim3 for **every** VIC-covered CalSim3 arc (~206, incl. `I_SLUIS`). |

Headline: **per-catchment** median KGE ≈ **0.67** (15cdec 0.67 / 9unimp 0.77 / 11obs 0.68;
VIC 0.66); **basin-level** median KGE **0.92 (15cdec), 0.95 (11obs), 0.93 (9unimp)** vs the
faithful per-basin reference (FLOW-UNIMPAIRED for rim systems) — aggregation cancels
per-catchment noise. Best-of-set covers **146 nodes**.

> A few GIS catchments have **no usable CalSim3 inflow series**: the Merced
> `I_MCD###`/`I_MSF###` sub-pieces (CalSim lumps the whole Merced into `I_MCLRE`, which is
> the Merced basin anchor — the merged layer makes Merced whole) and Goose Lake; and a few
> arcs (e.g. `I_RUB002`) have an identically-zero series because CalSim folds their flow into
> a parent node. These show in a basin's **footprint outline** but are not scored individually.

The engine lives in `sacsma.calsim` (`run_calsim`, `load_catchments`, `load_crosswalk`,
`derive_basin_nodes`, `build_merged_gis`, `basin_footprints`, `map_hrus_to_catchments`).

**Per-sub-arc bias-correction validation** (`subarc_validation_metrics`, mirrors CalSim's rim
`_2_qmap_historical_validation.py` *without* quantile mapping): for **every multi-arc basin**
in the crosswalk (≥2 `in_calsim3` sub-arcs) — the 6 distributed rim systems
(FOLS/OROV/SRBB/YUBA/ST/TU) **and** the multi-arc secondary basins (15cdec MKM, 11obs BLB, the
9unimp creeks Mokelumne/Bear/Cache/Cosumnes/Stony) — a multiplicative **monthly mean-ratio**
correction (mean CalSim3 / mean estimate, per calendar month) is learned on the **train** water
years (WY1922–1971) and applied on the **held-out test** years (WY1972–2018); the corrected
sub-arcs are then proportionally renormalized so each basin still sums to that estimate's
**anchor total** (`run_basin` for SAC, the basin's VIC total for VIC, both from `anchor_long`)
— "faithful sub-arcs without violating the anchor totals". Applied to every SAC set **and VIC**
(VIC corrected once per arc, deduplicated across sets). This is the *spatial* bias-correction
(a different factor per sub-arc) — distinct from the proportional sub-arc mass-balance
(`make_all(..., mass_balance=True)`, one factor per system, which only enforces the total and
does **not** improve per-catchment skill). On the test period it lifts median sub-arc skill
from KGE ≈0.67/|pbias| ≈20% to **KGE 0.75 (15cdec) / 0.78 (11obs) / 0.83 (9unimp) / 0.73 (VIC)**
and |pbias| to **~6–10%**.

> The standalone "unimpaired rim" comparison (`sacsma calsim --unimp`,
> `unimp_rim_*`) has been **retired** — the basin-level anchor now uses the same
> CalSim `FLOW-UNIMPAIRED` reference for every rim basin, and `anchor_hydrographs.png`
> draws those rivers. `data/reference/calsim_unimpaired_monthly.csv` (the source) is
> still used by the anchor. Headline vs FLOW-UNIMPAIRED: median monthly KGE **11obs 0.95**
> (calibrated to these gauges), **15cdec 0.92**.
