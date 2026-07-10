# `artifacts/` — simulated outputs & diagnostic figures

Generated outputs, organized by application: `cdec15/` (the 15-CDEC
diagnostics) and `calsim/` (the CalLite-domain diagnostics
`9unimp/`/`11obs/`/`12rim/`, the cross-compare `compare/`, and the
alternate-forcing runs `wgen_product_a/` + `historical_lto/`). The committed
copies are the published results of the current data/model state; regenerate
with:

```bash
sacsma plots --domain 15cdec              # -> artifacts/cdec15/
sacsma plots --domain 11obs               # -> artifacts/calsim/11obs/  (or 9unimp / 12rim)
sacsma calsim                             # -> artifacts/calsim/compare/
python -m sacsma.calsim.forcing_compare   # -> artifacts/calsim/forcing_compare/
```

## Per-domain diagnostics (`cdec15/`, `calsim/<domain>/`)

| Path | What |
|------|------|
| `figures/<BASIN>_diagnostics.png` | Per-basin 3-panel: sim-vs-observed time series with **separate calibration & validation skill** (dashed split + shaded validation), plus mean-monthly regimes for the **calibration and validation periods** on a shared y-scale. Daily vs the CDEC gage for `cdec15`; monthly vs the observed FNF for the CalLite domains. |
| `figures/skill_summary.png`       | KGE and percent bias (calibration vs validation) across the domain's watersheds (ordered **north→south** by mean HRU latitude, as is `metrics_<domain>.csv`); **4 in wide**. KGE panel is always the full **0–1** scale (values below 0 clipped, marked `↓`); the pbias panel uses one **fixed ±75% scale shared across all four calibration sets** (±10% guides), so bar heights are directly comparable set to set. cal/val use the same two colors (blue/orange) in both panels. |
| `figures/parity_vs_matlab.png`    | Python vs the original MATLAB `simflow` — exact-match proof (worst-case overlay, pooled 1:1 scatter, max daily \|Δ\| per basin). |
| `metrics_<domain>.csv`            | Per-basin calibration/validation KGE, NSE, pbias, r, and mean flow. |
| `figures/<BASIN>_diagnostics_calsim3.png`, `figures/skill_summary_calsim3.png`, `metrics_<domain>_calsim3.csv` | **CalSim3-basis** variant (`calsim.plots._make_calsim3_diagnostics`): the anchor run scored against **CalSim3's own unimpaired FNF** (TAF/month) instead of the observed-FNF calibration target, on the same cal/val windows. For the CalLite domains (`11obs`/`9unimp`) the sim is the anchor basis (full calibrated footprints; the `catchments.SCREENED_BASINS` over-reach cuts applied); for `cdec15` (via `--fnf-check`) it is the 15cdec model on its own footprint — which reads its honest pooled-GA bias (SHA/BND ≈ −16%) against CalSim3. (The earlier `*_fnf` cross-check against the 11obs/9unimp `fnf_<domain>_monthly` tables was retired 2026-07-07 — those targets' own per-basin offsets vs CalSim3 (`target_vs_calsim3.csv`, e.g. CalaverasRiver +4.8% → NHG read ~5 pp too low) leaked into the 15cdec scores; SHA/BND read an area-inflated −24%/−26% there.) **Parallel/non-destructive**: the fnf-target files above are untouched. `12rim` is skipped (not in the cross-compare). |

"Simulated" = `sacsma.model.run_basin` (the archived GA optimum).
"Observed" = `data/cdec15/gage.csv` (daily) or the domain's monthly FNF in
`data/calsim/`. "Reference" (parity) = the MATLAB `simflow` tables.
The `*_calsim3` variant instead scores against CalSim3's own unimpaired FNF
(`data/calsim/calsim_unimpaired_monthly.csv`); see `tmp/CALSIM3_FNF_FOOTPRINT.md` (local-only note).

## WGEN Product A runs (`calsim/wgen_product_a/`)

Every CalLite-domain watershed forward-simulated with the **WGEN Product A**
forcing (`data/calsim/forcing/wgen_product_a_<domain>.nc` — the CalSim3
stochastic-input pipeline's historical-parallel sequence: identical unsplit
precipitation, temperature detrended to 1991–2020; see `data/INVENTORY.md`).
Same archived GA optima, same model — only the forcing product changes
(`sacsma run <basin> --domain <d> --forcing wgen_product_a`, or
`run_basin(..., product="wgen_product_a")`; the committed tables concatenate
the per-watershed runs).

| Path | What |
|------|------|
| `flow_daily_<domain>.csv` | Long `[date, basin, flow]` daily simulated flow (mm/day, 1915–2018) for every watershed of `9unimp`/`11obs`/`12rim` — same format as the `simflow` reference tables. |

The detrended (warmer early-record) temperatures cost a consistent ~2–4% of
long-term runoff volume (median −3% per domain; daily correlation ≥ 0.999
everywhere — precipitation is unchanged). The per-watershed comparison vs the
Livneh-unsplit run lives in `calsim/forcing_compare/` (below).

## Historical LTO runs (`calsim/historical_lto/`)

The same watersheds forced with the **Historical LTO** climate
(`data/calsim/forcing/historical_lto_<domain>.nc` — the CalSim3 LTO-study
observed climate: the pre-correction **"split"** Livneh precipitation
lineage, **daily 1915–2021**, three years past the Livneh stores; see
`data/INVENTORY.md`). `sacsma run <basin> --domain <d> --forcing
historical_lto` / `run_basin(..., product="historical_lto")`.

| Path | What |
|------|------|
| `flow_daily_<domain>.csv` | Long `[date, basin, flow]` daily simulated flow (mm/day, **1915–2021**) for every watershed of `9unimp`/`11obs`/`12rim`. |

Unlike the WGEN Product A run (temperature-only difference), this is a
**different precipitation realization** — per-watershed differences reflect
the split-vs-unsplit precipitation basis end-to-end. The per-watershed
comparison vs the Livneh-unsplit run (1915–2018 overlap) lives in
`calsim/forcing_compare/` (below).

## Forcing comparison (`calsim/forcing_compare/`)

`python -m sacsma.calsim.forcing_compare` renders each alternate product's
comparison against the Livneh baseline run from the committed run tables (the
`simflow` reference + `flow_daily_<domain>.csv`), on the 1915–2018 overlap.
**Everything is split at 1950** — the split-vs-unsplit precipitation
difference is concentrated before 1950 (median |volume difference| ≈11% in
1915–1949 vs 1.5–3.2% in 1950–2018), and the WGEN detrending effect also
tapers (≈−5% → ≈−2%). Each climate product is applied to **two models** with
the **same figure structure** — SAC-SMA (this repo's daily runs) and **VIC**
(the CalSim3 pipeline's routed monthly runs,
`data/calsim/vic_routed_monthly[_<product>].csv`, aggregated to basins exactly
like the cross-compare anchor and converted to depth over the canonical CalSim
catchment areas) — showing the 9unimp + 11obs calibration sets, basins
north→south. Difference conventions: the split/unsplit figures report
**Unsplit − Split** (prefixes `split_unsplit_`, `vic_split_unsplit_`); WGEN
reports **Detrended − Baseline** (prefixes `wgen_`, `vic_wgen_`). Notably,
VIC's volume response to the temperature detrending is much weaker than
SAC-SMA's (≈−2.5..+0.7% vs ≈−9..−2.7% pre-1950) — it mostly shifts snowmelt
earlier instead. These are deck figures rendered at **600 dpi with ~6.5 pt
fonts**:

| Path (× `split_unsplit_`/`wgen_`/`vic_split_unsplit_`/`vic_wgen_`) | What |
|------|------|
| `figures/<p>_volume_by_period.png` | Per-watershed % volume difference vs the Livneh-unsplit run, 1915–1949 and 1950–2018 bars, one panel per domain. |
| `figures/<p>_annual_diff.png` | 5-yr rolling water-year volume difference, one coloured line per watershed, with a short cross-basin mean-\|difference\| panel below; 1950 marked. |
| `figures/<p>_regime_by_period_<domain>.png` | Mean-monthly regime, baseline vs product, both periods, one panel per basin of the domain, annotated with the per-period % volume difference. |
| `figures/agg_{split_unsplit,wgen}_annual_diff.png` | **Cross-model aggregate** (one per climate product): SAC-SMA and VIC on one axes — % difference of the aggregate WY volume summed over the **disjoint** watersheds (depth × canonical CalSim area; SHA and BLB dropped as double-counted: SHA ⊂ BND, BLB = StonyCreek's arcs), 5-yr rolling, per-period aggregates as dashed levels + legend values. |
| `split_unsplit_anchor_skill.csv` | **Anchor-skill effect of the split product**: per-basin KGE/NSE/pbias vs CalSim3 for the Livneh-unsplit baseline AND the `historical_lto` run, both re-simulated through the official anchor basis (full footprints; the `SCREENED_BASINS` over-reach cuts applied) and scored on identical months (`[set, basin, period, forcing, n_months, kge, nse, pbias]`; periods full / pre1950 / post1950, house WY1950 split). Unlike the figures above this re-simulates (a few minutes). |
| `figures/split_unsplit_skill_boxplot.png` | Boxplots of that table: KGE / NSE / pbias per period, baseline (grey) vs split (orange), all 20 anchor basins pooled and overlaid as points. Split pulls in the bad pre-1950 tail (TNL, Fresno) but slightly widens post-1950 volume spread; NSE improves modestly everywhere. |

## CalSim cross-compare (`calsim/compare/`)

`sacsma calsim` (or `python -m sacsma.calsim.compare`) cross-compares each
SAC-SMA calibration set (`15cdec`, `9unimp`, `11obs` — kept **separate**) and
**VIC** against the **CalSim3 historical inflow (the actual)** at the CalSim
inflow nodes (`data/calsim/gis/calsim3.gpkg`). Two views are produced:

**1. Per-catchment** — each CalSim3 inflow node scored individually (the set's local
runoff aggregated to that catchment vs CalSim3), drawn on the **merged** whole-basin layer
(`CalSim3_Merged`) so the cumulative single-node basins (Merced `I_MCLRE`, San Joaquin
`I_MLRTN`, Shasta, Trinity) appear as **whole catchments** rather than grey holes.  Below-rim
valley reaches (`I_SJR258/265`, `I_TUO054/105`) and zero-series arcs (`I_RUB002`) are excluded.
The per-sub-arc scores live in the **CSVs only**; the **maps and figures show skill at the
main-basin level** — every sub-area polygon is coloured by its watershed's basin-anchor score
(view 2 below), never its own sub-arc score.

| Path | What |
|------|------|
| `monthly_calsets.csv` | Long `[date, arc, node, source, flow_taf]` for source ∈ {each set, `calsim3`, `vic`}. |
| `calset_metrics.csv`      | Per (`node`, `set`) KGE/NSE/pbias/r vs CalSim3, plus `cov_frac`/`n_hru` (the set's honest HRU coverage of that catchment — low = extrapolated, less trustworthy). |
| `coverage_by_set.csv`     | Per (`set`, `cid`) assignment: `node`, `arc`, `basin`, `kind`, `cov_frac`, `n_hru`. `cov_frac` is the fraction of the catchment the set's HRUs actually sample. Node **inclusion is crosswalk-driven**, not gated by coverage. |
| `vic_full_metrics.csv`    | VIC vs CalSim3 for **every** VIC-covered CalSim3 arc (~206, incl. `I_SLUIS`). |
| `figures/<set>_coverage_map.png` | The set's catchments coloured by their **main basin's anchor NSE** (0–1; all sub-areas of a watershed share one colour), with a coloured **outline of each basin's HRU footprint** (the watershed it represents) + HRU cells. Each scored watershed is labelled in place (abbreviated basin name + NSE). |
| `figures/<set>_skill.png`        | Per-**basin** anchor KGE vs CalSim3, VIC alongside on the same basins. |
| `figures/shasta_footprint_panels.png` | **Shasta footprint-screening methods maps** (2×3: 5 panels + legend): the VIC routing grid with its Goose Lake over-reach red (`vic_gridinfo_I_SHSTA*.csv`, cells shaded by in-basin fraction), the full 11obs HRU cells (screened-out cells red), the 15cdec sub-grid HRU points, the CalSim3 SHSTA delineation (Goose Lake block dash-outlined), and the screened result: the GIS-screened 11obs cells drawn **over the full VIC no-Goose-Lake grid** — VIC still simulates all 546 of those cells (Goose Lake was its only cut), so terrain only VIC covers stays purple. Black SHSTA outline on every panel; red = outside the catchment / dropped by screening. **Single-basin methods illustration of `tmp/CALSIM3_FNF_FOOTPRINT.md` — the basin-level colouring rule doesn't apply here.** |
| `figures/{sns,chowchilla,tnl,fresno}_footprint_panels.png` | The other single-basin footprint maps (`compare.make_basin_footprint_maps`, configs `FOOTPRINT_MAP_BASINS`; grids `vic_gridinfo_{8RI_N_MEL,I_ESTMN,I_TRNTY,I_HNSLY}.csv` — the VIC grid is always used as-is). The story follows the basin's `SCREENED_BASINS` membership: **SNS and Chowchilla are screened** (out-of-catchment cells solid red = dropped; final panel = screened cells over the full VIC grid), **Trinity and Fresno are not** (cells extending outside the boundary red-hatched but **kept**; final panel = ALL HRU cells over the full VIC grid). The 15cdec panel appears only for SNS (NML). Notable: Trinity has no out-of-catchment cells and runs −5.0% effective vs the catchment; Fresno claims +17.5% effective (kept as-is) while VIC's grid is +0.0%; SNS/Chowchilla claim +7.6%/+12.6% full → screened. |
| `figures/hru_veg_15cdec.png`, `figures/hru_soil_15cdec.png` | **Whole-domain 15cdec HRU maps** by `veg_class` and `soil_class` (`compare.make_hru_attribute_maps`; categorical, tab20). Both are per-cell codes; the repo has **no code→name legend** (the study cites the Hansen et al. 2010 MODIS/IGBP land-cover lineage), so legends stay numeric. Input maps, not the basin-level cross-compare. |
| `figures/hru_kpet_15cdec.png` | The same 15cdec HRUs coloured by the **calibrated Hamon PET coefficient `Kpet`**. In the pooled optimum `Kpet` is regionalized on **soil zone alone** (single-valued per `soil_class`; `veg_class` adds nothing — verified), so this map mirrors `hru_soil_15cdec.png`, not the vegetation map. Exact lookup: `hru_kpet_by_soil_15cdec.csv`. |
| `figures/hru_kpet_calsim.png` | 11obs + 9unimp HRUs by `Kpet` (shared colourbar). `Kpet` is **per-basin uniform** here (each per-watershed calibration = one value), so it reads as a per-basin choropleth at HRU-cell resolution; basins share boundary cells with different `Kpet`, so every basin's full cell set is drawn (higher-`Kpet` last for a deterministic contested cell). |
| `hru_veg_kpet_15cdec.csv` | Per-`veg_class` HRU count + `Kpet` spread (min/median/max, distinct count). The spread within a class is **soil-driven** — `Kpet` is not a function of `veg_class`. |
| `hru_kpet_by_soil_15cdec.csv` | Exact **`soil_class` → `Kpet`** lookup (11 rows, 1:1) + HRU counts — the real 15cdec `Kpet` regionalization (`compare.kpet_soil_table`). |
| `subarc_validation_metrics.csv`  | Per-(`set`,`arc`) **per-sub-arc QMAP validation** (train/test split) raw-vs-corrected KGE/NSE/pbias on the held-out test period, with `anchor_kind` (see below). |
| `subarc_qmap_<set>.csv`          | **Full-period QMAP-corrected sub-arc series** (one per SAC set + `_vic`): `[date, arc, node, basin, anchor_kind, flow_taf_raw, flow_taf_qmap]` — the deliverable. |

### CalSim ↔ SAC-SMA basin maps (`calsim_sacsma_map`)

Basin-level maps on the composite of the two gauge-calibrated sets **9unimp + 11obs** (9unimp
wins where both cover an arc — the only overlap is Stony Creek / BLB); **15cdec keeps its own
per-set map but is not in the composite**. Every sub-area polygon is coloured by its **main
basin's anchor score** (the basin `run_basin` total vs its faithful CalSim3 reference, view 2
below), so all sub-areas of a watershed share one colour. NSE/KGE use a 0–1 sequential scale
(**plasma**, negatives clamp to the floor); **signed pbias** uses a diverging scale.
**Bend Bridge is left uncoloured** on these maps (`compare.MAP_EXCLUDE_BASINS`) — for the
nested Sacramento system only Shasta is drawn; BND keeps its scores in
`basin_map_metrics.csv` and every other anchor product. Every coloured watershed carries an
**in-map annotation** (abbreviated basin name + its value for the mapped metric, white-halo
text at a representative interior point of the basin; `compare._BASIN_ABBREV` shortens the
long 9unimp names).

| Path | What |
|------|------|
| `basin_map_metrics.csv` | Per `[set, basin, which (sac/vic), ref_kind, n_months, kge, nse, pbias]` — the anchor scores behind the maps. |
| `figures/calsim_sacsma_map_{nse,kge,pbias}.png` | SAC-SMA composite (9unimp + 11obs basins), one PNG per metric. |
| `figures/calsim_vic_map_{nse,kge,pbias}.png` | The same maps for **VIC**, on the same basins. |
| `figures/calsim_sacsma_minus_vic_{nse,kge,pbias}.png` | **SAC-SMA − VIC** basin-level difference, diverging (blue = SAC better, white = tie, red = VIC better; pbias compares \|bias\|). |

All maps share **one fixed extent**; **every figure is ≤6.5 in wide, rendered at ≥300 dpi,
with no text larger than 8 pt**, and uses **one consistent colour palette** (CalSim3 black,
15cdec blue, 9unimp green, 11obs orange, VIC purple). Long titles wrap automatically so they
never run off the page.

**2. Basin-level "anchor"** — uses **only the gauge-calibrated sets** (`compare.ANCHOR_SETS` =
**11obs** for the rim systems, **9unimp** for the creeks); **15cdec is excluded** from the
anchor (it is reservoir-calibrated with the ~−23% rim bias and contributes only per-catchment
sub-arcs). Each anchor-set basin (native `run_basin` total, on the **canonical CalSim
catchment area** `data/calsim/basin_area_<set>_calsim.csv`) is compared vs the **faithful
CalSim3 reference for that basin** (`ref_kind` column):

- **`unimp`** — if the basin maps to a CalSim **rim system** (`BASIN_RIM_SYSTEM`), the
  reference is that system's single **`FLOW-UNIMPAIRED`** whole-watershed series. This is
  the only correct target for systems like **Sac @ Bend Bridge** (`unimp_SRBB`), whose flow
  includes valley-floor/local accretion (Cottonwood, Battle, Cow, Mill, Thomes…) that the sum
  of individual INFLOW sub-arcs **misses** (~12% low: 598 vs 679 TAF/mo mean for SRBB).
- **`inflow_sum`** — otherwise (creeks, secondary basins with no aggregate series), the
  reference is the **sum of the basin's CalSim3 INFLOW sub-arcs**
  (`catchments.derive_basin_nodes`, a pure projection of the hand-edited crosswalk
  `data/calsim/calsim_crosswalk.csv`: `basin_<set>` column + `BASIN_NESTS` for
  Bend Bridge ⊇ Shasta).

Basin-level **VIC** uses each rim basin's one 8-River major-basin series (the crosswalk
`vic_basin`), avoiding double-counting nested inflows. Residual volume differences are
**honest depth biases** (real forcing/calibration bias, not area) — the SAC volume sits on
the same CalSim catchment area as its sub-arcs and the reference.

| Path | What |
|------|------|
| `anchor_monthly.csv` | Long `[date, set, basin, source, flow_taf, ref_kind]` — each basin's **full calibrated footprint** depth × the canonical CalSim catchment area, EXCEPT the **`catchments.SCREENED_BASINS`** (SHA, BND, SNS, ChowchillaRiver), which are GIS-screened (`screened_footprint`) because their footprints materially over-reach the catchment: SHA/BND the endorheic Goose Lake block (the exact parallel of VIC's `no_gooselake` fix), SNS/Chowchilla a delineation over-reach (revised 2026-07-08 from the earlier uniform screening: trimming ordinary boundary rasterization distorts the calibrated depth). |
| `anchor_metrics.csv`     | Per (`set`, `basin`, `source`) KGE/NSE/pbias vs CalSim3, with `ref_kind` (`unimp` \| `inflow_sum`) — same basis as `anchor_monthly.csv`. |
| `anchor_metrics_full.csv`, `anchor_monthly_full.csv` | The **everything-unscreened** anchor (`compare.make_anchor_full`) — the parallel view with the screened basins also on their complete footprints (incl. Goose Lake). |
| `anchor_screened_vs_full.csv`, `figures/anchor_screened_vs_full.png` | The delta between the unscreened parallel and the official anchor — differs **only at the screened basins** (SHA −8.9%→+0.1%; BND −1.9%→+4.8%, exposing an honest over-prediction the Goose dilution was cancelling; SNS −7.8%→−0.8%; ChowchillaRiver −14.3%→−4.3%); every other basin is identical in both views. Includes the **VIC benchmark** on the same months/reference (`pbias_vic`/`kge_vic`/`mean_vic_taf`; purple diamond). See `tmp/CALSIM3_FNF_FOOTPRINT.md`. |
| `target_vs_calsim3.csv` | How different the SAC-SMA **calibration target itself** (`fnf_<domain>` `obs_mm`, on the canonical CalSim area) is from CalSim3's own flow — the bias floor a perfect-fit model inherits. Most basins are within a few % (r≈1.0). A per-basin `class` (with `area_gis`/`area_implied`/`area_pub`) splits the rest: **`area_artifact`** (target normalized on the published area but scored on the CalSim area — ChowchillaRiver/SNS/YRS; documented, not re-normalized) vs **`product_offset`** (a real historical-FNF-vs-CalSim3 difference — CacheCreek, BearRiver, …). See `tmp/CALSIM3_FNF_FOOTPRINT.md` §3. |
| `basin_area_<set>_calsim.csv` | Copy of the canonical CalSim catchment areas used for the anchor volume. |
| `figures/anchor_skill_{kge,nse,pbias}.png` | **Vertical** dumbbell per (set, basin): SAC-SMA vs VIC (basins ordered **north→south** within each set; 9unimp basin names abbreviated — `compare._BASIN_ABBREV`). **15cdec is folded onto the same figure** after 9unimp/11obs, separated by a **dashed divider** — its numbers come from its own `anchor_monthly_15cdec.csv`/`anchor_metrics_15cdec.csv` (`compare.make_anchor_15cdec`), never merged into `anchor_monthly.csv`/`anchor_metrics.csv`/`ANCHOR_SETS` (15cdec stays excluded from everything else downstream — rolling skill, sub-arc QMAP). 15cdec's own calibration is **daily**; here its `run_basin` output is aggregated to **monthly** TAF like the other sets, so the KGE is monthly and comparable to VIC. |
| `anchor_metrics_by_period.csv`, `figures/anchor_skill_{kge,pbias}_{pre,post}1950.png` | The same anchor skill (11obs/9unimp **+ 15cdec**, same dashed-divider convention) re-scored on the **pre-/post-WY1950** months (house 1950 split; `compare.make_anchor_skill_periods`, standalone from `anchor_monthly.csv` + `anchor_monthly_15cdec.csv`). KGE on the 0–1 scale; the two pbias figures share one symmetric y-scale. |
| `figures/anchor_scatter.png` | Pooled basin inflow vs CalSim3 anchor-node sum, per set. |
| `figures/anchor_hydrographs.png` | Monthly hydrographs for the **8 main river indices** (CA 8-River Index: Sac@Bend Bridge, Feather, Yuba, American, Stanislaus, Tuolumne, Merced, San Joaquin — Shasta/Trinity/Whiskeytown excluded): each set's SAC-SMA run + VIC vs the CalSim3 FLOW-UNIMPAIRED reference (bold). |
| `figures/main_river_climatology.png` | **Mean-monthly (water-year O–S) climatology** of the 8 main river indices over the full period: SAC-SMA **11obs** vs **VIC** vs **CalSim** (FLOW-UNIMPAIRED), TAF/mo. |
| `rolling_skill_30yr.csv`, `rolling_skill_basin_30yr.csv`, `figures/rolling_*` | 30-yr rolling KGE/NSE/pbias/seasonal-mismatch on the basin anchors, per set (+ VIC). KGE/NSE axes are always the full **0–1** scale (below-zero values clipped); pbias/seasonal-mismatch bounds are shared across the anchor sets. |

Headline: **per-catchment** median KGE ≈ **0.67** (15cdec 0.67 / 9unimp 0.77 / 11obs 0.68;
VIC 0.66); **basin-level** median KGE **0.92 (15cdec), 0.95 (11obs), 0.93 (9unimp)** vs the
faithful per-basin reference (FLOW-UNIMPAIRED for rim systems) — aggregation cancels
per-catchment noise.

> A few GIS catchments have **no usable CalSim3 inflow series**: the Merced
> `I_MCD###`/`I_MSF###` sub-pieces (CalSim lumps the whole Merced into `I_MCLRE`, which is
> the Merced basin anchor — the merged layer makes Merced whole) and Goose Lake; and a few
> arcs (e.g. `I_RUB002`) have an identically-zero series because CalSim folds their flow into
> a parent node. These show in a basin's **footprint outline** but are not scored individually.

The engine lives in `sacsma.calsim.catchments` (`run_calsim`, `load_catchments`,
`load_crosswalk`, `derive_basin_nodes`, `basin_footprints`, `map_hrus_to_catchments`).

**Per-sub-arc QMAP validation** (`subarc_validation_metrics`, a faithful port of CalSim's rim
`_2_qmap_historical_validation.py`): for **every multi-arc basin** in the crosswalk (≥2
`in_calsim3` sub-arcs) — the 6 distributed rim systems (FOLS/OROV/SRBB/YUBA/ST/TU) **and** the
multi-arc secondary basins (15cdec MKM, 11obs BLB, the 9unimp creeks
Mokelumne/Bear/Cache/Cosumnes/Stony) — two steps, learned on the **train** water years
(WY1922–1971) and scored on the **held-out test** years (WY1972–2018): **(1) quantile mapping** —
each sub-arc is mapped, per calendar month, from its own distribution onto its CalSim3 `INFLOW`
distribution (empirical CDF within range, gamma tail beyond; `sacsma.calsim.qmap`); **(2)
mass-balance to the SAC-SMA simulated basin total** — the QMAPped sub-arcs are rescaled so each
basin sums to that estimate's **own `run_basin` total** (the basin's VIC total for VIC), so QMAP
fixes the per-catchment *shape* while the estimate keeps its own basin *volume* (`anchor_kind` =
`sac_sim`/`vic_sim`; a basin with no simulated total falls back to `own_sum`). A *nested cumulative
inflow* is included in each basin that lists it — `I_SHSTA` is both its own SHA basin **and** a
Bend Bridge sub-arc (`BASIN_NESTS`) — so a cumulative basin's sub-arcs reconstruct its `run_basin`
total. Applied to every SAC set **and VIC** (VIC corrected once per arc, deduplicated across
sets). On the test period it lifts median sub-arc skill from KGE ≈0.67/|pbias| ≈20% to **KGE 0.76
(15cdec) / 0.81 (11obs) / 0.86 (9unimp) / 0.75 (VIC)** and |pbias| to **~4–11%**.

### VIC precipitation basis — split vs unsplit

The VIC benchmark (`data/calsim/vic_routed_monthly.csv`) is built from the **unsplit**
VIC historical run (`…/vic/output/routed/Historical_Unsplit`), which shares its precipitation
basis with the SAC-SMA forcing (both unsplit — see `data/INVENTORY.md`), and uses the
`_no_gooselake` series at `I_SHSTA` and `8RI_SRBB`. The following diagnostic quantifies how
much the **unsplit** precip differs from the older **split** product per basin.

| Path | What |
|------|------|
| `vic_precip_split_vs_unsplit.csv` | Area-weighted basin-mean precip (mm/yr) for every 11obs & 9unimp basin: split, unsplit, and Δ% for the **full** period and **P1 (1915–49)** / **P2 (1950–2018)**. |
| `vic_precip_split_vs_unsplit.md`  | Tables + findings. |
| `figures/vic_precip_split_vs_unsplit.png` | Δ% by basin and period, faceted by domain (basins ordered **north→south**). |

Headline: the split/unsplit divergence is **overwhelmingly pre-1950** (P1 up to ±21% —
Trinity +21%, Bend Bridge −9%; unsplit wetter in the southern/Sierra basins, drier in the
northern Sacramento ones), collapsing to within **±2% after 1950** everywhere except
**Trinity (+8%)** — the one basin where the unsplit switch is materially consequential in the
modern era. Generated by the **calsim repo** script
`calsim3-stochastic-input-generation/data/GENERATED/mod_forcing/climate/precip_split_vs_unsplit.py`.
