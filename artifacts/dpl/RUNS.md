# dPL run track record

Structure: **canonical** runs live directly under `artifacts/dpl/<label>`.
Frozen-physics runs (`hamon_dense`, `hamon`, `pt`, `noah`) carry
`checkpoints/best.pt`, `train_log.csv`, `params_dpl.csv`, `metrics_<label>.csv`
(+ `params_canopy.csv` for `noah`). The two canonical SAC×LSTM **ensembles**
(`noah_lstm_feat`, `noah_lstm_resid`) hold `seed*/checkpoints/best.pt` +
per-seed `metrics_hybrid_<variant>.csv` and a top-level
`metrics_hybrid_<variant>.csv` scoring the **ensemble-mean** flow
(`hybrid.evaluate.score_ensemble`). Exploratory/superseded run artifacts were
pruned; their findings stay in the track record below (names there no longer
resolve to on-disk runs). `--physics` is REQUIRED with no default, so `testing/`
holds only gitignored local scratch; `hybrid/` now holds only the shared
Noah-lite frozen-sim cache. `fidelity/` is the numerics benchmark
(infrastructure, not a run). All skill numbers are pooled 15-basin mean KGE,
frozen-model scoring (numba; PT via `sacsma.pet_pt`, Noah-lite via
`sacsma.sma_noah_lite`) unless marked *(torch)* — only the full 7-param Noah
ET (`noah_grid*`) remains torch-only (no frozen core).

Standing methods shared by every canonical grid run: `physical` feature
variant, pooled 15cdec training (daily gage FNF, cal WY1989–2003 / val
WY2004–2018), CalSim3-footprint aggregation (`--calsim-footprint`), NNSE+log
loss + variance-matching term, cal-KGE selection every 2 epochs, truncated
no-grad spinup from 1978-10-01.

## Canonical lineage

| label | domain | delta vs predecessor | cal/val KGE |
|---|---|---|---|
| `hamon_dense` | 15cdec (7891 HRU) | — (the original dPL) | **0.806/0.840** (retrained 2026-07-14) |
| `hamon` | 15cdec_grid (2074 cells) | native-grid retrain + CalSim3 footprint | 0.807/0.829 |
| `pt` | 15cdec_grid | Priestley–Taylor PET (Bristow–Campbell Rn) + snow-cover albedo (0.6) + arid dewpoint depression (2 °C) | 0.799/0.826 |
| `noah` | 15cdec_grid | Noah-lite canopy ET (1 learned DOF `soil_chi`) on PT potential | 0.767/0.799 |
| `noah_lstm_feat` | 15cdec_grid | SAC×LSTM **feature** hybrid on `noah` physics — 5-seed ensemble mean | 0.923/0.869 |
| `noah_lstm_resid` | 15cdec_grid | SAC×LSTM **residual** hybrid on `noah` physics — 8-seed ensemble mean | **0.926/0.873** |

**2026-07-15 canonicalization + prune.** Renames: `pt_refined`→`pt` (the plain-PT
`pt` rung folded in, not kept as its own run), `pt_noah_lite`→`noah`; the two
SAC×LSTM ensembles promoted from `hybrid/` to top-level canonical
(`ens_feat_nl`→`noah_lstm_feat`, `ens_resid_nl`→`noah_lstm_resid`; scored on the
ensemble-mean flow). **Removed:** `pt_refined_ft` **and the entire ET/SWE
observation-loss infrastructure** (`loss.shape_pull_loss`/`level_hinge_loss`, the
`data.py` obs loaders, the `graphs.py`/`train.py` obs terms, the `config`/CLI
knobs). The obs work's findings are preserved in the track record below, but its
only consumer (`pt_refined_ft`) is retired, so the machinery went with it. Earlier
renames: `physical`→`hamon_dense`, `physical_grid_calsim`→`hamon`,
`physical_pt_calsim_refined`→`pt` (via `pt_refined`),
`noah_lite_pt_calsim`→`noah` (via `pt_noah_lite`).

## Track record (chronological)

### Feature/loss ablation — 15cdec fine-HRU domain, Hamon (2026-07-10/11)
- **`static`** — one-hot soil/veg statics, MSE. First working arm; proved the
  GA-prior init + bounded-sigmoid parameter net trains.
- **`static_nnse`** — NNSE loss. Better than MSE on the pooled objective →
  became the default.
- **`static_widebounds`** — widened parameter bounds. No win; GA bounds kept.
- **`static_adaptreg`** — Rahman-ALF adaptive per-basin weights. No win at
  the pooled optimum; off by default.
- **`climate_grouped`** — climate-statistics features + per-physics-group
  heads. Beaten by `physical`.
- **`physical`** (→ slot `hamon_dense`) — continuous soil/veg/terrain/LAI
  features. **Winner: val 0.840** (vs levers 0.836, smooth 0.838). Plain
  features beat every regularization lever. NHG (+23% bias) / FOL (α-damping) =
  structural ceiling. The original artifacts were lost 2026-07-15; **retrained
  from scratch 2026-07-14 under current defaults → cal 0.806 / val 0.840** (sel
  0.8014@ep34; val reproduces the recorded fine-HRU ceiling exactly, cal within
  plateau noise of the recorded 0.810). This val 0.840 is the fine-HRU ceiling
  the coarse-grid runs are measured against; `pt_refined_ft` recovers 0.837 of
  it on the 2074-cell grid.
- **`physical_levers`** — physical + spatial-reg/adaptive levers: 0.836,
  a wash. Was the hybrid track's default physics baseline until `--physics` was
  made required (2026-07-15); artifact then pruned (hybrids name a canonical
  export explicitly).
- **`physical_smooth`** — learned spatial smoother (gnn): 0.838, a wash.
- **`physical_seasonal`** — day-of-year harmonics on Kpet+recessions:
  seasonal recessions HURT; only seasonal Kpet helped (weakly).
- **`seasonal_kpet`** — seasonal Kpet, unbounded coeffs: diverged at lr 1e-3.
- **`seasonal_kpet_bounded`** — tanh-capped (±0.18): stable, a wash under
  flow-only training (foreshadowing: the DOF is unidentifiable from flow —
  see the 2026-07-15 obs-loss series where it becomes the key lever).

### Domain / footprint (2026-07-11/13)
- **`physical_grid`** — retrain on the native 1/16° grid (2074 cells, full
  footprint): val 0.834, −0.006 vs fine-HRU. Coarse grid recovers ~99% of
  skill; the residual is the lost orographic HRU-meteo downscaling (upstream
  CADWR product, not in repo).
- **`physical_grid_calsim`** (→ **`hamon`**) — CalSim3-footprint re-foot of
  11/15 basins (coarse grid over-reaches the true catchments +9–66%):
  skill-neutral (0.807/0.829), re-footed pbias better, YRS/MIL casualties.
  Footprint became standing method; recalibration folded in.
- **`hamon_grid_dynkpet`** — climate-state (wetness-index) dynamic Kpet.
  Stopped moot when the program pivoted to PT physics.

### Noah ET line (2026-07-12)
Diagnosis that launched it: Noah's deficit vs Hamon is a LEVEL deficit
present in calibration (flat cal→val), i.e. parameter identifiability — not
regime-shift extrapolation. Streamflow alone cannot identify a 7-param ET.
- **`noah_grid`** (v1) — full Jarvis canopy ET, uniform init: hurt dry
  basins (opposite-bias ET pattern, over-LAI 3.25 vs obs 1.3).
- **`noah_grid_v2`** — pinned observed veg_frac + seasonal LAI; 6 physiology
  params on a separate trunk.
- **`noah_grid_v3`** — best full Noah: **val 0.760** vs Hamon 0.834. The gap
  concentrated in 4 basins (NHG/SCC/TRM/BND = 55%).
- **`noah_grid_v4`–`v6`** — physics refinements: re-shuffled which basins
  fit, never closed the level gap (v5/v6: PT potential caused an SCC volume
  blowup).
- **`noah_lite_hamon`** — the minimal identifiable rebuild
  (AET = β(SM)^χ·PET, ONE learned DOF): the honest ablation floor on Hamon
  potential.
- **`noah_lite_pt`** — lite + PT potential (lifts the Kpet×Hamon ET ceiling).
- **`noah_lite_pt_calsim`** (→ **`pt_noah_lite`**) — + CalSim3 footprint; the
  canonical Noah. 1 DOF ≈ 7 DOF (v3) — confirming the non-identifiability
  diagnosis. Base model for the ET-obs screening. **Canonicalized to the frozen
  footing 2026-07-15**: a numba Noah-lite external-ET SAC core
  (`sacsma.sma_noah_lite`, the frozen mirror of the torch `canopy_lite` path —
  bit-exact vs torch `ninc_mode="dynamic"`, max |Δflow| 2e-13) lets it score
  through `run_basin` (`--et-scheme noah_lite` / `score_frozen`), the SAME
  full-footprint reference-SAC numerics as the Hamon/PT runs. **Frozen
  0.767/0.799** (torch was 0.759/0.792; the +0.008/+0.007 is the known
  full-vs-CalSim-footprint + variable-vs-fixed-`ninc` gap). `params_dpl.csv`
  now exported alongside `params_canopy.csv` (the canopy branch used to skip
  it). Both hybrid variants trained on this Noah-lite physics baseline
  (`--physics-et noah_lite`): residual 0.910/0.851, feature 0.912/0.851 — both
  match the pt_refined hybrids (~0.850), i.e. the LSTM erases the physics-baseline
  gap (local-only, gitignored; see the hybrid memory).
- **`pt_refined_noah_lite`** (2026-07-14, `testing/`, NOT promoted) — the two
  PT refinements (snow albedo 0.6 + arid dewpoint depression 2 °C) on the
  `pt_noah_lite` Noah-lite path. **A WASH in the mean (cal 0.759→0.763, val
  0.792→0.788) but a major basin RESHUFFLE** — same signature as the full-Noah
  v4–v6 refinements: it trades the flagship snow basins (BND −0.106, ORO/SHA/
  PNF/ISB −0.06 val) for the arid/transition basins (SCC +0.094, MIL +0.077,
  TRM +0.072, TLG +0.065, NHG +0.048 val); val |pbias| 10.0→9.1. Do-not-promote
  because it degrades Shasta + Bend Bridge. Mechanism: the refinements helped
  *plain-SAC* (`pt`→`pt_refined`) by reshaping the seasonal ET **cascade** that
  static `Kpet` cannot touch; on Noah-lite the learned `soil_chi` already
  absorbs the PET **level**, so lowering the potential only redistributes skill.
  The PT-refinement benefit is cascade-shape-specific, not universal.
- **`noah_lite_pt_calsim_et05`** — v1 ET obs loss (raw monthly central pull,
  λ=0.5, σ-floor 0.2): **degraded flow** 0.675→0.567 (baseline 0.759).
  Finding: the per-month pull pins LEVEL (products disagree 38–85%) as hard
  as PHASE (agree ~0.5 mo). Led to the v2 shape/level/SWE decomposition.

### Priestley–Taylor plain-SAC line (2026-07-13/14)
Rationale: energy-based PET for warming robustness (Wi et al. 2024) without
Noah's identifiability problem — PT drives the FROZEN SAC ET cascade.
- **`physical_pt_calsim`** (→ **`pt`**) — plain PT: 0.791/0.823. PT's summer
  +27% ET vs Hamon; Kpet re-optimizes level, cannot reshape season.
- **`physical_pt_calsim_refined`** (→ **`pt_refined`**) — + snow-cover albedo
  (PET collapses under the model's own Snow-17 pack) + arid dewpoint
  depression (both one-directional, Kpet-unabsorbable shape corrections):
  **0.799/0.826**, sel 0.7954@ep42. The baseline/donor for the obs work.
  Hamon reference on the same footing: 0.807/0.829.

### ET/SWE observation series — on `pt_refined` (2026-07-14/15) — RETIRED 2026-07-15
> The obs-loss code and the one run that used it (`pt_refined_ft`) were removed on
> 2026-07-15 (see the canonicalization note above). Findings kept here as the record.

Loss design v2.1: ET seasonal-SHAPE pull (level-blind, 1-ensemble-σ deadband,
Huber k=3) + ET level envelope + SWE shape pull (4 products, snow basins
auto-masked), λ=0.2 each, cal-window only, never a selection metric.
- **`physical_pt_refined_et_diverged`** — v2 as-built: NaN divergence
  (f32 backward overflow through the 366-day recurrence → inf×0 at a branch
  gate). Fix: Huber on both obs losses.
- **`physical_pt_refined_et_huber_nodeadband`** — Huber alone: identical
  peak-then-decay (0.748@ep12) — objective conflict burning the shared
  grad-clip budget. Fix: the deadband (zero force within product spread).
- **`physical_pt_refined_et`** — deadband+Huber, +ET only: 0.766/0.797.
  ET-shape RMS ~halved domain-wide (except NHG/ISB, worse).
- **`physical_pt_refined_etswe`** — +SWE: 0.767/0.811. **SWE strictly
  dominates +ET** (regularizer; peak month already matched at baseline).
  Key diagnosis: the "level-blind" shape pull LEAKS level ±100 mm/yr through
  the storage nonlinearity — that leak IS the flow cost. Water-balance
  closure check: P−Q closes at all 15 basins; the product bracket spans up
  to 72% of Q in arid basins (products cannot inform level; flow pins it
  ~10× tighter) — products = SHAPE only.
- **`physical_pt_refined_etswe_skpet`** — + seasonal Kpet (joint, from
  scratch): fast peak sel 0.767@ep8 (frozen 0.771/0.808, **|val pbias| 6.1
  vs 9.0 baseline** — seasonal ET timing fixes volume bias static Kpet
  can't), then Adam-vs-tanh saturation instability (all loss terms worse
  ep8→ep14, harmonics pinned at the ±0.25 joint cap, annual ET drifting
  10–17% inside the product bracket). Stopped ep14. Casualties NHG/TRM/YRS.
- **`pt_refined_ft`** (ex physical_pt_refined_etswe_skpet_ft) — PROMOTED
  CANONICAL: warm-start from `pt_refined` best.pt (`--init-from`,
  exact-equivalence verified), lr 2e-4, same obs λs, seasonal Kpet, and the
  level hinge re-targeted to the WATER-BALANCE anchor P−Q_obs ±15%
  (`--et-anchor-band`; replaces the too-wide product envelope; no
  basin-specific masking anywhere). ep0 sel = donor's 0.7954 exactly; best
  sel **0.8056@ep16** (early stop ep30). **Frozen 0.810/0.837 — beats the
  baseline (+0.011/+0.011), beats the Hamon anchor (0.807/0.829), matches
  the fine-HRU `hamon_dense` (0.810/0.840) on the coarse grid.** 12/15
  basins improved val; |val pbias| 9.0→7.3; NHG fixed generically by the
  anchor (val 0.799→0.826, pbias +6.8→+0.1 — no mask needed); ET-shape RMS
  deadband-z 0.639→0.411 (−36%) with NO basin degraded (the joint arms'
  NHG/ISB shape casualties are absent: ISB 0.418→0.201). Residual losers:
  PNF −0.034 / YRS −0.020 / SCC −0.018 val (positive-bias basins pushed
  higher). The obs information, given a flow-cheap knob (seasonal Kpet),
  a tight observed level constraint (P−Q), and the fine-tune regime
  (select within the flow-optimal plateau), IMPROVES flow rather than
  trading against it.

## Open items
- (none active). Canonical set is complete: `hamon_dense`, `hamon`, `pt`, `noah`,
  and the two SAC×LSTM ensembles `noah_lstm_feat` (0.923/0.869) /
  `noah_lstm_resid` (0.926/0.873) — the LSTM erases the physics-baseline gap
  (`noah` physics 0.767/0.799 → ~0.87 val hybrid). `pt_refined_noah_lite`
  resolved 2026-07-14 (wash + reshuffle, not promoted; see the Noah ET line).
