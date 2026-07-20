# dPL run track record

## Layout

Canonical runs live directly under `artifacts/dpl/<label>`. The frozen-physics runs
(`hamon_dense`, `hamon`, `pt`, `noah`, and the climate-adaptive `noah_ca`) each
carry `checkpoints/best.pt`, `train_log.csv`, `params_dpl.csv`, and
`metrics_<label>.csv` (plus `params_canopy.csv` for `noah`/`noah_ca`, and the
present-climate sim channel `frozen_sim_noah_ca.csv` for `noah_ca`).

`noah` also carries two torch daily dumps (date × basin, mm/day):
`daily_sim_noah_torch.csv`, the hybrids' sim channel, and
`daily_sim_noah_plus2C.csv` (with `metrics_noah_plus2C.csv`), the +2 °C teacher for
the ΔT-consistency loss. Channel and teacher come from one pipeline (torch
numerics), so the training-time response delta is numerics-consistent.

The `hybrid` (basic feature coupling on the `noah` torch channel, no day-of-year
inputs) and `hybrid_pet_dt` (adds the raw PT-potential input and a single +2 °C
ΔT-consistency loss) ensembles established the skill step and the +2 °C response
result. They are now the +2 °C-only predecessors of the `noah_ca` hybrid family
(`hybrid_base` / `hybrid_dtdp` / `lstm`; see the Phase-2 section below), which
rebuilds them on climate-adaptive physics and generalizes the single ΔT anchor to
the full (Δp, ΔT) response surface, so `hybrid_dtdp` supersedes `hybrid_pet_dt`
for climate work. Each ensemble holds `seed*/checkpoints/best.pt` and
per-seed `metrics_hybrid.csv`, plus a top-level `metrics_hybrid.csv` scoring the
ensemble-mean flow (`hybrid.evaluate.score_ensemble`). The residual coupling and
the first `noah`-based ensembles were retired 2026-07-16, and `noah_ft` (the
seasonal-melt fine-tune, canonical 2026-07-16→17) was demoted 2026-07-17 after a
new-basis head-to-head — see the track record; git history and the gitignored
`testing/noah_ft_region` hold the record.

Superseded run artifacts were pruned, but their findings stay in the track record
below (the names there no longer resolve to on-disk runs). `--physics` is required
with no default, so `testing/` holds only gitignored local scratch, and `fidelity/`
is the numerics benchmark, not a run. All skill numbers are pooled 15-basin mean
KGE under frozen-model scoring (numba; PT via `sacsma.pet_pt`, Noah-lite via
`sacsma.sma_noah_lite`) unless marked *(torch)* — the seasonal-melt fine-tunes and
the full 7-param Noah ET (`noah_grid*`) are torch-only, with no frozen core.

Standing methods shared by every canonical grid run: the `physical` feature
variant; pooled 15cdec training (daily gage FNF, cal WY1989–2003 / val WY2004–2018);
CalSim3-footprint aggregation (`--calsim-footprint`); an NNSE+log loss with a
variance-matching term; cal-KGE selection every 2 epochs; and a truncated no-grad
spinup from 1978-10-01.

## Canonical lineage

| label | domain | delta vs predecessor | cal/val KGE |
|---|---|---|---|
| `hamon_dense` | 15cdec (7891 HRU) | — (the original dPL) | **0.806/0.840** (retrained 2026-07-14) |
| `hamon` | 15cdec_grid (2074 cells) | native-grid retrain + CalSim3 footprint | 0.807/0.829 |
| `pt` | 15cdec_grid | Priestley–Taylor PET (Bristow–Campbell Rn) + snow-cover albedo (0.6) + arid dewpoint depression (2 °C) | 0.799/0.826 |
| `noah` | 15cdec_grid | Noah-lite canopy ET (1 learned DOF `soil_chi`) on PT potential | 0.767/0.799 |
| `noah_ca` | 15cdec_grid | `noah` retrained on `physical_climate` features (physiographic + 4 climate indices) → climate-ADAPTIVE physics: parameters recompute under perturbed climate. The physics basis for the current hybrid family (2026-07-19) | 0.779/0.804 |
| ~~`noah_ft`~~ | 15cdec_grid | DEMOTED 2026-07-17 — the seasonal-melt fine-tune of `noah` (0.765/0.799 torch): the new-basis head-to-head is a pooled wash vs `noah` with NHG + north-state-volume casualties, and its torch-only scoring taxed every consumer | — |
| `hybrid` | 15cdec_grid | SAC×LSTM feature coupling on `noah` physics (torch daily sim-cache channel), no doy inputs — 8-seed ensemble mean. The BASIC hybrid: the skill step; its unconstrained +2 °C response is untrustworthy (see 2026-07-17) | 0.917/0.869 |
| `hybrid_pet_dt` | 15cdec_grid | `hybrid` + the raw PT-potential input channel (`--pet-input`) + the temperature-consistency loss λ=0.3 (+2 °C `noah` torch teacher) — 8-seed ensemble mean. Same skill, physics-consistent climate response (+2 °C resp ratio 1.04, regime r 0.97). SUPERSEDED for climate work by `hybrid_dtdp` (2026-07-19) | 0.916/0.864 |
| `hybrid_base` | 15cdec_grid | SAC×LSTM feature coupling on `noah_ca` physics (`--pet-input --statics`, no doy), NO response loss — 3-seed ensemble mean. Best skill; +2 °C response over-strong (ratio 1.50) | 0.922/0.877 |
| `hybrid_dtdp` | 15cdec_grid | `hybrid_base` + the 14-anchor {−20,−10,0,+10,+20}%×{0,+2,+4 °C} (Δp, ΔT) response-consistency loss (λ0.18) vs the `noah_ca` adaptive teachers — 3-seed mean. THE climate-trustworthy model: tracks physics on both axes (+3 °C ratio 1.14, 15/15 signs); generalizes `hybrid_pet_dt` to the full surface | **0.873/0.849** |
| `lstm` | 15cdec_grid | pure data-driven control (`use_sim=False` — no SAC-SMA sim channel), same climate-adaptive statics — 3-seed mean. Good skill but physically nonsensical projection (+3 °C ratio −0.94, wrong-signed) | 0.909/0.835 |
| ~~`noah_lstm_feat`~~ | 15cdec_grid | RETIRED 2026-07-16 — feature hybrid on `noah` (5 seeds, 0.923/0.869); superseded by `hybrid` | — |
| ~~`noah_lstm_resid`~~ | 15cdec_grid | RETIRED 2026-07-16 — residual hybrid on `noah` (8 seeds, 0.926/0.873); the residual COUPLING was dropped entirely (regime-conditional volume injection, B1–B3) | — |

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

## Noah-line seasonal-timing program (2026-07-15/16, CONCLUDED — A1b promoted)

Why: the climatology shows noah→LSTM *hurts* NML/MRC/ORO (the basins where noah
already matches CalSim3 FNF, monthly KGE 0.90–0.94) while halving the val
seasonal mismatch everywhere else (0.092→~0.04). Diagnosis: ~37% of the LSTM
residual correction is a fixed winter→spring MELT shift (ORO −0.34 mm/d Feb /
+0.39 May), ~63% interannual; the damage at the good basins is val-period
VOLUME bias (NML β 0.954→0.901) injected by a calendar-keyed mean correction —
`sin/cos_doy` are LSTM inputs and nothing constrains the residual's long-run
mean. Program: obs-steered physics fine-tune on noah (the `pt_refined_ft`
recipe, resurrected) + hybrid application fixes (`--no-doy`,
`--resid-mean-lambda`). Scoreboard = `dpl.seasonal_compare.seasonal_physics_report`
(daily-gage val KGE decomposition, val seasonal mismatch, monthly-vs-CalSim3
KGE, correction seas_frac).

- **Obs-loss infra RESURRECTED** (from `2ff8076`, reverse of the `cdf3018`
  prune) into the seasonal-snow working tree: `loss.shape_pull_loss` /
  `level_hinge_loss`, the `data.py` obs loaders + P−Q anchor, config/CLI λs.
  Gates: λ=0 byte-identity vs the pre-merge run (loss + cal_kge to the last
  digit); graph==eager with ET+SWE+anchor+4 seasonal params on (2.6e-5 rel,
  selection bit-identical, 0 skips — first-ever run of swe-capture × seasonal
  snow). New: `--et-products` (single-product steering; one product requires
  the P−Q anchor; σ falls back to interannual spread + floor), and an
  **ep0-donor gate** in train.py (warm-start ep0 selection must reproduce the
  donor's sel cal-KGE).
- **⚠ TRUNCATED SPINUP IS UNSAFE FOR TRAINED dPL FIELDS.** The ep0 gate caught
  the canonical noah donor evaluating at 0.6552 under the 1978-10-01 truncated
  spinup vs its 0.7594 selection (full-spinup era); `--spinup-start 1915-01-02`
  reproduces 0.7594 EXACTLY. Cause: the learned field carries >10-yr state
  memory (lzfsm ≈ 4000 mm fills at ~1–2 mm/day) — the truncation was
  parity-verified on GA params only. Consequences: (1) every warm-start /
  fine-tune MUST run full spinup; (2) selection scores are NOT comparable
  across spinup bases (the seas_kpet control's 0.728 sel vs 0.754 scored gap
  is partly this); (3) the earlier "fresh-Adam transient wrecks warm-starts —
  from-scratch required" reading was WRONG — the warm start was exact all
  along, the evaluation basis was broken.
- **`testing/noah_seas_kpet`** — CONTROL: flow-only seasonal Kpet, from
  scratch (n_inc 5, 40 ep, truncated spinup). Torch-scored **0.754/0.790** vs
  noah-torch 0.759/0.792; val seasonal mismatch **0.097 vs noah 0.092**
  (UNCHANGED), CalSim3 KGE 0.848 vs 0.861. Confirms the retired program's
  conclusion on noah: flow alone cannot use the seasonal DOF — the obs signal
  identifies it (`seasonal_compare_seas_kpet_flowonly.csv`).
- **A2.0 pre-registered single-product pick: `fluxcom`** — minimizes RMS
  |annual product ET − (P−Q_obs)| over the 15 basins (61.5 mm/yr; next
  terraclimate 85.8, fldas 98.2, era5land 136, gleam 149; smallest mean bias
  +31; closest at 6/15 basins). The flow-consistent criterion lands on the
  same product the model's own ET level sits on (model ≈ FLUXCOM in the San
  Joaquin) — "closest to the model" and "closest to P−Q" agree because the
  model's level is flow-pinned (scratchpad a20_product_pick.py).
- **`testing/noah_ft_kpet`** (A1a) — the pt_refined_ft recipe VERBATIM on noah
  (`--init-from noah/best.pt --lr 2e-4 --patience 6 --seasonal Kpet` + λ
  0.2/0.2/0.2 + P−Q anchor ±15%, n_inc 10, FULL spinup; ep0 gate = 0.7594
  exact). **DOES NOT TRANSFER: selection never exceeded the donor** (early
  stop ep14; best.pt = the donor field). The final obs-shaped ckpt
  (final_ckpt/, scored for diagnosis only) is flow-neutral (0.757/0.794 vs
  noah-torch 0.759/0.792) with val seasonal mismatch UNCHANGED (0.091 vs
  0.092) — even though the net swung Kpet ±21% seasonally (amp med 0.18 on
  base 0.83). **Mechanism = the pt_refined_noah_lite lesson**: Noah-lite AET
  = β(SM)^χ·Kpet·PET — summer is water-limited (Kpet inert) and winter PET is
  tiny, so the seasonal-ET knob that reshaped the PT cascade is structurally
  damped before it reaches the hydrograph. Seasonal-ET timing is NOT a lever
  on the noah scheme.
- **`testing/noah_ft_snow`** (A1b) — A1a + `--seasonal Kpet,MFMAX,MFMIN,MBASE`
  (`--seasonal-amp-frac 0.10`). **D1 WINNER — the melt DOF works where the ET
  DOF was damped**: sel 0.7594→**0.7656@ep44** (the only arm to beat the
  donor; early stop ep58), torch-scored **0.765/0.799 vs noah 0.759/0.792**.
  Val seasonal mismatch 0.092→**0.085**; the program's target basins fixed
  WITHOUT the LSTM's volume injection: NML CalSim3 KGE 0.924→**0.939**
  (β 0.958), MRC 0.940→**0.970** (β 0.989) — the LSTM had dragged both below
  ~0.91 at β≈0.90. Also TLG 0.839→0.862, MIL 0.746→0.785, YRS 0.870→0.891.
  **The learned harmonics ARE the diagnosed winter→spring shift, in physics:**
  MBASE +~0.4°C in mid-winter (suppresses warm-spell melt), MFMAX amplitude
  added IN PHASE with Snow-17's Jun-21 sinusoid (stronger winter/spring melt
  contrast → later melt), Kpet peaking late-Jan (+0.2 on 0.79 — fills the
  known winter ET deficit, trims winter runoff). Costs: NHG CalSim3 0.751→
  0.647 (snow-free ⇒ its melt DOF is unconstrained by the SWE loss; β 1.075→
  0.930) and SHA/BND/ORO −0.03 via β −0.04 (the seasonal-Kpet level leak the
  ±15% anchor band tolerates — a tighter band is the candidate fix). Gap to
  the LSTM remains large (seas_mis 0.085 vs 0.043): the fixed harmonic can
  only address the ~37% climatological share, and captures ~1/4 of it.
- **`testing/noah_ft_1prod`** (A2) — A1b recipe + `--et-products fluxcom`
  (the pre-registered P−Q-closest product; single-product σ = interannual +
  0.1 floor, envelope replaced by the P−Q anchor). **LOSES the D2
  head-to-head: selection never exceeded the donor** (early stop ep14 like
  A1a; no instability — obs loss descended 0.77→0.50 cleanly) even though it
  carried the same melt DOF that lifted A1b to 0.7656. Final-ckpt diagnosis:
  0.757/0.797, seas_mis 0.084 (≈A1b — the melt DOF does the seasonal work in
  both). Verdict: the stricter single-product ET pull (σ at the floor,
  ~2.6× the init obs loss) BURNS the gradient budget that the consensus arm
  spent improving flow — consensus + P−Q anchor is the right use of the
  products, closing the single-product question. First launch was an A1b
  duplicate (the `--et-products` kwarg wasn't threaded into DplConfig in
  `_dpl_train` — argparse silently dropped it; fixed, and the launch-line
  product printout is now part of the gate check). **D2 WINNER: A1b
  (`noah_ft_snow`).** Seasonal arms are TORCH-scored — compare vs noah-torch
  0.759/0.792, never the frozen 0.767/0.799.
- **`testing/noah_lstm_resid_nodoy`** (B1) — residual hybrid WITHOUT the
  sin/cos day-of-year inputs (`--no-doy`, 5 dyn channels), seeds 0–2,
  canonical cfg, judged 3-member-mean vs canonical seeds 0–2 (never 3v8;
  `compare_3v3.csv`). **Doy is redundant but NOT causal**: pooled val 0.871 =
  0.871 (cal 0.923 vs 0.920 — zero skill cost), yet the val volume injection
  is UNCHANGED (mean |val β−1| 0.064 = 0.064; NML β 0.913 vs 0.914, MRC 0.948
  vs 0.964) — the LSTM reconstructs the same seasonal mean correction from
  tavg/sim. Falsifies the strong doy hypothesis; the bias lives in the
  residual's unconstrained MEAN → B2 is the live fix. D3: no 8-seed extension
  on B1 alone.
- **`testing/noah_lstm_resid_volpen_l{0.1,0.3,1.0}`** (B2) —
  `--resid-mean-lambda` screen on seed 0 (penalty = λ·mean_b(per-batch
  per-basin mean of the normalized residual)², basins ≥8 samples/batch).
  **INERT ON THE TARGET, with a clean mechanism**: sel cal 0.920/0.918/0.905
  (vs 0.923 plain) but seed-0 NML val β only 0.905→0.908/0.907/0.916 and
  mean |val β−1| flat 0.067→0.066/0.065/0.070 (λ=1.0 costs pooled val
  0.868→0.860 for +0.011 NML β). The penalty is satisfied ON THE CAL
  DISTRIBUTION — the val volume bias is a REGIME-CONDITIONAL correction that
  averages ~0 over cal but not over the shifted WY2004-18 climate; no
  cal-window penalty (zero- or cal-mean-anchored) can constrain it.
  **Track B conclusion: neither doy removal nor mean-penalties fix the val
  volume injection — the fix is to SHRINK the residual's job (better physics
  → B3). No 8-seed extension for B1/B2.**
- **`testing/noah_lstm_resid_ft`** (B3) — 8-seed residual ensemble on the
  **A1b physics** (`--physics GA --sim-cache
  testing/noah_ft_snow/daily_sim_noah_ft_snow.csv` — the torch daily dump IS
  the sim channel via the cache short-circuit; run_basin never executes; sim
  provenance = torch numerics, full 1915–2018). **HYPOTHESIS REFUTED**:
  ensemble-mean 0.931/0.870 vs canonical 0.926/0.873; CalSim3 0.877 vs 0.886;
  and the target-basin val β got WORSE (NML 0.891 vs 0.901, MRC 0.929 vs
  0.946) even though the A1b physics underneath has healthy β (0.958/0.989).
  The LSTM correction dominates whatever physics it sits on and re-injects
  its regime-conditional bias — the mirror of "the LSTM erases the physics
  gap": it erases physics IMPROVEMENTS too.
- **A3 (`--dynamic-params Kpet`) SKIPPED on evidence**: both A1a (seasonal)
  and the flow-only control show the Kpet channel is β(SM)-damped on noah —
  a climate-state Kpet routes through the same dead multiplier.
- **PROGRAM CONCLUSION (2026-07-16; combined scoreboard =
  `testing/noah_seasonal_program_scoreboard.csv`)**: (1) the ~37%
  climatological share of the LSTM's seasonal correction is partially
  absorbable in physics via MELT-timing DOF identified by SWE-shape obs
  (A1b: seas_mis 0.092→0.085, NML/MRC CalSim3 0.939/0.970 — ABOVE the
  hybrids' 0.78-0.87 at those basins — with healthy volume); ET-side seasonal
  levers are dead on the noah scheme. (2) The hybrids' val volume bias at the
  already-good basins is INTRINSIC to cal-only residual learning under
  climate shift — not fixable by doy removal (B1), mean penalties (B2), or a
  better physics baseline (B3). Practical reading: the LSTM ensembles remain
  the pooled skill ceiling (val 0.873), but AT the basins where the physics
  already matches CalSim3 (NML/MRC/ORO) the physics is the more trustworthy
  out-of-sample answer, and A1b widens exactly that margin. A1b promotion
  trade-offs if considered: torch-only scoring (seasonal params have no
  frozen core), NHG regression (snow-free ⇒ melt DOF unconstrained —
  candidate fix: weight melt harmonics by SWE participation), small
  SHA/BND/ORO volume drift (candidate fix: tighter `--et-anchor-band`).
- **PROMOTED 2026-07-16**: A1b adopted as canonical **`noah_ft`**
  (`testing/noah_ft_snow` → `artifacts/dpl/noah_ft`; re-evaluated in place,
  reproduces 0.765/0.799 torch and sel 0.7656 exactly). The trade-offs above
  were accepted at adoption; NHG melt-DOF weighting and a tighter anchor band
  stay open as refinement candidates.

## Canonical rebuild on noah_ft (2026-07-16)

> Historical record — `noah_ft` was demoted and the ensembles rebuilt on the
> `noah` torch channel the next day (next section). The design work below
> (residual prune, ΔT-consistency loss, PET input, the D2/D3 screens) all
> carries over unchanged; only the physics tier under the hybrids changed.

User decisions: adopt A1b as `noah_ft`; **drop the residual coupling
entirely** (full prune: code + `noah_lstm_resid` artifacts; git history is the
archive); the feature hybrid becomes THE `hybrid` — retrained on the `noah_ft`
sim channel (via `--sim-cache daily_sim_noah_ft.csv`), **no doy inputs**; and
add a **temperature-consistency loss** anchoring the hybrid's warming response
to the physics.

- **Residual prune + rename**: `hybrid/` package is feature-only
  (`HybridLSTM` Softplus head always; `variant`/`--resid-mean-lambda`
  removed; metrics file → `metrics_hybrid.csv`); climatology panel e →
  noah → noah_ft → Hybrid (noah_ft ingested from its daily dump via the new
  `TORCH_SIM` route); forcing-sensitivity → hamon/pt/noah/noah_ft + Hybrid
  (noah_ft detrended run streams the torch pipeline under the per-cell WGEN
  dT field — `evaluate.noah_torch_daily`, new); `seasonal_compare` references
  → noah / noah_ft / hybrid. **λ=0 gate PASSED**: 2-epoch feature run,
  pre-prune code (23f37d5 worktree) vs pruned code — `train_log.csv` loss +
  cal_kge byte-identical.
- **Temperature-consistency loss (design)**: teacher = `noah_ft` re-run with
  tavg/tmin/tmax + 2 °C (`sacsma dpl evaluate --temp-delta 2.0` →
  `daily_sim_noah_ft_plus2C.csv`); per batch the LSTM is forwarded a second
  time on the perturbed feature copy (temp channels +ΔT/σ in normalized
  space, sim channel = teacher sim/scale — the `_hybrid_flow` recipe at train
  time, same input-noise draw on both copies) and the DAILY response
  `pred_dt − pred` is pulled to the physics response `(sim_dt − sim)/scale`
  by MSE × `--temp-lambda`. Hypothesis: the val period is warmer than cal;
  the hybrids' regime-conditional val volume bias (B1–B3) is the LSTM's own
  temperature response extrapolating — anchoring dQ/dT to physics attacks it
  at the root. The WGEN detrending pattern is HELD OUT of training — the
  forcing-sensitivity figure becomes the independent check.
- **`testing/hybrid_base`** (8 seeds): the λ_T=0 baseline — feature, no-doy,
  statics, h64/dropout .35/noise .2, sim-cache = `daily_sim_noah_ft.csv`.
  Ensemble-mean **0.917/0.865** — matches the retired `noah`-based ensembles'
  pooled skill (feat 0.869 / resid 0.873 val): nothing lost moving to the
  noah_ft channel + dropping doy. The fallback canonical.
- **D2 temp-λ screen (λ ∈ {0.1, 0.3, 1.0} × seeds 0-2, judged 3v3 vs base
  seeds 0-2; `testing/d2_screen_compare.csv`)**:
  | group | cal/val KGE | mean \|val β−1\| | +2 °C resp ratio | regime r |
  |---|---|---|---|---|
  | base | 0.917/0.857 | 0.078 | **0.15** | 0.88 |
  | λ=0.1 | 0.919/0.863 | 0.076 | 0.36 | 0.96 |
  | **λ=0.3** | 0.914/0.860 | 0.077 | **0.78** | 0.97 |
  | λ=1.0 | 0.897/0.852 | 0.088 | 0.92 | 0.99 |
  (resp ratio = Σ hybrid ΔQ / Σ physics ΔQ under +2 °C, 3-member mean, all
  full-lookback days; regime r = monthly-regime correlation of the two ΔQs.)
  Findings: (1) the UNCONSTRAINED hybrid has almost no warming response
  (15% of physics; ORO −1.29 = wrong SIGN) — disqualifying for a climate
  application; (2) the loss dials the response in cleanly — λ=0.3 recovers
  78% of the physics response (r .97) at ZERO pooled-skill cost, λ=1.0 buys
  92% but costs cal 0.917→0.897; (3) the NML/MRC/ORO val-β hypothesis is
  REFUTED — anchoring dQ/dT leaves val β unchanged (NML 0.905→0.897): the
  regime-conditional bias is intrinsic to cal-only training (B1–B3 stands);
  the temp loss buys a physics-consistent CLIMATE RESPONSE, not a val-bias
  fix. **Winner: λ=0.3** → D3 (8 seeds, tl0.3 seeds 0-2 reused).
- **D3 → PROMOTED as canonical `hybrid`**: 8-seed λ=0.3 ensemble-mean
  **0.912/0.861** vs the λ=0 baseline's 0.917/0.865 — Δval −0.004, inside
  the ~0.005 gate, in exchange for the anchored +2 °C response (ratio 0.78,
  regime r 0.97; the unconstrained baseline delivers 0.15 with ORO
  wrong-signed). `testing/hybrid_tl_final` → `artifacts/dpl/hybrid`;
  `hybrid_base` kept in testing/ as the recorded λ=0 fallback. The WGEN
  detrending pattern was HELD OUT of training — the regenerated
  forcing-sensitivity figure is the independent temperature-response check.
  **Held-out WGEN verdict**: the monthly-regime response generalizes fully
  (Hybrid sits on the physics family: +300 TAF/mo winter gain, −380 June
  drawdown); the annual-volume response generalizes PARTIALLY (~⅓–½ of
  noah_ft's early-record signal — the summer-loss side under-delivers).
  Scoreboard (`hybrid/seasonal_compare_hybrid.csv`): the temp loss does NOT
  cost the hybrid its timing advantage — val seas_mis 0.047 vs noah_ft 0.085
  / noah 0.092; CalSim3 monthly KGE 0.867 (best of the three).
- **PET-input arm (`testing/hybrid_pet`, 3 seeds, λ=0; user-directed)**: the
  raw PT potential (basin-average, alb 0/dew 0 — exactly the noah_ft energy
  demand, recomputed from forcing via `hybrid.data.basin_pet_pt`, opt-in
  `--pet-input`, cached `basin_pet_pt_<domain>.csv`) added as a 6th dynamic
  channel alongside the temps. Judged 3v3 (`testing/pet_screen_compare.csv`):
  **PET is a SKILL lever, not a RESPONSE lever** — best pooled val of any
  group (0.921/**0.870** vs base 0.857 / canon 0.860) and best |val β−1|
  (0.074), but +2 °C resp ratio **0.19** ≈ the unconstrained 0.15 (ORO/NHG/
  SCC/NML still wrong-signed). Same "redundant, not causal" pattern as doy:
  the LSTM uses physics-shaped INPUTS as information; only the training-time
  anchor moves dQ/dT. PET under any ΔT/forcing recomputes EXACTLY (a
  deterministic function of T), so the perturbed training copy and the WGEN
  counterfactual are clean for pet-input checkpoints.
- **PET + temp-loss composition (`testing/hybrid_pet_tl`, 3 seeds, λ=0.3,
  +2 °C teacher)**: they TRADE, not stack — **strongest response of any arm
  (ratio 0.89, regime r 0.97; wrong-signed basins eliminated: ORO +0.17,
  NHG +0.15)** but the PET skill bonus is spent doing it (val 0.870→0.860 =
  canonical's level). Passes the promotion gate (val = canonical, response >
  canonical 0.78) — held pending the λ=0.1 screen below.
- **PET + λ=0.1 (`testing/hybrid_pet_tl0.1`, 3 seeds; resumed + judged
  2026-07-16)**: the light-anchor corner does NOT dominate — 0.920/0.863
  (+0.003 val over λ=0.3) but response ratio **0.70** < the no-PET canonical's
  0.78, regime r 0.94, and SHA/BND/ORO/NHG back at ~zero/wrong-signed. The
  skill-vs-response frontier is pet(0.870, 0.19) → pet_tl0.1(0.863, 0.70) →
  pet_tl(0.860, 0.89); the no-PET canonical (0.860, 0.78) is STRICTLY
  DOMINATED by pet_tl. **5-way winner: PET + λ=0.3** (equal skill, strongest
  + cleanest response).
- **FINAL PROMOTION (2026-07-16, user-directed naming + set)**: the winner ×8
  = **`hybrid_pet_dt`** (0.917/**0.864** ensemble-mean — skill-neutral vs the
  basic hybrid at 8 seeds, +2 °C response 0.89) and the λ=0 no-PET baseline
  `hybrid_base` ×8 promoted as **`hybrid`** (0.917/0.865, response 0.15) so
  the canonical set SHOWS the improvement: `hybrid` (the LSTM skill step) →
  `hybrid_pet_dt` (same skill, physics-consistent climate response). The
  intermediate no-PET λ=0.3 ensemble (0.912/0.861, response 0.78) is RETIRED
  — dominated by `hybrid_pet_dt` on every axis (git history: 55c4e3c).
  "dt" = the ΔT-consistency loss (formerly "tl"/temp-loss in the screens).
  Figures: climatology panel e = noah → noah_ft → Hybrid → Hybrid PET+dT;
  forcing-sensitivity carries BOTH ensembles (the flat-response basic Hybrid
  vs the physics-tracking PET+dT is the improvement exhibit).

## Region-basis rebuild + noah_ft demotion (2026-07-17)

Basis change: the unified region stores became the training basis (forcing
`33c62d8` — ×10-artifact-corrected Livneh-unsplit at the 4410-cell region
grid; obs `437c0f2`/`68248f1` — the GEE spec-v2 ET/SWE store + the
openet/modis referees; `dpl/data.py` ET/SWE defaults flipped `0f1bf8a`). At
the consumed level (15-basin monthly climatologies) the obs drift vs the
frozen legacy snapshot is small — ET rel RMS 1.1%, SWE 4.1%, snowy mask
unchanged 14/15 — the per-cell ERA5-Land drift damps out in the
multi-product basin means.

- **`testing/noah_ft_region`** — the A1b recipe re-run verbatim on the
  region basis (full spinup, ep0 gate). ep0 donor eval 0.75934 vs the
  historical 0.75941 (−7.3e-5 = the forcing ×10 fix reaching pre-1988
  spinup state only); best sel 0.7655@ep44, early stop ep58 — REPRODUCES
  the old canonical exactly (final metrics identical at 3 dp, 0.765/0.799
  torch). The region store is a training drop-in; nothing depended on the
  irreproducible legacy GEE snapshot.
- **Head-to-head (new basis, torch-vs-torch;
  `seasonal_physics_report` on both arms)**: pooled val KGE noah 0.7923 /
  noah_ft 0.7992 — a dead tie with the frozen-noah 0.799 the lineage table
  already carried; val seas_mis 0.0960/0.0851; CalSim3 monthly
  0.8562/0.8537 (wash); pooled val β 0.9716/0.9637. noah_ft wins the
  southern Sierra (MIL +0.044, SCC +0.046, ISB +0.035, TRM +0.026) and
  NML/MRC CalSim3 (0.930→0.940, 0.954→0.970), but loses NHG outright (val
  0.560→0.511, CalSim3 0.713→0.646) and bleeds north-state volume
  (SHA/BND/ORO/FOL β −0.02…−0.03). **USER DECISION: demote noah_ft.**
  `noah` is the canonical physics tier + ΔT teacher; the torch-only
  consumption tax (no frozen core for seasonal melt params — every consumer
  needed the daily-dump side channel) retired with it.
- **Ensemble rebuild on the `noah` torch channel**
  (`daily_sim_noah_torch.csv` sim cache + `daily_sim_noah_plus2C.csv`
  teacher — one pipeline for channel and teacher): `hybrid` ×8 →
  ensemble-mean **0.917/0.869** (vs 0.917/0.865 on the noah_ft channel —
  the basic hybrid GAINS +0.004 val); `hybrid_pet_dt` ×8 → **0.916/0.864**
  (vs 0.917/0.864 — identical). Per-seed val spread tightened (σ 0.003 vs
  0.008). The LSTM again erases the physics-tier difference — consistent
  with every earlier channel swap.
- **+2 °C response re-verified on the promoted pair**
  (`scratchpad/resp_ratio_promoted.py`, D2 convention): the basic `hybrid`
  is now WRONG-SIGNED — resp ratio **−0.57** (it *adds* annual flow under
  +2 °C where physics removes it; ORO −1.96, FOL −1.22, YRS −1.12; member
  spread +0.44…−1.02) even though its regime shape correlates (r 0.95).
  The old basic measured 0.15 — the unconstrained response is a lottery
  across retrains, and this draw landed wrong-signed, which SHARPENS the
  pair's story: without the ΔT anchor the hybrid's climate response is
  unusable. `hybrid_pet_dt`: resp ratio **1.04** (regime r 0.97; members
  0.74–1.42; residual per-basin scatter — MKM 2.7 overshoot, SHA/SCC/NHG
  ~zero — but every large-magnitude basin correct-signed) — the anchor
  survives the channel swap and now matches physics in the pooled sum
  (old channel: 0.89).
- **Progression exhibit (`hybrid_progression.png`/`.csv`; user-requested)**:
  the missing middle rung `testing/hybrid_pet_noah` (PET input, NO ΔT loss)
  trained ×8 on the same channel: ensemble-mean 0.921/**0.872** — the best
  val of the three arms (PET = skill lever, reconfirmed on the new basis) —
  but +2 °C resp ratio **0.24** (regime r 0.96): the response stays ~flat
  without the training-time anchor. The three-arm progression
  (0.869/−0.57 → 0.872/0.24 → 0.864/1.04) isolates the two levers: the PET
  INPUT buys ~+0.003 val and almost none of the response; the ΔT LOSS buys
  the full physics response for ~−0.008 val. Panel b localizes the basic
  arm's failure: its regime SHAPE is right (r 0.95) but it overshoots the
  winter/spring flow gain ~+80% while matching the summer loss, so the
  ANNUAL total comes out wrong-signed — the pathology lives in the annual
  sum, not the seasonal pattern. Canonical set unchanged (the pair); the
  PET-only rung stays in gitignored testing/ as the exhibit's middle point.
- **Trajectory head-room (recorded, NOT applied — user kept 60 epochs)**:
  the hybrid runs are epoch-capped, not converged — patience-12 never
  fires, most seeds run to ep58-59, 6/14 seeds still drift upward across
  the last evals, LR ≤1e-4 at the best epoch.
  `CosineAnnealingLR(T_max = n_epochs − warmup)` stretches with `--epochs`,
  so a 120-epoch run is the natural head-room experiment if the ensembles
  ever need another few thousandths.

## dt/dp climate-response surfaces (2026-07-18, branch `dtdp-response`)

Generalizes the +2 °C ΔT-consistency loss to PRECIPITATION and JOINT precip+temp
response consistency, with a per-watershed (Δprecip, ΔT) response-surface
diagnostic that scores whether the loss works.

- **Multi-anchor response loss** (`hybrid/{data,train}.py`, CLI `--response-grid`):
  the single ΔT term is now a list of (dp, dt) anchors. Each perturbs the feature
  copy — temps `+dt/σ`, precip RE-Z-SCORED `×(1+dp)` (multiplicative, not a level
  shift), PET recomputed under dt, sim channel = the physics run under the anchor —
  and MSE-pulls the hybrid's daily response `Q(dp,dt)−Q` toward physics.
  `apply_response_perturbation` is the one shared recipe (training + the sweep);
  the legacy `temp_*` knobs are the n=1 dt anchor (byte-identical math, verified).
  `--response-grid` = the 5 corners of {−10%,0,+10%}×{0,+3 °C}.
- **Physics engine = FROZEN noah-lite, torch-anchored**
  (`dtdp_response.physics_daily = base_torch + [frozen(dp,dt) − frozen(0,0)]`): the
  numba noah-lite core is ~4 s/run (vs ~14 min for the torch stream) and its (dp,dt)
  RESPONSE matches the torch noah to <0.3% on annual runoff (verified vs the torch
  ±10% teachers); exact at (0,0) so the hybrids' present-climate baseline is
  unperturbed. One source of truth for the teachers, the physics column, and the
  hybrids' perturbed sim channel. Turns the 25-point sweep from ~6 h into ~2 min.
- **`hybrid_pet_dtdp`** (`testing/`, gitignored scratch; 3 seeds matched to the raw
  `hybrid_pet_noah` — h64/dropout.35/noise.2/pet/statics/no-doy — + response λ=0.1
  on each of the 5 anchors): cal KGE **0.8994 / 0.9060 / 0.9035** (mean ~0.902 vs
  the raw's ~0.910 — the small, expected response-loss cost; no collapse).
- **Result** (`dtdp_response_metrics.csv` + `figures/dtdp_response/<BASIN>.png`;
  5×5 grid dp∈[−20,20]% × dt∈[0,4] °C, 4 metrics — total annual runoff, Apr–Jul
  freshet, max/min monthly — × 3 models, % change vs present climate, `×` = the
  supervised anchors):
  - PRECIP axis: both hybrids track physics (+10% precip annual: phys +20%, raw
    +17%, dtdp +18%) — precip is a direct input, so even the raw responds.
  - WARMING axis is the discriminator: the RAW hybrid is nearly flat and
    WRONG-SIGNED at several basins (+3 °C annual: SHA +1.3, ORO +4.5, NHG +4.5,
    SCC +4.3 — warming *raises* runoff), pooled **−0.9% vs physics −5.0%**; the
    dt/dp hybrid recovers the physical signal (**−4.2%**, right-signed 14/15). The
    same contrast reads off the annual/max-monthly contour TILT (raw vertical /
    wrong-tilted; dtdp tilts like physics). Apr–Jul freshet is the metric warming
    erodes most (all three capture it; dtdp closest to physics).
  - Verdict: the multi-anchor loss extends the ΔT result — it makes the hybrid's
    climate response trustworthy on BOTH the precip and warming axes, where the
    unconstrained PET hybrid is a skill lever only (reconfirms "PET = skill lever,
    not response lever"). Some southern-basin dtdp overshoot on warming
    (MKM/TLG/MRC/MIL/PNF/TRM −6…−9% vs phys −4…−6%) — a λ screen is the lever if
    tightening is wanted. Trained scratch ensembles stay local (like the raw);
    only the figures + CSV + code are tracked.
- **Denser grid + λ screen** (2026-07-18): the sweep grid was tightened to 9×9
  (dp step 5% / dt step 0.5 °C, 81 points) for smooth `contourf`, and a
  `hybrid_pet_dtdp_l0.3` variant (5 anchors, **λ=0.3**) trained to cal KGE
  0.886. λ=0.3 tightens per-basin +3 °C fidelity vs λ=0.1 (mean |err| vs physics
  2.67 pp, 15/15 signs, ratio 0.92) at ~0.016 more cal cost — southern overshoot
  is a pooling limit λ dents but doesn't remove.
- **Climate-static co-variation + 8-anchor grid** (`hybrid_pet_dtdp_cs8`,
  2026-07-18 — the recommended dt·dp variant): two changes over the above.
  (1) The hybrid's static net carries two CLIMATE statics (pmean, snowf) alongside
  the physiographic elev/flowlen; these now **co-vary with the perturbation** at
  both train and eval (`data.perturbed_static`: `pmean×(1+dp)`; `snowf` recomputed
  with the freeze threshold shifted by dt — invariant to uniform precip scaling, so
  it responds to dt only; both exact at (0,0), verified). (2) Anchors =
  **{−10%,0,+10%}×{0,+2,+4 °C}** (8 non-origin), λ=0.18. Cal KGE
  **0.8893 / 0.8931 / 0.8869** (~0.890; full skill, ~0.027 below the raw's 0.917 —
  the multi-anchor response-loss cost, same order as l0.3). **Best per-basin
  warming fidelity of any variant: +3 °C mean |err| vs physics 1.93 pp**
  (vs l0.3 2.67, raw 5.93), ratio 0.89, 14/15 signs. `dtdp_response_metrics_cs8.csv`
  + `figures/dtdp_response_cs8/<BASIN>.png` (8 anchors marked) +
  `figures/dtdp_lambda_compare.png`.
  - DECOMPOSITION (the raw column now also gets the co-varying statics at eval):
    the climate-static signal ALONE lifts the raw's pooled +3 °C response from the
    old flat/lottery (~−0.9%, ratio 0.24) to −6.1% (ratio 1.22) — directionally
    right on average — but leaves the per-basin response CATASTROPHICALLY
    miscalibrated: +24% over-response at TLG/MIL, wrong-signed at NHG/SCC, and an
    NHG min-monthly-flow blowup to **+125%** (arid basins amplify min-monthly % —
    the baseline min is ~0). The dt·dp response loss is what tames the per-basin
    scatter to physics (NHG min-flow → ~0, mean |err| 5.93→1.93). So: statics
    co-varying = necessary lever, response loss = the calibrator; neither alone.
  - Physics is still FROZEN here (dPL noah uses the physiographic `physical`
    variant, so its SAC params don't shift under climate — only its FORCING does).
    Making the noah backbone itself climate-adaptive (a `physical_climate` feature
    variant + noah retrain) is Phase 2 (below).

## Climate-adaptive physics + noah_ca hybrid family (Phase 2, 2026-07-18, branch `dtdp-response`)

Makes the dPL-noah backbone ITSELF climate-adaptive, adopts it as the physics
basis, and rebuilds the hybrid family on it.

- **`physical_climate` feature variant** (`features.py`, CLI `dpl train
  physical_climate`): the 23 physiographic `physical` features PLUS the 4 climate
  indices (p_mean/aridity/snow_frac/seasonality).  Retrained the noah on it
  (canonicalized 2026-07-19 to **`noah_ca/`**, exact-reconstructed noah cfg — only
  the variant differs): frozen cal/val **0.779/0.804 ≈ the canonical noah
  0.767/0.799**, so the indices cost no present-climate skill (torch selection cal
  0.745; killed early, annealed).
- **`noah_ca` = the climate-ADAPTIVE physics** (`adaptive_physics.py`): under
  (dp,dt) the params are RECOMPUTED by re-running the trained net on climate indices
  built from the perturbed forcing (`adaptive_params`; physiographic features + z-
  scoring frozen; exact at (0,0), verified max|d|=0).  Canonical labels: **`noah`** =
  canonical (physical, params frozen — forcing-only response); **`noah_ca`** =
  climate-adaptive (physical_climate, params co-vary).  Diagnostic (frozen vs
  adaptive params, same model): param adaptation AMPLIFIES warming-drying by
  **−1.3%/+2 °C → −2.3%/+4 °C** (Kpet +1→1.6%, lzsk −4→−8%), arid-concentrated
  (ISB max), robust across precip — a real space-for-time effect (2nd-order vs the
  forcing response).  Physics figure = 2 cols `[noah | noah_ca]` × 4 metrics, per
  watershed (`figures/adaptive_physics/`) AND per freshet-tercile regime
  (`figures/adaptive_physics_regimes/{snow,mix,rain}.png`, area-weighted); both
  from `adaptive_physics_metrics.csv`.  (`REGIMES` / `_aggregate_regime` are shared
  from `dtdp_response` by the physics + hybrid-family regime figures.)
- **noah_ca hybrid family** (`noah_ca_hybrids.py`; 3 seeds each, all on the noah_ca
  basis, `--pet-input --no-doy --statics` h64/drop.35/noise.2, 15cdec_grid;
  canonical dirs **`hybrid_base` / `hybrid_dtdp` / `lstm`** — the `noah_ca` infix
  is dropped since it is the family's default basis):
  - `base hybrid` — noah_ca sim channel, NO response loss: cal **0.922** / val
    **0.877**.
  - `dt·dp hybrid` — + **14-anchor {−20,−10,0,+10,+20}×{0,+2,+4}** response loss
    vs the noah_ca ADAPTIVE teachers, λ0.18: cal 0.873 / val 0.849.  The precip
    axis was extended to ±20% (from ±10%) so the SURFACE EDGES are supervised, not
    extrapolated — mean |err vs physics| on annual %Δ at the ±20% edge drops to
    **3.7 (dt·dp) vs 10.1 (base)**, and the dt·dp edge is only ~1.9× its interior
    (1.9→3.7) instead of running away.  Cost: ~0.018 cal / ~0.008 val vs the
    8-anchor, and the dp=0 interior warming ratio loosened 0.98→1.14 (mild
    over-response) — the interior↔edge trade of spreading 14 anchors.
  - `pure LSTM` (dir `lstm/`) — NO physics sim channel (`use_sim=False`, new toggle
    threaded through `feature_names`/`data`/`train`/`evaluate`); the pure data-driven
    control (no SAC-SMA connection at all), keeping only the SAME climate-adaptive
    statics: cal 0.909 / val **0.835**.
- **Surface metrics (revised 2026-07-19)**: the 4 rows are total annual runoff,
  Apr–Jul freshet, daily **Q99.9 (flood peak)**, daily **Q30 (low flow)** — the
  daily percentiles replace the old mean-monthly max/min.  The high-flow row is
  deliberately the extreme tail (Q99.9 ≈ top 37 days of 1915-2018), NOT Q98: in
  the noah_ca physics the snow-basin percentile response to +4 °C warming *crosses
  over* — Q95/Q98 fall (−12 %, the snowmelt-freshet shoulder, already carried by
  the freshet row) but the flood tail rises (Q99 −5.7 % → Q99.5 +4 % → **Q99.9
  +36 %, all 5 snow basins positive**; the whole top tail thickens, not one day).
  Mix basins rise even at Q98 (+8 %); rain basins are flat (no snow→rain
  amplification).  So Q99.9 is the *complement* of the freshet row: warming shrinks
  the snowmelt freshet but intensifies flood peaks.  The dt·dp `×` anchor grid is
  drawn on **every** column (all figure sets) for eye cross-comparison.
- **Evaluation window (2026-07-19)**: the response metrics reduce over **WY1951-1988
  + WY2004-2018** (`dtdp_response._eval_mask`), which (1) EXCLUDES the WY1989-2003 CAL
  window the hybrids + dt·dp loss trained on → the reported response is OUT-OF-SAMPLE,
  and (2) drops the 1915-1950 cold-start lead-in (the physics `run_basin` cold-starts
  1915 with SMA [0,0,100,100,100,0] / Snow-17 zeros; ~35 yr equilibrates every store
  incl. slow lztwc, and baseline/perturbed share the spin-up so it cancels in the %Δ
  regardless).  The ANNUAL response is period-insensitive (physics snow +4 °C −8.7 vs
  −9.0 full-record); Q99.9 is period-sensitive (physics snow +4 °C **+47** on this
  window vs +36 full-record — the pre-1950 record damped it), but the model RANKINGS
  hold in every period (dt·dp closest to physics, base over-, LSTM worst — and dt·dp
  tracks the flood peak even tighter on-window, 45.5 vs 47.2).
- **RESULT** (`noah_ca_hybrids_metrics.csv` + `figures/noah_ca_hybrids/<BASIN>.png`
  4-col physics/base/dtdp/lstm, `figures/noah_ca_regimes/{snow,mix,rain}.png`
  [freshet-tercile regime aggregates, area-weighted], + the 3-panel
  `figures/noah_ca_summary.png` = skill bars + warming-response and precip-response
  CURVES (line per model, physics the black reference; annual %Δ vs ΔT at Δp=0, and
  vs Δp held at +2 °C — dt·dp hugs physics on both axes, LSTM inverts/flattens):
  - **Physics sim channel buys GENERALIZATION**: pure LSTM ties base on cal
    (0.909 vs 0.922) but trails **~0.04 on val** (0.835 vs 0.877).  Val order:
    physics 0.804 < LSTM 0.835 < dtdp 0.857 < base 0.877.
  - **Only physics + the dt·dp loss gives a trustworthy warming response**:
    pure LSTM is **WRONG-SIGNED** (+5.6%, ratio **−0.94**, 5/15 signs) — a data-
    driven model, even with the climate-adaptive statics, learns warm⇒high-flow
    (seasonal melt) and extrapolates warming to MORE runoff.  Base over-responds
    (−9.0%, 1.50×, 13/15).  dt·dp tracks physics closely (−6.8%, **1.14×,
    15/15, err 1.9** at dp=0; full-grid annual err **2.6 vs base 7.6, LSTM 24.6**).
  - **Flood-peak (Q99.9) reproduction** — the extreme tail, NOT trained on (the
    dt·dp loss matches only the *bulk* daily response): +4 °C snow-basin flood peaks
    rise physics +36 % → dt·dp **+41 %** (closest) → base +48 % → pure LSTM **+67 %**
    (~2× physics).  Rain basins are the tell — physics ≈ 0 (−1 %), dt·dp tracks it
    (**−0.1 %**), but base (+10 %) and LSTM (**+23 %**) INVENT a warming-driven rain
    flood increase where there is none.  The dt·dp physics-consistency generalizes to
    the tail it never saw; the LSTM over-responds worst exactly where flood risk lives.
  - **Precip response held at +2 °C** (summary panel 3, pooled annual %Δ vs the
    +2 °C state): physics −35.5 % (−20 % precip) / +38.7 % (+20 %); dt·dp tracks it
    (−33.4 / +37.1), base under-responds (−26.4 / +31.1), and **pure LSTM is flat &
    inverted (+0.3 / −3.8)** — drying nudges flow UP, wetting DOWN.  Same failure
    mode as the warming axis: no physics channel ⇒ no trustworthy precip sensitivity.
    Verdict: base = best skill / over-strong response; **dt·dp = the
    climate-trustworthy model** (small skill cost); pure LSTM = good skill,
    physically nonsensical projection on BOTH axes = the case FOR physics.
- **Canonicalized 2026-07-19** out of gitignored `testing/` into tracked `noah_ca`
  (physics ckpt + params + `metrics_noah_ca.csv` + present sim channel
  `frozen_sim_noah_ca.csv`) / `hybrid_base` / `hybrid_dtdp` / `lstm` (3-seed
  ensembles, checkpoints tracked like `noah`/`hybrid`).  The two eval loaders
  (`hybrid.evaluate._load_data`, `dtdp_response._load_ensemble`) gained
  `physics_csv`/`sim_cache` overrides so the moved checkpoints' stale training-time
  paths don't bite; the regenerable adaptive (dp,dt) physics cache stays gitignored
  at `_adaptive_cache`.  Annual/response numbers verified BIT-IDENTICAL pre/post move.

## Open items
- Canonical set (2026-07-19): the physics ladder `hamon_dense`, `hamon`, `pt`,
  `noah`, and the climate-adaptive `noah_ca` (`physical_climate` features — the
  physics basis for the current hybrids); plus the `noah_ca` SAC×LSTM family
  `hybrid_base` (best skill 0.922/0.877), `hybrid_dtdp` (0.873/0.849, the
  climate-trustworthy model: full (Δp, ΔT) response loss, +3 °C ratio 1.14,
  15/15 signs) and the pure `lstm` control (0.909/0.835, +3 °C ratio −0.94
  wrong-signed = the case for physics). `hybrid_dtdp` SUPERSEDES `hybrid_pet_dt`
  for climate work; `hybrid`/`hybrid_pet_dt` (frozen-`noah` basis, +2 °C-only:
  0.917/0.869 and 0.916/0.864) are retained as the predecessors the (Δp, ΔT)
  loss generalizes, and `noah`'s torch daily dumps (the older hybrid sim channel
  + the +2 °C teacher) stay. `noah_ft` demoted 2026-07-17; its refinement
  candidates (SWE-participation-weighted melt harmonics, tighter anchor band) are
  moot unless the seasonal-melt line is revived.
- Hybrid val volume bias at NML/MRC/ORO remains intrinsic to cal-only
  training (B1–B3 + D2 all refute fixes); at those basins `noah` is the
  trustworthy out-of-sample answer.
- Region-store gap: statics rasters cover only the 2480 domain cells — the
  1930 footprint-only cells need a raster ingest before any full-region
  training (INVENTORY §data/region).
- `pt_refined_noah_lite` resolved 2026-07-14 (wash + reshuffle, not promoted;
  see the Noah ET line).
