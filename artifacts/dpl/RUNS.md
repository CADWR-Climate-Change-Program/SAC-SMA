# dPL run track record

Structure: **canonical** runs live directly under `artifacts/dpl/<label>`.
Frozen-physics runs (`hamon_dense`, `hamon`, `pt`, `noah`) carry
`checkpoints/best.pt`, `train_log.csv`, `params_dpl.csv`, `metrics_<label>.csv`
(+ `params_canopy.csv` for `noah`). `noah_ft` (the obs-steered seasonal-melt
fine-tune of `noah`) is **torch-only** — the frozen `run_basin` cannot
reconstruct its seasonal MFMAX/MFMIN/MBASE harmonics — so it carries
`metrics_noah_ft.csv` *(torch)* plus the load-bearing `daily_sim_noah_ft.csv`
(date×basin mm/day): every frozen-path consumer (climatology,
forcing-sensitivity, the hybrid sim channel) reads that dump instead of its
`params_dpl.csv`. The canonical SAC×LSTM **ensembles** — `hybrid` (basic
feature coupling on `noah_ft`, no doy inputs) and `hybrid_pet_dt` (+ the raw
PT-potential input + the ΔT-consistency loss; kept as a PAIR to show the
climate-response improvement) — hold `seed*/checkpoints/best.pt` + per-seed
`metrics_hybrid.csv` and a top-level `metrics_hybrid.csv` scoring the
**ensemble-mean** flow (`hybrid.evaluate.score_ensemble`); the residual
coupling and the two `noah`-based ensembles were retired 2026-07-16 (see
below). Exploratory/superseded run artifacts were pruned; their findings stay
in the track record below (names there no longer resolve to on-disk runs).
`--physics` is REQUIRED with no default, so `testing/` holds only gitignored
local scratch. `fidelity/` is the numerics benchmark (infrastructure, not a
run). All skill numbers are pooled 15-basin mean KGE, frozen-model scoring
(numba; PT via `sacsma.pet_pt`, Noah-lite via `sacsma.sma_noah_lite`) unless
marked *(torch)* — `noah_ft` and the full 7-param Noah ET (`noah_grid*`) are
torch-only (no frozen core).

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
| `noah_ft` | 15cdec_grid | warm-start fine-tune of `noah`: seasonal Kpet + MFMAX/MFMIN/MBASE melt harmonics, identified by ET/SWE-shape obs + P−Q anchor (flow-KGE selection) | 0.765/0.799 *(torch; vs noah-torch 0.759/0.792)* |
| `hybrid` | 15cdec_grid | SAC×LSTM feature coupling on `noah_ft` physics (sim-cache channel), no doy inputs — 8-seed ensemble mean. The BASIC hybrid: the skill step, with ~no warming response (+2 °C resp ratio 0.15) | 0.917/0.865 |
| `hybrid_pet_dt` | 15cdec_grid | `hybrid` + the raw PT-potential input channel (`--pet-input`) + the temperature-consistency loss λ=0.3 (+2 °C teacher) — 8-seed ensemble mean. Same skill, physics-consistent climate response (+2 °C resp ratio **0.89**, regime r 0.97) | **0.917/0.864** |
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

## Canonical rebuild on noah_ft (2026-07-16, in progress)

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

## Open items
- Canonical set (2026-07-16): `hamon_dense`, `hamon`, `pt`, `noah`,
  `noah_ft` *(torch, 0.765/0.799)*, and the two SAC×LSTM ensembles
  `hybrid` (basic: 0.917/0.865, +2 °C response 0.15) and `hybrid_pet_dt`
  (PET input + ΔT-consistency λ0.3: 0.917/0.864, response 0.89) — kept as a
  PAIR to show the climate-response improvement. `noah_lstm_feat`/
  `noah_lstm_resid` and the intermediate no-PET λ0.3 ensemble retired.
- noah_ft refinement candidates (not scheduled): SWE-participation-weighted
  melt harmonics (NHG), tighter `--et-anchor-band` (SHA/BND/ORO drift).
- Hybrid val volume bias at NML/MRC/ORO remains intrinsic to cal-only
  training (B1–B3 + D2 all refute fixes); at those basins `noah_ft` is the
  trustworthy out-of-sample answer.
- `pt_refined_noah_lite` resolved 2026-07-14 (wash + reshuffle, not promoted;
  see the Noah ET line).
