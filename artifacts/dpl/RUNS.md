# dPL run track record

## Layout

Canonical runs live directly under `artifacts/dpl/<label>`. The frozen-physics runs
(`hamon`, `pt`, and the climate-adaptive `noah`) each carry `checkpoints/best.pt`,
`train_log.csv`, `params_dpl.csv`, and `metrics_<label>.csv` (plus
`params_canopy.csv` for `noah`, and the present-climate sim channel
`frozen_sim_noah.csv`).

`noah` also carries a torch daily dump (date × basin, mm/day),
`daily_sim_noah_torch.csv`, the frozen-noah-basis hybrids' sim channel.

Runs on the multi-timescale `multifamily` domain sit in the group folder
`artifacts/dpl/multifamily/<label>`; their layout and the CalSim3 validation outputs they
carry are described in the multifamily section below.

The six tracked multifamily runs (folders under `artifacts/dpl/multifamily/`):

| name | run folder | trains on | status |
|---|---|---|---|
| `noah_cdec_uf_usgs` | `noah_cdec_uf_usgs` | 95 entities: 69 USGS daily gauges, 17 CDEC daily, 9 DWR-unimpaired monthly; family shares by observed water volume | first multifamily recipe (2026-09), kept for the record |
| `noah_cdec_uf_usgs_areaw` | `noah_cdec_uf_usgs_areaw` | the same 95 entities; family shares by footprint area | first multifamily recipe; the source of dPL-95's family shares and loss reference levels |
| `noah_cdec_uf` | `noah_cdec_uf` | 26 entities: 17 CDEC daily, 9 DWR-unimpaired monthly | first multifamily recipe; the control of the training program |
| **dPL-26** | `noah_cdec_uf_sacx_carry_px_aef` | the same 26 entities, on the recipe the training program ended with | adopted as base 2026-09-28 (then "run H"); replaced as base by dPL-95 |
| **dPL-95** | `noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef` | dPL-26's recipe on the 95 entities, family influence equalized | adopted as base 2026-09-29 (then "H95"); the recipe dPL-CalSim starts from |
| **dPL-CalSim** | `noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef` | dPL-95's entities + 64 CalSim3 arcs, with WY1976–85 held out of every family | adopted 2026-09-30; the current CalSim dPL, carries the CalSim3 rim-inflow product in `calsim_product/` |

Each run tracks the ten files of the multifamily layout (the selected checkpoint, the training log,
the entity metrics, the two parameter tables, `sim_daily_mm.npz`, the tier-1 and tier-2 score tables
and the atlas). dPL-CalSim adds its holdout scores (`metrics_entities_holdout.csv`,
`tier2/tier2_anchor_rescaled.csv`), `provenance/` and `calsim_product/`; dPL-95 has a small
`provenance/` with its share and loss-scale derivation. Every other folder under
`artifacts/dpl/multifamily/` is local and git-ignored: the other runs of the training program (R1,
R2, `noah_cdec_uf_sacx` and A–G2) and the `_failed_` / `_stopped_` / `_aborted_` / `_old_` folders.
The first three runs and the layout are in "Multi-timescale training on the multifamily domain",
dPL-26 and dPL-95 in "Multifamily training program: runs A–H and H95", dPL-CalSim and the product
in "CalSim3 rim-inflow product".

**2026-07-21 rename** (see Open items, below, for the full record): the
climate-adaptive physics and its hybrid family, canonicalized 2026-07-19 under the
`noah_ca`/`hybrid_base`/`hybrid_dtdp` names below, were promoted to the plain
top-level names `noah` / `hybrid` / `hybrid_dt` — they are now THE canonical
physics and hybrid family, not a parallel climate-adaptive track. The prior
frozen-noah-basis generation (the original `noah`, `hybrid`, `hybrid_pet_dt`, and
`hamon_dense`) moved to `superseded/{noah_noca, hybrid_noca, hybrid_dt_noca,
hamon_dense}` for lineage. Everywhere below that predates 2026-07-21 uses the old
names as originally written (this is a track record, not rewritten); read
`noah_ca`→`noah`, `hybrid_base`→`hybrid`, `hybrid_dtdp`→`hybrid_dt`, and the old
`noah`/`hybrid`/`hybrid_pet_dt`/`hamon_dense`→their `superseded/*_noca` (or
unchanged-name, for `hamon_dense`) counterparts.

The `hybrid` (basic feature coupling on the `noah` torch channel, no day-of-year
inputs) and `hybrid_pet_dt` (adds the raw PT-potential input and a single +2 °C
ΔT-consistency loss) ensembles — now `superseded/hybrid_noca` and
`superseded/hybrid_dt_noca` — established the skill step and the +2 °C response
result. They were the +2 °C-only predecessors of the climate-adaptive hybrid
family (now `hybrid` / `hybrid_dt` / `lstm`; see the Phase-2 section below), which
rebuilds them on climate-adaptive physics and generalizes the single ΔT anchor to
the full (Δp, ΔT) response surface, so `hybrid_dt` supersedes
`superseded/hybrid_dt_noca` for climate work. Each ensemble holds
`seed*/checkpoints/best.pt` and per-seed `metrics_hybrid.csv`, plus a top-level
`metrics_hybrid.csv` scoring the ensemble-mean flow
(`hybrid.evaluate.score_ensemble`). The residual coupling and the first
`noah`-based ensembles were retired 2026-07-16, and `noah_ft` (the seasonal-melt
fine-tune, canonical 2026-07-16→17) was demoted 2026-07-17 after a new-basis
head-to-head — see the track record; git history and the gitignored
`testing/noah_ft_region` hold the record.

Superseded run artifacts were pruned, but their findings stay in the track record
below (the names there no longer resolve to on-disk runs, except the
2026-07-21-renamed generation, which is retained on disk under `superseded/`).
`--physics` is required with no default, so `testing/` holds only gitignored local
scratch, and `noah/fidelity/` is the numerics benchmark, not a run. All skill
numbers are pooled 15-basin mean KGE under frozen-model scoring (numba; PT via
`sacsma.pet_pt`, Noah-lite via `sacsma.sma_noah_lite`) unless marked *(torch)* —
the seasonal-melt fine-tunes and the full 7-param Noah ET (`noah_grid*`) are
torch-only, with no frozen core.

Standing methods shared by every canonical grid run: the `physical` feature
variant; pooled 15cdec training (daily gage FNF, cal WY1989–2003 / val WY2004–2018);
CalSim3-footprint aggregation (`--calsim-footprint`); an NNSE+log loss with a
variance-matching term; cal-KGE selection every 2 epochs; and a truncated no-grad
spinup from 1978-10-01.

## Canonical lineage

Labels below are the **current, post-2026-07-21-rename** names (see Open items for
the rename record); descriptions still narrate each row's own history under the
names used at the time.

| label | domain | delta vs predecessor | cal/val KGE |
|---|---|---|---|
| `superseded/hamon_dense` | 15cdec (7891 HRU) | — (the original dPL); superseded by `hamon`, retained for lineage | **0.806/0.840** (retrained 2026-07-14) |
| `hamon` | 15cdec_grid (2074 cells) | native-grid retrain + CalSim3 footprint | 0.807/0.829 |
| `pt` | 15cdec_grid | Priestley–Taylor PET (Bristow–Campbell Rn) + snow-cover albedo (0.6) + arid dewpoint depression (2 °C) | 0.799/0.826 |
| `superseded/noah_noca` | 15cdec_grid | Noah-lite canopy ET (1 learned DOF `soil_chi`) on PT potential; climate-frozen, superseded 2026-07-21 by the climate-adaptive `noah` below | 0.767/0.799 |
| `noah` | 15cdec_grid | (was `noah_ca`) climate-frozen `noah` retrained on `physical_climate` features (physiographic + 4 climate indices) → climate-ADAPTIVE physics: parameters recompute under perturbed climate. Promoted to the plain `noah` name 2026-07-21; the physics basis for the current hybrid family (2026-07-19) | 0.779/0.804 |
| ~~`noah_ft`~~ | 15cdec_grid | DEMOTED 2026-07-17 — the seasonal-melt fine-tune of `noah` (0.765/0.799 torch): the new-basis head-to-head is a pooled wash vs `noah` with NHG + north-state-volume casualties, and its torch-only scoring taxed every consumer | — |
| `superseded/hybrid_noca` | 15cdec_grid | SAC×LSTM feature coupling on the climate-frozen `superseded/noah_noca` physics (torch daily sim-cache channel), no doy inputs — 8-seed ensemble mean. The BASIC hybrid: the skill step; its unconstrained +2 °C response is untrustworthy (see 2026-07-17); superseded 2026-07-21 by `hybrid` below | 0.917/0.869 |
| `superseded/hybrid_dt_noca` | 15cdec_grid | (was `hybrid_pet_dt`) `superseded/hybrid_noca` + the raw PT-potential input channel (`--pet-input`) + the temperature-consistency loss λ=0.3 (+2 °C `superseded/noah_noca` torch teacher) — 8-seed ensemble mean. Same skill, physics-consistent climate response (+2 °C resp ratio 1.04, regime r 0.97). SUPERSEDED for climate work by `hybrid_dt` below (2026-07-19), then renamed into `superseded/` 2026-07-21 | 0.916/0.864 |
| `hybrid` | 15cdec_grid | (was `hybrid_base`) SAC×LSTM feature coupling on the climate-adaptive `noah` physics (`--pet-input --statics`, no doy), NO response loss — 3-seed ensemble mean. Promoted to the plain `hybrid` name 2026-07-21; best skill, +2 °C response over-strong (ratio 1.50) | 0.922/0.877 |
| `hybrid_dt` | 15cdec_grid | (was `hybrid_dtdp`) `hybrid` + the 14-anchor {−20,−10,0,+10,+20}%×{0,+2,+4 °C} (Δp, ΔT) response-consistency loss (λ0.18) vs the `noah` adaptive teachers — 3-seed mean. Promoted to the plain `hybrid_dt` name 2026-07-21; THE climate-trustworthy model: tracks physics on both axes (+3 °C ratio 1.14, 15/15 signs); generalizes `superseded/hybrid_dt_noca` to the full surface | **0.873/0.849** |
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

## Multi-timescale training on the multifamily domain (2026-09)

**Domain and code.** `--domain multifamily` trains one parameter network on the entities of
`data/multifamily/` (69 `usgs_daily`, 17 `cdec_daily`, 9 `uf_monthly`), each over its own record
window inside the WY1950–2018 envelope: daily NNSE on the daily entities plus monthly NNSE on the
monthly ones, the gated and Huber-capped variance-matching term, family weighting
(`--mt-family-weight`), entity subsets (`--basins`) and exact warm starts (`--init-from`). Method:
`docs/part2.md`; store: `data/multifamily/README.md`.

**Layout.** Runs of this domain sit in the group folder `artifacts/dpl/multifamily/`, named by the
families they train on (a suffix names the family weighting where two runs share the families).
Each tracks ten files: `checkpoints/best.pt`, `train_log.csv`, `metrics_entities.csv`,
`params_dpl.csv`, `params_canopy.csv`, `sim_daily_mm.npz` (the simulated daily depth, entity × day
over the envelope: the physics channel a hybrid reads), `tier1/tier1_metrics.csv`,
`tier2/tier2_metrics.csv`, `tier2/tier2_arcs.csv` and `atlas/calsim_validation_atlas.html`
(self-contained; download it to view). Everything else in a run folder is git-ignored and
regenerates: `sacsma dpl evaluate <run>/checkpoints/best.pt` (metrics, parameters,
`sim_daily_mm.npz`, figures), `python -m sacsma.dpl.calsim_tier1 <run>`, `python -m
sacsma.dpl.calsim_tier2 <run>` (a forward pass; `--figures-only` redraws the maps from the tracked
CSV and, after a full tier-2 run has left `tier2_monthly.csv` in the folder, the regime figures)
and `python -m sacsma.dpl.calsim_atlas <run>`.  The atlases and `sim_daily_mm.npz` are git-LFS
files (`git lfs pull` after a clone).

**What the numbers are.** Entity metrics are calibration-window skill at each entity's native
timescale, scored by the torch pipeline (no frozen re-score, no held-out flow period). Validation
is the CalSim3 comparison over WY1950–84 only: tier 1 = the twenty training locations in monthly
volume, tier 2 = every rim `INFLOW` arc. At the ten anchored tier-1 locations the reference
coincides with the source of the monthly training target, so that score is a temporal holdout;
USGS creek records reach into WY1950–84: the atlas reports that overlap per location, and its
last tab sets the full-window scores against each location's trimmed window, the 20 or more
water years in which the creeks covered the least of it (`window = trimmed` in
`tier1/tier1_metrics.csv`; rule in `sacsma.dpl.calsim_windows`).

**Scoring spinup (2026-09-23).** The evaluator and tier 2 used to start the envelope from the ten
water years before it (WY1940–49). That left the slow lower-zone stores of trained fields far from
equilibrium: these fields put `lztwm` at its 5,000 mm ceiling, and under Noah-lite that store
loses water to ET only above a 5 % wilting fraction, so it takes centuries to settle. Both now
start from the timing-independent cycle spinup of `sacsma.dpl.spinup` (the envelope's first ten
water years looped 20 times from the frozen cold start; rule and evidence in
`artifacts/README.md`), which reads nothing before the envelope and so serves climate-perturbed
and stochastic forcing alike. Every score in the run entries and the table below is re-scored that
way. The parameters, the selected epochs and the selection scalars are training-time quantities
and do not change. For the three current runs (riva ≈ 0.85–0.9, lower-zone stores near empty) the
shift is small: tier-1 mean KGE +0.011 / +0.012 / +0.004, volume deficit 0.4 points smaller. For
the superseded riva = 0 `noah_cdec_uf`, whose stores fill, it is large: 0.810 → 0.856. As first
reported with the window spinup: `noah_cdec_uf_usgs` family means usgs 0.592 / cdec 0.772 / uf
0.859, tier-1 KGE mean 0.801 / median 0.812 (anchors 0.810, arc sums 0.791), median |bias| 7.2 %,
volume −1.9 %, trimmed 0.795 / 0.793 and −3.1 %, tier-2 median 0.601 (0.642 trained entities, 0.188
extrapolated); `noah_cdec_uf_usgs_areaw` 0.598 / 0.776 / 0.879, tier 1 0.813 / 0.822 (0.817 /
0.808), 7.2 %, −1.7 %, trimmed 0.808 / 0.805 and −2.9 %, tier 2 0.608 (0.644 / 0.224);
`noah_cdec_uf` cdec 0.788 / uf 0.891, tier 1 0.790 / 0.787 (0.791 / 0.789), 8.0 %, −3.2 %, trimmed
0.785 / 0.787 and −4.5 %, tier 2 0.573 (0.611 / 0.186).

Common recipe: `sacsma dpl train physical_climate --domain multifamily --et noah --noah-pet
priestley_taylor --canopy-lite --patience 10 --warmup-epochs 4 --chunk-grid water_year
--no-flowlen-feature`, seed 0, no `--calsim-footprint` (no effect on this domain). The chunks
follow the water years (52 × 365 + 17 × 366 days + a 92-day tail over the WY1950–2018 envelope), so
every month of the monthly entities is whole inside a chunk and in the loss (3,240 entity-months);
the parameter network reads 26 features — `flowlen` drives the routing but is no longer a network
input. All runs used `--nograd-window 256 --train-graph-segments 2`, which only splits the
CUDA-graph capture for GPU drivers that fault on whole-year graphs; the gradient is the same.
Training code at `cfa0fb3`; tier 2 at `fdd754e` (the extrapolated arcs' flow lengths traced on
the HydroSHEDS grid, 539 of 539 extension cells in each run).

Trainer options added 2026-09-22, all off by default (the three runs below predate them):
`--spinup-mode cycle` trains from the rule the evaluators score with (20 passes at the first
spinup, then from the previous state until a pass moves no basin's annual flow by more than
0.1 %); `--param-box name=lo:hi` narrows or pins a parameter's box (`riva=0:0`);
`--min-stop-epoch` arms the early stop no earlier than that epoch; `--dead-chunk-nograd` runs the
chunks that hold no scoreable observation forward only (bit-exact, epoch 352 → 201 s on the
26-entity set); `--diagnostics` writes a per-chunk loss and gradient log, the per-family terms of
each selection pass and per-selection snapshots with an EMA copy of the net. Feature variants `aef`
(AlphaEarth satellite embeddings: 16 principal components of the unit directions plus the vector
length, in place of every other input) and `aef_random` (its control, 17 random values per cell).

- **`multifamily/noah_cdec_uf_usgs`** — all three families, 95 entities; family shares by observed
  water volume, `--mt-family-weight usgs=0.27,cdec=0.54,uf=0.19`, `--epochs 120`. Selected epoch
  80 (early stop at 102); selection cal KGE 0.7400 (share-weighted family mean). Family mean cal
  KGE: usgs 0.587 / cdec 0.786 / uf 0.859. CalSim3 WY1950–84: tier-1 KGE mean 0.812 / median
  0.832 (anchors 0.826, arc sums 0.799), median |bias| 6.6 %, volume 31,629 vs 32,115 TAF/yr
  (−1.5 %) over the 19 locations of the totals (SHA sits inside UF6 and is scored, not summed);
  trimmed windows: KGE mean 0.807 / median 0.821, volume −2.8 %. Tier-2 median KGE 0.598 over 196
  arcs: 0.634 on the 155 arcs of trained entities, 0.200 on the 41 extrapolated arcs — 34 of
  those lie on cells USGS creeks trained on (median 0.209), 7 on cells no entity used (−0.324,
  the regionalization test).
- **`multifamily/noah_cdec_uf_usgs_areaw`** — the same 95 entities and recipe as
  `noah_cdec_uf_usgs` with the family shares by footprint area (the sum of the entities' footprints
  per family) instead of observed water volume,
  `--mt-family-weight usgs=0.216,cdec=0.558,uf=0.226`, `--epochs 120`. Selected epoch 78 (early
  stop at 100); selection cal KGE 0.7610 (the family mean weighted by these shares, so not the
  statistic behind the 0.7400 of `noah_cdec_uf_usgs`). Family mean cal KGE: usgs 0.592 / cdec
  0.791 / uf 0.879. CalSim3 WY1950–84: tier-1 KGE mean 0.825 / median 0.839 (anchors 0.835, arc
  sums 0.816), median |bias| 6.5 %, volume 31,702 vs 32,115 TAF/yr (−1.3 %); trimmed windows: KGE
  mean 0.822 / median 0.835, volume −2.5 %. Tier-2 median KGE 0.599 over 196 arcs: 0.643 on the
  155 arcs of trained entities, 0.233 on the 41 extrapolated arcs (34 on trained cells 0.251, 7 on
  unseen cells −0.297).
- **`multifamily/noah_cdec_uf`** — CDEC + DWR-unimpaired only, 26 entities through `--basins` (the
  17 `cdec_*` and 9 `uf_*` ids), `--mt-family-weight none` (one daily and one monthly family, so
  the two loss terms weigh equally), `--epochs 90`. Selected epoch 56 (early stop at 78);
  selection cal KGE 0.8241 (pooled). Family mean cal KGE: cdec 0.790 / uf 0.893. CalSim3 WY1950–84:
  tier-1 KGE mean 0.794 / median 0.792 (anchors 0.793, arc sums 0.796), median |bias| 8.0 %,
  volume 31,200 vs 32,115 TAF/yr (−2.8 %); trimmed windows: KGE mean 0.790 / median 0.796, volume
  −4.1 %. Tier-2 median KGE 0.576 over 196 arcs: 0.619 on the 155 arcs of trained entities, 0.194
  on the 41 extrapolated arcs (25 on trained cells 0.117, 16 on unseen cells 0.252 — without the
  creeks more of them are unseen).

**The two family weightings.** `noah_cdec_uf_usgs` and `noah_cdec_uf_usgs_areaw` differ in the
family shares only (usgs / cdec / uf: 0.27 / 0.54 / 0.19 by water volume, 0.216 / 0.558 / 0.226 by
footprint), with the same entities, seed and schedule. The footprint run selects two epochs earlier
and scores a little higher throughout: tier-1 mean KGE 0.825 against 0.812, higher at 17 of the 20
locations, by at most 0.057 (UF20, 0.657 against 0.599); volume −1.3 % against −1.5 %; tier-2
median 0.599 against 0.598. Both are single-seed, so the pair does not rank the two weightings.

**Reading the pair with and without the creeks.** `noah_cdec_uf_usgs` and `noah_cdec_uf` differ in
three things at once (the USGS family, the family weighting and the epoch cap), so their difference
is not a single-factor result, and the two selection scalars are not comparable (different entity
sets and statistics). Both are single-seed, so differences of a few hundredths in mean KGE are not
resolved (0.812 against 0.794, the run with the creeks higher at 13 of 20 locations). The
total-volume deficit is the clearer contrast: −1.5 % with the creeks against −2.8 % without, and at
the trimmed windows −2.8 % against −4.1 %.

**Superseded 2026-09-20.** The first runs under these names (code `92eefeb` / `c2faf29` /
`b147579`, merged at `eae478c`) had `flowlen` among the network inputs (27 features) and a fixed
366-day chunk grid that drifted off the water year, so months cut by a chunk boundary dropped
out of the monthly term (331 of each monthly entity's 360 months in the loss); their volume totals
counted Shasta twice (inside UF6). Their numbers, with the totals recomputed over the 19 locations:
`noah_cdec_uf_usgs` selected epoch 78, selection 0.7592, family means usgs 0.625 / cdec 0.790 /
uf 0.864, tier-1 KGE mean 0.809 / median 0.812 (trimmed 0.806 / 0.810), volume 31,181 vs 32,115
TAF/yr (−2.9 %; reported then as 37,047 vs 38,438, −3.6 %), tier-2 median 0.594 (0.640 trained
entities, 0.241 extrapolated); `noah_cdec_uf_usgs_areaw` epoch 78, selection 0.7715, family means
0.624 / 0.790 / 0.866, tier-1 0.808 / 0.811 (trimmed 0.805 / 0.809), volume −2.9 % (then −3.7 %),
tier-2 0.594 (0.641 / 0.222); `noah_cdec_uf` epoch 52, selection 0.8361, family means cdec 0.811 /
uf 0.884, tier-1 0.810 / 0.820 (trimmed 0.803 / 0.822), volume 29,272 vs 32,115 (−8.9 %; then
35,108 vs 38,438, −8.7 %), tier-2 0.587 (0.623 / 0.147). Retraining moved the tier-1 mean KGE by
−0.008 / +0.005 / −0.020 and the volume deficit from −2.9 / −2.9 / −8.9 % to −1.9 / −1.7 / −3.2 %:
skill about even, the volume bias smaller, most at the run without the creeks. The largest
single-location moves in the usgs run were CLE (0.773 → 0.701), UF7 (0.886 → 0.822) and UF4
(0.699 → 0.748). All of that was scored with the window spinup. Re-scored with the cycle spinup
(2026-09-23, the checkpoints kept locally in `_old_eae478c/`): `noah_cdec_uf_usgs` family means
usgs 0.621 / cdec 0.798 / uf 0.863, tier-1 0.816 / 0.818 (trimmed 0.814 / 0.813), volume −2.6 %,
tier-2 0.598 (0.637 / 0.236); `noah_cdec_uf_usgs_areaw` 0.619 / 0.798 / 0.867, tier 1 0.816 /
0.818 (0.813 / 0.814), −2.6 %, tier 2 0.604 (0.640 / 0.223); `noah_cdec_uf` cdec 0.812 / uf 0.883,
tier 1 0.856 / 0.863 (0.861 / 0.869), −4.6 %, tier 2 0.598 (0.661 / 0.157). On that basis
retraining moved the tier-1 mean KGE by −0.004 / +0.009 / −0.062. The run without the creeks lost
the most because it changed regime: the superseded field sits at riva = 0 (median pfree 0.48),
the retrained one at riva ≈ 0.85 with pfree 0.99, early in timing at 20 of 20 locations and with
median α 0.84 against 0.95. The window spinup had hidden the difference.

Tier 1 by location, CalSim3 WY1950–84, cycle spinup (from the three `tier1/tier1_metrics.csv`; the
UF9 second row scores the Yuba against the arcs its entity simulates):

| location | reference | CalSim3 TAF/yr | KGE `noah_cdec_uf_usgs` | bias % | KGE `noah_cdec_uf_usgs_areaw` | bias % | KGE `noah_cdec_uf` | bias % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SHA: Sacramento R. at Shasta | FLOW-UNIMPAIRED | 6,323 | 0.875 | -6.6 | 0.877 | -6.6 | 0.867 | -3.0 |
| CLE: Trinity R. at Trinity Dam | FLOW-UNIMPAIRED | 1,416 | 0.703 | +21.3 | 0.703 | +21.8 | 0.710 | +19.8 |
| UF6: Sacramento R. near Red Bluff | FLOW-UNIMPAIRED | 9,210 | 0.907 | -5.9 | 0.910 | -5.9 | 0.856 | -7.6 |
| UF7: Sacramento Valley east-side minor streams | arc sum | 1,328 | 0.845 | -1.5 | 0.846 | -2.7 | 0.826 | -3.0 |
| UF8: Feather R. near Oroville | FLOW-UNIMPAIRED | 4,941 | 0.830 | +0.5 | 0.831 | +1.0 | 0.784 | -4.4 |
| UF4: Stony Creek at Black Butte | arc sum | 505 | 0.751 | -13.5 | 0.742 | -13.7 | 0.697 | -19.4 |
| UF11: American R. at Fair Oaks | FLOW-UNIMPAIRED | 2,922 | 0.765 | -2.9 | 0.790 | -1.9 | 0.749 | -4.0 |
| UF9: Yuba R. at Smartville | FLOW-UNIMPAIRED | 2,546 | 0.761 | -0.8 | 0.777 | -0.1 | 0.771 | +5.4 |
| UF9: Yuba R. at Smartville | arc sum, covered arcs | 2,515 | 0.791 | +0.4 | 0.807 | +1.1 | 0.796 | +6.7 |
| UF10: Bear R. near Wheatland | arc sum | 384 | 0.862 | -8.0 | 0.907 | -6.3 | 0.881 | -5.8 |
| UF3: Cache Creek above Rumsey | arc sum | 718 | 0.864 | -6.9 | 0.867 | -8.2 | 0.790 | -17.6 |
| UF2: Putah Creek near Winters | arc sum | 425 | 0.887 | -5.5 | 0.879 | -7.2 | 0.963 | -3.3 |
| UF13: Cosumnes R. at Michigan Bar | arc sum | 421 | 0.738 | -3.8 | 0.749 | -3.0 | 0.775 | -4.9 |
| UF14: Mokelumne R. at Pardee | arc sum | 811 | 0.816 | +8.2 | 0.820 | +8.7 | 0.702 | +10.4 |
| UF16: Stanislaus R. at Melones | FLOW-UNIMPAIRED | 1,231 | 0.826 | +12.7 | 0.831 | +13.2 | 0.743 | +10.5 |
| UF15: Calaveras R. at Jenny Lind | arc sum | 178 | 0.835 | -13.6 | 0.867 | -11.3 | 0.811 | -16.8 |
| UF18: Tuolumne R. at Don Pedro | FLOW-UNIMPAIRED | 2,001 | 0.866 | -6.7 | 0.882 | -6.4 | 0.814 | -8.4 |
| UF19: Merced R. at Exchequer | FLOW-UNIMPAIRED | 1,025 | 0.881 | +8.4 | 0.895 | +7.9 | 0.841 | +11.3 |
| UF20: Chowchilla R. at Buchanan | arc sum | 78 | 0.599 | -26.6 | 0.657 | -23.2 | 0.689 | -18.5 |
| UF22: San Joaquin R. at Millerton | FLOW-UNIMPAIRED | 1,879 | 0.842 | -4.3 | 0.850 | -4.7 | 0.793 | -5.2 |
| UF21: Fresno R. near Daulton | arc sum | 95 | 0.789 | -2.9 | 0.826 | +0.3 | 0.824 | +8.7 |

## Multifamily training program: runs A–H and H95 (2026-09-22 → 29)

*Names. Later sections call the two adopted bases by what they train on: run H is **dPL-26**
and H95 is **dPL-95** (key in the CalSim3 rim-inflow product section). This section keeps the
letters it was written with.*

**Why.** The trained fields of the three runs of the section above share one regime: median riva
0.85–0.90, pfree 0.99, lzsk at its 0.003 floor, lztwm at its 5,000 mm ceiling on 95 % or more of
entity-cell rows and uztwm ≈ 1,000 mm, with tier-1 flows early in timing at 20 of 20 locations
(CalSim3 WY1950–84). The riparian term does the work of a summer ET sink that the Noah-lite path
lacks: with Noah-lite soil ET, the rest of the reference SAC ET block (the ADIMP ET, the upper-zone
free → tension rebalance, the lower-zone resupply) was skipped. From 2026-09-22 a series of
single-seed runs on the 26 entities of `noah_cdec_uf` changed one thing at a time (physics, the
TBPTT gradient, the network inputs, the loss), each scored through the full chain and read against
pass/fail bars written before its result. The series ended in two runs the user adopted as bases: H
(`noah_cdec_uf_sacx_carry_px_aef`, 26 entities, 2026-09-28) and H95
(`noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef`, 95 entities, 2026-09-29).

**Names and layout.** After the families, a run's name lists its recipe: `_rivapin` =
`--param-box riva=0:0`; `_seed1` = seed 1; `_sacx` = `--noah-sac-exchanges`; `_carry` =
`--tbptt-carry relative` (`_flux` = the flux carry); `_w2ft` / `_w3` = 2- or 3-water-year TBPTT
windows (`ft`: fine-tuned from A); `_tp` / `_g2` = the timing and peak loss terms, first and revised
form; `_px` = learned PXTEMP with the observation mask; `_aef` / `_aef64` = the feature variant (no
suffix: `physical_climate`); `_all` = `--mt-share-norm all`; `_kref05` = the frozen family loss
scale at p = 0.5. A folder named `_failed_` / `_stopped_` / `_aborted_` + run name + date holds a
launch that did not finish. H and H95 track the ten files of the multifamily layout; the other runs in
the table stay local and git-ignored, and each run folder records its code as `code_base.txt` +
`code_diff.patch`. The letters A–H are the labels the bars used.

**Recipe.** Every 26-entity run is the `noah_cdec_uf` recipe (the multifamily common recipe, `--basins`
with the 17 `cdec_*` and 9 `uf_*` ids, `--mt-family-weight none`, `--epochs 90`, seed 0 unless
named) plus the flags in the table, all with `--diagnostics --dead-chunk-nograd`; B through G2 used
`--patience 6`. Selection is the pooled cal KGE over the 26 entities (the statistic behind
`noah_cdec_uf`'s 0.8241), on the masked daily target for H. R1 and R2 ran on `17229ec` and
`noah_cdec_uf_sacx` on `e299947`, each with the working diff of the day; A onward ran on `889e671`
plus the options below.

**Trainer options added 2026-09-23 → 28**, all off by default:
- `--noah-sac-exchanges` (needs `--canopy-lite`; committed in `889e671`): restores the ADIMP ET(5),
  the upper-zone free → tension rebalance and the lower-zone free → tension resupply around the
  Noah-lite soil ET, in the torch path and the numba mirror (`sma_noah_lite`); exported
  `params_canopy.csv` files carry a `sac_exchanges` column.
- `--tbptt-carry relative|flux`: the state carried into each water-year chunk keeps its values, but
  the backward holds each SAC store's relative saturation fixed, so a larger capacity no longer
  looks like free deficit at every chunk start; `flux` also holds the lower-zone free stores'
  carried drainage flux fixed.
- `--tbptt-window-years 2|3` with `--graph-recompute-days 73`: each live chunk backpropagates
  through the previous one or two water years as a burn-in, the loss on its own year only; the
  recompute replays one captured 73-day CUDA graph and re-runs its segments in the backward, so
  graph memory stays one segment's.
- `--timing-lambda` and `--peak-lambda` (with `--peak-frac`, `--shape-min-days`,
  `--timing-vol-gate`; water-year grid only): a summer-recession timing term (normalized cumulative
  flow, 1 Jul–30 Sep) and a flood-peak term (the mean of the top 2 % of days, sim vs obs, over the
  record mean of that statistic, Huber-capped).
- `--learn-pxtemp` (`--pxtemp-box -1:3`, `--pxtemp-tau 1.0`): the Snow-17 rain/snow threshold learned
  per cell by a zero-init head (exactly the fixed 0 °C at init); the forward split stays hard and
  the gradient passes through a sigmoid surrogate of width τ, so the exported PXTEMP column runs
  as-is in the frozen `run_basin`.
- `--obs-mask CSV`: daily target observations masked out of training and scoring (the checkpoint
  carries the list). `data/cdec_fnf/fnf_daily_mask.csv` holds five confirmed artifact days: CLE
  1996-05-26, FOL 1993-02-24, YRS 1995-06-17, NML 2016-06-28, SHA 1996-01-26.
- `--mt-share-norm all`, `--mt-loss-ref` and `--mt-loss-ref-power`: see H95 below.
- Feature variant `aef64`: all 64 AlphaEarth unit-direction coordinates plus the length (65
  inputs), no PCA.

**The runs between the section above and H.** Single seed. Selection = best pooled 26-entity cal
KGE @ its epoch (early-stop epoch). Tier 1 = CalSim3 WY1950–84 KGE mean; "trimmed" = the trimmed
windows, full window at the 4 locations without one. Peaks and Shasta = CDEC daily, WY1988–2018;
"4-basin" = ORO, FOL, YRS, MKM. The "60 nested gauges" are the USGS gauges inside the trained
footprints that no 26-entity run trains on. The control is `noah_cdec_uf`: tier 1 0.794, trimmed
0.790, volume −2.8 %, early at 20 of 20 locations, median α 0.840.

| run | change | selection | verdict |
|---|---|---|---|
| `noah_cdec_uf_rivapin` (R1) | control + `--param-box riva=0:0` | 0.8314 @48 (70) | Leaves the regime by the pin (pfree 0.44, lzsk 0.015): early at 6 of 20 (control 20), median α 0.948, tier 1 0.853, trimmed 0.854, but volume −5.2 %. A pin, not a cause; superseded by `--noah-sac-exchanges`. |
| `noah_cdec_uf_seed1` (R2) | control, seed 1; stopped by hand at epoch 64 on its plateau | 0.8378 @60 | Same regime (riva 0.844, early 19 of 20), so the regime is systematic, not a seed draw. Tier 1 0.818 (control 0.794), volume −1.1 %: the seed pair behind the spreads the later bars use. |
| `noah_cdec_uf_sacx` | control + `--noah-sac-exchanges` | 0.8447 @58 (80) | The regime is gone without a pin: riva 0.0009, pfree 0.66, adimp 0.26 (control 0.016); early 5 of 20, α 0.945; tier 1 0.839, trimmed 0.844, volume −4.3 %. The physics of every later run. |
| `noah_cdec_uf_sacx_carry_aef` (A) | sacx + `--tbptt-carry relative` + `aef` inputs (17) | 0.8862 @62 (stopped by hand after 76) | The best 26-entity run until H: cdec 0.864, uf 0.928; tier 1 0.862, trimmed 0.869, volume −3.5 %; lztwm median 218 mm (sacx 4,378). The base of D–H. |
| `_aborted_noah_cdec_uf_sacx_carry_20260924` | B's first launch, started by a leftover queue while A was being scored | 0.4201 @0 | Killed after 2 epochs; relaunched as B. |
| `noah_cdec_uf_sacx_carry` (B) | A without `aef` (`physical_climate`, 26 inputs) | 0.8772 @62 (76) | The carry gives most of A's gain over sacx (0.8772 against 0.8447; uf 0.915 against 0.887). The embeddings add the rest in training and on the 60 nested gauges (B − A paired median ΔKGE −0.055, B higher at 20), but not on the 16 tier-2 arcs on cells no entity uses (B 0.457, A −0.047). |
| `noah_cdec_uf_sacx_carry_aef64` (C) | A with `aef64` (65 inputs) | 0.8904 @64 (78) | The highest selection and uf 0.942, but ORO 0.859 (A 0.896), trimmed 0.865 (0.869) and the unused-cell tier-2 arcs −0.243 (A −0.047): more embedding detail transfers worse. `aef` kept. |
| `noah_cdec_uf_sacx_flux_aef` (D) | A with `--tbptt-carry flux` | 0.8720 @60 (74) | Moves Shasta as intended (lzsk median 0.0058 against 0.0118, Jul–Nov volume −15.1 % against −27.5 %) but damps the Sierra flood peaks: ORO 0.815, 4-basin Q99.9 −25.4 % (A −5.5 %). Rejected. |
| `_failed_noah_cdec_uf_sacx_w2ft_aef_20260926` | E's first launch | — | Refused at start: the warm-start check expected statics that the `aef` inputs do not have; fixed in `train.py`, relaunched. |
| `noah_cdec_uf_sacx_w2ft_aef` (E) | fine-tune of A: `--init-from` A, `--tbptt-window-years 2 --graph-recompute-days 73`, lr 3e-4, 30 epochs, the final network scored | 0.8862 @0 (= A); 0.8778 @28 | Shasta Jul–Nov −15.1 % with the Sierra peaks kept (4-basin Q99.9 −7.1 %), but trimmed 0.847 (A 0.869) and no change on the 60 nested gauges (paired −0.001, E higher at 28). Rejected. |
| `noah_cdec_uf_sacx_w3_aef` (F) | A from scratch with 3-water-year windows (recompute 73) | 0.8704 @62 (76) | Fixes Shasta (Jul–Nov −3.5 %, volume +2.5 %, lzsk 0.0041) and uf 0.934, but loses the Sierra flood peaks: 4-basin KGE 0.789 (A 0.875), α 0.829, Q99.9 −27.2 %, five largest annual maxima −46.4 % (A −24.1 %); trimmed 0.854. Rejected; G and G2 try to keep its Shasta with A's peaks. |
| `_failed_noah_cdec_uf_sacx_w3_tp_aef_20260927` | G's first launch | — | Never trained: the machine crashed at launch. |
| `_stopped_noah_cdec_uf_sacx_w3_tp_aef_20260927` (G) | F + `--peak-lambda 1.0 --timing-lambda 24` (first forms) | 0.8292 @22 (stopped) | Stopped at epoch 22 (F 0.7998 at the same epoch) after a review of the two terms: the peak term gave every year an equal vote, so dry years carried it, and neither term was gated against near-zero flows. Replaced by G2. |
| `noah_cdec_uf_sacx_w3_g2_aef` (G2) | F + the revised terms, `--peak-lambda 1.0 --timing-lambda 3.5` | 0.8783 @54 (68) | Shasta held (Jul–Nov −0.5 %) but all four Sierra peak bars failed: KGE ORO/FOL/YRS/MKM 0.852/0.813/0.800/0.782 (A 0.896/0.893/0.863/0.846), α 0.864, Q99.9 −20.6 %, top five −40.8 %. A stayed the base; scoring stopped in tier 2 (no tier 2 or atlas). |
| `noah_cdec_uf_sacx_carry_px_aef` (H) | A + `--learn-pxtemp` + `--obs-mask` | 0.8946 @68 (90 epochs, no stop) | Adopted 2026-09-28; entry below. |
| `noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef` (H95) | H's recipe on 95 entities, family influence equalized | 0.8509 @92 (114), share-weighted family mean | Adopted 2026-09-29; entry below. |

- **`multifamily/noah_cdec_uf_sacx_carry_px_aef`** (run H) — A's recipe (26 entities, `aef`,
  `--noah-sac-exchanges --tbptt-carry relative`, `--epochs 90`, patience 10) plus `--learn-pxtemp
  --pxtemp-tau 1.0` (box −1 to 3 °C) and `--obs-mask data/cdec_fnf/fnf_daily_mask.csv`. Its target
  was Trinity (CLE), where A, with the rain/snow threshold fixed at 0 °C, was too wet in winter and
  too dry in May–Jun (WY1987–2018 daily volume bias Dec–Feb +40.1 %, May–Jun −10.0 %). Ran all 90
  epochs; selected epoch 68, selection 0.8946 on the masked target (A 0.8862 on the unmasked one;
  about 0.002 of the gap is the mask: +0.0017 on A's side, +0.0024 on H's). 48 steps were skipped for a non-finite gradient in epochs
  16–19 (1 / 6 / 23 / 18). Family mean cal KGE (masked daily target): cdec 0.874 (median 0.886) /
  uf 0.934.

  Bars against A, both on the masked target: Trinity passed, with CLE Dec–Feb +40.1 → +10.5 %,
  May–Jun −10.0 → −2.6 % and daily KGE 0.800 → 0.881. So did the basins with the same pattern (CLE,
  MKM, TRM, FOL, MRC): mean |Dec–Feb| + |May–Jun| 35.5 → 19.6 points, 4 of 5 better by more than 10.
  Guards: CDEC median KGE 0.879 → 0.886, 26-entity mean 0.888 → 0.895, uf mean 0.928 → 0.934, and
  the 60 nested gauges paired median ΔKGE +0.014 (90 % interval −0.012 to +0.044) with H higher at
  34 (all pass); ORO 0.896 → 0.838, a drop of 0.058 against a 0.040 seed spread (fail), with YRS
  −0.021, SHA −0.032, BND −0.011, NHG +0.057.

  Attribution: with PXTEMP reset to 0 on H's trained field, CLE goes back to Dec–Feb +57.5 % /
  May–Jun −45.9 % (KGE 0.766) and ORO to 0.903, so the learned threshold carries both the Trinity
  fix and the ORO loss; NHG's gain is not PXTEMP (0.922 with it reset). ORO's loss is in α (0.930 →
  0.849; Q99.9 −8.8 → −27.9 %). The threshold stores the precipitation of cool storm days (daily
  mean 0–4 °C) as snow, which moves ORO's winter volume to spring (Nov–Mar volume bias A +8 %, H
  −1 %) and damps moderate cool storms (five ORO events at 0.60–0.76 of the observed peak, 0.79–1.09
  with the threshold reset); the largest warm flood (1997-01-01) is unchanged. A uniform cap only
  trades one against the other: capped at 1.5 °C on H's field, CLE is back to Dec–Feb +30.0 % /
  May–Jun −22.7 % while ORO recovers only to 0.871. So the next step the bars prescribed for this
  outcome, a fresh run with a narrower box (−0.5 to 1.5 °C), was not run. Learned PXTEMP: median
  1.50 °C over entity-cell rows; basin medians 1.00 at SHA, 1.07 at BND, 1.5–2.1 at the other CDEC
  basins and 2.39 at CLE; no cell within 0.1 °C of a box edge.

  Not bars: the 8 arid USGS gauges outside the trained footprints, paired ΔKGE −0.139 and median
  |pbias| 37 → 76 %; the 18 DWR unimpaired sites, monthly KGE median WY1926–49 / 1950–84 / 1985–2014
  0.829 / 0.867 / 0.935 (A 0.829 / 0.893 / 0.925) and volume −9.8 / −7.9 / −3.9 % (A −7.2 / −5.1 /
  −1.1 %). CLE's cal-window volume bias is +6.4 %, inside the 10 % that would have called for a
  volume term. CalSim3 WY1950–84, out of sample at all 20 locations: tier-1 KGE mean 0.859 / median
  0.871 (anchors 0.865, arc sums 0.853), median |bias| 8.4 %, volume 29,995 vs 32,115 TAF/yr
  (−6.6 %); trimmed windows 0.863 / 0.877, volume −7.4 %. Tier-2 median 0.628 over 196 arcs: 0.654
  on the 155 arcs of trained entities, 0.108 on the 41 extrapolated arcs (25 on trained cells 0.124,
  16 on unseen cells −0.017).

  Decision: the user adopted H as the new base on 2026-09-28, with ORO's loss (α and the peaks of
  moderate cool storms, at ORO and less at YRS) recorded as a known cost. The candidate fix is a
  rain/snow partition that does not switch whole storm days, such as a snow fraction ramped over a
  temperature range; it changes the dPL snow step and has not been built.

- **`multifamily/noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef`** (run H95) — H's recipe
  (`aef`, `--noah-sac-exchanges --tbptt-carry relative --learn-pxtemp --pxtemp-tau 1.0 --obs-mask
  data/cdec_fnf/fnf_daily_mask.csv`, seed 0) on all 95 entities (69 `usgs_daily`, 17 `cdec_daily`,
  9 `uf_monthly`) with the footprint-area shares of `noah_cdec_uf_usgs_areaw`
  (`--mt-family-weight usgs=0.216,cdec=0.558,uf=0.226`), two weighting changes (`--mt-share-norm
  all`, `--mt-loss-ref usgs=0.7368,cdec=0.4202,uf=0.0980 --mt-loss-ref-power 0.5`), `--epochs 120`,
  patience 10.

  Why the weighting changes. Under the default `--mt-share-norm present` each chunk's loss is
  divided by the weight of the entities that chunk scores. USGS records start in WY1950 and the CDEC
  and uf targets in WY1985 or later, so the 35 WY1950–84 chunks score USGS alone at full weight, and
  the shares 0.216 / 0.558 / 0.226 came out about 0.57 / 0.34 / 0.10 over an epoch (both earlier
  95-entity runs trained this way). `all` divides by the weight of every entity of the run (daily
  86, monthly 9), so the shares hold as loss coefficients summed over the chunks (realized 0.234 /
  0.546 / 0.220). Coefficients are not influence, because the family losses are in different units:
  per entity, at the reference state below, usgs 0.7368 / cdec 0.4202 / uf 0.0980 (7.5 : 4.3 : 1).
  The frozen scale multiplies each family term by κ = L̄ / L_ref^p, with p = 0.5 and L̄ = 0.6179:
  usgs × 0.7198, cdec × 0.9532, uf × 1.9737, fixed for the run. L_ref are the per-entity loss levels
  of `noah_cdec_uf_usgs_areaw`'s best.pt re-scored with this loss; p = 0.5 is the power that brings
  each family's share of the optimizer step to its share at trained states (p = 1 would give uf
  about 1.4–1.5× its share). Selection keeps the nominal shares, never κ; the loss columns of
  `train_log.csv` are on the κ scale, so compare runs on KGE and selection only. The run's
  `provenance/` keeps the reference levels (`shares.txt`, `final_numbers.txt`) and the script that
  computed them (`levels_sim.py`).

  Training. Early stop at epoch 114 (11 stale selections); selected epoch 92, selection 0.8509 (the
  share-weighted family mean at the nominal shares, masked daily target; `noah_cdec_uf_usgs_areaw`
  logged 0.7610, and 0.7702 re-scored on the masked target: a confounded comparison, since the
  recipe differs). The `loss nan` on the epoch-114 line is the early-stop epoch, which runs a
  selection pass and no training chunk. Seven chunks were skipped for a non-finite gradient in
  epochs 8–9 (5 of 70 in epoch 9, over the flag line of 4 per epoch), in a gradient explosion
  through lzsk and lzpk at a few USGS creek cells with lzsk at its 0.003 floor. None followed after
  epoch 10, and selection had recovered by epoch 10 (0.7150). The weighting did what it was built
  to do: at best.pt the families' shares of the clipped, Adam-preconditioned step are 0.235 / 0.558
  / 0.207 against the coefficients 0.234 / 0.546 / 0.220 (pass). Loss-mass shares over epochs ≥ 10
  are 0.368 / 0.516 / 0.116, and the norm clip acted on 15.7 % of stepped chunks (epochs ≥ 5). The
  per-entity usgs : cdec loss ratio converged at 2.21, 1.26× the reference 1.75 (flagged above
  1.25: re-reference only for a next run).

  Skill (each entity's cal window, masked daily target). Family mean cal KGE: usgs 0.687 (median
  0.723) / cdec 0.880 (0.882) / uf 0.935 (0.940); H cdec 0.874 (0.886) / uf 0.934 (0.943);
  `noah_cdec_uf_usgs_areaw` usgs 0.592 / cdec 0.795 / uf 0.879 (confounded). Bars against H, with
  the preregistered one-seed spreads (cdec family mean 0.018, uf 0.005, single entity 0.04; a
  change inside its spread is unreadable):
  - uf, the success condition: 0.9351 against 0.9337, above the bar (0.924) but inside the spread,
    so unreadable; worst entity uf_13 0.959 → 0.923 (−0.035, inside 0.04).
  - cdec mean 0.880 against 0.874, unreadable. Guards: ORO 0.838 → 0.888 (α 0.849 → 0.901), SHA
    0.864 → 0.909 and BND 0.865 → 0.910 are readable gains; YRS 0.846 → 0.860 is unreadable; NHG
    0.924 → 0.873 is a readable fail (−0.051; α 0.988 → 0.905), back at A's 0.867, so H's NHG gain
    is lost. Also up beyond the spread: MKM +0.079, PNF +0.046.
  - Peaks, all pass: CDEC mean Q99.9 bias −10.3 % (H −12.2 %); flood-year
    (WY1986/95/97/98/2006/2011) Dec–Mar volume ORO −2.0 %, YRS −5.4 %, NML −2.1 % (H −7.4 / −10.5 /
    −14.3 %); the five moderate cool ORO storms of H's attribution at 0.65–0.92 of the observed peak
    (H 0.60–0.76).
  - USGS: family mean 0.687 against the old recipe's 0.592 (bar 0.562), higher at 54 of 69 gauges;
    confounded by the recipe.
  - Trinity, reported: CLE Dec–Feb +10.4 % (H +10.5), May–Jun −7.5 % (H −2.6), KGE 0.869 (0.881).

  CalSim3 WY1950–84 is in sample at the 16 tier-1 locations a trained creek covers (68 % of the USGS
  family's observation days fall in WY1950–84) and out of sample at UF2, UF4, UF10 and UF15. Tier-1
  KGE mean 0.857 / median 0.877 (anchors 0.867, arc sums 0.847), median |bias| 7.8 %, volume
  30,537 vs 32,115 TAF/yr (−4.9 %); trimmed windows 0.860 / 0.881, volume −5.8 % (H: 0.859 / 0.871,
  −6.6 %; trimmed 0.863 / 0.877, −7.4 %). The change against H, which trains no creek, is the same on
  the full and the trimmed windows (+0.001 on average, at most 0.025 at UF20), so no creek inflation
  of the WY1950–84 scores can be detected. By location (table below) the moves against H beyond
  0.04 are UF8 +0.043, UF3 +0.080 and UF14 +0.054 up, UF15 −0.071 and UF21 −0.042 down. Tier-2
  median 0.657 over 196 arcs (H 0.628): 0.684 on the 155 arcs of trained entities, 0.233 on the 41
  extrapolated arcs, of which 34 lie on cells the creeks trained on (0.347, partly in sample) and 7
  on cells no entity uses (−0.403; arcs of 2–12 TAF/yr). The 18 DWR sites, monthly KGE median
  WY1926–49 / 1950–84 / 1985–2014: 0.810 / 0.872 / 0.923 (H 0.829 / 0.867 / 0.935), about 1.8
  points wetter than H in each era (volume −8.1 / −6.1 / −2.1 %); NHG is down in all three eras
  (−0.123 / −0.081 / −0.108), ORO up in all three (+0.052 / +0.044 / +0.050).

  The learned PXTEMP differs from H's: median 1.18 °C over entity-cell rows (H 1.50); basin medians
  at or just below 0 °C at NHG (−0.24), CSN (−0.19) and seven of the nine uf basins (−0.19 to
  +0.01), 1.74 at SHA and 1.48 at BND; 130 of the 2,652 cells sit within 0.1 °C of the 3 °C cap
  (52 of them in the Shasta footprint), none in H.

  Decision: the bars' reading was to run the paired nominal arm (the same run and seed without the
  frozen scale) before deciding, because the NHG fail cannot be attributed to κ without it and the
  success condition (uf) was unreadable on one seed. The user adopted H95 as the new base on
  2026-09-29 without the nominal arm, a departure from that reading, with NHG's loss (α 0.988 →
  0.905) recorded as a known cost, as ORO's was for H. Neither the nominal arm nor a second seed has
  been run.

Tier 1 by location, CalSim3 WY1950–84, cycle spinup (from the two `tier1/tier1_metrics.csv`; in
sample for H95 at the 16 creek-covered locations, for H at none):

| location | reference | CalSim3 TAF/yr | KGE H | bias % | KGE H95 | bias % |
|---|---|---:|---:|---:|---:|---:|
| SHA: Sacramento R. at Shasta | FLOW-UNIMPAIRED | 6,323 | 0.856 | -13.2 | 0.878 | -11.6 |
| CLE: Trinity R. at Trinity Dam | FLOW-UNIMPAIRED | 1,416 | 0.645 | +28.0 | 0.653 | +26.9 |
| UF6: Sacramento R. near Red Bluff | FLOW-UNIMPAIRED | 9,210 | 0.851 | -12.5 | 0.878 | -11.5 |
| UF7: Sacramento Valley east-side minor streams | arc sum | 1,328 | 0.767 | -11.6 | 0.758 | -10.1 |
| UF8: Feather R. near Oroville | FLOW-UNIMPAIRED | 4,941 | 0.859 | -9.9 | 0.902 | -7.1 |
| UF4: Stony Creek at Black Butte | arc sum | 505 | 0.789 | -17.0 | 0.750 | -20.8 |
| UF11: American R. at Fair Oaks | FLOW-UNIMPAIRED | 2,922 | 0.943 | -2.0 | 0.946 | +1.6 |
| UF9: Yuba R. at Smartville | FLOW-UNIMPAIRED | 2,546 | 0.886 | -4.1 | 0.875 | -5.5 |
| UF9: Yuba R. at Smartville | arc sum, covered arcs | 2,515 | 0.924 | -2.9 | 0.912 | -4.3 |
| UF10: Bear R. near Wheatland | arc sum | 384 | 0.922 | -6.2 | 0.894 | -3.6 |
| UF3: Cache Creek above Rumsey | arc sum | 718 | 0.787 | -15.1 | 0.868 | -7.3 |
| UF2: Putah Creek near Winters | arc sum | 425 | 0.955 | -1.6 | 0.956 | -0.2 |
| UF13: Cosumnes R. at Michigan Bar | arc sum | 421 | 0.867 | -9.8 | 0.867 | -7.3 |
| UF14: Mokelumne R. at Pardee | arc sum | 811 | 0.875 | -5.5 | 0.929 | +2.9 |
| UF16: Stanislaus R. at Melones | FLOW-UNIMPAIRED | 1,231 | 0.922 | +0.2 | 0.893 | +8.2 |
| UF15: Calaveras R. at Jenny Lind | arc sum | 178 | 0.910 | -8.5 | 0.838 | -12.2 |
| UF18: Tuolumne R. at Don Pedro | FLOW-UNIMPAIRED | 2,001 | 0.951 | -2.3 | 0.924 | -3.0 |
| UF19: Merced R. at Exchequer | FLOW-UNIMPAIRED | 1,025 | 0.883 | +7.0 | 0.898 | +9.1 |
| UF20: Chowchilla R. at Buchanan | arc sum | 78 | 0.776 | -8.3 | 0.774 | -14.2 |
| UF22: San Joaquin R. at Millerton | FLOW-UNIMPAIRED | 1,879 | 0.853 | -13.3 | 0.823 | -12.9 |
| UF21: Fresno R. near Daulton | arc sum | 95 | 0.881 | -3.0 | 0.839 | -4.2 |

## CalSim3 rim-inflow product: dPL-CalSim, the sub-arc experiments and the adopted product (2026-09-28 → 10-01)

**Names.** From this section on, a run is called by what it trains on. The sections above keep
the letters they were written with.

| name | was | run folder | trains on |
|---|---|---|---|
| **dPL-26** | run H | `noah_cdec_uf_sacx_carry_px_aef` | 26 entities: CDEC daily and DWR-unimpaired monthly |
| **dPL-95** | H95 | `noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef` | dPL-26's entities + 69 USGS gauges |
| **dPL-CalSim** | S1 | `noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef` (named `s1_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef` when the run was made and until 2026-10-01; `provenance/RENAMED.txt`) | dPL-95's entities + 64 CalSim3 arcs, with WY1976–85 held out of every family |

dPL-CalSim's folder follows the naming of the training-program section (the families, then the
recipe) with three new parts: `_cs64` = the 64 CalSim3 arcs as a trained family (`--calsim-arcs
train_default`), `_ho7685` = `--holdout-wy 1976-1985`, `_ufx` = `--uf-train-start 1949-10-01`. The
files in the run's `provenance/` keep the old folder name: they are the record of the run as it was
made.

The **CalSim3 rim-inflow product** is dPL-CalSim's flow on the 196 rim arcs with the **share
model** on the arcs of the multi-arc systems (`sacsma.dpl.calsim_product`).

**Why.** CalSim3 takes monthly inflows on 196 rim arcs. The dPL has to supply them with skill
against the CalSim3 arc series on a decade no fit has seen, and with a climate response that stays
the dPL's own. dPL-95 could not be scored that way (50 of its USGS gauges trained inside any
candidate holdout), and the dPL's tier-2 skill on the small arcs was low (dPL-95 median 0.657 over
196 arcs). The program had two parts: a holdout-masked retrain (dPL-CalSim) and a series of
preregistered experiments on how to turn the dPL's arc flows into the product.

**Ground rules (user).** Holdout WY1976–85 in every family and every fit; nothing before WY1950;
the CalSim3 INFLOW series is the truth on every arc-month, with no tier weighting. No WGEN test:
the climate response is trained on ΔT {0, 1, 3, 4} °C × P {0.85, 1, 1.15} (11 points besides the
base) and validated on points no fit saw. A correction's response is scored against the
uncorrected dPL at the same point: dV (volume change, %) and dAJ (change of the April–July share,
pp), summarized over units per point; full gate = median ≤ 1 and p90 ≤ 3 pp, half gate = half
those. Each experiment has a prereg file frozen (sha256, read-only) before any of its holdout or
validation numbers existed; the holdout and the validation points are read once, at the end.
ET-observation products are in nothing below.

**dPL-CalSim** (branch `dpl/s1-calsim-retrain`; prereg, bars, readout and deviations in the run's
`provenance/`). dPL-95's recipe plus: `--holdout-wy 1976-1985`; the UF targets back-extended to
WY1950 (`--uf-train-start 1949-10-01`); 64 CalSim3 arcs as a fourth trained family
(`--calsim-arcs train_default`, absolute-volume loss); family shares re-solved so the realized
coefficients equal the footprint-area shares (usgs .1967, cdec .5070, uf .2053, calsim .0911),
selection at the same shares; seed 0. It trains on the region store after the second ×10 precip
correction (main `c5fae09`); every dPL-26 / dPL-95 comparator is on the store before it.

- *Interruption.* A thermal hibernation stopped the run in epoch 91 (2026-09-30 14:57). It was
  resumed from `last.pt` (epoch 90) with the same flags + `--resume`; the RNG stream and the EMA
  shadow are not restored.
- *Readout* (best epoch 116 of 120, flagged "not converged"; selection 0.8337, three-family
  0.8456 against dPL-95's 0.8522). Passes: the mechanism bars; the 64 trained arcs on training
  years (mean 0.715, median 0.771); **the primary holdout bar, arcs WY1976–85: 0.727 (bar 0.691;
  dPL-26 0.661)**; uf holdout 0.895 (bar 0.873); the USGS and second arc holdout guards.
  Unreadable within the spreads: the cdec family and its named basins. Readable guard fails:
  NHG 0.827 (bar 0.833; dPL-95 0.873, dPL-26 0.924) and the uf family on WY1986–2014, 0.896
  (bar 0.926; uf_03, uf_04, uf_07, uf_20 below their bars). By era the uf median is 0.913 on
  WY1950–84 (dPL-26 0.867) and 0.912 on WY1986–2014 (dPL-26 0.937): the back-extension buys the
  earlier era at the later one's cost.
- *Decision.* The frozen reading called for a no-arcs control after a readable guard fail. The
  user adopted dPL-CalSim as is on 2026-09-30 without it; NHG and uf WY1986–2014 are recorded
  as known costs (`provenance/DEVIATIONS.txt`).

**Tier-2 passes for the experiments.** `calsim_tier2 --scenarios NAME=DT:PS,...` runs several
climate points in one batched forward (identical to single passes at window 512). dPL-CalSim was
run at the base, the 11 training points, the 5 points of the first validation set (ΔT 2; P 0.92,
1.08 and their combinations) and a second, fresh validation set (ΔT 2.5; P 0.95, 1.05 and their
combinations). The passes the product needs are local in `<run>/tier2_scenarios/` (regenerable).

**Units.** 18 *systems*: 11 multi-arc closure groups, whose 139 arcs are the *share arcs*, and 7
single-arc systems (Shasta, Trinity, Merced, San Joaquin, New Hogan, uf_02, uf_20; Shasta is
anchored on SHA, not Bend Bridge). 50 *non-anchor arcs* have no closure group (the Bend
Bridge-only arcs and I_WKYTN among them). 139 + 7 + 50 = 196.

**The experiments.** Holdout numbers are the median monthly KGE over units, WY1976–85.

| experiment (prereg) | dPL | what was tried | result | outcome |
|---|---|---|---|---|
| arc correction v1 → v3 (`PREREG_CORRECTION_v1/2/3`) | dPL-26 | per arc, four constant factors on the routed runoff parts (quick, interflow, supplemental, primary), bounded [0.5, 2], with response rows; v2 trained the response on a delta grid, v3 split it into training and validation points | all arcs 0.782 / 0.775 / 0.771 (v1 / v2 / v3); v3 passes the full gate at all 5 validation points, worst p90 0.68 pp | frozen record on dPL-26; not refit on dPL-CalSim |
| LSTM hybrid (`PREREG_HYBRID_ALT_v1`) | dPL-26 | daily LSTM on the dPL states, daily + monthly loss on CDEC, UF and USGS, response loss | its first stage passes; the second fails its out-of-fold gate (uf 0.771 vs dPL-26 0.826, cdec 0.865 vs bar 0.895, usgs 0.483 vs 0.423) | stopped before any member run; validation points never read |
| sub-arc pipeline (`PREREG_SUBARC_S1`) | dPL-CalSim | system flow: gradient-boosted trees on log(CalSim3 / dPL) with water-year closure to a constant volume factor; non-anchor arcs: the same per arc; share arcs: the share model, anchored on the corrected system flow | systems 0.922 → 0.891; non-anchor 0.604 → 0.657; share 0.699 → 0.815; all arcs 0.690 → 0.806; full gate passes | the system correction loses to the dPL; the share model is kept |
| daily LSTM on the dPL (`PREREG_LSTM_S1`, two amendments) | dPL-CalSim | one pooled daily LSTM (dPL flow, precipitation, temperature), CalSim3 monthly targets, response penalty in training, systems ¾ of the loss; residual mode (dPL × bounded factor) and direct mode (flow outright); the share model re-fitted on the dPL and on the LSTM system flow | table below | frozen rule: the dPL at the systems, the residual LSTM at the non-anchor arcs, the share model on the dPL system flow (all arcs 0.827) |
| **user adoption, 2026-10-01** | dPL-CalSim | the dPL everywhere except the share arcs | all arcs **0.813** | **the product** |

`PREREG_LSTM_S1`, holdout WY1976–85 (median / mean / p10 of the unit KGEs):

| units | dPL-CalSim | volume factor | LSTM residual | LSTM direct (report only) | share model |
|---|---|---|---|---|---|
| 18 systems | **0.922** / 0.913 / 0.864 | 0.886 / 0.886 / 0.792 | 0.913 / 0.899 / 0.829 | 0.891 | – |
| 50 non-anchor arcs | **0.604** / 0.482 / 0.006 | 0.673 / 0.609 / 0.165 | 0.719 / 0.561 / 0.034 | 0.687 | – |
| 139 share arcs | 0.699 / 0.616 / 0.226 | – | – | – | **0.834** / 0.800 / 0.641 (dPL system flow); 0.831 (LSTM system flow) |

Bold = the adopted product. The volume factor is one constant per unit (training-year
Σ CalSim3 / Σ dPL); it keeps the dPL's response exactly and its timing.

- *Why every system correction loses.* On training years (out of fold) the corrections beat the
  dPL (dPL 0.898, factor 0.952, residual LSTM 0.955). On the holdout they do not, because the
  CalSim3 / dPL volume ratio drifts by decade (New Hogan 1.26–1.31 in the 1950s–70s, 1.02 in the
  1980s; Trinity 0.74–0.85, then 0.99) and WY1976–85 sits near 1.0 for several systems. A factor
  fitted on the other years moves those systems away from CalSim3. The factor beats the dPL in 5
  of the 7 training blocks and loses in WY2000–15 and on the holdout.
- *Share arcs.* The share model is the gain: 0.699 → 0.834 against the dPL arcs, with the dPL's
  own shares inside the closure at 0.776. Each system's arcs sum to the dPL system flow over
  every water year. Inside the year their sum differs from it by 2.5 % of the volume, and it
  scores the same against CalSim3 (11 multi-arc systems, holdout median 0.924 against the
  dPL's 0.928, mean 0.926 against 0.924).
  Timing improves with it: seasonal mismatch 11.6 % → 7.3 %, monthly r 0.924 → 0.948.
- *Non-anchor arcs.* The frozen rule adopted the residual LSTM on its median (0.719). The user
  chose the dPL (0.604) instead, after the holdout was read: a departure from the frozen
  adoption rule. Neither correction changes these arcs' timing (seasonal mismatch 10.9 % for the
  dPL, centre of timing 0.4 month early).
- *Direct mode.* Report-only by the second amendment. It trails the residual mode on both kinds
  of unit and fails the gates at most candidates for the non-anchor arcs.
- *Validation* (second, fresh set). The full gate passes for every model; for the adopted share
  model the worst p90 is 0.42 pp (dV) and 0.64 pp (dAJ).

**The adopted product (user, 2026-10-01).** dPL-CalSim's own flow at the 18 systems and the 50
non-anchor arcs; the share model (μ 0.03) at the 139 share arcs. All 196 arcs on WY1976–85:
median 0.813, mean 0.722, p10 0.425, 7 arcs below 0 (the dPL on every arc: 0.690 / 0.592 / 0.139,
13 below 0; the frozen-rule product: 0.827 / 0.742 / 0.537, 6 below 0). The climate response is
the dPL's own except at the share arcs, where it is held to the dPL's within the gate and each
system's water-year volume is the dPL's exactly.

`python -m sacsma.dpl.calsim_product fit <run>` refits it from the run's tier-2 passes and
writes `<run>/calsim_product/`: `share_model.pt`, `rim_inflow_monthly.csv` (196 arcs, WY1950–2015,
TAF), `product_metrics.csv`, `share_selection.csv`, `response_gate.csv`, `product_info.json`.
`calsim_product apply <run> --tier2 <pass>` gives the product for any other tier-2 pass with
runoff parts. The module's fit reproduces the experiment: the same selection table, and the
share-arc flows to 5e-8 relative.

**Against the original SAC-SMA** (Wi & Steinschneider GA optima mapped to the arcs,
`artifacts/calsim/compare/monthly_calsets.csv`; common units, WY1976–85). That decade is inside
the 11obs and 9unimp calibration periods and outside 15cdec's.

| set (units) | systems: SAC-SMA | systems: dPL-CalSim | share arcs: SAC-SMA | share arcs: SAC-SMA + QMAP | share arcs: share model |
|---|---|---|---|---|---|
| 15cdec (10 systems, 115 arcs) | 0.896 | 0.931 | 0.653 | – | 0.814 |
| 11obs (9, 109) | 0.885 | 0.933 | 0.682 | 0.813 | 0.819 |
| 9unimp (7, 21) | 0.903 | 0.904 | 0.731 | 0.880 | 0.870 |

**Temperature response at Shasta and Oroville** (uniform warming on every day, precipitation
unchanged, WY1950–2015).

| | +1 °C | +2 °C | +3 °C | +4 °C |
|---|---|---|---|---|
| volume, Shasta: SAC-SMA 11obs / 15cdec | −3.1 / −3.6 % | −6.0 / −7.0 | −9.0 / −10.4 | −11.8 / −13.7 |
| volume, Shasta: dPL-CalSim | 0.0 % | −0.3 | −0.9 | −1.7 |
| volume, Oroville: SAC-SMA 11obs / 15cdec | −4.4 / −3.0 % | −8.7 / −5.9 | −13.0 / −8.9 | −17.2 / −11.8 |
| volume, Oroville: dPL-CalSim | −0.1 % | −0.5 | −1.2 | −2.0 |
| April–July share, Shasta: SAC-SMA 11obs / 15cdec / dPL-CalSim | −1.2 / −2.9 / −3.0 pp | −2.0 / −4.6 / −5.0 | −2.5 / −5.6 / −6.2 | −2.8 / −6.3 / −6.9 |
| April–July share, Oroville: SAC-SMA 11obs / 15cdec / dPL-CalSim | −5.3 / −4.1 / −5.1 pp | −8.5 / −6.3 / −8.3 | −10.4 / −7.4 / −10.2 | −11.5 / −7.9 / −11.2 |

No warmed VIC run is on the machine; the only temperature-only VIC pair is the detrending
experiment (WGEN Product A baseline minus Livneh-unsplit, WY1922–49, basin-mean ΔT 0.84 °C at
Shasta and 1.24 °C at Oroville): VIC volume +0.6 % / −0.8 % against the original SAC-SMA's
−3.9 % / −7.0 % in the same experiment, with April–July share changes of −2.5 / −8.7 pp (VIC) and
−1.7 / −8.0 pp (SAC-SMA). The dPL's volume response is close to VIC's and several times weaker
than the Hamon-PET SAC-SMA's; the timing shifts agree across the three.

**Where it lives.** The run folder holds dPL-CalSim, its chain, its provenance and the product.
The experiments' own code, prereg files, hash sets and reports (the correction, the LSTM hybrid,
the sub-arc pipeline, the daily LSTM, the comparison scripts) are local-only, under
`tmp/calsim_hybrid/`; the share model is the one piece of them that is in the package.

**Open.** A longer product period (CalSim3 starts in WY1922, the product in WY1950); VIC on
warmed scenario weather for a like-for-like response curve; the uniform perturbations are a
placeholder for the WGEN daily scenario weather; the holdout WY1976–85 has now been read by
three experiments, so a further design choice read against it is not a clean test.

## Open items
- **Rename (2026-07-21):** the climate-adaptive physics and hybrid family
  canonicalized 2026-07-19 as `noah_ca` / `hybrid_base` / `hybrid_dtdp` are
  promoted to the plain top-level names `noah` / `hybrid` / `hybrid_dt` — they are
  now THE canonical noah physics and hybrid family, full stop, not a parallel
  climate-adaptive track alongside a still-current frozen one. The prior
  frozen-noah-basis generation (old `noah` 0.767/0.799, old `hybrid` 0.917/0.869,
  `hybrid_pet_dt` 0.916/0.864) and `hamon_dense` move to
  `superseded/{noah_noca, hybrid_noca, hybrid_dt_noca, hamon_dense}` to show the
  chain of improvement. `lstm` is unchanged. Every hardcoded consumer path
  (`sacsma/dpl/*.py`, `cli.py`) was repointed in the same pass; `fidelity/` moved
  to `noah/fidelity/`; top-level comparison CSVs/PNGs consolidated into
  `figures/` (dropping the `noah_ca`/`hybrid_pet_dt` infixes: `noah_ca_summary.png`
  → `noah_summary.png`, etc.). The `hybrid`/`hybrid_dt` checkpoints' baked
  `physics_csv`/`sim_cache` pointed at a since-deleted `testing/` scratch path
  from before their 2026-07-19 promotion (a latent bug, not caused by this
  rename) — patched to the canonical `noah/` files and verified reloadable.
  README.md and the Sphinx docs (`docs/`) were updated to the new names in the
  same pass; this file's history below keeps the names as originally written.
- **Figure-label follow-up (2026-07-21):** the rename above moved paths and
  file names but left several figure-internal legend/title strings on the old
  labels (`climatology.py`/`forcing_sensitivity.py` still said "Hybrid PET+dT";
  `noah_ca_hybrids.py`'s `hybrids/`/`noah_regimes/` column titles still said
  "noah_ca (physics)"/"base hybrid"/"dt·dp hybrid"/"pure LSTM"). Canonicalized to
  the plain model names throughout (`Noah`, `PT`, `Hamon`, `Hamon (dense)`,
  `Hybrid`, `Hybrid DT`, `LSTM`); cache-filename derivation in `climatology.py`/
  `forcing_sensitivity.py` was decoupled from the label text first (`_FROZEN_TAG`/
  `_MODEL_TAG`) so the rename didn't orphan the on-disk caches. All three figure
  sets (`cdec15_climatology_*`, `cdec15_forcing_sensitivity_*`, `hybrids/`/
  `noah_regimes/`/`noah_summary.png`) regenerated fully from cache (no GPU
  recompute — every (dp,dt) grid point and frozen/ensemble sim was already on
  disk). Separately, `superseded/hybrid_progression.{png,csv}` (the frozen-noah
  family's PET/λ ablation) was retired outright rather than kept, since its
  PET-only middle rung has no counterpart in the current family; a new
  `figures/hybrid_progression.{png,csv}` replaces it with the current chain's
  own progression (per-basin validation skill + the pooled warming-response
  curve for `noah` → `hybrid` → `hybrid_dt`).
- **Figure cleanup, round 2 (2026-07-21):** retired the redundant/superseded
  comparison exhibits and dropped stale naming prefixes now that `figures/`
  carries the canonical set. Removed: `dtdp_response_{cs8,l0.3}` (the CSV pair +
  per-basin folders; the λ-screening question that motivated them is answered,
  and `hybrids/` covers the current family's response comparison) and
  `dtdp_lambda_compare.{png,csv}` (the screening figure itself; the underlying
  `dtdp_response.make_lambda_compare` stays as a reusable function, just
  unwired); `compare_val_kge.png` (redundant with `hybrid_progression.png`'s
  skill panel — `hybrid/evaluate.py`'s `_dumbbell` helper was dropped along with
  it; `compare_ga_dpl_hybrid.csv` is kept, it still has independent value).
  Renamed: `noah_regimes/` → `hybrid_regimes/`, `noah_summary.png` →
  `hybrid_summary.png`, `adaptive_physics_metrics.csv`/`adaptive_physics/`/
  `adaptive_physics_regimes/` → `noah_climate_adaptive_metrics.csv`/
  `noah_climate_adaptive/`/`noah_climate_adaptive_regimes/` (the
  `adaptive_physics.py` module and its function names are unchanged — only the
  output paths moved), and every `cdec15_climatology_*`/`cdec15_forcing_sensitivity_*`
  figure dropped the `cdec15` prefix (`climatology_*`/`forcing_sensitivity_*`).
  All are pure renames/deletions — no recompute, no content change. Every
  hardcoded consumer (`sacsma/dpl/*.py`, `cli.py`) and doc reference
  (`appendix_b.md`'s B.6 figure list renumbered, `appendix_c.md`, `part2.md`
  Figures 5-6) updated in the same pass.
- **Figure cleanup, round 3 (2026-07-21):** `figures/hybrids/` (round 2 named it
  that to avoid colliding with the new `hybrid_regimes/`) → `figures/hybrid/`,
  parallel to `hybrid_regimes/`. Dropped the base (untagged) `dtdp_response/` +
  `dtdp_response_metrics.csv` too — `hybrid`/`hybrid_regimes` already cover the
  current family's response comparison, and the frozen-noah-basis 3-model
  comparison this represented has no live use now that the `_cs8`/`_l0.3`
  screening it supported (round 2) is gone; `dtdp_response.py` itself is
  unchanged and still supplies the shared response-window/regime-aggregation
  engine (`DP`/`DT`/`METRICS`/`REGIMES`/`_aggregate_regime`/`_eval_mask`) that
  `noah_ca_hybrids.py` and `adaptive_physics.py` both import. Also caught a gap
  the earlier label pass missed: `adaptive_physics.py`'s own 2-column
  `[noah | noah_ca]` physics-only figures (`figures/noah_climate_adaptive{,_regimes}/`)
  still had the old labels baked into the column titles and suptitle even after
  the folder itself was renamed — canonicalized `NOAH`/`CA_ADAPTIVE` to
  `"Noah"`/`"Noah (climate-adaptive)"` and regenerated (cache-hit, no recompute;
  81/81 (dp,dt) points were already on disk in both `_adaptive_cache` and the
  shared `testing/dtdp_cache`).
- Canonical set (2026-07-19, names above updated 2026-07-21): the physics ladder `hamon_dense`, `hamon`, `pt`,
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
- **Multifamily bases (2026-09-29).** H95 is the base dPL; its paired nominal arm, a second seed and
  a re-referenced κ (its usgs : cdec loss ratio ended at 1.26× the reference) have not been run. The
  CalSim-hybrid work stays on H for now: H's only overlap with the WY1976–85 holdout chosen for that
  work is the uf family in WY1985, while 68 % of the observation days H95's USGS creeks train on
  fall in WY1950–84. H95's recipe enters with the planned CalSim dPL retrain, which masks WY1976–85
  in every family and adds the observed CalSim3 rim arcs as a fourth, monthly family.
