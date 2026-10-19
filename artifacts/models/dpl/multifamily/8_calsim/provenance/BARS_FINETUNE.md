# Bars for the paired fine-tune (written 2026-10-16, before any run)

**The pair.** Both arms start from dPL-CalSim's `best.pt`: `--init-from`, lr 3e-4, 1 warmup
epoch, 15 epochs, CPU selection with patience 20. Both train on the same cleaned targets:
- the CDEC daily screen and the location masks, applied after the normalizers;
- the depth fixes for I_MFM008 and I_SFM005;
- Kings 11218501;
- the loss references re-measured on the cleaned targets (`levels_ft.py`).

| | Control (C) | Treatment (T) |
|---|---|---|
| Recipe | dPL-CalSim's | C plus three changes (next row) |
| Changes | — | CDEC monthly companion from WY1950 (cdec and cdecm 0.2754 each; companion screened, plus R3 at CLE, ISB, SCC E1+E2 and MIL E2); dry-season term with λ 0.1132 (10 % of the loss at dPL-CalSim's field; fixed denominators); flow-scaled riparian ET |
| `--mt-loss-ref` | usgs 0.6006, cdec 0.2827, uf 0.1088, calsim 0.3221 | same, plus cdecm 0.1173 |
| Scales | usgs 0.7325, cdec 1.0678, uf 1.7215, calsim 1.0004 | usgs 0.6653, cdec 0.9698, uf 1.5635, calsim 0.9086, cdecm 1.5053 |

dPL-CalSim (D) is reported beside both arms for context.

## Data and window

- **In sample:** WY1950–75 and WY1986–2018. No value of WY1976–85 is read; the fine-tune spends
  no held-out read.
- **Pre-1950:** WY1922–49, reported only.
- **Monthly CDEC FNF** (`fnf_monthly.csv`) at eight stations:
  - SHA (SIS), ORO (FTO), YRS, FOL (AMF): the Sacramento four;
  - MIL (SBF), TLG, MRC, NML (SNS, over 996 mi²): the San Joaquin four.

  Depth is over the registry area. The months in `fnf_monthly_screen.csv` and
  `forcing_era_mask.csv` are left out.
- **Flows and precipitation:**
  - Each arm's selected `best.pt` is run on the engine (CPU, the trained field, whole record).
  - Basin precipitation is Livneh, area-weighted over the entity's cells.

## Primary bars (T, absolute; C and D reported)

1. **Elasticity.** The Sacramento four's Aug–Sep elasticity is **at or below 1.0**. It is
   d ln(V_Aug-Sep) / d ln(P_annual), by OLS over the in-sample water years, on the four's summed
   volume and volume-weighted precipitation. The proposal measured D at 1.29 and the observed at
   about 0.6. D's value under this script is recorded below before any arm is read.
2. **Dry-third bias.** The Sacramento four's Aug–Sep bias in the driest third of years (terciles of
   the four's annual precipitation) is **within ±20 %**. SHA and ORO must each be **within ±25 %**.
   The proposal had D at about −30 %.
3. **San Joaquin recession.** The San Joaquin four's Sep/Jun ratio (four summed, median over
   in-sample years) is **at or below 0.06**. The proposal had D at 0.072, observed 0.040.

## Guards (T − C, paired over entities)

Scored with the multi-timescale evaluator's `metrics.csv` on the cleaned targets, in sample.

1. **CDEC daily KGE:** paired median **≥ −0.005**; SHA and ORO each **≥ −0.02**.
2. **USGS, UF and arc KGE:** paired medians **≥ −0.005**, each family separately.
3. **Peaks and tension water:**
   - Sierra flood peaks: per Sierra CDEC outlet, the mean of the 5 largest in-sample daily peaks
     over observed. T's median is **within 3 points** of C's.
   - `lztwm`: T/C, median over cells, **at most 1.4×**.
4. **Climate response:** +3 °C annual runoff and April–July change over the 15 CDEC watersheds (the
   ladder study's metric). T must be **within 1 point** of C's annual change and **within 2
   points** of its April–July change.

## Reading the result

- **All primary bars pass and every guard holds:** I recommend the full retrain on the treatment
  recipe; the user decides.
- **A guard fails or a primary bar is missed:** report and stop. Nothing is retrained without the
  user.
- **Reported, not bars:**
  - pre-1950: the Sacramento four's 1924 and 1931 annual and Aug–Sep bias, and WY1929–34;
  - C against D on the old and on the cleaned targets (what the cleaning alone did);
  - the dry-season term by family;
  - the in-sample monthly KGE of the eight stations.
- **Not part of the fine-tune:** the share model and product gates, and the WY1976–85 read. Both
  belong to the full retrain.

## D under this scoring (recorded before any arm is read)

Recorded 2026-10-16 ~22:20 by `score_ft.py run D` (`score/D/bars.json`, `score/run_D.log`). The
control arm was at epoch 2 of 15 and no arm had been scored.

**How the script scores** (fixed here, on D only):
- **Engine run.** The trained field of the 15 CDEC watersheds on the CPU engine, historical
  Livneh.
  - The bars, peaks and pre-1950 use the product's spin-up: WY1916–25 × 20 from cold, then
    1915-10-01 on.
  - The response uses cold-start runs over the whole record at +0 and +3 °C, as in the ladder
    study.
  - **Checks:** D's cold runs match its cached full-domain flows to within 4e-14 mm/day. Its
    response matches the ladder's tracked `response.csv` (−0.906 / −29.793 %).
- **Groups of four: member-drop.** Each year is taken over the members observed in every month of
  the quantity, as depth over their area. Sim, obs and precipitation use the same members, and
  the terciles use all four members' precipitation.
  - The first rule (drop the year when any member month is screened) removed 10 of the 59
    Sacramento years: 1961, 1966, 1988, 1991, 1992, 2004, 2007, 2008, 2012 and 2015. FOL's
    screened Aug/Sep months are zero-FNF months in the drought summers, so it took 1988, 1991,
    1992, 2008, 2012 and 2015 out of the dry third.
  - It also kept only 35 San Joaquin years.
  - `score_variants.py` compares the rules on D (`score/variants_D.log`).
- **Scorer check.** On the unmasked targets the script reproduces the proposal's D: elasticity
  1.290 (obs 0.608), dry third −30.5 %, Sep/Jun 0.0724 (obs 0.0402), 1924 −30/−63 %, 1931
  −28/−60 %.
- **Sierra peaks.** The program's Sierra four: ORO, FOL, YRS, MKM. Each outlet's peak value is the
  improvement plan's top-5 ratio (sim over obs, the five largest observed annual maxima) on the
  cleaned daily targets.
- **`lztwm`.** T/C paired on the (entity, cell) rows of `params_dpl.csv`, then the median over
  cells of the per-cell mean. 7 of D's 2,715 cells carry two values across entities.
- **C against D.** Each run's `sim_daily.npz` is rescored on the other target set. Rescoring D on
  its own targets reproduces its `metrics.csv` KGE to within 1.7e-8 (`check_rescore.py`).

**D's values:**

| Quantity | D | Observed | Bar on T |
|---|---|---|---|
| Sacramento-4 Aug–Sep elasticity (59 years, 10 short a member) | **1.202** | 0.510 | ≤ 1.0 |
| Sacramento-4 dry-third Aug–Sep bias (20 years) | **−31.4 %** | | ±20 % |
| SHA, dry third | **−31.0 %** | | ±25 % |
| ORO, dry third | **−44.0 %** | | ±25 % |
| Sacramento-4 mid / wet third | −12.9 / +4.0 % | | |
| San Joaquin-4 Sep/Jun (59 years, 24 short a member) | **0.0708** | 0.0402 | ≤ 0.06 |
| Sierra-4 top-5 ratio (ORO / FOL / YRS / MKM; median) | 76.0 / 78.8 / 67.9 / 77.6; **76.8** | | T within 3 points of C |
| `lztwm`, median over cells | 256.7 mm | | T/C ≤ 1.4 |
| +3 °C annual / Apr–Jul change, 15 CDEC mean | **−0.91 / −29.79 %** | | T within 1 / 2 points of C |

The in-sample monthly KGE of the eight stations (screened):

| SHA | ORO | YRS | FOL | MIL | TLG | MRC | NML | Sacramento-4 | San Joaquin-4 |
|---|---|---|---|---|---|---|---|---|---|
| 0.961 | 0.936 | 0.847 | 0.962 | 0.951 | 0.857 | 0.896 | 0.914 | 0.939 | 0.955 |

The pre-1950 Sacramento-4 bias, annual / Aug–Sep:

| 1924 | 1931 | WY1929–34 | WY1922–49 |
|---|---|---|---|
| −30 / −63 % | −28 / −60 % | −25 / −45 % | −14 / −24 % |

D's in-sample KGE medians, on the old targets and then the cleaned:

| Family | Old (n) | Cleaned (n) |
|---|---|---|
| CDEC daily | 0.8879 (17) | 0.8867 (17) |
| USGS | 0.7157 (69) | 0.7172 (60) |
| UF | 0.9262 (9) | 0.9336 (5) |
| Arcs | 0.7322 (64) | 0.7210 (60) |

**What the bars ask of T, against D:**
- elasticity down 0.20;
- the dry third up 11 points, ORO's up 19;
- Sep/Jun down 0.011.

## Reading the full retrain (written 2026-10-17 ~15:50, before it finished)

`7_calsim_ds` (R) was at epoch 108 of 120 and no value of it had been scored. It was launched
whatever the fine-tune's bars said (user, 2026-10-17 00:29). It is read with
`score_ft.py compare-full D 7_calsim_ft15_clean 7_calsim_ft15_ds 7_calsim_ds` ->
`score/compare_full.txt`:
- **Primary bars:** the same five, on R; D, C and T reported.
- **Guards:** the same, on R − D. R and D both train from scratch for 120 epochs, so D is
  the run R would replace. D's KGE is its `sim_daily.npz` rescored on the cleaned targets
  (`score/D/kge_clean.csv`), and R's `metrics.csv` is checked by rescoring R's
  `sim_daily.npz` on them.
- **Caveats:**
  - R trained on the cleaned targets and D did not, so R − D leans toward R.
  - R − C and R − T are reported beside it. C is D fine-tuned 15 epochs on the same targets.
  - A from-scratch run also differs from D by run-to-run noise. No second seed exists to size
    it.
- **Dry run.** With T standing in for R (`score/compare_full_dryrun.txt`), the script
  reproduces the fine-tune's T − C medians exactly. Against D on the cleaned targets, T's
  guards all hold: CDEC +0.0021, USGS +0.0098, UF +0.0004, arcs +0.0042, peaks −0.6 points,
  lztwm 1.27×, +3 °C +0.47 / −0.35 points.
- **Not part of this read:** the share-model refit, the product, the benchmark, the ladder
  study and the WY1976–85 read. Each waits for the user.

## Peak fine-tune (user 2026-10-17: "let's try a peak fine-tune"; written before launch)

**P** = `7_calsim_ds_ft15_pk` starts from R's `best.pt`: 15 epochs, lr 3e-4, 1 warmup epoch,
every option R's (`launch_peak.py`, checked by `check_peak_cfg.py`). It makes one change: the
daily CDEC record's share goes back to dPL-CalSim's 0.5508, beside the monthly companion's
0.2754, with the other shares unchanged
(`usgs=0.2366,cdec=0.5508,cdecm=0.2754,uf=0.1226,calsim=0.0900`). Renormalized, that is usgs
0.186, cdec 0.432, cdecm 0.216, uf 0.096 and calsim 0.071. The daily CDEC record then weighs
against every other family as in dPL-CalSim. The other families each lose 22 % of their share,
and the dry-season term's CDEC weight (cdec + cdecm) rises from 0.551 to 0.648.

R lost its peaks (Sierra-4 top-5 76.8 -> 66.6) and its daily CDEC skill (FOL, ORO, NML, YRS,
CLE; variability). R had halved the daily share to make room for the companion, and P tests
that as the cause.

Read with `score_ft.py compare-full D 7_calsim_ft15_clean 7_calsim_ds 7_calsim_ds_ft15_pk
compare_peak.txt DCRP` (columns D, C, R, P; guards on P − D, as R's were).

**P succeeds if both hold:**
1. **The peaks come back.**
   - Sierra-4 top-5 median within 3 points of D (at least 73.8).
   - CDEC daily paired median P − D at least −0.005.
   - SHA and ORO each P − D at least −0.02.
2. **R's gains are kept.**
   - The four Sacramento bars still pass: elasticity ≤ 1.0, dry third within ±20 %, SHA and
     ORO within ±25 %.
   - USGS, UF and arcs P − D each at least −0.005.
   - lztwm P/D at most 1.4×.
   - +3 °C within 1 point (annual) and 2 points (Apr–Jul) of D.

The San Joaquin Sep/Jun ratio is reported only; no run has met it.

**No control arm.** R fine-tuned 15 more epochs on its own shares was not run, so P − R mixes the
share change with 15 more epochs at lr 3e-4. P is the candidate either way.

**Then:**
- **If P succeeds:** I recommend carrying P forward to the CPU steps (share-model refit,
  product, benchmark, ladder study). The user decides.
- **If not:** report and stop.

## Deep-store run (user 2026-10-17: "launch it"; written before launch)

**S** = `7_calsim_ft45_deep` starts from dPL-CalSim's `best.pt` with the full retrain's recipe:
the target cleaning, the CDEC monthly companion, the dry-season term (λ 0.1132), flow-scaled
riparian ET, and the shares and loss references of `7_calsim_ds`. Four changes
(`deep/DESIGN.md`, `deep/review/REVIEW.md`; `launch_deep.py`, checked by `check_deep_cfg.py`):
- `--deep-store cascade`: a learned share of the supplemental drainage through a slow store of
  learned residence (2–10 years), outside the lower-zone deficit;
- `--deep-lr-mult 10`: the store head's learning rate at 10× (it starts at a share of 0.01);
- `--shape-min-days 180`: the variance term back at the masked outlets;
- the warm start: lr 3e-4, 1 warmup epoch, 45 epochs, patience 20.

A 1-epoch GPU smoke pair (12 entities, WY1987–90, graphs against eager) runs first. If the
epoch's loss differs by more than 1e-3 (relative), the launcher stops.

Read with `score_ft.py compare-full D 7_calsim_ft15_clean 7_calsim_ds 7_calsim_ft45_deep
compare_deep.txt DCRS` (columns D, C, R, S; guards on S − D).

**S succeeds if all hold:**
1. **The peaks are kept.**
   - Sierra-4 top-5 median within 3 points of D (at least 73.8).
   - CDEC daily paired median S − D at least −0.005.
   - SHA and ORO each S − D at least −0.02.
2. **The dry season is fixed.** The four Sacramento bars pass: elasticity ≤ 1.0, dry third within
   ±20 %, SHA and ORO within ±25 %.
3. **Nothing else is lost.**
   - USGS, UF and arcs S − D each at least −0.005.
   - lztwm S/D at most 1.4×.
   - +3 °C within 1 point (annual) and 2 points (Apr–Jul) of D.
   - Sacramento-4 in-sample monthly KGE at least 0.919 (D 0.939 − 0.02; the store's
     pre-test cost was −0.03, in alpha).

**The regime gate (at the epoch-15 and epoch-30 snapshots).** At ORO and FOL, the interflow
share of the top-5 peak days stays at least 20 %; the median `uzk` stays at least 0.6; the
Sierra-4 top-5 median stays at least 73.8. If the gate fails, I report it and give the user the
line to stop the run (Claude stops no process).

**Reported, not bars:**
- the San Joaquin Sep/Jun ratio (no run has met it);
- the store's share of the dry-third Jul–Sep flow at the Sacramento four;
- its end-of-September content in 1991 and 2014 against the wet years;
- the learned share and residence over the cells (`params_deep.csv`).

**No control arm.** The same run without the store was not trained, so S − D mixes the store
with 45 epochs of the full retrain's recipe. The nearest reference is the fine-tune's treatment
arm `7_calsim_ft15_ds` (15 epochs, no store, the default shape-min-days); its bars are in
`score/compare.txt`. C and R are the table's other columns.

**Then:**
- **If S succeeds:** I recommend carrying S forward to the CPU steps (share-model refit, which
  a run with the store needs; product, benchmark, ladder study). The user decides.
- **If not:** report and stop.

## Deep store from scratch (user 2026-10-18: "I want to do a training from scratch with this design"; written before launch)

**Where S stands.** S missed one item of its rule: ORO's dry third, at −27.5 % against ±25 %.
The user carries S forward anyway (2026-10-18: "I think S should be carried forward").

**X** = `7_calsim_ds_deep`. It is `7_calsim_ds`'s command from scratch, with S's three
deep-store options added and the flow-scaled riparian ET taken out (`launch_scratch.py`):
- **R's command:** 120 epochs, patience 20, lr 1e-3 with 4 warm-up epochs, seed 0.
- **The three options:**
  - `--deep-store cascade`;
  - `--deep-lr-mult 10`: the head's learning rate at 1e-2;
  - `--shape-min-days 180`.
- **No `--riparian flow`** (user 2026-10-18, on the recipe review for simplicity: "drop it, then
  launch the training"). The riparian ET is the reference form, as in dPL-CalSim, so the store is
  the run's only change to the model. Why:
  - the CPU pre-test (`deep/DESIGN.md`) passed the four Sacramento bars with the store on
    dPL-CalSim's field, under the reference form;
  - the peak diagnosis ranks the flow form as an enabler of R's no-interflow path (a
    hypothesis);
  - review 2 traced the extension arcs' summer excess to the flow form.
- **Checked:** `check_scratch_cfg.py` finds it differs from `7_calsim_ds` in exactly four
  fields: `shape_min_days`, `riparian` (flow → reference), `deep_store` and `deep_lr_mult`.
- **Control:** R is the nearest one, with the same seed and recipe, but no store,
  shape-min-days 300 and the flow form. X − R therefore mixes four changes. X is no longer S's
  design from scratch, so X − S mixes the riparian form with the start.
- **A smoke pair first**, as S's (1 epoch, 12 entities, WY1987–90, graphs against eager; the
  launcher stops if the epoch's loss differs by more than 1e-3 relative). The code changed since
  S's smoke (the external review's and review 2's fixes, `deep/worktree_before_fixes2.patch` is
  the tree before them), and the store has not run with the reference form on the GPU.
- **About 17 GPU-hours** (S ran at about 8.5 min an epoch).

Read with `score_ft.py compare-full D 7_calsim_ds 7_calsim_ft45_deep 7_calsim_ds_deep
compare_scratch.txt DRSX` (columns D, R, S, X; guards on X − D). The launcher runs it, then
`deep/store_report.py` on X's `best.pt`.

**X succeeds if all of S's items hold on X** (section "Deep-store run", items 1–3):
- the peaks kept;
- the four Sacramento bars;
- nothing else lost, including the Sacramento-4 monthly KGE at least 0.919.

**The regime gate, from scratch (epochs 40 and 60; `deep/watch_gate.py 7_calsim_ds_deep`).**
- **Epoch 40:** at ORO and at FOL, the interflow share of the top-5 peak days is at least
  15 %, and the median `uzk` at least 0.6.
- **Epoch 60:** the same, and the Sierra-4 top-5 median at least 70.
- **Why these values.** dPL-CalSim's own path from scratch cleared both:
  - epoch 40: 28 % and 20 %, `uzk` 0.83, Sierra-4 72.2;
  - epoch 60: 37 % and 31 %, `uzk` 0.91, Sierra-4 74.9.
  - R failed both: 0 %, `uzk` 0.29, Sierra-4 about 62.
- **If the gate fails:** I report it and give the user the line to stop the run. Claude stops
  no process.

**Reported, not bars:**
- the San Joaquin Sep/Jun ratio;
- the store report;
- X − S and X − R on every item;
- the tier-2 arcs on their own (`deep/arcs_score.py`, X added);
- the gate's values at epochs 10 and 20 (information only: dPL-CalSim's path or R's).

**Then:**
- **X succeeds:** I recommend X over S as the candidate carried forward. The user decides.
- **X fails an item S passes:** S stays the candidate. I report.
- **Otherwise:** I report both. The user decides.

**Outcome (2026-10-18).**
- **The gate failed at epoch 40:**
  - interflow 0 % at ORO and FOL;
  - `uzk` 0.361 / 0.370, flat since epoch 5.
- **The full bars at epoch 45** (`deep/xcheck/ASSESSMENT.md`, `score/compare_x_e045.txt`):
  - X fails 7 of S's 13 items, and is projected to fail 3-4 at epoch 120 (the peaks for certain);
  - it passes ORO's dry third, which S fails;
  - its extension arcs are better (Jul-Sep bias median +53 % against S's +235 %).
- **Stopped** on the user's go: "yes stop x and prepare the new fine tune".

## Deep store, warm start, reference riparian ET (user 2026-10-18: "yes stop x and prepare the new fine tune"; written before launch)

F = `7_calsim_ft45_deep_ref`. It is S's command less `--riparian flow` (`launch_ref.py`):
- dPL-CalSim's best.pt, 45 epochs at lr 3e-4 with a 1-epoch warm-up;
- the full retrain's recipe and cleaning;
- `--deep-store cascade`, `--deep-lr-mult 10`, `--shape-min-days 180`.

**Why** (`deep/xcheck/ASSESSMENT.md`):
- **S's dry-season pass rests partly on the flow form:**
  - at S's epoch 0 (D's net, flow form, store off), the dry third is already -7.5 and the wet third
    +29.4;
  - S then grew `riva` 7-8x (ORO 0.04 -> 0.30, FOL 0.06 -> 0.47) to take the wet summers back;
  - its extension arcs carry a +235 % median Jul-Sep bias.
- **From scratch this recipe misses the interflow regime** (R and X): the warm start is what keeps
  the peaks.
- **The CPU pre-test (`deep/DESIGN.md`)** put a uniform store on D's field under the reference
  form, untrained. Training has to lift Sac-4 mKGE by 0.01 and should trim the wet third:

  | Pre-test item | Value |
  |---|---|
  | peaks | 76.5 |
  | elasticity | 0.946 |
  | dry third | -16.2 |
  | SHA / ORO dry third | -20.8 / -21.1 |
  | wet third | +11.1 |
  | Sac-4 mKGE | 0.909 |

**Checked:**
- **`check_ref_cfg.py`** (`deep/check_ref_cfg.log`): the configuration differs from S's checkpoint
  in `riparian` only (flow -> reference), and the trainer's other arguments equal S's.
- **No smoke pair:**
  - X's pair (08:19, relative difference 0) ran this smoke command on this code;
  - the launcher refuses if a file under `sacsma/` changed since.
- **The difference F - S is the riparian form alone**, from the same start, seed and schedule.

**About 6.3 GPU-hours** (S's run).

**Read with:** `score_ft.py compare-full D 7_calsim_ft45_deep 7_calsim_ds_deep_e045
7_calsim_ft45_deep_ref compare_ref.txt DSXF`. Columns D, S, X at epoch 45, F; guards on F - D.
The launcher then runs:
- `deep/store_report.py` on F's best.pt;
- F's tier-2 pass with S's settings;
- `deep/arcs_ref.py`, which writes `deep/ref/arcs_score.txt`.

**F succeeds if all 13 of S's items hold on F** (section "Deep-store run", items 1-3):
- the peaks kept;
- the four Sacramento bars;
- nothing else lost, including the Sacramento-4 monthly KGE at least 0.919.

**The regime gate (epochs 15 and 30; `deep/watch_gate.py 7_calsim_ft45_deep_ref`):** S's:
- at ORO and at FOL, the interflow share of the top-5 peak days is at least 20 % and the median
  `uzk` at least 0.6;
- the Sierra-4 top-5 median is at least 73.8.
- **If the gate fails:** I report it and give the user the stop line.

**Reported, not bars:**
- `riva` at ORO, FOL and SHA (the gate's log and csv);
- the wet third;
- the San Joaquin Sep/Jun;
- the store report;
- F - S on every item;
- the tier-2 arcs, the extension arcs above all.

**Then:**
- **F succeeds:** I recommend F as the candidate carried forward. The user decides.
- **F fails no item that S passes:** I recommend F over S, for the same bars on the simpler
  physics. The user decides.
- **F fails an item that S passes:** I report both, with the riparian and arc evidence. The user
  decides.

**Outcome (2026-10-18 21:47): F SUCCEEDS, all 13 of S's items** (`score/compare_ref.txt`; best.pt =
the final net, selection 0.8427 against S's 0.8429). By BARS "Then", F is recommended as the
candidate carried forward; the user decides.

| Item | S | F | Bar |
|---|---|---|---|
| Sierra-4 | 76.8 | 77.6 | >= 73.8 |
| CDEC daily paired median vs D | +0.005 | +0.007 | >= -0.005 |
| SHA / ORO KGE vs D | -0.010 / -0.004 | -0.007 / -0.001 | >= -0.02 |
| elasticity | 0.777 | 0.802 | <= 1.0 |
| Sac-4 dry third | -19.9 | -18.1 | +-20 |
| SHA / ORO dry third | -21.0 / -27.5 (FAIL) | -20.0 / -22.9 | +-25 |
| usgs / uf / calsim vs D | +0.014 / -0.001 / +0.011 | +0.013 / -0.000 / +0.012 | >= -0.005 |
| lztwm / D | 1.26 | 0.89 | <= 1.4 |
| +3 C annual / Apr-Jul vs D | +0.16 / -0.48 | -0.31 / -0.61 | 1 / 2 points |
| Sac-4 monthly KGE | 0.956 | 0.954 | >= 0.919 |

Reported:
- San Joaquin Sep/Jun 0.072: it fails its 0.06 bar, as every run does.
- `riva` (ORO / FOL / SHA): 0.035 / 0.053 / 0.013 for F, against S's 0.30 / 0.47 / 0.16.
- Wet third -1.9 (S -4.6).
- Store: g ORO 0.21, FOL 0.16, SHA 0.43, tau about 8-9 years. The store gives 45 % of the Sac-4
  Jul-Sep flow in dry years (S 38.5 %).
- Extension arcs: Jul-Sep bias median -53 % (S +235 %, D -73 %) and monthly KGE 0.497 (S 0.510,
  D 0.543). All-arc KGE 0.691 (S 0.692).

How the run got there:
- **First launch (14:58):** it stopped early at epoch 23. The selection peaked at epoch 1 and
  patience was 20, so the scoring ran on epoch 1 (`score/compare_ref_best_e001.txt`).
- **Resume (18:43), on the user's "resume":**
  - `--patience 45`;
  - `--init-from` dropped, which the trainer requires on resume;
  - the empty `checkpoints/select/done` marker removed.
- **Provenance:** the checkpoints after the resume carry `init_from` empty. The warm start is
  dPL-CalSim's best.pt.
