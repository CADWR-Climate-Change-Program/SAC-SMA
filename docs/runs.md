# Runs

Which learned-parameter run is current, what each run scores, what was tried and not adopted,
and what is still open. The method is in [Learned parameters](learned_parameters.md) and
[CalSim3 rim inflows](calsim3_rim_inflows.md); where each file sits is in
[`artifacts/README.md`](../artifacts/README.md).

## Current model and product

The current model is dPL-CalSim (`multifamily/8_calsim`, the last rung of
[the ladder](#the-ladder)), and the current product is the CalSim3 rim-inflow product fitted on
it ([`artifacts/product/calsim3/`](../artifacts/product/calsim3/README.md)): monthly flow on the
196 rim arcs, TAF, WY1916–2018, one series per forcing (historical Livneh, WGEN Product A
scenarios 1 and 12).

Median monthly KGE against CalSim3 on the held-out water years 1976–85:

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.708 | 0.842 |
| 50 non-anchor arcs | 0.582 | 0.582 |
| 7 single-arc systems | 0.909 | 0.909 |
| All 196 | 0.685 | 0.809 |

On WY1922–1949, before any training year:

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.646 | 0.776 |
| 50 non-anchor arcs | 0.415 | 0.415 |
| 7 single-arc systems | 0.785 | 0.785 |
| All 196 | 0.633 | 0.762 |

Source: `artifacts/product/calsim3/historical_livneh_unsplit/product_metrics.csv`; the fit's own
scores, on its WY1950–2015 pass, are in `artifacts/product/calsim3/product_info.json`.
`sacsma verify product` checks that `sacsma dpl calsim product apply` reproduces every tracked
series.

dPL-CalSim trains one network on 159 entities: 69 USGS gauges, 17 CDEC records, the 9
unimpaired-flow subbasins (back to WY1950) and 64 CalSim3 arcs, on every year but WY1976–85,
which is held out of every family. Mean KGE by family; the training years are scored on the
cleaned targets, which keep 60 of the USGS gauges, 5 of the subbasins and 60 of the arcs, and
WY1976–85 on the records as they are:

| | usgs | cdec | uf | arcs | Tier 1 mean | Tier 2 median |
|---|---|---|---|---|---|---|
| Training years (`metrics.csv`) | 0.710 | 0.879 | 0.936 | 0.710 | – | – |
| WY1976–85 (`metrics_holdout.csv`, tier 1 and 2) | 0.639 | – | 0.878 | 0.705 | 0.904 | 0.685 |

Tier 1 is the mean over the 20 locations of `data/inputs/calsim3/tier1_sets.csv`, tier 2 the
median over the 196 arcs (`artifacts/results/dpl/multifamily/8_calsim/`). The CDEC daily
records are not scored on WY1976–85; [the ladder](#the-ladder) reads the outlets there on
monthly full natural flow.

## The ladder

The learned-parameter runs are one recipe, one change per rung, each trained from scratch but
the last, which fine-tunes a run trained from scratch. The rungs are named for their place on
the ladder, not for what they train on ([Conventions](conventions.md)).

Every rung uses the defaults of `sacsma dpl train`: the variance-normalized squared error with a
log-flow term and the record-form variance term, each chunk weighted by its days, dropout 0, a
gradient clip of 12, the state carried between water-year chunks with its relative saturation in
the gradient, the gradient through two water years, 120 epochs, and the `15cdec_grid` GA median as
the starting field. On top of them, every rung selects its network on the CPU every epoch
(`--cpu-select`, patience 20), masks the confirmed bad CDEC days
(`data/targets/cdec/fnf_daily_mask.csv`), maps `lzsk`, `lzpk`, `MFMAX` and `MFMIN` in log space
with `lzsk` down to 0.001, and adds a small penalty on the network's saturated outputs. The
commands are in [Reproduce](reproduce.md).

Rungs 1 to 7 train on the 15 CDEC watersheds, WY1989–2003. Rungs 2 to 7 run on the grid cells
and CalSim3 catchment outlines of the `multifamily` domain, restricted to the 15 CDEC records.

| Rung | Change from the rung above |
|---|---|
| GA parameters | the archived calibration, 7,891 HRUs (not a run) |
| `1_hru` | the network in place of the GA: SAC-SMA ET, Hamon PET, soil, vegetation and terrain inputs, the HRUs |
| `2_grid` | 1/16° grid cells and CalSim3 catchment outlines |
| `3_pt` | Priestley–Taylor PET |
| `4_noah` | soil-moisture-limited ET on observed vegetation, with the SAC-SMA exchange terms |
| `5_px` | a learned rain/snow threshold |
| `6_deep` | the deep store: a slow store behind the supplemental lower zone, with a learned share and a residence time of 2 to 10 years ([Learned parameters](learned_parameters.md#how-it-is-built)) |
| `7_aef` | AlphaEarth embeddings (`aef_u`) in place of every static input: soil, vegetation, terrain, elevation, position and flow length |
| `8_calsim` | dPL-CalSim: every family, every year but WY1976–85, family weights by footprint area with a fixed loss scale per family, and wider ranges for `MBASE`, `NMF`, `Velo` and `zperc`, trained from scratch without the deep store; then a 45-epoch fine-tune with the deep store, the cleaned targets, CDEC's monthly records from WY1950 and a dry-season term ([CalSim3 rim inflows](calsim3_rim_inflows.md#what-the-model-trains-on)) |
| `hybrid` | an LSTM on the forcing, `5_px`'s daily flow and 27 static attributes, 3 seeds |
| `hybrid_dt` | `hybrid` fine-tuned to keep `5_px`'s response to climate, 3 seeds |
| `lstm` | the LSTM without `5_px`'s flow (a control), 3 seeds |

Mean KGE over the 15 CDEC watersheds. WY1989–2003 and WY2004–18 are daily, from each run's
`metrics.csv` (the LSTM runs on the mean flow of their seeds). WY1976–85 is the held-out decade,
monthly, at the 13 outlets with a monthly record then, read once per run.

| Run | WY1989–2003 (training) | WY2004–18 | WY1976–85 |
|---|---|---|---|
| GA parameters | 0.808 | 0.768 | 0.769 |
| `1_hru` | 0.863 | 0.838 | 0.855 |
| `2_grid` | 0.854 | 0.826 | 0.836 |
| `3_pt` | 0.847 | 0.843 | 0.852 |
| `4_noah` | 0.858 | 0.844 | 0.850 |
| `5_px` | 0.868 | 0.841 | 0.859 |
| `6_deep` | 0.865 | 0.841 | 0.858 |
| `7_aef` | 0.859 | 0.839 | 0.853 |
| `8_calsim` | training | training | 0.913 |
| `hybrid` | 0.961 | 0.878 | 0.860 |
| `hybrid_dt` | 0.942 | 0.873 | 0.880 |
| `lstm` | 0.948 | 0.855 | 0.782 |

A mean can rest on one or two records. `steps.csv` of the ladder study pairs each rung with the
one before it record by record, with the number that gain and the median change. The GA to
`1_hru` on WY2004–18 gains 0.071 in the mean with 7 of 15 outlets up (SCC −0.18 to 0.62);
`7_aef` to `8_calsim` on the USGS gauges gains 0.124 in the mean with 30 of 49 up (median 0.03).

The same read on the regional records outside the 15 watersheds (mean KGE; USGS gauges daily,
unimpaired-flow subbasins and CalSim3 arcs monthly):

| Run | 49 USGS gauges | 9 UF subbasins | 64 CalSim3 arcs |
|---|---|---|---|
| `2_grid` | 0.489 | 0.815 | 0.634 |
| `3_pt` | 0.534 | 0.783 | 0.627 |
| `4_noah` | 0.535 | 0.772 | 0.629 |
| `5_px` | 0.522 | 0.803 | 0.630 |
| `6_deep` | 0.509 | 0.805 | 0.631 |
| `7_aef` | 0.515 | 0.844 | 0.659 |
| `8_calsim` | 0.639 | 0.878 | 0.705 |

Every rung is read on the grid statics in which every cell is a footprint mean ([the grid
README](../data/inputs/grid/README.md)). An earlier read had a single point sample at each cell
outside the 15-CDEC grid, with curvature, relief and slope on another scale: on it rungs 2 to 5
scored 0.35 to 0.39 on the USGS gauges and 0.66 to 0.73 on the subbasins. Inside the 15-CDEC
grid the means hold to 0.001 (`read_WY1976-85_statics.csv` in `8_calsim`'s provenance).

The response to a changed climate is the change in runoff against the run's own present climate,
in percent, averaged over the 15 CDEC watersheds on WY1951–75, WY1986–88 and WY2004–18; the
precipitation columns are annual runoff (`sacsma dpl study ladder`,
`artifacts/results/dpl/studies/ladder/ladder.csv`).

| Run | +3 °C, annual | +3 °C, April–July | Precipitation −10 % | Precipitation +10 % |
|---|---|---|---|---|
| GA parameters | −10.6 | −33.3 | −17.3 | +18.0 |
| `1_hru` | −12.0 | −33.7 | −18.3 | +19.1 |
| `2_grid` | −13.3 | −34.0 | −18.7 | +19.4 |
| `3_pt` | −3.4 | −29.8 | −17.8 | +18.7 |
| `4_noah` | −2.9 | −29.3 | −17.9 | +18.8 |
| `5_px` | −2.4 | −32.7 | −17.7 | +18.5 |
| `6_deep` | −2.4 | −33.1 | −17.9 | +18.7 |
| `7_aef` | −2.0 | −32.9 | −17.8 | +18.6 |
| `8_calsim` | −1.2 | −30.4 | −17.4 | +18.1 |
| `hybrid` | 0.0 | −22.8 | −14.4 | +14.2 |
| `hybrid_dt` | −1.5 | −29.4 | −16.2 | +16.8 |
| `lstm` | +10.5 | −5.1 | −6.6 | +6.3 |

The annual response at +3 °C is −11 to −13 % with Hamon PET (GA, `1_hru`, `2_grid`) and
−1 to −3 % with Priestley–Taylor PET (`3_pt` onward), while April–July falls 29 to 34 % in every
rung.

What the ladder shows (a step counts when the mean moves by 0.01 or more):

- The network generalizes better than the GA: +0.07 on WY2004–18 and +0.09 on WY1976–85.
- The grid costs about 0.01 to 0.02 at the outlets in both test windows. It is the form that
  covers the CalSim3 catchments.
- Priestley–Taylor PET gains at the outlets (+0.02 in both windows) and on the USGS gauges
  (+0.04), and loses on the unimpaired-flow subbasins (−0.03).
- The Noah-type ET and the learned rain/snow threshold leave the outlets unchanged in both
  windows. Noah-type ET loses on the subbasins (−0.01); the threshold gains on the subbasins
  (+0.03) and loses on the USGS gauges (−0.01).
- The deep store moves only the USGS gauges (−0.013; the median −0.002, 24 of 49 up). The
  outlets, subbasins and arcs stay within 0.005 in both test windows.
- AlphaEarth inputs gain on the regional records, most on the subbasins (+0.04, 6 of 9 up) and
  the arcs (+0.03, 40 of 64); USGS +0.006. The 15 outlets do not change in the mean (WY2004–18
  −0.002, WY1976–85 −0.005).
- dPL-CalSim's training (`8_calsim`: every family, the cleaned targets, the dry-season term)
  gains at the outlets (+0.06, 12 of 13 up), on the USGS gauges (+0.12, 30 of 49; median
  +0.03), on the subbasins (+0.03, 4 of 9; median −0.004) and on the arcs (+0.05, 42 of 64).
- The hybrid adds skill over its physics on WY2004–18 (+0.04) but not on WY1976–85, and
  loses its response to warming (0.0 % annual runoff at +3 °C against −2.4 % for `5_px`).
  `hybrid_dt` keeps most of the skill (+0.03 and +0.02) and about 60 % of the response
  ([Learned parameters](learned_parameters.md#the-lstm-hybrids-and-their-response-to-warming)).
  The LSTM without physics is at the rungs' level on WY2004–18 and far below them on WY1976–85.

## What is tracked and what is local

Each tracked run has a folder under `artifacts/models/` (checkpoints, training log, parameter
tables) and one under `artifacts/results/` (scores, figures; for dPL-CalSim also the simulated
series, the CalSim3 checks and the atlas). Everything else a command writes for the run goes to
the run's folder under `artifacts/_local/runs/`, which git ignores, as it ignores the caches
(`_local/cache/`), the scratch runs (`_local/testing/`) and the runs that were not adopted.

## Regenerating a run's outputs

```bash
sacsma dpl evaluate <run>/checkpoints/best.pt --score-holdout   # parameter tables, metrics (and the held-out years of a run with them)
sacsma dpl calsim tier1 <run> --score-holdout
sacsma dpl calsim tier2 <run> --trace-python <python of the sacsma-gis environment> --score-holdout
sacsma dpl calsim atlas <run> --score-holdout
```

`<run>` is any of the run's folders. The complete sequence, the product included, is in
[Reproduce](reproduce.md).

## Known costs

- Unimpaired-flow subbasins in WY1976–85: dPL-CalSim's mean KGE is 0.878 against 0.894 for the
  multi-family run that preceded the ladder. It is lower at 7 of the 9 subbasins, most at
  Chowchilla (0.76 against 0.83).
- The benchmark: on WGEN Product A scenario 1 against CDEC's monthly full natural flow,
  WY1991–2018, dPL-CalSim's median KGE over the 12 sites is 0.940, against 0.949 for the
  multi-family run that preceded the ladder. The loss is mostly at NML (0.87 against 0.96),
  TLG and CLE, with more volume in the wrong month (0.064 against 0.045). These are training
  years: the check is the move to the weather-generator forcing. See [Benchmark](benchmark.md).
- Snow-17 parameters fall outside the published ranges. With `MBASE` free down to −2 °C and
  `NMF` up to 0.5, dPL-CalSim puts `MBASE` below 0 °C in every cell (median −1.79; 17 % at −2)
  and `NMF` above 0.3, the largest Anderson gives, in every cell (median 0.47; 12 % at 0.5).
  Daily ground melt `DAYGM` is 0.018 mm/day at the median, against 0.24 with the range pinned,
  and below 0.01 in 32 % of the cells (inside its 0 to 0.3 range). The snowpack matches that of
  the same recipe with these ranges pinned (regional mean peak 191 against 192 mm), as do the
  WY1976–85 scores. The values fit this structure to flow; they are not measured melt properties.
- Flood peaks: the two-year gradient flattens the five largest annual peaks by about 4 points
  of their ratio to observed on WY1989–2003 (`2_grid` against a one-year control). Both
  under-predict them.
- Single seed: no rung has a second seed.
- `hybrid` on WY1976–85: level with `5_px` on the mean (0.860 against 0.859) and below it
  at 7 of the 13 outlets, most at MRC (0.784 against 0.937). On WY2004–18 `5_px` is ahead of
  it at SHA, BND and MRC.

## Tried and not adopted

One line per approach, so that it is not tried again without a reason. Numbers are as measured at
the time.

**Parameters on the 15 CDEC watersheds**

- Continuous soil, vegetation and terrain inputs beat one-hot classes; wider ranges, adaptive
  per-watershed loss weights, a spatial smoother and per-process network heads were a wash.
- Day-of-year parameters trained on flow alone: seasonal recession rates hurt, and flow does not
  identify a seasonal `Kpet`.
- A full Noah canopy ET with seven learned parameters: 0.760 in validation against 0.834 for Hamon
  on the same grid.
- ET and SWE products in the loss: a pull on ET level degraded flow, because the products
  disagree on level; a pull on seasonal shape helped Priestley–Taylor PET a little and did not
  transfer to the Noah-type ET, whose summer ET is limited by water.
- Seasonal melt parameters on the Noah-type ET: better in the southern Sierra, worse at NHG and in
  north-state volume, and not scorable through the reference model.
- A shortened spinup for trained parameters: trained fields hold more than ten years of state
  (0.655 from a 1978 start against 0.759 from the full record), so scoring starts cold at the
  beginning of the record.
- Parameters that respond to climate (four climate indices among the network's inputs): +0.009
  validation KGE over the same physics with fixed parameters; the ladder keeps inputs fixed in time.

**Multi-family training**

- The Noah-type ET without the SAC-SMA exchange terms: the trained fields used the riparian term as
  a summer ET sink and ran early at all 20 tier-1 locations.
- One-year gradients: the gradient on the slow lower-zone drainage and on `Kpet` at Shasta has the
  wrong sign. Against two years, on `2_grid`, Shasta's August–November flow is 0.57 of observed
  against 0.78.
- Gradients through three water years: they fixed Shasta's July to November volume and lost the
  Sierra flood peaks; timing and peak terms in the loss did not recover them.
- The state carry that also holds the drainage flux fixed in the gradient: it damped the Sierra
  flood peaks.
- All 64 AlphaEarth coordinates instead of 16 components: higher in training, lower at ORO and on
  arcs outside the trained cells.
- A cap of 1.5 °C on the learned rain/snow threshold: Trinity loses most of its gain.
- Family shares divided by the entities present in a chunk: the USGS family carried about 0.57 of
  the loss against a share of 0.22, because its records start earlier.
- Training the USGS gauges on every year: no clean held-out reading of the CalSim3 checks.
- Family shares by observed volume against by footprint area: not resolved on one seed.
- Wider ranges also for `PLWHC` (to 0), `UADJ` (to 0) and `MFMIN` (to 0.01), and `NMF` to 0.6
  (`7_calsim_fpb`): `PLWHC` and `UADJ` went below their published floors in every cell (medians
  0.004 and 0.005) and `MFMIN` in 55 %, `NMF` above 0.5 in 82 %, and the Noah-lite ET's
  slope at the wilting point drove one epoch's gradient to 5e8 (since capped,
  [Model equations](equations.md#6-parameter-table)). Selection score 0.829 against dPL-CalSim's
  0.831.
- Building the model by fine-tuning earlier runs (a one-year base, then two-year fine-tunes): on
  WY1976–85 the run trained from scratch on one recipe is level on the arcs and outlets, better
  on the USGS gauges (+0.011) and the product (+0.012), worse on the subbasins (−0.020).

**The dry season** (October 2026; `8_calsim`'s fine-tune came out of these)

- The cleaned targets, CDEC's monthly records, the dry-season term and a riparian ET
  proportional to the channel flow, trained from scratch (`7_calsim_ds`): the late summer came
  up (Sacramento four, driest third of the years, August–September −19 % against −31 %), but
  the lower zone grew, the percolation deficit stayed high and the interflow went to zero. The
  five largest annual peaks of four Sierra watersheds fell to 66.6 % of observed against 76.8 %,
  and the CDEC daily KGE at ORO by 0.05.
- The same recipe as a 15-epoch fine-tune of dPL-CalSim (`7_calsim_ft15_ds`): the wet years
  moved (wettest third −1.5 %, against +9.3 % for a control fine-tune on the cleaned targets
  alone), the dry ones did not (−30 % against −31 %).
- The from-scratch run fine-tuned 15 epochs with twice the daily CDEC weight
  (`7_calsim_ds_ft15_pk`): the peaks back to 69.0 %, the CDEC daily KGE not.
- The deep store from scratch on the recipe, without the flow-proportional riparian ET
  (`7_calsim_ds_deep`): no interflow from epoch 5 (Sierra peaks 63.6 % at epoch 40); stopped at
  epoch 49.
- The deep store fine-tune with the flow-proportional riparian ET (`7_calsim_ft45_deep`): every
  bar but ORO's dry third (−27.5 % against ±25 %), but the network grew `riva` from about 0.04 to
  0.30 at ORO and 0.47 at FOL, and the July–September flow of the extrapolated arcs to +235 % of
  CalSim3's (−53 % without it).
- The deep store fed from the primary drainage instead of the supplemental (a test on
  dPL-CalSim's field): Sierra peaks 72–75 % against 76.5 %.

**LSTM hybrids on the 15 CDEC watersheds**

- The LSTM predicting a correction to the simulated flow: the same skill, and the validation
  volume bias at NML, MRC and ORO stayed.
- Better physics under the LSTM: the hybrids score about the same on every physics run; the LSTM
  erases the difference.
- PET, mean temperature or the day of year as LSTM inputs: PET adds a little skill and almost no
  response to warming, mean temperature repeats Tmin and Tmax, and only the response term in the
  loss moves the response.
- A two-layer static encoder and a two-layer head: 15 distinct static vectors cannot identify them.
- Fewer response points (1, 5, 8 before the 14 of `hybrid_dt`): the error at ±20 % precipitation
  is larger.
- Dropout, input noise, weight decay and a log-flow term in the loss: they held the training KGE
  near 0.89 with no gain on WY2004–18.
- A wider network or longer training (hidden 128 or 256, 150 epochs): training KGE up to 0.98,
  WY2004–18 level or lower. A loss term on the variance ratio, a loss on every day of the
  window, or the statics scaling the last hidden state: no gain.
- AlphaEarth embeddings in the LSTM (`6_aef`'s flow, embedding statics, the embedding setting
  the initial state or scaling the inputs, its spread within a watershed, semi-distributed units
  by elevation band or embedding cluster): no gain on WY2004–18. A one-hot label scores about
  as well as any statics, so 15 watersheds cannot show what the embeddings add.

**The product** (median monthly KGE on the held-out WY1976–85)

- Constant factors per arc on the runoff components: 0.77 to 0.78 over all arcs.
- A daily LSTM on the model states: failed its out-of-fold check (unimpaired-flow subbasins 0.771
  against the model's 0.826).
- Corrections of the system flows (a constant volume factor, gradient-boosted trees, a daily
  LSTM): all lose to the model's own flow on held-out years and win on training years. The ratio
  of CalSim3 to the model drifts by decade.
- A daily LSTM on the non-anchor arcs: 0.719 against the model's 0.604; the model's flow was kept
  by choice, after the held-out numbers were read.
- An LSTM that gives the flow outright, not a correction: behind the correction form.

## Open

- The share model's response is fitted on uniform climate changes. Under WGEN weather it is
  checked on scenario 12 only.
- No warmed VIC run exists for a like-for-like comparison of the response.
- A rain/snow partition that does not switch whole storm days (the candidate fix for ORO) has not
  been built.
- The Stanislaus' late summer is low: dPL-CalSim's August–November flow at NML is 0.66 of
  observed over its training years (SHA 0.92, ORO 0.97).
