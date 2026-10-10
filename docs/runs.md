# Runs

Which learned-parameter run is current, what each run scores, what was tried and not adopted,
and what is still open. The method is in [Learned parameters](learned_parameters.md) and
[CalSim3 rim inflows](calsim3_rim_inflows.md); where each file sits is in
[`artifacts/README.md`](../artifacts/README.md).

## Current model and product

**dPL-CalSim** (`multifamily/7_calsim`, the last rung of [the ladder](#the-ladder)) and the
**CalSim3 rim-inflow product** fitted on it
([`artifacts/product/calsim3/`](../artifacts/product/calsim3/README.md)): monthly flow on the 196
rim arcs, TAF, WY1916–2018, one series per forcing (historical Livneh, WGEN Product A scenarios 1
and 12).

Median monthly KGE against CalSim3 on the held-out water years 1976–85:

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.712 | 0.847 |
| 50 non-anchor arcs | 0.562 | 0.562 |
| 7 single-arc systems | 0.896 | 0.896 |
| All 196 | 0.688 | 0.818 |

On WY1922–1949, before any training year:

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.656 | 0.778 |
| 50 non-anchor arcs | 0.433 | 0.433 |
| 7 single-arc systems | 0.809 | 0.809 |
| All 196 | 0.647 | 0.747 |

Source: `artifacts/product/calsim3/historical_livneh_unsplit/product_metrics.csv`; the fit's own scores,
on its WY1950–2015 pass, are in `artifacts/product/calsim3/product_info.json`. `sacsma verify product`
checks that `sacsma dpl calsim product apply` reproduces every tracked series.

dPL-CalSim trains one network on 159 entities: 69 USGS gauges, 17 CDEC records, the 9
unimpaired-flow subbasins (back to WY1950) and 64 CalSim3 arcs, on every year but WY1976–85,
which is held out of every family. Mean KGE by family:

| | usgs | cdec | uf | arcs | Tier 1 mean | Tier 2 median |
|---|---|---|---|---|---|---|
| Training years (`metrics.csv`) | 0.660 | 0.881 | 0.923 | 0.704 | – | – |
| WY1976–85 (`metrics_holdout.csv`, tier 1 and 2) | 0.638 | – | 0.874 | 0.706 | 0.897 | 0.688 |

Tier 1 is the mean over the 20 locations of `data/inputs/calsim3/tier1_sets.csv`, tier 2 the
median over the 196 arcs (`artifacts/results/dpl/multifamily/7_calsim/`). The CDEC daily records are
not scored on WY1976–85; [the ladder](#the-ladder) reads the outlets there on monthly full natural flow.

## The ladder

The learned-parameter runs are one recipe trained from scratch, one change per rung. The rungs
are named for their place on the ladder, not for what they train on
([Conventions](conventions.md)).

Every rung uses the defaults of `sacsma dpl train`: the variance-normalized squared error with a
log-flow term and the record-form variance term, each chunk weighted by its days, dropout 0, a
gradient clip of 12, the state carried between water-year chunks with its relative saturation in
the gradient, the gradient through two water years, 120 epochs, and the `15cdec_grid` GA median as
the starting field. On top of them, every rung selects its network on the CPU every epoch
(`--cpu-select`, patience 20), masks the confirmed bad CDEC days
(`data/targets/cdec/fnf_daily_mask.csv`), maps `lzsk`, `lzpk`, `MFMAX` and `MFMIN` in log space
with `lzsk` down to 0.001, and adds a small penalty on the network's saturated outputs. The
commands are in [Reproduce](reproduce.md).

Rungs 1 to 6 train on the 15 CDEC watersheds, WY1989–2003. Rungs 2 to 6 run on the grid cells and
CalSim3 catchment outlines of the `multifamily` domain, restricted to the 15 CDEC records.

| Rung | Change from the rung above |
|---|---|
| GA parameters | the archived calibration, 7,891 HRUs (not a run) |
| `1_hru` | the network in place of the GA: SAC-SMA ET, Hamon PET, soil, vegetation and terrain inputs, the HRUs |
| `2_grid` | 1/16° grid cells and CalSim3 catchment outlines |
| `3_pt` | Priestley–Taylor PET |
| `4_noah` | soil-moisture-limited ET on observed vegetation, with the SAC-SMA exchange terms |
| `5_px` | a learned rain/snow threshold |
| `6_aef` | AlphaEarth embeddings (`aef_u`) in place of soil, vegetation and terrain |
| `7_calsim` | dPL-CalSim: every family, every year but WY1976–85, family weights by footprint area with a fixed loss scale per family |
| `hybrid` | an LSTM on the forcing, `5_px`'s daily flow and 27 static attributes, 3 seeds |
| `hybrid_dt` | `hybrid` fine-tuned to keep `5_px`'s response to climate, 3 seeds |
| `lstm` | the LSTM without `5_px`'s flow (a control), 3 seeds |

Mean KGE over the 15 CDEC watersheds. WY1989–2003 and WY2004–18 are daily, from each run's
`metrics.csv` (the LSTM runs on the mean flow of their seeds). WY1976–85 is the one read of the
held-out decade, monthly, at the 13 outlets with a monthly record then.

| Run | WY1989–2003 (training) | WY2004–18 | WY1976–85 |
|---|---|---|---|
| GA parameters | 0.808 | 0.768 | 0.769 |
| `1_hru` | 0.863 | 0.838 | 0.855 |
| `2_grid` | 0.855 | 0.826 | 0.836 |
| `3_pt` | 0.847 | 0.843 | 0.852 |
| `4_noah` | 0.858 | 0.844 | 0.850 |
| `5_px` | 0.868 | 0.841 | 0.859 |
| `6_aef` | 0.863 | 0.837 | 0.866 |
| `7_calsim` | training | training | 0.899 |
| `hybrid` | 0.888 | 0.871 | 0.876 |
| `hybrid_dt` | 0.852 | 0.847 | 0.862 |
| `lstm` | 0.851 | 0.838 | 0.774 |

The same read on the regional records outside the 15 watersheds (mean KGE; USGS gauges daily,
unimpaired-flow subbasins and CalSim3 arcs monthly):

| Run | 49 USGS gauges | 9 UF subbasins | 64 CalSim3 arcs |
|---|---|---|---|
| `2_grid` | 0.370 | 0.734 | 0.604 |
| `3_pt` | 0.393 | 0.704 | 0.582 |
| `4_noah` | 0.345 | 0.661 | 0.580 |
| `5_px` | 0.352 | 0.710 | 0.591 |
| `6_aef` | 0.532 | 0.864 | 0.658 |
| `7_calsim` | 0.638 | 0.874 | 0.706 |

What the ladder shows (a step counts when the mean moves by 0.01 or more):

- **The network generalizes better than the GA.** +0.07 on WY2004–18 and +0.09 on WY1976–85.
- **The grid costs about 0.01 to 0.02** at the outlets in both test windows. It is the form that
  covers the CalSim3 catchments.
- **Priestley–Taylor PET gains at the outlets** (+0.02 in both windows) and on the USGS gauges,
  and loses on the unimpaired-flow subbasins and the arcs.
- **The Noah-type ET and the learned rain/snow threshold** leave the outlets unchanged in both
  windows. Noah-type ET loses on the USGS gauges and subbasins; the threshold gains on the
  subbasins (+0.05).
- **AlphaEarth inputs transfer.** No change at the 15 outlets, and better on every regional
  family (USGS +0.18, subbasins +0.15, arcs +0.07). On WY2004–18 they gain at MKM, FOL, PNF and
  ORO and lose at NHG, TRM and NML.
- **Training on every family** (`7_calsim`) gains at the outlets (+0.03) and in every family.
- **The hybrid adds skill** over its physics (+0.03 on WY2004–18, +0.02 on WY1976–85) and loses
  its response to warming (−0.3 % annual runoff at +3 °C against −2.4 % for `5_px`);
  `hybrid_dt` keeps most of the response and gives most of the skill back
  ([Learned parameters](learned_parameters.md#the-lstm-hybrids-and-their-response-to-warming)).
  The LSTM without physics is at the rungs' level on WY2004–18 and far below them on WY1976–85.

## What is tracked and what is local

Each tracked run has a folder under `artifacts/models/` (checkpoints, training log, parameter
tables) and one under `artifacts/results/` (scores, figures; for dPL-CalSim also the simulated
series, the CalSim3 checks and the atlas). Everything else a command writes for the run goes to the
run's folder under `artifacts/_local/runs/`, which git ignores, as it ignores the caches
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

- **Unimpaired-flow subbasins in WY1976–85.** dPL-CalSim scores 0.874, below the run it was
  adopted over (0.894), at each of the 9 subbasins.
- **The benchmark.** On WGEN Product A scenario 1 against CDEC's monthly full natural flow,
  WY1991–2018, dPL-CalSim's median KGE over the 12 sites is 0.925, against 0.949 for the run it
  was adopted over. The loss is mostly in the Sierra (TLG, NML, CLE, YRS), with more volume in the
  wrong month (0.063 against 0.045). These are training years: the check is the move to the
  weather-generator forcing. See [Benchmark](benchmark.md).
- **Flood peaks.** The two-year gradient flattens the five largest annual peaks by about 4 points
  of their ratio to observed on WY1989–2003 (`2_grid` against a one-year control). Both
  under-predict them.
- **Single seed.** No rung has a second seed.
- **`hybrid` on WY1976–85.** It is below `5_px` at SHA, ORO, YRS, FOL, TLG, MRC and MIL
  (−0.015), most at MRC (0.801 against 0.937). On WY2004–18
  `5_px` is ahead of it at BND, MRC, NHG, PNF and SHA.

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
- Building the model by fine-tuning earlier runs (a one-year base, then two-year fine-tunes): on
  WY1976–85 the run trained from scratch on one recipe is level on the arcs and outlets, better
  on the USGS gauges (+0.011) and the product (+0.012), worse on the subbasins (−0.020).

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
- The held-out decade WY1976–85 has been read six times.
- A rain/snow partition that does not switch whole storm days (the candidate fix for ORO) has not
  been built.
- **The Stanislaus' late summer is low**: dPL-CalSim's August–November flow at NML is 0.65 of
  observed over its training years (SHA 0.92, ORO 0.97).
- **`relief_m`** in the grid statics (`data/inputs/grid/soilveg_continuous.csv`) does not mean the
  same thing on every cell. Against the 3DEP elevation spread of the whole 1/16° cell (1,357 cells
  checked), it matches on the cells from the 15-CDEC grid and is 3.6 to 4.2 times smaller on those
  from the other sources, which `sample_gis.py` takes over a 25-pixel window.
