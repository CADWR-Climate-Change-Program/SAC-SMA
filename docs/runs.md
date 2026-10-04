# Runs

Which learned-parameter run is current, what each run scores, what was tried and not adopted,
and what is still open. The method is in [Learned parameters](learned_parameters.md) and
[CalSim3 rim inflows](calsim3_rim_inflows.md); where each file sits is in
[`artifacts/README.md`](../artifacts/README.md).

## Current model and product

**dPL-CalSim** (`noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef`),
adopted 2026-09-30, and the **CalSim3 rim-inflow product**
([`artifacts/product/calsim3/`](../artifacts/product/calsim3/README.md)), adopted 2026-10-01: monthly flow on
the 196 rim arcs, TAF, WY1916–2018, one series per forcing (historical Livneh, WGEN Product A
scenarios 1 and 12).

Median monthly KGE against CalSim3 on the held-out water years 1976–85:

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.699 | 0.834 |
| 50 non-anchor arcs | 0.604 | 0.604 |
| 7 single-arc systems | 0.870 | 0.870 |
| All 196 | 0.690 | 0.812 |

On WY1922–1949, before any training year:

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.666 | 0.783 |
| 50 non-anchor arcs | 0.417 | 0.417 |
| 7 single-arc systems | 0.778 | 0.778 |
| All 196 | 0.637 | 0.760 |

Source: `artifacts/product/calsim3/historical_livneh_unsplit/product_metrics.csv`; the fit's own scores,
on its WY1950–2015 pass, are in `artifacts/product/calsim3/product_info.json`. `sacsma verify product`
checks that `sacsma dpl calsim product apply` reproduces every tracked series.

## Multi-family runs

One parameter network trained on several families of flow record. Mean KGE of each family's
entities over their training years (the run's `metrics.csv`), and the two checks against
CalSim3: the mean over the 20 tier-1 locations and the median over the 196 arcs of tier 2. The
first five runs are checked over WY1950–84, dPL-CalSim over its held-out WY1976–85.

| Name | Folder | Entities | usgs | cdec | uf | arcs | Tier 1 mean | Tier 2 median |
|---|---|---|---|---|---|---|---|---|
| – | `noah_cdec_uf_usgs` (local) | 95 | 0.587 | 0.786 | 0.859 | – | 0.812 | 0.598 |
| – | `noah_cdec_uf_usgs_areaw` (local) | 95 | 0.592 | 0.791 | 0.879 | – | 0.825 | 0.599 |
| – | `noah_cdec_uf` (local) | 26 | – | 0.790 | 0.893 | – | 0.794 | 0.576 |
| dPL-26 | `noah_cdec_uf_sacx_carry_px_aef` | 26 | – | 0.874 | 0.934 | – | 0.859 | 0.628 |
| dPL-95 | `noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef` | 95 | 0.687 | 0.880 | 0.935 | – | 0.857 | 0.657 |
| **dPL-CalSim** | `noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef` | 159 | 0.691 | 0.872 | 0.929 | 0.715 | 0.908 | 0.690 |

The first three are the first recipe; they are kept locally, not tracked. dPL-26 was the base
from 2026-09-28, dPL-95 from 2026-09-29; dPL-CalSim starts from dPL-95's recipe. The parts of
a folder name are explained in the [Glossary](glossary.md#run-folder-names).

## Runs on the 15 CDEC watersheds

Mean daily KGE over the 15 watersheds, calibration WY1989–2003 and validation WY2004–2018, from
each run's `metrics.csv`. The parameter runs are scored as trained (their numerics and basin
weights); the LSTM ensembles on the mean flow of their seeds.

| Run | What changed | Calibration | Validation |
|---|---|---|---|
| GA parameters | the archived calibration, 7,891 HRUs (not a run) | 0.805 | 0.768 |
| `hamon_dense` | the network in place of the GA, nothing else | 0.802 | 0.838 |
| `hamon` | 1/16° grid cells in place of HRUs, CalSim3 catchment outlines | 0.807 | 0.829 |
| `pt` | Priestley–Taylor PET | 0.796 | 0.823 |
| `noah_noca` | soil-moisture-limited ET; `physical` inputs, so parameters fixed in time | 0.759 | 0.792 |
| `noah` | the same with `physical_climate` inputs: parameters that respond to climate | 0.771 | 0.801 |
| `hybrid` | an LSTM on top of `noah`'s simulated flow, 3 seeds | 0.926 | 0.875 |
| `hybrid_dt` | `hybrid` trained to keep `noah`'s response to climate, 3 seeds | 0.870 | 0.849 |
| `lstm` | the LSTM without the simulated flow (a control), 3 seeds | 0.909 | 0.835 |
| `hybrid_noca` (local) | an LSTM on `noah_noca`, 8 seeds | 0.917 | 0.869 |
| `hybrid_dt_noca` (local) | the same with a single +2 °C response term, 8 seeds | 0.916 | 0.864 |

`noah_noca` is the climate-frozen baseline of the `adaptive` study. The hybrids read `noah`'s daily
flow as trained, which the engine makes from its checkpoint (`sacsma dpl hybrid --physics noah`).

## What is tracked and what is local

Each tracked run has a folder under `artifacts/models/` (checkpoints, training log, parameter
tables, `provenance/`) and one under `artifacts/results/` (scores, simulated series, the
atlas). Everything else a command writes for the run goes to the run's folder under
`artifacts/_local/runs/`, which git ignores, as it ignores the caches (`_local/cache/`), the
scratch runs (`_local/testing/`) and the runs that were not adopted (`_local/runs/`).

## Regenerating a run's outputs

```bash
sacsma dpl evaluate <run>/checkpoints/best.pt    # parameter tables, metrics
sacsma dpl calsim tier1 <run>
sacsma dpl calsim tier2 <run> --trace-python <python of the sacsma-gis environment>
sacsma dpl calsim atlas <run>
```

`<run>` is any of the run's folders. The complete sequence, the product included, is in
[Reproduce](reproduce.md).

## Known costs

- **NHG.** Daily KGE 0.827 in dPL-CalSim (dPL-95 0.873, dPL-26 0.924).
- **Unimpaired-flow subbasins, WY1986–2014.** 0.896 against a bar of 0.926, after their targets
  were extended back to WY1950.
- **ORO in dPL-26.** 0.838 against 0.896 before the learned rain/snow threshold (a run that is
  kept locally, not tracked); dPL-95 is at 0.888.
- **Single seed.** No run has a second seed. dPL-95 was adopted without its paired run without
  the family loss scale, dPL-CalSim without its control without the arcs
  (`provenance/DEVIATIONS.txt` of its model folder).
- **`hybrid`.** Its volume bias moves between calibration and validation at MRC (+1.7 % to
  −8.5 %) and NML (−2.2 % to −7.3 %). For the earlier generation (`hybrid_noca`) the physics
  run was the better out-of-sample answer on monthly flow against CalSim3 at NML, MRC and ORO.
  That has not been re-read for this one, whose daily validation KGE is above `noah`'s at NML
  and ORO.

## Tried and not adopted

One line per result, so that it is not tried again without a reason. The runs behind these
lines are not tracked (those that still exist are under `artifacts/_local/`), and the numbers
are as recorded when the runs were made. The dated record they come from is the run record as
of commit `f4140c0` (`git show f4140c0:artifacts/dpl/RUNS.md`).

**Parameters on the 15 CDEC watersheds** (validation KGE unless stated)

- **Inputs and regularization.** Continuous soil, vegetation and terrain inputs beat one-hot
  classes. Wider parameter ranges, adaptive per-watershed loss weights, a spatial smoother and
  per-process network heads were a wash (0.836 to 0.838 against 0.840).
- **Day-of-year parameters trained on flow alone.** Seasonal recession rates hurt. A seasonal
  `Kpet` is a wash: flow does not identify it.
- **Full Noah canopy ET, seven learned parameters.** 0.760 at best against 0.834 for Hamon on
  the same grid, and no better than the one-parameter form `noah` uses.
- **ET and SWE products in the loss** (code removed). A monthly pull on ET level degraded
  flow, because the products disagree on level. A pull on seasonal shape with a water-balance
  level constraint and a seasonal `Kpet`, as a fine-tune of `pt`, reached 0.810 / 0.837
  (`pt` 0.799 / 0.826). The recipe did not transfer to the Noah-type ET, whose summer ET is
  limited by water, not by `Kpet`.
- **Seasonal melt parameters on the Noah-type ET.** 0.799 against 0.792 scored the same way:
  better at NML, MRC and in the southern Sierra, worse at NHG (0.511 against 0.560) and in
  north-state volume. Dropped, also because it could not be scored through the reference
  model.
- **A shortened spinup for trained parameters.** Trained fields hold more than ten years of
  state: one checkpoint scored 0.655 from a 1978 start and 0.759 from the full record. Scoring
  uses the cycle spinup.

**LSTM hybrids on the 15 CDEC watersheds**

- **The LSTM predicting a correction to the simulated flow** (instead of the flow). Same skill
  (0.873). Its validation volume bias at NML, MRC and ORO was not fixed by removing the
  day-of-year inputs, by a penalty on the mean correction, or by better physics underneath.
- **Better physics under the LSTM.** The hybrids score about the same on every physics run
  tried. The LSTM erases the difference.
- **PET as an LSTM input.** It adds a little skill (0.872 against 0.869) and almost no
  response to warming (0.24 of the physics response). Only the response term in the loss moves
  the response. Without it the response is a draw: 0.15 and −0.57 of the physics response in
  two trainings of one recipe.
- **Fewer response points.** One point at +2 °C (`hybrid_dt_noca`), then 5 and 8 points, were
  tried before the 14 of `hybrid_dt`. With precipitation changes of ±20 % among the points,
  the error at those edges is 3.7 points of annual change, against 10.1 for `hybrid`. The
  surfaces of the 5-point recipe (the retired `sacsma dpl study response`) are not kept.

**Multi-family training** (on the 26 entities of dPL-26 unless stated; one seed each)

- **Without the SAC-SMA exchange terms around the Noah-type ET** (the first three runs of the
  table above) the trained fields used the riparian term as a summer ET sink (`riva` 0.85 to
  0.90) and ran early at all 20 tier-1 locations. Pinning `riva` to zero removes the symptom,
  the exchange terms (`sacx`) the cause.
- **A second seed of the first recipe** landed in the same state and moved the tier-1 mean by
  0.024. That is the spread the later readings use.
- **State carry between year chunks** (`carry`) lifted the selection score from 0.845 to
  0.877. The variant that also holds the drainage flux fixed damped the Sierra flood peaks
  (99.9th-percentile flow −25 % against −6 %).
- **All 64 embedding coordinates** instead of 16 components: higher in training, lower at ORO
  (0.859 against 0.896) and on arcs outside the trained cells.
- **Gradients through two or three water years.** Three years fixed Shasta's July to November
  volume (−3.5 % against −27.5 %) and lost the Sierra peaks (the five largest annual maxima
  −46 % against −24 %). Added timing and peak terms in the loss did not recover them.
- **A lower cap on the learned rain/snow threshold.** Capped at 1.5 °C on dPL-26's field,
  Trinity loses most of its gain and ORO recovers only to 0.871. No such run was trained.
- **Family shares divided by the entities present in a chunk.** The USGS family then carried
  about 0.57 of the loss against a share of 0.22, because its records start earlier. `all` and
  the fixed family scale (`kref05`) correct that.
- **Family shares by observed volume or by footprint area** (the first two rows of the table):
  not resolved on one seed.

**The product** (median monthly KGE on the held-out WY1976–85)

- **Constant factors per arc on the runoff components**, fitted on dPL-26: 0.77 to 0.78 over
  all arcs. Not refitted on dPL-CalSim.
- **A daily LSTM on the model states**, fitted on dPL-26: failed its out-of-fold check
  (unimpaired-flow subbasins 0.771 against the model's 0.826).
- **Corrections of the system flows.** A constant volume factor (0.886), gradient-boosted
  trees (0.891) and a daily LSTM (0.913) all lose to the model's own flow (0.922). On training
  years they win (0.95 against 0.90). The ratio of CalSim3 to the model drifts by decade (New
  Hogan 1.26 to 1.31 before 1980, 1.02 in the 1980s), so a correction fitted on other years
  moves the held-out decade away from CalSim3.
- **A daily LSTM on the non-anchor arcs**: 0.719 against the model's 0.604. The plan's rule
  would have adopted it. The model's flow was kept by choice, after the held-out numbers were
  read.
- **An LSTM that gives the flow outright**, not a correction: behind the correction form on
  systems and on non-anchor arcs.

## Open

- The share model's response is fitted on uniform climate changes. Under WGEN weather it is
  checked on scenario 12 only: from scenario 1 the product misses the model's own share-arc
  response by 0.05 % in volume and 0.18 points in the April–July share at the median.
- No warmed VIC run exists for a like-for-like comparison of the response.
- The held-out decade WY1976–85 has been read by three experiments.
- A rain/snow partition that does not switch whole storm days (the candidate fix for ORO) has
  not been built.
