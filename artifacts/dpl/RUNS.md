# Learned-parameter runs: current state

What is in `artifacts/dpl/`, which run is current, what each run scores, what was tried and
not adopted, and what is still open. The method is in the guide:
[Learned parameters](../../docs/learned_parameters.md) and
[CalSim3 rim inflows](../../docs/calsim3_rim_inflows.md).

## Current model and product

**dPL-CalSim** (`multifamily/noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef`),
adopted 2026-09-30, and the **CalSim3 rim-inflow product** in its `calsim_product/` folder,
adopted 2026-10-01: monthly flow on the 196 rim arcs, WY1950–2015, TAF.

Median monthly KGE against CalSim3 on the held-out water years 1976–85:

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.699 | 0.834 |
| 50 non-anchor arcs | 0.604 | 0.604 |
| 7 single-arc systems | 0.870 | 0.870 |
| All 196 | 0.690 | 0.813 |

Source: `calsim_product/product_info.json`. `sacsma verify product` checks that
`sacsma dpl calsim product apply` reproduces the tracked `rim_inflow_monthly.csv`.

## Multi-family runs (`multifamily/`)

One parameter network trained on several families of flow record. Mean KGE of each family's
entities over their training years (`metrics_entities.csv`), and the two checks against CalSim3:
the mean over the 20 tier-1 locations and the median over the 196 arcs of tier 2. The first
five runs are checked over WY1950–84, dPL-CalSim over its held-out WY1976–85.

| Name | Folder | Entities | usgs | cdec | uf | arcs | Tier 1 mean | Tier 2 median |
|---|---|---|---|---|---|---|---|---|
| – | `noah_cdec_uf_usgs` | 95 | 0.587 | 0.786 | 0.859 | – | 0.812 | 0.598 |
| – | `noah_cdec_uf_usgs_areaw` | 95 | 0.592 | 0.791 | 0.879 | – | 0.825 | 0.599 |
| – | `noah_cdec_uf` | 26 | – | 0.790 | 0.893 | – | 0.794 | 0.576 |
| dPL-26 | `noah_cdec_uf_sacx_carry_px_aef` | 26 | – | 0.874 | 0.934 | – | 0.859 | 0.628 |
| dPL-95 | `noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef` | 95 | 0.687 | 0.880 | 0.935 | – | 0.857 | 0.657 |
| **dPL-CalSim** | `noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef` | 159 | 0.691 | 0.872 | 0.929 | 0.715 | 0.908 | 0.690 |

The first three are the first recipe and are kept for the record. dPL-26 was the base from
2026-09-28, dPL-95 from 2026-09-29; dPL-CalSim starts from dPL-95's recipe. The parts of a
folder name are explained in the [Glossary](../../docs/glossary.md#run-folder-names).

Each run tracks ten files: `checkpoints/best.pt`, `train_log.csv`, `metrics_entities.csv`,
`params_dpl.csv`, `params_canopy.csv`, `sim_daily_mm.npz`, `tier1/tier1_metrics.csv`,
`tier2/tier2_metrics.csv`, `tier2/tier2_arcs.csv` and `atlas/calsim_validation_atlas.html`. The
atlases and `sim_daily_mm.npz` are git-LFS files. dPL-95 adds a small `provenance/`. dPL-CalSim
adds `metrics_entities_holdout.csv`, `tier2/tier2_anchor_rescaled.csv`, `provenance/` (the plan
frozen before the run, its bars, the readout and the deviations) and `calsim_product/`.

## Runs on the 15 CDEC watersheds

Mean daily KGE over the 15 watersheds, calibration WY1989–2003 and validation WY2004–2018, from
each folder's `metrics_*.csv`. The parameter runs are scored through the reference model; the
LSTM ensembles are scored on the mean flow of their seeds.

| Folder | What it is | Calibration | Validation |
|---|---|---|---|
| `hamon` | learned parameters on 1/16° cells, Hamon PET | 0.817 | 0.836 |
| `pt` | Priestley–Taylor PET | 0.799 | 0.826 |
| `noah` | soil-moisture-limited ET, parameters that respond to climate | 0.779 | 0.804 |
| `hybrid` | LSTM on `noah`'s flow, 3 seeds | 0.922 | 0.877 |
| `hybrid_dt` | `hybrid` trained to keep `noah`'s response to climate, 3 seeds | 0.873 | 0.849 |
| `lstm` | LSTM without the physics (control), 3 seeds | 0.909 | 0.835 |
| `superseded/hamon_dense` | learned parameters on the original 7,891 HRUs | 0.806 | 0.840 |
| `superseded/noah_noca` | `noah` before its parameters responded to climate | 0.767 | 0.799 |
| `superseded/hybrid_noca` | LSTM on `noah_noca`, 8 seeds | 0.917 | 0.869 |
| `superseded/hybrid_dt_noca` | the same with a single +2 °C response term, 8 seeds | 0.916 | 0.864 |

The archived GA parameters score 0.805 / 0.768 on the same basis. `superseded/` is kept because
it shows the steps and because `superseded/noah_noca` still supplies the physics simulation of
the earlier hybrids. `noah/fidelity/` is the benchmark of the differentiable model against the
reference model, not a run. `figures/` holds the comparison figures and their tables.

## What is tracked and what is local

Tracked: the folders above. Everything under `artifacts/dpl/_local/` is ignored by git: caches
(`_local/cache/`), scratch runs (`_local/testing/`), evaluation scratch (`_local/eval/`) and the
runs of the training program that were not adopted (`_local/multifamily/`). In a tracked run
folder, the files beyond those listed above are also ignored and regenerate from the
checkpoint.

## Regenerating a run's outputs

```bash
sacsma dpl evaluate <run>/checkpoints/best.pt    # metrics, parameter tables, sim_daily_mm.npz
sacsma dpl calsim tier1 <run>
sacsma dpl calsim tier2 <run> --trace-python <python of the sacsma-gis environment>
sacsma dpl calsim atlas <run>
sacsma dpl calsim product fit <run>              # dPL-CalSim only; needs the scenario passes
```

The complete sequence is in [Reproduce](../../docs/reproduce.md); the files each command writes
are listed in [`artifacts/README.md`](../README.md).

## Known costs

- **NHG.** Daily KGE 0.827 in dPL-CalSim (dPL-95 0.873, dPL-26 0.924).
- **Unimpaired-flow subbasins, WY1986–2014.** 0.896 against a bar of 0.926, after their targets
  were extended back to WY1950.
- **ORO in dPL-26.** 0.838 against 0.896 before the learned rain/snow threshold (a run that is
  kept locally, not tracked); dPL-95 is at 0.888.
- **Single seed.** No run has a second seed. dPL-95 was adopted without its paired run without
  the family loss scale, dPL-CalSim without its control without the arcs
  (`provenance/DEVIATIONS.txt`).
- **`hybrid`.** Its volume bias moves between calibration and validation at MRC (+1.5 % to
  −8.5 %) and NML (−3.0 % to −6.6 %). For the earlier generation (`superseded/hybrid_noca`)
  the physics run was the better out-of-sample answer on monthly flow against CalSim3 at NML,
  MRC and ORO. That has not been re-read for this one, whose daily validation KGE is above
  `noah`'s at NML and ORO.

## Tried and not adopted

One line per result, so that it is not tried again without a reason. The runs behind these
lines are not tracked (those that still exist are under `_local/`), and the numbers are as
recorded when the runs were made. The dated record they come from is this file as of commit
`f4140c0` (`git show f4140c0:artifacts/dpl/RUNS.md`).

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
- **Fewer response points.** One point at +2 °C (`superseded/hybrid_dt_noca`), then 5 and 8
  points, were tried before the 14 of `hybrid_dt`. With precipitation changes of ±20 % among
  the points, the error at those edges is 3.7 points of annual change, against 10.1 for
  `hybrid`.

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

- The product covers WY1950–2015; CalSim3 starts in WY1922.
- The climate response of the product is trained and checked on uniform changes, which stand
  in for the WGEN daily scenario weather.
- No warmed VIC run exists for a like-for-like comparison of the response.
- The held-out decade WY1976–85 has been read by three experiments.
- A rain/snow partition that does not switch whole storm days (the candidate fix for ORO) has
  not been built.
