# CalSim3 rim inflows

CalSim3 takes monthly inflows on 196 rim arcs. The current model, **dPL-CalSim**, is a
[learned-parameter](learned_parameters.md) SAC-SMA trained on four kinds of flow record at
once, and the **CalSim3 rim-inflow product** is its monthly flow on those arcs, in TAF, from
WY1916 to 2018, under the historical weather and two WGEN weather sequences. This page says what the model trains on, how it is checked, what the product is and how
well it does on a decade nothing was fitted to.

## What the model trains on

One parameter network is trained on a set of *entities*. An entity is one flow record with the
grid cells that drain to it. Entities come in four *families*:

| Family | Entities | Record | Loss |
|---|---|---|---|
| `usgs_daily` | 69 USGS gauges inside the CalSim3 domain | daily, from WY1950 | daily |
| `cdec_daily` | 17 CDEC full-natural-flow stations | daily, from 1986 to 1988 (two from 1999) | daily |
| `uf_monthly` | 9 DWR unimpaired-flow subbasins | monthly | monthly |
| `calsim_monthly` | 64 CalSim3 arcs with a record of their own (of 196 in the table) | monthly | monthly volume |

The training domain is `--domain multifamily`. Each entity is scored over its own record inside
WY1950–2018; the unimpaired-flow subbasins over WY1985–2014, which dPL-CalSim extends back to
WY1950. Family weights follow the area the families cover, and a fixed scale per family
keeps one family's loss units from dominating the others.

Two runs are the steps to the current model. Each is named for what it trains on.

| Name | Trains on | Run folder (`artifacts/models/dpl/multifamily/`, `results/dpl/multifamily/`) |
|---|---|---|
| dPL-26 | 17 CDEC + 9 unimpaired-flow entities | `noah_cdec_uf_sacx_carry_px_aef` |
| **dPL-CalSim** | the same + 69 USGS gauges + 64 CalSim3 arcs, with WY1976–85 held out of every family | `noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef` |

Both use Priestley–Taylor PET, the soil-moisture-limited ET with the SAC-SMA exchange
terms kept, a learned rain/snow threshold, and the AlphaEarth inputs. The folder names spell
the recipe; the parts are listed in the [Glossary](glossary.md#run-folder-names).

## How a run is checked against CalSim3

| Check | What it scores | Command |
|---|---|---|
| Entities | each entity against its own record, at its own time step | `sacsma dpl evaluate <run>/checkpoints/best.pt` |
| Tier 1 | monthly volume at 20 training locations, against FLOW-UNIMPAIRED where a rim system has one and against the sum of its INFLOW arcs elsewhere | `sacsma dpl calsim tier1 <run>` |
| Tier 2 | every rim arc against its own CalSim3 series; arcs outside every trained footprint are simulated on their own cells | `sacsma dpl calsim tier2 <run> --trace-python <python of the sacsma-gis environment>` |
| Atlas | one HTML page per run: maps, a tab per location, the arc tables | `sacsma dpl calsim atlas <run>` |

Scoring starts every simulation from the same rule: the first ten water years of the scored
period looped 20 times from a cold start. Nothing before WY1950 is read, so the rule also
works for perturbed and stochastic climate. Each check's details are in
[How the checks work](#how-the-checks-work) below.

dPL-CalSim holds WY1976–85 out of every family, so its checks are on that decade. The CDEC
daily records start after it, so that family has no held-out score. From the run's tracked
tables:

| | Median monthly KGE, WY1976–85 |
|---|---|
| Tier 1, 20 locations | 0.92 |
| The 64 trained arcs | 0.73 |
| The 9 unimpaired-flow subbasins | 0.90 |
| All 196 arcs (tier 2) | 0.69 |

The whole-watershed flows are good. Single arcs inside a multi-arc watershed are weaker,
because the model's split of a watershed's flow among its arcs is poorer than the total. That
is what the product corrects.

## The product

Arcs are of three kinds.

| Kind | Arcs | What the product uses |
|---|---|---|
| Single-arc system | 7 | dPL-CalSim's flow |
| Share arc: an arc of one of the 11 multi-arc systems | 139 | the system's dPL-CalSim flow, divided among its arcs by the share model |
| Non-anchor arc: no system to close to | 50 | dPL-CalSim's flow |

A *system* is a group of arcs that closes to one CalSim3 anchor (18 systems: the 11 multi-arc
ones and the 7 single arcs).

**The share model** is a small network that gives each share arc's fraction of its system's
flow in a month. It reads the model's own shares and their lags, the fractions of the runoff
components, the system flow, the month and the arc area. Each system's arcs sum to the
dPL-CalSim system flow over every water year, so a system's volume, and its response to a
changed climate, stay the model's own. The arcs' response is held close to the model's by a
penalty in the fit and is checked at climate points the fit never saw.

Skill on the held-out decade, median monthly KGE against CalSim3
(`artifacts/product/calsim3/historical_livneh_unsplit/product_metrics.csv`):

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.70 | **0.83** |
| 50 non-anchor arcs | 0.60 | 0.60 |
| 7 single-arc systems | 0.87 | 0.87 |
| All 196 | 0.69 | **0.81** |

Over all 196 arcs the product's mean is 0.72 and its 10th percentile 0.42; 7 arcs are below
zero (13 for the model alone). At the five validation climate points the share arcs' response
differs from the model's by at most 0.42 % in volume and 0.64 points in the April–July share at
the 90th percentile (`artifacts/product/calsim3/response_gate.csv`), inside the limits set beforehand
(median 1, 90th percentile 3).

**One series per forcing.** Each series is one continuous run over the whole forcing record:
the spin-up loops WY1916–1925 twenty times from the cold start, the run goes on from October
1915 to December 2018, and the series holds its complete water years, WY1916–2018. The forcings are the historical Livneh grid (the training
forcing) and WGEN Product A scenarios 1 and 12 ([`artifacts/product/calsim3/`](../artifacts/product/calsim3/README.md)).
No training target and no fit reaches before WY1950. On WY1922–1949, against CalSim3, the
historical series scores (median monthly KGE):

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.67 | **0.78** |
| 50 non-anchor arcs | 0.42 | 0.42 |
| 7 single-arc systems | 0.78 | 0.78 |
| All 196 | 0.64 | **0.76** |

**Under the WGEN weather.** Over WY1916–2018 the product carries 27,540 TAF a year under the
historical weather, 27,740 under scenario 1 and 27,650 under scenario 12, and the April–July
share of the year's flow falls from 41.7 % to 38.6 % and 31.1 %. The warmer early record of
scenario 1 (its temperature is detrended to 1991–2020) moves the melt earlier and leaves the
volume within 1 %: the response the comparison of the calibrated sets finds in VIC, not in the
Hamon-PET calibrations, which lose about 3 % ([Calibrated SAC-SMA](calibrated_sacsma.md)). On
the share arcs the product keeps the model's own response. From scenario 1 to scenario 12 the
model's arc volume changes by −0.9 % and its April–July share by −5.7 points at the median;
the product differs from that by 0.05 % and 0.18 points at the median and by 0.11 % and 0.46
points at the 90th percentile, within half the limits set for the fit's validation points.

## What was tried and not adopted

Each with a plan frozen before its held-out numbers were read. The full list is in
[Runs](runs.md#tried-and-not-adopted).

- Corrections to the system flows (a constant volume factor, gradient-boosted trees, a daily
  LSTM) beat the model on training years and lose to it on the held-out decade, because the
  ratio between CalSim3 and the model drifts from decade to decade.
- A daily LSTM correction on the non-anchor arcs scored 0.72 against the model's 0.60 on the
  held-out decade. The model's own flow was kept there by choice.

## Known costs

- **NHG (New Hogan).** Daily KGE 0.83 in dPL-CalSim, against 0.92 in dPL-26.
- **Unimpaired-flow subbasins after WY1985.** Extending their targets back to WY1950 raised
  the score on the earlier years and lowered it on WY1986–2014 (0.90 against a bar of 0.93).
- **ORO (Oroville) in dPL-26.** The learned rain/snow threshold stores cool-storm precipitation
  as snow and damps moderate floods. dPL-CalSim recovered most of it (daily KGE 0.889 against 0.838).
- **One seed.** Each run is a single seed, and dPL-CalSim was adopted without the
  control runs its plan called for (`provenance/DEVIATIONS.txt` in its folder).
- **The held-out decade has been read** by three experiments, so a further design choice read
  against WY1976–85 is not a clean test.
- **The start is remembered for decades in three southern Sierra arcs.** In McClure (Merced),
  Hetch Hetchy (Tuolumne) and Millerton (San Joaquin) the slow lower-zone store keeps its
  starting state for decades. A run started in WY1950 instead of WY1916 differs there by a few
  TAF a year, mostly in dry years (17 % at McClure in WY1977). Over all arcs the median
  difference is below 0.03 % from WY1952 on.

## Open

- The share model is fitted on uniform climate changes (degrees added, precipitation scaled).
  Under WGEN weather it is checked on scenario 12 only.
- No warmed VIC run exists for a like-for-like comparison of the response.

## How the checks work

**Tier 1** scores monthly volume (TAF) at the twenty training locations of
`data/inputs/calsim3/tier1_sets.csv`: against FLOW-UNIMPAIRED where a rim system carries one
(ten anchors), against the sum of the member INFLOW arcs elsewhere (ten arc sums). Volumes use
the `CalSim3_Merged` polygon areas, so no third area enters. A location whose arcs all belong
to a larger one (Shasta inside Red Bluff) is *nested*: it enters the skill statistics but not
the volume totals (`nested_in`). The USGS creeks train over their whole records, which reach
into WY1950–84, so every location the creeks reach is also scored over a *trimmed window*: the
run of at least 20 water years inside WY1950–84 in which the creek gauges covered the least of
the location (`sacsma dpl calsim windows` derives it from the creek records alone, so it is the
same for every run). The full-window score is the one reported; the atlas sets the two side by
side. At the ten anchored locations the reference coincides with the source of a training
target, so that score is a temporal holdout rather than an independent reference.

**Tier 2** aggregates the runoff of every grid cell onto every rim INFLOW polygon and scores
each arc on its own series. Arcs no trained entity lists are simulated on their own cells, with
flow lengths traced to the polygon exit, and flagged `basis = extrapolated`.
`trained_cell_frac` gives each arc's share of area on cells the run trained on: an arc on a
USGS creek's footprint is extrapolated by construction while its cells were fitted, so the
regionalization test proper is the arcs at 0, which the summary and the atlas report apart.
The tracing needs the HydroSHEDS tiles and `rasterio`, which lives only in the `sacsma-gis`
environment, hence `--trace-python`; without it the extrapolated arcs fall back to
straight-line flow lengths × 1.5, which the output records. `--components parts` adds the
routed runoff components each arc's flow is made of (quick and interflow, supplemental and
primary baseflow), which the share model reads.

**The atlas** is one self-contained HTML page per run: what the run trained on and its family
weights, domain maps, a tab per location (scores, series, the arc table with each arc's
derivation class from `data/targets/calsim3/calsim3_arc_derivation.csv`), the arcs outside every
trained footprint, the footprints of each family, the USGS creek watersheds that overlap each
location with their records inside the window, and the full window against the trimmed ones.
Its images are embedded, so the page needs no other file.

**Runs with a holdout.** dPL-CalSim held WY1976–85 out of every family, so it is validated on
those years instead of WY1950–84: the evaluator adds `metrics_holdout.csv`; tier 1 and tier 2
score the windows `WY1976-85`, `WY1976-84` (comparable with runs that trained on WY1985) and
`WY1950-84_mixed`, and keep the held-out years out of the `train` window; tier 2 adds
`WY1976-85_own` (each arc's own gauge-record months) and `tier2_anchor_rescaled.csv`, each arc
after its system is rescaled to the CalSim3 anchor, an evaluation-only score.

## Commands

```bash
sacsma dpl calsim tier1 <run>
sacsma dpl calsim tier2 <run> --components parts --trace-python <python of the sacsma-gis environment>
sacsma dpl calsim atlas <run>
sacsma dpl calsim product fit <run>                              # -> artifacts/product/calsim3/
sacsma dpl calsim product apply --forcing <name>                # -> artifacts/product/calsim3/<name>/
sacsma verify product                                            # apply repeats every tracked series
```

`<run>` is any of the run's folders. The full sequence, from the data to the product, is in
[Reproduce](reproduce.md). The files each command writes are listed in
[`artifacts/results/dpl/README.md`](../artifacts/results/dpl/README.md) and
[`artifacts/product/calsim3/README.md`](../artifacts/product/calsim3/README.md).
