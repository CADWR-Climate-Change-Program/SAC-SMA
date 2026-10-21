# CalSim3 rim inflows

CalSim3 takes monthly inflows on 196 rim arcs. The current model, **dPL-CalSim** (`8_calsim`),
is a [learned-parameter](learned_parameters.md) SAC-SMA trained on four kinds of flow record at
once, and the **CalSim3 rim-inflow product** is its monthly flow on those arcs, in TAF, from
WY1916 to 2018, under the historical weather and two WGEN weather sequences. This page says what
the model trains on, how it is checked, what the product is and how well it does on a decade
nothing was fitted to.

## What the model trains on

One parameter network is trained on a set of *entities*. An entity is one flow record with the
grid cells that drain to it. Entities come in four *families*:

| Family | Entities | Record | Loss |
|---|---|---|---|
| `usgs_daily` | 69 USGS gauges inside the CalSim3 domain | daily, from WY1950 | daily |
| `cdec_daily` | 17 CDEC full-natural-flow stations | daily, from 1986 to 1988 (two from 1999); with it CDEC's monthly full natural flow at 15 of them, from WY1950 | daily; monthly for the monthly record |
| `uf_monthly` | 9 DWR unimpaired-flow subbasins | monthly | monthly |
| `calsim_monthly` | 64 CalSim3 arcs with a record of their own (of 196 in the table) | monthly | monthly volume |

The training domain is `--domain multifamily`. Each entity is scored over its own record inside
WY1950–2018; the unimpaired-flow subbasins over WY1985–2014, which dPL-CalSim extends back to
WY1950. Family weights follow the area the families cover, and a fixed scale per family
keeps one family's loss units from dominating the others.

dPL-CalSim is the run `8_calsim` (`artifacts/models/dpl/multifamily/8_calsim/`,
`artifacts/results/dpl/multifamily/8_calsim/`), the last rung of [the ladder](runs.md#the-ladder).
It keeps the physics and inputs of the rung below it, `7_aef`: Priestley–Taylor PET, the
soil-moisture-limited ET with the SAC-SMA exchange terms kept, a learned rain/snow threshold, the
deep store ([Learned parameters](learned_parameters.md#how-it-is-built)) and the AlphaEarth
inputs (`aef_u`). It trains on all 159 entities of the four families, on every water year but
WY1976–85, which is held out of every family, with wider ranges for four parameters the GA box
pins (`MBASE`, `NMF`, `Velo`, `zperc`; [Model equations](equations.md#6-parameter-table)).

It is trained in two stages:

1. The warm start trains that recipe without the deep store, from scratch, for 120 epochs. Its
   network is kept in the run's provenance (`provenance/warm_start.pt`).
2. The fine-tune runs 45 epochs from the warm start at a lower learning rate, with four
   additions:
   - The deep store, with its head new at the start.
   - Cleaned targets. The days and months that an audit of the training records found
     unreliable are left out: an unreliable dry season, the partners of CDEC's negative days,
     impaired summers, records that a USGS gauge at the same place already trains, and forcing
     eras the Livneh precipitation gets wrong. The tables are
     [the CDEC screens](../data/targets/cdec/README.md#the-training-screens-of-dpl-calsim) and
     [the location masks](../data/targets/usgs/README.md#the-location-mask). The Kings River
     gauge trains on its record with the powerhouse flow added, and two arcs on a corrected depth.
   - CDEC's monthly full natural flow at the CDEC sites, from WY1950, beside the daily
     records. It covers the decades before the daily records and the late summers where the daily
     record is screened out. The CDEC family's weight is split evenly between the two.
   - A dry-season term: for each entity and scored water year,
     $\lambda\,\big(\ln(V_{sim} + \epsilon) - \ln(V_{obs} + \epsilon)\big)^2$, with $V$ the
     July–September volume, $\epsilon$ a tenth of the entity's median observed July–September
     volume, and $\lambda = 0.113$, about a tenth of the objective at the warm start's field.

The recipe and the scores of every rung are in [Runs](runs.md#the-ladder).

## How a run is checked against CalSim3

| Check | What it scores | Command |
|---|---|---|
| Entities | each entity against its own record, at its own time step | `sacsma dpl evaluate <run>/checkpoints/best.pt --score-holdout` |
| Tier 1 | monthly volume at 20 training locations, against FLOW-UNIMPAIRED where a rim system has one and against the sum of its INFLOW arcs elsewhere | `sacsma dpl calsim tier1 <run> --score-holdout` |
| Tier 2 | every rim arc against its own CalSim3 series; arcs outside every trained footprint are simulated on their own cells | `sacsma dpl calsim tier2 <run> --score-holdout --trace-python <python of the sacsma-gis environment>` |
| Atlas | one HTML page per run: maps, a tab per location, the arc tables | `sacsma dpl calsim atlas <run> --score-holdout` |

A run with held-out water years is scored on them only when `--score-holdout` is given. Without
it the evaluator leaves those years unread, tier 1 and the atlas refuse the run, and tier 2
writes its flows and no score.

Scoring starts every simulation from the same rule: the first ten water years of the scored
period looped 20 times from a cold start. Nothing before WY1950 is read, so the rule also
works for perturbed and stochastic climate. Each check's details are in
[How the checks work](#how-the-checks-work) below.

dPL-CalSim holds WY1976–85 out of every family, so its checks are on that decade. The CDEC
daily records start after it, so that family has no held-out score. From the run's tracked
tables (`metrics_holdout.csv`, `tier1/tier1_metrics.csv`, `tier2/tier2_metrics.csv` in
`artifacts/results/dpl/multifamily/8_calsim/`):

| | Median monthly KGE, WY1976–85 |
|---|---|
| Tier 1, 20 locations | 0.92 |
| The 64 trained arcs | 0.73 |
| The 9 unimpaired-flow subbasins | 0.89 |
| All 196 arcs (tier 2) | 0.68 |

The means, with the 49 USGS gauges (daily), are in [Runs](runs.md#current-model-and-product).

The whole-watershed flows are good. Single arcs inside a multi-arc watershed are weaker,
because the model's split of a watershed's flow among its arcs is poorer than the total. The
product corrects that split.

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
| 139 share arcs | 0.71 | **0.84** |
| 50 non-anchor arcs | 0.58 | 0.58 |
| 7 single-arc systems | 0.91 | 0.91 |
| All 196 | 0.68 | **0.81** |

Over all 196 arcs the product's mean is 0.73 and its 10th percentile 0.48; 6 arcs are below
zero (14 for the model alone). At the five validation climate points the share arcs' response
differs from the model's by at most 0.32 % in volume and 0.73 points in the April–July share at
the 90th percentile (`artifacts/product/calsim3/response_gate.csv`), inside the limits set
beforehand (median 1, 90th percentile 3).

**One series per forcing.** Each series is one continuous run over the whole forcing record:
the spin-up loops WY1916–1925 twenty times from the cold start, the run goes on from October
1915 to December 2018, and the series holds its complete water years, WY1916–2018. The forcings
are the historical Livneh grid (the training forcing) and WGEN Product A scenarios 1 and 12
([`artifacts/product/calsim3/`](../artifacts/product/calsim3/README.md)).
No training target and no fit reaches before WY1950. On WY1922–1949, against CalSim3, the
historical series scores (median monthly KGE):

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.65 | **0.78** |
| 50 non-anchor arcs | 0.42 | 0.42 |
| 7 single-arc systems | 0.79 | 0.79 |
| All 196 | 0.63 | **0.76** |

**Under the WGEN weather.** Over WY1916–2018 the product carries 28,140 TAF a year under the
historical weather, 28,400 under scenario 1 and 28,420 under scenario 12, and the April–July
share of the year's flow falls from 40.9 % to 37.3 % and 28.8 %. The warmer early record of
scenario 1 (its temperature is detrended to 1991–2020) moves the melt earlier and leaves the
volume within 1 %: the response the comparison of the calibrated sets finds in VIC, not in the
Hamon-PET calibrations, which lose about 3 % ([Calibrated SAC-SMA](calibrated_sacsma.md)). On
the share arcs the product keeps the model's own response. From scenario 1 to scenario 12 the
model's arc volume changes by −0.3 % and its April–July share by −7.8 points at the median;
the product differs from that by 0.05 % and 0.20 points at the median and by 0.13 % and 0.59
points at the 90th percentile, within half the limits set for the fit's validation points.

## What was tried and not adopted

The corrections tried for the product, each with a plan frozen before its held-out numbers were
read, are listed in [Runs](runs.md#tried-and-not-adopted).

## Known costs

- The unimpaired-flow subbasins have a mean KGE of 0.88 on the held-out WY1976–85, lowest at
  Chowchilla (0.76) and Putah Creek (0.81). The cleaning leaves 5 of the 9 to train on (the
  whole records of Cache Creek, Red Bluff, the Cosumnes and the Fresno are masked); those 5
  score 0.94 over their training years (back to WY1950) and 0.88 on their DWR record without
  the held-out WY1985 (WY1986–2014, `kge_reg` in `metrics.csv`).
- MKM and TLG in the [benchmark](benchmark.md): on WGEN Product A scenario 1 against CDEC's
  monthly full natural flow, WY1991–2018, the median KGE over the 12 sites is 0.94; the lowest
  are MKM (0.86, volume −9 %), TLG (0.87, −6 %) and NML (0.87, +5 %).
- dPL-CalSim is a single seed; no rung of the ladder has a second.
- The model remembers its start for years: the deep store drains over 8 to 9 years. A run
  started in WY1950 (with the scoring spin-up, WY1950–59 looped) instead of WY1916 carries
  8.7 % more water over all arcs in WY1950, 1.2 to 1.7 % in WY1951–55 and less than 1 % from
  WY1956. The median arc differs by less than 0.1 % from WY1978 and less than 0.01 % from
  WY1991. The training and the scores start in WY1950; the product starts in WY1916.

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
the volume totals (`nested_in`). The Yuba's entity leaves out the two Deer Creek arcs, so the
Yuba is also scored against the sum of the arcs it simulates (`ref_kind = arcsum_covered`, a
row not counted among the twenty). At the ten anchored locations the reference coincides with
the source of a training target, so that score is a temporal holdout rather than an
independent reference.

**Tier 2** aggregates the runoff of every grid cell onto every rim INFLOW polygon and scores
each arc on its own series. Arcs no trained entity lists are simulated on their own cells, with
flow lengths traced to the polygon exit, and flagged `basis = extrapolated`.
`trained_cell_frac` gives each arc's share of area on cells the run trained on: an arc on a
USGS creek's footprint is extrapolated by construction while its cells were fitted, so the
regionalization test proper is the arcs at 0, which the summary and the atlas report apart.
The tracing needs the HydroSHEDS tiles and `rasterio`, which lives only in the `sacsma-gis`
environment, hence `--trace-python`; without it the extrapolated arcs fall back to
straight-line flow lengths × 1.5, which the output records. dPL-CalSim's traced lengths are
tracked (`models/dpl/multifamily/8_calsim/tier2_extension_cells.csv`), and `product apply`
reuses them, so applying the product needs no tracing. `--components parts` adds the
routed runoff components each arc's flow is made of (quick and interflow, supplemental and
primary baseflow), which the share model reads.

**The atlas** is one self-contained HTML page per run: what the run trained on and its family
weights, domain maps, a tab per location (scores, series, the arc table with each arc's
derivation class from `data/targets/calsim3/calsim3_arc_derivation.csv`), the arcs outside every
trained footprint, the footprints of each family, and the USGS creek watersheds that overlap
each location with their records inside the window. Its images are embedded, so the page needs
no other file.

**The windows.** dPL-CalSim holds WY1976–85 out of every family, and the evaluator scores those
years apart (`metrics_holdout.csv`). Tier 1 and tier 2 score each location and arc over four
windows (the `window` column):

| Window | What it scores |
|---|---|
| `WY1976-85` | the held-out decade: the reported score, and the one the maps, figures and summaries show |
| `WY1976-84` | the same without WY1985, the first year of the unimpaired-flow record |
| `WY1950-84_mixed` | WY1950–84, which mixes training years (WY1950–75) with held-out ones |
| `train` | the record window of the location's or arc's entity (WY1985–2014 for an unimpaired-flow subbasin, the record start to 2018 for a CDEC station), without the held-out years (`excluded_wy`) |

Tier 2 adds `WY1976-85_own`, the held-out months of each arc's own gauge record (73 arcs), and
`tier2_anchor_rescaled.csv`, each arc after its system is rescaled to the CalSim3 anchor, an
evaluation-only score.

## Commands

```bash
sacsma dpl calsim tier1 <run> --score-holdout
sacsma dpl calsim tier2 <run> --components parts --score-holdout --trace-python <python of the sacsma-gis environment>
sacsma dpl calsim atlas <run> --score-holdout
sacsma dpl calsim product fit <run> --score-holdout                 # -> artifacts/product/calsim3/
sacsma dpl calsim product apply --forcing <name> --score-holdout   # -> artifacts/product/calsim3/<name>/
sacsma verify product                                               # apply repeats every tracked series
```

A forcing longer than one pass, such as a 1,000-year WGEN sequence (a datetime64 axis spans 584
years), runs in segments. `--forcing` takes the path of a segment's store file, kept outside
`data/`, and `--start` and `--end` set the span of its pass. `--save-state` keeps the model state
on a pass's last day, and the next segment's pass starts from it with `--init-state` instead of
the spin-up, so the segments give the flow of one unbroken run; `sacsma verify learned` checks
that a run split this way matches it unsplit, bit for bit. The segments' passes are then joined
into one and applied with `--tier2 <joined pass> --centre-wy all`, which centres the share
model's inputs on the whole record rather than on WY1950-2015, since a synthetic sequence has no
such years.

`<run>` is any of the run's folders (`artifacts/models/dpl/multifamily/8_calsim`,
`artifacts/results/dpl/multifamily/8_calsim` or its folder under `artifacts/_local/runs/`). The
full sequence, from the data to the product, is in [Reproduce](reproduce.md). The files each
command writes are listed in
[`artifacts/results/dpl/README.md`](../artifacts/results/dpl/README.md) and
[`artifacts/product/calsim3/README.md`](../artifacts/product/calsim3/README.md).
