# CalSim3 rim inflows

CalSim3 takes monthly inflows on 196 rim arcs. The current model, **dPL-CalSim**, is a
[learned-parameter](learned_parameters.md) SAC-SMA trained on four kinds of flow record at
once, and the **CalSim3 rim-inflow product** is its monthly flow on those arcs, WY1950–2015, in
TAF. This page says what the model trains on, how it is checked, what the product is and how
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
WY1950–2018. Family weights follow the area the families cover, and a fixed scale per family
keeps one family's loss units from dominating the others.

Three runs are the steps to the current model. Each is named for what it trains on.

| Name | Trains on | Run folder under `artifacts/dpl/multifamily/` |
|---|---|---|
| dPL-26 | 17 CDEC + 9 unimpaired-flow entities | `noah_cdec_uf_sacx_carry_px_aef` |
| dPL-95 | the same + 69 USGS gauges | `noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef` |
| **dPL-CalSim** | the same + 64 CalSim3 arcs, with WY1976–85 held out of every family | `noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef` |

All three use Priestley–Taylor PET, the soil-moisture-limited ET with the SAC-SMA exchange
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
works for perturbed and stochastic climate.

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
(`calsim_product/product_info.json`):

| Arcs | dPL-CalSim | Product |
|---|---|---|
| 139 share arcs | 0.70 | **0.83** |
| 50 non-anchor arcs | 0.60 | 0.60 |
| 7 single-arc systems | 0.87 | 0.87 |
| All 196 | 0.69 | **0.81** |

Over all 196 arcs the product's mean is 0.72 and its 10th percentile 0.42; 7 arcs are below
zero (13 for the model alone). At the five validation climate points the share arcs' response
differs from the model's by at most 0.42 % in volume and 0.64 points in the April–July share at
the 90th percentile (`calsim_product/response_gate.csv`), inside the limits set beforehand
(median 1, 90th percentile 3).

## What was tried and not adopted

Each with a plan frozen before its held-out numbers were read. The full list is in
[`artifacts/dpl/RUNS.md`](../artifacts/dpl/RUNS.md#tried-and-not-adopted).

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
  as snow and damps moderate floods. dPL-95 recovered most of it.
- **One seed.** Each of the three runs is a single seed, and dPL-CalSim was adopted without the
  control runs its plan called for (`provenance/DEVIATIONS.txt` in its folder).
- **The held-out decade has been read** by three experiments, so a further design choice read
  against WY1976–85 is not a clean test.

## Open

- The product starts in WY1950; CalSim3 starts in WY1922.
- The climate points used so far are uniform changes (degrees added, precipitation scaled).
  They stand in for the WGEN daily scenario weather.
- No warmed VIC run exists for a like-for-like comparison of the response.

## Commands

```bash
sacsma dpl calsim tier1 <run>
sacsma dpl calsim tier2 <run> --components parts --trace-python <python of the sacsma-gis environment>
sacsma dpl calsim atlas <run>
sacsma dpl calsim product fit <run>                    # -> <run>/calsim_product/
sacsma dpl calsim product apply <run> --tier2 <dir>    # the product for another tier-2 pass
sacsma verify product                                  # apply reproduces the tracked product
```

The full sequence, from the data to the product, is in [Reproduce](reproduce.md). The files
each command writes are listed in [`artifacts/README.md`](../artifacts/README.md).
