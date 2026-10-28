# Glossary

Words that carry a specific meaning in this repository, and words that are used for more than
one thing.

## Domains, sets, families, entities

| Word | Meaning |
|---|---|
| Domain | What `--domain` selects: a group of watersheds with their modeling units. `15cdec`, `15cdec_grid`, `9unimp`, `11obs`, `12rim`, `multifamily`. |
| Calibration set | One of the four GA calibrations: CDEC15 (`15cdec`), Rim12 (`12rim`), Observed11 (`11obs`), Unimpaired9 (`9unimp`). |
| `15cdec_grid` | The 15 CDEC watersheds on 1/16° grid cells in place of the original HRUs. Its GA calibration is the starting field of every learned-parameter run trained from scratch; its forcing feeds the LSTM hybrids. |
| `multifamily` | The grid cells and CalSim3 catchment outlines the learned-parameter runs train on: for dPL-CalSim every USGS, CDEC and unimpaired-flow entity and the 64 tier-A CalSim3 arcs (159 of the registry's 291), for rungs 2 to 7 of the ladder the 15 CDEC records. |
| Entity | One flow record together with the grid cells that drain to it. |
| Family | The entities of one data source: `usgs_daily`, `cdec_daily`, `uf_monthly`, `calsim_monthly`. |
| HRU | Hydrologic response unit: a grid cell intersected with a soil class, the original modeling unit. |
| CalLite sets | The three per-watershed monthly calibrations (`12rim`, `11obs`, `9unimp`). |

## Flow data

| Word | Meaning |
|---|---|
| FNF | Full natural flow: the flow a river would carry without storage or diversion upstream, as estimated by CDEC. |
| Unimpaired flow | DWR's monthly estimate of the same thing for 24 Central Valley subbasins (the "UF" series). At the rim it is the same quantity as natural flow. |
| Rim arc | A CalSim3 `INFLOW` arc at the edge of the valley floor: where a watershed's runoff enters the CalSim3 network. 196 of them carry a series. |
| FLOW-UNIMPAIRED | CalSim3's whole-watershed unimpaired series for a rim system. |
| TAF | Thousand acre-feet. Monthly CalSim3, VIC and product volumes are in TAF per month. |
| WY | Water year, October through September, named for the year it ends in. |

## "Anchor" has two meanings

- In the comparison of the calibrated model with CalSim3 (`sacsma calsim`), the anchor is the
  whole-watershed score: a watershed's simulated flow against the CalSim3 reference for that
  watershed, as opposed to one arc at a time.
- For the rim arcs of the learned-parameter model, the anchor is the series a group of arcs must
  add up to (the *closure anchor*), which defines a *system*.

## Kinds of arc in the product

| Word | Meaning |
|---|---|
| System | A group of arcs that shares one anchor series. The product divides the model's flow of the group among its arcs. 18 in all: 11 with several arcs, 7 with one. |
| Share arc | An arc of a multi-arc system (139). The product divides the system's flow among them. |
| Single-arc system | A system of one arc (7). |
| Non-anchor arc | An arc the product does not close to a system (50): 39 with no anchor, 11 whose anchor is left out. |
| Share model | The small network that sets each share arc's fraction of its system's flow. |

## "Tier" has three unrelated meanings

- Tier 1 and tier 2 are the two checks of a learned-parameter run against CalSim3: tier 1 at 20
  training locations, tier 2 at every rim arc.
- In the USGS gauge table, `tier` is the flow regime of a gauge: 1 rain, 2 mixed, 3 snow. It
  says nothing about data quality.
- In the arc table, `tier` A–F grades how an arc's CalSim3 series was derived: A = its own
  gauge with a record in the training years that passes the naturalness screen, down to F = no
  record of its own in those years. X marks an arc that could not be graded (no series, not in
  the derivation table, or a record that was not screened). The 64 trained arcs are tier A.

## Models and runs

| Name | Meaning |
|---|---|
| GA parameters | The archived genetic-algorithm calibrations of Wi and Steinschneider (`artifacts/models/`, `ga_optimum.csv`). |
| Reference model | The NumPy/Numba implementation that reproduces the MATLAB simulations. Called the "frozen" model in the code. |
| dPL | Differentiable parameter learning: the network-trained parameters. |
| The ladder | The learned-parameter runs, one recipe with one change per rung, named for their rung: `1_hru`, `2_grid`, `3_pt`, `4_noah`, `5_px`, `6_deep`, `7_aef`, `8_calsim`. Each is trained from scratch except `8_calsim`, whose basis is trained from scratch and then fine-tuned ([Runs](runs.md#the-ladder)). |
| dPL-CalSim | The current learned model, the ladder's last rung (run folder `multifamily/8_calsim`): 159 entities of every family, 64 CalSim3 arcs among them, with WY1976–85 held out. |
| Basis, fine-tune | dPL-CalSim's two stages. The basis trains from scratch without the deep store (`checkpoints/basis.pt`); the fine-tune trains from it with the deep store, the cleaned targets, CDEC's monthly records and the dry-season term (`checkpoints/best.pt`). The product reads the fine-tune ([Training record](training_record.md)). |
| Deep store | A slow store behind SAC-SMA's supplemental lower zone, in the learned model only: it takes a learned share of that zone's drainage and returns it as baseflow over a learned residence time of 2 to 10 years ([Learned parameters](learned_parameters.md#how-it-is-built)). From rung `6_deep`. |
| `hybrid`, `hybrid_dt`, `lstm` | An LSTM on `5_px`'s daily flow; the same fine-tuned to keep `5_px`'s response to climate; an LSTM without it. |
| dPL-CalSim, BCM, VIC-CalSim3 | The three models of the [Benchmark](benchmark.md): the current learned model; the USGS Basin Characterization Model routed by its monthly post-processing, fitted per site on WY1991–2018; the VIC of the CalSim3 stochastic-input pipeline. |
| Product | The CalSim3 rim-inflow product: dPL-CalSim's monthly flow on the 196 rim arcs with the share model on the share arcs, WY1916–2018, one series per forcing (`artifacts/product/calsim3/`). The calibrated models deliver two more, named by their application: the CalLite files (`product/callite/`) and the daily flow of the 15 CDEC watersheds (`product/15cdec/`). |

## Other terms

| Word | Meaning |
|---|---|
| KGE | Kling–Gupta efficiency ([Model equations](equations.md#5-evaluation-metrics)). 1 is perfect. |
| Seasonal mismatch | The share of annual volume placed in the wrong month. |
| Footprint screening | Keeping only the modeling units inside the CalSim3 catchment for four watersheds whose calibrated outline over-reaches it ([Conventions](conventions.md)). |
| Holdout | Years whose observations are withheld from every fit (their forcing still runs). For dPL-CalSim: WY1976–85. Rungs 1 to 7 train on WY1989–2003 only. |
| Cycle spinup | The starting state used in scoring: the first ten water years looped 20 times from a cold start. |
| Local-only | On this machine and ignored by git: `tmp/` and `artifacts/_local/`. |
