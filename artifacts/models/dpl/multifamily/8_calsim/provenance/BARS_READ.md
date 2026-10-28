# Bars for the WY1976-85 read of 8_calsim and the two new rungs

**APPROVED by the user 2026-10-19 ("Yes, on those bars"): one read per run, 8_calsim after its product steps, 6_deep and 7_aef each once trained.**

Written 2026-10-19, before any WY1976-85 number of these runs is seen. The read happens only
after the user approves this page.

## Runs
- **The basis** (`checkpoints/basis.pt`, tracked): the reference. Its WY1976-85 numbers are its
  own read's (`provenance/read_WY1976-85_*.csv`, run `basis`); they are not re-read.
- **8_calsim** (was `7_calsim_ft45_deep_ref`; user 2026-10-19: adopt it as dPL-CalSim, rename):
  the basis fine-tuned 45 epochs (lr 3e-4) with the target cleaning, the monthly CDEC
  records back to WY1950, the dry-season term and the deep store (head at 10x the learning rate,
  shape-min-days 180), reference riparian ET. Passed all 13 bars of BARS_FINETUNE.md on the
  training years.
- **6_deep** and **7_aef** (training, `launch_rungs.py`): `5_px` and `6_aef` with the deep store,
  from scratch on the 15 CDEC watersheds, WY1989-2003. Read once each when trained.

## Before the read (training years, no held-out year; `score/compare_ref.txt`)
| | Basis | 8_calsim |
|---|---|---|
| Selection score (each on its own targets) | 0.8310 | 0.8427 |
| Sierra-4 peaks (top-5 median, points) | 76.8 | 77.6 |
| Sacramento-4 dry-third Aug-Sep bias (%) | -31.4 | -18.1 |
| SHA / ORO dry-third Aug-Sep bias (%) | -31.0 / -44.0 | -20.0 / -22.9 |
| Sacramento-4 Aug-Sep elasticity (observed 0.51) | 1.20 | 0.80 |
| Sacramento-4 monthly KGE | 0.939 | 0.954 |
| +3 degC annual / Apr-Jul (%) | -0.91 / -29.8 | -1.21 / -30.4 |
| CDEC daily / USGS / UF / arcs, paired median KGE against the basis | | +0.007 / +0.013 / -0.000 / +0.012 |

## What is read (once, CPU, `adopt/read_8calsim.py`)
The last adoption's six items, for 8_calsim:
1. the 13 outlets, monthly against CDEC FNF;
2. the 49 USGS gauges, daily;
3. the 9 UF subbasins, monthly;
4. the 64 CalSim3 arcs, monthly;
5. the 196 product arcs on WY1976-85;
6. the 196 product arcs on WY1922-49.

With them: the share model's response gate (8_calsim's own product), and the footprint split
(inside / partial / outside the 15cdec_grid cells) of every regional item.

Reported, no decision: the Aug-Sep volume bias (the dry bars' months) at the 13 outlets against
CDEC monthly FNF, both runs, over WY1976-85 and in the WY1976-77 drought.

The rungs, when trained: the 13 outlets monthly and the regional records (USGS, UF, arcs), as for
the other rungs; no decision rests on them, they fill the ladder's tables.

## The measure
Mean KGE difference over the shared items, with the paired median and the count up beside it.
Better: +0.01 or more; worse: -0.01 or less; no change in between (the last two reads' bar).

## Proposed decision
8_calsim is adopted (user, 2026-10-19). If it is "worse" on none of the six items and the
response gate passes, the adoption chain runs (scores, tiers, product, atlas, benchmark,
studies). If it is "worse" on any item or the gate fails, nothing else runs: Claude reports the
item to the user, who decides.

## Outcome (2026-10-19 09:40): 8_calsim HOLDS

Worse on no item; the share model's response gate passes (`adopt/read/read_WY1976-85.log`,
tables `adopt/read/read_WY1976-85_*.csv`). The training-year pass of the same script
(`--dry`, WY1989-2003) labelled every item "no change".

| Item (WY1976-85) | Basis | 8_calsim | Mean diff (median) [up/n] | Label |
|---|---|---|---|---|
| 13 outlets, monthly | 0.901 | 0.913 | +0.012 (+0.010) [10/13] | better |
| 49 USGS, daily | 0.640 | 0.639 | -0.001 (+0.003) [27/49] | no change |
| 9 UF, monthly | 0.875 | 0.878 | +0.003 (+0.000) [5/9] | no change |
| 64 arcs, monthly | 0.705 | 0.705 | +0.000 (+0.003) [34/64] | no change |
| Product, 196 arcs, WY1976-85 | 0.734 | 0.731 | -0.003 (+0.002) [104/196] | no change |
| Product, 196 arcs, WY1922-49 | 0.661 | 0.659 | -0.002 (+0.003) [105/196] | no change |

Reported, Aug-Sep volume bias against CDEC monthly FNF (%):

| | Basis | 8_calsim |
|---|---|---|
| WY1976-85, median over the 13 outlets | -11.1 | -8.1 |
| WY1976-85, SHA / ORO / FOL | -17.9 / -17.5 / +31.6 | -12.4 / -8.1 / +19.2 |
| WY1976-77, median over the 13 outlets | -5.0 | +7.0 |
| WY1976-77, SHA / ORO / FOL | -56.8 / -71.3 / +22.4 | -33.8 / -44.4 / +60.1 |
| WY1976-77, the four Sacramento outlets (sum of depths) | -55.9 | -32.2 |

In the drought the store lifts SHA's and ORO's late summer by about a third of the shortfall;
FOL, already high, goes higher (+60 %), and the mean over the outlets rises from +6.5 to +37.4.

## The rungs' read (2026-10-19, once each, `adopt/read_rungs.py`)

Mean KGE, WY1976-85; the change paired record by record, mean (median), records up:

| Step | Records | Before | After | Change | Up |
|---|---|---|---|---|---|
| 5_px -> 6_deep | outlets | 0.859 | 0.858 | -0.002 (-0.005) | 3/13 |
| 5_px -> 6_deep | usgs | 0.522 | 0.509 | -0.013 (-0.002) | 24/49 |
| 5_px -> 6_deep | uf | 0.803 | 0.805 | +0.002 (+0.005) | 5/9 |
| 5_px -> 6_deep | arcs | 0.630 | 0.631 | +0.001 (+0.004) | 35/64 |
| 6_deep -> 7_aef | outlets | 0.858 | 0.853 | -0.005 (-0.012) | 6/13 |
| 6_deep -> 7_aef | usgs | 0.509 | 0.515 | +0.006 (+0.007) | 26/49 |
| 6_deep -> 7_aef | uf | 0.805 | 0.844 | +0.039 (+0.049) | 6/9 |
| 6_deep -> 7_aef | arcs | 0.631 | 0.659 | +0.028 (+0.045) | 40/64 |

## Names
The bars above use the names of their day, except the basis. 8_calsim was trained as
`7_calsim_ft45_deep_ref` ("F" in `BARS_FINETUNE.md`, which holds the 13 items it passed on the
training years, scored in `compare_finetune.txt`); on adoption the user named the rungs `6_deep`,
`7_aef` and `8_calsim`. The reference is 8_calsim's basis, the network its fine-tune started from
("D" in `BARS_FINETUNE.md`); its rows in these tables are named `basis`. The tables of 0_ga to
5_px and of the LSTM models, `read_footprint.csv`, `read_monthly_ceiling.csv`,
`readout_2004_2018_retrained.csv`, `levels_ref.json` (the loss references the basis trained with)
and `read_WY1976-85_statics.csv` (rungs 2 to 6 on both grid statics; `6_aef` there is the rung of
its day, which `7_aef` replaces) are carried over from the basis's provenance.
`levels_finetune.json` holds the fine-tune's loss references and dry-season weight.

