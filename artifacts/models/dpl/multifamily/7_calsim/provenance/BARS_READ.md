# Bars for the WY1976-85 read of revised B (approved by the user 2026-10-14)

Written 2026-10-12 for B, revised 2026-10-14 for B2, before any WY1976-85 number of B or B2 is
seen. The read happens only after the user approves this page.

## Runs
- **dPL-CalSim** (`multifamily/7_calsim`, tracked): the reference. Its WY1976-85 numbers are
  already read (`provenance/read_WY1976-85_*.csv`, `metrics_holdout.csv`); they are not re-read
  on the old statics.
- **B2** (`7_calsim_fpb2`, revised B, user 2026-10-13): dPL-CalSim's config on the footprint
  statics with MBASE -2..5, NMF 0.05..0.5, Velo 0.5..10, zperc 1..1000, lzsk 0.001..0.5;
  PLWHC, UADJ and MFMIN at their default ranges; trained with the ET backward cap of
  `sacsma/dpl/physics/et_noah.py` (forward unchanged, so the engine scores it as is).
- **B** (`7_calsim_fpb`) is not read: it is kept as a data point (no better than dPL-CalSim on
  the training years, snow parameters outside the published ranges, a gradient blowup at e63).

## Before the read (no held-out year), done 2026-10-14
| | dPL-CalSim | B2 |
|---|---|---|
| Selection score (best, e120) | 0.8300 | 0.8310 |
| KGE, 15 outlets, WY1989-2003 / WY2004-18 | .859 / .876 | .864 / .875 |
| Apr-Jul share (obs .499) | .471 | .472 |
| +3 degC, annual / Apr-Jul, mean % over the 15 outlets | -1.22 / -30.4 | -0.91 / -29.8 |
| P -10 % / +10 %, annual | -17.6 / +18.3 | -17.3 / +18.1 |
| Benchmark, 12 CDEC sites, WGEN s1, WY1991-2018: median / mean KGE | .925 / .919 | .927 / .919 |
| Peak SWE, regional mean, today / +3 degC (mm) | 192 / 81 | 191 / 84 |

B2's field against the published Snow-17 ranges: MBASE < 0 in every cell (median -1.79, 17 % at
the -2 bound); NMF > 0.30 (Anderson's maximum) in every cell, none above 0.5 (median 0.47, 12 % at
the bound); PLWHC, UADJ, MFMIN inside their published ranges; DAYGM median 0.018 mm/day
(dPL-CalSim 0.24; 32 % of cells below 0.01, inside the published 0-0.3).

## What is read (once, CPU, one script)
Same items as the sixth read: 13 outlets monthly; 49 USGS gauges daily; 9 UF subbasins monthly;
64 CalSim3 arcs monthly; the 196 product arcs over WY1976-85 and WY1922-49 (product built from
the run with its own share-model fit), and the share model's response gate. Footprint split
(inside / partial / outside the 15cdec_grid cells) reported for every regional item.

In the same read, proposed: the ladder's regional records (49 USGS, 9 UF, 64 arcs) for rungs
2_grid to 6_aef on the footprint statics, their weights unchanged. No decision rests on it; it
replaces the regional table in runs.md and the ladder's regional figure, which were read on the
point-sampled statics (on the non-held-out years the physical rungs' out-of-footprint USGS
volume bias falls from +49-60 % to +12-21 %).

## The measure
Mean KGE difference over the shared items, with the count of items up and the paired median
beside it. Better: +0.01 or more; worse: -0.01 or less; no change in between (the sixth read's
bar).

## Proposed decisions
B2 carries a data fix (point-sampled statics on 24 % of the cells) and a model change (the
ranges), and the read cannot tell them apart.
1. B2 replaces dPL-CalSim if it is "worse" on none of the six items (13 outlets, USGS, UF, arcs,
   product WY1976-85, product WY1922-49) and the share model's response gate passes. MBASE below
   zero and NMF above 0.30 are then recorded in runs.md as known departures from the published
   ranges.
2. If B2 is "worse" on any item, the statics fix still has to be adopted: the statics-only run
   (A, dPL-CalSim's command unchanged) is trained then, and the ranges are judged as B2 against
   A on the bar above, without a second read of dPL-CalSim.

## Result (read 2026-10-14, once, CPU)
Mean KGE difference of B2 against dPL-CalSim (`read_WY1976-85_labels.csv`), n items, items up:

- usgs 7_calsim vs 7_calsim_point: +0.0024 (n 49, 28 up, median +0.0046): no change
- uf 7_calsim vs 7_calsim_point: +0.0005 (n 9, 4 up, median -0.0016): no change
- arcs 7_calsim vs 7_calsim_point: -0.0011 (n 64, 28 up, median -0.0013): no change
- outlets 7_calsim vs 7_calsim_point: +0.0026 (n 13, 8 up, median +0.0038): no change
- product kge_product_holdout 7_calsim vs 7_calsim_point: -0.0011 (n 196, 94 up, median -0.0008): no change
- product kge_product_early 7_calsim vs 7_calsim_point: -0.0020 (n 196, 88 up, median -0.0015): no change
- The share model's response gate: passed.

Decision 1 holds: B2 replaces dPL-CalSim. MBASE below zero and NMF above 0.30 are recorded in
runs.md as departures from the published Snow-17 ranges.

## Names
The bars above use the names of their day. B2 was trained as `7_calsim_fpb2`; on adoption the
user named it `7_calsim` (2026-10-14: "it is the final corrected 7th rung"), and the run it
replaces, dPL-CalSim of 2026-10-10, was named `7_calsim_point`. The tables in this
folder use the adopted names: `7_calsim` is B2, `7_calsim_point` the replaced run. Rungs 2_grid to
6_aef's regional records in `read_WY1976-85_region.csv` are on the footprint statics;
`read_WY1976-85_statics.csv` holds them on both statics. The 13 outlets of 0_ga to 6_aef and of the
LSTM models, and `readout_2004_2018_retrained.csv`, `read_footprint.csv`,
`read_monthly_ceiling.csv` and `levels_ref.json`, are carried over from the replaced run's
provenance (on the footprint statics the rungs' outlet means move by 0.0003 or less,
measured on WY1989-2018 before the read; the loss references are the ones B2 trained with).
