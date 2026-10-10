# Bars for the WY1976-85 read (APPROVED 2026-10-10 before 07:54)

Written before any WY1976-85 number of a ladder run or of 7_calsim has been seen. Drafted and approved before the read (the read started 07:54:40 by the clock; the ~stamps written this morning ran ahead of it), with one change: "go ahead. but make it mean kge instead" (the measure is the mean KGE difference, not the median). This is the decade's sixth read.

Inputs computed for this page from observations only (no model output):
- `checks/read_footprint.csv`: each entity's share of footprint area in the 15cdec_grid cells.
- `checks/read_monthly_ceiling.csv`: the daily CDEC gauge summed by month against CDEC monthly FNF.

## What is read

One script, `checks/read_1976_1985.py`, on the CPU, run once. Before that, the same script runs on WY1989-2003, which is training years for every run, to check that it runs and that the counts are right. Each model uses its best.pt: 7_calsim's best.pt is its final epoch, and 0_ga uses its saved flows. The hybrids are the retrained ones, as the mean flow of three seeds. dPL-CalSim means the tracked run (`..._w2ft15r10_aef`) and serves as the reference.

1. **Outlets, monthly (decision 16).**
   - What: CDEC monthly full natural flow at the 13 outlets that have their own monthly record (all except NHG and NML), 120 months.
   - How: `outlet_scores(monthly=True)` on each run's cold-start outlet flows, each run on its own domain.
   - Rows: 0_ga, 1_hru, 2_grid ... 6_aef, 7_calsim, hybrid, hybrid_dt, lstm, dPL-CalSim.
2. **Region (decision 3).**
   - What: on 7_calsim's 159 entities:
     - the 49 USGS gauges with at least 90 days in the decade (daily);
     - the 9 unimpaired-flow subbasins (monthly);
     - the 64 CalSim3 arcs (monthly; calsim3 basis, own-record basis reported).
   - How: `score_wy`.
   - Rows: 2_grid ... 6_aef, 7_calsim, and dPL-CalSim from its tracked `metrics_holdout.csv`. Rungs 0 and 1 and the hybrids have no regional row.
   - Check: 7_calsim's `score_wy` must equal its `dpl evaluate --score-holdout` (`metrics_holdout.csv`).
   - Split by footprint share in the 15cdec_grid cells: inside >= 0.9, outside <= 0.1, partial in between.

     | Family | Inside | Partial | Outside |
     |---|---|---|---|
     | USGS | 35 | 2 | 12 |
     | UF | 1 | 5 | 3 |
     | Arcs | 46 | 12 | 6 |

   - For rungs 2-6 every regional item is also a spatial transfer: they trained on the 15 outlets only.
3. **Product, 7_calsim only.**
   - Run `product apply --score-holdout` on `historical_livneh_unsplit` into `7_calsim/product`.
   - Score the 196 product arcs over WY1976-85 and over WY1922-49.
   - Check the share model's response gate (`response_gate.csv`).
   - Compare with the tracked product's `product_metrics.csv`.

## The measure

Each comparison takes the mean KGE difference over the same items (the items both rows score), and counts the items that go up.
- **Better:** the mean difference is +0.01 or more.
- **Worse:** the mean difference is -0.01 or less.
- **No change:** anything in between.

0.01 is the bar of the fourth and fifth reads, which used the median; this read uses the mean (the user). The three hybrid seeds spread over 0.013 on WY2004-18, so a smaller step is not a finding. A mean follows single items: one outlet or gauge with a very low KGE can carry a label (0_ga -> 1_hru below is SCC's). The paired median is printed beside every mean, with no bar.

## The bars

**B1. 7_calsim against dPL-CalSim.** These are the reversal bars of the fourth and fifth reads, with the mean in place of the median, plus the outlets. 7_calsim holds if it is "worse" on none of these:
- USGS (49);
- UF (9);
- arcs (64, calsim3 basis);
- the 13 outlets (monthly);
- the product arcs over WY1976-85 (196);
- the product arcs over WY1922-49 (196).

The response gate must also pass. Under decision 40, 7_calsim replaces dPL-CalSim whatever this shows. A "worse" goes into runs.md as a known cost.

**B2. The ladder steps.** Each step gets a label on the 13 outlets. Steps from rung 2 on also get one per regional family, with inside and outside kept apart. A step is **confirmed** when its outlet label on WY1976-85 matches its label on WY2004-18. The WY2004-18 labels below are already read (15 outlets, daily, `readout_2004_2018_retrained.csv`):

| Step | WY2004-18 mean difference (up at) | WY2004-18 label |
|---|---|---|
| 0_ga -> 1_hru | +0.0706 (7/15) | better (SCC: GA -0.18) |
| 1_hru -> 2_grid | -0.0126 (5/15) | worse |
| 2_grid -> 3_pt | +0.0172 (11/15) | better |
| 3_pt -> 4_noah | +0.0008 (8/15) | no change |
| 4_noah -> 5_px | -0.0033 (7/15) | no change |
| 5_px -> 6_aef | -0.0035 (9/15) | no change |
| 6_aef -> 7_calsim | (none: 7_calsim trains on WY2004-18) | no confirmation |

**B3. aef's transfer case (decision 28).** Compare 6_aef with 5_px over the 40 regional items outside or partial: USGS 14, UF 8, arcs 18. The result is reported as better, worse or no change. aef stays either way.

**B4. The hybrids.** On the 13 outlets (monthly):
- hybrid against 5_px: WY2004-18 better (+0.0301, 10/15);
- hybrid_dt against hybrid: WY2004-18 worse (-0.0241, 3/15);
- lstm against 5_px: WY2004-18 no change (-0.0023, 7/15).

Each is confirmed when its WY1976-85 label matches the WY2004-18 one.

**B5. The focus watersheds.** Added 2026-10-10 before 07:54, before the read, at the user's word: "our key focus is generally trinity, shasta, oroville, yuba, folsom, tuolumne, merced, stanislaus, and upper san joaquin". Same rule (mean difference, 0.01).
- **Outlets, monthly FNF:**
  - Seven come from the 13-outlet table: SHA, ORO, YRS, FOL, TLG, MRC and MIL (Upper San Joaquin).
  - Stanislaus has no outlet row (the user, before 07:54: "Arcs only, no outlet row"). Its training target, NML's daily FNF, starts in WY1987, and CDEC has no monthly NML series. Goodwin's monthly FNF (SNS) was looked at and set aside: against the NML target over WY1988-2018 it scores KGE 0.77 (about 10% more volume, about 25% more in April-May, 2-4 times less in August-October). Stanislaus is read through its product arcs: ST is exactly the 19 arcs of NML's footprint.
  - Trinity: the cdec_CLE entity's flow against TNL, Lewiston's monthly FNF, in depth over CLE's 692.86 mi2. CLE's monthly volume agrees with TNL (data/targets/cdec/README.md). Rows for 2_grid ... 6_aef, 7_calsim and dPL-CalSim (the user, before 07:54: "can't we do the same out of sample test of trinity/cle for the rungs 0-6?"): each run's field on the CLE entity, as score_wy runs it (its own training statistics, cycle spinup). For rungs 2-6 Trinity is a transfer in space and time (they trained on the 15 watersheds only); for 7_calsim and dPL-CalSim in time only (both trained on CLE's daily record from WY1986). Rungs 0 and 1 have no Trinity row: the GA has parameters for the 15 watersheds' HRUs only (the CalLite GA that covers Trinity was calibrated on Trinity's own record, a different model), and 1_hru's network does not run on the cells (decision 16).
- **Product:** the mean product KGE of each system's arcs, against CalSim3: TRIN (Trinity), SHAS, OROV, YUBA, FOLS, TU (Tuolumne), ME (Merced), ST (Stanislaus) and SJ (Upper San Joaquin). Each system is one item, so Folsom's 46 arcs count once.
- **Labels:**
  - 7_calsim against dPL-CalSim: on the 8 focus outlets (the seven and Trinity); on the 9 systems over WY1976-85; and over WY1922-49.
  - Every ladder step and the hybrids on the 7 focus outlets the rungs share (all but Trinity), each confirmed against the same 7 on WY2004-18.
  - Trinity's values for rungs 2-6 are printed with the steps, with no label (one item).
- Per-watershed values are printed for every row.

## Reported, no bar

**The monthly ceiling at each outlet.** This is the daily gauge summed by month against monthly FNF, WY1989-2018, complete months only.
- Median 0.996.
- Lowest: YRS 0.894, MKM 0.960 and MRC 0.964. All others are 0.985 or more.
- At YRS, the two records disagree by about 0.1 in KGE, so differences smaller than that are inside the disagreement.
- The daily record starts in WY1987, so the decade's own disagreement is unknown.

**Unpaired medians** for every row, family and split, with KGE r, alpha and beta at the outlets.

**Shasta and Bend Bridge:** Shasta's arc (I_SHSTA) and the Bend Bridge system, as in the earlier reads.

## After the read

- **Where the numbers go:** PLAN.md, and the WY1976-85 column of the runs.md ladder section (decision 35). The docs name the sixth read.
- **What stays fixed:** nothing retrains, reselects or changes on account of the read.

## Defaults (approved with the page)

1. The hybrids are in the outlet read (B4).
2. The band is 0.01.
3. "Confirmed" uses the outlets only. Confirming the regional steps would need a regional WY2004-18 read of rungs 2-6. That read was never made: WY2004-18 is a test window for those rungs, but only the outlets were read. I left it out.

## Result (the one read, 2026-10-10 07:54-08:00 by the clock; log logs/read_1976_1985.log, tables checks/read_WY1976-85_*.csv)

Checks: score_wy = each run's archived sim_daily to 1e-8 KGE; dPL-CalSim's score_wy = its tracked metrics_holdout.csv to 1e-16; its product scores = its tracked product_metrics.csv to 2e-8. Counts as written: 13 outlets, 49 USGS, 9 UF, 64 arcs, 196 product arcs, 40 items for B3.

**B1, 7_calsim against dPL-CalSim: worse on one item (UF), a known cost under decision 40.**
| Item | Mean difference | Median | Up | Label |
|---|---|---|---|---|
| USGS (49) | +0.011 | -0.015 | 22 | better (carried by the 12 outside gauges: mean 0.386 vs 0.274) |
| UF (9) | -0.020 | -0.021 | 0 | **worse** |
| Arcs (64) | -0.008 | 0.000 | 32 | no change |
| Outlets (13) | -0.002 | -0.003 | 6 | no change |
| Product arcs WY1976-85 (196) | +0.012 | +0.004 | 107 | better |
| Product arcs WY1922-49 (196) | +0.006 | +0.006 | 109 | no change |
| Response gate | pass (both) | | | |

**B2, the ladder at the 13 outlets (monthly): every step confirmed.** Mean KGE: 0_ga .769, 1_hru .855, 2_grid .836, 3_pt .852, 4_noah .850, 5_px .859, 6_aef .866, 7_calsim .899 (dPL-CalSim .900). Steps: 0->1 better (+0.085), 1->2 worse (-0.019), 2->3 better (+0.017), 3->4 / 4->5 / 5->6 no change, each as on WY2004-18; 6_aef -> 7_calsim better (+0.032). Region: 5_px -> 6_aef better in every family (USGS +0.180, UF +0.154, arcs +0.067); 6_aef -> 7_calsim better in every family (+0.106, +0.011, +0.048); 3_pt -> 4_noah worse on USGS (-0.047) and UF (-0.043); 2_grid -> 3_pt better on USGS (+0.023), worse on UF (-0.030) and arcs (-0.022); 4_noah -> 5_px better on UF (+0.048) and arcs (+0.011).

**B3, aef's transfer: better.** 6_aef - 5_px over the 40 items outside or partial: mean +0.289, median +0.121, 27 of 40 up.

**B4, the hybrids at the 13 outlets:** hybrid vs 5_px better (+0.016), confirmed; hybrid_dt vs hybrid worse (-0.013), confirmed; lstm vs 5_px worse (-0.085; WY2004-18 no change), not confirmed. Means: hybrid .876, hybrid_dt .862, lstm .774.

**B5, the focus watersheds.** 7_calsim vs dPL-CalSim: the 8 focus outlets -0.008 (no change, 4/8 up; Trinity .902 vs .929); the 9 systems' product +0.009 on WY1976-85 (no change, 7/9 up) and +0.008 on WY1922-49 (no change, 5/9). Ladder on the 7 focus outlets: confirmed except 2_grid -> 3_pt (worse -0.014; WY2004-18 better); 5_px -> 6_aef better (+0.018), 6_aef -> 7_calsim better (+0.010). Hybrids on the 7: hybrid vs 5_px worse (-0.015; WY2004-18 better; MRC .801 vs .937, YRS .855 vs .885, MIL .849 vs .862), hybrid_dt vs hybrid no change, lstm worse (-0.103). Trinity (CLE vs TNL): 2_grid .700, 3_pt .628, 4_noah .555, 5_px .664, 6_aef .817, 7_calsim .902, dPL-CalSim .929.

Reported: YRS scores above its WY1989-2018 ceiling (.894) in this decade (1_hru .949): the ceiling of the decade itself is unknown, as the page said.
