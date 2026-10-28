# Training record

What dPL-CalSim (`8_calsim`) trained on, stage by stage: which records are entities, what each
stage trained on and what it left out with the reason, how the inputs and the held-out years
enter, how the families are weighted, and what the rim-inflow product reads. The counts come
from the checkpoints through the trainer's own loader:

    sacsma dpl record 8_calsim     # -> artifacts/results/dpl/multifamily/8_calsim/training_record/

The four tables ([`stages.csv`](../artifacts/results/dpl/multifamily/8_calsim/training_record/stages.csv),
[`families.csv`](../artifacts/results/dpl/multifamily/8_calsim/training_record/families.csv),
[`entities.csv`](../artifacts/results/dpl/multifamily/8_calsim/training_record/entities.csv),
[`exclusions.csv`](../artifacts/results/dpl/multifamily/8_calsim/training_record/exclusions.csv))
hold every number on this page that is not marked otherwise. The command stops if a mask table
no longer holds what a stage masked, so the reasons are those the stage was trained with.
`sacsma verify record` writes the record again and compares it with the tracked tables.

## The two stages

| | Stage 1: basis | Stage 2: fine-tune |
|---|---|---|
| Checkpoint | `checkpoints/basis.pt` | `checkpoints/best.pt` |
| Starts from | scratch | the basis's network |
| Epochs, learning rate | 120, 1e-3 | 45, 3e-4 (the deep store's head at 10 times) |
| Physics | `7_aef`'s, without the deep store | the same, with the deep store |
| Targets | the records as stored | the cleaned records: the CDEC screens and the location masks |
| CDEC monthly full natural flow | no | yes, from WY1950 |
| Dry-season term | no | yes, λ = 0.1132 |
| Epoch kept, selection score | 120, 0.8310 | 45, 0.8427 |

Both stages train the same 159 entities and hold out the same years (WY1976–85). The product
reads only stage 2.

A checkpoint's configuration is not a verbatim launch record. The basis predates the
fine-tune's settings, which it reads as off. The fine-tune's was edited after training: the
retired `riparian` key was removed, `init_from` and the `target_series` path were rewritten, and
it records patience 45 for a run that began with 20 and was resumed (`docs/reproduce.md`,
`provenance/BARS_FINETUNE.md`).

## Which records are entities

The registry (`data/inputs/domains/multifamily/entities.csv`) holds 291 entities; dPL-CalSim
takes 159 of them.

| Family | Candidates | Entities | Left out, and why |
|---|---|---|---|
| USGS gauges, daily | the gauges of the neuralhyd-ca store | 69 | Gauges with less than 90 % of their delineated watershed inside the CalSim3 domain and Goose Lake; the store's 14 synthetic ids, which are CDEC watersheds already here ([USGS](../data/targets/usgs/README.md)) |
| CDEC full natural flow, daily | | 17 | The 15 CDEC watersheds, CLE and CSN. The candidate stations WHI and SNS do not train |
| DWR unimpaired-flow subbasins, monthly | 24 subbasins | 9 | 6 have no arcs, since no authoritative assignment to arcs exists (UF 1, 5, 12, 17, 23, 24: the valley floors, three minor-stream groups, the Tulare Lake Basin outflow). 9 are monthly copies of a long CDEC daily record (UF 8, 9, 11, 14, 15, 16, 18, 19, 22) ([DWR](../data/targets/dwr_unimpaired/README.md)) |
| CalSim3 arcs, monthly | 203 arcs in `arc_hierarchy.csv` | 64 | 7 are not rim arcs of the registry (3 without a polygon, 3 below the rim, the series-less valley node `I_SRBB_VAL`). Of the 196, only tier A trains (64 arcs, 25,523 own-gauge months). 31 more have an own-gauge record (16,487 months) and do not train: tier B fails the naturalness screen (7), C is a difference of gauges (10), D a reservoir mass balance (8 of the 14), X otherwise (6 of 7). Tiers E (72, donor split or proportioned) and F (22, no own record in the training years) have none ([CalSim3](../data/targets/calsim3/README.md)) |

The 7 arcs that are single-arc copies of an entity (`I_SHSTA`, `I_TRNTY`, `I_MLRTN`, `I_MCLRE`,
`I_NHGAN`, `I_PTH070`, `I_ESTMN`) are tiers D, F and X, so none trains twice as an arc.

## What each stage trains on

Units trained per family (`families.csv`), with the July–September units among them:

| Family | Stage 1 | Stage 2 |
|---|---|---|
| USGS gauges | 69 of 69, 666,694 days (Jul–Sep 169,569) | 60 of 69, 573,630 days (Jul–Sep 129,722) |
| CDEC, daily | 17 of 17, 176,898 days (Jul–Sep 42,439) | 17 of 17, 132,040 days (Jul–Sep 16,521) |
| CDEC, monthly | none | 15 of 17, 9,421 months (Jul–Sep 2,279) |
| DWR subbasins | 9 of 9, 5,940 months (Jul–Sep 1,485) | 5 of 9, 2,379 months (Jul–Sep 108) |
| CalSim3 arcs | 64 of 64, 25,523 months (Jul–Sep 6,382) | 60 of 64, 18,345 months (Jul–Sep 2,210) |

From a record to the units trained, per entity (`entities.csv`, in the loader's order):

- **The registry window.** USGS gauges their record within 1950–2018; CDEC sites from 1986 to
  1988 on (CSN and BND from 1999) to 2018; DWR subbasins WY1985–2014; arcs the arc's own-gauge
  months inside WY1950–75 and WY1986–2015 (`arc_obs_mask.csv`).
- **Replaced or removed days** (stage 2): `usgs_11218500` trains on the Kings River record with the
  powerhouse flow added (`usgs_11218501_mm.csv`; 13 days it lacks are removed). Five hand-checked
  CDEC days are removed in both stages (`obs_mask`).
- **The held-out WY1976–85**, every family: 154,718 USGS days and 108 subbasin months. The CDEC
  daily records and the arcs' own-gauge months hold none.
- **The subbasins' extension** back to WY1950, both stages: 312 months each (WY1950–75; WY1976–84
  are held out), 2,808 in all.
- **The screens and masks** (stage 2 only), below.
- **Depth fixes** (stage 2): `I_MFM008`'s target × 1.387 (a 68.4 mi² gauge on a 94.9 mi² arc) and
  `I_SFM005`'s × 0.855 (its gauge's area).

The CDEC monthly record (stage 2) comes from WY1950 at 15 sites: CLE through TNL (Trinity River at
Lewiston), the others at the station itself. The forcing-era masks start it at WY1986 at CLE, ISB
and SCC and take WY1962–75 out at MIL. NHG has no monthly station, and NML's (SNS, below New
Melones and Tulloch) is not the same point, so neither has one.

## What the fine-tune leaves out, and why

Units removed in stage 2 (`exclusions.csv`). A unit two rows name is counted under the stronger
reason, in the order whole record, train once, forcing era, years, the rest.

| Family | Reason | Units | Entities |
|---|---|---|---|
| CDEC, daily | `S`: the site's unreliable season, every year (July–November at CLE, FOL, MKM, YRS, TLG, NHG, NML, CSN; July–October at MRC, ORO) | 38,966 days | 10 |
| | `D2`: a month whose daily sum is more than 25 % off a trusted monthly value | 3,829 | 13 |
| | `D3`, `D1`, `F`: negative-day months, the partners of negative days, near-zero days in wet months | 2,063 | 14 |
| CDEC, monthly | forcing eras: WY1950–75 at CLE, ISB, SCC and WY1962–75 at MIL, where the Livneh precipitation drifts from PRISM's | 1,104 months | 4 |
| | `M1z`, `M1r`, `D2m`: zero or negative months, implausibly low months, months the review found wrong | 140 | 11 |
| USGS gauges | whole record (below) | 49,964 days | 9 |
| | July–September of impaired summers | 20,208 | 9 |
| | forcing eras (WY1950–75 at 11230500; WY1962–75 at 11199500 and 11237500) | 16,220 | 3 |
| | July–November at 11218500 (its replacement record is regulated then) | 3,305 | 1 |
| | years impaired (11433300 after 1962, 11433500 after 1961, 11341400 in WY1969) | 3,106 | 3 |
| | October at 11341400 after Box Canyon Dam | 248 | 1 |
| DWR subbasins | whole record (below) | 2,640 months | 4 |
| | July–September of impaired summers (UF 2 from 1957; UF 4, 10 every year; UF 7, 20 from WY1986) | 648 | 5 |
| | train once: UF 20's months that `usgs_11259000` already trains | 273 | 1 |
| CalSim3 arcs | train once: the months a USGS gauge at the same place already trains | 3,269 months | 8 |
| | July–September of impaired summers | 3,165 | 36 |
| | whole record (below) | 504 | 2 |
| | years: `I_SFY048` WY1995–2014, which are not its gauge's record | 240 | 1 |

The rules are those of the tables:
[the CDEC screens](../data/targets/cdec/README.md#the-training-screens-of-dpl-calsim) and
[the location masks](../data/targets/usgs/README.md#the-location-mask).

**The 17 entities stage 2 does not train**, with the audit's reason (October 2026; the audit's
working files are not tracked):

| Entity | Reason |
|---|---|
| `uf_03` Cache Creek | The published series is the outflow below Clear Lake and Indian Valley; its arcs are the lake inflows |
| `uf_06` Sacramento River near Red Bluff, `uf_13` Cosumnes | Copies of the CDEC monthly records of BND (SBB) and CSN, which train in stage 2 |
| `uf_21` Fresno River | The Big Creek ditch import, 8 to 10 % of the annual flow |
| `usgs_11257500`, `usgs_11258000` Fresno River | The same import, 19 to 24 % of the annual flow in WY1970 |
| `usgs_11418000` Yuba below Englebright | 0.63 to 0.85 of the Yuba's full natural flow: regulated |
| `usgs_11408850` Middle Yuba near Camptonville | The Milton-Bowman export, 15 to 41 % |
| `usgs_11413320`, `usgs_11413323` Deadwood Creek, Owl Gulch | Water diverted at the gauge to a powerhouse |
| `usgs_11396400` Sucker Run | Annual 0.46 times its neighbour's and summers 1.7 times: not natural in any season |
| `usgs_11208500` | Naturalness unverified, 1.9 mi² |
| `usgs_11525500` Trinity at Lewiston | The same record as TNL, CLE's monthly record |
| `cs_I_SLYCK`, `cs_I_WBR001` | Naturalness unverified |
| `cs_I_CFS001`, `cs_I_NMA003` | Train once: `usgs_11292500` and `usgs_11433260` train the same record |

An entity with no target keeps its place in its family's weight (below).

The Fresno River system therefore trains on nothing in stage 2: its subbasin and both gauges are
masked and none of its arcs is tier A. In stage 1 it trained on all three.

## Inputs, spin-up and the held-out years

- **Forcing**: `historical_livneh_unsplit` (daily precipitation, minimum and maximum
  temperature) in both stages.
- **The network sees AlphaEarth only**: the 16 leading components of each cell's 2017–2025 mean
  embedding and its length. No forcing, climate index, elevation or position enters it, so a
  cell has one parameter set whichever entity holds it. Latitude, elevation, flow length, the
  observed green fraction and the MODIS LAI climatology enter the physics directly.
- **PET** is Priestley–Taylor in both stages (`et_mode noah`; the configuration's `sac_pet` is
  not read).
- **Spin-up**: each epoch starts cold on 1939-10-01 and runs WY1940–49 once without gradient.
  The configuration's `spinup_start` (1978-10-01) falls inside the training period, so the
  trainer replaces it with ten years before WY1950. The forcing before October 1939 is not used.
- **The held-out WY1976–85 are held out as targets, not as forcing.** No target of those years is
  read in training or selection. Their forcing runs in every epoch, forward only, and carries the
  model's state into WY1986; WY1985 is the gradient-carrying burn-in of the WY1986 water year (each
  live water year runs from the start of the year before).

## How the families are weighted

The loss shares are fixed per family: usgs 0.2366, cdec 0.5508, uf 0.1226, calsim 0.0900 in stage
1; stage 2 splits the CDEC share evenly between the daily and the monthly records. Each family's
loss is divided by a fixed scale measured before training (`mt_loss_ref`), and each entity takes
its family's share divided by the family's entity count, scored or not.

What that gives, as the share of the coefficients summed over an epoch's water years (the
trainer's chunk log):

| | USGS | CDEC daily | CDEC monthly | DWR subbasins | CalSim3 arcs |
|---|---|---|---|---|---|
| Stage 1, before the scales | 0.197 | 0.507 | | 0.205 | 0.091 |
| Stage 1, with the scales | 0.132 | 0.459 | | 0.326 | 0.082 |
| Stage 2, before the scales | 0.166 | 0.236 | 0.426 | 0.099 | 0.073 |
| Stage 2, with the scales | 0.092 | 0.191 | 0.534 | 0.129 | 0.055 |

In stage 1 the shares before the scales are the footprint-area shares of the families exactly:
that is what "weighted by footprint area" means. Stage 2 kept stage 1's shares and changed the
targets, so its weights no longer follow the areas; the CDEC monthly record, which spans WY1950–2018
at 15 outlets, carries the most.

**The dry-season term** (stage 2) scores the July–September volume of each entity and water
year from one record: the CDEC monthly where its three months are present, else the monthly
target, else the daily record with at least 80 % of the days. Its weight per entity is the
family's share over its entity count (CDEC daily and monthly together), without the scales; a
record with no July–September flow in more than a tenth of its years is left out.

**Selection** keeps the epoch with the best training-year score: the mean over each family's
entities of the per-entity KGE (daily, or monthly sums), the four family means weighted by the
families' footprint areas (usgs 0.197, cdec 0.507, uf 0.205, calsim 0.091). The CDEC monthly
record is not in it. Both stages kept their last epoch.

## What the product reads

The rim-inflow product ([CalSim3 rim inflows](calsim3_rim_inflows.md)) uses stage 2's network
(`checkpoints/best.pt`) only.

- **Arcs.** The 196 rim arcs: 139 share arcs in 11 systems, 7 single-arc systems and 50
  non-anchor arcs. The valley node `I_SRBB_VAL` (682 mi²) is simulated in every pass, as part of
  Bend Bridge, but is not a CalSim3 arc and is not in the product.
- **Systems.** A system is a closure group of `arc_hierarchy.csv`. Its anchor is CalSim3's
  `FLOW-UNIMPAIRED` for 9 systems (FOLS, OROV, ST, YUBA, TU, ME, SJ, SHAS, TRIN) and a DWR
  unimpaired-flow series for the other 9 (UF 2, 4, 7, 10, 13, 14 for MKM, 15 for NHG, 20, 21).
  The product reads no anchor series: each system's arcs sum to dPL-CalSim's own system flow.
- **The share model's fit** reads the CalSim3 series of the 139 share arcs over WY1950–2015 less
  the held-out decade, every month. None of the training's masks applies: the share model learns
  CalSim3's division of a system's flow, including the months and arcs the model itself does not
  train on (extended, split, proportioned, impaired summers). Each arc's water-year volume is set
  by `k_a`, its ratio of CalSim3 to dPL-CalSim volume over those months.
- **The held-out decade in the fit** is out of the loss, the volume ratios, the input scaling and
  the selection of the penalty weight. It enters the centring of the share inputs (the means over
  WY1950–2015 of dPL-CalSim's own flows; no CalSim3 value).
- **Two passes.** The fit reads a pass that starts in WY1950 (WY1950–59 looped 20 times as
  spin-up); each series is a pass from October 1915 (WY1916–25 looped 20 times).
- **Routing.** An arc inside a trained entity is simulated on that entity's cells with their flow
  lengths, which run to the entity's outlet: for the 100 arcs with a CDEC or subbasin parent, to
  that gauge rather than the arc's own exit. The 32 arcs outside every entity are traced to where
  they leave their own polygon.

CalSim3 inflows the product does not give: `I_RUB002`, `I_FSL012` and `I_JBP006`, which have no
polygon in `CalSim3_Merged` (`I_RUB002`'s terrain is in `I_MFA025`'s), and `I_EBTML`, a CalSim3
series of about 28.5 TAF a year that no table of the repository lists.

## Open items

- `I_EBTML`: why it is outside the arc tables is not recorded.
- `I_SRBB_VAL`: its flow is simulated and dropped; whether the product should carry it is not
  decided.
