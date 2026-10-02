# data/targets/calsim3: CalSim3 historical rim inflows

The monthly historical series of DWR's CalSim3 model: the rim `INFLOW` arcs and the
whole-watershed `FLOW-UNIMPAIRED` series of the 11 rim systems, with the tables that record
how CalSim3 built each arc series and which of its months are the arc's own gauge record.
Role: target. The series have two uses. The dPL-CalSim model is fitted to arc-months of
them (the `calsim_monthly` family: the own-record months of the 64 `train_default` arcs). The
calibrated SAC-SMA cross-compare (`sacsma calsim`) only scores against them. A family fitted
by any model in this repository is a target, so they are kept here.

## Files

| File | What | Source |
|------|------|--------|
| `calsim3_inflow_monthly.csv` | `date, arc, flow_taf`: CalSim3 historical `INFLOW`, TAF/month on month-end dates, WY1922 to 2021. The reference of the cross-compare. 9.6 MB | Extracted from the CalSim3 SV DSS |
| `calsim_unimpaired_monthly.csv` | `date, system, flow_taf`: CalSim `FLOW-UNIMPAIRED` whole-watershed series of the 11 rim systems (SHAS, SRBB, OROV, YUBA, FOLS, ST, TU, ME, SJ, TRIN, WH), decade-merged 1920 to 2021. The per-basin anchor reference of the cross-compare. 0.4 MB | Extracted from the CalSim3 SV DSS |
| `calsim3_arc_derivation.csv` | How CalSim3 built the rim `INFLOW` series: 201 arcs (121 Sacramento, 80 San Joaquin), `arc, region, name, flow_type, method_class, method, gauge, period, record_frac, record_owner, tier1_set, notes, extension_period`. 0.04 MB | Transcribed by hand from DWR's *CalSim3 Hydrology Report* (2023), Tables 5-1 / 5-2 (Sacramento) and 5-5 / 5-6 (San Joaquin). Hand-maintained, never generated |
| `arc_hierarchy.csv` | The rim-arc target set, 203 rows (every `CalSim3_Merged` polygon plus every derivation arc): area, anchors, closure group, record and tier of each arc (columns below). 0.03 MB | `build_calsim_arcs.py` |
| `arc_obs_mask.csv` | `arc, date` (month-end): every arc-month that is the arc's own observed record inside the training water years WY1950 to 1975 and WY1986 to 2015. 42,010 rows on 95 arcs, 25,523 of them on the 64 tier-A arcs. 0.9 MB (LFS) | `build_calsim_arcs.py` |
| `calsim3_inflow_monthly_mm.csv` | `arc, date, depth_mm`: the 196 rim `INFLOW` series as mm/month over each arc's `SQ_MI`, full record WY1922 to 2021. 8.8 MB (LFS) | `build_calsim_arcs.py` |

Columns of `arc_hierarchy.csv`: `sq_mi` (the `CalSim3_Merged` `SQ_MI`); `status` (196
`rim_arc`, the series-less `I_SRBB_VAL`, 3 without a polygon, 3 excluded below the rim);
`containing_anchors` from inner to outer, `innermost_anchor`, `top_anchor`, `fu_systems`;
`residual` (a by-difference accretion arc); `closure_group`, `closure_anchor`,
`closure_excluded`; `duplicates_entity` (the 7 single-arc training entities);
`usgs_training_gauge`; `reg_screen` (the naturalness screen); `record_months` (own record in
the training water years, extension years carved out) and `record_months_listed` (the listed
gauge period alone); `tier`, `tier_listed`, `tier_note`; `train_default` (tier A and not a
duplicate: 64 arcs).

## How it is built

The two raw series (`calsim3_inflow_monthly.csv`, `calsim_unimpaired_monthly.csv`) were
extracted from the CalSim3 SV DSS and delivered; they cannot be rebuilt from a clone.
`calsim3_arc_derivation.csv` is edited by hand only.

The three derived tables are deterministic and are reproduced byte for byte from tracked
files (environment `sacsma`, from the repository root):

    python data/targets/calsim3/build_calsim_arcs.py [--data-dir data] [--out-dir <folder>]

It reads the two hand or delivered tables above, `data/inputs/calsim3/tier1_sets.csv`, layer
`CalSim3_Merged` of `data/inputs/calsim3/calsim3.gpkg`,
`data/targets/dwr_unimpaired/uf_locations.csv`, and the 95 base rows of
`data/inputs/domains/multifamily/entities.csv`. Run it before the three registry builders
when they are given `--calsim-arcs`. Starting without a registry, build the base registry
first (see [the registry README](../../inputs/domains/multifamily/README.md)).

## Checks

Gates of `build_calsim_arcs.py` (it stops if one fails): 196 rim arcs, arc ids unique, the
only node without a series is `I_SRBB_VAL`, the single-arc duplicates are exactly the 7
expected; the depth table holds 196 x 1200 rows and its TAF to mm to TAF round trip is
below 3e-16 relative (the gate is 1e-9); the mask has no duplicate row, its rows per arc
equal `record_months`, and no mask month falls in the held-out WY1976 to 1985 or before
WY1950.

## Know before using

- caution: an `INFLOW` series is CalSim3's reconstruction, not an observation. The derivation
  table groups the methods into eight classes (`method_class`): a gauged record extended by
  regression, a complete record, a split or proportioning of a donor gauge, a difference of
  gauges, a reservoir mass balance and others. Only the months of `arc_obs_mask.csv` are an
  arc's own gauge record: the listed gauge period minus every water year the report lists as
  a correlation extension.
- The mask covers every own-record arc (tiers A, B, C and the gauged D and X arcs). Which
  arcs train is the registry's choice; `train_default` is tier A.
- Tiers (`tier` = with the extension years carved out, `tier_listed` = the listed gauge
  period alone): A own gauge with a record in the training water years, passes the
  naturalness screen; B the same, fails it; C by difference of own gauges with a record,
  passes; D reservoir mass balance; E donor split or proportioned; F no own record in the
  training water years; X none of these (`tier_note` says why).
- The naturalness screen is the `REGULATED` list in the script, the complete record of a
  one-time review of the derivation arcs (195 screened, 40 flagged). The arcs it did not cover
  (`I_FSL012`, `I_JBP006`, `I_RUB002`, `I_SJR258`, `I_SJR265`, `I_TUO054`, and `I_ECHOL`,
  which the derivation table lacks) have an empty `reg_screen`.
- Anchors are the CalSim3 `FLOW-UNIMPAIRED` systems (`FU_<SYS>`: member arcs from
  `tier1_sets.csv`, with SRBB nesting SHAS and the valley node, TRIN = `I_TRNTY` alone, plus
  WH = `I_WKYTN`) and every `cdec_daily` or `uf_monthly` registry entity with an arc list.
  `innermost_anchor` is the smallest containing arc set, `top_anchor` the largest (ties: FU,
  then uf, then cdec). WH is `I_WKYTN` alone because its series is that arc's (WY1950 to 2015
  ratio 0.9956, r 0.996; with `I_CLR011` added the ratio would be 1.158).
- Closure groups (the water-year closure of `sacsma.dpl.calsim.arcs`): an arc in a
  `FLOW-UNIMPAIRED` system closes in its innermost system other than SRBB, so Shasta and
  Whiskeytown close on their own. An SRBB-only arc is excluded: the rim arcs sit about 973
  TAF/yr under the SRBB series by design, and the valley node holds the gap. An arc outside
  every system closes in its largest training-entity anchor, to that entity's DWR unimpaired
  monthly series (a CDEC entity uses the UF with the identical arc set). `uf_03`, the routed
  outflow below Clear Lake, is excluded.
- Depth uses `AF_PER_MM_MI2` of `sacsma.dpl.calsim.tier1` and the arc's `SQ_MI`. The depth
  table holds the arcs with both a series and a polygon; `I_RUB002`, `I_JBP006` and
  `I_FSL012` have no polygon, and the below-rim arcs are left out.
- `calsim3_arc_derivation.csv`: flow type, method flags, gauge and period are as printed.
  `extension_period` is the tables' *Period of Data Extension* (Table 5-2, printed pp. 5-23 to
  5-29 = PDF pp. 129 to 135 of `final_cs3_hydrologyreport_v2.pdf`; Table 5-6, pp. 5-45 to
  5-47 = PDF 151 to 153; read from the page images): the water years CalSim3 filled by the
  annual regression on a long-term station. Periods are separated by `; `, an arc's separate
  regression rows by ` | `, and a blank means the arc is in neither table (138 of 201 filled:
  96 Sacramento, 42 San Joaquin). `record_frac` is the share of WY1950 to 1984 inside the
  listed gauge period. `record_owner` is `arc` (the arc's own gauge) or `donor` (the
  neighbour or downstream record it was split or proportioned from), blank where no gauge
  period falls in that window. Arc ids follow `calsim3_inflow_monthly.csv`.
- Only typesetting was normalized in the transcription (`1922 -1 950` to `1922 - 1950`, en
  dashes, comma lists to `; `; the printed labels `I_MFA0025` and `I_MFA0023` are `I_MFA025`
  and `I_MFA023`).
- caution: printed errors are kept. `I_BCN010`'s extension `1922 - 1966` overlaps its own
  10/59 to 09/67 gauge record. The printed regression reproduces the CalSim3 series exactly
  in WY1922 to 1959 and WY1968 to 2015 but not in WY1960 to 1967, so the period is really
  1922 - 1959.

## Read by

- `calsim3_inflow_monthly.csv`: `sacsma.calsim.load_calsim3_monthly`, used by
  `sacsma.calsim.compare` (`sacsma calsim`) and `sacsma.dpl.calsim.tier1`.
- `calsim_unimpaired_monthly.csv`: `sacsma.calsim.compare.load_unimpaired_monthly`,
  `sacsma.dpl.calsim.tier1`, `sacsma.dpl.calsim.arcs` (closure anchors).
- `calsim3_inflow_monthly_mm.csv`, `arc_obs_mask.csv`: `sacsma.dpl.calsim.arcs`, and through
  it `sacsma.dpl.multi_timescale` and `sacsma.dpl.evaluate_multi_timescale` (the `cs_<ARC>`
  entities; the registry's `obs_store` column points at the depth table).
- `arc_hierarchy.csv`: `sacsma.dpl.calsim.arcs`, `sacsma.dpl.calsim.product`,
  `sacsma.dpl.train`; `data/inputs/domains/multifamily/build_entities.py --calsim-arcs`.
- `calsim3_arc_derivation.csv`: `sacsma.dpl.calsim.arcs`, `sacsma.dpl.calsim.atlas`.
- Both raw series: `data/targets/dwr_unimpaired/check_uf_locations.py`.
