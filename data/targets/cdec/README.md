# data/targets/cdec: CDEC daily full natural flow

Daily full natural flow (FNF) as computed and published by the California Data Exchange
Center (CDEC sensor 8, daily duration). Role: target. `gage_15cdec.csv` is the daily
calibration target of the 15cdec domain (pooled calibration, WY1989 to 2003) and of the
`cdec_daily` family of the learned-parameter (dPL) models. The other files are pulls of daily
FNF for stations outside those 15 watersheds. Every series is CDEC's own computation.

## Files

| File | What | Source |
|------|------|--------|
| `gage_15cdec.csv` | `date, basin, flow`: observed daily FNF of the 15 CDEC watersheds, 1986 to 2019, in mm/day (converted from cfs over `data/inputs/domains/15cdec/basin_area.csv`; negatives and sentinels are NaN). 5.8 MB | Delivered with the original study archive (Wi & Steinschneider) |
| `stations.csv` | `id, name, sensor, sensor_type, duration, lat, lon, available_from, available_to, note`: every CDEC station with daily FNF (29) | `cdec_fnf.py survey` |
| `fnf_daily.csv` | `station, date, flow_cfs`: daily FNF of the 4 pulled stations (CLE, CSN, SNS, WHI), cfs exactly as CDEC serves them, through 2018-12-31 (the end of the forcing) | `cdec_fnf.py pull` |
| `fnf_daily_mm.csv` | `station, date, depth_mm`: depth companion for the two stations with a defined depth area, CLE at 692.86 mi2 (the `I_TRNTY` arc area) and CSN at the UF 13 arc-sum area (539.1 mi2); negative-flow days dropped | `build_fnf_depth.py` |
| `fnf_daily_mask.csv` | `entity_id, date, value_mm, kind, partner_date, partner_mm, evidence`: daily target values confirmed as computation artifacts (5 rows) | Hand-maintained, never generated |

The 29 stations of the survey:

| Classification | n | Stations |
|---|---|---|
| In the 15cdec set, already in `gage_15cdec.csv` (CDEC id, with the basin code where it differs) | 15 | SIS (SHA), SBB (BND), FTO (ORO), YRS, AMF (FOL), MKM, NHG, NML, TLG, MRC, SBF (MIL), KGF (PNF), KWT (TRM), SCC, KRI (ISB) |
| Pulled into `fnf_daily.csv` | 4 | CLE, CSN (targets); WHI, SNS (candidates) |
| Record starts 2022-06, after the end of the forcing | 3 | MSS, PSH, SDT (the Shasta inflow arms) |
| Outside the domain: no CalSim3 arcs, no forcing | 7 | EFC, WFC, EWR, WWR, TRF, OWL, RRH |

## How it is built

`gage_15cdec.csv` was delivered; it cannot be rebuilt from a clone. The pulls need the
`sacsma` environment and network access to CDEC (station search, station metadata, CSV
servlet), and no local inputs:

    python data/targets/cdec/cdec_fnf.py            # survey + pull + verify
    python data/targets/cdec/cdec_fnf.py survey     # stations.csv only
    python data/targets/cdec/cdec_fnf.py pull       # fnf_daily.csv + verify [--start --end]
    python data/targets/cdec/cdec_fnf.py verify     # re-check an existing fnf_daily.csv

The depth companion needs only tracked files and is reproduced byte for byte. One of its
checks reads `data/targets/dwr_unimpaired/uf_monthly_mm.csv`, so that table is built first:

    python data/targets/dwr_unimpaired/build_uf_depth.py
    python data/targets/cdec/build_fnf_depth.py

Only stations with a stated depth area are converted (`FNF_DEPTH_AREAS` in the script); SNS
and WHI have none. A station added there comes with its own depth area.

## Checks

- The 15 aliases are proven identities: converted with the repository's own areas, each CDEC
  series reproduces `gage_15cdec.csv` at r = 1.0000 and volume ratio 1.0000 over 7,300 to
  11,700 overlapping days.
- The pulled stations, monthly sums against the monthly calibration targets of
  `data/targets/callite/` (complete months only, negatives masked):

| Station | Record | Months | r | Ratio | Summary |
|---|---|---|---|---|---|
| CLE | 1986-04 on | 224 | 0.9976 | 0.988 | Trinity daily target (arc `I_TRNTY`); about 1% low from a footprint difference |
| CSN | 1999-04 on | 137 | 0.9985 | 0.937 | Cosumnes daily target; the daily product misses about 6% of the volume |
| SNS | 1992-06 on | 170 | 0.998 | 1.036 | Candidate; a second Stanislaus record, larger footprint than NML |
| WHI | 2000-10 on | none | none | none | Candidate; the noisiest product of the four, no target to verify against |

- `build_fnf_depth.py` refuses to write unless the table holds exactly CLE and CSN, has no
  duplicate and no negative value, CLE's monthly volume agrees with the 11obs TNL target
  (r > 0.995, ratio 0.96 to 1.01) and CSN's with UF 13 (r > 0.99, ratio 0.84 to 0.99).

## Know before using

- `fnf_daily.csv` is cfs only and keeps negative days. Mask `flow_cfs < 0` before use:
  negative flow is a computation artifact. Daily FNF uses less data than the month-end
  computation and goes negative from reservoir-elevation noise; the monthly product is the
  reconciled volume. `gage_15cdec.csv` is already a depth with its negatives set to NaN.
- caution: dropping a negative day keeps its compensating positive partner, the day the
  reservoir-elevation error reverses, as a spurious spike (CLE 1996-05-26: +102 mm after
  -92 mm, the second-largest day of the Trinity record). `fnf_daily_mask.csv` lists such days.
  A row is added only when the day is confirmed against an independent record: the month's
  raw daily sum (negatives kept) against CDEC's reconciled monthly FNF, a nested USGS gauge
  showing no rise, or both. The 15 basins of `gage_15cdec.csv` already carry 22 such masks
  from the original study; the five rows here are what that pass missed (CLE had none).
- The mask is opt-in and covers any of the 17 `cdec_*` training entities, whichever file
  their series comes from: `sacsma dpl train --obs-mask data/targets/cdec/fnf_daily_mask.csv`
  removes the days from training and scoring, and the checkpoint carries the list. The series
  files themselves stay as they are.
- caution: the mask does not fix the upward bias of low-flow targets that comes from dropping
  negatives (August to September, stored against raw: CLE +40 %, FOL +48 %, NHG +97 %).
- CLE: the FNF is computed at Trinity Dam (692 mi2), while the monthly target basin TNL is
  defined at Lewiston, about 7 mi downstream (719 mi2); hence the ratio 0.988. Harmless in
  training, where each series is a depth over its own footprint.
- caution: CSN's daily FNF misses about 6% of the river's water (about 19,400 AF/yr). It is
  not a footprint or unit issue: CDEC's own monthly FNF (sensor 65) for the station matches
  the DWR unimpaired series within about 1%, so the deficit is specific to the daily
  computation.
- SNS is the Stanislaus at Goodwin Dam, a few miles below New Melones (NML, one of the 15).
  It is not a duplicate: its footprint is the 11obs SNS basin (980.45 mi2 against NML's 900),
  the two series correlate at only 0.932 daily, and different agencies compute them (USGS and
  USBR). Its daily product runs about 3.5% high against the 11obs monthly target. Using it
  would give the Stanislaus a daily and monthly pair, at the cost of double daily coverage.
- caution: WHI (Clear Creek at Whiskeytown Dam, CalSim3 arc `I_WKYTN`, computed by USBR) is
  noisy: 24% of days are negative (worst -1,144 cfs), and no monthly target exists to verify
  it against. Its record is kept from 2000-10, the published start; the servlet also returns
  an unpublished January to September 1990 fragment, dropped at build time (`RECORD_START`).
- Each station's rows in `fnf_daily.csv` start and end on a real value. Gaps inside a record
  are left as they are, and CDEC returns no rows for dates outside a station's record.
- `available_from` and `available_to` in `stations.csv` describe CDEC's data-collection
  entries, not an inventory of stored values, so they can understate the record. Many
  stations list several entries (an agency data exchange plus CDEC's own computed series from
  2013-10); the columns span all of them.
- The survey universe is CDEC's station search for sensor 8 at daily duration, not the
  daily-FNF report page, which omits reservoir stations that still carry the sensor (NML,
  NHG, WHI). A station absent from the search has no daily FNF on CDEC.

## Read by

- `gage_15cdec.csv`: `sacsma.cdec15.load_gage`, and through it `sacsma.cdec15.plots`
  (`sacsma plots --domain 15cdec`), `sacsma.calsim.plots`, `sacsma.dpl.data`,
  `sacsma.dpl.hybrid.data`, `sacsma.dpl.evaluate`, `sacsma.dpl.multi_timescale` and
  `sacsma.dpl.evaluate_multi_timescale`.
- `fnf_daily_mm.csv`: `sacsma.dpl.multi_timescale` and `sacsma.dpl.evaluate_multi_timescale`
  (the `cdec_CLE` and `cdec_CSN` entities, through the registry's `obs_store` column).
- `fnf_daily_mask.csv`: `sacsma dpl train --obs-mask`.
- `gage_15cdec.csv`, `fnf_daily.csv`, `stations.csv`: the registry builder
  `data/inputs/domains/multifamily/build_entities.py` (see
  [its README](../../inputs/domains/multifamily/README.md)).
