# data/targets/cdec: CDEC full natural flow, daily and monthly

Daily full natural flow (FNF) as computed and published by the California Data Exchange
Center (CDEC sensor 8, daily duration), and the monthly FNF of the benchmark sites (sensor 65,
monthly duration). Role: target. `gage_15cdec.csv` is the daily calibration target of the
15cdec domain (pooled calibration, WY1989 to 2003) and of the `cdec_daily` family of the
learned-parameter (dPL) models. The other daily files are pulls of daily FNF for stations
outside those 15 watersheds. `fnf_monthly.csv` is the monthly FNF, the reconciled month-end
volume, at the 17 `cdec_daily` sites of the learned-parameter registry (`sacsma benchmark` scores
13 of them). Every series is CDEC's own
computation.

## Files

| File | What | Source |
|------|------|--------|
| `gage_15cdec.csv` | `date, basin, flow`: observed daily FNF of the 15 CDEC watersheds, 1986 to 2019, in mm/day (converted from cfs over `data/inputs/domains/15cdec/basin_area.csv`; negatives and sentinels are NaN). 5.8 MB | Delivered with the original study archive (Wi & Steinschneider) |
| `stations.csv` | `id, name, sensor, sensor_type, duration, lat, lon, available_from, available_to, note`: every CDEC station with daily FNF (29) | `cdec_fnf.py survey` |
| `fnf_daily.csv` | `station, date, flow_cfs`: daily FNF of the 4 pulled stations (CLE, CSN, SNS, WHI), cfs exactly as CDEC serves them, through 2018-12-31 (the end of the forcing) | `cdec_fnf.py pull` |
| `fnf_daily_mm.csv` | `station, date, depth_mm`: depth companion for the two stations with a defined depth area, CLE at 692.86 mi2 (the `I_TRNTY` arc area) and CSN at the UF 13 arc-sum area (539.1 mi2); negative-flow days dropped | `build_fnf_depth.py` |
| `fnf_daily_mask.csv` | `entity_id, date, value_mm, kind, partner_date, partner_mm, evidence`: daily target values confirmed as computation artifacts (5 rows) | Hand-maintained, never generated |
| `fnf_monthly.csv` | `date, station, flow_af, flag`: monthly FNF (sensor 65) of the 17 `cdec_daily` sites at 16 stations (below), acre-feet per month and CDEC's data flag exactly as CDEC serves them (`flag` empty when none), stamped at the month end, WY1922 (SCC from WY1931, KRI from WY1930) to 2025-09; a missing month would be an empty `flow_af` (none so far). 19,764 rows, 0.46 MB | `cdec_fnf.py monthly` |

The 29 stations of the survey:

| Classification | n | Stations |
|---|---|---|
| In the 15cdec set, already in `gage_15cdec.csv` (CDEC id, with the basin code where it differs) | 15 | SIS (SHA), SBB (BND), FTO (ORO), YRS, AMF (FOL), MKM, NHG, NML, TLG, MRC, SBF (MIL), KGF (PNF), KWT (TRM), SCC, KRI (ISB) |
| Pulled into `fnf_daily.csv` | 4 | CLE, CSN (targets); WHI, SNS (candidates) |
| Record starts 2022-06, after the end of the forcing | 3 | MSS, PSH, SDT (the Shasta inflow arms) |
| Outside the domain: no CalSim3 arcs, no forcing | 7 | EFC, WFC, EWR, WWR, TRF, OWL, RRH |

The 17 `cdec_daily` sites and the station of their monthly FNF:

| Site | Monthly station | Note |
|---|---|---|
| SHA, BND, ORO, FOL, YRS, CSN, MKM, TLG, MRC, MIL, PNF, TRM, SCC, ISB | SIS, SBB, FTO, AMF, YRS, CSN, MKM, TLG, MRC, SBF, KGF, KWT, SCC, KRI | The daily station itself |
| CLE | TNL (Trinity River at Lewiston) | CLE (Trinity Dam) has no monthly series. TNL is the Trinity's Bulletin 120 forecast point, about 7 mi downstream, at 719 mi2 against CLE's 692.86 |
| NML | SNS (Stanislaus River at Goodwin Dam) | NML (New Melones) has no monthly series. SNS is the Stanislaus forecast point, below New Melones and Tulloch, at 996 mi2 (the area DWR normalises it by) against NML's 900 |
| NHG | none | No Calaveras River station on CDEC carries sensor 65 |

## How it is built

`gage_15cdec.csv` was delivered; it cannot be rebuilt from a clone. The pulls need the
`sacsma` environment and network access to CDEC (station search, station metadata, CSV
servlet), and no local inputs:

    python data/targets/cdec/cdec_fnf.py                 # survey + pull + verify + monthly
    python data/targets/cdec/cdec_fnf.py survey          # stations.csv only
    python data/targets/cdec/cdec_fnf.py pull            # fnf_daily.csv + verify [--start --end]
    python data/targets/cdec/cdec_fnf.py verify          # re-check an existing fnf_daily.csv
    python data/targets/cdec/cdec_fnf.py monthly         # fnf_monthly.csv + checks [--start --end]
    python data/targets/cdec/cdec_fnf.py verify-monthly  # re-check an existing fnf_monthly.csv

The monthly pull is one call to the CSV servlet (`SensorNums=65`, `dur_code=M`) for the 16
stations of `MONTHLY` in the script. Its checks read `gage_15cdec.csv`, `fnf_daily.csv`,
`fnf_daily_mask.csv`, `data/inputs/domains/15cdec/basin_area.csv` and, for the identity check,
`data/targets/callite/fnf_11obs_monthly.csv`. How the stations were found: CDEC's station
search for sensor 65 at monthly duration lists 80 stations, with 14 of the 17 daily stations
but not CLE, NML or NHG. The station pages of those three list only daily sensor-8 FNF, and the
servlet returns no rows for them. TNL and SNS are in the search, each with sensor 65 from
before 1922. The same search restricted to the Calaveras River basin returns no station (the
basin has 21, NHG among them). Two pulls on 2026-10-06 were byte-identical.

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
- The monthly FNF against the sum of the daily FNF the repository holds (`cdec_fnf.py
  verify-monthly`), WY1991 to 2018 (336 months). The daily records are read as the benchmark
  reads them: `gage_15cdec.csv` back to cfs over `data/inputs/domains/15cdec/basin_area.csv` for
  15 sites, `fnf_daily.csv` for CLE and CSN. A daily month counts only when every day is valid
  (finite, not negative, not in `fnf_daily_mask.csv`). The monthly record has all 336 months at
  every station. r and the volume ratio (monthly over daily) are taken on the complete daily
  months:

| Site | Monthly station | Monthly months | Complete daily months | r | Volume ratio |
|---|---|---|---|---|---|
| SHA | SIS | 336 | 315 | 0.9998 | 0.999 |
| CLE | TNL | 336 | 233 | 0.9972 | 1.016 |
| BND | SBB | 336 | 227 | 0.9992 | 1.004 |
| ORO | FTO | 336 | 279 | 0.9988 | 1.012 |
| FOL | AMF | 336 | 227 | 0.9991 | 0.992 |
| YRS | YRS | 336 | 234 | 0.9958 | 1.059 |
| CSN | CSN | 336 | 233 | 0.9992 | 1.060 |
| MKM | MKM | 336 | 217 | 0.9977 | 1.028 |
| NML | SNS | 336 | 217 | 0.9843 | 1.092 |
| NHG | none | 0 | 95 | none | none |
| TLG | TLG | 336 | 200 | 0.9995 | 1.004 |
| MRC | MRC | 336 | 329 | 0.9985 | 1.035 |
| MIL | SBF | 336 | 257 | 0.9996 | 1.001 |
| PNF | KGF | 336 | 308 | 0.9999 | 1.001 |
| TRM | KWT | 336 | 302 | 1.0000 | 0.999 |
| SCC | SCC | 336 | 267 | 0.9993 | 1.003 |
| ISB | KRI | 336 | 326 | 0.9998 | 1.000 |

  Nine sites agree within 1 % in volume and ORO within 1.2 %. Where the records part (the
  script also prints the ratio by season: December to March, April to July, August to
  November):
  - NML (SNS) +9.2 %, r 0.984: not the same point. SNS drains 10.7 % more area than NML, and
    NML's daily series is USBR's New Melones computation. The ratio moves with the season
    (April to July 1.173, August to November 0.471), so it is not an area factor alone.
  - YRS +5.9 % and CSN +6.0 %: the daily product misses water in the wet and the melt months
    (YRS 1.069 and 1.066, CSN 1.063 and 1.041). It is not an area or unit effect: both sides
    are volumes of CDEC's own series (CSN's daily is CDEC's cfs; YRS's daily depth goes back
    to cfs over the area it was made with, which reproduces CDEC's daily series). CSN's
    deficit is the known one (Know before using). YRS's was not known, and its cause is not
    established here.
  - MRC +3.5 % and MKM +2.8 %: smaller deficits of the same kind. MRC's is in every season,
    largest in the low-flow months (1.014, 1.033, 1.274); MKM's is in the melt months (April
    to July 1.043). Not explained here.
  - CLE (TNL) +1.6 %: the footprint. TNL drains 3.8 % more area, the reach from Trinity Dam
    to Lewiston. The CLE row of the table above (0.988 against the 11obs TNL target, over
    other months) shows the same difference.
  - In August to November, when the volumes are small, the two records part by 10 to 28 % at
    several sites (CLE 0.896, YRS 0.888, MKM 0.842, ORO 1.096, MRC 1.274, CSN 1.281).

  NHG has no monthly series; its daily record has 95 complete months of the 336.
- The monthly series is the record the `11obs` targets were made from. For the seven
  stations that are also `11obs` basins (AMF, FTO, MRC, SNS, TLG, TNL, YRS), `fnf_monthly.csv`
  equals `fnf_11obs_monthly.csv` to 1 AF in every month from 1922-01 to 2013-09 (r = 1.000000
  over 1,105 months to 2014-01), once the depth is multiplied by the area DWR normalised by:
  AMF 1885, FTO 3607, MRC 1040, SNS 996, TLG 1542, TNL 719, YRS 1100 mi2. Only the archive's
  last months (2013-10 to 2014-01; two at MRC, none at TLG and TNL) differ from CDEC's current
  values.

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
- caution: the daily FNF of YRS, MRC and MKM also runs below the monthly FNF over WY1991 to
  2018 (by 5.9 %, 3.5 % and 2.8 % of the volume, see Checks). These daily series are
  calibration and training targets (`gage_15cdec.csv`).
- `station` in `fnf_monthly.csv` is the CDEC station, not the site: CLE's monthly series is
  TNL's and NML's is SNS's, each over a larger footprint (the site table above).
- caution: TNL holds 0 AF with flag `N` in 1996-08 and 1996-09, kept as delivered; DWR's
  record (the `11obs` TNL target) holds 0 too. CLE's raw daily sums for those months are 4,060
  and 413 AF, with 13 and 15 negative days.
- `flag` is CDEC's `DATA_FLAG` as served: `r` (1,384 months), `e` (71) and `N` (2). CDEC
  serves no legend with the data. MKM carries `r` on 870 of its 1,248 months.
- CDEC revises recent months, so a later monthly pull can differ from this one. Several
  stations hold sensor 65 from before WY1922 (by their station pages: FTO, YRS, CSN, MKM, TLG,
  MRC, TNL and SNS, from 1900 to 1911); the pull starts at WY1922.
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
- `fnf_monthly.csv`: `sacsma.benchmark.flows` (`sacsma benchmark`: the observed monthly
  volume of the 10 benchmark sites whose own station carries it, and of CLE from TNL scaled to
  CLE's daily volume, read through `paths.cdec_fnf(data_dir, name="fnf_monthly.csv")`; SNS and
  the four Tulare stations are not used).
- `gage_15cdec.csv`, `fnf_daily_mm.csv`, `fnf_daily_mask.csv`, `stations.csv`:
  `sacsma.benchmark.flows` (`sacsma benchmark`: the observed monthly volume of NML, which
  has no usable monthly series, and the scale of CLE's; a month counts only when all its
  days are valid; the mask days are removed).
- `gage_15cdec.csv`, `fnf_daily.csv`, `stations.csv`: the registry builder
  `data/inputs/domains/multifamily/build_entities.py` (see
  [its README](../../inputs/domains/multifamily/README.md)).
