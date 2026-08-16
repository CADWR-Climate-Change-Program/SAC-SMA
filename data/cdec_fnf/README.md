# `data/cdec_fnf/` — CDEC daily full natural flow beyond the 15-CDEC store

Built by `dataprep/cdec_fnf.py` (survey 2026-08-03). Re-run: bare = everything;
`survey` / `pull [--start --end]` / `verify` run the steps separately.

| File | Contents |
|---|---|
| `stations.csv` | Every CDEC station with daily FNF (sensor 8, daily): id, name, lat/lon, advertised period. Classification of each station: the table below |
| `fnf_daily.csv` | `station, date, flow_cfs` — daily FNF for the 4 pulled stations, verbatim cfs, through 2018-12-31 (forcing end) |
| `fnf_daily_mm.csv` | `station, date, depth_mm` — derived depth companion for the two approved daily targets (built by `dataprep/build_obs_depth.py`): CLE at 692.86 mi² (the `I_TRNTY` arc area) and CSN at the UF 13 arc-sum (539.1 mi²); negative-flow days dropped |

## Survey

The universe is CDEC's **station search for sensor 8 at daily duration** (29
stations), not the daily-FNF report page, which omits reservoir stations that
still carry the sensor (NML, NHG, WHI). Absence from the search = no daily FNF
on CDEC.

| Classification | n | Stations |
|---|---|---|
| In 15_cdec — already in `data/cdec15/gage.csv` | 15 | SIS↔SHA, SBB↔BND, FTO↔ORO, YRS, AMF↔FOL, MKM, NHG, NML, TLG, MRC, SBF↔MIL, KGF↔PNF, KWT↔TRM, SCC, KRI↔ISB |
| Pulled — downloaded here | 4 | CLE, CSN (targets), WHI, SNS (candidates) |
| After 2018 — record starts 2022-06, after forcing end | 3 | MSS, PSH, SDT (the Shasta inflow arms) |
| Outside domain — no CalSim3 arcs / forcing | 7 | EFC, WFC, EWR, WWR, TRF, OWL, RRH |

## Verification (2026-08-03)

The 15 aliases are proven identities: converted with the repo's own areas,
each reproduces `gage.csv` at **r = 1.0000, volume ratio = 1.0000** over
7,300–11,700 overlapping days.

The pulled 4, monthly sums vs the repo's monthly calibration targets
(complete months only, negatives masked):

| Station | Record | Months | r | Ratio | Summary |
|---|---|---|---|---|---|
| CLE | 1986-04 → | 224 | 0.9976 | 0.988 | Trinity daily target (arc `I_TRNTY`); ~1% low from a footprint difference |
| CSN | 1999-04 → | 137 | 0.9985 | 0.937 | Cosumnes daily target; daily product misses ~6% of the volume |
| SNS | 1992-06 → | 170 | 0.998 | 1.036 | Candidate; a second Stanislaus record, larger footprint than NML |
| WHI | 2000-10 → | — | — | — | Candidate; noisiest product of the four, no verify target |

**CLE** — the FNF is computed at Trinity Dam (692 mi²), while the monthly
target basin TNL is defined at Lewiston, ~7 mi downstream (719 mi²). A
slightly smaller basin yields slightly less water, hence ratio 0.988. Harmless
for training because targets are trained in depth (volume ÷ own area), so each
series is normalized by its own footprint.

**CSN** — the daily FNF is missing ~6% of the river's water (~19,400 AF/yr).
Not a footprint or unit issue: CDEC's own *monthly* FNF (sensor 65) for the
same station matches the DWR Unimpaired Flow target within ~1%, so the
deficit is specific to the daily computation.

**SNS** — Stanislaus at Goodwin Dam, a few miles *below* New Melones (NML,
already a 15-CDEC target). Not a duplicate: its footprint is the 11obs SNS
basin (980.45 mi² vs NML's 900), the two series correlate only 0.932 daily,
and they are computed by different agencies (USGS vs USBR). The 1.036 ratio
says its daily product runs ~3.5% high against the 11obs monthly target. 
Adding it would give the Stanislaus a
daily+monthly pair (like CLE/TNL for Trinity) at the cost of double daily
coverage of one river.

**WHI** — Clear Creek at Whiskeytown Dam (CalSim3 arc `I_WKYTN`),
USBR-computed, kept from 2000-10 (the published start; the servlet also
returns an unpublished Jan–Sep 1990 fragment, dropped at build time via
`RECORD_START`). Noisy: 24% of days are negative (worst −1,144 cfs). No
monthly target exists in the repo to verify against, so treat it with the
most caution.

## Conventions

- **cfs only, verbatim** — no depth column. The training pipeline converts
  each series to depth (mm/day) with the observing site's own drainage area.
- **Negative days kept** — mask `flow_cfs < 0` before use (negative flow is
  a computation artifact). Per the FNF report notes, daily
  FNF uses less data than the month-end computation and goes negative from
  reservoir-elevation noise — the monthly product is the reconciled volume.
- Each station's rows start and end on a real value; gaps inside a record are
  left as they are, and CDEC returns no rows at all for dates outside a
  station's record.
- `available_from/to` in `stations.csv` describe CDEC's data-**collection**
  entries, not an inventory of stored values, so they can understate the
  record: WHI advertises 2000-10 but serves a 1990 stub. Many stations list
  several entries (an agency data exchange plus CDEC's own computed series
  from 2013-10); the columns span all of them. `fnf_daily.csv` keeps published
  records only — WHI's unpublished 1990 fragment is dropped at build time.
