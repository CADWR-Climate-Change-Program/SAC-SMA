# data/targets/callite: monthly full natural flow of the CalLite watershed sets

Monthly observed full natural flow (FNF) of the three per-watershed calibration domains:
`9unimp` (9 creek watersheds), `11obs` (11 major rim rivers) and `12rim` (12 rim-reservoir
inflows). Role: target. These are the records the archived per-watershed SAC-SMA calibrations
(Wi & Steinschneider) were fitted to, delivered with the study archive.

## Files

| File | What | Source |
|------|------|--------|
| `calib_<domain>_monthly.csv` | `basin, date, sim_mm, obs_mm, cal_start, cal_end`: the observed monthly FNF target (`obs_mm`, mm/month), the MATLAB monthly simulation (`sim_mm`) and the calibration window, over the calibration period only. 0.4 MB each | Parsed from each watershed's calibration log |
| `fnf_<domain>_monthly.csv` | `date, basin, obs_mm, cal_start, cal_end`: the full-period monthly observed FNF (1922 on), which allows validation outside the calibration window. 0.5 to 0.7 MB each | `9unimp`, `11obs`: historical FNF records. `12rim`: the CalSim SV DSS spreadsheet export (reservoir-inflow series ratio-matched to the calibration-log observations) |

`<domain>` is `9unimp`, `11obs` or `12rim`: six files.

## How it is built

Delivered; cannot be rebuilt from a clone.

## Checks

- The `9unimp` and `11obs` targets are DWR's published unimpaired flows.
  `python data/targets/dwr_unimpaired/dwr_unimpaired.py --verify` scores each published series
  against `fnf_<domain>_monthly.csv` and gets r = 1.000000 on 16 of the 18 mapped basins. The
  residual is area normalisation only: the mean ratio recovers the drainage area DWR
  normalised by (BND 8900 mi2, FTO 3607 mi2, AMF 1885 mi2). See
  [the DWR unimpaired README](../dwr_unimpaired/README.md).
- The two basins that are not exact are BLB and CacheCreek (below).

## Know before using

- caution: the `11obs` BLB target is not DWR's unimpaired series for Stony Creek. It spans
  1994 to 2014 and is a gauged Black Butte reservoir-inflow record (against the unimpaired
  series: annual r 0.07 to 0.997, ratio 0.63 to 6.2). The unimpaired series of the same
  watershed is the `9unimp` StonyCreek target.
- CacheCreek (`9unimp`) matches the published series exactly in 88 of 89 water years; only
  WY2010, the last year of its record here, departs.
- SHA and TNL (`11obs`) have no published unimpaired table of their own: Shasta is inside the
  Sacramento River near Red Bluff subbasin, and the Trinity is not a Central Valley subbasin.
- `obs_mm` is a depth over the domain's own basin area. The areas of `9unimp` and `11obs` are
  in `data/inputs/domains/<domain>/basin_area.csv`. The `12rim` table there holds the areas the
  study's CalLite wrapper converted its output with; whether they are the basis of the `12rim`
  `obs_mm` is not recorded.
- Basin codes are domain-specific: short codes in `11obs` and `12rim` (SHA, BND, ...),
  CamelCase names in `9unimp` (CacheCreek, StonyCreek, ...).

## Read by

- `sacsma.calsim.load_fnf_monthly` and `sacsma.calsim.load_calib_monthly`, used by
  `sacsma.calsim.plots` (`sacsma plots --domain <domain>`; the full-period file, with the
  calibration file as fallback) and `sacsma.calsim.compare` (the target scored against CalSim3).
- Checks in `data/targets/cdec/cdec_fnf.py`, `data/targets/cdec/build_fnf_depth.py` and
  `data/targets/dwr_unimpaired/dwr_unimpaired.py --verify`.
- `fnf_11obs_monthly.csv`: `data/inputs/domains/multifamily/build_entities.py`.
