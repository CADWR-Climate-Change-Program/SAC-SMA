# data/reference/dwr_swat: DWR's SWAT rim simulation and the valley and Delta totals

Two monthly tables (TAF, WY1922–2014) transcribed from DWR Bay-Delta Office, *Estimates of
Natural and Unimpaired Flows for the Central Valley of California: WY 1922-2014* (DRAFT, March
2016): the SWAT-simulated rim outflow the report labels "natural" (Appendix C, 18 subbasins),
and the six valley and Delta totals of the unimpaired flow (Appendix B, tables B-25 to B-30).
Role: reference. The unimpaired series of the same report are a target and live in
[`data/targets/dwr_unimpaired`](../../targets/dwr_unimpaired/README.md), which also describes
the report and the build.

## Files

| File | What | Source |
|---|---|---|
| `swat_monthly.csv` | `[date, uf, flow_taf]`, the SWAT rim simulation, 18 subbasins × 1116 month-end dates (1921-10-31 to 2014-09-30) = 20088 rows | Appendix C, as published |
| `delta_monthly.csv` | `[date, series, flow_taf]`, six derived totals: `SAC_VALLEY_OUTFLOW` (B-25), `EASTSIDE_OUTFLOW` (B-26), `SJ_VALLEY_OUTFLOW` (B-27), `DELTA_INFLOW` (B-28), `DELTA_OUTFLOW` (B-29), `DELTA_NET_USE` (B-30) | Appendix B, as published |

`uf` is the report's subbasin number (UF 1 to 24). The per-subbasin SWAT flags `has_swat`,
`swat_scale_appendix_d` and `swat_partial` are columns of
`data/targets/dwr_unimpaired/uf_locations.csv`.

## How it is built

Both files are written by `data/targets/dwr_unimpaired/dwr_unimpaired.py` (`sacsma`
environment, needs `pypdf`), in the same run that writes the unimpaired tables. The script
parses the text layer of the report PDF. The PDF is not redistributed; pass it with `--pdf`.

```bash
python data/targets/dwr_unimpaired/dwr_unimpaired.py --pdf <report.pdf>
python data/targets/dwr_unimpaired/dwr_unimpaired.py --verify
```

## Checks

- Completeness: 30 + 18 + 18 tables (Appendix B, C, D) × 93 water years.
- Identities of the totals, within rounding: B-25 = sum of UF 1–11, B-26 = sum of UF 12–15,
  B-27 = sum of UF 16–24, B-28 = B-25 + B-26 + B-27.
- Appendix C against Appendix D, which independently publishes simulated minus unimpaired
  (gate: `D == k·C − B` to ≤ 1.5 TAF, with `k` asserted against its expected value). 17 of the
  18 tables reconcile to ±1.04 TAF, the rounding floor (D is integer, C carries one decimal):
  13 exactly, and 5 after one constant factor (UF 4 ×1.0647, UF 9 ×0.9107, UF 10 ×1.0228,
  UF 15 ×0.9199, UF 22 ×0.9460). Appendix D is not stored; it is exactly derivable.
- `--verify` prints the SWAT fit to the unimpaired series. Over the 16 full-basis subbasins it
  runs 0.97–1.38× the unimpaired (monthly r 0.84–0.96). The widest gaps sit on the
  weakest-calibrated models (Chowchilla 1.38× at NSE 0.76, Fresno 1.37× at NSE 0.71), not on
  the most developed watersheds.

## Know before using

- At the rim, natural and unimpaired are the same quantity. Appendix C is not a different
  physical variable from Appendix B. The SWAT models were calibrated to the unimpaired series,
  so `swat_monthly.csv` is a model estimate of what
  `data/targets/dwr_unimpaired/uf_monthly.csv` measures, and the two differ by calibration
  residual. The report's natural flow departs from unimpaired only on the valley floor, where
  C2VSim adds riparian and wetland ET, stream and groundwater interaction and bank overflow
  (natural Delta inflow 21,533 against unimpaired 29,003 TAF/yr, Table 5-2).
- The SWAT runs: 23 SWAT2009 models, daily, on a 30 m DEM with 2001 USGS land use and STATSGO
  soils, driven by Hamlet and Lettenmaier (2005) 1/8° data extended with 4 km PRISM. They were
  calibrated and judged monthly against the unimpaired series, so they are model fits to
  Appendix B, not an independent observation of it. Reported skill (report Tables A-1, A-2):
  NSE 0.67–0.91, R² 0.68–0.91, weakest on the minor streams and the Tulare basin.
- Caution: `swat_monthly.csv` stores Appendix C as published. For UF 4, 9, 10, 15 and 22,
  Appendix D differences a scaled Appendix C. The report's stated mechanism is an area-ratio
  factor for small local drainage areas between a SWAT watershed outlet and its C2VSim stream
  inflow node (ch. 4). The report's Table 5-1 reproduces the scaled value for UF 9, 10, 15 and
  22 but the raw value for UF 4, so the basis is not uniform across the report.
  `swat_scale_appendix_d` carries the factor, so either basis is recoverable.
- Caution: two tables are not on the full-subbasin basis (flag `swat_partial`). Do not compare
  their level with `uf_monthly.csv`.
  - UF 5: table C-4 is captioned "Sacramento Valley West Side Minor Streams (Thomes and Elder
    Creeks only)". It is the one table that reconciles with Appendix D at no constant factor
    (implied ratio 1.22–2.00, volume ratio 0.64).
  - UF 7: table C-5 averages 1169 TAF/yr against 1410 in the report's own Table 5-1 for the
    same subbasin, a 17 % gap. The two are different aggregations of the east-side creek models
    (Table 5-1 sums Mill, Deer, Big Chico and a "Butte and Chico" node). C and D are mutually
    consistent; both fall short of the subbasin.
- Six subbasins have no SWAT table: the two valley floors (UF 1, 17), the San Joaquin
  east-side minor streams (UF 12), Tulare Lake Basin outflow and the San Joaquin west-side
  minor streams (UF 23, 24), and UF 6. The SWAT model there is Sacramento River at Shasta, not
  the larger Red Bluff subbasin.
- Published defects, stored verbatim:
  - Only months are stored. The printed annual totals are rounded independently of the months,
    so months need not sum to them (in Appendix B, 1085 of 2790 rows differ by ≤ 3 TAF).
  - Tables C-5 (UF 7) and C-17 (UF 21) have a broken Total column: every row repeats that
    row's October value. Table D-8 (UF 10) WY1922 drops a minus sign (the total prints `119`
    where its months sum to −120). In all three the monthly values are sound; the Appendix D
    reconciliation confirms them. The ingest gate whitelists these rows.
  - Caution: table B-29 (Delta Unimpaired Total Outflow, series `DELTA_OUTFLOW`) is corrupt in
    WY2014. All 12 months print as `396`, summing to 4752 against a printed total of 10879. Do
    not use that water year.
- The six totals belong to the unimpaired appendix (B). They are not the report's natural-flow
  estimate.

## Read by

No module of the `sacsma` package reads these two files. `dwr_unimpaired.py --verify` reads
`swat_monthly.csv` back to print the comparison above.
