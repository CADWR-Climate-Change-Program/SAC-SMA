# `artifacts/product/15cdec/`: daily flow of the 15 CDEC watersheds

The daily flow of the 15 CDEC reservoir watersheds from the archived GA calibration of `15cdec`
([`models/15cdec/`](../../models/README.md)) run by the reference model. Method and scores:
[Calibrated SAC-SMA](../../../docs/calibrated_sacsma.md).

The calibration's units are study-specific points off the 1/16° grid, and only the historical
Livneh forcing has been interpolated to them, so there is one folder:
`historical_livneh_unsplit/`.

The model runs from a cold start on 1915-01-01, the first day of the forcing record; the series
starts in October 1921, like the CalLite files, and ends in September 2018.

| File | What |
|---|---|
| `historical_livneh_unsplit/flow_daily.csv` | `date, basin, flow_mm, flow_cfs`: one row per watershed and day. `flow_mm` is mm/day over the watershed; `flow_cfs` converts it over the published drainage area of [`data/inputs/domains/15cdec/basin_area.csv`](../../../data/inputs/domains/README.md). |

```bash
sacsma product 15cdec          # CPU, under a minute
sacsma verify product          # the file is written again and compared
```
