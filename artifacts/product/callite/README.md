# `artifacts/product/callite/`: the CalLite inflow files

The monthly inflow of the three CalLite watershed sets, in the three files the original study's
MATLAB wrapper wrote for CalLite, from the archived GA calibrations
([`models/callite/`](../../models/README.md)) run by the reference model. Method and scores of
the calibrations: [Calibrated SAC-SMA](../../../docs/calibrated_sacsma.md).

One folder per forcing:

| Folder | Forcing |
|---|---|
| `historical_livneh_unsplit/` | Livneh, unsplit precipitation |
| `wgen_product_a/` | WGEN Product A scenario 1: the same precipitation, temperature detrended to a 1991–2020 baseline |
| `wgen_product_a_s12/` | WGEN Product A scenario 12: scenario 1 + 2 °C, extreme-tail rate 7 %/°C, wet-day mean unchanged |

Each model runs from a cold start on 1915-01-01, the first day of the forcing record, as the
original study ran it. The files start in October 1921, CalLite's first month, which leaves
more than six years of run-up (the wrapper kept five), and end in September 2018, the last
complete water year of the forcing: 1,164 months.

| File | Set | Columns | Unit |
|---|---|---|---|
| `SACSMA_Inflow_CalLite_9.txt` | `9unimp` | `Year, Month, BearRiver, CacheCreek, CalaverasRiver, ChowchillaRiver, CosumnesRiver, FresnoRiver, MokelumneRiver, PutahCreek, StonyCreek` | monthly total, mm |
| `SACSMA_Inflow_CalLite_11.txt` | `11obs` | `Year, Month, AMF, BLB, BND, FTO, MRC, SHA, SJF, SNS, TLG, TNL, YRS` | monthly total, mm |
| `SACSMA_Inflow_CalLite_12.txt` | `12rim` | the twelve rim inflows, below | monthly mean CFS, or monthly total TAF |

`Year` is the water year and `Month` the calendar month (`1922, 10` is October 1921).

`SACSMA_Inflow_CalLite_12.txt` has three header lines and no date columns: the end of the first
month (`31OCT1921 2400`) once per column, the CalLite DSS path of each column
(`/CALLITE/<node>/FLOW-INFLOW//1MON/2020D09E/`) and its unit. The columns, in order:

| 12rim watershed | CalLite node | Unit |
|---|---|---|
| `FOL_I` | `I_FOLSM` | CFS |
| `LK_MC` | `I_MCLRE` | TAF |
| `N_MEL` | `I_MELON` | TAF |
| `MILLE` | `I_MLRTN` | CFS |
| `PRD_C` | `I_MOKELUMNE` | CFS |
| `N_HOG` | `I_NHGAN` | TAF |
| `OROVI` | `I_OROVL` | CFS |
| `DPR_I` | `I_PEDRO` | TAF |
| `SHAST` | `I_SHSTA` | CFS |
| `TRINI` | `I_TRNTY` | CFS |
| `SMART` | `I_YUBA` | CFS |
| `WKYTN` | `I_WKYTN` | CFS |

The depths are converted over the drainage areas of
[`data/inputs/domains/12rim/basin_area.csv`](../../../data/inputs/domains/README.md), the areas the
wrapper used. TAF is the exact volume of the month (mean CFS × days × 1.9834711 / 1000); the
wrapper multiplied the mean CFS by 0.06, which is 2.4 % low in a 31-day month, 0.8 % high in a
30-day month and 8 % high in a 28-day February.

The same watershed in two sets is two calibrations against two targets: `12rim` was fitted
directly to CalLite's impaired inflows, `9unimp` and `11obs` to unimpaired flow, so over
WY1922–2018 under Livneh `PRD_C` carries about 40 % of `MokelumneRiver`'s volume and `SMART`
about 77 % of `YRS`'s. The areas match those of the same watersheds in `11obs` and `9unimp`
except at `TRINI` (676 mi², against 719 for `TNL`), with `PRD_C` and `N_HOG` within 0.3 mi².

```bash
sacsma product callite [--forcing <forcing> ...]    # CPU, about a minute for the three forcings
sacsma verify product                               # each file is written again and compared
```
