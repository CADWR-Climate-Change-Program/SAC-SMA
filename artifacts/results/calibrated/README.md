# `artifacts/results/calibrated/`: the calibrated SAC-SMA

Results of the archived GA calibrations run through the reference model
(`sacsma.model.run_basin`). Method and scores: [Calibrated SAC-SMA](../../../docs/calibrated_sacsma.md);
the scoring rules: [Conventions](../../../docs/conventions.md). Everything here redraws on CPU in
minutes.

| Folder | What | Command |
|---|---|---|
| `15cdec/`, `11obs/`, `9unimp/`, `12rim/` | Each calibration set against its own target: daily CDEC gauge for `15cdec`, monthly full natural flow for the others | `sacsma plots --domain <set>` (`--fnf-check` adds the CalSim3 scoring of `15cdec`) |
| `calsim3/` | Every set and VIC against CalSim3: watershed (anchor) and arc scores, maps, rolling skill, the per-arc quantile mapping | `sacsma calsim` |
| `vic_bcm/` | SAC-SMA, VIC and BCM v8 on one climate against one target, WY1989–2018 | `sacsma calsim --sacsma-vic-bcm` |
| `footprints/` | How each watershed's units sit on its CalSim3 catchment, and the HRU attribute maps | `sacsma calsim` |
| `forcing/` | The effect of the forcing product on both SAC-SMA and VIC, and the daily runs it compares | `sacsma calsim --forcing-compare` |

## Each set (`15cdec/`, `11obs/`, `9unimp/`, `12rim/`)

| File | What |
|---|---|
| `metrics.csv` | Per watershed: calibration and validation KGE, NSE, percent bias, r, mean flow |
| `metrics_calsim3.csv` | The same run scored against CalSim3's unimpaired flow on the same windows (`11obs`, `9unimp`; `15cdec` with `--fnf-check`) |
| `figures/<watershed>_diagnostics.png` | Simulated against observed flow with calibration and validation skill, and the mean-monthly regimes of both periods (`_calsim3`: against CalSim3) |
| `figures/skill_summary.png` | KGE and percent bias across the set's watersheds, north to south |
| `figures/parity_vs_matlab.png` | The reference model against the original MATLAB simulation |

## `calsim3/`

| File | What |
|---|---|
| `anchor_metrics.csv`, `anchor_monthly.csv` | Each watershed's total against its CalSim3 reference (FLOW-UNIMPAIRED for a rim system, else the sum of its INFLOW arcs; `ref_kind`), Observed11 and Unimpaired9 |
| `anchor_metrics_15cdec.csv`, `anchor_monthly_15cdec.csv` | The same for CDEC15, kept apart |
| `anchor_metrics_full.csv`, `anchor_monthly_full.csv`, `anchor_screened_vs_full.csv` | Without the footprint screening of SHA, BND, SNS and Chowchilla, and the difference |
| `anchor_metrics_by_period.csv` | The anchor scores before and from WY1950 |
| `calset_metrics.csv`, `monthly_calsets.csv`, `coverage_by_set.csv` | Every CalSim3 arc scored on its own, with each set's unit coverage of the arc |
| `subarc_qmap_<set>.csv` (and `_vic`), `subarc_validation_metrics.csv` | The per-arc quantile mapping, fitted on WY1922–1971 and scored on WY1972–2018 |
| `rolling_skill_30yr.csv`, `rolling_skill_basin_30yr.csv` | 30-year rolling skill at the anchors |
| `target_vs_calsim3.csv` | How far each set's calibration target sits from CalSim3 |
| `basin_map_metrics.csv`, `vic_full_metrics.csv` | The values behind the maps; VIC on every arc |
| `figures/` | Anchor dumbbells (`anchor_skill_*`, also by period), hydrographs and regimes, skill maps of SAC-SMA, VIC and their difference, coverage maps, rolling skill |

## `vic_bcm/`

`sacsma_vic_bcm_monthly.csv` (all years, so the window can be re-cut), `sacsma_vic_bcm_metrics.csv`
(per watershed and model, on identical months), `sacsma_vic_bcm_summary.csv` (pooled), and the
skill, regime and summary figures.

## `footprints/`

`figures/<watershed>_footprint_panels.png` (the screening method for SHA, SNS, Chowchilla,
Trinity and Fresno), `figures/hru_*` (vegetation and soil classes and `Kpet` of the units), and
the two tables behind the `Kpet` maps (`hru_veg_kpet_15cdec.csv`, `hru_kpet_by_soil_15cdec.csv`).

## `forcing/`

`figures/` holds each product against the Livneh baseline on both models, split at WY1950;
`split_unsplit_anchor_skill.csv` scores both precipitation lineages against CalSim3.
`wgen_product_a/` and `historical_lto/` hold the daily runs of the three CalLite sets under
those products (`sim_daily_<set>.csv`, `[date, basin, flow]` in mm/day), written by the
comparison when they are missing.
