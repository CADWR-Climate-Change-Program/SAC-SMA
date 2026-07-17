# Appendix C — Data and artifact manifest

All paths are relative to the modeling repository root (`sac-sma`). This appendix names
the authoritative source behind every number and figure in this document.

## C.1 Input data

| Item | Path | Notes |
|---|---|---|
| Baseline forcing (unsplit) | `data/cdec15/forcing/historical_livneh_unsplit.nc` (fine off-grid HRUs); `data/region/forcing/historical_livneh_unsplit.nc` (unified 1/16° region store read by every grid domain) | daily 1915–2018, grid-cell keyed |
| WGEN Product A forcing | `data/region/forcing/wgen_product_a.nc` | precip identical to unsplit; temperature detrended to 1991–2020 |
| Historical LTO forcing | `data/region/forcing/historical_lto.nc` | split-lineage Livneh precip, daily 1915–2021 |
| Satellite ET/SWE observations | `data/region/et_obs/`, `data/region/swe_obs/` | product ensembles at the 4,410-cell region grid; the Part II §9 observation losses |
| Region grid + precip audit | `data/region/grid_cells.csv`, `data/region/prcp_x10_artifacts.csv` | the unified-store cell index and the misplaced-decimal correction table |
| HRU tables | `data/cdec15/hruinfo.csv`, `data/calsim/hruinfo_<domain>.csv` | basin, cell key, area weight, elevation, flow length, soil/veg class |
| Archived GA optima | `data/cdec15/ga_optimum.csv` (pooled), `data/calsim/ga_optimum_<domain>.csv` (per-watershed) | 31 parameters per HRU row |
| Daily calibration target | `data/cdec15/gage.csv` + `data/cdec15/basin_area.csv` | CDEC full natural flow |
| Monthly calibration targets | `data/calsim/fnf_<domain>_monthly.csv`, `calib_<domain>_monthly.csv` | per-basin calibration windows carried in-table |
| MATLAB parity reference | `data/*/simflow*.csv` | the frozen-physics regression baseline |
| CalSim3 references | `data/calsim/calsim3_inflow_monthly.csv`, `calsim_unimpaired_monthly.csv` | INFLOW arcs; FLOW-UNIMPAIRED rim series |
| VIC benchmark | `data/calsim/vic_routed_monthly.csv` (+ `_wgen_product_a`, `_historical_lto`) | `no_gooselake` series at I_SHSTA / 8RI_SRBB |
| Arc crosswalk | `data/calsim/calsim_crosswalk.csv` | hand-maintained; maps every CalSim3 arc to rim system, VIC basin, and each SAC set |
| CalSim catchment areas | `data/calsim/basin_area_<set>_calsim.csv` | canonical areas used by the anchor scoring |
| Provenance manifest | `data/INVENTORY.md` | full data lineage documentation |

*Table C.1. Input data.*

## C.2 Part I result artifacts

| Item | Path |
|---|---|
| Per-set skill metrics | `artifacts/cdec15/metrics_15cdec.csv`; `artifacts/calsim/<domain>/metrics_<domain>.csv` (+ `_calsim3` variants) |
| Anchor cross-compare | `artifacts/calsim/compare/anchor_metrics.csv`, `anchor_metrics_15cdec.csv`, `anchor_metrics_by_period.csv`, `anchor_screened_vs_full.csv` |
| Target-vs-CalSim3 floor | `artifacts/calsim/compare/target_vs_calsim3.csv` |
| Per-arc metrics + QMAP validation | `artifacts/calsim/compare/calset_metrics.csv`, `subarc_validation_metrics.csv`, `subarc_qmap_*.csv` |
| Rolling skill | `artifacts/calsim/compare/rolling_skill_30yr.csv` (+ per-basin) |
| Forcing-sensitivity skill | `artifacts/calsim/forcing_compare/split_unsplit_anchor_skill.csv` |
| Alternate-forcing runs | `artifacts/calsim/wgen_product_a/flow_daily_<domain>.csv`, `artifacts/calsim/historical_lto/flow_daily_<domain>.csv` |
| Conventions documentation | `artifacts/README.md` |

*Table C.2. Part I result artifacts.*

Note: the per-basin WGEN/LTO volume-difference percentages of Part I §6–7 are computed
inside the forcing-comparison figures; `split_unsplit_anchor_skill.csv` is the one
standalone quantitative table.

## C.3 Part II (dPL) artifacts

| Item | Path |
|---|---|
| Run log / track record | `artifacts/dpl/RUNS.md` |
| Canonical physics runs | `artifacts/dpl/{hamon_dense, hamon, pt, noah}/`, each with `metrics_<run>.csv`, `params_dpl.csv`, `figures/` (Noah also `params_canopy.csv`) |
| Hybrid physics channel + teacher | `artifacts/dpl/noah/daily_sim_noah_torch.csv` (the ensembles' physics input channel, exported from the differentiable pipeline) + `daily_sim_noah_plus2C.csv` / `metrics_noah_plus2C.csv` (the +2 °C teacher) |
| Hybrid ensembles | `artifacts/dpl/hybrid/`, `artifacts/dpl/hybrid_pet_dt/` (ensemble-mean `metrics_hybrid.csv` + per-seed checkpoints) |
| Seasonal-timing scoreboard | `artifacts/dpl/hybrid_pet_dt/seasonal_compare_hybrid_pet_dt.csv` (per-basin val KGE, seasonal mismatch, CalSim3 KGE for `noah`/hybrids) |
| GA → dPL → hybrid comparison | `artifacts/dpl/compare_ga_dpl_hybrid.csv` + `compare_val_kge.png` |
| Hybrid progression exhibit | `artifacts/dpl/hybrid_progression.csv` + `.png` (three-arm skill vs +2 °C response) |
| Fidelity benchmark | `artifacts/dpl/fidelity/fidelity_benchmark.csv` + figure |
| Out-of-calibration climatology + dPL climate response | `artifacts/dpl/figures/cdec15_climatology_*.png`, `cdec15_forcing_sensitivity_*.png` |
| dPL implementation | `sacsma/dpl/` (training system, incl. `hybrid/` and `seasonal_compare.py`); frozen scoring mirrors `sacsma/pet_pt.py`, `sacsma/sma_noah_lite.py` |

*Table C.3. Part II (dPL) artifacts.*

## C.4 Reproduction commands

```
sacsma run <BASIN|ALL> [--domain 9unimp|11obs|12rim] [--forcing <product>]
sacsma plots --domain <set>          # per-set diagnostics
sacsma calsim                        # cross-compare -> artifacts/calsim/compare/
```

dPL training and evaluation run through `sacsma.dpl` (see `artifacts/dpl/RUNS.md` for
the exact command line of every canonical run); `sacsma dpl evaluate <checkpoint>
--temp-delta 2.0` dumps the temperature-perturbed teacher simulation used by the
hybrid's temperature-consistency loss. The cross-model climatology and
temperature-sensitivity figures regenerate through `sacsma dpl climatology` and
`sacsma.dpl.forcing_sensitivity` into `artifacts/dpl/figures/`.
