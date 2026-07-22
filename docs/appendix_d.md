# Appendix D: Part II (dPL) figures

The cross-model figure set behind Part II: the fidelity benchmark, the out-of-calibration climatology ladder, the seasonal-timing scoreboard, and the climate-response exhibits. Per-run skill bars and per-basin daily diagnostics are not reproduced here; they live with each run under `artifacts/dpl/<run>/figures/`. The source CSVs behind each figure are listed in Appendix E.

```{figure} ../artifacts/dpl/noah/fidelity/figures/fidelity_benchmark.png
:width: 6in

D.1. Differentiable-vs-frozen fidelity benchmark (archived GA parameters through both pipelines).
```

```{figure} ../artifacts/dpl/figures/climatology_a.png
:width: 6.2in

D.2. Out-of-calibration mean-monthly climatology vs CalSim3 FNF: GA optimum vs dPL `hamon_dense`.
```

```{figure} ../artifacts/dpl/figures/climatology_b.png
:width: 6.2in

D.3. Same: `hamon_dense` vs `hamon` (fine-HRU → native grid + footprint).
```

```{figure} ../artifacts/dpl/figures/climatology_c.png
:width: 6.2in

D.4. Same: `hamon` vs `pt` (Hamon → Priestley–Taylor).
```

```{figure} ../artifacts/dpl/figures/climatology_d.png
:width: 6.2in

D.5. Same: `pt` vs `noah` (E1–E5 cascade → climate-adaptive Noah-lite ET). The climate-frozen predecessor (`superseded/noah_noca`, 0.767/0.799 cal/val KGE) is superseded by this rung's 0.779/0.804.
```

```{figure} ../artifacts/dpl/figures/climatology_e.png
:width: 6.2in

D.6. Same: `noah` → the two LSTM hybrid ensembles (main-text Figure 1).
```

```{figure} ../artifacts/dpl/figures/climatology_summary.png
:width: 6.2in

D.7. Climatology summary: per-basin KGE, |bias|, seasonally misplaced volume across the lineage.
```

```{figure} ../artifacts/dpl/figures/climatology_summary_agg.png
:width: 6.2in

D.8. Same, basin-aggregated.
```

```{figure} ../artifacts/dpl/superseded/hybrid_dt_noca/seasonal_compare_hybrid_dt_noca.png
:width: 6.2in

D.9. Seasonal-timing scoreboard from the frozen-noah predecessor generation: per-basin validation KGE, seasonal mismatch, and CalSim3 monthly KGE for `noah_noca` / `hybrid_noca` / `hybrid_dt_noca`.
```

```{figure} ../artifacts/dpl/figures/forcing_sensitivity_rolling.png
:width: 6.2in

D.10. Temperature-detrending response of the dPL lineage (`hamon`/`pt`/`noah` and both current hybrid ensembles): 10-yr rolling aggregate flow response. The unconstrained `hybrid` versus the response-constrained `hybrid_dt` is the held-out check of the response-consistency loss.
```

```{figure} ../artifacts/dpl/figures/forcing_sensitivity_monthly_pre1950.png
:width: 6.2in

D.11. Same, pre-1950 mean-monthly regime response.
```

```{figure} ../artifacts/dpl/figures/hybrid_summary.png
:width: 6.2in

D.12. Skill and (Δprecip, ΔT) climate-response fidelity across the current hybrid family: `noah` physics, `hybrid`, `hybrid_dt`, and the no-physics `lstm` control (main-text Figure 2).
```

```{figure} ../artifacts/dpl/figures/hybrid_progression.png
:width: 6.2in

D.13. Skill and pooled warming-response curve for the current chain `noah` → `hybrid` → `hybrid_dt`: per-basin validation KGE, and annual runoff %change vs present climate along ΔT (Δprecip = 0). Updates the frozen-noah predecessor generation's progression exhibit (formerly `hybrid_noca` → +PET input → `hybrid_dt_noca`, retired with that generation) to the current family.
```
