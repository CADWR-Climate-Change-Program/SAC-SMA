# Appendix B — Figure compendium

The complete canonical figure set from the modeling repository, grouped by theme. All
figures are regenerated programmatically from the archived runs; the source CSVs behind
each figure are listed in Appendix C.

## B.1 Per-set calibration/validation skill

```{figure} ../artifacts/cdec15/figures/skill_summary.png
:width: 6in

B.1.1. CDEC15 (daily vs CDEC FNF): KGE and volume bias, calibration vs validation, basins north→south.
```

```{figure} ../artifacts/calsim/9unimp/figures/skill_summary.png
:width: 6in

B.1.2. Unimpaired9 (monthly vs unimpaired FNF).
```

```{figure} ../artifacts/calsim/11obs/figures/skill_summary.png
:width: 6in

B.1.3. Observed11 (monthly vs unimpaired gauge FNF).
```

```{figure} ../artifacts/calsim/12rim/figures/skill_summary.png
:width: 6in

B.1.4. Rim12 (monthly vs reservoir-inflow series).
```

```{figure} ../artifacts/cdec15/figures/skill_summary_calsim3.png
:width: 6in

B.1.5. CDEC15 re-scored against CalSim3 FNF on the same cal/val windows (the "honest bias" view; note the SHA/BND rim under-run).
```

```{figure} ../artifacts/cdec15/figures/parity_vs_matlab.png
:width: 6in

B.1.6. Python-port parity vs archived MATLAB simulations, CDEC15 (KGE ≈ 1.0 at every basin; the frozen-physics regression baseline).
```

## B.2 Cross-comparison vs VIC and CalSim3 (WY1950–2019)

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_kge.png
:width: 6.2in

B.2.1. Basin-level monthly KGE vs CalSim3 anchor: SAC-SMA vs VIC dumbbells (main-text Figure 1).
```

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_pbias.png
:width: 6.2in

B.2.2. Same, volume bias.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_nse.png
:width: 6.2in

B.2.3. Same, NSE.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_kge_pre1950.png
:width: 6.2in

B.2.4. Anchor KGE, pre-1950 window only.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_kge_post1950.png
:width: 6.2in

B.2.5. Anchor KGE, post-1950 window only.
```

```{figure} ../artifacts/calsim/compare/figures/calsim_sacsma_map_kge.png
:width: 5.2in

B.2.6. SAC-SMA composite basin map, monthly KGE vs CalSim3.
```

```{figure} ../artifacts/calsim/compare/figures/calsim_sacsma_map_pbias.png
:width: 5.2in

B.2.7. SAC-SMA composite basin map, volume bias.
```

```{figure} ../artifacts/calsim/compare/figures/calsim_vic_map_kge.png
:width: 5.2in

B.2.8. VIC benchmark basin map, monthly KGE.
```

```{figure} ../artifacts/calsim/compare/figures/calsim_sacsma_minus_vic_kge.png
:width: 5.2in

B.2.9. SAC-SMA minus VIC KGE difference map (main-text Figure 2).
```

```{figure} ../artifacts/calsim/compare/figures/calsim_sacsma_minus_vic_pbias.png
:width: 5.2in

B.2.10. SAC-SMA minus VIC volume-bias difference map.
```

```{figure} ../artifacts/calsim/compare/figures/rolling_skill_30yr.png
:width: 6.2in

B.2.11. 30-year rolling skill (KGE/NSE/bias/seasonal mismatch), anchor sets vs VIC.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_scatter.png
:width: 5in

B.2.12. Anchor scatter: SAC-SMA vs VIC KGE per basin.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_hydrographs.png
:width: 6.2in

B.2.13. Monthly hydrographs at the eight main river indices.
```

```{figure} ../artifacts/calsim/compare/figures/main_river_climatology.png
:width: 6.2in

B.2.14. Main-river mean-monthly climatology: SAC-SMA (Observed11) vs VIC vs CalSim3.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_pbias_vs_seasonal_11obs.png
:width: 5in

B.2.15. Volume bias vs seasonal mismatch, Observed11.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_pbias_vs_seasonal_9unimp.png
:width: 5in

B.2.16. Volume bias vs seasonal mismatch, Unimpaired9.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_screened_vs_full.png
:width: 6.2in

B.2.17. Screened vs full-footprint anchor scores (differs only at SHA, BND, SNS, Chowchilla).
```

## B.3 Footprint and input-attribute maps

```{figure} ../artifacts/calsim/compare/figures/shasta_footprint_panels.png
:width: 6.2in

B.3.1. Shasta footprint panels: the calibrated HRU footprint vs the CalSim3 catchment, including the endorheic Goose Lake block.
```

```{figure} ../artifacts/calsim/compare/figures/sns_footprint_panels.png
:width: 6.2in

B.3.2. Stanislaus (SNS) footprint panels (screened: delineation over-reach).
```

```{figure} ../artifacts/calsim/compare/figures/chowchilla_footprint_panels.png
:width: 6.2in

B.3.3. Chowchilla footprint panels (screened).
```

```{figure} ../artifacts/calsim/compare/figures/tnl_footprint_panels.png
:width: 6.2in

B.3.4. Trinity (TNL) footprint panels (unscreened; shown for contrast).
```

```{figure} ../artifacts/calsim/compare/figures/fresno_footprint_panels.png
:width: 6.2in

B.3.5. Fresno footprint panels (unscreened).
```

```{figure} ../artifacts/calsim/compare/figures/hru_veg_15cdec.png
:width: 5.2in

B.3.6. HRU vegetation classes, CDEC15 domain.
```

```{figure} ../artifacts/calsim/compare/figures/hru_soil_15cdec.png
:width: 5.2in

B.3.7. HRU soil classes, CDEC15 domain.
```

```{figure} ../artifacts/calsim/compare/figures/hru_kpet_15cdec.png
:width: 5.2in

B.3.8. Calibrated Hamon K$_{pet}$ field, CDEC15 (single-valued per soil class).
```

```{figure} ../artifacts/calsim/compare/figures/hru_kpet_calsim.png
:width: 5.2in

B.3.9. Calibrated K$_{pet}$, CalLite per-watershed sets (uniform per basin).
```

## B.4 Temperature-detrending sensitivity (WGEN Product A)

```{figure} ../artifacts/calsim/forcing_compare/figures/wgen_volume_by_period.png
:width: 6.2in

B.4.1. SAC-SMA per-watershed volume difference by period (Detrended − Baseline).
```

```{figure} ../artifacts/calsim/forcing_compare/figures/wgen_annual_diff.png
:width: 6.2in

B.4.2. SAC-SMA 5-yr rolling water-year volume difference.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/wgen_regime_by_period_9unimp.png
:width: 6.2in

B.4.3. SAC-SMA mean-monthly regime under both forcings, Unimpaired9.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/wgen_regime_by_period_11obs.png
:width: 6.2in

B.4.4. Same, Observed11.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_wgen_volume_by_period.png
:width: 6.2in

B.4.5. VIC per-watershed volume difference by period.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_wgen_annual_diff.png
:width: 6.2in

B.4.6. VIC 5-yr rolling volume difference.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_wgen_regime_by_period_9unimp.png
:width: 6.2in

B.4.7. VIC mean-monthly regime, Unimpaired9.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_wgen_regime_by_period_11obs.png
:width: 6.2in

B.4.8. VIC mean-monthly regime, Observed11.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/agg_wgen_annual_diff.png
:width: 6.2in

B.4.9. Aggregate SAC-SMA vs VIC volume response (main-text Figure 3).
```

## B.5 Split vs unsplit precipitation sensitivity (Historical LTO)

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_volume_by_period.png
:width: 6.2in

B.5.1. SAC-SMA per-watershed volume difference by period (Unsplit − Split).
```

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_annual_diff.png
:width: 6.2in

B.5.2. SAC-SMA 5-yr rolling volume difference.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_regime_by_period_9unimp.png
:width: 6.2in

B.5.3. SAC-SMA mean-monthly regime, Unimpaired9.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_regime_by_period_11obs.png
:width: 6.2in

B.5.4. Same, Observed11.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_split_unsplit_volume_by_period.png
:width: 6.2in

B.5.5. VIC per-watershed volume difference by period.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_split_unsplit_annual_diff.png
:width: 6.2in

B.5.6. VIC 5-yr rolling volume difference.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_split_unsplit_regime_by_period_9unimp.png
:width: 6.2in

B.5.7. VIC mean-monthly regime, Unimpaired9.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_split_unsplit_regime_by_period_11obs.png
:width: 6.2in

B.5.8. VIC mean-monthly regime, Observed11.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/agg_split_unsplit_annual_diff.png
:width: 6.2in

B.5.9. Aggregate SAC-SMA vs VIC volume response.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_skill_boxplot.png
:width: 6.2in

B.5.10. Anchor skill under unsplit vs split forcing, pooled boxplots by period (main-text Figure 4).
```

## B.6 dPL runs (Part II)

```{figure} ../artifacts/dpl/hamon_dense/figures/skill_summary.png
:width: 6in

B.6.1. dPL `hamon_dense` per-basin skill (frozen scoring, daily CDEC FNF).
```

```{figure} ../artifacts/dpl/hamon/figures/skill_summary.png
:width: 6in

B.6.2. dPL `hamon` (native grid + CalSim3 footprint) per-basin skill.
```

```{figure} ../artifacts/dpl/pt/figures/skill_summary.png
:width: 6in

B.6.3. dPL `pt` (Priestley–Taylor) per-basin skill.
```

```{figure} ../artifacts/dpl/noah/figures/skill_summary.png
:width: 6in

B.6.4. dPL `noah` (Noah-lite ET) per-basin skill.
```

```{figure} ../artifacts/dpl/compare_val_kge.png
:width: 6in

B.6.5. Per-basin validation KGE across the program: GA optimum → dPL (`hamon_dense`) → the two hybrid ensembles.
```

```{figure} ../artifacts/dpl/fidelity/figures/fidelity_benchmark.png
:width: 6in

B.6.6. Differentiable-vs-frozen fidelity benchmark (archived GA parameters through both pipelines).
```

```{figure} ../artifacts/dpl/figures/cdec15_climatology_a.png
:width: 6.2in

B.6.7. Out-of-calibration mean-monthly climatology vs CalSim3 FNF: GA optimum vs dPL `hamon_dense`.
```

```{figure} ../artifacts/dpl/figures/cdec15_climatology_b.png
:width: 6.2in

B.6.8. Same: `hamon_dense` vs `hamon` (fine-HRU → native grid + footprint).
```

```{figure} ../artifacts/dpl/figures/cdec15_climatology_c.png
:width: 6.2in

B.6.9. Same: `hamon` vs `pt` (Hamon → Priestley–Taylor).
```

```{figure} ../artifacts/dpl/figures/cdec15_climatology_d.png
:width: 6.2in

B.6.10. Same: `pt` vs `noah` (E1–E5 cascade → Noah-lite ET).
```

```{figure} ../artifacts/dpl/figures/cdec15_climatology_e.png
:width: 6.2in

B.6.11. Same: `noah` → the two LSTM hybrid ensembles (main-text Figure 5).
```

```{figure} ../artifacts/dpl/figures/cdec15_climatology_summary.png
:width: 6.2in

B.6.12. Climatology summary: per-basin KGE, |bias|, seasonally misplaced volume across the lineage.
```

```{figure} ../artifacts/dpl/figures/cdec15_climatology_summary_agg.png
:width: 6.2in

B.6.13. Same, basin-aggregated.
```

```{figure} ../artifacts/dpl/hybrid_pet_dt/seasonal_compare_hybrid_pet_dt.png
:width: 6.2in

B.6.14. Seasonal-timing scoreboard: per-basin validation KGE, seasonal mismatch, and CalSim3 monthly KGE for `noah` / `hybrid` / `hybrid_pet_dt`.
```

```{figure} ../artifacts/dpl/figures/cdec15_forcing_sensitivity_rolling.png
:width: 6.2in

B.6.15. Temperature-detrending response of the dPL lineage (`hamon`/`pt`/`noah` and both hybrid ensembles): 10-yr rolling aggregate flow response. The unconstrained basic `hybrid` versus the physics-tracking `hybrid_pet_dt` is the held-out check of the temperature-consistency loss.
```

```{figure} ../artifacts/dpl/figures/cdec15_forcing_sensitivity_monthly_pre1950.png
:width: 6.2in

B.6.16. Same, pre-1950 mean-monthly regime response.
```

```{figure} ../artifacts/dpl/hybrid_progression.png
:width: 6.2in

B.6.17. Hybrid progression exhibit: pooled skill and +2 °C response for `hybrid` → +PET input → `hybrid_pet_dt` (main-text Figure 6).
```

## B.7 Outlier-basin diagnostics

```{figure} ../artifacts/cdec15/figures/SHA_diagnostics.png
:width: 6.2in

B.7.1. SHA daily diagnostics (CDEC15 pooled set; the rim under-run).
```

```{figure} ../artifacts/cdec15/figures/NHG_diagnostics.png
:width: 6.2in

B.7.2. NHG daily diagnostics (structural positive bias; snow-free basin).
```

```{figure} ../artifacts/calsim/11obs/figures/TNL_diagnostics.png
:width: 6.2in

B.7.3. TNL monthly diagnostics (Observed11 validation outlier).
```

```{figure} ../artifacts/calsim/9unimp/figures/FresnoRiver_diagnostics.png
:width: 6.2in

B.7.4. Fresno River monthly diagnostics (Unimpaired9).
```

```{figure} ../artifacts/calsim/9unimp/figures/CosumnesRiver_diagnostics.png
:width: 6.2in

B.7.5. Cosumnes River monthly diagnostics (Unimpaired9 strong case).
```

```{figure} ../artifacts/calsim/12rim/figures/TRINI_diagnostics.png
:width: 6.2in

B.7.6. TRINI monthly diagnostics (Rim12; Trinity forcing-lineage sensitivity).
```
