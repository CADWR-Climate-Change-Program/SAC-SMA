# Appendix B — Figure compendium

The complete canonical figure set from the modeling repository, grouped by theme. All
figures are regenerated programmatically from the archived runs; the source CSVs behind
each figure are listed in Appendix C. Sections B.2, B.4, and B.5 also carry the full
narrative and tables behind Part I's VIC benchmark, summarized only briefly there.

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

The cross-comparison scores every set on one shared basis, monthly flows against the
CalSim3 anchor references over WY1950–2019 (table below). CDEC15's daily output is
aggregated to monthly here, so all three sets and VIC are directly comparable.

| Set | KGE | VIC KGE | \|pbias\| | VIC \|pbias\| | Seas. mism. | VIC seas. mism. |
|---|---|---|---|---|---|---|
| Observed11 | 0.91 | 0.77 | 4.2% | 8.5% | 3.3% | 7.6% |
| Unimpaired9 | 0.92 | 0.62 | 4.9% | 27.3% | 5.6% | 7.7% |
| CDEC15 | 0.87 | 0.77 | 7.6% | 10.0% | 8.2% | 7.1% |

*Mean basin-level skill against the CalSim3 anchor references, WY1950–2019, monthly.
VIC columns are means over the same basins as each set; \|pbias\| is the mean absolute
per-basin volume bias, and seasonal mismatch is the percentage of annual volume placed
in the wrong month.*

The seasonal-mismatch metric is the percentage of annual volume placed in the wrong
month (the total-variation distance between normalized mean-monthly regimes; Appendix
A.5). It separates timing errors from volume errors in a way KGE alone does not. The
anchor sets misplace 3–6% of annual volume seasonally against VIC's 8%; on the CDEC15
basins the pooled set misplaces 8%, slightly more than VIC on the same basins (7%),
which reflects its reservoir-oriented pooled calibration.

SAC-SMA outperforms VIC at nearly every basin, by +0.15 mean KGE on the rim set and
+0.30 on the creeks (B.2.1). The creek-set gap is wide because the margin is largest
where a complete CalSim hydrology is hardest, in the small southern creeks, where VIC
runs wet. At Fresno River VIC carries a +95% volume bias (KGE −0.18) against SAC-SMA's
+11% (0.87); Cosumnes is +35% versus +1.5%, Chowchilla +41% versus −4%, and Calaveras
+31% versus +5%. Rim basins are closer (at Shasta, SAC-SMA 0.94 and VIC 0.78). The
hardest rim basin for both models is Trinity (SAC-SMA 0.77, VIC 0.57) (B.2.9).

Four method-level findings follow.

- **Basin-level aggregation stabilizes the comparison.** At individual CalSim3 nodes
  the per-catchment median KGE is about 0.67 for the SAC sets (VIC 0.66), because
  sub-arc-scale noise (delineation mismatches, extrapolated small nodes) dominates;
  aggregation to basin anchors cancels most of it. A per-arc quantile-mapping
  correction with mass-balance rescaling (trained WY1922–1971, scored on held-out
  WY1972–2018) lifts the sub-arc median KGE to 0.76–0.86 depending on set, with volume
  bias magnitudes of 4–11%. That correction is the working route from basin-credible
  models to arc-level hydrology.
- **The footprint screening affects only the four screened basins.** Screening moves
  Shasta's anchor bias from −8.9% to +0.1% (the Goose Lake cut) and Chowchilla's from
  −14.3% to −4.3%; every other basin is unchanged.
- **CDEC15's rim bias is real but specific.** Scored against CalSim3, the pooled set
  under-runs the two big Sacramento rim systems by −24% (SHA) and −22% (BND). This is
  the reservoir-calibrated pooled optimum's known weakness, and the reason CDEC15 is
  kept off the official anchor basis. Its Tulare-basin coverage (PNF, TRM, SCC, ISB),
  which no CalLite set provides, is unaffected.
- **Skill is stable in time.** In every 30-year rolling window since 1922, SAC-SMA's
  anchor-set KGE stays within 0.86–0.96 (volume bias within ±8%) and exceeds VIC in
  the same window throughout.

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_kge.png
:width: 6.2in

B.2.1. Basin-level monthly KGE vs CalSim3 anchor: SAC-SMA vs VIC dumbbells.
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

B.2.9. SAC-SMA minus VIC KGE difference map.
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

Because the stochastic pipeline runs on detrended temperature, the response of each
hydrologic model to temperature detrending is itself a model property worth measuring.
The WGEN Product A forcing isolates it, since precipitation is identical to the
baseline and temperature is detrended to 1991–2020 (the early record warmed by up to
about 0.4 °C). Both models were re-run under both forcings with parameters unchanged,
and differences are reported as Detrended − Baseline on the shared 1915–2018 record,
split at WY1950.

**SAC-SMA responds with a volume loss.** Warming the early record raises Hamon PET and
snow-season ET, and the model loses 2–4% of long-term runoff volume (median near −3%
per domain), concentrated where the detrending is largest. Per-basin losses run −9 to
−2.7% before 1950, tapering to about −2% after. Daily flow correlation stays at or
above 0.999, so the response is almost entirely in volume rather than timing.

**VIC responds with a timing shift.** VIC's aggregate volume response is only −2.5 to
+0.7% pre-1950; it mostly shifts snowmelt earlier instead (B.4.9). The contrast
quantifies the liability of a temperature-only PET: any temperature adjustment feeds
directly into ET and volume, whereas an energy-budget model mostly re-times the season.

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

B.4.9. Aggregate SAC-SMA vs VIC volume response.
```

## B.5 Split vs unsplit precipitation sensitivity (Historical LTO)

The CalSim3 LTO study's observed climate carries the pre-correction "split" Livneh
precipitation lineage, whereas the baseline here (and the VIC benchmark's historical
run) uses the {cite:t}`pierce2021` unsplit product. Re-running SAC-SMA under the LTO
forcing (and comparing the VIC parallels) measures how much this lineage choice
matters.

**The difference is concentrated before 1950.** The median absolute water-year volume
difference is about 11% in 1915–1949 versus 1.5–3.2% in 1950–2018. Directionally, the
unsplit product is wetter in the southern and Sierra basins and drier in the northern
Sacramento systems before 1950 (spread up to ±21%), collapsing to within ±2% after 1950
everywhere except **Trinity (+8%)**. Trinity is the one basin where the lineage choice
remains consequential in the modern era, and it is also the weakest anchor basin in the
cross-comparison (B.2).

**On the full record, anchor skill differs little between the two forcings** (median
anchor KGE 0.934 unsplit versus 0.926 split), but the split forcing redistributes
skill. It recovers the weakest pre-1950 cases (Trinity pre-1950 KGE 0.40 → 0.83; Cache
0.63 → 0.78; Stony 0.73 → 0.87) while degrading several basins that were already in
good agreement (Cosumnes 0.97 → 0.87, Calaveras 0.92 → 0.81, Yuba 0.90 → 0.85) (B.5.10).
The practical reading is that pre-1950 disagreements with CalSim3 at a handful of basins
owe as much to the precipitation lineage as to the model, and conclusions drawn from
pre-1950 skill should carry that caveat.

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

B.5.10. Anchor skill under unsplit vs split forcing, pooled boxplots by period.
```

## B.6 dPL runs (Part II)

```{figure} ../artifacts/dpl/superseded/hamon_dense/figures/skill_summary.png
:width: 6in

B.6.1. dPL `hamon_dense` per-basin skill (frozen scoring, daily CDEC FNF); superseded
by `hamon`, retained for lineage.
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

B.6.4. dPL `noah` (climate-adaptive Noah-lite ET) per-basin skill. The climate-frozen
predecessor (`superseded/noah_noca`) scores 0.767/0.799 against this rung's
0.779/0.804.
```

```{figure} ../artifacts/dpl/noah/fidelity/figures/fidelity_benchmark.png
:width: 6in

B.6.5. Differentiable-vs-frozen fidelity benchmark (archived GA parameters through both pipelines).
```

```{figure} ../artifacts/dpl/figures/climatology_a.png
:width: 6.2in

B.6.6. Out-of-calibration mean-monthly climatology vs CalSim3 FNF: GA optimum vs dPL `hamon_dense`.
```

```{figure} ../artifacts/dpl/figures/climatology_b.png
:width: 6.2in

B.6.7. Same: `hamon_dense` vs `hamon` (fine-HRU → native grid + footprint).
```

```{figure} ../artifacts/dpl/figures/climatology_c.png
:width: 6.2in

B.6.8. Same: `hamon` vs `pt` (Hamon → Priestley–Taylor).
```

```{figure} ../artifacts/dpl/figures/climatology_d.png
:width: 6.2in

B.6.9. Same: `pt` vs `noah` (E1–E5 cascade → Noah-lite ET).
```

```{figure} ../artifacts/dpl/figures/climatology_e.png
:width: 6.2in

B.6.10. Same: `noah` → the two LSTM hybrid ensembles (main-text Figure 5).
```

```{figure} ../artifacts/dpl/figures/climatology_summary.png
:width: 6.2in

B.6.11. Climatology summary: per-basin KGE, |bias|, seasonally misplaced volume across the lineage.
```

```{figure} ../artifacts/dpl/figures/climatology_summary_agg.png
:width: 6.2in

B.6.12. Same, basin-aggregated.
```

```{figure} ../artifacts/dpl/superseded/hybrid_dt_noca/seasonal_compare_hybrid_dt_noca.png
:width: 6.2in

B.6.13. Seasonal-timing scoreboard from the frozen-noah predecessor generation:
per-basin validation KGE, seasonal mismatch, and CalSim3 monthly KGE for
`noah_noca` / `hybrid_noca` / `hybrid_dt_noca`.
```

```{figure} ../artifacts/dpl/figures/forcing_sensitivity_rolling.png
:width: 6.2in

B.6.14. Temperature-detrending response of the dPL lineage (`hamon`/`pt`/`noah` and both current hybrid ensembles): 10-yr rolling aggregate flow response. The unconstrained `hybrid` versus the response-constrained `hybrid_dt` is the held-out check of the response-consistency loss.
```

```{figure} ../artifacts/dpl/figures/forcing_sensitivity_monthly_pre1950.png
:width: 6.2in

B.6.15. Same, pre-1950 mean-monthly regime response.
```

```{figure} ../artifacts/dpl/figures/hybrid_summary.png
:width: 6.2in

B.6.16. Skill and (Δprecip, ΔT) climate-response fidelity across the current hybrid
family: `noah` physics, `hybrid`, `hybrid_dt`, and the no-physics `lstm` control
(main-text Figure 6).
```

```{figure} ../artifacts/dpl/figures/hybrid_progression.png
:width: 6.2in

B.6.17. Skill and pooled warming-response curve for the current chain
`noah` → `hybrid` → `hybrid_dt`: per-basin validation KGE, and annual runoff
%change vs present climate along ΔT (Δprecip = 0). Updates the frozen-noah
predecessor generation's progression exhibit (formerly `hybrid_noca` → +PET
input → `hybrid_dt_noca`, retired with that generation) to the current family.
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
