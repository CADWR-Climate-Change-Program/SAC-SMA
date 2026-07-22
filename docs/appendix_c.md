# Appendix C: VIC benchmark and forcing sensitivity

This appendix carries the full narrative, tables, and figures behind the three benchmark analyses summarized in Part I: the cross-comparison of every SAC-SMA calibration set against the VIC model used in CalSim3 development (C.1), the temperature-detrending sensitivity of both models under the WGEN Product A forcing (C.2), and the split-versus-unsplit precipitation sensitivity under the Historical LTO forcing (C.3). The source CSVs behind each figure are listed in Appendix E.

## C.1 Cross-comparison vs VIC and CalSim3 (WY1950–2019)

The cross-comparison scores every set on one shared basis, monthly flows against the CalSim3 anchor references over WY1950–2019 (table below). CDEC15's daily output is aggregated to monthly here, so all three sets and VIC are directly comparable.

| Set | KGE | VIC KGE | \|pbias\| | VIC \|pbias\| | Seas. mism. | VIC seas. mism. |
|---|---|---|---|---|---|---|
| Observed11 | 0.91 | 0.77 | 4.2% | 8.5% | 3.3% | 7.6% |
| Unimpaired9 | 0.92 | 0.62 | 4.9% | 27.3% | 5.6% | 7.7% |
| CDEC15 | 0.87 | 0.77 | 7.6% | 10.0% | 8.2% | 7.1% |

*Table C.1. Mean basin-level skill against the CalSim3 anchor references, WY1950–2019, monthly. VIC columns are means over the same basins as each set; \|pbias\| is the mean absolute per-basin volume bias, and seasonal mismatch is the percentage of annual volume placed in the wrong month.*

The seasonal-mismatch metric is the percentage of annual volume placed in the wrong month (the total-variation distance between normalized mean-monthly regimes; Appendix A.5). It separates timing errors from volume errors in a way KGE alone does not. The anchor sets misplace 3–6% of annual volume seasonally against VIC's 8%; on the CDEC15 basins the pooled set misplaces 8%, slightly more than VIC on the same basins (7%), which reflects its reservoir-oriented pooled calibration.

SAC-SMA outperforms VIC at nearly every basin, by +0.15 mean KGE on the rim set and +0.30 on the creeks (C.1.1). The creek-set gap is wide because the margin is largest where a complete CalSim hydrology is hardest, in the small southern creeks, where VIC runs wet. At Fresno River VIC carries a +95% volume bias (KGE −0.18) against SAC-SMA's +11% (0.87); Cosumnes is +35% versus +1.5%, Chowchilla +41% versus −4%, and Calaveras +31% versus +5%. Rim basins are closer (at Shasta, SAC-SMA 0.94 and VIC 0.78). The hardest rim basin for both models is Trinity (SAC-SMA 0.77, VIC 0.57) (C.1.9).

Four method-level findings follow.

- **Basin-level aggregation stabilizes the comparison.** At individual CalSim3 nodes the per-catchment median KGE is about 0.67 for the SAC sets (VIC 0.66), because sub-arc-scale noise (delineation mismatches, extrapolated small nodes) dominates; aggregation to basin anchors cancels most of it. A per-arc quantile-mapping correction with mass-balance rescaling (trained WY1922–1971, scored on held-out WY1972–2018) lifts the sub-arc median KGE to 0.76–0.86 depending on set, with volume bias magnitudes of 4–11%. That correction is the working route from basin-credible models to arc-level hydrology.
- **The footprint screening affects only the four screened basins.** Screening moves Shasta's anchor bias from −8.9% to +0.1% (the Goose Lake cut) and Chowchilla's from −14.3% to −4.3%; every other basin is unchanged.
- **CDEC15's rim bias is real but specific.** Scored against CalSim3, the pooled set under-runs the two big Sacramento rim systems by −24% (SHA) and −22% (BND). This is the reservoir-calibrated pooled optimum's known weakness, and the reason CDEC15 is kept off the official anchor basis. Its Tulare-basin coverage (PNF, TRM, SCC, ISB), which no CalLite set provides, is unaffected.
- **Skill is stable in time.** In every 30-year rolling window since 1922, SAC-SMA's anchor-set KGE stays within 0.86–0.96 (volume bias within ±8%) and exceeds VIC in the same window throughout.

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_kge.png
:width: 6.2in

C.1.1. Basin-level monthly KGE vs CalSim3 anchor: SAC-SMA vs VIC dumbbells.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_pbias.png
:width: 6.2in

C.1.2. Same, volume bias.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_nse.png
:width: 6.2in

C.1.3. Same, NSE.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_kge_pre1950.png
:width: 6.2in

C.1.4. Anchor KGE, pre-1950 window only.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_kge_post1950.png
:width: 6.2in

C.1.5. Anchor KGE, post-1950 window only.
```

```{figure} ../artifacts/calsim/compare/figures/calsim_sacsma_map_kge.png
:width: 5.2in

C.1.6. SAC-SMA composite basin map, monthly KGE vs CalSim3.
```

```{figure} ../artifacts/calsim/compare/figures/calsim_sacsma_map_pbias.png
:width: 5.2in

C.1.7. SAC-SMA composite basin map, volume bias.
```

```{figure} ../artifacts/calsim/compare/figures/calsim_vic_map_kge.png
:width: 5.2in

C.1.8. VIC benchmark basin map, monthly KGE.
```

```{figure} ../artifacts/calsim/compare/figures/calsim_sacsma_minus_vic_kge.png
:width: 5.2in

C.1.9. SAC-SMA minus VIC KGE difference map.
```

```{figure} ../artifacts/calsim/compare/figures/calsim_sacsma_minus_vic_pbias.png
:width: 5.2in

C.1.10. SAC-SMA minus VIC volume-bias difference map.
```

```{figure} ../artifacts/calsim/compare/figures/rolling_skill_30yr.png
:width: 6.2in

C.1.11. 30-year rolling skill (KGE/NSE/bias/seasonal mismatch), anchor sets vs VIC.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_scatter.png
:width: 5in

C.1.12. Anchor scatter: SAC-SMA vs VIC KGE per basin.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_hydrographs.png
:width: 6.2in

C.1.13. Monthly hydrographs at the eight main river indices.
```

```{figure} ../artifacts/calsim/compare/figures/main_river_climatology.png
:width: 6.2in

C.1.14. Main-river mean-monthly climatology: SAC-SMA (Observed11) vs VIC vs CalSim3.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_pbias_vs_seasonal_11obs.png
:width: 5in

C.1.15. Volume bias vs seasonal mismatch, Observed11.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_pbias_vs_seasonal_9unimp.png
:width: 5in

C.1.16. Volume bias vs seasonal mismatch, Unimpaired9.
```

```{figure} ../artifacts/calsim/compare/figures/anchor_screened_vs_full.png
:width: 6.2in

C.1.17. Screened vs full-footprint anchor scores (differs only at SHA, BND, SNS, Chowchilla).
```

## C.2 Temperature-detrending sensitivity (WGEN Product A)

Because the stochastic pipeline runs on detrended temperature, the response of each hydrologic model to temperature detrending is itself a model property worth measuring. The WGEN Product A forcing isolates it, since precipitation is identical to the baseline and temperature is detrended to 1991–2020 (the early record warmed by up to about 0.4 °C). Both models were re-run under both forcings with parameters unchanged, and differences are reported as Detrended − Baseline on the shared 1915–2018 record, split at WY1950.

**SAC-SMA responds with a volume loss.** Warming the early record raises Hamon PET and snow-season ET, and the model loses 2–4% of long-term runoff volume (median near −3% per domain), concentrated where the detrending is largest. Per-basin losses run −9 to −2.7% before 1950, tapering to about −2% after. Daily flow correlation stays at or above 0.999, so the response is almost entirely in volume rather than timing.

**VIC responds with a timing shift.** VIC's aggregate volume response is only −2.5 to +0.7% pre-1950; it mostly shifts snowmelt earlier instead (C.2.9). The contrast quantifies the liability of a temperature-only PET: any temperature adjustment feeds directly into ET and volume, whereas an energy-budget model mostly re-times the season.

```{figure} ../artifacts/calsim/forcing_compare/figures/wgen_volume_by_period.png
:width: 6.2in

C.2.1. SAC-SMA per-watershed volume difference by period (Detrended − Baseline).
```

```{figure} ../artifacts/calsim/forcing_compare/figures/wgen_annual_diff.png
:width: 6.2in

C.2.2. SAC-SMA 5-yr rolling water-year volume difference.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/wgen_regime_by_period_9unimp.png
:width: 6.2in

C.2.3. SAC-SMA mean-monthly regime under both forcings, Unimpaired9.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/wgen_regime_by_period_11obs.png
:width: 6.2in

C.2.4. Same, Observed11.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_wgen_volume_by_period.png
:width: 6.2in

C.2.5. VIC per-watershed volume difference by period.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_wgen_annual_diff.png
:width: 6.2in

C.2.6. VIC 5-yr rolling volume difference.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_wgen_regime_by_period_9unimp.png
:width: 6.2in

C.2.7. VIC mean-monthly regime, Unimpaired9.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_wgen_regime_by_period_11obs.png
:width: 6.2in

C.2.8. VIC mean-monthly regime, Observed11.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/agg_wgen_annual_diff.png
:width: 6.2in

C.2.9. Aggregate SAC-SMA vs VIC volume response.
```

## C.3 Split vs unsplit precipitation sensitivity (Historical LTO)

The CalSim3 LTO study's observed climate carries the pre-correction "split" Livneh precipitation lineage, whereas the baseline here (and the VIC benchmark's historical run) uses the {cite:t}`pierce2021` unsplit product. Re-running SAC-SMA under the LTO forcing (and comparing the VIC parallels) measures how much this lineage choice matters.

**The difference is concentrated before 1950.** The median absolute water-year volume difference is about 11% in 1915–1949 versus 1.5–3.2% in 1950–2018. Directionally, the unsplit product is wetter in the southern and Sierra basins and drier in the northern Sacramento systems before 1950 (spread up to ±21%), collapsing to within ±2% after 1950 everywhere except **Trinity (+8%)**. Trinity is the one basin where the lineage choice remains consequential in the modern era, and it is also the weakest anchor basin in the cross-comparison (C.1).

**On the full record, anchor skill differs little between the two forcings** (median anchor KGE 0.934 unsplit versus 0.926 split), but the split forcing redistributes skill. It recovers the weakest pre-1950 cases (Trinity pre-1950 KGE 0.40 → 0.83; Cache 0.63 → 0.78; Stony 0.73 → 0.87) while degrading several basins that were already in good agreement (Cosumnes 0.97 → 0.87, Calaveras 0.92 → 0.81, Yuba 0.90 → 0.85) (C.3.10). The practical reading is that pre-1950 disagreements with CalSim3 at a handful of basins owe as much to the precipitation lineage as to the model, and conclusions drawn from pre-1950 skill should carry that caveat.

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_volume_by_period.png
:width: 6.2in

C.3.1. SAC-SMA per-watershed volume difference by period (Unsplit − Split).
```

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_annual_diff.png
:width: 6.2in

C.3.2. SAC-SMA 5-yr rolling volume difference.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_regime_by_period_9unimp.png
:width: 6.2in

C.3.3. SAC-SMA mean-monthly regime, Unimpaired9.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_regime_by_period_11obs.png
:width: 6.2in

C.3.4. Same, Observed11.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_split_unsplit_volume_by_period.png
:width: 6.2in

C.3.5. VIC per-watershed volume difference by period.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_split_unsplit_annual_diff.png
:width: 6.2in

C.3.6. VIC 5-yr rolling volume difference.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_split_unsplit_regime_by_period_9unimp.png
:width: 6.2in

C.3.7. VIC mean-monthly regime, Unimpaired9.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/vic_split_unsplit_regime_by_period_11obs.png
:width: 6.2in

C.3.8. VIC mean-monthly regime, Observed11.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/agg_split_unsplit_annual_diff.png
:width: 6.2in

C.3.9. Aggregate SAC-SMA vs VIC volume response.
```

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_skill_boxplot.png
:width: 6.2in

C.3.10. Anchor skill under unsplit vs split forcing, pooled boxplots by period.
```
