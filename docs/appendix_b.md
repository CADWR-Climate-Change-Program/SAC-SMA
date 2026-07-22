# Appendix B: Part I evaluation figures

The canonical figure set behind Part I's evaluation of the current implementation: per-set calibration/validation skill, the footprint and input-attribute maps, and the outlier-basin diagnostics. The VIC benchmark and the two forcing-sensitivity analyses have their own appendix (Appendix C), as do the Part II dPL figures (Appendix D). All figures are regenerated programmatically from the archived runs; the source CSVs behind each figure are listed in Appendix E.

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

## B.2 Footprint and input-attribute maps

```{figure} ../artifacts/calsim/compare/figures/shasta_footprint_panels.png
:width: 6.2in

B.2.1. Shasta footprint panels: the calibrated HRU footprint vs the CalSim3 catchment, including the endorheic Goose Lake block.
```

```{figure} ../artifacts/calsim/compare/figures/sns_footprint_panels.png
:width: 6.2in

B.2.2. Stanislaus (SNS) footprint panels (screened: delineation over-reach).
```

```{figure} ../artifacts/calsim/compare/figures/chowchilla_footprint_panels.png
:width: 6.2in

B.2.3. Chowchilla footprint panels (screened).
```

```{figure} ../artifacts/calsim/compare/figures/tnl_footprint_panels.png
:width: 6.2in

B.2.4. Trinity (TNL) footprint panels (unscreened; shown for contrast).
```

```{figure} ../artifacts/calsim/compare/figures/fresno_footprint_panels.png
:width: 6.2in

B.2.5. Fresno footprint panels (unscreened).
```

```{figure} ../artifacts/calsim/compare/figures/hru_veg_15cdec.png
:width: 5.2in

B.2.6. HRU vegetation classes, CDEC15 domain.
```

```{figure} ../artifacts/calsim/compare/figures/hru_soil_15cdec.png
:width: 5.2in

B.2.7. HRU soil classes, CDEC15 domain.
```

```{figure} ../artifacts/calsim/compare/figures/hru_kpet_15cdec.png
:width: 5.2in

B.2.8. Calibrated Hamon K$_{pet}$ field, CDEC15 (single-valued per soil class).
```

```{figure} ../artifacts/calsim/compare/figures/hru_kpet_calsim.png
:width: 5.2in

B.2.9. Calibrated K$_{pet}$, CalLite per-watershed sets (uniform per basin).
```

## B.3 Outlier-basin diagnostics

```{figure} ../artifacts/cdec15/figures/SHA_diagnostics.png
:width: 6.2in

B.3.1. SHA daily diagnostics (CDEC15 pooled set; the rim under-run).
```

```{figure} ../artifacts/cdec15/figures/NHG_diagnostics.png
:width: 6.2in

B.3.2. NHG daily diagnostics (structural positive bias; snow-free basin).
```

```{figure} ../artifacts/calsim/11obs/figures/TNL_diagnostics.png
:width: 6.2in

B.3.3. TNL monthly diagnostics (Observed11 validation outlier).
```

```{figure} ../artifacts/calsim/9unimp/figures/FresnoRiver_diagnostics.png
:width: 6.2in

B.3.4. Fresno River monthly diagnostics (Unimpaired9).
```

```{figure} ../artifacts/calsim/9unimp/figures/CosumnesRiver_diagnostics.png
:width: 6.2in

B.3.5. Cosumnes River monthly diagnostics (Unimpaired9 strong case).
```

```{figure} ../artifacts/calsim/12rim/figures/TRINI_diagnostics.png
:width: 6.2in

B.3.6. TRINI monthly diagnostics (Rim12; Trinity forcing-lineage sensitivity).
```
