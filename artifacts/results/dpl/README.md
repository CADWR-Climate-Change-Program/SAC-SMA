# `artifacts/results/dpl/`: scores and figures of the learned-parameter runs

One folder per run, under the name of its folder in [`models/`](../../models/README.md) and in
the same group: `15cdec/` holds the runs fitted to the 15 CDEC watersheds, `multifamily/`
dPL-CalSim (`7_calsim`). Beside them, `fidelity/` and `studies/`. Method:
[Learned parameters](../../../docs/learned_parameters.md) and
[CalSim3 rim inflows](../../../docs/calsim3_rim_inflows.md); which run is current and what each
scores: [Runs](../../../docs/runs.md).

## The 15 CDEC watersheds (`15cdec/`)

| Path | What | Command |
|---|---|---|
| `<run>/metrics.csv` | Per watershed: calibration (WY1989–2003) and validation (WY2004–2018) KGE, NSE, percent bias, r, on the daily record with the days of `data/targets/cdec/fnf_daily_mask.csv` left out. The ladder's runs (`1_hru` to `6_aef`) are scored as trained, over the whole record from the cold start; the LSTM ensembles (`hybrid`, `hybrid_dt`, `lstm`) on the mean flow of their three seeds. | `sacsma dpl evaluate <run>/checkpoints/best.pt`; for an ensemble `sacsma.dpl.hybrid.evaluate.score_ensemble` |
| `<run>/figures/` | The ladder's runs: per-watershed diagnostics (`<watershed>_diagnostics.png`) and the skill summary | the same |

The daily flow of a run of the ladder, under any change of climate, is not kept: the engine
remakes it in under a minute (`sacsma.dpl.evaluate.basin_daily`, cached under
`_local/cache/basin_daily/`), and the hybrids and the studies read it from there.

## dPL-CalSim (`multifamily/7_calsim/`)

| Path | What | Command |
|---|---|---|
| `metrics.csv` | Per entity: KGE, NSE, percent bias over its training years | `sacsma dpl evaluate <run>/checkpoints/best.pt --score-holdout` |
| `metrics_holdout.csv` | The same over the held-out WY1976–85 | the same |
| `sim_daily.npz` | Daily flow of every trained entity (mm/day), WY1950–2018 | the same (CPU, a few minutes) |
| `tier1/tier1_metrics.csv` | The 20 tier-1 locations against CalSim3: one row per location × window × reference | `sacsma dpl calsim tier1 <run> --score-holdout` |
| `tier2/tier2_metrics.csv`, `tier2_arcs.csv` | Every rim arc against CalSim3, and each arc's coverage, parent entity and share of trained cells | `sacsma dpl calsim tier2 <run> --trace-python <python of sacsma-gis> --score-holdout` |
| `tier2/tier2_anchor_rescaled.csv` | Each arc after its system is rescaled to the CalSim3 anchor (an evaluation-only score) | the same |
| `atlas/calsim_validation_atlas.html` | One self-contained page: maps, a tab per location, the arcs outside every trained footprint, the training footprints | `sacsma dpl calsim atlas <run> --score-holdout` |

## The fidelity check and the studies

| Path | What | Command |
|---|---|---|
| `fidelity/` | The learned numerics (1 to 20 fixed sub-steps a day) against the reference model on the archived GA optimum of the 15 CDEC watersheds (`fidelity_benchmark.csv`, `figures/fidelity_benchmark.png`) | `sacsma dpl fidelity` |
| `studies/climatology/` | The mean-monthly regime of each step of the ladder, from the GA calibration to `6_aef`, and of the LSTM ensembles on `5_px`, against CalSim3, on the 11 watersheds CalSim3 covers | `sacsma dpl study climatology` |
| `studies/hybrids/` | Response surfaces (Δprecipitation × ΔT) and skill of `hybrid`, `hybrid_dt`, `lstm` against their physics run `5_px`; one figure per watershed in `basins/`; the GA, `5_px` and hybrid scores side by side (`compare_ga_dpl_hybrid.csv`) | `sacsma dpl study hybrids` |
| `studies/forcing/` | The effect of the WGEN temperature detrending on the rungs `2_grid` to `5_px` and the three LSTM ensembles | `sacsma dpl study forcing` |

Every command that takes `<run>` accepts the run's folder in `models/`, `results/` or
`_local/runs/`. The large files these commands also write (the monthly series behind tier 1
and tier 2, the daily arc flows, the maps and figures, the atlas images, the climate-point
passes, the per-entity figures of dPL-CalSim) go to the run's folder under `_local/runs/`.
