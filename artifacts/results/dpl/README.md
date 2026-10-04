# `artifacts/results/dpl/`: scores and figures of the learned-parameter runs

One folder per run, under the name of its folder in [`models/`](../../models/README.md). Method:
[Learned parameters](../../../docs/learned_parameters.md) and
[CalSim3 rim inflows](../../../docs/calsim3_rim_inflows.md); which run is current and what each
scores: [Runs](../../../docs/runs.md).

## The 15 CDEC watersheds (`15cdec/`)

| Path | What | Command |
|---|---|---|
| `<run>/metrics.csv` | Per watershed: calibration (WY1989–2003) and validation (WY2004–2018) KGE, NSE, percent bias, r. The parameter runs are scored as trained, over the whole record from the cold start; the LSTM ensembles on the mean flow of their seeds. | `sacsma dpl evaluate <run>/checkpoints/best.pt`; for an ensemble `sacsma.dpl.hybrid.evaluate.score_ensemble` |
| `<run>/figures/` | Per-watershed diagnostics and the skill summary of a parameter run | the same |
| `benchmark/` | The learned numerics (1 to 20 fixed sub-steps a day) against the reference model on the archived GA optimum (`fidelity_benchmark.csv`, `figures/`) | `sacsma dpl benchmark` |
| `studies/climatology/` | The regime of each step from the GA calibration to the hybrids, against CalSim3 | `sacsma dpl study climatology` |
| `studies/hybrids/` | Response surfaces (Δprecipitation × ΔT) and skill of `hybrid`, `hybrid_dt`, `lstm` against `noah`; the GA, dPL and hybrid scores side by side (`compare_ga_dpl_hybrid.csv`) | `sacsma dpl study hybrids` |
| `studies/adaptive/` | Response surfaces of the climate-frozen `noah_noca` against the climate-adaptive `noah` | `sacsma dpl study adaptive` |
| `studies/forcing/` | The effect of the WGEN temperature detrending on each run | `sacsma dpl study forcing` |

## The multi-family runs (`multifamily/`)

| Path | What | Command |
|---|---|---|
| `<run>/metrics.csv` | Per entity: KGE, NSE, percent bias over its training years | `sacsma dpl evaluate <run>/checkpoints/best.pt` |
| `<run>/metrics_holdout.csv` | dPL-CalSim: the same over the held-out WY1976–85 | the same |
| `<run>/sim_daily.npz` | Daily flow of every trained entity (mm/day), WY1950–2018 | the same (CPU, a few minutes) |
| `<run>/tier1/tier1_metrics.csv` | The 20 tier-1 locations against CalSim3: one row per location × window × reference | `sacsma dpl calsim tier1 <run>` |
| `<run>/tier2/tier2_metrics.csv`, `tier2_arcs.csv` | Every rim arc against CalSim3, and each arc's coverage, parent entity and share of trained cells | `sacsma dpl calsim tier2 <run> --trace-python <python of sacsma-gis>` |
| `<run>/tier2/tier2_anchor_rescaled.csv` | dPL-CalSim: each arc after its system is rescaled to the CalSim3 anchor (an evaluation-only score) | the same |
| `<run>/atlas/calsim_validation_atlas.html` | One self-contained page per run: maps, a tab per location, the arcs outside every trained footprint, the training footprints | `sacsma dpl calsim atlas <run>` |

The daily flow of a 15-CDEC parameter run, under any change of climate, is not kept: the engine
remakes it in under a minute (`sacsma.dpl.evaluate.basin_daily`, cached under
`_local/cache/basin_daily/`), and the hybrids and the studies read it from there.

Every command that takes `<run>` accepts the run's folder in `models/`, `results/` or
`_local/runs/`. The large files these commands also write (the monthly series behind tier 1
and tier 2, the daily arc flows, the maps and figures, the atlas images, the climate-point
passes) go to the run's folder under `_local/runs/`.
