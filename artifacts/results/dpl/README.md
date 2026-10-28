# `artifacts/results/dpl/`: scores and figures of the learned-parameter runs

One folder per run, under the name of its folder in [`models/`](../../models/README.md) and in
the same group: `15cdec/` holds the runs fitted to the 15 CDEC watersheds, `multifamily/`
dPL-CalSim (`8_calsim`). Beside them, `fidelity/` and `studies/`. Method:
[Learned parameters](../../../docs/learned_parameters.md) and
[CalSim3 rim inflows](../../../docs/calsim3_rim_inflows.md); which run is current and what each
scores: [Runs](../../../docs/runs.md).

## The 15 CDEC watersheds (`15cdec/`)

| Path | What | Command |
|---|---|---|
| `<run>/metrics.csv` | Per watershed: calibration (WY1989–2003) and validation (WY2004–2018) KGE, NSE, percent bias, r, on the daily record with the days of `data/targets/cdec/fnf_daily_mask.csv` left out. The ladder's runs (`1_hru` to `7_aef`) are scored as trained, over the whole record from the cold start; the LSTM ensembles (`hybrid`, `hybrid_dt`, `lstm`) on the mean flow of their three seeds. | `sacsma dpl evaluate <run>/checkpoints/best.pt`; for an ensemble `sacsma.dpl.hybrid.evaluate.score_ensemble` |
| `<run>/figures/` | The ladder's runs: per-watershed diagnostics (`<watershed>_diagnostics.png`) and the skill summary | the same |

The daily flow of a run of the ladder, under any change of climate, is not kept: the engine
remakes it in under a minute (`sacsma.dpl.evaluate.basin_daily`, cached under
`_local/cache/basin_daily/`), and the hybrids and the studies read it from there.

## dPL-CalSim (`multifamily/8_calsim/`)

| Path | What | Command |
|---|---|---|
| `metrics.csv` | Per entity: KGE, NSE, percent bias over its training years | `sacsma dpl evaluate <run>/checkpoints/best.pt --score-holdout` |
| `metrics_holdout.csv` | The same over the held-out WY1976–85 | the same |
| `sim_daily.npz` | Daily flow of every trained entity (mm/day), WY1950–2018 | the same (CPU, a few minutes) |
| `tier1/tier1_metrics.csv` | The 20 tier-1 locations against CalSim3: one row per location × window × reference | `sacsma dpl calsim tier1 <run> --score-holdout` |
| `tier2/tier2_metrics.csv`, `tier2_arcs.csv` | Every rim arc against CalSim3, and each arc's coverage, parent entity and share of trained cells | `sacsma dpl calsim tier2 <run> --trace-python <python of sacsma-gis> --score-holdout` |
| `tier2/tier2_anchor_rescaled.csv` | Each arc after its system is rescaled to the CalSim3 anchor (an evaluation-only score) | the same |
| `atlas/calsim_validation_atlas.html` | One self-contained page: maps, a tab per location, the arcs outside every trained footprint, the training footprints | `sacsma dpl calsim atlas <run> --score-holdout` |
| `training_record/` | What each training stage trained on, from its checkpoint through the trainer's own loader: `stages.csv` (stage 1 the basis, stage 2 the fine-tune: checkpoint, donor, epochs, the settings that choose the targets), `families.csv` (per stage and family: entities with a target, units trained, the July–September units, the CDEC monthly record), `entities.csv` (per stage and entity: every count from the registry record to the units trained), `exclusions.csv` (per stage, entity and reason: the units left out and the table that gives the reason). Read with [Training record](../../../docs/training_record.md) | `sacsma dpl record <run>` (CPU, under a minute) |

## The fidelity check and the studies

| Path | What | Command |
|---|---|---|
| `fidelity/` | The learned numerics (1 to 20 fixed sub-steps a day) against the reference model on the archived GA optimum of the 15 CDEC watersheds (`fidelity_benchmark.csv`, `figures/fidelity_benchmark.png`) | `sacsma dpl fidelity` |
| `studies/climatology/` | The mean-monthly regime of each step of the ladder, from the GA calibration to `7_aef`, and of the LSTM ensembles on `5_px`, against CalSim3, on the 11 watersheds CalSim3 covers. Monthly, over WY1950–75, WY1986–87 and WY2004–18 (without the training years and the held-out decade); a mean over the 11 is drawn with a whisker over their range | `sacsma dpl study climatology` |
| `studies/hybrids/` | Response surfaces (Δprecipitation × ΔT) and skill of `hybrid`, `hybrid_dt`, `lstm` against their physics run `5_px`; one figure per watershed in `basins/`; the GA, `5_px` and hybrid scores side by side (`compare_ga_dpl_hybrid.csv`). Each metric is the % change from the model's own present climate over WY1951–75, WY1986–88 and WY2004–18 (annual and April–July runoff, the 99.9th and 30th percentiles of daily flow); `hybrid_regimes/` pools five watersheds per tercile of `5_px`'s April–July share | `sacsma dpl study hybrids` |
| `studies/forcing/` | The effect of the WGEN temperature detrending on the rungs `2_grid` to `5_px` and the three LSTM ensembles: detrended minus historical flow over the 11 watersheds the WGEN stores cover, as a centred 10-year mean of water-year totals and as the mean monthly change before 1950 | `sacsma dpl study forcing` |
| `studies/ladder/` | Every rung of the ladder (the GA parameters to `8_calsim`) and the three LSTM ensembles side by side: mean KGE on WY1989–2003, WY2004–18 and the held-out WY1976–85 (13 outlets, and the USGS gauges, unimpaired-flow subbasins and CalSim3 arcs), and the change in annual and April–July runoff at +3 °C and at precipitation −10 % and +10 % (`ladder.csv`; per watershed in `response.csv`; each step paired record by record, with the number that gain and the median change, in `steps.csv`; `skill.png`, `response.png`, `skill_response.png`, `regional.png`). The WY1976–85 scores are copied from the read records in `models/` (`provenance/read_WY1976-85*.csv`), not recomputed, and the GA's WY1989–2003 and WY2004–18 scores from `8_calsim`'s `provenance/readout_2004_2018_retrained.csv`. `8_calsim` trains on every year but WY1976–85, so it has no score on WY1989–2003 or WY2004–18. The LSTM ensembles' response is that of `studies/hybrids/`. | `sacsma dpl study ladder` (CPU, after `hybrids`) |

Every command that takes `<run>` accepts the run's folder in `models/`, `results/` or
`_local/runs/`. The large files these commands also write (the monthly series behind tier 1
and tier 2, the daily arc flows, the maps and figures, the atlas images, the climate-point
passes, the per-entity figures of dPL-CalSim) go to the run's folder under `_local/runs/`.
