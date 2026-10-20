# `artifacts/models/`: every model

The archived calibrations of the original study and the learned-parameter runs. Each model's
scores and figures are in `results/` under the same name (see
[`artifacts/README.md`](../README.md)).

## The calibrated models

The genetic-algorithm optimum of each calibration domain, from the study archive (Wi &
Steinschneider), 31 parameters per row: `Kpet`, 16 SAC-SMA, 10 Snow-17 and 4 routing parameters.
The reference model (`sacsma.model`) runs them with the HRU tables of
[`data/inputs/domains/`](../../data/inputs/domains/README.md); method and scores:
[Calibrated SAC-SMA](../../docs/calibrated_sacsma.md). Kept by hand: no command writes them.

| File | What |
|---|---|
| `15cdec/ga_optimum.csv` | The pooled optimum of the 15 CDEC watersheds (KGE objective, WY1989–2003), one row per cell, keyed by `key` |
| `15cdec_grid/ga_optimum.csv` | The optimum on the coarse grid-aligned units of `15cdec_grid`, one row per `hruinfo` row, keyed by `basin` and `key`. Its area-weighted median over the rows, parameter by parameter, is also the starting field of every learned-parameter run (`sacsma.dpl.train`) |
| `callite/9unimp/`, `callite/11obs/`, `callite/12rim/` `ga_optimum.csv` | The per-watershed optima of the three CalLite sets, with a `basin` column: a cell shared by two watersheds holds different parameters in each, so filter by `basin` before indexing by `key` |

`sacsma verify parity` checks that they still reproduce the archived MATLAB simulation.

## The learned-parameter runs

One folder per run, under the name it was trained with; its large local files are in
`_local/runs/`. [Runs](../../docs/runs.md) says which run is current, what each scores and what
was tried and not adopted.

```
dpl/15cdec/        runs fitted to the 15 CDEC watersheds
  1_hru/ 2_grid/ 3_pt/ 4_noah/ 5_px/           the ladder: learned parameters, one change per rung
  6_deep/ 7_aef/
  hybrid/ hybrid_dt/ lstm/                      LSTM ensembles, three seeds each (seed0..2/)
dpl/multifamily/   runs fitted to every family of flow record
  8_calsim/                                     dPL-CalSim, the top rung of the ladder
```

`1_hru` runs on the HRUs of `15cdec`; `2_grid` to `7_aef` run on the `multifamily` domain
restricted to the 15 CDEC entities. The hybrids take the daily flow of `5_px` as an input
(`lstm` does not). [Runs](../../docs/runs.md#the-ladder) says what each rung changes and what
it scores.

| File | What | Written by |
|---|---|---|
| `checkpoints/best.pt` | The selected network. It stores its full configuration (`DplConfig`, the variant, the domain), so every other command needs only this file. | `sacsma dpl train` (`sacsma dpl hybrid` for the ensembles) |
| `checkpoints/last.pt` | The last epoch, for `--resume` | `sacsma dpl train` |
| `train_log.csv` | One row per epoch: losses, the selection score, the learning rate | `sacsma dpl train` (`sacsma dpl hybrid` for the ensembles) |
| `params_dpl.csv` | The learned SAC-SMA and Snow-17 parameters per modeling unit, in the shape of `ga_optimum.csv` | `sacsma dpl evaluate` |
| `params_canopy.csv` | The learned evapotranspiration parameters of the Noah-type runs (`4_noah` to `8_calsim`) | `sacsma dpl evaluate` |
| `params_deep.csv` | The deep store's share (`gdeep`), drainage rate (`kdeep`) and residence time in years per modeling unit, for the runs that have it (`6_deep`, `7_aef`, `8_calsim`) | `sacsma dpl evaluate` |
| `tier2_extension_cells.csv` | `8_calsim` only: the traced flow lengths of the 32 extrapolated tier-2 arcs (402 cells), which `product apply` reuses so that a pass needs no tracing and no `sacsma-gis` environment | by hand, from the run's tier-2 pass (`sacsma dpl calsim tier2 --trace-python`) |
| `provenance/` | `8_calsim` (dPL-CalSim): what it was adopted on. `BARS_FINETUNE.md` holds the bars of its fine-tune on the training years, written before the run, with the result appended (scored in `compare_finetune.txt`); `BARS_READ.md` the pass and fail levels of the WY1976–85 read, written before the read, with its result and the read of the rungs `6_deep` and `7_aef` appended; `read_WY1976-85_*.csv` the read's tables (labels, outlets, region, product, the late-summer volume bias in `dry_season`, and the regional records of rungs 2 to 6 on both grid statics in `statics`); `read_footprint.csv` and `read_monthly_ceiling.csv` its observation-only inputs; `readout_2004_2018_retrained.csv` the WY2004–18 readout; `levels_ref.json` and `levels_finetune.json` the loss-reference measurements behind `--mt-loss-ref` of the warm start and of the fine-tune; `warm_start.pt` the network the fine-tune started from. `hybrid`, `hybrid_dt`, `lstm`: `read_WY1976-85.csv`, the read of the held-out WY1976–85 at the 13 outlets (monthly, on the mean flow of the three seeds). Never edited; it keeps the names in use when it was written. | by hand |

A retrained run differs from the tracked one in the last digits (GPU arithmetic), so these
files are the record the results were made from. The per-epoch snapshots and the training
diagnostics logs are local.
