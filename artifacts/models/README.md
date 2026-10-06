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
| `15cdec_grid/ga_optimum.csv` | The optimum on the coarse grid-aligned units of `15cdec_grid`, one row per `hruinfo` row, keyed by `basin` and `key`. Also the initial parameter priors of the multi-family runs (`sacsma.dpl.train`) |
| `callite/9unimp/`, `callite/11obs/`, `callite/12rim/` `ga_optimum.csv` | The per-watershed optima of the three CalLite sets, with a `basin` column: a cell shared by two watersheds holds different parameters in each, so filter by `basin` before indexing by `key` |

`sacsma verify parity` checks that they still reproduce the archived MATLAB simulation.

## The learned-parameter runs

One folder per run, under the name it was trained with; its large local files are in
`_local/runs/`. Which run is current, what each scores and what was tried and not adopted:
[Runs](../../docs/runs.md).

```
dpl/15cdec/        runs on the 15 CDEC watersheds (domains 15cdec and 15cdec_grid)
  hamon_dense/ hamon/ pt/ noah_noca/ noah/       learned parameters
  hybrid/ hybrid_dt/ lstm/                       LSTM ensembles, three seeds each (seed0..2/)
dpl/multifamily/   runs on several families of flow record (domain multifamily)
  noah_cdec_uf_sacx_carry_px_aef/                         dPL-26
  noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_w2ft15r10_aef/   dPL-CalSim
```

The parts of a multi-family folder name are explained in the
[Glossary](../../docs/glossary.md#run-folder-names).

| File | What | Written by |
|---|---|---|
| `checkpoints/best.pt` | The selected network. It stores its full configuration (`DplConfig`, the variant, the domain), so every other command needs only this file. | `sacsma dpl train` (`sacsma dpl hybrid` for the ensembles) |
| `checkpoints/last.pt` | The last epoch, for `--resume` | `sacsma dpl train` |
| `train_log.csv` | One row per epoch: losses, the selection score, the learning rate | `sacsma dpl train` |
| `params_dpl.csv` | The learned SAC-SMA and Snow-17 parameters per modeling unit, in the shape of `ga_optimum.csv` | `sacsma dpl evaluate` |
| `params_canopy.csv` | The learned evapotranspiration parameters of the Noah-type runs | `sacsma dpl evaluate` |
| `provenance/` | dPL-CalSim: the check tables it was adopted on, against the bars written before each run (its fine-tunes on training years, then the one read of the held-out years). Never edited; it keeps the names of its day. | by hand |

A retrained run differs from the tracked one in the last digits (GPU arithmetic), so these
files are the record the results were made from. The per-epoch snapshots and the training
diagnostics logs are local.
