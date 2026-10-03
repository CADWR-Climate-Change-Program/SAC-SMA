# `artifacts/models/`: what training made

The learned-parameter runs, one folder per run, under the name it was trained with. A run's
scores and figures are in `results/` under the same name, its large local files in
`_local/runs/` (see [`artifacts/README.md`](../README.md)). Which run is current, what each
scores and what was tried and not adopted: [Runs](../../docs/runs.md).

```
dpl/15cdec/        runs on the 15 CDEC watersheds (domains 15cdec and 15cdec_grid)
  hamon_dense/ hamon/ pt/ noah_noca/ noah/       learned parameters
  hybrid/ hybrid_dt/ lstm/                       LSTM ensembles, three seeds each (seed0..2/)
dpl/multifamily/   runs on several families of flow record (domain multifamily)
  noah_cdec_uf_sacx_carry_px_aef/                         dPL-26
  noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef/   dPL-95
  noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef/   dPL-CalSim
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
| `provenance/` | dPL-95 and dPL-CalSim: the plan fixed before the held-out numbers were read, its bars, the readout and any departure. Never edited; it keeps the names and paths of its day. | by hand |

A retrained run differs from the tracked one in the last digits (GPU arithmetic), so these
files are the record the results were made from. The per-epoch snapshots and the training
diagnostics logs are local.
