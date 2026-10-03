# `artifacts/`: what the commands write

Four parts, by what a file is. Every path is set in [`sacsma/paths.py`](../sacsma/paths.py); the
method behind each result is in the [guide](../docs/README.md), the state of the runs in
[Runs](../docs/runs.md).

| Folder | What it holds | Written by | If it is lost |
|---|---|---|---|
| [`product/`](product/README.md) | The CalSim3 rim-inflow product: monthly flow on the 196 rim arcs, one folder per forcing, and the share model that makes it | `sacsma dpl calsim product` | Applying the share model to a tracked tier-2 pass rebuilds a series; the share model itself needs its fit passes |
| [`models/`](models/README.md) | What training made: checkpoints, training logs, the learned parameter tables, and each run's `provenance/` | `sacsma dpl train`, `sacsma dpl hybrid`, `sacsma dpl evaluate` (the parameter tables) | Hours of GPU, and a retrained run differs in the last digits |
| [`results/`](results/README.md) | What a command redraws from the models and the data: scores, simulated series, figures, atlases | `sacsma plots`, `sacsma calsim`, `sacsma dpl evaluate`, `sacsma dpl calsim tier1/tier2/atlas`, `sacsma dpl study`, `sacsma dpl benchmark` | Minutes on CPU, or a GPU pass for the multi-family runs |
| `_local/` | Not tracked: caches, scratch runs, the large outputs of every run, and the runs that were not adopted | the same commands | Nothing tracked depends on it |

A learned-parameter run has the same name in each part: `models/dpl/<group>/<run>/`,
`results/dpl/<group>/<run>/` and `_local/runs/dpl/<group>/<run>/`, where `<group>` is `15cdec`
or `multifamily`. Every command that takes a run accepts any of the three. `sacsma verify
artifacts` checks this layout: nothing tracked outside the three tracked parts, nothing
untracked inside them, and a model and a results folder for every run.
