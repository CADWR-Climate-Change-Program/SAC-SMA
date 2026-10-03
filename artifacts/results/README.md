# `artifacts/results/`: what a command redraws

Scores, simulated series, figures and atlases, each redrawn by one command from the models in
[`models/`](../models/README.md) and the data under [`data/`](../../data/README.md). Their
large companions (monthly dumps, maps, per-location figures, climate-point passes) go to the
same path under `artifacts/_local/`.

| Folder | What | README |
|---|---|---|
| `calibrated/` | The calibrated SAC-SMA (the archived GA optima through the reference model): each calibration set against its target, against CalSim3, against VIC and BCM, and under other forcing products | [calibrated](calibrated/README.md) |
| `dpl/15cdec/` | The learned-parameter runs and LSTM ensembles on the 15 CDEC watersheds, the benchmark of the differentiable model, and the four studies | [dpl](dpl/README.md) |
| `dpl/multifamily/` | The multi-family runs: scores per entity, the daily flow, tier 1, tier 2, the atlas | [dpl](dpl/README.md) |
