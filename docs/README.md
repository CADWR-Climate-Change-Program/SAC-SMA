# Guide

Nine short pages. Read the first four in order for the whole picture; the rest are reference.

| Page | What it answers |
|---|---|
| [The model](model.md) | What is simulated, on what units, with what forcing. How the Python model relates to the original MATLAB code. |
| [Calibrated SAC-SMA](calibrated_sacsma.md) | The four genetic-algorithm calibration sets, their skill, and their comparison with CalSim3. |
| [Learned parameters](learned_parameters.md) | How a neural network replaces the calibration, and what that showed on the 15 CDEC watersheds. |
| [CalSim3 rim inflows](calsim3_rim_inflows.md) | The current model (dPL-CalSim), how it is checked, and the monthly inflow product on the 196 rim arcs. |
| [Reproduce](reproduce.md) | Environments, data, and the commands from a clone to the product. What a clone cannot rebuild. |
| [Conventions](conventions.md) | The rules a change must respect: the frozen model, verification, scoring against CalSim3, data. |
| [Glossary](glossary.md) | Domains, families, entities, the two meanings of "anchor", the three of "tier", run names. |
| [Model equations](equations.md) | The governing equations as implemented, and the parameter table. |
| [References](references.md) | The literature cited. |

Beside the guide:

| File | What it is |
|---|---|
| [`data/README.md`](../data/README.md) | Every data folder: its role, its source, how it is built. |
| [`artifacts/README.md`](../artifacts/README.md) | Every result file and the command that writes it. |
| [`artifacts/dpl/RUNS.md`](../artifacts/dpl/RUNS.md) | The learned-parameter runs: what is current, what each scores, what was tried and not adopted. |

## Which model for which use

| Need | Use | Why |
|---|---|---|
| Monthly inflows on the CalSim3 rim arcs | the rim-inflow product of dPL-CalSim | scored on a held-out decade at all 196 arcs; whole-watershed volumes and their response to climate are the model's own |
| Daily flow at the 15 CDEC reservoir watersheds, present climate | `hybrid` or `hybrid_dt` | highest daily skill |
| The same under a changed climate | `hybrid_dt`, or `noah` for physics alone | `hybrid_dt` is trained to keep the physics' response; an LSTM without that training is not trustworthy under warming |
| CalLite inflows as in the original study | the calibrated sets `12rim`, `11obs`, `9unimp` | per-watershed monthly calibrations to those targets |
| The original MATLAB results | the reference model with the archived parameters | reproduces them to within 0.1 mm/day |

## State of the work, October 2026

- The reference model and the four calibrated sets are complete and unchanged.
- The comparison of the calibrated sets with CalSim3 and VIC is complete.
- dPL-CalSim is the current learned-parameter model and its rim-inflow product is adopted. Its
  known costs and open items are in [CalSim3 rim inflows](calsim3_rim_inflows.md) and
  [`artifacts/dpl/RUNS.md`](../artifacts/dpl/RUNS.md).
