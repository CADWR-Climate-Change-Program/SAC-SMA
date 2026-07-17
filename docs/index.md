# SAC-SMA Hydrologic Modeling for CalSim Stochastic Hydrology

*Current Implementation, Evaluation, and Differentiable Reimplementation*

**California Department of Water Resources — July 2026 (DRAFT)**

## Executive summary

The CalSim stochastic hydrology effort requires a complete-domain, best-available
hydrology, meaning daily-capable, monthly-deployable streamflow simulation for every
inflow arc in the CalSim system, suitable for driving stochastic and climate-adjusted
CalSim modeling. This document describes the hydrologic modeling program built around
the Sacramento Soil Moisture Accounting model (SAC-SMA) in two parts.

Part I documents the current implementation, a spatially distributed SAC-SMA (Hamon
PET, Snow-17 snow, Lohmann routing) developed by Wi and Steinschneider (Cornell
University) for CA DWR watershed studies, and since ported to Python with exact
numerical parity against the original MATLAB simulations. Four archived
genetic-algorithm (GA) calibrations cover the domain. One is a pooled daily calibration
to 15 CDEC full-natural-flow watersheds; the others are per-watershed monthly
calibrations to the CalLite/CalSim inflow sets (Rim12, Observed11, Unimpaired9).
Evaluated against CalSim3's own historical inflows at basin level (WY1950–2019),
SAC-SMA attains mean monthly KGE of 0.87 to 0.92 across calibration sets. The VIC
benchmark scores 0.62 to 0.77 on the same basis, and SAC-SMA leads at nearly every
basin, with the largest margins in the small, dry southern creeks. Part I closes with
two forcing-sensitivity analyses. Under temperature detrending (WGEN Product A),
SAC-SMA loses about 3% of long-term runoff volume, while VIC responds mostly by
shifting snowmelt timing. The split-versus-unsplit Livneh precipitation lineage
matters mainly before 1950, with median volume differences of roughly 11% pre-1950
shrinking to 1.5–3% afterward.

Part II describes the reimplementation of this modeling system under differentiable
parameter learning (dPL). The frozen SAC-SMA physics are re-expressed in a
differentiable framework so that a neural network mapping landscape attributes to the
full SAC-SMA parameter set can be trained end-to-end against streamflow by gradient
descent. A chain of progressively more physical variants is developed, from fine-HRU
Hamon (`hamon_dense`) to native-grid Hamon (`hamon`), Priestley–Taylor PET (`pt`), and
Noah-style soil-moisture-limited ET (`noah`). The chain is capped
by a pair of SAC×LSTM hybrid ensembles in which an LSTM ingests the physics simulation
as a feature: `hybrid`, the basic coupling, and `hybrid_pet_dt`, which adds the PET
input and a temperature-consistency loss that ties the network's warming response to
the physics. On the 15-CDEC daily benchmark the dPL models match or exceed the GA
optimum (pooled mean validation KGE of 0.767 for the GA versus 0.836–0.840 for dPL),
and the hybrid ensembles reach 0.864–0.869. Under a +2 °C test the unconstrained
hybrid's response is wrong-signed, adding annual flow where the physics remove it,
while `hybrid_pet_dt` matches the physics response at the same skill, which is why the
pair is kept. Part II closes with the proposed next
phase, which combines multi-timescale calibration (daily where daily FNF exists
post-1987, monthly elsewhere, with held-out WY1950–1987 monthly CalSim FNF validation)
and a nested training architecture that brings the sub-arc watersheds inside each
anchor basin, along with the monthly-only basins, directly into training.

```{toctree}
:maxdepth: 2
:caption: Contents

part1
part2
appendix_a
appendix_b
appendix_c
references
```
