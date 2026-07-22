# SAC-SMA for California Water Systems

*Model description, evaluation, and a differentiable, entity-blind reimplementation*

**California Department of Water Resources — July 2026 (DRAFT)**

## Executive summary

This document describes the hydrologic modeling program built around the Sacramento
Soil Moisture Accounting model (SAC-SMA), a spatially distributed model that
simulates streamflow from daily precipitation and temperature forcing across
California watersheds. Two calibrated applications sit on that shared model: a
general-purpose daily calibration to 15 CDEC reservoir watersheds, and a set of
monthly calibrations purpose-built for the CalSim/CalLite inflow domain. Part I
evaluates both. Part II reimplements the model under differentiable parameter
learning (dPL), a further generalization that is not tied to CalSim either.

Part I details the current implementation, ported to Python with exact numerical
parity against the original MATLAB simulations. Four archived
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
a climate-adaptive, Noah-style soil-moisture-limited ET (`noah`), whose parameters
recompute under a perturbed climate rather than staying fixed. The chain is capped by
a family of SAC×LSTM ensembles that ingest the physics simulation as a feature:
`hybrid`, the best-skill coupling; `hybrid_dt`, which adds a multi-anchor
(Δprecip, ΔT) response-consistency loss tying the network's climate response to the
physics; and a no-physics `lstm` control. On the 15-CDEC daily benchmark the dPL models
match or exceed the GA optimum (pooled mean validation KGE of 0.767 for the GA versus
0.826–0.840 for the physics ladder), and the hybrids reach 0.835–0.877. Scored on a
per-watershed climate-response surface, `hybrid` over-responds to warming (ratio 1.50)
and the no-physics `lstm` control is wrong-signed (ratio −0.94), while `hybrid_dt`
tracks the physics (ratio 1.14) at a modest cost in skill, including reproducing a
warming crossover, a shrinking snowmelt freshet alongside an intensifying flood peak,
on tails it was never trained on. Part II closes with the proposed next
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
