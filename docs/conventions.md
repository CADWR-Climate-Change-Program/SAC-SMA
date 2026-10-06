# Conventions

The rules a change to this repository has to respect, and the reasons for them. They cover the
frozen model, how results are checked, how the comparison with CalSim3 is scored, and how data
and results are kept.

## The reference model is frozen

- `sacsma/pet.py`, `snow17.py`, `sma.py`, `routing.py`, `metrics.py` and `_compat.py` reproduce
  the MATLAB numerics. They are not edited without re-running the parity check below. The
  parity check also guards `sacsma/engine.py`, which runs them (the routing convolution and the
  unit-hydrograph normalisation of the reference model are there).
- Quirks of the original code that are kept on purpose: the variable number of percolation
  sub-steps, the hard rain/snow split at `PXTEMP`, the squared ratio in the additional-impervious
  runoff, the storage clamps, and the two riparian-ET adjustments to channel inflow.
- Initial states everywhere: SAC-SMA `[0, 0, 100, 100, 100, 0]` mm, Snow-17 zeros.
- One engine runs every parameter field that is not being trained (`sacsma/engine.py`, CPU):
  the GA optima through the reference model, a trained field through the step it was trained
  with (`sacsma/sma_learned.py`, the torch step copied to Numba). A trained field is scored as
  trained, never through the reference numerics. Training alone runs in torch.
- Units: precipitation mm/day, temperature °C, flow mm/day over the watershed area. Monthly
  CalSim3, VIC and product volumes are TAF per month.

## Checking a change

There is no test suite. A change is checked by running the model.

```bash
sacsma verify            # imports, commands, links, artifacts, parity, learned, product
sacsma verify --quick    # the checks that need no model run
```

| Check | Passes when |
|---|---|
| `imports` | every module of the package imports |
| `cli` | every command builds its help |
| `links` | every relative link in the tracked markdown resolves |
| `artifacts` | the output tree keeps its layout: tracked files only in `product/`, `models/` and `results/`, each with its README; no untracked file in them; a model and a results folder for every run |
| `parity` | one watershed per domain matches the MATLAB simulation: KGE above 0.9999 and a largest daily difference below 0.1 mm/day |
| `learned` | the learned step of the CPU engine (`sacsma/sma_learned.py`) matches the torch step it copies, for one tracked run of each physics, within 1e-9 mm/day |
| `product` | `sacsma dpl calsim product apply` reproduces every tracked series of the rim-inflow product from the tier-2 pass kept beside it, to 1 part in 100,000 |

Run `parity` after any change that touches the model or the path the data takes into it, and
`learned` after any change to the torch physics or to `sacsma/sma_learned.py`: the two are
changed together.
Regenerated tables can differ from the tracked ones by about 1e-13 through floating-point
order; compare before committing and do not commit noise.

`sacsma dpl benchmark` runs the archived GA optimum through the learned numerics (the fixed
sub-steps the learned-parameter runs train with) and compares it with the reference model.

## One-way dependencies

- `sacsma.calsim` may import `sacsma.cdec15`. Never the reverse.
- Both depend on the core (`model`, `io`, the physics). The core imports neither.
- `sacsma.dpl` depends on the core. The core does not import `sacsma.dpl`.

## Scoring against CalSim3

These rules define the "anchor" scores of `sacsma calsim` (the comparison of the calibrated
model with CalSim3).

**The reference.** For a rim system the reference is CalSim3's single whole-watershed
FLOW-UNIMPAIRED series, because the sum of the system's INFLOW arcs leaves out valley-floor
accretion (about 12 % of the volume for the Sacramento at Bend Bridge). For every other
watershed the reference is the sum of its INFLOW arcs. Each row of the result tables records
which one was used (`ref_kind`).

**The area.** A watershed's simulated depth is turned into volume with the area of its CalSim3
catchment (the `CalSim3_Merged` polygons), not with a published drainage area. Depth biases
that remain are left visible; nothing is rescaled to hide them.

**Footprint screening covers four watersheds only:** SHA, BND, SNS and ChowchillaRiver
(`catchments.SCREENED_BASINS`). Their calibrated modeling units reach well beyond the CalSim3
catchment. SHA and BND include the Goose Lake block, about 1,000 mi² that drains to a closed
basin and never reaches the gauge (VIC makes the same correction with its `no_gooselake`
series). SNS and Chowchilla over-reach through their delineation. For these four, only the
units inside the CalSim3 catchment are kept, weighted by the overlapping area
(`catchments.screened_footprint`).

Every other watershed runs its full calibrated set of units. Trimming an ordinary boundary
would change the depth the calibration was fitted on without correcting anything.

The effect, on the full record (`anchor_screened_vs_full.csv`):

| Watershed | Volume bias, full → screened | KGE, full → screened |
|---|---|---|
| SHA | −8.9 % → +0.1 % | 0.869 → 0.939 |
| SNS | −7.8 % → −0.8 % | 0.878 → 0.958 |
| ChowchillaRiver | −14.3 % → −4.3 % | 0.790 → 0.921 |
| BND | −1.9 % → +4.8 % | 0.958 → 0.887 |

BND gets worse because the units that were removed had been hiding an over-prediction in its
foothill tributaries. The screened number is the honest one. The unscreened scores are kept
beside the official ones (`anchor_metrics_full.csv`, `anchor_monthly_full.csv`).

**What screening never touches.** The calibration targets (`fnf_<set>_monthly.csv`) and the
per-set diagnostics against them. The 15-CDEC set keeps its full set of units.

**The calibration targets are not adjusted either.** `target_vs_calsim3.csv` scores each target
against CalSim3 and labels each watershed `consistent`, `area_artifact` (the target depth was
normalized on a published area that differs from the CalSim3 area) or `product_offset` (a real
difference between the historical record and CalSim3). The offsets are documented, not removed:
changing a target's depth would put the area ratio into the calibration diagnostics.

**Sets.** Observed11 and Unimpaired9 are the anchor sets. CDEC15 is scored the same way as a
parallel track. Rim12 is not part of the comparison.

**Figures and maps show the watershed score.** Every catchment polygon is coloured by the
anchor score of its watershed, never by its own arc's score. Arc-level numbers are in the
tables only.

## Figures

- KGE axes run from 0 to 1. Percent-bias axes are fixed at ±75 % and shared across sets.
- Watersheds are ordered north to south.
- Calibration is on the left, validation on the right.
- Maps share one extent; figures are 300 dpi and at most 6.5 inches wide.

## Names

- **One set of names, the current one.** Pages, data, scripts, file names and code comments
  use the same names. A rename is applied everywhere in one change.
- **No record of old names is kept in the tree.** No alias tables, no notes on what something
  used to be called. Earlier text is in git.
- **The one exception is a run's `provenance/` folder.** Its files are evidence of what was
  fixed before a run's held-out numbers were read, so they are never edited and keep the names
  of their day.

## Data

The layout and the source of every file are in [`data/README.md`](../data/README.md).

- **Inputs, targets and references are kept apart.** Inputs are what a run reads to produce
  flow. Targets are the flow records a model is fitted to. References are series that results
  are only compared with.
- **Targets and references come from outside this repository.** Nothing in the `sacsma`
  package writes them; they change only through the script that sits in their folder.
- **Hand-maintained tables are never overwritten by a script:** the CalSim3 crosswalk, the
  tier-1 location table, the arc derivation table, the unimpaired-flow gauge table and the
  mask of bad observation days. The one exception is asked for by name:
  `sacsma dpl calsim windows --write` stores the two window columns of the tier-1 table.
- **No evapotranspiration product enters a loss, a selection rule or a prior.** The ET and
  snow products in the repository are references.
- **No machine paths in tracked files.** Locations outside the repository are read from an
  untracked local configuration file.

## Outputs

The output tree is described in [`artifacts/README.md`](../artifacts/README.md).

- **Every path under `data/` and `artifacts/` comes from `sacsma/paths.py`.** No path literal
  in the code.
- **Outputs are kept by role.** `product/` is what is delivered (one folder per application
  and forcing), `models/` every model (the archived calibrations and what training made),
  `results/` what a command redraws from them, `_local/` what is not tracked.
- **The archived calibrations are kept by hand.** The five `ga_optimum.csv` tables under
  `models/` (`15cdec`, `15cdec_grid`, `callite/9unimp`, `callite/11obs`, `callite/12rim`) are
  the study's GA optima; no command writes them, and `sacsma verify parity` checks them
  against the MATLAB simulation.
- **A run has one name in every part:** `models/dpl/<group>/<run>/`,
  `results/dpl/<group>/<run>/`, `_local/runs/dpl/<group>/<run>/`. A command that takes a run
  accepts any of the three.
- **Tracked folders hold only tracked files.** A command writes there the files that are
  tracked and everything else to the same path under `_local/`; `sacsma verify artifacts`
  checks it.

## Learned-parameter runs

- Validation years are never read in training or selection.
- An experiment's pass and fail levels are written down before its held-out numbers are read,
  and the held-out numbers are read once. A departure from the written plan is recorded in the
  run's `provenance/` folder.
- One job runs on the GPU at a time.
- Run folders keep the names they were made with; the files in a run's `provenance/` are the
  record of the run as it was made, including names and module paths of that time.
- Runs are named for what they train on (dPL-26, dPL-CalSim).
- What is not adopted stays local (`artifacts/_local/`, `tmp/`) and is not committed.
