# The model

A spatially distributed, daily rainfall–runoff model of California watersheds. Every modeling
unit runs the same four-step chain, and a watershed's flow is the area-weighted sum of its
units' routed runoff. The model and its original calibrations are by Wi and Steinschneider
([Wi & Steinschneider, 2022](references.md#wi2022); [2023](references.md#wimemo)).

## The chain

1. **Potential evapotranspiration.** Hamon PET ([Hamon, 1961](references.md#hamon1961)) scaled
   by a coefficient `Kpet`, with daylength from the CBM model
   ([Forsythe et al., 1995](references.md#forsythe1995)).
2. **Snow-17.** Snow accumulation and melt
   ([Anderson, 1973](references.md#anderson1973); [2006](references.md#anderson2006)), with a
   hard rain/snow split at the temperature `PXTEMP`.
3. **SAC-SMA.** Soil-moisture accounting
   ([Burnash et al., 1973](references.md#burnash1973); [Burnash, 1995](references.md#burnash1995)):
   16 parameters, six storages, five evapotranspiration terms. Rain plus melt from Snow-17 is
   its input.
4. **Lohmann routing.** A gamma hillslope unit hydrograph, then a linearized Saint-Venant
   channel response along the flow path to the outlet
   ([Lohmann et al., 1996](references.md#lohmann1996); [1998](references.md#lohmann1998)).
   Baseflow skips the hillslope step.

Each unit carries 31 parameters: `Kpet`, 16 for SAC-SMA, 10 for Snow-17 and 4 for routing. The
governing equations as implemented, and the parameter table with its ranges, are in
[Model equations](equations.md).

## Modeling units

| Domain | Units | Forcing points | Watersheds |
|---|---|---|---|
| `15cdec` | 7,891 watershed-HRU rows (6,033 distinct HRUs) | 6,033, one per HRU, off the 1/16° grid | 15 |
| `15cdec_grid` | 2,802 watershed-cell rows | 2,074 grid cells | 15 |
| `9unimp` | 466 HRUs | 414 grid cells | 9 |
| `11obs` | 2,448 HRUs | 1,770 grid cells | 11 |
| `12rim` | 1,756 HRUs | 1,594 grid cells | 12 |
| `multifamily` | 9,691 entity-cell rows | 2,849 grid cells | 291 entities |

In the original discretization ([Wi & Steinschneider, 2022](references.md#wi2022)) an HRU is
the intersection of a climate grid cell with a STATSGO soil class
([Miller & White, 1998](references.md#millerwhite1998)), with elevation from the SRTM 90 m DEM
([Jarvis et al., 2008](references.md#jarvis2008)) and a vegetation class from 1 km AVHRR land
cover ([Hansen et al., 2000](references.md#hansen2010)). A watershed lists every HRU it
contains, so nested and neighbouring watersheds repeat HRUs (all of SHA's lie inside BND),
which is why `15cdec` has more rows than HRUs. The `15cdec` HRUs take their temperature from
the grid cell through monthly lapse rates derived from MODIS land-surface temperature
([Wan, 2014](references.md#wan2014)). The grid domains use one parameter set per
1/16° cell and the cell's own meteorology. All of them sit on one 4,410-cell grid that covers
the modeling domains and the CalSim3 rim catchments. The domain names are explained in the
[Glossary](glossary.md).

## Forcing

Daily precipitation and temperature at 1/16° (about 6 km).

| Product | What it is | Period |
|---|---|---|
| `historical_livneh_unsplit` (default) | Precipitation from the extreme-preserving "unsplit" dataset of [Pierce et al. (2021)](references.md#pierce2021); temperature from [Livneh et al. (2013)](references.md#livneh2013), extended and bias-corrected with PRISM ([PRISM Climate Group, 2014](references.md#prism2014)) | 1915–2018 |
| `wgen_product_a` | The same precipitation, temperature detrended to 1991–2020: the historical-parallel sequence of the CalSim3 stochastic-input pipeline | 1915–2018 |
| `wgen_product_a_sNN` | A WGEN Product A climate scenario, stored as the change against `wgen_product_a` (built: `s12`, +2 °C) | 1915–2018 |
| `historical_lto` | The observed climate of the CalSim3 long-term-operations study: the earlier "split" Livneh precipitation | 1915–2021 |
| `aorc` | NOAA AORC | from WY1982 |

The alternates exist on the 1/16° grid only, so `15cdec` runs on the default product alone.
Daily mean temperature is (Tmax + Tmin) / 2. A small table of misplaced-decimal precipitation
values is corrected when the default store is built. Where each store comes from is in
[`data/README.md`](../data/README.md).

## Two implementations of the same physics

**The reference model** (`sacsma/pet.py`, `snow17.py`, `sma.py`, `routing.py`, driven by
`sacsma/model.py`) is NumPy and Numba. It reproduces the archived MATLAB simulations, and that
is the regression baseline for every change: `sacsma verify parity` requires KGE above 0.9999
and a largest daily difference below 0.1 mm/day on one watershed per domain.

| Domain | Watershed | KGE | Largest daily difference |
|---|---|---|---|
| `15cdec` | BND | 0.999997 | 0.0009 mm/day |
| `9unimp` | CacheCreek | 0.999989 | 0.039 mm/day |
| `11obs` | SHA | 0.999952 | 0.028 mm/day |
| `12rim` | SHAST | 0.999908 | 0.063 mm/day |

These files are frozen. Several quirks of the original code are kept on purpose because the
archived parameters were calibrated with them: the variable number of percolation sub-steps, the
squared ratio in the additional-impervious runoff, the storage clamps, and the riparian ET taken
from channel inflow. They are marked ★ in [Model equations](equations.md).

**The differentiable model** (`sacsma/dpl/physics/`) is the same chain in PyTorch, run for all
units at once so that parameters can be learned by gradient descent
([Learned parameters](learned_parameters.md)). With the reference numerics it matches the
reference model to 2 × 10⁻¹³ mm/day on all 15 CDEC watersheds
(`artifacts/dpl/noah/fidelity/fidelity_benchmark.csv`). It adds three options the reference
chain does not have: Priestley–Taylor PET, a soil-moisture-limited ET on observed vegetation
("Noah-lite"), and a learned rain/snow threshold. The first two have reference counterparts in
`sacsma/pet_pt.py` and `sacsma/sma_noah_lite.py`, so learned parameter tables can be run through
the reference model.
