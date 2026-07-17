# SAC-SMA (Python)

Distributed **SAC-SMA** for California watersheds: per-HRU **Hamon PET →
Snow-17 → Sacramento Soil Moisture Accounting → Lohmann routing**,
area-weighted to the watershed outlet. NumPy/Numba, daily time step.

The model and its archived GA calibrations were developed by **Sungwook Wi &
Scott Steinschneider** (Cornell / UMass Amherst) for the CA DWR watershed
studies; this repo runs those calibrations natively in Python (and reproduces
the original MATLAB simulations exactly — see Results).

## Two applications

The package is organized as a generic core plus two clearly separated
applications:

| | `sacsma.cdec15` | `sacsma.calsim` |
|---|---|---|
| Watersheds | 15 CDEC reservoir watersheds (SHA, BND, ORO, …) | The CalSim/CalLite domains: **`9unimp`** (9 CalLite "Unimpaired" creeks), **`11obs`** (11 observed gauges), **`12rim`** (12 Rim reservoir inflows) |
| Calibration | One **pooled** GA optimum, **daily** observed CDEC gage FNF target | **Per-watershed** GA optima, **monthly** observed FNF targets |
| Diagnostics | Daily cal/val skill (`sacsma plots --domain 15cdec`) | Monthly cal/val skill (`sacsma plots --domain 11obs` …) |
| Extras | GA calibration scaffold (`cdec15.calibrate`) | HRU→CalSim-catchment aggregation (`calsim.catchments`) and the CalSim3-vs-VIC cross-compare (`calsim.compare`) |

Data mirrors the split: `data/cdec15/` and `data/calsim/` — see
[`data/INVENTORY.md`](data/INVENTORY.md) for the full manifest and provenance.

## Forcing

All domains run on the same historical meteorology: **daily precipitation and
mean temperature, 1915–2018, on the 1/16° Livneh grid**, using the **unsplit
precipitation** basis (Pierce et al. 2021 storm-splitting correction applied;
temperature is Livneh, PRISM-adjusted and bias-corrected). The per-application
NetCDF stores are named for it — `data/*/forcing/historical_livneh_unsplit*.nc`
(git-LFS) — and share this precipitation basis with the VIC benchmark used in
the cross-compare.

The CalSim domains additionally ship two alternate forcing products from the
CalSim3 stochastic-input pipeline (select with `--forcing <product>` /
`run_basin(..., product=...)`; simulated flows committed under
`artifacts/calsim/<product>/`; provenance in
[`data/INVENTORY.md`](data/INVENTORY.md)):

- **`wgen_product_a`** — the DWR weather-generator historical-parallel
  sequence (identical unsplit precipitation; temperature detrended to a
  1991–2020 baseline), 1915–2018.
- **`historical_lto`** — the LTO-study observed climate (the pre-correction
  **"split"** Livneh precipitation lineage), **1915–2021**.

## Install

The forcing stores are tracked with **git-LFS** — install LFS before cloning
so they arrive as real files:

```bash
git lfs install                       # once per machine, before clone
mamba env create -f environment.yml   # or: conda env create -f environment.yml
mamba activate sacsma
pip install -e .
```

## Run

Forward-simulate a watershed (or all) from its archived GA optimum:

```bash
sacsma run BND                          # one CDEC basin -> mean daily flow
sacsma run ALL --out flow.csv           # all 15 CDEC basins -> CSV per basin
sacsma run CacheCreek --domain 9unimp   # a CalLite watershed
sacsma run ALL --domain 11obs           # or --domain 9unimp / 12rim
sacsma run ALL --parallel               # fan HRUs across cores (~8x)
sacsma run ALL --domain 11obs --forcing wgen_product_a   # WGEN Product A forcing
sacsma run ALL --domain 11obs --forcing historical_lto   # LTO split-basis climate (1915-2021)
```

```python
from sacsma.model import run_basin
df = run_basin("BND", data_dir="data")            # 15cdec (default)
df = run_basin("CacheCreek", domain="9unimp")     # DataFrame[date, flow] mm/day
```

Regenerate the calibration/validation diagnostics:

```bash
sacsma plots --domain 15cdec            # -> artifacts/cdec15/
sacsma plots --domain 11obs             # -> artifacts/calsim/11obs/
```

## Results — 15 CDEC basins

The Python model reproduces the original MATLAB simulated flow **exactly**
across all 15 basins over 1915–2018 (pooled KGE ≈ 1.0, max daily difference
< 0.02 mm/day) — and likewise for all three CalLite domains (KGE = 1.0):

Against the **observed** gage full-natural-flow, calibration skill matches the
published study (mean KGE ≈ 0.83), with separate calibration/validation
statistics per basin:

Per-basin diagnostics, the skill summary, the parity figure, and
`metrics_15cdec.csv` live in [`artifacts/cdec15/`](artifacts/cdec15/).

## CalSim cross-compare: CalSim3 vs VIC vs multi-set SAC-SMA

The HRUs are re-aggregated onto the **CalSim3 inflow catchments**
(`data/calsim/gis/calsim3.gpkg`) instead of the CDEC reservoir gauges, and the
result is benchmarked against the **CalSim3 historical inflow (the actual)**.
The cross-compare treats **each SAC-SMA calibration set separately** (`15cdec`,
`9unimp`, `11obs`) and reports **two views**:

- **Per-catchment** (`calset_metrics.csv`): each CalSim3 inflow node is scored
  individually (the set's local runoff aggregated to that catchment vs CalSim3).
  These per-sub-arc scores live in the CSVs; **the maps and figures show skill
  at the main-basin level** — every sub-area polygon is coloured by its
  watershed's basin-anchor score (`<set>_coverage_map.png`, with a coloured
  **outline of each basin's HRU footprint**; `<set>_skill.png` plots the
  per-basin KGE with **VIC** alongside).
- **Basin-level "anchor"** (`anchor_skill_{kge,nse,pbias}.png`, `anchor_scatter.png`):
  each set basin vs the **faithful CalSim3 reference for that basin** — the single
  **`FLOW-UNIMPAIRED`** whole-watershed series where the basin maps to a rim system
  (e.g. `unimp_SRBB` for Sac @ Bend Bridge, which captures the valley-floor accretion a
  sub-arc sum misses), otherwise the **sum of the basin's CalSim3 INFLOW sub-arcs**
  (creeks / secondary basins with no aggregate series). The chosen reference is
  recorded per row as `ref_kind`.

The basin → CalSim-node mapping is driven by **one hand-edited master
crosswalk**, [`data/calsim/calsim_crosswalk.csv`](data/calsim/calsim_crosswalk.csv)
(`[arc, system, unimp_anchor, vic_basin, basin_15cdec, basin_11obs,
basin_9unimp, in_calsim3]`) — the single source of truth aligning the SAC-SMA
sets, VIC, and CalSim3. It is **never auto-overwritten**: edit it to curate the
mapping. `catchments.derive_basin_nodes` simply projects it (a basin's nodes are
the arcs whose `basin_<set>` is that basin, plus the nesting rule Bend Bridge ⊇
Shasta; the four cumulative single-node systems — Merced, San Joaquin, Shasta,
Trinity — resolve to one whole-basin node).

```bash
sacsma calsim                                   # -> artifacts/calsim/compare/
python -m sacsma.calsim.compare --sets 15cdec 9unimp 11obs --parallel   # equivalent
```

Per-catchment median monthly **KGE vs CalSim3 ≈ 0.67** (VIC ≈ 0.63); at the
**basin level**, against the faithful per-basin reference (FLOW-UNIMPAIRED for
rim systems), skill jumps to median **KGE 0.94 (11obs), 0.91 (9unimp)**. See
[`artifacts/calsim/compare/`](artifacts/calsim/compare/) for metrics and
figures, and [`artifacts/README.md`](artifacts/README.md) for the full output
manifest (including the per-sub-arc QMAP bias-correction and the CalSim ↔
SAC-SMA basin maps).

> **Coverage note:** a few GIS catchments have **no usable CalSim3 inflow series**
> and cannot be scored individually — CalSim lumps the full Merced inflow into
> `I_MCLRE` (which *is* the Merced basin anchor, so Merced scores well at the basin
> level), Goose Lake is absent, and a handful of arcs (e.g. `I_RUB002`) have an
> identically-zero series because CalSim folds their flow into a parent node. These
> areas still appear inside the basin **footprint outline** but don't contribute to
> a scored sum.

## Differentiable parameter learning (dPL) + hybrid LSTM

`sacsma.dpl` is a **PyTorch reimplementation of the full daily pipeline**
(PET → Snow-17 → SAC-SMA → routing) that is differentiable end-to-end, so
SAC-SMA parameters can be *learned* rather than GA-calibrated. A parameter
network maps continuous basin attributes (soil / vegetation / terrain / LAI)
to the per-HRU parameters, trained by backprop through the model, **pooled
across the 15 CDEC basins** on the daily gage-FNF target (calibration
WY1989–2003 / validation WY2004–2018 — the same basis as `sacsma.cdec15`).

A **fidelity gate** anchors the port: the archived GA parameters pushed
through the torch forward reproduce the frozen NumPy/Numba reference to
numerical tolerance (`sacsma dpl benchmark` → `artifacts/dpl/fidelity/`), so
the differentiable model is the same model.

Two rungs of results (pooled 15-basin mean KGE, cal / val):

- **Learned physics** — the dPL parameter net alone. `hamon_dense` on the
  native fine-HRU grid reaches **val KGE 0.84** (matching the GA study's
  ceiling); coarse 1/16°-grid variants swap the evaporative physics —
  `hamon`, `pt` (Priestley–Taylor energy-based PET), `noah` (Noah-lite canopy
  ET with one learned DOF) — landing 0.80–0.83 val.
- **Hybrid SAC×LSTM** — the frozen daily SAC-SMA simulation is coupled to an
  LSTM as an input feature (the LSTM predicts flow directly on top of the
  physics), run as an 8-seed ensemble: **val KGE ≈ 0.87**. The paired
  `hybrid_pet_dt` variant adds a physics-shaped PET input channel and a
  **temperature-consistency loss** that anchors the model's +2 °C response to
  the physics model's — trading a hair of validation skill for a
  physically-consistent warming response (the plain hybrid's unconstrained
  response is unreliable), the version to use for climate projection.

```bash
sacsma dpl benchmark                    # fidelity vs the frozen reference
sacsma dpl train physical --pet priestley_taylor   # train a dPL parameter net
sacsma dpl hybrid --physics <params.csv> --statics # train a hybrid LSTM seed
sacsma dpl evaluate artifacts/dpl/noah/checkpoints/best.pt
sacsma dpl climatology                  # per-basin regime vs CalSim3 FNF
```

Canonical checkpoints, per-model metrics, the cross-model figures, and a
chronological track record of every experiment live in
[`artifacts/dpl/`](artifacts/dpl/) (see
[`artifacts/dpl/RUNS.md`](artifacts/dpl/RUNS.md)). This variant is torch-only
and GPU-oriented; the core `sacsma` package remains torch-free.

## License

MIT (see [`LICENSE`](LICENSE)). SAC-SMA model and calibrations by Wi &
Steinschneider for CA DWR.
