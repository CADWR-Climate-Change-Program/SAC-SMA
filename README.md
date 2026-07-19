# SAC-SMA (Python)

Distributed SAC-SMA for California watersheds: per-HRU Hamon PET → Snow-17 →
Sacramento Soil Moisture Accounting → Lohmann routing, area-weighted to the
watershed outlet. NumPy/Numba, daily time step.

The model and its genetic-algorithm (GA) calibrations were developed by Sungwook
Wi and Scott Steinschneider (Cornell / UMass Amherst) for the CA DWR watershed
studies. This repo runs those calibrations natively in Python, and reproduces the
original MATLAB simulations exactly (see [Results](#results)).

📖 **[Documentation site](https://cadwr-climate-change-program.github.io/SAC-SMA/)** —
the full technical report: current implementation, evaluation against CalSim3, and
the differentiable reimplementation. Start there for the methods and the numbers;
this README is the quick tour.

## Two applications

A generic core carries two clearly separated applications:

| | `sacsma.cdec15` | `sacsma.calsim` |
|---|---|---|
| Watersheds | 15 CDEC reservoir watersheds (SHA, BND, ORO, …) | The CalSim/CalLite domains: `9unimp` (9 Unimpaired creeks), `11obs` (11 observed gauges), `12rim` (12 Rim inflows) |
| Calibration | One pooled GA optimum, daily observed CDEC FNF target | Per-watershed GA optima, monthly observed FNF targets |
| Diagnostics | Daily cal/val skill (`sacsma plots --domain 15cdec`) | Monthly cal/val skill (`sacsma plots --domain 11obs` …) |

`calsim` may import `cdec15` (the cross-compare scores the CDEC set as one of its
inputs); never the reverse. Data mirrors the split under `data/cdec15/` and
`data/calsim/` — [`data/INVENTORY.md`](data/INVENTORY.md) is the full manifest.

## Forcing

Every grid-based domain runs on the same historical meteorology: daily
precipitation and temperature on the 1/16° Livneh grid, 1915–2018, on the
unsplit-precipitation basis (Pierce et al. 2021 storm-splitting correction). The
VIC benchmark shares that basis, so the SAC-SMA-vs-VIC comparison is
apples-to-apples on forcing.

The CalSim domains also ship two alternate products, selected with `--forcing`:
`wgen_product_a` (identical precipitation, temperature detrended to a 1991–2020
baseline) and `historical_lto` (the older "split" Livneh lineage, extended to
2021). Provenance for all three is in [`data/INVENTORY.md`](data/INVENTORY.md).

## Install

The forcing stores are tracked with git-LFS, so install LFS before cloning:

```bash
git lfs install                       # once per machine, before clone
mamba env create -f environment.yml   # or conda env create -f environment.yml
mamba activate sacsma
pip install -e .
```

## Run

```bash
sacsma run BND                          # one CDEC basin -> mean daily flow
sacsma run ALL --out flow.csv           # all 15, one column per basin
sacsma run CacheCreek --domain 9unimp   # a CalLite watershed
sacsma run ALL --domain 11obs --parallel                 # fan HRUs across cores
sacsma run ALL --domain 11obs --forcing wgen_product_a   # alternate forcing
sacsma run BND --start 2010-10-01 --spinup-years 20      # warm start at any year
sacsma plots --domain 15cdec            # diagnostics -> artifacts/cdec15/
sacsma calsim                           # CalSim cross-compare -> artifacts/calsim/compare/
```

```python
from sacsma.model import run_basin
df = run_basin("BND")                          # 15cdec (default)
df = run_basin("CacheCreek", domain="9unimp")  # DataFrame[date, flow] in mm/day
```

`--spinup-years N` (or `run_basin(..., spinup_years=N)`) prepends an N-year
climatological *average year* before the run window, so a run started at any
`--start` begins from an equilibrated state (soil moisture, snowpack, routing)
instead of the reference cold start. It is opt-in; leaving it off runs exactly as
before.

## Results

The Python model reproduces the MATLAB simulated flow exactly — pooled KGE ≈ 1.0
across all 15 CDEC basins and all three CalLite domains, with max daily
differences under 0.02 mm/day. Against the observed full-natural-flow, calibration
skill matches the published study (mean KGE ≈ 0.83). Per-basin diagnostics, the
skill summary, and the parity figure live in [`artifacts/cdec15/`](artifacts/cdec15/);
[`artifacts/README.md`](artifacts/README.md) is the output manifest.

**CalSim cross-compare.** The HRUs re-aggregate onto the CalSim3 inflow catchments
and score against CalSim3's own historical inflow, benchmarked against VIC. At the
basin level SAC-SMA reaches median monthly KGE of about 0.92–0.95 (VIC 0.62–0.77)
and leads at nearly every basin, with the largest margins in the small, dry
southern creeks. The conventions — the FLOW-UNIMPAIRED anchor for rim systems,
footprint screening, and the QMAP sub-arc correction — are covered in the
[report](https://cadwr-climate-change-program.github.io/SAC-SMA/) and
[`artifacts/README.md`](artifacts/README.md). The basin→node mapping is one
hand-edited crosswalk,
[`data/calsim/calsim_crosswalk.csv`](data/calsim/calsim_crosswalk.csv).

## Differentiable parameter learning (dPL)

`sacsma.dpl` re-expresses the full daily pipeline in PyTorch so the SAC-SMA
parameters can be *learned* by gradient descent instead of GA-calibrated. A
parameter network maps basin attributes (soil, vegetation, terrain, LAI) to the
per-HRU parameters, trained end-to-end and pooled across the 15 CDEC basins on the
daily FNF target. A fidelity gate anchors the port: the archived GA parameters
pushed through the torch model reproduce the frozen NumPy reference to numerical
tolerance, so it is demonstrably the same model. The skill numbers below are all
pooled validation KGE.

**Learned physics.** The parameter net alone reaches val KGE 0.84 on the fine-HRU
grid, matching the GA study's ceiling. Coarse-grid variants that swap the
evaporative physics (`hamon`, `pt`, `noah`) land 0.80–0.83. A climate-adaptive
variant, `noah_ca`, adds four climate indices to the attribute set so the learned
parameters co-vary with the forcing climate; it costs no present-day skill
(val ≈ 0.80) and is the physics the current hybrids build on.

**Hybrid SAC×LSTM.** An LSTM that reads the frozen simulation as one of its
features, run as a seed ensemble, lifts skill to val KGE ≈ 0.88 (`hybrid_base`).
Skill alone is misleading under a changing climate, though. The plain hybrid
over-responds to warming, and a pure `lstm` control with no physics channel comes
out wrong-signed: it learns that warm days mean high flow (the seasonal melt) and
then reads sustained warming as more runoff.

**A trustworthy warming response.** `hybrid_dtdp` adds a multi-anchor (Δp, ΔT)
response-consistency loss that pulls the hybrid's warming and precip response back
toward the physics. It follows the physics on both axes (warming ratio 1.14 and
right-signed at all 15 basins, versus 1.50 for `hybrid_base` and −0.94 for the pure
`lstm`) and gives up only a little skill (val 0.85). This is the one to use for
climate work. It replaces the earlier `hybrid_pet_dt`, which carried a single
+2 °C anchor on the frozen `noah`; `hybrid_dtdp` generalizes that to the full
(Δp, ΔT) surface on the climate-adaptive `noah_ca`.

The diagnostic behind these claims is a per-watershed (Δp, ΔT) response surface
(`sacsma.dpl.dtdp_response`), scored outside the calibration window. Warming
shrinks the April–July snowmelt freshet but intensifies the flood peaks
(daily Q99.9). `hybrid_dtdp` reproduces that crossover on a tail it was never
trained on; the pure LSTM roughly doubles it.

Checkpoints, per-model metrics, and a chronological log of every experiment live
in [`artifacts/dpl/`](artifacts/dpl/); see [`RUNS.md`](artifacts/dpl/RUNS.md).
This variant is torch-only and GPU-oriented; the core `sacsma` package stays
torch-free.

```bash
sacsma dpl benchmark                                   # fidelity vs the frozen reference
sacsma dpl train physical --pet priestley_taylor       # train a parameter net
sacsma dpl hybrid --physics <params.csv> --statics     # train a hybrid LSTM seed
sacsma dpl evaluate artifacts/dpl/noah/checkpoints/best.pt
python -m sacsma.dpl.noah_ca_hybrids                   # noah_ca hybrids + (Δp, ΔT) response surfaces
```

## License

MIT (see [`LICENSE`](LICENSE)). SAC-SMA model and calibrations by Wi &
Steinschneider for CA DWR.
