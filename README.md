# SAC-SMA for California water systems

A distributed, daily SAC-SMA model of California watersheds in Python: PET → Snow-17 →
Sacramento Soil Moisture Accounting → Lohmann routing, per modeling unit, summed to the
watershed outlet. The model and its original calibrations are by Sungwook Wi and Scott
Steinschneider (Cornell / UMass Amherst) for the California DWR watershed studies.

The repository holds three lines of work on that one model:

| | What it is | Read |
|---|---|---|
| **The model** | The reference implementation (NumPy/Numba). It reproduces the original MATLAB simulations. | [The model](docs/model.md) |
| **Calibrated SAC-SMA** | The original genetic-algorithm calibrations on four sets of watersheds, and their comparison with CalSim3's historical inflows. | [Calibrated SAC-SMA](docs/calibrated_sacsma.md) |
| **Learned parameters** | The same model in PyTorch, with parameters produced by a neural network trained on flow records. | [Learned parameters](docs/learned_parameters.md) |

**Current result.** The learned-parameter model **dPL-CalSim** and its **CalSim3 rim-inflow
product**: monthly flow on the 196 CalSim3 rim arcs, WY1916–2018, under the historical weather
and two WGEN weather sequences. On the water years 1976–85,
which no fit used, the product's median monthly KGE against CalSim3 is 0.81 over all 196 arcs.
See [CalSim3 rim inflows](docs/calsim3_rim_inflows.md).

The guide starts at [`docs/README.md`](docs/README.md).

## Install

The large data files are in git-LFS, so install LFS before cloning.

```bash
git lfs install                        # once per machine, before the clone
mamba env create -f environment.yml    # or: conda env create -f environment.yml
mamba activate sacsma
pip install -e .
```

`environment.yml` covers the model, the learned-parameter code (PyTorch with CUDA) and most
data scripts. The scripts that read rasters or pull from Earth Engine use a second environment,
`environment-gis.yml`. See [Reproduce](docs/reproduce.md).

## Run

```bash
sacsma run BND                                           # one CDEC watershed
sacsma run CacheCreek --domain 9unimp                    # a CalLite watershed
sacsma run ALL --domain 11obs --forcing wgen_product_a   # another forcing product
sacsma plots --domain 15cdec                             # diagnostics -> artifacts/results/calibrated/15cdec/
sacsma calsim                                            # comparison with CalSim3 -> artifacts/results/calibrated/calsim3/
sacsma dpl train --help                                  # learned parameters: train, evaluate, hybrid, calsim, study
sacsma verify                                            # imports, commands, links, artifacts, parity, product
```

```python
from sacsma.model import run_basin
df = run_basin("BND")                           # DataFrame[date, flow], mm/day
df = run_basin("CacheCreek", domain="9unimp")
```

`--spinup-years N` starts a run from an equilibrated state by first running N copies of an
average year. Without it a run starts from the reference cold start.

## Where things are

| Folder | Holds | Described in |
|---|---|---|
| `sacsma/` | the package: the model, the two calibrated applications (`cdec15`, `calsim`), the learned-parameter code (`dpl`) | [`docs/`](docs/README.md) |
| `data/` | inputs, calibration and training targets, references, and the scripts that build them | [`data/README.md`](data/README.md) |
| `artifacts/` | results: tables, figures, trained runs, the rim-inflow product | [`artifacts/README.md`](artifacts/README.md) |
| `docs/` | the guide | [`docs/README.md`](docs/README.md) |

The rules a change has to respect are in [Conventions](docs/conventions.md).

## License

MIT (see [`LICENSE`](LICENSE)). SAC-SMA model and calibrations by Wi & Steinschneider for
California DWR.
