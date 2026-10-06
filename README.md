# SAC-SMA for California water systems

A distributed, daily SAC-SMA model of California watersheds in Python: PET → Snow-17 →
Sacramento Soil Moisture Accounting → Lohmann routing for each modeling unit, area-weighted to
the watershed outlet. The model and its original genetic algorithm-based calibrations 
are by Sungwook Wi and Scott Steinschneider (Cornell / UMass Amherst) for California DWR.

| Model | Description | Doc |
|---|---|---|
| **SAC-SMA** | The reference implementation, which reproduces the original MATLAB simulations, and the CPU engine that runs every parameter field. | [The model](docs/model.md) |
| **SAC-SMA GA-Calibrated** | The original genetic-algorithm calibrations of four watershed sets, compared with CalSim3's historical inflows. | [Calibrated SAC-SMA](docs/calibrated_sacsma.md) |
| **SAC-SMA dPL-Calibrated** | Parameters from a neural network trained on flow records, in PyTorch. | [Learned parameters](docs/learned_parameters.md) |

The CalSim3 rim-inflow product of the learned-parameter model dPL-CalSim:
monthly flow on the 196 rim arcs, WY1916–2018, under the historical weather and two WGEN
sequences. On WY1976–85 validation period, median monthly KGE against CalSim3 is 0.81. See
[CalSim3 rim inflows](docs/calsim3_rim_inflows.md).

The guide starts at [`docs/README.md`](docs/README.md).

## Install

The large files are in git LFS, so install LFS before cloning.

```bash
git lfs install                        # once per machine, before the clone
mamba env create -f environment.yml    # or: conda env create -f environment.yml
mamba activate sacsma
pip install -e .
```

`environment.yml` covers the model and the learned-parameter code (PyTorch with CUDA). The few
scripts that read rasters or use Earth Engine need `environment-gis.yml`; see
[Reproduce](docs/reproduce.md).

## Run

The model runs on the CPU; only training needs a CUDA GPU. Every command writes under
`artifacts/`, and `--help` on a command lists its options.

### GA-Calibrated

One archived calibration per watershed set, chosen with `--domain`:

| `--domain` | Watersheds |
|---|---|
| `15cdec` (default) | `BND FOL ISB MIL MKM MRC NHG NML ORO PNF SCC SHA TLG TRM YRS` |
| `9unimp` | `BearRiver CacheCreek CalaverasRiver ChowchillaRiver CosumnesRiver FresnoRiver MokelumneRiver PutahCreek StonyCreek` |
| `11obs` | `AMF BLB BND FTO MRC SHA SJF SNS TLG TNL YRS` |
| `12rim` | `DPR_I FOL_I LK_MC MILLE N_HOG N_MEL OROVI PRD_C SHAST SMART TRINI WKYTN` |

The forcing is historical (unsplit) Livneh (non-temperature-detrended). 
The three CalLite sets (`9unimp`, `11obs`, `12rim`) also take `--forcing wgen_product_a`, 
`wgen_product_a_s12` or `historical_lto`.

```bash
# daily flow (mm/day)
sacsma run BND                                     # one watershed: prints a summary
sacsma run ALL --domain 9unimp --out flow.csv      # a whole set: flow_<watershed>.csv each
sacsma run ALL --domain 11obs --forcing wgen_product_a_s12 --start 1990-10-01

# diagnostics
sacsma plots --domain 15cdec      # skill and hydrographs vs the gauges -> artifacts/results/15cdec/
sacsma plots --domain 11obs       # a CalLite set -> artifacts/results/callite/11obs/
sacsma calsim                     # 15cdec, 9unimp, 11obs vs CalSim3 and VIC -> artifacts/results/calsim3/

# products
sacsma product callite            # the CalLite inflow files -> artifacts/product/callite/
sacsma product 15cdec             # daily flow of the 15 watersheds -> artifacts/product/15cdec/
```

```python
from sacsma.model import run_basins
flow = run_basins(domain="9unimp")   # a whole set in one run: date x watershed, mm/day
```

### dPL-Calibrated

The trained runs are in `artifacts/models/dpl/`: `15cdec/` holds the 15-watershed runs
(`hamon`, `hamon_dense`, `pt`, `noah`, `noah_noca`) and the LSTM models (`hybrid`,
`hybrid_dt`, and `lstm` without physics); `multifamily/` holds the CalSim3 runs dPL-26
and dPL-CalSim (their folder names are in [Runs](docs/runs.md)). `evaluate` takes a run's
checkpoint, the CalSim3 tools any of its folders; scores and figures go to the run's folder
under `artifacts/results/dpl/`.

```bash
# a 15-watershed run: parameter tables, then skill vs the gauges
sacsma dpl evaluate artifacts/models/dpl/15cdec/noah/checkpoints/best.pt

# a CalSim3 run (here dPL-CalSim)
RUN=artifacts/models/dpl/multifamily/noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef
sacsma dpl evaluate $RUN/checkpoints/best.pt       # skill at its gauges and unimpaired-flow sites
sacsma dpl calsim tier1 $RUN                       # vs CalSim3 at the anchor sets
sacsma dpl calsim tier2 $RUN --trace-python <python of the sacsma-gis environment>   # every rim arc
sacsma dpl calsim atlas $RUN                       # all of it on one HTML page
sacsma dpl calsim product apply --forcing wgen_product_a   # the rim-inflow product under a forcing

# studies of the 15-watershed runs -> artifacts/results/dpl/15cdec/studies/
sacsma dpl study hybrids          # also: climatology, adaptive, forcing
sacsma dpl benchmark              # the learned numerics vs the reference model
```

Training: `sacsma dpl train --help` and [Learned parameters](docs/learned_parameters.md).

### Check

```bash
sacsma verify --quick             # imports, commands, links, layout; no model run
sacsma verify                     # adds parity with the MATLAB runs, the learned step, the products
```

## Layout

| Folder | Holds |
|---|---|
| `sacsma/` | The package: the model and its engine, the calibrated applications (`cdec15`, `calsim`), the learned-parameter code (`dpl`). |
| `data/` | Inputs, targets, references and the scripts that build them ([`data/README.md`](data/README.md)). |
| `artifacts/` | The products, the models and the results ([`artifacts/README.md`](artifacts/README.md)). |
| `docs/` | The guide ([`docs/README.md`](docs/README.md)). |

The rules a change has to respect are in [Conventions](docs/conventions.md).

## License

MIT (see [`LICENSE`](LICENSE)). SAC-SMA model and calibrations by Wi & Steinschneider for
California DWR.
