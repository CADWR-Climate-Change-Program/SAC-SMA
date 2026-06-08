# SAC-SMA (Python)

A faithful **NumPy/Numba port** of the Sacramento Soil Moisture Accounting
(SAC-SMA) hydrologic model and its coupled modules — Hamon PET, Snow-17, and
Lohmann routing — for daily streamflow over California watersheds.

## Origin

The model was developed by **Sungwook Wi & Scott Steinschneider** (Cornell /
UMass Amherst) for the **CA DWR San Joaquin Watershed Studies**, as a spatially
distributed SAC-SMA built from ~6,000 hydrologic response units (HRUs) across 15
CDEC reservoir watersheds, calibrated with a pooled-KGE genetic algorithm.

The original implementation is **MATLAB** (the per-unit physics functions plus a
driver that loops over HRUs and area-sums routed flow to each gauge). This repo
**ports that MATLAB to Python** — matching the numerics so the Python forward run
reproduces the MATLAB output to floating-point tolerance — while reading clean,
Python-native data artifacts instead of the loose `.txt` reference files.

The MATLAB source and raw reference inputs were converted once by `sacsma/dataprep.py` 
into the organized `data/` store the model actually runs on.

## Install

The domain forcing stores (`data/forcing/historical_*.nc`, 53–774 MB) are tracked
with **git-LFS** — install LFS before cloning so they arrive as real files:

```bash
git lfs install                       # once per machine, before clone
mamba env create -f environment.yml   # or: conda env create -f environment.yml
mamba activate sacsma
pip install -e .
```

## Run

Forward-simulate a basin (or all) from the archived GA optimum. **Four**
calibration domains ship in the repo: **`15cdec`** (15 CDEC reservoir watersheds,
default), **`9unimp`** (9 "Unimpaired9" CalLite creeks — Bear, Cache, Calaveras,
Chowchilla, Cosumnes, Fresno, Mokelumne, Putah, Stony), **`11obs`** (11 observed
gauges), and **`12rim`** (12 Rim reservoir inflows):

```bash
sacsma run BND                          # one CDEC basin -> mean daily flow
sacsma run ALL --out flow.csv           # all 15 CDEC basins -> CSV per basin
sacsma run CacheCreek --domain 9unimp   # a 9-Unimpaired watershed
sacsma run ALL --domain 9unimp          # all 9 Unimpaired watersheds
sacsma run ALL --domain 11obs           # or --domain 12rim
```

All four domains reproduce their MATLAB `simflow` exactly. See
[`data/INVENTORY.md`](data/INVENTORY.md) for the full per-file data manifest.

```python
from sacsma.model import run_basin
df = run_basin("BND", data_dir="data")                    # 15cdec (default)
df = run_basin("CacheCreek", domain="9unimp")             # DataFrame[date, flow] mm/day
```

(9-Unimpaired parity: KGE = 1.0, max daily diff = 0.0 mm/day across all 9 watersheds.)

Regenerate the calibration/validation diagnostic figures:

```bash
python -m sacsma.plots                # -> artifacts/15cdec/
```

See [`data/README.md`](data/README.md) for the data store and the `dataprep`
ingest commands (forcing, GIS, reference, gage).

## Results — 15 CDEC basins

The Python port reproduces the MATLAB simulated flow **exactly** across all 15
basins over 1915–2018 (pooled KGE ≈ 1.0, max daily difference < 0.02 mm/day):

![Python vs MATLAB parity](artifacts/15cdec/figures/parity_vs_matlab.png)

Against the **observed** gage full-natural-flow, calibration skill matches the
published study (mean KGE ≈ 0.83), with separate calibration/validation
statistics per basin:

![Skill summary](artifacts/15cdec/figures/skill_summary.png)

Per-basin diagnostics, the skill summary, the parity figure, and
`metrics_15cdec.csv` live in [`artifacts/15cdec/`](artifacts/15cdec/).

## CalSim cross-compare: CalSim3 vs VIC vs multi-set SAC-SMA

The HRUs are re-aggregated onto the **CalSim3 inflow catchments** (the calsim-view
`watersheds.geojson`, ingested to `data/gis/calsim3.gpkg`; the `Inflow_arc` field
gives the CalSim node names directly) instead of the CDEC reservoir gauges, and the
result is benchmarked against the **CalSim3 historical inflow (the actual)**. The
cross-compare treats **each SAC-SMA calibration set separately** (`15cdec`,
`9unimp`, `11obs`) and reports **two views**:

- **Per-catchment** (`<set>_coverage_map.png`, `<set>_skill.png`): each CalSim3
  inflow node is scored individually (the set's local runoff aggregated to that
  catchment vs CalSim3), shaded as a 0–1 **NSE choropleth**, with a clean coloured
  **outline of each basin's HRU footprint** (the watershed it represents) and a
  per-node KGE plot with **VIC** alongside. `vic_coverage_map.png` shows VIC's NSE
  over **all ~200 VIC-covered Rim catchments** (incl. valley/westside nodes like San
  Luis, independent of any SAC-SMA set).
- **Basin-level "anchor"** (`anchor_skill_{kge,nse,pbias}.png`, `anchor_scatter.png`):
  each set basin vs the **faithful CalSim3 reference for that basin** — the single
  **`FLOW-UNIMPAIRED`** whole-watershed series where the basin maps to a rim system
  (e.g. `unimp_SRBB` for Sac @ Bend Bridge, which captures the valley-floor accretion a
  sub-arc sum misses), otherwise the **sum of the basin's CalSim3 INFLOW sub-arcs** (creeks /
  secondary basins with no aggregate series). The chosen reference is recorded per row as
  `ref_kind`. This subsumes the standalone unimpaired-rim comparison for the SAC sets.

The basin → CalSim-node mapping is driven by **one hand-edited master crosswalk**,
`data/reference/calsim_crosswalk.csv` (`[arc, system, unimp_anchor, vic_basin,
basin_15cdec, basin_11obs, basin_9unimp, in_calsim3]`) — the single source of truth that
aligns the three SAC-SMA sets, VIC, and CalSim3, and **is never auto-overwritten** (edit it
to curate the mapping). `calsim.derive_basin_nodes` simply projects it: a basin's nodes are
the arcs whose `basin_<set>` is that basin, plus a basin-nesting rule (Bend Bridge ⊇ Shasta,
so `BND` sums `I_SHSTA`); the four cumulative single-node systems (Merced = `I_MCLRE`, San
Joaquin = `I_MLRTN`, Shasta, Trinity) resolve to one whole-basin node, and a GIS-label alias
maps `I_BRYSA` → the CalSim3 series `I_PTH070` (Lake Berryessa = Putah Creek). The crosswalk
**replaces** the former `calsim_rim_anchor.csv`, `calsim_vic_name_mapping.csv`, and per-set
`calsim_basin_nodes_*.csv`.

A **merged GIS layer** (`CalSim3_Merged`, via `dataprep merge_gis`) dissolves sub-arcs that
CalSim3 does not model individually into the node that carries their flow (the Merced
`I_MCD###`/`I_MSF###` pieces → a whole `I_MCLRE`), so the cumulative basins appear as whole
catchments on the coverage maps; the original `CalSim3_And_GooseLake` is preserved.

```bash
python -m sacsma.dataprep gis       --src <watersheds.geojson>      # one-time GIS ingest
python -m sacsma.dataprep merge_gis                                 # one-time merged layer
# one-time crosswalk bootstrap, then hand-edit data/reference/calsim_crosswalk.csv:
python -c "from sacsma.compare import build_crosswalk; build_crosswalk('data', force=True)"
sacsma calsim                                                        # -> artifacts/calsim/
python -m sacsma.compare --sets 15cdec 9unimp 11obs                  # equivalent
```

Per-catchment median monthly **KGE vs CalSim3 ≈ 0.67** (VIC ≈ 0.63); at the
**basin level**, against the faithful per-basin reference (FLOW-UNIMPAIRED for rim systems),
skill jumps to median **KGE 0.92 (15cdec), 0.95 (11obs), 0.93 (9unimp)**. Best-of-set covers
~146 nodes.
See [`artifacts/calsim/`](artifacts/calsim/) for metrics, the basin-node CSVs, and
the figures below.

> **Coverage note:** a few GIS catchments have **no usable CalSim3 inflow series** and so
> cannot be scored individually — CalSim lumps the full Merced inflow into `I_MCLRE` (the
> `I_MCD###`/`I_MSF###` sub-pieces have no record, but `I_MCLRE` *is* the Merced basin
> anchor, so Merced scores well at the basin level), the Goose Lake area is absent, and a
> handful of arcs (e.g. `I_RUB002` on the Rubicon) have an identically-zero series because
> CalSim folds their flow into a parent node. These areas still appear inside the basin
> **footprint outline** (the true watershed extent) but don't contribute to a scored sum.

![15cdec coverage](artifacts/calsim/figures/15cdec_coverage_map.png)
![VIC NSE vs CalSim3](artifacts/calsim/figures/vic_coverage_map.png)
![Basin-level KGE](artifacts/calsim/figures/anchor_skill_kge.png)

### The 8 main river indices (anchor hydrographs)

For the rim basins the anchor's reference is the CalSim **`FLOW-UNIMPAIRED`** series
(see above), and `anchor_hydrographs.png` plots the monthly hydrographs of the **8 main
river indices** — the California 8-River Index: Sacramento @ Bend Bridge, Feather
(Oroville), Yuba, American (Folsom), Stanislaus, Tuolumne, Merced, and San Joaquin
(Millerton) — with each SAC-SMA set and VIC against the unimpaired reference (bold).
Shasta, Trinity, and Whiskeytown are deliberately excluded. SAC-SMA **11obs**
(calibrated to these rim gauges) reproduces the unimpaired hydrology at median monthly
**KGE 0.95**; **15cdec** at **0.92**. The source
`data/reference/calsim_unimpaired_monthly.csv` is ingested via
`python -m sacsma.dataprep unimp --src <calsim_sv.dss>`. (The former standalone
`sacsma calsim --unimp` rim comparison has been retired — the anchor subsumes it.)

![8 main river hydrographs](artifacts/calsim/figures/anchor_hydrographs.png)

## License

MIT (see [`LICENSE`](LICENSE)). Port of the Wi & Steinschneider SAC-SMA model
for CA DWR.
