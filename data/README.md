# `data/` — organized, Python-native artifacts

The tables here were built **once** from the original MATLAB-era reference
materials (kept local, never committed). Going forward there are two ingests,
each pointing at a **user-supplied source directory** and writing into `data/`:

```bash
# add / replace the domain forcing store  ->  data/forcing/<name>
python -m sacsma.dataprep forcing --src <meteo_dir> [--name historical_15cdec.nc]

# add / replace the watershed GeoPackage  ->  data/gis/<name>
# --src may be a dir of *.shp OR the calsim-view watersheds.geojson (normalized:
# Inflow_arc -> Connect_No node, Square_Mile -> SQ_MI) into the CalSim3_And_GooseLake layer
python -m sacsma.dataprep gis --src <shapefile_dir | watersheds.geojson> [--name calsim3.gpkg]

# add / replace the MATLAB simulated parity target  ->  data/reference/
# (full simflow_sacsma_<CODE>.txt series — NOT the stale *_short.txt drop)
python -m sacsma.dataprep reference --src <simflow_dir>

# add / replace the observed gage FNF calibration target  ->  data/reference/
# (FNF_<CODE>_cfs.txt in cfs; cleaned FNF_cfs_nan/ preferred, raw FNF_cfs/ for BND)
python -m sacsma.dataprep gage --src <fnf_dir>   # needs the area table first (tables)

# add / replace the VIC routed historical monthly flows (TAF) -> data/reference/
python -m sacsma.dataprep vic --src <dir of CS3_<name>_qmo.csv>

# add / replace the CalSim3 historical monthly inflow (TAF) -> data/reference/
# (.dss needs pydsstools<3, e.g. the csstochastic env; or a pre-extracted .csv [date,arc,flow_taf])
python -m sacsma.dataprep calsim3 --src <calsim_sv.dss | extracted.csv>

# add / replace the CalSim FLOW-UNIMPAIRED for the 11 rim systems (TAF) -> data/reference/
# (same SV .dss; or a pre-extracted .csv [date,system,flow_taf])
python -m sacsma.dataprep unimp --src <calsim_sv.dss | extracted.csv>

# (one-time bootstrap) RimInflowAnchor raw crosswalk -> data/reference/calsim_rim_anchor.csv
# (RimInflowAnchor.xlsx [needs openpyxl] or a pre-extracted .csv [arc,system,unimp_anchor]).
# Only needed to (re)bootstrap the master calsim_crosswalk.csv (see below).
python -m sacsma.dataprep rim_anchor --src <RimInflowAnchor.xlsx | extracted.csv>

# (one-time) write the MERGED whole-basin Rim layer (CalSim3_Merged) into calsim3.gpkg
python -m sacsma.dataprep merge_gis

# full-period monthly FNF for 12rim from a CalSim SV DSS spreadsheet export (needs openpyxl)
# -> data/reference/fnf_12rim_monthly.csv (reservoir-inflow series ratio-matched to the
#    calibration-log obs; zero-filled tail dropped; enables early-period validation)
python -m sacsma.dataprep fnf_xlsx --domain 12rim --src <DSS_DATA2.xlsx>
```

### The master crosswalk (`calsim_crosswalk.csv`) — single, hand-edited source of truth

`data/reference/calsim_crosswalk.csv` maps **every CalSim inflow arc** to its rim
`system`, `unimp_anchor`, `vic_basin`, and its owning basin in each SAC-SMA set
(`basin_15cdec`, `basin_11obs`, `basin_9unimp`), plus `in_calsim3` (usable non-zero
series). It is the **authoritative input** for the basin→node mapping, rim-system
membership, and VIC node names — **edit it by hand**; nothing in the pipeline
overwrites it. It supersedes the old `calsim_rim_anchor.csv`, `calsim_vic_name_mapping.csv`,
and per-set `calsim_basin_nodes_*.csv` (all removed).

Which sub-arcs are scored in the per-catchment view is **driven entirely by this crosswalk**
(an arc is scored for a set when it is assigned to a basin in `basin_<set>` and
`in_calsim3=True`) — there is **no automatic coverage threshold**. `calsim.COVERED_FRAC`
only labels a catchment `covered`/`partial`/`outside` for diagnostics, and
`calsim.SUBSYSTEM_FRAC` is used only by the one-time geographic bootstrap, not the live run.
To judge which sub-arcs are worth scoring, `coverage_by_set.csv` / `calset_metrics.csv`
report each set's honest `cov_frac` (HRU sample fraction of the catchment) and `n_hru`; curate
the crosswalk accordingly. Bootstrap it once (then curate):

```bash
python -c "from sacsma.compare import build_crosswalk; build_crosswalk('data', force=True)"
```

The VIC and CalSim3 monthly series are the benchmark for the CalSim inflow run
(see [`../README.md`](../README.md) and `sacsma.compare`): CalSim3 is the
"actual", SAC-SMA and VIC are both scored against it.  `calsim_unimpaired_monthly`
(the 11 rim systems' FLOW-UNIMPAIRED hydrology) is the **anchor's per-basin reference**
for rim basins and feeds the 8-main-river `anchor_hydrographs.png` (the former standalone
`--unimp` rim-system comparison has been retired — the anchor subsumes it).

`calsim_unimpaired_monthly` is also used to derive the **anchor area nudge**
`calsim_area_scale` → `data/reference/anchor_area_scale.csv` (`[set, basin, ref_kind,
area_before_mi2, scale, area_after_mi2, adj_pct, pbias/kge/nse before & after]`). It is a
**hand-editable, anchor-only** per-basin area adjustment (±10% cap) chosen to minimise
|pbias| vs the CalSim reference without degrading KGE/NSE; it scales the SAC-SMA volume in
*every* anchor output but leaves the gage cal/val and per-catchment areas untouched.
Bootstrap (then curate): `python -c "from sacsma.compare import build_anchor_area_scale;
build_anchor_area_scale('data', force=True)"`.

Diagnostic figures (sim vs. observed gage, calibration + validation) are written
to `artifacts/` by `python -m sacsma.plots`.

The one-time per-HRU table build (legacy) takes an explicit path to the
reference tree: `python -m sacsma.dataprep tables --reference-root <dir>`.

### A second domain: the 9 "Unimpaired9" CalLite watersheds (`9unimp`)

A separate per-watershed calibration set (Bear, Cache, Calaveras, Chowchilla,
Cosumnes, Fresno, Mokelumne, Putah, Stony — 414 grid cells). Ingest the HRU
tables, GA params, and reference simflow with one command, plus its own forcing
store:

```bash
python -m sacsma.dataprep unimp9 --hruinfo <Gridinfo dir> --calib <ga dir> --simflow <simflow dir>
python -m sacsma.dataprep forcing --src <9unimp meteo dir> --name historical_9unimp.nc
```

This writes `hru/hruinfo_<domain>.csv`, `params/ga_optimum_<domain>.csv`
(tagged with `basin` — shared cells carry per-watershed params),
`reference/simflow_<domain>.csv`, `reference/calib_<domain>_monthly.csv`
(the **observed monthly FNF** calibration target + MATLAB monthly sim + calibration
period, parsed from the calibration logs), and `forcing/historical_<domain>.nc`
(git-LFS). The same command works for `--domain {9unimp, 11obs, 12rim}` (12 Rim
reservoir inflows, 11 observed gauges, 9 unimpaired creeks — 32 CalLite watersheds).

Run with `run_basin(..., domain="11obs")` or `sacsma run <CODE> --domain 11obs`.
Per-domain calibrated-performance diagnostics (monthly sim vs observed FNF over the
calibration period, + exact MATLAB parity): `python -m sacsma.plots --domain 11obs`
-> `artifacts/<domain>/`.  All 32 reproduce their MATLAB simflow exactly; monthly
calibration KGE ranges ~0.85–0.99 (matching the published study).

## Layout

The **complete, current** per-file manifest — every file's size, what `dataprep`
command produces it, what code consumes it, and how it is tracked — lives in
[`INVENTORY.md`](INVENTORY.md). In brief:

- Four calibration **domains** (`15cdec`, `9unimp`, `11obs`, `12rim`), each with a
  `forcing/historical_<domain>.nc` (git-LFS), `hru/hruinfo_<domain>.csv`,
  `params/ga_optimum_<domain>.csv`, and `reference/simflow_<domain>.csv`
  (exact MATLAB parity target) — all tracked.
- Shared CalSim cross-compare reference under `reference/`: `calsim3_inflow_monthly`
  (the actual), `vic_routed_monthly`, `calsim_unimpaired_monthly` (anchor), and the
  two **hand-edited** sources of truth `calsim_crosswalk.csv` + `anchor_area_scale.csv`.
- `gis/calsim3.gpkg` (two layers: `CalSim3_And_GooseLake` + merged `CalSim3_Merged`).

Only the four `.nc` forcing stores are large (LFS); every table is a plain **CSV**
tracked normally — openable in Excel or a text editor without any script.
`.gitignore` whitelists exactly those four `.nc` and ignores any other `forcing/*`.

## Forcing is grid-cell, not per-basin

The forcing store is a property of the **meteo grid cells** (the set of
`meteo_<lat>_<lon>` files you ingest), independent of any basin delineation.
One grid cell can feed many HRUs. HRU-level attributes (`elev`, `flowlen`,
`area_weight`, soil/veg, `basin`) live in the **HRU table**, which references
the forcing by `key`. To ingest a different domain (e.g. CalSim watersheds),
point `--src` at that set of meteo files:

```bash
python -m sacsma.dataprep forcing --src <dir> --name calsim.nc
```

## Join keys

- `key = f"{lat:.6f}_{lon:.6f}"` links HRU attrs ↔ GA params ↔ meteo grid cell.
- `basin` is the CDEC code (SHA, BND, … ISB), derived from which
  `HRUinfo_<CODE>.txt` a row came from.

## Consuming the data

```python
from sacsma.model import run_basin
df = run_basin("BND", data_dir="data")     # reads data/forcing/historical_15cdec.nc
```
A legacy `.txt` path remains for auditability but requires explicit
`hruinfo_path` / `meteo_dir` / `ga_df` arguments (no in-package default).
