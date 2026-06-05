# `data/` — organized, Python-native artifacts

The tables here were built **once** from the original MATLAB-era reference
materials (kept local, never committed). Going forward there are two ingests,
each pointing at a **user-supplied source directory** and writing into `data/`:

```bash
# add / replace the domain forcing store  ->  data/forcing/<name>
python -m sacsma.dataprep forcing --src <meteo_dir> [--name historical_15cdec.nc]

# add / replace the watershed GeoPackage  ->  data/gis/<name>
python -m sacsma.dataprep gis --src <shapefile_dir> [--name calsim3.gpkg]

# add / replace the MATLAB simulated parity target  ->  data/reference/
# (full simflow_sacsma_<CODE>.txt series — NOT the stale *_short.txt drop)
python -m sacsma.dataprep reference --src <simflow_dir>

# add / replace the observed gage FNF calibration target  ->  data/reference/
# (FNF_<CODE>_cfs.txt in cfs; cleaned FNF_cfs_nan/ preferred, raw FNF_cfs/ for BND)
python -m sacsma.dataprep gage --src <fnf_dir>   # needs the area table first (tables)
```

Diagnostic figures (sim vs. observed gage, calibration + validation) are written
to `artifacts/` by `python -m sacsma.plots`.

The one-time per-HRU table build (legacy) takes an explicit path to the
reference tree: `python -m sacsma.dataprep tables --reference-root <dir>`.

## Layout

| Path | Contents | Tracked? |
|------|----------|----------|
| `hru/hruinfo_15cdec.parquet` | Per-HRU attributes (lat, lon, area_weight, elev, flowlen, soil_class, veg_class, basin_id) + CDEC `basin` code. 6033 HRUs. | ✅ yes (small) |
| `params/ga_optimum_15cdec.parquet` | Per-HRU 31 GA-optimum parameters, keyed by `key` (`lat_lon`). | ✅ yes (small) |
| `reference/simflow_15cdec.parquet` | MATLAB **simulated** flow, long `[date, basin, flow]` (mm/day, 1915–2018). Exact-parity target. | ✅ yes (small) |
| `reference/gage_15cdec.parquet` | **Observed** gage FNF, long `[date, basin, flow]` (mm/day, 1986–2019; missing days NaN). Calibration/validation target, all 15 basins. | ✅ yes (small) |
| `reference/basin_area_15cdec.parquet` | Per-basin drainage area `[basin, area_mi2]` — converts mm/day ↔ cfs. | ✅ yes (small) |
| `forcing/historical_15cdec.nc` | **One domain-wide** forcing store, dims `(key, time)`, vars `prcp`/`tavg`, coords `key`/`lat`/`lon`. 1915–2018 daily. | ✅ git-LFS (other `forcing/*` ignored) |
| `gis/calsim3.gpkg` | Single GeoPackage, one layer per ingested `*.shp`. | ✅ yes (~tens of MB) |

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
