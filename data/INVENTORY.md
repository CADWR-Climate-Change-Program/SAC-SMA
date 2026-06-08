# `data/` inventory

Complete manifest of the tracked data store: **what each file is, what produces
it, what consumes it, and how it is tracked.** Every file under `data/` is
referenced by package code — there are **no orphans**. Sizes are approximate.

Two files are **hand-edited sources of truth** and are *never auto-overwritten*
by the pipeline — edit them by hand and keep them in version control:
`reference/calsim_crosswalk.csv` and `reference/anchor_area_scale.csv`.

Four calibration **domains** ship: `15cdec` (15 CDEC reservoirs, default),
`9unimp` (9 CalLite Unimpaired creeks), `11obs` (11 observed gauges), `12rim`
(12 Rim reservoir inflows). The forcing store is **LFS**; everything else is a
small CSV/GeoPackage tracked normally — every table is plain CSV so it opens in
Excel or a text editor without any script.

## Forcing — `data/forcing/` (git-LFS)

| File | Size | Producer | Consumed by | Tracked |
|------|------|----------|-------------|---------|
| `historical_15cdec.nc` | 774 MB | `dataprep forcing --src <meteo> --name historical_15cdec.nc` | `io.load_forcing` → `model.run_basin` | LFS |
| `historical_11obs.nc`  | 240 MB | `dataprep forcing … --name historical_11obs.nc` | same (domain=11obs) | LFS |
| `historical_12rim.nc`  | 217 MB | `dataprep forcing … --name historical_12rim.nc` | same (domain=12rim) | LFS |
| `historical_9unimp.nc` | 53 MB  | `dataprep forcing … --name historical_9unimp.nc` | same (domain=9unimp) | LFS |

Dims `(key, time)`, vars `prcp`/`tavg`, 1915–2018 daily. A property of the meteo
**grid cells**, not basins (one cell can feed many HRUs). LFS rules live in
`.gitattributes`; `.gitignore` whitelists exactly these four `.nc` and ignores any
other `forcing/*`.

## Per-HRU tables — `data/hru/`, `data/params/` (normal git)

| File (×4 domains) | Size | Producer | Consumed by |
|-------------------|------|----------|-------------|
| `hru/hruinfo_<domain>.csv` | 0.04–0.63 MB | `dataprep tables` (15cdec) / `dataprep unimp9` (others) | `io.load_hru_table` → `model`, `calsim`, `plots` |
| `params/ga_optimum_<domain>.csv` | 0.20–2.4 MB | same | `io.load_params` / `parameters.load_ga_optimum` → `model` |

`hruinfo` = per-HRU `lat, lon, area_weight, elev, flowlen, soil_class, veg_class,
basin`. `ga_optimum` = the archived 31-parameter GA optimum per HRU, keyed by
`key` (`lat_lon`); the CalLite domains also carry a `basin` column (shared cells
hold per-watershed params).

## Reference — `data/reference/` (normal git)

### Parity & calibration targets (per domain)

| File | Size | Producer | Consumed by |
|------|------|----------|-------------|
| `simflow_<domain>.csv` (4) | 10–15 MB | `dataprep reference` (15cdec) / `unimp9` (others) | `io.load_reference` — exact MATLAB parity target (`tests/test_parity_simflow.py`, `plots`) |
| `gage_15cdec.csv` | 5.8 MB | `dataprep gage` | `io.load_gage` — daily observed FNF cal/val target (15cdec only) |
| `calib_<domain>_monthly.csv` (11obs, 12rim, 9unimp) | 0.37–0.40 MB | `dataprep unimp9` (calibration-log parse) | `io.load_calib_monthly` → `plots` (monthly cal target, fallback) |
| `fnf_<domain>_monthly.csv` (11obs, 9unimp, 12rim) | 0.5–0.6 MB | `dataprep unimp9` (11obs/9unimp); `dataprep fnf_xlsx` (12rim, from the DSS spreadsheet) | `io.load_fnf_monthly` → `plots` (full-period monthly FNF; enables validation) |
| `basin_area_<domain>.csv` (15cdec, 11obs, 9unimp) | <0.01 MB | `dataprep tables` / `unimp9` | `io.load_basin_area`, `calsim.basin_areas` — mm/day ↔ cfs |

> Asymmetry by design: `15cdec` uses a **daily** `gage_*` target; the CalLite
> domains use **monthly** `calib_*`/`fnf_*`. `12rim`'s `fnf_*` comes from the CalSim SV
> DSS spreadsheet export (`dataprep fnf_xlsx`, full period WY1922–2003); its `basin_area_*`
> is still pending (parity is mm/day; cfs/anchor compare for 12rim is pending — see CLAUDE.md).

### CalSim cross-compare reference (shared across sets)

| File | Size | Producer | Consumed by |
|------|------|----------|-------------|
| `calsim3_inflow_monthly.csv` | 9.6 MB | `dataprep calsim3` (DSS→csv) | `io.load_calsim3_monthly` → `compare` — the **actual** (truth) |
| `vic_routed_monthly.csv` | 10.4 MB | `dataprep vic` | `io.load_vic_monthly` → `compare` — VIC benchmark |
| `calsim_unimpaired_monthly.csv` | 0.43 MB | `dataprep unimp` | `compare.load_unimpaired_monthly` — per-basin **anchor** reference (11 rim systems) |
| **`calsim_crosswalk.csv`** 🔒 | 0.01 MB | `build_crosswalk(force=True)` once, then **hand-edited** | `calsim.load_crosswalk` / `derive_basin_nodes`, `compare.load_name_map` — single source of truth |
| **`anchor_area_scale.csv`** 🔒 | <0.01 MB | `build_anchor_area_scale(force=True)` once, then **hand-edited** | `compare.load_anchor_area_scale` — anchor-only ±10% area nudge |

🔒 = hand-edited; the pipeline refuses to overwrite it without `force=True`.

## GIS — `data/gis/` (normal git)

| File | Size | Producer | Consumed by |
|------|------|----------|-------------|
| `calsim3.gpkg` | 1.6 MB | `dataprep gis` (layer `CalSim3_And_GooseLake`) + `dataprep merge_gis` (layer `CalSim3_Merged`) | `calsim.load_catchments` / `compare` maps |

Two layers: the original `CalSim3_And_GooseLake` (preserved) and the merged
whole-basin `CalSim3_Merged` used by the per-catchment maps and `run_calsim`.

## Regeneration & tracking

Every artifact is reproducible from the **local-only** reference material under
`tmp/` (gitignored) via the `dataprep` commands above (see `data/README.md`). The
heavy `.nc` forcing stores are LFS; all other files are small and tracked
normally. To track the full four-domain store from a fresh state:

```bash
git lfs install
git add .gitattributes data/          # the three new .nc go to LFS automatically
git status && git lfs ls-files        # verify all four .nc are LFS pointers
git commit -m "Track all four domains (15cdec/9unimp/11obs/12rim) + shared reference"
```
