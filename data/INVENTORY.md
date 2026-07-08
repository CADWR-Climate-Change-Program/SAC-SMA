# `data/` inventory

Complete manifest of the tracked data store: **what each file is, where it came
from, and what consumes it.** The store is split by application — `cdec15/`
(the 15-CDEC domain) and `calsim/` (the CalSim/CalLite domains `9unimp`,
`11obs`, `12rim` plus the CalSim3/VIC references). Every file is referenced by
package code — there are **no orphans**. Sizes are approximate.

All tables are plain **CSV** (openable in Excel or a text editor); only the
gridded forcing stores are NetCDF, tracked with **git-LFS** (`data/**/*.nc` in
`.gitattributes`). One file is a **hand-edited source of truth** and is never
auto-overwritten: `calsim/calsim_crosswalk.csv` 🔒.

All of this data was derived once from the archived MATLAB-era study materials
of **Wi & Steinschneider** (Cornell / UMass Amherst; CA DWR watershed studies)
and from CalSim3/VIC model output. The one-time ingest scripts were retired and
live in git history (the pre-reorg checkpoint commit `ad89558` and earlier);
this file is the provenance record.

## Conventions

- **Join key:** `key = f"{lat:.6f}_{lon:.6f}"` links HRU attributes ↔ GA
  parameters ↔ forcing grid cell.
- **Forcing is grid-cell, not per-basin:** each store has dims `(key, time)`,
  vars `prcp` (mm/day) / `tavg` (°C), daily 1915-01-01 → 2018-12-31. One grid
  cell can feed many HRUs; HRU-level attributes (`elev`, `flowlen`,
  `area_weight`, soil/veg class, `basin`) live in the HRU table.
- **Units:** flow is mm/day (area-normalized); the `basin_area` tables convert
  mm/day ↔ cfs (`sacsma.io.cfs_to_mmday` / `mmday_to_cfs`). Monthly CalSim/VIC
  series are TAF/month.
- **Basin codes** are domain-specific: CDEC codes for `15cdec`/`11obs`/`12rim`
  (SHA, BND, …), CamelCase names for `9unimp` (CacheCreek, StonyCreek, …).

## Forcing provenance (all domains)

The historical meteorology is the **Livneh 1/16° gridded product with the
unsplit-precipitation basis** — precipitation carries the Pierce et al. (2021)
storm-splitting correction (no artificial multi-day splitting of storm
totals); temperature is Livneh, PRISM-adjusted and bias-corrected. The store
filenames carry the provenance (`historical_livneh_unsplit*.nc`). This is the
same precipitation basis as the VIC benchmark (`vic_routed_monthly.csv`,
routed from the VIC `Historical_Unsplit` run), so the SAC-SMA-vs-VIC
cross-compare is apples-to-apples on forcing.

The CalSim domains additionally carry the **WGEN Product A** forcing
(`wgen_product_a_<domain>.nc`) — scenario 1 of the DWR gridded weather
generator release ("Gridded Weather Generator Perturbations…", data.ca.gov),
the **historical-parallel sequence** used by the CalSim3 stochastic-input
pipeline (it forced the pipeline's VIC Product A validation run).
Empirically verified against the Livneh-unsplit store: **precipitation is
identical** (same unsplit basis, released rounded to 0.01 mm); the difference
is **temperature detrended to a 1991–2020 baseline** — the early record is
warmed (+0.40 °C in the 1910s, tapering linearly to ~0 by the 2010s).
`tavg = (tmax+tmin)/2`. Ingested 2026-07 from the per-cell ASCII files
(`BASE/WGEN/Product_A/1/meteo_<lat>_<lon>`, columns `year month day prcp tmax
tmin`) in the OneDrive copy of the `calsim3-stochastic-input-generation` data
store, for exactly the domain's HRU grid cells, written with the same schema/
compression as the Livneh stores. **`15cdec` has no WGEN store** — its HRU
points are off the 1/16° grid (study-specific centroids), so the release does
not cover them without a nearest-cell mapping the original study never
defined.

The third product is **Historical LTO** (`historical_lto_<domain>.nc`) — the
observed-climate VIC forcing carried over from the CalSim3 **LTO
(Long-Term Operations) study** (`BASE/Historical_Climate_LTO/1_Historical`,
per-cell `data_<lat>_<lon>` ASCII: `prcp tmax tmin wind`, no date columns,
implicit **daily 1915-01-01 → 2021-12-31** — three years past the other
stores; wind is dropped). This is the **pre-Pierce-2021 ("split") Livneh
precipitation lineage**: empirically vs the unsplit store, temperature is the
same product (Δ ≈ +0.02 °C uniform) but precipitation is a genuinely
different realization — daily correlation 0.83–0.93 per cell, annual totals
within ±7%, storm mass preserved but daily values differing well beyond the
storm-splitting signature. `tavg = (tmax+tmin)/2`, same schema/compression,
same HRU grid cells. One cell is absent from the release
(`41.46875_-122.15625`, Mt Shasta flank, ≤0.12% of the BND/SHA/SHAST areas)
and is **filled from its southern neighbor** `41.40625_-122.15625` (noted in
the store's `product` attribute). Ingested 2026-07; also excludes `15cdec`
(same off-grid reason as WGEN).

## `data/cdec15/` — the 15-CDEC application

| File | Size | What / provenance | Consumed by |
|------|------|-------------------|-------------|
| `forcing/historical_livneh_unsplit.nc` | 774 MB (LFS) | Livneh-unsplit daily forcing for the 6033 grid cells of the 15-CDEC domain (see above) | `io.load_forcing` → `model.run_basin` |
| `hruinfo.csv` | 0.6 MB | Per-HRU `lat, lon, area_weight, elev, flowlen, soil_class, veg_class, basin` (7891 rows; `area_weight` = per-basin percentage). From the study's HRUinfo tables | `io.load_hru_table` → `model`, `calsim`, plots |
| `ga_optimum.csv` | 2.4 MB | The archived **pooled** GA optimum (KGE objective, WY1989–2003), all 31 parameters expanded to every HRU, keyed by `key` | `io.load_params` → `model` |
| `gage.csv` | 5.8 MB | **Observed daily CDEC full-natural-flow** (the calibration target), 1986–2019, converted from cfs to mm/day over `basin_area.csv`; negatives/sentinels → NaN | `cdec15.load_gage` → `cdec15.plots` |
| `simflow.csv` | 15 MB | The original **MATLAB simulated** gauge flow, 1915–2018 — the exact-reproduction (parity) target | `io.load_reference` → plots, parity checks |
| `basin_area.csv` | <0.01 MB | Published drainage areas `[basin, area_mi2]` for the 15 basins | `io.load_basin_area` — mm/day ↔ cfs |

## `data/calsim/` — the CalSim/CalLite application

### Per-domain files (`9unimp`, `11obs`, `12rim`)

| File (×3 domains) | Size | What / provenance | Consumed by |
|-------------------|------|-------------------|-------------|
| `forcing/historical_livneh_unsplit_<domain>.nc` | 53–240 MB (LFS) | Livneh-unsplit daily forcing for the domain's grid cells | `io.load_forcing` → `model.run_basin` |
| `forcing/wgen_product_a_<domain>.nc` | 56–251 MB (LFS) | **WGEN Product A** daily forcing (1915–2018) for the same grid cells: identical unsplit precipitation, temperature detrended to 1991–2020 (see Forcing provenance) | `io.load_forcing(product="wgen_product_a")` → `sacsma run --forcing wgen_product_a` |
| `forcing/historical_lto_<domain>.nc` | 59–265 MB (LFS) | **Historical LTO** daily forcing (**1915–2021**) for the same grid cells: the LTO-study observed climate — pre-correction ("split") Livneh precipitation lineage; temperature matches the unsplit store (see Forcing provenance) | `io.load_forcing(product="historical_lto")` → `sacsma run --forcing historical_lto` |
| `hruinfo_<domain>.csv` | 0.04–0.19 MB | Per-HRU attribute table (as above); shared cells appear once per owning watershed | `io.load_hru_table` |
| `ga_optimum_<domain>.csv` | 0.2–1.0 MB | The archived **per-watershed** GA optima; carries a `basin` column (shared cells hold different params per watershed) | `io.load_params` |
| `simflow_<domain>.csv` | 10–12 MB | The MATLAB simulated flow — exact parity target (all 32 CalLite watersheds reproduce it exactly) | `io.load_reference` |
| `calib_<domain>_monthly.csv` | 0.4 MB | Monthly **observed FNF** calibration target + MATLAB monthly sim + calibration window, parsed from each watershed's calibration log | `calsim.load_calib_monthly` (fallback cal target) |
| `fnf_<domain>_monthly.csv` | 0.5–0.7 MB | **Full-period** monthly observed FNF (1922–) enabling out-of-calibration validation. 9unimp/11obs from historical-FNF records; **12rim** from the CalSim SV DSS spreadsheet export (reservoir-inflow series ratio-matched to the calibration-log obs) | `calsim.load_fnf_monthly` → `calsim.plots` |
| `basin_area_<domain>.csv` (9unimp, 11obs) | <0.01 MB | Authoritative drainage areas. `12rim` has none (parity is mm/day; it is not in the cross-compare) | `io.load_basin_area` |

### Shared cross-compare reference

| File | Size | What / provenance | Consumed by |
|------|------|-------------------|-------------|
| `calsim3_inflow_monthly.csv` | 9.6 MB | CalSim3 historical `INFLOW` (the **actual**, the cross-compare truth), extracted from the CalSim3 SV DSS | `calsim.load_calsim3_monthly` → `compare` |
| `vic_routed_monthly.csv` | 10 MB | **VIC** routed historical monthly flow (the benchmark), from the VIC `Historical_Unsplit` run — same precipitation basis as the SAC-SMA forcing. Keys `I_SHSTA` and `8RI_SRBB` hold the `_no_gooselake` variant (Goose Lake is endorheic — no real downstream inflow) | `calsim.load_vic_monthly` → `compare` |
| `calsim_unimpaired_monthly.csv` | 0.4 MB | CalSim `FLOW-UNIMPAIRED` whole-watershed series for the 11 rim systems (SHAS, SRBB, OROV, YUBA, FOLS, ST, TU, ME, SJ, TRIN, WH), decade-merged 1920–2021 from the SV DSS — the anchor's per-basin reference | `compare.load_unimpaired_monthly` |
| **`calsim_crosswalk.csv`** 🔒 | 0.01 MB | **Hand-edited master crosswalk** `[arc, system, unimp_anchor, vic_basin, basin_15cdec, basin_11obs, basin_9unimp, in_calsim3]` — the single source of truth for the basin→node mapping, rim-system membership, and VIC names. Bootstrapped geographically once, curated by hand ever since; **never auto-overwritten** | `catchments.load_crosswalk` / `derive_basin_nodes`, `compare.load_name_map` |
| `basin_area_<set>_calsim.csv` (15cdec, 9unimp, 11obs) | <0.01 MB | **Canonical CalSim catchment areas** per basin: the sum of the basin's merged-layer catchment `SQ_MI` (crosswalk nodes + nests + valley node) — puts the basin total on the same area as its sub-arcs and the CalSim reference | `catchments.basin_areas` → `compare` (anchor volume) |
| `screened_footprint_<domain>.csv` (9unimp, 11obs) | <0.05 MB | **Corrected (GIS-screened) footprint** `[basin, key, overlap_area_mi2]`: each basin's HRUs that fall inside its true CalSim catchment (merged-layer polygons + valley node), overlap-area weighted — the parallel, corrected anchor basis. Deterministic from `calsim3.gpkg` + `calsim_crosswalk.csv`; does **not** replace the full-footprint calibration basis. See `data/CALSIM3_FNF_FOOTPRINT.md` | `catchments.screened_footprint` → `compare.make_anchor_screened` |

Which sub-arcs are scored in the per-catchment view is **driven entirely by the
crosswalk** (an arc is scored for a set when `basin_<set>` assigns it and
`in_calsim3=True`) — there is no automatic coverage threshold.
`catchments.COVERED_FRAC` only labels a catchment covered/partial for
diagnostics; `coverage_by_set.csv` reports each set's honest `cov_frac` and
`n_hru` per catchment — curate the crosswalk accordingly.

### GIS

| File | Size | What / provenance | Consumed by |
|------|------|-------------------|-------------|
| `gis/calsim3.gpkg` | 1.6 MB | The CalSim3 inflow catchments, two layers | `catchments.load_catchments`, `compare` maps |

- **`CalSim3_And_GooseLake`** — the original calsim-view `watersheds.geojson`
  (DWR), normalized (`Inflow_arc` → `Connect_No` node names, `Square_Mile` →
  `SQ_MI`); 119/120 rim polygons match a CalSim node.
- **`CalSim3_Merged`** — a derived whole-basin layer: Rim polygons whose
  (alias-aware) arc has **no usable CalSim3 series** are dissolved into the
  node that actually carries their flow (e.g. the Merced `I_MCD###`/`I_MSF###`
  pieces → one whole `I_MCLRE`; `I_RUB002` → Folsom), and the `Type=="Valley"`
  polygons draining to a rim control point (Sac @ Bend Bridge, ~682 mi²) are
  dissolved into one series-less valley-accretion node `I_SRBB_VAL` so those
  HRUs are modelled explicitly. Used by `run_calsim` and the coverage maps.
  The GIS-label alias `I_BRYSA` → `I_PTH070` maps Lake Berryessa to its Putah
  Creek series.
