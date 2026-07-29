# `dataprep/`: region auxiliary-data build tools

Living tools that build and extend the **region auxiliary-data store** (`data/region/`): the compact, processed per-cell layers the dPL models draw on for training, fine-tuning, and new-basin setup, for any basin within the cdec15 + CalSim areas. Raw sources and the heavy daily forcing master stay on local disk; the repo carries only the compact processed stores plus these scripts to rebuild and extend them. The stores and their full provenance are catalogued in [`../data/INVENTORY.md`](../data/INVENTORY.md) (§`data/region`).

## The region grid

`build_region_grid.py` → `data/region/grid_cells.csv`: **4410 cells** of the 1/16° Livneh grid, the union of

- the four modeling domains (15cdec_grid ∪ 9unimp ∪ 11obs ∪ 12rim = 2480 cells), and
- the **full CalSim3 footprint**: every cell intersecting any polygon of `data/calsim/gis/calsim3.gpkg` (all 215 Rim watersheds including Goose Lake, plus the 170 Valley polygons), adding 1930 cells so every CalSim3 location is coverable.

Keys are normalized 5-decimal `<lat>_<lon>`; per-cell flags `in_<domain>` and `in_calsim3_fp` mark membership. Every ingest below targets this list.

## Layers

| layer | script | in-repo store | status |
|---|---|---|---|
| grid definition | `build_region_grid.py` | `grid_cells.csv` (0.1 MB) | done |
| statics: soil/veg + LAI climatology | `build_region_statics.py` | `{soilveg_continuous,lai_climatology}.csv` (~4 MB) | **partial: 2480/4410 cells**; footprint-only cells need a raster ingest (see below) |
| ET obs: gleam, fluxcom | `local_obs_region.py` | `et_obs/*.npz` | done (verified to 1e-7) |
| ET/SWE obs: terraclimate/fldas/era5land/daymet | `gee_obs_region.py` | `et_obs/*.npz`, `swe_obs/*.npz` | done (GEE spec v2, 2026-07-16) |
| ET referees: openet, modis | `gee_obs_region.py --products openet modis` | `et_obs/{openet,modis}_*.npz` | done (benchmark-only, 2026-07-17) |
| daily forcing master (raw) | `wgen_forcing.py` | local only (not in repo) | done |
| raw GIS rasters (soil/veg/terrain/LAI staging) | `download_gis.py` | local only (~89 GB, `D:\sacsma-data\raw_gis`) | staged + verified complete (2026-07-29); re-fetch is resumable |
| ×10 precip-artifact table | `wgen_forcing.py --scan-x10` | `prcp_x10_artifacts.csv` (frozen) | done |
| **unified region forcing** | `build_region_forcing.py` | `forcing/{historical_livneh_unsplit,wgen_product_a,historical_lto}.nc` (~3.1 GB LFS) | done; replaced the per-domain stores (2026-07-16) |
| **AORC forcing (1979–2025)** | `aorc_region.py` | `forcing/aorc.nc` | ⚠ **store held, not committed** — full pull finished 2026-07-29, but source fill values are unmasked and contaminate 5 of 47 years (see the open defect below) |
| **BCM monthly hydrology (WY1916–2018)** | `bcm_region.py` | `bcm/bcm_<scenario>_monthly.nc` + `bcm/bcm_<scenario>_catchments_monthly.csv` + `bcm/bcm_catchments.csv` (~200 MB) | done — Scenario 1 + Scenario 13, all six variables (2026-07-28) |

## AORC (`aorc_region.py`)

NOAA's Analysis of Record for Calibration v1.1 — 1-km hourly, CONUS, 1979–2025 —
aggregated to the region grid as a fourth forcing product. Because `product` is
just a filename stem, `data/region/forcing/aorc.nc` works as `--forcing aorc`
with no code change.

The source is a public Zarr store on S3 (`s3://noaa-nws-aorc-v1-1-1km/{year}.zarr`,
us-east-1, zarr v2 + consolidated metadata, `int16` + zstd-3, chunks
`(144 time, 128 lat, 256 lon)`). Its grid was **read from the store, not assumed**:

```
lat = 20.0   + 0.0083330*i,  i = 0..4200      (20 .. 55 N)
lon = -130.0 + 0.0083330*j,  j = 0..8400      (-130 .. -60 E)
```

i.e. a uniform 30 arcsec. Each region cell receives 49–64 AORC cells (mean 56.2
vs 56.3 theoretical) by box average.

**Volume and why it streams.** Zarr's read atom is the whole chunk, so a small
region cannot ask for less, and neither a coarser AORC product nor an Earth
Engine copy exists. Only 18 of the 27 candidate spatial chunks hold region
cells; the other 9 are Pacific fill. Measured on the 18 that matter:

| scope | wire | at 3.6 MB/s |
|---|---|---|
| `APCP` + `TMP` → `prcp`/`tmin`/`tmax` | ~175 GB | ~14 h |
| all 8 variables | ~1.1 TB | ~3.5 days |

A profile of the fetch loop puts **95.6 % of wall time in the network**
(decompress 1.3 %, aggregation 2.9 %), so the pull is bandwidth-bound and there
is nothing to gain from optimising the aggregation. Run it in `us-east-1`
instead and the same job transfers ~3 GB home rather than ~1.1 TB.

Chunk compressibility is strongly seasonal — a dry June precipitation chunk is
875 B against 9.44 MB raw (~10 000×), while a July temperature chunk is ~3.4 MB
(~2.8×) — so estimates from a single season mislead; the table above is taken
from summer chunks and is the conservative end.

The script **never writes the raw data**: each chunk is decompressed in memory,
box-averaged onto the 4410 cells, and discarded. Local cost is ~2.4 GB of
partials plus the final store.

**Resume.** Work is banked at two levels, both written atomically (temp file +
rename) so a hard kill cannot leave a truncated file:

- per time chunk → `tmp/aorc_parts/_hourly/<var>_<year>/<t>.npy` (~35 MB of
  transfer each, so an unexpected stop costs seconds)
- per variable-year → `tmp/aorc_parts/<var>_<year>.npz`, after which that
  year's hourly checkpoints are deleted

Re-running skips whatever is already banked, so an interrupted pull — dropped
VPN, sleeping laptop, killed process — resumes where it stopped.

```bash
python dataprep/aorc_region.py --years 2015 --vars APCP_surface TMP_2maboveground
python dataprep/aorc_region.py --all                     # everything, resumable
python dataprep/aorc_region.py --status                  # progress of a running pull
python dataprep/aorc_region.py --assemble --vars APCP_surface TMP_2maboveground
```

`--assemble` stitches whatever variables are banked over the years they share,
so the physics-critical `prcp`/`tmin`/`tmax` subset can land before the rest.
`--status` reports progress from the partials on disk rather than from a log,
which is the reliable source of truth for a multi-day run.

Because the full 8-variable pull takes days, run it detached rather than inside
a shell you might close. `tmp/run_aorc_pull.ps1` (local-only) does this: it
launches the pull hidden, restarts it if the process dies, and runs `--assemble`
once the pull exits cleanly. Variables are processed in the order listed above,
so `prcp`/`tmin`/`tmax` complete first and are assemblable long before the
radiation/humidity/wind fields finish.

**Day convention.** `--utc-offset` defaults to **0 (UTC calendar days)**, which
is what the committed Livneh store uses. This was measured, not assumed: scanning
all 24 whole-hour offsets for 2015 precipitation shows agreement falling off
monotonically away from UTC (r on the regional daily series 0.9565 at 0, 0.7624
at −8/PST, 0.5616 at −16).

**Verification.** 2015 against `historical_livneh_unsplit.nc` — independent
products, so agreement rather than identity is the gate:

| field | AORC | Livneh | bias | RMSE | r cell-day | r regional |
|---|---|---|---|---|---|---|
| `prcp` (mm/d) | 1.305 | 1.290 | +0.015 | 3.396 | 0.8234 | 0.9565 |
| `tmin` (°C) | 6.950 | 7.342 | −0.392 | 1.536 | 0.9795 | 0.9963 |
| `tmax` (°C) | 21.557 | 21.201 | +0.356 | 1.621 | 0.9865 | 0.9957 |

Per-cell annual precipitation totals correlate at 0.9755 (bias +1.13 %). Monthly
`tmax` differences stay within +0.02…+0.64 °C with no seasonal drift. The
signed pattern — `tmin` low, `tmax` high — is a slightly wider diurnal range in
AORC, which is what a 1-km hourly analysis should show against a
daily-disaggregated product. Wet-day fraction (>1 mm) 0.131 vs 0.143: AORC
concentrates precipitation into fewer, more intense days.

### ⚠ Open defect: source fill values are not masked (found 2026-07-29)

**The assembled `aorc.nc` is wrong and is deliberately not committed.** The
source arrays declare `missing_value: -32767` alongside the `scale_factor` that
`scale_factors()` reads, but nothing masks it, so fill is treated as data:
scaled, spatially averaged, and summed into the daily fields. A filled hour
becomes −3276.7, and a fully filled precipitation day −78 640 mm.

Only 0.019–0.034 % of values are affected, but the magnitude is such that the
domain-mean `prcp` over 1979–81 comes out at **−13.2 mm/day** instead of the
1.955 mm/day the clean values give. It is concentrated in whole-domain outages
in **5 of 47 years — 1979, 2000, 2001, 2009, 2024** (1979 = 2 full days ×
4410 cells, 2024 = 1 full day). The 2015 verification above never sampled one,
which is why it passed: it was clean, not representative.

The fix belongs in `fetch_hourly`, **not** in `daily_from_hourly` — the 1-km box
average (`acc /= counts`, a *static* per-cell count) runs on raw `int16` before
scaling, so filled and valid 1-km cells are blended and the contamination is not
even an integer multiple of the fill. Masking must drop filled cells from both
the sum and a *per-hour* count, which then raises the question the fix has to
answer: what a partly-observed hour, and a day with any missing hour, should
become (renormalise over valid cells, or NaN the day).

Recovery does not need another full pull. All 376 `(variable, year)` partials
survive in `tmp/aorc_parts` (1.88 GB), so only the 5 affected years × 8
variables — 40 partials, ~11 % of the transfer — need re-streaming, then
`--assemble`.

## BCM (`bcm_region.py`)

The USGS **Basin Characterization Model v8** run on the CalSim3 Weather Generator
scenarios — 270 m, monthly, WY1916–2018, California — aggregated to the region
grid *and* to the CalSim3 catchment polygons. Two scenarios are ingested, the
pair that brackets a +3 °C warming response at the same precipitation:

| key | ScienceBase item | scenario |
|---|---|---|
| `s01` | `69e7d5c9b66b0164d0f72e91` | Scenario 1, Baseline |
| `s13` | `69e91d03b66b0183fe17a443` | Scenario 13, 0 % ave ppt, **+3 °C** |

Both are the *hydrology* child of the Part I release (parent
`68029987d4be0210cdcc98d1`, 24 scenarios × {climate, hydrology, WY summaries}),
carrying all six BCM variables — `aet`, `cwd`, `pck`, `rch`, `run`, `str` — every
one **in millimetres** per the release's FGDC metadata.

The source is the public ScienceBase S3 bucket (`publishedS3Uri` in the item
JSON; the `sciencebase.usgs.gov/manager` URLs need auth and the catalog HTML
403s a plain fetch). One zip per (variable, ~decade) holds one ESRI ASCII grid
per month, `run1915oct.asc`. The grid was **read from the files and confirmed
against the release metadata, not assumed**:

```
ncols 3486  nrows 4477  cellsize 270 m  NODATA -9999
xllcorner -374495.84  yllcorner -616153.31
California Teale Albers (m), NAD83, lon_0 -120, false northing -4e6  =  EPSG:3310
```

The corner is written to varying precision across variables (`run` gives
−374495.8364 where `ppt` gives −374495.84375, 7 mm apart), so `check_layout`
tests the origin to a 1 m tolerance and everything else exactly.

**Volume and why it streams.** Each `.asc` is 156 MB of text, so the two
scenarios are **112 GB zipped / ~2.3 TB inflated** — the inflated form does not
fit on disk. Nothing raw is written: each member is range-fetched from S3,
inflated in memory, reduced to 4410 + 386 numbers, and discarded. Local cost is
the partials (a few MB) plus the two stores.

Every `.asc` is *fixed-width* (10-char fields, 2 decimals, constant row stride),
which the reader asserts per file and then exploits: it gathers only the 2.29 M
cells that fall in the region and decodes them arithmetically, at 0.29 s per
grid. That leaves the pull **entirely network-bound** — measured 1.0 MB/s on a
single stream, 6.5 MB/s at 16–20 concurrent members, with no gain beyond that
(and the link is shared with any AORC pull running alongside).

**Resume.** Banked at two levels, both written atomically (temp file + rename):

- per month → `tmp/bcm_parts/_months/<scenario>_<var>_<zip>/<yyyymm>.npz`
- per zip → `tmp/bcm_parts/<scenario>_<var>_WY<a>_<b>.npz`, after which that
  zip's monthly checkpoints are deleted

```bash
python dataprep/bcm_region.py --list                       # the zip inventory
python dataprep/bcm_region.py --scenarios s01 --vars run --zips WY1916_19
python dataprep/bcm_region.py --all                        # everything, resumable
python dataprep/bcm_region.py --status                     # progress from disk
python dataprep/bcm_region.py --assemble                   # partials -> stores
```

As with AORC, run the full pull detached — `tmp/run_bcm_pull.ps1` (local-only)
launches it hidden, restarts it if the process dies, and assembles on a clean
exit.

**Aggregation.** Each BCM cell centre is projected to lon/lat and assigned to the
1/16° region cell containing it, and to the CalSim3 catchment polygon
(`calsim3.gpkg`, `CalSim3_And_GooseLake`) containing it: 2,289,972 cells reach
the 4410 region cells (median 519 each) and 2,111,973 reach the 386 catchments
(median 1506). Targets are the mean over *valid* cells, so the NODATA mask never
biases a total. Two catchment polygons are smaller than a single 270-m cell
(`EMD007` 0.015 mi², `EBP030` 0.038 mi²) and contain no cell centre; each gets
the nearest cell rather than a hole, and `n_bcm_cells` in the outputs shows it.

**Coverage.** 98.6 % of the mapped cells carry data. The shortfall is BCM's
open-water mask: 3673 of 4410 region cells are fully valid, the rest clip a lake
or reservoir, and **five cells have no data at all** — a contiguous blob at
41.84–42.03 N / −120.41…−120.47 W ringed by partially masked neighbours (0.04 →
0.83 valid), i.e. **Goose Lake**. Those five sit inside the endorheic block the
repo already excludes from anchors (the `no_gooselake` convention), so the hole
lands where it does no harm; they are `NaN` in the store, not zero.

**Outputs.** Per scenario, `bcm_<scen>_monthly.nc` (4410 cells × 1236 months ×
6 variables, `n_bcm_cells` alongside) and `bcm_<scen>_catchments_monthly.csv`
(386 × 1236, long by `cid`/`month`). Catchment identity is static, so it lives
once in `bcm_catchments.csv` (`cid, node, ct_name, type, sq_mi, n_bcm_cells`)
rather than being repeated 1236 times per catchment — that alone is the
difference between a 43 MB and a 25 MB series file. The series are LFS-tracked
by a rule of their own; every other tracked CSV in the repo is under 0.5 MB.

**Verification.** BCM is an independent model, so the gate is agreement, not
identity. Two checks, both against data already in the repo.

*Snowpack vs the four region SWE products* (1988–2018, regional monthly series).
The month convention was scanned rather than assumed, and three of the four peak
at lag 0, so `pck` needs no shift:

| product | r @ lag −1 | **r @ lag 0** | r @ lag +1 | obs mean SWE |
|---|---|---|---|---|
| `fldas` | 0.735 | **0.923** | 0.799 | 12.3 mm |
| `era5land` | 0.861 | **0.912** | 0.681 | 26.9 mm |
| `terraclimate` | 0.641 | **0.896** | 0.805 | 18.6 mm |
| `daymet` | **0.673** | 0.561 | 0.302 | 19.0 mm |

BCM's regional mean is 37.8 mm, above every product — expected, since 270 m
resolves high-elevation snow that an 11-km product smooths away — and it peaks
in February and melts out by August, as do ERA5-Land and FLDAS. **Daymet is not
a usable referee here**: its own climatology peaks in *April* and never melts
out (Aug 9.5 mm, Sep 6.7 mm, against ~0 for BCM and ERA5-Land), and its apparent
preference for lag −1 is an artifact of that distorted seasonality.

*Total discharge (`run` + `rch`) vs the `11obs` FNF targets*, area-weighted onto
each basin's HRU footprint over WY1922–2018:

| | median | range |
|---|---|---|
| monthly r | 0.882 | 0.82 (TNL) … 0.94 (BLB) |
| annual r | 0.959 | 0.84 (TNL) … 0.97 (AMF, TLG) |
| pbias | +6.4 % | −18.7 % (SHA) … +20.9 % (TNL) |

The two most negative basins are **SHA and BND**, which is the expected sign for
exactly those two: their HRU footprints carry the endorheic Goose Lake block,
~1000 mi² that never reaches the gauge, so a footprint-mean depth is diluted
against an FNF target. That is the same defect `catchments.SCREENED_BASINS`
exists to correct, arrived at here from an independent model. (Renormalising the
weights over the valid cells — the five NaN cells are 0.95 % of SHA's weight and
0.71 % of BND's — moves the bias by under a point, so the shortfall is real
rather than a NaN-handling artifact.)

*Warming signature*, Scenario 13 − Scenario 1, regional. Fluxes as mm/yr, the
two state variables as mean storage:

| | S1 baseline | S13 (+3 °C) | |
|---|---|---|---|
| `pck` (mean SWE) | 38.7 mm | 14.0 mm | **−63.9 %** |
| `cwd` | 855 mm/yr | 917 mm/yr | +7.4 % |
| `run` | 129 mm/yr | 138 mm/yr | +7.1 % |
| `aet` | 359 mm/yr | 367 mm/yr | +2.1 % |
| `rch` | 165 mm/yr | 163 mm/yr | −1.1 % |
| `str` (mean storage) | 394 mm | 386 mm | −2.0 % |

Precipitation is unchanged between the two scenarios, so this is a pure
temperature response: snowpack collapses, deficit and soil drying rise, and
runoff *increases* slightly because rain that once fell as snow runs off in
winter instead of infiltrating as spring melt (`run` +7.1 % against `rch`
−1.1 %; total discharge +2.5 %).

## Verification

Every local ingest reproduces its committed or legacy predecessor before it lands. `local_obs_region.py --verify` reproduces the legacy 2074-cell npz (rel RMS < 1e-3, achieved 1e-7); `wgen_forcing.py --verify` reproduces the committed forcing stores from the master; `build_region_forcing.py` re-passes the SAC-SMA parity gate for every domain with a simflow reference (KGE > 0.9999).

**The GEE products are the exception.** The reproduce-the-snapshot gate failed: ERA5-Land shows genuine asset drift (GEE reprocesses its assets and the original pipeline is lost), so the snapshot is irreproducible in principle. The region GEE store is therefore **its own spec**: a cell-rectangle mean at each asset's native scale, with the asset versions recorded in each npz's `meta`. `gee_obs_region.py --verify` stays on as a drift *report* against the legacy snapshot, not a gate. At the level training consumes (15-basin monthly climatologies) the drift is small (ET rel RMS 1.1%, SWE 4.1%, snowy-basin mask unchanged), so anything built on the old snapshot re-runs cleanly on this basis. The frozen legacy npz stay as the record of what the pre-region models trained on.

## GEE export runbook (user-run)

Project = `ee-warnold` (EE-registered):

```bash
python dataprep/gee_obs_region.py --products all --project ee-warnold   # region burn, hours
python dataprep/gee_obs_region.py --verify --project ee-warnold         # optional drift report
```

Outputs land in `data/region/{et_obs,swe_obs}/*.npz` (LFS). The `dpl/data.py` `ET_DIR`/`SWE_DIR` defaults point at the in-repo store; `SACSMA_ET_DIR` / `SACSMA_SWE_DIR` override them to a frozen local snapshot.

## New-basin setup

A basin inside the region needs only a delineation and a gage/FNF target:

1. **Cells.** Select from `grid_cells.csv`, or intersect the delineation with the 1/16° grid.
2. **Forcing.** `python dataprep/wgen_forcing.py --cut <name> --cells <csv> --out-dir <dir>` writes `historical_livneh_unsplit_<name>.nc` (prcp + tavg) plus a per-cell tmin/tmax sidecar, with the ×10 artifact days corrected by default.
3. **Statics.** Rows from `{soilveg_continuous,lai_climatology}.csv` (currently the 2480 modeling-domain cells; footprint-only cells await the raster ingest).
4. **Obs losses.** The region npz cover the cells via the `data.py` defaults.

## Known gap

The statics stores cover only the 2480 modeling-domain cells. The 1930 footprint-only cells need a raster ingest (POLARIS / LANDFIRE / 3DEP / MODIS-LAI, see [`../data/raw_gis/SOURCES.md`](../data/raw_gis/SOURCES.md)), gated on reproducing the committed point-sample rows, before any full-region training.

The rasters themselves are **staged and complete** — 6261 files / 89 GB at `D:\sacsma-data\raw_gis`, off the repo drive. Point `SACSMA_RAW_GIS` at it (`data/raw_gis/sample_gis.py` still hard-codes the in-repo path, so it needs the tree there or an edit). `python dataprep/download_gis.py --status` inventories the stage; `--all` re-fetches anything missing, resumably. So the remaining work is the raster→cell aggregation, not the download.
