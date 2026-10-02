# data/reference/et: evapotranspiration reference products on the 1/16° grid

Nine evapotranspiration (ET) products, each reduced to the 4,410 cells of the grid
(`data/inputs/grid/grid_cells.csv`). Role: reference. These are reference observations: no code
in the `sacsma` package reads them. By the project's rule, ET observation products are never
used in a loss, in model selection, as priors, or in design (see
[Conventions](../../../docs/conventions.md)). All nine come from outside the repository; the
three scripts in this folder reduce them to the grid.

## Files

| File | What | Source |
|---|---|---|
| `gleam_cell_monthly.npz` | GLEAM v4.3a monthly `E`, mm/month, 1988-01 to 2018-12. Nearest-neighbour sample of the 0.1° global grid at the cell centre | `local_obs_region.py` |
| `fluxcom_cell_monthly.npz` | FLUXCOM RS_METEO CRUNCEP v8 monthly `LE` (0.5°, MJ m-2 d-1), nearest-neighbour sample. ET mm/month = LE / 2.45 × days in month. 1988-01 to 2016-12 (the product ends in 2016) | `local_obs_region.py` |
| `terraclimate_gee_cell_monthly.npz` | TerraClimate `aet` (`IDAHO_EPSCOR/TERRACLIMATE`, × 0.1), reduced at 4638 m, 1988-01 to 2018-12 | `gee_obs_region.py`, exported 2026-07-16 |
| `fldas_gee_cell_monthly.npz` | FLDAS Noah `Evap_tavg` (`NASA/FLDAS/NOAH01/C/GL/M/V001`, kg m-2 s-1 × 86400 × days in month), 11132 m, 1988-01 to 2018-12. NaN ≤0.05 % (FLDAS land mask) | `gee_obs_region.py`, exported 2026-07-16 |
| `era5land_gee_cell_monthly.npz` | ERA5-Land `total_evaporation_sum` (`ECMWF/ERA5_LAND/MONTHLY_AGGR`, m × −1000), 11132 m, 1988-01 to 2018-12 | `gee_obs_region.py`, exported 2026-07-16 |
| `openet_gee_cell_monthly.npz` | OpenET ensemble `et_ensemble_mad` (`projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0`), 30 m mosaic, 1999-10 to 2024-12, NaN 0 | `gee_obs_region.py`, exported 2026-07-17 |
| `modis_gee_cell_monthly.npz` | MOD16A2GF `ET` (`MODIS/061/MOD16A2GF`), 8-day ET summed by month × 0.1, 500 m, 2000-01 to 2025-12, NaN 0.14 % (unvegetated) | `gee_obs_region.py`, exported 2026-07-17 |
| `reitz2023_cell_annual.npz` | Reitz, Sanford and Saxe (2023), *Historical Evapotranspiration for the Conterminous U.S.* (USGS, doi:10.5066/P9EZ3VAS; WRR 59, e2022WR034012): water-year annual actual ET from the 30-arcsec rasters, mm/yr, WY1896–2018, NaN 0 | `reitz_et.py` |
| `reitz2023_cell_monthly.npz` | The same release's monthly maps: calendar months 1990-01 to 2018-09 (345), mm/month, NaN 0 | `reitz_et.py` |

All files are in LFS and cover all 4,410 cells. Each holds the arrays `keys` (cell keys),
`dates` (month start), `et` (cells × months, float32, mm/month), `lat` and `lon`. The Earth
Engine and Reitz files add `meta` (a text note of source and reduction). The Reitz annual file
differs: `et` is mm per water year, `wy` is the water year, and `dates` is the water-year start.

## How it is built

GLEAM and FLUXCOM: `local_obs_region.py`, `sacsma` environment. It reads the raw downloads
`gleam/v4.3a_monthly_E` and `fluxcom/RS_METEO_CRUNCEP_v8_LE_monthly` under the folder named by
`staging` in `data/local_paths.toml`. Run the check first; it must pass.

```bash
python data/reference/et/local_obs_region.py --verify
python data/reference/et/local_obs_region.py
```

Earth Engine products: `gee_obs_region.py`, `sacsma-gis` environment. It needs an authenticated
`earthengine-api` with a registered cloud project (`earthengine authenticate`; `--project`
defaults to `gee_project` in `data/local_paths.toml`). One run writes the ET files here and the
SWE files to `data/reference/swe/`. `--products all` covers the three ET and four SWE products
of 1988–2018 and takes a few hours; OpenET and MODIS run only when named.

```bash
python data/reference/et/gee_obs_region.py --products all --project <id>
python data/reference/et/gee_obs_region.py --products openet modis --project <id>
python data/reference/et/gee_obs_region.py --verify --project <id>   # optional drift report
```

Each value is the mean over the cell rectangle at the asset's native scale. A coarser scale
aliases the mean. On a network that inspects TLS, Earth Engine needs the Windows trust store
(`import pip._vendor.truststore as t; t.inject_into_ssl()` before `import ee`);
`gee_obs_region.py` does not inject it yet.

Reitz: `reitz_et.py`, `sacsma-gis` environment (rasterio). The release is staged by hand in the
folder named by `reitz_et` in `data/local_paths.toml` (default `tmp/reitz2023_et`). The
`*_monthly.zip` and `Irrigation_*_1980-2018.zip` files are ScienceBase cloud files (login, or a
captcha and e-mail request); the rest of the release was fetched with a helper script that is
not tracked. The monthly zips are Deflate64, which the script reads through 7-Zip (`SACSMA_7Z`,
else `7z` on PATH, else the default install location).

```bash
python data/reference/et/reitz_et.py --status
python data/reference/et/reitz_et.py --cut [--delete-source]
python data/reference/et/reitz_et.py --ingest
```

`--cut` windows every CONUS raster to California and writes DEFLATE GeoTIFFs to `<stage>/ca/`
(local only; 1.6 GB for 43 GB raw). `--ingest` reduces them to the cells with an exact-overlap
mean over the cell rectangle (cos-latitude row weights). All three scripts take `--data-dir`
(default: the repository's `data/`).

## Checks

- GLEAM, FLUXCOM: `--verify` reproduces the earlier 2,074-cell stores (`et_processed/` under
  `staging`). Required rel RMS < 1e-3; achieved 1e-7.
- Earth Engine products: the files are their own specification. Reproducing the earlier
  snapshot failed because Earth Engine reprocesses its assets and the snapshot's pipeline is
  lost (ERA5-Land: rel RMS about 0.2 under every reduction tried). `--verify` is therefore a
  drift report, not a gate: it exports the 2,074 cells of the earlier stores (`et_processed/`
  and `swe_processed/` under `staging`) and prints the difference. On 15-watershed monthly
  climatologies the drift is small: ET rel RMS 1.1 %, SWE 4.1 %.
- Reitz cut: every cut raster is verified bit-exact against the source window.
- Reitz annual: the cell mean agrees with a 600 × 600 supersample to 4e-7 m/yr.
- Reitz monthly (2026-09-23): each file month is a calendar month (the band description is its
  water year, `AET_1990_10` is `wy1991`, 345 of 345). Every complete water year WY1991–2018
  sums to the annual file to ≤2.4e-7, and `--ingest` refuses to write otherwise (a
  water-year-indexed or mm/month reading misses by 4–97 %). An independent supersample of the
  raw rasters agrees to ≤3e-5.

## Know before using

- Spans differ. OpenET and MODIS are kept at their full available span; they start after 1999
  and do not cover the WY1989–2003 calibration window. FLUXCOM ends in 2016.
- GLEAM and FLUXCOM are point samples of coarse grids; the Earth Engine and Reitz files are
  cell means.
- Earth Engine assets drift. A new export will not reproduce these files exactly; `meta`
  records the export date, the collection and band, and the scale.
- Reitz annual, what it is: a five-equation BMA ensemble (Schreiber, Budyko, Fu-Zhang with
  Hamon PET, Reitz-2017, Brutsaert-Stricker) whose weights were trained on annual
  `P − Q + ΔS_gw` at 1,858 Gages-II watersheds and spread by random forest. It is informed by
  the water balance, not closed on it, and is driven by PRISM.
- Caution (Reitz annual): irrigation supply enters only from WY1980. The 1,930 cells that are
  only in the CalSim3 footprint (valley) step +104 % at 1980 (257 to 523 mm/yr) while the 2,480
  modeling-domain cells move +3 %. Never difference across 1980 on the valley floor.
- Caution (Reitz): 18 of the 69 gauges of `data/targets/usgs/` are in its training set.
- Reitz monthly, what it is: the month-to-month pattern of one equation (Fu-Zhang with Hamon
  PET on same-month precipitation, no storage carry-over) rescaled to the annual ensemble. It
  carries no observed seasonal signal. Values are the rasters' mm/day × days in month (29-day
  leap Februaries). Only the three 1990–2018 monthly zips were fetched; the nine earlier ones,
  back to October 1895, were not. WY1990 is partial (January to September).
- Caution (Reitz monthly): summer ET follows same-month rain. 8.6 % of cell-months are exactly
  0 (29 % of Julys, 32 % of Augusts; TerraClimate 2.0 %, FLDAS 0.3 %, the others about 0).
  Above 2000 m in the Sierra, 16.6 % of July–September cell-months are 0, and the
  June–September interannual SD is 17–25 mm against 3–5 mm for GLEAM, FLUXCOM and OpenET. It
  is not a month-level ET reference in snow terrain.
- Caution (Reitz monthly): seasonal shape against the median of GLEAM, FLUXCOM, TerraClimate,
  FLDAS and ERA5-Land (WY1991–2016): September 1.9×, October 1.6×, January–April 0.63–0.74×.
  The annual 469 mm/yr sits inside their 410–564 spread.
- The whole Reitz monthly record is after 1980, so irrigation supply is in throughout (valley
  floor WY2001–18: 616 mm/yr against OpenET 750).

## Read by

No module of the `sacsma` package. `sacsma.paths.et_obs` names the folder for the build scripts.
