# data/inputs/grid: the 1/16° region grid and its static attributes

The cell list every grid-based store is built on, and what is known about each cell that does not change
in time: soil, vegetation, terrain, a leaf-area climatology and a satellite embedding. Role: input. Built
in this repository from public rasters (POLARIS, LANDFIRE, 3DEP, MODIS) and from Google Earth Engine.

## Files

| File | What | Source |
|---|---|---|
| `grid_cells.csv` | The 4,410 cells of the 1/16° Livneh grid: `key, lat, lon, in_15cdec_grid, in_9unimp, in_11obs, in_12rim, in_calsim3_fp`. The union of the four grid domains (2,480 cells) and the full CalSim3 footprint: every cell whose rectangle intersects a polygon of `data/inputs/calsim3/calsim3.gpkg` (both layers: all 215 Rim watersheds, Goose Lake included, and the 170 Valley polygons), which adds 1,930 cells. Keys are `<lat>_<lon>` normalized to 5 decimals. | `build_region_grid.py` |
| `soilveg_continuous.csv` | Per-cell soil, vegetation and terrain attributes, all 4,410 cells, no blank field. Columns: `sand`, `clay`, `ksat`, `theta_s` at six depths each (`_0_5` to `_100_200`), `polaris_gapfill`, `EVC`, `EVH`, `EVC_cover_pct`, `EVH_height_m`, `dem_elev`, `slope_deg`, `aspect_sin`, `aspect_cos`, `curvature`, `relief_m`, `lai_mean`, `lai_min`, `lai_max`, `lai_amp`, `lai_peak_doy`, and `src` (where the row came from). | `build_region_statics.py` |
| `lai_climatology.csv` | Per-cell MODIS leaf-area day-of-year climatology, 46 8-day values (`lai_doy001` to `lai_doy361`), and `src`. Same cells and consolidation. | `build_region_statics.py` |
| `aef_cell_mean.npz` (1.1 MB, LFS) | AlphaEarth Foundations satellite embeddings as one static 64-value vector per cell. Arrays: `keys`, `lat`, `lon`, `bands`, `years`, `dataset_version`, `emb` (4410 x 64, float32), `norm`, `n_years`, `valid_frac` (4410 x 9), `year_cos_min`, `meta`. | Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, `gee_aef_region.py` |
| `soilveg_continuous_raster.csv`, `lai_climatology_raster.csv` (not tracked) | Every grid cell sampled at its centre from the raw rasters. An intermediate: its content is folded into the two tracked tables, and nothing else reads it. | `sample_gis.py region` |
| `raw_gis/` (not tracked) | The default stage of the raw rasters, 89.5 GB. Normally kept on another drive and named by the `raw_gis` key of `data/local_paths.toml`. | `download_gis.py` |

## The raw rasters

Continuous soil, vegetation and terrain rasters, California and CONUS products in preference to global
ones. They give the dPL parameter network physical quantities in place of the `soil_class` and
`veg_class` codes of the HRU tables. Staged for lat 32 to 42 N, lon -125 to -114 W (110 one-degree
tiles), plus one tile at 42 to 43 N / -121 to -120 W, where the grid reaches 42.4 N at the Goose Lake
extension of BND (50 cells, all in `11obs` and `12rim`). 6,288 files: POLARIS 2,304 tiles / 55.1 GB, LANDFIRE 222 / 5.7 GB,
3DEP 90 / 3.8 GB, MODIS LAI 3,672 granules / 24.9 GB. POLARIS and 3DEP are CONUS-land products, so ocean
tiles return 404 and are skipped (95 and 89 land tiles in the original extent). LANDFIRE renders all 110.

| Product | What is staged | Units and encoding | Source and access | Layout under the stage root |
|---|---|---|---|---|
| POLARIS soil (Chaney et al. 2019, doi:10.1029/2018WR022797) | 30 m, CONUS, continuous and gap-free remap of SSURGO. `sand`, `clay`, `ksat`, `theta_s`, statistic `mean`, six depths (0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm). Staged by property, so other variables (silt, bd, om, theta_r, van Genuchten n/alpha/hb/lambda, pH) and statistics (mode, p5, p50, p95) can be added without a re-fetch. | sand, clay: %. `theta_s`: m³/m³ (0.35 to 0.76). `ksat`: log10(cm/hr), so a depth-weighted arithmetic mean of the stored value is the depth-geometric mean of conductivity. Nodata over open water. Tiles 3600 x 3600, EPSG:4326, exact 1° bounds, 1/3600° pixel, float32, nodata -9999. | `http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/<prop>/mean/<depth>/lat{S}{N}_lon{W}{E}.tif`. Plain HTTP, no login. | `polaris/PROPERTIES/v1.0/<prop>/mean/<depth>/lat{S}{N}_lon{W}{E}.tif` |
| LANDFIRE 2024 vegetation: Existing Vegetation Cover (EVC) and Height (EVH) | 30 m, CONUS, coded rasters banded by lifeform. | Bands: 100s = tree, 200s = shrub, 300s = herb; below 100 = water, developed, agriculture, sparse. The last two digits are cover % (EVC) or a height index (EVH). Height: tree `(v-100)` m, shrub `(v-200)*0.1` m, herb `(v-300)*0.1` m. Water class 11 is a valid code and decodes to cover 0. Tiles 3600 x 3600, EPSG:4326, int16, no nodata. | USGS ImageServer `exportImage`, no login: `https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/LF2024_{EVC,EVH}_CONUS/ImageServer`, with `bbox=<w>,<s>,<e>,<n>&bboxSR=4326&imageSR=4326&size=3600,3600&format=tiff&pixelType=S16&interpolation=RSP_NearestNeighbor&f=image`. | `landfire/{EVC,EVH}/{EVC,EVH}_lat{S}{N}_lon{W}{E}.tif` |
| MODIS MCD15A2H.061 leaf area index | 500 m, 8-day, combined Terra and Aqua, 2003 to 2022. Sinusoidal tiles h08v04, h08v05, h09v04, h09v05; 918 granules per tile. | Valid LAI 0 to 100, scale 0.1; values above 100 are fill (cloud, water, unfilled). Analytic sinusoidal georeference: R = 6371007.181 m, tile 1111950.5197 m, x0 = -20015109.354, y0 = 10007554.677, 2400 x 2400 per tile, pixel 463.3127 m. | NASA CMR granule API (`cmr.earthdata.nasa.gov/search/granules.json`), then the LP DAAC cloud bucket (`data.lpdaac.earthdatacloud.nasa.gov`) with an Earthdata login in `.netrc`. | `lai/mcd15a2h/<tile>/MCD15A2H.A{yyyy}{ddd}.<tile>.061.*.hdf` |
| USGS 3DEP 1 arc-second elevation | About 30 m seamless DEM. Gives slope, aspect (sin, cos), curvature (Laplacian) and relief (windowed elevation standard deviation). | Kept as published: tiles 3604 x 3604, EPSG:4269, 1° plus a 2-pixel overlap per side, float32, nodata -999999. | Public AWS `s3://prd-tnm/StagedProducts/Elevation/1/TIFF/current/` over HTTPS, no login. Tiles `n{N}w{W}` (north-edge latitude, west-corner longitude magnitude). | `dem/3dep_1as/USGS_1_n{N}w{W}.tif` |

Not fetched, optional: gNATSGO (USDA NRCS 30 m composite of SSURGO, STATSGO2 and RSS, a cross-check for
POLARIS) and the NLCD tree canopy cover (MRLC 30 m, overlaps LANDFIRE EVC).

## How it is built

From the repository root. `sample_gis.py` and `gee_aef_region.py` run in `sacsma-gis`
(`environment-gis.yml`); the others in `sacsma`.

```bash
python data/inputs/grid/build_region_grid.py              # -> grid_cells.csv (needs the wgen_ascii key)
python data/inputs/grid/download_gis.py --status          # inventory of the stage, no network
python data/inputs/grid/download_gis.py --all --jobs 8    # fetch what is missing (or --layers polaris)
python data/inputs/grid/sample_gis.py region              # every cell -> the two *_raster.csv
python data/inputs/grid/sample_gis.py 11obs               # a domain's sidecars (15cdec, 9unimp, 11obs, 12rim)
python data/inputs/grid/build_region_statics.py           # -> soilveg_continuous.csv, lai_climatology.csv

python data/inputs/grid/gee_aef_region.py --project <ee-project>     # the pull, 1 to 2 h, resumable
python data/inputs/grid/gee_aef_region.py --status
python data/inputs/grid/gee_aef_region.py --project <ee-project> --check 30 --check-year 2021
python data/inputs/grid/gee_aef_region.py --assemble                 # -> aef_cell_mean.npz
```

- The grid. Domain cells come from the `hruinfo.csv` of `15cdec_grid`, `9unimp`, `11obs` and `12rim`.
  Every cell must exist in the WGEN NonDetrend-Unsplit ASCII store, the source of the forcing master: a
  missing domain cell stops the build, and footprint cells absent from that land-only store are dropped
  (14 Delta and open-water cells).
- The stage. `download_gis.py` needs only `requests`. MODIS needs a `urs.earthdata.nasa.gov` entry in
  `.netrc` (`_netrc` on Windows). Files are written to a `.part` name and renamed when complete, so a rerun
  resumes. LANDFIRE is requested at `size=3600,3600` over a 1° box on purpose: it lands on the POLARIS
  pixel grid.
- The sampler. Each HRU or cell is a point, and the rasters are sampled there. Terrain derivatives use a
  window around the point, because one 30 m slope is noisy. Water is gap-filled and flagged: POLARIS nodata
  takes the nearest finite pixel in an 81-pixel window (`polaris_gapfill`), and an LAI pixel that is always
  fill (open water, permanent snow) borrows the nearest valid point's climatology (`lai_gapfill`). MODIS is
  read with pyhdf. The script is not importable by the `sacsma` package. `sample_gis.py <domain>` writes
  the per-HRU sidecars of a calibration domain (see [the domains README](../domains/README.md));
  `--layers polaris,landfire,terrain,lai` limits the layers.
- The consolidation. `build_region_statics.py` fills each cell from the first source that has it: the
  `15cdec_grid` sidecar (`src` = `cdec15_grid`, 2,074 cells), then the `11obs`, `12rim` and `9unimp`
  sidecars (`calsim_11obs` 203, `calsim_12rim` 4, `calsim_9unimp` 199), then the raster sample
  (`region_raster`, 1,930). It needs the untracked `*_raster.csv` files to cover the 1,930 footprint-only
  cells. Regenerate them only if the grid changes.
- Earth Engine. `gee_aef_region.py` needs an authenticated `earthengine-api` and a registered cloud project
  (`--project`, default the `gee_project` key). Behind a TLS-inspecting proxy the Windows trust store must
  be injected before `import ee` (`import pip._vendor.truststore as t; t.inject_into_ssl()`); the script
  does it itself. Each (year, unit) is banked atomically in `tmp/aef_parts` with its cell keys, empty
  results are refused, `run.json` pins `--chunk` and `--scale`, `run.lock` blocks a second pull, and a
  failed unit gives a non-zero exit, so a rerun fetches only what is missing. The pull costs about 140
  EECU-hours (the noncommercial Community tier allows 150 per month).
- The embedding. Each calendar year's 10 m pixel vectors are averaged over the cell rectangle, then the
  nine years 2017 to 2025 are averaged with equal weight. No `mosaic()`: the -120° line between UTM zones
  10N and 11N is a cell edge, so each cell is summed on its own zone's tiles (28 per year: 16 in 10N, 12 in
  11N) in native UTM; the other zone's pixels past -120° (about 84 m, slightly different) are never used.
  At 10 m this equals the zone mosaic's mean to 1e-12 for about a third less compute. The reduction runs at
  15 m, a nearest-neighbour lattice of the full-resolution level (4/9 of the pixels), at 12.8 instead of
  23.3 EECU-seconds per cell-year.

## Checks

- The stage is reproduced byte for byte by the downloader: SHA-256 of a re-fetched tile of each no-login
  product (POLARIS `sand/mean/0_5`, LANDFIRE `EVC`, 3DEP), the rendered LANDFIRE tile included.
- MODIS count: 918 granules per tile, checked against CMR. The window holds 920 8-day steps; CMR adds the
  composite starting 2002-12-27 (`A2002361`) and three are absent upstream (`A2016049`, `A2022097`,
  `A2022289`).
- Units were verified at sample time; tile geometry was read off the stage.
- The raster sample reproduces the point-sampled CalLite sidecars: 2,021 of 2,026 shared cells exactly. The
  5 others differ only in `lai_mean`: they are always-fill LAI cells, and the full grid offers a denser
  pool of neighbours to borrow from than one domain's HRU set.
- Statics coverage: 4,410 of 4,410 cells, zero blank fields.
- Domain cell sets from the `hruinfo` tables equal the key sets of the per-domain forcing stores of the
  time (2026-07-16).
- AlphaEarth: 4,410 cells, 9 of 9 years, `valid_frac` 1.0 everywhere (lakes, reservoirs and bays are
  embedded, not masked). The summed pixel weight equals the full-cell count to 1e-15 on tile corners and on
  the -120° cells. Against the full 10 m mean the 15 m lattice gives ≤ 2e-4 per band (≤ 9e-5 on tile-corner,
  zone-edge, lake and grid-extreme cells; 1.6 to 1.9e-4 on 90 random cells), cosine ≥ 0.9999998, length
  unchanged to ≤ 6e-5. `--check` passed on 90 random cells in 2017, 2021 and 2025.

## Know before using

- Two sampling conventions meet in the statics tables, marked by `src`. The `cdec15_grid` rows are means
  over the cell footprint (the convention the dPL parameter network was trained on, so they win where
  available); all other rows are point samples at the cell centre. Over shared cells the median difference
  in `dem_elev` is about 93 m.
- Depth aggregation is a modelling choice of `sacsma.dpl.features`, not of the sampler. Its `physical`
  variant collapses the six POLARIS depths into two zones, 0 to 30 cm and 30 to 200 cm (depth-weighted),
  and encodes `lai_peak_doy` as sin and cos. The six raw columns stay in the table.
- caution: LANDFIRE reports a failed request as HTTP 200 with a JSON body, so a size check alone does not
  catch it.
- caution: name each MODIS file from the URL basename. The collection's `producer_granule_id` omits
  `.hdf`, which hides the file from the sampler and defeats resume. The GDAL 3.12 HDF4 plugin fails under
  rasterio (DLL error 126); the classic e4ftl01 archive is dead.
- Topographic wetness index is not computed (it needs flow accumulation).
- AlphaEarth vectors are not unit length after averaging. `norm` (0.65 to 0.95, median 0.82) is low where a
  cell is mixed. `year_cos_min` (lowest cosine of a year's cell mean against `emb`, median 0.956) flags
  cells that changed between years: mostly surface water (the Tulare Lake bed reflooding in 2023, cosine
  0.47; Goose Lake; Colusa rice fallowing), also burn scars. A multi-year mean blends those states.
- caution: Earth Engine pyramid levels of this collection at 20 m and coarser are L2-renormalized block
  means, despite `pyramidingPolicy: MEAN` in the asset metadata. Reducing at 19 to 20 m inflates the mean's
  length by about 4e-3, and 16 m aliases (1e-3). `--scale` accepts only 10 or 15.
- AlphaEarth layers are calendar years, not water years, and all post-date the WY1989-2003 calibration
  window. `dataset_version` is 1.1 for 2017 and 2025, 1.0 for 2018 to 2024. Per-year cell means exist only
  in the local partials.
- What the embedding carries: PC1 (44 % of variance) tracks elevation (r -0.88), PC3 tracks LAI (+0.70).
  The cosine between east neighbours is 0.96, between random pairs 0.47.
- AlphaEarth licence, CC-BY 4.0: "The AlphaEarth Foundations Satellite Embedding dataset is produced by
  Google and Google DeepMind." Reference: Brown et al. (2025), arXiv:2507.22291.

## Read by

- `grid_cells.csv`: every build script that works on the grid (forcing, statics, the AlphaEarth pull, the
  ET, SWE and BCM references, the entity cells), and `sacsma.dpl.calsim.tier2`.
- `soilveg_continuous.csv`, `lai_climatology.csv`: `sacsma.io.soilveg_path` and `lai_climatology_path` for
  the `multifamily` domain, through them `sacsma.dpl.features` (attributes), `sacsma.io.load_hru_table`
  (`dem_elev` as cell elevation) and the canopy inputs of the Noah ET path (`EVC_cover_pct`, LAI).
- `aef_cell_mean.npz`: `sacsma.dpl.features`, for the `aef_u` inputs only (`6_aef` and `7_calsim`).
