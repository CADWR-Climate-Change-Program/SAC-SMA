# Raw California/CONUS GIS — provenance & manifest

Continuous soil, vegetation, and terrain rasters staged locally, for two purposes:
to replace the opaque one-hot `soil_class`/`veg_class` HRU features with physical
continua for the dPL parameter net (the `physical` feature variant), and to let the
domain extend from the 15 CDEC basins to all of California without re-downloading.
The preference is high-quality California/CONUS products over global ones. This tree
is gitignored (`data/raw_gis/*`); only this provenance doc, the reusable HRU sampler
(`sample_gis.py`), and the small per-HRU derived CSVs
(`data/<app>/soilveg_continuous*.csv`, `lai_climatology*.csv`) are committed.

**California extent fetched:** lat 32–42 N, lon −125 to −114 W (1° tiles;
ocean/out-of-CONUS tiles are skipped). Downloaders: `scratchpad/download_*.py`
(git history). Sampler: `sample_gis.py` (runs in the `sacsma-gis` conda env —
rasterio + pyhdf + pyproj; NOT importable by the core `sacsma` package).

**Staged (89 GB total):** POLARIS 2280 tiles / 55 GB · LANDFIRE 220 / 5.7 GB ·
3DEP 89 / 3.8 GB · MODIS LAI 3672 granules / 25 GB.

---

## Soil — POLARIS  ✅ staged
- **What:** 30 m, CONUS, continuous & gap-free ML remap of SSURGO. Minimal set
  parked: **sand, clay, ksat, theta_s** × `mean` × 6 depths (0-5, 5-15, 15-30,
  30-60, 60-100, 100-200 cm). Staged by-property so more variables/statistics
  (silt, bd, om, theta_r, van Genuchten n/alpha/hb/lambda, pH; mode/p5/p50/p95)
  can be added later without re-fetching.
- **Units (verified at sample time):** sand/clay **%**; `theta_s` **m³/m³**
  (0.35–0.76); `ksat` **log10(cm/hr)** (so a depth-weighted arithmetic mean of
  the stored value is the depth-*geometric* mean of conductivity — what the
  sampler does). Nodata over open water (reservoirs/lakes).
- **Source:** `http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/<prop>/mean/<depth>/lat{S}{N}_lon{W}{E}.tif`
  (Chaney et al. 2019, *WRR*, doi:10.1029/2018WR022797). Plain HTTP, no auth.
- **Layout:** `polaris/PROPERTIES/v1.0/<prop>/mean/<depth>/lat{S}{N}_lon{W}{E}.tif`

## Vegetation structure — LANDFIRE 2024 (EVC + EVH)  ✅ staged
- **What:** 30 m CONUS. Existing Vegetation Cover (EVC) + Height (EVH) — the
  continuous structural veg features.
- **Encoding (verified):** coded rasters, banded by lifeform. EVC/EVH last two
  digits = cover % (EVC) or a height index; bands 100s=Tree, 200s=Shrub,
  300s=Herb; <100 = water/developed/agriculture/sparse. Height decode:
  Tree `(v−100) m`, Shrub `(v−200)·0.1 m`, Herb `(v−300)·0.1 m`. Water class 11
  is a valid code (no nodata) → cover 0.
- **Source:** USGS ArcGIS ImageServer `exportImage` (bbox-clipped, no auth):
  `https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/LF2024_{EVC,EVH}_CONUS/ImageServer`
- **Layout:** `landfire/{EVC,EVH}/{EVC,EVH}_lat{S}{N}_lon{W}{E}.tif`

## LAI — MODIS MCD15A2H.061  ✅ staged
- **What:** 500 m, 8-day combined Terra+Aqua Leaf Area Index, **2003–2022**. Two
  uses: a QC-masked day-of-year **climatology** (46 8-day bins) as veg features,
  and the full climatology as the future Noah-ET canopy driver. CA sinusoidal
  tiles h08v04, h08v05, h09v04, h09v05.
- **Access:** NASA CMR granule API (`cmr.earthdata.nasa.gov/search/granules.json`)
  → LP DAAC protected cloud bucket (`data.lpdaac.earthdatacloud.nasa.gov`) via
  Earthdata `.netrc` (URS OAuth). The classic e4ftl01 HTTPS archive is dead.
- **Georef (analytic sinusoidal):** R=6371007.181 m; tile 1111950.5197 m;
  x₀=−20015109.354, y₀=10007554.677; 2400²/tile, pixel 463.3127 m. Valid LAI
  0–100 (×0.1); values >100 are fill (cloud/water/unfilled). Read with **pyhdf**
  (the GDAL 3.12 modular HDF4 plugin fails under rasterio, DLL err 126).
- **Layout:** `lai/mcd15a2h/<tile>/MCD15A2H.A{yyyy}{ddd}.<tile>.061.*.hdf`

## Terrain — USGS 3DEP 1 arc-second  ✅ staged
- **What:** ~30 m seamless DEM → slope, aspect (sin/cos), curvature (Laplacian),
  relief (windowed elevation std). TWI deferred (needs flow accumulation).
- **Source:** public AWS `s3://prd-tnm/StagedProducts/Elevation/1/TIFF/current/`
  (HTTPS, no auth), 1° tiles `n{N}w{W}` (north-edge lat, west-corner lon mag.).
- **Layout:** `dem/3dep_1as/USGS_1_n{N}w{W}.tif`

## Soil — gNATSGO (authoritative companion)  ⬜ optional, not yet fetched
- USDA NRCS 30 m gap-free composite (SSURGO+STATSGO2+RSS); a cross-check for
  POLARIS. Microsoft Planetary Computer COGs (`gnatsgo_rasters`) or soilDB WCS.

## Land cover — NLCD Tree Canopy Cover  ⬜ optional, not yet fetched
- MRLC 30 m % tree canopy (overlaps LANDFIRE EVC).

---

## HRU sampler (`sample_gis.py`) & derived products

`python sample_gis.py <domain> [--layers polaris,landfire,terrain,lai]` (in the
`sacsma-gis` env) samples every layer at each HRU point and writes, per domain
`<d>` ∈ {15cdec, 9unimp, 11obs, 12rim}:

- `data/<app>/soilveg_continuous<sfx>.csv` — one row per HRU (`hruinfo` order,
  keyed non-uniquely by `key`); the **durable RAW** values: POLARIS 4 props × 6
  depths, LANDFIRE EVC/EVH + decoded cover%/height, 3DEP elev/slope/aspect
  sin·cos/curvature/relief, LAI mean/min/max/amplitude/peak-DOY, plus
  `polaris_gapfill` / `lai_gapfill` provenance flags. (`<sfx>` = `` for 15cdec,
  `_<d>` for the calsim domains — matches `sacsma.io._sfx`.)
- `data/<app>/lai_climatology<sfx>.csv` — HRU × 46 8-day DOY LAI climatology
  (the Noah-ET canopy driver).

**Sampling conventions.** Each HRU is a point (its lat/lon), and rasters are
sampled there. Terrain derivatives use a windowed neighbourhood for sub-grid
roughness rather than a single noisy 30 m slope. Water bodies are gap-filled:
POLARIS nodata over reservoirs is backfilled with the nearest finite pixel in an
81-px window (a reservoir HRU's true soil is the surrounding land), and LAI pixels
that are always fill (open water or permanent snow) borrow the nearest valid HRU's
climatology. Both are flagged. For 15cdec, 35 HRUs are POLARIS-gap-filled and 239
LAI-gap-filled; otherwise the sample is complete (7891/7891 finite, no NaN).

**Depth aggregation** is a modelling choice made in `sacsma.dpl.features`, not the
sampler. The `physical` feature variant collapses the six POLARIS depths into two
SAC-SMA storage zones — surface/upper 0–30 cm (depth-weighted 0-5, 5-15, 15-30) and
deep/lower 30–200 cm (30-60, 60-100, 100-200) — for each of sand, clay, ksat, and
theta_s, and encodes LAI peak-DOY as sin/cos. The raw six-depth columns stay in the
CSV, so a different aggregation never requires re-sampling.

### Derived (committed) products consumed by the model
- `data/<app>/soilveg_continuous<sfx>.csv` + `lai_climatology<sfx>.csv` — per-HRU
  continuous soil/veg/terrain features; the `physical` variant of
  `sacsma.dpl.features.build_features` uses them in place of the one-hot
  soil/veg columns. Path resolvers: `sacsma.io.soilveg_path` /
  `sacsma.io.lai_climatology_path`.
