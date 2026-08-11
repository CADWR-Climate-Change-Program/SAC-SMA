# Raw California/CONUS GIS — provenance & manifest

Continuous soil, vegetation, and terrain rasters staged locally, for two purposes:
to replace the opaque one-hot `soil_class`/`veg_class` HRU features with physical
continua for the dPL parameter net (the `physical` feature variant), and to let the
domain extend from the 15 CDEC basins to all of California without re-downloading.
The preference is high-quality California/CONUS products over global ones. This tree
is gitignored (`data/raw_gis/*`); only this provenance doc, the reusable HRU sampler
(`sample_gis.py`), and the small per-HRU derived CSVs
(`data/<app>/soilveg_continuous*.csv`, `lai_climatology*.csv`) are committed.
The sampler's fifth mode, `sample_gis.py region` (full-region-grid point-sample,
see `../../dataprep/README.md`), writes `data/region/{soilveg_continuous,
lai_climatology}_raster.csv` — also gitignored, since its content is fully
folded into the committed `data/region/{soilveg_continuous,lai_climatology}.csv`
by `dataprep/build_region_statics.py` and nothing reads it directly.

**California extent fetched:** lat 32–42 N, lon −125 to −114 W (110 1° tiles;
POLARIS and 3DEP are CONUS-land products, so the ocean/out-of-CONUS tiles of
that grid 404 and are skipped — 95 and 89 land tiles respectively), **plus one
tile at 42–43 N / −121…−120 W** fetched 2026-08-10 — the region grid (built
from the full CalSim3-gpkg footprint) reaches 42.4 N, half a degree past the
original extent, at the Goose Lake extension of BND. Fetched with the same
downloader, one tile, not a general extent widening (the rest of that row has
no region cells).
Downloader: [`../../dataprep/download_gis.py`](../../dataprep/download_gis.py)
(needs only `requests`; runs in the plain `sacsma` env). Sampler:
`sample_gis.py` (runs in the `sacsma-gis` conda env — rasterio + pyhdf + pyproj;
NOT importable by the core `sacsma` package).

**Where it lives.** The tree is too large for the repo drive and is staged at
`D:\sacsma-data\raw_gis`; set **`SACSMA_RAW_GIS`** to point the downloader —
and, since 2026-08-10, the sampler — at it (the same override precedent as
`SACSMA_ET_DIR`/`SACSMA_SWE_DIR`; `sample_gis.py` previously hard-coded the
in-repo path, which is why the earlier region-footprint gap existed at all).
`python dataprep/download_gis.py --status` inventories the stage.

**Staged (89.5 GB total, verified complete 2026-07-29, +1 tile 2026-08-10):**
POLARIS 2304 tiles / 55.1 GB · LANDFIRE 222 / 5.7 GB · 3DEP 90 / 3.8 GB ·
MODIS LAI 3672 granules / 24.9 GB.

The downloader reproduces this stage **byte-identically** — spot-checked by
SHA-256 on a re-fetched tile of each no-auth product (POLARIS `sand/mean/0_5`,
LANDFIRE `EVC`, 3DEP), including the rendered LANDFIRE tile, so the recorded
`exportImage` parameters are the ones that produced it.

**Tile geometry** (read off the stage, not assumed): POLARIS and LANDFIRE are
3600×3600, EPSG:4326, on exact 1° bounds with a 1/3600° pixel — POLARIS float32
nodata −9999, LANDFIRE int16 with no nodata. 3DEP is kept as published:
3604×3604, EPSG:4269, 1° plus a 2-px overlap per side, float32 nodata −999999.

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
- **Request (matters):** `bbox=<w>,<s>,<e>,<n>&bboxSR=4326&imageSR=4326&size=3600,3600`
  `&format=tiff&pixelType=S16&interpolation=RSP_NearestNeighbor&f=image`. The
  `size=3600,3600` over a 1° bbox is deliberate — it lands the render on the
  **same 1/3600° grid as POLARIS**. All 110 tiles render (an ImageServer will
  serve any bbox, ocean included). Note the service reports failures as a
  *200 with a JSON body*, so a size check alone will not catch them.
- **Layout:** `landfire/{EVC,EVH}/{EVC,EVH}_lat{S}{N}_lon{W}{E}.tif`

## LAI — MODIS MCD15A2H.061  ✅ staged
- **What:** 500 m, 8-day combined Terra+Aqua Leaf Area Index, **2003–2022**. Two
  uses: a QC-masked day-of-year **climatology** (46 8-day bins) as veg features,
  and the full climatology as the future Noah-ET canopy driver. CA sinusoidal
  tiles h08v04, h08v05, h09v04, h09v05.
- **Access:** NASA CMR granule API (`cmr.earthdata.nasa.gov/search/granules.json`)
  → LP DAAC protected cloud bucket (`data.lpdaac.earthdatacloud.nasa.gov`) via
  Earthdata `.netrc` (URS OAuth). The classic e4ftl01 HTTPS archive is dead.
  Granules are found per tile with `readable_granule_name[]=*.<tile>.*` +
  `options[readable_granule_name][pattern]=true`, paged on `CMR-Search-After`.
  Name each file from the **URL basename**: this collection's
  `producer_granule_id` omits the `.hdf` extension, which would hide the file
  from the sampler's `*.hdf` glob and defeat resume.
- **Count:** 918 granules/tile (verified against CMR 2026-07-29). The window
  `2003-01-01 … 2022-12-31` holds 920 8-day steps; CMR returns granules
  *overlapping* it, adding the composite starting 2002-12-27 (`A2002361`), and
  three are absent upstream — `A2016049`, `A2022097`, `A2022289`. 920 − 3 + 1 = 918.
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
- `data/region/{soilveg_continuous,lai_climatology}.csv` — the region-grid
  consolidation of the sidecars above (`dataprep/build_region_statics.py`),
  4410/4410 cells. `sample_gis.py region`'s full-4410-cell point-sample layer
  (`data/region/{soilveg_continuous,lai_climatology}_raster.csv`, gitignored,
  regenerate on demand) is its final fill source but is not itself committed
  or read directly by the model — see `../../dataprep/README.md`.
