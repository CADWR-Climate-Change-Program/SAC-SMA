"""Sample the staged CA raster stack (POLARIS soil, LANDFIRE veg, 3DEP terrain,
MODIS LAI) at each HRU point -> per-HRU continuous soil/veg/terrain features that
replace the opaque one-hot ``soil_class``/``veg_class`` in the dPL parameter net.

Runs in the ``sacsma-gis`` env (``environment-gis.yml``: rasterio + pyhdf); NOT importable by the
core ``sacsma`` package (keeps the model path torch/gdal-free).  Reusable across
domains and for the future all-California extension.

Outputs (per HRU domain <d> in {15cdec, 9unimp, 11obs, 12rim}, each HRU a point):
  data/inputs/domains/<d>/soilveg_continuous.csv   one row per HRU (key), durable RAW values
  data/inputs/domains/<d>/lai_climatology.csv      HRU x 46 8-day DOY LAI climatology (Noah-ET)

Grid cells are sampled over their footprint (:func:`run_footprint`): each cell is the mean of
13 x 13 sub-points from corner to corner, and its terrain comes from the sub-point elevations.
``region`` does EVERY cell of the region grid (``data/inputs/grid/grid_cells.csv``, all 4410
cells, whatever modeling domain a cell belongs to):
  data/inputs/grid/soilveg_continuous_raster.csv
  data/inputs/grid/lai_climatology_raster.csv
from which ``data/inputs/grid/build_region_statics.py`` writes the region store; ``15cdec_grid``
does that domain's cells into its two sidecars (one row per cell).

Design notes
------------
* Each HRU is a POINT (its lat/lon); rasters are sampled at that point.  Terrain
  derivatives use a small window around the point (sub-grid roughness), since a
  single 30 m slope is noisy.  A grid cell's sub-points are sampled the same way.
* POLARIS stores RAW per-depth values (4 props x 6 depths = 24 cols) -- the
  durable artifact; depth aggregation into SAC-SMA zones is a *modelling* choice
  left to the feature builder.  ``ksat`` is log10(cm/hr); ``theta_s`` m3/m3;
  sand/clay %.
* MODIS LAI: analytic sinusoidal georef (R=6371007.181), QC-masked climatology.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import DATA, layout, local_path  # noqa: E402

#: Staging root: ``raw_gis`` in data/local_paths.toml (or ``SACSMA_RAW_GIS``), else
#: data/inputs/grid/raw_gis -- the same lookup as data/inputs/grid/download_gis.py.
RAW = local_path("raw_gis", layout.raw_gis(DATA))
POLARIS = RAW / "polaris" / "PROPERTIES" / "v1.0"
LANDFIRE = RAW / "landfire"
DEM = RAW / "dem" / "3dep_1as"
LAI = RAW / "lai" / "mcd15a2h"

PROPS = ("sand", "clay", "ksat", "theta_s")
DEPTHS = ("0_5", "5_15", "15_30", "30_60", "60_100", "100_200")

DOMAINS = ("15cdec", "9unimp", "11obs", "12rim")
#: the grid domain whose sidecars are cell-footprint means (:func:`run_footprint`)
FOOTPRINT_DOMAIN = "15cdec_grid"


def load_region_cells() -> pd.DataFrame:
    """Every cell of the region grid -- the full CalSim3-gpkg intersection with
    the 1/16-deg grid (4410 cells), regardless of modeling-domain membership."""
    grid = pd.read_csv(layout.grid_cells(DATA))
    return grid[["key", "lat", "lon"]].reset_index(drop=True)

# ---- MODIS sinusoidal constants (analytic georef) ------------------------
SIN_R = 6371007.181            # sphere radius (m)
SIN_T = 1111950.5196666666     # tile side (m)
SIN_X0 = -20015109.354         # global left edge x (m)
SIN_Y0 = 10007554.677          # global top edge y (m)
SIN_NPIX = 2400                # pixels/side for 500 m
SIN_P = SIN_T / SIN_NPIX       # pixel side (m) ~= 463.3127


def load_hrus(domain: str) -> pd.DataFrame:
    df = pd.read_csv(layout.hruinfo(DATA, domain))
    return df


# ==========================================================================
# Tile routing
# ==========================================================================
def polaris_path(lat: float, lon: float, prop: str, depth: str) -> Path:
    s, w = math.floor(lat), math.floor(lon)
    return POLARIS / prop / "mean" / depth / f"lat{s}{s + 1}_lon{w}{w + 1}.tif"


def landfire_path(lat: float, lon: float, prod: str) -> Path:
    s, w = math.floor(lat), math.floor(lon)
    return LANDFIRE / prod / f"{prod}_lat{s}{s + 1}_lon{w}{w + 1}.tif"


def dem_path(lat: float, lon: float) -> Path:
    n = math.ceil(lat)
    w = math.ceil(-lon)
    return DEM / f"USGS_1_n{n}w{w}.tif"


def lai_rc(lat: float, lon: float) -> tuple[str, int, int]:
    """(tile 'hHHvVV', row, col) for a WGS84 point in the MODIS sinusoidal grid."""
    latr, lonr = math.radians(lat), math.radians(lon)
    x = SIN_R * lonr * math.cos(latr)
    y = SIN_R * latr
    h = int((x - SIN_X0) // SIN_T)
    v = int((SIN_Y0 - y) // SIN_T)
    col = int((x - (SIN_X0 + h * SIN_T)) // SIN_P)
    row = int(((SIN_Y0 - v * SIN_T) - y) // SIN_P)
    col = min(max(col, 0), SIN_NPIX - 1)
    row = min(max(row, 0), SIN_NPIX - 1)
    return f"h{h:02d}v{v:02d}", row, col


# ==========================================================================
# POLARIS + LANDFIRE  (point sampling, grouped by tile)
# ==========================================================================
def _windowed_nearest(src, lon, lat, nod, win=81):
    """Nearest finite (non-nodata) pixel to a point, within a ``win``x``win``
    window -- fills POLARIS nodata over water bodies with the closest real soil."""
    from rasterio.windows import Window
    r, c = src.index(lon, lat)
    half = win // 2
    fill = nod if nod is not None else -9999.0
    w = src.read(1, window=Window(c - half, r - half, win, win),
                 boundless=True, fill_value=fill).astype(np.float64)
    if nod is not None:
        w[w == nod] = np.nan
    if not np.isfinite(w).any():
        return np.nan
    rr, cc = np.mgrid[0:win, 0:win]
    d2 = (rr - half) ** 2 + (cc - half) ** 2
    d2 = np.where(np.isfinite(w), d2, np.inf)
    k = np.argmin(d2)
    return float(w.flat[k])


def _sample_points_by_tile(hrus, path_fn, band_name, fill_window=0):
    """Sample one raster family at all HRU points; group by tile so each file
    opens once.  Returns (values, filled_mask) aligned to hrus row order.  When
    ``fill_window`` > 0, point samples that land on nodata are backfilled with
    the nearest finite pixel within that window."""
    import rasterio

    out = np.full(len(hrus), np.nan)
    filled = np.zeros(len(hrus), dtype=bool)
    groups: dict[Path, list[int]] = defaultdict(list)
    for i, (lat, lon) in enumerate(zip(hrus["lat"].to_numpy(), hrus["lon"].to_numpy())):
        groups[path_fn(lat, lon)].append(i)
    for path, idx in groups.items():
        if not path.exists():
            continue
        coords = [(hrus["lon"].iat[i], hrus["lat"].iat[i]) for i in idx]
        with rasterio.open(path) as src:
            nod = src.nodata
            vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float64)
            if nod is not None:
                vals[vals == nod] = np.nan
            if fill_window > 0:
                for j, i in enumerate(idx):
                    if not np.isfinite(vals[j]):
                        vals[j] = _windowed_nearest(
                            src, hrus["lon"].iat[i], hrus["lat"].iat[i], nod, fill_window)
                        filled[i] = np.isfinite(vals[j])
        for j, i in enumerate(idx):
            out[i] = vals[j]
    return out, filled


def sample_polaris(hrus) -> pd.DataFrame:
    cols = {}
    fill_flag = np.zeros(len(hrus), dtype=bool)
    for prop in PROPS:
        for depth in DEPTHS:
            name = f"{prop}_{depth}"
            vals, filled = _sample_points_by_tile(
                hrus, lambda la, lo, p=prop, d=depth: polaris_path(la, lo, p, d),
                name, fill_window=81)
            cols[name] = vals
            fill_flag |= filled
            print(f"    polaris {name}: {np.isfinite(vals).sum()}/{len(hrus)} finite"
                  f"  (+{int(filled.sum())} gapfilled)", flush=True)
    df = pd.DataFrame(cols, index=hrus.index)
    df["polaris_gapfill"] = fill_flag.astype(int)
    return df


def sample_landfire(hrus) -> pd.DataFrame:
    cols = {}
    for prod in ("EVC", "EVH"):
        vals, _ = _sample_points_by_tile(
            hrus, lambda la, lo, p=prod: landfire_path(la, lo, p), prod)
        cols[prod] = vals
        print(f"    landfire {prod}: "
              f"{np.isfinite(vals).sum()}/{len(hrus)} finite", flush=True)
    df = pd.DataFrame(cols, index=hrus.index)
    df["EVC_cover_pct"] = decode_evc_cover(df["EVC"].to_numpy())
    df["EVH_height_m"] = decode_evh_height(df["EVH"].to_numpy())
    return df


def decode_evc_cover(evc: np.ndarray) -> np.ndarray:
    """LANDFIRE EVC coded value -> continuous canopy cover %.  Veg is banded by
    lifeform (100s=Tree, 200s=Shrub, 300s=Herb) and the LAST TWO DIGITS are the
    cover %; non-veg codes (<100: water/developed/agriculture/sparse) -> 0 cover.
    Verified against the tile value distribution (see inspect_landfire)."""
    v = evc.astype(np.float64)
    cover = np.where((v >= 100) & (v <= 399), np.mod(v, 100), 0.0)
    cover[~np.isfinite(v)] = np.nan
    return cover


def decode_evh_height(evh: np.ndarray) -> np.ndarray:
    """LANDFIRE EVH coded value -> continuous height (m), banded by lifeform:
    Tree 100-199 => (v-100) m; Shrub 200-299 => (v-200)*0.1 m; Herb 300-399 =>
    (v-300)*0.1 m; non-veg (<100) -> 0.  Verified against the tile value set."""
    v = evh.astype(np.float64)
    h = np.zeros_like(v)
    tree = (v >= 100) & (v <= 199)
    shrub = (v >= 200) & (v <= 299)
    herb = (v >= 300) & (v <= 399)
    h[tree] = v[tree] - 100.0
    h[shrub] = (v[shrub] - 200.0) * 0.1
    h[herb] = (v[herb] - 300.0) * 0.1
    h[~np.isfinite(v)] = np.nan
    return h


# ==========================================================================
# Terrain (3DEP) -- windowed slope / aspect / curvature / relief
# ==========================================================================
def sample_terrain(hrus, win=25) -> pd.DataFrame:
    import rasterio
    from rasterio.windows import Window

    n = len(hrus)
    elev = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    asp_s = np.full(n, np.nan)
    asp_c = np.full(n, np.nan)
    curv = np.full(n, np.nan)
    relief = np.full(n, np.nan)

    groups: dict[Path, list[int]] = defaultdict(list)
    for i, (lat, lon) in enumerate(zip(hrus["lat"].to_numpy(), hrus["lon"].to_numpy())):
        groups[dem_path(lat, lon)].append(i)

    for path, idx in groups.items():
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            nod = src.nodata
            for i in idx:
                lat = hrus["lat"].iat[i]
                lon = hrus["lon"].iat[i]
                r, c = src.index(lon, lat)
                half = win // 2
                r0, c0 = r - half, c - half
                w = src.read(1, window=Window(c0, r0, win, win),
                             boundless=True, fill_value=np.nan).astype(np.float64)
                if nod is not None:
                    w[w == nod] = np.nan
                if not np.isfinite(w).any():
                    continue
                # cell size in meters (1 arc-sec ~ 30.87 m N-S)
                dy = 30.87
                dx = 30.87 * math.cos(math.radians(lat))
                gy, gx = np.gradient(w, dy, dx)          # d/drow(->-y), d/dcol(->x)
                # rows increase southward -> north gradient = -gy
                dzdx, dzdy = gx, -gy
                sl = np.sqrt(dzdx ** 2 + dzdy ** 2)
                elev[i] = w[half, half] if np.isfinite(w[half, half]) else np.nanmean(w)
                slope[i] = math.degrees(math.atan(np.nanmean(sl)))
                a = np.arctan2(np.nanmean(dzdy), -np.nanmean(dzdx))  # aspect rad
                asp_s[i] = math.sin(a)
                asp_c[i] = math.cos(a)
                lap = np.gradient(gx, dx, axis=1) + np.gradient(gy, dy, axis=0)
                curv[i] = np.nanmean(lap)
                relief[i] = np.nanstd(w)
    return pd.DataFrame({"dem_elev": elev, "slope_deg": slope,
                         "aspect_sin": asp_s, "aspect_cos": asp_c,
                         "curvature": curv, "relief_m": relief}, index=hrus.index)


# ==========================================================================
# MODIS LAI  -- streamed climatology (pyhdf)
# ==========================================================================
def sample_lai(hrus, verbose=True, max_files=None):
    from pyhdf.SD import SD, SDC

    n = len(hrus)
    tile_rc = [lai_rc(la, lo) for la, lo in
               zip(hrus["lat"].to_numpy(), hrus["lon"].to_numpy())]
    by_tile: dict[str, list[int]] = defaultdict(list)
    for i, (t, r, c) in enumerate(tile_rc):
        by_tile[t].append(i)

    NBIN = 46
    ssum = np.zeros((n, NBIN))
    scnt = np.zeros((n, NBIN))

    for tile, idx in by_tile.items():
        files = sorted(glob.glob(str(LAI / tile / f"MCD15A2H.*.{tile}.061.*.hdf")))
        if max_files is not None:
            files = files[:max_files]
        if verbose:
            print(f"    LAI {tile}: {len(idx)} HRUs, {len(files)} files", flush=True)
        rows = np.array([tile_rc[i][1] for i in idx])
        cols = np.array([tile_rc[i][2] for i in idx])
        idx = np.array(idx)
        for k, f in enumerate(files):
            base = os.path.basename(f)
            doy = int(base.split(".")[1][5:8])          # A{yyyy}{ddd}
            b = (doy - 1) // 8
            if b >= NBIN:
                b = NBIN - 1
            try:
                sd = SD(f, SDC.READ)
                lai = sd.select("Lai_500m").get()       # (2400,2400) uint8
                sd.end()
            except Exception as e:  # noqa: BLE001
                print(f"      LAI read fail {base}: {e!r}", flush=True)
                continue
            vals = lai[rows, cols].astype(np.float64)
            good = vals <= 100                          # fill/unfilled are >100
            v = np.where(good, vals * 0.1, 0.0)
            ssum[idx, b] += np.where(good, v, 0.0)
            scnt[idx, b] += good
            if verbose and (k + 1) % 500 == 0:
                print(f"      {tile} [{k + 1}/{len(files)}]", flush=True)

    with np.errstate(invalid="ignore"):
        clim = ssum / np.where(scnt > 0, scnt, np.nan)   # (n,46) NaN where no data
    clim = _fill_doy_gaps(clim)

    # HRUs whose MODIS pixel is always fill (open water/permanent snow) have an
    # all-NaN climatology: borrow the nearest valid HRU's climatology (lat/lon
    # nearest neighbour), and flag it.
    lat = hrus["lat"].to_numpy(np.float64)
    lon = hrus["lon"].to_numpy(np.float64)
    allnan = ~np.isfinite(clim).any(axis=1)
    gapfill = np.zeros(n, dtype=int)
    valid = np.where(~allnan)[0]
    if allnan.any() and valid.size:
        clat = np.cos(np.deg2rad(lat))
        for i in np.where(allnan)[0]:
            dx = (lon[valid] - lon[i]) * clat[i]
            dy = lat[valid] - lat[i]
            j = valid[np.argmin(dx * dx + dy * dy)]
            clim[i] = clim[j]
            gapfill[i] = 1

    lai_mean = np.nanmean(clim, axis=1)
    lai_min = np.nanmin(clim, axis=1)
    lai_max = np.nanmax(clim, axis=1)
    lai_amp = lai_max - lai_min
    peak_bin = np.nanargmax(np.where(np.isfinite(clim), clim, -1), axis=1)
    lai_peak_doy = peak_bin * 8 + 1
    print(f"    LAI gapfilled {int(gapfill.sum())} all-fill HRUs "
          f"(nearest-valid-HRU)", flush=True)
    summ = pd.DataFrame({"lai_mean": lai_mean, "lai_min": lai_min, "lai_max": lai_max,
                         "lai_amp": lai_amp, "lai_peak_doy": lai_peak_doy,
                         "lai_gapfill": gapfill}, index=hrus.index)
    clim_df = pd.DataFrame(clim, index=hrus.index,
                           columns=[f"lai_doy{b * 8 + 1:03d}" for b in range(NBIN)])
    return summ, clim_df


def _fill_doy_gaps(clim: np.ndarray) -> np.ndarray:
    """Circularly interpolate NaN 8-day bins per HRU (seasonal continuity)."""
    n, nb = clim.shape
    x = np.arange(nb)
    out = clim.copy()
    for i in range(n):
        row = clim[i]
        m = np.isfinite(row)
        if m.sum() == 0:
            continue
        if m.sum() == nb:
            continue
        # wrap by tiling three periods so endpoints interpolate across the year
        xt = np.concatenate([x[m] - nb, x[m], x[m] + nb])
        yt = np.concatenate([row[m], row[m], row[m]])
        out[i] = np.interp(x, xt, yt)
    return out


# ==========================================================================
# Cell footprint  -- the 13 x 13 sub-point mean of a 1/16-deg cell
# ==========================================================================
#: sub-points per side of a cell, from corner to corner (169 per cell)
FOOT_N = 13
CELL_DEG = 1.0 / 16.0
#: metres per degree of latitude for the footprint terrain
M_PER_DEG = 111320.0


def footprint_points(cells: pd.DataFrame) -> pd.DataFrame:
    """The FOOT_N x FOOT_N sub-points of each cell: ``np.linspace(-1/32, 1/32, 13)`` degrees
    about the centre in latitude and longitude, latitude-major (south to north), one row per
    sub-point with the cell's row number in ``cell``."""
    off = np.linspace(-0.5, 0.5, FOOT_N) * CELL_DEG
    n = len(cells)
    lat = (cells["lat"].to_numpy()[:, None, None] + off[None, :, None]).repeat(FOOT_N, axis=2)
    lon = (cells["lon"].to_numpy()[:, None, None] + off[None, None, :]).repeat(FOOT_N, axis=1)
    return pd.DataFrame({"cell": np.repeat(np.arange(n), FOOT_N * FOOT_N),
                         "lat": lat.ravel(), "lon": lon.ravel()})


def _tile_values(points, path_fn, fill_window=0):
    """The pixel under every point, as ``rasterio``'s ``sample`` gives it (``rowcol``; outside
    the tile its nodata, or 0 without one), read from each tile's whole band at once.  Nodata
    becomes NaN; with ``fill_window`` a NaN takes the nearest finite pixel in that window
    (:func:`_windowed_nearest`).  Returns (values, filled)."""
    import rasterio
    from rasterio.transform import rowcol

    lat, lon = points["lat"].to_numpy(), points["lon"].to_numpy()
    out = np.full(len(points), np.nan)
    filled = np.zeros(len(points), dtype=bool)
    groups: dict[Path, list[int]] = defaultdict(list)
    for i, (la, lo) in enumerate(zip(lat, lon)):
        groups[path_fn(la, lo)].append(i)
    for path, idx in groups.items():
        if not path.exists():
            continue
        idx = np.asarray(idx)
        with rasterio.open(path) as src:
            nod = src.nodata
            band = src.read(1)
            r, c = rowcol(src.transform, lon[idx], lat[idx])
            r, c = np.asarray(r), np.asarray(c)
            inside = (r >= 0) & (r < src.height) & (c >= 0) & (c < src.width)
            vals = np.full(len(idx), float(np.array(nod or 0, dtype=src.dtypes[0])))
            vals[inside] = band[r[inside], c[inside]]
            if nod is not None:
                vals[vals == nod] = np.nan
            if fill_window > 0:
                for j in np.flatnonzero(~np.isfinite(vals)):
                    vals[j] = _windowed_nearest(src, lon[idx[j]], lat[idx[j]], nod, fill_window)
                    filled[idx[j]] = np.isfinite(vals[j])
        out[idx] = vals
    return out, filled


def _point_elevation(points, win=25):
    """3DEP elevation at every point: the pixel under it, else the mean of the ``win`` window
    around it (:func:`sample_terrain`'s ``dem_elev``)."""
    import rasterio
    from rasterio.windows import Window

    z, _ = _tile_values(points, dem_path)
    lat, lon = points["lat"].to_numpy(), points["lon"].to_numpy()
    for i in np.flatnonzero(~np.isfinite(z)):
        path = dem_path(lat[i], lon[i])
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            r, c = src.index(lon[i], lat[i])
            half = win // 2
            w = src.read(1, window=Window(c - half, r - half, win, win), boundless=True,
                         fill_value=np.nan).astype(np.float64)
            if src.nodata is not None:
                w[w == src.nodata] = np.nan
            if np.isfinite(w).any():
                z[i] = np.nanmean(w)
    return z


def footprint_terrain(cells, z) -> pd.DataFrame:
    """Terrain of each cell from its sub-point elevations ``z`` (n_cells, 169): the mean
    elevation, the spread (``relief_m``, the standard deviation), and on the FOOT_N x FOOT_N
    grid (spacing CELL_DEG / (FOOT_N - 1)) the slope of the mean gradient magnitude, the mean
    Laplacian (``curvature``) and the aspect of the mean gradient."""
    n = len(cells)
    out = {k: np.full(n, np.nan) for k in ("dem_elev", "slope_deg", "aspect_sin", "aspect_cos",
                                            "curvature", "relief_m")}
    dy = CELL_DEG / (FOOT_N - 1) * M_PER_DEG
    for k in range(n):
        g = z[k].reshape(FOOT_N, FOOT_N)[::-1]                 # north up
        if not np.isfinite(g).all():
            continue
        dx = dy * math.cos(math.radians(cells["lat"].iat[k]))
        gy, gx = np.gradient(g, dy, dx)
        dzdx, dzdy = gx, -gy                                   # rows run south
        out["dem_elev"][k] = g.mean()
        out["relief_m"][k] = g.std()
        out["slope_deg"][k] = math.degrees(math.atan(np.mean(np.hypot(dzdx, dzdy))))
        a = np.arctan2(dzdy.mean(), -dzdx.mean())
        out["aspect_sin"][k], out["aspect_cos"][k] = math.sin(a), math.cos(a)
        out["curvature"][k] = (np.gradient(gx, dx, axis=1) + np.gradient(gy, dy, axis=0)).mean()
    return pd.DataFrame(out, index=cells.index)


def run_footprint(cells: pd.DataFrame, layers: set[str], dest: Path, cdest: Path,
                  key_col: str = "key") -> None:
    """Each cell of ``cells`` (``key, lat, lon``) as the mean of its 169 sub-points: POLARIS
    (``polaris_gapfill`` the share gap-filled), LANDFIRE codes and their decoded cover and
    height, :func:`footprint_terrain`, and the MODIS LAI climatology (the mean of the
    sub-points' climatologies, its summary columns from that mean; ``lai_gapfill`` the share
    of sub-points with no valid MODIS value, which take the climatology of the nearest valid
    sub-point among those sampled)."""
    pts = footprint_points(cells)
    print(f"[footprint] {len(cells)} cells, {len(pts)} sub-points  layers={sorted(layers)}",
          flush=True)
    nc = FOOT_N * FOOT_N

    def cell_mean(v):
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.asarray(v, np.float64).reshape(len(cells), nc), axis=1)

    parts = [cells[["key"]].reset_index(drop=True)]
    if "polaris" in layers:
        t, cols, fill = time.time(), {}, np.zeros(len(pts), dtype=bool)
        for prop in PROPS:
            for depth in DEPTHS:
                v, f = _tile_values(pts, lambda la, lo, p=prop, d=depth: polaris_path(la, lo, p, d),
                                    fill_window=81)
                cols[f"{prop}_{depth}"] = cell_mean(v)
                fill |= f
                print(f"    polaris {prop}_{depth}: {int(f.sum())} sub-points gap-filled", flush=True)
        cols["polaris_gapfill"] = cell_mean(fill)
        parts.append(pd.DataFrame(cols))
        print(f"  polaris done ({time.time() - t:.0f}s)", flush=True)
    if "landfire" in layers:
        t, cols = time.time(), {}
        for prod in ("EVC", "EVH"):
            v, _ = _tile_values(pts, lambda la, lo, p=prod: landfire_path(la, lo, p))
            cols[prod] = cell_mean(v)
            dec = decode_evc_cover(v) if prod == "EVC" else decode_evh_height(v)
            cols["EVC_cover_pct" if prod == "EVC" else "EVH_height_m"] = cell_mean(dec)
        parts.append(pd.DataFrame(cols)[["EVC", "EVH", "EVC_cover_pct", "EVH_height_m"]])
        print(f"  landfire done ({time.time() - t:.0f}s)", flush=True)
    if "terrain" in layers:
        t = time.time()
        z = _point_elevation(pts).reshape(len(cells), nc)
        parts.append(footprint_terrain(cells.reset_index(drop=True), z))
        print(f"  terrain done ({time.time() - t:.0f}s)", flush=True)
    clim = None
    if "lai" in layers:
        t = time.time()
        summ, cl = sample_lai(pts)
        clim = cl.to_numpy().reshape(len(cells), nc, -1).mean(axis=1)
        mx = clim.max(axis=1)
        mn = clim.min(axis=1)
        parts.append(pd.DataFrame({"lai_mean": clim.mean(axis=1), "lai_min": mn, "lai_max": mx,
                                   "lai_amp": mx - mn,
                                   "lai_peak_doy": np.argmax(clim, axis=1) * 8 + 1,
                                   "lai_gapfill": cell_mean(summ["lai_gapfill"])}))
        clim = pd.DataFrame(clim, columns=cl.columns)
        print(f"  lai done ({time.time() - t:.0f}s)", flush=True)
    out = pd.concat(parts, axis=1)
    out.to_csv(dest, index=False)
    print(f"WROTE {dest}  ({out.shape[0]}x{out.shape[1]})", flush=True)
    if clim is not None:
        keys = cells[["key"]].reset_index(drop=True).rename(columns={"key": key_col})
        pd.concat([keys, clim], axis=1).to_csv(cdest, index=False)
        print(f"WROTE {cdest}  ({clim.shape[0]}x{clim.shape[1]})", flush=True)


# ==========================================================================
# Driver
# ==========================================================================
def run(domain: str, layers: set[str]):
    if domain == "region":
        run_footprint(load_region_cells(), layers,
                      layout.raster_sample(DATA, "soilveg_continuous"),
                      layout.raster_sample(DATA, "lai_climatology"))
        return
    if domain == FOOTPRINT_DOMAIN:
        keys = load_hrus(domain)["key"].drop_duplicates()
        ll = keys.str.split("_", expand=True).astype(float)
        cells = pd.DataFrame({"key": keys.to_numpy(), "lat": ll[0].to_numpy(),
                              "lon": ll[1].to_numpy()})
        run_footprint(cells, layers, layout.soilveg(DATA, domain),
                      layout.lai_climatology(DATA, domain), key_col="cellkey")
        return
    hrus = load_hrus(domain)
    dest, cdest = layout.soilveg(DATA, domain), layout.lai_climatology(DATA, domain)
    print(f"[{domain}] {len(hrus)} HRUs  layers={sorted(layers)}", flush=True)
    parts = [hrus[["key"]].copy()]

    if "polaris" in layers:
        t = time.time()
        parts.append(sample_polaris(hrus))
        print(f"  polaris done ({time.time() - t:.0f}s)", flush=True)
    if "landfire" in layers:
        t = time.time()
        parts.append(sample_landfire(hrus))
        print(f"  landfire done ({time.time() - t:.0f}s)", flush=True)
    if "terrain" in layers:
        t = time.time()
        parts.append(sample_terrain(hrus))
        print(f"  terrain done ({time.time() - t:.0f}s)", flush=True)
    clim_df = None
    if "lai" in layers:
        t = time.time()
        summ, clim_df = sample_lai(hrus)
        parts.append(summ)
        print(f"  lai done ({time.time() - t:.0f}s)", flush=True)

    out = pd.concat(parts, axis=1)
    out.to_csv(dest, index=False)
    print(f"WROTE {dest}  ({out.shape[0]}x{out.shape[1]})", flush=True)
    if clim_df is not None:
        pd.concat([hrus[["key"]], clim_df], axis=1).to_csv(cdest, index=False)
        print(f"WROTE {cdest}  ({clim_df.shape[0]}x{clim_df.shape[1]})", flush=True)


def test(domain: str, nrows: int = 3):
    """Route + sample the first ``nrows`` HRUs for every layer and print, to
    sanity-check tile math and value ranges before a full run."""
    hrus = load_hrus(domain).head(nrows).reset_index(drop=True)
    print(f"=== TEST {domain} first {nrows} HRUs ===")
    for _, h in hrus.iterrows():
        la, lo = h["lat"], h["lon"]
        print(f"\nHRU {h['basin']} ({la:.4f},{lo:.4f}) soil={h['soil_class']} veg={h['veg_class']}")
        print(f"  polaris sand/0_5 -> {polaris_path(la, lo, 'sand', '0_5').name}"
              f"  exists={polaris_path(la, lo, 'sand', '0_5').exists()}")
        print(f"  landfire EVC     -> {landfire_path(la, lo, 'EVC').name}"
              f"  exists={landfire_path(la, lo, 'EVC').exists()}")
        print(f"  dem              -> {dem_path(la, lo).name}"
              f"  exists={dem_path(la, lo).exists()}")
        t, r, c = lai_rc(la, lo)
        print(f"  lai tile={t} row={r} col={c}"
              f"  tiledir_exists={(LAI / t).exists()}")
    print("\n--- sampling values ---")
    print("polaris:\n", sample_polaris(hrus).T)
    print("landfire:\n", sample_landfire(hrus).T)
    print("terrain:\n", sample_terrain(hrus).T)
    summ, clim = sample_lai(hrus, max_files=46)
    print("lai summary (46 files only):\n", summ.T)
    print("lai clim (first 10 bins):\n", clim.iloc[:, :10].T)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=[*DOMAINS, FOOTPRINT_DOMAIN, "region"])
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--nrows", type=int, default=3)
    ap.add_argument("--layers", default="polaris,landfire,terrain,lai")
    args = ap.parse_args()
    if args.test:
        test(args.domain, args.nrows)
    else:
        run(args.domain, set(args.layers.split(",")))
