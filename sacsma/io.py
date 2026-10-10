"""Data-store loaders, date helpers, and unit conversions.

The ``data/`` store is laid out by role (:mod:`sacsma.paths`; ``data/README.md`` is the
manifest): ``inputs/`` (forcing, the grid, the modeling domains, the CalSim3 geometry),
``targets/`` (the flow records models are fitted to) and ``reference/`` (series that
results are compared with).  Tables are plain CSV (openable in Excel / a text editor);
the gridded stores are NetCDF or npz (git-LFS).  The per-domain loaders
(:func:`load_hru_table`, :func:`load_params`, :func:`load_reference`, ...) resolve a
modeling ``domain`` string to its file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

#: Default modeling domain (the 15 CDEC reservoir watersheds).
DEFAULT_DOMAIN = "15cdec"
#: the coarse 1/16-deg grid-aligned parallel of 15cdec — one unit per native
#: Livneh cell (vs the ~3.8x-denser off-grid HRU cloud); see data/inputs/domains/15cdec_grid
#: and data/README.md.
CDEC15_GRID_DOMAIN = "15cdec_grid"
#: the CalSim/CalLite application's domains.
CALSIM_DOMAINS = ("9unimp", "11obs", "12rim")
#: the multi-timescale training domain (``data/inputs/domains/multifamily``): one
#: "basin" per training entity, cells/weights from ``entity_cells.csv``,
#: flow lengths from ``flowlens.csv``.
MULTI_TIMESCALE_DOMAIN = "multifamily"
#: 1/16-deg-grid-based domains — these read the UNIFIED region forcing stores
#: (``data/inputs/forcing/<product>.nc``: one file per product at the region
#: grid, prcp/tmin/tmax with tavg derived; built by
#: data/inputs/forcing/build_region_forcing.py).  The fine ``15cdec`` domain keeps its
#: own dense off-grid store (special interpolation treatment upstream).
REGION_DOMAINS = (CDEC15_GRID_DOMAIN, *CALSIM_DOMAINS, MULTI_TIMESCALE_DOMAIN)


#: Default forcing product (filename stem): the historical **Livneh-unsplit**
#: grid (Pierce-2021 unsplit precipitation basis; Livneh+PRISM temperature).
DEFAULT_FORCING = "historical_livneh_unsplit"


def forcing_path(
    data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN, product: str = DEFAULT_FORCING
) -> Path:
    """Full path of a domain's forcing store for a ``product`` (the filename stem).

    Products: :data:`DEFAULT_FORCING` (the historical Livneh-unsplit grid),
    ``wgen_product_a`` (WGEN Product A scenario 1 — the same unsplit
    precipitation, temperature detrended to the 1991-2020 baseline; CalSim
    domains only) and ``wgen_product_a_sNN`` (WGEN Product A climate scenario
    NN, a compact table store decoded against ``wgen_product_a`` by
    :mod:`sacsma.wgen_scenarios`; region domains only).  Grid-based domains
    (:data:`REGION_DOMAINS`) share one store per product; the fine ``15cdec``
    domain keeps its own dense store.  The layout is in :mod:`sacsma.paths`."""
    return paths.forcing(data_dir, domain, product)


def norm_grid_key(k: str) -> str:
    """Normalize a ``<lat>_<lon>`` cell key to the region store's 5-decimal
    convention (the calsim HRU tables carry 6-decimal fixed-format keys)."""
    lat, lon = str(k).split("_")
    return f"{round(float(lat), 5)}_{round(float(lon), 5)}"


def soilveg_path(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> Path:
    """Per-HRU continuous soil/veg/terrain feature table (POLARIS + LANDFIRE +
    3DEP + MODIS-LAI sampled at each HRU point; see ``data/inputs/grid/README.md``).
    One row per HRU in ``hruinfo`` order, keyed (non-uniquely) by ``key``.
    The multi-timescale domain reads the full-coverage REGION table (one row per grid
    cell, all 4,410 cells)."""
    return paths.soilveg(data_dir, domain)


def lai_climatology_path(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> Path:
    """Per-HRU 46-value 8-day MODIS-LAI day-of-year climatology (companion to
    :func:`soilveg_path`; the Noah-ET canopy driver).  The multi-timescale domain reads
    the full-coverage REGION table."""
    return paths.lai_climatology(data_dir, domain)


#: 1 cfs sustained for a day, spread over 1 mi^2, equals this many mm.
#: (1 cfs = 0.0283168 m^3/s; x86400 s; / (mi^2 = 2.589988e6 m^2); x1000 mm/m)
_CFS_DAY_PER_MI2_MM = 0.944628


def cfs_to_mmday(cfs, area_mi2):
    """Convert discharge (cfs) to area-normalized depth (mm/day) over ``area_mi2``."""
    return cfs * _CFS_DAY_PER_MI2_MM / area_mi2


def mmday_to_cfs(mmday, area_mi2):
    """Convert area-normalized depth (mm/day) over ``area_mi2`` to discharge (cfs)."""
    return mmday * area_mi2 / _CFS_DAY_PER_MI2_MM


# --------------------------------------------------------------------------
# Native (data/) loaders
# --------------------------------------------------------------------------
#: Columns parsed back to datetime64 when reading a native CSV table.
_DATE_COLS = ("date", "cal_start", "cal_end")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a native ``data/`` table (CSV), parsing date columns to datetime64.

    The native store is plain CSV so every table opens in Excel / a text editor
    without any script; date columns (:data:`_DATE_COLS`) round-trip to datetime.
    """
    path = Path(path)
    cols = pd.read_csv(path, nrows=0).columns
    parse = [c for c in _DATE_COLS if c in cols]
    return pd.read_csv(path, parse_dates=parse or None)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a native ``data/`` table as index-less CSV (the openable storage format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_hru_table(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> pd.DataFrame:
    """Per-HRU attribute table (with basin code) for a modeling ``domain``.

    The multi-timescale domain has no ``hruinfo`` file — its table is assembled on the
    hruinfo contract from the entity store: one row per (entity, cell) with
    ``basin`` = entity_id, ``area_weight`` = the square-overlap ``overlap_mi2``,
    ``flowlen`` from ``flowlens.csv`` (traced, meters), and ``elev`` joined
    per cell from the region statics (``dem_elev``)."""
    if domain != MULTI_TIMESCALE_DOMAIN:
        return read_table(paths.hruinfo(data_dir, domain))
    cells = read_table(paths.entity_cells(data_dir))
    fl = read_table(paths.flowlens(data_dir))
    hrus = cells.merge(fl[["entity_id", "key", "flowlen_m"]],
                       on=["entity_id", "key"], validate="one_to_one")
    if len(hrus) != len(cells):
        raise ValueError("flowlens.csv does not cover entity_cells.csv")
    sv = pd.read_csv(soilveg_path(data_dir, domain),
                     usecols=["key", "dem_elev"]).set_index("key")["dem_elev"]
    out = pd.DataFrame({
        "basin": hrus["entity_id"], "key": hrus["key"],
        "lat": hrus["lat"], "lon": hrus["lon"],
        "area_weight": hrus["overlap_mi2"],
        "elev": hrus["key"].map(sv), "flowlen": hrus["flowlen_m"],
    })
    if out["elev"].isna().any():
        n = int(out["elev"].isna().sum())
        raise ValueError(f"{n} entity cells missing dem_elev in the region "
                         "statics — rebuild data/inputs/grid/soilveg_continuous.csv")
    return out


def load_params(domain: str = DEFAULT_DOMAIN,
                artifacts_dir: str | Path = "artifacts") -> pd.DataFrame:
    """Per-HRU GA-optimum parameters for a ``domain`` (columns include ``key``), from its
    archived calibration in ``artifacts/models/`` (:func:`sacsma.paths.ga_optimum`).

    Not indexed: the pooled ``15cdec`` set has one param row per grid cell, but the
    per-watershed ``9unimp`` calibration repeats some shared cells with different
    params per ``basin``, so callers index by ``key`` (after filtering to a basin
    where a ``basin`` column is present).
    """
    return read_table(paths.ga_optimum(artifacts_dir, domain))


def load_reference(
    data_dir: str | Path = "data", basin: str | None = None, domain: str = DEFAULT_DOMAIN
) -> pd.DataFrame:
    """Reference MATLAB simulated flow for a ``domain`` (optionally one basin)."""
    df = read_table(paths.simflow(data_dir, domain))
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df


def load_basin_area(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> pd.DataFrame:
    """Per-basin drainage area table [basin, area_mi2] for a ``domain``."""
    return read_table(paths.basin_area(data_dir, domain))


def _ffill_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """Forward-fill NaN along ``axis`` (leading NaN are left alone)."""
    idx = np.where(~np.isnan(a), np.arange(a.shape[axis]).reshape(
        [-1 if d == axis else 1 for d in range(a.ndim)]), 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    return np.take_along_axis(a, idx, axis=axis)


def fill_missing_days(ds):
    """Persistence-fill missing forcing days from the **previous day**, per cell.

    A NaN forcing day propagates straight through Snow-17/SAC-SMA, so gaps in
    the source have to be closed before the model sees them.  The convention
    (decision 2026-07-30) is persistence: a missing day takes the adjacent —
    previous — day's value for that same cell, whether the gap is a handful of
    cells or the whole domain.  A leading gap has no previous day, so it takes
    the first observed day instead (back-fill).

    Only ``aorc`` currently carries NaN (see ``data/inputs/forcing/README.md``: an AORC
    source outage, not an artifact of our aggregation — the fill-masking fix
    turns unreported hours into NaN rather than fabricating 0.0).  The Livneh /
    WGEN / LTO stores are gap-free, and for them this returns each variable
    **untouched**, so the parity baseline is unaffected.

    The stored file keeps its NaN — the fill happens at load, so the record of
    what was actually missing is never overwritten.  The number of filled
    cell-days is reported in the ``nan_filled_cell_days`` attribute.
    """
    filled, out = 0, {}
    for v, da in ds.data_vars.items():
        a = da.values
        if ("time" not in da.dims or not np.issubdtype(a.dtype, np.floating)
                or not np.isnan(a).any()):
            out[v] = da
            continue
        filled += int(np.isnan(a).sum())
        ax = da.dims.index("time")
        a = _ffill_axis(a, ax)
        if np.isnan(a).any():                       # leading gap: no previous day
            a = np.flip(_ffill_axis(np.flip(a, ax), ax), ax)
        out[v] = da.copy(data=a)
    if not filled:
        return ds
    filled_ds = ds.copy()
    for v, da in out.items():
        filled_ds[v] = da
    filled_ds.attrs = {**ds.attrs, "nan_filled_cell_days": filled,
                       "nan_fill_method": "persistence (previous day), per cell"}
    return filled_ds


def load_forcing(
    data_dir: str | Path,
    domain: str = DEFAULT_DOMAIN,
    product: str = DEFAULT_FORCING,
):
    """Open the domain-wide forcing store (xarray Dataset, dims (key, time)).

    Grid cells indexed by ``key`` (``lat_lon``), shared across HRUs/basins;
    HRU-level attributes (elev, flowlen, area_weight, …) live in the HRU
    table, not here.

    Grid-based domains (:data:`REGION_DOMAINS`) are served from the UNIFIED
    region store: the domain's cells are selected (via its HRU table) and
    relabelled to the domain's native key strings, and ``tavg`` is derived as
    ``(tmax+tmin)/2`` (the committed stores' exact convention) — so the
    returned dataset holds ``prcp``/``tavg`` like the dense store, plus
    ``tmin``/``tmax``.  Gaps in the region store are
    persistence-filled at load — see :func:`fill_missing_days`.

    A WGEN climate-scenario product ``wgen_product_a_sNN`` is decoded exactly
    against ``wgen_product_a`` (:mod:`sacsma.wgen_scenarios`; raises on any
    fingerprint mismatch).
    """
    import xarray as xr

    from . import wgen_scenarios

    path = forcing_path(data_dir, domain, product)
    scen = wgen_scenarios.scenario_of(product) is not None
    if scen and domain not in REGION_DOMAINS:
        raise ValueError(f"{product} is a region-grid scenario store; domain {domain!r} "
                         "is not a region domain")
    ds = xr.open_dataset(path)
    if domain not in REGION_DOMAINS:
        # the dense 15cdec store is gap-free, and is left lazily-indexed on
        # purpose: fill_missing_days reads .values, which for that product
        # would pull ~2.7 GB into memory for nothing
        return ds
    hru_keys = load_hru_table(data_dir, domain)["key"].astype(str)
    uniq = list(dict.fromkeys(hru_keys))
    want = [norm_grid_key(k) for k in uniq]
    have = set(str(k) for k in ds["key"].values)
    absent = [u for u, w in zip(uniq, want, strict=True) if w not in have]
    if absent:
        raise KeyError(
            f"{len(absent)} {domain} cells absent from {path.name} (first: "
            f"{absent[:3]}) — e.g. outside the historical_lto release coverage")
    # persistence-fill before deriving tavg, so tavg is consistent with the
    # tmin/tmax the model actually gets (no-op unless the product has gaps).
    # Subset the variables first: aorc.nc carries nine, and filling the six
    # this function discards would load ~3x the data for nothing.
    if scen:
        ds.close()
        sub = fill_missing_days(wgen_scenarios.load_region_subset(data_dir, product, want))
    else:
        sub = fill_missing_days(ds[["prcp", "tmin", "tmax"]].sel(key=want))
    tavg = ((sub["tmin"].astype("float64") + sub["tmax"].astype("float64"))
            / 2.0).astype("float32")
    out = xr.Dataset(
        {"prcp": sub["prcp"], "tavg": tavg,
         "tmin": sub["tmin"], "tmax": sub["tmax"]},
        attrs=sub.attrs,          # carries fill_missing_days' nan_* provenance
    ).assign_coords(key=uniq)
    lat = np.array([float(k.split("_")[0]) for k in want])
    lon = np.array([float(k.split("_")[1]) for k in want])
    return out.assign_coords(lat=("key", lat), lon=("key", lon))


def doy_and_leap(dates: pd.Series | pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return (day_of_year int array, is_leap int array) for a date series."""
    dt = pd.DatetimeIndex(dates)
    doy = dt.dayofyear.to_numpy().astype(np.int64)
    is_leap = dt.is_leap_year.astype(np.int64)
    if hasattr(is_leap, "to_numpy"):
        is_leap = is_leap.to_numpy()
    return doy, np.asarray(is_leap, dtype=np.int64)
