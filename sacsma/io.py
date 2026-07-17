"""Data-store loaders, date helpers, and unit conversions.

The ``data/`` store is split by application — ``data/cdec15/`` (the 15-CDEC
domain) and ``data/calsim/`` (the CalSim/CalLite domains ``9unimp``/``11obs``/
``12rim`` plus the CalSim3/VIC references).  Everything is plain CSV (openable
in Excel / a text editor) except the gridded forcing stores (NetCDF, git-LFS).
The per-domain loaders (:func:`load_hru_table`, :func:`load_params`,
:func:`load_reference`, ...) resolve a modeling ``domain`` string to its file;
see ``data/INVENTORY.md`` for the full manifest and provenance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Default modeling domain (the 15 CDEC reservoir watersheds).
DEFAULT_DOMAIN = "15cdec"
#: the 15-CDEC application's domain.
CDEC15_DOMAIN = "15cdec"
#: the coarse 1/16-deg grid-aligned parallel of 15cdec — one unit per native
#: Livneh cell (vs the ~3.8x-denser off-grid HRU cloud); see data/cdec15_grid
#: and data/INVENTORY.md.
CDEC15_GRID_DOMAIN = "15cdec_grid"
#: the CalSim/CalLite application's domains.
CALSIM_DOMAINS = ("9unimp", "11obs", "12rim")
#: 1/16-deg-grid-based domains — these read the UNIFIED region forcing stores
#: (``data/region/forcing/<product>.nc``: one file per product at the region
#: grid, prcp/tmin/tmax with tavg derived; built by
#: dataprep/build_region_forcing.py).  The fine ``15cdec`` domain keeps its
#: own dense off-grid store (special interpolation treatment upstream).
REGION_DOMAINS = (CDEC15_GRID_DOMAIN, *CALSIM_DOMAINS)


def domain_dir(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> Path:
    """Application data directory: ``data/cdec15`` for 15cdec, ``data/cdec15_grid``
    for its grid parallel, ``data/calsim`` for the CalSim domains."""
    if domain == CDEC15_DOMAIN:
        return Path(data_dir) / "cdec15"
    if domain == CDEC15_GRID_DOMAIN:
        return Path(data_dir) / "cdec15_grid"
    if domain in CALSIM_DOMAINS:
        return Path(data_dir) / "calsim"
    raise ValueError(
        f"unknown domain {domain!r} (expected {CDEC15_DOMAIN}, {CDEC15_GRID_DOMAIN}, "
        f"or one of {CALSIM_DOMAINS})"
    )


def _sfx(domain: str) -> str:
    """Filename suffix: the 15cdec domains are unsuffixed (one domain per app dir);
    the calsim files carry ``_<domain>`` (three domains share the dir)."""
    return "" if domain in (CDEC15_DOMAIN, CDEC15_GRID_DOMAIN) else f"_{domain}"


#: Default forcing product (filename stem): the historical **Livneh-unsplit**
#: grid (Pierce-2021 unsplit precipitation basis; Livneh+PRISM temperature).
DEFAULT_FORCING = "historical_livneh_unsplit"


def forcing_name(domain: str = DEFAULT_DOMAIN, product: str = DEFAULT_FORCING) -> str:
    """Forcing store filename for a ``product`` (the filename stem).

    Products: :data:`DEFAULT_FORCING` (the historical Livneh-unsplit grid) and
    ``wgen_product_a`` (WGEN Product A scenario 1 — the same unsplit
    precipitation, temperature detrended to the 1991-2020 baseline; CalSim
    domains only).  See ``data/INVENTORY.md``.
    """
    return f"{product}{_sfx(domain)}.nc"


def forcing_path(
    data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN, product: str = DEFAULT_FORCING
) -> Path:
    """Full path of a domain's forcing store.

    Grid-based domains (:data:`REGION_DOMAINS`) share the unified region store
    ``data/region/forcing/<product>.nc``; the fine ``15cdec`` domain keeps its
    per-domain ``<app dir>/forcing/<name>.nc``."""
    if domain in REGION_DOMAINS:
        return Path(data_dir) / "region" / "forcing" / f"{product}.nc"
    return domain_dir(data_dir, domain) / "forcing" / forcing_name(domain, product)


def norm_grid_key(k: str) -> str:
    """Normalize a ``<lat>_<lon>`` cell key to the region store's 5-decimal
    convention (the calsim HRU tables carry 6-decimal fixed-format keys)."""
    lat, lon = str(k).split("_")
    return f"{round(float(lat), 5)}_{round(float(lon), 5)}"


def soilveg_path(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> Path:
    """Per-HRU continuous soil/veg/terrain feature table (POLARIS + LANDFIRE +
    3DEP + MODIS-LAI sampled at each HRU point; see ``data/raw_gis/SOURCES.md``).
    One row per HRU in ``hruinfo`` order, keyed (non-uniquely) by ``key``."""
    return domain_dir(data_dir, domain) / f"soilveg_continuous{_sfx(domain)}.csv"


def lai_climatology_path(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> Path:
    """Per-HRU 46-value 8-day MODIS-LAI day-of-year climatology (companion to
    :func:`soilveg_path`; the Noah-ET canopy driver)."""
    return domain_dir(data_dir, domain) / f"lai_climatology{_sfx(domain)}.csv"


#: Noah-lite observed-canopy clamps — keep in sync with dpl.config.CANOPY_BOUNDS
#: (veg_frac) and dpl.data._load_canopy_obs (the tiny observed-LAI floor, NOT the
#: learned 0.5 bound, which clamped ~half the driest basins' days).
_VEG_FRAC_LO, _VEG_FRAC_HI = 0.0, 1.0
_LAI_OBS_LO, _LAI_OBS_HI = 0.05, 6.0


def load_canopy_obs(
    data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN,
) -> tuple[pd.Series, pd.DataFrame]:
    """OBSERVED canopy structure for the Noah-lite ET path, keyed by grid-cell
    ``key`` — the torch-free core loader the frozen ``run_basin`` Noah-lite path
    uses (mirror of ``sacsma.dpl.data._load_canopy_obs``, which aligns the same
    inputs to the torch domain-tensor cell order).

    Returns ``(veg_frac, lai_lut)``:

    * ``veg_frac`` — pd.Series (index ``key``) of the LANDFIRE EVC cover fraction
      (``EVC_cover_pct`` / 100), clamped to [0, 1];
    * ``lai_lut`` — pd.DataFrame (index ``key``, columns day-of-year 1..366) of
      the per-cell daily LAI climatology, linearly interpolated from the 46
      8-day ``lai_doy*`` samples and clamped to the observed floor/ceiling.

    Rows are deduped by ``key`` (fine-HRU domains repeat identical per-cell rows).
    Raises ``FileNotFoundError`` if the sidecars are absent (only the grid
    domains — e.g. ``15cdec_grid`` — ship them)."""
    import re

    sv_path = soilveg_path(data_dir, domain)
    lai_path = lai_climatology_path(data_dir, domain)
    if not sv_path.exists() or not lai_path.exists():
        raise FileNotFoundError(
            f"Noah-lite canopy sidecars not found for domain {domain!r}: "
            f"{sv_path.name} / {lai_path.name}")

    sv = pd.read_csv(sv_path, usecols=["key", "EVC_cover_pct"]).set_index("key")
    sv = sv[~sv.index.duplicated()]
    veg_frac = (sv["EVC_cover_pct"] / 100.0).clip(_VEG_FRAC_LO, _VEG_FRAC_HI)
    veg_frac.name = "veg_frac"

    lai = pd.read_csv(lai_path).rename(columns={"cellkey": "key"}).set_index("key")
    lai = lai[~lai.index.duplicated()]
    doy_cols = sorted((c for c in lai.columns if c.startswith("lai_doy")),
                      key=lambda c: int(re.sub(r"\D", "", c)))
    sample_doys = np.array([int(re.sub(r"\D", "", c)) for c in doy_cols], float)
    target = np.arange(1, 367, dtype=float)                          # doy 1..366
    lut = np.vstack([np.interp(target, sample_doys, row)
                     for row in lai[doy_cols].to_numpy(np.float64)])
    lut = np.clip(lut, _LAI_OBS_LO, _LAI_OBS_HI)
    lai_lut = pd.DataFrame(lut, index=lai.index, columns=np.arange(1, 367))
    return veg_frac, lai_lut


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
    """Per-HRU attribute table (with basin code) for a modeling ``domain``."""
    return read_table(domain_dir(data_dir, domain) / f"hruinfo{_sfx(domain)}.csv")


def load_params(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> pd.DataFrame:
    """Per-HRU GA-optimum parameters for a ``domain`` (columns include ``key``).

    Not indexed: the pooled ``15cdec`` set has one param row per grid cell, but the
    per-watershed ``9unimp`` calibration repeats some shared cells with different
    params per ``basin``, so callers index by ``key`` (after filtering to a basin
    where a ``basin`` column is present).
    """
    return read_table(domain_dir(data_dir, domain) / f"ga_optimum{_sfx(domain)}.csv")


def load_reference(
    data_dir: str | Path = "data", basin: str | None = None, domain: str = DEFAULT_DOMAIN
) -> pd.DataFrame:
    """Reference MATLAB simulated flow for a ``domain`` (optionally one basin)."""
    df = read_table(domain_dir(data_dir, domain) / f"simflow{_sfx(domain)}.csv")
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df


def load_basin_area(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> pd.DataFrame:
    """Per-basin drainage area table [basin, area_mi2] for a ``domain``."""
    return read_table(domain_dir(data_dir, domain) / f"basin_area{_sfx(domain)}.csv")


def load_forcing(
    data_dir: str | Path,
    name: str | None = None,
    domain: str = DEFAULT_DOMAIN,
    product: str = DEFAULT_FORCING,
):
    """Open the domain-wide forcing store (xarray Dataset, dims (key, time)).

    Grid cells indexed by ``key`` (``lat_lon``), shared across HRUs/basins;
    HRU-level attributes (elev, flowlen, area_weight, …) live in the HRU
    table, not here.  ``name`` overrides the resolved filename entirely.

    Grid-based domains (:data:`REGION_DOMAINS`) are served from the UNIFIED
    region store: the domain's cells are selected (via its HRU table) and
    relabelled to the domain's native key strings, and ``tavg`` is derived as
    ``(tmax+tmin)/2`` (the committed stores' exact convention) — so the
    returned dataset looks exactly like the retired per-domain files
    (``prcp``/``tavg``), plus ``tmin``/``tmax``.
    """
    import xarray as xr

    if name:
        return xr.open_dataset(domain_dir(data_dir, domain) / "forcing" / name)
    path = forcing_path(data_dir, domain, product)
    ds = xr.open_dataset(path)
    if domain not in REGION_DOMAINS:
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
    sub = ds.sel(key=want)
    tavg = ((sub["tmin"].astype("float64") + sub["tmax"].astype("float64"))
            / 2.0).astype("float32")
    out = xr.Dataset(
        {"prcp": sub["prcp"], "tavg": tavg,
         "tmin": sub["tmin"], "tmax": sub["tmax"]},
        attrs=ds.attrs,
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
