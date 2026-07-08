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
#: the CalSim/CalLite application's domains.
CALSIM_DOMAINS = ("9unimp", "11obs", "12rim")


def domain_dir(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> Path:
    """Application data directory: ``data/cdec15`` for 15cdec, ``data/calsim`` else."""
    if domain == CDEC15_DOMAIN:
        return Path(data_dir) / "cdec15"
    if domain in CALSIM_DOMAINS:
        return Path(data_dir) / "calsim"
    raise ValueError(f"unknown domain {domain!r} (expected {CDEC15_DOMAIN} or one of {CALSIM_DOMAINS})")


def _sfx(domain: str) -> str:
    """Filename suffix: 15cdec files are unsuffixed (one domain per app dir);
    the calsim files carry ``_<domain>`` (three domains share the dir)."""
    return "" if domain == CDEC15_DOMAIN else f"_{domain}"


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
    """Full path of a domain's forcing store (``<app dir>/forcing/<name>.nc``)."""
    return domain_dir(data_dir, domain) / "forcing" / forcing_name(domain, product)


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

    One store per modeling ``domain`` and forcing ``product`` (default: the
    Livneh-unsplit historical grid) — grid cells indexed by ``key``
    (``lat_lon``), shared across HRUs/basins.  HRU-level attributes (elev,
    flowlen, area_weight, …) live in the HRU table, not here.  ``name``
    overrides the resolved filename entirely.
    """
    import xarray as xr

    d = domain_dir(data_dir, domain) / "forcing"
    return xr.open_dataset(d / name if name else forcing_path(data_dir, domain, product))


def doy_and_leap(dates: pd.Series | pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return (day_of_year int array, is_leap int array) for a date series."""
    dt = pd.DatetimeIndex(dates)
    doy = dt.dayofyear.to_numpy().astype(np.int64)
    is_leap = dt.is_leap_year.astype(np.int64)
    if hasattr(is_leap, "to_numpy"):
        is_leap = is_leap.to_numpy()
    return doy, np.asarray(is_leap, dtype=np.int64)
