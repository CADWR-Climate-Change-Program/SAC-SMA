"""Readers for the MATLAB-era reference data and date helpers.

File formats (all whitespace-delimited, no header):
  * meteo:    ``meteo_<lat>_<lon>`` -> year month day prcp tavg
  * HRUinfo:  ``HRUinfo_<CDEC>.txt`` -> lat lon area_weight elev flowlen soil veg b1 b2
  * simflow:  ``climate_historical/simflow_sacsma_<CDEC>_short.txt`` -> year month day value
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HRUINFO_COLUMNS = [
    "lat", "lon", "area_weight", "elev", "flowlen",
    "soil_class", "veg_class", "basin_id", "basin_id2",
]

#: Default domain-wide forcing store filename under ``data/forcing/``.
FORCING_NAME = "historical_15cdec.nc"

#: 1 cfs sustained for a day, spread over 1 mi^2, equals this many mm.
#: (1 cfs = 0.0283168 m^3/s; x86400 s; / (mi^2 = 2.589988e6 m^2); x1000 mm/m)
_CFS_DAY_PER_MI2_MM = 0.944628


def cfs_to_mmday(cfs, area_mi2):
    """Convert discharge (cfs) to area-normalized depth (mm/day) over ``area_mi2``."""
    return cfs * _CFS_DAY_PER_MI2_MM / area_mi2


def mmday_to_cfs(mmday, area_mi2):
    """Convert area-normalized depth (mm/day) over ``area_mi2`` to discharge (cfs)."""
    return mmday * area_mi2 / _CFS_DAY_PER_MI2_MM


def _read_ws(path, names, dtype=None) -> pd.DataFrame:
    """Fast whitespace-delimited read (C engine), version-robust.

    Uses ``delim_whitespace`` (C parser) — far faster than the regex
    ``sep=r"\\s+"`` Python engine on the large per-HRU meteo files.
    """
    kw = dict(header=None, names=names, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            return pd.read_csv(path, delim_whitespace=True, **kw)
        except (TypeError, ValueError):
            return pd.read_csv(path, sep=r"\s+", engine="python", **kw)


def meteo_filename(lat: float, lon: float) -> str:
    """Filename of the meteo grid cell for a given lat/lon (6 decimals)."""
    return f"meteo_{lat:.6f}_{lon:.6f}"


def parse_meteo_filename(name: str) -> tuple[str, float, float]:
    """Inverse of :func:`meteo_filename`: ``meteo_<lat>_<lon>`` -> (key, lat, lon).

    The grid-cell ``key`` (``lat_lon``) is the join key shared by the HRU
    table, the GA-parameter table, and the forcing store.
    """
    stem = Path(name).name
    if stem.startswith("meteo_"):
        stem = stem[len("meteo_"):]
    lat_s, lon_s = stem.split("_")
    return f"{lat_s}_{lon_s}", float(lat_s), float(lon_s)


def read_meteo(path: str | Path) -> pd.DataFrame:
    """Read a per-HRU meteo file -> DataFrame[date, prcp, tavg]."""
    df = _read_ws(
        path,
        ["year", "month", "day", "prcp", "tavg"],
        dtype={"year": "int16", "month": "int8", "day": "int8",
               "prcp": "float32", "tavg": "float32"},
    )
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    return df[["date", "prcp", "tavg"]]


def read_hruinfo(path: str | Path) -> pd.DataFrame:
    """Read an HRUinfo table -> DataFrame with :data:`HRUINFO_COLUMNS`."""
    df = _read_ws(path, HRUINFO_COLUMNS)
    from .parameters import latlon_key

    df["key"] = [latlon_key(la, lo) for la, lo in zip(df["lat"], df["lon"])]
    return df


def read_simflow(path: str | Path) -> pd.DataFrame:
    """Read a reference simulated-flow file -> DataFrame[date, flow]."""
    df = _read_ws(path, ["year", "month", "day", "flow"])
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    return df[["date", "flow"]]


# --------------------------------------------------------------------------
# Native (data/) loaders — produced by sacsma.dataprep
# --------------------------------------------------------------------------
def load_hru_table(data_dir: str | Path = "data") -> pd.DataFrame:
    """Per-HRU attribute table (with basin code)."""
    return pd.read_parquet(Path(data_dir) / "hru" / "hruinfo_15cdec.parquet")


def load_params(data_dir: str | Path = "data") -> pd.DataFrame:
    """Per-HRU GA-optimum parameters, indexed by ``key`` (lat_lon)."""
    df = pd.read_parquet(Path(data_dir) / "params" / "ga_optimum_15cdec.parquet")
    return df.set_index("key")


def load_reference(data_dir: str | Path = "data", basin: str | None = None) -> pd.DataFrame:
    """Reference MATLAB simulated flow (optionally filtered to one basin)."""
    df = pd.read_parquet(Path(data_dir) / "reference" / "simflow_15cdec.parquet")
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df


def load_gage(data_dir: str | Path = "data", basin: str | None = None) -> pd.DataFrame:
    """Observed gage FNF flow (calibration target, mm/day); optionally one basin."""
    df = pd.read_parquet(Path(data_dir) / "reference" / "gage_15cdec.parquet")
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df


def load_basin_area(data_dir: str | Path = "data") -> pd.DataFrame:
    """Per-basin drainage area table [basin, area_mi2]."""
    return pd.read_parquet(Path(data_dir) / "reference" / "basin_area_15cdec.parquet")


def load_forcing(data_dir: str | Path, name: str = FORCING_NAME):
    """Open the domain-wide forcing store (xarray Dataset, dims (key, time)).

    One store for the whole ingested meteo domain — grid cells indexed by
    ``key`` (``lat_lon``), shared across HRUs/basins.  HRU-level attributes
    (elev, flowlen, area_weight, …) live in the HRU table, not here.
    """
    import xarray as xr

    return xr.open_dataset(Path(data_dir) / "forcing" / name)


def doy_and_leap(dates: pd.Series | pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return (day_of_year int array, is_leap int array) for a date series."""
    dt = pd.DatetimeIndex(dates)
    doy = dt.dayofyear.to_numpy().astype(np.int64)
    is_leap = dt.is_leap_year.astype(np.int64)
    if hasattr(is_leap, "to_numpy"):
        is_leap = is_leap.to_numpy()
    return doy, np.asarray(is_leap, dtype=np.int64)
