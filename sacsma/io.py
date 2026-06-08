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

#: Default modeling domain (the 15 CDEC reservoir watersheds).
DEFAULT_DOMAIN = "15cdec"


def forcing_name(domain: str = DEFAULT_DOMAIN) -> str:
    """Domain-wide forcing store filename under ``data/forcing/``."""
    return f"historical_{domain}.nc"


#: Default domain-wide forcing store filename (back-compat constant).
FORCING_NAME = forcing_name(DEFAULT_DOMAIN)

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
    return read_table(Path(data_dir) / "hru" / f"hruinfo_{domain}.csv")


def load_params(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> pd.DataFrame:
    """Per-HRU GA-optimum parameters for a ``domain`` (columns include ``key``).

    Not indexed: the pooled ``15cdec`` set has one param row per grid cell, but the
    per-watershed ``9unimp`` calibration repeats some shared cells with different
    params per ``basin``, so callers index by ``key`` (after filtering to a basin
    where a ``basin`` column is present).
    """
    return read_table(Path(data_dir) / "params" / f"ga_optimum_{domain}.csv")


def load_reference(
    data_dir: str | Path = "data", basin: str | None = None, domain: str = DEFAULT_DOMAIN
) -> pd.DataFrame:
    """Reference MATLAB simulated flow for a ``domain`` (optionally one basin)."""
    df = read_table(Path(data_dir) / "reference" / f"simflow_{domain}.csv")
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df


def load_gage(data_dir: str | Path = "data", basin: str | None = None) -> pd.DataFrame:
    """Observed gage FNF flow (calibration target, mm/day); optionally one basin."""
    df = read_table(Path(data_dir) / "reference" / "gage_15cdec.csv")
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df


def load_basin_area(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> pd.DataFrame:
    """Per-basin drainage area table [basin, area_mi2] for a ``domain``."""
    return read_table(Path(data_dir) / "reference" / f"basin_area_{domain}.csv")


def load_calib_monthly(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN,
                       basin: str | None = None) -> pd.DataFrame:
    """Monthly calibration record [date, basin, sim_mm, obs_mm, cal_start, cal_end].

    The observed monthly full-natural-flow target (``obs_mm``) and the MATLAB
    monthly simulation (``sim_mm``) extracted from each CalLite watershed's
    calibration log; ``cal_start``/``cal_end`` bound the calibration period.
    """
    df = read_table(Path(data_dir) / "reference" / f"calib_{domain}_monthly.csv")
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df


def load_fnf_monthly(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN,
                     basin: str | None = None) -> pd.DataFrame:
    """Full-period monthly observed FNF [date, basin, obs_mm, cal_start, cal_end] (mm/month)."""
    df = read_table(Path(data_dir) / "reference" / f"fnf_{domain}_monthly.csv")
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df


def load_vic_monthly(data_dir: str | Path = "data") -> pd.DataFrame:
    """VIC routed historical monthly flow [date, vic_name, flow_taf] (TAF/month)."""
    return read_table(Path(data_dir) / "reference" / "vic_routed_monthly.csv")


def load_calsim3_monthly(data_dir: str | Path = "data") -> pd.DataFrame:
    """CalSim3 historical monthly inflow [date, arc, flow_taf] (TAF/month)."""
    return read_table(Path(data_dir) / "reference" / "calsim3_inflow_monthly.csv")


def load_forcing(data_dir: str | Path, name: str | None = None, domain: str = DEFAULT_DOMAIN):
    """Open the domain-wide forcing store (xarray Dataset, dims (key, time)).

    One store per modeling ``domain`` (``historical_<domain>.nc``) — grid cells
    indexed by ``key`` (``lat_lon``), shared across HRUs/basins.  HRU-level
    attributes (elev, flowlen, area_weight, …) live in the HRU table, not here.
    """
    import xarray as xr

    return xr.open_dataset(Path(data_dir) / "forcing" / (name or forcing_name(domain)))


def doy_and_leap(dates: pd.Series | pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return (day_of_year int array, is_leap int array) for a date series."""
    dt = pd.DatetimeIndex(dates)
    doy = dt.dayofyear.to_numpy().astype(np.int64)
    is_leap = dt.is_leap_year.astype(np.int64)
    if hasattr(is_leap, "to_numpy"):
        is_leap = is_leap.to_numpy()
    return doy, np.asarray(is_leap, dtype=np.int64)
