"""The CalSim/CalLite application: SAC-SMA for CalSim3 / CalLite modeling.

Three per-watershed calibration domains (Wi & Steinschneider), each calibrated
**monthly** against observed full-natural-flow:

* ``9unimp`` — the 9 CalLite "Unimpaired" creek watersheds (Bear, Cache,
  Calaveras, Chowchilla, Cosumnes, Fresno, Mokelumne, Putah, Stony).
* ``11obs`` — 11 observed-gauge watersheds (major rim rivers).
* ``12rim`` — 12 CalLite rim-reservoir-inflow watersheds.

On top of the forward runs, :mod:`sacsma.calsim.catchments` re-aggregates any
set's HRUs onto the CalSim3 inflow catchments, and :mod:`sacsma.calsim.compare`
scores every set (plus the 15-CDEC set and the VIC benchmark) against the
CalSim3 historical inflows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..io import DEFAULT_DOMAIN, read_table

#: the CalSim/CalLite calibration domains.
DOMAINS = ("9unimp", "11obs", "12rim")


def calsim_dir(data_dir: str | Path = "data") -> Path:
    """The CalSim/CalLite application's data directory (``data/calsim``)."""
    return Path(data_dir) / "calsim"


def load_calib_monthly(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN,
                       basin: str | None = None) -> pd.DataFrame:
    """Monthly calibration record [date, basin, sim_mm, obs_mm, cal_start, cal_end].

    The observed monthly full-natural-flow target (``obs_mm``) and the MATLAB
    monthly simulation (``sim_mm``) extracted from each CalLite watershed's
    calibration log; ``cal_start``/``cal_end`` bound the calibration period.
    """
    df = read_table(calsim_dir(data_dir) / f"calib_{domain}_monthly.csv")
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df


def load_fnf_monthly(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN,
                     basin: str | None = None) -> pd.DataFrame:
    """Full-period monthly observed FNF [date, basin, obs_mm, cal_start, cal_end] (mm/month)."""
    df = read_table(calsim_dir(data_dir) / f"fnf_{domain}_monthly.csv")
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df


def load_vic_monthly(data_dir: str | Path = "data") -> pd.DataFrame:
    """VIC routed historical monthly flow [date, vic_name, flow_taf] (TAF/month)."""
    return read_table(calsim_dir(data_dir) / "vic_routed_monthly.csv")


def load_calsim3_monthly(data_dir: str | Path = "data") -> pd.DataFrame:
    """CalSim3 historical monthly inflow [date, arc, flow_taf] (TAF/month)."""
    return read_table(calsim_dir(data_dir) / "calsim3_inflow_monthly.csv")
