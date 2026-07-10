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


def load_vic_monthly(data_dir: str | Path = "data",
                     product: str | None = None) -> pd.DataFrame:
    """VIC routed monthly flow [date, vic_name, flow_taf] (TAF/month).

    ``product=None`` is the historical baseline (the ``Historical_Unsplit`` VIC
    run — same unsplit precipitation basis as the SAC-SMA forcing).  The
    alternate-climate runs mirror the forcing products: ``historical_lto`` (the
    split-precipitation ``Historical`` run, 1915–2021) and ``wgen_product_a``
    (the detrended-temperature ``Product_A`` validation run).
    """
    sfx = f"_{product}" if product else ""
    return read_table(calsim_dir(data_dir) / f"vic_routed_monthly{sfx}.csv")


def load_vic_gridinfo(data_dir: str | Path = "data", node: str = "I_SHSTA", *,
                      no_gooselake: bool = False) -> pd.DataFrame:
    """VIC routing GridInfo [id, lat, lon, cell_km2, basin_km2] for one CalSim node.

    The station->grid-cell weight table of the CalSim3 VIC routing: one row per
    1/16-degree VIC cell fragment (``basin_km2`` is the cell area inside the
    node's basin; a couple of boundary cells appear as two fragments).  The
    ``no_gooselake`` variant drops the 94 cells of the endorheic Goose Lake
    over-reach (the basis of the ``_no_gooselake`` routed series).
    """
    sfx = "_no_gooselake" if no_gooselake else ""
    return read_table(calsim_dir(data_dir) / f"vic_gridinfo_{node}{sfx}.csv")


def load_calsim3_monthly(data_dir: str | Path = "data") -> pd.DataFrame:
    """CalSim3 historical monthly inflow [date, arc, flow_taf] (TAF/month)."""
    return read_table(calsim_dir(data_dir) / "calsim3_inflow_monthly.csv")
