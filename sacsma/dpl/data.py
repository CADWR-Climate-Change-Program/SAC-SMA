"""15cdec store -> torch tensors for the differentiable pipeline.

Wraps the existing loaders (``model.load_domain_forcing``, ``io.load_hru_table``,
``io.load_params``, ``cdec15.load_gage``) into a :class:`DomainTensors` bundle:
per-HRU static tensors, the basin aggregation matrix ``W`` (normalized
``area_weight`` per basin, exactly the ``model.py`` convention), and per-chunk
forcing gathers.  The big forcing arrays stay as CPU float32 NumPy (from
``DomainForcing``); each chunk is fancy-indexed to the HRU rows and moved to
the device on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..cdec15 import BASINS, CAL_END, load_gage
from ..io import load_hru_table, load_params
from ..model import DomainForcing, load_domain_forcing
from .config import PARAM_ORDER, validate_ga_optimum


@dataclass
class DomainTensors:
    dates: pd.DatetimeIndex
    doy: torch.Tensor          # (T,) float, on device
    is_leap: torch.Tensor      # (T,) bool, on device
    forcing: DomainForcing     # CPU float32 (cells, T)
    hrus: pd.DataFrame         # 7891 rows, reset index
    basins: tuple[str, ...]
    cell_idx: np.ndarray       # (N,) int rows into forcing arrays
    lat_rad: torch.Tensor      # (N,)
    elev: torch.Tensor         # (N,)
    flowlen: torch.Tensor      # (N,)
    W: torch.Tensor            # (B, N) basin aggregation (rows sum to 1)
    device: torch.device
    dtype: torch.dtype

    @property
    def n_hru(self) -> int:
        return len(self.cell_idx)

    @property
    def n_time(self) -> int:
        return len(self.dates)

    def chunk(self, t0: int, t1: int) -> tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor]:
        """(prcp, tavg, doy, is_leap) for days [t0, t1) gathered to HRU rows."""
        pr = torch.as_tensor(
            np.ascontiguousarray(self.forcing.prcp[self.cell_idx, t0:t1]),
        ).to(self.device, self.dtype)
        ta = torch.as_tensor(
            np.ascontiguousarray(self.forcing.tavg[self.cell_idx, t0:t1]),
        ).to(self.device, self.dtype)
        return pr, ta, self.doy[t0:t1], self.is_leap[t0:t1]

    def ga_params(self, data_dir: str = "data") -> dict[str, torch.Tensor]:
        """Archived GA optimum expanded to per-HRU (N,) tensors, bounds-asserted."""
        pdf = load_params(data_dir, domain="15cdec")
        validate_ga_optimum(pdf)
        merged = self.hrus.merge(pdf, on="key", how="left", suffixes=("", "_ga"))
        if merged[PARAM_ORDER[0]].isna().any():
            missing = merged.loc[merged[PARAM_ORDER[0]].isna(), "key"].unique()
            raise ValueError(f"{len(missing)} HRU keys missing from ga_optimum")
        return {
            name: torch.as_tensor(merged[name].to_numpy(np.float64)).to(
                self.device, self.dtype)
            for name in PARAM_ORDER
        }


@dataclass
class CalObs:
    """Observed daily gage FNF over the calibration window ONLY.

    Validation observations (after :data:`sacsma.cdec15.CAL_END`) are never
    materialized here — training and model selection cannot read them.
    ``obs_var`` is each basin's observed variance over its finite cal days,
    the fixed NNSE normalizer (so summing chunk losses reproduces the
    per-basin NSE denominator exactly).
    """

    t0: int                    # record index of the cal-window start
    t1: int                    # exclusive record index just past CAL_END
    obs: torch.Tensor          # (B, t1 - t0) mm/day, NaN where missing
    obs_var: torch.Tensor      # (B,)


def load_cal_obs(
    dom: DomainTensors,
    data_dir: str = "data",
    *,
    cal_start: str = "1988-10-01",
    cal_end: str = CAL_END,
) -> CalObs:
    t0 = int(dom.dates.searchsorted(pd.Timestamp(cal_start)))
    t1 = int(dom.dates.searchsorted(pd.Timestamp(cal_end))) + 1
    if dom.dates[t1 - 1] != pd.Timestamp(cal_end):
        raise ValueError(f"cal_end {cal_end} not in the forcing record")

    window = dom.dates[t0:t1]
    gage = load_gage(data_dir)
    arr = np.full((len(dom.basins), t1 - t0), np.nan)
    for b_i, b in enumerate(dom.basins):
        g = gage[gage["basin"] == b].set_index("date")["flow"]
        arr[b_i] = g.reindex(window).to_numpy(np.float64)
    var = np.nanvar(arr, axis=1)                 # population var over finite days
    return CalObs(
        t0=t0, t1=t1,
        obs=torch.as_tensor(arr).to(dom.device, dom.dtype),
        obs_var=torch.as_tensor(var).to(dom.device, dom.dtype),
    )


def load_domain_tensors(
    data_dir: str = "data",
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    basins: tuple[str, ...] | None = None,
) -> DomainTensors:
    device = torch.device(device)
    forcing = load_domain_forcing(data_dir, domain="15cdec")
    hrus = load_hru_table(data_dir, domain="15cdec")
    basins = tuple(basins if basins is not None else BASINS)
    hrus = hrus[hrus["basin"].isin(basins)].reset_index(drop=True)

    cell_idx = np.array([forcing.pos[k] for k in hrus["key"]], dtype=np.int64)
    lat_rad = torch.as_tensor(np.deg2rad(hrus["lat"].to_numpy(np.float64))).to(device, dtype)
    elev = torch.as_tensor(hrus["elev"].to_numpy(np.float64)).to(device, dtype)
    flowlen = torch.as_tensor(hrus["flowlen"].to_numpy(np.float64)).to(device, dtype)

    w = torch.zeros(len(basins), len(hrus), dtype=dtype)
    for b_i, b in enumerate(basins):
        rows = np.flatnonzero((hrus["basin"] == b).to_numpy())
        wt = hrus.loc[rows, "area_weight"].to_numpy(np.float64)
        w[b_i, rows] = torch.as_tensor(wt / wt.sum(), dtype=dtype)

    dates = forcing.dates
    doy = torch.as_tensor(forcing.doy.astype(np.float64)).to(device, dtype)
    is_leap = torch.as_tensor(forcing.is_leap.astype(bool)).to(device)

    return DomainTensors(dates=dates, doy=doy, is_leap=is_leap, forcing=forcing,
                         hrus=hrus, basins=basins, cell_idx=cell_idx,
                         lat_rad=lat_rad, elev=elev, flowlen=flowlen,
                         W=w.to(device), device=device, dtype=dtype)
