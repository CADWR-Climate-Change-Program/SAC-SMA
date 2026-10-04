"""Running the calibrated models: a domain's GA optimum through the reference model.

Per HRU: Hamon PET -> Snow-17 -> SAC-SMA -> Lohmann routing, from the cold start on the first day
of the forcing, on the engine (:mod:`sacsma.engine`); a basin's flow is the area-weighted sum of
its routed HRU flow (mm/day).  This is the run from the archived calibration.  The basins of a
domain run together, in one engine run (:func:`run_basins`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import parameters as P
from .engine import Field, Physics, simulate
from .io import (
    DEFAULT_DOMAIN,
    DEFAULT_FORCING,
    doy_and_leap,
    load_forcing,
    load_hru_table,
    load_params,
)


@dataclass
class DomainForcing:
    """A domain's forcing in memory, read once and shared by every run on it.

    ``prcp`` / ``tavg`` are ``(cells, days)`` as stored (float32); ``pos`` maps a grid-cell
    ``key`` to its row; ``doy`` / ``is_leap`` belong to ``dates``.  ``tmin`` / ``tmax`` are
    attached where a learned run with the Priestley-Taylor PET needs them."""

    pos: dict[str, int]
    prcp: np.ndarray
    tavg: np.ndarray
    dates: pd.DatetimeIndex
    doy: np.ndarray
    is_leap: np.ndarray
    tmin: np.ndarray | None = None
    tmax: np.ndarray | None = None


def load_domain_forcing(
    data_dir: str | Path,
    *,
    domain: str = DEFAULT_DOMAIN,
    start: str | None = None,
    end: str | None = None,
    product: str = DEFAULT_FORCING,
) -> DomainForcing:
    """Read the domain's forcing store ``product`` into memory, over ``start``..``end``.

    One contiguous read per variable (seconds); indexing ~2000 non-contiguous keys in the
    compressed store instead would decompress overlapping chunks per key (minutes)."""
    ds = load_forcing(data_dir, domain=domain, product=product)
    try:
        if start is not None or end is not None:
            ds = ds.sel(time=slice(start, end))
        dates = pd.DatetimeIndex(ds["time"].values)
        doy, is_leap = doy_and_leap(dates)
        prcp = ds["prcp"].values
        tavg = ds["tavg"].values
        pos = {str(k): i for i, k in enumerate(ds["key"].values)}
    finally:
        ds.close()
    return DomainForcing(pos=pos, prcp=prcp, tavg=tavg, dates=dates, doy=doy, is_leap=is_leap)


def simulate_ga(hrus: pd.DataFrame, params: pd.DataFrame, weights: np.ndarray,
                forcing: DomainForcing, output: str = "flow", *,
                physics: Physics | None = None) -> np.ndarray:
    """``weights`` (K, rows) sums over the rows of ``hrus`` (``key``, ``lat``, ``elev``,
    ``flowlen``), each run with the GA parameters in the same row of ``params``, over the whole
    of ``forcing`` from the cold start: (K, days) mm/day of the routed flow (``output="flow"``)
    or of the runoff before routing (``"runoff"``).  ``physics`` (default: the reference model)
    runs them through other numerics (the benchmark)."""
    keys = hrus["key"].to_numpy()
    col = lambda names: params[list(names)].to_numpy(float)  # noqa: E731
    field = Field.from_rows(
        physics or Physics(), np.fromiter((forcing.pos[k] for k in keys), np.int64, len(keys)),
        np.deg2rad(hrus["lat"].to_numpy(float)), hrus["elev"].to_numpy(float),
        hrus["flowlen"].to_numpy(float), params["Kpet"].to_numpy(float), col(P._SNOW_COLS),
        col(P._SMA_COLS), col(P._ROUT_COLS))
    return simulate(field, forcing, forcing.dates[0], forcing.dates[-1], weights, spinup="cold",
                    outputs=(output,))[output]


def area_weights(hrus: pd.DataFrame, basins) -> np.ndarray:
    """(basins, rows) each basin's HRU ``area_weight`` over the rows of ``hrus``, summing to 1."""
    codes = pd.Categorical(hrus["basin"], categories=list(basins)).codes
    missing = sorted(set(basins) - set(hrus["basin"]))
    if missing:
        raise ValueError(f"no HRUs for basin(s) {missing}")
    rows = np.flatnonzero(codes >= 0)
    W = np.zeros((len(basins), len(hrus)))
    W[codes[rows], rows] = hrus["area_weight"].to_numpy(float)[rows]
    return W / W.sum(axis=1, keepdims=True)


def run_basins(
    basins=None,
    *,
    data_dir: str | Path = "data",
    domain: str = DEFAULT_DOMAIN,
    start: str | None = None,
    end: str | None = None,
    forcing: DomainForcing | None = None,
    product: str = DEFAULT_FORCING,
    physics: Physics | None = None,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Forward-simulate ``basins`` of ``domain`` (default: all, sorted) from their GA optimum in
    one engine run: the daily gauge flow (mm/day), date x basin.  A basin's flow weighs its HRU
    rows by ``weights`` (basins, rows of the domain's HRU table; default :func:`area_weights`).
    ``forcing`` (:func:`load_domain_forcing`) reuses a read; without it the store ``product``
    is read over ``start``..``end``.  ``physics`` as in :func:`simulate_ga`."""
    if forcing is None:
        forcing = load_domain_forcing(data_dir, domain=domain, start=start, end=end,
                                      product=product)
    hrus = load_hru_table(data_dir, domain=domain)
    names = sorted(hrus["basin"].unique()) if basins is None else list(basins)
    W = area_weights(hrus, names) if weights is None else np.asarray(weights, float)
    use = W.any(axis=0)
    hrus = hrus[use].reset_index(drop=True)
    params = load_params(domain=domain)
    # a per-watershed calibration (a ``basin`` column) repeats shared cells with its own values
    on = ["basin", "key"] if "basin" in params.columns else ["key"]
    pr = hrus[on].merge(params, on=on, how="left", validate="many_to_one")
    if pr["Kpet"].isna().any():
        raise ValueError(f"{int(pr['Kpet'].isna().sum())} HRU rows have no GA parameters")
    flow = simulate_ga(hrus, pr, W[:, use], forcing, physics=physics)
    return pd.DataFrame(flow.T, index=forcing.dates.rename("date"), columns=names)
