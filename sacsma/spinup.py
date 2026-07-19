"""Optional climatological average-year spinup for :func:`sacsma.model.run_basin`.

Builds a repeated "average year" of forcing (a per-cell day-of-year climatology)
and prepends ``years`` of it to a run's forcing, so the model starts the real
window from an already-equilibrated state — soil moisture (including the slow
lower-zone stores), the Snow-17 pack, and the Lohmann routing ramp — instead of
the fixed cold start (SMA ``[0,0,100,100,100,0]`` / Snow-17 zeros) run from the
record start.

It is deliberately a **forcing-level** operation: it manipulates a
:class:`sacsma.model.DomainForcing` and reuses the existing run path unchanged, so
the frozen physics and the numba kernels are never touched and there is no effect
at all unless ``run_basin(..., spinup_years=...)`` is set.  Because the core
``lohmann`` convolves the whole inflow series, prepending forcing also warms the
routing ramp for free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import doy_and_leap


def average_year_luts(forcing) -> dict[str, np.ndarray]:
    """Per-cell day-of-year (1..366) climatological mean of each forcing variable.

    Returns ``{var: (n_cells, 366)}`` for ``'prcp'``/``'tavg'`` and, when present
    on the forcing, ``'tmin'``/``'tmax'``.  Day-of-year 366 (Feb 29) is filled
    from its leap-day samples; if a record never sees Feb 29 it falls back to the
    day-365 value.  The result is cached on the forcing so repeated basins in one
    ``ALL`` run reuse the same climatology.
    """
    cached = forcing._f64.get("_spinup_luts")
    if cached is not None:
        return cached

    doy = np.asarray(forcing.doy, dtype=np.int64)  # 1..366
    series = {"prcp": forcing.prcp, "tavg": forcing.tavg}
    if getattr(forcing, "tmin", None) is not None:
        series["tmin"] = forcing.tmin
    if getattr(forcing, "tmax", None) is not None:
        series["tmax"] = forcing.tmax

    n_cells = forcing.prcp.shape[0]
    luts: dict[str, np.ndarray] = {}
    for name, arr in series.items():
        lut = np.empty((n_cells, 366), dtype=arr.dtype)
        for d in range(1, 367):
            mask = doy == d
            if mask.any():
                lut[:, d - 1] = arr[:, mask].mean(axis=1)
            elif d > 1:  # no Feb-29 in the record: reuse day 365
                lut[:, d - 1] = lut[:, d - 2]
            else:  # pragma: no cover - a record with no doy==1 is impossible
                lut[:, d - 1] = arr.mean(axis=1)
        luts[name] = lut

    forcing._f64["_spinup_luts"] = luts
    return luts


def prepend_spinup(forcing, years: int = 20):
    """Prepend ``years`` climatological average-years to ``forcing``.

    Returns ``(extended_forcing, n_spin)``.  The spinup block uses real calendar
    dates immediately preceding ``forcing.dates[0]`` (so the leap structure is
    correct) and fills each day from the per-cell day-of-year climatology, keeping
    the seasonal cycle phase-correct across leap years.  The returned
    :class:`~sacsma.model.DomainForcing` shares ``pos`` and has the time axis
    extended by ``n_spin`` days; run the whole thing and drop the first ``n_spin``
    output rows to recover a warm-started run over the original dates.
    """
    from .model import DomainForcing  # local import avoids an import cycle

    if years <= 0:
        raise ValueError(f"spinup years must be positive, got {years!r}")

    luts = average_year_luts(forcing)
    start0 = pd.Timestamp(forcing.dates[0])
    spin_dates = pd.date_range(
        start0 - pd.DateOffset(years=years), start0 - pd.Timedelta(days=1), freq="D"
    )
    spin_doy, spin_leap = doy_and_leap(spin_dates)
    idx = spin_doy - 1  # 0..365 into the (n_cells, 366) LUT

    def _block(name: str) -> np.ndarray:
        return np.ascontiguousarray(luts[name][:, idx])

    ext_prcp = np.concatenate([_block("prcp"), forcing.prcp], axis=1)
    ext_tavg = np.concatenate([_block("tavg"), forcing.tavg], axis=1)
    ext_dates = spin_dates.append(pd.DatetimeIndex(forcing.dates))
    ext_doy = np.concatenate([spin_doy, np.asarray(forcing.doy, dtype=np.int64)])
    ext_leap = np.concatenate([spin_leap, np.asarray(forcing.is_leap, dtype=np.int64)])

    ext_tmin = ext_tmax = None
    if getattr(forcing, "tmin", None) is not None:
        ext_tmin = np.concatenate([_block("tmin"), forcing.tmin], axis=1)
        ext_tmax = np.concatenate([_block("tmax"), forcing.tmax], axis=1)

    extended = DomainForcing(
        pos=forcing.pos,
        prcp=ext_prcp,
        tavg=ext_tavg,
        dates=ext_dates,
        doy=ext_doy,
        is_leap=ext_leap,
        tmin=ext_tmin,
        tmax=ext_tmax,
    )
    return extended, len(spin_dates)
