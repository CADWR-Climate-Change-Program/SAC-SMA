"""Hamon potential evapotranspiration — faithful port of ``pet_hamon.m``.

Daylength uses the CBM model (Forsythe et al. 1995); PET is the Hamon
(1961) equation scaled by a coefficient ``Kpet``.  See the MATLAB
reference for the original formulation.
"""

from __future__ import annotations

import numpy as np

from ._compat import njit


@njit
def _hamon_core(tavg, doy, lat_rad, coeff):
    n = tavg.shape[0]
    pet = np.empty(n)
    sun_alt = np.sin(0.8333 * np.pi / 180.0)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    for i in range(n):
        d = doy[i]
        theta = 0.2163108 + 2.0 * np.arctan(0.9671396 * np.tan(0.0086 * (d - 186.0)))
        var_pi = np.arcsin(0.39795 * np.cos(theta))
        num = sun_alt + sin_lat * np.sin(var_pi)
        den = cos_lat * np.cos(var_pi)
        arg = num / den
        # CA latitudes keep arg in [-1, 1]; clamp defensively (no-op there).
        if arg > 1.0:
            arg = 1.0
        elif arg < -1.0:
            arg = -1.0
        daylight = 24.0 - (24.0 / np.pi) * np.arccos(arg)
        t = tavg[i]
        esat = 0.611 * np.exp(17.27 * t / (237.3 + t))
        pet[i] = coeff * 29.8 * daylight * (esat / (t + 273.2))
    return pet


def hamon_pet(
    tavg: np.ndarray,
    doy: np.ndarray,
    latitude_deg: float,
    coeff: float,
) -> np.ndarray:
    """Daily Hamon PET (mm/day).

    Parameters
    ----------
    tavg : (T,) array of daily mean air temperature (deg C).
    doy : (T,) int array of calendar day-of-year (1..365/366).
    latitude_deg : basin/HRU latitude in **degrees**.
    coeff : Hamon proportionality coefficient ``Kpet``.
    """
    tavg = np.asarray(tavg, dtype=float)
    doy = np.asarray(doy, dtype=float)
    return _hamon_core(tavg, doy, np.deg2rad(float(latitude_deg)), float(coeff))
