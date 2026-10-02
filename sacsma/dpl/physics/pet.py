"""Differentiable Hamon PET — torch mirror of the frozen ``sacsma.pet``.

Same CBM daylength (Forsythe et al. 1995) and Hamon (1961) equation as
``sacsma/pet.py::_hamon_core``, vectorised over HRUs.  ``Kpet`` multiplies a
coefficient-free base, so the base can be precomputed ONCE per record
(:func:`hamon_raw_pet`) and the only gradient path is through the coefficient:
``pet = kpet[:, None] * raw``.

Faithfulness notes vs the reference: the daylength ``arccos`` argument is
clamped to exactly [-1, 1] (a no-op at CA latitudes, zero-gradient only in the
clamped region); there is NO output clamp (the reference has none — PET is
positive by construction for tavg > -273.2 degC).
"""

from __future__ import annotations

import math

import numpy as np
import torch

_SUN_ALT = math.sin(0.8333 * math.pi / 180.0)


def hamon_raw_pet(
    tavg: torch.Tensor,       # (N, T) degC
    doy: torch.Tensor,        # (T,) or (N, T) float day-of-year 1..365/366
    lat_rad: torch.Tensor,    # (N,) latitude in radians
) -> torch.Tensor:
    """Coefficient-free Hamon PET base ``29.8 * daylight * esat/(tavg+273.2)``, (N, T)."""
    theta = 0.2163108 + 2.0 * torch.atan(0.9671396 * torch.tan(0.0086 * (doy - 186.0)))
    var_pi = torch.asin(0.39795 * torch.cos(theta))
    lat = lat_rad.unsqueeze(-1)
    arg = (_SUN_ALT + torch.sin(lat) * torch.sin(var_pi)) / (
        torch.cos(lat) * torch.cos(var_pi)
    )
    arg = arg.clamp(-1.0, 1.0)
    daylight = 24.0 - (24.0 / math.pi) * torch.acos(arg)
    esat = 0.611 * torch.exp(17.27 * tavg / (237.3 + tavg))
    return 29.8 * daylight * (esat / (tavg + 273.2))


def hamon_pet(
    tavg: torch.Tensor,
    doy: torch.Tensor,
    lat_rad: torch.Tensor,
    kpet: torch.Tensor,       # (N,)
) -> torch.Tensor:
    """Daily Hamon PET (mm/day), (N, T)."""
    return kpet.unsqueeze(-1) * hamon_raw_pet(tavg, doy, lat_rad)


def hamon_raw_pet_numpy(
    tavg: np.ndarray,         # (N, T) degC
    doy: np.ndarray,          # (T,) or (N, T)
    lat_rad: np.ndarray,      # (N,)
) -> np.ndarray:
    """NumPy twin of :func:`hamon_raw_pet` (float64) — used for the one-time
    precompute and for the Kpet-independent aridity climate index."""
    theta = 0.2163108 + 2.0 * np.arctan(0.9671396 * np.tan(0.0086 * (doy - 186.0)))
    var_pi = np.arcsin(0.39795 * np.cos(theta))
    lat = np.asarray(lat_rad, dtype=np.float64)[..., np.newaxis]
    arg = (_SUN_ALT + np.sin(lat) * np.sin(var_pi)) / (np.cos(lat) * np.cos(var_pi))
    arg = np.clip(arg, -1.0, 1.0)
    daylight = 24.0 - (24.0 / np.pi) * np.arccos(arg)
    tavg = np.asarray(tavg, dtype=np.float64)
    esat = 0.611 * np.exp(17.27 * tavg / (237.3 + tavg))
    return 29.8 * daylight * (esat / (tavg + 273.2))
