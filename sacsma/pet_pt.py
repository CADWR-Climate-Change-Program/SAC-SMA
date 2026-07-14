"""Priestley-Taylor potential ET (Numba) — frozen-pipeline mirror of the torch
``sacsma.dpl.et_noah.potential_et_priestley_taylor`` (with its snow-cover-albedo
and arid dewpoint-depression refinements) so PT-trained dPL exports score
through the fast ``run_basin`` path instead of streaming the full record
through the torch pipeline.

A NEW module: the frozen Hamon ``pet.py`` is untouched.  Constants and formula
order replicate the torch source EXACTLY (Bristow-Campbell shortwave from the
diurnal range, FAO-56 net longwave with dewpoint ~= tmin, PT over net
radiation, G ~= 0 daily) — any change here must be re-verified against the
torch pipeline (see the numba-vs-torch PT parity check).

Refinements (both default-off => the plain fixed-albedo / Tdew=Tmin PT):

* ``snow_albedo > 0`` blends the surface albedo from 0.23 toward it by the
  smooth snow-cover fraction ``1 - exp(-SWE/15mm)`` — the daily SWE comes from
  the SAME Snow-17 the pipeline already runs (its ``swe`` output was computed
  and discarded before), so PET collapses under a pack.
* ``dewpoint_depression > 0`` lowers Tdew below Tmin by up to that many degC,
  ramped over diurnal range 8..20 degC (aridity proxy), raising the net
  longwave loss in dry basins only.
"""

from __future__ import annotations

import numpy as np

from ._compat import njit

# constants — verbatim from sacsma.dpl.et_noah (keep in sync)
_ALPHA_PT = 1.26       # Priestley-Taylor coefficient
_ALBEDO = 0.23         # FAO-56 reference-surface (snow-free) albedo
_SIGMA_SB = 4.903e-9   # Stefan-Boltzmann, MJ K-4 m-2 day-1
_LAMBDA_MJ = 2.45      # latent heat of vaporisation, MJ/kg
_GSC = 0.0820          # solar constant, MJ m-2 min-1
_BC_A, _BC_B, _BC_C = 0.7, 0.007, 2.4   # Bristow-Campbell transmittance
_SNOW_COVER_SWE_REF = 15.0   # SWE (mm) e-folding of the snow-cover ramp
_DD_TD_LO = 8.0              # diurnal range (degC) below which air stays humid
_DD_TD_HI = 20.0             # full dewpoint depression at/above this range
_EPS = 1e-6


@njit
def _pt_core(tavg, tmin, tmax, doy, lat_rad, elev, swe,
             snow_albedo, dewpoint_depression):
    """Raw PT PET (mm/day) for one HRU: (T,) arrays + scalar lat_rad/elev.

    ``swe`` is the per-day Snow-17 SWE (only read when ``snow_albedo > 0``);
    ``snow_albedo``/``dewpoint_depression`` are the refinement knobs (0 = off).
    """
    n = tavg.shape[0]
    pet = np.empty(n)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    tan_lat = np.tan(lat_rad)
    p_kpa = 101.3 * ((293.0 - 0.0065 * elev) / 293.0) ** 5.26
    gamma = 0.000665 * p_kpa
    rso_coeff = 0.75 + 2e-5 * elev
    for i in range(n):
        td = tmax[i] - tmin[i]
        if td < 0.0:
            td = 0.0
        two_pi_doy = 2.0 * np.pi * doy[i] / 365.0
        dr = 1.0 + 0.033 * np.cos(two_pi_doy)
        decl = 0.409 * np.sin(two_pi_doy - 1.39)
        ws_arg = -tan_lat * np.tan(decl)
        if ws_arg > 1.0:
            ws_arg = 1.0
        elif ws_arg < -1.0:
            ws_arg = -1.0
        ws = np.arccos(ws_arg)
        ra = (24.0 * 60.0 / np.pi) * _GSC * dr * (
            ws * sin_lat * np.sin(decl)
            + cos_lat * np.cos(decl) * np.sin(ws))
        kt = _BC_A * (1.0 - np.exp(-_BC_B * td ** _BC_C))
        rs = kt * ra                                   # Bristow-Campbell shortwave
        rso = rso_coeff * ra                           # clear-sky
        albedo = _ALBEDO
        if snow_albedo > 0.0:                          # snow-cover blend
            s = swe[i]
            if s < 0.0:
                s = 0.0
            f = 1.0 - np.exp(-s / _SNOW_COVER_SWE_REF)
            albedo = _ALBEDO + (snow_albedo - _ALBEDO) * f
        rns = (1.0 - albedo) * rs
        dd = 0.0
        if dewpoint_depression > 0.0:                  # arid dewpoint depression
            frac = (td - _DD_TD_LO) / (_DD_TD_HI - _DD_TD_LO)
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            dd = dewpoint_depression * frac
        tdew = tmin[i] - dd
        ea = 0.6108 * np.exp(17.27 * tdew / (tdew + 237.3))
        if ea < 0.0:
            ea = 0.0
        rr = rs / (rso if rso > _EPS else _EPS)
        if rr < 0.3:
            rr = 0.3
        elif rr > 1.0:
            rr = 1.0
        cloud = 1.35 * rr - 0.35
        rnl = (_SIGMA_SB * ((tmax[i] + 273.16) ** 4 + (tmin[i] + 273.16) ** 4) / 2.0
               * (0.34 - 0.14 * np.sqrt(ea)) * cloud)
        rn = rns - rnl                                 # net radiation, MJ/m2/day
        if rn < 0.0:
            rn = 0.0
        es = 0.6108 * np.exp(17.27 * tavg[i] / (tavg[i] + 237.3))
        slope = 4098.0 * es / (tavg[i] + 237.3) ** 2
        p = _ALPHA_PT * slope / (slope + gamma) * rn / _LAMBDA_MJ
        pet[i] = p if p > 0.0 else 0.0
    return pet


def pt_raw_pet(
    tavg: np.ndarray,
    tmin: np.ndarray,
    tmax: np.ndarray,
    doy: np.ndarray,
    latitude_deg: float,
    elev: float,
    swe: np.ndarray | None = None,
    snow_albedo: float = 0.0,
    dewpoint_depression: float = 0.0,
) -> np.ndarray:
    """Daily raw Priestley-Taylor PET (mm/day) — scale by ``Kpet`` exactly as
    the Hamon PET is.  ``swe`` (per-day Snow-17 SWE, mm) is required when
    ``snow_albedo > 0``."""
    tavg = np.asarray(tavg, dtype=float)
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    doy = np.asarray(doy, dtype=float)
    if snow_albedo > 0.0:
        if swe is None:
            raise ValueError("snow_albedo > 0 requires the Snow-17 SWE series")
        swe = np.asarray(swe, dtype=float)
    else:
        swe = np.zeros_like(tavg)
    return _pt_core(tavg, tmin, tmax, doy, np.deg2rad(float(latitude_deg)),
                    float(elev), swe, float(snow_albedo),
                    float(dewpoint_depression))
