"""The Priestley-Taylor PET and the Noah-lite ET of the learned model.

**Priestley-Taylor potential ET** (:func:`potential_et_priestley_taylor`): net radiation from
the Bristow & Campbell (1984) shortwave (diurnal temperature range) and the FAO-56 net longwave
(dewpoint ~= tmin).  Real per-cell tmin/tmax are required; the callers raise if they are
absent.

**Noah-lite ET** (:func:`noah_lite_et`): bare-soil evaporation (Ek et al. 2003) and canopy
transpiration on the observed green fraction, withdrawn from the SAC-SMA tension stores ahead of
the water balance (Koren et al. 2010, NWS 53, reduced to its one streamflow-identifiable
parameter, the moisture exponent ``soil_chi``).  The wilting point and the upper-zone root
fraction are pinned.  ``sacsma.sma_learned`` and ``sacsma.pet_pt`` are the Numba copies.
"""

from __future__ import annotations

import math

import torch

_GSC = 0.0820          # solar constant, MJ m-2 min-1 (FAO-56)
# Bristow-Campbell transmittance coefficients (NWS 53 p.17)
_BC_A, _BC_B, _BC_C = 0.7, 0.007, 2.4

# -- Priestley-Taylor energy potential -----------------------------------------
_ALPHA_PT = 1.26       # Priestley-Taylor coefficient
_ALBEDO = 0.23         # FAO-56 reference-surface (snow-free) albedo
_SIGMA_SB = 4.903e-9   # Stefan-Boltzmann, MJ K-4 m-2 day-1 (FAO-56)
_LAMBDA_MJ = 2.45      # latent heat of vaporisation, MJ/kg (~20 degC)

# -- Beer's-law green-fraction seasonality (sigma tracks LAI phenology) -------
_BEER_K = 0.5          # canopy extinction coeff for shdfac = 1 - exp(-k*LAI)

# -- Noah-lite pinned constants -------------------------------------------------
_LITE_WILT = 0.05      # wilting point, fraction of tension capacity
_LITE_FROOT = 0.7      # upper-zone root fraction

_EPS = 1e-6


def _sat_vapour_kpa(t_c: torch.Tensor) -> torch.Tensor:
    """Saturation vapour pressure (kPa), Tetens (FAO-56 Eq 11)."""
    return 0.6108 * torch.exp(17.27 * t_c / (t_c + 237.3))


def _atm_pressure_kpa(elev: torch.Tensor) -> torch.Tensor:
    """Mean atmospheric pressure (kPa) from elevation, FAO-56 Eq 7."""
    return 101.3 * ((293.0 - 0.0065 * elev) / 293.0) ** 5.26


def potential_et_priestley_taylor(
    tavg: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor,   # (N, T) degC
    doy: torch.Tensor,        # (T,) or (N, T)
    lat_rad: torch.Tensor,    # (N,)
    elev: torch.Tensor,       # (N,)
) -> torch.Tensor:
    """Energy-based potential ET (mm/day), Priestley-Taylor over net radiation.

    Net radiation Rn = (1-albedo)*Rs - Rnl with Rs the Bristow-Campbell shortwave
    (diurnal range) and Rnl the FAO-56 net longwave (dewpoint~=tmin); then
    ``PET = alpha_PT * s/(s+gamma) * Rn / lambda`` (soil heat flux G~=0 daily).
    Fully vectorised over (N, T) — computed once per window in the driver.
    """
    lat = lat_rad.unsqueeze(-1)                        # (N, 1)
    el = elev.unsqueeze(-1)                            # (N, 1)
    td = (tmax - tmin).clamp_min(0.0)
    two_pi_doy = 2.0 * math.pi * doy / 365.0
    dr = 1.0 + 0.033 * torch.cos(two_pi_doy)
    decl = 0.409 * torch.sin(two_pi_doy - 1.39)
    ws = torch.acos((-torch.tan(lat) * torch.tan(decl)).clamp(-1.0, 1.0))
    ra = (24.0 * 60.0 / math.pi) * _GSC * dr * (       # extraterrestrial, MJ/m2/day
        ws * torch.sin(lat) * torch.sin(decl)
        + torch.cos(lat) * torch.cos(decl) * torch.sin(ws))
    kt = _BC_A * (1.0 - torch.exp(-_BC_B * td ** _BC_C))
    rs = kt * ra                                       # Bristow-Campbell shortwave
    rso = (0.75 + 2e-5 * el) * ra                      # clear-sky
    rns = (1.0 - _ALBEDO) * rs
    ea = _sat_vapour_kpa(tmin)                         # actual vapour pressure
    cloud = (1.35 * (rs / rso.clamp_min(_EPS)).clamp(0.3, 1.0) - 0.35)
    rnl = (_SIGMA_SB * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0
           * (0.34 - 0.14 * ea.clamp_min(0.0).sqrt()) * cloud)
    rn = rns - rnl                                     # net radiation, MJ/m2/day
    es = _sat_vapour_kpa(tavg)
    slope = 4098.0 * es / (tavg + 237.3) ** 2
    gamma = 0.000665 * _atm_pressure_kpa(el)
    pet = _ALPHA_PT * slope / (slope + gamma) * rn.clamp_min(0.0) / _LAMBDA_MJ
    return pet.clamp_min(0.0)


def noah_lite_et(
    uztwc: torch.Tensor, uzfwc: torch.Tensor, lztwc: torch.Tensor,   # (N,) mm
    ep: torch.Tensor,              # (N,) mm/day potential ET (Hamon or PT base)
    p: dict[str, torch.Tensor],    # SAC params (uztwm, lztwm)
    chi: torch.Tensor,             # (N,) learned moisture exponent soil_chi
    veg_frac: torch.Tensor,        # (N,) observed green-vegetation fraction (static)
    lai: torch.Tensor,             # (N,) observed leaf-area index of the day
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One day of Noah-lite ET; returns ``(uztwc, uzfwc, lztwc, tet)`` after the withdrawal.

    * bare soil (surface, fast dry-down): ``ed = (1-sig)*ep*avail_up^chi``
    * canopy (root zone, sustained):      ``et = sig*ep*sm_root^chi``

    ``sig`` is the green fraction, Beer's law on the LAI capped at the observed cover.
    """
    uztwm, lztwm = p["uztwm"], p["lztwm"]
    wilt, froot = _LITE_WILT, _LITE_FROOT
    sig = torch.minimum(veg_frac, 1.0 - torch.exp(-_BEER_K * lai))

    # root-zone available-water fractions (SAC tension saturations, wilt floor)
    sup = (uztwc / uztwm).clamp(0.0, 1.0)
    slo = (lztwc / lztwm).clamp(0.0, 1.0)
    avail_up = ((sup - wilt) / (1.0 - wilt)).clamp(0.0, 1.0)
    avail_lo = ((slo - wilt) / (1.0 - wilt)).clamp(0.0, 1.0)
    sm_root = froot * avail_up + (1.0 - froot) * avail_lo

    # one shared exponent; the bases are floored so the backward stays finite at avail -> 0
    ed = ((1.0 - sig) * ep * avail_up.clamp_min(_EPS) ** chi).clamp_min(0.0)
    et = (sig * ep * sm_root.clamp_min(_EPS) ** chi).clamp_min(0.0)

    # bare soil + upper transpiration from the upper zone (tension, then free); lower
    # transpiration from the lower tension
    et_up, et_lo = froot * et, (1.0 - froot) * et
    dem_up = ed + et_up
    w_upt = torch.minimum(dem_up, uztwc)
    uztwc = uztwc - w_upt
    w_upf = torch.minimum(dem_up - w_upt, uzfwc)
    uzfwc = uzfwc - w_upf
    w_lo = torch.minimum(et_lo, lztwc)
    lztwc = lztwc - w_lo
    return uztwc, uzfwc, lztwc, w_upt + w_upf + w_lo
