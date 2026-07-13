"""Noah canopy-resistance evapotranspiration for the differentiable SAC-SMA.

A torch implementation of the SAC-HT-CR evapotranspiration scheme of Koren,
Smith, Cui, Cosgrove, Werner & Zamora (2010, *NOAA Technical Report NWS 53*),
the same scheme the UMass/DWR "Decision Scaling Evaluation of Climate Risks to
the State Water Project" (2019, Annex F) proposes for the SWP watershed model.
It replaces SAC-SMA's single bulk-demand ET (the E1-E5 residual cascade) with
the Noah land-surface three-component formulation:

* **bare-soil evaporation** ``Ed`` (Ek et al. 2003, nonlinear, chi=2),
* **wet-canopy evaporation** ``Ec`` (Noilhan & Planton 1989) from an explicit
  canopy-interception store ``Wc`` (Terink et al. 2015),
* **transpiration** ``Et`` throttled by a Jarvis canopy resistance built from
  four stress factors (solar radiation, vapour-pressure deficit, air
  temperature, root-zone soil moisture; NWS 53 Eqs 3.16-3.20),

plus a **lower->upper tension redistribution** — the physical connection the
original SAC lacks, whose absence NWS 53 identifies as the cause of lower-zone
soil-moisture (and dry-basin) underestimation.

Why this is a NEW model, not a reparameterisation: the frozen ``sacsma.sma``
has the classic cascade, so a Noah-ET run is NOT scorable through
``run_basin`` — its skill is reported from the differentiable pipeline itself
(mass-balance validated).  The canopy parameters live in their own set
(:data:`sacsma.dpl.config.CANOPY_BOUNDS`); they are never written into the
28-vector ``ga_optimum`` export.

Reduced-input design (NWS 53 Sec 3.3): the scheme is driven by precipitation
and temperature only.  It is **tmin/tmax-native** — daily solar radiation via
Bristow & Campbell (1984) from the diurnal range ``Td = tmax - tmin`` and the
FAO-56 extraterrestrial radiation, and vapour pressure via the standard
dewpoint ~= tmin assumption.  When tmin/tmax are absent (the current forcing
store carries only tavg) a clearly-flagged fallback synthesises ``Td`` from a
fixed climatological amplitude and takes the dewpoint offset as a constant;
those two lines are the only non-faithful part and are removed once Livneh
tmin/tmax is re-ingested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# -- fixed constants (NWS 53 / Noah defaults) --------------------------------
_RCMAX = 5000.0        # cuticular (maximum) stomatal resistance, s/m
_RA = 230.0            # bulk atmospheric resistance, s/m (NWS 53 p.19)
_TREF_C = 25.0         # optimal transpiration temperature, degC
_CHI = 2.0             # Ek et al. (2003) nonlinear bare-soil exponent
_N_CANOPY = 0.5        # Noilhan-Planton wet-fraction exponent
_GSC = 0.0820          # solar constant, MJ m-2 min-1 (FAO-56)
# Bristow-Campbell transmittance coefficients (NWS 53 p.17)
_BC_A, _BC_B, _BC_C = 0.7, 0.007, 2.4
# tavg-only FALLBACK constant (removed once tmin/tmax are ingested): a fixed
# diurnal range, split symmetrically about tavg so tmin also serves as the
# dewpoint proxy for the VPD term.
_FALLBACK_TD = 10.0    # assumed diurnal range, degC

# -- Priestley-Taylor energy potential (noah_pet="priestley_taylor") ----------
_ALPHA_PT = 1.26       # Priestley-Taylor coefficient
_ALBEDO = 0.23         # FAO-56 reference-surface albedo
_SIGMA_SB = 4.903e-9   # Stefan-Boltzmann, MJ K-4 m-2 day-1 (FAO-56)
_LAMBDA_MJ = 2.45      # latent heat of vaporisation, MJ/kg (~20 degC)
# -- Beer's-law green-fraction seasonality (sigma tracks LAI phenology) -------
_BEER_K = 0.5          # canopy extinction coeff for shdfac = 1 - exp(-k*LAI)

# -- Noah-LITE fixed physical constants (canopy_lite path) --------------------
# The dropped, non-identifiable params are pinned here instead of learned:
# wilting point (fraction of tension capacity) and the upper-zone root fraction.
_LITE_WILT = 0.05      # wilting point (was learned wilt_frac in [0.05, 0.5])
_LITE_FROOT = 0.7      # upper-zone root fraction (was learned froot in [0.3, 0.9])

_EPS = 1e-6


@dataclass
class NoahCanopyState:
    """The one extra prognostic store the Noah ET scheme carries: canopy water."""

    wc: torch.Tensor          # (N,) intercepted canopy water, mm

    @classmethod
    def zeros(cls, n: int, device, dtype) -> NoahCanopyState:
        return cls(wc=torch.zeros(n, device=device, dtype=dtype))

    def detach(self) -> NoahCanopyState:
        return NoahCanopyState(wc=self.wc.detach())


def _sat_vapour_kpa(t_c: torch.Tensor) -> torch.Tensor:
    """Saturation vapour pressure (kPa), Tetens (FAO-56 Eq 11)."""
    return 0.6108 * torch.exp(17.27 * t_c / (t_c + 237.3))


def extraterrestrial_radiation(doy: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
    """Daily extraterrestrial radiation Ra (MJ m-2 day-1), FAO-56 Eq 21.

    Per-day/per-HRU native: ``doy`` scalar (or (N,)) and ``lat_rad`` (N,) ->
    (N,); the daily Noah ET step is the only caller (no time axis).
    """
    two_pi_doy = 2.0 * math.pi * doy / 365.0
    dr = 1.0 + 0.033 * torch.cos(two_pi_doy)                 # inverse earth-sun dist
    decl = 0.409 * torch.sin(two_pi_doy - 1.39)              # solar declination
    ws_arg = (-torch.tan(lat_rad) * torch.tan(decl)).clamp(-1.0, 1.0)
    ws = torch.acos(ws_arg)                                  # sunset hour angle
    return (24.0 * 60.0 / math.pi) * _GSC * dr * (
        ws * torch.sin(lat_rad) * torch.sin(decl)
        + torch.cos(lat_rad) * torch.cos(decl) * torch.sin(ws))


def solar_radiation_wm2(
    doy: torch.Tensor, lat_rad: torch.Tensor, td: torch.Tensor,
) -> torch.Tensor:
    """Daily-mean shortwave irradiance (W m-2) via Bristow & Campbell (1984).

    ``td`` is the diurnal temperature range tmax-tmin (degC).  Transmittance
    ``Kt = A(1 - exp(-B*Td^C))`` times FAO-56 extraterrestrial radiation, then
    MJ m-2 day-1 -> mean W m-2 (/0.0864).
    """
    ra = extraterrestrial_radiation(doy, lat_rad)
    kt = _BC_A * (1.0 - torch.exp(-_BC_B * td.clamp_min(0.0) ** _BC_C))
    return (kt * ra) / 0.0864


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

    Replaces the temperature-only Hamon PET the canopy params modulate: the
    Hamon ceiling (total ET <= Kpet*Hamon) makes Noah under-extract vs SAC.
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


def canopy_resistance_factor(
    tavg: torch.Tensor, tmin: torch.Tensor, td: torch.Tensor,
    doy: torch.Tensor, lat_rad: torch.Tensor, elev: torch.Tensor,
    sm_root: torch.Tensor,        # (N,) root-zone available-water fraction in [0,1]
    cp: dict[str, torch.Tensor],
    lai: torch.Tensor,            # (N,) observed leaf-area index (per-day seasonal)
) -> torch.Tensor:
    """Jarvis canopy factor Bc in (0, 1) (NWS 53 Eqs 3.16-3.20 + p.98 Bc).

    Bc = (1 + dg) / (1 + dg + Rc/Ra) with the total canopy resistance
    ``Rc = Rcmin / (LAI * Fsr * Fq * FT * Fsm)`` and dg = slope/gamma.
    Higher stress factors -> lower Rc -> larger Bc -> more transpiration.
    ``lai`` is the observed (pinned) leaf-area index, not a learned param.
    """
    rcmin = cp["rcmin"]
    rgl = cp["rgl"]
    hs = cp["hs"]

    # F_sr : photosynthetically-active radiation (0.55 of shortwave)
    rg = solar_radiation_wm2(doy, lat_rad, td)
    f = 0.55 * 2.0 * rg / (rgl * lai).clamp_min(_EPS)
    fsr = (rcmin / _RCMAX + f) / (1.0 + f)

    # F_q : vapour-pressure-deficit stress (dewpoint ~= tmin)
    p_kpa = _atm_pressure_kpa(elev)
    es = _sat_vapour_kpa(tavg)
    ea = _sat_vapour_kpa(tmin)
    qs = 0.622 * es / (p_kpa - 0.378 * es)
    qa = 0.622 * ea / (p_kpa - 0.378 * ea)
    fq = 1.0 / (1.0 + hs * (qs - qa).clamp_min(0.0))

    # F_T : air-temperature stress (parabola peaked at Tref)
    ft = (1.0 - 0.0016 * (_TREF_C - tavg) ** 2).clamp(0.01, 1.0)

    # F_sm : root-zone soil-moisture stress (already in [0,1])
    fsm = sm_root.clamp(0.01, 1.0)

    rc = rcmin / (lai * fsr * fq * ft * fsm).clamp_min(_EPS)

    # dg = slope of saturation vapour curve / psychrometric constant
    slope = 4098.0 * es / (tavg + 237.3) ** 2
    gamma = 0.000665 * p_kpa
    dg = slope / gamma
    return (1.0 + dg) / (1.0 + dg + rc / _RA)


def noah_et_step(
    st: dict[str, torch.Tensor],   # uztwc/uzfwc/lztwc/lzfsc/lzfpc/adimc + wc
    precip: torch.Tensor,          # (N,) mm/day rain+melt reaching the surface
    ep: torch.Tensor,              # (N,) mm/day potential evaporation (Hamon base)
    tavg: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor,
    doy: torch.Tensor, lat_rad: torch.Tensor, elev: torch.Tensor,
    p: dict[str, torch.Tensor],    # SAC params (uztwm, lztwm, ...)
    cp: dict[str, torch.Tensor],   # LEARNED canopy physiology (CANOPY_LEARNED_PARAMS)
    veg_frac: torch.Tensor,        # (N,) observed green-vegetation fraction (static)
    lai: torch.Tensor,             # (N,) observed leaf-area index (per-day seasonal)
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """One daily Noah ET step.

    Returns ``(effective_precip, new_state, tet)``: canopy-throughfall precip
    for the SAC water-balance step, the ET-withdrawn storages (incl. the
    updated ``wc``), and total actual ET (mm/day).  The SAC step then runs with
    ``et_mode="external"`` (no second withdrawal).

    ``veg_frac`` and ``lai`` are OBSERVED (pinned per cell), passed in rather
    than read from ``cp`` — only the physiology params are learned.
    """
    uztwc, uzfwc = st["uztwc"], st["uzfwc"]
    lztwc, lzfsc, lzfpc, adimc, wc = (
        st["lztwc"], st["lzfsc"], st["lzfpc"], st["adimc"], st["wc"])
    uztwm, lztwm = p["uztwm"], p["lztwm"]

    # green vegetation fraction tracks LAI phenology (Beer's law), capped at the
    # observed total cover: sparse/dormant (low LAI) -> less canopy, more bare
    # soil exposed; summer saturates at veg_frac.  Uses the seasonal lai already
    # passed, so no extra field is threaded.
    sig = torch.minimum(veg_frac, 1.0 - torch.exp(-_BEER_K * lai))
    wilt = cp["wilt_frac"]
    froot = cp["froot"]
    kdiff = cp["redist_k"]

    td = tmax - tmin

    # -- root-zone available-water fraction (SAC tension saturations) --------
    sup = (uztwc / uztwm).clamp(0.0, 1.0)
    slo = (lztwc / lztwm).clamp(0.0, 1.0)
    avail_up = ((sup - wilt) / (1.0 - wilt)).clamp(0.0, 1.0)
    avail_lo = ((slo - wilt) / (1.0 - wilt)).clamp(0.0, 1.0)
    sm_root = froot * avail_up + (1.0 - froot) * avail_lo

    # -- canopy interception + throughfall (Terink et al. 2015) --------------
    smax = (0.935 + 0.498 * lai - 0.00575 * lai * lai).clamp_min(_EPS)
    wc1 = wc + sig * precip
    throughfall = (wc1 - smax).clamp_min(0.0)
    wc2 = wc1 - throughfall
    eff_precip = (1.0 - sig) * precip + throughfall

    # (Wc/Smax)^0.5 has an INFINITE derivative at Wc=0 (the common dry-canopy
    # case) -> floor the base so the fractional-power backward stays finite
    # (offset ~3e-3, physically negligible).
    wet_frac = (wc2 / smax).clamp(_EPS, 1.0) ** _N_CANOPY

    # -- three ET components -------------------------------------------------
    # wet-canopy evaporation (capped by the canopy store)
    ec = torch.minimum(sig * ep * wet_frac, wc2)
    wc_new = wc2 - ec

    # transpiration (Jarvis canopy factor, suppressed by canopy wetness)
    bc = canopy_resistance_factor(tavg, tmin, td, doy, lat_rad, elev, sm_root, cp, lai)
    et = (sig * ep * bc * (1.0 - wet_frac)).clamp_min(0.0)

    # bare-soil direct evaporation (Ek et al. 2003 nonlinear); the exponent is a
    # LEARNED per-cell canopy param (soil_chi), so sparse-veg dry basins can
    # soften the avail^chi throttle (which collapses bare-soil ET to ~0 on dry
    # soil) instead of being stuck at the fixed chi=2.  Floor the base so the
    # fractional-power backward (chi<1 at avail->0) stays finite — same guard as
    # wet_frac; offset ~EPS^chi is physically negligible.
    ed = ((1.0 - sig) * ep * avail_up.clamp_min(_EPS) ** cp["soil_chi"]).clamp_min(0.0)

    # -- withdraw soil ET (bare soil + transpiration by root fraction) -------
    et_up, et_lo = froot * et, (1.0 - froot) * et
    # upper zone: bare soil + upper transpiration from tension, then free
    dem_up = ed + et_up
    w_upt = torch.minimum(dem_up, uztwc)
    uztwc = uztwc - w_upt
    w_upf = torch.minimum(dem_up - w_upt, uzfwc)
    uzfwc = uzfwc - w_upf
    # lower zone: lower transpiration from tension
    w_lo = torch.minimum(et_lo, lztwc)
    lztwc = lztwc - w_lo
    soil_et = w_upt + w_upf + w_lo

    # -- lower<->upper tension redistribution (the connection SAC lacks) -----
    # move water down the tension-saturation gradient; slo > sup pulls UP
    # (the dry-basin fix).  Clamp so neither store crosses 0 or capacity.
    sup2 = uztwc / uztwm
    slo2 = lztwc / lztwm
    flux = kdiff * (slo2 - sup2) * torch.minimum(uztwm, lztwm)
    flux = torch.clamp(flux, -lztwc, uztwm - uztwc)     # don't overfill UZ / drain LZ<0
    flux = torch.clamp(flux, uztwc - uztwm, lztwc)      # don't drain UZ<0 / overfill LZ
    uztwc = uztwc + flux
    lztwc = lztwc - flux

    tet = soil_et + ec
    new_state = {
        "uztwc": uztwc, "uzfwc": uzfwc, "lztwc": lztwc,
        "lzfsc": lzfsc, "lzfpc": lzfpc, "adimc": adimc, "wc": wc_new,
    }
    return eff_precip, new_state, tet


def noah_lite_et_step(
    st: dict[str, torch.Tensor],   # uztwc/uzfwc/lztwc/lzfsc/lzfpc/adimc + wc
    precip: torch.Tensor,          # (N,) mm/day rain+melt reaching the surface
    ep: torch.Tensor,              # (N,) mm/day potential ET (Hamon or PT base)
    p: dict[str, torch.Tensor],    # SAC params (uztwm, lztwm, ...)
    cp: dict[str, torch.Tensor],   # LEARNED canopy params — LITE: soil_chi only
    veg_frac: torch.Tensor,        # (N,) observed green-vegetation fraction (static)
    lai: torch.Tensor,             # (N,) observed leaf-area index (per-day seasonal)
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Minimal, identifiable Noah ET (``canopy_lite``): one daily step.

    ``AET = ed_bare + et_canopy`` on the pinned green fraction, with a SINGLE
    learned exponent ``soil_chi`` as the moisture-limiter shape — the one ET
    knob a streamflow calibration can identify.  The Jarvis transpiration
    resistance (rcmin/rgl/hs), the canopy-interception store and the learned
    UZ<->LZ redistribution are all dropped; the wilting point and root split are
    pinned (:data:`_LITE_WILT`, :data:`_LITE_FROOT`).

    * bare soil (surface, fast dry-down): ``ed = (1-sig)*ep*avail_up^chi``
    * canopy (root zone, sustained):      ``et = sig*ep*sm_root^chi``

    Returns ``(effective_precip, new_state, tet)`` exactly like
    :func:`noah_et_step` (interception off => effective precip == precip; the
    canopy store ``wc`` is carried through unchanged so the state shape and the
    graph buffers match the full path).
    """
    uztwc, uzfwc, lztwc = st["uztwc"], st["uzfwc"], st["lztwc"]
    uztwm, lztwm = p["uztwm"], p["lztwm"]
    chi = cp["soil_chi"]
    wilt, froot = _LITE_WILT, _LITE_FROOT

    # green vegetation fraction tracks LAI phenology (Beer's law), capped at the
    # observed total cover (same pinned structure as the full path).
    sig = torch.minimum(veg_frac, 1.0 - torch.exp(-_BEER_K * lai))

    # root-zone available-water fractions (SAC tension saturations, wilt floor)
    sup = (uztwc / uztwm).clamp(0.0, 1.0)
    slo = (lztwc / lztwm).clamp(0.0, 1.0)
    avail_up = ((sup - wilt) / (1.0 - wilt)).clamp(0.0, 1.0)
    avail_lo = ((slo - wilt) / (1.0 - wilt)).clamp(0.0, 1.0)
    sm_root = froot * avail_up + (1.0 - froot) * avail_lo

    # bare-soil (surface) + canopy (root zone) ET, ONE shared exponent.  Floor
    # the fractional-power bases so the backward stays finite at avail->0 (chi<1),
    # same guard as the full path (offset ~EPS^chi is physically negligible).
    ed = ((1.0 - sig) * ep * avail_up.clamp_min(_EPS) ** chi).clamp_min(0.0)
    et = (sig * ep * sm_root.clamp_min(_EPS) ** chi).clamp_min(0.0)

    # withdraw: bare soil + upper transpiration from UZ (tension then free);
    # lower transpiration from LZ tension — same allocation as the full path.
    et_up, et_lo = froot * et, (1.0 - froot) * et
    dem_up = ed + et_up
    w_upt = torch.minimum(dem_up, uztwc)
    uztwc = uztwc - w_upt
    w_upf = torch.minimum(dem_up - w_upt, uzfwc)
    uzfwc = uzfwc - w_upf
    w_lo = torch.minimum(et_lo, lztwc)
    lztwc = lztwc - w_lo
    tet = w_upt + w_upf + w_lo

    new_state = {
        "uztwc": uztwc, "uzfwc": uzfwc, "lztwc": lztwc,
        "lzfsc": st["lzfsc"], "lzfpc": st["lzfpc"], "adimc": st["adimc"],
        "wc": st["wc"],           # interception off — canopy store rides unchanged
    }
    return precip, new_state, tet


def canopy_inputs_fallback(
    tavg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesise (tmin, tmax) from tavg when the forcing lacks them.

    A stopgap for the current tavg-only store: a fixed diurnal amplitude and
    dewpoint offset.  NOT faithful — replaced by real Livneh tmin/tmax after
    re-ingest (:data:`_FALLBACK_TD`, :data:`_FALLBACK_DEWDEP`).
    """
    tmin = tavg - 0.5 * _FALLBACK_TD
    tmax = tavg + 0.5 * _FALLBACK_TD
    return tmin, tmax
