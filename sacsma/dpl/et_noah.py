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


def canopy_resistance_factor(
    tavg: torch.Tensor, tmin: torch.Tensor, td: torch.Tensor,
    doy: torch.Tensor, lat_rad: torch.Tensor, elev: torch.Tensor,
    sm_root: torch.Tensor,        # (N,) root-zone available-water fraction in [0,1]
    cp: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Jarvis canopy factor Bc in (0, 1) (NWS 53 Eqs 3.16-3.20 + p.98 Bc).

    Bc = (1 + dg) / (1 + dg + Rc/Ra) with the total canopy resistance
    ``Rc = Rcmin / (LAI * Fsr * Fq * FT * Fsm)`` and dg = slope/gamma.
    Higher stress factors -> lower Rc -> larger Bc -> more transpiration.
    """
    lai = cp["lai"]
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
    cp: dict[str, torch.Tensor],   # canopy params (CANOPY_BOUNDS)
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """One daily Noah ET step.

    Returns ``(effective_precip, new_state, tet)``: canopy-throughfall precip
    for the SAC water-balance step, the ET-withdrawn storages (incl. the
    updated ``wc``), and total actual ET (mm/day).  The SAC step then runs with
    ``et_mode="external"`` (no second withdrawal).
    """
    uztwc, uzfwc = st["uztwc"], st["uzfwc"]
    lztwc, lzfsc, lzfpc, adimc, wc = (
        st["lztwc"], st["lzfsc"], st["lzfpc"], st["adimc"], st["wc"])
    uztwm, lztwm = p["uztwm"], p["lztwm"]

    sig = cp["veg_frac"]
    lai = cp["lai"]
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
    bc = canopy_resistance_factor(tavg, tmin, td, doy, lat_rad, elev, sm_root, cp)
    et = (sig * ep * bc * (1.0 - wet_frac)).clamp_min(0.0)

    # bare-soil direct evaporation (Ek et al. 2003 nonlinear)
    ed = ((1.0 - sig) * ep * avail_up ** _CHI).clamp_min(0.0)

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
