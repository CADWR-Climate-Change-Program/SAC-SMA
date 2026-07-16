"""Vectorized differentiable pipeline: PET -> Snow-17 -> SAC-SMA -> routing.

Torch analog of ``sacsma.model``'s per-HRU loop, batched over all HRUs of a
basin (or the whole domain).  Supports chunked runs with carried state — the
Snow-17/SAC-SMA states plus the 106-day routing inflow history — so a long
record can be streamed under ``torch.no_grad()`` (fidelity benchmark, spinup)
or truncated-backprop trained chunk by chunk.

The routing unit hydrographs depend only on the parameters (not on time), so
the driver builds them once per parameter set (:func:`build_uh`) and reuses
them across chunks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import BOUNDS, CANOPY_BOUNDS
from .et_noah import (
    NoahCanopyState,
    dewpoint_depression_field,
    potential_et_priestley_taylor,
    snow_cover_albedo,
)
from .pet import hamon_raw_pet
from .routing import N_TAPS, build_uh, route
from .sma import SacState, run_sacsma
from .snow17 import Snow17State, run_snow17

#: angular frequency of the annual day-of-year cycle (mirrors sacsma.parameters).
_SEASONAL_OMEGA = 2.0 * math.pi / 365.0
#: recession rates that get a seasonal shape when the net emits their coeffs.
_SEASONAL_RECESSION = ("uzk", "lzpk", "lzsk")
#: Snow-17 melt parameters that get a day-of-year shape when the net emits their
#: coeffs — reconstructed to (N, T) and threaded per-step into run_snow17.
_SEASONAL_SNOW = ("MFMAX", "MFMIN", "MBASE")


def _seasonal(params: dict[str, torch.Tensor], name: str,
              doy: torch.Tensor,
              state: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable (N, T) reconstruction of a time-varying parameter.

    ``name(t) = clamp(mean + a_sin*sin(w*doy) + a_cos*cos(w*doy) + b*state(t), lo, hi)``
    — the day-of-year harmonic (``{name}_asin/_acos``) and/or the climate-state
    response (``{name}_dyn * state``) are added when their coeffs are present;
    zero coeffs give back the static ``params[name]`` field.
    """
    lo, hi = BOUNDS[name]
    v = params[name].unsqueeze(-1)                              # (N, 1) static base
    if f"{name}_asin" in params:
        v = (v + params[f"{name}_asin"].unsqueeze(-1) * torch.sin(_SEASONAL_OMEGA * doy)
             + params[f"{name}_acos"].unsqueeze(-1) * torch.cos(_SEASONAL_OMEGA * doy))
    if state is not None and f"{name}_dyn" in params:
        v = v + params[f"{name}_dyn"].unsqueeze(-1) * state     # (N,1)*(N,T) -> (N,T)
    return v.clamp(lo, hi)


@dataclass
class PipelineState:
    snow: Snow17State
    sac: SacState
    hist_surf: torch.Tensor   # (N, N_TAPS-1) unrouted direct-inflow history
    hist_base: torch.Tensor   # (N, N_TAPS-1) unrouted baseflow history
    canopy: NoahCanopyState | None = None   # only in et_mode="noah"

    def detach(self) -> PipelineState:
        return PipelineState(self.snow.detach(), self.sac.detach(),
                             self.hist_surf.detach(), self.hist_base.detach(),
                             None if self.canopy is None else self.canopy.detach())


def initial_state(
    n: int, device, dtype, *,
    init_mode: str = "reference",
    params: dict[str, torch.Tensor] | None = None,
    et_mode: str = "sac",
) -> PipelineState:
    """Reference protocol: Snow-17 zeros, SMA [0,0,100,100,100,0], no inflow history."""
    if init_mode == "reference":
        sac = SacState.reference_init(n, device, dtype)
    elif init_mode == "capacity":
        if params is None:
            raise ValueError("init_mode='capacity' needs the parameter dict")
        sac = SacState.capacity_init(params)
    else:
        raise ValueError(f"init_mode {init_mode!r}")
    hist = torch.zeros(n, N_TAPS - 1, device=device, dtype=dtype)
    canopy = NoahCanopyState.zeros(n, device, dtype) if et_mode == "noah" else None
    return PipelineState(snow=Snow17State.zeros(n, device, dtype), sac=sac,
                         hist_surf=hist, hist_base=hist.clone(), canopy=canopy)


def run_window(
    prcp: torch.Tensor,       # (N, T) mm/day
    tavg: torch.Tensor,       # (N, T) degC
    doy: torch.Tensor,        # (T,) float
    is_leap: torch.Tensor,    # (T,) bool
    lat_rad: torch.Tensor,    # (N,)
    elev: torch.Tensor,       # (N,)
    params: dict[str, torch.Tensor],       # (N,) per ga_optimum name (all 31)
    uh: tuple[torch.Tensor, torch.Tensor],  # (uh_direct, uh_base) from build_uh
    state: PipelineState,
    *,
    n_inc: int = 5,
    perc_mode: str = "reference",
    fracp_floor: float = 0.0,
    ninc_mode: str = "fixed",
    raw_pet: torch.Tensor | None = None,   # optional precomputed (N, T) PET base
    et_mode: str = "sac",
    canopy_params: dict[str, torch.Tensor] | None = None,  # CANOPY_LEARNED, (N,)
    tmin: torch.Tensor | None = None,      # (N, T); None -> tavg fallback
    tmax: torch.Tensor | None = None,
    veg_frac: torch.Tensor | None = None,  # (N,) observed veg fraction (Noah)
    lai: torch.Tensor | None = None,       # (N, T) observed seasonal LAI (Noah)
    noah_pet: str = "hamon",               # "hamon" | "priestley_taylor" (Noah)
    sac_pet: str = "hamon",                # PET source for the plain SAC ET path:
                                           # "priestley_taylor" swaps Hamon for the
                                           # energy-based PET (no Noah canopy module)
    pt_snow_albedo: float = 0.0,           # >0: raise PT albedo over snow (Snow-17
                                           # SWE-driven) toward this value; 0 = fixed
    pt_dewpoint_depression: float = 0.0,   # >0: max arid dewpoint depression (degC)
                                           # on the PT net-longwave term; 0 = Tdew=Tmin
    canopy_lite: bool = False,             # minimal 1-param Noah ET (et_noah.noah_lite_et_step)
    state_idx: torch.Tensor | None = None,  # (N, T) climate-state index (dynamic params)
    return_tet: bool = False,              # also return total ET (N, T) for closure
    return_swe: bool = False,              # also return Snow-17 SWE (N, T) (obs loss)
) -> tuple[torch.Tensor, PipelineState]:   # (+ tet, then swe, when requested)
    """One window through the full pipeline; returns (routed flow (N, T), state),
    with ``tet`` (total ET) appended when ``return_tet`` and the per-day Snow-17
    SWE appended when ``return_swe`` (both off by default — hot path unchanged;
    order is always flow, state, [tet], [swe]).

    ``et_mode="noah"`` runs the Noah canopy-resistance ET: ``canopy_params`` are
    the LEARNED physiology; ``veg_frac`` (static) and ``lai`` ((N,T) seasonal)
    are the OBSERVED canopy structure; ``tmin``/``tmax`` drive the radiation/VPD
    terms and are REQUIRED for the Noah ET and the Priestley-Taylor SAC PET (a
    missing pair raises — there is no synthetic tavg fallback).
    """
    # Snow-17 first: its SWE trajectory drives the snow-cover albedo of the
    # Priestley-Taylor PET below (and the SWE-observation loss via return_swe).
    # Snow-17 does not depend on the PET, so this reordering is value-identical
    # for every path that does not raise the albedo over snow (pt_snow_albedo == 0).
    albedo_swe = (raw_pet is None and pt_snow_albedo > 0.0
                  and ((et_mode == "sac" and sac_pet == "priestley_taylor")
                       or (et_mode == "noah" and noah_pet == "priestley_taylor")))
    need_swe = return_swe or albedo_swe
    # day-of-year melt-factor reconstruction (MFMAX/MFMIN/MBASE): (N, T) fields
    # threaded per-step into Snow-17.  Absent coeffs -> None -> static path.
    snow_seasonal = ({p: _seasonal(params, p, doy) for p in _SEASONAL_SNOW
                      if f"{p}_asin" in params} or None)
    snow_out = run_snow17(prcp, tavg, doy, is_leap, elev, params,
                          state=state.snow, return_swe=need_swe,
                          seasonal=snow_seasonal)
    eff_p, snow_state = snow_out[0], snow_out[1]
    swe = snow_out[2] if need_swe else None

    def _pt_potential():
        """Refined Priestley-Taylor potential ET: the energy-based PET plus the
        snow-cover-albedo and arid-dewpoint-depression corrections when enabled.
        Shared by the plain-SAC PT path and the Noah PT potential so both get the
        identical refinements (real per-cell tmin/tmax are REQUIRED — there is no
        tavg fallback: a synthetic diurnal range would diverge train from score)."""
        if tmin is None or tmax is None:
            raise ValueError(
                "priestley_taylor PET requires per-cell tmin/tmax")
        pt_kwargs = {}
        if albedo_swe:   # blend PT albedo toward snow where a pack is present
            pt_kwargs["albedo"] = snow_cover_albedo(swe, pt_snow_albedo)
        if pt_dewpoint_depression > 0.0:   # lower Tdew below Tmin in arid air
            pt_kwargs["dewpoint_depression"] = dewpoint_depression_field(
                tmin, tmax, pt_dewpoint_depression)
        return potential_et_priestley_taylor(
            tavg, tmin, tmax, doy, lat_rad, elev, **pt_kwargs)

    if raw_pet is None:
        if et_mode == "sac" and sac_pet == "priestley_taylor":
            # energy-based PET for the plain SAC ET (no Noah canopy) — Kpet still
            # scales it below, exactly as it scales Hamon.
            raw_pet = _pt_potential()
        else:
            raw_pet = hamon_raw_pet(tavg, doy, lat_rad)
    # Kpet time-varying reconstruction: day-of-year harmonic (seasonal) and/or a
    # climate-state response (dynamic) when the net emits their coeffs, else the
    # static per-HRU scalar.  Recessions stay seasonal-only (dynamic hurts).
    kpet = (_seasonal(params, "Kpet", doy, state_idx)
            if ("Kpet_asin" in params
                or ("Kpet_dyn" in params and state_idx is not None))
            else params["Kpet"].unsqueeze(-1))
    pet = kpet * raw_pet
    recession = ({p: _seasonal(params, p, doy) for p in _SEASONAL_RECESSION}
                 if "uzk_asin" in params else None)

    noah = None
    if et_mode == "noah":
        if canopy_params is None:
            raise ValueError("et_mode='noah' requires canopy_params")
        if veg_frac is None or lai is None:
            raise ValueError("et_mode='noah' requires observed veg_frac and lai")
        if tmin is None or tmax is None:
            raise ValueError("et_mode='noah' requires per-cell tmin/tmax")
        if noah_pet == "priestley_taylor":
            # energy-based potential replaces Hamon (still scaled by the learned
            # Kpet, now a mild global calibration knob on a physical PET);
            # inherits the snow-albedo / dewpoint refinements when enabled
            pet = kpet * _pt_potential()
        cp = canopy_params
        if state_idx is not None and any(k.endswith("_dyn") for k in cp):
            # climate-state response on canopy params (e.g. soil_chi): reconstruct
            # param(t)=clamp(base + b*state(t), lo, hi) as (N,T) in physical space
            # (the canopy head already did the sigmoid->[lo,hi] map for the base).
            cp = dict(cp)
            for k in [k for k in cp if k.endswith("_dyn")]:
                name = k[:-4]
                lo, hi = CANOPY_BOUNDS[name]
                b = cp.pop(k)
                cp[name] = (cp[name].unsqueeze(-1)
                            + b.unsqueeze(-1) * state_idx).clamp(lo, hi)   # (N,T)
        noah = {"tavg": tavg, "tmin": tmin, "tmax": tmax, "doy": doy,
                "lat_rad": lat_rad, "elev": elev, "cp": cp,
                "veg_frac": veg_frac, "lai": lai, "canopy": state.canopy,
                "lite": canopy_lite}
    surf, base, tet, sac_state = run_sacsma(pet, eff_p, params, state=state.sac,
                                            n_inc=n_inc, perc_mode=perc_mode,
                                            fracp_floor=fracp_floor,
                                            ninc_mode=ninc_mode,
                                            et_mode=et_mode, noah=noah,
                                            recession=recession)
    uh_direct, uh_base = uh
    flow = route(surf, uh_direct, state.hist_surf) + route(base, uh_base, state.hist_base)

    # carry the last N_TAPS-1 inflow days for the next chunk's convolution
    hist_surf = torch.cat([state.hist_surf, surf], dim=-1)[:, -(N_TAPS - 1):]
    hist_base = torch.cat([state.hist_base, base], dim=-1)[:, -(N_TAPS - 1):]
    new_state = PipelineState(snow=snow_state, sac=sac_state,
                              hist_surf=hist_surf, hist_base=hist_base,
                              canopy=noah["canopy"] if noah is not None else None)
    out = [flow, new_state]
    if return_tet:
        out.append(tet)
    if return_swe:
        out.append(swe)
    return tuple(out)


def routing_uh(params: dict[str, torch.Tensor], flowlen: torch.Tensor):
    """Build the per-HRU UH pair once per parameter set (reuse across chunks)."""
    return build_uh(params["Nres"], params["Kres"], params["Velo"], params["Diff"],
                    flowlen)
