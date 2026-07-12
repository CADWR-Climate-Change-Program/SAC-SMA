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

from .config import BOUNDS
from .et_noah import NoahCanopyState, canopy_inputs_fallback
from .pet import hamon_raw_pet
from .routing import N_TAPS, build_uh, route
from .sma import SacState, run_sacsma
from .snow17 import Snow17State, run_snow17

#: angular frequency of the annual day-of-year cycle (mirrors sacsma.parameters).
_SEASONAL_OMEGA = 2.0 * math.pi / 365.0
#: recession rates that get a seasonal shape when the net emits their coeffs.
_SEASONAL_RECESSION = ("uzk", "lzpk", "lzsk")


def _seasonal(params: dict[str, torch.Tensor], name: str,
              doy: torch.Tensor) -> torch.Tensor:
    """Differentiable (N, T) day-of-year reconstruction of a seasonal parameter.

    ``name(doy) = clamp(mean + a_sin*sin(w*doy) + a_cos*cos(w*doy), lo, hi)`` —
    the same harmonic the frozen model reconstructs (:mod:`sacsma.parameters`),
    so zero amplitude coefficients give back the static ``params[name]`` field.
    """
    lo, hi = BOUNDS[name]
    sin_t = torch.sin(_SEASONAL_OMEGA * doy)
    cos_t = torch.cos(_SEASONAL_OMEGA * doy)
    v = (params[name].unsqueeze(-1)
         + params[f"{name}_asin"].unsqueeze(-1) * sin_t
         + params[f"{name}_acos"].unsqueeze(-1) * cos_t)
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
    canopy_params: dict[str, torch.Tensor] | None = None,  # CANOPY_BOUNDS, (N,)
    tmin: torch.Tensor | None = None,      # (N, T); None -> tavg fallback
    tmax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, PipelineState]:
    """One window through the full pipeline; returns (routed flow (N, T), state).

    ``et_mode="noah"`` runs the Noah canopy-resistance ET (``canopy_params``
    required); ``tmin``/``tmax`` drive the radiation/VPD terms (a fixed-diurnal
    fallback is synthesised from ``tavg`` when they are absent).
    """
    if raw_pet is None:
        raw_pet = hamon_raw_pet(tavg, doy, lat_rad)
    # seasonal Kpet (day-of-year harmonic) when the net emits its coeffs, else
    # the static per-HRU scalar; likewise the seasonal recession override.
    kpet = (_seasonal(params, "Kpet", doy) if "Kpet_asin" in params
            else params["Kpet"].unsqueeze(-1))
    pet = kpet * raw_pet
    recession = ({p: _seasonal(params, p, doy) for p in _SEASONAL_RECESSION}
                 if "uzk_asin" in params else None)

    eff_p, snow_state = run_snow17(prcp, tavg, doy, is_leap, elev, params,
                                   state=state.snow)
    noah = None
    if et_mode == "noah":
        if canopy_params is None:
            raise ValueError("et_mode='noah' requires canopy_params")
        if tmin is None or tmax is None:
            tmin, tmax = canopy_inputs_fallback(tavg)
        noah = {"tavg": tavg, "tmin": tmin, "tmax": tmax, "doy": doy,
                "lat_rad": lat_rad, "elev": elev, "cp": canopy_params,
                "canopy": state.canopy}
    surf, base, _tet, sac_state = run_sacsma(pet, eff_p, params, state=state.sac,
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
    return flow, new_state


def routing_uh(params: dict[str, torch.Tensor], flowlen: torch.Tensor):
    """Build the per-HRU UH pair once per parameter set (reuse across chunks)."""
    return build_uh(params["Nres"], params["Kres"], params["Velo"], params["Diff"],
                    flowlen)
