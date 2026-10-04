"""Vectorized differentiable pipeline: PET -> Snow-17 -> SAC-SMA -> routing.

The torch model of a learned run, batched over the units of a domain and run window by window
with the state carried between windows -- the Snow-17 and SAC-SMA states and the 106-day routing
inflow history -- so a long record streams under ``torch.no_grad()`` (spin-up, selection) or
trains window by window (truncated backpropagation).  The physics options are a
:class:`sacsma.engine.Physics`, the description the CPU engine runs a trained field with.

The routing unit hydrographs depend on the parameters only; :func:`routing_uh` builds them once
per parameter set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .physics.et_noah import (
    dewpoint_depression_field,
    potential_et_priestley_taylor,
    snow_cover_albedo,
)
from .physics.pet import hamon_raw_pet
from .physics.routing import N_TAPS, build_uh, route
from .physics.sma import SacState, run_sacsma
from .physics.snow17 import Snow17State, run_snow17

if TYPE_CHECKING:
    from ..engine import Physics
    from .data import Window


@dataclass
class PipelineState:
    # snow / sac: per physics row; hist_*: per routed row -- the same N except under
    # cell dedup (distinct cells vs (entity, cell) rows)
    snow: Snow17State
    sac: SacState
    hist_surf: torch.Tensor   # (N, N_TAPS-1) unrouted direct-inflow history
    hist_base: torch.Tensor   # (N, N_TAPS-1) unrouted baseflow history
    # (4, N, N_TAPS-1) unrouted history of the runoff parts (quick, interflow,
    # supplemental, primary) -- carried only by run_window(return_parts=True)
    hist_parts: torch.Tensor | None = None

    def detach(self) -> PipelineState:
        return PipelineState(self.snow.detach(), self.sac.detach(),
                             self.hist_surf.detach(), self.hist_base.detach(),
                             None if self.hist_parts is None else self.hist_parts.detach())


def initial_state(n: int, device, dtype, *, n_rows: int | None = None) -> PipelineState:
    """The cold start: Snow-17 zeros, SAC-SMA [0, 0, 100, 100, 100, 0], no inflow history.

    ``n`` sizes the per-cell physics states; ``n_rows`` (default ``n``) the routing
    histories -- they differ under cell dedup (``DomainTensors.n_phys`` distinct
    cells vs ``n_hru`` routed rows)."""
    hist = torch.zeros(n if n_rows is None else n_rows, N_TAPS - 1,
                       device=device, dtype=dtype)
    return PipelineState(snow=Snow17State.zeros(n, device, dtype),
                         sac=SacState.reference_init(n, device, dtype),
                         hist_surf=hist, hist_base=hist.clone())


def run_window(
    w: Window,                # the forcing of the window on the physics rows
    lat_rad: torch.Tensor,    # (N,)
    elev: torch.Tensor,       # (N,)
    params: dict[str, torch.Tensor],       # (N,) per ga_optimum name (+ soil_chi, PXTEMP_tau)
    uh: tuple[torch.Tensor, torch.Tensor],  # (uh_direct, uh_base) from build_uh
    state: PipelineState,
    physics: Physics,
    *,
    veg_frac: torch.Tensor | None = None,  # (N,) observed green fraction (Noah-lite)
    return_components: bool = False,       # also return the routed (fast, slow) pair
    return_parts: bool = False,            # also return the 4 routed runoff parts
    row_cell: torch.Tensor | None = None,  # (R,) cell dedup: the physics row of each
                                           # routed row (DomainTensors.row_cell)
) -> tuple[torch.Tensor, PipelineState]:   # (+ components, parts, when requested)
    """One window through the full pipeline; returns (routed flow (N, T), state), with the
    routed runoff components ``(fast, slow)`` appended when ``return_components`` and the
    routed runoff parts ``(quick, interflow, supplemental, primary)`` when ``return_parts``.

    The components are the two routed terms whose sum IS ``flow``: ``fast`` = the SAC-SMA
    direct inflow (impervious + direct + surface + interflow) through the hillslope x channel
    UH, ``slow`` = primary + supplemental baseflow through the channel UH alone.  Both are net
    of SAC-SMA's riparian et4 channel-ET deduction and dry-channel clamp, which act on the
    aggregate direct inflow / baseflow before routing.

    The parts split those two further (:func:`sacsma.dpl.physics.sma.sacsma_step`
    ``return_parts``): ``quick`` = impervious + ADIMP direct + surface runoff and
    ``interflow``, each routed through the hillslope x channel UH (``quick + interflow =
    fast``), and ``supplemental`` / ``primary`` baseflow through the channel UH
    (``supplemental + primary = slow``), equal up to the rounding of the linear routing.
    Their routing history travels in ``state.hist_parts``; a state without it is accepted
    only with an empty routing history (a cold start), so a parts run streams its spin-up
    with ``return_parts`` too.

    ``row_cell`` (cell dedup, ``DplConfig.dedup_cells``): the forcing, statics, ``params``
    and the snow/SAC states are per DISTINCT cell, while ``uh`` and the routing histories
    are per routed row; each cell's surface and base runoff is gathered to its rows before
    the routing, so the flow comes back per routed row.

    The Priestley-Taylor PET needs the window's ``tmin``/``tmax`` and Noah-lite the observed
    ``veg_frac`` and the window's ``lai``; either raises without them.
    """
    pt = physics.pet == "priestley_taylor"
    # Snow-17 first: its SWE drives the snow-cover albedo of the Priestley-Taylor PET
    albedo_swe = pt and physics.pt_snow_albedo > 0.0
    snow_out = run_snow17(w.pr, w.ta, w.doy, w.leap, elev, params,
                          state=state.snow, return_swe=albedo_swe)
    eff_p, snow_state = snow_out[0], snow_out[1]

    if pt:
        if w.tmin is None or w.tmax is None:
            raise ValueError("the Priestley-Taylor PET needs per-cell tmin/tmax")
        pt_kwargs = {}
        if albedo_swe:   # blend the albedo toward snow where a pack is present
            pt_kwargs["albedo"] = snow_cover_albedo(snow_out[2], physics.pt_snow_albedo)
        if physics.pt_dewpoint_depression > 0.0:   # lower Tdew below Tmin in arid air
            pt_kwargs["dewpoint_depression"] = dewpoint_depression_field(
                w.tmin, w.tmax, physics.pt_dewpoint_depression)
        raw_pet = potential_et_priestley_taylor(w.ta, w.tmin, w.tmax, w.doy, lat_rad, elev,
                                                **pt_kwargs)
    else:
        raw_pet = hamon_raw_pet(w.ta, w.doy, lat_rad)
    pet = params["Kpet"].unsqueeze(-1) * raw_pet

    if physics.et == "noah_lite" and (veg_frac is None or w.lai is None):
        raise ValueError("the Noah-lite ET needs the observed veg_frac and lai")
    sac_out = run_sacsma(pet, eff_p, params, state.sac, physics, veg_frac=veg_frac,
                         lai=w.lai, return_parts=return_parts)
    surf, base, _tet, sac_state = sac_out[:4]
    if row_cell is not None:
        # cell dedup: every (entity, cell) row routes its cell's runoff through its
        # own flow length (the backward sums the rows' gradients onto the cell)
        surf = surf.index_select(0, row_cell)
        base = base.index_select(0, row_cell)
        if return_parts:
            # the six runoff parts go to the rows the same way (per cell they sum to
            # surf / base, so per row they still do; hist_parts is per routed row)
            sac_out = (*sac_out[:4], tuple(p.index_select(0, row_cell) for p in sac_out[4]))
    uh_direct, uh_base = uh
    if return_components or return_parts:
        # the same two routed terms and the same sum as the default expression
        fast = route(surf, uh_direct, state.hist_surf)
        slow = route(base, uh_base, state.hist_base)
        flow = fast + slow
    else:
        flow = route(surf, uh_direct, state.hist_surf) + route(base, uh_base, state.hist_base)

    hist_parts = None
    if return_parts:
        roimp, sdro, ssur, sif, bf_sup, bf_pri = sac_out[4]
        # (4, N, T) unrouted quick / interflow / supplemental / primary; quick is
        # summed in the order that reproduces the exact split (sma._apportion_parts)
        series = torch.stack([roimp + sdro + ssur, sif, bf_sup, bf_pri])
        hp = state.hist_parts
        if hp is None:
            if bool(state.hist_surf.any()) or bool(state.hist_base.any()):
                raise ValueError("return_parts needs the parts' routing history "
                                 "(state.hist_parts): stream the spinup with "
                                 "return_parts too, or start from an empty history")
            hp = torch.zeros((4,) + tuple(state.hist_surf.shape),
                             device=surf.device, dtype=surf.dtype)
        uhs = (uh_direct, uh_direct, uh_base, uh_base)
        parts = tuple(route(series[k], uhs[k], hp[k]) for k in range(4))
        hist_parts = torch.cat([hp, series], dim=-1)[..., -(N_TAPS - 1):]

    # carry the last N_TAPS-1 inflow days for the next window's convolution
    hist_surf = torch.cat([state.hist_surf, surf], dim=-1)[:, -(N_TAPS - 1):]
    hist_base = torch.cat([state.hist_base, base], dim=-1)[:, -(N_TAPS - 1):]
    new_state = PipelineState(snow=snow_state, sac=sac_state, hist_surf=hist_surf,
                              hist_base=hist_base, hist_parts=hist_parts)
    out = [flow, new_state]
    if return_components:
        out.append((fast, slow))
    if return_parts:
        out.append(parts)
    return tuple(out)


def routing_uh(params: dict[str, torch.Tensor], flowlen: torch.Tensor,
               row_cell: torch.Tensor | None = None):
    """Build the per-HRU UH pair once per parameter set (reuse across chunks).

    ``row_cell`` (cell dedup): ``params`` are per distinct cell and ``flowlen`` per
    routed row; the hillslope UH is built per cell and gathered, the channel UH
    per row from the cell's Velo/Diff and the row's flow length."""
    return build_uh(params["Nres"], params["Kres"], params["Velo"], params["Diff"],
                    flowlen, row_cell=row_cell)
