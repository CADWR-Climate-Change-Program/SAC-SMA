"""Differentiable Snow-17 — torch mirror of the frozen ``sacsma.snow17``.

Same ``_fix`` variant as the reference (HARD rain/snow split at ``PXTEMP``;
the TTI mixture stays disabled).  Every reference branch is expressed as a
``torch.where``/mask blend whose FORWARD values match the reference exactly;
only the gradients differ (subgradients at the branch points).

Fidelity restorations vs the prior ``tmp/src_dpl`` port:

* the seasonal melt factor is **leap-aware** again (365/366 days and
  DAYN-80/81, exactly as ``snow17.py:59-68`` in the reference);
* the ATI reset uses the reference's exact ``Deficit == 0`` test (the branch
  blends set the deficit to exact zeros, so equality is well-defined).

State (per HRU): ``W_i`` (ice water-equivalent), ``ATI``, ``W_q`` (liquid),
``Deficit`` — reference cold start is all zeros.  ``dtt = dtp = 24`` h.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class Snow17State:
    w_i: torch.Tensor
    ati: torch.Tensor
    w_q: torch.Tensor
    deficit: torch.Tensor

    @classmethod
    def zeros(cls, n: int, device, dtype) -> Snow17State:
        z = torch.zeros(n, device=device, dtype=dtype)
        return cls(w_i=z.clone(), ati=z.clone(), w_q=z.clone(), deficit=z.clone())

    def detach(self) -> Snow17State:
        return Snow17State(self.w_i.detach(), self.ati.detach(),
                           self.w_q.detach(), self.deficit.detach())


def _melt_factor(doy_t: torch.Tensor, is_leap_t: torch.Tensor,
                 mfmax: torch.Tensor, mfmin: torch.Tensor) -> torch.Tensor:
    """Seasonal non-rain melt factor, leap-aware (reference snow17.py:59-68)."""
    days = torch.where(is_leap_t, 366.0, 365.0)
    n_mar21 = doy_t - torch.where(is_leap_t, 81.0, 80.0)
    sv = 0.5 * torch.sin(n_mar21 * (2.0 * math.pi) / days) + 0.5
    # dtt/6 = 4; Av = 1
    return 4.0 * (sv * (mfmax - mfmin) + mfmin)


def snow17_step(
    state: Snow17State,
    prcp_t: torch.Tensor,     # (N,)
    tavg_t: torch.Tensor,     # (N,)
    doy_t: torch.Tensor,      # scalar tensor
    is_leap_t: torch.Tensor,  # scalar bool tensor
    elev: torch.Tensor,       # (N,)
    p: dict[str, torch.Tensor],
) -> tuple[Snow17State, torch.Tensor]:
    """One daily step; returns (new_state, outflow E = rain+melt leaving the pack)."""
    SCF, PXTEMP, MFMAX, MFMIN = p["SCF"], p["PXTEMP"], p["MFMAX"], p["MFMIN"]
    UADJ, MBASE, TIPM, PLWHC = p["UADJ"], p["MBASE"], p["TIPM"], p["PLWHC"]
    NMF, DAYGM = p["NMF"], p["DAYGM"]

    w_i, ati, w_q, deficit = state.w_i, state.ati, state.w_q, state.deficit
    dtype = prcp_t.dtype

    # ---- form of precipitation (hard split at PXTEMP) ----
    is_snow = (tavg_t <= PXTEMP).to(dtype)
    rain = (1.0 - is_snow) * prcp_t

    # ---- accumulation ----
    pn = is_snow * prcp_t * SCF
    w_i = w_i + pn

    # ---- seasonal melt factor + heat deficit exchange ----
    mf = _melt_factor(doy_t, is_leap_t, MFMAX, MFMIN)
    t_snow_new = torch.minimum(tavg_t, torch.zeros_like(tavg_t))
    delta_hd_snow = -(t_snow_new * pn) / 160.0          # 80 / 0.5
    delta_hd_t = NMF * 4.0 * (mf / MFMAX) * (ati - t_snow_new)   # dtp/6 = 4

    # ---- ATI update ----
    tipm_dtt = 1.0 - (1.0 - TIPM) ** 4.0                # dtt/6 = 4
    ati = torch.where(pn > 36.0, t_snow_new, ati + tipm_dtt * (tavg_t - ati))
    ati = torch.minimum(ati, torch.zeros_like(ati))

    # ---- melt (three-way branch) ----
    t_rain = tavg_t.clamp_min(0.0)
    e_sat = 2.7489e8 * torch.exp(-4278.63 / (tavg_t + 242.792))
    elev_h = elev / 100.0
    p_atm = 33.86 * (29.9 - 0.335 * elev_h + 0.00022 * elev_h ** 2.4)
    term1 = 6.12e-10 * 24.0 * ((tavg_t + 273.0) ** 4 - 273.0 ** 4)
    term2 = 0.0125 * rain * t_rain
    term3 = 8.5 * UADJ * 4.0 * ((0.9 * e_sat - 6.11) + 0.00057 * p_atm * tavg_t)
    melt_ros = (term1 + term2 + term3).clamp_min(0.0)
    melt_nonrain = (mf * (tavg_t - MBASE) + 0.0125 * rain * t_rain).clamp_min(0.0)

    is_ros = (rain > 6.0).to(dtype)                      # 0.25 * dtp
    is_warm = (1.0 - is_ros) * (tavg_t > MBASE).to(dtype)
    melt = is_ros * melt_ros + is_warm * melt_nonrain

    # ---- ripeness / liquid-water bookkeeping ----
    deficit = (deficit + delta_hd_snow + delta_hd_t).clamp_min(0.0)

    # Branch A: melt < w_i (pack persists)
    melt_a = torch.minimum(melt, w_i)
    w_i_a = w_i - melt_a
    deficit_a = torch.minimum(deficit, 0.33 * w_i_a)
    qw = melt_a + rain
    w_qx = PLWHC * w_i_a

    is_ripe = (qw + w_q > deficit_a * (1.0 + PLWHC) + w_qx).to(dtype)
    is_hold = ((qw >= deficit_a) & (qw + w_q <= deficit_a * (1.0 + PLWHC) + w_qx)).to(dtype)
    is_below = (1.0 - is_ripe) * (1.0 - is_hold)

    e_a = is_ripe * (qw + w_q - w_qx - deficit_a * (1.0 + PLWHC)).clamp_min(0.0)
    w_i_a_new = (is_ripe + is_hold) * (w_i_a + deficit_a) + is_below * (w_i_a + qw)
    w_q_a_new = is_ripe * w_qx + is_hold * (w_q + qw - deficit_a) + is_below * w_q
    deficit_a_new = is_below * (deficit_a - qw).clamp_min(0.0)

    # Branch B: melt >= w_i (pack fully melts; deficit untouched)
    e_b = w_i + w_q + rain

    in_a = (melt < w_i).to(dtype)
    in_b = 1.0 - in_a
    w_i = in_a * w_i_a_new
    w_q = in_a * w_q_a_new
    deficit = in_a * deficit_a_new + in_b * deficit
    e = in_a * e_a + in_b * e_b

    # reference: exact-equality ATI reset (blends above set deficit to exact 0)
    ati = torch.where(deficit == 0.0, torch.zeros_like(ati), ati)

    # ---- constant ground melt ----
    has_pack = (w_i > DAYGM).to(dtype)
    gmwlos = (DAYGM / w_i.clamp_min(1e-12)) * w_q
    e = e + has_pack * (gmwlos + DAYGM) + (1.0 - has_pack) * (w_i + w_q)
    w_i = has_pack * (w_i - DAYGM)
    w_q = has_pack * (w_q - gmwlos)

    return Snow17State(w_i=w_i, ati=ati, w_q=w_q, deficit=deficit), e


def run_snow17(
    prcp: torch.Tensor,       # (N, T) mm/day
    tavg: torch.Tensor,       # (N, T) degC
    doy: torch.Tensor,        # (T,) float
    is_leap: torch.Tensor,    # (T,) bool
    elev: torch.Tensor,       # (N,) m
    params: dict[str, torch.Tensor],   # (N,) each
    state: Snow17State | None = None,
    return_swe: bool = False,
    seasonal: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Snow17State]:
    """Run Snow-17 over a window; returns (outflow (N, T), final state).

    Under ``torch.no_grad()`` the output is written into a preallocated
    buffer; with gradients enabled the per-step outputs are stacked (autograd
    forbids in-place writes into leaves of the graph).

    With ``return_swe`` a third item is returned: the per-day snow water
    equivalent ``(N, T)`` (ice + retained liquid, end of step) — the driver
    feeds it to the Priestley-Taylor snow-cover albedo.

    ``seasonal`` optionally maps a melt-parameter name (``MFMAX``/``MFMIN``/
    ``MBASE``) to a per-day ``(N, T)`` field reconstructed by the caller from the
    net's day-of-year harmonic coeffs; each step overrides that scalar param with
    the day's slice.  ``None`` (the default) leaves every param static — the
    forward is byte-identical to the pre-seasonal path.
    """
    n, t_len = prcp.shape
    if state is None:
        state = Snow17State.zeros(n, prcp.device, prcp.dtype)
    grad = torch.is_grad_enabled() and (
        prcp.requires_grad or any(v.requires_grad for v in params.values())
        or (seasonal is not None and any(v.requires_grad for v in seasonal.values()))
    )
    out = None if grad else torch.empty_like(prcp)
    swe_out = torch.empty_like(prcp) if (return_swe and not grad) else None
    steps: list[torch.Tensor] = []
    swe_steps: list[torch.Tensor] = []
    for t in range(t_len):
        p_t = (params if seasonal is None
               else {**params, **{k: v[:, t] for k, v in seasonal.items()}})
        state, e_t = snow17_step(state, prcp[:, t], tavg[:, t], doy[t], is_leap[t],
                                 elev, p_t)
        if grad:
            steps.append(e_t)
            if return_swe:
                swe_steps.append(state.w_i + state.w_q)
        else:
            out[:, t] = e_t
            if return_swe:
                swe_out[:, t] = state.w_i + state.w_q
    if grad:
        out = torch.stack(steps, dim=-1)
        if return_swe:
            swe_out = torch.stack(swe_steps, dim=-1)
    if return_swe:
        return out, state, swe_out
    return out, state
