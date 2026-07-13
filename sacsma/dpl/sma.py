"""Differentiable SAC-SMA — torch mirror of the frozen ``sacsma.sma``.

Every reference branch/clamp (``sma.py::_sacsma_core``) is expressed as a
mask/`torch.where` blend whose FORWARD values match the reference exactly,
including the ``thres_zero = 1e-5`` and ``0.0001`` storage snaps, the
``(pinc + uzfwc) <= 0.01`` percolation short-circuit, the ``ratio**2`` ADIMP
runoff, and the reference et4/et5 channel-inflow adjustment (``et4`` from the
UNWEIGHTED ``et1+et2+et3`` — the prior tmp/src_dpl port weighted it too early).

The ONE structural deviation from the reference is the substep count: the
reference ``ninc = floor(1 + 0.2*(uzfwc + twx))`` is data-dependent (graph
shape would change per step/HRU); here it is a fixed ``n_inc`` (fidelity vs
``n_inc`` is quantified by ``sacsma dpl benchmark`` before any training).

``perc_mode`` selects the percolation numerics:

* ``"reference"`` — linear demand ``percm*uzfwc/uzfwm*(1+zperc*defr**rexp)``
  hard-capped at ``uzfwc`` then at the LZ deficit (exact frozen forward; the
  unclipped local Jacobian is huge — zperc reaches 500 — so this mode is for
  fidelity checks, not training);
* ``"implicit"`` — ``uzfwc*(1-exp(-k*dinc))`` implicit-Euler saturator;
* ``"tanh"`` — ``uzfwc*tanh(k)`` saturator.  Both bound the gradient in [0,1]
  and add a ``defr + 1e-6`` guard (NaN d/d(rexp) at defr=0) and a
  ``fracp_floor`` on the LZ fill-fraction denominator.

Parameter dict keys are the ga_optimum column names (lowercase SMA names);
bounds (config.BOUNDS) are a precondition — divisions by uztwm/uzfwm/lztwm/…
are unguarded exactly like the reference because the bounds keep them >= 1.
State order matches the reference vector: [uztwc, uzfwc, lztwc, lzfsc, lzfpc,
adimc]; reference cold start [0, 0, 100, 100, 100, 0].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_THRES = 1e-5    # reference thres_zero
_BF_EPS = 1e-4   # reference baseflow depletion snap (0.0001 — a DIFFERENT constant)


@dataclass
class SacState:
    uztwc: torch.Tensor
    uzfwc: torch.Tensor
    lztwc: torch.Tensor
    lzfsc: torch.Tensor
    lzfpc: torch.Tensor
    adimc: torch.Tensor

    @classmethod
    def reference_init(cls, n: int, device, dtype) -> SacState:
        """The frozen cold start [0, 0, 100, 100, 100, 0]."""
        z = torch.zeros(n, device=device, dtype=dtype)
        h = torch.full((n,), 100.0, device=device, dtype=dtype)
        return cls(uztwc=z.clone(), uzfwc=z.clone(), lztwc=h.clone(),
                   lzfsc=h.clone(), lzfpc=h.clone(), adimc=z.clone())

    @classmethod
    def capacity_init(cls, p: dict[str, torch.Tensor]) -> SacState:
        """Storages at capacity (the tmp/src_dpl convention; NOT the reference)."""
        return cls(uztwc=p["uztwm"].clone(), uzfwc=torch.zeros_like(p["uztwm"]),
                   lztwc=p["lztwm"].clone(), lzfsc=p["lzfsm"].clone(),
                   lzfpc=p["lzfpm"].clone(), adimc=(p["uztwm"] + p["lztwm"]).clone())

    def detach(self) -> SacState:
        return SacState(*(getattr(self, f).detach() for f in
                          ("uztwc", "uzfwc", "lztwc", "lzfsc", "lzfpc", "adimc")))


def _snap(x: torch.Tensor, thres: float = _THRES) -> torch.Tensor:
    """Reference storage snap: values below ``thres`` become exactly 0."""
    return torch.where(x < thres, torch.zeros_like(x), x)


def sacsma_step(
    state: SacState,
    pr_t: torch.Tensor,    # (N,) effective precip (Snow-17 outflow), mm/day
    pet_t: torch.Tensor,   # (N,) PET, mm/day
    p: dict[str, torch.Tensor],
    *,
    n_inc: int = 5,
    perc_mode: str = "reference",
    fracp_floor: float = 0.0,
    ninc_mode: str = "fixed",
    et_mode: str = "sac",
    eused_ext: torch.Tensor | None = None,
) -> tuple[SacState, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One daily step; returns (new_state, surf, base, tet) in mm/day.

    ``et_mode="sac"`` (default) runs the exact-parity E1-E5 cascade.
    ``et_mode="external"`` skips it: the caller (Noah ET) has already applied
    the soil withdrawals to ``state`` and passes ``pr_t`` as canopy-throughfall
    effective precip; ``eused_ext`` (the soil ET actually withdrawn) feeds the
    unchanged riparian ``et4`` channel adjustment.
    """
    uztwm, uzfwm, lztwm = p["uztwm"], p["uzfwm"], p["lztwm"]
    lzfpm, lzfsm = p["lzfpm"], p["lzfsm"]
    uzk, lzpk, lzsk = p["uzk"], p["lzpk"], p["lzsk"]
    zperc, rexp, pfree = p["zperc"], p["rexp"], p["pfree"]
    pctim, adimp, riva = p["pctim"], p["adimp"], p["riva"]
    side, rserv = p["side"], p["rserv"]

    uztwc, uzfwc, lztwc = state.uztwc, state.uzfwc, state.lztwc
    lzfsc, lzfpc, adimc = state.lzfsc, state.lzfpc, state.adimc
    dtype = pr_t.dtype
    zero = torch.zeros_like(uztwc)

    parea = 1.0 - adimp - pctim
    edmnd = pet_t

    if et_mode == "external":
        # ET applied upstream; no withdrawal here.  eused feeds et4 only.
        et1 = et2 = et3 = et5 = zero
        eused = eused_ext if eused_ext is not None else zero
    else:
        # ---- ET(1): upper-zone tension (min == reference subtract-then-correct) ----
        et1 = torch.minimum(edmnd * uztwc / uztwm, uztwc)
        uztwc = uztwc - et1
        red = edmnd - et1

        # ---- ET(2) / UZ rebalance (branch on tension exhaustion, exact <= 0) ----
        exhausted = (uztwc <= 0.0).to(dtype)
        et2 = exhausted * torch.minimum(red, uzfwc)
        uzfwc_ex = uzfwc - et2
        red = red - et2
        # else branch: rebalance if free fuller than tension, then thres snaps
        do_rb = (1.0 - exhausted) * (uztwc / uztwm < uzfwc / uzfwm).to(dtype)
        uzrat = (uztwc + uzfwc) / (uztwm + uzfwm)
        uztwc = do_rb * (uztwm * uzrat) + (1.0 - do_rb) * uztwc
        uzfwc_rb = do_rb * (uzfwm * uzrat) + (1.0 - do_rb) * uzfwc
        uzfwc = exhausted * uzfwc_ex + (1.0 - exhausted) * uzfwc_rb
        # reference snaps apply only on the non-exhausted path (no-ops on the other)
        not_ex = 1.0 - exhausted
        uztwc = torch.where((uztwc < _THRES) & (not_ex > 0), zero, uztwc)
        uzfwc = torch.where((uzfwc < _THRES) & (not_ex > 0), zero, uzfwc)

        # ---- ET(3): lower-zone tension ----
        et3 = torch.minimum(red * lztwc / (uztwm + lztwm), lztwc)
        lztwc = lztwc - et3

        # ---- resupply lower free -> lower tension ----
        saved = rserv * (lzfpm + lzfsm)
        ratlzt = lztwc / lztwm
        ratlz = (lztwc + lzfpc + lzfsc - saved) / (lztwm + lzfpm + lzfsm - saved)
        resup = (ratlzt < ratlz).to(dtype)
        dele = resup * (ratlz - ratlzt) * lztwm
        lztwc = lztwc + dele
        lzfsc_raw = lzfsc - dele
        lzfpc = lzfpc + torch.minimum(lzfsc_raw, zero)   # neg overdraw spills to primary
        lzfsc = lzfsc_raw.clamp_min(0.0)
        lztwc = _snap(lztwc)

        # ---- ET(5): ADIMP area (reference allows a negative demand — no lower clamp) ----
        et5_raw = et1 + (red + et2) * (adimc - et1 - uztwc) / (uztwm + lztwm)
        et5 = torch.minimum(et5_raw, adimc)
        adimc = adimc - et5
        et5 = et5 * adimp
        eused = et1 + et2 + et3

    # ---- throughfall split + impervious runoff ----
    twx_raw = pr_t + uztwc - uztwm
    has_excess = (twx_raw >= 0.0).to(dtype)
    uztwc = has_excess * uztwm + (1.0 - has_excess) * (uztwc + pr_t)
    twx = has_excess * twx_raw
    adimc = adimc + pr_t - twx
    roimp = pr_t * pctim

    # ---- substep loop ----
    # ninc_mode="fixed": n_inc substeps for every lane (the trainable numerics).
    # ninc_mode="dynamic": the reference's data-dependent
    # ninc = floor(1 + 0.2*(uzfwc + twx)) per lane, run to max(ninc) with
    # finished lanes masked through — EXACT frozen numerics (fidelity only:
    # the .item() sync and unbounded loop length make it untrainable).
    sbf = torch.zeros_like(uztwc)
    ssur = torch.zeros_like(uztwc)
    sif = torch.zeros_like(uztwc)
    sdro = torch.zeros_like(uztwc)

    if ninc_mode == "dynamic":
        ninc = torch.floor(1.0 + 0.2 * (uzfwc + twx))
        n_steps = int(ninc.max().item())
        dinc = 1.0 / ninc
        pinc = twx / ninc
    else:
        ninc = None
        n_steps = n_inc
        dinc = 1.0 / n_inc
        pinc = twx / n_inc
    duz = 1.0 - (1.0 - uzk) ** dinc
    dlzp = 1.0 - (1.0 - lzpk) ** dinc
    dlzs = 1.0 - (1.0 - lzsk) ** dinc
    percm = lzfpm * dlzp + lzfsm * dlzs

    for s in range(n_steps):
        # ADIMP direct runoff (hardcoded ratio**2, as the reference)
        ratio = ((adimc - uztwc) / lztwm).clamp_min(0.0)
        addro = pinc * ratio * ratio

        # baseflow depletion with the reference 0.0001 snap
        bf_p = lzfpc * dlzp
        lzfpc_raw = lzfpc - bf_p
        snap_p = (lzfpc_raw <= _BF_EPS).to(dtype)
        bf_p = bf_p + snap_p * lzfpc_raw
        lzfpc_s = (1.0 - snap_p) * lzfpc_raw

        bf_s = lzfsc * dlzs
        lzfsc_raw = lzfsc - bf_s
        snap_s = (lzfsc_raw <= _BF_EPS).to(dtype)
        bf_s = bf_s + snap_s * lzfsc_raw
        lzfsc_s = (1.0 - snap_s) * lzfsc_raw

        # percolation short-circuit: (pinc + uzfwc) <= 0.01 skips the whole block
        skip = ((pinc + uzfwc) <= 0.01).to(dtype)
        act = 1.0 - skip

        lz_deficit = (lztwm + lzfpm + lzfsm) - (lztwc + lzfpc_s + lzfsc_s)
        defr = (1.0 - (lztwc + lzfpc_s + lzfsc_s) / (lztwm + lzfpm + lzfsm)).clamp_min(0.0)
        if perc_mode == "reference":
            amp = 1.0 + zperc * defr ** rexp
            perc = torch.minimum(percm * uzfwc / uzfwm * amp, uzfwc)
        else:
            k_perc = (percm / uzfwm) * (1.0 + zperc * (defr + 1e-6) ** rexp)
            if perc_mode == "implicit":
                perc = uzfwc * (1.0 - torch.exp(-(k_perc * dinc).clamp_max(50.0)))
            else:  # "tanh"
                perc = uzfwc * torch.tanh(k_perc)
        # the reference "check" correction == cap at the LZ deficit
        perc = act * torch.minimum(perc, lz_deficit.clamp_min(0.0))
        uzfwc_a = uzfwc - perc

        # interflow
        dele_if = uzfwc_a * duz
        uzfwc_a = uzfwc_a - dele_if

        # distribute percolation: LZ tension first, overflow + pfree to free stores
        perct = perc * (1.0 - pfree)
        perct_in = torch.minimum(perct, lztwm - lztwc)
        lztwc_a = lztwc + perct_in
        percf = (perct - perct_in) + perc * pfree

        hpl = lzfpm / (lzfpm + lzfsm)
        ratlp = lzfpc_s / lzfpm
        ratls = lzfsc_s / lzfsm
        denom = ((1.0 - ratlp) + (1.0 - ratls)).clamp_min(max(fracp_floor, 1e-12))
        fracp = (hpl * 2.0 * (1.0 - ratlp) / denom).clamp_max(1.0)
        percs = percf * (1.0 - fracp)
        percs_in = torch.minimum(percs, lzfsm - lzfsc_s)
        lzfsc_a = lzfsc_s + percs_in
        into_p = percf - percs_in
        percp_in = torch.minimum(into_p, lzfpm - lzfpc_s)
        lzfpc_a = lzfpc_s + percp_in
        lztwc_a = lztwc_a + (into_p - percp_in)      # primary overflow -> tension

        # distribute pinc: fill UZ free, spill to surface (algebraic adsur rewrite)
        space_uz = uzfwm - uzfwc_a
        into_uz = torch.minimum(pinc, space_uz)
        sur = pinc - into_uz
        uzfwc_a = uzfwc_a + into_uz
        adsur = act * sur * (1.0 - ratio * ratio)

        # blend skip/active paths
        uzfwc_n = skip * (uzfwc + pinc) + act * uzfwc_a
        lztwc_n = skip * lztwc + act * lztwc_a
        lzfsc_n = skip * lzfsc_s + act * lzfsc_a
        lzfpc_n = skip * lzfpc_s + act * lzfpc_a

        # ADIMP water balance + overflow
        adimc_n = adimc + pinc - addro - adsur
        over = (adimc_n - (uztwm + lztwm)).clamp_min(0.0)
        addro = addro + over
        adimc_n = _snap(adimc_n - over)

        if ninc is None:
            live = None
            uzfwc, lztwc, lzfsc, lzfpc, adimc = uzfwc_n, lztwc_n, lzfsc_n, lzfpc_n, adimc_n
        else:
            # lanes whose own ninc is exhausted pass through unchanged
            live = (ninc > s).to(dtype)
            hold = 1.0 - live
            uzfwc = live * uzfwc_n + hold * uzfwc
            lztwc = live * lztwc_n + hold * lztwc
            lzfsc = live * lzfsc_n + hold * lzfsc
            lzfpc = live * lzfpc_n + hold * lzfpc
            adimc = live * adimc_n + hold * adimc
        gate = 1.0 if live is None else live
        sbf = sbf + gate * (bf_p + bf_s)
        sif = sif + gate * act * dele_if
        ssur = ssur + gate * act * (sur * parea + adsur * adimp)
        sdro = sdro + gate * (addro * adimp)

    # ---- aggregate channel inflow (reference order: et4 from UNWEIGHTED eused) ----
    # eused is set above per et_mode (et1+et2+et3 for "sac"; eused_ext for "external")
    sif = sif * parea
    bfcc = sbf * parea / (1.0 + side)
    base = bfcc
    surf = roimp + sdro + ssur + sif

    et4 = (edmnd - eused) * riva
    adimc = torch.maximum(adimc, uztwc)

    ch_inflow = surf + base - et4
    dry = (ch_inflow <= 0.0).to(dtype)
    et4 = dry * (surf + base) + (1.0 - dry) * et4
    half = 0.5 * et4
    surf_h = surf - half
    base_h = base - half
    # ch_inflow > 0 => at most one of the halves goes negative; the other absorbs it
    surf_w = (surf_h + torch.minimum(base_h, zero)).clamp_min(0.0)
    base_w = (base_h + torch.minimum(surf_h, zero)).clamp_min(0.0)
    surf = (1.0 - dry) * surf_w
    base = (1.0 - dry) * base_w

    tet = eused * parea + et4 + et5

    new_state = SacState(uztwc=uztwc, uzfwc=uzfwc, lztwc=lztwc,
                         lzfsc=lzfsc, lzfpc=lzfpc, adimc=adimc)
    return new_state, surf, base, tet


def run_sacsma(
    pet: torch.Tensor,       # (N, T) mm/day
    pr_eff: torch.Tensor,    # (N, T) mm/day (Snow-17 outflow)
    params: dict[str, torch.Tensor],   # (N,) each, ga_optimum names
    state: SacState | None = None,
    *,
    n_inc: int = 5,
    perc_mode: str = "reference",
    fracp_floor: float = 0.0,
    ninc_mode: str = "fixed",
    et_mode: str = "sac",
    noah: dict | None = None,
    recession: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SacState]:
    """Run SAC-SMA over a window; returns (surf, base, tet, final_state).

    ``et_mode="noah"`` runs the Noah canopy-resistance ET each day (interception
    reduces the precip that enters the SAC balance; the three ET components are
    withdrawn upstream of the water-balance step).  ``noah`` then carries the
    per-day drivers and the canopy state::

        {"tavg","tmin","tmax": (N,T); "doy": (T,); "lat_rad","elev": (N,);
         "cp": {canopy params}; "canopy": NoahCanopyState}

    The updated canopy ``wc`` is written back into ``noah["canopy"]``.
    """
    n, t_len = pet.shape
    if state is None:
        state = SacState.reference_init(n, pet.device, pet.dtype)
    if et_mode == "noah":
        from .et_noah import NoahCanopyState, noah_et_step, noah_lite_et_step
        if noah is None:
            raise ValueError("et_mode='noah' requires the noah driver dict")
        lite = bool(noah.get("lite"))
        cstate = noah.get("canopy") or NoahCanopyState.zeros(n, pet.device, pet.dtype)
        wc = cstate.wc
    grad = torch.is_grad_enabled() and (
        pet.requires_grad or pr_eff.requires_grad
        or any(v.requires_grad for v in params.values())
        or (recession is not None and any(v.requires_grad for v in recession.values()))
    )
    if grad:
        surf_s: list[torch.Tensor] = []
        base_s: list[torch.Tensor] = []
        tet_s: list[torch.Tensor] = []
    else:
        surf = torch.empty_like(pet)
        base = torch.empty_like(pet)
        tet = torch.empty_like(pet)
    for t in range(t_len):
        # per-day params: override the seasonal recession rates when supplied
        # (the day-of-year harmonic experiment); otherwise the shared dict.
        p_t = params if recession is None else {
            **params, "uzk": recession["uzk"][:, t],
            "lzpk": recession["lzpk"][:, t], "lzsk": recession["lzsk"][:, t]}
        if et_mode == "noah":
            st = {"uztwc": state.uztwc, "uzfwc": state.uzfwc, "lztwc": state.lztwc,
                  "lzfsc": state.lzfsc, "lzfpc": state.lzfpc, "adimc": state.adimc,
                  "wc": wc}
            doy_t = noah["doy"][t] if noah["doy"].dim() == 1 else noah["doy"][:, t]
            # index any (N,T) dynamic canopy param (e.g. soil_chi) to day t;
            # static (N,) canopy params pass through unchanged.
            cp_t = {k: (v[:, t] if v.dim() == 2 else v)
                    for k, v in noah["cp"].items()}
            if lite:
                eff_p, ns, et_soil = noah_lite_et_step(
                    st, pr_eff[:, t], pet[:, t], params, cp_t,
                    noah["veg_frac"], noah["lai"][:, t])
            else:
                eff_p, ns, et_soil = noah_et_step(
                    st, pr_eff[:, t], pet[:, t],
                    noah["tavg"][:, t], noah["tmin"][:, t], noah["tmax"][:, t],
                    doy_t, noah["lat_rad"], noah["elev"], params, cp_t,
                    noah["veg_frac"], noah["lai"][:, t])
            state = SacState(uztwc=ns["uztwc"], uzfwc=ns["uzfwc"], lztwc=ns["lztwc"],
                             lzfsc=ns["lzfsc"], lzfpc=ns["lzfpc"], adimc=ns["adimc"])
            wc = ns["wc"]
            # pass the Noah ET as eused so the step's existing eused*parea term
            # reports it (pervious-area weighted) — no double count.  te is then
            # tet_noah*parea + et4 (riparian channel).
            state, sf, bs, te = sacsma_step(
                state, eff_p, pet[:, t], p_t, n_inc=n_inc, perc_mode=perc_mode,
                fracp_floor=fracp_floor, ninc_mode=ninc_mode,
                et_mode="external", eused_ext=et_soil)
        else:
            state, sf, bs, te = sacsma_step(
                state, pr_eff[:, t], pet[:, t], p_t, n_inc=n_inc,
                perc_mode=perc_mode, fracp_floor=fracp_floor, ninc_mode=ninc_mode)
        if grad:
            surf_s.append(sf)
            base_s.append(bs)
            tet_s.append(te)
        else:
            surf[:, t] = sf
            base[:, t] = bs
            tet[:, t] = te
    if grad:
        surf = torch.stack(surf_s, dim=-1)
        base = torch.stack(base_s, dim=-1)
        tet = torch.stack(tet_s, dim=-1)
    if et_mode == "noah":
        noah["canopy"] = NoahCanopyState(wc=wc)
    return surf, base, tet, state
