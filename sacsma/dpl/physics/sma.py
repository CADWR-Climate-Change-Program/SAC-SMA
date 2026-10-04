"""Differentiable SAC-SMA: the torch step the learned parameter fields are trained with.

Every branch and clamp of the reference (``sacsma.sma._sacsma_core``) is written as a mask or
``torch.where`` blend with the reference's forward values: the ``thres_zero = 1e-5`` and
``0.0001`` storage snaps, the ``(pinc + uzfwc) <= 0.01`` percolation short-circuit, the
``ratio**2`` ADIMP runoff and the et4 channel-inflow adjustment (``et4`` from the unweighted
``et1 + et2 + et3``).  It departs from the reference in a fixed number of substeps a day
(``n_inc``; the reference takes ``floor(1 + 0.2 * (uzfwc + twx))``), a floor on the
denominator of the lower-zone fill fraction (``fracp_floor``) and the percolation cap at the
lower-zone deficit (``sacsma.sma_learned``, its Numba copy, lists the differences).

Parameter keys are the ga_optimum names (lowercase SMA names); the bounds (config.BOUNDS) keep
every capacity >= 1, so the divisions are unguarded as in the reference.  State order is the
reference vector [uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc], cold start [0, 0, 100, 100, 100, 0].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .et_noah import noah_lite_et

_THRES = 1e-5    # reference thres_zero
_BF_EPS = 1e-4   # reference baseflow depletion snap (0.0001 — a DIFFERENT constant)

#: the runoff parts ``sacsma_step(return_parts=True)`` returns, in order: the four
#: surface-side parts (impervious, ADIMP direct, surface, interflow) that sum to
#: ``surf`` and the two baseflow parts (supplemental, primary) that sum to ``base``.
PART_NAMES = ("roimp", "sdro", "ssur", "sif", "supplemental", "primary")


def _split2(total: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Split ``total`` (>= 0) into ``(x, y)`` in proportion to ``a, b >= 0`` so that
    ``x + y == total`` EXACTLY in floating point and ``x, y >= 0``: the larger share is
    ``total * w`` with ``w = max(a, b) / (a + b)`` in [0.5, 1], the smaller is the
    remainder ``total - total * w``, which Sterbenz's lemma makes exact.  A zero
    denominator (``a = b = 0``) gives equal halves.  NaN propagates."""
    den = a + b
    pos = den > 0
    a_big = a >= b
    big = torch.where(a_big, a, b)
    w = torch.where(pos, big / torch.where(pos, den, torch.ones_like(den)),
                    torch.full_like(den, 0.5))
    hi = total * w
    lo = total - hi
    return torch.where(a_big, hi, lo), torch.where(a_big, lo, hi)


def _apportion_parts(surf: torch.Tensor, base: torch.Tensor,
                     roimp: torch.Tensor, sdro: torch.Tensor, ssur: torch.Tensor,
                     sif: torch.Tensor, bf_sup: torch.Tensor, bf_pri: torch.Tensor):
    """The FINAL (post-et4, post-dry-clamp) ``surf`` / ``base`` apportioned over their
    parts in proportion to the parts' PRE-deduction values -> the six parts of
    :data:`PART_NAMES`.

    The weights are the pre-deduction values floored at 0 (the reference allows a
    negative ADIMP surface term, ``ratio > 1``, and a storage overdraw can make a
    baseflow part negative; such a part gets no share and the others carry the net
    amount); when every weight of a side is 0 the side splits equally.  The split is
    a tree of exact binary splits (:func:`_split2`) — ``surf -> (quick, sif)``,
    ``quick -> (roimp + sdro, ssur)``, ``roimp + sdro -> (roimp, sdro)``; ``base ->
    (supplemental, primary)`` — so, summed left to right, ``roimp + sdro + ssur + sif
    == surf`` and ``supplemental + primary == base`` bitwise, and no part is negative."""
    r, d, s, i = (v.clamp_min(0.0) for v in (roimp, sdro, ssur, sif))
    none = (((r + d) + s) + i) <= 0.0
    one = torch.ones_like(r)
    r, d, s, i = (torch.where(none, one, v) for v in (r, d, s, i))
    quick, sif_p = _split2(surf, (r + d) + s, i)
    imp, ssur_p = _split2(quick, r + d, s)
    roimp_p, sdro_p = _split2(imp, r, d)
    bs, bp = bf_sup.clamp_min(0.0), bf_pri.clamp_min(0.0)
    none_b = (bs + bp) <= 0.0
    bs, bp = torch.where(none_b, one, bs), torch.where(none_b, one, bp)
    sup_p, pri_p = _split2(base, bs, bp)
    return roimp_p, sdro_p, ssur_p, sif_p, sup_p, pri_p


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
    n_inc: int,
    fracp_floor: float,
    eused_ext: torch.Tensor | None = None,
    sac_exchanges: bool = False,
    uztwc_pre: torch.Tensor | None = None,
    return_parts: bool = False,
) -> tuple[SacState, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One daily step; returns (new_state, surf, base, tet) in mm/day.

    ``return_parts`` appends the runoff parts in :data:`PART_NAMES` order: ``roimp``
    (impervious), ``sdro`` (ADIMP direct), ``ssur`` (surface), ``sif`` (interflow) of
    ``surf`` and ``supplemental`` / ``primary`` baseflow of ``base`` -- the final ``surf`` /
    ``base`` (after the et4 channel-ET deduction and the dry-channel clamp) apportioned in
    proportion to the parts' pre-deduction values (:func:`_apportion_parts`), so the sums are
    exact.

    Without ``eused_ext`` the reference ET cascade E1-E5 runs.  With it the ET was withdrawn
    upstream (Noah-lite, :func:`run_sacsma`): ``eused_ext`` is the soil ET taken, which feeds
    the riparian ``et4`` only; with ``sac_exchanges`` the rest of the reference ET block runs
    after the withdrawal -- the upper free -> tension rebalance, the lower free -> tension
    resupply and the ADIMP ET(5), whose ET1 is the upper-tension withdrawal ``uztwc_pre -
    uztwc``.
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

    if eused_ext is not None:
        # ET applied upstream; no withdrawal here.  eused feeds et4 only.
        et1 = et2 = et3 = et5 = zero
        eused = eused_ext
        if sac_exchanges:
            if uztwc_pre is None:
                raise ValueError("sac_exchanges needs uztwc_pre (the UZ tension "
                                 "content before the external withdrawal)")
            # the external withdrawal stands in for E1-E3; the reference branch
            # below from the UZ rebalance on, with et1 = the UZ tension withdrawal
            # and red + et2 = edmnd - et1 (true on both reference branches)
            et1_ext = uztwc_pre - uztwc
            exhausted = (uztwc <= 0.0).to(dtype)
            do_rb = (1.0 - exhausted) * (uztwc / uztwm < uzfwc / uzfwm).to(dtype)
            uzrat = (uztwc + uzfwc) / (uztwm + uzfwm)
            uztwc = do_rb * (uztwm * uzrat) + (1.0 - do_rb) * uztwc
            uzfwc = do_rb * (uzfwm * uzrat) + (1.0 - do_rb) * uzfwc
            not_ex = 1.0 - exhausted
            uztwc = torch.where((uztwc < _THRES) & (not_ex > 0), zero, uztwc)
            uzfwc = torch.where((uzfwc < _THRES) & (not_ex > 0), zero, uzfwc)

            # ---- resupply lower free -> lower tension (reference block) ----
            saved = rserv * (lzfpm + lzfsm)
            ratlzt = lztwc / lztwm
            ratlz = (lztwc + lzfpc + lzfsc - saved) / (lztwm + lzfpm + lzfsm - saved)
            resup = (ratlzt < ratlz).to(dtype)
            dele = resup * (ratlz - ratlzt) * lztwm
            lztwc = lztwc + dele
            lzfsc_raw = lzfsc - dele
            lzfpc = lzfpc + torch.minimum(lzfsc_raw, zero)
            lzfsc = lzfsc_raw.clamp_min(0.0)
            lztwc = _snap(lztwc)

            # ---- ET(5): ADIMP area (reference form, no lower clamp) ----
            et5_raw = et1_ext + (edmnd - et1_ext) * (adimc - et1_ext - uztwc) / (uztwm + lztwm)
            et5 = torch.minimum(et5_raw, adimc)
            adimc = adimc - et5
            et5 = et5 * adimp
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

    # ---- substep loop: n_inc substeps for every lane ----
    sbf = torch.zeros_like(uztwc)
    ssur = torch.zeros_like(uztwc)
    sif = torch.zeros_like(uztwc)
    sdro = torch.zeros_like(uztwc)
    if return_parts:   # primary / supplemental baseflow kept apart (sbf is unchanged)
        sbf_p = torch.zeros_like(uztwc)
        sbf_s = torch.zeros_like(uztwc)

    dinc = 1.0 / n_inc
    pinc = twx / n_inc
    duz = 1.0 - (1.0 - uzk) ** dinc
    dlzp = 1.0 - (1.0 - lzpk) ** dinc
    dlzs = 1.0 - (1.0 - lzsk) ** dinc
    percm = lzfpm * dlzp + lzfsm * dlzs

    for _ in range(n_inc):
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
        amp = 1.0 + zperc * defr ** rexp
        perc = torch.minimum(percm * uzfwc / uzfwm * amp, uzfwc)
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

        uzfwc, lztwc, lzfsc, lzfpc, adimc = uzfwc_n, lztwc_n, lzfsc_n, lzfpc_n, adimc_n
        sbf = sbf + (bf_p + bf_s)
        if return_parts:
            sbf_p = sbf_p + bf_p
            sbf_s = sbf_s + bf_s
        sif = sif + act * dele_if
        ssur = ssur + act * (sur * parea + adsur * adimp)
        sdro = sdro + addro * adimp

    # ---- aggregate channel inflow (reference order: et4 from UNWEIGHTED eused) ----
    # eused: et1 + et2 + et3 of the cascade, or the upstream withdrawal
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
    if return_parts:
        # roimp/sdro/ssur/sif above are the pre-deduction surface parts; the
        # baseflow parts get the same parea / (1 + side) factor as bfcc
        parts = _apportion_parts(surf, base, roimp, sdro, ssur, sif,
                                 sbf_s * parea / (1.0 + side),
                                 sbf_p * parea / (1.0 + side))
        return new_state, surf, base, tet, parts
    return new_state, surf, base, tet


def run_sacsma(
    pet: torch.Tensor,       # (N, T) mm/day
    pr_eff: torch.Tensor,    # (N, T) mm/day (Snow-17 outflow)
    params: dict[str, torch.Tensor],   # (N,) each, ga_optimum names (+ soil_chi for Noah-lite)
    state: SacState,
    physics,                 # sacsma.engine.Physics
    *,
    veg_frac: torch.Tensor | None = None,   # (N,) observed green fraction (Noah-lite)
    lai: torch.Tensor | None = None,        # (N, T) observed LAI (Noah-lite)
    return_parts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SacState]:
    """Run SAC-SMA over a window; returns (surf, base, tet, final_state).

    ``return_parts`` appends the per-day runoff parts of :func:`sacsma_step`, six (N, T)
    tensors in :data:`PART_NAMES` order.

    ``physics.et == "noah_lite"`` withdraws the Noah-lite ET (:func:`.et_noah.noah_lite_et`,
    exponent ``params["soil_chi"]``) ahead of each day's water balance.
    """
    t_len = pet.shape[1]
    noah = physics.et == "noah_lite"
    step_kw = {"n_inc": physics.n_inc, "fracp_floor": physics.fracp_floor,
               "return_parts": return_parts}
    grad = torch.is_grad_enabled() and (
        pet.requires_grad or pr_eff.requires_grad
        or any(v.requires_grad for v in params.values()))
    if grad:
        surf_s: list[torch.Tensor] = []
        base_s: list[torch.Tensor] = []
        tet_s: list[torch.Tensor] = []
    else:
        surf = torch.empty_like(pet)
        base = torch.empty_like(pet)
        tet = torch.empty_like(pet)
    if return_parts:
        parts_s: list[tuple[torch.Tensor, ...]] = []
    for t in range(t_len):
        if noah:
            uztwc_pre = state.uztwc
            uztwc, uzfwc, lztwc, et_soil = noah_lite_et(
                state.uztwc, state.uzfwc, state.lztwc, pet[:, t], params, params["soil_chi"],
                veg_frac, lai[:, t])
            state = SacState(uztwc=uztwc, uzfwc=uzfwc, lztwc=lztwc, lzfsc=state.lzfsc,
                             lzfpc=state.lzfpc, adimc=state.adimc)
            # the Noah ET is the step's eused, so its eused*parea term reports it (no double
            # count): te = tet_noah*parea + et4 [+ et5 with sac_exchanges]
            step = sacsma_step(state, pr_eff[:, t], pet[:, t], params, eused_ext=et_soil,
                               sac_exchanges=physics.sac_exchanges, uztwc_pre=uztwc_pre,
                               **step_kw)
        else:
            step = sacsma_step(state, pr_eff[:, t], pet[:, t], params, **step_kw)
        state, sf, bs, te = step[:4]
        if return_parts:
            parts_s.append(step[4])
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
    if return_parts:
        parts = tuple(torch.stack([p[k] for p in parts_s], dim=-1)
                      for k in range(len(PART_NAMES)))
        return surf, base, tet, state, parts
    return surf, base, tet, state
