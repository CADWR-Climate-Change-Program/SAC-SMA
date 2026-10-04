"""SAC-SMA of the learned model, in Numba: the step a dPL network is trained with.

A learned parameter field is trained through the torch step (``sacsma.dpl.physics.sma.
sacsma_step``, with ``sacsma.dpl.physics.et_noah.noah_lite_et`` for the Noah-lite ET).
This module runs the same step on the CPU, written line for line in the torch forms, so a
trained field runs as it was trained, at Numba speed.  ``sacsma verify learned`` checks the two
against each other; any change to the torch step is made here too.

It is not the reference model (``sacsma.sma``, the frozen MATLAB transcription), and it differs
from it in three ways:

* a fixed number of substeps a day, ``n_inc`` (the reference takes
  ``floor(1 + 0.2 * (uzfwc + twx))``);
* a floor on the denominator of the lower-zone fill fraction, ``fracp_floor``;
* the percolation is capped at the lower-zone deficit and never negative, and the percolation
  split and the surface spill run on every active substep in ``min`` form.  Below capacity this
  is the reference to round-off; with a store above its capacity (the cold start
  ``[0, 0, 100, 100, 100, 0]`` puts the lower zone above many trained capacities) the reference
  percolates upward through its capacity check while this step drains the store.

``et_noah`` selects the evapotranspiration: False is the reference cascade E1-E5 (in the torch
forms); True is the Noah-lite external ET (bare soil + canopy on the observed green fraction,
one learned exponent ``soil_chi``), withdrawn ahead of the water balance, with
``sac_exchanges`` keeping the upper-zone rebalance, the lower-zone resupply and the ADIMP ET(5)
after it.

The runoff PARTS (``parts``, shape ``(6, T)``, or ``(0, 0)`` for none) are the net ``surf`` and
``base`` of each day apportioned in proportion to their pre-deduction parts, in the order of
``sacsma.dpl.physics.sma.PART_NAMES``: impervious, ADIMP direct, surface, interflow (sum
``surf``), supplemental, primary (sum ``base``), each sum exact.
"""

from __future__ import annotations

import numpy as np

from ._compat import njit

# the pinned Noah-lite constants of sacsma.dpl.physics.et_noah (keep in sync)
_BEER_K = 0.5      # green fraction = min(veg_frac, 1 - exp(-k LAI))
_WILT = 0.05       # wilting point, fraction of tension capacity
_FROOT = 0.7       # upper-zone root fraction
_EPS = 1e-6        # floor of the fractional-power bases
# the reference snaps (sacsma.dpl.physics.sma)
_THRES = 1e-5
_BF_EPS = 1e-4


@njit(inline="always")
def _mn(a, b):
    """``torch.minimum``: NaN when either is NaN (a cell with a missing input stays NaN)."""
    return a if (a < b or a != a) else b


@njit(inline="always")
def _mx(a, b):
    """``torch.maximum``."""
    return a if (a > b or a != a) else b


@njit(inline="always")
def _split2(total, a, b):
    """``total`` split in proportion to ``a, b >= 0``, the two shares summing to it exactly
    (``sacsma.dpl.physics.sma._split2``)."""
    den = a + b
    a_big = a >= b
    big = a if a_big else b
    w = big / den if den > 0.0 else 0.5
    hi = total * w
    lo = total - hi
    if a_big:
        return hi, lo
    return lo, hi


@njit
def sac_learned(pet, pr_eff, par, state, et_noah, veg_frac, lai, soil_chi, sac_exchanges,
                n_inc, fracp_floor, parts):
    """One HRU over a window.  ``pet`` and ``pr_eff`` (T,) mm/day; ``par`` the 16 SAC-SMA
    parameters; ``state`` the 6-vector; ``lai`` (T,) the observed LAI of each day (Noah-lite);
    ``veg_frac`` and ``soil_chi`` scalars.  Returns ``(surf, base, tet, new_state)`` in mm/day
    and fills ``parts`` when it has columns."""
    uztwm, uzfwm, lztwm, lzfpm, lzfsm = par[0], par[1], par[2], par[3], par[4]
    uzk, lzpk, lzsk, zperc, rexp, pfree = par[5], par[6], par[7], par[8], par[9], par[10]
    pctim, adimp, riva, side, rserv = par[11], par[12], par[13], par[14], par[15]
    uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc = (state[0], state[1], state[2], state[3],
                                                state[4], state[5])

    n = pet.shape[0]
    want_parts = parts.shape[1] > 0
    surf_o = np.empty(n)
    base_o = np.empty(n)
    tet_o = np.empty(n)
    parea = 1.0 - adimp - pctim
    dinc = 1.0 / n_inc
    duz = 1.0 - (1.0 - uzk) ** dinc
    dlzp = 1.0 - (1.0 - lzpk) ** dinc
    dlzs = 1.0 - (1.0 - lzsk) ** dinc
    percm = lzfpm * dlzp + lzfsm * dlzs
    hpl = lzfpm / (lzfpm + lzfsm)
    floor = _mx(fracp_floor, 1e-12)
    saved = rserv * (lzfpm + lzfsm)

    for i in range(n):
        edmnd = pet[i]
        pr = pr_eff[i]
        et5 = 0.0
        if et_noah:
            # ---- Noah-lite ET (et_noah.noah_lite_et), withdrawn ahead of the balance
            sig = _mn(veg_frac, 1.0 - np.exp(-_BEER_K * lai[i]))
            sup = _mn(_mx(uztwc / uztwm, 0.0), 1.0)
            slo = _mn(_mx(lztwc / lztwm, 0.0), 1.0)
            avail_up = _mn(_mx((sup - _WILT) / (1.0 - _WILT), 0.0), 1.0)
            avail_lo = _mn(_mx((slo - _WILT) / (1.0 - _WILT), 0.0), 1.0)
            sm_root = _FROOT * avail_up + (1.0 - _FROOT) * avail_lo
            ed = _mx((1.0 - sig) * edmnd * _mx(avail_up, _EPS) ** soil_chi, 0.0)
            et = _mx(sig * edmnd * _mx(sm_root, _EPS) ** soil_chi, 0.0)
            et_up = _FROOT * et
            et_lo = (1.0 - _FROOT) * et
            dem_up = ed + et_up
            uztwc_pre = uztwc
            w_upt = _mn(dem_up, uztwc)
            uztwc = uztwc - w_upt
            w_upf = _mn(dem_up - w_upt, uzfwc)
            uzfwc = uzfwc - w_upf
            w_lo = _mn(et_lo, lztwc)
            lztwc = lztwc - w_lo
            eused = w_upt + w_upf + w_lo
            if sac_exchanges:
                # the withdrawal stands in for E1-E3; the reference block from the UZ
                # rebalance on, ET1 = the UZ tension withdrawal
                et1 = uztwc_pre - uztwc
                if not (uztwc <= 0.0):
                    if uztwc / uztwm < uzfwc / uzfwm:
                        uzrat = (uztwc + uzfwc) / (uztwm + uzfwm)
                        uztwc = uztwm * uzrat
                        uzfwc = uzfwm * uzrat
                    if uztwc < _THRES:
                        uztwc = 0.0
                    if uzfwc < _THRES:
                        uzfwc = 0.0
                ratlzt = lztwc / lztwm
                ratlz = (lztwc + lzfpc + lzfsc - saved) / (lztwm + lzfpm + lzfsm - saved)
                dele = (ratlz - ratlzt) * lztwm if ratlzt < ratlz else 0.0
                lztwc = lztwc + dele
                lzfsc_raw = lzfsc - dele
                lzfpc = lzfpc + _mn(lzfsc_raw, 0.0)
                lzfsc = _mx(lzfsc_raw, 0.0)
                if lztwc < _THRES:
                    lztwc = 0.0
                et5 = _mn(et1 + (edmnd - et1) * (adimc - et1 - uztwc) / (uztwm + lztwm), adimc)
                adimc = adimc - et5
                et5 = et5 * adimp
        else:
            # ---- the reference cascade E1-E5
            et1 = _mn(edmnd * uztwc / uztwm, uztwc)
            uztwc = uztwc - et1
            red = edmnd - et1
            if uztwc <= 0.0:
                et2 = _mn(red, uzfwc)
                uzfwc = uzfwc - et2
                red = red - et2
            else:
                et2 = 0.0
                if uztwc / uztwm < uzfwc / uzfwm:
                    uzrat = (uztwc + uzfwc) / (uztwm + uzfwm)
                    uztwc = uztwm * uzrat
                    uzfwc = uzfwm * uzrat
                if uztwc < _THRES:
                    uztwc = 0.0
                if uzfwc < _THRES:
                    uzfwc = 0.0
            et3 = _mn(red * lztwc / (uztwm + lztwm), lztwc)
            lztwc = lztwc - et3
            ratlzt = lztwc / lztwm
            ratlz = (lztwc + lzfpc + lzfsc - saved) / (lztwm + lzfpm + lzfsm - saved)
            dele = (ratlz - ratlzt) * lztwm if ratlzt < ratlz else 0.0
            lztwc = lztwc + dele
            lzfsc_raw = lzfsc - dele
            lzfpc = lzfpc + _mn(lzfsc_raw, 0.0)
            lzfsc = _mx(lzfsc_raw, 0.0)
            if lztwc < _THRES:
                lztwc = 0.0
            et5 = _mn(et1 + (red + et2) * (adimc - et1 - uztwc) / (uztwm + lztwm), adimc)
            adimc = adimc - et5
            et5 = et5 * adimp
            eused = et1 + et2 + et3

        # ---- throughfall split + impervious runoff
        twx = pr + uztwc - uztwm
        if twx >= 0.0:
            uztwc = uztwm
        else:
            uztwc = uztwc + pr
            twx = 0.0
        adimc = adimc + pr - twx
        roimp = pr * pctim

        # ---- fixed substeps
        pinc = twx / n_inc
        sbf = sbf_p = sbf_s = ssur = sif = sdro = 0.0
        for _s in range(n_inc):
            ratio = _mx((adimc - uztwc) / lztwm, 0.0)
            addro = pinc * ratio * ratio
            bf_p = lzfpc * dlzp
            lzfpc = lzfpc - bf_p
            if lzfpc <= _BF_EPS:
                bf_p = bf_p + lzfpc
                lzfpc = 0.0
            bf_s = lzfsc * dlzs
            lzfsc = lzfsc - bf_s
            if lzfsc <= _BF_EPS:
                bf_s = bf_s + lzfsc
                lzfsc = 0.0
            adsur = 0.0
            if (pinc + uzfwc) <= 0.01:
                # percolation short-circuit
                uzfwc = uzfwc + pinc
            else:
                lz_def = (lztwm + lzfpm + lzfsm) - (lztwc + lzfpc + lzfsc)
                defr = _mx(1.0 - (lztwc + lzfpc + lzfsc) / (lztwm + lzfpm + lzfsm), 0.0)
                perc = _mn(percm * uzfwc / uzfwm * (1.0 + zperc * defr ** rexp), uzfwc)
                perc = _mn(perc, _mx(lz_def, 0.0))
                uzfwc = uzfwc - perc
                dele_if = uzfwc * duz
                uzfwc = uzfwc - dele_if
                sif = sif + dele_if
                # percolation: LZ tension first, the overflow and pfree to the free stores
                perct = perc * (1.0 - pfree)
                perct_in = _mn(perct, lztwm - lztwc)
                lztwc_a = lztwc + perct_in
                percf = (perct - perct_in) + perc * pfree
                ratlp = lzfpc / lzfpm
                ratls = lzfsc / lzfsm
                fracp = _mn(hpl * 2.0 * (1.0 - ratlp) / _mx((1.0 - ratlp) + (1.0 - ratls), floor),
                            1.0)
                percs = percf * (1.0 - fracp)
                percs_in = _mn(percs, lzfsm - lzfsc)
                lzfsc = lzfsc + percs_in
                into_p = percf - percs_in
                percp_in = _mn(into_p, lzfpm - lzfpc)
                lzfpc = lzfpc + percp_in
                lztwc = lztwc_a + (into_p - percp_in)
                # pinc fills UZ free water, the rest spills to the surface
                into_uz = _mn(pinc, uzfwm - uzfwc)
                sur = pinc - into_uz
                uzfwc = uzfwc + into_uz
                adsur = sur * (1.0 - ratio * ratio)
                ssur = ssur + (sur * parea + adsur * adimp)
            adimc = adimc + pinc - addro - adsur
            over = _mx(adimc - (uztwm + lztwm), 0.0)
            addro = addro + over
            adimc = adimc - over
            if adimc < _THRES:
                adimc = 0.0
            sbf = sbf + (bf_p + bf_s)
            sbf_p = sbf_p + bf_p
            sbf_s = sbf_s + bf_s
            sdro = sdro + addro * adimp

        # ---- channel inflow: et4 from the unweighted eused, the dry-channel clamp
        sif = sif * parea
        base = sbf * parea / (1.0 + side)
        surf = roimp + sdro + ssur + sif
        et4 = (edmnd - eused) * riva
        adimc = _mx(adimc, uztwc)
        if surf + base - et4 <= 0.0:
            et4 = surf + base
            surf_n = 0.0
            base_n = 0.0
        else:
            half = 0.5 * et4
            surf_h = surf - half
            base_h = base - half
            surf_n = _mx(surf_h + _mn(base_h, 0.0), 0.0)
            base_n = _mx(base_h + _mn(surf_h, 0.0), 0.0)
        surf_o[i] = surf_n
        base_o[i] = base_n
        tet_o[i] = eused * parea + et4 + et5

        if want_parts:
            r, d, s, f = _mx(roimp, 0.0), _mx(sdro, 0.0), _mx(ssur, 0.0), _mx(sif, 0.0)
            if ((r + d) + s) + f <= 0.0:
                r = d = s = f = 1.0
            quick, sif_p = _split2(surf_n, (r + d) + s, f)
            imp, ssur_p = _split2(quick, r + d, s)
            roimp_p, sdro_p = _split2(imp, r, d)
            bs = _mx(sbf_s * parea / (1.0 + side), 0.0)
            bp = _mx(sbf_p * parea / (1.0 + side), 0.0)
            if bs + bp <= 0.0:
                bs = bp = 1.0
            sup_p, pri_p = _split2(base_n, bs, bp)
            parts[0, i] = roimp_p
            parts[1, i] = sdro_p
            parts[2, i] = ssur_p
            parts[3, i] = sif_p
            parts[4, i] = sup_p
            parts[5, i] = pri_p

    new_state = np.array([uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc])
    return surf_o, base_o, tet_o, new_state
