"""SAC-SMA soil moisture accounting — faithful port of ``sma_sacramento.m``.

Preserves the MATLAB numerics exactly, including:
  * variable sub-daily substeps ``ninc = floor(1 + 0.2*(uzfwc+twx))``,
  * the hardcoded ``ratio**2`` ADIMP direct-runoff exponent,
  * the ``thres_zero = 1e-5`` and ``0.0001`` storage clamps,
  * the et4/et5 riparian/ADIMP ET adjustments to channel inflow.

Parameter order (16):
    uztwm, uzfwm, lztwm, lzfpm, lzfsm, uzk, lzpk, lzsk,
    zperc, rexp, pfree, pctim, adimp, riva, side, rserv
State (6): uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc.
"""

from __future__ import annotations

import numpy as np

from ._compat import njit

SACSMA_PARAM_NAMES = (
    "uztwm", "uzfwm", "lztwm", "lzfpm", "lzfsm",
    "uzk", "lzpk", "lzsk",
    "zperc", "rexp", "pfree",
    "pctim", "adimp", "riva", "side", "rserv",
)


@njit
def _sacsma_core(pet, pr_eff, par, state):
    uztwm = par[0]; uzfwm = par[1]; lztwm = par[2]; lzfpm = par[3]; lzfsm = par[4]
    uzk = par[5]; lzpk = par[6]; lzsk = par[7]
    zperc = par[8]; rexp = par[9]; pfree = par[10]
    pctim = par[11]; adimp = par[12]; riva = par[13]; side = par[14]; rserv = par[15]

    uztwc = state[0]; uzfwc = state[1]; lztwc = state[2]
    lzfsc = state[3]; lzfpc = state[4]; adimc = state[5]

    n = pet.shape[0]
    surf_tot = np.empty(n)
    base_tot = np.empty(n)
    tet_tot = np.empty(n)

    thres_zero = 0.00001
    parea = 1.0 - adimp - pctim

    for i in range(n):
        pr = pr_eff[i]
        edmnd = pet[i]

        # ET(1): upper-zone tension
        et1 = edmnd * uztwc / uztwm
        red = edmnd - et1
        uztwc = uztwc - et1

        et2 = 0.0
        if uztwc <= 0.0:
            et1 = et1 + uztwc
            uztwc = 0.0
            red = edmnd - et1
            if uzfwc < red:
                et2 = uzfwc
                uzfwc = 0.0
                red = red - et2
                if uztwc < thres_zero:
                    uztwc = 0.0
                if uzfwc < thres_zero:
                    uzfwc = 0.0
            else:
                et2 = red
                uzfwc = uzfwc - et2
                red = 0.0
        else:
            if (uztwc / uztwm) < (uzfwc / uzfwm):
                uzrat = (uztwc + uzfwc) / (uztwm + uzfwm)
                uztwc = uztwm * uzrat
                uzfwc = uzfwm * uzrat
            if uztwc < thres_zero:
                uztwc = 0.0
            if uzfwc < thres_zero:
                uzfwc = 0.0

        # ET(3): lower-zone tension
        et3 = red * lztwc / (uztwm + lztwm)
        lztwc = lztwc - et3
        if lztwc < 0.0:
            et3 = et3 + lztwc
            lztwc = 0.0

        # resupply lower free -> lower tension
        saved = rserv * (lzfpm + lzfsm)
        ratlzt = lztwc / lztwm
        ratlz = (lztwc + lzfpc + lzfsc - saved) / (lztwm + lzfpm + lzfsm - saved)
        if ratlzt < ratlz:
            dele = (ratlz - ratlzt) * lztwm
            lztwc = lztwc + dele
            lzfsc = lzfsc - dele
            if lzfsc < 0.0:
                lzfpc = lzfpc + lzfsc
                lzfsc = 0.0
        if lztwc < thres_zero:
            lztwc = 0.0

        # ET(5): ADIMP area
        et5 = et1 + (red + et2) * (adimc - et1 - uztwc) / (uztwm + lztwm)
        adimc = adimc - et5
        if adimc < 0.0:
            et5 = et5 + adimc
            adimc = 0.0
        et5 = et5 * adimp

        # rainfall in excess of UZ tension
        twx = pr + uztwc - uztwm
        if twx < 0.0:
            uztwc = uztwc + pr
            twx = 0.0
        else:
            uztwc = uztwm
        adimc = adimc + pr - twx

        roimp = pr * pctim

        sbf = 0.0
        ssur = 0.0
        sif = 0.0
        sperc = 0.0
        sdro = 0.0

        ninc = int(np.floor(1.0 + 0.2 * (uzfwc + twx)))
        dinc = 1.0 / ninc
        pinc = twx / ninc
        duz = 1.0 - (1.0 - uzk) ** dinc
        dlzp = 1.0 - (1.0 - lzpk) ** dinc
        dlzs = 1.0 - (1.0 - lzsk) ** dinc

        for _n in range(ninc):
            adsur = 0.0

            ratio = (adimc - uztwc) / lztwm
            if ratio < 0.0:
                ratio = 0.0
            addro = pinc * (ratio ** 2)

            bf_p = lzfpc * dlzp
            lzfpc = lzfpc - bf_p
            if lzfpc <= 0.0001:
                bf_p = bf_p + lzfpc
                lzfpc = 0.0
            sbf = sbf + bf_p

            bf_s = lzfsc * dlzs
            lzfsc = lzfsc - bf_s
            if lzfsc <= 0.0001:
                bf_s = bf_s + lzfsc
                lzfsc = 0.0
            sbf = sbf + bf_s

            if (pinc + uzfwc) <= 0.01:
                uzfwc = uzfwc + pinc
            else:
                percm = lzfpm * dlzp + lzfsm * dlzs
                perc = percm * uzfwc / uzfwm
                defr = 1.0 - (lztwc + lzfpc + lzfsc) / (lztwm + lzfpm + lzfsm)
                if defr < 0.0:
                    defr = 0.0
                perc = perc * (1.0 + zperc * (defr ** rexp))
                if perc >= uzfwc:
                    perc = uzfwc
                uzfwc = uzfwc - perc
                check = lztwc + lzfpc + lzfsc + perc - lztwm - lzfpm - lzfsm
                if check > 0.0:
                    perc = perc - check
                    uzfwc = uzfwc + check
                sperc = sperc + perc

                dele = uzfwc * duz
                sif = sif + dele
                uzfwc = uzfwc - dele

                perct = perc * (1.0 - pfree)
                if (perct + lztwc) <= lztwm:
                    lztwc = lztwc + perct
                    percf = 0.0
                else:
                    percf = lztwc + perct - lztwm
                    lztwc = lztwm
                percf = percf + (perc * pfree)
                if percf != 0.0:
                    hpl = lzfpm / (lzfpm + lzfsm)
                    ratlp = lzfpc / lzfpm
                    ratls = lzfsc / lzfsm
                    fracp = hpl * 2.0 * (1.0 - ratlp) / (1.0 - ratlp + 1.0 - ratls)
                    if fracp > 1.0:
                        fracp = 1.0
                    percp = percf * fracp
                    percs = percf - percp
                    lzfsc = lzfsc + percs
                    if lzfsc > lzfsm:
                        percs = percs - lzfsc + lzfsm
                        lzfsc = lzfsm
                    lzfpc = lzfpc + percf - percs
                    if lzfpc >= lzfpm:
                        excess = lzfpc - lzfpm
                        lztwc = lztwc + excess
                        lzfpc = lzfpm

                if pinc != 0.0:
                    if (pinc + uzfwc) <= uzfwm:
                        uzfwc = uzfwc + pinc
                    else:
                        sur = pinc + uzfwc - uzfwm
                        uzfwc = uzfwm
                        ssur = ssur + (sur * parea)
                        adsur = sur * (1.0 - addro / pinc)
                        ssur = ssur + adsur * adimp

            adimc = adimc + pinc - addro - adsur
            if adimc > (uztwm + lztwm):
                addro = addro + adimc - (uztwm + lztwm)
                adimc = uztwm + lztwm
            sdro = sdro + (addro * adimp)
            if adimc < thres_zero:
                adimc = 0.0

        # aggregate channel inflow
        eused = et1 + et2 + et3
        sif = sif * parea
        tbf = sbf * parea
        bfcc = tbf * (1.0 / (1.0 + side))

        base = bfcc
        surf = roimp + sdro + ssur + sif

        et4 = (edmnd - eused) * riva

        if adimc < uztwc:
            adimc = uztwc

        ch_inflow = surf + base - et4
        if ch_inflow <= 0.0:
            et4 = surf + base
            surf = 0.0
            base = 0.0
        else:
            surf = surf - et4 / 2.0
            base = base - et4 / 2.0
            if surf < 0.0:
                base = base + surf
                surf = 0.0
            if base < 0.0:
                surf = surf + base
                base = 0.0

        eused = eused * parea
        tet = eused + et4 + et5

        surf_tot[i] = surf
        base_tot[i] = base
        tet_tot[i] = tet

    new_state = np.array([uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc])
    return surf_tot, base_tot, tet_tot, new_state


@njit
def _sacsma_core_seasonal(pet, pr_eff, par, uzk_t, lzpk_t, lzsk_t, state):
    """Time-varying-recession variant of :func:`_sacsma_core`.

    ``uzk``/``lzpk``/``lzsk`` (par[5:8]) are supplied as per-day arrays instead
    of scalars (the dPL seasonal-parameter experiment).  It steps the FROZEN
    ``_sacsma_core`` one day at a time, injecting that day's recession rates into
    a copy of ``par`` and carrying state forward -- so each day's numerics are
    byte-identical to the reference core, and with constant arrays this reduces
    EXACTLY to ``_sacsma_core`` (the reference physics is never duplicated or
    edited).  Kpet-seasonality is applied upstream as a PET scaling, not here.
    """
    n = pet.shape[0]
    surf_tot = np.empty(n)
    base_tot = np.empty(n)
    tet_tot = np.empty(n)
    st = state.copy()
    par_i = par.copy()
    for i in range(n):
        par_i[5] = uzk_t[i]
        par_i[6] = lzpk_t[i]
        par_i[7] = lzsk_t[i]
        s, b, t, st = _sacsma_core(pet[i:i + 1], pr_eff[i:i + 1], par_i, st)
        surf_tot[i] = s[0]
        base_tot[i] = b[0]
        tet_tot[i] = t[0]
    return surf_tot, base_tot, tet_tot, st


# Initial state used by the archived 15-CDEC GA run:
#   UZ uztwc=uzfwc=adimc=0 ; LZ lztwc=lzfsc=lzfpc=100.
# Order matches the state vector: [uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc].
DEFAULT_INIT_STATE = np.array([0.0, 0.0, 100.0, 100.0, 100.0, 0.0])


def sac_sma(pet: np.ndarray, pr_eff: np.ndarray, par, init_state=None,
            recession=None):
    """Run SAC-SMA over a daily window.

    Returns ``(surf, base, tet, new_state)`` in mm/day.  ``pr_eff`` is the
    SNOW-17 outflow (effective precipitation).  ``recession``, if given, is a
    tuple of per-day ``(uzk, lzpk, lzsk)`` arrays (the seasonal-parameter path);
    ``recession=None`` uses the scalar ``par[5:8]`` and is bit-identical to the
    reference core.
    """
    pet = np.asarray(pet, dtype=float)
    pr_eff = np.asarray(pr_eff, dtype=float)
    par = np.asarray(par, dtype=float)
    if init_state is None:
        init_state = DEFAULT_INIT_STATE.copy()
    init_state = np.asarray(init_state, dtype=float)
    if recession is None:
        return _sacsma_core(pet, pr_eff, par, init_state)
    uzk_t, lzpk_t, lzsk_t = (np.asarray(a, dtype=float) for a in recession)
    return _sacsma_core_seasonal(pet, pr_eff, par, uzk_t, lzpk_t, lzsk_t, init_state)
