"""Noah-LITE evapotranspiration SAC-SMA core (Numba) — frozen-pipeline mirror of
the torch ``et_mode='noah'`` + ``canopy_lite`` path (``sacsma.dpl.et_noah.
noah_lite_et_step`` interleaved with ``sacsma.dpl.sma.sacsma_step(et_mode=
'external')``) so a Noah-lite-trained dPL export scores through the fast
``run_basin`` path instead of streaming the full record through the torch
pipeline.

A NEW module: the frozen ``sma.py`` is untouched.  The daily water-balance —
throughfall split, the variable-``ninc`` substep loop, and the channel-inflow
aggregation — is transcribed VERBATIM from ``sma._sacsma_core`` (same MATLAB
numerics: variable ``ninc = floor(1 + 0.2*(uzfwc + twx))``, the ``ratio**2``
ADIMP runoff, the ``thres_zero``/``0.0001`` storage clamps, the et4 riparian
channel adjustment).  Only the ET step differs:

* the frozen E1-E5 tension/free cascade is replaced by the Noah-lite two-term
  AET — ``ed_bare = (1-sig)*ep*avail_up**chi`` + ``et_canopy = sig*ep*
  sm_root**chi`` on the pinned green fraction ``sig``, ONE learned exponent
  ``soil_chi`` — withdrawn from the SAC storages upstream of the water balance
  (``et_mode='external'``);
* the lower-free -> lower-tension resupply and the ET(5) ADIMP withdrawal are
  DROPPED (they live inside the frozen ET branch, skipped in external mode);
* ``eused`` feeding the riparian ``et4 = (edmnd - eused)*riva`` is the Noah-lite
  soil ET actually withdrawn (``noah_lite_et_step``'s ``tet``), not E1-E3.

Interception is off (throughfall == precip), matching ``canopy_lite``.  The
green fraction, wilting point and root split are pinned to the same constants as
the torch path (kept in sync below).  Any change here must be re-verified
against the torch Noah-lite pipeline (run with ``ninc_mode='dynamic'`` for the
bit-exact reference numerics; see ``scratchpad/verify_noah_lite_numba.py``).
"""

from __future__ import annotations

import numpy as np

from ._compat import njit

# -- pinned Noah-lite constants — verbatim from sacsma.dpl.et_noah (keep in sync)
_BEER_K = 0.5      # canopy extinction: sig = min(veg_frac, 1 - exp(-k*LAI))
_LITE_WILT = 0.05  # wilting point (fraction of tension capacity)
_LITE_FROOT = 0.7  # upper-zone root fraction
_EPS = 1e-6        # fractional-power base floor (parity with the torch clamp_min)


@njit
def _sacsma_noah_lite_core(pet, pr_eff, veg_frac, lai, soil_chi, par, state):
    """Daily SAC-SMA with the Noah-lite external ET for one HRU.

    ``pet``/``pr_eff`` are (T,) mm/day (Kpet-scaled PET and Snow-17 outflow);
    ``veg_frac``/``soil_chi`` are scalars; ``lai`` is the (T,) daily observed LAI
    (per-day seasonal climatology, indexed by day-of-year upstream); ``par`` is
    the 16 SMA params, ``state`` the 6-vector cold start.  Returns
    ``(surf, base, tet, new_state)`` in mm/day — the frozen ``sac_sma`` contract.
    """
    uztwm = par[0]; uzfwm = par[1]; lztwm = par[2]; lzfpm = par[3]; lzfsm = par[4]
    uzk = par[5]; lzpk = par[6]; lzsk = par[7]
    zperc = par[8]; rexp = par[9]; pfree = par[10]
    pctim = par[11]; adimp = par[12]; riva = par[13]; side = par[14]
    # par[15] rserv is unused: the lower-free->tension resupply it governs lives
    # inside the frozen ET branch, dropped in the external-ET (Noah-lite) path.

    uztwc = state[0]; uzfwc = state[1]; lztwc = state[2]
    lzfsc = state[3]; lzfpc = state[4]; adimc = state[5]

    n = pet.shape[0]
    surf_tot = np.empty(n)
    base_tot = np.empty(n)
    tet_tot = np.empty(n)

    thres_zero = 0.00001
    parea = 1.0 - adimp - pctim
    wilt = _LITE_WILT
    froot = _LITE_FROOT

    for i in range(n):
        pr = pr_eff[i]
        edmnd = pet[i]

        # -- Noah-lite ET (replaces the frozen E1-E5 cascade) ----------------
        # green fraction tracks LAI phenology (Beer's law), capped at observed
        # cover; sparse/dormant -> more bare soil exposed.
        beer = 1.0 - np.exp(-_BEER_K * lai[i])
        sig = veg_frac if veg_frac < beer else beer

        # root-zone available-water fractions (SAC tension saturations, wilt floor)
        sup = uztwc / uztwm
        if sup < 0.0:
            sup = 0.0
        elif sup > 1.0:
            sup = 1.0
        slo = lztwc / lztwm
        if slo < 0.0:
            slo = 0.0
        elif slo > 1.0:
            slo = 1.0
        avail_up = (sup - wilt) / (1.0 - wilt)
        if avail_up < 0.0:
            avail_up = 0.0
        elif avail_up > 1.0:
            avail_up = 1.0
        avail_lo = (slo - wilt) / (1.0 - wilt)
        if avail_lo < 0.0:
            avail_lo = 0.0
        elif avail_lo > 1.0:
            avail_lo = 1.0
        sm_root = froot * avail_up + (1.0 - froot) * avail_lo

        # bare-soil (surface) + canopy (root zone) ET, ONE shared exponent;
        # floor the fractional-power base so chi<1 at avail->0 stays finite.
        base_up = avail_up if avail_up > _EPS else _EPS
        ed = (1.0 - sig) * edmnd * base_up ** soil_chi
        if ed < 0.0:
            ed = 0.0
        base_root = sm_root if sm_root > _EPS else _EPS
        et = sig * edmnd * base_root ** soil_chi
        if et < 0.0:
            et = 0.0

        # withdraw: bare soil + upper transpiration from UZ (tension then free);
        # lower transpiration from LZ tension — same allocation as the torch path.
        et_up = froot * et
        et_lo = (1.0 - froot) * et
        dem_up = ed + et_up
        w_upt = dem_up if dem_up < uztwc else uztwc
        uztwc = uztwc - w_upt
        rem_up = dem_up - w_upt
        w_upf = rem_up if rem_up < uzfwc else uzfwc
        uzfwc = uzfwc - w_upf
        w_lo = et_lo if et_lo < lztwc else lztwc
        lztwc = lztwc - w_lo
        eused = w_upt + w_upf + w_lo    # Noah-lite soil ET (feeds et4)

        # -- rainfall in excess of UZ tension (VERBATIM sma._sacsma_core) -----
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

        # -- aggregate channel inflow (eused = Noah-lite soil ET; et5 = 0) ----
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

        tet = eused * parea + et4    # et5 == 0 in the external-ET path

        surf_tot[i] = surf
        base_tot[i] = base
        tet_tot[i] = tet

    new_state = np.array([uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc])
    return surf_tot, base_tot, tet_tot, new_state


def sac_sma_noah_lite(
    pet: np.ndarray,
    pr_eff: np.ndarray,
    veg_frac: float,
    lai: np.ndarray,
    soil_chi: float,
    par,
    init_state=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the Noah-lite-ET SAC-SMA over a daily window (frozen-pipeline mirror
    of the torch ``canopy_lite`` path).  Returns ``(surf, base, tet, new_state)``
    in mm/day — same contract as :func:`sacsma.sma.sac_sma`.

    ``veg_frac``/``soil_chi`` are the per-HRU pinned green fraction and learned
    moisture-limiter exponent; ``lai`` is the (T,) observed daily LAI.
    """
    from .sma import DEFAULT_INIT_STATE

    pet = np.asarray(pet, dtype=float)
    pr_eff = np.asarray(pr_eff, dtype=float)
    lai = np.asarray(lai, dtype=float)
    par = np.asarray(par, dtype=float)
    if init_state is None:
        init_state = DEFAULT_INIT_STATE.copy()
    init_state = np.asarray(init_state, dtype=float)
    return _sacsma_noah_lite_core(pet, pr_eff, float(veg_frac), lai,
                                  float(soil_chi), par, init_state)
