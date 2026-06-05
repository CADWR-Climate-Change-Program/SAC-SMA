"""SNOW-17 snow accumulation/ablation — faithful port of ``snow_snow17_fix.m``.

This is the ``_fix`` variant: a HARD rain/snow split at ``PXTEMP`` (the TTI
linear mixture in the original is intentionally disabled — do not re-enable
it without breaking parity).

Parameter order (10):
    SCF, PXTEMP, MFMAX, MFMIN, UADJ, MBASE, TIPM, PLWHC, NMF, DAYGM
State (4): W_i (ice WE), ATI, W_q (liquid), Deficit (heat deficit).
``dtt = dtp = 24`` h.  The model outflow ``E`` (rain + melt leaving the
pack) is the effective precipitation fed to SAC-SMA.
"""

from __future__ import annotations

import numpy as np

from ._compat import njit

SNOW17_PARAM_NAMES = (
    "SCF", "PXTEMP", "MFMAX", "MFMIN", "UADJ",
    "MBASE", "TIPM", "PLWHC", "NMF", "DAYGM",
)


@njit
def _snow17_core(prcp, tavg, doy, is_leap, elev, par, state):
    SCF = par[0]; PXTEMP = par[1]; MFMAX = par[2]; MFMIN = par[3]; UADJ = par[4]
    MBASE = par[5]; TIPM = par[6]; PLWHC = par[7]; NMF = par[8]; DAYGM = par[9]

    W_i = state[0]; ATI = state[1]; W_q = state[2]; Deficit = state[3]

    n = prcp.shape[0]
    outflow = np.empty(n)
    melt = np.empty(n)
    swe = np.empty(n)
    intot = np.empty(n)

    dtt = 24.0
    dtp = 24.0

    for t in range(n):
        Ta = tavg[t]
        Pr = prcp[t]

        # ---- form of precipitation (hard split) ----
        if Ta <= PXTEMP:
            SNOW = Pr
            RAIN = 0.0
        else:
            SNOW = 0.0
            RAIN = Pr

        # ---- accumulation ----
        Pn = SNOW * SCF
        W_i = W_i + Pn

        # ---- seasonal non-rain melt factor ----
        DAYN = doy[t]
        if is_leap[t] == 1:
            days = 366.0
            N_Mar21 = DAYN - 81.0
        else:
            days = 365.0
            N_Mar21 = DAYN - 80.0
        Sv = (0.5 * np.sin((N_Mar21 * 2.0 * np.pi) / days)) + 0.5
        Av = 1.0
        Mf = (dtt / 6.0) * ((Sv * Av * (MFMAX - MFMIN)) + MFMIN)

        # ---- new-snow temperature & heat deficit ----
        if Ta < 0.0:
            T_snow_new = Ta
        else:
            T_snow_new = 0.0
        delta_HD_snow = -(T_snow_new * Pn) / (80.0 / 0.5)
        delta_HD_T = NMF * (dtp / 6.0) * (Mf / MFMAX) * (ATI - T_snow_new)

        if Pn > (1.5 * dtp):
            ATI = T_snow_new
        else:
            TIPM_dtt = 1.0 - ((1.0 - TIPM) ** (dtt / 6.0))
            ATI = ATI + TIPM_dtt * (Ta - ATI)
        if ATI > 0.0:
            ATI = 0.0

        # ---- snow melt ----
        if Ta > 0.0:
            T_rain = Ta
        else:
            T_rain = 0.0
        if RAIN > (0.25 * dtp):
            stefan = 6.12e-10
            e_sat = 2.7489e8 * np.exp(-4278.63 / (Ta + 242.792))
            P_atm = 33.86 * (29.9 - (0.335 * (elev / 100.0)) + (0.00022 * ((elev / 100.0) ** 2.4)))
            term1 = stefan * dtp * (((Ta + 273.0) ** 4) - (273.0 ** 4))
            term2 = 0.0125 * RAIN * T_rain
            term3 = 8.5 * UADJ * (dtp / 6.0) * ((0.9 * e_sat - 6.11) + (0.00057 * P_atm * Ta))
            Melt = term1 + term2 + term3
            if Melt < 0.0:
                Melt = 0.0
        elif (RAIN <= (0.25 * dtp)) and (Ta > MBASE):
            Melt = (Mf * (Ta - MBASE) * (dtp / dtt)) + (0.0125 * RAIN * T_rain)
            if Melt < 0.0:
                Melt = 0.0
        else:
            Melt = 0.0

        # ---- ripeness ----
        Deficit = Deficit + delta_HD_snow + delta_HD_T
        if Deficit < 0.0:
            Deficit = 0.0

        if Melt < W_i:
            W_i = W_i - Melt
            Qw = Melt + RAIN
            W_qx = PLWHC * W_i
            if Deficit > (0.33 * W_i):
                Deficit = 0.33 * W_i
            if (Qw + W_q) > (Deficit + Deficit * PLWHC + W_qx):
                E = Qw + W_q - W_qx - Deficit - (Deficit * PLWHC)
                W_i = W_i + Deficit
                W_q = W_qx
                Deficit = 0.0
            elif (Qw >= Deficit) and ((Qw + W_q) <= ((Deficit * (1.0 + PLWHC)) + W_qx)):
                E = 0.0
                W_i = W_i + Deficit
                W_q = W_q + Qw - Deficit
                Deficit = 0.0
            else:
                E = 0.0
                W_i = W_i + Qw
                Deficit = Deficit - Qw
        else:
            Melt = W_i + W_q
            W_i = 0.0
            W_q = 0.0
            Qw = Melt + RAIN
            E = Qw

        if Deficit == 0.0:
            ATI = 0.0

        # ---- constant ground melt ----
        if W_i > DAYGM:
            gmwlos = (DAYGM / W_i) * W_q
            gmslos = DAYGM
            gmro = gmwlos + gmslos
            W_i = W_i - gmslos
            W_q = W_q - gmwlos
            E = E + gmro
            SWE = W_i + W_q
        else:
            gmro = W_i + W_q
            W_i = 0.0
            W_q = 0.0
            E = E + gmro
            SWE = 0.0

        outflow[t] = E
        melt[t] = Melt
        swe[t] = SWE
        intot[t] = Pn + RAIN

    new_state = np.array([W_i, ATI, W_q, Deficit])
    return outflow, melt, swe, new_state, intot


def snow17(
    prcp: np.ndarray,
    tavg: np.ndarray,
    doy: np.ndarray,
    is_leap: np.ndarray,
    elev: float,
    par,
    init_state=None,
):
    """Run SNOW-17 over a daily forcing window.

    Returns ``(outflow, melt, swe, new_state, intot)`` where ``outflow`` is
    the effective precipitation (rain + melt) fed to SAC-SMA.
    """
    prcp = np.asarray(prcp, dtype=float)
    tavg = np.asarray(tavg, dtype=float)
    doy = np.asarray(doy, dtype=np.int64)
    is_leap = np.asarray(is_leap, dtype=np.int64)
    par = np.asarray(par, dtype=float)
    if init_state is None:
        init_state = np.zeros(4)
    init_state = np.asarray(init_state, dtype=float)
    return _snow17_core(prcp, tavg, doy, is_leap, float(elev), par, init_state)
