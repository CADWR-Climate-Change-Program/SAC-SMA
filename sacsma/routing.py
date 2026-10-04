"""Lohmann routing — vectorized NumPy port of ``rout_lohmann.m``.

A gamma-distribution HRU/hillslope unit hydrograph (UH) is combined with a
river-channel UH derived from the Green's function of the linearised
Saint-Venant equation, then convolved with the catchment direct runoff and
baseflow series.

The MATLAB ``O(TMAX^2)`` channel-UH double sum and the full-record
convolution are expressed here as ``np.convolve`` calls — mathematically
identical (a convolution IS that double sum) but C-speed, so routing no
longer needs Numba.

Parameter order (4): N (UH shape), K (UH scale), VELO (m/s), DIFF (m^2/s).
``flowlen`` is the travel distance (m) from the catchment outlet to the
watershed outlet; ``is_outlet=1`` collapses the channel UH to identity.
Constants follow the MATLAB: KE=12, UH_DAY=96, DT=3600 s, LE=48*50.
"""

from __future__ import annotations

import math

import numpy as np

from ._compat import njit

ROUTING_PARAM_NAMES = ("N", "K", "VELO", "DIFF")

_KE = 12
_UH_DAY = 96
_DT = 3600.0
_TMAX = _UH_DAY * 24      # 2304
_LE = 48 * 50             # 2400


def _river_uh(flowlen: float, velo: float, diff: float, is_outlet: int) -> np.ndarray:
    """Daily river-routing UH (length UH_DAY) from the Saint-Venant Green's fn."""
    if is_outlet == 1:
        uh = np.zeros(_UH_DAY)
        uh[0] = 1.0
        return uh

    t = _DT * np.arange(1, _LE + 1)
    pot = (velo * t - flowlen) ** 2 / (4.0 * diff * t)
    H = np.where(
        pot <= 69.0,
        flowlen / (2.0 * t * np.sqrt(np.pi * t * diff)) * np.exp(-np.minimum(pot, 69.0)),
        0.0,
    )
    s = H.sum()
    uhm = np.zeros(_LE)
    if s == 0.0:
        uhm[0] = 1.0
    else:
        uhm = H / s

    fr1 = np.zeros(_TMAX)
    fr1[:24] = 1.0 / 24.0
    # MATLAB FR(t,2) = sum_L FR(t-L,1)*UHM(L)  ==  causal convolution (offset 1).
    conv = np.convolve(fr1, uhm)
    fr2 = np.concatenate(([0.0], conv))[:_TMAX]
    return fr2.reshape(_UH_DAY, 24).sum(axis=1)


def _hru_uh_direct(n: float, k: float) -> np.ndarray:
    """HRU hillslope gamma UH (length KE), integrated per day as in MATLAB."""
    theta = 1.0 / k
    gN = math.gamma(n)
    # x grid: KE bins of [24*(i-1), 24*i], 1001 points each (matches MATLAB).
    i = np.arange(1, _KE + 1)[:, None]
    x = np.linspace(0.0, 1.0, 1001)[None, :] * 24.0 + 24.0 * (i - 1)
    dx = x[0, 1] - x[0, 0]
    integrand = (1.0 / theta / gN) * np.power(x / theta, n - 1.0) * np.exp(-x / theta)
    return integrand.sum(axis=1) * dx


def lohmann(inflow_direct, inflow_base, flowlen, route_par, is_outlet):
    """Route a catchment's direct runoff and baseflow to the watershed outlet.

    Returns ``(runoff, baseflow)`` where ``runoff`` is the total routed flow
    (direct + base) and ``baseflow`` is the routed baseflow component.
    """
    inflow_direct = np.asarray(inflow_direct, dtype=float)
    inflow_base = np.asarray(inflow_base, dtype=float)
    n, k, velo, diff = (float(p) for p in route_par)

    uh_river = _river_uh(float(flowlen), velo, diff, int(is_outlet))
    uh_hru_direct = _hru_uh_direct(n, k)
    uh_hru_base = np.zeros(_KE)
    uh_hru_base[0] = 1.0

    uh_direct = np.convolve(uh_hru_direct, uh_river)
    uh_base = np.convolve(uh_hru_base, uh_river)
    uh_direct = uh_direct / uh_direct.sum()
    uh_base = uh_base / uh_base.sum()

    m = inflow_direct.shape[0]
    directflow = np.convolve(inflow_direct, uh_direct)[:m]
    baseflow = np.convolve(inflow_base, uh_base)[:m]
    runoff = directflow + baseflow
    return runoff, baseflow


# ---------------------------------------------------------------------------
# Numba re-expression of the unit hydrographs and the convolution above, for the
# engine (:mod:`sacsma.engine`).  ``np.convolve`` and ``math.gamma`` keep the
# reference ``lohmann`` off the JIT path, so the convolutions are written out as
# loops here; ``lohmann`` stays the reference.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _convolve_full_nb(a, b):
    """Full discrete convolution ``a * b`` (length ``len(a)+len(b)-1``)."""
    la = a.shape[0]
    lb = b.shape[0]
    out = np.zeros(la + lb - 1)
    for i in range(la):
        ai = a[i]
        if ai == 0.0:
            continue
        for j in range(lb):
            out[i + j] += ai * b[j]
    return out


@njit(cache=True)
def _river_uh_nb(flowlen, velo, diff, is_outlet):
    """njit port of :func:`_river_uh` (daily channel UH, length ``_UH_DAY``)."""
    uh = np.zeros(96)
    if is_outlet == 1:
        uh[0] = 1.0
        return uh
    LE = 2400
    DT = 3600.0
    H = np.zeros(LE)
    s = 0.0
    for i in range(LE):
        t = DT * (i + 1)
        pot = (velo * t - flowlen) ** 2 / (4.0 * diff * t)
        if pot <= 69.0:
            H[i] = flowlen / (2.0 * t * np.sqrt(np.pi * t * diff)) * np.exp(-pot)
        s += H[i]
    uhm = np.zeros(LE)
    if s == 0.0:
        uhm[0] = 1.0
    else:
        for i in range(LE):
            uhm[i] = H[i] / s
    # fr1 is 1/24 over its first 24 entries and 0 elsewhere, so
    # conv(fr1, uhm)[kc] = (1/24) * sum_{ii=0..23} uhm[kc-ii]; fr2 = [0, conv...],
    # then UH(day j) = sum of fr2 over the 24 sub-steps of that day.
    inv24 = 1.0 / 24.0
    for j in range(96):
        acc = 0.0
        for r in range(24):
            k = j * 24 + r          # fr2 index (fr2[0] = 0)
            if k == 0:
                continue
            kc = k - 1              # conv(fr1, uhm) index
            c = 0.0
            for ii in range(24):
                idx = kc - ii
                if 0 <= idx < LE:
                    c += uhm[idx]
            acc += inv24 * c
        uh[j] = acc
    return uh


@njit(cache=True)
def _hru_uh_direct_nb(n, k):
    """njit port of :func:`_hru_uh_direct` (hillslope gamma UH, length ``_KE``)."""
    KE = 12
    theta = 1.0 / k
    gN = math.gamma(n)
    dx = 24.0 / 1000.0
    uh = np.zeros(KE)
    for i in range(KE):
        base = 24.0 * i
        acc = 0.0
        for j in range(1001):
            x = (j / 1000.0) * 24.0 + base
            acc += (1.0 / theta / gN) * (x / theta) ** (n - 1.0) * np.exp(-x / theta)
        uh[i] = acc * dx
    return uh
