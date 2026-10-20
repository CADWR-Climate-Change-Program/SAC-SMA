"""The simulation engine: a fixed parameter field over a forcing and period, on the CPU.

Every run of a parameter field that is not being trained goes through :func:`simulate`: the GA
calibrations through the reference model (``sacsma.sma``), a trained dPL field through the step
it was trained with (``sacsma.sma_learned``).  Snow-17, the PET and the unit hydrographs are the
frozen cores in both.

A :class:`Field` holds the parameters on physics units and the routed rows that read them.  A
unit is a forcing cell with one parameter set; rows with the same cell and parameters (the
(entity, cell) rows of a multifamily run) share a unit, so the physics runs once per unit and
the routing once per row, exactly as one physics run per row would.  :func:`simulate` returns
weighted sums of the routed rows.

The state at ``start`` is the reference cold start (snow empty, SAC-SMA ``[0, 0, 100, 100, 100,
0]``, routing empty) under ``spinup="cold"``; under ``"cycle"`` it is the cold start looped
``passes`` times over the first ``years`` years from ``start`` (``sacsma.dpl.spinup``); under
``"window"`` the run starts cold at ``window_start`` and the days before ``start`` are dropped.
A learned field with the deep store (``Physics.deep``) starts the store empty; under ``"window"``
its content at ``start`` is divided by ``1 - (1 - kdeep)**days``, the level a constant feed
reaches from empty, so the window starts near the store's steady state as the trainer's does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._compat import njit
from .pet import _hamon_core
from .pet_pt import _pt_core
from .routing import _KE, _UH_DAY, _convolve_full_nb, _hru_uh_direct_nb, _river_uh_nb
from .sma import DEFAULT_INIT_STATE, _sacsma_core
from .sma_learned import sac_learned
from .snow17 import _snow17_core

try:
    from numba import prange
except ImportError:  # pragma: no cover
    prange = range

#: taps of the routed unit hydrograph and the inflow days it reaches back
N_TAPS = _KE + _UH_DAY - 1      # 107
N_HIST = N_TAPS - 1
# The three kernels below are compiled in each process (``cache=False``, a few seconds): Numba
# does not invalidate a cached kernel when a function it calls changes in another file, so a
# cached kernel could keep running an edited sma_learned step.

#: the routed outputs: flow (= fast + slow), the routed direct and base inflow, and the four
#: runoff parts (quick + interflow = fast, supplemental + primary = slow)
OUTPUTS = ("flow", "fast", "slow", "quick", "interflow", "supplemental", "primary")
#: the deep store's diagnostics (a field with it): ``deep``, its outflow's share of the
#: supplemental part, routed as it; ``gwc``, its content (mm, not routed)
DEEP_OUTPUTS = ("deep", "gwc")


@dataclass(frozen=True)
class Physics:
    et: str = "sac"                      # "sac" | "noah_lite"
    pet: str = "hamon"                   # "hamon" | "priestley_taylor"
    learned: bool = False                # sma_learned instead of the reference sma
    n_inc: int = 10                      # learned: substeps a day
    fracp_floor: float = 1e-3            # learned
    deep: str = "none"                   # learned: "none" | "cascade" (the deep store)


@dataclass
class Field:
    physics: Physics
    cell: np.ndarray        # (U,) forcing row of each unit
    lat_rad: np.ndarray     # (U,)
    elev: np.ndarray        # (U,)
    kpet: np.ndarray        # (U,)
    snow: np.ndarray        # (U, 10) Snow-17 parameters (sacsma.parameters._SNOW_COLS)
    sma: np.ndarray         # (U, 16) SAC-SMA parameters (sacsma.parameters._SMA_COLS)
    veg: np.ndarray         # (U,) observed green-vegetation fraction (Noah-lite)
    chi: np.ndarray         # (U,) soil_chi (Noah-lite)
    lai: np.ndarray         # (U, 366) observed LAI by day of year (Noah-lite)
    deep: np.ndarray        # (U, 2) gdeep, kdeep (the deep store; zeros without it)
    unit: np.ndarray        # (R,) unit of each routed row
    flowlen: np.ndarray     # (R,)
    route: np.ndarray       # (R, 4) Nres, Kres, Velo, Diff

    @classmethod
    def from_rows(cls, physics: Physics, cell, lat_rad, elev, flowlen, kpet, snow, sma, route,
                  veg=None, chi=None, lai=None, deep=None) -> Field:
        """One row per routed row; rows with the same cell and the same physics inputs become
        one unit (``lai`` is ``(R, 366)``, ``deep`` ``(R, 2)``)."""
        f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731
        R = len(cell)
        noah = physics.et == "noah_lite"
        if physics.deep == "cascade" and deep is None:
            raise ValueError("a field with the deep store needs deep (R, 2): gdeep, kdeep")
        veg = f64(veg if noah else np.zeros(R))
        chi = f64(chi if noah else np.zeros(R))
        lai = f64(lai if noah else np.zeros((R, 1)))
        deep = f64(deep if physics.deep == "cascade" else np.zeros((R, 2)))
        cols = [np.asarray(cell, np.float64)[:, None], f64(lat_rad)[:, None], f64(elev)[:, None],
                f64(kpet)[:, None], f64(snow), f64(sma), veg[:, None], chi[:, None], lai, deep]
        key = np.ascontiguousarray(np.hstack(cols))
        _, first, unit = np.unique(key.view(np.dtype((np.void, key.shape[1] * 8))).ravel(),
                                   return_index=True, return_inverse=True)
        u = np.sort(first)                       # units in row order
        unit = np.searchsorted(u, first[unit.ravel()])
        pick = lambda a: np.ascontiguousarray(a[u])  # noqa: E731
        return cls(physics, pick(np.asarray(cell, np.int64)), pick(f64(lat_rad)), pick(f64(elev)),
                   pick(f64(kpet)), pick(f64(snow)), pick(f64(sma)), pick(veg), pick(chi),
                   pick(lai), pick(deep), unit.astype(np.int64), f64(flowlen), f64(route))


def block_end(dates: pd.DatetimeIndex, t0: int, years: int) -> int:
    """Index ``years`` years after ``dates[t0]`` (the same calendar day), clipped to the record."""
    return min(int(dates.searchsorted(dates[t0] + pd.DateOffset(years=years))), len(dates))


def year_index(dates: pd.DatetimeIndex, t0: int, t1: int) -> np.ndarray:
    """Year of block (0, 1, ...) of each day in ``[t0, t1)``, counted from ``dates[t0]``."""
    d = dates[t0:t1]
    ann = pd.DatetimeIndex([dates[t0] + pd.DateOffset(years=k)
                            for k in range(int((d[-1] - d[0]).days // 365) + 2)])
    return (ann.searchsorted(d, side="right") - 1).astype(np.int64)


@njit(parallel=True, cache=False)
def _units(prcp, tavg, tmin, tmax, doy_f, doy_i, leap, cell, lat_rad, elev, kpet, snow, sma,
           veg, chi, lai, et_noah, pt, learned, n_inc, floor, deep_on, deep,
           a, t0, te, passes, yidx, tfield, pscale, out, annual):
    """Units ``0..len(cell)``: the cycle spin-up over ``[t0, te)`` (``passes`` times), then the
    run over ``[a, t1)``.  ``out`` (U, S, N_HIST + t1 - t0): surf, base [, the four parts
    [, the deep store's outflow and content]], the first N_HIST days the inflow before
    ``t0``; ``annual`` (U, 2, Y): surf + base by block
    year over the last two spin-up passes.  ``tfield`` (1, 1) or (cells, days): the degC added
    to the temperatures.  ``deep_on``: the learned step's deep store on ``deep`` (U, 2), its
    content brought to steady state at ``t0`` after a window spin-up (module docstring)."""
    U, S, L = out.shape
    T = L - N_HIST
    t1 = t0 + T
    one = tfield.size == 1
    for u in prange(U):
        c = cell[u]
        n = t1 - a
        pr, ta, tn, tx = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
        for t in range(n):
            pr[t] = prcp[c, a + t]
            ta[t] = tavg[c, a + t]
            if pt:
                tn[t] = tmin[c, a + t]
                tx[t] = tmax[c, a + t]
            if pscale != 1.0:
                pr[t] = pr[t] * pscale
            d = tfield[0, 0] if one else tfield[c, a + t]
            if d != 0.0:
                ta[t] = ta[t] + d
                tn[t] = tn[t] + d
                tx[t] = tx[t] + d
        snow_st = np.zeros(4)
        if learned:                             # the reference six, then the deep store
            sac_st = np.zeros(7)
            sac_st[:6] = DEFAULT_INIT_STATE
        else:
            sac_st = DEFAULT_INIT_STATE.copy()
        for k in range(passes + 1):
            s0 = t0 - a if k < passes else 0
            s1 = te - a if k < passes else n
            eff, _m, _w, snow_st, _i = _snow17_core(pr[s0:s1], ta[s0:s1], doy_i[a + s0:a + s1],
                                                   leap[a + s0:a + s1], elev[u], snow[u], snow_st)
            if pt:
                pet = kpet[u] * _pt_core(ta[s0:s1], tn[s0:s1], tx[s0:s1], doy_f[a + s0:a + s1],
                                         lat_rad[u], elev[u])
            elif learned:
                pet = kpet[u] * _hamon_core(ta[s0:s1], doy_f[a + s0:a + s1], lat_rad[u], 1.0)
            else:
                pet = _hamon_core(ta[s0:s1], doy_f[a + s0:a + s1], lat_rad[u], kpet[u])
            parts = np.empty((max(S, 6), s1 - s0)) if S > 2 else np.empty((0, 0))
            if learned:
                lai_d = np.empty(s1 - s0)
                for t in range(s1 - s0):
                    lai_d[t] = lai[u, doy_i[a + s0 + t] - 1] if et_noah else 0.0
                m = t0 - a if (deep_on and k == passes) else 0    # a window spin-up's days
                if m > 0:
                    sf1, bs1, _t, sac_st = sac_learned(pet[:m], eff[:m], sma[u], sac_st, et_noah,
                                                       veg[u], lai_d[:m], chi[u], n_inc, floor,
                                                       deep_on, deep[u], parts[:, :m])
                    fill = -np.expm1(m * np.log1p(-deep[u, 1]))   # 1 - (1 - kdeep)**m
                    if fill > 0.0:
                        sac_st[6] = sac_st[6] / fill
                    sf2, bs2, _t, sac_st = sac_learned(pet[m:], eff[m:], sma[u], sac_st, et_noah,
                                                       veg[u], lai_d[m:], chi[u], n_inc, floor,
                                                       deep_on, deep[u], parts[:, m:])
                    surf = np.concatenate((sf1, sf2))
                    base = np.concatenate((bs1, bs2))
                else:
                    surf, base, _t, sac_st = sac_learned(pet, eff, sma[u], sac_st, et_noah,
                                                         veg[u], lai_d, chi[u], n_inc, floor,
                                                         deep_on, deep[u], parts)
            else:
                surf, base, _t, sac_st = _sacsma_core(pet, eff, sma[u], sac_st)
            if k < passes:
                if k >= passes - 2:
                    for t in range(s1 - s0):
                        annual[u, k - passes + 2, yidx[t]] += surf[t] + base[t]
                if k == passes - 1:     # the routing history: the last days of the last pass
                    m = min(N_HIST, s1 - s0)
                    for t in range(m):
                        j = N_HIST - m + t
                        i = s1 - s0 - m + t
                        out[u, 0, j] = surf[i]
                        out[u, 1, j] = base[i]
                        if S > 2:
                            out[u, 2, j] = parts[0, i] + parts[1, i] + parts[2, i]
                            out[u, 3, j] = parts[3, i]
                            out[u, 4, j] = parts[4, i]
                            out[u, 5, j] = parts[5, i]
                        if S > 6:
                            out[u, 6, j] = parts[6, i]
                            out[u, 7, j] = parts[7, i]
                continue
            # the run: [a, t0) is the inflow history of a window spin-up, [t0, t1) the output
            for i in range(n):
                j = N_HIST + i - (t0 - a)
                if j < 0:
                    continue
                out[u, 0, j] = surf[i]
                out[u, 1, j] = base[i]
                if S > 2:
                    out[u, 2, j] = parts[0, i] + parts[1, i] + parts[2, i]
                    out[u, 3, j] = parts[3, i]
                    out[u, 4, j] = parts[4, i]
                    out[u, 5, j] = parts[5, i]
                if S > 6:
                    out[u, 6, j] = parts[6, i]
                    out[u, 7, j] = parts[7, i]


@njit(parallel=True, cache=False)
def _uh(flowlen, route):
    """(R, 2, N_TAPS): the normalized direct (hillslope x channel) and base (channel) UHs."""
    R = flowlen.shape[0]
    out = np.empty((R, 2, N_TAPS))
    delta = np.zeros(12)
    delta[0] = 1.0
    for r in prange(R):
        riv = _river_uh_nb(flowlen[r], route[r, 2], route[r, 3], 1 if flowlen[r] == 0.0 else 0)
        d = _convolve_full_nb(_hru_uh_direct_nb(route[r, 0], route[r, 1]), riv)
        b = _convolve_full_nb(delta, riv)
        out[r, 0] = d / d.sum()
        out[r, 1] = b / b.sum()
    return out


@njit(parallel=True, cache=False)
def _route(series, unit_of, uh, out):
    """Rows ``r`` of a block, unit ``unit_of[r]`` of ``series``: ``out[r, 0]`` the flow (one
    accumulator per day, the direct convolution then the base one), then as many of
    :data:`OUTPUTS` and the deep store's ``deep`` as ``out`` has room for."""
    R = unit_of.shape[0]
    T = out.shape[2]
    for r in prange(R):
        x = series[unit_of[r]]
        ud = uh[r, 0]
        ub = uh[r, 1]
        for t in range(T):
            acc = 0.0
            for j in range(N_TAPS):
                acc += ud[j] * x[0, N_HIST + t - j]
            for j in range(N_TAPS):
                acc += ub[j] * x[1, N_HIST + t - j]
            out[r, 0, t] = acc
        for k in range(1, out.shape[1]):     # fast, slow, quick, interflow, supplemental,
            w = ud if (k == 1 or k == 3 or k == 4) else ub    # primary, deep
            for t in range(T):
                acc = 0.0
                for j in range(N_TAPS):
                    acc += w[j] * x[k - 1, N_HIST + t - j]
                out[r, k, t] = acc


def simulate(field: Field, forcing, start, end, weights: np.ndarray, *, spinup: str = "cycle",
             years: int = 10, passes: int = 20, window_start=None, outputs=("flow",),
             temp_delta=0.0, precip_scale: float = 1.0, zero_nan: bool = False,
             block_bytes: float = 2e9) -> dict:
    """Run ``field`` over ``forcing`` (a :class:`sacsma.model.DomainForcing`, with ``tmin`` /
    ``tmax`` for the Priestley-Taylor PET) from ``start`` to ``end`` (dates, inclusive) and
    return ``{output: weights @ routed rows}``, each ``(K, T)`` mm/day, for the names in
    ``outputs`` (:data:`OUTPUTS`, ``"runoff"``: surf + base, not routed, and for a field with the
    deep store :data:`DEEP_OUTPUTS`), plus ``"bad"`` (R,), the rows whose flow went NaN, and
    ``"spinup"``, a line on the spin-up.  A row that goes NaN
    raises, unless ``zero_nan``: then it counts as zero on every day, so that a caller can
    renormalise by the weight of the rows left (``bad``).  ``temp_delta`` (degC on
    tavg / tmin / tmax: a number, or a field on the forcing's rows and days) and
    ``precip_scale`` perturb the forcing, the spin-up included."""
    ph = field.physics
    dates = pd.DatetimeIndex(forcing.dates)
    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    t1 = int(dates.searchsorted(pd.Timestamp(end))) + 1
    if t0 >= t1 or t1 > len(dates) or dates[t0] != pd.Timestamp(start):
        raise ValueError(f"{start} .. {end} is outside the forcing {dates[0].date()} .. "
                         f"{dates[-1].date()}")
    bad_out = [o for o in outputs if o not in (*OUTPUTS, *DEEP_OUTPUTS, "runoff")]
    if bad_out:
        raise ValueError(f"outputs {bad_out} (of {OUTPUTS}, {DEEP_OUTPUTS} and runoff)")
    diag = any(o in DEEP_OUTPUTS for o in outputs)
    parts = diag or any(o in OUTPUTS[3:] for o in outputs)
    if not ph.learned and (parts or ph.et != "sac"):
        raise ValueError("the reference model has the SAC ET and no runoff parts")
    if diag and ph.deep != "cascade":
        raise ValueError(f"outputs {DEEP_OUTPUTS} need a field with the deep store")
    a, te, P = t0, t0, 0
    if spinup == "cycle":
        te, P = block_end(dates, t0, years), passes
    elif spinup == "window":
        a = int(dates.searchsorted(pd.Timestamp(window_start)))
    elif spinup != "cold":
        raise ValueError(f"spinup {spinup!r}")
    pt = ph.pet == "priestley_taylor"
    if pt and (forcing.tmin is None or forcing.tmax is None):
        raise ValueError("the Priestley-Taylor PET needs the per-cell tmin / tmax")
    yidx = year_index(dates, t0, te) if P else np.zeros(0, np.int64)
    T, S = t1 - t0, 8 if diag else 6 if parts else 2
    W = np.asarray(weights, np.float64)
    K, R = W.shape
    routed_out = set(outputs) - {"runoff", "gwc"}
    names = (*OUTPUTS, "deep")          # the routed outputs, in _route's order
    # _route fills the first nout of them: route up to the last one asked for
    nout = max((names.index(o) + 1 for o in routed_out), default=0)
    res: dict = {o: np.zeros((K, T)) for o in names[:nout]}
    runoff = np.zeros((K, T)) if "runoff" in outputs else None
    gwc = np.zeros((K, T)) if "gwc" in outputs else None
    bad = np.zeros(R, bool)
    uh = _uh(field.flowlen, field.route)
    rows_of = np.argsort(field.unit, kind="stable")
    bounds = np.searchsorted(field.unit[rows_of], np.arange(len(field.cell) + 1))
    step = max(1, int(block_bytes // ((S * (N_HIST + T) + nout * T) * 8)))
    tmin = forcing.tmin if pt else forcing.prcp[:1]
    tmax = forcing.tmax if pt else forcing.prcp[:1]
    doy_i = np.asarray(forcing.doy, np.int64)
    tfield = np.ascontiguousarray(np.broadcast_to(np.asarray(temp_delta, np.float64), (1, 1))
                                  if np.ndim(temp_delta) == 0 else temp_delta, np.float64)
    if np.ndim(temp_delta) and tfield.shape != forcing.prcp.shape:
        raise ValueError(f"temp_delta {tfield.shape} is not on the forcing's rows and days "
                         f"{forcing.prcp.shape}")
    annual = np.zeros((len(field.cell), 2, int(yidx.max()) + 1 if P else 1))
    for u0 in range(0, len(field.cell), step):
        u1 = min(u0 + step, len(field.cell))
        out = np.zeros((u1 - u0, S, N_HIST + T))
        sl = slice(u0, u1)
        _units(forcing.prcp, forcing.tavg, tmin, tmax, doy_i.astype(np.float64), doy_i,
               np.asarray(forcing.is_leap, np.int64), field.cell[sl], field.lat_rad[sl],
               field.elev[sl], field.kpet[sl], field.snow[sl], field.sma[sl], field.veg[sl],
               field.chi[sl], field.lai[sl], ph.et == "noah_lite", pt, ph.learned, ph.n_inc,
               ph.fracp_floor, ph.learned and ph.deep == "cascade",
               field.deep[sl], a, t0, te, P, yidx, tfield, float(precip_scale), out, annual[sl])
        rows = rows_of[bounds[u0]:bounds[u1]]
        loc = field.unit[rows] - u0
        if runoff is not None:
            q = out[loc, 0, N_HIST:] + out[loc, 1, N_HIST:]
            failed = np.isnan(q).any(axis=1)
            bad[rows] |= failed
            q[failed] = 0.0
            runoff += W[:, rows] @ q
        if gwc is not None:
            g = out[loc, 7, N_HIST:]
            failed = np.isnan(g).any(axis=1)
            bad[rows] |= failed              # (the flow may not be routed)
            gwc += W[:, rows] @ np.where(failed[:, None], 0.0, g)
        if not nout:
            continue
        routed = np.empty((len(rows), nout, T))
        _route(out, loc, uh[rows], routed)
        failed = np.isnan(routed[:, 0]).any(axis=1)
        if failed.any():
            bad[rows] |= failed
            routed[failed] = 0.0
        for k, o in enumerate(res):
            res[o] += W[:, rows] @ routed[:, k]
    for o in names[:nout]:
        if o not in outputs:
            del res[o]
    if runoff is not None:
        res["runoff"] = runoff
    if gwc is not None:
        res["gwc"] = gwc
    if bad.any() and not zero_nan:
        raise FloatingPointError(f"{int(bad.sum())} of {R} rows went NaN")
    res["bad"] = bad
    res["spinup"] = "cold start" if spinup == "cold" else (
        f"window spin-up from {dates[a].date()}" if spinup == "window" else
        spinup_line(dates, t0, te, P, _change(W, field.unit, annual)))
    return res


def spinup_line(dates, t0: int, t1: int, passes: int, change: float) -> str:
    """One line on a cycle spin-up over ``dates[t0:t1]``."""
    return (f"cycle spin-up over {dates[t0].date()}..{dates[t1 - 1].date()}: {passes} passes "
            f"from the cold start, last-pass annual flow change {change:.1e}")


def _change(W, unit, annual) -> float:
    """The largest annual change of the last spin-up pass over the weighted sums, relative to
    their mean annual flow (at least 1 mm); unrouted surf + base."""
    q = np.nan_to_num(np.einsum("kr,ry->ky", W, annual[unit, 1]))
    p = np.nan_to_num(np.einsum("kr,ry->ky", W, annual[unit, 0]))
    d = np.abs(q - p).max(axis=1) / np.maximum(np.abs(q.mean(axis=1)), 1.0)
    return float(d.max()) if d.size else float("nan")
