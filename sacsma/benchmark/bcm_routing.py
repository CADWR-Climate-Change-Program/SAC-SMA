"""BCM monthly routing: the USGS post-processing that turns BCM runoff and recharge into flow.

BCM (the Basin Characterization Model) has no channel routing: its ``run`` and ``rch`` are
water generated in a month.  The USGS turns them into a monthly streamflow at a gauge with three
reservoirs calibrated per basin, in the workbook ``CalBasins_v8_DWR_FNF_PRISM19.xlsx`` (one
sheet per basin, BCM v8 forced with PRISM, fitted to CDEC full natural flow over WY2000-2010).
This module is an exact port of the sheets' formulas.  The parameters and the sheets' own
numbers are extracted by ``data/reference/bcm/bcm_routing_params.py`` into
``bcm_routing_params.csv`` and ``bcm_routing_check.csv``; :func:`check` reproduces the cached
flow of all 11 sheets from the second.  :func:`fit` fits the same equations to a measured flow
(the refit of the Kings sheet, and the benchmark's own routing on BCM Scenario 1).

Every quantity is a volume in m3 per month, as in the sheets.  The exponents act on volumes,
so the release fraction of a store depends on the basin's size; :func:`route_depth` turns
depths into volumes with the area the parameters were fitted with (``area_m2``).

The sheet's columns, month ``t`` (``J``..``R`` of rows 14-145), with ``H`` = rch and ``I`` = run
in m3, ``A`` the antecedent storage and the previous month of row 14 read as ``A``:

=================  ======================================================================
``surface_store``  ``J[t] = 0 if J[t-1] + I[t] <= 0 else J[t-1] + I[t] - K[t]``
``K`` (release)    ``K[t] = 0 if J[t-1] <= 0 else SurfaceScale * J[t-1] ** SurfaceExp``
``shallow_store``  ``L[t] = L[t-1] + H[t] - M[t] - O[t]``
``M`` (release)    ``M[t] = 0 if L[t-1] < 0 else ShallowScale * L[t-1] ** ShallowExp``
``deep_store``     ``N[t] = A + min(L[t], 0)``
``deep``           ``O[t] = DeepScale * N[t-1] ** DeepExp`` (``N13`` is blank: ``O`` = 0 in month 1)
``rchrun``         ``P[t] = K[t + peak_lag] + M[t + recession_lag] + O[t]``
``bcm_flow``       ``Q[t] = AquiferRch * P[t]``
``impairment``     ``R[t] = (1 - AquiferRch) * P[t]``
=================  ======================================================================

Read before using:

- ``surface`` and ``shallow`` in the output are the two lagged release terms of ``P``
  (``K[t + peak_lag]`` and ``M[t + recession_lag]``), so ``surface + shallow + deep = rchrun``.
- A lag of 1 reads the release computed from the store at the end of month ``t``, which
  already holds month ``t``'s input: causal, no delay.  A recession lag of 2 (sheets 6 and 8)
  reads the shallow store after month ``t + 1``'s recharge: a one-month lead.  A peak lag of 0
  (sheet 23) reads ``K[t]``, computed from the store at the end of month ``t - 1``: the surface
  release is delayed by a month.
- The deep store never fills: ``N`` is the antecedent storage less any overdraft of the
  shallow store.  While the shallow store is not negative the deep flow is the constant
  ``DeepScale * A ** DeepExp``, a fixed baseflow in m3 whatever the basin's area, drawn out of
  the shallow store; once that store is overdrawn its own release stops and the deep flow
  falls with ``N``.
- The trailing months: with a lag ``k > 0`` the last ``k`` months of ``rchrun`` need a release
  from a row past the end of the series.  A release depends only on the store at the end of
  the month before it (``K[t+1]`` on ``J[t]``, ``M[t+1]`` on ``L[t]``), so a release one row past
  the end is fully determined; one two rows past the end needs the next month's input.
  ``tail="nan"`` (the default) computes every release the formulas determine and leaves NaN
  only where an input past the end is needed: no month is lost with lags of 0 or 1, the last
  month with a recession lag of 2 (sheets 6 and 8).  ``tail="zero"`` is the workbook, whose
  ``OFFSET`` reads the blank rows below the table as 0 (so its last month or two lack the
  surface and shallow terms), and is what :func:`check` uses.
- A NaN input propagates to every later month of its store (and so of the flow).  Excel reads a
  blank input as 0; the port does not.  A negative deep store (the shallow store below
  minus the antecedent storage) raises a fractional power of a negative number, ``#NUM!`` in the
  workbook, NaN here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths

PARAMS_FILE = "bcm_routing_params.csv"
CHECK_FILE = "bcm_routing_check.csv"

#: the output columns of :func:`route`
COLUMNS = ("surface_store", "shallow_store", "deep_store",
           "surface", "shallow", "deep", "rchrun", "bcm_flow", "impairment")


def load_params(data_dir: str | Path = "data") -> pd.DataFrame:
    """The routing parameters of the 11 calibrated sheets, indexed by ``sheet``.

    Columns: ``basin_no, name, station_id, measured_station, cdec_site, area_m2``, the seven
    calibrated parameters in the sheet's order (``K2``..``K8``: ``surface_scale, surface_exp,
    shallow_scale, shallow_exp, deep_scale, deep_exp, aquifer_rch``),
    ``antecedent_m3, peak_lag, recession_lag, cal_start, cal_end,
    stat_start, stat_end`` and the sheet's own ``r2, nse, pbias``."""
    return pd.read_csv(paths.bcm(data_dir, name=PARAMS_FILE), index_col="sheet")


def params_for(site: str, data_dir: str | Path = "data") -> pd.Series:
    """The parameter row applied at a CDEC site (``cdec_site``, e.g. ``"ORO"``): the refit where
    the site has one (``source`` ``refit``), else its workbook sheet."""
    p = load_params(data_dir)
    hit = p[p["cdec_site"] == site]
    if (hit["source"] == "refit").any():
        hit = hit[hit["source"] == "refit"]
    if len(hit) != 1:
        raise KeyError(f"no BCM routing sheet for site {site!r} (sites: {sorted(p['cdec_site'])})")
    return hit.iloc[0]


def to_m3(depth_mm, area_m2: float):
    """Depth in mm over ``area_m2`` to volume in m3, as the sheets do (``mm / 1000 * area``)."""
    return depth_mm / 1000 * area_m2


def _pow(x: float, e: float) -> float:
    # a fractional power of a negative number is #NUM! in the workbook
    return float("nan") if x < 0 else x ** e


def route(run_m3, rch_m3, params, tail: str = "nan") -> pd.DataFrame:
    """Route monthly BCM runoff and recharge volumes (m3) through one sheet's reservoirs.

    ``run_m3`` and ``rch_m3`` are equal-length monthly series (a Series keeps its index);
    ``params`` is a row of :func:`load_params` (or any mapping with the same keys).  Returns a
    DataFrame of :data:`COLUMNS` in m3, one row per month.  ``tail`` is ``"nan"`` or ``"zero"``
    (see the module docstring)."""
    if tail not in ("nan", "zero"):
        raise ValueError(f"tail must be 'nan' or 'zero', not {tail!r}")
    index = run_m3.index if isinstance(run_m3, pd.Series) else None
    run = np.asarray(run_m3, dtype=float)
    rch = np.asarray(rch_m3, dtype=float)
    if run.shape != rch.shape or run.ndim != 1:
        raise ValueError("run_m3 and rch_m3 must be 1-D and the same length")
    s_scale, s_exp = float(params["surface_scale"]), float(params["surface_exp"])
    h_scale, h_exp = float(params["shallow_scale"]), float(params["shallow_exp"])
    d_scale, d_exp = float(params["deep_scale"]), float(params["deep_exp"])
    aq = float(params["aquifer_rch"])
    a0 = float(params["antecedent_m3"])
    peak, rec = int(params["peak_lag"]), int(params["recession_lag"])

    if peak < 0 or rec < 0:
        raise ValueError("a negative lag reads before the first row; the sheets have none")

    n = len(run)
    ext = max(peak, rec)          # rows past the end that OFFSET reads
    run = np.concatenate([run, np.full(ext, np.nan)])
    rch = np.concatenate([rch, np.full(ext, np.nan)])
    # the sheet's columns J (surface store), K (its release), L (shallow store), M (its release),
    # N (deep store), O (deep flow)
    j_col, k_col, l_col, m_col, n_col, o_col = (np.empty(n + ext) for _ in range(6))
    j_prev = l_prev = a0          # row 14 reads the antecedent storage as the previous month
    n_prev = 0.0                  # N13 is blank: the deep flow of the first month is 0
    for t in range(n + ext):      # past the end the input is NaN, so only what it needs is NaN
        k = 0.0 if j_prev <= 0 else s_scale * _pow(j_prev, s_exp)
        j = 0.0 if j_prev + run[t] <= 0 else j_prev + run[t] - k
        m = 0.0 if l_prev < 0 else h_scale * _pow(l_prev, h_exp)
        o = d_scale * _pow(n_prev, d_exp)
        lt = l_prev + rch[t] - m - o
        nt = a0 + (0.0 if lt >= 0 else lt)          # NaN stays NaN
        j_col[t], k_col[t], l_col[t], m_col[t], n_col[t], o_col[t] = j, k, lt, m, nt, o
        j_prev, l_prev, n_prev = j, lt, nt
    if tail == "zero":            # the workbook: the rows below the table are blank, read as 0
        k_col[n:] = 0.0
        m_col[n:] = 0.0

    surface = k_col[peak:peak + n]
    shallow = m_col[rec:rec + n]
    deep = o_col[:n]
    rchrun = surface + shallow + deep
    out = pd.DataFrame({
        "surface_store": j_col[:n], "shallow_store": l_col[:n], "deep_store": n_col[:n],
        "surface": surface, "shallow": shallow, "deep": deep,
        "rchrun": rchrun, "bcm_flow": rchrun * aq, "impairment": (1 - aq) * rchrun,
    }, index=index)
    out.index.name = index.name if index is not None else "month"
    return out


def route_depth(run_mm, rch_mm, params, area_m2: float | None = None,
                tail: str = "nan") -> pd.DataFrame:
    """:func:`route` from depths in mm per month.

    ``area_m2`` defaults to the area the sheet was fitted with (``params["area_m2"]``), which
    keeps the fitted volume scale: the exponents, the antecedent storage and the constant deep
    flow are all in m3 of that basin.  The flow that comes out is a volume at the sheet's gauge
    (``AquiferRch`` was fitted to the measured volume there); divide it by the gauge's own
    drainage area for a depth."""
    area = float(params["area_m2"]) if area_m2 is None else float(area_m2)
    return route(to_m3(run_mm, area), to_m3(rch_mm, area), params, tail=tail)


def sheet_scores(measured, flow) -> dict[str, float]:
    """The workbook's statistics: ``r2`` (Excel ``RSQ``), ``nse`` (the sheet's ``NSS``,
    ``1 - mean((Q - E)^2) / VAR(E)`` with the sample variance) and ``pbias``
    (``100 * sum(E - Q) / sum(E)``, percent).  Months where either series is NaN are left out."""
    e = np.asarray(measured, dtype=float)
    q = np.asarray(flow, dtype=float)
    ok = np.isfinite(e) & np.isfinite(q)
    e, q = e[ok], q[ok]
    return {
        "r2": float(np.corrcoef(e, q)[0, 1] ** 2),
        "nse": float(1 - np.mean((q - e) ** 2) / np.var(e, ddof=1)),
        "pbias": float(100 * np.sum(e - q) / np.sum(e)),
    }


#: the free parameters of a refit and their bounds, around the range of the ten sheets that
#: are wired to their own basin (the antecedent storage on a log10 scale); the three scales stay
#: at 1, as on every sheet
FIT_BOUNDS = {"surface_exp": (0.97, 0.99999), "shallow_exp": (0.80, 0.9999),
              "deep_exp": (0.50, 0.95), "aquifer_rch": (0.50, 1.50),
              "log10_antecedent_m3": (6.0, 9.5)}
#: the (peak, recession) lags a refit tries: the causal ones only (a recession lag of 2 reads
#: next month's recharge)
FIT_LAGS = ((0, 1), (1, 1))
#: the sheets' volume tolerance, as a refit applies it: |PBIAS| within 1 % (the sheets reach 1.13)
FIT_PBIAS = 1.0


def fit(run_m3, rch_m3, measured_m3, window=None, seed: int = 0,
        objective: str = "nse") -> dict:
    """Fit a sheet's parameters to a measured monthly flow, the way the sheets were fitted.

    Maximises the sheet's NSE (:func:`sheet_scores`; ``objective="kge"``: the Kling-Gupta
    efficiency instead) over ``window`` (a boolean mask of the months scored; default every
    month) with ``|PBIAS|`` held within :data:`FIT_PBIAS`, over the
    :data:`FIT_BOUNDS` and each pair of :data:`FIT_LAGS`, with the workbook's own boundary rules
    (``tail="zero"``, the antecedent storage as the first month's previous store).  Differential
    evolution with a fixed seed, so a rerun gives the same parameters.  Returns the parameter
    row (the keys of :func:`load_params` that :func:`route` reads) with ``r2, nse, pbias``."""
    from scipy.optimize import differential_evolution

    from ..metrics import kge

    if objective not in ("nse", "kge"):
        raise ValueError(f"objective must be 'nse' or 'kge', not {objective!r}")
    run = np.asarray(run_m3, dtype=float)
    rch = np.asarray(rch_m3, dtype=float)
    e = np.asarray(measured_m3, dtype=float)
    win = np.ones(len(e), bool) if window is None else np.asarray(window, bool)
    names = list(FIT_BOUNDS)

    def row(x, peak, rec):
        p = dict(zip(names, x, strict=True))
        a = 10.0 ** p.pop("log10_antecedent_m3")
        return {"surface_scale": 1.0, "shallow_scale": 1.0, "deep_scale": 1.0, **p,
                "antecedent_m3": a, "peak_lag": peak, "recession_lag": rec}

    def loss(x, peak, rec):
        q = route(run, rch, row(x, peak, rec), tail="zero")["bcm_flow"].to_numpy()[win]
        if not np.all(np.isfinite(q)):
            return 1e6
        s = sheet_scores(e[win], q)
        score = s["nse"] if objective == "nse" else kge(q, e[win])
        return (1.0 - score) + 0.1 * max(0.0, abs(s["pbias"]) - FIT_PBIAS) ** 2

    best = None
    for peak, rec in FIT_LAGS:
        res = differential_evolution(loss, [FIT_BOUNDS[n] for n in names], args=(peak, rec),
                                     seed=seed, tol=1e-10, maxiter=2000, polish=True)
        if best is None or res.fun < best[0]:
            best = (res.fun, row(res.x, peak, rec))
    p = best[1]
    q = route(run, rch, p, tail="zero")["bcm_flow"].to_numpy()[win]
    return {**p, **sheet_scores(e[win], q), "kge": float(kge(q, e[win]))}


def check(data_dir: str | Path = "data") -> pd.DataFrame:
    """Run every sheet of the check table through :func:`route` (``tail="zero"``, as the
    workbook) and compare with the sheets' cached values.

    One row per sheet: ``max_rel_err_q`` and ``max_rel_err_p`` (largest ``|port - sheet| /
    |sheet|`` over the 132 months), and the statistics recomputed by the port over the sheet's
    own window (``r2, nse, pbias``) beside the sheet's (``r2_sheet, nse_sheet,
    pbias_sheet``)."""
    params = load_params(data_dir)
    tab = pd.read_csv(paths.bcm(data_dir, name=CHECK_FILE))
    rows = []
    for sheet, g in tab.groupby("sheet", sort=False):
        p = params.loc[sheet]
        g = g.sort_values("month")
        if not np.allclose(g["area_m2"], p["area_m2"], rtol=0, atol=0):
            raise ValueError(f"sheet {sheet}: the check table's area is not the parameter table's")
        r = route(to_m3(g["run_mm"].to_numpy(), p["area_m2"]),
                  to_m3(g["rch_mm"].to_numpy(), p["area_m2"]), p, tail="zero")
        q, q_sheet = r["bcm_flow"].to_numpy(), g["q_m3"].to_numpy()
        p_port, p_sheet = r["rchrun"].to_numpy(), g["p_m3"].to_numpy()
        q_err = np.abs(q - q_sheet) / np.abs(q_sheet)
        p_err = np.abs(p_port - p_sheet) / np.abs(p_sheet)
        win = (g["month"] >= p["stat_start"]).to_numpy() & (g["month"] <= p["stat_end"]).to_numpy()
        s = sheet_scores(g["measured_m3"].to_numpy()[win], r["bcm_flow"].to_numpy()[win])
        rows.append({"sheet": sheet, "cdec_site": p["cdec_site"], "months": len(g),
                     "max_rel_err_q": float(np.max(q_err)), "max_rel_err_p": float(np.max(p_err)),
                     "r2": s["r2"], "r2_sheet": p["r2"], "nse": s["nse"], "nse_sheet": p["nse"],
                     "pbias": s["pbias"], "pbias_sheet": p["pbias"]})
    return pd.DataFrame(rows).set_index("sheet")
