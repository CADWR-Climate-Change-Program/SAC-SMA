"""Per-calendar-month quantile mapping (QMAP) — faithful CalSim numerics port.

Direct port of CalSim's rim-inflow quantile mapping
``utils/quantile_mapping.qmap_single`` (the ``calsim3-stochastic-input-generation``
repo): within the training range each month is mapped through an **empirical CDF**
(Weibull plotting positions ``rank/(n+1)``); beyond it a **gamma-distribution tail**
(method-of-moments ``alpha=(mean/sd)**2``, ``beta=var/mean``) extrapolates, with the
non-exceedance probability capped to ``[0.0001, 0.9999]``.

Used by :mod:`sacsma.calsim.compare` to bias-correct each CalSim sub-arc toward the CalSim3
``INFLOW`` distribution before the basin mass-balance.  ``scipy`` is required (it ships
in the ``gis`` optional-dependency group, which the CalSim cross-compare already needs);
this module is imported **lazily** inside ``compare`` so the faithful forward-run path
never needs scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gamma

#: seed for the (only) stochastic step — tie-breaking among equal basis values —
#: so QMAP output is deterministic run-to-run (matches the CalSim reference's default).
QMAP_SEED = 42


def _std(x):
    # sample standard deviation (ddof=1), matching the CalSim reference (R's sd).
    return np.std(x, ddof=1)


def qmap_single(basis_sim, basis_hist, target, allow_negative=False):
    """Quantile-map ``basis_sim`` from the ``basis_hist`` distribution onto ``target``.

    Faithful port of CalSim ``qmap_single``.  Each input is a DataFrame with columns
    ``year, month, value``.  For every row of ``basis_sim`` the value is converted to a
    non-exceedance probability through ``basis_hist``'s monthly distribution (empirical
    within range, gamma tail beyond), then inverted through ``target``'s monthly
    distribution.  Returns ``basis_sim[[year, month]]`` plus ``quantile_mapped_value``.

    Within the empirical range the basis/target mappings are plain piecewise-linear
    interpolations, so ``np.interp`` is used directly (identical numerics to the reference's
    ``interp1d``, which was only ever evaluated inside the in-range branch — extrapolation
    never triggers — but far cheaper than rebuilding/calling a scipy interpolator per step).
    """
    rng = np.random.default_rng(QMAP_SEED)
    n = len(basis_sim)
    out_vals = np.full(n, np.nan)

    # per-month precompute: sorted basis/target samples, Weibull plotting positions,
    # gamma (alpha, beta) params, the gamma CDF over the basis support, and whether the
    # target month is degenerate (a single distinct value).
    parhat_basis = np.full((12, 2), np.nan)
    parhat_target = np.full((12, 2), np.nan)
    basis_sorted: list = [None] * 12
    target_sorted: list = [None] * 12
    eprob_basis: list = [None] * 12
    eprob_target: list = [None] * 12
    tprob_basis: list = [None] * 12
    target_degenerate = np.zeros(12, dtype=bool)   # precomputed once (was np.unique per step)

    for imon in range(12):
        month = imon + 1
        bm = np.sort(basis_hist.loc[basis_hist["month"] == month, "value"].to_numpy(float))
        tm = np.sort(target.loc[target["month"] == month, "value"].to_numpy(float))
        basis_sorted[imon] = bm
        target_sorted[imon] = tm
        eprob_basis[imon] = np.arange(1, len(bm) + 1) / (1 + len(bm))
        eprob_target[imon] = np.arange(1, len(tm) + 1) / (1 + len(tm))
        target_degenerate[imon] = len(tm) > 0 and len(np.unique(tm)) == 1

        if len(bm) > 1 and np.mean(bm) > 0:
            parhat_basis[imon] = [(np.mean(bm) / _std(bm)) ** 2, np.var(bm, ddof=1) / np.mean(bm)]
        if len(tm) > 1 and np.mean(tm) > 0:
            parhat_target[imon] = [(np.mean(tm) / _std(tm)) ** 2, np.var(tm, ddof=1) / np.mean(tm)]

        try:
            tprob_basis[imon] = gamma.cdf(np.unique(bm), parhat_basis[imon, 0],
                                          scale=parhat_basis[imon, 1])
        except Exception:
            tprob_basis[imon] = np.full(len(np.unique(bm)), np.nan)

    months = basis_sim["month"].to_numpy()
    values = basis_sim["value"].to_numpy(float)

    for j in range(n):
        selval = values[j]
        if not np.isfinite(selval):
            continue
        m = int(months[j]) - 1
        bm, tm = basis_sorted[m], target_sorted[m]
        ebas, etgt = eprob_basis[m], eprob_target[m]

        if len(tm) == 0 or len(bm) == 0:
            continue
        if target_degenerate[m]:              # degenerate target -> that single value
            out_vals[j] = tm[0]
            continue

        # 1) value -> non-exceedance probability through the BASIS distribution
        if bm[0] <= selval <= bm[-1]:         # within range -> empirical (piecewise linear)
            eq = np.where(bm == selval)[0]
            if len(eq) > 1:                   # tie -> random plotting position (seeded)
                nonexprob = ebas[rng.choice(eq)]
            elif len(bm) >= 2:
                nonexprob = float(np.interp(selval, bm, ebas))
            else:
                nonexprob = np.nan
        elif selval > bm[-1]:                 # upper tail -> gamma CDF delta
            cdf_sel = gamma.cdf(selval, parhat_basis[m, 0], scale=parhat_basis[m, 1])
            cdf_max = tprob_basis[m][-1] if len(tprob_basis[m]) else 0.0
            nonexprob = min(ebas[-1] + (cdf_sel - cdf_max), 0.9999)
        else:                                 # lower tail -> gamma CDF delta
            cdf_sel = gamma.cdf(selval, parhat_basis[m, 0], scale=parhat_basis[m, 1])
            cdf_min = tprob_basis[m][0] if len(tprob_basis[m]) else 0.0
            nonexprob = max(ebas[0] - (cdf_min - cdf_sel), 0.0001)

        if not np.isfinite(nonexprob):
            continue

        # 2) probability -> value through the TARGET distribution
        if etgt[0] <= nonexprob <= etgt[-1] and len(tm) >= 2:   # within range -> empirical
            out_vals[j] = float(np.interp(nonexprob, etgt, tm))
        elif not np.isnan(parhat_target[m, 0]):       # gamma quantile beyond empirical range
            if nonexprob > etgt[-1]:
                d = (gamma.ppf(nonexprob, parhat_target[m, 0], scale=parhat_target[m, 1])
                     - gamma.ppf(etgt[-1], parhat_target[m, 0], scale=parhat_target[m, 1]))
                out_vals[j] = tm[-1] + d
            elif nonexprob < etgt[0]:
                d = (gamma.ppf(etgt[0], parhat_target[m, 0], scale=parhat_target[m, 1])
                     - gamma.ppf(nonexprob, parhat_target[m, 0], scale=parhat_target[m, 1]))
                out_vals[j] = tm[0] - d
        elif not np.isnan(parhat_basis[m, 0]):        # fallback: basis gamma, multiplicative
            if nonexprob > etgt[-1]:
                d = (gamma.ppf(nonexprob, parhat_basis[m, 0], scale=parhat_basis[m, 1])
                     - gamma.ppf(etgt[-1], parhat_basis[m, 0], scale=parhat_basis[m, 1]))
                out_vals[j] = (1 + d / bm[-1]) * tm[-1]
            elif nonexprob < etgt[0]:
                d = (gamma.ppf(etgt[0], parhat_basis[m, 0], scale=parhat_basis[m, 1])
                     - gamma.ppf(nonexprob, parhat_basis[m, 0], scale=parhat_basis[m, 1]))
                if tm[0] >= 0:
                    out_vals[j] = (1 - d / bm[0]) * tm[0]
                else:
                    out_vals[j] = (1 + d / bm[0]) * tm[0]

    if not allow_negative:
        out_vals = np.maximum(0.0, out_vals)
    res = basis_sim[["year", "month"]].copy()
    res["quantile_mapped_value"] = out_vals
    return res


def qmap_series(est_full, est_train, ref_train, *, allow_negative=False):
    """Quantile-map a monthly estimate Series toward a reference, per calendar month.

    ``est_full``  — the estimate values to correct (the **source**; mapped one-for-one).
    ``est_train`` — the estimate over the training period (source CDF).
    ``ref_train`` — the reference over the training period (target CDF).

    All three are pandas Series with a monthly ``DatetimeIndex``.  Returns a Series
    aligned to ``est_full.index`` (NaN preserved where ``est_full`` is NaN).  If either
    training distribution has <2 valid points the estimate is returned unchanged.
    """
    def _frame(s):
        s = s.dropna()
        idx = pd.to_datetime(s.index)
        return pd.DataFrame({"year": idx.year, "month": idx.month, "value": s.to_numpy(float)})

    bh, tg = _frame(est_train), _frame(ref_train)
    if len(bh) < 2 or len(tg) < 2:
        return est_full.copy()

    idx = pd.to_datetime(est_full.index)
    bs = pd.DataFrame({"year": idx.year, "month": idx.month, "value": est_full.to_numpy(float)})
    out = qmap_single(bs, bh, tg, allow_negative=allow_negative)
    res = pd.Series(out["quantile_mapped_value"].to_numpy(), index=est_full.index)
    res[est_full.isna()] = np.nan
    return res
