"""Streamflow performance metrics (NSE, KGE, percent bias).

Used both for parity/evaluation and as GA calibration objectives.  KGE is
the pooled-GA objective used by Wi & Steinschneider.
"""

from __future__ import annotations

import numpy as np


def _align(sim: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    return sim[mask], obs[mask]


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency."""
    sim, obs = _align(sim, obs)
    if sim.size == 0:
        return np.nan
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((sim - obs) ** 2) / denom


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta efficiency (Gupta et al. 2009)."""
    sim, obs = _align(sim, obs)
    if sim.size == 0:
        return np.nan
    mu_s, mu_o = sim.mean(), obs.mean()
    sd_s, sd_o = sim.std(), obs.std()
    if mu_o == 0 or sd_o == 0 or sd_s == 0:
        return np.nan
    # Pearson r via reductions (no BLAS gemm; avoids np.corrcoef/np.cov).
    r = float(np.mean((sim - mu_s) * (obs - mu_o)) / (sd_s * sd_o))
    alpha = sd_s / sd_o
    beta = mu_s / mu_o
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def pearson(sim: np.ndarray, obs: np.ndarray) -> float:
    """Pearson correlation coefficient (BLAS-free; NaN pairs dropped).

    Computed via reductions (no ``np.corrcoef``/``np.cov`` gemm).  Returns NaN
    for fewer than 3 finite pairs or a zero-variance series.
    """
    sim, obs = _align(sim, obs)
    if sim.size < 3:
        return np.nan
    sd_s, sd_o = sim.std(), obs.std()
    if sd_s == 0 or sd_o == 0:
        return np.nan
    return float(np.mean((sim - sim.mean()) * (obs - obs.mean())) / (sd_s * sd_o))


def pbias(sim: np.ndarray, obs: np.ndarray) -> float:
    """Percent bias (%)."""
    sim, obs = _align(sim, obs)
    if sim.size == 0 or obs.sum() == 0:
        return np.nan
    return 100.0 * (sim.sum() - obs.sum()) / obs.sum()


def center_of_timing(dates, flow) -> float:
    """Flow-weighted center of timing (CT) on a **water-year (Oct–Sep)** basis.

    For each *complete* (12-month) water year, CT is the flow-weighted mean month
    **within** the water year — ``Oct=1, Nov=2, …, Sep=12`` (so WY2000 = Oct 1999 …
    Sep 2000); the value returned is the mean of those per-water-year CTs.  Larger CT =
    seasonal mass shifted **later** in the water year; the ``sim − ref`` difference is the
    seasonal timing bias in months (snowmelt earlier/later).  Expects monthly inputs.

    ``dates`` is any array-like of month-stamped dates (parallel to ``flow``); partial
    water years at the ends of the record are dropped, as are water years with non-positive
    total flow.  Returns NaN if no complete water year is available.
    """
    d = np.asarray(dates, dtype="datetime64[ns]").astype("datetime64[M]")
    q = np.asarray(flow, dtype=float)
    m = np.isfinite(q)
    d, q = d[m], q[m]
    if q.size == 0:
        return np.nan
    month = d.astype(int) % 12 + 1                 # calendar month 1..12
    wy = (d.astype("datetime64[Y]").astype(int) + 1970) + (month >= 10)  # Oct–Sep water year
    wm = (month - 10) % 12 + 1                      # position in the water year: Oct=1 … Sep=12
    cts = []
    for y in np.unique(wy):
        sel = wy == y
        if int(sel.sum()) < 12:                    # complete water years only
            continue
        tot = q[sel].sum()
        if tot > 0:
            cts.append(float((wm[sel] * q[sel]).sum() / tot))
    return float(np.mean(cts)) if cts else np.nan


def seasonal_mismatch(dates, sim, obs) -> float:
    """Seasonal-shape mismatch between two monthly hydrographs (a clearer single seasonal-bias
    number than center-of-timing).

    Each series is reduced to its 12-value mean-monthly regime (mean flow per calendar month)
    and normalised to sum to 1 (fraction of annual flow per month) -> ``p`` (sim), ``q`` (obs).
    The metric is the total-variation distance between those seasonal distributions::

        0.5 * Σ_m |p_m − q_m|   ==   1 − Σ_m min(p_m, q_m)

    bounded **[0, 1]** and read directly as the *fraction of annual flow delivered in the wrong
    month*: 0 = identical seasonality, 1 = no seasonal overlap.  It is **volume-independent**
    (a pure shape/timing error, orthogonal to :func:`pbias`).  Expects monthly inputs aligned
    to ``dates``; NaN if there is no overlapping finite data or a series has no positive flow.
    """
    d = np.asarray(dates, dtype="datetime64[ns]").astype("datetime64[M]")
    s = np.asarray(sim, dtype=float)
    o = np.asarray(obs, dtype=float)
    m = np.isfinite(s) & np.isfinite(o)
    if not m.any():
        return np.nan
    month = d[m].astype(int) % 12                    # 0..11 calendar-month bins
    s, o = s[m], o[m]
    ps = np.array([s[month == k].mean() if np.any(month == k) else np.nan for k in range(12)])
    qo = np.array([o[month == k].mean() if np.any(month == k) else np.nan for k in range(12)])
    keep = np.isfinite(ps) & np.isfinite(qo)
    ps, qo = ps[keep], qo[keep]
    if ps.sum() <= 0 or qo.sum() <= 0:
        return np.nan
    p, q = ps / ps.sum(), qo / qo.sum()
    return float(0.5 * np.abs(p - q).sum())
