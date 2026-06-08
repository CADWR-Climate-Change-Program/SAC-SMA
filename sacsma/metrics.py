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
