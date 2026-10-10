"""Training loss + selection metric.

Training loss (chunk-additive, MSE-family — NEVER chunk-local KGE, whose
mean/variance/correlation statistics are global to the record and biased/
unstable over one-year chunks): the per-basin variance-normalized squared error
(NNSE)

    sum_t (sim - obs)^2 / var_cal(obs)   over finite-obs days,

where var_cal(obs) is each basin's observed variance over the FULL calibration
window, computed once (a fixed constant) — so summing chunk losses reproduces
the per-basin NSE numerator/denominator exactly, and basins of very different
wetness weigh comparably.

Optional low-flow emphasis: ``+ lambda * (log(sim+eps) - log(obs+eps))^2``.

Optional variance matching: ``+ var_lambda * rho((std(sim) - std(obs)) /
sqrt(obs_var))`` per basin over the chunk's finite-obs days — the std difference
over the NNSE's own record normalizer, so a year weighs by its share of the
record variance as in the NNSE — with ``rho(d) = d^2`` up to ``|d| =
var_huber_cap`` and linear beyond (Huber; ``var_huber_cap <= 0`` keeps ``d^2``
throughout), on basin-chunks with at least ``shape_min_days`` valid days only.
Squared error alone is variance-damping — its optimum is ``alpha = r < 1`` (the
classic NSE peak-flattening), which the 2026-07-10 static run showed directly
(mean cal alpha 0.88 vs the GA's 1.08, costing ~0.1 KGE on the strong basins).  A
chunk std over ~366 days is a stable statistic, unlike chunk-local
correlation/mean ratios — this is NOT chunked KGE.

Every basin-chunk term is weighted by its valid days / 365, so a partial chunk
counts by its length.

Selection metric: pooled mean per-basin KGE over the full calibration record
(the GA-comparable exact objective), computed no-grad by the trainer.
"""

from __future__ import annotations

import torch


def masked_basin_loss(
    sim: torch.Tensor,        # (B, T) basin flow
    obs: torch.Tensor,        # (B, T) observed, NaN where missing
    obs_var: torch.Tensor,    # (B,) obs variance over the FULL cal window
    *,
    log_lambda: float = 0.0,
    log_eps: float = 0.01,
    var_lambda: float = 0.0,
    var_huber_cap: float = 1.0,
    shape_min_days: int = 300,
    weight: torch.Tensor | None = None,
    weight_total: float | None = None,
    min_days: int = 90,
) -> torch.Tensor:
    """Mean over valid basins of the normalized squared error on this chunk.

    ``weight`` (B,), if given, reweights the per-basin mean (the family weights)
    — renormalized by its own sum, so unit-mean weights leave the loss scale
    unchanged.  ``weight=None`` is byte-identical to no weighting.
    ``weight_total``, if given, replaces that renormalization: the weighted sum
    over the valid basins is divided by this FIXED total (the weight of every
    basin of the run, valid in this chunk or not — ``mt_share_norm="all"``),
    so a chunk's loss scales with the weight it holds.  ``None`` = unchanged.
    The variance term (the std difference over the record std) scores the
    basin-chunks with ``shape_min_days`` valid days, ``var_huber_cap`` the ``|d|``
    beyond which it grows linearly (``<= 0``: quadratic throughout); each basin's
    chunk terms are scaled by its valid days / 365.
    """
    finite = torch.isfinite(obs)
    n_fin = finite.sum(dim=1)                                   # (B,)
    n_safe = n_fin.clamp_min(1)
    obs_f = torch.where(finite, obs, torch.zeros_like(obs))
    sim_f = torch.where(finite, sim, torch.zeros_like(sim))

    se = (sim_f - obs_f) ** 2 * finite
    per_basin = se.sum(dim=1) / n_safe                          # mean SE per basin
    per_basin = per_basin / obs_var.clamp_min(1e-12)

    if log_lambda > 0.0:
        lse = (torch.log(sim_f.clamp_min(0.0) + log_eps)
               - torch.log(obs_f.clamp_min(0.0) + log_eps)) ** 2 * finite
        per_basin = per_basin + log_lambda * lse.sum(dim=1) / n_safe

    if var_lambda > 0.0:
        mo = (obs_f.sum(dim=1) / n_safe).unsqueeze(1)
        ms = (sim_f.sum(dim=1) / n_safe).unsqueeze(1)
        vo = ((obs_f - mo) ** 2 * finite).sum(dim=1) / n_safe
        vs = ((sim_f - ms) ** 2 * finite).sum(dim=1) / n_safe
        # the std difference over the record std: v_y * (alpha - 1)^2 with v_y the
        # chunk's share of the record variance, so the term weighs a year as the
        # NNSE does (W4, red team 2).  The denominator is a fixed constant, so a
        # near-flat chunk (an ephemeral gauge's zero-flow season) cannot blow it up
        # (its gradient norm stays <= 2 |d| / (sqrt(obs_var) sqrt(n))); whole water
        # years only (the 92-day envelope tail and sparse years drop out).  The
        # penalty is Huber-capped (quadratic to |d| = var_huber_cap, linear beyond)
        # so no chunk contributes unboundedly.  Branch-free on the tensors; the cap
        # is a Python-level choice (a torch.where against an infinite cap would put
        # NaN through the unselected branch).
        has_var = (n_fin >= shape_min_days).to(per_basin.dtype)
        d = ((vs.clamp_min(1e-12).sqrt() - vo.clamp_min(1e-12).sqrt())
             / obs_var.clamp_min(1e-12).sqrt())
        if var_huber_cap > 0.0:
            c = float(var_huber_cap)
            pen = torch.where(d.abs() <= c, d * d, 2.0 * c * d.abs() - c * c)
        else:
            pen = d * d
        per_basin = per_basin + var_lambda * has_var * pen

    # each basin-chunk by its valid days: a full year counts 1, the 92-day tail ~1/4
    per_basin = per_basin * (n_fin.to(per_basin.dtype) / 365.0)

    # branch-free mean over valid basins (CUDA-graph capturable: no host sync);
    # no valid basins -> 0.0 with the graph still alive through per_basin.
    valid = (n_fin >= min_days).to(per_basin.dtype)
    if weight is not None:
        valid = valid * weight
    if weight_total is not None:
        return (per_basin * valid).sum() / float(weight_total)
    return (per_basin * valid).sum() / valid.sum().clamp_min(1.0)


def kge_torch(sim: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Differentiable KGE over finite-obs pairs, per basin: (B, T) -> (B,).

    Mirror of ``sacsma.metrics.kge`` (Gupta 2009).  Used ONLY on full records
    (model selection / diagnostics) — never as a chunk loss.
    """
    finite = torch.isfinite(obs)
    n = finite.sum(dim=1).clamp_min(1)
    o = torch.where(finite, obs, torch.zeros_like(obs))
    s = torch.where(finite, sim, torch.zeros_like(sim))
    mo = o.sum(dim=1) / n
    ms = s.sum(dim=1) / n
    do = (o - mo.unsqueeze(1)) * finite
    ds = (s - ms.unsqueeze(1)) * finite
    so = torch.sqrt((do ** 2).sum(dim=1) / n)
    ss = torch.sqrt((ds ** 2).sum(dim=1) / n)
    r = (do * ds).sum(dim=1) / n / (so * ss).clamp_min(1e-12)
    alpha = ss / so.clamp_min(1e-12)
    beta = ms / mo.clamp_min(1e-12)
    return 1.0 - torch.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
