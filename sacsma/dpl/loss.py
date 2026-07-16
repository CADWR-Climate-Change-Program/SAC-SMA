"""Training loss + selection metric.

Training loss (chunk-additive, MSE-family — NEVER chunk-local KGE, whose
mean/variance/correlation statistics are global to the record and biased/
unstable over one-year chunks):

    nnse:  per-basin variance-normalized squared error
           sum_t (sim - obs)^2 / var_cal(obs)   over finite-obs days,
           where var_cal(obs) is each basin's observed variance over the FULL
           calibration window, computed once (a fixed constant) — so summing
           chunk losses reproduces the per-basin NSE numerator/denominator
           exactly, and basins of very different wetness weigh comparably.
    mse:   plain squared error (config alternative; wet basins dominate).

Optional low-flow emphasis: ``+ lambda * (log(sim+eps) - log(obs+eps))^2``.

Optional variance matching: ``+ var_lambda * (std(sim)/std(obs) - 1)^2`` per
basin over the chunk's finite-obs days.  Squared error alone is variance-
damping — its optimum is ``alpha = r < 1`` (the classic NSE peak-flattening),
which the 2026-07-10 static run showed directly (mean cal alpha 0.88 vs the
GA's 1.08, costing ~0.1 KGE on the strong basins).  A chunk std over ~366
days is a stable statistic, unlike chunk-local correlation/mean ratios —
this is NOT chunked KGE.

Selection metric: pooled mean per-basin KGE over the full calibration record
(the GA-comparable exact objective), computed no-grad by the trainer.
"""

from __future__ import annotations

import torch

from ..metrics import kge as kge_numpy  # noqa: F401  (re-export for trainers)


def masked_basin_loss(
    sim: torch.Tensor,        # (B, T) basin flow
    obs: torch.Tensor,        # (B, T) observed, NaN where missing
    obs_var: torch.Tensor,    # (B,) obs variance over the FULL cal window
    *,
    kind: str = "nnse",
    log_lambda: float = 0.0,
    log_eps: float = 0.01,
    var_lambda: float = 0.0,
    bias_lambda: float = 0.0,
    weight: torch.Tensor | None = None,
    min_days: int = 90,
) -> torch.Tensor:
    """Mean over valid basins of the (normalized) squared error on this chunk.

    ``weight`` (B,), if given, reweights the per-basin mean (adaptive per-basin
    training weights) — renormalized by its own sum, so unit-mean weights leave
    the loss scale unchanged.  ``weight=None`` is byte-identical to no weighting.
    """
    finite = torch.isfinite(obs)
    n_fin = finite.sum(dim=1)                                   # (B,)
    n_safe = n_fin.clamp_min(1)
    obs_f = torch.where(finite, obs, torch.zeros_like(obs))
    sim_f = torch.where(finite, sim, torch.zeros_like(sim))

    se = (sim_f - obs_f) ** 2 * finite
    per_basin = se.sum(dim=1) / n_safe                          # mean SE per basin
    if kind == "nnse":
        per_basin = per_basin / obs_var.clamp_min(1e-12)
    elif kind != "mse":
        raise ValueError(f"loss kind {kind!r}")

    if log_lambda > 0.0:
        lse = (torch.log(sim_f.clamp_min(0.0) + log_eps)
               - torch.log(obs_f.clamp_min(0.0) + log_eps)) ** 2 * finite
        per_basin = per_basin + log_lambda * lse.sum(dim=1) / n_safe

    if var_lambda > 0.0:
        mo = (obs_f.sum(dim=1) / n_safe).unsqueeze(1)
        ms = (sim_f.sum(dim=1) / n_safe).unsqueeze(1)
        vo = ((obs_f - mo) ** 2 * finite).sum(dim=1) / n_safe
        vs = ((sim_f - ms) ** 2 * finite).sum(dim=1) / n_safe
        alpha = vs.clamp_min(1e-12).sqrt() / vo.clamp_min(1e-12).sqrt()
        per_basin = per_basin + var_lambda * (alpha - 1.0) ** 2

    if bias_lambda > 0.0:
        # KGE beta term: per-basin chunk mean-ratio (sim/obs).  Penalizes volume
        # bias directly (the over-evaporation the squared-error loss tolerates).
        mo_b = obs_f.sum(dim=1) / n_safe
        ms_b = sim_f.sum(dim=1) / n_safe
        beta = ms_b / mo_b.clamp_min(1e-12)
        per_basin = per_basin + bias_lambda * (beta - 1.0) ** 2

    # branch-free mean over valid basins (CUDA-graph capturable: no host sync);
    # no valid basins -> 0.0 with the graph still alive through per_basin.
    valid = (n_fin >= min_days).to(per_basin.dtype)
    if weight is not None:
        valid = valid * weight
    return (per_basin * valid).sum() / valid.sum().clamp_min(1.0)


def shape_pull_loss(
    monthly: torch.Tensor,      # (B, M) model basin monthly values (ET sums / SWE means)
    mu: torch.Tensor,           # (B, M) ensemble-mean NORMALIZED cycle target
    sigma: torch.Tensor,        # (B, M) ensemble std of the normalized cycle (floored)
    mask: torch.Tensor,         # (M,) 1.0 for valid (complete, in-cal) month slots
    basin_w: torch.Tensor | None = None,   # (B,) participation (SWE: snow mask)
    huber_k: float = 3.0,       # quadratic within k sigma (past deadband), linear beyond
    deadband: float = 1.0,      # NO force within this many ensemble sigma
) -> torch.Tensor:
    """Inverse-variance pull of the model's NORMALIZED seasonal cycle toward the
    multi-product consensus shape — level-blind by construction.

    The model's monthly values are divided by their own masked-month mean before
    the z-score, so scaling the flux by any factor leaves the term unchanged:
    only the seasonal PATTERN (the thing the products agree on to ~0.5 month) is
    constrained; the annual volume is left entirely to the streamflow loss (and,
    for ET, the envelope hinge below).  Self-weighting via sigma: hard where the
    normalized cycles agree (timing of rise/fall), gentle where they scatter
    (peak amplitude).  Branch-free + capture-safe (no host sync); ``mask`` zeroes
    padding / partial / out-of-cal slots on both the values and the normalizer.

    ``deadband`` makes the pull a hinge like the level term: a month whose
    normalized value sits within the products' own combined spread (cross-product
    + interannual, in sigma units) feels ZERO force — the pull never claims to
    know the shape better than the ensemble disagrees with itself.  Without it,
    the quadratic well fights the streamflow loss over every month forever: the
    2026-07-14/15 +ET runs peaked at cal KGE 0.748@ep12 then decayed (flow loss
    rising, shape term flat — pure budget burn under the shared grad clip)
    while the no-obs baseline climbed monotonically past 0.789.

    ``huber_k`` bounds the per-slot gradient: beyond k sigma (past the deadband)
    the penalty turns linear (Huber), capping |dL/dz| at 2k.  Without it, a
    far-off month's z^2 backward seed through the 366-day state recurrence can
    overflow f32 to inf -> the branch-gate multiplies turn inf into NaN (the
    2026-07-14 divergence: skipped 7/15 steps on nonfinite grads).
    """
    msum = mask.sum().clamp_min(1.0)
    xbar = (monthly * mask).sum(-1, keepdim=True) / msum          # (B, 1)
    xhat = monthly / xbar.clamp_min(1e-3)                         # normalized cycle
    z = (xhat - mu) / sigma
    az = torch.relu(z.abs() - deadband)                           # outside-spread part
    c = az.clamp_max(huber_k)
    per = c * (2.0 * az - c) * mask.unsqueeze(0)                  # (B, M) Huber(az)^2
    if basin_w is not None:
        per = per * basin_w.unsqueeze(-1)
        denom = (mask.sum() * basin_w.sum()).clamp_min(1.0)
    else:
        denom = (mask.sum() * monthly.shape[0]).clamp_min(1.0)
    return per.sum() / denom


def level_hinge_loss(
    monthly: torch.Tensor,      # (B, M) model basin monthly totals (mm/month)
    mask: torch.Tensor,         # (M,) valid month slots
    lo: torch.Tensor,           # (B,) product-envelope LOW total over masked months
    hi: torch.Tensor,           # (B,) product-envelope HIGH total over masked months
    rel_floor: float = 0.05,
    huber_k: float = 3.0,       # quadratic within k widths outside, linear beyond
) -> torch.Tensor:
    """One-per-chunk volume backstop: ZERO anywhere inside the product min-max
    envelope, quadratic outside, scaled by the envelope width.

    The honest level constraint for a wide (~40%) product bracket: basins whose
    total sits inside the bracket feel NO force (no fight with the streamflow
    volume); basins outside every product (the arid over-ET signature) are
    pushed back to the boundary — never to the ensemble mean.  ``rel_floor``
    keeps the width from degenerating where products coincide.  ``huber_k``
    bounds the gradient (same rationale as :func:`shape_pull_loss`).
    Capture-safe.
    """
    total = (monthly * mask).sum(-1)                              # (B,)
    width = (hi - lo).clamp_min(rel_floor * hi.clamp_min(1e-6)).clamp_min(1e-6)
    over = torch.relu(total - hi) + torch.relu(lo - total)
    z = over / width
    c = z.clamp_max(huber_k)
    return (c * (2.0 * z - c)).mean()


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
