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

Optional variance matching: ``+ var_lambda * rho(std(sim)/std(obs) - 1)`` per
basin over the chunk's finite-obs days, with ``rho(d) = d^2`` up to
``|d| = var_huber_cap`` and linear beyond (Huber; ``var_huber_cap <= 0`` keeps
``d^2`` throughout), skipped for a basin-chunk whose observed variance is under
``var_gate_frac`` of the basin's record variance (defaults 1.0 and 1e-3).  Squared
error alone is variance-damping — its optimum is ``alpha = r < 1`` (the classic NSE peak-flattening),
which the 2026-07-10 static run showed directly (mean cal alpha 0.88 vs the
GA's 1.08, costing ~0.1 KGE on the strong basins).  A chunk std over ~366
days is a stable statistic, unlike chunk-local correlation/mean ratios —
this is NOT chunked KGE.

Optional hydrograph-shape terms (``timing_lambda``, ``peak_lambda``; daily rows
only), scored on basin-chunks with at least ``shape_min_days`` valid days and
scaled by per-basin RECORD constants computed once from the training obs
(:func:`record_references`), so drought years and short chunks cannot dominate
them and a zero-flow year stays bounded: timing = the summer RECESSION shape, the
mean squared difference of the normalized cumulative flow over 1 Jul - 30 Sep
(volume-blind, no pull on winter or on the flood peaks); peak = the mean of the
chunk's top ``peak_frac`` valid days, sim minus obs, over the record mean of that
statistic, Huber-capped (flood years carry it).

Selection metric: pooled mean per-basin KGE over the full calibration record
(the GA-comparable exact objective), computed no-grad by the trainer.
"""

from __future__ import annotations

import numpy as np
import torch

from ..metrics import kge as kge_numpy  # noqa: F401  (re-export for trainers)


#: the peak term's record constant below this (mm/day) switches the term off for
#: the basin (keeps the value finite; the smallest daily-entity constant is 0.39)
PEAK_REF_FLOOR = 0.01


def _top_mean(x: torch.Tensor, k: torch.Tensor, kmax: int) -> torch.Tensor:
    """Per row, the mean of the ``k`` largest values (``k`` (B,) integer, 1 <=
    k <= ``kmax``; ``kmax`` a Python int from the static shape).  Branch-free (topk
    + rank mask, no host sync), so a per-row ``k`` stays CUDA-graph capturable."""
    xs = torch.topk(x, kmax, dim=1).values
    rank = torch.arange(kmax, device=x.device)
    m = (rank[None, :] < k[:, None]).to(x.dtype)
    return (xs * m).sum(dim=1) / k.to(x.dtype)


def jul_sep_window(doy: torch.Tensor, leap: torch.Tensor) -> torch.Tensor:
    """(T,) bool: 1 Jul - 30 Sep, from the 1-based day of year and the leap flag
    (the timing term's recession window; capture-safe)."""
    lp = leap.to(doy.dtype)
    return (doy >= 182.0 + lp) & (doy <= 273.0 + lp)


def record_references(obs: np.ndarray, water_year: np.ndarray, *, min_days: int = 300,
                      peak_frac: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Per-basin record constants of the peak and timing terms, from the training
    observations (like ``obs_var``, computed once): over the water years of the
    window with at least ``min_days`` valid days, the mean of each year's observed
    top-``k`` mean (``k = max(1, round(peak_frac * valid days))``, the peak term's
    scale) and the mean of each year's observed mean daily flow (the timing term's
    volume scale).  0.0 for a basin with no such year, which gates both terms off.

    ``obs`` (B, T) mm/day with NaN where missing; ``water_year`` (T,) the water-year
    label of each day."""
    b = obs.shape[0]
    pk, vol = np.zeros(b), np.zeros(b)
    for i in range(b):
        tops, means = [], []
        for y in np.unique(water_year):
            o = obs[i, water_year == y]
            o = o[np.isfinite(o)]
            if len(o) < min_days:
                continue
            o = np.clip(o, 0.0, None)
            k = max(1, int(round(peak_frac * len(o))))
            tops.append(float(np.sort(o)[::-1][:k].mean()))
            means.append(float(o.mean()))
        if tops:
            pk[i], vol[i] = float(np.mean(tops)), float(np.mean(means))
    return pk, vol


def masked_basin_loss(
    sim: torch.Tensor,        # (B, T) basin flow
    obs: torch.Tensor,        # (B, T) observed, NaN where missing
    obs_var: torch.Tensor,    # (B,) obs variance over the FULL cal window
    *,
    kind: str = "nnse",
    log_lambda: float = 0.0,
    log_eps: float = 0.01,
    var_lambda: float = 0.0,
    var_gate_frac: float = 1e-3,
    var_huber_cap: float = 1.0,
    bias_lambda: float = 0.0,
    timing_lambda: float = 0.0,
    peak_lambda: float = 0.0,
    peak_frac: float = 0.02,
    peak_ref: torch.Tensor | None = None,
    vol_ref: torch.Tensor | None = None,
    timing_window: torch.Tensor | None = None,
    shape_min_days: int = 300,
    timing_vol_gate: float = 0.05,
    timing_min_window: int = 60,
    weight: torch.Tensor | None = None,
    weight_total: float | None = None,
    min_days: int = 90,
) -> torch.Tensor:
    """Mean over valid basins of the (normalized) squared error on this chunk.

    ``weight`` (B,), if given, reweights the per-basin mean (adaptive per-basin
    training weights) — renormalized by its own sum, so unit-mean weights leave
    the loss scale unchanged.  ``weight=None`` is byte-identical to no weighting.
    ``weight_total``, if given, replaces that renormalization: the weighted sum
    over the valid basins is divided by this FIXED total (the weight of every
    basin of the run, valid in this chunk or not — ``mt_share_norm="all"``),
    so a chunk's loss scales with the weight it holds.  ``None`` = unchanged.
    ``var_gate_frac`` is the share of the basin's record variance a chunk must
    carry for the variance term to apply; ``var_huber_cap`` the ``|alpha - 1|``
    beyond which that term grows linearly (``<= 0``: quadratic throughout).
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
        # a chunk whose obs are ~flat FOR THIS BASIN (an ephemeral gauge's
        # zero-flow season, a window-masked sliver) carries no variance
        # signal to match — the ratio explodes through the 1e-12 clamp
        # (1e6+ chunk losses occur on the multifamily domain's west-side
        # gauges), and an absolute floor alone still passes near-flat
        # chunks whose tiny denominator lets this one term hijack the
        # gradient.  The gate is therefore RELATIVE — the chunk must carry
        # >= var_gate_frac of the basin's full-record variance (the 1e-8
        # absolute floor stays a literal, for ~flat records) — and the
        # penalty is Huber-capped (quadratic to |alpha - 1| = var_huber_cap,
        # linear beyond) so no surviving chunk contributes unboundedly.
        # Skipped chunks keep their NNSE/log terms.  Branch-free on the
        # tensors; the cap is a Python-level choice (a torch.where against
        # an infinite cap would put NaN through the unselected branch).
        has_var = (vo > torch.clamp(var_gate_frac * obs_var, min=1e-8)).to(per_basin.dtype)
        alpha = vs.clamp_min(1e-12).sqrt() / vo.clamp_min(1e-12).sqrt()
        d = alpha - 1.0
        if var_huber_cap > 0.0:
            c = float(var_huber_cap)
            pen = torch.where(d.abs() <= c, d * d, 2.0 * c * d.abs() - c * c)
        else:
            pen = d * d
        per_basin = per_basin + var_lambda * has_var * pen

    if bias_lambda > 0.0:
        # KGE beta term: per-basin chunk mean-ratio (sim/obs).  Penalizes volume
        # bias directly (the over-evaporation the squared-error loss tolerates).
        mo_b = obs_f.sum(dim=1) / n_safe
        ms_b = sim_f.sum(dim=1) / n_safe
        beta = ms_b / mo_b.clamp_min(1e-12)
        per_basin = per_basin + bias_lambda * (beta - 1.0) ** 2

    if timing_lambda > 0.0 or peak_lambda > 0.0:
        # the two hydrograph-SHAPE terms score whole water years only: a basin-chunk
        # needs shape_min_days valid days (the 92-day envelope tail and sparse years
        # drop out) and a positive record constant (a basin without a qualifying
        # training year never gets them).  Gated rows add 0 and a 0 gradient.
        if ((peak_lambda > 0.0 and peak_ref is None)
                or (timing_lambda > 0.0 and (vol_ref is None or timing_window is None))):
            raise ValueError("the peak / timing terms need their record constants "
                             "(peak_ref / vol_ref, loss.record_references) and the timing "
                             "term its day window (timing_window, loss.jul_sep_window)")
        shape_ok = n_fin >= shape_min_days

    if timing_lambda > 0.0:
        # summer RECESSION timing: over the valid days of the chunk's 1 Jul - 30 Sep
        # window (timing_window, (T,) bool), the share of the window's volume passed
        # by each day, sim vs obs, mean squared.  Volume-blind (the level terms carry
        # volume) and zero outside the window, so it pulls on the recession shape —
        # a store that drains too fast or too slow — and never on winter volume or
        # the flood peaks (a whole-year cumulative curve did both, against the other
        # terms at most Sierra basins).  Missing days add nothing to either sum.
        # Gated off unless the window has timing_min_window valid days and an
        # observed mean of at least timing_vol_gate of the basin's record mean flow
        # (a dry summer has no recession to match; an all-zero one would leave the
        # observed curve at 0), and unless the simulated window volume is at least
        # 10 % of the observed one: below that there is no simulated recession to
        # shape, and a floored normalizer would turn the term into a summer-VOLUME
        # pull (at the untrained start ~80 % of the rows sit there, with 3x the
        # gradient of the level terms) — the level terms carry that deficit.  The
        # gate makes the gradient bounded by ~2 / (0.1 x observed window volume).
        m = finite & timing_window.reshape(1, -1)
        mf = m.to(sim_f.dtype)
        n_w = m.sum(dim=1)
        n_w_safe = n_w.clamp_min(1)
        cs = torch.cumsum(sim_f.clamp_min(0.0) * mf, dim=1)
        co = torch.cumsum(obs_f.clamp_min(0.0) * mf, dim=1)
        s_tot, o_tot = cs[:, -1], co[:, -1]
        cs = cs / torch.maximum(s_tot, 0.1 * o_tot).clamp_min(1e-6).unsqueeze(1)
        co = co / o_tot.clamp_min(1e-6).unsqueeze(1)
        v_ok = vol_ref > 0.0
        v_safe = torch.where(v_ok, vol_ref, torch.ones_like(vol_ref))
        gate = (shape_ok & v_ok & (n_w >= timing_min_window)
                & (o_tot / n_w_safe >= (timing_vol_gate * v_safe).clamp_min(1e-3))
                & (s_tot >= 0.1 * o_tot))
        per_basin = per_basin + timing_lambda * gate.to(per_basin.dtype) * (
            ((cs - co) ** 2 * mf).sum(dim=1) / n_w_safe)

    if peak_lambda > 0.0:
        # FLOOD PEAKS: the mean of the chunk's top ``peak_frac`` valid days, sim vs
        # obs, each sorted on its own (the flow-duration curve's high segment —
        # timing-free), k = max(1, round(peak_frac * valid days)).  The difference is
        # normalized by the basin's RECORD constant peak_ref (the record mean of the
        # yearly observed top-k mean), not by the chunk's own observed peaks, and
        # Huber-capped at 1: so a flood year, whose peaks are several times the
        # record mean, carries the term, a drought year's small peaks weigh little,
        # an over- and an under-prediction of the same size cost the same, and an
        # all-zero observed year stays bounded.  The other terms can trade the few
        # largest storm / rain-on-snow peaks of the wettest years away when a large
        # store buys low-flow skill (the 3-water-year TBPTT run F lost 30-50 % of
        # them at the Sierra snow basins while each year's variance ratio stayed
        # ~1).  Missing days are zero in both series and sort to the bottom.  A
        # record constant under PEAK_REF_FLOOR switches the term off for the basin.
        kmax = max(1, int(round(peak_frac * sim.shape[1])))
        kk = (torch.round(peak_frac * n_fin.to(sim.dtype)).clamp(1.0, float(kmax))
              .to(torch.long))
        ts = _top_mean(sim_f.clamp_min(0.0), kk, kmax)
        to = _top_mean(obs_f.clamp_min(0.0), kk, kmax)
        p_ok = peak_ref >= PEAK_REF_FLOOR
        e = (ts - to) / torch.where(p_ok, peak_ref, torch.ones_like(peak_ref))
        pen = torch.where(e.abs() <= 1.0, e * e, 2.0 * e.abs() - 1.0)
        per_basin = per_basin + peak_lambda * (shape_ok & p_ok).to(per_basin.dtype) * pen

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
