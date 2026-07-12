"""Opt-in training regularizers for dPL (default-off — a run that enables none
of them is byte-identical to the baseline).

Two independent levers, both targeting the small-sample (15-basin) regime:

* :func:`spatial_smoothness` — attribute-weighted geographic smoothness of the
  per-HRU parameter FIELD.  Because the net maps attributes -> params via a
  sigmoid onto ``[lo, hi]`` (log-interpolated for :data:`LOG_SPACE_PARAMS`),
  penalizing ``(theta_i - theta_j) / scale`` with ``scale = hi - lo`` (or the
  log-range) is EXACTLY penalizing the squared difference of the net's
  normalized outputs ``s in [0, 1]`` — so all 28 free params share one footing.
  Edges are within-basin geographic k-NN (:func:`build_neighbor_edges`), weighted
  ``exp(-attr_scale * attr_dist)`` so only HRUs that are BOTH geographically
  near and attribute-similar are tied (real elevation/soil contrasts survive).
  A complexity brake against overfitting that does NOT anchor to the GA optimum.

* :func:`adaptive_basin_weights` — Rahman-ALF per-basin loss weights: push the
  pooled loss toward the basins currently fitting worst, renormalized to unit
  mean so the overall loss scale is unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import BOUNDS, FREE_PARAMS, LOG_SPACE_PARAMS


def _basin_knn(hrus, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Directed within-basin geographic k-NN edges ``(src, dst)`` by lat/lon."""
    lat = hrus["lat"].to_numpy(np.float64)
    lon = hrus["lon"].to_numpy(np.float64)
    basin = hrus["basin"].to_numpy()

    src: list[int] = []
    dst: list[int] = []
    for b in np.unique(basin):
        idx = np.flatnonzero(basin == b)
        if idx.size < 2:
            continue
        pts = np.stack([lat[idx], lon[idx]], axis=1)             # (n, 2)
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)  # (n, n)
        np.fill_diagonal(d2, np.inf)
        kk = min(k, idx.size - 1)
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]         # k nearest per row
        for a in range(idx.size):
            for c in nn[a]:
                src.append(int(idx[a]))
                dst.append(int(idx[c]))
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)


def _attr_weights(x, edge_i: np.ndarray, edge_j: np.ndarray,
                  attr_scale: float) -> np.ndarray:
    """``exp(-attr_scale * d_attr / median)`` per edge (unnormalized)."""
    xi = np.asarray(x, np.float64)
    d_attr = np.sqrt(((xi[edge_i] - xi[edge_j]) ** 2).sum(axis=1))
    med = float(np.median(d_attr)) if d_attr.size else 1.0
    return np.exp(-attr_scale * d_attr / max(med, 1e-9))


def build_neighbor_edges(
    hrus, x, *, k: int = 8, attr_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-basin geographic k-NN edges with attribute-similarity weights.

    ``hrus`` is ``dom.hrus`` (same row order as the net input); ``x`` is the
    standardized feature matrix ``fs.x`` (N, F) used for attribute distances.
    Returns ``(edge_i, edge_j, edge_w)``: directed edges (each HRU -> its ``k``
    nearest same-basin neighbours by lat/lon) and ``exp(-attr_scale * d_attr)``
    weights normalized to unit mean (so the penalty magnitude is scale-stable).
    """
    edge_i, edge_j = _basin_knn(hrus, k)
    edge_w = _attr_weights(x, edge_i, edge_j, attr_scale)
    edge_w = edge_w / max(float(edge_w.mean()), 1e-12)           # unit mean
    return edge_i, edge_j, edge_w


def dense_neighbors(
    hrus, x, *, k: int = 8, attr_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Same neighborhoods in dense per-node form for the learned smoother
    (:class:`~sacsma.dpl.parameter_net.ParameterNet` ``gnn_k``): ``(idx, w)``
    both (N, k) — each row an HRU's same-basin nearest neighbours (self-padded
    where a small basin has < k) with attribute-similarity weights row-
    normalized to sum 1 (a weighted-mean aggregator; padding weight 0, so an
    isolated HRU aggregates nothing and the zero-init mixing layer leaves it
    untouched)."""
    edge_i, edge_j = _basin_knn(hrus, k)
    edge_w = _attr_weights(x, edge_i, edge_j, attr_scale)

    n = len(hrus)
    idx = np.tile(np.arange(n, dtype=np.int64)[:, None], (1, k))  # self-padded
    w = np.zeros((n, k), np.float64)
    fill = np.zeros(n, np.int64)
    for e in range(edge_i.size):
        i = edge_i[e]
        idx[i, fill[i]] = edge_j[e]
        w[i, fill[i]] = edge_w[e]
        fill[i] += 1
    row = w.sum(axis=1, keepdims=True)
    w = np.divide(w, row, out=np.zeros_like(w), where=row > 0)    # sum to 1 (or 0)
    return idx, w


def param_norm_scales(device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-free-param ``(scale, is_log)`` so ``(theta_i - theta_j) / scale``
    equals the difference of the net's normalized outputs (see module docstring)."""
    scale: list[float] = []
    is_log: list[bool] = []
    for p in FREE_PARAMS:
        lo, hi = BOUNDS[p]
        if p in LOG_SPACE_PARAMS:
            scale.append(float(np.log(hi) - np.log(lo)))
            is_log.append(True)
        else:
            scale.append(float(hi - lo))
            is_log.append(False)
    return (torch.tensor(scale, device=device, dtype=dtype),
            torch.tensor(is_log, device=device, dtype=torch.bool))


def spatial_smoothness(
    params: dict[str, torch.Tensor],
    edge_i: torch.Tensor, edge_j: torch.Tensor, edge_w: torch.Tensor,
    scale: torch.Tensor, is_log: torch.Tensor,
) -> torch.Tensor:
    """Edge-weighted mean squared difference of the net's normalized params.

    ``params`` is the per-HRU physical dict emitted by the net; the returned
    scalar is ``sum_e w_e * ||s_i - s_j||^2 / sum_e w_e`` over the free params,
    with ``s`` the normalized (sigmoid-space) parameter vector.  First-order in
    ``params`` (index_select only) — safe to backward outside the CUDA graph.
    """
    p = torch.stack([params[name] for name in FREE_PARAMS], dim=1)   # (N, P)
    pn = torch.where(is_log, torch.log(p.clamp_min(1e-8)), p) / scale
    d = pn.index_select(0, edge_i) - pn.index_select(0, edge_j)      # (E, P)
    per_edge = (d * d).sum(dim=1)                                    # (E,)
    return (edge_w * per_edge).sum() / edge_w.sum().clamp_min(1e-12)


def adaptive_basin_weights(
    kge: torch.Tensor, prev: torch.Tensor, *,
    beta: float = 1.0, momentum: float = 0.5,
    floor: float = 0.05, clip: float = 5.0,
) -> torch.Tensor:
    """Rahman-ALF per-basin weights from current cal KGE (both (B,) tensors).

    Target weight ``prop (1 - KGE)^beta`` (KGE floored so strong basins keep
    weight), momentum-blended with ``prev``, clamped to ``[1/clip, clip]`` and
    renormalized to unit mean so the pooled-loss scale is unchanged.
    """
    err = (1.0 - kge).clamp_min(floor)
    tgt = err.pow(beta)
    tgt = tgt / tgt.mean().clamp_min(1e-12)
    w = (1.0 - momentum) * prev + momentum * tgt
    w = w.clamp(1.0 / clip, clip)
    return w / w.mean().clamp_min(1e-12)
