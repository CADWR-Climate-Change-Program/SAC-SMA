"""The dPL parameter network g(attributes) -> physical parameters.

A flat MLP (the tmp/src_dpl "flat variant": encoder + single linear head)
whose sigmoid outputs are mapped into the GA feasible box ``config.BOUNDS`` —
log-space interpolation for the parameters whose bounds span decades
(``config.LOG_SPACE_PARAMS``) — or into a narrower per-parameter box
(``DplConfig.param_box``, :meth:`ParameterNet.set_box`).  Every free parameter is emitted PER HRU
("everything per-HRU"); ``config.FIXED_PARAMS`` (side/SCF/PXTEMP) are appended
as constants — except PXTEMP when ``pxtemp_learn`` gives it its own head.  A Noah-lite run
(``canopy``) adds the moisture exponent ``soil_chi`` from a head of its own.

GA-prior initialization (ported pattern): the head weights start at zero and
each bias at the logit of the (area-weighted median) archived GA value's
normalized position inside its bounds — so the untrained network reproduces a
GA-median uniform parameter field, and training departs from a hydrologically
sane starting point rather than mid-box noise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import (
    BOUNDS,
    FIXED_PARAMS,
    FREE_PARAMS,
    LOG_SPACE_PARAMS,
    PARAM_GROUPS,
    SOIL_CHI_BOUNDS,
)

_MIN_NORM = 0.02   # keep prior logits away from the sigmoid tails
#: buffers of earlier checkpoints that the network no longer has
_RETIRED_BUFFERS = ("_c_is_log",)


def _normalized_position(name: str, value: float) -> float:
    """Position of ``value`` in [lo, hi] on the head's (log or linear) scale."""
    lo, hi = BOUNDS[name]
    if name in LOG_SPACE_PARAMS:
        pos = (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        pos = (value - lo) / (hi - lo)
    return min(max(pos, _MIN_NORM), 1.0 - _MIN_NORM)


class ParameterNet(nn.Module):
    """(N, F) static features -> dict of (N,) physical parameters.

    ``grouped_heads=True`` (net-v2) replaces the single linear head with one
    small head per physics group (PET / SMA / Snow-17 / routing,
    :data:`sacsma.dpl.config.PARAM_GROUPS`) off the shared trunk, so the
    groups stop competing for the same output projection.  Group order
    concatenates exactly to FREE_PARAMS — everything downstream is identical.

    ``gnn_k > 0`` (net-v2) inserts ONE weighted-mean message-passing round
    over the within-basin geographic k-NN neighborhoods (the learned
    counterpart of the fixed ``spatial_reg`` smoother — the data decides where
    smoothing applies) between the encoder and the head(s):
    ``z <- z + mix(cat(z, sum_j w_ij z_j))``.  The mixing layer is ZERO-
    initialized, so the network is exactly the v1 forward at init (GA-prior
    parity preserved), and the neighbor tables (from
    :func:`sacsma.dpl.regularize.dense_neighbors`, row-normalized weights)
    are persistent buffers — baked into the checkpoint like the bounds, so
    evaluation needs no rebuild.  Gathers + matmuls only: CUDA-graph safe.
    """

    def __init__(self, n_features: int, *, hidden: int = 64, embed: int = 32,
                 dropout: float = 0.1, grouped_heads: bool = False,
                 gnn_k: int = 0, n_nodes: int | None = None,
                 canopy: bool = False,
                 pxtemp_learn: bool = False,
                 pxtemp_box: tuple[float, float] = (-1.0, 3.0),
                 pxtemp_tau: float = 1.0):
        super().__init__()
        self.grouped_heads = grouped_heads
        self.gnn_k = gnn_k
        self.canopy = bool(canopy)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed), nn.ReLU(),
        )
        if grouped_heads:
            self.heads = nn.ModuleDict(
                {g: nn.Linear(embed, len(ps)) for g, ps in PARAM_GROUPS.items()})
        else:
            self.head = nn.Linear(embed, len(FREE_PARAMS))
        if gnn_k > 0:
            if n_nodes is None:
                raise ValueError("gnn_k > 0 requires n_nodes (the fixed HRU count)")
            self.gnn_mix = nn.Linear(2 * embed, embed)
            with torch.no_grad():                 # exact identity at init
                self.gnn_mix.weight.zero_()
                self.gnn_mix.bias.zero_()
            self.register_buffer(
                "_nbr_idx",
                torch.arange(n_nodes, dtype=torch.int64).unsqueeze(1).repeat(1, gnn_k))
            self.register_buffer(
                "_nbr_w", torch.zeros(n_nodes, gnn_k, dtype=torch.float64))
        # bounds tensors in FREE_PARAMS order, on the head's scale
        lo = torch.tensor([BOUNDS[p][0] for p in FREE_PARAMS], dtype=torch.float64)
        hi = torch.tensor([BOUNDS[p][1] for p in FREE_PARAMS], dtype=torch.float64)
        is_log = torch.tensor([p in LOG_SPACE_PARAMS for p in FREE_PARAMS])
        self.register_buffer("_lo", torch.where(is_log, lo.log(), lo))
        self.register_buffer("_hi", torch.where(is_log, hi.log(), hi))
        self.register_buffer("_is_log", is_log)
        if self.canopy:
            # Noah-lite: the moisture exponent soil_chi off the shared trunk, zero-init so
            # sigmoid(0) = 0.5 starts every cell at the middle of its bounds
            self.canopy_head = nn.Linear(embed, 1)
            with torch.no_grad():
                self.canopy_head.weight.zero_()
                self.canopy_head.bias.zero_()
            self.register_buffer("_c_lo", torch.tensor([SOIL_CHI_BOUNDS[0]], dtype=torch.float64))
            self.register_buffer("_c_hi", torch.tensor([SOIL_CHI_BOUNDS[1]], dtype=torch.float64))

        # Learned Snow-17 rain/snow threshold: one zero-init output off the shared
        # trunk, mapped by tanh onto [lo, 0] / [0, hi] (piecewise, so tanh(0)=0
        # lands EXACTLY on the GA constant 0 degC — the untrained forward is the
        # fixed-threshold one).  Built last under a forked RNG so the other
        # modules' init and the training RNG stream match a run without it.
        self.pxtemp_learn = bool(pxtemp_learn)
        if self.pxtemp_learn:
            with torch.random.fork_rng(devices=[]):
                self.pxtemp_head = nn.Linear(embed, 1)
            with torch.no_grad():
                self.pxtemp_head.weight.zero_()
                self.pxtemp_head.bias.zero_()
            lo, hi = (float(v) for v in pxtemp_box)
            self.register_buffer("_px_box", torch.tensor([lo, hi], dtype=torch.float64))
            self.register_buffer("_px_tau", torch.tensor(float(pxtemp_tau),
                                                         dtype=torch.float64))

    @classmethod
    def from_checkpoint(cls, ck: dict, n_features: int) -> ParameterNet:
        """The network of the training checkpoint ``ck``, weights loaded (a ``gnn_k`` network
        carries its neighbor tables in them)."""
        nc = ck.get("net_config", {})
        gnn_k = nc.get("gnn_k", 0)
        net = cls(n_features, hidden=nc.get("hidden", 64), embed=nc.get("embed", 32),
                  dropout=nc.get("dropout", 0.1), grouped_heads=nc.get("grouped_heads", False),
                  gnn_k=gnn_k, n_nodes=ck["net"]["_nbr_idx"].shape[0] if gnn_k else None,
                  canopy=nc.get("canopy", False), pxtemp_learn=nc.get("pxtemp_learn", False),
                  pxtemp_box=tuple(nc.get("pxtemp_box", (-1.0, 3.0))),
                  pxtemp_tau=nc.get("pxtemp_tau", 1.0))
        net.load_state_dict({k: v for k, v in ck["net"].items() if k not in _RETIRED_BUFFERS})
        return net

    def set_neighbors(self, idx, w) -> None:
        """Load the (N, k) neighbor tables (numpy or tensor) into the buffers."""
        if self.gnn_k <= 0:
            raise ValueError("net was built without gnn_k")
        self._nbr_idx.copy_(torch.as_tensor(idx, dtype=torch.int64))
        self._nbr_w.copy_(torch.as_tensor(w, dtype=torch.float64))

    def set_box(self, box: dict[str, tuple[float, float]]) -> None:
        """Override the bounds box of named free parameters (physical units)
        in the ``_lo``/``_hi`` buffers the checkpoint carries; ``lo == hi``
        pins the parameter at that value."""
        with torch.no_grad():
            for p, (lo, hi) in box.items():
                i = FREE_PARAMS.index(p)
                if p in LOG_SPACE_PARAMS:
                    lo, hi = math.log(lo), math.log(hi)
                self._lo[i] = lo
                self._hi[i] = hi

    def _head_params(self) -> list[tuple[nn.Linear, tuple[str, ...]]]:
        if self.grouped_heads:
            return [(self.heads[g], ps) for g, ps in PARAM_GROUPS.items()]
        return [(self.head, FREE_PARAMS)]

    def init_from_priors(self, priors: dict[str, float],
                         box: dict[str, tuple[float, float]] | None = None) -> None:
        """Zero the head weights; set biases so the initial field == priors.
        A parameter narrowed by ``box`` (lo < hi) starts at its prior clamped
        into that box (its position measured in the box, not in ``BOUNDS``); a
        pinned one (lo == hi) is exact whatever its bias."""
        box = box or {}
        with torch.no_grad():
            for head, params in self._head_params():
                head.weight.zero_()
                for i, p in enumerate(params):
                    if p in box and box[p][0] < box[p][1]:
                        lo, hi = box[p]
                        v = min(max(priors[p], lo), hi)
                        if p in LOG_SPACE_PARAMS:
                            lo, hi, v = math.log(lo), math.log(hi), math.log(v)
                        pos = min(max((v - lo) / (hi - lo), _MIN_NORM), 1.0 - _MIN_NORM)
                    else:
                        pos = _normalized_position(p, priors[p])
                    head.bias[i] = math.log(pos / (1.0 - pos))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        if self.gnn_k > 0:
            m = (z[self._nbr_idx] * self._nbr_w.to(z.dtype).unsqueeze(-1)).sum(1)
            z = z + self.gnn_mix(torch.cat([z, m], dim=-1))
        if self.grouped_heads:                                   # order == FREE_PARAMS
            raw = torch.cat([self.heads[g](z) for g in PARAM_GROUPS], dim=-1)
        else:
            raw = self.head(z)
        s = torch.sigmoid(raw)                                   # (N, P) in (0,1)
        lo = self._lo.to(s.dtype)
        hi = self._hi.to(s.dtype)
        v = lo + s * (hi - lo)
        # back from log scale — double-where so exp() never sees the linear
        # columns' large physical values (exp(5000)=inf would leak NaN into the
        # where backward: 0-grad x inf)
        safe = torch.where(self._is_log, v, torch.zeros_like(v))
        v = torch.where(self._is_log, safe.exp(), v)
        out = {p: v[:, i] for i, p in enumerate(FREE_PARAMS)}
        n = x.shape[0]
        for p, c in FIXED_PARAMS.items():
            out[p] = torch.full((n,), c, device=x.device, dtype=s.dtype)
        if self.pxtemp_learn:
            # per-cell threshold in the box; PXTEMP_tau switches Snow-17 to the
            # straight-through split (hard forward, sigmoid-surrogate gradient)
            r = torch.tanh(self.pxtemp_head(z)[:, 0])
            box = self._px_box.to(r.dtype)
            out["PXTEMP"] = torch.where(r >= 0.0, r * box[1], -r * box[0])
            out["PXTEMP_tau"] = self._px_tau.to(r.dtype).repeat(n)   # (N,), contiguous
        if self.canopy:
            cs = torch.sigmoid(self.canopy_head(z))[:, 0]
            lo, hi = self._c_lo.to(cs.dtype), self._c_hi.to(cs.dtype)
            out["soil_chi"] = lo[0] + cs * (hi[0] - lo[0])
        return out


def ga_priors(params_df, hrus) -> dict[str, float]:
    """Area-weighted median of each free parameter over all HRUs (the init prior)."""
    merged = hrus.merge(params_df, on="key", how="left")
    w = merged["area_weight"].to_numpy(float)
    priors: dict[str, float] = {}
    for p in FREE_PARAMS:
        v = merged[p].to_numpy(float)
        order = v.argsort()
        cw = w[order].cumsum()
        priors[p] = float(v[order][cw.searchsorted(cw[-1] / 2.0)])
    return priors
