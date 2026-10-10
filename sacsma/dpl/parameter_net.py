"""The dPL parameter network g(attributes) -> physical parameters.

A flat MLP (the tmp/src_dpl "flat variant": encoder + single linear head)
whose sigmoid outputs are mapped into the GA feasible box ``config.BOUNDS`` —
log-space interpolation for the parameters whose bounds span decades
(``config.LOG_SPACE_PARAMS``).  Every free parameter is emitted PER HRU
("everything per-HRU"); ``config.FIXED_PARAMS`` (side/SCF/PXTEMP) are appended
as constants — except PXTEMP when ``pxtemp_learn`` gives it its own head.  A Noah-lite run
(``canopy``) adds the moisture exponent ``soil_chi`` from a head of its own.

GA-prior initialization (ported pattern): the head weights start at zero and
each bias at the logit of the (area-weighted median) archived GA value's
normalized position inside its bounds — so the untrained network reproduces a
GA-median uniform parameter field, and training departs from a hydrologically
sane starting point rather than mid-box noise.

Per-run mapping options (opt-in; the defaults are the mapping above): ``log_params`` maps more
parameters in log space and ``bounds`` replaces a parameter's ``BOUNDS`` box (it may widen it);
both live in the ``_lo``/``_hi``/``_is_log`` buffers, so a checkpoint carries them.
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
    SOIL_CHI_BOUNDS,
)

_MIN_NORM = 0.02   # keep prior logits away from the sigmoid tails
#: |pre-activation| beyond which the training's soft logit penalty applies
#: (``DplConfig.logit_penalty``); a tanh head's pre-activation counts double (tanh(u) =
#: 2 sigmoid(2u) - 1)
LOGIT_CAP = 6.0


def _normalized_position(name: str, value: float, bounds: dict | None = None,
                         log_params: tuple[str, ...] = ()) -> float:
    """Position of ``value`` in [lo, hi] on the head's (log or linear) scale."""
    lo, hi = (bounds or {}).get(name, BOUNDS[name])
    if name in LOG_SPACE_PARAMS or name in log_params:
        pos = (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        pos = (value - lo) / (hi - lo)
    return min(max(pos, _MIN_NORM), 1.0 - _MIN_NORM)


def logit_penalty(pre: torch.Tensor) -> torch.Tensor:
    """Mean over rows and outputs of (|pre| - LOGIT_CAP)^2 beyond the cap, 0 inside it
    (``pre``: the pre-activations of ``ParameterNet.forward(x, logits=True)``)."""
    return torch.relu(pre.abs() - LOGIT_CAP).square().mean()


class ParameterNet(nn.Module):
    """(N, F) static features -> dict of (N,) physical parameters: a two-layer encoder
    and one linear head over the free parameters, plus the zero-init heads of the
    learned PXTEMP (``pxtemp_learn``) and of the Noah-lite ``soil_chi`` (``canopy``)
    off the same trunk."""

    def __init__(self, n_features: int, *, hidden: int = 64, embed: int = 32,
                 dropout: float = 0.1,
                 canopy: bool = False,
                 pxtemp_learn: bool = False,
                 pxtemp_box: tuple[float, float] = (-1.0, 3.0),
                 pxtemp_tau: float = 1.0,
                 log_params: tuple[str, ...] = (),
                 bounds: dict[str, tuple[float, float]] | None = None):
        super().__init__()
        self.canopy = bool(canopy)
        self.log_params = tuple(log_params)
        self.bounds = dict(bounds or {})
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed), nn.ReLU(),
        )
        self.head = nn.Linear(embed, len(FREE_PARAMS))
        # bounds tensors in FREE_PARAMS order, on the head's scale
        lo = torch.tensor([BOUNDS[p][0] for p in FREE_PARAMS], dtype=torch.float64)
        hi = torch.tensor([BOUNDS[p][1] for p in FREE_PARAMS], dtype=torch.float64)
        is_log = torch.tensor([p in LOG_SPACE_PARAMS for p in FREE_PARAMS])
        self.register_buffer("_lo", torch.where(is_log, lo.log(), lo))
        self.register_buffer("_hi", torch.where(is_log, hi.log(), hi))
        self.register_buffer("_is_log", is_log)
        if self.log_params or self.bounds:
            self.set_mapping(self.log_params, self.bounds)
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
        """The network of the training checkpoint ``ck``, weights loaded (its parameter
        mapping in them)."""
        nc = ck["net_config"]
        net = cls(n_features, hidden=nc["hidden"], embed=nc["embed"], dropout=nc["dropout"],
                  canopy=nc["canopy"], pxtemp_learn=nc["pxtemp_learn"],
                  pxtemp_box=tuple(nc["pxtemp_box"]), pxtemp_tau=nc["pxtemp_tau"],
                  log_params=tuple(nc["log_params"]), bounds=nc["bounds"])
        net.load_state_dict(ck["net"])
        return net

    def set_mapping(self, log_params: tuple[str, ...] = (),
                    bounds: dict[str, tuple[float, float]] | None = None) -> None:
        """Map the named free parameters in log space (``log_params``) and/or into the box
        ``bounds`` (physical units, replacing ``BOUNDS``; it may widen it), in the
        ``_lo``/``_hi``/``_is_log`` buffers; every other parameter keeps its entries."""
        bounds = bounds or {}
        with torch.no_grad():
            for p in dict.fromkeys((*log_params, *bounds)):
                i = FREE_PARAMS.index(p)
                lo, hi = bounds.get(p, BOUNDS[p])
                log = p in LOG_SPACE_PARAMS or p in log_params
                if not lo < hi or (log and lo <= 0.0):
                    raise ValueError(f"{p}: box ({lo}, {hi}) is empty or not positive in log space")
                self._is_log[i] = log
                self._lo[i] = math.log(lo) if log else lo
                self._hi[i] = math.log(hi) if log else hi

    def init_from_priors(self, priors: dict[str, float]) -> None:
        """Zero the head weights; set biases so the initial field == priors.
        Positions are on the net's own mapping (``log_params``, ``bounds``)."""
        with torch.no_grad():
            self.head.weight.zero_()
            for i, p in enumerate(FREE_PARAMS):
                pos = _normalized_position(p, priors[p], self.bounds, self.log_params)
                self.head.bias[i] = math.log(pos / (1.0 - pos))

    def forward(self, x: torch.Tensor, *, logits: bool = False):
        """The parameter dict; with ``logits`` also the (N, K) pre-activations in sigmoid
        units (the free parameters, then 2u of the PXTEMP tanh head, then soil_chi's) that
        the logit penalty reads."""
        z = self.encoder(x)
        raw = self.head(z)
        pre = [raw]
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
            u = self.pxtemp_head(z)
            pre.append(2.0 * u)
            r = torch.tanh(u[:, 0])
            box = self._px_box.to(r.dtype)
            out["PXTEMP"] = torch.where(r >= 0.0, r * box[1], -r * box[0])
            out["PXTEMP_tau"] = self._px_tau.to(r.dtype).repeat(n)   # (N,), contiguous
        if self.canopy:
            c = self.canopy_head(z)
            pre.append(c)
            cs = torch.sigmoid(c)[:, 0]
            lo, hi = self._c_lo.to(cs.dtype), self._c_hi.to(cs.dtype)
            out["soil_chi"] = lo[0] + cs * (hi[0] - lo[0])
        if logits:
            return out, torch.cat(pre, dim=-1)
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
