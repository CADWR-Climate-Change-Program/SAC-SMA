"""The dPL parameter network g(attributes) -> physical parameters.

A flat MLP (the tmp/src_dpl "flat variant": encoder + single linear head)
whose sigmoid outputs are mapped into the GA feasible box ``config.BOUNDS`` —
log-space interpolation for the parameters whose bounds span decades
(``config.LOG_SPACE_PARAMS``).  Every free parameter is emitted PER HRU
("everything per-HRU"); ``config.FIXED_PARAMS`` (side/SCF/PXTEMP) are appended
as constants.

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
    CANOPY_BOUNDS,
    CANOPY_LEARNED_PARAMS,
    CANOPY_LITE_LEARNED,
    CANOPY_LOG_PARAMS,
    DYNAMIC_SAC_PARAMS,
    FIXED_PARAMS,
    FREE_PARAMS,
    LOG_SPACE_PARAMS,
    PARAM_GROUPS,
)

_MIN_NORM = 0.02   # keep prior logits away from the sigmoid tails


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
                 seasonal_params: tuple[str, ...] = (),
                 seasonal_amp: float = 0.18, seasonal_amp_frac: float = 0.10,
                 canopy: bool = False,
                 canopy_separate_trunk: bool = True,
                 canopy_lite: bool = False,
                 dynamic_params: tuple[str, ...] = (),
                 dynamic_amp: float = 0.5):
        super().__init__()
        self.grouped_heads = grouped_heads
        self.gnn_k = gnn_k
        self.seasonal_params = tuple(seasonal_params)
        self.seasonal_amp = float(seasonal_amp)
        self.canopy = bool(canopy)
        self.canopy_separate_trunk = bool(canopy_separate_trunk)
        self.canopy_lite = bool(canopy_lite)
        # which canopy params the head emits: LITE => soil_chi alone (the one
        # streamflow-identifiable knob); FULL => the 7 physiology params.
        self._canopy_learned = (CANOPY_LITE_LEARNED if self.canopy_lite
                                else CANOPY_LEARNED_PARAMS)
        # climate-state dynamic params, split by which trunk emits the coeff:
        # SAC params off the shared trunk z, canopy params off the canopy trunk zc.
        self.dynamic_amp = float(dynamic_amp)
        self._dyn_sac = tuple(p for p in dynamic_params if p in DYNAMIC_SAC_PARAMS)
        self._dyn_canopy = tuple(p for p in dynamic_params
                                 if p in CANOPY_LEARNED_PARAMS)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed), nn.ReLU(),
        )
        if grouped_heads:
            self.heads = nn.ModuleDict(
                {g: nn.Linear(embed, len(ps)) for g, ps in PARAM_GROUPS.items()})
        else:
            self.head = nn.Linear(embed, len(FREE_PARAMS))
        if self.seasonal_params:
            # 2 harmonic coeffs (a_sin, a_cos) per seasonal param off the shared
            # trunk; ZERO-initialized so the parameter field is EXACTLY static at
            # init (seasonality grows only if it lowers the loss) — the clean-
            # superset property that makes this a controlled ablation.  The
            # forward caps the raw output at |a| <= seasonal_amp via tanh so the
            # day-of-year swing cannot run away (unbounded coeffs diverged at
            # LR 1e-3); zero-init keeps tanh(0)=0 => exact static parity.
            self.seasonal_head = nn.Linear(embed, 2 * len(self.seasonal_params))
            with torch.no_grad():
                self.seasonal_head.weight.zero_()
                self.seasonal_head.bias.zero_()
            # per-param cap = frac*(hi-lo), repeated for (a_sin, a_cos); a static
            # buffer (baked into the state_dict like _lo/_hi) so eval restores it.
            caps = [seasonal_amp_frac * (BOUNDS[p][1] - BOUNDS[p][0])
                    for p in self.seasonal_params for _ in range(2)]
            self.register_buffer("_seasonal_cap",
                                 torch.tensor(caps, dtype=torch.float64))
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
            # Noah ET head: the LEARNED canopy params per node mapped into
            # CANOPY_BOUNDS (rcmin log-space) — FULL = 7 physiology params, LITE
            # = soil_chi alone (``self._canopy_learned``).  veg_frac + lai are
            # OBSERVED (pinned per cell), NOT emitted here.  ZERO-init the head so
            # sigmoid(0)=0.5 lands every param at its (log-aware) bound midpoint
            # — a sane start.  FULL puts the head on a SEPARATE encoder
            # (canopy_separate_trunk) so the 6 weak/non-identifiable physiology
            # params cannot perturb the GA-prior SAC pathway; LITE emits the one
            # soil_chi knob off the SHARED trunk z (no separate encoder needed).
            # Emitted in a SEPARATE out["_canopy"] dict; NEVER in ga_optimum.
            if self.canopy_separate_trunk:
                self.canopy_encoder = nn.Sequential(
                    nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, embed), nn.ReLU(),
                )
            self.canopy_head = nn.Linear(embed, len(self._canopy_learned))
            with torch.no_grad():
                self.canopy_head.weight.zero_()
                self.canopy_head.bias.zero_()
            clo = torch.tensor([CANOPY_BOUNDS[p][0] for p in self._canopy_learned],
                               dtype=torch.float64)
            chi = torch.tensor([CANOPY_BOUNDS[p][1] for p in self._canopy_learned],
                               dtype=torch.float64)
            c_is_log = torch.tensor(
                [p in CANOPY_LOG_PARAMS for p in self._canopy_learned])
            self.register_buffer("_c_lo", torch.where(c_is_log, clo.log(), clo))
            self.register_buffer("_c_hi", torch.where(c_is_log, chi.log(), chi))
            self.register_buffer("_c_is_log", c_is_log)

        # Climate-state DYNAMIC heads: one bounded coeff b per dynamic param,
        # ZERO-init so the field is EXACTLY static at init (clean superset, like
        # the seasonal head).  The physics reconstructs param(t)=clamp(base +
        # b*state(t), lo, hi).  SAC coeffs come off the shared trunk z, canopy
        # coeffs off the canopy trunk zc.
        if self._dyn_sac:
            self.dynamic_head = nn.Linear(embed, len(self._dyn_sac))
            with torch.no_grad():
                self.dynamic_head.weight.zero_()
                self.dynamic_head.bias.zero_()
        if self._dyn_canopy:
            self.canopy_dynamic_head = nn.Linear(embed, len(self._dyn_canopy))
            with torch.no_grad():
                self.canopy_dynamic_head.weight.zero_()
                self.canopy_dynamic_head.bias.zero_()

    def set_neighbors(self, idx, w) -> None:
        """Load the (N, k) neighbor tables (numpy or tensor) into the buffers."""
        if self.gnn_k <= 0:
            raise ValueError("net was built without gnn_k")
        self._nbr_idx.copy_(torch.as_tensor(idx, dtype=torch.int64))
        self._nbr_w.copy_(torch.as_tensor(w, dtype=torch.float64))

    def _head_params(self) -> list[tuple[nn.Linear, tuple[str, ...]]]:
        if self.grouped_heads:
            return [(self.heads[g], ps) for g, ps in PARAM_GROUPS.items()]
        return [(self.head, FREE_PARAMS)]

    def init_from_priors(self, priors: dict[str, float]) -> None:
        """Zero the head weights; set biases so the initial field == priors."""
        with torch.no_grad():
            for head, params in self._head_params():
                head.weight.zero_()
                for i, p in enumerate(params):
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
        if self.seasonal_params:
            # tanh-capped per-param: |a_sin|,|a_cos| <= frac*(hi-lo) (relative swing).
            sc = self._seasonal_cap.to(z.dtype) * torch.tanh(self.seasonal_head(z))  # (N, 2*S)
            for i, p in enumerate(self.seasonal_params):
                out[f"{p}_asin"] = sc[:, 2 * i]
                out[f"{p}_acos"] = sc[:, 2 * i + 1]
        if self._dyn_sac:
            # climate-state response coeff b per SAC dynamic param, off z; the
            # physics adds b*state(t) in the _seasonal reconstruction.
            ds = self.dynamic_amp * torch.tanh(self.dynamic_head(z))   # (N, |sac|)
            for i, p in enumerate(self._dyn_sac):
                out[f"{p}_dyn"] = ds[:, i]
        if self.canopy:
            # LEARNED canopy params (FULL: 7 physiology; LITE: soil_chi) via the
            # same sigmoid->[lo,hi]->double-where-exp map, off the separate canopy
            # trunk (FULL) or the shared z (LITE); returned SEPARATELY so they
            # never collide with ga_optimum names.
            zc = self.canopy_encoder(x) if self.canopy_separate_trunk else z
            cs = torch.sigmoid(self.canopy_head(zc))     # (N, |canopy_learned|) in (0,1)
            clo = self._c_lo.to(cs.dtype)
            chi = self._c_hi.to(cs.dtype)
            cv = clo + cs * (chi - clo)
            csafe = torch.where(self._c_is_log, cv, torch.zeros_like(cv))
            cv = torch.where(self._c_is_log, csafe.exp(), cv)
            out["_canopy"] = {p: cv[:, i]
                              for i, p in enumerate(self._canopy_learned)}
            if self._dyn_canopy:
                # climate-state response coeff b per canopy dynamic param, off zc;
                # forward.run_window reconstructs param(t)=clamp(base+b*state,lo,hi).
                dc = self.dynamic_amp * torch.tanh(self.canopy_dynamic_head(zc))
                for i, p in enumerate(self._dyn_canopy):
                    out["_canopy"][f"{p}_dyn"] = dc[:, i]
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
