"""dPL configuration: parameter bounds, fixed parameters, run/train settings.

Bounds are the ORIGINAL GA feasible ranges from the archived calibration setup
(``tmp/sacsma_module/sacramento_ga_15cdec_pool.txt``, parameter-description
block) — the same box the pooled optimum was drawn from, so every value in
``data/cdec15/ga_optimum.csv`` lies inside them (asserted by
:func:`validate_ga_optimum`).  ``side``, ``SCF`` and ``PXTEMP`` had degenerate
ranges there (held fixed) and stay fixed here.

TWO bounds are deliberately widened past the archived GA box for the dPL search
(bound-pinch probe, 2026-07-11 — the learned field pinned ~30%/23% of HRUs at
these limits and releasing them improved frozen cal/val KGE): ``rexp`` ceiling
10 -> 15 and ``lzsk`` floor 0.01 -> 0.003.  Both only EXPAND the box (never
exclude an archived GA value), so ``validate_ga_optimum`` still passes.  The
probe found NMF/Diff/UADJ pinned but inert (no daily-flow leverage), so those
stay at the archived limits.

This module imports no torch at module scope — it is safe to import from the
core package paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..parameters import _ROUT_COLS, _SMA_COLS, _SNOW_COLS

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

#: All 31 parameters in ga_optimum.csv column order: Kpet, 16 SMA, 10 Snow-17, 4 routing.
PARAM_ORDER: tuple[str, ...] = ("Kpet", *_SMA_COLS, *_SNOW_COLS, *_ROUT_COLS)

#: The 34 columns of data/cdec15/ga_optimum.csv (and of exported dPL tables).
GA_OPTIMUM_COLUMNS: tuple[str, ...] = ("key", "lat", "lon", *PARAM_ORDER)

#: GA feasible ranges (lo, hi) — archived setup file, verbatim.
BOUNDS: dict[str, tuple[float, float]] = {
    "Kpet":   (0.4, 2.5),
    "uztwm":  (1.0, 1000.0),
    "uzfwm":  (1.0, 1000.0),
    "lztwm":  (50.0, 5000.0),
    "lzfpm":  (50.0, 5000.0),
    "lzfsm":  (50.0, 5000.0),
    "uzk":    (0.01, 0.99),
    "lzpk":   (0.01, 0.5),
    "lzsk":   (0.003, 0.5),  # floor widened 0.01->0.003 for dPL (see note below)
    "zperc":  (1.0, 500.0),
    "rexp":   (1.0, 15.0),   # ceiling widened 10->15 for dPL (see note below)
    "pfree":  (0.0, 0.99),
    "pctim":  (0.0, 0.5),
    "adimp":  (0.0, 0.9),
    "riva":   (0.0, 0.9),
    "side":   (0.0, 0.0),   # fixed
    "rserv":  (0.0, 0.4),
    "SCF":    (1.0, 1.0),   # fixed
    "PXTEMP": (0.0, 0.0),   # fixed
    "MFMAX":  (0.05, 5.0),
    "MFMIN":  (0.05, 5.0),
    "UADJ":   (0.03, 0.2),
    "MBASE":  (0.0, 5.0),
    "TIPM":   (0.1, 1.0),
    "PLWHC":  (0.02, 0.3),
    "NMF":    (0.05, 0.3),
    "DAYGM":  (0.0, 0.3),
    "Nres":   (1.0, 20.0),
    "Kres":   (0.01, 0.99),
    "Velo":   (0.5, 5.0),
    "Diff":   (200.0, 4000.0),
}

#: Held fixed (degenerate GA ranges) — never emitted by the parameter net.
FIXED_PARAMS: dict[str, float] = {"side": 0.0, "SCF": 1.0, "PXTEMP": 0.0}

#: The 28 parameters the network learns, in PARAM_ORDER.
FREE_PARAMS: tuple[str, ...] = tuple(p for p in PARAM_ORDER if p not in FIXED_PARAMS)

#: Parameters whose bounds span >= 2 decades — mapped in log space by the net head.
LOG_SPACE_PARAMS: frozenset[str] = frozenset(
    {"uztwm", "uzfwm", "lztwm", "lzfpm", "lzfsm", "zperc"}
)

#: Physics groups for the optional grouped output heads (net-v2): insertion
#: order concatenates EXACTLY to FREE_PARAMS (asserted below).
PARAM_GROUPS: dict[str, tuple[str, ...]] = {
    "pet": ("Kpet",),
    "sma": tuple(p for p in _SMA_COLS if p not in ("side",)),
    "snow": tuple(p for p in _SNOW_COLS if p not in ("SCF", "PXTEMP")),
    "routing": tuple(_ROUT_COLS),
}
assert tuple(p for ps in PARAM_GROUPS.values() for p in ps) == FREE_PARAMS

# ---------------------------------------------------------------------------
# Noah canopy-resistance ET (et_mode="noah") — a SEPARATE parameter set
# ---------------------------------------------------------------------------
# These drive sacsma.dpl.et_noah and are NEVER part of PARAM_ORDER / the
# ga_optimum export (the frozen model has no Noah ET).  Bounds from NWS 53
# (Koren et al. 2010) and the Noah land-surface parameter tables.
CANOPY_BOUNDS: dict[str, tuple[float, float]] = {
    "rcmin":     (5.0, 400.0),   # min stomatal resistance, s/m (wetness-dependent)
    "lai":       (0.5, 6.0),     # leaf area index
    "veg_frac":  (0.0, 1.0),     # green vegetation fraction (sigma)
    "rgl":       (30.0, 150.0),  # solar-radiation limit, W/m2 (veg-class)
    "hs":        (30.0, 55.0),   # vapour-pressure-deficit coefficient
    "wilt_frac": (0.05, 0.5),    # wilting point as a fraction of tension capacity
    "froot":     (0.3, 0.9),     # fraction of roots in the SAC upper zone
    "redist_k":  (0.0, 0.5),     # lower<->upper tension redistribution rate
}

#: Canopy parameters mapped in log space (rcmin spans ~2 decades).
CANOPY_LOG_PARAMS: frozenset[str] = frozenset({"rcmin"})

CANOPY_PARAMS: tuple[str, ...] = tuple(CANOPY_BOUNDS)


def validate_ga_optimum(params_df) -> None:
    """Assert every archived GA value lies inside BOUNDS (call at startup)."""
    for name in PARAM_ORDER:
        lo, hi = BOUNDS[name]
        col = params_df[name]
        bad = (col < lo) | (col > hi)
        if bad.any():
            raise ValueError(
                f"ga_optimum column {name!r} has {int(bad.sum())} values outside "
                f"[{lo}, {hi}] (e.g. {float(col[bad].iloc[0])})"
            )


# ---------------------------------------------------------------------------
# Run / training configuration
# ---------------------------------------------------------------------------


@dataclass
class DplConfig:
    """Numerics + training settings for the differentiable model.

    The *forward numerics* block controls how far the torch model departs from
    the frozen reference; the named fidelity configs in
    :func:`sacsma.dpl.evaluate.fidelity_benchmark` are instances of this.
    """

    # -- forward numerics -------------------------------------------------
    #: fixed SAC-SMA substep count (reference uses data-dependent
    #: ``ninc = floor(1 + 0.2*(uzfwc + twx))`` — not batchable/differentiable).
    n_inc: int = 5
    #: "fixed" = n_inc substeps everywhere (trainable); "dynamic" = the exact
    #: reference per-lane ninc via masking (fidelity checks only — has a
    #: per-day .item() sync and unbounded loop length).
    ninc_mode: str = "fixed"
    #: percolation-cap treatment: "reference" = linear demand + hard min cap
    #: (exact frozen numerics apart from n_inc); "implicit" = implicit-Euler
    #: saturator exp(-k); "tanh" = tanh(k) saturator (both bound the Jacobian).
    perc_mode: str = "reference"
    #: epsilon of the smooth-relu ``0.5*(x + sqrt(x^2 + eps^2))`` used for
    #: storage floors during training; 0.0 -> exact relu/min/max clamps.
    smooth_eps: float = 0.0
    #: floor on the LZ free-water fill-fraction denominator (reference: none;
    #: training needs ~0.1 to bound the division backward at double saturation).
    fracp_floor: float = 0.0
    #: initial states: "reference" = SMA [0,0,100,100,100,0] + Snow-17 zeros
    #: (the frozen cold start); "capacity" = storages at capacity (tmp/src_dpl).
    init_mode: str = "reference"
    dtype: str = "float32"

    # -- device ------------------------------------------------------------
    device: str = "cuda"

    # -- training -----------------------------------------------------------
    loss: str = "nnse"          # "nnse" (variance-normalized MSE) | "mse"
    log_loss_lambda: float = 0.15
    log_loss_eps: float = 0.01  # mm/day
    #: per-chunk variance-matching penalty (std-ratio - 1)^2 — counters the
    #: squared-error variance damping (alpha -> r); NOT chunked KGE.
    var_loss_lambda: float = 1.0

    # -- regularizers (opt-in; ALL default-off => byte-identical baseline) ----
    #: attribute-weighted geographic smoothness of the per-HRU parameter FIELD
    #: (Feng 2023 "stable spatial patterns"): penalize squared differences of
    #: the net's NORMALIZED params across within-basin k-NN edges, edge-weighted
    #: by exp(-attr_scale * attr_dist) so only geographically-near AND
    #: attribute-similar HRUs are tied.  A small-sample complexity brake that
    #: does NOT anchor the field to the GA optimum.  0.0 disables (default).
    spatial_reg_lambda: float = 0.0
    spatial_reg_k: int = 8               # geographic neighbours per HRU
    spatial_reg_attr_scale: float = 1.0  # attr-distance decay (units: median dist)
    #: adaptive per-basin loss weights (Rahman-ALF): every selection eval,
    #: reweight the pooled loss ∝ (1 - cal_KGE)^beta toward the worst-fitting
    #: basins (SCC/ISB), momentum-blended and renormalized to unit mean.
    adaptive_loss: bool = False
    adaptive_loss_beta: float = 1.0
    adaptive_loss_momentum: float = 0.5
    adaptive_loss_floor: float = 0.05    # min (1 - KGE) so strong basins keep weight
    adaptive_loss_clip: float = 5.0      # per-basin weight clamp [1/clip, clip]

    # -- parameter net (net-v2 knobs; defaults = the v1 architecture) --------
    hidden: int = 64
    embed: int = 32
    dropout: float = 0.1
    grouped_heads: bool = False     # per-physics-group output heads
    fourier_k: int = 0              # spatial Fourier feature order (0 = off)
    #: learned spatial smoother (net-v2): ONE weighted-mean message-passing
    #: round over within-basin geographic k-NN neighborhoods, zero-init mixing
    #: (exact v1 at init).  0 = off.  The learned counterpart of spatial_reg.
    gnn_k: int = 0
    gnn_attr_scale: float = 1.0     # attr-distance decay of the neighbor weights
    #: parameters given a day-of-year harmonic shape (the net emits 2 zero-init
    #: coeffs each; physics reconstructs param(doy)=clamp(mean+a_sin*sin(w*doy)+
    #: a_cos*cos(w*doy), bounds)).  Empty = static field (default).  The frozen
    #: model reconstructs the identical series (sacsma.parameters), so exported
    #: seasonal params score exactly.
    seasonal_params: tuple[str, ...] = ()
    #: tanh amplitude cap on the harmonic coeffs (|a_sin|,|a_cos| <= this, in
    #: additive Kpet units): the day-of-year swing is hard-bounded so unbounded
    #: coeffs cannot diverge (they did at LR 1e-3).  0.18 ~ +/-25% of Kpet~1.
    seasonal_amp: float = 0.18
    lr: float = 1e-3
    lr_min: float = 1e-5        # cosine-annealed floor
    lr_warmup_epochs: int = 3   # linear warmup protects the GA-prior init
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    n_epochs: int = 60
    spinup_refresh_every: int = 1   # full-prefix no-grad spinup every k epochs
    cal_start: str = "1988-10-01"   # WY1989 start (cal end = cdec15.CAL_END)
    train_chunk_days: int = 366     # TBPTT chunk (fixed length; last chunk's
                                    # post-CAL_END days are NaN-masked in the loss)
    eval_every: int = 2             # full-cal no-grad KGE selection cadence
    patience: int = 10              # early stop after this many stale selections
    #: CUDA-graph capture of the day-stepped pipeline (eager is dispatch-bound:
    #: ~300 tiny kernels/day).  Falls back to eager on CPU or capture failure.
    use_cuda_graphs: bool = True
    nograd_window: int = 512        # replay window for spinup/selection streaming
    seed: int = 0
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.perc_mode not in ("reference", "implicit", "tanh"):
            raise ValueError(f"perc_mode {self.perc_mode!r}")
        if self.init_mode not in ("reference", "capacity"):
            raise ValueError(f"init_mode {self.init_mode!r}")
        if self.ninc_mode not in ("fixed", "dynamic"):
            raise ValueError(f"ninc_mode {self.ninc_mode!r}")
        if self.n_inc < 1:
            raise ValueError("n_inc must be >= 1")


def _ensure_conda_dlls_on_path() -> None:
    """Put the env's ``Library/bin`` on PATH (Windows conda).

    Torch's CUDA jiterator ops (e.g. ``lgamma``) JIT-compile through NVRTC,
    and NVRTC resolves its ``nvrtc-builtins64_*.dll`` with a plain
    ``LoadLibrary`` that searches PATH only — python's own DLL directories
    (``os.add_dll_directory``) don't apply.  Running the env's python by full
    path without conda activation therefore breaks those ops unless we add
    the directory here.
    """
    import sys

    lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(lib_bin):
        paths = os.environ.get("PATH", "").split(os.pathsep)
        if not any(os.path.normcase(p) == os.path.normcase(lib_bin) for p in paths):
            os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")


def pick_device(requested: str = "cuda"):
    """Resolve the torch device; honour ``SACSMA_DISABLE_CUDNN=1``.

    Training asserts CUDA unless ``cpu`` is requested explicitly — the dPL
    study is GPU-first by design.
    """
    import torch

    _ensure_conda_dlls_on_path()
    if os.environ.get("SACSMA_DISABLE_CUDNN"):
        torch.backends.cudnn.enabled = False
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available but a GPU run was requested (pass "
            "--device cpu to override explicitly; set SACSMA_DISABLE_CUDNN=1 "
            "if only cuDNN fails to load)."
        )
    return torch.device("cuda")
