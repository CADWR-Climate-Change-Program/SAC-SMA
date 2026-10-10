"""dPL configuration: parameter bounds, fixed parameters, run/train settings.

Bounds are the ORIGINAL GA feasible ranges from the archived calibration setup
(``tmp/sacsma_module/sacramento_ga_15cdec_pool.txt``, parameter-description
block) — the same box the pooled optimum was drawn from, so every value in
``artifacts/models/15cdec/ga_optimum.csv`` lies inside them.  ``side``, ``SCF`` and
``PXTEMP`` had degenerate
ranges there (held fixed) and stay fixed here.

TWO bounds are deliberately widened past the archived GA box for the dPL search
(bound-pinch probe, 2026-07-11 — the learned field pinned ~30%/23% of HRUs at
these limits and releasing them improved frozen cal/val KGE): ``rexp`` ceiling
10 -> 15 and ``lzsk`` floor 0.01 -> 0.003.  Both only EXPAND the box (never
exclude an archived GA value).  The
probe found NMF/Diff/UADJ pinned but inert (no daily-flow leverage), so those
stay at the archived limits.

This module imports no torch at module scope — it is safe to import from the
core package paths.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields

from ..parameters import _ROUT_COLS, _SMA_COLS, _SNOW_COLS

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

#: All 31 parameters in ga_optimum.csv column order: Kpet, 16 SMA, 10 Snow-17, 4 routing.
PARAM_ORDER: tuple[str, ...] = ("Kpet", *_SMA_COLS, *_SNOW_COLS, *_ROUT_COLS)


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

#: Bounds of the Noah-lite moisture exponent ``soil_chi`` (Ek et al. 2003 chi), the one
#: parameter of the Noah-lite ET the network learns; the green fraction and the LAI are
#: observed.  Never part of PARAM_ORDER or a ga_optimum export.
SOIL_CHI_BOUNDS: tuple[float, float] = (0.5, 2.5)


def config_from_checkpoint(ck: dict) -> DplConfig:
    """The training configuration of the checkpoint ``ck``, from its recorded fields (a field
    this configuration does not have is refused)."""
    rec = {k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()}
    unknown = sorted(set(rec) - {f.name for f in fields(DplConfig)})
    if unknown:
        raise ValueError(f"the checkpoint records fields DplConfig does not have: {unknown}")
    return DplConfig(**rec)


# ---------------------------------------------------------------------------
# Run / training configuration
# ---------------------------------------------------------------------------


@dataclass
class DplConfig:
    """Physics and training settings of a learned-parameter run (:meth:`physics` is the
    physics as the engine describes it)."""

    # -- forward numerics -------------------------------------------------
    #: fixed SAC-SMA substep count (the reference takes the data-dependent
    #: ``ninc = floor(1 + 0.2*(uzfwc + twx))``)
    n_inc: int = 10
    #: floor on the LZ free-water fill-fraction denominator (reference: none; it bounds the
    #: division's backward at double saturation)
    fracp_floor: float = 1e-3
    dtype: str = "float32"

    # -- device ------------------------------------------------------------
    device: str = "cuda"

    # -- training -----------------------------------------------------------
    #: the chunk loss (loss.masked_basin_loss): the variance-normalized squared error
    #: (NNSE), the low-flow log term (log_loss_lambda, log_loss_eps) and the variance
    #: term, every entity-chunk term weighted by its valid days / 365 (monthly: valid
    #: months / 12), so a partial chunk (the 92-day Oct-Dec 2018 envelope tail) counts
    #: by its length
    log_loss_lambda: float = 0.15
    log_loss_eps: float = 0.01  # mm/day
    #: per-chunk variance-matching penalty: the std difference over the basin's RECORD
    #: std (the NNSE's normalizer, so a year weighs by its share of the record
    #: variance), on basin-chunks with at least shape_min_days valid days — counters
    #: the squared-error variance damping (alpha -> r); NOT chunked KGE.
    var_loss_lambda: float = 1.0
    #: the Huber cap of the variance term: quadratic up to it, linear beyond; <= 0
    #: keeps the square throughout.
    var_huber_cap: float = 1.0
    #: the variance term scores a basin-chunk only with at least this many valid
    #: observed days (whole water years; the short envelope tail and sparse years
    #: drop out)
    shape_min_days: int = 300
    #: daily-target observations MASKED out of training and scoring, as
    #: ``"entity_id|YYYY-MM-DD"`` strings (the CLI reads them from a hand-edited
    #: CSV such as data/targets/cdec/fnf_daily_mask.csv; the checkpoint carries the
    #: list, so evaluation masks the identical days).  Empty = the stores as-is
    #: (default).
    obs_mask: tuple[str, ...] = ()
    #: water years held out of EVERY family, as an inclusive ``(first, last)`` pair,
    #: e.g. (1976, 1985) (calsim.arcs.HOLDOUT_WY): their targets are NaN after the
    #: registry n_obs audit and the obs_mask, before the NNSE normalizers — so the
    #: loss, the normalizers, the chunk liveness and selection never read them (a
    #: held-out water year with no other target becomes a dead chunk).  The
    #: checkpoint carries it; evaluation scores the same masked targets.
    #: Multi-timescale domain only.  () = off (default).
    holdout_wy: tuple[int, ...] = ()
    #: back-extend the uf_monthly TRAINING targets to this date (a first of the
    #: month, e.g. "1949-10-01" = WY1950): the months from it to each uf entity's
    #: registry train_start join its target (minus holdout_wy), read from the same
    #: store; the registry window and its n_obs audit are unchanged, the extension
    #: months are counted separately and must all be observed.  Multi-timescale
    #: domain only.  "" = off (default: the registry windows).
    uf_train_start: str = ""
    #: the training window, ``(first day, last day)`` of whole calendar months, e.g.
    #: ("1988-10-01", "2003-09-30") = the 15cdec domains' WY1989-2003: every target
    #: outside it is NaN (after the n_obs audit, the obs_mask, the holdout and the uf
    #: back-extension; counted per entity as ``n_out``) and the training chunks start at it
    #: (the spinup runs ahead of it).  Multi-timescale domain only.  () = the envelope
    #: (default).
    train_window: tuple[str, ...] = ()
    #: the CalSim3 rim-arc family (calsim_monthly) in the run: "train_default"
    #: appends the train_default arcs of data/targets/calsim3/arc_hierarchy.csv (tier A, own
    #: gauge record), in file order, as cs_<ARC> entities after the run's other
    #: entities, so a resume rebuilds the same order.  Multi-timescale domain only.
    #: "none" (default) = only the entities named or the 95 base entities.
    calsim_arcs: str = "none"

    #: multi-timescale family weighting (multifamily domain only): "none" =
    #: every valid daily entity weighs equally in the chunk mean and the
    #: monthly term adds with coefficient 1 (the baseline).  Numeric shares, e.g.
    #: "usgs=0.27,cdec=0.54,uf=0.19": each family's share of the loss
    #: (renormalized over the families present, entities equal within a
    #: family); the daily term is scaled by the daily families' shares and
    #: the monthly term by the monthly family's, and checkpoint selection is
    #: the same share-weighted mean of the family means.  The CalSim3 arcs
    #: (calsim_monthly, key "calsim") are a second monthly family with their own
    #: monthly term, denominator and frozen scale.  Ignored outside the
    #: multi-timescale domain.
    mt_family_weight: str = "none"
    #: how a chunk's loss normalizes the family-weighted entities (only with
    #: mt_family_weight shares): "present" (default) divides by the
    #: weight of the entities scored in THAT chunk, so a chunk holding one
    #: family gives it the whole daily term — on the 95-entity set the USGS
    #: creeks train WY1950-84 and CDEC/uf from WY1985, and nominal shares
    #: 0.216/0.558/0.226 come out ~0.57/0.34/0.10 over the chunks; "all"
    #: divides by the weight of EVERY entity of the run (the monthly term by
    #: the number of monthly entities), so each entity always carries
    #: share_f / n_f and a family's term scales with how many of its entities
    #: a chunk holds — summed over the chunks the shares hold as loss
    #: COEFFICIENTS, approximately (each family's total scales with its mean
    #: scored chunks per entity: 95-entity set 0.234/0.546/0.220).  They are
    #: not the families' shares of the loss or gradient: the monthly NNSE runs
    #: ~10x smaller per entity than the daily NNSE + log + var terms.
    mt_share_norm: str = "present"
    #: checkpoint-selection shares, when they must differ from the loss shares
    #: (numeric mt_family_weight only): "area" = the footprint-area shares of
    #: the run's families (the sum of each family's registry area_mi2 over the
    #: run's entities, renormalized) — the footprint-area rule, kept
    #: for selection when the loss shares are re-solved so their REALIZED
    #: coefficient shares hit those area shares; or numeric shares in the
    #: mt_family_weight syntax.  "" (default) = the mt_family_weight shares.
    mt_select_weight: str = ""
    #: FROZEN per-family loss scale (needs mt_share_norm="all"): "usgs=a,cdec=b,
    #: uf=c" = each family's per-entity chunk loss per unit coefficient (the
    #: chunk_log's sum l_f / sum c_f) at a REFERENCE state.  Every family term
    #: is multiplied by kappa_f = Lbar / L_ref_f^p, Lbar = sum_f share_f
    #: L_ref_f^p (p = mt_loss_ref_power; only the ratios of the values matter),
    #: fixed for the whole run: the objective stays linear.  p = 1: at the
    #: reference each family's share of the LOSS equals its share (the total
    #: loss there unchanged) — but not of the gradient: per unit loss the
    #: monthly NNSE gradient runs 1.4-1.9x the daily one at trained states, and
    #: p = 0.5 is what matches the optimizer-step shares there.  Selection keeps
    #: the nominal shares (or mt_select_weight's).  "" (default) = off,
    #: byte-identical.
    mt_loss_ref: str = ""
    #: the exponent p of mt_loss_ref (0 < p <= 1) — REQUIRED with mt_loss_ref
    #: (no default: p = 1 and p = 0.5 differ by 2x in the monthly weight), None
    #: without it
    mt_loss_ref_power: float | None = None

    # -- parameter net ---------------------------------------------------------
    hidden: int = 64
    embed: int = 32
    dropout: float = 0.0
    #: the head's mapping, per run: ``log_space_params`` are mapped in log space besides
    #: ``LOG_SPACE_PARAMS``, and ``param_bounds`` (``{name: (lo, hi)}``, physical units)
    #: replaces a parameter's ``BOUNDS`` box and may WIDEN it.  Both are written into the
    #: net's ``_lo``/``_hi``/``_is_log`` buffers, so a checkpoint carries them.  () / {} =
    #: the mapping of BOUNDS and LOG_SPACE_PARAMS (default).
    log_space_params: tuple[str, ...] = ()
    param_bounds: dict = field(default_factory=dict)
    #: weight of the soft penalty on the head pre-activations beyond |6|
    #: (``parameter_net.logit_penalty``: the mean of (|z| - 6)^2 over the rows and outputs),
    #: added to each live chunk's loss (not to the logged loss).  It keeps the outputs off the
    #: sigmoid tails, where a cell's gradient vanishes.  0.0 = off (default).
    logit_penalty: float = 0.0
    #: LEARN the Snow-17 rain/snow threshold PXTEMP per cell (otherwise the GA
    #: constant 0 degC of FIXED_PARAMS).  A separate zero-init head emits PXTEMP
    #: inside ``pxtemp_box`` (exactly 0 at init, so the untrained forward is the
    #: fixed-threshold one); the physics keeps the reference HARD split — forward
    #: values unchanged — and trains the threshold through a straight-through
    #: sigmoid surrogate of width ``pxtemp_tau`` degC.
    pxtemp_learn: bool = False
    pxtemp_box: tuple[float, float] = (-1.0, 3.0)
    pxtemp_tau: float = 1.0
    #: ET: "sac" = the SAC-SMA ET cascade E1-E5; "noah" = the Noah-lite ET
    #: (physics.et_noah.noah_lite_et: bare soil + canopy on the observed green fraction, one
    #: learned exponent soil_chi), which needs the observed veg_frac and LAI.  Noah-lite
    #: replaces only the reference E1-E3 withdrawals: the rest of the reference ET block
    #: runs after them as in ``sma._sacsma_core`` (the upper free -> tension rebalance, the
    #: lower free -> tension resupply ``rserv`` and the ADIMP ET(5)), with Noah's
    #: upper-tension withdrawal as ET1.
    et_mode: str = "sac"
    #: the PET of the Noah-lite ET: "hamon" or "priestley_taylor"
    noah_pet: str = "hamon"
    #: the PET of the SAC-SMA ET: "hamon" or "priestley_taylor" (the energy-based PET of
    #: physics.et_noah.potential_et_priestley_taylor; needs per-cell tmin/tmax)
    sac_pet: str = "hamon"
    lr: float = 1e-3
    lr_min: float = 1e-5        # cosine-annealed floor
    lr_warmup_epochs: int = 4   # linear warmup protects the GA-prior init
    weight_decay: float = 1e-5
    grad_clip: float = 12.0
    n_epochs: int = 120
    cal_start: str = "1988-10-01"   # WY1989 start (cal end = cdec15.CAL_END)
    #: No-grad spinup cold start.  Ten water years ahead of the cal window is
    #: enough because the window spans the record-wet WY1982-83, which clamps
    #: even the big arid-basin LZ tension stores at capacity and so resets the
    #: cold-start error EXACTLY (lztwc is the one multi-year memory — ~7-yr ET
    #: drawdown, no clamp in ordinary years; everything else converges inside
    #: 5 yr).  Measured vs the full prefix (training convention, trained
    #: params) on the worst-memory basins: MIL/NHG exact (KGE 1.000000,
    #: max|dQ| 5e-5), arid ISB/SCC keep <= 69 mm of lztwc residual but still
    #: clear the parity bar (KGE >= 0.999975, max|dQ| <= 3e-3 mm/day); a 5-yr
    #: window FAILS it (MIL 0.99933, d(lztwc) 363 mm).  Clamped to the record
    #: start — set <= "1915-01-01" for the exact frozen full-prefix convention.
    #: The trainer always spins up this way (the multifamily domain: the ten water
    #: years before its envelope), so it needs forcing before the window.
    spinup_start: str = "1978-10-01"
    #: the evaluators' cycle spinup (:mod:`sacsma.dpl.spinup`, ``sacsma.engine.simulate``):
    #: the window's own first ``spinup_years`` years looped ``spinup_passes`` times from
    #: the cold start
    spinup_years: int = 10
    spinup_passes: int = 20
    #: > 0: CUDA-graph ACTIVATION RECOMPUTE — ONE captured graph of this many days
    #: (fwd + bwd) replayed over every training window (graphs.RecomputeTrainWindow):
    #: the forward keeps only the segment-boundary states and the backward re-runs
    #: each segment from its stored state, so graph memory is one segment's whatever
    #: the window length, for about one extra forward per segment (days past the
    #: last whole segment run eagerly); 73 divides 365 and 730.  0 = no train graph
    #: (the training windows run eagerly).
    graph_recompute_days: int = 73
    eval_every: int = 2            # full-cal no-grad KGE selection cadence
    patience: int = 10              # early stop after this many stale selections
    #: the selection scored EVERY epoch on the CPU engine by a separate process
    #: (:mod:`sacsma.dpl.select_cpu`) while the GPU trains, in place of the GPU selection
    #: pass: ``patience`` then counts stale epochs, and the decisions trail the training
    #: by the scoring time.  False = the GPU pass every ``eval_every`` epochs (default).
    select_cpu: bool = False
    #: numerics-neutral training diagnostics: chunk_log.csv (per chunk: entities
    #: and loss by family and the pre-clip gradient norm), eval_terms.csv (selection epochs:
    #: the eval-mode chunk loss by family and term) and per-epoch net snapshots
    #: with an EMA shadow (``ema_decay`` per optimizer step; restarted from the
    #: net on a resume) under checkpoints/snapshots/.
    diagnostics: bool = False
    ema_decay: float = 0.995
    #: CUDA-graph capture of the day-stepped pipeline (eager is dispatch-bound:
    #: ~300 tiny kernels/day).  Falls back to eager on CPU or capture failure.
    use_cuda_graphs: bool = True
    nograd_window: int = 256        # replay window for spinup/selection streaming
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_inc < 1:
            raise ValueError("n_inc must be >= 1")
        if self.et_mode not in ("sac", "noah"):
            raise ValueError(f"et_mode {self.et_mode!r}")
        if self.noah_pet not in ("hamon", "priestley_taylor"):
            raise ValueError(f"noah_pet {self.noah_pet!r}")
        if self.sac_pet not in ("hamon", "priestley_taylor"):
            raise ValueError(f"sac_pet {self.sac_pet!r}")
        if self.mt_family_weight != "none":
            family_shares(self.mt_family_weight)   # raises on a bad spec
        if self.mt_share_norm not in ("present", "all"):
            raise ValueError(f"mt_share_norm {self.mt_share_norm!r}: 'present' or 'all'")
        if self.mt_share_norm == "all" and self.mt_family_weight == "none":
            raise ValueError("mt_share_norm='all' normalizes family SHARES — it needs "
                             "numeric mt_family_weight shares")
        if self.mt_select_weight:
            if self.mt_family_weight == "none":
                raise ValueError("mt_select_weight sets the SELECTION shares of a numeric "
                                 "mt_family_weight run — it needs numeric shares")
            if self.mt_select_weight != "area":
                family_shares(self.mt_select_weight)    # raises on a bad spec
        if self.mt_loss_ref:
            family_loss_refs(self.mt_loss_ref)          # raises on a bad spec
            if self.mt_share_norm != "all":
                # under "present" a chunk's daily denominator is its own scored
                # weight (kappa included), cancelling kappa in every chunk that
                # scores one daily family — the scale would not be the one logged
                raise ValueError("mt_loss_ref scales the family terms of the fixed-"
                                 "denominator loss — it needs mt_share_norm='all'")
            if self.mt_loss_ref_power is None:
                raise ValueError("mt_loss_ref needs an explicit mt_loss_ref_power "
                                 "(0 < p <= 1)")
            self.mt_loss_ref_power = float(self.mt_loss_ref_power)
            if not 0.0 < self.mt_loss_ref_power <= 1.0:
                raise ValueError(f"mt_loss_ref_power {self.mt_loss_ref_power} outside (0, 1]")
        elif self.mt_loss_ref_power is not None:
            raise ValueError("mt_loss_ref_power has no effect without mt_loss_ref")
        if self.nograd_window < 1:
            raise ValueError(f"nograd_window {self.nograd_window} < 1")
        if not 0 <= self.graph_recompute_days <= 366:
            raise ValueError(f"graph_recompute_days {self.graph_recompute_days} "
                             "outside [0, 366]")
        if not math.isfinite(self.var_huber_cap):
            raise ValueError(f"var_huber_cap {self.var_huber_cap} must be finite (<= 0 = uncapped)")
        if not self.grad_clip > 0.0:
            raise ValueError(f"grad_clip {self.grad_clip} must be > 0")
        if not 90 <= self.shape_min_days <= 366:
            raise ValueError(f"shape_min_days {self.shape_min_days} outside [90, 366]")
        if self.spinup_years < 1 or self.spinup_passes < 1:
            raise ValueError("spinup_years and spinup_passes must be >= 1")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay {self.ema_decay} outside (0, 1)")
        self.log_space_params = tuple(self.log_space_params)
        bad = [p for p in self.log_space_params if p not in FREE_PARAMS]
        if bad or len(set(self.log_space_params)) != len(self.log_space_params):
            raise ValueError(f"log_space_params {self.log_space_params}: learned parameters, "
                             f"each once ({', '.join(FREE_PARAMS)})")
        bounds = {}
        for name, lohi in dict(self.param_bounds).items():
            lo, hi = (float(v) for v in lohi)
            if name not in FREE_PARAMS or not (math.isfinite(lo) and math.isfinite(hi)
                                               and lo < hi):
                raise ValueError(f"param_bounds: {name} ({lo}, {hi}) must be a learned "
                                 "parameter with finite lo < hi")
            bounds[name] = (lo, hi)
        self.param_bounds = bounds
        for p in (*self.log_space_params, *bounds):
            if ((p in LOG_SPACE_PARAMS or p in self.log_space_params)
                    and {**BOUNDS, **bounds}[p][0] <= 0.0):
                raise ValueError(f"{p}: a log-space parameter needs a positive floor")
        if not (math.isfinite(self.logit_penalty) and self.logit_penalty >= 0.0):
            raise ValueError(f"logit_penalty {self.logit_penalty} must be finite and >= 0")
        plo, phi = (float(v) for v in self.pxtemp_box)
        if not (math.isfinite(plo) and math.isfinite(phi) and plo <= 0.0 <= phi
                and plo < phi):
            # the zero-init head starts at exactly 0 degC (the GA constant), so the
            # box must contain it
            raise ValueError(f"pxtemp_box {self.pxtemp_box} must satisfy lo <= 0 <= hi, "
                             "lo < hi")
        self.pxtemp_box = (plo, phi)
        if not (math.isfinite(self.pxtemp_tau) and self.pxtemp_tau > 0.0):
            raise ValueError(f"pxtemp_tau {self.pxtemp_tau} must be finite and > 0")
        if isinstance(self.obs_mask, str):      # tolerate a bare string
            self.obs_mask = tuple(s for s in self.obs_mask.split(",") if s)
        self.obs_mask = tuple(str(s) for s in self.obs_mask)
        from datetime import date as _date
        from datetime import timedelta as _timedelta
        for s in self.obs_mask:
            eid, sep, day = s.partition("|")
            try:
                _date.fromisoformat(day)
            except ValueError:
                sep = ""
            if not (sep and eid):
                raise ValueError(f"obs_mask entry {s!r}: expected 'entity_id|YYYY-MM-DD'")
        if len(set(self.obs_mask)) != len(self.obs_mask):
            raise ValueError("obs_mask has duplicate entries")
        if isinstance(self.holdout_wy, str):    # tolerate "1976-1985"
            self.holdout_wy = tuple(int(v) for v in self.holdout_wy.split("-") if v)
        self.holdout_wy = tuple(int(v) for v in self.holdout_wy)
        if self.holdout_wy and not (len(self.holdout_wy) == 2
                                    and 1950 <= self.holdout_wy[0] <= self.holdout_wy[1] <= 2018):
            raise ValueError(f"holdout_wy {self.holdout_wy}: expected (first, last) water "
                             "years with 1950 <= first <= last <= 2018")
        if self.uf_train_start:
            d = _date.fromisoformat(self.uf_train_start)
            if d.day != 1 or d < _date(1949, 10, 1):
                raise ValueError(f"uf_train_start {self.uf_train_start!r}: a first of the "
                                 "month on or after 1949-10-01 (the envelope start)")
            self.uf_train_start = d.isoformat()
        if self.train_window:
            a, b = (_date.fromisoformat(v) for v in self.train_window)
            if not (len(self.train_window) == 2 and a.day == 1
                    and (b + _timedelta(days=1)).day == 1
                    and _date(1949, 10, 1) <= a < b <= _date(2018, 12, 31)):
                raise ValueError(f"train_window {self.train_window}: (first, last) days of "
                                 "whole calendar months inside 1949-10-01..2018-12-31")
            self.train_window = (a.isoformat(), b.isoformat())
            if self.uf_train_start:
                raise ValueError("uf_train_start extends the uf targets that train_window "
                                 "cuts: use one or the other")
        if self.calsim_arcs not in ("none", "train_default"):
            raise ValueError(f"calsim_arcs {self.calsim_arcs!r}: 'none' or 'train_default'")
        from datetime import date
        if date.fromisoformat(self.spinup_start) >= \
                date.fromisoformat(self.cal_start):
            raise ValueError(
                f"spinup_start {self.spinup_start!r} must be before "
                f"cal_start {self.cal_start!r}")

    def physics(self):
        """The run's physics as a :class:`sacsma.engine.Physics` (the learned step)."""
        from ..engine import Physics

        noah = self.et_mode == "noah"
        return Physics(et="noah_lite" if noah else "sac",
                       pet=self.noah_pet if noah else self.sac_pet, learned=True,
                       n_inc=self.n_inc, fracp_floor=self.fracp_floor)


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


FAMILY_KEYS = {"usgs": "usgs_daily", "cdec": "cdec_daily", "uf": "uf_monthly",
               "calsim": "calsim_monthly"}
#: the monthly families: each is its own monthly NNSE term (own rows, share, denominator
#: and frozen scale) — the rest are daily
MONTHLY_FAMILIES = ("uf_monthly", "calsim_monthly")


def family_shares(spec: str) -> dict[str, float] | None:
    """Parse a numeric ``mt_family_weight`` spec ("usgs=0.27,cdec=0.54,uf=0.19")
    into ``{family_id: share}`` summing to 1 over the families named; ``None``
    for "none".  Family keys may be the short names or the registry family ids;
    every share must be positive."""
    if spec == "none":
        return None
    shares: dict[str, float] = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError(f"mt_family_weight {spec!r}: expected "
                             "family=share items or 'none'")
        key, val = (t.strip() for t in item.split("=", 1))
        fam = FAMILY_KEYS.get(key, key)
        if fam not in FAMILY_KEYS.values():
            raise ValueError(f"mt_family_weight {spec!r}: unknown family "
                             f"{key!r} (usgs, cdec, uf, calsim)")
        if fam in shares:
            raise ValueError(f"mt_family_weight {spec!r}: {key!r} repeated")
        try:
            share = float(val)
        except ValueError as e:
            raise ValueError(f"mt_family_weight {spec!r}: share {val!r}") from e
        if not (math.isfinite(share) and share > 0.0):
            raise ValueError(f"mt_family_weight {spec!r}: shares must be finite and > 0")
        shares[fam] = share
    tot = sum(shares.values())
    return {f: v / tot for f, v in shares.items()}


def family_loss_refs(spec: str) -> dict[str, float]:
    """Parse an ``mt_loss_ref`` spec ("usgs=a,cdec=b,uf=c,calsim=d") into
    ``{family_id: L_ref}`` — each family's per-entity chunk loss per unit
    coefficient at the reference state (unnormalized; only the ratios reach the
    loss, through ``kappa_f = Lbar / L_ref_f^p``)."""
    refs: dict[str, float] = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError(f"mt_loss_ref {spec!r}: expected family=level items")
        key, val = (t.strip() for t in item.split("=", 1))
        fam = FAMILY_KEYS.get(key, key)
        if fam not in FAMILY_KEYS.values():
            raise ValueError(f"mt_loss_ref {spec!r}: unknown family {key!r} "
                             "(usgs, cdec, uf, calsim)")
        if fam in refs:
            raise ValueError(f"mt_loss_ref {spec!r}: {key!r} repeated")
        try:
            v = float(val)
        except ValueError as e:
            raise ValueError(f"mt_loss_ref {spec!r}: level {val!r}") from e
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"mt_loss_ref {spec!r}: levels must be finite and > 0")
        refs[fam] = v
    return refs

