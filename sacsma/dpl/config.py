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
    "soil_chi":  (0.5, 2.5),     # bare-soil evap nonlinearity (Ek 2003 chi; was a
                                 # fixed 2.0 — LEARNED so sparse-veg dry basins can
                                 # lift bare-soil ET while wet basins keep it high)
}

#: Canopy parameters mapped in log space (rcmin spans ~2 decades).
CANOPY_LOG_PARAMS: frozenset[str] = frozenset({"rcmin"})

CANOPY_PARAMS: tuple[str, ...] = tuple(CANOPY_BOUNDS)

#: Canopy structure SUPPLIED FROM OBSERVATION (LANDFIRE EVC cover fraction and
#: the MODIS/Landsat LAI climatology), NOT learned — these two params scale ET
#: magnitude almost 1:1, and a uniform midpoint init (veg_frac 0.5, lai 3.25 vs
#: observed ~0.36 / ~1.3) drove the dry-basin ET-partition failure.  Pinning
#: them per-cell fixes the magnitude and removes the low-signal overfit; the net
#: then learns only the unobservable physiology below.  ``veg_frac`` is static
#: per cell; ``lai`` is the per-cell SEASONAL daily climatology (threaded like a
#: forcing).  Neither is ever a net output or part of ga_optimum.
CANOPY_OBSERVED_PARAMS: tuple[str, ...] = ("veg_frac", "lai")

#: The physiology parameters the net actually learns (unobservable): minimum
#: stomatal resistance, the radiation/VPD Jarvis coefficients, wilting point,
#: root split, and the lower->upper redistribution rate.  Order follows
#: CANOPY_BOUNDS insertion (so head columns stay stable).
CANOPY_LEARNED_PARAMS: tuple[str, ...] = tuple(
    p for p in CANOPY_PARAMS if p not in CANOPY_OBSERVED_PARAMS)

#: Noah-LITE (``canopy_lite=True``) learned set: ``soil_chi`` ALONE.  A
#: streamflow-only calibration cannot identify the full 7-param ET partition —
#: the three Jarvis resistance params (rcmin/rgl/hs) collapse into one
#: multiplicative factor confounded with Kpet, and froot/redist_k merely
#: re-implement SAC's own UZ<->LZ machinery.  Lite keeps the ONE identifiable
#: knob (the moisture-limiter exponent) and drops the rest (dropped params are
#: pinned at the physical constants in et_noah: ``_LITE_WILT``, ``_LITE_FROOT``;
#: the Jarvis transpiration + interception + redistribution terms are removed
#: entirely).  veg_frac + lai stay pinned from observation (0 DOF).
CANOPY_LITE_LEARNED: tuple[str, ...] = ("soil_chi",)

#: SAC parameters eligible for the climate-state dynamic response.  Kpet ONLY:
#: it is the ET-volume knob (`pet = Kpet * potential`) and already accepts a
#: per-day (N,T) field via forward._seasonal — no new physics threading.  The
#: recessions (uzk/lzpk/lzsk) are deliberately excluded: making them SEASONAL
#: already hurt (only seasonal Kpet helped), so a dynamic response would too.
DYNAMIC_SAC_PARAMS: tuple[str, ...] = ("Kpet",)


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
    #: per-chunk BIAS penalty (mean-ratio beta - 1)^2 — the KGE beta term the
    #: MSE/NNSE loss lacks (it penalizes correlation + variance but NOT volume
    #: bias, so the optimizer can trade wet-basin over-evaporation for dry-basin
    #: gains invisibly).  0.0 disables (default = byte-identical baseline); a
    #: chunk mean over ~366 days is a stable statistic, like the std-ratio above.
    bias_loss_lambda: float = 0.0

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
    #: ET/SWE-observation auxiliary losses (multi-product, cal-window ONLY,
    #: leakage-safe; obs are training regularizers, NEVER a selection criterion).
    #: Redesigned 2026-07-13 after the raw monthly central pull degraded
    #: streamflow (it pulled LEVEL, which the products disagree on 38-85%, as
    #: hard as PHASE, which they agree on to ~0.5 month).  Three separable terms:
    #: * ``et_loss_lambda`` — ET seasonal-SHAPE pull: model's monthly ET is
    #:   normalized by its own masked-month mean and pulled toward the products'
    #:   consensus NORMALIZED cycle (inverse-variance; sigma inflated by the
    #:   products' interannual shape spread, floored at ``shape_sigma_floor``).
    #:   Level-blind by construction — no volume fight with streamflow.
    #: * ``et_level_lambda`` — ET volume envelope HINGE: zero force inside the
    #:   product min-max total, quadratic outside (width-scaled).  Catches the
    #:   arid basins whose ET sits ABOVE every product without pulling anyone
    #:   toward the (untrustworthy) ensemble-mean volume.
    #: * ``swe_loss_lambda`` — SWE seasonal-SHAPE pull (same normalized form) on
    #:   the model's monthly-MEAN Snow-17 SWE vs 4 products; ~snow-free basins
    #:   masked out; NO SWE level term ever (85% peak-magnitude disagreement).
    #: 0.0 each disables; all zero => the obs path is absent = byte-identical.
    et_loss_lambda: float = 0.0
    et_level_lambda: float = 0.0
    swe_loss_lambda: float = 0.0
    shape_sigma_floor: float = 0.1   # absolute floor on the normalized-cycle sigma
    #: replace the level hinge's product min-max envelope with a WATER-BALANCE
    #: anchor: per-basin annual ET = cal-window mean(P) - mean(Q_obs) over the
    #: gage-covered days, spread over months by the products' consensus cycle,
    #: hinged at anchor*(1 -/+ band).  Observed and flow-consistent — far
    #: tighter than the product bracket (which spans up to 72% of Q in the arid
    #: basins and measurably could not stop the shape term's level leak: the
    #: 2026-07-15 seasonal-Kpet arm drifted annual ET 10-17% inside it).
    #: 0 = off (product envelope, exact prior behavior).
    et_anchor_band: float = 0.0
    #: restrict the ET obs target to a SUBSET of the 5 training products
    #: (e.g. ("fluxcom",) — the A2 single-product arm; the 2026-07-15 pre-
    #: registered pick minimizes RMS |annual product ET - (P-Q_obs)| over the
    #: 15 basins).  () = all 5 (consensus, prior behavior).  A SINGLE product
    #: has no cross-product spread — sigma degrades to the interannual spread
    #: floored at ``shape_sigma_floor`` (an intrinsically harder pull) and the
    #: min-max level envelope is degenerate, so one product REQUIRES the P-Q
    #: anchor (``et_anchor_band`` > 0) as the level constraint.
    et_products: tuple[str, ...] = ()
    #: warm-start checkpoint path: the net's weights are loaded strict=False
    #: BEFORE training (heads absent from the donor — e.g. a fresh seasonal
    #: head — keep their zero-init, so the run starts EXACTLY at the donor's
    #: parameter field).  Optimizer/scheduler start fresh; combine with a low
    #: lr for the fine-tune regime (obs losses select within the donor's flow-
    #: optimal plateau instead of fighting a from-scratch descent).  "" = off.
    init_from: str = ""

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
    #: PER-PARAMETER harmonic cap as a FRACTION of each seasonal param's bound
    #: range: |a_sin|,|a_cos| <= seasonal_amp_frac*(hi-lo).  Supersedes the flat
    #: ``seasonal_amp`` so a mixed set (Kpet + melt factors, whose native ranges
    #: differ by ~2x) gets a comparable RELATIVE day-of-year swing rather than the
    #: same additive one (0.10 -> Kpet +/-0.21, MFMAX/MFMIN +/-0.50, MBASE +/-0.50).
    seasonal_amp_frac: float = 0.10
    #: parameters made time-varying via a CLIMATE-STATE response (generalizes the
    #: seasonal harmonic): the net emits a bounded coeff b per param and the
    #: physics reconstructs param(t) = clamp(base + b*state(t), lo, hi), where
    #: state(t) is a cal-standardized rolling-precip wetness index.  SAC params
    #: must be in DYNAMIC_SAC_PARAMS (already (N,T)-capable via the seasonal path);
    #: canopy params in CANOPY_LEARNED_PARAMS (e.g. soil_chi).  Empty = static
    #: (default).  Zero-init coeffs => exact static field at init (clean superset).
    dynamic_params: tuple[str, ...] = ()
    dynamic_amp: float = 0.5     # tanh cap on |b| (state-response amplitude)
    dynamic_window: int = 365    # rolling-mean window (days) for the wetness state
    #: re-foot the basin aggregation (``dom.W``) onto the CalSim3 catchment
    #: geometry: each cell is area-weighted by its overlap fraction with the
    #: basin's CalSim3 catchment, so out-of-catchment cells drop and boundary
    #: cells down-weight.  Corrects the coarse 1/16-deg grid's systematic
    #: footprint over-reach (+9..+66% vs the true catchment).  Only basins with a
    #: CalSim3 catchment are re-footed (the 4 Tulare/Kern basins keep their full
    #: footprint).  Opt-in; default False => the exact area_weight aggregation.
    calsim_footprint: bool = False
    #: ET scheme: "sac" = the frozen Hamon PET (default; scorable through
    #: run_basin).  "noah" = the Noah canopy-resistance ET (et_noah.py) — NEW
    #: physics, NOT scorable through run_basin (skill via score_noah_torch).
    #: Requires ``canopy=True`` (the net's canopy head) and per-cell tmin/tmax.
    et_mode: str = "sac"
    #: Noah potential-ET source: "hamon" = the temperature-only Hamon PET the
    #: canopy params modulate (v1-v4; total ET <= Kpet*Hamon = a low ceiling that
    #: makes Noah under-extract vs SAC); "priestley_taylor" = an energy-based PET
    #: from Bristow-Campbell net radiation (FAO-56 Rn) — lifts the ceiling and
    #: removes the Kpet/canopy ET-scaling redundancy.  Only used when et_mode="noah".
    noah_pet: str = "hamon"
    #: PET source for the PLAIN SAC ET path (et_mode="sac"): "hamon" (default,
    #: the frozen temperature PET) or "priestley_taylor" — the energy-based PET
    #: (Bristow-Campbell Rn, from et_noah.potential_et_priestley_taylor) driving
    #: the frozen SAC ET cascade directly, with NO Noah canopy module.  A warming-
    #: robust ET without the canopy-parameter identifiability problems of Noah.
    sac_pet: str = "hamon"
    #: Priestley-Taylor refinements (any PT PET — sac_pet OR noah_pet =
    #: "priestley_taylor"; both default 0 => the plain fixed-albedo / Tdew=Tmin
    #: PT).  ``pt_snow_albedo``>0
    #: raises the PT net-radiation albedo toward this value over snow (driven by
    #: the model's own Snow-17 SWE, so PET collapses under a pack — a bright-snow
    #: value is ~0.5-0.7); ``pt_dewpoint_depression``>0 lowers the dewpoint up to
    #: this many degC below Tmin in ARID air (scaled by the diurnal range) so the
    #: net longwave loss is not under-counted in dry basins (FAO-56 arid Tdew).
    #: Neither is absorbable by the per-HRU Kpet (both are seasonal/spatial SHAPE
    #: corrections), unlike a global albedo/alpha which Kpet would just rescale.
    pt_snow_albedo: float = 0.0
    pt_dewpoint_depression: float = 0.0
    #: emit the learned CANOPY_LEARNED_PARAMS from a canopy head (needed for
    #: et_mode="noah"; veg_frac + seasonal lai come from observation, not here).
    canopy: bool = False
    #: give the canopy head its OWN encoder (decoupled from the SAC trunk) so
    #: the weak dry-basin canopy signal cannot corrupt the GA-prior SAC pathway
    #: through a shared embedding.  Only used when canopy=True.
    canopy_separate_trunk: bool = True
    #: Noah-LITE ET: the minimal, identifiable rebuild — AET = ed_bare +
    #: et_canopy on pinned veg with a SINGLE learned exponent (soil_chi); the
    #: Jarvis resistance (rcmin/rgl/hs), the learned root split (froot) and the
    #: UZ<->LZ redistribution (redist_k) are dropped, and the canopy head emits
    #: only CANOPY_LITE_LEARNED off the SHARED trunk (no separate encoder).
    #: Requires et_mode="noah"; noah_pet still selects Hamon | Priestley-Taylor.
    canopy_lite: bool = False
    lr: float = 1e-3
    lr_min: float = 1e-5        # cosine-annealed floor
    lr_warmup_epochs: int = 3   # linear warmup protects the GA-prior init
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    n_epochs: int = 60
    spinup_refresh_every: int = 1   # no-grad spinup every k epochs
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
    spinup_start: str = "1978-10-01"
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
        if self.et_mode not in ("sac", "noah"):
            raise ValueError(f"et_mode {self.et_mode!r}")
        if self.noah_pet not in ("hamon", "priestley_taylor"):
            raise ValueError(f"noah_pet {self.noah_pet!r}")
        if self.sac_pet not in ("hamon", "priestley_taylor"):
            raise ValueError(f"sac_pet {self.sac_pet!r}")
        if min(self.et_loss_lambda, self.et_level_lambda,
               self.swe_loss_lambda, self.shape_sigma_floor) < 0.0:
            raise ValueError("obs-loss lambdas / shape_sigma_floor must be >= 0")
        if not 0.0 <= self.et_anchor_band < 1.0:
            raise ValueError("et_anchor_band must be in [0, 1)")
        if self.et_anchor_band > 0.0 and self.et_level_lambda <= 0.0:
            raise ValueError(
                "et_anchor_band re-targets the level hinge — it needs "
                "et_level_lambda > 0 to have any effect")
        if isinstance(self.et_products, str):   # tolerate a bare CLI string
            self.et_products = tuple(p for p in self.et_products.split(",") if p)
        if (len(self.et_products) == 1
                and (self.et_loss_lambda > 0.0 or self.et_level_lambda > 0.0)
                and self.et_anchor_band <= 0.0):
            raise ValueError(
                "a single ET product has a degenerate min-max level envelope — "
                "et_products of length 1 requires the P-Q anchor "
                "(et_anchor_band > 0)")
        from datetime import date
        if date.fromisoformat(self.spinup_start) >= \
                date.fromisoformat(self.cal_start):
            raise ValueError(
                f"spinup_start {self.spinup_start!r} must be before "
                f"cal_start {self.cal_start!r}")
        if not 0.0 <= self.pt_snow_albedo < 1.0:
            raise ValueError("pt_snow_albedo must be in [0, 1)")
        if self.pt_dewpoint_depression < 0.0:
            raise ValueError("pt_dewpoint_depression must be >= 0")
        pt_active = (
            (self.et_mode == "sac" and self.sac_pet == "priestley_taylor")
            or (self.et_mode == "noah" and self.noah_pet == "priestley_taylor"))
        if (self.pt_snow_albedo > 0.0 or self.pt_dewpoint_depression > 0.0) \
                and not pt_active:
            raise ValueError(
                "pt_snow_albedo / pt_dewpoint_depression apply only to a "
                "Priestley-Taylor PET (sac_pet or noah_pet = 'priestley_taylor')")
        if self.dynamic_params:
            allowed = set(DYNAMIC_SAC_PARAMS) | set(CANOPY_LEARNED_PARAMS)
            bad = [p for p in self.dynamic_params if p not in allowed]
            if bad:
                raise ValueError(
                    f"dynamic_params {bad} not in {sorted(allowed)} "
                    f"(SAC dynamic limited to the (N,T)-capable set)")
        if self.et_mode == "noah":
            self.canopy = True   # the canopy head is required to emit CANOPY_PARAMS
        if self.canopy_lite:
            if self.et_mode != "noah":
                raise ValueError("canopy_lite requires et_mode='noah'")
            # lite emits only soil_chi off the shared trunk (the separate canopy
            # encoder existed to protect the SAC pathway from the 6 dropped params)
            self.canopy_separate_trunk = False


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
