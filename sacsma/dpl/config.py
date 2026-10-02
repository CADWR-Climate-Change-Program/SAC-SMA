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

import math

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
    #: per-chunk variance-matching penalty on alpha = std-ratio: (alpha - 1)^2 up
    #: to |alpha - 1| = var_huber_cap, linear beyond, skipped for a basin-chunk
    #: under var_gate_frac of the basin's record variance — counters the
    #: squared-error variance damping (alpha -> r); NOT chunked KGE.
    var_loss_lambda: float = 1.0
    #: the gate: a chunk's observed variance must exceed this share of the basin's
    #: full-record variance (an absolute floor of 1e-8 stays fixed in loss.py) for
    #: the term to apply.  0.0 = no gate beyond the floor.
    var_gate_frac: float = 1e-3
    #: the Huber cap on |alpha - 1|: quadratic up to it, linear beyond; <= 0 keeps
    #: (alpha - 1)^2 throughout.  Together, 0.0 and 0.0 are the loss the runs
    #: before 2026-09 (the 15cdec canonical set) trained with.
    var_huber_cap: float = 1.0
    #: per-chunk BIAS penalty (mean-ratio beta - 1)^2 — the KGE beta term the
    #: MSE/NNSE loss lacks (it penalizes correlation + variance but NOT volume
    #: bias, so the optimizer can trade wet-basin over-evaporation for dry-basin
    #: gains invisibly).  0.0 disables (default = byte-identical baseline); a
    #: chunk mean over ~366 days is a stable statistic, like the std-ratio above.
    bias_loss_lambda: float = 0.0
    #: per-chunk summer-RECESSION timing penalty: over 1 Jul - 30 Sep, the mean
    #: squared difference of the normalized cumulative flow (share of the window's
    #: volume passed by each day), sim vs obs — the recession shape, blind to volume,
    #: with no pull on winter or the flood peaks (a whole-year curve pulled winter
    #: volume against the other terms at most Sierra basins).  Daily entities only
    #: (monthly rows have no daily obs).  0.0 disables (default = byte-identical
    #: baseline).
    timing_loss_lambda: float = 0.0
    #: per-chunk FLOOD-PEAK penalty: the mean of the chunk's top peak_loss_frac
    #: valid days, sim vs obs (each sorted on its own — the flow-duration curve's
    #: high segment), their difference over the basin's RECORD mean of the yearly
    #: observed top mean, Huber-capped at 1 — flood years carry it, drought years
    #: weigh little.  Daily entities only.  0.0 disables (default = byte-identical
    #: baseline).
    peak_loss_lambda: float = 0.0
    #: the share of a chunk's valid days the peak term averages (0.02: 7 days a year)
    peak_loss_frac: float = 0.02
    #: the timing and peak terms score a basin-chunk only with at least this many
    #: valid observed days (whole water years; the short envelope tail and sparse
    #: years drop out); their record constants come from the same years
    shape_min_days: int = 300
    #: the timing term skips a basin-chunk whose observed Jul-Sep mean flow is under
    #: this share of the basin's record mean flow (a dry summer has no recession to
    #: match); it also needs 60 valid Jul-Sep days
    timing_vol_gate: float = 0.05
    #: daily-target observations MASKED out of training and scoring, as
    #: ``"entity_id|YYYY-MM-DD"`` strings (the CLI reads them from a hand-edited
    #: CSV such as data/cdec_fnf/fnf_daily_mask.csv; the checkpoint carries the
    #: list, so evaluation masks the identical days).  Multi-timescale domain
    #: only.  Empty = the stores as-is (default).
    obs_mask: tuple[str, ...] = ()
    #: water years held out of EVERY family, as an inclusive ``(first, last)`` pair,
    #: e.g. (1976, 1985) (calsim_arcs.HOLDOUT_WY): their targets are NaN after the
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
    #: the CalSim3 rim-arc family (calsim_monthly) in the run: "train_default"
    #: appends the train_default arcs of data/calsim/arc_hierarchy.csv (tier A, own
    #: gauge record), in file order, as cs_<ARC> entities after the run's other
    #: entities, so a resume rebuilds the same order.  Multi-timescale domain only.
    #: "none" (default) = only the entities named or the 95 base entities.
    calsim_arcs: str = "none"

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
    #: multi-timescale family weighting (multifamily domain only): "none" =
    #: every valid daily entity weighs equally in the chunk mean and the
    #: monthly term adds with coefficient 1 (the baseline); "equal" = shares
    #: 1:1:1 over the families the run trains (three families: thirds, the
    #: daily term x 2/3 and the monthly x 1/3; a subset without a family
    #: renormalizes over the rest).  Numeric shares, e.g.
    #: "usgs=0.27,cdec=0.54,uf=0.19": each family's share of the loss
    #: (renormalized over the families present, entities equal within a
    #: family); the daily term is scaled by the daily families' shares and
    #: the monthly term by the monthly family's, and checkpoint selection is
    #: the same share-weighted mean of the family means ("equal": the plain
    #: mean of the family means).  The CalSim3 arcs (calsim_monthly, key
    #: "calsim") are a second monthly family with their own monthly term,
    #: denominator and frozen scale.  Guards against combining with
    #: adaptive_loss (both drive the same per-basin weight vector).  Ignored
    #: outside the multi-timescale domain.
    mt_family_weight: str = "none"
    #: how a chunk's loss normalizes the family-weighted entities (only with
    #: mt_family_weight shares or "equal"): "present" (default) divides by the
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
    #: run's entities, renormalized) — the rule dPL-95's shares were set by, kept
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
    #: warm-start checkpoint path: the net's weights are loaded strict=False
    #: BEFORE training (heads absent from the donor — e.g. a fresh seasonal
    #: head — keep their zero-init, so the run starts EXACTLY at the donor's
    #: parameter field).  The donor's feature standardization is reused, so
    #: the start is exact even when this run trains a different entity subset
    #: (--basins) than the donor.  Optimizer/scheduler start fresh; combine
    #: with a low lr for the fine-tune regime (obs losses select within the
    #: donor's flow-optimal plateau instead of fighting a from-scratch
    #: descent).  "" = off.
    init_from: str = ""
    #: ep0-donor gate action when the epoch-0 selection does not reproduce
    #: the donor's sel cal KGE (|d| > 1e-3): "warn" prints and continues;
    #: "abort" raises, for unattended warm starts.  The gate compares the
    #: selection scalar directly when the donor trained the same entity set
    #: under the same statistic, and otherwise recomputes the donor's own
    #: statistic over its entities from this run's per-entity KGE.
    init_gate: str = "warn"

    # -- parameter net (net-v2 knobs; defaults = the v1 architecture) --------
    hidden: int = 64
    embed: int = 32
    dropout: float = 0.1
    grouped_heads: bool = False     # per-physics-group output heads
    fourier_k: int = 0              # spatial Fourier feature order (0 = off)
    #: whether the cell's flow length to its basin outlet is one of the net's
    #: continuous static inputs (features.CONTINUOUS_STATICS).  On a domain of
    #: nested entities (multifamily) the same cell then gets a different
    #: parameter set in every entity it belongs to, so False keeps the
    #: parameter field per cell; the routing reads flowlen from the HRU table
    #: either way.  True = the 15cdec canonical checkpoints' feature set.
    flowlen_feature: bool = True
    #: per-parameter override of the head's bounds box, ``{name: (lo, hi)}`` in
    #: physical units, inside ``BOUNDS``; ``lo == hi`` pins the parameter (its
    #: sigmoid is scaled by zero, so the value is exact and its head gets no
    #: gradient).  Applied to the net's ``_lo``/``_hi`` buffers after every load,
    #: so a checkpoint carries it and evaluation needs nothing further.
    param_box: dict = field(default_factory=dict)
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
    #: LEARN the Snow-17 rain/snow threshold PXTEMP per cell (otherwise the GA
    #: constant 0 degC of FIXED_PARAMS).  A separate zero-init head emits PXTEMP
    #: inside ``pxtemp_box`` (exactly 0 at init, so the untrained forward is the
    #: fixed-threshold one); the physics keeps the reference HARD split — forward
    #: values unchanged — and trains the threshold through a straight-through
    #: sigmoid surrogate of width ``pxtemp_tau`` degC.  The exported per-HRU
    #: PXTEMP column runs as-is in the frozen run_basin (same hard split).
    pxtemp_learn: bool = False
    pxtemp_box: tuple[float, float] = (-1.0, 3.0)
    pxtemp_tau: float = 1.0
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
    #: Noah ET replaces only the reference E1-E3 withdrawals; with this set the
    #: rest of the reference ET block runs after them as in ``sma._sacsma_core``:
    #: the upper free -> tension rebalance, the lower free -> tension resupply
    #: (``rserv``) and the ADIMP ET(5), with Noah's upper-tension withdrawal as
    #: ET1.  Off (the runs so far), the external-ET path skips all three, which
    #: leaves the riparian ``riva`` channel ET as the only sink for lower free
    #: water.  Requires the Noah-lite path (canopy_lite).
    noah_sac_exchanges: bool = False
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
    #: how the trainer's no-grad spinup reaches the state at the cal-window start
    #: (:mod:`sacsma.dpl.spinup`).  "window": stream from ``spinup_start`` (the
    #: multifamily domain: the ten water years before its envelope) — needs forcing
    #: before the window.  "cycle": timing-independent — loop the window's own first
    #: ``spinup_years`` years from the cold start ``spinup_passes`` times (the
    #: trainer's first spinup; each later one continues from the previous state for
    #: at least ``spinup_warm_passes`` passes and until a pass moves no basin's
    #: annual flow by more than ``spinup_warm_tol``, at most ``spinup_passes``).  The
    #: multifamily evaluators score with "cycle" whatever the training mode.
    spinup_mode: str = "window"
    spinup_years: int = 10
    spinup_passes: int = 20
    spinup_warm_passes: int = 2
    spinup_warm_tol: float = 1e-3
    train_chunk_days: int = 366     # TBPTT chunk (fixed length; last chunk's
                                    # post-CAL_END days are NaN-masked in the loss)
    #: how the TBPTT chunks tile the calibration window.  "fixed": train_chunk_days
    #: each from the window start, so the boundary drifts ~0.75 day/yr and cuts a
    #: calendar month almost every year — a month split between chunks never
    #: reaches the monthly-flow term (29 of the 360 DWR months on the multifamily
    #: domain).  "water_year": one chunk per water year, 1 Oct to 1 Oct (365 or
    #: 366 days; the last runs to the record end), every chunk holding twelve
    #: complete months.  The segmented graphs capture the 365-day length and a
    #: leap year's last day continues eagerly from the graphed state; the
    #: whole-chunk graph path captures each length.  Needs train_chunk_days ==
    #: 366 and a window that starts on 1 Oct.
    chunk_grid: str = "fixed"
    #: how the gradient sees the state carried into a TBPTT chunk.  "absolute": the
    #: detached SAC contents (the gradient treats them as constants, so a larger
    #: capacity looks like free deficit at every chunk start — the 1-water-year
    #: truncation overweights the tension capacities' pull 5-170x against the full
    #: sequence).  "relative": each incoming SAC content c is carried as
    #: c * (cap / cap.detach()) — the same value (x/x == 1 exactly), but the backward
    #: holds the relative saturation fixed, dc/dcap = c/cap (ADIMC against
    #: uztwm + lztwm).  Snow, routing history and canopy water pass unchanged.
    #: "flux": "relative" plus lzfsc * lzsk.detach()/lzsk and lzfpc * lzpk.detach()/lzpk
    #: — the backward also holds each lower-zone free store's carried drainage flux
    #: k*S fixed, dS/dk = -S/k: a faster store carries less water into the next water
    #: year, which the 1-water-year truncation otherwise never sees (it flips the sign
    #: of d loss / d lzsk at slow spring-fed basins such as Shasta).
    #: Segmented-graph and eager chunk paths only.
    tbptt_carry: str = "absolute"
    #: TBPTT window in water years (``chunk_grid = "water_year"`` only; 1-3).  1: each
    #: chunk backpropagates within its own water year.  2: overlapping windows — a
    #: live chunk runs the PREVIOUS water year and its own with autograd through
    #: both and the loss on its own year only (the previous year is a gradient-
    #: carrying burn-in), so the gradient sees how a parameter shapes the water
    #: carried into the scored year, which a 1-year chunk detaches.  The carried
    #: state still advances one water year per step; dead chunks advance it
    #: forward-only, and the first chunk and the short envelope tail keep 1-year
    #: windows.  About twice the compute and activation memory per live chunk
    #: (3: the previous two water years as the burn-in, about three times).  Needs
    #: ``dead_chunk_nograd`` on a domain with dead chunks.  On a domain whose FIRST
    #: chunk is live, the first n - 1 chunks run their 1-year windows eagerly beside
    #: the n-year graph (slower, and peak memory about n + 1 years).
    #: Segmented-graph and eager chunk paths only.
    tbptt_window_years: int = 1
    #: > 0: CUDA-graph ACTIVATION RECOMPUTE — ONE captured graph of this many days
    #: (fwd + bwd) replayed over every training window (graphs.RecomputeTrainWindow):
    #: the forward keeps only the segment-boundary states and the backward re-runs
    #: each segment from its stored state, so graph memory is one segment's whatever
    #: the window length (a multi-year ``tbptt_window_years`` window included), for
    #: about one extra forward per segment.  Replaces the segmented / whole-chunk
    #: train graphs (days past the last whole segment run eagerly); 73 divides 365,
    #: 730 and 1095.  0 = off (the captures above).
    graph_recompute_days: int = 0
    eval_every: int = 2            # full-cal no-grad KGE selection cadence
    patience: int = 10              # early stop after this many stale selections
    #: early stopping is armed from this epoch on: a stale streak that ends
    #: before it never stops the run (0 = armed from the start)
    min_stop_epoch: int = 0
    #: multi-timescale chunks with no scoreable observation (a 26-entity run's
    #: WY1950-84) run their forward without autograd — no backward, and no
    #: optimizer step either way.  The train-mode net(x) is still drawn once per
    #: chunk (same dropout stream) and the segmented graphs replay the same
    #: forward, so the carried state and every later chunk are unchanged.
    #: Segmented/eager chunk path only (the whole-chunk graph keeps its backward).
    dead_chunk_nograd: bool = False
    #: numerics-neutral training diagnostics: chunk_log.csv (per chunk: entities
    #: and loss by family — the family split on the segmented/eager chunk path
    #: only — and the pre-clip gradient norm), eval_terms.csv (selection epochs:
    #: the eval-mode chunk loss by family and term) and per-epoch net snapshots
    #: with an EMA shadow (``ema_decay`` per optimizer step; restarted from the
    #: net on a resume) under checkpoints/snapshots/.
    diagnostics: bool = False
    ema_decay: float = 0.995
    #: CUDA-graph capture of the day-stepped pipeline (eager is dispatch-bound:
    #: ~300 tiny kernels/day).  Falls back to eager on CPU or capture failure.
    use_cuda_graphs: bool = True
    nograd_window: int = 512        # replay window for spinup/selection streaming
    #: >1 splits the train-chunk capture into this many consecutive segment
    #: graphs (forward + backward each, autograd across them) -- for drivers
    #: that fault on very large graphs; 1 = the single whole-chunk graph.
    train_graph_segments: int = 1
    #: CELL-DEDUPLICATED forward (data.with_cell_dedup): on a domain whose HRU rows
    #: repeat grid cells (multifamily: one row per (entity, cell), 5,849 rows on
    #: 2,652 cells for the 95 entities) the parameter net, PET, Snow-17, Noah ET and
    #: SAC-SMA — with their states and the learned PXTEMP — run once per DISTINCT
    #: cell; each cell's surface/base runoff is gathered to its rows for the per-row
    #: routing (flow length, unit hydrograph, routing history) and the aggregation W.
    #: Refused unless every row of a cell carries identical net features and statics
    #: (a flowlen_feature=True net has per-row parameters) and with gnn_k > 0.  The
    #: eval-mode forward and its gradients equal the per-row forward's to float
    #: round-off; in training with dropout > 0 a shared cell draws ONE dropout mask
    #: per chunk instead of one per row, so the training noise (and RNG stream)
    #: differ.  False (default) = one physics column per row, byte-identical.
    dedup_cells: bool = False
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
        if self.init_gate not in ("warn", "abort"):
            raise ValueError(f"init_gate {self.init_gate!r}")
        if self.mt_family_weight not in ("none", "equal"):
            family_shares(self.mt_family_weight)   # raises on a bad spec
        if self.mt_share_norm not in ("present", "all"):
            raise ValueError(f"mt_share_norm {self.mt_share_norm!r}: 'present' or 'all'")
        if self.mt_share_norm == "all" and self.mt_family_weight == "none":
            raise ValueError("mt_share_norm='all' normalizes family SHARES — it needs "
                             "mt_family_weight shares or 'equal'")
        if self.mt_select_weight:
            if self.mt_family_weight in ("none", "equal"):
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
        if self.train_graph_segments < 1:
            raise ValueError(f"train_graph_segments {self.train_graph_segments} < 1")
        if self.dedup_cells and self.gnn_k > 0:
            raise ValueError("dedup_cells needs a per-cell parameter net (gnn_k == 0)")
        if self.nograd_window < 1:
            raise ValueError(f"nograd_window {self.nograd_window} < 1")
        if not 30 <= self.train_chunk_days <= 366:
            # > 366 days can hold a 14th complete calendar month, overflowing
            # the fixed 13-slot monthly buckets (ET_MAXM) — months past the
            # 13th would be dropped from the ET/SWE/monthly-flow targets
            # SILENTLY (et_chunk_target caps at maxm without error)
            raise ValueError(f"train_chunk_days {self.train_chunk_days} "
                             "outside [30, 366]")
        if self.chunk_grid not in ("fixed", "water_year"):
            raise ValueError(f"chunk_grid {self.chunk_grid!r}")
        if self.tbptt_carry not in ("absolute", "relative", "flux"):
            raise ValueError(f"tbptt_carry {self.tbptt_carry!r}")
        if not 0 <= self.graph_recompute_days <= 366:
            raise ValueError(f"graph_recompute_days {self.graph_recompute_days} "
                             "outside [0, 366]")
        if self.tbptt_window_years not in (1, 2, 3):
            raise ValueError(f"tbptt_window_years {self.tbptt_window_years!r}: 1, 2 or 3")
        if self.tbptt_window_years > 1 and self.chunk_grid != "water_year":
            raise ValueError(f"tbptt_window_years {self.tbptt_window_years} needs chunk_grid "
                             "'water_year' (the burn-in is the previous water-year chunks)")
        if self.chunk_grid == "water_year" and self.train_chunk_days != 366:
            # the water-year grid sets its own lengths; a different value would be
            # ignored silently
            raise ValueError("chunk_grid 'water_year' takes whole water years: leave "
                             f"train_chunk_days at 366 (got {self.train_chunk_days})")
        if not (math.isfinite(self.var_gate_frac) and self.var_gate_frac >= 0.0):
            raise ValueError(f"var_gate_frac {self.var_gate_frac} must be finite and >= 0")
        if not math.isfinite(self.var_huber_cap):
            raise ValueError(f"var_huber_cap {self.var_huber_cap} must be finite (<= 0 = uncapped)")
        if not (math.isfinite(self.timing_loss_lambda) and self.timing_loss_lambda >= 0.0
                and math.isfinite(self.peak_loss_lambda) and self.peak_loss_lambda >= 0.0):
            raise ValueError("timing_loss_lambda and peak_loss_lambda must be finite and >= 0")
        if not 0.0 < self.peak_loss_frac <= 0.5:
            raise ValueError(f"peak_loss_frac {self.peak_loss_frac} outside (0, 0.5]")
        if not 90 <= self.shape_min_days <= 366:
            raise ValueError(f"shape_min_days {self.shape_min_days} outside [90, 366]")
        if not 0.0 <= self.timing_vol_gate < 1.0:
            raise ValueError(f"timing_vol_gate {self.timing_vol_gate} outside [0, 1)")
        if ((self.timing_loss_lambda > 0.0 or self.peak_loss_lambda > 0.0)
                and self.chunk_grid != "water_year"):
            # the terms and their record constants are defined per water year; the
            # fixed grid drifts off it by a day a year
            raise ValueError("timing_loss_lambda / peak_loss_lambda need chunk_grid "
                             "'water_year' (they score whole water years)")
        if self.spinup_mode not in ("window", "cycle"):
            raise ValueError(f"spinup_mode {self.spinup_mode!r}")
        if (self.spinup_years < 1 or not 2 <= self.spinup_warm_passes <= self.spinup_passes
                or not self.spinup_warm_tol > 0.0):
            raise ValueError("spinup_years >= 1, 2 <= spinup_warm_passes <= spinup_passes "
                             "and spinup_warm_tol > 0 are required")
        if self.min_stop_epoch < 0:
            raise ValueError(f"min_stop_epoch {self.min_stop_epoch} < 0")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay {self.ema_decay} outside (0, 1)")
        box = {}
        for name, lohi in dict(self.param_box).items():
            if name not in FREE_PARAMS:
                raise ValueError(f"param_box: {name!r} is not a learned parameter "
                                 f"({', '.join(FREE_PARAMS)})")
            lo, hi = (float(v) for v in lohi)
            blo, bhi = BOUNDS[name]
            if not blo <= lo <= hi <= bhi:
                raise ValueError(f"param_box: {name} ({lo}, {hi}) must satisfy "
                                 f"{blo} <= lo <= hi <= {bhi}")
            box[name] = (lo, hi)
        self.param_box = box
        clash = sorted(set(box) & (set(self.seasonal_params) | set(self.dynamic_params)))
        if clash:
            # forward._seasonal clamps a time-varying parameter to BOUNDS, not the box
            raise ValueError(f"param_box {clash}: a boxed parameter cannot also be "
                             "seasonal or dynamic")
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
        if self.pxtemp_learn and "PXTEMP" in (set(self.seasonal_params)
                                              | set(self.dynamic_params)):
            raise ValueError("a learned PXTEMP cannot also be seasonal or dynamic")
        if isinstance(self.obs_mask, str):      # tolerate a bare string
            self.obs_mask = tuple(s for s in self.obs_mask.split(",") if s)
        self.obs_mask = tuple(str(s) for s in self.obs_mask)
        from datetime import date as _date
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
        if self.calsim_arcs not in ("none", "train_default"):
            raise ValueError(f"calsim_arcs {self.calsim_arcs!r}: 'none' or 'train_default'")
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
        if self.noah_sac_exchanges and not self.canopy_lite:
            raise ValueError("noah_sac_exchanges requires the Noah-lite path "
                             "(et_mode='noah' with canopy_lite)")


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
#: what ``mt_family_weight="equal"`` resolves to: shares 1:1:1, renormalized by the
#: trainer over the families the run holds (the trainer's numeric-shares path)
EQUAL_FAMILY_SHARES = {f: 1.0 for f in FAMILY_KEYS.values()}


def family_shares(spec: str) -> dict[str, float] | None:
    """Parse a numeric ``mt_family_weight`` spec ("usgs=0.27,cdec=0.54,uf=0.19")
    into ``{family_id: share}`` summing to 1 over the families named; ``None``
    for the keyword modes ("none", "equal").  Family keys may be the short
    names or the registry family ids; every share must be positive."""
    if spec in ("none", "equal"):
        return None
    shares: dict[str, float] = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError(f"mt_family_weight {spec!r}: expected "
                             "family=share items or 'none'/'equal'")
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
    """Parse an ``mt_loss_ref`` spec ("usgs=0.7368,cdec=0.4202,uf=0.0980") into
    ``{family_id: L_ref}`` — each family's per-entity chunk loss per unit
    coefficient at the reference state (unnormalized; only the ratios reach the
    loss, through ``kappa_f = Lbar / L_ref_f``)."""
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

