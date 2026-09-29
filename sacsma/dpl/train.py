"""dPL training: truncated no-grad spinup + water-year TBPTT + KGE selection.

Protocol (per epoch):

1. **Spinup** — no-grad forward from the reference cold start at
   ``cfg.spinup_start`` (default 1978-10-01, ten water years ahead of the
   cal window) to the day before the calibration window, under the CURRENT
   parameters.  The default window spans the record-wet WY1982-83, which
   clamps the LZ tension stores — the one multi-year memory (~7-yr ET
   drawdown in the big arid basins) — at capacity, resetting the cold
   start: measured against the full prefix, cal-window flow agrees to
   KGE 1.000000 on MIL/NHG and >= 0.999975 on arid ISB/SCC, max|dQ|
   <= 3e-3 mm/day (a 5-yr window fails the 0.9999 parity bar on MIL).
   Set ``spinup_start <= "1915-01-01"`` to restore the exact
   record-start convention.
   ``spinup_refresh_every`` > 1 reuses a cached state as an explicit
   approximation.
2. **Selection** (every ``eval_every`` epochs, pre-update) — continue no-grad
   through the full calibration window and score pooled mean per-basin KGE
   (the GA-comparable exact objective).  Best net -> ``checkpoints/best.pt``;
   early stop after ``patience`` stale evaluations.  Validation observations
   are never read (``load_cal_obs`` does not materialize them).
3. **TBPTT** — fixed-length chunks (366 days) covering WY1989-2003, state and
   the 106-day routing-tail history carried detached across chunk boundaries;
   one AdamW step per chunk on the chunk-additive NNSE loss (post-CAL_END
   days of the last chunk are NaN-masked).  ``cfg.chunk_grid="water_year"``
   tiles the window with whole water years instead (365/366 days; the
   graphs capture 365 and a leap year's last day continues eagerly from the
   graphed state), so no calendar month is ever split.

For the multi-timescale domain (``domain="multifamily"``) the same protocol
runs over the training entities on the registry envelope (WY1950-2018): each
daily entity joins the chunk NNSE window-masked (NaN outside its own
``train_start``/``train_end``), the monthly entities add a chunk-additive
monthly NNSE term (simulated daily flow bucketed to complete calendar
months — under the fixed grid a month the chunk boundary cuts never enters
it), and selection scores per-entity KGE at each entity's native
timescale — pooled mean by default; with ``mt_family_weight="equal"`` (loss
shares 1:1:1 over the families present) the selection scalar is the
family-mean-of-means, and with numeric family shares
("usgs=0.27,cdec=0.54,uf=0.19") the share-weighted family mean, so
checkpoint choice follows the training objective.

On CUDA the day-stepped pipeline runs as captured CUDA graphs
(:mod:`sacsma.dpl.graphs`) — eager execution is dispatch-bound.  Graph
capture failure (or ``--device cpu``) falls back to eager with identical
numerics.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..io import MULTI_TIMESCALE_DOMAIN, load_params, soilveg_path
from .config import (
    CANOPY_LEARNED_PARAMS,
    CANOPY_LITE_LEARNED,
    EQUAL_FAMILY_SHARES,
    DplConfig,
    family_loss_refs,
    family_shares,
    pick_device,
)
from .data import (
    CalObs,
    DomainTensors,
    et_chunk_target,
    load_cal_obs,
    load_domain_tensors,
    load_et_obs,
    load_swe_obs,
    shape_chunk_targets,
    water_balance_anchor,
    with_cell_dedup,
)
from .features import (AEF_STORE_VARIANTS, AEF_VARIANTS, CONTINUOUS_STATICS, FeatureSet,
                       aef_store, build_features)
from .forward import PipelineState, initial_state, routing_uh, run_window
from .loss import (PEAK_REF_FLOOR, jul_sep_window, kge_torch, level_hinge_loss,
                   masked_basin_loss, record_references, shape_pull_loss)
from .multi_timescale import (
    load_entity_obs,
    monthly_chunk_target,
    monthly_nnse_loss,
)
from .parameter_net import ParameterNet, ga_priors
from .regularize import (
    adaptive_basin_weights,
    build_neighbor_edges,
    param_norm_scales,
    spatial_smoothness,
)
from .spinup import (
    annual_totals,
    block_end,
    cycle_spinup,
    describe,
    cold_state,
    year_index,
)

_DTYPES = {"float32": torch.float32, "float64": torch.float64}

#: diagnostics file columns (cfg.diagnostics).  chunk_log.csv: one row per chunk
#: per epoch — ``n_`` live entities, ``c_`` coefficient mass and ``l_`` loss
#: contribution by family (the ``l_`` sum to ``loss``; ``lt_`` / ``lp_`` the timing
#: and peak terms' parts of ``l_``), ``gnorm`` the pre-clip gradient norm.
#: eval_terms.csv: one row per selection epoch — the eval-mode
#: loss on the training grid by family and term, averaged over the live chunks.
_CHUNK_LOG_COLS = ("epoch", "k", "start", "days", "live", "nograd", "stepped",
                   "loss", "gnorm", "n_usgs", "n_cdec", "n_uf", "c_usgs",
                   "c_cdec", "c_uf", "l_usgs", "l_cdec", "l_uf",
                   "lt_usgs", "lp_usgs", "lt_cdec", "lp_cdec")
_EVAL_TERM_COLS = ("epoch", "eval_loss", "usgs_nnse", "usgs_log", "usgs_var",
                   "usgs_timing", "usgs_peak", "cdec_nnse", "cdec_log", "cdec_var",
                   "cdec_timing", "cdec_peak", "uf_monthly")
#: DplConfig fields a --resume must repeat exactly (the objective and the TBPTT
#: scheme); the check skips a field an older checkpoint does not record
_RESUME_FIXED = ("loss", "log_loss_lambda", "log_loss_eps", "var_loss_lambda",
                 "var_gate_frac", "var_huber_cap", "bias_loss_lambda",
                 "timing_loss_lambda", "peak_loss_lambda", "peak_loss_frac",
                 "shape_min_days", "timing_vol_gate",
                 "mt_family_weight", "chunk_grid", "train_chunk_days",
                 "tbptt_carry", "tbptt_window_years", "spinup_mode",
                 "noah_sac_exchanges", "spatial_reg_lambda", "adaptive_loss",
                 "et_loss_lambda", "et_level_lambda", "swe_loss_lambda",
                 "obs_mask", "pxtemp_learn", "pxtemp_box", "pxtemp_tau",
                 "mt_share_norm", "mt_loss_ref", "mt_loss_ref_power", "dedup_cells")
#: fields added after checkpoints already existed, whose absence means the default:
#: an older checkpoint is compared as if it recorded these (a pre-mask checkpoint
#: resumed with --obs-mask must be refused, not skipped)
_RESUME_DEFAULTED = {"obs_mask": (), "pxtemp_learn": False, "pxtemp_box": (-1.0, 3.0),
                     "pxtemp_tau": 1.0, "mt_share_norm": "present", "mt_loss_ref": "",
                     "mt_loss_ref_power": None, "dedup_cells": False}


def _split_out(out: dict, et_mode: str):
    """Split a net forward into (PARAM_ORDER dict, canopy dict|None).  The
    canopy subdict must be peeled off before params reach run_window
    (``sma.py`` iterates ``params.values()``, which must all be tensors)."""
    if et_mode == "noah":
        return {k: v for k, v in out.items() if k != "_canopy"}, out.get("_canopy")
    return out, None


def _stream_nograd(
    dom: DomainTensors, cfg: DplConfig,
    params: dict[str, torch.Tensor], uh, t0: int, t1: int,
    state: PipelineState, *, graph=None, collect: bool = False,
    canopy_params: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor | None, PipelineState]:
    """No-grad basin flow over [t0, t1): graph replays + an eager remainder.

    ``params`` must be the PARAM_ORDER dict only (no ``_canopy`` subdict);
    ``canopy_params``/tmin/tmax drive the Noah ET path (``cfg.et_mode='noah'``);
    ``dom.chunk_tmm`` returns (None, None) for non-grid domains (tavg fallback)."""
    outs: list[torch.Tensor] = []
    t = t0
    if graph is not None:
        graph.set_state(state)
        while t + graph.length <= t1:
            pr, ta, doy, leap = dom.chunk(t, t + graph.length)
            tn, tx = dom.chunk_tmm(t, t + graph.length)
            basin = graph.replay(pr, ta, doy, leap, tn, tx,
                                 dom.chunk_lai(t, t + graph.length),
                                 dom.chunk_state(t, t + graph.length))
            if collect:
                outs.append(basin.clone())
            t += graph.length
        state = graph.get_state()
    with torch.no_grad():
        while t < t1:
            te = min(t + cfg.nograd_window, t1)
            pr, ta, doy, leap = dom.chunk(t, te)
            tn, tx = dom.chunk_tmm(t, te)
            flow, state = run_window(pr, ta, doy, leap, dom.phys_lat_rad, dom.phys_elev,
                                     params, uh, state, n_inc=cfg.n_inc,
                                     perc_mode=cfg.perc_mode,
                                     fracp_floor=cfg.fracp_floor,
                                     ninc_mode="fixed", et_mode=cfg.et_mode,
                                     canopy_params=canopy_params, tmin=tn, tmax=tx,
                                     veg_frac=dom.phys_veg_frac,
                                     lai=dom.chunk_lai(t, te),
                                     noah_pet=cfg.noah_pet, sac_pet=cfg.sac_pet,
                                     pt_snow_albedo=cfg.pt_snow_albedo,
                                     pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                                     canopy_lite=cfg.canopy_lite,
                                     sac_exchanges=cfg.noah_sac_exchanges,
                                     state_idx=dom.chunk_state(t, te),
                                     row_cell=dom.row_cell)
            if collect:
                outs.append(dom.W @ flow)
            t = te
    return (torch.cat(outs, dim=1) if collect else None), state


def _relative_carry(state: PipelineState, params: dict[str, torch.Tensor], *,
                    flux: bool = False) -> PipelineState:
    """The state carried into a TBPTT chunk with each SAC content ``c`` rewritten as
    ``c * (cap / cap.detach())`` (``DplConfig.tbptt_carry = "relative"``): the value is
    unchanged (x / x == 1 exactly for the positive capacities), but the backward sees
    the store's relative saturation as fixed, ``dc/dcap = c / cap``, instead of the
    detached content treating a larger capacity as free deficit.  ADIMC is carried
    against ``uztwm + lztwm``; snow, routing history and canopy water pass unchanged.

    ``flux=True`` (``tbptt_carry = "flux"``) also multiplies the lower-zone free
    contents by ``k.detach() / k`` of their drainage rates (lzfsc by lzsk, lzfpc by
    lzpk): again the same value, but the backward holds the carried drainage flux
    ``k * S`` fixed, ``dS/dk = -S/k`` — a faster store carries less water into the
    next water year, the dependence the 1-water-year truncation cannot see."""
    s = state.sac
    adim_cap = params["uztwm"] + params["lztwm"]

    def rel(c: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
        # clone: the product saves c for its backward, and after a graphed chunk c
        # is a view of the graph's static output buffer, which the next replay
        # overwrites in place (no version bump) — the backward would read the
        # END-of-chunk contents
        return c.detach().clone() * (cap / cap.detach())

    lzfsc = rel(s.lzfsc, params["lzfsm"])
    lzfpc = rel(s.lzfpc, params["lzfpm"])
    if flux:
        lzfsc = lzfsc * (params["lzsk"].detach() / params["lzsk"])
        lzfpc = lzfpc * (params["lzpk"].detach() / params["lzpk"])
    sac = type(s)(uztwc=rel(s.uztwc, params["uztwm"]), uzfwc=rel(s.uzfwc, params["uzfwm"]),
                  lztwc=rel(s.lztwc, params["lztwm"]), lzfsc=lzfsc,
                  lzfpc=lzfpc, adimc=rel(s.adimc, adim_cap))
    return PipelineState(snow=state.snow, sac=sac, hist_surf=state.hist_surf,
                         hist_base=state.hist_base, canopy=state.canopy)


def _obs_chunk(calobs: CalObs, c0: int, c1: int) -> torch.Tensor:
    """Cal-window obs for record days [c0, c1); NaN outside the window."""
    out = torch.full((calobs.obs.shape[0], c1 - c0), float("nan"),
                     device=calobs.obs.device, dtype=calobs.obs.dtype)
    lo, hi = max(c0, calobs.t0), min(c1, calobs.t1)
    if hi > lo:
        out[:, lo - c0:hi - c0] = calobs.obs[:, lo - calobs.t0:hi - calobs.t0]
    return out


def _chunk_grid(dates: pd.DatetimeIndex, t0: int, t1: int, n_time: int, *,
                mode: str, chunk: int) -> list[tuple[int, int]]:
    """The TBPTT chunks as ``(c0, ce)`` record-day pairs covering ``[t0, t1)``.

    ``fixed``: ``chunk`` days each from ``t0`` (the last one cut at the record end,
    ``n_time``) — the historical grid.  ``water_year``: from ``t0`` (which must be a
    1 Oct) to each following 1 Oct, so 365 or 366 days, the last chunk running to
    ``t1`` or the record end, whichever comes first — every calendar month whole."""
    if mode == "fixed":
        n = math.ceil((t1 - t0) / chunk)
        return [(t0 + k * chunk, min(t0 + (k + 1) * chunk, n_time)) for k in range(n)]
    if mode != "water_year":
        raise ValueError(f"chunk_grid {mode!r}")
    d0 = dates[t0]
    if (d0.month, d0.day) != (10, 1):
        raise ValueError(f"chunk_grid 'water_year' needs a window starting on 1 Oct, not {d0.date()}")
    starts, year = [t0], d0.year + 1
    while True:
        i = int(dates.searchsorted(pd.Timestamp(year=year, month=10, day=1)))
        if i >= t1:
            ends = starts[1:] + [min(t1, n_time)]
            return list(zip(starts, ends, strict=True))
        starts.append(i)
        year += 1


def _grid_summary(grid: list[tuple[int, int]]) -> str:
    """``70 chunks: 52 x 365 + 17 x 366 + 1 x 92`` — lengths by count."""
    from collections import Counter
    counts = Counter(ce - c0 for c0, ce in grid)
    parts = [f"{n} x {length}" for length, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    return f"{len(grid)} chunks: " + " + ".join(parts)


def _cal_kge(sim: torch.Tensor, obs: torch.Tensor,
             min_days: int = 90) -> tuple[torch.Tensor, float]:
    """Per-basin cal KGE vector (B,) and the pooled mean over valid basins."""
    k = kge_torch(sim.double(), obs.double())    # full-record stats in f64
    m = torch.isfinite(obs).sum(dim=1) >= min_days
    return k, float(k[m].mean())


def _feature_stats(fs: FeatureSet) -> dict:
    d = asdict(fs)
    d.pop("x")                       # rebuildable; keep checkpoints small
    return d


def _shares_stat(fam_means: dict[str, float], shares: dict[str, float]) -> float:
    """Share-weighted mean of the family means over the families present in
    ``fam_means`` (shares renormalized over those families)."""
    present = [f for f in fam_means if f in shares]
    if not present:
        return float("nan")
    tot = sum(shares[f] for f in present)
    return float(sum(shares[f] / tot * fam_means[f] for f in present))


def _donor_subset_stat(k: np.ndarray, valid: np.ndarray, in_donor: np.ndarray,
                       fams: np.ndarray | None, select: str | None,
                       shares: dict[str, float] | None = None) -> float:
    """The donor's selection statistic recomputed from this run's per-entity
    KGE vector ``k`` over the donor's entities: the pooled mean over valid
    donor entities, (``select == "family_mean"``) the mean over families of
    the per-family means, or (``"family_weighted"``) the share-weighted
    family mean with the donor's ``shares`` — the same arithmetic as the
    trainer's own selection, restricted to ``in_donor``."""
    m = valid & in_donor
    if not m.any():
        return float("nan")
    if select in ("family_mean", "family_weighted") and fams is not None:
        fam_means = {f: float(k[m & (fams == f)].mean())
                     for f in np.unique(fams[m])}
        if select == "family_weighted":
            return _shares_stat(fam_means, shares or {})
        return float(np.mean(list(fam_means.values())))
    return float(k[m].mean())


def train(
    variant: str = "static",
    data_dir: str = "data",
    out_dir: str | Path | None = None,
    cfg: DplConfig | None = None,
    *,
    resume: bool = False,
    basins: tuple[str, ...] | None = None,   # debug subset (default: all 15)
    domain: str = "15cdec",                  # 15cdec HRU cloud | 15cdec_grid
                                             # native grid | multifamily
                                             # multi-timescale entities
) -> Path:
    """Train one feature variant; returns the output directory."""
    cfg = cfg or DplConfig()
    if cfg.ninc_mode != "fixed":
        raise ValueError("training requires ninc_mode='fixed' (dynamic mode "
                         "has per-day host syncs and unbounded loop length)")
    dev = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    dtype = _DTYPES[cfg.dtype]
    out = Path(out_dir if out_dir is not None else f"artifacts/dpl/{variant}")
    ckdir = out / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.csv"

    dom = load_domain_tensors(
        data_dir, domain=domain, device=dev, dtype=dtype, basins=basins,
        dynamic_window=cfg.dynamic_window if cfg.dynamic_params else None,
        calsim_footprint=cfg.calsim_footprint)
    eobs = d_rows = m_rows = midx = None
    if domain == MULTI_TIMESCALE_DOMAIN:
        if (cfg.et_loss_lambda > 0.0 or cfg.et_level_lambda > 0.0
                or cfg.swe_loss_lambda > 0.0):
            raise ValueError("ET/SWE auxiliary losses are not wired for the "
                             "multi-timescale domain")
        if variant in ("static", "climate"):
            raise ValueError(
                "the multi-timescale domain has no zonal soil_class/"
                "veg_class (they exist only where the original zonal "
                "regionalization was drawn) — use the physical or embedding "
                "variants, whose inputs cover the whole region grid")
        eobs = load_entity_obs(dom, data_dir, obs_mask=cfg.obs_mask)
        if cfg.obs_mask:
            print(f"train: obs_mask — {len(cfg.obs_mask)} hand-confirmed daily "
                  f"observation(s) masked: {', '.join(cfg.obs_mask)}", flush=True)
        d_rows = torch.as_tensor(eobs.daily_rows, device=dev)
        m_rows = torch.as_tensor(eobs.monthly_rows, device=dev)
        # daily entities ride the existing chunk NNSE as a CalObs whose
        # monthly rows are all-NaN (they self-exclude via min_days); the
        # monthly rows train through the bucketed term below instead.
        obs_full = torch.full((len(dom.basins), eobs.t1 - eobs.t0),
                              float("nan"), device=dev, dtype=dtype)
        obs_full[d_rows] = eobs.obs_daily
        var_full = torch.ones(len(dom.basins), device=dev, dtype=dtype)
        var_full[d_rows] = eobs.var_daily
        calobs = CalObs(t0=eobs.t0, t1=eobs.t1, obs=obs_full, obs_var=var_full)
        d0, dl = dom.dates[eobs.t0], dom.dates[eobs.t1 - 1]
        if d0.day != 1 or (dl + pd.Timedelta(days=1)).day != 1:
            raise ValueError("the multi-timescale cal window must span whole "
                             "calendar months (monthly targets)")
        win = dom.dates[eobs.t0:eobs.t1]
        midx = torch.as_tensor(
            (win.year * 12 + (win.month - 1)).to_numpy()
            - int(eobs.month_code[0]), device=dev)
        print(f"train: multi-timescale domain — {len(eobs.daily_rows)} daily "
              f"+ {len(eobs.monthly_rows)} monthly entities, cal window "
              f"{d0.date()}..{dl.date()} "
              f"({int(torch.isfinite(eobs.obs_daily).sum())} daily + "
              f"{int(torch.isfinite(eobs.obs_monthly).sum())} monthly obs); "
              "chunk loss = daily NNSE + monthly NNSE (complete months)",
              flush=True)
    else:
        if cfg.obs_mask:
            raise ValueError("obs_mask is wired for the multi-timescale domain only")
        calobs = load_cal_obs(dom, data_dir, cal_start=cfg.cal_start)
    # the peak / timing terms' per-basin record constants, from the training obs
    # once (like obs_var): the record mean of the yearly observed top-k mean and of
    # the yearly mean flow, over the cal window's water years with shape_min_days
    # valid days; every chunk-loss call below takes them through shape_kw, and the
    # timing term's 1 Jul - 30 Sep day window as the chunk's slice of season_all
    win_d = dom.dates[calobs.t0:calobs.t1]
    pk_ref, v_ref = record_references(
        calobs.obs.detach().cpu().numpy(),
        np.asarray(win_d.year + (win_d.month >= 10)),
        min_days=cfg.shape_min_days, peak_frac=cfg.peak_loss_frac)
    shape_kw = dict(peak_frac=cfg.peak_loss_frac,
                    peak_ref=torch.as_tensor(pk_ref, device=dev, dtype=dtype),
                    vol_ref=torch.as_tensor(v_ref, device=dev, dtype=dtype),
                    shape_min_days=cfg.shape_min_days,
                    timing_vol_gate=cfg.timing_vol_gate)
    season_all = jul_sep_window(dom.doy, dom.is_leap)       # (n_time,) record-indexed
    if cfg.peak_loss_lambda > 0.0 or cfg.timing_loss_lambda > 0.0:
        print(f"train: shape terms (summer-recession timing {cfg.timing_loss_lambda}, "
              f"flood peak {cfg.peak_loss_lambda}) on basin-chunks with >= "
              f"{cfg.shape_min_days} valid days; record constants for "
              f"{int((pk_ref >= PEAK_REF_FLOOR).sum())} of {len(pk_ref)} basins", flush=True)
    # family weighting (multi-timescale only): each family's share of the
    # loss, renormalized over the families this run trains; entities weigh
    # equally within a family.  The daily term carries the daily families'
    # shares (their ratio set through the per-entity weights, the sum through
    # daily_scale) and the monthly term the monthly family's share.  "equal"
    # is shares 1:1:1 through the same path (three families: thirds, daily
    # x 2/3, monthly x 1/3); "none" = the unweighted baseline.
    fam_w = None
    daily_scale = monthly_scale = 1.0
    if eobs is not None and cfg.adaptive_loss:
        # adaptive weights would reach the daily term only — monthly_nnse_loss
        # takes no per-entity weights — silently skewing the documented
        # semantics for the 9 monthly entities
        raise ValueError("adaptive_loss is not wired for the multi-timescale "
                         "domain (the monthly term takes no per-entity "
                         "weights)")
    shares = None
    if eobs is not None:
        shares = (dict(EQUAL_FAMILY_SHARES) if cfg.mt_family_weight == "equal"
                  else family_shares(cfg.mt_family_weight))
    if shares is not None:
        fam = np.array(eobs.family)
        present = [f for f in shares if (fam == f).any()]
        if not present:
            raise ValueError(f"mt_family_weight {cfg.mt_family_weight!r} "
                             "names no family present in this run")
        tot = sum(shares[f] for f in present)
        shares = {f: shares[f] / tot for f in present}
        missing = [str(f) for f in sorted(set(fam)) if f not in shares]
        if missing:
            raise ValueError(f"mt_family_weight {cfg.mt_family_weight!r} "
                             f"gives no share to {missing} (present in this "
                             "run); name every family or drop it with --basins")
        w_np = np.zeros(len(fam))
        for f in ("usgs_daily", "cdec_daily"):
            msk = fam == f
            if msk.any():
                w_np[msk] = shares[f] / msk.sum()
        if (w_np > 0).any():
            # unit-MEAN scale over the daily entities (masked_basin_loss's
            # documented weight contract): the weighted mean itself is scale-
            # invariant, but tiny absolute weights would trip the loss's
            # clamp_min(1.0) denominator guard in sparse early chunks and
            # silently deflate the term (bound in 38/70 envelope chunks at raw
            # 1/n_family scale); a run without daily entities keeps fam_w None
            w_np *= (w_np > 0).sum() / w_np.sum()
            fam_w = torch.as_tensor(w_np, device=dev, dtype=dtype)
        daily_scale = sum(shares.get(f, 0.0) for f in ("usgs_daily", "cdec_daily"))
        monthly_scale = shares.get("uf_monthly", 0.0)
        print("train: family weighting by SHARES — "
              + ", ".join(f"{f.split('_')[0]} {s:.3f}" for f, s in shares.items())
              + f"; daily x {daily_scale:.3f}, monthly x {monthly_scale:.3f}; "
              + ("selection = mean of the family means (equal shares)"
                 if cfg.mt_family_weight == "equal" else
                 "selection = share-weighted family mean"), flush=True)
    # mt_share_norm="all": fixed loss denominators, so every entity carries
    # share_f / n_f in every chunk it is scored in (a chunk holding one family
    # no longer hands it the whole term); "present" keeps the per-chunk
    # renormalization (weight_total / mt_total None -> byte-identical)
    w_total = mt_total = None
    if shares is not None and cfg.mt_share_norm == "all":
        if fam_w is not None:
            w_total = float(fam_w.sum())
        if eobs is not None and len(eobs.monthly_rows):
            mt_total = float(len(eobs.monthly_rows))
        print(f"train: family shares normalized over ALL entities (mt_share_norm=all) — "
              f"daily denominator {w_total}, monthly {mt_total}: a chunk's family term "
              "scales with the share of that family's entities it scores", flush=True)
    # mt_loss_ref: the FROZEN per-family loss scale kappa_f = Lbar / L_ref_f.  Set
    # once, before any graph capture, and never changed: the daily families' kappa
    # goes into fam_w IN PLACE (the tensor every loss path and the captured
    # TrainChunk read), AFTER w_total was taken from the unscaled weights (so the
    # fixed denominator is unchanged and kappa is not normalized away); uf's goes
    # into monthly_scale, a Python float the TrainChunk bakes at construction —
    # correct only because it is constant for the run.  shares (selection) stay
    # nominal.  _loss_split then logs kappa-scaled c_<fam> and l_<fam>: loss ==
    # sum of l_<fam> and l_<fam> / c_<fam> is still the per-entity loss.
    if cfg.mt_loss_ref and shares is not None:
        refs = family_loss_refs(cfg.mt_loss_ref)
        missing = [f for f in shares if f not in refs]
        if missing:
            raise ValueError(f"mt_loss_ref {cfg.mt_loss_ref!r} gives no reference level "
                             f"to {missing} (trained in this run)")
        pw = cfg.mt_loss_ref_power
        lbar = sum(shares[f] * refs[f] ** pw for f in shares)
        kappa = {f: lbar / refs[f] ** pw for f in shares}
        if fam_w is not None:
            fam_w.mul_(torch.as_tensor([kappa.get(f, 1.0) for f in fam],
                                       device=dev, dtype=dtype))
        monthly_scale *= kappa.get("uf_monthly", 1.0)
        print("train: FROZEN family loss scale (mt_loss_ref) — reference per-entity levels "
              + ", ".join(f"{f.split('_')[0]} {refs[f]:.4g}" for f in shares)
              + f"; kappa = {lbar:.4g} / L_ref^{pw:g}: "
              + ", ".join(f"{f.split('_')[0]} x {kappa[f]:.4f}" for f in shares)
              + f" (monthly x {monthly_scale:.4f} in all); chunk_log c_/l_ carry kappa, "
              "selection keeps the nominal shares", flush=True)
    # truncated spinup start (clamped to the record start; == 0 restores the
    # exact frozen full-prefix convention)
    spin_req = int(dom.dates.searchsorted(pd.Timestamp(cfg.spinup_start)))
    if cfg.spinup_mode == "window" and eobs is not None and spin_req >= calobs.t0:
        # the default spinup_start (set for the 15cdec calibration window) sits
        # inside this cal window — keep the ten-water-year spinup convention
        # relative to the window
        ts = dom.dates[calobs.t0] - pd.DateOffset(years=10)
        spin_req = int(dom.dates.searchsorted(ts))
        print(f"train: spinup_start {cfg.spinup_start} is inside the cal "
              f"window — spinning up from {ts.date()}", flush=True)
    spin_t0 = min(spin_req, calobs.t0)
    # timing-independent spinup (cfg.spinup_mode "cycle"): the window's own first
    # spinup_years years, looped to convergence from the last converged state
    spin_t1 = block_end(dom.dates, calobs.t0, cfg.spinup_years)
    spin_cache: dict = {}
    spin_yidx = None
    if cfg.spinup_mode == "cycle":
        spin_yidx = year_index(dom.dates, calobs.t0, spin_t1)
        print(f"train: cycle spinup — {dom.dates[calobs.t0].date()}.."
              f"{dom.dates[spin_t1 - 1].date()} looped {cfg.spinup_passes} times from "
              f"the cold start, then from the previous state until a pass moves no "
              f"basin's annual flow by more than {cfg.spinup_warm_tol:g} "
              f"({cfg.spinup_warm_passes}..{cfg.spinup_passes} passes); nothing before "
              "the window is read", flush=True)
    etobs = sweobs = anchor_monthly = None
    if cfg.et_loss_lambda > 0.0 or cfg.et_level_lambda > 0.0:
        etobs = load_et_obs(dom, cal_start=cfg.cal_start,
                            products=cfg.et_products or None)
        print(f"train: ET obs loss ON (shape lambda={cfg.et_loss_lambda}, level "
              f"hinge lambda={cfg.et_level_lambda}, shape sigma floor="
              f"{cfg.shape_sigma_floor}) — normalized-cycle pull + min-max "
              f"envelope hinge over {etobs.products}; cal-window only, NOT a "
              "selection metric", flush=True)
        if cfg.et_anchor_band > 0.0:
            anchor_monthly = water_balance_anchor(dom, calobs, etobs)
            ann = anchor_monthly.sum(axis=1)
            print(f"train: ET level hinge re-targeted to the WATER-BALANCE "
                  f"anchor (P - Q_obs) +/- {cfg.et_anchor_band:.0%} "
                  f"(replaces the product min-max envelope): "
                  + ", ".join(f"{b} {a:.0f}"
                              for b, a in zip(dom.basins, ann, strict=True))
                  + " mm/yr", flush=True)
    if cfg.swe_loss_lambda > 0.0:
        sweobs = load_swe_obs(dom, cal_start=cfg.cal_start)
        n_snow = int(sweobs.basin_w.sum())
        print(f"train: SWE obs loss ON (shape lambda={cfg.swe_loss_lambda}, "
              f"sigma floor={cfg.shape_sigma_floor}) — normalized accumulation/"
              f"melt-cycle pull over {sweobs.products}; {n_snow}/{len(dom.basins)}"
              " snow basins (no SWE level term); cal-window only, NOT a "
              "selection metric", flush=True)
    _needs_climate = variant in ("climate", "physical_climate")
    _needs_physical = variant in ("physical", "physical_climate")
    # warm start: the donor checkpoint is read here, before the features are
    # built, so that its feature standardization (z-scoring stats, categories,
    # Fourier extent) is reused exactly as the evaluate path does.  Without
    # this a different entity subset (--basins) re-z-scores the HRU table and
    # the loaded weights start from a perturbed copy of the donor field.
    ick = None
    donor_stats = None
    # the embedding variants take no statics (build_features drops them), so a
    # warm-start donor's recorded statics are () too
    statics = (() if variant in AEF_VARIANTS else
               tuple(c for c in CONTINUOUS_STATICS
                     if cfg.flowlen_feature or c != "flowlen"))
    if cfg.init_from:
        if resume:
            raise ValueError("init_from and resume are mutually exclusive "
                             "(resume restores this run's own last.pt)")
        ick = torch.load(cfg.init_from, map_location=dev, weights_only=False)
        if ick.get("variant") != variant or ick.get("domain") != domain:
            raise ValueError(
                f"init_from checkpoint is variant={ick.get('variant')!r} "
                f"domain={ick.get('domain')!r}; this run is "
                f"{variant!r}/{domain!r}")
        if ick.get("features"):
            donor_stats = FeatureSet(x=np.empty((0, 0), dtype=np.float32),
                                     **ick["features"])
            if donor_stats.fourier_k != cfg.fourier_k:
                raise ValueError(
                    f"init_from checkpoint was trained with fourier_k="
                    f"{donor_stats.fourier_k}; this run asks for "
                    f"{cfg.fourier_k}")
            if tuple(donor_stats.statics) != statics:
                raise ValueError(
                    f"init_from checkpoint was trained on statics "
                    f"{tuple(donor_stats.statics)}; this run asks for {statics} "
                    "(flowlen_feature) — the first layer would not load")
    fs = build_features(
        dom.hrus, variant=variant,
        forcing=dom.forcing if _needs_climate else None,
        climate_window=(donor_stats.climate_window
                        if donor_stats is not None else None),
        climate_product=(donor_stats.climate_product
                         if donor_stats is not None else
                         ("historical_livneh_unsplit"
                          if _needs_climate else None)),
        fourier_k=cfg.fourier_k,
        physical_path=(soilveg_path(data_dir, domain)
                       if _needs_physical else None),
        stats=donor_stats,
        statics=statics,
        aef_path=aef_store(data_dir) if variant in AEF_STORE_VARIANTS else None,
    )
    x = torch.as_tensor(fs.x).to(dev, dtype)
    # cell dedup (opt-in): the per-cell physics and the parameter net run once per
    # distinct grid cell (x_net); routing and W stay per (entity, cell) row.  Off,
    # x_net IS x and every dom.phys_* accessor returns the row field unchanged.
    if cfg.dedup_cells:
        if cfg.gnn_k > 0:
            raise ValueError("dedup_cells needs a per-cell parameter net (gnn_k == 0): the "
                             "message passing runs over the HRU rows")
        dom = with_cell_dedup(dom, fs.x)
        if cfg.dropout > 0.0:
            print(f"train: dedup_cells with dropout {cfg.dropout:g} — one dropout mask "
                  "per distinct cell per chunk (the rows of a shared cell no longer "
                  "draw their own), so the training noise differs from a run without "
                  "dedup; the eval-mode forward is the same", flush=True)
    x_net = dom.phys_x(x)

    # -- opt-in regularizers (default-off => byte-identical to the baseline) --
    reg_lambda = cfg.spatial_reg_lambda
    e_i = e_j = e_w = p_scale = p_islog = None
    if reg_lambda > 0.0:
        ei_np, ej_np, ew_np = build_neighbor_edges(
            dom.hrus, fs.x, k=cfg.spatial_reg_k,
            attr_scale=cfg.spatial_reg_attr_scale)
        e_i = torch.as_tensor(ei_np, device=dev)
        e_j = torch.as_tensor(ej_np, device=dev)
        e_w = torch.as_tensor(ew_np, device=dev, dtype=dtype)
        p_scale, p_islog = param_norm_scales(dev, dtype)
        print(f"train: spatial reg lambda={reg_lambda} over {e_i.numel()} "
              f"within-basin k{cfg.spatial_reg_k} edges "
              f"(attr_scale={cfg.spatial_reg_attr_scale})", flush=True)
    # adaptive per-basin loss weights: the SAME buffer feeds the graph loss and
    # the eager loss and is updated in place at each selection eval.
    basin_w = (torch.ones(len(dom.basins), device=dev, dtype=dtype)
               if cfg.adaptive_loss else None)
    if basin_w is not None:
        print(f"train: adaptive per-basin loss weights on "
              f"(beta={cfg.adaptive_loss_beta}, momentum={cfg.adaptive_loss_momentum}, "
              f"floor={cfg.adaptive_loss_floor}, clip={cfg.adaptive_loss_clip})",
              flush=True)

    net = ParameterNet(x.shape[1], hidden=cfg.hidden, embed=cfg.embed,
                       dropout=cfg.dropout, grouped_heads=cfg.grouped_heads,
                       gnn_k=cfg.gnn_k,
                       n_nodes=x.shape[0] if cfg.gnn_k > 0 else None,
                       seasonal_params=cfg.seasonal_params,
                       seasonal_amp=cfg.seasonal_amp,
                       seasonal_amp_frac=cfg.seasonal_amp_frac,
                       canopy=cfg.canopy,
                       canopy_separate_trunk=cfg.canopy_separate_trunk,
                       canopy_lite=cfg.canopy_lite,
                       dynamic_params=cfg.dynamic_params,
                       dynamic_amp=cfg.dynamic_amp,
                       pxtemp_learn=cfg.pxtemp_learn,
                       pxtemp_box=cfg.pxtemp_box,
                       pxtemp_tau=cfg.pxtemp_tau,
                       ).to(device=dev, dtype=dtype)
    if cfg.pxtemp_learn:
        print(f"train: LEARNED rain/snow threshold PXTEMP per cell in "
              f"[{cfg.pxtemp_box[0]:g}, {cfg.pxtemp_box[1]:g}] degC — zero-init head "
              f"(exactly the fixed 0 degC at init); hard split forward, "
              f"straight-through sigmoid surrogate (tau {cfg.pxtemp_tau:g} degC)",
              flush=True)
    if cfg.seasonal_params:
        print(f"train: seasonal (day-of-year harmonic) params {cfg.seasonal_params} "
              f"— 2 zero-init coeffs each, tanh-capped per-param at "
              f"{cfg.seasonal_amp_frac:g}*(hi-lo) "
              f"(field is exactly static at init)", flush=True)
    if cfg.et_mode == "noah":
        faithful = dom.tmin is not None
        obs_canopy = dom.veg_frac is not None and dom.lai_lut is not None
        trunk = "separate" if cfg.canopy_separate_trunk else "shared"
        learned = CANOPY_LITE_LEARNED if cfg.canopy_lite else CANOPY_LEARNED_PARAMS
        kind = ("MINIMAL/LITE — beta(soil moisture)*PET" if cfg.canopy_lite
                else "full canopy-resistance")
        print(f"train: Noah {kind} ET ON (potential={cfg.noah_pet}; "
              f"canopy head: {len(learned)} LEARNED {learned} params/cell on a "
              f"{trunk} trunk, zero-init at bound midpoints; veg_frac + seasonal "
              f"LAI PINNED from observation={obs_canopy}; scored via torch "
              f"pipeline, NOT run_basin).  faithful per-cell tmin/tmax={faithful}"
              f"{'' if faithful else ' — WARNING using tavg fallback'}", flush=True)
        if not obs_canopy:
            raise ValueError("et_mode='noah' needs observed veg_frac + lai "
                             "(soilveg_continuous.csv + lai_climatology.csv)")
    if cfg.gnn_k > 0:
        from .regularize import dense_neighbors
        nb_idx, nb_w = dense_neighbors(dom.hrus, fs.x, k=cfg.gnn_k,
                                       attr_scale=cfg.gnn_attr_scale)
        net.set_neighbors(nb_idx, nb_w)
        print(f"train: learned spatial smoother on (gnn_k={cfg.gnn_k}, "
              f"attr_scale={cfg.gnn_attr_scale}; zero-init mixing = exact v1 "
              f"at init)", flush=True)
    if eobs is not None:
        # the multi-timescale store carries no GA table — the scalar init
        # prior (area-weighted median per param) comes from the region-grid
        # 15cdec GA params over the covered cells
        pdf = load_params(data_dir, domain="15cdec_grid").drop_duplicates("key")
        hrus_p = dom.hrus[dom.hrus["key"].isin(set(pdf["key"]))]
        if hrus_p.empty:            # debug subsets entirely off the 15cdec grid
            hrus_p = pdf[["key"]].assign(area_weight=1.0)
        print(f"train: init priors = 15cdec_grid GA median over "
              f"{len(hrus_p)}/{len(dom.hrus)} HRU rows", flush=True)
        priors = ga_priors(pdf, hrus_p)
    else:
        priors = ga_priors(load_params(data_dir, domain=domain), dom.hrus)
    net.init_from_priors(priors, box=cfg.param_box)
    donor_kge = float("nan")
    # ep0-donor gate mode: "same" compares the selection scalar directly
    # (same entity set, same statistic); "subset" recomputes the donor's own
    # statistic from this run's per-entity KGE vector over the donor's
    # entities (the donor trained a subset of this run's entities); None =
    # not applicable.
    donor_gate = None
    donor_basins: tuple[str, ...] = ()
    donor_select = None
    donor_shares = None
    if ick is not None:
        # strict=False: heads the donor lacks (e.g. a fresh seasonal head)
        # keep their zero-init, so training starts EXACTLY at the donor's
        # parameter field; donor keys the net lacks are a config error.
        missing, unexpected = net.load_state_dict(ick["net"], strict=False)
        if unexpected:
            raise ValueError(f"init_from checkpoint carries heads this net "
                             f"lacks: {sorted(unexpected)}")
        donor_basins = tuple(ick.get("basins") or ())
        this_select = ((("family_mean" if cfg.mt_family_weight == "equal"
                         else "family_weighted" if shares is not None
                         else "pooled")) if eobs is not None else None)
        donor_select = ick.get("mt_select") or "pooled"
        donor_shares = (family_shares(ick.get("cfg", {}).get("mt_family_weight", "none"))
                        if donor_select == "family_weighted" else None)
        # a checkpoint without an entity list (pre-multi-timescale) is taken
        # as the same entity set, as the gate always assumed for those
        if ((not donor_basins or donor_basins == tuple(dom.basins))
                and donor_select == (this_select or "pooled")
                and donor_select != "family_weighted"):
            donor_gate = "same"
        elif donor_basins and set(donor_basins) <= set(dom.basins):
            # also the route for share-weighted donors: their statistic is
            # recomputed with the donor's own shares
            donor_gate = "subset"
        print(f"train: warm-start from {cfg.init_from} (epoch {ick['epoch']}, "
              f"sel cal KGE {ick.get('cal_kge', float('nan')):.4f}, "
              f"{len(donor_basins) or 'n/a'} entities); feature "
              f"standardization {'reused from the donor' if donor_stats is not None else 'REBUILT (donor carries no feature stats)'}; "
              f"fresh zero-init heads: {sorted(missing) if missing else 'none'}; "
              "fresh optimizer/scheduler", flush=True)
        if donor_stats is None and cfg.init_gate == "abort":
            raise RuntimeError("init_from: the donor checkpoint carries no "
                               "feature standardization, so the warm start "
                               "cannot be exact (--init-gate abort)")
        if donor_gate == "subset":
            print(f"train: ep0-donor gate in subset mode — this run trains "
                  f"{len(dom.basins)} entities with selection "
                  f"{this_select!r}; the donor's {donor_select!r} statistic "
                  f"over its {len(donor_basins)} entities is recomputed from "
                  "the epoch-0 per-entity KGE and must reproduce its sel cal "
                  "KGE", flush=True)
        elif donor_gate is None:
            print(f"train: ep0-donor gate off — the donor's {len(donor_basins) or 'n/a'} "
                  f"entities are not a subset of this run's {len(dom.basins)}",
                  flush=True)
        donor_kge = float(ick.get("cal_kge", float("nan")))
    if cfg.param_box:
        # after the priors and any warm start (a donor's buffers carry the
        # donor's box); re-applied after a resume below
        lo0, hi0 = net._lo.clone(), net._hi.clone()
        net.set_box(cfg.param_box)
        print("train: parameter box — " + ", ".join(
            f"{p} pinned at {lo:g}" if lo == hi else f"{p} in [{lo:g}, {hi:g}]"
            for p, (lo, hi) in cfg.param_box.items()), flush=True)
        if donor_gate is not None and not (torch.equal(lo0, net._lo)
                                           and torch.equal(hi0, net._hi)):
            # the boxed net deliberately departs from the donor's field
            print("train: ep0-donor gate off — --param-box changes the donor's "
                  "parameter box", flush=True)
            donor_gate = None
    if cfg.pxtemp_learn:
        # a warm start loads a pxtemp donor's _px_box/_px_tau buffers; the
        # configured box and surrogate width win (as param_box does above)
        want_box = torch.tensor(cfg.pxtemp_box, dtype=net._px_box.dtype,
                                device=net._px_box.device)
        want_tau = torch.tensor(cfg.pxtemp_tau, dtype=net._px_tau.dtype,
                                device=net._px_tau.device)
        if not (torch.equal(net._px_box, want_box) and torch.equal(net._px_tau, want_tau)):
            with torch.no_grad():
                net._px_box.copy_(want_box)
                net._px_tau.copy_(want_tau)
            print(f"train: PXTEMP box/tau set to the configured {cfg.pxtemp_box} / "
                  f"{cfg.pxtemp_tau:g} (the donor carried others)", flush=True)
            if donor_gate is not None:
                print("train: ep0-donor gate off — the PXTEMP box changes the "
                      "donor's field", flush=True)
                donor_gate = None
    if ick is not None and cfg.init_gate == "abort" and (
            donor_gate is None or not math.isfinite(donor_kge)):
        raise RuntimeError(
            f"init_from: the ep0-donor gate cannot be armed (mode "
            f"{donor_gate!r}, donor sel cal KGE {donor_kge}); "
            "--init-gate abort requires a gate")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    warm = max(int(cfg.lr_warmup_epochs), 0)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.n_epochs - warm, 1), eta_min=cfg.lr_min)
    if warm > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1,
                                               total_iters=warm), cosine],
            milestones=[warm])
    else:
        sched = cosine

    start_epoch, best_kge, stale = 0, -math.inf, 0
    if not resume and log_path.exists():
        log_path.unlink()                        # fresh run, fresh log
    if resume and (ckdir / "last.pt").exists():
        ck = torch.load(ckdir / "last.pt", map_location=dev, weights_only=False)
        if ck.get("basins") is not None \
                and tuple(ck["basins"]) != tuple(dom.basins):
            raise ValueError(
                f"resume: checkpoint was trained on {len(ck['basins'])} "
                f"basins but this run loads {len(dom.basins)} — repeat the "
                "same --basins subset")
        # the objective and the TBPTT scheme come from the command line, not the
        # checkpoint: a resume that dropped or changed one of these flags would
        # continue silently on a different loss
        ck_cfg, now = ck.get("cfg") or {}, asdict(cfg)
        ck_eff = {**_RESUME_DEFAULTED, **ck_cfg}
        changed = [f"{k}: {ck_eff[k]!r} -> {now[k]!r}" for k in _RESUME_FIXED
                   if k in ck_eff and k in now and ck_eff[k] != now[k]]
        if changed:
            raise ValueError("resume: the loss / TBPTT settings differ from the "
                             "checkpoint's (repeat its flags): " + "; ".join(changed))
        if ((ck_cfg.get("timing_loss_lambda", 0.0) > 0.0
             or ck_cfg.get("peak_loss_lambda", 0.0) > 0.0) and "shape_min_days" not in ck_cfg):
            # the timing / peak terms of the checkpoint's code had different forms
            # (whole-year timing, relative peak error): the same flags would now
            # continue on a different loss
            raise ValueError("resume: the checkpoint trained the timing / peak terms of "
                             "the code before their 2026-09-27 revision; it cannot "
                             "continue on the revised forms")
        net.load_state_dict(ck["net"])
        if cfg.param_box:
            net.set_box(cfg.param_box)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_kge, stale = ck["epoch"] + 1, ck["best_kge"], ck["stale"]
        print(f"train: resumed at epoch {start_epoch} (best cal KGE {best_kge:.4f})",
              flush=True)

    # SWE snow-basin participation weights (shared by the graph and eager paths)
    swe_w = (torch.as_tensor(sweobs.basin_w, device=dev, dtype=dtype)
             if sweobs is not None else None)

    # -- the chunk grid and the CUDA-graph capture (falls back to eager) -----
    # The water-year grid is set by the calendar and names the lengths to
    # capture (365 and 366, the most frequent first; the short tail replays
    # eagerly).  The fixed grid follows the captured length — an OOM may halve
    # it — so it is built after capture.
    fixed = cfg.chunk_grid == "fixed"
    grid: list[tuple[int, int]] | None = None
    if fixed:
        cap_lens = [cfg.train_chunk_days, cfg.train_chunk_days // 2]
    else:
        from collections import Counter
        grid = _chunk_grid(dom.dates, calobs.t0, calobs.t1, dom.n_time,
                           mode=cfg.chunk_grid, chunk=cfg.train_chunk_days)
        cap_lens = [L for L, _ in Counter(ce - c0 for c0, ce in grid[:-1]).most_common()]
    nograd_g = None
    nograd_dead = None      # multi-year mode: forward-only window for dead chunks
    rec_g = None            # activation-recompute train window (graph_recompute_days)
    train_gs: dict = {}     # whole-chunk graphs by chunk length
    seg_gs: dict = {}       # segmented graphs by chunk length
    carry_grad = cfg.tbptt_carry in ("relative", "flux")
    if (carry_grad and cfg.use_cuda_graphs and dev.type == "cuda"
            and cfg.train_graph_segments == 1 and not cfg.graph_recompute_days):
        # fail before any capture (the check after capture stays as a backstop)
        raise ValueError(f"tbptt_carry={cfg.tbptt_carry!r} needs the segmented or eager "
                         "chunk path (--train-graph-segments >= 2, or --no-graphs)")
    n_win = cfg.tbptt_window_years
    win2 = n_win > 1                 # multi-year TBPTT windows
    if (win2 and cfg.use_cuda_graphs and dev.type == "cuda"
            and cfg.train_graph_segments == 1 and not cfg.graph_recompute_days):
        raise ValueError(f"tbptt_window_years {n_win} needs the segmented or eager chunk "
                         "path (--train-graph-segments >= 2, or --no-graphs)")
    if win2 and (etobs is not None or sweobs is not None):
        raise ValueError(f"tbptt_window_years {n_win} does not support the ET/SWE "
                         "auxiliary losses")
    if cfg.use_cuda_graphs and dev.type == "cuda":
        from .graphs import NoGradWindow, TrainChunk
        try:
            net.eval()
            with torch.no_grad():
                params0, canopy0 = _split_out(net(x_net), cfg.et_mode)
                uh0 = routing_uh(params0, dom.flowlen, row_cell=dom.row_cell)
            print("train: capturing CUDA graphs (no-grad window + train chunk) ...",
                  flush=True)
            nograd_g = NoGradWindow(dom, cfg, cfg.nograd_window, params0, uh0,
                                    canopy_params=canopy0)
        except Exception as e:  # noqa: BLE001 — any capture failure -> eager
            print(f"train: no-grad capture failed ({e!r}); running eager",
                  flush=True)
            nograd_g = None
        if (nograd_g is not None and win2 and cfg.dead_chunk_nograd and cap_lens
                and not cfg.graph_recompute_days):
            # multi-year mode captures no 1-year train graph: dead chunks replay a
            # forward-only window of the shortest chunk (static buffers only, no
            # activations), a leap year's last day eager
            try:
                nograd_dead = NoGradWindow(dom, cfg, min(cap_lens), params0, uh0,
                                           canopy_params=canopy0)
            except Exception as e:  # noqa: BLE001 — fall back to the stream window
                print(f"train: dead-chunk window capture failed ({e!r}); dead chunks "
                      f"stream through the {cfg.nograd_window}-day window", flush=True)
                torch.cuda.empty_cache()
        if nograd_g is not None and cfg.graph_recompute_days > 0:
            # activation recompute: ONE captured graph of graph_recompute_days days
            # replayed over any window, live or dead (graphs.RecomputeTrainWindow) —
            # the graph memory of one segment whatever the window length; replaces
            # the segmented and whole-chunk train captures
            from .graphs import RecomputeTrainWindow
            if etobs is not None or sweobs is not None:
                raise ValueError("graph_recompute_days does not support the ET/SWE "
                                 "auxiliary losses")
            try:
                rec_g = RecomputeTrainWindow(dom, cfg, cfg.graph_recompute_days,
                                             params0, uh0, canopy_params=canopy0,
                                             sample_c0=calobs.t0)
                print(f"train: activation-recompute train graph ({cfg.graph_recompute_days}"
                      " days, fwd+bwd, replayed per segment; backward re-runs each "
                      "segment from its stored state)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"train: recompute capture failed ({e!r}); eager chunks",
                      flush=True)
                torch.cuda.empty_cache()
        elif nograd_g is not None and cfg.train_graph_segments > 1:
            # segmented train-chunk graphs (drivers that fault on whole-year
            # captures): graphed run_window with autograd across segments;
            # net/routing/loss stay eager, so the chunk loop below takes its
            # eager branch with run_window swapped (same TBPTT gradient).  The
            # water-year grid captures its SHORTEST length only (365): a leap
            # year runs the graphed 365 days and its last day eagerly from the
            # graphed state, autograd through both — a second graph pool would
            # double the activation memory (it pushed an 8 GB GPU into paging).
            # No halving (the fixed grid keeps its length and runs eager on
            # failure, as before).
            from .graphs import SegmentedTrainWindow
            if etobs is not None or sweobs is not None:
                raise ValueError("train_graph_segments > 1 does not support "
                                 "the ET/SWE auxiliary losses")
            # tbptt_window_years n > 1: the one captured length is the shortest
            # n-year window (n x 365 days), in n times the segments (the same
            # per-segment size); a longer window runs its leap days eagerly, as a
            # leap year does on the 1-year path
            for clen in (cap_lens[:1] if fixed else
                         [n_win * min(cap_lens)] if win2 and cap_lens else
                         [min(cap_lens)] if cap_lens else []):
                n_seg = cfg.train_graph_segments * n_win
                try:
                    seg_gs[clen] = SegmentedTrainWindow(
                        dom, cfg, clen, n_seg,
                        params0, uh0, canopy_params=canopy0, sample_c0=calobs.t0)
                    print(f"train: segmented train-chunk graphs ({clen} days) -- "
                          f"{n_seg} x {seg_gs[clen].lens} days "
                          "(fwd+bwd captured per segment; net/routing/loss eager)",
                          flush=True)
                except torch.cuda.OutOfMemoryError:
                    n_len = sum(1 for c0, ce in (grid or []) if ce - c0 == clen)
                    print(f"train: segmented capture OOM at {clen} days — "
                          + ("eager chunks" if fixed else
                             f"the {n_win}-year windows run eager (slow)" if win2 else
                             f"the {n_len} chunks of that length run eager"), flush=True)
                    torch.cuda.empty_cache()
                except Exception as e:  # noqa: BLE001
                    print(f"train: segmented capture failed ({e!r}); eager chunks",
                          flush=True)
                    break
        elif nograd_g is not None:
            # whole-chunk fwd+bwd capture is the VRAM peak: on OOM the fixed
            # grid halves the chunk once (TBPTT detaches mid-water-year) before
            # giving up; the water-year grid captures each of its lengths and
            # runs the chunks of a length that will not fit eagerly.  The
            # multi-timescale monthly term is captured too (static target
            # buffers, mask-zero at capture); its short TAIL chunk — the
            # envelope ends at the forcing record — replays eagerly below.
            for clen in cap_lens:
                try:
                    train_gs[clen] = TrainChunk(
                        net, dom, cfg, clen, x_net, calobs.obs_var,
                        weight=fam_w if fam_w is not None else basin_w,
                        swe_basin_w=swe_w,
                        mt_rows=m_rows if eobs is not None else None,
                        mt_var=eobs.var_monthly if eobs is not None else None,
                        daily_scale=daily_scale, monthly_scale=monthly_scale,
                        shape_kw=shape_kw, weight_total=w_total, mt_total=mt_total)
                    if fixed:
                        break           # the first length that fits is the grid's
                except torch.cuda.OutOfMemoryError:
                    n_len = sum(1 for c0, ce in (grid or []) if ce - c0 == clen)
                    print(f"train: chunk capture OOM at {clen} days — "
                          + ("halving" if fixed else
                             f"the {n_len} chunks of that length run eager"), flush=True)
                    torch.cuda.empty_cache()
                    if not train_gs:
                        # nothing captured yet: drop the grads the failed attempt
                        # allocated (a captured graph's static grads must stay)
                        for p in net.parameters():
                            p.grad = None
                except Exception as e:  # noqa: BLE001
                    print(f"train: chunk capture failed ({e!r}); "
                          "eager chunks", flush=True)
                    break
        if not train_gs:
            for p in net.parameters():
                p.grad = None
    if grid is None:
        chunk = next(iter(train_gs)) if train_gs else cfg.train_chunk_days
        grid = _chunk_grid(dom.dates, calobs.t0, calobs.t1, dom.n_time,
                           mode="fixed", chunk=chunk)
    if carry_grad and train_gs:
        # the whole-chunk graphs own their state buffers (set_state copies values
        # in), so the carried state cannot hold a gradient path to the capacities
        raise ValueError(f"tbptt_carry={cfg.tbptt_carry!r} needs the segmented or eager "
                         "chunk path (--train-graph-segments >= 2, or no CUDA graphs)")
    graphed = sorted(set(train_gs) | set(seg_gs))
    other = sorted({ce - c0 for c0, ce in grid[:-1]} - set(graphed))
    if win2:
        print(f"train: {n_win}-water-year TBPTT windows — each live chunk runs the "
              f"previous {n_win - 1} water year(s) as a gradient-carrying burn-in, loss on "
              "its own year only"
              + (f"; graphed {graphed} days, a longer window's leap days eager"
                 if seg_gs else f"; recompute graph of {cfg.graph_recompute_days} days"
                 if rec_g is not None else "")
              + "; dead chunks advance forward-only; the first "
              "chunks and the envelope tail run shorter windows", flush=True)
    elif graphed and other:
        if seg_gs:
            print(f"train: chunks of {other} days run the graphed {min(seg_gs)} days and the rest "
                  "eagerly from the graphed state", flush=True)
        else:
            print(f"train: chunk lengths without a graph run eager: {other}", flush=True)

    def _mt_kge(sim: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Pooled per-entity cal KGE at each entity's NATIVE timescale: daily
        rows at daily stride, monthly rows on calendar-month sums of the same
        simulated flow (>= 12 obs months to enter the pool).  Prints the
        per-family means; the selection scalar is the pooled mean, or the
        family-mean-of-means under ``mt_family_weight="equal"``."""
        k = kge_torch(sim.double(), calobs.obs.double())
        sim_m = torch.zeros(len(eobs.monthly_rows), eobs.obs_monthly.shape[1],
                            device=sim.device, dtype=torch.float64)
        sim_m.index_add_(1, midx, sim[m_rows].double())
        k[m_rows] = kge_torch(sim_m, eobs.obs_monthly.double())
        valid = torch.isfinite(calobs.obs).sum(dim=1) >= 90
        valid[m_rows] = torch.isfinite(eobs.obs_monthly).sum(dim=1) >= 12
        fams = np.array(eobs.family)
        k_np, v_np = k.cpu().numpy(), valid.cpu().numpy()
        fam_means = {f: float(k_np[(fams == f) & v_np].mean())
                     for f in ("usgs_daily", "cdec_daily", "uf_monthly")
                     if ((fams == f) & v_np).any()}
        parts = "  ".join(
            f"{f.split('_')[0]} {m:.4f} (n={int(((fams == f) & v_np).sum())})"
            for f, m in fam_means.items())
        pooled = float(k[valid].mean())
        if cfg.mt_family_weight == "equal":
            # selection follows the training objective: with equal family
            # weighting, each family casts one equal vote on which epoch is
            # best (the pooled mean stays printed for cross-run comparison).
            fam_scalar = float(np.mean(list(fam_means.values())))
            print(f"    per-family cal KGE: {parts}  | sel family-mean "
                  f"{fam_scalar:.4f} (pooled {pooled:.4f})", flush=True)
            return k, fam_scalar
        if shares is not None:
            # share-weighted family mean over the families with a valid
            # entity, renormalized like the loss
            fam_scalar = _shares_stat(fam_means, shares)
            print(f"    per-family cal KGE: {parts}  | sel family-weighted "
                  f"{fam_scalar:.4f} (pooled {pooled:.4f})", flush=True)
            return k, fam_scalar
        print(f"    per-family cal KGE: {parts}", flush=True)
        return k, pooled

    def _spinup_and_maybe_eval(
        do_eval: bool,
    ) -> tuple[float, torch.Tensor | None, PipelineState]:
        """Fresh spinup under the current net; optionally the selection KGE
        (pooled scalar + the per-basin vector for adaptive weighting)."""
        net.eval()
        with torch.no_grad():
            pe, cp_e = _split_out(net(x_net), cfg.et_mode)
            ue = routing_uh(pe, dom.flowlen, row_cell=dom.row_cell)
        if nograd_g is not None:
            nograd_g.set_params(pe, ue, canopy_params=cp_e)
        st0 = initial_state(dom.n_phys, dev, dtype, init_mode=cfg.init_mode,
                            params=pe, et_mode=cfg.et_mode, n_rows=dom.n_hru)
        if cfg.spinup_mode == "cycle":
            from .graphs import _clone_state
            # the cold start and spinup_passes the first time; then from the previous
            # state until a pass moves no basin by more than spinup_warm_tol,
            # tracking the fixed point as the parameters move
            def _basins(a: int, b: int, s: PipelineState):
                sim_b, s = _stream_nograd(dom, cfg, pe, ue, a, b, s, graph=nograd_g,
                                          collect=True, canopy_params=cp_e)
                return annual_totals(sim_b, spin_yidx), s

            first = "state" not in spin_cache
            st, n_pass, change = cycle_spinup(
                _basins, calobs.t0, spin_t1,
                cold_state(dom, cfg, pe) if first else spin_cache["state"], cfg.spinup_passes,
                min_passes=None if first else cfg.spinup_warm_passes,
                until=None if first else cfg.spinup_warm_tol)
            spin_cache.update(state=_clone_state(st), cycles=n_pass, change=change)
            if first:
                print("train: " + describe(dom.dates, calobs.t0, spin_t1, n_pass, change),
                      flush=True)
        else:
            _, st = _stream_nograd(dom, cfg, pe, ue, spin_t0, calobs.t0, st0,
                                   graph=nograd_g, canopy_params=cp_e)
        pooled, per_basin = float("nan"), None
        if do_eval:
            sim, _ = _stream_nograd(dom, cfg, pe, ue, calobs.t0, calobs.t1, st,
                                    graph=nograd_g, collect=True, canopy_params=cp_e)
            per_basin, pooled = (_mt_kge(sim) if eobs is not None
                                 else _cal_kge(sim, calobs.obs))
            if cfg.diagnostics:
                eval_sim["sim"] = sim
        return pooled, per_basin, st

    eval_sim: dict[str, torch.Tensor] = {}   # the last selection pass's flow (diagnostics)

    def _save(path: Path, *, epoch: int, kge: float) -> None:
        torch.save({**_payload(epoch=epoch, kge=kge), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, path)

    def _payload(*, epoch: int, kge: float) -> dict:
        return {"net": net.state_dict(), "epoch": epoch,
                "best_kge": best_kge, "stale": stale, "cal_kge": kge,
                "mt_select": ((("family_mean"
                                if cfg.mt_family_weight == "equal" else
                                "family_weighted" if shares is not None
                                else "pooled"))
                              if eobs is not None else None),
                "cfg": asdict(cfg), "variant": variant, "domain": domain,
                "basins": list(dom.basins),
                "net_config": {"hidden": cfg.hidden, "embed": cfg.embed,
                               "dropout": cfg.dropout,
                               "grouped_heads": cfg.grouped_heads,
                               "gnn_k": cfg.gnn_k,
                               "seasonal_params": cfg.seasonal_params,
                               "seasonal_amp": cfg.seasonal_amp,
                               "seasonal_amp_frac": cfg.seasonal_amp_frac,
                               "canopy": cfg.canopy,
                               "canopy_separate_trunk":
                                   cfg.canopy_separate_trunk,
                               "canopy_lite": cfg.canopy_lite,
                               "dynamic_params": cfg.dynamic_params,
                               "dynamic_amp": cfg.dynamic_amp,
                               "dynamic_window": cfg.dynamic_window,
                               "pxtemp_learn": cfg.pxtemp_learn,
                               "pxtemp_box": cfg.pxtemp_box,
                               "pxtemp_tau": cfg.pxtemp_tau},
                "et_mode": cfg.et_mode,
                "features": _feature_stats(fs)}

    n_chunks = len(grid)

    # per-chunk obs targets — fixed given the chunk grid, so build once.  For
    # each chunk: the day->month bucket, the normalized-shape mu/sig (per-chunk
    # because normalization runs over the chunk's masked months), the slot mask,
    # and (ET only) the min-max level envelope.  None when the loss is off.
    et_targets: list | None = None
    swe_targets: list | None = None
    if etobs is not None or sweobs is not None:
        et_targets = [] if etobs is not None else None
        swe_targets = [] if sweobs is not None else None
        n_slots_total = 0
        for c0, ce in grid:
            bucket, cmon0, mask = et_chunk_target(
                dom.dates, c0, ce - c0, calobs.t0, calobs.t1)
            bucket_t = torch.as_tensor(bucket, device=dev, dtype=dtype)
            mask_t = torch.as_tensor(mask, device=dev, dtype=dtype)
            if etobs is not None:
                mu, sig, lo, hi = shape_chunk_targets(
                    etobs, cmon0, mask, sigma_floor=cfg.shape_sigma_floor)
                if anchor_monthly is not None:
                    # water-balance anchor: the hinge envelope becomes the
                    # anchor total of THIS chunk's masked months +/- the band
                    # (same seasonal resolution as the product envelope it
                    # replaces; only the lo/hi VALUES change — the loss and
                    # captured-graph paths are untouched)
                    tot = (anchor_monthly[:, cmon0] * mask).sum(axis=1)
                    lo = tot * (1.0 - cfg.et_anchor_band)
                    hi = tot * (1.0 + cfg.et_anchor_band)
                et_targets.append((
                    bucket_t,
                    torch.as_tensor(mu, device=dev, dtype=dtype),
                    torch.as_tensor(sig, device=dev, dtype=dtype),
                    mask_t,
                    torch.as_tensor(lo, device=dev, dtype=dtype),
                    torch.as_tensor(hi, device=dev, dtype=dtype)))
            if sweobs is not None:
                # SWE is a STATE: bucket columns divided by day counts -> the
                # matmul yields the monthly MEAN, matching the obs semantics.
                days = np.maximum(bucket.sum(axis=0, keepdims=True), 1.0)
                smu, ssig, _, _ = shape_chunk_targets(
                    sweobs, cmon0, mask, sigma_floor=cfg.shape_sigma_floor)
                swe_targets.append((
                    torch.as_tensor(bucket / days, device=dev, dtype=dtype),
                    torch.as_tensor(smu, device=dev, dtype=dtype),
                    torch.as_tensor(ssig, device=dev, dtype=dtype),
                    mask_t))
            n_slots_total += int(mask.sum())
        print(f"train: obs targets = {n_slots_total} complete months over "
              f"{n_chunks} chunks "
              f"(ET {'on' if etobs is not None else 'off'}, "
              f"SWE {'on' if sweobs is not None else 'off'})", flush=True)
    # per-chunk MONTHLY-FLOW targets (multi-timescale domain): fixed given the
    # chunk grid — the day->month bucket, each entity's observed months gathered
    # to the chunk's slots, and the finite-slot mask.  Simulated flow multiplies
    # the bucket per chunk.  A month the chunk boundary cuts gets no slot: under
    # the fixed grid that is one month most years, under the water-year grid none.
    mt_targets: list | None = None
    if eobs is not None:
        mt_targets = []
        n_ms = 0
        for c0, ce in grid:
            bucket, cols, mask = monthly_chunk_target(
                dom.dates, c0, ce - c0, calobs.t0, calobs.t1, eobs.month_code)
            tgt = eobs.obs_monthly[:, torch.as_tensor(cols, device=dev)]
            fin = (torch.isfinite(tgt)
                   & (torch.as_tensor(mask, device=dev) > 0).unsqueeze(0))
            mt_targets.append((
                torch.as_tensor(bucket, device=dev, dtype=dtype),
                torch.where(fin, tgt, torch.zeros_like(tgt)),
                fin.to(dtype)))
            n_ms += int(fin.sum())
        n_obs_months = int(torch.isfinite(eobs.obs_monthly).sum())
        print(f"train: monthly flow targets = {n_ms} entity-months over "
              f"{n_chunks} chunks ({n_obs_months} observed; "
              f"{n_obs_months - n_ms} cut by a chunk boundary)", flush=True)
    # a chunk carrying NO scoreable observation must not step the optimizer:
    # zero grads still move parameters through AdamW momentum + weight decay
    # (the multi-timescale envelope's tail — 40 days on the fixed grid, Oct-Dec
    # 2018 on the water-year grid — where every daily entity may fail min_days
    # and no monthly entity reaches Dec 2018).  Existing domains never produce
    # a dead chunk; their behavior is untouched.
    if eobs is not None:
        chunk_live = []
        for k, (c0, ce) in enumerate(grid):
            obs_c = _obs_chunk(calobs, c0, ce)
            live = bool((torch.isfinite(obs_c).sum(dim=1) >= 90).any())
            if mt_targets is not None:
                live = live or bool(mt_targets[k][2].sum() > 0)
            chunk_live.append(live)
        if not all(chunk_live):
            print(f"train: {chunk_live.count(False)} chunk(s) carry no "
                  "scoreable obs — forward-only (no optimizer step)",
                  flush=True)
    else:
        chunk_live = [True] * n_chunks
    skip_dead = cfg.dead_chunk_nograd and not all(chunk_live)
    if win2 and not all(chunk_live) and not skip_dead:
        raise ValueError(f"tbptt_window_years {n_win} needs --dead-chunk-nograd on a domain "
                         "with dead chunks (a dead chunk would otherwise run a full-year "
                         "eager forward and backward beside the multi-year graph)")
    if skip_dead:
        if reg_lambda > 0.0:
            # the penalty's own train-mode net(x) draws dropout in every chunk
            raise ValueError("dead_chunk_nograd does not combine with spatial_reg_lambda")
        n_dead = chunk_live.count(False)
        print(f"train: the {n_dead} dead chunk(s) run forward-only, without autograd "
              "(dead_chunk_nograd; segmented/eager chunk path)", flush=True)

    # -- diagnostics (cfg.diagnostics; numerics-neutral) ---------------------
    diag = cfg.diagnostics
    chunk_log_path, eval_terms_path = out / "chunk_log.csv", out / "eval_terms.csv"
    snapdir = ckdir / "snapshots"
    params_list = list(net.parameters())
    ema: list[torch.Tensor] = []
    if not resume:
        # a fresh run keeps no diagnostics of an earlier run in this folder
        for p in (chunk_log_path, eval_terms_path, *snapdir.glob("e*.pt")):
            p.unlink(missing_ok=True)
    if diag:
        snapdir.mkdir(exist_ok=True)
        if resume:
            # the resumed run rewrites the interrupted epoch: drop its rows; the kept
            # rows take the current columns (a file from before a column was added
            # would otherwise keep a header the appended rows no longer match)
            for p, cols in ((chunk_log_path, _CHUNK_LOG_COLS),
                            (eval_terms_path, _EVAL_TERM_COLS)):
                if p.exists():
                    d = pd.read_csv(p)
                    d[d["epoch"] < start_epoch].reindex(columns=cols).to_csv(p, index=False)
        # the EMA shadow restarts from the current net (also on a resume)
        ema = [p.detach().clone() for p in params_list]
        if train_gs:
            print("train: diagnostics — whole-chunk graphs: chunk_log.csv carries "
                  "loss and gradient norm only (the family split needs the "
                  "segmented/eager path)", flush=True)
        print(f"train: diagnostics on — {chunk_log_path.name}, "
              f"{eval_terms_path.name}, per-epoch snapshots + EMA shadow "
              f"(decay {cfg.ema_decay}) under {snapdir}", flush=True)
    fam_np = np.array(eobs.family) if eobs is not None else None

    def _snapshot(epoch: int, kge: float) -> None:
        """The net as selection scores it at this epoch (pre-update) plus the EMA
        shadow of the optimizer steps so far (``net_ema``, same keys) —
        loadable wherever best.pt is (no optimizer state)."""
        ema_sd = dict(net.state_dict())
        for (name, _), e in zip(net.named_parameters(), ema, strict=True):
            ema_sd[name] = e
        torch.save({**_payload(epoch=epoch, kge=kge), "net_ema": ema_sd},
                   snapdir / f"e{epoch:03d}.pt")

    def _loss_split(basin_flow: torch.Tensor, obs_c: torch.Tensor, k: int, *,
                    terms: bool = False) -> dict[str, float]:
        """The multi-timescale chunk loss as trained, split into additive
        contributions by family — ``l_<fam>`` (or, ``terms``, ``<fam>_nnse`` /
        ``_log`` / ``_var`` (variance + bias) / ``_timing`` / ``_peak`` and
        ``uf_monthly``) — with each family's live entity count ``n_<fam>`` and
        coefficient mass ``c_<fam>`` (its share of the chunk's daily-term weight
        times daily_scale; monthly_scale for uf)."""
        res: dict[str, float] = {}
        tw = season_all[grid[k][0]:grid[k][1]]      # the chunk's Jul-Sep day window
        with torch.no_grad():
            n_fin = torch.isfinite(obs_c).sum(dim=1)
            w_all = (fam_w if fam_w is not None
                     else torch.ones(len(dom.basins), device=dev, dtype=dtype))
            ok = n_fin >= 90                     # masked_basin_loss's min_days
            # the chunk's daily denominator: the fixed all-entity weight under
            # mt_share_norm="all", else the scored entities' weight
            w_tot = (w_total if w_total is not None
                     else max(float((w_all * ok).sum()), 1.0))
            for f in ("usgs_daily", "cdec_daily"):
                tag = f.split("_")[0]
                msk = torch.as_tensor(fam_np == f, device=dev)
                wf = w_all * msk
                w_f = float((wf * ok).sum())
                res[f"n_{tag}"] = int((ok & msk).sum())
                res[f"c_{tag}"] = daily_scale * w_f / w_tot
                sc = daily_scale * max(w_f, 1.0) / w_tot

                def _l(log_l: float, var_l: float, bias_l: float,
                       tim_l: float = 0.0, pk_l: float = 0.0) -> float:
                    return float(masked_basin_loss(
                        basin_flow, obs_c, calobs.obs_var, kind=cfg.loss,
                        log_lambda=log_l, log_eps=cfg.log_loss_eps,
                        var_lambda=var_l, var_gate_frac=cfg.var_gate_frac,
                        var_huber_cap=cfg.var_huber_cap, bias_lambda=bias_l,
                        timing_lambda=tim_l, peak_lambda=pk_l, **shape_kw,
                        timing_window=tw, weight=wf))

                full = _l(cfg.log_loss_lambda, cfg.var_loss_lambda, cfg.bias_loss_lambda,
                          cfg.timing_loss_lambda, cfg.peak_loss_lambda)
                if terms:
                    nnse = _l(0.0, 0.0, 0.0)
                    nl = _l(cfg.log_loss_lambda, 0.0, 0.0)
                    nlv = _l(cfg.log_loss_lambda, cfg.var_loss_lambda, cfg.bias_loss_lambda)
                    nlvt = (_l(cfg.log_loss_lambda, cfg.var_loss_lambda,
                               cfg.bias_loss_lambda, cfg.timing_loss_lambda)
                            if cfg.timing_loss_lambda > 0.0 else nlv)
                    res[f"{tag}_nnse"] = sc * nnse
                    res[f"{tag}_log"] = sc * (nl - nnse)
                    res[f"{tag}_var"] = sc * (nlv - nl)
                    res[f"{tag}_timing"] = sc * (nlvt - nlv)
                    res[f"{tag}_peak"] = sc * (full - nlvt)
                else:
                    res[f"l_{tag}"] = sc * full
                    # the shape terms' parts of l_<fam> (0 when off)
                    tp = cfg.timing_loss_lambda > 0.0 or cfg.peak_loss_lambda > 0.0
                    nlv = (_l(cfg.log_loss_lambda, cfg.var_loss_lambda, cfg.bias_loss_lambda)
                           if tp else full)
                    nlvt = (_l(cfg.log_loss_lambda, cfg.var_loss_lambda,
                               cfg.bias_loss_lambda, cfg.timing_loss_lambda)
                            if cfg.timing_loss_lambda > 0.0 else nlv)
                    res[f"lt_{tag}"] = sc * (nlvt - nlv)
                    res[f"lp_{tag}"] = sc * (full - nlvt)
            if mt_targets is not None:
                mb, mtgt, mfin = mt_targets[k]
                n_m = int((mfin.sum(dim=1) >= 1).sum())
                res["n_uf"] = n_m
                res["c_uf"] = (monthly_scale * n_m / mt_total if mt_total is not None
                               else monthly_scale if n_m else 0.0)
                res["uf_monthly" if terms else "l_uf"] = monthly_scale * float(
                    monthly_nnse_loss(basin_flow[m_rows] @ mb, mtgt, mfin,
                                      eobs.var_monthly, n_total=mt_total))
        return res

    def _eval_terms(epoch: int) -> None:
        """Eval-mode loss of the selection pass, chunk by chunk on the training
        grid, averaged over the live chunks like the logged training loss."""
        sim = eval_sim.pop("sim")
        acc: dict[str, float] = {}
        n = 0
        for k, (c0, ce) in enumerate(grid):
            if not chunk_live[k]:
                continue
            d = _loss_split(sim[:, c0 - calobs.t0:ce - calobs.t0],
                            _obs_chunk(calobs, c0, ce), k, terms=True)
            d = {key: v for key, v in d.items() if not key.startswith(("n_", "c_"))}
            d["eval_loss"] = sum(d.values())
            for key, v in d.items():
                acc[key] = acc.get(key, 0.0) + v
            n += 1
        row = {"epoch": epoch, **{key: v / max(n, 1) for key, v in acc.items()}}
        pd.DataFrame([row]).reindex(columns=_EVAL_TERM_COLS).to_csv(
            eval_terms_path, mode="a", index=False,
            header=not eval_terms_path.exists())

    print(f"train[{variant}]: {dom.n_hru} HRUs"
          + (f" on {dom.n_phys} distinct cells (cell dedup)" if dom.dedup is not None else "")
          + f", {len(dom.basins)} basins, "
          f"{x.shape[1]} features | cal days {calobs.t0}..{calobs.t1} "
          + ("(cycle spinup) " if cfg.spinup_mode == "cycle" else
             f"(spinup from day {spin_t0}"
             f"{' = record start' if spin_t0 == 0 else f' = {dom.dates[spin_t0].date()}'}) ")
          + f"({cfg.chunk_grid} grid: {_grid_summary(grid)}) | {cfg.loss} loss "
          f"(log lambda {cfg.log_loss_lambda}) | n_inc={cfg.n_inc} "
          f"perc={cfg.perc_mode} {cfg.dtype} on {dev.type}"
          + (f' [cuda-graphs {sorted(train_gs)}]' if train_gs else
             f' [cuda-graphs x{cfg.train_graph_segments} segments {sorted(seg_gs)}]' if seg_gs
             else f' [cuda-graphs recompute {cfg.graph_recompute_days} days]' if rec_g is not None
             else ' [eager]'),
          flush=True)

    # subset-mode donor gate: the selection validity mask (static, obs-only)
    # and the donor-entity membership, evaluated once
    gate_valid = gate_in_donor = gate_fams = None
    if donor_gate == "subset":
        v = torch.isfinite(calobs.obs).sum(dim=1) >= 90
        if eobs is not None:
            v[m_rows] = torch.isfinite(eobs.obs_monthly).sum(dim=1) >= 12
        gate_valid = v.cpu().numpy()
        gate_in_donor = np.isin(np.array(dom.basins), list(donor_basins))
        gate_fams = np.array(eobs.family) if eobs is not None else None

    state_spin: PipelineState | None = None
    stop = False
    for epoch in range(start_epoch, cfg.n_epochs):
        tic = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

        do_eval = epoch % cfg.eval_every == 0
        need_spin = state_spin is None or (
            epoch % max(cfg.spinup_refresh_every, 1) == 0)
        pooled, per_basin = float("nan"), None
        if do_eval or need_spin:
            pooled, per_basin, state_spin = _spinup_and_maybe_eval(do_eval)
        # refresh adaptive per-basin weights in place (basin_w aliases the graph
        # loss buffer AND the eager loss reads it — copy_, never reassign)
        if basin_w is not None and per_basin is not None:
            basin_w.copy_(adaptive_basin_weights(
                per_basin.to(dtype), basin_w, beta=cfg.adaptive_loss_beta,
                momentum=cfg.adaptive_loss_momentum,
                floor=cfg.adaptive_loss_floor, clip=cfg.adaptive_loss_clip))
        spin_s = time.time() - tic

        # ep0-donor gate: with zero-init extra heads the warm-started net IS the
        # donor field, so the pre-update epoch-0 selection must reproduce the
        # donor's sel cal-KGE.  A gap means a NUMERICS/CONFIG/DATA mismatch
        # (n_inc, spinup basis, domain/footprint flags, data revision).  In
        # "same" mode the selection scalar is compared directly; in "subset"
        # mode the donor's statistic is recomputed over its own entities.
        # --init-gate abort turns a failure into a hard stop (unattended runs).
        if (donor_gate is not None and epoch == start_epoch and do_eval
                and math.isfinite(donor_kge)):
            if donor_gate == "same":
                gate_val, how = pooled, "same entity set"
            else:
                gate_val = _donor_subset_stat(
                    per_basin.detach().cpu().numpy().astype(float),
                    gate_valid, gate_in_donor, gate_fams, donor_select,
                    donor_shares)
                how = (f"{donor_select} statistic over the donor's "
                       f"{len(donor_basins)} entities")
            gap = abs(gate_val - donor_kge)
            if not math.isfinite(gap) or gap > 1e-3:
                msg = (f"ep0-donor gate FAILED — {how}: {gate_val:.4f} vs "
                       f"donor {donor_kge:.4f} (|d|={gap:.4f} > 1e-3): this "
                       "run's numerics/config/data do NOT reproduce the "
                       "donor; check --n-inc / spinup / footprint / data "
                       "revision before trusting the fine-tune")
                if cfg.init_gate == "abort":
                    print(f"train: {msg} — aborting (--init-gate abort)",
                          flush=True)
                    raise RuntimeError(msg)
                print(f"train: WARNING — {msg}", flush=True)
            else:
                print(f"train: ep0-donor gate PASSED — {how}: {gate_val:.4f} "
                      f"vs donor {donor_kge:.4f}", flush=True)

        is_best = False
        if do_eval:
            if pooled > best_kge:
                best_kge, stale, is_best = pooled, 0, True
                _save(ckdir / "best.pt", epoch=epoch, kge=pooled)
            else:
                stale += 1
            if stale > cfg.patience and epoch >= cfg.min_stop_epoch:
                print(f"train: early stop at epoch {epoch} "
                      f"({stale} stale selections)", flush=True)
                stop = True
        if diag:
            _snapshot(epoch, pooled)
            if do_eval and eobs is not None:
                _eval_terms(epoch)

        # -- TBPTT chunks (one optimizer step per chunk) --------------------
        losses: list[float] = []
        reg_losses: list[float] = []
        chunk_rows: list[dict] = []
        skipped = 0
        if not stop:
            assert state_spin is not None   # first epoch always spins up
            net.train()
            # the carried state lives in the Python variable between chunks:
            # a graphed chunk takes it through set_state and hands it back
            # through get_state (a detached clone, routing history and canopy
            # included), so graph objects of different lengths and eager
            # chunks interleave freely
            if dev.type == "cuda" and (win2 or rec_g is not None):
                # hand the cached blocks of the spinup / selection forward back to
                # the driver before the training chunks (no numerical effect): on
                # an 8 GB WDDM card they would otherwise sit beside the train graph
                # and push the process into paging
                torch.cuda.empty_cache()
            state = state_spin
            starts: list = []       # (state carried into a chunk, its c0), newest last
            if win2:
                from .graphs import _clone_state
            for k, (c0, ce) in enumerate(grid):
                burn = starts[-(n_win - 1):] if win2 else []
                # a multi-year window uses this start one or more chunks LATER: it
                # must own its storage (a graphed window's end state is a view of
                # the graph's static output buffers, which the next replay
                # overwrites in place)
                starts = (starts + [(_clone_state(state) if win2 else state, c0)]
                          )[-max(n_win - 1, 1):]
                g = train_gs.get(ce - c0)
                dead_nograd = skip_dead and not chunk_live[k] and g is None
                # tbptt_window_years n > 1: a live full-year chunk that follows
                # n - 1 full-year chunks runs the window [their first c0, ce) from
                # the state carried into that chunk, the loss on [c0, ce) only
                two = (win2 and len(burn) == n_win - 1 and chunk_live[k]
                       and not dead_nograd and g is None and ce - c0 >= 365
                       and c0 - burn[0][1] >= 365 * (n_win - 1))
                w0 = burn[0][1] if two else c0
                # the multi-timescale cal window ends AT the forcing record,
                # so its LAST chunk is short (ce < c0 + the captured length)
                # and replays eagerly; every other domain has forcing headroom
                # past cal_end
                pr, ta, doy, leap = dom.chunk(w0, ce)
                tn, tx = dom.chunk_tmm(w0, ce)
                lai_c = dom.chunk_lai(w0, ce)
                st_c = dom.chunk_state(w0, ce)
                obs_c = _obs_chunk(calobs, c0, ce)
                et_tgt = et_targets[k] if et_targets is not None else None
                swe_tgt = swe_targets[k] if swe_targets is not None else None
                split: dict[str, float] = {}
                if g is not None:
                    g.set_state(state)
                    loss = g.run(pr, ta, doy, leap, obs_c, tn, tx, lai_c,
                                 st_c, et_target=et_tgt, swe_target=swe_tgt,
                                 mt_target=(mt_targets[k]
                                            if mt_targets is not None
                                            else None))
                    state = g.get_state()
                else:
                    # mixed mode (an eager chunk beside captured graphs): grads
                    # must STAY allocated — set_to_none would invalidate the
                    # captured graphs
                    if not dead_nograd:
                        opt.zero_grad(set_to_none=not train_gs)
                    # a dead chunk under dead_chunk_nograd: the same train-mode
                    # net(x) (one dropout draw) and the same graphed forward,
                    # without autograd
                    with torch.no_grad() if dead_nograd else contextlib.nullcontext():
                        params, cp = _split_out(net(x_net), cfg.et_mode)
                        uh = routing_uh(params, dom.flowlen, row_cell=dom.row_cell)
                        if two:
                            state = burn[0][0]      # the burn-in's carried state
                        if carry_grad and not dead_nograd:
                            state = _relative_carry(state, params,
                                                    flux=cfg.tbptt_carry == "flux")
                        # segmented CUDA graphs: graphed run_window over the longest
                        # captured length that fits, autograd across the segments
                        # (graphs.SegmentedTrainWindow); the days beyond it (a leap
                        # year's last day on the water-year grid) continue eagerly
                        # from the graphed state, autograd through both
                        n_g = max((L for L in seg_gs if L <= ce - w0), default=0)
                        if rec_g is not None:
                            # activation recompute over the whole window (under
                            # no_grad for a dead chunk: a plain graphed forward)
                            n_g = -1
                            flow_g, state_g = rec_g.forward(w0, ce - w0, params, uh, cp, state)
                        elif dead_nograd and win2 and nograd_g is not None:
                            # the multi-year mode captures no 1-year train graph:
                            # a dead chunk replays the forward-only dead-chunk
                            # window (the stream window if that capture failed)
                            n_g = -1
                            dg = nograd_dead if nograd_dead is not None else nograd_g
                            dg.set_params(params, uh, canopy_params=cp)
                            _, state_g = _stream_nograd(dom, cfg, params, uh, c0, ce, state,
                                                        graph=dg, canopy_params=cp)
                            flow_g = None
                        elif n_g:
                            flow_g, state_g = seg_gs[n_g].forward(w0, params, uh, cp, state)
                        else:
                            flow_g, state_g = None, state

                        def _eager(d0: int):
                            sl = slice(d0, None)
                            return run_window(
                                pr[:, sl], ta[:, sl], doy[sl], leap[sl], dom.phys_lat_rad,
                                dom.phys_elev,
                                params, uh, state_g, n_inc=cfg.n_inc, perc_mode=cfg.perc_mode,
                                fracp_floor=cfg.fracp_floor, ninc_mode="fixed",
                                et_mode=cfg.et_mode, canopy_params=cp,
                                tmin=None if tn is None else tn[:, sl],
                                tmax=None if tx is None else tx[:, sl],
                                veg_frac=dom.phys_veg_frac,
                                lai=None if lai_c is None else lai_c[:, sl],
                                noah_pet=cfg.noah_pet,
                                sac_pet=cfg.sac_pet, pt_snow_albedo=cfg.pt_snow_albedo,
                                pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                                canopy_lite=cfg.canopy_lite,
                                sac_exchanges=cfg.noah_sac_exchanges,
                                state_idx=None if st_c is None else st_c[:, sl],
                                return_tet=et_tgt is not None,
                                return_swe=swe_tgt is not None,
                                row_cell=dom.row_cell)

                        if n_g == -1 or n_g == ce - w0:
                            res = (flow_g, state_g)
                        elif n_g:
                            res_e = _eager(n_g)
                            res = (torch.cat([flow_g, res_e[0]], dim=1), res_e[1], *res_e[2:])
                        else:
                            res = _eager(0)
                        flow, state = res[0], res[1]
                        if two:
                            flow = flow[:, c0 - w0:]    # the scored year only
                    if dead_nograd:
                        # no loss, no backward, no step: the carried state is the
                        # one a full dead chunk leaves
                        if diag:
                            chunk_rows.append({"epoch": epoch, "k": k,
                                               "start": str(dom.dates[c0].date()),
                                               "days": ce - c0, "live": False,
                                               "nograd": True, "stepped": False})
                        continue
                    basin_flow = dom.W @ flow
                    loss_t = daily_scale * masked_basin_loss(
                        basin_flow, obs_c, calobs.obs_var, kind=cfg.loss,
                        log_lambda=cfg.log_loss_lambda, log_eps=cfg.log_loss_eps,
                        var_lambda=cfg.var_loss_lambda,
                        var_gate_frac=cfg.var_gate_frac,
                        var_huber_cap=cfg.var_huber_cap,
                        bias_lambda=cfg.bias_loss_lambda,
                        timing_lambda=cfg.timing_loss_lambda,
                        peak_lambda=cfg.peak_loss_lambda, **shape_kw,
                        timing_window=season_all[c0:ce],
                        weight=fam_w if fam_w is not None else basin_w,
                        weight_total=w_total)
                    if mt_targets is not None:
                        mb, mtgt, mfin = mt_targets[k]
                        loss_t = loss_t + monthly_scale * monthly_nnse_loss(
                            basin_flow[m_rows] @ mb, mtgt, mfin,
                            eobs.var_monthly, n_total=mt_total)
                    ri = 2
                    if et_tgt is not None:
                        et_monthly = (dom.W @ res[ri]) @ et_tgt[0]
                        ri += 1
                        if cfg.et_loss_lambda > 0.0:
                            loss_t = loss_t + cfg.et_loss_lambda * shape_pull_loss(
                                et_monthly, et_tgt[1], et_tgt[2], et_tgt[3])
                        if cfg.et_level_lambda > 0.0:
                            loss_t = loss_t + cfg.et_level_lambda * level_hinge_loss(
                                et_monthly, et_tgt[3], et_tgt[4], et_tgt[5])
                    if swe_tgt is not None:
                        swe_monthly = (dom.W @ res[ri]) @ swe_tgt[0]
                        loss_t = loss_t + cfg.swe_loss_lambda * shape_pull_loss(
                            swe_monthly, swe_tgt[1], swe_tgt[2], swe_tgt[3],
                            basin_w=swe_w)
                    loss_t.backward()
                    loss, state = float(loss_t.detach()), state.detach()
                    if diag and eobs is not None:
                        split = _loss_split(basin_flow, obs_c, k)
                # spatial-smoothness penalty: an eager net(x) whose gradient
                # ACCUMULATES onto the data-loss grads already in .grad (works
                # for both the graph and eager paths) — the CUDA graph is never
                # touched, so capture stability is unaffected.
                if reg_lambda > 0.0:
                    reg_params, _ = _split_out(net(x), cfg.et_mode)
                    reg_t = reg_lambda * spatial_smoothness(
                        reg_params, e_i, e_j, e_w, p_scale, p_islog)
                    reg_t.backward()
                    reg_losses.append(float(reg_t.detach()))
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                      cfg.grad_clip)
                stepped = False
                if not chunk_live[k]:
                    pass       # dead chunk: state advanced, no update
                elif math.isfinite(loss) and bool(torch.isfinite(norm)):
                    opt.step()
                    stepped = True
                    if diag:
                        with torch.no_grad():
                            for e, p in zip(ema, params_list, strict=True):
                                e.lerp_(p, 1.0 - cfg.ema_decay)
                else:
                    skipped += 1
                opt.zero_grad(set_to_none=False)
                if chunk_live[k]:
                    losses.append(loss)
                if diag:
                    # gnorm = the pre-clip total gradient norm
                    chunk_rows.append({"epoch": epoch, "k": k,
                                       "start": str(dom.dates[c0].date()),
                                       "days": ce - c0, "live": chunk_live[k],
                                       "nograd": False, "stepped": stepped,
                                       "loss": loss, "gnorm": float(norm), **split})
            sched.step()
        if chunk_rows:
            pd.DataFrame(chunk_rows).reindex(columns=_CHUNK_LOG_COLS).to_csv(
                chunk_log_path, mode="a", index=False,
                header=not chunk_log_path.exists())

        vram = (torch.cuda.max_memory_allocated(dev) / 2**20
                if dev.type == "cuda" else 0.0)
        row = {"epoch": epoch,
               "loss": (sum(losses) / len(losses)) if losses else float("nan"),
               "cal_kge": pooled, "lr": sched.get_last_lr()[0],
               "spinup_s": round(spin_s, 1),
               "epoch_s": round(time.time() - tic, 1),
               "peak_vram_mb": round(vram), "skipped_steps": skipped}
        if reg_lambda > 0.0:
            row["reg"] = (sum(reg_losses) / len(reg_losses)
                          if reg_losses else float("nan"))
        if cfg.spinup_mode == "cycle":
            # passes of this epoch's spinup (0: the state was reused) and the last
            # pass's largest basin annual-flow change (relative)
            row["spin_cycles"] = spin_cache.pop("cycles", 0)
            row["spin_change"] = spin_cache.pop("change", float("nan"))
        pd.DataFrame([row]).to_csv(log_path, mode="a", index=False,
                                   header=not log_path.exists())
        if not stop:
            _save(ckdir / "last.pt", epoch=epoch, kge=pooled)
        print(f"  epoch {epoch:3d}/{cfg.n_epochs}  loss {row['loss']:.4f}  "
              f"calKGE {pooled:.4f}{'*' if is_best else ' '}  "
              f"lr {row['lr']:.1e}  spin {spin_s:.0f}s  "
              f"epoch {row['epoch_s']:.0f}s  vram {row['peak_vram_mb']}MB"
              + (f"  reg {row['reg']:.4f}" if reg_lambda > 0.0 else "")
              + (f"  spin x{row['spin_cycles']}" if cfg.spinup_mode == "cycle" else "")
              + (f"  skipped {skipped}" if skipped else ""), flush=True)
        if stop:
            break

    # final selection on the post-training net (loop evals are pre-update).  An
    # early stop comes before the stop epoch's training, so that epoch's
    # selection (a stale score, in ``pooled``) already scored the final net.
    if not stop:
        pooled, _, _ = _spinup_and_maybe_eval(True)
        if diag:
            _snapshot(cfg.n_epochs, pooled)
            if eobs is not None:
                _eval_terms(cfg.n_epochs)
        if pooled > best_kge:
            best_kge = pooled
            _save(ckdir / "best.pt", epoch=cfg.n_epochs, kge=pooled)
    print(f"train[{variant}]: done — best selection cal KGE {best_kge:.4f} "
          f"(final {pooled:.4f}) -> {ckdir / 'best.pt'}", flush=True)
    return out
