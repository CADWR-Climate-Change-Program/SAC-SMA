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
2. **Selection** (every ``eval_every`` epochs, pre-update; with ``select_cpu`` every
   epoch on the CPU engine, :mod:`sacsma.dpl.select_cpu`) — continue no-grad
   through the full calibration window and score pooled mean per-basin KGE
   (the GA-comparable exact objective).  Best net -> ``checkpoints/best.pt``;
   early stop once more than ``patience`` evaluations in a row are stale.  Validation observations
   are never read (``load_cal_obs`` does not materialize them).
3. **TBPTT** — one chunk per water year (1 Oct to 1 Oct, 365 or 366 days, the
   last to the window or record end), so no calendar month is ever split; one
   AdamW step per chunk on the chunk-additive NNSE loss.  A live chunk runs the
   previous water year and its own with autograd through both and the loss on
   its own year only (the previous year is a gradient-carrying burn-in), so the
   gradient sees how a parameter shapes the water carried into the scored year;
   the state and the 106-day routing-tail history carried into the window are
   detached, the SAC contents carried relative to their capacities
   (:func:`_relative_carry`).  The carried state advances one water year per step;
   dead chunks (no scoreable observation) advance it forward-only, and the first
   chunk and the short envelope tail run one-year windows.  The recompute graph
   replays one captured segment over any window.

For the multi-timescale domain (``domain="multifamily"``) the same protocol
runs over the training entities on the registry envelope (WY1950-2018): each
daily entity joins the chunk NNSE window-masked (NaN outside its own
``train_start``/``train_end``), the monthly entities add a chunk-additive
monthly NNSE term (simulated daily flow bucketed to complete calendar
months), and selection scores per-entity KGE at each entity's native
timescale — pooled mean by default; with numeric family shares
("usgs=0.27,cdec=0.54,uf=0.19") the share-weighted family mean, so
checkpoint choice follows the training objective.

On CUDA the day-stepped pipeline runs as captured CUDA graphs
(:mod:`sacsma.dpl.graphs`) — eager execution is dispatch-bound.  Graph
capture failure (or ``--device cpu``) falls back to eager with identical
numerics.
"""

from __future__ import annotations

import atexit
import contextlib
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import paths
from ..io import MULTI_TIMESCALE_DOMAIN, load_hru_table, load_params, soilveg_path
from .config import (
    FAMILY_KEYS,
    MONTHLY_FAMILIES,
    DplConfig,
    config_from_checkpoint,
    family_loss_refs,
    family_shares,
    pick_device,
)
from .data import (
    CalObs,
    DomainTensors,
    load_cal_obs,
    load_domain_tensors,
)
from .features import FeatureSet, aef_store, build_features
from .forward import PipelineState, initial_state, routing_uh, run_window
from .loss import kge_torch, masked_basin_loss
from .multi_timescale import (
    cut_entity_obs,
    load_entity_obs,
    monthly_chunk_target,
    monthly_nnse_loss,
)
from .parameter_net import LOGIT_CAP, ParameterNet, ga_priors, logit_penalty
from .select_cpu import read_scores, replace_retry, selection_stat, start_scorer

_DTYPES = {"float32": torch.float32, "float64": torch.float64}

#: diagnostics file columns (cfg.diagnostics).  chunk_log.csv: one row per chunk
#: per epoch — ``n_`` live entities, ``c_`` coefficient mass and ``l_`` loss
#: contribution by family (the ``l_`` sum to ``loss``), ``gnorm`` the pre-clip gradient
#: norm.
#: eval_terms.csv: one row per selection epoch — the eval-mode
#: loss on the training grid by family and term, averaged over the live chunks.
_CHUNK_LOG_COLS = ("epoch", "k", "start", "days", "live", "nograd", "stepped",
                   "loss", "gnorm", "n_usgs", "n_cdec", "n_uf", "c_usgs",
                   "c_cdec", "c_uf", "l_usgs", "l_cdec", "l_uf")
_EVAL_TERM_COLS = ("epoch", "eval_loss", "usgs_nnse", "usgs_log", "usgs_var",
                   "cdec_nnse", "cdec_log", "cdec_var", "uf_monthly")
#: the CalSim3 arc family's columns, appended only when a run trains it (the files of
#: every other run keep the columns above)
_CS_CHUNK_LOG_COLS = ("n_cs", "c_cs", "l_cs")
_CS_EVAL_TERM_COLS = ("cs_monthly",)
#: DplConfig fields a --resume must repeat exactly (the objective)
_RESUME_FIXED = ("log_loss_lambda", "log_loss_eps", "var_loss_lambda", "var_huber_cap",
                 "shape_min_days", "mt_family_weight",
                 "obs_mask", "pxtemp_learn", "pxtemp_box", "pxtemp_tau",
                 "mt_share_norm", "mt_loss_ref", "mt_loss_ref_power",
                 "holdout_wy", "uf_train_start", "train_window", "calsim_arcs", "mt_select_weight",
                 "log_space_params", "param_bounds", "logit_penalty", "select_cpu",
                 "dropout", "grad_clip")


def _stream_nograd(
    dom: DomainTensors, cfg: DplConfig,
    params: dict[str, torch.Tensor], uh, t0: int, t1: int,
    state: PipelineState, *, graph=None, collect: bool = False,
) -> tuple[torch.Tensor | None, PipelineState]:
    """No-grad basin flow over [t0, t1): graph replays + an eager remainder."""
    outs: list[torch.Tensor] = []
    t = t0
    if graph is not None:
        graph.set_state(state)
        while t + graph.length <= t1:
            basin = graph.replay(dom.window(t, t + graph.length))
            if collect:
                outs.append(basin.clone())
            t += graph.length
        state = graph.get_state()
    physics = cfg.physics()
    with torch.no_grad():
        while t < t1:
            te = min(t + cfg.nograd_window, t1)
            flow, state = run_window(dom.window(t, te), dom.lat_rad, dom.elev,
                                     params, uh, state, physics, veg_frac=dom.veg_frac)
            if collect:
                outs.append(dom.W @ flow)
            t = te
    return (torch.cat(outs, dim=1) if collect else None), state


def _relative_carry(state: PipelineState, params: dict[str, torch.Tensor]) -> PipelineState:
    """The state carried into a TBPTT chunk with each SAC content ``c`` rewritten as
    ``c * (cap / cap.detach())``: the value is unchanged (x / x == 1 exactly for the
    positive capacities), but the backward sees the store's relative saturation as
    fixed, ``dc/dcap = c / cap``, instead of the detached content treating a larger
    capacity as free deficit at every window start (with one-year windows that
    truncation overweighted the tension capacities' pull 5-170x against the full
    sequence).  ADIMC is carried against ``uztwm + lztwm``; snow and the routing
    history pass unchanged."""
    s = state.sac
    adim_cap = params["uztwm"] + params["lztwm"]

    def rel(c: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
        # clone: the product saves c for its backward, and after a graphed chunk c
        # is a view of the graph's static output buffer, which the next replay
        # overwrites in place (no version bump) — the backward would read the
        # END-of-chunk contents
        return c.detach().clone() * (cap / cap.detach())

    sac = type(s)(uztwc=rel(s.uztwc, params["uztwm"]), uzfwc=rel(s.uzfwc, params["uzfwm"]),
                  lztwc=rel(s.lztwc, params["lztwm"]), lzfsc=rel(s.lzfsc, params["lzfsm"]),
                  lzfpc=rel(s.lzfpc, params["lzfpm"]), adimc=rel(s.adimc, adim_cap))
    return PipelineState(snow=state.snow, sac=sac, hist_surf=state.hist_surf,
                         hist_base=state.hist_base)


def _obs_chunk(calobs: CalObs, c0: int, c1: int) -> torch.Tensor:
    """Cal-window obs for record days [c0, c1); NaN outside the window."""
    out = torch.full((calobs.obs.shape[0], c1 - c0), float("nan"),
                     device=calobs.obs.device, dtype=calobs.obs.dtype)
    lo, hi = max(c0, calobs.t0), min(c1, calobs.t1)
    if hi > lo:
        out[:, lo - c0:hi - c0] = calobs.obs[:, lo - calobs.t0:hi - calobs.t0]
    return out


def _chunk_grid(dates: pd.DatetimeIndex, t0: int, t1: int,
                n_time: int) -> list[tuple[int, int]]:
    """The TBPTT chunks as ``(c0, ce)`` record-day pairs covering ``[t0, t1)``: from ``t0``
    (which must be a 1 Oct) to each following 1 Oct, so 365 or 366 days, the last chunk
    running to ``t1`` or the record end, whichever comes first — every calendar month
    whole."""
    d0 = dates[t0]
    if (d0.month, d0.day) != (10, 1):
        raise ValueError(f"the water-year chunks need a window starting on 1 Oct, not {d0.date()}")
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


def _cal_kge(sim: torch.Tensor, obs: torch.Tensor, min_days: int = 90) -> float:
    """The pooled mean per-basin cal KGE over the valid basins."""
    k = kge_torch(sim.double(), obs.double())    # full-record stats in f64
    m = torch.isfinite(obs).sum(dim=1) >= min_days
    return float(k[m].mean())


def _feature_stats(fs: FeatureSet) -> dict:
    d = asdict(fs)
    d.pop("x")                       # rebuildable; keep checkpoints small
    return d


def _with_calsim_arcs(data_dir: str, basins: tuple[str, ...] | None) -> tuple[str, ...]:
    """``DplConfig.calsim_arcs = "train_default"``: the run's entities (``basins``, or the
    registry's base entities) followed by the ``train_default`` arcs of
    ``arc_hierarchy.csv`` as ``cs_<ARC>`` registry entities, in hierarchy file order."""
    from .calsim.arcs import load_hierarchy
    from .multi_timescale import CALSIM_FAMILY

    reg = pd.read_csv(paths.entities(data_dir),
                      usecols=["entity_id", "family", "site_id"])
    if basins is None:
        basins = tuple(reg.loc[reg["family"] != CALSIM_FAMILY, "entity_id"])
    cs_reg = reg[reg["family"] == CALSIM_FAMILY].set_index("site_id")["entity_id"]
    if set(basins) & set(cs_reg):
        raise ValueError("calsim_arcs='train_default' appends the arc family itself — "
                         "drop the cs_ ids from --basins")
    hier = load_hierarchy(data_dir)
    # exactly True (a NaN would read True under astype(bool)); every one is tier A
    td = hier["train_default"].eq(True)
    if (hier.loc[td, "tier"] != "A").any():
        raise ValueError("arc_hierarchy: train_default arcs outside tier A "
                         f"{hier.loc[td & (hier['tier'] != 'A'), 'arc'].tolist()[:5]}")
    arcs = hier.loc[td, "arc"].tolist()
    missing = [a for a in arcs if a not in cs_reg.index]
    if missing:
        raise ValueError(f"calsim_arcs: train_default arcs without a registry entity "
                         f"{missing[:5]} (rebuild entities.csv with --calsim-arcs)")
    return tuple(basins) + tuple(cs_reg[a] for a in arcs)


def _area_shares(data_dir: str, basins, families=None) -> dict[str, float]:
    """Footprint-area family shares: each family's registry ``area_mi2`` summed over
    ``basins`` (restricted to ``families`` when given), renormalized — the footprint-area
    rule."""
    reg = pd.read_csv(paths.entities(data_dir),
                      usecols=["entity_id", "family", "area_mi2"]).set_index("entity_id")
    area = reg.loc[list(basins)].groupby("family", sort=False)["area_mi2"].sum()
    area = {f: float(area[f]) for f in FAMILY_KEYS.values()
            if f in area.index and (families is None or f in families)}
    tot = sum(area.values())
    return {f: a / tot for f, a in area.items()}


def _print_holdout(dom: DomainTensors, eobs, cfg: DplConfig) -> None:
    """The launch-gate counts of ``holdout_wy`` / ``uf_train_start``: values removed and
    entities touched per family, the (asserted) absence of any finite target inside the
    holdout, the daily entities still valid for selection (>= 90 days) and the months
    the uf back-extension adds."""
    fam = np.array(eobs.family)
    if cfg.holdout_wy:
        a, b = cfg.holdout_wy
        nh = np.array(eobs.n_holdout)
        win = dom.dates[eobs.t0:eobs.t1]
        wy_d = np.asarray(win.year + (win.month >= 10))
        mc = eobs.month_code
        wy_m = mc // 12 + (mc % 12 >= 9)
        left = (int(torch.isfinite(eobs.obs_daily[:, torch.as_tensor(
                    (wy_d >= a) & (wy_d <= b), device=eobs.obs_daily.device)]).sum())
                + int(torch.isfinite(eobs.obs_monthly[:, torch.as_tensor(
                    (wy_m >= a) & (wy_m <= b), device=eobs.obs_monthly.device)]).sum()))
        if left:
            raise RuntimeError(f"holdout WY{a}-{b}: {left} finite target(s) left inside it")
        parts = [f"{f.split('_')[0]} {int(nh[fam == f].sum())} "
                 f"({int(((fam == f) & (nh > 0)).sum())} of {int((fam == f).sum())} entities)"
                 for f in FAMILY_KEYS.values() if (fam == f).any()]
        n_d = torch.isfinite(eobs.obs_daily).sum(dim=1).cpu().numpy()
        dfam = fam[eobs.daily_rows]
        valid = [f"{f.split('_')[0]} {int(((dfam == f) & (n_d >= 90)).sum())}/"
                 f"{int((dfam == f).sum())}" for f in ("usgs_daily", "cdec_daily")
                 if (dfam == f).any()]
        print(f"train: holdout WY{a}-{b} blanked in every family (after the n_obs audit "
              "and obs_mask, before the normalizers) — removed " + ", ".join(parts)
              + "; 0 finite targets left inside it; daily entities still valid (>= 90 d): "
              + ", ".join(valid), flush=True)
    if cfg.uf_train_start:
        ne = np.array(eobs.n_ext)
        uf = fam == "uf_monthly"
        per = sorted(set(ne[uf].tolist()))
        print(f"train: uf back-extension from {cfg.uf_train_start} — +{int(ne[uf].sum())} "
              f"months over {int(uf.sum())} uf entities ({'/'.join(map(str, per))} each, "
              "holdout excluded, every one observed); registry windows and their n_obs "
              "audit unchanged", flush=True)


def train(
    variant: str = "physical",
    data_dir: str = "data",
    out_dir: str | Path | None = None,
    cfg: DplConfig | None = None,
    *,
    resume: bool = False,
    basins: tuple[str, ...] | None = None,   # a subset of the domain's entities
    domain: str = "15cdec",                  # 15cdec HRU cloud | 15cdec_grid
                                             # native grid | multifamily
                                             # multi-timescale entities
) -> Path:
    """Train one feature variant (``physical`` or ``aef_u``); returns the run's model
    folder.  ``out_dir`` names the run (any of its folders): checkpoints and the training
    log go to its model folder, the diagnostics logs and per-epoch snapshots to its local
    folder."""
    cfg = cfg or DplConfig()
    physics = cfg.physics()
    dev = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    dtype = _DTYPES[cfg.dtype]
    run = paths.run_roles(out_dir if out_dir is not None
                          else paths.dpl_run(run=variant, domain=domain))
    out = run.model
    ckdir = out / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.csv"

    if domain != MULTI_TIMESCALE_DOMAIN and (cfg.holdout_wy or cfg.uf_train_start
                                              or cfg.train_window
                                              or cfg.calsim_arcs != "none"):
        raise ValueError("holdout_wy / uf_train_start / train_window / calsim_arcs are wired "
                         "for the multi-timescale domain only")
    if cfg.calsim_arcs == "train_default":
        basins = _with_calsim_arcs(data_dir, basins)
    dom = load_domain_tensors(data_dir, domain=domain, device=dev, dtype=dtype, basins=basins)
    eobs = d_rows = m_rows = midx = None
    if domain == MULTI_TIMESCALE_DOMAIN:
        eobs = load_entity_obs(dom, data_dir, obs_mask=cfg.obs_mask,
                               holdout_wy=cfg.holdout_wy or None,
                               uf_train_start=cfg.uf_train_start or None,
                               train_window=cfg.train_window or None)
        if cfg.train_window:
            eobs = cut_entity_obs(eobs, dom.dates, cfg.train_window)
            print(f"train: training window {cfg.train_window[0]}..{cfg.train_window[1]} — "
                  f"{sum(eobs.n_out)} target value(s) outside it blanked; the chunks start at "
                  "it, the spinup runs ahead of it", flush=True)
        if cfg.obs_mask:
            print(f"train: obs_mask — {len(cfg.obs_mask)} hand-confirmed daily "
                  f"observation(s) masked: {', '.join(cfg.obs_mask)}", flush=True)
        if cfg.holdout_wy or cfg.uf_train_start:
            _print_holdout(dom, eobs, cfg)
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
        calobs = load_cal_obs(dom, data_dir, cal_start=cfg.cal_start, obs_mask=cfg.obs_mask)
    print(f"train: variance term over the record std (lambda {cfg.var_loss_lambda}, Huber "
          f"cap {cfg.var_huber_cap}) on basin-chunks with >= {cfg.shape_min_days} valid days; "
          "every entity-chunk term weighted by its valid days / 365 (months / 12)", flush=True)
    # family weighting (multi-timescale only): each family's share of the
    # loss, renormalized over the families this run trains; entities weigh
    # equally within a family.  The daily term carries the daily families'
    # shares (their ratio set through the per-entity weights, the sum through
    # daily_scale) and the monthly term the monthly family's share; "none" =
    # the unweighted baseline.
    fam_w = None
    daily_scale = monthly_scale = 1.0
    shares = None
    # the monthly families this run trains, and (share runs) their own scales
    m_fams: list[str] = []
    monthly_scale_f: dict[str, float] = {}
    if eobs is not None:
        m_fam_np = np.array(eobs.family)[eobs.monthly_rows]
        m_fams = [f for f in MONTHLY_FAMILIES if (m_fam_np == f).any()]
        shares = family_shares(cfg.mt_family_weight)
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
        # every monthly family is its own monthly term (a single one = the pooled term)
        monthly_scale_f = {f: shares.get(f, 0.0) for f in m_fams}
        print("train: family weighting by SHARES — "
              + ", ".join(f"{f.split('_')[0]} {s:.3f}" for f, s in shares.items())
              + f"; daily x {daily_scale:.3f}, "
              + (f"monthly x {monthly_scale:.3f}; " if len(m_fams) <= 1 else
                 "monthly " + ", ".join(f"{f.split('_')[0]} x {v:.3f}"
                                        for f, v in monthly_scale_f.items()) + "; ")
              + "selection = share-weighted family mean", flush=True)
    # checkpoint selection: the loss shares, unless mt_select_weight sets its own (e.g.
    # the area shares when the loss shares were re-solved to REALIZE them); with the
    # CalSim3 arcs, sel3 (logged, not selected on) = the footprint-area rule over the other
    # families — their footprint-area shares
    sel_shares, sel3_shares = shares, None
    if shares is not None and cfg.mt_select_weight:
        sw = (_area_shares(data_dir, dom.basins) if cfg.mt_select_weight == "area"
              else family_shares(cfg.mt_select_weight))
        miss = [str(f) for f in shares if f not in sw]
        if miss:
            raise ValueError(f"mt_select_weight {cfg.mt_select_weight!r} gives no share "
                             f"to {miss} (present in this run)")
        tot = sum(sw[f] for f in shares)
        sel_shares = {f: sw[f] / tot for f in shares}
        print(f"train: selection shares (mt_select_weight={cfg.mt_select_weight}) — "
              + ", ".join(f"{f.split('_')[0]} {s:.4f}" for f, s in sel_shares.items())
              + "; the loss keeps the shares above", flush=True)
    if shares is not None and "calsim_monthly" in shares and len(shares) > 1:
        sel3_shares = _area_shares(data_dir, dom.basins,
                                   families=[f for f in shares if f != "calsim_monthly"])
        print("train: sel3 (logged beside the selection scalar) = the footprint-area rule without "
              "the arcs, the footprint-area shares "
              + ", ".join(f"{f.split('_')[0]} {s:.4f}" for f, s in sel3_shares.items()),
              flush=True)
    # mt_share_norm="all": fixed loss denominators, so every entity carries
    # share_f / n_f in every chunk it is scored in (a chunk holding one family
    # no longer hands it the whole term); "present" keeps the per-chunk
    # renormalization (weight_total / mt_total None -> byte-identical)
    w_total = mt_total = None
    mt_total_f: dict[str, float] = {}       # per monthly family (its own entity count)
    if shares is not None and cfg.mt_share_norm == "all":
        if fam_w is not None:
            w_total = float(fam_w.sum())
        if eobs is not None and len(eobs.monthly_rows):
            mt_total = float(len(eobs.monthly_rows))
            mt_total_f = {f: float((m_fam_np == f).sum()) for f in m_fams}
        print(f"train: family shares normalized over ALL entities (mt_share_norm=all) — "
              f"daily denominator {w_total}, "
              + (f"monthly {mt_total}" if len(m_fams) <= 1 else
                 "monthly " + ", ".join(f"{f.split('_')[0]} {v}"
                                        for f, v in mt_total_f.items()))
              + ": a chunk's family term scales with the share of that family's entities "
              "it scores", flush=True)
    # mt_loss_ref: the FROZEN per-family loss scale kappa_f = Lbar / L_ref_f^p.  Set
    # once, before any graph capture, and never changed: the daily families' kappa
    # goes into fam_w IN PLACE (the tensor every loss path reads), AFTER w_total was
    # taken from the unscaled weights (so the fixed denominator is unchanged and kappa
    # is not normalized away); uf's goes into monthly_scale, a Python float constant
    # for the run.  shares (selection) stay
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
        monthly_scale_f = {f: v * kappa[f] for f, v in monthly_scale_f.items()}
        print("train: FROZEN family loss scale (mt_loss_ref) — reference per-entity levels "
              + ", ".join(f"{f.split('_')[0]} {refs[f]:.4g}" for f in shares)
              + f"; kappa = {lbar:.4g} / L_ref^{pw:g}: "
              + ", ".join(f"{f.split('_')[0]} x {kappa[f]:.4f}" for f in shares)
              + (f" (monthly x {monthly_scale:.4f} in all)" if len(m_fams) <= 1 else
                 " (monthly " + ", ".join(f"{f.split('_')[0]} x {v:.4f}"
                                          for f, v in monthly_scale_f.items()) + " in all)")
              + "; chunk_log c_/l_ carry kappa, selection keeps the "
              + ("nominal shares" if not cfg.mt_select_weight else
                 "mt_select_weight shares"), flush=True)
    # the monthly terms: (log tag, dom.basins rows, positions in eobs.monthly_rows or
    # None = all of them, scale, fixed denominator).  One pooled term over every
    # monthly row unless a share run holds more than one monthly family — then one per
    # family, each with its own share x kappa and entity count (the arcs must not
    # dilute uf's per-entity coefficient share/9 to share/73)
    m_groups: list[tuple] = []
    if eobs is not None and len(eobs.monthly_rows):
        if shares is None or m_fams == ["uf_monthly"]:
            m_groups = [("uf", m_rows, None, monthly_scale, mt_total)]
        elif len(m_fams) == 1:                  # the arcs as the only monthly family
            m_groups = [("cs", m_rows, None, monthly_scale_f[m_fams[0]], mt_total)]
        else:
            for f in m_fams:
                sel = np.flatnonzero(m_fam_np == f)
                m_groups.append(("uf" if f == "uf_monthly" else "cs",
                                 torch.as_tensor(eobs.monthly_rows[sel], device=dev),
                                 torch.as_tensor(sel, device=dev),
                                 monthly_scale_f[f], mt_total_f.get(f)))
    has_cs = any(g[0] == "cs" for g in m_groups)
    # truncated spinup start (clamped to the record start; == 0 restores the
    # exact frozen full-prefix convention)
    spin_req = int(dom.dates.searchsorted(pd.Timestamp(cfg.spinup_start)))
    if eobs is not None and spin_req >= calobs.t0:
        # the default spinup_start (set for the 15cdec calibration window) sits
        # inside this cal window — keep the ten-water-year spinup convention
        # relative to the window
        ts = dom.dates[calobs.t0] - pd.DateOffset(years=10)
        spin_req = int(dom.dates.searchsorted(ts))
        print(f"train: spinup_start {cfg.spinup_start} is inside the cal "
              f"window — spinning up from {ts.date()}", flush=True)
    spin_t0 = min(spin_req, calobs.t0)
    # the CalSim3 arc rows (cs_ entities) must not move the net's input scaling: it is
    # computed on the other entities' rows only, so the base rows see exactly the inputs
    # a run without the arcs would (the arc rows are scaled by the same statistics).
    # None when the run holds no cs_ entity.
    base_hru = None
    if eobs is not None and "calsim_monthly" in eobs.family:
        cs_ids = [b for b, f in zip(dom.basins, eobs.family, strict=True)
                  if f == "calsim_monthly"]
        base_hru = ~dom.hrus["basin"].isin(cs_ids).to_numpy()
        if not base_hru.any():
            raise ValueError("a run of CalSim3 arcs alone has no base rows to scale by")
    feat_kw = dict(variant=variant,
                   physical_path=(soilveg_path(data_dir, domain) if variant == "physical"
                                  else None),
                   aef_path=aef_store(data_dir) if variant == "aef_u" else None)
    stats = None
    if base_hru is not None:
        stats = build_features(dom.hrus[base_hru].reset_index(drop=True), **feat_kw)
        print(f"train: feature z-scoring from the {int(base_hru.sum())} "
              f"base-entity rows only (the {int((~base_hru).sum())} cs_ arc rows are "
              "scaled by them)", flush=True)
    fs = build_features(dom.hrus, stats=stats, **feat_kw)
    x = torch.as_tensor(fs.x).to(dev, dtype)

    # -- the soft logit penalty (opt-in) --
    pen_on = cfg.logit_penalty > 0.0
    if pen_on:
        print(f"train: logit penalty {cfg.logit_penalty:g} x mean (|z| - {LOGIT_CAP:g})^2 "
              "beyond the cap over the head pre-activations, in each live chunk's backward "
              "(not in the logged loss)", flush=True)

    net = ParameterNet(x.shape[1], hidden=cfg.hidden, embed=cfg.embed,
                       dropout=cfg.dropout,
                       canopy=cfg.et_mode == "noah",
                       pxtemp_learn=cfg.pxtemp_learn,
                       pxtemp_box=cfg.pxtemp_box,
                       pxtemp_tau=cfg.pxtemp_tau,
                       log_params=cfg.log_space_params,
                       bounds=cfg.param_bounds,
                       ).to(device=dev, dtype=dtype)
    if cfg.pxtemp_learn:
        print(f"train: LEARNED rain/snow threshold PXTEMP per cell in "
              f"[{cfg.pxtemp_box[0]:g}, {cfg.pxtemp_box[1]:g}] degC — zero-init head "
              f"(exactly the fixed 0 degC at init); hard split forward, "
              f"straight-through sigmoid surrogate (tau {cfg.pxtemp_tau:g} degC)",
              flush=True)
    if cfg.et_mode == "noah":
        if dom.veg_frac is None or dom.lai_lut is None:
            raise ValueError("et_mode='noah' needs observed veg_frac + lai "
                             "(soilveg_continuous.csv + lai_climatology.csv)")
        print(f"train: Noah-lite ET (potential {cfg.noah_pet}; soil_chi learned per cell, "
              "zero-init at its bound midpoint; veg_frac and seasonal LAI observed)",
              flush=True)
    # one starting field on every domain: the 15cdec_grid GA median over its own rows
    print("train: init priors = 15cdec_grid GA median over its own rows", flush=True)
    net.init_from_priors(ga_priors(load_params(domain="15cdec_grid"),
                                   load_hru_table(data_dir, "15cdec_grid")))
    if cfg.log_space_params or cfg.param_bounds:
        # the constructor wrote the mapping into the net's buffers (the priors are placed
        # on it)
        print("train: parameter mapping — "
              + (f"log space also {', '.join(cfg.log_space_params)}"
                 if cfg.log_space_params else "")
              + ("; " if cfg.log_space_params and cfg.param_bounds else "")
              + ", ".join(f"{p} in [{lo:g}, {hi:g}]" for p, (lo, hi) in cfg.param_bounds.items()),
              flush=True)
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
    # the CPU selection (cfg.select_cpu): its scorer process, how often it was restarted,
    # and the last epoch whose score the trainer took
    sel_proc, sel_restarts, sel_seen = None, 0, -1
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
        # the objective comes from the command line, not the checkpoint: a resume that
        # dropped or changed one of these flags would continue silently on a different loss
        ck_cfg = config_from_checkpoint(ck)
        then, now = asdict(ck_cfg), asdict(cfg)
        changed = [f"{k}: {then[k]!r} -> {now[k]!r}" for k in _RESUME_FIXED
                   if then[k] != now[k]]
        if changed:
            raise ValueError("resume: the loss settings differ from the checkpoint's "
                             "(repeat its flags): " + "; ".join(changed))
        if ck_cfg.physics() != cfg.physics():
            raise ValueError("resume: the physics differ from the checkpoint's (repeat its "
                             "flags)")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_kge, stale = ck["epoch"] + 1, ck["best_kge"], ck["stale"]
        sel_seen = ck.get("sel_seen", -1)
        print(f"train: resumed at epoch {start_epoch} (best cal KGE {best_kge:.4f})",
              flush=True)
    if resume and log_path.exists():
        # the resumed run rewrites the epochs from start_epoch on: drop their rows
        d = pd.read_csv(log_path)
        d[d["epoch"] < start_epoch].to_csv(log_path, index=False)

    # -- the water-year chunks and the CUDA-graph capture (falls back to eager) --
    # The chunks are set by the calendar; the recompute graph captures one segment.
    grid = _chunk_grid(dom.dates, calobs.t0, calobs.t1, dom.n_time)
    nograd_g = None
    rec_g = None            # activation-recompute train window (graph_recompute_days)
    if cfg.use_cuda_graphs and dev.type == "cuda":
        from .graphs import NoGradWindow
        try:
            net.eval()
            with torch.no_grad():
                params0 = net(x)
                uh0 = routing_uh(params0, dom.flowlen)
            print("train: capturing CUDA graphs (no-grad window + train chunk) ...",
                  flush=True)
            nograd_g = NoGradWindow(dom, cfg, cfg.nograd_window, params0, uh0)
        except Exception as e:  # noqa: BLE001 — any capture failure -> eager
            print(f"train: no-grad capture failed ({e!r}); running eager",
                  flush=True)
            nograd_g = None
        if nograd_g is not None and cfg.graph_recompute_days > 0:
            # activation recompute: ONE captured graph of graph_recompute_days days
            # replayed over any window, live or dead (graphs.RecomputeTrainWindow) —
            # the graph memory of one segment whatever the window length
            from .graphs import RecomputeTrainWindow
            try:
                rec_g = RecomputeTrainWindow(dom, cfg, cfg.graph_recompute_days,
                                             params0, uh0, sample_c0=calobs.t0)
                print(f"train: activation-recompute train graph ({cfg.graph_recompute_days}"
                      " days, fwd+bwd, replayed per segment; backward re-runs each "
                      "segment from its stored state)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"train: recompute capture failed ({e!r}); eager chunks",
                      flush=True)
                torch.cuda.empty_cache()
        for p in net.parameters():
            p.grad = None
    print("train: two-water-year TBPTT windows — each live chunk runs the previous water "
          "year as a gradient-carrying burn-in, loss on its own year only"
          + (f"; recompute graph of {cfg.graph_recompute_days} days"
             if rec_g is not None else "")
          + "; dead chunks advance forward-only; the first chunk and the envelope tail "
          "run one-year windows", flush=True)

    sel_extra: dict[str, float] = {}     # the last selection's sel3 (runs with arcs)

    def _mt_kge(sim: torch.Tensor) -> float:
        """The selection scalar of per-entity cal KGE at each entity's NATIVE timescale:
        daily rows at daily stride, monthly rows on calendar-month sums of the same
        simulated flow (>= 12 obs months to enter the pool).  Prints the
        per-family means; the selection scalar is the pooled mean, or the
        share-weighted family mean of a run with family shares."""
        scalar, fam_means, sel3, pooled = selection_stat(
            sim, calobs.obs, eobs=eobs, midx=midx, shares=shares, sel_shares=sel_shares,
            sel3_shares=sel3_shares)
        parts = "  ".join(f"{f.split('_')[0]} {m:.4f}" for f, m in fam_means.items())
        # with the CalSim3 arcs in the scalar, sel3 is logged beside it: the scalar a run
        # without arcs selects on (share runs: the other families' footprint-area shares)
        sel3_txt = ""
        if sel3 is not None:
            sel_extra["sel3"] = sel3
            sel3_txt = f" sel3 {sel3:.4f}"
        if shares is not None:
            # the share-weighted family mean over the families with a valid entity (at
            # mt_select_weight's shares if set)
            print(f"    per-family cal KGE: {parts}  | sel family-weighted "
                  f"{scalar:.4f}{sel3_txt} (pooled {pooled:.4f})", flush=True)
        else:
            if sel3_txt:
                parts += f"  |{sel3_txt} (pooled without the arcs)"
            print(f"    per-family cal KGE: {parts}", flush=True)
        return scalar

    def _spinup_and_maybe_eval(do_eval: bool) -> tuple[float, PipelineState]:
        """Fresh spinup under the current net; optionally the selection KGE (the
        selection scalar)."""
        net.eval()
        with torch.no_grad():
            pe = net(x)
            ue = routing_uh(pe, dom.flowlen)
        if nograd_g is not None:
            nograd_g.set_params(pe, ue)
        st0 = initial_state(dom.n_hru, dev, dtype)
        _, st = _stream_nograd(dom, cfg, pe, ue, spin_t0, calobs.t0, st0, graph=nograd_g)
        pooled = float("nan")
        if do_eval:
            sim, _ = _stream_nograd(dom, cfg, pe, ue, calobs.t0, calobs.t1, st,
                                    graph=nograd_g, collect=True)
            pooled = _mt_kge(sim) if eobs is not None else _cal_kge(sim, calobs.obs)
            if cfg.diagnostics:
                eval_sim["sim"] = sim
        return pooled, st

    eval_sim: dict[str, torch.Tensor] = {}   # the last selection pass's flow (diagnostics)

    def _save(path: Path, *, epoch: int, kge: float) -> None:
        # written, then renamed: a run stopped mid-write keeps the previous file
        tmp = path.with_suffix(".tmp")
        torch.save({**_payload(epoch=epoch, kge=kge), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, tmp)
        replace_retry(tmp, path)

    def _payload(*, epoch: int, kge: float) -> dict:
        # runs with the arcs: the selection's sel3 beside cal_kge (NaN with no
        # selection this epoch, like cal_kge)
        extra = ({"sel3": sel_extra["sel3"] if math.isfinite(kge) else float("nan")}
                 if sel_extra else {})
        # the resolved shares the selection scalar / sel3 were computed with
        if cfg.mt_select_weight and sel_shares is not None:
            extra["mt_select_shares"] = dict(sel_shares)
        if sel3_shares:
            extra["sel3_shares"] = dict(sel3_shares)
        if cfg.select_cpu:
            extra["sel_seen"] = sel_seen
        return {"net": net.state_dict(), "epoch": epoch, **extra,
                "best_kge": best_kge, "stale": stale, "cal_kge": kge,
                "mt_select": (("family_weighted" if shares is not None else "pooled")
                              if eobs is not None else None),
                "cfg": asdict(cfg), "variant": variant, "domain": domain,
                "basins": list(dom.basins),
                "net_config": {"hidden": cfg.hidden, "embed": cfg.embed,
                               "dropout": cfg.dropout,
                               "canopy": cfg.et_mode == "noah",
                               "pxtemp_learn": cfg.pxtemp_learn,
                               "pxtemp_box": cfg.pxtemp_box,
                               "pxtemp_tau": cfg.pxtemp_tau,
                               "log_params": cfg.log_space_params,
                               "bounds": cfg.param_bounds},
                "et_mode": cfg.et_mode,
                "features": _feature_stats(fs)}

    n_chunks = len(grid)

    # per-chunk MONTHLY-FLOW targets (multi-timescale domain): fixed given the
    # chunks — the day->month bucket, each entity's observed months gathered to
    # the chunk's slots, and the finite-slot mask.  Simulated flow multiplies the
    # bucket per chunk.  A month the chunk boundary cuts would get no slot; the
    # water-year chunks cut none.
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
    # a chunk carrying NO scoreable observation (a held-out or unobserved water year;
    # the multi-timescale envelope's tail, Oct-Dec 2018, where every daily entity may
    # fail min_days and no monthly entity reaches Dec 2018) runs its forward without
    # autograd and steps nothing: zero grads would still move parameters through AdamW
    # momentum + weight decay.  The train-mode net(x) is still drawn once per chunk
    # (same dropout stream) and the same graphed forward runs, so the carried state and
    # every later chunk are unchanged.  The 15cdec domains never produce a dead chunk.
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
                  "scoreable obs — forward-only, without autograd (no optimizer step)",
                  flush=True)
            if cfg.holdout_wy:
                print("train: dead chunks (water years): " + ", ".join(
                    str(dom.dates[c0].year + 1) for (c0, _), lv in zip(grid, chunk_live,
                                                                       strict=True)
                    if not lv), flush=True)
    else:
        chunk_live = [True] * n_chunks

    # -- diagnostics (cfg.diagnostics; numerics-neutral) ---------------------
    diag = cfg.diagnostics
    chunk_cols = _CHUNK_LOG_COLS + (_CS_CHUNK_LOG_COLS if has_cs else ())
    eval_cols = _EVAL_TERM_COLS + (_CS_EVAL_TERM_COLS if has_cs else ())
    chunk_log_path, eval_terms_path = run.local / "chunk_log.csv", run.local / "eval_terms.csv"
    snapdir = run.local / "checkpoints" / "snapshots"
    params_list = list(net.parameters())
    ema: list[torch.Tensor] = []
    if not resume:
        # a fresh run keeps no diagnostics of an earlier run in this folder
        for p in (chunk_log_path, eval_terms_path, *snapdir.glob("e*.pt")):
            p.unlink(missing_ok=True)
    if diag:
        snapdir.mkdir(parents=True, exist_ok=True)
        if resume:
            # the resumed run rewrites the interrupted epoch: drop its rows; the kept
            # rows take the current columns (a file from before a column was added
            # would otherwise keep a header the appended rows no longer match)
            for p, cols in ((chunk_log_path, chunk_cols), (eval_terms_path, eval_cols)):
                if p.exists():
                    d = pd.read_csv(p)
                    d[d["epoch"] < start_epoch].reindex(columns=cols).to_csv(p, index=False)
        # the EMA shadow restarts from the current net (also on a resume)
        ema = [p.detach().clone() for p in params_list]
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

    # the selection on the CPU (cfg.select_cpu): a candidate per epoch, scored by
    # select_cpu.main in its own process; its rows come back here in epoch order
    seldir = run.local / "checkpoints" / "select"

    def _candidate(epoch: int) -> None:
        """This epoch's selection candidate for the CPU scorer (written, then renamed)."""
        tmp = seldir / f"e{epoch:03d}.tmp"
        torch.save({**_payload(epoch=epoch, kge=float("nan")), "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "sel_setup": {"shares": shares, "sel_shares": sel_shares,
                                  "sel3_shares": sel3_shares}}, tmp)
        tmp.replace(seldir / f"e{epoch:03d}.pt")

    def _restart_scorer() -> None:
        """A scorer that ended before its work did starts again (three times in a row at
        most)."""
        nonlocal sel_proc, sel_restarts
        sel_restarts += 1
        if sel_restarts > 3:
            raise RuntimeError(f"the CPU selection scorer ended {sel_restarts} times (exit "
                               f"{sel_proc.returncode}; see {run.local / 'select_cpu.log'})")
        print(f"train: the CPU selection scorer ended (exit {sel_proc.returncode}); "
              "restarted", flush=True)
        sel_proc = start_scorer(run.local, os.path.abspath(data_dir))

    def _take_scores(final: bool = False) -> bool:
        """The scorer's new rows: a better candidate becomes best.pt, any other (a NaN
        score too) counts as a stale epoch; True once more than ``patience`` epochs are
        stale; the rows after the halting one are not taken (the GPU selection stops
        there).  The candidates stay until last.pt records ``sel_seen`` (a resume takes
        their rows again)."""
        nonlocal best_kge, stale, sel_seen, sel_restarts
        if not final and sel_proc.poll() is not None:
            _restart_scorer()
        halt = False
        for e, v, s3 in read_scores(run.local, sel_seen):
            if halt:
                break
            sel_restarts = 0                     # three restarts in a row without a row end it
            if not math.isfinite(v) and not math.isfinite(best_kge):
                raise RuntimeError(f"the CPU selection scored epoch {e} NaN before any finite "
                                   f"score (see {run.local / 'select_cpu.log'})")
            sel_seen = e
            c = seldir / f"e{e:03d}.pt"
            if v > best_kge:
                best_kge, stale = v, 0
                ck = torch.load(c, map_location="cpu", weights_only=False)
                ck.pop("sel_setup")
                ck.update(cal_kge=v, best_kge=v, stale=0)
                if math.isfinite(s3):
                    ck["sel3"] = s3
                torch.save(ck, ckdir / "best.tmp")
                replace_retry(ckdir / "best.tmp", ckdir / "best.pt")
            else:
                stale += 1
            halt = halt or stale > cfg.patience
        return halt

    def _mt_terms(basin_flow: torch.Tensor, k: int) -> list[tuple]:
        """Chunk ``k``'s monthly terms, one per ``m_groups`` entry: (tag, simulated
        complete-month sums, targets, finite mask, variances, scale, denominator) —
        a single group is the whole monthly block unsliced (the pooled term)."""
        mb, mtgt, mfin = mt_targets[k]
        out_t = []
        for tag, rows_g, sel, scale, n_tot in m_groups:
            if sel is None:
                out_t.append((tag, basin_flow[rows_g] @ mb, mtgt, mfin, eobs.var_monthly,
                              scale, n_tot))
            else:
                out_t.append((tag, basin_flow[rows_g] @ mb, mtgt[sel], mfin[sel],
                              eobs.var_monthly[sel], scale, n_tot))
        return out_t

    def _loss_split(basin_flow: torch.Tensor, obs_c: torch.Tensor, k: int, *,
                    terms: bool = False) -> dict[str, float]:
        """The multi-timescale chunk loss as trained, split into additive
        contributions by family — ``l_<fam>`` (or, ``terms``, ``<fam>_nnse`` /
        ``_log`` / ``_var`` and ``uf_monthly`` / ``cs_monthly``) — with each family's
        live entity count ``n_<fam>`` and coefficient mass ``c_<fam>`` (its share of the chunk's
        daily-term weight times daily_scale; its monthly term's scale for uf and
        the CalSim3 arcs, tag ``cs``)."""
        res: dict[str, float] = {}
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

                def _l(log_l: float, var_l: float) -> float:
                    return float(masked_basin_loss(
                        basin_flow, obs_c, calobs.obs_var,
                        log_lambda=log_l, log_eps=cfg.log_loss_eps,
                        var_lambda=var_l, var_huber_cap=cfg.var_huber_cap,
                        shape_min_days=cfg.shape_min_days, weight=wf))

                full = _l(cfg.log_loss_lambda, cfg.var_loss_lambda)
                if terms:
                    nnse = _l(0.0, 0.0)
                    nl = _l(cfg.log_loss_lambda, 0.0)
                    res[f"{tag}_nnse"] = sc * nnse
                    res[f"{tag}_log"] = sc * (nl - nnse)
                    res[f"{tag}_var"] = sc * (full - nl)
                else:
                    res[f"l_{tag}"] = sc * full
            if mt_targets is not None:
                for tag, sim_m, mtgt, mfin, var_m, scale, n_tot in _mt_terms(basin_flow, k):
                    n_m = int((mfin.sum(dim=1) >= 1).sum())
                    res[f"n_{tag}"] = n_m
                    res[f"c_{tag}"] = (scale * n_m / n_tot if n_tot is not None
                                       else scale if n_m else 0.0)
                    res[f"{tag}_monthly" if terms else f"l_{tag}"] = scale * float(
                        monthly_nnse_loss(sim_m, mtgt, mfin, var_m, n_total=n_tot))
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
        pd.DataFrame([row]).reindex(columns=eval_cols).to_csv(
            eval_terms_path, mode="a", index=False,
            header=not eval_terms_path.exists())

    print(f"train[{variant}]: {dom.n_hru} HRUs, {len(dom.basins)} basins, "
          f"{x.shape[1]} features | cal days {calobs.t0}..{calobs.t1} "
          f"(spinup from day {spin_t0}"
          f"{' = record start' if spin_t0 == 0 else f' = {dom.dates[spin_t0].date()}'}) "
          f"(water years: {_grid_summary(grid)}) | nnse loss "
          f"(log lambda {cfg.log_loss_lambda}) | n_inc={cfg.n_inc} "
          f"{cfg.dtype} on {dev.type}"
          + (f' [cuda-graphs recompute {cfg.graph_recompute_days} days]' if rec_g is not None
             else ' [eager]'),
          flush=True)

    if cfg.select_cpu:
        seldir.mkdir(parents=True, exist_ok=True)
        sel_csv = run.local / "select_cpu.csv"
        if resume:
            if (seldir / "done").exists() and not any(seldir.glob("e*.pt")):
                raise ValueError("resume: this run has finished (its selection is done)")
            # the epochs from start_epoch on run again: their candidates and rows go; the
            # earlier ones not yet taken (after sel_seen) stay, scored or not
            for p in (*seldir.glob("e*.tmp"), seldir / "done",
                      *(c for c in seldir.glob("e*.pt") if int(c.stem[1:]) >= start_epoch)):
                p.unlink(missing_ok=True)
            if sel_csv.exists():
                d = pd.read_csv(sel_csv)
                d[d["epoch"] < start_epoch].to_csv(sel_csv, index=False)
        else:
            for p in (*seldir.iterdir(), sel_csv, run.local / "select_cpu.log"):
                p.unlink(missing_ok=True)
        sel_proc = start_scorer(run.local, os.path.abspath(data_dir))
        atexit.register(lambda: sel_proc.poll() is None and sel_proc.kill())
        print(f"train: selection every epoch on the CPU engine (pid {sel_proc.pid}) -> "
              f"{run.local / 'select_cpu.csv'}; patience {cfg.patience} stale epochs",
              flush=True)

    stop = False
    for epoch in range(start_epoch, cfg.n_epochs):
        tic = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

        # with the CPU selection the GPU does not evaluate
        do_eval = not cfg.select_cpu and epoch % cfg.eval_every == 0
        pooled, state_spin = _spinup_and_maybe_eval(do_eval)
        spin_s = time.time() - tic

        is_best = False
        if cfg.select_cpu:
            _candidate(epoch)
            if _take_scores() and not stop:
                print(f"train: early stop at epoch {epoch} ({stale} stale epochs, scored "
                      f"to epoch {sel_seen})", flush=True)
                stop = True
        elif do_eval:
            if pooled > best_kge:
                best_kge, stale, is_best = pooled, 0, True
                _save(ckdir / "best.pt", epoch=epoch, kge=pooled)
            else:
                stale += 1
            if stale > cfg.patience:
                print(f"train: early stop at epoch {epoch} "
                      f"({stale} stale selections)", flush=True)
                stop = True
        if diag:
            _snapshot(epoch, pooled)
            if do_eval and eobs is not None:
                _eval_terms(epoch)

        # -- TBPTT chunks (one optimizer step per chunk) --------------------
        losses: list[float] = []
        pen_losses: list[float] = []
        chunk_rows: list[dict] = []
        skipped = 0
        if not stop:
            net.train()
            # the carried state lives in the Python variable between chunks
            if dev.type == "cuda":
                # hand the cached blocks of the spinup / selection forward back to
                # the driver before the training chunks (no numerical effect): on
                # an 8 GB WDDM card they would otherwise sit beside the train graph
                # and push the process into paging
                torch.cuda.empty_cache()
            state = state_spin
            from .graphs import _clone_state
            prev = None     # (the state carried into the previous chunk, its c0)
            for k, (c0, ce) in enumerate(grid):
                # the next chunk's window starts from this chunk's state: it must own
                # its storage (a graphed window's end state is a view of the graph's
                # static output buffers, which the next replay overwrites in place)
                burn, prev = prev, (_clone_state(state), c0)
                dead_nograd = not chunk_live[k]
                # a live full-year chunk that follows a full-year chunk runs the window
                # [that chunk's c0, ce) from the state carried into it, the loss on
                # [c0, ce) only
                two = (burn is not None and not dead_nograd
                       and ce - c0 >= 365 and c0 - burn[1] >= 365)
                w0 = burn[1] if two else c0
                obs_c = _obs_chunk(calobs, c0, ce)
                split: dict[str, float] = {}
                if not dead_nograd:
                    opt.zero_grad(set_to_none=True)
                # a dead chunk: the same train-mode net(x) (one dropout draw) and the
                # same graphed forward, without autograd
                with torch.no_grad() if dead_nograd else contextlib.nullcontext():
                    if pen_on and not dead_nograd:
                        # the same single forward (one dropout draw), pre-activations kept
                        params, pre = net(x, logits=True)
                    else:
                        params = net(x)
                    uh = routing_uh(params, dom.flowlen)
                    if two:
                        state = burn[0]         # the burn-in's carried state
                    if not dead_nograd:
                        state = _relative_carry(state, params)
                    if rec_g is not None:
                        # activation recompute over the whole window (under
                        # no_grad for a dead chunk: a plain graphed forward)
                        flow, state = rec_g.forward(w0, ce - w0, params, uh, state)
                    elif dead_nograd and nograd_g is not None:
                        # a dead chunk replays the forward-only stream window
                        nograd_g.set_params(params, uh)
                        _, state = _stream_nograd(dom, cfg, params, uh, c0, ce, state,
                                                  graph=nograd_g)
                        flow = None
                    else:
                        # no train graph: the whole window eagerly
                        flow, state = run_window(dom.window(w0, ce), dom.lat_rad, dom.elev,
                                                 params, uh, state, physics,
                                                 veg_frac=dom.veg_frac)
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
                    basin_flow, obs_c, calobs.obs_var,
                    log_lambda=cfg.log_loss_lambda, log_eps=cfg.log_loss_eps,
                    var_lambda=cfg.var_loss_lambda, var_huber_cap=cfg.var_huber_cap,
                    shape_min_days=cfg.shape_min_days,
                    weight=fam_w, weight_total=w_total)
                if mt_targets is not None:
                    for _, sim_m, mtgt, mfin, var_m, scale, n_tot in _mt_terms(
                            basin_flow, k):
                        loss_t = loss_t + scale * monthly_nnse_loss(
                            sim_m, mtgt, mfin, var_m, n_total=n_tot)
                if pen_on:
                    # the soft logit penalty: in the backward, not in the logged loss
                    pen_t = cfg.logit_penalty * logit_penalty(pre)
                    (loss_t + pen_t).backward()
                    pen_losses.append(float(pen_t.detach()))
                else:
                    loss_t.backward()
                loss, state = float(loss_t.detach()), state.detach()
                if diag and eobs is not None:
                    split = _loss_split(basin_flow, obs_c, k)
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                      cfg.grad_clip)
                stepped = False
                if math.isfinite(loss) and bool(torch.isfinite(norm)):
                    opt.step()
                    stepped = True
                    if diag:
                        with torch.no_grad():
                            for e, p in zip(ema, params_list, strict=True):
                                e.lerp_(p, 1.0 - cfg.ema_decay)
                else:
                    skipped += 1
                opt.zero_grad(set_to_none=False)
                losses.append(loss)
                if diag:
                    # gnorm = the pre-clip total gradient norm
                    chunk_rows.append({"epoch": epoch, "k": k,
                                       "start": str(dom.dates[c0].date()),
                                       "days": ce - c0, "live": True,
                                       "nograd": False, "stepped": stepped,
                                       "loss": loss, "gnorm": float(norm), **split})
            sched.step()
        if chunk_rows:
            pd.DataFrame(chunk_rows).reindex(columns=chunk_cols).to_csv(
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
        if base_hru is not None:        # a run with the arcs: the column every epoch
            row["sel3"] = sel_extra.get("sel3", float("nan")) if do_eval else float("nan")
        if pen_on:
            row["logit_pen"] = (sum(pen_losses) / len(pen_losses)
                                if pen_losses else float("nan"))
        pd.DataFrame([row]).to_csv(log_path, mode="a", index=False,
                                   header=not log_path.exists())
        if not stop:
            _save(ckdir / "last.pt", epoch=epoch, kge=pooled)
            for c in (seldir.glob("e*.pt") if cfg.select_cpu else ()):
                if int(c.stem[1:]) <= sel_seen:
                    c.unlink()
        print(f"  epoch {epoch:3d}/{cfg.n_epochs}  loss {row['loss']:.4f}  "
              + ("" if cfg.select_cpu else f"calKGE {pooled:.4f}{'*' if is_best else ' '}  ")
              + f"lr {row['lr']:.1e}  spin {spin_s:.0f}s  "
              f"epoch {row['epoch_s']:.0f}s  vram {row['peak_vram_mb']}MB"
              + (f"  logit pen {row['logit_pen']:.2e}" if pen_on else "")
              + (f"  cpu-sel to e{sel_seen}: best {best_kge:.4f}" if cfg.select_cpu else "")
              + (f"  skipped {skipped}" if skipped else ""), flush=True)
        if stop:
            break

    # final selection on the post-training net (loop evals are pre-update).  An
    # early stop comes before the stop epoch's training, so that epoch's
    # selection (a stale score, in ``pooled``) already scored the final net.
    if not stop and cfg.select_cpu:
        _candidate(cfg.n_epochs)
        if diag:
            _snapshot(cfg.n_epochs, float("nan"))
    elif not stop:
        pooled, _ = _spinup_and_maybe_eval(True)
        if diag:
            _snapshot(cfg.n_epochs, pooled)
            if eobs is not None:
                _eval_terms(cfg.n_epochs)
        if pooled > best_kge:
            best_kge = pooled
            _save(ckdir / "best.pt", epoch=cfg.n_epochs, kge=pooled)
    if cfg.select_cpu:
        # every candidate scored, then the scorer ends
        (seldir / "done").touch()
        while sel_proc.wait() != 0:
            _restart_scorer()
        if not stop:
            _take_scores(final=True)
        for c in seldir.glob("e*.pt"):
            c.unlink()
        pooled = read_scores(run.local, sel_seen - 1)[-1][1]
        # the train log's cal_kge (and sel3): the CPU selection's scores of each epoch's
        # candidate
        sc = pd.read_csv(run.local / "select_cpu.csv").set_index("epoch")
        tl = pd.read_csv(log_path)
        tl["cal_kge"] = tl["epoch"].map(sc["score"])
        if "sel3" in tl:
            tl["sel3"] = tl["epoch"].map(sc["sel3"])
        tl.to_csv(log_path, index=False)
    print(f"train[{variant}]: done — best selection cal KGE {best_kge:.4f} "
          f"(final {pooled:.4f}) -> {ckdir / 'best.pt'}", flush=True)
    return out
