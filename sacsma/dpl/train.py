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
   days of the last chunk are NaN-masked).

On CUDA the day-stepped pipeline runs as captured CUDA graphs
(:mod:`sacsma.dpl.graphs`) — eager execution is dispatch-bound.  Graph
capture failure (or ``--device cpu``) falls back to eager with identical
numerics.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..io import load_params, soilveg_path
from .config import (
    CANOPY_LEARNED_PARAMS,
    CANOPY_LITE_LEARNED,
    DplConfig,
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
)
from .features import FeatureSet, build_features
from .forward import PipelineState, initial_state, routing_uh, run_window
from .loss import kge_torch, level_hinge_loss, masked_basin_loss, shape_pull_loss
from .parameter_net import ParameterNet, ga_priors
from .regularize import (
    adaptive_basin_weights,
    build_neighbor_edges,
    param_norm_scales,
    spatial_smoothness,
)

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


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
            flow, state = run_window(pr, ta, doy, leap, dom.lat_rad, dom.elev,
                                     params, uh, state, n_inc=cfg.n_inc,
                                     perc_mode=cfg.perc_mode,
                                     fracp_floor=cfg.fracp_floor,
                                     ninc_mode="fixed", et_mode=cfg.et_mode,
                                     canopy_params=canopy_params, tmin=tn, tmax=tx,
                                     veg_frac=dom.veg_frac,
                                     lai=dom.chunk_lai(t, te),
                                     noah_pet=cfg.noah_pet, sac_pet=cfg.sac_pet,
                                     pt_snow_albedo=cfg.pt_snow_albedo,
                                     pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                                     canopy_lite=cfg.canopy_lite,
                                     state_idx=dom.chunk_state(t, te))
            if collect:
                outs.append(dom.W @ flow)
            t = te
    return (torch.cat(outs, dim=1) if collect else None), state


def _obs_chunk(calobs: CalObs, c0: int, c1: int) -> torch.Tensor:
    """Cal-window obs for record days [c0, c1); NaN outside the window."""
    out = torch.full((calobs.obs.shape[0], c1 - c0), float("nan"),
                     device=calobs.obs.device, dtype=calobs.obs.dtype)
    lo, hi = max(c0, calobs.t0), min(c1, calobs.t1)
    if hi > lo:
        out[:, lo - c0:hi - c0] = calobs.obs[:, lo - calobs.t0:hi - calobs.t0]
    return out


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


def train(
    variant: str = "static",
    data_dir: str = "data",
    out_dir: str | Path | None = None,
    cfg: DplConfig | None = None,
    *,
    resume: bool = False,
    basins: tuple[str, ...] | None = None,   # debug subset (default: all 15)
    domain: str = "15cdec",                  # 15cdec HRU cloud | 15cdec_grid native grid
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
    calobs = load_cal_obs(dom, data_dir, cal_start=cfg.cal_start)
    # truncated spinup start (clamped to the record start; == 0 restores the
    # exact frozen full-prefix convention)
    spin_t0 = min(int(dom.dates.searchsorted(pd.Timestamp(cfg.spinup_start))),
                  calobs.t0)
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
    fs = build_features(
        dom.hrus, variant=variant,
        forcing=dom.forcing if variant == "climate" else None,
        climate_product="historical_livneh_unsplit" if variant == "climate" else None,
        fourier_k=cfg.fourier_k,
        physical_path=(soilveg_path(data_dir, domain)
                       if variant == "physical" else None),
    )
    x = torch.as_tensor(fs.x).to(dev, dtype)

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
                       ).to(device=dev, dtype=dtype)
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
    priors = ga_priors(load_params(data_dir, domain=domain), dom.hrus)
    net.init_from_priors(priors)
    donor_kge = float("nan")
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
        # strict=False: heads the donor lacks (e.g. a fresh seasonal head)
        # keep their zero-init, so training starts EXACTLY at the donor's
        # parameter field; donor keys the net lacks are a config error.
        missing, unexpected = net.load_state_dict(ick["net"], strict=False)
        if unexpected:
            raise ValueError(f"init_from checkpoint carries heads this net "
                             f"lacks: {sorted(unexpected)}")
        print(f"train: warm-start from {cfg.init_from} (epoch {ick['epoch']}, "
              f"sel cal KGE {ick.get('cal_kge', float('nan')):.4f}); fresh "
              f"zero-init heads: {sorted(missing) if missing else 'none'}; "
              "fresh optimizer/scheduler", flush=True)
        donor_kge = float(ick.get("cal_kge", float("nan")))
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
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_kge, stale = ck["epoch"] + 1, ck["best_kge"], ck["stale"]
        print(f"train: resumed at epoch {start_epoch} (best cal KGE {best_kge:.4f})",
              flush=True)

    # SWE snow-basin participation weights (shared by the graph and eager paths)
    swe_w = (torch.as_tensor(sweobs.basin_w, device=dev, dtype=dtype)
             if sweobs is not None else None)

    # -- CUDA-graph capture (falls back to eager) ---------------------------
    nograd_g = train_g = None
    if cfg.use_cuda_graphs and dev.type == "cuda":
        from .graphs import NoGradWindow, TrainChunk
        try:
            net.eval()
            with torch.no_grad():
                params0, canopy0 = _split_out(net(x), cfg.et_mode)
                uh0 = routing_uh(params0, dom.flowlen)
            print("train: capturing CUDA graphs (no-grad window + train chunk) ...",
                  flush=True)
            nograd_g = NoGradWindow(dom, cfg, cfg.nograd_window, params0, uh0,
                                    canopy_params=canopy0)
        except Exception as e:  # noqa: BLE001 — any capture failure -> eager
            print(f"train: no-grad capture failed ({e!r}); running eager",
                  flush=True)
            nograd_g = None
        if nograd_g is not None:
            # whole-chunk fwd+bwd capture is the VRAM peak: on OOM, halve the
            # chunk once (TBPTT detaches mid-water-year) before giving up
            for clen in (cfg.train_chunk_days, cfg.train_chunk_days // 2):
                try:
                    train_g = TrainChunk(net, dom, cfg, clen, x, calobs.obs_var,
                                         weight=basin_w, swe_basin_w=swe_w)
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"train: chunk capture OOM at {clen} days — "
                          "halving", flush=True)
                    torch.cuda.empty_cache()
                except Exception as e:  # noqa: BLE001
                    print(f"train: chunk capture failed ({e!r}); "
                          "eager chunks", flush=True)
                    break
        if train_g is None:
            for p in net.parameters():
                p.grad = None

    def _spinup_and_maybe_eval(
        do_eval: bool,
    ) -> tuple[float, torch.Tensor | None, PipelineState]:
        """Fresh spinup under the current net; optionally the selection KGE
        (pooled scalar + the per-basin vector for adaptive weighting)."""
        net.eval()
        with torch.no_grad():
            pe, cp_e = _split_out(net(x), cfg.et_mode)
            ue = routing_uh(pe, dom.flowlen)
        if nograd_g is not None:
            nograd_g.set_params(pe, ue, canopy_params=cp_e)
        st0 = initial_state(dom.n_hru, dev, dtype, init_mode=cfg.init_mode,
                            params=pe, et_mode=cfg.et_mode)
        _, st = _stream_nograd(dom, cfg, pe, ue, spin_t0, calobs.t0, st0,
                               graph=nograd_g, canopy_params=cp_e)
        pooled, per_basin = float("nan"), None
        if do_eval:
            sim, _ = _stream_nograd(dom, cfg, pe, ue, calobs.t0, calobs.t1, st,
                                    graph=nograd_g, collect=True, canopy_params=cp_e)
            per_basin, pooled = _cal_kge(sim, calobs.obs)
        return pooled, per_basin, st

    def _save(path: Path, *, epoch: int, kge: float) -> None:
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch,
                    "best_kge": best_kge, "stale": stale, "cal_kge": kge,
                    "cfg": asdict(cfg), "variant": variant, "domain": domain,
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
                                   "dynamic_window": cfg.dynamic_window},
                    "et_mode": cfg.et_mode,
                    "features": _feature_stats(fs)}, path)

    chunk = train_g.length if train_g is not None else cfg.train_chunk_days
    n_chunks = math.ceil((calobs.t1 - calobs.t0) / chunk)

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
        for k in range(n_chunks):
            c0 = calobs.t0 + k * chunk
            bucket, cmon0, mask = et_chunk_target(
                dom.dates, c0, chunk, calobs.t0, calobs.t1)
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
    print(f"train[{variant}]: {dom.n_hru} HRUs, {len(dom.basins)} basins, "
          f"{x.shape[1]} features | cal days {calobs.t0}..{calobs.t1} "
          f"(spinup from day {spin_t0}"
          f"{' = record start' if spin_t0 == 0 else f' = {cfg.spinup_start}'}) "
          f"({n_chunks} x {chunk}-day chunks) | {cfg.loss} loss "
          f"(log lambda {cfg.log_loss_lambda}) | n_inc={cfg.n_inc} "
          f"perc={cfg.perc_mode} {cfg.dtype} on {dev.type}"
          f"{' [cuda-graphs]' if train_g is not None else ' [eager]'}",
          flush=True)

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
        # donor's sel cal-KGE.  A gap means a NUMERICS/CONFIG mismatch (n_inc,
        # spinup basis, domain/footprint flags) — kill the run and fix flags.
        if (cfg.init_from and epoch == start_epoch and do_eval
                and math.isfinite(donor_kge) and abs(pooled - donor_kge) > 1e-3):
            print(f"train: WARNING — ep0 cal KGE {pooled:.4f} != donor "
                  f"{donor_kge:.4f} (|d|={abs(pooled - donor_kge):.4f} > 1e-3): "
                  "this run's numerics/config do NOT reproduce the donor; "
                  "check --n-inc / spinup / footprint before trusting the "
                  "fine-tune", flush=True)

        is_best = False
        if do_eval:
            if pooled > best_kge:
                best_kge, stale, is_best = pooled, 0, True
                _save(ckdir / "best.pt", epoch=epoch, kge=pooled)
            else:
                stale += 1
            if stale > cfg.patience:
                print(f"train: early stop at epoch {epoch} "
                      f"({stale} stale selections)", flush=True)
                stop = True

        # -- TBPTT chunks (one optimizer step per chunk) --------------------
        losses: list[float] = []
        reg_losses: list[float] = []
        skipped = 0
        if not stop:
            assert state_spin is not None   # first epoch always spins up
            net.train()
            state = state_spin
            if train_g is not None:
                train_g.set_state(state_spin)
            for k in range(n_chunks):
                c0 = calobs.t0 + k * chunk
                pr, ta, doy, leap = dom.chunk(c0, c0 + chunk)
                tn, tx = dom.chunk_tmm(c0, c0 + chunk)
                lai_c = dom.chunk_lai(c0, c0 + chunk)
                st_c = dom.chunk_state(c0, c0 + chunk)
                obs_c = _obs_chunk(calobs, c0, c0 + chunk)
                et_tgt = et_targets[k] if et_targets is not None else None
                swe_tgt = swe_targets[k] if swe_targets is not None else None
                if train_g is not None:
                    loss = train_g.run(pr, ta, doy, leap, obs_c, tn, tx, lai_c,
                                       st_c, et_target=et_tgt, swe_target=swe_tgt)
                else:
                    opt.zero_grad(set_to_none=True)
                    params, cp = _split_out(net(x), cfg.et_mode)
                    uh = routing_uh(params, dom.flowlen)
                    res = run_window(
                        pr, ta, doy, leap, dom.lat_rad, dom.elev, params, uh,
                        state, n_inc=cfg.n_inc, perc_mode=cfg.perc_mode,
                        fracp_floor=cfg.fracp_floor, ninc_mode="fixed",
                        et_mode=cfg.et_mode, canopy_params=cp, tmin=tn, tmax=tx,
                        veg_frac=dom.veg_frac, lai=lai_c, noah_pet=cfg.noah_pet,
                        sac_pet=cfg.sac_pet, pt_snow_albedo=cfg.pt_snow_albedo,
                        pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                        canopy_lite=cfg.canopy_lite,
                        state_idx=st_c, return_tet=et_tgt is not None,
                        return_swe=swe_tgt is not None)
                    flow, state = res[0], res[1]
                    loss_t = masked_basin_loss(
                        dom.W @ flow, obs_c, calobs.obs_var, kind=cfg.loss,
                        log_lambda=cfg.log_loss_lambda, log_eps=cfg.log_loss_eps,
                        var_lambda=cfg.var_loss_lambda,
                        bias_lambda=cfg.bias_loss_lambda, weight=basin_w)
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
                if math.isfinite(loss) and bool(torch.isfinite(norm)):
                    opt.step()
                else:
                    skipped += 1
                opt.zero_grad(set_to_none=False)
                losses.append(loss)
            sched.step()

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
        pd.DataFrame([row]).to_csv(log_path, mode="a", index=False,
                                   header=not log_path.exists())
        if not stop:
            _save(ckdir / "last.pt", epoch=epoch, kge=pooled)
        print(f"  epoch {epoch:3d}/{cfg.n_epochs}  loss {row['loss']:.4f}  "
              f"calKGE {pooled:.4f}{'*' if is_best else ' '}  "
              f"lr {row['lr']:.1e}  spin {spin_s:.0f}s  "
              f"epoch {row['epoch_s']:.0f}s  vram {row['peak_vram_mb']}MB"
              + (f"  reg {row['reg']:.4f}" if reg_lambda > 0.0 else "")
              + (f"  skipped {skipped}" if skipped else ""), flush=True)
        if stop:
            break

    # final selection on the post-training net (loop evals are pre-update)
    pooled, _, _ = _spinup_and_maybe_eval(True)
    if pooled > best_kge:
        best_kge = pooled
        _save(ckdir / "best.pt", epoch=cfg.n_epochs, kge=pooled)
    print(f"train[{variant}]: done — best pooled cal KGE {best_kge:.4f} "
          f"(final {pooled:.4f}) -> {ckdir / 'best.pt'}", flush=True)
    return out
