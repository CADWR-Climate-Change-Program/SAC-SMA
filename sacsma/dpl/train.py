"""dPL training: full-prefix spinup + water-year TBPTT + no-grad KGE selection.

Protocol (per epoch):

1. **Spinup** — no-grad forward from the reference cold start at the record
   start (1915) to the day before the calibration window, under the CURRENT
   parameters, so the WY1989 initial state is exactly what the frozen model
   would carry (LZ stores hold multi-year memory).  ``spinup_refresh_every``
   > 1 reuses a cached state as an explicit approximation.
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

import pandas as pd
import torch

from ..io import load_params, soilveg_path
from .config import (
    CANOPY_LEARNED_PARAMS,
    CANOPY_LITE_LEARNED,
    DplConfig,
    pick_device,
)
from .data import CalObs, DomainTensors, load_cal_obs, load_domain_tensors
from .features import FeatureSet, build_features
from .forward import PipelineState, initial_state, routing_uh, run_window
from .loss import kge_torch, masked_basin_loss
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
                                     noah_pet=cfg.noah_pet,
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
        dynamic_window=cfg.dynamic_window if cfg.dynamic_params else None)
    calobs = load_cal_obs(dom, data_dir, cal_start=cfg.cal_start)
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
                       canopy=cfg.canopy,
                       canopy_separate_trunk=cfg.canopy_separate_trunk,
                       canopy_lite=cfg.canopy_lite,
                       dynamic_params=cfg.dynamic_params,
                       dynamic_amp=cfg.dynamic_amp,
                       ).to(device=dev, dtype=dtype)
    if cfg.seasonal_params:
        print(f"train: seasonal (day-of-year harmonic) params {cfg.seasonal_params} "
              f"— 2 zero-init coeffs each, tanh-capped at |a|<={cfg.seasonal_amp} "
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
                                         weight=basin_w)
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
        _, st = _stream_nograd(dom, cfg, pe, ue, 0, calobs.t0, st0,
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
    print(f"train[{variant}]: {dom.n_hru} HRUs, {len(dom.basins)} basins, "
          f"{x.shape[1]} features | cal days {calobs.t0}..{calobs.t1} "
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
                if train_g is not None:
                    loss = train_g.run(pr, ta, doy, leap, obs_c, tn, tx, lai_c, st_c)
                else:
                    opt.zero_grad(set_to_none=True)
                    params, cp = _split_out(net(x), cfg.et_mode)
                    uh = routing_uh(params, dom.flowlen)
                    flow, state = run_window(
                        pr, ta, doy, leap, dom.lat_rad, dom.elev, params, uh,
                        state, n_inc=cfg.n_inc, perc_mode=cfg.perc_mode,
                        fracp_floor=cfg.fracp_floor, ninc_mode="fixed",
                        et_mode=cfg.et_mode, canopy_params=cp, tmin=tn, tmax=tx,
                        veg_frac=dom.veg_frac, lai=lai_c, noah_pet=cfg.noah_pet,
                        canopy_lite=cfg.canopy_lite, state_idx=st_c)
                    loss_t = masked_basin_loss(
                        dom.W @ flow, obs_c, calobs.obs_var, kind=cfg.loss,
                        log_lambda=cfg.log_loss_lambda, log_eps=cfg.log_loss_eps,
                        var_lambda=cfg.var_loss_lambda,
                        bias_lambda=cfg.bias_loss_lambda, weight=basin_w)
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
