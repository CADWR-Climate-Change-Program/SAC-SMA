"""Train the hybrid LSTM: MSE-family loss, cal-KGE selection (no val leakage).

The target is per-basin std-normalized (so basins weigh comparably — the NNSE
spirit), the loss is plain MSE in that space (+ an optional low-flow log term,
+ the optional TEMPERATURE-CONSISTENCY term below).  KGE is used ONLY as the
no-grad selection metric, pooled over the CAL window (WY1989-2003); validation
(WY2004-2018) is never read during training or selection.  Denormalization
back to mm/day happens here.

Temperature-consistency loss (``temp_lambda``): every batch is forwarded a
second time on the temperature-perturbed feature copy (``HybridData.feat_dt``:
tavg/tmin/tmax + temp_delta, sim channel from the physics run under the same
delta) and the hybrid's daily response ``pred_dt - pred`` is pulled toward the
physics response ``(sim_dt - sim)/scale`` by MSE.  The LSTM keeps its
within-climate skill but inherits the physics' climate sensitivity — the
counter to the regime-conditional volume bias cal-only training injects under
a shifted validation climate.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ...metrics import kge
from ..config import pick_device
from .data import load_hybrid_data
from .model import HybridLSTM


@dataclass
class HybridConfig:
    hidden: int = 128
    static_embed: int = 16
    dropout: float = 0.15
    use_statics: bool = False
    #: feed sin/cos day-of-year to the LSTM.  False removes the only explicit
    #: calendar input — corrections must then key off forcing/state/sim, which
    #: blocks the doy-conditioned mean corrections that injected val-period
    #: volume bias at the basins the physics already had right (NML/MRC/ORO).
    use_doy: bool = True
    #: feed the raw PT potential (basin-average, alb 0 / dew 0 — exactly the
    #: noah/noah_ft energy demand, recomputed from forcing) as an input
    #: channel: a physics-shaped temperature pathway for the LSTM.
    use_pet: bool = False
    physics_domain: str = "15cdec"   # HRU resolution of the frozen sim + forcing
    pet_source: str = "hamon"        # "hamon" | "priestley_taylor" (match --physics)
    pt_snow_albedo: float = 0.0      # PT snow-albedo refinement (pt = 0.6)
    pt_dewpoint_depression: float = 0.0   # PT dewpoint refinement (pt = 2.0)
    physics_et_scheme: str = "sac"   # "sac" | "noah_lite" (Noah-lite external ET)
    canopy_csv: str = ""             # params_canopy.csv (soil_chi) for noah_lite
    #: temperature-consistency loss weight; 0 disables (no second forward).
    temp_lambda: float = 0.0
    #: the perturbation (degC) baked into ``temp_sim_cache`` — must match the
    #: --temp-delta the teacher sim was dumped with.
    temp_delta: float = 2.0
    #: teacher daily-sim CSV: the SAME physics as the sim channel, re-run with
    #: tavg/tmin/tmax + temp_delta (`sacsma dpl evaluate <ckpt> --temp-delta`).
    temp_sim_cache: str = ""
    lr: float = 4e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    n_epochs: int = 60
    warmup_epochs: int = 3
    lr_min: float = 1e-5
    batch_size: int = 512
    input_noise: float = 0.1         # gaussian input jitter (regularizer)
    log_lambda: float = 0.15         # low-flow log-space term
    log_eps: float = 0.01
    eval_every: int = 2
    patience: int = 12
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self):
        if self.temp_lambda > 0 and not self.temp_sim_cache:
            raise ValueError("temp_lambda > 0 requires temp_sim_cache "
                             "(the physics daily sim under temp_delta)")


def _denorm_flow(pred_norm, data, bb):
    """Normalized net output -> physical mm/day flow."""
    return pred_norm * data.scale[bb]


@torch.no_grad()
def predict_days(model, data, bb, tt, batch: int = 4096):
    model.eval()
    out = torch.empty(len(bb), device=data.device)
    for i in range(0, len(bb), batch):
        b = bb[i:i + batch]
        t = tt[i:i + batch]
        st = data.static[b] if data.static is not None else None
        out[i:i + batch] = _denorm_flow(model(data.gather_windows(b, t), st),
                                        data, b)
    return out


def pooled_kge(model, data, split: str) -> tuple[float, list[float]]:
    """Mean per-basin KGE of the predicted flow over ``split`` days."""
    bb, tt = data.eval_days(split)
    flow = predict_days(model, data, bb, tt).cpu().numpy()
    obs = data.obs[bb, tt].cpu().numpy()
    bnp = bb.cpu().numpy()
    ks = []
    for i in range(len(data.basins)):
        m = bnp == i
        k = kge(flow[m], obs[m])                # drops NaN-obs pairs internally
        ks.append(float(k))
    finite = [k for k in ks if np.isfinite(k)]
    return (float(np.mean(finite)) if finite else float("nan")), ks


def train_hybrid(cfg: HybridConfig, *, data_dir: str = "data",
                 out_dir: str | Path, physics_csv: str | Path | None = None,
                 sim_cache: str | Path | None = None) -> dict:
    dev = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if sim_cache is None:
        # physics-tagged so distinct baselines never share a stale sim cache
        # (et_scheme appended only when != "sac" so existing caches stay valid)
        tag = f"frozen_sim_{cfg.physics_domain}_{cfg.pet_source}"
        if cfg.physics_et_scheme != "sac":
            tag += f"_{cfg.physics_et_scheme}"
        sim_cache = out.parent / f"{tag}.csv"

    data = load_hybrid_data(data_dir, physics_csv=physics_csv,
                            sim_cache=sim_cache, use_statics=cfg.use_statics,
                            use_doy=cfg.use_doy, use_pet=cfg.use_pet,
                            domain=cfg.physics_domain, pet_source=cfg.pet_source,
                            pt_snow_albedo=cfg.pt_snow_albedo,
                            pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                            et_scheme=cfg.physics_et_scheme,
                            canopy_csv=cfg.canopy_csv or None,
                            temp_sim_cache=(cfg.temp_sim_cache or None)
                            if cfg.temp_lambda > 0 else None,
                            temp_delta=cfg.temp_delta,
                            device=dev)
    model = HybridLSTM(data.n_feat, data.n_static,
                       hidden=cfg.hidden, static_embed=cfg.static_embed,
                       dropout=cfg.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    warm = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=max(1, cfg.warmup_epochs))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cfg.n_epochs - cfg.warmup_epochs), eta_min=cfg.lr_min)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [warm, cos], milestones=[cfg.warmup_epochs])

    bt = data.train_bt
    m = len(bt)
    print(f"hybrid: {len(data.basins)} basins, {data.n_feat} dyn "
          f"+ {data.n_static} static feats, {m} cal samples on {dev}"
          + (f", temp loss dT={cfg.temp_delta:+.1f} lambda={cfg.temp_lambda}"
             if cfg.temp_lambda > 0 else ""), flush=True)

    best = -1e9
    best_state = None
    stale = 0
    log_rows = []
    for ep in range(cfg.n_epochs):
        model.train()
        perm = torch.randperm(m, device=dev)
        tot = 0.0
        nb = 0
        tic = time.time()
        for i in range(0, m, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            b = bt[idx, 0]
            t = bt[idx, 1]
            x = data.gather_windows(b, t)
            noise = (cfg.input_noise * torch.randn_like(x)
                     if cfg.input_noise > 0 else None)
            if noise is not None:
                x = x + noise
            st = data.static[b] if data.static is not None else None
            pred = model(x, st)
            targ = data.obs[b, t] / data.scale[b]
            loss = ((pred - targ) ** 2).mean()
            if cfg.log_lambda > 0:
                flow = (pred * data.scale[b]).clamp_min(0.0)
                o = data.obs[b, t]
                loss = loss + cfg.log_lambda * (
                    (torch.log(flow + cfg.log_eps)
                     - torch.log(o + cfg.log_eps)) ** 2).mean()
            if cfg.temp_lambda > 0:
                # second forward on the perturbed copy, SAME noise (so the
                # delta is not noise-dominated); anchor the hybrid's daily
                # response to the physics response, per-basin normalized.
                x_dt = data.gather_windows(b, t, feat=data.feat_dt)
                if noise is not None:
                    x_dt = x_dt + noise
                pred_dt = model(x_dt, st)
                d_phys = (data.sim_dt[b, t] - data.sim[b, t]) / data.scale[b]
                loss = loss + cfg.temp_lambda * (
                    ((pred_dt - pred) - d_phys) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()

        cal_kge = float("nan")
        if ep % cfg.eval_every == 0:
            cal_kge, _ = pooled_kge(model, data, "cal")
            if cal_kge > best:
                best = cal_kge
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        lr = opt.param_groups[0]["lr"]
        log_rows.append((ep, tot / max(nb, 1), cal_kge, lr, time.time() - tic))
        print(f"epoch {ep:3d}/{cfg.n_epochs}  loss {tot / max(nb, 1):.4f}  "
              f"calKGE {cal_kge:.4f}  lr {lr:.2e}  best {best:.4f}  "
              f"{time.time() - tic:.0f}s", flush=True)
        if stale >= cfg.patience:
            print(f"early stop at epoch {ep} (best cal KGE {best:.4f})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {"model": model.state_dict(), "cfg": asdict(cfg),
            "physics_csv": str(physics_csv) if physics_csv else None,
            "sim_cache": str(sim_cache), "n_feat": data.n_feat,
            "n_static": data.n_static, "best_cal_kge": best}
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.save(ckpt, out / "checkpoints" / "best.pt")
    pd_log = np.array(log_rows)
    np.savetxt(out / "train_log.csv", pd_log,
               header="epoch,loss,cal_kge,lr,epoch_s", delimiter=",", comments="")
    print(f"hybrid: best cal KGE {best:.4f} -> {out}", flush=True)
    return ckpt
