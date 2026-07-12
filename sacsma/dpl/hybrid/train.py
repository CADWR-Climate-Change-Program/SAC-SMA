"""Train the hybrid LSTM: MSE-family loss, cal-KGE selection (no val leakage).

The target is per-basin std-normalized (so basins weigh comparably — the NNSE
spirit), the loss is plain MSE in that space (+ an optional low-flow log term
for the feature variant).  KGE is used ONLY as the no-grad selection metric,
pooled over the CAL window (WY1989-2003); validation (WY2004-2018) is never
read during training or selection.  Denormalization back to mm/day and, for the
residual variant, the ``sim + correction`` reconstruction happen here.
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
    variant: str = "residual"        # "feature" | "residual"
    hidden: int = 128
    static_embed: int = 16
    dropout: float = 0.15
    use_statics: bool = False
    lr: float = 4e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    n_epochs: int = 60
    warmup_epochs: int = 3
    lr_min: float = 1e-5
    batch_size: int = 512
    input_noise: float = 0.1         # gaussian input jitter (regularizer)
    log_lambda: float = 0.15         # low-flow term (feature variant only)
    log_eps: float = 0.01
    eval_every: int = 2
    patience: int = 12
    seed: int = 0
    device: str = "cuda"


def _denorm_flow(pred_norm, data, bb, tt):
    """Normalized net output -> physical mm/day flow (per variant)."""
    flow = pred_norm * data.scale[bb]
    if data.variant == "residual":
        flow = data.sim[bb, tt] + flow
    return flow


@torch.no_grad()
def predict_days(model, data, bb, tt, batch: int = 4096):
    model.eval()
    out = torch.empty(len(bb), device=data.device)
    for i in range(0, len(bb), batch):
        b = bb[i:i + batch]
        t = tt[i:i + batch]
        st = data.static[b] if data.static is not None else None
        out[i:i + batch] = _denorm_flow(model(data.gather_windows(b, t), st),
                                        data, b, t)
    return out


def pooled_kge(model, data, split: str) -> tuple[float, list[float]]:
    """Mean per-basin KGE of the reconstructed flow over ``split`` days."""
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
        sim_cache = out.parent / "frozen_sim.csv"

    data = load_hybrid_data(data_dir, variant=cfg.variant, physics_csv=physics_csv,
                            sim_cache=sim_cache, use_statics=cfg.use_statics,
                            device=dev)
    model = HybridLSTM(data.n_feat, data.n_static, variant=cfg.variant,
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
    print(f"hybrid[{cfg.variant}]: {len(data.basins)} basins, {data.n_feat} dyn "
          f"+ {data.n_static} static feats, {m} cal samples on {dev}", flush=True)

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
            if cfg.input_noise > 0:
                x = x + cfg.input_noise * torch.randn_like(x)
            st = data.static[b] if data.static is not None else None
            pred = model(x, st)
            targ = data.target[b, t] / data.scale[b]
            loss = ((pred - targ) ** 2).mean()
            if cfg.variant == "feature" and cfg.log_lambda > 0:
                flow = (pred * data.scale[b]).clamp_min(0.0)
                o = data.obs[b, t]
                loss = loss + cfg.log_lambda * (
                    (torch.log(flow + cfg.log_eps)
                     - torch.log(o + cfg.log_eps)) ** 2).mean()
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
    ckpt = {"model": model.state_dict(), "cfg": asdict(cfg), "variant": cfg.variant,
            "physics_csv": str(physics_csv) if physics_csv else None,
            "sim_cache": str(sim_cache), "n_feat": data.n_feat,
            "n_static": data.n_static, "best_cal_kge": best}
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.save(ckpt, out / "checkpoints" / "best.pt")
    pd_log = np.array(log_rows)
    np.savetxt(out / "train_log.csv", pd_log,
               header="epoch,loss,cal_kge,lr,epoch_s", delimiter=",", comments="")
    print(f"hybrid[{cfg.variant}]: best cal KGE {best:.4f} -> {out}", flush=True)
    return ckpt
