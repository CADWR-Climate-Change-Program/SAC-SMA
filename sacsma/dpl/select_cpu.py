"""The training-year selection, scored every epoch on the CPU engine while the GPU trains
(``DplConfig.select_cpu``).

The trainer writes each epoch's candidate (the net as the GPU selection would score it, before
the epoch's updates) to ``<local>/checkpoints/select/eNNN.pt`` and starts :func:`main` in a
separate process.  The scorer runs every candidate, in epoch order, on the CPU engine over the
training window (the trainer's window spinup ahead of it), computes the trainer's own selection
statistic (:func:`selection_stat`) and adds a row to ``<local>/select_cpu.csv`` (the whole
file rewritten, then renamed: never a half-written row); the trainer reads the rows back, keeps
the best candidate as ``best.pt`` and counts stale epochs.  The scorer ends when the trainer
writes ``select/done`` and every candidate is scored, or when the trainer's process ends (its
pipe on the scorer's stdin closes).  A candidate whose simulation fails gets a NaN row; any
other failure ends the scorer, and the trainer starts another, which skips the epochs the
file already holds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import FAMILY_KEYS
from .loss import kge_torch


def _shares_stat(fam_means: dict[str, float], shares: dict[str, float]) -> float:
    """Share-weighted mean of the family means over the families present in
    ``fam_means`` (shares renormalized over those families)."""
    present = [f for f in fam_means if f in shares]
    if not present:
        return float("nan")
    tot = sum(shares[f] for f in present)
    return float(sum(shares[f] / tot * fam_means[f] for f in present))


def selection_stat(sim: torch.Tensor, obs: torch.Tensor, *, eobs=None, midx=None,
                   shares=None, sel_shares=None,
                   sel3_shares=None) -> tuple[float, dict, float | None, float]:
    """The trainer's selection statistic of the simulated flow ``sim`` (B, T) against ``obs``
    (B, T) on the cal window: (selection scalar, family means, sel3, pooled).
    Without ``eobs`` (the 15cdec domains): the pooled mean over the basins with >= 90 observed
    days.  With it (multi-timescale): daily rows at daily stride, monthly rows on calendar-month
    sums through ``midx`` (>= 12 months), and the scalar is the pooled mean or the
    ``sel_shares``-weighted family mean (a run with ``shares``); sel3 is the scalar without the
    CalSim3 arcs (None in a run without them)."""
    k = kge_torch(sim.double(), obs.double())
    valid = torch.isfinite(obs).sum(dim=1) >= 90
    if eobs is None:
        pooled = float(k[valid].mean())
        return pooled, {}, None, pooled
    m_rows = torch.as_tensor(eobs.monthly_rows, device=sim.device)
    sim_m = torch.zeros(len(eobs.monthly_rows), eobs.obs_monthly.shape[1],
                        device=sim.device, dtype=torch.float64)
    sim_m.index_add_(1, midx, sim[m_rows].double())
    k[m_rows] = kge_torch(sim_m, eobs.obs_monthly.double())
    valid[m_rows] = torch.isfinite(eobs.obs_monthly).sum(dim=1) >= 12
    fams = np.array(eobs.family)
    k_np, v_np = k.cpu().numpy(), valid.cpu().numpy()
    fam_means = {f: float(k_np[(fams == f) & v_np].mean()) for f in FAMILY_KEYS.values()
                 if ((fams == f) & v_np).any()}
    pooled = float(k[valid].mean())
    fm3 = {f: m for f, m in fam_means.items() if f != "calsim_monthly"}
    sel3 = None
    if len(fm3) < len(fam_means):
        sel3 = (_shares_stat(fm3, sel3_shares) if sel3_shares
                else _shares_stat(fm3, shares) if shares is not None
                else float(k_np[v_np & (fams != "calsim_monthly")].mean()))
    if shares is not None:
        scalar = _shares_stat(fam_means, sel_shares)
    else:
        scalar = pooled
    return scalar, fam_means, sel3, pooled


def start_scorer(local: Path, data_dir: str) -> subprocess.Popen:
    """:func:`main` in a CPU-only process that imports this same package (whatever the
    working directory puts first on its path), its stdin a pipe from this process; its output
    is added to ``select_cpu.log``."""
    root = str(Path(__file__).resolve().parents[2])
    code = (f"import sys; sys.path.insert(0, {root!r}); from sacsma.dpl.select_cpu import "
            f"main; main({str(local)!r}, {str(data_dir)!r}, with_parent=True)")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUNBUFFERED="1")
    with open(local / "select_cpu.log", "a", encoding="utf-8") as log:
        return subprocess.Popen([sys.executable, "-c", code], env=env, stdin=subprocess.PIPE,
                                stdout=log, stderr=subprocess.STDOUT)


def _setup(ckpt: Path, data_dir: str) -> dict:
    """The candidate's network, inputs and domain on the CPU and the training targets over its
    window, as the trainer builds them."""
    from ..io import MULTI_TIMESCALE_DOMAIN
    from .data import load_cal_obs
    from .evaluate import load_net_from_checkpoint
    from .multi_timescale import cut_entity_obs, load_entity_obs

    net, x, dom, cfg, ck = load_net_from_checkpoint(ckpt, data_dir, device="cpu")
    s: dict = dict(net=net, x=x, dom=dom, cfg=cfg, eobs=None, midx=None, **ck["sel_setup"])
    if ck.get("domain") == MULTI_TIMESCALE_DOMAIN:
        eobs = load_entity_obs(dom, data_dir, obs_mask=cfg.obs_mask,
                               holdout_wy=cfg.holdout_wy or None,
                               uf_train_start=cfg.uf_train_start or None,
                               train_window=cfg.train_window or None)
        if cfg.train_window:
            eobs = cut_entity_obs(eobs, dom.dates, cfg.train_window)
        obs = torch.full((len(dom.basins), eobs.t1 - eobs.t0), float("nan"),
                         dtype=torch.float64)
        obs[torch.as_tensor(eobs.daily_rows)] = eobs.obs_daily.double()
        win = dom.dates[eobs.t0:eobs.t1]
        s.update(eobs=eobs, obs=obs, t0=eobs.t0, t1=eobs.t1, midx=torch.as_tensor(
            (win.year * 12 + (win.month - 1)).to_numpy() - int(eobs.month_code[0])))
    else:
        cal = load_cal_obs(dom, data_dir, cal_start=cfg.cal_start, obs_mask=cfg.obs_mask)
        s.update(obs=cal.obs.double(), t0=cal.t0, t1=cal.t1)
    return s


def replace_retry(tmp: Path, dst: Path, copy: bool = True) -> None:
    """``tmp`` renamed over ``dst``; while another process has ``dst`` open (Windows refuses
    the rename then) retried for 10 s, then copied over it in place (``copy``) or raised."""
    for _ in range(100):
        try:
            tmp.replace(dst)
            return
        except PermissionError:
            time.sleep(0.1)
    if not copy:
        tmp.replace(dst)
    shutil.copyfile(tmp, dst)
    tmp.unlink()


def _write_rows(log: Path, rows: list[dict]) -> None:
    """``rows`` to ``log``: written whole, then renamed over it (:func:`replace_retry`; never
    copied, as the trainer could read a cut row: the scorer ends, the next one scores again)."""
    tmp = log.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    replace_retry(tmp, log, copy=False)


def _exit_with_parent() -> None:
    """Wait for the end of stdin (the trainer's process ended), then end this process."""
    while os.read(0, 4096):
        pass
    os._exit(0)


def main(local: str, data_dir: str, poll_s: float = 2.0, with_parent: bool = False) -> None:
    """Score every candidate under ``<local>/checkpoints/select/`` in epoch order until the
    trainer's ``done`` and the last candidate (see the module docstring), skipping the epochs
    ``select_cpu.csv`` already holds.  ``with_parent``: end when stdin closes (the trainer's
    process ended)."""
    from .evaluate import simulate_field

    if with_parent:
        # the raw descriptor, not sys.stdin: a daemon thread blocked in the buffered reader
        # aborts the interpreter's shutdown
        threading.Thread(target=_exit_with_parent, daemon=True).start()
    local = Path(local)
    sel, log = local / "checkpoints" / "select", local / "select_cpu.csv"
    rows = pd.read_csv(log).to_dict("records") if log.exists() else []
    scored = {int(r["epoch"]) for r in rows}
    s = None
    while True:
        todo = sorted(c for c in sel.glob("e*.pt") if int(c.stem[1:]) not in scored)
        if not todo:
            if (sel / "done").exists() and not any(
                    int(c.stem[1:]) not in scored for c in sel.glob("e*.pt")):
                break
            time.sleep(poll_s)
            continue
        c = todo[0]
        tic = time.time()
        if s is None:
            s = _setup(c, data_dir)
        ck = torch.load(c, map_location="cpu", weights_only=False)
        s["net"].load_state_dict(ck["net"])
        dom = s["dom"]
        try:
            flow = simulate_field(s["net"], s["x"], dom, s["cfg"], dom.dates[s["t0"]],
                                  dom.dates[s["t1"] - 1], dom.W.cpu().numpy(),
                                  spinup="window")["flow"]
            scalar, fam_means, sel3, pooled = selection_stat(
                torch.as_tensor(flow, dtype=torch.float64), s["obs"], eobs=s["eobs"],
                midx=s["midx"], shares=s["shares"], sel_shares=s["sel_shares"],
                sel3_shares=s["sel3_shares"])
        except (MemoryError, OSError):                      # the next scorer tries again
            raise
        except Exception:                                   # a NaN row: stale, never best
            traceback.print_exc()
            scalar, fam_means, sel3, pooled = float("nan"), {}, None, float("nan")
        rows.append({"epoch": int(ck["epoch"]), "score": scalar, "pooled": pooled,
                     "sel3": float("nan") if sel3 is None else sel3,
                     **{f"kge_{f.split('_')[0]}": m for f, m in fam_means.items()},
                     "secs": round(time.time() - tic, 1)})
        _write_rows(log, rows)
        scored.add(int(c.stem[1:]))
        print(f"select_cpu: epoch {rows[-1]['epoch']:3d}  {scalar:.4f}  "
              f"({rows[-1]['secs']:.0f}s)", flush=True)


def read_scores(local: Path, after: int) -> list[tuple[int, float, float]]:
    """The scorer's rows past epoch ``after``, in epoch order: (epoch, score, sel3)."""
    log = local / "select_cpu.csv"
    if not log.exists():
        return []
    d = pd.read_csv(log)
    d = d[d["epoch"] > after].sort_values("epoch")
    return [(int(e), float(v), float(s3))
            for e, v, s3 in zip(d["epoch"], d["score"], d["sel3"], strict=True)]
