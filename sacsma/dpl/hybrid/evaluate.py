"""Score the hybrid vs the observed gage, apples-to-apples with GA and dPL.

Reconstruct the daily flow (net output, clipped >= 0) and score it like a dPL run
(:func:`sacsma.dpl.evaluate.score_basins`) -> ``metrics.csv``.  ``compare_all`` merges the GA,
dPL and hybrid tables into one cal/val KGE comparison table, written by ``sacsma dpl study
hybrids`` (the per-basin dumbbell view is ``hybrid_progression.png``,
:func:`sacsma.dpl.studies.hybrids.make_hybrid_progression`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ... import paths
from ..config import pick_device
from ..evaluate import score_basins
from .data import data_for
from .model import HybridLSTM
from .train import predict_days


def _reconstruct(model, data) -> np.ndarray:
    """(B, T) hybrid flow (mm/day) at observed days (metrics need only those)."""
    bb, tt = data.eval_days("all")
    keep = torch.isfinite(data.obs[bb, tt])          # score only observed days
    bb, tt = bb[keep], tt[keep]
    flow = predict_days(model, data, bb, tt).clamp_min(0.0).cpu().numpy()
    pred = np.full((len(data.basins), len(data.dates)), np.nan)
    pred[bb.cpu().numpy(), tt.cpu().numpy()] = flow
    return pred


def _device() -> torch.device:
    try:
        return pick_device("cuda")
    except RuntimeError:
        return torch.device("cpu")


def _build_model(ck: dict, data, dev: torch.device) -> HybridLSTM:
    model = HybridLSTM(data.n_feat, data.n_static,
                       hidden=ck["cfg"]["hidden"],
                       static_embed=ck["cfg"]["static_embed"],
                       dropout=ck["cfg"]["dropout"]).to(dev)
    model.load_state_dict(ck["model"])
    return model


def _score_pred(pred: np.ndarray, data, data_dir: str, out: Path) -> pd.DataFrame:
    """Score a (B, T) daily-flow prediction against the gauges -> ``metrics.csv``."""
    out.mkdir(parents=True, exist_ok=True)
    return score_basins(pred, data.basins, data.dates, out, data_dir=data_dir, figures=False)


def score_hybrid(ckpt_path: str | Path, *, data_dir: str = "data",
                 out_dir: str | Path | None = None) -> pd.DataFrame:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # one member's score is local; the tracked score of an ensemble is its mean flow's
    out = (Path(out_dir) if out_dir is not None
           else paths.run_roles(Path(ckpt_path).parents[1]).local)
    dev = _device()
    data = data_for(ck, data_dir, dev)
    pred = _reconstruct(_build_model(ck, data, dev), data)
    return _score_pred(pred, data, data_dir, out)


def score_ensemble(ens_dir: str | Path, *, data_dir: str = "data",
                   out_dir: str | Path | None = None) -> pd.DataFrame:
    """Score the ENSEMBLE-MEAN daily flow across all trained seeds.

    Averages the per-seed reconstructed flow (mean of member flows — the
    canonical "keep full ensemble, use mean" convention) then scores it vs the
    gage exactly like :func:`score_hybrid` -> ``metrics.csv`` in the
    ensemble's results folder.  ``seed*/checkpoints/best.pt`` are the members; data is
    loaded once (every seed shares the physics/domain config)."""
    run = paths.run_roles(ens_dir)
    ens = run.model
    ckpts = sorted(ens.glob("seed*/checkpoints/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no seed*/checkpoints/best.pt under {ens}")
    out = Path(out_dir) if out_dir is not None else run.results
    dev = _device()
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    data = data_for(ck0, data_dir, dev)
    preds = []
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        preds.append(_reconstruct(_build_model(ck, data, dev), data))
    # every member shares the identical observed-day mask (same data.obs), so a
    # plain mean equals the per-cell nanmean without the all-NaN empty-slice warning
    pred = np.stack(preds, 0).mean(axis=0)
    print(f"ensemble {ens.name}: mean of {len(ckpts)} members", flush=True)
    return _score_pred(pred, data, data_dir, out)


def compare_all(out_dir: str | Path | None = None,
                *, ga_csv: str | Path | None = None,
                dpl_csv: str | Path | None = None,
                hybrid_csv: str | Path | None = None,
                pet_dt_csv: str | Path | None = None,
                ) -> pd.DataFrame:
    """Merge GA / dPL / hybrid-ensemble cal+val KGE into one comparison table, by default
    from the tracked score tables (the GA set, ``hamon_dense``, ``hybrid``, ``hybrid_dt``)
    into the ``hybrids`` study folder."""
    out = Path(out_dir) if out_dir is not None else paths.dpl_study(name="hybrids")
    ga_csv = ga_csv or paths.calibrated(name="15cdec") / "metrics.csv"
    dpl_csv = dpl_csv or paths.dpl_metrics(run="hamon_dense")
    hybrid_csv = hybrid_csv or paths.dpl_metrics(run="hybrid")
    pet_dt_csv = pet_dt_csv or paths.dpl_metrics(run="hybrid_dt")
    frames = {}
    for name, path in [("GA", ga_csv), ("dPL", dpl_csv),
                       ("hybrid", hybrid_csv),
                       ("hybrid_dt", pet_dt_csv)]:
        p = Path(path)
        if p.exists():
            d = pd.read_csv(p)[["basin", "cal_kge", "val_kge"]]
            frames[name] = d.rename(columns={"cal_kge": f"{name}_cal",
                                             "val_kge": f"{name}_val"})
    if "GA" not in frames:
        raise FileNotFoundError(f"need at least the GA table at {ga_csv}")
    merged = frames["GA"]
    for name, d in frames.items():
        if name != "GA":
            merged = merged.merge(d, on="basin", how="outer")
    csv = out / "compare_ga_dpl_hybrid.csv"
    merged.round(4).to_csv(csv, index=False)
    print(f"wrote {csv}", flush=True)
    return merged
