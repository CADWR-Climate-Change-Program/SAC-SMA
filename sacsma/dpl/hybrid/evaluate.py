"""Score the hybrid vs the observed gage, apples-to-apples with GA and dPL.

Reconstruct the daily flow (net output, clipped >= 0), split at
:data:`sacsma.cdec15.CAL_END`, and run the SAME ``_figures._period_stats``
used for GA/dPL -> ``metrics_hybrid.csv`` (identical columns to
``metrics_15cdec.csv``).  ``compare_all`` merges the GA, dPL and hybrid tables
into one cal/val KGE comparison table (the per-basin dumbbell view is now
``figures/hybrid_progression.png``, :func:`sacsma.dpl.noah_ca_hybrids.make_hybrid_progression`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..._figures import _period_stats
from ...cdec15 import CAL_END
from ...io import load_basin_area, mmday_to_cfs
from ..config import pick_device
from .data import load_hybrid_data
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


def _check_feature(ck: dict) -> None:
    """The residual variant was retired 2026-07-16 — its checkpoints are dead."""
    if ck.get("variant", "feature") != "feature":
        raise ValueError("residual hybrid checkpoints are retired (dropped "
                         "2026-07-16); only feature checkpoints can be scored")


def _load_data(ck: dict, data_dir: str, dev: torch.device,
               physics_csv: str | None = None, sim_cache: str | None = None):
    """Rebuild the HybridData for a checkpoint's physics config.

    ``physics_csv`` / ``sim_cache`` override the checkpoint-stored training-time
    paths (used when an ensemble was canonicalized out of ``testing/`` and its
    stored paths are now stale — the canonical caller passes the moved paths)."""
    _check_feature(ck)
    cfg = ck["cfg"]
    return load_hybrid_data(
        data_dir,
        physics_csv=physics_csv if physics_csv is not None else ck.get("physics_csv"),
        sim_cache=sim_cache if sim_cache is not None else ck.get("sim_cache"),
        use_statics=bool(ck["n_static"]),
        use_doy=cfg.get("use_doy", True),
        use_pet=cfg.get("use_pet", False),
        use_sim=cfg.get("use_sim", True),
        domain=cfg.get("physics_domain", "15cdec"),
        pet_source=cfg.get("pet_source", "hamon"),
        pt_snow_albedo=cfg.get("pt_snow_albedo", 0.0),
        pt_dewpoint_depression=cfg.get("pt_dewpoint_depression", 0.0),
        et_scheme=cfg.get("physics_et_scheme", "sac"),
        canopy_csv=cfg.get("canopy_csv") or None,
        device=dev)


def _build_model(ck: dict, data, dev: torch.device) -> HybridLSTM:
    _check_feature(ck)
    model = HybridLSTM(data.n_feat, data.n_static,
                       hidden=ck["cfg"]["hidden"],
                       static_embed=ck["cfg"]["static_embed"],
                       dropout=ck["cfg"]["dropout"]).to(dev)
    model.load_state_dict(ck["model"])
    return model


def _score_pred(pred: np.ndarray, data, data_dir: str, out: Path) -> pd.DataFrame:
    """Score a (B, T) daily-flow prediction vs the gage, cal/val split ->
    ``metrics_hybrid.csv`` (identical columns to metrics_15cdec.csv)."""
    obs = data.obs.cpu().numpy()
    is_cal = np.asarray(data.dates <= pd.Timestamp(CAL_END))
    try:
        areas = load_basin_area(data_dir, domain="15cdec").set_index(
            "basin")["area_mi2"].to_dict()
    except FileNotFoundError:
        areas = {}
    rows = []
    for i, b in enumerate(data.basins):
        cal = _period_stats(pred[i][is_cal], obs[i][is_cal])
        val = _period_stats(pred[i][~is_cal], obs[i][~is_cal])
        area = areas.get(b, np.nan)
        rows.append({
            "basin": b, "area_mi2": area,
            "cal_kge": cal.get("kge"), "cal_nse": cal.get("nse"),
            "cal_pbias": cal.get("pbias"), "cal_r": cal.get("r"),
            "cal_n": cal.get("n", 0),
            "val_kge": val.get("kge"), "val_nse": val.get("nse"),
            "val_pbias": val.get("pbias"), "val_r": val.get("r"),
            "val_n": val.get("n", 0),
            "obs_mean_mmday": cal.get("obs_mean"),
            "obs_mean_cfs": mmday_to_cfs(cal.get("obs_mean") or np.nan, area),
        })
        print(f"  {b}: CAL KGE={cal.get('kge', float('nan')):.3f} "
              f"VAL KGE={val.get('kge', float('nan')):.3f}", flush=True)
    metrics = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "metrics_hybrid.csv"
    metrics.round(4).to_csv(csv, index=False)
    print(f"wrote {csv}  (mean cal {metrics['cal_kge'].mean():.3f} / "
          f"val {metrics['val_kge'].mean():.3f})", flush=True)
    return metrics


def score_hybrid(ckpt_path: str | Path, *, data_dir: str = "data",
                 out_dir: str | Path | None = None) -> pd.DataFrame:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = Path(out_dir if out_dir is not None else Path(ckpt_path).parents[1])
    dev = _device()
    data = _load_data(ck, data_dir, dev)
    pred = _reconstruct(_build_model(ck, data, dev), data)
    return _score_pred(pred, data, data_dir, out)


def score_ensemble(ens_dir: str | Path, *, data_dir: str = "data",
                   out_dir: str | Path | None = None,
                   physics_csv: str | None = None,
                   sim_cache: str | None = None) -> pd.DataFrame:
    """Score the ENSEMBLE-MEAN daily flow across all trained seeds.

    Averages the per-seed reconstructed flow (mean of member flows — the
    canonical "keep full ensemble, use mean" convention) then scores it vs the
    gage exactly like :func:`score_hybrid` -> ``metrics_hybrid.csv`` at
    ``ens_dir``.  ``seed*/checkpoints/best.pt`` are the members; data is
    loaded once (every seed shares the physics/domain config).  ``physics_csv`` /
    ``sim_cache`` override the checkpoint-stored paths (for canonicalized
    ensembles whose training-time ``testing/`` paths are now stale)."""
    ens = Path(ens_dir)
    ckpts = sorted(ens.glob("seed*/checkpoints/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no seed*/checkpoints/best.pt under {ens}")
    out = Path(out_dir if out_dir is not None else ens)
    dev = _device()
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    data = _load_data(ck0, data_dir, dev, physics_csv=physics_csv,
                      sim_cache=sim_cache)
    preds = []
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        preds.append(_reconstruct(_build_model(ck, data, dev), data))
    # every member shares the identical observed-day mask (same data.obs), so a
    # plain mean equals the per-cell nanmean without the all-NaN empty-slice warning
    pred = np.stack(preds, 0).mean(axis=0)
    print(f"ensemble {ens.name}: mean of {len(ckpts)} members", flush=True)
    return _score_pred(pred, data, data_dir, out)


def compare_all(out_dir: str | Path = "artifacts/dpl",
                *, ga_csv: str | Path = "artifacts/cdec15/metrics_15cdec.csv",
                dpl_csv: str | Path =
                "artifacts/dpl/hamon_dense/metrics_hamon_dense.csv",
                hybrid_csv: str | Path =
                "artifacts/dpl/hybrid/metrics_hybrid.csv",
                pet_dt_csv: str | Path =
                "artifacts/dpl/hybrid_dt/metrics_hybrid.csv",
                ) -> pd.DataFrame:
    """Merge GA / dPL / hybrid-ensemble cal+val KGE into one comparison table."""
    out = Path(out_dir)
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
