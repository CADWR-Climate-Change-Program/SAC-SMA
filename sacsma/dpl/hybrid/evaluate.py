"""Score the hybrid vs the observed gage, apples-to-apples with GA and dPL.

Reconstruct the final daily flow (feature: net output; residual: sim +
correction, clipped >= 0), split at :data:`sacsma.cdec15.CAL_END`, and run the
SAME ``_figures._period_stats`` used for GA/dPL -> ``metrics_hybrid_<variant>.csv``
(identical columns to ``metrics_15cdec.csv``).  ``compare_all`` merges the GA,
dPL and both hybrid tables into one cal/val KGE comparison + a dumbbell figure.
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
    """(B, T) final hybrid flow (mm/day) at observed days (metrics need only those)."""
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


def _load_data(ck: dict, data_dir: str, dev: torch.device):
    """Rebuild the HybridData for a checkpoint's physics/variant config."""
    cfg = ck["cfg"]
    return load_hybrid_data(
        data_dir, variant=ck["variant"],
        physics_csv=ck.get("physics_csv"),
        sim_cache=ck.get("sim_cache"),
        use_statics=bool(ck["n_static"]),
        use_doy=cfg.get("use_doy", True),
        domain=cfg.get("physics_domain", "15cdec"),
        pet_source=cfg.get("pet_source", "hamon"),
        pt_snow_albedo=cfg.get("pt_snow_albedo", 0.0),
        pt_dewpoint_depression=cfg.get("pt_dewpoint_depression", 0.0),
        et_scheme=cfg.get("physics_et_scheme", "sac"),
        canopy_csv=cfg.get("canopy_csv") or None,
        device=dev)


def _build_model(ck: dict, data, dev: torch.device) -> HybridLSTM:
    model = HybridLSTM(data.n_feat, data.n_static, variant=ck["variant"],
                       hidden=ck["cfg"]["hidden"],
                       static_embed=ck["cfg"]["static_embed"],
                       dropout=ck["cfg"]["dropout"]).to(dev)
    model.load_state_dict(ck["model"])
    return model


def _score_pred(pred: np.ndarray, data, data_dir: str, out: Path,
                variant: str) -> pd.DataFrame:
    """Score a (B, T) daily-flow prediction vs the gage, cal/val split ->
    ``metrics_hybrid_<variant>.csv`` (identical columns to metrics_15cdec.csv)."""
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
    csv = out / f"metrics_hybrid_{variant}.csv"
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
    return _score_pred(pred, data, data_dir, out, ck["variant"])


def score_ensemble(ens_dir: str | Path, *, data_dir: str = "data",
                   out_dir: str | Path | None = None) -> pd.DataFrame:
    """Score the ENSEMBLE-MEAN daily flow across all trained seeds.

    Averages the per-seed reconstructed flow (mean of member flows — the
    canonical "keep full ensemble, use mean" convention) then scores it vs the
    gage exactly like :func:`score_hybrid` -> ``metrics_hybrid_<variant>.csv``
    at ``ens_dir``.  ``seed*/checkpoints/best.pt`` are the members; data is
    loaded once (every seed shares the physics/variant/domain config)."""
    ens = Path(ens_dir)
    ckpts = sorted(ens.glob("seed*/checkpoints/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no seed*/checkpoints/best.pt under {ens}")
    out = Path(out_dir if out_dir is not None else ens)
    dev = _device()
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    data = _load_data(ck0, data_dir, dev)
    preds = []
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        preds.append(_reconstruct(_build_model(ck, data, dev), data))
    # every member shares the identical observed-day mask (same data.obs), so a
    # plain mean equals the per-cell nanmean without the all-NaN empty-slice warning
    pred = np.stack(preds, 0).mean(axis=0)
    print(f"ensemble {ens.name}: mean of {len(ckpts)} members "
          f"({ck0['variant']})", flush=True)
    return _score_pred(pred, data, data_dir, out, ck0["variant"])


def compare_all(out_dir: str | Path = "artifacts/dpl",
                *, ga_csv: str | Path = "artifacts/cdec15/metrics_15cdec.csv",
                dpl_csv: str | Path =
                "artifacts/dpl/hamon_dense/metrics_hamon_dense.csv",
                feat_csv: str | Path =
                "artifacts/dpl/noah_lstm_feat/metrics_hybrid_feature.csv",
                resid_csv: str | Path =
                "artifacts/dpl/noah_lstm_resid/metrics_hybrid_residual.csv",
                ) -> pd.DataFrame:
    """Merge GA / dPL / Noah-LSTM feature+residual ensemble cal+val KGE + dumbbell."""
    out = Path(out_dir)
    frames = {}
    for name, path in [("GA", ga_csv), ("dPL", dpl_csv),
                       ("noah_lstm_feat", feat_csv),
                       ("noah_lstm_resid", resid_csv)]:
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
    _dumbbell(merged, out / "compare_val_kge.png")
    return merged


def _dumbbell(merged: pd.DataFrame, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [c for c in merged.columns if c.endswith("_val")]
    labels = [c[:-4] for c in cols]
    colors = {"GA": "#888888", "dPL": "#1f77b4",
              "noah_lstm_feat": "#ff7f0e", "noah_lstm_resid": "#2ca02c"}
    y = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(merged) + 1))
    for c, lab in zip(cols, labels, strict=True):
        ax.scatter(merged[c].clip(lower=0), y, s=36,
                   color=colors.get(lab, "k"), label=lab, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(merged["basin"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("validation KGE (WY2004-2018)")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)
