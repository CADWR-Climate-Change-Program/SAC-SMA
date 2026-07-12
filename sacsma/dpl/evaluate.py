"""Fidelity benchmark: the archived GA parameters through the torch forward
vs the frozen reference model — the Phase-1 go/no-go gate.

For each named numerics config the whole domain (7891 HRUs) is streamed
through the differentiable pipeline under ``torch.no_grad()`` over the full
1915-2018 record (reference protocol: cold start, no warmup drop), basin flow
aggregated with the exact ``run_basin`` weights, then scored per basin:

* sim-vs-sim daily KGE / NSE / pbias / max|delta| against the frozen truth;
* obs-scored cal/val KGE (split at ``CAL_END``) for both models -> deltas.

Gates (see the dpl plan): G1 structural — best ``ref-ninc*`` config KGE >=
0.999 and |dKGE_obs| <= 0.005 everywhere; G2 training numerics —
``train-default`` KGE >= 0.99, |dKGE_obs| <= 0.02; G3 precision — float32 vs
float64 same-config KGE >= 0.9999.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..cdec15 import BASINS, CAL_END, load_gage
from ..metrics import kge, nse, pbias
from ..model import run_basin
from .config import PARAM_ORDER, DplConfig, pick_device
from .data import DomainTensors, load_domain_tensors
from .forward import initial_state, routing_uh, run_window


def export_params(net: torch.nn.Module, dom: DomainTensors,
                  x: torch.Tensor) -> pd.DataFrame:
    """Learned per-HRU parameters as a ga_optimum-shaped table (+ ``basin``).

    Directly consumable by ``run_basin(..., params=...)``: the exact 34
    ``ga_optimum.csv`` columns keyed by grid-cell ``key``, plus the per-basin
    ``basin`` column the model.py filter uses — the ~1835 cells shared between
    basins intentionally carry per-basin values.  Fixed parameters
    (side/SCF/PXTEMP) come out at their GA constants.
    """
    net.eval()
    with torch.no_grad():
        out = net(x)
    df = dom.hrus[["basin", "key", "lat", "lon"]].copy()
    for p in PARAM_ORDER:
        df[p] = out[p].double().cpu().numpy()
    # seasonal harmonic coefficients (2 extra columns per seasonal param); the
    # frozen run_basin detects the ``_asin`` columns and reconstructs the same
    # day-of-year series.  Absent for a static net -> a plain ga_optimum table.
    for p in getattr(net, "seasonal_params", ()):
        df[f"{p}_asin"] = out[f"{p}_asin"].double().cpu().numpy()
        df[f"{p}_acos"] = out[f"{p}_acos"].double().cpu().numpy()
    return df

#: named numerics configs — (ninc, perc_mode, fracp_floor, dtype)
FIDELITY_CONFIGS: dict[str, DplConfig] = {
    # exact reference numerics (dynamic per-lane ninc): proves the port itself
    "ref-exact": DplConfig(ninc_mode="dynamic", perc_mode="reference", dtype="float64"),
    "ref-ninc1": DplConfig(n_inc=1, perc_mode="reference", dtype="float64"),
    "ref-ninc2": DplConfig(n_inc=2, perc_mode="reference", dtype="float64"),
    "ref-ninc5": DplConfig(n_inc=5, perc_mode="reference", dtype="float64"),
    "ref-ninc10": DplConfig(n_inc=10, perc_mode="reference", dtype="float64"),
    "ref-ninc20": DplConfig(n_inc=20, perc_mode="reference", dtype="float64"),
    "train-default": DplConfig(n_inc=5, perc_mode="implicit", fracp_floor=0.1,
                               dtype="float64"),
    "train-default-f32": DplConfig(n_inc=5, perc_mode="implicit", fracp_floor=0.1,
                                   dtype="float32"),
    "train-tanh": DplConfig(n_inc=5, perc_mode="tanh", fracp_floor=0.1,
                            dtype="float64"),
}

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def frozen_truth(data_dir: str = "data") -> dict[str, pd.DataFrame]:
    """Full-record frozen-model flow per basin (the parity-exact reference)."""
    from ..model import load_domain_forcing

    forcing = load_domain_forcing(data_dir, domain="15cdec")
    out: dict[str, pd.DataFrame] = {}
    for b in BASINS:
        out[b] = run_basin(b, data_dir=data_dir, domain="15cdec",
                           forcing=forcing, parallel=True)
    return out


def torch_domain_flow(
    dom: DomainTensors,
    params: dict[str, torch.Tensor],
    cfg: DplConfig,
    *,
    chunk_days: int = 4096,
    progress: bool = True,
) -> np.ndarray:
    """Stream the full record through the torch pipeline; (B, T) basin flow."""
    n, t_total = dom.n_hru, dom.n_time
    uh = routing_uh(params, dom.flowlen)
    state = initial_state(n, dom.device, dom.dtype, init_mode=cfg.init_mode,
                          params=params)
    basin_flow = torch.empty(len(dom.basins), t_total, device=dom.device,
                             dtype=dom.dtype)
    t0 = 0
    tic = time.time()
    with torch.no_grad():
        while t0 < t_total:
            t1 = min(t0 + chunk_days, t_total)
            pr, ta, doy, leap = dom.chunk(t0, t1)
            flow, state = run_window(pr, ta, doy, leap, dom.lat_rad, dom.elev,
                                     params, uh, state, n_inc=cfg.n_inc,
                                     perc_mode=cfg.perc_mode,
                                     fracp_floor=cfg.fracp_floor,
                                     ninc_mode=cfg.ninc_mode)
            basin_flow[:, t0:t1] = dom.W @ flow
            if progress:
                print(f"    days {t1}/{t_total}  ({time.time() - tic:.0f}s)", flush=True)
            t0 = t1
    return basin_flow.double().cpu().numpy()


def _obs_kge(sim: pd.Series, obs: pd.Series, dates: pd.DatetimeIndex,
             period: str) -> float:
    mask = (dates <= pd.Timestamp(CAL_END)) if period == "cal" else \
        (dates > pd.Timestamp(CAL_END))
    m = mask & np.isfinite(obs.to_numpy()) & np.isfinite(sim.to_numpy())
    if m.sum() < 90:
        return float("nan")
    return kge(sim.to_numpy()[m], obs.to_numpy()[m])


def score_frozen(
    params: pd.DataFrame,
    data_dir: str = "data",
    out_dir: str | Path = "artifacts/dpl/static",
    *,
    label: str = "dpl_static",
    cal_end: str = CAL_END,
    domain: str = "15cdec",
    parallel: bool = True,
) -> pd.DataFrame:
    """Score a parameter table through the FROZEN model vs the observed gage.

    The dPL reporting path: full-record ``run_basin(..., params=...)`` per
    basin (frozen physics — the torch pipeline is never a source of reported
    skill), daily cal/val split at ``cal_end``, per-basin diagnostics + skill
    summary in the cdec15 figure conventions ->
    ``<out_dir>/metrics_<label>.csv`` + ``figures/``.  Same columns as
    ``metrics_15cdec.csv`` so the GA-vs-dPL comparison is a plain merge.
    """
    from .._figures import (
        _period_stats,
        basin_diagnostics_fig,
        folsom_before_yuba,
        skill_summary_fig,
    )
    from ..io import load_basin_area, load_hru_table, mmday_to_cfs
    from ..model import load_domain_forcing

    out = Path(out_dir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    cal_end_ts = pd.Timestamp(cal_end)

    ref = "15cdec" if domain.endswith("_grid") else domain  # basins shared w/ 15cdec
    hru = load_hru_table(data_dir, domain=domain)
    basins = folsom_before_yuba(
        ref,
        hru.groupby("basin")["lat"].mean().sort_values(ascending=False).index.tolist())
    forcing = load_domain_forcing(data_dir, domain=domain)
    try:
        areas = load_basin_area(data_dir, domain=ref).set_index(
            "basin")["area_mi2"].to_dict()
    except FileNotFoundError:
        areas = {}

    records = []
    for b in basins:
        sim = run_basin(b, data_dir=data_dir, domain=domain, forcing=forcing,
                        params=params, parallel=parallel).rename(
                            columns={"flow": "flow_sim"})
        obs = load_gage(data_dir, basin=b)[["date", "flow"]].rename(
            columns={"flow": "flow_obs"})
        m = pd.merge(sim, obs, on="date", how="left").sort_values(
            "date").reset_index(drop=True)
        is_cal = m["date"] <= cal_end_ts
        cal = _period_stats(m.loc[is_cal, "flow_sim"].to_numpy(),
                            m.loc[is_cal, "flow_obs"].to_numpy())
        val = _period_stats(m.loc[~is_cal, "flow_sim"].to_numpy(),
                            m.loc[~is_cal, "flow_obs"].to_numpy())
        obs_dates = m.loc[m["flow_obs"].notna(), "date"]
        mplot = (m[m["date"] >= obs_dates.min()].reset_index(drop=True)
                 if not obs_dates.empty else m)
        basin_diagnostics_fig(b, mplot, cal_end_ts, cal, val,
                              figdir / f"{b}_diagnostics.png")
        area = areas.get(b, np.nan)
        records.append({
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

    metrics = pd.DataFrame(records)
    skill_summary_fig(metrics, figdir / "skill_summary.png")
    csv = out / f"metrics_{label}.csv"
    metrics.round(4).to_csv(csv, index=False)
    print(f"wrote {csv}", flush=True)
    return metrics


def evaluate_checkpoint(
    ckpt_path: str | Path,
    data_dir: str = "data",
    out_dir: str | Path | None = None,
    *,
    parallel: bool = True,
    label: str | None = None,
) -> pd.DataFrame:
    """best.pt -> params_dpl.csv -> frozen-model metrics (the full Phase-4 path)."""
    import numpy as _np

    from ..io import soilveg_path
    from .features import FeatureSet, build_features
    from .parameter_net import ParameterNet

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    variant = ck["variant"]
    domain = ck.get("domain", "15cdec")
    out = Path(out_dir if out_dir is not None else f"artifacts/dpl/{variant}")
    out.mkdir(parents=True, exist_ok=True)

    dom = load_domain_tensors(data_dir, domain=domain, device="cpu",
                              dtype=torch.float64)
    stats = FeatureSet(x=_np.empty((0, 0), dtype=_np.float32), **ck["features"])
    fs = build_features(dom.hrus, variant=variant,
                        forcing=dom.forcing if variant == "climate" else None,
                        climate_window=stats.climate_window,
                        climate_product=stats.climate_product,
                        physical_path=(soilveg_path(data_dir, domain)
                                       if variant == "physical" else None),
                        stats=stats)
    x = torch.as_tensor(fs.x).to("cpu", torch.float64)
    nc = ck.get("net_config", {})
    gnn_k = nc.get("gnn_k", 0)
    net = ParameterNet(x.shape[1], hidden=nc.get("hidden", 64),
                       embed=nc.get("embed", 32),
                       dropout=nc.get("dropout", 0.1),
                       grouped_heads=nc.get("grouped_heads", False),
                       gnn_k=gnn_k,
                       n_nodes=x.shape[0] if gnn_k > 0 else None,
                       seasonal_params=tuple(nc.get("seasonal_params", ())),
                       seasonal_amp=nc.get("seasonal_amp", 0.18)).double()
    net.load_state_dict(ck["net"])   # restores baked neighbor buffers too

    dpl_df = export_params(net, dom, x)
    pcsv = out / "params_dpl.csv"
    dpl_df.to_csv(pcsv, index=False)
    print(f"wrote {pcsv} ({len(dpl_df)} HRU rows, cal KGE at selection "
          f"{ck.get('cal_kge', float('nan')):.4f})", flush=True)
    return score_frozen(dpl_df, data_dir, out,
                        label=label if label is not None else f"dpl_{variant}",
                        domain=domain, parallel=parallel)


def fidelity_benchmark(
    data_dir: str = "data",
    out_dir: str = "artifacts/dpl/fidelity",
    *,
    configs: tuple[str, ...] | None = None,
    device: str = "cuda",
    chunk_days: int = 4096,
) -> pd.DataFrame:
    """Run the sweep; writes ``fidelity_benchmark.csv`` + a summary figure."""
    dev = pick_device(device)
    names = tuple(configs if configs is not None else FIDELITY_CONFIGS)
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    print("fidelity: frozen reference (run_basin, parallel) ...", flush=True)
    truth = frozen_truth(data_dir)
    dates = pd.DatetimeIndex(truth[BASINS[0]]["date"])
    gage = load_gage(data_dir)
    obs = {b: g.set_index("date")["flow"].reindex(dates)
           for b, g in gage.groupby("basin")}

    rows = []
    doms: dict[torch.dtype, DomainTensors] = {}
    for name in names:
        cfg = FIDELITY_CONFIGS[name]
        dtype = _DTYPES[cfg.dtype]
        if dtype not in doms:
            doms[dtype] = load_domain_tensors(data_dir, device=dev, dtype=dtype)
        dom = doms[dtype]
        ga = dom.ga_params(data_dir)
        print(f"fidelity: config {name} (n_inc={cfg.n_inc}, perc={cfg.perc_mode}, "
              f"{cfg.dtype}) ...", flush=True)
        tic = time.time()
        sim = torch_domain_flow(dom, ga, cfg, chunk_days=chunk_days)
        wall = time.time() - tic
        for b_i, b in enumerate(dom.basins):
            ref = truth[b]["flow"].to_numpy()
            s = sim[b_i]
            s_ser = pd.Series(s, index=dates)
            r_ser = pd.Series(ref, index=dates)
            ob = obs.get(b)
            rows.append({
                "config": name, "basin": b,
                "n_inc": cfg.n_inc, "perc_mode": cfg.perc_mode, "dtype": cfg.dtype,
                "kge_sim": kge(s, ref), "nse_sim": nse(s, ref),
                "pbias_sim": pbias(s, ref),
                "max_abs_diff": float(np.max(np.abs(s - ref))),
                "cal_kge_torch": _obs_kge(s_ser, ob, dates, "cal") if ob is not None else np.nan,
                "val_kge_torch": _obs_kge(s_ser, ob, dates, "val") if ob is not None else np.nan,
                "cal_kge_frozen": _obs_kge(r_ser, ob, dates, "cal") if ob is not None else np.nan,
                "val_kge_frozen": _obs_kge(r_ser, ob, dates, "val") if ob is not None else np.nan,
                "wall_s": round(wall, 1),
            })
        df_c = pd.DataFrame([r for r in rows if r["config"] == name])
        print(f"  {name}: sim-vs-sim KGE min {df_c['kge_sim'].min():.6f} | "
              f"max|d| {df_c['max_abs_diff'].max():.4f} mm/day | {wall:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df["d_cal_kge"] = df["cal_kge_torch"] - df["cal_kge_frozen"]
    df["d_val_kge"] = df["val_kge_torch"] - df["val_kge_frozen"]
    df.to_csv(out / "fidelity_benchmark.csv", index=False)
    _fidelity_figure(df, out / "figures" / "fidelity_benchmark.png")
    _print_gates(df)
    return df


def _print_gates(df: pd.DataFrame) -> None:
    ref = df[df["config"].str.startswith("ref-ninc")]
    if len(ref):
        best = ref.groupby("config")["kge_sim"].min().idxmax()
        b = ref[ref["config"] == best]
        g1 = (b["kge_sim"].min() >= 0.999
              and b["d_cal_kge"].abs().max() <= 0.005
              and b["d_val_kge"].abs().max() <= 0.005)
        print(f"G1 structural [{best}]: min KGE {b['kge_sim'].min():.6f}, "
              f"max|dKGE| {max(b['d_cal_kge'].abs().max(), b['d_val_kge'].abs().max()):.4f}"
              f" -> {'PASS' if g1 else 'FAIL'}")
    td = df[df["config"] == "train-default"]
    if len(td):
        g2 = (td["kge_sim"].min() >= 0.99
              and td["d_cal_kge"].abs().max() <= 0.02
              and td["d_val_kge"].abs().max() <= 0.02)
        print(f"G2 train-default: min KGE {td['kge_sim'].min():.6f} "
              f"-> {'PASS' if g2 else 'FAIL'}")
    f32 = df[df["config"] == "train-default-f32"]
    if len(td) and len(f32):
        m = td.merge(f32, on="basin", suffixes=("_64", "_32"))
        # float32-vs-float64 agreement, both scored against the same frozen truth
        g3 = (m["kge_sim_32"] - m["kge_sim_64"]).abs().max() <= 1e-3
        print(f"G3 precision: max |KGE_f32 - KGE_f64| "
              f"{(m['kge_sim_32'] - m['kge_sim_64']).abs().max():.5f} "
              f"-> {'PASS' if g3 else 'CHECK'}")


def _fidelity_figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = list(dict.fromkeys(df["config"]))
    basins = [b for b in BASINS if b in set(df["basin"])]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.2), dpi=300, sharex=True)
    x = np.arange(len(basins))
    width = 0.8 / len(configs)
    for c_i, c in enumerate(configs):
        d = df[df["config"] == c].set_index("basin").reindex(basins)
        axes[0].bar(x + c_i * width, d["kge_sim"], width, label=c)
        axes[1].bar(x + c_i * width, d["max_abs_diff"], width, label=c)
    axes[0].set_ylabel("sim-vs-sim daily KGE", fontsize=8)
    axes[0].set_ylim(0.95, 1.001)
    axes[0].axhline(0.999, color="0.4", lw=0.6, ls="--")
    axes[1].set_ylabel("max |torch − frozen| (mm/day)", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x + 0.4 - width / 2)
    axes[1].set_xticklabels(basins, fontsize=7, rotation=45)
    axes[0].legend(fontsize=6, ncol=2, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle("dPL torch forward vs frozen reference — archived GA parameters",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
