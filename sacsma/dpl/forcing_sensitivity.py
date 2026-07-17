"""Temperature-detrending sensitivity (WGEN Product A minus historical Livneh)
for the dPL hamon, pt and noah physics models plus the canonical Hybrid
ensembles (mean over seed members, on the noah physics baseline).

The ensembles' sac_sim channel is noah's TORCH daily run (their training
channel), so the detrended channel is rebuilt by streaming the torch pipeline
under the per-cell dT field (``evaluate.noah_torch_daily`` on the noah ckpt) —
numerics-matched to the channel the LSTMs were trained on (the displayed
``dPL noah`` series stays the frozen run_basin pair).  Because the WGEN
detrending never enters the hybrid's TRAINING (the temperature-consistency
loss trains on a scalar +2degC teacher), the Hybrid series here is an
INDEPENDENT check of its temperature response.

WGEN Product A detrends temperature to a 1991-2020 baseline (the early record is
warmed).  It is not packaged for the 15cdec application, so the detrending signal
is reconstructed as a temperature-level shift::

    dT(cell, day) = wgen_tavg - historical_tavg      (from the CalSim WGEN stores,
                                                       key-matched to 15cdec_grid;
                                                       0 for uncovered cells)

applied to tavg/tmin/tmax (precip is identical between the products).  Only the
11 rim basins whose grid cells are covered by the CalSim WGEN stores are shown
(92-100% cell coverage); the 4 Tulare basins have no WGEN and are dropped.

Two figures on the basin-aggregated flow change dQ = detrended - historical:
  * rolling (water-year) aggregate dQ over time (the detrending effect shrinks
    toward zero as the record approaches the 1991-2020 baseline);
  * the monthly dQ regime over the pre-1950 period (largest detrending effect).

Output: artifacts/dpl/figures/cdec15_forcing_sensitivity_*.png.
"""

from __future__ import annotations

import dataclasses as dc
from pathlib import Path

import numpy as np
import pandas as pd

from ..cdec15 import CAL_END
from ..io import load_forcing
from ..model import attach_tminmax, load_domain_forcing, run_basin
from .climatology import _WY, _WY_LABELS, _monthly_taf

DOMAIN = "15cdec_grid"
_PRE1950 = "1950-01-01"
#: pure-physics frozen sims (run_basin).  ``dPL pt`` is the refined PT cascade;
#: ``dPL noah`` is the frozen Noah-lite external-ET core (bit-exact vs torch).
MODELS: dict[str, dict] = {
    "dPL hamon": dict(csv="artifacts/dpl/hamon/params_dpl.csv",
                      pet="hamon", alb=0.0, dew=0.0),
    "dPL pt":    dict(csv="artifacts/dpl/pt/params_dpl.csv",
                      pet="priestley_taylor", alb=0.6, dew=2.0),
    "dPL noah":  dict(csv="artifacts/dpl/noah/params_dpl.csv",
                      pet="priestley_taylor", alb=0.0, dew=0.0, et_scheme="noah_lite",
                      canopy_csv="artifacts/dpl/noah/params_canopy.csv"),
}
#: the noah checkpoint: the ensembles' sac_sim channel is its TORCH daily run
#: (``daily_sim_noah_torch.csv`` baked into the seed ckpts), so the detrended
#: channel streams the torch pipeline under the dT field.
NOAH_CKPT = "artifacts/dpl/noah/checkpoints/best.pt"
#: the canonical Hybrid ENSEMBLES (mean over seed members); both sac_sim
#: channels are the noah physics, whose detrended torch run is re-fed as the
#: ensembles' detrended baseline.  ``Hybrid`` (plain: no PET, no dT loss) is
#: the improvement BASELINE — its near-flat/wrong-signed response against
#: ``Hybrid PET+dT`` (PT-potential input + temperature-consistency loss) is
#: the point of this figure.
ENSEMBLES: dict[str, str] = {
    "Hybrid":        "artifacts/dpl/hybrid",
    "Hybrid PET+dT": "artifacts/dpl/hybrid_pet_dt",
}
#: 2-D encoding so the series separate cleanly: COLOR = physics lineage (blue =
#: Hamon, red = PT cascade, green = Noah-lite); LINESTYLE = role (solid = pure
#: physics, dashed/dash-dot = LSTM ensembles).  Read the ET-scheme effect
#: across colours, physics-vs-LSTM within green.
STYLE: dict[str, dict] = {
    "dPL hamon":     dict(color="#1f77b4", lw=2.3, ls="-"),
    "dPL pt":        dict(color="#d62728", lw=2.3, ls="-"),
    "dPL noah":      dict(color="#2ca02c", lw=2.3, ls="-"),
    "Hybrid":        dict(color="#2ca02c", lw=2.0, ls="--"),
    "Hybrid PET+dT": dict(color="#2ca02c", lw=2.0, ls="-."),
}
#: marker by role (reinforces the linestyle on the monthly plot only; the rolling
#: time series stays marker-free).
_ROLE_MARKER = {"-": "o", "--": "s", "-.": "^"}


def _norm_key(k) -> str:
    a, b = str(k).split("_")
    return f"{round(float(a), 5)}_{round(float(b), 5)}"


def _delta_t(data_dir: str, f_hist) -> np.ndarray:
    """dT = wgen_tavg - historical_tavg per historical-cell row (0 = uncovered)."""
    wgen: dict[str, np.ndarray] = {}
    for d in ("9unimp", "11obs", "12rim"):
        ds = load_forcing(data_dir, domain=d, product="wgen_product_a")
        try:
            tav = ds["tavg"].values
            for i, k in enumerate(ds["key"].values):
                wgen.setdefault(_norm_key(k), tav[i])
        finally:
            ds.close()
    dT = np.zeros_like(f_hist.tavg)
    covered = 0
    for k, row in f_hist.pos.items():
        w = wgen.get(_norm_key(k))
        if w is not None:
            dT[row] = w.astype(f_hist.tavg.dtype) - f_hist.tavg[row]
            covered += 1
    print(f"  dT: {covered}/{len(f_hist.pos)} grid cells covered by CalSim WGEN",
          flush=True)
    return dT


def _forcings(data_dir: str):
    """Historical + detrended DomainForcing (tmin/tmax attached, temp shifted)."""
    f_hist = load_domain_forcing(data_dir, domain=DOMAIN)
    attach_tminmax(data_dir, DOMAIN, f_hist)
    dT = _delta_t(data_dir, f_hist)
    f_detr = dc.replace(f_hist, tavg=f_hist.tavg + dT, tmin=f_hist.tmin + dT,
                        tmax=f_hist.tmax + dT, _f64={})
    return f_hist, f_detr, dT


def _frozen_sim(forcing, spec: dict, basins, cache: Path | None = None) -> pd.DataFrame:
    """Daily basin flow (mm/day) under a given forcing for a frozen model."""
    if cache is not None and cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")
    params = pd.read_csv(spec["csv"])
    canopy = pd.read_csv(spec["canopy_csv"]) if spec.get("canopy_csv") else None
    et_scheme = spec.get("et_scheme", "sac")
    seasonal = any(str(c).endswith("_asin") for c in params.columns)
    cols = {}
    for b in basins:
        s = run_basin(b, data_dir="data", domain=DOMAIN, forcing=forcing,
                      params=params, parallel=not seasonal, pet_source=spec["pet"],
                      pt_snow_albedo=spec["alb"], pt_dewpoint_depression=spec["dew"],
                      et_scheme=et_scheme, canopy_params=canopy)
        cols[b] = s.set_index("date")["flow"]
    df = pd.DataFrame(cols)
    df.index.name = "date"
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    return df


def _hybrid_flow(data_dir, dT, dev, sim_detr: pd.DataFrame, ckpt: str):
    """(flow_hist, flow_detr) full-record daily hybrid flow (date x basin).

    Detrended features = historical features + dT_basin/sigma on tavg/tmin/tmax
    (additive under z-score) with the sac_sim channel re-fed by the detrended
    physics sim (per-basin ÷scale, as load_hybrid_data does).  The trained
    normalisation is reused exactly.  ``ckpt`` is one trained seed member's
    checkpoint."""
    import torch

    from .hybrid.data import _CAL_START, feature_names, load_hybrid_data
    from .hybrid.model import HybridLSTM
    from .hybrid.train import predict_days

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if ck.get("variant", "feature") != "feature":
        raise ValueError("residual hybrid checkpoints are retired (2026-07-16)")
    use_doy = cfg.get("use_doy", True)
    use_pet = cfg.get("use_pet", False)
    h = load_hybrid_data(
        data_dir, physics_csv=ck.get("physics_csv"),
        sim_cache=ck.get("sim_cache"), use_statics=bool(ck["n_static"]),
        use_doy=use_doy, use_pet=use_pet,
        domain=cfg.get("physics_domain", "15cdec"),
        pet_source=cfg.get("pet_source", "hamon"),
        pt_snow_albedo=cfg.get("pt_snow_albedo", 0.0),
        pt_dewpoint_depression=cfg.get("pt_dewpoint_depression", 0.0),
        et_scheme=cfg.get("physics_et_scheme", "sac"),
        canopy_csv=cfg.get("canopy_csv") or None, device=dev)
    model = HybridLSTM(h.n_feat, h.n_static, hidden=cfg["hidden"],
                       static_embed=cfg["static_embed"], dropout=cfg["dropout"]).to(dev)
    model.load_state_dict(ck["model"])

    # basin-level dT and the historical cal-window sigmas used in normalisation
    from .data import load_domain_tensors
    dom = load_domain_tensors(data_dir, domain=DOMAIN, device="cpu", dtype=torch.float64)
    W = dom.W.numpy()
    dTb = W @ dT[dom.cell_idx].astype(np.float64)                 # (B, T) basin dT
    lo = int(dom.dates.searchsorted(pd.Timestamp(_CAL_START)))
    hi = int(dom.dates.searchsorted(pd.Timestamp(CAL_END))) + 1
    tavg = W @ dom.forcing.tavg[dom.cell_idx].astype(np.float64)
    tmm = pd.read_csv(Path(data_dir) / "cdec15" / "basin_tminmax_livneh.csv",
                      parse_dates=["date"]).set_index("date")
    tmin = np.vstack([tmm[f"tmin_{b}"].reindex(dom.dates).to_numpy() for b in dom.basins])
    tmax = np.vstack([tmm[f"tmax_{b}"].reindex(dom.dates).to_numpy() for b in dom.basins])
    sd = {"tavg": tavg[:, lo:hi].std() + 1e-8, "tmin": tmin[:, lo:hi].std() + 1e-8,
          "tmax": tmax[:, lo:hi].std() + 1e-8}
    names = feature_names(use_doy, use_pet)
    idx = {n: names.index(n) for n in ("tavg", "tmin", "tmax", "sac_sim")}

    sim_d = np.vstack([sim_detr[b].reindex(dom.dates).to_numpy() for b in dom.basins])
    feat_d = h.feat.clone()
    dev_kw = dict(dtype=feat_d.dtype, device=feat_d.device)
    dTb_t = torch.as_tensor(dTb, **dev_kw)
    sim_d_t = torch.as_tensor(sim_d, **dev_kw)
    for n in ("tavg", "tmin", "tmax"):
        feat_d[:, :, idx[n]] += dTb_t / sd[n]
    if use_pet:
        # PET is deterministic in T — recompute exactly under the dT field,
        # normalized with the trained (historical cal-window) stats.
        from .hybrid.data import basin_pet_pt
        pet_h = basin_pet_pt(dom)
        pet_d = basin_pet_pt(dom, delta_t=dT)
        mu_p = pet_h[:, lo:hi].mean()
        sd_p = pet_h[:, lo:hi].std() + 1e-8
        feat_d[:, :, names.index("pet")] = torch.as_tensor(
            (pet_d - mu_p) / sd_p, **dev_kw)
    # per-basin target-matched scaling (as load_hybrid_data does)
    feat_d[:, :, idx["sac_sim"]] = sim_d_t / h.scale[:, None]
    h_detr = dc.replace(h, feat=feat_d, sim=sim_d_t)

    def _full(data):
        bb, tt = data.eval_days("all")
        flow = predict_days(model, data, bb, tt).clamp_min(0.0).cpu().numpy()
        arr = np.full((len(data.basins), len(data.dates)), np.nan)
        arr[bb.cpu().numpy(), tt.cpu().numpy()] = flow
        return pd.DataFrame(arr.T, index=data.dates, columns=list(data.basins))

    flow_h = _full(h)
    flow_d = _full(h_detr)
    return flow_h, flow_d


def _ensemble_flow(data_dir, dT, dev, sim_detr: pd.DataFrame, ens_dir: str):
    """(flow_hist, flow_detr) ENSEMBLE-MEAN daily hybrid flow — mean over all
    ``seed*/checkpoints/best.pt`` members of :func:`_hybrid_flow`."""
    ckpts = sorted(Path(ens_dir).glob("seed*/checkpoints/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no seed*/checkpoints/best.pt under {ens_dir}")
    fhs, fds = [], []
    for cp in ckpts:
        fh, fd = _hybrid_flow(data_dir, dT, dev, sim_detr, ckpt=str(cp))
        fhs.append(fh)
        fds.append(fd)
    n = len(ckpts)
    return sum(fhs) / n, sum(fds) / n


def assemble(data_dir: str = "data", *, device: str = "cuda") -> dict:
    """dQ (detrended - historical) daily mm/day per model, on the covered basins."""
    from ..calsim.catchments import basin_areas
    from ..cdec15 import BASINS
    from ..io import load_hru_table
    from .config import pick_device

    areas = basin_areas(data_dir, domain="15cdec")
    f_hist, f_detr, dT = _forcings(data_dir)
    cd = Path("artifacts/dpl/_climatology_cache")
    # basins covered by WGEN: dT non-zero for >=50% of their cells
    hru = load_hru_table(data_dir, domain=DOMAIN)
    covered = {_norm_key(k) for k, r in f_hist.pos.items() if np.any(dT[r] != 0.0)}
    frac = hru.assign(c=hru["key"].map(lambda k: _norm_key(k) in covered)
                      ).groupby("basin")["c"].mean()
    basins = [b for b in frac.index if frac[b] >= 0.5]

    # pure-physics frozen sims over ALL 15 basins (Tulare has dT=0 ->
    # detrended==historical), cached; dQ subsets to the covered basins.
    dq: dict[str, pd.DataFrame] = {}
    for label, spec in MODELS.items():
        tag = label.split()[-1]
        h = _frozen_sim(f_hist, spec, BASINS, cd / f"fs_{tag}_hist.csv")
        d = _frozen_sim(f_detr, spec, BASINS, cd / f"fs_{tag}_detr.csv")
        dq[label] = (d - h)[basins]
        print(f"  assembled {label}", flush=True)

    # the ensembles' detrended sac_sim channel: noah's TORCH run under the dT
    # field (numerics-matched to the training channel; cached).  Not displayed —
    # the dPL noah series above is the frozen run_basin pair.
    nt_detr_csv = cd / "fs_noah_torch_detr.csv"
    if nt_detr_csv.exists():
        nt_detr = pd.read_csv(nt_detr_csv, parse_dates=["date"]).set_index("date")
    else:
        from .evaluate import noah_torch_daily
        nt_detr = noah_torch_daily(NOAH_CKPT, data_dir=data_dir,
                                   temp_delta=dT)
        nt_detr_csv.parent.mkdir(parents=True, exist_ok=True)
        nt_detr.to_csv(nt_detr_csv)
    print("  assembled the noah torch detrended channel", flush=True)

    # the canonical Hybrid ENSEMBLES (mean over seed members) on the noah
    # detrended physics baseline
    dev = pick_device(device)
    for label, ens in ENSEMBLES.items():
        fh, fd = _ensemble_flow(data_dir, dT, dev, nt_detr, ens)
        dq[label] = (fd - fh)[basins]
        print(f"  assembled {label}", flush=True)

    # order north->south (reuse climatology ordering, restricted to covered)
    from .climatology import _basin_order
    order = _basin_order(data_dir, basins)
    return dict(dq=dq, order=order, areas=areas)


def _agg_monthly_taf(daily_dq: pd.DataFrame, areas, basins) -> pd.Series:
    """Basin-summed monthly TAF of a signed daily-mm/day dQ frame."""
    m = _monthly_taf(daily_dq[basins], areas)          # date x basin, signed TAF
    return m.sum(axis=1)


def _plot_rolling(data: dict, path: Path, window: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order, areas = data["order"], data["areas"]
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for label, dqf in data["dq"].items():
        m = _agg_monthly_taf(dqf, areas, order)        # monthly aggregate TAF
        wyear = m.index.year + (m.index.month >= 10)   # water-year total
        annual = m.groupby(wyear).sum()
        roll = annual.rolling(window, center=True, min_periods=window // 2).mean()
        ax.plot(roll.index, roll.values, label=label, **STYLE[label])
    ax.axhline(0, color="#888888", lw=0.8, zorder=1)
    ax.set_xlabel("water year")
    ax.set_ylabel(f"aggregate ΔQ  (TAF/yr, {window}-yr rolling mean)")
    ax.set_title("Temperature-detrending sensitivity over time — "
                 "detrended − historical, summed over 11 basins",
                 fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _plot_monthly(data: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order, areas = data["order"], data["areas"]
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for label, dqf in data["dq"].items():
        m = _agg_monthly_taf(dqf[dqf.index < pd.Timestamp(_PRE1950)], areas, order)
        reg = m.groupby(m.index.month).mean().reindex(_WY)
        st = STYLE[label]
        ax.plot(range(12), reg.values, marker=_ROLE_MARKER[st["ls"]], ms=5,
                label=label, **st)
    ax.axhline(0, color="#888888", lw=0.8, zorder=1)
    ax.set_xticks(range(12))
    ax.set_xticklabels(_WY_LABELS)
    ax.set_xlabel("water-year month (Oct → Sep)")
    ax.set_ylabel("aggregate ΔQ  (TAF/month)")
    ax.set_title("Monthly detrending sensitivity, pre-1950 — "
                 "detrended − historical, summed over 11 basins",
                 fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def make_forcing_sensitivity(data_dir: str = "data",
                             out_dir: str | Path = "artifacts/dpl",
                             *, device: str = "cuda") -> dict:
    data = assemble(data_dir, device=device)
    figdir = Path(out_dir) / "figures"
    _plot_rolling(data, figdir / "cdec15_forcing_sensitivity_rolling.png")
    _plot_monthly(data, figdir / "cdec15_forcing_sensitivity_monthly_pre1950.png")
    return data


if __name__ == "__main__":
    make_forcing_sensitivity()
