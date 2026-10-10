"""Temperature-detrending sensitivity (WGEN Product A minus historical Livneh)
for the ladder's physics rungs 2_grid (Hamon), 3_pt (Priestley-Taylor), 4_noah (Noah-type ET)
and 5_px (learned rain/snow threshold), and the three LSTM ensembles on 5_px (mean over seed
members): hybrid, hybrid_dt and lstm.

The physics runs run as trained under the per-cell dT field on their own domain's rows
(:func:`sacsma.dpl.evaluate.basin_daily`); the ensembles' physics channel is 5_px's detrended
series.  The WGEN detrending never enters the hybrids' training, so the ensembles here are an
INDEPENDENT check of their temperature response.  Their statics stay the present climate's:
the climate indices are computed over WY1989-2003, near WGEN's 1991-2020 baseline, where the
detrending is small.

WGEN Product A detrends temperature to a 1991-2020 baseline (the early record is
warmed).  It is not packaged for the 15cdec application, so the detrending signal
is reconstructed as a temperature-level shift::

    dT(cell, day) = wgen_tavg - historical_tavg      (from the CalSim WGEN stores,
                                                       key-matched to 15cdec_grid;
                                                       0 for uncovered cells)

applied to the temperatures (precip is identical between the products).  Only the
11 rim basins whose grid cells are covered by the CalSim WGEN stores are shown
(92-100% cell coverage); the 4 Tulare basins have no WGEN and are dropped.

Two figures on the basin-aggregated flow change dQ = detrended - historical:
  * rolling (water-year) aggregate dQ over time (the detrending effect shrinks
    toward zero as the record approaches the 1991-2020 baseline);
  * the monthly dQ regime over the pre-1950 period (largest detrending effect).

Output: ``artifacts/results/dpl/studies/forcing/forcing_sensitivity_*.png``.
"""

from __future__ import annotations

import dataclasses as dc
from pathlib import Path

import numpy as np
import pandas as pd

from ... import paths
from ...io import load_forcing
from ...model import load_domain_forcing
from .climatology import _WY, _WY_LABELS, _monthly_taf

DOMAIN = "15cdec_grid"
_PRE1950 = "1950-01-01"
#: the physics runs: label -> the ladder run
RUNS: dict[str, str] = {"2_grid": "2_grid", "3_pt": "3_pt", "4_noah": "4_noah", "5_px": "5_px"}
#: the LSTM ensembles (mean over seed members); hybrid and hybrid_dt read 5_px's detrended
#: flow as their physics channel, lstm has none
ENSEMBLES: dict[str, Path] = {
    "hybrid":    paths.dpl_run(run="hybrid"),
    "hybrid_dt": paths.dpl_run(run="hybrid_dt"),
    "lstm":      paths.dpl_run(run="lstm"),
}
#: COLOR = the physics (blue Hamon, red PT, olive Noah-type ET, green the learned threshold;
#: the ensembles on 5_px green, lstm gray); LINESTYLE = role (solid physics, broken LSTM)
STYLE: dict[str, dict] = {
    "2_grid":    dict(color="#1f77b4", lw=2.3, ls="-"),
    "3_pt":      dict(color="#d62728", lw=2.3, ls="-"),
    "4_noah":    dict(color="#bcbd22", lw=2.3, ls="-"),
    "5_px":      dict(color="#2ca02c", lw=2.3, ls="-"),
    "hybrid":    dict(color="#2ca02c", lw=2.0, ls="--"),
    "hybrid_dt": dict(color="#2ca02c", lw=2.0, ls="-."),
    "lstm":      dict(color="#7f7f7f", lw=2.0, ls=":"),
}
#: marker by role (reinforces the linestyle on the monthly plot only; the rolling
#: time series stays marker-free).
_ROLE_MARKER = {"-": "o", "--": "s", "-.": "^", ":": "D"}


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


def _hybrid_flow(data_dir, dT, dev, sim_detr: pd.DataFrame | None, ckpt: str):
    """(flow_hist, flow_detr) full-record daily hybrid flow (date x basin).

    Detrended features = historical features + dT_basin/sigma on the temperature channels
    (additive under z-score, with the trained normalisation ``h.norm``), and the physics
    channel, if any, re-fed by the detrended physics flow (per-basin ÷scale, as
    load_hybrid_data does).  ``dT`` is the field on the rows of the hybrid's forcing domain;
    ``ckpt`` is one trained seed member's checkpoint."""
    import torch

    from ..hybrid.data import data_for
    from ..hybrid.evaluate import _build_model
    from ..hybrid.train import predict_days

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    h = data_for(ck, data_dir, dev)
    model = _build_model(ck, h, dev)

    # basin-level dT on the hybrid's forcing domain
    from ..data import load_domain_tensors
    dom = load_domain_tensors(data_dir, domain=DOMAIN, device="cpu", dtype=torch.float64)
    dTb = dom.W.numpy() @ dT[dom.cell_idx].astype(np.float64)       # (B, T) basin dT
    idx = {n: i for i, n in enumerate(h.names)}
    feat_d = h.feat.clone()
    dev_kw = dict(dtype=feat_d.dtype, device=feat_d.device)
    dTb_t = torch.as_tensor(dTb, **dev_kw)
    for n in ("tmin", "tmax"):
        feat_d[:, :, idx[n]] += dTb_t / h.norm[n][1]
    rep = dict(feat=feat_d)
    if "sac_sim" in idx:
        sim_d = np.vstack([sim_detr[b].reindex(dom.dates).to_numpy() for b in dom.basins])
        sim_d_t = torch.as_tensor(sim_d, **dev_kw)
        feat_d[:, :, idx["sac_sim"]] = sim_d_t / h.scale[:, None]
        rep["sim"] = sim_d_t
    h_detr = dc.replace(h, **rep)

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
    from ...calsim.catchments import basin_areas
    from ...io import load_hru_table
    from ..config import pick_device

    areas = basin_areas(data_dir, domain="15cdec")
    f_hist = load_domain_forcing(data_dir, domain=DOMAIN)
    dT = _delta_t(data_dir, f_hist)
    # basins covered by WGEN: dT non-zero for >=50% of their cells
    hru = load_hru_table(data_dir, domain=DOMAIN)
    covered = {_norm_key(k) for k, r in f_hist.pos.items() if np.any(dT[r] != 0.0)}
    frac = hru.assign(c=hru["key"].map(lambda k: _norm_key(k) in covered)
                      ).groupby("basin")["c"].mean()
    basins = [b for b in frac.index if frac[b] >= 0.5]

    # the physics runs over all 15 basins (Tulare has dT = 0); dQ keeps the covered basins.
    # Each run takes dT on its own domain's forcing rows.
    import torch

    from ..evaluate import _loaded, basin_daily

    def detrended(run):
        f = _loaded(str(paths.dpl_checkpoint(run=run).resolve()), str(data_dir))[2].forcing
        return basin_daily(run, data_dir=data_dir, dt=_delta_t(data_dir, f))

    dq: dict[str, pd.DataFrame] = {}
    for label, run in RUNS.items():
        dq[label] = (detrended(run) - basin_daily(run, data_dir=data_dir))[basins]
        print(f"  assembled {label}", flush=True)

    # the LSTM ensembles (mean over seed members); the physics channel's detrended flow is
    # that of the ensembles' physics run (5_px)
    dev = pick_device(device)
    for label, ens in ENSEMBLES.items():
        first = sorted(Path(ens).glob("seed*/checkpoints/best.pt"))[0]
        physics = torch.load(first, map_location="cpu", weights_only=False)["cfg"]["physics"]
        sim_detr = detrended(physics) if physics else None
        fh, fd = _ensemble_flow(data_dir, dT, dev, sim_detr, ens)
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
                             out_dir: str | Path | None = None,
                             *, device: str = "cuda") -> dict:
    """The two figures into ``out_dir`` (default: the study's result folder)."""
    data = assemble(data_dir, device=device)
    figdir = Path(out_dir) if out_dir is not None else paths.dpl_study(name="forcing")
    _plot_rolling(data, figdir / "forcing_sensitivity_rolling.png")
    _plot_monthly(data, figdir / "forcing_sensitivity_monthly_pre1950.png")
    return data
