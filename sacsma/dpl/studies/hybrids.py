"""(Δprecip, ΔT) response surfaces + skill for the hybrid family.

The physics is the ladder's ``5_px``; the three LSTM ensembles are built on it:

  * ``5_px``      — the physics itself (the reference response);
  * ``hybrid``    — the LSTM with 5_px's daily flow as an input channel, NO response loss;
  * ``hybrid_dt`` — ``hybrid`` fine-tuned with the 14-anchor {−20%,−10%,0,+10%,+20%}×{0,+2,
    +4 °C} response-consistency loss against 5_px's response (λ=0.18);
  * ``lstm``      — no physics channel (``use_sim=False``) — the no-physics ablation.
All three take 5_px's physical inputs and the dPL's four climate indices as statics, the
indices recomputed under each climate.

Per-watershed figure: 4 metrics × 4 model columns, % change vs each model's own
present climate.  The physics daily flow is the ensembles' physics run as trained
(:func:`sacsma.dpl.evaluate.basin_daily`); the hybrids are forwarded on the perturbed features
(:func:`_ensemble_perturbed_daily`, with the climate statics co-varying).  Ensembles = mean
over the seed members.  The grid, the metrics and the regimes are those of
:mod:`.surfaces`.  Also writes ``compare_ga_dpl_hybrid.csv``, the GA, dPL and hybrid scores
side by side (:func:`sacsma.dpl.hybrid.evaluate.compare_all`).
"""
from __future__ import annotations

import dataclasses as dc
from pathlib import Path

import numpy as np
import pandas as pd

from ... import paths
from ..._figures import _CAL_COLOR, _VAL_COLOR, plt
from ...io import load_basin_area
from ..hybrid.train import RESPONSE_ANCHORS as ANCHORS
from .climatology import STYLE as CLIM_STYLE
from .climatology import _basin_order
from .surfaces import (
    DP,
    DT,
    METRICS,
    REGIME_TITLE,
    REGIMES,
    aggregate_regime,
    metrics_from_daily,
)

# the runs the figures compare; `hybrid` and `hybrid_dt` read 5_px's daily flow, `lstm`
# has no physics channel at all (use_sim=False)
BASE_DIR = paths.dpl_run(run="hybrid")
DTDP_DIR = paths.dpl_run(run="hybrid_dt")
LSTM_DIR = paths.dpl_run(run="lstm")
N_SEEDS = 3

#: figure-facing labels; PHYSICS/BASE/DTDP/LSTM are the dict keys used
#: throughout this module.
PHYSICS = "5_px"
BASE = "Hybrid"
DTDP = "Hybrid DT"
LSTM = "LSTM"
COL_ORDER = [PHYSICS, BASE, DTDP, LSTM]
ENSEMBLES = {BASE: BASE_DIR, DTDP: DTDP_DIR, LSTM: LSTM_DIR}
#: the run folder name of each figure label (for its score table)
RUN = {PHYSICS: "5_px", BASE: "hybrid", DTDP: "hybrid_dt", LSTM: "lstm"}
#: untracked per-(dp,dt) ensemble daily-flow cache, one folder per ensemble and checkpoint
#: content -- lets a metric-only change (e.g. a different percentile) reduce on CPU instead of
#: re-running the GPU sweep.
_HYBRID_CACHE = paths.local(name="cache/hybrid_daily")
_ENS_TAG = {BASE: "base", DTDP: "dtdp", LSTM: "lstm"}


def _out(out_dir: str | Path | None) -> Path:
    """The study's result folder, unless ``out_dir`` names another."""
    return Path(out_dir) if out_dir is not None else paths.dpl_study(name="hybrids")


def _hybrid_cache_path(folder: Path, dp: float, dt: float) -> Path:
    dp, dt = dp + 0.0, dt + 0.0                   # normalize IEEE -0.0 (arange)
    return folder / f"dp{dp:+.2f}_dt{dt:+.1f}.csv"


def _load_ensemble(ens_dir: str | Path, data_dir: str, dev, n_seeds: int):
    """Load the shared hybrid data + up to ``n_seeds`` member models once (every
    seed shares the physics/domain/normalization config)."""
    import torch

    from ..hybrid.data import data_for
    from ..hybrid.evaluate import _build_model

    ckpts = sorted(Path(ens_dir).glob("seed*/checkpoints/best.pt"))[:n_seeds]
    if not ckpts:
        raise FileNotFoundError(f"no seed*/checkpoints/best.pt under {ens_dir}")
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    h = data_for(ck0, data_dir, dev)
    models = []
    for cp in ckpts:
        models.append(_build_model(torch.load(cp, map_location="cpu", weights_only=False),
                                   h, dev))
    return h, models


def _ensemble_perturbed_daily(h, models, dp: float, dt: float,
                              sim_pert_df: pd.DataFrame) -> pd.DataFrame:
    """Ensemble-mean daily hybrid flow (date x basin, mm/day) under (dp, dt).

    Builds the perturbed feature tensor with the SAME recipe the training loss
    uses (``apply_response_perturbation`` on the stored normalization) and the
    physics sim channel re-fed from ``sim_pert_df`` (the physics run under (dp, dt)),
    then averages the seed members."""
    import torch

    from ..hybrid.data import apply_response_perturbation, perturbed_static
    from ..hybrid.train import predict_days

    base_feat = h.feat.cpu().numpy()
    prcp = h.prcp.cpu().numpy()
    scale = h.scale.cpu().numpy()
    sim_pert = np.vstack([sim_pert_df[b].reindex(h.dates).to_numpy(np.float64)
                          for b in h.basins])
    feat_p = apply_response_perturbation(
        base_feat, h.names, dp=float(dp), dt=float(dt), prcp_raw=prcp,
        norm=h.norm, sim_pert=sim_pert, scale=scale)
    feat_t = torch.as_tensor(feat_p).to(dtype=h.feat.dtype, device=h.device)
    sim_t = torch.as_tensor(sim_pert).to(dtype=h.feat.dtype, device=h.device)
    sp = perturbed_static(h.static_ing, float(dp), float(dt))   # climate statics co-vary
    hp = dc.replace(h, feat=feat_t, sim=sim_t,
                    static=torch.as_tensor(sp).to(dtype=h.static.dtype, device=h.device))

    bb, tt = hp.eval_days("all")
    accum = None
    for mdl in models:
        f = predict_days(mdl, hp, bb, tt).clamp_min(0.0).cpu().numpy()
        accum = f if accum is None else accum + f
    flow = accum / len(models)
    arr = np.full((len(hp.basins), len(hp.dates)), np.nan)
    arr[bb.cpu().numpy(), tt.cpu().numpy()] = flow
    return pd.DataFrame(arr.T, index=hp.dates, columns=list(hp.basins))


def assemble(data_dir: str = "data", *, device: str = "cuda",
             n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """Long metrics table: one row per (basin, model, dp, dt) with the 4 raw
    metrics + signed % change vs that model's own (0, 0) baseline.

    Each ensemble's perturbed daily flow per (dp,dt) is cached to
    :data:`_HYBRID_CACHE`, so a metric-only change (different percentile, new
    reduction) reduces from the cache on CPU in seconds instead of re-running the
    ~700 full-record GPU forwards.  The heavy objects (the hybrid data, seed
    models, device) are lazy-loaded ONLY on the first cache miss, so a fully
    cached re-run never touches the GPU.  The cache follows a retrain and a change to the
    engine (:func:`sacsma.dpl.evaluate.ensemble_key`)."""
    areas = load_basin_area(data_dir, domain="15cdec").set_index(
        "basin")["area_mi2"].to_dict()
    grid = [(float(dp), float(dt)) for dp in DP for dt in DT]

    # the ensembles' physics over the grid (cached by basin_daily) — the reference AND the
    # hybrids' perturbed physics channel
    import torch

    from ..evaluate import basin_daily

    first = sorted(Path(BASE_DIR).glob("seed*/checkpoints/best.pt"))[0]
    physics = torch.load(first, map_location="cpu", weights_only=False)["cfg"]["physics"]
    phys = {(dp, dt): basin_daily(physics, data_dir=data_dir, dp=dp, dt=dt) for dp, dt in grid}
    print(f"  physics: {len(grid)} (dp,dt) points", flush=True)

    # lazy holders — built only when a hybrid cache MISS actually needs them.
    _lazy: dict = {}

    def _dev():
        if "dev" not in _lazy:
            from ..config import pick_device
            _lazy["dev"] = pick_device(device)
        return _lazy["dev"]

    def _ens(label, ens_dir):
        if label not in _lazy:
            _lazy[label] = _load_ensemble(ens_dir, data_dir, _dev(), n_seeds)
        return _lazy[label]

    rows = []

    def _emit(model, dp, dt, met):
        for b in met.index:
            rows.append(dict(basin=b, model=model, dp=round(dp, 4),
                             dt=round(dt, 4), **{k: float(met.loc[b, k])
                                                 for k in ("annual", "freshet",
                                                           "q999", "q30")}))

    for dp, dt in grid:
        _emit(PHYSICS, dp, dt, metrics_from_daily(phys[(dp, dt)], areas))

    from ..evaluate import ensemble_key

    for label, ens in ENSEMBLES.items():
        folder, n_miss = _HYBRID_CACHE / f"{_ENS_TAG[label]}_{ensemble_key(ens)}", 0
        folder.mkdir(parents=True, exist_ok=True)
        for dp, dt in grid:
            cache = _hybrid_cache_path(folder, dp, dt)
            if cache.exists():
                daily = pd.read_csv(cache, index_col=0, parse_dates=[0])
            else:
                h, models = _ens(label, ens)
                daily = _ensemble_perturbed_daily(h, models, dp, dt, phys[(dp, dt)])
                daily.index.name = "date"
                daily.round(6).to_csv(cache)
                n_miss += 1
            _emit(label, dp, dt, metrics_from_daily(daily, areas))
        print(f"  {label}: {len(grid)} points ({n_miss} computed, "
              f"{len(grid) - n_miss} cached)", flush=True)

    tbl = pd.DataFrame(rows)
    base = tbl[(tbl.dp == 0.0) & (tbl.dt == 0.0)].set_index(["basin", "model"])
    for k, _ in METRICS:
        b0 = tbl.set_index(["basin", "model"]).index.map(base[k])
        tbl[f"pct_{k}"] = 100.0 * (tbl[k].to_numpy() / np.asarray(b0, float) - 1.0)
    return tbl


def _plot_surfaces(sub: pd.DataFrame, out: Path, title: str) -> None:
    """The four metrics (rows) of the four models (columns) as % change over the (Δprecip, ΔT)
    grid, one colour scale per metric, with the present climate and the response-loss anchors
    marked."""
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    X, Y = np.meshgrid(DP * 100.0, DT)
    adp, adt = np.array(ANCHORS, float).T
    fig, axes = plt.subplots(len(METRICS), len(COL_ORDER), figsize=(6.5, 8.2),
                             sharex=True, sharey=True, layout="constrained")
    for r, (mkey, mlab) in enumerate(METRICS):
        surf = {}
        for mdl in COL_ORDER:
            s = sub[sub.model == mdl]
            surf[mdl] = (s.pivot(index="dt", columns="dp", values=f"pct_{mkey}")
                         .reindex(index=DT, columns=DP).to_numpy())
        allv = np.concatenate([z.ravel() for z in surf.values()])
        vmax = max(float(np.nanpercentile(np.abs(allv), 98)), 1.0)
        levels = MaxNLocator(nbins=12, symmetric=True).tick_values(-vmax, vmax)
        vlim = float(max(abs(levels[0]), abs(levels[-1])))
        norm = Normalize(vmin=-vlim, vmax=vlim)
        cf = None
        for c, mdl in enumerate(COL_ORDER):
            ax = axes[r, c]
            cf = ax.contourf(X, Y, surf[mdl], levels=levels, cmap="RdBu",
                             norm=norm, extend="both")
            ax.contour(X, Y, surf[mdl], levels=levels, colors="0.35",
                       linewidths=0.25)
            ax.plot(adp * 100.0, adt, "x", ms=3.0, mew=0.8, color="0.15", zorder=5,
                    clip_on=False)
            ax.plot(0.0, 0.0, "o", ms=3.0, mfc="w", mec="0.1", mew=0.8, zorder=6,
                    clip_on=False)
            if r == 0:
                ax.set_title(RUN[mdl])
        cb = fig.colorbar(cf, ax=list(axes[r, :]), fraction=0.04, pad=0.02,
                          ticks=levels[::2])
        cb.set_label(f"{mlab} (%)")
    axes[0, 0].set_xticks([-10, 0, 10])
    axes[0, 0].set_yticks([0, 2, 4])
    fig.supxlabel("Precipitation change (%)")
    fig.supylabel("Temperature change (°C)")
    fig.suptitle(title, x=0.02, ha="left")
    marks = [Line2D([], [], ls="none", marker="o", ms=4, mfc="w", mec="0.1",
                    label="Present climate"),
             Line2D([], [], ls="none", marker="x", ms=4, mew=0.8, color="0.15",
                    label="Response-loss anchor")]
    fig.legend(handles=marks, loc="outside upper right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# skill + response summary and the regime-aggregate surfaces
# --------------------------------------------------------------------------- #
#: the line and marker of each model in the summary and progression figures: the physics
#: (5_px) is the black reference, the ensembles take their colours of the climatology study
_STY = {PHYSICS: dict(color="k", marker="o", ms=4.0, lw=1.6, zorder=5),
        **{lab: dict(color=CLIM_STYLE[RUN[lab]]["color"], marker=mk, ms=4.0, lw=1.2)
           for lab, mk in ((BASE, "s"), (DTDP, "^"), (LSTM, "v"))}}


def _mean_calval(csv: str | Path) -> tuple[float, float]:
    """(mean cal_kge, mean val_kge) of a run's ``metrics.csv``."""
    m = pd.read_csv(csv)
    return float(m["cal_kge"].mean()), float(m["val_kge"].mean())


def _skill_pairs() -> dict[str, tuple[float, float]]:
    """Ensemble-mean cal/val KGE per model, from the tracked metrics CSVs."""
    return {lab: _mean_calval(paths.dpl_metrics(run=RUN[lab])) for lab in (PHYSICS, *ENSEMBLES)}


def _pooled(tbl: pd.DataFrame, model: str, col: str, dp: float, dt: float):
    """Per-basin ``col`` for one model at (dp, dt) (index=basin)."""
    s = tbl[(tbl.model == model) & np.isclose(tbl.dp, dp) & np.isclose(tbl.dt, dt)]
    return s.set_index("basin")[col]


def _warming_curve(tbl: pd.DataFrame, model: str) -> list[float]:
    """The mean over the basins of the % change of annual runoff along ΔT, at Δprecip = 0."""
    return [float(_pooled(tbl, model, "pct_annual", 0.0, float(dt)).mean()) for dt in DT]


def make_summary(tbl: pd.DataFrame, out_dir: str | Path | None = None,
                 ) -> Path:
    """Skill and response of 5_px and the three ensembles: (1) the mean calibration and
    validation KGE over the basins (from the metrics CSVs), (2) the warming curve (the mean
    over the basins of the % change of annual runoff along ΔT, at Δprecip = 0), (3) the
    precipitation curve (the same along Δprecip at ΔT = +2 °C, against the +2 °C climate)."""
    skill = _skill_pairs()
    precip_curve = {}
    for m in COL_ORDER:
        a2 = _pooled(tbl, m, "annual", 0.0, 2.0)
        precip_curve[m] = [float((_pooled(tbl, m, "annual", float(dp), 2.0) / a2
                                  - 1.0).mean() * 100) for dp in DP]

    fig = plt.figure(figsize=(6.5, 5.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
    axk = fig.add_subplot(gs[0, :])
    axt = fig.add_subplot(gs[1, 0])
    axp = fig.add_subplot(gs[1, 1])

    x = np.arange(len(COL_ORDER))
    bw = 0.38
    axk.bar(x - bw / 2, [skill[m][0] for m in COL_ORDER], bw, label="Calibration",
            color=_CAL_COLOR)
    axk.bar(x + bw / 2, [skill[m][1] for m in COL_ORDER], bw, label="Validation",
            color=_VAL_COLOR)
    axk.set_ylim(0.0, 1.0)
    axk.set_ylabel("Mean KGE")
    axk.set_xticks(x)
    axk.set_xticklabels([RUN[m] for m in COL_ORDER])
    axk.set_title("Skill")
    axk.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    axt.axhline(0, color="0.6", lw=0.8)
    for m in COL_ORDER:
        axt.plot(DT, _warming_curve(tbl, m), label=RUN[m], **_STY[m])
    axt.set_xlabel("Temperature change (°C)")
    axt.set_ylabel("Annual runoff change (%)")
    axt.set_title("Warming")
    fig.legend(*axt.get_legend_handles_labels(), loc="outside lower center", ncol=len(COL_ORDER))

    axp.axhline(0, color="0.6", lw=0.8)
    for m in COL_ORDER:
        axp.plot(DP * 100.0, precip_curve[m], **_STY[m])
    axp.set_xlabel("Precipitation change (%)")
    axp.set_ylabel("Annual runoff change (%)")
    axp.set_title("Precipitation at +2 °C")

    out = _out(out_dir) / "hybrid_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def make_regime_surfaces(tbl: pd.DataFrame, data_dir: str = "data",
                         out_dir: str | Path | None = None) -> None:
    """One 4×4 response-surface figure per hydroclimate regime (:data:`REGIMES`),
    the group's basins pooled by area-weighted % change."""
    areas = load_basin_area(data_dir, domain="15cdec").set_index(
        "basin")["area_mi2"].to_dict()
    figdir = _out(out_dir) / "hybrid_regimes"
    for reg, basins in REGIMES.items():
        agg = aggregate_regime(tbl, basins, areas)
        _plot_surfaces(agg, figdir / f"{reg}.png",
                       f"{REGIME_TITLE[reg]}: {', '.join(basins)}")
    print(f"wrote {len(REGIMES)} regime figures -> {figdir}", flush=True)


def make_hybrid_progression(tbl: pd.DataFrame, data_dir: str = "data",
                            out_dir: str | Path | None = None) -> Path:
    """Two-panel progression exhibit for the chain 5_px (physics) -> Hybrid ->
    Hybrid DT: (a) per-basin validation skill, (b) the pooled warming-response
    curve."""
    MODELS = [PHYSICS, BASE, DTDP]

    order = _basin_order(data_dir, sorted(tbl["basin"].unique()))
    val = {lab: pd.read_csv(paths.dpl_metrics(run=RUN[lab])).set_index("basin")["val_kge"]
           for lab in MODELS}
    temp_curve = {m: _warming_curve(tbl, m) for m in MODELS}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.5, 4.8), layout="constrained")
    y = np.arange(len(order))
    for m in MODELS:
        v = val[m].reindex(order).to_numpy(float)
        sty = {**_STY[m], "ms": 5.0}
        axa.plot(np.clip(v, 0.0, None), y, ls="none", label=RUN[m], **sty)
        neg = np.isfinite(v) & (v < 0)
        if neg.any():
            axa.plot(np.zeros(neg.sum()), y[neg], "<", ms=5, color=sty["color"], ls="none",
                     clip_on=False)
    axa.invert_yaxis()
    axa.set_yticks(y)
    axa.set_yticklabels(order)
    axa.set_xlim(0, 1)
    axa.set_xlabel("Validation KGE (WY2004–2018)")
    axa.set_title("Skill")
    axa.legend(loc="center left")

    axb.axhline(0, color="0.6", lw=0.8)
    for m in MODELS:
        axb.plot(DT, temp_curve[m], label=RUN[m], **_STY[m])
    axb.set_xlabel("Temperature change (°C)")
    axb.set_ylabel("Annual runoff change (%)")
    axb.set_title("Warming")
    axb.legend(loc="lower left")

    out = _out(out_dir) / "hybrid_progression.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)

    rows = []
    for m in MODELS:
        row = {"model": m, "val_kge_mean": round(float(val[m].reindex(order).mean()), 4)}
        for dt, v in zip(DT.tolist(), temp_curve[m], strict=True):
            row[f"pct_annual_dt{dt:g}"] = round(v, 4)
        rows.append(row)
    csv = _out(out_dir) / "hybrid_progression.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {out}", flush=True)
    return out


def make_hybrids(data_dir: str = "data",
                 out_dir: str | Path | None = None,
                 *, device: str = "cuda", n_seeds: int = N_SEEDS,
                 regen: bool = False) -> pd.DataFrame:
    """Assemble (or reload) the metrics table, then render: one 4×4 response
    surface per watershed, one per hydroclimate regime, the 3-panel skill /
    response summary, the progression exhibit, and the GA / dPL / hybrid score table."""
    from ..hybrid.evaluate import compare_all

    out_dir = _out(out_dir)
    csv = out_dir / "hybrids_metrics.csv"
    if csv.exists() and not regen:
        tbl = pd.read_csv(csv)
        print(f"loaded {csv}", flush=True)
    else:
        tbl = assemble(data_dir, device=device, n_seeds=n_seeds)
        csv.parent.mkdir(parents=True, exist_ok=True)
        tbl.round(4).to_csv(csv, index=False)
        print(f"wrote {csv}", flush=True)

    order = _basin_order(data_dir, sorted(tbl["basin"].unique()))
    figdir = out_dir / "basins"
    for b in order:
        _plot_surfaces(tbl[tbl.basin == b], figdir / f"{b}.png", b)
    print(f"wrote {len(order)} figures -> {figdir}", flush=True)
    make_regime_surfaces(tbl, data_dir, out_dir)
    make_summary(tbl, out_dir)
    make_hybrid_progression(tbl, data_dir, out_dir)
    compare_all(out_dir)
    return tbl
