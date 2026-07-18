"""(Δprecip, ΔT) climate-response surfaces for the noah physics and the PET
hybrid ensembles (raw vs dt/dp-trained), watershed by watershed.

For each point on a (Δprecip, ΔT) grid, daily basin flow is recomputed three
ways and reduced to a mean-monthly TAF regime:

  * PHYSICS (noah) — the noah climate response under (dp, dt): the fast frozen
    numba noah-lite core (precip ×(1+dp), tavg/tmin/tmax + dt) carried on the
    torch present-climate baseline (``physics_daily``, see below);
  * HYBRID RAW — the PET hybrid ensemble trained with NO response loss,
    forwarded on the perturbed features (temps + dt, precip ×(1+dp), PET
    recomputed under dt, sim channel = the SAME physics run under (dp, dt));
  * HYBRID dt·dp — the PET hybrid ensemble trained with the multi-anchor dp/dt
    response-consistency loss, same perturbed forward.

Four metrics per watershed, each reported as **% change vs the (0, 0) baseline**
(the grid's own origin point, so the comparison is self-consistent):
  total annual runoff, Apr–Jul freshet volume ("seasonal runoff"), max monthly
  flow, min monthly flow — all from the full-record mean-monthly regime (the
  per-basin area cancels in a ratio, so the % change is area-independent).

Output: one compact 4×3 (metric × model) grid of interpolated response surfaces
per watershed (``artifacts/dpl/figures/dtdp_response/<BASIN>.png``) and a tidy
long metrics table (``artifacts/dpl/dtdp_response_metrics.csv``).

An independent check of the response loss: the WGEN/scalar training anchors are
the 5 corners of {−dp,0,+dp}×{0,+dt}; the surfaces here sweep the whole plane,
so most grid points are held out of training.
"""

from __future__ import annotations

import dataclasses as dc
from pathlib import Path

import numpy as np
import pandas as pd

from .._figures import plt  # noqa: F401  (applies the house rcParams)
from ..io import load_basin_area
from .climatology import _WY, _basin_order, _monthly_taf
from .evaluate import teacher_cache_path

DOMAIN = "15cdec_grid"
NOAH_DPL_CSV = "artifacts/dpl/noah/params_dpl.csv"       # frozen noah-lite SAC params
NOAH_CANOPY_CSV = "artifacts/dpl/noah/params_canopy.csv"  # + soil_chi
BASE_TORCH_CSV = "artifacts/dpl/noah/daily_sim_noah_torch.csv"  # torch present-climate base
RAW_DIR = "artifacts/dpl/testing/hybrid_pet_noah"        # PET input, no response loss
DTDP_DIR = "artifacts/dpl/testing/hybrid_pet_dtdp"       # PET input + dp/dt response loss
N_SEEDS = 3

#: model columns, left → right.
PHYSICS = "Physics (noah)"
RAW = "Hybrid raw"
DTDP = "Hybrid dt·dp"
MODEL_ORDER = [PHYSICS, RAW, DTDP]

#: metric rows: (key, pretty label).
METRICS: list[tuple[str, str]] = [
    ("annual", "Total annual runoff"),
    ("freshet", "Apr–Jul freshet"),
    ("maxmon", "Max monthly flow"),
    ("minmon", "Min monthly flow"),
]

#: response-surface grid — nodes sit exactly on the ±10% / +3 °C training
#: anchors, and bracket them at ±20% / +4 °C.  contourf interpolates between the
#: nodes, so a 5×5 grid gives clean surfaces at a tractable per-point (full-record
#: torch stream) cost; densify (e.g. step 0.05 / 0.5) if smoother is wanted.
DP = np.round(np.arange(-0.20, 0.2001, 0.10), 4)         # Δprecip fraction (5)
DT = np.round(np.arange(0.0, 4.0001, 1.0), 4)            # ΔT degC (5)

#: the dt·dp model's response-loss training anchors — the 5 non-origin corners of
#: {−dp,0,+dp}×{0,+dt} — marked on the surfaces so the supervised grid points are
#: distinguishable from the held-out (interpolated / extrapolated) region.
ANCHOR_DP, ANCHOR_DT = 0.10, 3.0
TRAIN_ANCHORS = [(-ANCHOR_DP, 0.0), (ANCHOR_DP, 0.0), (0.0, ANCHOR_DT),
                 (-ANCHOR_DP, ANCHOR_DT), (ANCHOR_DP, ANCHOR_DT)]


# --------------------------------------------------------------------------- #
# physics under (dp, dt): the FROZEN numba noah-lite climate response, carried
# on the torch present-climate baseline
# --------------------------------------------------------------------------- #
# The torch Noah stream is ~14 min/run; the frozen numba noah-lite core is the
# SAME physics (~4 s/run) and its (dp, dt) RESPONSE matches torch to <~0.3% on
# annual runoff (verified vs the torch teachers).  So the physics response —
# teachers, the physics column, and the hybrids' perturbed sim channel — is the
# fast frozen response ADDED to the torch baseline the hybrids trained on:
#     physics(dp, dt) = base_torch + [frozen(dp, dt) − frozen(0, 0)]
# which is EXACTLY the torch baseline at (0, 0) (so the hybrids' present-climate
# flow is unperturbed) and carries the frozen climate response elsewhere.
_F0 = None  # in-process cache of the loaded baseline DomainForcing


def _frozen_noah(dp: float, dt: float, data_dir: str = "data") -> pd.DataFrame:
    """Frozen numba noah-lite daily basin flow (date x basin, mm/day) under
    (Δprecip fraction ``dp``, ΔT ``dt``).  Cached to disk; the loaded baseline
    forcing is reused across calls."""
    import dataclasses as _dc

    cache = teacher_cache_path(dp, dt).with_name(
        f"frozen_dp{dp:+.2f}_dt{dt:+.1f}.csv")
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")
    global _F0
    from ..cdec15 import BASINS
    from ..model import attach_tminmax, load_domain_forcing, run_basin
    if _F0 is None:
        f = load_domain_forcing(data_dir, domain=DOMAIN)
        attach_tminmax(data_dir, DOMAIN, f)
        _F0 = f
    fp = _dc.replace(_F0, prcp=_F0.prcp * (1.0 + dp), tavg=_F0.tavg + dt,
                     tmin=_F0.tmin + dt, tmax=_F0.tmax + dt, _f64={})
    params = pd.read_csv(NOAH_DPL_CSV)
    canopy = pd.read_csv(NOAH_CANOPY_CSV)
    cols = {}
    for b in BASINS:
        s = run_basin(b, data_dir=data_dir, domain=DOMAIN, forcing=fp,
                      params=params, parallel=True, pet_source="priestley_taylor",
                      pt_snow_albedo=0.0, pt_dewpoint_depression=0.0,
                      et_scheme="noah_lite", canopy_params=canopy)
        cols[b] = s.set_index("date")["flow"]
    df = pd.DataFrame(cols)
    df.index.name = "date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(cache)
    return df


def physics_daily(dp: float, dt: float, data_dir: str = "data") -> pd.DataFrame:
    """Physics daily flow under (dp, dt): ``base_torch + [frozen(dp,dt) −
    frozen(0,0)]``.  The SINGLE source of truth for the response teachers, the
    physics column, and the hybrids' perturbed sim channel — cached to the
    teacher path shared by training and the sweep."""
    cache = teacher_cache_path(dp, dt)
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")
    base = pd.read_csv(BASE_TORCH_CSV, parse_dates=["date"]).set_index("date")
    fz = _frozen_noah(dp, dt, data_dir).reindex(base.index)[base.columns]
    fz0 = _frozen_noah(0.0, 0.0, data_dir).reindex(base.index)[base.columns]
    out = base + (fz - fz0)
    out.index.name = "date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(cache)
    return out


# --------------------------------------------------------------------------- #
# daily flow under (dp, dt): physics + perturbed hybrid ensemble
# --------------------------------------------------------------------------- #
def _load_ensemble(ens_dir: str, data_dir: str, dev, n_seeds: int):
    """Load the shared hybrid data + up to ``n_seeds`` member models once (every
    seed shares the physics/domain/normalization config)."""
    import torch

    from .hybrid.data import feature_names, load_hybrid_data
    from .hybrid.model import HybridLSTM

    ckpts = sorted(Path(ens_dir).glob("seed*/checkpoints/best.pt"))[:n_seeds]
    if not ckpts:
        raise FileNotFoundError(f"no seed*/checkpoints/best.pt under {ens_dir}")
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    cfg = ck0["cfg"]
    if ck0.get("variant", "feature") != "feature":
        raise ValueError("residual hybrid checkpoints are retired (2026-07-16)")
    h = load_hybrid_data(
        data_dir, physics_csv=ck0.get("physics_csv"),
        sim_cache=ck0.get("sim_cache"), use_statics=bool(ck0["n_static"]),
        use_doy=cfg.get("use_doy", True), use_pet=cfg.get("use_pet", False),
        domain=cfg.get("physics_domain", "15cdec"),
        pet_source=cfg.get("pet_source", "hamon"),
        pt_snow_albedo=cfg.get("pt_snow_albedo", 0.0),
        pt_dewpoint_depression=cfg.get("pt_dewpoint_depression", 0.0),
        et_scheme=cfg.get("physics_et_scheme", "sac"),
        canopy_csv=cfg.get("canopy_csv") or None, device=dev)
    models = []
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        mdl = HybridLSTM(h.n_feat, h.n_static, hidden=cfg["hidden"],
                         static_embed=cfg["static_embed"],
                         dropout=cfg["dropout"]).to(dev)
        mdl.load_state_dict(ck["model"])
        models.append(mdl)
    names = feature_names(cfg.get("use_doy", True), cfg.get("use_pet", False))
    return h, models, cfg, names


def _ensemble_perturbed_daily(h, dom, models, cfg, names, dp: float, dt: float,
                              sim_pert_df: pd.DataFrame,
                              pet_pert: np.ndarray | None = None) -> pd.DataFrame:
    """Ensemble-mean daily hybrid flow (date x basin, mm/day) under (dp, dt).

    Builds the perturbed feature tensor with the SAME recipe the training loss
    uses (``apply_response_perturbation`` on the stored normalization) and the
    physics sim channel re-fed from ``sim_pert_df`` (the noah run under (dp, dt)),
    then averages the seed members.  ``pet_pert`` (the raw PT PET under ΔT, (B,T))
    may be precomputed and passed in (it depends only on dt); computed here if the
    model uses PET and none is given."""
    import torch

    from .hybrid.data import apply_response_perturbation, basin_pet_pt
    from .hybrid.train import predict_days

    use_pet = cfg.get("use_pet", False)
    base_feat = h.feat.cpu().numpy()
    prcp = h.prcp.cpu().numpy()
    scale = h.scale.cpu().numpy()
    sim_pert = np.vstack([sim_pert_df[b].reindex(h.dates).to_numpy(np.float64)
                          for b in h.basins])
    if use_pet and pet_pert is None:
        pet_pert = basin_pet_pt(dom, delta_t=float(dt))
    if not use_pet:
        pet_pert = None
    feat_p = apply_response_perturbation(
        base_feat, names, dp=float(dp), dt=float(dt), prcp_raw=prcp,
        norm=h.norm, pet_pert=pet_pert, sim_pert=sim_pert, scale=scale)
    feat_t = torch.as_tensor(feat_p).to(dtype=h.feat.dtype, device=h.device)
    sim_t = torch.as_tensor(sim_pert).to(dtype=h.feat.dtype, device=h.device)
    hp = dc.replace(h, feat=feat_t, sim=sim_t)

    bb, tt = hp.eval_days("all")
    accum = None
    for mdl in models:
        f = predict_days(mdl, hp, bb, tt).clamp_min(0.0).cpu().numpy()
        accum = f if accum is None else accum + f
    flow = accum / len(models)
    arr = np.full((len(hp.basins), len(hp.dates)), np.nan)
    arr[bb.cpu().numpy(), tt.cpu().numpy()] = flow
    return pd.DataFrame(arr.T, index=hp.dates, columns=list(hp.basins))


# --------------------------------------------------------------------------- #
# daily mm/day -> mean-monthly TAF regime -> the four metrics
# --------------------------------------------------------------------------- #
def _metrics_from_daily(daily: pd.DataFrame, areas: dict[str, float]) -> pd.DataFrame:
    """Per-basin (annual, freshet, maxmon, minmon) TAF from the full-record
    mean-monthly regime (index=basin, cols=the 4 metric keys)."""
    m = _monthly_taf(daily, areas)                       # date x basin monthly TAF
    out = {}
    for b in m.columns:
        s = m[b].dropna()
        reg = s.groupby(s.index.month).mean().reindex(_WY)  # 12-pt WY regime
        out[b] = dict(annual=float(reg.sum()),
                      freshet=float(reg.loc[[4, 5, 6, 7]].sum()),
                      maxmon=float(reg.max()),
                      minmon=float(reg.min()))
    return pd.DataFrame(out).T


def assemble(data_dir: str = "data", *, device: str = "cuda",
             n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """Long metrics table: one row per (basin, model, dp, dt) with the 4 raw
    metrics + their signed % change vs the (0, 0) baseline."""
    import torch

    from .config import pick_device
    from .data import load_domain_tensors

    dev = pick_device(device)
    areas = load_basin_area(data_dir, domain="15cdec").set_index(
        "basin")["area_mi2"].to_dict()
    dom = load_domain_tensors(data_dir, domain=DOMAIN, device="cpu",
                              dtype=torch.float64)
    grid = [(float(dp), float(dt)) for dp in DP for dt in DT]

    # physics under every (dp, dt) — cached daily sims (single source of truth,
    # also the hybrids' perturbed sim channel).
    phys: dict[tuple[float, float], pd.DataFrame] = {}
    for dp, dt in grid:
        phys[(dp, dt)] = physics_daily(dp, dt, data_dir)
    print(f"  physics: {len(grid)} (dp,dt) points", flush=True)

    rows = []

    def _emit(model, dp, dt, met):
        for b in met.index:
            rows.append(dict(basin=b, model=model, dp=round(dp, 4),
                             dt=round(dt, 4), **{k: float(met.loc[b, k])
                                                 for k in ("annual", "freshet",
                                                           "maxmon", "minmon")}))

    for dp, dt in grid:
        _emit(PHYSICS, dp, dt, _metrics_from_daily(phys[(dp, dt)], areas))

    # PET depends only on ΔT — precompute once per distinct dt and reuse.
    from .hybrid.data import basin_pet_pt
    pet_by_dt = {float(dt): basin_pet_pt(dom, delta_t=float(dt)) for dt in DT}

    for label, ens in [(RAW, RAW_DIR), (DTDP, DTDP_DIR)]:
        h, models, cfg, names = _load_ensemble(ens, data_dir, dev, n_seeds)
        for dp, dt in grid:
            daily = _ensemble_perturbed_daily(h, dom, models, cfg, names, dp, dt,
                                              phys[(dp, dt)],
                                              pet_pert=pet_by_dt[dt])
            _emit(label, dp, dt, _metrics_from_daily(daily, areas))
        print(f"  {label}: {len(models)} seeds x {len(grid)} points", flush=True)

    tbl = pd.DataFrame(rows)
    # signed % change vs the (0,0) baseline per (basin, model)
    base = tbl[(tbl.dp == 0.0) & (tbl.dt == 0.0)].set_index(["basin", "model"])
    for k, _ in METRICS:
        b0 = tbl.set_index(["basin", "model"]).index.map(base[k])
        tbl[f"pct_{k}"] = 100.0 * (tbl[k].to_numpy() / np.asarray(b0, float) - 1.0)
    return tbl


# --------------------------------------------------------------------------- #
# per-watershed 4x3 response-surface figure
# --------------------------------------------------------------------------- #
def _plot_basin(basin: str, sub: pd.DataFrame, out: Path) -> None:
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    X, Y = np.meshgrid(DP * 100.0, DT)                   # (len(DT), len(DP))
    fig, axes = plt.subplots(len(METRICS), len(MODEL_ORDER), figsize=(7.0, 8.4),
                             sharex=True, sharey=True, constrained_layout=True)
    for r, (mkey, mlab) in enumerate(METRICS):
        surf = {}
        for mdl in MODEL_ORDER:
            s = sub[sub.model == mdl]
            surf[mdl] = (s.pivot(index="dt", columns="dp", values=f"pct_{mkey}")
                         .reindex(index=DT, columns=DP).to_numpy())
        allv = np.concatenate([z.ravel() for z in surf.values()])
        vmax = max(float(np.nanpercentile(np.abs(allv), 98)), 1.0)
        # nice round, symmetric levels so the shared colorbar reads cleanly
        levels = MaxNLocator(nbins=12, symmetric=True).tick_values(-vmax, vmax)
        vlim = float(max(abs(levels[0]), abs(levels[-1])))
        norm = Normalize(vmin=-vlim, vmax=vlim)
        cf = None
        for c, mdl in enumerate(MODEL_ORDER):
            ax = axes[r, c]
            Z = surf[mdl]
            cf = ax.contourf(X, Y, Z, levels=levels, cmap="RdBu", norm=norm,
                             extend="both")
            ax.contour(X, Y, Z, levels=levels, colors="0.35", linewidths=0.25)
            # dt·dp response-loss training anchors (supervised grid points)
            for adp, adt in TRAIN_ANCHORS:
                ax.plot(adp * 100.0, adt, marker="x", ms=3.5, mew=0.8,
                        color="0.15", zorder=5, clip_on=False)
            ax.plot(0.0, 0.0, marker="o", ms=3.0, mfc="w", mec="0.1", mew=0.8,
                    zorder=6, clip_on=False)
            if r == 0:
                ax.set_title(mdl, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{mlab}\nΔT (°C)", fontsize=7.5)
            if r == len(METRICS) - 1:
                ax.set_xlabel("Δprecip (%)", fontsize=7.5)
            ax.set_xticks([-20, -10, 0, 10, 20])
            ax.set_yticks([0, 1, 2, 3, 4])
            ax.tick_params(labelsize=6.5)
        cb = fig.colorbar(cf, ax=list(axes[r, :]), fraction=0.05, pad=0.01,
                          ticks=levels[::2])
        cb.set_label("% change", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle(
        f"{basin} — climate-response surfaces  (% change vs present climate)\n"
        "○ present climate      ×  dt·dp response-loss training anchors",
        fontsize=8.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def make_dtdp_response(data_dir: str = "data",
                       out_dir: str | Path = "artifacts/dpl",
                       *, device: str = "cuda", n_seeds: int = N_SEEDS,
                       regen: bool = False) -> pd.DataFrame:
    """Assemble (or reload) the metrics table, write the CSV + one figure per
    watershed (north → south)."""
    out_dir = Path(out_dir)
    csv = out_dir / "dtdp_response_metrics.csv"
    if csv.exists() and not regen:
        tbl = pd.read_csv(csv)
        print(f"loaded {csv}", flush=True)
    else:
        tbl = assemble(data_dir, device=device, n_seeds=n_seeds)
        csv.parent.mkdir(parents=True, exist_ok=True)
        tbl.round(4).to_csv(csv, index=False)
        print(f"wrote {csv}", flush=True)

    order = _basin_order(data_dir, sorted(tbl["basin"].unique()))
    figdir = out_dir / "figures" / "dtdp_response"
    for b in order:
        _plot_basin(b, tbl[tbl.basin == b], figdir / f"{b}.png")
    print(f"wrote {len(order)} figures -> {figdir}", flush=True)
    return tbl


if __name__ == "__main__":
    make_dtdp_response()
