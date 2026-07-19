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
  total annual runoff, Apr–Jul freshet volume ("seasonal runoff") — both from
  the full-record mean-monthly regime — plus the daily 99.9th-percentile (flood
  peak) and 30th-percentile (low flow).  Each enters only as a ratio, so the %
  change is area-independent.

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
from ..cdec15 import CAL_END
from ..io import load_basin_area
from .climatology import _WY, _basin_order, _monthly_taf
from .evaluate import teacher_cache_path

DOMAIN = "15cdec_grid"
NOAH_DPL_CSV = "artifacts/dpl/noah/params_dpl.csv"       # frozen noah-lite SAC params
NOAH_CANOPY_CSV = "artifacts/dpl/noah/params_canopy.csv"  # + soil_chi
BASE_TORCH_CSV = "artifacts/dpl/noah/daily_sim_noah_torch.csv"  # torch present-climate base
RAW_DIR = "artifacts/dpl/testing/hybrid_pet_noah"        # PET input, no response loss
DTDP_DIR = "artifacts/dpl/testing/hybrid_pet_dtdp"       # + dp/dt response loss, λ=0.1
DTDP_L03_DIR = "artifacts/dpl/testing/hybrid_pet_dtdp_l0.3"   # + response loss, λ=0.3
N_SEEDS = 3

#: model columns, left → right.
PHYSICS = "Physics (noah)"
RAW = "Hybrid raw"
DTDP = "Hybrid dt·dp"
MODEL_ORDER = [PHYSICS, RAW, DTDP]

#: metric rows: (key, pretty label).  ``q999``/``q30`` are DAILY-flow percentiles.
#: ``q999`` = the 99.9th percentile = the FLOOD PEAK (~top 37 days of the 1915-2018
#: record) — deliberately the extreme tail, not Q98: in snow basins Q98 tracks the
#: snowmelt-freshet shoulder (which the freshet row already carries and which
#: *declines* under warming), whereas the flood peak *intensifies* (snow→rain +
#: rain-on-snow), the complementary half of the warming story.  ``q30`` = low flow.
METRICS: list[tuple[str, str]] = [
    ("annual", "Total annual runoff"),
    ("freshet", "Apr–Jul freshet"),
    ("q999", "Daily Q99.9 (flood peak)"),
    ("q30", "Daily Q30 (low flow)"),
]

#: response-surface grid — nodes sit exactly on the ±10% / +3 °C training
#: anchors, and bracket them at ±20% / +4 °C.  contourf interpolates between the
#: nodes; the 9×9 grid (step 5% / 0.5 °C) gives smooth surfaces at a tractable
#: per-point (frozen noah-lite ~4 s) cost.  Coarsen (step 0.10 / 1.0 → 5×5) for a
#: quick pass, or densify further if wanted.
DP = np.round(np.arange(-0.20, 0.2001, 0.05), 4)         # Δprecip fraction (9)
DT = np.round(np.arange(0.0, 4.0001, 0.5), 4)            # ΔT degC (9)

#: the dt·dp model's response-loss training anchors — the 5 non-origin corners of
#: {−dp,0,+dp}×{0,+dt} — marked on the surfaces so the supervised grid points are
#: distinguishable from the held-out (interpolated / extrapolated) region.
ANCHOR_DP, ANCHOR_DT = 0.10, 3.0
TRAIN_ANCHORS = [(-ANCHOR_DP, 0.0), (ANCHOR_DP, 0.0), (0.0, ANCHOR_DT),
                 (-ANCHOR_DP, ANCHOR_DT), (ANCHOR_DP, ANCHOR_DT)]

#: hydroclimate regimes — freshet-fraction terciles (Apr–Jul runoff / annual of
#: the noah_ca physics baseline, the snowmelt-timing signature; 5 basins each,
#: snowmelt-strongest → weakest).  Shared by the hybrid-family and physics
#: regime-aggregate figures.
REGIMES: dict[str, list[str]] = {
    "snow": ["PNF", "MIL", "TLG", "ISB", "MRC"],
    "mix":  ["NML", "TRM", "MKM", "SCC", "FOL"],
    "rain": ["YRS", "ORO", "SHA", "BND", "NHG"],
}
_REGIME_TITLE = {"snow": "SNOW-dominated", "mix": "MIXED", "rain": "RAIN-dominated"}

#: response-surface EVALUATION window — the metrics reduce over this subset of the
#: daily record: WY1951-1988 (pre-cal) + WY2004-2018 (val).  It
#:   (1) EXCLUDES the WY1989-2003 CAL window the hybrids + dt·dp loss trained on,
#:       so the reported response is OUT-OF-SAMPLE; and
#:   (2) DROPS the 1915-1950 lead-in.  The physics runs cold-start from 1915 (SMA
#:       [0,0,100,100,100,0], Snow-17 zeros); ~35 yr equilibrates every store
#:       (incl. the slow multi-year lztwc) before WY1951, and the LSTM's 365-day
#:       lookback + its 1915-based sim channel are likewise warm.
#: Baseline and perturbed share the identical 1915 spin-up, so it cancels in the
#: %Δ regardless — the window just makes the eval OOS + spin-up-transient-free.
_CAL_START = "1988-10-01"       # WY1989 — the hybrids' training-window start
_EVAL_START = "1950-10-01"      # WY1951 — drop the 1915-1950 cold-start lead-in


def _eval_mask(idx) -> np.ndarray:
    """Boolean row mask for the response-evaluation window (see :data:`_EVAL_START`)."""
    ts = pd.DatetimeIndex(idx)
    in_cal = (ts >= pd.Timestamp(_CAL_START)) & (ts <= pd.Timestamp(CAL_END))
    return np.asarray((ts >= pd.Timestamp(_EVAL_START)) & ~in_cal)


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
def _load_ensemble(ens_dir: str, data_dir: str, dev, n_seeds: int,
                   *, physics_csv: str | None = None,
                   sim_cache: str | None = None):
    """Load the shared hybrid data + up to ``n_seeds`` member models once (every
    seed shares the physics/domain/normalization config).  ``physics_csv`` /
    ``sim_cache`` override the checkpoint-stored paths for ensembles that were
    canonicalized out of ``testing/`` (their training-time paths are now stale)."""
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
        data_dir,
        physics_csv=physics_csv if physics_csv is not None else ck0.get("physics_csv"),
        sim_cache=sim_cache if sim_cache is not None else ck0.get("sim_cache"),
        use_statics=bool(ck0["n_static"]),
        use_doy=cfg.get("use_doy", True), use_pet=cfg.get("use_pet", False),
        use_sim=cfg.get("use_sim", True),
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
    names = feature_names(cfg.get("use_doy", True), cfg.get("use_pet", False),
                          cfg.get("use_sim", True))
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

    from .hybrid.data import (apply_response_perturbation, basin_pet_pt,
                              perturbed_static)
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
    rep = dict(feat=feat_t, sim=sim_t)
    if h.static is not None and h.static_ing:      # CLIMATE statics co-vary too
        sp = perturbed_static(h.static_ing, float(dp), float(dt))
        rep["static"] = torch.as_tensor(sp).to(dtype=h.static.dtype,
                                               device=h.device)
    hp = dc.replace(h, **rep)

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
    """Per-basin (annual, freshet, q98, q30) from the full record (index=basin,
    cols=the 4 metric keys).

    ``annual``/``freshet`` are TAF from the mean-monthly regime; ``q999``/``q30``
    are the 99.9th (flood peak) / 30th (low flow) percentiles of the DAILY flow
    (mm/day).  Percentiles enter the surfaces only as a % change vs the (0,0)
    baseline, so the mm/day unit is immaterial (the area cancels).

    Reduced over the out-of-sample, spin-up-free window :func:`_eval_mask`
    (WY1951-1988 + WY2004-2018; excludes the CAL window + the 1915-1950 lead-in)."""
    daily = daily[_eval_mask(daily.index)]               # OOS + spin-up-free window
    m = _monthly_taf(daily, areas)                       # date x basin monthly TAF
    out = {}
    for b in m.columns:
        s = m[b].dropna()
        reg = s.groupby(s.index.month).mean().reindex(_WY)  # 12-pt WY regime
        d = daily[b].dropna().to_numpy()                    # daily mm/day
        out[b] = dict(annual=float(reg.sum()),
                      freshet=float(reg.loc[[4, 5, 6, 7]].sum()),
                      q999=float(np.percentile(d, 99.9)) if d.size else float("nan"),
                      q30=float(np.percentile(d, 30)) if d.size else float("nan"))
    return pd.DataFrame(out).T


def _aggregate_regime(tbl: pd.DataFrame, basins: list[str],
                      areas: dict[str, float]) -> pd.DataFrame:
    """Area-weighted mean of each model's per-basin % change over ``basins`` —
    one pooled surface per (model, dp, dt).  Model-agnostic: works for the 2-col
    physics table and the 4-col hybrid table alike."""
    w = np.array([areas[b] for b in basins], float)
    sub = tbl[tbl.basin.isin(basins)]
    rows = []
    for (model, dp, dt), g in sub.groupby(["model", "dp", "dt"]):
        g = g.set_index("basin").reindex(basins)
        r: dict[str, object] = {"basin": "AGG", "model": model, "dp": dp, "dt": dt}
        for k, _ in METRICS:
            r[f"pct_{k}"] = float(np.average(g[f"pct_{k}"].to_numpy(), weights=w))
        rows.append(r)
    return pd.DataFrame(rows)


def assemble(data_dir: str = "data", *, device: str = "cuda",
             n_seeds: int = N_SEEDS, dtdp_dir: str = DTDP_DIR) -> pd.DataFrame:
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
                                                           "q999", "q30")}))

    for dp, dt in grid:
        _emit(PHYSICS, dp, dt, _metrics_from_daily(phys[(dp, dt)], areas))

    # PET depends only on ΔT — precompute once per distinct dt and reuse.
    from .hybrid.data import basin_pet_pt
    pet_by_dt = {float(dt): basin_pet_pt(dom, delta_t=float(dt)) for dt in DT}

    for label, ens in [(RAW, RAW_DIR), (DTDP, dtdp_dir)]:
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
def _ensemble_anchors(dtdp_dir: str, n_seeds: int = N_SEEDS):
    """The (Δp, ΔT) response-loss anchors the ensemble was actually trained on
    (read from a member checkpoint's cfg); falls back to :data:`TRAIN_ANCHORS`."""
    import torch
    cps = sorted(Path(dtdp_dir).glob("seed*/checkpoints/best.pt"))[:n_seeds]
    if not cps:
        return TRAIN_ANCHORS
    cfg = torch.load(cps[0], map_location="cpu", weights_only=False)["cfg"]
    ancs = [(float(a["dp"]), float(a["dt"]))
            for a in cfg.get("response_anchors", [])]
    return ancs or TRAIN_ANCHORS


def _plot_basin(basin: str, sub: pd.DataFrame, out: Path,
                anchors=TRAIN_ANCHORS) -> None:
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
            for adp, adt in anchors:
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
                       dtdp_dir: str = DTDP_DIR, tag: str = "",
                       regen: bool = False) -> pd.DataFrame:
    """Assemble (or reload) the metrics table for the ``dtdp_dir`` ensemble, write
    ``dtdp_response_metrics{tag}.csv`` + one figure per watershed (north → south)
    under ``figures/dtdp_response{tag}/``."""
    out_dir = Path(out_dir)
    csv = out_dir / f"dtdp_response_metrics{tag}.csv"
    if csv.exists() and not regen:
        tbl = pd.read_csv(csv)
        print(f"loaded {csv}", flush=True)
    else:
        tbl = assemble(data_dir, device=device, n_seeds=n_seeds, dtdp_dir=dtdp_dir)
        csv.parent.mkdir(parents=True, exist_ok=True)
        tbl.round(4).to_csv(csv, index=False)
        print(f"wrote {csv}", flush=True)

    order = _basin_order(data_dir, sorted(tbl["basin"].unique()))
    anchors = _ensemble_anchors(dtdp_dir, n_seeds)
    figdir = out_dir / "figures" / f"dtdp_response{tag}"
    for b in order:
        _plot_basin(b, tbl[tbl.basin == b], figdir / f"{b}.png", anchors=anchors)
    print(f"wrote {len(order)} figures -> {figdir}", flush=True)
    return tbl


def _ens_cal_kge(ens_dir: str, n_seeds: int = N_SEEDS) -> float:
    """Mean per-seed selection cal KGE over an ensemble's checkpoints."""
    import torch
    ks = []
    for cp in sorted(Path(ens_dir).glob("seed*/checkpoints/best.pt"))[:n_seeds]:
        ks.append(float(torch.load(cp, map_location="cpu",
                                   weights_only=False)["best_cal_kge"]))
    return float(np.mean(ks)) if ks else float("nan")


def make_lambda_compare(dtdp_variants: dict[str, tuple[str, str]],
                        data_dir: str = "data",
                        out_dir: str | Path = "artifacts/dpl") -> pd.DataFrame:
    """Compare the warming-response fidelity of one or more dt·dp λ variants vs
    physics and the raw hybrid.  ``dtdp_variants[label] = (metrics_csv, ens_dir)``
    (physics + raw are read from the first csv — identical across variants).

    Writes a 2-panel figure (pooled annual %Δ along the pure-warming axis dp=0;
    per-basin +3 °C annual %Δ vs physics) + a summary CSV keyed by model."""
    labels = list(dtdp_variants)
    tbls = {lab: pd.read_csv(csv) for lab, (csv, _) in dtdp_variants.items()}
    t0 = tbls[labels[0]]
    dts = np.sort(t0["dt"].unique())

    def pooled(tbl, model, dt, dp=0.0, col="pct_annual"):
        s = tbl[(tbl.model == model) & np.isclose(tbl.dp, dp) & np.isclose(tbl.dt, dt)]
        return float(s[col].mean())

    # pooled warming curves (dp=0): physics, raw, each dt·dp variant
    curves = {PHYSICS: [pooled(t0, PHYSICS, dt) for dt in dts],
              RAW: [pooled(t0, RAW, dt) for dt in dts]}
    for lab in labels:
        curves[f"dt·dp {lab}"] = [pooled(tbls[lab], DTDP, dt) for dt in dts]

    # per-basin +3 °C annual response, and the summary rows
    order = _basin_order(data_dir, sorted(t0["basin"].unique()))
    at3 = {}
    for name, tbl, model in ([(PHYSICS, t0, PHYSICS), (RAW, t0, RAW)]
                             + [(f"dt·dp {lab}", tbls[lab], DTDP) for lab in labels]):
        at3[name] = (tbl[(tbl.model == model) & np.isclose(tbl.dp, 0.0)
                         & np.isclose(tbl.dt, 3.0)].set_index("basin")["pct_annual"]
                     .reindex(order))
    phys3 = at3[PHYSICS]
    ens_of = {f"dt·dp {lab}": d for lab, (_, d) in dtdp_variants.items()}
    ens_of[RAW] = RAW_DIR
    rows = []
    for name in at3:
        v = at3[name]
        ratio = float(v.sum() / phys3.sum()) if phys3.sum() else float("nan")
        sign = int((np.sign(v) == np.sign(phys3)).sum())
        cal = _ens_cal_kge(ens_of[name]) if name in ens_of else float("nan")
        rows.append(dict(model=name, warm3_pooled=round(float(v.mean()), 2),
                         resp_ratio_vs_phys=round(ratio, 2),
                         sign_correct_15=sign,
                         mean_abs_err_vs_phys=round(float((v - phys3).abs().mean()), 2),
                         mean_cal_kge=round(cal, 4) if np.isfinite(cal) else None))
    summary = pd.DataFrame(rows)

    # figure
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(9.0, 4.2),
                                   constrained_layout=True)
    sty = {PHYSICS: dict(color="k", lw=2.4, marker="o", ms=4, zorder=5),
           RAW: dict(color="#d62728", lw=1.8, marker="s", ms=3, ls="--")}
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, lab in enumerate(labels):
        sty[f"dt·dp {lab}"] = dict(color=palette[i % 4], lw=1.9, marker="^", ms=3)
    for name, y in curves.items():
        axc.plot(dts, y, label=name, **sty[name])
    axc.axhline(0, color="0.6", lw=0.7)
    axc.set_xlabel("ΔT (°C),  Δprecip = 0"); axc.set_ylabel("pooled annual runoff %Δ")
    axc.set_title("Warming response (15-basin mean)"); axc.legend(fontsize=7)
    axc.grid(alpha=0.3, lw=0.4)

    x = np.arange(len(order)); bw = 0.8 / max(len(at3) - 1, 1)
    axb.plot(x, phys3.to_numpy(), "ko-", ms=3, lw=1.2, label=PHYSICS, zorder=5)
    j = 0
    for name in at3:
        if name == PHYSICS:
            continue
        axb.bar(x + (j - (len(at3) - 2) / 2) * bw, at3[name].to_numpy(), bw,
                label=name, color=sty[name]["color"], alpha=0.85)
        j += 1
    axb.axhline(0, color="0.5", lw=0.7)
    axb.set_xticks(x); axb.set_xticklabels(order, rotation=90, fontsize=6.5)
    axb.set_ylabel("+3 °C annual runoff %Δ")
    axb.set_title("Per-basin +3 °C response vs physics"); axb.legend(fontsize=7)
    fig.suptitle("dt·dp response-loss strength — warming-response fidelity",
                 fontsize=10, fontweight="bold")
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figures" / "dtdp_lambda_compare.png", dpi=300)
    plt.close(fig)
    summary.to_csv(out_dir / "dtdp_lambda_compare.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"wrote {out_dir/'figures'/'dtdp_lambda_compare.png'}", flush=True)
    return summary


if __name__ == "__main__":
    make_dtdp_response()
