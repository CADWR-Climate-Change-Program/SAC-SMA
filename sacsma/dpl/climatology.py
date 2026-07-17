"""Per-watershed mean-monthly TAF regime for the 11 CalSim3-mapped CDEC basins,
as a five-step ablation vs the observed CalSim3 monthly FNF.

Each daily model sim (mm/day) is converted to monthly volume (TAF) with the
CalSim3-consistent catchment areas (``calsim.catchments.basin_areas``), then
averaged by calendar month over the **combined out-of-calibration period**
(WY1950-1987 pre-calibration + WY2004-2018 validation; the WY1989-2003
calibration window is excluded) into a 12-point water-year regime (Oct->Sep).
The observed reference is the CalSim3 full-natural-flow monthly series assembled
by ``calsim.compare.build_anchor_long`` (rim systems vs FLOW-UNIMPAIRED,
Mokelumne/Calaveras vs summed CalSim3 inflow arcs).  The 4 Tulare basins
(PNF/TRM/SCC/ISB) have no CalSim3 counterpart and are dropped.

Five figures, each an 11-basin (north->south) grid contrasting one ablation
step against CalSim3 FNF:
  a  GA SAC-SMA            vs  dPL hamon_dense            (learned parameters)
  b  dPL hamon_dense       vs  dPL hamon                  (fine HRU -> grid+footprint)
  c  dPL hamon             vs  dPL pt                     (Hamon -> Priestley-Taylor)
  d  dPL pt                vs  dPL noah                   (PT cascade -> Noah-lite ET)
  e  dPL noah -> Hybrid -> Hybrid PET+dT   (the LSTM step on the noah physics)

``Hybrid`` / ``Hybrid PET+dT`` are the CANONICAL seed ENSEMBLES (mean of member
daily flows) on the noah physics baseline — the sim channel is noah's TORCH
daily dump (``artifacts/dpl/noah/daily_sim_noah_torch.csv``), numerics-matched
to the +2 °C torch teacher.  (``noah_ft``, the obs-steered seasonal-melt
fine-tune, was DEMOTED 2026-07-17: pooled val ties frozen noah, CalSim3 a
wash, NHG + north-state volume worse — the head-to-head record lives in
``artifacts/dpl/RUNS.md``.)
Output: ``artifacts/dpl/figures/cdec15_climatology_{a..e}.png``.

A dPL-side artifact (needs torch for the hybrids) that reads the lightweight
CalSim3-FNF loader from ``calsim.compare``; it never makes calsim depend on torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io import load_hru_table, mmday_to_cfs

_AF_PER_CFS_DAY = 1.98347          # cfs-day -> acre-feet; /1000 -> TAF
#: combined out-of-calibration period (WY1950-1987 + WY2004-2018), cal excluded.
_PERIOD = [("1949-10-01", "1987-09-30"), ("2003-10-01", "2018-12-31")]
_WY = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
_WY_LABELS = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]

#: frozen-model sims: label -> run_basin spec (csv=None => archived GA optimum).
#: ``dPL pt`` IS the refined PT (snow-albedo 0.6 + dewpoint 2.0); ``dPL noah`` is
#: the Noah-lite external-ET canopy on PT potential.
FROZEN: dict[str, dict] = {
    "GA SAC-SMA":        dict(csv=None, domain="15cdec", pet="hamon", alb=0.0, dew=0.0),
    "dPL hamon_dense":   dict(csv="artifacts/dpl/hamon_dense/params_dpl.csv",
                              domain="15cdec", pet="hamon", alb=0.0, dew=0.0),
    "dPL hamon":         dict(csv="artifacts/dpl/hamon/params_dpl.csv",
                              domain="15cdec_grid", pet="hamon", alb=0.0, dew=0.0),
    "dPL pt":            dict(csv="artifacts/dpl/pt/params_dpl.csv",
                              domain="15cdec_grid", pet="priestley_taylor", alb=0.6, dew=2.0),
    "dPL noah":          dict(csv="artifacts/dpl/noah/params_dpl.csv",
                              domain="15cdec_grid", pet="priestley_taylor", alb=0.0, dew=0.0,
                              et_scheme="noah_lite",
                              canopy_csv="artifacts/dpl/noah/params_canopy.csv"),
}
#: torch-only sims: label -> canonical daily-sim CSV (for models the frozen
#: run_basin cannot reconstruct).  Empty since the noah_ft demotion (2026-07-17);
#: the ingestion route stays for future torch-only exports.
TORCH_SIM: dict[str, str] = {}
#: hybrid sims: label -> canonical ENSEMBLE dir (seed*/checkpoints/best.pt
#: averaged; physics settings read from the member ckpt cfg).  Both sit on the
#: noah physics baseline (sim channel = its torch daily dump): ``Hybrid`` is the
#: plain feature ensemble (no PET channel, no dT loss — the skill step),
#: ``Hybrid PET+dT`` adds the PT-potential input + the temperature-consistency
#: loss (the physics-consistent climate response, same skill).
HYBRID: dict[str, str] = {
    "Hybrid": "artifacts/dpl/hybrid",
    "Hybrid PET+dT": "artifacts/dpl/hybrid_pet_dt",
}

#: per-series line style (identity by hue; CalSim3 FNF emphasized in black).
STYLE: dict[str, dict] = {
    "CalSim3 FNF":        dict(color="#000000", lw=2.4, ls="--", marker="o", ms=4.5,
                               zorder=10),
    "GA SAC-SMA":         dict(color="#9e9e9e", lw=1.9),
    "dPL hamon_dense":    dict(color="#8c564b", lw=1.9),
    "dPL hamon":          dict(color="#1f77b4", lw=1.9),
    "dPL pt":             dict(color="#ff7f0e", lw=1.9),
    "dPL noah":           dict(color="#bcbd22", lw=2.0),
    "Hybrid":             dict(color="#9467bd", lw=2.1),
    "Hybrid PET+dT":      dict(color="#17becf", lw=2.1),
}

#: (tag, subtitle, [model labels]) -- each renders one 11-basin figure.
COMPARISONS: list[tuple[str, str, list[str]]] = [
    ("a", "learned parameters: GA SAC-SMA → dPL (same Hamon physics, fine HRUs)",
        ["GA SAC-SMA", "dPL hamon_dense"]),
    ("b", "resolution + footprint: fine 7891-HRU → 1/16° grid + CalSim3 footprint",
        ["dPL hamon_dense", "dPL hamon"]),
    ("c", "energy PET: Hamon → Priestley–Taylor (snow-albedo + dewpoint)",
        ["dPL hamon", "dPL pt"]),
    ("d", "canopy ET: PT cascade → Noah-lite external ET",
        ["dPL pt", "dPL noah"]),
    ("e", "the LSTM step: noah → Hybrid → +PET+dT",
        ["dPL noah", "Hybrid", "Hybrid PET+dT"]),
]


# --------------------------------------------------------------------------- #
# daily model sims (mm/day, date x basin) -- all cached to CSV
# --------------------------------------------------------------------------- #
def _daily_frozen(spec: dict, data_dir: str, cache: Path) -> pd.DataFrame:
    from .hybrid.data import build_frozen_sim
    return build_frozen_sim(data_dir, spec["csv"], cache=cache, domain=spec["domain"],
                            pet_source=spec["pet"], pt_snow_albedo=spec["alb"],
                            pt_dewpoint_depression=spec["dew"],
                            et_scheme=spec.get("et_scheme", "sac"),
                            canopy_csv=spec.get("canopy_csv"))


def _daily_ensemble(ens_dir: str, data_dir: str, device, cache: Path) -> pd.DataFrame:
    """Full-record ENSEMBLE-MEAN daily hybrid flow (mm/day) — mean over all
    ``seed*/checkpoints/best.pt`` members; cached to CSV.  Data is loaded once
    (every seed shares the physics/variant/domain config)."""
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")
    import torch

    from .hybrid.data import load_hybrid_data
    from .hybrid.model import HybridLSTM
    from .hybrid.train import predict_days

    ckpts = sorted(Path(ens_dir).glob("seed*/checkpoints/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no seed*/checkpoints/best.pt under {ens_dir}")
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    cfg = ck0["cfg"]
    if ck0.get("variant", "feature") != "feature":
        raise ValueError("residual hybrid checkpoints are retired (2026-07-16)")
    data = load_hybrid_data(
        data_dir, physics_csv=ck0.get("physics_csv"),
        sim_cache=ck0.get("sim_cache"), use_statics=bool(ck0["n_static"]),
        use_doy=cfg.get("use_doy", True),
        use_pet=cfg.get("use_pet", False),
        domain=cfg.get("physics_domain", "15cdec"),
        pet_source=cfg.get("pet_source", "hamon"),
        pt_snow_albedo=cfg.get("pt_snow_albedo", 0.0),
        pt_dewpoint_depression=cfg.get("pt_dewpoint_depression", 0.0),
        et_scheme=cfg.get("physics_et_scheme", "sac"),
        canopy_csv=cfg.get("canopy_csv") or None, device=device)
    bb, tt = data.eval_days("all")
    accum = None
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        model = HybridLSTM(data.n_feat, data.n_static,
                           hidden=cfg["hidden"], static_embed=cfg["static_embed"],
                           dropout=cfg["dropout"]).to(device)
        model.load_state_dict(ck["model"])
        f = predict_days(model, data, bb, tt).clamp_min(0.0).cpu().numpy()
        accum = f if accum is None else accum + f
    flow = accum / len(ckpts)
    arr = np.full((len(data.basins), len(data.dates)), np.nan)
    arr[bb.cpu().numpy(), tt.cpu().numpy()] = flow
    df = pd.DataFrame(arr.T, index=data.dates, columns=list(data.basins))
    df.index.name = "date"
    df.to_csv(cache)
    return df


# --------------------------------------------------------------------------- #
# daily mm/day -> monthly TAF -> combined-period WY mean-monthly regime
# --------------------------------------------------------------------------- #
def _monthly_taf(daily: pd.DataFrame, areas: dict[str, float]) -> pd.DataFrame:
    """Daily mm/day (date x basin) -> monthly TAF (month-start x basin).

    Gap-robust: monthly MEAN cfs x days-in-month (a complete month equals the
    daily cfs-day sum; a partial month is not volume-undercounted)."""
    out = {}
    for b in daily.columns:
        if b not in areas:
            continue
        cfs = pd.Series(mmday_to_cfs(daily[b].to_numpy(), areas[b]), index=daily.index)
        mean_cfs = cfs.resample("MS").mean()
        cfs_day = mean_cfs.to_numpy() * mean_cfs.index.days_in_month.to_numpy()
        out[b] = pd.Series(cfs_day * _AF_PER_CFS_DAY / 1000.0, index=mean_cfs.index)
    return pd.DataFrame(out)


def _in_period(idx) -> np.ndarray:
    """Boolean mask: months inside the combined out-of-calibration period.
    Accepts a DatetimeIndex or a monthly PeriodIndex."""
    ts = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else pd.DatetimeIndex(idx)
    mask = np.zeros(len(ts), bool)
    for lo, hi in _PERIOD:
        mask |= (ts >= pd.Timestamp(lo)) & (ts <= pd.Timestamp(hi))
    return mask


def _climatology(monthly_taf: pd.DataFrame) -> pd.DataFrame:
    """Monthly TAF (date x basin) -> WY 12-point mean-monthly regime over the
    combined out-of-calibration period."""
    m = monthly_taf[_in_period(monthly_taf.index)]
    clim = {b: m[b].groupby(m.index.month).mean().reindex(_WY) for b in m.columns}
    return pd.DataFrame(clim, index=_WY)


def _score(sim_m: pd.Series, obs_m: pd.Series) -> tuple[float, float, float, float]:
    """Monthly (KGE, NSE, pbias %, seasonal misplaced volume 0-1) of a model vs
    CalSim3 FNF over the combined period.  Aligns on monthly Periods (model sims
    are month-start, CalSim3 month-end)."""
    from ..metrics import kge, nse, pbias, seasonal_mismatch
    s, o = sim_m.copy(), obs_m.copy()
    s.index, o.index = s.index.to_period("M"), o.index.to_period("M")
    df = pd.concat({"s": s, "o": o}, axis=1).dropna()
    df = df[_in_period(df.index)]
    nan = float("nan")
    if len(df) < 12:
        return (nan, nan, nan, nan)
    sv, ov = df["s"].to_numpy(), df["o"].to_numpy()
    return (kge(sv, ov), nse(sv, ov), pbias(sv, ov),
            seasonal_mismatch(df.index.to_timestamp(), sv, ov))


def _calsim3_fnf(data_dir: str, cache: Path) -> pd.DataFrame:
    """CalSim3 monthly FNF (date x basin, TAF) for the mapped basins; cached."""
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")
    from ..calsim.compare import build_anchor_long
    al = build_anchor_long(data_dir, sets=("15cdec",))
    c3 = al[al["source"] == "calsim3"]
    piv = c3.pivot_table(index="date", columns="basin", values="flow_taf", aggfunc="sum")
    piv.index = pd.to_datetime(piv.index)
    piv.index.name = "date"
    piv.to_csv(cache)
    return piv


# --------------------------------------------------------------------------- #
# assembly + plotting
# --------------------------------------------------------------------------- #
def _basin_order(data_dir: str, keep: list[str]) -> list[str]:
    from .._figures import folsom_before_yuba
    hru = load_hru_table(data_dir, domain="15cdec")
    lat = hru.groupby("basin")["lat"].mean().sort_values(ascending=False)
    order = folsom_before_yuba("15cdec", lat.index.tolist())
    return [b for b in order if b in keep]


def assemble(data_dir: str = "data", *, device: str = "cuda") -> dict:
    """Compute every series' combined-period WY mean-monthly regime (TAF).
    Returns ``clim`` (label -> DataFrame[12 x basin]) and ``order`` (the 11
    CalSim3-mapped basins, north->south)."""
    from ..calsim.catchments import basin_areas
    from .config import pick_device

    areas = basin_areas(data_dir, domain="15cdec")
    cachedir = Path("artifacts/dpl/_climatology_cache")
    cachedir.mkdir(parents=True, exist_ok=True)

    monthly: dict[str, pd.DataFrame] = {}
    clim: dict[str, pd.DataFrame] = {}
    for label, spec in FROZEN.items():
        tag = label.split()[-1].lower()
        daily = _daily_frozen(spec, data_dir, cachedir / f"sim_{tag}.csv")
        monthly[label] = _monthly_taf(daily, areas)
        clim[label] = _climatology(monthly[label])
        print(f"  assembled {label}", flush=True)

    for label, csv in TORCH_SIM.items():   # torch-only exports: read the dump
        daily = pd.read_csv(csv, parse_dates=["date"]).set_index("date")
        monthly[label] = _monthly_taf(daily, areas)
        clim[label] = _climatology(monthly[label])
        print(f"  assembled {label} (torch daily sim)", flush=True)

    dev = pick_device(device)
    for label, ens_dir in HYBRID.items():
        tag = Path(ens_dir).name                # sim_hybrid / sim_hybrid_pet_dt
        daily = _daily_ensemble(ens_dir, data_dir, dev,
                                cachedir / f"sim_{tag}.csv")
        monthly[label] = _monthly_taf(daily, areas)
        clim[label] = _climatology(monthly[label])
        print(f"  assembled {label}", flush=True)

    obs = _calsim3_fnf(data_dir, cachedir / "calsim3_fnf_monthly.csv")
    clim["CalSim3 FNF"] = _climatology(obs)
    order = _basin_order(data_dir, list(obs.columns))

    # monthly KGE/NSE of each model vs CalSim3 FNF over the combined period
    metrics: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for label in [*FROZEN, *TORCH_SIM, *HYBRID]:
        metrics[label] = {b: _score(monthly[label][b], obs[b])
                          for b in order if b in monthly[label] and b in obs}
    print(f"  observed: CalSim3 FNF for {order}", flush=True)
    return dict(clim=clim, order=order, metrics=metrics)


def _bar_summary(ax, models: list[str], metrics: dict, order: list[str]) -> None:
    """Grouped bars in the spare 12th cell: mean KGE and NSE over the 11 basins
    per model (bar), with min-max whiskers.  Model-coloured to match the lines."""
    n = len(models)
    width = 0.8 / n
    gmin = gmax = 0.0
    for gi, mi in ((0, 0), (1, 1)):                # 0 -> KGE, 1 -> NSE
        for k, label in enumerate(models):
            vals = np.array([metrics[label][b][mi] for b in order
                             if b in metrics.get(label, {})
                             and np.isfinite(metrics[label][b][mi])])
            if not vals.size:
                continue
            x = gi + (k - (n - 1) / 2) * width
            mean = float(vals.mean())
            ax.bar(x, mean, width=width * 0.9, color=STYLE[label]["color"],
                   alpha=0.85, zorder=2)
            ax.errorbar(x, mean, yerr=[[mean - vals.min()], [vals.max() - mean]],
                        fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.5,
                        zorder=3)
            ax.text(x, vals.max() + 0.012, f"{mean:.2f}", ha="center", va="bottom",
                    fontsize=6.0, fontweight="bold", color=STYLE[label]["color"])
            gmin, gmax = min(gmin, vals.min()), max(gmax, vals.max())
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["KGE", "NSE"], fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(gmin - 0.05 if gmin < 0 else 0.0, min(1.15, gmax + 0.12))
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("mean skill, 11 basins (min–max)", fontsize=9.5, fontweight="bold")


def _plot_comparison(data: dict, models: list[str], tag: str, subtitle: str,
                     path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clim, order, metrics = data["clim"], data["order"], data["metrics"]
    series = ["CalSim3 FNF", *models]
    nrow, ncol = 4, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.0, 12.0))
    seen: dict[str, object] = {}
    for ax, basin in zip(axes.ravel(), order, strict=False):
        for label in series:
            if basin not in clim[label].columns:
                continue
            (ln,) = ax.plot(range(12), clim[label][basin].to_numpy(),
                            label=label, **STYLE[label])
            seen[label] = ln
        # KGE / NSE (monthly vs CalSim3 FNF, combined period), same colour
        ax.text(0.03, 0.965, "KGE / NSE", transform=ax.transAxes, fontsize=6.0,
                color="#888888", va="top", ha="left")
        for j, label in enumerate(models):
            mv = metrics.get(label, {}).get(basin)
            if mv is None or not np.isfinite(mv[0]):
                continue
            ax.text(0.03, 0.965 - (j + 1) * 0.072, f"{mv[0]:.2f} / {mv[1]:.2f}",
                    transform=ax.transAxes, fontsize=6.8, fontweight="bold",
                    color=STYLE[label]["color"], va="top", ha="left")
        ax.set_title(basin, fontsize=11, fontweight="bold")
        ax.set_xticks(range(12))
        ax.set_xticklabels(_WY_LABELS, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.margins(x=0.02)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(bottom=0)
    raveled = axes.ravel()
    _bar_summary(raveled[len(order)], models, metrics, order)
    for ax in raveled[len(order) + 1:]:
        ax.axis("off")
    leg = [s for s in series if s in seen]
    fig.legend([seen[s] for s in leg], leg, loc="lower center",
               ncol=len(leg), fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f"({tag})  {subtitle}", fontsize=13, fontweight="bold", y=0.998)
    fig.text(0.5, 0.966, "mean-monthly flow, TAF/month  •  "
             "WY1950–1987 + WY2004–2018 (calibration WY1989–2003 excluded)",
             ha="center", fontsize=9.5, color="#555555")
    fig.supylabel("mean-monthly flow (TAF/month)", fontsize=10)
    fig.supxlabel("water-year month (Oct → Sep)", fontsize=10, y=0.052)
    fig.tight_layout(rect=(0.015, 0.055, 1, 0.955))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _plot_metrics_bars(data: dict, path: Path) -> None:
    """All-series per-basin grouped bars on three metrics (KGE, |pbias|,
    seasonal misplaced volume), one metric per row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics, order = data["metrics"], data["order"]
    models = [*FROZEN, *TORCH_SIM, *HYBRID]
    n = len(models)
    width = 0.86 / n
    # (row label, value from a (kge,nse,pbias,smv) tuple, y-lower)
    rows = [
        ("KGE", lambda m: m[0], 0.0),
        ("| PBIAS |  (%)", lambda m: abs(m[2]), 0.0),
        ("seasonal misplaced volume  (%)", lambda m: m[3] * 100.0, 0.0),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14.0, 11.0), sharex=True)
    seen: dict[str, object] = {}
    for ax, (name, fn, ylo) in zip(axes, rows, strict=True):
        for i in range(len(order)):                 # alternate-column shading = clear groups
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color="#f4f4f4", zorder=0)
        for k, label in enumerate(models):
            xs, ys = [], []
            for i, b in enumerate(order):
                mv = metrics.get(label, {}).get(b)
                if mv is None or not np.isfinite(fn(mv)):
                    continue
                xs.append(i + (k - (n - 1) / 2) * width)
                ys.append(fn(mv))
            bc = ax.bar(xs, ys, width=width * 0.92, color=STYLE[label]["color"],
                        alpha=0.9, zorder=2, label=label)
            seen[label] = bc
        ax.set_ylabel(name, fontsize=10.5)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(bottom=ylo)
        ax.margins(x=0)
        ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=1)
    axes[0].set_ylim(0, 1.0)                          # KGE
    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels(order, fontsize=11, fontweight="bold")
    # two legend rows: 11 series in one row clip the outer entries at the edges
    ncol = (n + 1) // 2
    fig.legend([seen[m] for m in models], models, loc="lower center", ncol=ncol,
               fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.008))
    fig.suptitle("All series vs CalSim3 FNF — monthly skill by basin "
                 "(WY1950–1987 + WY2004–2018)", fontsize=13.5, fontweight="bold",
                 y=0.998)
    fig.tight_layout(rect=(0, 0.075, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _plot_metrics_bars_agg(data: dict, path: Path) -> None:
    """All-series skill AGGREGATED across the 11 basins: one bar per model (mean)
    with a min-max whisker, one metric per row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics, order = data["metrics"], data["order"]
    models = [*FROZEN, *TORCH_SIM, *HYBRID]
    rows = [
        ("KGE", lambda m: m[0], (0.0, 1.05)),
        ("| PBIAS |  (%)", lambda m: abs(m[2]), None),
        ("seasonal misplaced volume  (%)", lambda m: m[3] * 100.0, None),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.0), sharex=True)
    for ax, (name, fn, ylim) in zip(axes, rows, strict=True):
        gmax = 0.0
        for k, label in enumerate(models):
            vals = np.array([fn(metrics[label][b]) for b in order
                             if b in metrics.get(label, {})
                             and np.isfinite(fn(metrics[label][b]))])
            if not vals.size:
                continue
            mean = float(vals.mean())
            ax.bar(k, mean, width=0.7, color=STYLE[label]["color"], alpha=0.9, zorder=2)
            ax.errorbar(k, mean, yerr=[[mean - vals.min()], [vals.max() - mean]],
                        fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3, zorder=3)
            ax.annotate(f"{mean:.2f}", (k, vals.max()), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", color=STYLE[label]["color"])
            gmax = max(gmax, vals.max())
        ax.set_ylabel(name, fontsize=10.5)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.set_ylim(*(ylim if ylim else (0.0, gmax * 1.15)))
    axes[-1].set_xticks(range(len(models)))
    axes[-1].set_xticklabels(models, rotation=22, ha="right", fontsize=10)
    fig.suptitle("All series vs CalSim3 FNF — skill aggregated across the 11 basins "
                 "(mean, min–max)\nmonthly, WY1950–1987 + WY2004–2018",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def make_cdec15_climatology(data_dir: str = "data",
                            out_dir: str | Path = "artifacts/dpl",
                            *, device: str = "cuda") -> dict:
    """Assemble every series and render the five ablation figures + the
    all-series metric-bar summaries (per-basin and basin-aggregated)."""
    data = assemble(data_dir, device=device)
    figdir = Path(out_dir) / "figures"
    for tag, subtitle, models in COMPARISONS:
        _plot_comparison(data, models, tag, subtitle,
                         figdir / f"cdec15_climatology_{tag}.png")
    _plot_metrics_bars(data, figdir / "cdec15_climatology_summary.png")
    _plot_metrics_bars_agg(data, figdir / "cdec15_climatology_summary_agg.png")
    return data


if __name__ == "__main__":
    make_cdec15_climatology()
