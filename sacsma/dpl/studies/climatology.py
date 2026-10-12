"""Per-watershed mean-monthly TAF regime for the 11 CalSim3-mapped CDEC basins,
one figure per step of the ladder, vs the observed CalSim3 monthly FNF.

Each daily model sim (mm/day) is converted to monthly volume (TAF) with the
CalSim3-consistent catchment areas (``calsim.catchments.basin_areas``), then
averaged by calendar month over the **combined out-of-calibration period**
(WY1950-1975 + WY1986-1987 + WY2004-2018: the WY1989-2003 training years and the held-out
WY1976-85 are excluded) into a 12-point water-year regime (Oct->Sep).
The observed reference is the CalSim3 full-natural-flow monthly series assembled
by ``calsim.compare.build_anchor_long`` (rim systems vs FLOW-UNIMPAIRED,
Mokelumne/Calaveras vs summed CalSim3 inflow arcs).  The 4 Tulare basins
(PNF/TRM/SCC/ISB) have no CalSim3 counterpart and are dropped.

Seven figures, each an 11-basin (north->south) grid contrasting one ladder step against
CalSim3 FNF:
  a  GA      vs  1_hru    (the network in place of the GA)
  b  1_hru   vs  2_grid   (grid cells + CalSim3 outlines)
  c  2_grid  vs  3_pt     (Priestley-Taylor PET)
  d  3_pt    vs  4_noah   (Noah-type ET with the SAC exchanges)
  e  4_noah  vs  5_px     (learned rain/snow threshold)
  f  5_px    vs  6_aef    (aef_u inputs)
  g  5_px -> hybrid -> hybrid_dt, and lstm   (the LSTM step on 5_px)

The learned runs are run as trained (:func:`sacsma.dpl.evaluate.basin_daily`); the hybrids
are the seed ENSEMBLES (mean of member daily flows).
Output: ``artifacts/results/dpl/studies/climatology/climatology_{a..g}.png``.

A dPL-side artifact (needs torch for the hybrids) that reads the lightweight
CalSim3-FNF loader from ``calsim.compare``; it never makes calsim depend on torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ... import paths
from ..._figures import plt
from ...io import load_hru_table, mmday_to_cfs

_AF_PER_CFS_DAY = 1.98347          # cfs-day -> acre-feet; /1000 -> TAF
#: combined out-of-calibration period (WY1950-1975 + WY1986-1987 + WY2004-2018): the
#: training years and the held-out WY1976-85 excluded (decision 38 of the ladder).
_PERIOD = [("1949-10-01", "1975-09-30"), ("1985-10-01", "1987-09-30"),
           ("2003-10-01", "2018-09-30")]
_WY = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
_WY_LABELS = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]

#: the physics runs: label -> the ladder run (None: the GA calibration)
RUNS: dict[str, str | None] = {
    "GA": None, "1_hru": "1_hru", "2_grid": "2_grid", "3_pt": "3_pt", "4_noah": "4_noah",
    "5_px": "5_px", "6_aef": "6_aef",
}
#: the LSTM ensembles on 5_px: label -> ensemble dir (seed*/checkpoints/best.pt averaged)
HYBRID: dict[str, Path] = {
    "hybrid": paths.dpl_run(run="hybrid"),
    "hybrid_dt": paths.dpl_run(run="hybrid_dt"),
    "lstm": paths.dpl_run(run="lstm"),
}

#: per-series line style (identity by hue; CalSim3 FNF in black, drawn on top).
STYLE: dict[str, dict] = {
    "CalSim3 FNF":    dict(color="#000000", lw=1.6, zorder=10),
    "GA":             dict(color="#9e9e9e", lw=1.1),
    "1_hru":          dict(color="#8c564b", lw=1.1),
    "2_grid":         dict(color="#1f77b4", lw=1.1),
    "3_pt":           dict(color="#ff7f0e", lw=1.1),
    "4_noah":         dict(color="#bcbd22", lw=1.1),
    "5_px":           dict(color="#2ca02c", lw=1.1),
    "6_aef":          dict(color="#e377c2", lw=1.1),
    "hybrid":         dict(color="#9467bd", lw=1.1),
    "hybrid_dt":      dict(color="#17becf", lw=1.1),
    "lstm":           dict(color="#7f7f7f", lw=1.1),
}

#: (tag, title, [model labels]) -- each renders one 11-basin figure.
COMPARISONS: list[tuple[str, str, list[str]]] = [
    ("a", "Network in place of the GA", ["GA", "1_hru"]),
    ("b", "Grid cells and CalSim3 outlines", ["1_hru", "2_grid"]),
    ("c", "Priestley–Taylor PET", ["2_grid", "3_pt"]),
    ("d", "Noah-type ET", ["3_pt", "4_noah"]),
    ("e", "Learned rain/snow threshold", ["4_noah", "5_px"]),
    ("f", "aef_u inputs", ["5_px", "6_aef"]),
    ("g", "LSTM models on 5_px", ["5_px", "hybrid", "hybrid_dt", "lstm"]),
]
#: the rows of the all-series skill figures: (axis label, value from a (KGE, NSE, pbias,
#: seasonal mismatch) tuple)
_SKILL_ROWS = [
    ("KGE", lambda m: m[0]),
    ("Absolute percent bias", lambda m: abs(m[2])),
    ("Seasonal mismatch (%)", lambda m: m[3] * 100.0),
]


# --------------------------------------------------------------------------- #
# daily model sims (mm/day, date x basin) -- all cached to CSV
# --------------------------------------------------------------------------- #
def _daily_run(run: str | None, data_dir: str) -> pd.DataFrame:
    """Daily flow (date x basin, mm/day) of a learned run as trained, or of the GA calibration."""
    if run is not None:
        from ..evaluate import basin_daily

        return basin_daily(run, data_dir=data_dir)
    from ...cdec15 import BASINS
    from ...model import run_basins

    return run_basins(list(BASINS), data_dir=data_dir, domain="15cdec")


def _daily_ensemble(ens_dir: str, data_dir: str, device, cache: Path) -> pd.DataFrame:
    """Full-record ENSEMBLE-MEAN daily hybrid flow (mm/day) — mean over all
    ``seed*/checkpoints/best.pt`` members; cached to CSV.  Data is loaded once
    (every seed shares the physics/variant/domain config)."""
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")
    import torch

    from ..hybrid.data import data_for
    from ..hybrid.evaluate import _build_model
    from ..hybrid.train import predict_days

    ckpts = sorted(Path(ens_dir).glob("seed*/checkpoints/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no seed*/checkpoints/best.pt under {ens_dir}")
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    data = data_for(ck0, data_dir, device)
    bb, tt = data.eval_days("all")
    accum = None
    for cp in ckpts:
        model = _build_model(torch.load(cp, map_location="cpu", weights_only=False),
                             data, device)
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
    from ...metrics import kge, nse, pbias, seasonal_mismatch
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
    from ...calsim.compare import build_anchor_long
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
    from ..._figures import folsom_before_yuba
    hru = load_hru_table(data_dir, domain="15cdec")
    lat = hru.groupby("basin")["lat"].mean().sort_values(ascending=False)
    order = folsom_before_yuba("15cdec", lat.index.tolist())
    return [b for b in order if b in keep]


def assemble(data_dir: str = "data", *, device: str = "cuda") -> dict:
    """Compute every series' combined-period WY mean-monthly regime (TAF).
    Returns ``clim`` (label -> DataFrame[12 x basin]) and ``order`` (the 11
    CalSim3-mapped basins, north->south)."""
    from ...calsim.catchments import basin_areas
    from ..config import pick_device

    areas = basin_areas(data_dir, domain="15cdec")
    cachedir = paths.local(name="cache/climatology")
    cachedir.mkdir(parents=True, exist_ok=True)

    monthly: dict[str, pd.DataFrame] = {}
    clim: dict[str, pd.DataFrame] = {}
    for label, run in RUNS.items():
        monthly[label] = _monthly_taf(_daily_run(run, data_dir), areas)
        clim[label] = _climatology(monthly[label])
        print(f"  assembled {label}", flush=True)

    dev = pick_device(device)
    for label, ens_dir in HYBRID.items():
        from ..evaluate import ensemble_key

        daily = _daily_ensemble(ens_dir, data_dir, dev, cachedir / (
            f"sim_{Path(ens_dir).name}_{ensemble_key(ens_dir)}.csv"))
        monthly[label] = _monthly_taf(daily, areas)
        clim[label] = _climatology(monthly[label])
        print(f"  assembled {label}", flush=True)

    obs = _calsim3_fnf(data_dir, cachedir / "calsim3_fnf_monthly.csv")
    clim["CalSim3 FNF"] = _climatology(obs)
    order = _basin_order(data_dir, list(obs.columns))

    # monthly KGE/NSE of each model vs CalSim3 FNF over the combined period
    metrics: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for label in [*RUNS, *HYBRID]:
        metrics[label] = {b: _score(monthly[label][b], obs[b])
                          for b in order if b in monthly[label] and b in obs}
    print(f"  observed: CalSim3 FNF for {order}", flush=True)
    return dict(clim=clim, order=order, metrics=metrics)


def _mean_range_bar(ax, x: float, vals: np.ndarray, width: float, color: str,
                    *, skill: bool = False) -> None:
    """A bar at the mean of ``vals`` with a whisker over their range.  On a skill axis
    (``skill``: 0 to 1) a mean or a minimum below 0 is cut at 0 and marked there with a
    triangle."""
    mean, lo, hi = float(vals.mean()), float(vals.min()), float(vals.max())
    if skill:
        if lo < 0:
            ax.plot([x], [0.0], "v", ms=4, color=color, clip_on=False, zorder=4)
        mean, lo, hi = max(mean, 0.0), max(lo, 0.0), min(hi, 1.0)
    ax.bar(x, mean, width=width, color=color, zorder=2)
    ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor="0.2",
                elinewidth=0.8, capsize=2, zorder=3)


def _bar_summary(ax, models: list[str], metrics: dict, order: list[str]) -> None:
    """The spare cell of a comparison figure: KGE and NSE per model, the mean over the 11
    watersheds with their range, model-coloured to match the lines."""
    n = len(models)
    width = 0.8 / n
    for gi in (0, 1):                                  # 0 -> KGE, 1 -> NSE
        for k, label in enumerate(models):
            vals = np.array([metrics[label][b][gi] for b in order
                             if b in metrics.get(label, {})
                             and np.isfinite(metrics[label][b][gi])])
            if vals.size:
                _mean_range_bar(ax, gi + (k - (n - 1) / 2) * width, vals, width * 0.9,
                                STYLE[label]["color"], skill=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["KGE", "NSE"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Skill")


def _plot_comparison(data: dict, models: list[str], title: str, path: Path) -> None:
    """One ladder step: the mean-monthly regime of each watershed (north to south) against
    CalSim3 FNF, and in the spare cell the step's KGE and NSE over the watersheds."""
    clim, order, metrics = data["clim"], data["order"], data["metrics"]
    series = ["CalSim3 FNF", *models]
    ncol = 4
    nrow = int(np.ceil((len(order) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5, 1.9 * nrow + 0.9), squeeze=False,
                             layout="constrained")
    raveled = axes.ravel()
    for ax, basin in zip(raveled, order, strict=False):
        for label in series:
            if basin in clim[label].columns:
                ax.plot(range(12), clim[label][basin].to_numpy(), **STYLE[label])
        ax.set_title(basin)
        ax.set_xticks([0, 3, 6, 9])
        ax.set_xticklabels(["O", "J", "A", "J"])
        ax.set_ylim(bottom=0)
    _bar_summary(raveled[len(order)], models, metrics, order)
    for ax in raveled[len(order) + 1:]:
        ax.axis("off")
    handles = [plt.Line2D([], [], label=s, **STYLE[s]) for s in series]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(title)
    fig.supylabel("Mean monthly flow (TAF)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _pin_below_zero(ax, xs, ys, color: str) -> None:
    """Triangles at 0 for the KGE values below the 0 to 1 axis."""
    neg = [x for x, y in zip(xs, ys, strict=True) if y < 0]
    if neg:
        ax.plot(neg, np.zeros(len(neg)), "v", ms=4, color=color, ls="none", clip_on=False,
                zorder=4)


def _plot_metrics_bars(data: dict, path: Path) -> None:
    """Every series per watershed on the three metrics of :data:`_SKILL_ROWS`, one metric per
    row."""
    metrics, order = data["metrics"], data["order"]
    models = [*RUNS, *HYBRID]
    n = len(models)
    width = 0.8 / n
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True, layout="constrained")
    for ax, (name, fn) in zip(axes, _SKILL_ROWS, strict=True):
        for k, label in enumerate(models):
            xs, ys = [], []
            for i, b in enumerate(order):
                mv = metrics.get(label, {}).get(b)
                if mv is not None and np.isfinite(fn(mv)):
                    xs.append(i + (k - (n - 1) / 2) * width)
                    ys.append(fn(mv))
            ax.bar(xs, np.clip(ys, 0.0, None), width=width, color=STYLE[label]["color"],
                   label=label)
            if name == "KGE":
                _pin_below_zero(ax, xs, ys, STYLE[label]["color"])
        ax.set_ylabel(name)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(bottom=0.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Monthly skill by watershed")
    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels(order)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=(n + 1) // 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _plot_metrics_bars_agg(data: dict, path: Path) -> None:
    """Every series over the 11 watersheds on the three metrics of :data:`_SKILL_ROWS`: the
    mean with the range, one metric per row."""
    metrics, order = data["metrics"], data["order"]
    models = [*RUNS, *HYBRID]
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.0), sharex=True, layout="constrained")
    for ax, (name, fn) in zip(axes, _SKILL_ROWS, strict=True):
        for k, label in enumerate(models):
            vals = np.array([fn(metrics[label][b]) for b in order
                             if b in metrics.get(label, {})
                             and np.isfinite(fn(metrics[label][b]))])
            if vals.size:
                _mean_range_bar(ax, k, vals, 0.7, STYLE[label]["color"], skill=name == "KGE")
        ax.set_ylabel(name)
        ax.set_ylim(bottom=0.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Monthly skill, 11 watersheds")
    axes[-1].set_xticks(range(len(models)))
    axes[-1].set_xticklabels(models, rotation=45, ha="right", rotation_mode="anchor")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def make_cdec15_climatology(data_dir: str = "data",
                            out_dir: str | Path | None = None,
                            *, device: str = "cuda") -> dict:
    """Assemble every series and render the seven ladder figures + the
    all-series metric-bar summaries (per-basin and basin-aggregated) into ``out_dir``
    (default: the study's result folder)."""
    data = assemble(data_dir, device=device)
    figdir = Path(out_dir) if out_dir is not None else paths.dpl_study(name="climatology")
    for tag, title, models in COMPARISONS:
        _plot_comparison(data, models, title, figdir / f"climatology_{tag}.png")
    _plot_metrics_bars(data, figdir / "climatology_summary.png")
    _plot_metrics_bars_agg(data, figdir / "climatology_summary_agg.png")
    return data
