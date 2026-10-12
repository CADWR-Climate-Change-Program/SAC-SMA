"""Skill and climate response of every rung of the ladder (the GA parameters, ``1_hru`` to
``7_calsim``) and of the three LSTM models (``hybrid``, ``hybrid_dt``, ``lstm``).

Skill is the mean KGE over the 15 CDEC watersheds on the training years WY1989-2003 and the
test years WY2004-18 (daily, each run's ``metrics.csv``; the GA's from the ladder's read of the
same two windows, ``readout_2004_2018_retrained.csv``, since its own table splits at the end of
WY2003 and starts with each record), and on the held-out WY1976-85: monthly, at the 13 outlets
and on the regional records (USGS gauges, unimpaired-flow subbasins, CalSim3 arcs). The
held-out numbers are read from the tracked read records (``provenance/read_WY1976-85*.csv``),
never recomputed.

Response is the % change of annual and April-July runoff against the run's own present
climate, at +3 degC and at precipitation -10 % and +10 %, per watershed on the response window
of :mod:`.surfaces` (WY1976-85 left out), then averaged over the 15 watersheds. The GA runs its
optimum on the shifted forcing (:func:`sacsma.model.run_basins`), each learned rung runs as
trained (:func:`sacsma.dpl.evaluate.basin_daily`, cached, CPU); the LSTM models' numbers are
those of the hybrids study (``hybrids_metrics.csv``).

Output (``artifacts/results/dpl/studies/ladder/``): ``ladder.csv``, ``response.csv`` and the
figures ``skill.png``, ``response.png``, ``skill_response.png`` and ``regional.png``.
"""
from __future__ import annotations

import dataclasses as dc
from pathlib import Path

import numpy as np
import pandas as pd

from ... import paths
from ..._figures import plt
from ...io import load_basin_area
from .climatology import _basin_order
from .surfaces import metrics_from_daily

DOMAIN = paths.CDEC15
RUNGS = ("GA", "1_hru", "2_grid", "3_pt", "4_noah", "5_px", "6_aef", "7_calsim")
LSTMS = ("hybrid", "hybrid_dt", "lstm")
RUNS = RUNGS + LSTMS
#: (dp, dt): present climate, +3 degC, precipitation -10 % and +10 %
CLIMATES = ((0.0, 0.0), (0.0, 3.0), (-0.1, 0.0), (0.1, 0.0))
#: the regional families of the held-out read: (set in the read record, label)
FAMILIES = (("usgs", "49 USGS gauges"), ("uf", "9 UF subbasins"), ("arcs", "64 CalSim3 arcs"))
#: the name of a run in the ladder's read records, where it differs
_READ_NAME = {"GA": "0_ga"}
#: the hybrids study's label of each LSTM model
_HYBRIDS_LABEL = {"hybrid": "Hybrid", "hybrid_dt": "Hybrid DT", "lstm": "LSTM"}
_WINDOWS = (("kge_training", "WY1989–2003 (training)", "tab:blue"),
            ("kge_test", "WY2004–18 (test)", "tab:orange"),
            ("kge_heldout", "WY1976–85 (held out)", "0.45"))
_W, _DPI = 6.5, 300


def _out(out_dir: str | Path | None) -> Path:
    return Path(out_dir) if out_dir is not None else paths.dpl_study(name="ladder")


# ------------------------------------------------------------------------------ skill
def skill() -> pd.DataFrame:
    """One row per run (:data:`RUNS`): mean KGE on the training, test and held-out windows,
    and on the held-out regional records (``kge_<family>``); NaN where a run has no score."""
    prov = paths.dpl_run(run="7_calsim", domain=paths.MULTIFAMILY) / "provenance"
    readout = (pd.read_csv(prov / "readout_2004_2018_retrained.csv")
               .groupby(["run", "window"])["kge"].mean())
    outlets = pd.read_csv(prov / "read_WY1976-85_outlets.csv").groupby("run")["kge"].mean()
    region = pd.read_csv(prov / "read_WY1976-85_region.csv").groupby(["run", "set"])["kge"].mean()
    rows = []
    for run in RUNS:
        name = _READ_NAME.get(run, run)
        row = {"run": run}
        if run == "GA":
            row["kge_training"] = readout[(name, "WY1989-2003")]
            row["kge_test"] = readout[(name, "WY2004-18")]
        elif run != "7_calsim":
            m = pd.read_csv(paths.dpl_metrics(run=run))
            row["kge_training"], row["kge_test"] = m["cal_kge"].mean(), m["val_kge"].mean()
        if run in LSTMS:
            row["kge_heldout"] = pd.read_csv(paths.dpl_run(run=run) / "provenance"
                                             / "read_WY1976-85.csv")["kge"].mean()
        else:
            row["kge_heldout"] = outlets[name]
        for fam, _ in FAMILIES:
            row[f"kge_{fam}"] = region.get((name, fam), np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ response
def _shifted(forcing, dp: float, dt: float):
    """The domain forcing with precipitation x (1 + dp) and the temperatures + dt."""
    return dc.replace(forcing, prcp=forcing.prcp * (1.0 + dp), tavg=forcing.tavg + dt,
                      tmin=None if forcing.tmin is None else forcing.tmin + dt,
                      tmax=None if forcing.tmax is None else forcing.tmax + dt)


def response(data_dir: str = "data") -> pd.DataFrame:
    """One row per run, watershed and climate (:data:`CLIMATES`): annual and April-July runoff
    (``annual``, ``freshet``, TAF) and their % change against the run's present climate."""
    from ...model import load_domain_forcing, run_basins
    from ..evaluate import basin_daily

    areas = load_basin_area(data_dir, domain=DOMAIN).set_index("basin")["area_mi2"].to_dict()
    basins = _basin_order(data_dir, list(areas))
    forcing = load_domain_forcing(data_dir, domain=DOMAIN)
    rows = []
    for run in RUNGS:
        for dp, dt in CLIMATES:
            daily = (run_basins(basins, data_dir=data_dir, domain=DOMAIN,
                                forcing=_shifted(forcing, dp, dt)) if run == "GA"
                     else basin_daily(run, data_dir=data_dir, dp=dp, dt=dt))
            m = metrics_from_daily(daily[basins], areas).loc[basins, ["annual", "freshet"]]
            rows.append(m.rename_axis("basin").reset_index().assign(run=run, dp=dp, dt=dt))
        print(f"  {run}: {len(CLIMATES)} climates", flush=True)
    hyb = pd.read_csv(paths.dpl_study(name="hybrids") / "hybrids_metrics.csv")
    hyb = hyb.assign(run=hyb["model"].map({v: k for k, v in _HYBRIDS_LABEL.items()}),
                     dp=hyb["dp"].round(2), dt=hyb["dt"].round(2))
    for run in LSTMS:
        for dp, dt in CLIMATES:
            s = hyb[(hyb["run"] == run) & (hyb["dp"] == dp) & (hyb["dt"] == dt)]
            rows.append(s.set_index("basin").loc[basins, ["annual", "freshet"]]
                        .reset_index().assign(run=run, dp=dp, dt=dt))
    t = pd.concat(rows, ignore_index=True)[["run", "basin", "dp", "dt", "annual", "freshet"]]
    base = t[(t["dp"] == 0) & (t["dt"] == 0)].set_index(["run", "basin"])
    key = pd.MultiIndex.from_frame(t[["run", "basin"]])
    for k in ("annual", "freshet"):
        t[f"pct_{k}"] = 100.0 * (t[k].to_numpy() / base[k].reindex(key).to_numpy() - 1.0)
    return t


def response_summary(resp: pd.DataFrame) -> pd.DataFrame:
    """Per run, the mean over the watersheds of the % changes: annual and April-July at +3 degC,
    annual at precipitation -10 % and +10 %."""
    mean = resp.groupby(["run", "dp", "dt"])[["pct_annual", "pct_freshet"]].mean()
    return pd.DataFrame({
        "annual_pct_t+3": mean.xs((0.0, 3.0), level=("dp", "dt"))["pct_annual"],
        "aprjul_pct_t+3": mean.xs((0.0, 3.0), level=("dp", "dt"))["pct_freshet"],
        "annual_pct_p-10": mean.xs((-0.1, 0.0), level=("dp", "dt"))["pct_annual"],
        "annual_pct_p+10": mean.xs((0.1, 0.0), level=("dp", "dt"))["pct_annual"],
    }).reindex(list(RUNS)).rename_axis("run").reset_index()


# ------------------------------------------------------------------------------ figures
def _x(runs) -> np.ndarray:
    """Bar positions: the rungs, a half-step gap, the LSTM models."""
    return np.array([i + 0.5 * (r in LSTMS) for i, r in enumerate(runs)], float)


def _kge_bars(ax, x, vals, w, color, label) -> None:
    """KGE bars on the 0-1 axis; a value below 0 is pinned to 0 with a triangle."""
    v = np.asarray(vals, float)
    ax.bar(x, np.clip(v, 0, None), w, color=color, label=label)
    neg = np.isfinite(v) & (v < 0)
    ax.plot(x[neg], np.zeros(neg.sum()), "v", ms=5, color=color, clip_on=False)


def _run_ticks(ax, runs) -> None:
    ax.set_xticks(_x(runs))
    ax.set_xticklabels(runs, rotation=90)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def skill_fig(tbl: pd.DataFrame, path: Path) -> None:
    """Mean KGE by run on the three windows."""
    fig, ax = plt.subplots(figsize=(_W, 4.4))
    x, w = _x(tbl["run"]), 0.27
    for i, (col, label, color) in enumerate(_WINDOWS):
        _kge_bars(ax, x + (i - 1) * w, tbl[col], w, color, label)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean KGE")
    ax.set_title("Skill by run")
    _run_ticks(ax, tbl["run"])
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, path)


def response_fig(tbl: pd.DataFrame, path: Path) -> None:
    """Annual and April-July change at +3 degC, and annual change at precipitation +-10 %."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(_W, 5.6), sharex=True)
    x, w = _x(tbl["run"]), 0.38
    for ax, cols, title, ylim, loc in (
            (ax1, (("annual_pct_t+3", "Annual", "0.3"), ("aprjul_pct_t+3", "April–July", "0.7")),
             "+3 °C", (-40, 15), "upper left"),
            (ax2, (("annual_pct_p-10", "−10 %", "#b2182b"),
                   ("annual_pct_p+10", "+10 %", "#2166ac")),
             "Precipitation ±10 %", (-25, 30), "upper right")):
        for off, (col, label, color) in zip((-0.5, 0.5), cols, strict=True):
            ax.bar(x + off * w, tbl[col], w, color=color, label=label)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_ylim(*ylim)
        ax.set_ylabel("Runoff change (%)")
        ax.set_title(title)
        ax.legend(loc=loc, ncol=2)
    _run_ticks(ax2, tbl["run"])
    fig.tight_layout()
    _save(fig, path)


def _label_points(ax, xs, ys, names, ms: float) -> None:
    """Each point's name beside it, at the nearest spot (looking first away from the point's
    neighbours) where the name and its line keep clear of the other names, lines, points and the
    legend, inside the axes; a thin line joins a name set further off to its point."""
    from matplotlib.text import Text

    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px_pt = fig.dpi / 72
    pts = ax.transData.transform(np.c_[xs, ys])
    r = (0.5 * ms + 1) * px_pt                              # a point's radius, in pixels
    boxes = [tuple(ax.get_legend().get_window_extent(rend).extents)] if ax.get_legend() else []
    drawn: list[np.ndarray] = [p[None, :] for p in pts]     # points and sampled lines (pixels)
    frame = ax.get_window_extent(rend)

    def clear(samples, bx, gap) -> bool:
        return not ((samples[:, 0] > bx[0] - gap) & (samples[:, 0] < bx[2] + gap)
                    & (samples[:, 1] > bx[1] - gap) & (samples[:, 1] < bx[3] + gap)).any()

    near = [[j for j, q in enumerate(pts) if j != i and np.hypot(*(p - q)) < 60]
            for i, p in enumerate(pts)]
    for i in sorted(range(len(pts)), key=lambda k: -len(near[k])):
        v = pts[i] - pts[near[i]].mean(axis=0) if near[i] else np.array([1.0, 0.0])
        away = np.degrees(np.arctan2(v[1], v[0]))
        angles = sorted(range(0, 360, 15), key=lambda a: abs((a - away + 180) % 360 - 180))
        others = [d for k, d in enumerate(drawn) if k != i]
        placed = False
        for d in (6, 18, 30, 42, 54):
            for ang in angles:
                dx, dy = d * np.cos(np.radians(ang)), d * np.sin(np.radians(ang))
                ha = "left" if dx > 2 else "right" if dx < -2 else "center"
                va = "bottom" if dy > 2 else "top" if dy < -2 else "center"
                t = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy),
                                textcoords="offset points", ha=ha, va=va,
                                arrowprops=None if d == 6 else dict(
                                    arrowstyle="-", color="0.5", lw=0.6, shrinkA=1,
                                    shrinkB=r / px_pt))
                t.update_positions(rend)
                e = Text.get_window_extent(t, rend)     # the name, not its line
                bx = (e.x0, e.y0, e.x1, e.y1)
                end = pts[i] + np.array([dx, dy]) * px_pt
                seg = pts[i] + np.linspace(0, 1, 40)[:, None] * (end - pts[i])
                seg = seg[np.hypot(*(seg - pts[i]).T) > r]
                ok = (frame.x0 <= bx[0] and bx[2] <= frame.x1 and frame.y0 <= bx[1]
                      and bx[3] <= frame.y1
                      and all(clear(o, bx, r + 2) for o in others)
                      and all(b[2] < bx[0] - 2 or bx[2] < b[0] - 2 or b[3] < bx[1] - 2
                              or bx[3] < b[1] - 2 for b in boxes)
                      and (d == 6 or (all(clear(seg, b, 2) for b in boxes)
                                      and all(np.min(np.hypot(*(o[:, None, :] - seg[None]).T))
                                              > (r + 2 if len(o) == 1 else 4) for o in others))))
                if ok:
                    boxes.append(bx)
                    if d > 6:
                        drawn[i] = np.vstack([pts[i][None, :], seg])
                    placed = True
                    break
                t.remove()
            if placed:
                break


def skill_response_fig(tbl: pd.DataFrame, path: Path) -> None:
    """Held-out mean KGE against the annual change at +3 degC, one labelled point per run."""
    fig, ax = plt.subplots(figsize=(_W, 5.4))
    ms = 5
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    for runs, label, kw in ((RUNGS, "Rungs", dict(marker="o", color="0.2")),
                            (LSTMS, "LSTM models", dict(marker="s", mfc="white", mec="0.2"))):
        s = tbl[tbl["run"].isin(runs)]
        ax.plot(s["annual_pct_t+3"], s["kge_heldout"], ls="none", ms=ms, label=label, **kw)
    ax.set_ylim(0, 1)
    lo, hi = tbl["annual_pct_t+3"].min(), tbl["annual_pct_t+3"].max()
    ax.set_xlim(lo - 0.2 * (hi - lo), hi + 0.15 * (hi - lo))
    ax.set_xlabel("Annual runoff change at +3 °C (%)")
    ax.set_ylabel("Mean KGE, WY1976–85")
    ax.set_title("Held-out skill and warming response")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _label_points(ax, tbl["annual_pct_t+3"].to_numpy(), tbl["kge_heldout"].to_numpy(),
                  list(tbl["run"]), ms)
    _save(fig, path)


def regional_fig(tbl: pd.DataFrame, path: Path) -> None:
    """Held-out mean KGE by run on the three regional record families."""
    tbl = tbl.dropna(subset=[f"kge_{f}" for f, _ in FAMILIES])
    fig, ax = plt.subplots(figsize=(_W, 3.8))
    x, w = np.arange(len(tbl)), 0.27
    colors = ("#1b9e77", "#7570b3", "0.45")
    for i, ((fam, label), color) in enumerate(zip(FAMILIES, colors, strict=True)):
        _kge_bars(ax, x + (i - 1) * w, tbl[f"kge_{fam}"], w, color, label)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean KGE")
    ax.set_title("Regional records, WY1976–85")
    ax.set_xticks(x)
    ax.set_xticklabels(tbl["run"])
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)


def make_ladder(data_dir: str = "data", out_dir: str | Path | None = None) -> pd.DataFrame:
    """Write ``ladder.csv``, ``response.csv`` and the four figures into ``out_dir`` (default:
    the study's result folder); returns the ladder table."""
    out = _out(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resp = response(data_dir)
    resp.round(4).to_csv(out / "response.csv", index=False)
    tbl = skill().merge(response_summary(resp), on="run")
    tbl.round(4).to_csv(out / "ladder.csv", index=False)
    skill_fig(tbl, out / "skill.png")
    response_fig(tbl, out / "response.png")
    skill_response_fig(tbl, out / "skill_response.png")
    regional_fig(tbl, out / "regional.png")
    print(tbl.round(3).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return tbl
