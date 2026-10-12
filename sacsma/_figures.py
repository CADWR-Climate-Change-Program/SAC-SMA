"""Shared cal/val figure machinery used by both applications' diagnostics.

Domain-agnostic builders (parameterized by unit / labels / calibration window):
per-basin diagnostics, the domain skill summary, and the MATLAB parity figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# House style: Arial 10 pt for every text, no top or right spine, no grid, legends without a
# frame. Titles are short and name the subject; notes and sources go in the README, not on
# the figure.
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.titlesize": 10, "figure.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False,
})

from .metrics import kge, nse, pbias, pearson  # noqa: E402

#: House figure-axis ordering override: list the Folsom (American R) basin directly before
#: the Yuba, where the strict north->south latitude sort would put the Yuba first (its HRUs
#: sit slightly further north).  domain -> (folsom_basin, yuba_basin).
FOLSOM_BEFORE_YUBA = {
    "15cdec": ("FOL", "YRS"), "11obs": ("AMF", "YRS"), "12rim": ("FOL_I", "SMART"),
}


def folsom_before_yuba(domain: str, order: list[str]) -> list[str]:
    """Apply :data:`FOLSOM_BEFORE_YUBA` to a north->south basin list (edited in place)."""
    pair = FOLSOM_BEFORE_YUBA.get(domain)
    if pair and set(pair) <= set(order) and order.index(pair[0]) > order.index(pair[1]):
        order.insert(order.index(pair[1]), order.pop(order.index(pair[0])))
    return order


def _period_stats(sim: np.ndarray, obs: np.ndarray) -> dict:
    """Skill on finite (sim, obs) pairs; NaN-safe. Returns {} if no obs."""
    mask = np.isfinite(sim) & np.isfinite(obs)
    if mask.sum() == 0:
        return {"n": 0}
    s, o = sim[mask], obs[mask]
    return {
        "n": int(mask.sum()), "kge": kge(s, o), "nse": nse(s, o),
        "pbias": pbias(s, o), "r": pearson(s, o),
        "sim_mean": float(s.mean()), "obs_mean": float(o.mean()),
    }


def _stat_text(label: str, st: dict) -> str:
    """One line of a period's skill for a figure: its KGE and percent bias."""
    if st.get("n", 0) == 0:
        return f"{label}: no observations"
    return f"{label}  KGE {st['kge']:.2f}, bias {st['pbias']:+.0f} %"


def basin_diagnostics_fig(basin: str, m: pd.DataFrame, cal_end: pd.Timestamp,
                          cal: dict, val: dict, out: Path, unit: str = "mm/day",
                          obs_label: str = "Observed",
                          cal_start: pd.Timestamp | None = None) -> None:
    """Per-basin figure: the whole time series with the validation span shaded, and the
    mean-monthly regimes of calibration and validation on a shared y-scale.

    Calibration is the window ``[cal_start, cal_end]`` (validation = everything outside,
    including pre-calibration years); if ``cal_start`` is None, calibration is everything up
    to ``cal_end``.  Each period's KGE and bias sit above the series, calibration on the side
    of its window (always the left when ``cal_start`` is None: the 15cdec records start in
    different years, so a midpoint test would flip sides from basin to basin).
    """
    from matplotlib.patches import Patch

    d = m["date"]
    sim = m["flow_sim"].to_numpy()
    obs = m["flow_obs"].to_numpy()  # NaN where missing / outside obs record
    finite = np.isfinite(obs) & np.isfinite(sim)

    fig = plt.figure(figsize=(6.5, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
    fig.suptitle(basin)

    axts = fig.add_subplot(gs[0, :])
    axts.plot(d, obs, color="0.25", lw=0.7, label=obs_label)
    axts.plot(d, sim, color="tab:red", lw=0.7, alpha=0.8, label="Simulated")
    # shade the validation span(s) inside the plotted range and pin the x-limits to the data
    dmin, dmax = d.min(), d.max()
    if cal_start is not None and cal_start > dmin:
        axts.axvspan(dmin, min(cal_start, dmax), color="tab:blue", alpha=0.07, lw=0)
    if cal_end < dmax:
        axts.axvspan(max(cal_end, dmin), dmax, color="tab:blue", alpha=0.07, lw=0)
    axts.set_xlim(dmin, dmax)
    axts.set_ylabel(f"Flow ({unit})")
    ydata = np.nanmax([np.nanmax(sim), np.nanmax(obs)])
    axts.set_ylim(0, ydata * 1.3)
    handles, _ = axts.get_legend_handles_labels()
    handles.append(Patch(color="tab:blue", alpha=0.15, label="Validation"))
    axts.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    if cal_start is None:
        cal_on_left = True
    else:
        cal_mid = cal_start + (cal_end - cal_start) / 2
        cal_on_left = cal_mid < dmin + (dmax - dmin) / 2
    side = {True: (0.0, "left"), False: (1.0, "right")}
    for label, st, left in (("Calibration", cal, cal_on_left), ("Validation", val, not cal_on_left)):
        if label == "Validation" and st.get("n", 0) == 0:
            continue                     # too little out-of-calibration data (e.g. BLB)
        x, ha = side[left]
        axts.text(x, 1.02, _stat_text(label, st), transform=axts.transAxes, ha=ha, va="bottom")

    # mean-monthly regimes (on finite obs), water-year order Oct..Sep
    wy_months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    in_cal = d <= cal_end
    if cal_start is not None:
        in_cal = in_cal & (d >= cal_start)
    in_cal = in_cal.to_numpy()
    x = np.arange(12)
    axes_rg = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for axrg, mask, lab in zip(axes_rg, (finite & in_cal, finite & ~in_cal),
                               ("Calibration", "Validation")):
        if mask.sum():
            mf = m.loc[mask].assign(month=d[mask].dt.month)
            mm = mf.groupby("month")[["flow_obs", "flow_sim"]].mean().reindex(wy_months)
            axrg.plot(x, mm["flow_obs"], color="0.25", lw=1.4)
            axrg.plot(x, mm["flow_sim"], color="tab:red", lw=1.4)
        else:
            axrg.text(0.5, 0.5, "No observations", transform=axrg.transAxes,
                      ha="center", va="center", color="0.5")
        axrg.set_title(lab)
        axrg.set_xticks([0, 3, 6, 9])
        axrg.set_xticklabels(["O", "J", "A", "J"])
    axes_rg[0].set_ylabel(f"Mean ({unit})")
    top = max(ax_.get_ylim()[1] for ax_ in axes_rg)
    for ax_ in axes_rg:
        ax_.set_ylim(0, top)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# cal/val share these two colors across every panel/figure that shows them.
_CAL_COLOR = "tab:blue"
_VAL_COLOR = "tab:orange"
# Fixed pbias axis for skill_summary_fig, shared across all four calibration sets (15cdec,
# 9unimp, 11obs, 12rim) so bar heights are comparable set to set; sized to the largest
# committed |pbias| (15cdec SCC validation, ~69%) with headroom.
_SKILL_PBIAS_YLIM = 75.0


def skill_summary_fig(metrics: pd.DataFrame, out: Path) -> None:
    """KGE and percent bias (cal vs val) across all basins.

    House scale rules: the KGE panel is always the full 0–1 scale (a value below 0 is pinned
    to 0 with a triangle) so the domains' figures compare directly; the pbias panel uses one
    fixed scale (:data:`_SKILL_PBIAS_YLIM`) shared across all four calibration sets.  cal/val
    use the same two colors (:data:`_CAL_COLOR`/:data:`_VAL_COLOR`) in both panels.
    """
    metrics = metrics.copy()
    for c in ["cal_kge", "val_kge", "cal_pbias", "val_pbias"]:
        if c in metrics:
            metrics[c] = pd.to_numeric(metrics[c], errors="coerce")  # None -> NaN (skipped)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(2, 1, figsize=(min(6.5, max(4.0, 1.4 + 0.3 * len(x))), 5.5),
                           sharex=True)
    w = 0.38
    for off, col, label, color in ((-0.5, "cal_kge", "Calibration", _CAL_COLOR),
                                   (0.5, "val_kge", "Validation", _VAL_COLOR)):
        v = metrics[col].to_numpy(float)
        ax[0].bar(x + off * w, np.clip(v, 0, None), w, label=label, color=color)
        neg = np.isfinite(v) & (v < 0)
        if neg.any():
            ax[0].plot(x[neg] + off * w, np.zeros(neg.sum()), "v", ms=5, color=color,
                       clip_on=False)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("KGE")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    ax[1].bar(x - w / 2, metrics["cal_pbias"], w, color=_CAL_COLOR)
    ax[1].bar(x + w / 2, metrics["val_pbias"], w, color=_VAL_COLOR)
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].set_ylim(-_SKILL_PBIAS_YLIM, _SKILL_PBIAS_YLIM)
    ax[1].set_ylabel("Percent bias")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(metrics["basin"], rotation=90)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def parity_fig(parity: dict, out: Path) -> None:
    """The Python port against the MATLAB reference.

    ``parity`` = {basin: DataFrame[date, flow_sim, flow_ref]} over the full 1915–2018
    simulation period: the basin with the largest daily difference over its wettest four
    years, a pooled 1:1 scatter, and the largest daily difference per basin.
    """
    import matplotlib.dates as mdates

    maxdiff = {b: float(np.abs(df["flow_sim"] - df["flow_ref"]).max()) for b, df in parity.items()}
    worst = max(maxdiff, key=lambda b: maxdiff[b])
    alls = np.concatenate([df["flow_sim"].to_numpy() for df in parity.values()])
    allr = np.concatenate([df["flow_ref"].to_numpy() for df in parity.values()])

    fig = plt.figure(figsize=(6.5, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    fig.suptitle("Python port and MATLAB reference")

    axo = fig.add_subplot(gs[0, :])
    dfw = parity[worst]
    peak_year = dfw.loc[dfw["flow_ref"].idxmax(), "date"].year
    win = dfw[(dfw["date"].dt.year >= peak_year - 1) & (dfw["date"].dt.year <= peak_year + 2)]
    axo.plot(win["date"], win["flow_ref"], color="black", lw=2.4, alpha=0.5, label="MATLAB")
    axo.plot(win["date"], win["flow_sim"], color="tab:red", lw=0.9, ls="--", label="Python")
    axo.set_title(worst)
    axo.xaxis.set_major_locator(mdates.YearLocator())
    axo.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axo.set_ylabel("Flow (mm/day)")
    axo.legend(loc="upper right")

    axs = fig.add_subplot(gs[1, 0])
    n = allr.size
    idx = np.linspace(0, n - 1, min(n, 30000)).astype(int)
    hi = max(allr.max(), alls.max())
    axs.scatter(allr[idx], alls[idx], s=3, alpha=0.2, color="tab:blue", edgecolors="none")
    axs.plot([0, hi], [0, hi], color="0.4", lw=1, ls="--")
    axs.set_xlim(0, hi)
    axs.set_ylim(0, hi)
    axs.set_title(f"All basins, KGE {kge(alls, allr):.5f}")
    axs.set_xlabel("MATLAB (mm/day)")
    axs.set_ylabel("Python (mm/day)")

    axb = fig.add_subplot(gs[1, 1])
    items = sorted(maxdiff.items(), key=lambda kv: kv[1])
    axb.barh([k for k, _ in items], [v for _, v in items], color="tab:gray")
    axb.set_title("Largest daily difference")
    axb.set_xlabel("mm/day")

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
