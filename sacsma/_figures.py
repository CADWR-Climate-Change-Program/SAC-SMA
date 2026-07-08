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

# House style: nothing larger than 8pt (titles/labels included), small ticks/legends.
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.titlesize": 8,
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
    if st.get("n", 0) == 0:
        return f"{label}: no observations"
    return (f"{label} (n={st['n']:,})\n"
            f"KGE={st['kge']:.3f}  NSE={st['nse']:.3f}\n"
            f"pbias={st['pbias']:+.1f}%  r={st['r']:.3f}")


def basin_diagnostics_fig(basin: str, m: pd.DataFrame, cal_end: pd.Timestamp,
                          cal: dict, val: dict, out: Path, unit: str = "mm/day",
                          obs_label: str = "gage (observed)",
                          cal_start: pd.Timestamp | None = None,
                          title_obs: str | None = None) -> None:
    """Per-basin figure: full cal/val time series + a mean-monthly regime pair
    (calibration and validation on a shared y-scale).

    Calibration is the window ``[cal_start, cal_end]`` (validation = everything
    outside, including pre-calibration years); if ``cal_start`` is None, calibration
    is everything up to ``cal_end``.  ``title_obs`` names the observed basis in the
    suptitle (e.g. ``"FNF Flow"``, ``"CalSim3 FNF Flow"``); default keeps the
    step-based wording (``observed FNF flow`` / ``observed gage flow``).
    """
    d = m["date"]
    sim = m["flow_sim"].to_numpy()
    obs = m["flow_obs"].to_numpy()  # NaN where missing / outside obs record
    finite = np.isfinite(obs) & np.isfinite(sim)
    step = "Daily" if unit == "mm/day" else "Monthly"
    if title_obs is None:
        title_obs = f"observed {('FNF' if step == 'Monthly' else 'gage')} flow"

    fig = plt.figure(figsize=(6.5, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])  # taller top; squarer bottom panels
    fig.suptitle(f"{basin} — SAC-SMA simulated vs. {title_obs}",
                 fontsize=8, fontweight="bold")

    # --- top row (spans all 3 cols): cal/val time series ---
    axts = fig.add_subplot(gs[0, :])
    axts.plot(d, obs, color="0.25", lw=0.7, label=obs_label)
    axts.plot(d, sim, color="tab:red", lw=0.7, alpha=0.8, label="simulated")
    # shade the VALIDATION span(s) within the plotted (observed) range; calibration window is
    # left clear.  Clamp markers to [dmin, dmax] and pin the x-limits to the data so the axis
    # never extends past the observed record (e.g. when obs ends before the calibration end).
    dmin, dmax = d.min(), d.max()
    if cal_start is not None and cal_start > dmin:
        axts.axvspan(dmin, min(cal_start, dmax), color="tab:blue", alpha=0.05)
        axts.axvline(cal_start, color="tab:blue", lw=1.2, ls="--")
    if cal_end < dmax:
        axts.axvspan(max(cal_end, dmin), dmax, color="tab:blue", alpha=0.05)
        axts.axvline(cal_end, color="tab:blue", lw=1.2, ls="--")
    axts.set_xlim(dmin, dmax)
    axts.set_ylabel(f"flow ({unit})")
    axts.set_title(f"{step} flow — calibration (unshaded) vs validation (shaded)")
    # headroom so the stat boxes/legend sit above the data
    ydata = np.nanmax([np.nanmax(sim), np.nanmax(obs)])
    axts.set_ylim(top=ydata * 1.35)
    axts.legend(loc="upper center", fontsize=8, ncol=2)
    # per-period stat boxes in the top corners (clear of the centered legend): place each box
    # over its OWN period — CAL on the side of the calibration window's centre, VAL opposite.
    # cal_start is None only for 15cdec, where "calibration" is unbounded-below (everything up
    # to cal_end, per-basin obs coverage varies) — always CAL-left/VAL-right there rather than
    # comparing cal_end to the plotted range's midpoint, which flips per-basin depending on how
    # much post-cal_end validation data exists.  The rim domains (bounded [cal_start, cal_end]
    # window, possibly late in the record) keep the dynamic midpoint comparison.
    if cal_start is None:
        cal_on_left = True
    else:
        cal_mid = cal_start + (cal_end - cal_start) / 2
        cal_on_left = cal_mid < dmin + (dmax - dmin) / 2
    cal_x, cal_ha = (0.01, "left") if cal_on_left else (0.99, "right")
    val_x, val_ha = (0.99, "right") if cal_on_left else (0.01, "left")
    _box = dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9)
    axts.text(cal_x, 0.97, _stat_text("CAL", cal), transform=axts.transAxes,
              va="top", ha=cal_ha, fontsize=8, family="monospace", bbox=_box)
    if val.get("n", 0) > 0:  # omit the VAL box when there is too little out-of-cal data (e.g. BLB)
        axts.text(val_x, 0.97, _stat_text("VAL", val), transform=axts.transAxes,
                  va="top", ha=val_ha, fontsize=8, family="monospace", bbox=_box)

    # --- mean-monthly regimes (on finite obs), water-year order Oct..Sep:
    # calibration and validation side by side on a shared y-scale ---
    wy_months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    wy_labels = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]
    in_cal = d <= cal_end
    if cal_start is not None:
        in_cal = in_cal & (d >= cal_start)
    in_cal = in_cal.to_numpy()
    x = np.arange(12)
    axes_rg = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for axrg, mask, lab in zip(axes_rg, (finite & in_cal, finite & ~in_cal),
                               ("calibration", "validation")):
        if mask.sum():
            mf = m.loc[mask].assign(month=d[mask].dt.month)
            mm = mf.groupby("month")[["flow_obs", "flow_sim"]].mean().reindex(wy_months)
            axrg.plot(x, mm["flow_obs"], "o-", color="0.25", ms=3.5, label="obs")
            axrg.plot(x, mm["flow_sim"], "s-", color="tab:red", ms=3.5, label="sim")
            axrg.legend(fontsize=8)
        else:
            axrg.text(0.5, 0.5, "no observations", transform=axrg.transAxes,
                      ha="center", va="center", color="0.5")
        axrg.set_title(f"Mean monthly regime — {lab} (water year)")
        axrg.set_xlabel("water-year month"); axrg.set_ylabel(f"flow ({unit})")
        axrg.set_xticks(x); axrg.set_xticklabels(wy_labels)
    top = max(ax_.get_ylim()[1] for ax_ in axes_rg)
    for ax_ in axes_rg:
        ax_.set_ylim(0, top)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
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

    House scale rules: the KGE panel is always the full 0–1 scale (values
    below 0 are clipped and marked with a ``↓``) so the domains' figures compare
    directly; the pbias panel uses one **fixed** scale (:data:`_SKILL_PBIAS_YLIM`)
    shared across all four calibration sets. cal/val use the same two colors
    (:data:`_CAL_COLOR`/:data:`_VAL_COLOR`) in both panels.
    """
    metrics = metrics.copy()
    for c in ["cal_kge", "val_kge", "cal_pbias", "val_pbias"]:
        if c in metrics:
            metrics[c] = pd.to_numeric(metrics[c], errors="coerce")  # None -> NaN (skipped)
    fig, ax = plt.subplots(2, 1, figsize=(4.0, 5.5), sharex=True)
    x = np.arange(len(metrics))
    w = 0.35
    series = [(-0.5, "cal_kge", "KGE cal", _CAL_COLOR), (0.5, "val_kge", "KGE val", _VAL_COLOR)]
    for off, col, label, color in series:
        ax[0].bar(x + off * w, metrics[col], w, label=label, color=color)
        for i, v in enumerate(metrics[col]):
            if np.isfinite(v) and v < 0:
                ax[0].annotate("↓", (i + off * w, 0.01), ha="center", va="bottom",
                               fontsize=7, color=color)
    # median cal/val KGE across basins, drawn as dashed lines in the matching cal/val color
    for col, color, lab in [("cal_kge", _CAL_COLOR, "median cal"), ("val_kge", _VAL_COLOR, "median val")]:
        med = metrics[col].median()
        if np.isfinite(med):
            ax[0].axhline(med, color=color, lw=1.0, ls="--", label=lab)
    ax[0].axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("KGE (1 = perfect)")
    ax[0].set_title(f"SAC-SMA skill vs. observed flow ({len(metrics)} watersheds)")
    ax[0].legend(loc="lower right", fontsize=7, ncol=2)

    ax[1].bar(x - w / 2, metrics["cal_pbias"], w, label="cal", color=_CAL_COLOR)
    ax[1].bar(x + w / 2, metrics["val_pbias"], w, label="val", color=_VAL_COLOR)
    ax[1].axhline(0, color="0.4", lw=0.8)
    for yref in (-10.0, 10.0):
        ax[1].axhline(yref, color="0.75", lw=0.6, ls="--")
    ax[1].set_ylim(-_SKILL_PBIAS_YLIM, _SKILL_PBIAS_YLIM)
    ax[1].set_ylabel("percent bias (%)")
    ax[1].legend(loc="lower right", fontsize=7, ncol=2)
    ax[1].set_xticks(x); ax[1].set_xticklabels(metrics["basin"], rotation=90)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def parity_fig(parity: dict, out: Path) -> None:
    """Demonstrate the exact Python-port vs. MATLAB-reference match.

    ``parity`` = {basin: DataFrame[date, flow_sim, flow_ref]} over the full
    1915–2018 simulation period.  Shows a representative overlay (the worst-case
    basin), a pooled 1:1 scatter, and the max daily |Δ| per basin.
    """
    maxdiff = {b: float(np.abs(df["flow_sim"] - df["flow_ref"]).max()) for b, df in parity.items()}
    worst = max(maxdiff, key=lambda b: maxdiff[b])
    # pooled KGE across all basins/days
    alls = np.concatenate([df["flow_sim"].to_numpy() for df in parity.values()])
    allr = np.concatenate([df["flow_ref"].to_numpy() for df in parity.values()])
    pooled_kge = kge(alls, allr)

    fig = plt.figure(figsize=(6.5, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    fig.suptitle("Python port vs. MATLAB reference — exact reproduction of simflow",
                 fontsize=8, fontweight="bold")

    # top (wide): overlay for the worst-case basin over a wet window
    axo = fig.add_subplot(gs[0, :])
    dfw = parity[worst]
    # pick the 4-year window containing the largest flow
    peak_year = dfw.loc[dfw["flow_ref"].idxmax(), "date"].year
    win = dfw[(dfw["date"].dt.year >= peak_year - 1) & (dfw["date"].dt.year <= peak_year + 2)]
    axo.plot(win["date"], win["flow_ref"], color="black", lw=2.4, alpha=0.5, label="MATLAB reference")
    axo.plot(win["date"], win["flow_sim"], color="tab:red", lw=0.9, ls="--", label="Python port")
    axo.set_title(f"{worst} (worst case) — the two series are indistinguishable "
                  f"(max |Δ| = {maxdiff[worst]:.3f} mm/day)")
    axo.set_ylabel("flow (mm/day)")
    axo.legend(loc="upper right", fontsize=8)

    # bottom-left: pooled scatter (sampled) with 1:1
    axs = fig.add_subplot(gs[1, 0])
    n = allr.size
    idx = np.linspace(0, n - 1, min(n, 30000)).astype(int)
    hi = max(allr.max(), alls.max())
    axs.scatter(allr[idx], alls[idx], s=3, alpha=0.2, color="tab:blue", edgecolors="none")
    axs.plot([0, hi], [0, hi], color="0.4", lw=1, ls="--")
    axs.set_xlim(0, hi); axs.set_ylim(0, hi)
    axs.set_title(f"All basins, all days (pooled KGE = {pooled_kge:.5f})")
    axs.set_xlabel("MATLAB reference (mm/day)"); axs.set_ylabel("Python port (mm/day)")

    # bottom-right: max |Δ| per basin
    axb = fig.add_subplot(gs[1, 1])
    items = sorted(maxdiff.items(), key=lambda kv: kv[1])
    axb.barh([k for k, _ in items], [v for _, v in items], color="tab:gray")
    axb.set_title("Max daily |Δ| per basin (mm/day)")
    axb.set_xlabel("max |Python − MATLAB| (mm/day)")
    axb.axvline(0.05, color="tab:red", lw=0.8, ls="--")
    axb.text(0.98, 0.02, "Δ is just the reference text rounded to 8 decimals",
             transform=axb.transAxes, ha="right", va="bottom", fontsize=7, color="0.4")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=300)
    plt.close(fig)
