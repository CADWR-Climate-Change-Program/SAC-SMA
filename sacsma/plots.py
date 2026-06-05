"""Calibration/validation diagnostic figures: simulated vs. observed gage flow.

Runs the faithful forward model (``run_basin``) for the 15 CDEC basins and
compares to the observed gage full-natural-flow target in
``data/reference/gage_15cdec.parquet`` (missing days are NaN).  The record is
split at the calibration/validation boundary (default WY2004 start,
``2003-10-01``); skill statistics are reported **separately** for each period.
Writes per-basin diagnostics, a domain skill-summary, and a metrics CSV under
``artifacts/``.

Usage::

    python -m sacsma.plots                       # all 15 basins -> artifacts/
    python -m sacsma.plots --basins BND TRM      # a subset
    python -m sacsma.plots --cal-end 2003-09-30  # calibration/validation split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from .io import load_basin_area, load_gage, load_reference, mmday_to_cfs  # noqa: E402
from .metrics import kge, nse, pbias  # noqa: E402
from .model import load_domain_forcing, run_basin  # noqa: E402

CDEC15 = [
    "SHA", "BND", "ORO", "YRS", "FOL", "MKM", "NHG", "NML",
    "TLG", "MRC", "MIL", "PNF", "TRM", "SCC", "ISB",
]

#: Calibration period ends here (WY1989-WY2003); validation is everything after.
CAL_END = "2003-09-30"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    am, bm, sa, sb = a.mean(), b.mean(), a.std(), b.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.mean((a - am) * (b - bm)) / (sa * sb))  # BLAS-free


def _flow_duration(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)[::-1]
    exceed = np.arange(1, xs.size + 1) / (xs.size + 1) * 100.0
    return exceed, xs


def _period_stats(sim: np.ndarray, obs: np.ndarray) -> dict:
    """Skill on finite (sim, obs) pairs; NaN-safe. Returns {} if no obs."""
    mask = np.isfinite(sim) & np.isfinite(obs)
    if mask.sum() == 0:
        return {"n": 0}
    s, o = sim[mask], obs[mask]
    return {
        "n": int(mask.sum()), "kge": kge(s, o), "nse": nse(s, o),
        "pbias": pbias(s, o), "r": _pearson(s, o),
        "sim_mean": float(s.mean()), "obs_mean": float(o.mean()),
    }


def _stat_text(label: str, st: dict) -> str:
    if st.get("n", 0) == 0:
        return f"{label}: no observations"
    return (f"{label} (n={st['n']:,})\n"
            f"KGE={st['kge']:.3f}  NSE={st['nse']:.3f}\n"
            f"pbias={st['pbias']:+.1f}%  r={st['r']:.3f}")


def basin_diagnostics_fig(basin: str, m: pd.DataFrame, cal_end: pd.Timestamp,
                          cal: dict, val: dict, out: Path) -> None:
    """Per-basin figure: full cal/val time series + scatter / regime / FDC."""
    d = m["date"]
    sim = m["flow_sim"].to_numpy()
    obs = m["flow_obs"].to_numpy()  # NaN where missing / outside gage record
    finite = np.isfinite(obs) & np.isfinite(sim)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])
    fig.suptitle(f"{basin} — SAC-SMA simulated vs. observed gage flow",
                 fontsize=14, fontweight="bold")

    # --- top row (spans all 3 cols): cal/val time series ---
    axts = fig.add_subplot(gs[0, :])
    axts.plot(d, obs, color="0.25", lw=0.7, label="gage (observed)")
    axts.plot(d, sim, color="tab:red", lw=0.7, alpha=0.8, label="simulated")
    axts.axvline(cal_end, color="tab:blue", lw=1.2, ls="--")
    axts.axvspan(cal_end, d.max(), color="tab:blue", alpha=0.05)
    axts.set_ylabel("flow (mm/day)")
    axts.set_title("Daily flow — calibration (left of dashed line) vs validation (shaded)")
    # headroom so the stat boxes/legend sit above the data
    ydata = np.nanmax([np.nanmax(sim), np.nanmax(obs)])
    axts.set_ylim(top=ydata * 1.35)
    axts.legend(loc="upper center", fontsize=8, ncol=2)
    # per-period stat boxes in the top corners (clear of the centered legend)
    axts.text(0.01, 0.97, _stat_text("CAL", cal), transform=axts.transAxes,
              va="top", ha="left", fontsize=8, family="monospace",
              bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axts.text(0.99, 0.97, _stat_text("VAL", val), transform=axts.transAxes,
              va="top", ha="right", fontsize=8, family="monospace",
              bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    # finite-obs subset for the lower diagnostic panels
    s, o = sim[finite], obs[finite]

    # --- scatter ---
    axsc = fig.add_subplot(gs[1, 0])
    if s.size:
        hi = max(s.max(), o.max())
        axsc.scatter(o, s, s=4, alpha=0.25, color="tab:blue", edgecolors="none")
        axsc.plot([0, hi], [0, hi], color="0.4", lw=1, ls="--")
        axsc.set_xlim(0, hi); axsc.set_ylim(0, hi)
    axsc.set_title(f"Daily sim vs gage (r={cal.get('r', float('nan')):.3f})")
    axsc.set_xlabel("gage (mm/day)"); axsc.set_ylabel("sim (mm/day)")

    # --- mean-monthly regime (on finite obs) ---
    axrg = fig.add_subplot(gs[1, 1])
    mf = m.loc[finite].assign(month=d[finite].dt.month)
    mm = mf.groupby("month")[["flow_obs", "flow_sim"]].mean()
    axrg.plot(mm.index, mm["flow_obs"], "o-", color="0.25", label="gage")
    axrg.plot(mm.index, mm["flow_sim"], "s-", color="tab:red", label="sim")
    axrg.set_title("Mean monthly regime")
    axrg.set_xlabel("month"); axrg.set_ylabel("flow (mm/day)")
    axrg.set_xticks(range(1, 13)); axrg.legend(fontsize=8)

    # --- flow-duration curve ---
    axfd = fig.add_subplot(gs[1, 2])
    if s.size:
        eo, qo = _flow_duration(o); es, qs = _flow_duration(s)
        axfd.plot(eo, np.maximum(qo, 1e-4), color="0.25", lw=1.2, label="gage")
        axfd.plot(es, np.maximum(qs, 1e-4), color="tab:red", lw=1.2, label="sim")
        axfd.set_yscale("log")
    axfd.set_title("Flow-duration curve")
    axfd.set_xlabel("exceedance (%)"); axfd.set_ylabel("flow (mm/day, log)")
    axfd.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def skill_summary_fig(metrics: pd.DataFrame, out: Path) -> None:
    """KGE / NSE (cal vs val) and percent bias across all basins."""
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    x = np.arange(len(metrics))
    w = 0.2
    ax[0].bar(x - 1.5 * w, metrics["cal_kge"], w, label="KGE cal", color="tab:green")
    ax[0].bar(x - 0.5 * w, metrics["val_kge"], w, label="KGE val", color="tab:olive")
    ax[0].bar(x + 0.5 * w, metrics["cal_nse"], w, label="NSE cal", color="tab:blue")
    ax[0].bar(x + 1.5 * w, metrics["val_nse"], w, label="NSE val", color="tab:cyan")
    ax[0].axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax[0].set_ylim(top=1.05)
    ax[0].set_ylabel("skill (1 = perfect)")
    ax[0].set_title("SAC-SMA skill vs. observed gage flow, 15 CDEC basins")
    ax[0].legend(fontsize=8, ncol=4)

    colors = ["tab:red" if abs(v) > 10 else "tab:gray" for v in metrics["cal_pbias"].fillna(0)]
    ax[1].bar(x, metrics["cal_pbias"], color=colors)
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].set_ylabel("calibration percent bias (%)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(metrics["basin"], rotation=45)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
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

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    fig.suptitle("Python port vs. MATLAB reference — exact reproduction of simflow",
                 fontsize=14, fontweight="bold")

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
    axo.legend(loc="upper right", fontsize=9)

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
    fig.savefig(out, dpi=110)
    plt.close(fig)


def make_all(
    basins: list[str] | None = None,
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
    run: str = "15cdec",
    cal_end: str = CAL_END,
) -> pd.DataFrame:
    basins = basins or CDEC15
    art = Path(artifacts_dir) / run
    figdir = art / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    cal_end_ts = pd.Timestamp(cal_end)

    forcing = load_domain_forcing(data_dir)
    areas = load_basin_area(data_dir).set_index("basin")["area_mi2"].to_dict()

    records = []
    parity = {}
    for b in basins:
        sim = run_basin(b, data_dir=data_dir, forcing=forcing).rename(columns={"flow": "flow_sim"})
        obs = load_gage(data_dir, basin=b)[["date", "flow"]].rename(columns={"flow": "flow_obs"})
        # left join on sim so the full simulated span is plotted; obs is NaN
        # outside its record (and on its own missing days).
        m = pd.merge(sim, obs, on="date", how="left").sort_values("date").reset_index(drop=True)

        is_cal = m["date"] <= cal_end_ts
        cal = _period_stats(m.loc[is_cal, "flow_sim"].to_numpy(), m.loc[is_cal, "flow_obs"].to_numpy())
        val = _period_stats(m.loc[~is_cal, "flow_sim"].to_numpy(), m.loc[~is_cal, "flow_obs"].to_numpy())

        # restrict the plotted span to where we have obs (+ trailing val window)
        obs_dates = m.loc[m["flow_obs"].notna(), "date"]
        if not obs_dates.empty:
            mplot = m[m["date"] >= obs_dates.min()].reset_index(drop=True)
        else:
            mplot = m
        basin_diagnostics_fig(b, mplot, cal_end_ts, cal, val, figdir / f"{b}_diagnostics.png")

        # parity vs MATLAB reference (full period)
        ref = load_reference(data_dir, basin=b)[["date", "flow"]].rename(columns={"flow": "flow_ref"})
        pm = pd.merge(sim, ref, on="date", how="inner")
        if not pm.empty:
            parity[b] = pm

        area = areas.get(b, np.nan)
        records.append({
            "basin": b, "area_mi2": area,
            "cal_kge": cal.get("kge"), "cal_nse": cal.get("nse"),
            "cal_pbias": cal.get("pbias"), "cal_r": cal.get("r"), "cal_n": cal.get("n", 0),
            "val_kge": val.get("kge"), "val_nse": val.get("nse"),
            "val_pbias": val.get("pbias"), "val_r": val.get("r"), "val_n": val.get("n", 0),
            "obs_mean_mmday": cal.get("obs_mean"),
            "obs_mean_cfs": mmday_to_cfs(cal.get("obs_mean") or np.nan, area),
        })
        print(f"  {b}: CAL KGE={cal.get('kge', float('nan')):.3f} "
              f"NSE={cal.get('nse', float('nan')):.3f} pbias={cal.get('pbias', float('nan')):+.1f}% "
              f"| VAL n={val.get('n', 0)}", flush=True)

    metrics = pd.DataFrame(records)
    if not metrics.empty:
        skill_summary_fig(metrics, figdir / "skill_summary.png")
        csv = art / "metrics_15cdec.csv"
        metrics.round(4).to_csv(csv, index=False)
        print(f"wrote {csv} and {len(metrics)} basin figures + skill_summary.png")
    if parity:
        parity_fig(parity, figdir / "parity_vs_matlab.png")
        print(f"wrote parity_vs_matlab.png -> {figdir}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sacsma.plots", description="Calibration diagnostic figures")
    ap.add_argument("--basins", nargs="*", default=None, help="subset of CDEC IDs (default: all 15)")
    ap.add_argument("--data-dir", default="data", help="data store")
    ap.add_argument("--artifacts-dir", default="artifacts", help="output root")
    ap.add_argument("--run", default="15cdec", help="run name -> artifacts/<run>/ (default: 15cdec)")
    ap.add_argument("--cal-end", default=CAL_END, help="calibration/validation split date (YYYY-MM-DD)")
    args = ap.parse_args(argv)
    make_all(basins=args.basins, data_dir=args.data_dir, artifacts_dir=args.artifacts_dir,
             run=args.run, cal_end=args.cal_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
