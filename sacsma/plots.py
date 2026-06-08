"""Calibration/validation diagnostic figures: simulated vs. observed gage flow.

Runs the faithful forward model (``run_basin``) for the 15 CDEC basins and
compares to the observed gage full-natural-flow target in
``data/reference/gage_15cdec.csv`` (missing days are NaN).  The record is
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

# House style: 8pt text throughout (titles/labels), small ticks/legends.
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.titlesize": 9,
})

from .io import (  # noqa: E402
    DEFAULT_DOMAIN, load_basin_area, load_gage, load_hru_table, load_reference, mmday_to_cfs,
)
from .metrics import kge, nse, pbias, pearson  # noqa: E402
from .model import load_domain_forcing, run_basin  # noqa: E402

CDEC15 = [
    "SHA", "BND", "ORO", "YRS", "FOL", "MKM", "NHG", "NML",
    "TLG", "MRC", "MIL", "PNF", "TRM", "SCC", "ISB",
]

#: Calibration period ends here (WY1989-WY2003); validation is everything after.
CAL_END = "2003-09-30"


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
                          cal_start: pd.Timestamp | None = None) -> None:
    """Per-basin figure: full cal/val time series + scatter / regime / FDC.

    Calibration is the window ``[cal_start, cal_end]`` (validation = everything
    outside, including pre-calibration years); if ``cal_start`` is None, calibration
    is everything up to ``cal_end``.
    """
    d = m["date"]
    sim = m["flow_sim"].to_numpy()
    obs = m["flow_obs"].to_numpy()  # NaN where missing / outside obs record
    finite = np.isfinite(obs) & np.isfinite(sim)
    step = "Daily" if unit == "mm/day" else "Monthly"

    fig = plt.figure(figsize=(6.5, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2])  # taller top; squarer bottom panels
    fig.suptitle(f"{basin} — SAC-SMA simulated vs. observed {('FNF' if step=='Monthly' else 'gage')} flow",
                 fontsize=9, fontweight="bold")

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
    # over its OWN period — CAL on the side of the calibration window's centre, VAL opposite —
    # so it's correct for any domain (calibration is early for 15cdec, late for the rim sets).
    cal_mid = cal_start + (cal_end - cal_start) / 2 if cal_start is not None else cal_end
    cal_on_left = cal_mid < dmin + (dmax - dmin) / 2
    cal_x, cal_ha = (0.01, "left") if cal_on_left else (0.99, "right")
    val_x, val_ha = (0.99, "right") if cal_on_left else (0.01, "left")
    _box = dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9)
    axts.text(cal_x, 0.97, _stat_text("CAL", cal), transform=axts.transAxes,
              va="top", ha=cal_ha, fontsize=8, family="monospace", bbox=_box)
    if val.get("n", 0) > 0:  # omit the VAL box when there is too little out-of-cal data (e.g. BLB)
        axts.text(val_x, 0.97, _stat_text("VAL", val), transform=axts.transAxes,
                  va="top", ha=val_ha, fontsize=8, family="monospace", bbox=_box)

    # finite-obs subset for the lower diagnostic panels
    s, o = sim[finite], obs[finite]

    # --- scatter ---
    axsc = fig.add_subplot(gs[1, 0])
    if s.size:
        hi = max(s.max(), o.max())
        axsc.scatter(o, s, s=4, alpha=0.25, color="tab:blue", edgecolors="none")
        axsc.plot([0, hi], [0, hi], color="0.4", lw=1, ls="--")
        axsc.set_xlim(0, hi); axsc.set_ylim(0, hi)
    axsc.set_title(f"{step} sim vs obs\n(r={cal.get('r', float('nan')):.3f})")
    axsc.set_xlabel(f"observed ({unit})"); axsc.set_ylabel(f"sim ({unit})")

    # --- mean-monthly regime (on finite obs), water-year order Oct..Sep ---
    axrg = fig.add_subplot(gs[1, 1])
    wy_months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    wy_labels = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]
    mf = m.loc[finite].assign(month=d[finite].dt.month)
    mm = mf.groupby("month")[["flow_obs", "flow_sim"]].mean().reindex(wy_months)
    x = np.arange(12)
    axrg.plot(x, mm["flow_obs"], "o-", color="0.25", label="obs")
    axrg.plot(x, mm["flow_sim"], "s-", color="tab:red", label="sim")
    axrg.set_title("Mean monthly regime\n(water year)")
    axrg.set_xlabel("water-year month"); axrg.set_ylabel(f"flow ({unit})")
    axrg.set_xticks(x); axrg.set_xticklabels(wy_labels); axrg.legend(fontsize=8)

    # --- flow-duration curve ---
    axfd = fig.add_subplot(gs[1, 2])
    if s.size:
        eo, qo = _flow_duration(o); es, qs = _flow_duration(s)
        axfd.plot(eo, np.maximum(qo, 1e-4), color="0.25", lw=1.2, label="obs")
        axfd.plot(es, np.maximum(qs, 1e-4), color="tab:red", lw=1.2, label="sim")
        axfd.set_yscale("log")
    axfd.set_title("Flow-duration\ncurve")
    axfd.set_xlabel("exceedance (%)"); axfd.set_ylabel(f"flow ({unit}, log)")
    axfd.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def skill_summary_fig(metrics: pd.DataFrame, out: Path) -> None:
    """KGE / NSE (cal vs val) and percent bias across all basins."""
    metrics = metrics.copy()
    for c in ["cal_kge", "val_kge", "cal_nse", "val_nse", "cal_pbias", "val_pbias"]:
        if c in metrics:
            metrics[c] = pd.to_numeric(metrics[c], errors="coerce")  # None -> NaN (skipped)
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    x = np.arange(len(metrics))
    w = 0.2
    ax[0].bar(x - 1.5 * w, metrics["cal_kge"], w, label="KGE cal", color="tab:green")
    ax[0].bar(x - 0.5 * w, metrics["val_kge"], w, label="KGE val", color="tab:olive")
    ax[0].bar(x + 0.5 * w, metrics["cal_nse"], w, label="NSE cal", color="tab:blue")
    ax[0].bar(x + 1.5 * w, metrics["val_nse"], w, label="NSE val", color="tab:cyan")
    ax[0].axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax[0].set_ylim(top=1.05)
    ax[0].set_ylabel("skill (1 = perfect)")
    ax[0].set_title(f"SAC-SMA skill vs. observed flow ({len(metrics)} watersheds)")
    ax[0].legend(fontsize=8, ncol=4)

    colors = ["tab:red" if abs(v) > 10 else "tab:gray" for v in metrics["cal_pbias"].fillna(0)]
    ax[1].bar(x, metrics["cal_pbias"], color=colors)
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].set_ylabel("calibration percent bias (%)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(metrics["basin"], rotation=45)

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
                 fontsize=9, fontweight="bold")

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
    fig.savefig(out, dpi=300)
    plt.close(fig)


def _make_observed(basins, data_dir, forcing, areas, figdir, domain, cal_end_ts):
    """15-CDEC path: per-basin calibration/validation vs the observed gage."""
    records, parity = [], {}
    for b in basins:
        sim = run_basin(b, data_dir=data_dir, domain=domain, forcing=forcing).rename(columns={"flow": "flow_sim"})
        obs = load_gage(data_dir, basin=b)[["date", "flow"]].rename(columns={"flow": "flow_obs"})
        m = pd.merge(sim, obs, on="date", how="left").sort_values("date").reset_index(drop=True)

        is_cal = m["date"] <= cal_end_ts
        cal = _period_stats(m.loc[is_cal, "flow_sim"].to_numpy(), m.loc[is_cal, "flow_obs"].to_numpy())
        val = _period_stats(m.loc[~is_cal, "flow_sim"].to_numpy(), m.loc[~is_cal, "flow_obs"].to_numpy())

        obs_dates = m.loc[m["flow_obs"].notna(), "date"]
        mplot = m[m["date"] >= obs_dates.min()].reset_index(drop=True) if not obs_dates.empty else m
        basin_diagnostics_fig(b, mplot, cal_end_ts, cal, val, figdir / f"{b}_diagnostics.png")

        ref = load_reference(data_dir, basin=b, domain=domain)[["date", "flow"]].rename(columns={"flow": "flow_ref"})
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
              f"NSE={cal.get('nse', float('nan')):.3f} | VAL n={val.get('n', 0)}", flush=True)

    metrics = pd.DataFrame(records)
    if not metrics.empty:
        skill_summary_fig(metrics, figdir / "skill_summary.png")
    return metrics, parity


def _make_calib_monthly(basins, data_dir, forcing, figdir, domain):
    """CalLite path: MONTHLY calibrated/validated performance vs observed FNF + parity.

    Each watershed's simulated daily flow is summed to monthly mm and compared to
    the observed monthly full-natural-flow.  Where the **full-period** FNF is present
    (``fnf_<domain>_monthly``, 1922-) the record splits into the calibration window
    [cal_start, cal_end] and **validation** (everything outside, incl. pre-1948);
    otherwise it falls back to the calibration-log FNF (calibration period only).
    """
    from .io import load_calib_monthly, load_fnf_monthly

    try:
        obs_tbl = load_fnf_monthly(data_dir, domain=domain)  # full period (1922-)
    except FileNotFoundError:
        obs_tbl = load_calib_monthly(data_dir, domain=domain)  # calibration period only
    records, parity = [], {}
    for b in basins:
        sim_d = run_basin(b, data_dir=data_dir, domain=domain, forcing=forcing)
        # daily mm/day -> monthly total mm
        sm = (sim_d.assign(month=sim_d["date"].dt.to_period("M"))
              .groupby("month")["flow"].sum().reset_index())
        sm["date"] = sm["month"].dt.to_timestamp("M")
        sm = sm.rename(columns={"flow": "flow_sim"})[["date", "flow_sim"]]

        obs = obs_tbl[obs_tbl["basin"] == b]
        if obs.empty:
            continue
        cal_start_ts = pd.Timestamp(obs["cal_start"].iloc[0]) + pd.offsets.MonthEnd(0)
        cal_end_ts = pd.Timestamp(obs["cal_end"].iloc[0]) + pd.offsets.MonthEnd(0)
        m = pd.merge(sm, obs[["date", "obs_mm"]].rename(columns={"obs_mm": "flow_obs"}),
                     on="date", how="left").sort_values("date").reset_index(drop=True)

        is_cal = (m["date"] >= cal_start_ts) & (m["date"] <= cal_end_ts)
        cal = _period_stats(m.loc[is_cal, "flow_sim"].to_numpy(), m.loc[is_cal, "flow_obs"].to_numpy())
        val = _period_stats(m.loc[~is_cal, "flow_sim"].to_numpy(), m.loc[~is_cal, "flow_obs"].to_numpy())
        # require enough out-of-calibration months for a meaningful validation skill; a basin
        # whose record barely extends beyond the calibration window (e.g. BLB, ~14 months)
        # otherwise reports a garbage val stat — drop it.
        if val.get("n", 0) < 24:
            val = {"n": 0}
        # plot only the observed span: clip BOTH ends to [first obs, last obs] so the
        # simulation is not carried forward past the record (the right edge ends cleanly at
        # the calibration period; validation shows on the early, left part of the axis).
        obs_dates = m.loc[m["flow_obs"].notna(), "date"]
        mplot = (m[(m["date"] >= obs_dates.min()) & (m["date"] <= obs_dates.max())]
                 .reset_index(drop=True) if not obs_dates.empty else m)
        basin_diagnostics_fig(b, mplot, cal_end_ts, cal, val, figdir / f"{b}_diagnostics.png",
                              unit="mm/month", obs_label="observed FNF", cal_start=cal_start_ts)

        ref = load_reference(data_dir, basin=b, domain=domain)[["date", "flow"]].rename(columns={"flow": "flow_ref"})
        pm = pd.merge(sim_d.rename(columns={"flow": "flow_sim"}), ref, on="date", how="inner")
        if not pm.empty:
            parity[b] = pm

        records.append({
            "basin": b,
            "cal_kge": cal.get("kge"), "cal_nse": cal.get("nse"),
            "cal_pbias": cal.get("pbias"), "cal_r": cal.get("r"), "cal_n": cal.get("n", 0),
            "val_kge": val.get("kge"), "val_nse": val.get("nse"),
            "val_pbias": val.get("pbias"), "val_r": val.get("r"), "val_n": val.get("n", 0),
            "obs_mean_mmmon": cal.get("obs_mean"), "sim_mean_mmmon": cal.get("sim_mean"),
        })
        print(f"  {b}: CAL KGE={cal.get('kge', float('nan')):.3f} "
              f"NSE={cal.get('nse', float('nan')):.3f} pbias={cal.get('pbias', float('nan')):+.1f}% "
              f"(n={cal.get('n', 0)}) | VAL n={val.get('n', 0)}", flush=True)

    metrics = pd.DataFrame(records)
    if not metrics.empty:
        skill_summary_fig(metrics, figdir / "skill_summary.png")
    return metrics, parity


def make_all(
    basins: list[str] | None = None,
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
    run: str | None = None,
    cal_end: str = CAL_END,
    domain: str = DEFAULT_DOMAIN,
) -> pd.DataFrame:
    """Per-watershed diagnostics for a ``domain`` -> ``artifacts/<run>/``.

    ``15cdec`` compares against the observed gage (calibration/validation); the
    CalLite sets (no daily observed target) get simulated-flow diagnostics.  Both
    include the exact MATLAB parity figure.  ``run`` defaults to the domain name.
    """
    run = run or domain
    art = Path(artifacts_dir) / run
    figdir = art / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    cal_end_ts = pd.Timestamp(cal_end)
    if basins is None:
        basins = sorted(load_hru_table(data_dir, domain=domain)["basin"].unique())

    forcing = load_domain_forcing(data_dir, domain=domain)
    try:
        areas = load_basin_area(data_dir, domain=domain).set_index("basin")["area_mi2"].to_dict()
    except FileNotFoundError:
        areas = {}
    # 15cdec is calibrated/validated daily vs the observed gage; the CalLite sets
    # are calibrated monthly vs the observed FNF embedded in their calibration logs.
    if domain == "15cdec":
        metrics, parity = _make_observed(basins, data_dir, forcing, areas, figdir, domain, cal_end_ts)
    else:
        metrics, parity = _make_calib_monthly(basins, data_dir, forcing, figdir, domain)

    if not metrics.empty:
        csv = art / f"metrics_{domain}.csv"
        metrics.round(4).to_csv(csv, index=False)
        print(f"wrote {csv} and {len(metrics)} watershed figures")
    if parity:
        parity_fig(parity, figdir / "parity_vs_matlab.png")
        print(f"wrote parity_vs_matlab.png -> {figdir}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sacsma.plots", description="Per-watershed diagnostic figures")
    ap.add_argument("--basins", nargs="*", default=None, help="subset of watershed codes (default: all)")
    ap.add_argument("--domain", default="15cdec",
                    choices=["15cdec", "9unimp", "11obs", "12rim"],
                    help="calibration set (default: 15cdec)")
    ap.add_argument("--data-dir", default="data", help="data store")
    ap.add_argument("--artifacts-dir", default="artifacts", help="output root")
    ap.add_argument("--run", default=None, help="run name -> artifacts/<run>/ (default: domain)")
    ap.add_argument("--cal-end", default=CAL_END, help="calibration/validation split date (YYYY-MM-DD)")
    args = ap.parse_args(argv)
    make_all(basins=args.basins, data_dir=args.data_dir, artifacts_dir=args.artifacts_dir,
             run=args.run, cal_end=args.cal_end, domain=args.domain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
