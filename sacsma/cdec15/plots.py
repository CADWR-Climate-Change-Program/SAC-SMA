"""15-CDEC calibration/validation diagnostics: simulated vs. observed gage flow.

Runs the forward model (``run_basin``) for the 15 CDEC basins and compares to
the observed daily gage full-natural-flow target (missing days are NaN).  The
record is split at the calibration/validation boundary (:data:`CAL_END`,
WY2004 start); skill statistics are reported **separately** for each period.
Writes per-basin diagnostics, a domain skill-summary, and a metrics CSV under
``artifacts/cdec15/``.

Usage::

    sacsma plots --domain 15cdec                 # all 15 basins
    python -m sacsma.cdec15.plots --basins BND TRM
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .._figures import (
    _period_stats,
    basin_diagnostics_fig,
    folsom_before_yuba,
    parity_fig,
    skill_summary_fig,
)
from ..io import load_basin_area, load_hru_table, load_reference, mmday_to_cfs
from ..model import load_domain_forcing, run_basin
from . import CAL_END, DOMAIN, load_gage


def _make_observed(basins, data_dir, forcing, areas, figdir, cal_end_ts):
    """Per-basin calibration/validation vs the observed daily gage."""
    records, parity = [], {}
    for b in basins:
        sim = run_basin(b, data_dir=data_dir, domain=DOMAIN, forcing=forcing).rename(columns={"flow": "flow_sim"})
        obs = load_gage(data_dir, basin=b)[["date", "flow"]].rename(columns={"flow": "flow_obs"})
        m = pd.merge(sim, obs, on="date", how="left").sort_values("date").reset_index(drop=True)

        is_cal = m["date"] <= cal_end_ts
        cal = _period_stats(m.loc[is_cal, "flow_sim"].to_numpy(), m.loc[is_cal, "flow_obs"].to_numpy())
        val = _period_stats(m.loc[~is_cal, "flow_sim"].to_numpy(), m.loc[~is_cal, "flow_obs"].to_numpy())

        obs_dates = m.loc[m["flow_obs"].notna(), "date"]
        mplot = m[m["date"] >= obs_dates.min()].reset_index(drop=True) if not obs_dates.empty else m
        basin_diagnostics_fig(b, mplot, cal_end_ts, cal, val, figdir / f"{b}_diagnostics.png")

        ref = load_reference(data_dir, basin=b, domain=DOMAIN)[["date", "flow"]].rename(columns={"flow": "flow_ref"})
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


def make_all(
    basins: list[str] | None = None,
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
    run: str = "cdec15",
    cal_end: str = CAL_END,
) -> pd.DataFrame:
    """15-CDEC diagnostics -> ``artifacts/<run>/`` (default ``artifacts/cdec15/``).

    Daily calibration/validation vs the observed gage, plus the exact MATLAB
    parity figure.
    """
    art = Path(artifacts_dir) / run
    figdir = art / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    cal_end_ts = pd.Timestamp(cal_end)
    if basins is None:
        hru = load_hru_table(data_dir, domain=DOMAIN)
        # north -> south by mean HRU latitude, with the Folsom-before-Yuba override
        # (matches the cross-compare/figure convention)
        basins = folsom_before_yuba(
            DOMAIN, hru.groupby("basin")["lat"].mean().sort_values(ascending=False).index.tolist())

    forcing = load_domain_forcing(data_dir, domain=DOMAIN)
    try:
        areas = load_basin_area(data_dir, domain=DOMAIN).set_index("basin")["area_mi2"].to_dict()
    except FileNotFoundError:
        areas = {}
    metrics, parity = _make_observed(basins, data_dir, forcing, areas, figdir, cal_end_ts)

    if not metrics.empty:
        csv = art / f"metrics_{DOMAIN}.csv"
        metrics.round(4).to_csv(csv, index=False)
        print(f"wrote {csv} and {len(metrics)} watershed figures")
    if parity:
        parity_fig(parity, figdir / "parity_vs_matlab.png")
        print(f"wrote parity_vs_matlab.png -> {figdir}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sacsma.cdec15.plots",
                                 description="15-CDEC per-watershed diagnostic figures")
    ap.add_argument("--basins", nargs="*", default=None, help="subset of basin codes (default: all)")
    ap.add_argument("--data-dir", default="data", help="data store")
    ap.add_argument("--artifacts-dir", default="artifacts", help="output root")
    ap.add_argument("--run", default="cdec15", help="run name -> artifacts/<run>/")
    ap.add_argument("--cal-end", default=CAL_END, help="calibration/validation split date (YYYY-MM-DD)")
    args = ap.parse_args(argv)
    make_all(basins=args.basins, data_dir=args.data_dir, artifacts_dir=args.artifacts_dir,
             run=args.run, cal_end=args.cal_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
