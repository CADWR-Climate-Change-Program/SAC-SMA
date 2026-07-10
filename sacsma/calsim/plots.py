"""CalSim/CalLite calibration/validation diagnostics: monthly sim vs observed FNF.

Each watershed's simulated daily flow is summed to monthly mm and compared to
the observed monthly full-natural-flow for its domain (``9unimp``/``11obs``/
``12rim``).  Where the **full-period** FNF is present (``fnf_<domain>_monthly``,
1922-) the record splits into the calibration window [cal_start, cal_end] and
**validation** (everything outside); otherwise it falls back to the
calibration-log FNF (calibration period only).  Writes per-basin diagnostics,
a skill summary, and a metrics CSV under ``artifacts/calsim/<domain>/``.

Also home to :func:`make_cdec15_fnf_check`: the 15cdec basins scored MONTHLY against
**CalSim3's unimpaired FNF** (``*_calsim3`` figures + ``metrics_15cdec_calsim3.csv``)
instead of 15cdec's own daily CDEC gage -> ``artifacts/cdec15/``.  Lives
here rather than in ``sacsma.cdec15`` because ``sacsma.calsim`` is the side of the
dependency edge allowed to import ``sacsma.cdec15`` (never the reverse).

Each of these also emits a **CalSim3-basis** variant (``*_calsim3`` figures +
``metrics_*_calsim3.csv``, :func:`_make_calsim3_diagnostics`): the anchor run — on the
GIS-**corrected footprint** for the
CalLite domains — scored against **CalSim3's own unimpaired FNF** (TAF/month) instead of the
observed-FNF calibration target, split on the same calibration windows.  Non-destructive: the
fnf-target diagnostics are untouched.  See ``tmp/CALSIM3_FNF_FOOTPRINT.md``.

Usage::

    sacsma plots --domain 11obs                  # all 11 watersheds
    python -m sacsma.calsim.plots --domain 9unimp --basins CacheCreek
    sacsma plots --domain 15cdec --fnf-check      # + the FNF cross-check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import cdec15
from .._figures import _period_stats, basin_diagnostics_fig, parity_fig, skill_summary_fig
from ..io import load_reference
from ..model import load_domain_forcing, run_basin
from . import DOMAINS, load_calib_monthly, load_fnf_monthly
from .compare import _BASIN_ABBREV, basin_order_north_south


def _make_calib_monthly(basins, data_dir, forcing, figdir, domain):
    """MONTHLY calibrated/validated performance vs observed FNF + parity."""
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
                              unit="mm/month", obs_label="observed FNF", cal_start=cal_start_ts,
                              title_obs="FNF Flow")

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
        # abbreviate the long 9unimp creek names for the skill-summary x-axis only (CSV/other
        # figures keep the full name) so its rotated tick labels don't eat into the axis height
        # relative to the other domains' short codes.
        fig_metrics = metrics.assign(basin=metrics["basin"].map(lambda b: _BASIN_ABBREV.get(b, b)))
        skill_summary_fig(fig_metrics, figdir / "skill_summary.png")
    return metrics, parity


def _make_calsim3_diagnostics(basins, data_dir, domain, figdir, *, screened, cal_windows,
                              suffix="_calsim3", obs_label="CalSim3 unimpaired FNF",
                              label_map=None):
    """Per-basin diagnostics + skill summary of the SAC-SMA anchor run scored against **CalSim3's
    own unimpaired FNF** (TAF/month), split by each basin's calibration window.

    Built from :func:`sacsma.calsim.compare.build_anchor_long` — the same ``sac`` + ``calsim3``
    monthly series the cross-compare anchor uses (``screened=True`` applies the anchor's
    over-reach screening — :data:`~sacsma.calsim.catchments.SCREENED_BASINS`, SHA/BND/SNS/
    Chowchilla since 2026-07-08 — via :func:`~sacsma.calsim.catchments.screened_footprint`).
    This is a
    **parallel** view alongside the observed-FNF-target diagnostics: files carry ``suffix`` and the
    fnf-target outputs are untouched.  ``cal_windows`` is ``{basin: (cal_start_ts, cal_end_ts)}``.
    ``label_map`` overrides the skill-summary x labels (default: :data:`_BASIN_ABBREV`) — the
    15cdec caller uses it to append each basin's 11obs/9unimp equivalent, e.g. ``MKM (Moke)``.
    """
    from .catchments import screened_footprint
    from .compare import build_anchor_long

    fp = {domain: screened_footprint(data_dir, domain=domain)} if screened else None
    long = build_anchor_long(data_dir, (domain,), footprint=fp)
    records = []
    for b in basins:
        sub = long[long["basin"] == b]
        if sub.empty or b not in cal_windows:
            continue
        wide = sub.pivot_table(index="date", columns="source", values="flow_taf")
        if domain not in wide.columns or "calsim3" not in wide.columns:
            continue
        m = (pd.DataFrame({"date": wide.index, "flow_sim": wide[domain].to_numpy(),
                           "flow_obs": wide["calsim3"].to_numpy()})
             .dropna().sort_values("date").reset_index(drop=True))
        if m.empty:
            continue
        cal_start_ts, cal_end_ts = cal_windows[b]
        is_cal = (m["date"] >= cal_start_ts) & (m["date"] <= cal_end_ts)
        cal = _period_stats(m.loc[is_cal, "flow_sim"].to_numpy(),
                            m.loc[is_cal, "flow_obs"].to_numpy())
        val = _period_stats(m.loc[~is_cal, "flow_sim"].to_numpy(),
                            m.loc[~is_cal, "flow_obs"].to_numpy())
        if val.get("n", 0) < 24:                # too little out-of-cal data for a meaningful VAL
            val = {"n": 0}
        basin_diagnostics_fig(b, m, cal_end_ts, cal, val, figdir / f"{b}_diagnostics{suffix}.png",
                              unit="TAF/month", obs_label=obs_label, cal_start=cal_start_ts,
                              title_obs="CalSim3 FNF Flow")
        records.append({
            "basin": b,
            "cal_kge": cal.get("kge"), "cal_nse": cal.get("nse"),
            "cal_pbias": cal.get("pbias"), "cal_r": cal.get("r"), "cal_n": cal.get("n", 0),
            "val_kge": val.get("kge"), "val_nse": val.get("nse"),
            "val_pbias": val.get("pbias"), "val_r": val.get("r"), "val_n": val.get("n", 0),
            "obs_mean_taf": cal.get("obs_mean"), "sim_mean_taf": cal.get("sim_mean"),
        })
        print(f"  {b} (vs CalSim3): CAL KGE={cal.get('kge', float('nan')):.3f} "
              f"pbias={cal.get('pbias', float('nan')):+.1f}% (n={cal.get('n', 0)}) | "
              f"VAL n={val.get('n', 0)}", flush=True)
    metrics = pd.DataFrame(records)
    if not metrics.empty:
        lab = label_map if label_map is not None else {}
        fig_metrics = metrics.assign(
            basin=metrics["basin"].map(lambda x: lab.get(x, _BASIN_ABBREV.get(x, x))))
        skill_summary_fig(fig_metrics, figdir / f"skill_summary{suffix}.png")
    return metrics


#: 15cdec basin -> (calsim domain, calsim basin code) for the basins with a monthly
#: full-natural-flow counterpart, per ``data/calsim/calsim_crosswalk.csv``'s
#: ``basin_15cdec``/``basin_11obs``/``basin_9unimp`` columns (deduplicated).  ORO/FOL/
#: NML/MIL carry a *different* code in 11obs than in 15cdec (rim-gauge naming); MKM/NHG
#: only match in 9unimp.  PNF/TRM/SCC/ISB (Tulare Basin) have no CalSim3 rim arc and thus
#: no counterpart in either calsim domain.
CDEC15_FNF_MATCH = {
    "SHA": ("11obs", "SHA"), "BND": ("11obs", "BND"), "ORO": ("11obs", "FTO"),
    "FOL": ("11obs", "AMF"), "YRS": ("11obs", "YRS"), "NML": ("11obs", "SNS"),
    "TLG": ("11obs", "TLG"), "MRC": ("11obs", "MRC"), "MIL": ("11obs", "SJF"),
    "MKM": ("9unimp", "MokelumneRiver"), "NHG": ("9unimp", "CalaverasRiver"),
}


def make_cdec15_fnf_check(
    basins: list[str] | None = None,
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
    run: str = "cdec15",
) -> pd.DataFrame:
    """15cdec basins scored MONTHLY against CalSim3's unimpaired FNF -> ``artifacts/<run>/``.

    A second, independent diagnostic alongside :func:`sacsma.cdec15.plots.make_all`'s daily
    CDEC-gage diagnostics: same basins, same GA-calibrated model run, scored directly against
    **CalSim3's own unimpaired FNF** (15cdec's OWN CalSim system mapping + catchment area — no
    cross-domain target or area harmonization).  Calibration window = each basin's daily-gage
    record start through :data:`cdec15.CAL_END`; validation picks up everything outside it,
    including the pre-gage decades CalSim3 reaches back to.  Only the 11 basins with a CalSim3
    rim counterpart (:data:`CDEC15_FNF_MATCH` keys) are scoreable; PNF/TRM/SCC/ISB (Tulare
    Basin, no CalSim3 rim arc) are skipped.  Writes ``metrics_15cdec_calsim3.csv`` +
    ``*_diagnostics_calsim3.png`` + ``skill_summary_calsim3.png``.

    (The earlier fnf-basis variant — the same model scored against the 11obs/9unimp
    ``fnf_<domain>_monthly`` tables — was retired 2026-07-07: those targets are a different
    historical-FNF product whose per-basin offsets vs CalSim3 (see ``target_vs_calsim3.csv``,
    e.g. CalaverasRiver +4.8%) leaked into the 15cdec scores as spurious bias.)
    """
    art = Path(artifacts_dir) / run
    figdir = art / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    basins = list(basins) if basins is not None else list(CDEC15_FNF_MATCH)
    unmatched = [b for b in basins if b not in CDEC15_FNF_MATCH]
    if unmatched:
        print(f"skipping (no CalSim3 counterpart): {unmatched}")
        basins = [b for b in basins if b in CDEC15_FNF_MATCH]

    cal_windows = {}
    for b in basins:
        gage = cdec15.load_gage(data_dir, basin=b)
        gstart = gage.loc[gage["flow"].notna(), "date"].min()
        cal_windows[b] = (pd.Timestamp(gstart) + pd.offsets.MonthEnd(0),
                          pd.Timestamp(cdec15.CAL_END))
    # x labels carry each basin's 11obs/9unimp equivalent where the code differs, e.g.
    # "MKM (Moke)", "ORO (FTO)"; identical codes (SHA, BND, ...) stay plain.
    labels = {b: (b if fb == b else f"{b} ({_BASIN_ABBREV.get(fb, fb)})")
              for b, (_, fb) in CDEC15_FNF_MATCH.items()}
    print("CalSim3-basis check (15cdec model vs CalSim3 unimpaired FNF):")
    met_c3 = _make_calsim3_diagnostics(basins, data_dir, cdec15.DOMAIN, figdir,
                                       screened=False, cal_windows=cal_windows,
                                       label_map=labels)
    if not met_c3.empty:
        met_c3.round(4).to_csv(art / "metrics_15cdec_calsim3.csv", index=False)
        print(f"wrote metrics_15cdec_calsim3.csv and {len(met_c3)} _calsim3 figures")
    return met_c3


def make_all(
    domain: str,
    basins: list[str] | None = None,
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
    run: str | None = None,
) -> pd.DataFrame:
    """CalLite-domain diagnostics -> ``artifacts/calsim/<run>/`` (default: the domain name).

    Monthly calibration/validation vs the observed FNF, plus the exact MATLAB
    parity figure.
    """
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    run = run or domain
    art = Path(artifacts_dir) / "calsim" / run
    figdir = art / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    if basins is None:
        # north -> south with the Folsom-before-Yuba override (the cross-compare convention)
        basins = basin_order_north_south(data_dir, domain)

    forcing = load_domain_forcing(data_dir, domain=domain)
    metrics, parity = _make_calib_monthly(basins, data_dir, forcing, figdir, domain)

    if not metrics.empty:
        csv = art / f"metrics_{domain}.csv"
        metrics.round(4).to_csv(csv, index=False)
        print(f"wrote {csv} and {len(metrics)} watershed figures")
    if parity:
        parity_fig(parity, figdir / "parity_vs_matlab.png")
        print(f"wrote parity_vs_matlab.png -> {figdir}")

    # Parallel CalSim3-basis diagnostics: the GIS-corrected (screened) footprint scored against
    # CalSim3's own unimpaired FNF, same calibration windows.  Only the cross-compare anchor sets
    # (11obs/9unimp) have a CalSim node mapping; 12rim is skipped (not in the cross-compare).
    if domain in ("11obs", "9unimp"):
        cal_windows = {}
        try:
            obs_tbl = load_fnf_monthly(data_dir, domain=domain)
            for b in basins:
                ob = obs_tbl[obs_tbl["basin"] == b]
                if ob.empty:
                    continue
                cal_windows[b] = (pd.Timestamp(ob["cal_start"].iloc[0]) + pd.offsets.MonthEnd(0),
                                  pd.Timestamp(ob["cal_end"].iloc[0]) + pd.offsets.MonthEnd(0))
        except FileNotFoundError:
            pass
        if cal_windows:
            print("CalSim3-basis diagnostics (corrected footprint vs CalSim3 unimpaired FNF):")
            met_c3 = _make_calsim3_diagnostics(basins, data_dir, domain, figdir,
                                               screened=True, cal_windows=cal_windows)
            if not met_c3.empty:
                met_c3.round(4).to_csv(art / f"metrics_{domain}_calsim3.csv", index=False)
                print(f"wrote metrics_{domain}_calsim3.csv and {len(met_c3)} _calsim3 figures")
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sacsma.calsim.plots",
                                 description="CalLite-domain per-watershed diagnostic figures")
    ap.add_argument("--domain", required=True, choices=list(DOMAINS),
                    help="CalLite calibration set")
    ap.add_argument("--basins", nargs="*", default=None, help="subset of watershed codes (default: all)")
    ap.add_argument("--data-dir", default="data", help="data store")
    ap.add_argument("--artifacts-dir", default="artifacts", help="output root")
    ap.add_argument("--run", default=None,
                    help="run name -> artifacts/calsim/<run>/ (default: domain)")
    args = ap.parse_args(argv)
    make_all(domain=args.domain, basins=args.basins, data_dir=args.data_dir,
             artifacts_dir=args.artifacts_dir, run=args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
