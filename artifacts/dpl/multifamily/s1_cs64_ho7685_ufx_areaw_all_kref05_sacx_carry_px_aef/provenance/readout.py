"""S1 prereg readout: evaluate every bar of bars.json on the S1 run (PREREG_S1).

usage (cwd = wt_merge, PYTHONPATH = wt_merge, CPU):
  readout.py <s1_run_dir> [--bars bars.json] [--out DIR] [--logs DIR] [--p17 NPZ]
  --logs: folder holding chunk_log.csv / train_log.csv (default the run dir)
  --p17:  the run's P17b 1925 re-simulation (p17/p17b_s1.py sim), for HO-P17
Every S1 statistic is computed by s1lib.py, the module derive_bars.py computed the comparators with,
and is cross-checked against the package's own outputs (metrics_entities.csv kge_reg,
metrics_entities_holdout.csv, tier1_metrics.csv / tier2_metrics.csv holdout windows).
Writes <out>/readout.json + readout.txt (default <s1_run_dir>/../_readout_<name> is NOT used: --out
defaults to prereg/readout_<run name>).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import s1lib as L  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--bars", default=str(Path(__file__).parent / "bars.json"))
p.add_argument("--out", default=None)
p.add_argument("--logs", default=None)
p.add_argument("--p17", default=None)
a = p.parse_args()
run = Path(a.run)
logs = Path(a.logs) if a.logs else run
out = Path(a.out) if a.out else Path(__file__).parent / f"readout_{run.name}"
out.mkdir(parents=True, exist_ok=True)
B = json.loads(Path(a.bars).read_text())
bars = {b["id"]: b for b in B["bars"]}
A = set(L.tier_a_arcs())
lines, res, xchk = [], [], []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def f4(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.4f}"


def verdict(b, value, comp=None, thr=None):
    """pass / fail on the threshold; 'unreadable (...)' when |value - comparator| < spread."""
    thr = b.get("threshold") if thr is None else thr
    if value is None or not np.isfinite(value):
        return "n/a"
    ok = value >= thr if b.get("op", ">=") == ">=" else value < thr
    sp = b.get("spread")
    if sp and comp is not None and abs(value - comp) < sp:
        return "unreadable (holds)" if ok else "unreadable (below the bar)"
    if "fail_below" in b and value < b["fail_below"]:
        return "FAIL (hard)"
    if "fail_below" in b and not ok:
        return "cost (discuss)"
    return "pass" if ok else "fail"


def rec(bid, value, verdict_, detail="", **kw):
    b = bars[bid]
    res.append(dict(id=bid, cls=b["cls"], primary=bool(b.get("primary")), value=value, verdict=verdict_,
                    threshold=b.get("threshold"), comparator=b.get("comparator"), window=b.get("window"),
                    detail=detail, **kw))
    v = value if isinstance(value, str) else f4(value) if isinstance(value, (int, float)) else json.dumps(value)
    say(f"  {bid:12s} [{b['cls']}{', PRIMARY' if b.get('primary') else ''}] {v[:70]:<16s} -> {verdict_}"
        + (f"   ({detail})" if detail else ""))


ck = L.run_cfg(run)
cfg = ck["cfg"]
om = tuple(cfg.get("obs_mask", ()))
npz = run / "sim_daily_mm.npz"
ids = L.load_npz(npz)[0]
say(f"S1 readout — {run}  ({datetime.now(timezone.utc).isoformat(timespec='seconds')})")
say(f"checkpoint epoch {ck['epoch']}, selection {f4(ck['cal_kge'])}, sel3 {f4(ck['sel3'])}; holdout_wy "
    f"{cfg.get('holdout_wy')}, uf_train_start {cfg.get('uf_train_start')!r}, calsim_arcs {cfg.get('calsim_arcs')!r}; "
    f"{len(ids)} entities")
if not cfg.get("holdout_wy"):
    say("WARNING: this checkpoint has no holdout_wy — not an S1 run; the numbers below are for testing only")

# ---------------------------------------------------------------- statistics (s1lib, same as derive_bars)
ent = L.train_entity_kge(npz, om)
fam = L.family_stats(ent)
ho = L.holdout_entities(npz, om)
tw = L.cdec_twins(npz)
t1 = L.tier1_windows(run) if (run / "tier1" / "tier1_monthly.csv").exists() else None
t2 = L.tier2_windows(run) if (run / "tier2" / "tier2_monthly.csv").exists() else None
ar = L.anchor_rescaled(run) if t2 is not None else None

# ---------------------------------------------------------------- cross-checks against the package outputs
me = pd.read_csv(run / "metrics_entities.csv").set_index("entity_id")
col = "kge_reg" if "kge_reg" in me.columns else "kge"
d = (ent.set_index("entity_id")["kge"] - me[col].reindex(ent["entity_id"]).to_numpy()).abs()
xchk.append(("metrics_entities.csv " + col + " vs s1lib training-window KGE", float(d.max()), int(d.notna().sum())))
if (run / "metrics_entities_holdout.csv").exists():
    mh = pd.read_csv(run / "metrics_entities_holdout.csv").set_index(["entity_id", "window", "basis"])["kge"]
    d = (ho.set_index(["entity_id", "window", "basis"])["kge"] - mh.reindex(
        ho.set_index(["entity_id", "window", "basis"]).index).to_numpy()).abs()
    xchk.append(("metrics_entities_holdout.csv vs s1lib holdout_metrics (sim_daily_mm.npz)", float(d.max()), int(d.notna().sum())))
if t2 is not None and (run / "tier2" / "tier2_metrics.csv").exists():
    tm = pd.read_csv(run / "tier2" / "tier2_metrics.csv")
    for wn in ("WY1976-85", "WY1976-85_own"):
        x = tm[tm.window == wn].set_index("arc")["kge"]
        y = t2[t2.window == wn].set_index("arc")["kge"]
        dd = (x - y.reindex(x.index)).abs()
        xchk.append((f"tier2_metrics.csv {wn} vs s1lib (tier2_monthly.csv)", float(dd.max()), int(dd.notna().sum())))
if t1 is not None and (run / "tier1" / "tier1_metrics.csv").exists():
    tm = pd.read_csv(run / "tier1" / "tier1_metrics.csv")
    x = tm[(tm.window == "WY1976-84") & (tm.ref_kind == "anchor")].set_index("set_id")["kge"]
    y = t1[(t1.window == "WY1976-84") & (t1.ref_kind == "anchor")].set_index("set_id")["kge"]
    dd = (x - y.reindex(x.index)).abs()
    xchk.append(("tier1_metrics.csv WY1976-84 anchors vs s1lib (tier1_monthly.csv)", float(dd.max()), int(dd.notna().sum())))

# ---------------------------------------------------------------- mechanism
say("\nMechanism")
if (logs / "chunk_log.csv").exists():
    mech = L.mechanism(logs)
    b = bars["M-cs"]
    real = mech["realized_e0"]
    key = {"usgs": "usgs_daily", "cdec": "cdec_daily", "uf": "uf_monthly", "cs": "calsim_monthly"}
    dev = {f: real[f] - b["target"][key[f]] for f in real}
    rec("M-cs", {f: round(v, 4) for f, v in real.items()},
        "pass" if len(dev) == 4 and all(abs(v) <= b["tol"] for v in dev.values()) else "fail",
        ", ".join(f"{f} {v:+.4f}" for f, v in dev.items()))
    mass = mech["loss_mass"]
    e0 = mass[min(mass)]
    late = {ep: m for ep, m in mass.items() if ep >= 10}
    flags = [ep for ep, m in late.items() if "cs" in m and not 0.5 * e0["cs"] <= m["cs"] <= 2 * e0["cs"]]
    rec("M-mass", {f: round(v, 3) for f, v in (late[max(late)] if late else mass[max(mass)]).items()},
        "report" + (f" — FLAG calsim outside [0.5x, 2x] e0 at epochs {flags[:5]}" if flags else ""),
        "last epoch's loss-mass shares; e0 " + ", ".join(f"{f} {v:.3f}" for f, v in e0.items()))
    bestsel = ck["cal_kge"] is not None and np.isfinite(ck["cal_kge"])
    ok = mech["clip_share_e5"] < 0.5 if np.isfinite(mech["clip_share_e5"]) else True
    rec("M-clip", mech["clip_share_e5"], ("pass" if ok and bestsel else "fail")
        + (" — FLAG an epoch skipped > 4 of 70 chunks" if mech["skipped_max_epoch"] > 4 else ""),
        f"skipped steps {mech['skipped']} (max {mech['skipped_max_epoch']} in one epoch), best.pt selection "
        f"finite {bestsel}, epochs {mech['n_epochs']}")
else:
    say(f"  (no chunk_log.csv under {logs}: M-cs / M-mass / M-clip not read)")

# ---------------------------------------------------------------- training period
say("\nTraining period (holdout-masked targets)")
b = bars["S-sel"]
ep = ck["epoch"]
lo, hi = b["threshold"]
sel3_lib = L.shares_stat(fam, L.SEL3_S1)
rec("S-sel", float(ep) if ep is not None else float("nan"),
    "flag: early (< 40)" if ep is not None and ep < lo else "flag: not converged (>= 110)" if ep is not None and ep >= hi
    else "ok", f"selection {f4(ck['cal_kge'])}, sel3 logged {f4(ck['sel3'])} / re-scored {f4(sel3_lib)} vs H95 "
    f"{f4(b['comparator']['value'])} (report)")
if t2 is not None:
    x = t2[(t2.window == "arc_train") & t2.arc.isin(A)]["kge"]
    b = bars["A-trn"]
    ok = x.mean() >= b["threshold"]["mean"] and x.median() >= b["threshold"]["median"]
    cs = me[me["family"] == "calsim_monthly"]["kge"] if "family" in me.columns else pd.Series(dtype=float)
    rec("A-trn", {"mean": round(float(x.mean()), 4), "median": round(float(x.median()), 4), "n": int(x.notna().sum())},
        "pass" if ok else "fail", f"entity basis (metrics_entities calsim_monthly): mean {f4(cs.mean())} median "
        f"{f4(cs.median())} n={len(cs)}")
ek = ent.set_index("entity_id")["kge"]
for bid, stat in (("C1-mean", "mean"), ("C1-median", "median")):
    v = fam.get("cdec_daily", {}).get(stat, np.nan)
    rec(bid, v, verdict(bars[bid], v, bars[bid]["comparator"]["value"]), f"n={fam.get('cdec_daily', {}).get('n')}")
for e in ("ORO", "YRS", "SHA", "BND", "NHG"):
    bid = f"C1-{e}"
    v = float(ek.get(f"cdec_{e}", np.nan))
    extra = f"vs H {f4(bars[bid]['also']['H'])}" if e == "NHG" else ""
    rec(bid, v, verdict(bars[bid], v, bars[bid]["comparator"]["value"]), extra)
v = fam.get("uf_monthly", {}).get("mean", np.nan)
rec("U1-mean", v, verdict(bars["U1-mean"], v, bars["U1-mean"]["comparator"]["value"]),
    f"n={fam.get('uf_monthly', {}).get('n')}, WY1986-2014")
b = bars["U1-entity"]
per = {u: float(ek.get(u, np.nan)) for u in b["threshold"]}
fails = [u for u, v in per.items() if np.isfinite(v) and v < b["threshold"][u]]
rec("U1-entity", {u: round(v, 4) for u, v in per.items()}, "fail: " + ", ".join(fails) if fails else
    ("pass" if all(np.isfinite(list(per.values()))) else "partial (entities missing)"))
for bid, stat in (("G1u-mean", "mean"), ("G1u-median", "median")):
    v = fam.get("usgs_daily", {}).get(stat, np.nan)
    rec(bid, v, verdict(bars[bid], v, bars[bid]["comparator"]["value"]), f"n={fam.get('usgs_daily', {}).get('n')}")

# ---------------------------------------------------------------- holdout
say("\nHoldout WY1976-85")
if t2 is not None:
    b = bars["HO-A1"]
    x = t2[(t2.window == "WY1976-85_own") & t2.arc.isin(b["arcs"])]
    rec("HO-A1", float(x.kge.median()), verdict(b, float(x.kge.median())),
        f"n={int(x.kge.notna().sum())} of the {len(b['arcs'])} preregistered arcs; min n_months {int(x.n.min()) if len(x) else 0}")
    w = t2[(t2.window == "WY1976-85") & t2.has_ref]
    rec("HO-A2-all", float(w.kge.median()), verdict(bars["HO-A2-all"], float(w.kge.median())),
        f"n={int(w.kge.notna().sum())}; tier-A 64 median {f4(w[w.arc.isin(A)].kge.median())} (H "
        f"{f4(bars['HO-A2-all']['also']['H_tierA64'])})")
    nA = w[w.arc.isin(bars["HO-A2-nonA"]["arcs"])]
    rec("HO-A2-nonA", float(nA.kge.median()), verdict(bars["HO-A2-nonA"], float(nA.kge.median())),
        f"n={int(nA.kge.notna().sum())}")
u84 = ho[(ho.family == "uf_monthly") & (ho.window == "WY1976-84")].set_index("entity_id")["kge"]
u85 = ho[(ho.family == "uf_monthly") & (ho.window == "WY1976-85")].set_index("entity_id")["kge"]
b = bars["HO-U"]
rec("HO-U", float(u84.mean()), verdict(b, float(u84.mean()), b["comparator"]["value"]),
    f"n={int(u84.notna().sum())}; WY1976-85 mean {f4(u85.mean())} (H {f4(b['also']['H_WY1976_85'])}); "
    + ", ".join(f"{u} {f4(u84.get(u, np.nan))} (H {f4(b['comparator']['per_unit'].get(u))})" for u in b["also"]["flag_units"]))
t85 = tw[tw.window == "WY1976-85"]["kge"]
rec("HO-C-twins", float(t85.median()), verdict(bars["HO-C-twins"], float(t85.median()),
                                               bars["HO-C-twins"]["comparator"]["value"]), f"n={int(t85.notna().sum())}")
if t1 is not None:
    fu = t1[(t1.ref_kind == "anchor") & (t1.window == "WY1976-84")]["kge"]
    rec("HO-C-FU", float(fu.median()), verdict(bars["HO-C-FU"], float(fu.median()), bars["HO-C-FU"]["comparator"]["value"]),
        f"n={int(fu.notna().sum())}")
b = bars["HO-G"]
g = ho[(ho.family == "usgs_daily") & ho.entity_id.isin(b["gauges"]) & (ho.n >= 365)]
rec("HO-G", float(g.kge.median()), verdict(b, float(g.kge.median())),
    f"n={len(g)} of {len(b['gauges'])} gauges; mean {f4(g.kge.mean())} (H {f4(b['comparator']['mean'])}); "
    f"H95 in-sample {f4(b['also']['H95_in_sample'])} (never a bar)")
if ar is not None:
    x = ar[ar.window == "WY1976-85"].dropna(subset=["kge_resc"])
    gaps = {}
    if len(u84) and "uf_monthly" in fam:
        gaps["uf"] = fam["uf_monthly"]["mean"] - u84.mean()
    tw86 = tw[tw.window == "WY1986-2014"]["kge"]
    gaps["cdec_twins"] = tw86.median() - t85.median()
    if t1 is not None:
        gaps["fu_anchors"] = (t1[(t1.ref_kind == "anchor") & (t1.window == "WY1986-2014")].kge.median()
                              - t1[(t1.ref_kind == "anchor") & (t1.window == "WY1976-84")].kge.median())
    gaps["tierA_arcs"] = (t2[(t2.window == "arc_train") & t2.arc.isin(A)].kge.median()
                          - t2[(t2.window == "WY1976-85") & t2.arc.isin(A)].kge.median())
    hc = bars["HO-R"]["comparator"]
    rec("HO-R", {"abs_median": round(float(x.kge_abs.median()), 4), "resc_median": round(float(x.kge_resc.median()), 4)},
        "report", f"n={len(x)} (H abs {f4(hc['abs_median'])} / resc {f4(hc['resc_median'])}); gaps train-holdout "
        + ", ".join(f"{k} {v:+.3f} (H {hc['gaps'].get(k, float('nan')):+.3f})" for k, v in gaps.items()))
if a.p17 and Path(a.p17).exists():
    pp = L.p17b_sites(a.p17)
    val = {era: round(float(g.kge.median()), 4) for era, g in pp.groupby("era")}
    rec("HO-P17", val, "report", "H " + json.dumps(bars["HO-P17"]["comparator"].get("H")))

# ---------------------------------------------------------------- reading
say("\nCross-checks (S1 statistic vs the package's own file; max |d|, n)")
for name, dmax, n in xchk:
    say(f"  {name}: {dmax:.2e} (n={n})")
V = {r["id"]: r["verdict"] for r in res}
guards = [r["id"] for r in res if r["cls"] == "guard"]
g_fail = [g for g in guards if V[g].startswith(("fail", "FAIL"))]
g_cost = [g for g in guards if V[g].startswith("cost")]
atrn, hoa1, mcs = V.get("A-trn"), V.get("HO-A1"), V.get("M-cs")
if mcs == "fail":
    reading = "M-cs FAILED: the loss is not weighted as registered — stop, check the recipe before reading skill."
elif atrn == "fail":
    reading = "A-trn FAILED: the arc family did not train — stop, discuss (no downstream refit)."
elif atrn == "pass" and hoa1 != "pass":
    reading = "A-trn passes, HO-A1 does not: the arcs fitted but did not generalize — stop, discuss (no refit)."
elif g_fail:
    reading = (f"guard readable fail(s) {g_fail}: run the S1-noarcs control (same seed) before attributing "
               "them to the arcs.")
elif atrn == "pass" and hoa1 == "pass":
    reading = ("A-trn + HO-A1 pass and every guard holds" + (f" (recorded costs to discuss: {g_cost})" if g_cost else "")
               + ": S1 is the new CalSim dPL — tier-2 parts passes, v2/v3 refits, PREREG_HYBRID_ALT_v2.")
else:
    reading = "incomplete readout (a bar is n/a) — no reading."
say(f"\nREADING: {reading}")
(out / "readout.json").write_text(json.dumps(dict(run=str(run), bars_file=a.bars, results=res, cross_checks=xchk,
                                                  reading=reading), indent=1, default=str) + "\n")
(out / "readout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
ent.to_csv(out / "s1_train_entity_kge.csv", index=False)
ho.to_csv(out / "s1_holdout_entities.csv", index=False)
if t2 is not None:
    t2.to_csv(out / "s1_tier2_windows.csv", index=False)
print(f"wrote {out / 'readout.json'} and readout.txt")
