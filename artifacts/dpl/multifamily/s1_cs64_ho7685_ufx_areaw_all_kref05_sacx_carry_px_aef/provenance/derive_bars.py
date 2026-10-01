"""S1 prereg: re-derive every comparator number from the run folders and write bars.json.

Comparators (read-only, LIVE artifacts/dpl/multifamily):
  H   = noah_cdec_uf_sacx_carry_px_aef                      CLEAN on the holdout (uf/uf-nested: WY1976-84)
  H95 = noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef LEAKY on the holdout (trained 50 USGS gauges
        inside WY1976-85); training-period guards only
  H S0 = run H's USGS-69 entity pass (hybrid Stage 0) for the USGS holdout comparator
Every statistic comes from s1lib.py, the module readout.py applies to the S1 run.
CPU only, no forward.  usage (cwd = wt_merge, PYTHONPATH = wt_merge): derive_bars.py [out.json]
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import s1lib as L  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "bars.json"
SPREAD = {"cdec_mean": 0.018, "uf_mean": 0.005, "entity": 0.04}   # PREREG_H95 seed spreads


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


comp, units, src = {}, {}, {}
for tag, rd in L.RUNS.items():
    c = L.run_cfg(rd)
    om = tuple(c["cfg"].get("obs_mask", ()))
    npz = rd / "sim_daily_mm.npz"
    ent = L.train_entity_kge(npz, om)
    fam = L.family_stats(ent)
    ho = L.holdout_entities(npz, om)
    tw = L.cdec_twins(npz)
    t1 = L.tier1_windows(rd)
    t2 = L.tier2_windows(rd)
    ar = L.anchor_rescaled(rd)
    comp[tag] = dict(epoch=c["epoch"], logged_sel=r4(c["cal_kge"]), fam=fam,
                     sel3_h95_shares=r4(L.shares_stat(fam, L.SEL3_H95)))
    units[tag] = dict(ent=ent, ho=ho, tw=tw, t1=t1, t2=t2, ar=ar)
    src[tag] = {"run": str(rd), "best.pt": sha(rd / "checkpoints" / "best.pt"), "sim_daily_mm.npz": sha(npz),
                "tier1_monthly.csv": sha(rd / "tier1" / "tier1_monthly.csv"),
                "tier2_monthly.csv": sha(rd / "tier2" / "tier2_monthly.csv")}
s0 = L.holdout_entities(L.H_S0_PASS)
src["H_S0"] = {"npz": str(L.H_S0_PASS), "sha": sha(L.H_S0_PASS)}

A = set(L.tier_a_arcs())
bars = []


def add(**b):
    bars.append(b)


def ent_kge(tag, e):
    d = units[tag]["ent"].set_index("entity_id")["kge"]
    return float(d[e])


# ---------------------------------------------------------------- mechanism (S1 logs; no comparator)
add(id="M-cs", cls="bar", statistic="e0 realized coefficient shares sum(c_f/kappa_f) over stepped chunks, "
    "all 4 families", window="epoch 0, 60 stepped chunks", source="S1 chunk_log.csv",
    target=L.AREA4, tol=0.01, rule="every family within +/-0.01 of its footprint-area share",
    comparator=None)
add(id="M-mass", cls="report", statistic="loss-mass shares sum(l_f)/sum(l) per epoch (kappa scale)",
    window="every epoch; flag epochs >= 10 with calsim outside [0.5x, 2x] its e0 share",
    source="S1 chunk_log.csv", comparator=None)
add(id="M-clip", cls="bar", statistic="share of stepped chunks with gnorm > 1 in epochs >= 5",
    window="epochs >= 5", source="S1 chunk_log.csv", op="<", threshold=0.5, comparator=None,
    rule="< 0.5 and a finite best.pt selection (PREREG_H95 M3); flag any epoch skipping > 4 of 70 chunks")

# ---------------------------------------------------------------- training period (masked targets)
f95 = comp["H95"]["fam"]
add(id="S-sel", cls="flag", statistic="best epoch (checkpoint epoch); both selection scalars reported",
    window="selection epochs", source="S1 best.pt + train_log.csv", op="in", threshold=[40, 110],
    comparator=dict(run="H95", value=comp["H95"]["sel3_h95_shares"], what="sel3 (H95 shares .216/.558/.226) "
                    "re-scored on H95's evaluate sims, holdout-masked, registry windows", logged=0.8509,
                    note="S1's sel3 = the same rule at .2164/.5578/.2258 (moves it ~1e-4); not a bar"))
t2 = {k: units[k]["t2"] for k in units}
atr95 = t2["H95"][(t2["H95"].window == "arc_train") & t2["H95"].arc.isin(A)]["kge"]
add(id="A-trn", cls="bar", statistic="tier-A 64 arcs, KGE on their arc_obs_mask months (training WYs), "
    "tier-2 monthly TAF vs CalSim3 INFLOW; mean AND median", window="arc_obs_mask months (WY1950-75 + WY1986-2015)",
    source="tier2/tier2_monthly.csv", comparator=dict(run="H95 (untrained on arcs)", n=int(atr95.notna().sum()),
                                                      mean=r4(atr95.mean()), median=r4(atr95.median())),
    threshold=dict(mean=r4(atr95.mean() + 0.05), median=r4(atr95.median() + 0.03)), op=">=", spread=None,
    rule="mean >= H95 + 0.05 AND median >= H95 + 0.03; fail = the arcs did not train: stop, discuss",
    also_report="S1 metrics_entities.csv calsim_monthly kge (entity-cell basis, the trained target)")
cd = f95["cdec_daily"]
add(id="C1-mean", cls="guard", statistic="cdec family mean KGE (17), masked daily target",
    window="registry windows (WY1986+), obs_mask, holdout (0 cdec days)", source="sim_daily_mm.npz via load_entity_obs",
    comparator=dict(run="H95", value=r4(cd["mean"])), threshold=r4(cd["mean"] - 0.02), op=">=",
    spread=SPREAD["cdec_mean"], also=dict(H=r4(comp["H"]["fam"]["cdec_daily"]["mean"])))
add(id="C1-median", cls="guard", statistic="cdec family median KGE (17)", window="as C1-mean",
    source="as C1-mean", comparator=dict(run="H95", value=r4(cd["median"])), threshold=r4(cd["median"] - 0.02),
    op=">=", spread=SPREAD["cdec_mean"], also=dict(H=r4(comp["H"]["fam"]["cdec_daily"]["median"])))
for e in ("cdec_ORO", "cdec_YRS", "cdec_SHA", "cdec_BND", "cdec_NHG"):
    v = ent_kge("H95", e)
    add(id=f"C1-{e[5:]}", cls="guard", statistic=f"{e} KGE, masked daily target", window="as C1-mean",
        source="as C1-mean", comparator=dict(run="H95", value=r4(v)), threshold=r4(v - 0.04), op=">=",
        spread=SPREAD["entity"], also=dict(H=r4(ent_kge("H", e))),
        note=("H95's known cost vs H; reported against H as well" if e == "cdec_NHG" else None))
uf95 = units["H95"]["ent"][units["H95"]["ent"].family == "uf_monthly"].set_index("entity_id")["kge"]
add(id="U1-mean", cls="guard", statistic="uf family mean KGE (9) on the registry window minus the holdout "
    "= WY1986-2014 (S1: metrics_entities.csv kge_reg)", window="WY1986-2014 (348 months)",
    source="sim_daily_mm.npz via load_entity_obs (no back-extension)",
    comparator=dict(run="H95", value=r4(uf95.mean()), note="H95 on the SAME WY1986-2014 window (its logged "
                    "uf 0.9351 was WY1985-2014)"), threshold=r4(uf95.mean() - 0.01), op=">=",
    spread=SPREAD["uf_mean"], also=dict(H=r4(comp["H"]["fam"]["uf_monthly"]["mean"])))
add(id="U1-entity", cls="guard", statistic="no uf entity down > 0.04 vs H95, kge_reg WY1986-2014",
    window="WY1986-2014", source="as U1-mean", comparator=dict(run="H95", per_unit={k: r4(v) for k, v in uf95.items()}),
    threshold={k: r4(v - 0.04) for k, v in uf95.items()}, op=">=", spread=SPREAD["entity"])
us = f95["usgs_daily"]
add(id="G1u-mean", cls="guard", statistic="usgs family mean KGE (69), masked daily target (holdout out)",
    window="registry windows minus WY1976-85", source="sim_daily_mm.npz via load_entity_obs",
    comparator=dict(run="H95", value=r4(us["mean"])), threshold=r4(us["mean"] - 0.03), op=">=",
    fail_below=r4(us["mean"] - 0.05), spread=None,
    rule="below the threshold = the recorded cost of holdout + arcs (discuss); below fail_below FAILS")
add(id="G1u-median", cls="guard", statistic="usgs family median KGE (69)", window="as G1u-mean",
    source="as G1u-mean", comparator=dict(run="H95", value=r4(us["median"])), threshold=r4(us["median"] - 0.03),
    op=">=", spread=None)

# ---------------------------------------------------------------- holdout WY1976-85
own_h = t2["H"][(t2["H"].window == "WY1976-85_own") & (t2["H"].n >= 24)]
ho_a = own_h[own_h.arc.isin(A)]
own_95 = t2["H95"][(t2["H95"].window == "WY1976-85_own") & t2["H95"].arc.isin(ho_a.arc)]
add(id="HO-A1", cls="bar", primary=True, statistic="median KGE of the tier-A arcs with >= 24 own-record months "
    "inside WY1976-85 (fixed arc list below), tier-2 monthly TAF vs CalSim3 INFLOW on those months",
    window="WY1976-85 own-record months (calsim_arcs.own_record_months)", source="tier2/tier2_monthly.csv",
    arcs=sorted(ho_a.arc), comparator=dict(run="H", n=len(ho_a), value=r4(ho_a.kge.median())),
    threshold=r4(ho_a.kge.median() + 0.03), op=">=", spread=None,
    also=dict(H95_leaky=r4(own_95.kge.median()), H_all_own_arcs_ge24=dict(n=len(own_h), median=r4(own_h.kge.median()))))
w = {k: t2[k][(t2[k].window == "WY1976-85") & t2[k].has_ref] for k in t2}
nonA_h = w["H"][~w["H"].arc.isin(A)]
add(id="HO-A2-all", cls="guard", statistic="median KGE of every rim arc with a CalSim3 series (196)",
    window="WY1976-85, all months", source="tier2/tier2_monthly.csv",
    comparator=dict(run="H", n=int(w["H"].kge.notna().sum()), value=r4(w["H"].kge.median())),
    threshold=r4(w["H"].kge.median()), op=">=", spread=None,
    also=dict(H95_leaky=r4(w["H95"].kge.median()), H_tierA64=r4(w["H"][w["H"].arc.isin(A)].kge.median())))
add(id="HO-A2-nonA", cls="guard", statistic="median KGE of the non-tier-A arcs (132; regionalization check)",
    window="WY1976-85, all months", source="tier2/tier2_monthly.csv", arcs=sorted(nonA_h.arc),
    comparator=dict(run="H", n=int(nonA_h.kge.notna().sum()), value=r4(nonA_h.kge.median())),
    threshold=r4(nonA_h.kge.median() - 0.02), op=">=", spread=None)
hoh = units["H"]["ho"]
uf84 = hoh[(hoh.family == "uf_monthly") & (hoh.window == "WY1976-84")].set_index("entity_id")["kge"]
uf85 = hoh[(hoh.family == "uf_monthly") & (hoh.window == "WY1976-85")].set_index("entity_id")["kge"]
add(id="HO-U", cls="guard", statistic="uf family mean KGE (9), monthly depth vs DWR uf_monthly_mm",
    window="WY1976-84 (run H trained uf WY1985)", source="metrics_entities_holdout.csv (holdout_metrics)",
    comparator=dict(run="H", value=r4(uf84.mean()), per_unit={k: r4(v) for k, v in uf84.items()}),
    threshold=r4(uf84.mean() - 0.01), op=">=", spread=SPREAD["uf_mean"],
    also=dict(H_WY1976_85=r4(uf85.mean()), flag_units=["uf_02", "uf_20"],
              H95_leaky_WY1976_84=r4(units["H95"]["ho"].query("family == 'uf_monthly' and window == 'WY1976-84'").kge.mean())))
twh = units["H"]["tw"]
tw85 = twh[twh.window == "WY1976-85"].set_index("entity_id")["kge"]
add(id="HO-C-twins", cls="guard", statistic="median KGE of the 9 CDEC twins at DWR UF sites, monthly TAF "
    "(registry area) vs uf_monthly.csv", window="WY1976-85", source="sim_daily_mm.npz",
    comparator=dict(run="H", value=r4(tw85.median()), per_unit={k: r4(v) for k, v in tw85.items()}),
    threshold=r4(tw85.median() - 0.02), op=">=", spread=SPREAD["cdec_mean"])
t1h = units["H"]["t1"]
fu84 = t1h[(t1h.ref_kind == "anchor") & (t1h.window == "WY1976-84")].set_index("set_id")["kge"]
add(id="HO-C-FU", cls="guard", statistic="median KGE of the 10 tier-1 FLOW-UNIMPAIRED anchors, monthly TAF",
    window="WY1976-84 (uf-nested anchors; H trained uf WY1985)", source="tier1/tier1_monthly.csv",
    comparator=dict(run="H", value=r4(fu84.median()), per_unit={k: r4(v) for k, v in fu84.items()}),
    threshold=r4(fu84.median() - 0.02), op=">=", spread=SPREAD["cdec_mean"])
g_h = s0[(s0.family == "usgs_daily") & (s0.n >= 365)]
g_95 = units["H95"]["ho"].query("family == 'usgs_daily' and n >= 365")
add(id="HO-G", cls="guard", statistic="median daily KGE of the USGS gauges with >= 365 held-out days "
    "(registry window inside WY1976-85)", window="WY1976-85", source="S1 metrics_entities_holdout.csv; H: the "
    "hybrid Stage-0 base entity pass of run H (cycle spinup)", gauges=sorted(g_h.entity_id),
    comparator=dict(run="H (S0 pass)", n=len(g_h), value=r4(g_h.kge.median()), mean=r4(g_h.kge.mean())),
    threshold=r4(g_h.kge.median()), op=">=", spread=None,
    also=dict(H95_in_sample=r4(g_95.kge.median())), note="the S0 pass exists (09-29), so HO-G is a guard, "
    "not only a report; H95's value is in-sample and never a bar")
ar = {k: units[k]["ar"] for k in units}
arh = ar["H"][ar["H"].window == "WY1976-85"].dropna(subset=["kge_resc"])
add(id="HO-R", cls="report", statistic="anchor-rescaled arc KGE beside absolute (same months) + train-to-holdout "
    "gaps per family vs H", window="WY1976-85 (+ own-record months)",
    source="tier2/tier2_anchor_rescaled.csv (calsim_tier2.anchor_rescaled)",
    comparator=dict(run="H", n=len(arh), abs_median=r4(arh.kge_abs.median()), resc_median=r4(arh.kge_resc.median()),
                    gaps=dict(uf=r4(comp["H"]["fam"]["uf_monthly"]["mean"] - uf84.mean()),
                              cdec_twins=r4(twh[twh.window == "WY1986-2014"].kge.median() - tw85.median()),
                              fu_anchors=r4(t1h[(t1h.ref_kind == "anchor") & (t1h.window == "WY1986-2014")].kge.median()
                                            - fu84.median()),
                              tierA_arcs=r4(t2["H"][(t2["H"].window == "arc_train") & t2["H"].arc.isin(A)].kge.median()
                                            - w["H"][w["H"].arc.isin(A)].kge.median()))))
p17 = {}
for tag, fn in (("H", "noah_cdec_uf_sacx_carry_px_aef"), ("H95", "noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef")):
    f = L.P17B_DIR / f"{fn}_sim_1925.npz"
    if f.exists():
        p = L.p17b_sites(f)
        p17[tag] = {era: r4(g.kge.median()) for era, g in p.groupby("era")}
        src[f"P17b_{tag}"] = {"npz": str(f), "sha": sha(f)}
add(id="HO-P17", cls="report", statistic="P17b: median KGE over the 18 DWR UF sites by era (monthly TAF)",
    window=", ".join(L.P17_ERAS), source="p17/<run>_sim_1925.npz (p17b_s1.py)", comparator=p17)

res = dict(
    written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    tool="prereg/derive_bars.py", holdout_wy=list(L.HO), data_dir=L.DATA, sources=src,
    spreads=SPREAD,
    readability=("|S1 - comparator| < spread => 'unreadable' (PREREG_H95 rule); a guard is HELD unless a "
                 "readable fail; a bar PASSES only on a readable pass; no spread = literal"),
    known_input_change=("the region forcing may gain 92 misplaced-decimal x10 precip cell-days (1916-1992; "
                        "wgen_scen/x10) before launch: every comparator here comes from H / H95 run on the "
                        "UNFIXED store and none is re-derived; 4 pairs fall in WY1976-85 (1976-08-19, "
                        "1980-07-02/03, CalSim3-footprint cells, no entity), 78 in WY1950-75 (S1 training, "
                        "incl. cs_I_HON021 +151.7 mm basin mean on 1974-07-08, an arc_obs_mask month)"),
    comparators=comp, bars=bars)
OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
# per-unit tables beside the json
for tag, u in units.items():
    for k, df in u.items():
        df.to_csv(OUT.parent / f"units_{tag}_{k}.csv", index=False)
s0.to_csv(OUT.parent / "units_H_S0_ho.csv", index=False)
print(f"wrote {OUT} ({len(bars)} bars)")
for b in bars:
    print(f"  {b['id']:12s} {b['cls']:9s} thr={b.get('threshold')!s:.60s} comp={json.dumps(b.get('comparator'), default=str)[:150]}")
