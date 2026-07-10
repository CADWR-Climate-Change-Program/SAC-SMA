"""Cross-compare: CalSim3 (actual) vs VIC vs multi-set SAC-SMA at the CalSim nodes.

One consolidated benchmark.  CalSim3 historical ``INFLOW`` is the **reference**
("truth"); the **VIC** routed historical and **each SAC-SMA calibration set**
(15cdec / 9unimp / 11obs — kept SEPARATE) are scored against it, so the question
is how the SAC-SMA sets stack up against VIC and against each other when all are
measured against the same CalSim3 target.

Monthly inflow per CalSim node (TAF/month):
  * **calsim3** — CalSim3 historical ``INFLOW`` (``data/reference/calsim3_inflow_monthly.csv``) — REFERENCE.
  * **<set>**   — a SAC-SMA calibration set's CalSim run (``catchments.run_calsim`` live,
    daily local-runoff cfs aggregated to monthly volume), one ``source`` per set.
  * **vic**     — VIC routed historical (``data/reference/vic_routed_monthly.csv``).

Nodes are matched on the CalSim arc id ``I_<node>``; VIC names are resolved
through the hand-edited ``data/reference/calsim_crosswalk.csv`` (its ``vic_basin``
column).  Per-node scores go to the CSVs (``calset_metrics.csv`` etc.); the **maps and
figures present skill at the main-basin level** — every sub-area polygon is coloured by
its watershed's basin-anchor score (:func:`anchor_metrics`), not its own sub-arc score.
Artifacts -> ``artifacts/calsim/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cdec15 import DOMAIN as CDEC15
from ..io import read_table
from ..metrics import center_of_timing, kge, nse, pbias, pearson, seasonal_mismatch
from . import load_calsim3_monthly, load_vic_monthly

#: 1 cfs sustained for one day = 1.9834711 acre-feet.
_AF_PER_CFS_DAY = 1.9834711


def _cfs_day_to_taf(cfs_day):
    """Monthly cfs-day volume (sum of daily mean cfs over a month) -> thousand acre-feet."""
    return cfs_day * _AF_PER_CFS_DAY / 1000.0


#: the few CalSim node -> VIC node-name differences (VIC names are otherwise identical to
#: the CalSim arc).  This is **per-node** resolution for the per-catchment VIC comparison;
#: the **basin-level** VIC anchor (8-River major-basin totals) is the crosswalk ``vic_basin``.
_VIC_NODE_ALIAS = {"I_EBTML": "I_WBF006", "I_JBP006": "I_ANTLP",
                   "I_MLRTN_IMP": "I_MLRTN", "I422": "I_MSH015"}


def vic_node_name(arc) -> str:
    """VIC series name for a single CalSim node (identity, with a few legacy aliases)."""
    return _VIC_NODE_ALIAS.get(str(arc), str(arc))


def load_name_map(data_dir: str | Path = "data") -> dict[str, str]:
    """arc -> VIC **major-basin** series, from the crosswalk's ``vic_basin`` column (the
    single source of truth) — used for the basin-level anchor VIC comparison.  For
    per-node VIC use :func:`vic_node_name` instead."""
    from .catchments import load_crosswalk

    d = load_crosswalk(data_dir)
    m = dict(zip(d["arc"].astype(str), d["vic_basin"].astype(str)))
    return {k: v for k, v in m.items() if v and v != "nan"}


# --------------------------------------------------------------------------
# Cross-compare: each calibration set kept SEPARATE, scored vs CalSim3
# --------------------------------------------------------------------------
#: default calibration sets fed to the combined CalSim check (unimpaired + observed).
#: CDEC15 (the 15-CDEC application's set) is an *input* here — the dependency is one-way
#: (sacsma.calsim may import sacsma.cdec15, never the reverse).
DEFAULT_CALSETS = (CDEC15, "9unimp", "11obs")

#: Sets that receive the per-sub-arc QMAP adjustment AND form the basin-map composite.
#: 15cdec is deliberately excluded — it keeps its own per-set map but never enters the
#: adjustment/composite, which use 9unimp + 11obs (9unimp wins on overlap).
ADJUST_SETS = ("9unimp", "11obs")


def _calset_monthly_taf(domain: str, data_dir: str | Path = "data", *, covered_frac=None,
                        comp_cache=None, parallel=False):
    """Per-catchment monthly TAF for the catchments this set's HRUs **own** and score.

    Scored catchments are the basin -> node mapping (:func:`load_basin_nodes`) restricted
    to real CalSim3 inflow nodes the set's HRUs actually cover, and with the **cumulative
    single-node systems excluded** (Merced ``I_MCLRE``, San Joaquin ``I_MLRTN``, Shasta,
    Trinity) — those are whole-basin nodes scored only in the basin-level anchor view, not
    per local catchment.  Also carries the series-less valley-accretion nodes (``I_<SYS>_VAL``)
    for the sub-arc QMAP.  Returns ``(node_monthly, scored)``: ``node_monthly`` is monthly
    TAF per node (**raw**, true-area runoff — the coverage-map nudge is applied downstream in
    :func:`make_all`); ``scored`` carries ``[set, cid, node, arc, basin, kind, cov_frac]`` for
    the maps.  ``covered_frac`` only affects the internal :func:`run_calsim` weighting.
    """
    from .catchments import COVERED_FRAC, MERGED_LAYER, is_valley_arc, run_calsim, series_arc

    cf = COVERED_FRAC if covered_frac is None else covered_frac
    # the MERGED layer makes the cumulative single-node systems whole catchments, so they
    # are scored as one piece (Merced runoff vs I_MCLRE) instead of a sliver / grey hole.
    flows, cov, _map = run_calsim(data_dir, domain=domain, layer=MERGED_LAYER, covered_frac=cf,
                                  comp_cache=comp_cache, parallel=parallel)
    flows["arc"] = flows["node"].map(series_arc)
    nodes = load_basin_nodes(data_dir, domain)
    scored = nodes[nodes["in_calsim3"].astype(bool)].copy()   # cumulative now whole -> kept
    scored_arcs = set(scored["arc"])
    # also carry the series-less valley-accretion nodes (I_<SYS>_VAL): not scored on the maps,
    # but the sub-arc QMAP needs them so the real tributaries don't absorb the valley runoff.
    valley_arcs = {a for a in flows["arc"].unique() if is_valley_arc(a)}
    # per-arc monthly TAF over the scored + valley catchments (join on arc -> layer-robust)
    f = flows[flows["arc"].isin(scored_arcs | valley_arcs)].drop_duplicates(["arc", "date"]).copy()
    f["month"] = f["date"].dt.to_period("M")
    g = f.groupby(["arc", "month"], observed=True)["flow_cfs"].sum().reset_index()
    g["flow_taf"] = _cfs_day_to_taf(g["flow_cfs"])
    g["date"] = g["month"].dt.to_timestamp("M")
    g["node"] = g["arc"].str.slice(2)
    g["source"] = domain
    # keep only the scored arcs this set's HRUs actually cover (produced a flow)
    scored = scored[scored["arc"].isin(set(g["arc"]))].copy()
    scored["set"] = domain
    # attach the HONEST coverage this set's HRUs give each scored catchment (the crosswalk's
    # cov_frac is NaN); low cov_frac flags an extrapolated, less trustworthy per-node score.
    cov["arc"] = cov["node"].map(series_arc)
    cmap = cov.drop_duplicates("arc").set_index("arc")
    scored["cov_frac"] = scored["arc"].map(cmap["cov_frac"]).round(3)
    scored["n_hru"] = scored["arc"].map(cmap["n_hru"]).fillna(0).astype(int)
    return (g[["date", "node", "arc", "source", "flow_taf"]],
            scored[["set", "cid", "node", "arc", "basin", "kind", "cov_frac", "n_hru"]])


#: the 6 distributed rim systems that have a FLOW-UNIMPAIRED disaggregation anchor in
#: RimInflowAnchor (cumulative single-node systems SHAS/TRIN/ME/SJ need no sub-arc split).
_DISTRIB_RIM = ("FOLS", "OROV", "SRBB", "YUBA", "ST", "TU")


def _apply_anchor_mass_balance(long, data_dir, sets, anchor_long):
    """Proportional sub-arc (anchor mass-balance) adjustment — CalSim's rim-inflow
    `enforce_anchor_mass_balance` (`_2_qmap_historical_validation.py`).

    Within each distributed rim system, rescale **every estimate's** sub-arc flows so they
    sum to that system's anchor total, in proportion to each sub-arc's own value
    (``trib_adj = trib * anchor / sum_tribs``).  The anchor is the estimate's OWN basin
    aggregate — SAC-SMA: the area-nudged ``run_basin`` total (from ``anchor_long``); VIC:
    the 8-River index series — matching the reference's use of each estimate's own anchor.
    CalSim3 (the actual) and non-distributed arcs pass through unchanged."""
    from .catchments import BASIN_RIM_SYSTEM, load_crosswalk

    distrib = set(_DISTRIB_RIM)
    arc2sys = {str(a): s for a, s in zip(*[load_crosswalk(data_dir)[c] for c in ("arc", "system")])
               if s in distrib}

    def mend(s):
        return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("M")

    anchors: dict[tuple, float] = {}
    if anchor_long is not None and len(anchor_long):
        al = anchor_long.assign(date=mend(anchor_long["date"]))
        for st in sets:
            inv = {b: sy for b, sy in BASIN_RIM_SYSTEM.get(st, {}).items() if sy in distrib}
            a = al[(al["set"] == st) & (al["source"] == st) & (al["basin"].isin(inv))]
            for b, d, v in zip(a["basin"], a["date"], a["flow_taf"]):
                anchors[(st, inv[b], d)] = v
    vic = load_vic_monthly(data_dir).assign(date=lambda d: mend(d["date"]))
    for sy in distrib:
        s = vic[vic["vic_name"] == UNIMP_MAP[sy]["vic"][0]]   # the system's 8-River series
        for d, v in zip(s["date"], s["flow_taf"]):
            anchors[("vic", sy, d)] = v

    out = long.copy()
    out["_sys"] = out["arc"].map(arc2sys)
    est = out[(out["source"] != "calsim3") & out["_sys"].notna()].copy()
    if len(est):
        est["_m"] = mend(est["date"])
        est["_sum"] = est.groupby(["source", "_sys", "_m"])["flow_taf"].transform("sum")
        anc = np.array([anchors.get((src, sy, d), np.nan)
                        for src, sy, d in zip(est["source"], est["_sys"], est["_m"])], float)
        f = np.where((est["_sum"].to_numpy() > 0) & np.isfinite(anc),
                     anc / est["_sum"].to_numpy(), 1.0)
        out.loc[est.index, "flow_taf"] = est["flow_taf"].to_numpy() * f
    return out.drop(columns="_sys")


def build_calsets_long(
    data_dir: str | Path = "data", sets=DEFAULT_CALSETS, *, covered_frac=None,
    anchor_long=None, mass_balance=False, comp_cache=None, parallel=False
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Long [date, arc, node, source, flow_taf] for each calibration set + CalSim3 + VIC.

    ``source`` is the calibration set name, ``calsim3`` (the actual), or ``vic``.
    Restricted to the common period across all present sources.  Also returns the
    stacked per-set coverage table.  ``covered_frac`` overrides the "covered"
    threshold (None -> ``calsim.COVERED_FRAC``).  When ``anchor_long`` is provided, the
    **proportional sub-arc (anchor mass-balance) adjustment** is applied to every estimate
    (SAC sets + VIC) within the distributed rim systems (:func:`_apply_anchor_mass_balance`).
    """
    per = [_calset_monthly_taf(d, data_dir, covered_frac=covered_frac,
                               comp_cache=comp_cache, parallel=parallel)
           for d in sets]
    sac = pd.concat([p[0] for p in per], ignore_index=True)
    coverage = pd.concat([p[1] for p in per], ignore_index=True)
    sac_arcs = set(sac["arc"])
    arc_node = sac.drop_duplicates("arc").set_index("arc")["node"].to_dict()

    c3 = load_calsim3_monthly(data_dir)
    c3 = c3[c3["arc"].isin(sac_arcs)].assign(source="calsim3", node=lambda d: d["arc"].map(arc_node))

    vic = load_vic_monthly(data_dir)
    vic_avail = set(vic["vic_name"])
    rows = []
    for arc in sac_arcs:
        vname = vic_node_name(arc)   # per-node VIC series (identity + legacy aliases)
        if vname in vic_avail:
            s = vic[vic["vic_name"] == vname]
            rows.append(pd.DataFrame({"date": s["date"].to_numpy(), "arc": arc,
                                      "node": arc_node.get(arc), "flow_taf": s["flow_taf"].to_numpy()}))
    vicL = pd.concat(rows, ignore_index=True).assign(source="vic") if rows else \
        pd.DataFrame(columns=["date", "arc", "node", "flow_taf", "source"])

    long = pd.concat([
        sac[["date", "arc", "node", "source", "flow_taf"]],
        c3[["date", "arc", "node", "source", "flow_taf"]],
        vicL[["date", "arc", "node", "source", "flow_taf"]],
    ], ignore_index=True)

    bounds = long.groupby("source")["date"].agg(["min", "max"])
    start, end = bounds["min"].max(), bounds["max"].min()
    long = long[(long["date"] >= start) & (long["date"] <= end)].reset_index(drop=True)
    if mass_balance and anchor_long is not None:
        long = _apply_anchor_mass_balance(long, data_dir, sets, anchor_long)
    matched = sorted(set(c3["arc"]) & sac_arcs)
    return long, matched, coverage


def subarc_validation_metrics(
    data_dir: str | Path = "data", sets=DEFAULT_CALSETS, *,
    anchor_long=None, raw_long=None,
    train=("1921-10-01", "1971-09-30"), test=("1971-10-01", "2018-12-31"),
    ratio_clip=(0.1, 10.0), method="qmap",
) -> pd.DataFrame:
    """Per-sub-arc bias-correction **validation** (train/test split), applied to **every
    multi-arc basin** (≥2 in_calsim3 sub-arcs) in the crosswalk — the 6 distributed rim
    systems *and* the multi-arc secondary basins (15cdec MKM, 11obs BLB, the 9unimp creeks
    Mokelumne/Bear/Cache/Cosumnes/Stony) — for each SAC set AND VIC.

    Two faithful-to-CalSim steps, learned on the ``train`` water years and scored on the
    held-out ``test`` years:

    1. **Quantile mapping** (``method="qmap"``, default) — each sub-arc is mapped, per
       calendar month, from its own distribution onto its CalSim3 ``INFLOW`` distribution
       (empirical CDF within range, gamma tail beyond; :func:`sacsma._qmap.qmap_series`, a
       port of CalSim's ``utils/quantile_mapping.qmap_single``).
    2. **Mass-balance to the SAC-SMA simulated basin total** — the QMAPped sub-arcs are
       proportionally rescaled so each basin sums to **that estimate's own simulated
       unimpaired total**: the **un-nudged** ``run_basin`` series for a SAC set, the basin's
       VIC total for VIC (both from ``anchor_long``).  QMAP fixes the per-catchment *shape*
       while the estimate keeps its own basin *volume*.  This target is **not** the CalSim
       base (FLOW-UNIMPAIRED) and **not** the estimate's raw sub-arc sum.

    ``method="ratio"`` reproduces the previous release exactly (a multiplicative monthly
    mean-ratio renormalized to the group's own raw sub-arc sum).  VIC is corrected once per
    arc (deduplicated across sets).  Returns per (set, arc) raw-vs-corrected KGE/NSE/pbias on
    the **test** period only, plus the ``anchor_kind`` used (``sac_sim``/``vic_sim``/``own_sum``).
    A **nested cumulative inflow** is included in each basin that lists it — e.g. ``I_SHSTA`` is
    both its own SHA basin *and* a Bend Bridge sub-arc (``BASIN_NESTS``) — so a cumulative basin's
    sub-arcs (local tributaries + the upstream rim inflow) reconstruct its ``run_basin`` total.
    Such a nested inflow is **held fixed**: because Shasta is independently modeled as its own SHA
    basin, ``I_SHSTA`` is passed through **raw** (not QMAPped) and its volume is subtracted from the
    Bend Bridge anchor, so step 2's mass balance only redistributes the *remaining* basin volume
    across Bend Bridge's other (local) sub-arcs.
    The full-period corrected sub-arc **series** are emitted per-set (``subarc_qmap_<set>.csv``)
    by :func:`make_subarc_validation`; :func:`_subarc_validate` returns both.
    """
    return _subarc_validate(data_dir, sets, anchor_long=anchor_long, raw_long=raw_long,
                            train=train, test=test, ratio_clip=ratio_clip, method=method)[0]


def _subarc_validate(
    data_dir: str | Path = "data", sets=DEFAULT_CALSETS, *,
    anchor_long=None, raw_long=None,
    train=("1921-10-01", "1971-09-30"), test=("1971-10-01", "2018-12-31"),
    ratio_clip=(0.1, 10.0), method="qmap",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engine for :func:`subarc_validation_metrics`.  Returns ``(met, series)``: the
    test-period raw-vs-corrected metrics frame, and the **full-period** raw + corrected
    sub-arc long frame ``[date, set, arc, node, basin, anchor_kind, flow_taf_raw,
    flow_taf_qmap]`` for every SAC set + VIC (the QMAP deliverable)."""
    from .catchments import (
        BASIN_NESTS,
        BASIN_RIM_SYSTEM,
        VALLEY_SYSTEMS,
        derive_basin_nodes,
        valley_arc_for_system,
    )

    L = raw_long if raw_long is not None else build_calsets_long(data_dir, sets)[0]
    tr0, tr1 = pd.Timestamp(train[0]), pd.Timestamp(train[1])
    te0, te1 = pd.Timestamp(test[0]), pd.Timestamp(test[1])
    lo, hi = ratio_clip

    c3 = L[L["source"] == "calsim3"].pivot_table(index="date", columns="arc", values="flow_taf")
    Evic = L[L["source"] == "vic"].pivot_table(index="date", columns="arc", values="flow_taf")

    # QMAP mass-balance target: each estimate's own simulated basin total -> {(source, basin):
    # Series-by-month}.  Per the user's choice the area nudge enters HERE, on the anchor basin
    # total (the run_basin series carries it; VIC uses its own 8RI total) — and ONLY here; the
    # sub-arc INPUTS above are the un-nudged true-area per-catchment runoff.
    sac_tot: dict[tuple[str, str], pd.Series] = {}
    if method == "qmap":
        al = (anchor_long if anchor_long is not None
              else build_anchor_long(data_dir, sets, footprint=_screened_fp(data_dir, sets)))
        for (st_, basin_, src_), g in al.groupby(["set", "basin", "source"]):
            if src_ == "calsim3":                    # the CalSim base is NOT the target
                continue
            per = pd.PeriodIndex(pd.to_datetime(g["date"]), freq="M")
            s = pd.Series(g["flow_taf"].to_numpy(float), index=per).groupby(level=0).sum()
            sac_tot[("vic" if src_ == "vic" else str(st_), str(basin_))] = s

    def correct_ratio(E, arcs):
        """Legacy: per-sub-arc monthly mean-ratio correction renormalized to the group's OWN
        raw sub-arc total each month (the prior release's behaviour)."""
        corr = pd.DataFrame(index=E.index)
        tr = (E.index >= tr0) & (E.index <= tr1)
        for a in arcs:
            e, r = E[a], c3[a]
            d = pd.DataFrame({"e": e, "r": r, "m": e.index.month})
            dd = d[tr & d["e"].notna() & d["r"].notna()]
            gm = dd.groupby("m").agg(em=("e", "mean"), rm=("r", "mean"))
            ratio = (gm["rm"] / gm["em"]).clip(lo, hi).where(gm["em"] > 0, 1.0)
            corr[a] = e * d["m"].map(ratio).fillna(1.0)
        raw_sum = E[arcs].sum(axis=1)               # preserve the group's own basin total
        csum = corr.sum(axis=1)
        factor = (raw_sum / csum).where(csum > 0, 1.0)
        return corr.mul(factor, axis=0), "own_sum"

    def correct_qmap(E, arcs, set_label, basin, held=frozenset()):
        """QMAP each sub-arc toward its CalSim3 INFLOW (learned on the train years), then
        rescale so the basin sums to the estimate's SAC-SMA simulated total (``sac_tot``).

        Arcs in ``held`` (a nested cumulative inflow inherited via ``BASIN_NESTS`` — i.e.
        ``I_SHSTA`` in Bend Bridge, which is its own well-modeled SHA basin) are **passed
        through raw**: not QMAPped and not rescaled.  Their fixed volume is subtracted from
        the basin anchor so the mass balance only redistributes the *remaining* volume
        across the other (local) sub-arcs."""
        from .qmap import qmap_series

        tr = (E.index >= tr0) & (E.index <= tr1)
        corr = pd.DataFrame(index=E.index)
        for a in arcs:
            if a in held:                           # nested rim inflow (e.g. Shasta) -> raw
                corr[a] = E[a]
                continue
            e = E[a]
            r = c3[a].reindex(E.index)              # align CalSim3 to the estimate's grid
            corr[a] = qmap_series(e, e[tr], r[tr])
        free = [a for a in arcs if a not in held]
        csum = corr[free].sum(axis=1)               # only the free (QMAPped) arcs absorb the rescale
        held_sum = (corr[list(held)].sum(axis=1)    # fixed volume held back from the anchor
                    if held else pd.Series(0.0, index=E.index))
        tgt = sac_tot.get((set_label, basin))
        if tgt is not None:
            per = pd.PeriodIndex(pd.to_datetime(E.index), freq="M")
            anchor = pd.Series(tgt.reindex(per).to_numpy(), index=E.index)
            kind = "vic_sim" if set_label == "vic" else "sac_sim"
        else:                                       # graceful fallback: group's own raw sum
            anchor = E[arcs].sum(axis=1)
            kind = "own_sum"
        remaining = anchor - held_sum               # volume left for the free sub-arcs
        factor = (remaining / csum).where((csum > 0) & anchor.notna() & (remaining > 0), 1.0)
        corr[free] = corr[free].mul(factor, axis=0)
        return corr[arcs], kind

    def correct(E, arcs, set_label, basin, held=frozenset()):
        if method == "ratio":
            return correct_ratio(E, arcs)
        return correct_qmap(E, arcs, set_label, basin, held)

    def score_rows(E, Ec, arcs, set_label, basin, anchor_kind):
        te = (E.index >= te0) & (E.index <= te1)
        out = []
        for a in arcs:
            if a not in c3.columns:
                continue
            rr = c3[a]
            m = te & E[a].notna() & Ec[a].notna() & rr.notna()
            if int(m.sum()) < 12:
                continue
            er, cr, ac = E[a][m].to_numpy(), Ec[a][m].to_numpy(), rr[m].to_numpy()
            out.append({"set": set_label, "arc": str(a), "node": str(a)[2:], "basin": basin,
                        "anchor_kind": anchor_kind, "n_test": int(m.sum()),
                        "kge_raw": kge(er, ac), "nse_raw": nse(er, ac), "pbias_raw": pbias(er, ac),
                        "kge_corr": kge(cr, ac), "nse_corr": nse(cr, ac), "pbias_corr": pbias(cr, ac),
                        # signed mean-monthly volume bias (TAF), raw vs corrected
                        "tafbias_raw": float(er.mean() - ac.mean()),
                        "tafbias_corr": float(cr.mean() - ac.mean())})
        return out

    def series_rows(E, Ec, arcs, set_label, basin, anchor_kind):
        """Full-period raw + corrected sub-arc series (the QMAP deliverable)."""
        return [pd.DataFrame({"date": E.index, "set": set_label, "arc": str(a),
                              "node": str(a)[2:], "basin": basin, "anchor_kind": anchor_kind,
                              "flow_taf_raw": E[a].to_numpy(), "flow_taf_qmap": Ec[a].to_numpy()})
                for a in arcs if a in Ec.columns]

    rows: list = []
    series_parts: list = []
    vic_seen: set = set()
    for st in sets:
        nodes = derive_basin_nodes(data_dir, st)
        nodes = nodes[nodes["in_calsim3"].astype(bool)]
        Eset = L[L["source"] == st].pivot_table(index="date", columns="arc", values="flow_taf")
        cols = set(Eset.columns) & set(c3.columns)
        # Group by the nodes frame, NOT a per-arc dict: an arc can belong to several basins (a
        # nested cumulative inflow like I_SHSTA is both its own SHA basin AND a Bend Bridge
        # sub-arc, BASIN_NESTS), so it must join EACH basin that lists it — otherwise BND's
        # sub-arcs miss Shasta and can't reconstruct its run_basin total.
        sysmap = BASIN_RIM_SYSTEM.get(st, {})
        byb: dict[str, list] = {}
        valley_of: dict[str, str] = {}
        for basin, sub in nodes.groupby("basin"):
            a = [x for x in dict.fromkeys(sub["arc"].astype(str)) if x in cols]
            if not a:
                continue
            # explicit modeled valley-accretion node (I_<SYS>_VAL, series-less, not in `cols`)
            # for a rim basin whose CalSim control point carries ungauged main-stem accretion
            # (Bend Bridge): append it so it participates in the basin mass-balance, HELD.
            sysn = sysmap.get(str(basin))
            if sysn in VALLEY_SYSTEMS:
                va = valley_arc_for_system(sysn)
                if va in Eset.columns:
                    a = a + [va]
                    valley_of[str(basin)] = va
            byb[str(basin)] = a
        for basin, arcs in byb.items():
            if len(arcs) < 2:               # multi-arc basins only
                continue
            # arcs inherited from a nested basin (BASIN_NESTS: BND ⊇ SHA) are held FIXED — a
            # well-modeled cumulative inflow like Shasta passes through raw (no QMAP, no
            # rescale) and only the local sub-arcs absorb the basin mass balance.  The modeled
            # valley-accretion node is held the same way (a real SAC inflow over the ungauged
            # area), so the real tributaries no longer absorb the valley runoff.
            nested = BASIN_NESTS.get(str(basin), [])
            held = set(nodes.loc[nodes["basin"].isin(nested), "arc"].astype(str)) & set(arcs)
            if basin in valley_of:
                held = held | {valley_of[basin]}
            cb, kind = correct(Eset, arcs, st, basin, held)
            rows += score_rows(Eset, cb, arcs, st, basin, kind)
            series_parts += series_rows(Eset, cb, arcs, st, basin, kind)
            # VIC under this basin grouping, deduplicated across sets (rim arcs are shared)
            varcs = [a for a in arcs if a in Evic.columns and a not in vic_seen]
            if len(varcs) >= 2:
                cbv, kindv = correct(Evic, varcs, "vic", basin, held & set(varcs))
                rows += score_rows(Evic, cbv, varcs, "vic", basin, kindv)
                series_parts += series_rows(Evic, cbv, varcs, "vic", basin, kindv)
                vic_seen.update(varcs)
    met = pd.DataFrame(rows)
    series = pd.concat(series_parts, ignore_index=True) if series_parts else pd.DataFrame(
        columns=["date", "set", "arc", "node", "basin", "anchor_kind",
                 "flow_taf_raw", "flow_taf_qmap"])
    return met, series


def make_subarc_validation(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts",
                           run: str = "compare", sets=DEFAULT_CALSETS, *,
                           anchor_long=None, raw_long=None, met=None, series=None,
                           method="qmap") -> Path:
    """Write the per-sub-arc bias-correction validation (train/test) under ``artifacts/<run>/``:
    ``subarc_validation_metrics.csv`` (the scorecard),
    and — for ``method="qmap"`` — the **per-set QMAP-corrected sub-arc series**
    ``subarc_qmap_<set>.csv`` (the deliverable, one file per SAC set and VIC, distinct from the
    legacy monthly-ratio approach).  ``method`` selects the correction (``qmap`` default,
    ``ratio`` legacy — see :func:`subarc_validation_metrics`).  ``met``/``series`` may be passed
    in (already computed by :func:`make_all`) to avoid recompute."""
    out = Path(artifacts_dir) / "calsim" / run
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if met is None:
        met, series = _subarc_validate(data_dir, sets, anchor_long=anchor_long,
                                       raw_long=raw_long, method=method)
    met.to_csv(out / "subarc_validation_metrics.csv", index=False)
    # per-set QMAP-corrected sub-arc series — the deliverable, distinct from the ratio approach
    if method == "qmap" and series is not None and len(series):
        for s in series["set"].unique():
            series[series["set"] == s].to_csv(out / f"subarc_qmap_{s}.csv", index=False)
    for s in list(sets) + ["vic"]:
        d = met[met["set"] == s]
        if len(d):
            print(f"subarc-validation [{s}]: n={len(d)} test sub-arcs | "
                  f"median KGE {d['kge_raw'].median():.3f}->{d['kge_corr'].median():.3f}  "
                  f"NSE {d['nse_raw'].median():.3f}->{d['nse_corr'].median():.3f}  "
                  f"|pbias| {d['pbias_raw'].abs().median():.1f}->{d['pbias_corr'].abs().median():.1f}%")
    return out


def _arc_choropleth(data_dir, value_by_arc, title, cb_label, path, *,
                    cmap="managua", vmin=0.0, vmax=1.0, annot=None):
    """Map a per-arc value (Series indexed by CalSim arc) onto the merged Rim catchments:
    faint grey context + the scored catchments coloured by the value.  ``cmap`` may be a
    name or a Colormap object (e.g. the grey-centred diverging map for difference views);
    ``annot`` passes through to :func:`_nse_choropleth`."""
    from .catchments import MERGED_LAYER, load_catchments, series_arc

    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).copy()
    catch["arc"] = catch["node"].map(series_arc)
    v = pd.Series(value_by_arc).dropna()
    covered = catch[catch["arc"].isin(v.index)].copy()
    covered["nse"] = covered["arc"].map(v)
    _nse_choropleth(catch, covered, title, cb_label, path,
                    extent=_map_extent(data_dir), cmap_name=cmap, vmin=vmin, vmax=vmax,
                    annot=annot)


#: short in-map labels for the long 9unimp basin names (the 11obs / 15cdec CDEC codes are
#: already short and pass through unchanged).
_BASIN_ABBREV = {
    "BearRiver": "Bear", "CacheCreek": "Cache", "CalaverasRiver": "Calav",
    "ChowchillaRiver": "Chow", "CosumnesRiver": "Cosum", "FresnoRiver": "Fresno",
    "MokelumneRiver": "Moke", "PutahCreek": "Putah", "StonyCreek": "Stony",
}


def _basin_label_points(geoms: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """basin -> (lon, lat) anchor for the in-map annotations: a point guaranteed INSIDE the
    union of the basin's coloured sub-area polygons (``representative_point`` — the centroid
    of a crescent-shaped watershed can fall outside it).  ``geoms`` = a GeoDataFrame with
    ``basin`` + ``geometry`` columns (one row per sub-area polygon)."""
    from shapely.ops import unary_union

    pts = {}
    for b, grp in geoms.dropna(subset=["basin"]).groupby("basin"):
        p = unary_union(list(grp.geometry)).representative_point()
        pts[str(b)] = (float(p.x), float(p.y))
    return pts


def arc_basin_map(data_dir, set_name) -> dict[str, str]:
    """arc -> the set's OWN basin, straight from the crosswalk's ``basin_<set>`` column.

    Nest-free by construction: a nested cumulative inflow keeps its home basin (``I_SHSTA``
    -> SHA, not Bend Bridge, unlike ``derive_basin_nodes`` which expands ``BASIN_NESTS``), so
    on the basin-coloured maps the Shasta catchment carries SHA's own anchor skill while Bend
    Bridge colours only its local sub-areas.  Series-less member arcs (e.g. ``I_RUB002``) are
    kept: every sub-area of a watershed takes the basin colour."""
    from .catchments import load_crosswalk

    cw = load_crosswalk(data_dir)
    col = f"basin_{set_name}"
    if col not in cw.columns:
        return {}
    return {str(a): str(b) for a, b in zip(cw["arc"], cw[col]) if pd.notna(b)}


def _composite_arc_partition(data_dir, sets=ADJUST_SETS) -> dict[str, tuple[str, str]]:
    """arc -> ``(set, basin)`` over the union of the ``sets`` footprints; where more than one
    set assigns an arc the FIRST listed wins (9unimp over 11obs — the only overlap is Stony
    Creek / BLB, assigned to 9unimp)."""
    part: dict[str, tuple[str, str]] = {}
    for st in sets:
        for a, b in arc_basin_map(data_dir, st).items():
            part.setdefault(a, (st, b))
    return part


def basin_map_metrics(anchor_met, part) -> pd.DataFrame:
    """The basin-level scores behind the composite maps: for each ``(set, basin)`` of the arc
    partition, the set's own anchor skill (``which="sac"``) and VIC's skill on the same basin
    (``which="vic"``), read straight from :func:`anchor_metrics`.  Returns
    ``[set, basin, which, ref_kind, n_months, kge, nse, pbias]``."""
    am = anchor_met.set_index(["set", "basin", "source"]).sort_index()
    rows = []
    for st, b in sorted(set(part.values())):
        for which, src in (("sac", st), ("vic", "vic")):
            if (st, b, src) not in am.index:
                continue
            r = am.loc[(st, b, src)]
            rows.append({"set": st, "basin": b, "which": which, "ref_kind": r["ref_kind"],
                         "n_months": int(r["n_months"]), "kge": float(r["kge"]),
                         "nse": float(r["nse"]), "pbias": float(r["pbias"])})
    return pd.DataFrame(rows)


#: basins left uncoloured (grey) on the composite basin maps: Bend Bridge's valley-floor
#: sub-areas would dominate the upper Sacramento visually, so the nested Sacramento system
#: shows only Shasta.  Scores for these basins remain in all CSVs and dumbbell figures.
MAP_EXCLUDE_BASINS = ("BND",)


def make_basin_maps(data_dir, out, anchor_met, sets=ADJUST_SETS):
    """Write the CalSim<->SAC-SMA **basin-level** maps (one PNG per metric) + the metrics CSV.

    Every sub-area polygon is coloured by its MAIN BASIN's anchor skill — the basin's
    ``run_basin`` total vs its faithful CalSim3 reference (:func:`anchor_metrics`) — so all
    sub-areas of a watershed share one colour.  The partition is the union of the gauge-
    calibrated ``sets`` (9unimp priority on overlap).  Three layers: the SAC-SMA composite,
    VIC on the same basins, and the SAC-SMA − VIC **difference** (RdBu diverging, blue = SAC
    better, white = tie, red = VIC better).

    :data:`MAP_EXCLUDE_BASINS` (Bend Bridge) is left uncoloured on the maps — for the nested
    Sacramento system only Shasta is drawn; the excluded basins keep their scores in
    ``basin_map_metrics.csv`` and every other anchor product.

    Each coloured watershed is annotated in place (abbreviated basin name + its value for
    the mapped metric, at a representative interior point of the basin)."""
    from .catchments import MERGED_LAYER, load_catchments, series_arc

    part = _composite_arc_partition(data_dir, sets)
    met = basin_map_metrics(anchor_met, part)
    met.to_csv(out / "basin_map_metrics.csv", index=False)
    if met.empty:
        return met
    part = {a: sb for a, sb in part.items() if sb[1] not in MAP_EXCLUDE_BASINS}
    fig = out / "figures"
    key = met.set_index(["which", "set", "basin"]).sort_index()

    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).copy()
    catch["arc"] = catch["node"].map(series_arc)
    catch["basin"] = catch["arc"].map({a: b for a, (_, b) in part.items()})
    pts = _basin_label_points(catch)

    def by_basin(which, m):
        """(set, basin) -> the basin's metric value."""
        return {sb: float(key.loc[(which, *sb), m]) for sb in sorted(set(part.values()))
                if (which, *sb) in key.index}

    def by_arc(vals):
        """Expand a per-basin metric onto that basin's arcs (all sub-areas = basin value)."""
        return pd.Series({a: vals[sb] for a, sb in part.items() if sb in vals}, dtype=float)

    def annot(vals, fmt):
        """In-map labels: abbreviated basin name + formatted value at the basin's anchor point."""
        return [(*pts[b], f"{_BASIN_ABBREV.get(b, b)}\n{fmt.format(v)}")
                for (_, b), v in vals.items() if b in pts and np.isfinite(v)]

    # NSE/KGE: plasma (perceptually-uniform) over 0..1 (negatives clamp to the
    # floor); signed pbias: BrBG diverging (brown=under, teal=over, 0=light),
    # centred over -20..20%.
    spec = {"nse": ("NSE", "plasma", 0.0, 1.0, "{:.2f}"),
            "kge": ("KGE", "plasma", 0.0, 1.0, "{:.2f}"),
            "pbias": ("pbias (%)", "BrBG", -20.0, 20.0, "{:+.0f}%")}
    label = {"sac": f"SAC-SMA ({'+'.join(sets)} basins)", "vic": "VIC"}
    fname = {"sac": "calsim_sacsma_map", "vic": "calsim_vic_map"}
    for which in ("sac", "vic"):
        for m, (mlab, cmap, vmin, vmax, vfmt) in spec.items():
            vals = by_basin(which, m)
            _arc_choropleth(data_dir, by_arc(vals),
                            f"{label[which]} — basin-level {mlab} vs CalSim3 anchor "
                            f"(all sub-areas coloured by their watershed)",
                            f"basin {mlab} vs CalSim3 anchor", fig / f"{fname[which]}_{m}.png",
                            cmap=cmap, vmin=vmin, vmax=vmax, annot=annot(vals, vfmt))
    for m, lim, vfmt, cb in [("nse", 0.2, "{:+.2f}", "NSE difference (SAC − VIC)  [+ = SAC better]"),
                             ("kge", 0.2, "{:+.2f}", "KGE difference (SAC − VIC)  [+ = SAC better]"),
                             ("pbias", 20.0, "{:+.0f}%",
                              "|pbias| reduction vs VIC (%)  [+ = SAC better]")]:
        s, v = by_basin("sac", m), by_basin("vic", m)
        if m == "pbias":                             # closer-to-zero is better -> compare |bias|
            d = {sb: abs(v[sb]) - abs(s[sb]) for sb in s if sb in v}
        else:
            d = {sb: s[sb] - v[sb] for sb in s if sb in v}
        _arc_choropleth(data_dir, by_arc(d), f"SAC-SMA − VIC basin-level {spec[m][0]} vs CalSim3 anchor",
                        cb, fig / f"calsim_sacsma_minus_vic_{m}.png",
                        cmap="RdBu", vmin=-lim, vmax=lim, annot=annot(d, vfmt))
    sac_med = met[met["which"] == "sac"]["kge"].median()
    vic_med = met[met["which"] == "vic"]["kge"].median()
    print(f"basin maps: {met['basin'].nunique()} basins ({'+'.join(sets)} partition); "
          f"median basin KGE [sac={sac_med:.2f}  vic={vic_med:.2f}] -> {out}")
    return met


def vic_full_metrics(data_dir: str | Path = "data") -> pd.DataFrame:
    """VIC vs CalSim3 skill for **every** CalSim3 arc VIC has a series for (not just the
    arcs a SAC-SMA set covers) — so the VIC map spans the full CalSim3 footprint
    (San Luis ``I_SLUIS`` and the other valley/westside nodes included).

    Returns ``[arc, node, source='vic', n_months, kge, nse, pbias, r]`` over each arc's
    common period with CalSim3.
    """
    c3 = load_calsim3_monthly(data_dir)
    vic = load_vic_monthly(data_dir)
    vavail = set(vic["vic_name"])
    rows = []
    for arc, cs in c3.groupby("arc"):
        vname = vic_node_name(arc)   # per-node VIC series
        if vname not in vavail:
            continue
        vv = vic[vic["vic_name"] == vname]
        w = pd.merge(cs[["date", "flow_taf"]].rename(columns={"flow_taf": "calsim3"}),
                     vv[["date", "flow_taf"]].rename(columns={"flow_taf": "vic"}),
                     on="date").dropna()
        if len(w) < 12:
            continue
        sim, ref = w["vic"].to_numpy(), w["calsim3"].to_numpy()
        rows.append({"arc": str(arc), "node": str(arc)[2:], "source": "vic", "n_months": len(w),
                     "kge": kge(sim, ref), "nse": nse(sim, ref), "pbias": pbias(sim, ref),
                     "r": pearson(sim, ref)})
    return pd.DataFrame(rows)


def calset_metrics(long: pd.DataFrame, matched: list[str], candidates) -> pd.DataFrame:
    """Per (arc, candidate) skill vs CalSim3 over the node's common period.

    Every candidate present at a node (the SAC sets + VIC) is scored on the **same months**
    — the joint intersection of all those candidates and CalSim3 — so the per-node
    SAC-vs-VIC head-to-head is on an identical period, not each candidate's own overlap.
    """
    rows = []
    for arc in matched:
        sub = long[long["arc"] == arc]
        wide = sub.pivot_table(index="date", columns="source", values="flow_taf")
        if "calsim3" not in wide:
            continue
        node = sub["node"].iloc[0]
        present = [c for c in candidates if c in wide]
        common = wide[present + ["calsim3"]].dropna()   # one period shared by all candidates
        if len(common) < 12:
            continue
        ref = common["calsim3"].to_numpy()
        for cand in present:
            sim = common[cand].to_numpy()
            rows.append({
                "arc": arc, "node": node, "set": cand, "n_months": len(common),
                "kge": kge(sim, ref), "nse": nse(sim, ref), "pbias": pbias(sim, ref),
                "r": pearson(sim, ref), "mean_set_taf": float(sim.mean()),
                "mean_calsim3_taf": float(ref.mean()),
            })
    return pd.DataFrame(rows)


#: rolling basin-level products: each anchor set's SAC run, plus VIC scored on that SAME
#: set's basins (label, set, sim_source) — so VIC appears as two lines, one per anchor set.
_ROLLING_PRODUCTS = (
    ("11obs", "11obs", "11obs"),
    ("9unimp", "9unimp", "9unimp"),
    ("vic_11obs", "11obs", "vic"),
    ("vic_9unimp", "9unimp", "vic"),
)
#: (colour-key, linestyle, label) per rolling product — SAC sets keep their palette colour;
#: VIC keeps its colour and is distinguished into two lines by linestyle (set footprint).
#: The colour key is resolved against :data:`_COLORS` at plot time (defined later in the file).
_ROLLING_STYLE = {
    "11obs": ("11obs", "-", "SAC-SMA 11obs"),
    "9unimp": ("9unimp", "-", "SAC-SMA 9unimp"),
    "vic_11obs": ("vic", "-", "VIC (11obs basins)"),
    "vic_9unimp": ("vic", "--", "VIC (9unimp basins)"),
}


#: the four rolling metrics (key, axis label, fallback ylim as (lo, hi) with None = autoscale,
#: draw-zero-line) — shared by the median panels and the per-watershed decomposition figures.
#: KGE/NSE axes are always the full 0–1 scale (house rule); pbias/seasonal-mismatch bounds are
#: data-driven but shared across the anchor sets so the per-set figures compare directly.
_ROLLING_METRICS = (
    ("kge", "rolling KGE", (None, 1.02), False),
    ("nse", "rolling NSE", (None, 1.02), False),
    ("pbias", "rolling pbias (%)", None, True),
    ("seas_mismatch", "rolling seasonal mismatch", (0.0, None), False),
)


def _apply_ylim(ax, ylim):
    """Apply a possibly one-sided ylim: ``None`` = autoscale; ``(lo, hi)`` with either bound
    ``None`` clamps only that side."""
    if ylim is None:
        return
    lo, hi = ylim
    if lo is not None and hi is not None:
        ax.set_ylim(lo, hi)
    elif lo is not None:
        ax.set_ylim(bottom=lo)
    elif hi is not None:
        ax.set_ylim(top=hi)


def _padded_bounds(values, *, pad_frac: float = 0.05, fallback=None):
    """(lo, hi) y-bounds that extend just past the finite ``values`` by ``pad_frac`` of their
    range; returns ``fallback`` when there is nothing finite to bound."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return fallback
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * pad_frac or abs(hi) * pad_frac or 0.01
    return (lo - pad, hi + pad)


def rolling_anchor_basin_table(anchor_long: pd.DataFrame, *, sets=("11obs", "9unimp"),
                               window_years: int = 30, step_months: int = 1,
                               min_frac: float = 0.9) -> pd.DataFrame:
    """Per-(set, basin, source) rolling KGE / NSE / pbias / seasonal-mismatch vs CalSim3 at the
    basin anchor.  ``source`` is ``"sac"`` (the set's own SAC-SMA run) or ``"vic"``; each basin
    is scored over each ``window_years``-long window (stepped ``step_months``) against its
    CalSim3 anchor reference (FLOW-UNIMPAIRED for rim systems, else the basin's INFLOW-sub-arc
    sum), requiring ≥ ``min_frac`` valid months.  Returns ``[center, start, end, set, basin,
    source, metric, value]`` — the un-aggregated rolling series behind the median plot."""
    a = anchor_long.copy()
    a["date"] = pd.to_datetime(a["date"]).dt.to_period("M").dt.to_timestamp("M")
    grid = (pd.period_range(a["date"].min().to_period("M"), a["date"].max().to_period("M"),
                            freq="M").to_timestamp("M"))
    W = int(window_years * 12)
    if len(grid) < W:
        raise ValueError(f"rolling_anchor_basin_table: only {len(grid)} months < {W}-month window")
    min_count = int(round(min_frac * W))

    # per (set, basin, source): the sim series + the CalSim3 anchor ref reindexed onto the grid
    pre: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for st in sets:
        sub = a[a["set"] == st]
        for basin, g in sub.groupby("basin"):
            w = g.pivot_table(index="date", columns="source", values="flow_taf")
            if "calsim3" not in w:
                continue
            ref = w["calsim3"].reindex(grid).to_numpy(float)
            for col, src in ((st, "sac"), ("vic", "vic")):
                if col in w:
                    pre[(st, str(basin), src)] = (w[col].reindex(grid).to_numpy(float), ref)

    rows = []
    for i in range(0, len(grid) - W + 1, step_months):
        sl = slice(i, i + W)
        center, start, end = grid[i + W // 2], grid[i], grid[i + W - 1]
        wd = grid[sl]
        for (st, basin, src), (sim, ref) in pre.items():
            s, r = sim[sl], ref[sl]
            m = np.isfinite(s) & np.isfinite(r)
            if int(m.sum()) < min_count:
                continue
            sm, rm, dm = s[m], r[m], wd[m]
            for metric, val in (("kge", kge(sm, rm)), ("nse", nse(sm, rm)),
                                ("pbias", pbias(sm, rm)),
                                ("seas_mismatch", seasonal_mismatch(dm, sm, rm))):
                rows.append({"center": center, "start": start, "end": end, "set": st,
                             "basin": basin, "source": src, "metric": metric,
                             "value": float(val)})
    return pd.DataFrame(rows)


def _basin_to_product_medians(basin: pd.DataFrame, products=_ROLLING_PRODUCTS) -> pd.DataFrame:
    """Reduce the per-basin rolling table to the **median across each product's basins**,
    relabelled to the product names (11obs / 9unimp / vic_11obs / vic_9unimp)."""
    cols = ["center", "start", "end", "source", "metric", "value", "n_basins"]
    if basin.empty:
        return pd.DataFrame(columns=cols)
    lab = {(st, "sac" if sim == st else "vic"): label for label, st, sim in products}
    b = basin.assign(label=[lab.get((s, src)) for s, src in zip(basin["set"], basin["source"])])
    b = b[b["label"].notna()]
    agg = (b.groupby(["center", "start", "end", "label", "metric"], observed=True)["value"]
           .agg(value="median", n_basins="count").reset_index()
           .rename(columns={"label": "source"}))
    return agg[cols]


def rolling_anchor_skill_table(anchor_long: pd.DataFrame, *, window_years: int = 30,
                               step_months: int = 1, products=_ROLLING_PRODUCTS,
                               min_frac: float = 0.9) -> pd.DataFrame:
    """Rolling-window KGE / NSE / pbias / seasonal-mismatch vs CalSim3 at the basin-level anchor,
    reduced by **median across each product's basins**.  ``products`` is ``(label, set,
    sim_source)`` — each anchor set's own SAC run plus VIC scored on that same set's basins (VIC
    split into a 11obs-basins and a 9unimp-basins line).  Returns ``[center, start, end, source,
    metric, value, n_basins]`` (``source`` = product label).  Built on
    :func:`rolling_anchor_basin_table`."""
    sets = tuple(dict.fromkeys(p[1] for p in products))
    basin = rolling_anchor_basin_table(anchor_long, sets=sets, window_years=window_years,
                                       step_months=step_months, min_frac=min_frac)
    return _basin_to_product_medians(basin, products)


def _rolling_skill_fig(tbl, products, window_years, path):
    """Four stacked panels (KGE / NSE / pbias / seasonal mismatch), one line per product
    (x = window centre).  SAC sets keep their palette colour; VIC is split into two lines by
    linestyle.  ≤6.5in wide, 300 dpi, 8-pt text, compact legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [p[0] for p in products if (tbl["source"] == p[0]).any()]
    # house rule: skill (KGE/NSE) axes are always the full 0–1 scale.
    ylims = {"kge": (0.0, 1.0), "nse": (0.0, 1.0), "pbias": None, "seas_mismatch": (0.0, None)}
    fig, axes = plt.subplots(len(_ROLLING_METRICS), 1, figsize=(_MAP_W, 9.5), sharex=True)
    for ax, (m, ylab, _pbylim, zero) in zip(axes, _ROLLING_METRICS):
        for label in order:
            d = tbl[(tbl["source"] == label) & (tbl["metric"] == m)].sort_values("center")
            if not len(d):
                continue
            ckey, ls, leg = _ROLLING_STYLE.get(label, (label, "-", label))
            ax.plot(d["center"], d["value"], color=_COLORS.get(ckey, "k"), ls=ls, lw=1.5, label=leg)
        if zero:
            ax.axhline(0, color="0.6", lw=0.8)
        _apply_ylim(ax, ylims[m])
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(color="0.93", lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    axes[0].legend(loc="lower center", ncol=2, fontsize=6)
    axes[-1].set_xlabel(f"{window_years}-year window centre", fontsize=8)
    fig.suptitle(_wrap(f"Rolling {window_years}-yr skill vs CalSim3 FLOW-UNIMPAIRED anchors "
                       f"— median across each set's basins (VIC split onto the 11obs & 9unimp "
                       f"basins)", 84), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=_MAP_DPI); plt.close(fig)


def _rolling_basin_fig(basin_tbl, st, metric, mlabel, ylim, window_years, path):
    """Per (set, metric) rolling figure: SAC-SMA (top panel) and VIC (bottom panel), **one line
    per watershed**, shared y so the two panels are directly comparable.  KGE/NSE use the fixed
    0–1 skill scale (house rule; values below 0 are clipped); pbias/seasonal-mismatch bounds are
    computed over ALL sets so the per-set figures share a y-scale.  ≤6.5in wide, 300 dpi, 8-pt
    text, compact legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = basin_tbl[(basin_tbl["set"] == st) & (basin_tbl["metric"] == metric)]
    if d.empty:
        return
    if metric in ("kge", "nse"):
        ylim = (0.0, 1.02)
    else:
        ylim = _padded_bounds(basin_tbl[basin_tbl["metric"] == metric]["value"], fallback=ylim)
    basins = sorted(d["basin"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {b: cmap(i % 20) for i, b in enumerate(basins)}
    fig, axes = plt.subplots(2, 1, figsize=(_MAP_W, 6.5), sharex=True, sharey=True)
    for ax, src, title in ((axes[0], "sac", f"SAC-SMA {st}"),
                           (axes[1], "vic", f"VIC ({st} basins)")):
        for b in basins:
            g = d[(d["source"] == src) & (d["basin"] == b)].sort_values("center")
            if len(g):
                ax.plot(g["center"], g["value"], color=colors[b], lw=1.0)
        if metric == "pbias":
            ax.axhline(0, color="0.6", lw=0.8)
        _apply_ylim(ax, ylim)
        ax.set_ylabel(mlabel, fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.grid(color="0.93", lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    axes[-1].set_xlabel(f"{window_years}-year window centre", fontsize=8)
    handles = [Line2D([0], [0], color=colors[b], lw=1.4) for b in basins]
    fig.legend(handles, basins, loc="lower center", ncol=min(len(basins), 6), fontsize=6,
               frameon=False)
    fig.suptitle(_wrap(f"Rolling {window_years}-yr {mlabel.removeprefix('rolling ')} vs CalSim3 "
                       f"anchor by watershed — {st} (SAC-SMA top, VIC bottom)", 84), fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=_MAP_DPI); plt.close(fig)


def make_rolling_skill(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts",
                       run: str = "compare", *, anchor_long=None, sets=DEFAULT_CALSETS,
                       window_years: int = 30, step_months: int = 1) -> Path:
    """Rolling basin-level skill vs CalSim3 (FLOW-UNIMPAIRED anchors) for the SAC anchor sets +
    VIC (split onto each set's basins).  Writes the median table ``rolling_skill_<W>yr.csv`` +
    4-panel ``figures/rolling_skill_<W>yr.png`` (KGE/NSE/pbias/seasonal mismatch), the
    per-watershed table ``rolling_skill_basin_<W>yr.csv``, and per-(set, metric)
    SAC-top/VIC-bottom watershed figures ``figures/rolling_basin_<set>_<metric>.png``.
    ``anchor_long`` is the basin-level frame; if ``None`` it is read from ``anchor_monthly.csv``."""
    out = Path(artifacts_dir) / "calsim" / run
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if anchor_long is None:
        csv = out / "anchor_monthly.csv"
        anchor_long = (read_table(csv) if csv.exists()
                       else build_anchor_long(data_dir, sets,
                                              footprint=_screened_fp(data_dir, sets)))
    asets = tuple(s for s in dict.fromkeys(p[1] for p in _ROLLING_PRODUCTS)
                  if s in set(anchor_long["set"].astype(str)))
    basin = rolling_anchor_basin_table(anchor_long, sets=asets, window_years=window_years,
                                       step_months=step_months)
    basin.to_csv(out / f"rolling_skill_basin_{window_years}yr.csv", index=False)
    tbl = _basin_to_product_medians(basin, _ROLLING_PRODUCTS)
    tbl.to_csv(out / f"rolling_skill_{window_years}yr.csv", index=False)
    _rolling_skill_fig(tbl, _ROLLING_PRODUCTS, window_years,
                       out / "figures" / f"rolling_skill_{window_years}yr.png")
    # per-(set, metric) decomposition: SAC top / VIC bottom, one line per watershed.  Each figure
    # auto-bounds y just past its own set's min/max (over both panels); bounds are NOT shared
    # across sets.  The static _ROLLING_METRICS ylim is only a fallback for an all-NaN metric.
    for st in asets:
        for metric, mlabel, ylim, _zero in _ROLLING_METRICS:
            _rolling_basin_fig(basin, st, metric, mlabel, ylim, window_years,
                               out / "figures" / f"rolling_basin_{st}_{metric}.png")
    if tbl.empty:
        return out
    span = f"{tbl['center'].min().date()}..{tbl['center'].max().date()}"
    labels = [p[0] for p in _ROLLING_PRODUCTS if (tbl["source"] == p[0]).any()]
    msg = "  ".join(
        f"{lab}={tbl[(tbl['source']==lab)&(tbl['metric']=='kge')]['value'].median():.2f}"
        for lab in labels)
    print(f"rolling: {window_years}-yr basin anchors, {tbl['center'].nunique()} centres "
          f"({span}); median-over-time KGE [{msg}] -> {out}")
    return out


def _shared_period(*longs):
    """The single ``[start, end]`` common to every ``source`` across all given long frames.

    Used to score the anchor and per-catchment views over identical months even though their
    references (FLOW-UNIMPAIRED vs CalSim3 INFLOW) and VIC forms have different native spans.
    """
    starts, ends = [], []
    for L in longs:
        if not len(L):
            continue
        b = L.groupby("source")["date"].agg(["min", "max"])
        starts.append(b["min"].max())
        ends.append(b["max"].min())
    return max(starts), min(ends)


def make_all(
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
    run: str = "compare",
    sets=DEFAULT_CALSETS,
    covered_frac=None,
    mass_balance=False,
    parallel=False,
) -> Path:
    """Cross-compare each calibration set + VIC vs CalSim3; best-of set per node.

    ``mass_balance`` (default off) applies CalSim's proportional sub-arc (anchor
    mass-balance) adjustment to every estimate's per-catchment series before scoring
    (:func:`_apply_anchor_mass_balance`).  It is off by default because it does NOT improve
    per-catchment skill — our per-catchment error is the spatial split among a system's
    sub-arcs, which a single per-system rescale cannot correct.

    ``parallel`` (default off) fans the SAC-SMA model runs (the basin anchors via
    ``run_basin`` and the per-catchment local runoff via ``run_calsim``) across cores
    with the Numba ``prange`` kernels — the model results are unchanged (bit-exact for
    the per-catchment build, floating-tolerance for the routed anchors)."""
    out = Path(artifacts_dir) / "calsim" / run
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # Shared per-cell SMA-component cache: the anchor (run_basin, routed) and per-catchment
    # (run_calsim, local) builds need the same PET->Snow-17->SAC-SMA physics per HRU cell, so
    # compute it once and reuse — keyed by (domain, cell, params) for exact parity.
    comp_cache: dict = {}
    # the basin anchors drive BOTH the sub-arc QMAP mass-balance target and the basin-level
    # anchor view (built once).  The OFFICIAL anchor basis is the corrected (GIS-screened)
    # footprint: each CalLite anchor basin simulated only on the HRUs inside its true CalSim
    # catchment (screened_footprint), overlap-area weighted — consistent with the per-catchment
    # sub-arcs and VIC's no_gooselake substitution.  15cdec (off the 1/16-deg grid, not an
    # ANCHOR_SET) keeps its full footprint.  The CalSim catchment area
    # (basin_area_<set>_calsim.csv, consistent with the sub-arcs + reference) carries the
    # volume reconciliation, leaving honest depth biases (e.g. BND +4.8%, Fresno +29%) visible.
    anchor_long = build_anchor_long(data_dir, sets, comp_cache=comp_cache, parallel=parallel,
                                    footprint=_screened_fp(data_dir, sets))
    long, matched, coverage = build_calsets_long(data_dir, sets, covered_frac=covered_frac,
                                                 anchor_long=anchor_long, mass_balance=mass_balance,
                                                 comp_cache=comp_cache, parallel=parallel)
    # Clip BOTH views to one shared scoring period — the intersection of every source feeding
    # either the anchor (FLOW-UNIMPAIRED / 8RI-VIC) or the per-catchment (INFLOW / per-node VIC)
    # view — so they score identical months.  The references' native spans differ (FLOW-UNIMPAIRED
    # starts a year before CalSim3 INFLOW), so the anchor's rim basins lose those leading months.
    start, end = _shared_period(anchor_long, long)
    anchor_long = anchor_long[anchor_long["date"].between(start, end)].reset_index(drop=True)
    long = long[long["date"].between(start, end)].reset_index(drop=True)
    print(f"compare: shared scoring period {start.date()} .. {end.date()}")
    candidates = list(sets) + ["vic"]
    # The per-catchment series already sit on the CalSim catchment area (run_calsim uses each
    # catchment's SQ_MI).  With the area nudge retired, the coverage maps score these raw series
    # directly — no per-(set,basin) rescale — so they are consistent with the basin anchor (now
    # also on the summed CalSim area) and the sub-arc QMAP.
    met = calset_metrics(long, matched, candidates)
    # attach each set's honest HRU coverage of the node, so a low-coverage (extrapolated)
    # per-node score is visible in the metrics CSV (NaN for vic — not HRU-based).
    cov_key = coverage.drop_duplicates(["set", "arc"]).set_index(["set", "arc"])
    met = met.merge(cov_key[["cov_frac", "n_hru"]], left_on=["set", "arc"],
                    right_index=True, how="left")

    vic_full = vic_full_metrics(data_dir)
    long.to_csv(out / "monthly_calsets.csv", index=False)
    met.to_csv(out / "calset_metrics.csv", index=False)
    vic_full.to_csv(out / "vic_full_metrics.csv", index=False)
    coverage.sort_values(["set", "basin", "cov_frac"], ascending=[True, True, False]).to_csv(
        out / "coverage_by_set.csv", index=False)

    period = f"{long['date'].min().date()}..{long['date'].max().date()}"
    msg = "  ".join(f"{s}={met[met['set']==s]['kge'].median():.2f}"
                    for s in candidates if (met["set"] == s).any())
    print(f"compare: {len(matched)} nodes vs CalSim3 over {period}; median KGE [{msg}] -> {out}")

    # basin-level anchor skill for EVERY set (15cdec included) — this is what all the maps
    # and figures show: each sub-area polygon carries its MAIN BASIN's score, never its own
    # per-sub-arc score (per-sub-arc numbers stay available in calset_metrics.csv).
    anchor_met = anchor_metrics(anchor_long)
    # per-set basin-coloured NSE map (with watershed outlines) + per-set basin skill fig
    for s in sets:
        _calset_coverage_map(data_dir, s, anchor_met, out / "figures" / f"{s}_coverage_map.png")
        _calset_skill_fig(anchor_met, s, out / "figures" / f"{s}_skill.png")
    # basin-level (anchor) comparison: each basin vs the sum of its sub-nodes (15cdec is
    # folded onto the same dumbbells inside make_anchor, behind a dashed divider)
    make_anchor(data_dir, artifacts_dir, run, sets, anchor_long=anchor_long)
    # parallel full-footprint view + the screened-vs-full delta (with the VIC benchmark), plus
    # the calibration-target-vs-CalSim3 table; the fnf_* calibration basis is untouched
    # (see tmp/CALSIM3_FNF_FOOTPRINT.md).
    make_anchor_full(data_dir, artifacts_dir, run, sets, anchor_long=anchor_long,
                     period=(start, end), comp_cache=comp_cache, parallel=parallel)
    target_vs_calsim3(data_dir, sets=tuple(s for s in sets if s in ANCHOR_SETS)).to_csv(
        out / "target_vs_calsim3.csv", index=False)
    # per-sub-arc QMAP bias-correction validation (train/test) + SAC-sim mass balance + VIC.
    # ONLY the adjustment sets (9unimp + 11obs) are sub-arc-adjusted — 15cdec is scored raw
    # (per-catchment) and never enters the adjustment/composite (see ADJUST_SETS).
    adj_sets = tuple(s for s in ADJUST_SETS if s in sets)
    subarc_met, subarc_series = _subarc_validate(
        data_dir, adj_sets, anchor_long=anchor_long,
        raw_long=(None if mass_balance else long))
    make_subarc_validation(data_dir, artifacts_dir, run, adj_sets, met=subarc_met,
                           series=subarc_series)

    # CalSim<->SAC-SMA basin maps (9unimp + 11obs partition): basin-level NSE/KGE/pbias for
    # the SAC composite and VIC, plus SAC−VIC difference maps — all coloured per main basin.
    make_basin_maps(data_dir, out, anchor_met, sets=adj_sets)
    # footprint-screening methods maps (single-basin illustrations of
    # tmp/CALSIM3_FNF_FOOTPRINT.md: the VIC grid + the SAC HRU sets on the catchment)
    make_shasta_footprint_maps(data_dir, out)
    for title, set_name, basin, vic_node, cdec_basin, stem in FOOTPRINT_MAP_BASINS:
        make_basin_footprint_maps(data_dir, out, title=title, set_name=set_name,
                                  basin=basin, vic_node=vic_node,
                                  cdec_basin=cdec_basin, stem=stem)
    # whole-domain HRU attribute / calibrated-parameter maps (15cdec veg_class + Kpet;
    # 11obs/9unimp per-basin Kpet) + the per-veg-class Kpet summary and the exact
    # soil_class→Kpet lookup (the real 15cdec Kpet regionalization is soil-based)
    make_hru_attribute_maps(data_dir, out)
    hru_param_table(data_dir).to_csv(out / "hru_veg_kpet_15cdec.csv", index=False)
    kpet_soil_table(data_dir).to_csv(out / "hru_kpet_by_soil_15cdec.csv", index=False)
    # rolling 30-yr KGE/NSE/pbias vs CalSim3 at the basin-level FLOW-UNIMPAIRED anchors
    # (median across each anchor set's basins; VIC split onto the 11obs & 9unimp basins)
    make_rolling_skill(data_dir, artifacts_dir, run, anchor_long=anchor_long, sets=sets)
    return out


#: ONE consistent colour palette + labels used across EVERY figure.
_COLORS = {"calsim3": "#111111", "15cdec": "#2c7fb8", "9unimp": "#41ab5d",
           "11obs": "#d95f0e", "12rim": "#756bb1", "vic": "#984ea3"}
_LABELS = {"calsim3": "CalSim3", "15cdec": "SAC-SMA 15cdec", "9unimp": "SAC-SMA 9unimp",
           "11obs": "SAC-SMA 11obs", "12rim": "SAC-SMA 12rim", "vic": "VIC"}
_SET_COLORS = _COLORS                       # back-compat alias
_MAP_W, _MAP_DPI = 6.5, 300                 # every figure <=6.5in wide; all figures at 300 dpi


def _wrap(s, width=58):
    """Wrap a long figure title onto multiple lines so it never runs off the page."""
    import textwrap

    return "\n".join(textwrap.wrap(str(s), width=width)) if s else s


def _map_extent(data_dir):
    """Fixed (xmin, xmax, ymin, ymax) so every map shares the same extent — the merged
    Rim layer's bounds with a small pad."""
    from .catchments import MERGED_LAYER, load_catchments

    b = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).total_bounds
    px, py = 0.15 * (b[2] - b[0]) / 10, 0.15 * (b[3] - b[1]) / 10
    return (b[0] - px, b[2] + px, b[1] - py, b[3] + py)


def _nse_choropleth(catch, covered_gdf, title, cb_label, path, *, cells=None,
                    cells_label=None, subsystems=None, extent=None,
                    cmap_name="plasma", vmin=0.0, vmax=1.0, annot=None):
    """Draw a CalSim Rim map: ``covered_gdf`` (has an ``nse`` column) as a 0..1 NSE
    choropleth on top of all catchments in faint grey; optionally overlay HRU cells
    and thick **subsystem outlines** (``subsystems`` = a GeoSeries of basin unions).

    The NSE colour scale is fixed **0..1** for every map (negative NSE clamps to the
    floor).  Catchments not scored are drawn once as faint "not covered" context.
    ``annot`` = optional ``[(lon, lat, text), ...]`` in-map labels (basin abbreviation +
    value), drawn with a white halo so they read over any fill colour.
    """
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.patches import Patch

    cmap = cmap_name if isinstance(cmap_name, Colormap) else plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(_MAP_W, 8.0))
    catch.plot(ax=ax, facecolor="#f3f3f3", edgecolor="0.85", linewidth=0.2)  # not covered
    if len(covered_gdf):
        covered_gdf.plot(ax=ax, column="nse", cmap=cmap, norm=norm,
                         edgecolor="0.55", linewidth=0.2, missing_kwds={"color": "#cccccc"})
    handles = [Patch(facecolor="#f3f3f3", edgecolor="0.85", label="not covered")]
    if subsystems:
        # one distinct colour per basin sub-system (HRU footprint = the watershed extent)
        bcmap = plt.get_cmap("tab20")
        for i, (basin, geom) in enumerate(sorted(subsystems.items())):
            col = bcmap(i % 20)
            gpd.GeoSeries([geom], crs=catch.crs).boundary.plot(ax=ax, color=col, linewidth=1.4, zorder=4)
            handles.append(plt.Line2D([], [], color=col, lw=1.4, label=str(basin)))
    if cells is not None:
        ax.scatter(cells["lon"], cells["lat"], s=1.0, c="#222222", alpha=0.4, zorder=3)
        handles.append(plt.Line2D([], [], marker="o", ls="", color="#222222", markersize=3,
                                  label=cells_label or f"HRU cells ({len(cells)})"))
    if annot:
        for ax_x, ax_y, txt in annot:
            ax.annotate(txt, (ax_x, ax_y), ha="center", va="center", fontsize=5,
                        color="0.1", zorder=6, linespacing=1.1,
                        path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])

    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(cb_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.legend(handles=handles, loc="lower left", fontsize=6, framealpha=0.9, ncol=2,
              title=f"scored: {len(covered_gdf)}", title_fontsize=6)
    ax.set_title(_wrap(title), fontsize=8)
    ax.set_xlabel("lon", fontsize=8); ax.set_ylabel("lat", fontsize=8)
    ax.tick_params(labelsize=7); ax.set_aspect(1.25)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    fig.tight_layout()
    fig.savefig(path, dpi=_MAP_DPI)
    plt.close(fig)


def _calset_coverage_map(data_dir, set_name, anchor_met, path):
    """Per-set map: the set's CalSim3 catchments (the **merged** whole-basin layer) shaded by
    their MAIN BASIN's anchor NSE vs the basin's faithful CalSim3 reference
    (:func:`anchor_metrics`) — **all sub-areas of a watershed take the same colour** — on
    faint context, with a clean coloured outline of each basin's **HRU footprint** (the
    watershed it represents) + HRU cells.  The merged layer fills the cumulative single-node
    basins (Merced/Shasta/SJ) as whole catchments rather than leaving grey holes."""
    from .catchments import (
        MERGED_LAYER,
        basin_footprints,
        load_catchments,
        load_hru_cells,
        series_arc,
    )

    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).copy()
    catch["arc"] = catch["node"].map(series_arc)
    a2b = arc_basin_map(data_dir, set_name)
    nse_by_basin = (anchor_met[(anchor_met["set"] == set_name)
                               & (anchor_met["source"] == set_name)]
                    .dropna(subset=["nse"]).set_index("basin")["nse"].to_dict())
    scored = catch[catch["arc"].isin(a2b)].copy()
    scored["basin"] = scored["arc"].map(a2b)
    scored["nse"] = scored["basin"].map(nse_by_basin)
    scored = scored[scored["nse"].notna()]
    cells = load_hru_cells(data_dir, domain=set_name)
    footprints = basin_footprints(data_dir, set_name)
    pts = _basin_label_points(scored)
    annot = [(*pts[b], f"{_BASIN_ABBREV.get(b, b)}\n{v:.2f}")
             for b, v in nse_by_basin.items() if b in pts and np.isfinite(v)]
    _nse_choropleth(catch, scored,
                    f"{set_name}: basin-level NSE vs CalSim3 anchor (all sub-areas coloured by "
                    f"their watershed) + basin outlines ({set_name} HRUs)",
                    "basin monthly NSE vs CalSim3 anchor", path,
                    cells=cells, cells_label=f"{set_name} HRU cells ({len(cells)})",
                    subsystems=footprints, extent=_map_extent(data_dir), annot=annot)


# ---------------------------------------------------------------------------
# single-basin footprint-screening maps (methods figures)
# ---------------------------------------------------------------------------

#: red for terrain outside the CalSim3 catchment / dropped by the screening
_FP_RED = "#c02f1d"


def _cell_boxes(lat, lon):
    """1/16-degree grid-cell polygons around cell-center points (EPSG:4326)."""
    import geopandas as gpd
    from shapely import box

    h = 0.0625 / 2.0
    return gpd.GeoSeries([box(x - h, y - h, x + h, y + h)
                          for y, x in zip(lat, lon, strict=True)], crs="EPSG:4326")


def _vic_cells(data_dir, node, *, no_gooselake=False):
    """VIC GridInfo cells for ``node`` as boxes with the in-basin ``frac`` weight.

    A couple of boundary cells arrive as two fragments (sliver + main part) ->
    aggregate per (lat, lon) cell first.
    """
    import geopandas as gpd

    from . import load_vic_gridinfo

    g = (load_vic_gridinfo(data_dir, node=node, no_gooselake=no_gooselake)
         .groupby(["lat", "lon"], as_index=False)
         .agg(cell_km2=("cell_km2", "first"), basin_km2=("basin_km2", "sum")))
    g["frac"] = (g["basin_km2"] / g["cell_km2"]).clip(upper=1.0)
    return gpd.GeoDataFrame(g, geometry=_cell_boxes(g["lat"], g["lon"]), crs="EPSG:4326")


def _fill_cells(ax, geoms, color, frac=None, alpha=0.85, hatch=None, zorder=3):
    """Draw grid-cell polygons; ``frac`` scales each cell's fill alpha so partial
    (edge) cells read lighter."""
    import geopandas as gpd
    from matplotlib.colors import to_rgba

    if len(geoms) == 0:                     # e.g. a basin whose screening drops nothing
        return
    if frac is None:
        colors = to_rgba(color, alpha)
    else:
        colors = [to_rgba(color, 0.30 + 0.60 * float(f)) for f in frac]
    gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms, crs="EPSG:4326")).plot(
        ax=ax, color=colors, edgecolor=(color if hatch else "white"),
        linewidth=0.15, hatch=hatch, zorder=zorder)


def _fp_panel(ax, title, note, *, context, outline, extent, aspect):
    """One footprint-map panel: grey context catchments, the basin outline as the
    common reference, the stats note box, and the shared extent."""
    import geopandas as gpd

    context.plot(ax=ax, facecolor="#f3f3f3", edgecolor="0.85", linewidth=0.2, zorder=1)
    gpd.GeoSeries([outline], crs=context.crs).boundary.plot(
        ax=ax, color=_COLORS["calsim3"], linewidth=0.9, zorder=5)
    ax.set_title(title, fontsize=8)
    ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=6, va="bottom",
            ha="left", linespacing=1.3, zorder=6,
            bbox=dict(facecolor="white", edgecolor="0.7", lw=0.4, alpha=0.85,
                      boxstyle="round,pad=0.25"))
    xmin, ymin, xmax, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(aspect)
    ax.tick_params(labelsize=6)


def make_shasta_footprint_maps(data_dir: str | Path = "data",
                               out: str | Path = "artifacts/calsim/compare"):
    """The Shasta footprint-screening story as maps (single-basin methods figure).

    ``shasta_footprint_panels.png`` — 2x3 solo panels (same layout as
    :func:`make_basin_footprint_maps`): the VIC routing grid (original, with its
    endorheic Goose Lake over-reach red — the one place VIC's grid is ever
    corrected), the SAC-SMA 11obs HRUs (full, screened-out cells red), the 15cdec
    HRUs, the authoritative CalSim3 SHSTA delineation (Goose Lake dash-outlined),
    and the screened result the anchor scores: the GIS-screened 11obs HRUs drawn
    OVER the full VIC ``no_gooselake`` grid — VIC still simulates every one of those
    cells, so terrain only VIC covers stays visibly purple.  The sixth slot carries
    the legend.

    Every panel shares one extent and carries the black SHSTA outline as the common
    reference; red marks the Goose Lake over-reach (terrain outside the catchment /
    dropped by screening).  Unlike every other compare map this one is
    **single-basin** (SHA/SHSTA) — a methods illustration of
    ``tmp/CALSIM3_FNF_FOOTPRINT.md``, not a basin-level skill map.  The other
    single-basin counterparts render via :func:`make_basin_footprint_maps`
    (:data:`FOOTPRINT_MAP_BASINS`).
    """
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from shapely.ops import unary_union

    from ..io import load_basin_area, load_hru_table
    from . import load_vic_gridinfo
    from .catchments import (
        _EQ_CRS,
        _M2_PER_MI2,
        MERGED_LAYER,
        _hru_abs_area,
        load_catchments,
        screened_footprint,
    )

    figs = Path(out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    km2_per_mi2 = _M2_PER_MI2 / 1e6

    # --- CalSim3 delineation: the SHSTA rim polygon (+ every other rim as faint context)
    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True)
    shsta = catch[catch["node"] == "SHSTA"]
    shsta_geom = unary_union(list(shsta.geometry))
    calsim_mi2 = float(shsta["sq_mi"].sum())

    # --- VIC routing grid (GridInfo): per-cell in-basin fraction
    vic_all = _vic_cells(data_dir, "I_SHSTA")
    vic_kept = _vic_cells(data_dir, "I_SHSTA", no_gooselake=True)
    n_vic_all = len(load_vic_gridinfo(data_dir))            # row counts, incl. fragments
    n_vic_kept = len(load_vic_gridinfo(data_dir, no_gooselake=True))
    kept_keys = set(zip(vic_kept["lat"], vic_kept["lon"], strict=True))
    in_kept = [k in kept_keys for k in zip(vic_all["lat"], vic_all["lon"], strict=True)]
    vic_drop = vic_all[[not k for k in in_kept]]
    vic_orig_kept = vic_all[in_kept]
    vic_all_mi2 = float(vic_all["basin_km2"].sum() / km2_per_mi2)
    goose_geom = unary_union(list(vic_drop.geometry))       # the Goose Lake block

    # --- SAC-SMA 11obs SHA HRUs (1/16-deg cells) + the GIS screening that fixes them
    sha = (load_hru_table(data_dir, domain="11obs")
           .query("basin == 'SHA'").drop_duplicates("key").reset_index(drop=True))
    sha_boxes = _cell_boxes(sha["lat"], sha["lon"])
    scr = screened_footprint(data_dir, domain="11obs")
    scr = scr[scr["basin"] == "SHA"].set_index("key")["overlap_area_mi2"]
    kept11 = sha["key"].isin(scr.index).to_numpy()
    box_mi2 = sha_boxes.to_crs(_EQ_CRS).area.to_numpy() / _M2_PER_MI2
    ov_frac = np.clip(sha["key"].map(scr).to_numpy() / box_mi2, 0.0, 1.0)
    # 11obs quoted on its EFFECTIVE area basis — the reconstructed per-HRU drainage
    # areas its ``area_weight`` claims (same ``abs_area`` as :func:`load_hru_cells`),
    # the parallel of VIC's in-basin ``basin_km2`` — not the gross span of the drawn
    # boxes (edge HRUs are partial cells, so the box union would over-state it).
    abs_area = (_hru_abs_area(data_dir, "11obs").query("basin == 'SHA'")
                .drop_duplicates("key").set_index("key")["abs_area"])
    full11_mi2 = float(abs_area.sum())

    # --- SAC-SMA 15cdec SHA HRUs (irregular sub-grid points, off the 1/16-deg grid)
    cdec = (load_hru_table(data_dir, domain="15cdec")
            .query("basin == 'SHA'").drop_duplicates("key"))
    cdec_mi2 = float(load_basin_area(data_dir, domain="15cdec")
                     .set_index("basin").loc["SHA", "area_mi2"])

    # one shared extent: everything any panel draws, plus a small pad
    bounds = np.vstack([vic_all.total_bounds, np.asarray(shsta.total_bounds),
                        np.asarray(sha_boxes.total_bounds)])
    xmin, ymin = bounds[:, :2].min(axis=0) - 0.10
    xmax, ymax = bounds[:, 2:].max(axis=0) + 0.10

    def pct(a):
        return f"{100.0 * (a / calsim_mi2 - 1.0):+.1f}%"

    def panel(ax, title, note):
        _fp_panel(ax, title, note, context=catch, outline=shsta_geom,
                  extent=(xmin, ymin, xmax, ymax), aspect=1.30)

    fig, axes = plt.subplots(2, 3, figsize=(_MAP_W, 5.2), sharex=True, sharey=True)
    ax = axes.ravel()
    panel(ax[0], "(a) VIC routing grid — original",
          f"{n_vic_all} cells\n{vic_all_mi2:,.0f} mi$^2$ ({pct(vic_all_mi2)})")
    _fill_cells(ax[0], vic_orig_kept.geometry, _COLORS["vic"], frac=vic_orig_kept["frac"])
    _fill_cells(ax[0], vic_drop.geometry, _FP_RED, frac=vic_drop["frac"])
    panel(ax[1], "(b) SAC-SMA 11obs HRUs (full)",
          f"{len(sha)} HRUs\neffective {full11_mi2:,.0f} mi$^2$ ({pct(full11_mi2)})")
    _fill_cells(ax[1], sha_boxes[kept11], _COLORS["11obs"])
    _fill_cells(ax[1], sha_boxes[~kept11], _FP_RED)
    panel(ax[2], "(c) SAC-SMA 15cdec HRUs (SHA)",
          f"{len(cdec)} HRUs (sub-grid points)\n{cdec_mi2:,.0f} mi$^2$ ({pct(cdec_mi2)})")
    ax[2].scatter(cdec["lon"], cdec["lat"], s=0.8, c=_COLORS["15cdec"], alpha=0.7,
                  linewidths=0, zorder=3)
    panel(ax[3], "(d) CalSim3 delineation (SHSTA)",
          f"{calsim_mi2:,.0f} mi$^2$\nthe anchor basis")
    gpd.GeoSeries([shsta_geom], crs=catch.crs).plot(ax=ax[3], facecolor="#e0dcd3",
                                                    edgecolor="none", zorder=2)
    gpd.GeoSeries([goose_geom], crs=catch.crs).boundary.plot(
        ax=ax[3], color=_FP_RED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=4)
    gx, gy = goose_geom.centroid.x, goose_geom.centroid.y
    ax[3].annotate("Goose Lake\n(endorheic)", (gx, gy), ha="center", va="center",
                   fontsize=5.5, color=_FP_RED, zorder=6, linespacing=1.2,
                   path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
    panel(ax[4], "(e) screened — both models",
          f"SAC-SMA: {int(kept11.sum())}/{len(sha)} HRUs kept\n"
          f"VIC: {n_vic_kept}/{n_vic_all} cells kept")
    # VIC still SIMULATES its whole no-Goose-Lake grid (the screening cut only the
    # SAC-SMA HRU set) — draw the VIC cells underneath the screened HRUs so terrain
    # only VIC covers stays visibly purple.
    _fill_cells(ax[4], vic_kept.geometry, _COLORS["vic"], frac=vic_kept["frac"], zorder=2)
    _fill_cells(ax[4], sha_boxes[kept11], _COLORS["11obs"], frac=ov_frac[kept11])
    handles = [
        Line2D([], [], color=_COLORS["calsim3"], lw=0.9,
               label=f"CalSim3 SHSTA catchment ({calsim_mi2:,.0f} mi$^2$)"),
        Patch(facecolor=to_rgba(_COLORS["vic"], 0.7), label="VIC grid cell"),
        Patch(facecolor=to_rgba(_COLORS["11obs"], 0.85), label="11obs HRU cell"),
        Line2D([], [], marker="o", ls="", color=_COLORS["15cdec"], markersize=3,
               label="15cdec (SHA) HRU"),
        Patch(facecolor=to_rgba(_FP_RED, 0.8), label="Goose Lake over-reach (dropped)"),
    ]
    ax[5].axis("off")
    ax[5].legend(handles=handles, loc="center", fontsize=6, framealpha=0.9)
    ax[2].tick_params(labelbottom=True)     # sharex hides labels above the legend slot
    fig.suptitle("Shasta: screening the VIC grid and SAC-SMA HRU footprints to the "
                 "CalSim3 SHSTA catchment", fontsize=8.5)
    fig.supxlabel("lon", fontsize=7)
    fig.supylabel("lat", fontsize=7)
    fig.tight_layout(rect=(0.01, 0.0, 1, 1))
    fig.savefig(figs / "shasta_footprint_panels.png", dpi=_MAP_DPI)
    plt.close(fig)
    print(f"compare: wrote shasta_footprint_panels.png -> {figs}")


#: single-basin footprint-map configs for :func:`make_basin_footprint_maps` —
#: ``(display title, calibration set, basin, VIC GridInfo node, 15cdec basin or None,
#: output stem)``.  Each figure renders the screened or the not-screened story by the
#: basin's :data:`~.catchments.SCREENED_BASINS` membership (SNS and Chowchilla are
#: screened; Trinity and the Fresno River are not).  Trinity and the Fresno/Chowchilla
#: rivers have no 15cdec representation.
FOOTPRINT_MAP_BASINS = (
    ("Stanislaus", "11obs", "SNS", "8RI_N_MEL", "NML", "sns"),
    ("Chowchilla River", "9unimp", "ChowchillaRiver", "I_ESTMN", None, "chowchilla"),
    ("Trinity", "11obs", "TNL", "I_TRNTY", None, "tnl"),
    ("Fresno River", "9unimp", "FresnoRiver", "I_HNSLY", None, "fresno"),
)


def make_basin_footprint_maps(data_dir: str | Path = "data",
                              out: str | Path = "artifacts/calsim/compare", *,
                              title: str = "Stanislaus", set_name: str = "11obs",
                              basin: str = "SNS", vic_node: str = "8RI_N_MEL",
                              cdec_basin: str | None = "NML", stem: str = "sns"):
    """One anchor basin's footprint story as maps (single-basin methods figure).

    ``<stem>_footprint_panels.png`` — 2x3 solo panels: the VIC routing grid
    (``vic_node``, always used **as-is**), the SAC-SMA ``set_name`` HRUs, the 15cdec
    HRUs where the basin has them (``cdec_basin``), the CalSim3 delineation the basin
    owns, and the two model footprints the anchor actually scores, drawn together.
    The story branches on the basin's :data:`~.catchments.SCREENED_BASINS` membership:

    * **screened** (SNS, Chowchilla; Shasta has its own richer function) — the
      out-of-catchment HRU cells are solid red (dropped) and the final panel draws
      the GIS-screened cells (overlap-weighted alpha) over the full VIC grid;
    * **not screened** (Trinity, Fresno R.) — the anchor volume is simply the
      full-footprint area-weighted depth times the CalSim3 catchment area, so cells
      extending outside the catchment are red-hatched but **kept**, and the final
      panel draws ALL the set's HRU cells over the full VIC grid.

    The slot after the panels carries the legend.  Shared extent, black catchment
    outline; single-basin methods figures — the basin-level colouring rule does not
    apply.  ``make_all`` renders :data:`FOOTPRINT_MAP_BASINS`.
    """
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    from ..io import load_basin_area, load_hru_table
    from . import load_vic_gridinfo
    from .catchments import (
        _EQ_CRS,
        _M2_PER_MI2,
        BASIN_RIM_SYSTEM,
        MERGED_LAYER,
        SCREENED_BASINS,
        VALLEY_SYSTEMS,
        _hru_abs_area,
        derive_basin_nodes,
        load_catchments,
        screened_footprint,
        valley_arc_for_system,
    )

    screened = basin in SCREENED_BASINS

    figs = Path(out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    km2_per_mi2 = _M2_PER_MI2 / 1e6

    # --- CalSim3 delineation: the catchments the basin owns (screened_footprint's
    # own recipe: crosswalk nodes + the valley-accretion node where the system has one)
    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True)
    nodes = derive_basin_nodes(data_dir, set_name)
    own = set(nodes.loc[nodes["basin"] == basin, "node"].astype(str))
    sysn = BASIN_RIM_SYSTEM.get(set_name, {}).get(basin)
    if sysn in VALLEY_SYSTEMS:
        own.add(valley_arc_for_system(sysn)[2:])
    bcatch = catch[catch["node"].astype(str).isin(own)]
    u = unary_union(list(bcatch.geometry))
    # adjacent catchments union with hairline sliver gaps -> drop the interior rings
    # (and any sliver fragments) so only the true outer boundary draws
    parts = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
    bgeom = MultiPolygon([Polygon(p.exterior) for p in parts if p.area > 1e-6])
    calsim_mi2 = float(bcatch["sq_mi"].sum())

    # --- VIC routing grid (GridInfo): per-cell in-basin fraction, used as-is
    vic = _vic_cells(data_dir, vic_node)
    n_vic = len(load_vic_gridinfo(data_dir, node=vic_node))
    vic_mi2 = float(vic["basin_km2"].sum() / km2_per_mi2)

    # --- the set's HRUs (1/16-deg cells) + the catchment overlap.  For a screened
    # basin this IS the anchor's screening; for the others it is geometry only
    # (basins=(basin,) forces the per-basin diagnostic — every HRU stays kept).
    hru = load_hru_table(data_dir, domain=set_name)
    hru = hru[hru["basin"] == basin].drop_duplicates("key").reset_index(drop=True)
    hru_boxes = _cell_boxes(hru["lat"], hru["lon"])
    scr = screened_footprint(data_dir, domain=set_name, basins=(basin,))
    scr = scr[scr["basin"] == basin].set_index("key")["overlap_area_mi2"]
    inside = hru["key"].isin(scr.index).to_numpy()
    box_mi2 = hru_boxes.to_crs(_EQ_CRS).area.to_numpy() / _M2_PER_MI2
    ov_frac = np.clip(hru["key"].map(scr).to_numpy() / box_mi2, 0.0, 1.0)
    # effective area basis, as in the Shasta figure (see there)
    abs_area = _hru_abs_area(data_dir, set_name)
    abs_area = (abs_area[abs_area["basin"] == basin]
                .drop_duplicates("key").set_index("key")["abs_area"])
    full_mi2 = float(abs_area.sum())

    # --- 15cdec HRUs (irregular sub-grid points), only where the basin has them
    cdec = None
    if cdec_basin is not None:
        cdec = load_hru_table(data_dir, domain="15cdec")
        cdec = cdec[cdec["basin"] == cdec_basin].drop_duplicates("key")
        cdec_mi2 = float(load_basin_area(data_dir, domain="15cdec")
                         .set_index("basin").loc[cdec_basin, "area_mi2"])

    bounds = np.vstack([vic.total_bounds, np.asarray(bcatch.total_bounds),
                        np.asarray(hru_boxes.total_bounds)])
    xmin, ymin = bounds[:, :2].min(axis=0) - 0.10
    xmax, ymax = bounds[:, 2:].max(axis=0) + 0.10
    aspect = 1.0 / np.cos(np.radians((ymin + ymax) / 2.0))

    def pct(a):
        return f"{100.0 * (a / calsim_mi2 - 1.0):+.1f}%"

    def panel(ax, ttl, note):
        _fp_panel(ax, ttl, note, context=catch, outline=bgeom,
                  extent=(xmin, ymin, xmax, ymax), aspect=aspect)

    def draw_vic(a):
        _fill_cells(a, vic.geometry, _COLORS["vic"], frac=vic["frac"])

    def draw_full(a):
        if screened:
            _fill_cells(a, hru_boxes[inside], _COLORS[set_name])
            _fill_cells(a, hru_boxes[~inside], _FP_RED)      # dropped by the screening
        else:
            _fill_cells(a, hru_boxes, _COLORS[set_name])
            # cells extending outside the catchment are KEPT (not screened) — hatch only
            _fill_cells(a, hru_boxes[~inside], _FP_RED, alpha=0.0, hatch="////", zorder=4)

    def draw_cdec(a):
        a.scatter(cdec["lon"], cdec["lat"], s=0.8, c=_COLORS["15cdec"], alpha=0.7,
                  linewidths=0, zorder=3)

    def draw_delin(a):
        gpd.GeoSeries([bgeom], crs=catch.crs).plot(ax=a, facecolor="#e0dcd3",
                                                   edgecolor="none", zorder=2)

    def draw_scored(a):
        # the footprints the anchor actually scores: the full VIC grid (never screened
        # here) under the SAC-SMA cells — screened subset or the full set
        _fill_cells(a, vic.geometry, _COLORS["vic"], frac=vic["frac"], zorder=2)
        if screened:
            _fill_cells(a, hru_boxes[inside], _COLORS[set_name], frac=ov_frac[inside])
        else:
            _fill_cells(a, hru_boxes, _COLORS[set_name], alpha=0.62)

    panels = [
        (f"VIC routing grid ({vic_node})",
         f"{n_vic} cells — used as-is\n{vic_mi2:,.0f} mi$^2$ ({pct(vic_mi2)})",
         draw_vic),
        (f"SAC-SMA {set_name} HRUs" + (" (full)" if screened else ""),
         f"{len(hru)} HRUs — {f'{int(inside.sum())} kept' if screened else 'all kept'}\n"
         f"effective {full_mi2:,.0f} mi$^2$ ({pct(full_mi2)})",
         draw_full),
    ]
    if cdec is not None:
        panels.append((f"SAC-SMA 15cdec HRUs ({cdec_basin})",
                       f"{len(cdec)} HRUs (sub-grid points)\n"
                       f"{cdec_mi2:,.0f} mi$^2$ ({pct(cdec_mi2)})", draw_cdec))
    ncatch = len(bcatch)
    blab = _BASIN_ABBREV.get(basin, basin)
    panels += [
        (f"CalSim3 delineation ({blab})",
         f"{calsim_mi2:,.0f} mi$^2$ in {ncatch} catchment{'s' if ncatch > 1 else ''}\n"
         "the anchor basis", draw_delin),
        ("screened — both models" if screened else "scored footprints — both models",
         (f"SAC-SMA: {int(inside.sum())}/{len(hru)} HRUs kept" if screened
          else f"SAC-SMA: all {len(hru)} HRUs (not screened)")
         + f"\nVIC: all {n_vic} cells",
         draw_scored),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(_MAP_W, 5.2), sharex=True, sharey=True)
    ax = axes.ravel()
    for k, (ttl, note, draw) in enumerate(panels):
        panel(ax[k], f"({'abcde'[k]}) {ttl}", note)
        draw(ax[k])
    handles = [
        Line2D([], [], color=_COLORS["calsim3"], lw=0.9,
               label=f"CalSim3 {blab} catchment ({calsim_mi2:,.0f} mi$^2$)"),
        Patch(facecolor=to_rgba(_COLORS["vic"], 0.7), label="VIC grid cell"),
        Patch(facecolor=to_rgba(_COLORS[set_name], 0.85), label=f"{set_name} HRU cell"),
    ]
    if cdec is not None:
        handles.append(Line2D([], [], marker="o", ls="", color=_COLORS["15cdec"],
                              markersize=3, label=f"15cdec ({cdec_basin}) HRU"))
    if (~inside).any():                     # e.g. Trinity has no out-of-catchment cells
        handles.append(
            Patch(facecolor=to_rgba(_FP_RED, 0.8), label="outside catchment (screened out)")
            if screened else
            Patch(facecolor="none", edgecolor=_FP_RED, hatch="////", lw=0.6,
                  label="extends outside catchment (kept)"))
    ax[len(panels)].axis("off")
    ax[len(panels)].legend(handles=handles, loc="center", fontsize=6, framealpha=0.9)
    for k in range(len(panels) + 1, 6):
        ax[k].axis("off")
    for col in range(3):                          # sharex hides labels above off axes
        vis = [k for k in (col, col + 3) if k < len(panels)]
        if vis and vis[-1] != col + 3:
            ax[vis[-1]].tick_params(labelbottom=True)
    ax[0].xaxis.set_major_locator(MaxNLocator(4))  # small extents: default ticks collide
    fig.suptitle(f"{title}: screening the SAC-SMA footprint to the CalSim3 "
                 "catchment (VIC grid as-is)" if screened else
                 f"{title}: the model footprints vs the CalSim3 catchment — "
                 "not screened; both grids scored as-is", fontsize=8.5)
    fig.supxlabel("lon", fontsize=7)
    fig.supylabel("lat", fontsize=7)
    fig.tight_layout(rect=(0.01, 0.0, 1, 1))
    fig.savefig(figs / f"{stem}_footprint_panels.png", dpi=_MAP_DPI)
    plt.close(fig)
    print(f"compare: wrote {stem}_footprint_panels.png -> {figs}")


# ---------------------------------------------------------------------------
# HRU attribute / calibrated-parameter maps (whole-domain input figures)
# ---------------------------------------------------------------------------

def _hru_extent(lat, lon, pad=0.15):
    """(xmin, xmax, ymin, ymax) around HRU points with a small pad — these domains
    (esp. 15cdec, which reaches the southern Sierra) exceed the Rim :func:`_map_extent`."""
    return (lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad)


def _hru_map_context(ax, data_dir, extent):
    """Faint grey CalSim3 Rim catchments as a geographic backdrop + shared limits and
    cos(lat) aspect — the common base for every HRU attribute map."""
    from .catchments import load_catchments

    load_catchments(data_dir, rim_only=True).plot(
        ax=ax, facecolor="#f4f4f4", edgecolor="0.82", linewidth=0.2, zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad((extent[2] + extent[3]) / 2.0)))
    ax.tick_params(labelsize=6)
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def make_hru_attribute_maps(data_dir: str | Path = "data",
                            out: str | Path = "artifacts/calsim/compare"):
    """Whole-domain HRU attribute / calibrated-parameter maps (input figures).

    Four PNGs into ``<out>/figures/``:

    * ``hru_veg_15cdec.png`` — the 15cdec HRUs coloured categorically by ``veg_class``
      (a per-cell land-cover code; verified consistent across the pooled table's repeated
      keys).  Codes are shown as-is: they follow the study's MODIS/IGBP-lineage land-cover
      scheme (Hansen et al. 2010) but the repo carries no authoritative code→name legend,
      so the legend stays numeric to avoid mislabelling.
    * ``hru_soil_15cdec.png`` — the same HRUs coloured categorically by ``soil_class``
      (also a numeric code, no legend in the repo).  This is the map the calibrated ``Kpet``
      actually tracks (see below).
    * ``hru_kpet_15cdec.png`` — the same HRUs coloured by the **calibrated** Hamon PET
      coefficient ``Kpet``.  In the 15cdec pooled optimum ``Kpet`` is regionalized on
      **soil zone alone** (9 distinct values, 0.4–2.5, single-valued per ``soil_class``;
      ``veg_class`` adds nothing — verified), so this mirrors ``hru_soil_15cdec.png``, not the
      vegetation map.  The exact lookup is :func:`kpet_soil_table`.
    * ``hru_kpet_calsim.png`` — the 11obs + 9unimp HRUs coloured by ``Kpet`` on a shared
      colourbar.  ``Kpet`` is **per-basin uniform** in these per-watershed calibrations, so
      the map reads as a per-basin choropleth at HRU-cell resolution.  Basins share boundary
      cells with *different* ``Kpet`` (662 such cells in 11obs), so every basin's full cell
      set is drawn — higher-``Kpet`` basins last, so a contested cell resolves deterministically.

    Whole-domain input maps, not the basin-level cross-compare — the basin-colouring rule
    does not apply.  Repo map style (≤6.5in, 300 dpi).  Rendered by ``make_all``.
    """
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patheffects import withStroke

    from ..io import load_hru_table, load_params
    from .catchments import basin_footprints

    figs = Path(out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    halo = [withStroke(linewidth=1.6, foreground="white")]

    # --- Figures 1-2: 15cdec categorical attribute maps (veg_class, soil_class) ---
    h = load_hru_table(data_dir, domain="15cdec")
    ext = _hru_extent(h["lat"].to_numpy(), h["lon"].to_numpy())
    foots15 = basin_footprints(data_dir, domain="15cdec")

    def _cat_map(field, title, fname):
        classes = sorted(h[field].unique())
        counts = h[field].value_counts()
        cm = plt.get_cmap("tab20")
        col = {c: cm(i % 20) for i, c in enumerate(classes)}
        fig, ax = plt.subplots(figsize=(5.0, 6.6))
        _hru_map_context(ax, data_dir, ext)
        gpd.GeoSeries(list(foots15.values()), crs="EPSG:4326").boundary.plot(
            ax=ax, color="0.35", linewidth=0.5, zorder=3)
        for c in classes:
            hc = h[h[field] == c]
            ax.scatter(hc["lon"], hc["lat"], s=3.0, c=[col[c]], marker="s",
                       linewidths=0, zorder=4)
        handles = [Line2D([0], [0], marker="s", linestyle="", markersize=6,
                          markerfacecolor=col[c], markeredgewidth=0,
                          label=f"class {c}  (n={counts[c]})") for c in classes]
        ax.legend(handles=handles, title=field, fontsize=6, title_fontsize=7,
                  loc="upper right", framealpha=0.9, borderpad=0.5, handletextpad=0.4,
                  labelspacing=0.3)
        ax.set_title(f"{title}  ({len(h)} HRUs, {h['basin'].nunique()} watersheds)",
                     fontsize=9)
        ax.set_xlabel("lon", fontsize=7)
        ax.set_ylabel("lat", fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / fname, dpi=_MAP_DPI, bbox_inches="tight")
        plt.close(fig)

    _cat_map("veg_class", "15cdec HRUs by vegetation class", "hru_veg_15cdec.png")
    _cat_map("soil_class", "15cdec HRUs by soil class", "hru_soil_15cdec.png")

    # --- Figure 3: 15cdec calibrated Kpet (tracks soil_class) ---
    kp15 = load_params(data_dir, domain="15cdec").set_index("key")["Kpet"]
    h = h.assign(Kpet=h["key"].map(kp15))
    norm = Normalize(vmin=np.floor(h["Kpet"].min() * 10) / 10,
                     vmax=np.ceil(h["Kpet"].max() * 10) / 10)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(5.0, 6.6))
    _hru_map_context(ax, data_dir, ext)
    gpd.GeoSeries(list(foots15.values()), crs="EPSG:4326").boundary.plot(
        ax=ax, color="0.35", linewidth=0.5, zorder=3)
    ax.scatter(h["lon"], h["lat"], s=3.0, c=h["Kpet"], cmap=cmap, norm=norm,
               marker="s", linewidths=0, zorder=4)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("calibrated Hamon PET coefficient  Kpet", fontsize=8)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(f"15cdec HRUs by calibrated Kpet  ({len(h)} HRUs, "
                 f"{h['basin'].nunique()} watersheds)", fontsize=9)
    ax.set_xlabel("lon", fontsize=7)
    ax.set_ylabel("lat", fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "hru_kpet_15cdec.png", dpi=_MAP_DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: 11obs + 9unimp per-basin Kpet ---
    doms = ["11obs", "9unimp"]
    hru = {d: load_hru_table(data_dir, domain=d) for d in doms}
    par = {d: load_params(data_dir, domain=d).groupby("basin")["Kpet"].first() for d in doms}
    allh = np.concatenate([hru[d][["lat", "lon"]].to_numpy() for d in doms])
    cext = _hru_extent(allh[:, 0], allh[:, 1])
    kall = np.concatenate([par[d].to_numpy() for d in doms])
    norm = Normalize(vmin=np.floor(kall.min() * 10) / 10, vmax=np.ceil(kall.max() * 10) / 10)

    fig, axes = plt.subplots(1, 2, figsize=(_MAP_W, 4.6))
    for ax, d in zip(axes, doms, strict=True):
        _hru_map_context(ax, data_dir, cext)
        hb = hru[d].copy()
        hb["Kpet"] = hb["basin"].map(par[d])
        hb = hb.sort_values("Kpet")           # high Kpet last -> deterministic shared-cell colour
        gpd.GeoDataFrame(geometry=_cell_boxes(hb["lat"], hb["lon"])).plot(
            ax=ax, color=[to_rgba(cmap(norm(v))) for v in hb["Kpet"]],
            edgecolor="white", linewidth=0.05, zorder=3)
        for b, geom in basin_footprints(data_dir, domain=d).items():
            gpd.GeoSeries([geom], crs="EPSG:4326").boundary.plot(
                ax=ax, color="0.15", linewidth=0.4, zorder=4)
            pt = geom.representative_point()
            ax.text(pt.x, pt.y, f"{_BASIN_ABBREV.get(b, b)}\n{par[d][b]:.2f}", fontsize=5.0,
                    ha="center", va="center", zorder=6, path_effects=halo)
        ax.set_title(f"{d}  ({len(hb)} HRUs, {hb['basin'].nunique()} basins)", fontsize=8)
        ax.set_xlabel("lon", fontsize=7)
    axes[0].set_ylabel("lat", fontsize=7)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.045, pad=0.02)
    cb.set_label("Hamon PET coefficient  Kpet", fontsize=8)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle("SAC-SMA CalSim domains — Hamon PET coefficient (per-basin uniform)", fontsize=9)
    fig.savefig(figs / "hru_kpet_calsim.png", dpi=_MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print("compare: wrote hru_veg_15cdec.png, hru_soil_15cdec.png, hru_kpet_15cdec.png, "
          f"hru_kpet_calsim.png -> {figs}")


def hru_param_table(data_dir: str | Path = "data", domain: str = "15cdec") -> pd.DataFrame:
    """Per ``veg_class`` summary of the calibrated Hamon PET coefficient ``Kpet`` (the
    tabular companion to :func:`make_hru_attribute_maps`).

    In the 15cdec pooled optimum ``Kpet`` is regionalized by land-cover **and** soil zone —
    it is single-valued for each ``(veg_class, soil_class)`` pair but varies with soil within
    a ``veg_class`` — so this table reports the ``Kpet`` spread per class (median + range +
    number of distinct values) alongside the HRU count.  ``veg_class`` codes follow the
    study's MODIS/IGBP-lineage land-cover scheme (Hansen et al. 2010); the repo carries no
    authoritative code→name legend, so no name column is emitted.
    """
    from ..io import load_hru_table, load_params

    h = load_hru_table(data_dir, domain=domain)
    kp = load_params(data_dir, domain=domain)
    kp = (kp.drop_duplicates("basin") if "basin" in kp.columns and domain != "15cdec"
          else kp.drop_duplicates("key"))
    h = h.merge(kp[["key", "Kpet"]], on="key", how="left")
    g = h.groupby("veg_class")["Kpet"]
    tbl = pd.DataFrame({
        "veg_class": [int(c) for c in g.size().index],
        "n_hru": g.size().to_numpy(),
        "pct_domain": (100 * g.size() / len(h)).round(1).to_numpy(),
        "kpet_min": g.min().round(4).to_numpy(),
        "kpet_median": g.median().round(4).to_numpy(),
        "kpet_max": g.max().round(4).to_numpy(),
        "n_distinct_kpet": g.nunique().to_numpy(),
    }).sort_values("n_hru", ascending=False).reset_index(drop=True)
    return tbl


def kpet_soil_table(data_dir: str | Path = "data", domain: str = "15cdec") -> pd.DataFrame:
    """Exact ``soil_class → Kpet`` lookup for the 15cdec pooled optimum.

    The calibrated Hamon PET coefficient is regionalized on **soil zone alone**: within this
    domain ``Kpet`` is single-valued per ``soil_class`` and ``veg_class`` adds nothing (verified).
    So this 1:1 lookup — not the per-``veg_class`` spread in :func:`hru_param_table` — is the
    real parameterization; the ``veg_class`` map and the ``Kpet`` map look alike only because
    soil and land cover co-vary spatially.
    """
    from ..io import load_hru_table, load_params

    h = load_hru_table(data_dir, domain=domain)
    kp = load_params(data_dir, domain=domain).drop_duplicates("key")
    m = h.merge(kp[["key", "Kpet"]], on="key", how="left")
    g = m.groupby("soil_class")["Kpet"]
    tbl = pd.DataFrame({
        "soil_class": [int(c) for c in g.size().index],
        "kpet": g.first().round(6).to_numpy(),
        "n_hru": g.size().to_numpy(),
        "pct_domain": (100 * g.size() / len(m)).round(1).to_numpy(),
        "n_distinct_kpet": g.nunique().to_numpy(),   # 1 for every row (soil fixes Kpet)
    }).sort_values("n_hru", ascending=False).reset_index(drop=True)
    return tbl


def _calset_skill_fig(anchor_met, set_name, path):
    """Per-set **basin-level** skill: the set's anchor KGE per main basin vs CalSim3
    (:func:`anchor_metrics`), with VIC alongside on the same basins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = anchor_met[anchor_met["set"] == set_name]
    s = m[m["source"] == set_name].set_index("basin")
    if s.empty:
        return
    order = s.sort_values("kge", ascending=False).index
    v = m[m["source"] == "vic"].set_index("basin").reindex(order)

    fig, ax = plt.subplots(figsize=(_MAP_W, 4.0))
    x = np.arange(len(order))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.scatter(x, s.reindex(order)["kge"], s=22, alpha=0.9,
               color=_COLORS.get(set_name, "k"),
               label=f"{set_name} (med {s['kge'].median():.2f})")
    if v["kge"].notna().any():
        ax.scatter(x, v["kge"], s=18, alpha=0.8, color=_COLORS["vic"], marker="^",
                   label=f"VIC (med {v['kge'].median():.2f})")
    ax.set_ylim(-1, 1)
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=6, rotation=90)
    ax.set_ylabel("basin monthly KGE vs CalSim3 anchor", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="0.92", lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f"{set_name} basin-level KGE vs CalSim3 anchor (with VIC) — "
                 f"{len(order)} basins", fontsize=8)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=_MAP_DPI)
    plt.close(fig)


_ANCHOR_STYLE = {k: (_COLORS[k], _LABELS[k]) for k in _COLORS}   # the one shared palette

#: the 8 main river indices (CA "8-River Index" rim systems) for the anchor hydrographs —
#: 4 Sacramento-side + 4 San Joaquin-side (Shasta / Trinity / Whiskeytown deliberately dropped).
MAIN_RIVERS = [
    ("SRBB", "Sacramento R → Bend Bridge"),
    ("OROV", "Feather R → Oroville"),
    ("YUBA", "Yuba R → Smartsville"),
    ("FOLS", "American R → Folsom"),
    ("ST",   "Stanislaus R → New Melones"),
    ("TU",   "Tuolumne R → New Don Pedro"),
    ("ME",   "Merced R → Exchequer"),
    ("SJ",   "San Joaquin R → Millerton"),
]


def load_basin_nodes(data_dir: str | Path = "data", domain: str = "15cdec") -> pd.DataFrame:
    """Basin -> CalSim node mapping for a set, projected from the hand-edited crosswalk
    (:func:`calsim.derive_basin_nodes`).  ``in_calsim3`` is the crosswalk's flag (a node is
    usable only when its CalSim3 ``INFLOW`` series is non-zero — e.g. ``I_RUB002`` is folded
    into Folsom, so it is ``False``).  Edit ``calsim_crosswalk.csv`` to change the mapping."""
    from .catchments import derive_basin_nodes
    return derive_basin_nodes(data_dir, domain)


def _screened_basin_flow(basin, domain, fp_basin, forcing, hru_tbl, params_df, *, comp_cache=None):
    """Daily area-weighted flow (mm/day) for a basin's HRUs **screened to its true CalSim
    catchment** (:func:`sacsma.calsim.catchments.screened_footprint`), weighted by GIS-overlap
    area.  Routing is identical to :func:`sacsma.model.run_basin` (PET -> Snow-17 -> SAC-SMA ->
    Lohmann per HRU, then area-weight) — only the HRU **subset** and the **weights** differ, so
    the returned series is directly comparable to the full-footprint run.  ``comp_cache`` is the
    same (domain, key, params)-keyed cache the full anchor uses, so shared HRUs are computed once
    and are bit-identical across the two footprints."""
    from .. import parameters as P
    from ..model import default_is_outlet, run_hru_components_cached
    from ..routing import lohmann

    w = fp_basin.set_index("key")["overlap_area_mi2"]
    sub = hru_tbl[(hru_tbl["basin"] == basin) & (hru_tbl["key"].isin(w.index))]
    sub = sub.reset_index(drop=True)
    wnorm = sub["key"].map(w).to_numpy(dtype=float)
    wnorm = wnorm / wnorm.sum()
    dates, doy, is_leap = forcing.dates, forcing.doy, forcing.is_leap
    total = np.zeros(len(dates))
    for i, hru in enumerate(sub.itertuples(index=False)):
        c = forcing.pos[hru.key]
        ga_row = params_df.loc[hru.key]
        surf, base = run_hru_components_cached(
            comp_cache, domain, hru.key, forcing.prcp[c], forcing.tavg[c], doy, is_leap,
            lat=float(hru.lat), elev=float(hru.elev), ga_row=ga_row)
        is_outlet = default_is_outlet(float(hru.flowlen))
        runoff, _ = lohmann(surf, base, float(hru.flowlen), P.routing_par(ga_row), is_outlet)
        total += wnorm[i] * runoff
    return pd.DataFrame({"date": dates, "flow": total})


def _anchor_set_taf(domain, data_dir, nodes, forcing=None, *, comp_cache=None,
                    parallel=False, footprint=None):
    """Per basin monthly TAF for one set: SAC-SMA basin run + CalSim3 reference + VIC.

    The CalSim3 **reference** is chosen per basin (``ref_kind``):

    - ``unimp`` — if the basin maps to a CalSim rim system (``BASIN_RIM_SYSTEM``)
      that has a ``FLOW-UNIMPAIRED`` series, that single whole-watershed series is
      the faithful reference.  This is the only correct target for systems like Sac @
      Bend Bridge, whose flow includes valley-floor/local accretion that the sum of
      individual INFLOW sub-arcs does **not** capture (~12% low for SRBB).
    - ``inflow_sum`` — otherwise (creeks, secondary basins with no aggregate series),
      sum the basin's assigned CalSim3 INFLOW sub-arcs.

    The SAC-SMA volume uses the canonical CalSim catchment area (``basin_areas`` prefers
    ``basin_area_<domain>_calsim.csv``).
    """
    from ..io import mmday_to_cfs
    from ..model import run_basin
    from .catchments import BASIN_RIM_SYSTEM, basin_areas

    areas = basin_areas(data_dir, domain=domain)
    c3 = load_calsim3_monthly(data_dir)
    vic = load_vic_monthly(data_dir)
    arc2vic = load_name_map(data_dir)   # arc -> VIC major-basin series (crosswalk vic_basin)
    bsys = BASIN_RIM_SYSTEM.get(domain, {})
    unimp_by_sys = {s: g for s, g in load_unimpaired_monthly(data_dir).groupby("system")}
    summable = nodes[nodes["in_calsim3"].astype(bool)]
    # optional corrected-footprint override: for basins in `footprint`, replace the full-footprint
    # run_basin with the GIS-screened HRU subset (screened_footprint), overlap-area weighted.  The
    # CalSim3/VIC references below are unchanged, so the screened anchor is row-comparable to full.
    fp_basins: set = set()
    hru_tbl = pfull = None
    if footprint is not None and len(footprint):
        from ..io import load_hru_table, load_params
        fp_basins = set(footprint["basin"].unique())
        hru_tbl = load_hru_table(data_dir, domain=domain)
        pfull = load_params(data_dir, domain=domain)
    parts = []
    for basin, g in summable.groupby("basin"):
        if basin in fp_basins:
            pb = pfull[pfull["basin"] == basin] if "basin" in pfull.columns else pfull
            df = _screened_basin_flow(basin, domain, footprint[footprint["basin"] == basin],
                                      forcing, hru_tbl, pb.set_index("key"), comp_cache=comp_cache)
        else:
            df = run_basin(basin, data_dir=data_dir, domain=domain, forcing=forcing,
                           comp_cache=comp_cache, parallel=parallel)
        s = pd.Series(mmday_to_cfs(df["flow"].to_numpy(), areas[basin]),
                      index=pd.to_datetime(df["date"]))
        sac = _cfs_day_to_taf(s.groupby(s.index.to_period("M")).sum())
        sac.index = sac.index.to_timestamp("M")
        # Reference: faithful whole-watershed FLOW-UNIMPAIRED where a rim system exists,
        # else the sum of the basin's CalSim3 INFLOW sub-arcs.
        system = bsys.get(basin)
        if system is not None and system in unimp_by_sys:
            u = unimp_by_sys[system]
            uidx = pd.to_datetime(u["date"]).dt.to_period("M").dt.to_timestamp("M")
            cs = pd.Series(u["flow_taf"].to_numpy(), index=uidx).groupby(level=0).sum()
            ref_kind = "unimp"
        else:
            arcs = list(g["arc"].astype(str))
            cs = c3[c3["arc"].isin(arcs)].groupby("date")["flow_taf"].sum()
            ref_kind = "inflow_sum"
        # VIC at the basin level: a rim basin uses its ONE major-basin (8-River) series —
        # drop nested-inherited arcs (e.g. Shasta in Bend Bridge) so the 8RI total, which
        # already includes them, is not double-counted; secondary basins sum per-node VIC.
        own = bsys.get(basin)
        varcs = [a for a, sy in zip(g["arc"].astype(str), g["system"])
                 if own is None or sy == own]
        vnames = {arc2vic.get(a, a) for a in varcs}
        vv = vic[vic["vic_name"].isin(vnames)].groupby("date")["flow_taf"].sum()
        for src, series in [(domain, sac), ("calsim3", cs), ("vic", vv)]:
            if len(series):
                parts.append(pd.DataFrame({"date": series.index, "set": domain, "basin": basin,
                                           "source": src, "flow_taf": series.to_numpy(),
                                           "ref_kind": ref_kind}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_anchor_long(data_dir: str | Path = "data", sets=DEFAULT_CALSETS,
                      *, comp_cache=None, parallel=False, footprint=None,
                      product: str | None = None) -> pd.DataFrame:
    """Long [date, set, basin, source, flow_taf, ref_kind] for the basin-level anchor.

    The SAC volume is on the canonical CalSim catchment area.  Pass a shared
    ``comp_cache`` dict to reuse per-cell SMA components across the anchor and
    per-catchment builds — never share one across ``product``s (it is keyed per cell).
    ``footprint`` (a ``{domain: screened_footprint_df}`` dict) switches
    the basins it names onto the GIS-screened footprint (see :func:`_anchor_set_taf`) —
    in the **official** :func:`make_all` anchor that is
    :data:`~.catchments.SCREENED_BASINS` (SHA/BND Goose Lake + SNS/Chowchilla
    over-reach; every other basin runs its full calibrated footprint); omit it for the
    everything-unscreened view (the parallel artifact, :func:`make_anchor_full`).
    ``product`` selects an
    alternate forcing (e.g. ``historical_lto``; default = the Livneh-unsplit baseline)
    — the CalSim3/VIC reference columns are unaffected
    (:mod:`~sacsma.calsim.forcing_compare` uses this for the forcing-effect skill)."""
    from ..model import load_domain_forcing

    parts = []
    for dom in sets:
        nodes = load_basin_nodes(data_dir, dom)
        kw = {} if product is None else {"product": product}
        forcing = load_domain_forcing(data_dir, domain=dom, **kw)
        fp = footprint.get(dom) if footprint else None
        parts.append(_anchor_set_taf(dom, data_dir, nodes, forcing,
                                     comp_cache=comp_cache, parallel=parallel, footprint=fp))
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def _screened_fp(data_dir, sets) -> dict:
    """``{domain: screened_footprint}`` for the anchor sets — the OFFICIAL anchor basis.

    Since 2026-07-08 the screening covers ONLY :data:`~.catchments.SCREENED_BASINS`
    (SHA/BND — the endorheic Goose Lake block — plus SNS/ChowchillaRiver — delineation
    over-reach); every other basin runs on its full calibrated footprint, so its anchor
    volume is the full-footprint area-weighted depth times the canonical CalSim3
    catchment area.  Every standalone entry point
    that (re)builds the anchor must pass this to :func:`build_anchor_long` so it
    matches the :func:`make_all` anchor; only :func:`make_anchor_full` builds without
    it (the parallel everything-unscreened view)."""
    from .catchments import screened_footprint
    return {s: screened_footprint(data_dir, domain=s) for s in sets if s in ANCHOR_SETS}


def anchor_metrics(long: pd.DataFrame) -> pd.DataFrame:
    """Per (set, basin, source) skill vs CalSim3 over the basin's common period.

    The set's SAC-SMA run and VIC are scored on the **same months** — the intersection of
    the set, VIC and CalSim3 series — so the SAC-vs-VIC head-to-head is on an identical
    period rather than each source's own overlap with CalSim3.
    """
    rows = []
    for (st, basin), g in long.groupby(["set", "basin"]):
        wide = g.pivot_table(index="date", columns="source", values="flow_taf")
        if "calsim3" not in wide:
            continue
        rk = (g["ref_kind"].dropna().iloc[0] if "ref_kind" in g and g["ref_kind"].notna().any()
              else "")
        present = [s for s in [st, "vic"] if s in wide]
        common = wide[present + ["calsim3"]].dropna()   # one period shared by all candidates
        if len(common) < 12:
            continue
        ref = common["calsim3"].to_numpy()
        # center of timing (water-year Oct–Sep): sim vs ref over the same aligned months;
        # dct_mo > 0 means the simulated seasonal mass is later than CalSim3 (months).
        ct_ref = center_of_timing(common.index, ref)
        for src in present:
            sim = common[src].to_numpy()
            ct_sim = center_of_timing(common.index, sim)
            rows.append({"set": st, "basin": basin, "source": src, "ref_kind": rk,
                         "n_months": len(common),
                         "kge": kge(sim, ref), "nse": nse(sim, ref), "pbias": pbias(sim, ref),
                         "r": pearson(sim, ref), "mean_sim_taf": float(sim.mean()),
                         "mean_calsim3_taf": float(ref.mean()),
                         # seasonal-shape error: fraction of annual flow in the wrong month [0,1]
                         "seas_mismatch": seasonal_mismatch(common.index, sim, ref),
                         "ct_sim_mo": ct_sim, "ct_ref_mo": ct_ref, "dct_mo": ct_sim - ct_ref})
    return pd.DataFrame(rows)


#: the basin-level anchor uses only the GAUGE-calibrated sets — 11obs (rim gauges) and
#: 9unimp (creek gauges).  15cdec (reservoir-calibrated, the ~-23% rim bias) is excluded from
#: the anchor and appears only in the per-catchment / sub-arc best-of.
ANCHOR_SETS = ("11obs", "9unimp")


def make_anchor(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts",
                run: str = "compare", sets=DEFAULT_CALSETS, *, anchor_long=None) -> Path:
    """Basin-level anchor comparison: each set basin vs the sum of its CalSim3 nodes, using
    only the gauge-calibrated anchor sets (:data:`ANCHOR_SETS` = 11obs/9unimp; 15cdec is
    excluded from ``anchor_metrics.csv``/``anchor_monthly.csv`` and everything downstream of
    them — it is reservoir-calibrated and only contributes per-catchment sub-arcs elsewhere).
    ``anchor_long`` may be passed in (already built by :func:`make_all`) to avoid recompute.

    The three ``anchor_skill_{kge,nse,pbias}.png`` dumbbells ADD 15cdec on afterward (via
    :func:`make_anchor_15cdec`), behind a dashed divider (:func:`_anchor_dumbbell_fig`) — a
    display-only addition that does not touch the ANCHOR_SETS CSVs above.
    """
    out = Path(artifacts_dir) / "calsim" / run
    (out / "figures").mkdir(parents=True, exist_ok=True)
    long = (build_anchor_long(data_dir, sets, footprint=_screened_fp(data_dir, sets))
            if anchor_long is None else anchor_long)
    sets = tuple(s for s in sets if s in ANCHOR_SETS)        # 11obs / 9unimp only
    long = long[long["set"].isin(sets)].copy()
    met = anchor_metrics(long)
    long.to_csv(out / "anchor_monthly.csv", index=False)
    met.to_csv(out / "anchor_metrics.csv", index=False)
    # surface the canonical CalSim catchment areas (the basin total now sits on these)
    for s in sets:
        from . import calsim_dir
        area_csv = calsim_dir(data_dir) / f"basin_area_{s}_calsim.csv"
        if area_csv.exists():
            pd.read_csv(area_csv).to_csv(out / f"basin_area_{s}_calsim.csv", index=False)
    msg = "  ".join(f"{s}={met[(met['set']==s)&(met['source']==s)]['kge'].median():.2f}"
                    for s in sets if ((met['set'] == s) & (met['source'] == s)).any())
    print(f"anchor: basin-level vs CalSim3 (FLOW-UNIMPAIRED where a rim system exists, "
          f"else sum of INFLOW sub-nodes); median KGE [{msg}] -> {out}")

    # fold 15cdec onto the skill dumbbells only (own CSVs, own median print; never merged
    # into the ANCHOR_SETS met/CSVs above)
    has_cdec15 = anchor_long is None or CDEC15 in set(anchor_long["set"])
    long_15, met_15 = (make_anchor_15cdec(data_dir, artifacts_dir, run, anchor_long=anchor_long)
                      if has_cdec15 else (pd.DataFrame(), pd.DataFrame()))
    plot_sets = sets + (CDEC15,) if not met_15.empty else sets
    plot_met = pd.concat([met, met_15], ignore_index=True) if not met_15.empty else met

    for col, lab, xlim in [("kge", "monthly KGE", (0.0, 1.0)), ("nse", "monthly NSE", (0.0, 1.0)),
                           ("pbias", "percent bias (%)", None)]:
        _anchor_dumbbell_fig(plot_met, plot_sets, col, lab, xlim,
                             out / "figures" / f"anchor_skill_{col}.png", data_dir=data_dir)
    _anchor_scatter_fig(long, sets, out / "figures" / "anchor_scatter.png")
    for st in sets:                          # one |pbias| x seasonal-mismatch diagram per set
        _anchor_bias_diagram_fig(met, st, out / "figures" / f"anchor_pbias_vs_seasonal_{st}.png")
    _anchor_hydrograph_fig(long, sets, out / "figures" / "anchor_hydrographs.png")
    if "11obs" in set(long["set"]):
        _main_river_climatology_fig(long, out / "figures" / "main_river_climatology.png",
                                    climset="11obs")
    return out


def target_vs_calsim3(data_dir: str | Path = "data", sets=ANCHOR_SETS) -> pd.DataFrame:
    """How different is the SAC-SMA **calibration target itself** from CalSim3's own flow?

    The CalLite sets are GA-calibrated to an observed monthly FNF **depth**
    (``fnf_<domain>_monthly`` ``obs_mm``).  Expressed on the SAME canonical CalSim catchment area
    the cross-compare uses for the model (:func:`~sacsma.calsim.catchments.basin_areas` ->
    ``basin_area_<domain>_calsim.csv``), this scores that target volume directly against CalSim3's
    unimpaired FNF (``calsim_unimpaired_monthly`` for rim systems, else the summed ``I_`` INFLOW).
    ``pbias`` here is the bias a **perfect-fit** model would inherit from its target alone — the
    floor under the anchor scores, independent of model skill.

    ``area_implied`` is the area at which the target volume would exactly equal CalSim3; comparing
    it to ``area_gis`` (CalSim catchment) and ``area_pub`` (published drainage area) shows whether
    a target/CalSim3 gap is a real volume difference or an area-basis artifact.  ``class`` labels
    each basin: ``consistent`` (implied ~= area_gis), ``area_artifact`` (implied ~= published, not
    gis: the target was normalized on the published area but scored on the CalSim catchment — e.g.
    Chowchilla/SNS/YRS **before** the fnf area-harmonization), or ``product_offset`` (implied off
    BOTH areas: a genuine historical-FNF-vs-CalSim3 difference no area choice fixes).

    Returns ``[set, basin, ref_kind, n_months, kge, pbias, r, target_mean_taf, calsim3_mean_taf,
    area_gis, area_implied, area_pub, class]``.
    """
    from ..io import load_basin_area, mmday_to_cfs
    from ..metrics import kge as _kge
    from ..metrics import pbias as _pbias
    from ..metrics import pearson as _pearson
    from . import load_fnf_monthly
    from .catchments import BASIN_RIM_SYSTEM, basin_areas

    taf_per_mm_mi2 = mmday_to_cfs(1.0, 1.0) * _AF_PER_CFS_DAY / 1000.0   # exact vs the anchor path
    c3m = load_calsim3_monthly(data_dir)
    c3m = c3m.assign(month=pd.to_datetime(c3m["date"]).dt.to_period("M"))
    unimp_by_sys = {s: g.assign(month=pd.to_datetime(g["date"]).dt.to_period("M"))
                        .groupby("month")["flow_taf"].sum()   # collapse any dup (system,month) rows
                    for s, g in load_unimpaired_monthly(data_dir).groupby("system")}
    rows = []
    for dom in sets:
        nodes = load_basin_nodes(data_dir, dom)
        bsys = BASIN_RIM_SYSTEM.get(dom, {})
        areas = basin_areas(data_dir, domain=dom)
        try:                                             # published drainage area, for classing
            _pa = load_basin_area(data_dir, domain=dom).set_index("basin")["area_mi2"]
            pub = {str(b): float(a) for b, a in _pa.items()}
        except FileNotFoundError:
            pub = {}
        fnf = load_fnf_monthly(data_dir, domain=dom)
        fnf = fnf.assign(month=pd.to_datetime(fnf["date"]).dt.to_period("M"))
        for basin, g in nodes[nodes["in_calsim3"].astype(bool)].groupby("basin"):
            sysn = bsys.get(basin)
            if sysn is not None and sysn in unimp_by_sys:
                ref, ref_kind = unimp_by_sys[sysn], "unimp"
            else:
                arcs = list(g["arc"].astype(str))
                ref = c3m[c3m["arc"].isin(arcs)].groupby("month")["flow_taf"].sum()
                ref_kind = "inflow_sum"
            o = fnf[fnf["basin"] == basin].set_index("month")["obs_mm"]
            j = pd.concat([o.rename("obs"), ref.rename("ref")], axis=1).dropna()
            if len(j) < 12:
                continue
            area = float(areas[basin])
            tgt = j["obs"].to_numpy() * area * taf_per_mm_mi2
            ref_taf = j["ref"].to_numpy()
            osum = float(j["obs"].sum())
            implied = (ref_taf.sum() / (osum * taf_per_mm_mi2)) if osum else float("nan")
            # Class: is a target/CalSim3 gap an area-basis artifact or a real volume difference?
            # implied ~= area_gis -> consistent; implied ~= published (not gis) -> area_artifact
            # (target normalized on the published area, scored on the CalSim area); else a genuine
            # product difference (implied off BOTH areas), which no area choice fixes.
            area_pub = pub.get(basin, float("nan"))
            rel_gis = abs(implied - area) / area if (area and implied == implied) else float("nan")
            rel_pub = (abs(implied - area_pub) / area_pub
                       if (area_pub == area_pub and area_pub and implied == implied)
                       else float("nan"))
            if rel_gis == rel_gis and rel_gis < 0.03:
                cls = "consistent"
            elif rel_pub == rel_pub and rel_pub < 0.03:
                cls = "area_artifact"
            else:
                cls = "product_offset"
            rows.append({"set": dom, "basin": basin, "ref_kind": ref_kind, "n_months": len(j),
                         "kge": round(_kge(tgt, ref_taf), 3),
                         "pbias": round(_pbias(tgt, ref_taf), 1),
                         "r": round(_pearson(tgt, ref_taf), 3),
                         "target_mean_taf": round(float(tgt.mean()), 1),
                         "calsim3_mean_taf": round(float(ref_taf.mean()), 1),
                         "area_gis": round(area, 1), "area_implied": round(float(implied), 1),
                         "area_pub": round(area_pub, 1) if area_pub == area_pub else None,
                         "class": cls})
    return pd.DataFrame(rows)


def make_anchor_full(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts",
                     run: str = "compare", sets=DEFAULT_CALSETS, *, anchor_long=None,
                     period=None, comp_cache=None, parallel=False) -> Path:
    """Parallel full-HRU-footprint anchor + the screened-vs-full delta.

    The **official** anchor (``anchor_metrics.csv``/``anchor_monthly.csv`` and everything
    downstream — skill dumbbells, period splits, rolling skill, maps, the sub-arc QMAP
    mass-balance target) runs on the **corrected (GIS-screened) footprint**
    (:func:`~sacsma.calsim.catchments.screened_footprint`): each anchor basin simulated only on
    the HRUs inside its true CalSim catchment, overlap-area weighted — consistent with the
    per-catchment sub-arcs and with VIC's ``no_gooselake`` substitution.  This function keeps
    the original **full-footprint** view alive as the parallel artifact
    (``anchor_metrics_full.csv`` + ``anchor_monthly_full.csv``) and writes the delta table +
    figure (``anchor_screened_vs_full.csv``): pbias/KGE full -> screened per basin, plus the
    **VIC benchmark** (``pbias_vic``/``kge_vic``/``mean_vic_taf``; purple diamond) on the same
    months and reference.  Screening removes out-of-catchment dilution (SHA -8.9 -> +0.1,
    SNS -7.8 -> -0.8, Chowchilla -14.3 -> -4.3); for BND and Fresno R it removes a
    *compensating* dilution to expose an honest over-prediction (see
    ``tmp/CALSIM3_FNF_FOOTPRINT.md``).  ``anchor_long`` is the OFFICIAL (screened) long from
    :func:`make_all` (rebuilt here if omitted); ``period`` (a ``(start, end)`` pair) clips both
    views to identical months for a fair delta.  The ``fnf_<domain>_monthly.csv`` calibration
    basis and the fnf-target diagnostics are unaffected by the anchor basis."""
    out = Path(artifacts_dir) / "calsim" / run
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ssets = tuple(s for s in sets if s in ANCHOR_SETS)
    if anchor_long is None:
        anchor_long = build_anchor_long(data_dir, ssets, comp_cache=comp_cache, parallel=parallel,
                                        footprint=_screened_fp(data_dir, ssets))
    screened_long = anchor_long[anchor_long["set"].isin(ssets)].copy()
    full = build_anchor_long(data_dir, ssets, comp_cache=comp_cache, parallel=parallel)
    if period is not None:
        s0, e0 = period
        full = full[full["date"].between(s0, e0)].reset_index(drop=True)
        screened_long = screened_long[screened_long["date"].between(s0, e0)].reset_index(drop=True)
    full.to_csv(out / "anchor_monthly_full.csv", index=False)
    met_full = anchor_metrics(full)
    met_full.to_csv(out / "anchor_metrics_full.csv", index=False)
    met_scr = anchor_metrics(screened_long)

    def _sac(m):                              # the SAC-SMA row for each basin (source == set name)
        return m[m["source"] == m["set"]].set_index(["set", "basin"])
    f, s = _sac(met_full), _sac(met_scr)
    # VIC on the SAME months/reference (its rows are identical in full and screened longs —
    # only the SAC source changes with the footprint).  VIC handles its own Goose Lake
    # over-reach by series substitution (no_gooselake at I_SHSTA/8RI_SRBB), so it is the fair
    # third rail for the footprint question at SHA/BND.
    v = met_scr[met_scr["source"] == "vic"].set_index(["set", "basin"])
    cmp_rows = []
    for key in s.index:
        rs = s.loc[key]
        rf = f.loc[key] if key in f.index else None
        rv = v.loc[key] if key in v.index else None
        nan = float("nan")
        cmp_rows.append({
            "set": key[0], "basin": key[1], "ref_kind": rs["ref_kind"],
            "n_months": int(rs["n_months"]),
            "pbias_full": round(float(rf["pbias"]), 1) if rf is not None else nan,
            "pbias_screened": round(float(rs["pbias"]), 1),
            "pbias_vic": round(float(rv["pbias"]), 1) if rv is not None else nan,
            "kge_full": round(float(rf["kge"]), 3) if rf is not None else nan,
            "kge_screened": round(float(rs["kge"]), 3),
            "kge_vic": round(float(rv["kge"]), 3) if rv is not None else nan,
            "mean_sim_full_taf": round(float(rf["mean_sim_taf"]), 1) if rf is not None else nan,
            "mean_sim_screened_taf": round(float(rs["mean_sim_taf"]), 1),
            "mean_vic_taf": round(float(rv["mean_sim_taf"]), 1) if rv is not None else nan,
            "mean_calsim3_taf": round(float(rs["mean_calsim3_taf"]), 1)})
    cmp = pd.DataFrame(cmp_rows)
    cmp["dpbias"] = (cmp["pbias_screened"] - cmp["pbias_full"]).round(1)
    cmp["dkge"] = (cmp["kge_screened"] - cmp["kge_full"]).round(3)
    cmp = cmp.sort_values(["set", "basin"]).reset_index(drop=True)
    cmp.to_csv(out / "anchor_screened_vs_full.csv", index=False)
    _anchor_screened_fig(cmp, ssets, out / "figures" / "anchor_screened_vs_full.png",
                         data_dir=data_dir)
    med = {st: cmp[cmp["set"] == st]["pbias_screened"].abs().median() for st in ssets}
    print("anchor (screened official vs full-footprint parallel): median screened |pbias| "
          + "  ".join(f"{st}={med[st]:.1f}%" for st in ssets)
          + f" -> {out}/anchor_screened_vs_full.csv")
    return out


def _clipv(v, lo, hi):
    """Clamp a plotted value to the axis limits (off-scale markers sit at the edge)."""
    return min(max(v, lo), hi)


def _anchor_screened_fig(cmp, sets, path, data_dir="data"):
    """Full footprint -> GIS-screened corrected footprint per anchor basin, on pbias and KGE.
    Open marker = full (current anchor), filled = screened; a green connector means screening
    moved the value toward CalSim3 (pbias toward 0 / higher KGE), red = away.  The VIC
    benchmark (same months, same CalSim3 reference; its own footprint conventions, incl. the
    no_gooselake substitution at SHA/BND) is the purple diamond."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [(s, b) for s in sets
            for b in basin_order_north_south(data_dir, s, cmp[cmp["set"] == s]["basin"].unique())]
    if not rows:
        return
    labels = [f"{s}:{_BASIN_ABBREV.get(b, b)}" for s, b in rows]
    y = np.arange(len(rows))[::-1]                 # first (northern) basin at the top

    def get(s, b, c):
        m = cmp[(cmp["set"] == s) & (cmp["basin"] == b)]
        return float(m[c].iloc[0]) if len(m) else np.nan

    fig, ax = plt.subplots(1, 2, figsize=(_MAP_W, 0.30 * len(rows) + 1.3), sharey=True)
    vic_col = _ANCHOR_STYLE["vic"][0]
    # axis limits sized to the SAC values; VIC outliers (e.g. Fresno +95% / KGE -0.18) clamp to
    # the edge rather than stretching the shared scale (KGE axis stays 0-1 per the house rule).
    sac_pb = pd.concat([cmp["pbias_full"], cmp["pbias_screened"]]).dropna()
    pb_lim = (min(-25.0, float(sac_pb.min()) - 5.0), max(50.0, float(sac_pb.max()) + 5.0))
    for col_full, col_scr, col_vic, panel, lab, zero, lim in [
            ("pbias_full", "pbias_screened", "pbias_vic", ax[0], "percent bias (%)", True, pb_lim),
            ("kge_full", "kge_screened", "kge_vic", ax[1], "monthly KGE", False, (0.0, 1.0))]:
        for yi, (s, b) in zip(y, rows):
            vf, vs = get(s, b, col_full), get(s, b, col_scr)
            if np.isnan(vf) or np.isnan(vs):
                continue
            vf, vs = _clipv(vf, *lim), _clipv(vs, *lim)
            improved = (abs(vs) < abs(vf)) if zero else (vs > vf)
            col = "#2ca25f" if improved else "#cc3311"
            panel.plot([vf, vs], [yi, yi], color=col, lw=1.5, zorder=1)
            panel.scatter([vf], [yi], facecolor="white", edgecolor="0.4", s=26, zorder=2, lw=0.9)
            panel.scatter([vs], [yi], color=col, s=30, zorder=3, edgecolor="white", lw=0.5)
            vv = get(s, b, col_vic)
            if not np.isnan(vv):                # VIC benchmark, same months + reference
                panel.scatter([_clipv(vv, *lim)], [yi], color=vic_col, s=22, marker="D",
                              zorder=2.5, edgecolor="white", lw=0.4)
        panel.set_xlim(*lim)
        if zero:
            panel.axvline(0, color="0.6", lw=0.8)
        panel.set_xlabel(lab, fontsize=8)
        panel.tick_params(labelsize=7)
        panel.grid(axis="x", color="0.92", lw=0.6)
        panel.set_axisbelow(True)
    for i in range(1, len(rows)):                  # dashed divider at each set boundary
        if rows[i][0] != rows[i - 1][0]:
            for panel in ax:
                panel.axhline(y[i] + 0.5, color="0.5", lw=0.9, ls="--", zorder=0)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels, fontsize=6)
    ax[1].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="0.4",
               label="full footprint"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
               label="screened (corrected)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=vic_col, label="VIC"),
        Line2D([0], [0], color="#2ca25f", label="toward CalSim3"),
        Line2D([0], [0], color="#cc3311", label="away")], loc="best", fontsize=6)
    fig.suptitle("Anchor vs CalSim3 unimpaired FNF: full vs GIS-screened footprint", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=_MAP_DPI)
    plt.close(fig)


def make_anchor_15cdec(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts",
                       run: str = "compare",
                       *, anchor_long=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """15cdec's basin-level monthly SAC-SMA-vs-VIC series/metrics — computed and stored
    separately from :data:`ANCHOR_SETS`/``anchor_metrics.csv`` (15cdec is reservoir-
    calibrated and stays excluded from the gauge-calibrated anchor and everything
    downstream of it — rolling skill, period splits, sub-arc QMAP). Used by
    :func:`make_anchor`/:func:`make_anchor_skill_periods` to ADD 15cdec onto the anchor-skill
    dumbbells alongside 11obs/9unimp, behind a dashed divider; not a change to those sets'
    own convention.

    15cdec's own calibration target is DAILY (the CDEC gage); ``build_anchor_long``
    aggregates its ``run_basin`` output to **monthly** TAF (identical treatment to
    9unimp/11obs) so the KGE is comparable to VIC (monthly-only in this repo).
    ``anchor_long`` may be passed in (already built by :func:`make_all`, which includes
    15cdec by default via ``DEFAULT_CALSETS``) to avoid recompute. Writes
    ``anchor_monthly_15cdec.csv``/``anchor_metrics_15cdec.csv``; returns ``(long, met)``
    (both empty if 15cdec has no scoreable basins).
    """
    out = Path(artifacts_dir) / "calsim" / run
    out.mkdir(parents=True, exist_ok=True)
    long = (build_anchor_long(data_dir, sets=(CDEC15,)) if anchor_long is None
           else anchor_long[anchor_long["set"] == CDEC15].copy())
    if long.empty:
        return long, pd.DataFrame()
    met = anchor_metrics(long)
    long.to_csv(out / "anchor_monthly_15cdec.csv", index=False)
    met.to_csv(out / "anchor_metrics_15cdec.csv", index=False)
    sac_med = met[(met["set"] == CDEC15) & (met["source"] == CDEC15)]["kge"].median()
    vic_med = met[(met["set"] == CDEC15) & (met["source"] == "vic")]["kge"].median()
    print(f"anchor (15cdec, monthly): median KGE sac={sac_med:.2f} vic={vic_med:.2f} -> {out}")
    return long, met


def make_anchor_skill_periods(data_dir: str | Path = "data",
                              artifacts_dir: str | Path = "artifacts", run: str = "compare",
                              split: str = "1949-10-01") -> pd.DataFrame:
    """Pre-/post-1950 SAC-SMA-vs-VIC anchor dumbbells from the committed anchor series.

    Re-scores :func:`anchor_metrics` on the months before / from ``split`` (WY1950 by the
    house 1950 convention) using ``anchor_monthly.csv`` — no re-simulation.  Writes
    ``anchor_metrics_by_period.csv`` plus KGE and pbias dumbbells per period
    (``anchor_skill_{kge,pbias}_{pre,post}1950.png``); the two pbias figures share one
    symmetric y-scale (house rule), KGE stays on the full 0–1 scale.

    15cdec is folded in from ``anchor_monthly_15cdec.csv`` (if present, written by
    :func:`make_anchor_15cdec`) and re-scored/plotted the same way, appended after
    11obs/9unimp behind a dashed divider (:func:`_anchor_dumbbell_fig`) — it is included in
    ``anchor_metrics_by_period.csv`` for transparency but never in ``anchor_metrics.csv``.
    """
    out = Path(artifacts_dir) / "calsim" / run
    # round_trip: the default fast parser can be 1 ulp off, which would leak formatting
    # noise back into the committed CSVs on a re-run
    long = pd.read_csv(out / "anchor_monthly.csv", parse_dates=["date"],
                       float_precision="round_trip")
    cdec15_path = out / "anchor_monthly_15cdec.csv"
    has_cdec15 = cdec15_path.exists()
    if has_cdec15:
        long15 = pd.read_csv(cdec15_path, parse_dates=["date"], float_precision="round_trip")
        long = pd.concat([long, long15], ignore_index=True)
    plot_sets = ANCHOR_SETS + (CDEC15,) if has_cdec15 else ANCHOR_SETS
    split_ts = pd.Timestamp(split)

    def wy(ts):
        return ts.year + (1 if ts.month >= 10 else 0)

    mets = []
    for plab, sub in (("pre1950", long[long["date"] < split_ts]),
                      ("post1950", long[long["date"] >= split_ts])):
        met = anchor_metrics(sub)
        met.insert(0, "period", plab)
        met.insert(1, "window", f"WY{wy(sub['date'].min())}–{wy(sub['date'].max())}")
        mets.append(met)
    allmet = pd.concat(mets, ignore_index=True)
    allmet.to_csv(out / "anchor_metrics_by_period.csv", index=False)
    pb_lim = float(np.nanmax(np.abs(allmet["pbias"]))) * 1.08   # shared across the periods
    for met in mets:
        plab, window = met["period"].iloc[0], met["window"].iloc[0]
        for col, ylab, ylim in [("kge", "monthly KGE", (0.0, 1.0)),
                                ("pbias", "percent bias (%)", (-pb_lim, pb_lim))]:
            _anchor_dumbbell_fig(met, plot_sets, col, ylab, ylim,
                                 out / "figures" / f"anchor_skill_{col}_{plab}.png",
                                 data_dir=data_dir, title_suffix=f" — {window}")
        sac = met[met["set"] == met["source"]]
        vicm = met[met["source"] == "vic"]
        print(f"anchor by period [{plab} {window}]: median KGE sac={sac['kge'].median():.2f} "
              f"vic={vicm['kge'].median():.2f} | median |pbias| "
              f"sac={sac['pbias'].abs().median():.1f}% vic={vicm['pbias'].abs().median():.1f}%")
    return allmet


def basin_order_north_south(data_dir, domain, basins=None):
    """Basins ordered **north -> south** (descending area-weighted HRU latitude) for figure
    axes, from ``data/hru/hruinfo_<domain>.csv``, with the house Folsom-before-Yuba override
    (:func:`sacsma._figures.folsom_before_yuba`).  If ``basins`` is given, restrict/extend to
    exactly those (unknown basins kept at the end, in their given order)."""
    from .._figures import folsom_before_yuba
    from ..io import load_hru_table
    h = load_hru_table(data_dir, domain=domain)
    lat = (h.groupby("basin")[["lat", "area_weight"]]
             .apply(lambda d: np.average(d["lat"], weights=d["area_weight"]))
             .sort_values(ascending=False))
    order = folsom_before_yuba(domain, list(lat.index))
    if basins is not None:
        keep = set(basins)
        order = [b for b in order if b in keep] + [b for b in basins if b not in set(order)]
    return order


def _anchor_dumbbell_fig(met, sets, col, ylab, ylim, path, data_dir="data", title_suffix=""):
    """**Vertical** dumbbell per (set, basin): SAC-SMA vs VIC for one metric, grouped by
    set along the x-axis (basins as x ticks, metric on the y-axis).  Basins within each set
    are ordered **north -> south** (:func:`basin_order_north_south`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(s, b) for s in sets for b in
            basin_order_north_south(data_dir, s, met[(met["set"] == s)]["basin"].unique())]
    if not rows:
        return
    labels = [f"{s}:{_BASIN_ABBREV.get(b, b)}" for s, b in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(_MAP_W, 4.6))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylim is None or ylim[0] < 0 < ylim[1]:
        ax.axhline(0, color="0.7", lw=0.8)
    # dashed divider at every set boundary (e.g. 9unimp|11obs, 11obs|15cdec)
    for i in range(1, len(rows)):
        if rows[i][0] != rows[i - 1][0]:
            ax.axvline(i - 0.5, color="0.5", lw=0.9, ls="--", zorder=0)

    def val(st, basin, src):
        m = met[(met["set"] == st) & (met["basin"] == basin) & (met["source"] == src)]
        return float(m[col].iloc[0]) if len(m) else np.nan

    def clip(v):
        return min(max(v, ylim[0]), ylim[1]) if ylim is not None else v

    for xi, (st, basin) in zip(x, rows):
        a, b = val(st, basin, st), val(st, basin, "vic")
        vv = [clip(v) for v in (a, b) if not np.isnan(v)]
        if len(vv) > 1:
            ax.plot([xi, xi], [min(vv), max(vv)], color="0.8", lw=1.4, zorder=1)
        if not np.isnan(a):
            ax.scatter([xi], [clip(a)], color=_ANCHOR_STYLE[st][0], s=34, zorder=2,
                       edgecolor="white", linewidth=0.5)
        if not np.isnan(b):
            ax.scatter([xi], [clip(b)], color=_ANCHOR_STYLE["vic"][0], s=26, marker="D",
                       zorder=2, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel(ylab, fontsize=8); ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="0.92", lw=0.6); ax.set_axisbelow(True)
    ax.scatter([], [], color="#666", s=34, label="SAC-SMA (set)")
    ax.scatter([], [], color=_ANCHOR_STYLE["vic"][0], marker="D", s=26, label="VIC")
    ax.legend(loc="best", fontsize=7)
    ax.set_title(_wrap(f"Basin-level {ylab} vs CalSim3 (FLOW-UNIMPAIRED for rim systems, "
                       f"else sub-node sum){title_suffix}"), fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=_MAP_DPI); plt.close(fig)


def _anchor_scatter_fig(long, sets, path):
    """Pooled monthly: each set's basin inflow (y) vs CalSim3 sum (x), common log axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sets), figsize=(_MAP_W, max(2.6, _MAP_W / len(sets))))
    for a, st in zip(np.atleast_1d(axes), sets):
        w = long[long["set"] == st].pivot_table(index=["basin", "date"], columns="source",
                                                values="flow_taf")
        if "calsim3" not in w or st not in w:
            a.set_visible(False); continue
        d = w[["calsim3", st]].dropna()
        a.scatter(d["calsim3"], d[st], s=5, alpha=0.25, color=_ANCHOR_STYLE[st][0])
        lim = [1e-1, 1e5]
        a.plot(lim, lim, "k--", lw=0.8)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
        a.set_aspect("equal")
        a.set_xlabel("CalSim3 reference (unimpaired / sub-node sum, TAF/mo)", fontsize=8)
        a.set_ylabel(f"{_ANCHOR_STYLE[st][1]} (TAF/mo)", fontsize=8)
        a.set_title(f"{st} (KGE={kge(d[st].to_numpy(), d['calsim3'].to_numpy()):.2f})", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=_MAP_DPI); plt.close(fig)


def _anchor_bias_diagram_fig(met, st, path):
    """Volume-vs-seasonal-shape **error diagram** for ONE anchor set: absolute percent bias
    (x, volume-error magnitude) against the seasonal-distribution mismatch (y, the fraction of
    annual flow delivered in the wrong month; :func:`sacsma.metrics.seasonal_mismatch`) — one
    marker per basin for SAC-SMA and for VIC, with a faint connector joining each basin's SAC
    and VIC point.  Both axes are error magnitudes, so the **lower-left corner is a perfect
    match**.  ≤6.5in wide, 300 dpi, 8-pt text.  Reads :func:`anchor_metrics` (needs ``pbias`` +
    ``seas_mismatch``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "seas_mismatch" not in met.columns:
        return
    m = met[met["set"] == st].copy()
    m["abs_pbias"] = m["pbias"].abs()
    sac = m[m["source"] == st].set_index("basin")
    vic = m[m["source"] == "vic"].set_index("basin")
    if sac.empty:
        return
    fig, ax = plt.subplots(figsize=(_MAP_W, 5.0))
    for b in sac.index.intersection(vic.index):              # per-basin SAC -> VIC connector
        ax.plot([sac.loc[b, "abs_pbias"], vic.loc[b, "abs_pbias"]],
                [sac.loc[b, "seas_mismatch"], vic.loc[b, "seas_mismatch"]],
                color="0.85", lw=0.8, zorder=1)
    ax.scatter(sac["abs_pbias"], sac["seas_mismatch"], color=_COLORS.get(st, "k"), s=40,
               edgecolor="white", linewidth=0.5, zorder=3, label=_LABELS.get(st, st))
    ax.scatter(vic["abs_pbias"], vic["seas_mismatch"], color=_COLORS["vic"], s=30, marker="D",
               edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2, label="VIC")
    for src_df, col in ((sac, _COLORS.get(st, "k")), (vic, _COLORS["vic"])):
        for b, r in src_df.iterrows():
            ax.annotate(str(b), (r["abs_pbias"], r["seas_mismatch"]), fontsize=6, color=col,
                        xytext=(3, 2), textcoords="offset points", zorder=4)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel("|percent bias|  (%)   [volume-error magnitude]", fontsize=8)
    ax.set_ylabel("seasonal mismatch   [fraction of annual flow in the wrong month]", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=6)
    ax.set_title(_wrap(f"{_LABELS.get(st, st)} vs VIC — |pbias| × seasonal mismatch vs CalSim3 "
                       f"anchors ({st} basins; lower-left = perfect, grey = same basin)", 80),
                fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=_MAP_DPI); plt.close(fig)


def _anchor_hydrograph_fig(long, sets, path, *, window=("1985-10-01", "2005-09-30")):
    """Monthly hydrographs for the **8 main river indices** (CA 8-River Index rim systems;
    Shasta/Trinity/Whiskeytown excluded): each set's SAC-SMA basin run + VIC against the
    CalSim3 FLOW-UNIMPAIRED reference (bold).  Each set's basin is matched to its river via
    :data:`calsim.BASIN_RIM_SYSTEM`; the reference/VIC series are identical across sets for a
    given river, so they are de-duplicated by date."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .catchments import BASIN_RIM_SYSTEM

    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    win = long[(long["date"] >= lo) & (long["date"] <= hi)].copy()
    sys_of = {(st, b): sy for st in sets for b, sy in BASIN_RIM_SYSTEM.get(st, {}).items()}
    win["system"] = [sys_of.get((s, b)) for s, b in zip(win["set"], win["basin"])]

    fig, axes = plt.subplots(4, 2, figsize=(_MAP_W, 9.0), sharex=True)
    for ax, (sysn, label) in zip(axes.ravel(), MAIN_RIVERS):
        sub = win[win["system"] == sysn]
        if not len(sub):
            ax.set_visible(False); continue
        ref = sub[sub["source"] == "calsim3"].drop_duplicates("date").sort_values("date")
        if len(ref):
            ax.plot(ref["date"], ref["flow_taf"], color=_ANCHOR_STYLE["calsim3"][0], lw=1.9,
                    label="CalSim3 unimpaired", zorder=3)
        for src in list(sets) + ["vic"]:
            s = sub[sub["source"] == src].drop_duplicates("date").sort_values("date")
            if len(s):
                ax.plot(s["date"], s["flow_taf"], color=_ANCHOR_STYLE[src][0], lw=1.0,
                        alpha=0.85, label=_ANCHOR_STYLE[src][1])
        ax.set_title(label, fontsize=8)
        ax.set_ylabel("TAF/mo", fontsize=8)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(_wrap("8 main river indices: SAC-SMA & VIC vs CalSim3 FLOW-UNIMPAIRED (bold)", 70),
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=_MAP_DPI); plt.close(fig)


def _main_river_climatology_fig(long, path, *, climset="11obs"):
    """Mean-monthly (water-year O..S) climatology of the 8 main river indices over the FULL
    period: SAC-SMA (``climset``, default 11obs) vs VIC vs CalSim (FLOW-UNIMPAIRED), TAF/mo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .catchments import BASIN_RIM_SYSTEM

    inv = {sy: b for b, sy in BASIN_RIM_SYSTEM.get(climset, {}).items()}
    sub = long[long["set"] == climset].copy()
    sub["mon"] = pd.to_datetime(sub["date"]).dt.month
    wy = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    labels = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]
    order = [("calsim3", "CalSim (unimpaired)"), (climset, f"SAC-SMA {climset}"), ("vic", "VIC")]

    fig, axes = plt.subplots(4, 2, figsize=(_MAP_W, 9.0), sharex=True)
    for ax, (sysn, label) in zip(axes.ravel(), MAIN_RIVERS):
        b = inv.get(sysn)
        g = sub[sub["basin"] == b] if b is not None else sub.iloc[:0]
        if not len(g):
            ax.set_visible(False); continue
        for src, slabel in order:
            s = g[g["source"] == src]
            if len(s):
                clim = s.groupby("mon")["flow_taf"].mean().reindex(wy)
                ax.plot(range(12), clim.to_numpy(), color=_ANCHOR_STYLE[src][0],
                        lw=2.2 if src == "calsim3" else 1.4, marker="o", ms=3, label=slabel)
        ax.set_title(label, fontsize=8); ax.set_ylabel("mean TAF/mo", fontsize=8)
        ax.set_xticks(range(12)); ax.set_xticklabels(labels); ax.tick_params(labelsize=7)
        ax.grid(color="0.93", lw=0.6); ax.set_axisbelow(True)
    axes.ravel()[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(_wrap(f"8 main river indices — mean-monthly climatology, full period "
                       f"(SAC-SMA {climset}, VIC, CalSim unimpaired)", 70), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=_MAP_DPI); plt.close(fig)


# ==========================================================================
# CalSim FLOW-UNIMPAIRED rim systems (reference for the anchor + 8-river hydrographs)
# ==========================================================================
# The basin-level anchor (make_anchor / _anchor_set_taf) uses each rim system's
# FLOW-UNIMPAIRED series as the faithful whole-watershed reference, and the anchor
# hydrographs draw the 8 main river indices.  UNIMP_MAP keeps each system's label + VIC
# inflow-node candidate list.
UNIMP_MAP = {
    "SHAS": {"label": "Shasta (Sac R)",        "15cdec": "SHA", "11obs": "SHA", "vic": ["I_SHSTA"]},
    "SRBB": {"label": "Sac R @ Bend Bridge",   "15cdec": "BND", "11obs": "BND", "vic": ["8RI_SRBB"]},
    "OROV": {"label": "Oroville (Feather)",     "15cdec": "ORO", "11obs": "FTO", "vic": ["8RI_OROVI", "I_OROVL"]},
    "YUBA": {"label": "Yuba",                   "15cdec": "YRS", "11obs": "YRS", "vic": ["8RI_SMART", "I_ENGLB"]},
    "FOLS": {"label": "Folsom (American)",       "15cdec": "FOL", "11obs": "AMF", "vic": ["8RI_FOL_I", "I_FOLSM"]},
    "ST":   {"label": "Stanislaus",             "15cdec": "NML", "11obs": "SNS", "vic": ["8RI_N_MEL", "I_NMELN", "I_MELON"]},
    "TU":   {"label": "Tuolumne",               "15cdec": "TLG", "11obs": "TLG", "vic": ["8RI_DPR_I", "I_NDPR1", "I_DNPDR"]},
    "ME":   {"label": "Merced",                 "15cdec": "MRC", "11obs": "MRC", "vic": ["I_MCLRE", "I_EXCHQ"]},
    "SJ":   {"label": "San Joaquin",            "15cdec": "MIL", "11obs": "SJF", "vic": ["I_MLRTN", "I_MIL003"]},
    "TRIN": {"label": "Trinity",                "15cdec": None,  "11obs": "TNL", "vic": ["I_TRNTY"]},
    "WH":   {"label": "Whiskeytown (Clear Ck)",  "15cdec": None,  "11obs": None,  "vic": ["I_WKYTN"]},
}


def load_unimpaired_monthly(data_dir: str | Path = "data") -> pd.DataFrame:
    """CalSim FLOW-UNIMPAIRED monthly TAF for the 11 rim systems [date, system, flow_taf]."""
    from . import calsim_dir
    return read_table(calsim_dir(data_dir) / "calsim_unimpaired_monthly.csv")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="sacsma.calsim.compare",
        description="Cross-compare CalSim3 (actual) vs VIC vs multi-set SAC-SMA at the CalSim nodes",
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--run", default="compare", help="run name -> artifacts/calsim/<run>/")
    p.add_argument("--sets", nargs="+", default=None,
                   help="SAC-SMA calibration sets to score separately vs CalSim3 "
                        f"(default: {', '.join(DEFAULT_CALSETS)})")
    p.add_argument("--covered-frac", type=float, default=None,
                   help="informational 'covered'/'partial' status label only "
                        "(default: calsim.COVERED_FRAC); node inclusion is crosswalk-driven")
    p.add_argument("--mass-balance", action="store_true",
                   help="apply CalSim's proportional sub-arc (anchor mass-balance) adjustment "
                        "to the per-catchment estimates (does not improve per-catchment skill)")
    p.add_argument("--parallel", action="store_true",
                   help="fan the SAC-SMA model runs across cores (Numba prange); "
                        "results unchanged, ~6-8x faster on the model-run phase")
    args = p.parse_args(argv)
    sets = tuple(args.sets) if args.sets else DEFAULT_CALSETS
    make_all(args.data_dir, args.artifacts_dir, args.run, sets, covered_frac=args.covered_frac,
             mass_balance=args.mass_balance, parallel=args.parallel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
