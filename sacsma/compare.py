"""Cross-compare: CalSim3 (actual) vs VIC vs multi-set SAC-SMA at the CalSim nodes.

One consolidated benchmark.  CalSim3 historical ``INFLOW`` is the **reference**
("truth"); the **VIC** routed historical and **each SAC-SMA calibration set**
(15cdec / 9unimp / 11obs — kept SEPARATE) are scored against it, so the question
is how the SAC-SMA sets stack up against VIC and against each other when all are
measured against the same CalSim3 target.

Monthly inflow per CalSim node (TAF/month):
  * **calsim3** — CalSim3 historical ``INFLOW`` (``data/reference/calsim3_inflow_monthly.csv``) — REFERENCE.
  * **<set>**   — a SAC-SMA calibration set's CalSim run (``calsim.run_calsim`` live,
    daily local-runoff cfs aggregated to monthly volume), one ``source`` per set.
  * **vic**     — VIC routed historical (``data/reference/vic_routed_monthly.csv``).

Nodes are matched on the CalSim arc id ``I_<node>``; VIC names are resolved
through the hand-edited ``data/reference/calsim_crosswalk.csv`` (its ``vic_basin``
column).  Each candidate is scored over its overlap with CalSim3, and the
best-performing calibration set per node is reported.  Artifacts -> ``artifacts/calsim/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_calsim3_monthly, load_vic_monthly, read_table
from .metrics import kge, nse, pbias, pearson

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
    d = pd.read_csv(Path(data_dir) / "reference" / "calsim_crosswalk.csv")
    m = dict(zip(d["arc"].astype(str), d["vic_basin"].astype(str)))
    return {k: v for k, v in m.items() if v and v != "nan"}


# --------------------------------------------------------------------------
# Cross-compare: each calibration set kept SEPARATE, scored vs CalSim3
# --------------------------------------------------------------------------
#: default calibration sets fed to the combined CalSim check (unimpaired + observed).
DEFAULT_CALSETS = ("15cdec", "9unimp", "11obs")


def _calset_monthly_taf(domain: str, data_dir: str | Path = "data", *, covered_frac=None):
    """Per-catchment monthly TAF for the catchments this set's HRUs **own** and score.

    Scored catchments are the basin -> node mapping (:func:`load_basin_nodes`) restricted
    to real CalSim3 inflow nodes the set's HRUs actually cover, and with the **cumulative
    single-node systems excluded** (Merced ``I_MCLRE``, San Joaquin ``I_MLRTN``, Shasta,
    Trinity) — those are whole-basin nodes scored only in the basin-level anchor view, not
    per local catchment.  Returns ``(node_monthly, scored)``: ``node_monthly`` is monthly
    TAF per node; ``scored`` carries ``[set, cid, node, arc, basin, kind, cov_frac]`` for
    the maps.  ``covered_frac`` only affects the internal :func:`run_calsim` weighting.
    """
    from .calsim import COVERED_FRAC, MERGED_LAYER, run_calsim, series_arc

    cf = COVERED_FRAC if covered_frac is None else covered_frac
    # the MERGED layer makes the cumulative single-node systems whole catchments, so they
    # are scored as one piece (Merced runoff vs I_MCLRE) instead of a sliver / grey hole.
    flows, cov, _map = run_calsim(data_dir, domain=domain, layer=MERGED_LAYER, covered_frac=cf)
    flows["arc"] = flows["node"].map(series_arc)
    nodes = load_basin_nodes(data_dir, domain)
    scored = nodes[nodes["in_calsim3"].astype(bool)].copy()   # cumulative now whole -> kept
    scored_arcs = set(scored["arc"])
    # per-arc monthly TAF over the scored catchments (join on arc -> layer-robust)
    f = flows[flows["arc"].isin(scored_arcs)].drop_duplicates(["arc", "date"]).copy()
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
    from .calsim import BASIN_RIM_SYSTEM, load_crosswalk

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
    anchor_long=None, mass_balance=False
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Long [date, arc, node, source, flow_taf] for each calibration set + CalSim3 + VIC.

    ``source`` is the calibration set name, ``calsim3`` (the actual), or ``vic``.
    Restricted to the common period across all present sources.  Also returns the
    stacked per-set coverage table.  ``covered_frac`` overrides the "covered"
    threshold (None -> ``calsim.COVERED_FRAC``).  When ``anchor_long`` is provided, the
    **proportional sub-arc (anchor mass-balance) adjustment** is applied to every estimate
    (SAC sets + VIC) within the distributed rim systems (:func:`_apply_anchor_mass_balance`).
    """
    per = [_calset_monthly_taf(d, data_dir, covered_frac=covered_frac) for d in sets]
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
    ratio_clip=(0.1, 10.0),
) -> pd.DataFrame:
    """Per-sub-arc bias-correction **validation** (train/test split) + anchor mass-balance,
    applied to **every multi-arc basin** (≥2 in_calsim3 sub-arcs) in the crosswalk — the 6
    distributed rim systems *and* the multi-arc secondary basins (15cdec MKM, 11obs BLB, the
    9unimp creeks Mokelumne/Bear/Cache/Cosumnes/Stony) — for each SAC set AND VIC.

    For each sub-arc a multiplicative **monthly mean-ratio** correction (mean CalSim3 / mean
    estimate, by calendar month) is learned on the ``train`` water years and applied on the
    held-out ``test`` years (a non-QMAP per-sub-arc bias correction).  The corrected sub-arcs
    are then proportionally renormalized so each basin still sums to that estimate's anchor
    total (``run_basin`` for SAC, the basin's VIC total for VIC — both from ``anchor_long``) —
    "faithful sub-arcs without violating the anchor totals".  VIC is corrected once per arc
    (deduplicated across sets).  Returns per (set, arc) raw-vs-corrected KGE/NSE/pbias on the
    **test** period only.
    """
    from collections import defaultdict

    from .calsim import derive_basin_nodes

    L = raw_long if raw_long is not None else build_calsets_long(data_dir, sets)[0]
    tr0, tr1 = pd.Timestamp(train[0]), pd.Timestamp(train[1])
    te0, te1 = pd.Timestamp(test[0]), pd.Timestamp(test[1])
    lo, hi = ratio_clip

    c3 = L[L["source"] == "calsim3"].pivot_table(index="date", columns="arc", values="flow_taf")
    Evic = L[L["source"] == "vic"].pivot_table(index="date", columns="arc", values="flow_taf")

    def correct(E, arcs):
        """Per-sub-arc monthly mean-ratio correction (learned on the train years), renormalized
        so the group still sums to its OWN raw sub-arc total each month — preserving the
        estimate's per-catchment basin total ("don't violate the anchor total").  Using the
        group's own sub-arc sum (not ``run_basin``) is robust to nesting / partial coverage: it
        does NOT inflate a basin's few local sub-arcs up to a full-watershed total (the Bend
        Bridge bug, where SRBB's small local arcs were rescaled to include Shasta + valley)."""
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
        return corr.mul(factor, axis=0)

    def score_rows(E, Ec, arcs, set_label, basin):
        te = (E.index >= te0) & (E.index <= te1)
        out = []
        for a in arcs:
            if a not in c3.columns:
                continue
            rr = c3[a]
            m = te & E[a].notna() & rr.notna()
            if int(m.sum()) < 12:
                continue
            er, cr, ac = E[a][m].to_numpy(), Ec[a][m].to_numpy(), rr[m].to_numpy()
            out.append({"set": set_label, "arc": str(a), "node": str(a)[2:], "basin": basin,
                        "n_test": int(m.sum()),
                        "kge_raw": kge(er, ac), "nse_raw": nse(er, ac), "pbias_raw": pbias(er, ac),
                        "kge_corr": kge(cr, ac), "nse_corr": nse(cr, ac), "pbias_corr": pbias(cr, ac),
                        # signed mean-monthly volume bias (TAF), raw vs corrected
                        "tafbias_raw": float(er.mean() - ac.mean()),
                        "tafbias_corr": float(cr.mean() - ac.mean())})
        return out

    rows: list = []
    vic_seen: set = set()
    for st in sets:
        nodes = derive_basin_nodes(data_dir, st)
        nodes = nodes[nodes["in_calsim3"].astype(bool)]
        arc2basin = {str(a): b for a, b in zip(nodes["arc"], nodes["basin"])}
        Eset = L[L["source"] == st].pivot_table(index="date", columns="arc", values="flow_taf")
        byb: dict[str, list] = defaultdict(list)
        for a in Eset.columns:
            b = arc2basin.get(str(a))
            if b is not None and a in c3.columns:
                byb[b].append(a)
        for basin, arcs in byb.items():
            if len(arcs) < 2:               # multi-arc basins only
                continue
            cb = correct(Eset, arcs)
            rows += score_rows(Eset, cb, arcs, st, basin)
            # VIC under this basin grouping, deduplicated across sets (rim arcs are shared)
            varcs = [a for a in arcs if a in Evic.columns and a not in vic_seen]
            if len(varcs) >= 2:
                cbv = correct(Evic, varcs)
                rows += score_rows(Evic, cbv, varcs, "vic", basin)
                vic_seen.update(varcs)
    return pd.DataFrame(rows)


def make_subarc_validation(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts",
                           run: str = "calsim", sets=DEFAULT_CALSETS, *,
                           anchor_long=None, raw_long=None, met=None) -> Path:
    """Write the per-sub-arc bias-correction validation (train/test) under ``artifacts/<run>/``:
    ``subarc_validation_metrics.csv`` + a corrected best-of NSE coverage map.
    ``met`` may be passed in (already computed by :func:`make_all`) to avoid recompute."""
    out = Path(artifacts_dir) / run
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if met is None:
        met = subarc_validation_metrics(data_dir, sets, anchor_long=anchor_long, raw_long=raw_long)
    met.to_csv(out / "subarc_validation_metrics.csv", index=False)
    for s in list(sets) + ["vic"]:
        d = met[met["set"] == s]
        if len(d):
            print(f"subarc-validation [{s}]: n={len(d)} test sub-arcs | "
                  f"median KGE {d['kge_raw'].median():.3f}->{d['kge_corr'].median():.3f}  "
                  f"NSE {d['nse_raw'].median():.3f}->{d['nse_corr'].median():.3f}  "
                  f"|pbias| {d['pbias_raw'].abs().median():.1f}->{d['pbias_corr'].abs().median():.1f}%")
    _subarc_corrected_map(data_dir, met, sets, out / "figures" / "subarc_validation_map.png")
    return out


def _subarc_corrected_map(data_dir, met, sets, path):
    """Choropleth of the best-of **corrected** test-period NSE across sets (max over sets)."""
    from .calsim import MERGED_LAYER, load_catchments, series_arc

    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).copy()
    catch["arc"] = catch["node"].map(series_arc)
    best = met[met["set"].isin(sets)].dropna(subset=["nse_corr"]).groupby("arc")["nse_corr"].max()
    covered = catch[catch["arc"].isin(best.index)].copy()
    covered["nse"] = covered["arc"].map(best)
    _nse_choropleth(catch, covered,
                    f"Sub-arc bias-corrected (train/test) NSE vs CalSim3 — best over {', '.join(sets)} "
                    f"({len(covered)} sub-arcs, test period)",
                    "corrected monthly NSE vs CalSim3 (test)", path, extent=_map_extent(data_dir))


def _bestof_minus_vic(met, *, corr=False):
    """Per-arc (best-of-set NSE − VIC NSE) over arcs both cover.  ``corr`` uses the
    bias-corrected validation columns (``nse_corr``)."""
    col = "nse_corr" if corr else "nse"
    sets = [s for s in met["set"].unique() if s != "vic"]
    sr = met[met["set"].isin(sets)].dropna(subset=[col])
    best = sr.groupby("arc")[col].max()
    vic = met[met["set"] == "vic"].dropna(subset=[col]).set_index("arc")[col]
    common = best.index.intersection(vic.index)
    return (best.loc[common] - vic.loc[common]).rename("nse")


def _diff_choropleth(data_dir, diff_by_arc, title, path, *, vmin=-0.2, vmax=0.2,
                     cb_label="NSE difference (best-of − VIC)"):
    """Diverging choropleth of a per-arc difference (e.g. best-of − VIC), green = SAC better."""
    from .calsim import MERGED_LAYER, load_catchments, series_arc

    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).copy()
    catch["arc"] = catch["node"].map(series_arc)
    covered = catch[catch["arc"].isin(diff_by_arc.index)].copy()
    covered["nse"] = covered["arc"].map(diff_by_arc)
    _nse_choropleth(catch, covered, title, cb_label, path,
                    extent=_map_extent(data_dir), cmap_name="RdYlGn", vmin=vmin, vmax=vmax)


def _pbias_improvement(subarc_met, sets):
    """Per-arc change in |percent bias| from the sub-arc adjustment, best-of over ``sets``
    (the kge_corr-best set per arc): ``|pbias_raw| − |pbias_corr|`` on the held-out test
    period.  **Positive = improvement** (bias moved toward CalSim3); negative = worse."""
    sr = subarc_met[subarc_met["set"].isin(sets)].dropna(subset=["kge_corr"])
    best = sr.loc[sr.groupby("arc")["kge_corr"].idxmax()].set_index("arc")
    return (best["pbias_raw"].abs() - best["pbias_corr"].abs()).rename("nse")


def _cdf_vic_bestof_fig(met_full, subarc_met, sets, path):
    """CDF of NSE / KGE / pbias over arcs common to VIC and the SAC best-of: best-of vs VIC,
    **dashed = full period, solid = validation (post sub-arc adjustment)**."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def bestof(df, cols, by):                      # KGE-best set per arc, then take cols
        sr = df[df["set"].isin(sets)].dropna(subset=[by])
        return sr.loc[sr.groupby("arc")[by].idxmax()].set_index("arc")[cols]

    raw = ["kge", "nse", "pbias"]
    cor = ["kge_corr", "nse_corr", "pbias_corr"]
    ren = dict(zip(cor, raw))
    bof_full = bestof(met_full, raw, "kge")
    vic_full = met_full[met_full["set"] == "vic"].set_index("arc")[raw]
    bof_val = bestof(subarc_met, cor, "kge_corr").rename(columns=ren)
    vic_val = subarc_met[subarc_met["set"] == "vic"].set_index("arc")[cor].rename(columns=ren)
    common = bof_full.index.intersection(vic_full.index).intersection(
        bof_val.index).intersection(vic_val.index)

    def cdf(ax, vals, **kw):
        v = np.sort(np.asarray(vals, float)); v = v[np.isfinite(v)]
        if len(v):
            ax.plot(v, np.linspace(0, 1, len(v)), **kw)

    fig, axes = plt.subplots(1, 3, figsize=(_MAP_W, 2.7))
    for ax, (m, lab, xlim) in zip(axes, [("nse", "NSE", (-1, 1)), ("kge", "KGE", (-1, 1)),
                                         ("pbias", "pbias (%)", (-100, 100))]):
        cdf(ax, bof_full.loc[common, m], color=_COLORS["calsim3"], ls="--", lw=1.2)
        cdf(ax, bof_val.loc[common, m], color=_COLORS["calsim3"], ls="-", lw=1.5)
        cdf(ax, vic_full.loc[common, m], color=_COLORS["vic"], ls="--", lw=1.2)
        cdf(ax, vic_val.loc[common, m], color=_COLORS["vic"], ls="-", lw=1.5)
        ax.set_xlim(*xlim); ax.set_ylim(0, 1)
        ax.set_xlabel(lab, fontsize=8); ax.tick_params(labelsize=7)
        ax.grid(color="0.93", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("CDF", fontsize=8)
    h = [Line2D([], [], color=_COLORS["calsim3"], label="best-of"),
         Line2D([], [], color=_COLORS["vic"], label="VIC"),
         Line2D([], [], color="0.4", ls="--", label="full period"),
         Line2D([], [], color="0.4", ls="-", label="validation (post-adj)")]
    axes[0].legend(handles=h, fontsize=6, loc="upper right")
    fig.suptitle(f"VIC vs SAC best-of — CDF over {len(common)} common arcs", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=_MAP_DPI)
    plt.close(fig)


def _cdf_taf_bias_fig(met_full, subarc_met, sets, path):
    """CDF of the **signed mean-monthly volume bias (TAF)** over arcs common to VIC and the SAC
    best-of: best-of vs VIC, **dashed = full period, solid = validation (post sub-arc adj)**.
    Shows under- (negative) and over-prediction (positive) in actual acre-feet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mf = met_full.assign(bias=met_full["mean_set_taf"] - met_full["mean_calsim3_taf"])
    sr = mf[mf["set"].isin(sets)].dropna(subset=["kge"])
    bof_full = sr.loc[sr.groupby("arc")["kge"].idxmax()].set_index("arc")["bias"]
    vic_full = mf[mf["set"] == "vic"].set_index("arc")["bias"]
    sv = subarc_met[subarc_met["set"].isin(sets)].dropna(subset=["kge_corr"])
    bof_val = sv.loc[sv.groupby("arc")["kge_corr"].idxmax()].set_index("arc")["tafbias_corr"]
    vic_val = subarc_met[subarc_met["set"] == "vic"].set_index("arc")["tafbias_corr"]
    common = bof_full.index.intersection(vic_full.index).intersection(
        bof_val.index).intersection(vic_val.index)

    def cdf(ax, vals, **kw):
        v = np.sort(np.asarray(vals, float)); v = v[np.isfinite(v)]
        if len(v):
            ax.plot(v, np.linspace(0, 1, len(v)), **kw)

    fig, ax = plt.subplots(figsize=(_MAP_W, 3.2))
    cdf(ax, bof_full.loc[common], color=_COLORS["calsim3"], ls="--", lw=1.2)
    cdf(ax, bof_val.loc[common], color=_COLORS["calsim3"], ls="-", lw=1.5)
    cdf(ax, vic_full.loc[common], color=_COLORS["vic"], ls="--", lw=1.2)
    cdf(ax, vic_val.loc[common], color=_COLORS["vic"], ls="-", lw=1.5)
    lim = float(np.nanpercentile(np.abs(np.concatenate(
        [bof_full.loc[common], vic_full.loc[common]])), 97))
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(0, 1)
    ax.set_xlabel("mean-monthly volume bias (TAF)  [neg = under, pos = over]", fontsize=8)
    ax.set_ylabel("CDF", fontsize=8); ax.tick_params(labelsize=7)
    ax.grid(color="0.93", lw=0.5); ax.set_axisbelow(True)
    h = [Line2D([], [], color=_COLORS["calsim3"], label="best-of"),
         Line2D([], [], color=_COLORS["vic"], label="VIC"),
         Line2D([], [], color="0.4", ls="--", label="full period"),
         Line2D([], [], color="0.4", ls="-", label="validation (post-adj)")]
    ax.legend(handles=h, fontsize=7, loc="lower right")
    ax.set_title(f"Actual TAF bias CDF — VIC vs SAC best-of over {len(common)} common arcs",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=_MAP_DPI)
    plt.close(fig)


def build_crosswalk(data_dir: str | Path = "data", sets=DEFAULT_CALSETS, *,
                    force: bool = False) -> pd.DataFrame:
    """**Bootstrap** the master crosswalk aligning every CalSim inflow arc across all
    comparisons — run ONCE, then hand-edit ``data/reference/calsim_crosswalk.csv``.

    Columns ``[arc, system, unimp_anchor, vic_basin, basin_15cdec, basin_11obs,
    basin_9unimp, in_calsim3]``:

    * ``system`` / ``unimp_anchor`` — the RimInflowAnchor rim system (blank = non-rim).
    * ``vic_basin`` — the VIC major-basin series the arc rolls up to (the rim system's
      8-River anchor, or the arc's own VIC node for non-rim catchments).
    * ``basin_<set>`` — the **primary** owning basin in each SAC-SMA set (the basin whose
      rim system *is* the arc's home system, so nesting parents like Bend Bridge don't
      shadow Shasta; secondary arcs get their geographic owner).

    This is the SINGLE source of truth going forward.  To protect manual edits it
    **refuses to overwrite** an existing file unless ``force=True``; it reads the raw
    ``calsim_rim_anchor.csv`` + ``calsim_vic_name_mapping.csv`` (the bootstrap inputs).
    """
    out = Path(data_dir) / "reference" / "calsim_crosswalk.csv"
    if out.exists() and not force:
        print(f"crosswalk: {out} exists (hand-edited) — not overwriting (use force=True to rebuild)")
        return pd.read_csv(out)

    from .calsim import _bootstrap_geographic_nodes, load_rim_anchor

    anc = load_rim_anchor(data_dir)
    home = dict(zip(anc["arc"].astype(str), anc["system"]))
    c3 = load_calsim3_monthly(data_dir)
    nz = c3.groupby("arc")["flow_taf"].max()
    c3_arcs = set(nz[nz > 0].index.astype(str))
    vavail = set(load_vic_monthly(data_dir)["vic_name"])
    sys_vic = {s: next((n for n in m["vic"] if n in vavail), None) for s, m in UNIMP_MAP.items()}

    def primary_owner(domain: str) -> dict[str, str]:
        df = _bootstrap_geographic_nodes(data_dir, domain, calsim3_arcs=c3_arcs)
        out = {}
        for arc, g in df.groupby("arc"):
            if len(g) == 1:
                out[str(arc)] = g["basin"].iloc[0]
            else:  # shared (nested) arc -> the basin whose system is the arc's home
                cand = g[g["system"] == home.get(str(arc))]
                out[str(arc)] = (cand if len(cand) else g)["basin"].iloc[0]
        return out

    owners = {dom: primary_owner(dom) for dom in ("15cdec", "11obs", "9unimp")}
    rows = []
    for _, r in anc.iterrows():
        arc, sysn = str(r["arc"]), r["system"]
        vic = sys_vic.get(sysn) if pd.notna(sysn) else (arc if arc in vavail else None)
        rows.append({
            "arc": arc, "system": sysn, "unimp_anchor": r["unimp_anchor"], "vic_basin": vic,
            "basin_15cdec": owners["15cdec"].get(arc), "basin_11obs": owners["11obs"].get(arc),
            "basin_9unimp": owners["9unimp"].get(arc), "in_calsim3": arc in c3_arcs,
        })
    cw = pd.DataFrame(rows).sort_values("arc").reset_index(drop=True)
    out = Path(data_dir) / "reference" / "calsim_crosswalk.csv"
    cw.to_csv(out, index=False)
    print(f"crosswalk: {len(cw)} arcs ({cw[['basin_15cdec','basin_11obs','basin_9unimp']].notna().any(axis=1).sum()} "
          f"mapped to a set basin, {cw['vic_basin'].notna().sum()} to VIC) -> {out}")
    return cw


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
    """Per (arc, candidate) skill vs CalSim3 over the common period."""
    rows = []
    for arc in matched:
        sub = long[long["arc"] == arc]
        wide = sub.pivot_table(index="date", columns="source", values="flow_taf")
        if "calsim3" not in wide:
            continue
        node = sub["node"].iloc[0]
        for cand in candidates:
            if cand not in wide:
                continue
            d = wide[[cand, "calsim3"]].dropna()
            if len(d) < 12:
                continue
            sim, ref = d[cand].to_numpy(), d["calsim3"].to_numpy()
            rows.append({
                "arc": arc, "node": node, "set": cand, "n_months": len(d),
                "kge": kge(sim, ref), "nse": nse(sim, ref), "pbias": pbias(sim, ref),
                "r": pearson(sim, ref), "mean_set_taf": float(sim.mean()),
                "mean_calsim3_taf": float(ref.mean()),
            })
    return pd.DataFrame(rows)


def make_all(
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
    run: str = "calsim",
    sets=DEFAULT_CALSETS,
    covered_frac=None,
    mass_balance=False,
) -> Path:
    """Cross-compare each calibration set + VIC vs CalSim3; best-of set per node.

    ``mass_balance`` (default off) applies CalSim's proportional sub-arc (anchor
    mass-balance) adjustment to every estimate's per-catchment series before scoring
    (:func:`_apply_anchor_mass_balance`).  It is off by default because it does NOT improve
    per-catchment skill — our per-catchment error is the spatial split among a system's
    sub-arcs, which a single per-system rescale cannot correct."""
    out = Path(artifacts_dir) / run
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # the basin anchors (area-nudged run_basin totals) drive BOTH the proportional sub-arc
    # mass-balance in the per-catchment view and the basin-level anchor view (built once).
    anchor_long = build_anchor_long(data_dir, sets)
    long, matched, coverage = build_calsets_long(data_dir, sets, covered_frac=covered_frac,
                                                 anchor_long=anchor_long, mass_balance=mass_balance)
    candidates = list(sets) + ["vic"]
    met = calset_metrics(long, matched, candidates)
    # attach each set's honest HRU coverage of the node, so a low-coverage (extrapolated)
    # per-node score is visible in the metrics CSV (NaN for vic — not HRU-based).
    cov_key = coverage.drop_duplicates(["set", "arc"]).set_index(["set", "arc"])
    met = met.merge(cov_key[["cov_frac", "n_hru"]], left_on=["set", "arc"],
                    right_index=True, how="left")

    # best calibration set per node (exclude vic from the "best set" pick)
    setrows = met[met["set"].isin(sets)].dropna(subset=["kge"])
    idx = setrows.groupby("arc")["kge"].idxmax()
    best = setrows.loc[idx].sort_values("kge", ascending=False).reset_index(drop=True)

    vic_full = vic_full_metrics(data_dir)
    long.to_csv(out / "monthly_calsets.csv", index=False)
    met.to_csv(out / "calset_metrics.csv", index=False)
    best.to_csv(out / "best_of_set.csv", index=False)
    vic_full.to_csv(out / "vic_full_metrics.csv", index=False)
    coverage.sort_values(["set", "basin", "cov_frac"], ascending=[True, True, False]).to_csv(
        out / "coverage_by_set.csv", index=False)

    period = f"{long['date'].min().date()}..{long['date'].max().date()}"
    msg = "  ".join(f"{s}={met[met['set']==s]['kge'].median():.2f}"
                    for s in candidates if (met["set"] == s).any())
    print(f"compare: {len(matched)} nodes vs CalSim3 over {period}; "
          f"median KGE [{msg}]; best-of covers {best['arc'].nunique()} nodes -> {out}")

    # per-set per-catchment NSE map (with subsystem outlines) + per-set skill vs CalSim3
    for s in sets:
        _calset_coverage_map(data_dir, s, coverage, met, out / "figures" / f"{s}_coverage_map.png")
        _calset_skill_fig(met, s, out / "figures" / f"{s}_skill.png")
    _vic_coverage_map(data_dir, out / "figures" / "vic_coverage_map.png")
    _calsets_bestof_fig(best, sets, out / "figures" / "calsets_bestof.png")
    _bestof_coverage_map(data_dir, met, sets, out / "figures" / "calsets_bestof_map.png")
    # basin-level (anchor) comparison: each basin vs the sum of its sub-nodes
    make_anchor(data_dir, artifacts_dir, run, sets, anchor_long=anchor_long)
    # per-sub-arc bias-correction validation (train/test) + anchor mass-balance, SAC sets + VIC
    subarc_met = subarc_validation_metrics(data_dir, sets, anchor_long=anchor_long,
                                           raw_long=(None if mass_balance else long))
    make_subarc_validation(data_dir, artifacts_dir, run, sets, met=subarc_met)

    # best-of − VIC NSE difference maps (full period + post-adjustment validation), ±0.2 scale
    _diff_choropleth(data_dir, _bestof_minus_vic(met),
                     "Best-of − VIC monthly NSE vs CalSim3 (full period; green = SAC better)",
                     out / "figures" / "diff_bestof_vic_full.png")
    _diff_choropleth(data_dir, _bestof_minus_vic(subarc_met, corr=True),
                     "Best-of − VIC NSE vs CalSim3 (validation, post sub-arc adj; green = SAC better)",
                     out / "figures" / "diff_bestof_vic_validation.png")
    # sub-arc adjustment effect on |percent bias|: positive (green) = bias moved toward CalSim3
    _diff_choropleth(data_dir, _pbias_improvement(subarc_met, sets),
                     "Sub-arc adjustment |pbias| improvement vs CalSim3 "
                     "(best-of, validation; green = closer to truth)",
                     out / "figures" / "diff_pbias_improvement.png",
                     vmin=-20.0, vmax=20.0,
                     cb_label="|pbias| improvement (%)  [+ = closer to CalSim3]")
    # CDF over VIC∩best-of common arcs: best-of vs VIC, dashed=full / solid=validation
    _cdf_vic_bestof_fig(met, subarc_met, sets, out / "figures" / "cdf_vic_bestof.png")
    _cdf_taf_bias_fig(met, subarc_met, sets, out / "figures" / "cdf_taf_bias.png")
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
    from .calsim import MERGED_LAYER, load_catchments

    b = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).total_bounds
    px, py = 0.15 * (b[2] - b[0]) / 10, 0.15 * (b[3] - b[1]) / 10
    return (b[0] - px, b[2] + px, b[1] - py, b[3] + py)


def _nse_choropleth(catch, covered_gdf, title, cb_label, path, *, cells=None,
                    cells_label=None, subsystems=None, extent=None,
                    cmap_name="RdYlGn", vmin=0.0, vmax=1.0):
    """Draw a CalSim Rim map: ``covered_gdf`` (has an ``nse`` column) as a 0..1 NSE
    choropleth on top of all catchments in faint grey; optionally overlay HRU cells
    and thick **subsystem outlines** (``subsystems`` = a GeoSeries of basin unions).

    The NSE colour scale is fixed **0..1** for every map (negative NSE clamps to the
    floor).  Catchments not scored are drawn once as faint "not covered" context.
    """
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    cmap = plt.get_cmap(cmap_name)
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


def _calset_coverage_map(data_dir, set_name, coverage, met, path):
    """Per-set map: scored CalSim3 catchments (the **merged** whole-basin layer) shaded by
    that set's per-node NSE vs CalSim3 (0..1) on faint context, with a clean coloured
    outline of each basin's **HRU footprint** (the watershed it represents) + HRU cells.
    The merged layer fills the cumulative single-node basins (Merced/Shasta/SJ) as whole
    catchments rather than leaving grey holes."""
    from .calsim import MERGED_LAYER, basin_footprints, load_catchments, load_hru_cells, series_arc

    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).copy()
    catch["arc"] = catch["node"].map(series_arc)
    cov_s = coverage[coverage["set"] == set_name]
    nse_by_arc = (met[met["set"] == set_name].dropna(subset=["nse"])
                  .assign(arc=lambda d: "I_" + d["node"].astype(str))
                  .set_index("arc")["nse"].to_dict())
    scored = catch.merge(cov_s[["arc", "basin"]].drop_duplicates("arc"), on="arc", how="inner").copy()
    scored["nse"] = scored["arc"].map(nse_by_arc)
    cells = load_hru_cells(data_dir, domain=set_name)
    footprints = basin_footprints(data_dir, set_name)
    _nse_choropleth(catch, scored,
                    f"{set_name}: CalSim3 sub-node NSE + basin watershed outlines ({set_name} HRUs)",
                    "monthly NSE vs CalSim3 (scored catchments)", path,
                    cells=cells, cells_label=f"{set_name} HRU cells ({len(cells)})",
                    subsystems=footprints, extent=_map_extent(data_dir))


def _vic_coverage_map(data_dir, path):
    """Standalone map: **every** Rim catchment VIC has a CalSim3-comparable series for,
    shaded by VIC's NSE vs CalSim3 (0..1) — independent of any SAC-SMA set's coverage
    (so San Luis and the other valley/westside VIC nodes appear)."""
    from .calsim import MERGED_LAYER, load_catchments, series_arc

    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).copy()
    catch["sarc"] = catch["node"].map(series_arc)   # series arc per polygon (alias-aware)
    nse_by = vic_full_metrics(data_dir).dropna(subset=["nse"]).set_index("arc")["nse"].to_dict()
    covered = catch[catch["sarc"].isin(nse_by)].copy()
    covered["nse"] = covered["sarc"].map(nse_by)
    _nse_choropleth(catch, covered,
                    f"VIC vs CalSim3 — monthly NSE over all {len(covered)} VIC-covered Rim catchments",
                    "monthly NSE vs CalSim3 (VIC)", path, extent=_map_extent(data_dir))


def _calset_skill_fig(met, set_name, path):
    """Per-set skill on the arcs *this set* covers: set vs CalSim3, with VIC alongside."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = met[met["set"] == set_name].set_index("node")
    if s.empty:
        return
    order = s.sort_values("kge", ascending=False).index
    v = met[met["set"] == "vic"].set_index("node").reindex(order)

    fig, ax = plt.subplots(figsize=(_MAP_W, 4.0))
    x = np.arange(len(order))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.scatter(x, s.reindex(order)["kge"], s=10, alpha=0.85,
               color=_COLORS.get(set_name, "k"),
               label=f"{set_name} (med {s['kge'].median():.2f})")
    if v["kge"].notna().any():
        ax.scatter(x, v["kge"], s=10, alpha=0.7, color=_COLORS["vic"], marker="^",
                   label=f"VIC (med {v['kge'].median():.2f})")
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f"CalSim node covered by {set_name} (sorted by {set_name} KGE)", fontsize=8)
    ax.set_ylabel("monthly KGE vs CalSim3", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{set_name} vs CalSim3 (with VIC) on {len(order)} {set_name}-covered nodes", fontsize=8)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=_MAP_DPI)
    plt.close(fig)


def _bestof_coverage_map(data_dir, met, sets, path):
    """Choropleth of the **best achievable** monthly NSE vs CalSim3 across all calibration
    sets (max over sets per catchment), on the merged whole-basin layer — the spatial
    companion to ``best_of_set.csv`` / ``calsets_bestof.png``."""
    from .calsim import MERGED_LAYER, load_catchments, load_hru_cells, series_arc

    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).copy()
    catch["arc"] = catch["node"].map(series_arc)
    sr = met[met["set"].isin(sets)].dropna(subset=["nse"])
    best_nse = sr.groupby("arc")["nse"].max()
    covered = catch[catch["arc"].isin(best_nse.index)].copy()
    covered["nse"] = covered["arc"].map(best_nse)
    # HRU cells pooled across all sets (light context for the footprint extent)
    cells = pd.concat([load_hru_cells(data_dir, domain=s) for s in sets],
                      ignore_index=True).drop_duplicates("key")
    _nse_choropleth(catch, covered,
                    f"Best-of-set NSE vs CalSim3 (max over {', '.join(sets)}) — "
                    f"{len(covered)} catchments",
                    "best-of-set monthly NSE vs CalSim3", path,
                    cells=cells, cells_label=f"HRU cells (all sets, {len(cells)})",
                    extent=_map_extent(data_dir))


def _calsets_bestof_fig(best, sets, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(_MAP_W, 3.2))

    counts = best["set"].value_counts().reindex(list(sets)).fillna(0)
    ax[0].bar(counts.index, counts.values, color=[_COLORS.get(s, "k") for s in counts.index])
    ax[0].set_ylabel("nodes where this set is best", fontsize=8)
    ax[0].set_title(f"Best set per node (n={len(best)})", fontsize=8)
    ax[0].tick_params(labelsize=7)
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, int(v), ha="center", va="bottom", fontsize=7)

    ax[1].axhline(best["kge"].median(), color="0.7", ls="--", lw=0.8,
                  label=f"median {best['kge'].median():.2f}")
    bs = best.sort_values("kge", ascending=False).reset_index(drop=True)
    ax[1].scatter(np.arange(len(bs)), bs["kge"],
                  c=[_COLORS.get(s, "k") for s in bs["set"]], s=10)
    ax[1].set_ylim(-1, 1)
    ax[1].set_xlabel("CalSim node (sorted)", fontsize=8)
    ax[1].set_ylabel("best-of-set monthly KGE", fontsize=8)
    ax[1].set_title("Best achievable skill per node", fontsize=8)
    ax[1].tick_params(labelsize=7)
    ax[1].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=_MAP_DPI)
    plt.close(fig)


# ==========================================================================
# Basin-level anchor comparison (geographic winner-take-all sub-systems)
# ==========================================================================
# Each calibration-set basin is compared to the SUM of the CalSim3 rim-inflow nodes it
# comprises, per the hand-edited master crosswalk (data/reference/calsim_crosswalk.csv,
# projected by calsim.derive_basin_nodes).  This is the authoritative "anchor" view: the
# basin's total inflow vs CalSim's, rather than the per-catchment spatial view.
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
    from .calsim import derive_basin_nodes
    return derive_basin_nodes(data_dir, domain)


def load_anchor_area_scale(data_dir: str | Path = "data") -> dict[tuple[str, str], float]:
    """Per (set, basin) **anchor-only** area scale -> ``{(set, basin): scale}``.

    Empty if ``data/reference/anchor_area_scale.csv`` is absent.  ``scale =
    area_after_mi2 / area_before_mi2`` so hand-editing ``area_after_mi2`` just works.
    This nudges *only* the anchor volume conversion (not the gage cal/val or per-catchment
    views, which keep the true published areas)."""
    p = Path(data_dir) / "reference" / "anchor_area_scale.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    return {(str(r["set"]), str(r["basin"])): float(r["area_after_mi2"]) / float(r["area_before_mi2"])
            for _, r in d.iterrows() if float(r["area_before_mi2"]) > 0}


def _anchor_set_taf(domain, data_dir, nodes, forcing=None, *, apply_scale=True):
    """Per basin monthly TAF for one set: SAC-SMA basin run + CalSim3 reference + VIC.

    The CalSim3 **reference** is chosen per basin (``ref_kind``):

    - ``unimp`` — if the basin maps to a CalSim rim system (``BASIN_RIM_SYSTEM``)
      that has a ``FLOW-UNIMPAIRED`` series, that single whole-watershed series is
      the faithful reference.  This is the only correct target for systems like Sac @
      Bend Bridge, whose flow includes valley-floor/local accretion that the sum of
      individual INFLOW sub-arcs does **not** capture (~12% low for SRBB).
    - ``inflow_sum`` — otherwise (creeks, secondary basins with no aggregate series),
      sum the basin's assigned CalSim3 INFLOW sub-arcs.

    When ``apply_scale`` (default), the SAC-SMA volume is multiplied by the hand-editable
    per-basin :func:`load_anchor_area_scale` factor (an anchor-only area nudge that
    minimises |pbias| within a plausibility cap without degrading KGE/NSE).
    """
    from .calsim import BASIN_RIM_SYSTEM, basin_areas
    from .io import mmday_to_cfs
    from .model import run_basin

    areas = basin_areas(data_dir, domain=domain)
    scales = load_anchor_area_scale(data_dir) if apply_scale else {}
    c3 = load_calsim3_monthly(data_dir)
    vic = load_vic_monthly(data_dir)
    arc2vic = load_name_map(data_dir)   # arc -> VIC major-basin series (crosswalk vic_basin)
    bsys = BASIN_RIM_SYSTEM.get(domain, {})
    unimp_by_sys = {s: g for s, g in load_unimpaired_monthly(data_dir).groupby("system")}
    summable = nodes[nodes["in_calsim3"].astype(bool)]
    parts = []
    for basin, g in summable.groupby("basin"):
        df = run_basin(basin, data_dir=data_dir, domain=domain, forcing=forcing)
        s = pd.Series(mmday_to_cfs(df["flow"].to_numpy(), areas[basin]),
                      index=pd.to_datetime(df["date"]))
        sac = _cfs_day_to_taf(s.groupby(s.index.to_period("M")).sum())
        sac.index = sac.index.to_timestamp("M")
        sac = sac * scales.get((domain, basin), 1.0)   # anchor-only area nudge
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
                      *, apply_scale=True) -> pd.DataFrame:
    """Long [date, set, basin, source, flow_taf, ref_kind] for the basin-level anchor.

    ``apply_scale`` toggles the per-basin anchor area nudge (set False to get the raw,
    un-nudged series — used when *computing* the nudge in :func:`compute_anchor_area_scale`)."""
    from .model import load_domain_forcing

    parts = []
    for dom in sets:
        nodes = load_basin_nodes(data_dir, dom)
        forcing = load_domain_forcing(data_dir, domain=dom)
        parts.append(_anchor_set_taf(dom, data_dir, nodes, forcing, apply_scale=apply_scale))
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def compute_anchor_area_scale(data_dir: str | Path = "data", sets=DEFAULT_CALSETS,
                              *, cap: float = 0.10, tol: float = 0.0) -> pd.DataFrame:
    """Per (set, basin) anchor area nudge: the scale in ``[1-cap, 1+cap]`` that **minimises
    |pbias|** vs the CalSim3 reference **without degrading** KGE or NSE (by more than ``tol``,
    default 0 = strict no-harm — a basin is only nudged when it costs no skill).

    A pure area rescale is multiplicative on the SAC series, so it leaves correlation ``r``
    untouched and only moves the bias / variability terms.  Computed from the **un-nudged**
    anchor.  Returns before/after area + the total adjustment per basin."""
    from .calsim import basin_areas

    long = build_anchor_long(data_dir, sets, apply_scale=False)
    grid = np.linspace(1.0 - cap, 1.0 + cap, int(round(cap * 2000)) + 1)
    rows = []
    for dom in sets:
        areas = basin_areas(data_dir, domain=dom)
        for basin, g in long[long["set"] == dom].groupby("basin"):
            w = g.pivot_table(index="date", columns="source", values="flow_taf")
            if dom not in w or "calsim3" not in w:
                continue
            d = w[[dom, "calsim3"]].dropna()
            if len(d) < 12:
                continue
            sim, ref = d[dom].to_numpy(), d["calsim3"].to_numpy()
            k0, n0, p0 = kge(sim, ref), nse(sim, ref), pbias(sim, ref)
            best = (abs(p0), 1.0, k0, n0, p0)
            for c in grid:
                k, n, pp = kge(sim * c, ref), nse(sim * c, ref), pbias(sim * c, ref)
                if k >= k0 - tol - 1e-9 and n >= n0 - tol - 1e-9 and abs(pp) < best[0] - 1e-9:
                    best = (abs(pp), c, k, n, pp)
            _, c, k, n, pp = best
            rk = g["ref_kind"].dropna().iloc[0] if g["ref_kind"].notna().any() else ""
            ab = float(areas.get(basin, np.nan))
            rows.append({"set": dom, "basin": basin, "ref_kind": rk,
                         "area_before_mi2": round(ab, 1), "scale": round(c, 4),
                         "area_after_mi2": round(ab * c, 1), "adj_pct": round((c - 1) * 100, 1),
                         "pbias_before": round(p0, 1), "pbias_after": round(pp, 1),
                         "kge_before": round(k0, 3), "kge_after": round(k, 3),
                         "nse_before": round(n0, 3), "nse_after": round(n, 3)})
    return pd.DataFrame(rows)


def build_anchor_area_scale(data_dir: str | Path = "data", sets=DEFAULT_CALSETS,
                            *, cap: float = 0.10, force: bool = False) -> pd.DataFrame:
    """Write the hand-editable ``data/reference/anchor_area_scale.csv`` (guarded — refuses to
    overwrite an existing, possibly hand-edited file unless ``force``)."""
    out = Path(data_dir) / "reference" / "anchor_area_scale.csv"
    if out.exists() and not force:
        print(f"anchor_area_scale: {out} exists (hand-edited) — not overwriting (force=True to rebuild)")
        return pd.read_csv(out)
    df = compute_anchor_area_scale(data_dir, sets, cap=cap)
    df.to_csv(out, index=False)
    print(f"anchor_area_scale: wrote {len(df)} basins (cap ±{int(cap*100)}%); "
          f"median |pbias| {df['pbias_before'].abs().median():.1f}% -> "
          f"{df['pbias_after'].abs().median():.1f}% -> {out}")
    return df


def anchor_metrics(long: pd.DataFrame) -> pd.DataFrame:
    """Per (set, basin, source) skill vs CalSim3 over the basin's common period."""
    rows = []
    for (st, basin), g in long.groupby(["set", "basin"]):
        wide = g.pivot_table(index="date", columns="source", values="flow_taf")
        if "calsim3" not in wide:
            continue
        rk = (g["ref_kind"].dropna().iloc[0] if "ref_kind" in g and g["ref_kind"].notna().any()
              else "")
        for src in [st, "vic"]:
            if src not in wide:
                continue
            d = wide[[src, "calsim3"]].dropna()
            if len(d) < 12:
                continue
            sim, ref = d[src].to_numpy(), d["calsim3"].to_numpy()
            rows.append({"set": st, "basin": basin, "source": src, "ref_kind": rk,
                         "n_months": len(d),
                         "kge": kge(sim, ref), "nse": nse(sim, ref), "pbias": pbias(sim, ref),
                         "r": pearson(sim, ref), "mean_sim_taf": float(sim.mean()),
                         "mean_calsim3_taf": float(ref.mean())})
    return pd.DataFrame(rows)


#: the basin-level anchor uses only the GAUGE-calibrated sets — 11obs (rim gauges) and
#: 9unimp (creek gauges).  15cdec (reservoir-calibrated, the ~-23% rim bias) is excluded from
#: the anchor and appears only in the per-catchment / sub-arc best-of.
ANCHOR_SETS = ("11obs", "9unimp")


def make_anchor(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts",
                run: str = "calsim", sets=DEFAULT_CALSETS, *, anchor_long=None) -> Path:
    """Basin-level anchor comparison: each set basin vs the sum of its CalSim3 nodes, using
    only the gauge-calibrated anchor sets (:data:`ANCHOR_SETS` = 11obs/9unimp; 15cdec is
    excluded — it is reservoir-calibrated and only contributes per-catchment sub-arcs).
    ``anchor_long`` may be passed in (already built by :func:`make_all`) to avoid recompute."""
    out = Path(artifacts_dir) / run
    (out / "figures").mkdir(parents=True, exist_ok=True)
    long = build_anchor_long(data_dir, sets) if anchor_long is None else anchor_long
    sets = tuple(s for s in sets if s in ANCHOR_SETS)        # 11obs / 9unimp only
    long = long[long["set"].isin(sets)].copy()
    met = anchor_metrics(long)
    long.to_csv(out / "anchor_monthly.csv", index=False)
    met.to_csv(out / "anchor_metrics.csv", index=False)
    # surface the per-basin area nudge (before/after area + total adjustment) in the report
    scale_csv = Path(data_dir) / "reference" / "anchor_area_scale.csv"
    if scale_csv.exists():
        pd.read_csv(scale_csv).to_csv(out / "anchor_area_scale.csv", index=False)
    msg = "  ".join(f"{s}={met[(met['set']==s)&(met['source']==s)]['kge'].median():.2f}"
                    for s in sets if ((met['set'] == s) & (met['source'] == s)).any())
    print(f"anchor: basin-level vs CalSim3 (FLOW-UNIMPAIRED where a rim system exists, "
          f"else sum of INFLOW sub-nodes); median KGE [{msg}] -> {out}")
    for col, lab, xlim in [("kge", "monthly KGE", (0.0, 1.0)), ("nse", "monthly NSE", (0.0, 1.0)),
                           ("pbias", "percent bias (%)", None)]:
        _anchor_dumbbell_fig(met, sets, col, lab, xlim, out / "figures" / f"anchor_skill_{col}.png")
    _anchor_scatter_fig(long, sets, out / "figures" / "anchor_scatter.png")
    _anchor_hydrograph_fig(long, sets, out / "figures" / "anchor_hydrographs.png")
    if "11obs" in set(long["set"]):
        _main_river_climatology_fig(long, out / "figures" / "main_river_climatology.png",
                                    climset="11obs")
    return out


def _anchor_dumbbell_fig(met, sets, col, ylab, ylim, path):
    """**Vertical** dumbbell per (set, basin): SAC-SMA vs VIC for one metric, grouped by
    set along the x-axis (basins as x ticks, metric on the y-axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(s, b) for s in sets for b in
            sorted(met[(met["set"] == s)]["basin"].unique())]
    if not rows:
        return
    labels = [f"{s}:{b}" for s, b in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(_MAP_W, 4.6))
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.axhline(0, color="0.7", lw=0.8)

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
                       f"else sub-node sum)"), fontsize=8)
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
        a.set_title(f"{st} (KGE={kge(d[st].to_numpy(), d['calsim3'].to_numpy()):.2f})", fontsize=9)
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
    from .calsim import BASIN_RIM_SYSTEM

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
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("TAF/mo", fontsize=8)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(_wrap("8 main river indices: SAC-SMA & VIC vs CalSim3 FLOW-UNIMPAIRED (bold)", 70),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=_MAP_DPI); plt.close(fig)


def _main_river_climatology_fig(long, path, *, climset="11obs"):
    """Mean-monthly (water-year O..S) climatology of the 8 main river indices over the FULL
    period: SAC-SMA (``climset``, default 11obs) vs VIC vs CalSim (FLOW-UNIMPAIRED), TAF/mo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .calsim import BASIN_RIM_SYSTEM

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
        ax.set_title(label, fontsize=10); ax.set_ylabel("mean TAF/mo", fontsize=8)
        ax.set_xticks(range(12)); ax.set_xticklabels(labels); ax.tick_params(labelsize=7)
        ax.grid(color="0.93", lw=0.6); ax.set_axisbelow(True)
    axes.ravel()[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(_wrap(f"8 main river indices — mean-monthly climatology, full period "
                       f"(SAC-SMA {climset}, VIC, CalSim unimpaired)", 70), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=_MAP_DPI); plt.close(fig)


# ==========================================================================
# CalSim FLOW-UNIMPAIRED rim systems (reference for the anchor + 8-river hydrographs)
# ==========================================================================
# The basin-level anchor (make_anchor / _anchor_set_taf) uses each rim system's
# FLOW-UNIMPAIRED series as the faithful whole-watershed reference, and the anchor
# hydrographs draw the 8 main river indices.  UNIMP_MAP keeps each system's label + VIC
# inflow-node candidate list (used by the crosswalk bootstrap, build_crosswalk).
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
    return read_table(Path(data_dir) / "reference" / "calsim_unimpaired_monthly.csv")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="sacsma.compare",
        description="Cross-compare CalSim3 (actual) vs VIC vs multi-set SAC-SMA at the CalSim nodes",
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--run", default="calsim", help="run name -> artifacts/<run>/")
    p.add_argument("--sets", nargs="+", default=None,
                   help="SAC-SMA calibration sets to score separately vs CalSim3 "
                        f"(default: {', '.join(DEFAULT_CALSETS)})")
    p.add_argument("--covered-frac", type=float, default=None,
                   help="informational 'covered'/'partial' status label only "
                        "(default: calsim.COVERED_FRAC); node inclusion is crosswalk-driven")
    p.add_argument("--mass-balance", action="store_true",
                   help="apply CalSim's proportional sub-arc (anchor mass-balance) adjustment "
                        "to the per-catchment estimates (does not improve per-catchment skill)")
    args = p.parse_args(argv)
    sets = tuple(args.sets) if args.sets else DEFAULT_CALSETS
    make_all(args.data_dir, args.artifacts_dir, args.run, sets, covered_frac=args.covered_frac,
             mass_balance=args.mass_balance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
