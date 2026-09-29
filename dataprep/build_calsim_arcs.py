"""Build the CalSim3 rim-arc target set: hierarchy, observation mask and depth store.

Writes three derived tables under ``data/calsim/`` (deterministic; no hand edits):

* ``arc_hierarchy.csv`` — one row per CalSim3 rim node: every ``CalSim3_Merged``
  polygon (INFLOW arc id via ``series_arc``) plus every arc of
  ``calsim3_arc_derivation.csv`` (203 rows).  Area, anchor nesting, FLOW-UNIMPAIRED
  system, residual flag, closure group, entity duplicates, training-USGS provenance,
  the observation record in the training water years, and the record tier.
* ``arc_obs_mask.csv`` — ``arc, date`` (month-end), one row per arc-month that is the
  arc's OWN observed record inside the training water years WY1950-75 + WY1986-2015:
  the listed gauge period(s) (HR Tables 5-1/5-5, ``period``) minus every water year the
  report lists as a correlation extension (Tables 5-2/5-6, ``extension_period``).
  Written for every arc of the depth store whose record is its own (not a donor's):
  tiers A, B, C and the gauged D/X arcs — which rows TRAIN is the registry's choice
  (``train_default`` = tier A).
* ``calsim3_inflow_monthly_mm.csv`` — ``arc, date, depth_mm``: the 196 rim INFLOW
  series (arcs with a series and a polygon, below-rim ``EXCLUDE_ARCS`` left out) as
  depth over the arc's ``SQ_MI``, full record WY1922-2021, TAF -> mm with
  ``AF_PER_MM_MI2`` (the tier-1/tier-2 constant).

Anchors.  CalSim3 FLOW-UNIMPAIRED systems (``FU_<SYS>``: member arcs from
``tier1_sets.csv`` — the crosswalk's systems with SRBB nesting SHAS and the valley node,
TRIN = ``I_TRNTY`` alone — plus WH = ``I_WKYTN``) and every cdec_daily / uf_monthly
registry entity with an arc list.  ``innermost_anchor`` = the smallest containing arc
set, ``top_anchor`` = the largest (ties: FU, then uf, then cdec).

Closure groups (the water-year closure of ``sacsma.dpl.calsim_arcs``): an arc in a
FLOW-UNIMPAIRED system closes in its innermost FU system other than SRBB (so Shasta and
Whiskeytown close on their own); an SRBB-only arc is excluded (the rim arcs sit ~973
TAF/yr under FU SRBB by design, the valley node holds it); an arc outside every FU
system closes in its largest training-entity anchor, to that entity's DWR unimpaired
monthly series (a CDEC entity uses the UF with the identical arc set); ``uf_03`` (the
routed outflow below Clear Lake) is excluded.

Tiers (defined here; ``tier_listed`` = the listed gauge period alone, ``tier`` = with
the extension years carved out):
  A own gauge (gauged_extended / complete_record), record in the training WYs, passes
    the naturalness screen;  B the same, fails it;  C by-difference of own gauges with a
    record, passes;  D reservoir mass balance;  E donor split / proportioned;
  F no own record in the training WYs;  X none of these (tier_note says why).

The naturalness screen is the ``REGULATED`` list below, the complete record of a one-time
review of the derivation arcs (195 screened arcs, 40 flagged); the arcs it did not cover
(``UNSCREENED``) are NaN.

Usage (sacsma conda env, from the repo root):
    python dataprep/build_calsim_arcs.py [--data-dir data] [--out-dir data/calsim]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sacsma.calsim.catchments import (  # noqa: E402
    EXCLUDE_ARCS,
    MERGED_LAYER,
    load_catchments,
    series_arc,
)
from sacsma.dpl.calsim_arcs import (  # noqa: E402
    AF_PER_MM_MI2,
    CLOSURE_EXCLUDED,
    TRAIN_WY,
    in_record_months,
    mm_to_taf,
    taf_to_mm,
)

#: FLOW-UNIMPAIRED system WH (Whiskeytown) is not a tier-1 set: its series is I_WKYTN's
#: (WY1950-2015 ratio 0.9956, r 0.996; I_WKYTN + I_CLR011 would be 1.158)
EXTRA_FU = {"WH": ["I_WKYTN"]}
#: the naturalness screen (a one-time review, 2026-09-28; this list is its full record):
#: arcs flagged as regulated / not natural.  The screen covered every derivation arc but
#: I_FSL012, I_JBP006, I_RUB002, I_SJR258, I_SJR265, I_TUO054 (and I_ECHOL, which the
#: derivation table lacks): those are unscreened.
REGULATED = frozenset("""
I_BANOS I_BEARD I_BUR005 I_CAP000 I_CHC000 I_CLV026 I_DPC008 I_ENGLB I_FOLSM I_GERLE
I_GRZ012 I_ING008 I_LGRSV I_LOSVQ I_LPC007 I_LWSTN I_LYONS I_MFA001 I_MFA023 I_MFA025
I_MFS013 I_MNS000 I_MTMDW I_NFM010 I_ORG000 I_OROVL I_ORT014 I_QNT005 I_SCOTF I_SFA030
I_SFA040 I_SFD003 I_SFF008 I_SFF011 I_SFS030 I_SLO007 I_SLUIS I_STS072 I_TGC003 I_WBF006
""".split())
UNSCREENED = frozenset({"I_FSL012", "I_JBP006", "I_RUB002", "I_SJR258", "I_SJR265",
                        "I_TUO054", "I_ECHOL"})
#: the single-arc training entities (their arc is a duplicate target)
EXPECTED_DUPLICATES = {"I_SHSTA", "I_TRNTY", "I_MLRTN", "I_MCLRE", "I_NHGAN",
                       "I_PTH070", "I_ESTMN"}
DONOR_CLASSES = ("gauged_extended_split", "proportioned_ungauged")
_KIND_RANK = {"FU": 0, "uf": 1, "cdec": 2}


def _split(s) -> list[str]:
    return [a for a in str(s).split(";") if a and a != "nan"]


def _rank(anchor: str) -> int:
    return _KIND_RANK[anchor.split("_", 1)[0]]


def anchors(data_dir: Path, reg: pd.DataFrame) -> dict[str, list[str]]:
    t1 = pd.read_csv(data_dir / "calsim" / "tier1_sets.csv")
    out = {f"FU_{r.system}": _split(r.arcs) for r in t1.itertuples() if isinstance(r.system, str)}
    out.update({f"FU_{s}": a for s, a in EXTRA_FU.items()})
    for r in reg.itertuples():
        if r.family in ("cdec_daily", "uf_monthly") and _split(r.arcs):
            out[r.entity_id] = _split(r.arcs)
    return out


def _tier(method_class, has_series, in_deriv, n_months, reg_flag) -> tuple[str, str]:
    if not has_series:
        return "X", "seriesless"
    if not in_deriv:
        return "X", "not_in_derivation"
    if method_class == "reservoir_mass_balance":
        return "D", ""
    if method_class in DONOR_CLASSES:
        return "E", ""
    if n_months == 0:
        return "F", ""
    if method_class in ("gauged_extended", "complete_record"):
        if pd.isna(reg_flag):
            return "X", "own_gauge_unscreened"
        return ("B", "") if reg_flag else ("A", "")
    if method_class == "by_difference_accretion":
        if pd.isna(reg_flag):
            return "X", "by_difference_unscreened"
        return ("X", "by_difference_fails_screen") if reg_flag else ("C", "")
    return "X", str(method_class)


def build(data_dir: Path):
    deriv = pd.read_csv(data_dir / "calsim" / "calsim3_arc_derivation.csv")
    if "extension_period" not in deriv.columns:
        raise ValueError("calsim3_arc_derivation.csv lacks extension_period")
    reg = pd.read_csv(data_dir / "multifamily" / "entities.csv", dtype={"site_id": str})
    reg = reg[reg["family"] != "calsim_monthly"]                 # the base registry
    inflow = pd.read_csv(data_dir / "calsim" / "calsim3_inflow_monthly.csv",
                         parse_dates=["date"])
    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True)
    catch["arc"] = catch["node"].map(series_arc)
    assert catch["arc"].is_unique
    sq_mi = catch.set_index("arc")["sq_mi"].astype(float)
    series = set(inflow["arc"])
    uf_loc = pd.read_csv(data_dir / "dwr_unimpaired" / "uf_locations.csv")
    uf_by_arcs = {frozenset(_split(r.arcs)): int(r.uf)
                  for r in uf_loc.itertuples() if int(r.n_arcs) > 0}

    anc = anchors(data_dir, reg)
    usgs_ids = set(reg.loc[reg["family"] == "usgs_daily", "site_id"])
    dup = {}
    for r in reg.itertuples():
        a = _split(r.arcs)
        if len(a) == 1:
            dup[a[0]] = r.entity_id
    assert set(dup) == EXPECTED_DUPLICATES, sorted(dup)

    d = deriv.set_index("arc")
    universe = sorted(set(sq_mi.index) | set(d.index))
    rows = []
    for arc in universe:
        in_d = arc in d.index
        dr = d.loc[arc] if in_d else None
        mc = dr["method_class"] if in_d else np.nan
        has_series = arc in series
        cont = sorted((k for k, v in anc.items() if arc in v),
                      key=lambda k: (len(anc[k]), _rank(k), k))
        top = min(cont, key=lambda k: (-len(anc[k]), _rank(k), k)) if cont else ""
        fus = [k for k in cont if k.startswith("FU_")]
        # closure group
        group, excluded, ref = "", "", ""
        fu_close = [k for k in fus if k not in CLOSURE_EXCLUDED]
        if not has_series:
            excluded = "seriesless" + (f":{fus[0]}" if fus else "")
        elif fu_close:
            group = fu_close[0]
            ref = f"calsim_unimpaired:{group[3:]}"
        elif fus:                                            # SRBB-only arcs
            excluded = f"{fus[0]}:{CLOSURE_EXCLUDED[fus[0]]}"
        else:
            ents = [k for k in cont if not k.startswith("FU_")]
            if ents:
                g = min(ents, key=lambda k: (-len(anc[k]), _rank(k), k))
                if g in CLOSURE_EXCLUDED:
                    excluded = f"{g}:{CLOSURE_EXCLUDED[g]}"
                else:
                    group = g
                    ufn = (int(reg.set_index("entity_id").loc[g, "site_id"].split()[-1])
                           if g.startswith("uf_") else uf_by_arcs.get(frozenset(anc[g])))
                    ref = f"uf_monthly:{ufn}" if ufn is not None else f"entity:{g}"
            else:
                excluded = "unanchored"
        # record
        own = in_d and mc not in DONOR_CLASSES
        per = dr["period"] if in_d else np.nan
        ext = dr["extension_period"] if in_d else np.nan
        none = pd.PeriodIndex([], freq="M")
        m_list = in_record_months(per, ext, carve_extension=False) if own else none
        m_cut = in_record_months(per, ext) if own else none
        reg_flag = (np.nan if (arc in UNSCREENED or not in_d) else (arc in REGULATED))
        t_list, n_list = _tier(mc, has_series, in_d, len(m_list), reg_flag)
        t_cut, n_cut = _tier(mc, has_series, in_d, len(m_cut), reg_flag)
        gauge = str(dr["gauge"]) if in_d else ""
        hits = sorted(set(re.findall(r"(?<!\d)(\d{8})(?!\d)", gauge)) & usgs_ids)
        if arc in EXCLUDE_ARCS:
            status = "excluded_below_rim"
        elif not has_series:
            status = "seriesless_node"
        elif arc not in sq_mi.index:
            status = "no_polygon"
        else:
            status = "rim_arc"
        rows.append(dict(
            arc=arc, region=(dr["region"] if in_d else ""), status=status,
            sq_mi=float(sq_mi[arc]) if arc in sq_mi.index else np.nan,
            has_polygon=arc in sq_mi.index, has_series=has_series,
            method_class=mc, residual=(mc == "by_difference_accretion"),
            n_containing=len(cont), containing_anchors=";".join(cont),
            innermost_anchor=cont[0] if cont else "", top_anchor=top,
            fu_systems=";".join(k[3:] for k in fus),
            closure_group=group, closure_anchor=ref, closure_excluded=excluded,
            duplicates_entity=dup.get(arc, ""),
            usgs_training_gauge=";".join(hits),
            reg_screen=reg_flag,
            record_months_listed=len(m_list), record_months=len(m_cut),
            record_first=str(m_cut.min()) if len(m_cut) else "",
            record_last=str(m_cut.max()) if len(m_cut) else "",
            tier_listed=t_list, tier=t_cut, tier_note=n_cut,
        ))
    hier = pd.DataFrame(rows)
    hier["train_default"] = (hier["tier"] == "A") & (hier["duplicates_entity"] == "")

    # ---- depth store: the rim arcs with a series and a polygon
    rim = hier[(hier["status"] == "rim_arc")]
    arcs = list(rim["arc"])
    inf = inflow[inflow["arc"].isin(set(arcs))].copy()
    inf["sq_mi"] = inf["arc"].map(sq_mi)
    inf["depth_mm"] = taf_to_mm(inf["flow_taf"].to_numpy(), inf["sq_mi"].to_numpy())
    back = mm_to_taf(inf["depth_mm"].to_numpy(), inf["sq_mi"].to_numpy())
    taf = inf["flow_taf"].to_numpy()
    rel = np.abs(back - taf) / np.maximum(np.abs(taf), 1e-300)
    rel[taf == 0] = np.abs(back[taf == 0])
    depth = (inf.sort_values(["arc", "date"])[["arc", "date", "depth_mm"]]
             .reset_index(drop=True))
    depth["date"] = depth["date"].dt.strftime("%Y-%m-%d")

    # ---- observation mask: own-record arcs of the depth store
    mrows = []
    for r in rim.itertuples():
        if not r.record_months:
            continue
        dr = d.loc[r.arc]
        ms = in_record_months(dr["period"], dr["extension_period"])
        for p in ms:
            mrows.append((r.arc, p.end_time.strftime("%Y-%m-%d")))
    mask = pd.DataFrame(mrows, columns=["arc", "date"])
    return hier, depth, mask, float(rel.max()), anc


def gates(hier: pd.DataFrame, depth: pd.DataFrame, mask: pd.DataFrame, rt: float) -> dict:
    g = {}
    rim = hier[hier["status"] == "rim_arc"]
    g["n_rows"] = len(hier)
    g["n_rim_arcs"] = len(rim)
    assert len(rim) == 196, len(rim)
    assert hier["arc"].is_unique
    assert set(hier.loc[~hier["has_series"], "arc"]) == {"I_SRBB_VAL"}
    g["tier_listed_counts"] = hier["tier_listed"].value_counts().sort_index().to_dict()
    g["tier_counts"] = hier["tier"].value_counts().sort_index().to_dict()
    d = hier[hier["region"] != ""]                       # the 201 derivation arcs
    g["tier_listed_counts_derivation"] = d["tier_listed"].value_counts().sort_index().to_dict()
    g["n_tier_A_listed"] = int((hier["tier_listed"] == "A").sum())
    g["n_tier_A"] = int((hier["tier"] == "A").sum())
    g["n_train_default"] = int(hier["train_default"].sum())
    a = hier[hier["tier"] == "A"]
    g["tier_A_from_training_usgs"] = sorted(a.loc[a["usgs_training_gauge"] != "", "arc"])
    g["tier_A_duplicates"] = sorted(a.loc[a["duplicates_entity"] != "", "arc"])
    g["roundtrip_max_rel"] = rt
    assert rt < 1e-9, rt
    assert depth["arc"].nunique() == 196 and len(depth) == 196 * 1200, len(depth)
    assert not mask.duplicated().any()
    n = mask.groupby("arc").size()
    want = rim.set_index("arc")["record_months"]
    assert n.reindex(want.index).fillna(0).astype(int).equals(want.astype(int))
    # no mask month in the holdout or before WY1950
    p = pd.PeriodIndex(pd.to_datetime(mask["date"]), freq="M")
    wy = np.where(p.month >= 10, p.year + 1, p.year)
    ok = np.logical_or.reduce([(wy >= lo) & (wy <= hi) for lo, hi in TRAIN_WY])
    assert ok.all()
    g["mask_rows"] = len(mask)
    g["mask_arcs"] = int(mask["arc"].nunique())
    default_arcs = set(hier.loc[hier["train_default"], "arc"])
    g["mask_rows_train_default"] = int(mask["arc"].isin(default_arcs).sum())
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=Path("data"), type=Path)
    ap.add_argument("--out-dir", default=None, type=Path,
                    help="where the three tables go (default: <data-dir>/calsim)")
    args = ap.parse_args()
    out = args.out_dir or args.data_dir / "calsim"
    hier, depth, mask, rt, _ = build(args.data_dir)
    g = gates(hier, depth, mask, rt)
    out.mkdir(parents=True, exist_ok=True)
    hier.to_csv(out / "arc_hierarchy.csv", index=False)
    depth.to_csv(out / "calsim3_inflow_monthly_mm.csv", index=False)
    mask.to_csv(out / "arc_obs_mask.csv", index=False)
    for k, v in g.items():
        print(f"{k}: {v}")
    print(f"wrote {out}/arc_hierarchy.csv ({len(hier)}), calsim3_inflow_monthly_mm.csv "
          f"({len(depth)}), arc_obs_mask.csv ({len(mask)}); AF/(mm mi2) = {AF_PER_MM_MI2!r}")


if __name__ == "__main__":
    main()
