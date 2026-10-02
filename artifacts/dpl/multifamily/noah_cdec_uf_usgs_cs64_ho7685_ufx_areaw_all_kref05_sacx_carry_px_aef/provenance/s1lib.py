"""Shared statistics of the S1 prereg: derive_bars.py (comparators, from run H / H95 archives) and
readout.py (the S1 run) call the SAME functions, so a bar and its readout are one computation.

Everything reads archived run outputs (sim_daily_mm.npz, tier1/tier1_monthly.csv,
tier2/tier2_monthly.csv, checkpoints/best.pt cfg, chunk_log / train_log) plus the committed stores;
no forward pass, no GPU.  The package (wt_merge, PYTHONPATH) supplies the loaders and scorers:
  - training targets: sacsma.dpl.multi_timescale.load_entity_obs (the trainer's own loader) with the
    run's obs_mask and the WY1976-85 holdout, WITHOUT the uf back-extension = the registry window
    (evaluate's kge_reg basis); selection validity >= 90 daily / >= 12 monthly obs
  - holdout entity rows: sacsma.dpl.evaluate_multi_timescale.holdout_metrics
  - tier-1 / tier-2 window scores: the modules' own _score; own-record months:
    sacsma.dpl.calsim_arcs.own_record_months; anchor-rescaled: calsim_tier2.anchor_rescaled
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from sacsma.metrics import kge

HO = (1976, 1985)
LIVE = Path("<repo>")
MF = LIVE / "artifacts" / "dpl" / "multifamily"
SCR = Path("<scratch>")
DATA = str(SCR / "wt_merge" / "data")
RUNS = {"H": MF / "noah_cdec_uf_sacx_carry_px_aef",
        "H95": MF / "noah_cdec_uf_usgs_areaw_all_kref05_sacx_carry_px_aef"}
#: run H's USGS-69 entity pass (hybrid Stage 0, run H best.pt sha 41dd6f37, cycle spinup)
H_S0_PASS = SCR / "calsim_hybrid" / "hybrid_alt" / "s0" / "passes" / "base" / "entity_sim_daily.npz"
#: P17b re-simulations 1925-10..2018-12 (p17_gpu.py part b)
P17B_DIR = Path("<scratch-2>/train_alts/p17/p17b")
FAMS3 = ("usgs_daily", "cdec_daily", "uf_monthly")
#: H95's selection shares (its flags); S1's sel3 = the same rule at unrounded footprint-area shares
SEL3_H95 = {"usgs_daily": 0.216, "cdec_daily": 0.558, "uf_monthly": 0.226}
SEL3_S1 = {"usgs_daily": 0.2164, "cdec_daily": 0.5578, "uf_monthly": 0.2258}
AREA4 = {"usgs_daily": 0.1967, "cdec_daily": 0.5070, "uf_monthly": 0.2052, "calsim_monthly": 0.0911}
KAPPA4 = {"usgs": 0.7625, "cdec": 1.0111, "uf": 1.9412, "cs": 1.1002}
#: the 9 CDEC twins of DWR UF sites (P17b / scout convention: registry area_mi2 -> TAF)
TWINS = {8: "cdec_ORO", 9: "cdec_YRS", 11: "cdec_FOL", 14: "cdec_MKM", 15: "cdec_NHG",
         16: "cdec_NML", 18: "cdec_TLG", 19: "cdec_MRC", 22: "cdec_MIL"}
P17_SITES = {2: "uf_02", 3: "uf_03", 4: "uf_04", 6: "uf_06", 7: "uf_07", 10: "uf_10", 13: "uf_13",
             20: "uf_20", 21: "uf_21", **TWINS}
TAF_PER_MM_MI2 = 2589988.110336e-3 / 1233.48183754752 / 1e3


def wy_of(months) -> np.ndarray:
    p = pd.PeriodIndex(months, freq="M")
    return np.asarray(p.year + (p.month >= 10))


def registry() -> pd.DataFrame:
    return pd.read_csv(Path(DATA) / "multifamily" / "entities.csv", dtype={"site_id": str},
                       parse_dates=["train_start", "train_end"]).set_index("entity_id")


def tier_a_arcs() -> list[str]:
    h = pd.read_csv(Path(DATA) / "calsim" / "arc_hierarchy.csv")
    return sorted(h.loc[h["train_default"].eq(True), "arc"])


def load_npz(path) -> tuple[list[str], pd.DatetimeIndex, np.ndarray]:
    z = np.load(path)
    return list(map(str, z["entity_id"])), pd.DatetimeIndex(z["date"]), np.asarray(z["sim_mm"], np.float64)


def run_cfg(run_dir) -> dict:
    ck = torch.load(Path(run_dir) / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    return dict(cfg=ck.get("cfg") or {}, epoch=ck.get("epoch"), cal_kge=ck.get("cal_kge"),
                sel3=ck.get("sel3"), basins=ck.get("basins"))


# ------------------------------------------------------------------ training-window entity skill
def train_entity_kge(npz, obs_mask=(), holdout=HO) -> pd.DataFrame:
    """Per entity KGE on the TRAINING target with the holdout out, registry windows (no uf
    back-extension), the run's obs_mask: the trainer's loader on a stub domain over the npz dates.
    Selection validity: >= 90 daily / >= 12 monthly observations."""
    from sacsma.dpl.multi_timescale import load_entity_obs
    ids, dates, sim = load_npz(npz)
    dom = SimpleNamespace(basins=tuple(ids), dates=dates, device=torch.device("cpu"),
                          dtype=torch.float64)
    reg = registry()
    known = set(reg.index)
    mask = tuple(m for m in obs_mask if m.partition("|")[0] in known)
    eo = load_entity_obs(dom, DATA, cal_start=str(dates[0].date()), cal_end=str(dates[-1].date()),
                         obs_mask=mask, holdout_wy=holdout)
    od, om = eo.obs_daily.numpy(), eo.obs_monthly.numpy()
    per = dates.to_period("M")
    code = np.asarray(per.year * 12 + per.month - 1) - int(eo.month_code[0])
    rows = []
    for i, e in enumerate(ids):
        if i in set(eo.daily_rows):
            o, s, mn = od[int(np.flatnonzero(eo.daily_rows == i)[0])], sim[i], 90
        else:
            o = om[int(np.flatnonzero(eo.monthly_rows == i)[0])]
            s = np.zeros(om.shape[1])
            np.add.at(s, code, sim[i])
            mn = 12
        n = int(np.isfinite(o).sum())
        rows.append(dict(entity_id=e, family=reg.loc[e, "family"], n=n,
                         kge=kge(s, o) if n >= mn else np.nan))
    return pd.DataFrame(rows)


def family_stats(ent: pd.DataFrame) -> dict:
    out = {}
    for f, g in ent.groupby("family"):
        k = g["kge"].dropna()
        out[f] = dict(n=int(len(k)), mean=float(k.mean()), median=float(k.median()))
    return out


def shares_stat(fam: dict, shares: dict) -> float:
    have = {f: s for f, s in shares.items() if f in fam}
    tot = sum(have.values())
    return float(sum(s / tot * fam[f]["mean"] for f, s in have.items()))


# ------------------------------------------------------------------ holdout entity skill
def holdout_entities(npz, obs_mask=()) -> pd.DataFrame:
    from sacsma.dpl.evaluate_multi_timescale import holdout_metrics
    ids, dates, sim = load_npz(npz)
    return holdout_metrics(sim, dates, ids, DATA, HO, obs_mask=tuple(obs_mask))


def cdec_twins(npz) -> pd.DataFrame:
    """CDEC twins at the 9 DWR UF sites: monthly TAF (registry area) vs uf_monthly.csv."""
    ids, dates, sim = load_npz(npz)
    reg = registry()
    uf = pd.read_csv(Path(DATA) / "dwr_unimpaired" / "uf_monthly.csv", parse_dates=["date"])
    per = dates.to_period("M")
    rows = []
    for ufn, e in TWINS.items():
        if e not in ids:
            continue
        s = pd.Series(sim[ids.index(e)], index=per).groupby(level=0).sum() * reg.loc[e, "area_mi2"] * TAF_PER_MM_MI2
        g = uf[uf["uf"] == ufn]
        o = pd.Series(g["flow_taf"].to_numpy(), index=pd.PeriodIndex(g["date"], freq="M"))
        d = pd.DataFrame({"s": s, "o": o}).dropna()
        w = wy_of(d.index)
        for wn, (a, b) in {"WY1976-85": (1976, 1985), "WY1976-84": (1976, 1984),
                           "WY1986-2014": (1986, 2014)}.items():
            x = d[(w >= a) & (w <= b)]
            rows.append(dict(entity_id=e, uf=ufn, window=wn, n=len(x), kge=kge(x.s.to_numpy(), x.o.to_numpy())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ tier 1 / tier 2 windows
WINS = {"WY1976-85": (1976, 1985), "WY1976-84": (1976, 1984), "WY1986-2014": (1986, 2014)}


def tier1_windows(run_dir) -> pd.DataFrame:
    from sacsma.dpl.calsim_tier1 import _score
    t = pd.read_csv(Path(run_dir) / "tier1" / "tier1_monthly.csv")
    t["per"] = pd.PeriodIndex(t["month"], freq="M")
    rows = []
    for (sid, rk), g in t.groupby(["set_id", "ref_kind"], sort=False):
        g = g.set_index("per").sort_index()
        for wn, (a, b) in WINS.items():
            idx = pd.period_range(f"{a - 1}-10", f"{b}-09", freq="M")
            m = _score(idx, g["sim_taf"].reindex(idx).to_numpy(), g["ref_taf"].reindex(idx).to_numpy())
            rows.append(dict(set_id=sid, ref_kind=rk, window=wn, n=m["n_months"], kge=m["kge"]))
    return pd.DataFrame(rows)


def tier2_windows(run_dir) -> pd.DataFrame:
    """Per arc: WY1976-85 / WY1976-84 (all months), WY1976-85 own-record months, and the
    training-era own-record months of arc_obs_mask (``arc_train``)."""
    from sacsma.dpl.calsim_arcs import load_arc_mask, own_record_months
    from sacsma.dpl.calsim_tier2 import _score
    t = pd.read_csv(Path(run_dir) / "tier2" / "tier2_monthly.csv")
    t["per"] = pd.PeriodIndex(t["month"], freq="M")
    own = own_record_months(DATA, HO)
    trm = load_arc_mask(DATA)
    rows = []
    for arc, g in t.groupby("arc", sort=False):
        g = g.set_index("per").sort_index()
        s_all, r_all = g["sim_taf"], g["ref_taf"]
        specs = {wn: pd.period_range(f"{a - 1}-10", f"{b}-09", freq="M") for wn, (a, b) in WINS.items()}
        specs["WY1976-85_own"] = own.get(arc, pd.PeriodIndex([], freq="M"))
        specs["arc_train"] = trm.get(arc, pd.PeriodIndex([], freq="M"))
        for wn, idx in specs.items():
            idx = pd.PeriodIndex(idx, freq="M")
            m = _score(idx, s_all.reindex(idx).to_numpy(), r_all.reindex(idx).to_numpy()) if len(idx) \
                else dict(n_months=0, kge=np.nan)
            rows.append(dict(arc=arc, window=wn, n=m["n_months"], kge=m["kge"],
                             has_ref=bool(r_all.notna().any())))
    return pd.DataFrame(rows)


def anchor_rescaled(run_dir) -> pd.DataFrame:
    from sacsma.dpl.calsim_arcs import own_record_months
    from sacsma.dpl.calsim_tier1 import holdout_windows
    from sacsma.dpl.calsim_tier2 import anchor_rescaled as ar
    mon = pd.read_csv(Path(run_dir) / "tier2" / "tier2_monthly.csv")
    return ar(mon, DATA, windows=holdout_windows(HO), own=own_record_months(DATA, HO),
              own_window="WY1976-85")


# ------------------------------------------------------------------ P17b eras (report)
P17_ERAS = {"WY1926-49": (1926, 1949), "WY1950-84_mixed": (1950, 1984), "WY1976-85": (1976, 1985),
            "WY1976-84": (1976, 1984), "WY1986-2014": (1986, 2014)}


def p17b_sites(npz) -> pd.DataFrame:
    ids, dates, sim = load_npz(npz)
    reg = registry()
    uf = pd.read_csv(Path(DATA) / "dwr_unimpaired" / "uf_monthly.csv", parse_dates=["date"])
    per = dates.to_period("M")
    rows = []
    for ufn, e in P17_SITES.items():
        if e not in ids:
            continue
        s = pd.Series(sim[ids.index(e)], index=per).groupby(level=0).sum() * reg.loc[e, "area_mi2"] * TAF_PER_MM_MI2
        g = uf[uf["uf"] == ufn]
        o = pd.Series(g["flow_taf"].to_numpy(), index=pd.PeriodIndex(g["date"], freq="M"))
        d = pd.DataFrame({"s": s, "o": o}).dropna()
        w = wy_of(d.index)
        for era, (a, b) in P17_ERAS.items():
            x = d[(w >= a) & (w <= b)]
            rows.append(dict(entity_id=e, uf=ufn, era=era, n=len(x), kge=kge(x.s.to_numpy(), x.o.to_numpy()),
                             pbias=100 * (x.s.sum() - x.o.sum()) / x.o.sum() if len(x) else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ mechanism (S1 logs only)
FAM_LOG = ("usgs", "cdec", "uf", "cs")


def mechanism(run_dir) -> dict:
    """e0 realized coefficient shares (sum c_f / kappa_f over stepped chunks), loss-mass shares per
    epoch, clipped-chunk share (gnorm > 1) in epochs >= 5, NaN selections, best epoch."""
    cl = pd.read_csv(Path(run_dir) / "chunk_log.csv")
    tl = pd.read_csv(Path(run_dir) / "train_log.csv")
    st = cl[cl["stepped"].astype(str) == "True"].copy()
    for c in [c for c in st.columns if c[:2] in ("l_", "c_", "n_")] + ["gnorm", "loss"]:
        st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0.0)
    fams = [f for f in FAM_LOG if f"c_{f}" in st.columns]
    s0 = st[st["epoch"] == st["epoch"].min()]
    cs = {f: float((s0[f"c_{f}"] / KAPPA4[f]).sum()) for f in fams}
    z = sum(cs.values()) or 1.0
    mass = {}
    for ep, g in st.groupby("epoch"):
        lm = {f: float(g[f"l_{f}"].sum()) for f in fams}
        zl = sum(lm.values()) or 1.0
        mass[int(ep)] = {f: v / zl for f, v in lm.items()}
    late = st[st["epoch"] >= 5]
    return dict(realized_e0={f: v / z for f, v in cs.items()}, loss_mass=mass,
                clip_share_e5=float((late["gnorm"] > 1.0).mean()) if len(late) else float("nan"),
                n_epochs=int(tl["epoch"].max()) + 1,
                skipped=int(tl.get("skipped_steps", pd.Series([0])).sum()),
                skipped_max_epoch=int(tl.get("skipped_steps", pd.Series([0])).max()))
