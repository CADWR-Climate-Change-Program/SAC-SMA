"""Tier-1 CalSim3 validation of a multifamily dPL run that held water years out of training.

The twenty tier-1 locations (``data/inputs/calsim3/tier1_sets.csv``) are the training-target
watersheds expressed as CalSim3 arc sets.  They are scored in **volume** (TAF/month)
over the run's held-out water years against CalSim3: the ``FLOW-UNIMPAIRED`` series
where a system carries one (ten anchors), the sum of the member ``INFLOW`` arcs
elsewhere.  The simulation is the run's archived daily entity depth
(``sim_daily.npz`` — the area-weighted mean over the entity's cell set), summed to
calendar months and converted with the ``CalSim3_Merged`` ``SQ_MI`` of the arcs the
entity simulates, so no third area enters the comparison.

A set whose registry entity omits an arc is simulated on a footprint smaller than the arc
set (at present only the Yuba, whose entity leaves out the two Deer Creek arcs); it is
scored against the tier-1 reference as defined *and* against the sum of the arcs it does
simulate (``ref_kind = arcsum_covered``), and the covered area fraction is reported.
Cache Creek is counted like the others although its
arcs are lake inflows while its training record is the routed outflow; the set table's
``volume_scored`` flag is what includes or excludes a location from the aggregate.  A set
whose arcs all belong to a larger set (Shasta's ``I_SHSTA`` is one of Red Bluff's nine)
is *nested*: it keeps its own row and enters the skill statistics, but the volume totals
sum the unnested sets only, or Shasta's water would be counted twice.  Nesting is
derived from the arc lists (``nested_in`` column), never flagged by hand.

The run held water years out of training in every family (``DplConfig.holdout_wy``, read from
its checkpoint) and is validated over them: windows ``WY1976-85`` (the holdout) and
``WY1976-84`` (the holdout minus its last water year, comparable with runs that trained the
monthly family's WY1985), and WY1950-84 as ``WY1950-84_mixed`` (partly in-sample: the creeks,
the back-extended monthly family and the arcs train WY1950-75).  The ten anchored locations
coincide with the monthly training targets, so their held-out scores are a temporal holdout of
the training target rather than an independent reference; the arc-sum locations compare against
CalSim3's own inflow hydrology.  Each entity's own training window (the registry's
``train_start`` .. ``train_end``: WY1985-2014 for the monthly family, record start to 2018-12
for the daily families) is scored alongside as the in-sample comparison (``window = train``),
without the held-out water years (``excluded_wy``).  The regime figure and the summary are over
``WY1976-85``.

Usage::

    sacsma dpl calsim tier1 <run_dir> [--out DIR] [--data-dir data]

Writes ``tier1_metrics.csv`` (one row per set x reference x window) to the run's results
folder, ``tier1_monthly.csv`` (the aligned monthly volumes), ``tier1_regime_WY1976-85.png``
and the location figures to its local folder (both ``tier1/``; ``--out`` puts everything in
one folder), and prints the summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ... import paths
from ...io import read_table
from ...metrics import center_of_timing, kge, nse, pbias, pearson, seasonal_mismatch
from ...calsim import load_calsim3_monthly
from ...calsim.catchments import CALSIM_GPKG, MERGED_LAYER, series_arc

#: 1 mm of depth over 1 mi^2 in acre-feet: 2,589,988.11 m^2 x 1e-3 m / 1,233.4818 m^3 per AF.
AF_PER_MM_MI2 = 2589988.110336e-3 / 1233.48183754752

#: water years 1950-1984 as an inclusive month range.  The in-sample window is per entity (see
#: :func:`registry_windows`).
WINDOWS = {"WY1950-84": ("1949-10", "1984-09")}
MIXED_WINDOW = "WY1950-84"
#: a run that held water years out of training in every family (``DplConfig.holdout_wy``) is
#: validated over them: window ``WY<first>-<last>`` (e.g. ``WY1976-85``), plus the same
#: minus its last water year (``WY1976-84``: comparable with runs that trained the monthly
#: family's registry window, which starts in WY1985).  WY1950-84 is partly in-sample (the
#: creeks, the back-extended monthly family and the arcs train WY1950-75) and is kept under
#: this suffix.
MIXED_SUFFIX = "_mixed"
_WY_MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]


def wy_label(first: int, last: int) -> str:
    """``WY1976-85`` for water years 1976..1985 (last two digits, four across a century)."""
    return f"WY{first}-{last % 100:02d}" if last // 100 == first // 100 else f"WY{first}-{last}"


def window_range(name: str) -> tuple[str, str]:
    """The inclusive month range of a window name: :data:`WINDOWS` first, else a
    ``WY<first>-<last>`` name (any suffix after it ignored) as Oct of first-1 .. Sep of last."""
    if name in WINDOWS:
        return WINDOWS[name]
    import re
    m = re.match(r"^WY(\d{4})-(\d{2}|\d{4})", str(name))
    if not m:
        raise KeyError(f"window {name!r}")
    a = int(m.group(1))
    b = int(m.group(2)) if len(m.group(2)) == 4 else a // 100 * 100 + int(m.group(2))
    if b < a:
        b += 100
    return f"{a - 1}-10", f"{b}-09"


def holdout_windows(holdout_wy) -> dict[str, tuple[str, str]]:
    """The scoring windows of a run with ``holdout_wy`` (first, last): the holdout (the
    validation window), the holdout minus its last water year, and WY1950-84 as mixed."""
    a, b = int(holdout_wy[0]), int(holdout_wy[1])
    out = {wy_label(a, b): window_range(wy_label(a, b))}
    if b > a:
        out[wy_label(a, b - 1)] = window_range(wy_label(a, b - 1))
    out[MIXED_WINDOW + MIXED_SUFFIX] = WINDOWS[MIXED_WINDOW]
    return out


def _run_cfg(path: str | Path) -> dict:
    """The ``cfg`` of a run folder (its ``checkpoints/best.pt``) or a checkpoint file; ``{}``
    when the checkpoint does not exist."""
    p = Path(path)
    ck = p if p.is_file() else paths.run_roles(p).model / "checkpoints" / "best.pt"
    if not ck.exists():
        return {}
    import torch

    return torch.load(ck, map_location="cpu", weights_only=False).get("cfg") or {}


def run_holdout_wy(path: str | Path) -> tuple[int, ...]:
    """``DplConfig.holdout_wy`` of a run folder (its ``checkpoints/best.pt``) or a checkpoint
    file; ``()`` when the checkpoint has none (or does not exist)."""
    ho = _run_cfg(path).get("holdout_wy") or ()
    if isinstance(ho, str):
        ho = tuple(int(v) for v in ho.split("-") if v)
    return tuple(int(v) for v in ho)


def require_holdout(path: str | Path, tool: str) -> tuple[int, ...]:
    """The run's ``holdout_wy``; stops ``tool`` on a run that held no water years out of
    training, which the CalSim3 tiers do not validate."""
    ho = run_holdout_wy(path)
    if not ho:
        raise SystemExit(f"{tool}: {path} held no water years out of training "
                         "(DplConfig.holdout_wy); the CalSim3 tiers validate a holdout run")
    return ho


def protected_wy(path: str | Path) -> tuple[int, ...]:
    """The water years a CalSim-chain tool scores only with ``--score-holdout``: the run's
    ``holdout_wy``; for a run trained on a ``train_window`` (the 15-CDEC rungs) the shared
    held-out decade :data:`~sacsma.dpl.calsim.arcs.HOLDOUT_WY`, which its window leaves out;
    ``()`` for a run trained on every year."""
    from .arcs import HOLDOUT_WY

    ho = run_holdout_wy(path)
    if ho:
        return ho
    return tuple(HOLDOUT_WY) if _run_cfg(path).get("train_window") else ()


def refuse_protected(path: str | Path, score_holdout: bool, tool: str) -> None:
    """Stop ``tool`` (a scorer with nothing to write but scores) on a run with
    :func:`protected_wy`, unless ``--score-holdout`` asks for them."""
    pw = protected_wy(path)
    if pw and not score_holdout:
        raise SystemExit(f"{tool}: {path} holds {wy_label(*pw)} out; its scores read those "
                         "water years. Pass --score-holdout to score them.")


def holdout_month_mask(months: pd.PeriodIndex, holdout_wy) -> np.ndarray:
    """True on the months whose water year is inside ``holdout_wy`` (all False for ``()``)."""
    if not holdout_wy:
        return np.zeros(len(months), dtype=bool)
    wy = np.asarray(months.year + (months.month >= 10))
    return (wy >= int(holdout_wy[0])) & (wy <= int(holdout_wy[1]))


def _split(arcs) -> list[str]:
    if arcs is None or (isinstance(arcs, float) and np.isnan(arcs)):
        return []
    return [a for a in str(arcs).split(";") if a]


def load_sets(data_dir: str | Path = "data") -> pd.DataFrame:
    """The tier-1 arc-set table, ``arcs`` split into lists."""
    # only an empty cell is a missing value
    s = pd.read_csv(paths.tier1_sets(data_dir), keep_default_na=False, na_values=[""])
    s["arcs"] = s["arcs"].map(_split)
    s["system"] = s["system"].fillna("")
    s["note"] = s["note"].fillna("")
    s["volume_scored"] = s["volume_scored"].astype(bool)
    s["nested_in"] = s["set_id"].map(nested_sets(s)).fillna("")
    return s


def nested_sets(sets: pd.DataFrame) -> dict[str, str]:
    """``{set_id: parent set_id}`` for every set whose arcs are a proper subset of another
    set's (the smallest such parent).  Two sets that share arcs with neither containing the
    other would make the volume total ambiguous, so that is an error."""
    arcs = {s.set_id: set(s.arcs) for s in sets.itertuples(index=False)}
    area = dict(zip(sets.set_id, sets.area_mi2, strict=True))
    out: dict[str, str] = {}
    for a, sa in arcs.items():
        parents = [b for b, sb in arcs.items() if b != a and sa and sa < sb]
        if parents:
            out[a] = min(parents, key=lambda b: (float(area[b]), b))
        for b, sb in arcs.items():
            if b > a and (sa & sb) and not (sa <= sb or sb <= sa):
                raise ValueError(f"tier1_sets.csv: {a} and {b} share arcs {sorted(sa & sb)} "
                                 "but neither contains the other")
    return out


def arc_to_set(sets: pd.DataFrame) -> dict[str, str]:
    """``{arc: set_id}`` for every arc a tier-1 set lists — the smallest set where an arc
    belongs to several (Shasta's ``I_SHSTA`` goes to SHA, not to Red Bluff)."""
    out: dict[str, str] = {}
    for s in sets.sort_values("area_mi2", ascending=False).itertuples(index=False):
        for a in s.arcs:
            out[a] = s.set_id
    return out


def volume_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """The rows a volume total sums: volume-scored, main reference kind, not nested."""
    keep = metrics.volume_scored & metrics.ref_kind.isin(["anchor", "arcsum"])
    if "nested_in" in metrics.columns:
        keep &= metrics["nested_in"].fillna("").astype(str) == ""
    return metrics[keep]


def arc_areas(data_dir: str | Path = "data") -> dict[str, float]:
    """``CalSim3_Merged`` ``SQ_MI`` keyed by INFLOW-series arc id."""
    import geopandas as gpd
    g = gpd.read_file(paths.calsim3_gpkg(data_dir, CALSIM_GPKG), layer=MERGED_LAYER,
                      ignore_geometry=True)
    return {series_arc(n): float(a) for n, a in zip(g["Connect_No"], g["SQ_MI"], strict=True)}


def registry_arcs(data_dir: str | Path = "data") -> dict[str, list[str]]:
    reg = pd.read_csv(paths.entities(data_dir), dtype={"site_id": str})
    return {e: _split(a) for e, a in zip(reg["entity_id"], reg["arcs"], strict=True)}


def registry_windows(data_dir: str | Path = "data") -> dict[str, tuple[pd.Period, pd.Period]]:
    """Each entity's training window as (first month, last month) from the registry."""
    reg = pd.read_csv(paths.entities(data_dir), dtype={"site_id": str},
                      parse_dates=["train_start", "train_end"])
    return {e: (pd.Period(s, "M"), pd.Period(t, "M"))
            for e, s, t in zip(reg["entity_id"], reg["train_start"], reg["train_end"], strict=True)}


def footprint_areas(data_dir: str | Path = "data") -> dict[str, float]:
    cells = pd.read_csv(paths.entity_cells(data_dir))
    return cells.groupby("entity_id")["overlap_mi2"].sum().to_dict()


def load_run_monthly_depth(run_dir: str | Path) -> pd.DataFrame:
    """Monthly entity depth (mm/month, complete calendar months) from ``sim_daily.npz``."""
    z = np.load(paths.run_roles(run_dir).results / "sim_daily.npz")
    daily = pd.DataFrame(np.asarray(z["sim_mm"]).T.astype(float),
                         index=pd.to_datetime(z["date"]), columns=list(z["entity_id"]))
    monthly = daily.resample("ME").sum(min_count=1)
    complete = daily.resample("ME").count().iloc[:, 0].to_numpy() >= monthly.index.days_in_month
    monthly[~complete] = np.nan
    monthly.index = monthly.index.to_period("M")
    return monthly


def load_references(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(INFLOW arcs, FLOW-UNIMPAIRED systems) as month x series tables of TAF/month."""
    inflow = load_calsim3_monthly(data_dir)
    inflow["month"] = pd.to_datetime(inflow["date"]).dt.to_period("M")
    inflow = inflow.pivot(index="month", columns="arc", values="flow_taf")
    unimp = read_table(paths.calsim3_targets(data_dir, "calsim_unimpaired_monthly.csv"))
    unimp["month"] = pd.to_datetime(unimp["date"]).dt.to_period("M")
    unimp = unimp.pivot(index="month", columns="system", values="flow_taf")
    return inflow, unimp


def _score(months: pd.PeriodIndex, sim: np.ndarray, ref: np.ndarray) -> dict:
    m = np.isfinite(sim) & np.isfinite(ref)
    out = dict(n_months=int(m.sum()), kge=np.nan, nse=np.nan, pbias=np.nan, r=np.nan,
               alpha=np.nan, beta=np.nan, seas_mismatch=np.nan, ct_diff=np.nan,
               sim_taf_yr=np.nan, ref_taf_yr=np.nan)
    if m.sum() < 12:
        return out
    s, r = sim[m], ref[m]
    d = months[m].to_timestamp()
    out.update(kge=kge(s, r), nse=nse(s, r), pbias=pbias(s, r), r=pearson(s, r),
               alpha=s.std() / r.std() if r.std() > 0 else np.nan,
               beta=s.mean() / r.mean() if r.mean() != 0 else np.nan,
               seas_mismatch=seasonal_mismatch(d, s, r),
               ct_diff=center_of_timing(d, s) - center_of_timing(d, r),
               sim_taf_yr=12.0 * s.mean(), ref_taf_yr=12.0 * r.mean())
    return out


def _arcsum(inflow: pd.DataFrame, arcs: list[str], label: str = "") -> pd.Series:
    """Sum of the listed arcs' INFLOW series, NaN in any month one of them lacks.  Arcs
    without a series (valley nodes such as ``I_SRBB_VAL``) are left out and named once."""
    have = [a for a in arcs if a in inflow.columns]
    absent = [a for a in arcs if a not in inflow.columns]
    if absent:
        print(f"tier1: {label or 'arc sum'}: no CalSim3 series for {', '.join(absent)} — summing "
              f"the other {len(have)}", flush=True)
    if not have:
        return pd.Series(np.nan, index=inflow.index)
    return inflow[have].sum(axis=1, min_count=len(have))


def training_record_taf(entity_id: str, data_dir: str | Path = "data",
                        obs_mask: tuple[str, ...] = ()) -> pd.Series | None:
    """The entity's own observed record as monthly TAF (month PeriodIndex): the DWR
    monthly series for a ``uf_monthly`` entity, the daily depth store summed over complete
    months and converted with the registry's published ``area_mi2`` for a daily entity.
    ``None`` for families without a monthly-comparable store.  ``obs_mask`` (the run's
    ``DplConfig.obs_mask``) drops those days, so their months come out incomplete."""
    reg = pd.read_csv(paths.entities(data_dir),
                      dtype={"site_id": str}).set_index("entity_id")
    r = reg.loc[entity_id]
    if r["family"] == "uf_monthly":
        uf = pd.read_csv(paths.dwr_unimpaired(data_dir, "uf_monthly.csv"), parse_dates=["date"])
        s = uf[uf["uf"] == int(str(r["site_id"]).split()[-1])].set_index("date")["flow_taf"]
    elif r["family"] == "cdec_daily":
        path, col = str(r["obs_store"]).rsplit(":", 1)
        t = pd.read_csv(Path(data_dir) / path, parse_dates=["date"])
        key = "basin" if "basin" in t.columns else "station"
        t = t[t[key].astype(str) == str(r["site_id"])].set_index("date")[col].sort_index()
        t = t.where(t >= 0)
        drop = [pd.Timestamp(d) for e, _, d in (s.partition("|") for s in obs_mask)
                if e == entity_id]
        t = t.drop(drop, errors="ignore")
        m = t.resample("ME").sum(min_count=1)
        m[t.resample("ME").count().to_numpy() < m.index.days_in_month] = np.nan
        s = m * float(r["area_mi2"]) * AF_PER_MM_MI2 / 1000.0
    else:
        return None
    s.index = pd.DatetimeIndex(s.index).to_period("M")
    return s


def score_run(run_dir: str | Path, data_dir: str | Path = "data",
              holdout_wy: tuple[int, ...] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score one run.  Returns (metrics, monthly, panels): metrics has one row per
    set x reference kind x window (:func:`holdout_windows` and the entity's own ``train``
    window, which leaves the holdout out: column ``excluded_wy``); monthly holds the aligned
    sim/ref TAF series over the simulated record; panels holds the per-set series used by
    :func:`location_figure`.  ``holdout_wy``: default the run checkpoint's
    ``DplConfig.holdout_wy``."""
    sets = load_sets(data_dir)
    depth = load_run_monthly_depth(run_dir)
    obs_mask = _run_obs_mask(run_dir)
    ho = require_holdout(run_dir, "tier1") if holdout_wy is None else tuple(holdout_wy)
    ho_windows = holdout_windows(ho)
    inflow, unimp = load_references(data_dir)
    areas, ent_arcs, fp = arc_areas(data_dir), registry_arcs(data_dir), footprint_areas(data_dir)
    train_win = registry_windows(data_dir)
    span = depth.index
    rows, monthly, panels = [], [], {}
    for st in sets.itertuples(index=False):
        ent = st.entity_id
        if ent not in depth.columns:
            print(f"tier1: {st.set_id}: entity {ent} is not in this run — skipped")
            continue
        e_arcs = ent_arcs[ent]
        a_set = sum(areas.get(a, 0.0) for a in st.arcs)
        a_sim = sum(areas.get(a, 0.0) for a in e_arcs)
        if a_sim <= 0 or abs(a_sim - fp[ent]) / fp[ent] > 0.005:
            print(f"tier1: {st.set_id}: polygon area {a_sim:.1f} vs cell footprint "
                  f"{fp[ent]:.1f} mi2 — using the cell footprint")
            a_sim = fp[ent]
        covered = [a for a in st.arcs if a in e_arcs]
        missing = [a for a in st.arcs if a not in e_arcs]
        cover_frac = sum(areas.get(a, 0.0) for a in covered) / a_set
        sim = depth[ent] * a_sim * AF_PER_MM_MI2 / 1000.0            # TAF/month
        refs = {}
        if st.ref_kind == "anchor":
            refs["anchor"] = unimp[st.system]
        else:
            refs["arcsum"] = _arcsum(inflow, st.arcs, st.set_id)
        if missing:
            refs["arcsum_covered"] = _arcsum(inflow, covered, f"{st.set_id} (covered arcs)")
        t0, t1 = train_win[ent]
        windows = dict(ho_windows)
        windows["train"] = (str(t0), str(t1))
        for kind, ref in refs.items():
            for wname, (m0, m1) in windows.items():
                idx = pd.period_range(m0, m1, freq="M")
                s_w, r_w = sim.reindex(idx).to_numpy(), ref.reindex(idx).to_numpy()
                # the in-sample window never scores the held-out water years
                carve = holdout_month_mask(idx, ho) if wname == "train" else np.zeros(len(idx), bool)
                s_w, r_w = np.where(carve, np.nan, s_w), np.where(carve, np.nan, r_w)
                extra = {"excluded_wy": f"{ho[0]}-{ho[1]}" if carve.any() else ""}
                met = _score(idx, s_w, r_w)
                rows.append(dict(set_id=st.set_id, name=st.name, entity_id=ent, ref_kind=kind,
                                 system=st.system if kind == "anchor" else "", window=wname,
                                 win_start=m0, win_end=m1,
                                 volume_scored=bool(st.volume_scored) and kind != "arcsum_covered",
                                 nested_in=st.nested_in,
                                 area_set_mi2=a_set, area_sim_mi2=a_sim, cover_frac=cover_frac,
                                 n_arcs=len(st.arcs), n_arcs_missing=len(missing), **met,
                                 note=st.note, **extra))
            monthly.append(pd.DataFrame({"set_id": st.set_id, "ref_kind": kind,
                                         "month": span.astype(str),
                                         "sim_taf": sim.reindex(span).to_numpy(),
                                         "ref_taf": ref.reindex(span).to_numpy()}))
        main_kind = "anchor" if st.ref_kind == "anchor" else "arcsum"
        panels[st.set_id] = dict(
            row=st, kind=main_kind, sim=sim, ref=refs[main_kind].reindex(span), train=(t0, t1),
            covered=refs["arcsum_covered"].reindex(span) if missing else None,
            record=training_record_taf(ent, data_dir, obs_mask))
        vw = next(iter(ho_windows))
        panels[st.set_id].update(val_window=(vw, ho_windows[vw]), holdout_wy=ho)
    return pd.DataFrame(rows), pd.concat(monthly, ignore_index=True), panels


def _run_obs_mask(run_dir: str | Path) -> tuple[str, ...]:
    """The run's ``DplConfig.obs_mask`` from its best checkpoint (``()`` if none)."""
    ck = paths.run_roles(run_dir).model / "checkpoints" / "best.pt"
    if not ck.exists():
        return ()
    import torch

    cfg = torch.load(ck, map_location="cpu", weights_only=False).get("cfg") or {}
    return tuple(cfg.get("obs_mask", ()))


def location_figure(panel: dict, out: Path, run_label: str = "") -> None:
    """Per-location figure in the basin-diagnostics layout: the monthly volume series (validation
    window shaded, training window clear) with per-window stat boxes, and the two
    mean-monthly regimes.  The entity's own gauge record is overlaid where one exists,
    with its volume ratio to the CalSim3 reference in the legend; for a partial footprint
    the sum of the covered arcs is drawn dashed."""
    from ..._figures import _period_stats, _stat_text, plt
    st, sim, ref = panel["row"], panel["sim"], panel["ref"]
    idx = sim.index
    d = idx.to_timestamp()
    s, r = sim.to_numpy(), ref.to_numpy()
    kind = (f"FLOW-UNIMPAIRED {st.system}" if panel["kind"] == "anchor"
            else f"sum of {len(st.arcs)} INFLOW arcs")
    # the validation window is the run's held-out water years (score_run's panel)
    vname, vrange = panel["val_window"]
    v0, v1 = (pd.Period(x, "M") for x in vrange)
    c0, c1 = panel["train"]
    in_val = np.asarray((idx >= v0) & (idx <= v1))
    in_cal = np.asarray((idx >= c0) & (idx <= c1)) & ~holdout_month_mask(idx, panel["holdout_wy"])
    train_label = f"{c0}..{c1} minus WY{panel['holdout_wy'][0]}-{panel['holdout_wy'][1]}"
    val = _period_stats(s[in_val], r[in_val])
    cal = _period_stats(s[in_cal], r[in_cal])

    fig = plt.figure(figsize=(7.0, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
    fig.suptitle(f"{st.entity_id} — {st.name}\ndPL simulated vs CalSim3 {kind}"
                 + (f"   [{run_label}]" if run_label else ""), fontsize=8, fontweight="bold")
    ax = fig.add_subplot(gs[0, :])
    rec = panel.get("record")
    rec_label = None
    if rec is not None:
        rec = rec.reindex(idx).to_numpy()
        for wname, mask in (("validation", in_val), ("training window", in_cal)):
            k = mask & np.isfinite(rec) & np.isfinite(r)
            if k.sum() >= 12:
                rec_label = f"gauge record ({rec[k].sum() / r[k].sum():.2f}x CalSim3 over {wname})"
                break
        if rec_label is not None:
            ax.plot(d, rec, color="tab:green", lw=0.7, alpha=0.8, label=rec_label)
    ax.plot(d, r, color="0.25", lw=0.7, label=f"CalSim3 ({kind})")
    cov = panel.get("covered")
    if cov is not None:
        ax.plot(d, cov.to_numpy(), color="0.5", lw=0.7, ls="--", label="CalSim3, covered arcs only")
    ax.plot(d, s, color="tab:red", lw=0.7, alpha=0.8, label="dPL simulated")
    dmin, dmax = d.min(), d.max()
    ax.axvspan(dmin, c0.to_timestamp(), color="tab:blue", alpha=0.05)
    ax.axvline(c0.to_timestamp(), color="tab:blue", lw=1.2, ls="--")
    if c1.to_timestamp() < dmax:
        ax.axvspan(c1.to_timestamp(), dmax, color="tab:blue", alpha=0.05)
        ax.axvline(c1.to_timestamp(), color="tab:blue", lw=1.2, ls="--")
    ax.set_xlim(dmin, dmax)
    ax.set_ylabel("volume (TAF/month)")
    ax.set_title(f"Monthly volume — validation {vname} shaded, training window "
                 f"{train_label} clear", fontsize=8.5)
    ax.set_ylim(0, np.nanmax([np.nanmax(s), np.nanmax(r)]) * 1.5)
    # stat boxes in the two top corners; the single legend goes below the whole figure
    box = dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9)
    ax.text(0.01, 0.97, _stat_text(f"VAL {vname}", val), transform=ax.transAxes, va="top",
            ha="left", fontsize=8, family="monospace", bbox=box)
    ax.text(0.99, 0.97, _stat_text(f"CAL {train_label}", cal), transform=ax.transAxes, va="top",
            ha="right", fontsize=8, family="monospace", bbox=box)

    wy_months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x = np.arange(12)
    axes_rg = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    month = np.asarray(idx.month)
    for axrg, mask, lab in zip(axes_rg, (in_val, in_cal),
                               (f"validation {vname}", f"training {train_label}"), strict=True):
        fin = mask & np.isfinite(s) & np.isfinite(r)
        if fin.sum():
            def regime(y):
                out = []
                for mo in wy_months:
                    v = y[fin & (month == mo)]
                    out.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
                return out
            axrg.plot(x, regime(r), "o-", color="0.25", ms=3.5, label="CalSim3")
            if rec_label is not None and np.isfinite(rec[mask]).sum() >= 12:
                axrg.plot(x, regime(rec), "^-", color="tab:green", ms=3.0, label="gauge record")
            if cov is not None:
                axrg.plot(x, regime(cov.to_numpy()), "--", color="0.5", label="covered arcs")
            axrg.plot(x, regime(s), "s-", color="tab:red", ms=3.5, label="dPL")
        else:
            axrg.text(0.5, 0.5, "no reference", transform=axrg.transAxes,
                      ha="center", va="center", color="0.5")
        axrg.set_title(f"Regime — {lab}", fontsize=8)
        axrg.set_xlabel("water-year month")
        axrg.set_ylabel("volume (TAF/month)")
        axrg.set_xticks(x)
        axrg.set_xticklabels(list("ONDJFMAMJJAS"))
    top = max(a_.get_ylim()[1] for a_ in axes_rg)
    for a_ in axes_rg:
        a_.set_ylim(0, top)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _volume_text(g: pd.DataFrame) -> str:
    """``sim vs ref TAF/yr (bias)`` over the unnested rows of ``g``."""
    u = volume_rows(g)
    if not len(u) or not u.ref_taf_yr.sum():
        return "volume n/a"
    s, r = u.sim_taf_yr.sum(), u.ref_taf_yr.sum()
    return f"volume {s:,.0f} vs {r:,.0f} TAF/yr ({100.0 * (s / r - 1.0):+.1f}%)"


def summarize(metrics: pd.DataFrame, window: str) -> str:
    """Text summary: the volume-scored aggregate for ``window`` and the per-set table.  The
    skill statistics cover every volume-scored location; the volume totals leave out the
    nested ones (:func:`volume_rows`)."""
    lines = []
    m = metrics[(metrics.window == window)]
    v = m[m.volume_scored & m.ref_kind.isin(["anchor", "arcsum"])]
    nested = v[v["nested_in"].fillna("").astype(str) != ""] if "nested_in" in v.columns else v.iloc[0:0]
    lines.append(f"tier 1, {window}: {len(v)} volume-scored locations "
                 f"({(v.ref_kind == 'anchor').sum()} FLOW-UNIMPAIRED anchors, "
                 f"{(v.ref_kind == 'arcsum').sum()} arc sums)")
    if len(nested):
        lines.append("  volume totals exclude " + ", ".join(
            f"{r.set_id} (inside {r.nested_in})" for r in nested.itertuples(index=False))
            + ": nested, its water is already in the parent's")
    for label, g in (("all", v), ("anchors", v[v.ref_kind == "anchor"]),
                     ("arc sums", v[v.ref_kind == "arcsum"])):
        if len(g):
            lines.append(f"  {label:9s} KGE mean {g.kge.mean():.3f} median {g.kge.median():.3f} | "
                         f"NSE mean {g.nse.mean():.3f} | |pbias| median {g.pbias.abs().median():.1f}% | "
                         f"seasonal mismatch mean {g.seas_mismatch.mean():.3f} | " + _volume_text(g))
    cols = ["set_id", "ref_kind", "cover_frac", "n_months", "kge", "nse", "pbias", "r", "alpha",
            "beta", "seas_mismatch", "ct_diff", "sim_taf_yr", "ref_taf_yr"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        lines.append(m[cols].round(3).to_string(index=False))
    return "\n".join(lines)


def regime_figure(monthly: pd.DataFrame, metrics: pd.DataFrame, path: Path,
                  window: str, title: str = "") -> None:
    """Mean-monthly regime (Oct-Sep) per set over ``window``: reference vs simulation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m0, m1 = window_range(window)
    mon = monthly.copy()
    mon["period"] = pd.PeriodIndex(mon["month"], freq="M")
    mon = mon[(mon.period >= pd.Period(m0, "M")) & (mon.period <= pd.Period(m1, "M"))]
    mon["wm"] = (mon.period.dt.month - 10) % 12
    sets = list(dict.fromkeys(monthly.set_id))
    ncol = 5
    nrow = int(np.ceil(len(sets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
    met = metrics[metrics.window == window].set_index(["set_id", "ref_kind"])
    for ax, sid in zip(axes.ravel(), sets, strict=False):
        g = mon[mon.set_id == sid]
        main = g[g.ref_kind != "arcsum_covered"]
        reg = main.groupby("wm")[["sim_taf", "ref_taf"]].mean()
        kind = main.ref_kind.iloc[0]
        ax.plot(reg.index, reg.ref_taf, color="k", lw=1.8, label=f"CalSim3 ({kind})")
        ax.plot(reg.index, reg.sim_taf, color="tab:blue", lw=1.6, label="dPL")
        cov = g[g.ref_kind == "arcsum_covered"]
        if len(cov):
            rc = cov.groupby("wm")["ref_taf"].mean()
            ax.plot(rc.index, rc, color="0.5", lw=1.2, ls="--", label="CalSim3 (covered arcs)")
        r = met.loc[(sid, kind)]
        ax.set_title(f"{r.entity_id}  KGE {r.kge:.2f}  bias {r.pbias:+.0f}%  seas {r.seas_mismatch:.2f}",
                     fontsize=9)
        ax.set_xticks(range(12))
        ax.set_xticklabels([s[0] for s in _WY_MONTHS], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(sets):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(f"Tier 1 mean-monthly regime, {window} (TAF/month)  {title}", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None, prog=None) -> None:
    p = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", help="the run (any of its folders)")
    p.add_argument("--out", default=None,
                   help="one output folder for everything (default: tier1/ in the run's "
                        "results folder, the bulk in its local folder)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--label", default="", help="figure title suffix")
    p.add_argument("--no-figures", action="store_true",
                   help="skip the per-location figures (figures/<set_id>.png)")
    p.add_argument("--score-holdout", action="store_true",
                   help="score the run's held-out water years (protected_wy); without it "
                        "the run is refused")
    a = p.parse_args(argv)
    ho = require_holdout(a.run_dir, "tier1")
    refuse_protected(a.run_dir, a.score_holdout, "tier1")
    run = paths.run_roles(a.run_dir)
    out = Path(a.out) if a.out else run.results / "tier1"
    bulk = Path(a.out) if a.out else run.local / "tier1"
    out.mkdir(parents=True, exist_ok=True)
    bulk.mkdir(parents=True, exist_ok=True)
    label = a.label or run.results.name
    metrics, monthly, panels = score_run(a.run_dir, a.data_dir, holdout_wy=ho)
    metrics.to_csv(out / "tier1_metrics.csv", index=False)
    monthly.to_csv(bulk / "tier1_monthly.csv", index=False)
    # validated over the held-out water years (score_run)
    vwin = next(iter(holdout_windows(ho)))
    regime_figure(monthly, metrics, bulk / f"tier1_regime_{vwin}.png", window=vwin, title=label)
    n_fig = 1
    if not a.no_figures:
        (bulk / "figures").mkdir(exist_ok=True)
        for sid, panel in panels.items():
            location_figure(panel, bulk / "figures" / f"{sid}.png", run_label=label)
            n_fig += 1
    print(summarize(metrics, window=vwin))
    print(f"wrote {out / 'tier1_metrics.csv'}; tier1_monthly.csv and {n_fig} figures -> {bulk}")
