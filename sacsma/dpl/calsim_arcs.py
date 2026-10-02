"""CalSim3 rim-arc target set: record masks, volume basis and water-year closure.

The CalSim3 rim ``INFLOW`` arcs as a training / scoring target, built on three committed
tables (all written by ``dataprep/build_calsim_arcs.py``):

* ``data/calsim/arc_hierarchy.csv`` — one row per arc: ``SQ_MI``, containing anchors,
  FLOW-UNIMPAIRED system, residual (by-difference) flag, closure group, entity
  duplicates, training-USGS-gauge provenance, record tier (A-F) and ``train_default``;
* ``data/calsim/arc_obs_mask.csv`` — the arc-months that are an arc's OWN observed
  record inside the training water years (listed gauge period minus the report's
  correlation-extension years);
* ``data/calsim/calsim3_inflow_monthly_mm.csv`` — the INFLOW series as depth over each
  arc's ``SQ_MI`` (TAF -> mm with :data:`AF_PER_MM_MI2`, the tier-1/tier-2 constant).

The helpers here are torch-free: the builder, the ``calsim_monthly`` loader
(:mod:`sacsma.dpl.multi_timescale`) and the scoring side share them.

Volume basis.  A model arc or system depth becomes a volume with the CalSim3 catchment
area (``SQ_MI``; a system = the sum of its members), never with a footprint-overlap sum,
so model and CalSim3 volumes compare on one area convention (:func:`depth_to_taf`).

Water-year closure (:func:`close_water_years`), the rule adopted 2026-09-28: per
closure group and water year, the mismatch between the anchor (CalSim3
FLOW-UNIMPAIRED, or the training target where no FU exists) and the arc sum goes to the
group's by-difference residual arcs, spread over their months in proportion to their own
flow and floored at 0; what the residual arcs cannot absorb spills proportionally to the
other arcs of the group.  A group without residual arcs closes proportionally.  Sac R at
Bend Bridge (SRBB: its rim arcs sit ~973 TAF/yr under FU by design — the valley node
holds it) and the lake-routed ``uf_03`` Cache Creek are excluded; nested Shasta and
Whiskeytown close on their own.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .calsim_tier1 import AF_PER_MM_MI2

#: the dPL training water years (WY1976-85 is the holdout of every family; nothing
#: before WY1950) — inclusive (first, last) pairs
TRAIN_WY: tuple[tuple[int, int], ...] = ((1950, 1975), (1986, 2015))
HOLDOUT_WY = (1976, 1985)
#: "present" in a listed gauge period = the end of the CalSim3 historical record used here
PRESENT_WY = 2015
#: closure groups left out of the water-year closure, with the reason
CLOSURE_EXCLUDED = {
    "FU_SRBB": "srbb_valley_node_holds_the_gap",
    "uf_03": "lake_routed_anchor",
}
HIERARCHY_CSV = "arc_hierarchy.csv"
MASK_CSV = "arc_obs_mask.csv"
DEPTH_CSV = "calsim3_inflow_monthly_mm.csv"

_RANGE = re.compile(r"(\d{1,2})/(\d{2})\s*-\s*(?:(\d{1,2})/(\d{2})|(present))", re.I)
_YEARS = re.compile(r"^(\d{4})(?:\s*-\s*(\d{4}))?$")


# --------------------------------------------------------------------------- periods
def _yy(y: str) -> int:
    """Two-digit year of the report's MM/YY periods: 21-99 -> 19xx, 00-20 -> 20xx."""
    v = int(y)
    return 2000 + v if v <= 20 else 1900 + v


def water_year(months) -> np.ndarray:
    """Water year of each monthly Period (Oct-Sep, named by its September)."""
    p = pd.PeriodIndex(months, freq="M")
    return np.where(p.month >= 10, p.year + 1, p.year)


def parse_gauge_period(text) -> list[tuple[pd.Period, pd.Period]]:
    """Listed gauge period(s) of ``calsim3_arc_derivation.csv`` as inclusive monthly
    ranges.  Formats: ``MM/YY-MM/YY``, ``MM/YY-present`` (present = September of
    :data:`PRESENT_WY`), several ranges joined by ``and``, trailing notes ignored.
    Unparseable / 'none listed' / blank -> []."""
    if not isinstance(text, str):
        return []
    out = []
    for m0, y0, m1, y1, present in _RANGE.findall(text):
        start = pd.Period(year=_yy(y0), month=int(m0), freq="M")
        end = (pd.Period(year=PRESENT_WY, month=9, freq="M") if present
               else pd.Period(year=_yy(y1), month=int(m1), freq="M"))
        out.append((start, end))
    return out


def parse_extension_years(text) -> set[int]:
    """Water years named in the ``extension_period`` column (HR Tables 5-2/5-6 'Period
    of Data Extension'): ``YYYY - YYYY`` or ``YYYY`` items joined by ``;`` or ``|``,
    footnote markers like ``(fn 5)`` dropped."""
    if not isinstance(text, str) or not text.strip():
        return set()
    out: set[int] = set()
    for item in re.split(r"[;|]", re.sub(r"\([^)]*\)", "", text)):
        item = item.strip()
        if not item:
            continue
        m = _YEARS.match(item)
        if not m:
            raise ValueError(f"extension_period item {item!r} (in {text!r})")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        out.update(range(a, b + 1))
    return out


def train_wy_mask(wy: np.ndarray, train_wy=TRAIN_WY) -> np.ndarray:
    return np.logical_or.reduce([(wy >= a) & (wy <= b) for a, b in train_wy])


def in_record_months(period_text, extension_text=None, *, train_wy=TRAIN_WY,
                     carve_extension: bool = True) -> pd.PeriodIndex:
    """The arc's OWN observed months inside the training water years: the listed gauge
    period(s) minus every water year the report lists as a correlation extension
    (``carve_extension=False`` keeps the listed period whole)."""
    ranges = parse_gauge_period(period_text)
    if not ranges:
        return pd.PeriodIndex([], freq="M")
    months = pd.PeriodIndex(sorted({p for a, b in ranges
                                     for p in pd.period_range(a, b, freq="M")}), freq="M")
    wy = water_year(months)
    keep = train_wy_mask(wy, train_wy)
    if carve_extension:
        ext = parse_extension_years(extension_text)
        keep &= ~np.isin(wy, sorted(ext))
    return months[keep]


#: derivation classes whose listed gauge record belongs to a donor gauge, not to the arc
#: (``dataprep/build_calsim_arcs.py``: such arcs have no own-record months)
DONOR_CLASSES = ("gauged_extended_split", "proportioned_ungauged")


def own_record_months(data_dir: str | Path = "data", wy=HOLDOUT_WY) -> dict[str, pd.PeriodIndex]:
    """arc -> its OWN observed months inside the water years ``wy`` (first, last): the rule of
    ``arc_obs_mask.csv`` (a rim arc of the depth store whose derivation class is not a donor
    class; its listed gauge period minus the extension years, :func:`in_record_months`)
    applied to other water years — by default the holdout, which the mask leaves out.  Arcs
    without such months are absent."""
    der = pd.read_csv(Path(data_dir) / "calsim" / "calsim3_arc_derivation.csv").set_index("arc")
    hier = load_hierarchy(data_dir)
    own = hier[(hier["status"] == "rim_arc") & ~hier["method_class"].isin(DONOR_CLASSES)
               & hier["arc"].isin(der.index)]["arc"]
    out = {}
    for arc in own:
        ms = in_record_months(der.loc[arc, "period"], der.loc[arc, "extension_period"],
                              train_wy=(tuple(int(v) for v in wy),))
        if len(ms):
            out[arc] = ms
    return out


# --------------------------------------------------------------------------- volume basis
def taf_to_mm(taf, sq_mi):
    """TAF/month over ``sq_mi`` -> mm/month (the tier-1/tier-2 AF-per-mm-mi2 constant)."""
    return np.asarray(taf, dtype=np.float64) * 1000.0 / (np.asarray(sq_mi, dtype=np.float64)
                                                           * AF_PER_MM_MI2)


def mm_to_taf(mm, sq_mi):
    """mm/month over ``sq_mi`` -> TAF/month."""
    return np.asarray(mm, dtype=np.float64) * np.asarray(sq_mi, dtype=np.float64) \
        * AF_PER_MM_MI2 / 1000.0


def load_hierarchy(data_dir: str | Path = "data") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "calsim" / HIERARCHY_CSV)


def arc_areas(hier: pd.DataFrame) -> pd.Series:
    """``SQ_MI`` per arc (arcs without a CalSim3_Merged polygon are absent)."""
    h = hier[hier["sq_mi"].notna()]
    return h.set_index("arc")["sq_mi"].astype(float)


def closure_members(hier: pd.DataFrame) -> dict[str, list[str]]:
    """closure group -> its member arcs that carry a series (hierarchy order)."""
    h = hier[hier["closure_group"].notna() & hier["has_series"]]
    return {g: list(s["arc"]) for g, s in h.groupby("closure_group", sort=True)}


def depth_to_taf(depth_mm: pd.DataFrame, area_mi2) -> pd.DataFrame:
    """Monthly model depths (mm/month; columns = arcs or systems) -> TAF/month with each
    column's CalSim3 area (a Series/dict keyed by column, or one scalar)."""
    if np.isscalar(area_mi2):
        a = np.full(depth_mm.shape[1], float(area_mi2))
    else:
        a = np.array([float(area_mi2[c]) for c in depth_mm.columns])
    return pd.DataFrame(mm_to_taf(depth_mm.to_numpy(np.float64), a[None, :]),
                        index=depth_mm.index, columns=depth_mm.columns)


def monthly_depth_from_daily(daily_mm: pd.DataFrame) -> pd.DataFrame:
    """Daily depth (DatetimeIndex rows) -> complete calendar-month sums (PeriodIndex);
    incomplete months are NaN."""
    idx = pd.DatetimeIndex(daily_mm.index)
    per = idx.to_period("M")
    s = daily_mm.groupby(per).sum(min_count=1)
    n = pd.Series(1, index=idx).groupby(per).sum()
    full = n.to_numpy() == s.index.days_in_month
    s.loc[~full] = np.nan
    return s


def load_anchor_taf(data_dir: str | Path, hier: pd.DataFrame) -> pd.DataFrame:
    """Each closure group's anchor series, TAF/month (columns = closure groups): CalSim3
    FLOW-UNIMPAIRED for the ``FU_*`` groups, the DWR published unimpaired monthly (UF
    table) for the entity groups whose ``closure_anchor`` names one."""
    fu = pd.read_csv(Path(data_dir) / "calsim" / "calsim_unimpaired_monthly.csv",
                     parse_dates=["date"]).pivot(index="date", columns="system", values="flow_taf")
    fu.index = pd.PeriodIndex(fu.index, freq="M")
    uf = pd.read_csv(Path(data_dir) / "dwr_unimpaired" / "uf_monthly.csv", parse_dates=["date"])
    uf = uf.pivot(index="date", columns="uf", values="flow_taf")
    uf.index = pd.PeriodIndex(uf.index, freq="M")
    out = {}
    for g, ref in (hier.dropna(subset=["closure_group"])
                   .drop_duplicates("closure_group")[["closure_group", "closure_anchor"]]
                   .itertuples(index=False)):
        kind, _, key = str(ref).partition(":")
        if kind == "calsim_unimpaired":
            out[g] = fu[key]
        elif kind == "uf_monthly":
            out[g] = uf[int(key)]
    return pd.DataFrame(out)


def load_depth_store(data_dir: str | Path = "data") -> pd.DataFrame:
    """The arc depth store as a wide frame (monthly PeriodIndex x arcs), mm/month."""
    t = pd.read_csv(Path(data_dir) / "calsim" / DEPTH_CSV, parse_dates=["date"])
    w = t.pivot(index="date", columns="arc", values="depth_mm")
    w.index = pd.PeriodIndex(w.index, freq="M")
    return w


def load_arc_mask(data_dir: str | Path = "data") -> dict[str, pd.PeriodIndex]:
    """arc -> its trainable in-record months."""
    t = pd.read_csv(Path(data_dir) / "calsim" / MASK_CSV, parse_dates=["date"])
    return {a: pd.PeriodIndex(g["date"], freq="M") for a, g in t.groupby("arc", sort=False)}


# --------------------------------------------------------------------------- closure
def close_water_years(arc_taf: pd.DataFrame, anchor_taf: pd.DataFrame, hier: pd.DataFrame,
                      *, wy_range: tuple[int, int] = (1950, 2015), mode: str = "residual",
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close each group's arcs to its anchor over every complete water year.

    ``arc_taf`` / ``anchor_taf``: TAF/month, rows = monthly PeriodIndex, columns = arcs /
    closure groups.  ``mode="residual"`` is the user's rule (module docstring);
    ``"proportional"`` scales every member by anchor / arc sum.  Only water years in
    ``wy_range`` with all 12 months finite for the anchor and every member are closed;
    other months and arcs outside a closure group come back unchanged.

    Returns ``(closed, diag)``: the closed copy of ``arc_taf`` and one row per (group,
    water year) with ``arcs_taf`` / ``anchor_taf`` (water-year sums), ``mismatch_taf``
    (arcs - anchor), ``residual_taf`` (what the residual arcs carried), ``absorbed_taf`` /
    ``spilled_taf`` (the anchor-minus-arcs change taken by residual / other arcs),
    ``residual_short`` (the residual arcs hit their floor) and ``closed_err_taf``."""
    if mode not in ("residual", "proportional"):
        raise ValueError(f"mode {mode!r}")
    closed = arc_taf.copy()
    idx = pd.PeriodIndex(arc_taf.index, freq="M")
    wy_all = water_year(idx)
    resid = set(hier.loc[hier["residual"].astype(bool), "arc"])
    rows = []
    for g, members in closure_members(hier).items():
        if g in CLOSURE_EXCLUDED or g not in anchor_taf.columns:
            continue
        mem = [a for a in members if a in arc_taf.columns]
        if not mem:
            continue
        res_arcs = [a for a in mem if a in resid] if mode == "residual" else []
        oth_arcs = [a for a in mem if a not in res_arcs]
        anc = anchor_taf[g].reindex(idx)
        for y in range(wy_range[0], wy_range[1] + 1):
            sel = wy_all == y
            if sel.sum() != 12:
                continue
            block = arc_taf.loc[sel, mem].to_numpy(np.float64)
            a_m = anc.to_numpy(np.float64)[sel]
            if not (np.isfinite(block).all() and np.isfinite(a_m).all()):
                continue
            A, F = float(block.sum()), float(a_m.sum())
            D = F - A                                   # what the arcs must gain
            new = block.copy()
            jr = [mem.index(a) for a in res_arcs]
            jo = [mem.index(a) for a in oth_arcs]
            rtot = float(block[:, jr].sum()) if jr else 0.0
            otot = float(block[:, jo].sum()) if jo else 0.0
            absorbed, spilled = 0.0, 0.0
            # the residual arcs cannot take the whole mismatch: it exceeds what they
            # carry (floor binds), or they carry nothing to spread it over
            short = bool(jr) and (rtot + D < 0.0 or (rtot <= 0.0 and D != 0.0))
            if jr and rtot > 0:
                f = max(1.0 + D / rtot, 0.0)            # proportional over months, floor 0
                new[:, jr] = block[:, jr] * f
                absorbed = rtot * (f - 1.0)
            rem = D - absorbed
            if jo and rem != 0.0 and otot > 0:
                new[:, jo] = block[:, jo] * (1.0 + rem / otot)
                spilled = rem
            closed.loc[sel, mem] = new
            rows.append(dict(group=g, wy=y, n_arcs=len(mem), n_residual=len(jr),
                             arcs_taf=A, anchor_taf=F, mismatch_taf=A - F,
                             residual_taf=rtot, absorbed_taf=absorbed, spilled_taf=spilled,
                             residual_short=bool(short),
                             closed_err_taf=float(new.sum()) - F))
    return closed, pd.DataFrame(rows)
