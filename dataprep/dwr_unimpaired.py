"""Ingest DWR's published Central Valley unimpaired + SWAT natural flows (WY1922-2014).

Source is the DWR Bay-Delta Office report *Estimates of Natural and Unimpaired
Flows for the Central Valley of California: WY 1922-2014* (DRAFT, March 2016),
the fifth edition of the series that began as *Central Valley Natural Flow Data*
(1980) and became *California Central Valley Unimpaired Flow Data* in 1987.
Three of its appendices are tables of monthly volumes in TAF, 93 water years
each; the report has no machine-readable release, so this transcribes the PDF's
text layer.

    Appendix B   24 Central Valley subbasins (UF 1-24) + 6 valley/Delta totals
                 -- DWR's **unimpaired** flow: measured flow adjusted to remove
                 upstream storage, diversion, import and export.
    Appendix C   18 of those subbasins -- the **SWAT-simulated** rim outflow the
                 report labels "natural".
    Appendix D   the published "Simulated minus Unimpaired" difference.

**At the rim, natural and unimpaired are the same quantity** -- do not read
Appendix C as a different physical variable from Appendix B.  The report's
natural-flow estimate departs from unimpaired only on the *valley floor*, where
C2VSim adds riparian/wetland ET, stream-groundwater interaction and bank
overflow (natural Delta inflow 21,533 vs unimpaired 29,003 TAF/yr, Table 5-2).
Upstream, ch. 5 is explicit: "Upper rim watersheds ... are relatively
undeveloped.  Precipitation-runoff processes are assumed to be unchanged from
natural condition for a given climate.  Therefore, simulated natural outflows
from these watersheds should be similar to estimates of unimpaired flows.  ...
the SWAT models used to simulate the upper rim watersheds were **calibrated to
match unimpaired flows**."

So `swat_monthly.csv` is an independent *model* estimate of the same quantity
`uf_monthly.csv` measures -- a fourth rainfall-runoff model over the rim
watersheds this repo calibrates against -- and the spread between the two is
SWAT calibration residual, not a natural-vs-unimpaired signal.  The executive
summary says as much: the differences "were found to be small and therefore do
not bias conclusions regarding differences between natural and unimpaired
flows."  Consistent with that, the widest gaps sit on the weakest-calibrated
models (Chowchilla 1.38x at NSE 0.76, Fresno 1.37x at NSE 0.71) rather than on
the most developed watersheds.

**These 24 subbasins are the calibration targets of the `9unimp`/`11obs`
domains.**  Verified, not assumed: `--verify` scores each unimpaired series
against `data/calsim/fnf_<domain>_monthly.csv` and gets r = 1.000000 on 16 of
the 18 mapped basins, with the mean ratio recovering the area DWR normalised by
(UF 6/BND: 0.8920 x 9978.2 = 8900 mi^2, the official Sacramento R. above Bend
Bridge drainage area; UF 8/FTO: 0.9906 x 3641.5 = 3607 mi^2, Feather R. at
Oroville).  The two that are not exact:

* **UF 3 / CacheCreek** matches exactly in 88 of 89 water years; only WY2010,
  the last year of this repo's record, departs.
* **UF 4 / BLB is NOT this series.**  The `11obs` BLB target spans 1994-2014 and
  correlates only loosely (annual r 0.07-0.997, ratio 0.63-6.2), i.e. it is a
  gauged Black Butte reservoir-inflow record, not DWR's unimpaired estimate.
  UF 4 maps exactly to `StonyCreek` (`9unimp`), the same watershed and the same
  CalSim3 arcs.  Both basins are carried in `uf_locations.csv`; only the
  `9unimp` one is a transcription of this table.

**Appendix C is checked against Appendix D**, which independently publishes
simulated - unimpaired.  17 of the 18 tables reconcile to +-1.04 TAF, the
rounding floor (D is integer, C carries one decimal): 13 exactly, and 5 after a
single constant factor -- UF 4 x1.0647, UF 9 x0.9107, UF 10 x1.0228,
UF 15 x0.9199, UF 22 x0.9460.  The report's stated mechanism is an "area ratio
factor ... applied to consider rainfall-runoff from small local drainage areas
located between a SWAT watershed outlet and its corresponding C2VSim stream
inflow node" (ch. 4).  Table 5-1 independently reproduces the *scaled* value for
UF 9, 10, 15 and 22 -- but the *raw* Appendix C value for UF 4 -- so which basis
a given table uses is not uniform across the report.  `swat_monthly.csv` stores
Appendix C as published; `uf_locations.csv.swat_scale_appendix_d` carries the
factor so either basis is recoverable.

Two tables are **not on the full-subbasin basis**, so their level must not be
compared with `uf_monthly.csv`:

* **UF 5** -- C-4 is captioned *"Sacramento Valley West Side Minor Streams
  (Thomes and Elder Creeks only)"*, and it is the one table that does not
  reconcile with Appendix D at any constant factor (implied ratio 1.22-2.00,
  volume ratio 0.64).
* **UF 7** -- C-5 averages 1169 TAF/yr against 1410 in the report's own Table
  5-1 for the same subbasin, a 17 % gap: the two are different aggregations of
  the east-side creek models (Table 5-1 sums Mill, Deer, Big Chico and a
  "Butte and Chico" node).  Appendix D differences the Appendix C aggregation,
  so C and D are mutually consistent; both are short of the subbasin.

Both are flagged `swat_partial` in `uf_locations.csv`.

Provenance of the SWAT runs (report Appendix A): 23 SWAT2009 models, daily,
delineated on 30 m DEM with 2001 USGS land use and STATSGO soils, driven by
Hamlet & Lettenmaier (2005) 1/8 degree data extended with 4 km PRISM, calibrated
and judged at **monthly** level against the unimpaired series -- so these are
model *fits* to Appendix B, not an independent observation of it.  Reported skill
spans NSE 0.67-0.91 / R^2 0.68-0.91, weakest on the minor streams and the Tulare
basin.  Six subbasins have no Appendix C table: the two valley floors (UF 1, 17),
the San Joaquin east-side minor streams (UF 12), Tulare Lake Basin outflow and
the San Joaquin west-side minor streams (UF 23, 24), and **UF 6** -- the SWAT
model there is Sacramento R. at Shasta, not the larger Red Bluff subbasin.

Mapping to CalSim3 is taken from the hand-edited `calsim_crosswalk.csv` via
`catchments.derive_basin_nodes` (so basin nesting applies: UF 6 / Bend Bridge
picks up Shasta's `I_SHSTA`), plus the series-less valley-accretion node
`I_SRBB_VAL`.  17 of the 24 subbasins resolve to a calibration basin and its
arcs; the other 7 (valley floors, minor-stream groups, Tulare Lake Basin
outflow) have **no** arcs written -- they cover real CalSim3 catchments but no
authoritative subbasin-to-arc assignment exists in this repo, and guessing one
would be fabrication.

Defects in the published tables, transcribed verbatim and flagged:

* Monthly values and the annual total are **rounded independently**, so months
  need not sum to the printed total.  In Appendix B 1085 of 2790 rows differ by
  <= 3 TAF.  Only the months are stored.
* **Tables C-5 (UF 7) and C-17 (UF 21) have a broken Total column** -- every row
  repeats that row's October value instead of the annual sum.
* **Table D-8 (UF 10) WY1922 drops a minus sign**: the total prints `119` where
  its months sum to -120 (neighbouring rows print negative totals correctly).
* In all three the MONTHLY values are sound -- the Appendix D reconciliation
  confirms them independently -- and only months are stored, so nothing
  propagates.  The gate whitelists these rows rather than relaxing its tolerance.
* **Table B-29 (Delta Unimpaired Total Outflow), WY2014 is corrupt**: all 12
  months print as `396` and sum to 4752 against a printed total of 10879.  It is
  stored as published; do not use that row.

Outputs (``data/dwr_unimpaired/``):

    uf_monthly.csv        date, uf, flow_taf     -- unimpaired, 24 subbasins
    swat_monthly.csv      date, uf, flow_taf     -- SWAT rim simulation, 18 subbasins
    delta_monthly.csv     date, series, flow_taf -- the 6 derived totals
    uf_locations.csv      uf -> name, CDEC id, calibration basin, CalSim3 arcs

Usage
-----
    python dataprep/dwr_unimpaired.py --pdf <report.pdf>
    python dataprep/dwr_unimpaired.py --verify
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sacsma.io import write_table  # noqa: E402

#: default location of the source PDF (not redistributed with the repo).
DEFAULT_PDF = (Path.home() / "Downloads" / "Estimates-of-Natural-and-Unimpaired-Flows"
               "-for-the-Central-Valley-of-California-1922-2014.pdf")
#: appendix -> (first page, last page + 1) in the PDF's own 0-based page index.
APPENDIX_PAGES = {"B": (101, 162), "C": (163, 199), "D": (200, 245)}
#: water years covered by every table.
WY_FIRST, WY_LAST = 1922, 2014
#: water-year month order (Oct..Sep), matching the tables' column order.
WY_MONTHS = (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)
#: months are integer TAF rounded independently of the printed annual total.
ROUNDING_TOL = 6.0
#: Appendix D is integer and Appendix C carries one decimal, so a reconciled
#: monthly difference can miss by a shade over 1 TAF on rounding alone.
RECONCILE_TOL = 1.5

_CAPTION = "Table\\s+({}-\\d+)\\.\\s*(.+?)\\s*$"
_ROW = re.compile(r"^((?:19|20)\d{2})\s+(.+?)\s*$", re.M)
_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")
_UF_IN_CAPTION = re.compile(r"UF\s*(\d+)")

#: the 6 derived tables B-25..B-30, and the series name each is written under.
DELTA_SERIES = {
    "B-25": "SAC_VALLEY_OUTFLOW", "B-26": "EASTSIDE_OUTFLOW",
    "B-27": "SJ_VALLEY_OUTFLOW", "B-28": "DELTA_INFLOW",
    "B-29": "DELTA_OUTFLOW", "B-30": "DELTA_NET_USE",
}
#: which subbasins compose each valley total (asserted in `_check`).
COMPOSITION = {"B-25": range(1, 12), "B-26": range(12, 16), "B-27": range(16, 25)}
#: published rows whose printed Total is broken.  ``None`` = the whole table.
#: C-5/C-17 repeat October instead of summing; B-29 WY2014 prints 12x396 against a
#: total of 10879; D-8 WY1922 prints ``119`` where its months sum to -120 (a dropped
#: minus sign -- neighbouring rows print negative totals correctly).  In every case
#: the MONTHLY values are sound, which the Appendix D reconciliation confirms.
BROKEN_TOTAL = {("C-5", None), ("C-17", None), ("B-29", 2014), ("D-8", 1922)}
#: Appendix C tables that are NOT on the full-subbasin basis -- their level must
#: not be compared with the unimpaired series.  Flagged in ``uf_locations.csv``.
SWAT_PARTIAL = {
    5: "Thomes and Elder Creeks only (per the C-4 caption); does not reconcile "
       "with Appendix D at any constant factor",
    7: "a different aggregation of the east-side creek models than the report's own "
       "Table 5-1 for UF 7 (1169 vs 1410 TAF/yr); consistent with Appendix D, but "
       "short of the subbasin",
}
#: ...of which only UF 5 fails the Appendix D reconciliation, so only it is
#: exempt from that gate.  UF 7 reconciles at k=1.0 (D differences the same
#: aggregation C publishes) and is still checked.
SWAT_NO_RECONCILE = {5}
#: expected area-ratio factor between Appendix C and the series Appendix D
#: differences, as a regression guard on the reconciliation.
SWAT_SCALE_EXPECTED = {4: 1.0647, 9: 0.9107, 10: 1.0228, 15: 0.9199, 22: 0.9460}

#: UF number -> published name, CDEC id, and calibration basin per domain.
#: ``None`` basin = a valley floor / minor-stream group / Tulare outflow with no
#: calibration basin and no authoritative arc assignment.
LOCATIONS: dict[int, tuple[str, str | None, str | None, str | None]] = {
    1:  ("Sacramento Valley Floor", None, None, None),
    2:  ("Putah Creek near Winters", None, None, "PutahCreek"),
    3:  ("Cache Creek above Rumsey", None, None, "CacheCreek"),
    4:  ("Stony Creek at Black Butte", None, "BLB", "StonyCreek"),
    5:  ("Sacramento Valley West Side Minor Streams", None, None, None),
    6:  ("Sacramento River near Red Bluff", "SBB", "BND", None),
    7:  ("Sacramento Valley East Side Minor Streams", None, None, None),
    8:  ("Feather River near Oroville", "FTO", "FTO", None),
    9:  ("Yuba River at Smartville", "YRS", "YRS", None),
    10: ("Bear River near Wheatland", None, None, "BearRiver"),
    11: ("American River at Fair Oaks", "AMF", "AMF", None),
    12: ("San Joaquin Valley East Side Minor Streams", None, None, None),
    13: ("Cosumnes River at Michigan Bar", "CSN", None, "CosumnesRiver"),
    14: ("Mokelumne River at Pardee Reservoir", "PAR", None, "MokelumneRiver"),
    15: ("Calaveras River at Jenny Lind", None, None, "CalaverasRiver"),
    16: ("Stanislaus River at Melones Reservoir", "SNS", "SNS", None),
    17: ("San Joaquin Valley Floor", None, None, None),
    18: ("Tuolumne River at Don Pedro Reservoir", "TLG", "TLG", None),
    19: ("Merced River at Exchequer Reservoir", "MRC", "MRC", None),
    20: ("Chowchilla River at Buchanan Reservoir", None, None, "ChowchillaRiver"),
    21: ("Fresno River near Daulton", None, None, "FresnoRiver"),
    22: ("San Joaquin River at Millerton Reservoir", "SJF", "SJF", None),
    23: ("Tulare Lake Basin Outflow", None, None, None),
    24: ("San Joaquin Valley West Side Minor Streams", None, None, None),
}
#: per-UF caveats worth carrying into the data.
NOTES = {
    4: "11obs BLB is a gauged Black Butte reservoir-inflow record, NOT this table; "
       "the 9unimp StonyCreek target IS this table",
    5: "SWAT table covers Thomes and Elder Creeks only - not comparable to the "
       "full-subbasin unimpaired series",
    6: "includes Shasta (I_SHSTA) and the series-less valley-accretion node I_SRBB_VAL; "
       "no SWAT table (that model is Sacramento R. at Shasta, a smaller watershed)",
    7: "SWAT table is a smaller aggregation of the east-side creek models than the "
       "report's own Table 5-1 for UF 7 - not comparable to the unimpaired series",
}
#: rim systems whose CalSim catchment includes a series-less valley-accretion node.
_VALLEY_NODE = {"BND": "I_SRBB_VAL"}


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def read_tables(pdf: Path) -> dict[str, dict[str, dict[int, tuple[list[float], float]]]]:
    """Parse all three appendices into ``{letter: {table: {wy: (months, total)}}}``."""
    from pypdf import PdfReader

    pages = [(p.extract_text() or "") for p in PdfReader(str(pdf)).pages]
    need = max(hi for _, hi in APPENDIX_PAGES.values())
    if len(pages) < need:
        raise ValueError(f"{pdf.name}: {len(pages)} pages, expected >= {need} "
                         "-- not the March 2016 WY1922-2014 report?")
    out: dict[str, dict] = {}
    for letter, (lo, hi) in APPENDIX_PAGES.items():
        cap_re = re.compile(_CAPTION.format(letter), re.M)
        tabs: dict[str, dict] = {}
        cur = None
        for text in pages[lo:hi]:
            caps = cap_re.findall(text)
            if caps:
                name, caption = caps[0]
                if not caption.rstrip().endswith("contd."):   # new table, not a carry-over
                    cur = name
                    d = tabs.setdefault(cur, {"caption": caption, "rows": {}})
                    # C-4's first page omits the "UF 5 -" prefix its second page carries
                    # (and that second page also omits "contd."), so prefer whichever
                    # caption actually identifies the subbasin.
                    if (_UF_IN_CAPTION.search(caption)
                            and not _UF_IN_CAPTION.search(d["caption"])):
                        d["caption"] = caption
            if cur is None:
                continue
            for m in _ROW.finditer(text):
                toks = _NUM.findall(m.group(2))
                if len(toks) != 13:                           # 12 months + total, else not data
                    continue
                v = [float(t.replace(",", "")) for t in toks]
                tabs[cur]["rows"][int(m.group(1))] = (v[:12], v[12])
        out[letter] = tabs
    return out


def _uf_of(tabs: dict) -> dict[int, str]:
    """UF number -> table name, read from each table's caption (every table must name one)."""
    out = {}
    for name, d in tabs.items():
        m = _UF_IN_CAPTION.search(d["caption"])
        if m is None:
            raise ValueError(f"{name}: caption names no UF subbasin ({d['caption']!r})")
        out[int(m.group(1))] = name
    if len(out) != len(tabs):
        raise ValueError(f"two tables claim the same UF subbasin: {sorted(tabs)}")
    return out


def _months(tabs: dict, table: str) -> np.ndarray:
    """One table's monthly values as a (93, 12) array in water-year order."""
    return np.array([tabs[table]["rows"][wy][0] for wy in range(WY_FIRST, WY_LAST + 1)])


def _check(ap: dict) -> dict[int, float]:
    """Assert coverage, the report's internal identities, and the C-vs-D reconciliation.

    Returns the area-ratio factor per UF that reconciles Appendix C to Appendix D.
    """
    want_wy = list(range(WY_FIRST, WY_LAST + 1))
    expect = {"B": 30, "C": 18, "D": 18}
    for letter, tabs in ap.items():
        if len(tabs) != expect[letter]:
            raise ValueError(f"Appendix {letter}: {len(tabs)} tables, expected {expect[letter]}")
        for name, d in tabs.items():
            if sorted(d["rows"]) != want_wy:
                miss = sorted(set(want_wy) - set(d["rows"]))
                raise ValueError(f"{name}: water years missing {miss}")
            for wy, (mon, tot) in d["rows"].items():
                if (name, None) in BROKEN_TOTAL or (name, wy) in BROKEN_TOTAL:
                    continue
                if abs(sum(mon) - tot) > ROUNDING_TOL:
                    raise ValueError(f"{name} WY{wy}: months sum to {sum(mon):.1f} but the "
                                     f"printed total is {tot:.1f}")

    ann = pd.DataFrame({n: {wy: sum(m) for wy, (m, _) in d["rows"].items()}
                        for n, d in ap["B"].items()})
    for tab, ufs in COMPOSITION.items():
        e = (ann[[f"B-{u}" for u in ufs]].sum(axis=1) - ann[tab]).abs()
        if e.max() > 12 * ROUNDING_TOL:
            raise ValueError(f"{tab} != sum of UF {list(ufs)} (max {e.max():.0f} TAF/yr)")
    e = (ann[["B-25", "B-26", "B-27"]].sum(axis=1) - ann["B-28"]).abs()
    if e.max() > 12 * ROUNDING_TOL:
        raise ValueError(f"B-28 != B-25+B-26+B-27 (max {e.max():.0f} TAF/yr)")

    cuf, duf = _uf_of(ap["C"]), _uf_of(ap["D"])
    if sorted(cuf) != sorted(duf):
        raise ValueError(f"Appendix C covers UF {sorted(cuf)} but D covers {sorted(duf)}")
    scale = {}
    for uf in sorted(cuf):
        if uf in SWAT_NO_RECONCILE:                  # different composition, no factor exists
            continue
        c, d = _months(ap["C"], cuf[uf]), _months(ap["D"], duf[uf])
        b = _months(ap["B"], f"B-{uf}")
        if float(np.abs(d - (c - b)).max()) <= RECONCILE_TOL:
            scale[uf] = 1.0                          # D differences Appendix C as published
            continue
        k = float((d + b).sum() / c.sum())           # else: a single area-ratio factor?
        err = float(np.abs(d - (k * c - b)).max())
        if err > RECONCILE_TOL:
            raise ValueError(f"UF {uf}: Appendix D != k*C - B (k={k:.4f}, max err "
                             f"{err:.2f} TAF) -- extraction or basis mismatch")
        want = SWAT_SCALE_EXPECTED.get(uf)
        if want is None:
            raise ValueError(f"UF {uf}: reconciles only after an unexpected area-ratio "
                             f"factor {k:.4f}")
        if abs(k - want) > 0.001:
            raise ValueError(f"UF {uf}: area-ratio factor {k:.4f}, expected {want:.4f}")
        scale[uf] = round(k, 4)
    missing = set(SWAT_SCALE_EXPECTED) - {u for u, v in scale.items() if v != 1.0}
    if missing:
        raise ValueError(f"UF {sorted(missing)} were expected to need an area-ratio "
                         "factor but reconciled without one")
    return scale


def to_long(rows: dict[int, tuple[list[float], float]], key: str, label) -> pd.DataFrame:
    """One table's rows -> long ``[date, <key>, flow_taf]`` on month-END dates."""
    recs = []
    for wy in sorted(rows):
        for m, v in zip(WY_MONTHS, rows[wy][0], strict=True):
            y = wy - 1 if m >= 10 else wy
            recs.append((pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0), label, v))
    return pd.DataFrame(recs, columns=["date", key, "flow_taf"])


# --------------------------------------------------------------------------
# CalSim3 mapping
# --------------------------------------------------------------------------
def build_locations(data_dir: str | Path = "data", *, swat: dict[int, float] | None = None,
                    swat_uf: set[int] | None = None) -> pd.DataFrame:
    """UF location -> calibration basin, CalSim3 arcs, CalSim catchment area, SWAT flags."""
    from sacsma.calsim.catchments import calsim_basin_areas, derive_basin_nodes

    nodes, areas = {}, {}
    for dom in ("11obs", "9unimp"):
        n = derive_basin_nodes(data_dir, dom)
        nodes[dom] = {b: list(dict.fromkeys(g["arc"].astype(str)))
                      for b, g in n.groupby("basin")}
        areas[dom] = calsim_basin_areas(data_dir, dom)

    swat, swat_uf = swat or {}, swat_uf or set()
    recs = []
    for uf, (name, cdec, b11, b9) in sorted(LOCATIONS.items()):
        arcs: list[str] = []
        area = np.nan
        for dom, basin in (("11obs", b11), ("9unimp", b9)):
            if basin is None:
                continue
            got = list(nodes[dom].get(basin, []))
            if v := _VALLEY_NODE.get(basin):
                got.append(v)
            if arcs and sorted(got) != sorted(arcs):
                raise ValueError(f"UF {uf}: 11obs/9unimp disagree on arcs "
                                 f"({sorted(arcs)} vs {sorted(got)})")
            arcs = got
            area = round(float(areas[dom].get(basin, np.nan)), 2)
        recs.append({
            "uf": uf, "table": f"B-{uf}", "name": name, "cdec_id": cdec or "",
            "basin_11obs": b11 or "", "basin_9unimp": b9 or "",
            "n_arcs": len(arcs), "arcs": ";".join(sorted(arcs)),
            "area_mi2_calsim": area,
            "has_swat": uf in swat_uf,
            "swat_scale_appendix_d": swat.get(uf, ""),
            "swat_partial": SWAT_PARTIAL.get(uf, ""),
            "note": NOTES.get(uf, ""),
        })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------
# Verification against the repo's own calibration targets
# --------------------------------------------------------------------------
def verify(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the unimpaired series against `fnf_<domain>_monthly.csv`, and summarise the
    SWAT fit to the unimpaired series it was calibrated against (the rim residual)."""
    from sacsma.io import load_basin_area, mmday_to_cfs

    d = Path(data_dir) / "dwr_unimpaired"
    uf = pd.read_csv(d / "uf_monthly.csv", parse_dates=["date"])
    sw = pd.read_csv(d / "swat_monthly.csv", parse_dates=["date"])
    loc = pd.read_csv(d / "uf_locations.csv")

    fnf_rows = []
    for _, r in loc.iterrows():
        for dom in ("11obs", "9unimp"):
            basin = r[f"basin_{dom}"]
            if not isinstance(basin, str) or not basin:
                continue
            fnf = pd.read_csv(Path(data_dir) / "calsim" / f"fnf_{dom}_monthly.csv",
                              parse_dates=["date"])
            fnf = fnf[fnf["basin"] == basin]
            ab = load_basin_area(data_dir, domain=dom)
            a = float(ab.loc[ab["basin"] == basin, "area_mi2"].iloc[0])
            taf = mmday_to_cfs(fnf["obs_mm"], a) * 1.9834711 / 1000.0
            j = (uf[uf["uf"] == r["uf"]].set_index("date")["flow_taf"].rename("uf")
                 .to_frame().join(fnf.assign(taf=taf).set_index("date")["taf"],
                                  how="inner").dropna())
            fnf_rows.append({
                "uf": r["uf"], "basin": basin, "domain": dom, "n": len(j),
                "r": round(float(np.corrcoef(j["uf"], j["taf"])[0, 1]), 6),
                "ratio": round(j["uf"].sum() / j["taf"].sum(), 4),
                "implied_area_mi2": round(a * j["uf"].sum() / j["taf"].sum(), 1),
            })

    swat_rows = []
    for u in sorted(sw["uf"].unique()):
        j = (sw[sw["uf"] == u].set_index("date")["flow_taf"].rename("nat").to_frame()
             .join(uf[uf["uf"] == u].set_index("date")["flow_taf"].rename("unimp"),
                   how="inner").dropna())
        row = loc[loc["uf"] == u].iloc[0]
        swat_rows.append({
            "uf": u, "name": row["name"][:34],
            "swat_taf_yr": round(j["nat"].sum() / 93, 1),
            "unimp_taf_yr": round(j["unimp"].sum() / 93, 1),
            "swat_over_unimp": round(j["nat"].sum() / j["unimp"].sum(), 3),
            "r_monthly": round(float(np.corrcoef(j["nat"], j["unimp"])[0, 1]), 3),
            "partial": str(row["swat_partial"]) not in ("", "nan"),
        })
    return pd.DataFrame(fnf_rows), pd.DataFrame(swat_rows)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap_.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="source report PDF")
    ap_.add_argument("--data-dir", default="data")
    ap_.add_argument("--verify", action="store_true",
                     help="score the written tables against the repo's FNF targets")
    args = ap_.parse_args(argv)

    out_dir = Path(args.data_dir) / "dwr_unimpaired"
    if args.verify:
        fnf, swat = verify(args.data_dir)
        print("Unimpaired vs the repo's FNF calibration targets")
        print(fnf.to_string(index=False))
        print("\nSWAT rim simulation vs the unimpaired series it was calibrated to "
              "(WY1922-2014)\n-- the spread is calibration residual, not a "
              "natural-vs-unimpaired signal")
        print(swat.to_string(index=False))
        return 0

    if not args.pdf.exists():
        ap_.error(f"source PDF not found: {args.pdf}\n"
                  "Pass --pdf; the report is not redistributed with this repo.")
    ap = read_tables(args.pdf)
    scale = _check(ap)
    print(f"Appendix B/C/D: {[len(ap[k]) for k in 'BCD']} tables x "
          f"{WY_LAST - WY_FIRST + 1} water years; identities hold, "
          f"C reconciles to D ({len(scale)} UF, {sum(v != 1.0 for v in scale.values())} "
          "via an area-ratio factor)")

    cuf = _uf_of(ap["C"])
    monthly = pd.concat([to_long(ap["B"][f"B-{u}"]["rows"], "uf", u)
                         for u in sorted(LOCATIONS)], ignore_index=True)
    swat = pd.concat([to_long(ap["C"][t]["rows"], "uf", u) for u, t in sorted(cuf.items())],
                     ignore_index=True)
    delta = pd.concat([to_long(ap["B"][t]["rows"], "series", s)
                       for t, s in DELTA_SERIES.items()], ignore_index=True)
    loc = build_locations(args.data_dir, swat=scale, swat_uf=set(cuf))

    write_table(monthly.sort_values(["uf", "date"]), out_dir / "uf_monthly.csv")
    write_table(swat.sort_values(["uf", "date"]), out_dir / "swat_monthly.csv")
    write_table(delta.sort_values(["series", "date"]), out_dir / "delta_monthly.csv")
    write_table(loc, out_dir / "uf_locations.csv")
    print(f"uf_monthly.csv     {len(monthly):6d} rows  {monthly['uf'].nunique()} subbasins  "
          f"{monthly['date'].min().date()}..{monthly['date'].max().date()}")
    print(f"swat_monthly.csv   {len(swat):6d} rows  {swat['uf'].nunique()} subbasins")
    print(f"delta_monthly.csv  {len(delta):6d} rows  {len(DELTA_SERIES)} series")
    print(f"uf_locations.csv   {len(loc):6d} rows  "
          f"{int((loc['n_arcs'] > 0).sum())} mapped to CalSim3 arcs "
          f"({loc['n_arcs'].sum()} arcs total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
