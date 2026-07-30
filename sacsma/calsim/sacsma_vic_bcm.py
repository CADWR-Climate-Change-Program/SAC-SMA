"""Three-way model comparison — SAC-SMA vs VIC vs BCM — on WGEN Product A.

All three models are driven by the **same climate**: the CalSim3 stochastic
pipeline's historical-parallel sequence.  For SAC-SMA and VIC that is the
``wgen_product_a`` forcing product; for BCM it is Scenario 1 (Baseline) of the
USGS BCM v8 run on the CalSim3 Weather Generator scenarios, which is that same
sequence.  The reference is unchanged from the rest of the cross-compare — the
CalSim3 unimpaired FNF anchor (``FLOW-UNIMPAIRED`` for rim systems, else the sum
of the basin's ``INFLOW`` sub-arcs) — so this scores three independent
hydrologic models against one target on one climate.

Scope, all three fixed by the study request:

* **Period** — the most recent 30 water years fully covered by all three,
  **WY1989–WY2018** (:data:`WY_RANGE`).  BCM ends 2018-09, which is exactly
  WY2018, and is the binding constraint.
* **Basins** — the CalLite calibration domains ``11obs`` and ``9unimp`` pooled
  into ONE combined set of 19, not scored separately.  ``BLB`` (11obs) and
  ``StonyCreek`` (9unimp) are the same watershed — the same three CalSim arcs
  ``I_BLKBT``/``I_EPARK``/``I_SGRGE`` — so the 11obs copy is dropped and the
  9unimp one kept (:data:`DUPLICATE_BASINS`).
* **Output** — ``artifacts/calsim/compare/`` alongside the two-way anchor
  artifacts, under the ``sacsma_vic_bcm_`` prefix.

**How each model's basin volume is built**, and why they are comparable:

* *SAC-SMA* — the official screened-footprint anchor
  (:func:`~.compare.build_anchor_long` with the ``wgen_product_a`` forcing):
  full calibrated footprint everywhere except
  :data:`~.catchments.SCREENED_BASINS`, times the canonical CalSim3 catchment
  area.
* *BCM* — total discharge ``run + rch`` (the repo's established BCM discharge
  convention) on **the CalSim3 catchments themselves**: the per-polygon
  aggregation ``dataprep/bcm_region.py`` already produces, area-weighted over
  the catchments each basin owns, which sum to exactly that same canonical
  area.  No footprint and no screening enter — the endorheic Goose Lake block
  is its own polygon belonging to no rim catchment, so it is excluded by
  construction (:func:`bcm_catchment_nodes`).
* *VIC* — the routed monthly Product A series, per basin on the anchor's own
  VIC convention (one 8-River series for a rim basin, else the sum of its
  per-node series).

Note BCM is a water-balance model with no channel routing, so at a monthly step
its ``run + rch`` is water *generated*, not water *arriving*.  Over a 30-year
monthly comparison that difference is small, but it is a real reason to read
BCM's month-to-month timing more loosely than the two routed models'.

Entry point: :func:`make_all` (``sacsma calsim --sacsma-vic-bcm``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: water-year range of the comparison (inclusive), and the calendar window it implies.
WY_RANGE = (1989, 2018)

#: the pooled comparison set — both CalLite calibration domains, scored as one.
COMBINED_SETS = ("11obs", "9unimp")

#: basins dropped when pooling because another set already carries the same
#: watershed.  ``BLB`` (11obs) and ``StonyCreek`` (9unimp) are both Stony Creek
#: at Black Butte — identical CalSim arcs — so one has to go; the request keeps
#: the 9unimp copy.
DUPLICATE_BASINS = {"11obs": ("BLB",)}

#: the forcing product all three models share.
PRODUCT = "wgen_product_a"

#: BCM scenario driven by that same WGEN sequence (Scenario 1, Baseline).
BCM_SCENARIO = "s01"

#: BCM total discharge = fast runoff + recharge (see ``dataprep/README.md``).
BCM_DISCHARGE = ("run", "rch")

#: the three models plus the reference, in plot order.
MODELS = ("sac", "vic", "bcm")

#: figure palette — VIC and CalSim3 keep their cross-compare colours
#: (:data:`~.compare._COLORS`); SAC-SMA takes the 11obs orange since most pooled
#: basins come from that set, and BCM gets a new teal.
COLORS = {"sac": "#d95f0e", "vic": "#984ea3", "bcm": "#1b9e77", "calsim3": "#111111"}
LABELS = {"sac": "SAC-SMA", "vic": "VIC", "bcm": "BCM", "calsim3": "CalSim3 FNF"}

#: dumbbell marker per model — shape carries the model as well as colour, so the
#: skill figure stays readable in greyscale.  VIC keeps the cross-compare's
#: diamond (:func:`~.compare._anchor_dumbbell_fig`).
MARKERS = {"sac": "o", "vic": "D", "bcm": "s"}
MARKER_SIZE = {"sac": 30, "vic": 24, "bcm": 24}


def wy_window(wy_range=WY_RANGE) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calendar bounds of a water-year range, as month-end timestamps.

    WY *n* runs 1 Oct *n-1* .. 30 Sep *n*, so WY1989–2018 is 1988-10-31 ..
    2018-09-30 on the month-end index the anchor tables use.
    """
    a, b = wy_range
    return (pd.Timestamp(f"{a - 1}-10-31"), pd.Timestamp(f"{b}-09-30"))


def water_year(dates) -> np.ndarray:
    """Water year of a datetime-like series (Oct–Dec belong to the next WY)."""
    d = pd.DatetimeIndex(dates)
    return (d.year + (d.month >= 10).astype(int)).to_numpy()


# --------------------------------------------------------------------------
# BCM on the anchor's own footprint
# --------------------------------------------------------------------------
def bcm_catchment_nodes(data_dir: str | Path = "data") -> pd.DataFrame:
    """Assign every BCM catchment to its ``CalSim3_Merged`` rim node.

    Returns ``[cid, node_gl, node, arc, sq_mi]`` for the BCM catchments that sit
    inside a rim catchment (the rest — Tulare, the valley watersheds, Goose Lake
    — get no row and so enter no basin).

    ``dataprep/bcm_region.py`` aggregated BCM to the **``CalSim3_And_GooseLake``**
    layer (386 polygons, ``cid`` = row order), while the canonical basin areas and
    the screened footprint use **``CalSim3_Merged``** (200 polygons).  Merged is
    the *dissolve* of the rim part of And_GooseLake, so the two are the same
    geography at different aggregation — but they are NOT joinable on
    ``Connect_No``: Merged names each dissolved catchment for its CalSim INFLOW
    arc, so e.g. ``MCD021…MCD128`` become Merced's ``MCLRE``, ``TUO017/054/105``
    become Tuolumne's, ``PTH021+PTH024`` become ``PTH070``, and the Bend Bridge
    valley polygons become the synthetic ``SRBB_VAL``.  Matching on the name
    silently loses those basins, so the assignment is **geometric**: each
    sub-polygon goes to the merged polygon containing its representative point.

    Verified deterministic and exact — all 200 merged nodes are reconstructed
    from their members to within 2 % and the totals agree to 30 365.2 mi²
    (100.00 %).  Use a representative point, **not** overlap area: adjacent
    catchments share long boundaries, and a largest-overlap rule assigns polygons
    through slivers (it put the 14 452 mi² Tulare Lake Basin inside Millerton).

    ⚠ This is also why the catchment basis needs no Goose Lake screening: the
    endorheic block is its own polygon with a blank ``Connect_No``, so it is
    inside no rim catchment and joins to nothing.
    """
    import geopandas as gpd

    from .catchments import CALSIM_LAYER, MERGED_LAYER, load_catchments, series_arc

    gl = load_catchments(data_dir, layer=CALSIM_LAYER, rim_only=False)
    mg = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True)
    gl = gl.to_crs(3310)                          # equal-area, for a stable point-in-polygon
    mg = mg.to_crs(3310)
    pts = gl[["cid", "node", "sq_mi"]].copy()
    pts = gpd.GeoDataFrame(pts, geometry=gl.geometry.representative_point(), crs=gl.crs)
    j = gpd.sjoin(pts, mg[["node", "geometry"]].rename(columns={"node": "node_m"}),
                  how="inner", predicate="within").drop_duplicates("cid")
    out = pd.DataFrame({
        "cid": j["cid"].to_numpy(),
        "node_gl": j["node"].astype(str).to_numpy(),
        "node": j["node_m"].astype(str).to_numpy(),
        "sq_mi": j["sq_mi"].to_numpy(dtype=float),
    })
    out["arc"] = [series_arc(n) for n in out["node"]]
    return out.sort_values("cid").reset_index(drop=True)


def bcm_basin_catchments(data_dir: str | Path = "data", domain: str = "11obs") -> pd.DataFrame:
    """``[basin, cid, sq_mi]`` — the BCM catchments each anchor basin owns.

    Exactly the catchment set :func:`~.catchments.calsim_basin_areas` sums for the
    canonical area: the basin's crosswalk arcs (already carrying the
    ``BASIN_NESTS`` nesting, so BND includes Shasta) plus the modelled
    valley-accretion node for the rim systems that have one.  The per-basin
    ``sq_mi`` total is checked against the canonical area, so a future GIS or
    crosswalk edit that broke the correspondence would raise rather than quietly
    re-weight the comparison.
    """
    from .catchments import (BASIN_RIM_SYSTEM, VALLEY_SYSTEMS, basin_areas,
                             derive_basin_nodes, valley_arc_for_system)

    cat = bcm_catchment_nodes(data_dir)
    by_arc = {a: g for a, g in cat.groupby("arc")}
    nodes = derive_basin_nodes(data_dir, domain)
    areas = basin_areas(data_dir, domain=domain)
    sysmap = BASIN_RIM_SYSTEM.get(domain, {})
    parts = []
    for basin, g in nodes.groupby("basin"):
        arcs = list(dict.fromkeys(g["arc"].astype(str)))
        sysn = sysmap.get(str(basin))
        if sysn in VALLEY_SYSTEMS:
            arcs.append(valley_arc_for_system(sysn))
        sub = [by_arc[a] for a in arcs if a in by_arc]
        if not sub:
            continue                                       # no CalSim catchment (Tulare/Kern)
        d = pd.concat(sub, ignore_index=True).drop_duplicates("cid")
        got, want = d["sq_mi"].sum(), areas[str(basin)]
        if abs(got - want) > 0.02 * max(want, 1.0):
            raise ValueError(
                f"bcm_basin_catchments[{domain}/{basin}]: BCM catchments sum to "
                f"{got:.2f} mi2 but the canonical area is {want:.2f} mi2 — the "
                f"CalSim3 GIS layers and the crosswalk have diverged")
        parts.append(pd.DataFrame({"basin": str(basin), "cid": d["cid"].to_numpy(),
                                   "sq_mi": d["sq_mi"].to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def bcm_basin_monthly(data_dir: str | Path = "data", domain: str = "11obs", *,
                      scenario: str = BCM_SCENARIO,
                      variables=BCM_DISCHARGE) -> pd.DataFrame:
    """BCM basin discharge as monthly TAF — ``[date, basin, flow_taf]``.

    Built **directly on the CalSim3 catchments** (``bcm_<scenario>_catchments_
    monthly.csv``, the per-polygon aggregation ``dataprep/bcm_region.py`` already
    produces), not on a grid re-aggregation: each basin's volume is the
    area-weighted ``run + rch`` depth over the catchments it owns, times the same
    canonical CalSim3 area SAC-SMA's anchor uses.  Since those catchments sum to
    exactly that area (:func:`bcm_basin_catchments` enforces it), the volume is
    simply the sum of each catchment's own depth times its own area.

    This puts BCM on the true watershed rather than on a footprint, which is both
    the natural basis for an uncalibrated gridded model and the reason no Goose
    Lake screening is needed here — see :func:`bcm_catchment_nodes`.
    """
    from ..io import mmday_to_cfs
    from .compare import _cfs_day_to_taf

    path = Path(data_dir) / "region" / "bcm" / f"bcm_{scenario}_catchments_monthly.csv"
    raw = pd.read_csv(path, usecols=["cid", "month", *variables])
    raw["depth"] = sum(raw[v] for v in variables)
    dates = pd.to_datetime(raw["month"].astype(str), format="%Y%m").dt.to_period("M")
    raw["date"] = dates.dt.to_timestamp("M")

    own = bcm_basin_catchments(data_dir, domain)
    j = raw.merge(own, on="cid", how="inner")
    if j["depth"].isna().any():
        n = int(j["depth"].isna().sum())
        raise ValueError(f"bcm_basin_monthly[{domain}]: {n} NaN catchment-months in "
                         f"{path.name}")
    # volume-weighted: sum(depth_c * area_c) == area-weighted depth * basin area
    j["mm_mi2"] = j["depth"] * j["sq_mi"]
    agg = j.groupby(["basin", "date"], as_index=False).agg(
        mm_mi2=("mm_mi2", "sum"), area=("sq_mi", "sum"))
    agg["flow_taf"] = _cfs_day_to_taf(mmday_to_cfs(agg["mm_mi2"] / agg["area"],
                                                   agg["area"]))
    return agg[["date", "basin", "flow_taf"]].sort_values(
        ["basin", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# the combined three-model long table
# --------------------------------------------------------------------------
def build_long(data_dir: str | Path = "data", *, sets=COMBINED_SETS,
                        product: str = PRODUCT, scenario: str = BCM_SCENARIO,
                        parallel: bool = False) -> pd.DataFrame:
    """Long ``[date, set, basin, source, flow_taf, ref_kind]`` for the pooled set.

    ``source`` is one of ``sac``/``vic``/``bcm``/``calsim3``.  SAC-SMA and VIC come
    from the official anchor pipeline with **both** switched to ``product`` (the
    forcing for SAC-SMA, the matching routed table for VIC); CalSim3 is the
    forcing-independent reference.  Duplicated watersheds
    (:data:`DUPLICATE_BASINS`) are dropped here, so the result is the 19-basin
    combined set.

    NOTE this re-simulates every anchor basin under ``product`` — a few minutes.
    """
    from .compare import _screened_fp, build_anchor_long

    anchor = build_anchor_long(data_dir, sets, footprint=_screened_fp(data_dir, sets),
                               product=product, vic_product=product, parallel=parallel)
    anchor = anchor.copy()
    anchor["source"] = anchor["source"].where(~anchor["source"].isin(sets), "sac")

    ref = anchor[["set", "basin", "ref_kind"]].drop_duplicates()
    bcm = pd.concat(
        [bcm_basin_monthly(data_dir, dom, scenario=scenario).assign(set=dom) for dom in sets],
        ignore_index=True)
    bcm["source"] = "bcm"
    bcm = bcm.merge(ref, on=["set", "basin"], how="inner")

    long = pd.concat([anchor, bcm[anchor.columns]], ignore_index=True)
    drop = {(s, b) for s, bs in DUPLICATE_BASINS.items() for b in bs}
    mask = np.array([(s, b) not in drop for s, b in zip(long["set"], long["basin"])])
    long = long[mask].reset_index(drop=True)
    return long.sort_values(["set", "basin", "source", "date"]).reset_index(drop=True)


def skill_metrics(long: pd.DataFrame, *, wy_range=WY_RANGE) -> pd.DataFrame:
    """Per (basin, model) skill vs CalSim3 over ``wy_range``, on identical months.

    Every model is scored on the intersection of all four series inside the
    window, so the three-way head-to-head is on exactly the same months for each
    basin — and ``n_months`` is therefore shared across a basin's three rows.
    Also returns the mean volumes, so a reader can separate a bias in level from
    a bias in shape.
    """
    from ..metrics import kge, nse, pbias

    lo, hi = wy_window(wy_range)
    rows = []
    for (st, basin), g in long.groupby(["set", "basin"]):
        wide = g.pivot_table(index="date", columns="source", values="flow_taf")
        wide = wide[(wide.index >= lo) & (wide.index <= hi)]
        cols = [c for c in (*MODELS, "calsim3") if c in wide.columns]
        if "calsim3" not in cols:
            continue
        sub = wide[cols].dropna()
        if len(sub) < 12:
            continue
        ref = sub["calsim3"].to_numpy()
        for m in MODELS:
            if m not in sub:
                continue
            sim = sub[m].to_numpy()
            rows.append({
                "set": st, "basin": basin, "model": m,
                "ref_kind": str(g["ref_kind"].iloc[0]),
                "wy_start": wy_range[0], "wy_end": wy_range[1], "n_months": len(sub),
                "kge": kge(sim, ref), "nse": nse(sim, ref), "pbias": pbias(sim, ref),
                "r": float(np.corrcoef(sim, ref)[0, 1]),
                "mean_sim_taf": float(sim.mean()), "mean_calsim3_taf": float(ref.mean()),
            })
    return pd.DataFrame(rows)


def pooled_summary(met: pd.DataFrame) -> pd.DataFrame:
    """Pooled across basins: median/mean skill per model, plus the volume total.

    ``median_*`` is the headline (skill distributions across basins are skewed by
    the small dry creeks); ``mean_abs_pbias`` states the typical volume error
    without letting over- and under-prediction cancel.
    """
    rows = []
    for m, g in met.groupby("model"):
        rows.append({
            "model": m, "n_basins": len(g),
            "median_kge": g["kge"].median(), "mean_kge": g["kge"].mean(),
            "median_nse": g["nse"].median(),
            "median_pbias": g["pbias"].median(),
            "mean_abs_pbias": g["pbias"].abs().mean(),
            "median_r": g["r"].median(),
            "total_sim_taf": g["mean_sim_taf"].sum(),
            "total_calsim3_taf": g["mean_calsim3_taf"].sum(),
            "total_pbias": 100 * (g["mean_sim_taf"].sum() / g["mean_calsim3_taf"].sum() - 1),
            "n_kge_best": 0,
        })
    out = pd.DataFrame(rows).set_index("model")
    best = met.loc[met.groupby(["set", "basin"])["kge"].idxmax(), "model"].value_counts()
    out["n_kge_best"] = best.reindex(out.index).fillna(0).astype(int)
    return out.reset_index().sort_values("median_kge", ascending=False).reset_index(drop=True)


def _ordered_basins(data_dir, long_or_met) -> list[tuple[str, str]]:
    """Pooled basins as ``(set, basin)`` ordered north -> south across BOTH sets.

    The house order is per-domain (:func:`~.compare.basin_order_north_south`), but
    this set is pooled, so latitudes are compared across sets and the
    Folsom-before-Yuba override is applied to the merged list.
    """
    from .._figures import folsom_before_yuba
    from ..io import load_hru_table

    have = {(str(s), str(b)) for s, b in
            zip(long_or_met["set"], long_or_met["basin"])}
    lat = {}
    for dom in sorted({s for s, _ in have}):
        h = load_hru_table(data_dir, domain=dom)
        for basin, d in h.groupby("basin"):
            if (dom, str(basin)) in have:
                lat[(dom, str(basin))] = float(
                    np.average(d["lat"], weights=d["area_weight"]))
    order = sorted(lat, key=lambda k: -lat[k])
    names = folsom_before_yuba("11obs", [b for _, b in order])
    rank = {b: i for i, b in enumerate(names)}
    return sorted(order, key=lambda k: rank.get(k[1], len(rank)))


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
_W, _DPI = 6.5, 300           # house style: <=6.5in wide, 300 dpi, <=8pt text
_PBIAS_LIM = 75.0             # shared pbias scale, as elsewhere in the compare


def _label(basin: str) -> str:
    from .compare import _BASIN_ABBREV
    return _BASIN_ABBREV.get(basin, basin)


def _clip_marks(ax, x, vals, ylim, color):
    """Flag values pinned to an axis bound: ``v`` below the floor, ``^`` above the cap.

    The house rule fixes these axes (KGE 0–1, % bias ±75) so figures compare across
    the artifact, which means an off-scale value would otherwise vanish silently.
    """
    lo, hi = ylim
    for xi, v in zip(x, vals):
        if not np.isfinite(v):
            continue
        if v < lo:
            ax.plot([xi], [lo], marker="v", ms=4, color=color, mec="white", mew=0.4,
                    zorder=4, clip_on=False)
        elif v > hi:
            ax.plot([xi], [hi], marker="^", ms=4, color=color, mec="white", mew=0.4,
                    zorder=4, clip_on=False)


def skill_fig(met: pd.DataFrame, path, data_dir: str | Path = "data", *,
              wy_range=WY_RANGE) -> None:
    """Two panels — per-basin KGE and % bias vs CalSim3, as **vertical dumbbells**.

    One dumbbell per basin: a grey connector spanning the three models' values
    with a marker per model on it, the same idiom as
    :func:`~.compare._anchor_dumbbell_fig` — the connector makes the *spread*
    between models the thing you read first, and the gap for a basin where one
    model falls away is immediate.  Markers differ in shape as well as colour so
    the figure survives greyscale printing.

    KGE keeps the house full 0–1 axis and % bias the shared ±75 %, with values
    outside pinned to the bound and flagged (:func:`_clip_marks`).  Basins run
    north -> south across the pooled set; there are **no set dividers**, because
    the two calibration sets are pooled and the north–south order interleaves
    them, so a set boundary is not a place on this axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _ordered_basins(data_dir, met)
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(_W, 5.4), sharex=True)
    for ax, col, ylab, ylim in ((axes[0], "kge", "KGE", (0.0, 1.0)),
                                (axes[1], "pbias", "% bias vs CalSim3",
                                 (-_PBIAS_LIM, _PBIAS_LIM))):
        vals = {m: np.array([_get(met, s, b, m, col) for s, b in rows], dtype=float)
                for m in MODELS}
        for xi in x:                                   # connector first, under the markers
            v = [np.clip(vals[m][xi], *ylim) for m in MODELS
                 if np.isfinite(vals[m][xi])]
            if len(v) > 1:
                ax.plot([xi, xi], [min(v), max(v)], color="0.78", lw=1.5, zorder=1)
        for m in MODELS:
            ax.scatter(x, np.clip(vals[m], *ylim), s=MARKER_SIZE[m], marker=MARKERS[m],
                       color=COLORS[m], label=LABELS[m], zorder=3,
                       edgecolor="white", linewidth=0.4)
            _clip_marks(ax, x, vals[m], ylim, COLORS[m])
        ax.set_ylim(*ylim)
        if ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlim(-0.7, len(rows) - 0.3)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", fontsize=7, ncol=3, framealpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_label(b) for _, b in rows], fontsize=6, rotation=90)
    axes[0].set_title(f"SAC-SMA vs VIC vs BCM against the CalSim3 unimpaired FNF — "
                      f"WY{wy_range[0]}–{wy_range[1]}, WGEN Product A climate",
                      fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _get(met, st, basin, model, col):
    m = met[(met["set"] == st) & (met["basin"] == basin) & (met["model"] == model)]
    return float(m[col].iloc[0]) if len(m) else np.nan


def regime_fig(long: pd.DataFrame, path, data_dir: str | Path = "data", *,
               wy_range=WY_RANGE) -> None:
    """Mean-monthly regime per basin (Oct -> Sep), all three models over CalSim3.

    The seasonal diagram is where the three models separate most clearly — a
    volume bias shows as a level offset, a snow/melt-timing difference as a
    shifted or broadened spring limb.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = wy_window(wy_range)
    w = long[(long["date"] >= lo) & (long["date"] <= hi)]
    rows = _ordered_basins(data_dir, w)
    ncol = 4
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(_W, 1.35 * nrow), squeeze=False)
    wy_month = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for ax, (st, basin) in zip(axes.ravel(), rows):
        g = w[(w["set"] == st) & (w["basin"] == basin)]
        for src in ("calsim3", *MODELS):
            s = g[g["source"] == src]
            if s.empty:
                continue
            mm = s.groupby(pd.DatetimeIndex(s["date"]).month)["flow_taf"].mean()
            ax.plot(range(12), [mm.get(m, np.nan) for m in wy_month],
                    color=COLORS[src], lw=1.6 if src == "calsim3" else 1.1,
                    ls="-" if src == "calsim3" else "--", zorder=3 if src == "calsim3" else 2)
        ax.set_title(_label(basin), fontsize=7)
        ax.set_xticks([0, 3, 6, 9])
        ax.set_xticklabels(["O", "J", "A", "J"], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(color="0.93", lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=COLORS[s], lw=1.6 if s == "calsim3" else 1.1,
                          ls="-" if s == "calsim3" else "--", label=LABELS[s])
               for s in ("calsim3", *MODELS)]
    axes.ravel()[-1].legend(handles=handles, loc="center", fontsize=7, frameon=False)
    fig.suptitle(f"Mean-monthly regime (TAF/month), WY{wy_range[0]}–{wy_range[1]} "
                 f"— WGEN Product A", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def summary_fig(met: pd.DataFrame, path, *, wy_range=WY_RANGE) -> None:
    """Pooled across basins: the KGE / NSE / % bias distribution of each model.

    Box plus jittered per-basin points, so both the central tendency and the
    spread (which basin a model fails on) stay visible with only 19 basins.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)          # fixed jitter: figure is reproducible
    fig, axes = plt.subplots(1, 3, figsize=(_W, 2.6))
    panels = (("kge", "KGE", (0.0, 1.0)), ("nse", "NSE", (0.0, 1.0)),
              ("pbias", "% bias vs CalSim3", (-_PBIAS_LIM, _PBIAS_LIM)))
    for ax, (col, ylab, ylim) in zip(axes, panels):
        data = [met.loc[met["model"] == m, col].to_numpy(dtype=float) for m in MODELS]
        bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops={"color": "0.15", "lw": 1.2})
        for patch, m in zip(bp["boxes"], MODELS):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.28)
            patch.set_edgecolor(COLORS[m])
        for i, (m, v) in enumerate(zip(MODELS, data), start=1):
            jitter = i + rng.uniform(-0.13, 0.13, size=len(v))
            ax.scatter(jitter, np.clip(v, *ylim), s=11, color=COLORS[m], zorder=3,
                       edgecolor="white", linewidth=0.3)
            _clip_marks(ax, jitter, v, ylim, COLORS[m])
        ax.set_ylim(*ylim)
        if ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xticks(range(1, len(MODELS) + 1))
        ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=7)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"Pooled skill over the {met['basin'].nunique()} combined basins, "
                 f"WY{wy_range[0]}–{wy_range[1]}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
def make_all(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts",
             *, wy_range=WY_RANGE, product: str = PRODUCT, scenario: str = BCM_SCENARIO,
             parallel: bool = False) -> Path:
    """Build the whole three-way artifact into ``artifacts/calsim/compare/``.

    Writes ``sacsma_vic_bcm_monthly.csv`` (the full long table, all years — so the
    window can be re-cut without re-simulating), ``sacsma_vic_bcm_metrics.csv`` (per
    basin per model over ``wy_range``), ``sacsma_vic_bcm_summary.csv`` (pooled), and
    three figures under ``figures/``.
    """
    from ..io import write_table

    out = Path(artifacts_dir) / "calsim" / "compare"
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    long = build_long(data_dir, product=product, scenario=scenario,
                               parallel=parallel)
    met = skill_metrics(long, wy_range=wy_range)
    summ = pooled_summary(met)

    write_table(long, out / "sacsma_vic_bcm_monthly.csv")
    write_table(met, out / "sacsma_vic_bcm_metrics.csv")
    write_table(summ, out / "sacsma_vic_bcm_summary.csv")
    skill_fig(met, figs / "sacsma_vic_bcm_skill.png", data_dir, wy_range=wy_range)
    regime_fig(long, figs / "sacsma_vic_bcm_regime.png", data_dir, wy_range=wy_range)
    summary_fig(met, figs / "sacsma_vic_bcm_summary.png", wy_range=wy_range)

    print(f"sacsma_vic_bcm: {met['basin'].nunique()} basins x {len(MODELS)} models, "
          f"WY{wy_range[0]}-{wy_range[1]} ({int(met['n_months'].max())} months) -> {out}")
    print(summ.to_string(index=False))
    return out
