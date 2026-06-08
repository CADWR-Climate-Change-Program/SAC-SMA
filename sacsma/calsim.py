"""Forward-simulate CalSim3 inflow catchments from the 15-CDEC HRUs.

The 15-CDEC GA optimum gives every meteo grid cell (HRU) a calibrated parameter
set + forcing.  This module re-aggregates those HRUs onto the **CalSim3 inflow
catchments** in ``data/gis/calsim3.gpkg`` instead of the 15 CDEC reservoir
gauges, so the same physics produces an inflow series at each CalSim node.

Pipeline
--------
1. Load CalSim catchments (default: the authoritative ``CalSim3_And_GooseLake``
   layer, ``Rim`` inflows only — the mountain headwater terrain SAC-SMA models).
2. Point-in-polygon assign each unique HRU grid cell to the catchment that
   contains its centroid (a cell maps to at most one catchment).
3. Per covered catchment, area-weight the HRUs' **local runoff** (SMA
   ``surf + base``, mm/day — *not* the CDEC-routed flow, whose ``flowlen``
   targets the wrong outlet) and convert to cfs via the catchment ``SQ_MI``.

Catchments with no HRU centroid inside them are **outside** the 15-CDEC footprint
and cannot be simulated; they are reported in the coverage table.

Per-cell area weight = the **mean** of the cell's per-CDEC-basin area portions
(``area_weight% x basin_area``).  Cells shared between nested CDEC basins (e.g.
SHA inside BND) are thus counted once, so coverage fractions ``hru_area/SQ_MI``
land near 1.0 for well-covered catchments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import (
    DEFAULT_DOMAIN,
    load_basin_area,
    load_hru_table,
    load_params,
    mmday_to_cfs,
    read_table,
)
from .model import DomainForcing, load_domain_forcing, run_hru_components, run_hru_local

CALSIM_GPKG = "calsim3.gpkg"
CALSIM_LAYER = "CalSim3_And_GooseLake"
#: a catchment counts as "covered" once its HRU **grid-footprint** spans at least
#: this fraction of its CalSim ``SQ_MI`` (below it, but >0 HRUs, is "partial" and is
#: NOT scored).  Conservative by design: a partly-covered catchment would have its
#: area-weighted depth extrapolated to the full ``SQ_MI``, overstating sparse sets.
#: 0.6 sits in the gap between genuine coverage (a domain's own coarse-grid basins
#: reach ~0.72) and edge-only false positives (~0.41).
COVERED_FRAC = 0.6
#: cap a cell's footprint box at this side length (deg) so an isolated cell with a
#: distant nearest neighbour cannot claim an oversized footprint (1/16 deg meteo grid).
_MAX_FOOTPRINT_DEG = 0.0625


# --------------------------------------------------------------------------
# Catchments + HRU cells
# --------------------------------------------------------------------------
def load_catchments(
    data_dir: str | Path = "data",
    *,
    layer: str = CALSIM_LAYER,
    rim_only: bool = True,
    gpkg_name: str = CALSIM_GPKG,
):
    """Load CalSim inflow catchments as a GeoDataFrame.

    ``Connect_No`` (the CalSim inflow node id) is reused across a few distinct
    polygons (e.g. ``SHSTA`` labels both Shasta inflow and a Goose Lake area), so
    each polygon gets a unique ``cid``; ``node`` keeps the (non-unique) label.
    Columns: ``cid, node, name, ct_name, type, sq_mi, geometry``.
    """
    import geopandas as gpd

    g = gpd.read_file(Path(data_dir) / "gis" / gpkg_name, layer=layer)
    if rim_only:
        g = g[g["Type"] == "Rim"]
    g = g.reset_index(drop=True)
    return gpd.GeoDataFrame(
        {
            "cid": g.index.to_numpy(),
            "node": g["Connect_No"].to_numpy(),
            "name": g["Remarks"].to_numpy() if "Remarks" in g else None,
            "ct_name": g["CT_Name"].to_numpy() if "CT_Name" in g else None,
            "type": g["Type"].to_numpy(),
            "sq_mi": g["SQ_MI"].to_numpy(dtype=float),
            "geometry": g.geometry.to_numpy(),
        },
        crs=g.crs,
    )


def _cell_area_mi2(lat: np.ndarray, dlat: float = 0.0625, dlon: float = 0.0625) -> np.ndarray:
    """Geographic area (mi^2) of a ``dlat`` x ``dlon`` grid cell centered at ``lat``."""
    lat = np.asarray(lat, dtype=float)
    return (dlat * 69.0) * (dlon * 69.172 * np.cos(np.radians(lat)))


def load_hru_cells(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> pd.DataFrame:
    """Unique HRU grid cells with a true drainage area, key, lat/lon, elev.

    Per-cell area (mi^2) = ``area_weight%/100 x basin_area`` (the HRU's true area).
    For ``15cdec`` ``basin_area`` comes from the area table.  For the CalLite sets
    (regular 1/16 deg grids, no area table) ``basin_area`` is recovered from
    ``area_weight`` by anchoring each basin's largest cell to a full grid cell
    (geometric area at its latitude) -- this matches the published drainage areas
    to a few percent, far better than treating every cell as full.
    """
    h = _hru_abs_area(data_dir, domain)
    return h.groupby("key", as_index=False).agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        elev=("elev", "mean"),
        area_mi2=("abs_area", "mean"),
    )


def _hru_abs_area(data_dir: str | Path, domain: str) -> pd.DataFrame:
    """HRU table with a true per-HRU drainage area ``abs_area`` (mi^2).

    Uses the ``basin_area`` table when present (``15cdec``); otherwise recovers it
    from ``area_weight`` by anchoring each basin's largest cell to a full grid cell.
    """
    h = load_hru_table(data_dir, domain=domain)
    try:
        areas = load_basin_area(data_dir, domain=domain).set_index("basin")["area_mi2"].to_dict()
        h = h.assign(abs_area=[aw / 100.0 * areas[b] for aw, b in zip(h["area_weight"], h["basin"])])
    except FileNotFoundError:
        h = h.assign(_full=_cell_area_mi2(h["lat"].to_numpy()))
        gb = h.groupby("basin")
        # true HRU area = (area_weight / max_area_weight_in_basin) * full_cell_area;
        # the max-area_weight cell is treated as a full interior grid cell.
        h["abs_area"] = (h["area_weight"] / gb["area_weight"].transform("max")
                         * gb["_full"].transform("median"))
        h = h.drop(columns="_full")
    return h


def basin_areas(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN) -> dict[str, float]:
    """Per-basin drainage area (mi^2) for a domain -> ``{basin: area_mi2}``.

    Returns the **authoritative published areas** straight from ``basin_area_<domain>.csv``
    when present (the area table is the source of truth — used directly, NOT re-derived from
    per-HRU ``area_weight`` which need not sum to exactly 100 per basin).  Falls back to
    summing the reconstructed per-HRU areas for domains with no area table.
    """
    from .io import load_basin_area

    try:
        d = load_basin_area(data_dir, domain=domain)
        return {str(b): float(a) for b, a in zip(d["basin"], d["area_mi2"])}
    except FileNotFoundError:
        h = _hru_abs_area(data_dir, domain)
        return h.groupby("basin")["abs_area"].sum().to_dict()


#: equal-area CRS (California Albers, metres) and m^2 per mi^2 for area work.
_EQ_CRS = 3310
_M2_PER_MI2 = 2_589_988.0


def map_hrus_to_catchments(catchments, cells: pd.DataFrame, *,
                           covered_frac: float = COVERED_FRAC):
    """Assign HRUs to catchments by **grid-footprint area overlap** (not centroids).

    Each HRU owns its Voronoi cell **clipped to its true grid footprint** — a box of
    the HRU's own local nearest-neighbour spacing (capped at ``_MAX_FOOTPRINT_DEG``),
    centred on the cell.  This is resolution-correct (15cdec mixes ~0.021 deg and
    0.0625 deg grids) and, crucially, caps **edge** HRUs to their own cell so they
    cannot bleed into catchments they only graze (the old global-buffer Voronoi clip
    inflated edge-only catchments — e.g. THM028 read 0.83 vs a true 0.41).  **Weights**
    come from the clipped-cell overlap; **coverage** (``cov_frac``) is measured separately
    from the gap-free unclipped Voronoi, capped per HRU at its footprint area (see below),
    so a fully-sampled catchment reads ~1.0 while genuinely sparse ones stay low.

    Returns ``(mapping, coverage)``:
      * ``mapping``  — [cid, node, key, lat, lon, area_mi2] per (catchment, overlapping
        HRU); ``area_mi2`` is the (clipped) overlap area = the within-catchment weight.
        An HRU may appear in several catchments.
      * ``coverage`` — one row per catchment with ``n_hru, hru_area_mi2, cov_frac,
        status`` in {covered, partial, outside}.  ``cov_frac`` is the honest fraction of
        the catchment a set's HRUs actually sample; ``status`` uses ``covered_frac`` only
        as an informational label (it does NOT gate scoring — node inclusion is driven by
        the hand-edited crosswalk).
    """
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely import MultiPoint, box, voronoi_polygons

    pts = gpd.GeoDataFrame(
        cells[["key", "lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs=catchments.crs,
    )
    # Voronoi tessellation of the HRU points; associate each cell with its point.
    vor = voronoi_polygons(MultiPoint(list(pts.geometry)),
                           extend_to=box(*pts.total_bounds).buffer(0.2))
    vc = gpd.GeoDataFrame(geometry=list(vor.geoms), crs=catchments.crs)
    vc = gpd.sjoin(vc, pts[["key", "geometry"]], predicate="contains").drop(columns="index_right")

    # per-cell local spacing (median of the 4 nearest neighbours), capped -> footprint box
    lon = cells["lon"].to_numpy(); lat = cells["lat"].to_numpy()
    xy = np.c_[lon, lat]
    k = min(5, len(cells))
    nn = cKDTree(xy).query(xy, k=k)[0][:, 1:]
    s = np.minimum(np.median(nn, axis=1), _MAX_FOOTPRINT_DEG)
    fb = {key: box(x - h, y - h, x + h, y + h)
          for key, x, y, h in zip(cells["key"], lon, lat, s / 2.0)}
    boxes = gpd.GeoSeries([fb[k] for k in vc["key"]], index=vc.index, crs=catchments.crs)
    catch_eq = catchments[["cid", "node", "geometry"]].to_crs(_EQ_CRS)

    # (a) WEIGHTS — clip each Voronoi cell to its footprint box (anti-bleed: edge HRUs
    #     can't be weighted into catchments they only graze), go equal-area, overlap.
    vc_clip = gpd.GeoDataFrame({"key": vc["key"]},
                               geometry=vc.geometry.intersection(boxes),
                               crs=catchments.crs).to_crs(_EQ_CRS)
    ov = gpd.overlay(vc_clip, catch_eq, how="intersection", keep_geom_type=True)
    ov["area_mi2"] = ov.geometry.area / _M2_PER_MI2
    ov = ov[ov["area_mi2"] > 1e-6]
    ll = cells.set_index("key")
    mapping = ov[["cid", "node", "key", "area_mi2"]].copy()
    mapping["lat"] = mapping["key"].map(ll["lat"])
    mapping["lon"] = mapping["key"].map(ll["lon"])
    mapping = mapping[["cid", "node", "key", "lat", "lon", "area_mi2"]].reset_index(drop=True)

    # (b) COVERAGE — honest cov_frac: the UNCLIPPED Voronoi tiles the catchment with no
    #     gaps (a fully-sampled catchment -> ~1.0, vs the clipped-box sum which under-reads
    #     it to ~0.74), but each HRU's contribution is CAPPED at its grid-cell footprint
    #     area so a few far-apart HRUs can't "fill" a large catchment by nearest-neighbour
    #     tiling (genuinely sparse catchments stay low — e.g. I_DER001 ~0.03).
    box_area = dict(zip(vc["key"], boxes.to_crs(_EQ_CRS).area.to_numpy() / _M2_PER_MI2))
    vc_full = gpd.GeoDataFrame(vc[["key", "geometry"]], crs=catchments.crs).to_crs(_EQ_CRS)
    uov = gpd.overlay(vc_full, catch_eq, how="intersection", keep_geom_type=True)
    uov["vor_area"] = uov.geometry.area / _M2_PER_MI2
    uov["cov_area"] = np.minimum(uov["vor_area"], uov["key"].map(box_area))
    cov_area = uov.groupby("cid")["cov_area"].sum()

    agg = mapping.groupby("cid").agg(n_hru=("key", "nunique"), hru_area_mi2=("area_mi2", "sum"))
    cov = catchments.drop(columns="geometry").merge(agg, on="cid", how="left")
    cov["n_hru"] = cov["n_hru"].fillna(0).astype(int)
    cov["hru_area_mi2"] = cov["hru_area_mi2"].fillna(0.0)
    cov["cov_frac"] = (cov["cid"].map(cov_area) / cov["sq_mi"]).fillna(0.0)
    cov["status"] = np.where(
        cov["n_hru"] == 0,
        "outside",
        np.where(cov["cov_frac"] >= covered_frac, "covered", "partial"),
    )
    return mapping, cov


# --------------------------------------------------------------------------
# Basin -> CalSim node mapping: authoritative anchor crosswalk + geographic
# --------------------------------------------------------------------------
#: BOOTSTRAP-ONLY threshold: when geographically *building* the crosswalk
#: (:func:`_bootstrap_geographic_nodes`), a catchment joins a SECONDARY basin's
#: sub-system once that basin's HRUs cover this fraction of it AND it is the basin that
#: covers it most (winner-take-all).  The steady-state :func:`derive_basin_nodes` reads
#: the hand-edited crosswalk and IGNORES this — node membership is curated by hand.
SUBSYSTEM_FRAC = 0.5

#: calibration basin -> CalSim rim **system** (from RimInflowAnchor).  A basin mapped
#: here gets its rim system's *authoritative* member arcs (no geography); basins absent
#: here are **secondary** watersheds (Mokelumne, Putah/Berryessa, Cache, New Hogan, ...)
#: resolved geographically against the non-rim CalSim catchments.
BASIN_RIM_SYSTEM = {
    "15cdec": {"SHA": "SHAS", "BND": "SRBB", "ORO": "OROV", "YRS": "YUBA",
               "FOL": "FOLS", "NML": "ST", "TLG": "TU", "MRC": "ME", "MIL": "SJ"},
    "11obs": {"SHA": "SHAS", "BND": "SRBB", "FTO": "OROV", "YRS": "YUBA",
              "AMF": "FOLS", "SNS": "ST", "TLG": "TU", "MRC": "ME", "SJF": "SJ",
              "TNL": "TRIN"},
    "9unimp": {},
}
#: nesting: a rim system whose inflow **includes** a nested system's nodes.  The only
#: nest among the CalSim rim systems is Sac R @ Bend Bridge (SRBB) over Shasta (SHAS) —
#: so Bend Bridge's anchor sum includes ``I_SHSTA`` while Shasta is also scored alone.
SYSTEM_NESTS = {"SRBB": ["SHAS"]}
#: single-node **cumulative** systems: one CalSim node carries the *whole* basin inflow
#: (the GIS sub-catchments are not used in CalSim).  These nodes are scored only at the
#: basin level (anchor) and **excluded from the per-catchment map** — comparing a tiny
#: local sub-polygon's runoff to the full-basin series is meaningless.
CUMULATIVE_SYSTEMS = {"SHAS", "TRIN", "ME", "SJ"}
#: below-rim main-stem valley reaches: a CalSim3 ``INFLOW`` series exists but these are
#: NOT unimpaired rim inflows (e.g. San Joaquin below Millerton, Tuolumne below the
#: reservoirs), so they are excluded from both views.  Editable seed list.
EXCLUDE_ARCS = {"I_SJR258", "I_SJR265", "I_TUO054", "I_TUO105"}
#: GIS ``Connect_No`` labels that differ from the matching CalSim3 ``INFLOW`` series arc.
#: The GIS polygon for Lake Berryessa is ``BRYSA``; its CalSim3 inflow series is the
#: Putah Creek node ``I_PTH070`` — without this alias, Putah/Berryessa would be unscored.
GIS_ARC_ALIAS = {"I_BRYSA": "I_PTH070"}


def series_arc(node) -> str:
    """CalSim3 ``INFLOW`` series arc for a GIS catchment ``node`` (applies the alias)."""
    a = "I_" + str(node)
    return GIS_ARC_ALIAS.get(a, a)


#: basin-level nesting (for the basin->node mapping): a basin whose inflow **includes**
#: another basin's nodes.  The only nest is Sac R @ Bend Bridge over Shasta, so BND's
#: anchor sum picks up every node assigned to SHA (i.e. ``I_SHSTA``) on top of its own.
BASIN_NESTS = {"BND": ["SHA"]}


def load_crosswalk(data_dir: str | Path = "data") -> pd.DataFrame:
    """The single hand-editable master crosswalk (``data/reference/calsim_crosswalk.csv``).

    Columns ``[arc, system, unimp_anchor, vic_basin, basin_15cdec, basin_11obs,
    basin_9unimp, in_calsim3]``.  This is the **authoritative source** for the basin->node
    mapping, rim-system membership, and VIC node names — edit it by hand; nothing in the
    pipeline overwrites it.  (Bootstrap it once with :func:`sacsma.compare.build_crosswalk`.)
    """
    return pd.read_csv(Path(data_dir) / "reference" / "calsim_crosswalk.csv")


def load_rim_anchor(data_dir: str | Path = "data") -> pd.DataFrame:
    """Raw RimInflowAnchor crosswalk ``[arc, system, unimp_anchor]`` — **bootstrap only**
    (folded into :func:`load_crosswalk`'s ``calsim_crosswalk.csv`` going forward)."""
    return pd.read_csv(Path(data_dir) / "reference" / "calsim_rim_anchor.csv")


def _system_members(anchor: pd.DataFrame) -> dict[str, set[str]]:
    """Each rim system -> its full set of member arcs, expanding nesting (SRBB ⊇ SHAS).

    The crosswalk gives each arc one *home* system; :data:`SYSTEM_NESTS` then folds a
    nested system's arcs into its parent so Bend Bridge inherits Shasta's ``I_SHSTA``.
    """
    home = {sysn: set(g["arc"].astype(str))
            for sysn, g in anchor.dropna(subset=["system"]).groupby("system")}

    def expand(sysn: str, seen: set[str]) -> set[str]:
        if sysn in seen:
            return set()
        seen.add(sysn)
        out = set(home.get(sysn, set()))
        for nested in SYSTEM_NESTS.get(sysn, []):
            out |= expand(nested, seen)
        return out

    return {sysn: expand(sysn, set()) for sysn in set(home) | CUMULATIVE_SYSTEMS}


def _footprint_union(lon, lat, crs):
    """Union (equal-area) of per-cell grid-footprint boxes for a set of HRU points.

    Each box's side is the cell's own local nearest-neighbour spacing (capped at
    ``_MAX_FOOTPRINT_DEG``), so the union is the true terrain the HRUs represent at
    whatever local resolution the grid has.
    """
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely import box
    from shapely.ops import unary_union

    lon = np.asarray(lon, dtype=float); lat = np.asarray(lat, dtype=float)
    xy = np.c_[lon, lat]
    if len(xy) > 1:
        nn = cKDTree(xy).query(xy, k=min(5, len(xy)))[0][:, 1:]
        s = np.minimum(np.median(nn, axis=1), _MAX_FOOTPRINT_DEG)
    else:
        s = np.full(len(xy), _MAX_FOOTPRINT_DEG)
    boxes = gpd.GeoSeries([box(x - h, y - h, x + h, y + h)
                           for x, y, h in zip(lon, lat, s / 2.0)], crs=crs).to_crs(_EQ_CRS)
    return unary_union(list(boxes.values))


def derive_basin_nodes(
    data_dir: str | Path = "data",
    domain: str = DEFAULT_DOMAIN,
    *,
    calsim3_arcs=None,          # accepted for back-compat; in_calsim3 comes from the crosswalk
    rim_only: bool = True,      # unused (kept for back-compat)
    subsystem_frac: float = SUBSYSTEM_FRAC,
) -> pd.DataFrame:
    """Project the hand-editable crosswalk into a ``domain`` basin -> CalSim node table.

    The mapping is read straight from ``calsim_crosswalk.csv`` (``basin_<domain>`` column),
    with basin-level nesting applied (:data:`BASIN_NESTS`: Bend Bridge picks up Shasta's
    nodes).  No geography is computed here — edit the crosswalk to change assignments.  The
    one-time geographic bootstrap that *built* the crosswalk is :func:`_bootstrap_geographic_nodes`.

    Returns ``[basin, cid, node, arc, system, kind, cov_frac, in_calsim3]``.  ``kind`` ∈
    {``rim_member``, ``rim_cumulative``, ``secondary``}; ``in_calsim3`` flags a usable
    (non-zero) CalSim3 series.
    """
    cw = load_crosswalk(data_dir)
    col = f"basin_{domain}"
    if col not in cw.columns:
        raise KeyError(f"{col} not in calsim_crosswalk.csv (have {list(cw.columns)})")
    cumulative_arcs = set().union(*(_system_members(cw)[s] for s in CUMULATIVE_SYSTEMS))
    # arc -> catchment cid (original layer), best-effort for the maps/coverage table
    catch = load_catchments(data_dir, rim_only=True)
    arc2cid = {series_arc(n): int(c) for n, c in zip(catch["node"], catch["cid"])}

    def kind_of(arc: str, system) -> str:
        if pd.isna(system):
            return "secondary"
        return "rim_cumulative" if arc in cumulative_arcs else "rim_member"

    rows = []
    for basin in cw[col].dropna().unique():
        owners = {basin} | set(BASIN_NESTS.get(basin, []))
        sub = cw[cw[col].isin(owners)]
        for _, r in sub.iterrows():
            arc = str(r["arc"])
            rows.append({
                "basin": basin, "cid": arc2cid.get(arc, -1), "node": arc[2:], "arc": arc,
                "system": r["system"] if pd.notna(r["system"]) else None,
                "kind": kind_of(arc, r["system"]),
                "cov_frac": float("nan"),
                "in_calsim3": bool(r["in_calsim3"]),
            })
    cols = ["basin", "cid", "node", "arc", "system", "kind", "cov_frac", "in_calsim3"]
    return (pd.DataFrame(rows, columns=cols)
            .sort_values(["basin", "kind", "arc"]).reset_index(drop=True))


def _bootstrap_geographic_nodes(
    data_dir: str | Path = "data",
    domain: str = DEFAULT_DOMAIN,
    *,
    subsystem_frac: float = SUBSYSTEM_FRAC,
    calsim3_arcs=None,
    rim_only: bool = True,
) -> pd.DataFrame:
    """One-time geographic bootstrap used to *build* the crosswalk (rim membership from
    :func:`load_rim_anchor`; secondary basins by HRU-footprint winner-take-all).  Steady
    state reads the hand-edited crosswalk via :func:`derive_basin_nodes` instead.

    Two regimes, by basin: rim-system basins take their CalSim rim system's authoritative
    member arcs (nesting honoured; cumulative single nodes resolve to one whole-basin node);
    secondary basins get geographic winner-take-all over the non-rim catchments (excluding
    rim members and the below-rim :data:`EXCLUDE_ARCS`).  Same columns as
    :func:`derive_basin_nodes`.
    """
    from .io import load_hru_table

    catch = load_catchments(data_dir, rim_only=rim_only)
    catch_eq = catch.to_crs(_EQ_CRS)
    nodes = catch_eq["node"].astype(str).to_numpy()
    cids = catch_eq["cid"].to_numpy()
    arcs_of = np.array([series_arc(n) for n in nodes])
    geoms = catch_eq.geometry.to_numpy()
    sqmi = catch_eq["sq_mi"].to_numpy(dtype=float)
    arc2cid = {a: int(c) for a, c in zip(arcs_of, cids)}

    anchor = load_rim_anchor(data_dir)
    members = _system_members(anchor)
    cumulative_arcs = set().union(*(members[s] for s in CUMULATIVE_SYSTEMS))
    rim_member_arcs = set().union(*members.values())
    in_cs3 = set(map(str, calsim3_arcs)) if calsim3_arcs is not None else None

    h = load_hru_table(data_dir, domain=domain)
    basins = sorted(h["basin"].unique())
    bsys = BASIN_RIM_SYSTEM.get(domain, {})

    # per-basin geographic coverage of every catchment (for cov_frac + secondary WTA)
    cov = np.zeros((len(basins), len(catch_eq)))
    for i, b in enumerate(basins):
        hb = h[h["basin"] == b].drop_duplicates("key")
        foot = _footprint_union(hb["lon"].to_numpy(), hb["lat"].to_numpy(), catch.crs)
        cov[i] = np.array([g.intersection(foot).area for g in geoms]) / _M2_PER_MI2 / sqmi
    bi = {b: i for i, b in enumerate(basins)}
    cov_of = {a: i for i, a in enumerate(arcs_of)}  # arc -> catchment col (last wins; ok)

    def cov_frac(basin: str, arc: str) -> float:
        j = cov_of.get(arc)
        return round(float(cov[bi[basin], j]), 3) if j is not None else float("nan")

    rows = []
    # 1) rim-system basins: authoritative anchor membership
    for basin in basins:
        sysn = bsys.get(basin)
        if not sysn:
            continue
        for arc in sorted(members.get(sysn, set())):
            rows.append({
                "basin": basin, "cid": arc2cid.get(arc, -1), "node": arc[2:], "arc": arc,
                "system": sysn,
                "kind": "rim_cumulative" if arc in cumulative_arcs else "rim_member",
                "cov_frac": cov_frac(basin, arc),
                "in_calsim3": (arc in in_cs3) if in_cs3 is not None else None,
            })

    # 2) secondary basins: geographic winner-take-all over non-rim candidate catchments
    secondary = [b for b in basins if b not in bsys]
    if secondary:
        elig = np.array([
            (a not in rim_member_arcs) and (a not in EXCLUDE_ARCS)
            and (in_cs3 is None or a in in_cs3) and (a in arc2cid)
            for a in arcs_of
        ])
        sidx = [bi[b] for b in secondary]
        sub = cov[sidx]                      # secondary-basins x catchments
        win = np.array(sidx)[sub.argmax(axis=0)]
        winf = sub.max(axis=0)
        for j in np.where(elig & (winf >= subsystem_frac))[0]:
            basin = basins[int(win[j])]
            arc = str(arcs_of[j])
            rows.append({
                "basin": basin, "cid": int(cids[j]), "node": str(nodes[j]), "arc": arc,
                "system": None, "kind": "secondary", "cov_frac": round(float(cov[int(win[j]), j]), 3),
                "in_calsim3": (arc in in_cs3) if in_cs3 is not None else None,
            })

    cols = ["basin", "cid", "node", "arc", "system", "kind", "cov_frac", "in_calsim3"]
    return (pd.DataFrame(rows, columns=cols)
            .sort_values(["basin", "kind", "cov_frac"], ascending=[True, True, False])
            .reset_index(drop=True))


#: name of the merged "whole-basin" catchment layer written alongside the original.
MERGED_LAYER = "CalSim3_Merged"


def build_merged_gis(data_dir: str | Path = "data", *, sets=("15cdec", "11obs", "9unimp"),
                     out_layer: str = MERGED_LAYER) -> "object":
    """Write a **merged** Rim-catchment layer where sub-arcs that CalSim3 does not model
    individually are dissolved into the node that actually carries their flow — so the
    cumulative single-node basins (Merced, Shasta, San Joaquin, Trinity) become **whole**
    catchments instead of a usable anchor surrounded by series-less slivers.

    Rule per Rim polygon: keep it if its (alias-aware) ``series_arc`` has a **non-zero**
    CalSim3 series; otherwise absorb it into the usable member of the rim **system** its
    area falls in (cumulative systems have one member → the whole basin collapses to the
    anchor; distributed systems → the adjacent reservoir-inflow member).  Blank/junk
    polygons are dropped.  The result is dissolved by target arc (geometry unioned,
    ``SQ_MI`` summed) and written as a NEW layer ``out_layer`` in ``calsim3.gpkg`` —
    the original ``CalSim3_And_GooseLake`` is left untouched.
    """
    import geopandas as gpd
    from shapely.ops import unary_union

    gpkg = Path(data_dir) / "gis" / CALSIM_GPKG
    g = gpd.read_file(gpkg, layer=CALSIM_LAYER)
    rim = g[g["Type"] == "Rim"].reset_index(drop=True).copy()
    rim["sarc"] = rim["Connect_No"].astype(str).map(series_arc)

    c3 = read_table(Path(data_dir) / "reference" / "calsim3_inflow_monthly.csv")
    nz = c3.groupby("arc")["flow_taf"].max()
    usable = set(nz[nz > 0].index.astype(str))

    # rim system geographic footprint (union of its basins' HRU footprints, equal-area)
    members = _system_members(load_crosswalk(data_dir))
    sysfoot: dict[str, object] = {}
    for dom in sets:
        fp = basin_footprints(data_dir, dom)
        for basin, sysn in BASIN_RIM_SYSTEM.get(dom, {}).items():
            if basin in fp:
                sysfoot[sysn] = fp[basin] if sysn not in sysfoot else unary_union([sysfoot[sysn], fp[basin]])
    crs = rim.crs
    sysfoot_eq = {s: gpd.GeoSeries([f], crs=crs).to_crs(_EQ_CRS).iloc[0] for s, f in sysfoot.items()}
    rim_eq = rim.to_crs(_EQ_CRS)
    geoms = rim_eq.geometry.to_numpy()
    arc_idx: dict[str, list[int]] = {}
    for i, a in enumerate(rim["sarc"]):
        if a in usable:
            arc_idx.setdefault(a, []).append(i)

    def absorb(i: int) -> str | None:
        gi = geoms[i]
        # the rim system whose footprint this polygon most overlaps
        sysn = max(sysfoot_eq, key=lambda s: gi.intersection(sysfoot_eq[s]).area, default=None)
        if sysn is None or gi.intersection(sysfoot_eq[sysn]).area <= 0:
            return None
        cand = [a for a in members[sysn] if a in arc_idx]
        if not cand:
            return None
        # nearest usable member: longest shared boundary, else nearest centroid
        best = max(cand, key=lambda a: max(gi.boundary.intersection(geoms[j].boundary).length
                                           for j in arc_idx[a]))
        if max(gi.boundary.intersection(geoms[j].boundary).length
               for j in arc_idx[best]) <= 0:
            best = min(cand, key=lambda a: min(gi.centroid.distance(geoms[j].centroid)
                                               for j in arc_idx[a]))
        return best

    merge_to = []
    for i, a in enumerate(rim["sarc"]):
        if str(rim["Connect_No"].iloc[i]).strip() == "":
            merge_to.append(None)            # drop blank/junk polygons
        elif a in usable:
            merge_to.append(a)               # keep: modelled individually
        else:
            merge_to.append(absorb(i))
    rim["merge_to"] = merge_to
    rim = rim[rim["merge_to"].notna()].copy()

    diss = rim.dissolve(by="merge_to", aggfunc={"SQ_MI": "sum"}).reset_index()
    diss["Connect_No"] = diss["merge_to"].str.slice(2)
    diss["Type"] = "Rim"
    keep = [c for c in ("Connect_No", "Type", "SQ_MI", "CT_Name", "Remarks", "geometry") if c in diss]
    out = gpd.GeoDataFrame(diss[keep], geometry="geometry", crs=crs)
    out.to_file(gpkg, layer=out_layer, driver="GPKG")
    n_abs = int((rim["merge_to"] != rim["sarc"]).sum())
    print(f"merged GIS: {len(rim)} Rim polygons -> {len(out)} whole catchments "
          f"({n_abs} series-less absorbed) -> {gpkg}::{out_layer}")
    return out


def basin_footprints(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN,
                     *, simplify_deg: float = 0.01):
    """Per-basin HRU footprint union (lon/lat), the watershed each calibration basin
    represents — for drawing clean sub-system outlines on the coverage maps.

    Returns a dict ``{basin: shapely geometry}`` in the catchments' CRS (lon/lat),
    holes filled and lightly simplified so the outline is a single clean outer edge.
    """
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    from .io import load_hru_table

    crs = load_catchments(data_dir, rim_only=True).crs
    h = load_hru_table(data_dir, domain=domain)
    out = {}
    for b in sorted(h["basin"].unique()):
        hb = h[h["basin"] == b].drop_duplicates("key")
        foot = _footprint_union(hb["lon"].to_numpy(), hb["lat"].to_numpy(), crs)  # equal-area
        geom = gpd.GeoSeries([foot], crs=_EQ_CRS).to_crs(crs).iloc[0]
        geom = geom.buffer(0.01).buffer(-0.01).simplify(simplify_deg)  # close gaps -> outer edge
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        out[b] = MultiPolygon([Polygon(p.exterior) for p in polys if p.area > 0])  # drop holes
    return out


# --------------------------------------------------------------------------
# Special CalSim flowlens (for optional channel routing to the CalSim node)
# --------------------------------------------------------------------------
_EARTH_R_M = 6_371_000.0


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance (m) between arrays/scalars of lat/lon (degrees)."""
    la1, lo1, la2, lo2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    dphi = la2 - la1
    dlam = lo2 - lo1
    a = np.sin(dphi / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlam / 2) ** 2
    return 2 * _EARTH_R_M * np.arcsin(np.sqrt(a))


def assign_flowlens(mapping: pd.DataFrame, cells: pd.DataFrame, *, sinuosity: float = 1.4):
    """Per-HRU flow length (m) to each catchment's outlet, for CalSim-node routing.

    The native HRU ``flowlen`` is the distance to the *CDEC reservoir* outlet, which
    is wrong for a CalSim sub-node.  Lacking explicit node coordinates in the GIS,
    the catchment outlet (pour point) is taken as the **lowest-elevation HRU cell**;
    each HRU's flow length is the great-circle distance to it x ``sinuosity`` (a
    channel-meander factor).  The outlet cell gets ``flowlen_m=0``/``is_outlet=1``
    (Lohmann channel UH becomes identity there).  Returns ``mapping`` with added
    ``elev, flowlen_m, is_outlet`` columns.
    """
    m = mapping.merge(cells[["key", "elev"]], on="key", how="left")
    parts = []
    for _cid, g in m.groupby("cid", sort=False):
        g = g.copy()
        i = int(np.asarray(g["elev"].values).argmin())
        olat, olon = g["lat"].values[i], g["lon"].values[i]
        g["flowlen_m"] = _haversine_m(g["lat"].values, g["lon"].values, olat, olon) * sinuosity
        g["is_outlet"] = 0
        g.iloc[i, g.columns.get_loc("is_outlet")] = 1
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def run_calsim(
    data_dir: str | Path = "data",
    *,
    domain: str = DEFAULT_DOMAIN,
    layer: str = CALSIM_LAYER,
    rim_only: bool = True,
    start: str | None = None,
    end: str | None = None,
    forcing: DomainForcing | None = None,
    route: bool = False,
    sinuosity: float = 1.4,
    covered_frac: float = COVERED_FRAC,
    progress: bool = False,
):
    """Simulate area-weighted inflow for every covered catchment.

    With ``route=False`` (default) each catchment's inflow is the area-weighted
    HRU **local runoff** (SMA ``surf+base``).  With ``route=True`` each HRU's flow
    is first Lohmann-routed to the catchment outlet using a **special CalSim
    flowlen** (:func:`assign_flowlens`) before area-weighting — a physically routed
    daily inflow at the CalSim node (negligible at monthly aggregation, matters for
    daily shape).

    Returns ``(flows, coverage, mapping)`` where ``flows`` is long-format
    [date, cid, node, flow_mmday, flow_cfs] over all catchments that have >=1 HRU,
    and ``mapping`` carries the per-HRU ``flowlen_m``/``is_outlet`` columns.
    """
    catch = load_catchments(data_dir, layer=layer, rim_only=rim_only)
    cells = load_hru_cells(data_dir, domain=domain)
    mapping, cov = map_hrus_to_catchments(catch, cells, covered_frac=covered_frac)
    mapping = assign_flowlens(mapping, cells, sinuosity=sinuosity)

    if forcing is None:
        if progress:
            print("loading domain forcing once...", flush=True)
        forcing = load_domain_forcing(data_dir, domain=domain, start=start, end=end)
    # per-watershed calibrations repeat shared cells; one param set per cell suffices
    # for the CalSim aggregation, so keep the first.
    params = load_params(data_dir, domain=domain).drop_duplicates("key").set_index("key")

    # compute each unique HRU cell once; an HRU may feed several catchments.
    keys = mapping["key"].unique()
    meta = cells.set_index("key")
    label = "routed" if route else "local runoff"
    local: dict[str, np.ndarray] = {}
    comp: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, k in enumerate(keys):
        if progress and (i % 500 == 0):
            print(f"  {label}: HRU {i + 1}/{len(keys)}", flush=True)
        c = forcing.pos[k]
        args = (forcing.prcp[c], forcing.tavg[c], forcing.doy, forcing.is_leap)
        kw = dict(lat=float(meta.at[k, "lat"]), elev=float(meta.at[k, "elev"]), ga_row=params.loc[k])
        if route:
            comp[k] = run_hru_components(*args, **kw)
        else:
            local[k] = run_hru_local(*args, **kw)

    if route:
        from . import parameters as P
        from .routing import lohmann

    dates = forcing.dates
    sqmi = catch.set_index("cid")["sq_mi"].to_dict()
    frames = []
    for cid, grp in mapping.groupby("cid"):
        w = grp["area_mi2"].to_numpy(dtype=float)
        w = w / w.sum()
        depth = np.zeros(len(dates))
        if route:
            for wi, row in zip(w, grp.itertuples(index=False)):
                surf, base = comp[row.key]
                routed, _b = lohmann(surf, base, float(row.flowlen_m),
                                     P.routing_par(params.loc[row.key]), int(row.is_outlet))
                depth += wi * routed
        else:
            for wi, k in zip(w, grp["key"]):
                depth += wi * local[k]
        frames.append(pd.DataFrame({
            "date": dates,
            "cid": cid,
            "node": grp["node"].iloc[0],
            "flow_mmday": depth,
            "flow_cfs": mmday_to_cfs(depth, sqmi[cid]),
        }))
    flows = pd.concat(frames, ignore_index=True)
    return flows, cov, mapping
