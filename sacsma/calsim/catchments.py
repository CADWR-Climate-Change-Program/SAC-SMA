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

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import (
    DEFAULT_DOMAIN,
    load_basin_area,
    load_hru_table,
    load_params,
    mmday_to_cfs,
)
from ..model import (
    DomainForcing,
    load_domain_forcing,
    run_hru_components_cached,
    run_local_runoff_parallel,
)

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
#: the 1/16 deg climate-grid step.  9unimp/11obs/12rim HRUs ARE this regular lattice
#: (each point = one grid cell), so they get a simple square-cell overlap weighting;
#: 15cdec HRUs are irregular sub-grid centroids and use the Voronoi footprint path.
_GRID_STEP_DEG = 0.0625


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

    from . import calsim_dir
    g = gpd.read_file(calsim_dir(data_dir) / "gis" / gpkg_name, layer=layer)
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

    Prefers the **CalSim-consistent canonical area** ``basin_area_<domain>_calsim.csv`` when
    present — the summed GIS catchment area (:func:`calsim_basin_areas`) so a basin total sits
    on the SAME area as its sub-arcs and the CalSim reference.  Otherwise the **authoritative
    published areas** from ``basin_area_<domain>.csv`` (the source of truth, used directly, NOT
    re-derived from per-HRU ``area_weight``).  Falls back to summing reconstructed per-HRU areas.
    """
    from ..io import load_basin_area, read_table
    from . import calsim_dir
    calsim_csv = calsim_dir(data_dir) / f"basin_area_{domain}_calsim.csv"
    if calsim_csv.exists():
        d = read_table(calsim_csv)
        return {str(b): float(a) for b, a in zip(d["basin"], d["area_mi2"])}
    try:
        d = load_basin_area(data_dir, domain=domain)
        return {str(b): float(a) for b, a in zip(d["basin"], d["area_mi2"])}
    except FileNotFoundError:
        h = _hru_abs_area(data_dir, domain)
        return h.groupby("basin")["abs_area"].sum().to_dict()


def calsim_basin_areas(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN,
                       *, write: bool = False) -> dict[str, float]:
    """Per-basin **CalSim GIS catchment area** (mi^2): the sum of the merged-layer catchment
    ``SQ_MI`` the basin owns — its crosswalk nodes (:func:`derive_basin_nodes`) + ``BASIN_NESTS``
    nesting (Bend Bridge ⊇ Shasta) + the series-less valley-accretion node (``I_<SYS>_VAL``,
    e.g. Bend Bridge's SR_02/SR_03).  This is exactly the area :func:`run_calsim` already uses
    per sub-arc, so the basin total sits on the same area as its sub-arcs AND the CalSim
    reference (removing the SAC-volume-vs-CalSim-flow area mismatch).  Basins with no CalSim
    catchment (e.g. Tulare/Kern) keep their authoritative area.  ``write=True`` saves
    ``basin_area_<domain>_calsim.csv`` (``[basin, area_mi2, source]``)."""
    from ..io import load_basin_area, write_table

    try:
        ab = load_basin_area(data_dir, domain=domain)
        out = {str(b): float(a) for b, a in zip(ab["basin"], ab["area_mi2"])}
    except FileNotFoundError:
        h = _hru_abs_area(data_dir, domain).groupby("basin")["abs_area"].sum()
        out = {str(b): float(a) for b, a in h.items()}
    src = {b: "authoritative" for b in out}

    nodes = derive_basin_nodes(data_dir, domain)
    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True).copy()
    catch["arc"] = catch["node"].map(series_arc)
    arc_area = catch.groupby("arc")["sq_mi"].sum()
    for basin, g in nodes.groupby("basin"):
        arcs = list(dict.fromkeys(g["arc"].astype(str)))
        area = float(arc_area.reindex(arcs).fillna(0.0).sum())
        sysn = BASIN_RIM_SYSTEM.get(domain, {}).get(str(basin))
        if sysn in VALLEY_SYSTEMS:                     # add the modeled valley-accretion node
            area += float(arc_area.get(valley_arc_for_system(sysn), 0.0))
        if area > 0:
            out[str(basin)], src[str(basin)] = round(area, 2), "calsim"
    if write:
        df = pd.DataFrame({"basin": list(out), "area_mi2": [out[b] for b in out],
                           "source": [src[b] for b in out]})
        from . import calsim_dir
        write_table(df, calsim_dir(data_dir) / f"basin_area_{domain}_calsim.csv")
        print(f"calsim_basin_areas[{domain}]: {sum(v=='calsim' for v in src.values())} CalSim + "
              f"{sum(v=='authoritative' for v in src.values())} fallback -> {domain}_calsim.csv")
    return out


#: the ONLY basins whose HRU footprint is GIS-screened for the anchor (revised 2026-07-08):
#: basins whose footprint **materially over-reaches** its CalSim3 catchment — SHA and BND
#: (the ~1000 mi^2 **endorheic Goose Lake / Modoc block**, terrain whose runoff never
#: reaches the gauge; the exact parallel of VIC's ``no_gooselake`` fix for
#: I_SHSTA / 8RI_SRBB) plus SNS and ChowchillaRiver (delineation over-reach well beyond
#: the catchment, the same ``area_artifact`` family as their targets).  Every other basin
#: keeps its **full calibrated footprint**: the anchor volume is the full-footprint
#: area-weighted depth x the canonical CalSim3 catchment area, and trimming ordinary
#: boundary rasterization would distort the calibrated depth rather than fix anything.
SCREENED_BASINS = ("SHA", "BND", "SNS", "ChowchillaRiver")


def screened_footprint(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN,
                       *, basins: tuple[str, ...] | None = SCREENED_BASINS,
                       write: bool = False) -> pd.DataFrame:
    """Per-basin HRU set **screened to its true CalSim catchment**, with overlap-area weights.

    For each screened basin, keep only the HRUs whose grid-cell footprint overlaps the basin's
    CalSim GIS catchment (:func:`derive_basin_nodes` arcs + the ``<SYS>_VAL`` valley-accretion
    node, on the :data:`MERGED_LAYER`), and weight each retained HRU by that **overlap area**
    (mi^2) — the honest sub-area it contributes to the catchment — instead of the domain's own
    ``area_weight`` (which spans the basin's full HRU footprint).

    ``basins`` limits which basins are screened at all — default :data:`SCREENED_BASINS`
    (SHA/BND, the Goose Lake fix, + SNS/ChowchillaRiver, delineation over-reach; revised
    2026-07-08 from the earlier uniform rule): every other basin's anchor uses the full
    calibrated footprint — its full-footprint area-weighted depth times the canonical
    CalSim3 catchment area — so this returns no rows for it.  Pass ``basins=None`` (all) or
    an explicit tuple for per-basin geometry diagnostics (e.g. the footprint maps).  The
    screening is **deterministic** from tracked data (``calsim3.gpkg`` merged layer +
    ``calsim_crosswalk.csv``) — nothing here is hand-tuned; it reuses exactly the
    :func:`map_hrus_to_catchments` overlap the per-sub-arc cross-compare uses.

    Returns ``[basin, key, overlap_area_mi2]`` (one row per retained HRU per screened basin).
    It does **not** replace the full-footprint :func:`sacsma.model.run_basin` (the calibration
    basis).  ``write=True`` saves ``data/calsim/screened_footprint_<domain>.csv``.
    """
    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True)
    nodes = derive_basin_nodes(data_dir, domain)
    cells = load_hru_cells(data_dir, domain=domain)
    hru = load_hru_table(data_dir, domain=domain)
    sysmap = BASIN_RIM_SYSTEM.get(domain, {})
    parts = []
    basin_list = sorted(hru["basin"].unique())
    if basins is not None:
        basin_list = [b for b in basin_list if b in basins]
    for basin in basin_list:
        own = set(nodes.loc[nodes["basin"] == basin, "node"].astype(str))
        if not own:
            continue                                    # no CalSim catchment (e.g. Tulare/Kern)
        sysn = sysmap.get(basin)
        if sysn in VALLEY_SYSTEMS:                       # add the modeled valley-accretion node
            own.add(valley_arc_for_system(sysn)[2:])
        catch_b = catch[catch["node"].astype(str).isin(own)].reset_index(drop=True)
        if catch_b.empty:
            warnings.warn(                               # nodes present but no polygon resolves
                f"screened_footprint[{domain}]: basin {basin!r} crosswalk nodes {sorted(own)} "
                f"match no {MERGED_LAYER} polygon -> no screened footprint (falls back to full)",
                stacklevel=2)
            continue
        keys_b = set(hru.loc[hru["basin"] == basin, "key"])
        cells_b = cells[cells["key"].isin(keys_b)].reset_index(drop=True)
        mapping, _ = map_hrus_to_catchments(catch_b, cells_b)
        w = mapping.groupby("key")["area_mi2"].sum()
        if w.empty:                                      # HRU cells miss the catchment entirely
            warnings.warn(
                f"screened_footprint[{domain}]: basin {basin!r} HRU cells overlap none of its "
                f"catchment polygons -> no screened footprint (falls back to full)", stacklevel=2)
            continue
        parts.append(pd.DataFrame({"basin": basin, "key": w.index.to_numpy(),
                                   "overlap_area_mi2": w.to_numpy()}))
    cols = ["basin", "key", "overlap_area_mi2"]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    if write:
        from ..io import write_table
        from . import calsim_dir
        write_table(out, calsim_dir(data_dir) / f"screened_footprint_{domain}.csv")
        print(f"screened_footprint[{domain}]: {out['basin'].nunique()} basins, {len(out)} HRU rows "
              f"-> screened_footprint_{domain}.csv")
    return out


#: equal-area CRS (California Albers, metres) for area work.  Uses EPSG:3310's *parameters* but
#: on the **WGS84 datum**, so reprojecting the WGS84 (EPSG:4326) HRU/catchment geometries is a
#: pure projection with **no datum shift** — avoiding the WGS84->NAD83 transform pipeline, which
#: on some PROJ builds (e.g. PROJ 9.8) returns non-finite coords for 4326->3310 and collapses
#: every geometry to invalid (empty overlay).  Areas match EPSG:3310 to <0.001% (ellipsoid only).
_EQ_CRS = ("+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 "
           "+x_0=0 +y_0=-4000000 +datum=WGS84 +units=m +no_defs")
_M2_PER_MI2 = 2_589_988.0


def _is_regular_grid(cells: pd.DataFrame, *, step: float = _GRID_STEP_DEG,
                     tol: float = 1e-4, frac: float = 0.8) -> bool:
    """True if the HRU cells sit on a regular ``step``-deg lattice (each point is one
    grid cell — 9unimp/11obs/12rim), vs irregular sub-grid centroids (15cdec).  Tested
    by the fraction of nearest-neighbour spacings equal to ``step`` (1.0 for the regular
    grids, 0.0 for 15cdec)."""
    from scipy.spatial import cKDTree

    xy = np.c_[cells["lon"].to_numpy(dtype=float), cells["lat"].to_numpy(dtype=float)]
    if len(xy) < 2:
        return False
    d = cKDTree(xy).query(xy, k=2)[0][:, 1]
    return bool(np.mean(np.abs(d - step) < tol) >= frac)


def _square_cell_overlap(cells: pd.DataFrame, catch_eq, crs, step: float):
    """Weights from a simple regular **square grid**: each HRU point owns its ``step``-deg
    cell; overlap that with the catchments.  On a regular lattice the cells tile with no
    gaps or overlaps, so the overlap area is BOTH the within-catchment weight AND the
    covered area (no separate coverage estimate needed).  Returns ``(mapping, cov_area)``
    where ``mapping`` has [cid, node, key, area_mi2] and ``cov_area`` is per-cid mi^2."""
    import geopandas as gpd
    from shapely import box

    h = step / 2.0
    sq = gpd.GeoDataFrame(
        {"key": cells["key"].to_numpy()},
        geometry=[box(x - h, y - h, x + h, y + h)
                  for x, y in zip(cells["lon"].to_numpy(dtype=float),
                                  cells["lat"].to_numpy(dtype=float))],
        crs=crs,
    ).to_crs(_EQ_CRS)
    ov = gpd.overlay(sq, catch_eq, how="intersection", keep_geom_type=True)
    ov["area_mi2"] = ov.geometry.area / _M2_PER_MI2
    ov = ov[ov["area_mi2"] > 1e-6]
    mapping = ov[["cid", "node", "key", "area_mi2"]].copy()
    cov_area = mapping.groupby("cid")["area_mi2"].sum()   # cells tile -> overlap == covered
    return mapping, cov_area


def _voronoi_footprint_overlap(cells: pd.DataFrame, catch_eq, crs):
    """Weights for IRREGULAR HRU centroids (15cdec): each HRU owns its Voronoi cell clipped
    to its local grid footprint — a box of the HRU's own nearest-neighbour spacing (capped
    at ``_MAX_FOOTPRINT_DEG``).  This caps **edge** HRUs to their own cell so they cannot
    bleed into catchments they only graze.  Coverage is from the gap-free **unclipped**
    Voronoi, capped per HRU at its footprint area, so a fully-sampled catchment reads ~1.0
    while genuinely sparse ones stay low.  Returns ``(mapping, cov_area)``."""
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely import MultiPoint, box, voronoi_polygons

    pts = gpd.GeoDataFrame(
        cells[["key", "lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs=crs,
    )
    vor = voronoi_polygons(MultiPoint(list(pts.geometry)),
                           extend_to=box(*pts.total_bounds).buffer(0.2))
    vc = gpd.GeoDataFrame(geometry=list(vor.geoms), crs=crs)
    vc = gpd.sjoin(vc, pts[["key", "geometry"]], predicate="contains").drop(columns="index_right")

    # per-cell local spacing (median of the 4 nearest neighbours), capped -> footprint box
    lon = cells["lon"].to_numpy(); lat = cells["lat"].to_numpy()
    xy = np.c_[lon, lat]
    k = min(5, len(cells))
    nn = cKDTree(xy).query(xy, k=k)[0][:, 1:]
    s = np.minimum(np.median(nn, axis=1), _MAX_FOOTPRINT_DEG)
    fb = {key: box(x - hh, y - hh, x + hh, y + hh)
          for key, x, y, hh in zip(cells["key"], lon, lat, s / 2.0)}
    boxes = gpd.GeoSeries([fb[k] for k in vc["key"]], index=vc.index, crs=crs)

    # (a) WEIGHTS — Voronoi clipped to footprint box (anti-bleed), equal-area, overlap.
    vc_clip = gpd.GeoDataFrame({"key": vc["key"]},
                               geometry=vc.geometry.intersection(boxes),
                               crs=crs).to_crs(_EQ_CRS)
    ov = gpd.overlay(vc_clip, catch_eq, how="intersection", keep_geom_type=True)
    ov["area_mi2"] = ov.geometry.area / _M2_PER_MI2
    ov = ov[ov["area_mi2"] > 1e-6]
    mapping = ov[["cid", "node", "key", "area_mi2"]].copy()

    # (b) COVERAGE — unclipped Voronoi (gap-free) capped per HRU at its footprint area.
    box_area = dict(zip(vc["key"], boxes.to_crs(_EQ_CRS).area.to_numpy() / _M2_PER_MI2))
    vc_full = gpd.GeoDataFrame(vc[["key", "geometry"]], crs=crs).to_crs(_EQ_CRS)
    uov = gpd.overlay(vc_full, catch_eq, how="intersection", keep_geom_type=True)
    uov["vor_area"] = uov.geometry.area / _M2_PER_MI2
    uov["cov_area"] = np.minimum(uov["vor_area"], uov["key"].map(box_area))
    cov_area = uov.groupby("cid")["cov_area"].sum()
    return mapping, cov_area


def map_hrus_to_catchments(catchments, cells: pd.DataFrame, *,
                           covered_frac: float = COVERED_FRAC,
                           grid_step: float = _GRID_STEP_DEG):
    """Assign HRUs to catchments by **grid-footprint area overlap** (not centroids).

    Two regimes, auto-detected from the cell spacing (:func:`_is_regular_grid`):

    * **Regular 1/16-deg grid** (9unimp/11obs/12rim — each point is one grid cell): a
      simple square-cell overlap (:func:`_square_cell_overlap`).  The cells tile with no
      gaps/overlaps, so the overlap area is both the weight and the covered area.  (On a
      regular grid this is numerically identical to the Voronoi-footprint weights — the
      Voronoi cell of a regular lattice IS the grid square — just simpler.)
    * **Irregular sub-grid centroids** (15cdec): the Voronoi footprint scheme
      (:func:`_voronoi_footprint_overlap`), which a fixed square grid cannot replace —
      the centroids are ~0.021 deg apart so fixed 1/16-deg squares would overlap ~3x and
      double-count, over-weighting densely-sampled sub-regions.

    Returns ``(mapping, coverage)``:
      * ``mapping``  — [cid, node, key, lat, lon, area_mi2] per (catchment, overlapping
        HRU); ``area_mi2`` is the overlap area = the within-catchment weight.  An HRU may
        appear in several catchments.
      * ``coverage`` — one row per catchment with ``n_hru, hru_area_mi2, cov_frac,
        status`` in {covered, partial, outside}.  ``status`` uses ``covered_frac`` only as
        an informational label (it does NOT gate scoring — node inclusion is driven by the
        hand-edited crosswalk).
    """
    catch_eq = catchments[["cid", "node", "geometry"]].to_crs(_EQ_CRS)
    if _is_regular_grid(cells, step=grid_step):
        mapping, cov_area = _square_cell_overlap(cells, catch_eq, catchments.crs, grid_step)
    else:
        mapping, cov_area = _voronoi_footprint_overlap(cells, catch_eq, catchments.crs)

    ll = cells.set_index("key")
    mapping["lat"] = mapping["key"].map(ll["lat"])
    mapping["lon"] = mapping["key"].map(ll["lon"])
    mapping = mapping[["cid", "node", "key", "lat", "lon", "area_mi2"]].reset_index(drop=True)

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
# Basin -> CalSim node mapping: the hand-edited crosswalk
# --------------------------------------------------------------------------
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


#: rim **control points** whose unimpaired flow includes ungauged main-stem accretion that
#: CalSim represents only as ``Type=="Valley"`` polygons with **no** ``INFLOW`` series — the
#: "missing internal valley watersheds".  Only Sac R @ Bend Bridge (SRBB) has this among the
#: rim systems (~682 mi^2 of "Valley Watershed" draining to ``CT_BENDBRIDGE``); the other
#: valley control points (Colusa, Butte City, ...) are below-rim reaches, not rim inflow.
#: The merged GIS layer dissolves each into one series-less node ``<SYS>_VAL`` so the HRUs
#: there are modelled explicitly (a real SAC inflow) instead of being dropped as "outside".
VALLEY_CT_SYSTEM = {"CT_BENDBRIDGE": "SRBB"}
#: rim systems that carry a modelled valley-accretion node (derived from VALLEY_CT_SYSTEM).
VALLEY_SYSTEMS = frozenset(VALLEY_CT_SYSTEM.values())


def is_valley_arc(arc) -> bool:
    """True for a synthetic valley-accretion node arc (``I_<SYS>_VAL``) — modelled SAC
    runoff over the ungauged main-stem accretion, carrying no CalSim3 ``INFLOW`` series."""
    return str(arc).endswith("_VAL")


def valley_arc_for_system(system) -> str:
    """The valley-accretion node arc for a rim ``system`` (``I_<SYS>_VAL``)."""
    return f"I_{system}_VAL"


#: basin-level nesting (for the basin->node mapping): a basin whose inflow **includes**
#: another basin's nodes.  The only nest is Sac R @ Bend Bridge over Shasta, so BND's
#: anchor sum picks up every node assigned to SHA (i.e. ``I_SHSTA``) on top of its own.
BASIN_NESTS = {"BND": ["SHA"]}


def load_crosswalk(data_dir: str | Path = "data") -> pd.DataFrame:
    """The single hand-editable master crosswalk (``data/reference/calsim_crosswalk.csv``).

    Columns ``[arc, system, unimp_anchor, vic_basin, basin_15cdec, basin_11obs,
    basin_9unimp, in_calsim3]``.  This is the **authoritative source** for the basin->node
    mapping, rim-system membership, and VIC node names — edit it by hand; nothing in the
    pipeline overwrites it.
    """
    from . import calsim_dir
    return pd.read_csv(calsim_dir(data_dir) / "calsim_crosswalk.csv")


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
) -> pd.DataFrame:
    """Project the hand-editable crosswalk into a ``domain`` basin -> CalSim node table.

    The mapping is read straight from ``calsim_crosswalk.csv`` (``basin_<domain>`` column),
    with basin-level nesting applied (:data:`BASIN_NESTS`: Bend Bridge picks up Shasta's
    nodes).  No geography is computed here — edit the crosswalk to change assignments.

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


#: name of the merged "whole-basin" catchment layer (series-less sub-arcs dissolved into
#: the node carrying their flow + the ``<SYS>_VAL`` valley-accretion nodes); it lives in
#: ``calsim3.gpkg`` alongside the original layer.  See ``data/INVENTORY.md`` for how it
#: was derived.
MERGED_LAYER = "CalSim3_Merged"


def basin_footprints(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN,
                     *, simplify_deg: float = 0.01):
    """Per-basin HRU footprint union (lon/lat), the watershed each calibration basin
    represents — for drawing clean sub-system outlines on the coverage maps.

    Returns a dict ``{basin: shapely geometry}`` in the catchments' CRS (lon/lat),
    holes filled and lightly simplified so the outline is a single clean outer edge.
    """
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    from ..io import load_hru_table

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


def calsim_basin_polygons(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN):
    """Per-basin **CalSim3 catchment delineation** (lon/lat): the union of the merged-layer
    polygons the basin owns (crosswalk nodes + the valley-accretion node where the system
    has one) — the catchment geometry itself, NOT an HRU footprint.

    For building external eval sets directly on the CalSim3 delineations — the
    neuralhyd-ca LSTM comparison basis (2026-07-08): the LSTM has no HRU legacy, so it
    runs on the true catchment and its volume is depth x the canonical CalSim area
    (``basin_area_<domain>_calsim.csv``), matching the anchor convention exactly.  Node
    selection mirrors :func:`screened_footprint`; basins with no usable catchment (e.g.
    Tulare/Kern) fall back to their full HRU footprint (:func:`basin_footprints`).
    Hairline sliver rings between adjacent catchments are filled so each basin is a
    clean outer boundary.
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    full = basin_footprints(data_dir, domain)                # fallback only
    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True)
    nodes = derive_basin_nodes(data_dir, domain)
    sysmap = BASIN_RIM_SYSTEM.get(domain, {})
    out = {}
    for basin, geom in full.items():
        own = set(nodes.loc[nodes["basin"] == basin, "node"].astype(str))
        sysn = sysmap.get(basin)
        if sysn in VALLEY_SYSTEMS:
            own.add(valley_arc_for_system(sysn)[2:])
        catch_b = catch[catch["node"].astype(str).isin(own)]
        if not own or catch_b.empty:
            out[basin] = geom                                # no catchment -> HRU footprint
            continue
        u = unary_union(list(catch_b.geometry.values))
        polys = list(u.geoms) if isinstance(u, MultiPolygon) else [u]
        out[basin] = MultiPolygon([Polygon(p.exterior) for p in polys if p.area > 1e-6])
    return out


def screened_basin_footprints(data_dir: str | Path = "data", domain: str = DEFAULT_DOMAIN,
                              *, simplify_deg: float = 0.01):
    """Per-basin **GIS-screened** footprint (lon/lat): :func:`basin_footprints` clipped to
    the basin's own CalSim catchment polygons — the geometry counterpart of
    :func:`screened_footprint`, for building external eval sets on the clipped-HRU
    footprint.  Superseded for the LSTM comparison by :func:`calsim_basin_polygons`
    (the catchment delineation itself, 2026-07-08).  Node selection mirrors
    :func:`screened_footprint` exactly (crosswalk arcs + the valley-accretion node);
    basins with no usable catchment (e.g. Tulare/Kern) keep their full footprint.
    NOTE: clips EVERY basin — unlike the anchor, which screens only
    :data:`SCREENED_BASINS`.
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    full = basin_footprints(data_dir, domain, simplify_deg=simplify_deg)
    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True)
    nodes = derive_basin_nodes(data_dir, domain)
    sysmap = BASIN_RIM_SYSTEM.get(domain, {})
    out = {}
    for basin, geom in full.items():
        own = set(nodes.loc[nodes["basin"] == basin, "node"].astype(str))
        sysn = sysmap.get(basin)
        if sysn in VALLEY_SYSTEMS:
            own.add(valley_arc_for_system(sysn)[2:])
        catch_b = catch[catch["node"].astype(str).isin(own)]
        if not own or catch_b.empty:
            out[basin] = geom                            # same fallback as screened_footprint
            continue
        clipped = geom.intersection(unary_union(list(catch_b.geometry.values)))
        polys = ([clipped] if isinstance(clipped, Polygon)
                 else [p for p in getattr(clipped, "geoms", []) if isinstance(p, Polygon)])
        out[basin] = MultiPolygon([p for p in polys if p.area > 0]) if polys else geom
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
    comp_cache: dict | None = None,
    parallel: bool = False,
):
    """Simulate area-weighted inflow for every covered catchment.

    With ``route=False`` (default) each catchment's inflow is the area-weighted
    HRU **local runoff** (SMA ``surf+base``).  With ``route=True`` each HRU's flow
    is first Lohmann-routed to the catchment outlet using a **special CalSim
    flowlen** (:func:`assign_flowlens`) before area-weighting — a physically routed
    daily inflow at the CalSim node (negligible at monthly aggregation, matters for
    daily shape).

    ``parallel=True`` computes the per-cell local runoff via the Numba ``prange``
    kernel (:func:`sacsma.model.run_local_runoff_parallel`) — **bit-exact** vs the
    serial path (each cell is independent), only available for ``route=False``, and
    it bypasses ``comp_cache`` (the kernel computes every cell itself).

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
    if parallel and not route and run_local_runoff_parallel is not None:
        # bit-exact with the serial loop below (per-cell, no cross-HRU reduction);
        # bypasses comp_cache (the kernel computes every cell itself).
        mat = run_local_runoff_parallel(keys, meta, params, forcing)
        local = {k: mat[i] for i, k in enumerate(keys)}
    else:
        for i, k in enumerate(keys):
            if progress and (i % 500 == 0):
                print(f"  {label}: HRU {i + 1}/{len(keys)}", flush=True)
            c = forcing.pos[k]
            # components cached (and shared with the basin-anchor run_basin pass via comp_cache);
            # local runoff is just surf+base, identical to run_hru_local.
            surf, base = run_hru_components_cached(
                comp_cache, domain, k, forcing.prcp[c], forcing.tavg[c], forcing.doy, forcing.is_leap,
                lat=float(meta.at[k, "lat"]), elev=float(meta.at[k, "elev"]), ga_row=params.loc[k])
            if route:
                comp[k] = (surf, base)
            else:
                local[k] = surf + base

    if route:
        from .. import parameters as P
        from ..routing import lohmann

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
