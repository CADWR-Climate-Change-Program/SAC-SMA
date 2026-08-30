"""Build data/dpl_entities/entity_cells.csv — per-entity cell sets and weights.

One row per (entity, region grid cell) with the proportional square-overlap
area weight (overlap_mi2). Outlet coordinates stay in the
registry (entities.csv) only.

Cell-set source per registry `delineation`:

  arcs                   22 entities (9 uf_monthly, 13 cdec_daily). Union of
                         CalSim3_Merged polygons selected by the registry arc
                         list; per-cell weight = sum of cell-square x polygon
                         overlap across the entity's arcs (the arcs partition
                         the footprint).
  usgs_gpkg              69 usgs_daily. The gauge's delineated watershed
                         polygon (usgs/gis/usgs_watersheds.gpkg).
  sacsma_15cdec_gis      4 Tulare basins (ISB PNF SCC TRM — no CalSim
                         polygons exist). The original SAC-SMA boundary
                         polygons (data/cdec15/gis/SACSMA_15CDEC.geojson),
                         mapped square-overlap like the USGS watersheds.

I_RUB002 (FOL's arc list) has no CalSim3_Merged polygon — its terrain was
dissolved into MFA025, so the footprint stays complete and the arc id is the
one allowed skip.

Usage (sacsma conda env, from the repo root):
    python dataprep/build_entity_cells.py [--data-dir data]
        [--out data/dpl_entities/entity_cells.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sacsma.calsim.catchments import (  # noqa: E402
    MERGED_LAYER,
    load_catchments,
    map_hrus_to_catchments,
)
from sacsma.io import norm_grid_key  # noqa: E402

SQMI_PER_KM2 = 0.386102
# I_RUB002's terrain was dissolved into MFA025 (23.82 -> 141.92 mi^2).
DISSOLVED_ARCS = {"RUB002"}
TULARE = ("ISB", "PNF", "SCC", "TRM")
COLUMNS = ["entity_id", "key", "lat", "lon", "overlap_mi2"]


def _map_weights(catch: gpd.GeoDataFrame, cells: pd.DataFrame,
                 what: str) -> pd.DataFrame:
    mapping, coverage = map_hrus_to_catchments(catch, cells)
    bad = coverage[coverage["status"] != "covered"]
    if len(bad):
        print(f"{what}: {len(bad)} catchments not fully covered")
        print(bad[["node", "n_hru", "hru_area_mi2", "cov_frac", "status"]]
              .to_string(index=False))
    return mapping


def build(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (entity_cells, checks) — checks has one gate row per entity."""
    ent = pd.read_csv(data_dir / "dpl_entities" / "entities.csv",
                      dtype={"site_id": str})
    cells = pd.read_csv(
        data_dir / "region" / "grid_cells.csv")[["key", "lat", "lon"]]

    # CalSim arc entities. All 200 Merged polygons are Type=="Rim" and node
    # labels are unique, so one mapping pass serves every arc entity.
    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True)
    sq_mi = catch.set_index("node")["sq_mi"]
    arc_map = _map_weights(catch, cells, "calsim arcs")

    # USGS watersheds, shaped like a load_catchments frame. make_valid is
    # belt-and-braces: this layer is valid on this checkout, but at least
    # one other environment (different GEOS/PROJ) produced zero areas for
    # it without the repair.
    ws = gpd.read_file(data_dir / "usgs" / "gis" / "usgs_watersheds.gpkg")
    ws_catch = gpd.GeoDataFrame(
        {"cid": range(len(ws)), "node": ws["gid"].astype(str),
         "sq_mi": ws["area_km2_delineated"] * SQMI_PER_KM2},
        geometry=ws["geometry"].apply(make_valid), crs=ws.crs)
    ws_sqmi = ws_catch.set_index("node")["sq_mi"]
    usgs_map = _map_weights(ws_catch, cells, "usgs watersheds")

    # Tulare 4: the original SAC-SMA boundary polygons, mapped square-overlap
    # like every other polygon source. make_valid is belt-and-braces here
    # (all four are valid on this checkout; one of the other 11 is not).
    tul = gpd.read_file(data_dir / "cdec15" / "gis" / "SACSMA_15CDEC.geojson")
    tul = tul[tul["Name"].isin(TULARE)].reset_index(drop=True)
    tul_ea = tul.to_crs(5070)
    tul_catch = gpd.GeoDataFrame(
        {"cid": range(len(tul)), "node": tul["Name"],
         "sq_mi": tul_ea.geometry.area / 1e6 * SQMI_PER_KM2},
        geometry=tul.geometry.apply(make_valid), crs=tul.crs)
    tul_sqmi = tul_catch.set_index("node")["sq_mi"]
    tul_map = _map_weights(tul_catch, cells, "sacsma 15cdec (Tulare)")

    frames, checks = [], []
    for r in ent.itertuples():
        if r.delineation == "arcs":
            arcs = [a.removeprefix("I_") for a in str(r.arcs).split(";")]
            missing = set(arcs) - set(sq_mi.index)
            assert missing <= DISSOLVED_ARCS, (r.entity_id, sorted(missing))
            arcs = [a for a in arcs if a in sq_mi.index]
            sub = arc_map[arc_map["node"].isin(arcs)]
            g = (sub.groupby("key", as_index=False)
                    .agg(lat=("lat", "first"), lon=("lon", "first"),
                         overlap_mi2=("area_mi2", "sum")))
            geom_ref = float(sq_mi.loc[arcs].sum())
        elif r.delineation == "usgs_gpkg":
            sub = usgs_map[usgs_map["node"] == r.site_id]
            g = (sub.groupby("key", as_index=False)
                    .agg(lat=("lat", "first"), lon=("lon", "first"),
                         overlap_mi2=("area_mi2", "sum")))
            # unrounded delineated area — the registry's area_mi2 is rounded
            # to 2 decimals, which alone is >0.1% on the smallest basins
            geom_ref = float(ws_sqmi[r.site_id])
        elif r.delineation == "sacsma_15cdec_gis":
            sub = tul_map[tul_map["node"] == r.site_id]
            g = (sub.groupby("key", as_index=False)
                    .agg(lat=("lat", "first"), lon=("lon", "first"),
                         overlap_mi2=("area_mi2", "sum")))
            geom_ref = float(tul_sqmi[r.site_id])
        else:
            raise ValueError(f"{r.entity_id}: delineation {r.delineation!r}")
        assert len(g), f"{r.entity_id}: empty cell set"
        g = g.assign(entity_id=r.entity_id,
                     key=g["key"].map(norm_grid_key))
        frames.append(g[["entity_id", "key", "lat", "lon", "overlap_mi2"]])
        checks.append((r.entity_id, r.family, r.delineation, len(g),
                       float(g["overlap_mi2"].sum()), geom_ref,
                       float(r.area_mi2)))

    order = pd.CategoricalDtype(ent["entity_id"], ordered=True)
    df = (pd.concat(frames, ignore_index=True)
            .astype({"entity_id": order})
            .sort_values(["entity_id", "key"], ignore_index=True)
            .astype({"entity_id": str}))
    df["overlap_mi2"] = df["overlap_mi2"].round(6)
    checks = pd.DataFrame(checks, columns=[
        "entity_id", "family", "delineation", "n_cells", "overlap_sum",
        "geom_ref", "area_mi2"])
    checks["ratio"] = checks["overlap_sum"] / checks["geom_ref"]
    return df[COLUMNS], checks


def gates(df: pd.DataFrame, checks: pd.DataFrame, grid_keys: set) -> None:
    ent_ids = checks["entity_id"]
    assert len(checks) == 95 and ent_ids.is_unique
    assert set(df["entity_id"]) == set(ent_ids)
    assert not df.duplicated(["entity_id", "key"]).any()
    assert (df["overlap_mi2"] > 0).all()
    assert df["key"].isin(grid_keys).all()

    # Geometric consistency: mapped overlap reproduces each footprint's own
    # reference area (arc-attribute sum / delineated area / CADWR area).
    worst = (checks["ratio"] - 1).abs()
    off = checks[worst > 1e-3]
    if len(off):
        print("entities off their reference area by >0.1%:")
        print(off[["entity_id", "n_cells", "overlap_sum", "geom_ref",
                   "ratio"]].to_string(index=False))
    assert worst.max() < 2e-3, checks.loc[worst.idxmax()].to_dict()

    u = df[df["entity_id"].str.startswith("usgs_")]
    usgs_worst = (checks.loc[checks["family"] == "usgs_daily", "ratio"] - 1
                  ).abs().max()
    assert len(u) == 1625, len(u)
    assert u["key"].nunique() == 1109, u["key"].nunique()
    assert usgs_worst < 1e-3, usgs_worst

    # Every UF's arc-attribute sum must reproduce the registry area (the
    # uf_locations.area_mi2_calsim copy).
    ufs = checks[checks["family"] == "uf_monthly"]
    uf_dev = (ufs["geom_ref"] - ufs["area_mi2"]).abs()
    assert (uf_dev < 0.01).all(), ufs.loc[uf_dev >= 0.01].to_dict("records")

    # Arc-selection gates on the arc-attribute sums (geom_ref). The mapped
    # sums equal the polygons' true geometric areas; they sit up to ~0.14%
    # above geom_ref mostly because the SQ_MI attributes run systematically
    # ~0.05-0.12% below the true areas (single-arc uf_15 is +0.09% with no
    # overlap in play), plus a smaller once-per-arc count of arc-overlap
    # slivers (largest pairwise overlap 1.4% of the smaller arc;
    # entity-level effect <= 0.04%). Worst combined deviation: uf_10 +0.14%.
    named = checks.set_index("entity_id")["geom_ref"]
    # cdec_BND: 9,083.72 = the 8,401.7 strip+SHSTA union + I_SRBB_VAL (682.0),
    # added via EXTRA_ARCS in build_entities.py (the Bend Bridge FNF drainage
    # includes the valley floor; depth basis stays the published 8,900).
    for eid, want in (("cdec_CLE", 692.86), ("cdec_SHA", 6588.45),
                      ("cdec_FOL", 1863.78), ("cdec_BND", 9083.72)):
        assert abs(named[eid] - want) < 0.1, (eid, named[eid])

    tul = checks[checks["delineation"] == "sacsma_15cdec_gis"]
    assert dict(zip(tul["entity_id"], tul["n_cells"], strict=True)) == {
        "cdec_ISB": 177, "cdec_PNF": 133, "cdec_SCC": 40, "cdec_TRM": 62}

    # The training basis: 2,654 distinct cells. The de-dup drops left the
    # union unchanged at 2,646 (every dropped monthly twin's cells stay via
    # its daily twin; obs11_TNL's extra I_LWSTN cells via usgs_11525500);
    # the Tulare remap onto the SACSMA_15CDEC polygons then added 8 edge
    # cells (the inherited cell sets were a strict subset of the new).
    # Statics cover all 4,410 region cells (a77e4a8) — no gap from the
    # additions. An earlier full-rim tally logged 2,853 cells (all 200
    # Merged polygons; exact recompute 2,847).
    union = df["key"].nunique()
    assert union == 2654, union

    fam = checks.groupby("family").agg(
        entities=("entity_id", "size"), rows=("n_cells", "sum"),
        worst_dev=("ratio", lambda r: (r - 1).abs().max()))
    fam["cells"] = [df.loc[df["entity_id"].isin(
        checks.loc[checks["family"] == f, "entity_id"]), "key"].nunique()
        for f in fam.index]
    print(fam.to_string())
    print(f"gates ok: {len(df)} rows, {union} distinct cells; "
          f"USGS {len(u)}/1625 rows, {u['key'].nunique()}/1109 cells; "
          f"CLE {named['cdec_CLE']:.2f}, SHA {named['cdec_SHA']:.2f}, "
          f"FOL {named['cdec_FOL']:.2f}, BND {named['cdec_BND']:.2f} mi^2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=Path("data"), type=Path)
    ap.add_argument("--out",
                    default=Path("data/dpl_entities/entity_cells.csv"),
                    type=Path)
    args = ap.parse_args()

    df, checks = build(args.data_dir)
    grid_keys = set(pd.read_csv(
        args.data_dir / "region" / "grid_cells.csv")["key"])
    gates(df, checks, grid_keys)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, "
          f"{df['entity_id'].nunique()} entities")


if __name__ == "__main__":
    main()
