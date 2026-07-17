"""Build the canonical REGION 1/16-deg cell list -> data/region/grid_cells.csv.

The auxiliary-data region is the UNION of

* the four modeling domains' grid cells — ``15cdec_grid`` (2074 cells, from
  its hruinfo) plus the CalSim domains ``9unimp``/``11obs``/``12rim`` (their
  forcing-store keys), and
* the FULL CalSim3 footprint — every 1/16-deg cell whose rectangle intersects
  any polygon of ``data/calsim/gis/calsim3.gpkg`` (both layers: all Rim
  watersheds incl. Goose Lake, and the Valley polygons), so EVERY CalSim3 rim
  location is coverable, not just the ones the modeling domains carry.

Keys are normalized to the ``sacsma.io`` 5-decimal ``<lat>_<lon>`` convention
(``round(x, 5)``; the calsim stores carry 6-decimal fixed-format keys).
Membership flags ``in_<domain>`` + ``in_calsim3_fp`` record what touches each
cell.

Every region auxiliary ingest (GEE ET/SWE obs, GLEAM/FLUXCOM, statics
consolidation, the WGEN forcing master) targets THIS cell list.  Cells must
exist in the local WGEN NonDetrend-Unsplit statewide store — the daily
forcing master source (dataprep/wgen_forcing.py): a missing DOMAIN cell
aborts (key drift); missing FOOTPRINT-sweep cells are dropped with a report
(the 14 dropped are Delta/open-water cells the land-only store excludes).

Usage:
    python dataprep/build_region_grid.py [--wgen-dir <path>] [--out <csv>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

WGEN_DIR = r"C:\Users\warnold_la\Local\WGEN_NonDetrend_Unsplit_Statewide"
GPKG = "data/calsim/gis/calsim3.gpkg"
GPKG_LAYERS = ("CalSim3_And_GooseLake", "CalSim3_Merged")
DOMAINS = ("15cdec_grid", "9unimp", "11obs", "12rim")
CELL = 1.0 / 16.0


def norm_key(k: str) -> str:
    lat, lon = str(k).split("_")
    return f"{round(float(lat), 5)}_{round(float(lon), 5)}"


def domain_cells(root: Path, domain: str) -> set[str]:
    if domain == "15cdec_grid":
        hi = pd.read_csv(root / "data" / "cdec15_grid" / "hruinfo.csv")
        return {norm_key(k) for k in hi["key"].astype(str)}
    ds = xr.open_dataset(root / "data" / "calsim" / "forcing"
                         / f"historical_livneh_unsplit_{domain}.nc")
    keys = {norm_key(k) for k in np.asarray(ds["key"]).astype(str)}
    ds.close()
    return keys


def footprint_cells(root: Path) -> set[str]:
    """Cells whose 1/16-deg rectangle intersects the CalSim3 footprint."""
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union

    geoms = []
    for layer in GPKG_LAYERS:
        g = gpd.read_file(root / GPKG, layer=layer).to_crs(4326)
        geoms.append(unary_union(g.geometry.values))
    fp = unary_union(geoms)

    minx, miny, maxx, maxy = fp.bounds
    h = CELL / 2.0
    # Livneh cell centers sit at 0.03125 + k/16
    lat0 = np.floor((miny - 0.03125) / CELL) * CELL + 0.03125
    lon0 = np.floor((minx - 0.03125) / CELL) * CELL + 0.03125
    lats = np.arange(lat0, maxy + CELL, CELL)
    lons = np.arange(lon0, maxx + CELL, CELL)
    keys = []
    cells = []
    for lat in lats:
        for lon in lons:
            keys.append(f"{round(float(lat), 5)}_{round(float(lon), 5)}")
            cells.append(box(lon - h, lat - h, lon + h, lat + h))
    cand = gpd.GeoDataFrame({"key": keys}, geometry=cells, crs=4326)
    hit = cand[cand.intersects(fp)]
    return set(hit["key"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wgen-dir", default=WGEN_DIR)
    ap.add_argument("--out", default="data/region/grid_cells.csv")
    args = ap.parse_args()

    root = Path(".")
    members = {d: domain_cells(root, d) for d in DOMAINS}
    dom_union = set().union(*members.values())
    fp = footprint_cells(root)

    wgen = Path(args.wgen_dir)
    bad_dom = [k for k in sorted(dom_union) if not (wgen / f"data_{k}").exists()]
    if bad_dom:
        raise SystemExit(f"{len(bad_dom)} DOMAIN cells absent from the WGEN "
                         f"store (first: {bad_dom[:3]}) — key drift?")
    fp_dropped = sorted(k for k in fp - dom_union
                        if not (wgen / f"data_{k}").exists())
    if fp_dropped:
        print(f"NOTE: dropping {len(fp_dropped)} footprint-sweep cells absent "
              f"from the land-only WGEN store (Delta/open-water cells): "
              f"{fp_dropped[:5]}{'...' if len(fp_dropped) > 5 else ''}")
    fp -= set(fp_dropped)

    union = sorted(dom_union | fp,
                   key=lambda k: tuple(float(x) for x in k.split("_")))
    rows = []
    for k in union:
        lat, lon = (float(x) for x in k.split("_"))
        rows.append({"key": k, "lat": lat, "lon": lon,
                     **{f"in_{d}": int(k in members[d]) for d in DOMAINS},
                     "in_calsim3_fp": int(k in fp)})
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    counts = ", ".join(f"{d}: {len(members[d])}" for d in DOMAINS)
    n_new = len(union) - len(dom_union)
    print(f"wrote {out}: {len(df)} region cells ({counts}; calsim3 footprint: "
          f"{len(fp)}, adding {n_new} beyond the domains); "
          f"lat {df.lat.min():.5f}..{df.lat.max():.5f}, "
          f"lon {df.lon.min():.5f}..{df.lon.max():.5f}; all in WGEN store")


if __name__ == "__main__":
    main()
