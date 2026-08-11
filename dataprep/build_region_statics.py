"""Consolidate the per-domain static sidecars into the REGION statics store.

Writes ``data/region/soilveg_continuous.csv`` + ``data/region/lai_climatology.csv``
covering every cell of ``data/region/grid_cells.csv`` (4410 cells, the full
CalSim3-gpkg intersection with the 1/16-deg grid), by consolidating four
COMMITTED per-domain sidecars plus a full-coverage raster layer:

* ``data/cdec15_grid/{soilveg_continuous,lai_climatology}.csv`` — per-cell rows
  in the **cell-footprint-mean** convention (e.g. ``dem_elev`` = mean 3DEP
  elevation over the 1/16-deg cell).  This is the convention the dPL parameter
  net was trained on; it wins wherever available.
* ``data/calsim/{soilveg_continuous,lai_climatology}_<d>.csv`` — per-cell rows
  (repeated per HRU; identical within a key) in the **cell-center point-sample**
  convention (``dem_elev`` = 3DEP at the cell center; verified 2026-07-16).
  Fills the calsim-only cells.
* ``data/region/{soilveg_continuous,lai_climatology}_raster.csv`` — produced by
  ``data/raw_gis/sample_gis.py region`` (the ``sacsma-gis`` env): EVERY region
  cell, point-sampled at its center from the raw POLARIS/LANDFIRE/3DEP/MODIS-LAI
  stack, same convention as the calsim sidecars (verified to reproduce them
  exactly, see ``dataprep/README.md``).  Fills whatever the four sidecars above
  still miss, so the region store now covers the full 4410-cell grid with no
  documented gap.

The two point-sample-vs-footprint-mean conventions differ (median |Δdem_elev|
≈ 93 m over shared cells); the ``src`` column records which store each row
came from so the seam is explicit. Keys are normalized to the 5-decimal
``sacsma.io`` convention.

Usage:
    python dataprep/build_region_statics.py [--grid <csv>] [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

#: fill order for cells absent from cdec15_grid (values are identical across
#: the calsim domains except a handful of lai_* columns — reported at run time)
CALSIM_ORDER = ("11obs", "12rim", "9unimp")


def norm_key(k: str) -> str:
    lat, lon = str(k).split("_")
    return f"{round(float(lat), 5)}_{round(float(lon), 5)}"


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    key_col = "key" if "key" in df.columns else "cellkey"
    df["key"] = df[key_col].map(norm_key)
    if key_col != "key":
        df = df.drop(columns=[key_col])
    return df.drop_duplicates("key").set_index("key")


def consolidate(root: Path, grid: pd.DataFrame, stem: str, out: Path) -> None:
    prim = _load(root / "data" / "cdec15_grid" / f"{stem}.csv")
    cols = list(prim.columns)
    cal = {}
    for d in CALSIM_ORDER:
        df = _load(root / "data" / "calsim" / f"{stem}_{d}.csv")
        missing = set(cols) - set(df.columns)
        if missing:
            raise SystemExit(f"{stem}_{d}: missing columns {sorted(missing)}")
        extra = sorted(set(df.columns) - set(cols))
        if extra:   # e.g. the calsim-only lai_gapfill flag — schema follows cdec15_grid
            print(f"  {stem}_{d}: dropping calsim-only columns {extra}")
        cal[d] = df[cols]

    raster_path = root / "data" / "region" / f"{stem}_raster.csv"
    raster = None
    if raster_path.exists():
        rdf = _load(raster_path)
        missing = set(cols) - set(rdf.columns)
        if missing:
            raise SystemExit(f"{stem}_raster: missing columns {sorted(missing)}")
        extra = sorted(set(rdf.columns) - set(cols))
        if extra:   # e.g. lai_gapfill — schema follows cdec15_grid
            print(f"  {stem}_raster: dropping raster-only columns {extra}")
        raster = rdf[cols]

    # cross-domain consistency (same point-sample convention -> expect identical)
    disagree = 0
    for i, a in enumerate(CALSIM_ORDER):
        for b in CALSIM_ORDER[i + 1:]:
            shared = cal[a].index.intersection(cal[b].index)
            if len(shared):
                d = (cal[a].loc[shared].select_dtypes("number")
                     - cal[b].loc[shared].select_dtypes("number")).abs()
                disagree += int((d.max(axis=1) > 1e-6).sum())

    if raster is not None:
        # reproduction gate (dataprep/README.md): the raster layer shares the
        # calsim point-sample convention, so it should reproduce those rows
        # exactly wherever both cover a cell
        cal_all = pd.concat([cal[d] for d in CALSIM_ORDER]).groupby(level=0).first()
        shared = raster.index.intersection(cal_all.index)
        if len(shared):
            d = (raster.loc[shared].select_dtypes("number")
                 - cal_all.loc[shared].select_dtypes("number")).abs()
            n_mismatch = int((d.max(axis=1) > 1e-6).sum())
            print(f"  {stem}_raster: reproduces {len(shared)} calsim point-sample "
                  f"cells, {n_mismatch} mismatches >1e-6")

    prim = prim.assign(src="cdec15_grid")
    parts = [prim]
    covered = set(prim.index)
    for d in CALSIM_ORDER:
        keep = cal[d].index.difference(covered)
        parts.append(cal[d].loc[keep].assign(src=f"calsim_{d}"))
        covered |= set(keep)
    if raster is not None:
        keep = raster.index.difference(covered)
        parts.append(raster.loc[keep].assign(src="region_raster"))
        covered |= set(keep)
    df = pd.concat(parts).reindex(grid["key"])
    uncovered = df.index[df["src"].isna()]
    if len(uncovered):
        # cells no source could cover at all (e.g. a raster point that lands on
        # water/nodata even after the sampler's own gap-fill) — left out rather
        # than fabricated. Domain cells must be there.
        dom_cols = [c for c in grid.columns if c.startswith("in_")
                    and c != "in_calsim3_fp"]
        gd = grid.set_index("key")
        dom_miss = [k for k in uncovered if gd.loc[k, dom_cols].sum() > 0]
        if dom_miss:
            raise SystemExit(f"{stem}: {len(dom_miss)} DOMAIN cells uncovered "
                             f"(first: {dom_miss[:3]})")
        print(f"  {stem}: {len(uncovered)} cells have no statics source at all "
              "— left out (see dataprep/README.md)")
        df = df.dropna(subset=["src"])
    df.index.name = "key"
    df.to_csv(out)
    n_src = df["src"].value_counts().to_dict()
    print(f"wrote {out}: {len(df)} cells {n_src}"
          + (f"; {disagree} cross-calsim cell disagreements >1e-6" if disagree else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="data/region/grid_cells.csv")
    ap.add_argument("--out-dir", default="data/region")
    args = ap.parse_args()
    root = Path(".")
    grid = pd.read_csv(args.grid)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    consolidate(root, grid, "soilveg_continuous", out_dir / "soilveg_continuous.csv")
    consolidate(root, grid, "lai_climatology", out_dir / "lai_climatology.csv")


if __name__ == "__main__":
    main()
