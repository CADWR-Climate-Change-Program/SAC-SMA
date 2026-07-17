"""Consolidate the per-domain static sidecars into the REGION statics store.

Writes ``data/region/soilveg_continuous.csv`` + ``data/region/lai_climatology.csv``
covering every cell of ``data/region/grid_cells.csv``, by consolidating the four
COMMITTED per-domain sidecars (no raster work — the committed stores define
correctness):

* ``data/cdec15_grid/{soilveg_continuous,lai_climatology}.csv`` — per-cell rows
  in the **cell-footprint-mean** convention (e.g. ``dem_elev`` = mean 3DEP
  elevation over the 1/16-deg cell).  This is the convention the dPL parameter
  net was trained on; it wins wherever available.
* ``data/calsim/{soilveg_continuous,lai_climatology}_<d>.csv`` — per-cell rows
  (repeated per HRU; identical within a key) in the **cell-center point-sample**
  convention (``dem_elev`` = 3DEP at the cell center; verified 2026-07-16).
  Fills the calsim-only cells.

The two conventions differ (median |Δdem_elev| ≈ 93 m over shared cells); the
``src`` column records which store each row came from so the seam is explicit.
Keys are normalized to the 5-decimal ``sacsma.io`` convention.

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

    # cross-domain consistency (same point-sample convention -> expect identical)
    disagree = 0
    for i, a in enumerate(CALSIM_ORDER):
        for b in CALSIM_ORDER[i + 1:]:
            shared = cal[a].index.intersection(cal[b].index)
            if len(shared):
                d = (cal[a].loc[shared].select_dtypes("number")
                     - cal[b].loc[shared].select_dtypes("number")).abs()
                disagree += int((d.max(axis=1) > 1e-6).sum())

    prim = prim.assign(src="cdec15_grid")
    parts = [prim]
    covered = set(prim.index)
    for d in CALSIM_ORDER:
        keep = cal[d].index.difference(covered)
        parts.append(cal[d].loc[keep].assign(src=f"calsim_{d}"))
        covered |= set(keep)
    df = pd.concat(parts).reindex(grid["key"])
    uncovered = df.index[df["src"].isna()]
    if len(uncovered):
        # footprint-sweep cells beyond the modeling domains have no committed
        # sidecar — they stay ABSENT from the statics store (documented gap;
        # a verified raster ingest is the fill path). Domain cells must be there.
        dom_cols = [c for c in grid.columns if c.startswith("in_")
                    and c != "in_calsim3_fp"]
        gd = grid.set_index("key")
        dom_miss = [k for k in uncovered if gd.loc[k, dom_cols].sum() > 0]
        if dom_miss:
            raise SystemExit(f"{stem}: {len(dom_miss)} DOMAIN cells uncovered "
                             f"(first: {dom_miss[:3]})")
        print(f"  {stem}: {len(uncovered)} footprint-only cells have no "
              "committed statics source — left out (see dataprep/README.md)")
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
