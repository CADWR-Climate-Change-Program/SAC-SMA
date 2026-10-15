"""Write the REGION statics store from the cell-footprint sample of every region cell.

Writes ``data/inputs/grid/soilveg_continuous.csv`` + ``data/inputs/grid/lai_climatology.csv``
covering every cell of ``data/inputs/grid/grid_cells.csv`` (4410 cells, the full
CalSim3-gpkg intersection with the 1/16-deg grid) from
``data/inputs/grid/{soilveg_continuous,lai_climatology}_raster.csv``, the output of
``data/inputs/grid/sample_gis.py region`` (the ``sacsma-gis`` env): each cell the mean of its
13 x 13 sub-points, with the terrain from the sub-point elevations (``sample_gis.run_footprint``).

Gate: the ``15cdec_grid`` sidecars (``data/inputs/domains/15cdec_grid/``, the inputs the dPL
parameter network of the ladder's physical rungs was trained on) must be reproduced at every
one of their cells, to 1e-9 relative.  The one exception is the leaf area at cells with a
sub-point MODIS never fills (open water; ``lai_gapfill`` > 0 in the sample): such a sub-point
takes the climatology of the nearest valid sub-point, and the region sample has more
sub-points to draw from than the ``15cdec_grid`` one had, so the leaf area there must agree
within 0.02 (the peak day within one 8-day bin).  A cell the sample leaves without a value
stops the build if a modeling domain uses it and is left out otherwise.  Keys are normalized
to the 5-decimal ``sacsma.io`` convention.

Usage:
    python data/inputs/grid/build_region_statics.py [--grid <csv>] [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import layout  # noqa: E402

#: the domain whose committed sidecars the footprint sample must reproduce
GATE_DOMAIN = "15cdec_grid"
#: relative tolerance of the gate
GATE_RTOL = 1e-9
#: the gate's leaf-area tolerance at cells with an all-fill MODIS sub-point (LAI; peak day, d)
LAI_ATOL, PEAK_ATOL = 0.02, 8


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


def write_table(root: Path, grid: pd.DataFrame, stem: str, out: Path,
                lai_fill: pd.Series) -> None:
    """One of the two tables (``stem``) from the footprint sample, after the gate."""
    raster = _load(layout.raster_sample(root / "data", stem))
    gate = _load(layout.domain_dir(root / "data", GATE_DOMAIN) / f"{stem}.csv")
    missing = set(gate.columns) - set(raster.columns)
    if missing:
        raise SystemExit(f"{stem}_raster: missing columns {sorted(missing)}")
    raster = raster[list(gate.columns)]

    shared = gate.index.intersection(raster.index)
    if len(shared) != len(gate):
        raise SystemExit(f"{stem}: {len(gate) - len(shared)} {GATE_DOMAIN} cells not sampled")
    a = raster.loc[shared].to_numpy(np.float64)
    b = gate.loc[shared].to_numpy(np.float64)
    bad = ~np.isclose(a, b, rtol=GATE_RTOL, atol=0.0, equal_nan=True)
    lai = np.array([stem == "lai_climatology" or c.startswith("lai_") for c in gate.columns])
    filled = lai_fill.reindex(shared).fillna(0).to_numpy() > 0
    loose = filled[:, None] & lai[None, :]
    peak = np.array([c == "lai_peak_doy" for c in gate.columns])[None, :]
    near = np.abs(a - b) <= np.where(peak, PEAK_ATOL, LAI_ATOL)
    if (bad & ~(loose & near)).any():
        bad &= ~(loose & near)
        cols = gate.columns[bad.any(axis=0)].tolist()
        raise SystemExit(f"{stem}: the footprint sample differs from the {GATE_DOMAIN} sidecar "
                         f"at {int(bad.any(axis=1).sum())} cells (columns {cols})")
    n_lai = int((bad & loose).any(axis=1).sum())
    if n_lai:
        d = np.abs(a - b)[:, lai & ~peak[0]][(bad & loose).any(axis=1)]
        print(f"  {stem}: reproduces the {GATE_DOMAIN} sidecar at {len(shared) - n_lai} of "
              f"{len(shared)} cells; at the other {n_lai}, cells with an all-fill MODIS "
              f"sub-point, the leaf area is within {d.max():.4f}")
    else:
        print(f"  {stem}: reproduces the {GATE_DOMAIN} sidecar at all {len(shared)} cells")

    df = raster.reindex(grid["key"])
    uncovered = df.index[df.isna().all(axis=1)]
    if len(uncovered):
        dom_cols = [c for c in grid.columns if c.startswith("in_") and c != "in_calsim3_fp"]
        gd = grid.set_index("key")
        dom_miss = [k for k in uncovered if gd.loc[k, dom_cols].sum() > 0]
        if dom_miss:
            raise SystemExit(f"{stem}: {len(dom_miss)} DOMAIN cells uncovered "
                             f"(first: {dom_miss[:3]})")
        print(f"  {stem}: {len(uncovered)} cells have no value at all — left out")
        df = df.drop(index=uncovered)
    df.index.name = "key"
    df.to_csv(out)
    print(f"wrote {out}: {len(df)} cells")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(layout.grid_cells()))
    ap.add_argument("--out-dir", default=str(layout.grid_dir()))
    args = ap.parse_args()
    root = Path(".")
    grid = pd.read_csv(args.grid)
    grid["key"] = grid["key"].map(norm_key)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lai_fill = _load(layout.raster_sample(root / "data", "soilveg_continuous"))["lai_gapfill"]
    for stem in ("soilveg_continuous", "lai_climatology"):
        write_table(root, grid, stem, out_dir / f"{stem}.csv", lai_fill)


if __name__ == "__main__":
    main()
