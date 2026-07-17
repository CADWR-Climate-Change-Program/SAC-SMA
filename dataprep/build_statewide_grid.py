"""Build the canonical STATEWIDE 1/16-deg cell list -> data/statewide/grid_cells.csv.

The statewide grid is defined by the WGEN NonDetrend Unsplit Statewide store
(one ``data_<lat>_<lon>`` file per cell, 13,786 cells over California) — the
same 1/16-deg Livneh grid every domain store subsets.  Every statewide
auxiliary ingest (GEE ET/SWE obs, LAI, soilveg, tminmax, forcing) targets THIS
cell list, keyed ``<lat>_<lon>`` exactly as ``sacsma.io`` forcing keys.

Usage:
    python dataprep/build_statewide_grid.py [--wgen-dir <path>] [--out <csv>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

WGEN_DIR = r"C:\Users\warnold_la\Local\WGEN_NonDetrend_Unsplit_Statewide"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wgen-dir", default=WGEN_DIR)
    ap.add_argument("--out", default="data/statewide/grid_cells.csv")
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.wgen_dir).iterdir()):
        name = p.name
        if not name.startswith("data_"):
            continue
        _, lat, lon = name.split("_")
        # key convention = sacsma.io forcing keys / _norm_key: 5-decimal floats
        rows.append({"key": f"{round(float(lat), 5)}_{round(float(lon), 5)}",
                     "lat": float(lat), "lon": float(lon)})
    df = pd.DataFrame(rows).sort_values(["lat", "lon"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} cells, lat {df.lat.min():.5f}..{df.lat.max():.5f}, "
          f"lon {df.lon.min():.5f}..{df.lon.max():.5f}")


if __name__ == "__main__":
    main()
