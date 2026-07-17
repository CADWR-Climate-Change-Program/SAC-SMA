"""Build the canonical REGION 1/16-deg cell list -> data/region/grid_cells.csv.

The auxiliary-data region is the UNION of the four modeling domains' grid
cells — ``15cdec_grid`` (2074 cells, from its hruinfo) plus the CalSim domains
``9unimp``/``11obs``/``12rim`` (their forcing-store keys) — 2480 cells total.
Keys are normalized to the ``sacsma.io`` 5-decimal ``<lat>_<lon>`` convention
(``round(x, 5)``; the calsim stores carry 6-decimal fixed-format keys).
Membership flags ``in_<domain>`` record which domains touch each cell.

Every region auxiliary ingest (GEE ET/SWE obs, GLEAM/FLUXCOM, statics
consolidation, the WGEN forcing master) targets THIS cell list.  Each cell is
asserted present in the local WGEN NonDetrend-Unsplit statewide store — the
daily forcing master source (see dataprep/wgen_forcing.py).

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
DOMAINS = ("15cdec_grid", "9unimp", "11obs", "12rim")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wgen-dir", default=WGEN_DIR)
    ap.add_argument("--out", default="data/region/grid_cells.csv")
    args = ap.parse_args()

    root = Path(".")
    members = {d: domain_cells(root, d) for d in DOMAINS}
    union = sorted(set().union(*members.values()),
                   key=lambda k: tuple(float(x) for x in k.split("_")))

    wgen = Path(args.wgen_dir)
    missing = [k for k in union if not (wgen / f"data_{k}").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} region cells absent from the WGEN "
                         f"store (first: {missing[:3]}) — key drift?")

    rows = []
    for k in union:
        lat, lon = (float(x) for x in k.split("_"))
        rows.append({"key": k, "lat": lat, "lon": lon,
                     **{f"in_{d}": int(k in members[d]) for d in DOMAINS}})
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    counts = ", ".join(f"{d}: {len(members[d])}" for d in DOMAINS)
    print(f"wrote {out}: {len(df)} region cells ({counts}); "
          f"lat {df.lat.min():.5f}..{df.lat.max():.5f}, "
          f"lon {df.lon.min():.5f}..{df.lon.max():.5f}; all in WGEN store")


if __name__ == "__main__":
    main()
