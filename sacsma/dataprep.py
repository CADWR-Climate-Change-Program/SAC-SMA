"""Ingest source data into the organized, Python-native artifacts under ``data/``.

Going forward there are **two ingests** a user runs, each pointing at a
user-supplied source directory and writing into the tracked ``data/`` store::

    # add / replace the domain forcing store  ->  data/forcing/<name>
    python -m sacsma.dataprep forcing --src <meteo_dir> [--name historical_15cdec.nc]

    # add / replace the watershed GeoPackage  ->  data/gis/<name>
    python -m sacsma.dataprep gis --src <shapefile_dir> [--name calsim3.gpkg]

    # add / replace the MATLAB simulated parity target  ->  data/reference/
    python -m sacsma.dataprep reference --src <simflow_dir>

    # add / replace the observed gage FNF calibration target  ->  data/reference/
    python -m sacsma.dataprep gage --src <fnf_dir>

``meteo_dir`` holds ``meteo_<lat>_<lon>`` whitespace files (year month day prcp
tavg); ``shapefile_dir`` holds ``*.shp`` (one GeoPackage layer per shapefile);
``simflow_dir`` holds the **full** ``simflow_sacsma_<CODE>.txt`` MATLAB flow
series (the forward run reproduces these exactly — do NOT use the stale
``*_short.txt`` drop); ``fnf_dir`` holds ``FNF_<CODE>_cfs.txt`` daily CDEC
full-natural-flow (cfs, ``year month day flow_cfs``) for all 15 basins including
BND — converted to mm/day per basin area, negatives→NaN, covers calibration AND
validation (needs the drainage-area table built first).

The per-HRU attribute / GA-parameter tables and the per-basin drainage areas
were built **once** from the MATLAB-era reference materials and are committed
under ``data/``.  Rebuilding them is a legacy operation that needs an explicit
path to that reference tree (it is NOT wired to any in-repo default)::

    python -m sacsma.dataprep tables --reference-root <dir containing sacsma_module/>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .io import (
    FORCING_NAME,
    cfs_to_mmday,
    parse_meteo_filename,
    read_hruinfo,
    read_meteo,
    read_simflow,
)
from .parameters import load_ga_optimum

CDEC15 = [
    "SHA", "BND", "ORO", "YRS", "FOL", "MKM", "NHG", "NML",
    "TLG", "MRC", "MIL", "PNF", "TRM", "SCC", "ISB",
]


# --------------------------------------------------------------------------
# Forcing store (one domain-wide NetCDF, indexed by grid cell)  ->  data/forcing/
# --------------------------------------------------------------------------
def build_forcing(
    meteo_dir: Path | str,
    data_dir: Path | str = "data",
    out_name: str = FORCING_NAME,
    keys: list[str] | None = None,
) -> Path:
    """Ingest a directory of meteo grid-cell files into ONE forcing store.

    Forcing is a property of the meteo *grid cells*, independent of any basin
    delineation: the domain is whatever ``meteo_<lat>_<lon>`` files live in
    ``meteo_dir`` (or the explicit ``keys`` subset).  Writes
    ``<data_dir>/forcing/<out_name>`` with dims ``(key, time)`` and coords
    ``key`` (``lat_lon``), ``lat``, ``lon``.
    """
    import xarray as xr

    mdir = Path(meteo_dir)
    if keys is not None:
        files = [mdir / f"meteo_{k}" for k in keys]
    else:
        files = sorted(mdir.glob("meteo_*"))
    if not files:
        raise FileNotFoundError(f"No meteo_* files found in {mdir}")

    cell_keys: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    dates = None
    prcp = None
    tavg = None
    for i, path in enumerate(files):
        key, lat, lon = parse_meteo_filename(path.name)
        meteo = read_meteo(path)
        if dates is None:
            dates = pd.DatetimeIndex(meteo["date"])
            prcp = np.empty((len(files), len(dates)), dtype=np.float32)
            tavg = np.empty((len(files), len(dates)), dtype=np.float32)
        elif len(meteo) != len(dates):
            raise ValueError(f"cell {key}: {len(meteo)} days != {len(dates)}")
        cell_keys.append(key)
        lats.append(lat)
        lons.append(lon)
        prcp[i] = meteo["prcp"].to_numpy()
        tavg[i] = meteo["tavg"].to_numpy()
        if i % 500 == 0:
            print(f"  forcing: ingested {i + 1}/{len(files)} cells", flush=True)

    ds = xr.Dataset(
        data_vars={
            "prcp": (("key", "time"), prcp),
            "tavg": (("key", "time"), tavg),
        },
        coords={
            "key": np.array(cell_keys, dtype=object),
            "time": dates.values,
            "lat": ("key", np.array(lats)),
            "lon": ("key", np.array(lons)),
        },
        attrs={"units_prcp": "mm/day", "units_tavg": "degC", "n_cells": len(files)},
    )
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ("prcp", "tavg")}
    out_dir = Path(data_dir) / "forcing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out_name
    ds.to_netcdf(out, encoding=enc)
    print(f"forcing: {len(files)} cells x {len(dates)} days -> {out}")
    return out


# --------------------------------------------------------------------------
# GIS -> single GeoPackage  ->  data/gis/
# --------------------------------------------------------------------------
def build_gis_gpkg(
    shp_dir: Path | str,
    data_dir: Path | str = "data",
    out_name: str = "calsim3.gpkg",
) -> Path:
    """Merge every ``*.shp`` in ``shp_dir`` into one GeoPackage (one layer each)."""
    import geopandas as gpd

    shps = sorted(Path(shp_dir).glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No *.shp files found in {shp_dir}")
    out = Path(data_dir) / "gis" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()  # rewrite cleanly
    for shp in shps:
        gdf = gpd.read_file(shp)
        gdf.to_file(out, layer=shp.stem, driver="GPKG")
        print(f"gis: layer {shp.stem} ({len(gdf)} features)")
    print(f"gis: {len(shps)} layers -> {out}")
    return out


# --------------------------------------------------------------------------
# One-time reference tables (legacy): MATLAB reference -> data/{hru,params,reference}/
# --------------------------------------------------------------------------
def _reference_src(reference_root: Path) -> dict[str, Path]:
    """Paths within a user-supplied MATLAB reference tree (contains sacsma_module/)."""
    return {
        "hruinfo_dir": reference_root / "sacsma_module" / "hruinfo_15CDEC",
        "ga_file": reference_root / "sacsma_module" / "sacramento_ga_15cdec_pool.txt",
    }


def build_hru_table(reference_root: Path, data_dir: Path) -> pd.DataFrame:
    """Per-HRU attribute table with CDEC basin code (from per-basin HRUinfo)."""
    src = _reference_src(reference_root)
    frames = []
    for code in CDEC15:
        df = read_hruinfo(src["hruinfo_dir"] / f"HRUinfo_{code}.txt")
        df.insert(0, "basin", code)
        frames.append(df)
    table = pd.concat(frames, ignore_index=True)
    out = data_dir / "hru" / "hruinfo_15cdec.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"hru: {len(table)} HRUs -> {out}")
    return table


def _read_fnf_cfs(path: Path) -> pd.DataFrame:
    """Read one ``FNF_<CODE>_cfs.txt`` -> DataFrame[year, month, day, flow_cfs].

    Handles both the cleaned ``FNF_cfs_nan/`` variant (tab-separated, 5 cols with
    a trailing date string and an EMPTY field for missing days) and the raw
    ``FNF_cfs/`` variant (whitespace, 4 cols).
    """
    cols = ["year", "month", "day", "flow_cfs"]
    with open(path) as fh:
        first = fh.readline()
    if "\t" in first:  # cleaned variant (5 tab-separated cols)
        d = pd.read_csv(path, sep="\t", header=None,
                        names=cols + ["date"], usecols=cols)
    else:  # raw variant (4 whitespace cols)
        d = pd.read_csv(path, sep=r"\s+", engine="python", header=None, names=cols)
    return d


def build_gage(fnf_dir: Path | str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Build the observed gage record from per-basin daily CDEC FNF files (cfs).

    ``fnf_dir`` is the CDEC FNF tree; per basin the **cleaned**
    ``FNF_cfs_nan/FNF_<CODE>_cfs.txt`` is preferred (erroneous-peak QC applied),
    falling back to the raw ``FNF_cfs/FNF_<CODE>_cfs.txt`` (e.g. BND, which has no
    cleaned file) or ``FNF_<CODE>_cfs.txt`` directly under ``fnf_dir``.  Following
    the source QC (``cdec_fnf_clean.py``), any **negative** value (the
    ``-9999999`` sentinel and small reconstruction negatives) becomes NaN.  Flows
    are converted to mm/day via each basin's drainage area
    (``data/reference/basin_area_15cdec.parquet``) and span the full record
    (calibration **and** validation).  Writes
    ``data/reference/gage_15cdec.parquet`` [date, basin, flow].
    """
    fnf_dir = Path(fnf_dir)
    areas = (pd.read_parquet(Path(data_dir) / "reference" / "basin_area_15cdec.parquet")
             .set_index("basin")["area_mi2"].to_dict())
    frames = []
    for code in CDEC15:
        candidates = [fnf_dir / "FNF_cfs_nan" / f"FNF_{code}_cfs.txt",
                      fnf_dir / "FNF_cfs" / f"FNF_{code}_cfs.txt",
                      fnf_dir / f"FNF_{code}_cfs.txt"]
        f = next((c for c in candidates if c.exists()), None)
        if f is None:
            print(f"  (skip {code}: no FNF_{code}_cfs.txt found)")
            continue
        if code not in areas:
            raise KeyError(f"no drainage area for {code}; run `dataprep tables` first")
        d = _read_fnf_cfs(f)
        cfs = d["flow_cfs"].astype(float).where(d["flow_cfs"] >= 0)  # negatives -> NaN
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(d[["year", "month", "day"]]),
            "basin": code,
            "flow": cfs_to_mmday(cfs, areas[code]),
        }))
    if not frames:
        raise FileNotFoundError(f"No FNF_<CODE>_cfs.txt files found under {fnf_dir}")
    obs = pd.concat(frames, ignore_index=True).sort_values(["basin", "date"]).reset_index(drop=True)
    n_missing = int(obs["flow"].isna().sum())
    out = Path(data_dir) / "reference" / "gage_15cdec.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(out, index=False)
    print(f"gage: {len(obs)} rows across {obs['basin'].nunique()} basins "
          f"({n_missing} missing days NaN) "
          f"[{obs['date'].min().date()}..{obs['date'].max().date()}] -> {out}")
    return obs


def build_basin_area(area_csv: Path | str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Ingest per-basin drainage areas (mi^2) -> data/reference/basin_area_15cdec.parquet."""
    df = pd.read_csv(area_csv, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    out = Path(data_dir) / "reference" / "basin_area_15cdec.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"basin area: {len(df)} basins -> {out}")
    return df


def build_params(reference_root: Path, data_dir: Path) -> pd.DataFrame:
    """Per-HRU GA-optimum parameter table (31 params keyed by lat_lon)."""
    src = _reference_src(reference_root)
    params = load_ga_optimum(src["ga_file"]).reset_index()  # 'key' becomes a column
    out = data_dir / "params" / "ga_optimum_15cdec.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    params.to_parquet(out, index=False)
    print(f"params: {len(params)} HRUs -> {out}")
    return params


def build_reference(simflow_dir: Path | str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Reference MATLAB simulated gauge flow, long format [date, basin, flow].

    Reads the **full** per-basin series ``simflow_sacsma_<CODE>.txt`` (the whole
    1915–2018 simulation period) from ``simflow_dir``.  NOTE: the ``*_short.txt``
    files in the old ``climate_historical/`` drop are a DIFFERENT (stale) run and
    must not be used — the full files are the ones the forward run reproduces
    exactly.
    """
    simflow_dir = Path(simflow_dir)
    frames = []
    for code in CDEC15:
        path = simflow_dir / f"simflow_sacsma_{code}.txt"
        if not path.exists():
            print(f"  (skip reference {code}: {path.name} not found)")
            continue
        df = read_simflow(path)
        df.insert(1, "basin", code)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No simflow_sacsma_<CODE>.txt files found in {simflow_dir}")
    ref = pd.concat(frames, ignore_index=True)
    out = Path(data_dir) / "reference" / "simflow_15cdec.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(out, index=False)
    print(f"reference: {len(ref)} rows across {ref['basin'].nunique()} basins -> {out}")
    return ref


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sacsma.dataprep", description="Ingest source data into the data/ store"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("forcing", help="ingest a meteo dir -> data/forcing/<name>")
    pf.add_argument("--src", required=True, help="directory of meteo_<lat>_<lon> files")
    pf.add_argument("--name", default=FORCING_NAME, help="output forcing store filename")
    pf.add_argument("--data-dir", default="data", help="output data directory")

    pg = sub.add_parser("gis", help="ingest a shapefile dir -> data/gis/<name>")
    pg.add_argument("--src", required=True, help="directory of *.shp watershed shapes")
    pg.add_argument("--name", default="calsim3.gpkg", help="output GeoPackage filename")
    pg.add_argument("--data-dir", default="data", help="output data directory")

    pr = sub.add_parser(
        "reference",
        help="ingest MATLAB simflow_sacsma_<CODE>.txt -> data/reference/ (parity target)",
    )
    pr.add_argument("--src", required=True, help="directory of full simflow_sacsma_<CODE>.txt files")
    pr.add_argument("--data-dir", default="data", help="output data directory")

    pgg = sub.add_parser(
        "gage",
        help="ingest daily CDEC FNF (cfs) -> data/reference/gage_15cdec.parquet (calibration target)",
    )
    pgg.add_argument("--src", required=True, help="dir with FNF_<CODE>_cfs.txt (year month day flow_cfs)")
    pgg.add_argument("--data-dir", default="data", help="output data directory")

    pt = sub.add_parser(
        "tables",
        help="(one-time) build hru/params/area parquet from a MATLAB reference tree",
    )
    pt.add_argument(
        "--reference-root", required=True,
        help="directory containing sacsma_module/ (the MATLAB reference materials)",
    )
    pt.add_argument("--data-dir", default="data", help="output data directory")

    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)

    if args.command == "forcing":
        build_forcing(args.src, data_dir, out_name=args.name)
    elif args.command == "gis":
        build_gis_gpkg(args.src, data_dir, out_name=args.name)
    elif args.command == "reference":
        build_reference(args.src, data_dir)
    elif args.command == "gage":
        build_gage(args.src, data_dir)
    elif args.command == "tables":
        root = Path(args.reference_root)
        build_hru_table(root, data_dir)
        build_params(root, data_dir)
        area_csv = root / "basin_area_CDEC.csv"
        if area_csv.exists():
            build_basin_area(area_csv, data_dir)
        else:
            print(f"  (skip basin area: {area_csv} not found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
