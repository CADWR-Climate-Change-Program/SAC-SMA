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
    read_table,
    write_table,
)
from .parameters import latlon_key, load_ga_optimum

CDEC15 = [
    "SHA", "BND", "ORO", "YRS", "FOL", "MKM", "NHG", "NML",
    "TLG", "MRC", "MIL", "PNF", "TRM", "SCC", "ISB",
]

#: CalLite calibration sets (32 watersheds): domain -> GA-file tag.  Each is a
#: separate **per-watershed** calibration (monthly objective; Wi & Steinschneider
#: MEMO Rim12 / Observed11 / Unimpaired9), distinct from the pooled 15-CDEC set.
CALSETS = {"9unimp": "9uni", "11obs": "11obs", "12rim": "12rim"}


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
        _raw, lat, lon = parse_meteo_filename(path.name)
        # canonicalize the key to 6-decimal lat_lon so it joins the HRU/GA tables
        # regardless of how many decimals the meteo filename carries.
        key = latlon_key(lat, lon)
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
#: layer name the catchment reader expects inside the GeoPackage.
CALSIM_LAYER = "CalSim3_And_GooseLake"


def build_gis_gpkg(
    src: Path | str,
    data_dir: Path | str = "data",
    out_name: str = "calsim3.gpkg",
) -> Path:
    """Build ``data/gis/<out_name>`` from CalSim watershed GIS.

    ``src`` may be a **directory of shapefiles** (each becomes a layer, legacy) or a
    single **``.geojson``/``.json``** (e.g. the calsim-view ``watersheds.geojson``),
    which is normalised to the ``CalSim3_And_GooseLake`` layer schema the catchment
    reader expects (``Connect_No, Type, Remarks, CT_Name, SQ_MI``).  The geojson's
    clean ``Inflow_arc`` (``I_<node>``) drives ``Connect_No`` so node names match the
    CalSim inflow arcs exactly (119/120 rim nodes vs the older shapefile's 115).
    """
    import geopandas as gpd

    src = Path(src)
    out = Path(data_dir) / "gis" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()  # rewrite cleanly

    if src.is_file() and src.suffix.lower() in (".geojson", ".json"):
        g = gpd.read_file(src)
        # prefer the clean Inflow_arc (I_<node>) for the node code; fall back to Connect_No
        if "Inflow_arc" in g:
            iarc = g["Inflow_arc"].astype(str)
            node = iarc.str.replace(r"^I_", "", regex=True).str.strip()
            node = node.where(iarc.str.startswith("I_"), g.get("Connect_No"))
        else:
            node = g.get("Connect_No")
        norm = gpd.GeoDataFrame(
            {
                "Connect_No": node,
                "Type": g["Type"] if "Type" in g else None,
                "Remarks": g["Remarks"] if "Remarks" in g else None,
                "CT_Name": g["ClosureTerm_Name"] if "ClosureTerm_Name" in g else g.get("CT_Name"),
                "SQ_MI": (g["Square_Mile"] if "Square_Mile" in g else g.get("SQ_MI")).astype(float),
                "geometry": g.geometry.to_numpy(),
            },
            crs=g.crs,
        )
        norm.to_file(out, layer=CALSIM_LAYER, driver="GPKG")
        print(f"gis: {src.name} -> layer {CALSIM_LAYER} ({len(norm)} features; "
              f"{(norm['Type']=='Rim').sum()} Rim) -> {out}")
        return out

    shps = sorted(src.glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No *.shp or *.geojson found at {src}")
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
    out = data_dir / "hru" / "hruinfo_15cdec.csv"
    write_table(table, out)
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
    (``data/reference/basin_area_15cdec.csv``) and span the full record
    (calibration **and** validation).  Writes
    ``data/reference/gage_15cdec.csv`` [date, basin, flow].
    """
    fnf_dir = Path(fnf_dir)
    areas = (read_table(Path(data_dir) / "reference" / "basin_area_15cdec.csv")
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
    out = Path(data_dir) / "reference" / "gage_15cdec.csv"
    write_table(obs, out)
    print(f"gage: {len(obs)} rows across {obs['basin'].nunique()} basins "
          f"({n_missing} missing days NaN) "
          f"[{obs['date'].min().date()}..{obs['date'].max().date()}] -> {out}")
    return obs


def build_basin_area(area_csv: Path | str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Ingest per-basin drainage areas (mi^2) -> data/reference/basin_area_15cdec.csv."""
    df = pd.read_csv(area_csv, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    out = Path(data_dir) / "reference" / "basin_area_15cdec.csv"
    write_table(df, out)
    print(f"basin area: {len(df)} basins -> {out}")
    return df


#: drainage-area unit columns -> multiplier to mi^2.
_AREA_TO_MI2 = {"area_mi2": 1.0, "acre": 1.0 / 640.0, "area_km2": 1.0 / 2.589988,
                "area_ft2": 1.0 / 27_878_400.0}


def build_basin_area_domain(src: Path | str, domain: str,
                            data_dir: Path | str = "data") -> pd.DataFrame:
    """Ingest authoritative per-basin drainage areas for any ``domain`` ->
    ``data/reference/basin_area_<domain>.csv`` (the source of truth used by
    :func:`sacsma.calsim.basin_areas`).

    ``src`` is a ``.csv`` or ``.xlsx`` (needs openpyxl) with a ``basin`` column plus an
    area column whose name gives the unit: ``area_mi2`` | ``acre`` | ``area_km2`` |
    ``area_ft2`` (e.g. the ``tmp/basin_area_{CDEC.csv,11ObsInflows.xlsx,9Unimpaired.xlsx}``).
    """
    src = Path(src)
    df = (pd.read_excel(src) if src.suffix.lower() in (".xlsx", ".xls")
          else pd.read_csv(src, encoding="utf-8-sig"))
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    bcol = lower.get("basin") or df.columns[0]
    acol = next((lower[u] for u in _AREA_TO_MI2 if u in lower), None)
    if acol is None:
        raise ValueError(f"no area column in {src} (need one of {list(_AREA_TO_MI2)}); have {list(df.columns)}")
    out = pd.DataFrame({"basin": df[bcol].astype(str).str.strip(),
                        "area_mi2": df[acol].astype(float) * _AREA_TO_MI2[acol.lower()]})
    out = out.dropna(subset=["area_mi2"]).reset_index(drop=True)
    dst = Path(data_dir) / "reference" / f"basin_area_{domain}.csv"
    write_table(out, dst)
    print(f"basin area [{domain}]: {len(out)} basins from {acol} -> {dst}")
    return out


def build_params(reference_root: Path, data_dir: Path) -> pd.DataFrame:
    """Per-HRU GA-optimum parameter table (31 params keyed by lat_lon)."""
    src = _reference_src(reference_root)
    params = load_ga_optimum(src["ga_file"]).reset_index()  # 'key' becomes a column
    out = data_dir / "params" / "ga_optimum_15cdec.csv"
    write_table(params, out)
    print(f"params: {len(params)} HRUs -> {out}")
    return params


def build_vic(vic_dir: Path | str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Ingest VIC routed historical monthly flows -> data/reference/vic_routed_monthly.csv.

    ``vic_dir`` holds ``CS3_<name>_qmo.csv`` (no header: ``date,flow_taf``,
    month-end, **TAF/month**) — the VIC-routed historical inflow at each CalSim
    location (e.g. ``CS3_I_SHSTA_qmo.csv`` -> ``I_SHSTA``, ``CS3_8RI_OROVI_qmo``
    -> ``8RI_OROVI``).  Output is long ``[date, vic_name, flow_taf]``.
    """
    vic_dir = Path(vic_dir)
    files = sorted(vic_dir.glob("CS3_*_qmo.csv"))
    if not files:
        raise FileNotFoundError(f"No CS3_*_qmo.csv files found in {vic_dir}")
    frames = []
    for f in files:
        name = f.stem[len("CS3_"):-len("_qmo")] if f.stem.startswith("CS3_") else f.stem
        d = pd.read_csv(f, header=None, names=["date", "flow_taf"])
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(d["date"]),
            "vic_name": name,
            "flow_taf": pd.to_numeric(d["flow_taf"], errors="coerce"),
        }))
    vic = pd.concat(frames, ignore_index=True).sort_values(["vic_name", "date"]).reset_index(drop=True)
    out = Path(data_dir) / "reference" / "vic_routed_monthly.csv"
    write_table(vic, out)
    print(f"vic: {len(vic)} rows across {vic['vic_name'].nunique()} locations "
          f"[{vic['date'].min().date()}..{vic['date'].max().date()}] -> {out}")
    return vic


def _read_dss_monthly(dss_path: Path, cpart: str, *, key: str = "arc") -> pd.DataFrame:
    """Read monthly records of one ``cpart`` (C-part) for every B-part from a SV DSS.

    Returns long ``[date, <key>, flow_taf]`` (month-end), one merged series per
    B-part.  Self-contained: lazily imports ``pydsstools`` (needs ``pydsstools<3``,
    e.g. the ``csstochastic`` env) and creates a temporary Windows directory junction
    when the DSS path is long (the OneDrive data dir exceeds the Fortran 256-char
    limit).  D-part decade blocks are merged onto one month-end series per B-part.
    """
    import os
    import subprocess

    import numpy as np
    from pydsstools.heclib.dss import HecDss

    cpart = cpart.upper()
    link = Path("_dss_link_sacsma")
    use_junction = len(str(dss_path)) > 200
    if use_junction:
        if os.path.lexists(str(link)):
            subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(dss_path.parent)],
                       check=True, capture_output=True)
        work = str(link / dss_path.name)
    else:
        work = str(dss_path)
    try:
        frames = []
        with HecDss.Open(work, version=6, catalog_flag=True) as dss:
            paths = dss.getPathnameList("/*/*/*/*/1MON/*")
            buckets: dict[str, list[str]] = {}
            for p in paths:
                parts = p.strip("/").split("/")
                if len(parts) == 6 and parts[2].upper() == cpart:
                    buckets.setdefault(parts[1].upper(), []).append(p)
            for bpart, plist in buckets.items():
                master: dict = {}
                for p in sorted(plist, key=lambda x: x.strip("/").split("/")[3]):
                    ts = dss.read_ts(p, trim_missing=True)
                    vals = np.where(np.asarray(ts.values, dtype=float) <= -900, np.nan,
                                    np.asarray(ts.values, dtype=float))
                    idx = (pd.to_datetime(ts.pytimes).to_period("M") - 1).to_timestamp("M")
                    master.update(pd.Series(vals, index=idx).dropna().to_dict())
                if master:
                    s = pd.Series(master).sort_index()
                    frames.append(pd.DataFrame({"date": s.index, key: bpart, "flow_taf": s.values}))
    finally:
        if use_junction and os.path.lexists(str(link)):
            subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True)
    return pd.concat(frames, ignore_index=True)


def build_calsim3(src: Path | str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Ingest CalSim3 historical monthly inflows -> data/reference/calsim3_inflow_monthly.csv.

    ``src`` is either the CalSim SV **DSS** (``__calsim_sv_default__.dss``; needs
    ``pydsstools<3``) or a pre-extracted **CSV** with columns ``date,arc,flow_taf``.
    Each ``arc`` is the CalSim inflow node B-part (e.g. ``I_SHSTA``), C-part
    ``INFLOW``, **TAF/month**.  Output is long ``[date, arc, flow_taf]``.
    """
    src = Path(src)
    if src.suffix.lower() == ".dss":
        c3 = _read_dss_monthly(src, "INFLOW", key="arc")
    else:
        c3 = pd.read_csv(src)
        c3["date"] = pd.to_datetime(c3["date"])
    c3 = c3.sort_values(["arc", "date"]).reset_index(drop=True)
    out = Path(data_dir) / "reference" / "calsim3_inflow_monthly.csv"
    write_table(c3, out)
    print(f"calsim3: {len(c3)} rows across {c3['arc'].nunique()} arcs "
          f"[{c3['date'].min().date()}..{c3['date'].max().date()}] -> {out}")
    return c3


#: the 11 CalSim FLOW-UNIMPAIRED rim systems (B-part ``UNIMP_<SYS>``) -> short name.
UNIMP_SYSTEMS = {
    "UNIMP_SHAS": "SHAS", "UNIMP_SRBB": "SRBB", "UNIMP_OROV": "OROV",
    "UNIMP_YUBA": "YUBA", "UNIMP_FOLS": "FOLS", "UNIMP_ST": "ST",
    "UNIMP_TU": "TU", "UNIMP_ME": "ME", "UNIMP_SJ": "SJ",
    "UNIMP_TRIN": "TRIN", "UNIMP_WH": "WH",
}


def build_unimpaired(src: Path | str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Ingest CalSim ``FLOW-UNIMPAIRED`` for the 11 rim systems -> data/reference/.

    ``src`` is the CalSim SV **DSS** (``__calsim_sv_default__.dss``; needs
    ``pydsstools<3``) or a pre-extracted **CSV** ``[date, system, flow_taf]``.  Only
    the 11 base rim systems (``UNIMP_<SYS>``) are kept; the ``_UHH`` variants are
    dropped.  Output is long ``[date, system, flow_taf]`` (monthly TAF) ->
    ``data/reference/calsim_unimpaired_monthly.csv``.
    """
    src = Path(src)
    if src.suffix.lower() == ".dss":
        u = _read_dss_monthly(src, "FLOW-UNIMPAIRED", key="bpart")
        u = u[u["bpart"].isin(UNIMP_SYSTEMS)].copy()
        u["system"] = u["bpart"].map(UNIMP_SYSTEMS)
        u = u[["date", "system", "flow_taf"]]
    else:
        u = pd.read_csv(src)
        u["date"] = pd.to_datetime(u["date"])
    u = u.sort_values(["system", "date"]).reset_index(drop=True)
    out = Path(data_dir) / "reference" / "calsim_unimpaired_monthly.csv"
    write_table(u, out)
    print(f"unimpaired: {len(u)} rows across {u['system'].nunique()} systems "
          f"[{u['date'].min().date()}..{u['date'].max().date()}] -> {out}")
    return u


def build_rim_anchor(src: Path | str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Ingest the RimInflowAnchor crosswalk -> data/reference/calsim_rim_anchor.csv.

    ``src`` is ``RimInflowAnchor.xlsx`` (needs ``openpyxl``; sheet ``RimInflowAnchor``
    with columns ``Calsim_ID, CDEC_Watershed, VIC_Qmap_Anchor``) **or** a pre-extracted
    CSV with the same three columns renamed ``arc, system, unimp_anchor``.

    This is the **authoritative** map of every CalSim inflow ``arc`` to its rim
    ``system`` (one of the 10 ``FOLS OROV SRBB YUBA ST TU SHAS TRIN ME SJ``) and the
    ``UNIMP_<sys>`` anchor.  Arcs with a blank ``system`` are non-rim westside/valley
    catchments (e.g. ``I_PTH070`` = Putah/Berryessa) resolved geographically instead.
    Output ``[arc, system, unimp_anchor]`` (203 arcs; 120 assigned to a rim system).
    """
    src = Path(src)
    if src.suffix.lower() in (".xlsx", ".xls"):
        d = pd.read_excel(src, sheet_name="RimInflowAnchor")
        d = d[["Calsim_ID", "CDEC_Watershed", "VIC_Qmap_Anchor"]]
        d.columns = ["arc", "system", "unimp_anchor"]
    else:
        d = pd.read_csv(src)
    d = d.dropna(subset=["arc"]).drop_duplicates("arc").sort_values("arc").reset_index(drop=True)
    out = Path(data_dir) / "reference" / "calsim_rim_anchor.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"rim_anchor: {len(d)} arcs ({d['system'].notna().sum()} in a rim system) -> {out}")
    return d


def _parse_hist_fnf(path: Path) -> pd.DataFrame:
    """Parse a full-period historical FNF file -> long [date, val] (month-end).

    Handles the wide ``WaterYear,10,11,12,1..9`` layout (11-obs ``<CODE>_hist.csv``)
    and the long ``WaterYear,Month,Flow_TAF`` layout (9-unimp ``<Name>Historical.csv``).
    """
    df = pd.read_csv(path, na_values=["NA", "NaN", ""])
    cols = [str(c) for c in df.columns]
    if "Month" in cols:  # long layout
        valcol = "Flow_TAF" if "Flow_TAF" in cols else cols[-1]
        wy = df["WaterYear"].astype(int)
        mon = df["Month"].astype(int)
        val = pd.to_numeric(df[valcol], errors="coerce")
    else:  # wide layout: WaterYear + 12 month columns
        long = df.melt(id_vars="WaterYear", var_name="mon", value_name="val")
        wy = long["WaterYear"].astype(int)
        mon = long["mon"].astype(int)
        val = pd.to_numeric(long["val"], errors="coerce")
    yr = np.where(mon.to_numpy() >= 10, wy.to_numpy() - 1, wy.to_numpy())
    date = pd.to_datetime(dict(year=yr, month=mon, day=1)) + pd.offsets.MonthEnd(0)
    return pd.DataFrame({"date": date.to_numpy(), "val": val.to_numpy()}).dropna().reset_index(drop=True)


def _hist_fnf_file(hist_dir: Path, domain: str, code: str) -> Path | None:
    """Locate a watershed's full-period FNF file within ``hist_dir``."""
    if domain == "9unimp":
        cands = [hist_dir / f"{code}Historical.csv"]
    else:  # 11obs (SHA's series is the SIS gauge reconstruction)
        cands = [hist_dir / f"{code}_hist.csv", hist_dir / f"{code}_hist_sim.csv"]
        if code == "SHA":
            cands = [hist_dir / "SIS_hist_sim.csv", *cands]
    return next((c for c in cands if c.exists()), None)


def build_calib_fnf(domain: str, hist_dir: Path | str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Ingest full-period monthly FNF (1922-) -> data/reference/fnf_<domain>_monthly.csv.

    Each file's values are converted to **mm/month** with a per-basin ratio derived
    from the overlap with the calibration-log observed FNF (``calib_<domain>_monthly``),
    so the raw unit (TAF / normalized depth / etc.) does not need to be known.  The
    result spans the full record, enabling **validation** outside the calibration
    period.  Columns: [date, basin, obs_mm, cal_start, cal_end].
    """
    from .io import load_calib_monthly

    hist_dir = Path(hist_dir)
    calib = load_calib_monthly(data_dir, domain=domain)
    frames = []
    for code, g in calib.groupby("basin"):
        f = _hist_fnf_file(hist_dir, domain, code)
        if f is None:
            print(f"  (skip {code}: no historical FNF file in {hist_dir})")
            continue
        h = _parse_hist_fnf(f)
        m = h.merge(g[["date", "obs_mm"]], on="date")
        m = m[m["val"] > 0]
        if m.empty:
            print(f"  (skip {code}: no overlap with calibration record)")
            continue
        # robust per-basin ratio from the calibration overlap (median ignores any
        # corrupt source values), then a physical sanity filter on the result:
        # FNF can't be negative and no CA monthly depth approaches 3000 mm.
        r = (m["obs_mm"] / m["val"])
        ratio, spread = float(r.median()), float((r.std() / r.median()) if r.median() else np.nan)
        h["obs_mm"] = h["val"] * ratio
        bad = (h["obs_mm"] < 0) | (h["obs_mm"] > 3000)
        if bad.any():
            print(f"    ({code}: {int(bad.sum())} implausible month(s) -> NaN)")
        h.loc[bad, "obs_mm"] = np.nan
        h["basin"] = code
        h["cal_start"], h["cal_end"] = g["cal_start"].iloc[0], g["cal_end"].iloc[0]
        frames.append(h[["date", "basin", "obs_mm", "cal_start", "cal_end"]])
        flag = "  <- ratio not constant!" if spread > 0.02 else ""
        print(f"  {code}: {len(h)} months [{h['date'].min().date()}..{h['date'].max().date()}] "
              f"ratio={ratio:.3f} cv={spread:.4f}{flag}")
    out_df = pd.concat(frames, ignore_index=True).sort_values(["basin", "date"]).reset_index(drop=True)
    out = Path(data_dir) / "reference" / f"fnf_{domain}_monthly.csv"
    write_table(out_df, out)
    print(f"fnf: {len(out_df)} rows across {out_df['basin'].nunique()} watersheds -> {out}")
    return out_df


#: 12-Rim reservoir-inflow B-parts in a CalSim SV DSS *spreadsheet* export (basin code ->
#: inflow node).  Validated against the calibration-log obs over the 1951-2003 overlap
#: (monthly corr 0.86-1.00); Whiskeytown (an import-fed reservoir) is the weakest at 0.86,
#: every other basin is >0.96.
RIM12_DSS_INFLOW = {
    "SHAST": "I_SHSTA", "OROVI": "I_OROVL", "FOL_I": "I_FOLSM", "TRINI": "I_TRNTY",
    "WKYTN": "I_WKYTN", "N_MEL": "I_MELON", "N_HOG": "I_NHGAN", "PRD_C": "I_MOKELUMNE",
    "MILLE": "I_MLRTN", "SMART": "I_YUBA", "DPR_I": "I_PEDRO", "LK_MC": "I_MCLRE",
}


def _read_dss_xlsx_inflow(xlsx_path: Path | str,
                          sheet: str = "DSS_DATA (original)") -> dict[str, pd.Series]:
    """Decode a CalSim SV DSS *spreadsheet export* -> ``{bpart: Series(date -> monthly volume)}``.

    The export lays the DSS pathname parts in labelled rows (column 0 holds ``A``/``B``/
    ``C``/``Units``/``Type``); each data column is one series and the monthly time axis is
    the column whose values parse as month-end dates.  ``FLOW-INFLOW`` series in **CFS**
    (stored period-average) are converted to a monthly **volume** proxy (cfs x days-in-month);
    ``TAF`` series are already monthly volume.  One volume series per ``I_<node>`` inflow B-part.
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
    lab = raw.iloc[:, 0].astype("string")

    def row_for(tag: str) -> int:
        hit = raw.index[lab.str.fullmatch(tag, case=False).fillna(False)]
        if len(hit) == 0:
            raise ValueError(f"{xlsx_path} [{sheet}]: no '{tag}' label row in column 0")
        return int(hit[0])

    rb, rc, ru, rt = row_for("B"), row_for("C"), row_for("Units"), row_for("Type")
    bpart = raw.iloc[rb].apply(lambda v: str(v) if pd.notna(v) else "")
    cpart = raw.iloc[rc].apply(lambda v: str(v).upper() if pd.notna(v) else "")
    units = raw.iloc[ru].apply(lambda v: str(v).upper() if pd.notna(v) else "")
    data = raw.iloc[rt + 1:]

    # date axis = the column whose parsed timestamps span the widest range.  (Selecting by
    # COUNT fails: the sequential row-index column of plain ints parses as epoch-nanoseconds,
    # giving 1008 bogus ~1970 "dates"; the real monthly axis spans decades, so span wins.)
    def _date_span_days(c: int) -> float:
        d = pd.to_datetime(data.iloc[:, c], errors="coerce")
        return (d.max() - d.min()).days if d.notna().sum() >= 24 else -1.0

    best = max(range(min(16, raw.shape[1])), key=_date_span_days)
    dates = pd.to_datetime(data.iloc[:, best], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    dim = dates.dt.days_in_month.to_numpy(dtype=float)
    out: dict[str, pd.Series] = {}
    for c in range(raw.shape[1]):
        if bpart[c].startswith("I_") and cpart[c] == "FLOW-INFLOW":
            v = pd.to_numeric(data.iloc[:, c], errors="coerce").to_numpy(dtype=float)
            vol = v * dim if units[c] == "CFS" else v
            s = pd.Series(vol, index=dates.to_numpy())
            out[bpart[c]] = s[s.index.notna()].dropna()
    return out


def build_calib_fnf_xlsx(domain: str, xlsx_path: Path | str, data_dir: Path | str = "data", *,
                         sheet: str = "DSS_DATA (original)",
                         crosswalk: dict[str, str] | None = None) -> pd.DataFrame:
    """Full-period monthly FNF for a domain from a CalSim SV DSS *spreadsheet* export
    (e.g. ``DSS_DATA2.xlsx`` for ``12rim``) -> ``data/reference/fnf_<domain>_monthly.csv``.

    Each basin's reservoir-inflow series (``crosswalk`` basin -> ``I_<node>`` B-part,
    default :data:`RIM12_DSS_INFLOW`) is converted to **mm/month** with a robust per-basin
    **median ratio** from the overlap with the calibration-log observed FNF
    (``calib_<domain>_monthly``) — exactly like :func:`build_calib_fnf`, so the source unit
    need not be known.  The result spans the full record (~WY1922-), enabling **validation**
    on the pre-calibration years.  The overlap correlation is reported as a mapping sanity
    check (low -> wrong series).  Columns [date, basin, obs_mm, cal_start, cal_end].
    """
    from .io import load_calib_monthly

    cw = crosswalk or RIM12_DSS_INFLOW
    series = _read_dss_xlsx_inflow(xlsx_path, sheet=sheet)
    calib = load_calib_monthly(data_dir, domain=domain)
    frames = []
    for code, g in calib.groupby("basin"):
        bp = cw.get(code)
        s = series.get(bp) if bp else None
        if s is None or s.empty:
            print(f"  (skip {code}: no inflow series {bp} in {Path(xlsx_path).name})")
            continue
        h = pd.DataFrame({"date": s.index, "val": s.to_numpy()})
        m = h.merge(g[["date", "obs_mm"]], on="date")
        m = m[(m["val"] > 0) & (m["obs_mm"] > 0)]
        if m.empty:
            print(f"  (skip {code}: no overlap with calibration record)")
            continue
        a, b = m["val"].to_numpy(), m["obs_mm"].to_numpy()
        ratio = float(np.median(b / a))
        corr = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))  # BLAS-free
        h["obs_mm"] = h["val"] * ratio
        # a monthly rim-reservoir inflow of exactly 0 is non-physical: the DSS source's
        # zero-filled tail (the San Joaquin reservoirs go to 0 after ~2001-07) is missing
        # data, not real flow -> drop it (NaN) so it neither plots flat nor scores.
        bad = (h["val"] <= 0) | (h["obs_mm"] < 0) | (h["obs_mm"] > 3000)
        h.loc[bad, "obs_mm"] = np.nan
        h["basin"] = code
        h["cal_start"], h["cal_end"] = g["cal_start"].iloc[0], g["cal_end"].iloc[0]
        frames.append(h[["date", "basin", "obs_mm", "cal_start", "cal_end"]])
        flag = "  <- low corr, check mapping!" if corr < 0.8 else ""
        print(f"  {code} <- {bp}: {len(h)} months "
              f"[{h['date'].min().date()}..{h['date'].max().date()}] r={corr:.3f}{flag}")
    out_df = pd.concat(frames, ignore_index=True).sort_values(["basin", "date"]).reset_index(drop=True)
    out = Path(data_dir) / "reference" / f"fnf_{domain}_monthly.csv"
    write_table(out_df, out)
    print(f"fnf[xlsx]: {len(out_df)} rows across {out_df['basin'].nunique()} watersheds -> {out}")
    return out_df


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
    out = Path(data_dir) / "reference" / "simflow_15cdec.csv"
    write_table(ref, out)
    print(f"reference: {len(ref)} rows across {ref['basin'].nunique()} basins -> {out}")
    return ref


# --------------------------------------------------------------------------
# CalLite calibration sets (9unimp / 11obs / 12rim) — per-watershed calibration
# --------------------------------------------------------------------------
def _parse_calib_result(path: Path):
    """Parse a calibration log's monthly CALIBRATION RESULT table + cal period.

    Returns ``(cal_start, cal_end, DataFrame[date, sim_mm, obs_mm])`` where the
    table is the monthly ``YEAR MON SIM FLOW(mm) OBS FLOW(mm)`` calibration record
    (OBS = the observed monthly full-natural-flow target, mm/month).
    """
    import re

    lines = Path(path).read_text(errors="replace").splitlines()
    cal_start = cal_end = None
    rstart = None
    for i, ln in enumerate(lines):
        if "CALIBRATION PERIOD" in ln:
            m = re.search(r"(\d{4})/(\d+)/\d+\s*-\s*(\d{4})/(\d+)/\d+", ln)
            if m:
                cal_start = f"{m.group(1)}-{int(m.group(2)):02d}"
                cal_end = f"{m.group(3)}-{int(m.group(4)):02d}"
        if "CALIBRATION RESULT" in ln:
            rstart = i
    rows = []
    if rstart is not None:
        for ln in lines[rstart + 1:]:
            p = ln.split()
            if len(p) == 4 and p[0].isdigit():
                rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3])))
    df = pd.DataFrame(rows, columns=["year", "mon", "sim_mm", "obs_mm"])
    if not df.empty:
        df["date"] = (pd.to_datetime(dict(year=df["year"], month=df["mon"], day=1))
                      + pd.offsets.MonthEnd(0))
        df = df[["date", "sim_mm", "obs_mm"]]
    return cal_start, cal_end, df



def build_calset(
    domain: str,
    hruinfo_dir: Path | str,
    calib_dir: Path | str,
    simflow_dir: Path | str,
    data_dir: Path | str = "data",
    ga_tag: str | None = None,
) -> None:
    """Ingest one CalLite calibration set into the ``<domain>`` data store.

    Watershed codes are discovered from ``Gridinfo_<CODE>_new.txt``.  Builds three
    parallel artifacts (mirroring the 15-CDEC ones):
      * ``hru/hruinfo_<domain>.csv``       from ``Gridinfo_<CODE>_new.txt``
      * ``params/ga_optimum_<domain>.csv`` from ``sacramento_ga_<tag>_<CODE>.txt``
        (tagged with ``basin``; cells shared between watersheds carry per-watershed
        params, so the table is keyed by ``(basin, key)``)
      * ``reference/simflow_<domain>.csv`` from ``simflow_sacsma_<CODE>.txt``

    ``ga_tag`` defaults from :data:`CALSETS` (``9unimp``->``9uni`` etc.).  The
    matching forcing store is built separately with
    ``dataprep forcing --src <meteo dir> --name historical_<domain>.nc``.
    """
    tag = ga_tag or CALSETS.get(domain, domain)
    hdir, cdir, sdir = Path(hruinfo_dir), Path(calib_dir), Path(simflow_dir)
    dd = Path(data_dir)
    codes = sorted(p.name[len("Gridinfo_"):-len("_new.txt")]
                   for p in hdir.glob("Gridinfo_*_new.txt"))
    if not codes:
        raise FileNotFoundError(f"No Gridinfo_*_new.txt files in {hdir}")

    hru_frames, par_frames, ref_frames, cal_frames = [], [], [], []
    for code in codes:
        h = read_hruinfo(hdir / f"Gridinfo_{code}_new.txt")
        h.insert(0, "basin", code)
        hru_frames.append(h)

        cfile = cdir / f"sacramento_ga_{tag}_{code}.txt"
        p = load_ga_optimum(cfile).reset_index()
        p.insert(0, "basin", code)
        par_frames.append(p)

        r = read_simflow(sdir / f"simflow_sacsma_{code}.txt")
        r.insert(1, "basin", code)
        ref_frames.append(r)

        # observed monthly FNF (calibration target) embedded in the calibration log
        cs, ce, cal = _parse_calib_result(cfile)
        if not cal.empty:
            cal.insert(0, "basin", code)
            cal["cal_start"], cal["cal_end"] = cs, ce
            cal_frames.append(cal)

    hru = pd.concat(hru_frames, ignore_index=True)
    par = pd.concat(par_frames, ignore_index=True)
    ref = pd.concat(ref_frames, ignore_index=True)
    outputs = [("hru", f"hruinfo_{domain}", hru),
               ("params", f"ga_optimum_{domain}", par),
               ("reference", f"simflow_{domain}", ref)]
    if cal_frames:
        outputs.append(("reference", f"calib_{domain}_monthly", pd.concat(cal_frames, ignore_index=True)))

    for sub, name, df in outputs:
        out = dd / sub / f"{name}.csv"
        write_table(df, out)
    ncal = sum(len(c) for c in cal_frames)
    print(f"{domain}: {len(hru)} HRU rows ({hru['key'].nunique()} cells), "
          f"{len(par)} param rows, {len(ref)} simflow rows, {ncal} monthly-FNF rows "
          f"across {hru['basin'].nunique()} watersheds ({', '.join(codes)}) -> {dd}")


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

    pg = sub.add_parser("gis", help="ingest a shapefile dir OR a watersheds .geojson -> data/gis/<name>")
    pg.add_argument("--src", required=True,
                    help="directory of *.shp watershed shapes, or a single watersheds .geojson")
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
        help="ingest daily CDEC FNF (cfs) -> data/reference/gage_15cdec.csv (calibration target)",
    )
    pgg.add_argument("--src", required=True, help="dir with FNF_<CODE>_cfs.txt (year month day flow_cfs)")
    pgg.add_argument("--data-dir", default="data", help="output data directory")

    pv = sub.add_parser(
        "vic",
        help="ingest VIC routed monthly CSVs -> data/reference/vic_routed_monthly.csv",
    )
    pv.add_argument("--src", required=True, help="dir with CS3_<name>_qmo.csv (date,flow_taf)")
    pv.add_argument("--data-dir", default="data", help="output data directory")

    pc3 = sub.add_parser(
        "calsim3",
        help="ingest CalSim3 historical inflow DSS/CSV -> data/reference/calsim3_inflow_monthly.csv",
    )
    pc3.add_argument("--src", required=True,
                     help="CalSim SV .dss (needs pydsstools<3) or pre-extracted .csv [date,arc,flow_taf]")
    pc3.add_argument("--data-dir", default="data", help="output data directory")

    pun = sub.add_parser(
        "unimp",
        help="ingest CalSim FLOW-UNIMPAIRED (11 rim systems) -> data/reference/calsim_unimpaired_monthly.csv",
    )
    pun.add_argument("--src", required=True,
                     help="CalSim SV .dss (needs pydsstools<3) or pre-extracted .csv [date,system,flow_taf]")
    pun.add_argument("--data-dir", default="data", help="output data directory")

    pmg = sub.add_parser(
        "merge_gis",
        help="write the merged whole-basin Rim layer (CalSim3_Merged) into calsim3.gpkg",
    )
    pmg.add_argument("--data-dir", default="data", help="data directory (reads + writes calsim3.gpkg)")

    pra = sub.add_parser(
        "rim_anchor",
        help="ingest RimInflowAnchor.xlsx (arc->rim system crosswalk) -> data/reference/calsim_rim_anchor.csv",
    )
    pra.add_argument("--src", required=True,
                     help="RimInflowAnchor.xlsx (needs openpyxl) or pre-extracted .csv [arc,system,unimp_anchor]")
    pra.add_argument("--data-dir", default="data", help="output data directory")

    par = sub.add_parser(
        "area",
        help="ingest authoritative per-basin drainage areas -> data/reference/basin_area_<domain>.csv",
    )
    par.add_argument("--domain", required=True, help="output domain (15cdec/11obs/9unimp/12rim/...)")
    par.add_argument("--src", required=True,
                     help="csv/xlsx with a 'basin' column + an area column named "
                          "area_mi2 | acre | area_km2 | area_ft2 (xlsx needs openpyxl)")
    par.add_argument("--data-dir", default="data", help="output data directory")

    pu = sub.add_parser(
        "calset",
        help="(one-time) build a CalLite calibration set (9unimp/11obs/12rim) data store",
    )
    pu.add_argument("--domain", required=True, choices=sorted(CALSETS),
                    help="calibration set / output domain")
    pu.add_argument("--hruinfo", required=True, help="dir of Gridinfo_<CODE>_new.txt")
    pu.add_argument("--calib", required=True, help="dir of sacramento_ga_<tag>_<CODE>.txt")
    pu.add_argument("--simflow", required=True, help="dir of simflow_sacsma_<CODE>.txt")
    pu.add_argument("--data-dir", default="data", help="output data directory")

    pf = sub.add_parser(
        "fnf",
        help="(one-time) ingest full-period monthly FNF -> data/reference/fnf_<domain>_monthly.csv",
    )
    pf.add_argument("--domain", required=True, choices=sorted(CALSETS), help="calibration set")
    pf.add_argument("--src", required=True, help="dir of historical FNF files (full period)")
    pf.add_argument("--data-dir", default="data", help="output data directory")

    pfx = sub.add_parser(
        "fnf_xlsx",
        help="(one-time) full-period monthly FNF from a CalSim SV DSS spreadsheet export "
             "(e.g. DSS_DATA2.xlsx for 12rim) -> data/reference/fnf_<domain>_monthly.csv",
    )
    pfx.add_argument("--domain", required=True, choices=sorted(CALSETS), help="calibration set")
    pfx.add_argument("--src", required=True, help="DSS spreadsheet export .xlsx (needs openpyxl)")
    pfx.add_argument("--sheet", default="DSS_DATA (original)", help="worksheet name")
    pfx.add_argument("--data-dir", default="data", help="output data directory")

    pt = sub.add_parser(
        "tables",
        help="(one-time) build hru/params/area CSV tables from a MATLAB reference tree",
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
    elif args.command == "vic":
        build_vic(args.src, data_dir)
    elif args.command == "calsim3":
        build_calsim3(args.src, data_dir)
    elif args.command == "unimp":
        build_unimpaired(args.src, data_dir)
    elif args.command == "rim_anchor":
        build_rim_anchor(args.src, data_dir)
    elif args.command == "area":
        build_basin_area_domain(args.src, args.domain, data_dir)
    elif args.command == "merge_gis":
        from .calsim import build_merged_gis
        build_merged_gis(data_dir)
    elif args.command == "calset":
        build_calset(args.domain, args.hruinfo, args.calib, args.simflow, data_dir)
    elif args.command == "fnf":
        build_calib_fnf(args.domain, args.src, data_dir)
    elif args.command == "fnf_xlsx":
        build_calib_fnf_xlsx(args.domain, args.src, data_dir, sheet=args.sheet)
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
