"""Ingest the cleaned USGS daily flows for gauges inside the CalSim3 domain.

Source is the training store of the sibling **neuralhyd-ca** repo, whose QA/QC
pipeline retrieves USGS parameter ``00060`` (discharge) and screens it — the
"cleaned" flows.  This script does not re-clean anything; it selects the gauges
that belong to the CalSim3 domain and republishes them in this repo's shape.

    <neuralhyd-ca>/data/training/flow.zarr        224 basins x 25202 days, cfs
    <neuralhyd-ca>/data/training/watersheds/      delineations + area table

Facts established by reading the source, not assumed:

* **Units are cfs.**  Mean flow over the largest basins works out to 143-965
  mm/yr as cfs, which is the right band for California; read as mm/day the same
  numbers imply millions of mm/yr.  ``clean_flows.py`` labels its plots "Flow
  (cfs)" and retrieves NWIS ``00060``.
* **``tier`` is a hydrologic regime, not a data-quality grade** — 1 rain,
  2 mixed, 3 snow (``analyse_flow_extremes.TIER_LABELS``).  It does not rank
  record quality and must not be used to filter on that basis.
* **14 of the 224 ids are synthetic** ``99xxxxxxx`` footprints that
  ``build_sacsma_basins.py`` injected into that training set — they ARE the
  SAC-SMA/CDEC basins, already in this repo under their own names, so they are
  excluded by default (``--include-synthetic`` keeps them).
* **Time is int32 days since 1970-01-01**, contiguous daily 1950-01-01 onward;
  the epoch is asserted at run time rather than trusted.
* Records are **sparse** — median completeness ~44 % of the 1950-2019 window.

Selection: a gauge is "inside the CalSim domain" when at least ``--min-frac``
(default 0.90) of its delineated watershed area falls inside the union of
``data/calsim/gis/calsim3.gpkg`` layer ``CalSim3_And_GooseLake``, measured in
EPSG:3310 (equal-area).  90 % keeps basins that merely clip a boundary cell
during rasterisation, the same tolerance the repo applies to footprint
over-reach elsewhere; nothing in this dataset falls between 50 % and 90 %.

Units written: **cfs is canonical** (verbatim from the source, reproducible
against USGS) plus a derived ``flow_mm`` at mm/day, since the rest of the repo
is area-normalised.  The conversion uses the **delineated** polygon area — the
basin the flow record is actually attributed to here — not the USGS-reported
drainage area; both are in ``gauges.csv`` along with their ratio, so any
consumer can re-derive.  mm/day = cfs * 2.4465755 / area_km2.

Outputs (``data/usgs/``):

    flow_daily.nc            flow_cfs + flow_mm, (gauge, time) float32, LFS
    gauges.csv               id, name, lat/lon, both areas, tier, coverage
    gis/usgs_watersheds.gpkg the delineations, subset, EPSG:4326

Needs **zarr**, so it runs in the ``neuralhyd`` conda env, not ``sacsma``
(which has no zarr).  Everything downstream just reads the .nc.  Station names,
coordinates and USGS drainage areas are fetched from the NWIS site service;
``--no-nwis`` skips that and leaves those columns blank.

Usage
-----
    conda run -n neuralhyd python dataprep/usgs_flows.py
    conda run -n neuralhyd python dataprep/usgs_flows.py --min-frac 0.999
    conda run -n neuralhyd python dataprep/usgs_flows.py --verify
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NH_DIR = Path(os.environ.get("SACSMA_NEURALHYD_DIR")
              or r"C:\Users\warnold_la\Local\repos\neuralhyd-ca\data\training")
OUT_DIR = REPO / "data" / "usgs"
CALSIM_GPKG = REPO / "data" / "calsim" / "gis" / "calsim3.gpkg"
CALSIM_LAYER = "CalSim3_And_GooseLake"

EPOCH = np.datetime64("1970-01-01", "D")
EQUAL_AREA = 3310                     # California Teale Albers
#: 1 cfs over 1 km2 for one day, in mm:  0.0283168466 m3/s * 86400 s / 1e6 m2 * 1000
CFS_KM2_TO_MM = 0.0283168466 * 86400 / 1e6 * 1000
SQMI_TO_KM2 = 2.589988
TIER_LABEL = {1: "rain", 2: "mixed", 3: "snow"}
NWIS_SITE = "https://waterservices.usgs.gov/nwis/site/"


def is_synthetic(gid: str) -> bool:
    """``99xxxxxxx`` ids are SAC-SMA basin footprints, not USGS gauges."""
    return gid.startswith("99") and len(gid) == 9


def load_flow(nh_dir: Path):
    """flow.zarr -> (ids, dates, cfs[gauge, time], tier)."""
    import zarr

    g = zarr.open_group(str(nh_dir / "flow.zarr"), mode="r")
    ids = np.array([str(b) for b in g["basin"][:]])
    raw = np.asarray(g["time"][:], dtype=np.int64)
    if not np.array_equal(np.diff(raw), np.ones(len(raw) - 1, np.int64)):
        raise SystemExit("flow.zarr time axis is not contiguous daily")
    dates = EPOCH + raw.astype("timedelta64[D]")
    if dates[0] != np.datetime64("1950-01-01"):
        raise SystemExit(f"unexpected epoch: time[0] decodes to {dates[0]}, not 1950-01-01")
    return ids, dates, np.asarray(g["flow"][:], np.float32), np.asarray(g["tier"][:])


def footprint_fraction(nh_dir: Path, gpkg: Path, layer: str) -> gpd.GeoDataFrame:
    """Each delineation with the fraction of its area inside the CalSim footprint."""
    poly = gpd.read_file(nh_dir / "watersheds" / "watersheds.geojson")
    poly["gid"] = poly["Pour Point ID"].astype(str)
    poly = poly.to_crs(EQUAL_AREA)
    foot = gpd.read_file(gpkg, layer=layer).to_crs(EQUAL_AREA).union_all()
    poly["area_km2_delineated"] = poly.geometry.area / 1e6
    poly["frac_in_calsim"] = poly.geometry.intersection(foot).area / poly.geometry.area
    return poly


def nwis_sites(ids: list[str]) -> pd.DataFrame:
    """Authoritative station name, coordinates and drainage area from NWIS."""
    import time

    import requests

    cols = ["site_no", "station_nm", "dec_lat_va", "dec_long_va", "drain_area_va"]
    frames = []
    for i in range(0, len(ids), 50):             # the service caps the site list
        chunk = ids[i:i + 50]
        text = ""
        for attempt in range(5):                 # NWIS 503s freely under load
            try:
                r = requests.get(NWIS_SITE, timeout=120, params={
                    "format": "rdb", "sites": ",".join(chunk),
                    "siteOutput": "expanded", "siteStatus": "all"})
                r.raise_for_status()
                text = r.text
                break
            except Exception:                    # noqa: BLE001 - retry anything transient
                if attempt == 4:
                    raise
                time.sleep(3 * 2**attempt)
        body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
        tab = pd.read_csv(io.StringIO(body), sep="\t", dtype=str)
        frames.append(tab.iloc[1:])              # row 0 is the RDB format spec
    site = pd.concat(frames, ignore_index=True)
    missing = [c for c in cols if c not in site.columns]
    if missing:
        raise RuntimeError(f"NWIS response missing {missing}")
    site = site[cols].rename(columns={
        "site_no": "gid", "station_nm": "station_name",
        "dec_lat_va": "lat", "dec_long_va": "lon"})
    for c in ("lat", "lon", "drain_area_va"):
        site[c] = pd.to_numeric(site[c], errors="coerce")
    site["area_km2_usgs"] = site.pop("drain_area_va") * SQMI_TO_KM2
    return site.drop_duplicates("gid")


def build(nh_dir: Path, out_dir: Path, min_frac: float,
          include_synthetic: bool, use_nwis: bool) -> None:
    import xarray as xr

    ids, dates, cfs, tier = load_flow(nh_dir)
    print(f"flow.zarr: {len(ids)} basins x {len(dates)} days "
          f"({dates[0]}..{dates[-1]})")

    poly = footprint_fraction(nh_dir, CALSIM_GPKG, CALSIM_LAYER)
    frac = dict(zip(poly["gid"], poly["frac_in_calsim"]))
    darea = dict(zip(poly["gid"], poly["area_km2_delineated"]))

    keep = []
    for k, gid in enumerate(ids):
        if not include_synthetic and is_synthetic(gid):
            continue
        if frac.get(gid, 0.0) >= min_frac:
            keep.append(k)
    keep = np.array(keep)
    if not len(keep):
        raise SystemExit("no gauges selected")
    sel_ids = ids[keep]
    print(f"selected {len(keep)} gauges (>= {min_frac:.0%} of area inside "
          f"{CALSIM_LAYER}, synthetic {'kept' if include_synthetic else 'excluded'})")

    sel_cfs = cfs[keep]
    if np.nanmin(sel_cfs) < 0:
        n = int(np.nansum(sel_cfs < 0))
        print(f"  WARNING: {n} negative discharge values in the source", file=sys.stderr)

    area = np.array([darea[g] for g in sel_ids], np.float64)
    sel_mm = (sel_cfs * CFS_KM2_TO_MM / area[:, None]).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"flow_cfs": (("gauge", "time"), sel_cfs),
         "flow_mm": (("gauge", "time"), sel_mm)},
        coords={"gauge": sel_ids.astype(str), "time": dates},
        attrs={
            "title": "Cleaned USGS daily discharge, gauges inside the CalSim3 domain",
            "source": ("neuralhyd-ca data/training/flow.zarr (USGS NWIS 00060, "
                       "QA/QC'd); selection >= "
                       f"{min_frac:.3f} of delineated area inside "
                       f"{CALSIM_GPKG.name}:{CALSIM_LAYER} (EPSG:{EQUAL_AREA})"),
            "units": ("flow_cfs: ft3/s as delivered (canonical). flow_mm: mm/day, "
                      "cfs * %.7f / area_km2_delineated" % CFS_KM2_TO_MM),
        })
    ds["flow_cfs"].attrs["units"] = "ft3/s"
    ds["flow_mm"].attrs["units"] = "mm/day"
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    nc = out_dir / "flow_daily.nc"
    ds.to_netcdf(nc, encoding=enc)
    print(f"wrote {nc}  ({nc.stat().st_size / 1e6:.1f} MB)")

    finite = np.isfinite(sel_cfs)
    first = [str(dates[f.argmax()])[:10] if f.any() else "" for f in finite]
    last = [str(dates[len(dates) - 1 - f[::-1].argmax()])[:10] if f.any() else ""
            for f in finite]
    tbl = pd.DataFrame({
        "gid": sel_ids,
        "tier": tier[keep],
        "tier_label": [TIER_LABEL.get(int(t), "?") for t in tier[keep]],
        # 6 dp, not 3: the smallest basin here is 4.9 km2, where 3 dp already
        # costs 1e-4 relative and makes the published mm/day irreproducible
        # from this table.  Precision has to beat the float32 series itself.
        "area_km2_delineated": area.round(6),
        "frac_in_calsim": [round(frac[g], 5) for g in sel_ids],
        "n_obs": finite.sum(axis=1),
        "coverage_frac": (finite.mean(axis=1)).round(4),
        "first_obs": first,
        "last_obs": last,
        "synthetic": [is_synthetic(g) for g in sel_ids],
    })
    if use_nwis:
        real = [g for g in sel_ids if not is_synthetic(g)]
        try:
            site = nwis_sites(real)
            tbl = tbl.merge(site, on="gid", how="left")
            tbl["area_ratio_delin_over_usgs"] = (
                tbl["area_km2_delineated"] / tbl["area_km2_usgs"]).round(4)
            got = tbl["station_name"].notna().sum()
            print(f"NWIS: matched {got}/{len(real)} sites")
            off = tbl["area_ratio_delin_over_usgs"]
            bad = tbl[(off < 0.9) | (off > 1.1)]
            if len(bad):
                print(f"  {len(bad)} gauge(s) whose delineated area differs from the "
                      f"USGS drainage area by >10% -- mm/day there depends on which "
                      f"you trust:")
                for _, r in bad.iterrows():
                    print(f"    {r['gid']}  delineated {r['area_km2_delineated']:8.1f} "
                          f"vs USGS {r['area_km2_usgs']:8.1f} km2  "
                          f"(x{r['area_ratio_delin_over_usgs']})")
        except Exception as exc:                 # noqa: BLE001 - metadata is optional
            print(f"NWIS lookup failed ({exc}); name/coords left blank", file=sys.stderr)

    csv = out_dir / "gauges.csv"
    tbl.sort_values("gid").to_csv(csv, index=False)
    print(f"wrote {csv}  ({len(tbl)} rows)")

    gis = out_dir / "gis"
    gis.mkdir(exist_ok=True)
    sub = poly[poly["gid"].isin(set(sel_ids))].copy()
    sub = sub[["gid", "area_km2_delineated", "frac_in_calsim", "geometry"]]
    gpkg = gis / "usgs_watersheds.gpkg"
    # EPSG:4326 to match data/calsim/gis/calsim3.gpkg
    sub.to_crs(4326).to_file(gpkg, layer="usgs_watersheds", driver="GPKG")
    print(f"wrote {gpkg}  ({gpkg.stat().st_size / 1e6:.1f} MB, {len(sub)} polygons)")


def verify(out_dir: Path) -> int:
    import xarray as xr

    nc, csv = out_dir / "flow_daily.nc", out_dir / "gauges.csv"
    if not nc.exists():
        print(f"missing {nc}", file=sys.stderr)
        return 1
    ds = xr.open_dataset(nc)
    tbl = pd.read_csv(csv, dtype={"gid": str})
    rc = 0

    gauges = [str(g) for g in ds.gauge.values]
    if sorted(gauges) != sorted(tbl["gid"]):
        print("gauges.csv and flow_daily.nc disagree on the gauge list", file=sys.stderr)
        rc = 1

    cfs = ds["flow_cfs"].values
    mm = ds["flow_mm"].values
    area = tbl.set_index("gid").loc[gauges, "area_km2_delineated"].to_numpy()
    back = mm * area[:, None] / CFS_KM2_TO_MM
    ok = np.isfinite(cfs)
    err = np.abs(back[ok] - cfs[ok]) / np.maximum(np.abs(cfs[ok]), 1e-6)
    print(f"round-trip mm -> cfs: max rel err {np.nanmax(err):.2e} "
          f"over {ok.sum():,} values")
    # float32 storage of flow_mm alone costs ~1e-7 relative; anything near 1e-4
    # means gauges.csv no longer carries enough area precision to re-derive
    if np.nanmax(err) > 1e-5:
        print("  round-trip error too large", file=sys.stderr)
        rc = 1

    neg = int(np.nansum(cfs < 0))
    print(f"negative discharge values: {neg}")

    # the real check on units + areas: a wrong factor or a wrong drainage area
    # throws the runoff depth orders of magnitude out of the California band
    depth = np.nanmean(mm, axis=1) * 365.25
    print(f"mean annual runoff depth mm/yr: min {depth.min():.0f}, "
          f"median {np.median(depth):.0f}, max {depth.max():.0f}")
    off = (depth < 2) | (depth > 2500)
    if off.any():
        for k in np.nonzero(off)[0]:
            print(f"  implausible: {gauges[k]} {depth[k]:.0f} mm/yr", file=sys.stderr)
        rc = 1
    cov = tbl["coverage_frac"]
    print(f"gauges {len(tbl)}; coverage median {cov.median():.1%}, "
          f"min {cov.min():.1%}, max {cov.max():.1%}")
    print(f"tier split: " + ", ".join(
        f"T{t}={int((tbl['tier'] == t).sum())}" for t in sorted(tbl["tier"].unique())))
    print(f"time {str(ds.time.values[0])[:10]}..{str(ds.time.values[-1])[:10]}, "
          f"n={ds.sizes['time']}")
    ds.close()
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--nh-dir", type=Path, default=NH_DIR, help=f"[{NH_DIR}]")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help=f"[{OUT_DIR}]")
    ap.add_argument("--min-frac", type=float, default=0.90,
                    help="minimum fraction of watershed area inside CalSim3 [0.90]")
    ap.add_argument("--include-synthetic", action="store_true",
                    help="keep the 99xxxxxxx SAC-SMA footprints (excluded by default)")
    ap.add_argument("--no-nwis", action="store_true", help="skip the NWIS metadata lookup")
    ap.add_argument("--verify", action="store_true", help="check the written store")
    a = ap.parse_args(argv)

    if a.verify:
        return verify(a.out_dir)
    if not a.nh_dir.exists():
        raise SystemExit(f"neuralhyd-ca training dir not found: {a.nh_dir}")
    build(a.nh_dir, a.out_dir, a.min_frac, a.include_synthetic, not a.no_nwis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
