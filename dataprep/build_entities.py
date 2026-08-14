"""Build data/dpl_entities/entities.csv — the multi-timescale training-entity registry.

One row per training entity
(site x timescale x family), carrying: delineation arcs, depth-conversion area,
outlet coordinate, training window, observation count and lineage, and flags.

Families:
  uf_monthly   18 arc-mapped UF subbasins   (uf_locations.csv / uf_monthly.csv)
  usgs_daily    69 reference gauges          (gauges.csv / flow_daily.nc)
  cdec_daily    15 committed basins + CLE + CSN  (cdec15/gage.csv, cdec_fnf/)
  obs11_monthly SHA + TNL                    (fnf_11obs_monthly.csv)

Outlet coordinates, one source per family:
  - UF entities: data/dwr_unimpaired/uf_gauges.csv — the hand-maintained UF
    pour-point table (all 18 arc-mapped UFs, per-row source + note; UF 7 has
    no gauge by construction). Rows sourced `cdec_stations:*` are copies of
    cdec_fnf/stations.csv coordinates; this script asserts they still match
    (drift gate). 
  - USGS entities: gauges.csv lat/lon.
  - CDEC and obs11 entities: the station row in cdec_fnf/stations.csv
    (obs11_TNL: USGS gauge 11525500).

The uf_monthly family carries TWO area columns:

  - area_mi2 — the default: the sum of the entity's CalSim arc areas.
  - area_mi2_swat  — Appendix A SWAT *model* areas, the alternate; read from
    uf_gauges.csv (area_mi2_swat + area_source per row; 16 of 18 UFs — UF 6
    and UF 7 have no usable single model).

  - I_RUB002 (UF 11 / FOL lists) has no CalSim3_Merged polygon — its terrain
    was dissolved into MFA025, so coverage is complete.
  - Some outlets sit a few km outside their polygons (dam/valley-floor
    stations below the delineation terminus). The coordinates are correct —
    do not "fix" them; flagged outlet_below_delineation (BELOW_DAM below).
  - UF 7 is a composite of east-side creeks with no gauge by construction.

Usage (sacsma conda env):
    python dataprep/build_entities.py [--data-dir data] [--out data/dpl_entities/entities.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

SQMI_PER_KM2 = 0.386102
# CDEC daily-FNF station id per cdec15 basin (data/cdec_fnf/README.md alias table).
CDEC_STATION = {
    "SHA": "SIS", "BND": "SBB", "ORO": "FTO", "FOL": "AMF", "MIL": "SBF",
    "PNF": "KGF", "TRM": "KWT", "ISB": "KRI",
    "YRS": "YRS", "MKM": "MKM", "NHG": "NHG", "NML": "NML", "TLG": "TLG",
    "MRC": "MRC", "SCC": "SCC",
}
TULARE = ("ISB", "PNF", "SCC", "TRM")
# BASIN_NESTS (catchments.py): BND as a watershed includes SHA's arc.
BASIN_NESTS = {"BND": ["SHA"]}
# SWAT model areas live in uf_gauges.csv (area_mi2_swat + area_source per
# row) — decision-5 alternate, indicative only; 16 of 18 UFs carry one.
UF_FLAGS = {
    3: "obs_routed_through_lakes", 7: "obs_includes_valley_floor;no_gauge_composite",
    10: "calsim_ref_wetter_summers",
}
# Entities whose (correct) gauge/dam coordinate sits well below the
# delineation terminus (EPSG:3310/5070 km): BND 12.2,
# MKM/UF14 13.0, FOL/UF11 9.5, UF10 9.3 (below Camp Far West), MRC/UF19 7.9,
# ORO/UF08 5.2, UF15 4.7 (below New Hogan).
BELOW_DAM = {"uf_08", "uf_10", "uf_11", "uf_14", "uf_15",
             "uf_19", "cdec_ORO", "cdec_FOL", "cdec_BND", "cdec_MKM",
             "cdec_MRC"}
MONTHLY_START, MONTHLY_END = "1984-10-01", "2014-09-30"
FORCING_END = "2018-12-31"


def _entity(entity_id, family, timescale, site_id, name, delineation, arcs,
            area, area_swat, olat, olon, osrc, t0, t1, n_obs, obs_store,
            flags=""):
    return dict(entity_id=entity_id, family=family, timescale=timescale,
                site_id=site_id, name=name, delineation=delineation,
                arcs=arcs, area_mi2=area, area_mi2_swat=area_swat,
                outlet_lat=olat, outlet_lon=olon, outlet_source=osrc,
                train_start=t0, train_end=t1, n_obs=n_obs,
                obs_store=obs_store, flags=flags)


def _add_flag(row: dict, flag: str) -> None:
    row["flags"] = f"{row['flags']};{flag}" if row["flags"] else flag


def build(data_dir: Path) -> pd.DataFrame:
    uf = pd.read_csv(data_dir / "dwr_unimpaired" / "uf_locations.csv")
    ufg = pd.read_csv(data_dir / "dwr_unimpaired" / "uf_gauges.csv").set_index("uf")
    ufm = pd.read_csv(data_dir / "dwr_unimpaired" / "uf_monthly.csv",
                      parse_dates=["date"])
    gauges = pd.read_csv(data_dir / "usgs" / "gauges.csv", dtype={"gid": str})
    cw = pd.read_csv(data_dir / "calsim" / "calsim_crosswalk.csv")
    barea = pd.read_csv(data_dir / "cdec15" / "basin_area.csv").set_index("basin")
    gage = pd.read_csv(data_dir / "cdec15" / "gage.csv", parse_dates=["date"])
    fnf = pd.read_csv(data_dir / "cdec_fnf" / "fnf_daily.csv", parse_dates=["date"])
    stations = pd.read_csv(data_dir / "cdec_fnf" / "stations.csv").set_index("id")
    obs11 = pd.read_csv(data_dir / "calsim" / "fnf_11obs_monthly.csv",
                        parse_dates=["date"])
    merged = gpd.read_file(data_dir / "calsim" / "gis" / "calsim3.gpkg",
                           layer="CalSim3_Merged", ignore_geometry=True)
    sq_mi = merged.set_index("Connect_No")["SQ_MI"]  # node ids carry no I_ prefix

    def arc_area(arcs: list[str]) -> float:
        return float(sum(sq_mi[a.removeprefix("I_")] for a in arcs))

    cw_arcs = {b: sorted(g["arc"]) for b, g in
               cw[cw["basin_15cdec"].notna()].groupby("basin_15cdec")}

    # Per-gauge finite-day counts from the daily store (window == full record).
    with xr.open_dataset(data_dir / "usgs" / "flow_daily.nc") as ds:
        usgs_nobs = ds["flow_mm"].notnull().sum("time").to_pandas()
        usgs_nobs.index = usgs_nobs.index.astype(str)

    ufm_win = ufm[(ufm["date"] >= MONTHLY_START) & (ufm["date"] <= MONTHLY_END)]
    uf_nobs = ufm_win.dropna(subset=["flow_taf"]).groupby("uf").size()

    rows = []

    # --- uf_monthly ------------------------------------------------------
    # Drift gate: uf_gauges rows that copy stations.csv must still match it.
    for n, g in ufg.iterrows():
        src = str(g["gauge_source"])
        if src.startswith("cdec_stations:"):
            st = src.split(":", 1)[1]
            slat, slon = stations.loc[st, ["lat", "lon"]]
            assert abs(g["lat"] - slat) < 1e-6 and abs(g["lon"] - slon) < 1e-6, \
                f"uf_gauges.csv uf {n} ({src}) drifted from stations.csv"

    for r in uf[uf["n_arcs"] > 0].itertuples():
        n = int(r.uf)
        g = ufg.loc[n]
        has_coord = pd.notna(g["lat"])
        olat = float(g["lat"]) if has_coord else None
        olon = float(g["lon"]) if has_coord else None
        osrc = f"uf_gauges:{g['gauge_source']}" if has_coord else ""
        swat = float(g["area_mi2_swat"]) if pd.notna(g["area_mi2_swat"]) else None
        row = _entity(
            f"uf_{n:02d}", "uf_monthly", "monthly", f"UF {n}", r.name,
            "arcs", r.arcs, round(float(r.area_mi2_calsim), 2), swat,
            olat, olon, osrc, MONTHLY_START, MONTHLY_END, int(uf_nobs[n]),
            "dwr_unimpaired/uf_monthly.csv:flow_taf", UF_FLAGS.get(n, ""))
        rows.append(row)

    # --- usgs_daily -------------------------------------------------------
    for r in gauges.itertuples():
        rows.append(_entity(
            f"usgs_{r.gid}", "usgs_daily", "daily", r.gid, r.station_name,
            "usgs_gpkg", "", round(r.area_km2_delineated * SQMI_PER_KM2, 2),
            None, r.lat, r.lon, f"usgs_gauges:{r.gid}", r.first_obs,
            r.last_obs, int(usgs_nobs[r.gid]), "usgs/flow_daily.nc:flow_mm"))

    # --- cdec_daily: the 15 committed basins ------------------------------
    valid = gage.dropna(subset=["flow"])
    spans = valid.groupby("basin")["date"].agg(["min", "max"])
    for basin in sorted(barea.index):
        arcs = list(cw_arcs.get(basin, []))
        for nest in BASIN_NESTS.get(basin, []):
            arcs += cw_arcs.get(nest, [])
        st = CDEC_STATION[basin]
        olat, olon = stations.loc[st, ["lat", "lon"]]
        t0 = spans.loc[basin, "min"].date().isoformat()
        t1 = min(spans.loc[basin, "max"].date().isoformat(), FORCING_END)
        n_obs = int(((valid["basin"] == basin) & (valid["date"] >= t0)
                     & (valid["date"] <= t1)).sum())
        row = _entity(
            f"cdec_{basin}", "cdec_daily", "daily", basin,
            f"CDEC {st} full natural flow",
            "cdec15_grid_footprint" if basin in TULARE else "arcs",
            ";".join(sorted(arcs)), float(barea.loc[basin, "area_mi2"]), None,
            olat, olon, f"cdec_stations:{st}", t0, t1, n_obs,
            "cdec15/gage.csv:flow",
            "train_only;inherited_cadwr_footprint" if basin in TULARE else "")
        if basin == "PNF":
            # The CADWR and CalSim maps draw the Kings-San Joaquin divide in
            # slightly different places, so ~2.5% of PNF's area also falls
            # inside MIL's polygon and is counted in both basins. The true
            # overlap is smaller (whole cells were counted whenever their
            # center fell inside MIL). 
            _add_flag(row, "footprint_overlaps_MIL_2.5pct")
        rows.append(row)

    # --- cdec_daily: CLE + CSN --------------------------------------------
    fvalid = fnf[(fnf["flow_cfs"] >= 0) & fnf["flow_cfs"].notna()]
    fspans = fvalid.groupby("station")["date"].agg(["min", "max"])
    uf13 = uf.loc[uf["uf"] == 13].iloc[0]
    for st, arcs, area, flags in (
            ("CLE", "I_TRNTY", round(arc_area(["I_TRNTY"]), 2),
             "fnf_computed_at_trinity_dam"),
            ("CSN", uf13["arcs"], round(float(uf13["area_mi2_calsim"]), 2),
             "daily_runs_6pct_below_monthly")):
        t0 = fspans.loc[st, "min"].date().isoformat()
        t1 = min(fspans.loc[st, "max"].date().isoformat(), FORCING_END)
        n_obs = int(((fvalid["station"] == st) & (fvalid["date"] >= t0)
                     & (fvalid["date"] <= t1)).sum())
        rows.append(_entity(
            f"cdec_{st}", "cdec_daily", "daily", st,
            str(stations.loc[st, "name"]), "arcs", arcs, area, None,
            stations.loc[st, "lat"], stations.loc[st, "lon"],
            f"cdec_stations:{st}", t0, t1, n_obs,
            "cdec_fnf/fnf_daily.csv:flow_cfs", flags))

    # --- obs11_monthly: SHA + TNL -----------------------------------------
    lew = gauges.set_index("gid").loc["11525500"]

    def obs11_nobs(basin: str, end: str) -> int:
        m = obs11[(obs11["basin"] == basin) & (obs11["date"] >= MONTHLY_START)
                  & (obs11["date"] <= end)]
        return int(m["obs_mm"].notna().sum())

    rows.append(_entity(
        "obs11_SHA", "obs11_monthly", "monthly", "SHA",
        "Sacramento R at Shasta (11obs FNF)", "arcs", "I_SHSTA",
        round(arc_area(["I_SHSTA"]), 2), None,
        stations.loc["SIS", "lat"], stations.loc["SIS", "lon"],
        "cdec_stations:SIS", MONTHLY_START, MONTHLY_END, obs11_nobs("SHA", MONTHLY_END),
        "calsim/fnf_11obs_monthly.csv:obs_mm", "obs_already_mm"))
    rows.append(_entity(
        "obs11_TNL", "obs11_monthly", "monthly", "TNL",
        "Trinity R (11obs FNF)", "arcs", "I_LWSTN;I_TRNTY",
        round(arc_area(["I_TRNTY", "I_LWSTN"]), 2), None,
        lew["lat"], lew["lon"], "usgs_gauges:11525500",
        MONTHLY_START, "2013-09-30", obs11_nobs("TNL", "2013-09-30"),
        "calsim/fnf_11obs_monthly.csv:obs_mm",
        "obs_already_mm;obs_ends_2014-01"))

    df = pd.DataFrame(rows)
    df.loc[df["entity_id"].isin(BELOW_DAM), "flags"] = (
        df.loc[df["entity_id"].isin(BELOW_DAM), "flags"]
        .map(lambda f: f"{f};outlet_below_delineation" if f
             else "outlet_below_delineation"))

    fam_rank = {"uf_monthly": 0, "usgs_daily": 1, "cdec_daily": 2,
                "obs11_monthly": 3}
    return (df.assign(_r=df["family"].map(fam_rank))
              .sort_values(["_r", "entity_id"]).drop(columns="_r")
              .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--out", default=Path("data/dpl_entities/entities.csv"), type=Path)
    args = ap.parse_args()

    df = build(args.data_dir)
    counts = df["family"].value_counts()
    assert counts["uf_monthly"] == 18, counts
    assert counts["usgs_daily"] == 69, counts
    assert counts["cdec_daily"] == 17, counts
    assert counts["obs11_monthly"] == 2, counts
    assert df["entity_id"].is_unique

    by_fam = df.groupby("family")["n_obs"].sum()
    # 6,480 DWR site-months;
    # 821,412 USGS gauge-days over the reference windows.
    assert by_fam["uf_monthly"] == 6480, by_fam
    assert by_fam["usgs_daily"] == 821412, by_fam

    tnl = df.loc[df["entity_id"] == "obs11_TNL", "area_mi2"].iloc[0]
    assert abs(tnl - 719.0) < 0.1, tnl
    missing = df[df["outlet_lat"].isna()]["entity_id"].tolist()
    assert missing == ["uf_07"], missing  # composite, null by design

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} entities")
    print(counts.to_string())
    print("n_obs by family:", dict(by_fam))
    print(f"gates ok: TNL {tnl} mi^2; outlets missing: {missing} (by design)")


if __name__ == "__main__":
    main()
