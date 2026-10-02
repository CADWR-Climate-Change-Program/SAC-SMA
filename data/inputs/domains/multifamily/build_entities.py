"""Build data/inputs/domains/multifamily/entities.csv — the multi-timescale training-entity registry.

One row per training entity
(site x timescale x family), carrying: delineation arcs, depth-conversion area,
outlet coordinate, true record start, training window, observation count and
lineage, and flags.

record_start — the source's true data-availability start (train_start/end
stay the training-window cut); one source per family:
  - UF entities: first non-null month in uf_monthly.csv (the published
    record, WY1922-2014); the store's month-end stamps are normalized to
    the month start to match the window columns' convention.
  - USGS entities: gauges.csv first_obs (train window == record window).
  - CDEC entities: the earlier of the station's advertised daily-FNF start
    (the CDEC stations.csv, available_from) and the committed store's first
    day - CADWR's gage_15cdec.csv can predate CDEC's advertised entry and vice
    versa.

Families:
  uf_monthly    9 arc-mapped UF subbasins   (uf_locations.csv / uf_monthly.csv;
                18 built, 9 dropped as monthly twins of long-record dailies)
  usgs_daily   69 reference gauges          (gauges.csv / flow_daily.nc)
  cdec_daily   17 = 15 committed + CLE + CSN  (gage_15cdec.csv, fnf_daily.csv)
  obs11_monthly SHA + TNL built for lineage, both dropped (SIS/CLE daily twins)
  calsim_monthly  196 CalSim3 rim INFLOW arcs — ONLY with --calsim-arcs, appended
                after the 95 (whose rows stay byte-identical): one entity per arc
                of data/targets/calsim3/calsim3_inflow_monthly_mm.csv, cs_<ARC>, the arc's
                SQ_MI as area, n_obs = its trainable months under
                data/targets/calsim3/arc_obs_mask.csv, tier/duplicate/training-USGS flags
                from data/targets/calsim3/arc_hierarchy.csv (data/targets/calsim3/build_calsim_arcs.py
                builds all three).  No outlet coordinate: flow lengths trace to
                the footprint exit, like uf_07.

Outlet coordinates, one source per family:
  - UF entities: data/targets/dwr_unimpaired/uf_gauges.csv (hand-maintained; rows
    tagged `cdec_stations:*` copy the CDEC stations.csv coordinates and are
    drift-gated against it below).
  - USGS entities: gauges.csv lat/lon.
  - CDEC and obs11 entities: the station row in the CDEC stations.csv
    (obs11_TNL: USGS gauge 11525500).

The uf_monthly family carries TWO area columns:

  - area_mi2 — the default: the sum of the entity's CalSim arc areas.
  - area_mi2_swat  — Appendix A SWAT *model* areas, the alternate; read from
    uf_gauges.csv (area_mi2_swat + area_source per row; 16 of 18 UFs — UF 6
    and UF 7 have no usable single model).

  - I_RUB002 (UF 11 / FOL lists) has no CalSim3_Merged polygon — its terrain
    was dissolved into MFA025, so coverage is complete.
  - Some outlets sit a few km outside their polygons (dam/valley-floor
    stations below the delineation terminus). The coordinates are the
    stations' own and are kept; flagged outlet_below_delineation (BELOW_DAM
    below).
  - UF 7 is a composite of east-side creeks with no gauge by construction.

Usage (sacsma conda env):
    python data/inputs/domains/multifamily/build_entities.py [--data-dir data]
        [--out CSV] [--calsim-arcs]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import DATA, layout  # noqa: E402


def _store(path: Path, column: str) -> str:
    """How the registry names an observation store: its path relative to the data root,
    then the column that holds the series."""
    return f"{layout.relative(path)}:{column}"


SQMI_PER_KM2 = 0.386102
# CDEC daily-FNF station id per cdec15 basin (data/targets/cdec/README.md alias table).
CDEC_STATION = {
    "SHA": "SIS", "BND": "SBB", "ORO": "FTO", "FOL": "AMF", "MIL": "SBF",
    "PNF": "KGF", "TRM": "KWT", "ISB": "KRI",
    "YRS": "YRS", "MKM": "MKM", "NHG": "NHG", "NML": "NML", "TLG": "TLG",
    "MRC": "MRC", "SCC": "SCC",
}
TULARE = ("ISB", "PNF", "SCC", "TRM")
# BASIN_NESTS (catchments.py): BND as a watershed includes SHA's arc.
BASIN_NESTS = {"BND": ["SHA"]}
# Arcs the crosswalk cannot assign (no inflow series) that still belong in a
# training footprint.  BND's FNF is naturalized at Bend Bridge, whose published
# 8,900 mi^2 drainage includes the valley floor carried only by the series-less
# I_SRBB_VAL node; sacsma/calsim/catchments.py (VALLEY_SYSTEMS) already adds
# that node to BND's CalSim catchment -- this keeps the registry consistent.
EXTRA_ARCS = {"BND": ["I_SRBB_VAL"]}

# The inverse correction: arcs the crosswalk files under a basin's CalSim
# system whose water the observing gauge never sees.  Deer Creek joins the
# Yuba below the Smartville gauge, so CalSim correctly books its two arcs
# as Yuba-system inflow, but the YRS full natural flow (naturalized at the
# gauge, published 1,108 mi^2) contains none of that water -- keeping them
# would average ~64 mi^2 of unobserved runoff into the training depth.
# The crosswalk itself is untouched: it states CalSim's delivery topology,
# and validation aggregations keep using it.
TRIM_ARCS = {"YRS": ["I_DER001", "I_DER004"]}
# SWAT model areas live in uf_gauges.csv (area_mi2_swat + area_source per
# row)
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
# De-dup rule: a watershed does NOT train at both daily and monthly unless
# its daily record is short (starts ~2000 or later). Each monthly entity
# below shares its arc set with a CDEC daily whose record is contiguous back to
# 1985-88 (advertised starts in the CDEC stations.csv; completeness verified
# against gage_15cdec.csv / fnf_daily.csv: SHA 99.7%, TLG 90.7% scattered-gap,
# NML 98.1%, MIL 98.3%, CLE 94.5% usable), so the monthly twin is dropped.
# The short-daily pairs STAY: uf_13 (CSN 1999-04) and uf_06 (SBB 1999-05).
# obs11_TNL is CLE's twin at 96% shared area (TNL adds I_LWSTN, whose cells
# stay in the training basis via usgs_11525500).
DEDUP_DROPS = {
    "uf_08": "cdec_ORO",      # FTO 1985-04
    "uf_09": "cdec_YRS",      # YRS 1987-05
    "uf_11": "cdec_FOL",      # AMF 1987-06
    "uf_14": "cdec_MKM",      # MKM 1986-04
    "uf_15": "cdec_NHG",      # NHG 1987-05
    "uf_16": "cdec_NML",      # NML 1987-06
    "uf_18": "cdec_TLG",      # TLG 1986-04
    "uf_19": "cdec_MRC",      # MRC 1988-01
    "uf_22": "cdec_MIL",      # SBF 1987-05
    "obs11_SHA": "cdec_SHA",  # SIS 1987-05
    "obs11_TNL": "cdec_CLE",  # CLE 1986-04
}

# Target-validity drops (built like the de-dup twins, then removed); empty
# by default.  UF 3 is the documented case: its observation is the routed outflow at Rumsey,
# below Clear Lake and Indian Valley, while its four arcs are the inflows
# CalSim routes through those lakes itself (arc sum +16% volume, r 0.877 vs
# the obs -- storage attenuation plus net lake evaporation, a shape
# difference no scaling fixes), so a lake-free cell parameterization can
# only fit that series by learning the lakes' behavior.  uf_03 is kept in
# the registry (flag obs_routed_through_lakes) so a run can include it when
# it is the only supervision of the Cache Creek cells; whether it trains is
# the run's choice (``--basins``).  Add "uf_03" here to drop it at build time.
TARGET_DROPS: tuple[str, ...] = ()

#: the CalSim3 arc family (--calsim-arcs): training window = WY1950-2015 (the arc
#: mask removes the WY1976-85 holdout and every non-record month)
CALSIM_FAMILY = "calsim_monthly"
CALSIM_TRAIN_START, CALSIM_TRAIN_END = "1949-10-01", "2015-09-30"
CALSIM_OBS_STORE = _store(layout.calsim3_targets(name="calsim3_inflow_monthly_mm.csv"), "depth_mm")


def _entity(entity_id, family, timescale, site_id, name, delineation, arcs,
            area, area_swat, olat, olon, osrc, rec0, t0, t1, n_obs,
            obs_store, flags=""):
    return dict(entity_id=entity_id, family=family, timescale=timescale,
                site_id=site_id, name=name, delineation=delineation,
                arcs=arcs, area_mi2=area, area_mi2_swat=area_swat,
                outlet_lat=olat, outlet_lon=olon, outlet_source=osrc,
                record_start=rec0, train_start=t0, train_end=t1, n_obs=n_obs,
                obs_store=obs_store, flags=flags)


def _add_flag(row: dict, flag: str) -> None:
    row["flags"] = f"{row['flags']};{flag}" if row["flags"] else flag


def build(data_dir: Path) -> pd.DataFrame:
    uf = pd.read_csv(layout.dwr_unimpaired(data_dir, "uf_locations.csv"))
    ufg = pd.read_csv(layout.dwr_unimpaired(data_dir, "uf_gauges.csv")).set_index("uf")
    ufm = pd.read_csv(layout.dwr_unimpaired(data_dir, "uf_monthly.csv"),
                      parse_dates=["date"])
    gauges = pd.read_csv(layout.usgs_gauges(data_dir), dtype={"gid": str})
    cw = pd.read_csv(layout.crosswalk(data_dir))
    barea = pd.read_csv(layout.basin_area(data_dir, "15cdec")).set_index("basin")
    gage = pd.read_csv(layout.gage_15cdec(data_dir), parse_dates=["date"])
    fnf = pd.read_csv(layout.cdec_fnf(data_dir, "fnf_daily.csv"), parse_dates=["date"])
    stations = pd.read_csv(layout.cdec_fnf(data_dir, "stations.csv")).set_index("id")
    obs11 = pd.read_csv(layout.callite_fnf(data_dir, "11obs"),
                        parse_dates=["date"])
    merged = gpd.read_file(layout.calsim3_gpkg(data_dir),
                           layer="CalSim3_Merged", ignore_geometry=True)
    sq_mi = merged.set_index("Connect_No")["SQ_MI"]  # node ids carry no I_ prefix

    def arc_area(arcs: list[str]) -> float:
        return float(sum(sq_mi[a.removeprefix("I_")] for a in arcs))

    cw_arcs = {b: sorted(g["arc"]) for b, g in
               cw[cw["basin_15cdec"].notna()].groupby("basin_15cdec")}

    # Per-gauge finite-day counts from the daily store (window == full record).
    with xr.open_dataset(layout.usgs_flow(data_dir)) as ds:
        usgs_nobs = ds["flow_mm"].notnull().sum("time").to_pandas()
        usgs_nobs.index = usgs_nobs.index.astype(str)

    ufm_win = ufm[(ufm["date"] >= MONTHLY_START) & (ufm["date"] <= MONTHLY_END)]
    uf_nobs = ufm_win.dropna(subset=["flow_taf"]).groupby("uf").size()
    def _month_start(ts: pd.Timestamp) -> str:
        return ts.to_period("M").start_time.date().isoformat()

    uf_rec = (ufm.dropna(subset=["flow_taf"]).groupby("uf")["date"].min()
              .map(_month_start))
    obs11_rec = (obs11.dropna(subset=["obs_mm"]).groupby("basin")["date"]
                 .min().map(_month_start))

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
            olat, olon, osrc, uf_rec[n],
            MONTHLY_START, MONTHLY_END, int(uf_nobs[n]),
            _store(layout.dwr_unimpaired(name="uf_monthly_mm.csv"), "depth_mm"), UF_FLAGS.get(n, ""))
        rows.append(row)

    # --- usgs_daily -------------------------------------------------------
    for r in gauges.itertuples():
        rows.append(_entity(
            f"usgs_{r.gid}", "usgs_daily", "daily", r.gid, r.station_name,
            "usgs_gpkg", "", round(r.area_km2_delineated * SQMI_PER_KM2, 2),
            None, r.lat, r.lon, f"usgs_gauges:{r.gid}", r.first_obs,
            r.first_obs, r.last_obs, int(usgs_nobs[r.gid]),
            _store(layout.usgs_flow(), "flow_mm")))

    # --- cdec_daily: the 15 committed basins ------------------------------
    valid = gage.dropna(subset=["flow"])
    spans = valid.groupby("basin")["date"].agg(["min", "max"])
    for basin in sorted(barea.index):
        arcs = list(cw_arcs.get(basin, []))
        for nest in BASIN_NESTS.get(basin, []):
            arcs += cw_arcs.get(nest, [])
        arcs += EXTRA_ARCS.get(basin, [])
        arcs = [a for a in arcs if a not in TRIM_ARCS.get(basin, ())]
        st = CDEC_STATION[basin]
        olat, olon = stations.loc[st, ["lat", "lon"]]
        t0 = spans.loc[basin, "min"].date().isoformat()
        t1 = min(spans.loc[basin, "max"].date().isoformat(), FORCING_END)
        n_obs = int(((valid["basin"] == basin) & (valid["date"] >= t0)
                     & (valid["date"] <= t1)).sum())
        rec0 = min(str(stations.loc[st, "available_from"]), t0)
        row = _entity(
            f"cdec_{basin}", "cdec_daily", "daily", basin,
            f"CDEC {st} full natural flow",
            "sacsma_15cdec_gis" if basin in TULARE else "arcs",
            ";".join(sorted(arcs)), float(barea.loc[basin, "area_mi2"]), None,
            olat, olon, f"cdec_stations:{st}", rec0, t0, t1, n_obs,
            _store(layout.gage_15cdec(), "flow"),
            "train_only" if basin in TULARE else "")
        # PNF carries no overlap flag: on the SACSMA_15CDEC polygons the PNF x
        # MLRTN overlap measures 0.09% (whole-cell counting on the cdec15_grid
        # cell sets overstates it at 2.5%).
        if basin == "TRM":
            # The SACSMA_15CDEC polygon measures 575.8 mi^2 vs the published
            # 561; area_mi2 keeps the published value as the depth basis.
            _add_flag(row, "polygon_2.6pct_above_published_area")
        if basin == "BND":
            _add_flag(row, "footprint_includes_valley_node")
        if basin == "YRS":
            _add_flag(row, "footprint_excludes_below_gauge_arcs")
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
            f"cdec_stations:{st}",
            min(str(stations.loc[st, "available_from"]), t0), t0, t1, n_obs,
            _store(layout.cdec_fnf(name="fnf_daily_mm.csv"), "depth_mm"), flags))

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
        "cdec_stations:SIS", obs11_rec["SHA"],
        MONTHLY_START, MONTHLY_END, obs11_nobs("SHA", MONTHLY_END),
        _store(layout.callite_fnf(domain="11obs"), "obs_mm"), "obs_already_mm"))
    rows.append(_entity(
        "obs11_TNL", "obs11_monthly", "monthly", "TNL",
        "Trinity R (11obs FNF)", "arcs", "I_LWSTN;I_TRNTY",
        round(arc_area(["I_TRNTY", "I_LWSTN"]), 2), None,
        lew["lat"], lew["lon"], "usgs_gauges:11525500",
        obs11_rec["TNL"],
        MONTHLY_START, "2013-09-30", obs11_nobs("TNL", "2013-09-30"),
        _store(layout.callite_fnf(domain="11obs"), "obs_mm"),
        "obs_already_mm;obs_ends_2014-01"))

    df = pd.DataFrame(rows)
    df.loc[df["entity_id"].isin(BELOW_DAM), "flags"] = (
        df.loc[df["entity_id"].isin(BELOW_DAM), "flags"]
        .map(lambda f: f"{f};outlet_below_delineation" if f
             else "outlet_below_delineation"))

    # De-dup: every candidate row is built first (keeps the lineage and lets
    # the twin assertions run), then the monthly twins are dropped.
    ids = set(df["entity_id"])
    assert set(DEDUP_DROPS) <= ids and set(DEDUP_DROPS.values()) <= ids
    assert set(TARGET_DROPS) <= ids
    df = df[~df["entity_id"].isin(DEDUP_DROPS)]
    df = df[~df["entity_id"].isin(TARGET_DROPS)]

    fam_rank = {"uf_monthly": 0, "usgs_daily": 1, "cdec_daily": 2,
                "obs11_monthly": 3}
    return (df.assign(_r=df["family"].map(fam_rank))
              .sort_values(["_r", "entity_id"]).drop(columns="_r")
              .reset_index(drop=True))


def build_calsim(data_dir: Path) -> pd.DataFrame:
    """The calsim_monthly rows: one per arc of the depth store (hierarchy order)."""
    cdir = layout.calsim3_targets(data_dir, "")
    hier = pd.read_csv(cdir / "arc_hierarchy.csv").set_index("arc")
    depth = pd.read_csv(cdir / "calsim3_inflow_monthly_mm.csv", parse_dates=["date"])
    mask = pd.read_csv(cdir / "arc_obs_mask.csv", parse_dates=["date"])
    deriv = pd.read_csv(cdir / "calsim3_arc_derivation.csv").set_index("arc")
    arcs = [a for a in hier.index if a in set(depth["arc"])]
    fin = depth[depth["depth_mm"].notna()]
    rec0 = fin.groupby("arc")["date"].min()
    t0, t1 = pd.Timestamp(CALSIM_TRAIN_START), pd.Timestamp(CALSIM_TRAIN_END)
    # trainable months: in the mask, finite in the store, inside the window
    m = mask.merge(fin[["arc", "date"]], on=["arc", "date"])
    m = m[(m["date"] >= t0) & (m["date"] <= t1)]
    n_obs = m.groupby("arc").size()
    rows = []
    for arc in arcs:
        h = hier.loc[arc]
        flags = [f"tier_{h['tier']}"]
        if bool(h["train_default"]):
            flags.append("train_default")
        if isinstance(h["duplicates_entity"], str) and h["duplicates_entity"]:
            flags.append(f"duplicates_{h['duplicates_entity']}")
        if isinstance(h["usgs_training_gauge"], str) and h["usgs_training_gauge"]:
            flags += [f"training_usgs_{g}" for g in h["usgs_training_gauge"].split(";")]
        name = (str(deriv.loc[arc, "name"]) if arc in deriv.index
                else f"{arc} (CalSim3 rim inflow)")
        rows.append(_entity(
            f"cs_{arc}", CALSIM_FAMILY, "monthly", arc, name, "arcs", arc,
            float(h["sq_mi"]), None, None, None, "",
            rec0[arc].to_period("M").start_time.date().isoformat(),
            CALSIM_TRAIN_START, CALSIM_TRAIN_END, int(n_obs.get(arc, 0)),
            CALSIM_OBS_STORE, ";".join(flags)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=DATA, type=Path)
    ap.add_argument("--out", default=None, type=Path,
                    help="default: the registry of --data-dir")
    ap.add_argument("--calsim-arcs", action="store_true",
                    help="also append the calsim_monthly family (196 CalSim3 rim arcs) "
                         "after the 95 base entities")
    args = ap.parse_args()

    args.out = args.out or layout.entities(args.data_dir)
    df = build(args.data_dir)
    counts = df["family"].value_counts()
    assert counts["uf_monthly"] == 9, counts
    assert counts["usgs_daily"] == 69, counts
    assert counts["cdec_daily"] == 17, counts
    assert "obs11_monthly" not in counts, counts  # both rows are DEDUP_DROPS
    assert df["entity_id"].is_unique

    by_fam = df.groupby("family")["n_obs"].sum()
    # 3,240 DWR site-months (9 UFs x 360; 6,480 before the de-dup drops);
    # 821,412 USGS gauge-days over the reference windows.
    assert by_fam["uf_monthly"] == 3240, by_fam
    assert by_fam["usgs_daily"] == 821412, by_fam

    cle = df.loc[df["entity_id"] == "cdec_CLE", "area_mi2"].iloc[0]
    assert abs(cle - 692.86) < 0.1, cle
    late = df[df["record_start"] > df["train_start"]]
    assert late.empty, late["entity_id"].tolist()
    missing = df[df["outlet_lat"].isna()]["entity_id"].tolist()
    assert missing == ["uf_07"], missing  # composite, null by design

    if args.calsim_arcs:
        cs = build_calsim(args.data_dir)
        assert len(cs) == 196 and cs["entity_id"].is_unique, len(cs)
        assert not set(cs["entity_id"]) & set(df["entity_id"])
        n_default = int(cs["flags"].str.split(";").map(lambda f: "train_default" in f).sum())
        assert n_default == 64, n_default
        df = pd.concat([df, cs], ignore_index=True)
        print(f"calsim_monthly: {len(cs)} arcs appended; {n_default} train_default; "
              f"n_obs {int(cs['n_obs'].sum())} arc-months "
              f"({int((cs['n_obs'] > 0).sum())} arcs with an own record)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} entities")
    print(counts.to_string())
    print("n_obs by family:", dict(by_fam))
    print(f"gates ok: CLE {cle} mi^2; outlets missing: {missing} (by design)")


if __name__ == "__main__":
    main()
