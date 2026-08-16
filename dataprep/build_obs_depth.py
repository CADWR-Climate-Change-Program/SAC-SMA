"""Build the derived depth-series companions of the two non-depth stores.

Each table lives beside its raw source, keyed by the source's own ids —
no entity vocabulary:

* ``data/dwr_unimpaired/uf_monthly_mm.csv`` (``uf, date, depth_mm``) —
  ``uf_monthly.csv`` TAF volumes converted to mm/month for the 18
  arc-mapped UF subbasins at the CalSim arc-sum areas
  (``uf_locations.area_mi2_calsim``; the depth basis stays swappable by
  rerunning with the Appendix A SWAT areas).  Full published record
  (WY1922-2014, month-end stamps).
* ``data/cdec_fnf/fnf_daily_mm.csv`` (``station, date, depth_mm``) —
  ``fnf_daily.csv`` cfs converted to mm/day (USGS formula) for the two
  approved daily targets: CLE at 692.86 mi^2 (the I_TRNTY arc area, the
  crosswalk convention for Trinity) and CSN at the UF 13 arc-sum area
  (same four arcs as its delineation).  Negative-flow days are dropped
  (computation artifacts per the store README).

Consumers select their sites and windows (the training registry's
``obs_store`` column points here; windows live in the registry).

Usage (sacsma conda env, from the repo root):
    python dataprep/build_obs_depth.py [--data-dir data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SQMI_PER_KM2 = 0.386102
CFS_TO_MM_PER_KM2 = 2.4465755   # mm/day = cfs x this / area_km2 (USGS)
TAF_TO_MM_PER_KM2 = 1233.48184  # mm = TAF x this / area_km2

#: The APPROVED daily-FNF depth conversions: station -> depth area (mi^2).
#: An area here is a decided convention, not a lookup - only stations whose
#: depth basis has been ruled on are converted (fnf_daily.csv also carries
#: the SNS and WHI candidates; they join this table if/when approved, with
#: their own area rulings).  ``None`` = read the UF 13 arc-sum from
#: uf_locations.csv at build time (CSN shares that delineation exactly).
FNF_DEPTH_AREAS: dict[str, float | None] = {
    # CLE FNF is computed at Trinity Dam; the crosswalk maps it to the
    # I_TRNTY arc (692.86 mi^2), which is also its training footprint.
    "CLE": 692.86,
    # CSN sits at Michigan Bar; its delineation IS UF 13's four arcs.
    "CSN": None,
}


def build_uf(data_dir: Path) -> pd.DataFrame:
    uf = pd.read_csv(data_dir / "dwr_unimpaired" / "uf_locations.csv")
    ufm = pd.read_csv(data_dir / "dwr_unimpaired" / "uf_monthly.csv",
                      parse_dates=["date"])
    frames = []
    for r in uf[uf["n_arcs"] > 0].itertuples():
        area_km2 = float(r.area_mi2_calsim) / SQMI_PER_KM2
        g = (ufm[(ufm["uf"] == int(r.uf)) & ufm["flow_taf"].notna()]
             .sort_values("date"))
        frames.append(pd.DataFrame({
            "uf": int(r.uf), "date": g["date"],
            "depth_mm": (g["flow_taf"] * TAF_TO_MM_PER_KM2
                         / area_km2).round(4)}))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["date"].dt.date.astype(str)
    return df


def build_fnf(data_dir: Path) -> pd.DataFrame:
    uf = pd.read_csv(data_dir / "dwr_unimpaired" / "uf_locations.csv")
    csn_area = float(uf.loc[uf["uf"] == 13, "area_mi2_calsim"].iloc[0])
    assert abs(csn_area - 539.1) < 0.1, csn_area
    fnf = pd.read_csv(data_dir / "cdec_fnf" / "fnf_daily.csv",
                      parse_dates=["date"])
    frames = []
    for st, area_mi2 in FNF_DEPTH_AREAS.items():
        area_km2 = (area_mi2 if area_mi2 is not None else csn_area) \
            / SQMI_PER_KM2
        g = (fnf[(fnf["station"] == st) & fnf["flow_cfs"].notna()
                 & (fnf["flow_cfs"] >= 0)].sort_values("date"))
        frames.append(pd.DataFrame({
            "station": st, "date": g["date"],
            "depth_mm": (g["flow_cfs"] * CFS_TO_MM_PER_KM2
                         / area_km2).round(6)}))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["date"].dt.date.astype(str)
    return df


def gates(ufmm: pd.DataFrame, fnfmm: pd.DataFrame, data_dir: Path) -> None:
    uf = pd.read_csv(data_dir / "dwr_unimpaired" / "uf_locations.csv")
    obs11 = pd.read_csv(data_dir / "calsim" / "fnf_11obs_monthly.csv",
                        parse_dates=["date"])

    # UF: every arc-mapped subbasin, full record, WY1985-2014 complete
    assert ufmm["uf"].nunique() == 18
    assert not ufmm.duplicated(["uf", "date"]).any()
    assert ufmm["depth_mm"].notna().all() and (ufmm["depth_mm"] >= 0).all()
    d = pd.to_datetime(ufmm["date"])
    inwin = ufmm[(d >= "1984-10-01") & (d <= "2014-09-30")]
    per = inwin.groupby("uf").size()
    assert (per == 360).all(), per[per != 360].to_dict()

    # FNF: the two stations, no negatives, no duplicates
    assert set(fnfmm["station"]) == {"CLE", "CSN"}
    assert not fnfmm.duplicated(["station", "date"]).any()
    assert (fnfmm["depth_mm"] >= 0).all()

    def monthly_volume(g, area_mi2):
        gg = g.assign(dt=pd.to_datetime(g["date"]))
        gg["m"] = gg["dt"].dt.to_period("M")
        agg = gg.groupby("m").agg(mm=("depth_mm", "sum"),
                                  n=("depth_mm", "size"))
        agg = agg[agg["n"] >= agg.index.days_in_month]
        return agg["mm"] * area_mi2

    # CLE volume vs the 11obs TNL reference (Trinity Dam vs Lewiston
    # footprint difference -> a few % low; store-verified ~0.987 / r 0.998)
    cle = monthly_volume(fnfmm[fnfmm["station"] == "CLE"],
                         FNF_DEPTH_AREAS["CLE"])
    tnl = obs11[obs11["basin"] == "TNL"].dropna(subset=["obs_mm"])
    tnl_v = pd.Series((tnl["obs_mm"] * 719.0).to_numpy(),
                      index=tnl["date"].dt.to_period("M"))
    j = pd.concat([cle, tnl_v], axis=1, join="inner").dropna()
    r = np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
    ratio = j.iloc[:, 0].sum() / j.iloc[:, 1].sum()
    print(f"CLE vs 11obs TNL: n={len(j)} months, r={r:.4f}, ratio={ratio:.3f}")
    assert r > 0.995 and 0.96 < ratio < 1.01, (r, ratio)

    # CSN volume vs the UF 13 monthly series (same delineation; the daily
    # product's documented ~6% low bias)
    csn_area = float(uf.loc[uf["uf"] == 13, "area_mi2_calsim"].iloc[0])
    csn = monthly_volume(fnfmm[fnfmm["station"] == "CSN"], csn_area)
    u13 = ufmm[ufmm["uf"] == 13]
    u13_v = pd.Series((u13["depth_mm"] * csn_area).to_numpy(),
                      index=pd.to_datetime(u13["date"]).dt.to_period("M"))
    j = pd.concat([csn, u13_v], axis=1, join="inner").dropna()
    r = np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
    ratio = j.iloc[:, 0].sum() / j.iloc[:, 1].sum()
    print(f"CSN vs UF 13:     n={len(j)} months, r={r:.4f}, ratio={ratio:.3f}")
    assert r > 0.99 and 0.84 < ratio < 0.99, (r, ratio)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=Path("data"), type=Path)
    args = ap.parse_args()

    ufmm = build_uf(args.data_dir)
    fnfmm = build_fnf(args.data_dir)
    gates(ufmm, fnfmm, args.data_dir)

    p1 = args.data_dir / "dwr_unimpaired" / "uf_monthly_mm.csv"
    p2 = args.data_dir / "cdec_fnf" / "fnf_daily_mm.csv"
    ufmm.to_csv(p1, index=False)
    fnfmm.to_csv(p2, index=False)
    print(f"wrote {p1}: {len(ufmm)} rows, {ufmm['uf'].nunique()} UFs")
    print(f"wrote {p2}: {len(fnfmm)} rows, "
          f"{fnfmm['station'].nunique()} stations")


if __name__ == "__main__":
    main()
