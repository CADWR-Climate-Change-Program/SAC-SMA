"""Build ``fnf_daily_mm.csv``: the CDEC daily full natural flows as depth.

``fnf_daily_mm.csv`` (``station, date, depth_mm``) is ``fnf_daily.csv`` (cfs) converted to
mm/day (USGS formula) for the two stations with a defined depth area: CLE at 692.86 mi^2 (the
I_TRNTY arc area, the crosswalk convention for Trinity) and CSN at the UF 13 arc-sum area (same
four arcs as its delineation).  Negative-flow days are dropped (computation artifacts, see the
README beside this script).  The table is keyed by the source's own ids, no entity vocabulary:
consumers select their sites and windows (the training registry's ``obs_store`` column points
here; windows live in the registry).

Two checks read other stores: CLE against the 11obs TNL monthly target, and CSN against UF 13
of ``uf_monthly_mm.csv`` (built by ``data/targets/dwr_unimpaired/build_uf_depth.py``: run that
script first).

Usage (sacsma conda env):
    python data/targets/cdec/build_fnf_depth.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import DATA, layout  # noqa: E402

SQMI_PER_KM2 = 0.386102
CFS_TO_MM_PER_KM2 = 2.4465755   # mm/day = cfs x this / area_km2 (USGS)

#: The daily-FNF depth conversions: station -> depth area (mi^2).
#: An area here is a stated convention, not a lookup - only stations with a
#: defined depth basis are converted (fnf_daily.csv also carries SNS and WHI,
#: which have no defined depth area and are not converted; a station added
#: here comes with its own depth area).  ``None`` = read the UF 13 arc-sum from
#: uf_locations.csv at build time (CSN shares that delineation exactly).
FNF_DEPTH_AREAS: dict[str, float | None] = {
    # CLE FNF is computed at Trinity Dam; the crosswalk maps it to the
    # I_TRNTY arc (692.86 mi^2), which is also its training footprint.
    "CLE": 692.86,
    # CSN sits at Michigan Bar; its delineation IS UF 13's four arcs.
    "CSN": None,
}


def _csn_area(data_dir: Path) -> float:
    uf = pd.read_csv(layout.dwr_unimpaired(data_dir, "uf_locations.csv"))
    return float(uf.loc[uf["uf"] == 13, "area_mi2_calsim"].iloc[0])


def build_fnf(data_dir: Path) -> pd.DataFrame:
    csn_area = _csn_area(data_dir)
    assert abs(csn_area - 539.1) < 0.1, csn_area
    fnf = pd.read_csv(layout.cdec_fnf(data_dir, "fnf_daily.csv"),
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


def gates(fnfmm: pd.DataFrame, data_dir: Path) -> None:
    obs11 = pd.read_csv(layout.callite_fnf(data_dir, "11obs"), parse_dates=["date"])
    ufmm = pd.read_csv(layout.dwr_unimpaired(data_dir, "uf_monthly_mm.csv"))

    # the two stations, no negatives, no duplicates
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
    csn_area = _csn_area(data_dir)
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
    ap.add_argument("--data-dir", default=DATA, type=Path)
    args = ap.parse_args()

    fnfmm = build_fnf(args.data_dir)
    gates(fnfmm, args.data_dir)
    out = layout.cdec_fnf(args.data_dir, "fnf_daily_mm.csv")
    fnfmm.to_csv(out, index=False)
    print(f"wrote {out}: {len(fnfmm)} rows, {fnfmm['station'].nunique()} stations")


if __name__ == "__main__":
    main()
