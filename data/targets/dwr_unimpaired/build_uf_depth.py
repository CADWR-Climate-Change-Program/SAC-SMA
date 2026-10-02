"""Build ``uf_monthly_mm.csv``: the DWR unimpaired flows as depth.

``uf_monthly_mm.csv`` (``uf, date, depth_mm``) is ``uf_monthly.csv`` (TAF) converted to
mm/month for the 18 arc-mapped UF subbasins at the CalSim arc-sum areas
(``uf_locations.area_mi2_calsim``; the depth basis stays swappable by rerunning with the
Appendix A SWAT areas).  Full published record (WY1922-2014, month-end stamps).  The table is
keyed by the source's own ids, no entity vocabulary: consumers select their sites and windows
(the training registry's ``obs_store`` column points here; windows live in the registry).

The companion for the CDEC daily store is ``data/targets/cdec/build_fnf_depth.py``, which
reads this table for one of its checks: run this script first.

Usage (sacsma conda env):
    python data/targets/dwr_unimpaired/build_uf_depth.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import DATA, layout  # noqa: E402

SQMI_PER_KM2 = 0.386102
TAF_TO_MM_PER_KM2 = 1233.48184  # mm = TAF x this / area_km2


def build_uf(data_dir: Path) -> pd.DataFrame:
    uf = pd.read_csv(layout.dwr_unimpaired(data_dir, "uf_locations.csv"))
    ufm = pd.read_csv(layout.dwr_unimpaired(data_dir, "uf_monthly.csv"),
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


def gates(ufmm: pd.DataFrame) -> None:
    # every arc-mapped subbasin, full record, WY1985-2014 complete
    assert ufmm["uf"].nunique() == 18
    assert not ufmm.duplicated(["uf", "date"]).any()
    assert ufmm["depth_mm"].notna().all() and (ufmm["depth_mm"] >= 0).all()
    d = pd.to_datetime(ufmm["date"])
    inwin = ufmm[(d >= "1984-10-01") & (d <= "2014-09-30")]
    per = inwin.groupby("uf").size()
    assert (per == 360).all(), per[per != 360].to_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=DATA, type=Path)
    args = ap.parse_args()

    ufmm = build_uf(args.data_dir)
    gates(ufmm)
    out = layout.dwr_unimpaired(args.data_dir, "uf_monthly_mm.csv")
    ufmm.to_csv(out, index=False)
    print(f"wrote {out}: {len(ufmm)} rows, {ufmm['uf'].nunique()} UFs")


if __name__ == "__main__":
    main()
