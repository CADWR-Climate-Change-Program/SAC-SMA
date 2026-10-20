"""The replacement record of usgs_11218500 (Kings River below North Fork near Trimmer):
USGS 11218501, the river plus the Kings River powerhouse near Trimmer, which carries part of
the river past the gauge.  The combined record is the flow the watershed makes; the river-only
gauge in ``flow_daily.nc`` misses the powerhouse share (annual river/combined 0.72 to 0.88).

Fetches NWIS daily mean discharge (parameter ``00060``, statistic ``00003``) over
1962-01-01..1993-12-31 (the record runs to 1993-09-30, every value approved) and writes
``usgs_11218501_mm.csv`` (``date,value_mm``): depth over usgs_11218500's area, with the factor
``flow_daily.nc`` uses for that gauge (its ``flow_mm`` over ``flow_cfs``), so the two series
share one area.

    (sacsma env)  python data/targets/usgs/usgs_11218501.py            # fetch and write
    (sacsma env)  python data/targets/usgs/usgs_11218501.py --verify   # fetch and compare

Checks, on every run: the factor is one constant over the record; the combined record is not
below the river-only one (more than 0.5 cfs on at most 5 days); the annual river/combined ratio
lies in 0.70-0.90.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import DATA, layout  # noqa: E402

OUT = layout.usgs_series(DATA, "11218501")
NWIS = "https://waterservices.usgs.gov/nwis/dv/"
SPAN = ("1962-01-01", "1993-12-31")


def fetch() -> pd.Series:
    """NWIS daily mean discharge of 11218501, cfs."""
    import pip._vendor.truststore as truststore  # the system's certificates (a proxy's CA)
    truststore.inject_into_ssl()
    import requests

    r = requests.get(NWIS, timeout=120, params={
        "format": "rdb", "sites": "11218501", "parameterCd": "00060", "statCd": "00003",
        "startDT": SPAN[0], "endDT": SPAN[1]})
    r.raise_for_status()
    rdb = pd.read_csv(io.StringIO(r.text), sep="\t", comment="#", dtype=str).iloc[1:]
    col = [c for c in rdb.columns if c.endswith("_00060_00003")]
    assert len(col) == 1, rdb.columns.tolist()
    q = pd.to_numeric(rdb[col[0]], errors="coerce")
    q.index = pd.DatetimeIndex(rdb["datetime"])
    return q


def factor() -> tuple[float, pd.Series]:
    """mm/day per cfs of usgs_11218500 in ``flow_daily.nc``, and its river-only cfs record."""
    ds = xr.open_dataset(layout.usgs_flow(DATA))
    g = [str(x).zfill(8) for x in ds["gauge"].values].index("11218500")
    cfs = pd.Series(ds["flow_cfs"].values[g].astype(float),
                    index=pd.DatetimeIndex(ds["time"].values))
    mm = pd.Series(ds["flow_mm"].values[g].astype(float), index=cfs.index)
    ok = (cfs > 0) & mm.notna()
    f = mm[ok] / cfs[ok]
    assert f.std() / f.mean() < 1e-6, f.describe()
    return float(f.mean()), cfs


def build() -> pd.Series:
    q = fetch()
    fac, riv = factor()
    j = pd.concat([q.rename("comb"), riv.rename("riv")], axis=1, sort=True).dropna()
    j = j[j.riv > 0]
    below = int((j.comb < j.riv - 0.5).sum())
    a = j.groupby(j.index.year + (j.index.month >= 10))[["comb", "riv"]].sum()
    ratio = a.riv / a.comb
    assert below <= 5 and ratio.between(0.70, 0.90).all(), (below, ratio.describe())
    out = (q * fac).rename("value_mm").dropna()
    out.index.name = "date"
    print(f"11218501: {out.index.min().date()}..{out.index.max().date()}, {len(out)} days; "
          f"{fac:.8e} mm/day per cfs; {len(j)} days in common with the river record, combined "
          f"below it on {below}; annual river/combined {ratio.min():.2f}-{ratio.max():.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="usgs_11218500's replacement record, USGS 11218501")
    ap.add_argument("--verify", action="store_true",
                    help="compare a fresh fetch with the tracked file instead of writing it")
    a = ap.parse_args()
    out = build()
    if not a.verify:
        out.to_frame().to_csv(OUT, float_format="%.6f")
        print(f"wrote {OUT.name}")
        return
    old = pd.read_csv(OUT, index_col="date", parse_dates=True)["value_mm"]
    d = (out.round(6) - old).abs()
    assert out.index.equals(old.index), "the dates differ"
    print(f"verify: {len(old)} days, largest difference {d.max():.2e} mm/day")
    assert d.max() < 1e-6


if __name__ == "__main__":
    main()
