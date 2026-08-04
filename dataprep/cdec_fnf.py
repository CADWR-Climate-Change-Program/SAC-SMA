"""CDEC daily full natural flow (sensor 8, daily duration).

Step 1: survey every CDEC station carrying daily FNF -> stations.csv.
Step 2: download the usable ones -> fnf_daily.csv (cfs, exactly as CDEC
serves them — negative days included; mask flow_cfs < 0 before use).
Method, classification, and verification results: data/cdec_fnf/README.md.

Usage:
    python dataprep/cdec_fnf.py            # survey + pull + verify
    python dataprep/cdec_fnf.py survey     # stations.csv only
    python dataprep/cdec_fnf.py pull       # fnf_daily.csv + verify [--start --end]
    python dataprep/cdec_fnf.py verify     # re-check an existing fnf_daily.csv
"""
from __future__ import annotations

import argparse
import io as _io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sacsma.io import cfs_to_mmday, load_basin_area, read_table, write_table  # noqa: E402

SERVLET = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
STA_META = "https://cdec.water.ca.gov/dynamicapp/staMeta"
FNF_SEARCH = ("https://cdec.water.ca.gov/dynamicapp/staSearch?"
              "sensor_chk=on&sensor=8&dur_chk=on&dur=D&display=sta")

#: stations downloaded here -> (verify basin, its basin_area domain), or
#: (None, None) where the repo has no monthly target to check against.
#: Why these, and what the rest of the universe is: data/cdec_fnf/README.md
PULL = {
    "CLE": ("TNL", "11obs"),
    "CSN": ("CosumnesRiver", "9unimp"),
    "SNS": ("SNS", "11obs"),
    "WHI": (None, None),
}

START = "1986-01-01"   # before the earliest pulled record (CLE 1986-04-01)
END = "2018-12-31"     # Livneh forcing end = training hard stop

#: per-station record start. WHI's published record begins 2000-10; the
#: servlet also returns an unpublished Jan-Sep 1990 fragment — drop it so the
#: store holds published records only.
RECORD_START = {"WHI": "2000-10-01"}


def fnf_universe() -> dict[str, str]:
    """All stations carrying daily FNF, from CDEC's station search: {id: name}."""
    resp = requests.get(FNF_SEARCH, timeout=120)
    resp.raise_for_status()
    rows = re.findall(
        r"station_id=([A-Z0-9]{3})'>[A-Z0-9]{3}</a></td>\s*<td[^>]*>([^<]+)</td>",
        resp.text)
    uni = {sid: name.strip() for sid, name in rows}
    if not 20 <= len(uni) <= 60:
        raise ValueError(f"station search parse looks wrong: {len(uni)} stations")
    return uni


def station_meta(sid: str) -> dict:
    """lat, lon and the daily-FNF period of record from the staMeta page.

    A station often lists SEVERAL daily-FNF rows — an agency data-exchange
    record plus CDEC's own computed one (usually from 2013-10) — so span every
    matching row, or the advertised record looks decades shorter than it is.
    """
    resp = requests.get(STA_META, params={"station_id": sid}, timeout=120)
    resp.raise_for_status()
    text = re.sub(r"<[^>]+>", "|", resp.text)
    lat = float(re.search(r"Latitude\|+([0-9.\-]+)", text).group(1))
    lon = float(re.search(r"Longitude\|+([0-9.\-]+)", text).group(1))
    starts, ends = [], []
    for row in re.findall(r"<tr>(.*?)</tr>", resp.text, re.S):
        cells = [re.sub(r"<[^>]+>", " ", c).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if (len(cells) >= 6 and cells[0].upper().startswith("FULL NATURAL FLOW")
                and "daily" in cells[2]):
            m = re.search(r"([\d/]+)\s+to\s+(present|[\d/]+)", cells[-1])
            if m:
                starts.append(pd.to_datetime(m.group(1)))
                ends.append(m.group(2))
    if "present" in ends:
        avail_to = "present"
    else:
        avail_to = max(pd.to_datetime(e) for e in ends).date().isoformat() if ends else ""
    return {"lat": lat, "lon": lon,
            "available_from": min(starts).date().isoformat() if starts else "",
            "available_to": avail_to}


def build_stations(uni: dict[str, str]) -> pd.DataFrame:
    rows = []
    for sid, name in sorted(uni.items()):
        note = ("data in fnf_daily.csv; mask flow_cfs < 0 before use"
                if sid in PULL else "")
        rows.append({"id": sid, "name": name, "sensor": 8,
                     "sensor_type": "FULL NATURAL FLOW", "duration": "daily",
                     **station_meta(sid), "note": note})
        print(f"{sid}  {name}", flush=True)
    return pd.DataFrame(rows)


def fetch_daily_fnf(start: str, end: str) -> pd.DataFrame:
    """One servlet call for all PULL stations -> long table (verbatim cfs)."""
    resp = requests.get(SERVLET, params={
        "Stations": ",".join(sorted(PULL)), "SensorNums": "8", "dur_code": "D",
        "Start": start, "End": end}, timeout=300)
    resp.raise_for_status()
    raw = pd.read_csv(_io.StringIO(resp.text))
    raw.columns = [c.strip().upper().replace(" ", "_") for c in raw.columns]
    need = {"STATION_ID", "DATE_TIME", "VALUE", "UNITS"}
    if not need <= set(raw.columns):
        raise ValueError(f"servlet response is missing {sorted(need - set(raw.columns))}"
                         f" (returned {list(raw.columns)})")
    units = set(raw["UNITS"].dropna().str.upper())
    if units - {"CFS"}:
        raise ValueError(f"expected CFS only, servlet returned {units}")
    return pd.DataFrame({
        "station": raw["STATION_ID"].str.strip(),
        "date": pd.to_datetime(raw["DATE_TIME"], format="%Y%m%d %H%M"),
        "flow_cfs": pd.to_numeric(raw["VALUE"], errors="coerce"),  # '---' -> NaN
    })


def build_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Trim each station to its finite record; keep values verbatim."""
    frames = []
    for sta in sorted(PULL):
        g = raw[raw["station"] == sta]
        if sta in RECORD_START:
            g = g[g["date"] >= pd.Timestamp(RECORD_START[sta])]
        finite = np.flatnonzero(g["flow_cfs"].notna())
        if not len(finite):
            raise ValueError(f"{sta} returned no finite values")
        g = g.iloc[finite[0]:finite[-1] + 1]
        print(f"{sta}: {g['date'].min():%Y-%m-%d} .. {g['date'].max():%Y-%m-%d}  "
              f"{len(g)} days, {int(g['flow_cfs'].isna().sum())} missing, "
              f"{int((g['flow_cfs'] < 0).sum())} negative (kept)", flush=True)
        frames.append(g)
    return (pd.concat(frames).sort_values(["station", "date"])
            .reset_index(drop=True))


def verify(gage: pd.DataFrame, data_dir: str = "data") -> None:
    """Monthly sums vs the repo's monthly calibration targets, where one exists."""
    for sta, (basin, domain) in sorted(PULL.items()):
        if basin is None:
            print(f"verify {sta}: no independent monthly target — skipped", flush=True)
            continue
        area = load_basin_area(data_dir, domain=domain).set_index("basin")
        area_mi2 = float(area.loc[basin, "area_mi2"])
        tgt = read_table(Path(data_dir) / "calsim" / f"calib_{domain}_monthly.csv")
        cfs = gage[gage["station"] == sta].set_index("date")["flow_cfs"]
        mm = cfs_to_mmday(cfs.where(cfs >= 0.0), area_mi2)
        by_month = mm.groupby(pd.Grouper(freq="ME"))
        monthly = by_month.sum()
        monthly = monthly[by_month.count() == monthly.index.days_in_month]
        t = tgt[tgt["basin"] == basin].set_index("date")["obs_mm"]
        both = (monthly.rename("cdec").to_frame()
                .join(t.rename("target"), how="inner").dropna())
        # element-wise Pearson r (pandas .corr trips the env's MKL crash)
        xm = both["cdec"] - both["cdec"].mean()
        ym = both["target"] - both["target"].mean()
        r = float((xm * ym).sum() / np.sqrt((xm ** 2).sum() * (ym ** 2).sum()))
        print(f"verify {sta}/{basin}: {len(both)} complete months  r={r:.4f}  "
              f"mean-ratio={both['cdec'].mean() / both['target'].mean():.4f}  "
              f"(area {area_mi2:.2f} mi2)", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="data/cdec_fnf")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("survey", help="write stations.csv only")
    pull = sub.add_parser("pull", help="download fnf_daily.csv and verify")
    pull.add_argument("--start", default=START)
    pull.add_argument("--end", default=END)
    sub.add_parser("verify", help="re-check an existing fnf_daily.csv")
    args = ap.parse_args(argv)
    out = Path(args.dir)

    if args.cmd in (None, "survey"):
        stations = build_stations(fnf_universe())
        path = write_table(stations, str(out / "stations.csv"))
        print(f"wrote {path} ({len(stations)} stations)")
    if args.cmd in (None, "pull"):
        gage = build_table(fetch_daily_fnf(getattr(args, "start", START),
                                           getattr(args, "end", END)))
        path = write_table(gage, str(out / "fnf_daily.csv"))
        print(f"wrote {path} ({len(gage)} rows)")
        verify(gage)
    if args.cmd == "verify":
        verify(read_table(str(out / "fnf_daily.csv")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
