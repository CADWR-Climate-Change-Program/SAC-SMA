"""CDEC full natural flow: daily (sensor 8) and monthly (sensor 65).

Step 1: survey every CDEC station carrying daily FNF -> stations.csv.
Step 2: download the usable ones -> fnf_daily.csv (cfs, exactly as CDEC
serves them — negative days included; mask flow_cfs < 0 before use).
Step 3: download the monthly FNF of the 17 benchmark sites -> fnf_monthly.csv
(acre-feet per month and CDEC's data flag, exactly as CDEC serves them), and
check it against the summed daily records of the repository.
Method, classification, and verification results: data/targets/cdec/README.md.

Usage:
    python data/targets/cdec/cdec_fnf.py                 # survey + pull + verify + monthly
    python data/targets/cdec/cdec_fnf.py survey          # stations.csv only
    python data/targets/cdec/cdec_fnf.py pull            # fnf_daily.csv + verify [--start --end]
    python data/targets/cdec/cdec_fnf.py verify          # re-check an existing fnf_daily.csv
    python data/targets/cdec/cdec_fnf.py monthly         # fnf_monthly.csv + check [--start --end]
    python data/targets/cdec/cdec_fnf.py verify-monthly  # re-check an existing fnf_monthly.csv
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

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import DATA, layout  # noqa: E402
from sacsma.io import (cfs_to_mmday, load_basin_area, mmday_to_cfs, read_table,  # noqa: E402
                       write_table)

SERVLET = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
STA_META = "https://cdec.water.ca.gov/dynamicapp/staMeta"
FNF_SEARCH = ("https://cdec.water.ca.gov/dynamicapp/staSearch?"
              "sensor_chk=on&sensor=8&dur_chk=on&dur=D&display=sta")

#: stations downloaded here -> (verify basin, its basin_area domain), or
#: (None, None) where the repo has no monthly target to check against.
#: Why these, and what the rest of the universe is: data/targets/cdec/README.md
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

#: the monthly FNF (sensor 65, acre-feet per month) of the 17 benchmark sites
#: (``sacsma.benchmark.flows.SITES``): site -> the CDEC station that carries it, None where
#: CDEC has none. CLE and NML have no monthly series; the Bulletin 120 forecast point of
#: their river, a few miles downstream, carries it (TNL, the Trinity at Lewiston; SNS, the
#: Stanislaus at Goodwin Dam). No Calaveras River station carries sensor 65 (NHG).
MONTHLY = {
    "SHA": "SIS", "CLE": "TNL", "BND": "SBB", "ORO": "FTO", "FOL": "AMF", "YRS": "YRS",
    "CSN": "CSN", "MKM": "MKM", "NML": "SNS", "NHG": None, "TLG": "TLG", "MRC": "MRC",
    "MIL": "SBF", "PNF": "KGF", "TRM": "KWT", "SCC": "SCC", "ISB": "KRI",
}
MONTHLY_START = "1921-10-01"   # WY1922; a station whose record starts later starts there
MONTHLY_END = "2025-09-30"     # WY2025
#: water years of the check against the summed daily records (the benchmark's window)
CHECK_WY = (1991, 2018)
#: acre-feet in one cfs sustained for a day (86,400 ft3 / 43,560 ft3)
AF_PER_CFS_DAY = 86400.0 / 43560.0
#: acre-feet in 1 mm over 1 mi2 (2,589,988.11 m2 x 1e-3 m / 1,233.4818 m3 per AF)
AF_PER_MM_MI2 = 2589988.110336e-3 / 1233.48183754752
#: the seasons of the check's volume ratios, by calendar month
SEASONS = {"DJFM": (12, 1, 2, 3), "AMJJ": (4, 5, 6, 7), "ASON": (8, 9, 10, 11)}


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


def verify(gage: pd.DataFrame, data_dir: str | Path = DATA) -> None:
    """Monthly sums vs the repo's monthly calibration targets, where one exists."""
    for sta, (basin, domain) in sorted(PULL.items()):
        if basin is None:
            print(f"verify {sta}: no independent monthly target — skipped", flush=True)
            continue
        area = load_basin_area(data_dir, domain=domain).set_index("basin")
        area_mi2 = float(area.loc[basin, "area_mi2"])
        tgt = read_table(layout.callite_calib(data_dir, domain))
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


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Element-wise Pearson r (pandas .corr trips the env's MKL crash)."""
    xm, ym = x - x.mean(), y - y.mean()
    return float((xm * ym).sum() / np.sqrt((xm ** 2).sum() * (ym ** 2).sum()))


def fetch_monthly_fnf(start: str = MONTHLY_START, end: str = MONTHLY_END) -> pd.DataFrame:
    """One servlet call for the MONTHLY stations -> long table (verbatim AF and flag)."""
    stations = sorted({s for s in MONTHLY.values() if s})
    resp = requests.get(SERVLET, params={
        "Stations": ",".join(stations), "SensorNums": "65", "dur_code": "M",
        "Start": start, "End": end}, timeout=(30, 300))
    resp.raise_for_status()
    raw = pd.read_csv(_io.StringIO(resp.text), dtype=str, keep_default_na=False)
    raw.columns = [c.strip().upper().replace(" ", "_") for c in raw.columns]
    need = {"STATION_ID", "SENSOR_NUMBER", "DURATION", "DATE_TIME", "VALUE", "DATA_FLAG",
            "UNITS"}
    if not need <= set(raw.columns):
        raise ValueError(f"servlet response is missing {sorted(need - set(raw.columns))}"
                         f" (returned {list(raw.columns)})")
    raw = raw.apply(lambda c: c.str.strip())
    for col, want in (("UNITS", {"AF"}), ("SENSOR_NUMBER", {"65"}), ("DURATION", {"M"})):
        if set(raw[col]) != want:
            raise ValueError(f"expected {col} {want}, servlet returned {set(raw[col])}")
    absent = sorted(set(stations) - set(raw["STATION_ID"]))
    if absent:
        raise ValueError(f"no monthly FNF rows for {absent}")
    # DATE TIME is the first of the month (OBS DATE is not consistent: month end early in
    # the record, month start later); the table stamps the month end
    date = pd.to_datetime(raw["DATE_TIME"], format="%Y%m%d %H%M")
    if (date.dt.day != 1).any():
        raise ValueError("expected DATE TIME on the first of each month")
    return pd.DataFrame({
        "date": date + pd.offsets.MonthEnd(0),
        "station": raw["STATION_ID"],
        "flow_af": pd.to_numeric(raw["VALUE"], errors="coerce"),  # '---' -> NaN
        "flag": raw["DATA_FLAG"],
    })


def build_monthly_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Each station from its first to its last value, one row per month (a missing value
    stays empty); values and flags verbatim (whole acre-feet stay integers)."""
    frames = []
    for sta in sorted(set(raw["station"])):
        g = raw[raw["station"] == sta].drop_duplicates()
        if g["date"].duplicated().any():
            raise ValueError(f"{sta}: conflicting rows for one month")
        g = g.set_index("date").sort_index()
        finite = g.index[g["flow_af"].notna()]
        if not len(finite):
            raise ValueError(f"{sta} returned no finite values")
        g = g.reindex(pd.date_range(finite[0], finite[-1], freq="ME", name="date"))
        g["station"] = sta
        g["flag"] = g["flag"].fillna("")
        flags = g["flag"].replace("", "none").value_counts().to_dict()
        print(f"{sta}: {g.index.min():%Y-%m} .. {g.index.max():%Y-%m}  {len(g)} months, "
              f"{int(g['flow_af'].isna().sum())} missing, "
              f"{int((g['flow_af'] < 0).sum())} negative, flags {flags}", flush=True)
        frames.append(g.reset_index())
    out = pd.concat(frames, ignore_index=True)[["date", "station", "flow_af", "flag"]]
    if (out["flow_af"].dropna() % 1 == 0).all():
        out["flow_af"] = out["flow_af"].astype("Int64")
    return out


def daily_af(data_dir: str | Path = DATA) -> pd.DataFrame:
    """The repository's daily FNF of the 17 sites as volume (AF/day, date x site), as the
    benchmark reads it: a day counts only when it is finite, not negative and not in
    ``fnf_daily_mask.csv``. The 15 of ``gage_15cdec.csv`` go back to cfs over the 15cdec
    areas they were made with; CLE and CSN come from ``fnf_daily.csv``."""
    area = load_basin_area(data_dir, domain=layout.CDEC15).set_index("basin")["area_mi2"]
    gage = read_table(layout.cdec_fnf(data_dir, "gage_15cdec.csv"))
    wide = gage[gage["basin"].isin(list(MONTHLY))].pivot(index="date", columns="basin",
                                                          values="flow")
    cfs = mmday_to_cfs(wide, area.reindex(wide.columns).to_numpy())
    fnf = read_table(layout.cdec_fnf(data_dir, "fnf_daily.csv"))
    rest = [s for s in MONTHLY if s not in cfs.columns]
    fnf = fnf[fnf["station"].isin(rest)].pivot(index="date", columns="station",
                                               values="flow_cfs")
    cfs = cfs.join(fnf, how="outer")
    absent = sorted(set(MONTHLY) - set(cfs.columns))
    if absent:
        raise ValueError(f"no daily FNF for {absent}")
    cfs = cfs.where(cfs >= 0.0)
    mask = read_table(layout.cdec_fnf(data_dir, "fnf_daily_mask.csv"))
    for eid, day in zip(mask["entity_id"], mask["date"], strict=True):
        site = eid.removeprefix("cdec_")
        if site in cfs.columns and day in cfs.index:
            cfs.loc[day, site] = np.nan
    return cfs[list(MONTHLY)] * AF_PER_CFS_DAY


def verify_monthly(monthly: pd.DataFrame, data_dir: str | Path = DATA,
                   wy: tuple[int, int] = CHECK_WY) -> pd.DataFrame:
    """Each site's monthly FNF against the sum of its daily FNF over water years ``wy``, on
    the months both records have; a daily month counts only when every day is valid."""
    months = pd.date_range(f"{wy[0] - 1}-10-31", f"{wy[1]}-09-30", freq="ME", name="date")
    by = daily_af(data_dir).groupby(pd.Grouper(freq="ME"))
    count = by.count()
    full = count.eq(pd.Series(count.index.days_in_month, index=count.index), axis=0)
    dsum = by.sum().where(full).reindex(months)
    mon = (monthly.pivot(index="date", columns="station", values="flow_af")
           .astype(float).reindex(months))
    rows = []
    for site, sta in MONTHLY.items():
        m = mon[sta] if sta in mon.columns else pd.Series(np.nan, index=months)
        d = dsum[site]
        both = (m.notna() & d.notna()).to_numpy()
        x, y = m.to_numpy()[both], d.to_numpy()[both]
        row = {"site": site, "monthly_station": sta or "", "monthly_months": int(m.notna().sum()),
               "daily_months": int(d.notna().sum()), "both": int(both.sum()),
               "r": _pearson(x, y) if len(x) > 2 else np.nan,
               "ratio": x.sum() / y.sum() if len(x) else np.nan}
        for season, cal in SEASONS.items():
            s = np.isin(months.month[both], cal)
            row[f"ratio_{season}"] = x[s].sum() / y[s].sum() if s.any() else np.nan
        rows.append(row)
        print(f"verify-monthly {site:<3} ({sta or '-':<3}) WY{wy[0]}-{wy[1]}: "
              f"monthly {row['monthly_months']}, complete daily {row['daily_months']}, "
              f"both {row['both']}  r={row['r']:.4f}  volume ratio={row['ratio']:.4f}  ("
              + ", ".join(f"{s} {row[f'ratio_{s}']:.3f}" for s in SEASONS) + ")", flush=True)
    return pd.DataFrame(rows)


def verify_against_11obs(monthly: pd.DataFrame, data_dir: str | Path = DATA) -> None:
    """Each station that is also an ``11obs`` basin against that basin's full-period target
    (``fnf_11obs_monthly.csv``, DWR's published record as a depth): the median ratio is the
    area DWR normalised by; the months that depart from it by more than 1 AF are listed."""
    tgt = read_table(layout.callite_fnf(data_dir, "11obs"))
    area = load_basin_area(data_dir, domain="11obs").set_index("basin")["area_mi2"]
    for b in sorted(set(tgt["basin"]) & set(monthly["station"])):
        t = tgt[tgt["basin"] == b].set_index("date")["obs_mm"] * area[b] * AF_PER_MM_MI2
        t.index = t.index + pd.offsets.MonthEnd(0)
        m = monthly[monthly["station"] == b].set_index("date")["flow_af"].astype(float)
        j = pd.concat([m.rename("m"), t.rename("t")], axis=1, join="inner").dropna()
        k = float((j["m"] / j["t"])[j["t"] > 0].median())
        off = j.index[(j["m"] - k * j["t"]).abs() > 1.0]
        print(f"verify-monthly {b} vs 11obs: {len(j)} months {j.index.min():%Y-%m}.."
              f"{j.index.max():%Y-%m}  r={_pearson(j['m'].to_numpy(), j['t'].to_numpy()):.6f}"
              f"  ratio {k:.4f} = area {area[b] * k:.1f} mi2 (11obs {area[b]:.1f});"
              f" {len(off)} months off by > 1 AF {[f'{d:%Y-%m}' for d in off]}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(layout.cdec_fnf(DATA, "")))
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("survey", help="write stations.csv only")
    pull = sub.add_parser("pull", help="download fnf_daily.csv and verify")
    pull.add_argument("--start", default=START)
    pull.add_argument("--end", default=END)
    sub.add_parser("verify", help="re-check an existing fnf_daily.csv")
    mon = sub.add_parser("monthly", help="download fnf_monthly.csv and check it")
    mon.add_argument("--start", default=MONTHLY_START)
    mon.add_argument("--end", default=MONTHLY_END)
    sub.add_parser("verify-monthly", help="re-check an existing fnf_monthly.csv")
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
    if args.cmd in (None, "monthly"):
        monthly = build_monthly_table(fetch_monthly_fnf(getattr(args, "start", MONTHLY_START),
                                                        getattr(args, "end", MONTHLY_END)))
        path = write_table(monthly, str(out / "fnf_monthly.csv"))
        print(f"wrote {path} ({len(monthly)} rows)")
        verify_monthly(monthly)
        verify_against_11obs(monthly)
    if args.cmd == "verify-monthly":
        monthly = read_table(str(out / "fnf_monthly.csv"))
        verify_monthly(monthly)
        verify_against_11obs(monthly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
