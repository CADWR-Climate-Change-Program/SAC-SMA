"""Verify ``data/dwr_unimpaired/uf_locations.csv`` against independent sources.

The 24 UF subbasins of DWR's *Estimates of Natural and Unimpaired Flows for the
Central Valley of California, WY 1922-2014* (2016 draft) have no published GIS
layer — the table's arc sets reconstruct them from CalSim3 catchments.  This
script checks every claim in the table against a source that did not produce it

1. **Set consistency** — each UF's arc list vs ``calsim_crosswalk.csv``, BOTH
   directions (the reverse direction catches Fresno-style omissions).
2. **Area closure** — member-arc ``SQ_MI`` sums (``calsim3.gpkg`` merged layer)
   vs ``area_mi2_calsim``.  Exact by construction; internal typo check only.
3. **Independent area + outlet identity** — USGS NWIS station name, coordinates
   and published drainage area at each outlet gauge, plus CDEC staMeta
   coordinates where a ``cdec_id`` exists.  The key independent check.
4. **Volume closure** — arc-summed ``calsim3_inflow_monthly.csv`` vs the DWR
   unimpaired series (``uf_monthly.csv``) over WY1950-84 (ratio + Pearson r),
   plus the ``UNIMP_<sys>`` anchor where one exists.
5. **Geometry** — one map per UF (member arcs + outlet pins), all UFs dissolved
   onto one overview (tiling: no double-claims, no unexplained gaps — the
   reconstruction of the report's Figure 2-1), and an 18-panel contact sheet.

Plus the Paynes Creek ruling: the NLDI-delineated Bend Bridge watershed
(USGS 11377100) is intersected with I_PYN001 (and control arcs) — 0.4 %%
overlap vs 99.9 %% for a true member proves the creek joins BELOW the gauge,
so I_PYN001 is correctly excluded from UF 6.

Web calls (NWIS site service, CDEC staMeta, NLDI basin) are cached in
``artifacts/uf_check/web_cache.json`` + ``nldi_bend_basin.json``; with the
committed caches the script reruns offline and byte-reproducibly.

Outputs
-------
    data/dwr_unimpaired/uf_outlets.csv    verified UF -> USGS gauge mapping
                                          (site, name, lat/lon, published DA,
                                          CDEC coords, offset km)
    artifacts/uf_check/report_table.csv   per-UF results (areas, volumes, outlets)
    artifacts/uf_check/findings.md        flags + notes
    artifacts/uf_check/figures/uf_NN.png  per-UF map, outlet pinned
    artifacts/uf_check/figures/uf_dissolved_overview.png   Figure-2-1 analogue
    artifacts/uf_check/figures/uf_all_grid.png             18-panel contact sheet
    artifacts/uf_check/uf_dissolved.gpkg  the dissolved UF polygons (QGIS-ready)

Needs geopandas + pyogrio + matplotlib + pillow (any env that can read the
repo's gpkg).  Usage::

    python dataprep/check_uf_locations.py
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.validation import make_valid

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "uf_check"
FIG = OUT / "figures"
WY = (1950, 1984)  # volume-closure window, water years inclusive

SKIP = {1: "valley floor, n_arcs=0", 12: "valley floor, n_arcs=0",
        17: "valley floor, n_arcs=0", 23: "valley floor, n_arcs=0",
        24: "valley floor, n_arcs=0",
        5: "no arc set by design (west-side minor streams)"}

#: (uf, arc) pairs expected to fail the naive lookups, with the reason
EXPECTED = {
    (6, "I_SHSTA"): "SHA nests inside BND (BASIN_NESTS in catchments.py)",
    (6, "I_SRBB_VAL"): "series-less valley-accretion node; no crosswalk row or inflow series by design",
}
#: unassigned arcs adjacent to a UF outlet, highlighted on that UF's map.
#: Both were ruled OUT on 2026-08-04: PYN001 joins below the Bend Bridge gauge
#: (NLDI overlap 0.4 %); PARDE is the Mokelumne-Hill-gauge-to-Pardee-dam
#: increment and DWR's B-14 series matches the gauge footprint (544 mi2) exactly.
ADJACENT = {"I_PYN001": 6, "I_PARDE": 14}

#: NWIS candidates per UF + name keywords (every group must match, any word of
#: a group suffices).  Sites were resolved 2026-08-04 by checking NWIS's own
#: station names; the first fetched candidate passing the keyword test wins.
NWIS_CAND = {
    2:  (["11454000"], [["PUTAH"], ["WINTERS"]]),
    3:  (["11451760", "11451800"], [["CACHE"], ["RUMSEY"]]),
    4:  (["11388000", "11387500"], [["STONY"], ["BLACK BUTTE"]]),
    6:  (["11377100"], [["BEND BRIDGE"]]),
    8:  (["11407000"], [["FEATHER"], ["OROVILLE"]]),
    9:  (["11419000"], [["YUBA"], ["SMARTSVILLE", "SMARTVILLE"]]),
    10: (["11424000"], [["BEAR"], ["WHEATLAND"]]),
    11: (["11446500"], [["AMERICAN"], ["FAIR OAKS"]]),
    13: (["11335000"], [["COSUMNES"], ["MICHIGAN BAR"]]),
    14: (["11319500"], [["MOKELUMNE"]]),
    # the historical Jenny Lind gauge no longer exists in NWIS; New Hogan anchors
    15: (["11308900"], [["CALAVERAS"]]),
    16: (["11299000", "11302000"], [["MELONES", "STANISLAUS"]]),
    18: (["11288000", "11289650"], [["TUOLUMNE"]]),
    19: (["11270000", "11270900"], [["MERCED"]]),
    20: (["11259000"], [["CHOWCHILLA"]]),
    21: (["11258000", "11257500"], [["FRESNO"], ["DAULTON"]]),
    22: (["11250100", "11251000"], [["MILLERTON", "SAN JOAQUIN"]]),
}

UA = {"User-Agent": "SAC-SMA uf_locations check (CADWR Climate Change Program)"}
NLDI = "https://api.water.usgs.gov/nldi"  # api.waterdata.usgs.gov paths 404


# ---------------------------------------------------------------- web helpers
def _rdb(text: str) -> pd.DataFrame:
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    return pd.DataFrame([l.split("\t") for l in lines[2:]], columns=lines[0].split("\t"))


def nwis_site(site_no: str) -> dict | None:
    url = ("https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=%s"
           "&siteOutput=expanded" % site_no)
    try:
        df = _rdb(urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                         timeout=30).read().decode())
        if df.empty:
            return None
        r = df.iloc[0]
        return {"site_no": r["site_no"], "station_nm": r["station_nm"],
                "lat": float(r["dec_lat_va"]), "lon": float(r["dec_long_va"]),
                "da_mi2": float(r["drain_area_va"]) if r.get("drain_area_va", "").strip() else np.nan}
    except Exception as e:  # noqa: BLE001 - report and continue, gauge stays blank
        print("  NWIS fail %s: %s" % (site_no, e))
        return None


def cdec_coords(sta: str) -> tuple[float, float] | None:
    url = "https://cdec.water.ca.gov/dynamicapp/staMeta?station_id=%s" % sta
    try:
        html = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                      timeout=30).read().decode(errors="replace")
        lat = re.search(r"Latitude</t[dh]>\s*<td[^>]*>\s*([0-9.]+)", html)
        lon = re.search(r"Longitude</t[dh]>\s*<td[^>]*>\s*(-?[0-9.]+)", html)
        if not lat:
            lat = re.search(r"Latitude\D{0,40}?([34][0-9]\.[0-9]+)", html)
            lon = re.search(r"Longitude\D{0,40}?(-?1[12][0-9]\.[0-9]+)", html)
        if lat and lon:
            lo = float(lon.group(1))
            return float(lat.group(1)), (lo if lo < 0 else -lo)
    except Exception as e:  # noqa: BLE001
        print("  CDEC fail %s: %s" % (sta, e))
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------- main
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    locs = pd.read_csv(REPO / "data/dwr_unimpaired/uf_locations.csv")
    xw = pd.read_csv(REPO / "data/calsim/calsim_crosswalk.csv").set_index("arc")
    g = gpd.read_file(REPO / "data/calsim/gis/calsim3.gpkg", layer="CalSim3_Merged")
    g["arc"] = "I_" + g["Connect_No"].astype(str)
    g = g.set_index("arc")

    inf = pd.read_csv(REPO / "data/calsim/calsim3_inflow_monthly.csv", parse_dates=["date"])
    unimp = pd.read_csv(REPO / "data/calsim/calsim_unimpaired_monthly.csv", parse_dates=["date"])
    ufm = pd.read_csv(REPO / "data/dwr_unimpaired/uf_monthly.csv", parse_dates=["date"])
    for df in (inf, unimp, ufm):
        df["wy"] = df["date"].dt.year + (df["date"].dt.month >= 10)
    infw = inf[inf.wy.between(*WY)].pivot(index="date", columns="arc", values="flow_taf")
    unw = unimp[unimp.wy.between(*WY)].pivot(index="date", columns="system", values="flow_taf")
    ufw = ufm[ufm.wy.between(*WY)].pivot(index="date", columns="uf", values="flow_taf")

    #: UF -> CalSim UNIMP whole-watershed anchor (SRBB already includes SHAS)
    sys_anchor = {6: "SRBB", 8: "OROV", 9: "YUBA", 11: "FOLS",
                  16: "ST", 18: "TU", 19: "ME", 22: "SJ"}

    cache_f = OUT / "web_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {"nwis": {}, "cdec": {}}
    findings, rows, outlets, dissolved = [], [], [], []

    for _, uf in locs.iterrows():
        n = int(uf.uf)
        if n in SKIP:
            rows.append({"uf": n, "name": uf["name"], "status": "SKIP: " + SKIP[n]})
            continue
        arcs = uf.arcs.split(";")
        row = {"uf": n, "name": uf["name"], "status": "checked", "n_arcs": len(arcs)}

        # ---- 1. crosswalk set consistency, both directions
        keys = {(c, uf[c]) for c in ("basin_11obs", "basin_9unimp") if isinstance(uf[c], str)}
        fwd, rev = [], []
        for a in arcs:
            if a not in xw.index:
                note = EXPECTED.get((n, a))
                fwd.append("%s absent from crosswalk%s" % (a, " [expected: %s]" % note if note else ""))
                continue
            got = {(c, xw.loc[a, c]) for c in ("basin_11obs", "basin_9unimp")
                   if isinstance(xw.loc[a, c], str)}
            if keys and not (keys & got):
                note = EXPECTED.get((n, a))
                fwd.append("%s -> %s in crosswalk%s"
                           % (a, ",".join(sorted(v for _, v in got)) or "unassigned",
                              " [expected: %s]" % note if note else ""))
        if not keys:
            findings.append(("note", n, "no basin key; the crosswalk carries no record that "
                             "these %d arcs form UF %d" % (len(arcs), n)))
        for col, val in keys:
            for a in sorted(set(xw.index[xw[col] == val]) - set(arcs)):
                rev.append("%s has %s=%s but is not in the UF %d arc set" % (a, col, val, n))
        row["xwalk_fwd"] = "; ".join(fwd) or "ok"
        row["xwalk_rev"] = "; ".join(rev) or ("ok" if keys else "n/a (no basin key)")
        findings += [("flag", n, "crosswalk fwd: " + t) for t in fwd if "[expected" not in t]
        findings += [("flag", n, "crosswalk rev: " + t) for t in rev]

        # ---- 2. area closure vs gpkg
        have_poly = [a for a in arcs if a in g.index]
        gsum = float(g.loc[have_poly, "SQ_MI"].sum())
        row.update(area_csv=uf.area_mi2_calsim,
                   area_dpct=round(100 * (gsum / uf.area_mi2_calsim - 1), 2),
                   arcs_no_polygon=";".join(a for a in arcs if a not in g.index))
        if abs(row["area_dpct"]) > 1:
            findings.append(("flag", n, "area closure: gpkg %.1f vs csv %.1f (%+.1f%%)"
                             % (gsum, uf.area_mi2_calsim, row["area_dpct"])))
        for a in arcs:
            if a not in g.index and (n, a) not in EXPECTED:
                findings.append(("flag" if xw.loc[a, "in_calsim3"] else "note", n,
                                 "no polygon in CalSim3_Merged for %s (in_calsim3=%s)"
                                 % (a, xw.loc[a, "in_calsim3"] if a in xw.index else "?")))

        # ---- 3. volume closure WY1950-84
        have_ser = [a for a in arcs if a in infw.columns]
        s = infw[have_ser].sum(axis=1)
        tgt = ufw[n]
        common = s.index.intersection(tgt.index)
        s, tgt = s.loc[common], tgt.loc[common]
        row.update(arcs_no_series=";".join(a for a in arcs if a not in infw.columns),
                   arcsum_tafyr=round(12 * s.mean(), 1),
                   dwr_unimp_tafyr=round(12 * tgt.mean(), 1),
                   vol_dpct=round(100 * (s.mean() / tgt.mean() - 1), 1),
                   vol_r=round(float(np.corrcoef(s, tgt)[0, 1]), 4))
        if abs(row["vol_dpct"]) > 5:
            findings.append(("flag", n, "volume closure: arc sum %.0f vs DWR %.0f TAF/yr "
                             "(%+.1f%%, r=%.3f)" % (12 * s.mean(), 12 * tgt.mean(),
                                                    row["vol_dpct"], row["vol_r"])))
        if n in sys_anchor and sys_anchor[n] in unw.columns:
            u = unw[sys_anchor[n]].loc[common]
            row.update(unimp_sys=sys_anchor[n],
                       arcsum_vs_unimp_dpct=round(100 * (s.mean() / u.mean() - 1), 1),
                       unimp_vs_dwr_dpct=round(100 * (u.mean() / tgt.mean() - 1), 1))

        # ---- 4. outlet ground truth (NWIS + CDEC, cached)
        cand, kw = NWIS_CAND.get(n, ([], []))
        fetched = []
        for c in cand:
            info = cache["nwis"].get(c) or nwis_site(c)
            if info:
                cache["nwis"][c] = info
                fetched.append(info)
        best = next((f for f in fetched
                     if all(any(k in f["station_nm"].upper() for k in grp) for grp in kw)), None)
        if best is None and fetched:
            best = fetched[0]
            findings.append(("flag", n, "NWIS name check FAILED for %s; got %s"
                             % (cand, [f["station_nm"] for f in fetched])))
        # report_table keeps only the check results (site number = join key);
        # outlet identity (name, coords, offsets) lives solely in uf_outlets.csv
        if best:
            row.update(usgs_site=best["site_no"], usgs_da_mi2=best["da_mi2"])
            if best["da_mi2"] == best["da_mi2"]:
                row["nwis_da_dpct"] = round(100 * (uf.area_mi2_calsim / best["da_mi2"] - 1), 1)
        cd = None
        if isinstance(uf.cdec_id, str):
            cd = cache["cdec"].get(uf.cdec_id)
            cd = tuple(cd) if cd else cdec_coords(uf.cdec_id)
            if cd:
                cache["cdec"][uf.cdec_id] = cd
        outlets.append({"uf": n, "name": uf["name"],
                        "cdec_id": uf.cdec_id if isinstance(uf.cdec_id, str) else "",
                        "usgs_site": best["site_no"] if best else "",
                        "usgs_name": best["station_nm"] if best else "",
                        "usgs_lat": best["lat"] if best else np.nan,
                        "usgs_lon": best["lon"] if best else np.nan,
                        "usgs_da_mi2": best["da_mi2"] if best else np.nan,
                        "cdec_lat": cd[0] if cd else np.nan,
                        "cdec_lon": cd[1] if cd else np.nan,
                        "usgs_cdec_km": round(haversine_km(best["lat"], best["lon"], *cd), 2)
                                        if best and cd else np.nan})

        # ---- 5a. per-UF map
        fig, ax = plt.subplots(figsize=(7, 7.5))
        g.plot(ax=ax, facecolor="0.94", edgecolor="0.75", linewidth=0.4)
        gm = g.loc[have_poly]
        gm.plot(ax=ax, facecolor="#7ba7cc", edgecolor="#2c5d8a", linewidth=0.8, alpha=0.85)
        for a, owner in ADJACENT.items():
            if owner == n and a in g.index:
                g.loc[[a]].plot(ax=ax, facecolor="#e8a54b", edgecolor="#a35f00",
                                linewidth=1.2, alpha=0.9)
                c = g.loc[a].geometry.centroid
                ax.annotate(a + " (adjacent, non-member)", (c.x, c.y), fontsize=7,
                            ha="center", color="#7a4400")
        for a in have_poly:
            c = gm.loc[a].geometry.centroid
            ax.annotate(a[2:], (c.x, c.y), fontsize=5.5, ha="center", color="#1a3a5c")
        if best:
            ax.plot(best["lon"], best["lat"], marker="*", ms=16, mec="k", mfc="red",
                    ls="none", label="USGS %s" % best["site_no"], zorder=5)
        if cd:
            ax.plot(cd[1], cd[0], marker="x", ms=9, color="k", ls="none",
                    label="CDEC %s" % uf.cdec_id, zorder=5)
        tb = gm.total_bounds
        xs = [tb[0], tb[2]] + ([best["lon"]] if best else [])
        ys = [tb[1], tb[3]] + ([best["lat"]] if best else [])
        ax.set_xlim(min(xs) - 0.15, max(xs) + 0.15)
        ax.set_ylim(min(ys) - 0.15, max(ys) + 0.15)
        ax.set_title("UF %d — %s\ncsv %.0f mi² | gpkg %.0f mi² | NWIS DA %s mi²"
                     % (n, uf["name"], uf.area_mi2_calsim, gsum,
                        ("%.0f" % best["da_mi2"]) if best and best["da_mi2"] == best["da_mi2"] else "n/a"),
                     fontsize=10)
        if best or cd:
            ax.legend(loc="best", fontsize=8)
        ax.set_aspect(1 / math.cos(math.radians(float(np.mean(ys)))))
        fig.tight_layout()
        fig.savefig(FIG / ("uf_%02d.png" % n), dpi=110)
        plt.close(fig)

        dissolved.append({"uf": n, "name": uf["name"],
                          "geometry": make_valid(g.loc[have_poly].union_all())})
        rows.append(row)

    cache_f.write_text(json.dumps(cache, indent=1))

    # ---- 5b. dissolve: tiling, overlaps, Figure-2-1 analogue
    claims = {a for _, uf in locs.iterrows() if isinstance(uf.arcs, str) for a in uf.arcs.split(";")}
    unassigned = g.loc[[a for a in g.index if a not in claims]]
    diss = gpd.GeoDataFrame(dissolved, crs=g.crs)
    diss.to_file(OUT / "uf_dissolved.gpkg", layer="uf_dissolved", driver="GPKG")
    d2 = diss.to_crs(3310)
    d2["geometry"] = d2.geometry.buffer(0)
    for i in range(len(d2)):
        for j in range(i + 1, len(d2)):
            ov = d2.geometry.iloc[i].intersection(d2.geometry.iloc[j]).area / 2.59e6
            if ov > 0.5:  # slivers along shared divides run 0.02-0.25 mi2
                findings.append(("flag", int(d2.uf.iloc[i]),
                                 "dissolved overlap with UF %d: %.2f mi2" % (d2.uf.iloc[j], ov)))
    fig, ax = plt.subplots(figsize=(9, 12))
    g.plot(ax=ax, facecolor="0.96", edgecolor="0.8", linewidth=0.3)
    for i, r in diss.iterrows():
        gpd.GeoSeries([r.geometry], crs=g.crs).plot(
            ax=ax, facecolor=plt.cm.tab20(i % 20), alpha=0.65, edgecolor="k", linewidth=1.0)
        c = r.geometry.representative_point()
        ax.annotate(str(r.uf), (c.x, c.y), fontsize=13, fontweight="bold", ha="center")
    unassigned.plot(ax=ax, facecolor="none", edgecolor="#b06000", linewidth=0.7,
                    hatch="///", alpha=0.5)
    ax.set_title("UF subbasins dissolved from CalSim3 arc sets\n"
                 "(numbers = UF; hatched = rim arcs assigned to no UF) — compare to DWR Fig. 2-1")
    ax.set_aspect(1 / math.cos(math.radians(39)))
    fig.tight_layout()
    fig.savefig(FIG / "uf_dissolved_overview.png", dpi=110)
    plt.close(fig)
    print("tiling: %d rim arcs, %d assigned, %d unassigned (%.0f mi2)"
          % (len(g), len(claims & set(g.index)), len(unassigned), unassigned.SQ_MI.sum()))

    # ---- 5c. contact sheet
    from PIL import Image
    files = sorted(FIG.glob("uf_[0-9][0-9].png"))
    W = 320
    thumbs = []
    for f in files:
        im = Image.open(f)
        thumbs.append(im.resize((W, int(im.height * W / im.width)), Image.LANCZOS))
    ph = max(t.height for t in thumbs)
    cols = 5
    sheet = Image.new("RGB", (cols * W, -(-len(thumbs) // cols) * ph), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * W, (i // cols) * ph))
    sheet.convert("P", palette=Image.ADAPTIVE, colors=256).save(
        FIG / "uf_all_grid.png", optimize=True)

    # ---- Paynes Creek ruling (NLDI Bend Bridge watershed vs I_PYN001)
    basin_f = OUT / "nldi_bend_basin.json"
    try:
        if not basin_f.exists():
            url = NLDI + "/linked-data/nwissite/USGS-11377100/basin?f=json"
            basin_f.write_bytes(urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=60).read())
        basin = make_valid(shape(json.loads(basin_f.read_text())["features"][0]["geometry"]))
        for a in ("I_PYN001", "I_SCW008", "I_ANT011"):
            p = make_valid(g.to_crs(4326).loc[a].geometry)
            frac = p.intersection(basin).area / p.area
            findings.append(("note", 6, "NLDI Bend Bridge watershed overlap: %s %.1f%%" % (a, 100 * frac)))
            if a == "I_PYN001" and frac > 0.5:
                findings.append(("flag", 6, "I_PYN001 falls INSIDE the Bend Bridge watershed "
                                 "— the 2026-08-04 exclusion ruling no longer holds"))
    except Exception as e:  # noqa: BLE001 - offline runs keep the cached ruling
        print("  NLDI check skipped: %s" % e)

    # ------------------------------------------------------------- write out
    pd.DataFrame(rows).to_csv(OUT / "report_table.csv", index=False)
    pd.DataFrame(outlets).to_csv(REPO / "data/dwr_unimpaired/uf_outlets.csv", index=False)
    with open(OUT / "findings.md", "w", encoding="utf-8") as f:
        f.write("# uf_locations.csv verification — findings (volume window WY%d-%d)\n" % WY)
        for sev in ("flag", "note"):
            f.write("\n## %s\n\n" % ("FLAGS" if sev == "flag" else "Notes"))
            for s, n, t in findings:
                if s == sev:
                    f.write("- UF %d: %s\n" % (n, t))
    nflag = sum(1 for s, _, _ in findings if s == "flag")
    print("%d flags, %d notes -> %s" % (nflag, len(findings) - nflag, OUT / "findings.md"))


if __name__ == "__main__":
    main()
