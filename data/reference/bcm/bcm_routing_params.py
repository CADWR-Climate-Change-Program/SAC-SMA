"""Extract the BCM monthly routing parameters from the USGS workbook, and check the port.

``CalBasins_v8_DWR_FNF_PRISM19.xlsx`` (beside this script, not tracked) holds the USGS
post-processing ("second-order correction") of BCM v8: BCM runoff and recharge, forced with
PRISM, routed through three reservoirs per basin and fitted to CDEC full natural flow over
WY2000-2010.  One sheet per calibrated basin (``6``, ``8``, ``9``, ``11``, ``13``, ``14``,
``16``, ``17``, ``18``, ``21``, ``23``) reads its BCM input from the sheet ``Cal_outfile``.

The script refuses to write unless the workbook has the structure the port assumes:

- every formula of columns F to T, rows 14 (1999-10) to 145 (2010-09), of every sheet is the
  template below, in relative (R1C1) form;
- each sheet's area (``D5``) and inputs (``F``, ``G``) read the ``Cal_outfile`` block of its own
  basin number, 132 consecutive months from 1999-10, with one area;
- no formula reads a ``Cal_outfile`` column other than ``rch_mm``, ``run_mm`` and
  ``Basin_area_m^2`` (``rchrunscaler`` enters nothing);
- the parameter labels and statistic formulas are where the port reads them.

It writes ``bcm_routing_params.csv`` (one row per sheet), ``bcm_routing_check.csv`` (each
sheet's inputs, measured flow and cached ``P`` and ``Q``, 132 months) and ``bcm_cal_outfile.csv``
(the BCM input of every basin of ``Cal_outfile``), then runs the port
(``sacsma.benchmark.bcm_routing.check``) on them.

The refit.  Sheet 23 is mis-wired: it routes ``Cal_outfile`` basin 23 (816 mi2, a warm, low
basin) against the flow of the Kings at Pine Flat, and needs ``AquiferRch`` 4.65 to make up the
volume.  Basin 22 is the Kings at Pine Flat (1,523 mi2).  :data:`REFITS` fits the sheet's
equations to basin 22 against sheet 23's measured column over the sheet's own window, the way
the sheets were fitted (``sacsma.benchmark.bcm_routing.fit``), and adds the result to the
parameter table as a row of ``source`` ``refit``; the workbook rows are ``source`` ``workbook``.
The refit reads only the tracked tables, so ``--refit`` runs without the workbook.

Usage
-----
    python data/reference/bcm/bcm_routing_params.py           # extract, refit, write, check
    python data/reference/bcm/bcm_routing_params.py --refit   # refit only (no workbook needed)
    python data/reference/bcm/bcm_routing_params.py --check   # check only (no workbook needed)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "_paths.py").is_file())))
from _paths import DATA, REPO, layout  # noqa: E402

WORKBOOK = layout.bcm(DATA, "CalBasins_v8_DWR_FNF_PRISM19.xlsx")
OUT_PARAMS = layout.bcm(DATA, "bcm_routing_params.csv")
OUT_CHECK = layout.bcm(DATA, "bcm_routing_check.csv")
OUT_CAL = layout.bcm(DATA, "bcm_cal_outfile.csv")

#: sheet -> (CDEC site of the benchmark it is applied at, CDEC station whose full natural flow
#: the sheet's measured column reproduces).  Found by comparing the measured column with every
#: CDEC record (``--check`` prints the evidence).  Sheet 16 was fitted to the Stanislaus at
#: Goodwin (SNS), the benchmark site is New Melones (NML) on the same river.
SITES = {6: ("SHA", "SIS"), 8: ("ORO", "FTO"), 9: ("YRS", "YRS"), 11: ("FOL", "AMF"),
         13: ("CSN", "CSN"), 14: ("MKM", "MKM"), 16: ("NML", "SNS"), 17: ("TLG", "TLG"),
         18: ("MRC", "MRC"), 21: ("MIL", "SBF"), 23: ("PNF", "KGF")}
#: basin of Cal_outfile -> the refit made on it: the sheet whose measured column it is fitted to,
#: the CDEC station of that column and the benchmark site it is applied at
REFITS = {22: {"name": "22-Kings (refit)", "measured_sheet": 23, "measured_station": "KGF",
               "cdec_site": "PNF"}}
FIRST, LAST = 14, 145          # the rows of 1999-10 and 2010-09
N_MONTHS = LAST - FIRST + 1

#: column -> R1C1 form of its formula, row 14 then rows 15-145
TEMPLATE = {
    "H": ("=(R[0]C[-2]/1000)*R5C4",) * 2,
    "I": ("=(R[0]C[-2]/1000)*R5C4",) * 2,
    "J": ("=IF((R[-4]C[1]+R[0]C[-1])<=0,0,(R[-4]C[1]+R[0]C[-1]-R[0]C[1]))",
          "=IF((R[-1]C[0]+R[0]C[-1])<=0,0,(R[-1]C[0]+R[0]C[-1]-R[0]C[1]))"),
    "K": ("=IF(R[-4]C[0]<=0,0,(R[-4]C[0]^R3C[0])*R2C[0])",
          "=IF(R[-1]C[-1]<=0,0,(R[-1]C[-1]^R3C[0])*R2C[0])"),
    "L": ("=(R[-4]C[-1]+R[0]C[-4]-R[0]C[1]-R[0]C[3])",
          "=(R[-1]C[0]+R[0]C[-4]-R[0]C[1]-R[0]C[3])"),
    "M": ("=IF(R10C11<0,0,R4C[-2]*(R[-4]C[-2])^(R5C[-2]))",
          "=IF(R[-1]C[-1]<0,0,R4C[-2]*(R[-1]C[-1])^(R5C[-2]))"),
    "N": ("=R10C[-3]+IF(R[0]C[-2]<0,R[0]C[-2],0)",) * 2,
    "O": ("=R6C[-4]*(R[-1]C[-1])^R7C[-4]",) * 2,
    "P": ("=(OFFSET(R[0]C[-5],R10C15,0))+(OFFSET(R[0]C[-3],R10C16,0))+R[0]C[-1]",) * 2,
    "Q": ("=R[0]C[-1]*R8C[-6]",) * 2,
    "R": ("=(1-R8C[-7])*R[0]C[-2]",) * 2,
    "S": ("=(R[0]C[-2]-R[0]C[-14])^2",) * 2,
    "T": ("=(R[0]C[-15]-R[0]C[-3])*100",) * 2,
}
#: label cell -> expected text, and the value cell beside it
LABELS = {"J2": "SurfaceScale", "J3": "SurfaceExp", "J4": "ShallowScale", "J5": "ShallowExp",
          "J6": "DeepScale", "J7": "DeepExp", "J8": "AquiferRch", "J10": "Antecedent storage",
          "O9": "peak", "P9": "recession", "C5": "Area (m2):", "L4": "Monthly r2:",
          "L5": "NSS:", "L6": "PBIAS:", "E13": "(m3)", "F12": "BCM rch", "G12": "BCM run",
          "Q12": "BCM flow"}
PARAM_CELLS = {"surface_scale": "K2", "surface_exp": "K3", "shallow_scale": "K4",
               "shallow_exp": "K5", "deep_scale": "K6", "deep_exp": "K7", "aquifer_rch": "K8",
               "antecedent_m3": "K10", "peak_lag": "O10", "recession_lag": "P10"}
#: the Cal_outfile columns a sheet may read: rch_mm, run_mm, Basin_area_m^2
CAL_COLUMNS = {"R": "rch_mm", "S": "run_mm", "V": "Basin_area_m^2"}

_REF = re.compile(r"((?:'[^']+'|[A-Za-z_][A-Za-z0-9_ ]*)!)?(\$?)([A-Z]{1,3})(\$?)(\d+)")


def _col(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def _r1c1(formula: str, row: int, col: int) -> str:
    """A1 references to R1C1: relative ``R[dr]C[dc]``, absolute ``R5C4``."""
    def sub(m):
        sheet, dc, c, dr, r = m.groups()
        cs = f"C{_col(c)}" if dc else f"C[{_col(c) - col}]"
        rs = f"R{r}" if dr else f"R[{int(r) - row}]"
        return (sheet or "") + rs + cs
    return _REF.sub(sub, formula)


def _read(data_only: bool) -> dict[str, dict[tuple[int, int], object]]:
    """Every non-empty cell of the sheets the port needs, ``{sheet: {(row, col): value}}``."""
    import openpyxl
    wb = openpyxl.load_workbook(WORKBOOK, data_only=data_only, read_only=True)
    out = {}
    for name in ["Cal_outfile", *map(str, SITES)]:
        ws = wb[name]
        max_row = 3200 if name == "Cal_outfile" else 470
        cells = {}
        for r, row in enumerate(ws.iter_rows(min_row=1, max_row=max_row, values_only=True), 1):
            for c, v in enumerate(row, 1):
                if v is not None:
                    cells[(r, c)] = v
        out[name] = cells
    wb.close()
    return out


def _a1(cell: str) -> tuple[int, int]:
    m = re.fullmatch(r"([A-Z]+)(\d+)", cell)
    return int(m.group(2)), _col(m.group(1))


def _stat_rows(formula: str) -> tuple[int, int]:
    """First and last row of the E range of a statistic formula (``RSQ(E15:E463,...)``)."""
    m = re.search(r"E(\d+):E(\d+)", formula)
    return int(m.group(1)), int(m.group(2))


def extract() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not WORKBOOK.is_file():
        sys.exit(f"workbook not found: {WORKBOOK} (delivered with the BCM release, not tracked)")
    f, v = _read(data_only=False), _read(data_only=True)
    cal_f, cal_v = f["Cal_outfile"], v["Cal_outfile"]
    hdr = {cal_v[(1, c)]: c for r, c in cal_v if r == 1}
    assert [hdr[n] for n in ("Year", "Month", "Basin", "rch_mm", "run_mm", "Basin_area_m^2",
                             "rchrunscaler")] == [2, 3, 4, 18, 19, 22, 27], hdr
    assert not any(isinstance(x, str) and x.startswith("=") for x in cal_f.values()), \
        "Cal_outfile holds formulas"

    params, check = [], []
    for sheet, (site, station) in SITES.items():
        sf, sv = f[str(sheet)], v[str(sheet)]

        def cell_f(a, sf=sf):
            return sf.get(_a1(a))

        def cell_v(a, sv=sv):
            return sv.get(_a1(a))

        for a, text in LABELS.items():
            assert str(cell_v(a)).strip() == text, (sheet, a, cell_v(a), text)
        assert cell_v("D1") == sheet, (sheet, cell_v("D1"))

        # every formula of F..T, rows 14..145, against the template
        for r in range(FIRST, LAST + 1):
            for col, (first, rest) in TEMPLATE.items():
                got = _r1c1(sf[(r, _col(col))], r, _col(col))
                assert got == (first if r == FIRST else rest), (sheet, f"{col}{r}", got)
        # the inputs: F (rch) and G (run) read Cal_outfile R and S, one row per month
        m = re.fullmatch(r"=Cal_outfile!\$V\$(\d+)", cell_f("D5"))
        block = int(m.group(1))
        for r in range(FIRST, LAST + 1):
            assert sf[(r, _col("F"))] == f"=Cal_outfile!R{block + r - FIRST}", (sheet, r)
            assert sf[(r, _col("G"))] == f"=Cal_outfile!S{block + r - FIRST}", (sheet, r)
        rows = range(block, block + N_MONTHS)
        assert {cal_v[(r, 4)] for r in rows} == {sheet}, (sheet, "block of another basin")
        assert {cal_v[(r, 22)] for r in rows} == {cell_v("D5")}, (sheet, "area varies or differs")
        months = pd.period_range("1999-10", periods=N_MONTHS, freq="M")
        assert [(cal_v[(r, 2)], cal_v[(r, 3)]) for r in rows] == [(p.year, p.month) for p in months]
        dates = [pd.Timestamp(sv[(r, 3)]).to_period("M") for r in range(FIRST, LAST + 1)]
        assert dates == list(months), (sheet, "dates")

        # the statistics: one window per sheet (rows a..145; blank rows past 145 are ignored)
        a, b = _stat_rows(cell_f("M4"))
        assert cell_f("M4") == f"=RSQ(E{a}:E{b},Q{a}:Q{b})", cell_f("M4")
        assert cell_f("M2") == f"=AVERAGE(S{a}:S{b})" and cell_f("M3") == f"=VAR(E{a}:E{b})"
        assert cell_f("M5") == "=1-(M2/M3)"
        pbias = re.fullmatch(r"=\(SUM\(T(\d+):T(\d+)\)\)/\(SUM\(E\1:E\2\)\)", cell_f("M6"))
        pa, pb = pbias.groups()
        assert int(pa) == a and min(b, int(pb)) >= LAST, (sheet, a, b, pa, pb)

        p = {"sheet": sheet, "source": "workbook", "basin_no": sheet, "name": cell_v("D2"),
             "station_id": cell_v("D3"), "measured_station": station, "cdec_site": site,
             "area_m2": cell_v("D5")}
        p.update({k: cell_v(c) for k, c in PARAM_CELLS.items()})
        p.update({"cal_start": str(months[0]), "cal_end": str(months[-1]),
                  "stat_start": str(months[a - FIRST]), "stat_end": str(months[-1]),
                  "r2": cell_v("M4"), "nse": cell_v("M5"), "pbias": cell_v("M6")})
        params.append(p)
        for i, r in enumerate(range(FIRST, LAST + 1)):
            check.append({"sheet": sheet, "month": str(months[i]),
                          "rch_mm": sv[(r, _col("F"))], "run_mm": sv[(r, _col("G"))],
                          "area_m2": cell_v("D5"), "measured_m3": float(sv[(r, _col("E"))]),
                          "p_m3": sv[(r, _col("P"))], "q_m3": sv[(r, _col("Q"))]})

    # no formula in the sheets reads another Cal_outfile column (rchrunscaler is AA)
    for name in map(str, SITES):
        for (r, c), x in f[name].items():
            if isinstance(x, str):
                for col in re.findall(r"Cal_outfile!\$?([A-Z]+)\$?\d+", x):
                    assert col in CAL_COLUMNS, (name, r, c, x)
    # the BCM input of every basin (Cal_outfile holds values only, checked above)
    cal = [{"basin_no": cal_v[(r, 4)], "month": f"{cal_v[(r, 2)]:04d}-{cal_v[(r, 3)]:02d}",
            "rch_mm": cal_v[(r, 18)], "run_mm": cal_v[(r, 19)], "area_m2": cal_v[(r, 22)]}
           for r in sorted({r for r, _ in cal_v}) if r > 1 and cal_v.get((r, 4)) is not None]
    return pd.DataFrame(params), pd.DataFrame(check), pd.DataFrame(cal)


def refit(params: pd.DataFrame, check: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """The :data:`REFITS` rows of the parameter table, from the tracked tables only."""
    from sacsma.benchmark.bcm_routing import fit, to_m3

    rows = []
    for basin, spec in REFITS.items():
        b = cal[cal["basin_no"] == basin].sort_values("month")
        m = check[check["sheet"] == spec["measured_sheet"]].sort_values("month")
        if list(b["month"]) != list(m["month"]) or b["area_m2"].nunique() != 1:
            raise ValueError(f"basin {basin}: months or area do not line up with sheet "
                             f"{spec['measured_sheet']}")
        sheet = params.set_index("sheet").loc[spec["measured_sheet"]]
        area = float(b["area_m2"].iloc[0])
        window = m["month"].between(sheet["stat_start"], sheet["stat_end"]).to_numpy()
        p = fit(to_m3(b["run_mm"].to_numpy(), area), to_m3(b["rch_mm"].to_numpy(), area),
                m["measured_m3"].to_numpy(), window=window)
        rows.append({"sheet": basin, "source": "refit", "basin_no": basin, "name": spec["name"],
                     "station_id": "", "measured_station": spec["measured_station"],
                     "cdec_site": spec["cdec_site"], "area_m2": area,
                     **{k: p[k] for k in ("surface_scale", "surface_exp", "shallow_scale",
                                          "shallow_exp", "deep_scale", "deep_exp", "aquifer_rch",
                                          "antecedent_m3", "peak_lag", "recession_lag")},
                     "cal_start": sheet["cal_start"], "cal_end": sheet["cal_end"],
                     "stat_start": sheet["stat_start"], "stat_end": sheet["stat_end"],
                     "r2": p["r2"], "nse": p["nse"], "pbias": p["pbias"]})
    return pd.DataFrame(rows)


def _fnf_monthly_m3(site: str) -> pd.Series:
    """Monthly CDEC full natural flow volume (m3) of a site, complete months only."""
    if site in ("CSN", "SNS"):
        d = pd.read_csv(layout.cdec_fnf(DATA, "fnf_daily.csv"), parse_dates=["date"])
        d = d[d["station"] == site]
        m3 = d["flow_cfs"].where(d["flow_cfs"] >= 0) * 0.0283168466 * 86400
    else:
        d = pd.read_csv(layout.gage_15cdec(DATA), parse_dates=["date"])
        d = d[d["basin"] == site]
        area = pd.read_csv(layout.basin_area(DATA, "15cdec")).set_index("basin")["area_mi2"][site]
        m3 = d["flow"] / 1000 * area * 2589988.110336
    s = pd.Series(m3.to_numpy(), index=d["date"].dt.to_period("M"))
    g = s.groupby(level=0)
    full = g.count() == g.size().index.days_in_month
    return g.sum()[full]


def run_check() -> int:
    from sacsma.benchmark.bcm_routing import check, load_params

    res = check(DATA)
    pd.set_option("display.width", 200)
    print("port against the sheets' cached values (tail='zero', as the workbook):")
    print(res.to_string(float_format=lambda x: f"{x:.3g}" if abs(x) < 1e-3 else f"{x:.4f}"))
    bad = res[res["max_rel_err_q"] > 1e-9]

    print("\nthe measured column against CDEC full natural flow (complete months):")
    params = load_params(DATA)
    tab = pd.read_csv(OUT_CHECK)
    tab["month"] = pd.PeriodIndex(tab["month"], freq="M")
    for sheet, g in tab.groupby("sheet", sort=False):
        e = g.set_index("month")["measured_m3"]
        p = params.loc[sheet]
        out = []
        # the benchmark site, and the station the sheet was fitted to where that is another
        sites = [p["cdec_site"]] + (["SNS"] if p["cdec_site"] == "NML" else [])
        for site in sites:
            c = _fnf_monthly_m3(site).reindex(e.index).dropna()
            r = np.corrcoef(e[c.index], c)[0, 1]
            ratio = e[c.index].sum() / c.sum()
            out.append(f"{site}: {len(c)} months, r {r:.4f}, volume ratio {ratio:.3f}")
        print(f"  sheet {sheet:>2} ({p['name']}): " + "; ".join(out))
    if len(bad):
        print(f"\nFAIL: {len(bad)} sheet(s) above 1e-9 relative: {list(bad.index)}")
        return 1
    print("\nOK: every sheet's Q reproduced within 1e-9 relative")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="run the port against the check table only")
    ap.add_argument("--refit", action="store_true",
                    help="redo the refits from the tracked tables only")
    args = ap.parse_args()
    if args.refit:
        params = pd.read_csv(OUT_PARAMS, float_precision="round_trip")
        params = params[params["source"] == "workbook"]
        check = pd.read_csv(OUT_CHECK, float_precision="round_trip")
        cal = pd.read_csv(OUT_CAL, float_precision="round_trip")
    elif not args.check:
        params, check, cal = extract()
        check.to_csv(OUT_CHECK, index=False)
        cal.to_csv(OUT_CAL, index=False)
        print(f"wrote {OUT_CHECK.relative_to(REPO).as_posix()} ({len(check)} rows) and "
              f"{OUT_CAL.relative_to(REPO).as_posix()} ({len(cal)} rows)")
    if not args.check:
        fitted = refit(params, check, cal)
        print(fitted.drop(columns=["station_id"]).to_string(index=False))
        params = pd.concat([params, fitted], ignore_index=True)
        params.to_csv(OUT_PARAMS, index=False)
        print(f"wrote {OUT_PARAMS.relative_to(REPO).as_posix()} ({len(params)} rows: "
              f"{(params['source'] == 'workbook').sum()} sheets, {len(fitted)} refit)")
    return run_check()


if __name__ == "__main__":
    sys.exit(main())
