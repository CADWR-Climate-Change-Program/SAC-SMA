"""The products of the calibrated models: CalLite monthly inflows and 15-CDEC daily flow.

Each is the archived GA calibration of its sets (``artifacts/models/``) run by the reference
model under one forcing, from the cold start at the beginning of the forcing record, as the
original study ran it.  The series start in October 1921, the first month of CalLite, which
leaves the model more than six years of run-up, and end with the last complete water year
of the forcing (September 2018).

``callite`` writes, per forcing, the three CalLite input files of the original study's
wrapper (``artifacts/product/callite/<forcing>/``):

* ``SACSMA_Inflow_CalLite_9.txt`` and ``SACSMA_Inflow_CalLite_11.txt``: ``Year, Month`` and
  one column per watershed, the monthly total in mm.  ``Year`` is the water year.
* ``SACSMA_Inflow_CalLite_12.txt``: the twelve rim inflows under their CalLite DSS paths,
  three header lines (start of the record, path, unit) and one row per month: monthly mean
  CFS, or TAF for ``I_MCLRE``, ``I_MELON``, ``I_NHGAN`` and ``I_PEDRO``, over the areas of
  ``data/inputs/domains/12rim/basin_area.csv``.

``15cdec`` writes ``flow_daily.csv`` (``date, basin, flow_mm, flow_cfs``) for the historical
forcing, the only one its off-grid units have.

``sacsma product {callite,15cdec} [--forcing NAME ...]`` writes them (under a minute each);
``sacsma verify product`` repeats every tracked series.  The rim-inflow product of the learned
model is made by ``sacsma dpl calsim product``.
"""

from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .io import load_basin_area, mmday_to_cfs

#: the forcings each product carries
FORCINGS = {
    paths.CALLITE_DIR: ("historical_livneh_unsplit", "wgen_product_a", "wgen_product_a_s12"),
    paths.CDEC15: ("historical_livneh_unsplit",),
}
#: the period of every series: October 1921 (CalLite's first month) to September 2018
START, END = "1921-10-01", "2018-09-30"
#: acre-feet in one cfs-day
AF_PER_CFS_DAY = 1.9834711
#: CalLite file, column order and number format of each set (the original study's wrapper)
CALLITE_FILES = {
    "9unimp": ("SACSMA_Inflow_CalLite_9.txt",
               ("BearRiver", "CacheCreek", "CalaverasRiver", "ChowchillaRiver", "CosumnesRiver",
                "FresnoRiver", "MokelumneRiver", "PutahCreek", "StonyCreek"), "%14.6f"),
    "11obs": ("SACSMA_Inflow_CalLite_11.txt",
              ("AMF", "BLB", "BND", "FTO", "MRC", "SHA", "SJF", "SNS", "TLG", "TNL", "YRS"),
              "%10.6f"),
}
#: the twelve rim inflows: (12rim basin, CalLite node, unit)
CALLITE_12 = (("FOL_I", "I_FOLSM", "CFS"), ("LK_MC", "I_MCLRE", "TAF"), ("N_MEL", "I_MELON", "TAF"),
              ("MILLE", "I_MLRTN", "CFS"), ("PRD_C", "I_MOKELUMNE", "CFS"),
              ("N_HOG", "I_NHGAN", "TAF"), ("OROVI", "I_OROVL", "CFS"), ("DPR_I", "I_PEDRO", "TAF"),
              ("SHAST", "I_SHSTA", "CFS"), ("TRINI", "I_TRNTY", "CFS"), ("SMART", "I_YUBA", "CFS"),
              ("WKYTN", "I_WKYTN", "CFS"))
CALLITE_12_FILE = "SACSMA_Inflow_CalLite_12.txt"
#: the DSS F-part of the CalLite inflow paths
CALLITE_VERSION = "2020D09E"


def simulate(domain: str, forcing: str, data_dir: str | Path = "data") -> pd.DataFrame:
    """Daily flow of every watershed of a calibration set under a forcing:
    ``[date, basin, flow_mm]`` from :data:`START` to :data:`END`."""
    from .model import run_basins

    flow = run_basins(data_dir=data_dir, domain=domain, product=forcing).loc[START:END]
    return flow.melt(ignore_index=False, var_name="basin", value_name="flow_mm").reset_index()


def _monthly_mm(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly totals (mm), months by water year: index ``(Year, Month)``, one column per
    watershed."""
    d = pd.to_datetime(daily["date"])
    m = daily.assign(Year=d.dt.year + (d.dt.month >= 10), Month=d.dt.month)
    t = m.pivot_table(index=["Year", "Month"], columns="basin", values="flow_mm", aggfunc="sum")
    order = sorted(t.index, key=lambda ym: (ym[0], (ym[1] - 10) % 12))
    return t.loc[order]


def _rows(index, values: np.ndarray, fmt: str, lead: str = "%4.0f, %2.0f, ") -> list[str]:
    return [((lead % ym) if lead else "") + ", ".join(fmt % v for v in row)
            for ym, row in zip(index, values, strict=True)]


def callite_files(forcing: str, data_dir: str | Path = "data") -> dict[str, str]:
    """The three CalLite input files of a forcing: ``{file name: text}``."""
    out = {}
    for domain, (name, cols, fmt) in CALLITE_FILES.items():
        t = _monthly_mm(simulate(domain, forcing, data_dir))[list(cols)]
        lines = ["Year, Month, " + ", ".join(cols)] + _rows(t.index, t.to_numpy(), fmt)
        out[name] = "\n".join(lines) + "\n"
    t = _monthly_mm(simulate("12rim", forcing, data_dir))
    area = load_basin_area(data_dir, domain="12rim").set_index("basin")["area_mi2"]
    days = np.array([calendar.monthrange(y - (m >= 10), m)[1] for y, m in t.index], float)
    cols = []
    for basin, _, unit in CALLITE_12:
        cfs = mmday_to_cfs(t[basin].to_numpy() / days, float(area[basin]))
        cols.append(cfs * days * AF_PER_CFS_DAY / 1000.0 if unit == "TAF" else cfs)
    wy, m = t.index[0]                  # the end of the first month, e.g. 31OCT1921 2400
    y = wy - (m >= 10)
    stamp = f"{calendar.monthrange(y, m)[1]}{calendar.month_abbr[m].upper()}{y} 2400"
    head = [",".join([stamp] * len(CALLITE_12)),
            ",".join(f"/CALLITE/{node}/FLOW-INFLOW//1MON/{CALLITE_VERSION}/"
                     for _, node, _ in CALLITE_12),
            ",".join(unit for _, _, unit in CALLITE_12)]
    out[CALLITE_12_FILE] = "\n".join(head + _rows(t.index, np.column_stack(cols), "%1.2f",
                                                  lead="")) + "\n"
    return out


def cdec15_files(forcing: str, data_dir: str | Path = "data") -> dict[str, str]:
    """The daily flow of the 15 CDEC watersheds: ``{"flow_daily.csv": text}``, columns
    ``date, basin, flow_mm, flow_cfs``."""
    d = simulate(paths.CDEC15, forcing, data_dir)
    area = load_basin_area(data_dir, domain=paths.CDEC15).set_index("basin")["area_mi2"]
    d["flow_cfs"] = mmday_to_cfs(d["flow_mm"], d["basin"].map(area)).round(3)
    d["flow_mm"] = d["flow_mm"].round(6)
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    return {"flow_daily.csv": d.to_csv(index=False, lineterminator="\n")}


def files(app: str, forcing: str, data_dir: str | Path = "data") -> dict[str, str]:
    """The files of the product ``app`` (``callite`` or ``15cdec``) under one forcing:
    ``{file name: text}``."""
    if forcing not in FORCINGS[app]:
        raise ValueError(f"{app} carries {FORCINGS[app]}, not {forcing!r}")
    return (callite_files if app == paths.CALLITE_DIR else cdec15_files)(forcing, data_dir)


def write(app: str, forcings=None, data_dir: str | Path = "data",
          artifacts_dir: str | Path = "artifacts", log=print) -> list[Path]:
    """Write the product ``app`` for ``forcings`` (default: every forcing it carries); returns
    the files written."""
    written = []
    for forcing in forcings or FORCINGS[app]:
        out = paths.product(artifacts_dir, forcing=forcing, app=app)
        for name, text in files(app, forcing, data_dir).items():
            out.mkdir(parents=True, exist_ok=True)
            (out / name).write_text(text, newline="\n")
            written.append(out / name)
        log(f"product {app}: {forcing} -> {out}")
    return written
