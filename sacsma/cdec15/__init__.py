"""The 15-CDEC application: daily streamflow for 15 CDEC reservoir watersheds.

Calibration setup (Wi & Steinschneider): a single **pooled** GA optimum across
all 15 basins (KGE objective, WY1989–2003), targeting the **daily observed CDEC
full-natural-flow** gage record (``load_gage``).  Diagnostics
(:mod:`sacsma.cdec15.plots`) report calibration and validation skill
separately either side of :data:`CAL_END`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..io import domain_dir, read_table

#: the modeling domain / calibration-set name (data files, forcing store).
DOMAIN = "15cdec"

#: the 15 CDEC reservoir watersheds (upstream-to-downstream Sacramento then San Joaquin/Tulare).
BASINS = [
    "SHA", "BND", "ORO", "YRS", "FOL", "MKM", "NHG", "NML",
    "TLG", "MRC", "MIL", "PNF", "TRM", "SCC", "ISB",
]

#: calibration period ends here (WY1989–WY2003); validation is everything after.
CAL_END = "2003-09-30"


def load_gage(data_dir: str | Path = "data", basin: str | None = None) -> pd.DataFrame:
    """Observed daily CDEC gage FNF (calibration target, mm/day); optionally one basin."""
    df = read_table(domain_dir(data_dir, DOMAIN) / "gage.csv")
    if basin is not None:
        df = df[df["basin"] == basin].reset_index(drop=True)
    return df
