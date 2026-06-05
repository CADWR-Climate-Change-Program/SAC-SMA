"""Parameter definitions and loaders for the pre-done GA calibration.

The archived ``GA OPTIMUM PARAMETER`` table (``sacramento_ga_15cdec_pool.txt``)
already expands all 31 parameters to every one of the 6033 HRUs, so a forward
run just looks up each HRU's row by its ``lat_lon`` key.  It is ingested once
into ``data/params/`` (see :mod:`sacsma.dataprep`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Column order of the GA OPTIMUM PARAMETER table (after HRU_Lat, HRU_Lon).
GA_OPT_COLUMNS = [
    "lat", "lon",
    "Kpet",
    "uztwm", "uzfwm", "lztwm", "lzfpm", "lzfsm",
    "uzk", "lzpk", "lzsk",
    "zperc", "rexp", "pfree", "pctim", "adimp", "riva", "side", "rserv",
    "SCF", "PXTEMP", "MFMAX", "MFMIN", "UADJ", "MBASE", "TIPM", "PLWHC", "NMF", "DAYGM",
    "Nres", "Kres", "Velo", "Diff",
]

# Sub-vectors in the order each physics module expects them.
_SMA_COLS = [
    "uztwm", "uzfwm", "lztwm", "lzfpm", "lzfsm",
    "uzk", "lzpk", "lzsk",
    "zperc", "rexp", "pfree", "pctim", "adimp", "riva", "side", "rserv",
]
_SNOW_COLS = [
    "SCF", "PXTEMP", "MFMAX", "MFMIN", "UADJ",
    "MBASE", "TIPM", "PLWHC", "NMF", "DAYGM",
]
_ROUT_COLS = ["Nres", "Kres", "Velo", "Diff"]


def latlon_key(lat: float, lon: float) -> str:
    """Canonical 6-decimal ``lat_lon`` join key (matches meteo filenames)."""
    return f"{lat:.6f}_{lon:.6f}"


def load_ga_optimum(path: str | Path) -> pd.DataFrame:
    """Load the per-HRU GA OPTIMUM PARAMETER table.

    Returns a DataFrame with :data:`GA_OPT_COLUMNS` columns and a ``key``
    index (``lat_lon``) for fast per-HRU lookup.
    """
    path = Path(path)
    rows: list[list[float]] = []
    in_table = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "HRU_Lat" in line and "HRU_Lon" in line:
                in_table = True
                continue
            if not in_table:
                continue
            parts = line.split()
            if len(parts) != len(GA_OPT_COLUMNS):
                # blank line or non-data -> end of table once we've started
                if rows:
                    break
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                if rows:
                    break
                continue
    if not rows:
        raise ValueError(f"No GA OPTIMUM PARAMETER rows found in {path}")
    df = pd.DataFrame(rows, columns=GA_OPT_COLUMNS)
    df["key"] = [latlon_key(la, lo) for la, lo in zip(df["lat"], df["lon"])]
    return df.set_index("key")


def sma_par(row) -> np.ndarray:
    """16-element SAC-SMA parameter vector from a GA table row."""
    return np.array([row[c] for c in _SMA_COLS], dtype=float)


def snow_par(row) -> np.ndarray:
    """10-element SNOW-17 parameter vector from a GA table row."""
    return np.array([row[c] for c in _SNOW_COLS], dtype=float)


def routing_par(row) -> np.ndarray:
    """4-element Lohmann routing parameter vector from a GA table row."""
    return np.array([row[c] for c in _ROUT_COLS], dtype=float)


def kpet(row) -> float:
    """Hamon coefficient from a GA table row."""
    return float(row["Kpet"])
