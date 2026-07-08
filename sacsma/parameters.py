"""Parameter-vector extraction from the per-HRU GA-optimum tables.

Each domain's ``ga_optimum`` table carries all 31 calibrated parameters per
HRU (keyed by the ``lat_lon`` grid-cell ``key``); these helpers slice one row
into the vectors each physics module expects.
"""

from __future__ import annotations

import numpy as np

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
