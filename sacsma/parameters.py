"""The parameter columns of the per-HRU GA-optimum tables.

Each domain's ``ga_optimum`` table carries all 31 calibrated parameters per HRU (keyed by the
``lat_lon`` grid-cell ``key``): ``Kpet`` and the groups below, each in the order its physics
module reads it.
"""

from __future__ import annotations

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
