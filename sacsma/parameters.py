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


# ---------------------------------------------------------------------------
# Seasonal (day-of-year harmonic) parameters — the dPL dynamic-parameter path.
# A seasonal row carries, per seasonal parameter ``P``, two extra columns
# ``P_asin`` / ``P_acos`` (the base ``P`` column stays the annual MEAN):
#     P(doy) = clip(P_mean + P_asin*sin(w*doy) + P_acos*cos(w*doy), lo, hi)
# with ``w = 2*pi/365``.  Bounds MIRROR sacsma.dpl.config.BOUNDS (kept here so
# the core stays torch/dpl-free — dpl imports core, never the reverse).  The
# frozen model reconstructs the same series so seasonal params score exactly.
# ---------------------------------------------------------------------------
SEASONAL_PARAMS: tuple[str, ...] = ("Kpet", "uzk", "lzpk", "lzsk")
_SEASONAL_BOUNDS: dict[str, tuple[float, float]] = {
    "Kpet": (0.4, 2.5), "uzk": (0.01, 0.99), "lzpk": (0.01, 0.5), "lzsk": (0.003, 0.5),
}
_SEASONAL_OMEGA = 2.0 * np.pi / 365.0


def is_seasonal(row) -> bool:
    """True if a GA/dPL row carries seasonal harmonic coefficient columns."""
    return any(f"{p}_asin" in row for p in SEASONAL_PARAMS)


def _harmonic(row, name: str, doy: np.ndarray) -> np.ndarray:
    """Reconstruct ``name``'s per-day series from its mean + harmonic coeffs.

    Missing coefficient columns => zero amplitude (that parameter stays static),
    so a partially-seasonal row degrades gracefully to the static value.
    """
    mean = float(row[name])
    asin = float(row[f"{name}_asin"]) if f"{name}_asin" in row else 0.0
    acos = float(row[f"{name}_acos"]) if f"{name}_acos" in row else 0.0
    lo, hi = _SEASONAL_BOUNDS[name]
    v = mean + asin * np.sin(_SEASONAL_OMEGA * doy) + acos * np.cos(_SEASONAL_OMEGA * doy)
    return np.clip(v, lo, hi)


def kpet_series(row, doy) -> np.ndarray:
    """Per-day Hamon coefficient ``Kpet(doy)`` from a seasonal row."""
    return _harmonic(row, "Kpet", np.asarray(doy, dtype=float))


def recession_series(row, doy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-day ``(uzk, lzpk, lzsk)`` recession-rate series from a seasonal row."""
    doy = np.asarray(doy, dtype=float)
    return (_harmonic(row, "uzk", doy), _harmonic(row, "lzpk", doy),
            _harmonic(row, "lzsk", doy))
