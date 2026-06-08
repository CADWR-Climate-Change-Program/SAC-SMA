"""sacsma — faithful NumPy/Numba port of the Sacramento Soil Moisture
Accounting model (Wi & Steinschneider; CA DWR San Joaquin Watershed Studies).

Public API:
  * :func:`hamon_pet`, :func:`snow17`, :func:`sac_sma`, :func:`lohmann`
    — the four faithful physics ports.
  * :func:`run_hru`, :func:`run_basin` — coupled pipeline / basin driver.
  * :func:`load_ga_optimum` — load the archived per-HRU calibration.
  * :mod:`sacsma.metrics`, :mod:`sacsma.calibrate` — eval + GA scaffold.
"""

from __future__ import annotations

from .metrics import kge, nse, pbias, pearson
from .model import run_basin, run_hru
from .parameters import load_ga_optimum
from .pet import hamon_pet
from .routing import lohmann
from .sma import sac_sma
from .snow17 import snow17

__version__ = "0.1.0"

__all__ = [
    "hamon_pet",
    "snow17",
    "sac_sma",
    "lohmann",
    "run_hru",
    "run_basin",
    "load_ga_optimum",
    "kge",
    "nse",
    "pbias",
    "pearson",
    "__version__",
]
