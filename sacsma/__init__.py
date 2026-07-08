"""sacsma — distributed SAC-SMA (Hamon PET / Snow-17 / SAC-SMA / Lohmann)
for California watersheds (Wi & Steinschneider; CA DWR Watershed Studies).

Public API:
  * :func:`hamon_pet`, :func:`snow17`, :func:`sac_sma`, :func:`lohmann`
    — the four physics modules.
  * :func:`run_hru`, :func:`run_basin` — coupled pipeline / basin driver.
  * :mod:`sacsma.metrics` — skill metrics.
  * :mod:`sacsma.cdec15` — the 15-CDEC application (daily gage calibration).
  * :mod:`sacsma.calsim` — the CalSim/CalLite application (9unimp/11obs/12rim
    monthly calibrations + CalSim3 cross-compare).
"""

from __future__ import annotations

from .metrics import kge, nse, pbias, pearson
from .model import run_basin, run_hru
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
    "kge",
    "nse",
    "pbias",
    "pearson",
    "__version__",
]
