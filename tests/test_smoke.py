"""Shape/finiteness smoke tests for the physics ports (no big data needed)."""

import numpy as np

from sacsma.model import run_hru
from sacsma.pet import hamon_pet
from sacsma.routing import lohmann
from sacsma.sma import sac_sma
from sacsma.snow17 import snow17

GA_ROW = {
    "Kpet": 0.9,
    "uztwm": 476.2, "uzfwm": 809.0, "lztwm": 88.2, "lzfpm": 301.9, "lzfsm": 3631.5,
    "uzk": 0.56, "lzpk": 0.143, "lzsk": 0.0295,
    "zperc": 44.1, "rexp": 9.42, "pfree": 0.247,
    "pctim": 0.0, "adimp": 0.088, "riva": 0.0, "side": 0.0, "rserv": 0.4,
    "SCF": 1.0, "PXTEMP": 0.0, "MFMAX": 0.905, "MFMIN": 0.05, "UADJ": 0.063,
    "MBASE": 0.0, "TIPM": 1.0, "PLWHC": 0.02, "NMF": 0.3, "DAYGM": 0.134,
    "Nres": 5.7, "Kres": 0.87, "Velo": 3.9, "Diff": 3476.0,
}


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    prcp = rng.gamma(0.5, 5.0, n)
    tavg = 10.0 + 12.0 * np.sin(np.arange(n) * 2 * np.pi / 365.0) + rng.normal(0, 2, n)
    doy = (np.arange(n) % 365 + 1).astype(np.int64)
    is_leap = np.zeros(n, dtype=np.int64)
    return prcp, tavg, doy, is_leap


def test_pet():
    _, tavg, doy, _ = _synthetic()
    pet = hamon_pet(tavg, doy, 40.0, 0.9)
    assert pet.shape == tavg.shape
    assert np.all(np.isfinite(pet))
    assert np.all(pet >= 0)


def test_snow17():
    prcp, tavg, doy, is_leap = _synthetic()
    outflow, melt, swe, state, intot = snow17(prcp, tavg, doy, is_leap, 1500.0, list(
        GA_ROW[k] for k in ("SCF", "PXTEMP", "MFMAX", "MFMIN", "UADJ",
                            "MBASE", "TIPM", "PLWHC", "NMF", "DAYGM")))
    assert outflow.shape == prcp.shape
    assert np.all(np.isfinite(outflow))
    assert np.all(swe >= 0)
    assert state.shape == (4,)


def test_sacsma():
    prcp, tavg, doy, _ = _synthetic()
    pet = hamon_pet(tavg, doy, 40.0, 0.9)
    par = [GA_ROW[k] for k in ("uztwm", "uzfwm", "lztwm", "lzfpm", "lzfsm",
                               "uzk", "lzpk", "lzsk", "zperc", "rexp", "pfree",
                               "pctim", "adimp", "riva", "side", "rserv")]
    surf, base, tet, state = sac_sma(pet, prcp, par)
    assert surf.shape == prcp.shape
    assert np.all(np.isfinite(surf)) and np.all(np.isfinite(base))
    assert np.all(surf >= 0) and np.all(base >= 0)
    assert state.shape == (6,)


def test_lohmann_conserves_to_unit_uh():
    n = 200
    direct = np.zeros(n)
    base = np.zeros(n)
    direct[10] = 100.0  # impulse
    runoff, baseflow = lohmann(direct, base, 50000.0, [5.7, 0.87, 3.9, 3476.0], 0)
    assert runoff.shape == (n,)
    assert np.all(np.isfinite(runoff))
    # UH is mass-conserving up to truncation; routed mass <= input mass.
    assert runoff.sum() <= 100.0 + 1e-6


def test_run_hru():
    prcp, tavg, doy, is_leap = _synthetic()
    q = run_hru(prcp, tavg, doy, is_leap, lat=40.0, elev=1500.0,
                flowlen=50000.0, ga_row=GA_ROW)
    assert q.shape == prcp.shape
    assert np.all(np.isfinite(q)) and np.all(q >= 0)
