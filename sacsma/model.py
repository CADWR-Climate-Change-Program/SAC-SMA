"""Coupled per-HRU pipeline and basin aggregation driver.

Per HRU: Hamon PET -> SNOW-17 -> SAC-SMA -> Lohmann routing.  Basin flow is
the area-weighted sum of routed HRU flow (mm/day) at the gauge.

This reproduces the distributed SAC-SMA forward run from the archived GA
optimum (the run-from-pre-done-calibration path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import parameters as P
from ._compat import HAVE_NUMBA, njit
from .io import (
    DEFAULT_DOMAIN,
    DEFAULT_FORCING,
    doy_and_leap,
    forcing_path,
    load_forcing,
    load_params,
)
from .pet import _hamon_core, hamon_pet
from .routing import _lohmann_core_nb, lohmann
from .sma import _sacsma_core, sac_sma
from .snow17 import _snow17_core, snow17

if HAVE_NUMBA:
    from numba import prange

    @njit(parallel=True)
    def _basin_kernel(
        prcp_cells, tavg_cells, cell_idx, doy_f, doy_i, is_leap,
        lat_rad, elev, flowlen, is_outlet,
        kpet, snow_par, sma_par, rout_par, wnorm,
    ):
        """Area-weighted routed basin flow, fanned across HRUs with ``prange``.

        Each HRU runs the full PET -> Snow-17 -> SAC-SMA -> Lohmann pipeline using the
        same njit cores as the serial path; ``total`` is a flat 1-D array reduction so
        memory stays at one time series per thread.  ``prcp_cells``/``tavg_cells`` are the
        shared ``(n_cell, T)`` forcing; ``cell_idx[h]`` selects HRU ``h``'s grid cell."""
        T = prcp_cells.shape[1]
        nh = cell_idx.shape[0]
        total = np.zeros(T)
        for h in prange(nh):
            c = cell_idx[h]
            pr = prcp_cells[c]
            tv = tavg_cells[c]
            pet = _hamon_core(tv, doy_f, lat_rad[h], kpet[h])
            snow_init = np.zeros(4)
            eff = _snow17_core(pr, tv, doy_i, is_leap, elev[h], snow_par[h], snow_init)[0]
            sma_init = np.empty(6)
            sma_init[0] = 0.0; sma_init[1] = 0.0; sma_init[2] = 100.0
            sma_init[3] = 100.0; sma_init[4] = 100.0; sma_init[5] = 0.0
            surf, base, _t, _s = _sacsma_core(pet, eff, sma_par[h], sma_init)
            runoff = _lohmann_core_nb(surf, base, flowlen[h], rout_par[h], is_outlet[h])
            total += wnorm[h] * runoff
        return total

    @njit(parallel=True)
    def _local_runoff_kernel(
        prcp_cells, tavg_cells, cell_idx, doy_f, doy_i, is_leap,
        lat_rad, elev, kpet, snow_par, sma_par,
    ):
        """Per-HRU **local runoff** depth ``surf + base`` (mm/day, un-routed), one row
        per cell, fanned across cores.  Each ``prange`` iteration writes its own row, so
        the result is **bit-exact** vs the serial path (no cross-HRU reduction).  Backs
        the local-runoff aggregation in :func:`sacsma.calsim.catchments.run_calsim`."""
        nk = cell_idx.shape[0]
        T = prcp_cells.shape[1]
        out = np.empty((nk, T))
        for h in prange(nk):
            c = cell_idx[h]
            pr = prcp_cells[c]
            tv = tavg_cells[c]
            pet = _hamon_core(tv, doy_f, lat_rad[h], kpet[h])
            snow_init = np.zeros(4)
            eff = _snow17_core(pr, tv, doy_i, is_leap, elev[h], snow_par[h], snow_init)[0]
            sma_init = np.empty(6)
            sma_init[0] = 0.0; sma_init[1] = 0.0; sma_init[2] = 100.0
            sma_init[3] = 100.0; sma_init[4] = 100.0; sma_init[5] = 0.0
            surf, base, _t, _s = _sacsma_core(pet, eff, sma_par[h], sma_init)
            for t in range(T):
                out[h, t] = surf[t] + base[t]
        return out
else:  # pragma: no cover - Numba absent
    _basin_kernel = None
    _local_runoff_kernel = None


def default_is_outlet(flowlen: float) -> int:
    """Default outlet rule: the HRU at the watershed outlet has flowlen 0.

    NOTE: confirm this against the MATLAB driver's outlet/flowlen convention.
    """
    return 1 if flowlen == 0.0 else 0


def _pet_and_recession(tavg, doy, lat, ga_row):
    """PET series and the optional seasonal ``(uzk, lzpk, lzsk)`` override.

    Static rows: PET = ``hamon_pet(..., Kpet)`` and ``recession=None`` (the
    bit-identical reference path).  Seasonal rows (day-of-year harmonics): PET =
    ``Kpet(doy) * rawPET`` and the three recession rates become per-day arrays,
    selecting the seasonal SAC-SMA path (:func:`sacsma.sma._sacsma_core_seasonal`).
    """
    if P.is_seasonal(ga_row):
        pet = P.kpet_series(ga_row, doy) * hamon_pet(tavg, doy, lat, 1.0)
        return pet, P.recession_series(ga_row, doy)
    return hamon_pet(tavg, doy, lat, P.kpet(ga_row)), None


def run_hru(
    prcp: np.ndarray,
    tavg: np.ndarray,
    doy: np.ndarray,
    is_leap: np.ndarray,
    *,
    lat: float,
    elev: float,
    flowlen: float,
    ga_row,
    is_outlet: int | None = None,
) -> np.ndarray:
    """Run the full coupled pipeline for one HRU; return routed flow (mm/day)."""
    pet, recession = _pet_and_recession(tavg, doy, lat, ga_row)
    eff_p = snow17(prcp, tavg, doy, is_leap, elev, P.snow_par(ga_row))[0]
    surf, base, _tet, _state = sac_sma(pet, eff_p, P.sma_par(ga_row), recession=recession)
    if is_outlet is None:
        is_outlet = default_is_outlet(flowlen)
    runoff, _baseflow = lohmann(surf, base, flowlen, P.routing_par(ga_row), is_outlet)
    return runoff


def run_hru_components(
    prcp: np.ndarray,
    tavg: np.ndarray,
    doy: np.ndarray,
    is_leap: np.ndarray,
    *,
    lat: float,
    elev: float,
    ga_row,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-HRU SMA outputs ``(surf, base)`` (mm/day), **before** channel routing.

    PET -> SNOW-17 -> SAC-SMA.  Returned separately so an arbitrary sub-catchment
    (e.g. a CalSim node) can either area-weight the un-routed runoff (``surf+base``)
    or Lohmann-route with a catchment-specific ``flowlen``.  See :mod:`sacsma.calsim.catchments`.
    """
    pet, recession = _pet_and_recession(tavg, doy, lat, ga_row)
    eff_p = snow17(prcp, tavg, doy, is_leap, elev, P.snow_par(ga_row))[0]
    surf, base, _tet, _state = sac_sma(pet, eff_p, P.sma_par(ga_row), recession=recession)
    return surf, base


def run_hru_local(
    prcp: np.ndarray,
    tavg: np.ndarray,
    doy: np.ndarray,
    is_leap: np.ndarray,
    *,
    lat: float,
    elev: float,
    ga_row,
) -> np.ndarray:
    """Local runoff depth (mm/day) generated by one HRU (``surf + base``)."""
    surf, base = run_hru_components(prcp, tavg, doy, is_leap, lat=lat, elev=elev, ga_row=ga_row)
    return surf + base


def _comp_key(domain: str, key: str, lat: float, elev: float, ga_row):
    """Hashable identity of everything feeding the SMA components ``(surf, base)``: the
    ``(domain, cell)`` (which fix the forcing) plus lat/elev and the PET/Snow-17/SAC-SMA
    params.  Routing params are deliberately excluded — they only affect the downstream
    Lohmann routing, not the components — so the cache is shared between the routed
    (``run_basin``) and local (``run_calsim``) aggregations."""
    return (domain, key, float(lat), float(elev), float(P.kpet(ga_row)),
            tuple(float(x) for x in P.snow_par(ga_row)),
            tuple(float(x) for x in P.sma_par(ga_row)))


def run_hru_components_cached(
    comp_cache: dict | None,
    domain: str,
    key: str,
    prcp: np.ndarray,
    tavg: np.ndarray,
    doy: np.ndarray,
    is_leap: np.ndarray,
    *,
    lat: float,
    elev: float,
    ga_row,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`run_hru_components` with an optional cross-builder cache.

    Within a single cross-compare the same per-cell PET->Snow-17->SAC-SMA physics is needed
    by BOTH the basin anchor (``run_basin``, routed to the CDEC gauge) and the per-catchment
    view (``run_calsim``, local runoff at the CalSim node).  Passing one ``comp_cache`` dict to
    both computes each cell's components **once**.  The cache key (:func:`_comp_key`) includes
    the params, so a 9unimp cell shared between watersheds with different params is computed
    per param set — numerically identical to the un-cached path.  Returned arrays are treated
    as read-only by all consumers (Lohmann routing and area-weighting only read them)."""
    if comp_cache is None:
        return run_hru_components(prcp, tavg, doy, is_leap, lat=lat, elev=elev, ga_row=ga_row)
    ck = _comp_key(domain, key, lat, elev, ga_row)
    sb = comp_cache.get(ck)
    if sb is None:
        sb = run_hru_components(prcp, tavg, doy, is_leap, lat=lat, elev=elev, ga_row=ga_row)
        comp_cache[ck] = sb
    return sb


def run_basin(
    basin: str,
    *,
    data_dir: str | Path | None = "data",
    domain: str = DEFAULT_DOMAIN,
    start: str | None = None,
    end: str | None = None,
    progress: bool = False,
    forcing: DomainForcing | None = None,
    comp_cache: dict | None = None,
    parallel: bool = False,
    product: str = DEFAULT_FORCING,
    params: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forward-simulate one basin from the GA optimum.

    ``data_dir`` points at the organized ``data/`` store and the ``domain``
    forcing store is used.  ``domain`` selects the application/calibration set
    (``"15cdec"`` default, or one of the CalSim/CalLite domains); ``product``
    selects the forcing store (default: the historical Livneh-unsplit grid —
    e.g. ``"wgen_product_a"`` for the WGEN historical-parallel sequence).

    For multi-basin runs, build the domain forcing once with
    :func:`load_domain_forcing` and pass it as ``forcing`` so the ~900 MB/var
    read happens a single time across all basins.

    ``params`` substitutes an alternate parameter table for the domain's
    archived GA optimum (same columns as ``ga_optimum.csv``; keyed by ``key``
    with an optional per-basin ``basin`` column — the dPL exports use this).
    Everything else, including the physics, is unchanged.

    Returns DataFrame[date, flow] of area-weighted gauge flow (mm/day).
    """
    if forcing is None and (
        data_dir is None
        or not forcing_path(data_dir, domain, product).exists()
    ):
        raise FileNotFoundError(
            f"forcing store not found: {forcing_path(data_dir or 'data', domain, product)}"
        )
    return _run_basin_native(
        basin, data_dir=data_dir, domain=domain, start=start, end=end,
        progress=progress, forcing=forcing, comp_cache=comp_cache, parallel=parallel,
        product=product, params=params,
    )


@dataclass
class DomainForcing:
    """Domain-wide forcing read into memory once, reusable across basins.

    ``prcp``/``tavg`` are the full ``(n_cells, n_time)`` arrays (stored as
    float32 to halve memory); ``pos`` maps a grid-cell ``key`` to its row.
    ``doy``/``is_leap`` are precomputed for the (possibly time-sliced) ``dates``.
    """

    pos: dict[str, int]
    prcp: np.ndarray
    tavg: np.ndarray
    dates: pd.DatetimeIndex
    doy: np.ndarray
    is_leap: np.ndarray
    _f64: dict = field(default_factory=dict, repr=False, compare=False)

    def forcing_f64(self) -> tuple[np.ndarray, np.ndarray]:
        """``(prcp, tavg)`` as C-contiguous float64, converted **once** and cached.

        The physics cores run in float64 (the serial path up-converts each row), so
        the parallel kernel needs the whole store in float64.  Converting once here
        (vs per basin) keeps the cost off the per-basin path; the float32 originals
        are retained so serial-only runs keep their smaller footprint."""
        if "prcp" not in self._f64:
            self._f64["prcp"] = np.ascontiguousarray(self.prcp, dtype=np.float64)
            self._f64["tavg"] = np.ascontiguousarray(self.tavg, dtype=np.float64)
        return self._f64["prcp"], self._f64["tavg"]


def load_domain_forcing(
    data_dir: str | Path,
    *,
    domain: str = DEFAULT_DOMAIN,
    start: str | None = None,
    end: str | None = None,
    product: str = DEFAULT_FORCING,
) -> DomainForcing:
    """Read the whole forcing store into memory ONCE (reuse across basins).

    The store is zlib-compressed; xarray/HDF5 fancy-indexing ~2000 non-contiguous
    keys re-decompresses overlapping chunks per key and takes minutes, whereas a
    single contiguous read of each variable is ~4s and NumPy row-indexing is
    instant.  For multi-basin runs, build this once and pass it to every
    :func:`run_basin` call so the ~900 MB/var read happens a single time.
    """
    ds = load_forcing(data_dir, domain=domain, product=product)
    try:
        if start is not None or end is not None:
            ds = ds.sel(time=slice(start, end))
        dates = pd.DatetimeIndex(ds["time"].values)
        doy, is_leap = doy_and_leap(dates)
        prcp = ds["prcp"].values
        tavg = ds["tavg"].values
        pos = {str(k): i for i, k in enumerate(ds["key"].values)}
    finally:
        ds.close()
    return DomainForcing(pos=pos, prcp=prcp, tavg=tavg, dates=dates, doy=doy, is_leap=is_leap)


def _run_basin_native(
    basin: str,
    *,
    data_dir: str | Path | None = None,
    domain: str = DEFAULT_DOMAIN,
    start: str | None = None,
    end: str | None = None,
    progress: bool = False,
    forcing: DomainForcing | None = None,
    comp_cache: dict | None = None,
    parallel: bool = False,
    product: str = DEFAULT_FORCING,
    params: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Native-path basin run from the organized ``data/`` artifacts.

    Forcing is the domain-wide grid-cell store; HRU attributes (elev, flowlen,
    area_weight, lat) come from the HRU table.  Each HRU pulls its grid cell's
    forcing by ``key``.  Pass a preloaded ``forcing`` (from
    :func:`load_domain_forcing`) to avoid re-reading the store per basin.

    ``parallel=True`` fans the HRUs across cores via :func:`_basin_kernel` (Numba
    ``prange``); the result matches the serial path to floating tolerance (the
    only difference is reduction order).  It bypasses ``comp_cache``.
    """
    from .io import load_hru_table

    dd: str | Path = data_dir if data_dir is not None else "data"
    if forcing is None:
        forcing = load_domain_forcing(dd, domain=domain, start=start, end=end, product=product)
    params_df = params if params is not None else load_params(dd, domain=domain)
    # per-watershed calibrations (e.g. 9unimp) repeat shared cells with different
    # params per basin; filter to this basin before indexing by key.
    if "basin" in params_df.columns:
        params_df = params_df[params_df["basin"] == basin]
    params_df = params_df.set_index("key")
    # Seasonal (day-of-year harmonic) params run only on the serial path — the
    # numba kernels take scalar params.  Fall back to serial rather than
    # silently dropping the seasonality.
    if parallel and any(str(c).endswith("_asin") for c in params_df.columns):
        parallel = False
    hrus = load_hru_table(dd, domain=domain)
    sub = hrus[hrus["basin"] == basin].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No HRUs for basin {basin} in {data_dir}")
    # Source HRU weights are PER-BASIN PERCENTAGES (sum to 100); normalize to
    # area fractions (sum to 1) before area-weighting the routed flow.
    wnorm = sub["area_weight"].to_numpy(dtype=float)
    wnorm = wnorm / wnorm.sum()

    dates, doy, is_leap = forcing.dates, forcing.doy, forcing.is_leap

    if parallel and _basin_kernel is not None:
        total = _run_basin_parallel(sub, params_df, forcing, wnorm)
        return pd.DataFrame({"date": dates, "flow": total})

    total = np.zeros(len(dates))
    n = len(sub)
    for i, hru in enumerate(sub.itertuples(index=False)):
        if progress and (i % 200 == 0):
            print(f"  {basin}: HRU {i + 1}/{n}", flush=True)
        c = forcing.pos[hru.key]
        ga_row = params_df.loc[hru.key]
        # PET -> Snow-17 -> SAC-SMA (cached across the anchor/per-catchment builds), then
        # Lohmann-route to the gauge — equivalent to run_hru() but reusing shared components.
        surf, base = run_hru_components_cached(
            comp_cache, domain, hru.key, forcing.prcp[c], forcing.tavg[c], doy, is_leap,
            lat=float(hru.lat), elev=float(hru.elev), ga_row=ga_row,
        )
        is_outlet = default_is_outlet(float(hru.flowlen))
        runoff, _baseflow = lohmann(surf, base, float(hru.flowlen), P.routing_par(ga_row), is_outlet)
        total += wnorm[i] * runoff
    return pd.DataFrame({"date": dates, "flow": total})


def _run_basin_parallel(sub, params_df, forcing: DomainForcing, wnorm: np.ndarray) -> np.ndarray:
    """Build the flat per-HRU array bundle and call :func:`_basin_kernel`.

    All per-HRU pandas lookups happen here (vectorized, once); the kernel sees only
    contiguous NumPy arrays.  The full ``(n_cell, T)`` forcing is passed by reference
    with a per-HRU ``cell_idx`` so it is never duplicated.
    """
    keys = sub["key"].to_numpy()
    cell_idx = np.fromiter((forcing.pos[k] for k in keys), dtype=np.int64, count=len(keys))
    pr = params_df.loc[keys]
    kpet = np.ascontiguousarray(pr["Kpet"].to_numpy(dtype=float))
    snow_par = np.ascontiguousarray(pr[list(P._SNOW_COLS)].to_numpy(dtype=float))
    sma_par = np.ascontiguousarray(pr[list(P._SMA_COLS)].to_numpy(dtype=float))
    rout_par = np.ascontiguousarray(pr[list(P._ROUT_COLS)].to_numpy(dtype=float))
    lat_rad = np.ascontiguousarray(np.deg2rad(sub["lat"].to_numpy(dtype=float)))
    elev = np.ascontiguousarray(sub["elev"].to_numpy(dtype=float))
    flowlen = np.ascontiguousarray(sub["flowlen"].to_numpy(dtype=float))
    is_outlet = (flowlen == 0.0).astype(np.int64)
    prcp_cells, tavg_cells = forcing.forcing_f64()
    doy_f = forcing.doy.astype(np.float64)
    doy_i = forcing.doy.astype(np.int64)
    is_leap = forcing.is_leap.astype(np.int64)
    return _basin_kernel(
        prcp_cells, tavg_cells, cell_idx, doy_f, doy_i, is_leap,
        lat_rad, elev, flowlen, is_outlet,
        kpet, snow_par, sma_par, rout_par, np.ascontiguousarray(wnorm),
    )


def run_local_runoff_parallel(keys, meta, params, forcing: DomainForcing) -> np.ndarray:
    """Local runoff depth (``surf + base``, mm/day) for ``keys``, fanned across cores.

    Returns an ``(len(keys), T)`` matrix, row ``i`` = cell ``keys[i]``'s un-routed runoff,
    **bit-exact** vs computing each cell with :func:`run_hru_local`.  ``meta`` is the cell
    table indexed by ``key`` (lat/elev); ``params`` is the GA table indexed by ``key``.
    Used by :func:`sacsma.calsim.catchments.run_calsim` (``parallel=True``)."""
    keys = np.asarray(keys)
    cell_idx = np.fromiter((forcing.pos[k] for k in keys), dtype=np.int64, count=len(keys))
    pr = params.loc[keys]
    kpet = np.ascontiguousarray(pr["Kpet"].to_numpy(dtype=float))
    snow_par = np.ascontiguousarray(pr[list(P._SNOW_COLS)].to_numpy(dtype=float))
    sma_par = np.ascontiguousarray(pr[list(P._SMA_COLS)].to_numpy(dtype=float))
    mm = meta.loc[keys]
    lat_rad = np.ascontiguousarray(np.deg2rad(mm["lat"].to_numpy(dtype=float)))
    elev = np.ascontiguousarray(mm["elev"].to_numpy(dtype=float))
    prcp_cells, tavg_cells = forcing.forcing_f64()
    doy_f = forcing.doy.astype(np.float64)
    doy_i = forcing.doy.astype(np.int64)
    is_leap = forcing.is_leap.astype(np.int64)
    return _local_runoff_kernel(
        prcp_cells, tavg_cells, cell_idx, doy_f, doy_i, is_leap,
        lat_rad, elev, kpet, snow_par, sma_par,
    )
