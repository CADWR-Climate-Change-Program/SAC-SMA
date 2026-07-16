"""15cdec store -> torch tensors for the differentiable pipeline.

Wraps the existing loaders (``model.load_domain_forcing``, ``io.load_hru_table``,
``io.load_params``, ``cdec15.load_gage``) into a :class:`DomainTensors` bundle:
per-HRU static tensors, the basin aggregation matrix ``W`` (normalized
``area_weight`` per basin, exactly the ``model.py`` convention), and per-chunk
forcing gathers.  The big forcing arrays stay as CPU float32 NumPy (from
``DomainForcing``); each chunk is fancy-indexed to the HRU rows and moved to
the device on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..cdec15 import BASINS, CAL_END, load_gage
from ..io import domain_dir, load_hru_table, load_params
from ..model import DomainForcing, load_domain_forcing
from .config import PARAM_ORDER, validate_ga_optimum


@dataclass
class DomainTensors:
    dates: pd.DatetimeIndex
    doy: torch.Tensor          # (T,) float, on device
    is_leap: torch.Tensor      # (T,) bool, on device
    forcing: DomainForcing     # CPU float32 (cells, T)
    hrus: pd.DataFrame         # 7891 rows, reset index
    basins: tuple[str, ...]
    cell_idx: np.ndarray       # (N,) int rows into forcing arrays
    lat_rad: torch.Tensor      # (N,)
    elev: torch.Tensor         # (N,)
    flowlen: torch.Tensor      # (N,)
    W: torch.Tensor            # (B, N) basin aggregation (rows sum to 1)
    device: torch.device
    dtype: torch.dtype
    #: per-cell (n_cells, T) Tmin/Tmax for the Noah ET path (CPU float32, like
    #: forcing.prcp); None unless the domain has a per-cell tminmax sidecar
    #: (only 15cdec_grid — its cells are ON the WGEN lattice).
    tmin: np.ndarray | None = None
    tmax: np.ndarray | None = None
    #: OBSERVED canopy structure for the Noah ET path (only 15cdec_grid): the
    #: per-HRU green-vegetation fraction (static, on-device (N,)) and the per-CELL
    #: daily LAI climatology look-up (CPU float32 (n_cells, 366), indexed by doy).
    #: Pinned inputs — never learned.  None unless the domain ships the soilveg/
    #: LAI sidecars.
    veg_frac: torch.Tensor | None = None
    lai_lut: np.ndarray | None = None
    #: climate-STATE index for dynamic (time-varying) parameters: a per-cell
    #: (n_cells, T) rolling-precip wetness signal, CAL-standardized (no val
    #: leakage), clamped ~[-3,3] (CPU float32, like tmin).  None unless a dynamic
    #: run requested it (drought -> negative, wet -> positive).
    state: np.ndarray | None = None

    @property
    def n_hru(self) -> int:
        return len(self.cell_idx)

    @property
    def n_time(self) -> int:
        return len(self.dates)

    def chunk(self, t0: int, t1: int) -> tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor]:
        """(prcp, tavg, doy, is_leap) for days [t0, t1) gathered to HRU rows."""
        pr = torch.as_tensor(
            np.ascontiguousarray(self.forcing.prcp[self.cell_idx, t0:t1]),
        ).to(self.device, self.dtype)
        ta = torch.as_tensor(
            np.ascontiguousarray(self.forcing.tavg[self.cell_idx, t0:t1]),
        ).to(self.device, self.dtype)
        return pr, ta, self.doy[t0:t1], self.is_leap[t0:t1]

    def chunk_tmm(self, t0: int, t1: int):
        """(tmin, tmax) for days [t0, t1) gathered to HRU rows; (None, None) if
        the domain has no per-cell Tmin/Tmax (Noah ET then uses the tavg fallback)."""
        if self.tmin is None or self.tmax is None:
            return None, None
        tn = torch.as_tensor(
            np.ascontiguousarray(self.tmin[self.cell_idx, t0:t1]),
        ).to(self.device, self.dtype)
        tx = torch.as_tensor(
            np.ascontiguousarray(self.tmax[self.cell_idx, t0:t1]),
        ).to(self.device, self.dtype)
        return tn, tx

    def chunk_lai(self, t0: int, t1: int):
        """Observed daily LAI (N, t1-t0) for the Noah ET path, gathered to HRU
        rows by each day's day-of-year; None if the domain has no LAI sidecar."""
        if self.lai_lut is None:
            return None
        doy_idx = self.forcing.doy[t0:t1].astype(np.int64) - 1   # 0..365
        lai = self.lai_lut[self.cell_idx][:, doy_idx]            # (N, t1-t0)
        return torch.as_tensor(np.ascontiguousarray(lai)).to(self.device, self.dtype)

    def chunk_state(self, t0: int, t1: int):
        """Climate-state index (N, t1-t0) for days [t0, t1) gathered to HRU rows;
        None if the domain has no dynamic-parameter state field."""
        if self.state is None:
            return None
        s = self.state[self.cell_idx, t0:t1]
        return torch.as_tensor(np.ascontiguousarray(s)).to(self.device, self.dtype)

    def ga_params(self, data_dir: str = "data") -> dict[str, torch.Tensor]:
        """Archived GA optimum expanded to per-HRU (N,) tensors, bounds-asserted."""
        pdf = load_params(data_dir, domain="15cdec")
        validate_ga_optimum(pdf)
        merged = self.hrus.merge(pdf, on="key", how="left", suffixes=("", "_ga"))
        if merged[PARAM_ORDER[0]].isna().any():
            missing = merged.loc[merged[PARAM_ORDER[0]].isna(), "key"].unique()
            raise ValueError(f"{len(missing)} HRU keys missing from ga_optimum")
        return {
            name: torch.as_tensor(merged[name].to_numpy(np.float64)).to(
                self.device, self.dtype)
            for name in PARAM_ORDER
        }


@dataclass
class CalObs:
    """Observed daily gage FNF over the calibration window ONLY.

    Validation observations (after :data:`sacsma.cdec15.CAL_END`) are never
    materialized here — training and model selection cannot read them.
    ``obs_var`` is each basin's observed variance over its finite cal days,
    the fixed NNSE normalizer (so summing chunk losses reproduces the
    per-basin NSE denominator exactly).
    """

    t0: int                    # record index of the cal-window start
    t1: int                    # exclusive record index just past CAL_END
    obs: torch.Tensor          # (B, t1 - t0) mm/day, NaN where missing
    obs_var: torch.Tensor      # (B,)


def load_cal_obs(
    dom: DomainTensors,
    data_dir: str = "data",
    *,
    cal_start: str = "1988-10-01",
    cal_end: str = CAL_END,
) -> CalObs:
    t0 = int(dom.dates.searchsorted(pd.Timestamp(cal_start)))
    t1 = int(dom.dates.searchsorted(pd.Timestamp(cal_end))) + 1
    if dom.dates[t1 - 1] != pd.Timestamp(cal_end):
        raise ValueError(f"cal_end {cal_end} not in the forcing record")

    window = dom.dates[t0:t1]
    gage = load_gage(data_dir)
    arr = np.full((len(dom.basins), t1 - t0), np.nan)
    for b_i, b in enumerate(dom.basins):
        g = gage[gage["basin"] == b].set_index("date")["flow"]
        arr[b_i] = g.reindex(window).to_numpy(np.float64)
    var = np.nanvar(arr, axis=1)                 # population var over finite days
    return CalObs(
        t0=t0, t1=t1,
        obs=torch.as_tensor(arr).to(dom.device, dom.dtype),
        obs_var=torch.as_tensor(var).to(dom.device, dom.dtype),
    )


def _compute_state_index(forcing, dates, window: int, cal_end: str) -> np.ndarray:
    """Per-cell climate-state (wetness) index for dynamic parameters: the
    ``window``-day trailing-mean precipitation, standardized with CALIBRATION-
    period mean/std only (no val leakage) and clamped to [-3, 3].  Drought reads
    negative, wet years positive.  ``(n_cells, T)`` float32."""
    prcp = forcing.prcp.astype(np.float64)                      # (n_cells, T)
    csum = np.cumsum(prcp, axis=1)
    roll = np.empty_like(prcp)
    roll[:, :window] = csum[:, :window] / np.arange(1, window + 1)   # expanding start
    roll[:, window:] = (csum[:, window:] - csum[:, :-window]) / window
    cal = dates <= pd.Timestamp(cal_end)
    mu = roll[:, cal].mean(axis=1, keepdims=True)
    sd = roll[:, cal].std(axis=1, keepdims=True).clip(min=1e-6)
    return np.clip((roll - mu) / sd, -3.0, 3.0).astype(np.float32)


def _calsim_footprint_weights(hrus: pd.DataFrame, basins: tuple[str, ...],
                              base_w: np.ndarray, data_dir: str) -> tuple[np.ndarray, list[str]]:
    """Re-weight ``base_w`` rows by each cell's overlap fraction with the basin's
    CalSim3 catchment (out-of-catchment cells -> 0, boundary cells down-weighted).

    Only basins with a real CalSim3 catchment (rim + geographically-resolved
    secondary nodes in the crosswalk) are re-footed; basins without one
    (Tulare/Kern: PNF/TRM/SCC/ISB) keep their full ``area_weight`` row.  Geometry
    comes from the CalSim3 ``15cdec`` catchment polygons (crosswalk column
    ``basin_15cdec``); the coarse cells are 1/16-deg squares overlapped in the
    equal-area CRS.  Heavy geo deps are imported lazily (only when opted in)."""
    import geopandas as gpd
    from shapely import box

    from ..calsim.catchments import (
        _EQ_CRS,
        _M2_PER_MI2,
        calsim_basin_polygons,
        derive_basin_nodes,
    )

    polys = calsim_basin_polygons(data_dir, "15cdec")            # basin -> catchment geom
    have_catchment = set(derive_basin_nodes(data_dir, "15cdec")["basin"].astype(str))
    step_h = (1.0 / 16.0) / 2.0
    w = base_w.copy()
    refooted: list[str] = []
    for bi, b in enumerate(basins):
        if b not in have_catchment or polys.get(b) is None:
            continue                                             # no catchment -> full footprint
        m = (hrus["basin"] == b).to_numpy()
        sub = hrus.loc[m, ["key", "lat", "lon"]].drop_duplicates("key")
        sq = gpd.GeoDataFrame(
            {"key": sub["key"].to_numpy()},
            geometry=[box(x - step_h, y - step_h, x + step_h, y + step_h)
                      for x, y in zip(sub["lon"].astype(float),
                                      sub["lat"].astype(float), strict=True)],
            crs="EPSG:4326").to_crs(_EQ_CRS)
        cell_mi2 = sq.geometry.area.to_numpy() / _M2_PER_MI2
        poly = gpd.GeoDataFrame(geometry=[polys[b]], crs="EPSG:4326").to_crs(_EQ_CRS)
        ov = gpd.overlay(sq[["key", "geometry"]], poly, how="intersection", keep_geom_type=True)
        if ov.empty:
            continue
        ov_mi2 = ov.geometry.area.to_numpy() / _M2_PER_MI2
        ov_by_key = pd.Series(ov_mi2, index=ov["key"].to_numpy()).groupby(level=0).sum()
        frac = (ov_by_key.reindex(sub["key"]).fillna(0.0).to_numpy() / cell_mi2).clip(0, 1)
        fmap = dict(zip(sub["key"].to_numpy(), frac, strict=True))
        fh = np.where(m, hrus["key"].map(fmap).fillna(0.0).to_numpy(), 0.0)
        wb = base_w[bi] * fh
        if wb.sum() <= 0:
            continue
        w[bi] = wb / wb.sum()
        refooted.append(b)
    return w, refooted


def load_domain_tensors(
    data_dir: str = "data",
    *,
    domain: str = "15cdec",
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    basins: tuple[str, ...] | None = None,
    dynamic_window: int | None = None,
    calsim_footprint: bool = False,
) -> DomainTensors:
    device = torch.device(device)
    forcing = load_domain_forcing(data_dir, domain=domain)
    tmin_cells, tmax_cells = _load_percell_tminmax(data_dir, domain, forcing)
    veg_cells, lai_lut = _load_canopy_obs(data_dir, domain, forcing)
    state_cells = (None if dynamic_window is None else
                   _compute_state_index(forcing, forcing.dates, dynamic_window, CAL_END))
    hrus = load_hru_table(data_dir, domain=domain)
    basins = tuple(basins if basins is not None else BASINS)
    hrus = hrus[hrus["basin"].isin(basins)].reset_index(drop=True)

    cell_idx = np.array([forcing.pos[k] for k in hrus["key"]], dtype=np.int64)
    lat_rad = torch.as_tensor(np.deg2rad(hrus["lat"].to_numpy(np.float64))).to(device, dtype)
    elev = torch.as_tensor(hrus["elev"].to_numpy(np.float64)).to(device, dtype)
    flowlen = torch.as_tensor(hrus["flowlen"].to_numpy(np.float64)).to(device, dtype)

    w_np = np.zeros((len(basins), len(hrus)), dtype=np.float64)
    for b_i, b in enumerate(basins):
        rows = np.flatnonzero((hrus["basin"] == b).to_numpy())
        wt = hrus.loc[rows, "area_weight"].to_numpy(np.float64)
        w_np[b_i, rows] = wt / wt.sum()
    if calsim_footprint:
        w_np, refooted = _calsim_footprint_weights(hrus, basins, w_np, data_dir)
        print(f"load_domain_tensors: CalSim3 footprint re-foot applied to "
              f"{len(refooted)}/{len(basins)} basins {refooted} (others keep full "
              f"footprint)", flush=True)
    w = torch.as_tensor(w_np, dtype=dtype)

    dates = forcing.dates
    doy = torch.as_tensor(forcing.doy.astype(np.float64)).to(device, dtype)
    is_leap = torch.as_tensor(forcing.is_leap.astype(bool)).to(device)

    veg_frac = (None if veg_cells is None else
                torch.as_tensor(veg_cells[cell_idx].astype(np.float64)).to(
                    device, dtype))
    return DomainTensors(dates=dates, doy=doy, is_leap=is_leap, forcing=forcing,
                         hrus=hrus, basins=basins, cell_idx=cell_idx,
                         lat_rad=lat_rad, elev=elev, flowlen=flowlen,
                         W=w.to(device), device=device, dtype=dtype,
                         tmin=tmin_cells, tmax=tmax_cells,
                         veg_frac=veg_frac, lai_lut=lai_lut, state=state_cells)


def _load_canopy_obs(data_dir: str, domain: str, forcing):
    """OBSERVED canopy structure for the Noah ET path, aligned to ``forcing``
    cell order — or (None, None) if the sidecars are absent.

    Returns ``(veg_frac_cells (n_cells,), lai_lut (n_cells, 366))``:

    * ``veg_frac`` = LANDFIRE EVC cover fraction (``EVC_cover_pct`` / 100), from
      ``<domain>/soilveg_continuous.csv``, clamped to CANOPY_BOUNDS.
    * ``lai_lut`` = the per-cell daily LAI climatology, linearly interpolated
      from the 46 8-day samples (``lai_doy001..361``) in
      ``<domain>/lai_climatology.csv`` onto day-of-year 1..366 (winter tail
      flat-held past the last sample), clamped to CANOPY_BOUNDS.

    Both are PINNED inputs (never learned).  Only 15cdec_grid ships them.
    """
    import re

    from .config import CANOPY_BOUNDS

    ddir = domain_dir(data_dir, domain)
    sv_path = ddir / "soilveg_continuous.csv"
    lai_path = ddir / "lai_climatology.csv"
    if not sv_path.exists() or not lai_path.exists():
        return None, None

    vlo, vhi = CANOPY_BOUNDS["veg_frac"]
    # observed LAI keeps only a tiny positive floor (numerical safety) — NOT the
    # learned-param 0.5 bound, which clamped ~half of the driest basins' days and
    # spuriously inflated their canopy conductance.
    llo, lhi = 0.05, CANOPY_BOUNDS["lai"][1]

    # The fine-HRU domains sample these per HRU, so cells shared between HRUs
    # repeat their (identical) row — dedupe to a unique per-cell index before
    # reindexing onto the forcing cell order (grid domains are already unique).
    sv = pd.read_csv(sv_path, usecols=["key", "EVC_cover_pct"]).set_index("key")
    sv = sv[~sv.index.duplicated()]
    veg = (sv["EVC_cover_pct"].reindex(forcing.pos).to_numpy(np.float64) / 100.0)
    veg_frac = np.clip(veg, vlo, vhi).astype(np.float32)              # (n_cells,)

    lai = pd.read_csv(lai_path).rename(columns={"cellkey": "key"}).set_index("key")
    lai = lai[~lai.index.duplicated()]
    doy_cols = sorted((c for c in lai.columns if c.startswith("lai_doy")),
                      key=lambda c: int(re.sub(r"\D", "", c)))
    sample_doys = np.array([int(re.sub(r"\D", "", c)) for c in doy_cols], float)
    samples = lai.reindex(forcing.pos)[doy_cols].to_numpy(np.float64)  # (n_cells, 46)
    target = np.arange(1, 367, dtype=float)                          # doy 1..366
    lut = np.vstack([np.interp(target, sample_doys, row) for row in samples])
    lai_lut = np.clip(lut, llo, lhi).astype(np.float32)              # (n_cells, 366)
    return veg_frac, lai_lut


def _load_percell_tminmax(data_dir: str, domain: str, forcing):
    """Per-cell (n_cells, T) Tmin/Tmax aligned to ``forcing`` cell order, from
    ``<domain>/tminmax_livneh_percell.nc`` — or (None, None) if absent (the
    Noah/PT paths then RAISE at run time; there is no tavg fallback).  Only
    15cdec_grid ships this sidecar (its cells sit on the WGEN lattice)."""
    path = domain_dir(data_dir, domain) / "tminmax_livneh_percell.nc"
    if not path.exists():
        return None, None
    import xarray as xr

    ds = xr.open_dataset(path)
    key_row = {str(k): i for i, k in enumerate(ds["key"].values)}
    order = np.array([key_row[k] for k in forcing.pos], dtype=np.int64)  # forcing order
    tmin = ds["tmin"].values[order].astype(np.float32)
    tmax = ds["tmax"].values[order].astype(np.float32)
    return tmin, tmax
