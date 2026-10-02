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

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .. import paths
from ..cdec15 import BASINS, CAL_END, load_gage
from ..io import MULTI_TIMESCALE_DOMAIN, load_hru_table, load_params
from ..model import DomainForcing, load_domain_forcing
from .config import PARAM_ORDER, validate_ga_optimum


@dataclass
class CellDedup:
    """The distinct-cell physics basis of a :class:`DomainTensors` whose HRU rows
    repeat grid cells (``DplConfig.dedup_cells``; built by :func:`with_cell_dedup`).

    On the multi-timescale domain one row is an (entity, cell) pair, so a cell that
    several nested entities share appears once per entity, with identical forcing,
    statics and parameter-net inputs.  With a ``CellDedup`` attached, the per-cell
    physics — parameter net, PET, Snow-17, Noah ET, SAC-SMA and their states — runs
    once per DISTINCT cell (``U`` of them); only the routing (per-row flow length,
    unit hydrograph and routing-inflow history) and the aggregation ``W`` stay per
    row, fed by gathering each cell's surface/base runoff to its rows (``row_cell``).
    """

    row_cell: torch.Tensor     # (N,) int64 on device: each HRU row's distinct-cell index
    first_row: torch.Tensor    # (U,) int64 on device: the first HRU row of each cell
    cell_idx: np.ndarray       # (U,) int rows into the forcing arrays
    lat_rad: torch.Tensor      # (U,)
    elev: torch.Tensor         # (U,)
    veg_frac: torch.Tensor | None = None   # (U,) (Noah ET domains)


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
    #: distinct-cell physics basis (``DplConfig.dedup_cells``, :func:`with_cell_dedup`);
    #: None = the per-row physics (every HRU row runs its own column — the default)
    dedup: CellDedup | None = None

    @property
    def n_hru(self) -> int:
        return len(self.cell_idx)

    @property
    def n_time(self) -> int:
        return len(self.dates)

    # -- the physics rows: the HRU rows, or the distinct cells under cell dedup.
    # The chunk_* gathers below, the state vectors and the parameter net run on
    # these; routing, the routing history, flowlen and W stay per HRU row.
    @property
    def n_phys(self) -> int:
        """Rows of the per-cell physics: ``n_hru``, or the distinct cells (dedup)."""
        return self.n_hru if self.dedup is None else len(self.dedup.cell_idx)

    @property
    def phys_idx(self) -> np.ndarray:
        """Forcing-array rows of the physics rows."""
        return self.cell_idx if self.dedup is None else self.dedup.cell_idx

    @property
    def phys_lat_rad(self) -> torch.Tensor:
        return self.lat_rad if self.dedup is None else self.dedup.lat_rad

    @property
    def phys_elev(self) -> torch.Tensor:
        return self.elev if self.dedup is None else self.dedup.elev

    @property
    def phys_veg_frac(self) -> torch.Tensor | None:
        return self.veg_frac if self.dedup is None else self.dedup.veg_frac

    @property
    def row_cell(self) -> torch.Tensor | None:
        """(N,) physics row of each HRU row under cell dedup, else None (the
        ``row_cell`` argument of ``forward.run_window`` / ``routing_uh``)."""
        return None if self.dedup is None else self.dedup.row_cell

    def phys_x(self, x: torch.Tensor) -> torch.Tensor:
        """The parameter-net input of the physics rows: ``x`` itself, or its first
        row per distinct cell under cell dedup (rows of a cell are identical —
        :func:`with_cell_dedup` asserts it)."""
        return x if self.dedup is None else x.index_select(0, self.dedup.first_row)

    def chunk(self, t0: int, t1: int) -> tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor]:
        """(prcp, tavg, doy, is_leap) for days [t0, t1) gathered to the physics
        rows (the HRU rows; the distinct cells under cell dedup)."""
        idx = self.phys_idx
        pr = torch.as_tensor(
            np.ascontiguousarray(self.forcing.prcp[idx, t0:t1]),
        ).to(self.device, self.dtype)
        ta = torch.as_tensor(
            np.ascontiguousarray(self.forcing.tavg[idx, t0:t1]),
        ).to(self.device, self.dtype)
        return pr, ta, self.doy[t0:t1], self.is_leap[t0:t1]

    def chunk_tmm(self, t0: int, t1: int):
        """(tmin, tmax) for days [t0, t1) gathered to the physics rows; (None, None)
        if the domain has no per-cell Tmin/Tmax (Noah ET then uses the tavg fallback)."""
        if self.tmin is None or self.tmax is None:
            return None, None
        idx = self.phys_idx
        tn = torch.as_tensor(
            np.ascontiguousarray(self.tmin[idx, t0:t1]),
        ).to(self.device, self.dtype)
        tx = torch.as_tensor(
            np.ascontiguousarray(self.tmax[idx, t0:t1]),
        ).to(self.device, self.dtype)
        return tn, tx

    def chunk_lai(self, t0: int, t1: int):
        """Observed daily LAI (physics rows, t1-t0) for the Noah ET path, gathered
        by each day's day-of-year; None if the domain has no LAI sidecar."""
        if self.lai_lut is None:
            return None
        doy_idx = self.forcing.doy[t0:t1].astype(np.int64) - 1   # 0..365
        lai = self.lai_lut[self.phys_idx][:, doy_idx]            # (N, t1-t0)
        return torch.as_tensor(np.ascontiguousarray(lai)).to(self.device, self.dtype)

    def chunk_state(self, t0: int, t1: int):
        """Climate-state index (physics rows, t1-t0) for days [t0, t1); None if the
        domain has no dynamic-parameter state field."""
        if self.state is None:
            return None
        s = self.state[self.phys_idx, t0:t1]
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


def _rows_differing(a: np.ndarray, rep: np.ndarray) -> np.ndarray:
    """(N,) bool: rows of ``a`` not bitwise-equal to ``rep`` (NaN == NaN)."""
    a = np.asarray(a).reshape(len(a), -1)
    rep = np.asarray(rep).reshape(len(rep), -1)
    ne = a != rep
    if np.issubdtype(a.dtype, np.floating):
        ne &= ~(np.isnan(a) & np.isnan(rep))
    return ne.any(axis=1)


def with_cell_dedup(dom: DomainTensors, x=None, *, verbose: bool = True) -> DomainTensors:
    """A copy of ``dom`` whose per-cell physics runs once per DISTINCT grid cell
    (``DplConfig.dedup_cells``; see :class:`CellDedup`).  The row-level fields —
    ``hrus``, ``cell_idx``, ``lat_rad``/``elev``/``veg_frac``, ``flowlen``, ``W`` —
    are unchanged; the physics accessors (``n_phys``, ``phys_*``, ``chunk*``,
    ``phys_x``, ``row_cell``) switch to the distinct cells, in first-appearance order.

    Refuses (``ValueError``) when rows that share a cell do not carry identical
    physics inputs: the parameter-net features ``x`` (``(N, F)`` array or tensor —
    pass it: e.g. a net with the flow length as a feature gives every (entity,
    cell) row its own parameters), the latitude, elevation and observed veg
    fraction.  Forcing, LAI and the climate-state index are per cell by
    construction (gathered through ``cell_idx``)."""
    if dom.dedup is not None:
        return dom
    ci = np.asarray(dom.cell_idx)
    _, first, inv = np.unique(ci, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")          # distinct cells, first-appearance order
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    first_row = first[order]                          # (U,) first HRU row of each cell
    row_cell = rank[inv.reshape(-1)]                  # (N,) cell of each HRU row
    rep = first_row[row_cell]                         # (N,) the representative row

    def _t(v):
        return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    checks = {"lat_rad": dom.lat_rad, "elev": dom.elev, "veg_frac": dom.veg_frac,
              "parameter-net features x": x}
    for name, v in checks.items():
        if v is None:
            continue
        a = _t(v)
        if len(a) != len(ci):
            raise ValueError(f"cell dedup: {name} has {len(a)} rows, the domain {len(ci)}")
        bad = np.flatnonzero(_rows_differing(a, a[rep]))
        if len(bad):
            ex = ", ".join(f"{dom.hrus['basin'].iat[i]}/{dom.hrus['key'].iat[i]}"
                           for i in bad[:5])
            raise ValueError(
                f"cell dedup refused: {len(bad)} HRU row(s) differ in {name} from the "
                f"other rows of their grid cell (e.g. {ex}) — the per-cell physics would "
                "not be shared; run without dedup_cells (a flow-length feature, "
                "flowlen_feature=True, gives every (entity, cell) row its own parameters)")
    dev = dom.device
    fr_t = torch.as_tensor(first_row, dtype=torch.int64, device=dev)
    dd = CellDedup(
        row_cell=torch.as_tensor(row_cell, dtype=torch.int64, device=dev),
        first_row=fr_t,
        cell_idx=ci[first_row],
        lat_rad=dom.lat_rad.index_select(0, fr_t),
        elev=dom.elev.index_select(0, fr_t),
        veg_frac=None if dom.veg_frac is None else dom.veg_frac.index_select(0, fr_t))
    if verbose:
        print(f"cell dedup: {len(ci)} HRU rows -> {len(first_row)} distinct cells "
              f"({len(ci) / max(len(first_row), 1):.2f} rows per cell): per-cell physics "
              "once per cell, routing + aggregation per row", flush=True)
    return dataclasses.replace(dom, dedup=dd)


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


#: max complete calendar months a fixed-length TBPTT chunk can hold (<=12 for a
#: 366-day chunk; 13 for headroom).  The per-chunk monthly target is padded to this.
CHUNK_MAXM = 13


def month_chunk_target(dates: pd.DatetimeIndex, c0: int, length: int,
                       cal_t0: int, cal_t1: int, maxm: int = CHUNK_MAXM):
    """Monthly-bucket target for a TBPTT chunk [c0, c0+length): a (length, maxm)
    day->month-slot sum matrix and the 0-based calendar month of each slot, for
    the calendar months lying COMPLETELY inside both the chunk and the cal window
    [cal_t0, cal_t1).  Split/partial months (chunk boundaries, post-CAL_END) get
    no slot (mask 0) — a partial-month sum isn't comparable to a full-month
    target.  Returns (bucket, cal_month0 (maxm,), mask (maxm,))."""
    c1 = c0 + length
    all_codes = (dates.year * 12 + dates.month).to_numpy()
    codes = all_codes[c0:c1]
    d_month = dates.month.to_numpy()[c0:c1]
    bucket = np.zeros((length, maxm), np.float64)
    cal_month0 = np.zeros(maxm, np.int64)
    mask = np.zeros(maxm, np.float64)
    slot = 0
    for code in pd.unique(codes):
        local = np.nonzero(codes == code)[0]
        g = np.nonzero(all_codes == code)[0]
        g = g[(g >= cal_t0) & (g < cal_t1)]
        if len(g) and g.min() >= c0 and g.max() < c1 and len(local) == len(g) \
                and slot < maxm:
            bucket[local, slot] = 1.0
            cal_month0[slot] = d_month[local[0]] - 1
            mask[slot] = 1.0
            slot += 1
    return bucket, cal_month0, mask


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
    if basins is None:
        if domain == MULTI_TIMESCALE_DOMAIN:
            # registry order — one "basin" per training entity; the CalSim3 arc
            # family (``build_entities.py --calsim-arcs``) is opt-in: its entities
            # load only when named, so the full domain stays the 95 base entities
            reg = pd.read_csv(
                paths.entities(data_dir),
                usecols=["entity_id", "family"])
            basins = tuple(reg.loc[reg["family"] != "calsim_monthly", "entity_id"])
        else:
            basins = tuple(BASINS)
    else:
        basins = tuple(basins)
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
      from the 46 8-day samples (``lai_doy001..361``) onto day-of-year 1..366
      (winter tail flat-held past the last sample), clamped to CANOPY_BOUNDS.

    Both are PINNED inputs (never learned).  Paths resolve through the
    suffix-aware ``io.soilveg_path``/``io.lai_climatology_path`` helpers, so
    suffixed calsim domains and the multi-timescale domain (region tables) find their
    sidecars.
    """
    import re

    from ..io import lai_climatology_path, soilveg_path
    from .config import CANOPY_BOUNDS

    sv_path = soilveg_path(data_dir, domain)
    lai_path = lai_climatology_path(data_dir, domain)
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
    the unified region forcing store (``data/inputs/forcing``) — or
    (None, None) for non-grid domains (the Noah/PT paths then RAISE at run
    time; there is no tavg fallback)."""
    from ..io import REGION_DOMAINS, forcing_path, norm_grid_key

    if domain not in REGION_DOMAINS:
        return None, None
    path = forcing_path(data_dir, domain)
    if not path.exists():
        return None, None
    import xarray as xr

    ds = xr.open_dataset(path)
    key_row = {str(k): i for i, k in enumerate(ds["key"].values)}
    order = np.array([key_row[norm_grid_key(k)] for k in forcing.pos],
                     dtype=np.int64)  # forcing order
    tmin = ds["tmin"].values[order].astype(np.float32)
    tmax = ds["tmax"].values[order].astype(np.float32)
    ds.close()
    return tmin, tmax
