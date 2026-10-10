"""A domain's data as torch tensors for the differentiable pipeline.

Wraps the loaders (``model.load_domain_forcing``, ``io.load_hru_table``, ``cdec15.load_gage``)
into a :class:`DomainTensors` bundle: per-HRU static tensors, the basin aggregation matrix
``W`` (normalized ``area_weight`` per basin, the ``model.py`` convention) and the forcing by
window (:class:`Window`).  The forcing arrays stay CPU float32 NumPy (from ``DomainForcing``);
a window is gathered to the HRU rows and moved to the device on demand.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .. import paths
from ..cdec15 import BASINS, CAL_END, load_gage
from ..io import DEFAULT_FORCING, MULTI_TIMESCALE_DOMAIN, load_hru_table
from ..model import DomainForcing, load_domain_forcing


@dataclass
class Window:
    """One window of the forcing on the HRU rows: ``(rows, days)`` tensors and the calendar
    of the days, with the daily ``tmin`` / ``tmax`` where the domain has them (the
    Priestley-Taylor PET) and the observed ``lai`` where it has the canopy tables (Noah-lite)."""

    pr: torch.Tensor
    ta: torch.Tensor
    doy: torch.Tensor
    leap: torch.Tensor
    tmin: torch.Tensor | None = None
    tmax: torch.Tensor | None = None
    lai: torch.Tensor | None = None

    def _map(self, f_row, f_day) -> Window:
        return Window(*(None if v is None else (f_day(v) if v.dim() == 1 else f_row(v))
                        for v in (getattr(self, k.name) for k in dataclasses.fields(self))))

    def days(self, sl: slice) -> Window:
        """The days ``sl`` of the window."""
        return self._map(lambda v: v[:, sl], lambda v: v[sl])

    def rows(self, idx) -> Window:
        """The rows ``idx`` of the window."""
        return self._map(lambda v: v[idx], lambda v: v)

    def copy_(self, src: Window) -> None:
        """Copy ``src`` into these buffers (the static inputs of a captured graph)."""
        for k in dataclasses.fields(self):
            buf = getattr(self, k.name)
            if buf is not None:
                buf.copy_(getattr(src, k.name))


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

    @property
    def n_hru(self) -> int:
        return len(self.cell_idx)

    @property
    def n_time(self) -> int:
        return len(self.dates)

    def _rows(self, a: np.ndarray, t0: int, t1: int) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a[self.cell_idx, t0:t1])).to(
            self.device, self.dtype)

    def window(self, t0: int, t1: int) -> Window:
        """The forcing of days [t0, t1) on the HRU rows."""
        lai = None
        if self.lai_lut is not None:
            doy_idx = self.forcing.doy[t0:t1].astype(np.int64) - 1   # 0..365
            lai = torch.as_tensor(np.ascontiguousarray(
                self.lai_lut[self.cell_idx][:, doy_idx])).to(self.device, self.dtype)
        tmm = self.tmin is not None and self.tmax is not None
        return Window(self._rows(self.forcing.prcp, t0, t1), self._rows(self.forcing.tavg, t0, t1),
                      self.doy[t0:t1], self.is_leap[t0:t1],
                      self._rows(self.tmin, t0, t1) if tmm else None,
                      self._rows(self.tmax, t0, t1) if tmm else None, lai)

    def window_buffers(self, length: int, physics) -> Window:
        """Zero buffers of a ``length``-day window for the inputs ``physics``
        (:class:`sacsma.engine.Physics`) reads -- the static inputs of a captured graph."""
        z = lambda: torch.zeros(self.n_hru, length, device=self.device, dtype=self.dtype)  # noqa: E731
        pt = physics.pet == "priestley_taylor"
        return Window(z(), z(), torch.zeros(length, device=self.device, dtype=self.doy.dtype),
                      torch.zeros(length, device=self.device, dtype=torch.bool),
                      z() if pt else None, z() if pt else None,
                      z() if physics.et == "noah_lite" else None)


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
    obs_mask: tuple[str, ...] = (),
) -> CalObs:
    """The basins' gage flow over the cal window.  ``obs_mask`` (``DplConfig.obs_mask``,
    ``"cdec_<BASIN>|YYYY-MM-DD"``, the registry ids of the 15 CDEC basins) drops those days
    before the normalizers; an entry for another entity or a day outside the window is
    skipped, an in-window day that is not observed raises (a stale entry)."""
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
    for m in obs_mask:
        eid, _, day = m.partition("|")
        b = eid.removeprefix("cdec_")
        k = int(window.searchsorted(pd.Timestamp(day)))
        if b not in dom.basins or k >= len(window) or window[k] != pd.Timestamp(day):
            continue                             # another entity, or outside the window
        b_i = list(dom.basins).index(b)
        if not np.isfinite(arr[b_i, k]):
            raise ValueError(f"obs_mask {m!r}: not an observed in-window day")
        arr[b_i, k] = np.nan
    var = np.nanvar(arr, axis=1)                 # population var over finite days
    return CalObs(
        t0=t0, t1=t1,
        obs=torch.as_tensor(arr).to(dom.device, dom.dtype),
        obs_var=torch.as_tensor(var).to(dom.device, dom.dtype),
    )


#: max complete calendar months a TBPTT chunk can hold (12 for a water-year chunk;
#: 13 for headroom).  The per-chunk monthly target is padded to this.
CHUNK_MAXM = 13


def month_chunk_target(dates: pd.DatetimeIndex, c0: int, length: int,
                       cal_t0: int, cal_t1: int, maxm: int = CHUNK_MAXM):
    """Monthly-bucket target for a TBPTT chunk [c0, c0+length): a (length, maxm)
    day->month-slot sum matrix for the calendar months lying COMPLETELY inside both
    the chunk and the cal window [cal_t0, cal_t1).  Split/partial months (chunk
    boundaries, post-CAL_END) get no slot (mask 0) — a partial-month sum isn't
    comparable to a full-month target.  Returns (bucket, mask (maxm,))."""
    c1 = c0 + length
    all_codes = (dates.year * 12 + dates.month).to_numpy()
    codes = all_codes[c0:c1]
    bucket = np.zeros((length, maxm), np.float64)
    mask = np.zeros(maxm, np.float64)
    slot = 0
    for code in pd.unique(codes):
        local = np.nonzero(codes == code)[0]
        g = np.nonzero(all_codes == code)[0]
        g = g[(g >= cal_t0) & (g < cal_t1)]
        if len(g) and g.min() >= c0 and g.max() < c1 and len(local) == len(g) \
                and slot < maxm:
            bucket[local, slot] = 1.0
            mask[slot] = 1.0
            slot += 1
    return bucket, mask


def load_domain_tensors(
    data_dir: str = "data",
    *,
    domain: str = "15cdec",
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    basins: tuple[str, ...] | None = None,
    product: str = DEFAULT_FORCING,
) -> DomainTensors:
    device = torch.device(device)
    forcing = load_domain_forcing(data_dir, domain=domain, product=product)
    tmin_cells, tmax_cells = _load_percell_tminmax(data_dir, domain, forcing, product)
    veg_cells, lai_lut = _load_canopy_obs(data_dir, domain, forcing)
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
                         veg_frac=veg_frac, lai_lut=lai_lut)


def _load_canopy_obs(data_dir: str, domain: str, forcing):
    """OBSERVED canopy structure for the Noah ET path, aligned to ``forcing``
    cell order — or (None, None) if the sidecars are absent.

    Returns ``(veg_frac_cells (n_cells,), lai_lut (n_cells, 366))``:

    * ``veg_frac`` = LANDFIRE EVC cover fraction (``EVC_cover_pct`` / 100), from
      ``<domain>/soilveg_continuous.csv``, clamped to [0, 1].
    * ``lai_lut`` = the per-cell daily LAI climatology, linearly interpolated
      from the 46 8-day samples (``lai_doy001..361``) onto day-of-year 1..366
      (winter tail flat-held past the last sample), clamped to [0.05, 6].

    Both are PINNED inputs (never learned).  Paths resolve through the
    suffix-aware ``io.soilveg_path``/``io.lai_climatology_path`` helpers, so
    suffixed calsim domains and the multi-timescale domain (region tables) find their
    sidecars.
    """
    import re

    from ..io import lai_climatology_path, soilveg_path

    sv_path = soilveg_path(data_dir, domain)
    lai_path = lai_climatology_path(data_dir, domain)
    if not sv_path.exists() or not lai_path.exists():
        return None, None

    vlo, vhi = 0.0, 1.0
    llo, lhi = 0.05, 6.0      # a small positive floor for numerical safety

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


def _load_percell_tminmax(data_dir: str, domain: str, forcing, product: str = DEFAULT_FORCING):
    """Per-cell (n_cells, T) Tmin/Tmax aligned to ``forcing`` cell order, from
    the unified region forcing store of ``product`` (``data/inputs/forcing``; a WGEN
    scenario store is decoded against its base) — or (None, None) for non-grid domains
    (the Noah/PT paths then RAISE at run time; there is no tavg fallback)."""
    from .. import wgen_scenarios
    from ..io import REGION_DOMAINS, forcing_path, norm_grid_key

    if domain not in REGION_DOMAINS:
        return None, None
    if wgen_scenarios.scenario_of(product) is not None:
        want = [norm_grid_key(k) for k in forcing.pos]
        ds = wgen_scenarios.load_region_subset(data_dir, product, want, ("tmin", "tmax"))
        return (ds["tmin"].values.astype(np.float32), ds["tmax"].values.astype(np.float32))
    path = forcing_path(data_dir, domain, product)
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
