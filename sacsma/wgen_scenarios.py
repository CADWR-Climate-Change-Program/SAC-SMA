"""WGEN Product A climate scenarios (``wgen_product_a_sNN``): exact decode from the compact store.

The DWR WGEN Product A scenarios 2-30 are thermodynamic perturbations of scenario 1, which is the
committed ``data/region/forcing/wgen_product_a.nc``:

- **temperature** is exactly scenario 1 + dT on every day and cell (tmin and tmax alike), so it is not
  stored; only dT is (in hundredths of a degree, ``dT_hundredths``);
- **precipitation** is a per-cell, per-calendar-month quantile map of the scenario-1 value, so each
  scenario file ``data/region/forcing/wgen_product_a_sNN.nc`` stores it as a TABLE: for every
  (cell, calendar month), the scenario value at each distinct scenario-1 wet value (0.01 mm),
  delta-coded (int32), plus an int8 residual per scenario-1 wet day (rounding-tie splits; almost all
  zero) and an override list of exact (cell, day, value) triples (the days where WGEN's q99
  extreme-tail rule splits one table bin, and the few days dry in scenario 1 but wet here).

Decode is integer-only (np.unique, cumsum, gather), so it is platform-independent, and it is checked
per cell by CRC32 against scenario 1 (``s1_crc32``: the input) and against the source
(``prcp_crc32``: the output). The loaded values equal float32 of the Box release's 2-decimal text
exactly, with the release's inverted tmin/tmax pairs sorted as in scenario 1.

Table order (defined here, never stored): per cell, the sorted distinct codes
``month * 2**20 + h1`` over the cell's scenario-1 wet days (h1 = scenario-1 hundredths); the store
concatenates cells in its ``key`` order (``table_count`` / ``wet_count`` give the slices). Built and
verified by ``dataprep/wgen_product_a_scenarios.py``; see ``data/INVENTORY.md``.
"""
from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np

from . import paths

BASE_PRODUCT = "wgen_product_a"
CODEC = "wgen_table_v1"
SHIFT = 2 ** 20                       # h1 < 2**20 (region max 63,139 hundredths)
_RE = re.compile(r"^wgen_product_a_s(\d{2})$")


def scenario_of(product: str) -> int | None:
    """Scenario number of a ``wgen_product_a_sNN`` product name (2..30), else None."""
    m = _RE.match(str(product))
    return int(m.group(1)) if m else None


def crc(h: np.ndarray) -> int:
    """CRC32 of an integer series as little-endian int32 (the store's fingerprint)."""
    return zlib.crc32(np.ascontiguousarray(h, dtype="<i4").tobytes())


def hundredths(x: np.ndarray) -> np.ndarray:
    """float32 values that are exact hundredths -> int64 hundredths (via float64; never float32)."""
    return np.rint(np.asarray(x, np.float64) * 100.0).astype(np.int64)


def month_of(time) -> np.ndarray:
    import pandas as pd
    return pd.DatetimeIndex(time).month.to_numpy().astype(np.int64)


def decode_cell(h1: np.ndarray, mon: np.ndarray, dtable: np.ndarray, resid: np.ndarray,
                overrides=()) -> np.ndarray:
    """One cell: scenario-1 hundredths ``h1`` (int64, time) -> the scenario's hundredths (int64)."""
    wet = h1 > 0
    code = mon[wet] * SHIFT + h1[wet]
    ux, inv = np.unique(code, return_inverse=True)
    if len(ux) != len(dtable) or int(wet.sum()) != len(resid):
        raise ValueError(f"table/wet size mismatch: {len(ux)} vs {len(dtable)}, "
                         f"{int(wet.sum())} vs {len(resid)}")
    first = np.ones(len(ux), bool)
    first[1:] = (ux[1:] // SHIFT) != (ux[:-1] // SHIFT)
    cs = np.cumsum(dtable.astype(np.int64))
    start = np.maximum.accumulate(np.where(first, np.arange(len(ux)), 0))
    yk = cs - np.concatenate([[0], cs])[start]                 # per-segment cumsum = table values
    out = np.zeros_like(h1)
    out[wet] = yk[inv] + resid.astype(np.int64)
    for d, v in overrides:
        out[d] = v
    return out


def load_region_subset(data_dir: str | Path, product: str, want: list[str],
                       variables=("prcp", "tmin", "tmax")):
    """(key=want, time) Dataset of ``variables`` for a scenario product, decoded exactly.

    ``want`` holds region-store keys (5-decimal ``lat_lon``). Raises on any CRC mismatch, so a changed
    scenario-1 store or a corrupt table fails loud instead of returning wrong forcing."""
    s = scenario_of(product)
    if s is None:
        raise ValueError(f"not a WGEN scenario product: {product!r}")
    fdir = paths.forcing_dir(data_dir)
    return decode_region(fdir / f"{BASE_PRODUCT}.nc", fdir / f"{product}.nc", want, variables)


def decode_region(base_path: str | Path, scen_path: str | Path, want: list[str],
                  variables=("prcp", "tmin", "tmax")):
    """:func:`load_region_subset` on explicit paths (the scenario-1 store and one scenario file)."""
    import netCDF4
    import xarray as xr

    base = xr.open_dataset(base_path)
    try:
        sub = base[["prcp", "tmin", "tmax"]].sel(key=want).load()
    finally:
        base.close()
    z = netCDF4.Dataset(scen_path)
    product = Path(scen_path).stem
    try:
        s = int(z.getncattr("scenario"))
        if z.getncattr("codec") != CODEC or scenario_of(product) not in (None, s):
            raise ValueError(f"{Path(scen_path).name}: codec/scenario attributes do not match")
        dT_h = int(z.getncattr("dT_hundredths"))
        keys = [str(k) for k in z["key"][:]]
        pos = {k: i for i, k in enumerate(keys)}
        tc = np.asarray(z["table_count"][:], np.int64)
        wc = np.asarray(z["wet_count"][:], np.int64)
        t0 = np.concatenate([[0], np.cumsum(tc)])
        w0 = np.concatenate([[0], np.cumsum(wc)])
        s1crc = np.asarray(z["s1_crc32"][:], np.int64)
        pcrc = np.asarray(z["prcp_crc32"][:], np.int64)
        idx = [pos[k] for k in want]
        allc = len(set(idx)) > len(keys) // 2 and "prcp" in variables
        if allc:                                               # one sequential read beats many slices
            dt_all = np.asarray(z["dprcp_table"][:])
            rs_all = np.asarray(z["prcp_resid"][:])
        ok_, od_, ov_ = (np.asarray(z[v][:], np.int64) for v in
                         ("override_key", "override_day", "override_value"))
        ovr: dict[int, list] = {}
        for k_, d_, v_ in zip(ok_, od_, ov_):
            ovr.setdefault(int(k_), []).append((int(d_), int(v_)))
        mon = month_of(sub["time"].values)
        H1 = hundredths(sub["prcp"].values)
        P = np.empty(H1.shape, np.float32)
        for n, i in enumerate(idx):
            if crc(H1[n]) != s1crc[i]:
                raise ValueError(f"{product}: scenario-1 fingerprint mismatch at {keys[i]} "
                                 f"({BASE_PRODUCT}.nc changed?)")
            if "prcp" not in variables:                        # tmin/tmax only: s1 check suffices
                continue
            if allc:
                dt, rs = dt_all[t0[i]:t0[i + 1]], rs_all[w0[i]:w0[i + 1]]
            else:
                dt = np.asarray(z["dprcp_table"][t0[i]:t0[i + 1]])
                rs = np.asarray(z["prcp_resid"][w0[i]:w0[i + 1]])
            h = decode_cell(H1[n], mon, dt, rs, ovr.get(i, ()))
            if crc(h) != pcrc[i]:
                raise ValueError(f"{product}: decoded precipitation fingerprint mismatch at {keys[i]}")
            P[n] = (h / 100.0).astype(np.float32)
        attrs = {a: z.getncattr(a) for a in ("scenario", "dT_C", "cc_pct_per_C", "dmean_pct",
                                             "F_extreme", "source")}
    finally:
        z.close()
    out = {}
    for v in variables:
        if v == "prcp":
            out[v] = sub["prcp"].copy(data=P)
        else:
            out[v] = sub[v].copy(data=((hundredths(sub[v].values) + dT_h) / 100.0).astype(np.float32))
    return xr.Dataset(out, attrs={**sub.attrs, **attrs, "product": product})
