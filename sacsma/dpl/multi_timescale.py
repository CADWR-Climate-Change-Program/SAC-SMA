"""Per-entity observation bundle for the multi-timescale domain.

Companion to :func:`sacsma.dpl.data.load_domain_tensors` with
``domain="multifamily"``: loads every entity's target series from its
native store, masks it to the entity's OWN training window (the registry's
``train_start``/``train_end``) inside a single global envelope, and computes
each entity's NNSE variance normalizer over its own record.

Daily entities (usgs_daily + cdec_daily) form a dense ``(D, T_env)`` matrix
like :class:`sacsma.dpl.data.CalObs`; monthly entities (uf_monthly) get a
``(M, n_months)`` matrix over the envelope's calendar months, consumed by
the monthly flow-loss term (simulated daily flow bucketed to complete
months, the ``month_chunk_target`` machinery).

Sources, per family (the registry's ``obs_store`` column):

* ``usgs_daily``  — ``data/usgs/flow_daily.nc`` ``flow_mm`` by gauge id.
* ``cdec_daily``  — the 15 committed basins from ``data/cdec15/gage.csv``
  (mm/day); CLE + CSN from ``data/cdec_fnf/fnf_daily_mm.csv`` (the
  derived depth companion of the raw cfs store).
* ``uf_monthly``  — ``data/dwr_unimpaired/uf_monthly_mm.csv`` (mm/month,
  month-end stamps; the derived depth companion of ``uf_monthly.csv``).
* ``calsim_monthly`` — ``data/calsim/calsim3_inflow_monthly_mm.csv`` (the
  CalSim3 rim INFLOW arcs as mm/month over each arc's ``SQ_MI``), kept only on
  the arc-months of ``data/calsim/arc_obs_mask.csv`` (the arc's own gauge
  record in the training water years; see :mod:`sacsma.dpl.calsim.arcs`).
  Registry rows exist only when the registry was built with
  ``--calsim-arcs``; they load only when a run names them (``--basins``).

Every entity's finite in-window count is asserted against the registry's
``n_obs`` — the loader cannot silently drift from the audited store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..cdec15 import load_gage
from ..io import MULTI_TIMESCALE_DOMAIN, domain_dir
from .data import DomainTensors, month_chunk_target

#: global training envelope (WY1950-2018; forcing ends 2018-12-31).
ENVELOPE_START = "1949-10-01"
ENVELOPE_END = "2018-12-31"
#: the CalSim3 arc family (registry rows from ``build_entities.py --calsim-arcs``)
CALSIM_FAMILY = "calsim_monthly"


def _calsim_store(data_dir: str, obs_store: str) -> tuple[pd.DataFrame, dict]:
    """The arc depth store (wide, monthly PeriodIndex x arc) and the arc mask."""
    from .calsim.arcs import load_arc_mask
    path, col = obs_store.rsplit(":", 1)
    t = pd.read_csv(Path(data_dir) / path, parse_dates=["date"])
    w = t.pivot(index="date", columns="arc", values=col)
    w.index = pd.PeriodIndex(w.index, freq="M")
    return w, load_arc_mask(data_dir)


@dataclass
class EntityObs:
    """Observed targets over the envelope, window-masked per entity.

    ``obs_daily`` rows follow ``daily_rows`` (indices into ``dom.basins``);
    ``obs_monthly`` rows follow ``monthly_rows``.  ``month_code`` maps each
    monthly column to ``year * 12 + (month - 1)`` so a chunk's complete
    months land in the right column regardless of chunk boundaries.
    ``var_*`` are population variances over each entity's finite in-window
    values — the fixed NNSE denominators.
    """

    t0: int
    t1: int
    family: tuple[str, ...]            # per dom.basins entry
    daily_rows: np.ndarray             # (D,) indices into dom.basins
    obs_daily: torch.Tensor            # (D, t1 - t0) NaN where unobserved
    var_daily: torch.Tensor            # (D,)
    monthly_rows: np.ndarray           # (M,) indices into dom.basins
    month_code: np.ndarray             # (n_months,) year*12 + month0
    obs_monthly: torch.Tensor          # (M, n_months) mm/month, NaN-masked
    var_monthly: torch.Tensor          # (M,)
    #: per dom.basins entry: observations removed by obs_mask (0 without one), so
    #: scoring can check its counts against the registry n_obs
    n_masked: tuple[int, ...] = ()
    #: per dom.basins entry: in-window observations removed by the water-year
    #: holdout (``holdout_wy``; 0 without one)
    n_holdout: tuple[int, ...] = ()
    #: per dom.basins entry: months ADDED by the uf back-extension
    #: (``uf_train_start``; 0 without one) — scored = n_obs - n_masked -
    #: n_holdout + n_ext
    n_ext: tuple[int, ...] = ()


def load_entity_obs(
    dom: DomainTensors,
    data_dir: str = "data",
    *,
    cal_start: str = ENVELOPE_START,
    cal_end: str = ENVELOPE_END,
    obs_mask: tuple[str, ...] = (),
    holdout_wy: tuple[int, int] | None = None,
    uf_train_start: str | None = None,
) -> EntityObs:
    """``obs_mask`` (``DplConfig.obs_mask``, ``"entity_id|YYYY-MM-DD"``) drops those
    daily observations.  An entry for a registry entity outside this run's basins
    is skipped; an unknown entity, a monthly entity of this run (the mask is
    daily-only), or a day that is not an observed in-window day raises (a stale
    entry must not silently mask nothing).

    ``holdout_wy`` (``DplConfig.holdout_wy``, inclusive water years) blanks every
    family's targets in those water years — after the registry n_obs audit and
    the obs_mask, before the NNSE variances — and counts them per entity
    (``n_holdout``); a calsim_monthly entity must lose none (its arc mask already
    excludes the holdout: a nonzero count is a stale mask).  ``uf_train_start``
    (``DplConfig.uf_train_start``) adds each uf_monthly entity's months from that
    date to its registry train_start (minus the holdout) from the same store,
    after the audit; every one must be observed (``n_ext``)."""
    ddir = domain_dir(data_dir, MULTI_TIMESCALE_DOMAIN)
    reg = pd.read_csv(ddir / "entities.csv", dtype={"site_id": str},
                      parse_dates=["train_start", "train_end"])
    known_ids = set(reg["entity_id"])
    reg = reg.set_index("entity_id").loc[list(dom.basins)]

    t0 = int(dom.dates.searchsorted(pd.Timestamp(cal_start)))
    t1 = int(dom.dates.searchsorted(pd.Timestamp(cal_end))) + 1
    if dom.dates[t1 - 1] != pd.Timestamp(cal_end):
        raise ValueError(f"cal_end {cal_end} not in the forcing record")
    window = dom.dates[t0:t1]

    ufmm = pd.read_csv(Path(data_dir) / "dwr_unimpaired" / "uf_monthly_mm.csv",
                       parse_dates=["date"])
    fnfmm = pd.read_csv(Path(data_dir) / "cdec_fnf" / "fnf_daily_mm.csv",
                        parse_dates=["date"])
    gage = load_gage(data_dir)

    import xarray as xr
    usgs = xr.open_dataset(f"{data_dir}/usgs/flow_daily.nc")

    daily_rows, monthly_rows = [], []
    daily_arrs, n_obs_want = [], []
    for i, eid in enumerate(dom.basins):
        r = reg.loc[eid]
        if r["timescale"] == "monthly":
            monthly_rows.append(i)
            continue
        daily_rows.append(i)
        n_obs_want.append(int(r["n_obs"]))
        if r["family"] == "usgs_daily":
            s = pd.Series(
                usgs["flow_mm"].sel(gauge=r["site_id"]).values,
                index=pd.DatetimeIndex(usgs["time"].values))
        elif eid in ("cdec_CLE", "cdec_CSN"):
            g = fnfmm[fnfmm["station"] == r["site_id"]]
            s = pd.Series(g["depth_mm"].to_numpy(), index=g["date"])
        else:                                   # 15 committed cdec basins
            g = gage[gage["basin"] == r["site_id"]]
            s = pd.Series(g["flow"].to_numpy(),
                          index=pd.DatetimeIndex(g["date"]))
        arr = s.reindex(window).to_numpy(np.float64, copy=True)
        mask = (window >= r["train_start"]) & (window <= r["train_end"])
        arr[~mask] = np.nan
        daily_arrs.append(arr)
    usgs.close()

    obs_d = np.stack(daily_arrs, 0) if daily_arrs else np.zeros((0, t1 - t0))
    n_fin = np.isfinite(obs_d).sum(axis=1)
    want = np.array(n_obs_want)
    if not np.array_equal(n_fin, want):
        bad = [(dom.basins[daily_rows[j]], int(n_fin[j]), int(want[j]))
               for j in np.flatnonzero(n_fin != want)]
        raise ValueError(f"daily obs counts diverge from the registry n_obs: "
                         f"{bad[:5]}")
    # hand-confirmed bad observations — applied after the n_obs check, which
    # verifies the stores themselves
    daily_ids = [dom.basins[i] for i in daily_rows]
    n_masked = [0] * len(dom.basins)
    for s in obs_mask:
        eid, _, day = s.partition("|")
        if eid not in known_ids:
            raise ValueError(f"obs_mask {s!r}: unknown entity {eid!r}")
        if eid not in dom.basins:
            continue                            # the mask covers other runs too
        if eid not in daily_ids:
            raise ValueError(f"obs_mask {s!r}: {eid} is a monthly entity — the "
                             "mask covers daily targets only")
        j, k = daily_ids.index(eid), int(window.searchsorted(pd.Timestamp(day)))
        if (k >= len(window) or window[k] != pd.Timestamp(day)
                or not np.isfinite(obs_d[j, k])):
            raise ValueError(f"obs_mask {s!r}: not an observed in-window day")
        obs_d[j, k] = np.nan
        n_masked[daily_rows[j]] += 1
    # the water-year holdout — after the audit and the mask, before the normalizers
    n_holdout = [0] * len(dom.basins)
    n_ext = [0] * len(dom.basins)
    if holdout_wy:
        wy_d = np.asarray(window.year + (window.month >= 10))
        ho_d = (wy_d >= holdout_wy[0]) & (wy_d <= holdout_wy[1])
        cnt = np.isfinite(obs_d[:, ho_d]).sum(axis=1)
        obs_d[:, ho_d] = np.nan
        for j, i in enumerate(daily_rows):
            n_holdout[i] = int(cnt[j])
    var_d = np.nanvar(obs_d, axis=1)

    # ---- monthly entities over the envelope's calendar months -----------
    m_start = pd.Timestamp(cal_start).to_period("M")
    m_end = pd.Timestamp(cal_end).to_period("M")
    months = pd.period_range(m_start, m_end, freq="M")
    month_code = (months.year * 12 + (months.month - 1)).to_numpy()
    wy_m = np.asarray(months.year + (months.month >= 10))
    ho_m = ((wy_m >= holdout_wy[0]) & (wy_m <= holdout_wy[1]) if holdout_wy
            else np.zeros(len(months), dtype=bool))
    obs_m = np.full((len(monthly_rows), len(months)), np.nan)
    cs_store = None
    for j, i in enumerate(monthly_rows):
        eid = dom.basins[i]
        r = reg.loc[eid]
        if r["family"] == CALSIM_FAMILY:
            if cs_store is None:
                cs_store = _calsim_store(data_dir, r["obs_store"])
            depth, arc_mask = cs_store
            arc = r["site_id"]
            # months fully inside the entity window AND in the arc's own record
            keep = ((months.start_time >= r["train_start"])
                    & (months.end_time <= r["train_end"] + pd.Timedelta(days=1))
                    & months.isin(arc_mask.get(arc, pd.PeriodIndex([], freq="M"))))
            vals = depth[arc].reindex(months).to_numpy(np.float64, copy=True)
            vals[~keep] = np.nan
            if not np.isfinite(vals).any():
                raise ValueError(f"{eid}: no observed arc-month under the arc mask "
                                 "(not an own-record arc — see arc_hierarchy.csv tier)")
        else:
            g = ufmm[ufmm["uf"] == int(r["site_id"].split()[1])]
            per = pd.PeriodIndex(g["date"], freq="M")
            s = pd.Series(g["depth_mm"].to_numpy(), index=per)
            # months fully inside the entity window
            keep = (months.start_time >= r["train_start"]) \
                & (months.end_time <= r["train_end"] + pd.Timedelta(days=1))
            vals = s.reindex(months).to_numpy(np.float64, copy=True)
            vals[~keep] = np.nan
        n = int(np.isfinite(vals).sum())
        if n != int(r["n_obs"]):
            raise ValueError(f"{eid}: {n} in-window months vs registry "
                             f"n_obs {int(r['n_obs'])}")
        if holdout_wy:
            n_holdout[i] = int(np.isfinite(vals[ho_m]).sum())
            if r["family"] == CALSIM_FAMILY and n_holdout[i]:
                raise ValueError(f"{eid}: {n_holdout[i]} arc-month(s) inside the holdout "
                                 f"WY{holdout_wy[0]}-{holdout_wy[1]} — arc_obs_mask.csv "
                                 "is stale (it must exclude the holdout)")
            vals[ho_m] = np.nan
        if uf_train_start and r["family"] == "uf_monthly":
            ts = pd.Timestamp(uf_train_start)
            if ts >= r["train_start"]:
                raise ValueError(f"{eid}: uf_train_start {uf_train_start} is not before "
                                 f"its registry train_start {r['train_start'].date()}")
            ext = ((months.start_time >= ts) & (months.end_time < r["train_start"])
                   & ~ho_m)
            ev = s.reindex(months).to_numpy(np.float64, copy=True)[ext]
            if not np.isfinite(ev).all():
                raise ValueError(f"{eid}: {int((~np.isfinite(ev)).sum())} of {len(ev)} "
                                 f"back-extension months from {uf_train_start} missing "
                                 "in the store")
            vals[ext] = ev
            n_ext[i] = int(ext.sum())
        obs_m[j] = vals
    var_m = (np.nanvar(obs_m, axis=1) if len(monthly_rows)
             else np.zeros(0))

    if (np.concatenate([var_d, var_m]) <= 0).any():
        raise ValueError("zero observed variance for at least one entity")

    return EntityObs(
        t0=t0, t1=t1, family=tuple(reg["family"]),
        daily_rows=np.array(daily_rows, dtype=np.int64),
        obs_daily=torch.as_tensor(obs_d).to(dom.device, dom.dtype),
        var_daily=torch.as_tensor(var_d).to(dom.device, dom.dtype),
        monthly_rows=np.array(monthly_rows, dtype=np.int64),
        month_code=month_code,
        obs_monthly=torch.as_tensor(obs_m).to(dom.device, dom.dtype),
        var_monthly=torch.as_tensor(var_m).to(dom.device, dom.dtype),
        n_masked=tuple(n_masked),
        n_holdout=tuple(n_holdout),
        n_ext=tuple(n_ext),
    )


def monthly_chunk_target(
    dates: pd.DatetimeIndex, c0: int, length: int,
    cal_t0: int, cal_t1: int, month_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Day->month bucket for one TBPTT chunk plus each slot's COLUMN in the
    envelope month grid (:attr:`EntityObs.month_code`).

    Wraps :func:`sacsma.dpl.data.month_chunk_target` — a calendar month gets a
    slot only when it lies completely inside both the chunk and the cal
    window, so split/partial months never enter the loss (a fixed 366-day grid
    from 1 Oct 1949 splits one month most years; the water-year grid,
    ``DplConfig.chunk_grid``, splits none) — and recovers each slot's absolute
    month, so the simulated monthly sum lands in the right ``obs_monthly``
    column regardless of where the chunk grid cuts.  Returns numpy ``(bucket
    (length, maxm), cols (maxm,), mask (maxm,))``; masked slots have ``cols``
    0 (gate on ``mask`` before comparing).
    """
    bucket, _, mask = month_chunk_target(dates, c0, length, cal_t0, cal_t1)
    if mask.sum() >= bucket.shape[1]:
        # all slots used ⇒ a further complete month may have been dropped
        # silently (month_chunk_target caps at maxm without error); DplConfig
        # bounds train_chunk_days <= 366 exactly to keep this unreachable (a
        # 1-Oct-aligned 365/366-day chunk holds twelve complete months)
        raise ValueError(f"monthly bucket slots exhausted for a {length}-day "
                         "chunk — shorten the chunk")
    cols = np.zeros(bucket.shape[1], np.int64)
    for s in np.flatnonzero(mask > 0):
        d = dates[c0 + int(np.flatnonzero(bucket[:, s])[0])]
        col = d.year * 12 + (d.month - 1) - int(month_code[0])
        if not 0 <= col < len(month_code):
            raise ValueError(
                f"chunk month {d.year}-{d.month:02d} outside the envelope")
        cols[s] = col
    return bucket, cols, mask


def monthly_nnse_loss(
    sim_monthly: torch.Tensor,   # (M, maxm) simulated complete-month sums
    tgt: torch.Tensor,           # (M, maxm) observed mm/month, 0 where invalid
    fin: torch.Tensor,           # (M, maxm) 1.0 on finite in-window slots
    var_monthly: torch.Tensor,   # (M,) fixed envelope variances
    min_months: int = 1,
    n_total: float | None = None,
) -> torch.Tensor:
    """Chunk-additive NNSE over the monthly entities — the monthly mirror of
    :func:`sacsma.dpl.loss.masked_basin_loss`: per-entity mean squared error
    over the chunk's valid month slots, normalized by the FIXED per-entity
    variance (:attr:`EntityObs.var_monthly`, so summed chunk losses keep the
    NSE numerator/denominator structure), averaged over entities with at
    least ``min_months`` valid slots.  Branch-free (no host sync).
    ``n_total`` (``mt_share_norm="all"``): divide by this fixed entity count
    instead of the chunk's valid count (``None`` = unchanged)."""
    n = fin.sum(dim=1)
    se = (sim_monthly - tgt) ** 2 * fin
    per = se.sum(dim=1) / n.clamp_min(1.0) / var_monthly.clamp_min(1e-12)
    valid = (n >= min_months).to(per.dtype)
    if n_total is not None:
        return (per * valid).sum() / float(n_total)
    return (per * valid).sum() / valid.sum().clamp_min(1.0)
