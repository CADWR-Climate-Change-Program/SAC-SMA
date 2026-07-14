"""Assemble the per-basin daily matrix for the hybrid LSTM.

For each of the 15 CDEC basins, on the shared 1915->2018 daily record:

* dynamic features (z-scored on the CAL window): basin area-weighted precip &
  tavg (``W @ HRU forcing``), the frozen SAC-SMA sim, and sin/cos day-of-year;
* the observed gage FNF target (``load_gage``, NaN outside gage coverage);
* optional per-basin statics (elev, flowlen, cal precip mean, snow fraction);
* the temporal split at :data:`sacsma.cdec15.CAL_END` and the 365-day-lookback
  sample index for training / evaluation.

The frozen physics comes from ``run_basin`` under a chosen parameter table
(dPL ``physical_levers``, the seasonal winner, or GA) — cached to CSV so the
~15 basin runs happen once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...cdec15 import BASINS, CAL_END, load_gage
from ...model import load_domain_forcing, run_basin
from ..data import load_domain_tensors

SEQ_LEN = 365                     # lookback window (last day = target); == spinup
_OMEGA = 2.0 * np.pi / 365.0
_CAL_START = "1988-10-01"         # WY1989 (matches DplConfig.cal_start)
#: dynamic feature order fed to the LSTM (statics are appended by the model).
DYNAMIC_FEATURES: tuple[str, ...] = (
    "precip", "tavg", "tmin", "tmax", "sac_sim", "sin_doy", "cos_doy")
#: basin-average daily Tmin/Tmax (pre-ingested from the WGEN 1/16-deg grid;
#: see scratchpad/ingest_tminmax.py).  Adds the diurnal-range signal a single
#: tavg discards; basin tavg reproduces the stored forcing to 0.37 degC.
TMINMAX_CSV = "basin_tminmax_livneh.csv"


def build_frozen_sim(
    data_dir: str = "data",
    physics_csv: str | Path | None = None,
    cache: str | Path | None = None,
) -> pd.DataFrame:
    """Frozen SAC-SMA daily sim (mm/day) for all 15 basins: index=date, cols=basins.

    ``physics_csv`` = a ga_optimum-shaped parameter table (with a ``basin``
    column) e.g. ``artifacts/dpl/testing/physical_levers/params_dpl.csv``; ``None`` uses
    the archived GA optimum.  Cached to ``cache`` if given.
    """
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")
    forcing = load_domain_forcing(data_dir, domain="15cdec")
    params = pd.read_csv(physics_csv) if physics_csv is not None else None
    cols = {}
    for b in BASINS:
        s = run_basin(b, data_dir=data_dir, domain="15cdec", forcing=forcing,
                      params=params, parallel=(params is None or
                                               not _has_seasonal(params)))
        cols[b] = s.set_index("date")["flow"]
    df = pd.DataFrame(cols)
    df.index.name = "date"
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    return df


def _has_seasonal(params: pd.DataFrame) -> bool:
    return any(str(c).endswith("_asin") for c in params.columns)


@dataclass
class HybridData:
    """Everything the hybrid trainer/evaluator need, all aligned on ``dates``."""

    variant: str                    # "feature" | "residual"
    dates: pd.DatetimeIndex         # (T,)
    feat: torch.Tensor              # (B, T, F) z-scored dynamics + sin/cos doy
    static: torch.Tensor | None     # (B, S) z-scored, or None
    obs: torch.Tensor               # (B, T) mm/day, NaN where missing
    sim: torch.Tensor               # (B, T) mm/day frozen SAC-SMA
    target: torch.Tensor            # (B, T) mm/day: obs (feature) or obs-sim (residual)
    scale: torch.Tensor             # (B,) per-basin cal-window target std (denorm)
    is_cal: torch.Tensor            # (T,) bool: date <= CAL_END (scoring split)
    train_bt: torch.Tensor          # (M, 2) [basin, day] training samples
    basins: tuple[str, ...]
    device: torch.device

    @property
    def n_feat(self) -> int:
        return self.feat.shape[-1]

    @property
    def n_static(self) -> int:
        return 0 if self.static is None else self.static.shape[-1]

    def gather_windows(self, b_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """(K,) basin & end-day indices -> (K, SEQ_LEN, F) lookback windows."""
        rel = torch.arange(-SEQ_LEN + 1, 1, device=self.device)      # (SEQ_LEN,)
        tt = t_idx[:, None] + rel[None, :]                           # (K, SEQ_LEN)
        bb = b_idx[:, None].expand(-1, SEQ_LEN)                      # (K, SEQ_LEN)
        return self.feat[bb, tt]                                     # (K, SEQ_LEN, F)

    def eval_days(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(basin, day) index of every full-lookback day in cal|val|all."""
        T = len(self.dates)
        day = torch.arange(SEQ_LEN - 1, T, device=self.device)
        if split == "cal":
            day = day[self.is_cal[day]]
        elif split == "val":
            day = day[~self.is_cal[day]]
        elif split != "all":
            raise ValueError(split)
        b = torch.arange(len(self.basins), device=self.device)
        bb = b[:, None].expand(-1, len(day)).reshape(-1)
        tt = day[None, :].expand(len(self.basins), -1).reshape(-1)
        return bb, tt


def load_hybrid_data(
    data_dir: str = "data",
    *,
    variant: str = "residual",
    physics_csv: str | Path | None = None,
    sim_cache: str | Path | None = None,
    use_statics: bool = False,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> HybridData:
    device = torch.device(device)
    dom = load_domain_tensors(data_dir, device="cpu", dtype=torch.float64)
    dates = dom.dates
    T = len(dates)
    basins = dom.basins
    B = len(basins)

    # -- basin-level forcing (area-weighted HRU -> outlet) --------------------
    W = dom.W.numpy()                                       # (B, N) rows sum 1
    prcp = W @ dom.forcing.prcp[dom.cell_idx].astype(np.float64)   # (B, T) mm/day
    tavg = W @ dom.forcing.tavg[dom.cell_idx].astype(np.float64)   # (B, T) degC

    # basin-average Tmin/Tmax (same W-average as tavg, pre-ingested to CSV)
    tmm = pd.read_csv(Path(data_dir) / "cdec15" / TMINMAX_CSV,
                      parse_dates=["date"]).set_index("date")
    tmin = np.vstack([tmm[f"tmin_{b}"].reindex(dates).to_numpy(np.float64)
                      for b in basins])                     # (B, T) degC
    tmax = np.vstack([tmm[f"tmax_{b}"].reindex(dates).to_numpy(np.float64)
                      for b in basins])                     # (B, T) degC

    # -- frozen SAC-SMA sim ---------------------------------------------------
    sim_df = build_frozen_sim(data_dir, physics_csv, cache=sim_cache)
    sim = np.vstack([sim_df[b].reindex(dates).to_numpy(np.float64) for b in basins])

    # -- observed gage FNF ----------------------------------------------------
    gage = load_gage(data_dir)
    obs = np.full((B, T), np.nan)
    for i, b in enumerate(basins):
        g = gage[gage["basin"] == b].set_index("date")["flow"]
        obs[i] = g.reindex(dates).to_numpy(np.float64)

    # -- splits ---------------------------------------------------------------
    is_cal = np.asarray(dates <= pd.Timestamp(CAL_END))
    cal_lo = int(dates.searchsorted(pd.Timestamp(_CAL_START)))
    cal_hi = int(dates.searchsorted(pd.Timestamp(CAL_END))) + 1
    cal_slc = slice(cal_lo, cal_hi)                         # WY1989..2003 training

    # -- target + per-basin denorm scale (cal-window std) ---------------------
    # (computed BEFORE the dynamic features so the FEATURE variant can put its
    #  sim channel on the same per-basin scale as the target — see below.)
    target = obs.copy() if variant == "feature" else (obs - sim)
    tw = target[:, cal_slc]
    scale = np.array([np.nanstd(tw[i]) for i in range(B)]) + 1e-8

    # -- dynamic features -----------------------------------------------------
    # precip/tavg: pooled z-score (cross-basin forcing comparability).  sac_sim:
    # pooled z-score for the RESIDUAL variant (there sim is added back as a
    # physical baseline OUTSIDE the normalization, so its input scaling is free);
    # but the FEATURE variant must OUTPUT flow, and it can only reproduce the
    # physics baseline (flow = sim) if the sim channel shares the target's
    # per-basin scale — otherwise "copy the physics" demands a per-basin
    # ÷scale[b] the pooled, entity-blind net cannot represent.  So the feature
    # variant scales sim by the per-basin target std, matching obs/scale[b].
    doy = dom.forcing.doy.astype(np.float64)               # (T,)
    sin_doy = np.sin(_OMEGA * doy)
    cos_doy = np.cos(_OMEGA * doy)
    dyn = {"precip": prcp, "tavg": tavg, "tmin": tmin, "tmax": tmax, "sac_sim": sim}
    feat_cols = []
    for name in DYNAMIC_FEATURES:
        if name == "sac_sim" and variant == "feature":
            feat_cols.append(sim / scale[:, None])         # per-basin, target-matched
        elif name in dyn:
            a = dyn[name]
            mu = a[:, cal_slc].mean()
            sd = a[:, cal_slc].std() + 1e-8
            feat_cols.append((a - mu) / sd)
        elif name == "sin_doy":
            feat_cols.append(np.broadcast_to(sin_doy, (B, T)))
        elif name == "cos_doy":
            feat_cols.append(np.broadcast_to(cos_doy, (B, T)))
    feat = np.stack(feat_cols, axis=-1)                    # (B, T, F)

    # -- training sample index (finite obs, full lookback, cal window) --------
    finite = np.isfinite(obs)
    valid = finite.copy()
    valid[:, :SEQ_LEN - 1] = False
    valid[:, :cal_lo] = False
    valid[:, cal_hi:] = False
    b_ix, t_ix = np.nonzero(valid)
    train_bt = np.stack([b_ix, t_ix], axis=1)

    # -- optional per-basin statics (z-scored across basins) ------------------
    static = None
    if use_statics:
        elev = W @ dom.elev.numpy()                        # area-wt mean elev (m)
        flen = W @ dom.flowlen.numpy()                     # area-wt mean flowlen
        pmean = prcp[:, cal_slc].mean(axis=1)              # cal mean daily precip
        snowf = ((prcp[:, cal_slc] * (tavg[:, cal_slc] <= 0.0)).sum(axis=1)
                 / prcp[:, cal_slc].sum(axis=1))           # snow fraction
        s = np.stack([elev, flen, pmean, snowf], axis=1)   # (B, 4)
        s = (s - s.mean(axis=0)) / (s.std(axis=0) + 1e-8)
        static = torch.as_tensor(s).to(device, dtype)

    def _t(a):
        return torch.as_tensor(a).to(device, dtype)

    return HybridData(
        variant=variant, dates=dates,
        feat=_t(feat), static=static,
        obs=_t(obs), sim=_t(sim), target=_t(target), scale=_t(scale),
        is_cal=torch.as_tensor(is_cal).to(device),
        train_bt=torch.as_tensor(train_bt, dtype=torch.long).to(device),
        basins=basins, device=device,
    )
