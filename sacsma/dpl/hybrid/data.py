"""Assemble the per-basin daily matrix for the hybrid LSTM.

For each of the 15 CDEC basins, on the shared 1915->2018 daily record:

* dynamic features (z-scored on the CAL window): basin area-weighted precip
  (``W @ HRU forcing``), Tmin/Tmax and the physics run's daily flow (none for the
  pure LSTM);
* the observed gage FNF target (``load_gage``, NaN outside gage coverage, the
  confirmed bad gauge days dropped);
* per-basin statics: the physical inputs of a dPL run and the dPL's four climate
  indices;
* the temporal split at :data:`sacsma.cdec15.CAL_END` and the 365-day-lookback
  sample index for training / evaluation;
* optionally one or more CLIMATE-PERTURBED copies of the feature tensor
  (``feat_anchors``) for the response-consistency loss: each anchor shifts
  tmin/tmax by ``dt`` and re-scales precip by ``×(1+dp)`` in normalized
  space, and re-feeds the sim channel from the physics run under the same (dp, dt).

The physics is a 15-CDEC learned run, named in the configuration (``physics``): its daily flow
as trained, and under each anchor's climate, comes from
:func:`sacsma.dpl.evaluate.basin_daily` (the engine, cached).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ... import paths
from ...cdec15 import CAL_END, load_gage
from ..data import load_domain_tensors

SEQ_LEN = 365                     # lookback window (last day = target); == spinup
_CAL_START = "1988-10-01"         # WY1989 (matches DplConfig.cal_start)
DOMAIN = "15cdec_grid"            # the domain of the forcing features
#: dynamic feature order fed to the LSTM (statics are appended by the model)
DYNAMIC_FEATURES: tuple[str, ...] = ("precip", "tmin", "tmax", "sac_sim")


def feature_names(use_sim: bool = True) -> tuple[str, ...]:
    """The active dynamic-channel names, in DYNAMIC_FEATURES order.

    ``use_sim=False`` drops the ``sac_sim`` physics channel entirely — a PURE
    LSTM on the meteorology + statics, the no-physics baseline for the hybrid
    ablation."""
    return DYNAMIC_FEATURES if use_sim else DYNAMIC_FEATURES[:-1]


def apply_response_perturbation(feat: np.ndarray, names: tuple[str, ...], *,
                                dp: float, dt: float, prcp_raw: np.ndarray,
                                norm: dict[str, tuple[float, float]],
                                sim_pert: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return a (B, T, F) copy of ``feat`` under a climate anchor (Δprecip
    fraction ``dp``, ΔT ``dt`` degC).

    The single recipe shared by the training-time response-consistency loss and
    the (dp, dt) response-surface evaluation, so the perturbation is identical
    on both sides:

    * temperatures: additive z-shift ``+= dt/σ`` on tmin/tmax (the
      z-scored channels shift by the raw ΔT over their std);
    * precip: RE-Z-SCORED at ``×(1+dp)`` — ``(prcp_raw*(1+dp) − μ)/σ`` — because
      precip is multiplicative, not a level shift (dp=0 reproduces the channel
      exactly);
    * sim channel: re-fed from ``sim_pert`` (the physics run under the anchor),
      per-basin ``÷scale`` as ``load_hybrid_data`` does.

    ``norm[name] = (μ, σ)`` are the base pooled z-score stats; ``prcp_raw`` and
    ``sim_pert`` are (B, T) mm/day.  At ``dp=0, dt=0`` (with ``sim_pert`` = the
    baseline sim) this returns ``feat`` unchanged."""
    idx = {n: i for i, n in enumerate(names)}
    fa = feat.copy()
    for n in ("tmin", "tmax"):
        fa[:, :, idx[n]] += dt / norm[n][1]
    mu_p, sd_p = norm["precip"]
    fa[:, :, idx["precip"]] = (prcp_raw * (1.0 + dp) - mu_p) / sd_p
    if "sac_sim" in idx:                         # absent for a pure LSTM
        fa[:, :, idx["sac_sim"]] = sim_pert / scale[:, None]
    return fa


def perturbed_static(ing: dict, dp: float, dt: float) -> np.ndarray:
    """Climate-perturbed z-scored statics (B, S) under (Δp, ΔT): the dPL run's physical
    inputs stay fixed and its four climate indices are recomputed (:func:`_basin_climate`).
    Normalized with the present-climate stats in ``ing`` so the trained scaling is reused.
    ``ing`` = ``{phys, W, hrus, forcing, window, s_mean, s_std}`` from
    :func:`load_hybrid_data`."""
    s_p = np.hstack([ing["phys"], _basin_climate(ing, dp, dt)])
    return (s_p - ing["s_mean"]) / ing["s_std"]


def _basin_climate(ing: dict, dp: float, dt: float) -> np.ndarray:
    """(B, 4) the dPL's climate indices (:data:`sacsma.dpl.features.CLIMATE_INDICES`) on the
    dPL run's rows over ``ing["window"]`` in a climate changed by (dp, dt), weighted to the
    basins as the run weighs its rows (``ing["W"]``)."""
    import dataclasses

    from ..features import CLIMATE_INDICES, climate_indices

    f = ing["forcing"]
    if dp or dt:
        f = dataclasses.replace(f, prcp=f.prcp * (1.0 + dp), tavg=f.tavg + dt)
    ci = climate_indices(ing["hrus"], f, window=ing["window"])
    return ing["W"] @ ci[list(CLIMATE_INDICES)].to_numpy(np.float64)


@dataclass
class HybridData:
    """Everything the hybrid trainer/evaluator need, all aligned on ``dates``."""

    dates: pd.DatetimeIndex         # (T,)
    feat: torch.Tensor              # (B, T, F) z-scored dynamics
    static: torch.Tensor            # (B, S) z-scored
    obs: torch.Tensor               # (B, T) mm/day, NaN where missing (= target)
    sim: torch.Tensor               # (B, T) mm/day frozen SAC-SMA
    scale: torch.Tensor             # (B,) per-basin cal-window obs std (denorm)
    is_cal: torch.Tensor            # (T,) bool: the training water years (selection split)
    is_val: torch.Tensor            # (T,) bool: the test years WY2004-18
    train_bt: torch.Tensor          # (M, 2) [basin, day] training samples
    basins: tuple[str, ...]
    device: torch.device
    #: per-anchor perturbed copies of ``feat`` and the matching physics teacher
    #: sim (mm/day), one per (Δprecip, ΔT) response anchor — empty unless a
    #: response-consistency loss is on.  ``feat_anchors[i]`` shifts temps by
    #: ΔT/σ, re-z-scores precip at ×(1+Δp), and re-feeds
    #: the sim channel from ``sim_anchors[i]`` (the physics teacher under the
    #: anchor).  Built by :func:`apply_response_perturbation`.
    feat_anchors: list = field(default_factory=list)
    sim_anchors: list = field(default_factory=list)
    #: base pooled z-score stats ``{channel: (μ, σ)}`` and the raw basin precip
    #: (B, T) mm/day — reused by the (dp, dt) response surfaces to rebuild
    #: perturbed features with the exact trained normalization
    #: (:func:`apply_response_perturbation`).
    norm: dict = field(default_factory=dict)
    prcp: "torch.Tensor | None" = None
    #: per-anchor climate-perturbed statics (parallel to feat_anchors), and the
    #: raw ingredients ({phys,W,hrus,forcing,window,s_mean,s_std}) to rebuild perturbed
    #: statics at (Δp,ΔT) eval time via :func:`perturbed_static`.
    static_anchors: list = field(default_factory=list)
    static_ing: dict = field(default_factory=dict)
    #: the active dynamic-channel names, in ``feat``'s order (:func:`feature_names`)
    names: tuple = ()

    @property
    def n_feat(self) -> int:
        return self.feat.shape[-1]

    @property
    def n_static(self) -> int:
        return self.static.shape[-1]

    def gather_windows(self, b_idx: torch.Tensor, t_idx: torch.Tensor,
                       feat: torch.Tensor | None = None) -> torch.Tensor:
        """(K,) basin & end-day indices -> (K, SEQ_LEN, F) lookback windows
        (from ``feat`` if given — e.g. ``feat_dt`` — else ``self.feat``)."""
        src = self.feat if feat is None else feat
        rel = torch.arange(-SEQ_LEN + 1, 1, device=self.device)      # (SEQ_LEN,)
        tt = t_idx[:, None] + rel[None, :]                           # (K, SEQ_LEN)
        bb = b_idx[:, None].expand(-1, SEQ_LEN)                      # (K, SEQ_LEN)
        return src[bb, tt]                                           # (K, SEQ_LEN, F)

    def eval_days(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(basin, day) index of every full-lookback day in cal|val|all."""
        T = len(self.dates)
        day = torch.arange(SEQ_LEN - 1, T, device=self.device)
        if split == "cal":
            day = day[self.is_cal[day]]
        elif split == "val":
            day = day[self.is_val[day]]
        elif split != "all":
            raise ValueError(split)
        b = torch.arange(len(self.basins), device=self.device)
        bb = b[:, None].expand(-1, len(day)).reshape(-1)
        tt = day[None, :].expand(len(self.basins), -1).reshape(-1)
        return bb, tt


def load_hybrid_data(
    data_dir: str = "data",
    *,
    physics: str = "",
    statics_from: str,
    use_sim: bool = True,
    response_anchors: tuple = (),
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> HybridData:
    """The hybrid's data on the basins of :data:`DOMAIN`: forcing, the daily flow of the
    15-CDEC run ``physics`` (with ``use_sim``), the gauges, per-basin statics, and one perturbed
    copy per response anchor (``{"dp", "dt"}``).  The statics are the inputs of
    ``statics_from`` (a dPL checkpoint of the ``physical`` variant) as it trained them and the
    dPL's four climate indices over the training window, both weighted to the basins as the run
    weighs its rows.  The confirmed bad gauge days
    (:func:`sacsma.dpl.evaluate.daily_obs_mask`) are dropped from the targets, the selection and
    the scores."""
    if use_sim and not physics:
        raise ValueError("the physics channel needs a physics run (use_sim=False for a pure LSTM)")
    if not use_sim and response_anchors:
        raise ValueError("use_sim=False (pure LSTM) is incompatible with response "
                         "anchors — the response-consistency loss needs the "
                         "physics sim channel")
    device = torch.device(device)
    dom = load_domain_tensors(data_dir, domain=DOMAIN, device="cpu",
                              dtype=torch.float64)
    dates = dom.dates
    T = len(dates)
    basins = dom.basins
    B = len(basins)

    # -- basin-level forcing (area-weighted HRU -> outlet) --------------------
    W = dom.W.numpy()                                       # (B, N) rows sum 1
    prcp = W @ dom.forcing.prcp[dom.cell_idx].astype(np.float64)   # (B, T) mm/day

    # basin-average Tmin/Tmax (same W-average as prcp, pre-ingested to CSV)
    tmm = pd.read_csv(paths.basin_tminmax(data_dir),
                      parse_dates=["date"]).set_index("date")
    tmin = np.vstack([tmm[f"tmin_{b}"].reindex(dates).to_numpy(np.float64)
                      for b in basins])                     # (B, T) degC
    tmax = np.vstack([tmm[f"tmax_{b}"].reindex(dates).to_numpy(np.float64)
                      for b in basins])                     # (B, T) degC

    # -- the physics run's flow (skipped for a pure LSTM: no physics channel) ----
    def _sim(dp: float = 0.0, dt: float = 0.0) -> np.ndarray:
        from ..evaluate import basin_daily

        df = basin_daily(physics, data_dir=data_dir, dp=dp, dt=dt)
        return np.vstack([df[b].reindex(dates).to_numpy(np.float64) for b in basins])

    if use_sim:
        sim = _sim()
    else:
        sim = np.zeros((B, T))   # placeholder: no sac_sim channel, no anchors

    # -- observed gage FNF ----------------------------------------------------
    gage = load_gage(data_dir)
    obs = np.full((B, T), np.nan)
    for i, b in enumerate(basins):
        g = gage[gage["basin"] == b].set_index("date")["flow"]
        obs[i] = g.reindex(dates).to_numpy(np.float64)
    from ..evaluate import daily_obs_mask
    for m in daily_obs_mask(data_dir):
        e, _, day = m.partition("|")
        b = e.removeprefix("cdec_")
        if b in basins and pd.Timestamp(day) in dates:
            obs[basins.index(b), dates.get_loc(pd.Timestamp(day))] = np.nan

    # -- splits ---------------------------------------------------------------
    from ..evaluate import CAL_WINDOW, VAL_WINDOW
    is_cal = np.asarray((dates >= pd.Timestamp(CAL_WINDOW[0]))
                        & (dates <= pd.Timestamp(CAL_WINDOW[1])))
    is_val = np.asarray((dates >= pd.Timestamp(VAL_WINDOW[0]))
                        & (dates <= pd.Timestamp(VAL_WINDOW[1])))
    cal_lo = int(dates.searchsorted(pd.Timestamp(_CAL_START)))
    cal_hi = int(dates.searchsorted(pd.Timestamp(CAL_END))) + 1
    cal_slc = slice(cal_lo, cal_hi)                         # WY1989..2003 training

    # -- per-basin denorm scale (cal-window obs std) ---------------------------
    # (computed BEFORE the dynamic features so the sim channel can share the
    #  target's per-basin scale — see below.)
    tw = obs[:, cal_slc]
    scale = np.array([np.nanstd(tw[i]) for i in range(B)]) + 1e-8

    # -- dynamic features -----------------------------------------------------
    # precip/tmin/tmax: pooled z-score (cross-basin forcing comparability).
    # sac_sim: the net must OUTPUT flow, and it can only reproduce the physics
    # baseline (flow = sim) if the sim channel shares the target's per-basin
    # scale — otherwise "copy the physics" demands a per-basin ÷scale[b] the
    # pooled, entity-blind net cannot represent.  So sim is scaled by the
    # per-basin target std, matching obs/scale[b].
    dyn = {"precip": prcp, "tmin": tmin, "tmax": tmax}
    names = feature_names(use_sim)
    feat_cols = []
    mu_pooled: dict[str, float] = {}                       # cal-window pooled stats
    sd_pooled: dict[str, float] = {}
    for name in names:
        if name == "sac_sim":
            feat_cols.append(sim / scale[:, None])         # per-basin, target-matched
        else:
            a = dyn[name]
            mu = a[:, cal_slc].mean()
            sd = a[:, cal_slc].std() + 1e-8
            mu_pooled[name] = float(mu)
            sd_pooled[name] = float(sd)
            feat_cols.append((a - mu) / sd)
    feat = np.stack(feat_cols, axis=-1)                    # (B, T, F)

    # -- response-perturbed copies (response-consistency loss) ----------------
    # One per (Δprecip, ΔT) anchor: the SAME recipe the (dp, dt) response-surface
    # evaluation uses (apply_response_perturbation), applied at load time.  Each
    # re-feeds the sim channel from the physics run under the anchor's (dp, dt).
    norm = {n: (mu_pooled[n], sd_pooled[n]) for n in mu_pooled}
    feat_anchors: list[np.ndarray] = []
    sim_anchors: list[np.ndarray] = []
    for anc in response_anchors:
        sim_a = _sim(float(anc["dp"]), float(anc["dt"]))
        feat_a = apply_response_perturbation(
            feat, names, dp=float(anc["dp"]), dt=float(anc["dt"]),
            prcp_raw=prcp, norm=norm, sim_pert=sim_a, scale=scale)
        feat_anchors.append(feat_a)
        sim_anchors.append(sim_a)

    # -- training sample index (finite obs, full lookback, cal window) --------
    finite = np.isfinite(obs)
    valid = finite.copy()
    valid[:, :SEQ_LEN - 1] = False
    valid[:, :cal_lo] = False
    valid[:, cal_hi:] = False
    b_ix, t_ix = np.nonzero(valid)
    train_bt = np.stack([b_ix, t_ix], axis=1)

    # -- per-basin statics (z-scored across basins) ---------------------------
    # the run's physical inputs are fixed; its four climate indices co-vary with a
    # response anchor's (Δp,ΔT) (see perturbed_static).
    from ..evaluate import _loaded

    _, x, sdom, _, sck = _loaded(str(paths.dpl_checkpoint(run=statics_from).resolve()),
                                 str(data_dir))
    if sck["variant"] != "physical":
        raise ValueError(f"statics_from: a physical-variant run, not {sck['variant']!r}")
    sbasins = [str(b).removeprefix("cdec_") for b in sdom.basins]
    Ws = sdom.W.numpy()[[sbasins.index(b) for b in basins]]
    static_ing = dict(phys=Ws @ x.numpy(), W=Ws, hrus=sdom.hrus, forcing=sdom.forcing,
                      window=(_CAL_START, CAL_END))
    s = np.hstack([static_ing["phys"], _basin_climate(static_ing, 0.0, 0.0)])
    static_ing.update(s_mean=s.mean(axis=0), s_std=s.std(axis=0) + 1e-8)
    static = torch.as_tensor((s - static_ing["s_mean"]) / static_ing["s_std"]).to(device, dtype)
    static_np_anchors = [perturbed_static(static_ing, float(anc["dp"]), float(anc["dt"]))
                         for anc in response_anchors]      # parallel to feat_anchors

    def _t(a):
        return torch.as_tensor(a).to(device, dtype)

    return HybridData(
        dates=dates,
        feat=_t(feat), static=static,
        obs=_t(obs), sim=_t(sim), scale=_t(scale),
        is_cal=torch.as_tensor(is_cal).to(device),
        is_val=torch.as_tensor(is_val).to(device),
        train_bt=torch.as_tensor(train_bt, dtype=torch.long).to(device),
        basins=basins, device=device,
        feat_anchors=[_t(fa) for fa in feat_anchors],
        sim_anchors=[_t(sa) for sa in sim_anchors],
        norm=norm, prcp=_t(prcp),
        static_anchors=[_t(sa) for sa in static_np_anchors],
        static_ing=static_ing,
        names=tuple(names),
    )


def data_for(ck: dict, data_dir: str = "data", device: torch.device | str = "cuda",
             **kw) -> HybridData:
    """:func:`load_hybrid_data` for a hybrid checkpoint, as it was trained (``kw``: e.g. the
    response anchors)."""
    cfg = ck["cfg"]
    return load_hybrid_data(data_dir, physics=cfg["physics"], statics_from=cfg["statics_from"],
                            use_sim=cfg["use_sim"], device=device, **kw)
