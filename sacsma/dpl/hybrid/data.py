"""Assemble the per-basin daily matrix for the hybrid LSTM.

For each of the 15 CDEC basins, on the shared 1915->2018 daily record:

* dynamic features (z-scored on the CAL window): basin area-weighted precip &
  tavg (``W @ HRU forcing``), Tmin/Tmax, the frozen SAC-SMA sim, and (optional,
  off for the canonical hybrid) sin/cos day-of-year;
* the observed gage FNF target (``load_gage``, NaN outside gage coverage);
* optional per-basin statics (elev, flowlen, cal precip mean, snow fraction);
* the temporal split at :data:`sacsma.cdec15.CAL_END` and the 365-day-lookback
  sample index for training / evaluation;
* optionally one or more CLIMATE-PERTURBED copies of the feature tensor
  (``feat_anchors``) for the response-consistency loss: each anchor shifts
  tavg/tmin/tmax by ``dt`` and re-scales precip by ``×(1+dp)`` in normalized
  space, recomputes PET under ``dt``, and re-feeds the sim channel from the
  physics run under the same (dp, dt).

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
_OMEGA = 2.0 * np.pi / 365.0
_CAL_START = "1988-10-01"         # WY1989 (matches DplConfig.cal_start)
#: dynamic feature order fed to the LSTM (statics are appended by the model).
#: ``pet`` (raw PT potential — the physics' energy-demand signal) and the
#: sin/cos day-of-year are OPT-IN; see :func:`feature_names`.
DYNAMIC_FEATURES: tuple[str, ...] = (
    "precip", "tavg", "tmin", "tmax", "pet", "sac_sim", "sin_doy", "cos_doy")


def feature_names(use_doy: bool, use_pet: bool,
                  use_sim: bool = True) -> tuple[str, ...]:
    """The active dynamic-channel names, in DYNAMIC_FEATURES order.

    Every consumer that needs channel indices (training, evaluation, the
    forcing-sensitivity counterfactuals) derives them from THIS filter so old
    checkpoints (no pet, doy on) keep their layout.

    ``use_sim=False`` drops the ``sac_sim`` physics channel entirely — a PURE
    LSTM on the meteorology (+ optional PET + statics), the no-physics baseline
    for the hybrid ablation."""
    drop = set()
    if not use_doy:
        drop |= {"sin_doy", "cos_doy"}
    if not use_pet:
        drop.add("pet")
    if not use_sim:
        drop.add("sac_sim")
    return tuple(n for n in DYNAMIC_FEATURES if n not in drop)

def basin_pet_pt(dom, delta_t: float | np.ndarray = 0.0) -> np.ndarray:
    """(B, T) basin-average raw PT PET (mm/day) — alb 0 / dew 0, i.e. exactly
    the potential the noah physics sees, BEFORE the learned Kpet.

    Recomputed from the per-cell forcing + tmin/tmax sidecar (deterministic and
    parameter-free), so any counterfactual is exact: ``delta_t`` shifts all
    three temperatures — a scalar degC (warming), or a per-forcing-row
    ``(rows, T)`` field (e.g. the WGEN detrending delta)."""
    from ...pet_pt import pt_raw_pet

    if dom.tmin is None or dom.tmax is None:
        raise ValueError("the pet input channel needs the per-cell tmin/tmax "
                         "sidecar (15cdec_grid domain)")
    d = (np.asarray(delta_t)[dom.cell_idx]
         if not isinstance(delta_t, (int, float)) else float(delta_t))
    tavg = dom.forcing.tavg[dom.cell_idx].astype(np.float64) + d
    tmin = dom.tmin[dom.cell_idx].astype(np.float64) + d
    tmax = dom.tmax[dom.cell_idx].astype(np.float64) + d
    doy = dom.forcing.doy.astype(np.float64)
    lat = dom.hrus["lat"].to_numpy(np.float64)
    elev = dom.hrus["elev"].to_numpy(np.float64)
    pet = np.empty_like(tavg)
    for i in range(tavg.shape[0]):
        pet[i] = pt_raw_pet(tavg[i], tmin[i], tmax[i], doy, lat[i], elev[i])
    return dom.W.numpy() @ pet


def apply_response_perturbation(feat: np.ndarray, names: tuple[str, ...], *,
                                dp: float, dt: float, prcp_raw: np.ndarray,
                                norm: dict[str, tuple[float, float]],
                                pet_pert: np.ndarray | None,
                                sim_pert: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return a (B, T, F) copy of ``feat`` under a climate anchor (Δprecip
    fraction ``dp``, ΔT ``dt`` degC).

    The single recipe shared by the training-time response-consistency loss and
    the (dp, dt) response-surface evaluation, so the perturbation is identical
    on both sides:

    * temperatures: additive z-shift ``+= dt/σ`` on tavg/tmin/tmax (the
      z-scored channels shift by the raw ΔT over their std);
    * precip: RE-Z-SCORED at ``×(1+dp)`` — ``(prcp_raw*(1+dp) − μ)/σ`` — because
      precip is multiplicative, not a level shift (dp=0 reproduces the channel
      exactly);
    * PET (if present): recomputed under ΔT (``pet_pert``) and re-z-scored (PET
      is a deterministic function of T; dp is irrelevant to it);
    * sim channel: re-fed from ``sim_pert`` (the physics run under the anchor),
      per-basin ``÷scale`` as ``load_hybrid_data`` does.

    ``norm[name] = (μ, σ)`` are the base pooled z-score stats; ``prcp_raw`` and
    ``pet_pert`` are (B, T) mm/day; ``sim_pert`` is (B, T) mm/day.  At
    ``dp=0, dt=0`` (with ``sim_pert`` = the baseline sim) this returns ``feat``
    unchanged."""
    idx = {n: i for i, n in enumerate(names)}
    fa = feat.copy()
    for n in ("tavg", "tmin", "tmax"):
        fa[:, :, idx[n]] += dt / norm[n][1]
    if "precip" in idx:
        mu_p, sd_p = norm["precip"]
        fa[:, :, idx["precip"]] = (prcp_raw * (1.0 + dp) - mu_p) / sd_p
    if "pet" in idx and pet_pert is not None:
        mu_e, sd_e = norm["pet"]
        fa[:, :, idx["pet"]] = (pet_pert - mu_e) / sd_e
    if "sac_sim" in idx:                         # absent for a pure LSTM
        fa[:, :, idx["sac_sim"]] = sim_pert / scale[:, None]
    return fa


def perturbed_static(ing: dict, dp: float, dt: float) -> np.ndarray:
    """Climate-perturbed z-scored statics (B, 4) under (Δp, ΔT): the two CLIMATE
    statics co-vary — ``pmean → pmean×(1+dp)`` and ``snowf`` is recomputed with the
    (tavg+ΔT) freezing mask (snowf is invariant to uniform precip scaling — the
    (1+dp) cancels); ``elev``/``flowlen`` stay fixed.  Normalized with the
    present-climate stats in ``ing`` so the trained scaling is reused.  ``ing`` =
    ``{elev, flen, pcal, tcal, s_mean, s_std}`` from :func:`load_hybrid_data`."""
    pcal, tcal = ing["pcal"], ing["tcal"]
    pmean_p = pcal.mean(axis=1) * (1.0 + dp)
    snowf_p = ((pcal * ((tcal + dt) <= 0.0)).sum(axis=1)
               / np.maximum(pcal.sum(axis=1), 1e-9))
    s_p = np.stack([ing["elev"], ing["flen"], pmean_p, snowf_p], axis=1)
    return (s_p - ing["s_mean"]) / ing["s_std"]


@dataclass
class HybridData:
    """Everything the hybrid trainer/evaluator need, all aligned on ``dates``."""

    dates: pd.DatetimeIndex         # (T,)
    feat: torch.Tensor              # (B, T, F) z-scored dynamics (+ sin/cos doy)
    static: torch.Tensor | None     # (B, S) z-scored, or None
    obs: torch.Tensor               # (B, T) mm/day, NaN where missing (= target)
    sim: torch.Tensor               # (B, T) mm/day frozen SAC-SMA
    scale: torch.Tensor             # (B,) per-basin cal-window obs std (denorm)
    is_cal: torch.Tensor            # (T,) bool: date <= CAL_END (scoring split)
    train_bt: torch.Tensor          # (M, 2) [basin, day] training samples
    basins: tuple[str, ...]
    device: torch.device
    #: per-anchor perturbed copies of ``feat`` and the matching physics teacher
    #: sim (mm/day), one per (Δprecip, ΔT) response anchor — empty unless a
    #: response-consistency loss is on.  ``feat_anchors[i]`` shifts temps by
    #: ΔT/σ, re-z-scores precip at ×(1+Δp), recomputes PET under ΔT, and re-feeds
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
    #: raw ingredients ({elev,flen,pcal,tcal,s_mean,s_std}) to rebuild perturbed
    #: statics at (Δp,ΔT) eval time via :func:`perturbed_static`.
    static_anchors: list = field(default_factory=list)
    static_ing: dict = field(default_factory=dict)

    @property
    def n_feat(self) -> int:
        return self.feat.shape[-1]

    @property
    def n_static(self) -> int:
        return 0 if self.static is None else self.static.shape[-1]

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
    physics: str = "",
    use_statics: bool = False,
    use_doy: bool = True,
    use_pet: bool = False,
    use_sim: bool = True,
    domain: str = "15cdec",
    response_anchors: tuple = (),
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> HybridData:
    """The hybrid's data on ``domain``'s basins: forcing, the daily flow of the 15-CDEC run
    ``physics`` (with ``use_sim``), the gauges, and one perturbed copy per response anchor
    (``{"dp", "dt"}``)."""
    if use_sim and not physics:
        raise ValueError("the physics channel needs a physics run (use_sim=False for a pure LSTM)")
    if not use_sim and response_anchors:
        raise ValueError("use_sim=False (pure LSTM) is incompatible with response "
                         "anchors — the response-consistency loss needs the "
                         "physics sim channel")
    device = torch.device(device)
    dom = load_domain_tensors(data_dir, domain=domain, device="cpu",
                              dtype=torch.float64)
    dates = dom.dates
    T = len(dates)
    basins = dom.basins
    B = len(basins)

    # -- basin-level forcing (area-weighted HRU -> outlet) --------------------
    W = dom.W.numpy()                                       # (B, N) rows sum 1
    prcp = W @ dom.forcing.prcp[dom.cell_idx].astype(np.float64)   # (B, T) mm/day
    tavg = W @ dom.forcing.tavg[dom.cell_idx].astype(np.float64)   # (B, T) degC

    # basin-average Tmin/Tmax (same W-average as tavg, pre-ingested to CSV)
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

    # -- splits ---------------------------------------------------------------
    is_cal = np.asarray(dates <= pd.Timestamp(CAL_END))
    cal_lo = int(dates.searchsorted(pd.Timestamp(_CAL_START)))
    cal_hi = int(dates.searchsorted(pd.Timestamp(CAL_END))) + 1
    cal_slc = slice(cal_lo, cal_hi)                         # WY1989..2003 training

    # -- per-basin denorm scale (cal-window obs std) ---------------------------
    # (computed BEFORE the dynamic features so the sim channel can share the
    #  target's per-basin scale — see below.)
    tw = obs[:, cal_slc]
    scale = np.array([np.nanstd(tw[i]) for i in range(B)]) + 1e-8

    # -- dynamic features -----------------------------------------------------
    # precip/tavg/tmin/tmax: pooled z-score (cross-basin forcing comparability).
    # sac_sim: the net must OUTPUT flow, and it can only reproduce the physics
    # baseline (flow = sim) if the sim channel shares the target's per-basin
    # scale — otherwise "copy the physics" demands a per-basin ÷scale[b] the
    # pooled, entity-blind net cannot represent.  So sim is scaled by the
    # per-basin target std, matching obs/scale[b].
    doy = dom.forcing.doy.astype(np.float64)               # (T,)
    sin_doy = np.sin(_OMEGA * doy)
    cos_doy = np.cos(_OMEGA * doy)
    dyn = {"precip": prcp, "tavg": tavg, "tmin": tmin, "tmax": tmax, "sac_sim": sim}
    # use_pet adds the raw PT potential (the physics' energy-demand signal) as
    # an input channel — recomputed from the forcing (deterministic, cached).
    if use_pet:
        pet_cache = paths.local(name="cache/climatology") / f"basin_pet_pt_{domain}.csv"
        if pet_cache.exists():
            pdf = pd.read_csv(pet_cache, parse_dates=["date"]).set_index("date")
            pet_b = np.vstack([pdf[b].reindex(dates).to_numpy(np.float64)
                               for b in basins])
        else:
            pet_b = basin_pet_pt(dom)
            pet_cache.parent.mkdir(parents=True, exist_ok=True)
            pdf = pd.DataFrame(pet_b.T, index=dates, columns=list(basins))
            pdf.index.name = "date"
            pdf.to_csv(pet_cache)
            print(f"hybrid: cached basin PT PET -> {pet_cache}", flush=True)
        dyn["pet"] = pet_b
    # use_doy=False drops the sin/cos day-of-year channels: the sim channel
    # already carries the calendar (Snow-17 melt sinusoid, seasonal LAI, PT
    # radiation), and an explicit doy input is what lets the net learn a
    # calendar-keyed mean correction that carries unchecked into validation.
    names = feature_names(use_doy, use_pet, use_sim)
    feat_cols = []
    mu_pooled: dict[str, float] = {}                       # cal-window pooled stats
    sd_pooled: dict[str, float] = {}
    for name in names:
        if name == "sac_sim":
            feat_cols.append(sim / scale[:, None])         # per-basin, target-matched
        elif name in dyn:
            a = dyn[name]
            mu = a[:, cal_slc].mean()
            sd = a[:, cal_slc].std() + 1e-8
            mu_pooled[name] = float(mu)
            sd_pooled[name] = float(sd)
            feat_cols.append((a - mu) / sd)
        elif name == "sin_doy":
            feat_cols.append(np.broadcast_to(sin_doy, (B, T)))
        elif name == "cos_doy":
            feat_cols.append(np.broadcast_to(cos_doy, (B, T)))
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
        pet_a = basin_pet_pt(dom, delta_t=float(anc["dt"])) if use_pet else None
        feat_a = apply_response_perturbation(
            feat, names, dp=float(anc["dp"]), dt=float(anc["dt"]),
            prcp_raw=prcp, norm=norm, pet_pert=pet_a, sim_pert=sim_a, scale=scale)
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

    # -- optional per-basin statics (z-scored across basins) ------------------
    # elev/flowlen are physiographic (fixed); pmean/snowf are CLIMATE statics
    # that co-vary with a response anchor's (Δp,ΔT) (see perturbed_static).
    static = None
    static_np_anchors: list[np.ndarray] = []
    static_ing: dict = {}
    if use_statics:
        elev = W @ dom.elev.numpy()                        # area-wt mean elev (m)
        flen = W @ dom.flowlen.numpy()                     # area-wt mean flowlen
        pcal = prcp[:, cal_slc]                            # (B, n_cal) mm/day
        tcal = tavg[:, cal_slc]                            # (B, n_cal) degC
        pmean = pcal.mean(axis=1)                          # cal mean daily precip
        snowf = ((pcal * (tcal <= 0.0)).sum(axis=1)
                 / np.maximum(pcal.sum(axis=1), 1e-9))     # snow fraction
        s = np.stack([elev, flen, pmean, snowf], axis=1)   # (B, 4)
        s_mean = s.mean(axis=0)
        s_std = s.std(axis=0) + 1e-8
        static = torch.as_tensor((s - s_mean) / s_std).to(device, dtype)
        static_ing = dict(elev=elev, flen=flen, pcal=pcal, tcal=tcal,
                          s_mean=s_mean, s_std=s_std)
        for anc in response_anchors:                       # parallel to feat_anchors
            static_np_anchors.append(perturbed_static(
                static_ing, float(anc["dp"]), float(anc["dt"])))

    def _t(a):
        return torch.as_tensor(a).to(device, dtype)

    return HybridData(
        dates=dates,
        feat=_t(feat), static=static,
        obs=_t(obs), sim=_t(sim), scale=_t(scale),
        is_cal=torch.as_tensor(is_cal).to(device),
        train_bt=torch.as_tensor(train_bt, dtype=torch.long).to(device),
        basins=basins, device=device,
        feat_anchors=[_t(fa) for fa in feat_anchors],
        sim_anchors=[_t(sa) for sa in sim_anchors],
        norm=norm, prcp=_t(prcp),
        static_anchors=[_t(sa) for sa in static_np_anchors],
        static_ing=static_ing,
    )


def data_for(ck: dict, data_dir: str = "data", device: torch.device | str = "cuda",
             **kw) -> HybridData:
    """:func:`load_hybrid_data` for a hybrid checkpoint, as it was trained (``kw``: e.g. the
    response anchors)."""
    if ck.get("variant", "feature") != "feature":
        raise ValueError("residual hybrid checkpoints are retired; only feature checkpoints "
                         "can be scored")
    cfg = ck["cfg"]
    return load_hybrid_data(
        data_dir, physics=cfg.get("physics", ""), use_statics=bool(ck["n_static"]),
        use_doy=cfg.get("use_doy", True), use_pet=cfg.get("use_pet", False),
        use_sim=cfg.get("use_sim", True), domain=cfg.get("physics_domain", "15cdec"),
        device=device, **kw)
