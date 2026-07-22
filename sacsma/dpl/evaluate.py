"""Fidelity benchmark: the archived GA parameters through the torch forward
vs the frozen reference model — the Phase-1 go/no-go gate.

For each named numerics config the whole domain (7891 HRUs) is streamed
through the differentiable pipeline under ``torch.no_grad()`` over the full
1915-2018 record (reference protocol: cold start, no warmup drop), basin flow
aggregated with the exact ``run_basin`` weights, then scored per basin:

* sim-vs-sim daily KGE / NSE / pbias / max|delta| against the frozen truth;
* obs-scored cal/val KGE (split at ``CAL_END``) for both models -> deltas.

Gates (see the dpl plan): G1 structural — best ``ref-ninc*`` config KGE >=
0.999 and |dKGE_obs| <= 0.005 everywhere; G2 training numerics —
``train-default`` KGE >= 0.99, |dKGE_obs| <= 0.02; G3 precision — float32 vs
float64 same-config KGE >= 0.9999.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..cdec15 import BASINS, CAL_END, load_gage
from ..metrics import kge, nse, pbias
from ..model import run_basin
from .config import CANOPY_LEARNED_PARAMS, PARAM_ORDER, DplConfig, pick_device
from .data import DomainTensors, load_domain_tensors
from .forward import initial_state, routing_uh, run_window


def export_params(net: torch.nn.Module, dom: DomainTensors,
                  x: torch.Tensor) -> pd.DataFrame:
    """Learned per-HRU parameters as a ga_optimum-shaped table (+ ``basin``).

    Directly consumable by ``run_basin(..., params=...)``: the exact 34
    ``ga_optimum.csv`` columns keyed by grid-cell ``key``, plus the per-basin
    ``basin`` column the model.py filter uses — the ~1835 cells shared between
    basins intentionally carry per-basin values.  Fixed parameters
    (side/SCF/PXTEMP) come out at their GA constants.
    """
    net.eval()
    with torch.no_grad():
        out = net(x)
    df = dom.hrus[["basin", "key", "lat", "lon"]].copy()
    for p in PARAM_ORDER:
        df[p] = out[p].double().cpu().numpy()
    # seasonal harmonic coefficients (2 extra columns per seasonal param); the
    # frozen run_basin detects the ``_asin`` columns and reconstructs the same
    # day-of-year series.  Absent for a static net -> a plain ga_optimum table.
    for p in getattr(net, "seasonal_params", ()):
        df[f"{p}_asin"] = out[f"{p}_asin"].double().cpu().numpy()
        df[f"{p}_acos"] = out[f"{p}_acos"].double().cpu().numpy()
    return df


def export_canopy_params(net: torch.nn.Module, dom: DomainTensors,
                         x: torch.Tensor) -> pd.DataFrame:
    """Learned per-HRU Noah canopy-ET physiology params keyed by key/basin, plus
    the OBSERVED (pinned) veg_frac and annual-mean LAI for reference.  Kept
    SEPARATE from :func:`export_params` — canopy params must NEVER enter the
    ga_optimum / PARAM_ORDER export (the frozen run_basin has no Noah ET)."""
    net.eval()
    with torch.no_grad():
        cp = net(x)["_canopy"]
    df = dom.hrus[["basin", "key", "lat", "lon"]].copy()
    # LITE nets emit only soil_chi; FULL nets the 7 physiology params.  Use the
    # net's actual learned set (fall back to the full list for older nets).
    learned = getattr(net, "_canopy_learned", CANOPY_LEARNED_PARAMS)
    for p in learned:
        df[p] = cp[p].double().cpu().numpy()
    if dom.veg_frac is not None:                 # pinned observed structure
        df["veg_frac_obs"] = dom.veg_frac.double().cpu().numpy()
    if dom.lai_lut is not None:
        df["lai_obs_mean"] = dom.lai_lut[dom.cell_idx].mean(axis=1)
    return df

#: named numerics configs — (ninc, perc_mode, fracp_floor, dtype)
FIDELITY_CONFIGS: dict[str, DplConfig] = {
    # exact reference numerics (dynamic per-lane ninc): proves the port itself
    "ref-exact": DplConfig(ninc_mode="dynamic", perc_mode="reference", dtype="float64"),
    "ref-ninc1": DplConfig(n_inc=1, perc_mode="reference", dtype="float64"),
    "ref-ninc2": DplConfig(n_inc=2, perc_mode="reference", dtype="float64"),
    "ref-ninc5": DplConfig(n_inc=5, perc_mode="reference", dtype="float64"),
    "ref-ninc10": DplConfig(n_inc=10, perc_mode="reference", dtype="float64"),
    "ref-ninc20": DplConfig(n_inc=20, perc_mode="reference", dtype="float64"),
    "train-default": DplConfig(n_inc=5, perc_mode="implicit", fracp_floor=0.1,
                               dtype="float64"),
    "train-default-f32": DplConfig(n_inc=5, perc_mode="implicit", fracp_floor=0.1,
                                   dtype="float32"),
    "train-tanh": DplConfig(n_inc=5, perc_mode="tanh", fracp_floor=0.1,
                            dtype="float64"),
}

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def frozen_truth(data_dir: str = "data") -> dict[str, pd.DataFrame]:
    """Full-record frozen-model flow per basin (the parity-exact reference)."""
    from ..model import load_domain_forcing

    forcing = load_domain_forcing(data_dir, domain="15cdec")
    out: dict[str, pd.DataFrame] = {}
    for b in BASINS:
        out[b] = run_basin(b, data_dir=data_dir, domain="15cdec",
                           forcing=forcing, parallel=True)
    return out


def torch_domain_flow(
    dom: DomainTensors,
    params: dict[str, torch.Tensor],
    cfg: DplConfig,
    *,
    chunk_days: int = 4096,
    progress: bool = True,
) -> np.ndarray:
    """Stream the full record through the torch pipeline; (B, T) basin flow."""
    n, t_total = dom.n_hru, dom.n_time
    uh = routing_uh(params, dom.flowlen)
    state = initial_state(n, dom.device, dom.dtype, init_mode=cfg.init_mode,
                          params=params)
    basin_flow = torch.empty(len(dom.basins), t_total, device=dom.device,
                             dtype=dom.dtype)
    t0 = 0
    tic = time.time()
    with torch.no_grad():
        while t0 < t_total:
            t1 = min(t0 + chunk_days, t_total)
            pr, ta, doy, leap = dom.chunk(t0, t1)
            tn, tx = dom.chunk_tmm(t0, t1)   # real per-cell tmin/tmax (required by
                                             # the PT PET; None for a Hamon domain)
            flow, state = run_window(pr, ta, doy, leap, dom.lat_rad, dom.elev,
                                     params, uh, state, n_inc=cfg.n_inc,
                                     perc_mode=cfg.perc_mode,
                                     fracp_floor=cfg.fracp_floor,
                                     ninc_mode=cfg.ninc_mode, sac_pet=cfg.sac_pet,
                                     tmin=tn, tmax=tx,
                                     pt_snow_albedo=cfg.pt_snow_albedo,
                                     pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                                     state_idx=dom.chunk_state(t0, t1))
            basin_flow[:, t0:t1] = dom.W @ flow
            if progress:
                print(f"    days {t1}/{t_total}  ({time.time() - tic:.0f}s)", flush=True)
            t0 = t1
    return basin_flow.double().cpu().numpy()


def _obs_kge(sim: pd.Series, obs: pd.Series, dates: pd.DatetimeIndex,
             period: str) -> float:
    mask = (dates <= pd.Timestamp(CAL_END)) if period == "cal" else \
        (dates > pd.Timestamp(CAL_END))
    m = mask & np.isfinite(obs.to_numpy()) & np.isfinite(sim.to_numpy())
    if m.sum() < 90:
        return float("nan")
    return kge(sim.to_numpy()[m], obs.to_numpy()[m])


def score_frozen(
    params: pd.DataFrame,
    data_dir: str = "data",
    out_dir: str | Path = "artifacts/dpl/static",
    *,
    label: str = "dpl_static",
    cal_end: str = CAL_END,
    domain: str = "15cdec",
    parallel: bool = True,
    pet_source: str = "hamon",
    pt_snow_albedo: float = 0.0,
    pt_dewpoint_depression: float = 0.0,
    et_scheme: str = "sac",
    canopy_params: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score a parameter table through the FROZEN model vs the observed gage.

    The dPL reporting path: full-record ``run_basin(..., params=...)`` per
    basin (frozen physics — the torch pipeline is never a source of reported
    skill), daily cal/val split at ``cal_end``, per-basin diagnostics + skill
    summary in the cdec15 figure conventions ->
    ``<out_dir>/metrics_<label>.csv`` + ``figures/``.  Same columns as
    ``metrics_15cdec.csv`` so the GA-vs-dPL comparison is a plain merge.

    ``pet_source="priestley_taylor"`` scores a PT-trained export through the
    numba PT PET (``sacsma.pet_pt``) with the given refinement knobs — the
    same fast frozen-numerics convention as the Hamon exports.

    ``et_scheme="noah_lite"`` scores a Noah-lite (``canopy_lite``) export through
    the numba Noah-lite external-ET SAC core (``sacsma.sma_noah_lite``): PT PET
    forced, the learned per-HRU ``soil_chi`` taken from ``canopy_params`` (a
    ``params_canopy.csv`` table).  Same frozen-numerics convention — the Noah-ET
    dPL then reports on the identical footing as the Hamon/PT runs."""
    from .._figures import (
        _period_stats,
        basin_diagnostics_fig,
        folsom_before_yuba,
        skill_summary_fig,
    )
    from ..io import load_basin_area, load_hru_table, mmday_to_cfs
    from ..model import load_domain_forcing

    out = Path(out_dir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    cal_end_ts = pd.Timestamp(cal_end)

    ref = "15cdec" if domain.endswith("_grid") else domain  # basins shared w/ 15cdec
    hru = load_hru_table(data_dir, domain=domain)
    basins = folsom_before_yuba(
        ref,
        hru.groupby("basin")["lat"].mean().sort_values(ascending=False).index.tolist())
    forcing = load_domain_forcing(data_dir, domain=domain)
    try:
        areas = load_basin_area(data_dir, domain=ref).set_index(
            "basin")["area_mi2"].to_dict()
    except FileNotFoundError:
        areas = {}

    records = []
    for b in basins:
        sim = run_basin(b, data_dir=data_dir, domain=domain, forcing=forcing,
                        params=params, parallel=parallel,
                        pet_source=pet_source,
                        pt_snow_albedo=pt_snow_albedo,
                        pt_dewpoint_depression=pt_dewpoint_depression,
                        et_scheme=et_scheme, canopy_params=canopy_params).rename(
                            columns={"flow": "flow_sim"})
        obs = load_gage(data_dir, basin=b)[["date", "flow"]].rename(
            columns={"flow": "flow_obs"})
        m = pd.merge(sim, obs, on="date", how="left").sort_values(
            "date").reset_index(drop=True)
        is_cal = m["date"] <= cal_end_ts
        cal = _period_stats(m.loc[is_cal, "flow_sim"].to_numpy(),
                            m.loc[is_cal, "flow_obs"].to_numpy())
        val = _period_stats(m.loc[~is_cal, "flow_sim"].to_numpy(),
                            m.loc[~is_cal, "flow_obs"].to_numpy())
        obs_dates = m.loc[m["flow_obs"].notna(), "date"]
        mplot = (m[m["date"] >= obs_dates.min()].reset_index(drop=True)
                 if not obs_dates.empty else m)
        basin_diagnostics_fig(b, mplot, cal_end_ts, cal, val,
                              figdir / f"{b}_diagnostics.png")
        area = areas.get(b, np.nan)
        records.append({
            "basin": b, "area_mi2": area,
            "cal_kge": cal.get("kge"), "cal_nse": cal.get("nse"),
            "cal_pbias": cal.get("pbias"), "cal_r": cal.get("r"),
            "cal_n": cal.get("n", 0),
            "val_kge": val.get("kge"), "val_nse": val.get("nse"),
            "val_pbias": val.get("pbias"), "val_r": val.get("r"),
            "val_n": val.get("n", 0),
            "obs_mean_mmday": cal.get("obs_mean"),
            "obs_mean_cfs": mmday_to_cfs(cal.get("obs_mean") or np.nan, area),
        })
        print(f"  {b}: CAL KGE={cal.get('kge', float('nan')):.3f} "
              f"VAL KGE={val.get('kge', float('nan')):.3f}", flush=True)

    metrics = pd.DataFrame(records)
    skill_summary_fig(metrics, figdir / "skill_summary.png")
    csv = out / f"metrics_{label}.csv"
    metrics.round(4).to_csv(csv, index=False)
    print(f"wrote {csv}", flush=True)
    return metrics


def _pipeline_storage(st) -> torch.Tensor:
    """Total liquid-equivalent water (N,) held in a PipelineState — for the
    Noah mass-balance closure.  Snow SWE + the 5 SAC stores + canopy wc +
    in-transit routing history.  adimc is EXCLUDED (it overlaps uztwc/uzfwc)."""
    s = (st.snow.w_i + st.snow.w_q
         + st.sac.uztwc + st.sac.uzfwc + st.sac.lztwc + st.sac.lzfsc + st.sac.lzfpc
         + st.hist_surf.sum(-1) + st.hist_base.sum(-1))
    if st.canopy is not None:
        s = s + st.canopy.wc
    return s


def _noah_stream(net: torch.nn.Module, x: torch.Tensor, dom: DomainTensors,
                 cfg: DplConfig, *, temp_delta: float | np.ndarray = 0.0,
                 precip_scale: float = 1.0, chunk_days: int = 4096) -> dict:
    """Stream the full record through the torch Noah pipeline -> the basin daily
    flow (``sim``, (B, T) ndarray mm/day) + per-HRU ET sums + water-balance
    closure.  ``temp_delta`` adds to tavg/tmin/tmax: a scalar degC (warming
    projection), or a per-forcing-row ``(rows, T)`` array (e.g. the WGEN
    detrending field) gathered to HRU rows via ``dom.cell_idx``.  ``precip_scale``
    MULTIPLIES precip (e.g. 1.1 for +10%): the precipitation-perturbation knob
    parallel to ``temp_delta``, applied before the model AND the closure sum so
    the water balance stays consistent under the counterfactual."""
    net.eval()
    with torch.no_grad():
        o = net(x)
    params = {k: v for k, v in o.items() if k != "_canopy"}
    cp = o.get("_canopy")
    uh = routing_uh(params, dom.flowlen)
    n, tt = dom.n_hru, dom.n_time
    st0 = initial_state(n, dom.device, dom.dtype, init_mode=cfg.init_mode,
                        params=params, et_mode="noah")
    state = st0
    basin = torch.empty(len(dom.basins), tt, device=dom.device, dtype=dom.dtype)
    sum_pr = torch.zeros(n, device=dom.device, dtype=dom.dtype)
    sum_fl = torch.zeros(n, device=dom.device, dtype=dom.dtype)
    sum_et = torch.zeros(n, device=dom.device, dtype=dom.dtype)
    dt_field = None
    if not isinstance(temp_delta, (int, float)):
        dt_field = torch.as_tensor(np.ascontiguousarray(
            np.asarray(temp_delta)[dom.cell_idx])).to(dom.device, dom.dtype)
    with torch.no_grad():
        t0 = 0
        while t0 < tt:
            t1 = min(t0 + chunk_days, tt)
            pr, ta, doy, leap = dom.chunk(t0, t1)
            tn, tx = dom.chunk_tmm(t0, t1)
            if precip_scale != 1.0:             # multiplicative precip perturbation
                pr = pr * precip_scale
            if dt_field is not None:            # per-cell temperature field
                d = dt_field[:, t0:t1]
                ta = ta + d
                if tn is not None:
                    tn, tx = tn + d, tx + d
            elif temp_delta:                    # scalar warming perturbation
                ta = ta + temp_delta
                if tn is not None:
                    tn, tx = tn + temp_delta, tx + temp_delta
            flow, state, tet = run_window(
                pr, ta, doy, leap, dom.lat_rad, dom.elev, params, uh, state,
                n_inc=cfg.n_inc, perc_mode=cfg.perc_mode,
                fracp_floor=cfg.fracp_floor, ninc_mode=cfg.ninc_mode,
                et_mode="noah", canopy_params=cp, tmin=tn, tmax=tx,
                veg_frac=dom.veg_frac, lai=dom.chunk_lai(t0, t1),
                noah_pet=cfg.noah_pet, canopy_lite=cfg.canopy_lite,
                pt_snow_albedo=cfg.pt_snow_albedo,
                pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                state_idx=dom.chunk_state(t0, t1),
                return_tet=True)
            basin[:, t0:t1] = dom.W @ flow
            sum_pr += pr.sum(1)
            sum_fl += flow.sum(1)
            sum_et += tet.sum(1)
            t0 = t1

    # per-HRU water-balance closure: Σprcp ≈ Σflow + Σtet + ΔS (the routing-tail
    # residual is the only expected slack over the full record)
    dS = _pipeline_storage(state) - _pipeline_storage(st0)
    resid = (sum_pr - sum_fl - sum_et - dS).abs()
    closure_rel = float((resid / sum_pr.clamp_min(1e-6)).max())
    return dict(sim=basin.double().cpu().numpy(), sum_et=sum_et,
                closure_rel=closure_rel)


def score_noah_torch(net: torch.nn.Module, x: torch.Tensor, dom: DomainTensors,
                     cfg: DplConfig, *, data_dir: str = "data",
                     out_dir: str | Path = "artifacts/dpl/noah_grid",
                     label: str = "dpl_noah", temp_delta: float = 0.0,
                     chunk_days: int = 4096, cal_end: str = CAL_END) -> pd.DataFrame:
    """Score a Noah-ET net through the TORCH pipeline (Noah is NEW physics — NOT
    scorable via ``run_basin``).  Streams the full record with ``et_mode='noah'``
    + per-cell tmin/tmax, aggregates ``dom.W @ flow`` to the outlet, and scores
    cal/val KGE vs the gage (same columns as ``score_frozen``).  Also reports
    the per-basin ET partition and a per-HRU water-balance closure.  ``temp_delta``
    adds ΔT to tavg/tmin/tmax (a one-knob warming-projection run)."""
    from .._figures import _period_stats, folsom_before_yuba, skill_summary_fig
    from ..io import load_basin_area, load_hru_table, mmday_to_cfs

    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    res = _noah_stream(net, x, dom, cfg, temp_delta=temp_delta,
                       chunk_days=chunk_days)
    sim = res["sim"]
    closure_rel = res["closure_rel"]
    sum_et = res["sum_et"]
    years = dom.n_time / 365.25

    hru = load_hru_table(data_dir, domain="15cdec")
    basins = folsom_before_yuba(
        "15cdec", hru.groupby("basin")["lat"].mean().sort_values(
            ascending=False).index.tolist())
    b_index = {b: i for i, b in enumerate(dom.basins)}
    try:
        areas = load_basin_area(data_dir, domain="15cdec").set_index(
            "basin")["area_mi2"].to_dict()
    except FileNotFoundError:
        areas = {}
    et_mmyr = dom.W.cpu().numpy() @ (sum_et / years).cpu().numpy()   # basin ET
    is_cal = dom.dates <= pd.Timestamp(cal_end)

    rows = []
    for b in basins:
        i = b_index[b]
        obs = load_gage(data_dir, basin=b).set_index("date")["flow"].reindex(
            dom.dates).to_numpy()
        cal = _period_stats(sim[i][is_cal], obs[is_cal])
        val = _period_stats(sim[i][~is_cal], obs[~is_cal])
        area = areas.get(b, np.nan)
        rows.append({
            "basin": b, "area_mi2": area,
            "cal_kge": cal.get("kge"), "cal_nse": cal.get("nse"),
            "cal_pbias": cal.get("pbias"), "cal_r": cal.get("r"),
            "cal_n": cal.get("n", 0),
            "val_kge": val.get("kge"), "val_nse": val.get("nse"),
            "val_pbias": val.get("pbias"), "val_r": val.get("r"),
            "val_n": val.get("n", 0),
            "obs_mean_mmday": cal.get("obs_mean"),
            "obs_mean_cfs": mmday_to_cfs(cal.get("obs_mean") or np.nan, area),
            "noah_et_mmyr": float(et_mmyr[i]),
        })
        print(f"  {b}: CAL KGE={cal.get('kge', float('nan')):.3f} "
              f"VAL KGE={val.get('kge', float('nan')):.3f}  "
              f"ET={et_mmyr[i]:.0f}mm/yr", flush=True)

    metrics = pd.DataFrame(rows)
    if not temp_delta:      # a perturbed run must not clobber the run's figure
        skill_summary_fig(metrics, out / "figures" / "skill_summary.png")
    lab = label if not temp_delta else f"{label}_plus{temp_delta:g}C"
    csv = out / f"metrics_{lab}.csv"
    metrics.round(4).to_csv(csv, index=False)
    # daily sim (date x basin, mm/day) — the torch reporting path's only route to
    # a daily series (run_basin can't reconstruct a seasonal/Noah net); consumed
    # by the seasonal-vs-LSTM comparison (sacsma.dpl.seasonal_compare).
    daily_csv = out / f"daily_sim_{lab}.csv"
    pd.DataFrame(sim.T, index=dom.dates, columns=list(dom.basins)).rename_axis(
        "date").to_csv(daily_csv)
    print(f"wrote {daily_csv} (daily sim mm/day)", flush=True)
    scc_isb = metrics.set_index("basin")["val_kge"].reindex(
        ["SCC", "ISB"]).round(3).tolist()
    print(f"wrote {csv}  (mean cal {metrics['cal_kge'].mean():.3f} / "
          f"val {metrics['val_kge'].mean():.3f}; SCC/ISB val {scc_isb})", flush=True)
    print(f"[closure] max per-HRU |sum(prcp-flow-tet)-dS|/sum(prcp) = {closure_rel:.2e}",
          flush=True)
    return metrics


def score_sac_torch(net: torch.nn.Module, x: torch.Tensor, dom: DomainTensors,
                    cfg: DplConfig, *, data_dir: str = "data",
                    out_dir: str | Path = "artifacts/dpl/dynamic",
                    label: str = "dpl_dynamic", cal_end: str = CAL_END) -> pd.DataFrame:
    """Score a Hamon (et_mode='sac') net through the TORCH pipeline — needed when
    the parameter field is TIME-VARYING (dynamic params), which the frozen
    run_basin cannot reconstruct.  Same per-basin columns as score_frozen."""
    from .._figures import _period_stats, folsom_before_yuba, skill_summary_fig
    from ..io import load_basin_area, load_hru_table, mmday_to_cfs

    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    net.eval()
    with torch.no_grad():
        params = {k: v for k, v in net(x).items() if k != "_canopy"}
    sim = torch_domain_flow(dom, params, cfg, progress=False)     # (B, T)

    hru = load_hru_table(data_dir, domain="15cdec")
    basins = folsom_before_yuba(
        "15cdec", hru.groupby("basin")["lat"].mean().sort_values(
            ascending=False).index.tolist())
    b_index = {b: i for i, b in enumerate(dom.basins)}
    try:
        areas = load_basin_area(data_dir, domain="15cdec").set_index(
            "basin")["area_mi2"].to_dict()
    except FileNotFoundError:
        areas = {}
    is_cal = dom.dates <= pd.Timestamp(cal_end)
    rows = []
    for b in basins:
        i = b_index[b]
        obs = load_gage(data_dir, basin=b).set_index("date")["flow"].reindex(
            dom.dates).to_numpy()
        cal = _period_stats(sim[i][is_cal], obs[is_cal])
        val = _period_stats(sim[i][~is_cal], obs[~is_cal])
        area = areas.get(b, np.nan)
        rows.append({
            "basin": b, "area_mi2": area,
            "cal_kge": cal.get("kge"), "cal_nse": cal.get("nse"),
            "cal_pbias": cal.get("pbias"), "cal_r": cal.get("r"),
            "cal_n": cal.get("n", 0),
            "val_kge": val.get("kge"), "val_nse": val.get("nse"),
            "val_pbias": val.get("pbias"), "val_r": val.get("r"),
            "val_n": val.get("n", 0),
            "obs_mean_mmday": cal.get("obs_mean"),
            "obs_mean_cfs": mmday_to_cfs(cal.get("obs_mean") or np.nan, area),
        })
        print(f"  {b}: CAL KGE={cal.get('kge', float('nan')):.3f} "
              f"VAL KGE={val.get('kge', float('nan')):.3f}", flush=True)
    metrics = pd.DataFrame(rows)
    skill_summary_fig(metrics, out / "figures" / "skill_summary.png")
    csv = out / f"metrics_{label}.csv"
    metrics.round(4).to_csv(csv, index=False)
    print(f"wrote {csv}  (mean cal {metrics['cal_kge'].mean():.3f} / "
          f"val {metrics['val_kge'].mean():.3f})", flush=True)
    return metrics


def load_net_from_checkpoint(
    ckpt_path: str | Path,
    data_dir: str = "data",
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.nn.Module, torch.Tensor, DomainTensors, DplConfig, dict]:
    """Rebuild ``(net, x, dom, cfg, ck)`` from a training checkpoint — the
    shared front half of :func:`evaluate_checkpoint` (feature rebuild +
    ParameterNet restore).  ``device=None`` -> cuda if available, else cpu."""
    import dataclasses as _dc

    import numpy as _np

    from ..io import soilveg_path
    from .features import FeatureSet, build_features
    from .parameter_net import ParameterNet

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    variant = ck["variant"]
    domain = ck.get("domain", "15cdec")
    nc = ck.get("net_config", {})
    # rebuild the training config tolerantly: checkpoints persist the cfg dict
    # verbatim, so fields REMOVED in later schema versions (e.g. the retired
    # et_loss_sigma_floor) must be dropped, not crash the scoring.
    known = {f.name for f in _dc.fields(DplConfig)}
    dropped = sorted(set(ck["cfg"]) - known)
    if dropped:
        print(f"note: dropping retired cfg keys from checkpoint: {dropped}",
              flush=True)
    cfg = DplConfig(**{k: v for k, v in ck["cfg"].items() if k in known})
    dyn = tuple(nc.get("dynamic_params", ()))
    if device is None:
        try:
            dev = pick_device("cuda")
        except RuntimeError:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)
    dom = load_domain_tensors(data_dir, domain=domain, device=dev,
                              dtype=torch.float64,
                              dynamic_window=(nc.get("dynamic_window", 365)
                                              if dyn else None),
                              calsim_footprint=ck.get("cfg", {}).get(
                                  "calsim_footprint", False))
    stats = FeatureSet(x=_np.empty((0, 0), dtype=_np.float32), **ck["features"])
    fs = build_features(dom.hrus, variant=variant,
                        forcing=(dom.forcing if variant in ("climate",
                                 "physical_climate") else None),
                        climate_window=stats.climate_window,
                        climate_product=stats.climate_product,
                        physical_path=(soilveg_path(data_dir, domain)
                                       if variant in ("physical",
                                       "physical_climate") else None),
                        stats=stats)
    x = torch.as_tensor(fs.x).to(dev, torch.float64)
    gnn_k = nc.get("gnn_k", 0)
    net = ParameterNet(x.shape[1], hidden=nc.get("hidden", 64),
                       embed=nc.get("embed", 32),
                       dropout=nc.get("dropout", 0.1),
                       grouped_heads=nc.get("grouped_heads", False),
                       gnn_k=gnn_k,
                       n_nodes=x.shape[0] if gnn_k > 0 else None,
                       seasonal_params=tuple(nc.get("seasonal_params", ())),
                       seasonal_amp=nc.get("seasonal_amp", 0.18),
                       seasonal_amp_frac=nc.get("seasonal_amp_frac", 0.10),
                       canopy=nc.get("canopy", False),
                       canopy_separate_trunk=nc.get("canopy_separate_trunk", True),
                       canopy_lite=nc.get("canopy_lite", False),
                       dynamic_params=dyn,
                       dynamic_amp=nc.get("dynamic_amp", 0.5),
                       ).to(dev, torch.float64)
    net.load_state_dict(ck["net"])   # restores baked neighbor buffers too
    return net, x, dom, cfg, ck


def noah_torch_daily(ckpt_path: str | Path, *, data_dir: str = "data",
                     temp_delta: float | np.ndarray = 0.0,
                     precip_scale: float = 1.0,
                     chunk_days: int = 4096,
                     device: torch.device | str | None = None) -> pd.DataFrame:
    """Daily basin flow (date x basin, mm/day) from a torch-scored Noah
    checkpoint, optionally under a climate perturbation: ``temp_delta`` adds to
    tavg/tmin/tmax (a scalar degC, or a per-forcing-row ``(rows, T)`` field e.g.
    the WGEN detrending delta) and ``precip_scale`` multiplies precip (1.1 =
    +10%).  Pure compute — writes nothing; the counterfactual runner for
    torch-side (dp, dt) response surfaces and the forcing-sensitivity detrended
    noah sim channel."""
    net, x, dom, cfg, ck = load_net_from_checkpoint(ckpt_path, data_dir,
                                                    device=device)
    if not ck.get("net_config", {}).get("canopy", False):
        raise ValueError("noah_torch_daily needs a Noah (canopy) checkpoint")
    res = _noah_stream(net, x, dom, cfg, temp_delta=temp_delta,
                       precip_scale=precip_scale, chunk_days=chunk_days)
    df = pd.DataFrame(res["sim"].T, index=dom.dates, columns=list(dom.basins))
    df.index.name = "date"
    return df


#: cache root for the (Δprecip, ΔT) noah physics teachers / response-surface runs.
_DTDP_CACHE = "artifacts/dpl/testing/dtdp_cache"


def teacher_cache_path(dp: float, dt: float,
                       cache_dir: str | Path = _DTDP_CACHE) -> Path:
    """Cache path for the noah daily sim under (Δprecip fraction dp, ΔT dt)."""
    return Path(cache_dir) / f"noah_dp{dp:+.2f}_dt{dt:+.1f}.csv"


def noah_teacher_daily(dp: float, dt: float, ckpt_path: str | Path, *,
                       data_dir: str = "data", cache_dir: str | Path = _DTDP_CACHE,
                       device: torch.device | str | None = None) -> pd.DataFrame:
    """Cached noah daily sim (date x basin, mm/day) under (Δprecip fraction ``dp``,
    ΔT ``dt`` degC).  The SINGLE source of truth for the dt/dp response teachers,
    the physics response column, and the hybrids' perturbed sim channel — so the
    numerics match on the training and evaluation sides.  ``dp=dt=0`` reproduces
    the noah checkpoint's full torch daily sim (the physics parity anchor)."""
    cache = teacher_cache_path(dp, dt, cache_dir)
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")
    df = noah_torch_daily(ckpt_path, data_dir=data_dir, temp_delta=float(dt),
                          precip_scale=1.0 + float(dp), device=device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def corner_anchors(dp: float, dt: float) -> list[tuple[float, float]]:
    """The 5 non-origin corners of {−dp, 0, +dp} × {0, +dt}:
    (−dp,0), (+dp,0), (0,dt), (−dp,dt), (+dp,dt)."""
    return [(-dp, 0.0), (dp, 0.0), (0.0, dt), (-dp, dt), (dp, dt)]


def grid_anchors(dps, dts) -> list[tuple[float, float]]:
    """The full (Δprecip, ΔT) grid ``dps × dts`` minus the (0, 0) origin — a wider
    / interior anchor set for supervising the response over more of the plane
    (not just the corners)."""
    return [(float(dp), float(dt)) for dp in dps for dt in dts
            if not (float(dp) == 0.0 and float(dt) == 0.0)]


def evaluate_checkpoint(
    ckpt_path: str | Path,
    data_dir: str = "data",
    out_dir: str | Path | None = None,
    *,
    parallel: bool = True,
    label: str | None = None,
    temp_delta: float = 0.0,
) -> pd.DataFrame:
    """best.pt -> params_dpl.csv -> frozen-model metrics (the full Phase-4 path).

    ``temp_delta`` != 0 re-runs the TORCH scorer with tavg/tmin/tmax + delta and
    dumps the perturbed daily sim (label suffixed ``_plus<delta>C``) — the
    teacher for the hybrid temperature-consistency loss.  Torch-scored
    checkpoints only (seasonal/canopy/dynamic)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    variant = ck["variant"]
    domain = ck.get("domain", "15cdec")
    nc = ck.get("net_config", {})
    canopy = nc.get("canopy", False)
    dyn = tuple(nc.get("dynamic_params", ()))
    # canopy (Noah ET) and dynamic (time-varying) params are torch-only; a
    # STATIC checkpoint — Hamon OR Priestley-Taylor — scores through the fast
    # numba run_basin (PT via sacsma.pet_pt, verified vs the torch pipeline).
    torch_score = canopy or bool(dyn)
    if temp_delta and not canopy:
        raise ValueError("--temp-delta teacher dumps run through the torch Noah "
                         "scorer only (canopy checkpoints)")
    if out_dir is not None:
        out = Path(out_dir)
    else:
        # default NEXT TO the checkpoint (<run>/checkpoints/x.pt -> <run>/).
        # The old artifacts/dpl/<variant> fallback silently CLOBBERED whatever
        # unrelated run shared the variant name (it overwrote the hamon
        # `physical` arm's outputs on 2026-07-15); it remains only for a bare
        # checkpoint outside the standard run layout.
        ckp = Path(ckpt_path).resolve()
        out = (ckp.parent.parent if ckp.parent.name == "checkpoints"
               else Path(f"artifacts/dpl/{variant}"))
        if label is None and ckp.parent.name == "checkpoints":
            label = out.name       # metrics_<run>.csv, not metrics_dpl_<variant>
    out.mkdir(parents=True, exist_ok=True)

    if torch_score:
        try:
            dev = pick_device("cuda")
        except RuntimeError:
            dev = torch.device("cpu")
    else:
        dev = torch.device("cpu")
    net, x, dom, cfg, ck = load_net_from_checkpoint(ckpt_path, data_dir,
                                                    device=dev)

    if canopy:   # Noah ET — a SEPARATE canopy-param table (kept OUT of ga_optimum)
        ccsv = out / "params_canopy.csv"
        canopy_df = export_canopy_params(net, dom, x)
        canopy_df.to_csv(ccsv, index=False)
        print(f"wrote {ccsv} (Noah canopy params; kept OUT of ga_optimum)",
              flush=True)
        # The learned SAC params still export (canopy runs used to skip this,
        # leaving noah with no params_dpl.csv) — the Noah-lite frozen
        # path and the hybrid physics baseline both need them.
        dpl_df = export_params(net, dom, x)
        pcsv = out / "params_dpl.csv"
        dpl_df.to_csv(pcsv, index=False)
        print(f"wrote {pcsv} ({len(dpl_df)} HRU rows, cal KGE at selection "
              f"{ck.get('cal_kge', float('nan')):.4f})", flush=True)
        lite = nc.get("canopy_lite", False)
        if (lite and cfg.noah_pet == "priestley_taylor"
                and not cfg.seasonal_params and not temp_delta):
            # canonical Noah-lite: score through the fast frozen numba core
            # (sma_noah_lite), verified bit-exact vs the torch pipeline — the
            # SAME frozen-numerics footing as the Hamon/PT exports.  A SEASONAL
            # Noah-lite export cannot use this path — the frozen run_basin raises
            # on day-of-year params (model.py:run_hru_components_noah_lite) and
            # the exported params_dpl.csv carries {MFMAX,...}_asin columns the
            # frozen path cannot read (never feed it to build_frozen_sim) — so it
            # falls through to the torch pipeline below.
            print("canopy_lite + PT -> frozen Noah-lite scoring "
                  "(sacsma.sma_noah_lite)", flush=True)
            pt_alb = cfg.pt_snow_albedo or 0.0
            pt_dd = cfg.pt_dewpoint_depression or 0.0
            return score_frozen(
                dpl_df, data_dir, out,
                label=label if label is not None else f"dpl_{variant}",
                domain=domain, parallel=parallel, pet_source="priestley_taylor",
                pt_snow_albedo=pt_alb, pt_dewpoint_depression=pt_dd,
                et_scheme="noah_lite", canopy_params=canopy_df)
        # full Noah ET (7-param Jarvis) or Hamon-potential lite: NO frozen core
        # -> the torch pipeline reports skill (mass-balance validated).
        return score_noah_torch(net, x, dom, cfg, data_dir=data_dir, out_dir=out,
                                temp_delta=temp_delta,
                                label=label if label is not None
                                else f"dpl_{variant}_noah")

    dpl_df = export_params(net, dom, x)
    pcsv = out / "params_dpl.csv"
    dpl_df.to_csv(pcsv, index=False)
    print(f"wrote {pcsv} ({len(dpl_df)} HRU rows, cal KGE at selection "
          f"{ck.get('cal_kge', float('nan')):.4f})", flush=True)
    if dyn:   # time-varying params — the frozen run_basin can't reconstruct them
        print(f"dynamic params {dyn} -> torch scoring (run_basin can't reconstruct)",
              flush=True)
        return score_sac_torch(net, x, dom, cfg, data_dir=data_dir, out_dir=out,
                               label=label if label is not None else f"dpl_{variant}")
    if cfg.sac_pet != "hamon":
        print(f"sac_pet={cfg.sac_pet} -> frozen scoring via the numba PT PET "
              f"(snow_albedo={cfg.pt_snow_albedo}, "
              f"dewpoint_depression={cfg.pt_dewpoint_depression})", flush=True)
    return score_frozen(dpl_df, data_dir, out,
                        label=label if label is not None else f"dpl_{variant}",
                        domain=domain, parallel=parallel,
                        pet_source=cfg.sac_pet,
                        pt_snow_albedo=cfg.pt_snow_albedo,
                        pt_dewpoint_depression=cfg.pt_dewpoint_depression)


def fidelity_benchmark(
    data_dir: str = "data",
    out_dir: str = "artifacts/dpl/noah/fidelity",
    *,
    configs: tuple[str, ...] | None = None,
    device: str = "cuda",
    chunk_days: int = 4096,
) -> pd.DataFrame:
    """Run the sweep; writes ``fidelity_benchmark.csv`` + a summary figure."""
    dev = pick_device(device)
    names = tuple(configs if configs is not None else FIDELITY_CONFIGS)
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    print("fidelity: frozen reference (run_basin, parallel) ...", flush=True)
    truth = frozen_truth(data_dir)
    dates = pd.DatetimeIndex(truth[BASINS[0]]["date"])
    gage = load_gage(data_dir)
    obs = {b: g.set_index("date")["flow"].reindex(dates)
           for b, g in gage.groupby("basin")}

    rows = []
    doms: dict[torch.dtype, DomainTensors] = {}
    for name in names:
        cfg = FIDELITY_CONFIGS[name]
        dtype = _DTYPES[cfg.dtype]
        if dtype not in doms:
            doms[dtype] = load_domain_tensors(data_dir, device=dev, dtype=dtype)
        dom = doms[dtype]
        ga = dom.ga_params(data_dir)
        print(f"fidelity: config {name} (n_inc={cfg.n_inc}, perc={cfg.perc_mode}, "
              f"{cfg.dtype}) ...", flush=True)
        tic = time.time()
        sim = torch_domain_flow(dom, ga, cfg, chunk_days=chunk_days)
        wall = time.time() - tic
        for b_i, b in enumerate(dom.basins):
            ref = truth[b]["flow"].to_numpy()
            s = sim[b_i]
            s_ser = pd.Series(s, index=dates)
            r_ser = pd.Series(ref, index=dates)
            ob = obs.get(b)
            rows.append({
                "config": name, "basin": b,
                "n_inc": cfg.n_inc, "perc_mode": cfg.perc_mode, "dtype": cfg.dtype,
                "kge_sim": kge(s, ref), "nse_sim": nse(s, ref),
                "pbias_sim": pbias(s, ref),
                "max_abs_diff": float(np.max(np.abs(s - ref))),
                "cal_kge_torch": _obs_kge(s_ser, ob, dates, "cal") if ob is not None else np.nan,
                "val_kge_torch": _obs_kge(s_ser, ob, dates, "val") if ob is not None else np.nan,
                "cal_kge_frozen": _obs_kge(r_ser, ob, dates, "cal") if ob is not None else np.nan,
                "val_kge_frozen": _obs_kge(r_ser, ob, dates, "val") if ob is not None else np.nan,
                "wall_s": round(wall, 1),
            })
        df_c = pd.DataFrame([r for r in rows if r["config"] == name])
        print(f"  {name}: sim-vs-sim KGE min {df_c['kge_sim'].min():.6f} | "
              f"max|d| {df_c['max_abs_diff'].max():.4f} mm/day | {wall:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df["d_cal_kge"] = df["cal_kge_torch"] - df["cal_kge_frozen"]
    df["d_val_kge"] = df["val_kge_torch"] - df["val_kge_frozen"]
    df.to_csv(out / "fidelity_benchmark.csv", index=False)
    _fidelity_figure(df, out / "figures" / "fidelity_benchmark.png")
    _print_gates(df)
    return df


def _print_gates(df: pd.DataFrame) -> None:
    ref = df[df["config"].str.startswith("ref-ninc")]
    if len(ref):
        best = ref.groupby("config")["kge_sim"].min().idxmax()
        b = ref[ref["config"] == best]
        g1 = (b["kge_sim"].min() >= 0.999
              and b["d_cal_kge"].abs().max() <= 0.005
              and b["d_val_kge"].abs().max() <= 0.005)
        print(f"G1 structural [{best}]: min KGE {b['kge_sim'].min():.6f}, "
              f"max|dKGE| {max(b['d_cal_kge'].abs().max(), b['d_val_kge'].abs().max()):.4f}"
              f" -> {'PASS' if g1 else 'FAIL'}")
    td = df[df["config"] == "train-default"]
    if len(td):
        g2 = (td["kge_sim"].min() >= 0.99
              and td["d_cal_kge"].abs().max() <= 0.02
              and td["d_val_kge"].abs().max() <= 0.02)
        print(f"G2 train-default: min KGE {td['kge_sim'].min():.6f} "
              f"-> {'PASS' if g2 else 'FAIL'}")
    f32 = df[df["config"] == "train-default-f32"]
    if len(td) and len(f32):
        m = td.merge(f32, on="basin", suffixes=("_64", "_32"))
        # float32-vs-float64 agreement, both scored against the same frozen truth
        g3 = (m["kge_sim_32"] - m["kge_sim_64"]).abs().max() <= 1e-3
        print(f"G3 precision: max |KGE_f32 - KGE_f64| "
              f"{(m['kge_sim_32'] - m['kge_sim_64']).abs().max():.5f} "
              f"-> {'PASS' if g3 else 'CHECK'}")


def _fidelity_figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = list(dict.fromkeys(df["config"]))
    basins = [b for b in BASINS if b in set(df["basin"])]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.2), dpi=300, sharex=True)
    x = np.arange(len(basins))
    width = 0.8 / len(configs)
    for c_i, c in enumerate(configs):
        d = df[df["config"] == c].set_index("basin").reindex(basins)
        axes[0].bar(x + c_i * width, d["kge_sim"], width, label=c)
        axes[1].bar(x + c_i * width, d["max_abs_diff"], width, label=c)
    axes[0].set_ylabel("sim-vs-sim daily KGE", fontsize=8)
    axes[0].set_ylim(0.95, 1.001)
    axes[0].axhline(0.999, color="0.4", lw=0.6, ls="--")
    axes[1].set_ylabel("max |torch − frozen| (mm/day)", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x + 0.4 - width / 2)
    axes[1].set_xticklabels(basins, fontsize=7, rotation=45)
    axes[0].legend(fontsize=6, ncol=2, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle("dPL torch forward vs frozen reference — archived GA parameters",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
