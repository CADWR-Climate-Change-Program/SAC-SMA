"""A trained run: rebuilt from its checkpoint, run as trained, scored.

:func:`load_net_from_checkpoint` rebuilds the network, its inputs and the domain;
:func:`learned_field` and :func:`simulate_field` run the trained field on the CPU engine with
the physics it was trained with (:mod:`sacsma.engine`); :func:`evaluate_checkpoint` scores a
15-CDEC run against the gauges (a multifamily run: :mod:`.evaluate_multi_timescale`);
:func:`basin_daily` is its daily flow under a changed climate, for the hybrids and the studies.
:func:`fidelity_benchmark` runs the GA optimum through the learned numerics against the reference
model.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import paths
from ..cdec15 import BASINS, CAL_END, load_gage
from ..io import DEFAULT_FORCING
from ..metrics import kge, nse, pbias
from ..model import run_basins
from .config import PARAM_ORDER, DplConfig, config_from_checkpoint, pick_device
from .data import DomainTensors, load_domain_tensors


def export_params(net: torch.nn.Module, dom: DomainTensors,
                  x: torch.Tensor) -> pd.DataFrame:
    """Learned per-HRU parameters as a ga_optimum-shaped table (+ ``basin``): the 34
    ``ga_optimum.csv`` columns keyed by grid-cell ``key``, plus the per-basin ``basin`` column
    (the cells shared between basins carry per-basin values).  Fixed parameters
    (side/SCF/PXTEMP) come out at their GA constants -- PXTEMP at its learned per-cell value
    for a ``pxtemp_learn`` net."""
    net.eval()
    with torch.no_grad():
        out = net(x)
    df = dom.hrus[["basin", "key", "lat", "lon"]].copy()
    for p in PARAM_ORDER:
        df[p] = out[p].double().cpu().numpy()
    return df


def export_canopy_params(net: torch.nn.Module, dom: DomainTensors,
                         x: torch.Tensor) -> pd.DataFrame:
    """The learned Noah-lite exponent ``soil_chi`` per HRU keyed by key/basin, with the
    observed (pinned) veg_frac and annual-mean LAI beside it.  Kept apart from
    :func:`export_params`: the reference model has no Noah ET."""
    net.eval()
    with torch.no_grad():
        chi = net(x)["soil_chi"]
    df = dom.hrus[["basin", "key", "lat", "lon"]].copy()
    df["soil_chi"] = chi.double().cpu().numpy()
    if dom.veg_frac is not None:
        df["veg_frac_obs"] = dom.veg_frac.double().cpu().numpy()
    if dom.lai_lut is not None:
        df["lai_obs_mean"] = dom.lai_lut[dom.cell_idx].mean(axis=1)
    return df


def _obs_kge(sim: pd.Series, obs: pd.Series, dates: pd.DatetimeIndex,
             period: str) -> float:
    mask = (dates <= pd.Timestamp(CAL_END)) if period == "cal" else \
        (dates > pd.Timestamp(CAL_END))
    m = mask & np.isfinite(obs.to_numpy()) & np.isfinite(sim.to_numpy())
    if m.sum() < 90:
        return float("nan")
    return kge(sim.to_numpy()[m], obs.to_numpy()[m])


def daily_obs_mask(data_dir: str = "data") -> tuple[str, ...]:
    """The confirmed bad CDEC gauge days (``fnf_daily_mask.csv``) as ``"cdec_<B>|YYYY-MM-DD"``,
    the entries of ``DplConfig.obs_mask``."""
    from .cli import _obs_mask

    return _obs_mask(str(paths.cdec_fnf(data_dir, "fnf_daily_mask.csv")))


#: the 15-CDEC training water years (WY1989-2003) and test years (WY2004-18)
CAL_WINDOW = ("1988-10-01", CAL_END)
VAL_WINDOW = ("2003-10-01", "2018-09-30")


def score_basins(sim: np.ndarray, basins, dates: pd.DatetimeIndex, out: Path, *,
                 data_dir: str = "data", figures: bool = True,
                 cal: tuple[str, str] = CAL_WINDOW, val: tuple[str, str] = VAL_WINDOW,
                 obs_mask: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Score daily basin flow ``sim`` (B, T mm/day on ``basins`` x ``dates``; NaN days are
    skipped) against the gauges on the training water years ``cal`` and the test years ``val``,
    the days of ``obs_mask`` left out (``None``: :func:`daily_obs_mask`), north to south:
    ``metrics.csv`` in ``out`` (the columns of the calibrated model's), and with ``figures``
    the diagnostics under ``out/figures/``."""
    from .._figures import (
        _period_stats,
        basin_diagnostics_fig,
        folsom_before_yuba,
        skill_summary_fig,
    )
    from ..io import load_basin_area, load_hru_table, mmday_to_cfs

    cal_end_ts = pd.Timestamp(cal[1])
    masked = daily_obs_mask(data_dir) if obs_mask is None else obs_mask
    lat = load_hru_table(data_dir, domain="15cdec").groupby("basin")["lat"].mean()
    order = folsom_before_yuba("15cdec", lat.sort_values(ascending=False).index.tolist())
    areas = load_basin_area(data_dir, domain="15cdec").set_index("basin")["area_mi2"].to_dict()
    is_cal = np.asarray((dates >= pd.Timestamp(cal[0])) & (dates <= cal_end_ts))
    is_val = np.asarray((dates >= pd.Timestamp(val[0])) & (dates <= pd.Timestamp(val[1])))
    figdir = out / "figures"
    if figures:
        figdir.mkdir(parents=True, exist_ok=True)
    records = []
    for b in [b for b in order if b in basins]:
        s = sim[list(basins).index(b)]
        obs = load_gage(data_dir, basin=b).set_index("date")["flow"].reindex(dates)
        for m in masked:
            e, _, day = m.partition("|")
            if e.removeprefix("cdec_") == b and pd.Timestamp(day) in obs.index:
                obs[pd.Timestamp(day)] = np.nan
        cs = _period_stats(s[is_cal], obs.to_numpy()[is_cal])
        vs = _period_stats(s[is_val], obs.to_numpy()[is_val])
        if figures:
            m = pd.DataFrame({"date": dates, "flow_sim": s, "flow_obs": obs.to_numpy()})
            m = m[m["date"] >= obs.first_valid_index()].reset_index(drop=True)
            basin_diagnostics_fig(b, m, cal_end_ts, cs, vs, figdir / f"{b}_diagnostics.png")
        area = areas.get(b, np.nan)
        records.append({
            "basin": b, "area_mi2": area,
            "cal_kge": cs.get("kge"), "cal_nse": cs.get("nse"),
            "cal_pbias": cs.get("pbias"), "cal_r": cs.get("r"), "cal_n": cs.get("n", 0),
            "val_kge": vs.get("kge"), "val_nse": vs.get("nse"),
            "val_pbias": vs.get("pbias"), "val_r": vs.get("r"), "val_n": vs.get("n", 0),
            "obs_mean_mmday": cs.get("obs_mean"),
            "obs_mean_cfs": mmday_to_cfs(cs.get("obs_mean") or np.nan, area),
        })
        print(f"  {b}: CAL KGE={cs.get('kge', float('nan')):.3f} "
              f"VAL KGE={vs.get('kge', float('nan')):.3f}", flush=True)
    metrics = pd.DataFrame(records)
    if figures:
        skill_summary_fig(metrics, figdir / "skill_summary.png")
    metrics.round(4).to_csv(out / "metrics.csv", index=False)
    print(f"wrote {out / 'metrics.csv'}  (mean cal {metrics['cal_kge'].mean():.3f} / "
          f"val {metrics['val_kge'].mean():.3f})", flush=True)
    return metrics


def load_net_from_checkpoint(
    ckpt_path: str | Path,
    data_dir: str = "data",
    *,
    device: torch.device | str | None = None,
    product: str = DEFAULT_FORCING,
) -> tuple[torch.nn.Module, torch.Tensor, DomainTensors, DplConfig, dict]:
    """Rebuild ``(net, x, dom, cfg, ck)`` from a training checkpoint: the domain (the
    checkpoint's basins, on the forcing ``product``), the network inputs and the network.
    ``device=None`` -> cuda if available, else cpu."""
    from .parameter_net import ParameterNet

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ck)
    if device is None:
        try:
            dev = pick_device("cuda")
        except RuntimeError:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)
    dom = load_domain_tensors(data_dir, domain=ck["domain"], device=dev, dtype=torch.float64,
                              basins=tuple(ck["basins"]), product=product)
    x = checkpoint_features(ck, dom, data_dir)
    net = ParameterNet.from_checkpoint(ck, x.shape[1]).to(dev, torch.float64)
    return net, x, dom, cfg, ck


def checkpoint_features(ck: dict, dom: DomainTensors, data_dir: str = "data", *,
                        domain: str | None = None) -> torch.Tensor:
    """The network inputs of the checkpoint's variant on ``dom``'s rows, scaled with its
    training statistics: the ``physical`` variant its statics from ``domain``'s table
    (default: the checkpoint's), ``aef_u`` the AlphaEarth store."""
    from ..io import soilveg_path
    from .features import FeatureSet, aef_store, build_features

    variant = ck["variant"]
    stats = FeatureSet(x=np.empty((0, 0), dtype=np.float32), **ck["features"])
    fs = build_features(dom.hrus, variant=variant,
                        physical_path=(soilveg_path(data_dir, domain or ck["domain"])
                                       if variant == "physical" else None),
                        aef_path=aef_store(data_dir) if variant == "aef_u" else None,
                        stats=stats)
    return torch.as_tensor(fs.x).to(dom.device, torch.float64)


def learned_field(net: torch.nn.Module, x: torch.Tensor, dom: DomainTensors, cfg: DplConfig):
    """The trained field of ``net`` on ``dom``'s rows as a :class:`sacsma.engine.Field` (the
    network evaluated once, in eval mode, in float64) with the physics it was trained with."""
    from .. import parameters as P
    from ..engine import Field

    net.eval()
    with torch.no_grad():
        out = net(x)
    noah = cfg.et_mode == "noah"
    a = lambda t: t.detach().cpu().double().numpy()  # noqa: E731
    cols = lambda names: np.stack([a(out[k]) for k in names], axis=1)  # noqa: E731
    cell = np.asarray(dom.cell_idx, np.int64)
    return Field.from_rows(cfg.physics(), cell, a(dom.lat_rad), a(dom.elev), a(dom.flowlen),
                           a(out["Kpet"]), cols(P._SNOW_COLS), cols(P._SMA_COLS),
                           cols(P._ROUT_COLS), veg=a(dom.veg_frac) if noah else None,
                           chi=a(out["soil_chi"]) if noah else None,
                           lai=dom.lai_lut[cell] if noah else None)


def simulate_field(net: torch.nn.Module, x: torch.Tensor, dom: DomainTensors, cfg: DplConfig,
                   start, end, weights, *, spinup: str = "cycle", **kw) -> dict:
    """The trained field over ``[start, end]`` on the CPU engine (:func:`sacsma.engine.simulate`),
    the state at ``start`` from the run's spin-up settings; ``weights`` (K, rows)."""
    from ..engine import simulate
    from .spinup import window_start

    forcing = dataclasses.replace(dom.forcing, tmin=dom.tmin, tmax=dom.tmax)
    t0 = int(dom.dates.searchsorted(pd.Timestamp(start)))
    return simulate(learned_field(net, x, dom, cfg), forcing, start, end, weights, spinup=spinup,
                    years=cfg.spinup_years, passes=cfg.spinup_passes,
                    window_start=dom.dates[window_start(dom.dates, t0, cfg.spinup_start)], **kw)


#: the engine's code: a change to it re-runs a cached daily flow
ENGINE_CODE = tuple(Path(__file__).parents[1] / f for f in (
    "engine.py", "sma_learned.py", "sma.py", "snow17.py", "pet.py", "pet_pt.py", "routing.py"))


def content_key(*files) -> str:
    """The first 10 hex digits of the SHA-1 of the files' bytes, in order: a cache key that
    follows a retrained checkpoint."""
    import hashlib

    h = hashlib.sha1()
    for f in files:
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:10]


def ensemble_key(ens_dir: str | Path) -> str:
    """:func:`content_key` of an LSTM ensemble: its members' checkpoints and, for a physics input,
    the checkpoint of the physics run and the engine's code (:data:`ENGINE_CODE`)."""
    members = sorted(Path(ens_dir).glob("seed*/checkpoints/best.pt"))
    if not members:
        raise FileNotFoundError(f"no seed*/checkpoints/best.pt under {ens_dir}")
    physics = torch.load(members[0], map_location="cpu", weights_only=False)["cfg"].get("physics")
    run = [paths.dpl_checkpoint(run=physics), *ENGINE_CODE] if physics else []
    return content_key(*members, *run)


@functools.lru_cache(maxsize=2)
def _loaded(ckpt: str, data_dir: str):
    """:func:`load_net_from_checkpoint` on the CPU, kept for the next call on the same run."""
    return load_net_from_checkpoint(ckpt, data_dir, device="cpu")


def basin_daily(run: str | Path, *, data_dir: str = "data", dp: float = 0.0,
                dt: float | np.ndarray = 0.0) -> pd.DataFrame:
    """Daily flow (date x basin, mm/day) of a 15-CDEC run (its name, or a checkpoint path; a
    multi-timescale run's ``cdec_<BASIN>`` entities are named by basin) over
    the whole record from the cold start, on the engine as trained (its numerics, its basin
    weights), in a climate changed by ``dp`` (precipitation x (1 + dp)) and ``dt`` (degC on the
    temperatures: a number, or a field on the domain's forcing rows and days).

    Cached under ``artifacts/_local/cache/basin_daily/`` by run, by the content of the
    checkpoint and of the engine's code, and by climate; clear it after a change to the
    forcing."""
    import hashlib

    ckpt = paths.dpl_checkpoint(run=str(run)).resolve()
    dp = float(dp) + 0.0                                    # -0.0 -> 0.0
    if np.ndim(dt) == 0:
        dt = float(dt) + 0.0
        tag = f"dt{dt:+g}"
    else:
        dt = np.ascontiguousarray(dt, np.float64)
        tag = f"dt{hashlib.sha1(dt.tobytes()).hexdigest()[:10]}"
    key = content_key(ckpt, *ENGINE_CODE)
    cache = (paths.local(name="cache/basin_daily") / f"{ckpt.parents[1].name}_{key}"
             / f"dp{dp:+g}_{tag}.npz")
    if cache.exists():
        z = np.load(cache)
        return pd.DataFrame(z["flow"].T, index=pd.DatetimeIndex(z["dates"], name="date"),
                            columns=[str(b).removeprefix("cdec_") for b in z["basins"]])
    net, x, dom, cfg, _ = _loaded(str(ckpt), str(data_dir))
    flow = simulate_field(net, x, dom, cfg, dom.dates[0], dom.dates[-1], dom.W.cpu().numpy(),
                          spinup="cold", precip_scale=1.0 + dp, temp_delta=dt)["flow"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, flow=flow, dates=dom.dates.values, basins=np.array(dom.basins))
    return pd.DataFrame(flow.T, index=dom.dates.rename("date"),
                        columns=[str(b).removeprefix("cdec_") for b in dom.basins])


def evaluate_checkpoint(ckpt_path: str | Path, data_dir: str = "data",
                        out_dir: str | Path | None = None) -> pd.DataFrame:
    """A 15-CDEC run: its parameter tables (``params_dpl.csv``, and ``params_canopy.csv`` for a
    Noah-ET field) to the model folder, then its daily flow as trained (:func:`basin_daily`)
    scored against the gauges (:func:`score_basins`) into the results folder."""
    # the run's folders: by default the run of the checkpoint (<run>/checkpoints/x.pt)
    ckp = Path(ckpt_path).resolve()
    run = paths.run_roles(out_dir if out_dir is not None else ckp.parents[1])
    run.model.mkdir(parents=True, exist_ok=True)
    run.results.mkdir(parents=True, exist_ok=True)
    net, x, dom, cfg, _ = _loaded(str(ckp), str(data_dir))
    export_params(net, dom, x).to_csv(run.model / "params_dpl.csv", index=False)
    if cfg.et_mode == "noah":
        export_canopy_params(net, dom, x).to_csv(run.model / "params_canopy.csv", index=False)
    print(f"wrote the parameter tables -> {run.model}", flush=True)
    daily = basin_daily(ckp, data_dir=data_dir)
    return score_basins(daily.to_numpy().T, list(daily.columns), daily.index, run.results,
                        data_dir=data_dir)


#: the substeps a day the benchmark runs the learned numerics with
BENCHMARK_N_INC = (1, 2, 5, 10, 20)


def fidelity_benchmark(data_dir: str = "data", out_dir: str | Path | None = None) -> pd.DataFrame:
    """The archived GA optimum of the 15 CDEC watersheds through the learned numerics
    (:mod:`sacsma.sma_learned`, ``n_inc`` substeps a day for each of :data:`BENCHMARK_N_INC`,
    the training ``fracp_floor``) against the reference model, over the whole record from the
    cold start, on the engine: per watershed and ``n_inc``, the daily KGE / NSE / bias /
    largest difference between the two and the calibration / validation KGE of each against
    the gauge.  Writes ``fidelity_benchmark.csv`` and a figure into ``out_dir`` (default: the
    benchmark's result folder)."""
    from ..engine import Physics
    from ..model import load_domain_forcing

    out = Path(out_dir) if out_dir is not None else paths.dpl_fidelity()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    forcing = load_domain_forcing(data_dir, domain="15cdec")
    dates = forcing.dates
    gage = load_gage(data_dir)
    obs = {b: g.set_index("date")["flow"].reindex(dates) for b, g in gage.groupby("basin")}

    def flow(physics: Physics | None = None) -> pd.DataFrame:
        return run_basins(list(BASINS), data_dir=data_dir, domain="15cdec", forcing=forcing,
                          physics=physics)

    ref = flow()
    rows = []
    for n_inc in BENCHMARK_N_INC:
        sim = flow(Physics(learned=True, n_inc=n_inc))
        for b in BASINS:
            s, r = sim[b], ref[b]
            ob = obs.get(b)
            gauge = [np.nan if ob is None else _obs_kge(q, ob, dates, per)
                     for q in (s, r) for per in ("cal", "val")]
            rows.append({"n_inc": n_inc, "basin": b,
                         "kge_sim": kge(s.to_numpy(), r.to_numpy()),
                         "nse_sim": nse(s.to_numpy(), r.to_numpy()),
                         "pbias_sim": pbias(s.to_numpy(), r.to_numpy()),
                         "max_abs_diff": float(np.max(np.abs(s - r))),
                         **dict(zip(("cal_kge_learned", "val_kge_learned", "cal_kge_reference",
                                     "val_kge_reference"), gauge, strict=True))})
        d = pd.DataFrame(rows[-len(BASINS):])
        print(f"  n_inc {n_inc:2d}: learned vs reference KGE min {d['kge_sim'].min():.6f} | "
              f"max |d| {d['max_abs_diff'].max():.4f} mm/day", flush=True)
    df = pd.DataFrame(rows)
    df["d_cal_kge"] = df["cal_kge_learned"] - df["cal_kge_reference"]
    df["d_val_kge"] = df["val_kge_learned"] - df["val_kge_reference"]
    df.to_csv(out / "fidelity_benchmark.csv", index=False)
    _fidelity_figure(df, out / "figures" / "fidelity_benchmark.png")
    print(f"wrote {out / 'fidelity_benchmark.csv'}", flush=True)
    return df


def _fidelity_figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_incs = list(dict.fromkeys(df["n_inc"]))
    basins = [b for b in BASINS if b in set(df["basin"])]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.2), dpi=300, sharex=True)
    x = np.arange(len(basins))
    width = 0.8 / len(n_incs)
    for c_i, c in enumerate(n_incs):
        d = df[df["n_inc"] == c].set_index("basin").reindex(basins)
        axes[0].bar(x + c_i * width, d["kge_sim"], width, label=f"n_inc {c}")
        axes[1].bar(x + c_i * width, d["max_abs_diff"], width, label=f"n_inc {c}")
    axes[0].set_ylabel("daily KGE, learned vs reference", fontsize=8)
    axes[0].set_ylim(0.8, 1.001)
    axes[0].axhline(0.999, color="0.4", lw=0.6, ls="--")
    axes[1].set_ylabel("max |learned − reference| (mm/day)", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x + 0.4 - width / 2)
    axes[1].set_xticklabels(basins, fontsize=7, rotation=45)
    axes[0].legend(fontsize=6, ncol=5, frameon=False, loc="lower center")
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle("Learned numerics vs the reference model — archived GA optimum", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
