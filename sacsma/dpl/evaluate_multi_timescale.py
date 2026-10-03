"""Per-entity evaluation of a multi-timescale (``multifamily``) checkpoint.

The multi-timescale counterpart of :func:`sacsma.dpl.evaluate.evaluate_checkpoint`:
rebuild the net, stream the full envelope once under ``torch.no_grad()`` (the
state at its start from the timing-independent cycle spinup of
:mod:`sacsma.dpl.spinup`, whatever spinup the run trained with), then
score every entity over its OWN registry window at its NATIVE timescale —
daily entities on days, monthly entities on calendar-month sums of the same
simulated flow.  There is no held-out flow validation in this domain by
design (validation happens against CalSim elsewhere), so all metrics are
calibration-window skill — except for a run with ``holdout_wy``, whose
held-out water years are scored apart (``metrics_holdout.csv``).

Outputs (the parameter table in the run's model folder, the scores and the daily depth in its
results folder, the figures in its local folder; :func:`sacsma.paths.run_roles`):

* ``params_dpl.csv`` — the learned per-(entity, cell) parameter field.
* ``metrics.csv`` — one row per entity: family, timescale, KGE,
  NSE, pbias, r, alpha, beta, n days/months scored (asserted == the
  registry ``n_obs``).  A run with ``holdout_wy`` / ``uf_train_start`` adds
  ``n_holdout`` / ``n_ext`` and the registry-window scores ``n_reg`` /
  ``kge_reg`` / ``nse_reg`` / ``pbias_reg`` (holdout and back-extension out).
* ``metrics_holdout.csv`` — only for a run with ``holdout_wy``: the held-out water
  years scored apart (:func:`holdout_metrics`: USGS daily, uf monthly, the calsim arcs on
  their own gauge-record months and on all months).
* ``sim_daily.npz`` — the simulated daily basin depth (entities x days)
  over the envelope, for downstream figures/analyses.
* ``figures/skill_by_family.png`` — per-entity KGE by family (fixed 0-1 axis).
* ``figures/entities/<entity_id>.png`` — the per-basin diagnostics figure of
  ``sacsma._figures`` (time series + mean-monthly regimes) at the entity's
  native timescale; by default the cdec_daily + uf_monthly entities
  (``hydrographs="all"`` adds the USGS entities, ``"none"`` skips).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import paths
from .._figures import _period_stats, basin_diagnostics_fig, plt
from ..io import MULTI_TIMESCALE_DOMAIN
from ..metrics import kge, nse, pbias, pearson
from .evaluate import export_canopy_params, export_params, load_net_from_checkpoint
from .forward import routing_uh
from .multi_timescale import EntityObs, load_entity_obs
from .spinup import spin_state, stream_rows

#: hydrograph families drawn by default (69 USGS figures on request only)
_REVIEW_FAMILIES = ("cdec_daily", "uf_monthly")


def _entity_flow(net, x, dom, cfg, *, spinup: str = "cycle",
                 dedup_cells: bool = False) -> tuple[np.ndarray, int, int]:
    """Stream the trained field over the envelope; returns the basin daily
    depth ``(B, t1-t0)`` (numpy float64) plus the envelope indices.

    The state at the envelope start comes from ``spinup``
    (:func:`sacsma.dpl.spinup.spin_state`): ``"cycle"`` (default) loops the
    envelope's own first ``cfg.spinup_years`` years ``cfg.spinup_passes`` times
    from the cold start — nothing before the envelope is read; ``"window"`` is the
    legacy ten water years ahead of it (or ``cfg.spinup_start`` when earlier).  Eager ``run_window``
    pieces (no CUDA graphs — a single full pass does not need them).
    ``dedup_cells``: the per-cell physics once per distinct cell
    (:func:`sacsma.dpl.data.with_cell_dedup`)."""
    from .data import with_cell_dedup
    from .multi_timescale import ENVELOPE_END, ENVELOPE_START

    t0 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_START)))
    t1 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_END))) + 1
    if dedup_cells:
        dom = with_cell_dedup(dom, x)

    net.eval()
    with torch.no_grad():
        out = net(dom.phys_x(x))
        canopy = out.pop("_canopy", None)
        uh = routing_uh(out, dom.flowlen, row_cell=dom.row_cell)
        state, how = spin_state(dom, cfg, out, uh, canopy, t0, mode=spinup,
                                agg=lambda f: dom.W @ f)
        print(f"eval: {how}", flush=True)
        sim, _ = stream_rows(dom, cfg, out, uh, canopy, t0, t1, state,
                             lambda f: (dom.W @ f).cpu())
    return sim.double().numpy(), t0, t1


def _components(s: np.ndarray, o: np.ndarray) -> tuple[float, float]:
    """KGE alpha (std ratio) and beta (mean ratio) on finite pairs."""
    alpha = float(s.std() / max(o.std(), 1e-12))
    beta = float(s.mean() / max(o.mean(), 1e-12))
    return alpha, beta


def _scores(s: np.ndarray, o: np.ndarray, min_n: int) -> dict:
    """n finite pairs and the KGE family of scores (NaN below ``min_n`` pairs)."""
    fin = np.isfinite(s) & np.isfinite(o)
    out = dict(n=int(fin.sum()), kge=np.nan, nse=np.nan, pbias=np.nan, r=np.nan,
               alpha=np.nan, beta=np.nan)
    if out["n"] >= min_n:
        sf, of = s[fin], o[fin]
        alpha, beta = _components(sf, of)
        out.update(kge=kge(sf, of), nse=nse(sf, of), pbias=pbias(sf, of), r=pearson(sf, of),
                   alpha=alpha, beta=beta)
    return out


def holdout_metrics(sim: np.ndarray, dates: pd.DatetimeIndex, basins, data_dir: str = "data",
                    holdout_wy: tuple[int, int] = (1976, 1985), *,
                    obs_mask: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per-entity skill over the water years a run held out of training
    (``metrics_holdout.csv``); ``sim`` is the ``(entities, days)`` daily depth over
    ``dates`` (the evaluator's stream or an archived ``sim_daily.npz``).  Rows, one per
    entity x window x basis:

    * daily entities — the native store on the held-out days inside the registry window
      (``obs_mask`` days out): the values the holdout removed from the target (``basis =
      target``; KGE from 90 days).  The CDEC daily stores start after WY1985, so only the USGS
      creeks have rows.
    * ``uf_monthly`` — the DWR monthly depth store over every held-out month (``target``), and
      over the held-out water years before the entity's registry train_start (``WY1976-84``
      for a WY1985 start: the window runs trained on the registry window never saw).
    * ``calsim_monthly`` — the arc depth store on the arc's OWN gauge-record months inside the
      holdout (``own_record``, :func:`sacsma.dpl.calsim.arcs.own_record_months`) and on every
      held-out month (``calsim3``: CalSim3 INFLOW whether gauged or not).

    Monthly rows are calendar-month sums of the simulated flow (complete months), KGE from
    12 months.  Nothing here enters training or selection."""
    import xarray as xr

    from ..cdec15 import load_gage
    from .calsim.arcs import load_depth_store, monthly_depth_from_daily, own_record_months
    from .calsim.tier1 import wy_label
    a_wy, b_wy = int(holdout_wy[0]), int(holdout_wy[1])
    basins = list(basins)
    dates = pd.DatetimeIndex(dates)
    wy_d = np.asarray(dates.year + (dates.month >= 10))
    ho_d = (wy_d >= a_wy) & (wy_d <= b_wy)
    reg = pd.read_csv(paths.entities(data_dir),
                      dtype={"site_id": str}, parse_dates=["train_start", "train_end"])
    reg = reg.set_index("entity_id").loc[basins]
    msim = monthly_depth_from_daily(pd.DataFrame(np.asarray(sim, np.float64).T, index=dates,
                                                 columns=basins))
    months = pd.PeriodIndex(msim.index, freq="M")
    wy_m = np.asarray(months.year + (months.month >= 10))
    ho_m = (wy_m >= a_wy) & (wy_m <= b_wy)
    label = wy_label(a_wy, b_wy)
    fams = set(reg["family"])
    usgs = (xr.open_dataset(paths.usgs_flow(data_dir)) if "usgs_daily" in fams else None)
    gage = load_gage(data_dir) if "cdec_daily" in fams else None
    fnfmm = (pd.read_csv(paths.cdec_fnf(data_dir, "fnf_daily_mm.csv"), parse_dates=["date"])
             if "cdec_daily" in fams else None)
    ufmm = (pd.read_csv(paths.dwr_unimpaired(data_dir, "uf_monthly_mm.csv"),
                        parse_dates=["date"]) if "uf_monthly" in fams else None)
    arcs = load_depth_store(data_dir) if "calsim_monthly" in fams else None
    own = own_record_months(data_dir, (a_wy, b_wy)) if arcs is not None else {}
    rows = []

    def add(eid, r, window, basis, sc):
        rows.append(dict(entity_id=eid, family=r["family"], timescale=r["timescale"],
                         site_id=r["site_id"], window=window, basis=basis, **sc))

    for i, eid in enumerate(basins):
        r = reg.loc[eid]
        if r["timescale"] == "daily":
            if r["family"] == "usgs_daily":
                s = pd.Series(usgs["flow_mm"].sel(gauge=r["site_id"]).values,
                              index=pd.DatetimeIndex(usgs["time"].values))
            elif eid in ("cdec_CLE", "cdec_CSN"):
                g = fnfmm[fnfmm["station"] == r["site_id"]]
                s = pd.Series(g["depth_mm"].to_numpy(), index=g["date"])
            else:
                g = gage[gage["basin"] == r["site_id"]]
                s = pd.Series(g["flow"].to_numpy(), index=pd.DatetimeIndex(g["date"]))
            o = s.reindex(dates).to_numpy(np.float64, copy=True)
            keep = ho_d & np.asarray((dates >= r["train_start"]) & (dates <= r["train_end"]))
            for m in obs_mask:
                e, _, day = m.partition("|")
                if e == eid and pd.Timestamp(day) in dates:
                    keep[dates.get_loc(pd.Timestamp(day))] = False
            o[~keep] = np.nan
            if np.isfinite(o).any():
                add(eid, r, label, "target", _scores(np.asarray(sim[i], np.float64), o, 90))
            continue
        s = msim[eid].to_numpy()
        if r["family"] == "uf_monthly":
            g = ufmm[ufmm["uf"] == int(str(r["site_id"]).split()[1])]
            o = (pd.Series(g["depth_mm"].to_numpy(), index=pd.PeriodIndex(g["date"], freq="M"))
                 .reindex(months).to_numpy(np.float64))
            add(eid, r, label, "target", _scores(s, np.where(ho_m, o, np.nan), 12))
            y0 = int(r["train_start"].year + (r["train_start"].month >= 10))
            if a_wy < y0 <= b_wy:
                pre = ho_m & (wy_m < y0)
                add(eid, r, wy_label(a_wy, y0 - 1), "target",
                    _scores(s, np.where(pre, o, np.nan), 12))
        elif r["family"] == "calsim_monthly":
            arc = str(r["site_id"])
            o = arcs[arc].reindex(months).to_numpy(np.float64) if arc in arcs.columns \
                else np.full(len(months), np.nan)
            mine = np.asarray(months.isin(own.get(arc, pd.PeriodIndex([], freq="M"))))
            add(eid, r, label, "own_record", _scores(s, np.where(ho_m & mine, o, np.nan), 12))
            add(eid, r, label, "calsim3", _scores(s, np.where(ho_m, o, np.nan), 12))
    if usgs is not None:
        usgs.close()
    return pd.DataFrame(rows)


def _skill_by_family_fig(met: pd.DataFrame, out: Path) -> None:
    """Sorted per-entity cal KGE, one panel per family (fixed 0-1 scale,
    negatives clipped and marked)."""
    fams = [f for f in ("usgs_daily", "cdec_daily", "uf_monthly", "calsim_monthly")
            if (met["family"] == f).any()]
    fig, axes = plt.subplots(
        1, len(fams), figsize=(2.6 + 1.6 * len(fams), 3.2),
        gridspec_kw={"width_ratios": [max((met["family"] == f).sum(), 4)
                                      for f in fams]})
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams, strict=True):
        sub = met[met["family"] == fam].sort_values("kge", ascending=False)
        x = np.arange(len(sub))
        ax.bar(x, sub["kge"].clip(lower=0.0), color="tab:blue", width=0.8)
        for i, v in enumerate(sub["kge"]):
            if np.isfinite(v) and v < 0:
                ax.annotate("↓", (i, 0.01), ha="center", va="bottom",
                            fontsize=7, color="tab:blue")
        med = sub["kge"].median()
        ax.axhline(med, color="tab:red", lw=1.0, ls="--")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{fam} (n={len(sub)}, median {med:.3f})")
        if fam in ("usgs_daily", "calsim_monthly"):
            ax.set_xticks([])
            ax.set_xlabel("entities (sorted)")
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(sub["site_id"], rotation=90)
    axes[0].set_ylabel("cal KGE (native timescale)")
    fig.suptitle("Multi-timescale entities — per-entity calibration KGE",
                 fontsize=8, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def evaluate_checkpoint_mt(
    ckpt_path: str | Path,
    data_dir: str = "data",
    out_dir: str | Path | None = None,
    *,
    hydrographs: str = "review",   # review (cdec+uf) | all | none
    device: str | None = None,     # None = cuda if available
    spinup: str = "cycle",         # cycle (timing-independent) | window (legacy)
    dedup_cells: bool = False,     # per-cell physics once per distinct cell
) -> pd.DataFrame:
    """Score a ``multifamily`` checkpoint per entity; returns the metrics.  ``out_dir`` names
    the run (any of its folders; default: the run of the checkpoint): the parameter tables go
    to its model folder, the scores and ``sim_daily.npz`` to its results folder, the
    figures to its local folder."""
    if hydrographs not in ("review", "all", "none"):
        raise ValueError(f"hydrographs {hydrographs!r}")
    net, x, dom, cfg, ck = load_net_from_checkpoint(ckpt_path, data_dir,
                                                    device=device)
    if ck.get("domain") != MULTI_TIMESCALE_DOMAIN:
        raise ValueError(f"checkpoint domain {ck.get('domain')!r} is not "
                         f"{MULTI_TIMESCALE_DOMAIN!r}")
    ckp = Path(ckpt_path).resolve()
    run = paths.run_roles(out_dir if out_dir is not None else (
        ckp.parent.parent if ckp.parent.name == "checkpoints" else paths.local(name="eval")))
    out = run.results
    out.mkdir(parents=True, exist_ok=True)
    figdir = run.local / "figures" / "entities"
    figdir.mkdir(parents=True, exist_ok=True)

    run.model.mkdir(parents=True, exist_ok=True)
    dpl_df = export_params(net, dom, x)
    dpl_df.to_csv(run.model / "params_dpl.csv", index=False)
    if ck.get("net_config", {}).get("canopy", False):
        export_canopy_params(net, dom, x).to_csv(
            run.model / "params_canopy.csv", index=False)
    print(f"wrote {run.model / 'params_dpl.csv'} ({len(dpl_df)} entity-cell rows, "
          f"sel cal KGE {ck.get('cal_kge', float('nan')):.4f})", flush=True)

    # scored against the run's own training target (its obs_mask, water-year
    # holdout and uf back-extension, if any — all from the checkpoint's cfg)
    eobs: EntityObs = load_entity_obs(dom, data_dir, obs_mask=cfg.obs_mask,
                                      holdout_wy=cfg.holdout_wy or None,
                                      uf_train_start=cfg.uf_train_start or None)
    # the holdout / back-extension counts are columns only for a run that used them
    ho_cols = bool(cfg.holdout_wy or cfg.uf_train_start)
    print(f"eval: streaming the envelope "
          f"({dom.n_hru} HRUs, {len(dom.basins)} entities, {cfg.dtype} "
          f"config scored in float64 on {dom.device.type}) ...", flush=True)
    sim, t0, t1 = _entity_flow(net, x, dom, cfg, spinup=spinup,
                               dedup_cells=dedup_cells)
    assert (t0, t1) == (eobs.t0, eobs.t1)
    dates = dom.dates[t0:t1]
    np.savez_compressed(
        out / "sim_daily.npz", entity_id=np.array(dom.basins),
        date=dates.to_numpy().astype("datetime64[D]").astype(str),
        sim_mm=sim.astype(np.float32))

    reg = pd.read_csv(paths.entities(data_dir), dtype={"site_id": str})
    reg = reg.set_index("entity_id").loc[list(dom.basins)]
    midx = (dates.year * 12 + (dates.month - 1)).to_numpy() \
        - int(eobs.month_code[0])
    obs_d = eobs.obs_daily.cpu().numpy()
    obs_m = eobs.obs_monthly.cpu().numpy()

    rows, figs = [], []
    for i, eid in enumerate(dom.basins):
        r = reg.loc[eid]
        if r["timescale"] == "daily":
            j = int(np.flatnonzero(eobs.daily_rows == i)[0])
            o, s = obs_d[j], sim[i]
            m = pd.DataFrame({"date": dates, "flow_sim": s, "flow_obs": o})
            unit = "mm/day"
        else:
            j = int(np.flatnonzero(eobs.monthly_rows == i)[0])
            sm = np.zeros(obs_m.shape[1])
            np.add.at(sm, midx, sim[i])
            o, s = obs_m[j], sm
            mdates = pd.PeriodIndex(
                pd.period_range(dates[0], dates[-1], freq="M")).to_timestamp()
            m = pd.DataFrame({"date": mdates, "flow_sim": s, "flow_obs": o})
            unit = "mm/month"
        fin = np.isfinite(o)
        sf, of = s[fin], o[fin]
        alpha, beta = _components(sf, of)
        reg_cols = {}
        if ho_cols:
            # the same target on the REGISTRY window only (so minus the holdout, and
            # without the uf back-extension): uf reads like-for-like with runs trained
            # on the registry windows (a daily entity: == the columns before)
            dts = pd.DatetimeIndex(m["date"])
            ends = dts + pd.offsets.MonthEnd(0) if r["timescale"] != "daily" else dts
            rw = fin & np.asarray((dts >= pd.Timestamp(r["train_start"]))
                                  & (ends <= pd.Timestamp(r["train_end"])))
            reg_cols = {"n_reg": int(rw.sum()), "kge_reg": kge(s[rw], o[rw]),
                        "nse_reg": nse(s[rw], o[rw]), "pbias_reg": pbias(s[rw], o[rw])}
        rows.append({
            "entity_id": eid, "family": r["family"],
            "timescale": r["timescale"], "site_id": r["site_id"],
            "name": r["name"], "area_mi2": r["area_mi2"],
            "n_obs": int(r["n_obs"]), "n_masked": int(eobs.n_masked[i]),
            **({"n_holdout": int(eobs.n_holdout[i]), "n_ext": int(eobs.n_ext[i])}
               if ho_cols else {}),
            "n_scored": int(fin.sum()),
            "kge": kge(sf, of), "nse": nse(sf, of), "pbias": pbias(sf, of),
            "r": pearson(sf, of), "alpha": alpha, "beta": beta, **reg_cols,
        })
        want_fig = (hydrographs == "all"
                    or (hydrographs == "review"
                        and r["family"] in _REVIEW_FAMILIES))
        if want_fig:
            # a back-extended uf entity's training window starts at uf_train_start
            ts = pd.Timestamp(r["train_start"])
            if cfg.uf_train_start and r["family"] == "uf_monthly":
                ts = min(ts, pd.Timestamp(cfg.uf_train_start))
            figs.append((eid, m, unit, ts, pd.Timestamp(r["train_end"])))

    met = pd.DataFrame(rows)
    want = met["n_obs"] - met["n_masked"]
    if ho_cols:
        want = want - met["n_holdout"] + met["n_ext"]
    bad = met[met["n_scored"] != want]
    if len(bad):
        raise AssertionError(f"scored counts diverge from the registry "
                             f"n_obs: {bad['entity_id'].tolist()[:5]}")
    if ho_cols:
        # registry window = the scored target minus the back-extension months
        bad = met[met["n_reg"] != met["n_scored"] - met["n_ext"]]
        if len(bad):
            raise AssertionError(f"registry-window counts diverge: "
                                 f"{bad['entity_id'].tolist()[:5]}")
    met.to_csv(out / "metrics.csv", index=False)
    fam_tbl = (met.groupby("family")[["kge", "nse", "pbias"]
                                     + (["kge_reg"] if ho_cols else [])]
               .median().round(3))
    print(f"wrote {out / 'metrics.csv'} ({len(met)} entities); "
          "family medians:", flush=True)
    print(fam_tbl.to_string(), flush=True)
    if cfg.holdout_wy:
        # the held-out water years, scored apart (never part of the training target above)
        hm = holdout_metrics(sim, dates, dom.basins, data_dir, cfg.holdout_wy,
                             obs_mask=tuple(cfg.obs_mask))
        # the daily rows are exactly the values the holdout took out of the target
        d = hm[hm["timescale"] == "daily"].set_index("entity_id")["n"]
        want_d = {e: int(eobs.n_holdout[i]) for i, e in enumerate(dom.basins)
                  if int(eobs.n_holdout[i]) and reg.loc[e, "timescale"] == "daily"}
        if d.to_dict() != want_d:
            raise AssertionError("holdout daily counts diverge from the removed target values")
        hm.to_csv(out / "metrics_holdout.csv", index=False)
        print(f"wrote {out / 'metrics_holdout.csv'} ({len(hm)} rows, WY"
              f"{cfg.holdout_wy[0]}-{cfg.holdout_wy[1]}); median KGE by family x window x basis:",
              flush=True)
        print(hm.groupby(["family", "window", "basis"])["kge"].agg(["count", "median"])
              .round(3).to_string(), flush=True)

    _skill_by_family_fig(met, run.local / "figures" / "skill_by_family.png")
    for eid, m, unit, ts, te in figs:
        # everything is calibration in this domain — obs outside the window
        # are masked upstream, so the figure's "validation" panels stay empty
        cal = _period_stats(m["flow_sim"].to_numpy(), m["flow_obs"].to_numpy())
        basin_diagnostics_fig(
            eid, m, cal_end=te, cal=cal, val={"n": 0},
            out=figdir / f"{eid}.png", unit=unit, cal_start=ts,
            obs_label="observed (native store)",
            title_obs="observed flow (training window)")
    n_figs = len(figs) + 1
    print(f"wrote {n_figs} figures under {run.local / 'figures'}", flush=True)
    return met
