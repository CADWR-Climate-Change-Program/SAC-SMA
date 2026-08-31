"""Per-entity evaluation of a multi-timescale (``multifamily``) checkpoint.

The multi-timescale counterpart of :func:`sacsma.dpl.evaluate.evaluate_checkpoint`:
rebuild the net, stream the full envelope once under ``torch.no_grad()`` (same
spinup convention as training — ten water years ahead of the window), then
score every entity over its OWN registry window at its NATIVE timescale —
daily entities on days, monthly entities on calendar-month sums of the same
simulated flow.  There is no held-out flow validation in this domain by
design (validation happens against CalSim elsewhere), so all metrics are
calibration-window skill.

Outputs (next to the checkpoint, like the other domains):

* ``params_dpl.csv`` — the learned per-(entity, cell) parameter field.
* ``metrics_entities.csv`` — one row per entity: family, timescale, KGE,
  NSE, pbias, r, alpha, beta, n days/months scored (asserted == the
  registry ``n_obs``).
* ``sim_daily_mm.npz`` — the simulated daily basin depth (entities x days)
  over the envelope, for downstream figures/analyses.
* ``figures/skill_by_family.png`` — per-entity KGE by family (house style).
* ``figures/entities/<entity_id>.png`` — the house per-basin diagnostics
  figure (time series + mean-monthly regimes) at the entity's native
  timescale; by default the reviewable cdec_daily + uf_monthly set
  (``hydrographs="all"`` draws all entities, ``"none"`` skips).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .._figures import _period_stats, basin_diagnostics_fig, plt
from ..io import MULTI_TIMESCALE_DOMAIN, domain_dir
from ..metrics import kge, nse, pbias, pearson
from .evaluate import export_canopy_params, export_params, load_net_from_checkpoint
from .forward import initial_state, routing_uh, run_window
from .multi_timescale import EntityObs, load_entity_obs

#: hydrograph families drawn by default (69 USGS figures on request only)
_REVIEW_FAMILIES = ("cdec_daily", "uf_monthly")


def _entity_flow(net, x, dom, cfg) -> tuple[np.ndarray, int, int]:
    """Stream the trained field over the envelope; returns the basin daily
    depth ``(B, t1-t0)`` (numpy float64) plus the envelope indices.

    Mirrors the trainer's no-grad protocol: cold start ten water years ahead
    of the window (clamped to the record start), eager ``run_window`` chunks
    (no CUDA graphs — a single full pass does not need them)."""
    from .multi_timescale import ENVELOPE_END, ENVELOPE_START

    t0 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_START)))
    t1 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_END))) + 1
    # the trainer's spinup basis exactly: honor an explicit early
    # cfg.spinup_start; redirect the (15cdec-era) in-window default to ten
    # water years ahead of the envelope
    spin_req = int(dom.dates.searchsorted(pd.Timestamp(cfg.spinup_start)))
    if spin_req >= t0:
        spin_req = int(dom.dates.searchsorted(
            dom.dates[t0] - pd.DateOffset(years=10)))
    spin = max(min(spin_req, t0), 0)

    net.eval()
    with torch.no_grad():
        out = net(x)
        canopy = out.pop("_canopy", None)
        uh = routing_uh(out, dom.flowlen)
        state = initial_state(dom.n_hru, dom.device, dom.dtype,
                              init_mode=cfg.init_mode, params=out,
                              et_mode=cfg.et_mode)
        sims = []
        t = spin
        while t < t1:
            te = min(t + 512, t1)
            pr, ta, doy, leap = dom.chunk(t, te)
            tn, tx = dom.chunk_tmm(t, te)
            flow, state = run_window(
                pr, ta, doy, leap, dom.lat_rad, dom.elev, out, uh, state,
                n_inc=cfg.n_inc, perc_mode=cfg.perc_mode,
                fracp_floor=cfg.fracp_floor, ninc_mode="fixed",
                et_mode=cfg.et_mode, canopy_params=canopy, tmin=tn, tmax=tx,
                veg_frac=dom.veg_frac, lai=dom.chunk_lai(t, te),
                noah_pet=cfg.noah_pet, sac_pet=cfg.sac_pet,
                pt_snow_albedo=cfg.pt_snow_albedo,
                pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                canopy_lite=cfg.canopy_lite, state_idx=dom.chunk_state(t, te))
            if te > t0:
                sims.append((dom.W @ flow)[:, max(t0 - t, 0):].cpu())
            t = te
    return torch.cat(sims, dim=1).double().numpy(), t0, t1


def _components(s: np.ndarray, o: np.ndarray) -> tuple[float, float]:
    """KGE alpha (std ratio) and beta (mean ratio) on finite pairs."""
    alpha = float(s.std() / max(o.std(), 1e-12))
    beta = float(s.mean() / max(o.mean(), 1e-12))
    return alpha, beta


def _skill_by_family_fig(met: pd.DataFrame, out: Path) -> None:
    """Sorted per-entity cal KGE, one panel per family (house scale 0-1,
    negatives clipped and marked)."""
    fams = [f for f in ("usgs_daily", "cdec_daily", "uf_monthly")
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
        if fam == "usgs_daily":
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
) -> pd.DataFrame:
    """Score a ``multifamily`` checkpoint per entity; returns the metrics."""
    if hydrographs not in ("review", "all", "none"):
        raise ValueError(f"hydrographs {hydrographs!r}")
    net, x, dom, cfg, ck = load_net_from_checkpoint(ckpt_path, data_dir,
                                                    device=device)
    if ck.get("domain") != MULTI_TIMESCALE_DOMAIN:
        raise ValueError(f"checkpoint domain {ck.get('domain')!r} is not "
                         f"{MULTI_TIMESCALE_DOMAIN!r}")
    ckp = Path(ckpt_path).resolve()
    out = Path(out_dir) if out_dir is not None else (
        ckp.parent.parent if ckp.parent.name == "checkpoints"
        else Path("artifacts/multifamily/eval"))
    figdir = out / "figures" / "entities"
    figdir.mkdir(parents=True, exist_ok=True)

    dpl_df = export_params(net, dom, x)
    dpl_df.to_csv(out / "params_dpl.csv", index=False)
    if ck.get("net_config", {}).get("canopy", False):
        export_canopy_params(net, dom, x).to_csv(
            out / "params_canopy.csv", index=False)
    print(f"wrote {out / 'params_dpl.csv'} ({len(dpl_df)} entity-cell rows, "
          f"sel cal KGE {ck.get('cal_kge', float('nan')):.4f})", flush=True)

    eobs: EntityObs = load_entity_obs(dom, data_dir)
    print(f"eval: streaming the envelope "
          f"({dom.n_hru} HRUs, {len(dom.basins)} entities, {cfg.dtype} "
          f"config scored in float64 on {dom.device.type}) ...", flush=True)
    sim, t0, t1 = _entity_flow(net, x, dom, cfg)
    assert (t0, t1) == (eobs.t0, eobs.t1)
    dates = dom.dates[t0:t1]
    np.savez_compressed(
        out / "sim_daily_mm.npz", entity_id=np.array(dom.basins),
        date=dates.to_numpy().astype("datetime64[D]").astype(str),
        sim_mm=sim.astype(np.float32))

    reg = pd.read_csv(domain_dir(data_dir, MULTI_TIMESCALE_DOMAIN)
                      / "entities.csv", dtype={"site_id": str})
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
        rows.append({
            "entity_id": eid, "family": r["family"],
            "timescale": r["timescale"], "site_id": r["site_id"],
            "name": r["name"], "area_mi2": r["area_mi2"],
            "n_obs": int(r["n_obs"]), "n_scored": int(fin.sum()),
            "kge": kge(sf, of), "nse": nse(sf, of), "pbias": pbias(sf, of),
            "r": pearson(sf, of), "alpha": alpha, "beta": beta,
        })
        want_fig = (hydrographs == "all"
                    or (hydrographs == "review"
                        and r["family"] in _REVIEW_FAMILIES))
        if want_fig:
            figs.append((eid, m, unit,
                         pd.Timestamp(r["train_start"]),
                         pd.Timestamp(r["train_end"])))

    met = pd.DataFrame(rows)
    bad = met[met["n_scored"] != met["n_obs"]]
    if len(bad):
        raise AssertionError(f"scored counts diverge from the registry "
                             f"n_obs: {bad['entity_id'].tolist()[:5]}")
    met.to_csv(out / "metrics_entities.csv", index=False)
    fam_tbl = (met.groupby("family")[["kge", "nse", "pbias"]]
               .median().round(3))
    print(f"wrote {out / 'metrics_entities.csv'} ({len(met)} entities); "
          "family medians:", flush=True)
    print(fam_tbl.to_string(), flush=True)

    _skill_by_family_fig(met, out / "figures" / "skill_by_family.png")
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
    print(f"wrote {n_figs} figures under {out / 'figures'}", flush=True)
    return met
