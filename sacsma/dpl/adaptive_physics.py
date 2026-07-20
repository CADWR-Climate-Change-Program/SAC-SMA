"""Physics-only (Δprecip, ΔT) response surfaces: the canonical ``noah`` vs the
climate-adaptive ``noah_ca``.

Canonical physics model-type labels (used here, in the figures, and in RUNS.md):

  * ``noah``            — the canonical dPL-noah (the ``physical`` feature
    backbone: 23 physiographic soil/veg/terrain/LAI features).  Its learned SAC
    + canopy parameters are a CLIMATE-FROZEN regionalization — under a (Δp, ΔT)
    perturbation only the FORCING changes.  This is the Phase-1 physics reference.
  * ``noah_ca``         — the climate-ADAPTIVE dPL-noah (the ``physical_climate``
    backbone: the same 23 physiographic features PLUS the 4 climate indices
    p_mean / aridity / snow_frac / seasonality).  Verified frozen cal/val
    0.779/0.804 ≈ the canonical noah (0.767/0.799), so the added indices cost no
    present-climate skill.  Under a perturbation its parameters are RECOMPUTED
    from the perturbed climate indices (a space-for-time response) — so BOTH its
    forcing and its parameters co-vary with the climate.

The per-watershed figure is 4 metrics × 2 columns ``[ noah | noah_ca ]``, each a
filled contour of % change vs that model's own present climate — the canonical
physics vs the climate-adaptive physics.

Physics is the fast frozen numba noah-lite core (``run_basin`` with
``et_scheme='noah_lite'``, PT potential) — the same core that scores the frozen
checkpoints.  ``noah`` reuses the Phase-1 cache
(:func:`dtdp_response._frozen_noah`); ``noah_ca`` runs are cached under
``artifacts/dpl/_adaptive_cache/``.
"""
from __future__ import annotations

import dataclasses as dc
from pathlib import Path

import numpy as np
import pandas as pd

from .._figures import plt  # noqa: F401  (house rcParams)
from ..io import load_basin_area
from .climatology import _basin_order
from .dtdp_response import (DOMAIN, DP, DT, METRICS, REGIMES, _REGIME_TITLE,
                            _aggregate_regime, _frozen_noah, _metrics_from_daily)

CA_CKPT = "artifacts/dpl/noah_ca/checkpoints/best.pt"
_CA_CACHE = Path("artifacts/dpl/_adaptive_cache")

#: canonical physics model-type labels (left → right in the figure).
NOAH = "noah"
CA_ADAPTIVE = "noah_ca (adaptive)"
COL_ORDER = [NOAH, CA_ADAPTIVE]

#: the noah_ca dt·dp hybrid's 14 response-loss anchors (mirrors
#: ``noah_ca_hybrids.ANCHORS``) — drawn on these physics surfaces too so the eye
#: can cross-compare the same (Δp, ΔT) reference grid across all figure sets.
ANCHORS = [(dp, dt) for dp in (-0.2, -0.1, 0.0, 0.1, 0.2) for dt in (0.0, 2.0, 4.0)
           if not (dp == 0.0 and dt == 0.0)]

_CA: dict | None = None   # in-process cache of the loaded noah_ca net + baseline
_FA = None                # in-process cache of the noah_ca numba forcing


def _load_ca(data_dir: str = "data") -> dict:
    """Load the noah_ca net + present-climate (baseline) params once."""
    global _CA
    if _CA is not None:
        return _CA
    import numpy as _np

    from .evaluate import (export_canopy_params, export_params,
                           load_net_from_checkpoint)
    from .features import FeatureSet
    net, x0, dom, cfg, ck = load_net_from_checkpoint(CA_CKPT, data_dir)
    if ck.get("variant") != "physical_climate":
        raise ValueError(f"noah_ca ckpt must be physical_climate, got "
                         f"{ck.get('variant')!r}")
    stats = FeatureSet(x=_np.empty((0, 0), _np.float32), **ck["features"])
    _CA = dict(net=net, x0=x0, dom=dom, stats=stats,
               base_dpl=export_params(net, dom, x0),
               base_can=export_canopy_params(net, dom, x0))
    return _CA


def adaptive_params(dp: float, dt: float, data_dir: str = "data"):
    """(dpl_df, canopy_df) with the climate indices recomputed on (dp,dt)-perturbed
    forcing (physiographic features + z-scoring unchanged); exact at (0,0)."""
    import torch

    from ..io import soilveg_path
    from .evaluate import export_canopy_params, export_params
    from .features import build_features
    C = _load_ca(data_dir)
    if dp == 0.0 and dt == 0.0:
        return C["base_dpl"], C["base_can"]
    dom, stats, net, x0 = C["dom"], C["stats"], C["net"], C["x0"]
    f = dom.forcing   # climate_indices uses prcp+tavg only (tmin/tmax may be None)
    fp = dc.replace(f, prcp=f.prcp * (1.0 + dp), tavg=f.tavg + dt, _f64={})
    fs = build_features(dom.hrus, variant="physical_climate", forcing=fp,
                        climate_window=stats.climate_window,
                        climate_product=stats.climate_product,
                        physical_path=soilveg_path(data_dir, DOMAIN), stats=stats)
    x = torch.as_tensor(fs.x).to(x0.device, x0.dtype)
    return export_params(net, dom, x), export_canopy_params(net, dom, x)


def noah_ca_daily(dp: float, dt: float, mode: str,
                  data_dir: str = "data") -> pd.DataFrame:
    """noah_ca daily basin flow (date x basin, mm/day) under (dp,dt), ``mode`` in
    {``static``, ``adaptive``}.  Static = present-climate params on perturbed
    forcing; adaptive = params recomputed under the perturbed climate.  Cached."""
    dp, dt = dp + 0.0, dt + 0.0    # normalize IEEE -0.0 -> +0.0 (arange artifact)
    cache = _CA_CACHE / f"{mode}_dp{dp:+.2f}_dt{dt:+.1f}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")
    global _FA
    from ..cdec15 import BASINS
    from ..model import attach_tminmax, load_domain_forcing, run_basin
    if _FA is None:
        f = load_domain_forcing(data_dir, domain=DOMAIN)
        attach_tminmax(data_dir, DOMAIN, f)
        _FA = f
    if mode == "adaptive":
        dpl, can = adaptive_params(dp, dt, data_dir)
    elif mode == "static":
        C = _load_ca(data_dir)
        dpl, can = C["base_dpl"], C["base_can"]
    else:
        raise ValueError(f"mode {mode!r}")
    fp = dc.replace(_FA, prcp=_FA.prcp * (1.0 + dp), tavg=_FA.tavg + dt,
                    tmin=_FA.tmin + dt, tmax=_FA.tmax + dt, _f64={})
    cols = {}
    for b in BASINS:
        s = run_basin(b, data_dir=data_dir, domain=DOMAIN, forcing=fp, params=dpl,
                      parallel=True, pet_source="priestley_taylor",
                      pt_snow_albedo=0.0, pt_dewpoint_depression=0.0,
                      et_scheme="noah_lite", canopy_params=can)
        cols[b] = s.set_index("date")["flow"]
    df = pd.DataFrame(cols)
    df.index.name = "date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(cache)
    return df


def assemble(data_dir: str = "data") -> pd.DataFrame:
    """Long metrics table: one row per (basin, model, dp, dt) with the 4 raw
    metrics + their signed % change vs that model's own (0, 0) baseline."""
    areas = load_basin_area(data_dir, domain="15cdec").set_index(
        "basin")["area_mi2"].to_dict()
    grid = [(float(dp), float(dt)) for dp in DP for dt in DT]
    rows = []

    def _emit(model, dp, dt, met):
        for b in met.index:
            rows.append(dict(basin=b, model=model, dp=round(dp, 4),
                             dt=round(dt, 4), **{k: float(met.loc[b, k])
                                                 for k in ("annual", "freshet",
                                                           "q999", "q30")}))

    for i, (dp, dt) in enumerate(grid):
        _emit(NOAH, dp, dt, _metrics_from_daily(_frozen_noah(dp, dt, data_dir), areas))
        _emit(CA_ADAPTIVE, dp, dt,
              _metrics_from_daily(noah_ca_daily(dp, dt, "adaptive", data_dir), areas))
        print(f"  [{i + 1}/{len(grid)}] ({dp:+.2f},{dt:+.1f}) done", flush=True)

    tbl = pd.DataFrame(rows)
    base = tbl[(tbl.dp == 0.0) & (tbl.dt == 0.0)].set_index(["basin", "model"])
    for k, _ in METRICS:
        b0 = tbl.set_index(["basin", "model"]).index.map(base[k])
        tbl[f"pct_{k}"] = 100.0 * (tbl[k].to_numpy() / np.asarray(b0, float) - 1.0)
    return tbl


def _plot_basin(basin: str, sub: pd.DataFrame, out: Path,
                title: str | None = None) -> None:
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    X, Y = np.meshgrid(DP * 100.0, DT)
    fig, axes = plt.subplots(len(METRICS), len(COL_ORDER), figsize=(5.2, 8.4),
                             sharex=True, sharey=True, constrained_layout=True)
    for r, (mkey, mlab) in enumerate(METRICS):
        surf = {}
        for mdl in COL_ORDER:
            s = sub[sub.model == mdl]
            surf[mdl] = (s.pivot(index="dt", columns="dp", values=f"pct_{mkey}")
                         .reindex(index=DT, columns=DP).to_numpy())
        allv = np.concatenate([z.ravel() for z in surf.values()])
        vmax = max(float(np.nanpercentile(np.abs(allv), 98)), 1.0)
        levels = MaxNLocator(nbins=12, symmetric=True).tick_values(-vmax, vmax)
        vlim = float(max(abs(levels[0]), abs(levels[-1])))
        norm = Normalize(vmin=-vlim, vmax=vlim)
        cf = None
        for c, mdl in enumerate(COL_ORDER):
            ax = axes[r, c]
            Z = surf[mdl]
            cf = ax.contourf(X, Y, Z, levels=levels, cmap="RdBu", norm=norm,
                             extend="both")
            ax.contour(X, Y, Z, levels=levels, colors="0.35", linewidths=0.25)
            for adp, adt in ANCHORS:             # dt·dp anchor grid (cross-ref)
                ax.plot(adp * 100.0, adt, marker="x", ms=3.0, mew=0.8,
                        color="0.15", zorder=5, clip_on=False)
            ax.plot(0.0, 0.0, marker="o", ms=3.0, mfc="w", mec="0.1", mew=0.8,
                    zorder=6, clip_on=False)
            if r == 0:
                ax.set_title(mdl, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{mlab}\nΔT (°C)", fontsize=7.5)
            if r == len(METRICS) - 1:
                ax.set_xlabel("Δprecip (%)", fontsize=7.5)
            ax.set_xticks([-20, -10, 0, 10, 20])
            ax.set_yticks([0, 1, 2, 3, 4])
            ax.tick_params(labelsize=6.5)
        cb = fig.colorbar(cf, ax=list(axes[r, :]), fraction=0.05, pad=0.01,
                          ticks=levels[::2])
        cb.set_label("% change", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle(
        title if title is not None else
        f"{basin} — physics climate-response surfaces (% change vs present)\n"
        "○ present   ×  dt·dp anchors   noah = frozen   noah_ca = adaptive",
        fontsize=8.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def make_regime_physics_surfaces(tbl: pd.DataFrame, data_dir: str = "data",
                                 out_dir: str | Path = "artifacts/dpl") -> None:
    """One 4×2 ``[noah | noah_ca]`` physics response-surface figure per
    hydroclimate regime (:data:`REGIMES`), the group's basins pooled by
    area-weighted % change."""
    areas = load_basin_area(data_dir, domain="15cdec").set_index(
        "basin")["area_mi2"].to_dict()
    figdir = Path(out_dir) / "figures" / "adaptive_physics_regimes"
    for reg, basins in REGIMES.items():
        agg = _aggregate_regime(tbl, basins, areas)
        title = (f"{_REGIME_TITLE[reg]} regime · {len(basins)} basins: "
                 f"{' '.join(basins)}\n"
                 "area-weighted % change   ○ present   ×  dt·dp anchors")
        _plot_basin(reg, agg, figdir / f"{reg}.png", title=title)
    print(f"wrote {len(REGIMES)} regime figures -> {figdir}", flush=True)


def make_adaptive_physics_surfaces(data_dir: str = "data",
                                   out_dir: str | Path = "artifacts/dpl",
                                   *, regen: bool = False) -> pd.DataFrame:
    """Assemble (or reload) the noah / noah_ca metrics table + one 4×2 physics
    response-surface figure per watershed (north → south) and per regime."""
    out_dir = Path(out_dir)
    csv = out_dir / "adaptive_physics_metrics.csv"
    if csv.exists() and not regen:
        tbl = pd.read_csv(csv)
        print(f"loaded {csv}", flush=True)
    else:
        tbl = assemble(data_dir)
        csv.parent.mkdir(parents=True, exist_ok=True)
        tbl.round(4).to_csv(csv, index=False)
        print(f"wrote {csv}", flush=True)

    order = _basin_order(data_dir, sorted(tbl["basin"].unique()))
    figdir = out_dir / "figures" / "adaptive_physics"
    for b in order:
        _plot_basin(b, tbl[tbl.basin == b], figdir / f"{b}.png")
    print(f"wrote {len(order)} figures -> {figdir}", flush=True)
    make_regime_physics_surfaces(tbl, data_dir, out_dir)
    return tbl


if __name__ == "__main__":
    make_adaptive_physics_surfaces()
