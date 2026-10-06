"""Physics-only (Δprecip, ΔT) response surfaces: climate-frozen against
climate-adaptive parameters.

Two runs, left to right in the figure:

  * ``noah_noca`` (column "Noah") — the ``physical`` inputs (23
    physiographic soil/veg/terrain/LAI features).  Its learned SAC + canopy
    parameters are a CLIMATE-FROZEN regionalization — under a (Δp, ΔT)
    perturbation only the FORCING changes.
  * ``noah`` (column "Noah (climate-adaptive)") — the ``physical_climate``
    inputs: the same 23 physiographic features PLUS the 4 climate indices
    p_mean / aridity / snow_frac / seasonality.  Frozen cal/val 0.771/0.801
    against 0.759/0.792, so the added indices cost no present-climate skill.
    Under a perturbation its parameters are RECOMPUTED from the perturbed
    climate indices (a space-for-time response) — so BOTH its forcing and its
    parameters co-vary with the climate.

The per-watershed figure is 4 metrics × 2 columns, each a filled contour of %
change vs that run's own present climate.

Both run as trained on the engine (:func:`sacsma.dpl.evaluate.basin_daily`, cached).  The
grid, the metrics and the regimes are those of :mod:`.surfaces`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ... import paths
from ..._figures import plt  # noqa: F401  (house rcParams)
from ...io import load_basin_area
from ..hybrid.train import RESPONSE_ANCHORS as ANCHORS
from .climatology import _basin_order
from .surfaces import (
    DP,
    DT,
    METRICS,
    REGIME_TITLE,
    REGIMES,
    aggregate_regime,
    metrics_from_daily,
)


def _out(out_dir: str | Path | None) -> Path:
    """The study's result folder, unless ``out_dir`` names another."""
    return Path(out_dir) if out_dir is not None else paths.dpl_study(name="adaptive")


#: canonical physics model-type labels (left → right in the figure).
NOAH = "Noah"
CA_ADAPTIVE = "Noah (climate-adaptive)"
COL_ORDER = [NOAH, CA_ADAPTIVE]


def assemble(data_dir: str = "data") -> pd.DataFrame:
    """Long metrics table: one row per (basin, model, dp, dt) with the 4 raw
    metrics + their signed % change vs that model's own (0, 0) baseline."""
    from ..evaluate import basin_daily

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
        _emit(NOAH, dp, dt, metrics_from_daily(
            basin_daily("noah_noca", data_dir=data_dir, dp=dp, dt=dt), areas))
        _emit(CA_ADAPTIVE, dp, dt, metrics_from_daily(
            basin_daily("noah", data_dir=data_dir, dp=dp, dt=dt), areas))
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
        "○ present   ×  Δp·ΔT anchors   (right col: params co-vary with climate)",
        fontsize=8.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def make_regime_physics_surfaces(tbl: pd.DataFrame, data_dir: str = "data",
                                 out_dir: str | Path | None = None) -> None:
    """One 4×2 ``[noah_noca | noah]`` physics response-surface figure per
    hydroclimate regime (:data:`REGIMES`), the group's basins pooled by
    area-weighted % change."""
    areas = load_basin_area(data_dir, domain="15cdec").set_index(
        "basin")["area_mi2"].to_dict()
    figdir = _out(out_dir) / "noah_climate_adaptive_regimes"
    for reg, basins in REGIMES.items():
        agg = aggregate_regime(tbl, basins, areas)
        title = (f"{REGIME_TITLE[reg]} regime · {len(basins)} basins: "
                 f"{' '.join(basins)}\n"
                 "area-weighted % change   ○ present   ×  Δp·ΔT anchors")
        _plot_basin(reg, agg, figdir / f"{reg}.png", title=title)
    print(f"wrote {len(REGIMES)} regime figures -> {figdir}", flush=True)


def make_adaptive_physics_surfaces(data_dir: str = "data",
                                   out_dir: str | Path | None = None,
                                   *, regen: bool = False) -> pd.DataFrame:
    """Assemble (or reload) the noah_noca / noah metrics table + one 4×2 physics
    response-surface figure per watershed (north → south) and per regime, into ``out_dir``
    (default: the study's result folder)."""
    out_dir = _out(out_dir)
    csv = out_dir / "noah_climate_adaptive_metrics.csv"
    if csv.exists() and not regen:
        tbl = pd.read_csv(csv)
        print(f"loaded {csv}", flush=True)
    else:
        tbl = assemble(data_dir)
        csv.parent.mkdir(parents=True, exist_ok=True)
        tbl.round(4).to_csv(csv, index=False)
        print(f"wrote {csv}", flush=True)

    order = _basin_order(data_dir, sorted(tbl["basin"].unique()))
    figdir = out_dir / "noah_climate_adaptive"
    for b in order:
        _plot_basin(b, tbl[tbl.basin == b], figdir / f"{b}.png")
    print(f"wrote {len(order)} figures -> {figdir}", flush=True)
    make_regime_physics_surfaces(tbl, data_dir, out_dir)
    return tbl


if __name__ == "__main__":
    make_adaptive_physics_surfaces()
