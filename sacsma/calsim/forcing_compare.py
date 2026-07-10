"""Forcing-product flow comparisons vs the Livneh-unsplit baseline run.

Two alternate climate products, each applied to TWO models — **SAC-SMA** (this
repo's daily runs) and **VIC** (the CalSim3 pipeline's routed monthly runs) —
with the SAME figure structure for every set:

* ``historical_lto`` — the split-precipitation LTO climate, labelled simply
  **Split** vs the **Unsplit** baseline, reported as **Unsplit − Split**
  (prefixes ``split_unsplit_`` / ``vic_split_unsplit_``).  The difference is
  concentrated **before 1950** (median |volume difference| ~11% in 1915-1949
  vs ~2-3% in 1950-2018).
* ``wgen_product_a`` — detrended temperature, identical precipitation,
  labelled **Detrended** vs the **Baseline** run, reported as
  **Detrended − Baseline** (prefixes ``wgen_`` / ``vic_wgen_``; negative —
  detrending warms the early record most, so the effect tapers ≈−5% → ≈−2%).

Every figure separates the 1915-1949 / 1950-2018 periods and shows the CalLite
calibration sets 9unimp + 11obs (:data:`SETS`), basins north->south.

Inputs:

* SAC-SMA — the committed run tables: the parity-exact ``simflow`` reference
  (the Livneh-unsplit run) and ``artifacts/calsim/<product>/flow_daily_<domain>.csv``
  (regenerate with ``sacsma run ALL --domain <d> --forcing <product>``).
* VIC — the routed monthly tables ``data/calsim/vic_routed_monthly[_<product>].csv``
  (TAF/month; the ``Historical_Unsplit`` baseline / ``Historical`` split /
  ``Product_A`` detrended runs), aggregated to basins exactly like the
  cross-compare anchor (crosswalk ``vic_basin`` major-basin series; a rim basin
  keeps only its own system's series) and converted to depth over the canonical
  CalSim catchment area.

Outputs -> ``artifacts/calsim/forcing_compare/figures/``, per set prefix:

* ``<p>_volume_by_period.png`` — % volume difference per watershed, one bar
  per period, one panel per domain.
* ``<p>_annual_diff.png``      — 5-yr rolling water-year volume difference,
  one coloured line per watershed, with a short cross-basin mean-|difference|
  panel below; 1950 marked.
* ``<p>_regime_by_period_<domain>.png`` — mean-monthly regime, baseline vs
  product, both periods, one panel per basin.

Plus one **cross-model aggregate** figure per climate product
(``agg_split_unsplit_annual_diff.png``, ``agg_wgen_annual_diff.png``): SAC-SMA
and VIC on one axes, the % difference of the aggregate water-year volume
summed over the **disjoint** watersheds (per-basin depth × canonical CalSim
catchment area; :data:`AGG_EXCLUDE` drops the double-counted SHA/BLB), 5-yr
rolling, with each model's per-period aggregate as dashed level lines.

And the **anchor-skill artifact** for the split product
(``split_unsplit_anchor_skill.csv`` + ``figures/split_unsplit_skill_boxplot.png``):
both forcings re-simulated through the screened-footprint anchor
(:func:`sacsma.calsim.compare.build_anchor_long` with ``product=``) and scored
vs CalSim3 per basin (KGE/NSE/pbias; full + pre/post-WY1950, identical months).

Usage::

    python -m sacsma.calsim.forcing_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .._figures import plt  # Agg backend
from ..io import load_reference
from .compare import _BASIN_ABBREV, basin_order_north_south

_FIG_W, _DPI = 6.5, 600  # <=6.5in wide; high-dpi small-font deck figures
_RC = {
    "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
    "figure.titlesize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
}

#: analysis periods: the storm-splitting artifact is concentrated before 1950.
PERIODS = (
    ("1915–1949", "1915-01", "1949-12"),
    ("1950–2018", "1950-01", "2018-12"),
)

#: the calibration sets shown in every figure (the alternate products exist only
#: for the CalSim domains; 12rim duplicates the 11obs rim systems and is omitted).
SETS = ("9unimp", "11obs")

#: figure sets, keyed by output prefix: each compares one ``model``'s run under an
#: alternate climate ``product`` against that model's Livneh-unsplit baseline run.
#: ``title`` prefixes the figure titles; ``label``/``base`` name the product/baseline
#: lines; ``sign`` sets the difference convention (+1 = product − baseline, −1 =
#: baseline − product) with ``pct_phrase`` stating it in the axis labels/titles;
#: ``legend_loc`` = where the volume-bar legend has empty space.  The VIC sets reuse
#: their climate product's styling — the ``vic_`` prefix and titles carry the model.
PRODUCTS = {
    "split_unsplit": dict(
        model="SAC-SMA", product="historical_lto",
        title="Split precipitation", label="Split", base="Unsplit",
        sign=-1.0, pct_phrase="Unsplit − Split",
        c1="#d95f0e", c2="#fdbe85", legend_loc="upper right",
    ),
    "wgen": dict(
        model="SAC-SMA", product="wgen_product_a",
        title="WGEN Product A (detrended temperature)",
        label="Detrended", base="Baseline",
        sign=1.0, pct_phrase="Detrended − Baseline",
        c1="#2c7fb8", c2="#a6cbe3", legend_loc="upper left",
    ),
    "vic_split_unsplit": dict(
        model="VIC", product="historical_lto",
        title="Split precipitation", label="Split", base="Unsplit",
        sign=-1.0, pct_phrase="Unsplit − Split",
        c1="#d95f0e", c2="#fdbe85", legend_loc="upper right",
    ),
    "vic_wgen": dict(
        model="VIC", product="wgen_product_a",
        title="WGEN Product A (detrended temperature)",
        label="Detrended", base="Baseline",
        sign=1.0, pct_phrase="Detrended − Baseline",
        c1="#2c7fb8", c2="#a6cbe3", legend_loc="upper left",
    ),
}


# --------------------------------------------------------------------------
# per-model basin series: {basin: (baseline, product)} monthly mm (PeriodIndex)
# --------------------------------------------------------------------------

def _monthly_mm(df: pd.DataFrame) -> pd.Series:
    """Daily ``[date, flow]`` (mm/day) -> monthly total mm, PeriodIndex[M]."""
    return df.groupby(df["date"].dt.to_period("M"))["flow"].sum()


def _sacsma_series(data_dir, calsim_art: Path, cfg: dict, domain: str) -> dict:
    """SAC-SMA per-basin monthly mm: ``simflow`` reference vs the product run table."""
    run = pd.read_csv(calsim_art / cfg["product"] / f"flow_daily_{domain}.csv",
                      parse_dates=["date"])
    return {b: (_monthly_mm(load_reference(data_dir, basin=b, domain=domain)),
                _monthly_mm(run[run["basin"] == b]))
            for b in basin_order_north_south(data_dir, domain)}


def _vic_series(data_dir, cfg: dict, domain: str) -> dict:
    """VIC per-basin monthly mm: the ``Historical_Unsplit`` baseline vs the product run.

    Basin aggregation is identical to the cross-compare anchor
    (:func:`sacsma.calsim.compare._anchor_set_taf`): each basin sums its crosswalk
    ``vic_basin`` major-basin series, a rim basin keeping only its own system's series
    (the 8-River total already includes nested arcs).  TAF/month is converted to depth
    over the canonical CalSim catchment area, so panels read like the SAC-SMA ones.
    """
    from ..io import mmday_to_cfs
    from . import load_vic_monthly
    from .catchments import BASIN_RIM_SYSTEM, basin_areas
    from .compare import _AF_PER_CFS_DAY, load_basin_nodes, load_name_map

    taf_per_mm_mi2 = mmday_to_cfs(1.0, 1.0) * _AF_PER_CFS_DAY / 1000.0
    tables = [load_vic_monthly(data_dir), load_vic_monthly(data_dir, product=cfg["product"])]
    for i, t in enumerate(tables):
        tables[i] = t.assign(month=pd.to_datetime(t["date"]).dt.to_period("M"))
    nodes = load_basin_nodes(data_dir, domain)
    summable = nodes[nodes["in_calsim3"].astype(bool)]
    arc2vic = load_name_map(data_dir)
    bsys = BASIN_RIM_SYSTEM.get(domain, {})
    areas = basin_areas(data_dir, domain=domain)
    out = {}
    for basin, g in summable.groupby("basin"):
        own = bsys.get(basin)
        varcs = [a for a, sy in zip(g["arc"].astype(str), g["system"], strict=True)
                 if own is None or sy == own]
        vnames = {arc2vic.get(a, a) for a in varcs}
        out[basin] = tuple(
            t[t["vic_name"].isin(vnames)].groupby("month")["flow_taf"].sum()
            / (areas[basin] * taf_per_mm_mi2)
            for t in tables)
    return out


def _series_for(data_dir, calsim_art: Path, cfg: dict, domain: str) -> dict:
    return (_vic_series(data_dir, cfg, domain) if cfg["model"] == "VIC"
            else _sacsma_series(data_dir, calsim_art, cfg, domain))


# --------------------------------------------------------------------------
# period aggregates (all from the monthly mm series)
# --------------------------------------------------------------------------

def _window(m: pd.Series, a: str, b: str) -> pd.Series:
    return m[(m.index >= pd.Period(a, "M")) & (m.index <= pd.Period(b, "M"))]


def _pct_diff_by_period(base: pd.Series, run: pd.Series, sign: float = 1.0) -> list[float]:
    """% volume difference for each analysis period, in % of the base run:
    run − base (``sign=+1``) or base − run (``sign=-1``)."""
    return [float(sign * 100.0 * (_window(run, a, b).mean() / _window(base, a, b).mean() - 1.0))
            for _, a, b in PERIODS]


def _wy_annual(m: pd.Series) -> pd.Series:
    """Water-year total flow (mm) indexed by WY end year, complete WYs only."""
    wy = m.index.year + (m.index.month >= 10).astype(int)
    ann = m.groupby(wy).sum()
    return ann.loc[(ann.index >= 1916) & (ann.index <= 2018)]


def _mean_monthly(m: pd.Series, a: str, b: str) -> pd.Series:
    """Water-year-ordered (O..S) mean monthly flow (mm/month) over [a, b]."""
    sub = _window(m, a, b)
    clim = sub.groupby(sub.index.month).mean()
    return clim.reindex([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def volume_by_period_fig(series: dict, orders: dict, cfg: dict, path: Path) -> None:
    """Horizontal bars: % volume difference vs baseline, one bar per period."""
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            len(orders), 1, figsize=(_FIG_W, 0.225 * sum(map(len, orders.values()))),
            sharex=True, gridspec_kw={"height_ratios": [len(o) for o in orders.values()]},
        )
        for ax, (domain, order) in zip(axes, orders.items(), strict=True):
            vals = {b: _pct_diff_by_period(*series[domain][b], cfg["sign"]) for b in order}
            y = range(len(order))
            for k, ((plabel, _, _), color) in enumerate(
                    zip(PERIODS, (cfg["c1"], cfg["c2"]), strict=True)):
                off = 0.2 if k == 0 else -0.2
                ax.barh([i + off for i in y], [vals[b][k] for b in order],
                        height=0.38, color=color, label=plabel)
            ax.axvline(0, color="0.4", lw=0.7)
            ax.set_yticks(list(y), order)
            ax.invert_yaxis()  # north at top
            ax.set_ylabel(domain)
            ax.grid(axis="x", color="0.9", lw=0.5)
            ax.set_axisbelow(True)
        axes[0].legend(loc=cfg["legend_loc"], frameon=False, title=None)
        axes[-1].set_xlabel(f"{cfg['pct_phrase']} {cfg['model']} runoff volume (%)")
        fig.suptitle(f"{cfg['title']}: {cfg['model']} runoff volume difference by period "
                     "(basins north→south)")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)


def annual_diff_fig(series: dict, orders: dict, cfg: dict, path: Path) -> None:
    """5-yr rolling water-year volume difference: one coloured line per watershed
    (legend below) plus a short cross-basin mean-|difference| panel."""
    labels, diffs = [], []
    for domain in SETS:
        for b in orders[domain]:
            base, r = (_wy_annual(m) for m in series[domain][b])
            labels.append(_BASIN_ABBREV.get(b, b))
            diffs.append((cfg["sign"] * 100.0 * (r / base - 1.0))
                         .rolling(5, center=True).mean())
    d = pd.concat(diffs, axis=1)
    with plt.rc_context(_RC):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(_FIG_W, 3.6), sharex=True,
                                      gridspec_kw={"height_ratios": (3, 1)})
        colors = plt.get_cmap("tab20").colors
        for k, lab in enumerate(labels):
            ax.plot(d.index, d.iloc[:, k], color=colors[k % len(colors)],
                    lw=0.7, label=lab)
        ax2.plot(d.index, d.abs().mean(axis=1), color="0.1", lw=1.2)
        ax2.set_ylabel("mean |diff| (%)")
        ax2.set_ylim(bottom=0)
        fig.legend(*ax.get_legend_handles_labels(), ncol=7, loc="lower center",
                   frameon=False, fontsize=5)
        ax.axhline(0, color="0.4", lw=0.7)
        for a in (ax, ax2):
            a.axvline(1950, color="0.2", lw=0.8, ls=":")
            a.set_xlim(1916, 2018)
            a.grid(color="0.92", lw=0.5)
            a.set_axisbelow(True)
        ax.text(1951, 0.93, "1950", fontsize=5.5, color="0.2",
                transform=ax.get_xaxis_transform())
        ax2.set_xlabel("water year")
        ax.set_ylabel("volume difference (%)")
        ax.set_title(f"{cfg['title']}: {cfg['model']} water-year runoff volume difference "
                     f"({cfg['pct_phrase']})\n(5-yr rolling, one line per 9unimp/11obs "
                     "watershed)")
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)


def regime_domain_fig(series_d: dict, order: list, cfg: dict, domain: str,
                      path: Path) -> None:
    """Mean-monthly regime, baseline vs product, both periods: one panel per basin
    of ``domain`` (north -> south); solid = baseline, dashed = product, dark =
    1915-1949, light = 1950-2018.  Each panel is annotated top-right with the
    per-period volume difference (``sign`` convention, period-coloured)."""
    ncol = 4 if len(order) > 9 else 3
    nrow = -(-len(order) // ncol)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(_FIG_W, 1.25 * nrow + 1.0),
                                 sharex=True)
        for ax in axes.flat[len(order):]:
            ax.set_visible(False)
        for ax, basin in zip(axes.flat, order, strict=False):  # trailing axes hidden
            base, r = series_d[basin]
            for (plabel, a, b), color in zip(PERIODS, (cfg["c1"], cfg["c2"]),
                                             strict=True):
                ax.plot(range(12), _mean_monthly(base, a, b), color=color, lw=1.1,
                        label=f"{cfg['base']} {plabel}")
                ax.plot(range(12), _mean_monthly(r, a, b), color=color, lw=1.1,
                        ls="--", label=f"{cfg['label']} {plabel}")
            for v, color, dy in zip(_pct_diff_by_period(base, r, cfg["sign"]),
                                    (cfg["c1"], cfg["c2"]), (0.0, 0.12), strict=True):
                ax.text(0.96, 0.94 - dy, f"{v:+.1f}%", transform=ax.transAxes,
                        ha="right", va="top", color=color, fontsize=5.5)
            ax.set_title(basin)
            ax.set_xticks(range(12), list("ONDJFMAMJJAS"))
            ax.grid(color="0.92", lw=0.5)
            ax.set_axisbelow(True)
        for i in range(nrow):
            axes[i, 0].set_ylabel("mm/month")
        for j in range(ncol):  # x tick labels on each column's last VISIBLE panel
            vis = [i for i in range(nrow) if i * ncol + j < len(order)]
            if vis and vis[-1] < nrow - 1:
                axes[vis[-1], j].tick_params(labelbottom=True)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, 0.925), frameon=False)
        fig.suptitle(f"{cfg['title']}: {cfg['model']} mean-monthly runoff regime — {domain}\n"
                     f"(north→south; % = {cfg['pct_phrase']} volume)")
        fig.tight_layout(rect=(0, 0, 1, 0.885))
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)


#: watersheds excluded from the cross-model AGGREGATE so it sums disjoint drainage:
#: SHA is nested inside BND (SRBB ⊇ SHAS), and 11obs BLB is the same three ST arcs
#: (I_BLKBT/I_EPARK/I_SGRGE) as 9unimp StonyCreek — each would be double-counted.
AGG_EXCLUDE = {("11obs", "SHA"), ("11obs", "BLB")}


def _aggregate(series: dict, orders: dict, data_dir) -> tuple[pd.Series, pd.Series]:
    """(baseline, product) total monthly volume (mm·mi², units cancel in the % diff)
    across the disjoint 9unimp+11obs watersheds: per-basin depth × canonical CalSim
    catchment area, :data:`AGG_EXCLUDE` dropped."""
    from .catchments import basin_areas

    base_parts, run_parts = [], []
    for domain in SETS:
        areas = basin_areas(data_dir, domain=domain)
        for b in orders[domain]:
            if (domain, b) in AGG_EXCLUDE:
                continue
            base, run = series[domain][b]
            base_parts.append(base * areas[b])
            run_parts.append(run * areas[b])
    return (pd.concat(base_parts, axis=1).dropna().sum(axis=1),
            pd.concat(run_parts, axis=1).dropna().sum(axis=1))


def agg_models_fig(model_pairs: dict, cfg: dict, path: Path) -> None:
    """SAC-SMA vs VIC on one axes: % difference of the aggregate water-year volume
    (:func:`_aggregate`), 5-yr rolling; each model's per-period aggregate difference
    is drawn as dashed level lines and quoted in the legend."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(_FIG_W, 2.6))
        for (model, (base, run)), color in zip(model_pairs.items(),
                                               ("#c02f1d", "#2c7fb8"), strict=True):
            ann_b, ann_r = _wy_annual(base), _wy_annual(run)
            diff = (cfg["sign"] * 100.0 * (ann_r / ann_b - 1.0)).rolling(5, center=True).mean()
            pcts = _pct_diff_by_period(base, run, cfg["sign"])
            ax.plot(diff.index, diff, color=color, lw=1.2,
                    label=f"{model}   {PERIODS[0][0]}: {pcts[0]:+.1f}%   "
                          f"{PERIODS[1][0]}: {pcts[1]:+.1f}%")
            for (_, a, b), v in zip(PERIODS, pcts, strict=True):
                ax.plot([max(int(a[:4]), 1916), min(int(b[:4]), 2018)], [v, v],
                        color=color, lw=0.8, ls="--", alpha=0.8)
        ax.axhline(0, color="0.4", lw=0.7)
        ax.axvline(1950, color="0.2", lw=0.8, ls=":")
        ax.text(1951, 0.93, "1950", fontsize=5.5, color="0.2",
                transform=ax.get_xaxis_transform())
        ax.set_xlim(1916, 2018)
        ax.grid(color="0.92", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("water year")
        ax.set_ylabel("aggregate volume difference (%)")
        ax.legend(loc="best", frameon=False)
        ax.set_title(f"{cfg['title']}: aggregate water-year runoff volume difference "
                     f"({cfg['pct_phrase']}), SAC-SMA vs VIC\n(total over the disjoint "
                     "9unimp+11obs watersheds, 5-yr rolling; dashed = period aggregate)")
        fig.tight_layout()
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)


# --------------------------------------------------------------------------
# anchor-skill effect: both climates scored vs CalSim3 on the screened anchor
# --------------------------------------------------------------------------

#: the house WY1950 split for the skill periods
_SKILL_SPLIT = "1949-10-01"


def split_unsplit_skill_table(data_dir: str | Path = "data") -> pd.DataFrame:
    """Anchor-basin skill vs CalSim3 under both climates: the Livneh-unsplit baseline
    vs the split-lineage ``historical_lto`` run.

    Both forcings go through the official screened-footprint anchor pipeline
    (:func:`~sacsma.calsim.compare.build_anchor_long`) and are scored per basin on
    **identical months** — the intersection of the two runs, VIC and CalSim3 (keeping
    VIC in the intersection keeps the unsplit rows consistent with the compare run's
    ``anchor_metrics.csv``) — for the full period plus the house pre/post-WY1950
    split (:data:`_SKILL_SPLIT`).  Returns
    ``[set, basin, period, forcing, n_months, kge, nse, pbias]`` (``forcing`` in
    {unsplit, lto}).  NOTE: unlike the rest of this module (which reads committed run
    tables) this re-simulates every anchor basin under both forcings — a few minutes.
    """
    from ..metrics import kge, nse, pbias
    from .compare import ANCHOR_SETS, _screened_fp, build_anchor_long

    fp = _screened_fp(data_dir, ANCHOR_SETS)
    base = build_anchor_long(data_dir, ANCHOR_SETS, footprint=fp)
    lto = build_anchor_long(data_dir, ANCHOR_SETS, footprint=fp,
                            product="historical_lto")
    split_ts = pd.Timestamp(_SKILL_SPLIT)
    rows = []
    for (dom, basin), g in base.groupby(["set", "basin"]):
        wide = g.pivot_table(index="date", columns="source", values="flow_taf")
        if "calsim3" not in wide:
            continue
        run_l = lto[(lto["set"] == dom) & (lto["basin"] == basin)
                    & (lto["source"] == dom)]
        wide = (wide.rename(columns={dom: "unsplit"})
                .join(run_l.set_index("date")["flow_taf"].rename("lto")))
        cols = ["unsplit", "lto", "calsim3"] + (["vic"] if "vic" in wide else [])
        common = wide[cols].dropna()
        for plab, sub in (("full", common),
                          ("pre1950", common[common.index < split_ts]),
                          ("post1950", common[common.index >= split_ts])):
            if len(sub) < 12:
                continue
            ref = sub["calsim3"].to_numpy()
            for frc in ("unsplit", "lto"):
                sim = sub[frc].to_numpy()
                rows.append({"set": dom, "basin": basin, "period": plab,
                             "forcing": frc, "n_months": len(sub),
                             "kge": kge(sim, ref), "nse": nse(sim, ref),
                             "pbias": pbias(sim, ref)})
    return pd.DataFrame(rows)


def skill_boxplot_fig(met: pd.DataFrame, path: Path) -> None:
    """Boxplots of the per-basin anchor skill (KGE / NSE / pbias vs CalSim3), baseline
    vs split product, per period, all anchor basins pooled and overlaid as jittered
    points.  House axis rules: KGE/NSE on the full 0-1 scale, pbias symmetric about 0."""
    import numpy as np

    cfg = PRODUCTS["split_unsplit"]
    periods = [("full", "Full"), ("pre1950", "Pre-1950"), ("post1950", "Post-1950")]
    forcings = [("unsplit", f"{cfg['base']} (Livneh baseline)", "0.55"),
                ("lto", f"{cfg['label']} ({cfg['product']})", cfg["c1"])]
    metrics = [("kge", "KGE"), ("nse", "NSE"), ("pbias", "pbias (%)")]
    rng = np.random.default_rng(7)
    plim = float(np.ceil(met["pbias"].abs().max() / 5.0) * 5.0)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 3, figsize=(_FIG_W, 2.9))
        for ax, (m, mlab) in zip(axes, metrics, strict=True):
            for gi, (plab, _) in enumerate(periods):
                for fi, (frc, _, color) in enumerate(forcings):
                    v = met[(met["period"] == plab)
                            & (met["forcing"] == frc)][m].to_numpy()
                    x = gi + (fi - 0.5) * 0.38
                    ax.boxplot(v, positions=[x], widths=0.30, patch_artist=True,
                               showfliers=False,
                               medianprops=dict(color="0.1", lw=1.1),
                               boxprops=dict(facecolor=color, edgecolor="0.25",
                                             lw=0.6, alpha=0.55),
                               whiskerprops=dict(color="0.35", lw=0.7),
                               capprops=dict(color="0.35", lw=0.7))
                    ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=4, zorder=3,
                               facecolor=color, edgecolor="0.2", linewidth=0.25)
            ax.set_xticks(range(len(periods)))
            ax.set_xticklabels([p[1] for p in periods])
            ax.set_title(mlab)
            ax.grid(axis="y", color="0.92", lw=0.5)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.6, len(periods) - 0.4)
            if m == "pbias":
                ax.set_ylim(-plim, plim)
                ax.axhline(0, color="0.4", lw=0.7)
            else:
                ax.set_ylim(0, 1)          # house rule: KGE/NSE on the full 0-1 axis
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="0.25", lw=0.6,
                                 alpha=0.55) for _, _, c in forcings]
        fig.legend(handles, [lab for _, lab, _ in forcings], loc="lower center",
                   ncol=2, frameon=False)
        fig.suptitle("SAC-SMA vs CalSim3 unimpaired FNF — forcing-product effect on "
                     "the screened-anchor skill (11obs + 9unimp basins)")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)


def make_split_unsplit_skill(data_dir: str | Path = "data",
                             artifacts_dir: str | Path = "artifacts") -> pd.DataFrame:
    """The split-product skill artifact: ``split_unsplit_anchor_skill.csv`` +
    ``figures/split_unsplit_skill_boxplot.png`` (see
    :func:`split_unsplit_skill_table`)."""
    out = Path(artifacts_dir) / "calsim" / "forcing_compare"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    met = split_unsplit_skill_table(data_dir)
    met.to_csv(out / "split_unsplit_anchor_skill.csv", index=False)
    skill_boxplot_fig(met, out / "figures" / "split_unsplit_skill_boxplot.png")
    med = (met[met["period"] == "full"]
           .groupby("forcing")[["kge", "nse", "pbias"]].median())
    print("split_unsplit anchor skill vs CalSim3 (screened): full-period medians\n"
          f"{med.round(3).to_string()}\n-> {out}")
    return met


def make_all(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts") -> Path:
    """Forcing-comparison figures -> ``artifacts/calsim/forcing_compare/figures/``."""
    calsim_art = Path(artifacts_dir) / "calsim"
    figdir = calsim_art / "forcing_compare" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    orders = {d: basin_order_north_south(data_dir, d) for d in SETS}
    all_series = {}
    for prefix, cfg in PRODUCTS.items():
        series = all_series[prefix] = {d: _series_for(data_dir, calsim_art, cfg, d)
                                       for d in SETS}
        volume_by_period_fig(series, orders, cfg,
                             figdir / f"{prefix}_volume_by_period.png")
        annual_diff_fig(series, orders, cfg,
                        figdir / f"{prefix}_annual_diff.png")
        for domain in SETS:
            regime_domain_fig(series[domain], orders[domain], cfg, domain,
                              figdir / f"{prefix}_regime_by_period_{domain}.png")
        print(f"wrote {prefix}_* figures -> {figdir}")
    # cross-model aggregate: SAC-SMA vs VIC per climate product
    for sac_key, vic_key in (("split_unsplit", "vic_split_unsplit"), ("wgen", "vic_wgen")):
        pairs = {"SAC-SMA": _aggregate(all_series[sac_key], orders, data_dir),
                 "VIC": _aggregate(all_series[vic_key], orders, data_dir)}
        agg_models_fig(pairs, PRODUCTS[sac_key],
                       figdir / f"agg_{sac_key}_annual_diff.png")
        print(f"wrote agg_{sac_key}_annual_diff.png -> {figdir}")
    # skill effect of the split product: both forcings re-simulated through the
    # screened anchor and scored vs CalSim3 (per-basin CSV + boxplot; a few minutes —
    # unlike the figures above, this does not read committed run tables)
    make_split_unsplit_skill(data_dir, artifacts_dir)
    return figdir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sacsma.calsim.forcing_compare",
                                 description="Forcing-product volume/regime comparison figures")
    ap.add_argument("--data-dir", default="data", help="data store")
    ap.add_argument("--artifacts-dir", default="artifacts", help="output root")
    args = ap.parse_args(argv)
    make_all(data_dir=args.data_dir, artifacts_dir=args.artifacts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
