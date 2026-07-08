"""Forcing-product flow comparisons vs the Livneh-unsplit baseline run.

Two separate figure sets, one per alternate forcing product:

* ``historical_lto`` — the split-precipitation LTO climate, labelled simply
  **Split** vs the **Unsplit** baseline in the figures. The split-vs-unsplit
  difference is concentrated **before 1950** (median |volume difference| ~11%
  in 1915-1949 vs ~2-3% in 1950-2018), so every figure separates the two
  periods. The volume-bar and annual-difference figures show the CalLite
  calibration sets 9unimp + 11obs only.
* ``wgen_product_a`` — detrended temperature, identical precipitation. The
  effect also tapers in time (detrending warms the early record most).

Inputs are the committed run tables: the parity-exact ``simflow`` reference
(the Livneh-unsplit run) and ``artifacts/calsim/<product>/flow_daily_<domain>.csv``
(regenerate with ``sacsma run ALL --domain <d> --forcing <product>``).

Outputs -> ``artifacts/calsim/forcing_compare/figures/``, per product prefix
(``split_unsplit_``, ``wgen_``):

* ``<p>_volume_by_period.png`` — % volume difference per watershed (the
  split/unsplit figures report **Unsplit − Split**; WGEN reports run − base),
  1915-1949 and 1950-2018 bars, basins north->south per domain.
* ``<p>_annual_diff.png``      — 5-yr rolling water-year volume difference per
  basin, same sign convention; the split/unsplit figure colours the lines per
  watershed with a short cross-basin mean-|difference| panel below, WGEN keeps
  the anonymous grey fan with the median overlaid; 1950 marked.
* regime figures — mean-monthly regime, baseline vs product, both periods:
  ``split_unsplit_regime_by_period_<domain>.png`` (every basin of 9unimp /
  11obs, one panel each) and ``wgen_regime_by_period.png`` (one row per
  period, three example watersheds).

Usage::

    python -m sacsma.calsim.forcing_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .._figures import plt  # Agg backend
from ..io import load_reference
from . import DOMAINS
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
    ("1915–1949", "1915-01-01", "1949-12-31"),
    ("1950–2018", "1950-01-01", "2018-12-31"),
)

#: per-product figure configuration.  ``title`` prefixes the figure titles; ``label``/
#: ``base`` name the product/baseline lines; ``vs_run``/``xlabel``/``bar_body``/
#: ``annual_body`` phrase the reference run and the volume-bar / annual-diff titles;
#: ``domains`` = the calibration sets in the volume-bar + annual-diff figures;
#: ``by_basin`` colours the annual-diff lines per watershed (with a legend) and adds a
#: short cross-basin mean-|difference| panel, instead of the anonymous grey fan with a
#: median overlay; ``median_signed`` = the grey-fan median statistic (signed for WGEN —
#: every watershed is negative, so a |difference| median would plot positive);
#: ``legend_loc`` = where the volume-bar legend has empty space; ``regime_per_domain``
#: swaps the three-example-watershed regime figure for one all-basin figure per domain;
#: ``sign``/``diff_phrase`` set the difference convention (split/unsplit figures report
#: **Unsplit − Split**, so positive = the Unsplit run is larger; WGEN reports run − base).
PRODUCTS = {
    "historical_lto": dict(
        prefix="split_unsplit", title="Split precipitation", label="Split", base="Unsplit",
        vs_run="the Unsplit run", xlabel="Unsplit − Split SAC-SMA runoff volume (%)",
        sign=-1.0,
        bar_body="SAC-SMA runoff volume difference by period (basins north→south)",
        annual_body="SAC-SMA water-year runoff volume difference (Unsplit − Split) "
                    "(5-yr rolling, one line per 9unimp/11obs watershed)",
        c1="#d95f0e", c2="#fdbe85", domains=("9unimp", "11obs"),
        by_basin=True, median_signed=False, legend_loc="upper right", regime_per_domain=True,
    ),
    "wgen_product_a": dict(
        prefix="wgen", title="WGEN Product A (detrended temperature)",
        label="WGEN Product A (detrended temperature)", base="Livneh unsplit (baseline)",
        vs_run="the baseline run", xlabel="volume difference vs Livneh-unsplit run (%)",
        sign=1.0,
        bar_body="volume difference by period (basins north→south)",
        annual_body="water-year volume difference vs the baseline run "
                    "(5-yr rolling, one line per watershed)",
        c1="#2c7fb8", c2="#a6cbe3", domains=DOMAINS,
        by_basin=False, median_signed=True, legend_loc="upper left", regime_per_domain=False,
    ),
}
_BASE_COLOR = "#111111"

#: (domain, basin) regime panels for the subset-style regime figure
#: (``regime_per_domain=False``): chosen for snowmelt timing vs foothill volume.
REGIME_BASINS = {
    "wgen_product_a": (("11obs", "AMF"), ("11obs", "SJF"), ("9unimp", "CosumnesRiver")),
}


def _product_daily(calsim_art: Path, product: str, domain: str) -> pd.DataFrame:
    return pd.read_csv(calsim_art / product / f"flow_daily_{domain}.csv", parse_dates=["date"])


def _pct_diff_by_period(base: pd.DataFrame, run: pd.DataFrame,
                        sign: float = 1.0) -> list[float]:
    """% volume difference for each analysis period, in % of the base run:
    run − base (``sign=+1``) or base − run (``sign=-1``)."""
    out = []
    for _, a, b in PERIODS:
        bm = base[(base["date"] >= a) & (base["date"] <= b)]["flow"].mean()
        rm = run[(run["date"] >= a) & (run["date"] <= b)]["flow"].mean()
        out.append(sign * 100.0 * (rm / bm - 1.0))
    return out


def volume_by_period_fig(data_dir, calsim_art: Path, product: str, path: Path) -> None:
    """Horizontal bars: % volume difference vs baseline, one bar per period."""
    cfg = PRODUCTS[product]
    orders = {d: basin_order_north_south(data_dir, d) for d in cfg["domains"]}
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            len(orders), 1, figsize=(_FIG_W, 0.225 * sum(map(len, orders.values()))),
            sharex=True, gridspec_kw={"height_ratios": [len(o) for o in orders.values()]},
        )
        for ax, (domain, order) in zip(axes, orders.items()):
            run = _product_daily(calsim_art, product, domain)
            vals = {b: _pct_diff_by_period(load_reference(data_dir, basin=b, domain=domain),
                                           run[run["basin"] == b], cfg["sign"]) for b in order}
            y = range(len(order))
            for k, ((plabel, _, _), color) in enumerate(zip(PERIODS, (cfg["c1"], cfg["c2"]))):
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
        axes[-1].set_xlabel(cfg["xlabel"])
        fig.suptitle(f"{cfg['title']}: {cfg['bar_body']}")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)


def _wy_annual(df: pd.DataFrame) -> pd.Series:
    """Water-year total flow (mm) indexed by WY end year, complete WYs only."""
    wy = df["date"].dt.year + (df["date"].dt.month >= 10).astype(int)
    ann = df.groupby(wy)["flow"].sum()
    return ann.loc[(ann.index >= 1916) & (ann.index <= 2018)]


def annual_diff_fig(data_dir, calsim_art: Path, product: str, path: Path) -> None:
    """5-yr rolling water-year volume difference per basin.  ``by_basin`` products
    get one coloured line per watershed plus a short cross-basin mean-|difference|
    panel below; otherwise a grey fan with the cross-basin median overlaid (signed
    or |difference| per the product's ``median_signed``)."""
    cfg = PRODUCTS[product]
    labels, diffs = [], []
    for domain in cfg["domains"]:
        run = _product_daily(calsim_art, product, domain)
        for b in basin_order_north_south(data_dir, domain):
            base = _wy_annual(load_reference(data_dir, basin=b, domain=domain))
            r = _wy_annual(run[run["basin"] == b])
            labels.append(_BASIN_ABBREV.get(b, b))
            diffs.append((cfg["sign"] * 100.0 * (r / base - 1.0))
                         .rolling(5, center=True).mean())
    d = pd.concat(diffs, axis=1)
    with plt.rc_context(_RC):
        if cfg["by_basin"]:
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
            axes = (ax, ax2)
        else:
            med = d.median(axis=1) if cfg["median_signed"] else d.abs().median(axis=1)
            stat = "median difference" if cfg["median_signed"] else "median |difference|"
            fig, ax = plt.subplots(figsize=(_FIG_W, 2.2))
            ax.plot(d.index, d.to_numpy(), color="0.75", lw=0.4)
            ax.plot(d.index, med, color=cfg["c1"], lw=1.4,
                    label=f"{stat}, {d.shape[1]} watersheds")
            ax.legend(loc="upper right", frameon=False)
            axes = (ax,)
        ax.axhline(0, color="0.4", lw=0.7)
        for a in axes:
            a.axvline(1950, color="0.2", lw=0.8, ls=":")
            a.set_xlim(1916, 2018)
            a.grid(color="0.92", lw=0.5)
            a.set_axisbelow(True)
        ax.text(1951, 0.93, "1950", fontsize=5.5, color="0.2",
                transform=ax.get_xaxis_transform())
        axes[-1].set_xlabel("water year")
        ax.set_ylabel("volume difference (%)")
        ax.set_title(f"{cfg['title']}: {cfg['annual_body']}")
        fig.tight_layout(rect=(0, 0.14, 1, 1) if cfg["by_basin"] else None)
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)


def _mean_monthly(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Water-year-ordered (O..S) mean monthly flow (mm/month) over [a, b]."""
    sub = df[(df["date"] >= a) & (df["date"] <= b)]
    m = (sub.assign(mo=sub["date"].dt.to_period("M")).groupby("mo")["flow"].sum())
    clim = m.groupby(m.index.month).mean()
    return clim.reindex([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def regime_by_period_fig(data_dir, calsim_art: Path, product: str, path: Path) -> None:
    """Mean-monthly regime, baseline vs product: rows = periods, cols = basins."""
    cfg = PRODUCTS[product]
    basins = REGIME_BASINS[product]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(len(PERIODS), len(basins), figsize=(_FIG_W, 3.8),
                                 sharex=True)
        for i, (plabel, a, b) in enumerate(PERIODS):
            for j, (domain, basin) in enumerate(basins):
                ax = axes[i, j]
                base = load_reference(data_dir, basin=basin, domain=domain)
                run = _product_daily(calsim_art, product, domain)
                ax.plot(range(12), _mean_monthly(base, a, b), color=_BASE_COLOR,
                        lw=1.4, label=cfg["base"])
                ax.plot(range(12), _mean_monthly(run[run["basin"] == basin], a, b),
                        color=cfg["c1"], lw=1.1, label=cfg["label"])
                if i == 0:
                    ax.set_title(f"{basin} ({domain})")
                ax.set_xticks(range(12), list("ONDJFMAMJJAS"))
                ax.grid(color="0.92", lw=0.5)
                ax.set_axisbelow(True)
            axes[i, 0].set_ylabel(f"{plabel}\nmm/month")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="upper center",
                   bbox_to_anchor=(0.5, 0.955), frameon=False)
        fig.suptitle(f"{cfg['title']}: mean-monthly regime by period, vs {cfg['vs_run']}")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)


def regime_domain_fig(data_dir, calsim_art: Path, product: str, domain: str,
                      path: Path) -> None:
    """Mean-monthly regime, baseline vs product, both periods: one panel per basin
    of ``domain`` (north -> south); solid = baseline, dashed = product, dark =
    1915-1949, light = 1950-2018.  Each panel is annotated top-right with the
    per-period volume difference (``sign`` convention, period-coloured)."""
    cfg = PRODUCTS[product]
    order = basin_order_north_south(data_dir, domain)
    run = _product_daily(calsim_art, product, domain)
    ncol = 4 if len(order) > 9 else 3
    nrow = -(-len(order) // ncol)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(_FIG_W, 1.25 * nrow + 1.0),
                                 sharex=True)
        for ax in axes.flat[len(order):]:
            ax.set_visible(False)
        for ax, basin in zip(axes.flat, order, strict=False):  # trailing axes hidden
            base = load_reference(data_dir, basin=basin, domain=domain)
            r = run[run["basin"] == basin]
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
                   bbox_to_anchor=(0.5, 0.965), frameon=False)
        fig.suptitle(f"{cfg['title']}: SAC-SMA mean-monthly runoff regime — {domain} "
                     f"(north→south; % = {cfg['base']} − {cfg['label']} volume)")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=_DPI)
        plt.close(fig)


def make_all(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts") -> Path:
    """Forcing-comparison figures -> ``artifacts/calsim/forcing_compare/figures/``."""
    calsim_art = Path(artifacts_dir) / "calsim"
    figdir = calsim_art / "forcing_compare" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for product, cfg in PRODUCTS.items():
        prefix = cfg["prefix"]
        volume_by_period_fig(data_dir, calsim_art, product,
                             figdir / f"{prefix}_volume_by_period.png")
        annual_diff_fig(data_dir, calsim_art, product,
                        figdir / f"{prefix}_annual_diff.png")
        if cfg["regime_per_domain"]:
            for domain in cfg["domains"]:
                regime_domain_fig(data_dir, calsim_art, product, domain,
                                  figdir / f"{prefix}_regime_by_period_{domain}.png")
        else:
            regime_by_period_fig(data_dir, calsim_art, product,
                                 figdir / f"{prefix}_regime_by_period.png")
        print(f"wrote {prefix}_* figures -> {figdir}")
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
