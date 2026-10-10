"""The benchmark at the CDEC full-natural-flow sites: three models against observed CDEC FNF,
monthly, WY1991-2018 (``sacsma benchmark``).

The models, all on WGEN Product A scenario 1:

==================  =======================================================================
``dpl``             dPL-CalSim, its trained field on the 12 ``cdec_*`` entities (:mod:`.flows`)
``bcm``             BCM v8 Scenario 1, ``run`` and ``rch`` routed by the USGS monthly
                    equations, fitted per site on WY1991-2018 (:func:`.gridded.bcm_fit`)
``vic_calsim3``     the CalSim3 pipeline's routed VIC at the site's CalSim3 node
                    (:mod:`.flows`)
==================  =======================================================================

Scores (:func:`metrics`): per site, every model present there is scored on the same months,
the months of the window where the observation and every one of those models have a value.  A
month of observations is CDEC's monthly full natural flow (:func:`.flows.observed_monthly`).
:func:`summary` pools the scores over the sites where all three models are scored.

:func:`make_all` writes the tables and figures to ``artifacts/results/benchmark/``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from . import flows, gridded

#: the three models, in table and plot order
MODELS = ("dpl", "bcm", "vic_calsim3")
#: the observation every model is scored against
REF = "obs"
LABELS = {"obs": "CDEC FNF", "dpl": "dPL-CalSim", "bcm": "BCM",
          "vic_calsim3": "VIC-CalSim3"}
#: categorical colours in :data:`MODELS` order (checked for colour-blind separation of
#: neighbours); the observation is black
COLORS = {"obs": "#111111", "dpl": "#2c7fb8", "bcm": "#1b9e77",
          "vic_calsim3": "#984ea3"}
#: the marker of each model, so the skill figure reads without colour (the triangles are kept
#: for values pinned to an axis bound)
MARKERS = {"dpl": "P", "bcm": "s", "vic_calsim3": "D"}
MARKER_SIZE = {"dpl": 30, "bcm": 22, "vic_calsim3": 18}
#: the forcing of the benchmark and the BCM scenario on the same climate
FORCING = flows.FORCING
BCM_SCENARIO = gridded.BCM_SCENARIO[FORCING]
#: the scoring window (water years)
WY = flows.WY
#: the result folder (``paths.calibrated``)
RESULT = "benchmark"

_W, _DPI = 6.5, 300            # house style: at most 6.5 in wide, 300 dpi, 8 pt text
_PBIAS_LIM = 75.0              # the shared percent-bias axis
_WY_MONTHS = (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)


# ------------------------------------------------------------------------------ flows
def monthly(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts", *,
            wy=WY, log=print, bcm_depth=None, bcm_params=None) -> pd.DataFrame:
    """``[date, site, model, flow_taf]``: the observation and the three models, every month end
    of the window x site x model, NaN where there is no value.  ``site`` and ``model`` are
    ordered categoricals (north to south; ``obs`` then :data:`MODELS`)."""
    months = flows._water_months(wy)
    base = flows.monthly_flows(FORCING, data_dir, artifacts_dir, wy=wy,
                               models=("obs", "dpl", "vic_calsim3"), log=log)
    t = time.perf_counter()
    bcm = flows._tidy(gridded.bcm_routed(data_dir, BCM_SCENARIO, bcm_depth, bcm_params), "bcm",
                      months)
    if log:
        log(f"benchmark flows: bcm ({time.perf_counter() - t:.1f} s)")
    out = pd.concat([base.astype({"site": str, "model": str}), bcm], ignore_index=True)
    out["site"] = pd.Categorical(out["site"], categories=list(flows.SITES), ordered=True)
    out["model"] = pd.Categorical(out["model"], categories=[REF, *MODELS], ordered=True)
    return out.sort_values(["model", "site", "date"]).reset_index(drop=True)


def _wide(long: pd.DataFrame, site: str) -> pd.DataFrame:
    """One site's flows, month end x model (TAF)."""
    g = long[long["site"] == site]
    return g.pivot_table(index="date", columns="model", values="flow_taf", observed=False,
                         dropna=False)


def _present(wide: pd.DataFrame) -> list[str]:
    """The models with a series at a site."""
    return [m for m in MODELS if m in wide and wide[m].notna().any()]


def _scored(wide: pd.DataFrame) -> pd.DataFrame:
    """The months a site is scored on: the observation and every model present all have a
    value."""
    return wide[[REF, *_present(wide)]].dropna()


def _wy(d: pd.Timestamp) -> int:
    return d.year + (d.month >= 10)


# ------------------------------------------------------------------------------ scores
def _skill(dates, sim, obs) -> dict:
    from ..metrics import kge, nse, pbias, pearson, seasonal_mismatch

    return dict(kge=kge(sim, obs), nse=nse(sim, obs), pbias=pbias(sim, obs),
                r=pearson(sim, obs), seas_mismatch=seasonal_mismatch(dates, sim, obs))


def metrics(long: pd.DataFrame) -> pd.DataFrame:
    """Per site and model present: ``[site, model, n_months, wy_start, wy_end, kge, nse,
    pbias, r, seas_mismatch, mean_sim_taf, mean_obs_taf]`` on the site's scored months
    (shared by every model at the site).  ``pbias`` is percent, positive when the model is
    high; ``seas_mismatch`` is the fraction of the annual flow in the wrong month
    (:func:`sacsma.metrics.seasonal_mismatch`)."""
    rows = []
    for site in flows.SITES:
        sc = _scored(_wide(long, site))
        if len(sc) < 12:
            continue
        obs = sc[REF].to_numpy(float)
        for m in [c for c in MODELS if c in sc]:
            sim = sc[m].to_numpy(float)
            rows.append(dict(site=site, model=m, n_months=len(sc), wy_start=_wy(sc.index[0]),
                             wy_end=_wy(sc.index[-1]), **_skill(sc.index, sim, obs),
                             mean_sim_taf=float(sim.mean()), mean_obs_taf=float(obs.mean())))
    return pd.DataFrame(rows)


def common_sites(met: pd.DataFrame) -> list[str]:
    """The sites where all three models are scored, north to south."""
    have = met.groupby("site")["model"].apply(set)
    return [s for s in flows.SITES if s in have and have[s] >= set(MODELS)]


def summary(met: pd.DataFrame) -> pd.DataFrame:
    """Each model pooled over the sites where all three models are scored
    (:func:`common_sites`, the same sites for every model): median and mean KGE, median NSE,
    percent bias (median, mean absolute, and of the summed volume), median r and seasonal
    mismatch, and the number of sites where the model has the highest KGE."""
    met = met[met["site"].isin(common_sites(met))]
    best = met.loc[met.groupby("site")["kge"].idxmax(), "model"].value_counts()
    rows = []
    for m in MODELS:
        g = met[met["model"] == m]
        rows.append(dict(
            model=m, n_sites=len(g), sites=";".join(g["site"]),
            median_kge=g["kge"].median(), mean_kge=g["kge"].mean(),
            median_nse=g["nse"].median(), median_pbias=g["pbias"].median(),
            mean_abs_pbias=g["pbias"].abs().mean(), median_r=g["r"].median(),
            median_seas_mismatch=g["seas_mismatch"].median(),
            total_pbias=100 * (g["mean_sim_taf"].sum() / g["mean_obs_taf"].sum() - 1),
            n_kge_best=int(best.get(m, 0))))
    return pd.DataFrame(rows)


def bcm_routing_effect(long: pd.DataFrame, unrouted: pd.DataFrame,
                       workbook: pd.DataFrame) -> pd.DataFrame:
    """BCM routed by the benchmark's fit, against the USGS sheets as delivered and against
    ``run + rch`` unrouted, per site with a fit, on the site's scored months:
    ``[site, n_months, <metric>_fit, <metric>_workbook, <metric>_unrouted]`` (``_workbook`` NaN
    where the site has no sheet)."""
    rows = []
    for site in flows.SITES:
        wide = _wide(long, site)
        if "bcm" not in _present(wide):
            continue
        sc = _scored(wide)
        obs = sc[REF].to_numpy(float)
        row = dict(site=site, n_months=len(sc))
        for tag, sim in (("fit", sc["bcm"]), ("workbook", workbook[site].reindex(sc.index)),
                         ("unrouted", unrouted[site].reindex(sc.index))):
            x = sim.to_numpy(float)
            ok = np.isfinite(x)
            sk = (_skill(sc.index[ok], x[ok], obs[ok]) if ok.sum() >= 12 else
                  dict.fromkeys(("kge", "nse", "pbias", "r", "seas_mismatch"), np.nan))
            row.update({f"{k}_{tag}": v for k, v in sk.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def site_table(long: pd.DataFrame, met: pd.DataFrame, data_dir: str | Path = "data",
               bcm_params: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per site: its CDEC record, areas, footprint, the observed months of the BCM fit
    (``bcm_fit_months``, from ``bcm_params``) and the USGS sheet kept for comparison, the
    models scored and the months of observation in the window."""
    from . import bcm_routing

    st = flows.site_table(data_dir).drop(columns=["order", "entity_id"])
    fp = gridded.footprint_table(data_dir)
    have = set(gridded.bcm_sites(data_dir))
    obs = long[long["model"] == REF].dropna(subset=["flow_taf"])
    rows = []
    for _, r in st.iterrows():
        s = r["site"]
        o = obs[obs["site"] == s]["date"]
        sh = bcm_routing.params_for(s, data_dir) if s in have else None
        f = fp[fp["site"] == s].iloc[0]
        rows.append(dict(
            site=s, cdec_id=r["cdec_id"], name=r["name"], area_mi2=r["area_mi2"],
            n_cells=int(f["n_cells"]), footprint_mi2=float(f["footprint_mi2"]),
            vic_calsim3_area_mi2=r["vic_area_mi2"],
            bcm_fit_months=(pd.NA if bcm_params is None or s not in bcm_params.index
                            else int(bcm_params.loc[s, "n_months"])),
            bcm_sheet=None if sh is None else sh["name"],
            models=";".join(met[met["site"] == s]["model"]),
            obs_months=len(o), obs_first_wy=_wy(o.min()) if len(o) else np.nan))
    return pd.DataFrame(rows).astype({"obs_first_wy": "Int64", "bcm_fit_months": "Int64"})


# ------------------------------------------------------------------------------ figures
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .. import _figures  # noqa: F401  (the house rcParams)
    return plt


def _save(fig, path: Path) -> None:
    """Save a figure at the house resolution as a 256-colour PNG (line figures lose nothing
    visible and the 17-panel hydrograph shrinks from 2.2 to 0.9 MB)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, dpi=_DPI, format="png")
    im = Image.open(buf).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                 dither=Image.Dither.NONE)
    im.save(path, optimize=True, dpi=(_DPI, _DPI))


def _clip_marks(ax, x, vals, ylim, color):
    """Pin a value outside the fixed axis to its bound with a triangle (``v`` below, ``^``
    above), so it stays visible."""
    lo, hi = ylim
    for xi, v in zip(x, vals, strict=True):
        if np.isfinite(v) and (v < lo or v > hi):
            ax.plot([xi], [lo if v < lo else hi], marker="v" if v < lo else "^", ms=4,
                    color=color, mec="white", mew=0.4, zorder=4, clip_on=False)


def _late(met: pd.DataFrame) -> dict[str, int]:
    """``{site: first water year scored}`` for the sites whose observation starts after the
    window does (BND, CSN)."""
    first = met.groupby("site")["wy_start"].first()
    return {s: int(first[s]) for s in flows.SITES if s in first and first[s] > WY[0]}


def _site_label(site: str, met: pd.DataFrame) -> str:
    """The site code, starred when its observation starts after the window does."""
    return f"{site}*" if site in _late(met) else site


def _late_note(met: pd.DataFrame) -> str:
    """The footnote of the starred sites, as a sentence ("" when every site starts with the
    window)."""
    late = _late(met)
    return ("* scored from " + ", ".join(f"WY{y} ({s})" for s, y in late.items()) + "."
            if late else "")


def skill_fig(met: pd.DataFrame, path: Path) -> None:
    """KGE (0 to 1) and percent bias (+-75 %) per site and model, as vertical dumbbells: a
    grey bar spans the models at a site, one marker per model.  Sites north to south."""
    plt = _plt()
    sites = [s for s in flows.SITES if s in set(met["site"])]
    x = np.arange(len(sites))
    fig, axes = plt.subplots(2, 1, figsize=(_W, 5.2), sharex=True)
    for ax, col, ylab, ylim in ((axes[0], "kge", "KGE", (0.0, 1.0)),
                                (axes[1], "pbias", "Percent bias", (-_PBIAS_LIM, _PBIAS_LIM))):
        vals = {m: np.array([_get(met, s, m, col) for s in sites]) for m in MODELS}
        for xi in x:
            v = [np.clip(vals[m][xi], *ylim) for m in MODELS if np.isfinite(vals[m][xi])]
            if len(v) > 1:
                ax.plot([xi, xi], [min(v), max(v)], color="0.8", lw=1.5, zorder=1)
        for m in MODELS:
            ax.scatter(x, np.clip(vals[m], *ylim), s=MARKER_SIZE[m], marker=MARKERS[m],
                       color=COLORS[m], label=LABELS[m], zorder=3, edgecolor="white",
                       linewidth=0.4)
            _clip_marks(ax, x, vals[m], ylim, COLORS[m])
        ax.set_ylim(*ylim)
        if ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlim(-0.6, len(sites) - 0.4)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", ncol=5, fontsize=7, framealpha=0.95, handletextpad=0.2,
                   columnspacing=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_site_label(s, met) for s in sites], rotation=90)
    axes[0].set_title(f"Monthly skill against CDEC full natural flow, WY{WY[0]}–{WY[1]}")
    fig.text(0.01, 0.005, f"{_late_note(met)}\nWGEN Product A scenario 1.  A value outside an "
             "axis is pinned to it with a triangle.", fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save(fig, path)
    plt.close(fig)


def _get(met, site, model, col) -> float:
    m = met[(met["site"] == site) & (met["model"] == model)]
    return float(m[col].iloc[0]) if len(m) else np.nan


def hydrograph_fig(long: pd.DataFrame, met: pd.DataFrame, path: Path,
                   titles: dict[str, str] | None = None) -> None:
    """Monthly flow over the window, one panel per site, north to south: the observation in
    black (a month with an invalid day is a gap), each model in its colour.  ``titles`` adds a
    name to each panel's site code."""
    plt = _plt()
    titles = titles or {}
    sites = list(flows.SITES)
    height = 1.05 * len(sites) + 0.75
    fig, axes = plt.subplots(len(sites), 1, figsize=(_W, height), sharex=True)
    for ax, site in zip(axes, sites, strict=True):
        wide = _wide(long, site)
        for m in _present(wide):
            ax.plot(wide.index, wide[m], color=COLORS[m], lw=0.6, alpha=0.9, label=LABELS[m])
        ax.plot(wide.index, wide[REF], color=COLORS[REF], lw=0.9, marker="o", ms=1.2,
                label=LABELS[REF], zorder=4)
        ax.set_ylim(bottom=0)
        ax.text(0.005, 0.95, f"{_site_label(site, met)}  {titles.get(site, '')}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        ax.set_ylabel("TAF/mo", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(color="0.93", lw=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.005)
    handles = [plt.Line2D([], [], color=COLORS[m], lw=1.4 if m == REF else 1.0,
                          label=LABELS[m]) for m in (REF, *MODELS)]
    fig.suptitle(f"Monthly flow (TAF/month), WY{WY[0]}–{WY[1]}.  {_late_note(met)}",
                 y=1 - 0.08 / height)
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.26 / height))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.3 / height, top=1 - 0.48 / height,
                        hspace=0.12)
    _save(fig, path)
    plt.close(fig)


#: the river of each site, for the panel titles
RIVERS = {"SHA": "Sacramento", "CLE": "Trinity", "BND": "Sacramento", "ORO": "Feather",
          "FOL": "American", "YRS": "Yuba", "CSN": "Cosumnes", "MKM": "Mokelumne",
          "NML": "Stanislaus", "TLG": "Tuolumne", "MRC": "Merced",
          "MIL": "San Joaquin"}


def climatology_fig(long: pd.DataFrame, met: pd.DataFrame, path: Path) -> None:
    """Mean flow per calendar month (October to September) at each site over its scored
    months, the observation in black and each model in its colour."""
    plt = _plt()
    sites = [s for s in flows.SITES if s in set(met["site"])]
    ncol = 4
    nrow = int(np.ceil((len(sites) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(_W, 1.45 * nrow + 0.4), squeeze=False)
    for ax, site in zip(axes.ravel(), sites, strict=False):
        sc = _scored(_wide(long, site))
        clim = sc.groupby(sc.index.month).mean().reindex(list(_WY_MONTHS))
        for m in [c for c in MODELS if c in sc]:
            ax.plot(range(12), clim[m], color=COLORS[m], lw=1.0, marker=MARKERS[m], ms=1.8)
        ax.plot(range(12), clim[REF], color=COLORS[REF], lw=1.6, zorder=4)
        ax.set_title(f"{_site_label(site, met)} ({len(sc)} months)", fontsize=7, pad=2)
        ax.set_xticks([0, 3, 6, 9])
        ax.set_xticklabels(["O", "J", "A", "J"], fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_ylim(bottom=0)
        ax.grid(color="0.93", lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[len(sites):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=COLORS[m], lw=1.6 if m == REF else 1.0,
                          marker=None if m == REF else MARKERS[m], ms=3, label=LABELS[m])
               for m in (REF, *MODELS)]
    axes.ravel()[-1].legend(handles=handles, loc="center", fontsize=7, frameon=False)
    fig.suptitle(f"Mean monthly flow (TAF/month) on the months scored, WY{WY[0]}–{WY[1]}.  "
                 f"{_late_note(met)}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)
    plt.close(fig)


# ------------------------------------------------------------------------------ the run
def make_all(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts") -> Path:
    """Build the benchmark into ``<artifacts_dir>/results/benchmark/``: ``monthly.csv`` (the
    flows of the window, TAF), ``metrics.csv`` (per site and model), ``summary.csv`` (pooled),
    ``sites.csv``, ``bcm_routing_fit.csv`` (the fitted routing parameters),
    ``bcm_routing_effect.csv`` and three figures."""
    t0 = time.perf_counter()
    out = paths.calibrated(artifacts_dir, RESULT)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    depth = gridded.bcm_depth(data_dir, BCM_SCENARIO)
    t = time.perf_counter()
    fit = gridded.bcm_fit(data_dir, artifacts_dir, BCM_SCENARIO, depth)
    print(f"benchmark BCM fit: {len(fit)} sites ({time.perf_counter() - t:.0f} s)")
    long = monthly(data_dir, artifacts_dir, bcm_depth=depth, bcm_params=fit)
    met = metrics(long)
    summ = summary(met)
    effect = bcm_routing_effect(long, gridded.bcm_unrouted(data_dir, BCM_SCENARIO, depth),
                                gridded.bcm_workbook(data_dir, BCM_SCENARIO, depth))
    sites = site_table(long, met, data_dir, fit)

    long.dropna(subset=["flow_taf"]).assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")) \
        .to_csv(out / "monthly.csv", index=False, float_format="%.4f")
    for df, name in ((met, "metrics.csv"), (summ, "summary.csv"), (sites, "sites.csv"),
                     (effect, "bcm_routing_effect.csv")):
        df.to_csv(out / name, index=False)
    fit.reset_index().to_csv(out / "bcm_routing_fit.csv", index=False)
    skill_fig(met, figs / "skill.png")
    hydrograph_fig(long, met, figs / "hydrographs.png",
                   {s: f"{RIVERS.get(s, '')} (CDEC {c})" for s, c in
                    zip(sites["site"], sites["cdec_id"], strict=True)})
    climatology_fig(long, met, figs / "climatology.png")

    print(f"benchmark: {met['site'].nunique()} sites, {len(MODELS)} models, WY{WY[0]}-{WY[1]} "
          f"-> {out} ({time.perf_counter() - t0:.0f} s)")
    cols = ["model", "n_sites", "median_kge", "median_nse", "median_pbias", "mean_abs_pbias",
            "n_kge_best"]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return out
