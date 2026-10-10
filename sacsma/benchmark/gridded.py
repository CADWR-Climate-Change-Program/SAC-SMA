"""The gridded reference model of the benchmark at the CDEC sites: BCM v8 routed by its monthly
post-processing.

BCM is a 1/16-degree region store (``data/reference/bcm/bcm_s01_monthly.nc``) averaged over one
footprint per site (:func:`footprints`): the cells of the site's ``cdec_<site>`` entity of the
multifamily registry, weighted by ``overlap_mi2`` (the dPL convention).  The registry footprints
of SHA and BND stop south of the endorheic Goose Lake block, so nothing is screened.  A cell with
no value in a month is left out and the weights are renormalised over the others (BCM masks open
water).

* **BCM** (:func:`bcm_routed`).  The footprint means of ``run`` and ``rch`` (Scenario 1, the
  WGEN Product A climate), turned into volume with the site's registry area, go through the
  USGS routing equations (:mod:`.bcm_routing`) with parameters fitted here, per site, on this
  same Scenario 1 input against the observed monthly flow of :data:`FIT_WY`, the whole scoring
  window, the years dPL-CalSim was trained on (:func:`bcm_fit`: the sheets' volume tolerance,
  KGE in place of their NSE).  The
  routing runs from the first month of the store (1915-10), so the stores are spun up long
  before the scoring window.  A site with fewer than :data:`MIN_FIT_MONTHS` observed months in
  the window gets no series.
  The USGS sheets themselves, fitted on PRISM-forced BCM, are kept for comparison
  (:func:`bcm_workbook`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from . import bcm_routing
from .flows import SITES, TAF_PER_MM_MI2, WY, _registry, entity

#: the BCM scenario of each forcing (Scenario 1 is the WGEN Product A climate)
BCM_SCENARIO = {"wgen_product_a": "s01"}
#: cubic metres in one thousand acre-feet
M3_PER_TAF = 1233.48183754752 * 1000.0
#: the water years the benchmark's BCM routing is fitted on: the whole scoring window, as
#: dPL-CalSim was trained on it
FIT_WY = WY
#: the fewest observed months in :data:`FIT_WY` a site needs for a fit (five years)
MIN_FIT_MONTHS = 60
#: the objective of the fit: KGE, the benchmark's headline score, in place of the sheets' NSE,
#: whose optimum damps the monthly variance
FIT_OBJECTIVE = "kge"


# --------------------------------------------------------------------------- footprints
def footprints(data_dir: str | Path = "data") -> pd.DataFrame:
    """``[site, key, overlap_mi2, weight]``: each site's registry cells, ``weight`` the overlap
    normalised to 1 per site."""
    cells = pd.read_csv(paths.entity_cells(data_dir))
    parts = []
    for site in SITES:
        c = cells[cells["entity_id"] == entity(site)][["key", "overlap_mi2"]]
        if c.empty:
            raise ValueError(f"{entity(site)} has no cells in {paths.entity_cells(data_dir)}")
        c = c.groupby("key", as_index=False, sort=False)["overlap_mi2"].sum()
        parts.append(c.assign(site=site, weight=c["overlap_mi2"] / c["overlap_mi2"].sum()))
    return pd.concat(parts, ignore_index=True)[["site", "key", "overlap_mi2", "weight"]]


def footprint_table(data_dir: str | Path = "data") -> pd.DataFrame:
    """``[site, n_cells, footprint_mi2]``: the size of each footprint."""
    fp = footprints(data_dir)
    return (fp.groupby("site", sort=False)
              .agg(n_cells=("key", "size"), footprint_mi2=("overlap_mi2", "sum"))
              .reindex(list(SITES)).reset_index())


def _site_means(store: Path, variables, data_dir) -> dict[str, pd.DataFrame]:
    """Footprint means of ``variables`` of a region store: ``{var: month end x site}`` in the
    store's units, the weights renormalised over the cells with a value each month."""
    import xarray as xr

    fp = footprints(data_dir)
    keys = list(dict.fromkeys(fp["key"]))
    W = np.zeros((len(SITES), len(keys)))
    col = {k: j for j, k in enumerate(keys)}
    for i, site in enumerate(SITES):
        f = fp[fp["site"] == site]
        W[i, f["key"].map(col).to_numpy()] = f["weight"].to_numpy(float)
    with xr.open_dataset(store) as ds:
        absent = sorted(set(keys) - set(ds["key"].values.tolist()))
        if absent:
            raise ValueError(f"{store.name}: {len(absent)} footprint cells are not in the store")
        sub = ds.sel(key=keys)
        months = pd.DatetimeIndex(sub["time"].values) + pd.offsets.MonthEnd(0)
        out = {}
        for v in variables:
            x = sub[v].values.astype(float)                         # (key, month)
            ok = np.isfinite(x)
            num = W @ np.where(ok, x, 0.0)
            den = W @ ok
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(den > 0, num / den, np.nan)
            out[v] = pd.DataFrame(mean.T, index=months.rename("date"), columns=list(SITES))
    return out


# ---------------------------------------------------------------------------------- BCM
def bcm_sites(data_dir: str | Path = "data") -> list[str]:
    """The sites with a USGS routing sheet (or its refit), north to south."""
    have = set(bcm_routing.load_params(data_dir)["cdec_site"])
    return [s for s in SITES if s in have]


def bcm_depth(data_dir: str | Path = "data", scenario: str = "s01") -> dict[str, pd.DataFrame]:
    """BCM ``run`` and ``rch`` on each site's footprint: ``{"run", "rch"}`` -> month end x
    site, mm per month, 1915-10 to 2018-09."""
    return _site_means(paths.bcm(data_dir, f"bcm_{scenario}_monthly.nc"), ("run", "rch"),
                       data_dir)


def _volumes(depth, site, area_mi2) -> tuple[pd.Series, pd.Series]:
    run, rch = depth["run"][site], depth["rch"][site]
    if run.isna().any() or rch.isna().any():
        raise ValueError(f"BCM at {site}: a month with no valid cell")
    a = area_mi2 * 2589988.110336
    return bcm_routing.to_m3(run, a), bcm_routing.to_m3(rch, a)


def _fit_site(task):
    """One site's fit (module level, so the process pool can send it)."""
    site, run, rch, obs, window = task
    return site, bcm_routing.fit(run, rch, obs, window=window, objective=FIT_OBJECTIVE)


def bcm_fit(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts",
            scenario: str = "s01", depth: dict[str, pd.DataFrame] | None = None,
            obs: pd.DataFrame | None = None, log=print) -> pd.DataFrame:
    """The benchmark's BCM routing parameters, one row per site, indexed by ``site``.

    For each site, :func:`sacsma.benchmark.bcm_routing.fit` (objective :data:`FIT_OBJECTIVE`,
    ``|PBIAS|`` within 1 %) on the Scenario 1 ``run`` and
    ``rch`` of its footprint times its registry area, routed over the whole record from
    1915-10, against the observed monthly volume (``obs``, TAF, month end x site; default
    :func:`.flows.observed_monthly`) over the months of :data:`FIT_WY` that have one.  The
    sites with fewer than :data:`MIN_FIT_MONTHS` such months are left out.  Columns: the keys
    :func:`~sacsma.benchmark.bcm_routing.route` reads, ``area_m2``, ``n_months`` and the fit's
    ``r2, nse, pbias`` (the sheets' statistics) and ``kge``.  The fits run in parallel
    processes and are cached under ``cache/benchmark/``, keyed by the inputs and the code."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    from ..dpl.evaluate import content_key
    from . import flows
    from .flows import cache_dir, observed_monthly

    key = content_key(paths.bcm(data_dir, f"bcm_{scenario}_monthly.nc"),
                      paths.cdec_fnf(data_dir, "fnf_monthly.csv"), paths.gage_15cdec(data_dir),
                      paths.cdec_fnf(data_dir, "fnf_daily_mm.csv"),
                      paths.cdec_fnf(data_dir, "fnf_daily_mask.csv"), paths.entities(data_dir),
                      paths.entity_cells(data_dir), Path(bcm_routing.__file__), Path(__file__),
                      Path(flows.__file__))
    cache = cache_dir(artifacts_dir) / f"bcm_fit_{scenario}_{key}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col="site", float_precision="round_trip")
    depth = bcm_depth(data_dir, scenario) if depth is None else depth
    obs = observed_monthly(data_dir) if obs is None else obs
    area = _registry(data_dir)["area_mi2"]
    months = depth["run"].index
    in_wy = (months >= pd.Timestamp(f"{FIT_WY[0] - 1}-10-01")) & \
            (months <= pd.Timestamp(f"{FIT_WY[1]}-09-30"))
    tasks, skipped = [], {}
    for site in SITES:
        o = obs[site].reindex(months) * M3_PER_TAF
        window = in_wy & o.notna().to_numpy()
        if window.sum() < MIN_FIT_MONTHS:
            skipped[site] = int(window.sum())
            continue
        run, rch = _volumes(depth, site, float(area[site]))
        tasks.append((site, run.to_numpy(), rch.to_numpy(), o.to_numpy(), window))
    if log and skipped:
        log(f"benchmark BCM fit: no fit for {skipped} (observed months in WY{FIT_WY[0]}-"
            f"{FIT_WY[1]}, fewer than {MIN_FIT_MONTHS})")
    with ProcessPoolExecutor(max_workers=max(1, min(len(tasks), (os.cpu_count() or 2) - 1))) as ex:
        fitted = dict(ex.map(_fit_site, tasks))
    rows = []
    for site, _, _, _, window in tasks:
        rows.append({"site": site, **fitted[site], "area_m2": float(area[site]) * 2589988.110336,
                     "n_months": int(window.sum())})
    out = pd.DataFrame(rows).set_index("site")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out


def bcm_routed(data_dir: str | Path = "data", scenario: str = "s01",
               depth: dict[str, pd.DataFrame] | None = None,
               params: pd.DataFrame | None = None) -> pd.DataFrame:
    """BCM routed to each site's gauge by the benchmark's fitted parameters (``params``, the
    table of :func:`bcm_fit`; TAF/month, month end x site; NaN columns for the sites without a
    fit).  The routing runs over the whole record from the antecedent storage, on the depth
    times the site's registry area."""
    depth = bcm_depth(data_dir, scenario) if depth is None else depth
    params = bcm_fit(data_dir, scenario=scenario, depth=depth) if params is None else params
    area = _registry(data_dir)["area_mi2"]
    out = pd.DataFrame(np.nan, index=depth["run"].index, columns=list(SITES))
    for site, p in params.loc[[s for s in SITES if s in params.index]].iterrows():
        run, rch = _volumes(depth, site, float(area[site]))
        out[site] = bcm_routing.route(run, rch, p)["bcm_flow"].to_numpy() / M3_PER_TAF
    return out


def bcm_workbook(data_dir: str | Path = "data", scenario: str = "s01",
                 depth: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """BCM routed by the USGS sheets as delivered (the Kings by its refit), for comparison
    (TAF/month, month end x site; NaN columns for the sites without a sheet).  The routing runs
    over the whole record from the sheet's antecedent storage, on the depth times the sheet's
    fitted area; the last month is NaN where the sheet's recession lag reads a month past the
    end (SHA, ORO)."""
    depth = bcm_depth(data_dir, scenario) if depth is None else depth
    out = pd.DataFrame(np.nan, index=depth["run"].index, columns=list(SITES))
    for site in bcm_sites(data_dir):
        run, rch = depth["run"][site], depth["rch"][site]
        if run.isna().any() or rch.isna().any():
            raise ValueError(f"BCM {scenario} at {site}: a month with no valid cell")
        r = bcm_routing.route_depth(run, rch, bcm_routing.params_for(site, data_dir))
        out[site] = r["bcm_flow"].to_numpy() / M3_PER_TAF
    return out


def bcm_unrouted(data_dir: str | Path = "data", scenario: str = "s01",
                 depth: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """BCM ``run + rch`` with no routing, over each site's registry area (TAF/month, month end
    x site): the series the routing replaces, kept for the routing's effect."""
    depth = bcm_depth(data_dir, scenario) if depth is None else depth
    area = _registry(data_dir)["area_mi2"].reindex(list(SITES)).to_numpy(float)
    return (depth["run"] + depth["rch"]) * (area * TAF_PER_MM_MI2)

