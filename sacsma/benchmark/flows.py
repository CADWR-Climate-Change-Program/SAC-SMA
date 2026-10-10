"""Monthly flow at the CDEC full-natural-flow watersheds: observed, dPL-CalSim and the CalSim3
pipeline's VIC, on one forcing.

:func:`monthly_flows` is the whole of it: a tidy frame ``[date, site, model, flow_taf]``, one
row per month end, site and model over a water-year window (default WY1991-2018), NaN where a
model has no value.  The models:

* ``obs``  CDEC's monthly full natural flow (sensor 65, ``fnf_monthly.csv``) where the site's
  own station publishes it (:func:`observed_monthly`); for CLE, that of the station just
  downstream scaled to CLE's volume (:data:`MONTHLY_FROM`); for NML, which has neither
  (:data:`DAILY_ONLY`), the monthly sum of the daily FNF as the ``cdec_daily`` entities of the
  multifamily registry read it (:func:`observed_daily`: the registry's ``obs_store``, the
  hand-kept mask of bad days applied), a month counting only when every one of its days is
  valid.
* ``dpl``  dPL-CalSim (:data:`DPL_RUN`), its trained field on the 12 ``cdec_*`` entities on
  the CPU engine as the run is scored: the multifamily envelope (1949-10-01 to 2018-12-31),
  its state at the start from the cycle spin-up of the run's settings (:func:`dpl_daily`).
* ``vic_calsim3`` the routed monthly VIC of the CalSim3 pipeline on the matching climate
  (:data:`VIC_TABLES`), in TAF as delivered, at the CalSim3 node the existing 15-CDEC track
  of ``sacsma calsim`` maps each site to (:func:`vic_nodes`).

One area per site (:func:`site_table`, ``area_mi2``): the registry's, which is the area the
observed depth was made with (the published drainage area of the 15, the CalSim3 arc area of
CLE and CSN).  The depth model (``dpl``) is turned into volume with that area, so a score on
depth and a score on volume are the same score.  VIC cannot be rescaled: it is the
flow of its CalSim3 node's catchment, whose area (``vic_area_mi2``) can differ from the site's.

The dPL run is cached under ``artifacts/_local/cache/benchmark/`` (:func:`cache_dir`), keyed by
the content of the checkpoint and entity tables and of the engine's code, and by forcing name:
clear the folder after a forcing store changes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths

#: the 12 CDEC FNF sites of the benchmark, the ``cdec_daily`` entities of the registry outside
#: the Tulare basin less NHG (no monthly full natural flow, and 95 complete months of its daily
#: record in the window, none in October), north to south: descending area-weighted latitude of
#: the registry entity cells, the American (FOL) listed before the Yuba
#: (``sacsma._figures.FOLSOM_BEFORE_YUBA``)
SITES = ("SHA", "CLE", "BND", "ORO", "FOL", "YRS", "CSN", "MKM", "NML", "TLG", "MRC", "MIL")
#: the models of the frame, in plot order
MODELS = ("obs", "dpl", "vic_calsim3")
#: the forcing every model runs on (WGEN Product A scenario 1)
FORCING = "wgen_product_a"
#: the scoring window, water years (inclusive): the WGEN temperature detrending is small here
WY = (1991, 2018)
#: dPL-CalSim, the learned-parameter run of the multifamily domain
DPL_RUN = "7_calsim"
#: the sites of the 15-CDEC set whose VIC is the sum of the per-node series of their registry
#: arcs, not the basin-level series of the ``sacsma calsim`` 15-CDEC track: that series reaches
#: below the gauge (the Yuba's ``8RI_SMART`` holds Deer Creek, ``I_DER001`` and ``I_DER004``)
VIC_PER_NODE = ("YRS",)
#: the sites whose CDEC station has no monthly full natural flow (sensor 65) and no usable
#: station downstream, scored on the monthly sum of their daily record: NML is published only at
#: SNS, whose monthly series parts from NML's daily record with the season (August to November
#: 0.47 of it)
DAILY_ONLY = ("NML",)
#: the sites scored on the monthly full natural flow of the station just downstream, scaled to
#: the site's volume: CLE (Trinity Lake) on TNL (Trinity River at Lewiston, 3.8 % more area,
#: r 0.997 with CLE's complete daily months, 1.6 % more volume).  The scale is the site's daily
#: volume over the station's monthly volume on the months both cover (:func:`observed_monthly`)
MONTHLY_FROM = {"CLE": "TNL"}
#: the routed VIC table of each forcing (``paths.vic_routed`` product; None = the baseline)
VIC_TABLES = {"historical_livneh_unsplit": None, "wgen_product_a": "wgen_product_a",
              "historical_lto": "historical_lto"}
#: thousand acre-feet in 1 mm over 1 mi2 (2,589,988.11 m2 x 1e-3 m / 1,233.4818 m3 per AF)
TAF_PER_MM_MI2 = 2589988.110336e-3 / 1233.48183754752 / 1000.0
#: the registry family of the sites
_FAMILY = "cdec_daily"


def entity(site: str) -> str:
    """The registry entity of a site (``cdec_<site>``)."""
    return f"cdec_{site}"


def cache_dir(artifacts_dir: str | Path = "artifacts") -> Path:
    """The untracked folder of the benchmark's cached runs."""
    return paths.local(artifacts_dir, "cache/benchmark")


def _registry(data_dir) -> pd.DataFrame:
    reg = pd.read_csv(paths.entities(data_dir), dtype={"site_id": str})
    reg = reg[reg["family"] == _FAMILY].set_index("site_id")
    missing = sorted(set(SITES) - set(reg.index))
    if missing:
        raise ValueError(f"no {_FAMILY} registry entity for {missing}")
    return reg.loc[list(SITES)]


def _water_months(wy) -> pd.DatetimeIndex:
    """Month ends of water years ``wy[0]..wy[1]``."""
    return pd.date_range(f"{wy[0] - 1}-10-31", f"{wy[1]}-09-30", freq="ME")


# ----------------------------------------------------------------------------- the sites
def site_table(data_dir: str | Path = "data") -> pd.DataFrame:
    """``[site, name, area_mi2, order, cdec_id, entity_id, vic_area_mi2]``, north to south.

    ``name`` is the CDEC station name; ``area_mi2`` the registry area (the observed depth's);
    ``vic_area_mi2`` the CalSim3 catchment area of the site's VIC node (NaN without one)."""
    reg = _registry(data_dir)
    st = pd.read_csv(paths.cdec_fnf(data_dir, "stations.csv")).set_index("id")["name"]
    cdec_id = reg["outlet_source"].str.split(":").str[-1]
    vic_area = {s: a for s, (_, a) in vic_nodes(data_dir).items()}
    return pd.DataFrame({
        "site": list(SITES), "name": [st.get(c, "") for c in cdec_id],
        "area_mi2": reg["area_mi2"].to_numpy(float), "order": np.arange(len(SITES)),
        "cdec_id": cdec_id.to_numpy(), "entity_id": reg["entity_id"].to_numpy(),
        "vic_area_mi2": [vic_area.get(s, np.nan) for s in SITES]})


def _taf(depth_mm_month: pd.DataFrame, data_dir) -> pd.DataFrame:
    """Monthly depth (mm, month x site) to TAF over each site's area."""
    area = _registry(data_dir)["area_mi2"]
    return depth_mm_month * (area.reindex(depth_mm_month.columns).to_numpy() * TAF_PER_MM_MI2)


def _monthly_sum(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily mm/day (date x site) to monthly mm (month end x site); a month with a missing or
    NaN day is NaN."""
    days = pd.date_range(daily.index.min().to_period("M").start_time,
                         daily.index.max().to_period("M").end_time.normalize(), freq="D")
    d = daily.reindex(days)
    total = d.resample("ME").sum()
    valid = d.notna().resample("ME").sum()
    out = total.where(valid.eq(pd.Series(total.index.days_in_month, index=total.index), axis=0))
    out.index.name = "date"
    return out


# ------------------------------------------------------------------------------ observed
def observed_daily(data_dir: str | Path = "data") -> pd.DataFrame:
    """The observed daily FNF depth (mm/day, date x site) from each entity's ``obs_store``,
    the days of ``fnf_daily_mask.csv`` set to NaN (negative and missing days are NaN in the
    stores).  The registry's training window is not applied: it is a training choice."""
    from ..cdec15 import load_gage

    reg = _registry(data_dir)
    gage_store = paths.relative(paths.gage_15cdec(data_dir), data_dir)
    fnf_store = paths.relative(paths.cdec_fnf(data_dir, "fnf_daily_mm.csv"), data_dir)
    gage = fnf = None
    cols = {}
    for site, r in reg.iterrows():
        store, col = r["obs_store"].rsplit(":", 1)
        if store == gage_store:
            gage = load_gage(data_dir) if gage is None else gage
            g = gage[gage["basin"] == site]
            cols[site] = pd.Series(g[col].to_numpy(float), index=pd.DatetimeIndex(g["date"]))
        elif store == fnf_store:
            if fnf is None:
                fnf = pd.read_csv(paths.cdec_fnf(data_dir, "fnf_daily_mm.csv"),
                                  parse_dates=["date"])
            g = fnf[fnf["station"] == r["outlet_source"].split(":")[-1]]
            cols[site] = pd.Series(g[col].to_numpy(float), index=pd.DatetimeIndex(g["date"]))
        else:
            raise ValueError(f"{site}: unknown observation store {store!r}")
    obs = pd.DataFrame(cols).sort_index()
    obs.index.name = "date"
    mask = pd.read_csv(paths.cdec_fnf(data_dir, "fnf_daily_mask.csv"), parse_dates=["date"])
    ent = {entity(s): s for s in SITES}
    for eid, day in zip(mask["entity_id"], mask["date"], strict=True):
        if eid in ent and day in obs.index:
            obs.loc[day, ent[eid]] = np.nan
    return obs[list(SITES)]


def observed_monthly(data_dir: str | Path = "data") -> pd.DataFrame:
    """The observed monthly volume (TAF, month end x site): CDEC's monthly full natural flow at
    the site's station; for :data:`MONTHLY_FROM` at the station named there, times the site's
    daily volume over that station's monthly volume on the complete daily months; for
    :data:`DAILY_ONLY` the sum of the daily record (a month with any invalid day is NaN)."""
    daily = _taf(_monthly_sum(observed_daily(data_dir)[[*DAILY_ONLY, *MONTHLY_FROM]]),
                 data_dir)
    m = pd.read_csv(paths.cdec_fnf(data_dir, "fnf_monthly.csv"), parse_dates=["date"])
    m = m.pivot(index="date", columns="station", values="flow_af") / 1000.0
    out = daily.reindex(daily.index.union(m.index))     # the monthly record reaches WY1922
    station = site_table(data_dir).set_index("site")["cdec_id"]
    for site in SITES:
        if site in DAILY_ONLY:
            continue
        if site in MONTHLY_FROM:
            sub = m[MONTHLY_FROM[site]].reindex(out.index)
            both = out[site].notna() & sub.notna()
            out[site] = sub * (out.loc[both, site].sum() / sub[both].sum())
            continue
        if station[site] not in m:
            raise KeyError(f"{site}: no monthly full natural flow for station {station[site]}")
        out[site] = m[station[site]]
    out.index.name = "date"
    return out[list(SITES)]


# ----------------------------------------------------------------------------- dPL model
def _engine_code():
    from ..dpl.evaluate import ENGINE_CODE, content_key

    return ENGINE_CODE, content_key


def _cached(path: Path, run) -> pd.DataFrame:
    """A daily frame (date x site) cached as npz at ``path``, made by ``run()`` when missing."""
    if path.exists():
        z = np.load(path)
        return pd.DataFrame(z["flow"].T, index=pd.DatetimeIndex(z["dates"], name="date"),
                            columns=z["sites"].tolist())
    df = run()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, flow=df.to_numpy().T, dates=df.index.values,
             sites=np.array([str(c) for c in df.columns]))
    return df


def dpl_field(forcing: str = FORCING, data_dir: str | Path = "data",
              artifacts_dir: str | Path = "artifacts", run: str = DPL_RUN):
    """``(net, x, dom, cfg)`` of the run on the 12 ``cdec_*`` entities and ``forcing``, on the
    CPU in float64: :func:`sacsma.dpl.evaluate.load_net_from_checkpoint` with the entity set
    narrowed to the sites (the field is per row: a row's parameters do not depend on the
    other rows)."""
    import torch

    from ..dpl.config import config_from_checkpoint
    from ..dpl.data import load_domain_tensors
    from ..dpl.evaluate import checkpoint_features
    from ..dpl.parameter_net import ParameterNet

    ck = torch.load(paths.dpl_checkpoint(artifacts_dir, run, paths.MULTIFAMILY),
                    map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ck)
    ents = tuple(entity(s) for s in SITES)
    absent = sorted(set(ents) - set(ck["basins"]))
    if absent:
        raise ValueError(f"{run} did not train {absent}")
    dom = load_domain_tensors(data_dir, domain=ck["domain"], device="cpu", dtype=torch.float64,
                              basins=ents, product=forcing)
    x = checkpoint_features(ck, dom, data_dir)
    net = ParameterNet.from_checkpoint(ck, x.shape[1]).to("cpu", torch.float64)
    return net, x, dom, cfg


def dpl_daily(forcing: str = FORCING, data_dir: str | Path = "data",
              artifacts_dir: str | Path = "artifacts", run: str = DPL_RUN) -> pd.DataFrame:
    """dPL-CalSim's daily flow (mm/day, date x site) on ``forcing`` over the multifamily
    envelope, as the run is scored (:mod:`sacsma.dpl.evaluate_multi_timescale`): its trained
    field on the CPU engine, the state at 1949-10-01 from the cycle spin-up (cached)."""
    from ..dpl.evaluate import simulate_field
    from ..dpl.multi_timescale import ENVELOPE_END, ENVELOPE_START

    code, key_of = _engine_code()
    ckpt = paths.dpl_checkpoint(artifacts_dir, run, paths.MULTIFAMILY)
    key = key_of(ckpt, paths.entity_cells(data_dir), paths.flowlens(data_dir), *code)

    def go() -> pd.DataFrame:
        net, x, dom, cfg = dpl_field(forcing, data_dir, artifacts_dir, run)
        r = simulate_field(net, x, dom, cfg, ENVELOPE_START, ENVELOPE_END, dom.W.cpu().numpy(),
                           spinup="cycle")
        t0 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_START)))
        dates = dom.dates[t0:t0 + r["flow"].shape[1]]
        return pd.DataFrame(r["flow"].T, index=dates.rename("date"),
                            columns=[b.removeprefix("cdec_") for b in dom.basins])

    df = _cached(cache_dir(artifacts_dir) / f"dpl_{run}_{key}" / f"{forcing}.npz", go)
    return df[list(SITES)]


# ------------------------------------------------------------------------------- VIC
def vic_nodes(data_dir: str | Path = "data") -> dict[str, tuple[list[str], float]]:
    """``{site: (VIC series, CalSim3 catchment area mi2)}`` for the sites with a CalSim3 node.

    The sites of the 15-CDEC set use the basin-level VIC of the ``sacsma calsim`` 15-CDEC
    track (:func:`sacsma.calsim.compare._anchor_set_taf`): a rim watershed its one 8-River or
    cumulative series (``I_SHSTA`` and ``8RI_SRBB`` hold the no-Goose-Lake routing), MKM the
    sum of its per-node series; the area is the set's CalSim3 catchment area.  CLE,
    CSN and the sites of :data:`VIC_PER_NODE` sum the per-node series of their registry arcs
    (Trinity Lake ``I_TRNTY``; the four Cosumnes arcs above Michigan Bar; the 16 Yuba arcs above
    Smartsville), on the arcs' summed area."""
    from ..calsim.catchments import BASIN_RIM_SYSTEM, basin_areas
    from ..calsim.compare import load_basin_nodes, load_name_map

    arc2vic = load_name_map(data_dir)
    out: dict[str, tuple[list[str], float]] = {}
    nodes = load_basin_nodes(data_dir, paths.CDEC15)
    nodes = nodes[nodes["in_calsim3"].astype(bool)]
    bsys = BASIN_RIM_SYSTEM.get(paths.CDEC15, {})
    areas = basin_areas(data_dir, domain=paths.CDEC15)
    for site, g in nodes.groupby("basin"):
        if site in VIC_PER_NODE:
            continue
        own = bsys.get(site)
        arcs = [a for a, sy in zip(g["arc"].astype(str), g["system"], strict=True)
                if own is None or sy == own]
        out[str(site)] = (sorted({arc2vic.get(a, a) for a in arcs}), float(areas[site]))
    reg = _registry(data_dir)
    hier = pd.read_csv(paths.calsim3_targets(data_dir, "arc_hierarchy.csv")).set_index("arc")
    for site in SITES:
        r = reg.loc[site]
        if site in out or r["delineation"] != "arcs":
            continue
        arcs = str(r["arcs"]).split(";")
        # the name map sends an arc to its basin-level series; a per-node site keeps the arc's own
        names = arcs if site in VIC_PER_NODE else [arc2vic.get(a, a) for a in arcs]
        out[site] = (sorted(set(names)),
                     float(hier["sq_mi"].reindex(arcs).sum()))
    return {s: out[s] for s in SITES if s in out}


def vic_monthly(forcing: str = FORCING, data_dir: str | Path = "data") -> pd.DataFrame:
    """The CalSim3 pipeline's routed VIC on ``forcing``'s climate (TAF, month end x site);
    NaN columns for the sites without a node."""
    from ..calsim import load_vic_monthly

    if forcing not in VIC_TABLES:
        raise ValueError(f"no routed VIC table for {forcing!r} (have {list(VIC_TABLES)})")
    v = load_vic_monthly(data_dir, product=VIC_TABLES[forcing])
    v = v.assign(date=pd.to_datetime(v["date"]) + pd.offsets.MonthEnd(0))
    w = v.pivot_table(index="date", columns="vic_name", values="flow_taf", aggfunc="sum")
    nodes = vic_nodes(data_dir)
    out = pd.DataFrame(index=w.index, columns=list(SITES), dtype=float)
    for site, (names, _) in nodes.items():
        absent = sorted(set(names) - set(w.columns))
        if absent:
            raise KeyError(f"{site}: VIC series {absent} not in the table")
        out[site] = w[names].sum(axis=1, min_count=len(names))
    out.index.name = "date"
    return out


# ------------------------------------------------------------------------------- the frame
def _tidy(wide: pd.DataFrame, model: str, months: pd.DatetimeIndex) -> pd.DataFrame:
    w = wide.reindex(index=months, columns=list(SITES))
    w.index.name = "date"
    t = w.reset_index().melt(id_vars="date", var_name="site", value_name="flow_taf")
    return t.assign(model=model)[["date", "site", "model", "flow_taf"]]


def monthly_flows(forcing: str = FORCING, data_dir: str | Path = "data",
                  artifacts_dir: str | Path = "artifacts", *, wy=WY, models=MODELS,
                  log=print) -> pd.DataFrame:
    """The monthly flow of ``models`` at the sites on ``forcing`` (``obs`` is the same for every
    forcing): tidy ``[date, site, model, flow_taf]``, every month end of water years
    ``wy[0]..wy[1]`` x site x model, NaN where a model has no value (a month of observations
    with an invalid day, a site a model does not cover).  Sites north to south."""
    months = _water_months(wy)
    make = {
        "obs": lambda: observed_monthly(data_dir),
        "dpl": lambda: _taf(_monthly_sum(dpl_daily(forcing, data_dir, artifacts_dir)), data_dir),
        "vic_calsim3": lambda: vic_monthly(forcing, data_dir),
    }
    parts = []
    for m in models:
        if m not in make:
            raise ValueError(f"unknown model {m!r} (one of {MODELS})")
        t = time.perf_counter()
        parts.append(_tidy(make[m](), m, months))
        if log:
            log(f"benchmark flows: {m} on {forcing} ({time.perf_counter() - t:.1f} s)")
    out = pd.concat(parts, ignore_index=True)
    out["site"] = pd.Categorical(out["site"], categories=list(SITES), ordered=True)
    out["model"] = pd.Categorical(out["model"], categories=list(models), ordered=True)
    return out.sort_values(["model", "site", "date"]).reset_index(drop=True)
