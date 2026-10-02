"""Tier-2 CalSim3 validation of a multifamily dPL run: every rim INFLOW arc on its own.

The run archive holds entity aggregates only, so tier 2 re-runs the checkpoint forward over
the envelope (the evaluator's streaming protocol, eager, no-grad) and aggregates the
per-cell runoff onto each ``CalSim3_Merged`` rim polygon with square-cell overlap weights.
Where nested entities share a cell, the arc takes the (entity, cell) row of the most local
trained entity whose registry arc list holds the arc — the crosswalk's own convention.
Arcs that no trained entity lists have no HRU rows in the run; by default they are
**extrapolated**: their region cells get flow lengths traced to the polygon exit on the
HydroSHEDS grid (the entity builder's exit-of-footprint mode, run in a subprocess because
the raster and vector GDAL stacks must not share a process), the trained parameter network
maps their attributes to parameters, and they are simulated and scored like the others,
flagged ``basis = extrapolated``.  ``--no-extend`` restricts tier 2 to the trained
footprints.  ``basis`` says how an arc was simulated, not whether the loss saw its cells: a
USGS creek gauge has no arc list, so an arc on its footprint is extrapolated by construction
although its cells were fitted.  ``trained_cell_frac`` (the share of each arc's area on cells
the run trained on) tells those apart; the regionalization test proper is the arcs at 0.

Each arc's monthly volume (mean depth over its covered cells x the polygon ``SQ_MI``) is
scored against its CalSim3 ``INFLOW`` series over WY1950-84, and over the parent entity's
own training window as the in-sample comparison.  The valley node ``I_SRBB_VAL`` is
simulated for its volume but has no series to score against.  The converse is ``I_RUB002``
(the lower Rubicon accretion): it has a CalSim3 series but no merged-layer polygon, so tier
2 does not simulate it, and ``tier2_not_simulated.csv`` (polygons only) does not list it.

A checkpoint with ``DplConfig.holdout_wy`` (water years held out of training in every family) is
scored over them instead (:func:`sacsma.dpl.calsim_tier1.holdout_windows`): ``WY1976-85``, the
same on each arc's own gauge-record months only (``WY1976-85_own``, :func:`sacsma.dpl.calsim_arcs.
own_record_months`), ``WY1976-84`` and ``WY1950-84_mixed``, and the ``train`` window without the
held-out water years (``excluded_wy``); the maps, regime figures and summary are over
``WY1976-85``, and ``tier2_anchor_rescaled.csv`` (:func:`anchor_rescaled`, eval-only) is written
beside the absolute scores.  With ``calsim_monthly`` entities in the run each trained arc is its
own parent (the smallest listing entity), so its rows are the trained cs_ rows.

Usage::

    python -m sacsma.dpl.calsim_tier2 <run_dir | checkpoint.pt> [--out DIR] [--data-dir data]
                                  [--device cpu|cuda] [--no-maps] [--no-extend]
                                  [--tiles-dir tmp/hydrosheds] [--trace-python PY] [--figures-only]
                                  [--label NAME] [--dedup-cells] [--components [fastslow|parts]]
                                  [--extension-cells CSV] [--temp-delta DT] [--precip-scale S]
                                  [--anchor-rescaled] [--scenarios NAME=DT:PS,...]
                                  [--batch-window DAYS]

Writes ``tier2_metrics.csv`` (one row per arc x window), ``tier2_monthly.csv``,
``tier2_arcs.csv`` (coverage, parent entity and trained-cell share per arc),
``tier2_not_simulated.csv``, ``tier2_extension_cells.csv``, ``tier2_sim_daily.npz``, the KGE /
bias maps and the regime figures under ``figures/``, and prints the summary.  When the run
folder holds ``sim_daily_mm.npz`` the re-run's entity aggregates are checked against it.
``--figures-only`` redraws from the CSVs already in ``--out``: the maps from the tracked
``tier2_metrics.csv``, the regime figures only where the untracked ``tier2_monthly.csv`` of a
previous full run is present.

Opt-in extras, all off by default: ``--components`` also writes
``tier2_components_monthly.csv`` (per arc and month the routed FAST and SLOW runoff
components, ``fast_taf + slow_taf = total_taf = sim_taf``); ``--components parts`` adds the
four routed runoff parts ``quick_taf`` (impervious + ADIMP direct + surface) and
``interflow_taf`` (``quick + interflow = fast``), ``supplemental_taf`` and ``primary_taf``
baseflow (``supplemental + primary = slow``).  Components and parts are NET of SAC-SMA's
riparian et4 channel-ET deduction (the parts: the net inflow apportioned in proportion to
their pre-deduction values).  ``--extension-cells`` reuses a previous run's
``tier2_extension_cells.csv`` flow lengths instead of tracing; ``--temp-delta`` /
``--precip-scale`` re-run under a PLACEHOLDER uniform climate perturbation (degC added to
tavg/tmin/tmax, precip multiplied, spin-up included) — a stand-in until the WGEN daily
scenario weather is supplied.  Any extra, and ``--dedup-cells``, also writes
``tier2_run_info.json`` (the extras, the dedup setting, the perturbation and its PLACEHOLDER
flag).  ``--scenarios t1=1:1,p85=0:0.85,...`` runs several such perturbations in ONE batched
forward pass (:func:`stream_batch`: the rows replicated once per scenario, parameters and
routing kernels evaluated once) and writes each scenario's CSVs, exactly as a single pass
would, into ``--out/NAME`` (default ``<run>/tier2_scenarios/NAME``), without maps or figures;
it does not combine with ``--dedup-cells``.  ``--batch-window`` is the days per streamed chunk
and bounds device memory only (results differ between windows by round-off).

Needs the ``dpl`` extra (torch) and a source checkout: the extrapolated arcs run the tracer
of ``dataprep/build_flowlens.py`` (``--trace-cells``) in a subprocess, which needs
``rasterio`` and the HydroSHEDS v2 tiles (read from ``--tiles-dir`` when complete, otherwise
streamed from the HydroSHEDS server; the size check needs network, or a local tile's
``.ok`` marker); ``--trace-python`` names an interpreter that has rasterio when this one
does not (the ``sacsma-gis`` env).  Without them those arcs fall back to straight-line x 1.5
lengths, which the log states; ``--no-extend`` needs neither.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..metrics import center_of_timing, kge, nse, pbias, pearson, seasonal_mismatch
from ..calsim.catchments import (_EQ_CRS, _GRID_STEP_DEG, EXCLUDE_ARCS, MERGED_LAYER,
                         _square_cell_overlap, load_catchments, load_crosswalk, series_arc)
from .calsim_tier1 import (AF_PER_MM_MI2, VALIDATION_WINDOW, WINDOWS, arc_to_set,
                    holdout_month_mask, holdout_windows, load_references, load_sets, registry_arcs,
                    registry_windows, run_holdout_wy, window_range)

#: suffix of the window that scores an arc on its OWN gauge-record months inside the holdout
#: (:func:`sacsma.dpl.calsim_arcs.own_record_months`), for a run with ``holdout_wy``
OWN_SUFFIX = "_own"


def rim_polygons(data_dir: str | Path = "data"):
    """The merged rim catchments with their INFLOW-series arc id (below-rim reaches dropped)."""
    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True)
    catch["arc"] = catch["node"].map(series_arc)
    return catch[~catch["arc"].isin(EXCLUDE_ARCS)].reset_index(drop=True)


def cell_arc_overlap(hrus: pd.DataFrame, catch) -> pd.DataFrame:
    """Square-cell x rim-polygon overlap areas: ``[key, arc, area_mi2]`` for every
    distinct cell of the run's HRU table."""
    cells = hrus.drop_duplicates("key")[["key", "lat", "lon"]].reset_index(drop=True)
    catch_eq = catch[["cid", "node", "geometry"]].to_crs(_EQ_CRS)
    mapping, _ = _square_cell_overlap(cells, catch_eq, "EPSG:4326", _GRID_STEP_DEG)
    mapping["arc"] = mapping["node"].map(series_arc)
    return mapping[["key", "arc", "area_mi2"]].reset_index(drop=True)


def parent_entities(basins, data_dir: str | Path = "data") -> dict[str, str]:
    """arc -> the most local trained entity whose registry arc list holds the arc."""
    reg = pd.read_csv(Path(data_dir) / "multifamily" / "entities.csv",
                      dtype={"site_id": str}).set_index("entity_id")
    arcs = registry_arcs(data_dir)
    out = {}
    for arc in sorted({a for b in basins for a in arcs[b]}):
        cands = [b for b in basins if arc in arcs[b]]
        out[arc] = min(cands, key=lambda b: (float(reg.loc[b, "area_mi2"]), b))
    return out


def arc_weights(dom, catch, mapping: pd.DataFrame, parent: dict[str, str]):
    """Aggregation matrix ``(A, N)`` over the run's HRU rows (rows sum to 1 = mean depth over
    the covered cells) plus the per-arc table [arc, node, entity, sq_mi, covered_mi2,
    cover_frac, n_cells]."""
    hrus = dom.hrus
    row_of = {(b, k): i for i, (b, k) in enumerate(zip(hrus["basin"], hrus["key"], strict=True))}
    sq_mi = catch.groupby("arc")["sq_mi"].sum()
    arcs = [a for a in sq_mi.index if a in parent]
    W = np.zeros((len(arcs), len(hrus)), dtype=np.float64)
    rows = []
    by_arc = {a: g for a, g in mapping.groupby("arc")}
    for j, arc in enumerate(arcs):
        ent = parent[arc]
        g = by_arc.get(arc)
        idx, wts = [], []
        if g is not None:
            for key, a in zip(g["key"], g["area_mi2"], strict=True):
                i = row_of.get((ent, key))
                if i is not None:
                    idx.append(i)
                    wts.append(float(a))
        covered = float(sum(wts))
        if covered > 0:
            W[j, idx] = np.asarray(wts) / covered
        rows.append(dict(arc=arc, node=arc[2:], entity=ent, sq_mi=float(sq_mi[arc]),
                         covered_mi2=covered, cover_frac=covered / float(sq_mi[arc]),
                         n_cells=len(idx)))
    return W, pd.DataFrame(rows)


def region_arc_overlap(catch, data_dir: str | Path = "data") -> pd.DataFrame:
    """Square-cell x rim-polygon overlap of the whole region grid, ``[key, lat, lon, arc,
    area_mi2]``: every arc's cells whether or not the run trained on them."""
    grid = pd.read_csv(Path(data_dir) / "region" / "grid_cells.csv", usecols=["key", "lat", "lon"])
    mapping, _ = _square_cell_overlap(grid, catch[["cid", "node", "geometry"]].to_crs(_EQ_CRS),
                                      "EPSG:4326", _GRID_STEP_DEG)
    mapping["arc"] = mapping["node"].map(series_arc)
    g = grid.set_index("key")
    mapping["lat"] = mapping["key"].map(g["lat"])
    mapping["lon"] = mapping["key"].map(g["lon"])
    return mapping[["key", "lat", "lon", "arc", "area_mi2"]].reset_index(drop=True)


def trained_cell_share(region_map: pd.DataFrame, trained_keys) -> pd.Series:
    """Per arc (the index), the share of its overlap area on cells the run trained on: 1 for
    an arc wholly inside trained footprints, 0 for one the loss never saw.  An arc no trained
    entity lists can still sit on trained cells where a USGS creek gauge covers it — its
    parameters were fitted through that gauge and only the routing to the arc is new."""
    on = region_map["key"].isin(set(trained_keys))
    tot = region_map.groupby("arc")["area_mi2"].sum()
    hit = region_map[on].groupby("arc")["area_mi2"].sum().reindex(tot.index).fillna(0.0)
    return (hit / tot).rename("trained_cell_frac")


def uncovered_arc_cells(catch, parent: dict[str, str], data_dir: str | Path = "data",
                        region_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """HRU rows for the rim polygons no trained entity lists: every region grid cell whose
    square overlaps the polygon, weight = overlap area, elevation from the region statics.
    Columns ``[basin, key, lat, lon, area_weight, elev]`` (basin = the arc id).  Pass the
    :func:`region_arc_overlap` table to reuse it."""
    from ..io import soilveg_path
    if region_map is None:
        region_map = region_arc_overlap(catch, data_dir)
    unc = region_map[~region_map["arc"].isin(parent)]
    if unc.empty:
        return pd.DataFrame(columns=["basin", "key", "lat", "lon", "area_weight", "elev"])
    sv = pd.read_csv(soilveg_path(data_dir, "multifamily"), usecols=["key", "dem_elev"]).set_index("key")["dem_elev"]
    out = pd.DataFrame({"basin": unc["arc"], "key": unc["key"], "lat": unc["lat"], "lon": unc["lon"],
                        "area_weight": unc["area_mi2"], "elev": unc["key"].map(sv)})
    out = out[out["elev"].notna()].sort_values(["basin", "key"]).reset_index(drop=True)
    return out


_BUILD_FLOWLENS = Path(__file__).resolve().parents[2] / "dataprep" / "build_flowlens.py"


def trace_arc_cells(cells_csv: str | Path, out_csv: str | Path, tiles_dir: str | Path = "tmp/hydrosheds") -> None:
    """Flow length of every row of ``cells_csv`` to where its HydroSHEDS flow path leaves its
    arc's cell footprint (``build_flowlens.trace_cells_to_exit``).  Kept for the in-process
    case; :func:`_flowlens_for` runs the tracer as ``dataprep/build_flowlens.py --trace-cells``
    in a subprocess, which needs rasterio and pandas only."""
    sys.path.insert(0, str(_BUILD_FLOWLENS.parent))
    import build_flowlens as bf  # noqa: E402
    bf.trace_cells_to_exit(Path(cells_csv), Path(out_csv), Path(tiles_dir))


def _tracer_env(trace_python: str | None) -> dict[str, str] | None:
    """Environment for the tracer subprocess: unchanged for this interpreter; for another
    one, PATH without this environment's own directories (``pick_device`` prepends
    ``sys.prefix/Library/bin``, whose ``gdal.dll`` rasterio in the child would pick up)."""
    if not trace_python or Path(trace_python).resolve() == Path(sys.executable).resolve():
        return None
    env = dict(os.environ)
    prefix = os.path.normcase(os.path.abspath(sys.prefix)) + os.sep
    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep)
        if not (os.path.normcase(os.path.abspath(p)) + os.sep).startswith(prefix))
    return env


def _flowlens_for(cells: pd.DataFrame, tiles_dir: str | Path,
                  trace_python: str | None = None) -> pd.DataFrame:
    """Add ``flowlen`` (m) and ``flowlen_method`` to the extension HRU rows, tracing in a
    subprocess (``dataprep/build_flowlens.py --trace-cells`` under ``trace_python``, default
    this interpreter — an environment with rasterio; the raster and vector GDAL stacks must
    not share a process); falls back to straight-line x 1.5 to each arc's lowest cell if
    tracing fails, and says so.  Another interpreter gets a PATH without this environment's
    own directories: ``pick_device`` puts this env's ``Library/bin`` (its GDAL) on PATH for
    NVRTC, and rasterio in the child would bind to that ``gdal.dll`` instead of its own."""
    with tempfile.TemporaryDirectory() as td:
        cin, cout = Path(td) / "cells.csv", Path(td) / "flowlens.csv"
        cells.to_csv(cin, index=False)
        r = subprocess.run([trace_python or sys.executable, str(_BUILD_FLOWLENS), "--trace-cells",
                            str(cin), str(cout), "--tiles-dir", str(tiles_dir)],
                           capture_output=True, text=True, env=_tracer_env(trace_python))
        if r.returncode == 0 and cout.exists():
            fl = pd.read_csv(cout)
            out = cells.merge(fl, on=["basin", "key"], how="left")
            out = out.rename(columns={"flowlen_m": "flowlen", "method": "flowlen_method"})
            n_tr = int((out.flowlen_method != "fallback").sum())
            print(f"tier2: traced flow lengths for {n_tr}/{len(out)} extension cells", flush=True)
            return out
        print("tier2: flow-length tracing failed, using straight-line x 1.5 to each arc's lowest "
              f"cell\n{r.stderr[-800:]}", flush=True)
    out = cells.copy()
    out["flowlen"] = np.nan
    for arc, sub in out.groupby("basin"):
        j = sub["elev"].idxmin()
        lat0, lon0 = out.loc[j, "lat"], out.loc[j, "lon"]
        p1, p2 = np.deg2rad(sub.lat), np.deg2rad(lat0)
        a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lon0 - sub.lon) / 2) ** 2
        out.loc[sub.index, "flowlen"] = 2 * 6371008.8 * np.arcsin(np.sqrt(a)) * 1.5
    out["flowlen_method"] = "straight_x1.5"
    return out


def _net_for_hrus(ckpt: str | Path, hrus: pd.DataFrame, data_dir: str | Path, device=None):
    """Rebuild (net, x, dom, cfg) from a checkpoint for an arbitrary HRU table whose ``basin``
    column names the aggregation units — the evaluator's front half with the entity-store
    loader swapped for the given rows."""
    import dataclasses as _dc

    import sacsma.dpl.data as D
    from .config import DplConfig
    from .data import load_domain_tensors
    from .features import AEF_STORE_VARIANTS, FeatureSet, aef_store, build_features
    from .parameter_net import ParameterNet
    from ..io import soilveg_path
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    known = {f.name for f in _dc.fields(DplConfig)}
    cfg = DplConfig(**{k: v for k, v in ck["cfg"].items() if k in known})
    nc = ck.get("net_config", {})
    if nc.get("gnn_k", 0):
        raise ValueError("extrapolation needs a per-cell network (gnn_k == 0)")
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basins = tuple(dict.fromkeys(hrus["basin"]))
    # the entity-store loader is consulted twice: by the domain-tensor builder (its own
    # imported name) and by the forcing reader, which subsets the region store to the
    # table's cells (io's module attribute) — swap both for the extension rows
    import sacsma.io as IO
    orig_d, orig_io = D.load_hru_table, IO.load_hru_table
    D.load_hru_table = IO.load_hru_table = lambda *a, **k: hrus.copy()
    try:
        dom = load_domain_tensors(data_dir, domain=ck["domain"], device=dev, dtype=torch.float64,
                                  basins=basins, dynamic_window=None, calsim_footprint=False)
    finally:
        D.load_hru_table, IO.load_hru_table = orig_d, orig_io
    stats = FeatureSet(x=np.empty((0, 0), dtype=np.float32), **ck["features"])
    variant = ck["variant"]
    fs = build_features(dom.hrus, variant=variant,
                        forcing=dom.forcing if variant in ("climate", "physical_climate") else None,
                        climate_window=stats.climate_window, climate_product=stats.climate_product,
                        physical_path=(soilveg_path(data_dir, ck["domain"])
                                       if variant in ("physical", "physical_climate") else None),
                        aef_path=aef_store(data_dir) if variant in AEF_STORE_VARIANTS else None,
                        stats=stats)
    x = torch.as_tensor(fs.x).to(dev, torch.float64)
    net = ParameterNet(x.shape[1], hidden=nc.get("hidden", 64), embed=nc.get("embed", 32),
                       dropout=nc.get("dropout", 0.1), grouped_heads=nc.get("grouped_heads", False),
                       gnn_k=0, n_nodes=None, seasonal_params=tuple(nc.get("seasonal_params", ())),
                       seasonal_amp=nc.get("seasonal_amp", 0.18),
                       seasonal_amp_frac=nc.get("seasonal_amp_frac", 0.10),
                       canopy=nc.get("canopy", False),
                       canopy_separate_trunk=nc.get("canopy_separate_trunk", True),
                       canopy_lite=nc.get("canopy_lite", False),
                       dynamic_params=tuple(nc.get("dynamic_params", ())),
                       dynamic_amp=nc.get("dynamic_amp", 0.5),
                       pxtemp_learn=nc.get("pxtemp_learn", False),
                       pxtemp_box=tuple(nc.get("pxtemp_box", (-1.0, 3.0))),
                       pxtemp_tau=nc.get("pxtemp_tau", 1.0)).to(dev, torch.float64)
    net.load_state_dict(ck["net"])
    return net, x, dom, cfg


#: the ``components`` modes of :func:`stream` / :func:`score_run` / ``--components``:
#: ``fastslow`` = the routed FAST / SLOW pair; ``parts`` = that pair plus the four
#: routed runoff PARTS (:data:`PART_COLUMNS`).
COMPONENT_MODES = ("fastslow", "parts")
#: the parts columns (TAF) of ``tier2_components_monthly.csv`` under ``--components parts``
PART_COLUMNS = ("quick", "interflow", "supplemental", "primary")


def _component_mode(components) -> str | None:
    """Normalise ``components``: falsy -> None, True -> ``fastslow``, else a mode name."""
    if not components:
        return None
    if components is True:
        return "fastslow"
    if components not in COMPONENT_MODES:
        raise ValueError(f"components {components!r} (one of {COMPONENT_MODES})")
    return components


def stream(net, x, dom, cfg, W_arc: np.ndarray, *, spinup: str = "cycle",
           components: bool | str = False, temp_delta: float = 0.0, precip_scale: float = 1.0,
           dedup_cells: bool = False):
    """Stream the trained field over the envelope (the evaluator's protocol, including its
    ``spinup``: :func:`sacsma.dpl.spinup.spin_state`) and return (arc depth (A, T), entity
    depth (B, T), t0, t1, bad_rows) in mm/day.  A cell whose physics returns NaN
    (extrapolated parameters at their bounds) is zeroed and flagged in ``bad_rows`` so it
    cannot blank every aggregate that carries a zero weight on it.  ``dedup_cells``: the
    per-cell physics runs once per distinct cell (:func:`sacsma.dpl.data.with_cell_dedup`);
    ``W_arc``, the flows (components and parts included) and ``bad_rows`` stay per HRU row.

    ``components`` (True / ``"fastslow"``) appends the arc depths of the routed fast and
    slow runoff components ((A, T) each, ``fast + slow`` = the arc depth; NaN cell-days
    zeroed in both, as in the total); ``"parts"`` appends after them the arc depths of the
    four routed runoff parts, quick (impervious + ADIMP direct + surface), interflow,
    supplemental and primary baseflow (``quick + interflow = fast``, ``supplemental +
    primary = slow`` up to rounding), and streams the spin-up with the parts on so their
    routing history is exact.  Components and parts are all NET of SAC-SMA's riparian et4
    channel-ET deduction (and dry-channel clamp), which act before routing.
    ``temp_delta`` / ``precip_scale`` are a PLACEHOLDER uniform climate perturbation (degC
    added to tavg/tmin/tmax, precip multiplied), applied to the spin-up and the envelope
    alike, until the WGEN daily scenario weather is supplied."""
    from .data import with_cell_dedup
    from .forward import routing_uh
    from .multi_timescale import ENVELOPE_END, ENVELOPE_START
    from .spinup import spin_state, stream_rows

    mode = _component_mode(components)
    parts = mode == "parts"
    t0 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_START)))
    t1 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_END))) + 1
    if dedup_cells:
        dom = with_cell_dedup(dom, x)
    Wa = torch.as_tensor(W_arc).to(dom.device, dom.dtype)
    bad = torch.zeros(dom.n_hru, dtype=torch.bool, device=dom.device)
    pert = {}
    if temp_delta or precip_scale != 1.0:
        pert = dict(temp_delta=float(temp_delta), precip_scale=float(precip_scale))

    def agg(flow: torch.Tensor) -> torch.Tensor:
        nan_rows = torch.isnan(flow).any(dim=1)
        if nan_rows.any():
            bad.logical_or_(nan_rows)
            flow = torch.nan_to_num(flow, nan=0.0)
        return torch.cat([Wa @ flow, dom.W @ flow])

    def agg_parts(flow: torch.Tensor, *comps: torch.Tensor) -> torch.Tensor:
        # the total exactly as agg() forms it; the components zeroed on the total's NaN days
        nan = torch.isnan(flow)
        if nan.any():
            comps = tuple(c.masked_fill(nan, 0.0) for c in comps)
        return torch.cat([agg(flow)] + [Wa @ c for c in comps])

    net.eval()
    with torch.no_grad():
        out = net(dom.phys_x(x))
        canopy = out.pop("_canopy", None)
        uh = routing_uh(out, dom.flowlen, row_cell=dom.row_cell)
        skw = dict(pert, parts=True) if parts else pert
        state, how = spin_state(dom, cfg, out, uh, canopy, t0, mode=spinup,
                                agg=lambda f: dom.W @ torch.nan_to_num(f, nan=0.0), **skw)
        print(f"tier2: {how}", flush=True)
        if mode:
            rows, _ = stream_rows(dom, cfg, out, uh, canopy, t0, t1, state,
                                  lambda f, *c: agg_parts(f, *c).cpu(), components=True, **skw)
        else:
            rows, _ = stream_rows(dom, cfg, out, uh, canopy, t0, t1, state, lambda f: agg(f).cpu(),
                                  **pert)
    rows = rows.double().numpy()
    na, nb = len(W_arc), dom.W.shape[0]
    res = (rows[:na], rows[na:na + nb], t0, t1, bad.cpu().numpy())
    if mode:
        n_comp = 2 + (len(PART_COLUMNS) if parts else 0)
        res = res + tuple(rows[na + nb + k * na:na + nb + (k + 1) * na] for k in range(n_comp))
    return res


def _replicate_domain(dom, S: int):
    """``dom`` with every HRU row repeated ``S`` times, scenario-major (rows ``s*N .. s*N+N-1``
    are scenario ``s``'s copy of rows ``0 .. N-1``).  Forcing, LAI and the climate-state index
    stay per cell (gathered through the tiled ``cell_idx``); ``W`` is replaced by an empty
    placeholder because :func:`stream_batch` aggregates each scenario's slice itself."""
    import dataclasses
    if dom.dedup is not None:
        raise ValueError("scenario batching does not combine with dedup_cells")
    return dataclasses.replace(
        dom, hrus=pd.concat([dom.hrus] * S, ignore_index=True), cell_idx=np.tile(dom.cell_idx, S),
        lat_rad=dom.lat_rad.repeat(S), elev=dom.elev.repeat(S), flowlen=dom.flowlen.repeat(S),
        veg_frac=None if dom.veg_frac is None else dom.veg_frac.repeat(S),
        W=torch.zeros((0, dom.n_hru * S), dtype=dom.dtype, device=dom.device))


def _tile_rows(v, S: int):
    """Repeat a per-row tensor (or a dict / tuple of them) ``S`` times along dim 0."""
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: _tile_rows(t, S) for k, t in v.items()}
    if isinstance(v, (tuple, list)):
        return type(v)(_tile_rows(t, S) for t in v)
    return v.repeat(S, *([1] * (v.dim() - 1)))


def stream_batch(net, x, dom, cfg, W_arc: np.ndarray, scenarios, *, spinup: str = "cycle",
                 components: bool | str = False, window: int = 128) -> list[tuple]:
    """:func:`stream` for several uniform climate perturbations at once: ``scenarios`` is a
    list of ``(temp_delta, precip_scale)``; returns one :func:`stream` result tuple per
    scenario, in order.  The domain's rows are replicated once per scenario
    (:func:`_replicate_domain`) and each copy's forcing is perturbed by its own values
    inside :func:`sacsma.dpl.spinup.stream_rows`, so one eager pass (spin-up included)
    replaces ``len(scenarios)`` passes.  The parameter net and the routing kernels are
    evaluated ONCE on the original rows and tiled, so every scenario sees bit-identical
    parameters; each scenario's flow slice is aggregated exactly as :func:`stream` does.
    ``window`` (days per streamed chunk) bounds device memory only."""
    from .forward import routing_uh
    from .multi_timescale import ENVELOPE_END, ENVELOPE_START
    from .spinup import spin_state, stream_rows

    mode = _component_mode(components)
    parts = mode == "parts"
    S, N = len(scenarios), dom.n_hru
    t0 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_START)))
    t1 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_END))) + 1
    Wa = torch.as_tensor(W_arc).to(dom.device, dom.dtype)
    Wb = dom.W
    domS = _replicate_domain(dom, S)
    bad = torch.zeros(S * N, dtype=torch.bool, device=dom.device)
    td = torch.as_tensor([float(s[0]) for s in scenarios], dtype=dom.dtype,
                         device=dom.device).repeat_interleave(N)
    ps = torch.as_tensor([float(s[1]) for s in scenarios], dtype=dom.dtype,
                         device=dom.device).repeat_interleave(N)
    pert = dict(temp_delta=td, precip_scale=ps)

    def sl(t: torch.Tensor, s: int) -> torch.Tensor:
        return t[s * N:(s + 1) * N]

    def agg_one(flow: torch.Tensor, s: int) -> torch.Tensor:      # stream()'s agg on one slice
        nan_rows = torch.isnan(flow).any(dim=1)
        if nan_rows.any():
            bad[s * N:(s + 1) * N].logical_or_(nan_rows)
            flow = torch.nan_to_num(flow, nan=0.0)
        return torch.cat([Wa @ flow, Wb @ flow])

    def agg_parts_one(flow: torch.Tensor, comps, s: int) -> torch.Tensor:
        nan = torch.isnan(flow)
        if nan.any():
            comps = tuple(c.masked_fill(nan, 0.0) for c in comps)
        return torch.cat([agg_one(flow, s)] + [Wa @ c for c in comps])

    def agg(flow: torch.Tensor) -> torch.Tensor:
        return torch.cat([agg_one(sl(flow, s), s) for s in range(S)]).cpu()

    def agg_parts(flow: torch.Tensor, *comps: torch.Tensor) -> torch.Tensor:
        return torch.cat([agg_parts_one(sl(flow, s), tuple(sl(c, s) for c in comps), s)
                          for s in range(S)]).cpu()

    net.eval()
    with torch.no_grad():
        out = net(dom.phys_x(x))
        canopy = out.pop("_canopy", None)
        uh = routing_uh(out, dom.flowlen, row_cell=dom.row_cell)
        outS, canopyS, uhS = _tile_rows(out, S), _tile_rows(canopy, S), _tile_rows(uh, S)
        skw = dict(pert, parts=True) if parts else pert
        state, how = spin_state(
            domS, cfg, outS, uhS, canopyS, t0, mode=spinup, window=window,
            agg=lambda f: torch.cat([Wb @ torch.nan_to_num(sl(f, s), nan=0.0) for s in range(S)]),
            **skw)
        print(f"tier2 batch ({S} scenarios): {how}", flush=True)
        if mode:
            rows, _ = stream_rows(domS, cfg, outS, uhS, canopyS, t0, t1, state, agg_parts,
                                  components=True, window=window, **skw)
        else:
            rows, _ = stream_rows(domS, cfg, outS, uhS, canopyS, t0, t1, state, agg,
                                  window=window, **pert)
    rows = rows.double().numpy()
    na, nb = len(W_arc), Wb.shape[0]
    n_comp = (2 + (len(PART_COLUMNS) if parts else 0)) if mode else 0
    per = na + nb + n_comp * na
    badn = bad.cpu().numpy()
    res = []
    for s in range(S):
        r = rows[s * per:(s + 1) * per]
        one = (r[:na], r[na:na + nb], t0, t1, badn[s * N:(s + 1) * N])
        if mode:
            one = one + tuple(r[na + nb + k * na:na + nb + (k + 1) * na] for k in range(n_comp))
        res.append(one)
    return res


def _drop_bad_cells(sim: np.ndarray, W: np.ndarray, bad: np.ndarray, hrus: pd.DataFrame,
                    label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Renormalise aggregates whose cells went NaN (those cells contributed zero) and
    report them.  Returns (sim, n_bad_cells per aggregate, bad weight fraction per aggregate)."""
    n_bad = (W[:, bad] > 0).sum(axis=1)
    w_bad = W[:, bad].sum(axis=1)
    if bad.any():
        cells = hrus.loc[bad, ["basin", "key", "elev", "flowlen"]]
        print(f"tier2: {int(bad.sum())} {label} cell row(s) returned NaN and were dropped from their "
              f"aggregates (parameters extrapolated to their bounds):\n{cells.to_string(index=False)}", flush=True)
        keep = w_bad < 1.0
        sim[keep] = sim[keep] / (1.0 - w_bad[keep])[:, None]
        sim[~keep] = np.nan
    return sim, n_bad, w_bad


def _renorm_like_bad(sim: np.ndarray, W: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """The renormalisation of :func:`_drop_bad_cells`, silently, for the component
    aggregates (the same divisor per aggregate as their total)."""
    if bad.any():
        w_bad = W[:, bad].sum(axis=1)
        keep = w_bad < 1.0
        sim[keep] = sim[keep] / (1.0 - w_bad[keep])[:, None]
        sim[~keep] = np.nan
    return sim


def _reuse_flowlens(cells: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Add ``flowlen`` / ``flowlen_method`` to the extension HRU rows from a previous run's
    ``tier2_extension_cells.csv`` instead of tracing again; the two must hold the same
    (arc, cell) rows at the same positions (the extension footprint is a pure function of
    the polygons, the region grid and the trained entities), each (basin, key) once, and the
    merge must keep the row count (a duplicated key would add HRU rows and change the
    aggregation weights)."""
    old = pd.read_csv(path, float_precision="round_trip")
    got = cells.merge(old[["basin", "key", "flowlen", "flowlen_method"]], on=["basin", "key"], how="left")
    same = (len(old) == len(cells)
            and len(got) == len(cells)
            and not old.duplicated(["basin", "key"]).any()
            and not cells.duplicated(["basin", "key"]).any()
            and (old["basin"].to_numpy() == cells["basin"].to_numpy()).all()
            and (old["key"].to_numpy() == cells["key"].to_numpy()).all()
            and np.allclose(old["area_weight"].to_numpy(), cells["area_weight"].to_numpy(), rtol=1e-9, atol=0))
    if not same or got["flowlen"].isna().any():
        raise ValueError(f"{path}: its extension rows do not match this run's "
                         f"({len(old)} vs {len(cells)} rows, {len(got)} after the merge; duplicated "
                         f"(basin, key): {int(old.duplicated(['basin', 'key']).sum())} there, "
                         f"{int(cells.duplicated(['basin', 'key']).sum())} here) — trace again instead")
    print(f"tier2: reused traced flow lengths for {len(got)} extension cells from {path}", flush=True)
    return got


def _monthly_taf(depth: np.ndarray, dates: pd.DatetimeIndex, area_mi2: np.ndarray) -> pd.DataFrame:
    """(A, T) daily depth -> month x arc TAF over complete calendar months."""
    df = pd.DataFrame(depth.T, index=dates)
    m = df.resample("ME").sum(min_count=1)
    complete = df.resample("ME").count().iloc[:, 0].to_numpy() >= m.index.days_in_month
    m[~complete] = np.nan
    m.index = m.index.to_period("M")
    return m * area_mi2[None, :] * AF_PER_MM_MI2 / 1000.0


def _score(months: pd.PeriodIndex, sim: np.ndarray, ref: np.ndarray) -> dict:
    m = np.isfinite(sim) & np.isfinite(ref)
    out = dict(n_months=int(m.sum()), kge=np.nan, nse=np.nan, pbias=np.nan, r=np.nan,
               alpha=np.nan, beta=np.nan, seas_mismatch=np.nan, ct_diff=np.nan,
               sim_taf_yr=np.nan, ref_taf_yr=np.nan)
    if m.sum() < 12:
        if np.isfinite(sim).sum() >= 12:
            out["sim_taf_yr"] = 12.0 * np.nanmean(sim)
        return out
    s, r = sim[m], ref[m]
    d = months[m].to_timestamp()
    out.update(kge=kge(s, r), nse=nse(s, r), pbias=pbias(s, r), r=pearson(s, r),
               alpha=s.std() / r.std() if r.std() > 0 else np.nan,
               beta=s.mean() / r.mean() if r.mean() != 0 else np.nan,
               seas_mismatch=seasonal_mismatch(d, s, r),
               ct_diff=center_of_timing(d, s) - center_of_timing(d, r),
               sim_taf_yr=12.0 * s.mean(), ref_taf_yr=12.0 * r.mean())
    return out


def anchor_rescaled(monthly: pd.DataFrame, data_dir: str | Path = "data", *,
                    windows: dict[str, tuple[str, str]], own: dict | None = None,
                    own_window: str | None = None, wy_range: tuple[int, int] = (1950, 2015)
                    ) -> pd.DataFrame:
    """Eval-only ANCHOR-RESCALED arc scores beside the absolute ones.

    ``monthly`` is ``tier2_monthly.csv`` (``arc, month, sim_taf, ref_taf``).  Per closure group
    of ``data/calsim/arc_hierarchy.csv`` and complete water year, every simulated member arc is
    scaled by anchor / simulated arc sum (:func:`sacsma.dpl.calsim_arcs.close_water_years`,
    ``mode="proportional"``: the arcs' volume-weighted shares of the anchor, the anchor being
    CalSim3 FLOW-UNIMPAIRED or the DWR unimpaired record of the group), so the score measures the
    split of the anchor's water-year volume among arcs and months, not the system volume.  Groups
    left out of the closure (``CLOSURE_EXCLUDED``), groups with a member polygon this run did not
    simulate, arcs in no group and water years the closure skips have no rescaled value.  Both scores are over the same months (those with a rescaled
    value) per window; ``own`` (arc -> months) adds ``own_window``, the window of that name in
    ``windows`` restricted to each arc's own months."""
    from .calsim_arcs import (close_water_years, closure_members, load_anchor_taf, load_hierarchy,
                              water_year)
    mon = monthly.copy()
    mon["period"] = pd.PeriodIndex(mon["month"], freq="M")
    sim = mon.pivot(index="period", columns="arc", values="sim_taf")
    ref = mon.pivot(index="period", columns="arc", values="ref_taf")
    hier = load_hierarchy(data_dir)
    anchor = load_anchor_taf(data_dir, hier)
    # a group closes only when every member with a polygon is simulated: a partial run (e.g.
    # --no-extend) would hand the whole anchor to the few arcs it has
    poly = set(hier.loc[hier["has_polygon"].astype(bool), "arc"])
    part = [g for g, mem in closure_members(hier).items()
            if any(m in poly and m not in sim.columns for m in mem)]
    anchor = anchor.drop(columns=[g for g in part if g in anchor.columns])
    closed, diag = close_water_years(sim, anchor, hier, wy_range=wy_range, mode="proportional")
    grp = hier.dropna(subset=["closure_group"]).set_index("arc")["closure_group"]
    done = set(zip(diag["group"], diag["wy"], strict=True)) if len(diag) else set()
    wy = water_year(sim.index)
    rows = []
    specs = dict(windows)
    if own is not None and own_window:
        specs[own_window + OWN_SUFFIX] = windows[own_window]
    for arc in sim.columns:
        g = grp.get(arc)
        ok = (np.array([(g, int(y)) in done for y in wy]) if isinstance(g, str)
              else np.zeros(len(wy), bool))
        for wname, (m0, m1) in specs.items():
            idx = pd.period_range(m0, m1, freq="M")
            pos = sim.index.get_indexer(idx)
            keep = np.where(pos >= 0, ok[np.clip(pos, 0, None)], False)
            if wname.endswith(OWN_SUFFIX):
                keep &= idx.isin(own.get(arc, pd.PeriodIndex([], freq="M")))
            r = ref[arc].reindex(idx).to_numpy()
            s_abs = np.where(keep, sim[arc].reindex(idx).to_numpy(), np.nan)
            s_res = np.where(keep, closed[arc].reindex(idx).to_numpy(), np.nan)
            a_ = _score(idx, s_abs, np.where(keep, r, np.nan))
            b_ = _score(idx, s_res, np.where(keep, r, np.nan))
            keys = ("kge", "nse", "pbias", "r", "alpha", "beta")
            rows.append(dict(arc=arc, closure_group=g if isinstance(g, str) else "", window=wname,
                             n_months=a_["n_months"], **{f"{k}_abs": a_[k] for k in keys},
                             **{f"{k}_resc": b_[k] for k in keys}))
    return pd.DataFrame(rows)


def score_run(ckpt: str | Path, data_dir: str | Path = "data", *, device=None, run_dir=None,
              extend: bool = True, tiles_dir: str | Path = "tmp/hydrosheds", out: Path | None = None,
              trace_python: str | None = None, spinup: str = "cycle",
              components: bool | str = False, temp_delta: float = 0.0, precip_scale: float = 1.0,
              extension_cells: str | Path | None = None, dedup_cells: bool = False,
              scenarios=None, outs=None, batch_window: int = 128):
    """Returns (metrics, monthly, arcs, not_sim, entity_check).  ``extend`` simulates the
    arcs outside every trained footprint on their own region cells (``basis = extrapolated``);
    ``spinup`` and ``dedup_cells`` are the evaluator's (:func:`stream`).

    Opt-in extras (all off by default, leaving the outputs unchanged): ``components`` (True
    / ``"fastslow"``) also returns, as a sixth element, the per-arc monthly routed FAST and
    SLOW components (``[arc, month, fast_taf, slow_taf, total_taf]``, ``total_taf`` =
    ``sim_taf``); ``"parts"`` adds the four routed runoff parts before them
    (``quick_taf, interflow_taf, supplemental_taf, primary_taf``; ``quick + interflow =
    fast``, ``supplemental + primary = slow``).  All are NET of the riparian et4 channel-ET
    deduction (:func:`stream`).  ``temp_delta`` / ``precip_scale`` re-run under a
    PLACEHOLDER uniform climate perturbation (:func:`stream`), which skips the archived
    entity check; ``extension_cells`` reuses a previous run's ``tier2_extension_cells.csv``
    flow lengths instead of tracing.  When any extra or ``dedup_cells`` is on and ``out`` is
    given, ``tier2_run_info.json`` records them (the perturbation with its PLACEHOLDER flag),
    so a perturbed folder cannot be mistaken for a baseline one.

    ``scenarios`` (a list of ``(temp_delta, precip_scale)``, with ``outs`` one output
    folder each) runs them all in ONE batched forward pass (:func:`stream_batch`;
    ``temp_delta`` / ``precip_scale`` are then ignored) and returns a list with one
    result per scenario, each exactly what the single-scenario call returns.
    ``batch_window`` is the batched stream's chunk length (device memory only)."""
    from .evaluate import load_net_from_checkpoint
    mode = _component_mode(components)
    batch = scenarios is not None
    if batch:
        if outs is None or len(outs) != len(scenarios):
            raise ValueError("scenarios needs one output folder each (outs)")
        if dedup_cells:
            raise ValueError("scenario batching does not combine with dedup_cells")
        todo = [(float(t), float(s), None if o is None else Path(o))
                for (t, s), o in zip(scenarios, outs, strict=True)]
        print(f"tier2: {len(todo)} PLACEHOLDER uniform climate scenarios in one batched pass "
              "(tavg/tmin/tmax + degC, precip x; spin-up included): "
              + ", ".join(f"{t:+g}C x{s:g}" for t, s, _ in todo), flush=True)
    else:
        todo = [(float(temp_delta), float(precip_scale), out)]
    for t_, s_, o_ in todo:
        pert_ = bool(t_) or s_ != 1.0
        if pert_ and not batch:
            print(f"tier2: PLACEHOLDER climate perturbation — tavg/tmin/tmax {t_:+g} degC, "
                  f"precip x{s_:g}, spin-up included (uniform stand-in until the WGEN "
                  "daily scenario weather is supplied)", flush=True)
        if o_ is not None and (mode or pert_ or extension_cells is not None or dedup_cells):
            _write_run_info(o_, ckpt=ckpt, data_dir=data_dir, spinup=spinup, extend=extend,
                            dedup_cells=dedup_cells, components=mode, temp_delta=t_,
                            precip_scale=s_, extension_cells=extension_cells)
    net, x, dom, cfg, ck = load_net_from_checkpoint(ckpt, data_dir, device=device)
    ho = tuple(int(v) for v in (getattr(cfg, "holdout_wy", ()) or ()))
    catch = rim_polygons(data_dir)
    mapping = cell_arc_overlap(dom.hrus, catch)
    parent = parent_entities(dom.basins, data_dir)
    W_arc, arcs = arc_weights(dom, catch, mapping, parent)
    arcs["basis"] = "trained entity"
    # every arc's share of area on the cells this run trained on, from the full region grid
    region_map = region_arc_overlap(catch, data_dir)
    share = trained_cell_share(region_map, dom.hrus["key"].unique())
    arcs["trained_cell_frac"] = arcs["arc"].map(share).fillna(0.0)
    print(f"tier2: {len(catch)} rim polygons; {len(arcs)} arcs inside the {len(dom.basins)} trained "
          f"entities; {len(dom.hrus)} HRU rows on {dom.device.type}", flush=True)
    # the extension cells (arcs outside every trained footprint) are prepared ahead of the streams
    ext = None
    if extend:
        ext = uncovered_arc_cells(catch, parent, data_dir, region_map=region_map)
        if len(ext):
            if extension_cells is not None:
                ext = _reuse_flowlens(ext, extension_cells)
            else:
                ext = _flowlens_for(ext, tiles_dir, trace_python)
            for _, _, o_ in todo:
                if o_ is not None:
                    ext.to_csv(o_ / "tier2_extension_cells.csv", index=False)
            net2, x2, dom2, cfg2 = _net_for_hrus(ckpt, ext, data_dir, device=device)
            W2 = dom2.W.cpu().numpy()
        else:
            ext = None
    print("tier2: streaming the envelope ...", flush=True)

    def _single_skw(t_, s_):
        k = {"components": mode} if mode else {}
        if bool(t_) or s_ != 1.0:
            k.update(temp_delta=t_, precip_scale=s_)
        return k

    if batch:
        res1 = stream_batch(net, x, dom, cfg, W_arc, [(t_, s_) for t_, s_, _ in todo],
                            spinup=spinup, components=mode, window=batch_window)
    else:
        res1 = [stream(net, x, dom, cfg, W_arc, spinup=spinup, dedup_cells=dedup_cells,
                       **_single_skw(todo[0][0], todo[0][1]))]
    res2 = [None] * len(todo)
    if ext is not None:
        print(f"tier2: extrapolating {len(dom2.basins)} uncovered arcs on {len(dom2.hrus)} region "
              f"cells ...", flush=True)
        if batch:
            res2 = stream_batch(net2, x2, dom2, cfg2, W2, [(t_, s_) for t_, s_, _ in todo],
                                spinup=spinup, components=mode, window=batch_window)
        else:
            res2 = [stream(net2, x2, dom2, cfg2, W2, spinup=spinup, dedup_cells=dedup_cells,
                           **_single_skw(todo[0][0], todo[0][1]))]

    def _finish(sres, sres2, out, perturbed, arcs):
        arcs = arcs.copy()
        sim_arc, sim_ent, t0, t1, bad = sres[:5]
        sim_arc, arcs["n_nan_cells"], arcs["nan_weight_frac"] = _drop_bad_cells(
            sim_arc, W_arc, bad, dom.hrus, "trained-footprint")
        if mode:   # [fast, slow] (+ [quick, interflow, supplemental, primary]) arc depths
            comp_arc = [_renorm_like_bad(c, W_arc, bad) for c in sres[5:]]
        dates = dom.dates[t0:t1]
        if sres2 is not None:
            sim_ext, _, t0e, t1e, bad2 = sres2[:5]
            assert (t0e, t1e) == (t0, t1)
            sim_ext, n_bad2, w_bad2 = _drop_bad_cells(sim_ext, W2, bad2, dom2.hrus, "extrapolated")
            if mode:
                comp_arc = [np.concatenate([c, _renorm_like_bad(c2, W2, bad2)], axis=0)
                            for c, c2 in zip(comp_arc, sres2[5:], strict=True)]
            sq_mi = catch.groupby("arc")["sq_mi"].sum()
            cov = ext.groupby("basin")["area_weight"].sum()
            rows = [dict(arc=a, node=a[2:], entity="", sq_mi=float(sq_mi[a]), covered_mi2=float(cov[a]),
                         cover_frac=float(cov[a] / sq_mi[a]), n_cells=int((ext.basin == a).sum()),
                         basis="extrapolated", trained_cell_frac=float(share.get(a, 0.0)),
                         n_nan_cells=int(n_bad2[j]), nan_weight_frac=float(w_bad2[j]))
                    for j, a in enumerate(dom2.basins)]
            arcs = pd.concat([arcs, pd.DataFrame(rows)], ignore_index=True)
            sim_arc = np.concatenate([sim_arc, sim_ext], axis=0)
        # the daily arc series, so scoring and figures can be redone without the forward pass
        if out is not None:
            np.savez_compressed(out / "tier2_sim_daily.npz", arc=np.array(arcs["arc"]),
                                date=dates.to_numpy().astype("datetime64[D]").astype(str),
                                sim_mm=sim_arc.astype(np.float32))

        entity_check = None
        if perturbed and run_dir is not None:
            print("tier2: entity check against the archived sim_daily_mm.npz skipped (perturbed "
                  "forcing)", flush=True)
        elif run_dir is not None and (Path(run_dir) / "sim_daily_mm.npz").exists():
            z = np.load(Path(run_dir) / "sim_daily_mm.npz")
            ref = pd.DataFrame(np.asarray(z["sim_mm"]).T.astype(float),
                               columns=list(z["entity_id"]))
            mine = pd.DataFrame(sim_ent.T, columns=list(dom.basins))
            common = [b for b in dom.basins if b in ref.columns]
            d = (mine[common].to_numpy() - ref[common].to_numpy())
            entity_check = dict(n_entities=len(common), max_abs_diff_mm=float(np.nanmax(np.abs(d))),
                                rel_rmse=float(np.sqrt(np.nanmean(d ** 2))
                                               / np.nanmean(ref[common].to_numpy())))
            print(f"tier2: entity re-run vs archived sim_daily_mm.npz over {len(common)} entities: "
                  f"max |diff| {entity_check['max_abs_diff_mm']:.3g} mm/day, relative RMSE "
                  f"{entity_check['rel_rmse']:.2e}", flush=True)

        inflow, _ = load_references(data_dir)
        xw = load_crosswalk(data_dir).set_index("arc")
        sets = load_sets(data_dir)
        # the smallest set names an arc that several list (I_SHSTA -> SHA)
        in_set = arc_to_set(sets)
        train_win = registry_windows(data_dir)
        taf = _monthly_taf(sim_arc, dates, arcs["sq_mi"].to_numpy())
        taf.columns = list(arcs["arc"])
        ho_win = holdout_windows(ho) if ho else None
        own = {}
        if ho:
            from .calsim_arcs import own_record_months
            own = own_record_months(data_dir, ho)
        rows, monthly = [], []
        for a in arcs.itertuples(index=False):
            ref = inflow[a.arc] if a.arc in inflow.columns else pd.Series(np.nan, index=taf.index)
            # in-sample window: the parent entity's training window; for an extrapolated arc there
            # is none, so the monthly family's window stands in as the comparison period
            t0m, t1m = (train_win[a.entity] if a.entity
                        else (pd.Period("1984-10", "M"), pd.Period("2014-09", "M")))
            if ho_win is not None:
                # a holdout run: the held-out water years (all months, and the arc's own
                # gauge-record months among them), WY1950-84 as mixed, and the train window
                # without the holdout
                hw = next(iter(ho_win))
                windows = dict(ho_win)
                windows[hw + OWN_SUFFIX] = ho_win[hw]
                windows["train"] = (str(t0m), str(t1m))
            else:
                windows = {VALIDATION_WINDOW: WINDOWS[VALIDATION_WINDOW],
                           "train": (str(t0m), str(t1m))}
            for wname, (m0, m1) in windows.items():
                idx = pd.period_range(m0, m1, freq="M")
                s_w, r_w = taf[a.arc].reindex(idx).to_numpy(), ref.reindex(idx).to_numpy()
                extra = {}
                if ho_win is not None:
                    if wname == "train":
                        drop = holdout_month_mask(idx, ho)
                    elif wname.endswith(OWN_SUFFIX):
                        drop = ~idx.isin(own.get(a.arc, pd.PeriodIndex([], freq="M")))
                    else:
                        drop = np.zeros(len(idx), dtype=bool)
                    s_w, r_w = np.where(drop, np.nan, s_w), np.where(drop, np.nan, r_w)
                    extra = {"excluded_wy": f"{ho[0]}-{ho[1]}" if wname == "train" else ""}
                met = _score(idx, s_w, r_w)
                rows.append(dict(arc=a.arc, node=a.node, entity=a.entity, basis=a.basis,
                                 trained_cell_frac=float(a.trained_cell_frac),
                                 system=xw["system"].get(a.arc, ""),
                                 tier1_set=in_set.get(a.arc, ""),
                                 sq_mi=a.sq_mi, cover_frac=a.cover_frac, n_cells=a.n_cells,
                                 n_nan_cells=int(a.n_nan_cells),
                                 nan_weight_frac=float(a.nan_weight_frac),
                                 has_series=a.arc in inflow.columns, window=wname,
                                 win_start=m0, win_end=m1, **met, **extra))
            monthly.append(pd.DataFrame({"arc": a.arc, "month": taf.index.astype(str),
                                         "sim_taf": taf[a.arc].to_numpy(),
                                         "ref_taf": ref.reindex(taf.index).to_numpy()}))
        metrics = pd.DataFrame(rows)
        # arcs with a series that this run does not simulate
        simulated = set(arcs["arc"])
        missing = [a for a in catch["arc"].unique() if a not in simulated and a in inflow.columns]
        v0, v1 = WINDOWS[VALIDATION_WINDOW] if ho_win is None else next(iter(ho_win.values()))
        idx = pd.period_range(v0, v1, freq="M")
        not_sim = pd.DataFrame({
            "arc": missing,
            "system": [xw["system"].get(a, "") for a in missing],
            "sq_mi": [float(catch.loc[catch["arc"] == a, "sq_mi"].sum()) for a in missing],
            "ref_taf_yr": [12.0 * inflow[a].reindex(idx).mean() for a in missing]})
        if mode:
            area = arcs["sq_mi"].to_numpy()
            names = ["fast", "slow"] + (list(PART_COLUMNS) if mode == "parts" else [])
            tafs = dict(zip(names, (_monthly_taf(c, dates, area) for c in comp_arc), strict=True))
            order = (list(PART_COLUMNS) if mode == "parts" else []) + ["fast", "slow"]
            comp = pd.concat([pd.DataFrame({"arc": a, "month": taf.index.astype(str),
                                            **{f"{n}_taf": tafs[n].iloc[:, j].to_numpy()
                                               for n in order},
                                            "total_taf": taf[a].to_numpy()})
                              for j, a in enumerate(arcs["arc"])], ignore_index=True)
            return metrics, pd.concat(monthly, ignore_index=True), arcs, not_sim, entity_check, comp
        return metrics, pd.concat(monthly, ignore_index=True), arcs, not_sim, entity_check

    results = [_finish(r1, r2, o_, bool(t_) or s_ != 1.0, arcs)
               for (t_, s_, o_), r1, r2 in zip(todo, res1, res2, strict=True)]
    return results if batch else results[0]


def _write_run_info(out: Path, **info) -> Path:
    """``tier2_run_info.json``: the opt-in extras a tier-2 folder was made with.  A
    perturbed run is flagged ``placeholder_perturbation`` (the uniform dT / precip-scale
    stand-in for the WGEN scenario weather): its CSVs are still scored against HISTORICAL
    CalSim3, so its skill columns are not a baseline."""
    import json
    from datetime import datetime, timezone
    perturbed = bool(info["temp_delta"]) or info["precip_scale"] != 1.0
    rec = dict(
        tool="sacsma.dpl.calsim_tier2",
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        checkpoint=str(info["ckpt"]), data_dir=str(info["data_dir"]), spinup=info["spinup"],
        extend=bool(info["extend"]), dedup_cells=bool(info["dedup_cells"]),
        components=info["components"],
        component_columns=(None if not info["components"] else
                           (list(PART_COLUMNS) if info["components"] == "parts" else [])
                           + ["fast", "slow", "total"]),
        components_note=("routed runoff components, all NET of SAC-SMA's riparian et4 "
                         "channel-ET deduction and dry-channel clamp" if info["components"] else None),
        extension_cells=None if info["extension_cells"] is None else str(info["extension_cells"]),
        temp_delta_degC=info["temp_delta"], precip_scale=info["precip_scale"],
        perturbed=perturbed, placeholder_perturbation=perturbed,
        perturbation_note=("PLACEHOLDER uniform climate perturbation (degC added to "
                           "tavg/tmin/tmax, precip multiplied, spin-up included) until the WGEN "
                           "daily scenario weather is supplied; tier2_metrics/monthly still score "
                           "against historical CalSim3" if perturbed else None))
    path = Path(out) / "tier2_run_info.json"
    path.write_text(json.dumps(rec, indent=1) + "\n")
    return path


def _stat_line(m: pd.DataFrame) -> str:
    w = m.ref_taf_yr / m.ref_taf_yr.sum()
    return (f"KGE median {m.kge.median():.3f} mean {m.kge.mean():.3f} volume-weighted "
            f"{(m.kge * w).sum():.3f} | NSE median {m.nse.median():.3f} | |pbias| median "
            f"{m.pbias.abs().median():.1f}% | seasonal mismatch median {m.seas_mismatch.median():.3f} | "
            f"KGE >= 0.7: {(m.kge >= 0.7).sum()}, 0.5-0.7: {((m.kge >= 0.5) & (m.kge < 0.7)).sum()}, "
            f"< 0.5: {(m.kge < 0.5).sum()} | volume {m.sim_taf_yr.sum():,.0f} vs {m.ref_taf_yr.sum():,.0f} TAF/yr")


def summarize(metrics: pd.DataFrame, not_sim: pd.DataFrame, window: str = VALIDATION_WINDOW) -> str:
    m = metrics[(metrics.window == window) & metrics.has_series & metrics.kge.notna()]
    tot_ref = m.ref_taf_yr.sum() + not_sim.ref_taf_yr.sum()
    lines = [f"tier 2, {window}: {len(m)} rim arcs scored ({m.ref_taf_yr.sum():,.0f} of "
             f"{tot_ref:,.0f} TAF/yr of rim inflow); {len(not_sim)} arcs with a series not simulated "
             f"({not_sim.ref_taf_yr.sum():,.0f} TAF/yr)"]
    lines.append("  all scored:       " + _stat_line(m))
    for basis, g in m.groupby("basis"):
        lines.append(f"  {basis:17s} " + f"(n={len(g)}) " + _stat_line(g))
    ext = m[m.basis == "extrapolated"]
    if len(ext) and "trained_cell_frac" in ext.columns:
        # the regionalization test is the arcs on cells the loss never saw; the others overlap a
        # trained creek footprint and only their routing to the arc is new
        for label, g in (("  on unseen cells ", ext[ext.trained_cell_frac <= 0]),
                         ("  on trained cells", ext[ext.trained_cell_frac > 0])):
            if len(g):
                lines.append(f"  {label:17s} " + f"(n={len(g)}) " + _stat_line(g))
    m = m[m.basis == "trained entity"]
    g = m.groupby("entity").agg(n=("arc", "size"), kge_med=("kge", "median"), kge_min=("kge", "min"),
                                pbias_med=("pbias", "median"), sim=("sim_taf_yr", "sum"), ref=("ref_taf_yr", "sum"))
    g["vol_ratio"] = g.sim / g.ref
    lines.append("  by parent entity (n arcs, median KGE, min KGE, median pbias, volume ratio):")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        lines.append(g.sort_values("kge_med", ascending=False).round(3).to_string())
        cols = ["arc", "entity", "system", "tier1_set", "sq_mi", "cover_frac", "kge", "nse", "pbias", "r",
                "alpha", "beta", "seas_mismatch", "sim_taf_yr", "ref_taf_yr"]
        lines.append("  worst 15 trained-footprint arcs by KGE:")
        lines.append(m.sort_values("kge")[cols].head(15).round(3).to_string(index=False))
        e = metrics[(metrics.window == window) & metrics.has_series & metrics.kge.notna()
                    & (metrics.basis == "extrapolated")]
        if len(e):
            ecols = cols[:6] + (["trained_cell_frac"] if "trained_cell_frac" in e.columns else []) + cols[6:]
            lines.append("  extrapolated arcs (no trained entity lists them; trained_cell_frac = share of the "
                         "arc's area on cells the run trained on), by reference volume:")
            lines.append(e.sort_values("ref_taf_yr", ascending=False)[ecols].round(3).to_string(index=False))
        if len(not_sim):
            lines.append("  not simulated:")
            lines.append(not_sim.sort_values("ref_taf_yr", ascending=False).round(1).to_string(index=False))
    return "\n".join(lines)


def maps(metrics: pd.DataFrame, out: Path, label: str, data_dir: str | Path = "data",
         window: str = VALIDATION_WINDOW) -> None:
    from ..calsim.compare import _arc_choropleth
    m = metrics[(metrics.window == window) & metrics.has_series].set_index("arc")
    _arc_choropleth(data_dir, m["kge"], f"Tier 2 — KGE per rim arc, {window}  [{label}]", "KGE",
                    out / f"tier2_kge_{window}.png", cmap="plasma", vmin=0.0, vmax=1.0)
    _arc_choropleth(data_dir, m["pbias"], f"Tier 2 — volume bias per rim arc, {window}  [{label}]",
                    "pbias (%)", out / f"tier2_pbias_{window}.png", cmap="BrBG", vmin=-50.0, vmax=50.0)


#: single-arc tier-1 sets share one regime figure instead of one figure each
_SINGLE_ARC_GROUP = "single-arc sets"


def regime_figures(out: Path, data_dir: str | Path = "data", label: str = "",
                   window: str = VALIDATION_WINDOW) -> list[Path]:
    """One mean-monthly-regime figure per tier-1 set (the anchor system where one exists),
    every rim arc of the set as a panel — simulated arcs show CalSim3 vs dPL with their score,
    arcs this run does not simulate show the CalSim3 regime alone on a shaded panel.  Arcs in
    no tier-1 set go to an "unconstrained" figure.  Reads the CSVs written by :func:`main`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not (out / "tier2_monthly.csv").exists():
        # the monthly table is not tracked: a fresh clone holds the metrics only
        print(f"tier2: no {out / 'tier2_monthly.csv'} (written by the forward pass, not tracked) — "
              "regime figures skipped", flush=True)
        return []
    metrics = pd.read_csv(out / "tier2_metrics.csv")
    monthly = pd.read_csv(out / "tier2_monthly.csv")
    inflow, _ = load_references(data_dir)
    catch = rim_polygons(data_dir)
    sq_mi = catch.groupby("arc")["sq_mi"].sum()
    xw = load_crosswalk(data_dir).set_index("arc")["system"].fillna("")
    sets = load_sets(data_dir)
    # every set that lists an arc gets it on its figure (Shasta's I_SHSTA is on SHA's and Red Bluff's)
    sets_of: dict[str, list[str]] = {}
    for s in sets.itertuples(index=False):
        for a in s.arcs:
            sets_of.setdefault(a, []).append(s.set_id)
    set_name = {s.set_id: s.name for s in sets.itertuples(index=False)}
    set_system = {s.set_id: s.system for s in sets.itertuples(index=False)}
    m0, m1 = window_range(window)
    idx = pd.period_range(m0, m1, freq="M")
    wm = (idx.month - 10) % 12
    met = metrics[(metrics.window == window)].set_index("arc")
    mon = monthly.copy()
    mon["period"] = pd.PeriodIndex(mon["month"], freq="M")
    mon = mon[(mon.period >= idx[0]) & (mon.period <= idx[-1])]
    mon["wm"] = (mon.period.dt.month - 10) % 12
    sim_reg = mon.groupby(["arc", "wm"])["sim_taf"].mean().unstack()

    def ref_regime(arc):
        if arc not in inflow.columns:
            return None
        s = inflow[arc].reindex(idx).to_numpy()
        return np.array([np.nanmean(s[wm == k]) if np.isfinite(s[wm == k]).any() else np.nan
                         for k in range(12)])

    # group every rim arc: its tier-1 set(s), else its crosswalk system, else unconstrained
    groups: dict[str, list[str]] = {}
    for arc in sq_mi.index:
        if arc in sets_of:
            for g in sets_of[arc]:
                groups.setdefault(g, []).append(arc)
            continue
        if xw.get(arc, ""):
            g = next((s for s, sy in set_system.items() if sy == xw[arc]), xw[arc])
        else:
            g = "unconstrained"
        groups.setdefault(g, []).append(arc)
    single = [g for g, arcs in groups.items() if len(arcs) == 1 and g != "unconstrained"]
    if len(single) > 1:
        groups[_SINGLE_ARC_GROUP] = [groups.pop(g)[0] for g in single]

    paths = []
    for g, arcs in groups.items():
        arcs = sorted(arcs, key=lambda a: -float(sq_mi[a]))
        ncol = min(5, max(len(arcs), 2))
        nrow = int(np.ceil(len(arcs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), squeeze=False)
        n_sim = 0
        for ax, arc in zip(axes.ravel(), arcs, strict=False):
            r = ref_regime(arc)
            if r is not None:
                ax.plot(range(12), r, color="k", lw=1.8, label="CalSim3 INFLOW")
            head = f"{arc}  {float(sq_mi[arc]):,.0f} mi²"
            if arc in sim_reg.index:
                ax.plot(range(12), sim_reg.loc[arc].reindex(range(12)), color="tab:blue", lw=1.6, label="dPL")
                n_sim += 1
                extrap = arc in met.index and str(met.loc[arc, "basis"]) == "extrapolated"
                if extrap:
                    ax.set_facecolor("#fff4d6")
                if arc in met.index and np.isfinite(met.loc[arc, "kge"]):
                    head += f"\nKGE {met.loc[arc, 'kge']:.2f}  bias {met.loc[arc, 'pbias']:+.0f}%  seas {met.loc[arc, 'seas_mismatch']:.2f}"
                    if extrap:
                        head += "  (extrapolated)"
                else:
                    head += "\nsimulated, no reference series"
            else:
                ax.set_facecolor("0.93")
                head += "\nnot simulated in this run"
            ax.set_title(head, fontsize=7.5)
            ax.set_xticks(range(12))
            ax.set_xticklabels(list("ONDJFMAMJJAS"), fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(arcs):]:
            ax.axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower right", fontsize=8, frameon=False)
        title = (f"{g} — {set_name[g]}" if g in set_name else g)
        if g in set_system and set_system[g]:
            title += f"  (anchor {set_system[g]})"
        fig.suptitle(f"Tier 2 mean-monthly regime, {window} (TAF/month): {title}  |  {n_sim} of "
                     f"{len(arcs)} arcs simulated  [{label}]", fontsize=10)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        path = out / "figures" / f"tier2_regime_{g.replace(' ', '_')}_{window}.png"
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def _parse_scenarios(spec: str) -> list[tuple[str, float, float]]:
    out = []
    for item in spec.split(","):
        name, val = item.split("=")
        dt, ps = val.split(":")
        out.append((name.strip(), float(dt), float(ps)))
    if len({n for n, _, _ in out}) != len(out):
        raise ValueError(f"--scenarios: duplicate names in {spec!r}")
    return out


def _main_batch(a, ckpt: Path, run_dir: Path) -> None:
    """``--scenarios``: one batched pass, then each scenario's CSVs exactly as a single pass
    writes them (plus the anchor-rescaled scores of a holdout run), into ``--out/NAME``."""
    sc = _parse_scenarios(a.scenarios)
    root = Path(a.out) if a.out else run_dir / "tier2_scenarios"
    outs = [root / n for n, _, _ in sc]
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
    extra = {"components": a.components} if a.components else {}
    if a.extension_cells:
        extra["extension_cells"] = a.extension_cells
    results = score_run(ckpt, a.data_dir, device=a.device, run_dir=run_dir, extend=not a.no_extend,
                        tiles_dir=a.tiles_dir, out=None, trace_python=a.trace_python,
                        spinup=a.spinup, scenarios=[(t, s) for _, t, s in sc], outs=outs,
                        batch_window=a.batch_window, **extra)
    ho = run_holdout_wy(ckpt)
    ho_win = holdout_windows(ho) if ho else None
    vwin = next(iter(ho_win)) if ho_win else VALIDATION_WINDOW
    for (name, t, s), o, res in zip(sc, outs, results, strict=True):
        metrics, monthly, arcs, not_sim, _ = res[:5]
        metrics.to_csv(o / "tier2_metrics.csv", index=False)
        monthly.to_csv(o / "tier2_monthly.csv", index=False)
        arcs.to_csv(o / "tier2_arcs.csv", index=False)
        not_sim.to_csv(o / "tier2_not_simulated.csv", index=False)
        if a.components:
            res[5].to_csv(o / "tier2_components_monthly.csv", index=False)
        if ho or a.anchor_rescaled:
            from .calsim_arcs import own_record_months
            wins = ho_win or {VALIDATION_WINDOW: WINDOWS[VALIDATION_WINDOW]}
            ar = anchor_rescaled(monthly, a.data_dir, windows=wins,
                                 own=own_record_months(a.data_dir, ho) if ho else None,
                                 own_window=vwin if ho else None)
            ar.to_csv(o / "tier2_anchor_rescaled.csv", index=False)
        print(f"tier2 batch: {name} ({t:+g} degC, precip x{s:g}) -> {o}", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", nargs="?", help="run folder (uses checkpoints/best.pt) or a checkpoint file")
    p.add_argument("--out", default=None, help="output folder (default <run>/tier2)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None, help="cpu | cuda (default: cuda if available)")
    p.add_argument("--label", default="")
    p.add_argument("--no-maps", action="store_true")
    p.add_argument("--no-extend", action="store_true",
                   help="do not simulate the arcs outside every trained footprint")
    p.add_argument("--tiles-dir", default="tmp/hydrosheds", help="HydroSHEDS DIR/ACC tiles")
    p.add_argument("--trace-python", default=None,
                   help="interpreter for the flow-length tracer subprocess (an environment "
                        "with rasterio, e.g. the sacsma-gis env; default: this one)")
    p.add_argument("--spinup", default="cycle", choices=["cycle", "window"],
                   help="state at the envelope start: cycle = loop its first ten water years "
                        "20 times from the cold start (timing-independent, default); window = "
                        "the legacy ten water years before it")
    p.add_argument("--dedup-cells", action="store_true",
                   help="run the per-cell physics once per distinct grid cell (routing and "
                        "aggregation per row); same flows to float round-off, less compute")
    p.add_argument("--figures-only", action="store_true",
                   help="only (re)draw the maps and the per-set regime figures from the CSVs "
                        "already in --out (no forward pass)")
    p.add_argument("--components", nargs="?", const="fastslow", default=None,
                   choices=COMPONENT_MODES,
                   help="also write tier2_components_monthly.csv: per arc and month the routed "
                        "FAST (direct + surface + interflow, hillslope x channel UH) and SLOW "
                        "(baseflow, channel UH) components, fast_taf + slow_taf = total_taf "
                        "(= sim_taf); '--components parts' adds quick_taf (impervious + ADIMP "
                        "direct + surface), interflow_taf (quick + interflow = fast), "
                        "supplemental_taf and primary_taf (supplemental + primary = slow).  "
                        "All are NET of the riparian et4 channel-ET deduction")
    p.add_argument("--temp-delta", type=float, default=0.0,
                   help="PLACEHOLDER climate perturbation: degC added uniformly to tavg/tmin/tmax "
                        "(spin-up included) until the WGEN daily scenario weather is supplied")
    p.add_argument("--precip-scale", type=float, default=1.0,
                   help="PLACEHOLDER climate perturbation: multiplies precip uniformly "
                        "(spin-up included) until the WGEN daily scenario weather is supplied")
    p.add_argument("--extension-cells", default=None,
                   help="reuse the flow lengths of a previous run's tier2_extension_cells.csv "
                        "(same checkpoint's entity set) instead of tracing")
    p.add_argument("--anchor-rescaled", action="store_true",
                   help="also write tier2_anchor_rescaled.csv (eval-only anchor-rescaled arc "
                        "scores beside the absolute ones; always on for a run with holdout_wy); "
                        "with "
                        "--figures-only it is computed from the existing tier2_monthly.csv")
    p.add_argument("--scenarios", default=None, metavar="NAME=DT:PS,...",
                   help="several PLACEHOLDER uniform perturbations in ONE batched forward pass "
                        "(stream_batch), e.g. 't1=1:1,p85=0:0.85'; each writes its CSVs into "
                        "--out/NAME (no maps or figures; --temp-delta/--precip-scale ignored)")
    p.add_argument("--batch-window", type=int, default=128,
                   help="days per streamed chunk under --scenarios (device memory only)")
    p.add_argument("--trace-only", nargs=2, metavar=("CELLS_CSV", "OUT_CSV"), default=None,
                   help=argparse.SUPPRESS)   # the flow-length subprocess entry point
    a = p.parse_args(argv)
    if a.trace_only:
        trace_arc_cells(a.trace_only[0], a.trace_only[1], a.tiles_dir)
        return
    if not a.run:
        p.error("run is required")
    run = Path(a.run)
    ckpt = run / "checkpoints" / "best.pt" if run.is_dir() else run
    run_dir = run if run.is_dir() else run.parent.parent
    if a.scenarios:
        _main_batch(a, ckpt, run_dir)
        return
    out = Path(a.out) if a.out else run_dir / "tier2"
    out.mkdir(parents=True, exist_ok=True)
    label = a.label or run_dir.name
    # a run that held water years out of training is validated over them (score_run)
    ho = run_holdout_wy(ckpt)
    ho_win = holdout_windows(ho) if ho else None
    vwin = next(iter(ho_win)) if ho_win else VALIDATION_WINDOW
    if not a.figures_only:
        extra = {}
        if a.components:
            extra["components"] = a.components
        if a.temp_delta or a.precip_scale != 1.0:
            extra.update(temp_delta=a.temp_delta, precip_scale=a.precip_scale)
        if a.extension_cells:
            extra["extension_cells"] = a.extension_cells
        res = score_run(ckpt, a.data_dir, device=a.device, run_dir=run_dir,
                        extend=not a.no_extend, tiles_dir=a.tiles_dir, out=out,
                        trace_python=a.trace_python,
                        spinup=a.spinup, dedup_cells=a.dedup_cells, **extra)
        metrics, monthly, arcs, not_sim, _ = res[:5]
        metrics.to_csv(out / "tier2_metrics.csv", index=False)
        monthly.to_csv(out / "tier2_monthly.csv", index=False)
        arcs.to_csv(out / "tier2_arcs.csv", index=False)
        not_sim.to_csv(out / "tier2_not_simulated.csv", index=False)
        if a.components:
            res[5].to_csv(out / "tier2_components_monthly.csv", index=False)
        print(summarize(metrics, not_sim, window=vwin))
        print(f"wrote {out / 'tier2_metrics.csv'}, tier2_monthly.csv, tier2_arcs.csv, tier2_not_simulated.csv"
              + (", tier2_components_monthly.csv" if a.components else ""))
    else:
        metrics = pd.read_csv(out / "tier2_metrics.csv")
    if (ho or a.anchor_rescaled) and (out / "tier2_monthly.csv").exists():
        from .calsim_arcs import own_record_months
        wins = ho_win or {VALIDATION_WINDOW: WINDOWS[VALIDATION_WINDOW]}
        ar = anchor_rescaled(pd.read_csv(out / "tier2_monthly.csv"), a.data_dir, windows=wins,
                             own=own_record_months(a.data_dir, ho) if ho else None,
                             own_window=vwin if ho else None)
        ar.to_csv(out / "tier2_anchor_rescaled.csv", index=False)
        g = ar[ar.window == vwin].dropna(subset=["kge_resc"])
        print(f"tier2: anchor-rescaled (eval-only), {vwin}: {len(g)} arcs, KGE median absolute "
              f"{g.kge_abs.median():.3f} vs rescaled {g.kge_resc.median():.3f} on the same months "
              f"-> {out / 'tier2_anchor_rescaled.csv'}", flush=True)
    if not a.no_maps:
        try:
            maps(metrics, out, label, a.data_dir, window=vwin)
        except Exception as e:  # noqa: BLE001 — maps are a convenience on top of the CSVs
            print(f"tier2: maps skipped ({e})")
    paths = regime_figures(out, a.data_dir, label, window=vwin)
    print(f"wrote {len(paths)} regime figures under {out / 'figures'}")


if __name__ == "__main__":
    main()
