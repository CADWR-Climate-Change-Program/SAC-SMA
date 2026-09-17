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
footprints.

Each arc's monthly volume (mean depth over its covered cells x the polygon ``SQ_MI``) is
scored against its CalSim3 ``INFLOW`` series over WY1950-84, and over the parent entity's
own training window as the in-sample comparison.  The valley node ``I_SRBB_VAL`` is
simulated for its volume but has no series to score against.

Usage::

    python -m sacsma.calsim.tier2 <run_dir | checkpoint.pt> [--out DIR] [--data-dir data]
                                  [--device cpu|cuda] [--no-maps] [--no-extend]
                                  [--tiles-dir tmp/hydrosheds] [--figures-only] [--label NAME]

Writes ``tier2_metrics.csv`` (one row per arc x window), ``tier2_monthly.csv``,
``tier2_arcs.csv`` (coverage and parent entity per arc), ``tier2_not_simulated.csv``,
``tier2_extension_cells.csv``, ``tier2_sim_daily.npz``, the KGE / bias maps and the regime
figures under ``figures/``, and prints the summary.  When the run folder holds
``sim_daily_mm.npz`` the re-run's entity aggregates are checked against it.

Needs the ``dpl`` extra (torch) and a source checkout: the extrapolated arcs import the
tracer of ``dataprep/build_flowlens.py``, which needs ``rasterio`` and the HydroSHEDS v2
tiles (read from ``--tiles-dir`` when complete, otherwise streamed from the HydroSHEDS
server; the size check needs network either way); without them those arcs fall back to
straight-line x 1.5 lengths, and ``--no-extend`` needs neither.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..metrics import center_of_timing, kge, nse, pbias, pearson, seasonal_mismatch
from .catchments import (_EQ_CRS, _GRID_STEP_DEG, EXCLUDE_ARCS, MERGED_LAYER,
                         _square_cell_overlap, load_catchments, load_crosswalk, series_arc)
from .tier1 import (AF_PER_MM_MI2, VALIDATION_WINDOW, WINDOWS, load_references, load_sets,
                    registry_arcs, registry_windows)


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


def uncovered_arc_cells(catch, parent: dict[str, str], data_dir: str | Path = "data") -> pd.DataFrame:
    """HRU rows for the rim polygons no trained entity lists: every region grid cell whose
    square overlaps the polygon, weight = overlap area, elevation from the region statics.
    Columns ``[basin, key, lat, lon, area_weight, elev]`` (basin = the arc id)."""
    from ..io import soilveg_path
    grid = pd.read_csv(Path(data_dir) / "region" / "grid_cells.csv", usecols=["key", "lat", "lon"])
    unc = catch[~catch["arc"].isin(parent)]
    if unc.empty:
        return pd.DataFrame(columns=["basin", "key", "lat", "lon", "area_weight", "elev"])
    mapping, _ = _square_cell_overlap(grid, unc[["cid", "node", "geometry"]].to_crs(_EQ_CRS),
                                      "EPSG:4326", _GRID_STEP_DEG)
    mapping["basin"] = mapping["node"].map(series_arc)
    sv = pd.read_csv(soilveg_path(data_dir, "multifamily"), usecols=["key", "dem_elev"]).set_index("key")["dem_elev"]
    g = grid.set_index("key")
    out = pd.DataFrame({"basin": mapping["basin"], "key": mapping["key"],
                        "lat": mapping["key"].map(g["lat"]), "lon": mapping["key"].map(g["lon"]),
                        "area_weight": mapping["area_mi2"], "elev": mapping["key"].map(sv)})
    out = out[out["elev"].notna()].sort_values(["basin", "key"]).reset_index(drop=True)
    return out


def trace_arc_cells(cells_csv: str | Path, out_csv: str | Path, tiles_dir: str | Path = "tmp/hydrosheds") -> None:
    """Flow length of every row of ``cells_csv`` to where its HydroSHEDS flow path leaves its
    arc's cell footprint (the entity builder's exit-of-footprint mode); untraceable cells take
    the arc's median traced length.  Runs in its own process (rasterio only, no geopandas)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dataprep"))
    import build_flowlens as bf  # noqa: E402
    cells = pd.read_csv(cells_csv)
    dir_m, acc_m = bf.Mosaic(Path(tiles_dir), "DIR"), bf.Mosaic(Path(tiles_dir), "ACC")
    frames = []
    for arc, sub in cells.groupby("basin", sort=False):
        sub = sub.reset_index(drop=True)
        m = 0.10
        dirr, tr = dir_m.read(sub.lat.min() - m, sub.lat.max() + m, sub.lon.min() - m, sub.lon.max() + m)
        exit_mask = np.zeros(dirr.shape, dtype=bool)
        inv = ~tr
        for cr in sub.itertuples():
            c0, r0 = inv * (cr.lon - bf.HALF, cr.lat + bf.HALF)
            c1, r1 = inv * (cr.lon + bf.HALF, cr.lat - bf.HALF)
            exit_mask[max(int(r0), 0):int(r1) + 1, max(int(c0), 0):int(c1) + 1] = True
        res = bf.trace_entity(dirr, tr, sub, acc_m, None, exit_mask, None)
        ok = res.flowlen_m.notna() & (res.method != "fallback")
        if ok.any():
            res.loc[~ok, "flowlen_m"] = res.flowlen_m[ok].median()
        else:  # nothing traced: straight line to the lowest cell x a typical sinuosity
            j = int(sub["elev"].idxmin())
            res["flowlen_m"] = bf.haversine_m(sub.lat, sub.lon, sub.lat[j], sub.lon[j]) * 1.5
        res.insert(0, "basin", arc)
        frames.append(res[["basin", "key", "flowlen_m", "method"]])
        print(f"trace {arc:11s} cells {len(sub):3d} traced {int(ok.sum()):3d}", flush=True)
    dir_m.close()
    acc_m.close()
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)


def _flowlens_for(cells: pd.DataFrame, tiles_dir: str | Path) -> pd.DataFrame:
    """Add ``flowlen`` (m) and ``flowlen_method`` to the extension HRU rows, tracing in a
    subprocess; falls back to straight-line x 1.5 to each arc's lowest cell if tracing fails."""
    with tempfile.TemporaryDirectory() as td:
        cin, cout = Path(td) / "cells.csv", Path(td) / "flowlens.csv"
        cells.to_csv(cin, index=False)
        r = subprocess.run([sys.executable, "-m", "sacsma.calsim.tier2", "--trace-only", str(cin),
                            str(cout), "--tiles-dir", str(tiles_dir)], capture_output=True, text=True)
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
    from ..dpl.config import DplConfig
    from ..dpl.data import load_domain_tensors
    from ..dpl.features import FeatureSet, build_features
    from ..dpl.parameter_net import ParameterNet
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
                       dynamic_amp=nc.get("dynamic_amp", 0.5)).to(dev, torch.float64)
    net.load_state_dict(ck["net"])
    return net, x, dom, cfg


def stream(net, x, dom, cfg, W_arc: np.ndarray):
    """Stream the trained field over the envelope (the evaluator's protocol) and return
    (arc depth (A, T), entity depth (B, T), t0, t1, bad_rows) in mm/day.  A cell whose
    physics returns NaN (extrapolated parameters at their bounds) is zeroed and flagged in
    ``bad_rows`` so it cannot blank every aggregate that carries a zero weight on it."""
    from ..dpl.forward import initial_state, routing_uh, run_window
    from ..dpl.multi_timescale import ENVELOPE_END, ENVELOPE_START

    t0 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_START)))
    t1 = int(dom.dates.searchsorted(pd.Timestamp(ENVELOPE_END))) + 1
    spin_req = int(dom.dates.searchsorted(pd.Timestamp(cfg.spinup_start)))
    if spin_req >= t0:
        spin_req = int(dom.dates.searchsorted(dom.dates[t0] - pd.DateOffset(years=10)))
    spin = max(min(spin_req, t0), 0)
    Wa = torch.as_tensor(W_arc).to(dom.device, dom.dtype)
    net.eval()
    with torch.no_grad():
        out = net(x)
        canopy = out.pop("_canopy", None)
        uh = routing_uh(out, dom.flowlen)
        state = initial_state(dom.n_hru, dom.device, dom.dtype,
                              init_mode=cfg.init_mode, params=out, et_mode=cfg.et_mode)
        arcs, ents = [], []
        bad = torch.zeros(dom.n_hru, dtype=torch.bool, device=dom.device)
        t = spin
        while t < t1:
            te = min(t + 512, t1)
            pr, ta, doy, leap = dom.chunk(t, te)
            tn, tx = dom.chunk_tmm(t, te)
            flow, state = run_window(
                pr, ta, doy, leap, dom.lat_rad, dom.elev, out, uh, state,
                n_inc=cfg.n_inc, perc_mode=cfg.perc_mode, fracp_floor=cfg.fracp_floor,
                ninc_mode="fixed", et_mode=cfg.et_mode, canopy_params=canopy, tmin=tn, tmax=tx,
                veg_frac=dom.veg_frac, lai=dom.chunk_lai(t, te), noah_pet=cfg.noah_pet,
                sac_pet=cfg.sac_pet, pt_snow_albedo=cfg.pt_snow_albedo,
                pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                canopy_lite=cfg.canopy_lite, state_idx=dom.chunk_state(t, te))
            nan_rows = torch.isnan(flow).any(dim=1)
            if nan_rows.any():
                bad |= nan_rows
                flow = torch.nan_to_num(flow, nan=0.0)
            if te > t0:
                k = max(t0 - t, 0)
                arcs.append((Wa @ flow)[:, k:].cpu())
                ents.append((dom.W @ flow)[:, k:].cpu())
            t = te
    return (torch.cat(arcs, dim=1).double().numpy(), torch.cat(ents, dim=1).double().numpy(),
            t0, t1, bad.cpu().numpy())


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


def score_run(ckpt: str | Path, data_dir: str | Path = "data", *, device=None, run_dir=None,
              extend: bool = True, tiles_dir: str | Path = "tmp/hydrosheds", out: Path | None = None):
    """Returns (metrics, monthly, arcs, not_sim, entity_check).  ``extend`` simulates the
    arcs outside every trained footprint on their own region cells (``basis = extrapolated``)."""
    from ..dpl.evaluate import load_net_from_checkpoint
    net, x, dom, cfg, ck = load_net_from_checkpoint(ckpt, data_dir, device=device)
    catch = rim_polygons(data_dir)
    mapping = cell_arc_overlap(dom.hrus, catch)
    parent = parent_entities(dom.basins, data_dir)
    W_arc, arcs = arc_weights(dom, catch, mapping, parent)
    arcs["basis"] = "trained entity"
    print(f"tier2: {len(catch)} rim polygons; {len(arcs)} arcs inside the {len(dom.basins)} trained "
          f"entities; {len(dom.hrus)} HRU rows on {dom.device.type}", flush=True)
    print("tier2: streaming the envelope ...", flush=True)
    sim_arc, sim_ent, t0, t1, bad = stream(net, x, dom, cfg, W_arc)
    sim_arc, arcs["n_nan_cells"], arcs["nan_weight_frac"] = _drop_bad_cells(sim_arc, W_arc, bad, dom.hrus, "trained-footprint")
    dates = dom.dates[t0:t1]
    if extend:
        ext = uncovered_arc_cells(catch, parent, data_dir)
        if len(ext):
            ext = _flowlens_for(ext, tiles_dir)
            if out is not None:
                ext.to_csv(out / "tier2_extension_cells.csv", index=False)
            net2, x2, dom2, cfg2 = _net_for_hrus(ckpt, ext, data_dir, device=device)
            print(f"tier2: extrapolating {len(dom2.basins)} uncovered arcs on {len(dom2.hrus)} region "
                  f"cells ...", flush=True)
            W2 = dom2.W.cpu().numpy()
            sim_ext, _, t0e, t1e, bad2 = stream(net2, x2, dom2, cfg2, W2)
            assert (t0e, t1e) == (t0, t1)
            sim_ext, n_bad2, w_bad2 = _drop_bad_cells(sim_ext, W2, bad2, dom2.hrus, "extrapolated")
            sq_mi = catch.groupby("arc")["sq_mi"].sum()
            cov = ext.groupby("basin")["area_weight"].sum()
            rows = [dict(arc=a, node=a[2:], entity="", sq_mi=float(sq_mi[a]), covered_mi2=float(cov[a]),
                         cover_frac=float(cov[a] / sq_mi[a]), n_cells=int((ext.basin == a).sum()),
                         basis="extrapolated", n_nan_cells=int(n_bad2[j]), nan_weight_frac=float(w_bad2[j]))
                    for j, a in enumerate(dom2.basins)]
            arcs = pd.concat([arcs, pd.DataFrame(rows)], ignore_index=True)
            sim_arc = np.concatenate([sim_arc, sim_ext], axis=0)
    if out is not None:   # the daily arc series, so scoring and figures can be redone without the forward pass
        np.savez_compressed(out / "tier2_sim_daily.npz", arc=np.array(arcs["arc"]),
                            date=dates.to_numpy().astype("datetime64[D]").astype(str),
                            sim_mm=sim_arc.astype(np.float32))

    entity_check = None
    if run_dir is not None and (Path(run_dir) / "sim_daily_mm.npz").exists():
        z = np.load(Path(run_dir) / "sim_daily_mm.npz")
        ref = pd.DataFrame(np.asarray(z["sim_mm"]).T.astype(float), columns=list(z["entity_id"]))
        mine = pd.DataFrame(sim_ent.T, columns=list(dom.basins))
        common = [b for b in dom.basins if b in ref.columns]
        d = (mine[common].to_numpy() - ref[common].to_numpy())
        entity_check = dict(n_entities=len(common), max_abs_diff_mm=float(np.nanmax(np.abs(d))),
                            rel_rmse=float(np.sqrt(np.nanmean(d ** 2)) / np.nanmean(ref[common].to_numpy())))
        print(f"tier2: entity re-run vs archived sim_daily_mm.npz over {len(common)} entities: "
              f"max |diff| {entity_check['max_abs_diff_mm']:.3g} mm/day, relative RMSE "
              f"{entity_check['rel_rmse']:.2e}", flush=True)

    inflow, _ = load_references(data_dir)
    xw = load_crosswalk(data_dir).set_index("arc")
    sets = load_sets(data_dir)
    in_set = {a: s.set_id for s in sets.itertuples(index=False) for a in s.arcs}
    train_win = registry_windows(data_dir)
    taf = _monthly_taf(sim_arc, dates, arcs["sq_mi"].to_numpy())
    taf.columns = list(arcs["arc"])
    rows, monthly = [], []
    for a in arcs.itertuples(index=False):
        ref = inflow[a.arc] if a.arc in inflow.columns else pd.Series(np.nan, index=taf.index)
        # in-sample window: the parent entity's training window; for an extrapolated arc there
        # is none, so the monthly family's window stands in as the comparison period
        t0m, t1m = train_win[a.entity] if a.entity else (pd.Period("1984-10", "M"), pd.Period("2014-09", "M"))
        windows = {VALIDATION_WINDOW: WINDOWS[VALIDATION_WINDOW], "train": (str(t0m), str(t1m))}
        for wname, (m0, m1) in windows.items():
            idx = pd.period_range(m0, m1, freq="M")
            met = _score(idx, taf[a.arc].reindex(idx).to_numpy(), ref.reindex(idx).to_numpy())
            rows.append(dict(arc=a.arc, node=a.node, entity=a.entity, basis=a.basis,
                             system=xw["system"].get(a.arc, ""), tier1_set=in_set.get(a.arc, ""),
                             sq_mi=a.sq_mi, cover_frac=a.cover_frac, n_cells=a.n_cells,
                             n_nan_cells=int(a.n_nan_cells), nan_weight_frac=float(a.nan_weight_frac),
                             has_series=a.arc in inflow.columns, window=wname,
                             win_start=m0, win_end=m1, **met))
        monthly.append(pd.DataFrame({"arc": a.arc, "month": taf.index.astype(str),
                                     "sim_taf": taf[a.arc].to_numpy(),
                                     "ref_taf": ref.reindex(taf.index).to_numpy()}))
    metrics = pd.DataFrame(rows)
    # arcs with a series that this run does not simulate
    simulated = set(arcs["arc"])
    missing = [a for a in catch["arc"].unique() if a not in simulated and a in inflow.columns]
    v0, v1 = WINDOWS[VALIDATION_WINDOW]
    idx = pd.period_range(v0, v1, freq="M")
    not_sim = pd.DataFrame({
        "arc": missing,
        "system": [xw["system"].get(a, "") for a in missing],
        "sq_mi": [float(catch.loc[catch["arc"] == a, "sq_mi"].sum()) for a in missing],
        "ref_taf_yr": [12.0 * inflow[a].reindex(idx).mean() for a in missing]})
    return metrics, pd.concat(monthly, ignore_index=True), arcs, not_sim, entity_check


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
            lines.append("  extrapolated arcs (outside every trained footprint), by reference volume:")
            lines.append(e.sort_values("ref_taf_yr", ascending=False)[cols].round(3).to_string(index=False))
        if len(not_sim):
            lines.append("  not simulated:")
            lines.append(not_sim.sort_values("ref_taf_yr", ascending=False).round(1).to_string(index=False))
    return "\n".join(lines)


def maps(metrics: pd.DataFrame, out: Path, label: str, window: str = VALIDATION_WINDOW) -> None:
    from .compare import _arc_choropleth
    m = metrics[(metrics.window == window) & metrics.has_series].set_index("arc")
    _arc_choropleth("data", m["kge"], f"Tier 2 — KGE per rim arc, {window}  [{label}]", "KGE",
                    out / f"tier2_kge_{window}.png", cmap="plasma", vmin=0.0, vmax=1.0)
    _arc_choropleth("data", m["pbias"], f"Tier 2 — volume bias per rim arc, {window}  [{label}]",
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
    metrics = pd.read_csv(out / "tier2_metrics.csv")
    monthly = pd.read_csv(out / "tier2_monthly.csv")
    inflow, _ = load_references(data_dir)
    catch = rim_polygons(data_dir)
    sq_mi = catch.groupby("arc")["sq_mi"].sum()
    xw = load_crosswalk(data_dir).set_index("arc")["system"].fillna("")
    sets = load_sets(data_dir)
    in_set = {a: s.set_id for s in sets.itertuples(index=False) for a in s.arcs}
    set_name = {s.set_id: s.name for s in sets.itertuples(index=False)}
    set_system = {s.set_id: s.system for s in sets.itertuples(index=False)}
    m0, m1 = WINDOWS[window]
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

    # group every rim arc: its tier-1 set, else its crosswalk system, else unconstrained
    groups: dict[str, list[str]] = {}
    for arc in sq_mi.index:
        if arc in in_set:
            g = in_set[arc]
        elif xw.get(arc, ""):
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
    p.add_argument("--figures-only", action="store_true",
                   help="only (re)draw the per-set regime figures from the CSVs already in --out")
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
    out = Path(a.out) if a.out else run_dir / "tier2"
    out.mkdir(parents=True, exist_ok=True)
    label = a.label or run_dir.name
    if not a.figures_only:
        metrics, monthly, arcs, not_sim, _ = score_run(ckpt, a.data_dir, device=a.device, run_dir=run_dir,
                                                       extend=not a.no_extend, tiles_dir=a.tiles_dir, out=out)
        metrics.to_csv(out / "tier2_metrics.csv", index=False)
        monthly.to_csv(out / "tier2_monthly.csv", index=False)
        arcs.to_csv(out / "tier2_arcs.csv", index=False)
        not_sim.to_csv(out / "tier2_not_simulated.csv", index=False)
        print(summarize(metrics, not_sim))
        if not a.no_maps:
            try:
                maps(metrics, out, label)
            except Exception as e:  # noqa: BLE001 — maps are a convenience on top of the CSVs
                print(f"tier2: maps skipped ({e})")
        print(f"wrote {out / 'tier2_metrics.csv'}, tier2_monthly.csv, tier2_arcs.csv, tier2_not_simulated.csv")
    paths = regime_figures(out, a.data_dir, label)
    print(f"wrote {len(paths)} regime figures under {out / 'figures'}")


if __name__ == "__main__":
    main()
