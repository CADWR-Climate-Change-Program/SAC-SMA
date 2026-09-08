"""CalSim3 validation atlas: where every validated location is, and how the run scored on it.

Builds, from a finished tier-1 output folder (:mod:`sacsma.calsim.tier1`) and, when given, the
run's tier-2 folder (:mod:`sacsma.calsim.tier2`):

* ``atlas/tier1_map_kge_<window>.png`` and ``atlas/tier1_map_pbias_<window>.png`` — the
  CalSim3 rim domain with every tier-1 arc set dissolved and coloured by its score, the
  set id and its numbers written inside each basin;
* ``atlas/<set_id>_map.png`` and ``atlas/<set_id>_zoom.png`` — the set highlighted on
  the domain map with its registry outlet, and a zoom naming its arcs, coloured by tier-2
  KGE when the tier-2 metrics are available;
* ``atlas/family_<family>.png`` — the training footprints and gauges of each target family;
* ``atlas/calsim_validation_atlas.html`` — a self-contained page (images embedded): a domain
  tab with the tier-1 and tier-2 maps and the summary table, one tab per location with its
  tier-1 metrics, time series, tier-2 sub-arc table and regime figure and its maps, a tab for
  the unconstrained arcs, and a tab of training footprints by family;
* ``atlas/atlas.md`` — the tier-1 content as Markdown with image links.

Usage::

    python -m sacsma.calsim.atlas <tier1_out_dir> [--data-dir data] [--tier2-dir ...] [--label ...]
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .catchments import MERGED_LAYER, load_catchments, series_arc
from .tier1 import VALIDATION_WINDOW, load_sets


def _rim(data_dir):
    catch = load_catchments(data_dir, layer=MERGED_LAYER, rim_only=True)
    if catch.crs is not None and not catch.crs.is_geographic:
        catch = catch.to_crs("EPSG:4326")
    catch["arc"] = catch["node"].map(series_arc)
    return catch


def _set_geoms(catch, sets):
    """Per set: (GeoDataFrame of its polygons, dissolved geometry)."""
    from shapely.ops import unary_union
    out = {}
    for s in sets.itertuples(index=False):
        g = catch[catch["arc"].isin(s.arcs)]
        out[s.set_id] = (g, unary_union(list(g.geometry)) if len(g) else None)
    return out


def _outlets(data_dir):
    reg = pd.read_csv(Path(data_dir) / "multifamily" / "entities.csv", dtype={"site_id": str})
    return {r.entity_id: (r.outlet_lat, r.outlet_lon) for r in reg.itertuples(index=False)
            if pd.notna(r.outlet_lat)}


def _aspect(catch):
    minx, miny, maxx, maxy = catch.total_bounds
    return 1.0 / np.cos(np.deg2rad(0.5 * (miny + maxy)))


def daily_monthly_overlap(data_dir: str | Path, run_dir: Path | None):
    """Where a daily-family entity and a monthly-family entity of the run both train: the
    intersection of the two families' cell-footprint unions (1/16-degree cell squares from
    ``entity_cells.csv``).  The trained entities are read from the run's ``sim_daily_mm.npz``
    when available, else every registry entity is assumed trained."""
    from shapely import box
    from shapely.ops import unary_union
    reg = pd.read_csv(Path(data_dir) / "multifamily" / "entities.csv", dtype={"site_id": str})
    trained = None
    if run_dir is not None and (Path(run_dir) / "sim_daily_mm.npz").exists():
        trained = set(np.load(Path(run_dir) / "sim_daily_mm.npz", allow_pickle=True)["entity_id"].tolist())
    if trained is not None:
        reg = reg[reg["entity_id"].isin(trained)]
    cells = pd.read_csv(Path(data_dir) / "multifamily" / "entity_cells.csv")
    cells = cells[cells["entity_id"].isin(reg["entity_id"])]
    fam = reg.set_index("entity_id")["timescale"]
    h = 0.03125
    unions = {}
    for ts in ("daily", "monthly"):
        sub = cells[cells["entity_id"].map(fam) == ts].drop_duplicates("key")
        unions[ts] = unary_union([box(x - h, y - h, x + h, y + h)
                                  for x, y in zip(sub["lon"], sub["lat"], strict=True)]) if len(sub) else None
    if unions["daily"] is None or unions["monthly"] is None:
        return None, trained is not None
    ov = unions["daily"].intersection(unions["monthly"])
    return (ov if not ov.is_empty else None), trained is not None


FAMILY_LABEL = {"cdec_daily": "CDEC daily full natural flow",
                "uf_monthly": "DWR Appendix B monthly unimpaired flow",
                "usgs_daily": "USGS daily unimpaired gauges"}
FAMILY_COLOR = {"cdec_daily": "tab:blue", "uf_monthly": "tab:green", "usgs_daily": "tab:orange"}


def _entity_footprints(data_dir: str | Path, entity_ids=None):
    """Per entity: the union of its 1/16-degree cell squares, as a GeoDataFrame with the
    registry attributes (family, name, area, outlet)."""
    import geopandas as gpd
    from shapely import box
    from shapely.ops import unary_union
    reg = pd.read_csv(Path(data_dir) / "multifamily" / "entities.csv", dtype={"site_id": str})
    if entity_ids is not None:
        reg = reg[reg["entity_id"].isin(entity_ids)]
    cells = pd.read_csv(Path(data_dir) / "multifamily" / "entity_cells.csv")
    cells = cells[cells["entity_id"].isin(reg["entity_id"])]
    h = 0.03125
    geoms = {eid: unary_union([box(x - h, y - h, x + h, y + h)
                               for x, y in zip(g["lon"], g["lat"], strict=True)])
             for eid, g in cells.groupby("entity_id")}
    reg = reg[reg["entity_id"].isin(geoms)].copy()
    return gpd.GeoDataFrame(reg, geometry=[geoms[e] for e in reg["entity_id"]], crs="EPSG:4326")


def family_maps(catch, data_dir, out: Path, trained=None, label: str = "") -> dict:
    """One map per family: its entities' footprints outlined and labelled, gauges marked, the
    CalSim3 rim domain behind.  Returns {family: (path, table rows)}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fp = _entity_footprints(data_dir, trained)
    asp = _aspect(catch)
    out_map = {}
    for fam in ("cdec_daily", "uf_monthly", "usgs_daily"):
        g = fp[fp["family"] == fam].sort_values("area_mi2", ascending=False).reset_index(drop=True)
        if not len(g):
            continue
        bx = g.total_bounds
        cb = catch.total_bounds
        minx, miny = min(bx[0], cb[0]) - 0.2, min(bx[1], cb[1]) - 0.2
        maxx, maxy = max(bx[2], cb[2]) + 0.2, max(bx[3], cb[3]) + 0.2
        h_in = 10.0
        w_in = max(5.0, min(9.5, h_in * (maxx - minx) / max((maxy - miny) * asp, 1e-9)))
        fig, ax = plt.subplots(figsize=(w_in, h_in))
        catch.plot(ax=ax, color="0.94", edgecolor="0.82", linewidth=0.3)
        g.plot(ax=ax, color=FAMILY_COLOR[fam], edgecolor="k", linewidth=0.6, alpha=0.45)
        rows = []
        big = fam != "usgs_daily"
        for i, r in enumerate(g.itertuples(index=False), start=1):
            tag = r.entity_id if big else str(i)
            has_out = pd.notna(r.outlet_lat) and pd.notna(r.outlet_lon)
            if has_out:
                ax.plot(r.outlet_lon, r.outlet_lat, marker="o", ms=4.5, color="red", mec="k",
                        mew=0.4, ls="none", zorder=5)
            px, py = ((r.outlet_lon, r.outlet_lat) if has_out
                      else (r.geometry.representative_point().x, r.geometry.representative_point().y))
            ax.annotate(tag, (px, py), xytext=(3, 3), textcoords="offset points",
                        fontsize=6.5 if big else 5.5, fontweight="bold" if big else "normal",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75), zorder=6)
            rows.append((tag, r.entity_id, str(r.site_id), r.name, float(r.area_mi2),
                         str(r.train_start)[:7], str(r.train_end)[:7], int(r.n_obs)))
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=FAMILY_COLOR[fam], edgecolor="k", alpha=0.45,
                                 label=f"{FAMILY_LABEL[fam]} ({len(g)})"),
                           Line2D([], [], marker="o", color="red", mec="k", ls="none", ms=5,
                                  label="gauge / pour point")],
                  loc="lower left", fontsize=8, frameon=True)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect(asp)
        ax.set_title(f"{FAMILY_LABEL[fam]}: training footprints"
                     + (f"\n[{label}]" if label else ""), fontsize=9)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        path = out / f"family_{fam}.png"
        fig.savefig(path, dpi=135)
        plt.close(fig)
        out_map[fam] = (path, rows)
    return out_map


def _hatch(ax, overlap, crs, label="daily + monthly targets overlap"):
    import geopandas as gpd
    from matplotlib.patches import Patch
    if overlap is None:
        return None
    gpd.GeoSeries([overlap], crs=crs).plot(ax=ax, facecolor="none", edgecolor="0.15", hatch="////", linewidth=0.0)
    return Patch(facecolor="none", edgecolor="0.15", hatch="////", label=label)


def domain_maps(catch, sets, geoms, metrics, out: Path, label: str, window: str,
                overlap=None, overlap_note: str = "") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    m = metrics[(metrics.window == window) & metrics.ref_kind.isin(["anchor", "arcsum"])].set_index("set_id")
    S = sets.set_index("set_id")
    order = sorted(geoms, key=lambda k: -float(S.loc[k, "area_mi2"]))
    paths = []
    for col, cmap_name, vmin, vmax, cb in (("kge", "plasma", 0.0, 1.0, "KGE"),
                                           ("pbias", "BrBG", -40.0, 40.0, "volume bias (%)")):
        fig, ax = plt.subplots(figsize=(11.0, 10.5))
        catch.plot(ax=ax, color="0.94", edgecolor="0.8", linewidth=0.3)
        cmap, norm = colormaps[cmap_name], Normalize(vmin, vmax)
        for sid in order:                       # big sets first so nested ones draw on top
            g, diss = geoms[sid]
            if diss is None or sid not in m.index:
                continue
            v = float(m.loc[sid, col])
            gpd.GeoSeries([diss], crs=catch.crs).plot(
                ax=ax, color=cmap(norm(v)) if np.isfinite(v) else "0.7", edgecolor="k", linewidth=0.6)
        minx, miny, maxx, maxy = catch.total_bounds
        labels = []
        for sid in order:
            g, diss = geoms[sid]
            if diss is None or sid not in m.index:
                continue
            own = diss
            for other, (_, od) in geoms.items():      # label a nesting set on its own ground
                if other != sid and od is not None and od.area < diss.area \
                        and od.intersection(diss).area > 0.9 * od.area:
                    own = own.difference(od)
            p = (own if not own.is_empty else diss).representative_point()
            r = m.loc[sid]
            txt = (f"{sid}  KGE {r.kge:.2f}  {r.pbias:+.0f}%" if np.isfinite(r.kge) else f"{sid}  not scored")
            if not bool(S.loc[sid, "volume_scored"]):
                txt += "\n(not aggregated)"
            labels.append((sid, p.x, p.y, txt, float(S.loc[sid, "area_mi2"])))
        box = dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", alpha=0.95)
        # big basins carry their label inside; the rest go to a side column with a leader line
        inside = [t for t in labels if t[4] >= 1500]
        side = [t for t in labels if t[4] < 1500]
        for sid, x, y, txt, _ in inside:
            ax.annotate(txt.replace("  KGE", "\nKGE"), (x, y), ha="center", va="center", fontsize=9,
                        fontweight="bold", bbox=box)
        mid = 0.5 * (minx + maxx)
        for col_sign, group in ((-1, [t for t in side if t[1] < mid]), (+1, [t for t in side if t[1] >= mid])):
            group = sorted(group, key=lambda t: t[2])
            if not group:
                continue
            xs = (minx - 0.3) if col_sign < 0 else (maxx + 0.3)
            ys = np.linspace(miny + 0.15, maxy - 0.15, len(group)) if len(group) > 1 else [group[0][2]]
            for (sid, x, y, txt, _), yy in zip(group, ys, strict=True):
                ax.annotate(txt, (x, y), xytext=(xs, yy), ha="right" if col_sign < 0 else "left", va="center",
                            fontsize=9, fontweight="bold", bbox=box,
                            arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8, shrinkB=2))
        hp = _hatch(ax, overlap, catch.crs)
        if hp is not None:
            ax.legend(handles=[hp], loc="lower left", fontsize=9, frameon=True)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, shrink=0.6, label=cb, pad=0.02)
        ax.set_xlim(minx - 1.9, maxx + 2.0)
        ax.set_ylim(miny - 0.2, maxy + 0.2)
        ax.set_aspect(_aspect(catch))
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"Tier 1 — {cb} per location, {window}  [{label}]\n"
                     "grey: rim arcs in no tier-1 set; nested sets drawn on top; "
                     "hatched: trained at both daily and monthly timescale", fontsize=10.5)
        fig.tight_layout()
        path = out / f"tier1_map_{col}_{window}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def location_maps(sid: str, row, catch, geoms, outlets, out: Path, overlap=None, t2=None) -> tuple[Path, Path]:
    """The set on the domain map (with its registry outlet) and a zoom naming its arcs; with ``t2`` (tier-2
    metrics indexed by arc) the zoom colours each arc by its tier-2 KGE."""
    import matplotlib
    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    g, diss = geoms[sid]
    asp = _aspect(catch)
    o = outlets.get(row.entity_id)
    # domain map
    fig, ax = plt.subplots(figsize=(4.2, 5.2))
    catch.plot(ax=ax, color="0.93", edgecolor="0.8", linewidth=0.3)
    if diss is not None:
        gpd.GeoSeries([diss], crs=catch.crs).plot(ax=ax, color="tab:orange", edgecolor="k", linewidth=0.6)
    hp = _hatch(ax, overlap, catch.crs, label="daily + monthly overlap")
    if o is not None:
        ax.plot(o[1], o[0], marker="*", color="red", ms=10, mec="k", mew=0.5, ls="none")
    if hp is not None:
        ax.legend(handles=[hp], loc="lower left", fontsize=7, frameon=True)
    minx, miny, maxx, maxy = catch.total_bounds
    ax.set_xlim(minx - 0.15, maxx + 0.15)
    ax.set_ylim(miny - 0.15, maxy + 0.15)
    ax.set_aspect(asp)
    ax.set_title(f"{sid} in the CalSim3 rim domain", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    p_map = out / f"{sid}_map.png"
    fig.savefig(p_map, dpi=120)
    plt.close(fig)
    # zoom
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    if diss is not None:
        bx = diss.bounds
        pad = max(0.12, 0.12 * max(bx[2] - bx[0], bx[3] - bx[1]))
        catch.plot(ax=ax, color="0.94", edgecolor="0.75", linewidth=0.4)
        gg = g.sort_values("sq_mi", ascending=False).reset_index(drop=True)
        if t2 is not None and "kge" in t2.columns:
            import matplotlib as mpl
            key = "arc" if "arc" in gg.columns else "node"
            kge = gg[key].map(t2["kge"])
            norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
            cm = colormaps["plasma"]
            cols = [cm(norm(k)) if k == k else (0.85, 0.85, 0.85, 1.0) for k in kge]
            gg.plot(ax=ax, color=cols, edgecolor="k", linewidth=0.5, alpha=0.9)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label="tier-2 KGE WY1950-84 (grey: no series)")
        else:
            cmap = colormaps["tab20"]
            gg.plot(ax=ax, color=[cmap(i % 20) for i in range(len(gg))], edgecolor="k", linewidth=0.5, alpha=0.85)
        n_label = len(gg) if len(gg) <= 25 else 12
        for i, r in enumerate(gg.itertuples(index=False)):
            if i >= n_label:
                break
            p = r.geometry.representative_point()
            ax.annotate(r.node, (p.x, p.y), ha="center", va="center", fontsize=6 if len(gg) > 8 else 7.5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        if o is not None:
            ax.plot(o[1], o[0], marker="*", color="red", ms=13, mec="k", mew=0.6, ls="none", label="registry outlet")
            ax.legend(fontsize=7, loc="lower left")
        ax.set_xlim(bx[0] - pad, bx[2] + pad)
        ax.set_ylim(bx[1] - pad, bx[3] + pad)
        ax.set_aspect(asp)
    sub = f"{len(row.arcs)} arc(s), {row.area_mi2:,.0f} mi²" + ("" if len(g) <= 25 else " — labels on the 12 largest")
    ax.set_title(f"{sid}: {sub}", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    p_zoom = out / f"{sid}_zoom.png"
    fig.savefig(p_zoom, dpi=120)
    plt.close(fig)
    return p_map, p_zoom


def _b64(path: Path, *, jpeg_quality: int | None = None) -> str:
    """Data URI of an image; optionally re-encoded as JPEG to keep the page small."""
    if jpeg_quality is None:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[3])
    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _t2_table(rows, e, *, show_set: bool, show_cover: bool) -> list[str]:
    """HTML rows of a tier-2 arc table, ordered by reference volume."""
    has_cls = "method_class" in rows.columns
    out = ["<table><tr><th>arc</th>"
           + ("<th>tier-1 set</th>" if show_set else "")
           + "<th>mi²</th>"
           + ("<th>derivation</th><th>record share</th>" if has_cls else "")
           + "<th>basis</th>"
           + ("<th>trained cells</th>" if show_cover else "")
           + "<th>KGE</th><th>NSE</th><th>bias</th><th>r</th><th>seas. mismatch</th>"
             "<th>sim / ref TAF/yr</th></tr>"]
    for r in rows.sort_values("ref_taf_yr", ascending=False).itertuples(index=False):
        cls = ""
        if has_cls:
            mc = "" if r.method_class != r.method_class else str(r.method_class).replace("_", " ")
            rf = getattr(r, "record_frac", float("nan"))
            own = getattr(r, "record_owner", "")
            own = "" if own != own else str(own)
            mf = "" if rf != rf else (f"{rf:.0%}" + (" (donor)" if own == "donor" else ""))
            cls = f"<td class='l'>{e(mc)}</td><td>{mf}</td>"
        st = f"<td class='l'>{e(str(getattr(r, 'tier1_set', '') or '—'))}</td>" if show_set else ""
        cov = ""
        if show_cover:
            cf = getattr(r, "trained_cell_frac", float("nan"))
            cov = f"<td>{'' if cf != cf else f'{cf:.0%}'}</td>"
        out.append(f"<tr><td class='l'>{e(r.arc)}</td>{st}<td>{r.sq_mi:,.0f}</td>{cls}"
                   f"<td class='l'>{e(str(r.basis))}</td>{cov}"
                   f"<td>{_fmt(r.kge, '.3f')}</td><td>{_fmt(r.nse, '.3f')}</td>"
                   f"<td>{_fmt(r.pbias, '+.0f')}{'' if r.pbias != r.pbias else '%'}</td>"
                   f"<td>{_fmt(r.r, '.3f')}</td><td>{_fmt(r.seas_mismatch, '.3f')}</td>"
                   f"<td>{r.sim_taf_yr:,.0f} / {_fmt(r.ref_taf_yr, ',.0f')}</td></tr>")
    out.append("</table>")
    return out


def _fmt(v, spec: str) -> str:
    """Formatted number, or '' for NaN."""
    return "" if v != v else format(v, spec)


def _metrics_rows(metrics, sid, window):
    m = metrics[(metrics.set_id == sid)]
    rows = []
    for r in m[m.ref_kind != "arcsum_covered"].itertuples(index=False):
        wl = r.window if r.window == window else f"training window {r.win_start}..{r.win_end}"
        rows.append((wl, r))
    for r in m[(m.ref_kind == "arcsum_covered") & (m.window == window)].itertuples(index=False):
        rows.append((f"{window}, covered arcs only ({100 * r.cover_frac:.1f}% of the set)", r))
    return rows


def write_html(sets, metrics, out: Path, label: str, window: str, maps: list[Path], t1_dir: Path,
               t2=None, t2_dir: Path | None = None, fam_maps: dict | None = None) -> Path:
    e = html.escape
    t2 = t2 if t2 is not None else pd.DataFrame()
    fam_maps = fam_maps or {}
    v = metrics[(metrics.window == window) & metrics.ref_kind.isin(["anchor", "arcsum"])].set_index("set_id")
    ids = [s.set_id for s in sets.itertuples(index=False) if s.set_id in v.index]
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
    header{padding:12px 18px;background:#1f3b57;color:#fff}
    header h1{margin:0;font-size:18px} header p{margin:4px 0 0;font-size:12px;opacity:.9}
    nav{display:flex;flex-wrap:wrap;gap:4px;padding:10px 18px;background:#e9eef3;position:sticky;top:0;z-index:5}
    nav button{border:1px solid #9fb3c8;background:#fff;border-radius:4px;padding:4px 9px;font-size:12px;cursor:pointer}
    nav button.on{background:#1f3b57;color:#fff;border-color:#1f3b57}
    section{display:none;padding:14px 18px} section.on{display:block}
    table{border-collapse:collapse;font-size:12px;margin:8px 0 12px} th,td{border:1px solid #cfd8e0;padding:3px 7px;text-align:right}
    th:first-child,td:first-child,td.l{text-align:left} tr.pick{cursor:pointer} tr.pick:hover{background:#eef4fa}
    .row{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}
    .row img{max-width:100%;height:auto;border:1px solid #ddd;background:#fff;cursor:zoom-in}
    .domain img{width:min(100%,1000px)} .locmaps img{width:min(48%,620px)} .ts img{width:min(100%,1200px)}
    img.big{width:100%!important;max-width:none;cursor:zoom-out}
    h2{font-size:16px;margin:4px 0 6px} h3{font-size:13px;margin:12px 0 4px;color:#1f3b57}
    .note{font-size:12px;color:#555;margin:0 0 8px}
    """
    js = """
    function show(id){document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===id));
      document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));
      try{localStorage.setItem('atlastab',id)}catch(e){} window.scrollTo(0,0)}
    document.addEventListener('DOMContentLoaded',()=>{let t='domain';try{t=localStorage.getItem('atlastab')||'domain'}catch(e){}
      if(!document.getElementById(t)) t='domain'; show(t);
      document.querySelectorAll('img').forEach(i=>i.addEventListener('click',()=>i.classList.toggle('big')))});
    """
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>CalSim3 validation atlas {e(label)}</title>",
             f"<style>{css}</style><script>{js}</script></head><body>",
             f"<header><h1>CalSim3 validation atlas — {e(label)}</h1>"
             f"<p>Validation window {e(window)}, monthly volume against CalSim3. Tier 1: the twenty trained locations, "
             "FLOW-UNIMPAIRED at the anchored systems and the sum of the member INFLOW arcs elsewhere. Tier 2: every rim "
             "inflow arc on its own, shown under the set it belongs to and, for the arcs outside every set, on the "
             "unconstrained-arcs tab. Red star = the registry outlet of the trained entity.</p></header>",
             "<nav><button data-t='domain' onclick=\"show('domain')\">Domain</button>"]
    ent_of = {s.set_id: s.entity_id for s in sets.itertuples(index=False)}
    parts += [f"<button data-t='{e(sid)}' onclick=\"show('{e(sid)}')\">{e(ent_of[sid])}</button>" for sid in ids]
    if len(t2):
        parts.append("<button data-t='unconstrained' onclick=\"show('unconstrained')\">"
                     "unconstrained arcs</button>")
    if fam_maps:
        parts.append("<button data-t='footprints' onclick=\"show('footprints')\">footprints</button>")
    parts.append("</nav>")
    # domain tab
    parts.append("<section id='domain'><h2>All locations</h2><p class='note'>Click a row to open its tab.</p>")
    parts.append("<table><tr><th>training entity</th><th>location</th><th>reference</th><th>arcs</th><th>mi²</th><th>KGE</th><th>NSE</th>"
                 "<th>bias</th><th>r</th><th>seas. mismatch</th><th>sim / ref TAF/yr</th><th>aggregated</th></tr>")
    for s in sets.itertuples(index=False):
        if s.set_id not in v.index:
            continue
        r = v.loc[s.set_id]
        ref = f"UNIMP {s.system}" if s.ref_kind == "anchor" else "arc sum"
        parts.append(f"<tr class='pick' onclick=\"show('{e(s.set_id)}')\"><td>{e(s.entity_id)}</td><td class='l'>{e(s.name)}</td>"
                     f"<td class='l'>{e(ref)}</td><td>{len(s.arcs)}</td><td>{s.area_mi2:,.0f}</td><td>{r.kge:.3f}</td>"
                     f"<td>{r.nse:.3f}</td><td>{r.pbias:+.1f}%</td><td>{r.r:.3f}</td><td>{r.seas_mismatch:.3f}</td>"
                     f"<td>{r.sim_taf_yr:,.0f} / {r.ref_taf_yr:,.0f}</td><td>{'yes' if s.volume_scored else 'no'}</td></tr>")
    parts.append("</table><p class='note'>Click any image to enlarge it.</p><div class='row domain'>")
    for p in maps:
        parts.append(f"<img src='{_b64(p, jpeg_quality=90)}' alt='{e(p.stem)}'>")
    parts.append("</div>")
    if t2_dir is not None:
        t2maps = [t2_dir / f"tier2_kge_{window}.png", t2_dir / f"tier2_pbias_{window}.png"]
        if any(q.exists() for q in t2maps):
            parts.append("<h3>Tier 2: every rim arc</h3><div class='row domain'>")
            for q in t2maps:
                if q.exists():
                    parts.append(f"<img src='{_b64(q, jpeg_quality=88)}' alt='{e(q.stem)}'>")
            parts.append("</div>")
    parts.append("</section>")
    # one tab per location
    for s in sets.itertuples(index=False):
        sid = s.set_id
        if sid not in v.index:
            continue
        ref = (f"FLOW-UNIMPAIRED {s.system}" if s.ref_kind == "anchor" else f"sum of {len(s.arcs)} INFLOW arcs")
        parts.append(f"<section id='{e(sid)}'><h2>{e(s.entity_id)} — {e(s.name)}</h2>")
        parts.append(f"<p class='note'>Tier-1 set <code>{e(sid)}</code>, trained on <code>{e(s.entity_id)}</code>, {len(s.arcs)} arc(s), {s.area_mi2:,.0f} mi², "
                     f"reference {e(ref)}." + (f" {e(s.note)}." if s.note else "") +
                     (" Not included in the volume-scored aggregate." if not s.volume_scored else "") + "</p>")
        parts.append("<table><tr><th>window</th><th>months</th><th>KGE</th><th>NSE</th><th>bias</th><th>r</th><th>alpha</th>"
                     "<th>beta</th><th>seas. mismatch</th><th>CT diff (mo)</th><th>sim / ref TAF/yr</th></tr>")
        for wl, r in _metrics_rows(metrics, sid, window):
            parts.append(f"<tr><td class='l'>{e(wl)}</td><td>{int(r.n_months)}</td><td>{r.kge:.3f}</td><td>{r.nse:.3f}</td>"
                         f"<td>{r.pbias:+.1f}%</td><td>{r.r:.3f}</td><td>{r.alpha:.2f}</td><td>{r.beta:.2f}</td>"
                         f"<td>{r.seas_mismatch:.3f}</td><td>{r.ct_diff:+.2f}</td><td>{r.sim_taf_yr:,.0f} / {r.ref_taf_yr:,.0f}</td></tr>")
        parts.append("</table>")
        parts.append("<h3>Time series and regimes</h3><div class='row ts'>")
        ts = t1_dir / "figures" / f"{sid}.png"
        if ts.exists():
            parts.append(f"<img src='{_b64(ts, jpeg_quality=88)}' alt='{e(sid)} time series'>")
        parts.append("</div>")
        if len(t2) and "tier1_set" in t2.columns:
            sub2 = t2[t2.tier1_set == sid].sort_values("ref_taf_yr", ascending=False)
            if len(sub2):
                has_cls = "method_class" in sub2.columns
                parts.append(f"<h3>Tier 2: the {len(sub2)} sub-arc(s) of this set, {e(window)}</h3>")
                parts.append("<p class='note'>Each arc's monthly volume against its own CalSim3 INFLOW series. "
                             "Derivation and record share come from the Hydrology Report's Tables 5-1/5-2 and "
                             "5-5/5-6: the share is the fraction of WY1950-84 covered by the gauge record listed for "
                             "the arc, and (donor) marks a record that belongs to the neighbour or downstream gauge "
                             "the arc was split or proportioned from, not to the arc itself. Outside the record the "
                             "reference is a regression on an index gauge with a borrowed monthly shape, so those "
                             "scores are bounded by the reference, not by the model.</p>")
                parts.append("<table><tr><th>arc</th><th>mi²</th>"
                             + ("<th>derivation</th><th>record share</th>" if has_cls else "")
                             + "<th>basis</th><th>KGE</th><th>NSE</th><th>bias</th><th>r</th><th>seas. mismatch</th>"
                               "<th>sim / ref TAF/yr</th></tr>")
                for r in sub2.itertuples(index=False):
                    cls = ""
                    if has_cls:
                        mc = "" if r.method_class != r.method_class else str(r.method_class).replace("_", " ")
                        rf = getattr(r, "record_frac", float("nan"))
                        own = getattr(r, "record_owner", "")
                        own = "" if own != own else str(own)
                        mf = "" if rf != rf else (f"{rf:.0%}" + (" (donor)" if own == "donor" else ""))
                        cls = f"<td class='l'>{e(mc)}</td><td>{mf}</td>"
                    parts.append(f"<tr><td class='l'>{e(r.arc)}</td><td>{r.sq_mi:,.0f}</td>{cls}"
                                 f"<td class='l'>{e(str(r.basis))}</td><td>{_fmt(r.kge, '.3f')}</td>"
                                 f"<td>{_fmt(r.nse, '.3f')}</td><td>{_fmt(r.pbias, '+.0f')}{'' if r.pbias != r.pbias else '%'}</td>"
                                 f"<td>{_fmt(r.r, '.3f')}</td><td>{_fmt(r.seas_mismatch, '.3f')}</td>"
                                 f"<td>{r.sim_taf_yr:,.0f} / {_fmt(r.ref_taf_yr, ',.0f')}</td></tr>")
                parts.append("</table>")
                if t2_dir is not None:
                    fig2 = t2_dir / "figures" / f"tier2_regime_{sid}_{window}.png"
                    if not fig2.exists() and len(sub2) == 1:
                        fig2 = t2_dir / "figures" / f"tier2_regime_single-arc_sets_{window}.png"
                    if fig2.exists():
                        parts.append("<div class='row ts'>"
                                     + f"<img src='{_b64(fig2, jpeg_quality=88)}' alt='{e(sid)} tier-2 regimes'>"
                                     + "</div>")
        parts.append("<h3>Location</h3>")
        parts.append("<p class='note'>Arcs: " + e(", ".join(s.arcs)) + "</p>")
        parts.append("<div class='row locmaps'>")
        for suffix in ("_map", "_zoom"):
            p = out / f"{sid}{suffix}.png"
            if p.exists():
                parts.append(f"<img src='{_b64(p, jpeg_quality=88)}' alt='{e(sid + suffix)}'>")
        parts.append("</div></section>")
    if len(t2):
        ext = t2[t2.basis == "extrapolated"]
        if len(ext):
            n_in_set = int((ext.tier1_set.astype(str) != "").sum()) if "tier1_set" in ext.columns else 0
            parts.append("<section id='unconstrained'><h2>Unconstrained arcs: the regionalization test</h2>")
            parts.append(f"<p class='note'>{len(ext)} arc(s), {ext.ref_taf_yr.sum():,.0f} TAF/yr of CalSim3 rim "
                         f"inflow ({100 * ext.ref_taf_yr.sum() / max(t2.ref_taf_yr.sum(), 1):.1f}% of the total). "
                         "These lie outside every trained entity footprint: their cells are taken from the region "
                         "grid, given flow lengths traced to the polygon exit, and simulated with the trained "
                         "parameter network. Nothing in the loss ever touched them, so they test the "
                         "regionalization alone. The trained-cells column, where present, is the share of each "
                         "arc's cell weight that nonetheless lies inside a trained footprint, which happens where a "
                         "gauged creek overlaps the arc."
                         + (f" {n_in_set} of them belong to a tier-1 set and also appear on its tab." if n_in_set else "")
                         + f" With the {len(t2) - len(ext) + n_in_set} arcs on the set tabs these cover all "
                           f"{len(t2)} scored rim arcs.</p>")
            parts += _t2_table(ext, e, show_set=True, show_cover="trained_cell_frac" in ext.columns)
            if t2_dir is not None:
                fig = t2_dir / "figures" / f"tier2_regime_unconstrained_{window}.png"
                if fig.exists():
                    parts.append("<div class='row ts'>"
                                 + f"<img src='{_b64(fig, jpeg_quality=88)}' alt='unconstrained arc regimes'>"
                                 + "</div>")
                ns = t2_dir / "tier2_not_simulated.csv"
                if ns.exists():
                    miss = pd.read_csv(ns)
                    if len(miss):
                        parts.append(f"<p class='note'>Not simulated: {e(', '.join(miss.iloc[:, 0].astype(str)))}</p>")
            parts.append("</section>")
    if fam_maps:
        parts.append("<section id='footprints'><h2>Training footprints by target family</h2>")
        parts.append("<p class='note'>What the model was fitted to, family by family. Each footprint is the union "
                     "of the 1/16-degree forcing cells assigned to an entity, which is the area whose runoff the "
                     "entity's observation constrains; red dots are the gauge or pour point from the registry, and "
                     "the pale polygons behind are the CalSim3 rim watersheds. Nested entities overlap: the "
                     "Sacramento above Red Bluff contains Shasta, and several CDEC basins contain USGS creeks.</p>")
        for fam, (path, rows) in fam_maps.items():
            parts.append(f"<h3>{e(FAMILY_LABEL.get(fam, fam))} ({len(rows)})</h3>")
            parts.append("<div class='row domain'>"
                         + f"<img src='{_b64(path, jpeg_quality=90)}' alt='{e(fam)} footprints'>"
                         + "</div>")
            parts.append("<table><tr><th>map</th><th>entity</th><th>site</th><th>name</th><th>mi²</th>"
                         "<th>training window</th><th>observations</th></tr>")
            for tag, eid, site, name, area, t0, t1_, nobs in rows:
                parts.append(f"<tr><td>{e(tag)}</td><td class='l'>{e(eid)}</td><td class='l'>{e(site)}</td>"
                             f"<td class='l'>{e(name)}</td><td>{area:,.0f}</td>"
                             f"<td class='l'>{e(t0)} to {e(t1_)}</td><td>{nobs:,}</td></tr>")
            parts.append("</table>")
        parts.append("</section>")
    parts.append("</body></html>")
    path = out / "calsim_validation_atlas.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_markdown(sets, metrics, out: Path, label: str, window: str, maps: list[Path]) -> Path:
    lines = [f"# CalSim3 validation atlas (tier 1) — {label}", "",
             f"Validation window {window}, monthly volume against CalSim3 (FLOW-UNIMPAIRED at the anchored "
             "systems, the sum of the member INFLOW arcs elsewhere). Per location: the domain map with the "
             "arc set highlighted and its registry outlet (red star), a zoom naming the arcs, and the "
             "tier-1 time-series figure. The tabbed version, which also carries tier 2, the unconstrained "
             "arcs and the training footprints, is `calsim_validation_atlas.html`.",
             "", "## Domain maps", ""]
    for p in maps:
        lines += [f"![{p.stem}]({p.name})", ""]
    v = metrics[(metrics.window == window) & metrics.ref_kind.isin(["anchor", "arcsum"])].set_index("set_id")
    lines += ["## Locations", "",
              "| set | location | reference | arcs | mi² | KGE | NSE | bias | r | seas. mismatch | sim / ref TAF/yr | aggregated |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sets.itertuples(index=False):
        if s.set_id not in v.index:
            continue
        r = v.loc[s.set_id]
        ref = f"UNIMP {s.system}" if s.ref_kind == "anchor" else "arc sum"
        lines.append(f"| {s.set_id} | {s.name} | {ref} | {len(s.arcs)} | {s.area_mi2:,.0f} | {r.kge:.3f} | {r.nse:.3f} | "
                     f"{r.pbias:+.1f}% | {r.r:.3f} | {r.seas_mismatch:.3f} | {r.sim_taf_yr:,.0f} / {r.ref_taf_yr:,.0f} | "
                     f"{'yes' if s.volume_scored else 'no'} |")
    lines.append("")
    for s in sets.itertuples(index=False):
        if s.set_id not in v.index:
            continue
        ref = (f"FLOW-UNIMPAIRED {s.system}" if s.ref_kind == "anchor" else f"sum of {len(s.arcs)} INFLOW arcs")
        lines += [f"### {s.set_id} — {s.name}", "",
                  f"Entity `{s.entity_id}`, {len(s.arcs)} arc(s), {s.area_mi2:,.0f} mi², reference {ref}."
                  + (f" {s.note}." if s.note else ""), "",
                  "| window | months | KGE | NSE | bias | r | alpha | beta | seas. mismatch | sim / ref TAF/yr |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for wl, r in _metrics_rows(metrics, s.set_id, window):
            lines.append(f"| {wl} | {int(r.n_months)} | {r.kge:.3f} | {r.nse:.3f} | {r.pbias:+.1f}% | {r.r:.3f} | "
                         f"{r.alpha:.2f} | {r.beta:.2f} | {r.seas_mismatch:.3f} | {r.sim_taf_yr:,.0f} / {r.ref_taf_yr:,.0f} |")
        lines += ["", f"![{s.set_id} map]({s.set_id}_map.png) ![{s.set_id} zoom]({s.set_id}_zoom.png)", "",
                  f"![{s.set_id} time series](../figures/{s.set_id}.png)", ""]
    path = out / "atlas.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tier1_dir", help="a tier-1 output folder (tier1_metrics.csv + figures/<set>.png)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--label", default="")
    p.add_argument("--out", default=None, help="default <tier1_dir>/atlas")
    p.add_argument("--run-dir", default=None,
                   help="run folder with sim_daily_mm.npz, to know which entities trained (for the "
                        "daily/monthly overlap hatching; default <tier1_dir>/..)")
    p.add_argument("--tier2-dir", default=None,
                   help="a tier-2 output folder (tier2_metrics.csv, maps, figures/) to embed per set; "
                        "default <run-dir>/tier2 when it exists")
    p.add_argument("--arc-derivation", default=None,
                   help="per-arc derivation table (default data/calsim/calsim3_arc_derivation.csv when present)")
    a = p.parse_args(argv)
    t1 = Path(a.tier1_dir)
    out = Path(a.out) if a.out else t1 / "atlas"
    out.mkdir(parents=True, exist_ok=True)
    label = a.label or t1.parent.name
    metrics = pd.read_csv(t1 / "tier1_metrics.csv")
    sets = load_sets(a.data_dir)
    catch = _rim(a.data_dir)
    geoms = _set_geoms(catch, sets)
    outlets = _outlets(a.data_dir)
    run_dir = Path(a.run_dir) if a.run_dir else t1.parent
    overlap, from_run = daily_monthly_overlap(a.data_dir, run_dir)
    t2_dir = Path(a.tier2_dir) if a.tier2_dir else run_dir / "tier2"
    t2 = None
    if (t2_dir / "tier2_metrics.csv").exists():
        t2 = pd.read_csv(t2_dir / "tier2_metrics.csv")
        t2 = t2[t2.window == VALIDATION_WINDOW].copy()
        deriv = Path(a.arc_derivation) if a.arc_derivation else Path(a.data_dir) / "calsim" / "calsim3_arc_derivation.csv"
        if deriv.exists():
            keep = [c for c in ("arc", "method_class", "record_frac", "record_owner") if c in pd.read_csv(deriv, nrows=0).columns]
            d = pd.read_csv(deriv)[keep]
            t2 = t2.merge(d, on="arc", how="left")
        # every set's arcs belong to that set's tab, whatever the tier-2 parent assignment was
        s_of = {arc: st.set_id for st in sets.itertuples(index=False) for arc in st.arcs}
        t2["tier1_set"] = t2["arc"].map(s_of).fillna("")
        cov = t2_dir / "tier2_extrapolated_trained_cover.csv"
        if cov.exists():
            c = pd.read_csv(cov).rename(columns={"basin": "arc"})[["arc", "trained_cell_frac"]]
            t2 = t2.merge(c, on="arc", how="left")
        print(f"atlas: tier-2 block from {t2_dir} ({len(t2)} arcs"
              + (", with derivation classes)" if deriv.exists() else ")"))
    else:
        t2_dir = None
    t2_idx = t2.set_index("arc") if t2 is not None else None
    note = "; hatched: cells trained at both daily and monthly timescale" + ("" if from_run else " (all registry entities assumed)")
    maps = domain_maps(catch, sets, geoms, metrics, out, label, VALIDATION_WINDOW, overlap=overlap, overlap_note=note)
    trained = None
    if (run_dir / "sim_daily_mm.npz").exists():
        trained = set(np.load(run_dir / "sim_daily_mm.npz", allow_pickle=True)["entity_id"].tolist())
    fam_maps = family_maps(catch, a.data_dir, out, trained=trained, label=label)
    print("atlas: family footprint maps for " + ", ".join(f"{k} ({len(v[1])})" for k, v in fam_maps.items())
          + ("" if trained else " (all registry entities: the run has no sim_daily_mm.npz)"))
    n = 0
    for s in sets.itertuples(index=False):
        if not (t1 / "figures" / f"{s.set_id}.png").exists():
            print(f"atlas: no tier-1 figure for {s.set_id}, skipped")
            continue
        location_maps(s.set_id, s, catch, geoms, outlets, out, overlap=overlap, t2=t2_idx)
        n += 1
    page = write_html(sets, metrics, out, label, VALIDATION_WINDOW, maps, t1, t2=t2, t2_dir=t2_dir,
                      fam_maps=fam_maps)
    md = write_markdown(sets, metrics, out, label, VALIDATION_WINDOW, maps)
    print(f"wrote {n} location map pairs, {len(maps)} domain maps, {page} ({page.stat().st_size / 1e6:.1f} MB) and {md}")


if __name__ == "__main__":
    main()
