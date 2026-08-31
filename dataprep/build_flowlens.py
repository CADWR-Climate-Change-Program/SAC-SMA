"""Build data/multifamily/flowlens.csv — per-entity traced flow lengths.

Traces each entity cell downstream along the HydroSHEDS v2 1-arcsec flow
directions (TanDEM-X basis) to the entity outlet; ``flowlen_m`` is the
accumulated along-path distance in meters.

Conventions:

- **Start pixel per cell** (the "channel" convention): the highest
  flow-accumulation pixel inside the cell square whose path reaches the
  entity outlet (top-5 candidates by accumulation, then the cell
  center). The value is channel-to-outlet distance — the within-cell
  hillslope delay belongs to the cell's runoff response, not the
  routing UH — and the outlet cell naturally gets ~0 (identity UH).
- **Outlet snap**: the registry outlet coordinate moves to the nearest
  pixel (within ~2 km) whose accumulation-implied upstream area falls
  within [0.2x, 5x] of the registry ``area_mi2``. Nearest-in-band, not
  patch max: below dams the patch max sits on the same river a few km
  downstream of the station coordinate.
- **uf_07** (gauge-less composite of east-side creeks, one entity with
  one outlet per sub-arc): each cell traces until its path exits the
  union of the entity's cell squares — the exit is the creek's valley
  outlet (instantaneous delivery there, per the multi-outlet design).
- **Fallback** for cells none of whose candidates drain to the outlet
  (square-overlap edge cells whose ground truly drains elsewhere):
  ``flowlen_m`` = haversine(cell center -> outlet) x the entity's
  median traced sinuosity; ``method = "fallback"``.

Inputs: data/multifamily/{entities,entity_cells}.csv and the
HydroSHEDS v2 DIR + ACC tiles for the four 10-degree tiles covering the
domain (~6 GB, auto-downloaded to ``tmp/hydrosheds/`` with size
validation; not in git — re-fetched on demand).

Requires ``rasterio`` (pip-installed in the sacsma env). Deliberately
imports NO geopandas/pyogrio: the two GDAL stacks stay in separate
processes.

Usage (sacsma conda env, from the repo root):
    python dataprep/build_flowlens.py [--data-dir data]
        [--tiles-dir tmp/hydrosheds] [--only cdec_MKM,uf_07]
        [--out data/multifamily/flowlens.csv]
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

ARCSEC_M = 30.87  # meters per arc-second of latitude
PX = 1.0 / 3600.0  # 1-arcsec pixel, degrees
HALF = 0.03125  # half a 1/16-degree region cell
TILES = ("n30w120", "n30w130", "n40w120", "n40w130")
URL = "https://data.hydrosheds.org/file/hydrosheds-v2/{p}/1s/{t}_{p}_1s_v2r0.tif"
# ESRI D8: value -> (drow, dcol); 0 = sink, 247/255 = nodata/ocean
D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
      16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
SQMI_PER_KM2 = 0.386102
MAX_STEPS = 4_000_000


_UA = {"User-Agent": "sacsma-dataprep/1.0"}
_SIZES: dict[str, int] = {}


def remote_size(url: str) -> int:
    """Content-Length via HEAD (UA required); range-GET fallback."""
    if url not in _SIZES:
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD", headers=_UA))
            _SIZES[url] = int(r.headers["Content-Length"])
        except urllib.error.HTTPError:
            r = urllib.request.urlopen(urllib.request.Request(
                url, headers={**_UA, "Range": "bytes=0-0"}))
            _SIZES[url] = int(r.headers["Content-Range"].rsplit("/", 1)[1])
    return _SIZES[url]


def _fetch(url: str, dst: Path, want: int) -> None:
    """Streamed download with UA header and byte-range resume."""
    have = dst.stat().st_size if dst.exists() else 0
    if have >= want:
        dst.unlink()
        have = 0
    headers = dict(_UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r:
        resumed = r.status == 206
        with open(dst, "ab" if (have and resumed) else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)


def ensure_tiles(tiles_dir: Path) -> None:
    """Download any missing/partial DIR+ACC tile (size-validated)."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for p in ("DIR", "ACC"):
        for t in TILES:
            url = URL.format(p=p, t=t)
            dst = tiles_dir / url.rsplit("/", 1)[1]
            want = remote_size(url)
            if dst.exists() and dst.stat().st_size == want:
                continue
            print(f"fetching {dst.name} ({want/1e6:.0f} MB)", flush=True)
            for attempt in range(8):
                try:
                    _fetch(url, dst, want)
                except (urllib.error.HTTPError, urllib.error.URLError,
                        ConnectionError, TimeoutError) as e:
                    print(f"  retry {attempt + 1}: {e}", flush=True)
                if dst.exists() and dst.stat().st_size == want:
                    break
            assert dst.stat().st_size == want, (dst, dst.stat().st_size, want)


class Mosaic:
    """Windowed reader over the four 10-degree tiles of one product.

    Opens the local tile when it is complete (size-validated against the
    server); otherwise reads the remote COG — a partially downloaded
    local file must never be read."""

    def __init__(self, tiles_dir: Path, prod: str):
        self.srcs = {}
        for t in TILES:
            url = URL.format(p=prod, t=t)
            local = tiles_dir / url.rsplit("/", 1)[1]
            if local.exists() and local.stat().st_size == remote_size(url):
                self.srcs[t] = rasterio.open(local)
            else:
                self.srcs[t] = rasterio.open("/vsicurl/" + url)

    def read(self, lat0, lat1, lon0, lon1):
        """Return (array, transform) for the bbox; row 0 = north."""
        gx0 = int(np.floor(lon0 / PX))
        gx1 = int(np.ceil(lon1 / PX))
        gy0 = int(np.floor(-lat1 / PX))
        gy1 = int(np.ceil(-lat0 / PX))
        out = None
        for src in self.srcs.values():
            tx0 = round(src.transform.c / PX)
            ty0 = round(-src.transform.f / PX)
            c0 = max(gx0 - tx0, 0); c1 = min(gx1 - tx0, src.width)
            r0 = max(gy0 - ty0, 0); r1 = min(gy1 - ty0, src.height)
            if c1 <= c0 or r1 <= r0:
                continue
            arr = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            if out is None:
                out = np.zeros((gy1 - gy0, gx1 - gx0), dtype=arr.dtype)
            out[r0 + ty0 - gy0:r1 + ty0 - gy0,
                c0 + tx0 - gx0:c1 + tx0 - gx0] = arr
        tr = Affine(PX, 0, gx0 * PX, 0, -PX, -gy0 * PX)
        return out, tr

    def close(self):
        for s in self.srcs.values():
            s.close()


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def snap_outlet(acc_m: Mosaic, lat, lon, area_mi2):
    """Nearest pixel within ~2 km whose implied upstream area is in
    [0.2x, 5x] of area_mi2; widening to [0.05x, 20x] then patch max."""
    k = 66 * PX  # ~2 km
    acc, tr = acc_m.read(lat - k, lat + k, lon - k, lon + k)
    px_area = ARCSEC_M * ARCSEC_M * np.cos(np.deg2rad(lat))  # m^2 per pixel
    imp = np.nan_to_num(acc, nan=0.0) * px_area / 2.589988e6  # mi^2
    inv = ~tr
    jc, jr = inv * (lon, lat)
    for lo, hi in ((0.2, 5.0), (0.05, 20.0)):
        cand = np.argwhere((imp >= lo * area_mi2) & (imp <= hi * area_mi2))
        if len(cand):
            d2 = (cand[:, 0] - jr) ** 2 + (cand[:, 1] - jc) ** 2
            j = tuple(cand[int(np.argmin(d2))])
            break
    else:
        j = np.unravel_index(np.argmax(imp), imp.shape)
    slon, slat = tr * (j[1] + 0.5, j[0] + 0.5)
    return slat, slon, float(imp[j])


def trace_entity(dirr, tr, cells, acc_m, outlet=None, exit_mask=None,
                 area_mi2=None):
    """Trace every cell; returns DataFrame(key, flowlen_m, method).

    outlet=(lat, lon): terminate at that pixel (memoized distances).
    exit_mask: bool array on dirr's grid — terminate on leaving True.
    area_mi2: entity area — candidate pixels are limited to implied
    upstream area <= 1.3x it (a pixel carrying more water than the whole
    basin cannot drain to its outlet; decisive for sub-cell basins).
    """
    H, W = dirr.shape
    inv = ~tr
    lat_row = np.array([(tr * (0, r + 0.5))[1] for r in range(H)])
    dy = ARCSEC_M
    dxr = ARCSEC_M * np.cos(np.deg2rad(lat_row))
    dist = np.full((H, W), np.nan, dtype=np.float32)
    lost = np.zeros((H, W), dtype=bool)  # pixels known NOT to reach the outlet

    def px_of(lat, lon):
        c, r = inv * (lon, lat)
        return int(r), int(c)

    if outlet is not None:
        orow, ocol = px_of(*outlet)
        if 0 <= orow < H and 0 <= ocol < W:
            dist[orow, ocol] = 0.0

    def trace(r, c):
        path, rr, cc = [], r, c
        d = np.nan
        for _ in range(MAX_STEPS):
            inside = 0 <= rr < H and 0 <= cc < W
            if exit_mask is not None and (not inside or not exit_mask[rr, cc]):
                d = 0.0  # left the entity footprint = reached its outlet
                break
            if not inside or lost[rr, cc]:
                break
            if not np.isnan(dist[rr, cc]):
                d = dist[rr, cc]
                break
            v = int(dirr[rr, cc])
            if v not in D8:
                break
            drow, dcol = D8[v]
            step = np.hypot(drow * dy, dcol * dxr[rr]) if (drow and dcol) \
                else (dy if drow else dxr[rr])
            path.append((rr, cc, step))
            rr, cc = rr + drow, cc + dcol
        if np.isnan(d):
            for pr, pc, _ in path:
                lost[pr, pc] = True
            return np.nan
        total = d
        for pr, pc, st in reversed(path):
            total += st
            dist[pr, pc] = total
        return dist[r, c] if path else d

    rows = []
    for cr in cells.itertuples():
        acc, atr = acc_m.read(cr.lat - HALF, cr.lat + HALF,
                              cr.lon - HALF, cr.lon + HALF)
        imp = np.nan_to_num(acc, nan=-1.0)
        if area_mi2 is not None:
            px_area = ARCSEC_M * ARCSEC_M * np.cos(np.deg2rad(cr.lat))
            imp = np.where(imp * px_area / 2.589988e6 <= 1.3 * area_mi2,
                           imp, -1.0)
        flat = np.argsort(imp.ravel())[::-1][:20]
        flat = flat[imp.ravel()[flat] > 0]
        cand = []
        for i in flat:
            plon, plat = atr * (i % acc.shape[1] + 0.5, i // acc.shape[1] + 0.5)
            cand.append(px_of(plat, plon))
        cand.append(px_of(cr.lat, cr.lon))
        got, method, s_lat, s_lon = np.nan, "fallback", np.nan, np.nan
        for n, (r, c) in enumerate(cand):
            if not (0 <= r < H and 0 <= c < W):
                continue
            got = trace(r, c)
            if not np.isnan(got):
                method = "channel" if n < len(cand) - 1 else "center"
                s_lon, s_lat = tr * (c + 0.5, r + 0.5)
                break
        rows.append((cr.key, got, method, s_lat, s_lon))
    return pd.DataFrame(rows, columns=["key", "flowlen_m", "method",
                                       "start_lat", "start_lon"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=Path("data"), type=Path)
    ap.add_argument("--tiles-dir", default=Path("tmp/hydrosheds"), type=Path)
    ap.add_argument("--only", default="",
                    help="comma-separated entity_ids (testing)")
    ap.add_argument("--out", default=Path("data/multifamily/flowlens.csv"),
                    type=Path)
    ap.add_argument("--debug-out", default=None, type=Path,
                    help="also write flowlen_starts.csv (per-cell start "
                         "pixel) and flowlen_outlets.csv (snapped outlets) "
                         "to this directory")
    args = ap.parse_args()

    if not args.only:  # test runs may not have complete local tiles yet
        ensure_tiles(args.tiles_dir)
    ent = pd.read_csv(args.data_dir / "multifamily" / "entities.csv",
                      dtype={"site_id": str})
    cells = pd.read_csv(args.data_dir / "multifamily" / "entity_cells.csv")
    if args.only:
        keep = args.only.split(",")
        ent = ent[ent.entity_id.isin(keep)]
    dir_m = Mosaic(args.tiles_dir, "DIR")
    acc_m = Mosaic(args.tiles_dir, "ACC")

    frames, report = [], []
    for e in ent.itertuples():
        sub = cells[cells.entity_id == e.entity_id].reset_index(drop=True)
        m = 0.10
        lat0, lat1 = sub.lat.min() - m, sub.lat.max() + m
        lon0, lon1 = sub.lon.min() - m, sub.lon.max() + m
        acc_imp = np.nan
        if pd.notna(e.outlet_lat):
            lat0 = min(lat0, e.outlet_lat - m); lat1 = max(lat1, e.outlet_lat + m)
            lon0 = min(lon0, e.outlet_lon - m); lon1 = max(lon1, e.outlet_lon + m)
            slat, slon, acc_imp = snap_outlet(
                acc_m, e.outlet_lat, e.outlet_lon, e.area_mi2)
            outlet, exit_mask = (slat, slon), None
            dirr, tr = dir_m.read(lat0, lat1, lon0, lon1)
        else:  # uf_07: exit-of-footprint termination
            outlet = None
            dirr, tr = dir_m.read(lat0, lat1, lon0, lon1)
            exit_mask = np.zeros(dirr.shape, dtype=bool)
            inv = ~tr
            for cr in sub.itertuples():
                c0, r0 = inv * (cr.lon - HALF, cr.lat + HALF)
                c1, r1 = inv * (cr.lon + HALF, cr.lat - HALF)
                exit_mask[max(int(r0), 0):int(r1) + 1,
                          max(int(c0), 0):int(c1) + 1] = True
        res = trace_entity(dirr, tr, sub, acc_m, outlet, exit_mask,
                           e.area_mi2 if outlet is not None else None)
        res.insert(0, "entity_id", e.entity_id)

        # fallback: haversine x entity median traced sinuosity
        ok = res.flowlen_m.notna() & (res.method != "fallback")
        if outlet is not None:
            straight = haversine_m(sub.lat, sub.lon, outlet[0], outlet[1])
            sin_med = float(np.median(
                (res.flowlen_m[ok] / straight[ok]).clip(1.0, 3.0))) if ok.any() else 1.4
            res.loc[~ok, "flowlen_m"] = (straight[~ok] * sin_med).round(1)
        else:
            sin_med = np.nan
            res.loc[~ok, "flowlen_m"] = res.flowlen_m[ok].median()
        res["flowlen_m"] = res.flowlen_m.round(1)

        fb_w = sub.loc[(~ok).to_numpy(), "overlap_mi2"].sum()
        report.append((e.entity_id, len(sub), int(ok.sum()),
                       100 * fb_w / sub.overlap_mi2.sum(),
                       acc_imp / e.area_mi2 if pd.notna(e.outlet_lat) else np.nan,
                       sin_med,
                       outlet[0] if outlet else np.nan,
                       outlet[1] if outlet else np.nan))
        frames.append(res)
        print(f"{e.entity_id:16s} cells {len(sub):4d} traced {int(ok.sum()):4d} "
              f"fb_w {100 * fb_w / sub.overlap_mi2.sum():5.1f}% "
              f"snapACC/area {report[-1][4]:6.2f} sin {sin_med:4.2f}"
              if pd.notna(e.outlet_lat) else
              f"{e.entity_id:16s} cells {len(sub):4d} traced {int(ok.sum()):4d} "
              f"(exit-mode)")
    dir_m.close(); acc_m.close()

    df = pd.concat(frames, ignore_index=True)
    rep = pd.DataFrame(report, columns=[
        "entity_id", "n_cells", "n_traced", "fallback_wpct", "accarea_ratio",
        "sinuosity_med", "snap_lat", "snap_lon"])
    if args.debug_out:
        args.debug_out.mkdir(parents=True, exist_ok=True)
        df[["entity_id", "key", "method", "start_lat", "start_lon"]].to_csv(
            args.debug_out / "flowlen_starts.csv", index=False)
        rep.to_csv(args.debug_out / "flowlen_outlets.csv", index=False)
    df = df[["entity_id", "key", "flowlen_m", "method"]]
    rep = rep.drop(columns=["snap_lat", "snap_lon"])

    # ---- gates -----------------------------------------------------------
    full = not args.only
    assert df.flowlen_m.notna().all() and (df.flowlen_m >= 0).all()
    assert df.flowlen_m.max() < 7e5, df.flowlen_m.max()
    if full:
        want = cells[["entity_id", "key"]]
        assert len(df) == len(want) and not df.duplicated(
            ["entity_id", "key"]).any()
        assert set(map(tuple, df[["entity_id", "key"]].values)) \
            == set(map(tuple, want.values))
        # Fallback is the DESIGNED handling for ground that does not drain
        # through the outlet (below-outlet valley cells; square-overlap
        # edge cells; sub-cell basins), so it is reported, not forbidden.
        # Gates: per-entity sanity (<50% of weight on fallback for any
        # entity bigger than a few cells) and a global ceiling.
        high = rep[rep.fallback_wpct > 25].sort_values(
            "fallback_wpct", ascending=False)
        if len(high):
            print("entities with >25% fallback weight (reported):")
            print(high.to_string(index=False))
        bad = rep[(rep.fallback_wpct > 50) & (rep.n_cells > 4)]
        assert bad.empty, bad.to_string(index=False)
        w = cells.merge(df[["entity_id", "key", "method"]],
                        on=["entity_id", "key"])
        gshare = (w.loc[w.method == "fallback", "overlap_mi2"].sum()
                  / w.overlap_mi2.sum())
        print(f"global fallback weight share: {100 * gshare:.1f}%")
        assert gshare < 0.12, gshare
        # archived CADWR comparison, where an archive exists
        hru = pd.read_csv(args.data_dir / "cdec15_grid" / "hruinfo.csv")
        for b, g in hru.groupby("basin"):
            mm = df[df.entity_id == f"cdec_{b}"].merge(
                g[["key", "flowlen"]], on="key")
            if len(mm) > 5:
                r = np.corrcoef(mm.flowlen_m, mm.flowlen)[0, 1]
                print(f"vs CADWR {b}: n={len(mm)} r={r:.4f} "
                      f"med ratio={float((mm.flowlen_m/mm.flowlen).median()):.3f}")
    print(rep.sort_values("fallback_wpct", ascending=False)
          .head(8).to_string(index=False))

    if full or args.out.name != "flowlens.csv":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}: {len(df)} rows, "
              f"{df.entity_id.nunique()} entities")
    else:
        print("--only run: not writing the store")


if __name__ == "__main__":
    main()
