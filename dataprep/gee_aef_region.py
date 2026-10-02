"""REGION AlphaEarth satellite embeddings (Google Earth Engine) -> one multi-year mean per 1/16-deg cell.

Source: ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` -- AlphaEarth Foundations
(Brown et al. 2025, arXiv:2507.22291), CC-BY 4.0: "The AlphaEarth Foundations
Satellite Embedding dataset is produced by Google and Google DeepMind."  One
image per UTM tile (163.84 km, 10 m) per CALENDAR year 2017-2025; bands
A00..A63 are a unit-length 64-d embedding per pixel (stored int8, served
dequantized as sign(q)*(q/127.5)**2).  DATASET_VERSION is per year (1.1 for
2017 + 2025, 1.0 for 2018-2024 as of 2026-09) -- recorded from the images,
not the catalog prose.

Product: ``data/region/aef/aef_cell_mean.npz`` -- for each of the 4410 region
cells the mean of the pixel vectors over the cell rectangle, then averaged
over the years (equal weight per year): a single static snapshot.  The
per-year cell means stay in the local partials (``tmp/aef_parts``).

**Method (measured and adversarially reviewed 2026-09-22 before the burn):**

- **No mosaic().**  The -120 deg UTM 10N/11N line is a cell edge, so every
  cell lies in one zone and is reduced on its own zone's tiles only, in the
  tile's native UTM grid; the per-tile weighted sums are combined.  Tiles of
  one zone never overlap (every year: 28 tiles, 16 x 10N + 12 x 11N, on the
  163.84 km grid) and cover every cell exactly: sum(w) / expected = 1 to 1e-15
  on tile corners and the -120 cells.  At 10 m this equals the zone mosaic's
  mean to 1e-12, at ~1/3 fewer EECU.  (The other zone carries ~84 m of valid,
  slightly different pixels past -120; they are never used.)
- **scale 15 m, native UTM** (``--scale`` accepts only 10 or 15).  EE's
  pyramid levels >= 20 m are L2-RENORMALIZED block means (despite
  pyramidingPolicy MEAN in the asset metadata): asking for 19-20 m reads them
  and inflates |mean| by ~4e-3.  At 15 m EE reads the full-resolution level on
  a nearest-neighbour lattice that takes 4/9 of the 10 m pixels: <= 2e-4 per
  band (<= 9e-5 on tile-corner / zone-edge / lake / grid-extreme cells, 1.6-1.9e-4
  on 90 random cells; a tiny same-sign bias), cos >= 0.9999998, |mean| unchanged
  (<= 6e-5), vs the full 10 m mean, at
  12.8 vs 23.3 EECU-s per cell-year -- the whole burn is ~140 EECU-h (the
  noncommercial Community tier is 150 EECU-h/month).  16 m aliases (1e-3), so
  never "any value below 19".  ``--check`` re-verifies against an independent
  zone-mosaic reduction at 10 m.
- mean = sum(v*w) / sum(w) with EE's fractional boundary weights (``w`` is a
  constant band carrying A00's exact mask); ``valid_frac`` = sum(w) / the same
  sum of an unmasked constant on the same grid.

**Resume / safety.**  Each (year, unit) is banked atomically as
``tmp/aef_parts/<year>/<unit>.npz`` with the cell KEYS it holds; a re-run skips
banked units.  An empty or null result is never banked.  ``run.json`` pins
``--chunk``/``--scale`` for the whole burn and ``run.lock`` stops two runs from
spending quota on the same units.  Ctrl+C cancels the queued units (in-flight
requests finish and are banked).

Needs an authenticated earthengine-api (``sacsma`` env) and a registered cloud
project (``--project``).  Behind the DWR TLS proxy the Windows trust store has
to be injected before ``import ee`` -- done below when pip's vendored
truststore is importable.

Usage
-----
    python dataprep/gee_aef_region.py --dry-run                      # what would run
    python dataprep/gee_aef_region.py --project <ee-project> --years 2021 --max-units 2   # smoke test
    python dataprep/gee_aef_region.py --project <ee-project> --check 30 --check-year 2021
    python dataprep/gee_aef_region.py --project <ee-project>           # the burn, resumable
    python dataprep/gee_aef_region.py --status
    python dataprep/gee_aef_region.py --assemble
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import local_value

COLL = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]
YEARS = list(range(2017, 2026))
ZONES = {"10N": "EPSG:32610", "11N": "EPSG:32611"}
ZONE_EDGE = -120.0                         # 10N | 11N, a cell edge on the 1/16 lattice
CELL_DEG = 1.0 / 16.0
SCALES = (10.0, 15.0)                      # the verified reduction scales (see docstring)
ATTRIBUTION = ("The AlphaEarth Foundations Satellite Embedding dataset is produced "
               "by Google and Google DeepMind. (CC-BY 4.0; Brown et al. 2025, "
               "arXiv:2507.22291)")

REPO = Path(__file__).resolve().parents[1]
GRID_CSV = REPO / "data" / "region" / "grid_cells.csv"
OUT = REPO / "data" / "region" / "aef" / "aef_cell_mean.npz"
PARTS = REPO / "tmp" / "aef_parts"         # tmp/ is gitignored, local-only

#: EE error text -> what to do.  SPLIT: the request is too big, halve it (no
#: retry).  FATAL: deterministic, abort the burn.  Anything else (429 / 5xx /
#: "too many concurrent aggregations" / dropped connections, and the reduced
#: concurrency of an over-quota "restricted" project) is retried with backoff.
SPLIT = ("timed out", "deadline", "memory limit", "too large")
FATAL = ("permission", "not found", "not authorized", "does not have", "did not match",
         "not registered", "invalid argument", "unknown band")


class Fatal(RuntimeError):
    """An EE error that retrying cannot fix."""


# ------------------------------------------------------------------ small utils
def _atomic_npz(path: Path, **arrays) -> None:
    """Write ``arrays`` so a kill mid-write can never leave a half-file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)


def _load(path: Path) -> dict:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:                               # noqa: BLE001
        sys.exit(f"{path}: unreadable ({e}) -- delete it and re-run")


def _ee(project: str | None):
    try:
        import pip._vendor.truststore as truststore     # DWR TLS proxy
        truststore.inject_into_ssl()
    except Exception:                                    # noqa: BLE001
        pass
    try:
        import ee
        ee.Initialize(project=project)
        # EE's interactive limit is 5 min; without a client deadline a dropped
        # connection hangs forever (seen once in the 2026-09-22 burn: 1 unit, >1 h).
        ee.data.setDeadline(360_000)
    except Exception as e:                               # noqa: BLE001
        sys.exit(f"earthengine-api not ready ({e}); run `earthengine authenticate` "
                 "and pass --project <an-EE-registered-cloud-project>")
    return ee


def _grid() -> pd.DataFrame:
    g = pd.read_csv(GRID_CSV)
    g["key"] = g["key"].astype(str)
    g["zone"] = np.where(g["lon"] < ZONE_EDGE, "10N", "11N")
    h = CELL_DEG / 2.0
    straddle = (g["lon"] - h < ZONE_EDGE) & (g["lon"] + h > ZONE_EDGE)
    if straddle.any() or g["lon"].min() - h < -126.0 or g["lon"].max() + h > -114.0:
        sys.exit("grid_cells.csv: a cell straddles a UTM zone edge or leaves 10N/11N")
    return g


def _units(g: pd.DataFrame, chunk: int) -> list[tuple[str, str, np.ndarray]]:
    """Work units: (zone, name, row indices) -- one 1-degree block of one zone,
    split to <= ``chunk`` cells, so a request touches few tiles."""
    out = []
    for zone in ZONES:
        z = g[g["zone"] == zone]
        blk = (np.floor(z["lat"]).astype(int).astype(str) + "_"
               + (-np.floor(z["lon"])).astype(int).astype(str))
        for b, rows in z.groupby(blk, sort=True):
            idx = rows.sort_values(["lat", "lon"]).index.to_numpy()
            for k, lo in enumerate(range(0, len(idx), chunk)):
                out.append((zone, f"{zone}_{b}_{k}", idx[lo:lo + chunk]))
    return out


def _fc(ee, g: pd.DataFrame, idx: np.ndarray):
    """Cell rectangles exactly as dataprep/gee_obs_region.py builds them."""
    h = CELL_DEG / 2.0
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Rectangle([g.lon[i] - h, g.lat[i] - h, g.lon[i] + h, g.lat[i] + h],
                                         proj="EPSG:4326", geodesic=False), {"i": int(i)})
        for i in idx])


def _bbox(ee, g: pd.DataFrame, idx: np.ndarray):
    h = CELL_DEG / 2.0
    s = g.loc[idx]
    return ee.Geometry.Rectangle([s.lon.min() - h, s.lat.min() - h, s.lon.max() + h, s.lat.max() + h],
                                 proj="EPSG:4326", geodesic=False)


def _retry(fn, what: str, tries: int = 4):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:                           # noqa: BLE001
            msg = str(e).lower()
            if any(s in msg for s in SPLIT):
                raise
            if any(s in msg for s in FATAL):
                raise Fatal(f"{what}: {e}") from e
            if k == tries - 1:
                raise
            wait = min(300, 30 * 2 ** k)
            print(f"    retry {what} in {wait}s ({str(e)[:120]})", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _lock() -> Path:
    """One burn at a time.  A lock left by a killed run is taken over when its
    PID is gone."""
    lock = PARTS / "run.lock"
    PARTS.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock
        except FileExistsError:
            pid = lock.read_text().strip()
            if os.name == "nt":
                alive = pid in subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                              capture_output=True, text=True).stdout
            else:
                alive = Path(f"/proc/{pid}").exists()
            if alive:
                sys.exit(f"another run (PID {pid}) holds {lock}")
            print(f"  taking over the stale lock of PID {pid}", flush=True)
            lock.unlink(missing_ok=True)
    sys.exit(f"could not take {lock}")


def _manifest(chunk: int, scale: float) -> None:
    """Pin chunk + scale + the grid for the whole burn (unit names and cell
    sets must not change between runs)."""
    want = {"chunk": chunk, "scale": scale,
            "grid_sha1": hashlib.sha1(GRID_CSV.read_bytes()).hexdigest()}
    path = PARTS / "run.json"
    if path.exists():
        have = json.loads(path.read_text())
        diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
        if diff:
            sys.exit(f"{path} was banked with a different setup {diff} (banked, now): "
                     "finish with the banked settings, or move tmp/aef_parts aside")
    else:
        PARTS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(want, indent=1))


# --------------------------------------------------------------- the reduction
def _reduce(ee, g: pd.DataFrame, year: int, zone: str, idx: np.ndarray, scale: float):
    """Per-tile weighted sums of the embedding over cells ``idx`` (one zone, one
    year) -> (S [n,64], W [n], image ids).  Halves the request when EE says it
    is too big."""
    fc = _fc(ee, g, idx)
    coll = (ee.ImageCollection(COLL).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filter(ee.Filter.eq("UTM_ZONE", zone)).filterBounds(_bbox(ee, g, idx)))

    def per_tile(img):
        w = img.select(0).multiply(0).add(1).rename("w")       # A00's exact mask
        return (img.addBands(w)
                .reduceRegions(collection=fc.filterBounds(img.geometry()),
                               reducer=ee.Reducer.sum(), crs=img.projection().crs(),
                               scale=scale)
                .map(lambda f: f.setGeometry(None).set("img", img.get("system:index"))))

    try:
        res = _retry(lambda: coll.map(per_tile).flatten().getInfo(),
                     f"{year} {zone} n={len(idx)}")
    except Fatal:
        raise
    except Exception as e:                               # noqa: BLE001
        if not any(s in str(e).lower() for s in SPLIT) or len(idx) < 2:
            raise
        h = len(idx) // 2
        print(f"    {year} {zone}: too big at n={len(idx)} -> split", flush=True)
        a = _reduce(ee, g, year, zone, idx[:h], scale)
        b = _reduce(ee, g, year, zone, idx[h:], scale)
        return np.vstack([a[0], b[0]]), np.concatenate([a[1], b[1]]), sorted(set(a[2]) | set(b[2]))
    pos = {int(i): k for k, i in enumerate(idx)}
    S = np.zeros((len(idx), 64))
    W = np.zeros(len(idx))
    imgs = set()
    for f in res["features"]:
        p = f["properties"]
        k = pos[int(p["i"])]
        v = [p[b] for b in BANDS]            # Reducer.sum gives 0, never null, on no data:
        if p["w"] is None or any(x is None for x in v):   # a None or a missing key is a
            raise RuntimeError(f"{year} {zone}: null band sum (cell row {p['i']})")  # schema change
        S[k] += v
        W[k] += p["w"]
        imgs.add(p["img"])
    return S, W, sorted(imgs)


def _inventory(ee, g: pd.DataFrame, years: list[int]) -> None:
    """Per year: the tile images over the grid + their version properties."""
    h = CELL_DEG / 2.0
    box = ee.Geometry.Rectangle([g.lon.min() - h, g.lat.min() - h, g.lon.max() + h, g.lat.max() + h],
                                proj="EPSG:4326", geodesic=False)
    for y in years:
        path = PARTS / f"inventory_{y}.json"
        if path.exists():
            continue
        c = (ee.ImageCollection(COLL).filterDate(f"{y}-01-01", f"{y + 1}-01-01")
             .filterBounds(box).filter(ee.Filter.inList("UTM_ZONE", list(ZONES))))
        props = ["UTM_ZONE", "DATASET_VERSION", "MODEL_VERSION", "PROCESSING_SOFTWARE_VERSION"]
        rows = _retry(lambda: c.map(lambda im: ee.Feature(None, im.toDictionary(props).set(
            "id", im.get("system:index")))).getInfo(), f"inventory {y}")
        recs = [f["properties"] for f in rows["features"]]
        PARTS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(recs, indent=1))
        vers = sorted({(r.get("DATASET_VERSION"), r.get("MODEL_VERSION"),
                        r.get("PROCESSING_SOFTWARE_VERSION")) for r in recs})
        print(f"  inventory {y}: {len(recs)} tiles, versions {vers}", flush=True)


def _expected(ee, g: pd.DataFrame, scale: float) -> None:
    """Fractional pixel count of each FULL cell on the same grid (unmasked
    constant) -> the denominator of valid_frac.  Costs ~nothing to compute."""
    path = PARTS / f"expected_{scale:g}m.npz"
    if path.exists():
        return
    n = np.full(len(g), np.nan)
    for zone, crs in ZONES.items():
        idx = g.index[g["zone"] == zone].to_numpy()
        for lo in range(0, len(idx), 1000):
            sub = idx[lo:lo + 1000]
            r = _retry(lambda: ee.Image.constant(1).rename("n").reduceRegions(
                collection=_fc(ee, g, sub), reducer=ee.Reducer.sum(), crs=crs, scale=scale)
                .map(lambda f: f.setGeometry(None)).getInfo(), f"expected {zone}")
            for f in r["features"]:   # one band, one reducer -> property "sum"
                n[int(f["properties"]["i"])] = f["properties"]["sum"]
    if np.isnan(n).any():
        sys.exit(f"expected counts missing for {int(np.isnan(n).sum())} cells")
    _atomic_npz(path, n=n, keys=g["key"].to_numpy().astype("U"))
    print(f"  expected pixel counts @ {scale:g} m: median {np.nanmedian(n):.0f}", flush=True)


def _todo(g: pd.DataFrame, years: list[int], chunk: int, max_units: int | None):
    units = _units(g, chunk)
    if max_units is not None:
        units = units[:max_units]
    return [(y, z, name, idx) for y in years for z, name, idx in units
            if not (PARTS / str(y) / f"{name}.npz").exists()], len(units)


def run(ee, years: list[int], chunk: int, scale: float, workers: int,
        max_units: int | None) -> None:
    g = _grid()
    lock = _lock()
    try:
        _manifest(chunk, scale)
        _inventory(ee, g, years)
        _expected(ee, g, scale)
        todo, nunits = _todo(g, years, chunk, max_units)
        print(f"{len(todo)} units to reduce ({nunits} per year x {len(years)} years; "
              f"<= {chunk} cells each) @ {scale:g} m, {workers} workers", flush=True)
        keys = g["key"].to_numpy().astype("U")

        def one(job):
            y, z, name, idx = job
            t0 = time.time()
            S, W, imgs = _reduce(ee, g, y, z, idx, scale)
            if not imgs or (W <= 0).any():   # every cell is fully covered, every year
                raise RuntimeError(f"{int((W <= 0).sum())}/{len(idx)} cells without valid "
                                   f"pixels (tiles {imgs}) -- not banked")
            _atomic_npz(PARTS / str(y) / f"{name}.npz", idx=idx, keys=keys[idx], S=S, W=W,
                        imgs=np.array(imgs), scale=np.array(scale))
            return len(idx), time.time() - t0

        done = failed = streak = 0
        t0 = time.time()
        ex = cf.ThreadPoolExecutor(workers)
        futs = {ex.submit(one, j): j for j in todo}
        try:
            for fut in cf.as_completed(futs):
                y, _, name, _ = futs[fut]
                try:
                    n, dt = fut.result()
                except Fatal as e:
                    print(f"  FATAL {y} {name}: {e}", flush=True)
                    raise
                except Exception as e:                   # noqa: BLE001
                    failed += 1
                    streak += 1
                    print(f"  FAIL {y} {name}: {str(e)[:300]}", flush=True)
                    if streak >= 2 * workers:
                        raise RuntimeError(f"{streak} consecutive failures -- stopping")
                    continue
                done += 1
                streak = 0
                print(f"  {done}/{len(todo)} {y} {name}: {n} cells {dt:5.1f}s  "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        except BaseException:
            print("stopping: cancelling queued units (in-flight ones finish and are banked)",
                  flush=True)
            ex.shutdown(wait=True, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    finally:
        lock.unlink(missing_ok=True)
    status()
    if failed:
        sys.exit(f"{failed} units failed (not banked) -- re-run to retry them")


def status() -> None:
    g = _grid()
    man = PARTS / "run.json"
    print(f"  setup: {man.read_text().strip() if man.exists() else '(no run yet)'}")
    for y in YEARS:
        d = PARTS / str(y)
        parts = sorted(d.glob("*.npz")) if d.exists() else []
        seen = np.zeros(len(g), int)
        for p in parts:
            seen[_load(p)["idx"]] += 1
        dup = int((seen > 1).sum())
        print(f"  {y}: {len(parts):4d} partials, {int((seen > 0).sum()):5d}/{len(g)} cells"
              + (f"  ({dup} banked twice!)" if dup else ""))


# ----------------------------------------------------------------- assembling
def _year_means(g: pd.DataFrame, y: int, scale: float):
    """One year's cell means from its partials, placed by cell KEY (each cell
    exactly once)."""
    gi = pd.Index(g["key"])
    S = np.zeros((len(g), 64))
    W = np.zeros(len(g))
    seen = np.zeros(len(g), int)
    imgs: set[str] = set()
    for p in sorted((PARTS / str(y)).glob("*.npz")):
        z = _load(p)
        if float(z["scale"]) != scale:
            sys.exit(f"{p}: banked at {float(z['scale']):g} m, assembling {scale:g} m")
        pos = gi.get_indexer(z["keys"].astype(str))
        if (pos < 0).any():
            sys.exit(f"{p}: {int((pos < 0).sum())} banked keys not in grid_cells.csv "
                     "(grid changed since banking?)")
        S[pos] += z["S"]
        W[pos] += z["W"]
        seen[pos] += 1
        imgs |= set(z["imgs"].tolist())
    if (seen > 1).any():
        sys.exit(f"{y}: {int((seen > 1).sum())} cells banked twice -- delete "
                 f"tmp/aef_parts/{y} and re-run that year")
    with np.errstate(invalid="ignore", divide="ignore"):
        return S / W[:, None], W, seen, imgs


def assemble(scale: float) -> None:
    g = _grid()
    man = PARTS / "run.json"
    if man.exists() and json.loads(man.read_text()).get("scale") != scale:
        sys.exit(f"the partials were banked at {json.loads(man.read_text())['scale']:g} m "
                 f"(run.json), not {scale:g} m")
    ep = PARTS / f"expected_{scale:g}m.npz"
    if not ep.exists():
        sys.exit(f"no expected counts at {scale:g} m ({ep}) -- nothing banked at this scale")
    e = _load(ep)
    exp = pd.Series(e["n"], index=e["keys"].astype(str)).reindex(g["key"]).to_numpy()
    means, fracs, vers, ntiles = [], [], [], []
    for y in YEARS:
        m, W, seen, imgs = _year_means(g, y, scale)
        if (seen == 0).any():
            sys.exit(f"{y}: {int((seen == 0).sum())} cells not banked yet -- finish the run")
        inv_path = PARTS / f"inventory_{y}.json"
        if not inv_path.exists():
            sys.exit(f"{inv_path} missing -- re-run to rebuild the inventory")
        v = {r["id"]: r.get("DATASET_VERSION") for r in json.loads(inv_path.read_text())}
        unknown = sorted(set(imgs) - set(v))
        vset = sorted({str(v.get(i)) for i in imgs})
        if unknown or len(vset) != 1:   # the collection was republished mid-burn
            sys.exit(f"{y}: tiles {unknown[:3]} not in {inv_path.name} or mixed "
                     f"DATASET_VERSION {vset} -- delete tmp/aef_parts/{y} + its inventory "
                     "and re-run that year")
        vers.append("/".join(vset))
        ntiles.append(len(imgs))
        means.append(m)
        fracs.append(W / exp)
    E = np.stack(means, axis=1)                          # (cells, years, 64)
    F = np.stack(fracs, axis=1)                          # (cells, years)
    if np.nanmax(F) > 1 + 1e-6:  # tiles of one zone overlapping would double-count
        c, yy = np.unravel_index(np.nanargmax(F), F.shape)
        sys.exit(f"valid_frac {np.nanmax(F):.6f} > 1 at {g['key'][c]} {YEARS[yy]}: "
                 "a pixel was counted twice")
    ok = F > 0
    if not ok.all():
        print(f"  WARNING {int((~ok).sum())} cell-years without valid pixels (left out of the mean)")
    n_years = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        emb = np.where(ok[..., None], E, 0.0).sum(axis=1) / n_years[:, None]
        norm = np.linalg.norm(emb, axis=1)
        cos = (np.einsum("cyk,ck->cy", E, emb)
               / (np.linalg.norm(E, axis=2) * norm[:, None]))
    how = ("full 10 m" if scale == 10 else
           "a 15 m nearest-neighbour lattice of the full-resolution level (4/9 of the 10 m "
           "pixels; <= 2e-4 per band, cos >= 0.9999998 vs the full 10 m mean)")
    meta = (f"exported {pd.Timestamp.today().date()} | {COLL} A00..A63, calendar years "
            f"{YEARS[0]}-{YEARS[-1]}, per-year DATASET_VERSION in `dataset_version` | per "
            f"cell and year: mean of the pixel vectors over the 1/16-deg rectangle, each "
            f"cell reduced on its own UTM zone's tiles in the native UTM grid (no mosaic), "
            f"read on {how}; `emb` = the equal-weight mean over the years | NOT unit length: "
            f"`norm` < 1 measures within-cell (and between-year) heterogeneity | "
            f"{ATTRIBUTION}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, keys=g["key"].to_numpy(dtype=object), lat=g["lat"].to_numpy(),
        lon=g["lon"].to_numpy(), bands=np.array(BANDS), years=np.array(YEARS),
        dataset_version=np.array(vers), emb=emb.astype(np.float32),
        norm=norm.astype(np.float32), n_years=n_years.astype(np.int8),
        valid_frac=F.astype(np.float32), year_cos_min=np.nanmin(cos, axis=1).astype(np.float32),
        meta=np.array(meta))
    bad = int(np.isnan(norm).sum())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB): {len(g)} cells x 64"
          + (f" ({bad} cells NaN)" if bad else "")
          + f", norm {np.nanmin(norm):.3f}-{np.nanmax(norm):.3f} (median {np.nanmedian(norm):.3f}), "
          f"valid_frac min {np.nanmin(F):.4f}, year-vs-mean cos min {np.nanmin(cos):.3f} "
          f"(median {np.nanmedian(cos):.3f}); tiles/yr {ntiles}; versions {vers}")


# ------------------------------------------------------------- verification
def check(ee, n: int, year: int, scale: float, seed: int = 0) -> None:
    """Banked partials vs an INDEPENDENT reduction of ``n`` random banked cells:
    Reducer.mean on each zone's own tile mosaic at the full 10 m in the native
    UTM grid.  Gate: max|d| < 2e-4 and | |m| - |m10| | < 1e-3 (reading a
    renormalized pyramid level shows up as |m| inflated by ~4e-3)."""
    g = _grid()
    m, _, seen, _ = _year_means(g, year, scale)
    pool = np.flatnonzero(seen == 1)
    if pool.size == 0:
        sys.exit(f"nothing banked for {year} yet")
    idx = np.random.default_rng(seed).choice(pool, size=min(n, pool.size), replace=False)
    ref = np.full((idx.size, 64), np.nan)
    pos = {int(i): k for k, i in enumerate(idx)}
    for zone, crs in ZONES.items():
        mine = idx[g["zone"].to_numpy()[idx] == zone]
        mos = (ee.ImageCollection(COLL).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
               .filter(ee.Filter.eq("UTM_ZONE", zone)).mosaic())
        for lo in range(0, mine.size, 50):
            sub = mine[lo:lo + 50]
            r = _retry(lambda: mos.reduceRegions(collection=_fc(ee, g, sub),
                                                 reducer=ee.Reducer.mean(), crs=crs, scale=10)
                       .map(lambda f: f.setGeometry(None)).getInfo(), "check")
            for f in r["features"]:
                p = f["properties"]
                if p.get("A00") is not None:
                    ref[pos[int(p["i"])]] = [p[b] for b in BANDS]
    got = m[idx]
    d = np.abs(got - ref)
    dn = np.linalg.norm(got, axis=1) - np.linalg.norm(ref, axis=1)
    cos = np.einsum("ck,ck->c", got, ref) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
    ok = bool(np.nanmax(d) < 2e-4 and np.nanmax(np.abs(dn)) < 1e-3 and not np.isnan(ref).any())
    print(f"check {year}, {idx.size} cells vs the zone-mosaic 10 m mean: max|d| {np.nanmax(d):.2e}, "
          f"min cos {np.nanmin(cos):.8f}, d|m| mean {np.nanmean(dn):+.1e} max "
          f"{np.nanmax(np.abs(dn)):.1e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--project", default=local_value("gee_project"),
                    help="EE-registered cloud project id (default: gee_project in "
                         "dataprep/local_paths.toml)")
    ap.add_argument("--years", nargs="+", type=int, default=YEARS, choices=YEARS)
    ap.add_argument("--chunk", type=int, default=150, help="max cells per request")
    ap.add_argument("--scale", type=float, default=15.0, choices=SCALES,
                    help="reduction scale in native UTM (m): 15 = verified lattice of the "
                         "full-resolution level (default), 10 = every pixel at ~1.8x the "
                         "EECU; 16 aliases and >= 19 reads renormalized pyramid levels")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-units", type=int, default=None,
                    help="only the first N units per year (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="list what would run, no EE")
    ap.add_argument("--status", action="store_true", help="what is banked so far")
    ap.add_argument("--assemble", action="store_true", help="partials -> the store")
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="verify N random banked cells against a 10 m zone-mosaic reduction")
    ap.add_argument("--check-year", type=int, default=2021, choices=YEARS)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):               # EE errors can carry non-ASCII
        sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
    if args.status:
        return status()
    if args.assemble:
        return assemble(args.scale)
    if args.dry_run:
        todo, nunits = _todo(_grid(), args.years, args.chunk, args.max_units)
        return print(f"{len(todo)} of {nunits * len(args.years)} units to reduce "
                     f"@ {args.scale:g} m")
    ee = _ee(args.project)
    if args.check:
        return check(ee, args.check, args.check_year, args.scale)
    run(ee, args.years, args.chunk, args.scale, args.workers, args.max_units)


if __name__ == "__main__":
    main()
