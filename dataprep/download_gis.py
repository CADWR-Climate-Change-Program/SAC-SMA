"""Stage the raw California/CONUS GIS rasters that ``data/raw_gis/sample_gis.py`` samples.

Fetches the four products behind the continuous soil/veg/terrain HRU features
(the dPL ``physical`` feature variant) into exactly the tree the sampler reads.
Provenance, units, and encodings: ``data/raw_gis/SOURCES.md``.

    polaris/PROPERTIES/v1.0/<prop>/mean/<depth>/lat{S}{N}_lon{W}{E}.tif
    landfire/{EVC,EVH}/{EVC,EVH}_lat{S}{N}_lon{W}{E}.tif
    dem/3dep_1as/USGS_1_n{N}w{W}.tif
    lai/mcd15a2h/<tile>/MCD15A2H.A{yyyy}{ddd}.<tile>.061.*.hdf

**Extent.** 1-degree tiles over lat 32-42 N, lon -125 to -114 W = 110 tiles.
POLARIS and 3DEP are CONUS-land products, so the ocean/out-of-CONUS tiles of
that grid simply 404 -- they are counted as ``absent`` and are not an error.
The 2026-07 stage that the committed per-HRU CSVs were sampled from is 95 land
tiles for POLARIS and 89 for 3DEP; LANDFIRE serves all 110 (an ImageServer
renders any bbox).

**Tile geometry** (verified by reading the 2026-07 stage, not assumed):
POLARIS, LANDFIRE   3600x3600, EPSG:4326, exact 1-degree bounds, 1/3600 deg pixel
POLARIS             float32, nodata -9999
LANDFIRE            int16, no nodata (water class 11 is a valid code)
3DEP                3604x3604, EPSG:4269, 1 degree + a 2-px overlap per side,
                    float32, nodata -999999 -- served as published, not re-cut
LANDFIRE is requested at ``size=3600,3600`` over the tile bbox precisely so it
lands on the POLARIS 1/3600-degree grid; do not "simplify" that to a default.

**Auth.** POLARIS (plain HTTP), LANDFIRE, and 3DEP need none.  MODIS LAI goes
through NASA Earthdata: put a ``urs.earthdata.nasa.gov`` entry in your
``~/.netrc`` (``_netrc`` on Windows).  ``requests`` re-applies netrc auth per
host across the LP DAAC redirect chain, so no custom auth handler is needed.
The classic e4ftl01 HTTPS archive is dead; granules are discovered through the
CMR granule API and pulled from the protected cloud bucket.

Only ``requests`` is needed -- this runs in the plain ``sacsma`` env.  The
sampler itself needs the ``sacsma-gis`` env (rasterio + pyhdf + pyproj).

Usage
-----
    python dataprep/download_gis.py --status                    # inventory, no network
    python dataprep/download_gis.py --layers polaris            # one product
    python dataprep/download_gis.py --all --jobs 8              # everything, resumable
    python dataprep/download_gis.py --all --dry-run             # what would be fetched

Files are written via a ``.part`` temporary and renamed only on a complete
body, so an interrupted run -- a dropped VPN, a sleeping laptop -- never leaves
a truncated tile that a later resume would mistake for finished.  Re-running
skips what is already on disk.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import threading
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]

#: Staging root.  ``SACSMA_RAW_GIS`` overrides it -- the 2026-07 stage lives on
#: ``D:\sacsma-data\raw_gis``, off the repo drive, because the tree is ~89 GB.
#: Mirrors the SACSMA_ET_DIR / SACSMA_SWE_DIR precedent in dataprep/README.md.
DEFAULT_ROOT = Path(os.environ.get("SACSMA_RAW_GIS") or REPO / "data" / "raw_gis")

# --- California extent: 1-degree tiles ---------------------------------------
LAT_S = range(32, 42)      # south edge of each tile row
LON_W = range(-125, -114)  # west edge of each tile column

#: the 2026-07 stage the committed soilveg/lai CSVs were sampled from, for --status
REFERENCE = {"polaris": 2280, "landfire": 220, "terrain": 89, "lai": 3672}

LAYERS = ("polaris", "landfire", "terrain", "lai")
#: CONUS-land products -- ocean tiles of the request grid 404 rather than exist
LAND_ONLY = ("polaris", "terrain")


def tiles() -> list[tuple[int, int]]:
    return [(s, w) for s in LAT_S for w in LON_W]


def tile_stem(s: int, w: int) -> str:
    """POLARIS/LANDFIRE tile stem: ``lat{S}{N}_lon{W}{E}`` (e.g. lat3839_lon-122-121)."""
    return f"lat{s}{s + 1}_lon{w}{w + 1}"


# --- POLARIS (Chaney et al. 2019, WRR, doi:10.1029/2018WR022797) -------------
POLARIS_BASE = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"
#: the minimal set the sampler consumes; stage more without re-fetching these
PROPS = ("sand", "clay", "ksat", "theta_s")
DEPTHS = ("0_5", "5_15", "15_30", "30_60", "60_100", "100_200")


def plan_polaris(root: Path):
    out = root / "polaris" / "PROPERTIES" / "v1.0"
    for prop in PROPS:
        for depth in DEPTHS:
            for s, w in tiles():
                stem = tile_stem(s, w)
                yield (
                    f"{POLARIS_BASE}/{prop}/mean/{depth}/{stem}.tif",
                    out / prop / "mean" / depth / f"{stem}.tif",
                    None,
                )


# --- LANDFIRE 2024 EVC/EVH (USGS ArcGIS ImageServer, bbox-clipped) -----------
LANDFIRE_BASE = (
    "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024"
    "/LF2024_{layer}_CONUS/ImageServer/exportImage"
)
LF_PX = 3600  # 1 deg / 3600 -> the POLARIS 1/3600-deg grid


def plan_landfire(root: Path):
    for layer in ("EVC", "EVH"):
        for s, w in tiles():
            stem = tile_stem(s, w)
            yield (
                LANDFIRE_BASE.format(layer=layer),
                root / "landfire" / layer / f"{layer}_{stem}.tif",
                {
                    "bbox": f"{w},{s},{w + 1},{s + 1}",
                    "bboxSR": "4326",
                    "imageSR": "4326",
                    "size": f"{LF_PX},{LF_PX}",
                    "format": "tiff",
                    "pixelType": "S16",
                    "interpolation": "RSP_NearestNeighbor",
                    "f": "image",
                },
            )


# --- 3DEP 1 arc-second (public AWS prd-tnm, no auth) -------------------------
TNM_BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current"


def plan_terrain(root: Path):
    for s, w in tiles():
        # tiles are named by NORTH-edge lat and WEST-corner lon magnitude
        cell = f"n{s + 1}w{abs(w)}"
        yield (
            f"{TNM_BASE}/{cell}/USGS_1_{cell}.tif",
            root / "dem" / "3dep_1as" / f"USGS_1_{cell}.tif",
            None,
        )


# --- MODIS MCD15A2H.061 LAI (CMR discovery -> LP DAAC cloud, Earthdata auth) --
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
LAI_TILES = ("h08v04", "h08v05", "h09v04", "h09v05")
#: CMR returns granules OVERLAPPING the window, so the 8-day composite starting
#: 2002-12-27 (A2002361) comes with it -- that is why the stage holds 918/tile,
#: not 920: 920 in-window - 3 absent upstream (A2016049, A2022097, A2022289) + 1.
LAI_TEMPORAL = "2003-01-01T00:00:00Z,2022-12-31T23:59:59Z"


def _get_retrying(session, url, params, headers, retries=4, timeout=120):
    """GET with backoff.  A 20-year listing is several paged queries and CMR
    drops connections often enough that one hiccup must not lose the layer."""
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception:  # noqa: BLE001 - retry anything transient
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def cmr_granules(session: requests.Session, tile: str, temporal: str):
    """Every MCD15A2H.061 granule for ``tile`` in ``temporal`` -> [(name, url)]."""
    params = {
        "short_name": "MCD15A2H",
        "version": "061",
        "temporal": temporal,
        "readable_granule_name[]": f"*.{tile}.*",
        "options[readable_granule_name][pattern]": "true",
        "page_size": "2000",
        "sort_key": "start_date",
    }
    headers: dict[str, str] = {}
    found = []
    while True:
        r = _get_retrying(session, CMR_URL, params, headers)
        entries = r.json().get("feed", {}).get("entry", [])
        if not entries:
            break
        for e in entries:
            href = next(
                (
                    ln["href"]
                    for ln in e.get("links", [])
                    if ln.get("href", "").endswith(".hdf")
                    and ln.get("rel", "").endswith("/data#")
                ),
                None,
            )
            if href:
                # name from the URL, NOT producer_granule_id -- CMR reports the
                # id for this collection WITHOUT the .hdf extension, which would
                # both hide the file from the sampler's *.hdf glob and defeat
                # resume (every granule would look un-fetched on the next run).
                found.append((href.rsplit("/", 1)[-1], href))
        after = r.headers.get("CMR-Search-After")
        if not after:
            break
        headers["CMR-Search-After"] = after
    return found


def plan_lai(root: Path, session: requests.Session, temporal: str):
    for tile in LAI_TILES:
        granules = cmr_granules(session, tile, temporal)
        print(f"  CMR {tile}: {len(granules)} granules", flush=True)
        for name, url in granules:
            yield (url, root / "lai" / "mcd15a2h" / tile / name, None)


# --- fetch -------------------------------------------------------------------
_print_lock = threading.Lock()


def fetch(session, url, dest, params, retries=4, timeout=300) -> str:
    """Download ``url`` -> ``dest`` atomically.  'skip' | 'ok' | 'absent' | 'fail'."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # with_name, not with_suffix: MODIS granule names are full of dots
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            with session.get(url, params=params, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    return "absent"
                r.raise_for_status()
                # an ImageServer reports failure as a 200 JSON error body
                if "json" in r.headers.get("Content-Type", "").lower():
                    raise RuntimeError(f"service error: {r.text[:200]}")
                declared = int(r.headers.get("Content-Length") or 0)
                written = 0
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        written += len(chunk)
                if declared and written != declared:
                    raise OSError(f"short read {written}/{declared} B")
                if written == 0:
                    raise OSError("empty body")
            part.replace(dest)
            return "ok"
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            part.unlink(missing_ok=True)
            if attempt == retries - 1:
                with _print_lock:
                    print(f"  FAIL {dest.name}: {exc}", file=sys.stderr, flush=True)
                return "fail"
            time.sleep(2**attempt)
    return "fail"


def run_layer(name, items, session, jobs, dry_run) -> dict[str, int]:
    items = list(items)
    todo = [it for it in items if not (it[1].exists() and it[1].stat().st_size > 0)]
    print(f"[{name}] {len(items)} on the request grid, "
          f"{len(items) - len(todo)} on disk, {len(todo)} to try")
    if todo and name in LAND_ONLY:
        print(f"  ({name} is CONUS-land only -- ocean tiles of the grid "
              f"404 and are counted 'absent')")
    if dry_run or not todo:
        return {"skip": len(items) - len(todo), "todo": len(todo)}

    tally: dict[str, int] = {}
    done = 0
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(fetch, session, *it): it for it in todo}
        for fut in cf.as_completed(futures):
            status = fut.result()
            tally[status] = tally.get(status, 0) + 1
            done += 1
            if done % 25 == 0 or done == len(todo):
                with _print_lock:
                    print(f"  [{name}] {done}/{len(todo)}  {tally}", flush=True)
    return tally


# --- status ------------------------------------------------------------------
def status(root: Path) -> None:
    print(f"root: {root}{'' if root.exists() else '   (does not exist)'}\n")
    counts = {
        "polaris": (root / "polaris", "*.tif"),
        "landfire": (root / "landfire", "*.tif"),
        "terrain": (root / "dem", "*.tif"),
        "lai": (root / "lai", "*.hdf"),
    }
    total_n = total_b = 0
    print(f"{'layer':10s} {'files':>7s} {'GB':>7s}  {'ref':>6s}  delta")
    for layer, (d, pat) in counts.items():
        files = list(d.rglob(pat)) if d.exists() else []
        n = len(files)
        b = sum(f.stat().st_size for f in files)
        total_n, total_b = total_n + n, total_b + b
        ref = REFERENCE[layer]
        delta = "complete" if n == ref else f"{n - ref:+d} vs 2026-07 stage"
        print(f"{layer:10s} {n:7d} {b / 1e9:7.1f}  {ref:6d}  {delta}")
    print(f"{'TOTAL':10s} {total_n:7d} {total_b / 1e9:7.1f}")
    if root.exists():
        parts = list(root.rglob("*.part"))
        if parts:
            print(f"\n{len(parts)} interrupted .part file(s) -- re-run to finish")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"stage root [{DEFAULT_ROOT}]")
    ap.add_argument("--layers", default="", help=f"comma-separated: {','.join(LAYERS)}")
    ap.add_argument("--all", action="store_true", help="all four layers")
    ap.add_argument("--status", action="store_true", help="inventory the stage, no network")
    ap.add_argument("--dry-run", action="store_true", help="report what would be fetched")
    ap.add_argument("--jobs", type=int, default=6, help="concurrent downloads [6]")
    ap.add_argument("--retries", type=int, default=4, help="attempts per file [4]")
    ap.add_argument("--lai-temporal", default=LAI_TEMPORAL, help=f"CMR window [{LAI_TEMPORAL}]")
    args = ap.parse_args(argv)

    if args.status:
        status(args.root)
        return 0

    selected = LAYERS if args.all else tuple(x for x in args.layers.split(",") if x)
    if not selected:
        ap.error("choose --layers <l1,l2> or --all (or --status)")
    unknown = set(selected) - set(LAYERS)
    if unknown:
        ap.error(f"unknown layer(s) {sorted(unknown)}; valid: {', '.join(LAYERS)}")

    session = requests.Session()
    session.headers["User-Agent"] = "sacsma-dataprep/1.0"
    builders = {
        "polaris": lambda: plan_polaris(args.root),
        "landfire": lambda: plan_landfire(args.root),
        "terrain": lambda: plan_terrain(args.root),
        "lai": lambda: plan_lai(args.root, session, args.lai_temporal),
    }

    rc = 0
    for layer in selected:
        tally = run_layer(layer, builders[layer](), session, args.jobs, args.dry_run)
        if tally.get("fail"):
            rc = 1
        if tally.get("absent"):
            print(f"[{layer}] {tally['absent']} tile(s) absent upstream (ocean / out-of-CONUS)")
    if not args.dry_run:
        print()
        status(args.root)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
