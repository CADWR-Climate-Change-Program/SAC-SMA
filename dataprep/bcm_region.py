"""Build the BCM region stores: USGS 270-m monthly BCM -> 1/16-deg grid + CalSim3 catchments.

Streams the Basin Characterization Model (v8) Weather-Generator-scenario data
release from its public ScienceBase S3 bucket, aggregates each monthly 270-m
grid onto the 4410-cell region grid *and* the CalSim3 catchment polygons **in
memory**, and keeps only the aggregates.  The ~112 GB of zipped ASCII (~2.3 TB
inflated) is never written to disk; the two output stores are ~100 MB each.

The source (verified against the live bucket and the release's FGDC metadata,
not assumed):

    https://prod-is-usgs-sb-prod-publish.s3.amazonaws.com/<item>/<Scenario>_<var>_WY<a>_<b>.zip
    zip -> one deflate member per month, named e.g. ``run1915oct.asc``
    ESRI ASCII grid, ncols 3486 x nrows 4477, cellsize 270 m, NODATA -9999
    xllcorner -374495.843750  yllcorner -616153.312500
    California Teale Albers (meters), NAD83, lon_0 -120, false northing -4e6  =  EPSG:3310
    every hydrology variable is in millimetres

Each ``.asc`` is 156 MB of *fixed-width* text (10-char fields, 2 decimals), so
rows are found by arithmetic and the ~2.4 M cells that fall in the region are
gathered directly -- no whole-grid float conversion.

**Resume.** Work is banked at two levels, both written atomically (temp file +
rename) so a hard kill cannot leave a truncated file:

  - per month  -> ``tmp/bcm_parts/_months/<scenario>_<var>/<yyyymm>.npz``
  - per zip    -> ``tmp/bcm_parts/<scenario>_<var>_WY<a>_<b>.npz``, after which
    that zip's monthly checkpoints are deleted

Re-running skips whatever is already banked.

Usage
-----
    python dataprep/bcm_region.py --list                        # what is on offer
    python dataprep/bcm_region.py --scenarios s01 --vars run --zips WY1916_19
    python dataprep/bcm_region.py --all                         # everything, resumable
    python dataprep/bcm_region.py --assemble                    # partials -> stores
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --- source -------------------------------------------------------------------
SB_ITEM = "https://www.sciencebase.gov/catalog/item/{item}?format=json"

#: scenario key -> (ScienceBase item id, human label).  Both items are the
#: *hydrology* child of the Part I release (parent 68029987d4be0210cdcc98d1).
SCENARIOS: dict[str, tuple[str, str]] = {
    "s01": ("69e7d5c9b66b0164d0f72e91", "Scenario 1 Baseline"),
    "s13": ("69e91d03b66b0183fe17a443", "Scenario 13 0% ave ppt, +3 degC"),
}
#: BCM hydrology variables, all in mm
VARIABLES = ("aet", "cwd", "pck", "rch", "run", "str")
VAR_LONG = {
    "aet": "actual evapotranspiration",
    "cwd": "climatic water deficit",
    "pck": "snowpack (snow water equivalent)",
    "rch": "recharge",
    "run": "runoff",
    "str": "soil moisture storage",
}
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec")

# --- grid (from the .asc headers / FGDC metadata) -----------------------------
NCOLS, NROWS = 3486, 4477
XLL, YLL, CELL = -374495.843750, -616153.312500, 270.0
NODATA = -9999.0
BCM_CRS = "EPSG:3310"                        # California Teale Albers, NAD83

REPO = Path(__file__).resolve().parents[1]
GRID_CSV = REPO / "data" / "region" / "grid_cells.csv"
GPKG = REPO / "data" / "calsim" / "gis" / "calsim3.gpkg"
GPKG_LAYER = "CalSim3_And_GooseLake"
OUT_DIR = REPO / "data" / "region" / "bcm"
PARTS = REPO / "tmp" / "bcm_parts"           # tmp/ is gitignored, local-only
INDEX_NPZ = PARTS / "_index.npz"
RES = 1 / 16


# ------------------------------------------------------------------ small utils
def _session(pool: int = 32) -> requests.Session:
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=pool,
                                                      pool_maxsize=pool))
    return s


def _atomic_npz(path: Path, **arrays) -> None:
    """Write ``arrays`` so a kill mid-write can never leave a half-file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)


#: ScienceBase serves published item files from the ``-publish`` bucket, but a few
#: files' ``publishedS3Uri`` still names the internal ``-content`` bucket, which
#: 403s even though the object *is* in ``-publish`` (1 of these 132 zips:
#: Scenario1_aet_WY2000_09.zip).  So the recorded host is a hint, not the truth.
S3_PUBLISH = "https://prod-is-usgs-sb-prod-publish.s3.amazonaws.com"
_S3_HOST = re.compile(r"https://prod-is-usgs-sb-prod-\w+\.s3\.amazonaws\.com")


def s3_variants(url: str) -> list[str]:
    """The hosts to try for a ScienceBase file: publish bucket first."""
    pub = _S3_HOST.sub(S3_PUBLISH, url)
    return [pub] if pub == url else [pub, url]


def _get(sess: requests.Session, url: str, headers=None, tries: int = 6) -> bytes:
    urls = s3_variants(url)
    last: Exception = RuntimeError(url)
    for attempt in range(tries):
        for u in urls:
            try:
                r = sess.get(u, headers=headers, timeout=300)
                r.raise_for_status()
                return r.content
            except Exception as exc:
                last = exc
        if attempt < tries - 1:
            time.sleep(2 ** attempt)
    raise last


# ------------------------------------------------------------------- item files
def item_files(sess: requests.Session, scenario: str) -> pd.DataFrame:
    """The zip inventory of a scenario item: var, wy span, size, S3 url (cached)."""
    item, _ = SCENARIOS[scenario]
    cache = PARTS / f"_item_{scenario}.json"
    if cache.exists():
        files = json.loads(cache.read_text())
    else:
        files = json.loads(_get(sess, SB_ITEM.format(item=item)))["files"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(files))
    rows = []
    for f in files:
        stem = Path(f["name"]).stem                     # Scenario1_run_WY1916_19
        parts = stem.split("_")
        if len(parts) != 4 or parts[1] not in VARIABLES:
            continue
        rows.append(dict(scenario=scenario, var=parts[1],
                         zip=f"{parts[2]}_{parts[3]}", name=f["name"],
                         size=int(f["size"]), url=f["publishedS3Uri"]))
    out = pd.DataFrame(rows).sort_values(["var", "zip"]).reset_index(drop=True)
    if out.empty:
        sys.exit(f"{scenario}: no recognisable zips in item {item}")
    return out


# ------------------------------------------------------------------ zip reading
def zip_members(sess: requests.Session, url: str, tail: int = 262144) -> list[dict]:
    """Central-directory listing of a remote zip, via one range request."""
    b = _get(sess, url, headers={"Range": f"bytes=-{tail}"})
    i = b.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError(f"no end-of-central-directory in the last {tail} B of {url}")
    n_entries = struct.unpack("<H", b[i + 10:i + 12])[0]
    out, p = [], 0
    while True:
        p = b.find(b"PK\x01\x02", p)
        if p < 0:
            break
        f = struct.unpack("<IHHHHHHIIIHHHHHII", b[p:p + 46])
        nlen, elen, clen = f[10], f[11], f[12]
        out.append(dict(name=b[p + 46:p + 46 + nlen].decode("utf-8", "replace"),
                        method=f[4], comp=f[8], uncomp=f[9], offset=f[16]))
        p += 46 + nlen + elen + clen
    if len(out) != n_entries:
        raise RuntimeError(f"{url}: central directory truncated "
                           f"({len(out)}/{n_entries} entries in a {tail} B tail)")
    return out


def member_bytes(sess: requests.Session, url: str, m: dict) -> bytes:
    """Inflate one zip member straight from S3 -- nothing touches disk."""
    end = m["offset"] + 30 + 65535 + m["comp"]          # local header + name + extra
    raw = _get(sess, url, headers={"Range": f"bytes={m['offset']}-{end}"})
    sig, _, _, method, _, _, _, _, _, nlen, elen = struct.unpack("<IHHHHHIIIHH", raw[:30])
    if sig != 0x04034B50:
        raise RuntimeError(f"{url}#{m['name']}: bad local header signature {sig:#x}")
    data = raw[30 + nlen + elen: 30 + nlen + elen + m["comp"]]
    if len(data) != m["comp"]:
        raise RuntimeError(f"{url}#{m['name']}: short read "
                           f"({len(data)}/{m['comp']} B)")
    if method == 0:
        buf = data
    elif method == 8:
        buf = zlib.decompressobj(-15).decompress(data, m["uncomp"] + 1)
    else:
        raise RuntimeError(f"{url}#{m['name']}: unsupported compression {method}")
    if len(buf) != m["uncomp"]:
        raise RuntimeError(f"{url}#{m['name']}: inflated {len(buf)} B, "
                           f"expected {m['uncomp']}")
    return buf


# --------------------------------------------------------------- ASCII grid I/O
def grid_layout(buf: bytes) -> dict:
    """Header values plus the fixed-width geometry the fast reader needs.

    ESRI ASCII is only *usually* fixed-width; this asserts it for the file in
    hand (constant row stride, constant field width, decimal point in a constant
    column) so the arithmetic reader can never silently mis-read a grid.
    """
    head_end = 0
    for _ in range(6):
        head_end = buf.index(b"\n", head_end) + 1
    hdr = {}
    for line in buf[:head_end].decode("ascii").splitlines():
        k, v = line.split()
        hdr[k.lower()] = float(v)
    ncols, nrows = int(hdr["ncols"]), int(hdr["nrows"])
    body = len(buf) - head_end
    if body % nrows:
        raise RuntimeError(f"body {body} B is not a whole number of {nrows} rows "
                           "-- not fixed-width")
    stride = body // nrows
    eol = 2 if buf[head_end + stride - 2:head_end + stride] == b"\r\n" else 1
    width, rem = divmod(stride - eol, ncols)
    if rem:
        raise RuntimeError(f"row of {stride - eol} B is not {ncols} equal fields")
    row0 = np.frombuffer(buf[head_end:head_end + stride - eol], np.uint8)
    dots = np.nonzero(row0.reshape(ncols, width)[0] == ord("."))[0]
    if dots.size != 1 or not (row0.reshape(ncols, width)[:, dots[0]] == ord(".")).all():
        raise RuntimeError("decimal point is not in a constant column "
                           "-- not fixed-width")
    return dict(ncols=ncols, nrows=nrows, xll=hdr["xllcorner"], yll=hdr["yllcorner"],
                cell=hdr["cellsize"], nodata=hdr["nodata_value"],
                head=head_end, stride=stride, width=width, dot=int(dots[0]))


#: the corner is written to varying precision across variables (the ``run``
#: grids give -374495.8364, the ``ppt`` grids -374495.84375 -- 7 mm apart), so
#: the origin is checked to a tolerance, not for equality.  Anything beyond a
#: metre would be a different grid.
ORIGIN_TOL_M = 1.0


def check_layout(lay: dict) -> None:
    """The release is one single grid; refuse anything that is not it."""
    bad = {k: (lay[k], v) for k, v in
           dict(ncols=NCOLS, nrows=NROWS, cell=CELL, nodata=NODATA).items()
           if lay[k] != v}
    bad |= {k: (lay[k], v) for k, v in dict(xll=XLL, yll=YLL).items()
            if abs(lay[k] - v) > ORIGIN_TOL_M}
    if bad:
        raise RuntimeError(f"grid does not match the release geometry: {bad}")


def read_cells(buf: bytes, lay: dict, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Values at ``(rows, cols)`` of a fixed-width ASCII grid, as float32.

    Gathers only the requested fields, then decodes them arithmetically: at 2.4 M
    of 15.6 M cells this is ~6x cheaper than converting the whole grid.
    """
    w, dot = lay["width"], lay["dot"]
    ndec = w - dot - 1
    flat = np.frombuffer(buf, np.uint8, offset=lay["head"])
    grid = flat.reshape(lay["nrows"], lay["stride"])[:, :lay["ncols"] * w]
    sel = grid.reshape(lay["nrows"], lay["ncols"], w)[rows, cols]     # (n, w) uint8
    digit = np.where((sel >= 48) & (sel <= 57), sel - 48, 0).astype(np.int32)
    whole = digit[:, :dot] @ (10 ** np.arange(dot - 1, -1, -1, dtype=np.int32))
    frac = digit[:, dot + 1:] @ (10 ** np.arange(ndec - 1, -1, -1, dtype=np.int32))
    val = whole.astype(np.float32) + frac.astype(np.float32) / np.float32(10 ** ndec)
    np.negative(val, out=val, where=(sel[:, :dot] == ord("-")).any(1))
    return val


# ---------------------------------------------------------------------- indexing
def load_region_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rg = pd.read_csv(GRID_CSV)
    return (rg["key"].to_numpy(str), rg["lat"].to_numpy(float),
            rg["lon"].to_numpy(float))


def build_index(verbose: bool = True) -> dict:
    """Map every BCM cell whose centre lands in the region onto its targets.

    Returns the BCM (row, col) of those cells plus, for each, the region-cell
    index and the CalSim3 catchment index (-1 = none).  Cached to ``_index.npz``.
    """
    if INDEX_NPZ.exists():
        with np.load(INDEX_NPZ, allow_pickle=False) as z:
            idx = {k: z[k] for k in z.files}
        return idx

    import geopandas as gpd
    import shapely
    from pyproj import Transformer

    keys, rlat, rlon = load_region_grid()
    lat_ix = np.rint(rlat / RES - 0.5).astype(np.int64)
    lon_ix = np.rint(rlon / RES - 0.5).astype(np.int64)
    lut = {(a, o): n for n, (a, o) in enumerate(zip(lat_ix, lon_ix))}

    tr = Transformer.from_crs(BCM_CRS, "EPSG:4326", always_xy=True)
    xc = XLL + (np.arange(NCOLS) + 0.5) * CELL
    rows_l, cols_l, dst_l, lon_l, lat_l = [], [], [], [], []
    for i0 in range(0, NROWS, 256):                      # row blocks: bounded memory
        i1 = min(NROWS, i0 + 256)
        # ESRI ASCII row 0 is the *top* row
        yc = YLL + (NROWS - 0.5 - np.arange(i0, i1)) * CELL
        xx, yy = np.meshgrid(xc, yc)
        lon, lat = tr.transform(xx.ravel(), yy.ravel())
        a = np.rint(lat / RES - 0.5).astype(np.int64)
        o = np.rint(lon / RES - 0.5).astype(np.int64)
        dst = np.fromiter((lut.get((aa, oo), -1) for aa, oo in zip(a, o)),
                          np.int32, count=a.size)
        hit = np.nonzero(dst >= 0)[0]
        if hit.size:
            rows_l.append((i0 + hit // NCOLS).astype(np.int32))
            cols_l.append((hit % NCOLS).astype(np.int32))
            dst_l.append(dst[hit])
            lon_l.append(lon[hit])
            lat_l.append(lat[hit])
        if verbose and (i0 // 256) % 4 == 0:
            print(f"  rows {i0:5d}/{NROWS}  {sum(x.size for x in dst_l):,} cells",
                  end="\r", flush=True)
    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    dst_region = np.concatenate(dst_l)
    lon = np.concatenate(lon_l)
    lat = np.concatenate(lat_l)
    if verbose:
        print(f"  {rows.size:,} BCM cells land in the {len(keys)} region cells      ")

    # CalSim3 catchments: assign each selected cell centre to a polygon
    g = gpd.read_file(GPKG, layer=GPKG_LAYER).reset_index(drop=True)
    dst_catch = np.full(rows.size, -1, np.int32)
    pts_x, pts_y = lon, lat
    for cid, geom in enumerate(g.geometry.to_numpy()):
        x0, y0, x1, y1 = geom.bounds
        cand = np.nonzero((pts_x >= x0) & (pts_x <= x1)
                          & (pts_y >= y0) & (pts_y <= y1) & (dst_catch < 0))[0]
        if not cand.size:
            continue
        inside = shapely.contains_xy(geom, pts_x[cand], pts_y[cand])
        dst_catch[cand[inside]] = cid
        if verbose and cid % 40 == 0:
            print(f"  catchments {cid:3d}/{len(g)}  "
                  f"{int((dst_catch >= 0).sum()):,} cells assigned", end="\r", flush=True)
    # a polygon smaller than a 270-m cell can contain no cell centre (the layer has
    # two such slivers, 0.015 and 0.038 mi^2).  Give each the single nearest cell
    # rather than leaving a hole -- at that size the nearest centre is the cell the
    # polygon sits in.
    covered = np.bincount(dst_catch[dst_catch >= 0], minlength=len(g))
    for cid in np.nonzero(covered == 0)[0]:
        c = g.geometry.iloc[cid].centroid
        n = int(np.argmin((pts_x - c.x) ** 2 + (pts_y - c.y) ** 2))
        dst_catch[n] = cid
        if verbose:
            print(f"  sub-cell catchment {g['Connect_No'].iloc[cid]} "
                  f"({g['SQ_MI'].iloc[cid]:.4f} mi2) -> nearest cell")
    if verbose:
        print(f"  {int((dst_catch >= 0).sum()):,} cells in {len(g)} CalSim3 "
              f"catchments                    ")

    idx = dict(
        rows=rows, cols=cols, dst_region=dst_region, dst_catch=dst_catch,
        region_keys=keys, region_n=np.array([len(keys)]),
        catch_n=np.array([len(g)]),
        catch_node=g["Connect_No"].to_numpy(str),
        catch_name=g["CT_Name"].to_numpy(str) if "CT_Name" in g else
        np.array([""] * len(g)),
        catch_type=g["Type"].to_numpy(str),
        catch_sqmi=g["SQ_MI"].to_numpy(float),
    )
    INDEX_NPZ.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(INDEX_NPZ, **idx)
    return idx


def aggregate(val: np.ndarray, idx: dict) -> dict[str, np.ndarray]:
    """Cell values -> region-cell and catchment means, ignoring NODATA."""
    ok = val > NODATA + 1.0
    out = {}
    for tag, dst, n in (("region", idx["dst_region"], int(idx["region_n"][0])),
                        ("catch", idx["dst_catch"], int(idx["catch_n"][0]))):
        m = ok if tag == "region" else (ok & (dst >= 0))
        d = dst[m].astype(np.int64)
        cnt = np.bincount(d, minlength=n)
        tot = np.bincount(d, weights=val[m].astype(np.float64), minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{tag}_mean"] = np.where(cnt > 0, tot / np.maximum(cnt, 1),
                                          np.nan).astype(np.float32)
        out[f"{tag}_count"] = cnt.astype(np.int32)
    return out


# -------------------------------------------------------------------- the pull
def month_key(name: str, var: str) -> str:
    """``run1915oct.asc`` -> ``191510`` (the calendar month, not the water year)."""
    stem = Path(name).stem.lower()
    if not stem.startswith(var):
        raise RuntimeError(f"member {name} does not start with {var!r}")
    rest = stem[len(var):]
    year, mon = int(rest[:4]), rest[4:7]
    if mon not in MONTHS:
        raise RuntimeError(f"member {name}: unrecognised month {mon!r}")
    return f"{year:04d}{MONTHS.index(mon) + 1:02d}"


def part_path(scenario: str, var: str, zipname: str) -> Path:
    return PARTS / f"{scenario}_{var}_{zipname}.npz"


def run_zip(sess: requests.Session, row: pd.Series, idx: dict, workers: int,
            layout: dict | None) -> dict | None:
    """Aggregate every month in one remote zip; bank per month, then per zip."""
    p = part_path(row.scenario, row["var"], row["zip"])
    if p.exists():
        return layout
    t0 = time.time()
    ckpt = PARTS / "_months" / f"{row.scenario}_{row['var']}_{row['zip']}"
    members = [m for m in zip_members(sess, row.url) if m["name"].endswith(".asc")]
    members.sort(key=lambda m: month_key(m["name"], row["var"]))
    rows_i, cols_i = idx["rows"], idx["cols"]
    lay_box = [layout]

    def one(m):
        mk = month_key(m["name"], row["var"])
        cp = ckpt / f"{mk}.npz"
        if cp.exists():
            try:
                with np.load(cp, allow_pickle=False) as z:
                    return mk, {k: z[k] for k in z.files}
            except Exception:
                cp.unlink(missing_ok=True)          # truncated by a hard kill
        buf = member_bytes(sess, row.url, m)
        lay = lay_box[0]
        if lay is None:                             # first grid of the run
            lay = grid_layout(buf)
            check_layout(lay)
            lay_box[0] = lay
        val = read_cells(buf, lay, rows_i, cols_i)
        del buf
        agg = aggregate(val, idx)
        _atomic_npz(cp, **agg)
        return mk, agg

    if lay_box[0] is None:                          # establish the layout serially
        mk, agg = one(members[0])
        got = {mk: agg}
        rest = members[1:]
    else:
        got, rest = {}, members
    with cf.ThreadPoolExecutor(workers) as ex:
        for mk, agg in ex.map(one, rest):
            got[mk] = agg

    months = sorted(got)
    banked = {"months": np.array(months)}
    for field in ("region_mean", "catch_mean"):
        banked[field] = np.stack([got[m][field] for m in months])
    for field in ("region_count", "catch_count"):
        banked[field] = got[months[0]][field]       # static; checked below
        for m in months:
            if not np.array_equal(got[m][field], banked[field]):
                raise RuntimeError(f"{row['name']}: {field} varies between months "
                                   f"(first vs {m}) -- the NODATA mask is not static")
    _atomic_npz(p, **banked)
    shutil.rmtree(ckpt, ignore_errors=True)
    mb = row["size"] / 1e6
    print(f"  {row.scenario} {row['var']:<3s} {row['zip']:<10s} {len(months):4d} mo  "
          f"{mb:7.0f} MB  {time.time() - t0:6.1f}s "
          f"({mb / max(time.time() - t0, 1e-9):5.1f} MB/s)  -> {p.name}", flush=True)
    return lay_box[0]


# ------------------------------------------------------------------- assembling
def assemble(scenarios: list[str], variables: list[str]) -> None:
    """Stitch the per-zip partials into the region-grid and catchment stores."""
    import xarray as xr

    idx = build_index(verbose=False)
    keys = idx["region_keys"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for scen in scenarios:
        have = {}
        for var in variables:
            parts = sorted(PARTS.glob(f"{scen}_{var}_WY*.npz"))
            if not parts:
                continue
            frames = []
            for p in parts:
                with np.load(p, allow_pickle=False) as z:
                    frames.append((z["months"], z["region_mean"], z["catch_mean"],
                                   z["region_count"], z["catch_count"]))
            months = np.concatenate([f[0] for f in frames])
            order = np.argsort(months)
            if len(set(months.tolist())) != months.size:
                sys.exit(f"{scen} {var}: duplicate months across partials")
            have[var] = dict(
                months=months[order],
                region=np.concatenate([f[1] for f in frames])[order],
                catch=np.concatenate([f[2] for f in frames])[order],
                region_count=frames[0][3], catch_count=frames[0][4])
        if not have:
            print(f"{scen}: nothing banked yet")
            continue
        months = sorted(set.intersection(*(set(v["months"].tolist())
                                           for v in have.values())))
        dropped = {v: len(have[v]["months"]) - len(months) for v in have}
        if any(dropped.values()):
            print(f"{scen}: trimming to the {len(months)} month(s) every variable "
                  f"has (dropped per var: {dropped})")
        time_ax = np.array([np.datetime64(f"{m[:4]}-{m[4:]}-01") for m in months])
        pick = {v: np.searchsorted(have[v]["months"], months) for v in have}

        ds = xr.Dataset(
            {v: (("key", "time"), have[v]["region"][pick[v]].T.astype(np.float32))
             for v in have},
            coords={"key": keys, "time": time_ax},
            attrs=dict(
                title=f"BCM v8 monthly hydrology, {SCENARIOS[scen][1]}, region store",
                source=("USGS ScienceBase item "
                        f"{SCENARIOS[scen][0]} (California 270-m Basin "
                        "Characterization Model using Weather Generator Scenarios, "
                        "Part I); 270-m monthly ASCII grids (EPSG:3310) averaged "
                        "over each 1/16-deg region cell"),
                units="mm",
                resolution="1/16 deg (aggregated from 270 m)",
            ),
        )
        for v in have:
            ds[v].attrs.update(long_name=VAR_LONG[v], units="mm")
        ds["n_bcm_cells"] = ("key", have[next(iter(have))]["region_count"])
        ds["n_bcm_cells"].attrs["long_name"] = "270-m BCM cells averaged per region cell"
        nc = OUT_DIR / f"bcm_{scen}_monthly.nc"
        ds.to_netcdf(nc, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})

        # the catchment identity is static, so it lives in its own 386-row table
        # rather than being repeated once per month in a 477k-row series
        ncat = int(idx["catch_n"][0])
        look = pd.DataFrame({
            "cid": np.arange(ncat),
            "node": idx["catch_node"],
            "ct_name": idx["catch_name"],
            "type": idx["catch_type"],
            "sq_mi": idx["catch_sqmi"],
            "n_bcm_cells": have[next(iter(have))]["catch_count"],
        })
        look.to_csv(OUT_DIR / "bcm_catchments.csv", index=False, float_format="%.6g")
        cat = pd.DataFrame({"cid": np.repeat(np.arange(ncat), len(months)),
                            "month": np.tile(months, ncat)})
        for v in have:
            cat[v] = have[v]["catch"][pick[v]].T.reshape(-1)
        csv = OUT_DIR / f"bcm_{scen}_catchments_monthly.csv"
        cat.to_csv(csv, index=False, float_format="%.3f")
        print(f"wrote {nc.name} ({nc.stat().st_size / 1e6:.0f} MB)  "
              f"vars={[v for v in ds.data_vars if v != 'n_bcm_cells']}  "
              f"{months[0]}..{months[-1]}\n"
              f"      {csv.name} ({csv.stat().st_size / 1e6:.0f} MB)  "
              f"{ncat} catchments x {len(months)} months")


# -------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=None, choices=list(SCENARIOS))
    ap.add_argument("--vars", nargs="*", default=None, choices=list(VARIABLES))
    ap.add_argument("--zips", nargs="*", default=None,
                    help="e.g. WY1916_19 (default: every span)")
    ap.add_argument("--all", action="store_true", help="every scenario and variable")
    ap.add_argument("--list", action="store_true", help="inventory, then exit")
    ap.add_argument("--index", action="store_true", help="build the index, then exit")
    ap.add_argument("--assemble", action="store_true", help="partials -> stores")
    ap.add_argument("--status", action="store_true", help="what is banked so far")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel month fetch+decode (each holds a 156 MB grid)")
    a = ap.parse_args()

    scenarios = a.scenarios or list(SCENARIOS)
    variables = a.vars or list(VARIABLES)

    if a.assemble:
        assemble(scenarios, variables)
        return
    if a.index:
        build_index()
        return

    sess = _session(max(32, a.workers * 4))
    inv = pd.concat([item_files(sess, s) for s in scenarios], ignore_index=True)
    inv = inv[inv["var"].isin(variables)]
    if a.zips:
        inv = inv[inv["zip"].isin(a.zips)]
    if inv.empty:
        sys.exit("nothing matches that selection")
    if a.list:
        print(inv.to_string(index=False,
                            columns=["scenario", "var", "zip", "size", "name"]))
        print(f"\n{len(inv)} zips, {inv['size'].sum() / 1e9:.1f} GB")
        return
    if a.status:
        inv["done"] = [part_path(r.scenario, r["var"], r["zip"]).exists()
                       for _, r in inv.iterrows()]
        grid = (inv.groupby(["scenario", "var"])
                .agg(zips=("done", "size"), done=("done", "sum"),
                     GB=("size", lambda s: s.sum() / 1e9),
                     GB_done=("size", lambda s: s[inv.loc[s.index, "done"]].sum() / 1e9)))
        print(grid.to_string(float_format=lambda x: f"{x:6.1f}"))
        gb, gbd = inv["size"].sum() / 1e9, inv.loc[inv["done"], "size"].sum() / 1e9
        part = sum(f.stat().st_size for f in PARTS.glob("_months/*/*.npz"))
        print(f"\n{int(inv['done'].sum())}/{len(inv)} zips  {gbd:.1f}/{gb:.1f} GB "
              f"({100 * gbd / gb:.1f}%)   in-flight month checkpoints: "
              f"{part / 1e6:.0f} MB")
        return
    if not (a.all or a.scenarios or a.vars or a.zips):
        ap.error("give --scenarios/--vars/--zips, or --all, or --assemble")

    print("region grid + CalSim3 catchments -> BCM 270-m index ...", flush=True)
    idx = build_index()
    todo = [r for _, r in inv.iterrows()
            if not part_path(r.scenario, r["var"], r["zip"]).exists()]
    done = len(inv) - len(todo)
    print(f"{len(todo)} zips to pull, {sum(r['size'] for r in todo) / 1e9:.1f} GB "
          f"[{done} already banked]", flush=True)

    # A zip that cannot be read must not block the other 111 GB -- and must not
    # deadlock the restart wrapper by being first in the list every time.  Each
    # pass does everything it can; only what is still missing is retried.
    layout, failed = None, []
    for attempt in range(1, 4):
        pending = [r for r in todo
                   if not part_path(r.scenario, r["var"], r["zip"]).exists()]
        if not pending:
            break
        if attempt > 1:
            print(f"retry pass {attempt}: {len(pending)} zip(s) left", flush=True)
        failed = []
        for r in pending:
            try:
                layout = run_zip(sess, r, idx, a.workers, layout)
            except Exception as exc:
                failed.append({"name": r["name"], "error": repr(exc)})
                print(f"  FAILED {r['name']}: {exc!r}", flush=True)
    if failed:
        (PARTS / "_failed.json").write_text(json.dumps(failed, indent=1))
        print(f"\n{len(failed)} zip(s) unread after 3 passes -- see "
              f"{PARTS / '_failed.json'}.  Everything else is banked; rerun to "
              f"retry just these.", flush=True)
    else:
        (PARTS / "_failed.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
