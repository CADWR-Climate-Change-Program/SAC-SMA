"""Reitz, Sanford & Saxe (2023) historical ET: cut the CONUS rasters to California
and ingest them onto the region grid.

Source: *Historical Evapotranspiration for the Conterminous U.S.*, USGS data
release doi:10.5066/P9EZ3VAS (ScienceBase item 64135576d34eb496d1ce3d2e); paper
WRR 59, e2022WR034012.  30-arcsec (~800 m) NAD83 float32 GeoTIFFs, Oct 1895 ..
Sep 2018, shipped as decadal zips of whole-CONUS rasters (87 MB each raw):

* ``AET_<y0>_<y1>.zip``          annual WATER-YEAR totals, ``AET_<WY>.tif``, m/yr
* ``AET_<y0>_<y1>_monthly.zip``  monthly rates, mm/day, ``AET_<YYYY>_<MM>.tif`` in
  CALENDAR months (``AET_1990_10`` carries the band description ``wy1991``);
  ~4.7 GB per decade, ~58 GB.  These zips are **Deflate64**, which Python's
  ``zipfile`` cannot inflate: their members are read through 7-Zip
  (``SACSMA_7Z``, else ``7z`` on PATH, else the default install path), each
  checked against the zip's CRC-32
* side layers: long-term mean, BMA variance, per-equation weight maps (nested
  zips), irrigation fractions, GRACE storage change.

California is ~7 % of the CONUS frame, so the cut is what makes the monthly
record keepable: each source tif is windowed to ``BBOX``, rewritten as a
DEFLATE/predictor GeoTIFF, **verified bit-exact against the source window**, and
only then may the source zip be deleted (``--delete-source``).

Staging root is local-only (``tmp/reitz2023_et``; ``reitz_et`` in
dataprep/local_paths.toml or ``SACSMA_REITZ_ET`` overrides it).  The ``*_monthly.zip`` and
``Irrigation_*_1980-2018.zip`` files are ScienceBase *cloud* files (login or a
captcha/e-mail request) -- drop them into the staging root by hand; everything
else is fetched by ``tmp/reitz2023_et/get_reitz2023.py``.

What the numbers are (read before using the seasonal shape): the ANNUAL maps are
a 5-equation weighted ensemble whose weights were trained on annual
``P - Q + dS_gw`` at 1,858 Gages-II watersheds; the MONTHLY maps are the
month-to-month pattern of ONE equation (Fu-Zhang with Hamon PET) rescaled to sum
to the water-year annual map -- no observed seasonal signal, no storage
carry-over.  Irrigation supply enters only 1980-2018.

RUN ORDER (``sacsma-gis`` env, ``environment-gis.yml`` -- needs rasterio):
  1. ``python dataprep/reitz_et.py --status``
  2. ``python dataprep/reitz_et.py --cut [--delete-source]``
     -> ``<stage>/ca/<zip stem>/<name>.tif`` + ``<stage>/ca/manifest.csv``
  3. ``python dataprep/reitz_et.py --ingest``
     -> ``data/region/et_obs/reitz2023_cell_{annual,monthly}.npz``
"""

from __future__ import annotations

import argparse
import calendar
import io
import math
import os
import re
import shutil
import subprocess
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window

from _paths import local_path

REPO = Path(__file__).resolve().parents[1]
GRID_CSV = REPO / "data" / "region" / "grid_cells.csv"
DEFAULT_STAGE = local_path("reitz_et", REPO / "tmp" / "reitz2023_et")

#: (west, south, east, north), degrees.  The whole state plus the out-of-state
#: headwaters the region grid reaches (Goose Lake to 42.44 N, the east-slope
#: Nevada strip) -- deliberately wider than the 4410 cells so a new basin never
#: needs a re-pull.  Snapped OUTWARD to source pixel edges at cut time.
BBOX = (-124.75, 32.25, -114.0, 42.75)

#: 1/16-deg region cell half-width.
HALF = 1.0 / 32.0

ANNUAL_RE = re.compile(r"^AET_(\d{4})\.tif$", re.I)
#: Verified on the 1990-2018 monthly zips (2026-09-23): calendar year + month.
MONTHLY_RE = re.compile(r"^AET_(\d{4})_(0[1-9]|1[0-2])\.tif$", re.I)

#: Compression methods ``zipfile`` inflates itself; anything else (the monthly
#: zips' Deflate64, method 9) goes through 7-Zip.
_PY_METHODS = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}


# --------------------------------------------------------------------- cut ----
def _sevenzip() -> str:
    for exe in (os.environ.get("SACSMA_7Z"), shutil.which("7z"),
                r"C:\Program Files\7-Zip\7z.exe"):
        if exe and Path(exe).exists():
            return exe
    raise SystemExit("this zip is Deflate64, which Python's zipfile cannot inflate: "
                     "install 7-Zip or point SACSMA_7Z at its 7z executable")


def _read_7z(zpath: Path, info: zipfile.ZipInfo) -> bytes:
    """One member through ``7z e -so``, accepted only if its size and CRC-32
    match the zip's central directory."""
    r = subprocess.run([_sevenzip(), "e", "-so", "-bd", str(zpath), info.filename],
                       capture_output=True)
    data = r.stdout
    if r.returncode or len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise IOError(f"7-Zip read of {info.filename} in {zpath.name} failed "
                      f"(exit {r.returncode}, {len(data)} of {info.file_size} bytes): "
                      + r.stderr.decode(errors="replace").strip()[-300:])
    return data


def _iter_tifs(zpath: Path):
    """Yield (member_id, out_subdir, name, loader) for every tif in a zip,
    descending into nested zips (Weight_maps.zip).  ``loader()`` decompresses the
    member on demand, so a resumed run skips finished rasters without reading them."""
    def walk(zf: zipfile.ZipFile, prefix: str, subdir: Path):
        for info in zf.infolist():
            low = info.filename.lower()
            if info.compress_type not in _PY_METHODS and prefix:
                raise NotImplementedError(f"{prefix}{info.filename}: a nested member "
                                          "7-Zip cannot address")
            if low.endswith(".tif"):
                load = ((lambda zf=zf, info=info: zf.read(info))
                        if info.compress_type in _PY_METHODS
                        else (lambda info=info: _read_7z(zpath, info)))
                yield prefix + info.filename, subdir, Path(info.filename).name, load
            elif low.endswith(".zip"):
                inner = zipfile.ZipFile(io.BytesIO(zf.read(info)))
                yield from walk(inner, prefix + info.filename + "!",
                                subdir / Path(info.filename).stem)
    with zipfile.ZipFile(zpath) as zf:
        yield from walk(zf, "", Path(zpath.stem))


def _window(src) -> Window:
    """BBOX -> pixel window, snapped outward, clipped to the raster."""
    t = src.transform
    w, s, e, n = BBOX
    c0 = max(0, math.floor((w - t.c) / t.a + 1e-9))
    c1 = min(src.width, math.ceil((e - t.c) / t.a - 1e-9))
    r0 = max(0, math.floor((n - t.f) / t.e + 1e-9))          # t.e < 0
    r1 = min(src.height, math.ceil((s - t.f) / t.e - 1e-9))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("raster does not intersect BBOX")
    return Window(c0, r0, c1 - c0, r1 - r0)


def _cut_one(data: bytes, out: Path) -> dict | None:
    """Window one tif to BBOX, write compressed, verify bit-exact.  Returns None
    for a raster that cannot be cut (no georeferencing -- the release ships one
    figure image as a .tif -- or no overlap with BBOX); its source is then kept."""
    with MemoryFile(data) as mf, mf.open() as src:
        if src.crs is None:
            return None
        try:
            win = _window(src)
        except ValueError:
            return None
        arr = src.read(window=win)
        prof = src.profile.copy()
        tags = src.tags()
        # band descriptions carry the monthly naming proof (AET_1990_10 -> wy1991)
        descs = src.descriptions
        btags = [src.tags(b) for b in src.indexes]
        prof.update(driver="GTiff", width=int(win.width), height=int(win.height),
                    transform=src.window_transform(win), compress="deflate",
                    zlevel=9, predictor=3 if arr.dtype.kind == "f" else 2)
        tiled = min(win.width, win.height) >= 256
        prof.update(tiled=tiled)
        for k in ("blockxsize", "blockysize"):
            prof.pop(k, None)
        if tiled:
            prof.update(blockxsize=256, blockysize=256)
        nodata = src.nodata
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with rasterio.open(part, "w", **prof) as dst:
        dst.write(arr)
        dst.update_tags(**tags)
        for b, (d, t) in enumerate(zip(descs, btags), start=1):
            if d:
                dst.set_band_description(b, d)
            if t:
                dst.update_tags(b, **t)
    with rasterio.open(part) as chk:                          # the gate
        back = chk.read()
        same = (np.array_equal(back, arr, equal_nan=True)
                and chk.transform.almost_equals(prof["transform"]) and chk.crs == prof["crs"]
                and chk.descriptions == descs)
    if not same:
        part.unlink()
        raise IOError(f"cut of {out.name} does not reproduce the source window")
    part.replace(out)
    valid = np.isfinite(arr) if nodata is None else (arr != nodata) & np.isfinite(arr)
    return dict(rows=int(win.height), cols=int(win.width), src_bytes=len(data),
                out_bytes=out.stat().st_size, valid_frac=round(float(valid.mean()), 4),
                mean=float(arr[valid].mean()) if valid.any() else float("nan"))


def _load_manifest(path: Path) -> pd.DataFrame:
    cols = ["source", "member", "out", "rows", "cols", "src_bytes", "out_bytes",
            "valid_frac", "mean"]
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)


def cut(stage: Path, delete_source: bool, only: list[str] | None) -> None:
    ca = stage / "ca"
    mpath = ca / "manifest.csv"
    man = _load_manifest(mpath)
    rec = {(r["source"], r["member"]): r for r in man.to_dict("records")}

    def bank() -> None:
        if not rec:                                           # an empty csv has no header
            return
        ca.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(list(rec.values())).to_csv(mpath, index=False)

    sources = sorted(p for p in stage.iterdir() if p.suffix.lower() in (".zip", ".tif"))
    if only:
        sources = [p for p in sources if any(o in p.name for o in only)]
    freed = 0
    for src in sources:
        n = new = 0
        if src.suffix.lower() == ".tif":
            items = [(src.name, Path("_loose"), src.name, src.read_bytes)]
        else:
            items = _iter_tifs(src)
        for member, sub, name, load in items:
            n += 1
            out = ca / sub / name
            if (src.name, member) in rec and out.exists():
                continue
            info = _cut_one(load(), out)
            if info is None:
                print(f"skip  {member} in {src.name}: not georeferenced / outside "
                      "BBOX (source kept)")
                continue
            rec[(src.name, member)] = dict(source=src.name, member=member,
                                           out=out.relative_to(stage).as_posix(), **info)
            new += 1
            if new % 25 == 0:                                 # bank progress
                bank()
        bank()
        mine = [r for r in rec.values() if r["source"] == src.name]
        if not mine:
            if n == 0:
                print(f"skip  {src.name}: no rasters inside (left in place)")
            continue
        ratio = sum(r["src_bytes"] for r in mine) / max(1, sum(r["out_bytes"] for r in mine))
        msg = f"cut   {src.name}: {n} rasters ({new} new), {ratio:.0f}x smaller than raw"
        if delete_source and len(mine) == n and all((stage / r["out"]).exists() for r in mine):
            freed += src.stat().st_size
            src.unlink()
            msg += " -- source deleted"
        print(msg, flush=True)
    total = sum(f.stat().st_size for f in ca.rglob("*.tif")) if ca.exists() else 0
    print(f"CA store: {total / 1e9:.2f} GB in {ca}"
          + (f"; freed {freed / 1e9:.2f} GB of source zips" if freed else ""))


def status(stage: Path) -> None:
    man = _load_manifest(stage / "ca" / "manifest.csv")
    print(f"stage: {stage}   BBOX (W,S,E,N): {BBOX}")
    for p in sorted(stage.iterdir()):
        if p.suffix.lower() not in (".zip", ".tif"):
            continue
        k = int((man["source"] == p.name).sum())
        print(f"  {p.name:42s} {p.stat().st_size / 1e6:9.1f} MB   {k:4d} rasters cut")
    gone = sorted(set(man["source"]) - {p.name for p in stage.iterdir()})
    for s in gone:
        m = man[man["source"] == s]
        print(f"  {s:42s}   (deleted)      {len(m):4d} rasters cut, "
              f"{m['out_bytes'].sum() / 1e6:.1f} MB kept")
    if len(man):
        print(f"CA store: {man['out_bytes'].sum() / 1e9:.2f} GB kept for "
              f"{man['src_bytes'].sum() / 1e9:.1f} GB of raw raster")


# ------------------------------------------------------------------ ingest ----
def _overlap(lo: np.ndarray, hi: np.ndarray, origin: float, step: float, n: int):
    """1-D exact interval overlap of [lo, hi] with a pixel axis that starts at
    ``origin`` and advances by ``step`` (>0).  Returns (first index, (cells, K)
    weights); out-of-raster pixels get weight 0."""
    i0 = np.floor((lo - origin) / step + 1e-9).astype(int)
    k = int(np.ceil((hi - lo).max() / step)) + 1
    idx = i0[:, None] + np.arange(k)[None, :]
    a = origin + idx * step
    w = np.clip(np.minimum(a + step, hi[:, None]) - np.maximum(a, lo[:, None]), 0, None)
    w[(idx < 0) | (idx >= n)] = 0.0
    return np.clip(idx, 0, n - 1), w


class CellMean:
    """Area-weighted mean of a raster over each 1/16-deg cell rectangle: exact
    fractional pixel overlap (a cell spans 7.5 x 7.5 source pixels, so edges
    never align), cos(lat) per row, nodata excluded and renormalised."""

    def __init__(self, src, lats: np.ndarray, lons: np.ndarray):
        t = src.transform
        self.key = (tuple(t)[:6], src.width, src.height)
        # rows run north -> south: measure "down" from the top edge
        self.r, wr = _overlap(t.f - (lats + HALF), t.f - (lats - HALF), 0.0, -t.e, src.height)
        self.c, wc = _overlap(lons - HALF, lons + HALF, t.c, t.a, src.width)
        rowlat = t.f + (self.r + 0.5) * t.e
        wr = wr * np.cos(np.deg2rad(rowlat))
        self.w = wr[:, :, None] * wc[:, None, :]

    def __call__(self, arr: np.ndarray, nodata) -> np.ndarray:
        v = arr[self.r[:, :, None], self.c[:, None, :]].astype(np.float64)
        ok = np.isfinite(v) if nodata is None else (v != nodata) & np.isfinite(v)
        den = (self.w * ok).sum(axis=(1, 2))
        num = (self.w * np.where(ok, v, 0.0)).sum(axis=(1, 2))
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _month_of(name: str) -> pd.Timestamp | None:
    m = MONTHLY_RE.match(name)
    return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1) if m else None


def _reduce(files: dict, grid: pd.DataFrame, scale) -> np.ndarray:
    out = np.full((len(grid), len(files)), np.nan, dtype=np.float32)
    cm = None
    for j, (stamp, path) in enumerate(sorted(files.items())):
        with rasterio.open(path) as src:
            if cm is None or cm.key != (tuple(src.transform)[:6], src.width, src.height):
                cm = CellMean(src, grid["lat"].to_numpy(), grid["lon"].to_numpy())
            out[:, j] = cm(src.read(1), src.nodata) * scale(stamp)
    return out


def ingest(stage: Path, out_root: Path) -> None:
    ca = stage / "ca"
    grid = pd.read_csv(GRID_CSV)
    common = dict(keys=grid["key"].astype(str).to_numpy(),
                  lat=grid["lat"].to_numpy(), lon=grid["lon"].to_numpy())
    src_tag = ("Reitz, Sanford & Saxe 2023, doi:10.5066/P9EZ3VAS | exact-overlap "
               "cell-rectangle mean of the 30-arcsec rasters | ingested "
               f"{pd.Timestamp.today().date()}")

    afiles = [(int(m.group(1)), p) for p in ca.glob("AET_*/AET_*.tif")
              if "monthly" not in p.parent.name.lower() and (m := ANNUAL_RE.match(p.name))]
    annual = dict(afiles)
    if len(annual) != len(afiles):
        raise SystemExit(f"two annual rasters carry the same water year -- check {ca} "
                         "for a stale cut")
    et_wy = wy = None
    if annual:
        et_wy = _reduce(annual, grid, lambda _: 1000.0)       # m/yr -> mm/yr
        wy = np.array(sorted(annual))
        out = out_root / "et_obs" / "reitz2023_cell_annual.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, wy=wy, et=et_wy,
            dates=pd.to_datetime([f"{y - 1}-10-01" for y in wy]).to_numpy(),
            meta=np.array(src_tag + " | WATER-YEAR totals, mm/yr; `dates` = WY start"),
            **common)
        print(f"wrote {out}: {et_wy.shape[0]} cells x WY{wy[0]}-{wy[-1]} "
              f"(domain-mean {np.nanmean(et_wy):.0f} mm/yr, "
              f"NaN frac {float(np.isnan(et_wy).mean()):.4f})")
    else:
        print("no annual CA rasters -- run --cut first")

    mfiles = [p for d in ca.glob("AET_*monthly*") for p in d.rglob("*.tif")]
    if not mfiles:
        print("no monthly CA rasters (the *_monthly.zip files are ScienceBase "
              "cloud-gated) -- monthly store skipped")
        return
    monthly = {_month_of(p.name): p for p in mfiles}
    if None in monthly:
        bad = [p.name for p in mfiles if _month_of(p.name) is None][:8]
        raise SystemExit(f"cannot parse year/month from monthly rasters, e.g. {bad} "
                         "-- extend MONTHLY_RE")
    if len(monthly) != len(mfiles):
        raise SystemExit("two monthly rasters carry the same year/month -- check "
                         f"{ca} for a stale cut")
    et = _reduce(monthly, grid,                               # mm/day -> mm/month
                 lambda d: calendar.monthrange(d.year, d.month)[1])
    dates = pd.DatetimeIndex(sorted(monthly))
    gaps = pd.date_range(dates[0], dates[-1], freq="MS").difference(dates)
    # The release rescales each water year's months to the annual ensemble map,
    # so every complete water year must add up to the annual store.  This is
    # also the gate on the naming convention (calendar months, mm/day): a
    # water-year-indexed or mm/month reading misses by 4 % to 97 %.
    wy_m = dates.year + (dates.month >= 10)
    full = [y for y in np.unique(wy_m) if (wy_m == y).sum() == 12 and y in annual]
    if not full:
        raise SystemExit("no complete water year shared with the annual store -- "
                         "cannot check the monthly rasters, nothing written")
    assert et_wy is not None and wy is not None
    msum = np.stack([et[:, wy_m == y].sum(axis=1) for y in full], axis=1)
    ann = et_wy[:, np.searchsorted(wy, full)]
    rel = np.where(ann > 0, np.abs(msum / np.where(ann > 0, ann, 1.0) - 1.0), np.abs(msum))
    print(f"water-year closure vs the annual store, {len(full)} WYs "
          f"{full[0]}-{full[-1]}: |months/annual - 1| median {np.nanmedian(rel):.1e}, "
          f"99th pct {np.nanpercentile(rel, 99):.1e}, max {np.nanmax(rel):.1e}")
    if not np.nanmax(rel) < 1e-5:
        raise SystemExit("monthly rasters do not close on the annual store -- "
                         "nothing written")
    out = out_root / "et_obs" / "reitz2023_cell_monthly.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, dates=dates.to_numpy(), et=et,
                        meta=np.array(src_tag + " | mm/month (mm/day x days)"), **common)
    print(f"wrote {out}: {et.shape[0]} cells x {len(dates)} months "
          f"{dates[0]:%Y-%m}..{dates[-1]:%Y-%m} ({len(gaps)} missing months; "
          f"domain-mean {np.nanmean(et) * 12:.0f} mm/yr, "
          f"NaN frac {float(np.isnan(et).mean()):.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--cut", action="store_true",
                    help="window every tif in every staged zip to BBOX (resumable)")
    ap.add_argument("--only", nargs="+", metavar="SUBSTR",
                    help="with --cut: restrict to sources whose name contains any of these")
    ap.add_argument("--delete-source", action="store_true",
                    help="with --cut: delete a source zip once EVERY raster in it "
                         "is cut and verified bit-exact")
    ap.add_argument("--ingest", action="store_true",
                    help="CA rasters -> per-cell npz on the region grid")
    ap.add_argument("--out-root", type=Path, default=REPO / "data" / "region")
    args = ap.parse_args()
    if not (args.status or args.cut or args.ingest):
        ap.error("pick at least one of --status / --cut / --ingest")
    if args.cut:
        cut(args.stage, args.delete_source, args.only)
    if args.ingest:
        ingest(args.stage, args.out_root)
    if args.status:
        status(args.stage)


if __name__ == "__main__":
    main()
