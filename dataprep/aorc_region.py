"""Build the AORC region forcing store: NOAA AORC v1.1 1-km hourly -> daily 1/16 deg.

Streams the AORC Zarr store on S3 chunk-by-chunk, aggregates each hourly field
onto the 4410-cell region grid *in memory*, and keeps only the daily result.
The ~950 GB of compressed source is never written to disk -- per (variable,
year) partials are ~6 MB each, so the whole job costs ~2.4 GB of local storage.

The source store (verified against the live bucket, not assumed):

    s3://noaa-nws-aorc-v1-1-1km/{year}.zarr          zarr v2, consolidated
    lat = 20.0   + 0.0083330*i,  i = 0..4200         (20 .. 55 N)
    lon = -130.0 + 0.0083330*j,  j = 0..8400         (-130 .. -60 E)
    arrays (time, latitude, longitude) int16 + zstd-3, chunks (144, 128, 256)

A 144-hour time chunk is exactly 6 days, and hour 0 of each store is 00:00 UTC
on 1 January, so no *UTC* day ever straddles a chunk boundary.

**Day convention.** ``--utc-offset`` defaults to 0 (UTC calendar days) because
that is what the committed Livneh region store actually uses -- established by
scanning all 24 whole-hour offsets for 2015 precipitation against
``historical_livneh_unsplit.nc``.  Agreement falls off monotonically as the
window is shifted away from UTC:

    offset    0  ->  r_regional 0.9565   r_cellday 0.8234
    offset   -8  ->  r_regional 0.7624   r_cellday 0.6603   (PST midnight)
    offset  -16  ->  r_regional 0.5616   r_cellday 0.4584

A non-zero offset makes local day D span UTC hours [24D-offset, 24D-offset+24),
so each year additionally needs the first time chunk of the following year.

Usage
-----
    python dataprep/aorc_region.py --years 2020                # one year, all vars
    python dataprep/aorc_region.py --years 1979-2025 --vars APCP_surface
    python dataprep/aorc_region.py --all                       # everything, resumable
    python dataprep/aorc_region.py --assemble                  # partials -> aorc.nc

Re-running skips completed (variable, year) partials, so an interrupted job --
a dropped VPN, a sleeping laptop -- resumes exactly where it stopped.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import zstandard as zstd

# --- source geometry (verified against the live store) ------------------------
BUCKET = "https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com"
LAT0, LON0, DD = 20.0, -130.0, 0.0083330
NLAT, NLON = 4201, 8401
CT, CI, CJ = 144, 128, 256
YEAR_MIN, YEAR_MAX = 1979, 2025
#: every AORC array declares ``missing_value``/``fill_value`` -32767 (int16),
#: right beside the ``scale_factor`` that ``scale_factors`` reads.  Scaling it
#: as data is what contaminated the 2026-07-28 pull: a filled hour became
#: -3276.7 and a fully filled precipitation day -78 640 mm, in 5 of 47 years.
FILL = np.int16(-32767)

REPO = Path(__file__).resolve().parents[1]
GRID_CSV = REPO / "data" / "region" / "grid_cells.csv"
OUT_NC = REPO / "data" / "region" / "forcing" / "aorc.nc"
PARTS = REPO / "tmp" / "aorc_parts"           # tmp/ is gitignored, local-only
RES = 1 / 16

#: AORC variable -> (output field(s), how each day is reduced from the hourly series)
#: 'sum' for precipitation, 'min'/'max' for the temperature extremes, else 'mean'.
VARIABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "APCP_surface": (("prcp", "sum"),),
    "TMP_2maboveground": (("tmin", "min"), ("tmax", "max")),
    "DLWRF_surface": (("dlwrf", "mean"),),
    "DSWRF_surface": (("dswrf", "mean"),),
    "PRES_surface": (("pres", "mean"),),
    "SPFH_2maboveground": (("spfh", "mean"),),
    "UGRD_10maboveground": (("ugrd", "mean"),),
    "VGRD_10maboveground": (("vgrd", "mean"),),
}
#: fields carrying a unit conversion applied after scale_factor (K -> degC)
KELVIN = {"tmin", "tmax"}


def _atomic_npy(path: Path, arr: np.ndarray) -> None:
    """Write ``arr`` so a kill mid-write can never leave a half-file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:            # via a handle: np.save would append .npy
        np.save(fh, arr, allow_pickle=False)
    tmp.replace(path)


def _session(pool: int = 64) -> requests.Session:
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool))
    return s


def hours_in_year(y: int) -> int:
    return 8784 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 8760


def load_region_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Region cell keys and their 1/16-deg centre coordinates, in file order."""
    rg = pd.read_csv(GRID_CSV)
    latc = next(c for c in rg.columns if "lat" in c.lower())
    lonc = next(c for c in rg.columns if "lon" in c.lower())
    lat = rg[latc].to_numpy(float)
    lon = rg[lonc].to_numpy(float)
    keys = np.array([f"{a:.5f}_{o:.5f}" for a, o in zip(lat, lon)])
    return keys, lat, lon


def build_index() -> dict:
    """Map every AORC cell in the window onto its containing region cell.

    Returns the chunk window plus, for each spatial chunk, the flat AORC-cell
    offsets that land in the region grid and the region-cell index they feed.
    """
    keys, rlat, rlon = load_region_grid()
    # region cell -> position, via the integer lattice its centre sits on
    lat_ix = np.rint(rlat / RES - 0.5).astype(np.int64)
    lon_ix = np.rint(rlon / RES - 0.5).astype(np.int64)
    lut = {(a, o): n for n, (a, o) in enumerate(zip(lat_ix, lon_ix))}

    pad = RES / 2 + 2 * DD
    i0 = max(0, int(np.floor((rlat.min() - pad - LAT0) / DD)))
    i1 = min(NLAT, int(np.ceil((rlat.max() + pad - LAT0) / DD)) + 1)
    j0 = max(0, int(np.floor((rlon.min() - pad - LON0) / DD)))
    j1 = min(NLON, int(np.ceil((rlon.max() + pad - LON0) / DD)) + 1)
    ci0, ci1 = i0 // CI, -(-i1 // CI)
    cj0, cj1 = j0 // CJ, -(-j1 // CJ)

    chunks = {}
    ncells = len(keys)
    counts = np.zeros(ncells, np.int64)
    for ci in range(ci0, ci1):
        lat = LAT0 + (ci * CI + np.arange(CI)) * DD
        a_ix = np.rint(lat / RES - 0.5).astype(np.int64)
        for cj in range(cj0, cj1):
            lon = LON0 + (cj * CJ + np.arange(CJ)) * DD
            o_ix = np.rint(lon / RES - 0.5).astype(np.int64)
            aa, oo = np.meshgrid(a_ix, o_ix, indexing="ij")
            dest = np.full(aa.size, -1, np.int64)
            flat_a, flat_o = aa.ravel(), oo.ravel()
            for n, (a, o) in enumerate(zip(flat_a, flat_o)):
                d = lut.get((a, o))
                if d is not None:
                    dest[n] = d
            hit = dest >= 0
            if hit.any():
                src = np.nonzero(hit)[0].astype(np.int32)
                dst = dest[hit].astype(np.int32)
                chunks[(ci, cj)] = (src, dst)
                counts += np.bincount(dst, minlength=ncells)
    if (counts == 0).any():
        raise RuntimeError(f"{int((counts == 0).sum())} region cells got no AORC cell")
    return dict(keys=keys, chunks=chunks, counts=counts, ncells=ncells)


def scale_factors(sess: requests.Session, year: int = 2020) -> dict[str, float]:
    meta = sess.get(f"{BUCKET}/{year}.zarr/.zmetadata", timeout=120).json()["metadata"]
    return {v: float(meta[f"{v}/.zattrs"].get("scale_factor", 1.0)) for v in VARIABLES}


def fetch_hourly(sess, var: str, year: int, idx: dict, workers: int,
                 tchunks: list[int] | None = None) -> np.ndarray:
    """Hourly region-cell means for ``year``: array (nhours, ncells) float32.

    Fetched one time chunk at a time so at most ``len(idx['chunks'])`` decoded
    blocks (~250 MB) are ever resident, while still keeping the link saturated.
    """
    nh = hours_in_year(year)
    want = list(range(-(-nh // CT))) if tchunks is None else tchunks
    ncells = idx["ncells"]
    out = np.zeros((nh, ncells), np.float32)
    raw_bytes = CT * CI * CJ * 2
    ckpt = PARTS / "_hourly" / f"{var}_{year}"

    def one(job):
        t, ci, cj = job
        url = f"{BUCKET}/{year}.zarr/{var}/{t}.{ci}.{cj}"
        for attempt in range(6):
            try:
                r = sess.get(url, timeout=180)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                # a ZstdDecompressor is NOT thread-safe -- one per call
                raw = zstd.ZstdDecompressor().decompress(
                    r.content, max_output_size=raw_bytes)
                return np.frombuffer(raw, "<i2").reshape(CT, CI * CJ)
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        return None

    spatial = list(idx["chunks"])
    with cf.ThreadPoolExecutor(workers) as ex:
        for t in want:
            h0 = t * CT
            n = min(CT, nh - h0)
            if n <= 0:
                continue
            # per-time-chunk checkpoint: ~35 MB of transfer, so an unexpected
            # stop costs seconds of work, not the whole year
            cp = ckpt / f"{t:03d}.npy"
            if cp.exists():
                try:
                    out[h0:h0 + n] = np.load(cp)
                    continue
                except Exception:
                    cp.unlink(missing_ok=True)      # truncated by a hard kill
            acc = np.zeros((n, ncells), np.float32)
            nval = np.zeros((n, ncells), np.float32)
            jobs = [(t, ci, cj) for (ci, cj) in spatial]
            for (ci, cj), block in zip(spatial, ex.map(one, jobs)):
                if block is None:
                    continue
                src, dst = idx["chunks"][(ci, cj)]
                raw = block[:n][:, src]
                # AORC marks missing as int16 FILL.  Averaging it as if it were
                # data is what poisoned the first pull, so drop it from BOTH the
                # sum and the divisor: each hour becomes the mean over the 1-km
                # cells that actually reported.  The divisor must be counted per
                # hour -- a static per-cell count also mis-divides whenever a
                # spatial chunk 404s.
                good = raw != FILL
                vals = np.where(good, raw, 0).astype(np.float32)
                # one bincount over (hour, region-cell) instead of n scattered adds
                offs = (np.arange(n, dtype=np.int64)[:, None] * ncells
                        + dst[None, :].astype(np.int64)).ravel()
                acc += np.bincount(
                    offs, weights=vals.ravel(), minlength=n * ncells
                ).reshape(n, ncells).astype(np.float32)
                nval += np.bincount(
                    offs, weights=good.ravel(), minlength=n * ncells
                ).reshape(n, ncells).astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                acc = np.where(nval > 0, acc / nval, np.nan).astype(np.float32)
            out[h0:h0 + n] = acc
            _atomic_npy(cp, acc)
    return out


def daily_from_hourly(hourly: np.ndarray, tail: np.ndarray | None,
                      var: str, scale: float, offset_h: int) -> dict[str, np.ndarray]:
    """Reduce an hourly (nhours, ncells) block to daily fields on local days."""
    shift = -offset_h                      # UTC hour at which the local day starts
    series = hourly if tail is None else np.concatenate([hourly, tail], 0)
    usable = series[shift:]
    ndays = usable.shape[0] // 24
    if ndays == 0:
        raise RuntimeError("not enough hours for a single local day")
    block = usable[: ndays * 24].reshape(ndays, 24, -1) * np.float32(scale)
    # An hour whose 1-km box reported nothing is NaN (see fetch_hourly).  Reduce
    # over the hours that did report, and surrender a day only when none did.
    # A daily SUM therefore under-reports a partly observed day rather than
    # inflating it to a full 24 h -- the deliberate convention, since inventing
    # precipitation for unobserved hours is the worse failure.
    dead = np.all(np.isnan(block), axis=1)
    fields = {}
    for name, how in VARIABLES[var]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
            if how == "sum":
                v = np.nansum(block, 1)      # 0.0 for an all-NaN day -> masked below
            elif how == "min":
                v = np.nanmin(block, 1)
            elif how == "max":
                v = np.nanmax(block, 1)
            else:
                v = np.nanmean(block, 1)
        v = np.where(dead, np.nan, v)
        if name in KELVIN:
            v = v - np.float32(273.15)
        fields[name] = v.astype(np.float32)
    return fields


def part_path(var: str, year: int) -> Path:
    return PARTS / f"{var}_{year}.npz"


def run_year(sess, var: str, year: int, idx: dict, scale: float,
             workers: int, offset_h: int) -> None:
    p = part_path(var, year)
    if p.exists():
        return
    t0 = time.time()
    hourly = fetch_hourly(sess, var, year, idx, workers)
    # local day 0 starts at UTC hour ``shift``; the tail of the year therefore
    # needs the first ``shift`` hours of the following year to close its last day
    shift = -offset_h
    tail = None
    if shift > 0 and year < YEAR_MAX:
        tail = fetch_hourly(sess, var, year + 1, idx, workers, tchunks=[0])[:shift]
    fields = daily_from_hourly(hourly, tail, var, scale, offset_h)
    PARTS.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".npz.part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **fields)
    tmp.replace(p)
    # the year is banked -- drop its hourly checkpoints
    shutil.rmtree(PARTS / "_hourly" / f"{var}_{year}", ignore_errors=True)
    n = next(iter(fields.values())).shape[0]
    print(f"  {var:<20s} {year}  {n:4d} days  {time.time() - t0:6.1f}s  -> {p.name}",
          flush=True)


def assemble(offset_h: int, variables: list[str] | None = None) -> None:
    """Stitch the per-(variable, year) partials into the region forcing store."""
    import xarray as xr

    keys, _, _ = load_region_grid()
    seen = sorted({int(f.stem.rsplit("_", 1)[1]) for f in PARTS.glob("*.npz")})
    if not seen:
        sys.exit(f"no partials in {PARTS}")
    # assemble whatever variables have been pulled so far, over the years they
    # all share -- so the physics-critical subset can land before the rest
    use = [v for v in (variables or VARIABLES) if any(part_path(v, y).exists()
                                                      for y in seen)]
    if not use:
        sys.exit("no partials for the requested variables")
    years = [y for y in seen if all(part_path(v, y).exists() for v in use)]
    skipped = [y for y in seen if y not in years]
    if skipped:
        print(f"skipping {len(skipped)} year(s) missing some variable: {skipped}")
    if not years:
        sys.exit(f"no year has all of {use}")
    print(f"assembling {use} over {len(years)} year(s): {years[0]}..{years[-1]}")
    names = [n for v in use for n, _ in VARIABLES[v]]

    data: dict[str, list[np.ndarray]] = {n: [] for n in names}
    times: list[np.ndarray] = []
    for y in years:
        ndays = set()
        for var in use:
            with np.load(part_path(var, y)) as z:
                for n in z.files:
                    data[n].append(z[n])
                    ndays.add(z[n].shape[0])
        if len(ndays) != 1:
            sys.exit(f"{y}: variables disagree on day count {sorted(ndays)}")
        times.append(np.datetime64(f"{y}-01-01")
                     + np.arange(ndays.pop()).astype("timedelta64[D]"))

    time_ax = np.concatenate(times)
    ds = xr.Dataset(
        {n: (("key", "time"), np.concatenate(v, 0).T.astype(np.float32))
         for n, v in data.items() if v},
        coords={"key": keys, "time": time_ax},
        attrs={
            "title": "NOAA AORC v1.1 daily forcing, unified region store",
            "source": (f"s3://noaa-nws-aorc-v1-1-1km {{year}}.zarr, 1-km hourly, "
                       f"box-averaged to the 1/16-deg region grid; daily on local "
                       f"days (UTC{offset_h:+d})"),
            "resolution": "1/16 deg (aggregated from 0.0083330 deg / 30 arcsec)",
        },
    )
    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(OUT_NC, encoding=enc)
    print(f"wrote {OUT_NC}  ({OUT_NC.stat().st_size / 1e9:.2f} GB)  "
          f"vars={list(ds.data_vars)}  time={ds.time.values[0]}..{ds.time.values[-1]}")


def status() -> None:
    """Progress of a long-running pull, read from what is banked on disk."""
    years = list(range(YEAR_MIN, YEAR_MAX + 1))
    total = len(VARIABLES) * len(years)
    done_by_var = {v: sum(part_path(v, y).exists() for y in years) for v in VARIABLES}
    done = sum(done_by_var.values())
    print(f"banked {done}/{total} (variable, year) partials  [{100 * done / total:5.1f}%]")
    for v, n in done_by_var.items():
        bar = "#" * int(30 * n / len(years))
        print(f"  {v:<20s} {n:2d}/{len(years)}  {bar}")

    live = sorted((PARTS / "_hourly").glob("*/")) if (PARTS / "_hourly").exists() else []
    for d in live:
        got = len(list(d.glob("*.npy")))
        if got:
            print(f"  in flight: {d.name} {got}/61 time chunks")

    sz = sum(f.stat().st_size for f in PARTS.rglob("*") if f.is_file())
    print(f"  local partial store: {sz / 1e9:.2f} GB")
    if done and done < total:
        print(f"  remaining: {total - done} partials")


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [y for y in out if not YEAR_MIN <= y <= YEAR_MAX]
    if bad:
        sys.exit(f"years outside AORC coverage {YEAR_MIN}-{YEAR_MAX}: {bad}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default=None, help="e.g. 2020 or 1979-2025 or 1990,1991")
    ap.add_argument("--vars", nargs="*", default=None, choices=list(VARIABLES))
    ap.add_argument("--all", action="store_true", help="every variable, 1979-2025")
    ap.add_argument("--assemble", action="store_true", help="partials -> aorc.nc")
    ap.add_argument("--status", action="store_true", help="progress of a running pull")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel chunk fetches (measured: saturates at 16)")
    ap.add_argument("--utc-offset", type=int, default=0,
                    help="day-window offset from UTC; 0 (default) matches the "
                         "committed Livneh region store, -8 would be PST midnight")
    a = ap.parse_args()

    if a.status:
        status()
        return
    if a.assemble:
        assemble(a.utc_offset, a.vars)
        return

    years = parse_years(a.years) if a.years else (
        list(range(YEAR_MIN, YEAR_MAX + 1)) if a.all else None)
    if years is None:
        ap.error("give --years, or --all, or --assemble")
    variables = a.vars or list(VARIABLES)

    print(f"region grid -> AORC index ...", flush=True)
    idx = build_index()
    nchunk = len(idx["chunks"])
    print(f"  {idx['ncells']} region cells, {nchunk} spatial chunks, "
          f"{idx['counts'].min()}-{idx['counts'].max()} AORC cells per region cell")

    sess = _session(max(64, a.workers * 2))
    scales = scale_factors(sess)
    todo = [(v, y) for v in variables for y in years if not part_path(v, y).exists()]
    print(f"{len(todo)} (variable, year) partials to fetch "
          f"[{len(variables) * len(years) - len(todo)} already done]", flush=True)

    for var, year in todo:
        run_year(sess, var, year, idx, scales[var], a.workers, a.utc_offset)


if __name__ == "__main__":
    main()
