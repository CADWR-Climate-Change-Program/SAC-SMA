"""Trimmed tier-1 validation windows: the years of least USGS creek coverage at every location.

The USGS daily creek gauges train over their whole records, which reach into the tier-1
validation window (water years 1950-1984); the CDEC and DWR-unimpaired targets only start
after it.  Tier 1 therefore scores every location the creeks reach twice: over the full window and
over a **trimmed window**, the years in which the registry's creeks covered the least of it.  This
module derives that window from the data alone (the registry's creek family, their
delineated watersheds and their daily records), so it is the same for every run and two runs
stay comparable:

* the creek coverage of a location in a water year is the share of its area under creek
  gauges that have data, averaged over the year's months
  (:func:`sacsma.dpl.calsim_atlas.creek_overlap`, all registry creeks taken as trained);
* every contiguous run of at least ``--min-years`` (20) water years inside the validation
  window is a candidate; the pick is the lowest mean coverage, ties going to the longer and
  then the later window;
* a location where no run of years lowers the coverage keeps all 35 years: the locations no
  creek reaches.  ``--min-gain`` raises that bar to a gain in coverage (as a share of the
  location's area; default 0, any gain).

The trimmed window can start late (the large creeks stopped early) or end early (they started
late).  It never extends past the validation window: from water year 1985 the CDEC and
DWR-unimpaired targets are in training.  What it leaves is not always small: the coverage
inside the trimmed window is reported next to the coverage over the full one, and where the
two are close the trimmed score mostly measures the change of years.  The first and last
water year of each location's window are the ``val_start_wy`` and ``val_end_wy`` columns of
``data/calsim/tier1_sets.csv``, which :mod:`sacsma.dpl.calsim_tier1` scores as
``window = trimmed``.

Usage::

    python -m sacsma.dpl.calsim_windows [--data-dir data] [--min-years 20] [--min-gain 0] [--write]

Prints the per-location table and checks it against the set table (exit status 1 on a
difference); ``--write`` stores the derived values in the set table instead.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from ..calsim import calsim_dir
from .calsim_tier1 import VALIDATION_WINDOW, load_sets

MIN_YEARS = 20
MIN_GAIN = 0.0
_TIE = 1e-9
_COLUMNS = ("val_start_wy", "val_end_wy")


def pick_window(cover_by_wy: dict[int, float], min_years: int = MIN_YEARS,
                min_gain: float = MIN_GAIN) -> dict:
    """The trimmed window of one location from its creek coverage by water year."""
    years = sorted(cover_by_wy)
    if not 1 <= min_years <= len(years):
        raise ValueError(f"min_years = {min_years}: a window needs 1 to {len(years)} water years")
    c = pd.Series([cover_by_wy[y] for y in years], index=years, dtype=float)
    full = float(c.mean())
    runs = [(float(c.loc[a:b].mean()), a, b) for i, a in enumerate(years) for b in years[i + min_years - 1:]]
    low = min(m for m, _, _ in runs)
    # means within _TIE of the lowest are one tie, which goes to the longer and then the later window
    cov, a, b = min((r for r in runs if r[0] - low <= _TIE), key=lambda r: (-(r[2] - r[1] + 1), -r[2]))
    gain = full - cov
    trimmed = (a, b) != (years[0], years[-1]) and gain >= min_gain
    if not trimmed:
        a, b, cov = years[0], years[-1], full
    return dict(start_wy=a, end_wy=b, n_years=b - a + 1, cover_full=full, cover=float(cov),
                best_gain=gain, trimmed=trimmed)


def derive(data_dir: str | Path = "data", min_years: int = MIN_YEARS,
           min_gain: float = MIN_GAIN) -> pd.DataFrame:
    """One row per tier-1 location: its creek coverage and the window the rule gives."""
    from .calsim_atlas import _rim, _set_geoms, creek_overlap
    sets = load_sets(data_dir)
    geoms = _set_geoms(_rim(data_dir), sets)
    creeks = creek_overlap(sets, geoms, data_dir, trained=None, window=VALIDATION_WINDOW)
    rows = []
    for s in sets.itertuples(index=False):
        p = pick_window(creeks[s.set_id][1]["cover_by_wy"], min_years, min_gain)
        rows.append(dict(set_id=s.set_id, entity_id=s.entity_id, val_start_wy=int(s.val_start_wy),
                         val_end_wy=int(s.val_end_wy), **p))
    return pd.DataFrame(rows)


def write_windows(data_dir: str | Path, windows: dict[str, tuple[int, int]]) -> Path:
    """Store ``val_start_wy`` and ``val_end_wy`` in the set table, leaving every other cell's value as written."""
    path = calsim_dir(data_dir) / "tier1_sets.csv"
    eol = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r]
    head = rows[0]
    for j, col in enumerate(_COLUMNS):
        if col not in head:
            # the two columns sit side by side in this order, ahead of the free-text note
            k = (head.index(_COLUMNS[j - 1]) + 1 if j and _COLUMNS[j - 1] in head else
                 head.index(_COLUMNS[1]) if not j and _COLUMNS[1] in head else
                 head.index("note") if "note" in head else len(head))
            head.insert(k, col)
            for r in rows[1:]:
                r.insert(k, "")
    sid = head.index("set_id")
    for r in rows[1:]:
        for col, wy in zip(_COLUMNS, windows[r[sid]], strict=True):
            r[head.index(col)] = str(wy)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator=eol).writerows(rows)
    return path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--min-years", type=int, default=MIN_YEARS,
                   help="shortest window in water years (default 20, the value the atlas, the comparison page "
                        "and the docs state; keep it with --write)")
    p.add_argument("--min-gain", type=float, default=MIN_GAIN,
                   help="keep the full window unless the coverage drops by at least this share of the area")
    p.add_argument("--write", action="store_true",
                   help="store the derived val_start_wy and val_end_wy in tier1_sets.csv")
    a = p.parse_args(argv)
    t = derive(a.data_dir, a.min_years, a.min_gain)
    show = t.assign(window=[f"WY{r.start_wy}-{r.end_wy}" for r in t.itertuples(index=False)])
    for c in ("cover_full", "cover", "best_gain"):
        show[c] = (100 * show[c]).round(1)
    print(f"creek coverage inside {VALIDATION_WINDOW} (% of the location's area) and the window of at least "
          f"{a.min_years} water years with the lowest coverage"
          + (f", kept only for a gain of at least {100 * a.min_gain:g} points" if a.min_gain > 0 else ""))
    print(show[["set_id", "entity_id", "cover_full", "best_gain", "window", "n_years", "cover", "trimmed"]]
          .to_string(index=False))
    if a.write:
        path = write_windows(a.data_dir, {r.set_id: (r.start_wy, r.end_wy) for r in t.itertuples(index=False)})
        print(f"wrote {' and '.join(_COLUMNS)} for {len(t)} locations ({int(t.trimmed.sum())} trimmed) to {path}")
        return
    bad = t[(t.start_wy != t.val_start_wy) | (t.end_wy != t.val_end_wy)]
    if len(bad):
        print("the set table differs from the rule at: "
              + ", ".join(f"{r.set_id} (table WY{r.val_start_wy}-{r.val_end_wy}, rule WY{r.start_wy}-{r.end_wy})"
                          for r in bad.itertuples(index=False)))
        sys.exit(1)
    print(f"the set table agrees with the rule at all {len(t)} locations ({int(t.trimmed.sum())} trimmed)")


if __name__ == "__main__":
    main()
