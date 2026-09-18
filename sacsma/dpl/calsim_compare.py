"""Tier-1 CalSim3 validation of two multifamily runs side by side.

Reads the ``tier1/tier1_metrics.csv`` of two runs scored by :mod:`sacsma.dpl.calsim_tier1` and
writes one comparison: the aggregate statistics and the twenty locations, each over the full
validation window and over the **trimmed windows** (the trimmed window at the locations that
have one, the full window elsewhere; see :mod:`sacsma.dpl.calsim_windows`).  The two runs are
always compared over the same months.  The creek-coverage columns are the share of each
location's area-months under the registry's USGS creek gauges with data, which is what the
trimmed windows are chosen from; they do not depend on either run.

Usage::

    python -m sacsma.dpl.calsim_compare <run_a> <run_b> [--label-a A] [--label-b B]
                                        [--out DIR] [--data-dir data] [--no-coverage]

Writes ``tier1_comparison.md``, ``tier1_comparison.html`` and ``tier1_comparison.csv`` under
``--out`` (default: the current folder) and prints the Markdown.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from .calsim_tier1 import TRIMMED_WINDOW, VALIDATION_WINDOW, load_sets

_MAIN = ["anchor", "arcsum"]


def _load(run: Path) -> pd.DataFrame:
    m = pd.read_csv(Path(run) / "tier1" / "tier1_metrics.csv")
    return m[m.ref_kind.isin(_MAIN) & m.volume_scored]


def _trimmed(m: pd.DataFrame) -> pd.DataFrame:
    """One row per location: the trimmed window where the location has one, else the full window."""
    full = m[m.window == VALIDATION_WINDOW].set_index("set_id")
    trim = m[m.window == TRIMMED_WINDOW].set_index("set_id")
    return pd.concat([full[~full.index.isin(trim.index)], trim]).reindex(full.index)


def _coverage(data_dir: str | Path) -> dict[str, dict[int, float]]:
    from .calsim_atlas import _rim, _set_geoms, creek_overlap
    sets = load_sets(data_dir)
    geoms = _set_geoms(_rim(data_dir), sets)
    creeks = creek_overlap(sets, geoms, data_dir, trained=None, window=VALIDATION_WINDOW)
    return {s: v[1]["cover_by_wy"] for s, v in creeks.items()}


def compare(run_a: str | Path, run_b: str | Path, data_dir: str | Path = "data",
            coverage: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-location table, aggregate table) of run A against run B."""
    a, b = _load(Path(run_a)), _load(Path(run_b))
    fa, fb = (m[m.window == VALIDATION_WINDOW].set_index("set_id") for m in (a, b))
    ra, rb = _trimmed(a), _trimmed(b)
    if list(fa.index) != list(fb.index):
        raise ValueError("the two runs do not score the same tier-1 locations")
    if not (ra.win_start.equals(rb.win_start) and ra.win_end.equals(rb.win_end)):
        raise ValueError("the two runs were scored over different trimmed windows: re-run calsim_tier1 on both")
    cov = _coverage(data_dir) if coverage else {}
    rows = []
    for s in fa.index:
        w0, w1 = str(ra.loc[s, "win_start"]), str(ra.loc[s, "win_end"])
        wy0, wy1 = int(w0[:4]) + 1, int(w1[:4])
        c = cov.get(s)
        rows.append(dict(
            set_id=s, entity_id=fa.loc[s, "entity_id"], location=str(fa.loc[s, "name"]).split(" (")[0],
            reference="FLOW-UNIMPAIRED" if fa.loc[s, "ref_kind"] == "anchor" else "arc sum",
            ref_taf_yr=fa.loc[s, "ref_taf_yr"], window=f"WY{wy0}-{str(wy1)[2:]}", years=wy1 - wy0 + 1,
            trimmed=ra.loc[s, "window"] == TRIMMED_WINDOW,
            cover_full=float(np.mean(list(c.values()))) if c else np.nan,
            cover_window=float(np.mean([v for y, v in c.items() if wy0 <= y <= wy1])) if c else np.nan,
            kge_a_full=fa.loc[s, "kge"], kge_b_full=fb.loc[s, "kge"], kge_a=ra.loc[s, "kge"], kge_b=rb.loc[s, "kge"],
            nse_a_full=fa.loc[s, "nse"], nse_b_full=fb.loc[s, "nse"], nse_a=ra.loc[s, "nse"], nse_b=rb.loc[s, "nse"],
            bias_a_full=fa.loc[s, "pbias"], bias_b_full=fb.loc[s, "pbias"], bias_a=ra.loc[s, "pbias"], bias_b=rb.loc[s, "pbias"]))
    loc = pd.DataFrame(rows)

    def agg(m: pd.DataFrame, other: pd.DataFrame) -> dict:
        anchors, arcs = m[m.ref_kind == "anchor"], m[m.ref_kind == "arcsum"]
        return {"locations": len(m), "KGE mean": m.kge.mean(), "KGE median": m.kge.median(),
                "KGE mean, FLOW-UNIMPAIRED anchors": anchors.kge.mean(), "KGE mean, arc sums": arcs.kge.mean(),
                "NSE mean": m.nse.mean(), "absolute bias median, %": m.pbias.abs().median(), "absolute bias mean, %": m.pbias.abs().mean(),
                "sum of the locations' mean annual volumes, simulated TAF/yr": m.sim_taf_yr.sum(), "sum of the locations' mean annual volumes, CalSim3 TAF/yr": m.ref_taf_yr.sum(),
                "volume-weighted bias, %": 100.0 * (m.sim_taf_yr.sum() / m.ref_taf_yr.sum() - 1.0),
                "seasonal mismatch mean": m.seas_mismatch.mean(),
                "locations with the higher KGE": int((m.kge.to_numpy() > other.kge.to_numpy()).sum()),
                "locations with the smaller absolute bias": int((m.pbias.abs().to_numpy() < other.pbias.abs().to_numpy()).sum())}
    summ = pd.DataFrame({("A", "full"): agg(fa, fb), ("B", "full"): agg(fb, fa),
                         ("A", "trimmed"): agg(ra, rb), ("B", "trimmed"): agg(rb, ra)})
    return loc, summ


def _fmt(name: str, v) -> str:
    if name.startswith("locations"):
        return f"{int(v)}"
    if "TAF" in name:
        return f"{v:,.0f}"
    if "%" in name:
        return f"{v:+.1f}" if "volume-weighted bias" in name else f"{v:.1f}"
    return f"{v:.3f}"


def run_table(loc: pd.DataFrame, k: str) -> tuple[list[str], list[list[str]]]:
    """One run on its own (``k`` = "a" or "b"): header and rows, full window against the trimmed
    window, closed by the mean over the locations."""
    hdr = ["location", "trimmed window", "years", f"creek coverage, {VALIDATION_WINDOW}", "creek coverage, trimmed",
           f"KGE, {VALIDATION_WINDOW}", "KGE, trimmed", f"NSE, {VALIDATION_WINDOW}", "NSE, trimmed",
           f"bias %, {VALIDATION_WINDOW}", "bias %, trimmed"]
    rows = []
    for r in loc.itertuples(index=False):
        d = r._asdict()
        cov = ([f"{100 * r.cover_full:.1f}%", f"{100 * r.cover_window:.1f}%"] if np.isfinite(r.cover_full) else ["", ""])
        rows.append([f"{r.set_id}: {r.location}", r.window, str(r.years), *cov,
                     f"{d[f'kge_{k}_full']:.3f}", f"{d[f'kge_{k}']:.3f}", f"{d[f'nse_{k}_full']:.3f}", f"{d[f'nse_{k}']:.3f}",
                     f"{d[f'bias_{k}_full']:+.1f}", f"{d[f'bias_{k}']:+.1f}"])
    rows.append(["mean of the locations (absolute bias in the bias columns)", "", "", "", "",
                 f"{loc[f'kge_{k}_full'].mean():.3f}", f"{loc[f'kge_{k}'].mean():.3f}",
                 f"{loc[f'nse_{k}_full'].mean():.3f}", f"{loc[f'nse_{k}'].mean():.3f}",
                 f"{loc[f'bias_{k}_full'].abs().mean():.1f}", f"{loc[f'bias_{k}'].abs().mean():.1f}"])
    return hdr, rows


def _legend(n_trim: int) -> str:
    return (f"Full window = {VALIDATION_WINDOW}; trimmed windows = at the {n_trim} locations the USGS creek gauges reach, "
            "the run of at least 20 water years inside it in which they covered the least of the location (it can start "
            "late or end early), the full window elsewhere. Under the trimmed windows each location's mean annual volume "
            f"is taken over its own window, so the CalSim3 sum differs from the {VALIDATION_WINDOW} one although the "
            "reference is the same. Both runs are scored over the same months. Creek coverage = share of the location's "
            "area-months under the registry's USGS creek gauges with data (the same for every run).")


def to_markdown(loc: pd.DataFrame, summ: pd.DataFrame, label_a: str, label_b: str) -> str:
    n_trim = int(loc.trimmed.sum())
    rep = f"trimmed windows (at {n_trim} of the {len(loc)} locations)"
    lines = [f"Tier 1, CalSim3 monthly volume: `{label_a}` against `{label_b}`", "", _legend(n_trim), "",
             f"| statistic | `{label_a}`, {VALIDATION_WINDOW} | `{label_b}`, {VALIDATION_WINDOW} | `{label_a}`, {rep} | `{label_b}`, {rep} |",
             "|---|---:|---:|---:|---:|"]
    for name in summ.index:
        lines.append(f"| {name} | " + " | ".join(_fmt(name, summ.loc[name, k]) for k in summ.columns) + " |")
    lines += ["", f"| location | reference | CalSim3 TAF/yr | trimmed window | years | creek coverage, {VALIDATION_WINDOW} | creek coverage, trimmed | "
                  f"KGE `{label_a}`, {VALIDATION_WINDOW} | KGE `{label_b}`, {VALIDATION_WINDOW} | KGE `{label_a}`, trimmed | KGE `{label_b}`, trimmed | "
                  f"bias % `{label_a}`, {VALIDATION_WINDOW} | bias % `{label_b}`, {VALIDATION_WINDOW} | bias % `{label_a}`, trimmed | bias % `{label_b}`, trimmed |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in loc.itertuples(index=False):
        b = "**" if r.trimmed else ""
        cov = (f"{100 * r.cover_full:.1f}% | {100 * r.cover_window:.1f}%" if np.isfinite(r.cover_full) else " | ")
        lines.append(f"| {b}{r.set_id}: {r.location}{b} | {r.reference} | {r.ref_taf_yr:,.0f} | {b}{r.window}{b} | {r.years} | {cov} | "
                     f"{r.kge_a_full:.3f} | {r.kge_b_full:.3f} | {r.kge_a:.3f} | {r.kge_b:.3f} | "
                     f"{r.bias_a_full:+.1f} | {r.bias_b_full:+.1f} | {r.bias_a:+.1f} | {r.bias_b:+.1f} |")
    for k, lab in (("a", label_a), ("b", label_b)):
        hdr, rows = run_table(loc, k)
        lines += ["", f"`{lab}` on its own: full window against the trimmed window (trimmed locations in bold)", "",
                  "| " + " | ".join(hdr) + " |", "|---|---|" + "---:|" * (len(hdr) - 2)]
        for c, trimmed in zip(rows, list(loc.trimmed) + [False], strict=True):
            b = "**" if trimmed else ""
            lines.append(f"| {b}{c[0]}{b} | {b}{c[1]}{b} | " + " | ".join(c[2:]) + " |")
    return "\n".join(lines) + "\n"


def to_html(loc: pd.DataFrame, summ: pd.DataFrame, label_a: str, label_b: str) -> str:
    e = html.escape
    n_trim = int(loc.trimmed.sum())
    css = ("body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#fafafa;color:#222}"
           "header{padding:12px 18px;background:#1f3b57;color:#fff}header h1{margin:0;font-size:18px}"
           "header p{margin:4px 0 0;font-size:12px;opacity:.9}main{padding:14px 18px}"
           "table{border-collapse:collapse;font-size:12px;margin:8px 0 16px}th,td{border:1px solid #cfd8e0;padding:3px 7px;text-align:right}"
           "th:first-child,td:first-child,td.l,th.l{text-align:left}th.g{text-align:center;background:#e9eef3}"
           "tr.trim td{background:#fff6e0}td.win{font-weight:600;white-space:nowrap}h2{font-size:15px;margin:10px 0 4px}.note{font-size:12px;color:#555;max-width:1100px}")

    def pair(x, y, spec, better):          # bold the better of the two runs
        bx, by = better(x, y), better(y, x)
        return (f"<td>{'<b>' if bx else ''}{format(x, spec)}{'</b>' if bx else ''}</td>"
                f"<td>{'<b>' if by else ''}{format(y, spec)}{'</b>' if by else ''}</td>")
    hi = lambda x, y: x > y                # noqa: E731
    lo = lambda x, y: abs(x) < abs(y)      # noqa: E731
    p = [f"<!doctype html><html><head><meta charset='utf-8'><title>Tier-1 comparison</title><style>{css}</style></head><body>",
         f"<header><h1>Tier 1 against CalSim3 — {e(label_a)} and {e(label_b)}</h1>"
         f"<p>Monthly volume at the twenty training locations. {e(_legend(n_trim))} Bold = the better of the two runs.</p>"
         "</header><main>",
         "<h2>Overall</h2><table><tr><th rowspan='2'>statistic</th>"
         f"<th class='g' colspan='2'>{e(VALIDATION_WINDOW)}</th><th class='g' colspan='2'>trimmed windows</th></tr>"
         f"<tr><th>{e(label_a)}</th><th>{e(label_b)}</th><th>{e(label_a)}</th><th>{e(label_b)}</th></tr>"]
    for name in summ.index:
        p.append(f"<tr><td class='l'>{e(name)}</td>" + "".join(f"<td>{e(_fmt(name, summ.loc[name, k]))}</td>" for k in summ.columns) + "</tr>")
    p += ["</table><h2>By location</h2><p class='note'>Shaded rows have a trimmed window.</p>",
          "<table><tr><th rowspan='2'>location</th><th rowspan='2' class='l'>reference</th><th rowspan='2'>CalSim3 TAF/yr</th>"
          "<th rowspan='2'>trimmed window</th><th rowspan='2'>years</th><th class='g' colspan='2'>creek coverage</th>"
          f"<th class='g' colspan='2'>KGE, {e(VALIDATION_WINDOW)}</th><th class='g' colspan='2'>KGE, trimmed</th>"
          f"<th class='g' colspan='2'>bias %, {e(VALIDATION_WINDOW)}</th><th class='g' colspan='2'>bias %, trimmed</th></tr><tr>"
          f"<th>{e(VALIDATION_WINDOW)}</th><th>trimmed</th>" + f"<th>{e(label_a)}</th><th>{e(label_b)}</th>" * 4 + "</tr>"]
    for r in loc.itertuples(index=False):
        cov = (f"<td>{100 * r.cover_full:.1f}%</td><td>{100 * r.cover_window:.1f}%</td>" if np.isfinite(r.cover_full) else "<td></td><td></td>")
        cls = " class='trim'" if r.trimmed else ""
        p.append(f"<tr{cls}><td class='l'>{e(r.set_id)}: {e(r.location)} ({e(r.entity_id)})</td>"
                 f"<td class='l'>{e(r.reference)}</td><td>{r.ref_taf_yr:,.0f}</td><td class='win'>{e(r.window)}</td><td>{r.years}</td>{cov}"
                 + pair(r.kge_a_full, r.kge_b_full, ".3f", hi) + pair(r.kge_a, r.kge_b, ".3f", hi)
                 + pair(r.bias_a_full, r.bias_b_full, "+.1f", lo) + pair(r.bias_a, r.bias_b, "+.1f", lo) + "</tr>")
    p.append("</table>")
    for k, lab in (("a", label_a), ("b", label_b)):
        hdr, rows = run_table(loc, k)
        p.append(f"<h2>{e(lab)} on its own</h2><p class='note'>Full window against the trimmed window; shaded rows "
                 "have a trimmed window.</p><table><tr>" + "".join(f"<th>{e(h)}</th>" for h in hdr) + "</tr>")
        for c, trimmed in zip(rows, list(loc.trimmed) + [False], strict=True):
            cls = " class='trim'" if trimmed else ""
            p.append(f"<tr{cls}><td class='l'>{e(c[0])}</td><td class='win'>{e(c[1])}</td>"
                     + "".join(f"<td>{e(x)}</td>" for x in c[2:]) + "</tr>")
        p.append("</table>")
    p.append("</main></body></html>")
    return "\n".join(p)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--label-a", default="")
    p.add_argument("--label-b", default="")
    p.add_argument("--out", default=".", help="output folder (default: the current folder)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-coverage", action="store_true", help="skip the creek-coverage columns (no GIS needed)")
    a = p.parse_args(argv)
    la, lb = a.label_a or Path(a.run_a).name, a.label_b or Path(a.run_b).name
    loc, summ = compare(a.run_a, a.run_b, a.data_dir, coverage=not a.no_coverage)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(loc, summ, la, lb)
    (out / "tier1_comparison.md").write_text(md, encoding="utf-8")
    (out / "tier1_comparison.html").write_text(to_html(loc, summ, la, lb), encoding="utf-8")
    loc.to_csv(out / "tier1_comparison.csv", index=False)
    print(md)
    print(f"wrote tier1_comparison.md, .html and .csv under {out}")


if __name__ == "__main__":
    main()
