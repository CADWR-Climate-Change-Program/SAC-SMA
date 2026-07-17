"""Regenerate the MyST doc pages from the pandoc report sources.

The narrative pages (`index`, `part1`, `part2`, `appendix_*`, `references`) are
DERIVED from the pandoc report markdown (`00_front.md` … `05_appendix_c.md` +
`references.bib`) that also builds the Word deliverable. When that source
changes, re-run this to refresh `docs/`:

    python docs/_sync_from_pandoc.py --src "<path to the report source folder>"

The source folder defaults to $SACSMA_REPORT_SRC. It is NOT tracked in this
repo (it lives in the report author's document store); `docs/*.md` are the
committed, buildable copies. Transforms applied: strip `{-}`/`{.unnumbered}`;
demote part1/part2 headings after the first (one H1 per page); `[@k]`/`[@a; @b]`
→ `{cite:p}`, bare `@k` → `{cite:t}`; `![cap](artifacts/..){width=Xin}` →
`{figure}` (path → `../artifacts`, caption kept); pandoc `: Table N.` captions →
italic paragraph; `~pet~` → `$_{pet}$`; the `::: {#refs}` block → a References
page with `{bibliography}`; and the Part II "Next steps" section → a pointer to
the repo issue tracker.

This file is not part of the Sphinx build (Sphinx ingests only .md/.rst).
"""
import argparse
import os
import re
import shutil
from pathlib import Path

ISSUES_URL = "https://github.com/CADWR-Climate-Change-Program/SAC-SMA/issues"
DST = Path(__file__).resolve().parent


def strip_unnumbered(t):
    return re.sub(r"[ \t]*\{(?:-|\.unnumbered)\}[ \t]*$", "", t, flags=re.M)


def demote_after_first(t):
    """Keep the first heading as the page H1; push every later heading down one
    level so each page has a single top-level title."""
    out, seen, in_code = [], False, False
    for ln in t.split("\n"):
        if ln.lstrip().startswith("```"):
            in_code = not in_code
            out.append(ln)
            continue
        if not in_code and re.match(r"^#{1,6}\s+\S", ln):
            if seen:
                out.append("#" + ln)
            else:
                seen = True
                out.append(ln)
        else:
            out.append(ln)
    return "\n".join(out)


def convert_citations(t):
    def bracket(m):
        keys = re.findall(r"@([A-Za-z][\w:]*)", m.group(1))
        return "{cite:p}`" + ",".join(keys) + "`" if keys else m.group(0)
    t = re.sub(r"\[([^\]]*@[^\]]*)\]", bracket, t)
    t = re.sub(r"(?<![\w`\[])@([A-Za-z][\w]+)", r"{cite:t}`\1`", t)
    return t


def convert_subscripts(t):
    return re.sub(r"~([A-Za-z0-9]+)~", r"$_{\1}$", t)


def convert_images(t):
    pat = re.compile(r"!\[(?P<cap>.*?)\]\((?P<path>[^)]*)\)(?P<attrs>\{[^}]*\})?",
                     re.S)

    def repl(m):
        cap = " ".join(m.group("cap").split())
        path = m.group("path").strip()
        if path.startswith("artifacts/"):
            path = "../" + path
        w = re.search(r"width\s*=\s*([0-9.]+\w+)", m.group("attrs") or "")
        head = [f"```{{figure}} {path}"]
        if w:
            head.append(f":width: {w.group(1)}")
        head += ["", cap, "```"]
        return "\n".join(head)
    return pat.sub(repl, t)


def convert_table_captions(t):
    lines, out, i = t.split("\n"), [], 0
    while i < len(lines):
        m = re.match(r"^:\s+(.*)$", lines[i])
        if m:
            cap = [m.group(1)]
            i += 1
            while i < len(lines) and lines[i].strip():
                cap.append(lines[i].strip())
                i += 1
            out.append("*" + " ".join(cap).strip() + "*")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def drop_refs_block(t):
    return re.split(r"\n#\s+References\b", t)[0].rstrip() + "\n"


def replace_next_steps(t):
    head = re.split(r"\n#\s+Next steps\b", t)[0].rstrip()
    pointer = (
        "\n\n# Next steps\n\n"
        "The next-phase workstreams that extend this system toward the full "
        "CalSim domain — multi-timescale calibration, nested sub-arc training, a "
        "fuller training domain, and the physics/hybrid refinement options — are "
        "tracked as open issues in the modeling repository:\n\n"
        f"<{ISSUES_URL}>\n"
    )
    return head + pointer + "\n"


def process(t, *, demote):
    t = strip_unnumbered(t)
    if demote:
        t = demote_after_first(t)
    t = convert_images(t)
    t = convert_table_captions(t)
    t = convert_citations(t)
    t = convert_subscripts(t)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=os.environ.get("SACSMA_REPORT_SRC"),
                    help="report source folder (00_front.md … references.bib); "
                         "defaults to $SACSMA_REPORT_SRC")
    args = ap.parse_args()
    if not args.src:
        ap.error("no --src and $SACSMA_REPORT_SRC unset")
    src = Path(args.src)

    (DST / "part1.md").write_text(
        process((src / "01_part1.md").read_text(encoding="utf-8"), demote=True),
        encoding="utf-8")

    p2 = replace_next_steps(drop_refs_block(
        (src / "02_part2.md").read_text(encoding="utf-8")))
    (DST / "part2.md").write_text(process(p2, demote=True), encoding="utf-8")

    for s, d in [("03_appendix_a.md", "appendix_a.md"),
                 ("04_appendix_b.md", "appendix_b.md"),
                 ("05_appendix_c.md", "appendix_c.md")]:
        (DST / d).write_text(
            process((src / s).read_text(encoding="utf-8"), demote=False),
            encoding="utf-8")

    (DST / "references.md").write_text(
        "# References\n\n```{bibliography}\n:all:\n```\n", encoding="utf-8")

    front = (src / "00_front.md").read_text(encoding="utf-8")
    front = re.sub(r"^---.*?---\s*", "", front, flags=re.S)
    front = re.sub(r"^#\s+Executive summary\s*", "",
                   strip_unnumbered(front)).strip()
    front = convert_citations(front)
    (DST / "index.md").write_text(
        "# SAC-SMA Hydrologic Modeling for CalSim Stochastic Hydrology\n\n"
        "*Current Implementation, Evaluation, and Differentiable "
        "Reimplementation*\n\n"
        "**California Department of Water Resources — July 2026 (DRAFT)**\n\n"
        "## Executive summary\n\n" + front + "\n\n"
        "```{toctree}\n:maxdepth: 2\n:caption: Contents\n\n"
        "part1\npart2\nappendix_a\nappendix_b\nappendix_c\nreferences\n```\n",
        encoding="utf-8")

    shutil.copy(src / "references.bib", DST / "references.bib")
    print("regenerated docs pages from", src)


if __name__ == "__main__":
    main()
