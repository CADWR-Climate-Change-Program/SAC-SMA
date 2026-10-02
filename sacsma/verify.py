"""``sacsma verify``: check that the installation works and the tracked results reproduce.

The repository has no test suite; the model is verified by running it.  The checks, in order:

``imports``
    every module of the package imports (modules that need torch are skipped, with a note,
    when torch is not installed)
``cli``
    every command builds its arguments and prints its help
``links``
    every relative link and image in the tracked markdown files resolves, section anchors
    included
``parity``
    the Python model against the archived MATLAB simulation, one watershed per domain
    (:data:`PARITY_BASINS`): KGE > 0.9999 and max daily difference < 0.1 mm/day
``product``
    the share model applied to the base tier-2 pass reproduces the tracked
    ``rim_inflow_monthly.csv`` of every run that carries a product (needs the run's local
    tier-2 files; skipped when they are absent)

``--quick`` runs the first three, which need no model run.  The exit code is the number of
failed checks.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import pkgutil
import re
import subprocess
from pathlib import Path

#: one watershed per domain for the parity check
PARITY_BASINS = (("15cdec", "BND"), ("9unimp", "CacheCreek"), ("11obs", "SHA"),
                 ("12rim", "SHAST"))
PARITY_KGE = 0.9999
PARITY_MAX_MM = 0.1
#: relative tolerance of the product check (the fit and ``apply`` agree to about 3e-7)
PRODUCT_RTOL = 1e-5

CHECKS = ("imports", "cli", "links", "parity", "product")
QUICK = ("imports", "cli", "links")


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


# ------------------------------------------------------------------------------- imports
def check_imports(**_) -> tuple[bool, str]:
    import sacsma

    torch_ok = _has_torch()
    bad, skipped, n = [], 0, 0
    for m in pkgutil.walk_packages(sacsma.__path__, "sacsma."):
        if m.name.endswith("__main__"):
            continue
        n += 1
        try:
            importlib.import_module(m.name)
        except ImportError as e:
            if not torch_ok and "torch" in str(e).lower():
                skipped += 1
            else:
                bad.append(f"{m.name}: {e}")
        except Exception as e:  # noqa: BLE001 — a module that fails to import is the finding
            bad.append(f"{m.name}: {e!r}")
    note = f"{n - len(bad) - skipped} of {n} modules import"
    if skipped:
        note += f" ({skipped} need torch, not installed)"
    return not bad, note + "".join(f"\n    {b}" for b in bad)


# ----------------------------------------------------------------------------------- cli
def _leaves(parser, path=()):
    import argparse

    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if not subs:
        yield path
        return
    for name, sp in subs[0].choices.items():
        yield from _leaves(sp, (*path, name))


def check_cli(**_) -> tuple[bool, str]:
    from .cli import build_parser, main
    from .dpl.cli import CALSIM_TOOLS

    torch_ok = _has_torch()
    bad, n, skipped = [], 0, 0
    for path in _leaves(build_parser()):
        if path[:2] == ("dpl", "calsim") and path[2] in CALSIM_TOOLS and not torch_ok:
            skipped += 1            # the tool's module imports torch
            continue
        n += 1
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                main([*path, "--help"])
            bad.append("sacsma " + " ".join(path) + ": --help did not exit")
        except SystemExit as e:
            if e.code not in (0, None):
                bad.append("sacsma " + " ".join(path) + f": exit {e.code}")
        except Exception as e:  # noqa: BLE001
            bad.append("sacsma " + " ".join(path) + f": {e!r}")
    note = f"{n - len(bad)} of {n} commands print their help"
    if skipped:
        note += f" ({skipped} need torch, not installed)"
    return not bad, note + "".join(f"\n    {b}" for b in bad)


# --------------------------------------------------------------------------------- links
_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_FENCE = re.compile(r"^(```|~~~).*?^\1\s*$", re.S | re.M)
_INLINE = re.compile(r"`[^`\n]*`")
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.M)


def _slug(heading: str) -> str:
    """GitHub's anchor of a heading: lower case, punctuation dropped, spaces to hyphens."""
    h = re.sub(r"`", "", heading.strip().lower())
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h)          # a link keeps its text
    h = re.sub(r"[^\w\- ]", "", h, flags=re.UNICODE)
    return h.replace(" ", "-")


def _anchors(md: Path) -> set[str]:
    text = _FENCE.sub("", md.read_text(encoding="utf-8", errors="replace"))
    out: set[str] = set()
    for h in _HEADING.findall(text):
        s, k = _slug(h), 0
        while (s if k == 0 else f"{s}-{k}") in out:         # repeated headings get -1, -2, ...
            k += 1
        out.add(s if k == 0 else f"{s}-{k}")
    return out


def _tracked_markdown(root: Path) -> list[Path]:
    try:
        r = subprocess.run(["git", "-C", str(root), "ls-files", "*.md"], capture_output=True,
                           text=True, check=True)
        return [root / p for p in r.stdout.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        skip = {"tmp", ".git", "node_modules"}
        return [p for p in root.rglob("*.md") if not skip & set(p.relative_to(root).parts)]


def check_links(root: str | Path = ".", **_) -> tuple[bool, str]:
    root = Path(root).resolve()
    files = _tracked_markdown(root)
    bad, n = [], 0
    cache: dict[Path, set[str]] = {}
    for md in files:
        if not md.exists():
            continue
        text = _INLINE.sub("", _FENCE.sub("", md.read_text(encoding="utf-8", errors="replace")))
        for target in _LINK.findall(text):
            if re.match(r"^[a-z][a-z0-9+.\-]*:", target, re.I):     # http:, mailto:, ...
                continue
            n += 1
            path, _, anchor = target.partition("#")
            dest = (md.parent / path).resolve() if path else md
            where = f"{md.relative_to(root).as_posix()} -> {target}"
            if not dest.exists():
                bad.append(f"{where}: no such file")
            elif anchor and dest.suffix.lower() == ".md":
                if dest not in cache:
                    cache[dest] = _anchors(dest)
                if anchor.lower() not in cache[dest]:
                    bad.append(f"{where}: no such section")
    return not bad, (f"{n - len(bad)} of {n} relative links resolve in {len(files)} markdown "
                     "files" + "".join(f"\n    {b}" for b in bad))


# -------------------------------------------------------------------------------- parity
def check_parity(data_dir: str = "data", **_) -> tuple[bool, str]:
    import numpy as np
    import pandas as pd

    from .io import load_reference
    from .metrics import kge
    from .model import run_basin

    ok, rows = True, []
    for domain, basin in PARITY_BASINS:
        sim = run_basin(basin, data_dir=data_dir, domain=domain)
        ref = load_reference(data_dir, basin, domain=domain)
        s = pd.Series(sim["flow"].to_numpy(), index=pd.DatetimeIndex(sim["date"]))
        r = pd.Series(ref["flow"].to_numpy(), index=pd.DatetimeIndex(ref["date"]))
        j = s.index.intersection(r.index)
        k = kge(s[j].to_numpy(), r[j].to_numpy())
        d = float(np.abs(s[j].to_numpy() - r[j].to_numpy()).max())
        good = k > PARITY_KGE and d < PARITY_MAX_MM
        ok &= good
        rows.append(f"\n    {domain:7s} {basin:11s} KGE {k:.6f}  max |d| {d:.4f} mm/day"
                    + ("" if good else "   FAIL"))
    return ok, (f"KGE > {PARITY_KGE} and max |d| < {PARITY_MAX_MM} mm/day against the MATLAB "
                "simulation" + "".join(rows))


# ------------------------------------------------------------------------------- product
def check_product(data_dir: str = "data", artifacts_dir: str = "artifacts",
                  **_) -> tuple[bool | None, str]:
    runs = sorted(Path(artifacts_dir).glob("dpl/multifamily/*/calsim_product/share_model.pt"))
    if not runs:
        return None, "no run carries a rim-inflow product"
    if not _has_torch():
        return None, "torch is not installed"
    import numpy as np
    import pandas as pd
    import torch

    from .dpl.calsim.product import apply

    ok, rows, done = True, [], 0
    for pt in runs:
        run = pt.parents[1]
        tier2 = run / "tier2"
        if not (tier2 / "tier2_components_monthly.csv").exists():
            rows.append(f"\n    {run.name}: skipped, the base tier-2 pass with runoff parts is "
                        "local-only and not here (sacsma dpl calsim tier2 <run> --components parts)")
            continue
        got = apply(torch.load(pt, weights_only=False), tier2, data_dir)
        ref = pd.read_csv(pt.parent / "rim_inflow_monthly.csv")
        m = ref.merge(got, on=["arc", "month"], suffixes=("_ref", ""))
        rel = float((np.abs(m.taf - m.taf_ref) / np.maximum(np.abs(m.taf_ref), 1e-6)).max())
        good = len(m) == len(ref) == len(got) and rel < PRODUCT_RTOL
        ok &= good
        done += 1
        rows.append(f"\n    {run.name}: {len(m)} arc-months, max relative difference {rel:.1e}"
                    + ("" if good else "   FAIL"))
    return (ok if done else None), "apply reproduces the tracked product" + "".join(rows)


_FUNCS = {"imports": check_imports, "cli": check_cli, "links": check_links,
          "parity": check_parity, "product": check_product}


def run_checks(checks=None, *, data_dir: str = "data", quick: bool = False) -> int:
    """Run the named checks (default: all, or the three quick ones); print one line each;
    return the number that failed."""
    names = tuple(checks) if checks else (QUICK if quick else CHECKS)
    unknown = [c for c in names if c not in _FUNCS]
    if unknown:
        raise SystemExit(f"unknown check(s) {unknown}; choose from {', '.join(CHECKS)}")
    failed = 0
    for name in names:
        ok, note = _FUNCS[name](data_dir=data_dir)
        status = "skip" if ok is None else "ok  " if ok else "FAIL"
        failed += ok is False
        print(f"{status}  {name:8s} {note}", flush=True)
    print(f"verify: {len(names) - failed} of {len(names)} checks passed"
          + ("" if not failed else f", {failed} FAILED"))
    return failed
