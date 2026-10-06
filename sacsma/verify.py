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
``artifacts``
    the output tree keeps its layout (``artifacts/README.md``): every tracked file sits in
    ``product/``, ``models/`` or ``results/``, each of which has its README; nothing under
    ``_local/`` is tracked; no file a command wrote sits untracked in a tracked folder; every
    learned-parameter run has its model folder and its results folder
``parity``
    the Python model against the archived MATLAB simulation, one watershed per domain
    (:data:`PARITY_BASINS`): KGE > 0.9999 and max daily difference < 0.1 mm/day
``learned``
    the learned model's step on the CPU engine (``sacsma.sma_learned``) against the torch step it
    mirrors, one tracked run per physics (:data:`LEARNED_RUNS`): 32 rows, two years from the
    cold start, the flow and its six routed components within 1e-9 mm/day
``product``
    the share model applied to the tier-2 pass kept beside each forcing's series
    (``artifacts/product/calsim3/<forcing>/tier2/``) reproduces the tracked
    ``rim_inflow_monthly.csv``; the products of the calibrated models
    (``artifacts/product/{callite,15cdec}/<forcing>/``, :mod:`sacsma.product`) are written again
    and match the tracked files to one unit in the last printed digit

``--quick`` runs the first four, which need no model run.  The exit code is the number of
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

from . import paths

#: one watershed per domain for the parity check
PARITY_BASINS = (("15cdec", "BND"), ("9unimp", "CacheCreek"), ("11obs", "SHA"),
                 ("12rim", "SHAST"))
PARITY_KGE = 0.9999
PARITY_MAX_MM = 0.1
#: one tracked run per physics of the learned step: Noah-lite with the SAC exchanges and a
#: learned PXTEMP, Noah-lite, SAC ET with the refined Priestley-Taylor PET, SAC ET with Hamon
LEARNED_RUNS = (("multifamily",
                 "noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_w2ft15r10_aef"),
                ("15cdec", "noah"), ("15cdec", "pt"), ("15cdec", "hamon_dense"))
LEARNED_MAX_MM = 1e-9
#: relative tolerance of the product check (the fit and ``apply`` agree to about 3e-7)
PRODUCT_RTOL = 1e-5

CHECKS = ("imports", "cli", "links", "artifacts", "parity", "learned", "product")
QUICK = ("imports", "cli", "links", "artifacts")
#: the tracked parts of the output tree; ``_local/`` is ignored
ARTIFACT_PARTS = ("product", "models", "results")


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
_HTML_ID = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"", re.I)


def _slug(heading: str) -> str:
    """GitHub's anchor of a heading: lower case, punctuation dropped, spaces to hyphens."""
    h = re.sub(r"`", "", heading.strip().lower())
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h)          # a link keeps its text
    h = re.sub(r"[^\w\- ]", "", h, flags=re.UNICODE)
    return h.replace(" ", "-")


def _anchors(md: Path) -> set[str]:
    """Heading anchors of a markdown file plus the explicit ``<a id="...">`` ones."""
    text = _FENCE.sub("", md.read_text(encoding="utf-8", errors="replace"))
    out: set[str] = {a.lower() for a in _HTML_ID.findall(text)}
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


# ----------------------------------------------------------------------------- artifacts
def _git_files(root: Path, *args: str) -> list[str]:
    r = subprocess.run(["git", "-C", str(root), "ls-files", *args, "--", "artifacts"],
                       capture_output=True, text=True, check=True)
    return [p for p in r.stdout.splitlines() if p]


def check_artifacts(root: str | Path = ".", **_) -> tuple[bool | None, str]:
    root = Path(root).resolve()
    try:
        tracked = _git_files(root)
        stray = _git_files(root, "--others", "--exclude-standard")
    except (OSError, subprocess.CalledProcessError):
        return None, "not a git checkout"
    bad = []
    parts = {Path(f).parts[1] if len(Path(f).parts) > 2 else "" for f in tracked}
    for f in tracked:
        q = Path(f).parts
        if f != "artifacts/README.md" and (len(q) < 3 or q[1] not in ARTIFACT_PARTS):
            bad.append(f"tracked outside {', '.join(ARTIFACT_PARTS)}: {f}")
    bad += [f"no README.md in artifacts/{d}/" for d in ARTIFACT_PARTS
            if d in parts and f"artifacts/{d}/README.md" not in tracked]
    bad += [f"untracked in a tracked folder: {f}" for f in stray]
    runs: dict[tuple[str, str], set[str]] = {}
    for f in tracked:
        q = Path(f).parts
        if (len(q) > 5 and q[1] in ("models", "results") and q[2] == "dpl"
                and q[4] not in ("benchmark", "studies")):
            runs.setdefault((q[3], q[4]), set()).add(q[1])
    bad += [f"{g}/{r} has only a {next(iter(roles))} folder"
            for (g, r), roles in sorted(runs.items()) if roles != {"models", "results"}]
    rows = [f"\n    {b}" for b in bad[:20]]
    if len(bad) > 20:
        rows.append(f"\n    ... and {len(bad) - 20} more")
    return not bad, (f"{len(tracked)} tracked files, {len(runs)} learned-parameter runs with "
                     "a model and a results folder" + "".join(rows))


# -------------------------------------------------------------------------------- parity
def check_parity(data_dir: str = "data", **_) -> tuple[bool, str]:
    import numpy as np
    import pandas as pd

    from .io import load_reference
    from .metrics import kge
    from .model import run_basins

    ok, rows = True, []
    sims = {d: run_basins([b for d2, b in PARITY_BASINS if d2 == d], data_dir=data_dir, domain=d)
            for d in dict.fromkeys(d for d, _ in PARITY_BASINS)}
    for domain, basin in PARITY_BASINS:
        s = sims[domain][basin]
        ref = load_reference(data_dir, basin, domain=domain)
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


# ------------------------------------------------------------------------------- learned
def check_learned(data_dir: str = "data", artifacts_dir: str = "artifacts",
                  **_) -> tuple[bool | None, str]:
    if not _has_torch():
        return None, "torch is not installed"
    import numpy as np
    import torch

    from .dpl.evaluate import load_net_from_checkpoint, simulate_field
    from .dpl.forward import initial_state, routing_uh, run_window
    from .engine import OUTPUTS

    ok, rows, T = True, [], 730
    for group, run in LEARNED_RUNS:
        ck = paths.dpl_checkpoint(artifacts_dir, run, group)
        net, x, dom, cfg, _ = load_net_from_checkpoint(ck, data_dir, device="cpu")
        pick = np.unique(np.linspace(0, dom.n_hru - 1, 32).astype(int))
        W = np.zeros((len(pick), dom.n_hru))
        W[np.arange(len(pick)), pick] = 1.0
        got = simulate_field(net, x, dom, cfg, dom.dates[0], dom.dates[T - 1], W, spinup="cold",
                             outputs=OUTPUTS)
        rt = torch.as_tensor(pick)
        with torch.no_grad():
            p = {k: v[rt] for k, v in net(x).items()}
            res = run_window(
                dom.window(0, T).rows(rt), dom.lat_rad[rt], dom.elev[rt], p,
                routing_uh(p, dom.flowlen[rt]), initial_state(len(pick), dom.device, dom.dtype),
                cfg.physics(), veg_frac=None if dom.veg_frac is None else dom.veg_frac[rt],
                return_components=True, return_parts=True)
        ref = dict(zip(OUTPUTS, [res[0], *res[2], *res[3]], strict=True))
        d = max(float(np.abs(got[o] - ref[o].numpy()).max()) for o in OUTPUTS)
        good = d < LEARNED_MAX_MM
        ok &= good
        rows.append(f"\n    {group}/{run[:40]:40s} max |d| {d:.1e} mm/day"
                    + ("" if good else "   FAIL"))
    return ok, (f"the engine's learned step against torch within {LEARNED_MAX_MM:g} mm/day"
                + "".join(rows))


# ------------------------------------------------------------------------------- product
def _calsim3_rows(data_dir: str, artifacts_dir: str) -> list[tuple[bool | None, str]]:
    """The share model applied to each forcing's tier-2 pass against its tracked series."""
    root = paths.product(artifacts_dir, app="calsim3")
    series = sorted(root.glob("*/rim_inflow_monthly.csv"))
    if not (root / "share_model.pt").exists() or not series:
        return [(None, f"calsim3: no rim-inflow product under {root}")]
    if not _has_torch():
        return [(None, "calsim3: torch is not installed")]
    import numpy as np
    import pandas as pd
    import torch

    from .dpl.calsim.product import PASS_FILES, apply

    model = torch.load(root / "share_model.pt", weights_only=False)
    rows = []
    for csv in series:
        tier2 = csv.parent / "tier2"
        if not all((tier2 / name).exists() for name in PASS_FILES):
            rows.append((None, f"calsim3/{csv.parent.name}: skipped, {tier2} lacks the pass "
                               f"({', '.join(PASS_FILES)})"))
            continue
        ref = pd.read_csv(csv)
        per = pd.PeriodIndex(ref.month.unique(), freq="M")
        wy = per.year + (per.month >= 10)
        got = apply(model, tier2, data_dir, wy=(int(wy.min()), int(wy.max())))
        m = ref.merge(got, on=["arc", "month"], suffixes=("_ref", ""))
        rel = float((np.abs(m.taf - m.taf_ref) / np.maximum(np.abs(m.taf_ref), 1e-6)).max())
        good = len(m) == len(ref) == len(got) and rel < PRODUCT_RTOL
        rows.append((good, f"calsim3/{csv.parent.name}: {len(m)} arc-months, max relative "
                           f"difference {rel:.1e}" + ("" if good else "   FAIL")))
    return rows


def _last_digits(ref: str, new: str) -> float:
    """Largest difference between two tables of one writer, in units of the last printed
    decimal of each number (``inf`` when a line, a word or the shape differs)."""
    a, b = ref.splitlines(), new.splitlines()
    if len(a) != len(b):
        return float("inf")
    worst = 0.0
    for x, y in zip(a, b, strict=True):
        if x == y:
            continue
        u, v = x.split(","), y.split(",")
        if len(u) != len(v):
            return float("inf")
        for s, t in zip(u, v, strict=True):
            if s == t:
                continue
            try:
                d = abs(float(s) - float(t))
            except ValueError:
                return float("inf")
            dec = max(len(s.strip().partition(".")[2]), len(t.strip().partition(".")[2]))
            worst = max(worst, d * 10 ** dec)
    return worst


def _calibrated_rows(data_dir: str, artifacts_dir: str) -> list[tuple[bool | None, str]]:
    """Each tracked product of the calibrated models written again and compared."""
    from . import product

    rows = []
    for app, forcings in product.FORCINGS.items():
        for forcing in forcings:
            out = paths.product(artifacts_dir, forcing=forcing, app=app)
            if not out.is_dir():
                rows.append((None, f"{app}/{forcing}: not written"))
                continue
            worst = 0.0
            for name, text in product.files(app, forcing, data_dir).items():
                ref = out / name
                worst = max(worst, _last_digits(ref.read_text(), text) if ref.exists()
                            else float("inf"))
            good = worst <= 1.0
            rows.append((good, f"{app}/{forcing}: largest difference {worst:g} in the last "
                               "printed digit" + ("" if good else "   FAIL")))
    return rows


def check_product(data_dir: str = "data", artifacts_dir: str = "artifacts",
                  **_) -> tuple[bool | None, str]:
    rows = _calsim3_rows(data_dir, artifacts_dir) + _calibrated_rows(data_dir, artifacts_dir)
    done = [good for good, _ in rows if good is not None]
    return ((all(done) if done else None),
            "every tracked product is written again" + "".join(f"\n    {r}" for _, r in rows))


_FUNCS = {"imports": check_imports, "cli": check_cli, "links": check_links,
          "artifacts": check_artifacts, "parity": check_parity, "learned": check_learned,
          "product": check_product}


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
