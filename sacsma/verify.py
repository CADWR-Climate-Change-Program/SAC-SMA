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
``bcm``
    the port of the BCM routing workbook (:mod:`sacsma.benchmark.bcm_routing`) reproduces the
    cached flow of each of its 11 calibration sheets (``bcm_routing_check.csv``) within
    :data:`BCM_RTOL`, and the sheet's r2, NSE and percent bias
``parity``
    the Python model against the archived MATLAB simulation, every watershed of the four
    calibrated domains on the forcing MATLAB ran on (:func:`sacsma.model.load_matlab_forcing`):
    KGE > 0.9999 and max daily difference < 0.1 mm/day
``learned``
    the learned model's step on the CPU engine (``sacsma.sma_learned``) against the torch step it
    mirrors, one tracked run per physics (:data:`LEARNED_RUNS`): 32 rows, two years from the
    cold start, the flow and its six routed components within 1e-9 mm/day (with the deep
    store's outflow and content where the run has it, and there the trainer's window spin-up,
    the store brought to its steady state, against the engine's); on each run the same two
    years split in two, the second year from the state the first left (``spinup="state"``),
    bitwise equal to them unsplit; then, on the first run without the store, the store at a
    zero share bitwise equal to the field without it
``product``
    the share model applied to the tier-2 pass kept beside each forcing's series
    (``artifacts/product/calsim3/<forcing>/tier2/``) reproduces the tracked
    ``rim_inflow_monthly.csv``; the products of the calibrated models
    (``artifacts/product/{callite,15cdec}/<forcing>/``, :mod:`sacsma.product`) are written again
    and match the tracked files to one unit in the last printed digit

``--quick`` runs the first five, which need no model run.  The exit code is the number of
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

#: the domains of the parity check, every watershed of each
PARITY_DOMAINS = (paths.CDEC15, *paths.CALLITE)
PARITY_KGE = 0.9999
PARITY_MAX_MM = 0.1
#: one tracked run per physics of the learned step: Noah-lite with the SAC exchanges, a
#: learned PXTEMP and the deep store (dPL-CalSim), Noah-lite with the SAC exchanges, SAC ET
#: with the Priestley-Taylor PET, SAC ET with Hamon on the HRUs (the ladder's rungs,
#: docs/runs.md)
LEARNED_RUNS = (("multifamily", "8_calsim"), ("15cdec", "4_noah"), ("15cdec", "3_pt"),
                ("15cdec", "1_hru"))
LEARNED_MAX_MM = 1e-9
#: relative tolerance of the product check (the fit and ``apply`` agree to about 3e-7)
PRODUCT_RTOL = 1e-5
#: relative tolerance of the BCM routing port against the workbook's cached flow (it agrees to
#: about 4e-14); the sheet statistics are held to :data:`BCM_STAT_TOL`
BCM_RTOL = 1e-9
BCM_STAT_TOL = 1e-9

CHECKS = ("imports", "cli", "links", "artifacts", "bcm", "parity", "learned", "product")
QUICK = ("imports", "cli", "links", "artifacts", "bcm")
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
                and q[3] in (paths.CDEC15, paths.MULTIFAMILY)):
            runs.setdefault((q[3], q[4]), set()).add(q[1])
    bad += [f"{g}/{r} has only a {next(iter(roles))} folder"
            for (g, r), roles in sorted(runs.items()) if roles != {"models", "results"}]
    rows = [f"\n    {b}" for b in bad[:20]]
    if len(bad) > 20:
        rows.append(f"\n    ... and {len(bad) - 20} more")
    return not bad, (f"{len(tracked)} tracked files, {len(runs)} learned-parameter runs with "
                     "a model and a results folder" + "".join(rows))


# ----------------------------------------------------------------------------------- bcm
def check_bcm(data_dir: str = "data", **_) -> tuple[bool, str]:
    from .benchmark.bcm_routing import check

    c = check(data_dir)
    stat = max(float((c[k] - c[f"{k}_sheet"]).abs().max()) for k in ("r2", "nse", "pbias"))
    q = float(c["max_rel_err_q"].max())
    ok = q <= BCM_RTOL and stat <= BCM_STAT_TOL
    return ok, (f"{len(c)} routing sheets, {int(c['months'].sum())} months: largest relative "
                f"difference in flow {q:.1e}, in the sheet statistics {stat:.1e}"
                + ("" if ok else "   FAIL"))


# -------------------------------------------------------------------------------- parity
def check_parity(data_dir: str = "data", **_) -> tuple[bool, str]:
    import numpy as np
    import pandas as pd

    from .io import load_reference
    from .metrics import kge
    from .model import load_matlab_forcing, run_basins

    ok, rows = True, []
    for domain in PARITY_DOMAINS:
        sim = run_basins(data_dir=data_dir, domain=domain,
                         forcing=load_matlab_forcing(data_dir, domain=domain))
        worst = None
        for basin, ref in load_reference(data_dir, domain=domain).groupby("basin"):
            r = pd.Series(ref["flow"].to_numpy(), index=pd.DatetimeIndex(ref["date"]))
            s = sim[basin]
            j = s.index.intersection(r.index)
            k = kge(s[j].to_numpy(), r[j].to_numpy())
            d = float(np.abs(s[j].to_numpy() - r[j].to_numpy()).max())
            good = k > PARITY_KGE and d < PARITY_MAX_MM
            ok &= good
            if not good:
                rows.append(f"\n    {domain:7s} {basin:15s} KGE {k:.6f}  "
                            f"max |d| {d:.4f} mm/day   FAIL")
            if worst is None or d > worst[2]:
                worst = (basin, k, d)
        rows.append(f"\n    {domain:7s} {sim.shape[1]:2d} watersheds, largest at {worst[0]}: "
                    f"KGE {worst[1]:.6f}  max |d| {worst[2]:.4f} mm/day")
    return ok, (f"KGE > {PARITY_KGE} and max |d| < {PARITY_MAX_MM} mm/day against the MATLAB "
                "simulation, every watershed, on the forcing it ran on" + "".join(rows))


# ------------------------------------------------------------------------------- learned
def _learned_rows(dom):
    """32 rows spread over the domain, and the weights that pick each alone."""
    import numpy as np

    pick = np.unique(np.linspace(0, dom.n_hru - 1, 32).astype(int))
    W = np.zeros((len(pick), dom.n_hru))
    W[np.arange(len(pick)), pick] = 1.0
    return pick, W


def _learned_diff(net, x, dom, cfg, outputs, T: int = 730) -> float:
    """The largest |difference| over ``outputs`` between the engine's learned step and the
    torch step on :func:`_learned_rows`, ``T`` days from the cold start."""
    import numpy as np
    import torch

    from .dpl.evaluate import simulate_field
    from .dpl.forward import initial_state, routing_uh, run_window

    pick, W = _learned_rows(dom)
    got = simulate_field(net, x, dom, cfg, dom.dates[0], dom.dates[T - 1], W, spinup="cold",
                         outputs=outputs)
    rt = torch.as_tensor(pick)
    with torch.no_grad():
        p = {k: v[rt] for k, v in net(x).items()}
        res = run_window(
            dom.window(0, T).rows(rt), dom.lat_rad[rt], dom.elev[rt], p,
            routing_uh(p, dom.flowlen[rt]), initial_state(len(pick), dom.device, dom.dtype),
            cfg.physics(), veg_frac=None if dom.veg_frac is None else dom.veg_frac[rt],
            return_components=True, return_parts=True)
    ref = dict(zip(outputs, [res[0], *res[2], *res[3]], strict=True))
    return max(float(np.abs(got[o] - ref[o].numpy()).max()) for o in outputs)


def _window_diff(net, x, dom, cfg, years: int = 3, T: int = 365) -> float:
    """The trainer's window spin-up against the engine's (``spinup="window"``), the deep
    store brought to its steady state in both: :func:`sacsma.dpl.train._stream_nograd` from
    the cold start over ``years`` years, :func:`sacsma.dpl.train._steady_fill`, then ``T``
    days; the largest |difference| in flow on :func:`_learned_rows`."""
    import dataclasses

    import numpy as np
    import pandas as pd
    import torch

    from .dpl.evaluate import simulate_field
    from .dpl.forward import initial_state, routing_uh
    from .dpl.train import _steady_fill, _stream_nograd

    pick, W = _learned_rows(dom)
    rt = torch.as_tensor(pick)
    t0 = int(dom.dates.searchsorted(dom.dates[0] + pd.DateOffset(years=years)))
    cfgw = dataclasses.replace(cfg, spinup_start=str(dom.dates[0].date()))
    got = simulate_field(net, x, dom, cfgw, dom.dates[t0], dom.dates[t0 + T - 1], W,
                         spinup="window")
    sub = dataclasses.replace(
        dom, hrus=dom.hrus.iloc[pick].reset_index(drop=True), cell_idx=dom.cell_idx[pick],
        lat_rad=dom.lat_rad[rt], elev=dom.elev[rt], flowlen=dom.flowlen[rt],
        veg_frac=None if dom.veg_frac is None else dom.veg_frac[rt],
        W=torch.eye(len(pick), dtype=dom.dtype), basins=tuple(f"r{i}" for i in pick))
    with torch.no_grad():
        p = {k: v[rt] for k, v in net(x).items()}
        uh = routing_uh(p, sub.flowlen)
        _, st = _stream_nograd(sub, cfgw, p, uh, 0, t0,
                               initial_state(len(pick), dom.device, dom.dtype))
        st.sac.gwc = _steady_fill(st.sac.gwc, p["kdeep"], t0)
        sim, _ = _stream_nograd(sub, cfgw, p, uh, t0, t0 + T, st, collect=True)
    return float(np.abs(got["flow"] - sim.numpy()).max())


def _handoff_same(net, x, dom, cfg, outputs, T: int = 730) -> bool:
    """``T`` days from the cold start on :func:`_learned_rows`, split in two (the second half
    from the state the first kept), bitwise equal to the run unsplit over ``outputs``."""
    import numpy as np

    from .dpl.evaluate import simulate_field

    _pick, W = _learned_rows(dom)
    h = T // 2
    run = lambda t0, t1, **kw: simulate_field(  # noqa: E731
        net, x, dom, cfg, dom.dates[t0], dom.dates[t1 - 1], W, outputs=outputs, **kw)
    whole = run(0, T, spinup="cold")
    first = run(0, h, spinup="cold", keep_state=True)
    second = run(h, T, spinup="state", state=first["state"])
    return all(np.array_equal(np.hstack([first[o], second[o]]), whole[o]) for o in outputs)


def check_learned(data_dir: str = "data", artifacts_dir: str = "artifacts",
                  **_) -> tuple[bool | None, str]:
    if not _has_torch():
        return None, "torch is not installed"
    import dataclasses

    import numpy as np

    from .dpl.evaluate import learned_field, load_net_from_checkpoint
    from .engine import DEEP_OUTPUTS, OUTPUTS, simulate

    ok, rows, zero_share = True, [], False
    for group, run in LEARNED_RUNS:
        ck = paths.dpl_checkpoint(artifacts_dir, run, group)
        net, x, dom, cfg, _raw = load_net_from_checkpoint(ck, data_dir, device="cpu")
        deep = cfg.deep_store != "none"
        outs = (*OUTPUTS, *DEEP_OUTPUTS) if deep else OUTPUTS
        d = _learned_diff(net, x, dom, cfg, outs)
        good = d < LEARNED_MAX_MM
        ok &= good
        rows.append(f"\n    {group}/{run[:40]:40s} max |d| {d:.1e} mm/day"
                    + (" (with the deep store)" if deep else "") + ("" if good else "   FAIL"))
        same = _handoff_same(net, x, dom, cfg, outs)
        ok &= same
        rows.append(f"\n    {'  + split in two, from the state kept':46s} "
                    + ("bitwise" if same else "DIFFERENT   FAIL"))
        if deep:
            w = _window_diff(net, x, dom, cfg)
            good = w < LEARNED_MAX_MM
            ok &= good
            rows.append(f"\n    {'  + the window spin-up, trainer against engine':46s} "
                        f"max |d| {w:.1e} mm/day" + ("" if good else "   FAIL"))
        if deep or zero_share:
            continue
        zero_share = True
        # the store at gdeep 0 against the field without it, bitwise, through the engine's
        # window spin-up (the run split at its start)
        cfgd = dataclasses.replace(cfg, deep_store="cascade")
        f0 = learned_field(net, x, dom, cfg)
        fz = dataclasses.replace(f0, physics=cfgd.physics(), deep=np.column_stack(
            [np.zeros(len(f0.cell)), np.full(len(f0.cell), 5e-4)]))
        forcing = dataclasses.replace(dom.forcing, tmin=dom.tmin, tmax=dom.tmax)
        _pick, W = _learned_rows(dom)
        a, b = (simulate(f, forcing, dom.dates[365], dom.dates[365 + 729], W, spinup="window",
                         window_start=dom.dates[0], outputs=OUTPUTS)    # a 365-day spin-up
                for f in (f0, fz))
        same = all(np.array_equal(a[o], b[o]) for o in OUTPUTS)
        ok &= same
        rows.append(f"\n    {'  + the deep store at gdeep 0 == without it':46s} "
                    + ("bitwise" if same else "DIFFERENT   FAIL"))
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
          "artifacts": check_artifacts, "bcm": check_bcm, "parity": check_parity,
          "learned": check_learned, "product": check_product}


def run_checks(checks=None, *, data_dir: str = "data", quick: bool = False) -> int:
    """Run the named checks (default: all, or the quick ones); print one line each;
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
