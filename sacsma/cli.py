"""Command-line interface: ``sacsma run | plots | calsim | benchmark | product | verify | dpl``.

``run`` forward-simulates a watershed from its archived GA optimum; ``plots``
writes the per-watershed calibration/validation diagnostics for a domain;
``calsim`` runs the CalSim3-vs-VIC-vs-SAC-SMA cross-compare; ``benchmark`` scores three models
against observed CDEC full natural flow; ``product`` writes the products of the calibrated
models; ``verify`` checks the installation.  The ``dpl`` commands (differentiable parameter
learning) are defined in :mod:`sacsma.dpl.cli`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import calsim as _calsim_pkg
from . import cdec15 as _cdec15_pkg
from .io import DEFAULT_FORCING, load_hru_table

#: selectable modeling domains (calibration sets): the 15-CDEC application + CalLite sets.
DOMAINS = [_cdec15_pkg.DOMAIN, *_calsim_pkg.DOMAINS]


def _run(args: argparse.Namespace) -> int:
    from .model import run_basins

    domain = args.domain
    if args.basin.upper() == "ALL":
        # basin codes are domain-specific; read them from the HRU table
        basins = sorted(load_hru_table(args.data_dir, domain=domain)["basin"].unique())
    elif domain == _cdec15_pkg.DOMAIN:
        basins = [args.basin.upper()]
    else:
        basins = [args.basin]  # CalLite basin codes are case-sensitive (CamelCase / mixed)

    flow = run_basins(basins, data_dir=args.data_dir, domain=domain, start=args.start,
                      end=args.end, product=args.forcing or DEFAULT_FORCING)
    for basin in basins:
        df = flow[basin].rename("flow").reset_index()
        if args.out:
            out = Path(args.out)
            if len(basins) > 1:
                out = out.with_name(f"{out.stem}_{basin}{out.suffix or '.csv'}")
            df.to_csv(out, index=False)
            print(f"{basin}: wrote {len(df)} days -> {out}")
        else:
            print(f"{basin}: {len(df)} days, mean flow {df['flow'].mean():.4f} mm/day")
    return 0


def _plots(args: argparse.Namespace) -> int:
    if args.domain == _cdec15_pkg.DOMAIN:
        from .cdec15 import plots as p

        p.make_all(basins=args.basins, data_dir=args.data_dir,
                   artifacts_dir=args.artifacts_dir)
        if args.fnf_check:
            from .calsim.plots import make_cdec15_fnf_check

            make_cdec15_fnf_check(basins=args.basins, data_dir=args.data_dir,
                                  artifacts_dir=args.artifacts_dir)
    else:
        from .calsim import plots as p

        p.make_all(domain=args.domain, basins=args.basins, data_dir=args.data_dir,
                   artifacts_dir=args.artifacts_dir)
    return 0


def _calsim(args: argparse.Namespace) -> int:
    from .calsim.compare import DEFAULT_CALSETS, make_all

    if getattr(args, "forcing_compare", False):
        from .calsim.forcing_compare import make_all as fc_all
        fc_all(data_dir=args.data_dir, artifacts_dir=args.artifacts_dir)
        return 0
    sets = tuple(args.sets) if args.sets else DEFAULT_CALSETS
    make_all(args.data_dir, args.artifacts_dir, sets,
             covered_frac=getattr(args, "covered_frac", None))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .benchmark.report import make_all

    make_all(args.data_dir, args.artifacts_dir)
    return 0


def _product(args: argparse.Namespace) -> int:
    from .product import write

    write(args.app, args.forcing, args.data_dir, args.artifacts_dir)
    return 0


def _verify(args: argparse.Namespace) -> int:
    from .verify import run_checks

    return run_checks(args.checks, data_dir=args.data_dir, quick=args.quick)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sacsma", description="Distributed SAC-SMA for CA watersheds")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="forward-simulate a basin (or ALL) for a domain")
    run.add_argument("basin", help="basin ID (e.g. BND, or BearRiver for 9unimp) or ALL")
    run.add_argument(
        "--domain", default=_cdec15_pkg.DOMAIN, choices=DOMAINS,
        help="calibration set / forcing store (default: 15cdec)",
    )
    run.add_argument(
        "--data-dir", default="data",
        help="organized data/ store to read (default: data)",
    )
    run.add_argument("--forcing", default=None, metavar="PRODUCT",
                     help="forcing product (store filename stem), e.g. wgen_product_a "
                          "for the WGEN historical-parallel sequence, or "
                          "wgen_product_a_s12 for WGEN climate scenario 12 (CalSim "
                          "domains only; default: the historical Livneh-unsplit store)")
    run.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    run.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    run.add_argument("--out", default=None, help="output CSV path")
    run.set_defaults(func=_run)

    pl = sub.add_parser("plots", help="per-watershed cal/val diagnostic figures for a domain")
    pl.add_argument("--domain", default=_cdec15_pkg.DOMAIN, choices=DOMAINS,
                    help="calibration set (default: 15cdec)")
    pl.add_argument("--basins", nargs="*", default=None,
                    help="subset of watershed codes (default: all)")
    pl.add_argument("--data-dir", default="data", help="data store")
    pl.add_argument("--artifacts-dir", default="artifacts",
                    help="output root (-> <root>/results/15cdec/ or "
                         "<root>/results/callite/<domain>/)")
    pl.add_argument("--fnf-check", action="store_true",
                    help="15cdec only: also score the same basins MONTHLY against CalSim3's "
                         "unimpaired FNF (longer independent validation window) -> extra "
                         "*_diagnostics_calsim3.png figures + metrics_calsim3.csv")
    pl.set_defaults(func=_plots)

    cs = sub.add_parser(
        "calsim",
        help="cross-compare CalSim3 (actual) vs VIC vs multi-set SAC-SMA "
             "-> artifacts/results/calsim3/ and footprints/",
    )
    cs.add_argument("--data-dir", default="data", help="organized data/ store")
    cs.add_argument("--artifacts-dir", default="artifacts", help="output root")
    cs.add_argument("--sets", nargs="+", default=None,
                    help="SAC-SMA calibration sets to score separately vs CalSim3 "
                         "(default: 15cdec 9unimp 11obs)")
    cs.add_argument("--covered-frac", type=float, default=None,
                    help="informational 'covered'/'partial' status label only "
                         "(default: catchments.COVERED_FRAC); inclusion is crosswalk-driven")
    cs.add_argument("--forcing-compare", action="store_true",
                    help="instead of the standard cross-compare, draw the forcing-product "
                         "volume and regime figures -> artifacts/results/forcing/")
    cs.set_defaults(func=_calsim)

    bm = sub.add_parser(
        "benchmark",
        help="dPL-CalSim, BCM and VIC-CalSim3 against observed CDEC full natural flow at "
             "12 CDEC sites, monthly, WY1991-2018, on WGEN Product A scenario 1 "
             "-> artifacts/results/benchmark/",
    )
    bm.add_argument("--data-dir", default="data", help="organized data/ store")
    bm.add_argument("--artifacts-dir", default="artifacts", help="output root")
    bm.set_defaults(func=_benchmark)

    pr = sub.add_parser(
        "product",
        help="the products of the calibrated models: the three CalLite inflow files per "
             "forcing -> artifacts/product/callite/<forcing>/, or the daily flow of the 15 "
             "CDEC watersheds -> artifacts/product/15cdec/<forcing>/",
    )
    pr.add_argument("app", choices=("callite", "15cdec"))
    pr.add_argument("--forcing", nargs="+", default=None, metavar="PRODUCT",
                    help="forcings to write (default: every forcing the product carries: "
                         "callite historical_livneh_unsplit wgen_product_a wgen_product_a_s12, "
                         "15cdec historical_livneh_unsplit)")
    pr.add_argument("--data-dir", default="data", help="organized data/ store")
    pr.add_argument("--artifacts-dir", default="artifacts", help="output root")
    pr.set_defaults(func=_product)

    vf = sub.add_parser(
        "verify",
        help="check the installation: imports, commands, document links, the layout of "
             "artifacts/, the BCM routing port, parity with the MATLAB reference, the learned "
             "step against torch, and the tracked products",
    )
    vf.add_argument("checks", nargs="*", metavar="CHECK",
                    help="subset of: imports cli links artifacts bcm parity learned product "
                         "(default: all)")
    vf.add_argument("--quick", action="store_true",
                    help="only the checks that need no model run (imports, cli, links, "
                         "artifacts, bcm)")
    vf.add_argument("--data-dir", default="data", help="organized data/ store")
    vf.set_defaults(func=_verify)

    from .dpl.cli import register as register_dpl   # argument definitions only (no torch)

    register_dpl(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    from .dpl.cli import dispatch_tool

    rc = dispatch_tool(argv)    # `dpl calsim <tool> ...` goes to the tool's own parser
    if rc is not None:
        return rc
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
