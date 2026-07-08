"""Command-line interface: ``sacsma run | plots | calsim``.

``run`` forward-simulates a watershed from its archived GA optimum; ``plots``
writes the per-watershed calibration/validation diagnostics for a domain; and
``calsim`` runs the CalSim3-vs-VIC-vs-SAC-SMA cross-compare.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import calsim as _calsim_pkg
from . import cdec15 as _cdec15_pkg
from .io import DEFAULT_FORCING, forcing_path, load_hru_table

#: selectable modeling domains (calibration sets): the 15-CDEC application + CalLite sets.
DOMAINS = [_cdec15_pkg.DOMAIN, *_calsim_pkg.DOMAINS]


def _run(args: argparse.Namespace) -> int:
    from .model import load_domain_forcing, run_basin

    domain = args.domain
    if args.basin.upper() == "ALL":
        # basin codes are domain-specific; read them from the HRU table
        basins = sorted(load_hru_table(args.data_dir, domain=domain)["basin"].unique())
    elif domain == _cdec15_pkg.DOMAIN:
        basins = [args.basin.upper()]
    else:
        basins = [args.basin]  # CalLite basin codes are case-sensitive (CamelCase / mixed)

    # For multi-basin native runs, read the ~900 MB/var forcing store ONCE and
    # reuse it across every basin instead of re-reading it per basin.
    product = args.forcing or DEFAULT_FORCING
    forcing = None
    if (
        len(basins) > 1
        and args.data_dir is not None
        and forcing_path(args.data_dir, domain, product).exists()
    ):
        print(f"loading domain forcing once for all basins ({product})...", flush=True)
        forcing = load_domain_forcing(args.data_dir, domain=domain, start=args.start,
                                      end=args.end, product=product)

    for basin in basins:
        df = run_basin(
            basin,
            data_dir=args.data_dir,
            domain=domain,
            start=args.start,
            end=args.end,
            progress=args.progress,
            forcing=forcing,
            parallel=args.parallel,
            product=product,
        )
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
                   artifacts_dir=args.artifacts_dir, run=args.run or "cdec15")
        if args.fnf_check:
            from .calsim.plots import make_cdec15_fnf_check

            make_cdec15_fnf_check(basins=args.basins, data_dir=args.data_dir,
                                  artifacts_dir=args.artifacts_dir, run=args.run or "cdec15")
    else:
        from .calsim import plots as p

        p.make_all(domain=args.domain, basins=args.basins, data_dir=args.data_dir,
                   artifacts_dir=args.artifacts_dir, run=args.run)
    return 0


def _calsim(args: argparse.Namespace) -> int:
    from .calsim.compare import DEFAULT_CALSETS, make_all

    sets = tuple(args.sets) if args.sets else DEFAULT_CALSETS
    make_all(args.data_dir, args.artifacts_dir, args.run, sets,
             covered_frac=getattr(args, "covered_frac", None), parallel=args.parallel)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sacsma", description="Distributed SAC-SMA for CA watersheds")
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
                          "for the WGEN historical-parallel sequence (CalSim domains "
                          "only; default: the historical Livneh-unsplit store)")
    run.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    run.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    run.add_argument("--out", default=None, help="output CSV path")
    run.add_argument("--progress", action="store_true", help="print HRU progress")
    run.add_argument("--parallel", action="store_true",
                     help="fan HRUs across cores (Numba prange); ~8x faster, matches "
                          "the serial result to floating tolerance")
    run.set_defaults(func=_run)

    pl = sub.add_parser("plots", help="per-watershed cal/val diagnostic figures for a domain")
    pl.add_argument("--domain", default=_cdec15_pkg.DOMAIN, choices=DOMAINS,
                    help="calibration set (default: 15cdec)")
    pl.add_argument("--basins", nargs="*", default=None,
                    help="subset of watershed codes (default: all)")
    pl.add_argument("--data-dir", default="data", help="data store")
    pl.add_argument("--artifacts-dir", default="artifacts", help="output root")
    pl.add_argument("--run", default=None,
                    help="run name (default: cdec15 -> artifacts/cdec15/, or the domain "
                         "-> artifacts/calsim/<domain>/)")
    pl.add_argument("--fnf-check", action="store_true",
                    help="15cdec only: also score the same basins MONTHLY against CalSim3's "
                         "unimpaired FNF (longer independent validation window) -> extra "
                         "*_diagnostics_calsim3.png figures + metrics_15cdec_calsim3.csv")
    pl.set_defaults(func=_plots)

    cs = sub.add_parser(
        "calsim",
        help="cross-compare CalSim3 (actual) vs VIC vs multi-set SAC-SMA -> artifacts/calsim/<run>/",
    )
    cs.add_argument("--data-dir", default="data", help="organized data/ store")
    cs.add_argument("--artifacts-dir", default="artifacts", help="output root")
    cs.add_argument("--run", default="compare", help="run name -> artifacts/calsim/<run>/")
    cs.add_argument("--sets", nargs="+", default=None,
                    help="SAC-SMA calibration sets to score separately vs CalSim3 "
                         "(default: 15cdec 9unimp 11obs)")
    cs.add_argument("--covered-frac", type=float, default=None,
                    help="informational 'covered'/'partial' status label only "
                         "(default: catchments.COVERED_FRAC); inclusion is crosswalk-driven")
    cs.add_argument("--parallel", action="store_true",
                    help="fan the SAC-SMA model runs across cores (Numba prange); "
                         "results unchanged, ~8x faster on the model-run phase")
    cs.set_defaults(func=_calsim)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
