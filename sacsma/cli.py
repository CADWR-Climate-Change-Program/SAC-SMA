"""Command-line interface: ``sacsma run <BASIN>``.

Forward-simulate a CDEC basin from the archived GA optimum and (optionally)
write the gauge flow to CSV.  Calibration subcommands will be added with the
GA milestone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import forcing_name, load_hru_table

CDEC15 = [
    "SHA", "BND", "ORO", "YRS", "FOL", "MKM", "NHG", "NML",
    "TLG", "MRC", "MIL", "PNF", "TRM", "SCC", "ISB",
]
#: selectable modeling domains (calibration sets).
DOMAINS = ["15cdec", "9unimp", "11obs", "12rim"]


def _run(args: argparse.Namespace) -> int:
    from .model import load_domain_forcing, run_basin

    domain = args.domain
    if args.basin.upper() == "ALL":
        # basin codes are domain-specific; read them from the HRU table
        basins = sorted(load_hru_table(args.data_dir, domain=domain)["basin"].unique())
    elif domain == "15cdec":
        basins = [args.basin.upper()]
    else:
        basins = [args.basin]  # CalLite basin codes are case-sensitive (CamelCase / mixed)

    # For multi-basin native runs, read the ~900 MB/var forcing store ONCE and
    # reuse it across every basin instead of re-reading it per basin.
    forcing = None
    if (
        len(basins) > 1
        and args.data_dir is not None
        and (Path(args.data_dir) / "forcing" / forcing_name(domain)).exists()
    ):
        print("loading domain forcing once for all basins...", flush=True)
        forcing = load_domain_forcing(args.data_dir, domain=domain, start=args.start, end=args.end)

    for basin in basins:
        df = run_basin(
            basin,
            data_dir=args.data_dir,
            domain=domain,
            start=args.start,
            end=args.end,
            progress=args.progress,
            forcing=forcing,
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


def _calsim(args: argparse.Namespace) -> int:
    from .compare import DEFAULT_CALSETS, make_all

    sets = tuple(args.sets) if args.sets else DEFAULT_CALSETS
    make_all(args.data_dir, args.artifacts_dir, args.run, sets,
             covered_frac=getattr(args, "covered_frac", None))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sacsma", description="SAC-SMA forward simulation")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="forward-simulate a basin (or ALL) for a domain")
    run.add_argument("basin", help="basin ID (e.g. BND, or BearRiver for 9unimp) or ALL")
    run.add_argument(
        "--domain", default="15cdec", choices=DOMAINS,
        help="calibration set / forcing store (default: 15cdec)",
    )
    run.add_argument(
        "--data-dir", default="data",
        help="organized data/ store to read (default: data)",
    )
    run.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    run.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    run.add_argument("--out", default=None, help="output CSV path")
    run.add_argument("--progress", action="store_true", help="print HRU progress")
    run.set_defaults(func=_run)

    cs = sub.add_parser(
        "calsim",
        help="cross-compare CalSim3 (actual) vs VIC vs multi-set SAC-SMA -> artifacts/<run>/",
    )
    cs.add_argument("--data-dir", default="data", help="organized data/ store")
    cs.add_argument("--artifacts-dir", default="artifacts", help="output root")
    cs.add_argument("--run", default="calsim", help="run name -> artifacts/<run>/")
    cs.add_argument("--sets", nargs="+", default=None,
                    help="SAC-SMA calibration sets to score separately vs CalSim3 "
                         "(default: 15cdec 9unimp 11obs)")
    cs.add_argument("--covered-frac", type=float, default=None,
                    help="informational 'covered'/'partial' status label only "
                         "(default: calsim.COVERED_FRAC); inclusion is crosswalk-driven")
    cs.set_defaults(func=_calsim)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
