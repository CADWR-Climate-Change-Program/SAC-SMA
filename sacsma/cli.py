"""Command-line interface: ``sacsma run <BASIN>``.

Forward-simulate a CDEC basin from the archived GA optimum and (optionally)
write the gauge flow to CSV.  Calibration subcommands will be added with the
GA milestone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import FORCING_NAME

CDEC15 = [
    "SHA", "BND", "ORO", "YRS", "FOL", "MKM", "NHG", "NML",
    "TLG", "MRC", "MIL", "PNF", "TRM", "SCC", "ISB",
]


def _run(args: argparse.Namespace) -> int:
    from .model import load_domain_forcing, run_basin

    basins = CDEC15 if args.basin.upper() == "ALL" else [args.basin.upper()]

    # For multi-basin native runs, read the ~900 MB/var forcing store ONCE and
    # reuse it across every basin instead of re-reading it per basin.
    forcing = None
    if (
        len(basins) > 1
        and args.data_dir is not None
        and (Path(args.data_dir) / "forcing" / FORCING_NAME).exists()
    ):
        print("loading domain forcing once for all basins...", flush=True)
        forcing = load_domain_forcing(args.data_dir, start=args.start, end=args.end)

    for basin in basins:
        df = run_basin(
            basin,
            data_dir=args.data_dir,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sacsma", description="SAC-SMA forward simulation")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="forward-simulate a CDEC basin (or ALL)")
    run.add_argument("basin", help="CDEC ID (e.g. BND) or ALL")
    run.add_argument(
        "--data-dir", default="data",
        help="organized data/ store to read (default: data)",
    )
    run.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    run.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    run.add_argument("--out", default=None, help="output CSV path")
    run.add_argument("--progress", action="store_true", help="print HRU progress")
    run.set_defaults(func=_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
