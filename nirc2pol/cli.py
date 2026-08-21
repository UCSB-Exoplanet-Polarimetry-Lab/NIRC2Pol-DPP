"""``nirc2pol-reduce``: the standard reduction, from the command line.

A thin wrapper. Everything the reduction does is in
:func:`nirc2pol.polmode.run`, and everything it *chooses* is in the TOML
config, so this file adds only argument parsing, log level, and the two
lines printed at the end.

    nirc2pol-reduce --template > my_night.toml   # every option and default
    nirc2pol-reduce my_night.toml                # run it

``--template`` matters more than it looks: an installed copy has no
repository to copy an example config out of, so this is where a config
comes from.
"""

from __future__ import annotations

import argparse
import logging
import sys

from nirc2pol import __version__
from nirc2pol.polmode import run
from nirc2pol.reduction.config import ReductionConfig

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def build_parser():
    """The argument parser, exposed so the options can be inspected.

    Returns
    -------
    argparse.ArgumentParser
        Parses ``config``, ``--template``, ``--log-level`` and ``--version``.
    """
    parser = argparse.ArgumentParser(
        prog="nirc2pol-reduce",
        description="Reduce one night of Keck/NIRC2-Pol data, raw frames to "
                    "Stokes products.",
        epilog="Every choice lives in the config file. Instrument constants "
               "-- plate scale, detector epochs -- do not: they are "
               "properties of the hardware and ship with the package.")
    parser.add_argument(
        "config", nargs="?",
        help="TOML config for this reduction, as written by --template.")
    parser.add_argument(
        "--template", action="store_true",
        help="Print a config listing every option with its default and "
             "allowed values, then exit. Redirect it to a file and edit it.")
    parser.add_argument(
        "--log-level", default="INFO", choices=LOG_LEVELS, metavar="LEVEL",
        help=f"Console verbosity, one of {', '.join(LOG_LEVELS)} "
             "(default: INFO). The reduction log written beside the products "
             "is unaffected.")
    parser.add_argument(
        "--version", action="version", version=f"nirc2pol-dpp {__version__}")
    return parser


def main(argv=None):
    """Run one reduction from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        Process exit status: 0 on success, 2 when the config is missing or
        will not validate. Anything the reduction itself raises is left to
        propagate -- a traceback says more than a summary would.
    """
    args = build_parser().parse_args(argv)

    if args.template:
        print(ReductionConfig.template().rstrip("\n"))
        return 0

    if args.config is None:
        build_parser().error(
            "a config file is required. Write one with:\n"
            "    nirc2pol-reduce --template > my_night.toml")

    # --log-level is the *console* level. ReductionLog attaches its own file
    # handler at INFO and raises the root logger to match, so setting the root
    # level here would be overruled a moment later -- and a quiet console
    # should still leave a full record on disk. Set the console handler
    # instead, and keep the root low enough to let its records through.
    level = getattr(logging, args.log_level)
    logging.basicConfig(level=min(level, logging.INFO),
                        format="%(levelname)s %(message)s")
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)

    try:
        cfg = ReductionConfig.from_toml(args.config)
    except FileNotFoundError:
        print(f"No config file at {args.config}. Write one with:\n"
              f"    nirc2pol-reduce --template > my_night.toml",
              file=sys.stderr)
        return 2
    except ValueError as exc:
        # ReductionConfig validates on construction and says what is wrong
        # and why, so pass its message through rather than a traceback.
        print(f"{args.config}: {exc}", file=sys.stderr)
        return 2

    products = run(cfg, config_path=args.config)

    writer = products["writer"]
    run_log = products["run_log"]
    print(f"Done: {len(products['stokes_cubes'])} HWP cycles -> Stokes "
          f"products in {writer.output_dir}")
    print(f"Log:  {run_log.path} ({run_log.warnings} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
