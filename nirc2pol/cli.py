"""The command line: ``nirc2pol-reduce`` and ``nirc2pol-combine``.

A thin wrapper. Everything the reduction does is in
:func:`nirc2pol.polmode.run`, and everything it *chooses* is in the TOML
config, so this file adds only argument parsing, log level, and the two
lines printed at the end.

    nirc2pol-reduce --template > my_night.toml   # every option and default
    nirc2pol-reduce my_night.toml                # run it

``nirc2pol-combine`` joins reductions that have already been run, by
median-combining their per-cycle Stokes cubes -- see
:mod:`nirc2pol.combine` for why that is the safe place to join two nights.

``--template`` matters more than it looks: an installed copy has no
repository to copy an example config out of, so this is where a config
comes from. Both commands have one.
"""

from __future__ import annotations

import argparse
import logging
import sys

from nirc2pol import __version__
from nirc2pol.combine import CombineConfig
from nirc2pol.combine import run as run_combine
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


def _configure_logging(log_level):
    """Set the *console* level, leaving the log file at INFO.

    ReductionLog attaches its own file handler at INFO and raises the root
    logger to match, so setting the root level here would be overruled a
    moment later -- and a quiet console should still leave a full record on
    disk. Set the console handler instead, and keep the root low enough to
    let its records through.
    """
    level = getattr(logging, log_level)
    logging.basicConfig(level=min(level, logging.INFO),
                        format="%(levelname)s %(message)s")
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)


def _load(config_cls, path, prog):
    """Read a config, or report the problem the way a command should.

    Returns
    -------
    config or int
        The config, or the exit status to return -- 2 for anything the user
        can fix by editing the file or the path.
    """
    try:
        return config_cls.from_toml(path)
    except FileNotFoundError:
        print(f"No config file at {path}. Write one with:\n"
              f"    {prog} --template > my_config.toml", file=sys.stderr)
        return 2
    except ValueError as exc:
        # The config classes validate on construction and say what is wrong
        # and why, so pass the message through rather than a traceback.
        print(f"{path}: {exc}", file=sys.stderr)
        return 2


def main(argv=None):
    """Run one reduction from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        Process exit status: 0 on success, 2 when the config is missing,
        will not validate, or points at data that is not there. Anything
        else the reduction raises is left to propagate -- a traceback says
        more than a summary would.
    """
    args = build_parser().parse_args(argv)

    if args.template:
        print(ReductionConfig.template().rstrip("\n"))
        return 0

    if args.config is None:
        build_parser().error(
            "a config file is required. Write one with:\n"
            "    nirc2pol-reduce --template > my_night.toml")

    _configure_logging(args.log_level)

    cfg = _load(ReductionConfig, args.config, "nirc2pol-reduce")
    if isinstance(cfg, int):
        return cfg

    try:
        products = run(cfg, config_path=args.config)
    except FileNotFoundError as exc:
        # Data that is not where the config says it is: the user's to fix,
        # and the message already explains it. Anything else propagates --
        # a traceback says more about a bug than a summary would.
        print(exc, file=sys.stderr)
        return 2

    writer = products["writer"]
    run_log = products["run_log"]
    print(f"Done: {len(products['stokes_cubes'])} HWP cycles -> Stokes "
          f"products in {writer.output_dir}")
    print(f"Log:  {run_log.path} ({run_log.warnings} warnings)")
    return 0


def build_combine_parser():
    """The argument parser for ``nirc2pol-combine``."""
    parser = argparse.ArgumentParser(
        prog="nirc2pol-combine",
        description="Combine reductions that have already been run, by "
                    "median-combining their per-cycle Stokes cubes.",
        epilog="Each input must have been reduced on its own, with its own "
               "darks, flats and beam geometry. That is what makes combining "
               "across nights safe, and it is why this is a separate step "
               "rather than one reduction over both nights' frames.")
    parser.add_argument(
        "config", nargs="?",
        help="TOML config for the combination, as written by --template.")
    parser.add_argument(
        "--template", action="store_true",
        help="Print a config listing every option with its default, then "
             "exit. Redirect it to a file and edit it.")
    parser.add_argument(
        "--log-level", default="INFO", choices=LOG_LEVELS, metavar="LEVEL",
        help=f"Console verbosity, one of {', '.join(LOG_LEVELS)} "
             "(default: INFO).")
    parser.add_argument(
        "--version", action="version", version=f"nirc2pol-dpp {__version__}")
    return parser


def main_combine(argv=None):
    """Combine several reductions from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0 on success, 2 when the config is missing, will not validate, or
        points at reductions whose per-cycle cubes are not there.
    """
    args = build_combine_parser().parse_args(argv)

    if args.template:
        print(CombineConfig.template().rstrip("\n"))
        return 0

    if args.config is None:
        build_combine_parser().error(
            "a config file is required. Write one with:\n"
            "    nirc2pol-combine --template > combined.toml")

    _configure_logging(args.log_level)

    cfg = _load(CombineConfig, args.config, "nirc2pol-combine")
    if isinstance(cfg, int):
        return cfg

    try:
        products = run_combine(cfg, config_path=args.config)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    writer, run_log = products["writer"], products["run_log"]
    print(f"Done: {len(products['cubes'])} HWP cycles from "
          f"{len(products['sources'])} reduction(s) -> {writer.output_dir}")
    print(f"Log:  {run_log.path} ({run_log.warnings} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
