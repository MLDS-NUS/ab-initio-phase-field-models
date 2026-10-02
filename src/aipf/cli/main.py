"""The ``aipf`` console entry point.

Subcommands are registered here and implemented in sibling modules. Every
subcommand takes ``--system``, whose defaults come from that system's SYSTEM
object, so naming a system is enough to reproduce its published settings.
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aipf",
        description="Ab initio phase-field models.",
    )
    p.add_argument("--version", action="store_true",
                   help="print the package version and exit")
    sub = p.add_subparsers(dest="command", metavar="<command>")
    from aipf.cli import data_cmd, diagnose_cmd, modes_cmd, train_cmd
    data_cmd.add_parser(sub)
    train_cmd.add_parser(sub)
    diagnose_cmd.add_parser(sub)
    modes_cmd.add_parser(sub)
    __import__("aipf.cli.rollout_cmd", fromlist=["add_parser"]).add_parser(sub)
    __import__("aipf.cli.md_cmd", fromlist=["add_parser"]).add_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from aipf import __version__
        print(__version__)
        return 0
    if not args.command:
        parser.print_help()
        return 0
    from aipf.paths import ConfigError, MissingLocation
    from aipf.site import MissingSiteFact
    try:
        return args.func(args)
    except (MissingLocation, MissingSiteFact, ConfigError) as refused:
        # an undeclared location or site fact, or an unreadable aipf.toml: a refusal, not a crash
        print(f"aipf {args.command}: {refused}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
