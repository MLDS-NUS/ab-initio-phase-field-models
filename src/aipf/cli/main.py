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
    from aipf.system import UnknownSystem
    try:
        return args.func(args)
    except (MissingLocation, MissingSiteFact, ConfigError, UnknownSystem) as refused:
        # an undeclared location, site fact or system, or an unreadable aipf.toml: a refusal, not a
        # crash
        print(f"{_verb(args)}: {refused}", file=sys.stderr)
        return 2


#: The attributes that hold a command's verb or driver (``aipf md doctor``, ``aipf rollout slab``).
_VERB_ATTRIBUTES = ("md_command", "data_command", "driver")


def _verb(args: argparse.Namespace) -> str:
    """The command as it was typed, up to its verb: ``aipf md doctor``, ``aipf train``."""
    words = ["aipf", args.command]
    words += [getattr(args, name) for name in _VERB_ATTRIBUTES if getattr(args, name, None)]
    return " ".join(words)


if __name__ == "__main__":
    sys.exit(main())
