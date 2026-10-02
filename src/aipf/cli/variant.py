"""``--variant``: build, train or diagnose a model a system declares beside its production one."""
from __future__ import annotations

import argparse
import sys


def add_argument(p: argparse.ArgumentParser) -> None:
    """The ``--variant NAME`` flag; absent means the system's production model."""
    p.add_argument("--variant", default=None, metavar="NAME",
                   help="one of the system's declared variants (System.variants), whose "
                        "functional, mobility and checkpoint replace the production ones; "
                        "omitted: the production model")


def load_system(args: argparse.Namespace):
    """``load(args.system)``, as variant ``args.variant`` when one is named; ``None`` after printing a refusal."""
    from aipf.system import load

    system = load(args.system)
    if args.variant is None:
        return system
    try:
        return system.variant(args.variant)
    except KeyError as refused:
        print(f"--variant: {refused.args[0]}", file=sys.stderr)
        return None
