"""``aipf modes`` -- extract one state point's Fourier modes.

The coarse-graining width and the cutoff are the two numbers the archive is
identified by afterwards, and neither is inferable from a trajectory, so
both are required. The algorithm choices are named arguments with real
alternatives: they are passed only when they are asked for, so that what
the extractor itself declares stays the single answer to "what was it
extracted with".
"""
from __future__ import annotations

import argparse
import sys


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "modes", help="extract one state point's Fourier modes",
        description="Extract the modes of one state point, cache them "
                    "under this system's modes tier, and print where the "
                    "archive landed.")
    p.add_argument("--system", required=True,
                   help="the system whose farm holds the run, by name or "
                        "by folder")
    p.add_argument("--tag", required=True,
                   help="the state point, by the name it is filed under")
    p.add_argument("--sigma", required=True, type=float,
                   help="the coarse-graining width the run is declared at, "
                        "recorded with the archive")
    p.add_argument("--k-cut", required=True, type=float, dest="k_cut",
                   help="the largest wavenumber kept")
    p.add_argument("--ordering", default=None,
                   help="how the kept modes are ordered; the extractor's "
                        "own choice answers when this is not given, and an "
                        "unknown name is refused with the known ones")
    p.add_argument("--route", default=None,
                   help="how the atom sum is evaluated; the extractor's "
                        "own choice answers when this is not given, and an "
                        "unknown name is refused with the known ones")
    p.set_defaults(func=run)


def unindexed_tag(system, tag: str) -> str | None:
    """Why ``tag`` names no state point of the archive, naming its root; ``None`` when it does."""
    from aipf.data import index
    try:
        index.record_for_tag(system, tag)
    except (KeyError, FileNotFoundError) as refused:
        return (f"--tag {tag}: {refused.args[0]} (the archive root is "
                f"{system.paths.data_root()})")
    return None


def run(args: argparse.Namespace) -> int:
    from aipf.system import load

    system = load(args.system)
    refused = unindexed_tag(system, args.tag)
    if refused is not None:
        print(refused, file=sys.stderr)
        return 2
    # Not re-exported from the package, so the submodule is named.
    from aipf.pipeline.modes import modes
    named = {name: value
             for name, value in (("ordering", args.ordering),
                                 ("route", args.route))
             if value is not None}
    record = modes(system, args.tag, sigma=args.sigma, k_cut=args.k_cut,
                   **named)
    print(record.provenance["cache_dir"])
    return 0
