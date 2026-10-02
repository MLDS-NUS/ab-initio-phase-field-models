"""``aipf diagnose`` -- run named diagnostic stages over one checkpoint.

``--stage`` is required and repeatable; thresholds are forwarded from the system's ``defaults["diagnose"]``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aipf.cli import variant
from aipf.paths import PUBLISHED_DIRNAME

#: The key of ``system.defaults`` this verb forwards, unchanged, as the
#: driver's declared keywords.
_DECLARED_KEY = "diagnose"

#: How the temperature grid is spelled: the two ends and the step, the
#: inclusive form the driver's own grid helper expands.
_T_GRID_SPELLING = "LO,HI,STEP"

#: The count spelling: the two ends and the number of points, ``np.linspace``.
_T_GRID_N_SPELLING = "LO,HI,N"


def _t_grid(text: str):
    """``LO,HI,STEP`` as the inclusive grid the driver builds from it."""
    from aipf.diagnose.run import grid
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a temperature grid. Spell it "
            f"{_T_GRID_SPELLING}, the two ends and the step")
    try:
        return grid(tuple(float(v) for v in parts))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a temperature grid: three numbers, "
            f"{_T_GRID_SPELLING}")


def _t_grid_n(text: str):
    """``LO,HI,N`` as ``np.linspace(LO, HI, N)``."""
    from aipf.diagnose.run import grid_n
    parts = text.split(",")
    try:
        if len(parts) != 3:
            raise ValueError(text)
        return grid_n((float(parts[0]), float(parts[1]), float(parts[2])))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a temperature grid: spell it "
            f"{_T_GRID_N_SPELLING}, the two ends and an integer count")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    # the stages are the driver's own tuple, read from the stdlib-only module so `aipf --help` does
    # not import torch
    from aipf.diagnose.stages import STAGES

    p = subparsers.add_parser(
        "diagnose", help="run diagnostic stages over one checkpoint",
        description="Diagnose one checkpoint and write a directory named "
                    "by its digest. Which stages run is asked for, never "
                    "assumed; the thresholds each stage needs come from the "
                    f"system's declared {_DECLARED_KEY!r} block.")
    p.add_argument("--system", required=True,
                   help="the system whose declaration the diagnosis reads, "
                        "by name or by folder")
    p.add_argument("--ckpt", required=True,
                   help=f"the checkpoint file to diagnose, or {PUBLISHED_DIRNAME!r} "
                        f"for the one the system declares (its digest is "
                        f"verified before it is read)")
    variant.add_argument(p)
    p.add_argument("--stage", required=True, action="append", choices=STAGES,
                   help="a stage to run, repeatable and needed at least "
                        "once; each costs minutes, so none is implied")
    p.add_argument("--out", default=None,
                   help="where the digest-named directory is written "
                        "(default: this system's declared diagnose tier)")
    p.add_argument("--pressure", action="append", type=float, default=None,
                   help="an isobar for the phase_diagram stage, repeatable; "
                        "that stage refuses to run without one")
    grids = p.add_mutually_exclusive_group()
    grids.add_argument("--T-grid", dest="t_grid", default=None, type=_t_grid,
                       metavar=_T_GRID_SPELLING,
                       help="where to cut those isobars, as the two ends and "
                            "the step; the phase_diagram stage refuses to run "
                            "without it, and the one-field stage reads its "
                            "declared grid when it is not given")
    grids.add_argument("--T-grid-n", dest="t_grid", default=None,
                       type=_t_grid_n, metavar=_T_GRID_N_SPELLING,
                       help="the temperature grid as the two ends and a "
                            "count (np.linspace), for a grid a step cannot "
                            "spell exactly")
    p.add_argument("--override-declared", action="store_true",
                   help="run the one-field stage on a --T-grid/--T-grid-n that differs "
                        "from the declared defaults['diagnose']['one_field']['T_grid']; "
                        "without it such a grid is refused")
    p.set_defaults(func=run)


def _published(system) -> Path | None:
    """The declared checkpoint, tracked copy first, digest-verified; ``None`` after printing why not."""
    if system.checkpoint is None:
        print(f"--ckpt {PUBLISHED_DIRNAME}: system {system.name!r} declares no "
              f"checkpoint", file=sys.stderr)
        return None
    try:
        return system.resolve_checkpoint()
    except (FileNotFoundError, ValueError) as refused:
        print(f"--ckpt {PUBLISHED_DIRNAME}: the checkpoint system {system.name!r} "
              f"declares is refused: {refused}", file=sys.stderr)
        return None


def run(args: argparse.Namespace) -> int:
    system = variant.load_system(args)
    if system is None:
        return 2
    declared = dict(system.defaults.get(_DECLARED_KEY, {}))
    ckpt = args.ckpt
    if ckpt == PUBLISHED_DIRNAME:
        ckpt = _published(system)
        if ckpt is None:
            return 2
    from aipf.diagnose.run import check_declared, request_T_grid
    from aipf.diagnose.run import run as diagnose
    try:
        request_T_grid(declared, tuple(args.stage), tuple(args.pressure or ()),
                       args.t_grid, override_declared=args.override_declared)
        check_declared(declared, tuple(args.stage))
    except ValueError as refused:
        print(f"aipf diagnose: {refused}", file=sys.stderr)
        return 2
    out_dir = diagnose(system, ckpt,
                       stages=tuple(args.stage),
                       out=None if args.out is None else Path(args.out),
                       pressures=tuple(args.pressure or ()),
                       T_grid=args.t_grid,
                       override_declared=args.override_declared,
                       **declared)
    print(out_dir)
    return 0
