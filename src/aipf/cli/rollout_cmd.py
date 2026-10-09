"""``aipf rollout {spinodal,slab}`` -- the two basic checks on a trained functional.

Every flag is required but ``--precision`` (default ``fp32``, the published rollouts' precision; ``fp64``
runs the solver on a float64 copy of the model and the cast initial state, and is recorded in the
output's declaration, so it gets its own directory); the solver, archive and grid come off
``system.defaults["rollout"]`` and the noise off ``system.noise``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aipf.cli import variant
from aipf.paths import PUBLISHED_DIRNAME


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "rollout", help="roll a trained functional from a measured run",
        description="Integrate the declared functional from frame 0 of an "
                    "archived run and compare with that run. Nothing below "
                    "has a default but --precision.")
    p.add_argument("driver", choices=("spinodal", "slab"))
    p.add_argument("--system", required=True)
    variant.add_argument(p)
    p.add_argument("--ckpt", required=True,
                   help=f"{PUBLISHED_DIRNAME!r} (the system's checkpoint, md5-checked) or "
                        "a path")
    p.add_argument("--run", required=True,
                   help="the archived run, under the declared modes tree")
    p.add_argument("--seeds", required=True, nargs="*", type=int,
                   help="one noisy rollout per seed; none given: one "
                        "deterministic rollout")
    p.add_argument("--t-end", required=True, type=float)
    p.add_argument("--dt", required=True, type=float)
    p.add_argument("--save-ps", required=True, type=float)
    p.add_argument("--device", required=True)
    p.add_argument("--out", required=True,
                   help="output root; 'data' is data/<system>/rollout")
    p.add_argument("--precision", choices=("fp32", "fp64"), default="fp32",
                   help="fp32 (default, as published) or fp64: the model and the initial "
                        "state cast to float64, recorded in the output declaration")
    p.set_defaults(func=run)


def refusal(system, driver: str, ckpt: str) -> str | None:
    """Why the declaration or the checkpoint cannot run ``driver``, naming the missing key; ``None`` when they can."""
    from aipf.rollout.spinodal import declared, resolve_checkpoint, trust_domain
    from aipf.solve.noise import declared_noise
    try:
        declared(system, driver)
        trust_domain(system)
        declared_noise(system)
    except ValueError as refused:
        return str(refused)
    try:
        resolve_checkpoint(system, ckpt)
    except (FileNotFoundError, ValueError) as refused:
        return f"--ckpt {ckpt}: {refused}"
    return None


def run(args: argparse.Namespace) -> int:
    from aipf.rollout.slab import slab
    from aipf.rollout.spinodal import spinodal
    system = variant.load_system(args)
    if system is None:
        return 2
    refused = refusal(system, args.driver, args.ckpt)
    if refused is not None:
        print(f"aipf rollout {args.driver}: {refused}", file=sys.stderr)
        return 2
    driver = spinodal if args.driver == "spinodal" else slab
    out = None if args.out == "data" else Path(args.out)
    print(driver(system, args.ckpt, run=args.run,
                 seeds=tuple(args.seeds), t_end=args.t_end, dt=args.dt,
                 save_ps=args.save_ps, device=args.device, out=out,
                 precision=args.precision))
    return 0
