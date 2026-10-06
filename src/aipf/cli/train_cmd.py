"""``aipf train`` -- train a declared system's functional.

Run name, seed, length and sources are flags with no default; everything else comes off ``system.defaults``."""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence, Tuple, Union

from aipf.cli import variant
from aipf.paths import PUBLISHED_DIRNAME

#: How one training source is spelled; SUBDIR is relative to the system's declared source root. A
#: two-dimensional model's source (``--projection``) spells two lengths, ``GX,GY``.
SOURCE_SPELLING = "NAME=SUBDIR:PATTERN:GX,GY,GZ"

#: ``--source declared``: the system's own ``defaults["training"]["sources"]``, exclusions included.
DECLARED = "declared"

#: ``--source sample``: the system's bundled sample, ``data/<system>/sample``, whose sources are
#: ``defaults["training"]["sample"]``; no raw root is read.
SAMPLE = "sample"

#: The ``--source`` keywords that name a whole set and stand alone.
_KEYWORDS = (DECLARED, SAMPLE)

#: A field grid is three lengths. A source that gives fewer has not said
#: what shape its fields are.
_GRID_RANK = 3

#: The grid of a source read through ``--projection`` is two lengths, ``(Gx, Gy)``.
_PROJECTED_GRID_RANK = 2

#: ``--projection``'s values, :data:`aipf.train.projection.PROJECTIONS` spelled here so the parser
#: imports no training module.
PROJECTIONS = ("kz0-volumetric", "kz0-areal")


def source_row(text: str) -> Union[str, Tuple[str, str, str, Tuple[int, ...]]]:
    """One ``--source`` argument as the ``(name, subdir, pattern, grid)`` row
    ``aipf.train.fit._specs_from`` takes, or :data:`DECLARED` or :data:`SAMPLE`."""
    if text in _KEYWORDS:
        return text
    name, sep, rest = text.partition("=")
    parts = rest.split(":")
    if not sep or not name or len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a source. Spell it {SOURCE_SPELLING}: a name, "
            f"the subdirectory of the fields root it is cut from, the glob "
            f"that selects its files, and its grid")
    subdir, pattern, grid_text = parts
    try:
        grid = tuple(int(g) for g in grid_text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{grid_text!r} is not a grid: {_GRID_RANK} integers, "
            f"comma-separated, as in {SOURCE_SPELLING}")
    if subdir.startswith("/"):
        raise argparse.ArgumentTypeError(
            f"{text!r} names an absolute tree {subdir!r}; SUBDIR is relative "
            f"to the system's declared source root "
            f"(defaults['training']['source_root'])")
    if len(grid) not in (_GRID_RANK, _PROJECTED_GRID_RANK) or not subdir or not pattern:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a source. Spell it {SOURCE_SPELLING}: the "
            f"subdirectory, the glob and {_GRID_RANK} grid lengths are all "
            f"required ({_PROJECTED_GRID_RANK}, GX,GY, for a two-dimensional "
            f"model trained through --projection)")
    return (name, subdir, pattern, grid)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train", help="train this system's declared functional",
        description="Train the functional a system declares and write a "
                    "signed run directory. Nothing below has a default: a "
                    "run that did not state one of these is not a run "
                    "anyone can reproduce.")
    p.add_argument("--system", required=True,
                   help="the system to train, by name or by folder")
    variant.add_argument(p)
    p.add_argument("--run", required=True, dest="run_name",
                   help="the run's name, and the directory it writes to "
                        "under this system's checkpoint tier")
    p.add_argument("--seed", required=True, type=int,
                   help="the one seed every stream of the run is derived "
                        "from: the parameters, the batch order, the draws")
    length = p.add_mutually_exclusive_group(required=True)
    length.add_argument("--steps", type=int,
                        help="how long to train, in optimiser steps "
                             "(exactly one of --steps and --epochs)")
    length.add_argument("--epochs", type=int,
                        help="how long to train, in epochs "
                             "(exactly one of --steps and --epochs)")
    p.add_argument("--source", required=True, action="append",
                   type=source_row, metavar=SOURCE_SPELLING,
                   help="a training source, repeatable and needed at least "
                        "once; SUBDIR is relative to the system's declared "
                        "source root. Its loss weight is the system's declared one. "
                        "`--source declared` alone trains on the system's declared "
                        "sources (defaults['training']['sources']). `--source sample` "
                        "alone trains on the small sample bundled in the repository, "
                        "data/<system>/sample (defaults['training']['sample']), and reads "
                        "no raw data root: it exercises the pipeline and gives a model "
                        "with no physics in it")
    p.add_argument(f"--init-from-{PUBLISHED_DIRNAME}", dest="init_from_published",
                   action="store_true",
                   help="start from the weights of the checkpoint this "
                        "system declares (the repository's tracked copy "
                        "first, then the raw root), digest-checked "
                        "before it is read")
    p.add_argument("--resume-optimizer", required=True, choices=("yes", "no"),
                   help="whether to continue the optimizer and schedule "
                        "saved in --init-from-published's checkpoint ('yes') "
                        "or build both fresh ('no'). REQUIRED, because the "
                        "two are different experiments and not two settings "
                        "of one: a fresh schedule starts in its linear "
                        "warm-up, where a step moves no float32 parameter, "
                        "and a resumed one starts at the rate the saved run "
                        "had reached")
    p.add_argument("--anchors", required=True, choices=("declared", "none"),
                   help="whether to train the measured anchor tables this "
                        "system declares ('declared') or the drift term "
                        "alone ('none'). REQUIRED, because the anchors are "
                        "most of the loss -- 94 per cent of it at step 0 on "
                        "the first system measured -- so a run without them "
                        "is a different experiment and not a cheaper "
                        "version of the same one. A system that declares no "
                        "tables trains the drift term whichever is passed")
    p.add_argument("--projection", default=None, choices=PROJECTIONS,
                   help="train a two-dimensional model (a factory model declared on a two-axis "
                        "grid) on the k_z = 0 plane of the three-dimensional archives: "
                        "'kz0-volumetric' scatters rho_k / V_ref (the z mean of the density), "
                        "'kz0-areal' rho_k / (Lx_ref Ly_ref) (the density integrated along z). "
                        "The archives must record their reference cell, every --source grid is "
                        "GX,GY, and only the drift term trains (--anchors none). Off by default: "
                        "a three-dimensional model reads the archives as they are")
    p.add_argument("--log-every-step", action="store_true",
                   help="also write the per-step loss series")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                   help="where to train (default auto: cuda when torch sees a GPU, else cpu); "
                        "cuda without one is refused before the run directory is made")
    p.add_argument("--deterministic", action="store_true",
                   help="ask torch for deterministic kernels for this run (slower; what a "
                        "bit-for-bit comparison needs). Off by default, and then no switch is touched")
    p.add_argument("--pbs", action="store_true",
                   help="do not train here: write job.pbs in the run directory, which runs this same "
                        "command without --pbs in this environment, and submit it to the site's queue "
                        "([site] pbs_queue, pbs_project, pbs_gpus, pbs_walltime_h); print the job id")
    p.add_argument("--walltime-h", type=float, default=None,
                   help="with --pbs: the job's wall-clock limit in hours (default: [site] pbs_walltime_h)")
    p.add_argument("--dry-run", action="store_true",
                   help="with --pbs: write the job file and submit nothing")
    p.set_defaults(func=run)


def _anchors_of(args: argparse.Namespace):
    """``--anchors`` as ``fit`` takes it: ``"declared"`` -> ``None`` (the declaration answers),
    ``"none"`` -> ``NO_ANCHORS`` (drift-only)."""
    from aipf.train.anchors import NO_ANCHORS
    return None if args.anchors == "declared" else NO_ANCHORS


def _projection_refusal(args: argparse.Namespace, table) -> Optional[str]:
    """Why ``--projection`` and the source grids (or ``--anchors``) contradict each other, or ``None``.
    A projected source's grid is two lengths and a three-dimensional one's three; the anchor tables
    are three-dimensional only."""
    rank = _GRID_RANK if args.projection is None else _PROJECTED_GRID_RANK
    wrong = [row[0] for row in table if len(row[3]) != rank]
    if wrong and args.projection is None:
        return (f"--source {', '.join(wrong)}: a two-length grid GX,GY is a two-dimensional "
                f"model's, which trains on the k_z = 0 plane of the archives: pass --projection "
                f"(one of {', '.join(PROJECTIONS)})")
    if wrong:
        return (f"--projection {args.projection}: --source {', '.join(wrong)} spells "
                f"{_GRID_RANK} grid lengths; the k_z = 0 plane scatters onto a two-axis grid, "
                f"GX,GY")
    if args.projection is not None and args.anchors != "none":
        return (f"--projection {args.projection} trains a two-dimensional model, and the anchor "
                f"tables are three-dimensional only: pass --anchors none (the drift term alone)")
    return None


def unmatched_sources(system, table) -> list:
    """One refusal per source whose glob selects no directory of the archive once its excluded runs
    (a row's optional fifth field) are removed, naming its root."""
    from aipf.train.fit import source_root
    root = source_root(system)
    refusals = []
    for name, subdir, pattern, *rest in table:
        tree = root / subdir
        excluded = set(rest[1]) if len(rest) > 1 else set()
        if not any(p.is_dir() and p.name not in excluded for p in tree.glob(pattern)):
            also = (f", once its {len(excluded)} excluded run(s) are removed"
                    if excluded else "")
            refusals.append(
                f"--source {name}: {pattern!r} matches no run directory "
                f"under {tree}{also} (the archive root is {root})")
    return refusals


def _source_table(system, given) -> Optional[Sequence[Tuple]]:
    """The ``--source`` rows as given, or the system's declared ones for ``--source declared`` (and for
    ``--source sample``, on the system :func:`_on_sample` returned); ``None`` after printing a refusal."""
    keyword = next((g for g in given if g in _KEYWORDS), None)
    if keyword is None:
        return tuple(given)
    if len(given) > 1:
        print(f"--source {keyword} stands alone: it names the system's whole "
              f"{'declared set' if keyword == DECLARED else 'bundled sample'}, "
              f"so it cannot be combined with another --source", file=sys.stderr)
        return None
    from aipf.train.fit import declared_source_table
    try:
        return declared_source_table(system)
    except (KeyError, ValueError) as refused:
        print(f"--source {keyword}: {refused.args[0]}", file=sys.stderr)
        return None


def _on_sample(system, given):
    """``system`` itself, or under ``--source sample`` the system as it trains on its bundled sample
    (:func:`aipf.train.fit.sample_system`); ``None`` after printing a refusal."""
    if SAMPLE not in given:
        return system
    from aipf.train.fit import sample_system
    try:
        return sample_system(system)
    except (KeyError, FileNotFoundError) as refused:
        print(f"--source {SAMPLE}: {refused.args[0]}", file=sys.stderr)
        return None


def _absolute_if_path(text: str) -> str:
    """A ``--system`` given as a folder, made absolute (the job runs in another directory); a name as is."""
    from pathlib import Path

    from aipf.system import _looks_like_a_path
    return str(Path(text).resolve()) if _looks_like_a_path(text) else text


def _job_command(args: argparse.Namespace) -> list:
    """This command as the batch job runs it: no ``--pbs``, no ``--walltime-h``, no ``--dry-run``."""
    command = ["aipf", "train", "--system", _absolute_if_path(args.system)]
    if args.variant is not None:
        command += ["--variant", args.variant]
    command += ["--run", args.run_name, "--seed", str(args.seed)]
    command += (["--steps", str(args.steps)] if args.steps is not None
                else ["--epochs", str(args.epochs)])
    for row in args.source:
        if row in _KEYWORDS:
            command += ["--source", row]
            continue
        name, subdir, pattern, grid = row
        command += ["--source", f"{name}={subdir}:{pattern}:{','.join(str(g) for g in grid)}"]
    if args.init_from_published:
        command.append(f"--init-from-{PUBLISHED_DIRNAME}")
    command += ["--resume-optimizer", args.resume_optimizer, "--anchors", args.anchors]
    if args.projection is not None:
        command += ["--projection", args.projection]
    if args.log_every_step:
        command.append("--log-every-step")
    command += ["--device", args.device]
    if args.deterministic:
        command.append("--deterministic")
    return command


def _submit(args: argparse.Namespace, system) -> int:
    """Write ``<run directory>/job.pbs`` and submit it unless ``--dry-run``; print the file and the job id.

    The job is handed the data root and the system's raw root this submission resolved."""
    from aipf import paths
    from aipf.data import index
    from aipf.md.scheduler import PbsSpec, SchedulerError, batch_job, job_name
    from aipf.site import Site
    spec = PbsSpec.from_site(Site.load(), walltime_h=args.walltime_h)
    env = {"AIPF_DATA": str(paths.data_root())}
    try:
        if not isinstance(system.paths, paths.SamplePaths):  # the sample is found in the checkout
            env[paths.raw_env_name(system.name)] = str(system.paths.raw())
    except paths.MissingLocation:
        pass                                      # the job refuses the same way, naming it
    workdir = index.ckpt_dir(system) / args.run_name
    job = batch_job(spec, name=job_name(f"train-{args.run_name}"), command=_job_command(args),
                    workdir=workdir, env=env)
    try:
        backend = spec.backend()
        if args.dry_run:
            print(backend.write(job))
            return 0
        handle = backend.submit(job)
    except SchedulerError as refused:
        print(f"aipf train --pbs: {refused}", file=sys.stderr)
        return 2
    print(job.script_path)
    print(handle.job_id)
    return 0


def run(args: argparse.Namespace) -> int:
    system = variant.load_system(args)
    if system is None:
        return 2
    system = _on_sample(system, args.source)
    if system is None:
        return 2
    if (args.walltime_h is not None or args.dry_run) and not args.pbs:
        print("--walltime-h and --dry-run belong to a batch submission and need --pbs",
              file=sys.stderr)
        return 2

    init_from = None
    if args.init_from_published:
        init_from = system.checkpoint
        if init_from is None:
            print(f"--init-from-{PUBLISHED_DIRNAME}: system {system.name!r} declares no "
                  f"checkpoint, so there are no weights to start from",
                  file=sys.stderr)
            return 2

    # fit's own refusals, made before anything is written (and before a job is submitted):
    # a reserved run name, the starting weights found and digest-checked
    from aipf.train.fit import ReservedRunName, pre_run_checks
    try:
        init_path = pre_run_checks(system, args.run_name, init_from)
    except ReservedRunName as refused:
        print(f"aipf train: {refused}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as refused:
        print(f"--init-from-{PUBLISHED_DIRNAME}: the checkpoint system {system.name!r} "
              f"declares is refused: {refused}", file=sys.stderr)
        return 2

    resume_optimizer = args.resume_optimizer == "yes"
    # refused here, by name: the two flags contradict each other (`fit` refuses it too)
    if resume_optimizer and init_from is None:
        print("--resume-optimizer yes needs --init-from-published: there is "
              "no saved optimizer to continue without a checkpoint to read "
              "it from", file=sys.stderr)
        return 2

    table = _source_table(system, args.source)
    if table is None:
        return 2
    refusal = _projection_refusal(args, table)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2
    refusals = unmatched_sources(system, table)
    if refusals:
        print("\n".join(refusals), file=sys.stderr)
        return 2

    if args.pbs:
        # a batch job would only fail on it later: refused here, by name, before a job exists
        if args.projection is not None:
            from aipf.train.fit import DimensionMismatch, _specs_from, check_dimensions
            try:
                check_dimensions(system, _specs_from(system, table, [row[0] for row in table]),
                                 projection=args.projection, anchors=_anchors_of(args))
            except DimensionMismatch as refused:
                print(f"aipf train: --projection {args.projection}: {refused}", file=sys.stderr)
                return 2
        if resume_optimizer:
            from aipf.train.fit import OptimizerLayoutMismatch, check_optimizer_resumable
            try:
                check_optimizer_resumable(system, init_from, init_path)
            except (OptimizerLayoutMismatch, KeyError) as refused:
                print(f"aipf train: --resume-optimizer yes: {str(refused.args[0]).rstrip('.')}.",
                      file=sys.stderr)
                return 2
        return _submit(args, system)

    # refused here, by name, before any run directory exists (`fit` refuses it too)
    from aipf.train.fit import DeviceUnavailable, resolve_device
    try:
        resolve_device(args.device)
    except DeviceUnavailable as refused:
        print(f"--device {args.device}: {refused}", file=sys.stderr)
        return 2

    # the driver's own helper assembles the specs, so the CLI and a Python caller agree
    from aipf.train.fit import _specs_from, fit
    sources = _specs_from(system, table, [row[0] for row in table])

    anchors = _anchors_of(args)

    # a saved optimizer whose parameter groups do not fit this model's in count, shape or
    # recorded order is refused by fit, by name, before the run directory exists; every
    # published checkpoint is refused this way (they record no parameter names)
    from aipf.train.fit import DimensionMismatch, OptimizerLayoutMismatch
    try:
        run_dir = fit(system, run_name=args.run_name, sources=sources,
                      seed=args.seed, steps=args.steps, epochs=args.epochs,
                      init_from=init_from, resume_optimizer=resume_optimizer,
                      anchors=anchors, log_every_step=args.log_every_step,
                      device=args.device, deterministic=args.deterministic,
                      **({} if args.projection is None else {"projection": args.projection}))
    except OptimizerLayoutMismatch as refused:
        print(f"aipf train: --resume-optimizer yes: {str(refused).rstrip('.')}.", file=sys.stderr)
        return 2
    except DimensionMismatch as refused:
        print(f"aipf train: {refused}", file=sys.stderr)
        return 2
    print(run_dir)
    return 0
