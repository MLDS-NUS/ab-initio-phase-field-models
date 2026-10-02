"""``aipf md run`` -- one state point from a built-in deck, written, pre-flighted, and run or not.

Every value the deck needs beyond the request and the system is declared: by the system, under
``defaults["md"][<template>]``, or in ``--declared`` (JSON: ``point``, ``values``, optional
``dt_equil_ps``), and ``--set KEY=VALUE`` wins over both: a request field or ``dt_equil_ps`` by that
name, a deck value otherwise. A refusal is exit 2 naming the key. ``aipf md doctor`` is one call of
:func:`aipf.md.doctor.examine_site` on the declared :class:`aipf.site.Site`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

#: The declaration's keys besides the request's fields: the deck values, the preparation timestep,
#: and the system's engine-settings campaign the run belongs to (``--campaign`` wins).
_DECLARED_KEYS = ("point", "values", "dt_equil_ps", "campaign")


def _pair(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"{text!r} is not KEY=VALUE")
    return key, value


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("md", help="molecular-dynamics runs",
                              description="Molecular-dynamics runs.")
    verbs = p.add_subparsers(dest="md_command", metavar="<verb>", required=True)
    r = verbs.add_parser(
        "run", help="fill a built-in deck for one state point, pre-flight it, run it",
        description="Fill a built-in deck for one state point and write it to "
                    "OUT/in.lammps, check the binary lists every style the deck "
                    "uses, then run it unless --dry-run.")
    r.add_argument("--system", required=True, help="the system, by name or folder")
    r.add_argument("--template", required=True,
                   help="a built-in deck by name (aipf.md.templates.DECKS)")
    r.add_argument("--declared", default=None, type=Path,
                   help="JSON with 'point' (the request) and 'values' (the holes "
                        "the request and the system do not fill); without it, the "
                        "system's own defaults['md'][TEMPLATE]")
    r.add_argument("--set", action="append", default=[], type=_pair,
                   metavar="KEY=VALUE",
                   help="one more value, winning over the declaration: a request field "
                        "(prod_ps, equil_ps, dump_every_ps, n_atoms, T, x, seed, ...) or "
                        "dt_equil_ps by that name, read as JSON; any other KEY is a deck value")
    r.add_argument("--lammps", default=None,
                   help="the simulator binary (default: the site's, AIPF_LAMMPS or [site] lammps "
                        "in aipf.toml)")
    r.add_argument("--out", required=True, type=Path, help="the run directory")
    r.add_argument("--campaign", default=None,
                   help="the system's declared engine-settings campaign, "
                        "required when the system declares any")
    r.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                   help="where a deck of a system that declares a machine-learned potential runs: "
                        "that route computes on a GPU only, so cpu is refused and auto needs a "
                        "visible GPU (or --pbs). A pair-potential deck runs the binary on the host "
                        "whatever this says")
    r.add_argument("--pbs", action="store_true",
                   help="do not run here: write OUT/job.pbs, which runs this same command without "
                        "--pbs in this environment, and submit it to the site's queue "
                        "([site] pbs_queue, pbs_project, pbs_gpus, pbs_walltime_h); print the job id")
    r.add_argument("--walltime-h", type=float, default=None,
                   help="with --pbs: the job's wall-clock limit in hours (default: [site] pbs_walltime_h)")
    r.add_argument("--dry-run", action="store_true",
                   help="write the deck and the pre-flight (and with --pbs the job file); run and "
                        "submit nothing")
    r.set_defaults(func=_run)

    d = verbs.add_parser(
        "doctor", help="whether this site runs the molecular dynamics (aipf.md.doctor.examine_site)",
        description="Check the declared site (aipf.toml [site], AIPF_* variables): the lammps python "
                    "module imports, the binary and the module list the mliap pair style, the declared "
                    "MACE potential loads, two atoms run ten steps through it on --device, and the "
                    "binary lists every style of every deck family. Exit 0 when every check is ok, "
                    "1 otherwise.")
    d.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                   help="where the ten steps run (default auto: cuda when a GPU is visible)")
    d.add_argument("--deep", action="store_true",
                   help="also import the accelerated kernels (seconds to minutes)")
    d.add_argument("--system", default=None,
                   help="check the MACE potential this system runs ([site.mace_potential] <system>, "
                        "else the plain key mace_potential); default: one report per system that "
                        "declares an ML-IAP potential, or the plain key when none does")
    d.set_defaults(func=_doctor)


def _site_lammps(given: str | None, verb: str) -> str | None:
    """``--lammps`` as given, else the site's binary; ``None`` after printing the refusal."""
    if given is not None:
        return given
    from aipf.site import MissingSiteFact, Site
    try:
        return str(Site.load().require("lammps").lammps)
    except MissingSiteFact as refused:
        print(f"aipf md {verb}: --lammps is not given and {refused}", file=sys.stderr)
        return None


def _doctor(args: argparse.Namespace) -> int:
    from aipf.md.doctor import examine_site, systems_with_potential
    from aipf.site import Site
    # the potential each system runs (Site.for_system), every declaring system unless one is named
    names = [args.system] if args.system is not None else (systems_with_potential() or None)
    diagnosis = examine_site(Site.load(), device=args.device, deep=args.deep,
                             potential_systems=names)
    print(diagnosis)
    return 0 if diagnosis.can_run is True else 1


def _value(text: str):
    """A ``--set`` value for a request field: JSON (numbers, ``null``, lists), else the text itself."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def _request(declared: dict, source: str, pairs) -> tuple[dict, dict, object]:
    """``(point, values, dt_equil_ps)``: the declaration with ``--set`` applied; ``ValueError`` names a missing key."""
    from aipf.md.request import StatePoint
    fields = {f.name: f for f in dataclasses.fields(StatePoint)}
    point = dict(declared.get("point") or {})
    values = dict(declared.get("values") or {})
    dt_equil_ps = declared.get("dt_equil_ps")
    for key, text in pairs:
        if key in fields:
            point[key] = _value(text)
        elif key == "dt_equil_ps":
            dt_equil_ps = _value(text)
        else:
            values[key] = text
    required = [name for name, f in fields.items()
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    missing = [name for name in required if name not in point]
    if missing:
        raise ValueError(f"{source}: the request declares no {missing} under 'point'; "
                         f"declare it there or pass --set {missing[0]}=VALUE")
    unknown = sorted(set(point) - set(fields))
    if unknown:
        raise ValueError(f"{source}: unknown request field(s) {unknown} under 'point'; "
                         f"the fields are {sorted(fields)}")
    return point, values, dt_equil_ps


def _refuse(message: str) -> int:
    print(f"aipf md run: {message}", file=sys.stderr)
    return 2


def _job_command(args: argparse.Namespace) -> list[str]:
    """This command as the batch job runs it: no ``--pbs``, no ``--dry-run``, every path absolute."""
    from aipf.system import _looks_like_a_path

    def absolute(text: str) -> str:              # a folder or a binary given as a path; a name as is
        return str(Path(text).resolve()) if _looks_like_a_path(text) else text

    command = ["aipf", "md", "run", "--system", absolute(args.system), "--template", args.template]
    if args.declared is not None:
        command += ["--declared", str(args.declared.resolve())]
    for key, value in args.set:
        command += ["--set", f"{key}={value}"]
    if args.lammps is not None:
        command += ["--lammps", absolute(args.lammps)]
    command += ["--out", str(args.out.resolve())]
    if args.campaign is not None:
        command += ["--campaign", args.campaign]
    return command + ["--device", args.device]


def _leave(code: int) -> int:
    """Leave after an in-process run without the accelerator's teardown (it aborts on this build)."""
    from aipf.md.engine import exit_cleanly
    exit_cleanly(code)
    return code                                   # not reached


def _run(args: argparse.Namespace) -> int:
    from aipf.md.run import run
    from aipf.md.scheduler import PbsSpec, SchedulerError
    from aipf.site import Site
    from aipf.system import load

    system = load(args.system)
    lammps = _site_lammps(args.lammps, "run")
    if lammps is None:
        return 2
    if args.declared is not None:
        declared, source = json.loads(args.declared.read_text()), str(args.declared)
    else:
        declared = system.defaults.get("md", {}).get(args.template)
        source = f"system {system.name!r} defaults['md'][{args.template!r}]"
        if declared is None:
            return _refuse(
                f"--declared is not given and system {system.name!r} declares no "
                f"defaults['md'][{args.template!r}]; declare the deck's values "
                f"in one of the two")
    unknown = sorted(set(declared) - set(_DECLARED_KEYS))
    if unknown:
        return _refuse(f"{source}: unknown key(s) {unknown}; a declaration holds {list(_DECLARED_KEYS)}")
    if args.walltime_h is not None and not args.pbs:
        return _refuse("--walltime-h is the batch job's limit and needs --pbs")
    try:
        pbs = PbsSpec.from_site(Site.load(), walltime_h=args.walltime_h) if args.pbs else None
        point, values, dt_equil_ps = _request(declared, source, args.set)
        record = run(system, template=args.template, point=point,
                     values=values, out=args.out, lammps=lammps,
                     dry_run=args.dry_run,
                     campaign=args.campaign if args.campaign is not None else declared.get("campaign"),
                     dt_equil_ps=dt_equil_ps, device=args.device, pbs=pbs,
                     job_command=_job_command(args) if args.pbs else None)
    except (ValueError, TypeError, KeyError, NotImplementedError, SchedulerError) as refused:
        # a hole nobody filled (DeckError), a request that does not validate, an undeclared
        # campaign, a device or a potential this deck cannot run with, a job the queue refused:
        # all before anything was run
        return _refuse(str(refused.args[0]) if refused.args else repr(refused))
    print(record.deck_path)
    print(record.diagnosis)
    print(record.command)
    if record.job_path is not None:
        print(record.job_path)
        if record.job_id is not None:
            print(record.job_id)
    if record.result is not None:
        print(record.result.status)
        code = 0 if record.result.ok else 1
        return _leave(code) if record.route == "in_process" else code
    return 0 if record.diagnosis.can_run is True else 1
