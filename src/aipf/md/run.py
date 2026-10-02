"""One state point, from a named template to a deck on disk, checked, and run here, submitted, or not.

``run`` fills a built-in deck from a request, the system and the caller's declared values, writes
it, checks the binary against every style it names, and runs it only when that check passes and
``dry_run`` is false. Two routes:

* a deck whose interactions are given (a pair potential in the declared values) runs the binary
  on the deck file, as a separate process;
* a deck of a system that declares a machine-learned potential (``defaults["md"]["potential"]``)
  gets its pair lines from that potential, resolved from the site (``Site.for_system``) and
  checked against the declared md5, and runs in this interpreter through the ML-IAP coupling, on
  a GPU only.

With ``pbs`` the run is not made here: ``out/job.pbs`` runs ``job_command`` (the same command,
without the submission) in this environment, and is submitted unless ``dry_run``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from aipf.md.doctor import Diagnosis, alloc_conf, deck_styles, gpu_visible
from aipf.md.engine import UNIFIED_STYLE, Result, check_deck, pair_commands, plan, strip_environment
from aipf.md.engine import run as run_deck
from aipf.md.request import StatePoint
from aipf.md.templates import DECKS, Deck, DeckError, fill
from aipf.system import MdSettings, System

__all__ = ["DECK_FILE", "DEVICES", "JOB_FILE", "JOB_LOG", "JOB_RECORD", "LOG_FILE", "POTENTIAL_KEY",
           "RECORD_FILE", "DeviceRefused", "RunRecord", "deck_named", "declared_potential",
           "point_values", "run", "settings_values"]

#: What ``run`` writes into ``out``.
DECK_FILE, LOG_FILE, RECORD_FILE = "in.lammps", "log.lammps", "run.json"
#: What a ``pbs`` run adds: the job file, the job's own log, and the submission's record.
JOB_FILE, JOB_LOG, JOB_RECORD = "job.pbs", "job.log", "job.json"
#: The key under ``defaults["md"]`` that declares a system's machine-learned potential.
POTENTIAL_KEY = "potential"
#: Where a run computes.
DEVICES = ("auto", "cpu", "cuda")

_PAIR_ROUTE, _MODEL_ROUTE = "input_file", "in_process"

#: Why the model route has no host path (measured: Kokkos built for CUDA refuses to start without a
#: device, and the MACE coupling refuses host tensors; with that refusal lifted the model segfaults).
_CPU_REFUSAL = (
    "the ML-IAP route (a MACE potential through the mliap/kk pair style) computes on a GPU only: "
    "the coupling refuses host arrays, and a CUDA build of Kokkos does not start without a device. "
    "Pass --device cuda on a node with a GPU, or submit with --pbs")


class DeviceRefused(ValueError):
    """The device asked for cannot run this deck; nothing was written."""


@dataclass(frozen=True)
class RunRecord:
    """The deck as written, the command that runs it, the pre-flight, and the result (``None`` when
    nothing ran here). ``route`` is how it runs; ``job_path``/``job_id`` are set by a ``pbs`` run."""

    deck_path: Path
    command: str
    diagnosis: Diagnosis
    result: Result | None
    route: str = _PAIR_ROUTE
    job_path: Path | None = None
    job_id: str | None = None


def deck_named(name: str) -> Deck:
    """The built-in deck called ``name``; a refusal lists every name."""
    for deck in DECKS.values():
        if deck.name == name:
            return deck
    raise DeckError(f"no built-in deck named {name!r}; there are "
                    f"{sorted({d.name for d in DECKS.values()})}")


def settings_values(settings: MdSettings, point: StatePoint) -> dict[str, str]:
    """The placeholders one declared campaign's engine knobs fill; ``P`` only when the request sets one."""
    values = {"UNITS": settings.units, "SKIN": repr(float(settings.skin)),
              "NEIGH_MODIFY": settings.neigh_modify,
              "N_THERMO": str(settings.thermo_every),
              "TDAMP": repr(float(settings.T_damp)),
              "PDAMP": repr(float(settings.P_damp))}
    if point.P is not None:
        values["P"] = repr(settings.deck_pressure(point.P))
    return values


def point_values(point: StatePoint) -> dict[str, str]:
    """``N_ATOMS`` and ``X`` for a declared configuration that builds its atoms, when the request sets them."""
    values = {}
    if point.n_atoms is not None:
        values["N_ATOMS"] = str(int(point.n_atoms))
    if not isinstance(point.x, tuple):
        values["X"] = repr(float(point.x))
    return values


def declared_potential(system: System) -> Mapping[str, Any] | None:
    """The system's ``defaults["md"]["potential"]`` (``{"md5": ...}``), or ``None`` when it declares none."""
    declared = ((system.defaults or {}).get("md") or {}).get(POTENTIAL_KEY)
    return declared or None


def _point(point: StatePoint | Mapping[str, Any]) -> StatePoint:
    if isinstance(point, StatePoint):
        return point
    fields = dict(point)
    if isinstance(fields.get("x"), list):
        fields["x"] = tuple(fields["x"])
    return StatePoint(**fields)


def _names_the_model_style(text: str) -> bool:
    return any(line.split()[:2] and line.split()[1].split("/")[0] == UNIFIED_STYLE
               for line in text.splitlines() if line.startswith("pair_style"))


def _resolve_potential(system: System, declared: Mapping[str, Any]):
    """The site's potential, checked against the declared md5 and the system's typed species."""
    from aipf import paths
    from aipf.md.potentials import PotentialError, resolve
    from aipf.site import FACTS, Site, per_system_variable
    md5 = declared.get("md5")
    if not md5:
        raise PotentialError(
            f"system {system.name!r} declares defaults['md'][{POTENTIAL_KEY!r}] without an md5, so "
            f"nothing says which file its data were generated with; declare {{'md5': ...}}")
    remedy = (f"set {per_system_variable('mace_potential', system.name)}, or {system.name} = "
              f"\"/path\" under [site.mace_potential] in {paths._where_config()} "
              f"({FACTS['mace_potential']} serves every system)")
    potential = resolve(Site.load().for_system("mace_potential", system.name), engine="mace-mliap",
                        md5=md5, remedy=remedy)
    typed = [name for name, _ in sorted(system.atom_types.items(), key=lambda item: item[1])]
    absent = [name for name in typed if name not in potential.species]
    if absent:
        raise PotentialError(
            f"the potential at {potential.path} knows {list(potential.species)} and system "
            f"{system.name!r} types {typed}: {absent} would have no interactions")
    return potential, typed


def _model_pair(potential, typed: Sequence[str]) -> str:
    """The pair lines of the in-process route, the coefficients in the deck's TYPE order."""
    head = pair_commands(potential, route=_MODEL_ROUTE).splitlines()[0]
    return f"{head}\npair_coeff      * * {' '.join(typed)}"


def _device(device: str, *, submitting: bool) -> str:
    """``cuda`` for the model route, or :class:`DeviceRefused` saying why not (before anything is written)."""
    if device not in DEVICES:
        raise DeviceRefused(f"device is one of {DEVICES}, got {device!r}")
    if device == "cpu":
        raise DeviceRefused(f"--device cpu: {_CPU_REFUSAL}")
    if submitting:
        return "cuda"                      # the job's node has the GPU, not this one
    visible, seen = gpu_visible()
    if not visible:
        asked = "--device cuda" if device == "cuda" else "--device auto"
        raise DeviceRefused(f"{asked} and no GPU is visible here ({seen}); {_CPU_REFUSAL}")
    return "cuda"


def run(system: System, *, template: str, point: StatePoint | Mapping[str, Any],
        values: Mapping[str, Any], out: str | Path, lammps: str | Path | None = None,
        dry_run: bool = False, campaign: str | None = None,
        dt_equil_ps: float | None = None, device: str = "auto",
        pbs: Any | None = None, job_command: Sequence[str] | None = None) -> RunRecord:
    """Fill ``template`` for ``point`` and write ``out/in.lammps``; pre-flight it against ``lammps``;
    run it here, or submit ``job_command`` with ``pbs`` (a :class:`aipf.md.scheduler.PbsSpec`), unless ``dry_run``.

    ``values`` fill the holes neither the request nor the system determines and win over derived ones;
    ``campaign`` names the system's declared engine knobs (required when the system declares any);
    ``lammps`` defaults to the site's binary. Refusals (a hole, the device, the potential) are raised
    before anything is written.

    ``system``: the declared system (masses, preparation, engine knobs, potential).
    ``template``: the built-in deck's name (:func:`deck_named`).
    ``point``: the state point, a :class:`~aipf.md.request.StatePoint` or its manifest record.
    ``values``: deck values the request and the system do not determine.
    ``out``: the run directory, created; the deck is ``out/in.lammps``.
    ``lammps``: the binary to pre-flight and run, or ``None`` for the site's.
    ``dry_run``: write and pre-flight, run or submit nothing.
    ``campaign``: the system's engine-knob campaign this run belongs to.
    ``dt_equil_ps``: the equilibration timestep in ps, when it differs from the production one.
    ``device``: one of :data:`DEVICES`; a model deck runs on ``cuda`` only.
    ``pbs``: a :class:`aipf.md.scheduler.PbsSpec` to submit with, or ``None`` to run here.
    ``job_command``: the command the batch job runs (required with ``pbs``).
    """
    point = _point(point)
    deck = deck_named(template)
    derived: dict[str, Any] = {}
    if system.md_settings:
        if campaign is None:
            raise ValueError(
                f"system {system.name!r} declares engine settings for "
                f"{sorted(system.md_settings)}; name the campaign this run "
                f"belongs to")
        derived.update(settings_values(system.md(campaign), point))
    elif campaign is not None:
        system.md(campaign)                      # raises, naming what is declared
    derived.update(point_values(point))
    derived.update(values)
    if pbs is not None and not job_command:
        raise ValueError("a pbs run submits a command, and none was given")

    declared = declared_potential(system)
    model_route = "PAIR" not in derived and declared is not None
    potential = None
    if model_route:
        _device(device, submitting=pbs is not None)
        strip_environment()                      # before the model's libraries are imported
        potential, typed = _resolve_potential(system, declared)
        derived["PAIR"] = _model_pair(potential, typed)
    elif "PAIR" in derived and _names_the_model_style(str(derived["PAIR"])):
        raise ValueError(
            f"the declared values write the {UNIFIED_STYLE!r} pair style themselves; the model "
            f"route writes its pair lines from the potential the system declares under "
            f"defaults['md'][{POTENTIAL_KEY!r}], so drop PAIR from the values")
    route = _MODEL_ROUTE if model_route else _PAIR_ROUTE

    if lammps is None:
        from aipf.site import Site
        lammps = Site.load().require("lammps").lammps
    text = fill(deck, point, system, derived, dt_equil_ps=dt_equil_ps)
    check_deck(text, route=route)

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    deck_path = out / DECK_FILE
    deck_path.write_text(text, encoding="utf-8")
    diagnosis = Diagnosis((alloc_conf(), deck_styles(lammps, text)))
    if model_route:
        launch = plan(_MODEL_ROUTE, log=LOG_FILE, gpus=1)
        command = f"in process: lammps.lammps(cmdargs={list(launch.args)})"
    else:
        launch = plan(_PAIR_ROUTE, log=LOG_FILE, gpus=None, deck_path=DECK_FILE,
                      uses_model=False)
        command = launch.command(lammps)

    result, job_path, job_id = None, None, None
    if not dry_run and diagnosis.can_run is not True:
        raise RuntimeError(f"the pre-flight did not pass, nothing was run or submitted:\n{diagnosis}")
    if pbs is not None:
        job_path, job_id = _submit(pbs, out=out, command=job_command, dry_run=dry_run,
                                   lammps=lammps, potential=potential, system=system.name)
    elif not dry_run:
        if model_route:
            result = run_deck(launch, text, potential=potential, workdir=out)
        else:
            result = run_deck(launch, text, binary=lammps, workdir=out)
    record = {"system": system.name, "template": template, "campaign": campaign,
              "point": asdict(point), "values": {k: str(v) for k, v in values.items()},
              "deck": DECK_FILE, "command": command, "cwd": str(out), "route": route,
              "device": "cuda" if model_route else "host",
              "potential": None if potential is None else potential.to_meta_fields()["potential"],
              "dry_run": dry_run, "diagnosis": diagnosis.to_records(),
              "job": None if job_path is None else {"file": JOB_FILE, "id": job_id},
              "result": None if result is None else asdict(result)}
    (out / RECORD_FILE).write_text(json.dumps(record, indent=1, default=str))
    return RunRecord(deck_path=deck_path, command=command, diagnosis=diagnosis,
                     result=result, route=route, job_path=job_path, job_id=job_id)


def _submit(pbs: Any, *, out: Path, command: Sequence[str], dry_run: bool,
            lammps: str | Path, potential: Any, system: str) -> tuple[Path, str | None]:
    """Write ``out/job.pbs`` and submit it unless ``dry_run``; the job is handed the site facts this
    submission resolved, so it runs the same binary and the same potential."""
    from aipf.md.scheduler import batch_job, job_name
    from aipf.site import FACTS, per_system_variable
    env = {FACTS["lammps"]: str(lammps)}
    if potential is not None:
        env[per_system_variable("mace_potential", system)] = str(potential.path)
    job = batch_job(pbs, name=job_name(f"md-{out.name}"), command=command, workdir=out,
                    env=env, script_name=JOB_FILE, log=JOB_LOG)
    backend = pbs.backend()
    if dry_run:
        return backend.write(job), None
    handle = backend.submit(job)
    (out / JOB_RECORD).write_text(json.dumps(handle.to_record(), indent=1))
    return job.script_path, handle.job_id
