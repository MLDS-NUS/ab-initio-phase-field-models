"""Driving the molecular-dynamics engine: the launch vocabulary, the two routes and their checks.

``in_process``
    The engine runs as a library in this interpreter. The model is loaded into the interface
    before any command, and the pair style carries a literal token instead of a path.
``input_file``
    A separate executable reads a deck. The pair style names the model's path and a
    ghost-neighbour flag, and ``-sf`` supplies the accelerator suffix.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

from aipf.md.potentials import Potential

__all__ = [
    "BROKEN_STYLES",
    "EngineError",
    "KOKKOS_PACKAGE",
    "REMOVED_FROM_ENVIRONMENT",
    "ROUTES",
    "UNIFIED_TOKEN",
    "Launch",
    "Result",
    "check_deck",
    "clean_environment",
    "exit_cleanly",
    "kokkos_args",
    "normalise_route",
    "pair_commands",
    "plan",
    "run",
    "strip_environment",
    "unavailable_reason",
]


# --- The vocabulary ---

#: The two ways the engine is driven.
ROUTES = frozenset({"in_process", "input_file"})

#: Accepted spellings, mapped to the canonical route name.
_ROUTE_ALIASES: dict[str, str] = {
    "in-process": "in_process",
    "inprocess": "in_process",
    "library": "in_process",
    "python": "in_process",
    "input-file": "input_file",
    "inputfile": "input_file",
    "deck": "input_file",
    "subprocess": "input_file",
}

#: The unified pair style, bare; the accelerator suffix is a separate fact.
UNIFIED_STYLE = "mliap"

#: The accelerator suffix, as a command-line value and as a style-name tail.
ACCELERATOR = "kk"

#: The literal token standing where a model path would go on the in-process route.
UNIFIED_TOKEN = "EXISTS"

#: The ghost-neighbour flag written after the model path on the input-file route.
GHOST_NEIGHBOURS_OFF = "0"

#: The only accelerator package combination that runs (the other corners: _KOKKOS_CORNERS).
KOKKOS_PACKAGE: tuple[str, ...] = ("newton", "on", "neigh", "half")

#: Environment variables a run must not inherit, removed by name and recorded on the launch.
REMOVED_FROM_ENVIRONMENT: tuple[str, ...] = ("PYTORCH_CUDA_ALLOC_CONF",)

_REMOVAL_REASONS: dict[str, str] = {
    "PYTORCH_CUDA_ALLOC_CONF": (
        "the expanding-segments allocator is correct for training and breaks "
        "the run, with a host-pointer failure deep in the tensor library"
    ),
}

#: Pair styles that exist and do not run, refused by name with the reason.
BROKEN_STYLES: dict[str, str] = {
    f"mace/{ACCELERATOR}": (
        "the dedicated accelerated pair style is broken upstream: it requires "
        "newton on, while the accelerator's device neighbour list on that "
        "path requires newton off, and no build reconciles them. Use the "
        f"unified {UNIFIED_STYLE} style, which accepts newton on with a half "
        "list and requests the full list it needs internally"
    ),
}

#: The measured failure at each (newton, neigh) corner of the accelerator package.
_KOKKOS_CORNERS: dict[tuple[str, str], str] = {
    ("on", "half"): "runs",
    ("off", "half"): "the unified pair style refuses: it requires newton on",
    ("on", "full"): ("the accelerator refuses at the package line itself: a "
                     "full device list requires newton off"),
    ("off", "full"): ("the unified pair style refuses, for the same reason "
                      "as the half list with newton off"),
}


class EngineError(ValueError):
    """A run that cannot be set up, and why. A run that starts and then fails is a ``Result``."""


def normalise_route(name: object) -> str:
    """The canonical route name, or a refusal naming what was given."""
    if not isinstance(name, str):
        raise EngineError(
            f"a route is named by a string, got {type(name).__name__}")
    key = name.strip().replace(" ", "_")
    canonical = _ROUTE_ALIASES.get(key, key)
    if canonical not in ROUTES:
        raise EngineError(
            f"unknown route {name!r}; this module drives "
            f"{sorted(ROUTES)} and accepts "
            f"{sorted(_ROUTE_ALIASES)} as spellings of them")
    return canonical


# --- The environment ---


def clean_environment(
    env: Mapping[str, str],
) -> tuple[dict[str, str], tuple[tuple[str, str], ...]]:
    """A copy of ``env`` without the variables a run must not inherit.

    Returns the copy and the removed ``(name, value)`` pairs."""
    kept = dict(env)
    removed = []
    for name in REMOVED_FROM_ENVIRONMENT:
        if name in kept:
            removed.append((name, kept.pop(name)))
    return kept, tuple(removed)


def strip_environment(
    env: dict[str, str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Remove those variables from ``env`` (default: this process's environment), in place.

    The in-process route's call, made before the first import. Returns what was removed."""
    target = os.environ if env is None else env
    removed = []
    for name in REMOVED_FROM_ENVIRONMENT:
        if name in target:
            removed.append((name, target[name]))
            del target[name]
    return tuple(removed)


def removal_reason(name: str) -> str:
    """Why a variable is removed, for a report that names it."""
    return _REMOVAL_REASONS.get(name, "")


# --- The command line ---


def _count(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EngineError(f"{name} is a whole number, got {value!r}")
    if value < 1:
        raise EngineError(f"{name} must be at least 1, got {value}")
    return value


def kokkos_args(*, gpus: int | None = 1, threads: int | None = None) -> tuple[str, ...]:
    """The accelerator arguments in the one combination that runs.

    ``gpus`` = devices this process owns; ``gpus=None`` with ``threads`` runs the accelerator on the
    host (a host-only build), which serves the pair-potential decks only: :func:`plan` refuses it for a
    model run."""
    if gpus is None and threads is None:
        raise EngineError("the accelerator needs devices (gpus) or host threads (threads)")
    enable = ["-k", "on"]
    if gpus is not None:
        enable += ["g", str(_count("gpus", gpus))]
    if threads is not None:
        enable += ["t", str(_count("threads", threads))]
    return tuple(enable) + ("-sf", ACCELERATOR, "-pk", "kokkos") + KOKKOS_PACKAGE


def _check_kokkos(args: Sequence[str]) -> None:
    """Refuse an accelerator package line that is not the one that works."""
    if "-pk" not in args:
        return
    index = list(args).index("-pk")
    tail = list(args[index + 1:])
    if not tail or tail[0] != "kokkos":
        return
    settings = {}
    rest = tail[1:]
        # A repeated keyword takes its LAST value (measured).
    for key, value in zip(rest[::2], rest[1::2]):
        settings[key] = value
    wanted = dict(zip(KOKKOS_PACKAGE[::2], KOKKOS_PACKAGE[1::2]))
    for key, value in wanted.items():
        given = settings.get(key)
        if given is None:
            raise EngineError(
                f"the accelerator package line sets no {key!r}; the only "
                f"combination that runs is {' '.join(KOKKOS_PACKAGE)}")
        if given != value:
            corner = (settings.get("newton", wanted["newton"]),
                      settings.get("neigh", wanted["neigh"]))
            measured = _KOKKOS_CORNERS.get(
                corner, "the run fails at setup, and the message names "
                        "neither this line nor this package")
            raise EngineError(
                f"the accelerator package line sets {key} {given!r} and the "
                f"only combination that runs is "
                f"{' '.join(KOKKOS_PACKAGE)}. With "
                f"newton {corner[0]} neigh {corner[1]}, {measured}")


def _has_accelerator(args: Sequence[str]) -> bool:
    """Whether the command line both enables (``-k on``) and applies (``-sf``) the accelerator.

    Each without the other fails (measured)."""
    items = list(args)
    pairs = list(zip(items, items[1:]))
    enabled = any(a == "-k" and b == "on" for a, b in pairs)
    suffixed = any(a == "-sf" and b == ACCELERATOR for a, b in pairs)
    return enabled and suffixed


# --- The pair style, which is where the two routes differ ---


def pair_commands(potential: Potential, *, route: str,
                  ghost_neighbours: bool = False,
                  accelerated: bool | None = None) -> str:
    """The ``pair_style`` and ``pair_coeff`` lines that give the deck its interactions.

    In process the style carries the literal token; from an input file it names the model path
    and the ghost-neighbour flag. Coefficients follow the MODEL's element order. ``accelerated``
    defaults to True in process and False from an input file (where ``-sf`` applies it).
    """
    canonical = normalise_route(route)
    if accelerated is None:
        accelerated = canonical == "in_process"
    if not isinstance(potential, Potential):
        raise EngineError(
            f"pair commands are written from a resolved potential, got "
            f"{type(potential).__name__}")
    if not potential.species:
        raise EngineError(
            "the potential declares no elements, so there is no order to "
            "write the pair coefficients in")

    style = UNIFIED_STYLE + (f"/{ACCELERATOR}" if accelerated else "")
    if canonical == "in_process":
        if ghost_neighbours:
            raise EngineError(
                "the ghost-neighbour flag is written after a model path, and "
                "the in-process route writes no path: the interface already "
                "holds the model. Ask for it on the input-file route")
        head = f"pair_style      {style} unified {UNIFIED_TOKEN}"
    else:
        flag = "1" if ghost_neighbours else GHOST_NEIGHBOURS_OFF
        head = (f"pair_style      {style} unified "
                f"{potential.path} {flag}")
    labels = " ".join(potential.species)
    return f"{head}\npair_coeff      * * {labels}"


def check_deck(text: str, *, route: str) -> None:
    """Refuse a deck whose interactions cannot run on this route.

    Refused: a broken-upstream style; a model path AND the token; the token on the input-file
    route; a path on the in-process route (it runs, with the deck's model, not the loaded one).
    """
    canonical = normalise_route(route)
    if not isinstance(text, str):
        raise EngineError(f"a deck is text, got {type(text).__name__}")

    for line in text.splitlines():
        words = line.split()
        if len(words) < 2 or words[0] != "pair_style":
            continue
        style = words[1]
        if style in BROKEN_STYLES:
            raise EngineError(
                f"the deck asks for pair style {style!r}: "
                f"{BROKEN_STYLES[style]}")
        if style not in (UNIFIED_STYLE, f"{UNIFIED_STYLE}/{ACCELERATOR}"):
            continue
        rest = words[2:]
        if not rest or rest[0] != "unified":
            continue
        arguments = rest[1:]
        if UNIFIED_TOKEN in arguments and len(arguments) > 1:
            raise EngineError(
                f"the deck writes both a model argument and the literal "
                f"{UNIFIED_TOKEN!r}: {line.strip()!r}. That composite is not "
                f"either working spelling. In process the style takes the "
                f"token ALONE, because the interface already holds the "
                f"model; from an input file it takes the model's path and a "
                f"ghost-neighbour flag, and no token")
        if (canonical == "in_process" and arguments
                and arguments[0] != UNIFIED_TOKEN):
            raise EngineError(
                f"the deck names a model on the in-process route: "
                f"{line.strip()!r}. The model is loaded into the interface "
                f"before any command is parsed, so the style takes "
                f"{UNIFIED_TOKEN!r} where a path would go. This run would "
                f"NOT fail: measured, the deck's model computes and the "
                f"loaded one is discarded silently, so the record would name "
                f"a file the run never used")
        if canonical == "input_file" and arguments[:1] == [UNIFIED_TOKEN]:
            raise EngineError(
                f"the deck writes {UNIFIED_TOKEN!r} on the input-file route: "
                f"{line.strip()!r}. Nothing has loaded a model into the "
                f"interface, because there is no interpreter holding one, so "
                f"the style needs the model's path")


# --- One resolved launch ---


@dataclass(frozen=True)
class Launch:
    """Everything decided before anything starts: the route, the engine ``args``, the child
    ``env`` (empty in process), the ``removed`` variables, ``uses_model`` (makes the accelerator
    mandatory) and ``deck_path`` (input-file route only)."""

    route: str
    args: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    removed: tuple[tuple[str, str], ...] = ()
    uses_model: bool = False
    deck_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", normalise_route(self.route))
        object.__setattr__(self, "args", tuple(str(a) for a in self.args))
        _check_kokkos(self.args)
        if self.uses_model and not _has_accelerator(self.args):
            raise EngineError(
                "a model run needs the accelerator suffix on the command "
                "line. Without it the unified interface reaches for a call "
                "that only the accelerated coupling defines, and the run "
                "stops at the first step with that call's name and nothing "
                "else")
        if self.route == "input_file" and self.deck_path is None:
            raise EngineError(
                "the input-file route reads a deck from a file and none was "
                "named")
        if self.route == "in_process" and self.deck_path is not None:
            raise EngineError(
                f"the in-process route issues commands directly and needs no "
                f"deck file, but one was named ({self.deck_path!r})")

    @property
    def accelerated(self) -> bool:
        """Whether the accelerator suffix is applied to styles."""
        return _has_accelerator(self.args)

    def argv(self, binary: str | Path,
             launcher: Sequence[str] = ()) -> tuple[str, ...]:
        """The whole command for the input-file route, with ``launcher`` in front."""
        if self.route != "input_file":
            raise EngineError(
                f"route {self.route!r} runs in this interpreter and has no "
                f"command line to build")
        return tuple(str(t) for t in launcher) + (str(binary),) + self.args

    def command(self, binary: str | Path,
                launcher: Sequence[str] = ()) -> str:
        """That command as one quoted line, for a record or a job file."""
        return shlex.join(self.argv(binary, launcher))


def plan(route: str, *, log: str | Path | None = None,
         screen: str | None = "none",
         gpus: int | None = 1,
         threads: int | None = None,
         deck_path: str | Path | None = None,
         variables: Mapping[str, Any] | None = None,
         env: Mapping[str, str] | None = None,
         uses_model: bool = True) -> Launch:
    """Resolve a launch: the arguments, the environment, and the checks.

    ``gpus=None`` means no device and is refused with a model; ``threads`` (no-model launches) puts
    the accelerator on the host (:func:`kokkos_args`); ``variables`` become
    ``-var`` arguments; ``env`` (input-file route only) is copied minus the removed variables.
    """
    canonical = normalise_route(route)
    if uses_model and gpus is None:
        raise EngineError(
            "a model run needs a device: the unaccelerated unified interface "
            "is missing a call the model's coupling makes, and on "
            "host threads the coupling refuses the host arrays. Give this launch "
            "a device, or say it runs no model")

    args: list[str] = []
    if screen is not None:
        args += ["-screen", str(screen)]
    if log is not None:
        args += ["-log", str(log)]
    if gpus is not None or threads is not None:
        args += list(kokkos_args(gpus=gpus, threads=threads))
    for name, value in (variables or {}).items():
        if not isinstance(name, str) or not name:
            raise EngineError(f"a variable is named by a string, got {name!r}")
        if value is None:
            raise EngineError(
                f"variable {name!r} has no value; a command-line variable "
                f"with nothing after it consumes the next argument instead")
        args += ["-var", name, str(value)]

    if canonical == "input_file":
        if deck_path is None:
            raise EngineError(
                "the input-file route reads a deck from a file and none was "
                "named")
        args += ["-in", str(deck_path)]
        source = os.environ if env is None else env
        child, removed = clean_environment(source)
    else:
        if deck_path is not None:
            raise EngineError(
                f"the in-process route issues commands directly and needs no "
                f"deck file, but one was named ({str(deck_path)!r})")
        if env is not None:
            raise EngineError(
                "the in-process route has no child to hand an environment "
                "to. Its variables are stripped from this process instead, "
                "before the first import, by strip_environment()")
        child, removed = {}, ()

    return Launch(route=canonical, args=tuple(args), env=child,
                  removed=removed, uses_model=uses_model,
                  deck_path=None if canonical == "in_process"
                  else str(deck_path))


# --- What came back ---


@dataclass(frozen=True)
class Result:
    """One finished attempt, whether or not it worked.

    A run that starts and then fails is recorded here, not raised."""

    status: str
    route: str
    seconds: float
    n_atoms: int | None = None
    returncode: int | None = None
    log: str | None = None
    command: str | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the run finished."""
        return self.status == "ok"

    def to_meta_fields(self) -> dict[str, Any]:
        """The metadata keys only the run can fill: ``status`` and, when known, ``n_atoms``."""
        fields: dict[str, Any] = {"status": self.status}
        if self.n_atoms is not None:
            fields["n_atoms"] = self.n_atoms
        return fields


# --- Whether this interpreter, on this machine, can run at all ---


def unavailable_reason(route: str, *,
                       binary: str | Path | None = None) -> str | None:
    """Why this route cannot run here, or ``None`` if it can.

    In process needs the engine's bindings importable here; input file needs an executable.
    """
    canonical = normalise_route(route)
    if canonical == "in_process":
        import importlib.util

        if importlib.util.find_spec("lammps") is None:
            return ("the engine's Python bindings are not importable in this "
                    "interpreter. They are installed by the engine's own "
                    "build, into the environment that build was linked "
                    "against, so this route runs only from that environment")
        return None

    if binary is None:
        return "no engine executable was named"
    import shutil

    found = shutil.which(str(binary))
    if found is None:
        candidate = Path(binary)
        if not candidate.is_file():
            return f"no engine executable at {str(binary)!r}"
        if not os.access(candidate, os.X_OK):
            return f"the engine at {str(binary)!r} is not executable"
    return None


# --- Running ---


def run(launch: Launch, deck: str, *,
        potential: Potential | None = None,
        binary: str | Path | None = None,
        launcher: Sequence[str] = (),
        workdir: str | Path | None = None,
        timeout_s: float | None = None) -> Result:
    """Run one deck VERBATIM, and say what happened.

    Refusals before start are raised; a run that starts and then fails is recorded on the result.
    """
    if not isinstance(launch, Launch):
        raise EngineError(
            f"run takes a resolved launch, got {type(launch).__name__}")
    check_deck(deck, route=launch.route)
    if launch.uses_model and potential is None:
        raise EngineError(
            "this launch says it runs a model and none was given")
    if potential is not None and not launch.uses_model:
        raise EngineError(
            "a potential was given to a launch that says it runs no model, "
            "so nothing would load it and the deck's own interactions would "
            "run instead, silently")

    if launch.route == "in_process":
        return _run_in_process(launch, deck, potential=potential,
                               workdir=workdir)
    return _run_input_file(launch, deck, binary=binary, launcher=launcher,
                           workdir=workdir, timeout_s=timeout_s)


def _log_from(args: Sequence[str]) -> str | None:
    items = list(args)
    for name, value in zip(items, items[1:]):
        if name == "-log":
            return value
    return None


def _run_in_process(launch: Launch, deck: str, *,
                    potential: Potential | None,
                    workdir: str | Path | None) -> Result:
    """Drive the engine as a library: activate the interface, load the model, then issue commands."""
    reason = unavailable_reason("in_process")
    if reason is not None:
        raise EngineError(f"cannot run in this interpreter: {reason}")
    if potential is not None and potential.model is None:
        raise EngineError(
            "the potential carries no loaded model. Resolve it first: the "
            "load is the expensive step and doing it twice can answer "
            "differently")

    import lammps
    import lammps.mliap

    started = time.monotonic()
    here = Path.cwd()
    try:
        if workdir is not None:
            os.chdir(str(workdir))
        try:
            engine = lammps.lammps(cmdargs=list(launch.args))
        except OSError as exc:
            raise EngineError(
                f"the engine's shared library failed to load: {exc}. This is "
                f"loader order, not a missing install: the library search "
                f"path reached a different copy of a system library first. "
                f"Put the environment's own library directory ahead of "
                f"whatever the session inherited") from exc
        try:
            if potential is not None:
                lammps.mliap.activate_mliappy_kokkos(engine)
                lammps.mliap.load_unified_kokkos(potential.model)
            engine.commands_string(deck)
            n_atoms = int(engine.get_natoms())
        finally:
            engine.close()
    except EngineError:
        raise
    except Exception as exc:                      # noqa: BLE001
        return Result(status=f"error: {exc}", route=launch.route,
                      seconds=time.monotonic() - started,
                      log=_log_from(launch.args),
                      detail=traceback.format_exc())
    finally:
        os.chdir(here)

    return Result(status="ok", route=launch.route,
                  seconds=time.monotonic() - started,
                  n_atoms=n_atoms, log=_log_from(launch.args))


def _run_input_file(launch: Launch, deck: str, *,
                    binary: str | Path | None,
                    launcher: Sequence[str],
                    workdir: str | Path | None,
                    timeout_s: float | None) -> Result:
    """Write the deck where the launch says, and run the executable on it."""
    if binary is None:
        raise EngineError(
            "the input-file route runs an executable and none was named")
    reason = unavailable_reason("input_file", binary=binary)
    if reason is not None:
        raise EngineError(f"cannot run here: {reason}")

    root = Path.cwd() if workdir is None else Path(workdir)
    assert launch.deck_path is not None                # by construction
    target = root / launch.deck_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(deck, encoding="utf-8")

    argv = launch.argv(binary, launcher)
    started = time.monotonic()
    try:
        finished = subprocess.run(list(argv), cwd=str(root), env=launch.env,
                                  capture_output=True, text=True,
                                  timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return Result(status=f"error: exceeded {timeout_s} s",
                      route=launch.route, seconds=time.monotonic() - started,
                      log=_log_from(launch.args),
                      command=launch.command(binary, launcher))
    except OSError as exc:
        return Result(status=f"error: {exc}", route=launch.route,
                      seconds=time.monotonic() - started,
                      log=_log_from(launch.args),
                      command=launch.command(binary, launcher),
                      detail=traceback.format_exc())

    status = "ok" if finished.returncode == 0 else (
        f"error: exit {finished.returncode}")
    return Result(status=status, route=launch.route,
                  seconds=time.monotonic() - started,
                  returncode=finished.returncode,
                  log=_log_from(launch.args),
                  command=launch.command(binary, launcher),
                  detail=(finished.stderr or None)
                  if finished.returncode != 0 else None)


# --- Leaving ---


def exit_cleanly(status: int = 0) -> NoReturn:
    """Flush the streams and leave by ``os._exit``, skipping the accelerator's teardown.

    That teardown aborts on this build and turns a finished run's exit status into a signal.
    """
    try:
        sys.stdout.flush()
    except Exception:                                  # noqa: BLE001
        pass
    try:
        sys.stderr.flush()
    except Exception:                                  # noqa: BLE001
        pass
    os._exit(status)
