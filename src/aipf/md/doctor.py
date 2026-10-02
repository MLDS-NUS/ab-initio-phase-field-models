"""What this machine can and cannot run, answered before a campaign starts.

Every check is cheap and made in advance from a login shell. Each has five outcomes
(:class:`Verdict`): OK, DEGRADED (right numbers, extra cost), BROKEN, UNKNOWN (not
inspected, with the reason), and NOT_APPLICABLE (the check does not apply to what the site
declared or the device asked for; it never stops a run). A timeout is UNKNOWN, never a failure. The facts come from one
subprocess in the interpreter that will run the job; how the engine is invoked is not here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aipf.md.potentials import (
    E3NN_DISTRIBUTION,
    REQUIRED_E3NN,
    PotentialError,
)
from aipf.md.potentials import resolve as _resolve_potential

__all__ = [
    "ALLOCATOR_ENV",
    "CHECKS",
    "Diagnosis",
    "Finding",
    "KERNEL_TIMEOUT_S",
    "PROBE_TIMEOUT_S",
    "Probe",
    "TOOL_TIMEOUT_S",
    "Verdict",
    "accelerator_libraries",
    "alloc_conf",
    "batch_client",
    "SITE_CHECKS",
    "deck_families",
    "deck_styles",
    "deserialiser",
    "devices",
    "examine",
    "examine_site",
    "gpu_visible",
    "loader_path",
    "resolve_device",
    "site_environment",
    "kernels",
    "lammps_build",
    "lammps_module",
    "potential",
    "probe",
]

#: The checks, in report order; every report carries every one.
CHECKS: tuple[str, ...] = (
    "allocator",
    "deserialiser",
    "accelerator_libraries",
    "accelerator_kernels",
    "lammps_module",
    "lammps_build",
    "devices",
    "batch_client",
    "potential",
)

#: The allocator setting that is correct for training and fatal for a run.
ALLOCATOR_ENV = "PYTORCH_CUDA_ALLOC_CONF"
_BREAKING_ALLOCATOR_OPTION = "expandable_segments"
_ON = frozenset({"true", "1", "yes"})

#: Distribution metadata the probe reads, without importing the libraries.
_DISTRIBUTIONS: tuple[str, ...] = (
    E3NN_DISTRIBUTION,
    "torch",
    "mace-torch",
    "lammps",
    "nvidia-cublas-cu12",
    "cuequivariance-torch",
    "cuequivariance-ops-torch-cu12",
)

#: Modules the probe looks for without importing them.
_MODULES: tuple[str, ...] = ("torch", "lammps", "cuequivariance_ops_torch")

#: Libraries the probe itself must not have imported.
_HEAVY = ("torch", E3NN_DISTRIBUTION, "cuequivariance_ops_torch")

#: The accelerated kernels, the minimum linear-algebra library they need, and the two
#: interpreter-owned library directories the loader must find.
KERNEL_MODULE = "cuequivariance_ops_torch"
KERNEL_DISTRIBUTION = "cuequivariance-ops-torch-cu12"
CUBLAS_DISTRIBUTION = "nvidia-cublas-cu12"
MIN_CUBLAS = (12, 5)
_ACCELERATOR_DIRS: tuple[str, ...] = ("cublas", "cuda_nvrtc")

#: Path fragments marking another installation of the same CUDA stack (a heuristic).
_CUDA_MARKERS = ("cuda", "cublas", "nvidia", "nvhpc")

#: Measured time cost per model evaluation of a model whose kernels were downgraded.
FALLBACK_COST = "2.49 times"

#: The simulator, as a module, as a library and as a binary.
SIMULATOR_MODULE = "lammps"
SIMULATOR_LIBRARY = "liblammps.so"
SIMULATOR_BINARY = "lmp"
#: The pair style that works, and the one that is broken upstream.
WANTED_PAIR_STYLE = "mliap/kk"
BROKEN_PAIR_STYLE = "mace/kk"
#: The two entry points that load a model before the pair style is parsed.
SIMULATOR_ENTRY_POINTS: tuple[str, ...] = (
    "activate_mliappy_kokkos", "load_unified_kokkos")
_SIMULATOR_SUBMODULE = "lammps.mliap"

#: What the help text calls the device backend, and the packages the run needs.
_DEVICE_BACKEND = "CUDA"
_WANTED_PACKAGES = ("KOKKOS", "ML-IAP")

DEVICE_TOOL = "nvidia-smi"
DEVICE_SELECTION_ENV = "CUDA_VISIBLE_DEVICES"
BATCH_CLIENT = "qsub"

#: Timeouts, in seconds; each reports UNKNOWN when it fires.
PROBE_TIMEOUT_S = 60.0
KERNEL_TIMEOUT_S = 300.0
TOOL_TIMEOUT_S = 30.0


# --- The vocabulary ---


class Verdict(str, Enum):
    """What one check found. ``UNKNOWN``: the question was not answered; ``DEGRADED``: the right
    numbers at extra cost; ``NOT_APPLICABLE``: the check does not apply here (a route the site did
    not declare, or one the device asked for cannot take), which :attr:`Diagnosis.can_run` ignores."""

    OK = "ok"
    DEGRADED = "degraded"
    BROKEN = "broken"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "n/a"


@dataclass(frozen=True)
class Finding:
    """One check's answer: ``check``, ``verdict``, a non-empty ``detail``, a ``remedy``
    (required when BROKEN or DEGRADED), and ``raw`` tool output or ``None``."""

    check: str
    verdict: Verdict
    detail: str
    remedy: str = ""
    raw: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.check, str) or not self.check:
            raise ValueError(f"a finding must name its check, got {self.check!r}")
        if not isinstance(self.verdict, Verdict):
            raise TypeError(
                f"verdict must be a Verdict, got {self.verdict!r}")
        if not self.detail:
            if self.verdict is Verdict.UNKNOWN:
                raise ValueError(
                    f"{self.check}: a check that was not inspected has to say "
                    f"why it was not, and this detail is empty; "
                    f"'not inspected' with no reason is indistinguishable "
                    f"from 'nobody wrote the check'")
            raise ValueError(
                f"{self.check}: a finding must say what it saw, and this "
                f"detail is empty")
        if self.verdict in (Verdict.BROKEN, Verdict.DEGRADED) and not self.remedy:
            raise ValueError(
                f"{self.check}: a {self.verdict.value} finding must carry a "
                f"remedy; the point of inspecting in advance is that there is "
                f"still time to act, and a verdict with no action is a "
                f"complaint")


@dataclass(frozen=True)
class Probe:
    """What one interpreter said about itself. ``facts`` is ``None`` when it could not be
    asked, and ``reason`` then says why; checks read that as "not inspected"."""

    facts: Mapping[str, Any] | None = None
    reason: str = ""
    raw: str = ""


@dataclass(frozen=True)
class Diagnosis:
    """Every check's answer, in one object; checks that were not run are present and say so."""

    findings: tuple[Finding, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for item in self.findings:
            if not isinstance(item, Finding):
                raise TypeError(f"findings must be Findings, got {item!r}")
            if item.check in seen:
                raise ValueError(
                    f"two findings for {item.check!r}; one of them would never "
                    f"be read, and which one is an accident of ordering")
            seen.add(item.check)

    def _with(self, verdict: Verdict) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.verdict is verdict)

    @property
    def passed(self) -> tuple[Finding, ...]:
        return self._with(Verdict.OK)

    @property
    def degraded(self) -> tuple[Finding, ...]:
        return self._with(Verdict.DEGRADED)

    @property
    def broken(self) -> tuple[Finding, ...]:
        return self._with(Verdict.BROKEN)

    @property
    def not_inspected(self) -> tuple[Finding, ...]:
        return self._with(Verdict.UNKNOWN)

    @property
    def not_applicable(self) -> tuple[Finding, ...]:
        return self._with(Verdict.NOT_APPLICABLE)

    @property
    def can_run(self) -> bool | None:
        """``True``, ``False``, or ``None`` when something was not inspected and nothing
        inspected is broken. A check that does not apply counts for nothing."""
        if self.broken:
            return False
        if self.not_inspected:
            return None
        return True

    def find(self, check: str) -> Finding:
        """The answer for one named check; raises ``KeyError`` rather than returning ``None``."""
        for item in self.findings:
            if item.check == check:
                return item
        raise KeyError(
            f"no check named {check!r}; this report carries "
            f"{[f.check for f in self.findings]}")

    def to_records(self) -> list[dict[str, Any]]:
        """Plain data, for a run record, which is written as JSON."""
        return [{"check": f.check, "verdict": f.verdict.value,
                 "detail": f.detail, "remedy": f.remedy, "raw": f.raw}
                for f in self.findings]

    def __str__(self) -> str:
        lines = []
        for f in self.findings:
            lines.append(f"[{f.verdict.value:<13}] {f.check}: {f.detail}")
            if f.remedy:
                lines.append(f"{'':<16}-> {f.remedy}")
        summary = (f"{len(self.passed)} ok, {len(self.degraded)} degraded, "
                   f"{len(self.broken)} broken, {len(self.not_inspected)} "
                   f"not inspected")
        if self.not_applicable:
            summary += f", {len(self.not_applicable)} n/a"
        lines.append(summary)
        return "\n".join(lines)


def _not_inspected(check: str, detail: str) -> Finding:
    return Finding(check, Verdict.UNKNOWN, detail)


# --- Asking an interpreter about itself ---


#: Runs in the interpreter under test and prints one JSON record.
_PROBE_SOURCE = r"""
import ctypes, json, os, sys, sysconfig
import importlib.metadata as _meta
import importlib.util as _util

want = json.loads(sys.argv[1])

versions = {}
for name in want["distributions"]:
    try:
        versions[name] = _meta.version(name)
    except Exception:
        versions[name] = None

modules = {}
for name in want["modules"]:
    try:
        modules[name] = _util.find_spec(name) is not None
    except Exception:
        modules[name] = False

site = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
dirs = {}
for name in want["accelerator_dirs"]:
    where = os.path.join(site, name, "lib")
    dirs[name] = where if os.path.isdir(where) else None

facts = {
    "executable": sys.executable,
    "python_version": ".".join(str(part) for part in sys.version_info[:3]),
    "versions": versions,
    "modules": modules,
    "accelerator_dirs": dirs,
    "ld_library_path": [p for p in
                        os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p],
    "lammps": None,
}

if modules.get(want["simulator_module"]):
    found = {"origin": None, "shared_library": None, "dlopen": "",
             "entry_points": None, "entry_points_error": ""}
    try:
        simulator = __import__(want["simulator_module"])
        found["origin"] = getattr(simulator, "__file__", None)
        beside = os.path.join(os.path.dirname(found["origin"] or ""),
                              want["shared_library"])
        found["shared_library"] = beside if os.path.isfile(beside) else None
        try:
            ctypes.CDLL(found["shared_library"] or want["shared_library"])
            found["dlopen"] = "ok"
        except OSError as exc:
            found["dlopen"] = str(exc)
        try:
            couple = __import__(want["simulator_submodule"], fromlist=["x"])
        except Exception as exc:
            found["entry_points_error"] = "%s: %s" % (type(exc).__name__, exc)
        else:
            found["entry_points"] = [name for name in want["entry_points"]
                                     if hasattr(couple, name)]
    except Exception as exc:
        found["entry_points_error"] = "%s: %s" % (type(exc).__name__, exc)
    facts["lammps"] = found

facts["heavy_imports"] = [name for name in want["heavy"] if name in sys.modules]
print(json.dumps(facts))
"""


def _run(command: Sequence[str], *, timeout: float,
         env: Mapping[str, str] | None = None) -> Any:
    return subprocess.run(list(command), capture_output=True, text=True,
                          timeout=timeout, env=None if env is None else dict(env))


def _child_environment(env: Mapping[str, str] | None) -> dict[str, str]:
    """The child's environment: the caller's (default: this process's), with bytecode writing off."""
    child = dict(os.environ if env is None else env)
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    return child


def probe(python: str | Path | None = None, *, timeout: float = PROBE_TIMEOUT_S,
          env: Mapping[str, str] | None = None,
          runner: Callable[..., Any] | None = None) -> Probe:
    """Ask one interpreter what it has, without importing anything expensive.

    ``python`` defaults to this interpreter; a timeout gives a probe with no facts and a
    reason; ``runner`` is a seam for tests."""
    interpreter = str(python or sys.executable)
    run = runner or _run
    request = json.dumps({
        "distributions": list(_DISTRIBUTIONS),
        "modules": list(_MODULES),
        "accelerator_dirs": list(_ACCELERATOR_DIRS),
        "simulator_module": SIMULATOR_MODULE,
        "simulator_submodule": _SIMULATOR_SUBMODULE,
        "shared_library": SIMULATOR_LIBRARY,
        "entry_points": list(SIMULATOR_ENTRY_POINTS),
        "heavy": list(_HEAVY),
    })
    try:
        result = run([interpreter, "-c", _PROBE_SOURCE, request],
                     timeout=timeout, env=_child_environment(env))
    except subprocess.TimeoutExpired:
        return Probe(None, f"{interpreter} did not answer within "
                           f"{timeout:g} s, which on a shared node means a "
                           f"busy machine as often as a broken one")
    except OSError as exc:
        return Probe(None, f"{interpreter} could not be run: {exc}")
    if result.returncode != 0:
        return Probe(None,
                     f"{interpreter} exited {result.returncode} while being "
                     f"asked what it has",
                     raw=_tail(result.stderr or result.stdout))
    try:
        facts = json.loads(result.stdout)
    except ValueError:
        return Probe(None,
                     f"{interpreter} did not answer with a readable record",
                     raw=_tail(result.stdout))
    return Probe(facts, raw=result.stdout)


def _tail(text: str | None, lines: int = 8) -> str:
    """The last few lines of some output. The cause is usually at the end."""
    if not text:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


# --- The checks ---


def _allocator_options(raw: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for part in raw.split(","):
        key, _, value = part.partition(":")
        options[key.strip().lower()] = value.strip().lower()
    return options


def alloc_conf(environ: Mapping[str, str] | None = None) -> Finding:
    """The allocator setting that is right for training and fatal for a run (never UNKNOWN)."""
    values = os.environ if environ is None else environ
    raw = values.get(ALLOCATOR_ENV)
    if not raw:
        return Finding("allocator", Verdict.OK, f"{ALLOCATOR_ENV} is not set")
    options = _allocator_options(raw)
    if options.get(_BREAKING_ALLOCATOR_OPTION, "") in _ON:
        return Finding(
            "allocator", Verdict.BROKEN,
            f"{ALLOCATOR_ENV} is set to {raw!r}, and "
            f"{_BREAKING_ALLOCATOR_OPTION} crashes this engine inside the "
            f"tensor library",
            remedy=f"unset {ALLOCATOR_ENV} for the run, or drop that one "
                   f"option from it. It is correct for training and only for "
                   f"training, so a shell used for both carries it across")
    return Finding("allocator", Verdict.OK,
                   f"{ALLOCATOR_ENV} is {raw!r}, which does not carry "
                   f"{_BREAKING_ALLOCATOR_OPTION}")


def deserialiser(probed: Probe, *,
                 required: str | None = REQUIRED_E3NN) -> Finding:
    """Whether the library that reads the saved model is at the required version.

    ``required=None`` (an engine that does not use it) is "not inspected"."""
    if required is None:
        return _not_inspected(
            "deserialiser",
            f"not inspected: the engine named does not read its models "
            f"through {E3NN_DISTRIBUTION}, so its version decides nothing here")
    if probed.facts is None:
        return _not_inspected("deserialiser", f"not inspected: {probed.reason}")
    where = probed.facts.get("executable", "the interpreter")
    version = (probed.facts.get("versions") or {}).get(E3NN_DISTRIBUTION)
    if version is None:
        return Finding(
            "deserialiser", Verdict.BROKEN,
            f"{E3NN_DISTRIBUTION} is not installed in {where}, and the saved "
            f"models cannot be read without it",
            remedy=f"install {E3NN_DISTRIBUTION}=={required} into that "
                   f"interpreter, or point the run at the one that has it")
    if version != required:
        return Finding(
            "deserialiser", Verdict.BROKEN,
            f"{E3NN_DISTRIBUTION} {version} is installed in {where} and the "
            f"saved models need exactly {required}; another version fails to "
            f"read them with an error that names neither",
            remedy=f"pin {E3NN_DISTRIBUTION}=={required} in that interpreter")
    return Finding("deserialiser", Verdict.OK,
                   f"{E3NN_DISTRIBUTION} {version} in {where}")


def _version_parts(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.split("."):
        digits = "".join(c for c in piece if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _looks_like_cuda(directory: str) -> bool:
    lowered = directory.lower()
    return any(marker in lowered for marker in _CUDA_MARKERS)


def accelerator_libraries(probed: Probe) -> Finding:
    """Whether the fast kernels could be reached here, asked cheaply.

    Predicts from installs and loader-path order; :func:`kernels` settles it by importing them."""
    if probed.facts is None:
        return _not_inspected("accelerator_libraries",
                              f"not inspected: {probed.reason}")
    facts = probed.facts
    versions = facts.get("versions") or {}
    modules = facts.get("modules") or {}
    dirs = facts.get("accelerator_dirs") or {}
    entries = list(facts.get("ld_library_path") or [])
    problems: list[str] = []

    if not modules.get(KERNEL_MODULE):
        problems.append(
            f"{KERNEL_MODULE} is not installed, so there are no accelerated "
            f"kernels to fall back FROM")
    cublas = versions.get(CUBLAS_DISTRIBUTION)
    wanted = ".".join(str(part) for part in MIN_CUBLAS)
    if cublas is None:
        problems.append(
            f"no {CUBLAS_DISTRIBUTION} is installed, and the kernels need at "
            f"least {wanted}")
    elif _version_parts(cublas) < MIN_CUBLAS:
        problems.append(
            f"{CUBLAS_DISTRIBUTION} {cublas} is installed and the kernels "
            f"need at least {wanted}; measured, the older one exports the "
            f"entry point they call zero times")

    missing = [name for name in _ACCELERATOR_DIRS
               if not dirs.get(name) or dirs[name] not in entries]
    if missing:
        problems.append(
            f"the interpreter's own {' and '.join(missing)} directories are "
            f"not on LD_LIBRARY_PATH, so the loader has nowhere to find what "
            f"the kernels link against")
    else:
        first = min(entries.index(dirs[name]) for name in _ACCELERATOR_DIRS)
        ahead = [p for p in entries[:first] if _looks_like_cuda(p)]
        if ahead:
            problems.append(
                f"{ahead[0]} comes ahead of the interpreter's own directories "
                f"on LD_LIBRARY_PATH and may supply an older library")

    if problems:
        return Finding(
            "accelerator_libraries", Verdict.DEGRADED,
            "; ".join(problems) + ". A model converted in this configuration "
            f"keeps the reference implementation for good, measured at "
            f"{FALLBACK_COST} the cost of the accelerated one. Only an import "
            f"of {KERNEL_MODULE} settles it",
            remedy=f"put the interpreter's own "
                   f"{' and '.join(_ACCELERATOR_DIRS)} directories at the "
                   f"FRONT of LD_LIBRARY_PATH before converting or running, "
                   f"then confirm with the deep check")
    return Finding(
        "accelerator_libraries", Verdict.OK,
        f"{KERNEL_MODULE} is installed with {CUBLAS_DISTRIBUTION} {cublas}, "
        f"and the interpreter's own {' and '.join(_ACCELERATOR_DIRS)} "
        f"directories lead LD_LIBRARY_PATH")


def kernels(python: str | Path | None = None, *,
            timeout: float = KERNEL_TIMEOUT_S,
            env: Mapping[str, str] | None = None,
            runner: Callable[..., Any] | None = None) -> Finding:
    """Whether the fast kernels really load here, by a real import (slow: run when asked)."""
    interpreter = str(python or sys.executable)
    run = runner or _run
    try:
        result = run([interpreter, "-c", f"import {KERNEL_MODULE}"],
                     timeout=timeout, env=_child_environment(env))
    except subprocess.TimeoutExpired:
        return _not_inspected(
            "accelerator_kernels",
            f"not inspected: importing {KERNEL_MODULE} in {interpreter} did "
            f"not finish within {timeout:g} s. A busy node and a broken "
            f"environment look the same from here, so this is not a failure")
    except OSError as exc:
        return _not_inspected(
            "accelerator_kernels",
            f"not inspected: {interpreter} could not be run: {exc}")
    if result.returncode != 0:
        return Finding(
            "accelerator_kernels", Verdict.DEGRADED,
            f"{KERNEL_MODULE} cannot be imported in {interpreter}, so a model "
            f"converted here keeps the reference implementation for good, "
            f"measured at {FALLBACK_COST} the cost of the accelerated one",
            remedy="put the interpreter's own linear-algebra and runtime-"
                   "compiler directories at the front of LD_LIBRARY_PATH, and "
                   "convert again; an already converted model keeps whatever "
                   "was baked into it",
            raw=_tail(result.stderr or result.stdout))
    return Finding("accelerator_kernels", Verdict.OK,
                   f"{KERNEL_MODULE} imports in {interpreter}")


def lammps_module(probed: Probe) -> Finding:
    """Whether the simulator can be driven from here: the module imports, its shared library
    loads, and the model-loading entry points exist."""
    if probed.facts is None:
        return _not_inspected("lammps_module", f"not inspected: {probed.reason}")
    facts = probed.facts
    where = facts.get("executable", "the interpreter")
    if not (facts.get("modules") or {}).get(SIMULATOR_MODULE):
        return Finding(
            "lammps_module", Verdict.BROKEN,
            f"there is no importable {SIMULATOR_MODULE} module in {where}, so "
            f"nothing here can drive a run",
            remedy=f"point the run at the interpreter that has it, or build "
                   f"the simulator with PKG_PYTHON and BUILD_SHARED_LIBS=ON "
                   f"and install its python module into this interpreter. "
                   f"This package detects that build and never attempts it")
    found = facts.get("lammps") or {}
    version = (facts.get("versions") or {}).get(SIMULATOR_MODULE)
    loaded = found.get("dlopen")
    if loaded != "ok":
        if found.get("shared_library"):
            return Finding(
                "lammps_module", Verdict.BROKEN,
                f"{found['shared_library']} is there and will not load: "
                f"{loaded}",
                remedy="put the interpreter's own library directory ahead of "
                       "any site module on LD_LIBRARY_PATH; a failure naming "
                       "some third library is the loader resolving a symbol "
                       "against the wrong copy of it",
                raw=loaded)
        return Finding(
            "lammps_module", Verdict.BROKEN,
            f"the {SIMULATOR_MODULE} module in {where} has no "
            f"{SIMULATOR_LIBRARY} beside it, and loading one by name failed: "
            f"{loaded}",
            remedy=f"rebuild with BUILD_SHARED_LIBS=ON and install the python "
                   f"module from that build, or put {SIMULATOR_LIBRARY} on "
                   f"LD_LIBRARY_PATH",
            raw=loaded)
    entry_points = found.get("entry_points")
    if entry_points is None:
        error = found.get("entry_points_error") or ""
        if error:
            return Finding(
                "lammps_module", Verdict.BROKEN,
                f"{_SIMULATOR_SUBMODULE} cannot be imported in {where}, so a "
                f"model cannot be loaded into the interface before the pair "
                f"style is parsed: {error}",
                remedy=f"build the simulator with its {' and '.join(_WANTED_PACKAGES)} "
                       f"packages and install that build's python module",
                raw=error)
        return _not_inspected(
            "lammps_module",
            f"not inspected: the {SIMULATOR_MODULE} module in {where} loads, "
            f"and nothing looked for its model-loading entry points")
    absent = [name for name in SIMULATOR_ENTRY_POINTS if name not in entry_points]
    if absent:
        return Finding(
            "lammps_module", Verdict.BROKEN,
            f"{_SIMULATOR_SUBMODULE} in {where} is missing "
            f"{' and '.join(absent)}; the model is loaded into the interface "
            f"through those before the pair style is parsed, and the "
            f"names without the suffix are a different, non-accelerated "
            f"interface",
            remedy=f"build the simulator with its "
                   f"{' and '.join(_WANTED_PACKAGES)} packages and install "
                   f"that build's python module")
    return Finding(
        "lammps_module", Verdict.OK,
        f"{SIMULATOR_MODULE} {version} in {where}, {SIMULATOR_LIBRARY} loads, "
        f"and {' and '.join(SIMULATOR_ENTRY_POINTS)} are there")


def _has_style(text: str, style: str) -> bool:
    """Whether ``style`` is listed as a whole token, not as part of a longer name."""
    return any(token == style for token in text.replace(",", " ").split())


def lammps_build(binary: str | Path | None = None, *,
                 timeout: float = TOOL_TIMEOUT_S,
                 runner: Callable[..., Any] | None = None,
                 which: Callable[[str], str | None] | None = None) -> Finding:
    """What the simulator binary was built with, read from its help text; never builds it."""
    locate = which or shutil.which
    if binary is None:
        found = locate(SIMULATOR_BINARY)
        if not found:
            return _not_inspected(
                "lammps_build",
                f"not inspected: no binary named {SIMULATOR_BINARY} on PATH "
                f"and none was named. This module does not build the "
                f"simulator, it detects it, so it has nothing to look at")
        binary = found
    binary = str(binary)
    run = runner or _run
    try:
        result = run([binary, "-h"], timeout=timeout)
    except subprocess.TimeoutExpired:
        return _not_inspected(
            "lammps_build",
            f"not inspected: {binary} did not print its help within "
            f"{timeout:g} s")
    except OSError as exc:
        return _not_inspected("lammps_build",
                              f"not inspected: {binary} could not be run: {exc}")
    if result.returncode != 0:
        return Finding(
            "lammps_build", Verdict.BROKEN,
            f"{binary} exited {result.returncode} instead of printing its own "
            f"help, so it cannot run anything either",
            remedy="check that the binary is complete and that the libraries "
                   "it links against are on the loader path",
            raw=_tail(result.stderr or result.stdout))
    text = f"{result.stdout}\n{result.stderr}"
    banner = next((line.strip() for line in result.stdout.splitlines()
                   if "Large-scale" in line), "")
    headline = f"{binary} ({banner})" if banner else binary
    if not _has_style(text, WANTED_PAIR_STYLE):
        return Finding(
            "lammps_build", Verdict.BROKEN,
            f"{headline} has no {WANTED_PAIR_STYLE} pair style, "
            f"which is the only working route for this engine",
            remedy=f"build the simulator with its "
                   f"{' and '.join(_WANTED_PACKAGES)} packages, a device "
                   f"backend, PKG_PYTHON and BUILD_SHARED_LIBS=ON, and name "
                   f"THAT binary",
            raw=banner)
    backend = next((line.strip() for line in text.splitlines()
                    if "package API" in line), "")
    if not backend:
        return _not_inspected(
            "lammps_build",
            f"not inspected: {binary} lists {WANTED_PAIR_STYLE} and its help "
            f"does not say which backend its accelerator package was built "
            f"with, so whether it can use a device is unknown")
    if _DEVICE_BACKEND not in backend:
        return Finding(
            "lammps_build", Verdict.BROKEN,
            f"{binary} has {WANTED_PAIR_STYLE} but its accelerator package "
            f"was built without {_DEVICE_BACKEND} ({backend!r}); the "
            f"processor-only interface fails inside the run on a missing "
            f"exchange, not at startup",
            remedy=f"rebuild the accelerator package with a "
                   f"{_DEVICE_BACKEND} backend for the device on this node",
            raw=backend)
    note = ""
    if _has_style(text, BROKEN_PAIR_STYLE):
        note = (f", and also {BROKEN_PAIR_STYLE}, which is broken upstream on "
                f"a device and must not be used")
    return Finding(
        "lammps_build", Verdict.OK,
        f"{headline} has {WANTED_PAIR_STYLE} with {backend}{note}")


def devices(*, required: bool = True, environ: Mapping[str, str] | None = None,
            timeout: float = TOOL_TIMEOUT_S,
            runner: Callable[..., Any] | None = None,
            which: Callable[[str], str | None] | None = None) -> Finding:
    """Whether a device is visible, and whether anything can use it.

    ``required`` is the campaign's; an emptied selection variable hides a present device."""
    locate = which or shutil.which
    tool = locate(DEVICE_TOOL)
    if not tool:
        return _not_inspected(
            "devices",
            f"not inspected: no {DEVICE_TOOL} on PATH, so nothing here can "
            f"say what devices this node has")
    run = runner or _run
    try:
        result = run([tool, "-L"], timeout=timeout)
    except subprocess.TimeoutExpired:
        return _not_inspected(
            "devices",
            f"not inspected: {tool} did not answer within {timeout:g} s")
    except OSError as exc:
        return _not_inspected("devices",
                              f"not inspected: {tool} could not be run: {exc}")
    if result.returncode != 0:
        return Finding(
            "devices", Verdict.BROKEN,
            f"{tool} exited {result.returncode}, so this node has no usable "
            f"device even if it has the hardware",
            remedy="run where the driver answers, which on a batch farm means "
                   "submitting to a queue that provides a device",
            raw=_tail(result.stderr or result.stdout))
    listed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    values = os.environ if environ is None else environ
    selection = values.get(DEVICE_SELECTION_ENV)
    if required and selection is not None and not selection.strip():
        return Finding(
            "devices", Verdict.BROKEN,
            f"{DEVICE_SELECTION_ENV} is set to {selection!r}, which hides "
            f"every one of the {len(listed)} devices this node has from "
            f"anything that would use one",
            remedy=f"unset {DEVICE_SELECTION_ENV} or name the device index "
                   f"the run should take")
    if not listed:
        if required:
            return Finding(
                "devices", Verdict.BROKEN,
                f"{tool} lists no device, and the route this engine takes has "
                f"no processor-only path",
                remedy="submit to a queue that provides a device")
        return Finding("devices", Verdict.OK,
                       f"{tool} lists no device, and the campaign said it "
                       f"does not need one")
    return Finding("devices", Verdict.OK,
                   f"{len(listed)} device(s): {listed[0]}")


def batch_client(*, required: bool = False,
                 which: Callable[[str], str | None] | None = None) -> Finding:
    """Whether a campaign can be submitted from here; absence matters only when ``required``."""
    locate = which or shutil.which
    found = locate(BATCH_CLIENT)
    if found:
        return Finding("batch_client", Verdict.OK, f"{found} is on PATH")
    if required:
        return Finding(
            "batch_client", Verdict.BROKEN,
            f"no {BATCH_CLIENT} on PATH, and the campaign asked to submit",
            remedy=f"submit from a node that has the batch client, or run the "
                   f"campaign with the local backend")
    return Finding("batch_client", Verdict.OK,
                   f"no {BATCH_CLIENT} on PATH, so only the local backend is "
                   f"available here")


def potential(path: str | Path | None = None, *, engine: str | None = None,
              resolver: Callable[..., Any] | None = None,
              **kwargs: Any) -> Finding:
    """Whether the model file is one this stack can run, and at what cost.

    Resolution is delegated to the potentials module; its refusals become findings."""
    if path is None:
        return _not_inspected(
            "potential",
            "not inspected: no model file was named, so nothing was resolved")
    if engine is None:
        raise ValueError(
            "a model was named without an engine; the engine is never "
            "defaulted, because assuming one for a record that does not say "
            "is how a cross-check arm ends up mislabelled")
    resolve = resolver or _resolve_potential
    try:
        resolved = resolve(path, engine=engine, **kwargs)
    except PotentialError as exc:
        return Finding(
            "potential", Verdict.BROKEN, str(exc),
            remedy="name a model this stack can run, or convert the one you "
                   "have into the form the run interface takes")
    except ImportError as exc:
        return _not_inspected(
            "potential",
            f"not inspected: the model at {path} could not be read here "
            f"because a library it needs is missing: {exc}")
    downgraded = resolved.fallback_kernels
    if downgraded is None:
        return _not_inspected(
            "potential",
            f"not inspected: {resolved.path} loaded and could not be walked, "
            f"so whether its fast kernels were downgraded is unknown. That is "
            f"not the same answer as a model with none")
    if downgraded:
        return Finding(
            "potential", Verdict.DEGRADED,
            f"{resolved.path} was converted where the fast kernels could not "
            f"be reached: {len(downgraded)} node(s) hold the reference "
            f"implementation ({', '.join(downgraded)}). The numbers are the "
            f"same and the cost is {FALLBACK_COST} higher",
            remedy="convert the model again somewhere the kernels import, and "
                   "check this again on the new file; nothing about the run "
                   "can undo what the conversion baked in")
    return Finding(
        "potential", Verdict.OK,
        f"{resolved.path} is in the run format, {resolved.dtype}, elements "
        f"{list(resolved.species)}, with no downgraded kernels")


# --- One deck against one binary ---

#: Commands whose style word the binary must list, and that word's position in the command.
DECK_STYLE_COMMANDS: dict[str, int] = {"pair_style": 1, "fix": 3, "compute": 3,
                                       "atom_style": 1}


def styles_of(text: str) -> tuple[str, ...]:
    """The styles a deck names (pair, fix, compute, atom), in order, once each; ``hybrid`` sub-styles are not read."""
    found: list[str] = []
    for line in text.splitlines():
        words = line.split("#", 1)[0].split()
        where = DECK_STYLE_COMMANDS.get(words[0]) if words else None
        if where is not None and len(words) > where and words[where] not in found:
            found.append(words[where])
    return tuple(found)


def deck_styles(binary: str | Path, text: str, *,
                timeout: float = TOOL_TIMEOUT_S,
                runner: Callable[..., Any] | None = None) -> Finding:
    """Whether ``binary``'s help lists every style the deck ``text`` names; OK, BROKEN naming the missing
    ones, or UNKNOWN when the help cannot be read."""
    wanted = styles_of(text)
    run = runner or _run
    try:
        result = run([str(binary), "-h"], timeout=timeout)
    except subprocess.TimeoutExpired:
        return _not_inspected("deck_styles", f"not inspected: {binary} did not "
                              f"print its help within {timeout:g} s")
    except OSError as exc:
        return _not_inspected("deck_styles",
                              f"not inspected: {binary} could not be run: {exc}")
    if result.returncode != 0:
        return Finding("deck_styles", Verdict.BROKEN,
                       f"{binary} exited {result.returncode} instead of printing "
                       f"its own help, so it cannot run the deck either",
                       remedy="name a complete simulator binary",
                       raw=_tail(result.stderr or result.stdout))
    help_text = f"{result.stdout}\n{result.stderr}"
    missing = [style for style in wanted if not _has_style(help_text, style)]
    if missing:
        return Finding("deck_styles", Verdict.BROKEN,
                       f"{binary} lists no {', '.join(missing)}, which the deck "
                       f"uses; the run would stop at that command",
                       remedy="name a binary built with the packages that "
                              "provide those styles")
    return Finding("deck_styles", Verdict.OK,
                   f"{binary} lists every style the deck uses: "
                   f"{', '.join(wanted)}")


# --- All of it ---


def examine(*, python: str | Path | None = None,
            prober: Callable[..., Probe] | None = None,
            lammps_binary: str | Path | None = None,
            model: str | Path | None = None,
            engine: str | None = None,
            require_e3nn: str | None = REQUIRED_E3NN,
            deep: bool = False,
            require_device: bool = True,
            require_batch_client: bool = False,
            environ: Mapping[str, str] | None = None,
            env: Mapping[str, str] | None = None,
            which: Callable[[str], str | None] | None = None,
            kernel_runner: Callable[..., Any] | None = None,
            probe_timeout: float = PROBE_TIMEOUT_S,
            kernel_timeout: float = KERNEL_TIMEOUT_S,
            tool_timeout: float = TOOL_TIMEOUT_S) -> Diagnosis:
    """Every check, in one pass, with the ones nobody ran saying so.

    One probe of ``python`` feeds every check that reads one; ``deep`` imports the kernels;
    a ``model`` without an ``engine`` is refused."""
    if model is not None and engine is None:
        raise ValueError(
            "a model was named without an engine; the engine is never "
            "defaulted here")
    ask = prober or probe
    probed = ask(python, timeout=probe_timeout, env=env)

    findings = [
        alloc_conf(environ),
        deserialiser(probed, required=require_e3nn),
        accelerator_libraries(probed),
    ]
    if deep:
        findings.append(kernels(python, timeout=kernel_timeout, env=env,
                                runner=kernel_runner))
    else:
        findings.append(_not_inspected(
            "accelerator_kernels",
            f"not inspected: importing {KERNEL_MODULE} costs seconds to "
            f"minutes, so it is only done when asked for with deep=True"))
    findings.extend([
        lammps_module(probed),
        lammps_build(lammps_binary, timeout=tool_timeout, which=which),
        devices(required=require_device, environ=environ,
                timeout=tool_timeout, which=which),
        batch_client(required=require_batch_client, which=which),
        potential(model, engine=engine),
    ])
    # Built in the order CHECKS declares (a test pins it).
    return Diagnosis(tuple(findings))


# --- The site, end to end: what `aipf md doctor` runs ---

#: The checks of :func:`examine_site`, in report order; ``kokkos_device`` only on a device run,
#: ``accelerator_kernels`` only when deep, then one ``deck_styles:<family>`` per deck family.
SITE_CHECKS: tuple[str, ...] = (
    "python_module", "mliap_style", "potential_loads", "ten_steps", "kokkos_device",
    "accelerator_kernels")
DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")

#: How long the child that imports the simulator, loads the model and runs ten steps may take.
SITE_TIMEOUT_S = 900.0
#: The ten-step run: steps, edge of the cubic two-atom cell (Angstrom), temperature (K).
TEN_STEPS, CELL_EDGE, CELL_T = 10, 2.9, 300.0
_MARK = "AIPF-DOCTOR "
#: The ML-IAP route runs on a device only: a CUDA build's Kokkos refuses to start without one, and
#: on a host-only build the MACE coupling refuses its host arrays (and, with that refusal lifted,
#: the model evaluation crashes). On cpu the model is never started.
CPU_TEN_STEPS = "ML-IAP runs on cuda only, use --device cuda"
NO_POTENTIAL = "no mace_potential declared (needed for ML-IAP decks only)"


def _remedy_for(fact: str) -> str:
    """How to declare one site fact, in the words :class:`aipf.site.MissingSiteFact` uses."""
    from aipf.site import FACTS
    return f"set {FACTS[fact]}, or {fact} = \"/path\" under [site] in aipf.toml"


def gpu_visible(*, environ: Mapping[str, str] | None = None,
                timeout: float = TOOL_TIMEOUT_S,
                runner: Callable[..., Any] | None = None,
                which: Callable[[str], str | None] | None = None) -> tuple[bool, str]:
    """Whether a device is visible to a process started now, and what was seen (never raises)."""
    found = devices(required=True, environ=environ, timeout=timeout, runner=runner, which=which)
    return found.verdict is Verdict.OK, found.detail


def resolve_device(device: str, **kwargs: Any) -> tuple[str, str]:
    """``(cpu|cuda, why)``: ``auto`` is ``cuda`` when a device is visible, else ``cpu``."""
    if device not in DEVICES:
        raise ValueError(f"device is one of {DEVICES}, got {device!r}")
    if device != "auto":
        return device, f"--device {device}"
    visible, seen = gpu_visible(**kwargs)
    return ("cuda" if visible else "cpu"), f"--device auto: {seen}"


def loader_path(prefix: str | Path, inherited: str = "",
                cuda_lib: str | Path | None = None) -> str:
    """The loader path for a child of the environment at ``prefix``: its own cuBLAS and nvrtc,
    its ``lib``, the site CUDA library directory, then whatever was inherited.

    The environment's libraries come first, so an OpenMP or C++ runtime of a site module that
    happens to be on the inherited path (an incompatible ``libgomp`` is the measured case) is
    never the one a library of this environment resolves."""
    root = Path(prefix)
    first: list[str] = []
    for lib in sorted(root.glob("lib/python3*")):
        site = lib / "site-packages" / "nvidia"
        first += [str(site / name / "lib") for name in _ACCELERATOR_DIRS
                  if (site / name / "lib").is_dir()]
    first.append(str(root / "lib"))
    if cuda_lib is not None:
        first.append(str(cuda_lib))
    rest = [p for p in inherited.split(os.pathsep) if p and p not in first]
    return os.pathsep.join(first + rest)


def site_environment(python: str | Path, *, cuda_lib: str | Path | None = None,
                     env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment the doctor's child runs in: the caller's, minus what a run must not inherit,
    with :func:`loader_path` for the interpreter's own prefix."""
    from aipf.md.engine import clean_environment
    child, _removed = clean_environment(_child_environment(env))
    prefix = Path(python).resolve().parent.parent
    child["LD_LIBRARY_PATH"] = loader_path(prefix, child.get("LD_LIBRARY_PATH", ""), cuda_lib)
    child["PYTHONNOUSERSITE"] = "1"
    return child


def _two_atom_deck(potential: Any, *, pair: str) -> str:
    """Two atoms of the model's first two elements in a periodic cube, ten steps of NVE."""
    from ase.data import atomic_masses, atomic_numbers
    species = list(potential.species)
    first, second = species[0], species[1 % len(species)]
    half = CELL_EDGE / 2
    masses = [atomic_masses[atomic_numbers[name]] for name in (first, second)]
    return "\n".join([
        "units           metal",
        "boundary        p p p",
        "atom_style      atomic",
        "atom_modify     map yes",
        f"region          cell block 0 {CELL_EDGE} 0 {CELL_EDGE} 0 {CELL_EDGE}",
        "create_box      2 cell",
        "create_atoms    1 single 0 0 0",
        f"create_atoms    2 single {half} {half} {half}",
        f"mass            1 {masses[0]:.4f}",
        f"mass            2 {masses[1]:.4f}",
        pair.replace(f"* * {' '.join(species)}", f"* * {first} {second}"),
        f"velocity        all create {CELL_T} 4928459 mom yes",
        "timestep        0.001",
        "fix             nve all nve",
        "thermo          5",
        "thermo_style    custom step temp pe etotal",
        f"run             {TEN_STEPS}",
        "",
    ])


def _site_child_main(raw: str) -> None:
    """Runs in the interpreter under test: import the simulator, load the model, run ten steps.

    Prints one marked JSON record of facts and leaves by :func:`aipf.md.engine.exit_cleanly`."""
    import tempfile
    import traceback

    want = json.loads(raw)
    facts: dict[str, Any] = {"executable": sys.executable}

    def finish() -> None:
        print(_MARK + json.dumps(facts), flush=True)
        from aipf.md.engine import exit_cleanly
        exit_cleanly(0)

    # The model is read before the simulator's library is loaded: that library is loaded with
    # its symbols global, and a TorchScript load after it fails inside the allocator (measured).
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        facts["torch"] = {"version": torch.__version__, "cuda": cuda,
                          "device_name": torch.cuda.get_device_name(0) if cuda else None}
    except Exception as exc:  # noqa: BLE001
        facts["torch"] = {"error": f"{type(exc).__name__}: {exc}"}

    potential = None
    if want.get("potential"):
        try:
            from aipf.md.potentials import installed_version, resolve
            potential = resolve(want["potential"], engine="mace-mliap")
            facts["potential"] = {
                "ok": True, "path": str(potential.path), "species": list(potential.species),
                "dtype": potential.dtype, "cutoff": potential.cutoff,
                "fallback_kernels": (None if potential.fallback_kernels is None
                                     else list(potential.fallback_kernels)),
                "e3nn": installed_version(E3NN_DISTRIBUTION)}
        except Exception as exc:  # noqa: BLE001
            facts["potential"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        import lammps
        engine = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
        try:
            config = engine.accelerator_config or {}
            facts["module"] = {
                "ok": True, "origin": getattr(lammps, "__file__", None),
                "version": engine.version(), "packages": list(engine.installed_packages),
                "kokkos_api": list((config.get("KOKKOS") or {}).get("api") or []),
                "mliap_styles": [s for s in engine.available_styles("pair")
                                 if s.split("/")[0] == "mliap"]}
        finally:
            engine.close()
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        facts["module"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "raw": traceback.format_exc()}

    if want.get("run") and potential is not None and facts["module"].get("ok"):
        from aipf.md.engine import pair_commands, plan
        from aipf.md.engine import run as run_deck
        with tempfile.TemporaryDirectory(prefix="aipf-doctor-") as work:
            log = Path(work) / "log.lammps"
            try:
                launch = plan("in_process", log=str(log), screen="none", gpus=1)
                deck = _two_atom_deck(potential, pair=pair_commands(potential, route="in_process"))
                result = run_deck(launch, deck, potential=potential)
                text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
                rows = [line.split() for line in text.splitlines()
                        if line.split()[:1] == [str(TEN_STEPS)]]
                facts["ten_steps"] = {
                    "ok": result.ok, "status": result.status, "n_atoms": result.n_atoms,
                    "seconds": round(result.seconds, 2), "args": list(launch.args),
                    "last_row": rows[-1] if rows else None,
                    "detail": _tail(result.detail or "", 12) or _tail(text, 12)}
            except Exception as exc:  # noqa: BLE001
                facts["ten_steps"] = {"ok": False, "status": f"{type(exc).__name__}: {exc}",
                                      "detail": traceback.format_exc()}
    finish()


def _site_child(python: str, want: Mapping[str, Any], *, env: Mapping[str, str],
                timeout: float, runner: Callable[..., Any] | None) -> tuple[dict | None, str, str, bool]:
    """``(facts, reason, raw, answered)``: what the child said, or ``None`` with why it said nothing;
    ``answered`` is False when it could not be asked (a timeout, an interpreter that does not run),
    True when it ran and died before reporting."""
    run = runner or _run
    command = [python, "-c",
               "import sys; from aipf.md.doctor import _site_child_main; _site_child_main(sys.argv[1])",
               json.dumps(dict(want))]
    try:
        result = run(command, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, (f"{python} did not import the simulator, load the model and run "
                      f"{TEN_STEPS} steps within {timeout:g} s; a busy node and a broken "
                      f"environment look the same from here"), "", False
    except OSError as exc:
        return None, f"{python} could not be run: {exc}", "", False
    out = result.stdout or ""
    marked = [line[len(_MARK):] for line in out.splitlines() if line.startswith(_MARK)]
    raw = _tail(result.stderr or out, 12)
    if not marked:
        return None, (f"{python} exited {result.returncode} before it reported "
                      f"(a crash in the simulator or the model's libraries)"), raw, True
    try:
        return json.loads(marked[-1]), "", raw, True
    except ValueError:
        return None, f"{python} reported an unreadable record", raw, True


def _module_finding(facts: dict | None, reason: str, raw: str) -> Finding:
    if facts is None:
        return _not_inspected("python_module", f"not inspected: {reason}")
    module = facts.get("module") or {}
    where = facts.get("executable", "the interpreter")
    if not module.get("ok"):
        return Finding(
            "python_module", Verdict.BROKEN,
            f"`import lammps` and starting it fail in {where}: {module.get('error')}",
            remedy="build LAMMPS into this environment (bash env/build-lammps.sh); a failure "
                   "naming a third library is the loader resolving it from another installation",
            raw=module.get("raw") or raw)
    return Finding(
        "python_module", Verdict.OK,
        f"lammps {module.get('version')} imports and starts in {where} "
        f"({module.get('origin')}), packages {', '.join(module.get('packages') or [])}")


def _mliap_finding(facts: dict | None, binary: Path | None, help_text: str | None,
                   help_problem: str) -> Finding:
    module_styles = None if facts is None else (facts.get("module") or {}).get("mliap_styles")
    problems = []
    seen = []
    if module_styles is not None:
        if "mliap" in module_styles:
            seen.append(f"the python module lists {', '.join(module_styles)}")
        else:
            problems.append("the python module lists no mliap pair style")
    if binary is None:
        problems.append("no LAMMPS binary is declared")
    elif help_text is None:
        problems.append(help_problem)
    elif _has_style(help_text, UNIFIED_PAIR_STYLE):
        seen.append(f"{binary} lists {UNIFIED_PAIR_STYLE}")
    else:
        problems.append(f"{binary} lists no {UNIFIED_PAIR_STYLE} pair style")
    if problems:
        remedy = ("build LAMMPS with ML-IAP, its Python coupling and Kokkos "
                  "(bash env/build-lammps.sh)")
        if binary is None:
            remedy = _remedy_for("lammps") + "; " + remedy
        return Finding("mliap_style", Verdict.BROKEN, "; ".join(problems + seen), remedy=remedy)
    if module_styles is None:
        return _not_inspected("mliap_style", "not inspected: the python module did not report; "
                              + "; ".join(seen))
    return Finding("mliap_style", Verdict.OK, "; ".join(seen))


def _potential_finding(path: Path | None, facts: dict | None, reason: str) -> Finding:
    if path is None:
        return Finding("potential_loads", Verdict.NOT_APPLICABLE, NO_POTENTIAL)
    if facts is None:
        return _not_inspected("potential_loads", f"not inspected: {reason}")
    found = facts.get("potential") or {}
    if not found.get("ok"):
        return Finding(
            "potential_loads", Verdict.BROKEN, f"{path} does not load: {found.get('error')}",
            remedy=f"declare a MACE model converted for ML-IAP (mace_create_lammps_model "
                   f"--format=mliap), and keep {E3NN_DISTRIBUTION}=={REQUIRED_E3NN} installed")
    kernels_note = ""
    if found.get("fallback_kernels"):
        kernels_note = (f"; {len(found['fallback_kernels'])} node(s) hold the reference "
                        f"implementation of the fast kernels ({FALLBACK_COST} the cost)")
    return Finding(
        "potential_loads", Verdict.OK,
        f"{found.get('path')} loads with {E3NN_DISTRIBUTION} {found.get('e3nn')}: elements "
        f"{found.get('species')}, {found.get('dtype')}, cutoff {found.get('cutoff')}{kernels_note}")


def _ten_steps_finding(device: str, facts: dict | None, reason: str, raw: str,
                       answered: bool, blocked: str) -> Finding:
    if blocked:
        return _not_inspected("ten_steps", f"not inspected: {blocked}")
    if facts is None:
        if not answered:
            return _not_inspected("ten_steps", f"not inspected: {reason}")
        return Finding("ten_steps", Verdict.BROKEN, reason,
                       remedy="read the raw output: the child died inside the simulator or "
                              "the model's libraries", raw=raw)
    steps = facts.get("ten_steps")
    if steps is None:
        return _not_inspected("ten_steps", "not inspected: the simulator or the model did not "
                                           "load, so there was nothing to run")
    if not steps.get("ok") or steps.get("n_atoms") != 2 or not steps.get("last_row"):
        return Finding(
            "ten_steps", Verdict.BROKEN,
            f"two atoms for {TEN_STEPS} steps on {device} did not finish: {steps.get('status')}",
            remedy="read the raw output; the run's command line was "
                   + " ".join(steps.get("args") or []),
            raw=steps.get("detail") or raw)
    row = steps["last_row"]
    return Finding(
        "ten_steps", Verdict.OK,
        f"two atoms ran {TEN_STEPS} steps on {device} in {steps.get('seconds')} s "
        f"(step {row[0]}: T {row[1]} K, pe {row[2]} eV, etotal {row[3]} eV)")


def _kokkos_finding(facts: dict | None, reason: str, visible: bool, seen: str,
                    help_text: str | None, binary: Path | None) -> Finding:
    api = None if facts is None else (facts.get("module") or {}).get("kokkos_api")
    torch_facts = {} if facts is None else (facts.get("torch") or {})
    problems = []
    if not visible:
        problems.append(f"no device is visible ({seen})")
    if api is not None and "cuda" not in [a.lower() for a in api]:
        problems.append(f"the python module's Kokkos backends are {api}, without CUDA")
    if help_text is not None:
        backend = next((line.strip() for line in help_text.splitlines()
                        if "package API" in line), "")
        if backend and _DEVICE_BACKEND not in backend:
            problems.append(f"{binary} says {backend!r}")
    if facts is not None and not torch_facts.get("cuda"):
        problems.append(f"torch in {facts.get('executable')} sees no CUDA device")
    if problems:
        return Finding("kokkos_device", Verdict.BROKEN, "; ".join(problems),
                       remedy="run on a GPU node with a LAMMPS whose Kokkos was built for CUDA "
                              "(bash env/build-lammps.sh --cuda)")
    if api is None:
        return _not_inspected("kokkos_device", f"not inspected: {reason}")
    return Finding("kokkos_device", Verdict.OK,
                   f"Kokkos backends {api}, device {torch_facts.get('device_name')} ({seen})")


#: The unified pair style, bare, as a binary's help lists it.
UNIFIED_PAIR_STYLE = "mliap"


def deck_families(systems: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """``(family, text)`` for every built-in deck, then every deck a system declares
    (``defaults["md"]``) with its declared values, whose pair styles the deck leaves as holes.

    ``systems=None`` reads every system :func:`aipf.system.available` lists."""
    from aipf.md.templates import DECKS
    from aipf import system as systems_module
    families = {deck.name: deck.text for deck in DECKS.values()}
    out = [(name, text) for name, text in sorted(families.items())]
    names = systems_module.available() if systems is None else list(systems)
    for name in names:
        declared = (systems_module.load(name).defaults or {}).get("md") or {}
        for template, spec in sorted(declared.items()):
            values = (spec or {}).get("values") or {}
            text = "\n".join([families.get(template, "")]
                             + [str(v) for v in values.values()])
            out.append((f"{name}/{template}", text))
    return out


def examine_site(site: Any, *, device: str = "auto", deep: bool = False,
                 python: str | Path | None = None,
                 systems: Sequence[str] | None = None,
                 env: Mapping[str, str] | None = None,
                 runner: Callable[..., Any] | None = None,
                 tool_runner: Callable[..., Any] | None = None,
                 which: Callable[[str], str | None] | None = None,
                 timeout: float = SITE_TIMEOUT_S,
                 tool_timeout: float = TOOL_TIMEOUT_S) -> Diagnosis:
    """Whether this site runs the package's molecular dynamics, end to end, on ``device``.

    Reads the declared :class:`aipf.site.Site`: the interpreter (``lammps_python``, else this one)
    imports ``lammps`` and starts it (``python_module``); the module and the binary list the
    ``mliap`` pair style (``mliap_style``); the declared MACE potential loads (``potential_loads``);
    two atoms of its elements run ten steps through :func:`aipf.md.engine.run` on cuda (``ten_steps``;
    on cpu, or with no potential declared, it and an undeclared ``potential_loads`` are n/a);
    on ``cuda``, Kokkos has a CUDA backend and a device is visible (``kokkos_device``); with
    ``deep``, the fast kernels import (``accelerator_kernels``); then the binary lists every style
    of every deck family (``deck_styles:<family>``, :func:`deck_families`). The heavy checks run in
    one child of the interpreter, with :func:`site_environment`, so a crash is a finding.
    """
    if device not in DEVICES:
        raise ValueError(f"device is one of {DEVICES}, got {device!r}")
    interpreter = str(python or site.lammps_python or sys.executable)
    visible, seen = gpu_visible(environ=env, timeout=tool_timeout, runner=tool_runner,
                                which=which)
    resolved = device if device != "auto" else ("cuda" if visible else "cpu")
    child_env = site_environment(interpreter, cuda_lib=site.cuda_lib, env=env)

    binary = site.lammps
    help_text, help_problem = None, ""
    if binary is not None:
        try:
            shown = (tool_runner or _run)([str(binary), "-h"], timeout=tool_timeout, env=child_env)
            if shown.returncode == 0:
                help_text = f"{shown.stdout}\n{shown.stderr}"
            else:
                help_problem = (f"{binary} exited {shown.returncode} instead of printing its "
                                f"help: {_tail(shown.stderr or shown.stdout, 3)}")
        except subprocess.TimeoutExpired:
            help_problem = f"{binary} did not print its help within {tool_timeout:g} s"
        except OSError as exc:
            help_problem = f"{binary} could not be run: {exc}"

    # The model is started only on cuda with a declared potential and a visible device.
    blocked = ""
    if resolved == "cuda" and not visible:
        blocked = f"--device cuda and no device is visible ({seen})"
    run = resolved == "cuda" and site.mace_potential is not None and not blocked
    want = {"potential": None if site.mace_potential is None else str(site.mace_potential),
            "run": run, "device": resolved}
    facts, reason, raw, answered = _site_child(interpreter, want, env=child_env,
                                               timeout=timeout, runner=runner)
    findings = [
        _module_finding(facts, reason, raw),
        _mliap_finding(facts, binary, help_text, help_problem),
    ]
    if site.mace_potential is None:
        # No potential declared: the site runs no ML-IAP deck, so what only that route needs
        # (the python module, the mliap style) is reported and does not stop the pair decks.
        findings = [Finding(f.check, Verdict.NOT_APPLICABLE,
                            f"{f.detail} (needed for ML-IAP decks only; {NO_POTENTIAL})",
                            f.remedy, f.raw)
                    if f.verdict in (Verdict.BROKEN, Verdict.UNKNOWN) else f for f in findings]
    findings.append(_potential_finding(site.mace_potential, facts, reason))
    if resolved == "cpu":
        findings.append(Finding("ten_steps", Verdict.NOT_APPLICABLE, CPU_TEN_STEPS))
    elif site.mace_potential is None:
        findings.append(Finding("ten_steps", Verdict.NOT_APPLICABLE, NO_POTENTIAL))
    else:
        findings.append(_ten_steps_finding(resolved, facts, reason, raw, answered, blocked))
    if resolved == "cuda":
        findings.append(_kokkos_finding(facts, reason, visible, seen, help_text, binary))
    if deep:
        findings.append(kernels(interpreter, env=child_env))
    for family, text in deck_families(systems):
        if binary is None:
            found = _not_inspected("deck_styles", f"not inspected: no LAMMPS binary is "
                                                  f"declared ({_remedy_for('lammps')})")
        else:
            found = deck_styles(binary, text, timeout=tool_timeout, runner=tool_runner)
        findings.append(Finding(f"deck_styles:{family}", found.verdict, found.detail,
                                found.remedy, found.raw))
    return Diagnosis(tuple(findings))
