"""Resolve a machine-learned potential file to a :class:`Potential` record, or refuse it.

The record is a resolved path, the checksum of its bytes, and what the run interface
reads off the model. Every refusal is one that can be made on a login node before a job
is queued; downgraded fast kernels are recorded and warned about, not refused.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "E3NN_DISTRIBUTION",
    "ENGINES",
    "MLIAP_ATTRIBUTES",
    "REQUIRED_E3NN",
    "Potential",
    "PotentialError",
    "check_e3nn",
    "describe",
    "fallback_kernels",
    "installed_version",
    "md5_of",
    "normalise_engine",
    "resolve",
    "sha256_of",
]

#: Pinned exactly: the archived models deserialise under this version only.
E3NN_DISTRIBUTION = "e3nn"
REQUIRED_E3NN = "0.5.1"

ENGINES = frozenset({"mace-mliap"})

#: Archived manifest spellings -> the canonical name.
_ENGINE_ALIASES: dict[str, str] = {
    "mace": "mace-mliap",
    "mace_mliap": "mace-mliap",
    "mliap": "mace-mliap",
}

#: Archived engines refused by name, each with the reason.
_ENGINES_REFUSED: dict[str, str] = {
    "pet": (
        "it runs under a different pair style, metatomic, backed by its own "
        "runtime; a model saved for it needs torch extension classes that "
        "only that runtime registers, so this interpreter cannot even read "
        "the file (measured: the load fails on an unknown extension type "
        "name). Drive it from whatever provides that runtime"
    ),
}

#: Attributes the run interface reads off a converted model.
MLIAP_ATTRIBUTES: tuple[str, ...] = (
    "element_types", "num_species", "rcutfac", "ndescriptors", "nparams",
)

#: The reference implementation's name as a method, and as a class-name suffix.
_FALLBACK_METHOD = "naive"
_FALLBACK_CLASS_SUFFIX = "Naive"

#: Checksum read size, bytes.
_CHUNK = 1 << 20


class PotentialError(ValueError):
    """A potential that cannot be run, and why."""


def normalise_engine(name: object) -> str:
    """The canonical engine name, or a refusal that names the engine given."""
    if not isinstance(name, str):
        raise TypeError(f"engine must be a string, got {name!r}")
    stripped = name.strip()
    if stripped in _ENGINES_REFUSED:
        raise PotentialError(
            f"engine {stripped!r} is not one this resolver drives: "
            f"{_ENGINES_REFUSED[stripped]}")
    canonical = _ENGINE_ALIASES.get(stripped, stripped)
    if canonical not in ENGINES:
        raise PotentialError(
            f"unknown engine {name!r} (read as {canonical!r}); this resolver "
            f"drives {sorted(ENGINES)}, accepts the archived spellings "
            f"{sorted(_ENGINE_ALIASES)}, and refuses "
            f"{sorted(_ENGINES_REFUSED)} by name")
    return canonical


def sha256_of(path: str | Path) -> str:
    """SHA-256 of the file's bytes (not its modification time)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def md5_of(path: str | Path) -> str:
    """MD5 of the file's bytes: the checksum a system declares its potential by."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_version(distribution: str) -> str | None:
    """The version that is LIVE in this interpreter, or ``None``.

    An already-imported module wins over the distribution metadata; nothing is imported.
    """
    imported = sys.modules.get(distribution)
    version = getattr(imported, "__version__", None)
    if isinstance(version, str) and version:
        return version
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_e3nn(required: str = REQUIRED_E3NN) -> str | None:
    """The problem with the deserialisation library, or ``None`` if none (returns, never raises)."""
    live = installed_version(E3NN_DISTRIBUTION)
    if live is None:
        return (f"{E3NN_DISTRIBUTION} is not installed; the saved models need "
                f"exactly {required} to be read")
    if live != required:
        return (f"{E3NN_DISTRIBUTION} {live} is installed but the saved "
                f"models need exactly {required}; a different version fails "
                f"to deserialise them, with an error that names neither")
    return None


def fallback_kernels(tree: Any) -> tuple[str, ...] | None:
    """Nodes that asked for a fast kernel but hold the reference implementation.

    ``()`` means walked and clean; ``None`` means the object could not be walked.
    """
    walk = getattr(tree, "named_modules", None)
    if not callable(walk):
        return None
    found: list[str] = []
    for name, node in walk():
        requested = getattr(node, "method", None)
        built = getattr(node, "m", None)
        if not isinstance(requested, str) or built is None:
            continue
        if requested == _FALLBACK_METHOD:
            continue
        if type(built).__name__.endswith(_FALLBACK_CLASS_SUFFIX):
            found.append(name)
    return tuple(found)


def _dtype_name(value: object) -> str | None:
    """A plain name for a dtype, whether it arrives as one or as a string."""
    if value is None:
        return None
    text = str(value)
    marker = "torch."
    return text[len(marker):] if text.startswith(marker) else text


def describe(model: Any) -> dict[str, Any]:
    """What the run interface would read off this object (duck-typed on the attributes), or a refusal."""
    missing = [name for name in MLIAP_ATTRIBUTES if not hasattr(model, name)]
    if missing:
        raise PotentialError(
            f"loaded a {type(model).__name__}, which is not in the form the "
            f"run interface takes: it is missing {missing}. The run interface "
            f"reads {list(MLIAP_ATTRIBUTES)}. A trained model has to be "
            f"converted before it can be run")

    species = model.element_types
    if (isinstance(species, (str, bytes)) or
            not isinstance(species, Sequence) or
            not all(isinstance(name, str) for name in species)):
        raise PotentialError(
            f"the model's element list is {species!r}, which is not a "
            f"sequence of element names; it sets the order the pair "
            f"coefficients are written in")
    if not species:
        raise PotentialError(
            "the model declares no elements, so there is no order to write "
            "the pair coefficients in")

    declared = model.num_species
    if declared != len(species):
        raise PotentialError(
            f"the model declares {declared} species but lists {len(species)} "
            f"element names {list(species)}; the count sizes the run "
            f"interface's arrays and the list names the coefficients, so a "
            f"disagreement between them is silent")

    return {
        "species": tuple(species),
        "dtype": _dtype_name(getattr(model, "dtype", None)),
        "cutoff": float(model.rcutfac),
        "fallback_kernels": fallback_kernels(getattr(model, "model", model)),
    }


@dataclass(frozen=True)
class Potential:
    """One resolved potential. ``path`` is symlink-resolved; ``species`` is in the model's
    own order (the pair-coefficient order); ``model`` is the loaded object, outside
    comparison and repr.
    """

    path: Path
    sha256: str
    engine: str
    species: tuple[str, ...]
    dtype: str | None
    cutoff: float
    fallback_kernels: tuple[str, ...] | None
    model: Any = field(default=None, compare=False, repr=False)

    def to_meta_fields(self) -> dict[str, Any]:
        """The two metadata keys this resolution determines, ``engine`` and ``potential``, as JSON data."""
        return {
            "engine": self.engine,
            "potential": {
                "path": str(self.path),
                "sha256": self.sha256,
                "species": list(self.species),
                "dtype": self.dtype,
                "cutoff": self.cutoff,
                "fallback_kernels": (
                    None if self.fallback_kernels is None
                    else list(self.fallback_kernels)),
            },
        }

    def require_species(self, expected: Sequence[str]) -> None:
        """Refuse an element order that disagrees with the caller's."""
        wanted = tuple(expected)
        if wanted != self.species:
            raise PotentialError(
                f"the model's element order is {list(self.species)} but the "
                f"caller declared {list(wanted)}; the pair coefficients are "
                f"written in the model's order, so a disagreement runs "
                f"quietly with the species swapped")


def _torch_load(path: Path) -> Any:
    """Read a saved model (a pickled module, so ``weights_only=False``); torch is imported here only."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def resolve(path: str | Path, *, engine: str,
            require_e3nn: str | None = REQUIRED_E3NN,
            species: Sequence[str] | None = None,
            refuse_fallback_kernels: bool = False,
            loader: Callable[[Path], Any] | None = None,
            md5: str | None = None, remedy: str | None = None) -> Potential:
    """Resolve a potential, or refuse it; checks run cheapest first.

    ``engine`` is required; ``require_e3nn=None`` skips the version check; ``species`` is
    the caller's element order; ``loader`` defaults to ``torch.load``; ``md5``, when given,
    is the checksum the file must have, checked before it is loaded (``remedy`` says where the file is
    declared, for that refusal).
    """
    canonical = normalise_engine(engine)

    given = Path(path)
    if not given.exists():
        raise PotentialError(f"no potential at {given}")
    if not given.is_file():
        raise PotentialError(f"{given} is not a file")
    resolved = given.resolve()

    if md5 is not None:
        found = md5_of(resolved)
        if found != str(md5).strip().lower():
            raise PotentialError(
                f"the potential at {resolved} has md5 {found}, and the system declares "
                f"{md5}: it is a different file from the one the system's data were "
                f"generated with. Point the site's potential at the declared file"
                + (f": {remedy}" if remedy else ""))

    if require_e3nn is not None:
        problem = check_e3nn(require_e3nn)
        if problem is not None:
            raise PotentialError(f"cannot load {resolved}: {problem}")

    read = loader if loader is not None else _torch_load
    try:
        model = read(resolved)
    except Exception as exc:
        detail = ""
        if require_e3nn is not None:
            detail = (f" ({E3NN_DISTRIBUTION} "
                      f"{installed_version(E3NN_DISTRIBUTION)} is installed)")
        raise PotentialError(
            f"cannot load the potential at {resolved}{detail}: "
            f"{type(exc).__name__}: {exc}") from exc

    described = describe(model)

    potential = Potential(
        path=resolved,
        sha256=sha256_of(resolved),
        engine=canonical,
        species=described["species"],
        dtype=described["dtype"],
        cutoff=described["cutoff"],
        fallback_kernels=described["fallback_kernels"],
        model=model,
    )

    if species is not None:
        potential.require_species(species)

    downgraded = potential.fallback_kernels
    if downgraded:
        message = (
            f"the potential at {resolved} was converted where its fast "
            f"kernels could not be loaded, so {len(downgraded)} node(s) fell "
            f"back to the reference implementation: {list(downgraded)}. It "
            f"will run correctly and roughly two and a half times slower. "
            f"Reconvert it somewhere the accelerated library loads")
        if refuse_fallback_kernels:
            raise PotentialError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    return potential
