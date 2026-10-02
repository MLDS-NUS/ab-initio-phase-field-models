"""A molecular-dynamics request (:class:`StatePoint`) and a campaign (a list of them).

A state point says what to simulate, never what came back: outcomes are recorded by
:mod:`aipf.data.meta`. There are no defaults; only ``P``, ``n_atoms`` and ``dump_every_ps``
are nullable. A campaign's only rule is distinct tags.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, replace
from typing import Any

from aipf.data.meta import ENSEMBLES, GEOMETRIES

__all__ = [
    "DUMP_FROM",
    "PRESSURE_CONTROLLED",
    "StatePoint",
    "campaign_from_records",
    "campaign_to_records",
    "normalise_ensemble",
    "normalise_geometry",
]

#: When the dump starts: ``"prod"``, ``"equil"`` (keeps the melt frames), ``"none"`` (no trajectory).
DUMP_FROM = frozenset({"prod", "equil", "none"})

#: Ensembles in which a target pressure is set rather than measured.
PRESSURE_CONTROLLED = frozenset({"NPT", "NPT_z"})

#: Archived ensemble spellings -> canonical names in :data:`aipf.data.meta.ENSEMBLES`.
_ENSEMBLE_ALIASES: dict[str, str] = {
    "npt": "NPT",
    "npt_iso": "NPT",
    "npt_z": "NPT_z",
    "nvt": "NVT",
    "nve": "NVE",
    "langevin": "NVT",
    # Positional dynamics with no inertia, kept apart from "langevin".
    "brownian": "langevin_overdamped",
    "overdamped": "langevin_overdamped",
}

#: Archived run-shape names -> canonical geometry (the protocol part is dropped).
_GEOMETRY_ALIASES: dict[str, str] = {
    "slab_meltfirst": "slab",
    "slab_overdamped": "slab",
    "homogeneous_brownian": "cube",
    "spinodal_cube": "cube",
    "cube_mix": "cube",
    "eos_grid": "eos",
}


def _normalise(name: object, aliases: dict[str, str], allowed: frozenset[str],
               what: str) -> str:
    """One canonical name, or a refusal that names both sides."""
    if not isinstance(name, str):
        raise TypeError(f"{what} must be a string, got {name!r}")
    stripped = name.strip()
    canonical = aliases.get(stripped, stripped)
    if canonical not in allowed:
        raise ValueError(
            f"unknown {what} {name!r} (read as {canonical!r}); "
            f"expected one of {sorted(allowed)} "
            f"or an archived spelling {sorted(aliases)}")
    return canonical


def normalise_ensemble(name: object) -> str:
    """The canonical ensemble name for an archived spelling."""
    return _normalise(name, _ENSEMBLE_ALIASES, ENSEMBLES, "ensemble")


def normalise_geometry(name: object) -> str:
    """The canonical geometry name for an archived spelling."""
    return _normalise(name, _GEOMETRY_ALIASES, GEOMETRIES, "geometry")


def _number(value: object, what: str) -> float:
    """A finite real number. Booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be a number, got {value!r}")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return value


def _fraction(value: object, what: str) -> float:
    """A composition fraction in the closed interval [0, 1]; both endpoints occur."""
    number = _number(value, what)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{what} must lie in [0, 1], got {number!r}")
    return number


@dataclass(frozen=True)
class StatePoint:
    """One molecular-dynamics request.

    ``T`` in the system's temperature unit; ``dt_ps``, ``equil_ps``, ``prod_ps``,
    ``dump_every_ps`` in ps; ``x`` one fraction, or one per region in geometry order;
    ``dump_from`` in :data:`DUMP_FROM`, ``"none"`` exactly when ``dump_every_ps`` is None;
    ``P`` None only where the ensemble does not control it; ``n_atoms`` None where the
    count is an outcome; ``parent`` the tag this run continues (then ``equil_ps=0``).
    """

    geometry: str
    ensemble: str
    T: float
    x: float | tuple[float, ...]
    dt_ps: float
    equil_ps: float
    prod_ps: float
    dump_every_ps: float | None
    dump_from: str
    seed: int
    P: float | None = None
    n_atoms: int | None = None
    parent: str | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "geometry", normalise_geometry(self.geometry))
        set_(self, "ensemble", normalise_ensemble(self.ensemble))

        set_(self, "T", _number(self.T, "T"))
        if self.T <= 0.0:
            raise ValueError(f"T must be positive, got {self.T!r}")

        if isinstance(self.x, (list, tuple)):
            parts = tuple(_fraction(v, "x") for v in self.x)
            if len(parts) < 2:
                raise ValueError(
                    "a sequence composition describes a build made of regions "
                    f"and needs at least two, got {self.x!r}; a uniform build "
                    "takes a single number")
            set_(self, "x", parts)
        else:
            set_(self, "x", _fraction(self.x, "x"))

        set_(self, "dt_ps", _number(self.dt_ps, "dt_ps"))
        if self.dt_ps <= 0.0:
            raise ValueError(f"dt_ps must be positive, got {self.dt_ps!r}")

        set_(self, "equil_ps", _number(self.equil_ps, "equil_ps"))
        if self.equil_ps < 0.0:
            raise ValueError(
                f"equil_ps must not be negative, got {self.equil_ps!r}")

        set_(self, "prod_ps", _number(self.prod_ps, "prod_ps"))
        if self.prod_ps <= 0.0:
            raise ValueError(f"prod_ps must be positive, got {self.prod_ps!r}")

        if self.dump_from not in DUMP_FROM:
            raise ValueError(
                f"unknown dump_from {self.dump_from!r}, "
                f"expected one of {sorted(DUMP_FROM)}")

        if self.dump_every_ps is None:
            if self.dump_from != "none":
                raise ValueError(
                    f"dump_from is {self.dump_from!r} but no dump cadence was "
                    "given; a run that writes no trajectory says "
                    'dump_from="none"')
        else:
            set_(self, "dump_every_ps",
                 _number(self.dump_every_ps, "dump_every_ps"))
            if self.dump_from == "none":
                raise ValueError(
                    f"dump_from is \"none\" but a cadence of "
                    f"{self.dump_every_ps!r} was given")
            if self.dump_every_ps < self.dt_ps:
                raise ValueError(
                    f"dump_every_ps {self.dump_every_ps!r} is shorter than the "
                    f"timestep {self.dt_ps!r}; a trajectory cannot be written "
                    "more often than it is integrated")
            if self.dump_every_ps > self.prod_ps:
                raise ValueError(
                    f"dump_every_ps {self.dump_every_ps!r} exceeds the "
                    f"production length {self.prod_ps!r}, so the run would "
                    "keep no production frame")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(f"seed must be an integer, got {self.seed!r}")

        if self.P is None:
            if self.ensemble in PRESSURE_CONTROLLED:
                raise ValueError(
                    f"ensemble {self.ensemble!r} controls the pressure, so a "
                    "target pressure is required; leave P unset only where "
                    "the pressure is measured rather than set")
        else:
            set_(self, "P", _number(self.P, "P"))
            if self.P < 0.0:
                raise ValueError(f"P must not be negative, got {self.P!r}")

        if self.n_atoms is not None:
            if isinstance(self.n_atoms, bool) or not isinstance(self.n_atoms, int):
                raise TypeError(
                    f"n_atoms must be an integer, got {self.n_atoms!r}")
            if self.n_atoms <= 0:
                raise ValueError(
                    f"n_atoms must be positive, got {self.n_atoms!r}")

        if self.parent is not None:
            if not isinstance(self.parent, str) or not self.parent.strip():
                raise ValueError(
                    f"parent must name a state point, got {self.parent!r}")
            set_(self, "parent", self.parent.strip())
            if self.equil_ps != 0.0:
                raise ValueError(
                    f"a continuation of {self.parent!r} re-equilibrates for "
                    f"{self.equil_ps!r}; a run that resumes an equilibrated "
                    "configuration takes equil_ps=0")

    @property
    def total_ps(self) -> float:
        """Simulated time, equilibration and production together."""
        return self.equil_ps + self.prod_ps

    @property
    def n_steps(self) -> int:
        """Integrator steps for the whole run at the production timestep."""
        return int(round(self.total_ps / self.dt_ps))

    @property
    def n_frames(self) -> int:
        """Frames the trajectory will hold, counting the one at time zero; 0 when nothing is written."""
        if self.dump_every_ps is None:
            return 0
        span = self.total_ps if self.dump_from == "equil" else self.prod_ps
        return int(round(span / self.dump_every_ps)) + 1

    @property
    def tag(self) -> str:
        """A distinct, readable label; nothing reads physics back out of it."""
        if isinstance(self.x, tuple):
            composition = "-".join(f"{v:.2f}" for v in self.x)
        else:
            composition = f"{self.x:.2f}"
        parts = [self.geometry, f"x{composition}", f"T{self.T:g}"]
        if self.P is not None:
            parts.append(f"P{self.P:g}")
        parts.append(f"s{self.seed}")
        if self.parent is not None:
            parts.append(f"ext{self.prod_ps:g}")
        return "_".join(parts)

    def to_record(self) -> dict[str, Any]:
        """A plain dictionary, ready to be written into a manifest."""
        return {
            "tag": self.tag,
            "geometry": self.geometry,
            "ensemble": self.ensemble,
            "T": self.T,
            "x": list(self.x) if isinstance(self.x, tuple) else self.x,
            "P": self.P,
            "n_atoms": self.n_atoms,
            "dt_ps": self.dt_ps,
            "equil_ps": self.equil_ps,
            "prod_ps": self.prod_ps,
            "dump_every_ps": self.dump_every_ps,
            "dump_from": self.dump_from,
            "seed": self.seed,
            "parent": self.parent,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "StatePoint":
        """Rebuild a request from a manifest record; a ``tag`` key is ignored (it is derived)."""
        unknown = set(record) - set(_RECORD_KEYS) - {"tag"}
        if unknown:
            raise ValueError(
                f"unknown key(s) {sorted(unknown)} in state-point record; "
                f"known keys are {sorted(_RECORD_KEYS)}")
        missing = [k for k in _REQUIRED_RECORD_KEYS if k not in record]
        if missing:
            raise ValueError(
                f"state-point record is missing {sorted(missing)}")
        return cls(**{k: record[k] for k in _RECORD_KEYS if k in record})

    def to_meta_fields(self) -> dict[str, Any]:
        """The metadata keys this request determines, and only those (a partial record)."""
        return {
            "tag": self.tag,
            "geometry": self.geometry,
            "ensemble": self.ensemble,
            "T_K": self.T,
            "P_GPa": self.P,
            "n_atoms": self.n_atoms,
            "dt_ps": self.dt_ps,
            "dump_every_ps": self.dump_every_ps,
            "n_frames": self.n_frames,
            "seed": self.seed,
        }

    def continuation(self, prod_ps: float) -> "StatePoint":
        """A request that resumes this one for a further ``prod_ps``."""
        return replace(self, parent=self.tag, equil_ps=0.0, prod_ps=prod_ps)


#: Record keys, in constructor order.
_RECORD_KEYS: tuple[str, ...] = tuple(
    f.name for f in StatePoint.__dataclass_fields__.values())

#: The keys a record must carry: the fields with no default.
_REQUIRED_RECORD_KEYS: tuple[str, ...] = tuple(
    f.name for f in StatePoint.__dataclass_fields__.values()
    if f.default is MISSING and f.default_factory is MISSING)


def _check_distinct(tags: list[str]) -> None:
    """Refuse a campaign whose tags repeat (two requests would write into one place)."""
    seen: set[str] = set()
    clashes: list[str] = []
    for tag in tags:
        if tag in seen and tag not in clashes:
            clashes.append(tag)
        seen.add(tag)
    if clashes:
        raise ValueError(
            f"campaign has {len(clashes)} repeated tag(s): {sorted(clashes)}")


def campaign_to_records(campaign: list[StatePoint]) -> list[dict[str, Any]]:
    """A campaign as plain records. The list of records IS the manifest."""
    records = [point.to_record() for point in campaign]
    _check_distinct([r["tag"] for r in records])
    return records


def campaign_from_records(records: list[dict[str, Any]]) -> list[StatePoint]:
    """A campaign rebuilt from manifest records."""
    campaign = [StatePoint.from_record(r) for r in records]
    _check_distinct([p.tag for p in campaign])
    return campaign
