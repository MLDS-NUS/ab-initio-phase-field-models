"""Trajectory metadata: an explicit ``ensemble`` and the ``box.varying`` axes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = 1

ENSEMBLES = frozenset({"NVE", "NVT", "NPT", "NPT_z", "langevin_overdamped"})
GEOMETRIES = frozenset({"cube", "slab", "ball", "column", "vext", "eos", "quench"})

#: Axes that may move under each ensemble.
_VARYING_ALLOWED: dict[str, frozenset[str]] = {
    "NVE": frozenset(),
    "NVT": frozenset(),
    "langevin_overdamped": frozenset(),
    "NPT_z": frozenset({"z"}),
    "NPT": frozenset({"x", "y", "z"}),
}



def moving_axes(ensemble: str) -> list[str]:
    """The axes the box moves along under ``ensemble``, sorted: every axis it lets move, since a
    barostat that acts on an axis moves it (``NPT``: x, y and z; ``NPT_z``: z); none for an unknown one."""
    return sorted(_VARYING_ALLOWED.get(ensemble, ()))


#: Ensembles in which the pressure is a controlled variable.
_PRESSURE_CONTROLLED = frozenset({"NPT", "NPT_z"})

_REQUIRED = (
    "schema_version", "system", "tag", "geometry", "ensemble",
    "T_K", "composition", "box", "n_atoms", "dt_ps",
    "dump_every_ps", "n_frames", "engine", "seed", "status",
)


@dataclass
class Meta:
    """One state point's metadata. ``extra`` carries engine-specific keys."""

    schema_version: int
    system: str
    tag: str
    geometry: str
    ensemble: str
    T_K: float
    P_GPa: float | None
    composition: dict
    box: dict
    n_atoms: int
    dt_ps: float
    dump_every_ps: float
    n_frames: int
    engine: str
    potential: dict | None
    seed: int
    status: str
    extra: dict = field(default_factory=dict)


def validate(d: dict) -> list[str]:
    """Problems with a metadata dictionary. Empty means valid."""
    problems: list[str] = []

    for key in _REQUIRED:
        if key not in d:
            problems.append(f"missing required key {key!r}")
    if problems:
        return problems

    if d["schema_version"] != SCHEMA_VERSION:
        problems.append(
            f"schema_version {d['schema_version']} != {SCHEMA_VERSION}")

    ensemble = d["ensemble"]
    if ensemble not in ENSEMBLES:
        problems.append(
            f"unknown ensemble {ensemble!r}, expected one of {sorted(ENSEMBLES)}")
    if d["geometry"] not in GEOMETRIES:
        problems.append(f"unknown geometry {d['geometry']!r}")

    box = d.get("box")
    if not isinstance(box, dict):
        problems.append(f"box must be a mapping, got {box!r}")
        return problems
    varying = set(box.get("varying", []))
    allowed = _VARYING_ALLOWED.get(ensemble)
    if allowed is not None and not varying <= allowed:
        problems.append(
            f"box.varying {sorted(varying)} is impossible under ensemble "
            f"{ensemble!r}, which allows {sorted(allowed)}")

    # Type before value: validate returns problems, it never raises.
    for key in ("n_frames", "dt_ps", "dump_every_ps", "T_K"):
        value = d.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{key} must be a number, got {value!r}")
        elif value <= 0:
            problems.append(f"{key} must be positive, got {value!r}")

    return problems


def supports_pressure_anchor(d: dict) -> bool:
    """Whether ``L_P`` may be computed: a pressure-controlled ensemble with ``P_GPa`` recorded.

    Never raises: "no" whenever the answer is unclear.
    """
    return (d.get("ensemble") in _PRESSURE_CONTROLLED
            and d.get("P_GPa") is not None)
