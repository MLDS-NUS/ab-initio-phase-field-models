"""A deck, its ``@NAME@`` placeholders, and what a request, a system and a caller fill in.

``@NAME@`` (upper case, delimited both sides) cannot collide with the engine's
``${name}`` or a lone ``@``. The request fills physics (T, seed, dt, step counts),
the system fills material (masses, preparation temperature), the caller the rest,
including ``@P@`` (no unit conversion here). Unfilled, ``None`` and mid-command
empty values are refused; an empty whole-line placeholder drops the line.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aipf.md.request import StatePoint
from aipf.system import System

__all__ = [
    "Deck",
    "DeckError",
    "check",
    "fill",
    "from_file",
    "from_text",
    "placeholders",
    "render",
    "values_from",
]

#: A hole: upper case, delimited both sides, so ``${name}`` and a lone ``@`` pass through.
_PLACEHOLDER = re.compile(r"@([A-Z][A-Z0-9_]*)@")

#: Substitution nesting cap: a value containing its own name is refused, not a hang.
_MAX_DEPTH = 8

class DeckError(ValueError):
    """A deck that cannot be filled, and why."""


def placeholders(text: str) -> tuple[str, ...]:
    """The holes in a deck, in order of first appearance, each once."""
    seen: list[str] = []
    for match in _PLACEHOLDER.finditer(text):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    return tuple(seen)


@dataclass(frozen=True)
class Deck:
    """One deck: text, source, produced dataset, geometry/ensemble, and deck-level defaults."""

    name: str
    text: str
    source: str
    produced: str | None = None
    geometry: str | None = None
    ensemble: str | None = None
    defaults: dict[str, str] = field(default_factory=dict)

    @property
    def placeholders(self) -> tuple[str, ...]:
        """The holes, in order of first appearance."""
        return placeholders(self.text)

    @property
    def prepares(self) -> bool:
        """Whether the deck runs an equilibration stage before production."""
        return "N_EQUIL" in self.placeholders

    @property
    def melts(self) -> bool:
        """Whether that stage runs at the system's preparation temperature (not for a liquid start)."""
        return "T_PREP" in self.placeholders

    @property
    def writes_trajectory(self) -> bool:
        """Whether the deck writes frames: a generated dump slot or its own ``N_DUMP`` cadence."""
        return bool({"N_DUMP", "DUMP_PROD", "DUMP_EQUIL"}
                    & set(self.placeholders))

    @property
    def dumps_from_equilibration(self) -> bool:
        """Whether the deck can start its trajectory before production."""
        return "DUMP_EQUIL" in self.placeholders


def from_text(text: str, *, name: str = "deck",
              source: str = "<text>") -> Deck:
    """A deck someone wrote, as a string."""
    if not isinstance(text, str):
        raise DeckError(f"a deck is text, got {type(text).__name__}")
    return Deck(name=name, text=text, source=source)


def from_file(path: str | Path) -> Deck:
    """A deck someone wrote, read from a file; a missing path and a directory are refused apart."""
    location = Path(path)
    if not location.exists():
        raise DeckError(f"no deck at {location}")
    if not location.is_file():
        raise DeckError(f"{location} is not a file")
    return Deck(name=location.name, text=location.read_text(encoding="utf-8"),
                source=str(location))


def _expand(line: str, values: Mapping[str, Any], where: str,
            depth: int = 0) -> list[str]:
    """One line of a deck, filled, as the lines it becomes; recursive, values may hold placeholders.

    A whole-line placeholder may resolve to nothing and drops the line; elsewhere empty is refused.
    """
    if depth > _MAX_DEPTH:
        raise DeckError(
            f"{where}: {list(placeholders(line))} is still unresolved "
            f"{_MAX_DEPTH} levels into {line.strip()!r}; a value that "
            f"contains its own placeholder cannot be resolved")
    if not _PLACEHOLDER.search(line):
        return [line]
    alone = _PLACEHOLDER.fullmatch(line.strip()) is not None

    def substitute(match: re.Match) -> str:
        name = match.group(1)
        if name not in values:
            raise DeckError(
                f"{where}: nothing to fill @{name}@ with. The deck asks for "
                f"{list(placeholders(line))} on the line {line.strip()!r}; "
                f"{sorted(values)} was given")
        value = values[name]
        if value is None:
            raise DeckError(
                f"{where}: @{name}@ was given None, which is not a value a "
                f"deck can carry. A request that does not set this quantity "
                f"needs a deck that does not ask for it")
        rendered = str(value)
        if rendered == "" and not alone:
            raise DeckError(
                f"{where}: @{name}@ was given nothing, in the middle of "
                f"{line.strip()!r}. An empty value is only allowed where the "
                f"placeholder is the whole line, and then the line is dropped")
        return rendered

    filled = _PLACEHOLDER.sub(substitute, line)
    if not filled.strip():
        return []
    out: list[str] = []
    for part in filled.splitlines():
        out.extend(_expand(part, values, where, depth + 1))
    return out


def render(deck: Deck, values: Mapping[str, Any]) -> str:
    """Fill a deck's holes and change nothing else; deck defaults lose, unused values are ignored."""
    merged = dict(deck.defaults)
    merged.update(values)
    out: list[str] = []
    for line in deck.text.splitlines():
        out.extend(_expand(line, merged, f"deck {deck.name!r}"))
    return "\n".join(out)


def _number(value: float) -> str:
    """``repr(float)``, which round-trips; ``%g`` would round to six significant figures."""
    return repr(float(value))


def _steps(span_ps: float, dt_ps: float) -> int:
    """Integrator steps for a span, rounded, not truncated (``0.7 / 0.1`` is 6.999999999999999)."""
    return int(round(span_ps / dt_ps))


def _preparation(point: StatePoint, system: System) -> tuple[float, float]:
    """(start, end) temperatures of the first stage, one of three forms.

    OFFSET: target + ramp_offset down to target; ABSOLUTE: melt_T held; NEITHER: at the target.
    """
    rules = system.preparation
    if rules.melt_T is not None and rules.ramp_offset is not None:
        raise DeckError(
            f"system {system.name!r} declares both an absolute preparation "
            f"temperature ({rules.melt_T!r}) and an offset "
            f"({rules.ramp_offset!r}). They are alternatives, not a pair: one "
            f"does not depend on the target and the other does, so a system "
            f"declaring both describes two different preparations and nothing "
            f"here can choose between them")
    if rules.melt_T is not None:
        start = end = float(rules.melt_T)
    elif rules.ramp_offset is not None:
        start, end = point.T + float(rules.ramp_offset), point.T
    else:
        start = end = point.T
    # Only the start can land wrong: the end is melt_T or the (positive) target.
    if start <= 0.0:
        raise DeckError(
            f"system {system.name!r} prepares a run at T={_number(start)}, "
            f"which is not a temperature. Target {_number(point.T)}, "
            f"offset {rules.ramp_offset!r}, absolute {rules.melt_T!r}")
    return start, end


def _masses(system: System) -> str | None:
    """A mass line per numeric type, in dump-type order, or ``None`` when no masses are declared.

    A typed species with no mass is refused: the engine needs one per type.
    """
    if not system.masses:
        return None
    if not system.atom_types:
        raise DeckError(
            f"system {system.name!r} declares masses {dict(system.masses)} "
            f"but assigns no numeric types, so there is no type to write a "
            f"mass for. A deck indexes a mass by the type a trajectory dump "
            f"carries, not by the channel name")
    lines = []
    for species, atom_type in sorted(system.atom_types.items(),
                                     key=lambda item: item[1]):
        if species not in system.masses:
            raise DeckError(
                f"system {system.name!r} assigns numeric type {atom_type} to "
                f"{species!r} but declares no mass for it. A dump may carry a "
                f"type that is not a density channel, and the engine still "
                f"needs its mass")
        lines.append(f"mass            {atom_type} "
                     f"{system.masses[species]!r}")
    return "\n".join(lines)


def _dump_block() -> str:
    """The lines that open a trajectory, clock reset first; the block carries its own placeholders."""
    return ("@DUMP_RESET@\n"
            "dump            traj all custom @N_DUMP@ @TRAJECTORY@ "
            "id type x y z\n"
            "@DUMP_MODIFY@")


def values_from(point: StatePoint, system: System, *,
                dt_equil_ps: float | None = None) -> dict[str, str]:
    """Placeholder values the request and the system determine; keys are omitted rather than faked.

    ``dt_equil_ps``: the equilibration timestep, when it differs from the production one.
    """
    dt_equil = point.dt_ps if dt_equil_ps is None else float(dt_equil_ps)
    if dt_equil <= 0.0:
        raise DeckError(f"dt_equil_ps must be positive, got {dt_equil_ps!r}")

    start, end = _preparation(point, system)
    values: dict[str, str] = {
        "T": _number(point.T),
        "T_PREP": _number(start),
        "T_PREP_END": _number(end),
        "SEED": str(point.seed),
        "DT": _number(point.dt_ps),
        "DT_EQUIL": _number(dt_equil),
        "N_EQUIL": str(_steps(point.equil_ps, dt_equil)),
        "N_PROD": str(_steps(point.prod_ps, point.dt_ps)),
        "DUMP_EQUIL": "",
        "DUMP_PROD": "",
    }

    masses = _masses(system)
    if masses is not None:
        values["MASSES"] = masses

    if point.dump_every_ps is not None:
        values["N_DUMP"] = str(_steps(point.dump_every_ps, point.dt_ps))
        slot = "DUMP_EQUIL" if point.dump_from == "equil" else "DUMP_PROD"
        values[slot] = _dump_block()

    return values


def check(deck: Deck, point: StatePoint) -> None:
    """Refuse a deck and a request that describe different runs.

    What a deck asks for is checked for every deck; what it omits only for a shipped one.
    """
    if deck.prepares and point.equil_ps == 0.0:
        raise DeckError(
            f"deck {deck.name!r} has an equilibration stage and the request "
            f"gives it no length. Every archived continuation sets equil_ps "
            f"to zero because the configuration it resumes is already "
            f"equilibrated, and preparing it again would destroy the state "
            f"the leg exists to grow. Use a deck with no preparation stage")
    if deck.writes_trajectory and point.dump_every_ps is None:
        raise DeckError(
            f"deck {deck.name!r} writes a trajectory and the request sets no "
            f"cadence")

    if deck.geometry is None:
        return

    if deck.geometry != point.geometry:
        raise DeckError(
            f"deck {deck.name!r} is for geometry {deck.geometry!r} and the "
            f"request is for {point.geometry!r}")
    if deck.ensemble != point.ensemble:
        raise DeckError(
            f"deck {deck.name!r} integrates {deck.ensemble!r} and the "
            f"request asks for {point.ensemble!r}")
    if not deck.prepares and point.equil_ps > 0.0:
        raise DeckError(
            f"deck {deck.name!r} has no equilibration stage and the request "
            f"asks for {point.equil_ps!r} ps of one, which would be silently "
            f"dropped")
    if not deck.writes_trajectory and point.dump_every_ps is not None:
        raise DeckError(
            f"deck {deck.name!r} writes no trajectory and the request asks "
            f"for frames every {point.dump_every_ps!r} ps")
    if point.dump_from == "equil" and not deck.dumps_from_equilibration:
        raise DeckError(
            f"the request keeps its equilibration frames and deck "
            f"{deck.name!r} has nowhere to put them. Its equilibration runs "
            f"at its own timestep, so those frames would not be on the "
            f"production cadence, and no archived campaign on that protocol "
            f"asked for them")


def fill(deck: Deck, point: StatePoint, system: System,
         values: Mapping[str, Any] | None = None, *,
         dt_equil_ps: float | None = None) -> str:
    """Check a deck against a request, then fill it; the caller's ``values`` win over derived ones."""
    check(deck, point)
    merged = dict(values_from(point, system, dt_equil_ps=dt_equil_ps))
    if values:
        merged.update(values)
    return render(deck, merged)
