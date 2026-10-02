"""The front door to the solver: validate the whole declaration once, before anything is paid for.
One spelling for the lower bound (``lo=`` in every mode; ``floor=`` is refused), and a declaration
nothing reads (``hi=`` outside ``"box"``, noise on a deterministic integrator) is refused."""
from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping

import torch

from aipf.system import System

from . import integrators
from .declare import UNDECLARED
from .noise import check_noise_declaration
from .projection import STATE_PROJ_MODES, check_state_projection

#: The integrators :func:`build` constructs a solver for (``"sde"`` selects ``rollout_sde``).
INTEGRATORS = ("euler", "heun", "sde")

#: Keywords this door owns and never forwards under the caller's name.
_RESERVED = ("method", "state_proj", "floor")


@functools.lru_cache(maxsize=None)
def _declarable(integrator: str) -> frozenset:
    """Every keyword a declaration may carry for one integrator, read off the rollout's signature (cached)."""
    fn = (integrators.rollout_sde if integrator == "sde"
          else integrators.rollout_deterministic)
    names = {name for name, p in inspect.signature(fn).parameters.items()
             if p.kind is p.KEYWORD_ONLY}
    names -= set(_RESERVED)
    names.add("lo")
    if integrator == "sde":
        names.add("kBT_noise")
    return frozenset(names)


def _check_keywords(integrator: str, declared: Mapping[str, Any]) -> None:
    """Refuse a keyword no rollout for this integrator would read."""
    allowed = _declarable(integrator)
    for name in declared:
        if name in allowed:
            continue
        if name in ("noise_mode", "sigma_noise", "kBT_noise"):
            raise ValueError(
                f"{name}= is a noise declaration and integrator="
                f"{integrator!r} is deterministic, so nothing would read "
                f"it. Use integrator='sde' to inject noise, or drop the "
                f"declaration.")
        if name == "floor":
            raise ValueError(
                "floor= is the rollout's name for the lower bound; this "
                "door spells it lo= for all three state_proj modes, so "
                "that choosing a mode does not also mean changing the "
                "keyword. Pass lo=.")
        raise ValueError(
            f"{name}= is not a keyword any rollout accepts. Declarable "
            f"here: {sorted(allowed)}.")


def _check_bounds(system: System, state_proj: str,
                  declared: Mapping[str, Any]) -> None:
    """Validate the state-projection declaration in this door's spelling, via ``check_state_projection``."""
    if state_proj is UNDECLARED or state_proj not in STATE_PROJ_MODES:
        check_state_projection(state_proj, UNDECLARED, None)     # raises
    if state_proj == "floor" and "lo" not in declared:
        raise ValueError(
            "state_proj='floor' needs lo=: the per-cell lower bound the "
            "projection holds the field at (0.0 asks for no projection at "
            "all). It is the rollout's floor= under this door's one "
            "spelling. There is no default -- see aipf.solve.declare.")
    if state_proj != "box" and "hi" in declared:
        raise ValueError(
            f"hi= is read only by state_proj='box' (the upper bound of the "
            f"two-sided box); state_proj={state_proj!r} would ignore it. "
            f"Declaring a bound nothing applies is the ambiguity this door "
            f"exists to remove.")
    if state_proj != "domain" and "domain" in declared:
        raise ValueError(
            f"domain= is read only by state_proj='domain' (the trapezoid "
            f"the projection is onto); state_proj={state_proj!r} would "
            f"ignore it.")
    floor = declared["lo"] if state_proj == "floor" else UNDECLARED
    lo = UNDECLARED if state_proj == "floor" else declared.get("lo",
                                                               UNDECLARED)
    check_state_projection(state_proj, floor, declared.get("domain"),
                           lo=lo, hi=declared.get("hi", UNDECLARED))
    domain = declared.get("domain")
    if domain is not None and domain.n_species != system.n_species:
        raise ValueError(
            f"domain= has n_species={domain.n_species} and the declared "
            f"system has n_species={system.n_species}. A trust domain is "
            f"one system's measured data-coverage extremes, and one "
            f"system's corners under another system's rollout is a "
            f"predicate that is false everywhere rather than an error -- "
            f"see aipf.solve.trust_domain.")


def _validate(system: System, integrator: str, state_proj: str,
              declared: Mapping[str, Any]) -> None:
    """Every check solver keywords must pass; run at :func:`build` and again at :meth:`Solver.run`."""
    _check_keywords(integrator, declared)
    _check_bounds(system, state_proj, declared)
    if "clamp_rho" not in declared:
        raise ValueError(
            "clamp_rho must be declared: a positive float (the lower "
            "bound the model's pointwise inputs are clamped at, which is "
            "a guard on what the network SEES, not on what the state is), "
            "or None to run the network unguarded. There is no default -- "
            "see aipf.solve.declare.")
    # Validate the value too, at zero model evaluations.
    integrators._validated_clamp_rho(declared["clamp_rho"])
    if integrator == "sde":
        check_noise_declaration(declared.get("noise_mode", UNDECLARED),
                                declared.get("sigma_noise", UNDECLARED))
        if "kBT_noise" not in declared:
            raise ValueError(
                "integrator='sde' needs kBT_noise=: the noise amplitude, "
                "in the energy unit the model's free energy is in. 0.0 is "
                "the deterministic limit and is bit-for-bit the "
                "deterministic step, so it is a statement rather than an "
                "omission.")


def _rollout_keywords(state_proj: str,
                      declared: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate one declaration into the rollout's own keyword names."""
    kw = dict(declared)
    if state_proj == "floor" and "lo" in kw:
        kw["floor"] = kw.pop("lo")
    return kw


@dataclass(frozen=True)
class Solver:
    """A fully declared, frozen solver (``declared`` is read-only); ``run`` takes what changes per run."""

    system: System
    integrator: str
    dt: float
    state_proj: str
    declared: Mapping[str, Any] = field(default_factory=dict)

    def run(self, model: torch.nn.Module, rho_hat0: torch.Tensor,
            boxes: torch.Tensor, T: torch.Tensor, n_steps: int,
            **per_run: Any) -> torch.Tensor:
        """Advance ``rho_hat0`` for ``n_steps``; ``per_run`` overrides this call only and is re-validated."""
        if rho_hat0.shape[1] != self.system.n_species:
            raise ValueError(
                f"rho_hat0 has {rho_hat0.shape[1]} density channels and "
                f"the declared system has n_species="
                f"{self.system.n_species}")
        merged = {**self.declared, **per_run}
        _validate(self.system, self.integrator, self.state_proj, merged)
        kw = _rollout_keywords(self.state_proj, merged)
        if self.integrator == "sde":
            kBT_noise = kw.pop("kBT_noise")
            return integrators.rollout_sde(
                model, rho_hat0, boxes, T, self.dt, n_steps, kBT_noise,
                state_proj=self.state_proj, **kw)
        return integrators.rollout_deterministic(
            model, rho_hat0, boxes, T, self.dt, n_steps,
            method=self.integrator, state_proj=self.state_proj, **kw)


def build(system: System, *, integrator: str, dt: float, state_proj: str,
          **declared: Any) -> Solver:
    """Validate every declaration now, before a model is loaded; returns a :class:`Solver`."""
    if not isinstance(system, System):
        raise TypeError(
            f"build's first argument is the System the run is for, got "
            f"{type(system).__name__}. Core carries no system's numbers, "
            f"so there is nothing to fall back to.")
    if integrator not in INTEGRATORS:
        raise ValueError(
            f"integrator must be one of {list(INTEGRATORS)}, got "
            f"{integrator!r}")
    dt_value = float(dt)
    if not (dt_value > 0.0) or dt_value == float("inf"):
        raise ValueError(
            f"dt must be a positive, finite step in the model's own time "
            f"unit, got {dt!r}")
    _validate(system, integrator, state_proj, declared)
    return Solver(system, integrator, dt_value, state_proj,
                  MappingProxyType(dict(declared)))
