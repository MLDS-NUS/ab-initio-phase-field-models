"""``aipf.solve``: integrators, the state-projection stack, and the rollout guards.
Operates on a ``FreeEnergyModel`` through its four protocol methods only. Rules: one declared
admissible set plus an exact ``k=0`` mass restore (no band mask); per-step Nyquist hermitianisation;
noise colour and every physics knob declared, never defaulted (:mod:`aipf.solve.declare`)."""
from __future__ import annotations

from .declare import UNDECLARED
from .trust_domain import TrustDomain, in_domain, project_trust_domain
from .projection import (STATE_PROJ_MODES, check_state_projection,
                         hermitianize, project_state)
from .guards import assert_finite, clamped_inputs
from .noise import (M_STAB_MODES, NOISE_MODES, build_noise_filter,
                    check_m_stab, check_noise_declaration, declared_noise)
from .integrators import (
    step_euler,
    step_heun,
    step_sde_euler_maruyama,
    rollout_deterministic,
    rollout_sde,
)
# ``build`` is not re-exported (it would shadow its submodule).
from .build import INTEGRATORS, Solver

__all__ = [
    "UNDECLARED",
    "TrustDomain",
    "in_domain",
    "project_trust_domain",
    "STATE_PROJ_MODES",
    "check_state_projection",
    "hermitianize",
    "project_state",
    "assert_finite",
    "clamped_inputs",
    "NOISE_MODES",
    "M_STAB_MODES",
    "check_m_stab",
    "declared_noise",
    "build_noise_filter",
    "check_noise_declaration",
    "step_euler",
    "step_heun",
    "step_sde_euler_maruyama",
    "rollout_deterministic",
    "rollout_sde",
    "INTEGRATORS",
    "Solver",
]
