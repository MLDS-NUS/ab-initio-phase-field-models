"""Thermodynamic and dynamic diagnostics for the free-energy ladder.

:mod:`~aipf.diagnose.thermo` (binodal, spinodal, ``T_c``), :mod:`~aipf.diagnose.dynamics` (growth rates,
``S(k)``, rollout drift), :mod:`~aipf.diagnose.kappa`, :mod:`~aipf.diagnose.reference`."""
from __future__ import annotations

# Names resolve on first access so importing the package does not import torch.
# Reach the driver as ``from aipf.diagnose.run import run``.

_SOURCES = {
    "finite_difference_rate": "aipf.diagnose.dynamics",
    "growth_rate": "aipf.diagnose.dynamics",
    "most_unstable_k": "aipf.diagnose.dynamics",
    "rollout_drift": "aipf.diagnose.dynamics",
    "structure_factor": "aipf.diagnose.dynamics",
    "nonlocal_kernel_kappa_eff": "aipf.diagnose.kappa",
    "ReferenceDataset": "aipf.diagnose.reference",
    "for_system": "aipf.diagnose.reference",
    "binodal": "aipf.diagnose.thermo",
    "binodal_convex_hull": "aipf.diagnose.thermo",
    "binodal_mu_roots": "aipf.diagnose.thermo",
    "critical_temperature": "aipf.diagnose.thermo",
    "dome_apex": "aipf.diagnose.thermo",
    "dome_tail": "aipf.diagnose.thermo",
    "spinodal": "aipf.diagnose.thermo",
}


def __getattr__(name: str):
    """One exported name, imported the first time it is asked for."""
    source = _SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(source), name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_SOURCES))


__all__ = [
    "binodal",
    "binodal_convex_hull",
    "binodal_mu_roots",
    "spinodal",
    "critical_temperature",
    "dome_apex",
    "dome_tail",
    "nonlocal_kernel_kappa_eff",
    "ReferenceDataset",
    "for_system",
    "growth_rate",
    "most_unstable_k",
    "structure_factor",
    "finite_difference_rate",
    "rollout_drift",
]
