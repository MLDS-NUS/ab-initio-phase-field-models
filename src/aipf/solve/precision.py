"""Which precision a rollout runs in: the state's and the model's dtype, never a separate switch.
A ``complex64`` state with a float32 model runs the float32 path the published rollouts ran; a ``complex128``
state with a float64 model runs float64 / complex128 throughout. A mixed pair is refused, naming both dtypes
and the cast that makes them agree. :data:`PRECISIONS` names the two for the drivers and the CLI
(``--precision``), which cast the model and the initial state before the solver (:func:`cast_pair`)."""
from __future__ import annotations

from typing import Set, Tuple

import torch

#: ``precision`` name -> (real dtype, complex dtype); ``"fp32"`` is the default everywhere.
PRECISIONS = {"fp32": (torch.float32, torch.complex64),
              "fp64": (torch.float64, torch.complex128)}


def check_precision(precision) -> str:
    """``precision`` if it is a key of :data:`PRECISIONS`; otherwise ``ValueError`` naming both."""
    if precision not in PRECISIONS:
        raise ValueError(
            f"precision must be one of {list(PRECISIONS)}, got {precision!r}")
    return precision


def state_real_dtype(rho_hat: torch.Tensor) -> torch.dtype:
    """The real dtype of a state: ``float32`` for ``complex64``, ``float64`` for ``complex128``."""
    return rho_hat.real.dtype if rho_hat.is_complex() else rho_hat.dtype


def model_float_dtypes(model) -> Set[torch.dtype]:
    """The floating dtypes of ``model``'s parameters and of its operator set's mode buffer ``ops.NX``.
    Other buffers are not read: a float32 model may carry a float64 constant (a reference ``kBT``)."""
    out: Set[torch.dtype] = set()
    params = getattr(model, "parameters", None)
    if callable(params):
        out.update(p.dtype for p in params() if p.is_floating_point())
    nx = getattr(getattr(model, "ops", None), "NX", None)
    if isinstance(nx, torch.Tensor) and nx.is_floating_point():
        out.add(nx.dtype)
    return out


def _name(dtypes) -> str:
    return " and ".join(sorted(str(d).replace("torch.", "") for d in dtypes))


def working_dtypes(model, rho_hat: torch.Tensor, what: str) -> Tuple[torch.dtype, torch.dtype]:
    """``(real, complex)`` dtypes of the run: float64/complex128 for a complex128 state and a float64 model,
    otherwise the float32 path's. A float64 state with a model that is not float64, or a float32 state with a
    model that has float64 parameters, is refused."""
    real = state_real_dtype(rho_hat)
    have = model_float_dtypes(model)
    if real == torch.float64:
        if have - {torch.float64}:
            raise ValueError(
                f"{what}: the state is {str(rho_hat.dtype).replace('torch.', '')} and the model "
                f"{type(model).__name__} is {_name(have)}; a float64 rollout needs both in float64. "
                f"Cast the model with model.double(), or roll out in float32 with "
                f"rho_hat.to(torch.complex64)")
        return torch.float64, torch.complex128
    if torch.float64 in have:
        raise ValueError(
            f"{what}: the state is {str(rho_hat.dtype).replace('torch.', '')} and the model "
            f"{type(model).__name__} is {_name(have)}; a float64 model rolls out a float64 state. "
            f"Cast the state with rho_hat.to(torch.complex128), or the model back with model.float()")
    return torch.float32, torch.complex64


def cast_pair(model, rho_hat: torch.Tensor, precision: str):
    """``(model, rho_hat)`` in ``precision``; ``"fp32"`` returns both untouched (no copy, no cast)."""
    check_precision(precision)
    if precision == "fp32":
        return model, rho_hat
    real, cplx = PRECISIONS[precision]
    return model.to(real), rho_hat.to(cplx)
