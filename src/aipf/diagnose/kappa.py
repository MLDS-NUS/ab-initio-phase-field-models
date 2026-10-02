"""``kappa_eff`` read out of a TRAINED rung-3 kernel: a property of rung 3, never a ladder row."""
from __future__ import annotations

from typing import Any

import torch


def nonlocal_kernel_kappa_eff(model_or_kernel: Any) -> torch.Tensor:
    """``kappa_eff`` of a rung-3 assembly (its ``.kernel``) or a bare ``PairKernel``, duck-typed.

    Raises ``AttributeError`` for anything without a callable ``kappa_eff`` (rungs 1, 2, 4)."""
    kernel = getattr(model_or_kernel, "kernel", model_or_kernel)
    fn = getattr(kernel, "kappa_eff", None)
    if not callable(fn):
        raise AttributeError(
            f"{type(model_or_kernel).__name__!r} carries no rung-3 pair "
            f"kernel with a kappa_eff() method. kappa_eff is a diagnostic "
            f"read out of a TRAINED rung-3 kernel (PairKernel.kappa_eff, "
            f"aipf.functional.kernels) -- never a general ladder quantity, "
            f"and never a rung-2 result. Pass a rung-3 "
            f"model (its .kernel attribute) or a PairKernel directly."
        )
    return fn()
