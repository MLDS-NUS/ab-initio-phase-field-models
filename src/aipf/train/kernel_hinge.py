"""``L_W``, the pair kernel's square-gradient hinge (:func:`aipf.losses.extras.l_w`) on ``n_k``
wavenumbers ``linspace(k_max / n_k, k_max, n_k)``, trained once a step when a system weighs it.

A declaration spells it as a saved checkpoint does (``lambda_wpsd``; ``wpsd_kappa``, ``wpsd_k_max``,
``wpsd_n_k``, ``wpsd_margin``), and :func:`declared_fields` maps it as ``ckpt_compat`` maps that
checkpoint: the weight to ``lambda_W``, the shape into ``extra_experiment_config``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from aipf.losses.extras import l_w
from aipf.losses.names import WEIGHT_MAP_A

from .config import TrainConfig

#: The term this module feeds.
HINGE_TERM = "L_W"

#: The declared (and saved) spelling of the hinge's weight.
WEIGHT_KEY: str = WEIGHT_MAP_A.to_code(HINGE_TERM)

#: The declared (and saved) spelling of the hinge's shape, kept under these keys in
#: ``extra_experiment_config``; every one is required when the weight is not zero.
SHAPE_KEYS: Tuple[str, ...] = ("wpsd_kappa", "wpsd_k_max", "wpsd_n_k", "wpsd_margin")


def declared_fields(declared: Mapping[str, Any]) -> Dict[str, Any]:
    """The :class:`TrainConfig` fields the hinge keys of one declaration give, ``{}`` when it declares
    none: ``lambda_W`` from :data:`WEIGHT_KEY`, and ``extra_experiment_config`` holding whichever of
    :data:`SHAPE_KEYS` it declares. A declaration naming both ``lambda_W`` and :data:`WEIGHT_KEY` is
    refused, since the two would be one weight declared twice."""
    out: Dict[str, Any] = {}
    if WEIGHT_KEY in declared:
        if "lambda_W" in declared:
            raise ValueError(
                f"the declaration names both lambda_W={declared['lambda_W']!r} and "
                f"{WEIGHT_KEY}={declared[WEIGHT_KEY]!r}, the same weight twice: keep one")
        out["lambda_W"] = float(declared[WEIGHT_KEY])
    shape = {key: declared[key] for key in SHAPE_KEYS if key in declared}
    if shape:
        out["extra_experiment_config"] = shape
    return out


@dataclass(frozen=True)
class KernelHinge:
    """The hinge's shape: ``kappa``, the wavenumber reach ``k_max``, the count ``n_k`` and ``margin``."""

    kappa: float
    k_max: float
    n_k: int
    margin: float

    def __post_init__(self) -> None:
        if int(self.n_k) < 1:
            raise ValueError(f"wpsd_n_k={self.n_k!r}: the hinge needs at least one wavenumber")
        if not float(self.k_max) > 0.0:
            raise ValueError(f"wpsd_k_max={self.k_max!r}: the wavenumbers reach a positive k_max")

    @classmethod
    def from_config(cls, cfg: TrainConfig) -> Optional["KernelHinge"]:
        """The hinge ``cfg`` trains, ``None`` at ``lambda_W == 0``; a non-zero weight without every one
        of :data:`SHAPE_KEYS` in ``extra_experiment_config`` is refused, naming the missing keys."""
        if float(cfg.lambda_W) == 0.0:
            return None
        extra = cfg.extra_experiment_config
        missing = [key for key in SHAPE_KEYS if key not in extra]
        if missing:
            raise KeyError(
                f"lambda_W={cfg.lambda_W!r} trains the kernel hinge, and its shape {missing} "
                f"is not declared: this package has no default for any of {list(SHAPE_KEYS)}")
        return cls(kappa=float(extra["wpsd_kappa"]), k_max=float(extra["wpsd_k_max"]),
                   n_k=int(extra["wpsd_n_k"]), margin=float(extra["wpsd_margin"]))

    def wavenumbers(self, *, dtype: torch.dtype,
                    device: torch.device | str | None = None) -> torch.Tensor:
        """``linspace(k_max / n_k, k_max, n_k)``, made in ``dtype``."""
        return torch.linspace(self.k_max / self.n_k, self.k_max, int(self.n_k),
                              dtype=dtype, device=device)

    @staticmethod
    def check(model) -> None:
        """Refuse a model the hinge cannot read: no pair kernel, or one whose ``W_hat`` needs a grid."""
        kernel = getattr(model, "kernel", None)
        if kernel is None or not callable(getattr(kernel, "w_hat", None)):
            raise NotImplementedError(
                "lambda_W trains a hinge on the pair kernel's W_hat(k), and this model has no "
                "pair kernel")
        if getattr(getattr(kernel, "evaluator", None), "reads_geometry", False):
            raise NotImplementedError(
                "this kernel's W_hat reads a grid and a box, and the hinge evaluates it on a "
                "wavenumber line that has neither")

    def loss(self, model) -> torch.Tensor:
        """``l_w(W_hat(k) - W_hat(0), k)`` in the model's dtype and on its device, with a graph."""
        p = next(model.parameters())
        k = self.wavenumbers(dtype=p.dtype, device=p.device)
        W = model.kernel.w_hat(torch.cat([k.new_zeros(1), k]))      # (n_k + 1, n, n)
        return l_w(W[1:] - W[0], k, kappa=self.kappa, margin=self.margin)

    def provenance(self) -> Dict[str, Any]:
        return {"term": HINGE_TERM, "kappa": self.kappa, "k_max": self.k_max,
                "n_k": int(self.n_k), "margin": self.margin}


__all__ = ["HINGE_TERM", "KernelHinge", "SHAPE_KEYS", "WEIGHT_KEY", "declared_fields"]
