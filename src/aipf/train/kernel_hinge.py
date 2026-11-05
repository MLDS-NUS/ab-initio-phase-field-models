"""``L_W``, the pair kernel's square-gradient hinge (:func:`aipf.losses.extras.l_w`) on ``n_k``
wavenumbers ``linspace(k_max / n_k, k_max, n_k)``, trained once a step when a system weighs it.

Why it exists: the floor ``W_hat(k) - W_hat(0) >= kappa k^2`` gives the pair kernel a positive
square-gradient stiffness at every ``k``, so a homogeneous state can only lose stability at ``k = 0``
(no finite-``k`` checkerboard). With ``W_hat(k_max) ~ 0`` it also pins ``W_hat(0) <= -kappa k_max^2``
(about ``-18`` for the Fe-B declaration, ``kappa = 2``, ``k_max = 3``; the published kernel
has ``W_hat(0)`` eigenvalues of about -17.5): an attractive ``k = 0`` kernel, and
for Fe-B that kernel is what carries the miscibility gap, at a measurable cost to the ``L_S`` and
``L_bulk`` fit. It is not an optional regulariser: a run that sets the weight to 0 loses the gap.

A declaration spells it as a saved checkpoint does (``lambda_wpsd``; ``wpsd_kappa``, ``wpsd_k_max``,
``wpsd_n_k``, ``wpsd_margin``), and :func:`declared_fields` maps it as ``ckpt_compat`` maps that
checkpoint: the weight to ``lambda_W``, the shape into ``extra_experiment_config``.

Two optional keys (:data:`OPTIONAL_KEYS`) change what is hinged, and neither is written anywhere when
it is not declared. ``wpsd_subspace="number_conserving"`` reads ``lambda_min`` of the reduced block
``B^T (W_hat(k) - W_hat(0)) B`` on the number-conserving subspace ``1^perp``
(:func:`aipf.losses.extras.number_conserving_basis`), leaving the total-number direction free: a
liquid's ``S_NN(k)`` rises toward its first peak, so the full matrix contradicts the data in that
direction at every ``k``. ``wpsd_k_sat`` saturates the floor at ``kappa min(k, k_sat)^2``, so a hinge
reaching the grid's corner does not demand an ever larger stiffness there. ``W_hat`` itself is flat
beyond the radial transform's ``kernel_k_table_max`` (``AnalyticRadialTransform`` clamps ``|k|``)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from aipf.losses.extras import l_w, number_conserving_basis
from aipf.losses.names import WEIGHT_MAP_A
from aipf.spectral import model_ndim, refuse_two_dimensions

from .config import TrainConfig

#: The term this module feeds.
HINGE_TERM = "L_W"

#: The declared (and saved) spelling of the hinge's weight.
WEIGHT_KEY: str = WEIGHT_MAP_A.to_code(HINGE_TERM)

#: The declared (and saved) spelling of the hinge's shape, kept under these keys in
#: ``extra_experiment_config``; every one is required when the weight is not zero.
SHAPE_KEYS: Tuple[str, ...] = ("wpsd_kappa", "wpsd_k_max", "wpsd_n_k", "wpsd_margin")

#: The hinge's optional keys, kept beside :data:`SHAPE_KEYS`: the subspace it reads (one of
#: :data:`SUBSPACES`) and the wavenumber its floor saturates at. Absent is the full matrix, unsaturated.
OPTIONAL_KEYS: Tuple[str, ...] = ("wpsd_subspace", "wpsd_k_sat")

#: The subspaces ``wpsd_subspace`` names: ``None`` (the full matrix) or ``"number_conserving"``.
SUBSPACES: Tuple[Optional[str], ...] = (None, "number_conserving")


def declared_fields(declared: Mapping[str, Any]) -> Dict[str, Any]:
    """The :class:`TrainConfig` fields the hinge keys of one declaration give, ``{}`` when it declares
    none: ``lambda_W`` from :data:`WEIGHT_KEY`, and ``extra_experiment_config`` holding whichever of
    :data:`SHAPE_KEYS` and :data:`OPTIONAL_KEYS` it declares. A declaration naming both ``lambda_W``
    and :data:`WEIGHT_KEY` is refused, since the two would be one weight declared twice."""
    out: Dict[str, Any] = {}
    if WEIGHT_KEY in declared:
        if "lambda_W" in declared:
            raise ValueError(
                f"the declaration names both lambda_W={declared['lambda_W']!r} and "
                f"{WEIGHT_KEY}={declared[WEIGHT_KEY]!r}, the same weight twice: keep one")
        out["lambda_W"] = float(declared[WEIGHT_KEY])
    shape = {key: declared[key] for key in SHAPE_KEYS + OPTIONAL_KEYS if key in declared}
    if shape:
        out["extra_experiment_config"] = shape
    return out


@dataclass(frozen=True)
class KernelHinge:
    """The hinge's shape: ``kappa``, the wavenumber reach ``k_max``, the count ``n_k`` and ``margin``;
    ``weight`` is the ``lambda_W`` it trains at (recorded, not compared: one shape is one hinge).
    ``subspace`` (:data:`SUBSPACES`) and ``k_sat`` are the optional keys, ``None`` for the full matrix
    and an unsaturated floor."""

    kappa: float
    k_max: float
    n_k: int
    margin: float
    weight: Optional[float] = field(default=None, compare=False)
    subspace: Optional[str] = None
    k_sat: Optional[float] = None

    def __post_init__(self) -> None:
        if int(self.n_k) < 1:
            raise ValueError(f"wpsd_n_k={self.n_k!r}: the hinge needs at least one wavenumber")
        if not float(self.k_max) > 0.0:
            raise ValueError(f"wpsd_k_max={self.k_max!r}: the wavenumbers reach a positive k_max")
        if self.subspace not in SUBSPACES:
            raise ValueError(f"wpsd_subspace={self.subspace!r} is not one of {SUBSPACES}")
        if self.k_sat is not None and (isinstance(self.k_sat, bool)
                                       or not math.isfinite(float(self.k_sat))
                                       or not float(self.k_sat) > 0.0):
            raise ValueError(f"wpsd_k_sat={self.k_sat!r}: the floor saturates at a positive, finite k")

    @classmethod
    def from_config(cls, cfg: TrainConfig) -> Optional["KernelHinge"]:
        """The hinge ``cfg`` trains, ``None`` at ``lambda_W == 0``; a non-zero weight without every one
        of :data:`SHAPE_KEYS` in ``extra_experiment_config`` is refused, naming the missing keys.
        :data:`OPTIONAL_KEYS` are read when present."""
        if float(cfg.lambda_W) == 0.0:
            return None
        extra = cfg.extra_experiment_config
        missing = [key for key in SHAPE_KEYS if key not in extra]
        if missing:
            raise KeyError(
                f"lambda_W={cfg.lambda_W!r} trains the kernel hinge, and its shape {missing} "
                f"is not declared: this package has no default for any of {list(SHAPE_KEYS)}")
        k_sat = extra.get("wpsd_k_sat")
        if isinstance(k_sat, bool):
            raise ValueError(f"wpsd_k_sat={k_sat!r}: the floor saturates at a positive, finite k")
        return cls(kappa=float(extra["wpsd_kappa"]), k_max=float(extra["wpsd_k_max"]),
                   n_k=int(extra["wpsd_n_k"]), margin=float(extra["wpsd_margin"]),
                   weight=float(cfg.lambda_W), subspace=extra.get("wpsd_subspace"),
                   k_sat=None if k_sat is None else float(k_sat))

    def wavenumbers(self, *, dtype: torch.dtype,
                    device: torch.device | str | None = None) -> torch.Tensor:
        """``linspace(k_max / n_k, k_max, n_k)``, made in ``dtype``."""
        return torch.linspace(self.k_max / self.n_k, self.k_max, int(self.n_k),
                              dtype=dtype, device=device)

    @staticmethod
    def check(model) -> None:
        """Refuse a model the hinge cannot read: no pair kernel, or one whose ``W_hat`` needs a grid.
        A two-dimensional model is refused too."""
        refuse_two_dimensions(model_ndim(model), "the kernel hinge (lambda_W)")
        kernel = getattr(model, "kernel", None)
        if kernel is None or not callable(getattr(kernel, "w_hat", None)):
            raise NotImplementedError(
                "lambda_W trains a hinge on the pair kernel's W_hat(k), and this model has no "
                "pair kernel")
        if getattr(getattr(kernel, "evaluator", None), "reads_geometry", False):
            raise NotImplementedError(
                "this kernel's W_hat reads a grid and a box, and the hinge evaluates it on a "
                "wavenumber line that has neither")

    def check_subspace(self, model) -> None:
        """Refuse a declared subspace the model's kernel does not have: ``number_conserving`` at one
        channel, where ``1^perp`` is ``{0}``. Nothing to check without a subspace."""
        if self.subspace is not None:
            number_conserving_basis(_channels(model.kernel))

    def basis(self, n: int, *, dtype: torch.dtype,
              device: torch.device | str | None = None) -> Optional[torch.Tensor]:
        """The ``(n, n - 1)`` basis :func:`~aipf.losses.extras.l_w` reduces ``D`` on, ``None`` for the
        full matrix."""
        if self.subspace is None:
            return None
        return number_conserving_basis(n, dtype=dtype, device=device)

    def loss(self, model) -> torch.Tensor:
        """``l_w(W_hat(k) - W_hat(0), k)`` in the model's dtype and on its device, with a graph; on the
        declared subspace and with the declared saturation, when there are."""
        refuse_two_dimensions(model_ndim(model), "the kernel hinge (lambda_W)")
        p = next(model.parameters())
        k = self.wavenumbers(dtype=p.dtype, device=p.device)
        W = model.kernel.w_hat(torch.cat([k.new_zeros(1), k]))      # (n_k + 1, n, n)
        return l_w(W[1:] - W[0], k, kappa=self.kappa, margin=self.margin,
                   basis=self.basis(W.shape[-1], dtype=W.dtype, device=W.device),
                   k_sat=self.k_sat)

    def provenance(self) -> Dict[str, Any]:
        """What ``MANIFEST.json`` records under ``kernel_hinge``: the term, its shape and the weight
        ``lambda_W`` it trained at (``None`` for a hinge built without one); ``subspace`` and ``k_sat``
        only when they are set."""
        out = {"term": HINGE_TERM, "kappa": self.kappa, "k_max": self.k_max,
               "n_k": int(self.n_k), "margin": self.margin,
               "lambda_W": None if self.weight is None else float(self.weight)}
        if self.subspace is not None:
            out["subspace"] = self.subspace
        if self.k_sat is not None:
            out["k_sat"] = float(self.k_sat)
        return out


def _channels(kernel) -> int:
    """The kernel's number of channels: its ``n_species``, else the size of ``W_hat(0)``."""
    n = getattr(kernel, "n_species", None)
    if n is not None:
        return int(n)
    p = next(kernel.parameters())
    with torch.no_grad():
        return int(kernel.w_hat(torch.zeros(1, dtype=p.dtype, device=p.device)).shape[-1])


__all__ = ["HINGE_TERM", "KernelHinge", "OPTIONAL_KEYS", "SHAPE_KEYS", "SUBSPACES", "WEIGHT_KEY",
           "declared_fields"]
