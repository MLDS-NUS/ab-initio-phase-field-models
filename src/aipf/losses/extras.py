"""L_W: a non-canonical extra, the square-gradient hinge ``W_hat(k) - W_hat(0) >= kappa k^2``.

The ``kappa k^2`` floor gives the pair kernel a positive square-gradient stiffness at every ``k``, so
instability can only start at ``k = 0``; with ``W_hat(k_max) ~ 0`` it pins
``W_hat(0) <= -kappa k_max^2``, the attractive kernel that carries a miscibility gap (Fe-B: ``-18``).
A run that weighs it at 0 loses that gap (:mod:`aipf.train.kernel_hinge`)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: Keeps the square root's gradient finite at an exactly degenerate matrix.
EIGMIN_SQRT_FLOOR = 1e-30


def eigmin_sym2(D: torch.Tensor) -> torch.Tensor:
    """Smaller eigenvalue of symmetric ``(..., 2, 2)`` matrices, ``(a + c)/2 - sqrt(((a - c)/2)^2 + b^2)``:
    one expression on every device, where ``eigvalsh`` at a near-degenerate ``D`` depends on the backend.

    Symmetry is assumed, not checked: ``b`` is ``D[..., 0, 1]`` and ``D[..., 1, 0]`` is never read, so
    the gradient reaches the upper off-diagonal entry only. A pair kernel's ``W_hat`` is assembled
    symmetric, so the two entries are one value."""
    if D.shape[-2:] != (2, 2):
        raise ValueError(f"eigmin_sym2 takes (..., 2, 2) matrices, got {tuple(D.shape)}")
    a = D[..., 0, 0]
    b = D[..., 0, 1]
    c = D[..., 1, 1]
    return 0.5 * (a + c) - torch.sqrt(((a - c) * 0.5) ** 2 + b * b + EIGMIN_SQRT_FLOOR)


def l_w(D: torch.Tensor, k: torch.Tensor, *, kappa: float,
        margin: float = 0.0) -> torch.Tensor:
    """``mean(relu(margin + kappa k^2 - lambda_min(D))^2)``, ``D = W_hat(k) - W_hat(0)`` ``(..., n, n)``;
    two channels take :func:`eigmin_sym2`, others ``eigvalsh``. ``kappa`` is required (no default)."""
    eigmin = (eigmin_sym2(D) if D.shape[-1] == 2
              else torch.linalg.eigvalsh(D)[..., 0])
    margin_k = margin + kappa * k * k
    return F.relu(-eigmin + margin_k).pow(2).mean()
