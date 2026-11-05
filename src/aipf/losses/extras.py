"""L_W: a non-canonical extra, the square-gradient hinge ``W_hat(k) - W_hat(0) >= kappa k^2``.

The ``kappa k^2`` floor gives the pair kernel a positive square-gradient stiffness at every ``k``, so
instability can only start at ``k = 0``; with ``W_hat(k_max) ~ 0`` it pins
``W_hat(0) <= -kappa k_max^2``, the attractive kernel that carries a miscibility gap (Fe-B: about ``-18``).
A run that weighs it at 0 loses that gap (:mod:`aipf.train.kernel_hinge`).

Optionally the hinge is read on the number-conserving subspace alone (the reduced block
``B^T D B``, :func:`number_conserving_basis`) and its floor saturates at ``k_sat``."""
from __future__ import annotations

import math
from typing import Optional

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


def number_conserving_basis(n: int, *, dtype: torch.dtype = torch.float64,
                            device: torch.device | str | None = None) -> torch.Tensor:
    """``(n, n - 1)``, an orthonormal basis ``B`` of ``1^perp``, the compositions that keep the total
    number: the Helmert columns ``b_j = (1, ..., 1, -j, 0, ..., 0) / sqrt(j (j + 1))``, ``j`` ones, so
    ``(1, -1)/sqrt(2)`` at two channels and ``(1, -1, 0)/sqrt(2)``, ``(1, 1, -2)/sqrt(6)`` at three.

    Any other orthonormal basis of ``1^perp`` is ``B Q`` with ``Q`` orthogonal, and ``(B Q)^T D (B Q) =
    Q^T (B^T D B) Q`` has the same eigenvalues: ``lambda_min`` of the reduced block does not depend on the
    choice. One channel is refused, since ``1^perp`` is then ``{0}``."""
    n = int(n)
    if n < 2:
        raise ValueError(
            f"wpsd_subspace='number_conserving' at {n} channel(s): the number-conserving subspace "
            f"1^perp of one channel is {{0}}, so there is nothing to hinge; declare no subspace")
    B = torch.zeros(n, n - 1, dtype=dtype, device=device)
    for j in range(1, n):
        B[:j, j - 1] = 1.0
        B[j, j - 1] = -float(j)
        B[:, j - 1] /= math.sqrt(j * (j + 1))
    return B


def l_w(D: torch.Tensor, k: torch.Tensor, *, kappa: float,
        margin: float = 0.0, basis: Optional[torch.Tensor] = None,
        k_sat: Optional[float] = None) -> torch.Tensor:
    """``mean(relu(margin + kappa k^2 - lambda_min(D))^2)``, ``D = W_hat(k) - W_hat(0)`` ``(..., n, n)``;
    two channels take :func:`eigmin_sym2`, others ``eigvalsh``. ``kappa`` is required (no default).

    ``basis`` ``(n, n - 1)`` (orthonormal, :func:`number_conserving_basis`) reads ``lambda_min`` of the
    reduced block ``B^T D B`` instead: a scalar at ``n = 2``, :func:`eigmin_sym2` at ``n = 3``,
    ``eigvalsh`` beyond. The projector ``P D P`` (``P = B B^T``) would not do: it carries a spurious zero
    eigenvalue along ``1``. ``k_sat`` saturates the floor at ``kappa min(k, k_sat)^2``."""
    if basis is None:
        eigmin = (eigmin_sym2(D) if D.shape[-1] == 2
                  else torch.linalg.eigvalsh(D)[..., 0])
    else:
        basis = basis.to(dtype=D.dtype, device=D.device)
        R = basis.transpose(-1, -2) @ D @ basis
        m = R.shape[-1]
        eigmin = (R[..., 0, 0] if m == 1 else eigmin_sym2(R) if m == 2
                  else torch.linalg.eigvalsh(R)[..., 0])
    k_floor = k if k_sat is None else torch.clamp(k, max=float(k_sat))
    margin_k = margin + kappa * k_floor * k_floor
    return F.relu(-eigmin + margin_k).pow(2).mean()
