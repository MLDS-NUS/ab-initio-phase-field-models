"""L_M: the mobility anchor, a relative squared Frobenius residual."""
from __future__ import annotations

import torch


def l_m(M_pred: torch.Tensor, M_target: torch.Tensor, *,
        row_weight: torch.Tensor | None = None,
        eps: float = 1e-30) -> torch.Tensor:
    """``mean_rows(|M_pred - M_target|_F^2 / max(|M_target|_F^2, eps))``, optionally ``row_weight``-weighted.

    ``M_pred``, ``M_target``: ``(P, n_species, n_species)``."""
    num = (M_pred - M_target).pow(2).sum(dim=(-2, -1))
    den = M_target.pow(2).sum(dim=(-2, -1)).clamp(min=eps)
    per_row = num / den
    if row_weight is None:
        return per_row.mean()
    w = row_weight.to(per_row.dtype)
    return (w * per_row).sum() / w.sum().clamp(min=eps)
