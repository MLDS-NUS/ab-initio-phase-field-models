"""L_M: the mobility anchor, a relative squared Frobenius residual."""
from __future__ import annotations

import torch


def l_m(M_pred: torch.Tensor, M_target: torch.Tensor, *,
        row_weight: torch.Tensor | None = None,
        eps: float = 1e-30,
        active: torch.Tensor | None = None) -> torch.Tensor:
    """``mean_rows(|M_pred - M_target|_F^2 / max(|M_target|_F^2, eps))``, optionally ``row_weight``-weighted.

    ``M_pred``, ``M_target``: ``(P, n_species, n_species)``. ``active`` ``(P, n_species)`` bool, each row's
    present channels: given, both norms run over the row's active block alone."""
    if active is not None:
        # mask the inputs, not the residual: a non-finite entry outside the block reaches no gradient
        pairs = active.unsqueeze(-1) & active.unsqueeze(-2)
        M_pred = torch.where(pairs, M_pred, 0.0)
        M_target = torch.where(pairs, M_target, 0.0)
    num = (M_pred - M_target).pow(2).sum(dim=(-2, -1))
    den = M_target.pow(2).sum(dim=(-2, -1)).clamp(min=eps)
    per_row = num / den
    if row_weight is None:
        return per_row.mean()
    w = row_weight.to(per_row.dtype)
    return (w * per_row).sum() / w.sum().clamp(min=eps)
