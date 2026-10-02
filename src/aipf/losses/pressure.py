"""L_P: the pressure (isobar) anchor on a caller-computed Euler pressure ``P = mu . rho - f``.

Modes: ``absolute`` (measured target per state) and ``variance`` (one isobar group agrees with itself)."""
from __future__ import annotations

from typing import Optional

import torch


def l_p_absolute(P_model: torch.Tensor, P_target: torch.Tensor, *,
                 floor: Optional[float] = None) -> torch.Tensor:
    """``mean[((P_model - P_target) / den)^2]``, ``den = P_target``, or ``max(|P_target|, floor)`` with a
    ``floor`` (a zero target is otherwise a division by zero); both ``(P,)``."""
    if floor is None:
        return ((P_model - P_target) / P_target).pow(2).mean()
    den = P_target.abs().clamp(min=float(floor))
    return ((P_model - P_target) / den).pow(2).mean()


def l_p_variance(P_model: torch.Tensor, group_id: torch.Tensor, *,
                  p_ref: float) -> torch.Tensor:
    """``mean_g var_g(P_model) / p_ref^2`` over dense ``group_id`` ``0..n_groups-1``, all ``(P,)``.

    ``p_ref`` is the system's reference pressure in ``P_model``'s unit: required, never defaulted."""
    n_groups = int(group_id.max().item()) + 1
    total = P_model.new_zeros(())
    for g in range(n_groups):
        Pg = P_model[group_id == g]
        total = total + ((Pg - Pg.mean()) ** 2).mean()
    return total / (n_groups * p_ref ** 2)
