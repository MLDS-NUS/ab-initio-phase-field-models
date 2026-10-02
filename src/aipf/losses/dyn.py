"""L_dyn: the weak-form drift residual."""
from __future__ import annotations

import torch


def l_dyn(residual: torch.Tensor, k2: torch.Tensor, mult: torch.Tensor,
          band: torch.Tensor, *, alpha: float, eps: float = 1e-12,
          weight_eps: float = 1e-12,
          sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """``sum(band * mult * (alpha*r2 + (1-alpha)*r2/(k2+eps))) / n_band``, then a (weighted) batch mean.

    ``residual`` is ``(B, n_species, *grid)`` in k-space; ``alpha`` is required. ``eps`` floors ``k2`` only,
    ``weight_eps`` floors ``sum(sample_weight)``."""
    r2 = residual.abs().pow(2) if residual.is_complex() else residual.pow(2)
    r2 = r2 * mult
    per_mode = alpha * r2 + (1.0 - alpha) * r2 / (k2 + eps)
    per_mode = per_mode * band
    reduce_dims = tuple(range(1, per_mode.dim()))
    band_dims = tuple(range(1, band.dim()))
    n_band = band.to(per_mode.dtype).sum(dim=band_dims).clamp(min=1)
    per_sample = per_mode.sum(dim=reduce_dims) / n_band
    if sample_weight is None:
        return per_sample.mean()
    w = sample_weight.to(per_sample.dtype)
    return (w * per_sample).sum() / (w.sum() + weight_eps)
