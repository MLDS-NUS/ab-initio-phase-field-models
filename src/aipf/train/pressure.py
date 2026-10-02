"""The model-side Euler pressure ``L_P`` needs, built from ``FreeEnergyModel`` protocol methods only.

The field-wide term is quadratic in ``rho``, so it contributes half its ``mu . rho`` to ``f``."""
from __future__ import annotations

import torch

from aipf.functional.base import FreeEnergyModel


def pressure_from_model(model: FreeEnergyModel, rho: torch.Tensor,
                         boxes: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """``P = 1/2 (mu + mu_local) . rho - f_local``, ``mu_local = d f_local / d rho``; ``(P,)`` out.

    ``rho`` ``(P, n_species)``, ``boxes`` ``(P, 3)``, ``T`` ``(P,)``; differentiable (``create_graph=True``)."""
    n_points, n_species = rho.shape
    rho_grid = rho.reshape(n_points, n_species, 1, 1, 1)
    mu = model.chemical_potential(rho_grid, boxes, T).reshape(n_points,
                                                              n_species)

    leaf = rho_grid.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        f_grid = model.bulk_free_energy_density(leaf, T)
        mu_local = torch.autograd.grad(f_grid.sum(), leaf,
                                       create_graph=True)[0]
    mu_local = mu_local.reshape(n_points, n_species)
    f = model.bulk_free_energy_density(rho_grid, T).reshape(n_points)
    return 0.5 * ((mu + mu_local) * rho).sum(dim=-1) - f
