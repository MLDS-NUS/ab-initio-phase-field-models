"""The factory toy of :mod:`toy_factory_model`, written once for either number of spatial axes.

The same functional, a square gradient written as a pair kernel ``W(k) = kappa k^2`` on a polynomial
local free energy with a constant, full, positive definite mobility, but every axis count, permutation
and einsum read off the declared grid (``self.ops.ndim``), so one class builds on ``(Gx, Gy, Gz)`` and on
``(Gx, Gy)``. The parameter names are the 3D toy's, so a state dict moves between the two by its
parameters alone. :class:`LinearLocal` swaps in ``f = a rho^2 / 2`` for the tests that need a linear
model, whose stationary spectrum is known in closed form."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aipf.spectral import (CHANNEL_LAST, CHANNEL_SECOND, FLUX_EINSUM, OpsCache,
                           k_squared)
from aipf.system import System

from toy_factory_model import PolynomialLocal, SquareGradientKernel


class LinearLocal(nn.Module):
    """``f = sum_i a rho_i^2 / 2``, a fixed (not trained) curvature ``a``; ``rho`` ``(P, n)``."""

    def __init__(self, a: float):
        super().__init__()
        self.register_buffer("a", torch.tensor(float(a)))

    def f_pointwise(self, rho: torch.Tensor, kBT: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.a * (rho ** 2).sum(-1)

    def mu_pointwise(self, rho: torch.Tensor, kBT: torch.Tensor) -> torch.Tensor:
        return self.a * rho


class NdimToyModel(nn.Module):
    """The four ``FreeEnergyModel`` methods, ``_cache``, ``ops``, ``kernel.w_hat`` and ``f_local``, on the
    declared ``grid`` of two or three axes. ``linear_a`` replaces the polynomial local part by
    :class:`LinearLocal`."""

    def __init__(self, grid, n_species: int, *, nyquist_mask: bool, kappa: float, kB: float,
                 linear_a: float | None = None):
        super().__init__()
        self.n_species = int(n_species)
        self.kB = float(kB)
        self._cache = OpsCache(grid, self.n_species, nyquist_mask=nyquist_mask)
        self.ops = self._cache.ops
        self.kernel = SquareGradientKernel(self.n_species, kappa)
        self.f_local = (PolynomialLocal(self.n_species) if linear_a is None
                        else LinearLocal(linear_a))
        n_lower = self.n_species * (self.n_species + 1) // 2
        self.mobility_raw = nn.Parameter(torch.zeros(n_lower))

    @property
    def ndim(self) -> int:
        return self._cache.ndim

    def mobility_matrix(self) -> torch.Tensor:
        """``L L^T + 0.1 I``, ``L`` lower triangular with a softplus diagonal."""
        n = self.n_species
        rows, cols = torch.tril_indices(n, n)
        L = torch.zeros(n, n, dtype=self.mobility_raw.dtype, device=self.mobility_raw.device)
        L = L.index_put((rows, cols), self.mobility_raw)
        L = L - torch.diag(torch.diagonal(L)) + torch.diag(F.softplus(torch.diagonal(L)))
        return L @ L.T + 0.1 * torch.eye(n, dtype=L.dtype, device=L.device)

    def _grid(self, x: torch.Tensor) -> tuple:
        return tuple(int(g) for g in x.shape[-self.ndim:])

    def _flat(self, rho: torch.Tensor) -> torch.Tensor:
        return rho.permute(*CHANNEL_LAST[self.ndim]).reshape(-1, self.n_species)

    def _kbt_flat(self, T: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        B, grid = rho.shape[0], self._grid(rho)
        ones = (1,) * self.ndim
        return (self.kB * T).to(rho.dtype).view(B, *ones).expand(B, *grid).reshape(-1)

    def bulk_free_energy_density(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        f = self.f_local.f_pointwise(self._flat(rho), self._kbt_flat(T, rho))
        return f.view(rho.shape[0], 1, *self._grid(rho))

    def chemical_potential(self, rho: torch.Tensor, boxes: torch.Tensor,
                           T: torch.Tensor) -> torch.Tensor:
        B, grid = rho.shape[0], self._grid(rho)
        ops = self._cache.ops_for_grid(grid, rho.device)
        mu = self.f_local.mu_pointwise(self._flat(rho), self._kbt_flat(T, rho))
        mu = mu.view(B, *grid, self.n_species).permute(*CHANNEL_SECOND[self.ndim])
        kmag = torch.sqrt(k_squared(ops.k_axes(boxes)))[:, 0]
        W = self.kernel.w_hat(kmag)
        rho_hat = ops.rfft(rho).permute(*CHANNEL_LAST[self.ndim])
        term = torch.einsum("b...ij,b...j->b...i", W.to(rho_hat.dtype), rho_hat)
        return mu + ops.irfft(term.permute(*CHANNEL_SECOND[self.ndim]))

    def mobility(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        n, B = self.n_species, rho.shape[0]
        M = self.mobility_matrix().to(rho.dtype)
        return M.view(1, n, n, *(1,) * self.ndim).expand(B, n, n, *self._grid(rho))

    def forward(self, rho_hat: torch.Tensor, boxes: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        ops = self._cache.ops_for(rho_hat.shape, rho_hat.device)
        N = math.prod(ops.grid)
        rho = ops.irfft(rho_hat * N)
        mu = self.chemical_potential(rho, boxes, T)
        M = self.mobility(rho, T)
        ks = ops.k_axes(boxes)
        grads = ops.grad_hat(ops.rfft(mu), *ks)
        J = [torch.einsum(FLUX_EINSUM[self.ndim], M, ops.irfft(g)) for g in grads]
        return ops.div_hat(*[ops.rfft(j) / N for j in J], *ks)


def build_ndim_toy(system: System, **overrides) -> NdimToyModel:
    """The factory: ``factory(system, **overrides)``, on whatever grid the declaration names."""
    kw = system.functional.kwargs
    return NdimToyModel(kw["grid"], system.n_species,
                        **{**dict(nyquist_mask=kw["nyquist_mask"], kappa=kw["kappa"],
                                  kB=system.constants["kB"]), **overrides})
