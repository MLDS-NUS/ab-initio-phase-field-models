"""Landau and Flory-Huggins free energies: closed-form basis expansions, untrained.
``landau``: ``f = sum_s [(1/8) a0 (T - Tc) psi_s^2 + (b/16) psi_s^4]``, ``psi_s = 2 rho_s - 1``.
``fh``: ``f = w rho (1 - rho) + T [rho ln rho + (1 - rho) ln(1 - rho)]``; at ``n_species > 1`` the
enthalpy is a Redlich-Kister polynomial in ``z_s = rho_s / rho_ref_s - 1``.
Every constant is a required constructor argument. The formulas are the module functions below, which
the rung-2 square-gradient forms call too (one implementation of each)."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from .base import MODEL_REGISTRY
from aipf.spectral import OpsCache

__all__ = ["Landau", "FloryHuggins", "landau_f", "landau_mu", "regular_solution_f",
           "regular_solution_mu", "ideal_mixing_f", "ideal_mixing_mu", "landau_curvature",
           "regular_solution_curvature"]


def landau_f(psi, T, a0, b, T_c):
    """``(a0/8)(T - T_c) psi^2 + (b/16) psi^4``, elementwise, ``psi = 2 rho - 1``."""
    return 0.125 * a0 * (T - T_c) * psi ** 2 + (b / 16.0) * psi ** 4


def landau_mu(psi, T, a0, b, T_c):
    """``d landau_f / d rho = (a0/2)(T - T_c) psi + (b/2) psi^3``, elementwise."""
    return 0.5 * a0 * (T - T_c) * psi + 0.5 * b * psi ** 3


def landau_curvature(T, a0, T_c):
    """``d2 landau_f / d rho2`` at ``rho = 1/2``: ``a0 (T - T_c)``."""
    return a0 * (T - T_c)


def regular_solution_curvature(T, w):
    """``d2 (regular_solution_f + ideal_mixing_f) / d rho2`` at ``rho = 1/2``: ``4 T - 2 w``."""
    return 4.0 * T - 2.0 * w


def regular_solution_f(rho, w):
    """``w rho (1 - rho)``, elementwise."""
    return w * rho * (1.0 - rho)


def regular_solution_mu(rho, w):
    """``w (1 - 2 rho)``, elementwise."""
    return w * (1.0 - 2.0 * rho)


def ideal_mixing_f(rho, T):
    """``T [rho ln rho + (1 - rho) ln(1 - rho)]``, elementwise, ``rho`` in ``(0, 1)``."""
    return T * (rho * torch.log(rho) + (1.0 - rho) * torch.log1p(-rho))


def ideal_mixing_mu(rho, T):
    """``T ln(rho / (1 - rho))``, elementwise, as ``T (ln rho - ln1p(-rho))``."""
    return T * (torch.log(rho) - torch.log1p(-rho))


def _compositions(total: int, parts: int):
    """Every tuple of ``parts`` non-negative ints summing to ``total``, first index slowest."""
    if parts == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in _compositions(total - i, parts - 1):
            yield (i,) + rest


class _RedlichKister(nn.Module):
    """Monomials in ``z_s = rho_s/rho_ref_s - 1`` of total degree in ``{0} u {2 .. degree}``.
    Powers by repeated multiplication, not ``torch.pow`` (NaN second derivative at ``z == 0``)."""

    def __init__(self, n_species: int, rho_ref: Sequence[float], degree: int):
        super().__init__()
        if len(rho_ref) != n_species:
            raise ValueError(
                f"rho_ref={tuple(rho_ref)!r} has {len(rho_ref)} entries but "
                f"n_species={n_species}"
            )
        if degree < 2:
            raise ValueError(f"degree must be >= 2, got {degree}")
        self.n_species = int(n_species)
        self.degree = int(degree)
        exps = [(0,) * n_species]
        for total in range(2, self.degree + 1):
            exps.extend(_compositions(total, n_species))
        self.register_buffer("exps", torch.tensor(exps, dtype=torch.long))
        self.register_buffer(
            "rho_ref", torch.tensor([float(r) for r in rho_ref], dtype=torch.float32)
        )
        self.coefs = nn.Parameter(torch.zeros(len(exps)))

    def _powers(self, z: torch.Tensor) -> torch.Tensor:
        """``(..., degree + 1, n_species)``: ``z_s^0 .. z_s^degree``."""
        pows = [torch.ones_like(z)]
        for _ in range(self.degree):
            pows.append(pows[-1] * z)
        return torch.stack(pows, dim=-2)

    def forward(self, rho_last: torch.Tensor) -> torch.Tensor:
        """``(..., n_species)`` (species last) to the summed polynomial ``(...,)``."""
        z = rho_last / self.rho_ref.to(rho_last.dtype) - 1.0
        P = self._powers(z)
        vals = torch.stack(
            [P[..., self.exps[:, s], s] for s in range(self.n_species)], dim=-1
        )
        mono = vals.prod(dim=-1)
        return mono @ self.coefs.to(rho_last.dtype)

    def d_drho(self, rho_last: torch.Tensor) -> torch.Tensor:
        """Analytic ``d(forward)/d(rho_last)``, same shape as ``rho_last``:
        ``c * alpha_s * z_s^{alpha_s - 1} * prod_{t != s} z_t^{alpha_t} / rho_ref_s`` per monomial."""
        z = rho_last / self.rho_ref.to(rho_last.dtype) - 1.0
        P = self._powers(z)
        all_vals = torch.stack(
            [P[..., self.exps[:, s], s] for s in range(self.n_species)], dim=-1
        )   # (..., M, n_species)
        coefs = self.coefs.to(rho_last.dtype)
        grads = []
        for s in range(self.n_species):
            alpha_s = self.exps[:, s]
            lower_idx = (alpha_s - 1).clamp(min=0)
            z_pow = P[..., lower_idx, s]
            others = torch.cat(
                (all_vals[..., :s], all_vals[..., s + 1:]), dim=-1
            )
            prod_others = (
                others.prod(dim=-1) if others.shape[-1] > 0
                else torch.ones_like(z_pow)
            )
            term = alpha_s.to(rho_last.dtype) * z_pow * prod_others
            d_dz_s = term @ coefs
            grads.append(d_dz_s / self.rho_ref[s].to(rho_last.dtype))
        return torch.stack(grads, dim=-1)


class _BasisExpansionRung(nn.Module):
    """Shared Model-B ``forward``: ``d(rho_hat)/dt = +div_hat(M * grad_hat(mu_hat))``, diagonal ``M``."""

    def __init__(self, grid: Tuple[int, int, int], n_species: int, *,
                 nyquist_mask: bool):
        super().__init__()
        self._cache = OpsCache(grid, n_species, nyquist_mask=nyquist_mask)
        self.ops = self._cache.ops
        self.n_species = int(n_species)

    @staticmethod
    def _T_broadcast(T: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """``T`` ``(B,)`` reshaped to broadcast against ``reference``."""
        shape = [T.shape[0]] + [1] * (reference.dim() - 1)
        return T.reshape(shape).to(reference.dtype)

    def forward(self, rho_hat: torch.Tensor, boxes: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        ops = self._cache.ops_for(rho_hat.shape, rho_hat.device)
        N = ops.grid[0] * ops.grid[1] * ops.grid[2]
        rho = ops.irfft(rho_hat * N)
        mu = self.chemical_potential(rho, boxes, T)
        M = self.mobility(rho, T)
        kx, ky, kz = ops.k_axes(boxes)
        mu_hat = ops.rfft(mu) / N
        gx, gy, gz = ops.grad_hat(mu_hat * N, kx, ky, kz)
        grad_mu = torch.stack(
            [ops.irfft(gx), ops.irfft(gy), ops.irfft(gz)], dim=2)
        J = M.unsqueeze(2) * grad_mu
        Jx_hat = ops.rfft(J[:, :, 0]) / N
        Jy_hat = ops.rfft(J[:, :, 1]) / N
        Jz_hat = ops.rfft(J[:, :, 2]) / N
        return ops.div_hat(Jx_hat, Jy_hat, Jz_hat, kx, ky, kz)


class Landau(_BasisExpansionRung):
    """``f = sum_s (1/8) a0 (T - Tc) psi_s^2 + (b/16) psi_s^4``, shared learnable scalars, no cross term."""

    def __init__(self, grid: Tuple[int, int, int], n_species: int,
                 a0: float, b: float, T_c: float, gamma: float, *,
                 nyquist_mask: bool):
        super().__init__(grid, n_species, nyquist_mask=nyquist_mask)
        self.a0 = nn.Parameter(torch.tensor(float(a0)))
        self.b = nn.Parameter(torch.tensor(float(b)))
        self.T_c = nn.Parameter(torch.tensor(float(T_c)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))

    def chemical_potential(self, rho: torch.Tensor, boxes: torch.Tensor,
                            T: torch.Tensor) -> torch.Tensor:
        del boxes
        psi = 2.0 * rho - 1.0
        T_b = self._T_broadcast(T, rho)
        return landau_mu(psi, T_b, self.a0, self.b, self.T_c)

    def bulk_free_energy_density(self, rho: torch.Tensor,
                                  T: torch.Tensor) -> torch.Tensor:
        psi = 2.0 * rho - 1.0
        T_b = self._T_broadcast(T, rho)
        return landau_f(psi, T_b, self.a0, self.b, self.T_c).sum(dim=1, keepdim=True)

    def mobility(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        del T
        return self.gamma * rho * (1.0 - rho)


class FloryHuggins(_BasisExpansionRung):
    """Regular-solution enthalpy (``w``, or Redlich-Kister with ``rho_ref`` at ``n_species > 1``)
    plus the per-channel ideal-mixing entropy."""

    def __init__(self, grid: Tuple[int, int, int], n_species: int,
                 gamma: float, *, w: float | None = None,
                 rho_ref: Sequence[float] | None = None,
                 degree: int = 4, rho_eps: float = 1e-6,
                 nyquist_mask: bool):
        super().__init__(grid, n_species, nyquist_mask=nyquist_mask)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.rho_eps = float(rho_eps)
        if n_species == 1:
            if w is None:
                raise ValueError(
                    "fh at n_species=1 is the regular-solution form and "
                    "requires w; rho_ref/degree are unused there"
                )
            self.w = nn.Parameter(torch.tensor(float(w)))
            self.rk = None
        else:
            if rho_ref is None:
                raise ValueError(
                    "fh at n_species>1 replaces w with the Redlich-Kister "
                    "enthalpy expansion and requires rho_ref, one reference "
                    "density per channel"
                )
            self.rk = _RedlichKister(n_species, rho_ref, degree)
            self.w = None

    def _safe(self, rho: torch.Tensor) -> torch.Tensor:
        return rho.clamp(self.rho_eps, 1.0 - self.rho_eps)

    def chemical_potential(self, rho: torch.Tensor, boxes: torch.Tensor,
                            T: torch.Tensor) -> torch.Tensor:
        del boxes
        T_b = self._T_broadcast(T, rho)
        rho_s = self._safe(rho)
        entropy_mu = ideal_mixing_mu(rho_s, T_b)
        if self.rk is None:
            enthalpy_mu = regular_solution_mu(rho_s, self.w)
        else:
            rho_last = rho.movedim(1, -1)
            d = self.rk.d_drho(rho_last)
            enthalpy_mu = d.movedim(-1, 1)
        return enthalpy_mu + entropy_mu

    def bulk_free_energy_density(self, rho: torch.Tensor,
                                  T: torch.Tensor) -> torch.Tensor:
        T_b = self._T_broadcast(T, rho)
        rho_s = self._safe(rho)
        entropy = ideal_mixing_f(rho_s, T_b).sum(dim=1, keepdim=True)
        if self.rk is None:
            enthalpy = regular_solution_f(rho_s, self.w)
        else:
            rho_last = rho.movedim(1, -1)
            u = self.rk(rho_last)
            enthalpy = u.unsqueeze(1)
        return enthalpy + entropy

    def mobility(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        del T
        rho_s = self._safe(rho)
        return self.gamma * rho_s * (1.0 - rho_s)


#: Rung-1 form names to classes.
MODEL_REGISTRY.register("landau", Landau)
MODEL_REGISTRY.register("fh", FloryHuggins)
