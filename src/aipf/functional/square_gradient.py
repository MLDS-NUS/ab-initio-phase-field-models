"""Square gradient: a local free energy plus a trainable square-gradient term.
``F = int f_loc(rho, T) + (1/2) grad(rho)^T kappa grad(rho)``. The declared ``local`` form (:data:`LOCAL_FORMS`):
``"quadratic_quartic"``, untrained, ``kappa`` an ``n x n`` SPD matrix and a constant mobility;
``"landau"`` and ``"flory_huggins"``, one field, the formulas of :mod:`aipf.functional.landau`,
``kappa = exp(_log_kappa)`` and a declared :class:`aipf.mobility.Mobility`.
The gradient term in ``mu`` uses ``SpectralOps.laplacian``, not ``div_hat(grad_hat(.))`` (Nyquist)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MODEL_REGISTRY
from .landau import (ideal_mixing_f, ideal_mixing_mu, landau_curvature, landau_f, landau_mu,
                     regular_solution_curvature, regular_solution_f, regular_solution_mu)
from aipf.mobility import Mobility
from aipf.spectral import OpsCache, SpectralOps

__all__ = ["SquareGradient", "LOCAL_FORMS"]

#: The declared local forms, each with the bulk constants it reads.
LOCAL_FORMS = ("quadratic_quartic", "landau", "flory_huggins")
_BULK_CONSTANTS = {"quadratic_quartic": (), "landau": ("a0", "b", "T_c"),
                   "flory_huggins": ("w",)}

#: Mobility shapes a closed-form local form composes: none of them is a net, whose width this rung takes no
#: argument for.
_NET_FREE_SHAPES = ("fixed", "constant", "lattice_scalar")


def _to_log(x: float) -> float:
    """``log(x)`` for a positive initial value; the stored parameter of a positive constant."""
    if x <= 0:
        raise ValueError(f"a log-parametrised initial value must be positive, got {x}")
    return math.log(x)


def _inv_softplus(y: float) -> float:
    """``x`` such that ``softplus(x) == y``, for ``y > 0``."""
    return math.log(math.expm1(y))


def _n_lower_triangular(n: int) -> int:
    """Parameter count of an ``n x n`` lower-triangular Cholesky factor."""
    return n * (n + 1) // 2


def _cholesky_to_matrix(raw: torch.Tensor, n: int) -> torch.Tensor:
    """``L @ L.T`` from the flat row-major lower triangle ``raw`` (length ``n*(n+1)/2``).
    Softplus diagonal, free off-diagonal: positive definite for every ``raw``."""
    if raw.shape[-1] != _n_lower_triangular(n):
        raise ValueError(
            f"raw has {raw.shape[-1]} entries; an n={n} Cholesky factor "
            f"needs {_n_lower_triangular(n)}")
    zero = raw[..., 0] * 0.0  # correctly shaped/typed/device zero
    idx = 0
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if j > i:
                row.append(zero)
            else:
                val = raw[..., idx]
                idx += 1
                if j == i:
                    val = F.softplus(val)
                row.append(val)
        rows.append(torch.stack(row, dim=-1))
    L = torch.stack(rows, dim=-2)
    return L @ L.transpose(-1, -2)


def _init_diagonal_cholesky(n: int, diag_value: float) -> torch.Tensor:
    """Raw Cholesky parameters giving ``L @ L.T == diag_value * I``."""
    raw = torch.zeros(_n_lower_triangular(n))
    diag_raw = _inv_softplus(math.sqrt(diag_value))
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                raw[idx] = diag_raw
            idx += 1
    return raw


class SquareGradient(nn.Module):
    """Local free energy plus a trainable square-gradient penalty, ``local`` one of :data:`LOCAL_FORMS`.
    ``kappa_init`` is the starting ``kappa`` (the diagonal of the matrix under ``"quadratic_quartic"``).
    The closed forms read their bulk constants (``a0, b, T_c`` or ``w``), ``rho_eps`` and the mobility
    arguments; ``"quadratic_quartic"`` reads none of them."""

    def __init__(self, grid: Tuple[int, int, int], n_species: int, *,
                 local: str, nyquist_mask: bool, kappa_init: float,
                 w: Optional[float] = None, a0: Optional[float] = None,
                 b: Optional[float] = None, T_c: Optional[float] = None,
                 rho_eps: Optional[float] = None, kB: Optional[float] = None,
                 mobility_prefactor: Optional[str] = None,
                 mobility_shape: Optional[str] = None,
                 mobility_t_form: Optional[str] = None,
                 mobility_shape_init: Optional[float] = None,
                 mobility_t_ref: Optional[float] = None,
                 mobility_activation_energy_init: Optional[float] = None):
        super().__init__()
        if n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {n_species}")
        if local not in LOCAL_FORMS:
            raise ValueError(f"local={local!r} is not one of {LOCAL_FORMS}")
        self.n_species = int(n_species)
        self.local = local
        n = self.n_species
        constants = {"w": w, "a0": a0, "b": b, "T_c": T_c}
        for name, value in constants.items():
            reads = name in _BULK_CONSTANTS[local]
            if reads and value is None:
                raise ValueError(f"local={local!r} requires {name}")
            if not reads and value is not None:
                raise ValueError(f"local={local!r} reads no {name}, got {name}={value!r}")
        mobility = dict(mobility_prefactor=mobility_prefactor, mobility_shape=mobility_shape,
                        mobility_t_form=mobility_t_form, mobility_shape_init=mobility_shape_init,
                        mobility_t_ref=mobility_t_ref,
                        mobility_activation_energy_init=mobility_activation_energy_init, kB=kB)

        self._cache = OpsCache(grid, n, nyquist_mask=nyquist_mask)
        self.ops = self._cache.ops

        if local == "quadratic_quartic":
            given = sorted(k for k, v in {**mobility, "rho_eps": rho_eps}.items()
                           if v is not None)
            if given:
                raise ValueError(f"local='quadratic_quartic' has its own constant mobility and "
                                 f"reads none of {given}")
            self._kappa_chol_raw = nn.Parameter(
                _init_diagonal_cholesky(n, float(kappa_init)))
            # f_loc(rho) = (1/2) rho^T A rho + (1/4) sum_i b_i rho_i^4; b > 0 via softplus.
            self._A_raw = nn.Parameter(torch.zeros(n, n))
            self._b_raw = nn.Parameter(torch.zeros(n))
            self._m_raw = nn.Parameter(torch.zeros(n))
            return

        if n != 1:
            raise ValueError(f"local={local!r} is a one-field form; got n_species={n}")
        if rho_eps is None:
            raise ValueError(f"local={local!r} requires rho_eps, the fraction clamp")
        if mobility_shape not in _NET_FREE_SHAPES:
            raise ValueError(f"local={local!r} composes a mobility of shape one of "
                             f"{_NET_FREE_SHAPES}; got mobility_shape={mobility_shape!r}")
        self.rho_eps = float(rho_eps)
        # Registration order is the published checkpoints' state_dict order.
        if local == "flory_huggins":
            self.w = nn.Parameter(torch.tensor(float(w)))
            self._log_kappa = nn.Parameter(torch.tensor(_to_log(kappa_init)))
        else:
            self._log_a0 = nn.Parameter(torch.tensor(_to_log(a0)))
            self._log_b = nn.Parameter(torch.tensor(_to_log(b)))
            self._log_kappa = nn.Parameter(torch.tensor(_to_log(kappa_init)))
            self.T_c = nn.Parameter(torch.tensor(float(T_c)))
        # `_mobility`: `mobility` is the protocol method (as NonlocalKernel).
        self._mobility = Mobility(
            n, mobility_prefactor, mobility_shape, mobility_t_form,
            shape_init=mobility_shape_init, t_ref=mobility_t_ref,
            activation_energy_init=mobility_activation_energy_init, kB=kB,
            **({"rho_eps": self.rho_eps} if mobility_shape == "lattice_scalar" else {}))

    # -- the constants ------------------------------------------------------------------

    @property
    def closed_form(self) -> bool:
        """A one-field closed form (``"landau"`` or ``"flory_huggins"``), not the polynomial."""
        return self.local != "quadratic_quartic"

    @property
    def a0(self) -> torch.Tensor:
        return torch.exp(self._log_a0)

    @property
    def b(self) -> torch.Tensor:
        return torch.exp(self._log_b)

    def _kappa_matrix(self) -> torch.Tensor:
        """``(n_species, n_species)``, SPD by construction."""
        if self.closed_form:
            return self.kappa.view(1, 1)
        return _cholesky_to_matrix(self._kappa_chol_raw, self.n_species)

    @property
    def kappa(self) -> torch.Tensor:
        """The ``(n, n)`` gradient-penalty matrix; a 0-d tensor at ``n_species == 1``."""
        if self.closed_form:
            return torch.exp(self._log_kappa)
        K = self._kappa_matrix()
        if self.n_species == 1:
            return K[0, 0]
        return K

    def curvature_at_wavevector(self, k: torch.Tensor) -> torch.Tensor:
        """The gradient term's curvature ``kappa |k|^2``, ``(*k.shape, n, n)``."""
        K = self._kappa_matrix().to(k.dtype)
        return (k * k)[..., None, None] * K

    # -- the local free energy ----------------------------------------------------------

    def _A_matrix(self) -> torch.Tensor:
        return 0.5 * (self._A_raw + self._A_raw.transpose(-1, -2))

    @staticmethod
    def _T_like(T: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """``T`` with trailing axes added until it broadcasts against ``reference``, in its dtype."""
        while T.dim() < reference.dim():
            T = T.unsqueeze(-1)
        return T.to(reference.dtype)

    def _closed_mu(self, rho: torch.Tensor, T_b: torch.Tensor) -> torch.Tensor:
        """``d f_loc / d rho`` of a closed form, elementwise."""
        if self.local == "flory_huggins":
            rho_s = rho.clamp(self.rho_eps, 1.0 - self.rho_eps)
            return regular_solution_mu(rho_s, self.w) + ideal_mixing_mu(rho_s, T_b)
        return landau_mu(2.0 * rho - 1.0, T_b, self.a0, self.b, self.T_c)

    def _closed_f(self, rho: torch.Tensor, T_b: torch.Tensor) -> torch.Tensor:
        """``f_loc`` of a closed form, elementwise."""
        if self.local == "flory_huggins":
            rho_s = rho.clamp(self.rho_eps, 1.0 - self.rho_eps)
            return regular_solution_f(rho_s, self.w) + ideal_mixing_f(rho_s, T_b)
        return landau_f(2.0 * rho - 1.0, T_b, self.a0, self.b, self.T_c)

    def _df_loc(self, rho: torch.Tensor) -> torch.Tensor:
        """``d f_loc / d rho_i`` of the polynomial, shape ``(B, n_species, Gx, Gy, Gz)``."""
        A = self._A_matrix()
        b = F.softplus(self._b_raw)
        linear = torch.einsum("ij,bjxyz->bixyz", A, rho)
        cubic = b.view(1, -1, 1, 1, 1) * rho.pow(3)
        return linear + cubic

    def bulk_free_energy_density(self, rho: torch.Tensor,
                                  T: torch.Tensor) -> torch.Tensor:
        """``f_loc(rho, T)``, pointwise, summed over species; no gradient term (the polynomial reads no ``T``)."""
        if self.closed_form:
            return self._closed_f(rho, self._T_like(T, rho))
        A = self._A_matrix()
        b = F.softplus(self._b_raw)
        quad = 0.5 * torch.einsum("ij,bixyz,bjxyz->bxyz", A, rho, rho)
        quart = 0.25 * torch.einsum("i,bixyz->bxyz", b, rho.pow(4))
        return (quad + quart).unsqueeze(1)

    def bulk_free_energy_curve(self, phi, T) -> torch.Tensor:
        """A closed form's ``f_loc`` on a 1-D grid ``phi`` at one ``T``, in the parameters' dtype."""
        self._require_closed_form("bulk_free_energy_curve")
        p = next(self.parameters())
        ph = torch.as_tensor(phi, dtype=p.dtype, device=p.device).reshape(-1)
        return self._closed_f(ph, self._T_like(
            torch.tensor(float(T), dtype=p.dtype, device=p.device), ph))

    def bulk_curvature(self, T: torch.Tensor) -> torch.Tensor:
        """``d2 f_loc / d rho2`` at ``rho = 1/2`` in closed form, differentiable."""
        self._require_closed_form("bulk_curvature")
        if self.local == "flory_huggins":
            return regular_solution_curvature(T.to(self.w.dtype).to(self.w.device), self.w)
        return landau_curvature(T.to(self.T_c.dtype).to(self.T_c.device), self.a0, self.T_c)

    @torch.no_grad()
    def Tc_curvature(self) -> float:
        """The temperature where :meth:`bulk_curvature` vanishes: ``w / 2`` or ``T_c``."""
        self._require_closed_form("Tc_curvature")
        if self.local == "flory_huggins":
            return float(self.w) / 2.0
        return float(self.T_c)

    def _require_closed_form(self, what: str) -> None:
        if not self.closed_form:
            raise NotImplementedError(
                f"{what} is the closed form of 'landau' and 'flory_huggins'; this rung's "
                f"local form is {self.local!r}")

    # -- mu, M and the drift ------------------------------------------------------------

    def _ops_for_real(self, rho: torch.Tensor) -> SpectralOps:
        """The :class:`SpectralOps` for ``rho``'s real-space grid (``Gzr = Gz // 2 + 1``)."""
        Gx, Gy, Gz = (int(s) for s in rho.shape[-3:])
        return self._cache.ops_for((Gx, Gy, Gz // 2 + 1), rho.device)

    def chemical_potential(self, rho: torch.Tensor, boxes: torch.Tensor,
                            T: torch.Tensor) -> torch.Tensor:
        """``mu = d f_loc/d rho - kappa @ laplacian(rho)``, real space."""
        ops = self._ops_for_real(rho)
        k2 = ops.k2(boxes)
        rho_hat = ops.rfft(rho)
        lap_hat = ops.laplacian(rho_hat, k2)
        lap = ops.irfft(lap_hat)
        if self.closed_form:
            # bulk, then minus kappa times the Laplacian
            return self._closed_mu(rho, self._T_like(T, rho)) - self.kappa * lap
        kappa = self._kappa_matrix()
        gradient_term = -torch.einsum("ij,bjxyz->bixyz", kappa, lap)
        return self._df_loc(rho) + gradient_term

    def gradient_energy_density(self, rho: torch.Tensor,
                                 boxes: torch.Tensor) -> torch.Tensor:
        """``(1/2) sum_ij kappa_ij grad(rho_i) . grad(rho_j)``, real space (a diagnostic)."""
        ops = self._ops_for_real(rho)
        kx, ky, kz = ops.k_axes(boxes)
        rho_hat = ops.rfft(rho)
        gx_hat, gy_hat, gz_hat = ops.grad_hat(rho_hat, kx, ky, kz)
        gx, gy, gz = ops.irfft(gx_hat), ops.irfft(gy_hat), ops.irfft(gz_hat)
        dot = (torch.einsum("bixyz,bjxyz->bijxyz", gx, gx)
               + torch.einsum("bixyz,bjxyz->bijxyz", gy, gy)
               + torch.einsum("bixyz,bjxyz->bijxyz", gz, gz))
        kappa = self._kappa_matrix()
        energy = 0.5 * torch.einsum("ij,bijxyz->bxyz", kappa, dot)
        return energy.unsqueeze(1)

    def mobility(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """The polynomial: one positive scalar per species, ``(B, n, Gx, Gy, Gz)``, ``T`` unused.
        A closed form: the composed :class:`~aipf.mobility.Mobility`, ``(B, n, n, Gx, Gy, Gz)``."""
        if self.closed_form:
            B, n = rho.shape[0], self.n_species
            Gx, Gy, Gz = (int(g) for g in rho.shape[-3:])
            rho_flat = rho.permute(0, 2, 3, 4, 1).reshape(-1, n)
            T_flat = T.view(B, 1, 1, 1).expand(B, Gx, Gy, Gz).reshape(-1)
            M_flat = self._mobility(rho_flat, T_flat)
            return M_flat.view(B, Gx, Gy, Gz, n, n).permute(0, 4, 5, 1, 2, 3)
        m = F.softplus(self._m_raw)
        shape = (1, self.n_species) + (1,) * (rho.dim() - 2)
        return m.view(shape).expand_as(rho)

    def forward(self, rho_hat: torch.Tensor, boxes: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        """The Model B right-hand side ``+div(M grad mu)``, k-space in and out."""
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
        # d(rho)/dt = -div(J), J = -M grad(mu): hence +div(M grad(mu)).
        if self.closed_form:
            J = torch.stack([torch.einsum("bijxyz,bjxyz->bixyz", M, grad_mu[:, :, a])
                             for a in range(3)], dim=2)
        else:
            J = M.unsqueeze(2) * grad_mu
        Jx_hat = ops.rfft(J[:, :, 0]) / N
        Jy_hat = ops.rfft(J[:, :, 1]) / N
        Jz_hat = ops.rfft(J[:, :, 2]) / N
        return ops.div_hat(Jx_hat, Jy_hat, Jz_hat, kx, ky, kz)


#: Registered by name.
MODEL_REGISTRY.register("square_gradient", SquareGradient)
