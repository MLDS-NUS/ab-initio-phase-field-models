"""The equivariant neural operator: implemented and equivariance-tested, not trained.
Each layer: a real radial gain ``c_l(|k|, T)``, the channel mix ``a * I + b * (ones - I)``, one scalar
bias, a pointwise ``tanh``; equivariant to translation, channel permutation and axis relabelling."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .base import MODEL_REGISTRY
from aipf.spectral import OpsCache, refuse_two_dimensions

__all__ = ["NeuralOperator"]

#: Hidden width of every radial-gain MLP and the mobility gate MLP (architecture, not physics).
_DEFAULT_HIDDEN = 8

#: Default depth.
_DEFAULT_LAYERS = 3


class _RadialLayer(nn.Module):
    """One equivariant layer: radial gain, permutation-symmetric channel mix, scalar bias, ``tanh``."""

    def __init__(self, hidden: int = _DEFAULT_HIDDEN) -> None:
        super().__init__()
        # Input (|k|, T); one real gain per k-point, shared across channels.
        self._radial = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.mix_diag = nn.Parameter(torch.tensor(1.0))
        self.mix_off = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def _radial_gain(self, k_mag: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """``c(|k|, T)``, real, shape ``(B, 1, Gx, Gy, Gzr)``."""
        t = T.reshape(-1, 1, 1, 1, 1).expand_as(k_mag)
        x = torch.stack((k_mag, t), dim=-1)
        return self._radial(x).squeeze(-1)

    def _mix(self, z: torch.Tensor) -> torch.Tensor:
        """``a * I + b * (ones - I)`` along the channel axis, without materialising the matrix."""
        total = z.sum(dim=1, keepdim=True)
        return self.mix_diag * z + self.mix_off * (total - z)

    def forward(self, v_hat: torch.Tensor, k_mag: torch.Tensor,
                T: torch.Tensor, ops) -> torch.Tensor:
        gain = self._radial_gain(k_mag, T).to(v_hat.dtype)
        z_hat = gain * v_hat
        z = ops.irfft(z_hat)
        mixed = self._mix(z) + self.bias
        out = torch.tanh(mixed)
        return ops.rfft(out)


class NeuralOperator(nn.Module):
    """The equivariant neural operator: a stack of :class:`_RadialLayer`; ``mu`` is the stack's direct output.
    ``bulk_free_energy_density`` is exactly zero (no local part). ``mobility`` is a gated
    ``a * I + b * (ones - I)`` with ``|b| <= a/(n-1)`` (PSD by construction)."""

    def __init__(self, grid: Tuple[int, int, int], n_species: int,
                 n_layers: int = _DEFAULT_LAYERS,
                 hidden: int = _DEFAULT_HIDDEN, *, nyquist_mask: bool) -> None:
        super().__init__()
        refuse_two_dimensions(len(grid), "the neural_operator functional")
        if n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {n_species}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        Gx, Gy, Gz = grid
        self.grid = (int(Gx), int(Gy), int(Gz))
        self.n_species = int(n_species)
        self._ops_cache = OpsCache(self.grid, self.n_species,
                                   nyquist_mask=nyquist_mask)
        self.ops = self._ops_cache.ops
        self.layers = nn.ModuleList(
            _RadialLayer(hidden) for _ in range(n_layers))

        # Mobility a*I + b*(ones-I): softplus a, b = a*tanh(m_off)/(n-1), PSD by construction.
        self.m_diag = nn.Parameter(torch.tensor(0.0))
        self.m_off = nn.Parameter(torch.tensor(0.0))
        self._mobility_gate = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    # -- internal helpers ---------------------------------------------

    def _ops_for_real(self, real_shape, device) -> "torch.nn.Module":
        Gx, Gy, Gz = (int(s) for s in real_shape[-3:])
        return self._ops_cache.ops_for((Gx, Gy, Gz // 2 + 1), device=device)

    def _mu_field(self, rho: torch.Tensor, boxes: torch.Tensor,
                  T: torch.Tensor, ops) -> torch.Tensor:
        k_mag = torch.sqrt(torch.clamp(ops.k2(boxes), min=0.0))
        v_hat = ops.rfft(rho)
        for layer in self.layers:
            v_hat = layer(v_hat, k_mag, T, ops)
        return ops.irfft(v_hat)

    def _species_symmetric_matrix(self) -> torch.Tensor:
        n = self.n_species
        a = torch.nn.functional.softplus(self.m_diag)
        if n > 1:
            b = a * torch.tanh(self.m_off) / (n - 1)
        else:
            b = torch.zeros_like(a)
        eye = torch.eye(n, device=a.device, dtype=a.dtype)
        ones = torch.ones(n, n, device=a.device, dtype=a.dtype)
        return a * eye + b * (ones - eye)

    # -- FreeEnergyModel protocol ---------------------------------------

    def chemical_potential(self, rho: torch.Tensor, boxes: torch.Tensor,
                            T: torch.Tensor) -> torch.Tensor:
        ops = self._ops_for_real(rho.shape, rho.device)
        return self._mu_field(rho, boxes, T, ops)

    def bulk_free_energy_density(self, rho: torch.Tensor,
                                  T: torch.Tensor) -> torch.Tensor:
        del T
        return torch.zeros(rho.shape[0], 1, *rho.shape[2:],
                            dtype=rho.dtype, device=rho.device)

    def mobility(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        rho_bar = rho.mean(dim=1, keepdim=True)
        t = T.reshape(-1, 1, 1, 1, 1).expand_as(rho_bar)
        gate_in = torch.stack((rho_bar, t), dim=-1)
        gate = torch.sigmoid(self._mobility_gate(gate_in).squeeze(-1))
        Mss = self._species_symmetric_matrix()
        n = self.n_species
        Mss = Mss.view(1, n, n, 1, 1, 1)
        return gate.unsqueeze(1) * Mss

    def forward(self, rho_hat: torch.Tensor, boxes: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        ops = self._ops_cache.ops_for(rho_hat.shape, device=rho_hat.device)
        rho = ops.irfft(rho_hat)
        mu = self._mu_field(rho, boxes, T, ops)
        M = self.mobility(rho, T)

        kx, ky, kz = ops.k_axes(boxes)
        mu_hat = ops.rfft(mu)
        gx_hat, gy_hat, gz_hat = ops.grad_hat(mu_hat, kx, ky, kz)
        gx, gy, gz = ops.irfft(gx_hat), ops.irfft(gy_hat), ops.irfft(gz_hat)

        # d(rho)/dt = -div(J), J = -M grad(mu): hence +div(M grad(mu)).
        Jx = torch.einsum("bijxyz,bjxyz->bixyz", M, gx)
        Jy = torch.einsum("bijxyz,bjxyz->bixyz", M, gy)
        Jz = torch.einsum("bijxyz,bjxyz->bixyz", M, gz)

        Jx_hat, Jy_hat, Jz_hat = ops.rfft(Jx), ops.rfft(Jy), ops.rfft(Jz)
        return ops.div_hat(Jx_hat, Jy_hat, Jz_hat, kx, ky, kz)


#: Registered under the ladder's name for this rung (`aipf.system.FUNCTIONAL_FORMS`).
MODEL_REGISTRY.register("neural_operator", NeuralOperator)
