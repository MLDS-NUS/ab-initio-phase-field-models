"""Batched, per-sample-box k-space operators on a fixed integer mode grid.
``k_axis[b] = 2*pi * n_axis / L_axis[b]``; only the integer mode grids are batch-independent.
Every operator set declares whether grad and div zero the Nyquist wavenumber (``nyquist_mask``, no default)."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

TWO_PI = 2.0 * math.pi

#: Admissible ``nyquist_mask`` values: zero the Nyquist wavenumber in grad and div, or keep it.
NYQUIST_MASKS = (True, False)


def check_nyquist_mask(value) -> bool:
    """``value`` if it is one of :data:`NYQUIST_MASKS`; otherwise ``ValueError`` naming both."""
    if not isinstance(value, bool):
        raise ValueError(
            f"nyquist_mask={value!r} is not one of {NYQUIST_MASKS}: True zeroes "
            f"the Nyquist wavenumber in the odd-order operators (grad, div), "
            f"False keeps it. Which one a model was trained with is declared, "
            f"never assumed")
    return value


class SpectralOps(nn.Module):
    """k-space operators for one real-space ``grid`` ``(Gx, Gy, Gz)``; ``n_species`` is recorded only.
    ``nyquist_mask`` (required, :data:`NYQUIST_MASKS`): ``False`` makes ``grad_hat``/``div_hat`` plain ``i k``."""

    def __init__(self, grid: Tuple[int, int, int], n_species: int, *,
                 nyquist_mask: bool):
        super().__init__()
        self.nyquist_mask = check_nyquist_mask(nyquist_mask)
        if n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {n_species}")
        Gx, Gy, Gz = grid
        self.grid = (int(Gx), int(Gy), int(Gz))
        self.n_species = int(n_species)
        self.Gzr = Gz // 2 + 1
        nx = torch.fft.fftfreq(Gx) * Gx                     # 0..7,-8..-1
        ny = torch.fft.fftfreq(Gy) * Gy
        nz = torch.arange(self.Gzr, dtype=torch.float32)    # rfft half axis
        self.register_buffer("NX", nx.view(Gx, 1, 1))
        self.register_buffer("NY", ny.view(1, Gy, 1))
        self.register_buffer("NZ", nz.view(1, 1, self.Gzr))
        # Real-FFT multiplicity: sum(|full fft|^2) == sum(MULT * |rfft|^2).
        mult = torch.full((1, 1, 1, 1, self.Gzr), 2.0)
        mult[..., 0] = 1.0
        if Gz % 2 == 0:
            mult[..., -1] = 1.0
        self.register_buffer("MULT", mult)
        # Odd-order operators zero the Nyquist wavenumber (i*k of a real coefficient); even-order keep it.
        mx = torch.ones(Gx)
        if Gx % 2 == 0 and nyquist_mask:
            mx[Gx // 2] = 0.0          # fftfreq puts Nyquist (-Gx/2) at index Gx/2
        my = torch.ones(Gy)
        if Gy % 2 == 0 and nyquist_mask:
            my[Gy // 2] = 0.0
        mz = torch.ones(self.Gzr)
        if Gz % 2 == 0 and nyquist_mask:
            mz[-1] = 0.0               # rfft half axis: Nyquist is the last index
        # Non-persistent: saved checkpoints must load strictly (NX/NY/NZ/MULT stay persistent for the same
        # reason).
        self.register_buffer("MX_ODD", mx.view(Gx, 1, 1), persistent=False)
        self.register_buffer("MY_ODD", my.view(1, Gy, 1), persistent=False)
        self.register_buffer("MZ_ODD", mz.view(1, 1, self.Gzr),
                             persistent=False)

    def k_axes(self, boxes: torch.Tensor):
        """Per-sample angular wavenumbers from ``boxes`` ``(B, 3)``;
        ``kx, ky, kz`` are each ``(B, 1, Gx, Gy, Gzr)``."""
        B = boxes.shape[0]
        L = boxes.to(self.NX.dtype) if boxes.dtype != self.NX.dtype else boxes
        inv = TWO_PI / L                                     # (B,3)
        shape = (B, 1, *self.grid[:2], self.Gzr)
        kx = (self.NX * inv[:, 0].view(B, 1, 1, 1, 1)).expand(shape)
        ky = (self.NY * inv[:, 1].view(B, 1, 1, 1, 1)).expand(shape)
        kz = (self.NZ * inv[:, 2].view(B, 1, 1, 1, 1)).expand(shape)
        return kx, ky, kz

    def k2(self, boxes: torch.Tensor) -> torch.Tensor:
        kx, ky, kz = self.k_axes(boxes)
        return kx * kx + ky * ky + kz * kz

    def band_mask(self, boxes: torch.Tensor, k_max: float) -> torch.Tensor:
        """Boolean mask of modes with ``0 < |k| <= k_max``, per sample."""
        k2 = self.k2(boxes)
        return (k2 > 0) & (k2 <= k_max * k_max)

    def sigma_filter(self, boxes: torch.Tensor, sigma: float) -> torch.Tensor:
        return torch.exp(-self.k2(boxes) * (sigma * sigma / 2.0))

    def rfft(self, f: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfftn(f, dim=(-3, -2, -1))

    def irfft(self, f_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfftn(f_hat, s=self.grid, dim=(-3, -2, -1))

    # Masking is per axis, not by |k|.
    def grad_hat(self, f_hat, kx, ky, kz):
        return (1j * (kx * self.MX_ODD) * f_hat,
                1j * (ky * self.MY_ODD) * f_hat,
                1j * (kz * self.MZ_ODD) * f_hat)

    def div_hat(self, Jx_hat, Jy_hat, Jz_hat, kx, ky, kz):
        return 1j * (kx * self.MX_ODD * Jx_hat
                     + ky * self.MY_ODD * Jy_hat
                     + kz * self.MZ_ODD * Jz_hat)

    def laplacian(self, f_hat: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
        """``-|k|^2 * f_hat``; even-order, so no Nyquist mask."""
        return -k2 * f_hat


class OpsCache:
    """A per-grid-shape cache of :class:`SpectralOps`, seeded with the owner's grid and grown lazily.
    Only the seed (:attr:`ops`) is a registered submodule; later entries are moved device on lookup.
    ``nyquist_mask`` (required, :data:`NYQUIST_MASKS`) applies to every entry."""

    def __init__(self, grid: Tuple[int, int, int], n_species: int, *,
                 nyquist_mask: bool):
        self.grid = (int(grid[0]), int(grid[1]), int(grid[2]))
        self.n_species = int(n_species)
        self.nyquist_mask = check_nyquist_mask(nyquist_mask)
        self.ops = self._new(self.grid)
        self._cache: Dict[Tuple[int, int, int], SpectralOps] = {
            self.grid: self.ops}

    def ops_for(self, shape, device=None) -> SpectralOps:
        """The :class:`SpectralOps` for an rfft shape ``(..., Gx, Gy, Gzr)``; ``Gz = 2 * (Gzr - 1)``."""
        Gx, Gy, Gzr = (int(s) for s in shape[-3:])
        return self.ops_for_grid((Gx, Gy, 2 * (Gzr - 1)), device)

    def ops_for_grid(self, grid, device=None) -> SpectralOps:
        """The :class:`SpectralOps` for a full real-space grid (use this for odd ``Gz`` or ``(1, 1, 1)``)."""
        grid = tuple(int(g) for g in grid)
        if len(grid) != 3 or min(grid) < 1:
            raise ValueError(
                f"grid must be three positive integers, got {grid!r}")
        if device is None:
            device = self.ops.NX.device
        want = torch.device(device)
        ops = self._cache.get(grid)
        if ops is None or ops.NX.device != want:
            ops = (self.ops if grid == self.grid
                   else self._new(grid)).to(want)
            self._cache[grid] = ops
        return ops

    def _new(self, grid) -> SpectralOps:
        return SpectralOps(grid, self.n_species, nyquist_mask=self.nyquist_mask)
