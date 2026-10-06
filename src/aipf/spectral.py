"""Batched, per-sample-box k-space operators on a fixed integer mode grid.
``k_axis[b] = 2*pi * n_axis / L_axis[b]``; only the integer mode grids are batch-independent.
Every operator set declares whether grad and div zero the Nyquist wavenumber (``nyquist_mask``, no default).
The number of spatial axes is declared by the grid's length and read back as ``ops.ndim``, never from a
tensor's shape: :class:`SpectralOps` is three-dimensional, :class:`SpectralOps2D` two-dimensional, and
:func:`make_ops` picks one by ``len(grid)``."""
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

    #: Spatial axes; a class attribute, so the state dict is the one this class always had.
    ndim = 3

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


class SpectralOps2D(nn.Module):
    """k-space operators for one real-space ``grid`` ``(Gx, Gy)``: :class:`SpectralOps` with two axes.
    The half (``rfft``) axis is ``y``, so a state is ``(B, n, Gx, Gyr)``, ``Gyr = Gy // 2 + 1``, and the
    boxes are ``(B, 2)`` ``(Lx, Ly)``. The buffers mean what the three-dimensional ones do: ``NX``
    ``(Gx, 1)``, ``NY`` ``(1, Gyr)`` and ``MULT`` ``(1, 1, 1, Gyr)`` are persistent, the odd-order masks are
    not. ``nyquist_mask`` (required, :data:`NYQUIST_MASKS`) as there."""

    #: Spatial axes; a class attribute, not a buffer.
    ndim = 2

    def __init__(self, grid: Tuple[int, int], n_species: int, *,
                 nyquist_mask: bool):
        super().__init__()
        self.nyquist_mask = check_nyquist_mask(nyquist_mask)
        if n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {n_species}")
        Gx, Gy = grid
        self.grid = (int(Gx), int(Gy))
        self.n_species = int(n_species)
        self.Gyr = Gy // 2 + 1
        nx = torch.fft.fftfreq(Gx) * Gx
        ny = torch.arange(self.Gyr, dtype=torch.float32)    # rfft half axis
        self.register_buffer("NX", nx.view(Gx, 1))
        self.register_buffer("NY", ny.view(1, self.Gyr))
        # Real-FFT multiplicity along the half axis: sum(|full fft|^2) == sum(MULT * |rfft|^2).
        mult = torch.full((1, 1, 1, self.Gyr), 2.0)
        mult[..., 0] = 1.0
        if Gy % 2 == 0:
            mult[..., -1] = 1.0
        self.register_buffer("MULT", mult)
        mx = torch.ones(Gx)
        if Gx % 2 == 0 and nyquist_mask:
            mx[Gx // 2] = 0.0
        my = torch.ones(self.Gyr)
        if Gy % 2 == 0 and nyquist_mask:
            my[-1] = 0.0               # rfft half axis: Nyquist is the last index
        self.register_buffer("MX_ODD", mx.view(Gx, 1), persistent=False)
        self.register_buffer("MY_ODD", my.view(1, self.Gyr), persistent=False)

    def k_axes(self, boxes: torch.Tensor):
        """Per-sample angular wavenumbers from ``boxes`` ``(B, 2)``; ``kx, ky`` each ``(B, 1, Gx, Gyr)``."""
        if boxes.dim() != 2 or boxes.shape[1] != 2:
            raise ValueError(
                f"a two-dimensional operator set reads boxes (B, 2), (Lx, Ly); got shape "
                f"{tuple(boxes.shape)}")
        B = boxes.shape[0]
        L = boxes.to(self.NX.dtype) if boxes.dtype != self.NX.dtype else boxes
        inv = TWO_PI / L                                     # (B,2)
        shape = (B, 1, self.grid[0], self.Gyr)
        kx = (self.NX * inv[:, 0].view(B, 1, 1, 1)).expand(shape)
        ky = (self.NY * inv[:, 1].view(B, 1, 1, 1)).expand(shape)
        return kx, ky

    def k2(self, boxes: torch.Tensor) -> torch.Tensor:
        kx, ky = self.k_axes(boxes)
        return kx * kx + ky * ky

    def band_mask(self, boxes: torch.Tensor, k_max: float) -> torch.Tensor:
        """Boolean mask of modes with ``0 < |k| <= k_max``, per sample."""
        k2 = self.k2(boxes)
        return (k2 > 0) & (k2 <= k_max * k_max)

    def sigma_filter(self, boxes: torch.Tensor, sigma: float) -> torch.Tensor:
        return torch.exp(-self.k2(boxes) * (sigma * sigma / 2.0))

    def rfft(self, f: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfftn(f, dim=(-2, -1))

    def irfft(self, f_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfftn(f_hat, s=self.grid, dim=(-2, -1))

    def grad_hat(self, f_hat, kx, ky):
        return (1j * (kx * self.MX_ODD) * f_hat,
                1j * (ky * self.MY_ODD) * f_hat)

    def div_hat(self, Jx_hat, Jy_hat, kx, ky):
        return 1j * (kx * self.MX_ODD * Jx_hat
                     + ky * self.MY_ODD * Jy_hat)

    def laplacian(self, f_hat: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
        """``-|k|^2 * f_hat``; even-order, so no Nyquist mask."""
        return -k2 * f_hat


def make_ops(grid, n_species: int, *, nyquist_mask: bool):
    """The operator set for ``grid``: :class:`SpectralOps` for three axes, :class:`SpectralOps2D` for two."""
    if len(grid) == 3:
        return SpectralOps(grid, n_species, nyquist_mask=nyquist_mask)
    if len(grid) == 2:
        return SpectralOps2D(grid, n_species, nyquist_mask=nyquist_mask)
    raise ValueError(
        f"grid {tuple(grid)!r} has {len(grid)} axes; an operator set has three "
        f"(Gx, Gy, Gz) or two (Gx, Gy)")


def k_squared(ks) -> torch.Tensor:
    """``|k|^2`` from :meth:`SpectralOps.k_axes` (or the 2D one), summed in axis order."""
    if len(ks) == 3:
        kx, ky, kz = ks
        return kx * kx + ky * ky + kz * kz
    kx, ky = ks
    return kx * kx + ky * ky


#: Per number of spatial axes, the literal index and permutation each call site uses (the
#: three-dimensional entries are the spellings the code always had):
#: the ``k = 0`` entry of a half-spectrum,
DC = {3: (Ellipsis, 0, 0, 0), 2: (Ellipsis, 0, 0)}
#: ``(B, n, *grid)`` -> ``(B, *grid, n)`` and back,
CHANNEL_LAST = {3: (0, 2, 3, 4, 1), 2: (0, 2, 3, 1)}
CHANNEL_SECOND = {3: (0, 4, 1, 2, 3), 2: (0, 3, 1, 2)}
#: ``(B, n, n, *grid)`` -> ``(B, *grid, n, n)`` and back,
MATRIX_LAST = {3: (0, 3, 4, 5, 1, 2), 2: (0, 3, 4, 1, 2)}
MATRIX_SECOND = {3: (0, 4, 5, 1, 2, 3), 2: (0, 3, 4, 1, 2)}
#: and the flux ``J_i = M_ij g_j`` per cell.
FLUX_EINSUM = {3: "bijxyz,bjxyz->bixyz", 2: "bijxy,bjxy->bixy"}


def half_spectrum_grid(half_shape, declared=None) -> Tuple[int, ...]:
    """The real-space grid of a two-dimensional half spectrum ``(Gx, Gyr)``: the ``declared`` grid when its
    half spectrum has that shape (an odd ``Gy`` included), else ``(Gx, 2 * (Gyr - 1))``."""
    Gx, Gyr = (int(s) for s in half_shape)
    if declared is not None and len(declared) == 2:
        dx, dy = (int(g) for g in declared)
        if (dx, dy // 2 + 1) == (Gx, Gyr):
            return (dx, dy)
    return (Gx, 2 * (Gyr - 1))


def ops_ndim(ops) -> int:
    """``ops.ndim``; 3 for an operator set that declares none (one written before two dimensions)."""
    return int(getattr(ops, "ndim", 3))


#: The operator classes, one per number of spatial axes.
OPS_CLASSES = (SpectralOps, SpectralOps2D)


def model_ndim(model) -> int:
    """The spatial axes a model's operator set was built on (``model.ops.ndim``); 3 for a model with none."""
    return int(getattr(getattr(model, "ops", None), "ndim", 3))


def refuse_two_dimensions(ndim: int, what: str) -> None:
    """``NotImplementedError`` naming ``what`` when ``ndim == 2``; nothing otherwise."""
    if ndim == 2:
        raise NotImplementedError(
            f"{what} is three-dimensional only: it is written for a (Gx, Gy, Gz) grid and the "
            f"(B, 3) boxes of a bulk simulation cell, and a two-dimensional model has neither. "
            f"See docs/reference/functional.md, 'Two dimensions', for what runs in 2D")


class OpsCache:
    """A per-grid-shape cache of :class:`SpectralOps`, seeded with the owner's grid and grown lazily.
    Only the seed (:attr:`ops`) is a registered submodule; later entries are moved device on lookup.
    ``nyquist_mask`` (required, :data:`NYQUIST_MASKS`) applies to every entry. A two-axis seed ``grid``
    makes a cache of :class:`SpectralOps2D`; :attr:`ndim` is the seed's, and every lookup reads that many
    trailing axes."""

    def __init__(self, grid: Tuple[int, int, int], n_species: int, *,
                 nyquist_mask: bool):
        if len(grid) == 2:
            self.grid = (int(grid[0]), int(grid[1]))
        else:
            self.grid = (int(grid[0]), int(grid[1]), int(grid[2]))
        self.ndim = len(self.grid)
        self.n_species = int(n_species)
        self.nyquist_mask = check_nyquist_mask(nyquist_mask)
        self.ops = self._new(self.grid)
        self._cache: Dict[Tuple[int, int, int], SpectralOps] = {
            self.grid: self.ops}

    def ops_for(self, shape, device=None) -> SpectralOps:
        """The :class:`SpectralOps` for an rfft shape ``(..., Gx, Gy, Gzr)``; ``Gz = 2 * (Gzr - 1)``.
        In two dimensions ``(..., Gx, Gyr)``: the seed grid when it has that half spectrum, else
        ``Gy = 2 * (Gyr - 1)`` (:func:`half_spectrum_grid`)."""
        if self.ndim == 2:
            return self.ops_for_grid(half_spectrum_grid(shape[-2:], self.grid), device)
        Gx, Gy, Gzr = (int(s) for s in shape[-3:])
        return self.ops_for_grid((Gx, Gy, 2 * (Gzr - 1)), device)

    def ops_for_grid(self, grid, device=None) -> SpectralOps:
        """The :class:`SpectralOps` for a full real-space grid (use this for odd ``Gz`` or ``(1, 1, 1)``)."""
        grid = tuple(int(g) for g in grid)
        if self.ndim == 2:
            if len(grid) != 2 or min(grid) < 1:
                raise ValueError(
                    f"grid must be two positive integers (this cache is two-dimensional), "
                    f"got {grid!r}")
        elif len(grid) != 3 or min(grid) < 1:
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
        return make_ops(grid, self.n_species, nyquist_mask=self.nyquist_mask)
