"""Atom positions to a smooth density field on a periodic grid.

    rho_c(x) = sum_i g_sigma(x - r_i)   over the atoms of channel c, g_sigma the normalised Gaussian

A number density: a channel integrates to its atom count. Channel first, ``(n_species, Gx, Gy, Gz)``.
:data:`METHODS`: ``"mesh"`` (cloud-in-cell, ``exp(-k^2 sigma^2 / 2)`` over ``sinc^2(k dx / 2)``)
and ``"direct"`` (nearest-image sum truncated at ``cutoff_in_sigma``, a reference).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

TWO_PI = 2.0 * math.pi

#: The named coarse-graining methods.
METHODS = ("mesh", "direct")

#: The named compute devices. Only ``"auto"`` adapts: an absent named accelerator raises.
DEVICES = ("auto", "cpu", "cuda")

#: Guard on the shape-factor divisor. It is never the active term.
_SHAPE_FACTOR_FLOOR = 1e-12

#: Grid points per block of the direct sum. Peak memory only: the answer does not depend on it.
_DEFAULT_BLOCK_SIZE = 4096


def channel_atom_types(system) -> tuple[int, ...]:
    """The dump type of each density channel of ``system``, in channel order."""
    if not system.atom_types:
        raise ValueError(
            f"system {system.name!r} declares no atom_types, so there is no "
            f"way to tell which atoms of a dump feed which channel"
        )
    return tuple(system.atom_types[name] for name in system.species)


def density_field(
    positions,
    types,
    box_bounds,
    grid,
    *,
    sigma: float,
    atom_types: Sequence[int],
    method: str,
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
    cutoff_in_sigma: float | None = None,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Coarse-grain one frame into a ``(n_species, Gx, Gy, Gz)`` density field on the resolved device.

    ``box_bounds`` ``(3, 2)`` periodic, ``sigma`` in coordinate units, ``grid`` one integer or three.
    ``cutoff_in_sigma`` is required by ``"direct"``, refused by ``"mesh"``.
    """
    if method not in METHODS:
        raise ValueError(f"method={method!r} is not one of {METHODS}")
    resolved = _resolve_device(device)
    grid = _as_grid(grid)
    sigma = float(sigma)
    if not sigma > 0.0:
        raise ValueError(f"sigma={sigma!r} is not positive")
    channels = _as_atom_types(atom_types)
    lo, lengths = _box(box_bounds)
    pos, typ = _as_frame(positions, types, dtype, resolved)

    if method == "direct":
        if cutoff_in_sigma is None:
            raise ValueError(
                "method='direct' needs cutoff_in_sigma: the sum has to stop "
                "somewhere, and where it stops is the whole accuracy of it"
            )
        cutoff = float(cutoff_in_sigma)
        if not cutoff > 0.0:
            raise ValueError(f"cutoff_in_sigma={cutoff!r} is not positive")
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError(f"block_size={block_size!r} is below one")
        return _density_direct(pos, typ, lo, lengths, grid, sigma, channels,
                               cutoff, block_size, dtype, resolved)

    if cutoff_in_sigma is not None:
        raise ValueError(
            "method='mesh' has no cutoff_in_sigma to honour: it applies the "
            "kernel in k space and truncates nothing. Accepting the argument "
            "and ignoring it would report a truncation that never happened"
        )
    return _density_mesh(pos, typ, lo, lengths, grid, sigma, channels,
                         dtype, resolved)


def binder_cumulant(density, *, contrast: Sequence[float]) -> float:
    """The spatial Binder cumulant ``1 - <psi^4> / (3 <psi^2>^2)``, ``psi = sum_c contrast[c] * density[c]``.

    Raw moments, accumulated in double.
    """
    field = torch.as_tensor(density)
    if field.ndim != 4:
        raise ValueError(
            f"density has shape {tuple(field.shape)}, not (n_species, Gx, "
            f"Gy, Gz): the cumulant needs to know which axis is the channel"
        )
    weights = tuple(float(w) for w in contrast)
    if len(weights) != field.shape[0]:
        raise ValueError(
            f"contrast has {len(weights)} weights for {field.shape[0]} "
            f"channels: the order parameter has to say something about each"
        )
    field = field.to(torch.float64)
    psi = torch.zeros(field.shape[1:], dtype=torch.float64,
                      device=field.device)
    for weight, channel in zip(weights, field):
        psi = psi + weight * channel
    second = float((psi * psi).mean())
    if second == 0.0:
        return 0.0
    fourth = float((psi * psi * psi * psi).mean())
    return 1.0 - fourth / (3.0 * second * second)


def _resolve_device(device: str) -> torch.device:
    """A named device, or a refusal. A named accelerator that is absent raises."""
    if device not in DEVICES:
        raise ValueError(f"device={device!r} is not one of {DEVICES}")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was asked for and no accelerator is visible. Pass "
            "device='auto' to accept whatever this machine has; a silent fall "
            "back changes the arithmetic under a caller who asked for one "
            "thing and was given another"
        )
    return torch.device(device)


class TwoDimensionalGridError(NotImplementedError, ValueError):
    """A two-entry grid: not implemented (the deposit is three-dimensional), and a ``ValueError`` as a grid of
    any other wrong length is."""


def _as_grid(grid) -> tuple[int, int, int]:
    """One integer, or three, into a per-axis resolution."""
    try:
        n_axes = len(grid)
    except TypeError:
        axes = (int(grid),) * 3
    else:
        if n_axes == 2:
            raise TwoDimensionalGridError(
                f"grid has 2 entries: the density field is deposited from three-dimensional "
                f"atom positions onto a (Gx, Gy, Gz) mesh, and a two-dimensional grid is not "
                f"implemented here; deposit in 3D and average along the axis instead")
        if n_axes != 3:
            raise ValueError(
                f"grid has {n_axes} entries: it is one resolution for a "
                f"cubic grid or three, one per axis"
            )
        axes = tuple(int(v) for v in grid)
    if any(v < 1 for v in axes):
        raise ValueError(f"grid={axes} has an axis below one point")
    return axes


def _as_atom_types(atom_types: Sequence[int]) -> tuple[int, ...]:
    """The declared dump type per channel. A repeated type is refused."""
    channels = tuple(int(t) for t in atom_types)
    if not channels:
        raise ValueError(
            "atom_types is empty: with no dump type declared there is no "
            "channel to fill and the result would be an empty field"
        )
    if len(set(channels)) != len(channels):
        repeated = sorted({t for t in channels if channels.count(t) > 1})
        raise ValueError(
            f"atom_types={channels} repeats {repeated}: the same atoms would "
            f"be deposited into more than one channel and the total density "
            f"would be counted twice"
        )
    return channels


def _box(box_bounds):
    """Per-axis origin and edge length, in double precision."""
    bounds = torch.as_tensor(box_bounds, dtype=torch.float64)
    if bounds.shape != (3, 2):
        raise ValueError(
            f"box_bounds has shape {tuple(bounds.shape)}, not (3, 2): one "
            f"[lo, hi] pair per axis"
        )
    lo = tuple(float(v) for v in bounds[:, 0])
    lengths = tuple(float(hi) - low for low, hi in zip(lo, bounds[:, 1]))
    for axis, length in enumerate(lengths):
        if not length > 0.0:
            raise ValueError(
                f"box_bounds axis {axis} has length {length!r}: an edge that "
                f"is not positive gives a grid spacing that is not positive"
            )
    return lo, lengths


def _as_frame(positions, types, dtype: torch.dtype, device: torch.device):
    """Positions and dump types as tensors, shapes agreeing."""
    pos = torch.as_tensor(positions, dtype=dtype, device=device)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"positions has shape {tuple(pos.shape)}, not (N, 3)"
        )
    typ = torch.as_tensor(types, device=device).to(torch.long)
    if typ.ndim != 1 or typ.shape[0] != pos.shape[0]:
        raise ValueError(
            f"types has shape {tuple(typ.shape)} for {pos.shape[0]} "
            f"positions: one dump type per atom"
        )
    return pos, typ


def _mesh_kernel(grid, spacing, sigma, dtype, device) -> torch.Tensor:
    """The k-space factor ``exp(-k^2 sigma^2 / 2) / sinc^2(k dx / 2)``, deconvolving the deposition."""
    axes = []
    for points, step in zip(grid, spacing):
        axis = TWO_PI * torch.fft.fftfreq(points, d=step, dtype=dtype)
        axes.append(axis.to(device))
    k_x, k_y, k_z = axes
    k_squared = (k_x[:, None, None] * k_x[:, None, None]
                 + k_y[None, :, None] * k_y[None, :, None]
                 + k_z[None, None, :] * k_z[None, None, :])
    gaussian = torch.exp(-0.5 * (sigma ** 2) * k_squared)

    shape_x = torch.sinc(k_x * spacing[0] / TWO_PI) ** 2
    shape_y = torch.sinc(k_y * spacing[1] / TWO_PI) ** 2
    shape_z = torch.sinc(k_z * spacing[2] / TWO_PI) ** 2
    shape = (shape_x[:, None, None] * shape_y[None, :, None]
             * shape_z[None, None, :])
    return gaussian / (shape + _SHAPE_FACTOR_FLOOR)


def _deposit(fractional, grid) -> torch.Tensor:
    """Cloud-in-cell deposition into the eight surrounding cells, ``fractional`` wrapped into ``[0, G)``."""
    points = torch.tensor(grid, dtype=torch.long, device=fractional.device)
    lower = torch.floor(fractional).long()
    frac = fractional - lower.to(fractional.dtype)
    lower = lower % points
    upper = (lower + 1) % points
    stride_y, stride_z = grid[1], grid[2]

    flat = torch.zeros(grid[0] * grid[1] * grid[2],
                       dtype=fractional.dtype, device=fractional.device)
    for corner_x in (0, 1):
        for corner_y in (0, 1):
            for corner_z in (0, 1):
                weight_x = frac[:, 0] if corner_x else 1 - frac[:, 0]
                weight_y = frac[:, 1] if corner_y else 1 - frac[:, 1]
                weight_z = frac[:, 2] if corner_z else 1 - frac[:, 2]
                index_x = upper[:, 0] if corner_x else lower[:, 0]
                index_y = upper[:, 1] if corner_y else lower[:, 1]
                index_z = upper[:, 2] if corner_z else lower[:, 2]
                index = (index_x * stride_y + index_y) * stride_z + index_z
                flat.scatter_add_(0, index, weight_x * weight_y * weight_z)
    return flat


def _density_mesh(pos, typ, lo, lengths, grid, sigma, channels, dtype, device):
    spacing = tuple(length / points for length, points in zip(lengths, grid))
    cell_volume = spacing[0] * spacing[1] * spacing[2]

    origin = torch.tensor(lo, dtype=dtype, device=device)
    step = torch.tensor(spacing, dtype=dtype, device=device)
    points = torch.tensor(grid, dtype=dtype, device=device)
    fractional = (pos - origin) / step
    fractional = fractional - torch.floor(fractional / points) * points

    kernel = _mesh_kernel(grid, spacing, sigma, dtype, device)
    out = torch.zeros((len(channels), *grid), dtype=dtype, device=device)
    for channel, dump_type in enumerate(channels):
        selected = typ == dump_type
        if not bool(selected.any()):
            continue
        flat = _deposit(fractional[selected], grid)
        counts = flat.view(*grid) / cell_volume
        out[channel] = torch.fft.ifftn(torch.fft.fftn(counts) * kernel).real
    return out


def _grid_points(lo, lengths, grid, dtype, device) -> torch.Tensor:
    """``(Gx*Gy*Gz, 3)`` grid nodes ``lo + i * spacing``, row-major, the points the mesh method samples."""
    axes = []
    for low, length, count in zip(lo, lengths, grid):
        step = length / count
        axis = torch.arange(count, dtype=torch.float64) * step + low
        axes.append(axis.to(dtype=dtype, device=device))
    mesh_x, mesh_y, mesh_z = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1),
                        mesh_z.reshape(-1)], dim=1)


def _density_direct(pos, typ, lo, lengths, grid, sigma, channels, cutoff,
                    block_size, dtype, device):
    nodes = _grid_points(lo, lengths, grid, dtype, device)
    edges = torch.tensor(lengths, dtype=dtype, device=device)
    n_nodes = nodes.shape[0]

    normalisation = 1.0 / (2.0 * math.pi * sigma ** 2) ** 1.5
    two_sigma_squared = 2.0 * sigma ** 2
    cutoff_squared = (cutoff * sigma) ** 2

    out = torch.zeros((len(channels), *grid), dtype=dtype, device=device)
    for channel, dump_type in enumerate(channels):
        selected = typ == dump_type
        if not bool(selected.any()):
            continue
        atoms = pos[selected]
        summed = torch.zeros(n_nodes, dtype=dtype, device=device)
        for start in range(0, n_nodes, block_size):
            stop = min(start + block_size, n_nodes)
            delta = nodes[start:stop].unsqueeze(1) - atoms.unsqueeze(0)
            delta = delta - torch.round(delta / edges) * edges
            distance_squared = torch.sum(delta ** 2, dim=2)
            weight = torch.where(
                distance_squared < cutoff_squared,
                torch.exp(-distance_squared / two_sigma_squared),
                torch.zeros_like(distance_squared),
            )
            summed[start:stop] = torch.sum(weight, dim=1)
        out[channel] = (summed * normalisation).view(*grid)
    return out
