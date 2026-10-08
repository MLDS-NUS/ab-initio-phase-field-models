"""Rung 3's pair-kernel term: radial functions ``W_ab(r)`` and the route to ``Ŵ(|k|)``.
``R_cut`` and the quadrature and k-table sizes are System fields, never defaulted here.
An evaluator with ``reads_geometry = True`` is handed the real-space grid and the boxes as well as ``|k|``.
In two dimensions the radial route is the Hankel transform (:func:`hankel_transform`); the lattice sum,
``Ŵ(0)`` and ``kappa_eff`` are three-dimensional only."""
from __future__ import annotations

import math
from typing import Dict, Optional, Protocol, Tuple

import torch
import torch.nn as nn

FOUR_PI = 4.0 * math.pi
TWO_PI = 2.0 * math.pi

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
}


def _pair_list(n_species: int) -> Tuple[Tuple[int, int], ...]:
    """The ``n(n+1)/2`` unordered species-pair indices ``(i, j)``, ``i <= j``, in a fixed order."""
    if n_species < 1:
        raise ValueError(f"n_species must be >= 1, got {n_species}")
    return tuple((i, j) for i in range(n_species) for j in range(i, n_species))


def _assemble_symmetric(values: torch.Tensor, pairs: Tuple[Tuple[int, int], ...],
                         n_species: int) -> torch.Tensor:
    """Stack ``(n_pairs, ...)`` per-pair values into a symmetric ``(..., n, n)`` matrix."""
    index = {p: idx for idx, p in enumerate(pairs)}
    rows = []
    for i in range(n_species):
        row = [values[index[(i, j)] if i <= j else index[(j, i)]]
               for j in range(n_species)]
        rows.append(torch.stack(row, dim=-1))
    return torch.stack(rows, dim=-2)


def radial_fourier_transform(w_vals: torch.Tensor, r: torch.Tensor,
                              k: torch.Tensor) -> torch.Tensor:
    """``Ŵ(k) = 4*pi*int r^2 W(r) sinc(kr) dr``, by trapezoidal quadrature.
    ``w_vals`` ``(..., Q)`` at points ``r`` ``(Q,)``, ``k`` any shape -> ``(*w_vals.shape[:-1], *k.shape)``.
    ``torch.sinc`` is normalised, hence ``k*r/pi``."""
    kr = k.reshape(-1, 1) * r                                   # (n_k, Q)
    sinc = torch.sinc(kr / torch.pi)
    integrand = w_vals.unsqueeze(-2) * (r * r) * sinc           # (..., n_k, Q)
    out = FOUR_PI * torch.trapezoid(integrand, r, dim=-1)       # (..., n_k)
    return out.reshape(*w_vals.shape[:-1], *k.shape)


def hankel_transform(w_vals: torch.Tensor, r: torch.Tensor,
                     k: torch.Tensor) -> torch.Tensor:
    """``Ŵ(k) = 2*pi*int r W(r) J0(kr) dr``, the two-dimensional transform of a radial ``W``, by trapezoidal
    quadrature; shapes as :func:`radial_fourier_transform`."""
    kr = k.reshape(-1, 1) * r                                   # (n_k, Q)
    j0 = torch.special.bessel_j0(kr)
    integrand = w_vals.unsqueeze(-2) * r * j0                   # (..., n_k, Q)
    out = TWO_PI * torch.trapezoid(integrand, r, dim=-1)        # (..., n_k)
    return out.reshape(*w_vals.shape[:-1], *k.shape)


#: The admissible ``dim`` of :class:`AnalyticRadialTransform`: the 3D sine transform or the 2D Hankel one.
RADIAL_TRANSFORM_DIMS = (3, 2)


class QuinticEnvelope:
    """``f_c(u) = 1 - 10*u^3 + 15*u^4 - 6*u^5``, ``u = r/R_cut`` clamped at 1 (an exact hard cutoff).
    Optional Gaussian tail ``exp(-r^2 / (4*sigma^2))``; ``tail_sigma=None`` is off."""

    def __init__(self, R_cut: float, tail_sigma: Optional[float] = None):
        self.R_cut = float(R_cut)
        self.tail_sigma = float(tail_sigma) if tail_sigma is not None else None

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        u = (r / self.R_cut).clamp(max=1.0)
        env = 1 - 10 * u ** 3 + 15 * u ** 4 - 6 * u ** 5
        if self.tail_sigma is not None:
            env = env * torch.exp(-r * r / (4.0 * self.tail_sigma ** 2))
        return env


class RadialKernelSet(nn.Module):
    """The trainable ``W_ab(r)``: one radial MLP per unordered species pair, times ``envelope``."""

    def __init__(self, n_species: int, envelope: QuinticEnvelope,
                 hidden: int = 16, activation: str = "gelu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; have "
                f"{sorted(_ACTIVATIONS)}")
        self.n_species = int(n_species)
        self.pairs = _pair_list(self.n_species)
        self.envelope = envelope
        act = _ACTIVATIONS[activation]
        self.nets = nn.ModuleList(
            nn.Sequential(nn.Linear(1, hidden), act(),
                          nn.Linear(hidden, hidden), act(),
                          nn.Linear(hidden, 1))
            for _ in self.pairs)

    def w_of_r(self, r: torch.Tensor) -> torch.Tensor:
        """``(n_pairs, *r.shape)`` radial values at ``r``, envelope applied."""
        env = self.envelope(r)
        return torch.stack([net(r.unsqueeze(-1)).squeeze(-1)
                            for net in self.nets]) * env

    def matrix_of_r(self, r: torch.Tensor) -> torch.Tensor:
        """``(*r.shape, n_species, n_species)`` symmetric ``W(r)`` matrix."""
        return _assemble_symmetric(self.w_of_r(r), self.pairs, self.n_species)


class WHatEvaluator(Protocol):
    """The structural seam from a :class:`RadialKernelSet` to ``Ŵ(k)``."""

    def w_hat(self, radial_set: RadialKernelSet,
              k: torch.Tensor) -> torch.Tensor:
        """``(*k.shape, radial_set.n_species, radial_set.n_species)``."""
        ...


class AnalyticRadialTransform(nn.Module):
    """``Ŵ(k) = 4*pi*int r^2 W(r) sinc(kr) dr`` on a fixed 1-D k table, linearly interpolated to ``|k|``.
    Reads no grid and no box. ``R_cut`` must match the radial set's. ``dim=2`` takes the Hankel transform
    ``2*pi*int r W(r) J0(kr) dr`` instead, on the same buffers (``dim`` is an attribute, not state)."""

    def __init__(self, R_cut: float, n_quad: int, n_k_table: int,
                 k_table_max: float, *, dim: int = 3):
        super().__init__()
        if dim not in RADIAL_TRANSFORM_DIMS:
            raise ValueError(
                f"dim={dim!r} is not one of {RADIAL_TRANSFORM_DIMS}: 3 is the sine transform "
                f"of a radial W in three dimensions, 2 the Hankel transform in two")
        self.dim = int(dim)
        self.k_table_max = float(k_table_max)
        self.register_buffer(
            "r_quad", torch.linspace(0.0, float(R_cut), int(n_quad)))
        self.register_buffer(
            "k_table", torch.linspace(0.0, self.k_table_max, int(n_k_table)))

    def w_hat(self, radial_set: RadialKernelSet,
              k: torch.Tensor) -> torch.Tensor:
        w_vals = radial_set.w_of_r(self.r_quad)                       # (n_pairs, Q)
        if self.dim == 2:
            table = hankel_transform(w_vals, self.r_quad, self.k_table)
        else:
            table = radial_fourier_transform(w_vals, self.r_quad, self.k_table)  # (n_pairs, K)

        K = self.k_table.shape[0]
        x = (k.abs().clamp(max=self.k_table_max) / self.k_table_max) * (K - 1)
        i0 = x.floor().long().clamp(max=K - 2)
        frac = x - i0.to(x.dtype)
        flat0, flat1 = i0.reshape(-1), (i0 + 1).reshape(-1)
        n_pairs = table.shape[0]
        t0 = table[:, flat0].reshape(n_pairs, *k.shape)
        t1 = table[:, flat1].reshape(n_pairs, *k.shape)
        w_pairs = t0 + frac.unsqueeze(0) * (t1 - t0)                  # (n_pairs, *k.shape)

        return _assemble_symmetric(w_pairs, radial_set.pairs, radial_set.n_species)


class LatticeSumTransform(nn.Module):
    """``Ŵ = dV * rfftn(W(r_ij))`` with ``r_ij`` the periodic nearest-image distance between grid nodes.
    Spacing ``(boxes / grid)`` is taken in float64 and rounded to the radial nets' dtype; the transform
    therefore depends on the grid and box, never only on ``|k|``."""

    reads_geometry = True

    def __init__(self):
        super().__init__()
        self._r_cache: Dict[tuple, torch.Tensor] = {}

    def distances(self, grid: Tuple[int, int, int], spacing: Tuple[float, float, float],
                  device, dtype) -> torch.Tensor:
        """``(Gx, Gy, Gz)`` nearest-image distances, ``sqrt(sum_a (min(n_a, G_a - n_a) d_a)^2)``."""
        key = (tuple(grid), tuple(spacing), str(device), dtype)
        cached = self._r_cache.get(key)
        if cached is not None:
            return cached
        D, H, W = (int(g) for g in grid)
        dx, dy, dz = spacing
        zz = torch.arange(D, device=device, dtype=dtype)
        yy = torch.arange(H, device=device, dtype=dtype)
        xx = torch.arange(W, device=device, dtype=dtype)
        ZZ, YY, XX = torch.meshgrid(zz, yy, xx, indexing="ij")
        ZZ = torch.minimum(ZZ, D - ZZ) * dx
        YY = torch.minimum(YY, H - YY) * dy
        XX = torch.minimum(XX, W - XX) * dz
        r = torch.sqrt(ZZ ** 2 + YY ** 2 + XX ** 2)
        self._r_cache[key] = r
        return r

    def spacing(self, grid, box: torch.Tensor, dtype) -> Tuple[float, float, float]:
        g = torch.tensor([float(v) for v in grid], dtype=torch.float64, device=box.device)
        return tuple((box.to(torch.float64) / g).to(dtype).tolist())

    def w_grid(self, radial_set: RadialKernelSet, grid, spacing) -> torch.Tensor:
        """``(n_pairs, Gx, Gy, Gz)`` real-space ``W`` at the nodes' nearest-image distances."""
        p = next(radial_set.parameters())
        r = self.distances(grid, spacing, p.device, p.dtype)
        return radial_set.w_of_r(r)

    def w_hat(self, radial_set: RadialKernelSet, k: torch.Tensor, *,
              grid=None, boxes=None) -> torch.Tensor:
        """``(B, Gx, Gy, Gzr, n, n)`` complex; ``k`` only fixes the batch and is not read."""
        if grid is None or boxes is None:
            raise NotImplementedError(
                "kernel_evaluator='lattice_sum' was asked for W_hat at a bare "
                "|k| (no grid, no boxes), as a W_hat(0) read-off does: a lattice "
                "sum over grid nodes has no value there, so this combination is "
                "not supported. Pass grid= and boxes=, or use "
                "PairKernel.w_hat_zero_radial for the radial W_hat(0)")
        if len(grid) == 2:
            raise NotImplementedError(
                "kernel_evaluator='lattice_sum' is three-dimensional only: it sums W over the "
                "nodes of a (Gx, Gy, Gz) grid. A two-dimensional model takes the Hankel transform "
                "(AnalyticRadialTransform(dim=2))")
        p = next(radial_set.parameters())
        out = []
        for b in range(boxes.shape[0]):
            sp = self.spacing(grid, boxes[b], p.dtype)
            W = self.w_grid(radial_set, grid, sp)
            dV = sp[0] * sp[1] * sp[2]
            out.append(_assemble_symmetric(dV * torch.fft.rfftn(W, dim=(-3, -2, -1)),
                                           radial_set.pairs, radial_set.n_species))
        return torch.stack(out)


#: Admissible names for an evaluator a rung builds itself (beside ``None``, the k table).
KERNEL_EVALUATORS: Tuple[str, ...] = ("lattice_sum",)


class PairKernel(nn.Module):
    """Rung 3's pair-kernel term: radial functions plus a required ``Ŵ(k)`` evaluator.
    :meth:`w_hat` is ``(*k.shape, n, n)``; :meth:`kappa_eff` is a diagnostic, never trained."""

    def __init__(self, radial_set: RadialKernelSet, evaluator: WHatEvaluator,
                 n_quad: Optional[int]):
        super().__init__()
        self.radial_set = radial_set
        self.evaluator = evaluator
        # None: no quadrature is declared, and kappa_eff refuses rather than pick one.
        self.register_buffer(
            "kappa_r_quad",
            None if n_quad is None
            else torch.linspace(0.0, radial_set.envelope.R_cut, int(n_quad)))

    @property
    def n_species(self) -> int:
        return self.radial_set.n_species

    def _refuse_two_dimensions(self, what: str) -> None:
        if getattr(self.evaluator, "dim", 3) == 2:
            raise NotImplementedError(
                f"{what} is the three-dimensional moment of W(r); this kernel's evaluator is "
                f"two-dimensional (the Hankel transform), whose moments are other integrals and "
                f"are not provided")

    def w_hat(self, k: torch.Tensor, *, grid=None, boxes=None) -> torch.Tensor:
        """The evaluator's ``Ŵ``; the grid and boxes reach it only when it ``reads_geometry``."""
        if getattr(self.evaluator, "reads_geometry", False):
            return self.evaluator.w_hat(self.radial_set, k, grid=grid, boxes=boxes)
        return self.evaluator.w_hat(self.radial_set, k)

    @torch.no_grad()
    def w_hat_zero_radial(self, r_max: float, n_points: int) -> torch.Tensor:
        """``(n, n)`` ``Ŵ(0) = 4*pi*int_0^r_max r^2 W(r) dr``, trapezoid on ``linspace(0, r_max, n_points)``
        in the radial nets' dtype."""
        return self.w_hat_zero_quadrature(r_max, n_points)

    def w_hat_zero_quadrature(self, r_max: float, n_points: int) -> torch.Tensor:
        """:meth:`w_hat_zero_radial` with its graph kept, for a loss that trains through it."""
        self._refuse_two_dimensions("W_hat(0) = 4*pi*int r^2 W(r) dr")
        p = next(self.radial_set.parameters())
        r = torch.linspace(0.0, float(r_max), int(n_points), device=p.device, dtype=p.dtype)
        w = self.radial_set.w_of_r(r)
        vals = torch.stack([4.0 * torch.pi * torch.trapz(r ** 2 * w[i], r)
                            for i in range(w.shape[0])])
        return _assemble_symmetric(vals, self.radial_set.pairs, self.radial_set.n_species)

    @torch.no_grad()
    def kappa_eff(self) -> torch.Tensor:
        """``-(2*pi/3) * int r^4 W(r) dr``, so ``Ŵ(k) = Ŵ(0) + kappa_eff*k^2 + O(k^4)``."""
        self._refuse_two_dimensions("kappa_eff = -(2*pi/3) int r^4 W(r) dr")
        r = self.kappa_r_quad
        if r is None:
            raise ValueError(
                "kappa_eff integrates on the declared kernel_n_quad points, and "
                "this kernel was built without them (its evaluator needs no "
                "quadrature); declare kernel_n_quad to read kappa_eff")
        w = self.radial_set.w_of_r(r)                                # (n_pairs, Q)
        kap = -(2.0 * torch.pi / 3.0) * torch.trapezoid(r ** 4 * w, r, dim=-1)
        return _assemble_symmetric(kap, self.radial_set.pairs, self.radial_set.n_species)
