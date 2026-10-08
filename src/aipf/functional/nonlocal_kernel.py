"""Nonlocal-kernel functional, the trained one: local free energy plus a nonlocal pair kernel.
``F = int [ f_loc(rho, kBT) + (1/2) rho^T (W * rho) ] dr``, assembled from ``FLocal``, ``PairKernel``
and ``Mobility``. ``kB`` is required: ``FLocal`` takes ``kB * T``, ``Mobility`` takes the literal ``T``.
Construction order (FLocal, kernel, mobility, T-basis heads last) fixes the RNG stream.
``kernel_argument``: ``"density"`` convolves ``W`` with ``rho``; ``"difference"`` with ``rho - rho_ref``,
so the pair term is ``(1/2) (rho - rho_ref)^T (W * (rho - rho_ref))`` (the drift is the same).
A two-axis ``grid`` builds the two-dimensional functional: :class:`aipf.spectral.SpectralOps2D`, ``(B, 2)``
boxes and the Hankel transform of the same radial ``W`` (``AnalyticRadialTransform(dim=2)``)."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .base import MODEL_REGISTRY
from .kernels import (
    KERNEL_EVALUATORS,
    AnalyticRadialTransform,
    LatticeSumTransform,
    PairKernel,
    QuinticEnvelope,
    RadialKernelSet,
    WHatEvaluator,
)
from .local_forms import FLocal
from aipf.mobility import Mobility
from aipf.spectral import (CHANNEL_LAST, CHANNEL_SECOND, DC, FLUX_EINSUM,
                           MATRIX_SECOND, OpsCache, k_squared)

__all__ = ["NonlocalKernel", "KERNEL_ARGUMENTS"]

#: What the pair kernel convolves.
KERNEL_ARGUMENTS: Tuple[str, ...] = ("density", "difference")


class NonlocalKernel(nn.Module):
    """The nonlocal-kernel functional: ``FLocal`` + ``PairKernel`` + ``Mobility``.
    ``kB`` is the Boltzmann constant in the unit of ``kBT_ref`` (``8.617333262e-5`` eV/K, or ``1.0`` reduced).
    ``kernel_evaluator=None`` builds :class:`AnalyticRadialTransform`; other keywords go to a submodule."""

    def __init__(
        self,
        grid: Tuple[int, int, int],
        n_species: int,
        kB: float,
        rho_ref: Sequence[float],
        h_g: int,
        R_cut: float,
        kernel_n_quad: Optional[int] = None,
        kernel_n_k_table: Optional[int] = None,
        kernel_k_table_max: Optional[float] = None,
        mobility_prefactor: str = "mole_fraction",
        mobility_shape: str = "mlp_rho",
        mobility_t_form: str = "none",
        *,
        # Required: "gas" or "lattice" (local_forms.IDEAL_FORMS).
        ideal_form: str,
        # Required only by the forms that read them; FLocal raises when one is needed and None.
        kBT_ref: Optional[float] = None,
        h_u: Optional[int] = None,
        # Required: grad and div zero the Nyquist wavenumber or not (spectral.NYQUIST_MASKS).
        nyquist_mask: bool,
        f_exc_form: str = "split",
        u_form: str = "mlp",
        u_degree: int = 4,
        u_parity: str = "full",
        u_variable: str = "ratio",
        gauge_fix: bool = False,
        g_exc_form: str = "icnn",
        g_symmetry: str = "none",
        icnn_output_bias: bool = True,
        kernel_argument: str = "density",
        enable_TlnT: bool = False,
        enable_T2: bool = False,
        tbasis_ortho_window: Optional[Sequence[float]] = None,
        tbasis_ortho_points: Optional[int] = None,
        h_g_hat: Optional[int] = None,
        h_g_tilde: Optional[int] = None,
        h_joint: Optional[int] = None,
        joint_depth: Optional[int] = None,
        local_input_scale: bool = False,
        local_activation: str = "gelu",
        rho_eps: float = 1e-6,
        kernel_hidden: int = 16,
        kernel_activation: str = "gelu",
        kernel_tail_sigma: Optional[float] = None,
        kernel_evaluator: Optional[Union[WHatEvaluator, str]] = None,
        mobility_shape_init: Optional[torch.Tensor] = None,
        mobility_hidden: int = 32,
        mobility_activation: str = "gelu",
        mobility_t_ref: Optional[float] = None,
        mobility_activation_energy_init: Optional[Sequence[float]] = None,
        mobility_input_ref: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {n_species}")
        if kernel_argument not in KERNEL_ARGUMENTS:
            raise ValueError(f"kernel_argument={kernel_argument!r} not in {KERNEL_ARGUMENTS}")
        if isinstance(kernel_evaluator, str) and kernel_evaluator not in KERNEL_EVALUATORS:
            raise ValueError(f"kernel_evaluator={kernel_evaluator!r} not in {KERNEL_EVALUATORS}")
        if kernel_evaluator is None and (kernel_n_k_table is None
                                          or kernel_k_table_max is None
                                          or kernel_n_quad is None):
            raise ValueError(
                "kernel_n_quad, kernel_n_k_table and kernel_k_table_max are "
                "required to build the default AnalyticRadialTransform "
                "evaluator; supply all three, or name another evaluator")
        if f_exc_form == "joint" and joint_depth is None:
            raise ValueError(
                "joint_depth is required when f_exc_form='joint': the joint "
                "net's number of hidden layers is part of the model, and a "
                "checkpoint of another depth does not load into it")

        if len(grid) == 2:
            if kernel_evaluator == "lattice_sum":
                raise NotImplementedError(
                    "kernel_evaluator='lattice_sum' is three-dimensional only; a two-dimensional "
                    "nonlocal_kernel takes the Hankel transform (kernel_evaluator=None)")
            if kernel_evaluator is not None and getattr(kernel_evaluator, "dim", 3) != 2:
                raise ValueError(
                    f"grid {tuple(grid)} is two-dimensional and the kernel_evaluator "
                    f"{type(kernel_evaluator).__name__} does not declare dim=2; a three-"
                    f"dimensional transform of W would be read as a two-dimensional one")
            Gx, Gy = grid
            self.grid = (int(Gx), int(Gy))
        else:
            if kernel_evaluator is not None and getattr(kernel_evaluator, "dim", 3) != 3:
                raise ValueError(
                    f"grid {tuple(grid)} is three-dimensional and the kernel_evaluator "
                    f"{type(kernel_evaluator).__name__} declares dim={kernel_evaluator.dim!r}; a "
                    f"two-dimensional transform of W would be read as a three-dimensional one")
            Gx, Gy, Gz = grid
            self.grid = (int(Gx), int(Gy), int(Gz))
        self.n_species = int(n_species)
        self.kB = float(kB)

        self._cache = OpsCache(self.grid, self.n_species,
                               nyquist_mask=nyquist_mask)
        self.ops = self._cache.ops

        # --- build order is load-bearing (RNG stream): 1. FLocal, T-heads deferred
        self.f_local = FLocal(
            self.n_species, rho_ref, kBT_ref, h_u, h_g,
            f_exc_form=f_exc_form, u_form=u_form, u_degree=u_degree,
            u_parity=u_parity, u_variable=u_variable, gauge_fix=gauge_fix,
            g_exc_form=g_exc_form, g_symmetry=g_symmetry,
            icnn_output_bias=icnn_output_bias,
            enable_TlnT=enable_TlnT, enable_T2=enable_T2,
            tbasis_ortho_window=tbasis_ortho_window,
            tbasis_ortho_points=tbasis_ortho_points,
            h_g_hat=h_g_hat, h_g_tilde=h_g_tilde, h_joint=h_joint,
            **({} if joint_depth is None else {"joint_depth": joint_depth}),
            activation=local_activation, rho_eps=rho_eps,
            ideal_form=ideal_form,
            defer_T_heads=True,
            input_scale=local_input_scale,
        )

        # --- 2. the pair kernel
        envelope = QuinticEnvelope(R_cut, kernel_tail_sigma)
        radial_set = RadialKernelSet(self.n_species, envelope,
                                      hidden=kernel_hidden,
                                      activation=kernel_activation)
        if kernel_evaluator == "lattice_sum":
            evaluator: WHatEvaluator = LatticeSumTransform()
        elif kernel_evaluator is not None:
            evaluator = kernel_evaluator
        else:
            evaluator = AnalyticRadialTransform(
                R_cut, kernel_n_quad, kernel_n_k_table, kernel_k_table_max,
                **({"dim": 2} if len(self.grid) == 2 else {}))
        self.kernel = PairKernel(radial_set, evaluator, n_quad=kernel_n_quad)
        self.kernel_argument = kernel_argument
        # Non-persistent: what the kernel convolves is declared, never read from a state dict.
        self.register_buffer(
            "kernel_centre",
            torch.as_tensor(rho_ref, dtype=torch.float32)
            if kernel_argument == "difference" else None,
            persistent=False)

        # --- 3. mobility (`_mobility`: `mobility` is the protocol method)
        self._mobility = Mobility(
            self.n_species, mobility_prefactor, mobility_shape,
            mobility_t_form, shape_init=mobility_shape_init,
            hidden=mobility_hidden, activation=mobility_activation,
            t_ref=mobility_t_ref,
            activation_energy_init=mobility_activation_energy_init,
            kB=self.kB, input_ref=mobility_input_ref,
            **({"rho_eps": rho_eps} if mobility_shape == "lattice_scalar" else {}),
        )

        # --- 4. last, the deferred T-basis heads
        self.f_local.build_T_heads()

    def _ops_for_real(self, rho: torch.Tensor):
        """The ops for a real-space tensor's own grid, looked up by the full grid."""
        return self._cache.ops_for_grid(rho.shape[-self._cache.ndim:], rho.device)

    def _flatten_channel_last(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, n_species, Gx, Gy, Gz)`` -> ``(B*Gx*Gy*Gz, n_species)``, permute before reshape."""
        return x.permute(*CHANNEL_LAST[self._cache.ndim]).reshape(-1, self.n_species)

    def _unflatten_channel_second(self, flat: torch.Tensor, batch: int,
                                   grid: Tuple[int, int, int]) -> torch.Tensor:
        """Inverse of :meth:`_flatten_channel_last` on the caller's ``grid``."""
        axes = tuple(int(g) for g in grid)
        n = flat.shape[-1]
        return flat.view(batch, *axes, n).permute(*CHANNEL_SECOND[self._cache.ndim])

    def _broadcast_scalar(self, value: torch.Tensor, batch: int,
                           grid: Tuple[int, int, int]) -> torch.Tensor:
        """``(B,)`` -> ``(B*Gx*Gy*Gz,)``, one copy per grid point."""
        return value.view(batch, *(1,) * len(grid)).expand(batch, *grid).reshape(-1)

    def bulk_free_energy_density(self, rho: torch.Tensor,
                                  T: torch.Tensor) -> torch.Tensor:
        """``f_loc(rho, kBT)``, pointwise, real space; no kernel term."""
        B = rho.shape[0]
        grid = rho.shape[-self._cache.ndim:]
        kBT = self.kB * T
        rho_flat = self._flatten_channel_last(rho)
        kBT_flat = self._broadcast_scalar(kBT, B, grid)
        f_flat = self.f_local.f_pointwise(rho_flat, kBT_flat)   # (-1,)
        return f_flat.view(B, 1, *grid)

    def mobility(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """``M(rho, T)``, ``(B, n_species, n_species, Gx, Gy, Gz)``, from the literal ``T``."""
        B = rho.shape[0]
        grid = rho.shape[-self._cache.ndim:]
        rho_flat = self._flatten_channel_last(rho)
        T_flat = self._broadcast_scalar(T, B, grid)
        M_flat = self._mobility(rho_flat, T_flat)                # (-1, n, n)
        n = self.n_species
        M = M_flat.view(B, *grid, n, n).permute(*MATRIX_SECOND[self._cache.ndim])
        return M

    def _mu_hat(self, rho: torch.Tensor, boxes: torch.Tensor,
                T: torch.Tensor,
                rho_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``mu_hat`` in k-space at mode scale; the kernel term uses the caller's ``rho_hat`` when given."""
        ops = self._ops_for_real(rho)
        ndim = self._cache.ndim
        N = math.prod(ops.grid)
        B = rho.shape[0]
        grid = ops.grid
        kBT = self.kB * T

        # pointwise mu, on the flattened (-1, n_species) view.
        rho_flat = self._flatten_channel_last(rho)
        kBT_flat = self._broadcast_scalar(kBT, B, grid)
        mu_flat = self.f_local.mu_pointwise(rho_flat, kBT_flat)
        mu = self._unflatten_channel_second(mu_flat, B, grid)

        # the kernel term, in k-space: mu_hat += einsum(W_hat(|k|), rho_hat).
        kmag = torch.sqrt(k_squared(ops.k_axes(boxes)))[:, 0]     # (B,Gx,Gy,Gzr)
        Wm = self.kernel.w_hat(kmag, grid=grid, boxes=boxes)      # (B,Gx,Gy,Gzr,n,n)

        mu_hat = ops.rfft(mu) / N
        if rho_hat is None:
            rho_hat = ops.rfft(self.kernel_offset(rho)) / N       # same "mode" scale as mu_hat
        elif self.kernel_centre is not None:
            rho_hat = rho_hat.clone()                             # the centre sits in the k = 0 mode
            rho_hat[DC[ndim]] -= self.kernel_centre.to(rho_hat.real.dtype).view(1, -1)
        rho_hat_last = rho_hat.permute(*CHANNEL_LAST[ndim])       # (B,Gx,Gy,Gzr,n)
        kernel_term = torch.einsum("b...ij,b...j->b...i",
                                    Wm.to(rho_hat_last.dtype), rho_hat_last)
        return mu_hat + kernel_term.permute(*CHANNEL_SECOND[ndim])  # (B,n,Gx,Gy,Gzr)

    def kernel_offset(self, rho: torch.Tensor) -> torch.Tensor:
        """What the kernel convolves: ``rho`` (channel axis 1), minus ``rho_ref`` under ``"difference"``."""
        if self.kernel_centre is None:
            return rho
        c = self.kernel_centre.to(dtype=rho.dtype, device=rho.device)
        return rho - c.view(1, -1, *([1] * (rho.dim() - 2)))

    def uniform_free_energy(self, rho: torch.Tensor, T: torch.Tensor,
                            w0: torch.Tensor) -> torch.Tensor:
        """``f_loc(rho) + (1/2) d^T w0 d`` per uniform state, ``d`` = :meth:`kernel_offset`; ``rho`` ``(P, n)``.
        ``w0`` ``(n, n)`` is the caller's ``Ŵ(0)`` (e.g. :meth:`PairKernel.w_hat_zero_radial`)."""
        f = self.f_local.f_pointwise(rho, self.kB * T)
        d = self.kernel_offset(rho)
        return f + 0.5 * torch.einsum("pi,ij,pj->p", d, w0.to(rho.dtype), d)

    def _chemical_potential_is_overridden(self) -> bool:
        """True when ``chemical_potential`` is replaced on the instance, a subclass or the class.
        Compares against ``_BASE_CHEMICAL_POTENTIAL`` captured at import."""
        return ("chemical_potential" in self.__dict__
                or type(self).chemical_potential
                is not _BASE_CHEMICAL_POTENTIAL)

    def chemical_potential(self, rho: torch.Tensor, boxes: torch.Tensor,
                            T: torch.Tensor) -> torch.Tensor:
        """``mu = d(f_loc)/d(rho) + W_hat(|k|) @ rho_hat``, real space in and out."""
        ops = self._ops_for_real(rho)
        N = math.prod(ops.grid)
        return ops.irfft(self._mu_hat(rho, boxes, T) * N)

    def forward(self, rho_hat: torch.Tensor, boxes: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        """The Model B right-hand side ``+div(M grad mu)``, k-space in and out."""
        ops = self._cache.ops_for(rho_hat.shape, rho_hat.device)
        ndim = self._cache.ndim
        N = math.prod(ops.grid)
        rho = ops.irfft(rho_hat * N)
        M = self.mobility(rho, T)                                 # (B,n,n,Gx,Gy,Gz)

        ks = ops.k_axes(boxes)
        # k-space fast path unless `chemical_potential` has been overridden.
        if self._chemical_potential_is_overridden():
            mu_hat = ops.rfft(self.chemical_potential(rho, boxes, T)) / N
        else:
            mu_hat = self._mu_hat(rho, boxes, T, rho_hat=rho_hat)
        grads = ops.grad_hat(mu_hat * N, *ks)
        grad_mu = torch.stack(
            [ops.irfft(g) for g in grads], dim=2)                 # (B,n,3,Gx,Gy,Gz)

        # d(rho)/dt = -div(J), J = -M grad(mu): hence +div(M grad(mu)).
        J = [torch.einsum(FLUX_EINSUM[ndim], M, grad_mu[:, :, d])
             for d in range(ndim)]
        J_hat = [ops.rfft(j) / N for j in J]
        return ops.div_hat(*J_hat, *ks)


#: The unoverridden real-space seam, captured at import.
_BASE_CHEMICAL_POTENTIAL = NonlocalKernel.chemical_potential

#: Registered by name.
MODEL_REGISTRY.register("nonlocal_kernel", NonlocalKernel)
