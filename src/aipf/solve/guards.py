"""Rollout-time guards, and ``kappa_roll``: a numerical stabilisation dial, never trained and not
part of any model's free energy (unlike rung 2's ``kappa`` and the diagnostic ``kappa_eff``)."""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

from aipf.spectral import k_squared, ops_ndim


def assert_finite(x: torch.Tensor, where: str) -> None:
    """Raise ``FloatingPointError``, naming ``where``, on the first non-finite value."""
    finite = torch.isfinite(x.real) if x.is_complex() else torch.isfinite(x)
    if not bool(finite.all()):
        bad = int((~finite).sum())
        raise FloatingPointError(
            f"{where}: {bad} non-finite value(s) (NaN or Inf). A rollout "
            f"that reaches here has already left the physical state and "
            f"every step after this one is meaningless.")
    if x.is_complex():
        finite_im = torch.isfinite(x.imag)
        if not bool(finite_im.all()):
            bad = int((~finite_im).sum())
            raise FloatingPointError(
                f"{where}: {bad} non-finite imaginary value(s) (NaN or Inf).")


@contextmanager
def clamped_inputs(model: torch.nn.Module, floor: float) -> Iterator[torch.nn.Module]:
    """Clamp every ``rho`` seen by ``chemical_potential`` and ``mobility`` at ``floor``; restore on exit.
    An input guard: the state is untouched. ``floor = 0`` still clamps; there is no off value."""
    mu0 = model.chemical_potential
    mob0 = model.mobility
    # Restore each name to an instance attribute only if it was one before.
    had_mu = "chemical_potential" in model.__dict__
    had_mob = "mobility" in model.__dict__

    def _mu(rho, boxes, T, _mu0=mu0, _floor=floor):
        return _mu0(rho.clamp(min=_floor), boxes, T)

    def _mob(rho, T, _mob0=mob0, _floor=floor):
        return _mob0(rho.clamp(min=_floor), T)

    model.chemical_potential = _mu
    model.mobility = _mob
    try:
        yield model
    finally:
        if had_mu:
            model.chemical_potential = mu0
        else:
            del model.chemical_potential
        if had_mob:
            model.mobility = mob0
        else:
            del model.mobility


def kappa_roll_correction(model: torch.nn.Module, rho: torch.Tensor,
                          rho_hat: torch.Tensor, T: torch.Tensor,
                          ops, kx: torch.Tensor, ky: torch.Tensor,
                          kz: Optional[torch.Tensor], kappa_roll: float) -> torch.Tensor:
    """The extra flux ``div(M grad(-kappa_roll * lap(rho)))`` in k-space, through ``model.mobility`` only.
    Zero when ``kappa_roll`` is falsy. ``kz`` is ``None`` for a two-dimensional ``ops``."""
    if not kappa_roll:
        return torch.zeros_like(rho_hat)
    ks = (kx, ky, kz)[:ops_ndim(ops)]
    N = math.prod(ops.grid)
    k2 = k_squared(ks)
    extra_mu_hat = float(kappa_roll) * k2 * rho_hat
    grads = ops.grad_hat(extra_mu_hat * N, *ks)
    grad_extra_mu = torch.stack(
        [ops.irfft(g) for g in grads], dim=2)                  # (B,n,ndim,...)
    M = model.mobility(rho, T)                                 # (B,n,n,...)
    J = [torch.einsum("bij...,bj...->bi...", M, grad_extra_mu[:, :, d])
         for d in range(len(ks))]
    J_hat = [ops.rfft(j) / N for j in J]
    return ops.div_hat(*J_hat, *ks)
