"""Integrators, deterministic and SDE, generic over any ``FreeEnergyModel`` (protocol methods only).
Mass is exact: every step is a linear combination of k-space divergences, zero at ``k=0``.
External drive (off by default, bit-for-bit absent when off): ``v_ext`` adds ``V_i(r)`` to ``mu_i``;
``T_field`` adds ``psi_i(r) = mu^loc_i(rho(r), T(r)) - mu^loc_i(rho(r), T)``, both as ``div[M grad psi]``.
Under noise ``T_field`` also scales the amplitude per cell (local-equilibrium convention).
Two dimensions are declared by the model's operator set (``model.ops.ndim``), never read off a shape; the
noise there needs the cell's extent along the averaged axis, ``depth`` (:func:`check_depth`)."""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch

from aipf.spectral import (CHANNEL_LAST, CHANNEL_SECOND, FLUX_EINSUM,
                           MATRIX_LAST, MATRIX_SECOND, SpectralOps,
                           half_spectrum_grid, make_ops, model_ndim, ops_ndim)
from .declare import UNDECLARED
from .guards import assert_finite, clamped_inputs, kappa_roll_correction
from .noise import build_noise_filter, check_depth, check_noise_declaration
from .projection import (check_state_projection, hermitianize,
                         project_state)
from .trust_domain import TrustDomain


def _ops_for(rho_hat: torch.Tensor, ops: Optional[SpectralOps],
             n_species: int, model) -> SpectralOps:
    """``ops``, or one on ``rho_hat``'s grid carrying the model's own declared ``nyquist_mask``."""
    if ops is not None:
        return ops
    declared = getattr(getattr(model, "ops", None), "nyquist_mask", None)
    if declared is None:
        raise ValueError(
            "no operator set was passed and the model has none (model.ops) "
            "declaring nyquist_mask; pass ops=, built with the model's declaration")
    if model_ndim(model) == 2:
        grid = half_spectrum_grid(rho_hat.shape[-2:], model.ops.grid)
        return make_ops(grid, n_species, nyquist_mask=declared).to(rho_hat.device)
    Gx, Gy, Gzr = (int(s) for s in rho_hat.shape[-3:])
    grid = (Gx, Gy, 2 * (Gzr - 1))
    return SpectralOps(grid, n_species, nyquist_mask=declared).to(rho_hat.device)


# ---------------------------------------------------------------------------
# the external drive
# ---------------------------------------------------------------------------

def _shape_text(shape) -> str:
    return "(" + ", ".join(str(v) for v in shape) + ")"


def _validated_T_field(T_field, rho: torch.Tensor, ndim: int = 3) -> torch.Tensor:
    """``T_field`` as a ``(B, *grid)`` tensor, finite and strictly positive; ``grid`` is ``rho``'s last
    ``ndim`` (declared) axes."""
    B = rho.shape[0]
    grid = tuple(int(g) for g in rho.shape[-ndim:])
    Tf = torch.as_tensor(T_field, dtype=rho.dtype, device=rho.device)
    if tuple(Tf.shape) == grid:
        Tf = Tf.unsqueeze(0).expand(B, *grid)
    if tuple(Tf.shape) != (B, *grid):
        raise ValueError(
            f"T_field must have shape {_shape_text((B, *grid))} or "
            f"{_shape_text(grid)} -- one "
            f"temperature per cell of the rollout's FULL real-space grid, "
            f"no species axis, in the same unit as the protocol's T -- got "
            f"{tuple(Tf.shape)}")
    if not bool(torch.isfinite(Tf).all()) or float(Tf.min()) <= 0.0:
        raise ValueError(
            f"T_field must be finite and strictly positive everywhere; got "
            f"min {float(Tf.min())}, max {float(Tf.max())}")
    return Tf


def _validated_v_ext(v_ext, rho: torch.Tensor, ndim: int = 3) -> torch.Tensor:
    """``v_ext`` as a ``(B, n_species, *grid)`` tensor, checked; ``grid`` is ``rho``'s last ``ndim`` axes."""
    B, n = rho.shape[0], rho.shape[1]
    grid = tuple(int(g) for g in rho.shape[-ndim:])
    V = torch.as_tensor(v_ext, dtype=rho.dtype, device=rho.device)
    if tuple(V.shape) == (n, *grid):
        V = V.unsqueeze(0).expand(B, n, *grid)
    if tuple(V.shape) != (B, n, *grid):
        raise ValueError(
            f"v_ext must have shape {_shape_text((B, n, *grid))} or "
            f"{_shape_text((n, *grid))} -- one "
            f"potential per species on the rollout's FULL real-space grid, "
            f"in the same energy unit as the model's mu -- got "
            f"{tuple(V.shape)}")
    if not bool(torch.isfinite(V).all()):
        raise ValueError("v_ext must be finite everywhere")
    return V


def _local_mu_shift(model: torch.nn.Module, rho: torch.Tensor,
                    T: torch.Tensor, T_field, ndim: int = 3) -> torch.Tensor:
    """``mu^loc(rho(r), T(r)) - mu^loc(rho(r), T)``, pointwise, from ``bulk_free_energy_density`` by autograd.
    Both temperatures in one call, so a uniform ``T_field == T`` gives exactly zero."""
    B, n = rho.shape[0], rho.shape[1]
    grid = tuple(int(g) for g in rho.shape[-ndim:])
    Tf = _validated_T_field(T_field, rho, ndim)
    P = B * math.prod(grid)
    ones = (1,) * ndim

    cells = rho.permute(*CHANNEL_LAST[ndim]).reshape(P, n)
    T_ref = T.to(rho.dtype).view(B, *ones).expand(B, *grid).reshape(P)
    T_two = torch.cat([Tf.reshape(P), T_ref], dim=0)
    rho_two = torch.cat([cells, cells], dim=0).view(2 * P, n, *ones)

    with torch.enable_grad():
        r = rho_two.detach().requires_grad_(True)
        f = model.bulk_free_energy_density(r, T_two)
        (g,) = torch.autograd.grad(f.sum(), r)
    g = g.reshape(2 * P, n)
    return (g[:P] - g[P:]).view(B, *grid, n).permute(*CHANNEL_SECOND[ndim])


def _external_potential(model: torch.nn.Module, rho: torch.Tensor,
                        T: torch.Tensor, v_ext, T_field, ndim: int = 3
                        ) -> Optional[torch.Tensor]:
    """The additive potential ``psi_i(r)`` on ``mu_i``, or ``None`` when neither drive is on."""
    psi = None
    if v_ext is not None:
        psi = _validated_v_ext(v_ext, rho, ndim)
    if T_field is not None:
        shift = _local_mu_shift(model, rho, T, T_field, ndim)
        psi = shift if psi is None else psi + shift
    return psi


def _external_flux_hat(model: torch.nn.Module, rho: torch.Tensor,
                       T: torch.Tensor, ops: SpectralOps, kx, ky, kz,
                       psi: torch.Tensor) -> torch.Tensor:
    """``div[M(rho) grad psi]`` in the state's spectral convention; zero at ``k=0``.
    ``kz`` is ``None`` for a two-dimensional ``ops``."""
    ndim = ops_ndim(ops)
    ks = (kx, ky, kz)[:ndim]
    N = math.prod(ops.grid)
    grads = ops.grad_hat(ops.rfft(psi), *ks)
    grad_psi = torch.stack([ops.irfft(g) for g in grads], dim=2)
    M = model.mobility(rho, T)
    J = [torch.einsum(FLUX_EINSUM[ndim], M, grad_psi[:, :, d])
         for d in range(ndim)]
    return ops.div_hat(*[ops.rfft(j) / N for j in J], *ks)


def _xyz(ks) -> tuple:
    """``ops.k_axes``'s wavevectors as ``(kx, ky, kz)``, ``kz`` ``None`` in two dimensions."""
    return tuple(ks) + (None,) * (3 - len(ks))


def _rhs(model: torch.nn.Module, rho_hat: torch.Tensor, boxes: torch.Tensor,
         T: torch.Tensor, ops: SpectralOps, kappa_roll: float,
         v_ext=None, T_field=None) -> torch.Tensor:
    """``model.forward`` plus ``kappa_roll``'s term and the external flux (one model call when off)."""
    F_hat = model.forward(rho_hat, boxes, T)
    if not kappa_roll and v_ext is None and T_field is None:
        return F_hat
    N = math.prod(ops.grid)
    kx, ky, kz = _xyz(ops.k_axes(boxes))
    rho = ops.irfft(rho_hat * N)
    if kappa_roll:
        F_hat = F_hat + kappa_roll_correction(
            model, rho, rho_hat, T, ops, kx, ky, kz, kappa_roll)
    psi = _external_potential(model, rho, T, v_ext, T_field, ops_ndim(ops))
    if psi is not None:
        F_hat = F_hat + _external_flux_hat(model, rho, T, ops, kx, ky, kz,
                                           psi)
    return F_hat


# ---------------------------------------------------------------------------
# deterministic integrators
# ---------------------------------------------------------------------------

def step_euler(model: torch.nn.Module, rho_hat: torch.Tensor,
               boxes: torch.Tensor, T: torch.Tensor, dt: float, *,
               ops: Optional[SpectralOps] = None,
               kappa_roll: float = 0.0,
               v_ext=None, T_field=None) -> torch.Tensor:
    """Forward Euler: ``rho_hat + dt * rhs(rho_hat)``."""
    n_species = rho_hat.shape[1]
    ops = _ops_for(rho_hat, ops, n_species, model)
    rhs = _rhs(model, rho_hat, boxes, T, ops, kappa_roll, v_ext, T_field)
    return rho_hat + dt * rhs


def step_heun(model: torch.nn.Module, rho_hat: torch.Tensor,
             boxes: torch.Tensor, T: torch.Tensor, dt: float, *,
             ops: Optional[SpectralOps] = None,
             kappa_roll: float = 0.0,
             v_ext=None, T_field=None) -> torch.Tensor:
    """Explicit trapezoidal (Heun / SSPRK2) predictor-corrector."""
    n_species = rho_hat.shape[1]
    ops = _ops_for(rho_hat, ops, n_species, model)
    k1 = _rhs(model, rho_hat, boxes, T, ops, kappa_roll, v_ext, T_field)
    predictor = rho_hat + dt * k1
    k2 = _rhs(model, predictor, boxes, T, ops, kappa_roll, v_ext, T_field)
    return rho_hat + 0.5 * dt * (k1 + k2)


_DETERMINISTIC_STEPS = {"euler": step_euler, "heun": step_heun}


def _validated_clamp_rho(clamp_rho) -> Optional[float]:
    """``clamp_rho`` as the floor it means, or ``None`` for no guard (no wrapper installed).
    ``True``, ``False`` and ``0`` are refused: each reads two ways."""
    if clamp_rho is UNDECLARED:
        raise ValueError(
            "clamp_rho must be declared: the floor applied to the density "
            "the model's own pointwise methods are EVALUATED at (not to "
            "the state -- see aipf.solve.guards.clamped_inputs), or None "
            "for no guard at all. There is no default, deliberately: "
            "False reads as no guard and the published rollouts passed "
            "1e-3. See aipf.solve.declare.")
    if clamp_rho is None:
        return None
    if clamp_rho is True:
        raise ValueError(
            "clamp_rho=True is not carried: it would mean 'the rung's "
            "own rho_eps', a rung-private "
            "attribute a protocol-only core cannot read, so the value "
            "would depend on which rung was loaded. Pass the floor "
            "itself, or None for no guard.")
    value = float(clamp_rho)
    if value != value or value == float("inf"):
        raise ValueError(f"clamp_rho must be finite, got {clamp_rho!r}")
    if clamp_rho is False or value == 0.0:
        raise ValueError(
            f"clamp_rho={clamp_rho!r} is refused because it reads two ways "
            f"and neither can be chosen for you. Read as a truth value "
            f"(`if clamp_rho:`, '0 disables') it means NO GUARD -- spell "
            f"that None. "
            f"Read as this package's arithmetic it means a real clamp at "
            f"zero, which is a guard and, on a rung that watches its "
            f"real-space seam, a measurable change to the drift -- for "
            f"that, wrap the call: `with clamped_inputs(model, 0.0): "
            f"rollout_...(..., clamp_rho=None)`, which installs exactly "
            f"the wrapper this argument would have. See aipf.solve.declare.")
    return value


def _input_guard(model, guard_floor: Optional[float]):
    """The ``clamp_rho`` guard's context, or ``nullcontext`` for no guard."""
    if guard_floor is None:
        return nullcontext(model)
    return clamped_inputs(model, guard_floor)


def rollout_deterministic(model: torch.nn.Module, rho_hat0: torch.Tensor,
                          boxes: torch.Tensor, T: torch.Tensor, dt: float,
                          n_steps: int, *, method: str = "euler",
                          state_proj=UNDECLARED, floor=UNDECLARED,
                          domain: Optional[TrustDomain] = None,
                          lo=UNDECLARED, hi=UNDECLARED,
                          clamp_rho=UNDECLARED,
                          kappa_roll: float = 0.0,
                          v_ext=None, T_field=None,
                          save_every: int = 1,
                          check_finite: bool = True) -> torch.Tensor:
    """Advance ``rho_hat0`` for ``n_steps``; returns ``(n_saved, B, n_species, Gx, Gy, Gzr)``.
    Per step: update, :func:`hermitianize`, :func:`project_state` (this order is load-bearing).
    ``state_proj``, ``floor`` and ``clamp_rho`` are declared, never defaulted."""
    if method not in _DETERMINISTIC_STEPS:
        raise ValueError(
            f"method must be one of {sorted(_DETERMINISTIC_STEPS)}, got "
            f"{method!r}")
    step_fn = _DETERMINISTIC_STEPS[method]
    # Both declarations are checked before the first model evaluation.
    check_state_projection(state_proj, floor, domain, lo=lo, hi=hi)
    guard_floor = _validated_clamp_rho(clamp_rho)
    n_species = rho_hat0.shape[1]
    ops = _ops_for(rho_hat0, None, n_species, model)

    rho_hat = rho_hat0
    traj: List[torch.Tensor] = [rho_hat.clone()]
    with _input_guard(model, guard_floor):
        for step in range(1, n_steps + 1):
            rho_hat = step_fn(model, rho_hat, boxes, T, dt, ops=ops,
                              kappa_roll=kappa_roll, v_ext=v_ext,
                              T_field=T_field)
            rho_hat = hermitianize(rho_hat, ops)
            rho_hat = project_state(rho_hat, ops, floor,
                                    state_proj=state_proj, domain=domain,
                                    lo=lo, hi=hi)
            if check_finite:
                assert_finite(rho_hat, f"rollout_deterministic step {step}")
            if step % save_every == 0 or step == n_steps:
                traj.append(rho_hat.clone())
    return torch.stack(traj)


# ---------------------------------------------------------------------------
# SDE integrator
# ---------------------------------------------------------------------------

def _cell_volume(boxes: torch.Tensor, grid: Tuple[int, int, int],
                 depth: Optional[float] = None) -> torch.Tensor:
    """``dV`` per sample; in two dimensions the cell's area times the declared ``depth``."""
    if len(grid) == 2:
        Gx, Gy = grid
        return (boxes[:, 0] / Gx) * (boxes[:, 1] / Gy) * float(depth)
    Gx, Gy, Gz = grid
    return (boxes[:, 0] / Gx) * (boxes[:, 1] / Gy) * (boxes[:, 2] / Gz)


def step_sde_euler_maruyama(model: torch.nn.Module, rho_hat: torch.Tensor,
                            boxes: torch.Tensor, T: torch.Tensor, dt: float,
                            kBT_noise: float, *,
                            noise_mode=UNDECLARED,
                            sigma_noise=UNDECLARED,
                            ops: Optional[SpectralOps] = None,
                            kappa_roll: float = 0.0,
                            v_ext=None, T_field=None,
                            generator: Optional[torch.Generator] = None,
                            depth: Optional[float] = None
                            ) -> torch.Tensor:
    """Euler-Maruyama for FDT-consistent conservative (Model B) noise:
        drho_i = div[M_ij grad(mu_j)] dt + div(zeta_i) dt,
        zeta_i(r) = sqrt(2 kBT_noise / (dV dt)) L_ij(rho(r)) w_j(r),  L L^T = M(rho(r), T),
    with the noise divergence multiplied by the declared filter ``G(k)``; zero at ``k=0``. Ito convention.
    Stationary: ``"none"`` gives ``S(k) = kBT H(k)^-1``, ``"gaussian"`` gives ``S(k) = G(k)^2 kBT H(k)^-1``.
    ``kBT_noise == 0`` is bit-for-bit :func:`step_euler`; ``None`` is refused. A two-dimensional model
    declares ``depth`` (:func:`check_depth`); a three-dimensional one leaves it unset."""
    n_species = rho_hat.shape[1]
    ops = _ops_for(rho_hat, ops, n_species, model)
    ndim = ops_ndim(ops)
    if ndim != 2 and depth is not None:
        check_depth(depth, ndim, True)
    # Validate the declaration before the deterministic delegation; build the filter after it.
    check_noise_declaration(noise_mode, sigma_noise)

    if kBT_noise is None:
        raise ValueError(
            "kBT_noise=None is refused: read as 'the thermal energy at "
            "T', honouring it silently would need a Boltzmann "
            "constant this package's core deliberately does not carry. "
            "Pass the thermal energy explicitly in the same unit as the "
            "model's free energy, or 0.0 for a deterministic step.")
    kBT = float(kBT_noise)
    if not (kBT >= 0.0):
        raise ValueError(
            f"kBT_noise must be a non-negative energy (0.0 means no "
            f"noise), got {kBT_noise!r}")
    if kBT == 0.0:
        return step_euler(model, rho_hat, boxes, T, dt, ops=ops,
                          kappa_roll=kappa_roll, v_ext=v_ext,
                          T_field=T_field)
    depth = check_depth(depth, ndim, True)

    filt = build_noise_filter(ops, boxes, noise_mode, sigma_noise)
    N = math.prod(ops.grid)
    grid = ops.grid
    B, n = rho_hat.shape[0], n_species
    ones = (1,) * ndim

    drift_hat = _rhs(model, rho_hat, boxes, T, ops, kappa_roll, v_ext,
                     T_field)
    rho = ops.irfft(rho_hat * N)

    M = model.mobility(rho, T)                                  # (B,n,n,*grid)
    M_flat = M.permute(*MATRIX_LAST[ndim]).reshape(-1, n, n)
    eye = torch.eye(n, device=rho.device, dtype=M_flat.dtype)
    L_flat = torch.linalg.cholesky(M_flat + 1e-12 * eye)         # (P,n,n)

    # One draw per species, direction and cell: (B, n, ndim, *grid).
    w = torch.randn(B, n, ndim, *grid, generator=generator,
                    device=rho.device, dtype=rho.dtype)
    if T_field is not None:
        # Local-equilibrium FDT: amplitude times sqrt(T(r)/T), exactly 1 for a uniform field.
        Tf = _validated_T_field(T_field, rho, ndim)
        w = w * torch.sqrt(
            Tf / T.to(rho.dtype).view(B, *ones)).unsqueeze(1).unsqueeze(1)
    w_flat = w.permute(*MATRIX_LAST[ndim]).reshape(-1, n, ndim)  # (P,n,ndim)
    zeta_flat = torch.einsum("pij,pjd->pid", L_flat, w_flat)     # (P,n,ndim)

    dV = _cell_volume(boxes, grid, depth)                        # (B,)
    amp = torch.sqrt(2.0 * kBT / (dV * dt))                      # (B,)
    amp_flat = amp.view(B, *ones).expand(B, *grid).reshape(-1, 1, 1)
    zeta_flat = zeta_flat * amp_flat
    zeta = zeta_flat.reshape(B, *grid, n, ndim).permute(*MATRIX_SECOND[ndim])

    ks = ops.k_axes(boxes)
    z_hat = [ops.rfft(zeta[:, :, d]) / N for d in range(ndim)]
    noise_div_hat = ops.div_hat(*z_hat, *ks)
    if filt is not None:
        noise_div_hat = noise_div_hat * filt
    # div_hat is already 0 at k=0; enforced anyway.
    noise_div_hat[(slice(None), slice(None)) + (0,) * ndim] = 0

    return rho_hat + dt * drift_hat + dt * noise_div_hat


def rollout_sde(model: torch.nn.Module, rho_hat0: torch.Tensor,
                boxes: torch.Tensor, T: torch.Tensor, dt: float,
                n_steps: int, kBT_noise: float, *,
                noise_mode=UNDECLARED, sigma_noise=UNDECLARED,
                state_proj=UNDECLARED, floor=UNDECLARED,
                domain: Optional[TrustDomain] = None,
                lo=UNDECLARED, hi=UNDECLARED,
                clamp_rho=UNDECLARED,
                kappa_roll: float = 0.0, v_ext=None, T_field=None,
                save_every: int = 1,
                generator: Optional[torch.Generator] = None,
                check_finite: bool = True,
                depth: Optional[float] = None) -> torch.Tensor:
    """The SDE analogue of :func:`rollout_deterministic`: noise is part of the update, then
    hermitianise, then project. No ``rho_floor`` fallback. ``depth``: a two-dimensional model's cell
    extent along the averaged axis, required under noise (:func:`check_depth`); unset in 3D."""
    check_state_projection(state_proj, floor, domain, lo=lo, hi=hi)
    guard_floor = _validated_clamp_rho(clamp_rho)
    n_species = rho_hat0.shape[1]
    ops = _ops_for(rho_hat0, None, n_species, model)
    if ops_ndim(ops) == 2 or depth is not None:
        noisy = kBT_noise is not None and float(kBT_noise) != 0.0
        check_depth(depth, ops_ndim(ops), noisy)

    rho_hat = rho_hat0
    traj: List[torch.Tensor] = [rho_hat.clone()]
    with _input_guard(model, guard_floor):
        for step in range(1, n_steps + 1):
            rho_hat = step_sde_euler_maruyama(
                model, rho_hat, boxes, T, dt, kBT_noise,
                noise_mode=noise_mode, sigma_noise=sigma_noise, ops=ops,
                kappa_roll=kappa_roll, v_ext=v_ext, T_field=T_field,
                generator=generator, depth=depth)
            rho_hat = hermitianize(rho_hat, ops)
            rho_hat = project_state(rho_hat, ops, floor,
                                    state_proj=state_proj, domain=domain,
                                    lo=lo, hi=hi)
            if check_finite:
                assert_finite(rho_hat, f"rollout_sde step {step}")
            if step % save_every == 0 or step == n_steps:
                traj.append(rho_hat.clone())
    return torch.stack(traj)
