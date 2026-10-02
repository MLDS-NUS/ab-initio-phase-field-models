"""The semi-implicit (IMEX) scheme the published rollouts ran, deterministic and with conserved noise.
Frozen operator ``A = I + dt k^2 M_s H(k)``, ``H = Hess f_loc(rho_bar) + W_hat(k)``; per step
``rho^{n+1} = A^-1 (rho^n + dt F(rho^n) + dt k^2 M_s H rho^n + dt n_hat)``, then hermitianise, then project.
Reads the model's ``kernel.w_hat`` and ``f_local`` (a pair-kernel functional); other forms are refused."""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, List, Optional

import torch

from aipf.solve import hermitianize, project_state
from aipf.solve.declare import UNDECLARED
from aipf.solve.noise import build_noise_filter, check_m_stab
from aipf.solve.projection import project_state_uniform_shift
from aipf.solve.trust_domain import TrustDomain
from aipf.spectral import SpectralOps

from .noise import (NOISE_EVAL_FRAC, check_noise_eval, draw_w,
                    max_norm_mobility, noise_div_hat, zeta_from_w)

#: How the state projection restores mass: ``"shift"`` writes ``k=0`` back (the published rollouts),
#: ``"headroom"`` is :func:`aipf.solve.project_state` (never below ``lo``).
MASS_RESTORES = ("shift", "headroom")


def _require_pair_kernel(model) -> None:
    if not (hasattr(model, "kernel") and hasattr(model.kernel, "w_hat")
            and hasattr(model, "f_local")
            and hasattr(model.f_local, "f_pointwise")
            and hasattr(model.f_local, "mu_pointwise")):
        raise TypeError(
            f"{type(model).__name__} exposes no kernel.w_hat and "
            f"f_local.{{f_pointwise, mu_pointwise}}; the semi-implicit "
            f"operator is frozen from exactly those. Use aipf.solve's "
            f"explicit integrators for this functional form.")


@contextmanager
def pointwise_guard(model, floor: Optional[float]) -> Iterator[None]:
    """Clamp the density the pointwise ``mu`` and the mobility read at ``floor`` (``None``: no guard).
    The kernel term and the state are untouched."""
    if floor is None:
        yield
        return
    f_local = model.f_local
    had_mu = "mu_pointwise" in f_local.__dict__
    had_mob = "mobility" in model.__dict__
    mu0, mob0 = f_local.mu_pointwise, model.mobility

    def _mu(rho, kBT, _mu0=mu0, _eps=float(floor)):
        return _mu0(rho.clamp(min=_eps), kBT)

    def _mob(rho, T, _mob0=mob0, _eps=float(floor)):
        return _mob0(rho.clamp(min=_eps), T)

    f_local.mu_pointwise = _mu
    model.mobility = _mob
    try:
        yield
    finally:
        if had_mu:
            f_local.mu_pointwise = mu0
        else:
            del f_local.mu_pointwise
        if had_mob:
            model.mobility = mob0
        else:
            del model.mobility


def _full_grid(rho_hat0: torch.Tensor, what: str) -> tuple:
    if int(rho_hat0.shape[0]) != 1:
        raise ValueError(f"{what} is single-field; got batch {int(rho_hat0.shape[0])}")
    Gx, Gy, Gzr = (int(s) for s in rho_hat0.shape[-3:])
    return Gx, Gy, 2 * (Gzr - 1)


@contextmanager
def pointwise_fields(model, rho_hat0: torch.Tensor, kbt_field=None,
                     v_ext=None) -> Iterator[Optional[torch.Tensor]]:
    """Per-cell ``kBT`` (``(Gx, Gy, Gz)``, energy) replaces the pointwise ``mu``'s scalar one, then a static
    ``v_ext`` (``(n, Gx, Gy, Gz)``, energy) is added to it; full-grid calls only. Yields the ``kBT`` field."""
    device, n = rho_hat0.device, int(rho_hat0.shape[1])
    kbt_t = V = None
    if kbt_field is not None:
        grid = _full_grid(rho_hat0, "kbt_field")
        kbt_t = torch.as_tensor(kbt_field, dtype=torch.float32, device=device)
        if tuple(kbt_t.shape) != grid:
            raise ValueError(f"kbt_field must have shape {grid} (one kBT per cell, no "
                             f"species axis), got {tuple(kbt_t.shape)}")
        if not bool(torch.isfinite(kbt_t).all()) or float(kbt_t.min()) <= 0.0:
            raise ValueError(f"kbt_field must be finite and positive everywhere; got "
                             f"min {float(kbt_t.min())}, max {float(kbt_t.max())}")
    if v_ext is not None:
        grid = _full_grid(rho_hat0, "v_ext")
        V = torch.as_tensor(v_ext, dtype=torch.float32, device=device)
        if tuple(V.shape) != (n, *grid):
            raise ValueError(f"v_ext must have shape {(n, *grid)} (species on the full "
                             f"grid), got {tuple(V.shape)}")
    f_local = model.f_local
    had_mu = "mu_pointwise" in f_local.__dict__
    mu0 = f_local.mu_pointwise
    if kbt_t is not None:
        def _mu_kbt(rho, kBT_flat, _mu0=f_local.mu_pointwise, _k=kbt_t.reshape(-1)):
            if rho.shape[0] == _k.shape[0]:
                kBT_flat = _k
            return _mu0(rho, kBT_flat)

        f_local.mu_pointwise = _mu_kbt
    if V is not None:
        def _mu_vext(rho, kBT_flat, _mu0=f_local.mu_pointwise,
                     _V=V.permute(1, 2, 3, 0).reshape(-1, n)):
            mu = _mu0(rho, kBT_flat)
            if rho.shape[0] == _V.shape[0]:
                mu = mu + _V
            return mu

        f_local.mu_pointwise = _mu_vext
    try:
        yield kbt_t
    finally:
        if had_mu:
            f_local.mu_pointwise = mu0
        elif "mu_pointwise" in f_local.__dict__:
            del f_local.mu_pointwise


def local_hessian(model, rho_bar: torch.Tensor, kBT: float) -> torch.Tensor:
    """``d^2 f_loc / d rho^2`` at ``rho_bar`` (shape ``(n,)``), by double autograd; ``(n, n)``."""
    rho = rho_bar.detach().clone().unsqueeze(0).requires_grad_(True)
    kBT_t = torch.tensor([float(kBT)], dtype=rho.dtype, device=rho.device)
    with torch.enable_grad():
        f = model.f_local.f_pointwise(rho, kBT_t).sum()
        g = torch.autograd.grad(f, rho, create_graph=True)[0].squeeze(0)
        H = torch.stack([torch.autograd.grad(g[a], rho, retain_graph=True)[0]
                         .squeeze(0) for a in range(rho.shape[-1])])
    return H.detach()


def _inverse(A: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of ``(..., 2, 2)``; ``torch.linalg.inv`` for any other size."""
    if A.shape[-1] != 2:
        return torch.linalg.inv(A)
    a, b = A[..., 0, 0], A[..., 0, 1]
    c, d = A[..., 1, 0], A[..., 1, 1]
    det = a * d - b * c
    inv = torch.stack([torch.stack([d, -b], -1), torch.stack([-c, a], -1)], -2)
    return inv / det.unsqueeze(-1).unsqueeze(-1)


def _check_projection(state_proj, state_clamp, domain, mass_restore) -> None:
    if mass_restore not in MASS_RESTORES:
        raise ValueError(
            f"mass_restore must be one of {list(MASS_RESTORES)}, got "
            f"{mass_restore!r}: 'shift' is the published rollouts' uniform k=0 "
            f"write-back, 'headroom' is aipf.solve.project_state")
    if state_proj not in ("floor", "domain"):
        raise ValueError(
            f"state_proj must be 'floor' or 'domain', got {state_proj!r}")
    if state_clamp is UNDECLARED:
        raise ValueError(
            "state_clamp must be declared: the 'floor' mode's per-cell floor, "
            "and the 'domain' mode's lo= under mass_restore='headroom'")
    if state_proj == "domain" and not isinstance(domain, TrustDomain):
        raise ValueError("state_proj='domain' requires a declared TrustDomain")


def _project(rho_hat, ops, state_proj, state_clamp, domain, mass_restore):
    if mass_restore == "shift":
        return project_state_uniform_shift(rho_hat, ops, state_proj=state_proj,
                                           floor=float(state_clamp),
                                           domain=domain)
    if state_proj == "floor":
        return project_state(rho_hat, ops, float(state_clamp),
                             state_proj="floor")
    return project_state(rho_hat, ops, UNDECLARED, state_proj="domain",
                         domain=domain, lo=float(state_clamp))


@torch.no_grad()
def rollout_imex(model, rho_hat0: torch.Tensor, box: torch.Tensor, T: float,
                 dt: float, n_steps: int, *, kB: float, m_stab=UNDECLARED,
                 state_proj=UNDECLARED, state_clamp=UNDECLARED,
                 domain: Optional[TrustDomain] = None,
                 mass_restore=UNDECLARED, clamp_rho=UNDECLARED,
                 noise: Optional[dict] = None, save_every: int = 1,
                 generator: Optional[torch.Generator] = None,
                 v_ext=None, kbt_field=None) -> torch.Tensor:
    """Advance one field ``rho_hat0`` ``(1, n, Gx, Gy, Gzr)`` in box ``(3,)`` at scalar ``T``; returns the
    saved states ``(n_saved, n, Gx, Gy, Gzr)`` on the CPU. ``noise=None`` is deterministic; otherwise a
    dict with ``kBT_noise``, ``noise_scale``, ``noise_mode``, ``sigma_noise``, ``noise_eval``,
    ``predictor_floor``. ``v_ext``/``kbt_field``: a static external potential and a per-cell ``kBT``
    (:func:`pointwise_fields`). Every other knob is declared."""
    _require_pair_kernel(model)
    m_stab = check_m_stab(m_stab)
    _check_projection(state_proj, state_clamp, domain, mass_restore)
    if clamp_rho is UNDECLARED:
        raise ValueError(
            "clamp_rho must be declared: the floor the pointwise mu and the "
            "mobility read the density at, or None for no guard")
    if noise is not None:
        noise = dict(noise)
        check_noise_eval(noise["noise_eval"])
        if float(noise["kBT_noise"]) * float(noise["noise_scale"]) == 0.0:
            noise = None
    model.eval()
    device = rho_hat0.device
    boxes = box.unsqueeze(0).to(device)
    rho_hat = rho_hat0.clone()
    n = rho_hat.shape[1]
    rho_bar = rho_hat0[0, :, 0, 0, 0].real.clone()
    kBT = kB * float(T)
    T_t = torch.tensor([float(T)], device=device)
    with pointwise_guard(model, clamp_rho), \
            pointwise_fields(model, rho_hat0, kbt_field, v_ext) as kbt_t:
        w_scale = None
        if kbt_t is not None:
            if noise is not None:
                w_scale = torch.sqrt(kbt_t / kBT)
            kBT = float(kbt_t.mean(dtype=torch.float64))
            T_t = torch.tensor([kBT / kB], device=device)
        return _run(model, rho_hat, boxes, T_t, kBT, rho_bar, n, dt, n_steps,
                    save_every, m_stab, state_proj, state_clamp, domain,
                    mass_restore, noise, generator, w_scale)


def _run(model, rho_hat, boxes, T_t, kBT, rho_bar, n, dt, n_steps, save_every,
         m_stab, state_proj, state_clamp, domain, mass_restore, noise,
         generator, w_scale=None) -> torch.Tensor:
    device = rho_hat.device
    Hb = local_hessian(model, rho_bar, kBT)
    Gx, Gy, Gzr = (int(s) for s in rho_hat.shape[-3:])
    ops = SpectralOps((Gx, Gy, 2 * (Gzr - 1)), n,
                      nyquist_mask=model.ops.nyquist_mask).to(device)
    N = ops.grid[0] * ops.grid[1] * ops.grid[2]
    kx, ky, kz = ops.k_axes(boxes)
    kmag = torch.sqrt(kx * kx + ky * ky + kz * kz)[:, 0]
    k2 = (kmag * kmag).unsqueeze(-1).unsqueeze(-1)
    H_k = model.kernel.w_hat(kmag) + Hb
    if m_stab == "mean":
        M_s = model.mobility(rho_bar.view(1, n, 1, 1, 1), T_t)[0, :, :, 0, 0, 0]
    else:
        M_s = max_norm_mobility(model, ops.irfft(rho_hat * N), T_t)
    dtL = dt * k2 * torch.einsum("ij,...jk->...ik", M_s, H_k)
    A_inv = _inverse(torch.eye(n, device=device) + dtL).to(torch.complex64)
    MH = dtL.to(torch.complex64)

    frac = filt = amp = None
    if noise is not None:
        filt = build_noise_filter(ops, boxes, noise["noise_mode"],
                                  noise["sigma_noise"])
        dV = float(boxes[0].prod()) / N
        amp = float(noise["noise_scale"]) * math.sqrt(
            2.0 * float(noise["kBT_noise"]) / (dV * dt))
        frac = NOISE_EVAL_FRAC.get(noise["noise_eval"])
        if frac is not None:
            A_pred_inv = _inverse(torch.eye(n, device=device)
                                  + frac * dtL).to(torch.complex64)
            MH_pred = (frac * dtL).to(torch.complex64)
            amp_floor = float(noise["predictor_floor"])

    traj: List[torch.Tensor] = [rho_hat[0].cpu()]
    for step in range(1, n_steps + 1):
        F_hat = model(rho_hat, boxes, T_t)
        rh = rho_hat.permute(0, 2, 3, 4, 1).unsqueeze(-1)
        Fh = F_hat.permute(0, 2, 3, 4, 1).unsqueeze(-1)
        if noise is None:
            rhs = rh + dt * Fh + MH @ rh
        else:
            rho_now = ops.irfft(rho_hat * N)
            w = draw_w(ops.grid, n, generator, device, rho_now.dtype)
            if w_scale is not None:
                w = w * w_scale
            if frac is None:
                zeta = zeta_from_w(model, rho_now, T_t, amp, w)
            else:
                zeta0 = zeta_from_w(model, rho_now, T_t, amp, w)
                n0_hat = noise_div_hat(ops, zeta0, kx, ky, kz, filt)
                star = (A_pred_inv @ (rh + (frac * dt) * Fh + MH_pred @ rh)) \
                    .squeeze(-1).permute(0, 4, 1, 2, 3)
                rho_amp = ops.irfft((star + (frac * dt) * n0_hat) * N)
                zeta = zeta_from_w(model, rho_amp.clamp(min=amp_floor), T_t,
                                   amp, w)
            n_hat = noise_div_hat(ops, zeta, kx, ky, kz, filt)
            nh = n_hat.permute(0, 2, 3, 4, 1).unsqueeze(-1)
            rhs = rh + dt * Fh + MH @ rh + dt * nh
        rho_hat = (A_inv @ rhs).squeeze(-1).permute(0, 4, 1, 2, 3)
        rho_hat = hermitianize(rho_hat, ops)
        rho_hat = _project(rho_hat, ops, state_proj, state_clamp, domain,
                           mass_restore)
        if step % save_every == 0 or step == n_steps:
            traj.append(rho_hat[0].cpu())
    return torch.stack(traj)
