"""Nyquist repair and the state-projection stack a rollout runs every step.
``hermitianize`` projects onto real-field half-spectra each step. ``project_state`` alternates one
declared admissible set with an exact ``k=0`` mass restore, and applies no band mask. Modes:
``"floor"`` (``rho >= floor``), ``"box"`` (``lo <= rho <= hi``),
``"domain"`` (the trust trapezoid, then clamped at ``lo > 0``)."""
from __future__ import annotations

import math
from typing import Optional

import torch

from aipf.spectral import (CHANNEL_LAST, CHANNEL_SECOND, DC, SpectralOps,
                           ops_ndim)

from .declare import UNDECLARED
from .trust_domain import TrustDomain, project_trust_domain

#: The admissible ``state_proj`` targets.
STATE_PROJ_MODES = ("floor", "box", "domain")

#: (clamp, restore-mass) alternations per call.
STATE_PROJ_ITERS = 3


def hermitianize(rho_hat: torch.Tensor, ops: SpectralOps) -> torch.Tensor:
    """Project ``rho_hat`` onto the ``rfft`` of real fields; ``k=0`` restored bitwise with ``Im = 0``.
    In two dimensions the Nyquist line is the half axis's last column, repaired the same way."""
    dc_at = DC[ops_ndim(ops)]
    N = math.prod(ops.grid)
    dc = rho_hat[dc_at].clone()
    out = ops.rfft(ops.irfft(rho_hat * N)) / N
    out[dc_at] = dc.real
    return out


def _bound(value, ref: torch.Tensor, name: str, ndim: int = 3) -> torch.Tensor:
    """A scalar or per-channel bound, broadcastable against a ``(B, n, *grid)`` field of ``ndim`` axes."""
    t = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    if t.ndim == 0:
        return t
    if t.ndim != 1 or int(t.shape[0]) != int(ref.shape[1]):
        raise ValueError(
            f"{name} must be a scalar or one number per channel "
            f"({int(ref.shape[1])}), got shape {tuple(t.shape)}")
    return t.view(1, -1, *(1,) * ndim)


def restore_mass(rho_r: torch.Tensor, target_mean: torch.Tensor,
                 lo, ndim: int = 3) -> torch.Tensor:
    """Move each channel's mean to ``target_mean`` without pushing any cell below ``lo``:
    ``rho' = rho + max(d, 0) + min(d, 0) * (rho - lo) / mean(rho - lo)``; full drain if ``target_mean < lo``.
    ``ndim`` is the field's declared number of spatial axes (the trailing ones averaged over)."""
    dims = (-3, -2, -1) if ndim == 3 else (-2, -1)
    cell = (Ellipsis,) + (None,) * ndim
    d = target_mean - rho_r.mean(dim=dims)                    # (B, n)
    head = (rho_r - lo).clamp(min=0.0)
    head_mean = head.mean(dim=dims)
    tiny = torch.finfo(rho_r.dtype).tiny
    frac = (d.clamp(max=0.0) / head_mean.clamp(min=tiny)).clamp(min=-1.0)
    return (rho_r + d.clamp(min=0.0)[cell]
            + frac[cell] * head)


def check_state_projection(state_proj, floor, domain, *, lo=UNDECLARED,
                           hi=UNDECLARED) -> None:
    """Validate a state-projection declaration; raises exactly what :func:`project_state` raises.
    ``floor`` is read by ``"floor"`` only; ``"box"`` needs ``lo`` and ``hi``;
    ``"domain"`` needs a ``TrustDomain`` and ``lo > 0``."""
    if state_proj is UNDECLARED:
        raise ValueError(
            "state_proj must be declared: 'floor' (project onto "
            "rho >= floor), 'box' (project "
            "onto lo <= rho <= hi) or 'domain' "
            "(Euclidean projection onto the experiment's declared trust "
            "trapezoid, which is what the published rollouts used). "
            "There is no default, deliberately -- see aipf.solve.declare.")
    if state_proj not in STATE_PROJ_MODES:
        raise ValueError(
            f"state_proj must be one of {list(STATE_PROJ_MODES)}, got "
            f"{state_proj!r}")
    if state_proj == "floor":
        if floor is UNDECLARED:
            raise ValueError(
                "floor must be declared alongside state_proj='floor': it "
                "is that mode's per-cell density floor (0.0 asks for no "
                "projection at all), and it is NOT consulted for "
                "state_proj='box' or state_proj='domain', which declare "
                "their bounds as lo= and hi=. There is no default -- see "
                "aipf.solve.declare.")
        fl = float(floor)
        if fl != fl or abs(fl) == float("inf"):
            raise ValueError(f"floor must be finite, got {floor!r}")
    if state_proj == "domain":
        if domain is None:
            raise ValueError(
                "state_proj='domain' requires a declared TrustDomain: this "
                "package's core carries no trapezoid corners, so there is "
                "nothing for it to fall back to. See "
                "aipf.solve.trust_domain.")
        if not isinstance(domain, TrustDomain):
            raise ValueError(
                f"domain must be an aipf.solve.trust_domain.TrustDomain, "
                f"got {type(domain).__name__}")
        if lo is UNDECLARED or not (float(lo) > 0.0):
            raise ValueError(
                "state_proj='domain' projects onto a trapezoid whose two "
                "edges lie ON the axes, so a density can leave it at zero "
                "or slightly negative; pass lo=<positive float> for the "
                "mass restore, or state_proj='floor' / 'box'")
    elif state_proj == "box":
        if lo is UNDECLARED or hi is UNDECLARED:
            raise ValueError("state_proj='box' needs both lo= and hi=")


def project_state(rho_hat: torch.Tensor, ops: SpectralOps, floor,
                  *, state_proj=UNDECLARED,
                  domain: Optional[TrustDomain] = None,
                  lo=UNDECLARED, hi=UNDECLARED,
                  iters: int = STATE_PROJ_ITERS) -> torch.Tensor:
    """Project ``rho_hat`` onto the declared admissible set, preserving ``k=0`` exactly.
    Per pass: irfft, project (exit if admissible), :func:`restore_mass`, rfft, write the original ``k=0``.
    ``"floor"`` with ``floor <= 0`` returns the input untouched."""
    check_state_projection(state_proj, floor, domain, lo=lo, hi=hi)
    if state_proj == "floor":
        lo_value, hi_value = float(floor), None
        if lo_value <= 0.0:
            return rho_hat
    elif state_proj == "box":
        lo_value, hi_value = lo, hi
    else:
        lo_value, hi_value = float(lo), None

    ndim = ops_ndim(ops)
    dc_at = DC[ndim]
    N = math.prod(ops.grid)
    dc = rho_hat[dc_at].clone()
    target = dc.real
    for _ in range(iters):
        rho_r = ops.irfft(rho_hat * N)
        lo_t = _bound(lo_value, rho_r, "lo", ndim)
        hi_t = None if hi_value is None else _bound(hi_value, rho_r, "hi", ndim)
        if state_proj == "domain":
            B = rho_r.shape[0]
            grid = rho_r.shape[-ndim:]
            n = rho_r.shape[1]
            flat = rho_r.permute(*CHANNEL_LAST[ndim]).reshape(-1, n)
            proj = project_trust_domain(flat, domain)
            # Identity, not equality: `project_trust_domain` returns its argument when all rows are inside.
            if proj is flat:
                if bool((rho_r >= lo_t).all()):
                    break
            else:
                rho_r = proj.reshape(B, *grid, n).permute(*CHANNEL_SECOND[ndim])
            rho_r = rho_r.clamp(min=lo_t)
        elif state_proj == "box":
            if bool((rho_r >= lo_t).all()) and bool((rho_r <= hi_t).all()):
                break
            rho_r = rho_r.clamp(min=lo_t).clamp(max=hi_t)
        else:
            if bool((rho_r >= lo_t).all()):
                break
            rho_r = rho_r.clamp(min=lo_t)
        rho_r = restore_mass(rho_r, target, lo_t, ndim)
        rho_hat = ops.rfft(rho_r) / N
        rho_hat[dc_at] = dc
    return rho_hat


#: The admissible sets :func:`project_state_uniform_shift` projects onto.
SHIFT_PROJ_MODES = ("floor", "domain")


def project_state_uniform_shift(rho_hat: torch.Tensor, ops: SpectralOps, *,
                                state_proj, floor,
                                domain: Optional[TrustDomain],
                                iters: int = STATE_PROJ_ITERS) -> torch.Tensor:
    """The published rollouts' projection: per pass irfft, project (exit if admissible),
    rfft, write the original ``k=0`` (a uniform shift). ``"domain"`` has NO positive floor and a cell can
    come back at zero or below; ``"floor"`` with ``floor <= 0`` is a no-op."""
    if state_proj not in SHIFT_PROJ_MODES:
        raise ValueError(
            f"state_proj must be one of {list(SHIFT_PROJ_MODES)}, got "
            f"{state_proj!r}")
    if state_proj == "domain" and not isinstance(domain, TrustDomain):
        raise ValueError(
            "state_proj='domain' requires a declared TrustDomain; core "
            "carries no trapezoid corners")
    floor = float(floor)
    if state_proj == "floor" and floor <= 0.0:
        return rho_hat
    ndim = ops_ndim(ops)
    dc_at = DC[ndim]
    N = math.prod(ops.grid)
    dc = rho_hat[dc_at].clone()
    for _ in range(iters):
        rho_r = ops.irfft(rho_hat * N)
        if state_proj == "domain":
            B, grid, n = rho_r.shape[0], rho_r.shape[-ndim:], rho_r.shape[1]
            flat = rho_r.permute(*CHANNEL_LAST[ndim]).reshape(-1, n)
            proj = project_trust_domain(flat, domain)
            if proj is flat:
                break
            rho_r = proj.reshape(B, *grid, n).permute(*CHANNEL_SECOND[ndim])
        else:
            if bool((rho_r >= floor).all()):
                break
            rho_r = rho_r.clamp(min=floor)
        rho_hat = ops.rfft(rho_r) / N
        rho_hat[dc_at] = dc
    return rho_hat
