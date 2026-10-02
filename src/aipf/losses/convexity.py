"""L_conv and L_Gamma: the two convexity penalties on a caller-computed Hessian."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_CONV_FORMS = ("hinge", "hinge0", "softplus")


def l_conv(H: torch.Tensor, *, form: str = "hinge",
           margin: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Soft ``lambda_min(H) >= margin`` penalty; returns ``(loss, {"viol_frac", "lam_worst"})``.

    ``form``: ``"hinge"`` ``relu(margin - lam)^2``, ``"hinge0"`` the same at margin 0,
    ``"softplus"`` ``0.02 * softplus((margin - lam) / 0.02)``."""
    if form not in _CONV_FORMS:
        raise ValueError(f"form={form!r}, expected one of {_CONV_FORMS}")
    lam_min = torch.linalg.eigvalsh(H)[..., 0]
    if form == "hinge":
        loss = F.relu(margin - lam_min).pow(2).mean()
    elif form == "hinge0":
        loss = F.relu(-lam_min).pow(2).mean()
    else:  # softplus
        loss = 0.02 * F.softplus((margin - lam_min) / 0.02).mean()
    aux = {"viol_frac": float((lam_min.detach() < 0).float().mean()),
           "lam_worst": float(lam_min.detach().min())}
    return loss, aux


def gamma_closed_form(H: torch.Tensor, n_density: torch.Tensor,
                       prefactor: torch.Tensor, v: torch.Tensor,
                       kBT: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """``Gamma = n_density * prefactor * det(H) / (kBT * (v^T H v))``, denominator signed-clamped at ``eps``."""
    det = torch.linalg.det(H)
    q = torch.einsum("...i,...ij,...j->...", v, H, v)
    sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
    q_safe = sign * torch.clamp(q.abs(), min=eps)
    return n_density * prefactor * det / (kBT * q_safe)


def gamma_on_path(H: torch.Tensor, n_density: torch.Tensor, x: torch.Tensor,
                  kBT: torch.Tensor, *, x_channel: int, eps: float) -> torch.Tensor:
    """Two channels, ``x`` the fraction of channel ``x_channel``: ``n x (1-x) det(H) / (kBT q)``,
    ``q = x^2 H_cc + 2x(1-x) H_01 + (1-x)^2 H_oo`` signed-clamped at ``eps``; the 2x2 algebra written out."""
    if int(x_channel) not in (0, 1):
        raise ValueError(f"x_channel={x_channel!r}: two channels, 0 or 1")
    c, o = int(x_channel), 1 - int(x_channel)
    H00, H01, H11 = H[..., 0, 0], H[..., 0, 1], H[..., 1, 1]
    Hcc, Hoo = H[..., c, c], H[..., o, o]
    det = H00 * H11 - H01 * H01
    q = x * x * Hcc + 2.0 * x * (1.0 - x) * H01 + (1.0 - x) * (1.0 - x) * Hoo
    sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
    q_safe = sign * torch.clamp(q.abs(), min=eps)
    return n_density * x * (1.0 - x) * det / (kBT * q_safe)


def l_gamma(gamma: torch.Tensor, *, h: float) -> tuple[torch.Tensor, dict]:
    """Soft ``d^2(Gamma)/dx^2 >= 0`` on grid spacing ``h`` (five-point stencil, interior ``[..., 2:-2]``).

    Returns ``(loss, {"viol_frac", "d2_worst"})``."""
    d2 = (-gamma[..., :-4] + 16.0 * gamma[..., 1:-3] - 30.0 * gamma[..., 2:-2]
         + 16.0 * gamma[..., 3:-1] - gamma[..., 4:]) / (12.0 * h * h)
    loss = F.relu(-d2).pow(2).mean()
    aux = {"viol_frac": float((d2.detach() < 0).float().mean()),
           "d2_worst": float(d2.detach().min())}
    return loss, aux
