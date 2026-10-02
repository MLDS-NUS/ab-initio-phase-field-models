"""L_S and L_bulk: the two static-structure-factor anchors, model ``H`` against ``k_BT . S(k)^-1``.

``L_S`` acts on shells ``0 < k <= k_fit`` (masked mean over padded shells); ``L_bulk`` is ``k = 0``
through ``S_cc(0) = k_BT . z^T H(0)^-1 z / rho_tot``. The ``L_bulk`` residual form is declared."""
from __future__ import annotations

import torch

_RESIDUAL_FORMS = ("relative_inverse", "sigma_chi2")


def sinv_rootinv(Sinv: torch.Tensor) -> torch.Tensor:
    """Symmetric inverse square root ``R = V diag(lambda^-1/2) V^T``, so ``R . Sinv . R = I``;
    eigenvalues floored at 1e-12."""
    evals, evecs = torch.linalg.eigh(Sinv)
    inv_root = evals.clamp(min=1e-12).rsqrt()
    return evecs @ torch.diag_embed(inv_root) @ evecs.transpose(-1, -2)


def l_s(H: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *,
        metric: str = "rel_frob", target_rootinv: torch.Tensor | None = None,
        row_weight: torch.Tensor | None = None) -> torch.Tensor:
    """The k-resolved shell anchor, masked-mean over (row, shell); ``H``, ``target`` ``(..., n, n)``.

    ``metric``: ``"rel_frob"`` ``||H - target||_F^2 / ||target||_F^2`` or ``"s_metric"``
    ``||R H R - I||_F^2`` with ``R = target^-1/2``."""
    if metric == "rel_frob":
        diff = H - target
        num = diff.pow(2).sum(dim=(-2, -1))
        den = target.pow(2).sum(dim=(-2, -1)).clamp(min=1e-30)
        per = num / den
    elif metric == "s_metric":
        R = target_rootinv if target_rootinv is not None else sinv_rootinv(target)
        A = R @ H @ R
        n = A.shape[-1]
        eye = torch.eye(n, dtype=A.dtype, device=A.device)
        per = (A - eye).pow(2).sum(dim=(-2, -1))
    else:
        raise ValueError(f"unknown metric {metric!r}, expected 'rel_frob' or "
                         f"'s_metric'")
    m = mask.to(per.dtype)
    if row_weight is not None:
        m = m * row_weight.to(per.dtype).unsqueeze(-1)
    return (per * m).sum() / m.sum().clamp(min=1.0)


def l_bulk(H0: torch.Tensor, kBT: torch.Tensor, zvec: torch.Tensor,
           rho_tot: torch.Tensor, target: torch.Tensor, *, residual: str,
           sigma: torch.Tensor | None = None,
           row_weight: torch.Tensor | None = None, jitter: float = 1e-10,
           clamp: float | None = None) -> torch.Tensor:
    """The k=0 bulk anchor on ``H0`` ``(P, n, n)``, ``zvec`` ``(P, n)``,
    ``kBT``/``rho_tot``/``target`` ``(P,)``.

    ``residual``: ``"relative_inverse"`` ``(target/model - 1)**2`` (optional ``clamp``) or
    ``"sigma_chi2"`` ``((1/model - 1/target)/sigma)**2`` (needs ``sigma``)."""
    if residual not in _RESIDUAL_FORMS:
        raise ValueError(f"residual={residual!r}, expected one of "
                         f"{_RESIDUAL_FORMS}")
    n = H0.shape[-1]
    eye = torch.eye(n, dtype=H0.dtype, device=H0.device)
    y = torch.linalg.solve(H0 + jitter * eye, zvec.unsqueeze(-1)).squeeze(-1)
    q = (zvec * y).sum(-1)  # z^T H0^-1 z, per row

    if residual == "relative_inverse":
        model = kBT * q / rho_tot
        resid = target / model - 1.0
        if clamp is not None:
            resid = resid.clamp(min=-clamp, max=clamp)
    else:  # sigma_chi2
        if sigma is None:
            raise ValueError("residual='sigma_chi2' requires sigma")
        model_inv = rho_tot / (kBT * q)
        target_inv = 1.0 / target
        resid = (model_inv - target_inv) / sigma

    per_row = resid.pow(2)
    if row_weight is None:
        return per_row.mean()
    w = row_weight.to(per_row.dtype)
    return (w * per_row).sum() / w.sum().clamp(min=1e-12)
