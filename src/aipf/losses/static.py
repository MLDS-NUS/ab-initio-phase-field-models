"""L_S and L_bulk: the two static-structure-factor anchors, model ``H`` against ``k_BT . S(k)^-1``.

``L_S`` acts on shells ``0 < k <= k_fit`` (masked mean over padded shells); ``L_bulk`` is ``k = 0``
through ``S_cc(0) = k_BT . z^T H(0)^-1 z / rho_tot``, a scalar for two channels and the
``(n-1) x (n-1)`` concentration block ``k_BT . Z^T H_PP(0)^-1 Z / rho_tot`` beyond. The ``L_bulk``
residual form is declared."""
from __future__ import annotations

import torch

_RESIDUAL_FORMS = ("relative_inverse", "sigma_chi2", "s_metric")


def sinv_rootinv(Sinv: torch.Tensor) -> torch.Tensor:
    """Symmetric inverse square root ``R = V diag(lambda^-1/2) V^T``, so ``R . Sinv . R = I``;
    eigenvalues floored at 1e-12."""
    evals, evecs = torch.linalg.eigh(Sinv)
    inv_root = evals.clamp(min=1e-12).rsqrt()
    return evecs @ torch.diag_embed(inv_root) @ evecs.transpose(-1, -2)


def _pairs(active: torch.Tensor) -> torch.Tensor:
    """``(..., n, n)`` bool: both channels of the pair present, from ``(..., n)``."""
    return active.unsqueeze(-1) & active.unsqueeze(-2)


def _principal_block(H: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """``H`` with every row and column outside the active principal block replaced by the identity's."""
    n = H.shape[-1]
    eye = torch.eye(n, dtype=H.dtype, device=H.device)
    return torch.where(pairs, H, eye)


def l_s(H: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *,
        metric: str = "rel_frob", target_rootinv: torch.Tensor | None = None,
        row_weight: torch.Tensor | None = None,
        active: torch.Tensor | None = None) -> torch.Tensor:
    """The k-resolved shell anchor, masked-mean over (row, shell); ``H``, ``target`` ``(..., n, n)``.

    ``metric``: ``"rel_frob"`` ``||H - target||_F^2 / ||target||_F^2`` or ``"s_metric"``
    ``||R H R - I||_F^2`` with ``R = target^-1/2``.
    ``active`` ``(P, n)`` bool, each row's present channels, with ``H`` ``(P, shells, n, n)``: given, a row
    is compared on its active principal block alone (``H`` is the identity outside it, and only active
    pairs enter the residual and the denominator); ``target`` and ``target_rootinv`` then carry the
    identity outside it."""
    pairs = None
    if active is not None:
        if H.dim() != 4 or active.shape != (H.shape[0], H.shape[-1]):
            raise ValueError(
                f"active has shape {tuple(active.shape)} against H "
                f"{tuple(H.shape)}; the masked shell anchor takes H "
                f"(P, shells, n, n) and one (P, n) mask row per state")
        pairs = _pairs(active).unsqueeze(1)          # (P, 1, n, n), over the shells
        H = _principal_block(H, pairs)
    if metric == "rel_frob":
        diff = H - target
        num = diff.pow(2)
        den = target.pow(2)
        if pairs is not None:
            num = torch.where(pairs, num, 0.0)
            den = torch.where(pairs, den, 0.0)
        num = num.sum(dim=(-2, -1))
        den = den.sum(dim=(-2, -1)).clamp(min=1e-30)
        per = num / den
    elif metric == "s_metric":
        R = target_rootinv if target_rootinv is not None else sinv_rootinv(target)
        A = R @ H @ R
        n = A.shape[-1]
        eye = torch.eye(n, dtype=A.dtype, device=A.device)
        sq = (A - eye).pow(2)
        if pairs is not None:
            sq = torch.where(pairs, sq, 0.0)
        per = sq.sum(dim=(-2, -1))
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
           clamp: float | None = None,
           target_rootinv: torch.Tensor | None = None,
           active: torch.Tensor | None = None,
           zmask: torch.Tensor | None = None) -> torch.Tensor:
    """The k=0 bulk anchor on ``H0`` ``(P, n, n)``, ``kBT``/``rho_tot`` ``(P,)``.

    A scalar batch, ``zvec`` ``(P, n)`` and ``target`` ``(P,)``: ``residual`` ``"relative_inverse"``
    ``(target/model - 1)**2`` (optional ``clamp``) or ``"sigma_chi2"`` ``((1/model - 1/target)/sigma)**2``
    (needs ``sigma``).

    A matrix batch, the concentration block: ``zvec`` ``(P, n, n-1)`` (the columns ``z_i``, zero beyond a
    row's ``m`` independent species), ``target`` and ``target_rootinv`` ``R = target^-1/2`` ``(P, n-1, n-1)``
    (the identity beyond ``m``), ``active`` ``(P, n)`` and ``zmask`` ``(P, n-1)`` bool. ``residual``
    ``"s_metric"`` ``||R S_model R - I||_F^2`` over the ``m x m`` block, with
    ``S_model = kBT Z^T H_PP^-1 Z / rho_tot`` on the active principal block of ``H0``. A matrix batch under
    a scalar residual is refused, and so is a scalar batch under ``"s_metric"``."""
    if residual not in _RESIDUAL_FORMS:
        raise ValueError(f"residual={residual!r}, expected one of "
                         f"{_RESIDUAL_FORMS}")
    if residual == "s_metric" or zvec.dim() == 3:
        per_row = _block_residual(H0, kBT, zvec, rho_tot, target,
                                  residual=residual, sigma=sigma,
                                  jitter=jitter, clamp=clamp,
                                  target_rootinv=target_rootinv,
                                  active=active, zmask=zmask)
    else:
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


def _block_residual(H0, kBT, zvec, rho_tot, target, *, residual, sigma,
                    jitter, clamp, target_rootinv, active, zmask):
    """``(P,)`` ``||R S_model R - I||_F^2`` on each row's concentration block (:func:`l_bulk`)."""
    if residual != "s_metric":
        raise ValueError(
            f"zvec has shape {tuple(zvec.shape)}, the columns of an "
            f"(n-1) x (n-1) concentration block, and residual={residual!r} "
            f"compares one scalar S_cc(0); declare 's_metric' for the block")
    missing = [name for name, value in (("target_rootinv", target_rootinv),
                                        ("active", active), ("zmask", zmask))
               if value is None]
    if zvec.dim() != 3 or target.dim() != 3 or missing:
        raise ValueError(
            f"residual='s_metric' compares the concentration block and needs "
            f"a matrix batch, zvec (P, n, n-1) and target (P, n-1, n-1) with "
            f"target_rootinv, active and zmask; got zvec "
            f"{tuple(zvec.shape)}, target {tuple(target.shape)}"
            + (f", and no {missing}" if missing else ""))
    if sigma is not None or clamp is not None:
        raise ValueError(
            "residual='s_metric' takes neither sigma nor clamp: the block is "
            "weighted by its row weights alone")
    n = H0.shape[-1]
    eye = torch.eye(n, dtype=H0.dtype, device=H0.device)
    H = _principal_block(H0, _pairs(active))
    y = torch.linalg.solve(H + jitter * eye, zvec)            # (P, n, n-1)
    model = (kBT / rho_tot)[:, None, None] * (zvec.transpose(-1, -2) @ y)
    m_eye = torch.eye(model.shape[-1], dtype=model.dtype, device=model.device)
    sq = (target_rootinv @ model @ target_rootinv - m_eye).pow(2)
    return torch.where(_pairs(zmask), sq, 0.0).sum(dim=(-2, -1))
