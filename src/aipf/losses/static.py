"""L_S and L_bulk: the two static-structure-factor anchors, model ``H`` against ``k_BT . S(k)^-1``.

``L_S`` acts on shells ``0 < k <= k_fit`` (masked mean over padded shells); ``L_bulk`` is ``k = 0``
through ``S_cc(0) = k_BT . z^T H(0)^-1 z / rho_tot``, a scalar for two channels and the
``(n-1) x (n-1)`` concentration block ``k_BT . Z^T H_PP(0)^-1 Z / rho_tot`` beyond. The ``L_bulk``
residual form is declared: the block is scored in ``S`` space (``"s_metric"``) or in stiffness space,
``S^-1`` (``"s_metric_inverse"``, the block's ``"relative_inverse"``)."""
from __future__ import annotations

import torch

_RESIDUAL_FORMS = ("relative_inverse", "sigma_chi2", "s_metric", "s_metric_inverse")
#: The forms that score the concentration block of a matrix batch; the others score one ``S_cc(0)``.
_BLOCK_FORMS = ("s_metric", "s_metric_inverse")


def sinv_rootinv(Sinv: torch.Tensor) -> torch.Tensor:
    """Symmetric inverse square root ``R = V diag(lambda^-1/2) V^T``, so ``R . Sinv . R = I``;
    eigenvalues floored at 1e-12."""
    evals, evecs = torch.linalg.eigh(Sinv)
    inv_root = evals.clamp(min=1e-12).rsqrt()
    return evecs @ torch.diag_embed(inv_root) @ evecs.transpose(-1, -2)


def spd_root(S: torch.Tensor) -> torch.Tensor:
    """Symmetric square root ``V diag(lambda^1/2) V^T``, so ``root . root = S``; eigenvalues floored at
    0. The counterpart of :func:`sinv_rootinv`, for the ``"s_metric_inverse"`` residual's ``S_t^1/2``."""
    evals, evecs = torch.linalg.eigh(S)
    root = evals.clamp(min=0.0).sqrt()
    return evecs @ torch.diag_embed(root) @ evecs.transpose(-1, -2)


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
           zmask: torch.Tensor | None = None,
           target_root: torch.Tensor | None = None) -> torch.Tensor:
    """The k=0 bulk anchor on ``H0`` ``(P, n, n)``, ``kBT``/``rho_tot`` ``(P,)``.

    A scalar batch, ``zvec`` ``(P, n)`` and ``target`` ``(P,)``: ``residual`` ``"relative_inverse"``
    ``(target/model - 1)**2`` (optional ``clamp``) or ``"sigma_chi2"`` ``((1/model - 1/target)/sigma)**2``
    (needs ``sigma``).

    A matrix batch, the concentration block: ``zvec`` ``(P, n, n-1)`` (the columns ``z_i``, zero beyond a
    row's ``m`` independent species), ``target`` and ``target_rootinv`` ``R = target^-1/2`` ``(P, n-1, n-1)``
    (the identity beyond ``m``), ``active`` ``(P, n)`` and ``zmask`` ``(P, n-1)`` bool. ``residual``
    ``"s_metric"`` ``||R S_model R - I||_F^2`` over the ``m x m`` block, with
    ``S_model = kBT Z^T H_PP^-1 Z / rho_tot`` on the active principal block of ``H0``; or
    ``"s_metric_inverse"`` ``||T S_model^-1 T - I||_F^2`` with ``T = target_root = target^1/2``
    ``(P, n-1, n-1)`` (the identity beyond ``m``), its stiffness-space form, which is
    ``"relative_inverse"`` at ``m = 1`` and smooth where ``H_PP`` is singular along a concentration
    direction (:func:`concentration_stiffness`; ``jitter`` is not used there). A matrix batch under a
    scalar residual is refused, and so is a scalar batch under either block form."""
    if residual not in _RESIDUAL_FORMS:
        raise ValueError(f"residual={residual!r}, expected one of "
                         f"{_RESIDUAL_FORMS}")
    if residual in _BLOCK_FORMS or zvec.dim() == 3:
        per_row = _block_residual(H0, kBT, zvec, rho_tot, target,
                                  residual=residual, sigma=sigma,
                                  jitter=jitter, clamp=clamp,
                                  target_rootinv=target_rootinv,
                                  active=active, zmask=zmask,
                                  target_root=target_root)
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
                    jitter, clamp, target_rootinv, active, zmask, target_root=None):
    """``(P,)`` the block residual of each row's concentration block (:func:`l_bulk`)."""
    if residual not in _BLOCK_FORMS:
        raise ValueError(
            f"zvec has shape {tuple(zvec.shape)}, the columns of an "
            f"(n-1) x (n-1) concentration block, and residual={residual!r} "
            f"compares one scalar S_cc(0); declare 's_metric' or "
            f"'s_metric_inverse' for the block")
    root_name, root = (("target_rootinv", target_rootinv) if residual == "s_metric"
                       else ("target_root", target_root))
    missing = [name for name, value in ((root_name, root),
                                        ("active", active), ("zmask", zmask))
               if value is None]
    if zvec.dim() != 3 or target.dim() != 3 or missing:
        raise ValueError(
            f"residual={residual!r} compares the concentration block and needs "
            f"a matrix batch, zvec (P, n, n-1) and target (P, n-1, n-1) with "
            f"{root_name}, active and zmask; got zvec "
            f"{tuple(zvec.shape)}, target {tuple(target.shape)}"
            + (f", and no {missing}" if missing else ""))
    if sigma is not None or clamp is not None:
        raise ValueError(
            f"residual={residual!r} takes neither sigma nor clamp: the block is "
            "weighted by its row weights alone")
    n = H0.shape[-1]
    eye = torch.eye(n, dtype=H0.dtype, device=H0.device)
    H = _principal_block(H0, _pairs(active))
    if residual == "s_metric":
        y = torch.linalg.solve(H + jitter * eye, zvec)            # (P, n, n-1)
        model = (kBT / rho_tot)[:, None, None] * (zvec.transpose(-1, -2) @ y)
        m_eye = torch.eye(model.shape[-1], dtype=model.dtype, device=model.device)
        sq = (target_rootinv @ model @ target_rootinv - m_eye).pow(2)
    else:
        stiffness = (rho_tot / kBT)[:, None, None] * concentration_stiffness(
            H, zvec, active, zmask)
        m_eye = torch.eye(stiffness.shape[-1], dtype=stiffness.dtype,
                          device=stiffness.device)
        sq = (target_root @ stiffness @ target_root - m_eye).pow(2)
    return torch.where(_pairs(zmask), sq, 0.0).sum(dim=(-2, -1))


def concentration_stiffness(H: torch.Tensor, zvec: torch.Tensor, active: torch.Tensor,
                            zmask: torch.Tensor) -> torch.Tensor:
    """``(P, n-1, n-1)`` ``(Z^T H^-1 Z)^-1`` on each row's ``m`` independent species, the identity beyond,
    without inverting ``H``; ``H`` ``(P, n, n)`` is the identity outside the active block (as
    :func:`_principal_block` leaves it), ``zvec``, ``active``, ``zmask`` as in :func:`l_bulk`.

    On the active block ``Q = [Z  u]``, ``u = 1_P``, which is never in ``span Z``: ``Q`` maps the
    coordinates ``(c, beta)`` to ``sum_i c_i (e_i - x_i 1) + beta 1``, which is ``c_i - s + beta`` on an
    independent species ``i`` and ``beta - s`` on the dependent one (``s = sum_i c_i x_i``), and that is
    zero only for ``c = 0``, ``beta = 0``, whatever the fractions; ``det Q = +-1``. With
    ``A = Q^-1 H Q^-T``, ``A^-1 = Q^T H^-1 Q``, whose ``ZZ`` block is ``Z^T H^-1 Z``; so its inverse is the
    Schur complement ``A_ZZ - A_Zu A_uu^-1 A_uZ``, a polynomial in ``H`` over the scalar ``A_uu = w^T H w``
    (``w`` the last row of ``Q^-1``, the row's fractions when they sum to one on ``P``), the total-number
    stiffness. It is smooth wherever ``A_uu != 0``, through any singularity of ``H`` along a concentration
    direction. Off the active block each column beyond ``m`` is the unit vector of one absent channel,
    so ``Q`` is invertible on every row and those entries come out as the identity."""
    P, n, _ = zvec.shape
    if not torch.equal(zmask.sum(-1), active.sum(-1) - 1):
        raise ValueError(
            "zmask marks m = (present species) - 1 independent columns on every "
            "row; it does not here")
    with torch.no_grad():
        dtype = H.dtype
        absent = ~active
        # the columns beyond m, in order, take the absent channels' unit vectors, in channel order
        order = torch.argsort(zmask.to(torch.int8), dim=-1, stable=True)
        rank = (torch.cumsum(absent.to(torch.int64), dim=-1) - 1).clamp(min=0, max=n - 2)
        col = torch.gather(order, -1, rank)                          # (P, n)
        pad = (torch.nn.functional.one_hot(col, n - 1).to(dtype)
               * absent.to(dtype).unsqueeze(-1))                     # (P, n, n-1)
        Q = torch.cat([torch.where(zmask.unsqueeze(-2), zvec.to(dtype), pad),
                       active.to(dtype).unsqueeze(-1)], dim=-1)      # (P, n, n)
        Q_inv = torch.linalg.inv(Q)
    A = Q_inv @ H @ Q_inv.transpose(-1, -2)
    A_zz, A_zu = A[..., :-1, :-1], A[..., :-1, -1:]
    A_uz, A_uu = A[..., -1:, :-1], A[..., -1:, -1:]
    return A_zz - (A_zu @ A_uz) / A_uu
