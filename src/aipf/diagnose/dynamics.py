"""Growth rates, the static structure factor, and rollout drift (the ``l_dyn`` residual on a stored
trajectory)."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from aipf.losses.dyn import l_dyn
from aipf.spectral import SpectralOps, model_ndim, ops_ndim, refuse_two_dimensions

WHatFn = Callable[[torch.Tensor], torch.Tensor]


def growth_rate(hessian: torch.Tensor, mobility: torch.Tensor,
                 k: torch.Tensor, *, w_hat: Optional[WHatFn] = None
                 ) -> torch.Tensor:
    r"""``lambda(k) = -k^2 * max Re eig[M @ (H + w_hat(k))]``, the linear Model-B dispersion about a
    uniform state.

    ``hessian``, ``mobility``: ``(n_species, n_species)``; ``k``: any shape, ``|k|``; ``w_hat`` optional
    ``k -> (*k.shape, n, n)``. Positive ``lambda`` = growing mode. Returns shape ``k.shape``."""
    hessian = torch.as_tensor(hessian, dtype=torch.float64)
    mobility = torch.as_tensor(mobility, dtype=torch.float64)
    k = torch.as_tensor(k, dtype=torch.float64)
    n = hessian.shape[-1]
    if hessian.shape[-2:] != (n, n):
        raise ValueError(f"hessian must be square, got {tuple(hessian.shape)}")
    if mobility.shape[-2:] != (n, n):
        raise ValueError(
            f"mobility shape {tuple(mobility.shape)} does not match "
            f"hessian's n_species={n}"
        )
    target_shape = tuple(k.shape) + (n, n)
    if w_hat is not None:
        Wk = w_hat(k).to(torch.float64)
        if tuple(Wk.shape) != target_shape:
            raise ValueError(
                f"w_hat(k) returned shape {tuple(Wk.shape)}, expected "
                f"{target_shape}"
            )
        A = hessian.expand(target_shape) + Wk
    else:
        A = hessian.expand(target_shape).clone()
    M = mobility.expand(target_shape)
    MA = torch.matmul(M, A)
    lam_max = torch.linalg.eigvals(MA).real.max(dim=-1).values
    return -(k * k) * lam_max


def most_unstable_k(hessian: torch.Tensor, mobility: torch.Tensor,
                     k_grid: torch.Tensor, *, w_hat: Optional[WHatFn] = None
                     ) -> Tuple[float, float]:
    """``(k*, lambda(k*))`` by grid search over the caller's ``k_grid``."""
    lam = growth_rate(hessian, mobility, torch.as_tensor(k_grid), w_hat=w_hat)
    i = int(torch.argmax(lam))
    k_grid_t = torch.as_tensor(k_grid, dtype=torch.float64)
    return float(k_grid_t[i]), float(lam[i])


def structure_factor(rho_hat: torch.Tensor, ops: SpectralOps,
                      boxes: torch.Tensor, n_bins: int, *,
                      k_max: Optional[float] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``S = <sum_species MULT * |rho_hat|^2>`` in ``n_bins`` equal shells of ``[0, k_max]``.

    ``rho_hat`` ``(B, n_species, Gx, Gy, Gzr)``; returns ``(k_centers (n_bins,), S (B, n_bins))``,
    NaN in empty shells."""
    refuse_two_dimensions(ops_ndim(ops), "structure_factor")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    kmag = ops.k2(boxes).sqrt()                                   # (B,1,Gx,Gy,Gzr)
    power = (rho_hat.abs() ** 2 * ops.MULT).sum(dim=1, keepdim=True)
    B = rho_hat.shape[0]
    k_top = float(kmag.max()) if k_max is None else float(k_max)
    edges = torch.linspace(0.0, k_top, n_bins + 1, dtype=kmag.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kflat = kmag.reshape(B, -1)
    pflat = power.reshape(B, -1)
    S = torch.full((B, n_bins), float("nan"), dtype=power.dtype)
    for b in range(B):
        idx = torch.bucketize(kflat[b], edges[1:-1])
        for j in range(n_bins):
            sel = idx == j
            if bool(sel.any()):
                S[b, j] = pflat[b, sel].mean()
    return centers, S


def finite_difference_rate(rho_hat_traj: torch.Tensor, dt: float
                            ) -> torch.Tensor:
    """``(rho_hat[t+1] - rho_hat[t]) / dt``; ``(T_frames, B, n_species, Gx, Gy, Gzr)`` in,
    ``T_frames - 1`` out."""
    if rho_hat_traj.shape[0] < 2:
        raise ValueError(
            f"rho_hat_traj has {rho_hat_traj.shape[0]} frame(s); need at "
            f"least 2 to form a finite-difference rate"
        )
    return (rho_hat_traj[1:] - rho_hat_traj[:-1]) / dt


def rollout_drift(model, rho_hat_traj: torch.Tensor, boxes_traj: torch.Tensor,
                   T_traj: torch.Tensor, dt: float, ops: SpectralOps, *,
                   alpha: float, k_max: float) -> torch.Tensor:
    """:func:`aipf.losses.dyn.l_dyn` between the model's rate and the finite-difference rate of a
    stored trajectory.

    Trajectories ``(T_frames, B, ...)`` with per-frame boxes and ``T``; ``alpha``, ``k_max`` required.
    Returns ``(T_frames - 1,)``."""
    refuse_two_dimensions(model_ndim(model), "rollout_drift")
    actual = finite_difference_rate(rho_hat_traj, dt)
    n_frames = rho_hat_traj.shape[0]
    per_frame = []
    for t in range(n_frames - 1):
        boxes = boxes_traj[t]
        predicted = model.forward(rho_hat_traj[t], boxes, T_traj[t])
        k2 = ops.k2(boxes)
        band = ops.band_mask(boxes, k_max)
        residual = actual[t] - predicted
        per_frame.append(l_dyn(residual, k2, ops.MULT, band, alpha=alpha))
    return torch.stack(per_frame)
