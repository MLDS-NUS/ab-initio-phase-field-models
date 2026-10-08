"""The conserved noise of the semi-implicit scheme: draw, contract with ``L L^T = M``, take the coloured
spectral divergence; and ``m_stab="max"``'s frozen mobility.
``zeta_{i,d}(r) = amp L_ij(rho(r)) w_{j,d}(r)``, ``n_hat_i = G(k) i k_d zeta_hat_{i,d}``, ``n_hat(0) = 0``."""
from __future__ import annotations

import math
from typing import Optional

import torch

from aipf.spectral import DC, MATRIX_LAST, ops_ndim

#: Where the noise amplitude is evaluated: ``"ito"`` at ``rho^n``; the others at a predictor
#: advanced by this fraction of the full step, noise included.
NOISE_EVAL_FRAC = {"midpoint": 0.5, "kinetic": 1.0}
NOISE_EVAL_MODES = ("ito",) + tuple(NOISE_EVAL_FRAC)


def check_noise_eval(noise_eval) -> str:
    """``noise_eval`` as declared, or a refusal naming the admissible values."""
    if noise_eval not in NOISE_EVAL_MODES:
        raise ValueError(
            f"noise_eval must be one of {list(NOISE_EVAL_MODES)}, got "
            f"{noise_eval!r}")
    return noise_eval


def cell_mobilities(model, rho: torch.Tensor, T: torch.Tensor,
                    ndim: int = 3) -> torch.Tensor:
    """``M`` at every cell of a single field ``(1, n, *grid)`` of ``ndim`` declared axes, as ``(N, n, n)``
    in C order."""
    n = rho.shape[1]
    M = model.mobility(rho, T)                                   # (1,n,n,*grid)
    return M.permute(*MATRIX_LAST[ndim]).reshape(-1, n, n)


def max_norm_mobility(model, rho: torch.Tensor, T: torch.Tensor,
                      ndim: int = 3) -> torch.Tensor:
    """The one cell's whole ``M`` whose largest eigenvalue is the largest in the field (not an entrywise max)."""
    cells = cell_mobilities(model, rho, T, ndim)
    lam_max = torch.linalg.eigvalsh(cells)[:, -1]
    return cells[torch.argmax(lam_max)]


def draw_w(grid, n: int, generator: Optional[torch.Generator], device,
           dtype) -> torch.Tensor:
    """iid ``N(0, 1)``, shape ``(1, n, ndim, *grid)`` = (batch, species, direction, cell),
    ``ndim = len(grid)``: ``(1, n, 3, Gx, Gy, Gz)`` in three dimensions."""
    return torch.randn((1, n, len(grid), *grid), generator=generator, device=device,
                       dtype=dtype)


#: ``L_ij w_jd`` per cell, per number of spatial axes.
_ZETA_EINSUM = {3: "xyzij,bjdxyz->bidxyz", 2: "xyij,bjdxy->bidxy"}


def zeta_from_w(model, rho_at: torch.Tensor, T: torch.Tensor, amp: float,
                w: torch.Tensor, ndim: int = 3) -> torch.Tensor:
    """``amp L(rho_at) w`` with ``L L^T = M(rho_at) + 1e-12 I``; single field only, ``ndim`` declared axes."""
    if rho_at.shape[0] != 1:
        raise ValueError(f"the noise is single-field; got batch {rho_at.shape[0]}")
    n = rho_at.shape[1]
    grid = tuple(int(g) for g in rho_at.shape[-ndim:])
    M_cells = cell_mobilities(model, rho_at, T, ndim)
    eye = torch.eye(n, device=M_cells.device, dtype=M_cells.dtype)
    L_cells = torch.linalg.cholesky(M_cells + 1e-12 * eye)
    L_grid = L_cells.view(*grid, n, n)
    return amp * torch.einsum(_ZETA_EINSUM[ndim], L_grid, w)


def noise_div_hat(ops, zeta: torch.Tensor, kx, ky, kz,
                  filt: Optional[torch.Tensor]) -> torch.Tensor:
    """``G(k) (i k . zeta_hat)`` in the state's convention (``rfft / N``), zero at ``k = 0``.
    ``kz`` is ``None`` for a two-dimensional ``ops``."""
    ndim = ops_ndim(ops)
    ks = (kx, ky, kz)[:ndim]
    N = math.prod(ops.grid)
    z_hat = [ops.rfft(zeta[:, :, d]) / N for d in range(ndim)]
    n_hat = ops.div_hat(*z_hat, *ks)
    if filt is not None:
        n_hat = n_hat * filt
    n_hat[DC[ndim]] = 0
    return n_hat
