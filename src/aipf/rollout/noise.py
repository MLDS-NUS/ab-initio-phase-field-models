"""The conserved noise of the semi-implicit scheme: draw, contract with ``L L^T = M``, take the coloured
spectral divergence; and ``m_stab="max"``'s frozen mobility.
``zeta_{i,d}(r) = amp L_ij(rho(r)) w_{j,d}(r)``, ``n_hat_i = G(k) i k_d zeta_hat_{i,d}``, ``n_hat(0) = 0``."""
from __future__ import annotations

from typing import Optional

import torch

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


def cell_mobilities(model, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """``M`` at every cell of a single field ``(1, n, Gx, Gy, Gz)``, as ``(N, n, n)`` in C order."""
    n = rho.shape[1]
    M = model.mobility(rho, T)                                   # (1,n,n,Gx,Gy,Gz)
    return M.permute(0, 3, 4, 5, 1, 2).reshape(-1, n, n)


def max_norm_mobility(model, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """The one cell's whole ``M`` whose largest eigenvalue is the largest in the field (not an entrywise max)."""
    cells = cell_mobilities(model, rho, T)
    lam_max = torch.linalg.eigvalsh(cells)[:, -1]
    return cells[torch.argmax(lam_max)]


def draw_w(grid, n: int, generator: Optional[torch.Generator], device,
           dtype) -> torch.Tensor:
    """iid ``N(0, 1)``, shape ``(1, n, 3, Gx, Gy, Gz)`` = (batch, species, direction, cell)."""
    return torch.randn((1, n, 3, *grid), generator=generator, device=device,
                       dtype=dtype)


def zeta_from_w(model, rho_at: torch.Tensor, T: torch.Tensor, amp: float,
                w: torch.Tensor) -> torch.Tensor:
    """``amp L(rho_at) w`` with ``L L^T = M(rho_at) + 1e-12 I``; single field only."""
    if rho_at.shape[0] != 1:
        raise ValueError(f"the noise is single-field; got batch {rho_at.shape[0]}")
    n = rho_at.shape[1]
    grid = tuple(int(g) for g in rho_at.shape[-3:])
    M_cells = cell_mobilities(model, rho_at, T)
    eye = torch.eye(n, device=M_cells.device, dtype=M_cells.dtype)
    L_cells = torch.linalg.cholesky(M_cells + 1e-12 * eye)
    L_grid = L_cells.view(*grid, n, n)
    return amp * torch.einsum("xyzij,bjdxyz->bidxyz", L_grid, w)


def noise_div_hat(ops, zeta: torch.Tensor, kx, ky, kz,
                  filt: Optional[torch.Tensor]) -> torch.Tensor:
    """``G(k) (i k . zeta_hat)`` in the state's convention (``rfft / N``), zero at ``k = 0``."""
    N = ops.grid[0] * ops.grid[1] * ops.grid[2]
    zx_hat = ops.rfft(zeta[:, :, 0]) / N
    zy_hat = ops.rfft(zeta[:, :, 1]) / N
    zz_hat = ops.rfft(zeta[:, :, 2]) / N
    n_hat = ops.div_hat(zx_hat, zy_hat, zz_hat, kx, ky, kz)
    if filt is not None:
        n_hat = n_hat * filt
    n_hat[..., 0, 0, 0] = 0
    return n_hat
