"""The paired structure-factor FDT gate: roll the noisy scheme from a homogeneous state above the critical
temperature and compare the stationary spectrum with ``S_pred = G(k)^2 eps kBT H(k)^-1``,
``H = Hess f_loc(rho_bar) + W_hat(k)``, per mode. Tier 1 (small ``eps``): band means in a window. Tier 2:
every noise convention, one seed, distinct paths and a bounded spread. ``precision="fp64"`` runs the gate on
the model cast to float64 (:mod:`aipf.solve.precision`); the default is the float32 gate as published."""
from __future__ import annotations

import numpy as np
import torch

from aipf.solve.noise import declared_noise
from aipf.solve.precision import (PRECISIONS, cast_pair, check_precision,
                                  model_float_dtypes)
from aipf.solve.trust_domain import in_domain
from aipf.spectral import model_ndim, refuse_two_dimensions
from aipf.system import System

from .imex import local_hessian, rollout_imex
from .observables import (measure_S_modes, mode_wavevectors, s_pred_from_H,
                          trace_and_cc, weighted_mean)
from .spinodal import noise_for, trust_domain


def homogeneous_state(rho_bar, grid, device,
                      dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """Only ``k = 0`` set, to ``rho_bar``; ``dtype`` ``complex64`` or ``complex128``."""
    refuse_two_dimensions(len(grid), "the FDT gate")
    Gx, Gy, Gz = (int(g) for g in grid)
    h = torch.zeros((1, len(rho_bar), Gx, Gy, Gz // 2 + 1),
                    dtype=dtype, device=device)
    h[0, :, 0, 0, 0] = torch.as_tensor(np.asarray(rho_bar),
                                       dtype=dtype, device=device)
    return h


def hessian_of_k(model, rho_bar, kBT: float, kmag) -> np.ndarray:
    """``H(k) = Hess f_loc(rho_bar) + W_hat(k)`` per mode, float64 ``(M, n, n)``; evaluated in float32, or
    in float64 for a float64 model."""
    device = next(model.parameters()).device
    real = (torch.float64 if torch.float64 in model_float_dtypes(model)
            else torch.float32)
    rb = torch.as_tensor(np.asarray(rho_bar), dtype=real,
                         device=device)
    Hb = local_hessian(model, rb, kBT)
    with torch.no_grad():
        k = torch.as_tensor(np.asarray(kmag), dtype=real,
                            device=device)
        return (model.kernel.w_hat(k) + Hb).double().cpu().numpy()


def run_one(system: System, model, decl: dict, *, box, rho_bar, grid, T: float,
            eps: float, noise_eval: str, t_end: float, dt: float,
            save_dt: float, burn_in: float, k_band: float, seed: int,
            device: str, v_ext=None, kbt_field=None,
            precision: str = "fp32") -> dict:
    """One noisy rollout with the projection off (``"floor"`` at 0) and its per-mode ratios to ``S_pred``.
    ``v_ext``/``kbt_field`` pass through to the scheme; ``S_pred`` assumes them uniform (``kbt_field = kB T``).
    ``precision="fp64"`` casts the model in place (``model.double()``) and builds the state and box in
    float64."""
    refuse_two_dimensions(model_ndim(model), "the FDT gate")
    refuse_two_dimensions(len(grid), "the FDT gate")
    real, cplx = PRECISIONS[check_precision(precision)]
    h0 = homogeneous_state(rho_bar, grid, device, cplx)
    model, h0 = cast_pair(model, h0, precision)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    noise = dict(noise_for(system, decl, T, eps), noise_eval=noise_eval)
    kB = float(system.constants["kB"])
    traj = rollout_imex(
        model, h0,
        torch.as_tensor(box, dtype=real), T, dt,
        int(round(t_end / dt)), kB=kB,
        m_stab=declared_noise(system)["m_stab"], state_proj="floor",
        state_clamp=0.0, domain=trust_domain(system), mass_restore="shift",
        clamp_rho=decl["solver"]["clamp_rho"], noise=noise,
        save_every=max(1, int(round(save_dt / dt))), generator=gen,
        v_ext=v_ext, kbt_field=kbt_field)
    frames = traj.numpy()
    frames = frames[int(round(burn_in * len(frames))):]
    V = float(np.prod(box))
    kmag, mult = mode_wavevectors(box, grid)
    S = measure_S_modes(frames, V)
    kf, wf = kmag.ravel(), mult.ravel()
    band = (kf > 1e-12) & (kf <= k_band)
    kb, wb = kf[band], wf[band]
    n = S.shape[-1]
    S_meas = S.reshape(-1, n, n)[band]
    H = hessian_of_k(model, rho_bar, kB * T, kb)
    S_pred = s_pred_from_H(H, kB * T * eps, kb, noise["sigma_noise"])
    tr_m, cc_m = trace_and_cc(S_meas)
    tr_p, cc_p = trace_and_cc(S_pred)
    fields = [np.fft.irfftn(f * np.prod(grid), s=grid, axes=(-3, -2, -1))
              for f in frames]
    rho_min = min(float(r.min()) for r in fields)
    domain = trust_domain(system)
    out = sum(int((~in_domain(torch.as_tensor(np.moveaxis(r, 0, -1)),
                                domain)).sum()) for r in fields)
    return {"T": float(T), "eps": float(eps), "noise_eval": noise_eval,
            "k_mode": kb, "w_mode": wb, "r_tr_mode": tr_m / tr_p,
            "r_cc_mode": cc_m / cc_p, "n_frames": int(len(frames)),
            "band_mean_tr": weighted_mean(tr_m / tr_p, wb),
            "band_mean_cc": weighted_mean(cc_m / cc_p, wb),
            "rho_min": rho_min,
            "out_of_domain_fraction": out / (len(fields) * int(np.prod(grid)))}
