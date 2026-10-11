"""The semi-implicit (IMEX) scheme the published rollouts ran, deterministic and with conserved noise.
Frozen operator ``A = I + dt k^2 M_s H(k)``, ``H = Hess f_loc(rho_bar) + W_hat(k)``; per step
``rho^{n+1} = A^-1 (rho^n + dt F(rho^n) + dt k^2 M_s H rho^n + dt n_hat)``, then hermitianise, then project.
Reads the model's ``kernel.w_hat`` and ``f_local`` (a pair-kernel functional); other forms are refused.
A model with a ``stabilizer_mobility(rho, T)`` method sets ``M_s`` itself, and ``m_stab`` is then left
undeclared (:func:`stabilizer_mobility`). A two-dimensional model (``model.ops.ndim == 2``) runs the same
scheme on ``(1, n, Gx, Gyr)`` in a ``(2,)`` box; its noise needs ``depth``
(:func:`aipf.solve.noise.check_depth`). Precision is the state's and the model's
(:func:`aipf.solve.precision.working_dtypes`): a ``complex128`` state with a float64 model builds every
operator, field and noise draw in float64 / complex128; a ``complex64`` state with a float32 model is the
float32 scheme the published rollouts ran; a mixed pair is refused.
Opt-in, off by default: ``m_stab="bound"`` freezes ``M_s`` at an upper bound of every cell's mobility
(:func:`bound_ratio`), and ``refresh_every`` re-freezes ``M_s`` (any source) from the current state."""
from __future__ import annotations

import logging
import math
import numbers
from contextlib import contextmanager, nullcontext
from typing import Iterator, List, Optional

import torch

from aipf.solve import hermitianize, project_state
from aipf.solve.declare import UNDECLARED
from aipf.solve.noise import build_noise_filter, check_depth, check_m_stab
from aipf.solve.precision import working_dtypes
from aipf.solve.projection import project_state_uniform_shift
from aipf.solve.trust_domain import TrustDomain
from aipf.spectral import (CHANNEL_LAST, CHANNEL_SECOND, SpectralOps,
                           exact_mode_indices, half_spectrum_grid, k_squared,
                           make_ops, model_ndim)

from .noise import (NOISE_EVAL_FRAC, cell_mobilities, check_noise_eval, draw_w,
                    max_norm_mobility, noise_div_hat, zeta_from_w)

#: How the state projection restores mass: ``"shift"`` writes ``k=0`` back (the published rollouts),
#: ``"headroom"`` is :func:`aipf.solve.project_state` (never below ``lo``).
MASS_RESTORES = ("shift", "headroom")

_LOG = logging.getLogger(__name__)

#: Relative tolerance of the stabiliser's symmetry and positive semi-definiteness checks.
_STABILIZER_RTOL = 1e-6

#: The opt-in ``m_stab`` of :func:`rollout_imex` beside the declared ``"mean"`` / ``"max"``: ``M_s`` an upper
#: bound of every cell's mobility. A rollout option, not a ``Noise.m_stab`` declaration.
BOUND = "bound"

#: ``M(rho_bar)``'s eigenvalues are floored at this times the largest before it is whitened
#: (:func:`bound_ratio`).
BOUND_RIDGE = 1e-12


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


def _stabilizer_hook(model, m_stab):
    """The model's ``stabilizer_mobility``, or ``None`` with ``m_stab`` checked; a declared ``m_stab``
    beside the hook is refused, since the two would each set ``M_s``."""
    hook = getattr(model, "stabilizer_mobility", None)
    if hook is None:
        if not (isinstance(m_stab, str) and m_stab == BOUND):
            check_m_stab(m_stab)
        return None
    if m_stab is not UNDECLARED:
        raise ValueError(
            f"m_stab={m_stab!r} is declared and {type(model).__name__} has a "
            f"stabilizer_mobility, which sets M_s itself; leave m_stab undeclared for "
            f"this model, or roll out one without the method")
    return hook


def stabilizer_mobility(hook, rho: torch.Tensor, T_t: torch.Tensor) -> torch.Tensor:
    """``M_s`` from a model's hook, called once on the real-space field at ``t = 0`` ``(1, n, Gx, Gy, Gz)``
    (``(1, n, Gx, Gy)`` in two dimensions) and ``T_t`` ``(1,)``; refused unless ``(n, n)``, finite,
    symmetric and positive semi-definite to a relative ``_STABILIZER_RTOL``."""
    n = int(rho.shape[1])
    M = hook(rho, T_t)
    if not isinstance(M, torch.Tensor) or tuple(M.shape) != (n, n):
        shape = tuple(M.shape) if isinstance(M, torch.Tensor) else type(M).__name__
        raise ValueError(
            f"stabilizer_mobility returned {shape}, not an ({n}, {n}) tensor: M_s is one "
            f"matrix over the species, frozen for the whole rollout")
    M = M.detach().to(device=rho.device, dtype=rho.dtype)
    if not bool(torch.isfinite(M).all()):
        raise ValueError(f"stabilizer_mobility returned a non-finite M_s: {M.tolist()}")
    tol = _STABILIZER_RTOL * float(M.abs().max())
    if float((M - M.T).abs().max()) > tol:
        raise ValueError(
            f"stabilizer_mobility returned a non-symmetric M_s: {M.tolist()}. A mobility "
            f"is symmetric (Onsager), and the frozen operator is built on that")
    lowest = float(torch.linalg.eigvalsh(M.double())[0])
    if lowest < -tol:
        raise ValueError(
            f"stabilizer_mobility returned an M_s with eigenvalue {lowest!r} < 0: "
            f"{M.tolist()}. A negative mobility makes the implicit operator "
            f"anti-diffusive, which is what it exists to damp")
    return M


def _check_bound_margin(m_stab, bound_margin, hook) -> Optional[float]:
    """``bound_margin`` as a float for ``m_stab="bound"`` (``None``: 1.0), finite and ``>= 1``; refused beside
    any other ``M_s`` source, where it would be ignored."""
    if not (isinstance(m_stab, str) and m_stab == BOUND):
        if bound_margin is not None:
            source = (f"{type(hook.__self__).__name__}.stabilizer_mobility sets M_s"
                      if hook is not None and hasattr(hook, "__self__") else f"m_stab is {m_stab!r}")
            raise ValueError(
                f"bound_margin={bound_margin!r} is a setting of m_stab='bound' only, and {source}; "
                f"leave bound_margin=None")
        return None
    if bound_margin is None:
        return 1.0
    if isinstance(bound_margin, bool) or not isinstance(bound_margin, numbers.Real):
        raise ValueError(f"bound_margin must be a finite number >= 1, got {bound_margin!r}")
    margin = float(bound_margin)
    if not (math.isfinite(margin) and margin >= 1.0):
        raise ValueError(
            f"bound_margin must be a finite number >= 1 (M_s = bound_margin * max(r, 1) * M(rho_bar) "
            f"bounds every cell's mobility by M_s / bound_margin), got {bound_margin!r}")
    return margin


def _check_refresh_every(refresh_every) -> Optional[int]:
    """``refresh_every`` as an int ``>= 1``, or ``None`` (``M_s`` frozen once, at ``t = 0``)."""
    if refresh_every is None:
        return None
    if (isinstance(refresh_every, bool) or not isinstance(refresh_every, numbers.Integral)
            or int(refresh_every) < 1):
        raise ValueError(
            f"refresh_every must be None (M_s frozen once, at t = 0) or an int >= 1 (M_s re-frozen "
            f"from the state every refresh_every steps), got {refresh_every!r}")
    return int(refresh_every)


def _mean_mobility(model, rho_bar: torch.Tensor, T_t: torch.Tensor, n: int,
                   ndim: int) -> torch.Tensor:
    """``m_stab="mean"``'s ``M_s``: ``model.mobility`` at ``rho_bar`` as a one-cell field, ``(n, n)``."""
    if ndim == 2:
        return model.mobility(rho_bar.view(1, n, 1, 1), T_t)[0, :, :, 0, 0]
    return model.mobility(rho_bar.view(1, n, 1, 1, 1), T_t)[0, :, :, 0, 0, 0]


def _floored_eigh(M_bar: torch.Tensor) -> tuple:
    """``(lam, lam_floored, V)`` of the symmetrised ``M_bar`` in float64, ``lam`` ascending and floored at
    :data:`BOUND_RIDGE` times the largest."""
    Mb = M_bar.detach().to(torch.float64)
    lam, V = torch.linalg.eigh(0.5 * (Mb + Mb.T))
    return lam, lam.clamp(min=BOUND_RIDGE * float(lam[-1])), V


def _whitened_ratios(lam_floored: torch.Tensor, V: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    W = (V * lam_floored.rsqrt()) @ V.T
    C = W @ cells.detach().to(device=W.device, dtype=torch.float64) @ W
    return torch.linalg.eigvalsh(0.5 * (C + C.transpose(-1, -2)))[..., -1]


def generalised_ratios(M_bar: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """The largest generalised eigenvalue of each ``cells[..., :, :]`` against ``M_bar`` (``max_x x^T M_cell x /
    x^T M_bar x``), shape ``cells.shape[:-2]``, float64 whatever the inputs' dtype: ``eigh`` of the symmetrised
    ``M_bar``, its eigenvalues floored at :data:`BOUND_RIDGE` times the largest, the whitening
    ``W = V diag(lambda^-1/2) V^T``, then ``eigvalsh`` of the symmetrised ``W M_cell W``. A singular ``M_bar``
    gives a finite ratio against the floored one; an ``M_bar`` with no positive eigenvalue, or a non-finite
    input, gives a non-finite one."""
    _, lam_floored, V = _floored_eigh(M_bar)
    return _whitened_ratios(lam_floored, V, cells)


def _bound_stabiliser(model, rho: torch.Tensor, rho_bar: torch.Tensor, T_t: torch.Tensor, n: int,
                      ndim: int, margin: float) -> tuple:
    """``(M_s, cells, info)`` of ``m_stab="bound"`` on the real field ``rho``. ``M_bar`` is built as ``"mean"``
    builds it; ``ratios`` are the cells' largest generalised eigenvalues against it (float64, C order),
    ``r_raw`` their maximum and ``r_used = max(r_raw, 1)``. ``M_s = margin * r_used * M_bar`` in ``M_bar``'s
    dtype (the product taken in float64), or, when the eigenvalue floor raises any eigenvalue of ``M_bar``
    (``floored``), ``margin * r_used * V diag(lam_floored) V^T``, so the bound holds along ``M_bar``'s null
    directions too. A non-finite ``M_bar``, cell, ``r`` or ``M_s`` is refused, naming which."""
    M_bar = _mean_mobility(model, rho_bar, T_t, n, ndim)
    if not bool(torch.isfinite(M_bar).all()):
        raise ValueError(
            f"m_stab='bound': M(rho_bar), the mobility at the mean density {rho_bar.tolist()}, is "
            f"non-finite: {M_bar.tolist()}")
    cells = cell_mobilities(model, rho, T_t, ndim)
    finite = torch.isfinite(cells).flatten(1).all(-1)
    if not bool(finite.all()):
        raise ValueError(
            f"m_stab='bound': the mobility is non-finite at {int((~finite).sum())} of "
            f"{int(finite.numel())} cells (a non-finite state, or a mobility undefined there); "
            f"r cannot bound it")
    lam, lam_floored, V = _floored_eigh(M_bar)
    if not float(lam[-1]) > 0.0:
        raise ValueError(
            f"m_stab='bound': M(rho_bar) has no positive eigenvalue (largest {float(lam[-1])!r}): "
            f"{M_bar.tolist()}; no multiple of it bounds the cells' mobility")
    ratios = _whitened_ratios(lam_floored, V, cells)
    if not bool(torch.isfinite(ratios).all()):
        raise ValueError(
            f"m_stab='bound': r, the cells' largest generalised eigenvalue against M(rho_bar), is "
            f"non-finite (max {float(ratios.max())!r})")
    i = int(torch.argmax(ratios))
    r_raw = float(ratios[i])
    r_used = max(r_raw, 1.0)
    floored = not torch.equal(lam_floored, lam)
    if floored:
        M_f = (V * lam_floored) @ V.T
        M_s = (margin * r_used * (0.5 * (M_f + M_f.T))).to(M_bar.dtype)
    else:
        M_s = (margin * r_used * M_bar.to(torch.float64)).to(M_bar.dtype)
    if not bool(torch.isfinite(M_s).all()):
        raise ValueError(
            f"m_stab='bound': M_s = {margin!r} * {r_used!r} * M(rho_bar) is non-finite in "
            f"{M_bar.dtype}")
    cond = float(lam[-1] / lam[0]) if float(lam[0]) > 0.0 else float("inf")
    info = {"r_raw": r_raw, "r_used": r_used, "index": i, "ratios": ratios, "M_bar": M_bar,
            "cond": cond, "floored": floored}
    return M_s, cells, info


def _cell_index(i: int, grid) -> tuple:
    """The grid index ``(ix, iy[, iz])`` of the C-order flat cell ``i``."""
    out = []
    for g in reversed(tuple(int(g) for g in grid)):
        i, r = divmod(i, g)
        out.append(r)
    return tuple(reversed(out))


def _full_grid(rho_hat0: torch.Tensor, what: str, ndim: int = 3, declared=None) -> tuple:
    if int(rho_hat0.shape[0]) != 1:
        raise ValueError(f"{what} is single-field; got batch {int(rho_hat0.shape[0])}")
    if ndim == 2:
        return half_spectrum_grid(rho_hat0.shape[-2:], declared)
    Gx, Gy, Gzr = (int(s) for s in rho_hat0.shape[-3:])
    return Gx, Gy, 2 * (Gzr - 1)


@contextmanager
def pointwise_fields(model, rho_hat0: torch.Tensor, kbt_field=None,
                     v_ext=None, dtype: torch.dtype = torch.float32
                     ) -> Iterator[Optional[torch.Tensor]]:
    """Per-cell ``kBT`` (``(Gx, Gy, Gz)``, energy) replaces the pointwise ``mu``'s scalar one, then a static
    ``v_ext`` (``(n, Gx, Gy, Gz)``, energy) is added to it; full-grid calls only. Yields the ``kBT`` field.
    Two-dimensional model: ``(Gx, Gy)`` and ``(n, Gx, Gy)``. Both are built in ``dtype``, the run's real
    dtype."""
    device, n = rho_hat0.device, int(rho_hat0.shape[1])
    ndim = model_ndim(model)
    declared = getattr(getattr(model, "ops", None), "grid", None)
    kbt_t = V = None
    if kbt_field is not None:
        grid = _full_grid(rho_hat0, "kbt_field", ndim, declared)
        kbt_t = torch.as_tensor(kbt_field, dtype=dtype, device=device)
        if tuple(kbt_t.shape) != grid:
            raise ValueError(f"kbt_field must have shape {grid} (one kBT per cell, no "
                             f"species axis), got {tuple(kbt_t.shape)}")
        if not bool(torch.isfinite(kbt_t).all()) or float(kbt_t.min()) <= 0.0:
            raise ValueError(f"kbt_field must be finite and positive everywhere; got "
                             f"min {float(kbt_t.min())}, max {float(kbt_t.max())}")
    if v_ext is not None:
        grid = _full_grid(rho_hat0, "v_ext", ndim, declared)
        V = torch.as_tensor(v_ext, dtype=dtype, device=device)
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
                     _V=V.permute(*((1, 2, 3, 0) if ndim == 3 else (1, 2, 0))).reshape(-1, n)):
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


def _mean_density(rho_hat: torch.Tensor, ndim: int) -> torch.Tensor:
    """``rho_bar``: the state's ``k = 0`` mode, real, ``(n,)``."""
    if ndim == 2:
        return rho_hat[0, :, 0, 0].real.clone()
    return rho_hat[0, :, 0, 0, 0].real.clone()


def _scheme_ops(model, rho_hat: torch.Tensor, n: int, ndim: int, real: torch.dtype, device):
    """The scheme's own operator set on the state's grid: float32 as it always was, else in ``real`` with
    exact integer mode indices."""
    ops_kw = {} if real == torch.float32 else {"dtype": real}
    if ndim == 2:
        return make_ops(half_spectrum_grid(rho_hat.shape[-2:], model.ops.grid), n,
                        nyquist_mask=model.ops.nyquist_mask, **ops_kw).to(device)
    Gx, Gy, Gzr = (int(s) for s in rho_hat.shape[-3:])
    return SpectralOps((Gx, Gy, 2 * (Gzr - 1)), n,
                       nyquist_mask=model.ops.nyquist_mask, **ops_kw).to(device)


@torch.no_grad()
def bound_ratio(model, rho_hat: torch.Tensor, T, *, clamp_rho: Optional[float] = None) -> dict:
    """What ``m_stab="bound"`` reads off the state ``rho_hat`` ``(1, n, Gx, Gy, Gzr)`` (``(1, n, Gx, Gyr)`` in
    two dimensions) at temperature ``T``, without rolling out: for a caller to check how far a ``"mean"`` run's
    frozen ``M(rho_bar)`` is from bounding the field's mobility.

    ``M_bar = model.mobility`` at the state's ``k = 0`` mode, exactly as ``m_stab="mean"`` builds it; the cells'
    ``model.mobility`` on the real field; per cell the largest generalised eigenvalue against ``M_bar``
    (:func:`generalised_ratios`). Returns a dict: ``r_raw`` the largest (float), ``r_used = max(r_raw, 1)``,
    ``cell`` its grid index, ``ratios`` float64 on the grid, ``M_bar`` ``(n, n)`` in the run's dtype, ``cond``
    its condition number (``inf`` if not positive definite), ``floored`` whether the eigenvalue floor raised
    any of its eigenvalues, and ``M_s`` the ``"bound"`` stabiliser at ``bound_margin = 1``. A stiff
    frozen-coefficient step over-corrects a decaying mode where ``r_raw > 2`` under ``"mean"``.
    ``clamp_rho`` is :func:`rollout_imex`'s mobility floor; a run with a ``kbt_field`` reads ``T`` as the
    field's mean ``kBT / kB``. Precision follows the state and the model, as in :func:`rollout_imex`; a
    non-finite ``M_bar``, cell or ``r`` is refused."""
    if int(rho_hat.shape[0]) != 1:
        raise ValueError(f"bound_ratio is single-field; got batch {int(rho_hat.shape[0])}")
    ndim = model_ndim(model)
    real, _ = working_dtypes(model, rho_hat, "bound_ratio")
    fp64 = real == torch.float64
    device, n = rho_hat.device, int(rho_hat.shape[1])
    t_kw = {"dtype": real} if fp64 else {}
    T_t = torch.tensor([float(T)], device=device, **t_kw)
    rho_bar = _mean_density(rho_hat, ndim)
    exact = exact_mode_indices(model) if fp64 else nullcontext()
    with exact, pointwise_guard(model, clamp_rho):
        ops = _scheme_ops(model, rho_hat, n, ndim, real, device)
        rho = ops.irfft(rho_hat * math.prod(ops.grid))
        M_s, _, info = _bound_stabiliser(model, rho, rho_bar, T_t, n, ndim, 1.0)
    grid = tuple(int(g) for g in ops.grid)
    return {"r_raw": info["r_raw"], "r_used": info["r_used"],
            "cell": _cell_index(info["index"], grid), "ratios": info["ratios"].reshape(grid),
            "M_bar": info["M_bar"], "cond": info["cond"], "floored": info["floored"], "M_s": M_s}


def _identity(n: int, device, real: torch.dtype) -> torch.Tensor:
    """``I`` ``(n, n)``: the float32 path's ``torch.eye`` exactly as it always was, else in ``real``."""
    if real == torch.float32:
        return torch.eye(n, device=device)
    return torch.eye(n, device=device, dtype=real)


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
                 v_ext=None, kbt_field=None,
                 depth: Optional[float] = None,
                 bound_margin: Optional[float] = None,
                 refresh_every: Optional[int] = None,
                 stabiliser_log: Optional[list] = None) -> torch.Tensor:
    """Advance one field ``rho_hat0`` ``(1, n, Gx, Gy, Gzr)`` in box ``(3,)`` at scalar ``T``; returns the
    saved states ``(n_saved, n, Gx, Gy, Gzr)`` on the CPU. ``noise=None`` is deterministic; otherwise a
    dict with ``kBT_noise``, ``noise_scale``, ``noise_mode``, ``sigma_noise``, ``noise_eval``,
    ``predictor_floor``. ``v_ext``/``kbt_field``: a static external potential and a per-cell ``kBT``
    (:func:`pointwise_fields`). Every other knob is declared, except ``m_stab`` for a model with a
    ``stabilizer_mobility``, which must leave it undeclared. A two-dimensional model takes
    ``(1, n, Gx, Gyr)`` in a ``(2,)`` box, and its noise needs ``depth``: ``dV = dA * depth``.
    Precision follows the state and the model (module docstring); a float64 run casts ``box``, ``v_ext`` and
    ``kbt_field`` to float64 (give ``box`` in float64 when its lengths are not float32 numbers), reads the
    model's operator sets with exact integer mode indices for the call (:func:`aipf.spectral.exact_mode_indices`,
    restored after), and its noise draws are float64, a different random stream from a float32 run of the
    same seed.

    Opt-in, each off by default (the call is then exactly the one above): ``m_stab="bound"`` freezes
    ``M_s = bound_margin * max(r, 1) * M(rho_bar)``, ``M(rho_bar)`` as ``"mean"`` builds it and ``r`` the
    largest generalised eigenvalue of any cell's mobility against it (:func:`bound_ratio`; ``M(rho_bar)``
    with its eigenvalues floored when the floor acts). It is a bound only if ``model.mobility(rho)`` bounds,
    in the Loewner order, the mobility the model's forward pass applies. ``bound_margin`` (``None``: 1.0,
    else finite ``>= 1``) is declared with ``"bound"`` only, and a model with a ``stabilizer_mobility``
    refuses ``"bound"`` as it refuses any declared ``m_stab``. ``refresh_every=N`` re-freezes ``M_s`` from the
    current state, from the same source, before steps ``N + 1, 2N + 1, ...``, and rebuilds every operator from
    it as a new call would: the run equals ``ceil(n_steps / N)`` calls of ``N`` steps chained on the last
    state, bit for bit. ``stabiliser_log``, a list, receives one dict per freeze (``step``, ``source``, and for
    ``"bound"`` ``r_raw``, ``r_used``, ``margin``, ``cell``, ``ratio_min`` / ``_mean`` / ``_max`` against
    ``M(rho_bar)``, ``cond``, ``floored`` and ``ratio_after`` against ``M_s``)."""
    _require_pair_kernel(model)
    hook = _stabilizer_hook(model, m_stab)
    bound_margin = _check_bound_margin(m_stab, bound_margin, hook)
    refresh_every = _check_refresh_every(refresh_every)
    if stabiliser_log is not None and not isinstance(stabiliser_log, list):
        raise TypeError(f"stabiliser_log must be None or a list, got {type(stabiliser_log).__name__}")
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
    ndim = model_ndim(model)
    if ndim == 2 or depth is not None:
        depth = check_depth(depth, ndim, noise is not None)
    real, cplx = working_dtypes(model, rho_hat0, "rollout_imex")
    fp64 = real == torch.float64
    model.eval()
    device = rho_hat0.device
    if fp64:
        boxes = box.to(device=device, dtype=real).unsqueeze(0)
    else:
        boxes = box.unsqueeze(0).to(device)
    rho_hat = rho_hat0.clone()
    n = rho_hat.shape[1]
    rho_bar = _mean_density(rho_hat0, ndim)
    kBT = kB * float(T)
    t_kw = {"dtype": real} if fp64 else {}
    T_t = torch.tensor([float(T)], device=device, **t_kw)
    exact = exact_mode_indices(model) if fp64 else nullcontext()
    with exact, pointwise_guard(model, clamp_rho), \
            pointwise_fields(model, rho_hat0, kbt_field, v_ext, real) as kbt_t:
        w_scale = None
        if kbt_t is not None:
            if noise is not None:
                w_scale = torch.sqrt(kbt_t / kBT)
            kBT = float(kbt_t.mean(dtype=torch.float64))
            T_t = torch.tensor([kBT / kB], device=device, **t_kw)
        return _run(model, rho_hat, boxes, T_t, kBT, rho_bar, n, dt, n_steps,
                    save_every, m_stab, state_proj, state_clamp, domain,
                    mass_restore, noise, generator, w_scale, hook,
                    ndim=ndim, depth=depth, real=real, cplx=cplx,
                    bound_margin=bound_margin, refresh_every=refresh_every,
                    stabiliser_log=stabiliser_log)


def _frozen_mobility(model, rho_hat, ops, T_t, rho_bar, n, ndim, m_stab, hook,
                     bound_margin, stabiliser_log, step: int) -> torch.Tensor:
    """``M_s`` from the state ``rho_hat`` after ``step`` steps: the hook, ``"mean"``, ``"max"`` or ``"bound"``;
    one record appended to ``stabiliser_log`` (a list) when one is given."""
    N = math.prod(ops.grid)
    record = None
    if hook is not None:
        M_s = stabilizer_mobility(hook, ops.irfft(rho_hat * N), T_t)
        source = "hook"
        if step == 0:
            _LOG.info("M_s from %s.stabilizer_mobility", type(model).__name__)
    elif m_stab == "mean":
        M_s = _mean_mobility(model, rho_bar, T_t, n, ndim)
        source = "mean"
        if step == 0:
            _LOG.debug("M_s from m_stab='mean'")
    elif m_stab == BOUND:
        M_s, cells, info = _bound_stabiliser(model, ops.irfft(rho_hat * N), rho_bar, T_t, n, ndim,
                                             bound_margin)
        source = "bound"
        if stabiliser_log is not None:
            ratios = info["ratios"]
            record = {"r_raw": info["r_raw"], "r_used": info["r_used"], "margin": bound_margin,
                      "cell": _cell_index(info["index"], ops.grid),
                      "ratio_min": float(ratios.min()), "ratio_mean": float(ratios.mean()),
                      "ratio_max": float(ratios.max()), "cond": info["cond"],
                      "floored": info["floored"],
                      "ratio_after": float(generalised_ratios(M_s, cells).max())}
        if step == 0:
            _LOG.debug("M_s from m_stab='bound', r = %r", info["r_raw"])
    else:
        M_s = max_norm_mobility(model, ops.irfft(rho_hat * N), T_t, ndim)
        source = "max"
        if step == 0:
            _LOG.debug("M_s from m_stab='max'")
    if step != 0:
        _LOG.debug("M_s re-frozen after step %d from %s", step, source)
    if stabiliser_log is not None:
        stabiliser_log.append({"step": int(step), "source": source, **(record or {})})
    return M_s


def _operators(M_s, H_k, k2, dt, n, device, real, cplx, frac) -> tuple:
    """``(A_inv, MH, A_pred_inv, MH_pred)`` frozen at ``M_s``; the predictor's pair is ``None`` without
    ``frac``."""
    dtL = dt * k2 * torch.einsum("ij,...jk->...ik", M_s, H_k)
    A_inv = _inverse(_identity(n, device, real) + dtL).to(cplx)
    MH = dtL.to(cplx)
    if frac is None:
        return A_inv, MH, None, None
    A_pred_inv = _inverse(_identity(n, device, real) + frac * dtL).to(cplx)
    MH_pred = (frac * dtL).to(cplx)
    return A_inv, MH, A_pred_inv, MH_pred


def _run(model, rho_hat, boxes, T_t, kBT, rho_bar, n, dt, n_steps, save_every,
         m_stab, state_proj, state_clamp, domain, mass_restore, noise,
         generator, w_scale=None, hook=None, *, ndim: int = 3,
         depth: Optional[float] = None, real: torch.dtype = torch.float32,
         cplx: torch.dtype = torch.complex64, bound_margin: Optional[float] = None,
         refresh_every: Optional[int] = None,
         stabiliser_log: Optional[list] = None) -> torch.Tensor:
    device = rho_hat.device
    Hb = local_hessian(model, rho_bar, kBT)
    # float64: every buffer float64, the mode indices exact integers; float32: the set it always was.
    ops = _scheme_ops(model, rho_hat, n, ndim, real, device)
    N = math.prod(ops.grid)
    ks = ops.k_axes(boxes)
    kx, ky, kz = tuple(ks) + (None,) * (3 - ndim)
    kmag = torch.sqrt(k_squared(ks))[:, 0]
    k2 = (kmag * kmag).unsqueeze(-1).unsqueeze(-1)
    W_k = model.kernel.w_hat(kmag)
    H_k = W_k + Hb
    M_s = _frozen_mobility(model, rho_hat, ops, T_t, rho_bar, n, ndim, m_stab, hook,
                           bound_margin, stabiliser_log, 0)

    frac = filt = amp = amp_floor = None
    if noise is not None:
        filt = build_noise_filter(ops, boxes, noise["noise_mode"],
                                  noise["sigma_noise"])
        dV = float(boxes[0].prod()) / N
        if ndim == 2:
            dV = dV * depth
        amp = float(noise["noise_scale"]) * math.sqrt(
            2.0 * float(noise["kBT_noise"]) / (dV * dt))
        frac = NOISE_EVAL_FRAC.get(noise["noise_eval"])
        if frac is not None:
            amp_floor = float(noise["predictor_floor"])
    A_inv, MH, A_pred_inv, MH_pred = _operators(M_s, H_k, k2, dt, n, device, real, cplx, frac)

    to_last, to_second = CHANNEL_LAST[ndim], CHANNEL_SECOND[ndim]
    traj: List[torch.Tensor] = [rho_hat[0].cpu()]
    for step in range(1, n_steps + 1):
        if refresh_every is not None and step > 1 and (step - 1) % refresh_every == 0:
            # What a new call from this state would build; rho_bar (k = 0) is conserved, and the
            # Hessian at it is rebuilt only if a projection moved it.
            now = _mean_density(rho_hat, ndim)
            if not torch.equal(now, rho_bar):
                rho_bar = now
                H_k = W_k + local_hessian(model, rho_bar, kBT)
            M_s = _frozen_mobility(model, rho_hat, ops, T_t, rho_bar, n, ndim, m_stab, hook,
                                   bound_margin, stabiliser_log, step - 1)
            A_inv, MH, A_pred_inv, MH_pred = _operators(M_s, H_k, k2, dt, n, device, real,
                                                        cplx, frac)
        F_hat = model(rho_hat, boxes, T_t)
        rh = rho_hat.permute(*to_last).unsqueeze(-1)
        Fh = F_hat.permute(*to_last).unsqueeze(-1)
        if noise is None:
            rhs = rh + dt * Fh + MH @ rh
        else:
            rho_now = ops.irfft(rho_hat * N)
            w = draw_w(ops.grid, n, generator, device, rho_now.dtype)
            if w_scale is not None:
                w = w * w_scale
            if frac is None:
                zeta = zeta_from_w(model, rho_now, T_t, amp, w, ndim)
            else:
                zeta0 = zeta_from_w(model, rho_now, T_t, amp, w, ndim)
                n0_hat = noise_div_hat(ops, zeta0, kx, ky, kz, filt)
                star = (A_pred_inv @ (rh + (frac * dt) * Fh + MH_pred @ rh)) \
                    .squeeze(-1).permute(*to_second)
                rho_amp = ops.irfft((star + (frac * dt) * n0_hat) * N)
                zeta = zeta_from_w(model, rho_amp.clamp(min=amp_floor), T_t,
                                   amp, w, ndim)
            n_hat = noise_div_hat(ops, zeta, kx, ky, kz, filt)
            nh = n_hat.permute(*to_last).unsqueeze(-1)
            rhs = rh + dt * Fh + MH @ rh + dt * nh
        rho_hat = (A_inv @ rhs).squeeze(-1).permute(*to_second)
        rho_hat = hermitianize(rho_hat, ops)
        rho_hat = _project(rho_hat, ops, state_proj, state_clamp, domain,
                           mass_restore)
        if step % save_every == 0 or step == n_steps:
            traj.append(rho_hat[0].cpu())
    return torch.stack(traj)
