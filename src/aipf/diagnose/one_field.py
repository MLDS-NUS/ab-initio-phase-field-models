"""The one-field, fixed-density phase diagram: binodal, spinodal, a fixed-exponent dome fit, ``T_c``.
The field is a lattice fraction ``phi``; no pressure, no isobar. Every threshold is declared.
Two read-offs (:data:`READOFFS`): ``"mu_roots"`` (:func:`phase_diagram`, the roots of ``mu``) and
``"bulk_minima"`` (:func:`bulk_minima`, a closed form's minima and ``f''`` sign on a grid)."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, curve_fit

#: The declared read-offs, each with the keys its block must carry.
READOFFS = {
    "mu_roots": ("dtype", "phi_grid", "phi_clip", "min_width", "bracket_eps", "mu_w0", "tc_w0",
                 "ising"),
    "bulk_minima": ("dtype", "phi_grid", "edge_trim", "centre"),
}

#: Floor of ``1 - T/T_c`` inside the dome fit's power law (arithmetic guard, not a threshold).
_POWER_BASE_FLOOR = 1e-12


def _check_model(model) -> None:
    f = model.f_local
    if model.n_species != 1 or f.ideal_form != "lattice":
        raise NotImplementedError(
            "the one-field route reads one lattice fraction; this model has "
            f"n_species={model.n_species} and ideal_form={f.ideal_form!r}")
    if f.f_exc_form != "split" or f.enable_TlnT or f.enable_T2:
        raise NotImplementedError(
            "the one-field route sums the split form's energy and entropy "
            "slopes; a joint net or a T-basis head is not supported")


def _centre(model) -> float:
    c = model.kernel_centre
    return 0.0 if c is None else float(c[0])


def uniform_mu(model, *, clip: float, w0: float) -> Callable:
    """``mu(phi, T) = u'(r) + kB T (ln r - ln(1-r) + g'(r)) + w0 (phi - c)``, ``r`` = ``phi`` clipped to
    ``[clip, 1 - clip]``, nets in the model's dtype, the rest float64; ``c`` the kernel centre."""
    _check_model(model)
    f, kB, c = model.f_local, float(model.kB), _centre(model)
    dtype = next(model.parameters()).dtype

    def mu(phi: np.ndarray, T: float) -> np.ndarray:
        r = np.clip(phi, clip, 1 - clip)
        rt = torch.tensor(r, dtype=dtype).reshape(-1, 1)
        du = f.du_drho(rt).reshape(-1).detach().numpy()
        s = np.log(r) - np.log1p(-r)
        s = s + f.dg_drho(rt).reshape(-1).detach().numpy()
        return du + (kB * T) * s + w0 * (phi - c)
    return mu


def binodal_at(mu: Callable, T: float, phi: np.ndarray, *, min_width: float,
               centre: float, bracket_eps: float) -> Optional[Tuple[float, float]]:
    """The roots of ``mu = 0`` either side of the symmetric ``centre`` (cubic spline, brentq); the right one
    mirrored as ``2 centre - left`` when it has no bracket. ``None`` if narrower than ``min_width``."""
    mus = mu(phi, T)
    mu_spl = CubicSpline(phi, mus)
    lo, hi = float(phi[0]), float(phi[-1])
    eps, half = float(bracket_eps), float(centre)
    try:
        if mu_spl(lo + eps) * mu_spl(half - eps) >= 0:
            return None
        left = float(brentq(mu_spl, lo + eps, half - eps))
    except (ValueError, RuntimeError):
        return None
    try:
        if mu_spl(half + eps) * mu_spl(hi - eps) >= 0:
            right = 2 * half - left
        else:
            right = float(brentq(mu_spl, half + eps, hi - eps))
    except (ValueError, RuntimeError):
        right = 2 * half - left
    if (right - left) < min_width:
        return None
    return left, right


def spinodal_at(mu: Callable, T: float, phi: np.ndarray) -> Optional[Tuple[float, float]]:
    """The outermost sign changes of ``d mu / d phi`` (finite difference, spline, brentq per bracket)."""
    mus = mu(phi, T)
    dmu = np.diff(mus) / np.diff(phi)
    mid = 0.5 * (phi[:-1] + phi[1:])
    sign = np.sign(dmu)
    cross = np.where(sign[:-1] * sign[1:] < 0)[0]
    if len(cross) < 2:
        return None
    dmu_spl = CubicSpline(mid, dmu)
    roots = []
    for i in cross:
        a, b = float(mid[i]), float(mid[i + 1])
        try:
            roots.append(float(brentq(dmu_spl, a, b)))
        except (ValueError, RuntimeError):
            pass
    if len(roots) < 2:
        return None
    return min(roots), max(roots)


def fit_dome(Ts, left, right, *, T_max: float, beta: float, B0: float,
             Tc0_floor: float, Tc0_offset: float, B_bounds: Tuple[float, float],
             Tc_bounds_offset: Tuple[float, float], maxfev: int) -> Optional[dict]:
    """``(right - left)/2 = (B/2)(1 - T/T_c)^beta``, ``beta`` fixed, on ``T <= T_max``.
    Start ``(B0, max(T_top + Tc0_offset, Tc0_floor))``; ``T_c`` bounded to ``T_top + Tc_bounds_offset``."""
    Ts = np.asarray(Ts)
    left = np.asarray(left)
    right = np.asarray(right)
    finite = np.isfinite(left) & np.isfinite(right) & (Ts <= T_max)
    if finite.sum() < 2:
        return None
    T_fit = Ts[finite]
    delta = 0.5 * (right[finite] - left[finite])

    def model(T, B, T_c):
        arg = np.clip(1.0 - T / T_c, _POWER_BASE_FLOOR, None)
        return (B / 2.0) * arg ** beta

    p0 = [B0, max(T_fit.max() + Tc0_offset, Tc0_floor)]
    bounds = ([B_bounds[0], T_fit.max() + Tc_bounds_offset[0]],
              [B_bounds[1], T_fit.max() + Tc_bounds_offset[1]])
    try:
        popt, _ = curve_fit(model, T_fit, delta, p0=p0, bounds=bounds,
                            maxfev=maxfev)
    except Exception:
        return None
    return {"B": float(popt[0]), "T_c": float(popt[1]), "beta": float(beta)}


def tc_curvature(model, w0: torch.Tensor) -> float:
    """``T_c = -(u''(c) + w0) / (kB (1/c + 1/(1-c) + g''(c)))``, ``c = rho_ref``: where the uniform curvature
    at the symmetric point vanishes (exact for the split form with no T-basis head)."""
    _check_model(model)
    f = model.f_local
    p = next(model.parameters())
    c = float(f.u_net.rho_ref[0]) if hasattr(f.u_net, "rho_ref") else _centre(model)

    def second(fn):
        phi = torch.full((1, 1), c, dtype=p.dtype, requires_grad=True)
        with torch.enable_grad():
            y = fn(phi).sum()
            g1 = torch.autograd.grad(y, phi, create_graph=True)[0]
            g2 = torch.autograd.grad(g1.sum(), phi, create_graph=True)[0]
        return g2.squeeze()

    e_pp = second(f.u_net)
    g_pp = second(f._g())
    id_pp = 1.0 / c + 1.0 / (1.0 - c)
    return float(-(e_pp + w0.to(e_pp.dtype)) / (float(model.kB) * (id_pp + g_pp)))


def phase_diagram(model, T_grid: np.ndarray, phi: np.ndarray,
                  spec: Mapping[str, Any]) -> dict:
    """Every array of the one-field diagnosis, under fixed key names."""
    mu_w0 = float(model.kernel.w_hat_zero_radial(**dict(spec["mu_w0"]))[0, 0].item())
    mu = uniform_mu(model, clip=float(spec["phi_clip"]), w0=mu_w0)
    centre = _centre(model)
    rows = {k: [] for k in ("binodal_L", "binodal_R", "spinodal_L", "spinodal_R")}
    for T in T_grid:
        b = binodal_at(mu, T, phi, min_width=float(spec["min_width"]),
                       centre=centre, bracket_eps=float(spec["bracket_eps"]))
        s = spinodal_at(mu, T, phi)
        rows["binodal_L"].append(b[0] if b else np.nan)
        rows["binodal_R"].append(b[1] if b else np.nan)
        rows["spinodal_L"].append(s[0] if s else np.nan)
        rows["spinodal_R"].append(s[1] if s else np.nan)
    out = {"T": np.asarray(T_grid), **{k: np.array(v) for k, v in rows.items()}}
    ising = spec["ising"]
    fit = None if ising is None else fit_dome(
        T_grid, out["binodal_L"], out["binodal_R"], **dict(ising))
    tc_w0 = model.kernel.w_hat_zero_radial(**dict(spec["tc_w0"]))[0, 0]
    out["Tc_exact"] = tc_curvature(model, tc_w0)
    out["ising_T_max"] = np.nan if ising is None else float(ising["T_max"])
    for key in ("B", "T_c", "beta"):
        out[f"ising_{key}"] = fit[key] if fit else np.nan
    out["mu_w0"] = mu_w0
    return out


def bulk_minima(model, T_grid: np.ndarray, phi: np.ndarray, *, edge_trim: int,
                centre: float) -> dict:
    """Per ``T``: where ``f''`` (``np.gradient`` twice, ``edge_trim`` end points dropped) is negative, the
    spinodal is its outermost grid points and the binodal the grid minimum of ``f`` below ``centre``, mirrored.
    ``T_c`` is the model's closed ``Tc_curvature``. Keys as :func:`phase_diagram`'s."""
    for method in ("bulk_free_energy_curve", "Tc_curvature"):
        if not callable(getattr(model, method, None)):
            raise NotImplementedError(
                f"the bulk_minima read-off needs the closed form's {method}; "
                f"{type(model).__name__} has none")
    k = int(edge_trim)
    if k < 1:
        raise ValueError(f"edge_trim={edge_trim!r}: at least one end point is dropped")
    nT = len(T_grid)
    out = {key: np.full(nT, np.nan)
           for key in ("binodal_lo", "binodal_hi", "spinodal_lo", "spinodal_hi")}
    for i, T in enumerate(T_grid):
        f = model.bulk_free_energy_curve(phi, float(T)).detach().numpy()
        fpp = np.gradient(np.gradient(f, phi), phi)
        neg = np.nonzero(fpp[k:-k] < 0)[0]
        if neg.size:
            out["spinodal_lo"][i], out["spinodal_hi"][i] = phi[k:-k][neg[0]], phi[k:-k][neg[-1]]
            half = phi < centre
            out["binodal_lo"][i] = phi[half][np.argmin(f[half])]
            out["binodal_hi"][i] = 2 * centre - out["binodal_lo"][i]
    return {"Tc": model.Tc_curvature(), "T_grid": np.asarray(T_grid, float), **out}
