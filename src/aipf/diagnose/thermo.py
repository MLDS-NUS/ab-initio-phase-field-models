"""Binodal, spinodal and the critical temperature, each by a declared route with no default.

Works on one scalar order parameter; the caller supplies ``mu``/``f`` as ``(order_parameter, T) -> float``.
Binodal routes: ``mu_roots``, ``convex_hull``. ``T_c`` methods: ``ising_fit``, ``bracket``, ``dome_apex``."""
from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import brentq, curve_fit

MuFn = Callable[[float, float], float]
FFn = Callable[[float, float], float]

# Admissible binodal routes and T_c methods.
BINODAL_ROUTES: Tuple[str, ...] = ("mu_roots", "convex_hull")

TC_METHODS: Tuple[str, ...] = ("ising_fit", "bracket", "dome_apex")

Anchor = Union[float, Literal["auto"]]


def binodal_mu_roots(
    mu: MuFn, T: float, domain: Tuple[float, float], *,
    symmetry_point: float, rel_eps: float = 1e-6,
) -> Optional[Tuple[float, float]]:
    """The two non-trivial roots of ``mu(x, T) = 0`` either side of the required ``symmetry_point``.

    Returns ``(x_lo, x_hi)`` sorted, or ``None`` if either half of ``domain`` has no sign change."""
    lo, hi = float(domain[0]), float(domain[1])
    if not lo < symmetry_point < hi:
        raise ValueError(
            f"symmetry_point={symmetry_point} is not inside domain={domain}"
        )
    span = hi - lo
    eps = rel_eps * span

    def _signed_root(a: float, b: float) -> Optional[float]:
        fa, fb = mu(a, T), mu(b, T)
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            return None
        return brentq(lambda x: mu(x, T), a, b, xtol=1e-12)

    left = _signed_root(lo + eps, symmetry_point - eps)
    right = _signed_root(symmetry_point + eps, hi - eps)
    if left is None or right is None:
        return None
    return (min(left, right), max(left, right))


def _cross(o: Tuple[float, float], a: Tuple[float, float],
           b: Tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def lower_hull_indices(xs: Sequence[float], ys: Sequence[float]) -> list:
    """Indices of the lower convex hull of ``(xs[i], ys[i])``, ``xs`` sorted ascending (monotone chain)."""
    hull: list = []
    for i in range(len(xs)):
        p = (xs[i], ys[i])
        while len(hull) >= 2 and _cross(
            (xs[hull[-2]], ys[hull[-2]]),
            (xs[hull[-1]], ys[hull[-1]]),
            p,
        ) <= 0:
            hull.pop()
        hull.append(i)
    return hull


def _most_unstable_x(f: FFn, T: float, xs: Sequence[float],
                      margin: int) -> float:
    """The grid point of most negative discrete curvature of ``f(., T)`` (resolves ``anchor="auto"``)."""
    n = len(xs)
    lo_i, hi_i = margin, n - margin
    if hi_i <= lo_i:
        raise ValueError(
            f"grid of {n} points is too short for margin={margin} on both "
            f"sides while resolving anchor='auto'"
        )
    best_i, best_curv = None, None
    for i in range(lo_i, hi_i):
        h_lo, h_hi = xs[i] - xs[i - 1], xs[i + 1] - xs[i]
        # non-uniform-safe three-point second difference.
        f0, fp, fm = f(xs[i], T), f(xs[i + 1], T), f(xs[i - 1], T)
        curv = 2.0 * (h_lo * fp - (h_lo + h_hi) * f0 + h_hi * fm) / (
            h_lo * h_hi * (h_lo + h_hi))
        if best_curv is None or curv < best_curv:
            best_curv, best_i = curv, i
    return xs[best_i]


def binodal_convex_hull(
    f: FFn, T: float, grid: Sequence[float], *, anchor: Anchor,
    curvature_margin: int = 2,
) -> Optional[Tuple[float, float]]:
    """The lower-hull facet of ``f`` on ascending ``grid`` bracketing ``anchor`` (a float or
    ``"auto"``, required).

    Returns ``(x_lo, x_hi)``, or ``None`` when ``anchor`` sits on the hull (single phase there)."""
    xs = [float(x) for x in grid]
    if xs != sorted(xs):
        raise ValueError("grid must be sorted ascending")
    if len(xs) < 2 * curvature_margin + 1:
        raise ValueError(
            f"grid of {len(xs)} points is too short for curvature_margin="
            f"{curvature_margin}"
        )
    fs = [float(f(x, T)) for x in xs]
    hull_idx = lower_hull_indices(xs, fs)
    hull_x = [xs[i] for i in hull_idx]

    if anchor == "auto":
        a = _most_unstable_x(f, T, xs, curvature_margin)
    else:
        a = float(anchor)
        if not xs[0] < a < xs[-1]:
            raise ValueError(
                f"anchor={a} is outside the grid domain ({xs[0]}, {xs[-1]})"
            )

    for edge_lo, edge_hi in zip(hull_x[:-1], hull_x[1:]):
        if not edge_lo <= a <= edge_hi:
            continue
        if edge_hi - edge_lo <= (xs[1] - xs[0]) * 1.5:
            return None
        if edge_lo < a < edge_hi:
            return (edge_lo, edge_hi)
        return None   # anchor coincides with a hull vertex: single phase.
    raise AssertionError(
        f"anchor={a} not bracketed by any hull segment of {hull_x!r}: "
        f"the hull must always contain its own endpoints"
    )


def binodal(route: str, *, T: float, **kwargs) -> Optional[Tuple[float, float]]:
    """Dispatch on a required ``route`` in :data:`BINODAL_ROUTES`, forwarding ``**kwargs`` unchanged."""
    if route == "mu_roots":
        return binodal_mu_roots(T=T, **kwargs)
    if route == "convex_hull":
        return binodal_convex_hull(T=T, **kwargs)
    raise ValueError(f"unknown binodal route {route!r}; have {BINODAL_ROUTES}")


def spinodal(mu: MuFn, T: float, domain: Tuple[float, float], *,
             n_grid: int = 400) -> list:
    """Roots of ``d(mu)/dx = 0`` (``d2f/dx2 = 0``): central difference on ``n_grid`` points, then
    ``brentq``."""
    lo, hi = float(domain[0]), float(domain[1])
    h = (hi - lo) / (4.0 * n_grid)
    xs = np.linspace(lo, hi, n_grid)

    def dmu(x: float) -> float:
        return (mu(min(x + h, hi), T) - mu(max(x - h, lo), T)) / (2.0 * h)

    vals = [dmu(float(x)) for x in xs]
    roots = []
    for a, b, fa, fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if fa == 0.0:
            roots.append(float(a))
            continue
        if fa * fb < 0.0:
            roots.append(brentq(dmu, float(a), float(b), xtol=1e-12))
    return sorted(roots)


def critical_temperature(
    binodal_at: Callable[[float], Optional[Tuple[float, float]]],
    T_grid: Sequence[float], *, method: str, beta: Optional[float] = None,
    T_max_fit: Optional[float] = None, **fit_kwargs,
) -> float:
    """``T_c`` by a required ``method`` in :data:`TC_METHODS`; ``binodal_at(T)`` returns ``None``
    when single-phase.

    ``ising_fit``: ``delta(T) = (B/2) (1 - T/T_c)^beta``, ``beta`` required; ``bracket``: midpoint of the last
    two-phase and first single-phase ``T``; ``dome_apex``: :func:`dome_apex`, keywords forwarded."""
    if method not in TC_METHODS:
        raise ValueError(f"unknown method {method!r}; have {TC_METHODS}")
    Ts = sorted(float(T) for T in T_grid)
    results = [(T, binodal_at(T)) for T in Ts]

    if method == "dome_apex":
        lo = np.array([r[0] if r is not None else np.nan for _, r in results])
        hi = np.array([r[1] if r is not None else np.nan for _, r in results])
        return float(dome_apex(np.array(Ts), lo, hi, **fit_kwargs)[0])

    if method == "bracket":
        two_phase_Ts = [T for T, r in results if r is not None]
        if not two_phase_Ts:
            raise ValueError("no two-phase T in T_grid to bracket from")
        T_last = max(two_phase_Ts)
        above = [T for T, r in results if r is None and T > T_last]
        if not above:
            raise ValueError(
                "no single-phase T above the two-phase region in T_grid: "
                "T_grid does not bracket a transition"
            )
        return 0.5 * (T_last + min(above))

    # method == "ising_fit"
    if beta is None:
        raise ValueError(
            "method='ising_fit' requires beta: the Ising exponent is a "
            "declared modelling choice (one system fixes 0.325), never a "
            "package default"
        )
    two_phase = [(T, r) for T, r in results if r is not None]
    if T_max_fit is not None:
        two_phase = [(T, r) for T, r in two_phase if T <= T_max_fit]
    if len(two_phase) < 3:
        raise ValueError(
            f"only {len(two_phase)} two-phase point(s) in T_grid "
            f"(T_max_fit={T_max_fit}); need at least 3 to fit T_c"
        )
    Ts_fit = np.array([T for T, _ in two_phase], dtype=np.float64)
    delta = np.array([0.5 * (r[1] - r[0]) for _, r in two_phase],
                      dtype=np.float64)

    def model(T, B, T_c):
        arg = np.clip(1.0 - T / T_c, 1e-12, None)
        return (B / 2.0) * arg ** beta

    p0 = [float(delta.max()) * 2.0, float(Ts_fit.max()) * 1.05]
    popt, _ = curve_fit(model, Ts_fit, delta, p0=p0, maxfev=20000)
    return float(popt[1])


def dome_tail(T: Sequence[float], bin_lo: Sequence[float],
              bin_hi: Sequence[float], *, central_lo_max: float,
              tail_fraction: float):
    """``(central, tail, T_last)``: ties with finite branches and ``bin_lo < central_lo_max``; the
    tail narrower
    than ``tail_fraction`` of the widest; ``T_last`` the highest clean (non-tail) central tie. Cuts required."""
    T = np.asarray(T, dtype=np.float64)
    bin_lo = np.asarray(bin_lo, dtype=np.float64)
    bin_hi = np.asarray(bin_hi, dtype=np.float64)
    central = (np.isfinite(bin_lo) & np.isfinite(bin_hi)
               & (bin_lo < central_lo_max))
    tail = np.zeros_like(central)
    nan = np.float64(np.nan)
    if not central.any():
        return central, tail, nan
    w = bin_hi - bin_lo
    w_max = float(w[central].max())
    if not (np.isfinite(w_max) and w_max > 0):
        return central, tail, nan
    tail = central & (w < tail_fraction * w_max)
    clean = central & ~tail
    T_last = np.float64(T[clean].max()) if clean.any() else nan
    return central, tail, T_last


def dome_apex(T: Sequence[float], bin_lo: Sequence[float],
              bin_hi: Sequence[float], *, T_step: float,
              central_lo_max: float, tail_fraction: float,
              min_tail_points: int, max_overshoot_steps: float):
    """``(T_c, w0, T_last)`` from ``w(T)^2 = a (T_c - T)`` fitted over :func:`dome_tail`'s tail;
    ``T_c = intercept / a``, ``w0 = sqrt(a)``.

    Any failed guard returns ``(nan, nan, T_last)``; every threshold is required."""
    nan = np.float64(np.nan)
    T = np.asarray(T, dtype=np.float64)
    bin_lo = np.asarray(bin_lo, dtype=np.float64)
    bin_hi = np.asarray(bin_hi, dtype=np.float64)
    _central, tail, T_last = dome_tail(
        T, bin_lo, bin_hi, central_lo_max=central_lo_max,
        tail_fraction=tail_fraction)
    if not np.isfinite(T_last):
        return nan, nan, nan
    if int(tail.sum()) < int(min_tail_points):
        return nan, nan, T_last
    T_tail, w2_tail = T[tail], (bin_hi[tail] - bin_lo[tail]) ** 2
    try:
        slope, intercept = np.polyfit(T_tail, w2_tail, 1)
    except Exception:
        return nan, nan, T_last
    a = -float(slope)
    if not (np.isfinite(a) and a > 0 and np.isfinite(intercept)):
        return nan, nan, T_last
    T_c = float(intercept) / a
    if (not np.isfinite(T_c) or T_c < T_last
            or T_c > T_last + float(max_overshoot_steps) * float(T_step)):
        return nan, nan, T_last
    return np.float64(T_c), np.float64(np.sqrt(a)), T_last
