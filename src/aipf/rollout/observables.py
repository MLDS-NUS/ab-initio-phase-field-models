"""What the two checks measure, on real fields ``rho (F, n, Gx, Gy, Gz)``; the order parameter is
``c = rho[:, x_channel] / sum_i rho_i``. Spinodal: ``Phi`` (density-weighted ``Var(c)``), radially binned
``S_cc(k)`` on integer wavenumbers, ``L = L_box / k1``. Slab: the profile ``c(z)``, its plateaus and
10-90 width. FDT gate: the stationary 2x2 spectrum against ``G(k)^2 kBT H(k)^-1``."""
from __future__ import annotations

import numpy as np

from aipf.spectral import refuse_two_dimensions


def _refuse_two_dimensional_field(a, what: str) -> None:
    """A ``(F, n, Gx, Gy)`` field is two-dimensional; these measures read ``(F, n, Gx, Gy, Gz)``."""
    refuse_two_dimensions(2 if np.ndim(a) == 4 else 3, what)


def weighted_var_c(rho, x_channel: int):
    """``(Phi (F,), c, cbar)``: density-weighted ``Var(c)``, the field ``c`` and its weighted mean."""
    tot = np.maximum(rho.sum(axis=1), 1e-9)
    ax = tuple(range(1, tot.ndim))
    w = tot / tot.sum(axis=ax, keepdims=True)
    c = rho[:, x_channel] / tot
    cbar = (w * c).sum(axis=ax, keepdims=True)
    return (w * (c - cbar) ** 2).sum(axis=ax), c, cbar


def decomp_metrics(rho, box_L, x_channel: int):
    """``(Phi (F,), L (F,), Sk (F, G/2-1))`` on a cubic grid; ``Sk`` on integer wavenumbers ``1..G/2-1``,
    ``L = box_L / k1`` with ``k1`` the ``S``-weighted mean integer wavenumber."""
    _refuse_two_dimensional_field(rho, "decomp_metrics")
    ng = rho.shape[-1]
    Phi, c, cbar = weighted_var_c(rho, x_channel)
    S = np.abs(np.fft.fftn(c - cbar, axes=(1, 2, 3))) ** 2
    kf = np.fft.fftfreq(ng) * ng
    KX, KY, KZ = np.meshgrid(kf, kf, kf, indexing="ij")
    kbin = np.sqrt(KX**2 + KY**2 + KZ**2).round().astype(int)
    nb = ng // 2
    Sk = np.stack([S[:, kbin == k].mean(axis=1) for k in range(1, nb)], axis=1)
    kvals = np.arange(1, nb)
    k1 = (kvals * Sk).sum(axis=1) / np.maximum(Sk.sum(axis=1), 1e-12)
    L = box_L / np.maximum(k1, 1e-6)
    return Phi, L, Sk


def conc_profile(rho, x_channel: int, axis: int):
    """``c`` along the spatial ``axis`` (0, 1 or 2), density-weighted over the other two; ``(F, G_axis)``."""
    _refuse_two_dimensional_field(rho, "conc_profile")
    lateral = tuple(2 + a for a in range(3) if a != axis)
    lat = np.asarray(rho).sum(axis=lateral, dtype=np.float64)       # (F, n, G)
    return lat[:, x_channel] / np.maximum(lat.sum(axis=1), 1e-9)


def extract_plateaus(profile):
    """``(low, high)``: the means of the lowest and highest quarter of the profile's samples."""
    s = np.sort(np.asarray(profile))
    q = max(1, len(s) // 4)
    return float(s[:q].mean()), float(s[-q:].mean())


def _level_crossings(prof, level):
    """Fractional indices where the periodic profile crosses ``level`` (a sample on it counts)."""
    d = np.asarray(prof, dtype=np.float64) - float(level)
    n = len(d)
    out = []
    for i in range(n):
        j = (i + 1) % n
        if d[i] == 0.0:
            out.append(float(i))
        elif d[i] * d[j] < 0.0:
            out.append(i + d[i] / (d[i] - d[j]))
    return np.asarray(out, dtype=np.float64)


def interface_width(prof, dz: float) -> float:
    """10-90 width of the sharper interface, levels relative to the profile's own plateaus; NaN without a gap."""
    p = np.asarray(prof, dtype=np.float64).ravel()
    lo, hi = extract_plateaus(p)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    z10 = _level_crossings(p, lo + 0.1 * (hi - lo))
    z90 = _level_crossings(p, lo + 0.9 * (hi - lo))
    if not len(z10) or not len(z90):
        return float("nan")
    n = len(p)
    d = np.abs(z10[:, None] - z90[None, :]) % n
    return float(np.minimum(d, n - d).min() * dz)


def field_frames(t_model, t_md, stride: int):
    """``(sel_model, idx_md)``: every ``stride``-th model save inside the MD window, each with the MD frame
    nearest in time."""
    t_model = np.asarray(t_model, dtype=np.float64)
    t_md = np.asarray(t_md, dtype=np.float64)
    sel = np.arange(0, len(t_model), max(1, int(stride)))
    sel = sel[t_model[sel] <= t_md[-1] + 1e-9]
    idx = np.abs(t_md[None, :] - t_model[sel][:, None]).argmin(axis=1)
    return sel, idx


# ------------------------------------------------------------------ the FDT gate

def mode_wavevectors(box, grid):
    """``(|k|, multiplicity)`` of every stored rfft mode, ``(Gx, Gy, Gzr)``; multiplicity 2 off the
    ``k_z = 0`` and Nyquist planes."""
    refuse_two_dimensions(len(grid), "mode_wavevectors")
    Gx, Gy, Gz = (int(g) for g in grid)
    Gzr = Gz // 2 + 1
    kx = (2.0 * np.pi * np.fft.fftfreq(Gx) * Gx / box[0]).reshape(Gx, 1, 1)
    ky = (2.0 * np.pi * np.fft.fftfreq(Gy) * Gy / box[1]).reshape(1, Gy, 1)
    kz = (2.0 * np.pi * np.arange(Gzr, dtype=np.float64) / box[2]).reshape(1, 1, Gzr)
    kmag = np.broadcast_to(np.sqrt(kx ** 2 + ky ** 2 + kz ** 2), (Gx, Gy, Gzr)).copy()
    mult = np.full((1, 1, Gzr), 2.0)
    mult[..., 0] = 1.0
    if Gz % 2 == 0:
        mult[..., -1] = 1.0
    return kmag, np.broadcast_to(mult, (Gx, Gy, Gzr)).copy()


def measure_S_modes(frames, V: float):
    """``S_ij(k) = V mean_F[rho_hat_i conj(rho_hat_j)]`` from stored ``(F, n, Gx, Gy, Gzr)`` states."""
    _refuse_two_dimensional_field(frames, "measure_S_modes")
    a = np.asarray(frames).astype(np.complex128)
    return np.einsum("fixyz,fjxyz->xyzij", a, a.conj()) * (V / a.shape[0])


def trace_and_cc(S):
    """``(tr S, S_cc)``, ``S_cc = S_00 + S_11 - 2 Re S_01``, from a ``(..., 2, 2)`` spectrum."""
    tr = np.real(S[..., 0, 0] + S[..., 1, 1])
    cc = np.real(S[..., 0, 0] + S[..., 1, 1] - 2.0 * S[..., 0, 1])
    return tr, cc


def s_pred_from_H(H, kBT: float, kmag, sigma: float):
    """``G(k)^2 kBT H(k)^-1``, ``G = exp(-k^2 sigma^2 / 2)``, per mode."""
    k = np.asarray(kmag, dtype=np.float64)
    G2 = np.exp(-(k ** 2) * float(sigma) ** 2)
    return G2[:, None, None] * float(kBT) * np.linalg.inv(np.asarray(H, dtype=np.float64))


def weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float((values * weights).sum() / weights.sum())


def band_means(res, k_max: float):
    """Multiplicity-weighted means of the per-mode ``tr`` and ``cc`` ratios over ``0 < |k| <= k_max``."""
    sel = res["k_mode"] <= float(k_max) + 1e-12
    w = res["w_mode"][sel]
    return (weighted_mean(res["r_tr_mode"][sel], w),
            weighted_mean(res["r_cc_mode"][sel], w))


def tier1_ok(res, k_max: float, window) -> bool:
    """Both band-mean ratios inside ``window = (lo, hi)``."""
    tr, cc = band_means(res, k_max)
    return window[0] <= tr <= window[1] and window[0] <= cc <= window[1]


def path_divergence(by_eval: dict, k_max: float, reference: str = "ito"):
    """``{convention: (median relative difference of tr ratio vs reference, fraction of modes differing)}``."""
    ref = by_eval[reference]
    sel = ref["k_mode"] <= float(k_max) + 1e-12
    a = np.asarray(ref["r_tr_mode"])[sel]
    out = {}
    for mode, r in by_eval.items():
        if mode == reference:
            continue
        b = np.asarray(r["r_tr_mode"])[sel]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(b - a) / np.abs(a)
        rel = rel[np.isfinite(rel)]
        out[mode] = (float(np.median(rel)) if rel.size else 0.0,
                     float(np.count_nonzero(b != a) / max(1, len(a))))
    return out


def convention_spread(by_eval: dict, k_max: float):
    """``(max - min) / mean`` of the band means across conventions, for ``tr`` and ``cc``."""
    means = [band_means(r, k_max) for r in by_eval.values()]
    out = []
    for j in range(2):
        vals = [m[j] for m in means]
        out.append(0.0 if len(vals) < 2 else
                   (max(vals) - min(vals)) / (sum(vals) / len(vals)))
    return tuple(out)


def tier2_ok(by_eval: dict, conventions, k_max: float, spread_tol: float,
             min_divergence: float, min_fraction: float):
    """``(ok, detail)``: every convention present, each non-ito one distinct from ito (the positive
    control), and the band-mean spread within ``spread_tol``."""
    missing = [m for m in conventions if m not in by_eval]
    if missing:
        return False, {"missing": missing}
    div = path_divergence(by_eval, k_max)
    spread = convention_spread(by_eval, k_max)
    control = bool(div) and all(d >= min_divergence and f >= min_fraction
                                for d, f in div.values())
    bound = max(spread) <= spread_tol
    return bool(control and bound), {"divergence": div, "spread": spread,
                                     "control_ok": control, "bound_ok": bound}
