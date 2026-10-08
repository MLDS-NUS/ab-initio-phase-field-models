"""The noise colour filter ``G(k)``, which must be declared: white ``S(k) = kBT H(k)^-1``,
coloured ``S(k) = G(k)^2 kBT H(k)^-1``, ``G(k) = exp(-k^2 sigma^2 / 2)``, ``sigma`` = coarse-graining length.
A two-dimensional model's noise also declares ``depth`` (:func:`check_depth`)."""
from __future__ import annotations

from typing import Optional

import torch

# One sentinel for every declared-not-defaulted knob (:mod:`aipf.solve.declare`).
from .declare import UNDECLARED

# The admissible ``noise_mode`` and ``m_stab`` values, one source.
from aipf.system import M_STAB_MODES, NOISE_MODES


def _require_declared(noise_mode, sigma_noise) -> None:
    if noise_mode is UNDECLARED:
        raise ValueError(
            "noise_mode must be declared: 'gaussian' (coloured, "
            "G(k) = exp(-k^2 sigma_noise^2 / 2), which is what a model "
            "written at a finite coarse-graining length needs) or 'none' "
            "(white, G = 1). There is no default, deliberately: the two "
            "differ by a factor of 1/exp(-4) = 54.598 wherever "
            "k*sigma_noise = 2, and a default is how that factor once "
            "entered four production runs unremarked.")
    if sigma_noise is UNDECLARED:
        raise ValueError(
            f"sigma_noise must be declared alongside noise_mode="
            f"{noise_mode!r}: a positive filter width for "
            f"noise_mode='gaussian', or None for noise_mode='none'. There "
            f"is no default -- see aipf.solve.noise.")


def check_noise_declaration(noise_mode, sigma_noise) -> None:
    """Validate ``noise_mode``/``sigma_noise`` without building; raises as :func:`build_noise_filter` does."""
    _require_declared(noise_mode, sigma_noise)
    if noise_mode == "none":
        if sigma_noise is not None:
            raise ValueError(
                f"noise_mode='none' takes sigma_noise=None (G = 1, no "
                f"filter); got sigma_noise={sigma_noise!r}. Declaring a "
                f"width and then not applying it is the ambiguity this "
                f"module exists to remove.")
        return
    if noise_mode == "gaussian":
        if sigma_noise is None:
            raise ValueError(
                "noise_mode='gaussian' requires a positive sigma_noise: "
                "the coarse-graining length the model's field is defined "
                "at, not a free parameter.")
        sigma = float(sigma_noise)
        if not (sigma > 0.0) or sigma != sigma or sigma == float("inf"):
            raise ValueError(
                f"sigma_noise must be finite and positive, got {sigma!r}")
        return
    raise ValueError(
        f"noise_mode must be one of {list(NOISE_MODES)}, got {noise_mode!r}")


def build_noise_filter(ops, boxes: torch.Tensor, noise_mode,
                       sigma_noise) -> Optional[torch.Tensor]:
    """``G(k)`` as a real ``(B, 1, Gx, Gy, Gzr)`` tensor for ``"gaussian"``, or ``None`` for ``"none"``.
    ``sigma_noise`` is in the length unit of ``boxes``; delegates to ``ops.sigma_filter``."""
    check_noise_declaration(noise_mode, sigma_noise)
    if noise_mode == "none":
        return None
    return ops.sigma_filter(boxes, float(sigma_noise))


def check_depth(depth, ndim: int, noisy: bool) -> Optional[float]:
    """``depth``, the extent along the averaged axis the noise of a two-dimensional model needs.
    Required (finite, positive) for a noisy 2D step, refused in 3D, unread by a deterministic 2D one."""
    if ndim != 2:
        if depth is not None:
            raise ValueError(
                f"depth={depth!r} is declared for a three-dimensional model: depth is the "
                f"extent of a two-dimensional model's cell along the axis it averages, and a "
                f"three-dimensional cell's volume is read off its box. Leave depth unset")
        return None
    if depth is None:
        if noisy:
            raise ValueError(
                "a two-dimensional model's noise needs depth=: the variance is "
                "2 kBT / (dV dt) with dV = dA * depth, and which densities the model "
                "carries decides it. depth=1.0 for areal densities (per unit area); the "
                "reference cell's Lz for volumetric densities averaged along z. There is "
                "no default")
        return None
    value = float(depth)
    if not (value > 0.0) or value == float("inf"):
        raise ValueError(f"depth must be finite and positive, got {depth!r}")
    return value


def check_m_stab(m_stab) -> str:
    """``m_stab`` as declared, or a refusal naming ``"mean"`` and ``"max"``."""
    if m_stab is UNDECLARED:
        raise ValueError(
            f"m_stab must be declared, one of {list(M_STAB_MODES)}: 'mean' "
            f"freezes the implicit operator at M(mean density), 'max' at the "
            f"initial field's largest-norm cell. There is no default: under "
            f"noise the frozen matrix sets the stationary distribution.")
    if m_stab not in M_STAB_MODES:
        raise ValueError(
            f"m_stab must be one of {list(M_STAB_MODES)}, got {m_stab!r}")
    return m_stab


def declared_noise(system) -> dict:
    """``{"noise_mode", "sigma_noise", "m_stab"}`` from ``system.noise``; ``sigma_noise`` is DERIVED from
    ``system.defaults["sigma"]`` (``None`` for ``"none"``). Raises if the system declares no noise."""
    noise = getattr(system, "noise", None)
    if noise is None:
        raise ValueError(
            f"system {system.name!r} declares no noise: add "
            f"noise=Noise(mode=..., m_stab=...) with mode one of "
            f"{list(NOISE_MODES)} and m_stab one of {list(M_STAB_MODES)}")
    sigma = None
    if noise.mode == "gaussian":
        if "sigma" not in system.defaults:
            raise ValueError(
                f"system {system.name!r} declares gaussian noise but no "
                f"defaults['sigma']: the filter width IS the coarse-graining "
                f"length, and there is no second knob for it")
        sigma = float(system.defaults["sigma"])
    check_noise_declaration(noise.mode, sigma)
    return {"noise_mode": noise.mode, "sigma_noise": sigma,
            "m_stab": check_m_stab(noise.m_stab)}
