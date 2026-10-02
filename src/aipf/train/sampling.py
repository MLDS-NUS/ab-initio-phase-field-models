"""Explicit-generator samplers, on a box and on a trust domain: ``generator`` is required, never the
global torch stream."""
from __future__ import annotations

import math
from typing import Sequence

import torch


def sample_uniform(n: int, low: Sequence[float], high: Sequence[float], *,
                    generator: torch.Generator,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(n, d)`` points uniform on the box ``[low, high]`` (each length ``d``), drawn from ``generator``."""
    if n < 0:
        raise ValueError(f"n={n!r} must be >= 0")
    low_t = torch.as_tensor(low, dtype=dtype)
    high_t = torch.as_tensor(high, dtype=dtype)
    if low_t.shape != high_t.shape:
        raise ValueError(
            f"low has shape {tuple(low_t.shape)}, high has shape "
            f"{tuple(high_t.shape)}: they must match")
    u = torch.rand(n, *low_t.shape, generator=generator, dtype=dtype,
                    device=generator.device)
    return low_t + (high_t - low_t) * u


#: Admissible temperature measures for a trust-domain draw.
T_MEASURES = ("uniform", "log_uniform")


def sample_trust_domain(n: int, domain, *, T_measure: str,
                        generator: torch.Generator) -> tuple:
    """``(rho (n, d), T (n,))`` float32: ``rho`` uniform in ``domain`` by rejection from ``[0, outer]^d``
    (``2n`` candidates a round), then ``T`` on ``domain.T_range`` by ``T_measure`` (:data:`T_MEASURES`)."""
    if T_measure not in T_MEASURES:
        raise ValueError(f"T_measure={T_measure!r} is not one of {T_MEASURES}")
    if domain.T_range is None:
        raise ValueError(
            "the trust domain declares no T_range, so there is no "
            "temperature to draw the penalty's points at")
    box = torch.tensor(domain.outer)
    chunks, have = [], 0
    while have < n:
        cand = torch.rand(2 * n, domain.n_species, generator=generator) * box
        inner = cand[:, 0] / domain.inner[0]
        outer = cand[:, 0] / domain.outer[0]
        for i in range(1, domain.n_species):
            inner = inner + cand[:, i] / domain.inner[i]
            outer = outer + cand[:, i] / domain.outer[i]
        keep = (inner >= 1.0) & (outer <= 1.0)
        chunks.append(cand[keep])
        have += int(keep.sum())
    rho = torch.cat(chunks)[:n]
    u = torch.rand(n, generator=generator)
    lo, hi = domain.T_range
    if T_measure == "log_uniform":
        T = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    else:
        T = lo + u * (hi - lo)
    return rho, T
