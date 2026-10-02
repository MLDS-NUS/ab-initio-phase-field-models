"""The trust-domain predicate: core owns the geometric test, never the numbers.
``rho_i >= 0``, ``sum_i rho_i / inner_i >= 1``, ``sum_i rho_i / outer_i <= 1``, plus an optional ``T_range``.
The experiment declares the :class:`~aipf.system.TrustDomain`; there is no default."""
from __future__ import annotations

from typing import Optional

import torch

from aipf.system import TrustDomain


def in_domain(rho: torch.Tensor, domain: Optional[TrustDomain],
              T: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Boolean mask ``rho.shape[:-1]``: is each ``(..., n_species)`` row inside ``domain``.
    ``domain=None`` raises. ``T`` is required exactly when ``domain.T_range`` is set."""
    if domain is None:
        raise ValueError(
            "in_domain requires a declared TrustDomain: core carries no "
            "default trapezoid corners. A system's "
            "trust domain belongs in that system's own experiment module, "
            "e.g. experiments/<name>/system.py, not in aipf.solve.")
    if rho.shape[-1] != domain.n_species:
        raise ValueError(
            f"rho has {rho.shape[-1]} channels but domain declares "
            f"{domain.n_species} (inner={domain.inner!r})")
    inner = torch.as_tensor(domain.inner, dtype=rho.dtype, device=rho.device)
    outer = torch.as_tensor(domain.outer, dtype=rho.dtype, device=rho.device)

    nonneg = (rho >= 0.0).all(dim=-1)
    inside_inner_edge = (rho / inner).sum(dim=-1) >= 1.0
    inside_outer_edge = (rho / outer).sum(dim=-1) <= 1.0
    mask = nonneg & inside_inner_edge & inside_outer_edge

    if domain.T_range is not None:
        if T is None:
            raise ValueError(
                f"domain declares T_range={domain.T_range!r} but no T was "
                f"given: this domain's density trapezoid is not the whole "
                f"predicate")
        lo, hi = domain.T_range
        T_t = torch.as_tensor(T, dtype=rho.dtype, device=rho.device)
        mask = mask & (T_t >= lo) & (T_t <= hi)
    return mask


def project_trust_domain(rho: torch.Tensor, domain: TrustDomain) -> torch.Tensor:
    """Euclidean projection of out-of-domain ``(..., n_species)`` rows onto the trapezoid boundary.
    Exact at ``n_species = 2`` (nearest of four edge segments); other counts raise."""
    if domain.n_species != 2:
        raise NotImplementedError(
            f"project_trust_domain's closed-form edge projection is exact "
            f"only at n_species=2 (a 4-edge trapezoid); domain has "
            f"n_species={domain.n_species}")
    inner_h, inner_k = domain.inner
    outer_h, outer_k = domain.outer
    verts = torch.tensor(
        [(inner_h, 0.0), (outer_h, 0.0), (0.0, outer_k), (0.0, inner_k)],
        dtype=rho.dtype, device=rho.device)
    a = verts
    b = verts.roll(-1, dims=0)

    mask = in_domain(rho, domain)
    if bool(mask.all()):
        return rho
    flat = rho.reshape(-1, 2)
    inside = mask.reshape(-1)
    out = flat.clone()
    p = flat[~inside]
    best_d = None
    best_q = None
    for i in range(4):
        ai, bi = a[i], b[i]
        ab = bi - ai
        t = ((p - ai) @ ab / (ab @ ab)).clamp(0.0, 1.0)
        q = ai + t.unsqueeze(-1) * ab
        d = ((p - q) ** 2).sum(-1)
        if best_d is None:
            best_d, best_q = d, q
        else:
            better = d < best_d
            best_d = torch.where(better, d, best_d)
            best_q = torch.where(better.unsqueeze(-1), q, best_q)
    out[~inside] = best_q
    return out.reshape(rho.shape)
