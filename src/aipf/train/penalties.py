"""The two hinge penalties' probes: points drawn from the declared ``penalty_seed``, then ``L_conv``
on the system's trust domain and ``L_Gamma`` along its measured equation-of-state paths."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from aipf.data.rows import measured_rows
from aipf.losses.convexity import gamma_on_path, l_conv, l_gamma
from aipf.system import System, TrustDomain

from .anchors import EOS_COLUMNS, _columns, _curvature, _digest, _resolve, _w_hat_of
from .config import TrainConfig
from .sampling import T_MEASURES, sample_trust_domain

#: The keys a declared ``gamma_paths`` block carries.
GAMMA_PATH_KEYS: Tuple[str, ...] = (
    "eos_csvs", "eos_columns", "x_grid", "poly_degree", "T_measure", "q_floor",
)

#: The two terms this module feeds, in canonical order.
PENALTY_TERMS: Tuple[str, ...] = ("L_conv", "L_Gamma")


def _hessian(model, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """The symmetrised ``d2 f_loc / d rho2`` at ``(P,)`` points, ``(P, n, n)``, with a graph."""
    H = _curvature(model, rho, T)
    return 0.5 * (H + H.transpose(1, 2))


def _dtype_device(model):
    p = next(model.parameters())
    return p.dtype, p.device


@dataclass(frozen=True)
class ConvexityProbe:
    """``L_conv``: ``n`` states uniform in ``domain``, temperatures on its ``T_range`` by ``T_measure``."""

    domain: TrustDomain
    n: int
    T_measure: str
    form: str
    margin: float

    def __post_init__(self) -> None:
        if int(self.n) < 1:
            raise ValueError(
                f"conv_samples={self.n!r}: L_conv is trained, so it needs at "
                f"least one point a step")
        if self.T_measure not in T_MEASURES:
            raise ValueError(
                f"conv_T_measure={self.T_measure!r} is not one of {T_MEASURES}")

    def draw(self, generator: torch.Generator):
        return sample_trust_domain(int(self.n), self.domain,
                                   T_measure=self.T_measure,
                                   generator=generator)

    def loss(self, model, points) -> Tuple[torch.Tensor, dict]:
        rho, T = points
        dtype, device = _dtype_device(model)
        H = _hessian(model, rho.to(device=device, dtype=dtype),
                     T.to(device=device, dtype=dtype))
        return l_conv(H, form=self.form, margin=self.margin)


@dataclass(frozen=True)
class PathDraw:
    """One step's ``L_Gamma`` paths: a manifold index, a temperature and ``n(x)`` per path (float64)."""

    index: torch.Tensor
    T: torch.Tensor
    n_path: torch.Tensor


@dataclass(frozen=True)
class GammaPaths:
    """``L_Gamma``'s paths: ``n(x)`` per manifold (axis 0) and temperature node (axis 1) on ``x_grid``.
    ``x`` is the fraction of channel ``x_channel``; ``kB`` turns a temperature into ``kB T``."""

    n_table: torch.Tensor
    T_nodes: torch.Tensor
    x_grid: torch.Tensor
    x_channel: int
    kB: float
    n_paths: int
    T_measure: str
    q_floor: float
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, system: System, *, n_paths: int,
             eos_csvs: Mapping[Any, Any], eos_columns: Mapping[str, Any],
             x_grid: Tuple[float, float, float], poly_degree: int,
             T_measure: str, q_floor: float) -> "GammaPaths":
        """Per manifold, a degree-``poly_degree`` fit of ``n(x)`` at each temperature node, on
        ``arange(lo, hi + h/2, h)``; a manifold with a hole, or a different node set, is refused."""
        if system.n_species != 2:
            raise NotImplementedError(
                f"L_Gamma walks one composition axis, a path through the "
                f"simplex only at two channels; system {system.name!r} has "
                f"{system.n_species}")
        if int(n_paths) < 1:
            raise ValueError(
                f"gamma_pt_samples={n_paths!r}: L_Gamma is trained, so it "
                f"needs at least one path a step")
        if T_measure not in T_MEASURES:
            raise ValueError(
                f"gamma_paths T_measure={T_measure!r} is not one of "
                f"{T_MEASURES}")
        paths = _resolve(system.paths.raw(), eos_csvs, "gamma_paths")
        col = _columns(eos_columns, EOS_COLUMNS, "the declared gamma_paths columns")
        lo, hi, h = (float(v) for v in x_grid)
        x_np = np.arange(lo, hi + 0.5 * h, h)
        T_ref, tables = None, []
        for _value, path in paths:
            rows = np.asarray(list(measured_rows(path, **col)), dtype=float)
            xs, Ts, ns = rows[:, 0], rows[:, 1], rows[:, 2]
            xg, Tg = np.unique(xs), np.unique(Ts)
            grid = np.full((len(xg), len(Tg)), np.nan)
            for x, T, n in zip(xs, Ts, ns):
                grid[np.searchsorted(xg, x), np.searchsorted(Tg, T)] = n
            if np.isnan(grid).any():
                raise ValueError(
                    f"{path} has holes in its (x, T) grid; a fit per "
                    f"temperature node needs every node")
            if T_ref is None:
                T_ref = Tg
            elif not np.array_equal(Tg, T_ref):
                raise ValueError(
                    f"{path} has temperature nodes {Tg}, not the {T_ref} of "
                    f"the first manifold")
            table = np.empty((len(Tg), len(x_np)))
            for j in range(len(Tg)):
                table[j] = np.polynomial.Polynomial.fit(
                    xg, grid[:, j], int(poly_degree))(x_np)
            tables.append(table)
        return cls(
            n_table=torch.tensor(np.stack(tables), dtype=torch.float64),
            T_nodes=torch.tensor(T_ref, dtype=torch.float64),
            x_grid=torch.tensor(x_np, dtype=torch.float64),
            x_channel=int(system.table_keys["x_channel"]),
            kB=float(system.constants["kB"]), n_paths=int(n_paths),
            T_measure=str(T_measure), q_floor=float(q_floor),
            provenance={"eos_csvs": tuple((str(p), _digest(p))
                                          for _v, p in paths)})

    def draw(self, generator: torch.Generator) -> PathDraw:
        """A manifold uniformly, then a temperature by ``T_measure`` between the end nodes, ``n(x)``
        linear between the two bracketing node rows."""
        n_P, n_T, _ = self.n_table.shape
        S = int(self.n_paths)
        index = torch.randint(0, n_P, (S,), generator=generator)
        u = torch.rand(S, generator=generator).to(self.T_nodes.dtype)
        T_lo, T_hi = self.T_nodes[0], self.T_nodes[-1]
        if self.T_measure == "log_uniform":
            T = torch.exp(math.log(T_lo) + u * (math.log(T_hi) - math.log(T_lo)))
        else:
            T = T_lo + u * (T_hi - T_lo)
        j = torch.clamp(torch.searchsorted(self.T_nodes, T) - 1, 0, n_T - 2)
        Tj, Tj1 = self.T_nodes[j], self.T_nodes[j + 1]
        tT = (T - Tj) / (Tj1 - Tj)
        n_path = (self.n_table[index, j] * (1.0 - tT)[:, None]
                  + self.n_table[index, j + 1] * tT[:, None])
        return PathDraw(index=index, T=T, n_path=n_path)

    def points(self, draw: PathDraw, dtype: torch.dtype) -> Dict[str, Any]:
        """The flattened ``(S * n_x,)`` states of a draw in ``dtype``: ``rho`` ``(., 2)``, ``T``, ``kBT``,
        ``x``, ``n``; ``h`` the grid step; ``shape`` ``(S, n_x)``."""
        S, n_x = draw.n_path.shape
        n_path = draw.n_path.to(dtype)
        x_row = self.x_grid.to(dtype)
        T_col = draw.T.to(dtype)
        x = x_row.unsqueeze(0).expand(S, n_x)
        channel = [(1.0 - x) * n_path, (1.0 - x) * n_path]
        channel[self.x_channel] = x * n_path
        return {"rho": torch.stack(channel, dim=-1).reshape(-1, 2),
                "T": T_col.unsqueeze(-1).expand(S, n_x).reshape(-1),
                "kBT": (self.kB * T_col).unsqueeze(-1).expand(S, n_x).reshape(-1),
                "x": x.reshape(-1), "n": n_path.reshape(-1),
                "h": float(x_row[1] - x_row[0]), "shape": (S, n_x)}

    def loss(self, model, draw: PathDraw) -> Tuple[torch.Tensor, dict]:
        """``l_gamma`` of ``Gamma(x)`` from ``Hess f_loc + W_hat(0)`` along each drawn path."""
        dtype, device = _dtype_device(model)
        pts = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in self.points(draw, dtype).items()}
        H = (_hessian(model, pts["rho"], pts["T"])
             + _w_hat_of(model)(torch.zeros(1, dtype=dtype, device=device))[0])
        gamma = gamma_on_path(H, pts["n"], pts["x"], pts["kBT"],
                              x_channel=self.x_channel, eps=self.q_floor)
        return l_gamma(gamma.reshape(pts["shape"]), h=pts["h"])


class Penalties:
    """Both probes on one generator seeded by the declared ``penalty_seed``; ``conv`` draws first."""

    def __init__(self, seed: int, conv: Optional[ConvexityProbe],
                 gamma: Optional[GammaPaths]) -> None:
        if conv is None and gamma is None:
            raise ValueError("Penalties with neither probe trains nothing")
        self.seed = int(seed)
        self.conv = conv
        self.gamma = gamma
        self.generator = torch.Generator().manual_seed(self.seed)

    def terms(self) -> Tuple[str, ...]:
        fed = (self.conv is not None, self.gamma is not None)
        return tuple(name for name, on in zip(PENALTY_TERMS, fed) if on)

    def draw(self):
        """This step's points, ``(conv points or None, PathDraw or None)``."""
        c = None if self.conv is None else self.conv.draw(self.generator)
        g = None if self.gamma is None else self.gamma.draw(self.generator)
        return c, g

    def losses(self, model, draws=None):
        """``(conv pair or None, gamma pair or None)`` on ``draws``, or on a fresh :meth:`draw`."""
        c, g = self.draw() if draws is None else draws
        return (None if c is None else self.conv.loss(model, c),
                None if g is None else self.gamma.loss(model, g))

    def provenance(self) -> Dict[str, Any]:
        return {"penalty_seed": self.seed, "terms": list(self.terms()),
                "gamma": (None if self.gamma is None
                          else dict(self.gamma.provenance))}


def penalties_from_system(system: System, cfg: TrainConfig) -> Optional[Penalties]:
    """The probes this system declares for the weights ``cfg`` carries, or ``None``.
    ``L_conv`` needs ``system.trust_domain``; ``L_Gamma`` needs ``defaults["training"]["gamma_paths"]``."""
    block = system.defaults.get("training") or {}
    conv = None
    if float(cfg.lambda_conv) != 0.0 and system.trust_domain is not None:
        if "conv_T_measure" not in block:
            raise KeyError(
                f"system {system.name!r} trains L_conv on its trust domain and "
                f"declares no defaults['training']['conv_T_measure'], one of "
                f"{T_MEASURES}")
        conv = ConvexityProbe(domain=system.trust_domain,
                              n=int(cfg.conv_samples),
                              T_measure=str(block["conv_T_measure"]),
                              form=str(cfg.conv_penalty),
                              margin=float(cfg.conv_margin))
    gamma = None
    if float(cfg.lambda_gamma) != 0.0 and "gamma_paths" in block:
        declared = dict(block["gamma_paths"])
        missing = [k for k in GAMMA_PATH_KEYS if k not in declared]
        extra = sorted(set(declared) - set(GAMMA_PATH_KEYS))
        if missing or extra:
            raise KeyError(
                f"system {system.name!r} gamma_paths: missing {missing}, not "
                f"read {extra}")
        gamma = GammaPaths.load(system, n_paths=int(cfg.gamma_pt_samples),
                                **declared)
    if conv is None and gamma is None:
        return None
    if cfg.penalty_seed is None:
        fed = [n for n, probe in zip(PENALTY_TERMS, (conv, gamma)) if probe]
        raise KeyError(
            f"system {system.name!r} trains {fed} and declares no "
            f"defaults['training']['penalty_seed']: the probe points come "
            f"from that stream and nowhere else")
    return Penalties(cfg.penalty_seed, conv, gamma)


__all__ = ["ConvexityProbe", "GammaPaths", "PathDraw", "Penalties",
           "PENALTY_TERMS", "penalties_from_system"]
