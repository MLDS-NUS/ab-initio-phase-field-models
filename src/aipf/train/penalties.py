"""The two hinge penalties' probes: points drawn from the declared ``penalty_seed``, then ``L_conv``
on the system's trust domain and ``L_Gamma`` along its measured equation-of-state paths.

``L_conv`` may instead be declared on the anchor tables' single-phase states (``conv_states="anchors"``,
:class:`StateStability`): ``lambda_min[d2 f_loc + W_hat(k)] >= conv_margin`` on each ``anchor_bulk``
row's active block, at every ``k`` of a declared line from ``0`` on. It draws nothing."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from aipf.data.rows import measured_rows
from aipf.losses.convexity import gamma_on_path, l_conv, l_conv_eigmin, l_gamma
from aipf.spectral import model_ndim, refuse_two_dimensions
from aipf.system import System, TrustDomain

from .anchors import (EOS_COLUMNS, AnchorTables, _columns, _curvature, _digest, _resolve,
                      _w_hat_of)
from .config import TrainConfig
from .sampling import T_MEASURES, sample_trust_domain

#: The keys a declared ``gamma_paths`` block carries.
GAMMA_PATH_KEYS: Tuple[str, ...] = (
    "eos_csvs", "eos_columns", "x_grid", "poly_degree", "T_measure", "q_floor",
)

#: The two terms this module feeds, in canonical order.
PENALTY_TERMS: Tuple[str, ...] = ("L_conv", "L_Gamma")

#: Where ``L_conv`` reads its states, declared as ``conv_states``: ``"trust_domain"`` (points drawn in
#: the system's trust domain, at ``k = 0``; the default) or ``"anchors"`` (the ``anchor_bulk`` rows, on
#: a declared wavenumber line).
CONV_STATES: Tuple[str, ...] = ("trust_domain", "anchors")

#: The declared (and saved) spelling of ``L_conv``'s source, kept in ``extra_experiment_config`` and
#: only when declared: ``conv_states`` (one of :data:`CONV_STATES`) and ``conv_k``, the line
#: ``(n_k, k_max)`` the ``"anchors"`` source reads (``linspace(0, k_max, n_k)``).
STATE_KEYS: Tuple[str, ...] = ("conv_states", "conv_k")


def declared_fields(declared: Mapping[str, Any]) -> Dict[str, Any]:
    """The :class:`TrainConfig` fields the ``L_conv`` source keys of one declaration give, ``{}`` when
    it declares none: ``extra_experiment_config`` holding whichever of :data:`STATE_KEYS` it declares,
    as :func:`aipf.train.kernel_hinge.declared_fields` keeps the hinge's shape."""
    keys = {key: declared[key] for key in STATE_KEYS if key in declared}
    if "conv_k" in keys:
        keys["conv_k"] = list(conv_line(keys["conv_k"]))
    return {"extra_experiment_config": keys} if keys else {}


def conv_line(value: Any) -> Tuple[int, float]:
    """A declared ``conv_k`` as ``(n_k, k_max)``: two finite numbers (a list, a tuple or a 1-D array),
    ``n_k >= 2`` a whole one (the line holds ``k = 0`` and ``k_max``) and ``k_max > 0``; anything else
    is refused naming the key (a mapping, a string, a fractional ``n_k``, a bool, a non-finite or
    non-numeric entry, a line too short)."""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence) \
            or len(value) != 2:
        raise ValueError(f"conv_k={value!r}: declare the line as (n_k, k_max), two numbers")
    n_k, k_max = value
    try:
        n, k = float(n_k), float(k_max)
    except (TypeError, ValueError) as error:
        raise ValueError(f"conv_k={value!r}: declare the line as (n_k, k_max), two numbers") \
            from error
    if isinstance(n_k, bool) or isinstance(k_max, bool) or not (math.isfinite(n)
                                                                and math.isfinite(k)):
        raise ValueError(f"conv_k={value!r}: declare the line as (n_k, k_max), two finite numbers")
    if n != int(n):
        raise ValueError(f"conv_k={value!r}: n_k={n_k!r} counts wavenumbers, a whole number")
    if n < 2:
        raise ValueError(
            f"conv_k={value!r}: the line runs from k = 0 to k_max, at least two wavenumbers")
    if not k > 0.0:
        raise ValueError(f"conv_k={value!r}: the line reaches a positive k_max")
    return int(n), k


def anchor_line(cfg: TrainConfig) -> Optional[Tuple[int, float]]:
    """The ``(n_k, k_max)`` an anchors-sourced ``L_conv`` of ``cfg`` reads, ``None`` when ``cfg`` trains
    no such term. Every refusal that needs no table is made here (the source's value, ``conv_k``
    without it, ``conv_k`` missing or malformed), so a caller can make them before anything is loaded."""
    states = conv_states(cfg)
    if float(cfg.lambda_conv) == 0.0 or states != "anchors":
        return None
    extra = cfg.extra_experiment_config
    if "conv_k" not in extra:
        raise KeyError(
            f"lambda_conv={cfg.lambda_conv!r} with conv_states='anchors' and no conv_k = "
            f"(n_k, k_max), the wavenumber line it reads: this package has no default for it")
    return conv_line(extra["conv_k"])


def conv_states(cfg: TrainConfig) -> str:
    """The declared ``conv_states`` of ``cfg``, ``"trust_domain"`` when it declares none; another
    value is refused, and so is a ``conv_k`` without the ``"anchors"`` source, which alone reads it."""
    extra = cfg.extra_experiment_config
    states = extra.get("conv_states", "trust_domain")
    if states not in CONV_STATES:
        raise ValueError(f"conv_states={states!r} is not one of {CONV_STATES}")
    if states != "anchors" and "conv_k" in extra:
        raise ValueError(
            f"conv_k={extra['conv_k']!r} is declared with conv_states={states!r}: the wavenumber "
            f"line is read by conv_states='anchors' alone, so it would change nothing")
    return states


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
        refuse_two_dimensions(model_ndim(model), "the convexity penalty")
        rho, T = points
        dtype, device = _dtype_device(model)
        H = _hessian(model, rho.to(device=device, dtype=dtype),
                     T.to(device=device, dtype=dtype))
        return l_conv(H, form=self.form, margin=self.margin)


#: One active pattern's rows: the active channels ``idx``, the rows that share them.
_Group = Tuple[torch.Tensor, torch.Tensor]


def _active_groups(active: torch.Tensor) -> Tuple[Tuple[_Group, ...], torch.Tensor, torch.Tensor]:
    """``active`` ``(P, n)`` bool grouped by pattern: ``(groups, rows, order)``, each group the active
    channels and the rows sharing them, ``rows`` ``(R,)`` the kept rows in row order and ``order`` the
    permutation from the groups' concatenation to it. A pattern with fewer than two active channels is
    left out. Computed once: the tables' masks never change."""
    groups, kept = [], []
    for pattern in torch.unique(active, dim=0):
        idx = torch.nonzero(pattern).flatten()
        if idx.numel() < 2:
            continue
        rows = torch.nonzero((active == pattern).all(-1)).flatten()
        groups.append((idx, rows))
        kept.append(rows)
    if not groups:
        return (), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    rows = torch.cat(kept)
    order = torch.argsort(rows)
    return tuple(groups), rows[order], order


def _active_eigmin(H: torch.Tensor, groups: Sequence[_Group], order: torch.Tensor) -> torch.Tensor:
    """``lambda_min`` of each kept row's active principal block of ``H`` ``(P, K, n, n)``, ``(R, K)`` in
    row order: every block is taken at its own size (no padding)."""
    lams = [torch.linalg.eigvalsh(H[rows][..., idx[:, None], idx])[..., 0] for idx, rows in groups]
    return torch.cat(lams)[order]


@dataclass(frozen=True)
class StateStability:
    """``L_conv`` on the anchor tables' single-phase states: ``lambda_min[d2 f_loc + W_hat(k)] >= margin``
    on each ``anchor_bulk`` row's active principal block, at ``n_k`` wavenumbers ``linspace(0, k_max,
    n_k)``. ``active`` ``(P, n)`` is the tables' ``bulk_active`` (``bulk_rho > 0`` at two channels).

    The ``k = 0`` slice is the bulk anchor's own ``H0`` (its ``W_hat(0)`` route included) and every other
    slice is the same curvature plus the ``W_hat(k)`` ``L_S`` reads (the kernel's evaluator), so the
    penalty evaluates the model nowhere ``AnchorTables.batch`` does not already."""

    n_k: int
    k_max: float
    form: str
    margin: float
    active: torch.Tensor = field(repr=False, compare=False)
    #: The active-pattern groups of ``active`` per device, built once (:func:`_active_groups`).
    _groups: Dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._groups["cpu"] = _active_groups(self.active.detach().cpu())
        if int(self.n_k) < 2:
            raise ValueError(
                f"conv_k n_k={self.n_k!r}: the line runs from k = 0 to k_max, at least two "
                f"wavenumbers")
        if not float(self.k_max) > 0.0:
            raise ValueError(f"conv_k k_max={self.k_max!r}: the line reaches a positive k_max")

    @classmethod
    def from_tables(cls, tables: AnchorTables, *, n_k: int, k_max: float, form: str,
                    margin: float) -> "StateStability":
        """The probe on ``tables``' ``anchor_bulk`` rows; tables with none, or whose every row has
        fewer than two species present, are refused."""
        active = (tables.bulk_active if tables.bulk_active is not None
                  else tables.bulk_rho > 0)
        if not int((active.sum(-1) >= 2).sum()):
            # a table-dependent refusal: made when the tables are loaded
            raise ValueError(
                f"conv_states='anchors' reads the anchor_bulk rows and these tables have "
                f"{int(tables.bulk_rho.shape[0])}, none with two species present: declare "
                f"conv_states='trust_domain', or tables that keep single-phase bulk rows")
        return cls(n_k=int(n_k), k_max=float(k_max), form=str(form), margin=float(margin),
                   active=active.detach().clone())

    def wavenumbers(self, *, dtype: torch.dtype,
                    device: torch.device | str | None = None) -> torch.Tensor:
        """``linspace(0, k_max, n_k)``, made in ``dtype``."""
        return torch.linspace(0.0, self.k_max, int(self.n_k), dtype=dtype, device=device)

    @staticmethod
    def check(model) -> None:
        """Refuse a model whose ``W_hat`` the line cannot read: a two-dimensional one, and a kernel
        evaluated as a lattice sum (it needs a grid and a box, which a wavenumber line has not)."""
        refuse_two_dimensions(model_ndim(model), "the stability penalty (conv_states='anchors')")
        if getattr(getattr(getattr(model, "kernel", None), "evaluator", None),
                   "reads_geometry", False):
            raise NotImplementedError(
                "conv_states='anchors' reads W_hat(k) on a wavenumber line, as L_S does, and this "
                "kernel's W_hat reads a grid and a box: declare conv_states='trust_domain'")

    def eigmin(self, model, bulk: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor,
                                                                torch.Tensor]:
        """``(lam (R, n_k), rows (R,), k (n_k,))`` on one step's ``anchor_bulk`` entry, built with
        ``AnchorTables.batch(..., curvature=True)``; ``H`` is symmetrised before its blocks are read."""
        if "curvature" not in bulk:
            raise KeyError(
                "conv_states='anchors' reads the bulk anchor's curvature: build the batch with "
                "AnchorTables.batch(model, kB, curvature=True)")
        local, H0 = bulk["curvature"], bulk["H0"]
        if local.shape[0] != self.active.shape[0]:
            raise ValueError(
                f"the anchor_bulk entry has {local.shape[0]} rows and this probe was built on "
                f"{self.active.shape[0]}: build it from the tables the batch came from")
        k = self.wavenumbers(dtype=local.dtype, device=local.device)
        W = _w_hat_of(model)(k[1:])                                   # (n_k - 1, n, n)
        H = torch.cat([H0.unsqueeze(1), local.unsqueeze(1) + W], dim=1)   # (P, n_k, n, n)
        H = 0.5 * (H + H.transpose(-1, -2))
        groups, rows, order = self.groups(H.device)
        return _active_eigmin(H, groups, order), rows, k

    def groups(self, device: torch.device | str) -> Tuple[Tuple[_Group, ...], torch.Tensor,
                                                          torch.Tensor]:
        """:func:`_active_groups` of ``active`` with every index on ``device``, moved there once."""
        key = str(torch.device(device))
        if key not in self._groups:
            groups, rows, order = self._groups["cpu"]
            self._groups[key] = (tuple((i.to(device), r.to(device)) for i, r in groups),
                                 rows.to(device), order.to(device))
        return self._groups[key]

    def loss(self, model, bulk: Mapping[str, Any]) -> Tuple[torch.Tensor, dict]:
        """:func:`~aipf.losses.convexity.l_conv_eigmin` over every row and wavenumber of the line."""
        lam, _rows, _k = self.eigmin(model, bulk)
        return l_conv_eigmin(lam, form=self.form, margin=self.margin)

    def provenance(self) -> Dict[str, Any]:
        """What ``MANIFEST.json`` records under ``penalties["conv_states"]``."""
        return {"states": "anchors", "n_k": int(self.n_k), "k_max": float(self.k_max),
                "rows": int(self._groups["cpu"][1].numel())}


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
        refuse_two_dimensions(model_ndim(model), "the Gamma-path penalty")
        dtype, device = _dtype_device(model)
        pts = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in self.points(draw, dtype).items()}
        H = (_hessian(model, pts["rho"], pts["T"])
             + _w_hat_of(model)(torch.zeros(1, dtype=dtype, device=device))[0])
        gamma = gamma_on_path(H, pts["n"], pts["x"], pts["kBT"],
                              x_channel=self.x_channel, eps=self.q_floor)
        return l_gamma(gamma.reshape(pts["shape"]), h=pts["h"])


class Penalties:
    """Both probes on one generator seeded by the declared ``penalty_seed``; ``conv`` draws first.
    A :class:`StateStability` ``conv`` draws nothing and reads the step's anchor batch instead; with no
    probe that draws, ``seed`` may be ``None``."""

    def __init__(self, seed: Optional[int],
                 conv: Optional[Union[ConvexityProbe, StateStability]],
                 gamma: Optional[GammaPaths]) -> None:
        if conv is None and gamma is None:
            raise ValueError("Penalties with neither probe trains nothing")
        self.conv = conv
        self.gamma = gamma
        if seed is None and (gamma is not None or isinstance(conv, ConvexityProbe)):
            raise ValueError("a probe that draws points needs a penalty_seed")
        self.seed = None if seed is None else int(seed)
        self.generator = (None if self.seed is None
                          else torch.Generator().manual_seed(self.seed))

    @property
    def reads_anchors(self) -> bool:
        """Whether ``L_conv`` reads the step's anchor batch (``conv_states="anchors"``)."""
        return isinstance(self.conv, StateStability)

    def terms(self) -> Tuple[str, ...]:
        fed = (self.conv is not None, self.gamma is not None)
        return tuple(name for name, on in zip(PENALTY_TERMS, fed) if on)

    def check(self, model) -> None:
        """Refuse, before training, a model the anchors-sourced ``L_conv`` cannot read."""
        if self.reads_anchors:
            self.conv.check(model)

    def draw(self):
        """This step's points, ``(conv points or None, PathDraw or None)``."""
        c = (None if self.conv is None or self.reads_anchors
             else self.conv.draw(self.generator))
        g = None if self.gamma is None else self.gamma.draw(self.generator)
        return c, g

    def losses(self, model, draws=None, *, anchor_batch: Optional[Mapping[str, Any]] = None):
        """``(conv pair or None, gamma pair or None)`` on ``draws``, or on a fresh :meth:`draw`; an
        anchors-sourced ``L_conv`` reads ``anchor_batch["anchor_bulk"]``, built with
        ``curvature=True``."""
        c, g = self.draw() if draws is None else draws
        if self.reads_anchors:
            if anchor_batch is None or "anchor_bulk" not in anchor_batch:
                raise ValueError(
                    "conv_states='anchors' reads this step's anchor_bulk entry and none was given: "
                    "pass anchor_batch=AnchorTables.batch(model, kB, curvature=True)")
            conv = self.conv.loss(model, anchor_batch["anchor_bulk"])
        else:
            conv = None if c is None else self.conv.loss(model, c)
        return conv, None if g is None else self.gamma.loss(model, g)

    def provenance(self) -> Dict[str, Any]:
        out = {"penalty_seed": self.seed, "terms": list(self.terms()),
               "gamma": (None if self.gamma is None
                         else dict(self.gamma.provenance))}
        if self.reads_anchors:
            out["conv_states"] = self.conv.provenance()
        return out


def penalties_from_system(system: System, cfg: TrainConfig,
                          anchors: Optional[AnchorTables] = None) -> Optional[Penalties]:
    """The probes this system declares for the weights ``cfg`` carries, or ``None``.
    ``L_conv`` needs ``system.trust_domain``, or with ``conv_states="anchors"`` the loaded ``anchors``
    (refused by name when there are none) and ``conv_k``; ``L_Gamma`` needs
    ``defaults["training"]["gamma_paths"]``."""
    block = system.defaults.get("training") or {}
    conv = None
    line = anchor_line(cfg)
    if line is not None:
        if not isinstance(anchors, AnchorTables):
            raise ValueError(
                f"system {system.name!r} trains L_conv with conv_states='anchors', which reads the "
                f"anchor_bulk rows, and this run loads no anchor tables (anchors=NO_ANCHORS, aipf "
                f"train --anchors none, or no defaults['training']['tables']): load them, or "
                f"declare conv_states='trust_domain'")
        n_k, k_max = line
        conv = StateStability.from_tables(anchors, n_k=n_k, k_max=k_max,
                                          form=str(cfg.conv_penalty),
                                          margin=float(cfg.conv_margin))
    elif float(cfg.lambda_conv) != 0.0 and system.trust_domain is not None:
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
    drawing = (isinstance(conv, ConvexityProbe), gamma is not None)
    if cfg.penalty_seed is None and any(drawing):
        fed = [n for n, probe in zip(PENALTY_TERMS, drawing) if probe]
        raise KeyError(
            f"system {system.name!r} trains {fed} and declares no "
            f"defaults['training']['penalty_seed']: the probe points come "
            f"from that stream and nowhere else")
    return Penalties(cfg.penalty_seed, conv, gamma)


__all__ = ["CONV_STATES", "ConvexityProbe", "GammaPaths", "PathDraw", "Penalties",
           "PENALTY_TERMS", "STATE_KEYS", "StateStability", "anchor_line", "conv_line",
           "conv_states", "declared_fields", "penalties_from_system"]
