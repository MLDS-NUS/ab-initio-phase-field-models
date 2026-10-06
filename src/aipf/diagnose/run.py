"""The diagnosis driver: a checkpoint in, a directory of measured curves out.

Output lands under the checkpoint's md5 digest. Every threshold is a required declaration
(the system's ``defaults["diagnose"]``); nothing here has a default. Stages and outputs:
``phase_diagram`` -> ``P<P>GPa/phase_diagram.npz``; ``dome`` -> ``dome_Pgrid.npz``;
``tc`` -> ``tc.json`` (runs ``dome``); ``kappa`` -> ``kappa.json``;
``stability_map`` -> ``stability_map.npz`` (``lambda_min`` and ``Gamma`` of ``H_tot(0)`` on each isobar);
``one_field_phase_diagram`` -> ``one_field_phase_diagram.npz`` (:mod:`.one_field`); plus ``summary.csv``."""
from __future__ import annotations

import copy
import csv as _csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from aipf.data.rows import measured_rows
from aipf.functional.build import build, factory_name
from aipf.system import Checkpoint, System
from aipf.train.ckpt_compat import CONFIG_SCHEMA_TAGS, NEW_CONFIG_SCHEMA_TAG
from aipf.train.checkpoint_formats import (kmodes_model_state_dict, load_kmodes_into,
                                           load_lightning_hparams_into)
from aipf.train.pressure import pressure_from_model

from . import kappa as _kappa
from . import one_field as _one_field
from . import thermo
from .stages import STAGES

_ARANGE_PAD = 1e-9


def _need(declared: Mapping[str, Any], key: str) -> Any:
    """One declared value, or a ``KeyError`` naming the missing declaration."""
    if key not in declared:
        raise KeyError(
            f"{key!r} is not declared. This module has no defaults: every "
            f"threshold of the diagnosis is a measured choice of one system "
            f"and belongs in that system's defaults['diagnose'], not here. "
            f"Declared here: {sorted(declared)}")
    return declared[key]


def grid(spec: Sequence[float]) -> np.ndarray:
    """``(lo, hi, step)`` as an inclusive grid; the command line expands ``LO,HI,STEP`` through this."""
    lo, hi, step = (float(v) for v in spec)
    return np.arange(lo, hi + _ARANGE_PAD, step)


def grid_n(spec: Sequence[float]) -> np.ndarray:
    """``(lo, hi, n)`` as ``np.linspace(lo, hi, n)``, both ends included; ``LO,HI,N`` on the command line."""
    lo, hi, n = spec
    if int(n) != n or int(n) < 2:
        raise ValueError(f"grid_n needs an integer count >= 2, got {n!r}")
    return np.linspace(float(lo), float(hi), int(n))


def declared_grid(spec: Any) -> np.ndarray:
    """A declared grid: ``(lo, hi, step)``, ``{"lo", "hi", "step"}`` (:func:`grid`) or
    ``{"lo", "hi", "n"}`` (:func:`grid_n`)."""
    if isinstance(spec, Mapping):
        if set(spec) == {"lo", "hi", "n"}:
            return grid_n((spec["lo"], spec["hi"], spec["n"]))
        if set(spec) == {"lo", "hi", "step"}:
            return grid((spec["lo"], spec["hi"], spec["step"]))
        raise ValueError(f"a declared grid mapping has keys lo, hi and one of "
                         f"step or n; got {sorted(spec)}")
    return grid(spec)


#: The stage whose temperature grid is declared, and where: ``defaults["diagnose"]["one_field"]["T_grid"]``.
_DECLARED_T_GRID_STAGE = "one_field_phase_diagram"
_DECLARED_T_GRID_KEY = "defaults['diagnose']['one_field']['T_grid']"


def request_T_grid(declared: Mapping[str, Any], stages: Sequence[str],
                   pressures: Sequence[float], T_grid: Optional[Sequence[float]],
                   *, override_declared: bool) -> tuple[Optional[np.ndarray], str]:
    """The grid the asked stages run on and where it came from (``"declared"``, ``"command line"``,
    ``"override"`` or ``"none"``); ``ValueError`` naming the missing or contradicted key.

    ``declared``: the system's ``defaults["diagnose"]`` block.
    ``stages``: the stages asked for (:data:`STAGES`).
    ``pressures``: the isobars asked for, in the manifolds' pressure unit (GPa for the shipped systems).
    ``T_grid``: the temperatures given on the command line, or ``None``.
    ``override_declared``: let ``T_grid`` replace a declared one-field grid it contradicts."""
    given = None if T_grid is None else np.asarray(T_grid, dtype=np.float64)
    for stage, what in (("phase_diagram", "which isobars to cut and where to cut them"),
                        ("stability_map", "which isobars to walk and where to cut them")):
        if stage in stages and (not len(pressures) or given is None):
            raise ValueError(
                f"the {stage} stage needs both `pressures` (--pressure) and "
                f"`T_grid` (--T-grid or --T-grid-n): {what} are the "
                f"question being asked, not a setting")
    if _DECLARED_T_GRID_STAGE not in stages:
        return given, ("none" if given is None else "command line")
    block = declared.get("one_field")
    spec = None if block is None else block.get("T_grid")
    if spec is None:
        if given is None:
            raise ValueError(
                f"the {_DECLARED_T_GRID_STAGE} stage needs a temperature grid: "
                f"declare {_DECLARED_T_GRID_KEY} (lo, hi and step or n), or pass "
                f"--T-grid / --T-grid-n")
        return given, "command line"
    own = declared_grid(spec)
    if given is None:
        return own, "declared"
    if given.shape == own.shape and np.allclose(given, own, rtol=0.0, atol=1e-12):
        return own, "declared"
    if not override_declared:
        raise ValueError(
            f"the temperature grid given ({len(given)} points, {given[0]:g} to "
            f"{given[-1]:g}) is not the declared {_DECLARED_T_GRID_KEY} = {dict(spec)} "
            f"({len(own)} points). The declaration is the one source: drop the "
            f"flag, or pass --override-declared to run on the given grid")
    return given, "override"


#: The binodal routes the ``phase_diagram`` stage admits for ``binodal_route``. ``thermo.binodal``
#: also has ``mu_roots``, which needs a ``mu`` and a symmetry point the isobaric ``g(x)`` of a
#: two-species system does not have; the one-field stage reads it (``one_field["readoff"]``).
PHASE_DIAGRAM_BINODAL_ROUTES = ("convex_hull",)


def check_diagnosable(system: System) -> None:
    """Refuse a functional built by a factory: the stages read a rung's own parts (``f_local``, ``kernel``,
    ``kB``), which such a model need not have."""
    functional = getattr(system, "functional", None)
    if functional is not None and getattr(functional, "factory", None) is not None:
        raise ValueError(
            f"system {system.name!r} declares its functional by a factory "
            f"({factory_name(functional.factory)}), and the diagnosis reads the parts of this "
            f"package's own rungs; a model this package does not define is diagnosed by "
            f"its own code")


def check_declared(declared: Mapping[str, Any], stages: Sequence[str]) -> None:
    """The declaration refusals made before anything is written; ``ValueError`` naming the key.

    ``phase_diagram``: ``binodal_route`` is one of :data:`PHASE_DIAGRAM_BINODAL_ROUTES`."""
    if "phase_diagram" in stages and "binodal_route" in declared:
        route = declared["binodal_route"]
        if route not in PHASE_DIAGRAM_BINODAL_ROUTES:
            raise ValueError(
                f"binodal_route={route!r} is not a route the phase_diagram stage runs: it "
                f"admits {PHASE_DIAGRAM_BINODAL_ROUTES}. 'mu_roots' needs mu and a symmetry "
                f"point, which an isobaric g(x) does not have; it is the one-field read-off "
                f"(defaults['diagnose']['one_field']['readoff'], stage one_field_phase_diagram)")


def resolve_checkpoint(system: System, ckpt: Any) -> Path:
    """The file to diagnose: a :class:`~aipf.system.Checkpoint` is digest-verified, a path only
    checked to exist."""
    if isinstance(ckpt, Checkpoint):
        return system.resolve_checkpoint()
    path = Path(ckpt)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    return path


def _state_dict(saved: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """The model's tensors: ``model_state_dict`` when the file carries this package's ``config_schema`` tag,
    else ``state_dict`` then ``model_state_dict``. A ``config_schema`` tag this package does not know is refused."""
    tag = saved.get("config_schema")
    if tag is not None and tag not in CONFIG_SCHEMA_TAGS:
        raise ValueError(
            f"the checkpoint declares config_schema={tag!r}, which this package does not read; "
            f"it reads {list(CONFIG_SCHEMA_TAGS)} and the untagged Lightning-hparams and k-modes layouts")
    if tag in CONFIG_SCHEMA_TAGS:
        if "model_state_dict" not in saved:
            raise KeyError(
                f"the checkpoint declares config_schema="
                f"{NEW_CONFIG_SCHEMA_TAG!r} and holds no "
                f"'model_state_dict'; it holds {sorted(saved)[:12]}")
        return saved["model_state_dict"]
    for key in ("state_dict", "model_state_dict"):
        if key in saved:
            return saved[key]
    raise KeyError(
        f"the checkpoint holds no model tensors under 'state_dict' or "
        f"'model_state_dict'; it holds {sorted(saved)[:12]}")


def load_model(system: System, path: Path) -> torch.nn.Module:
    """The system's declared functional (:func:`aipf.functional.build`) with this file's weights, in float64.

    Checkpoints this package wrote load directly; Lightning-hparams ones go through ``load_lightning_hparams_into``."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = _state_dict(saved)
    model = build(system)
    if _is_kmodes(saved):
        return load_kmodes_into(model, saved).double().eval()
    if set(state) <= set(model.state_dict()):
        merged = dict(model.state_dict())
        merged.update(state)
        model.load_state_dict(merged, strict=True)
    else:
        load_lightning_hparams_into(model, state)
    return model.double().eval()


def _is_kmodes(saved: Mapping[str, Any]) -> bool:
    """A k-modes checkpoint: every tensor has a k-modes rename rule."""
    if saved.get("config_schema") in CONFIG_SCHEMA_TAGS:
        return False
    try:
        kmodes_model_state_dict(saved)
    except (KeyError, TypeError):
        return False
    return True


def load_manifold(path: Path, *, x_column: str, T_column: str,
                  n_columns: Sequence[str], status_column: str,
                  status_ok: str):
    """A measured ``n(x, T)`` table as ``(x_grid, T_grid, n)``, via
    :func:`aipf.data.rows.measured_rows`; holes raise.

    ``path``: the table (CSV). ``x_column``, ``T_column``: the composition and temperature columns.
    ``n_columns``: alternative number-density columns, the first present wins.
    ``status_column``, ``status_ok``: the column a row's status is in, and the value that keeps it."""
    rows = list(measured_rows(path, x_column=x_column, T_column=T_column,
                              n_columns=n_columns,
                              status_column=status_column,
                              status_ok=status_ok))
    xs, Ts, ns = map(np.asarray, zip(*rows) if rows else ((), (), ()))
    x_grid, T_grid = np.unique(xs), np.unique(Ts)
    grid = np.full((len(x_grid), len(T_grid)), np.nan)
    for x, T, n in zip(xs, Ts, ns):
        grid[np.searchsorted(x_grid, x), np.searchsorted(T_grid, T)] = n
    if np.isnan(grid).any():
        raise ValueError(
            f"{path}: the manifold has holes, and the density path is fitted "
            f"along a whole temperature column")
    return x_grid, T_grid, grid


#: How a manifold's ``n(x)`` is fitted: ``"column"`` blends two temperature columns then fits (holes
#: refused); ``"per_row"`` fits each column on its own measured nodes, then blends the two fits.
MANIFOLD_FITS = ("column", "per_row")


class RowFitManifold(tuple):
    """``(x_grid, T_grid, n)`` with NaN holes, fitted per temperature row by :func:`manifold_density`."""


def load_manifold_rows(path: Path, *, x_column: str, T_column: str,
                       n_columns: Sequence[str], status_column: str,
                       status_ok: str, x_channel: int,
                       min_nodes: int) -> RowFitManifold:
    """A measured ``n(x, T)`` table that may have holes; ``x`` becomes the channel-1 fraction.

    A temperature row with fewer than ``min_nodes`` measured compositions is dropped.
    ``path``, ``x_column``, ``T_column``, ``n_columns``, ``status_column``, ``status_ok``: as :func:`load_manifold`.
    ``x_channel``: the channel the table's composition column is the fraction of.
    ``min_nodes``: the measured compositions a temperature row needs."""
    rows = list(measured_rows(path, x_column=x_column, T_column=T_column,
                              n_columns=n_columns,
                              status_column=status_column,
                              status_ok=status_ok))
    xs = np.asarray([_channel_one(x, x_channel) for x, _T, _n in rows])
    Ts = np.asarray([T for _x, T, _n in rows])
    ns = np.asarray([n for _x, _T, n in rows])
    x_grid, T_grid = np.unique(xs), np.unique(Ts)
    grid = np.full((len(x_grid), len(T_grid)), np.nan)
    for x, T, n in zip(xs, Ts, ns):
        grid[np.searchsorted(x_grid, x), np.searchsorted(T_grid, T)] = n
    thick = np.isfinite(grid).sum(axis=0) >= int(min_nodes)
    if not thick.any():
        raise ValueError(
            f"{path}: no temperature row has {min_nodes} measured nodes")
    return RowFitManifold((x_grid, T_grid[thick], grid[:, thick]))


def _channel_one(x: float, x_channel: int) -> float:
    """A two-channel composition label as the fraction of channel 1."""
    if int(x_channel) == 1:
        return x
    if int(x_channel) == 0:
        return 1.0 - x
    raise ValueError(
        f"x_channel={x_channel!r}: a two-channel composition labels channel "
        f"0 or channel 1")


def _row_fit(x_grid, column, poly_degree: int):
    """One temperature row fitted on its finite nodes, over the whole grid's domain."""
    m = np.isfinite(column)
    if int(m.sum()) < int(poly_degree) + 1:
        raise ValueError(
            f"a manifold row has {int(m.sum())} nodes and a degree-"
            f"{poly_degree} fit needs {int(poly_degree) + 1}")
    return np.polynomial.Polynomial.fit(
        x_grid[m], column[m], int(poly_degree),
        domain=[float(x_grid[0]), float(x_grid[-1])])


def manifold_density(manifold, x, T: float, *, poly_degree: int) -> np.ndarray:
    """``n(x)`` at ``T``: linear in ``T`` between columns, then a degree-``poly_degree`` global
    polynomial fit in ``x`` (a :class:`RowFitManifold`: fit each column, then blend the fits).

    ``manifold``: a measured table (:func:`load_manifold` or :func:`load_manifold_rows`).
    ``x``: channel-1 fractions. ``T``: the temperature, in the table's unit (K).
    ``poly_degree``: the degree of the fit in ``x``."""
    x_grid, T_grid, grid = manifold
    if isinstance(manifold, RowFitManifold):
        if len(T_grid) == 1:
            return _row_fit(x_grid, grid[:, 0], poly_degree)(x)
        j = int(np.clip(np.searchsorted(T_grid, T) - 1, 0, len(T_grid) - 2))
        t = (T - T_grid[j]) / (T_grid[j + 1] - T_grid[j])
        lo = _row_fit(x_grid, grid[:, j], poly_degree)
        hi = _row_fit(x_grid, grid[:, j + 1], poly_degree)
        return (lo * (1.0 - t) + hi * t)(x)
    j = int(np.clip(np.searchsorted(T_grid, T) - 1, 0, len(T_grid) - 2))
    t = (T - T_grid[j]) / (T_grid[j + 1] - T_grid[j])
    column = grid[:, j] * (1 - t) + grid[:, j + 1] * t
    return np.polynomial.Polynomial.fit(x_grid, column, poly_degree)(x)


def _states(x: np.ndarray, n: np.ndarray, n_species: int) -> np.ndarray:
    """``(len(x), 2)`` densities ``[(1 - x) n, x n]`` on a composition path; two species only."""
    if n_species != 2:
        raise NotImplementedError(
            f"the isobar construction walks ONE composition axis, which is a "
            f"path through a {n_species}-component simplex only when there "
            f"are two components; this system has {n_species}. A one-field "
            f"system is diagnosed by the one_field_phase_diagram stage")
    return np.stack([(1.0 - x) * n, x * n], axis=1)


def _tensor(rho: np.ndarray):
    r = torch.tensor(np.ascontiguousarray(rho), dtype=torch.float64)
    return r, torch.ones(r.shape[0], 3, dtype=torch.float64)


#: How :func:`model_pressure` evaluates the Euler pressure: ``"protocol"`` (the model protocol's
#: ``chemical_potential`` on single-point grids, the isobar's scale rows in one call) or
#: ``"closed_form"`` (a pair-kernel rung's ``mu_local . rho - f_local + 1/2 rho . W_hat(0) rho``, one
#: call per scale row). Equal in exact arithmetic, not in the last bit, and a bisection reads that bit.
PRESSURE_ROUTES = ("protocol", "closed_form")


def _w_hat_zero(model, dtype, what: str) -> torch.Tensor:
    """``W_hat(0)`` ``(n, n)`` of a pair-kernel rung whose evaluator reads ``|k|`` alone; refuses otherwise."""
    if getattr(model.kernel.evaluator, "reads_geometry", False):
        raise NotImplementedError(
            f"{what} reads W_hat(0) at a bare |k| = 0, and this model's kernel "
            f"evaluator (kernel_evaluator='lattice_sum') has no value there "
            f"without a grid and a box: {what} under kernel_evaluator="
            f"'lattice_sum' is not supported")
    return model.kernel.w_hat(torch.zeros(1, dtype=dtype))[0]


def model_pressure(model, rho: np.ndarray, T: float, *,
                   route: str) -> np.ndarray:
    """``P(rho)`` in the model's own energy-per-volume unit, by the declared ``route`` (:data:`PRESSURE_ROUTES`).

    ``model``: the functional (``"closed_form"`` needs a pair-kernel one). ``rho``: uniform states,
    ``(P, n_species)``, in the model's density unit. ``T``: the temperature (K). ``route``: one of
    :data:`PRESSURE_ROUTES`."""
    if route == "closed_form":
        r = torch.tensor(np.ascontiguousarray(rho), dtype=torch.float64)
        kBT = torch.full((r.shape[0],), model.kB * float(T),
                         dtype=torch.float64)
        mu_local = model.f_local.mu_pointwise(r, kBT)
        with torch.no_grad():
            f_local = model.f_local.f_pointwise(r, kBT)
            w0 = _w_hat_zero(model, torch.float64, "the closed-form pressure")
            quad = 0.5 * torch.einsum("ni,ij,nj->n", r, w0, r)
            return ((mu_local * r).sum(-1) - f_local + quad).numpy()
    if route != "protocol":
        raise ValueError(
            f"pressure route {route!r} is not one of {PRESSURE_ROUTES}")
    r, boxes = _tensor(rho)
    T_vec = torch.full((r.shape[0],), float(T), dtype=torch.float64)
    with torch.no_grad():
        return pressure_from_model(model, r, boxes, T_vec).numpy()


def bulk_free_energy(model, rho: np.ndarray, T: float) -> np.ndarray:
    """``f(rho) = f_local + 1/2 (mu - mu_local) . rho`` per unit volume, including the field-wide
    (quadratic) term."""
    if getattr(model, "kernel_centre", None) is not None:
        raise NotImplementedError(
            "bulk_free_energy adds 1/2 (mu - mu_local) . rho, the pair term of a "
            "kernel that convolves rho; this model's kernel convolves rho - "
            "rho_ref (kernel_argument='difference'), where that sum is a "
            "different number. bulk_free_energy under kernel_argument="
            "'difference' is not supported; NonlocalKernel.uniform_free_energy "
            "is the uniform curve of that form")
    r, boxes = _tensor(rho)
    n_points, n_species = r.shape
    T_vec = torch.full((n_points,), float(T), dtype=torch.float64)
    grid = r.reshape(n_points, n_species, 1, 1, 1)
    with torch.no_grad():
        mu = model.chemical_potential(grid, boxes, T_vec).reshape(
            n_points, n_species)
    leaf = grid.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        mu_local = torch.autograd.grad(
            model.bulk_free_energy_density(leaf, T_vec).sum(), leaf)[0]
    mu_local = mu_local.reshape(n_points, n_species).detach()
    with torch.no_grad():
        f_local = model.bulk_free_energy_density(grid, T_vec).reshape(n_points)
    return (f_local + 0.5 * ((mu - mu_local) * r).sum(-1)).numpy()


def isobar_scale(model, P: float, x: np.ndarray, T: float, *, manifold,
                 pressure_unit: float, poly_degree: int, s_lo: float,
                 s_hi: float, n_scan: int, n_iter: int,
                 n_species: int, pressure_route: str) -> np.ndarray:
    """``s(x)`` with ``P_model(s n_manifold(x), T) = P``; NaN where the window has no rising (stable) root.

    ``model``: the functional. ``P``: the isobar, in the unit one ``pressure_unit`` of model pressure is.
    ``x``: channel-1 fractions along the path. ``T``: the temperature (K).
    ``manifold``: the measured ``n(x, T)`` table the scale multiplies.
    ``pressure_unit``: one unit of ``P`` in the model's energy-per-volume unit.
    ``poly_degree``: the manifold fit's degree in ``x``.
    ``s_lo``, ``s_hi``: the scale window searched, as multiples of the manifold density.
    ``n_scan``: points of the bracketing scan over the window.
    ``n_iter``: bisection steps after the scan.
    ``n_species``: the channel count (two).
    ``pressure_route``: one of :data:`PRESSURE_ROUTES`."""
    P_target = float(P) * float(pressure_unit)
    n_eos = manifold_density(manifold, x, T, poly_degree=poly_degree)

    def residual(s: np.ndarray) -> np.ndarray:
        """``P_model - P`` on one or several scale rows."""
        s = np.atleast_2d(np.asarray(s, dtype=np.float64))
        if pressure_route == "closed_form":
            _states(x, n_eos, n_species)  # refuses anything but two channels
            # ((1 - x) n) s, one call per row: the rounding the published isobars carry
            return np.stack([
                model_pressure(model, np.stack([(1.0 - x) * n_eos * row,
                                                x * n_eos * row], axis=1),
                               T, route=pressure_route) - P_target
                for row in s])
        rho = np.concatenate([_states(x, n_eos * row, n_species)
                              for row in s], axis=0)
        out = model_pressure(model, rho, T, route=pressure_route) - P_target
        return out.reshape(s.shape)

    s_scan = np.linspace(float(s_lo), float(s_hi), int(n_scan))
    R = residual(np.repeat(s_scan[:, None], len(x), axis=1))
    left, right = R[:-1], R[1:]
    bracket = (left * right <= 0.0) & ~((left == 0.0) & (right == 0.0))
    rising = right > left
    mid = 0.5 * (s_scan[:-1] + s_scan[1:])
    # Any rising bracket beats every falling one; nearest to the manifold wins among them.
    cost = np.abs(mid[:, None] - 1.0) + np.where(rising, 0.0, 10.0)
    cost[~bracket] = np.inf
    cost[:, ~(bracket & rising).any(axis=0)] = np.inf

    lo = np.full(len(x), np.nan)
    hi = np.full(len(x), np.nan)
    ok = np.isfinite(cost.min(axis=0))
    idx = cost.argmin(axis=0)
    lo[ok] = s_scan[:-1][idx[ok]]
    hi[ok] = s_scan[1:][idx[ok]]
    for _ in range(int(n_iter)):
        mid_s = 0.5 * (lo + hi)
        r_mid, r_lo = residual(np.stack([
            np.where(np.isfinite(mid_s), mid_s, 1.0),
            np.where(np.isfinite(lo), lo, 1.0)]))
        crossed = (r_lo * r_mid) <= 0.0
        hi = np.where(crossed, mid_s, hi)
        lo = np.where(crossed, lo, mid_s)
    return 0.5 * (lo + hi)


def isobar_density(model, P: float, x, T: float, *, manifold,
                   pressure_unit: float, poly_degree: int, s_lo: float,
                   s_hi: float, n_scan: int, n_iter: int,
                   n_species: int, pressure_route: str) -> np.ndarray:
    """``n(x) = s(x) n_manifold(x)`` on the model's own ``P`` isobar at ``T``; raises if no point has a state.

    Every parameter as :func:`isobar_scale`."""
    x = np.asarray(x, dtype=np.float64)
    s = isobar_scale(model, P, x, T, manifold=manifold,
                     pressure_unit=pressure_unit, poly_degree=poly_degree,
                     s_lo=s_lo, s_hi=s_hi, n_scan=n_scan, n_iter=n_iter,
                     n_species=n_species, pressure_route=pressure_route)
    n = manifold_density(manifold, x, T, poly_degree=poly_degree) * s
    n = np.where(np.isfinite(n) & (n > 0.0), n, np.nan)
    if not np.isfinite(n).any():
        raise ValueError(
            f"the model has no state at P = {float(P):g} anywhere on the "
            f"T = {float(T):g} isotherm, over a density scan window of "
            f"[{s_lo:g}, {s_hi:g}] around the manifold. Either the target "
            f"pressure is not the one the manifold was built at, or this "
            f"model carries no absolute-pressure anchor")
    return n


def _ties(x: np.ndarray, g: np.ndarray, min_gap: float):
    """Lower-hull index pairs of ``g(x)`` spanning more than ``min_gap`` (the ties)."""
    hull = thermo.lower_hull_indices(x, g)
    return [(a, b) for a, b in zip(hull[:-1], hull[1:])
            if x[b] - x[a] > min_gap]


def tangent_free_energy(model, x: np.ndarray, n: np.ndarray, T: float,
                        P: float, *, pressure_unit: float,
                        n_species: int) -> np.ndarray:
    """``g = (f + P)/n`` along a solved isobar, whose double tangent is the coexistence condition.

    ``model``: the functional. ``x``: channel-1 fractions. ``n``: the total number density on the isobar.
    ``T``: the temperature (K). ``P``: the isobar, in ``pressure_unit``'s unit.
    ``pressure_unit``: one unit of ``P`` in the model's energy-per-volume unit.
    ``n_species``: the channel count (two)."""
    rho = _states(x, n, n_species)
    f = bulk_free_energy(model, rho, T)
    return (f + float(P) * float(pressure_unit)) / n


def binodal_on_isobar(model, P: float, T: float, x: np.ndarray, *, manifold,
                      pressure_unit: float, poly_degree: int, s_lo: float,
                      s_hi: float, n_scan: int, n_iter: int,
                      min_path_points: int, min_tie_gap: float,
                      central_lo_max: float, central_width_max: float,
                      n_species: int, pressure_route: str):
    """``(x_lo, x_hi, max|s - 1|)`` at ``(P, T)``: the widest central tie (``x_lo < central_lo_max``,
    width ``< central_width_max``) of ``g`` on the isobar, else the widest tie; NaN pair if none.

    ``model``: the functional. ``P``: the isobar, in ``pressure_unit``'s unit. ``T``: the temperature (K).
    ``x``: the channel-1 fractions the isobar is solved on.
    ``manifold``, ``pressure_unit``, ``poly_degree``, ``s_lo``, ``s_hi``, ``n_scan``, ``n_iter``: as
    :func:`isobar_scale`.
    ``min_path_points``: solved points below which the isobar gives no binodal.
    ``min_tie_gap``: the narrowest composition gap a lower-hull facet must span to count as a tie.
    ``central_lo_max``: a central tie's left end lies below this fraction.
    ``central_width_max``: a central tie is narrower than this.
    ``n_species``: the channel count (two). ``pressure_route``: one of :data:`PRESSURE_ROUTES`."""
    s = isobar_scale(model, P, x, T, manifold=manifold,
                     pressure_unit=pressure_unit, poly_degree=poly_degree,
                     s_lo=s_lo, s_hi=s_hi, n_scan=n_scan, n_iter=n_iter,
                     n_species=n_species, pressure_route=pressure_route)
    ok = np.isfinite(s)
    if int(ok.sum()) < int(min_path_points):
        return np.nan, np.nan, np.nan
    xs = x[ok]
    n = manifold_density(manifold, xs, T, poly_degree=poly_degree) * s[ok]
    g = tangent_free_energy(model, xs, n, T, P, pressure_unit=pressure_unit,
                            n_species=n_species)
    tilt = float(np.nanmax(np.abs(s - 1.0)))
    ties = _ties(xs, g, float(min_tie_gap))
    if not ties:
        return np.nan, np.nan, tilt
    central = [(a, b) for a, b in ties
               if xs[a] < central_lo_max
               and (xs[b] - xs[a]) < central_width_max]
    a, b = max(central or ties, key=lambda ab: xs[ab[1]] - xs[ab[0]])
    return float(xs[a]), float(xs[b]), tilt


def nearest_anchor(P: float, anchors: Sequence[float]) -> float:
    """The measured manifold pressure nearest ``P``; an exact midpoint breaks toward the higher one."""
    return float(min(anchors, key=lambda a: (abs(P - a), -a)))


def _apex(declared: Mapping[str, Any], T_grid: np.ndarray, bin_lo: np.ndarray,
          bin_hi: np.ndarray, T_step: float):
    """``(T_c, w0, T_last)`` by the declared ``tc_method``; non-apex methods return NaN
    ``w0``/``T_last``."""
    method = _need(declared, "tc_method")
    if method == "dome_apex":
        return thermo.dome_apex(T_grid, bin_lo, bin_hi, T_step=T_step,
                                **dict(_need(declared, "apex")))
    rows = {float(T): (lo, hi) for T, lo, hi in zip(T_grid, bin_lo, bin_hi)}

    def at(T: float):
        lo, hi = rows[float(T)]
        return None if not np.isfinite(lo) else (float(lo), float(hi))

    nan = np.float64(np.nan)
    try:
        T_c = thermo.critical_temperature(
            at, T_grid, method=method,
            **dict(declared.get("tc_kwargs", {})))
    except ValueError:
        return nan, nan, nan
    return np.float64(T_c), nan, nan


def manifolds(system: System, declared: Mapping[str, Any]):
    """Every declared manifold, loaded once and keyed by its pressure. A relative path is read from the
    system's tracked tables (``experiments/<system>/eos``) first, then from its raw root."""
    from aipf.paths import tracked_eos

    columns = dict(_need(declared, "eos_columns"))
    fit = _need(declared, "manifold_fit")
    if fit not in MANIFOLD_FITS:
        raise ValueError(f"manifold_fit={fit!r} is not one of {MANIFOLD_FITS}")
    x_channel = int(system.table_keys["x_channel"])
    tracked = tracked_eos(system.name)
    out = {}
    for pressure, relative in dict(_need(declared, "eos_csvs")).items():
        path = Path(relative)
        if path.is_absolute():
            raise ValueError(
                f"the manifold declared at {pressure} is an absolute path; "
                f"locations are resolved under the system's own root so that "
                f"the environment can move it")
        # the tracked copy first; the raw root is asked for only when there is none
        root = tracked if (tracked / path).is_file() else system.paths.raw()
        if fit == "per_row":
            out[float(pressure)] = load_manifold_rows(
                root / path, x_channel=x_channel,
                min_nodes=int(_need(declared, "manifold_min_nodes")),
                **columns)
            continue
        if x_channel != 1:
            raise NotImplementedError(
                "manifold_fit='column' reads the composition column as the "
                "channel-1 fraction; a channel-0 label is read by 'per_row'")
        out[float(pressure)] = load_manifold(root / path, **columns)
    return out


def _dome_scan(model, system: System, declared: Mapping[str, Any],
               out_dir: Path) -> dict:
    """The dense pressure scan, and its apex per slice."""
    spec = dict(_need(declared, "dome"))
    by_pressure = manifolds(system, declared)
    anchors = tuple(float(a) for a in spec["anchor_pressures"])
    T_grid = declared_grid(spec["T_grid"])
    P_grid = declared_grid(spec["pressures"])
    x = declared_grid(spec["x_grid"])
    close_after = int(spec["close_after"])
    isobar = dict(_need(declared, "isobar"))
    T_step = float(T_grid[1] - T_grid[0]) if len(T_grid) > 1 else np.nan

    bin_lo = np.full((len(P_grid), len(T_grid)), np.nan)
    bin_hi = np.full((len(P_grid), len(T_grid)), np.nan)
    Tc_fit = np.full(len(P_grid), np.nan)
    w0 = np.full(len(P_grid), np.nan)
    T_last = np.full(len(P_grid), np.nan)
    anchor_P = np.full(len(P_grid), np.nan)

    for pi, P in enumerate(P_grid):
        anchor = nearest_anchor(float(P), anchors)
        anchor_P[pi] = anchor
        manifold = by_pressure[anchor]
        blank = 0
        for ti, T in enumerate(T_grid):
            lo, hi, _tilt = binodal_on_isobar(
                model, float(P), float(T), x, manifold=manifold,
                pressure_unit=float(_need(declared, "pressure_unit")),
                poly_degree=int(_need(declared, "poly_degree")),
                min_path_points=int(_need(declared, "min_path_points")),
                min_tie_gap=float(_need(declared, "min_tie_gap")),
                central_lo_max=float(_need(declared, "central_lo_max")),
                central_width_max=float(_need(declared, "central_width_max")),
                n_species=system.n_species, **isobar)
            bin_lo[pi, ti], bin_hi[pi, ti] = lo, hi
            blank = 0 if np.isfinite(lo) else blank + 1
            if blank >= close_after:
                break
        Tc_fit[pi], w0[pi], T_last[pi] = _apex(
            declared, T_grid, bin_lo[pi], bin_hi[pi], T_step)

    # Not named ``grid``: that would shadow the module-level ``grid()`` in this body.
    scan = dict(T=T_grid, P_grid=P_grid, bin_lo=bin_lo, bin_hi=bin_hi,
                Tc_fit=Tc_fit, w0=w0, T_last=T_last, anchor_P=anchor_P)
    np.savez(out_dir / "dome_Pgrid.npz", **scan)
    return scan


def _phase_diagram(model, system: System, pressures: Sequence[float],
                   T_grid: np.ndarray, declared: Mapping[str, Any],
                   out_dir: Path) -> list:
    """One isobaric slice per requested pressure, under fixed key names."""
    by_pressure = manifolds(system, declared)
    isobar = dict(_need(declared, "isobar"))
    unit = float(_need(declared, "pressure_unit"))
    degree = int(_need(declared, "poly_degree"))
    route = _need(declared, "binodal_route")
    check_declared(declared, ("phase_diagram",))
    x_bar = _need(declared, "x_bar")
    x_bar = x_bar if x_bar == "auto" else float(x_bar)  # "auto": thermo.binodal_convex_hull
    x = declared_grid(_need(declared, "x_grid"))
    iso_x = declared_grid(_need(declared, "iso_x_grid"))
    min_tie_gap = float(_need(declared, "min_tie_gap"))
    T_step = float(T_grid[1] - T_grid[0]) if len(T_grid) > 1 else np.nan
    written = []

    for P in pressures:
        manifold = by_pressure[float(P)]
        n_T = len(T_grid)
        out = {key: np.full(n_T, np.nan) for key in
               ("bin_lo", "bin_hi", "bin1d_lo", "bin1d_hi",
                "spmat_lo", "spmat_hi")}
        iso_s = np.full((n_T, len(iso_x)), np.nan)
        for ti, T in enumerate(T_grid):
            s = isobar_scale(model, float(P), x, float(T), manifold=manifold,
                             pressure_unit=unit, poly_degree=degree,
                             n_species=system.n_species, **isobar)
            iso_s[ti] = isobar_scale(
                model, float(P), iso_x, float(T), manifold=manifold,
                pressure_unit=unit, poly_degree=degree,
                n_species=system.n_species, **isobar)
            ok = np.isfinite(s)
            if int(ok.sum()) < int(_need(declared, "min_path_points")):
                continue
            xs = x[ok]
            n = manifold_density(manifold, xs, float(T),
                                 poly_degree=degree) * s[ok]
            g = tangent_free_energy(model, xs, n, float(T), float(P),
                                    pressure_unit=unit,
                                    n_species=system.n_species)

            # the declared route, under the declared lever anchor
            tie = thermo.binodal(route, T=float(T),
                                 f=lambda xx, _T, _x=xs, _g=g: float(
                                     np.interp(xx, _x, _g)),
                                 grid=xs, anchor=x_bar)
            if tie is not None:
                out["bin_lo"][ti], out["bin_hi"][ti] = tie

            # the widest tie overlapping the concave stretch
            slope = np.gradient(g, xs)
            ties = _ties(xs, g, min_tie_gap)
            if ties:
                curvature = np.gradient(slope, xs)
                concave = np.where(curvature < 0.0)[0]
                pool = ties
                if concave.size:
                    lo_x, hi_x = xs[concave[0]], xs[concave[-1]]
                    pool = [t for t in ties
                            if xs[t[0]] <= hi_x and xs[t[1]] >= lo_x] or ties
                a, b = max(pool, key=lambda ab: xs[ab[1]] - xs[ab[0]])
                out["bin1d_lo"][ti], out["bin1d_hi"][ti] = xs[a], xs[b]

            # the linearly unstable interval, from the same curve
            roots = thermo.spinodal(
                lambda xx, _T, _x=xs, _d=slope: float(np.interp(xx, _x, _d)),
                float(T), (float(xs[0]), float(xs[-1])))
            if len(roots) >= 2:
                out["spmat_lo"][ti], out["spmat_hi"][ti] = roots[0], roots[-1]

        Tc_fit, w0, _T_last = _apex(declared, T_grid, out["bin_lo"],
                                     out["bin_hi"], T_step)
        res = dict(T=np.asarray(T_grid, dtype=np.float64), **out,
                   Tc_fit=Tc_fit, w0=w0, iso_x=iso_x, iso_s=iso_s,
                   isobar_P_GPa=np.float64(P),
                   isobar="model")
        target = out_dir / f"P{int(P)}GPa"
        target.mkdir(parents=True, exist_ok=True)
        np.savez(target / "phase_diagram.npz", **res)
        written.append(res)
    return written


def bulk_hessian(model, rho: np.ndarray, T: float, dtype) -> np.ndarray:
    """``H_tot(0) = sym(d2 f_loc / d rho2) + W_hat(0)`` per state, ``(P, n, n)`` float64, evaluated in ``dtype``.

    ``model``: a pair-kernel functional. ``rho``: uniform states, ``(P, n)``. ``T``: temperatures (K).
    ``dtype``: the torch dtype the Hessian is evaluated in."""
    r = torch.as_tensor(rho, dtype=dtype).requires_grad_(True)
    kBT = model.kB * torch.as_tensor(np.full(r.shape[0], float(T)), dtype=dtype)
    with torch.enable_grad():
        f = model.f_local.f_pointwise(r, kBT)
        g = torch.autograd.grad(f.sum(), r, create_graph=True)[0]
        H = torch.stack([torch.autograd.grad(g[:, i].sum(), r,
                                             retain_graph=True)[0]
                         for i in range(r.shape[1])], dim=1)
    H = 0.5 * (H + H.transpose(1, 2))
    w0 = _w_hat_zero(model, dtype, "the bulk Hessian (and the stability map)")
    return (H + w0).detach().double().numpy()


def _stability_map(model, system: System, pressures: Sequence[float],
                   T_grid: np.ndarray, declared: Mapping[str, Any],
                   out_dir: Path) -> Path:
    """``lambda_min(H_tot(0))`` and ``Gamma = x0 x1 / S_cc(0)`` on each model isobar, one file."""
    if system.n_species != 2:
        raise NotImplementedError(
            "the stability map walks one composition axis; it needs two "
            f"channels and this system has {system.n_species}")
    spec = dict(_need(declared, "stability_map"))
    lo, hi, count = spec["x_points"]
    x = np.linspace(float(lo), float(hi), int(count))
    dtype = {"float32": torch.float32, "float64": torch.float64}[
        spec["hessian_dtype"]]
    min_points = int(spec["min_points"])
    by_pressure = manifolds(system, declared)
    isobar = dict(_need(declared, "isobar"))
    unit = float(_need(declared, "pressure_unit"))
    degree = int(_need(declared, "poly_degree"))

    # every isobar first, in the model's own float64
    paths = {}
    for P in pressures:
        for T in T_grid:
            s = isobar_scale(model, float(P), x, float(T),
                             manifold=by_pressure[float(P)], pressure_unit=unit,
                             poly_degree=degree, n_species=system.n_species,
                             **isobar)
            paths[(float(P), float(T))] = manifold_density(
                by_pressure[float(P)], x, float(T), poly_degree=degree) * s

    # a copy: casting every buffer to `dtype` and back would lose the float64 ones for later stages
    model = copy.deepcopy(model).to(dtype)
    out = {"T": np.asarray(T_grid, dtype=np.float64), "x_int": x}
    for P in pressures:
        lam = np.full((len(T_grid), len(x)), np.nan)
        gam = np.full((len(T_grid), len(x)), np.nan)
        for i, T in enumerate(T_grid):
            n = paths[(float(P), float(T))]
            m = np.isfinite(n)
            if int(m.sum()) < min_points:
                continue
            x0 = 1.0 - x[m]                  # channel-0 fraction; x is channel 1's
            rho = np.stack([x0 * n[m], (1 - x0) * n[m]], -1)
            H = bulk_hessian(model, rho, float(T), dtype)
            lam[i, m] = np.linalg.eigvalsh(H)[:, 0]
            x1 = 1.0 - x0
            z = np.stack([x1, -(1.0 - x1)], -1)
            y = np.linalg.solve(H, z[..., None])[..., 0]
            scc0 = (model.kB * np.full(int(m.sum()), float(T))
                    * (z * y).sum(-1) / n[m])
            gam[i, m] = x0 * (1 - x0) / scc0
        out[f"lam_P{int(P)}"] = lam
        out[f"Gamma_P{int(P)}"] = gam
    path = out_dir / "stability_map.npz"
    np.savez(path, **out)
    return path


#: The ``"mu_roots"`` read-off's declared block: every key required.
ONE_FIELD_KEYS = _one_field.READOFFS["mu_roots"]


def _one_field_stage(model, system: System, T_grid: np.ndarray,
                     declared: Mapping[str, Any], out_dir: Path) -> Path:
    """The fixed-density one-field diagram by the declared ``readoff`` (:data:`.one_field.READOFFS`), in the
    declared dtype."""
    if system.n_species != 1:
        raise NotImplementedError(
            f"the one-field stage reads one fraction; this system has "
            f"{system.n_species} channels (the isobar stages read two)")
    spec = dict(_need(declared, "one_field"))
    readoff = _need(spec, "readoff")
    if readoff not in _one_field.READOFFS:
        raise ValueError(f"one_field readoff={readoff!r} is not one of "
                         f"{sorted(_one_field.READOFFS)}")
    for key in _one_field.READOFFS[readoff]:
        _need(spec, key)
    dtype = {"float32": torch.float32, "float64": torch.float64}[spec["dtype"]]
    model = copy.deepcopy(model).to(dtype)
    T_grid = np.asarray(T_grid, dtype=np.float64)
    phi = declared_grid(spec["phi_grid"])
    if readoff == "bulk_minima":
        out = _one_field.bulk_minima(model, T_grid, phi, edge_trim=spec["edge_trim"],
                                     centre=spec["centre"])
    else:
        out = _one_field.phase_diagram(model, T_grid, phi, spec)
    path = out_dir / "one_field_phase_diagram.npz"
    np.savez(path, **out)
    return path


def _summary_rows(pressures, results) -> list:
    rows = []
    for P, res in zip(pressures, results):
        finite = np.isfinite(res["bin_lo"])
        T = res["T"]
        T_c = float(T[finite].max()) if finite.any() else float("nan")
        if finite.any():
            mid = float(np.median(T[finite]))
            i = int(np.argmin(np.abs(T - mid)))
        else:
            i = 0
        rows.append({
            "P_GPa": float(P),
            "T_c_K": T_c,
            "T_c_fit_K": float(res["Tc_fit"]),
            "T_mid_K": float(T[i]),
            "bin_lo_mid": float(res["bin_lo"][i]),
            "bin_hi_mid": float(res["bin_hi"][i]),
            "max_abs_iso_s_minus_1": float(
                np.nanmax(np.abs(res["iso_s"] - 1.0)))
            if np.isfinite(res["iso_s"]).any() else float("nan"),
        })
    return rows


def _write_summary(path: Path, rows: list) -> None:
    with open(path, "w", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(system: System, ckpt: Any, *, stages: Sequence[str],
        out: Optional[Path] = None, pressures: Sequence[float] = (),
        T_grid: Optional[Sequence[float]] = None, override_declared: bool = False,
        **declared: Any) -> Path:
    """Diagnose one checkpoint into ``<out>/<md5[:12]>``; returns that directory.

    ``stages`` required, from :data:`STAGES`; every other keyword is declared and recorded in
    ``MANIFEST.json``. The one-field stage's grid is the declared one (:func:`request_T_grid`).

    ``system``: the declared system. ``ckpt``: its :class:`~aipf.system.Checkpoint` or a file path.
    ``stages``: the stages to run. ``out``: the parent directory; ``None`` is the farm's ``diagnose/``.
    ``pressures``: the isobars, in the manifolds' pressure unit (GPa for the shipped systems).
    ``T_grid``: the temperatures (K). ``override_declared``: as :func:`request_T_grid`.
    ``declared``: the system's ``defaults["diagnose"]`` keywords, forwarded to the stages."""
    check_diagnosable(system)
    bad = sorted(set(stages) - set(STAGES))
    if bad:
        raise ValueError(f"unknown stage(s) {bad}; one of {STAGES}")
    if not stages:
        raise ValueError(f"no stage asked for; one or more of {STAGES}")
    T_grid, T_grid_source = request_T_grid(declared, stages, pressures, T_grid,
                                           override_declared=override_declared)
    check_declared(declared, stages)

    path = resolve_checkpoint(system, ckpt)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if out is None:
        from aipf.data import index
        out = index.diagnose_dir(system)
    out_dir = Path(out) / digest[:12]
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(system, path)
    written: dict = {}

    if "phase_diagram" in stages:
        results = _phase_diagram(model, system, pressures,
                                 np.asarray(T_grid, dtype=np.float64),
                                 declared, out_dir)
        written["phase_diagram"] = [f"P{int(P)}GPa/phase_diagram.npz"
                                    for P in pressures]
        rows = _summary_rows(pressures, results)
        if rows:
            _write_summary(out_dir / "summary.csv", rows)
            written["summary"] = "summary.csv"

    if "stability_map" in stages:
        _stability_map(model, system, pressures,
                       np.asarray(T_grid, dtype=np.float64), declared,
                       out_dir)
        written["stability_map"] = "stability_map.npz"

    if "one_field_phase_diagram" in stages:
        _one_field_stage(model, system, T_grid, declared, out_dir)
        written["one_field_phase_diagram"] = "one_field_phase_diagram.npz"

    if "dome" in stages or "tc" in stages:
        scan = _dome_scan(model, system, declared, out_dir)
        written["dome"] = "dome_Pgrid.npz"
        if "tc" in stages:
            (out_dir / "tc.json").write_text(json.dumps({
                "method": _need(declared, "tc_method"),
                "P": scan["P_grid"].tolist(),
                "Tc_fit": scan["Tc_fit"].tolist(),
                "w0": scan["w0"].tolist(),
                "T_last": scan["T_last"].tolist(),
            }, indent=1))
            written["tc"] = "tc.json"

    if "kappa" in stages:
        value = _kappa.nonlocal_kernel_kappa_eff(model)
        (out_dir / "kappa.json").write_text(json.dumps(
            {"kappa_eff": np.asarray(value.detach()).tolist()}, indent=1))
        written["kappa"] = "kappa.json"

    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "system": system.name,
        "checkpoint": str(path),
        "md5": digest,
        "stages": list(stages),
        "pressures": [float(P) for P in pressures],
        "T_grid": None if T_grid is None else [float(T) for T in T_grid],
        "T_grid_source": T_grid_source,
        "outputs": written,
        "declared": {key: repr(value) for key, value in declared.items()},
        "written_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    return out_dir
