"""The measured anchor tables, loaded once, and the four batch entries (``anchor_M``, ``anchor_S``,
``anchor_bulk``, ``anchor_P``) :meth:`aipf.train.lit_module.LitModule.compute_losses` reads.

Targets are read once in :meth:`AnchorTables.load`; the model side is rebuilt by every
:meth:`AnchorTables.batch`."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from aipf.data.rows import measured_rows
from aipf.paths import tracked_tables
from aipf.spectral import model_ndim, refuse_two_dimensions
from aipf.train.pressure import pressure_from_model

_LOG = logging.getLogger(__name__)

#: The keys a declared ``columns`` block must carry.
TABLE_COLUMNS: Tuple[str, ...] = (
    "phase",         # bool per row: may this row anchor at all; one name, or one per role (PHASE_ROLES)
    "temperature",   # the row's temperature, in the floor's own unit
    "densities",     # one name per channel, IN CHANNEL ORDER
    "composition",   # the fraction the bulk projector is built from
    "mobility",      # the measured mobility matrix per row
    "shells",        # the shell wavenumbers per row
    "structure",     # the measured structure factor per row and shell
    "bulk_target",   # the zero-wavenumber concentration target
    "bulk_sigma",    # its uncertainty
    "bulk_ok",       # the extrapolation flag, when the file carries one
)

#: The roles a per-role ``phase`` mapping names: the mobility table, and the structure table (shells and bulk).
PHASE_ROLES: Tuple[str, ...] = ("mobility", "structure")

#: The keys a declared ``row_weight`` block carries, when it is not ``None``.
ROW_WEIGHT_KEYS: Tuple[str, ...] = ("value", "error", "floor")

#: The coordinate a declaration's tables are keyed by: ``"pressure"`` (one file per pressure, rows carry
#: their densities) or ``"temperature"`` (one file per role, rows at one declared state, one per temperature).
TABLE_KEYS: Tuple[str, ...] = ("pressure", "temperature")

#: How the bulk anchor evaluates ``W_hat(0)``, declared as ``tables["w0"]["route"]``: ``"evaluator"``
#: (the kernel's own ``w_hat`` at ``k = 0``), ``"lattice"`` (the lattice sum on the declared grid and box)
#: or ``"radial"`` (``4 pi int_0^r_max r^2 W dr`` on declared points).
W0_ROUTES: Tuple[str, ...] = ("evaluator", "lattice", "radial")

#: The keys a ``"temperature"``-keyed ``columns`` block must carry.
TEMPERATURE_COLUMNS: Tuple[str, ...] = (
    "temperature",          # the mobility table's temperature column
    "mobility",             # its measured scalar mobility column
    "records",              # the structure table's per-sample records
    "record_temperature",   # a record's temperature entry
    "structure_zero",       # a record's zero-wavenumber structure factor entry
    "temperatures",         # the structure table's temperature list, the row order
)

#: The keys a declared ``eos_columns`` block must carry (the five the diagnosis driver declares).
EOS_COLUMNS: Tuple[str, ...] = (
    "x_column", "T_column", "n_columns", "status_column", "status_ok",
)


# -- provenance and path resolution --
def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(root, declared: Mapping[Any, Any], what: str, tracked: Optional[Path] = None
              ) -> Tuple[Tuple[float, Path], ...]:
    """``{control value: relative path}`` to ``((value, absolute path), ...)``, in declaration order.
    A relative path is looked for under ``tracked`` (the system's tracked tables) first, then under
    ``root`` (a path, or a callable returning one, called only when needed: the raw root)."""
    out = []
    for value, rel in dict(declared).items():
        path = Path(rel)
        if not path.is_absolute():
            if tracked is not None and (tracked / path).is_file():
                path = tracked / path
            else:
                path = (root() if callable(root) else root) / path
        if not path.is_file():
            raise FileNotFoundError(
                f"{what}: the table declared at control value {value!r} is "
                f"not a file: {path}. The declaration names it relative to "
                f"this system's tracked tables"
                + (f" ({tracked})" if tracked is not None else "")
                + " and then its raw root, which is where it is looked for")
        out.append((float(value), path))
    if not out:
        raise ValueError(
            f"{what}: no table is declared. An anchor tier with no table is "
            f"not a weaker anchor, it is a different loss, so this is a "
            f"refusal and not a warning")
    return tuple(out)


def _columns(declared: Mapping[str, Any], required: Sequence[str],
             what: str) -> Dict[str, Any]:
    missing = [name for name in required if name not in declared]
    if missing:
        raise KeyError(
            f"{what} does not declare {missing}: the measured tables are a "
            f"file format and this package holds none of their column "
            f"names, so every one of {list(required)} has to be declared")
    return {name: declared[name] for name in required}


# -- the row filters the measured tables were cut with --
def _phase_column(col, role: str) -> str:
    """The row-gate column for ``role``: the one declared name, or that role's entry of a mapping."""
    phase = col["phase"]
    if isinstance(phase, Mapping):
        if role not in phase:
            raise KeyError(
                f"the declared phase columns {dict(phase)} name no column "
                f"for the {role!r} table; declare one per role "
                f"{list(PHASE_ROLES)}, or one name for every table")
        return phase[role]
    return phase


def _keep(archive, col, t_min: Optional[float], role: str) -> np.ndarray:
    name = _phase_column(col, role)
    if name not in archive:
        raise KeyError(
            f"the {role} table carries no declared phase column {name!r}")
    keep = archive[name].astype(bool)
    if t_min is not None:
        keep = keep & (archive[col["temperature"]] > float(t_min))
    return keep


def _densities(archive, col, keep) -> np.ndarray:
    return np.stack([archive[name][keep] for name in col["densities"]],
                    axis=-1)


def _f32(array) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def _mobility_rows(path: Path, col, t_min: Optional[float]):
    """``(rho, T, M_target)``: the measured transport matrix, per state."""
    with np.load(path) as z:
        keep = _keep(z, col, t_min, "mobility")
        return (_f32(_densities(z, col, keep)),
                _f32(z[col["temperature"]][keep]),
                _f32(z[col["mobility"]][keep]))


def _relative_error_weights(archive, keep, spec) -> np.ndarray:
    """``(P,)`` float32 row weights ``1 / max(|error/value|, floor)^2``, normalised to mean one.

    A non-finite ratio weighs zero; a table with no finite weight falls back to ones."""
    n = int(keep.sum())
    rel = np.abs(np.asarray(archive[spec["error"]], float)[keep]
                 / np.asarray(archive[spec["value"]], float)[keep])
    rel = np.where(np.isfinite(rel), rel, np.inf)
    rel = np.maximum(rel, float(spec["floor"]))
    w = 1.0 / rel ** 2
    if not np.isfinite(w).any() or w.sum() == 0:
        return np.ones(n, dtype=np.float32)
    w = np.where(np.isfinite(w), w, 0.0)
    w = w / w.mean()
    return w.astype(np.float32)


def _row_weights(archive, keep, spec) -> Optional[torch.Tensor]:
    """The declared row weights, or ``None`` when ``spec`` is ``None`` (the unweighted mean, not ones)."""
    if spec is None:
        return None
    return torch.from_numpy(_relative_error_weights(archive, keep, spec))


def _cat_weights(parts: Sequence[Optional[torch.Tensor]]):
    """Concatenated row weights, or ``None`` when no table is weighted."""
    if all(p is None for p in parts):
        return None
    return torch.cat(list(parts))


def _shell_rows(path: Path, col, k_fit: float, kB: float,
                t_min: Optional[float], weight_spec=None):
    """``(rho, T, k, target, mask)``: shells ``0 < k <= k_fit`` with a finite positive-determinant matrix,
    ``target = k_B T . S(k)^-1``; short rows repeat their last shell behind a False mask."""
    with np.load(path) as z:
        keep = _keep(z, col, t_min, "structure")
        rho = _densities(z, col, keep)
        T = z[col["temperature"]][keep].astype(np.float64)
        k_all = z[col["shells"]][keep].astype(np.float64)
        S_all = z[col["structure"]][keep].astype(np.float64)
        weight = _row_weights(z, keep, weight_spec)

    n_rows = int(keep.sum())
    n_channels = S_all.shape[-1]
    kBT = float(kB) * T
    rows_k, rows_target = [], []
    for row in range(n_rows):
        k_row = k_all[row]
        in_band = np.isfinite(k_row) & (k_row > 0.0) & (k_row <= k_fit)
        ks, targets = [], []
        for shell in np.nonzero(in_band)[0]:
            measured = S_all[row, shell]
            if not np.all(np.isfinite(measured)):
                continue
            if _determinant(measured) <= 0.0:
                _LOG.warning("shell anchor: dropping a non-positive-definite "
                              "shell (row %d, k=%.4f) of %s",
                              row, k_row[shell], path)
                continue
            ks.append(k_row[shell])
            targets.append(kBT[row] * np.linalg.inv(measured))
        rows_k.append(np.asarray(ks, dtype=np.float64))
        rows_target.append(np.asarray(targets, dtype=np.float64).reshape(
            -1, n_channels, n_channels))
        if not ks:
            _LOG.warning("shell anchor: row %d of %s has no valid in-band "
                          "shell", row, path)

    width = max(max((len(r) for r in rows_k), default=0), 1)
    k_pad = np.zeros((n_rows, width), dtype=np.float32)
    target_pad = np.zeros((n_rows, width, n_channels, n_channels),
                          dtype=np.float32)
    mask = np.zeros((n_rows, width), dtype=bool)
    for row in range(n_rows):
        n_valid = len(rows_k[row])
        if n_valid == 0:
            continue
        mask[row, :n_valid] = True
        k_pad[row, :n_valid] = rows_k[row]
        target_pad[row, :n_valid] = rows_target[row]
        if n_valid < width:
            k_pad[row, n_valid:] = rows_k[row][-1]
            target_pad[row, n_valid:] = rows_target[row][-1]
    return (_f32(rho), _f32(T), torch.from_numpy(k_pad),
            torch.from_numpy(target_pad), torch.from_numpy(mask), weight)


def _determinant(matrix: np.ndarray) -> float:
    """The 2x2 cross-multiplication the measured table was filtered with; ``det`` beyond two channels."""
    if matrix.shape == (2, 2):
        return float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    return float(np.linalg.det(matrix))


def _bulk_rows(path: Path, col, t_min: Optional[float], x_channel: int,
               weight_spec=None):
    """``(rho, T, zvec, rho_tot, target, weight)``, the zero-wavenumber anchor, ``zvec = (x1, -(1 - x1))``
    with ``x1`` the channel-1 fraction (``1 - x`` when the composition column labels channel 0).
    ``bulk_sigma``/``bulk_ok`` ``None``: the phase gate alone decides, the target need only be finite."""
    with np.load(path) as z:
        keep = _keep(z, col, t_min, "structure")
        target_all = z[col["bulk_target"]]
        keep = keep & np.isfinite(target_all)
        if col["bulk_sigma"] is None:
            if col["bulk_ok"] is not None:
                raise ValueError(
                    "bulk_ok is declared without bulk_sigma; the health "
                    "flag and its fallback heuristic come as a pair")
            return _bulk_finish(z, col, keep, target_all, x_channel,
                                weight_spec)
        sigma_all = z[col["bulk_sigma"]]
        keep = keep & np.isfinite(sigma_all)
        if col["bulk_ok"] in z:
            keep = keep & z[col["bulk_ok"]].astype(bool)
        else:
            _LOG.warning("bulk anchor: %s carries no %r column, falling back "
                          "to the historical sigma < 0.5 * target heuristic",
                          path, col["bulk_ok"])
            keep = keep & (sigma_all < 0.5 * target_all)
        return _bulk_finish(z, col, keep, target_all, x_channel, weight_spec)


def _bulk_finish(z, col, keep, target_all, x_channel: int, weight_spec):
    rho = _densities(z, col, keep)
    x = z[col["composition"]][keep].astype(np.float64)
    x1 = _channel_one_fraction(x, x_channel)
    return (_f32(rho), _f32(z[col["temperature"]][keep]),
            _f32(np.stack([x1, -(1.0 - x1)], axis=-1)),
            _f32(rho.sum(-1)), _f32(target_all[keep]),
            _row_weights(z, keep, weight_spec))


def _channel_one_fraction(x, x_channel: int):
    """A two-channel composition label as the fraction of channel 1: ``x`` itself, or ``1 - x``."""
    if int(x_channel) == 1:
        return x
    if int(x_channel) == 0:
        return 1.0 - x
    raise ValueError(
        f"x_channel={x_channel!r}: a two-channel composition labels channel "
        f"0 or channel 1")


def _eos_rows(path: Path, col, pressure: float, pressure_unit: float,
              x_channel: int):
    """``(rho, T, P_target)``: one absolute-pressure anchor per ok row, at the declared pressure
    times ``pressure_unit``; ``x`` is the fraction of channel ``x_channel``."""
    rho, T = [], []
    for x, temperature, n in measured_rows(path, **{name: col[name]
                                                    for name in EOS_COLUMNS}):
        if int(x_channel) == 1:
            rho.append([(1.0 - x) * n, x * n])
        elif int(x_channel) == 0:
            rho.append([x * n, (1.0 - x) * n])
        else:
            raise ValueError(
                f"x_channel={x_channel!r}: a two-channel composition labels "
                f"channel 0 or channel 1")
        T.append(temperature)
    if not rho:
        raise ValueError(f"no usable equation-of-state row in {path}")
    target = np.full(len(rho), float(pressure) * float(pressure_unit))
    return (_f32(np.asarray(rho)), _f32(np.asarray(T)), _f32(target))


def _refuse_beyond_two_channels(n_channels: int, what: str) -> None:
    if n_channels != 2:
        raise NotImplementedError(
            f"{what} walks ONE composition axis, which is a path through an "
            f"{n_channels}-component simplex only when there are two "
            f"components")


def _w0_route(declared: Mapping[str, Any]) -> Dict[str, Any]:
    """A ``w0`` declaration checked: a route of :data:`W0_ROUTES`, with ``r_max`` and ``n_points``
    for ``"radial"`` and only then."""
    declared = dict(declared)
    route = declared.pop("route", None)
    if route not in W0_ROUTES:
        raise ValueError(
            f"w0 route={route!r} is not one of {W0_ROUTES}: how the bulk anchor "
            f"evaluates W_hat(0) is declared, never assumed")
    wanted = {"r_max", "n_points"} if route == "radial" else set()
    if set(declared) != wanted:
        raise ValueError(
            f"w0 route {route!r} takes exactly {sorted(wanted)}, got "
            f"{sorted(declared)}")
    if route == "radial":
        declared = {"r_max": float(declared["r_max"]),
                    "n_points": int(declared["n_points"])}
    return {"route": route, **declared}


def _state_rows(state: Sequence[float], n_rows: int) -> torch.Tensor:
    """The declared state, one row per temperature, ``(P, n)``."""
    return _f32(np.tile(np.asarray(state, dtype=np.float64), (n_rows, 1)))


def _mobility_by_temperature(path: Path, col, state):
    """``(rho, T, M_target)`` with ``M_target`` ``(P, 1, 1)``: one row per temperature, all at ``state``."""
    with np.load(path) as z:
        T = np.asarray(z[col["temperature"]], dtype=np.float64)
        M = np.asarray(z[col["mobility"]], dtype=np.float64)
    return _state_rows(state, T.size), _f32(T), _f32(M.reshape(-1, 1, 1))


def _structure_by_temperature(path: Path, col, state, rho_total: float):
    """``(rho, T, zvec, rho_tot, target, sigma)``: per temperature, the mean of its records' ``S(0)``
    and the standard error of its inverse ``sem / mean^2``."""
    with np.load(path, allow_pickle=True) as z:
        records = list(z[col["records"]])
        temperatures = np.asarray(z[col["temperatures"]], dtype=np.float64)
    target, sigma = [], []
    for T in temperatures:
        samples = np.array([r[col["structure_zero"]] for r in records
                            if abs(float(r[col["record_temperature"]]) - T)
                            < _SAME_TEMPERATURE], dtype=np.float64)
        if samples.size < 2:
            raise ValueError(
                f"{path}: temperature {T} has {samples.size} record(s); the "
                f"standard error of its mean needs at least two")
        mean = samples.mean()
        target.append(mean)
        sigma.append(samples.std(ddof=1) / np.sqrt(samples.size) / mean ** 2)
    n = temperatures.size
    return (_state_rows(state, n), _f32(temperatures),
            _f32(np.ones((n, len(state)))), _f32(np.full(n, rho_total)),
            _f32(target), _f32(sigma))


#: How close two temperatures are to be the same row of a record-per-sample table.
_SAME_TEMPERATURE = 1e-9


def _no_rows(*shape_tail: int) -> torch.Tensor:
    """An empty float32 table with the given trailing shape."""
    return torch.zeros((0,) + tuple(shape_tail), dtype=torch.float32)


# -- the loaded tables --
class _NoAnchors:
    """The marker type for a deliberate drift-only run: one instance, compared by identity."""

    __slots__ = ()

    def __repr__(self) -> str:                      # pragma: no cover
        return "<no-anchors>"


#: Sole instance of :class:`_NoAnchors`: a drift-only run on a system that declares anchors.
#: Not ``None``, which means "the declaration answers".
NO_ANCHORS = _NoAnchors()


@dataclass(frozen=True)
class AnchorTables:
    """The three measured tables, loaded (targets and their state points only), plus provenance."""

    #: ``anchor_M``: the state points, and the measured mobility matrices.
    mobility_rho: torch.Tensor
    mobility_T: torch.Tensor
    mobility_target: torch.Tensor

    #: ``anchor_S``: state points, shell wavenumbers, ``k_B T . S(k)^-1``, shell mask, target inverse root.
    shell_rho: torch.Tensor
    shell_T: torch.Tensor
    shell_k: torch.Tensor
    shell_target: torch.Tensor
    shell_mask: torch.Tensor
    shell_target_rootinv: torch.Tensor
    shell_weight: Optional[torch.Tensor]

    #: ``anchor_bulk``: the zero-wavenumber limit.
    bulk_rho: torch.Tensor
    bulk_T: torch.Tensor
    bulk_zvec: torch.Tensor
    bulk_rho_tot: torch.Tensor
    bulk_target: torch.Tensor
    bulk_weight: Optional[torch.Tensor]
    #: The target's own uncertainty, for the ``sigma_chi2`` residual; ``None`` when the tables carry none.
    bulk_sigma: Optional[torch.Tensor]

    #: ``anchor_P``: equation-of-state states and pressures, in the model's energy-per-volume unit.
    pressure_rho: torch.Tensor
    pressure_T: torch.Tensor
    pressure_target: torch.Tensor

    #: The relative-residual denominator's floor ``max(|P_target|, floor)``, model unit; ``None`` = ``|P_target|``.
    pressure_floor: Optional[float]

    #: ``{role: ((path, sha256), ...)}``, one entry per file actually read.
    provenance: Mapping[str, Any]

    #: The system's declared ``(grid, box)``, which a lattice-sum kernel needs for ``W_hat(0)``; ``None``
    #: when either is undeclared, and such a kernel then refuses by name.
    geometry: Optional[Tuple[Tuple[int, int, int], Tuple[float, float, float]]]

    #: The declared ``W_hat(0)`` route of the bulk anchor (:func:`_w0_route`).
    w0: Mapping[str, Any]

    # -- loading ---------------------------------------------------------
    @classmethod
    def load(cls, system, *, key: str, **declared) -> "AnchorTables":
        """Read the declared tables, keyed by ``key`` (one of :data:`TABLE_KEYS`); the rest is that
        form's arguments (:meth:`load_by_pressure`, :meth:`load_by_temperature`)."""
        if key == "pressure":
            return cls.load_by_pressure(system, **declared)
        if key == "temperature":
            return cls.load_by_temperature(system, **declared)
        raise ValueError(
            f"tables key={key!r} is not one of {TABLE_KEYS}: the coordinate "
            f"the tables are keyed by is declared, never assumed")

    @classmethod
    def load_by_temperature(cls, system, *, m_table: str, s_table: str,
                            state: Sequence[float], columns: Mapping[str, Any],
                            rho_total: float,
                            w0: Mapping[str, Any]) -> "AnchorTables":
        """One mobility and one structure table, one row per temperature, all at the declared ``state``.

        Feeds ``anchor_M`` and ``anchor_bulk`` only; one channel. Paths as in :meth:`load_by_pressure`."""
        w0 = _w0_route(w0)
        state = tuple(float(v) for v in state)
        if len(state) != 1 or int(system.n_species) != 1:
            raise NotImplementedError(
                f"temperature-keyed tables hold one scalar mobility and one "
                f"S(0) per row, which is one channel; state={state} on "
                f"{system.n_species} channel(s) is not that")
        root, tracked = system.paths.raw, tracked_tables(system.name)
        (_, m_path), = _resolve(root, {0.0: m_table}, "m_table", tracked)
        (_, s_path), = _resolve(root, {0.0: s_table}, "s_table", tracked)
        col = _columns(columns, TEMPERATURE_COLUMNS,
                       "the declared temperature-keyed table columns")
        m_rho, m_T, m_target = _mobility_by_temperature(m_path, col, state)
        b_rho, b_T, b_zvec, b_tot, b_target, b_sigma = \
            _structure_by_temperature(s_path, col, state, float(rho_total))
        n = len(state)
        return cls(
            mobility_rho=m_rho, mobility_T=m_T, mobility_target=m_target,
            shell_rho=_no_rows(n), shell_T=_no_rows(),
            shell_k=_no_rows(1), shell_target=_no_rows(1, n, n),
            shell_mask=torch.zeros((0, 1), dtype=torch.bool),
            shell_target_rootinv=_no_rows(1, n, n), shell_weight=None,
            bulk_rho=b_rho, bulk_T=b_T, bulk_zvec=b_zvec, bulk_rho_tot=b_tot,
            bulk_target=b_target, bulk_weight=None, bulk_sigma=b_sigma,
            pressure_rho=_no_rows(n), pressure_T=_no_rows(),
            pressure_target=_no_rows(), pressure_floor=None,
            provenance={"m_table": ((str(m_path), _digest(m_path)),),
                        "s_table": ((str(s_path), _digest(s_path)),),
                        "eos_csvs": ()},
            geometry=_geometry(system), w0=w0)

    @classmethod
    def load_by_pressure(cls, system, *, m_table: Mapping[Any, Any],
                         s_table: Mapping[Any, Any],
                         eos_csvs: Mapping[Any, Any], pressure_unit: float,
                         columns: Mapping[str, Any],
                         eos_columns: Mapping[str, Any],
                         row_weight: Optional[Mapping[str, Any]],
                         pressure_floor: Optional[float],
                         w0: Mapping[str, Any]) -> "AnchorTables":
        """Read the three declared tables (tracked copy first, then ``system.paths.raw()``); every argument required,
        no partial load.

        Declared on ``system``: ``anchor_rules.T_min_by_pressure``, ``defaults["k_fit_stat"]``,
        ``constants["kB"]``.
        ``pressure_unit`` is one unit of the keys' pressure in the model's energy-per-volume unit.
        ``row_weight``: ``None`` (equal rows) or ``{"value", "error", "floor"}`` columns of the structure table.
        ``pressure_floor``: ``None`` or a floor on ``L_P``'s denominator, in the keys' pressure unit.
        Compositions are read as the fraction of channel ``system.table_keys["x_channel"]``."""
        w0 = _w0_route(w0)
        root, tracked = system.paths.raw, tracked_tables(system.name)
        m_paths = _resolve(root, m_table, "m_table", tracked)
        s_paths = _resolve(root, s_table, "s_table", tracked)
        eos_paths = _resolve(root, eos_csvs, "eos_csvs", tracked)
        col = _columns(columns, TABLE_COLUMNS, "the declared table columns")
        eos_col = _columns(eos_columns, EOS_COLUMNS,
                           "the declared manifold columns")
        # The zero-wavenumber projector and the manifold's own
        # composition-to-density map both walk ONE composition axis.
        _refuse_beyond_two_channels(len(col["densities"]),
                                    "the composition-axis anchors")
        phase = col["phase"]
        if isinstance(phase, Mapping) and set(phase) != set(PHASE_ROLES):
            raise KeyError(
                f"the declared phase columns name roles {sorted(phase)}; a "
                f"per-role mapping names exactly {list(PHASE_ROLES)}")
        if row_weight is not None:
            row_weight = _columns(row_weight, ROW_WEIGHT_KEYS,
                                  "the declared row weighting")
        x_channel = int(system.table_keys["x_channel"])

        floors = dict(getattr(system.anchor_rules, "T_min_by_pressure", {})
                      or {})
        k_fit = float(_declared(system.defaults, "k_fit_stat"))
        kB = float(_declared(system.constants, "kB"))

        def floor_at(value: float) -> Optional[float]:
            for key, floor in floors.items():
                if float(key) == float(value):
                    return float(floor)
            return None

        m_parts = [_mobility_rows(p, col, floor_at(v)) for v, p in m_paths]
        s_parts = [_shell_rows(p, col, k_fit, kB, floor_at(v), row_weight)
                   for v, p in s_paths]
        b_parts = [_bulk_rows(p, col, floor_at(v), x_channel, row_weight)
                   for v, p in s_paths]
        e_parts = [_eos_rows(p, eos_col, v, pressure_unit, x_channel)
                   for v, p in eos_paths]

        m_rho, m_T, m_target = _cat(m_parts)
        b_rho, b_T, b_zvec, b_tot, b_target = _cat([p[:5] for p in b_parts])
        b_weight = _cat_weights([p[5] for p in b_parts])
        e_rho, e_T, e_target = _cat(e_parts)
        s_rho, s_T, s_k, s_target, s_mask = _cat_shells(
            [p[:5] for p in s_parts])
        s_weight = _cat_weights([p[5] for p in s_parts])

        from aipf.losses.static import sinv_rootinv

        return cls(
            mobility_rho=m_rho, mobility_T=m_T, mobility_target=m_target,
            shell_rho=s_rho, shell_T=s_T, shell_k=s_k,
            shell_target=s_target, shell_mask=s_mask,
            shell_target_rootinv=sinv_rootinv(s_target),
            shell_weight=s_weight,
            bulk_rho=b_rho, bulk_T=b_T, bulk_zvec=b_zvec,
            bulk_rho_tot=b_tot, bulk_target=b_target, bulk_weight=b_weight,
            bulk_sigma=None, pressure_rho=e_rho, pressure_T=e_T, pressure_target=e_target,
            pressure_floor=(None if pressure_floor is None
                            else float(pressure_floor) * float(pressure_unit)),
            provenance={
                "m_table": tuple((str(p), _digest(p)) for _v, p in m_paths),
                "s_table": tuple((str(p), _digest(p)) for _v, p in s_paths),
                "eos_csvs": tuple((str(p), _digest(p)) for _v, p in eos_paths),
            },
            geometry=_geometry(system), w0=w0,
        )

    # -- the batch -------------------------------------------------------
    def batch(self, model, kB: float) -> Dict[str, Dict[str, Any]]:
        """The entries ``compute_losses`` reads, model side rebuilt with a graph (a per-step object);
        an anchor with no rows is left out."""
        refuse_two_dimensions(model_ndim(model), "the anchor losses")
        w_hat = _w_hat_of(model)
        out: Dict[str, Dict[str, Any]] = {}
        if self.mobility_rho.shape[0]:
            out["anchor_M"] = {
                "M_pred": _mobility_at(model, self.mobility_rho,
                                       self.mobility_T),
                "M_target": self.mobility_target,
            }
        if self.shell_rho.shape[0]:
            H = (_curvature(model, self.shell_rho, self.shell_T).unsqueeze(1)
                 + w_hat(self.shell_k))
            out["anchor_S"] = {
                "H": H, "target": self.shell_target, "mask": self.shell_mask,
                "target_rootinv": self.shell_target_rootinv,
                "row_weight": self.shell_weight,
            }
        if self.bulk_rho.shape[0]:
            H0 = (_curvature(model, self.bulk_rho, self.bulk_T)
                  + _w_hat_zero(model, w_hat, self.bulk_rho, self.geometry,
                                self.w0))
            out["anchor_bulk"] = {
                "H0": H0, "kBT": float(kB) * self.bulk_T,
                "zvec": self.bulk_zvec, "rho_tot": self.bulk_rho_tot,
                "target": self.bulk_target, "row_weight": self.bulk_weight,
                "sigma": self.bulk_sigma,
            }
        if self.pressure_rho.shape[0]:
            # the box is irrelevant: each row is a single-point grid, carrying the zero wavevector only
            boxes = torch.ones(self.pressure_rho.shape[0], 3,
                               dtype=self.pressure_rho.dtype,
                               device=self.pressure_rho.device)
            out["anchor_P"] = {
                "P_model": pressure_from_model(model, self.pressure_rho,
                                               boxes, self.pressure_T),
                "P_target": self.pressure_target,
                "p_floor": self.pressure_floor,
            }
        return out

    def terms(self) -> Tuple[str, ...]:
        """The canonical loss terms these tables feed: one per anchor with at least one row."""
        rows = self.rows()
        return tuple(name for name, group in zip(
            ("L_M", "L_S", "L_bulk", "L_P"),
            ("anchor_M", "anchor_S", "anchor_bulk", "anchor_P")) if rows[group])

    def rows(self) -> Dict[str, int]:
        """How many rows each anchor kept, for a run's manifest."""
        return {"anchor_M": int(self.mobility_rho.shape[0]),
                "anchor_S": int(self.shell_rho.shape[0]),
                "anchor_bulk": int(self.bulk_rho.shape[0]),
                "anchor_P": int(self.pressure_rho.shape[0])}


# -- helpers --
def _declared(block: Mapping[str, Any], name: str) -> Any:
    if name not in block:
        raise KeyError(
            f"the anchor tier needs {name!r} and this system does not "
            f"declare it. It is a property of the measurement, so there is "
            f"no core value to fall back on")
    return block[name]


def _cat(parts: Sequence[Sequence[torch.Tensor]]):
    """Row-wise concatenation of several tables' fixed-shape tensors."""
    if len(parts) == 1:
        return tuple(parts[0])
    return tuple(torch.cat([p[i] for p in parts])
                 for i in range(len(parts[0])))


def _cat_shells(parts):
    """Concatenate shell tables, each padded to the widest shell axis by its last shell; the mask is
    zero-padded."""
    if len(parts) == 1:
        return tuple(parts[0])
    width = max(p[2].shape[1] for p in parts)

    def pad(t, target):
        if t.shape[1] == target:
            return t
        tail = t[:, -1:].expand(t.shape[0], target - t.shape[1], *t.shape[2:])
        return torch.cat([t, tail.clone()], dim=1)

    def pad_mask(m):
        return torch.nn.functional.pad(m, (0, width - m.shape[1]))

    return (torch.cat([p[0] for p in parts]),
            torch.cat([p[1] for p in parts]),
            torch.cat([pad(p[2], width) for p in parts]),
            torch.cat([pad(p[3], width) for p in parts]),
            torch.cat([pad_mask(p[4]) for p in parts]))


#: The attribute a square-gradient rung keeps its coefficient under (``kappa k^2``, not a ``w_hat``).
_SQUARE_GRADIENT_ATTR = "kappa"


def _w_hat_of(model):
    """The field-wide term's curvature at ``k``: the pair kernel's ``w_hat``, a square-gradient rung's
    ``curvature_at_wavevector`` (``kappa k^2``), zero for a purely local rung, and a refusal for a rung that
    carries a ``kappa`` and no such method."""
    fn = getattr(getattr(model, "kernel", None), "w_hat", None)
    if fn is not None:
        return fn
    fn = getattr(model, "curvature_at_wavevector", None)
    if fn is not None:
        return fn
    if getattr(model, _SQUARE_GRADIENT_ATTR, None) is not None:
        raise NotImplementedError(
            f"this functional has no pair kernel but does carry a "
            f"{_SQUARE_GRADIENT_ATTR!r}, so its curvature at a wavevector "
            f"is not zero and this module cannot read it: there is no "
            f"protocol method for it. Add 'curvature_at_wavevector' to the "
            f"model protocol and implement it on this rung -- returning "
            f"zero here instead would drop a real term from the static "
            f"anchors' target comparison and say so nowhere")
    if not hasattr(model, "n_species"):
        raise ValueError(
            f"{type(model).__name__} has no kernel.w_hat, no curvature_at_wavevector and no "
            f"n_species, so the anchors can neither read its field-wide curvature nor size a zero "
            f"one; give the model one of them, or train it with --anchors none")
    n = int(getattr(model, "n_species"))

    def zero(k: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*k.shape, n, n, dtype=k.dtype, device=k.device)

    return zero


def _geometry(system):
    """``(grid, box)`` from the functional's declared ``grid`` and ``defaults["box"]``; ``None`` if either
    is undeclared."""
    grid = (system.functional.kwargs.get("grid")
            if system.functional is not None else None)
    box = system.defaults.get("box")
    if grid is None or box is None:
        return None
    return (tuple(int(g) for g in grid), tuple(float(b) for b in box))


def _w_hat_zero(model, w_hat, rho: torch.Tensor, geometry,
                w0: Mapping[str, Any]) -> torch.Tensor:
    """``W_hat(0)`` per row, ``(P, n, n)``, by the declared route; ``"lattice"`` without a declared grid
    and box is the evaluator's own refusal."""
    kernel = getattr(model, "kernel", None)
    k_zero = torch.zeros(rho.shape[0], 1, dtype=rho.dtype, device=rho.device)
    route = w0["route"]
    if route != "evaluator" and kernel is None:
        raise ValueError(
            f"w0 route {route!r} needs a pair kernel and this functional has "
            f"none; declare 'evaluator'")
    if route == "evaluator":
        return w_hat(k_zero)[:, 0]
    if route == "radial":
        if not hasattr(kernel, "w_hat_zero_quadrature"):
            raise ValueError(
                f"w0 route 'radial' integrates the radial kernel through "
                f"kernel.w_hat_zero_quadrature, and {type(kernel).__name__} has none; declare "
                f"'evaluator', or train with --anchors none")
        zero = kernel.w_hat_zero_quadrature(w0["r_max"], w0["n_points"])
        return zero.to(rho.dtype).expand(rho.shape[0], *zero.shape)
    if not hasattr(kernel, "evaluator"):
        raise ValueError(
            f"w0 route 'lattice' reads the lattice sum through kernel.evaluator, and "
            f"{type(kernel).__name__} has none; declare 'evaluator', or train with --anchors none")
    if not getattr(kernel.evaluator, "reads_geometry", False):
        raise ValueError(
            f"w0 route 'lattice' needs a kernel evaluated as a lattice sum, and "
            f"this one is {type(kernel.evaluator).__name__}; declare 'evaluator'")
    if geometry is None:
        return w_hat(k_zero)[:, 0]
    grid, box = geometry
    whole = w_hat(k_zero, grid=grid,
                  boxes=torch.tensor([box], dtype=torch.float64, device=rho.device))
    zero = whole[:, 0, 0, 0].real.to(rho.dtype)
    return zero.expand(rho.shape[0], *zero.shape[1:])


def _mobility_at(model, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """``M(rho, T)`` at ``(P,)`` state points, ``(P, n, n)``, each row passed as a single-point grid."""
    n_points, n_species = rho.shape
    M = model.mobility(rho.reshape(n_points, n_species, 1, 1, 1), T)
    return M.reshape(n_points, n_species, n_species)


def _curvature(model, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """``d2 f_loc / d rho2`` at ``(P,)`` state points, ``(P, n, n)``: local density only,
    ``create_graph=True``."""
    n_points, n_species = rho.shape
    leaf = rho.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        f = model.bulk_free_energy_density(
            leaf.reshape(n_points, n_species, 1, 1, 1), T)
        grad = torch.autograd.grad(f.sum(), leaf, create_graph=True)[0]
        rows = [torch.autograd.grad(grad[:, i].sum(), leaf,
                                    create_graph=True, retain_graph=True)[0]
                for i in range(n_species)]
    return torch.stack(rows, dim=1)


__all__ = ["AnchorTables", "TABLE_COLUMNS", "EOS_COLUMNS", "TABLE_KEYS",
           "TEMPERATURE_COLUMNS", "W0_ROUTES"]
