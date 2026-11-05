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

from aipf.data.rows import FRACTION_SUM_TOL, measured_rows
from aipf.paths import tracked_tables
from aipf.spectral import model_ndim, refuse_two_dimensions
from aipf.train.pressure import pressure_from_model

_LOG = logging.getLogger(__name__)

#: The keys a declared ``columns`` block must carry.
TABLE_COLUMNS: Tuple[str, ...] = (
    "phase",         # bool per row: may this row anchor at all; one name, or one per role (PHASE_ROLES)
    "temperature",   # the row's temperature, in the floor's own unit
    "densities",     # one name per channel, IN CHANNEL ORDER
    "composition",   # the fraction the bulk projector is built from; one name per channel beyond two
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

#: The vector form of :data:`EOS_COLUMNS`, beyond two channels: ``x_columns`` names one fraction per
#: channel, in channel order.
EOS_VECTOR_COLUMNS: Tuple[str, ...] = (
    "x_columns", "T_column", "n_columns", "status_column", "status_ok",
)

#: The ``columns`` key the vector form adds: the ``(rows, n)`` bool mask of the species present in a row.
ACTIVE_COLUMN = "active"


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
    """``(rho, T, M_target)``: the measured transport matrix, per state; the vector form appends
    ``active`` and keeps rows with two species or more, the matrix zero outside their active block."""
    with np.load(path) as z:
        keep = _keep(z, col, t_min, "mobility")
        if ACTIVE_COLUMN not in col:
            return (_f32(_densities(z, col, keep)),
                    _f32(z[col["temperature"]][keep]),
                    _f32(z[col["mobility"]][keep]))
        active = _species_present(z, col, path)
        keep = keep & (active.sum(-1) >= 2)
        measured = np.asarray(z[col["mobility"]], dtype=np.float64)[keep]
        rho = _densities(z, col, keep)
        T = z[col["temperature"]][keep]
    active = active[keep]
    pairs = active[:, :, None] & active[:, None, :]
    bad = np.nonzero((pairs & ~np.isfinite(measured)).any(axis=(-2, -1)))[0]
    if bad.size:
        raise ValueError(
            f"{path}: row {int(np.nonzero(keep)[0][bad[0]])} has a non-finite "
            f"entry in its active block of {col['mobility']!r}; L_M has no "
            f"filter for one, so it is refused rather than passed on")
    return (_f32(rho), _f32(T), _f32(np.where(pairs, measured, 0.0)),
            torch.from_numpy(active))


def _species_present(archive, col, path: Path) -> np.ndarray:
    """``(rows, n)`` bool, the declared ``active`` column of a vector-form table, checked against the
    fractions and the densities on every row: ``active == (x > 0) == (rho > 0)``, fractions summing to one
    within :data:`FRACTION_SUM_TOL`."""
    names = tuple(col["composition"])
    if col[ACTIVE_COLUMN] not in archive:
        raise KeyError(
            f"{path} carries no declared active column "
            f"{col[ACTIVE_COLUMN]!r}")
    active = np.asarray(archive[col[ACTIVE_COLUMN]])
    if active.dtype != np.bool_:
        raise ValueError(
            f"{path}: the active column {col[ACTIVE_COLUMN]!r} is "
            f"{active.dtype}, not bool; a mask is written as one, so that a "
            f"fraction or a density stored there is not read as a presence")
    x = np.stack([np.asarray(archive[name], np.float64) for name in names],
                 axis=-1)
    rho = np.stack([np.asarray(archive[name], np.float64)
                    for name in col["densities"]], axis=-1)
    if active.shape != x.shape:
        raise ValueError(
            f"{path}: the active column {col[ACTIVE_COLUMN]!r} has shape "
            f"{tuple(active.shape)} against {x.shape[0]} rows of "
            f"{x.shape[1]} channels")
    for what, present in (("x > 0", x > 0.0), ("rho > 0", rho > 0.0)):
        rows = np.nonzero((active != present).any(axis=-1))[0]
        if rows.size:
            row = int(rows[0])
            raise ValueError(
                f"{path}: row {row} has active={active[row].tolist()} and "
                f"{what} is {present[row].tolist()}; a species is present "
                f"exactly where its fraction and its density are positive")
    rows = np.nonzero(~(np.abs(x.sum(-1) - 1.0) <= FRACTION_SUM_TOL))[0]
    if rows.size:
        row = int(rows[0])
        raise ValueError(
            f"{path}: row {row} has fractions {x[row].tolist()} summing to "
            f"{x[row].sum()!r}, not to one within {FRACTION_SUM_TOL:g}")
    return active


def _relative_error_weights(archive, keep, spec) -> np.ndarray:
    """``(P,)`` float32 row weights ``1 / max(|error/value|, floor)^2``, normalised to mean one;
    ``value`` ``None`` means ``error`` is already relative, ``1 / max(|error|, floor)^2``.

    A non-finite ratio weighs zero; a table with no finite weight falls back to ones."""
    n = int(keep.sum())
    if spec["value"] is None:
        rel = np.abs(np.asarray(archive[spec["error"]], float)[keep])
    else:
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
    """``(rho, T, k, target, mask, weight)``: shells ``0 < k <= k_fit`` with a finite positive-determinant
    matrix, ``target = k_B T . S(k)^-1``; short rows repeat their last shell behind a False mask.

    The vector form appends ``active``, keeps rows with two species or more, and reads each row on its
    active block ``S_PP(k)`` alone: the test, the inverse and ``target`` (the identity outside it)."""
    with np.load(path) as z:
        keep = _keep(z, col, t_min, "structure")
        active = None
        if ACTIVE_COLUMN in col:
            active = _species_present(z, col, path)
            keep = keep & (active.sum(-1) >= 2)
            active = active[keep]
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
        present = None if active is None else np.nonzero(active[row])[0]
        for shell in np.nonzero(in_band)[0]:
            measured = S_all[row, shell]
            if present is not None:
                measured = measured[np.ix_(present, present)]
            if not np.all(np.isfinite(measured)):
                continue
            if _determinant(measured) <= 0.0:
                _LOG.warning("shell anchor: dropping a non-positive-definite "
                              "shell (row %d, k=%.4f) of %s",
                              row, k_row[shell], path)
                continue
            ks.append(k_row[shell])
            if present is None:
                targets.append(kBT[row] * np.linalg.inv(measured))
            else:
                targets.append(_embedded(kBT[row] * np.linalg.inv(measured),
                                         present, n_channels))
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
    out = (_f32(rho), _f32(T), torch.from_numpy(k_pad),
           torch.from_numpy(target_pad), torch.from_numpy(mask), weight)
    if active is None:
        return out
    return out + (torch.from_numpy(active),)


def _embedded(block: np.ndarray, present: np.ndarray, n: int) -> np.ndarray:
    """``(n, n)``: ``block`` on the rows and columns ``present``, the identity's entries elsewhere."""
    out = np.eye(n)
    out[np.ix_(present, present)] = block
    return out


def _rootinv_on_blocks(target: torch.Tensor, active: torch.Tensor,
                       root=None) -> torch.Tensor:
    """:func:`~aipf.losses.static.sinv_rootinv` (or ``root``, another matrix function) of each row's
    active principal block of ``target`` ``(P, ..., n, n)`` (``active`` ``(P, n)``), the identity outside
    it: exactly block-diagonal."""
    from aipf.losses.static import sinv_rootinv

    if root is None:
        root = sinv_rootinv
    n = target.shape[-1]
    out = torch.eye(n, dtype=target.dtype).expand(target.shape).clone()
    for pattern in torch.unique(active, dim=0):
        rows = (active == pattern).all(-1)
        idx = torch.nonzero(pattern).flatten()
        block = target[rows][..., idx[:, None], idx]
        sub = out[rows]
        sub[..., idx[:, None], idx] = root(block)
        out[rows] = sub
    return out


def _determinant(matrix: np.ndarray) -> float:
    """The 2x2 cross-multiplication the measured table was filtered with; ``det`` beyond two channels."""
    if matrix.shape == (2, 2):
        return float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    return float(np.linalg.det(matrix))


def _bulk_rows(path: Path, col, t_min: Optional[float], x_channel: int,
               weight_spec=None):
    """``(rho, T, zvec, rho_tot, target, weight)``, the zero-wavenumber anchor, ``zvec = (x1, -(1 - x1))``
    with ``x1`` the channel-1 fraction (``1 - x`` when the composition column labels channel 0).
    ``bulk_sigma``/``bulk_ok`` ``None``: the phase gate alone decides, the target need only be finite.
    The vector form is :func:`_bulk_block_rows`."""
    if ACTIVE_COLUMN in col:
        return _bulk_block_rows(path, col, t_min, weight_spec)
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


def _bulk_block_rows(path: Path, col, t_min: Optional[float], weight_spec):
    """The vector form's zero-wavenumber anchor on the concentration block, per row with ``m + 1 >= 2``
    present species ``P``: ``(rho, T, Z, rho_tot, target, weight, active, zmask, target_rootinv,
    target_root)``.

    The independent species ``I`` are the first ``m`` present ones in channel order, and ``Z`` ``(P, n, n-1)``
    has the columns ``z_i = e_i - x_i a`` (``a`` the row's active indicator), zero beyond ``m``. ``target``
    is the leading ``m x m`` block of the ``(rows, n-1, n-1)`` column, kept when finite (and refused when
    not positive definite), padded by the identity; ``zmask`` marks the ``m`` columns, ``target_rootinv``
    and ``target_root`` are its inverse square root and its square root on that block."""
    if col["bulk_sigma"] is not None or col["bulk_ok"] is not None:
        raise ValueError(
            f"bulk_sigma={col['bulk_sigma']!r} and bulk_ok={col['bulk_ok']!r} "
            f"are declared with a vector composition; their sigma < 0.5 * "
            f"target test is a scalar one with no meaning on a matrix, so "
            f"the vector form declares both None and the phase gate decides")
    with np.load(path) as z:
        keep = _keep(z, col, t_min, "structure")
        active = _species_present(z, col, path)
        keep = keep & (active.sum(-1) >= 2)
        n = active.shape[1]
        target_all = np.asarray(z[col["bulk_target"]], dtype=np.float64)
        if target_all.shape != (active.shape[0], n - 1, n - 1):
            raise ValueError(
                f"{path}: {col['bulk_target']!r} has shape "
                f"{tuple(target_all.shape)}; the concentration block of "
                f"{n} channels is ({active.shape[0]}, {n - 1}, {n - 1})")
        m_all = active.sum(-1) - 1
        finite = np.array([np.isfinite(target_all[r, :m, :m]).all()
                           for r, m in enumerate(m_all)], dtype=bool)
        keep = keep & finite
        x = np.stack([np.asarray(z[name], np.float64)[keep]
                      for name in col["composition"]], axis=-1)
        rho = _densities(z, col, keep)
        T = z[col["temperature"]][keep]
        weight = _row_weights(z, keep, weight_spec)
    rows = np.nonzero(keep)[0]
    active, target_all = active[keep], target_all[keep]
    n_rows = rows.size
    zvec = np.zeros((n_rows, n, n - 1))
    target = np.tile(np.eye(n - 1), (n_rows, 1, 1))
    zmask = np.zeros((n_rows, n - 1), dtype=bool)
    for r in range(n_rows):
        a = active[r].astype(np.float64)
        independent = np.nonzero(active[r])[0][:-1]
        m = independent.size
        for j, i in enumerate(independent):
            zvec[r, :, j] = -x[r, i] * a
            zvec[r, i, j] += 1.0
        block = target_all[r, :m, :m]
        if not np.all(np.linalg.eigvalsh(0.5 * (block + block.T)) > 0.0):
            raise ValueError(
                f"{path}: row {int(rows[r])} has a {m} x {m} "
                f"{col['bulk_target']!r} block that is not positive "
                f"definite, which no structure factor is")
        target[r, :m, :m] = block
        zmask[r, :m] = True
    zmask_t = torch.from_numpy(zmask)
    from aipf.losses.static import spd_root
    return (_f32(rho), _f32(T), _f32(zvec), _f32(rho.sum(-1)), _f32(target),
            weight, torch.from_numpy(active), zmask_t,
            _f32(_rootinv_on_blocks(torch.from_numpy(target), zmask_t)),
            _f32(_rootinv_on_blocks(torch.from_numpy(target), zmask_t, spd_root)))


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
              x_channel: Optional[int]):
    """``(rho, T, P_target)``: one absolute-pressure anchor per ok row, at the declared pressure
    times ``pressure_unit``; ``x`` is the fraction of channel ``x_channel``, or with ``x_columns``
    one fraction per channel and ``rho = n x`` (every row, pure ones too)."""
    if "x_columns" in col:
        rows = [(n * np.asarray(x, dtype=np.float64), temperature)
                for x, temperature, n in measured_rows(
                    path, **{name: col[name] for name in EOS_VECTOR_COLUMNS})]
        if not rows:
            raise ValueError(f"no usable equation-of-state row in {path}")
        target = np.full(len(rows), float(pressure) * float(pressure_unit))
        return (_f32(np.stack([r[0] for r in rows])),
                _f32(np.asarray([r[1] for r in rows])), _f32(target))
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


def _vector_form(columns: Mapping[str, Any], eos_columns: Mapping[str, Any],
                 n_channels: int) -> bool:
    """Whether a pressure-keyed declaration takes the vector form (``composition`` a tuple, ``x_columns``,
    ``active``; beyond two channels) or the single-name form (one ``composition`` and ``x_column``, read
    with ``table_keys["x_channel"]``; two channels). Each count of channels has exactly one form."""
    composition = columns.get("composition")
    vector = isinstance(composition, (tuple, list))
    eos_vector = "x_columns" in eos_columns
    if eos_vector and "x_column" in eos_columns:
        raise ValueError(
            "eos_columns declares both x_column and x_columns: the manifold's "
            "composition is one fraction or one per channel, not both")
    if vector != eos_vector:
        raise ValueError(
            f"columns['composition']={composition!r} and eos_columns "
            f"{'x_columns' if eos_vector else 'x_column'} mix the two forms: "
            f"a tuple composition goes with x_columns and one name with "
            f"x_column")
    if not vector:
        if ACTIVE_COLUMN in columns:
            raise ValueError(
                f"columns[{ACTIVE_COLUMN!r}] is read by the vector form only, "
                f"and this declaration names one composition column; a "
                f"declared column that has no effect is refused")
        if n_channels > 2:
            raise ValueError(
                f"columns['composition']={composition!r} is one name and the "
                f"tables have {n_channels} channels: one fraction cannot say "
                f"which channel it labels, or what the others hold. Declare "
                f"one fraction column per channel, columns['composition'] = "
                f"(...), with eos_columns['x_columns'] and "
                f"columns[{ACTIVE_COLUMN!r}]")
        # One channel: the zero-wavenumber projector and the manifold's own
        # composition-to-density map both walk ONE composition axis, and one channel has none.
        _refuse_beyond_two_channels(n_channels, "the composition-axis anchors")
        return False
    if n_channels < 3:
        raise ValueError(
            f"columns['composition']={tuple(composition)!r} is the vector form "
            f"and the tables have {n_channels} channels; {n_channels} channels "
            f"declare one composition name and x_column, read with "
            f"table_keys['x_channel'], so that each count of channels has "
            f"exactly one code path")
    for what, names in (("columns['composition']", composition),
                        ("eos_columns['x_columns']", eos_columns["x_columns"])):
        if isinstance(names, str) or not all(isinstance(n, str) for n in names):
            raise ValueError(
                f"{what}={names!r} is not a tuple of column names, one per "
                f"channel in channel order")
        if len(tuple(names)) != n_channels:
            raise ValueError(
                f"{what}={tuple(names)!r} has {len(tuple(names))} names "
                f"against {n_channels} density channels: one per channel, in "
                f"channel order")
    if ACTIVE_COLUMN not in columns:
        raise KeyError(
            f"the vector form declares columns[{ACTIVE_COLUMN!r}], the "
            f"species present in each row; it is not inferred from the "
            f"fractions, it is checked against them")
    return True


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

    #: The vector form only (``None`` otherwise): each row's present species ``(P, n)`` bool, per anchor.
    #: There ``mobility_target`` is zero and ``shell_target``/``shell_target_rootinv`` the identity outside a
    #: row's active block (a row with no valid shell keeps zeros, behind its False mask, as two channels
    #: do), ``bulk_zvec`` is ``(P, n, n-1)`` and ``bulk_target`` ``(P, n-1, n-1)``, with
    #: ``bulk_zmask`` ``(P, n-1)`` marking the independent species, ``bulk_target_rootinv`` the target's
    #: inverse square root on them and ``bulk_target_root`` its square root (``"s_metric_inverse"``).
    mobility_active: Optional[torch.Tensor] = None
    shell_active: Optional[torch.Tensor] = None
    bulk_active: Optional[torch.Tensor] = None
    bulk_zmask: Optional[torch.Tensor] = None
    bulk_target_rootinv: Optional[torch.Tensor] = None
    bulk_target_root: Optional[torch.Tensor] = None

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
        Two channels: ``columns["composition"]`` and ``eos_columns["x_column"]`` name one fraction, of
        channel ``system.table_keys["x_channel"]``. Beyond two (the vector form): one fraction per channel,
        ``columns["composition"]`` a tuple and ``eos_columns["x_columns"]``, plus ``columns["active"]``,
        and ``row_weight["value"]`` may be ``None`` (``error`` already relative); rows with fewer than two
        species feed ``L_P`` alone."""
        w0 = _w0_route(w0)
        root, tracked = system.paths.raw, tracked_tables(system.name)
        m_paths = _resolve(root, m_table, "m_table", tracked)
        s_paths = _resolve(root, s_table, "s_table", tracked)
        eos_paths = _resolve(root, eos_csvs, "eos_csvs", tracked)
        col = _columns(columns, TABLE_COLUMNS, "the declared table columns")
        vector = _vector_form(columns, eos_columns, len(col["densities"]))
        if vector:
            col[ACTIVE_COLUMN] = columns[ACTIVE_COLUMN]
            eos_col = _columns(eos_columns, EOS_VECTOR_COLUMNS,
                               "the declared manifold columns")
        else:
            eos_col = _columns(eos_columns, EOS_COLUMNS,
                               "the declared manifold columns")
        phase = col["phase"]
        if isinstance(phase, Mapping) and set(phase) != set(PHASE_ROLES):
            raise KeyError(
                f"the declared phase columns name roles {sorted(phase)}; a "
                f"per-role mapping names exactly {list(PHASE_ROLES)}")
        if row_weight is not None:
            row_weight = _columns(row_weight, ROW_WEIGHT_KEYS,
                                  "the declared row weighting")
        x_channel = None if vector else int(system.table_keys["x_channel"])

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

        m_rho, m_T, m_target = _cat([p[:3] for p in m_parts])
        b_rho, b_T, b_zvec, b_tot, b_target = _cat([p[:5] for p in b_parts])
        b_weight = _cat_weights([p[5] for p in b_parts])
        e_rho, e_T, e_target = _cat(e_parts)
        s_rho, s_T, s_k, s_target, s_mask = _cat_shells(
            [p[:5] for p in s_parts])
        s_weight = _cat_weights([p[5] for p in s_parts])

        from aipf.losses.static import sinv_rootinv

        masks: Dict[str, Optional[torch.Tensor]] = {}
        if vector:
            s_active = torch.cat([p[6] for p in s_parts])
            masks = dict(
                mobility_active=torch.cat([p[3] for p in m_parts]),
                shell_active=s_active,
                bulk_active=torch.cat([p[6] for p in b_parts]),
                bulk_zmask=torch.cat([p[7] for p in b_parts]),
                bulk_target_rootinv=torch.cat([p[8] for p in b_parts]),
                bulk_target_root=torch.cat([p[9] for p in b_parts]))
            s_rootinv = _rootinv_on_blocks(s_target, s_active)
        else:
            s_rootinv = sinv_rootinv(s_target)

        return cls(
            mobility_rho=m_rho, mobility_T=m_T, mobility_target=m_target,
            shell_rho=s_rho, shell_T=s_T, shell_k=s_k,
            shell_target=s_target, shell_mask=s_mask,
            shell_target_rootinv=s_rootinv,
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
            geometry=_geometry(system), w0=w0, **masks,
        )

    # -- the batch -------------------------------------------------------
    def batch(self, model, kB: float, *, curvature: bool = False) -> Dict[str, Dict[str, Any]]:
        """The entries ``compute_losses`` reads, model side rebuilt with a graph (a per-step object);
        an anchor with no rows is left out. The vector form adds each row's ``active`` mask (and the bulk
        anchor's ``zmask``, ``target_rootinv`` and ``target_root``): the losses compare the active
        principal block of ``H`` and ``M`` alone. ``curvature=True`` also keeps the bulk anchor's
        ``d2 f_loc / d rho2``, the one its ``H0`` is built from, under ``"curvature"``, for a penalty that
        reads it at other ``k``."""
        refuse_two_dimensions(model_ndim(model), "the anchor losses")
        w_hat = _w_hat_of(model)
        out: Dict[str, Dict[str, Any]] = {}
        if self.mobility_rho.shape[0]:
            out["anchor_M"] = {
                "M_pred": _mobility_at(model, self.mobility_rho,
                                       self.mobility_T),
                "M_target": self.mobility_target,
            }
            if self.mobility_active is not None:
                out["anchor_M"]["active"] = self.mobility_active
        if self.shell_rho.shape[0]:
            H = (_curvature(model, self.shell_rho, self.shell_T).unsqueeze(1)
                 + w_hat(self.shell_k))
            out["anchor_S"] = {
                "H": H, "target": self.shell_target, "mask": self.shell_mask,
                "target_rootinv": self.shell_target_rootinv,
                "row_weight": self.shell_weight,
            }
            if self.shell_active is not None:
                out["anchor_S"]["active"] = self.shell_active
        if self.bulk_rho.shape[0]:
            local = _curvature(model, self.bulk_rho, self.bulk_T)
            H0 = (local
                  + _w_hat_zero(model, w_hat, self.bulk_rho, self.geometry,
                                self.w0))
            out["anchor_bulk"] = {
                "H0": H0, "kBT": float(kB) * self.bulk_T,
                "zvec": self.bulk_zvec, "rho_tot": self.bulk_rho_tot,
                "target": self.bulk_target, "row_weight": self.bulk_weight,
                "sigma": self.bulk_sigma,
            }
            if curvature:
                out["anchor_bulk"]["curvature"] = local
            if self.bulk_active is not None:
                out["anchor_bulk"].update(
                    active=self.bulk_active, zmask=self.bulk_zmask,
                    target_rootinv=self.bulk_target_rootinv,
                    target_root=self.bulk_target_root)
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


__all__ = ["AnchorTables", "TABLE_COLUMNS", "EOS_COLUMNS", "EOS_VECTOR_COLUMNS",
           "ACTIVE_COLUMN", "TABLE_KEYS", "TEMPERATURE_COLUMNS", "W0_ROUTES"]
