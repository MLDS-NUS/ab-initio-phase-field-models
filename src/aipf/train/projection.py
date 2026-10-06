"""The ``k_z = 0`` plane of a three-dimensional mode archive, read as the timeline a two-dimensional model
trains on.

A mode archive is three-dimensional: labels ``(n_x, n_y, n_z)`` and the atom sums ``rho_k`` of a bulk
cell. :func:`project_kz0` keeps the labels with ``n_z == 0``, drops their third index, and reads the run
in the reference cell's ``(Lx, Ly)``. The window's measure is then ``A_ref * depth``,
``A_ref = Lx_ref Ly_ref``, so the half spectrum a window scatters is

* volumetric (``areal=False``, ``depth = Lz_ref``): ``rho_hat = rho_k / V_ref``, the coefficient of the
  z mean of the three-dimensional density, in particles per unit volume;
* areal (``areal=True``, ``depth = 1``): ``rho_hat = rho_k / A_ref = Lz_ref * (the volumetric one)``, the
  coefficient of the density integrated along z, in particles per unit area.

The same ``depth`` is what a noisy two-dimensional rollout of the trained model declares
(:func:`projection_depth`), since its cell volume is ``dA * depth``. Only an archive that records its
reference cell can be projected: the projection needs one ``Lz`` for every frame, which the frames' own
boxes of an NPT run do not have. See docs/reference/data.md, "The k_z = 0 projection"."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .dataset import ModeRun

#: The projections a two-dimensional training run reads its three-dimensional archives through: the
#: ``k_z = 0`` plane in volumetric densities (``"kz0-volumetric"``) or in areal ones (``"kz0-areal"``).
PROJECTIONS = ("kz0-volumetric", "kz0-areal")


def check_projection(projection: Optional[str]) -> Optional[str]:
    """``projection`` when it is ``None`` or one of :data:`PROJECTIONS`; otherwise ``ValueError``."""
    if projection is not None and projection not in PROJECTIONS:
        raise ValueError(
            f"projection={projection!r} is not one of {PROJECTIONS}: the k_z = 0 plane in volumetric "
            f"densities (rho_k / V_ref) or in areal ones (rho_k / (Lx_ref Ly_ref)); None reads the "
            f"archive in three dimensions")
    return projection


def is_areal(projection: str) -> bool:
    """Whether one of :data:`PROJECTIONS` carries areal densities."""
    return check_projection(projection) == "kz0-areal"


def _no_cell(where: str) -> ValueError:
    return ValueError(
        f"{where} records no reference cell, so it has no single Lz to project along: the k_z = 0 "
        f"projection reads every frame in one cell (Lx_ref, Ly_ref) and divides by Lz_ref. Write the "
        f"archive with modes_from_dump(reference_box=...), a stated cell (Lx, Ly, Lz) or "
        f"'first_frame'; the default time-mean rule records none")


def _depth_of(cell: np.ndarray, areal: bool) -> float:
    return 1.0 if areal else float(cell[2])


def project_kz0(run: ModeRun, *, areal: bool) -> ModeRun:
    """``run``'s ``k_z = 0`` plane as a two-dimensional run: the labels with ``n_z == 0`` as ``(n_x, n_y)``,
    their amplitudes unchanged, the boxes and the reference cell cut to ``(Lx, Ly)``, and
    :attr:`ModeRun.depth` set (``Lz_ref`` volumetric, ``1.0`` areal), which a window's volume is
    multiplied by.

    Refuses a run without a reference cell, a run that is not three-dimensional, and one with no
    ``n_z == 0`` label."""
    if run.reference_box is None:
        raise _no_cell(f"run {run.tag!r}")
    labels = np.asarray(run.labels)
    cell = np.asarray(run.reference_box, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != 3 or cell.shape != (3,) or run.depth is not None:
        raise ValueError(
            f"run {run.tag!r} is not a three-dimensional run (labels {tuple(labels.shape)}, reference "
            f"cell {tuple(cell.shape)}): the projection reads (n_x, n_y, n_z) labels once")
    keep = labels[:, 2] == 0
    if not keep.any():
        raise ValueError(
            f"run {run.tag!r} has no label with n_z == 0: its k_z = 0 plane is empty")
    boxes = np.asarray(run.boxes)
    return dataclasses.replace(
        run,
        amplitudes=np.ascontiguousarray(run.amplitudes[..., keep]),
        labels=np.ascontiguousarray(labels[keep][:, :2]),
        boxes=np.ascontiguousarray(boxes[..., :2]),
        reference_box=cell[:2].copy(),
        depth=_depth_of(cell, areal))


def projection_depth(run_or_dir: Union[ModeRun, str, Path], *, areal: bool) -> float:
    """The ``depth`` a noisy two-dimensional rollout declares for a model trained through the ``k_z = 0``
    projection of this archive: ``1.0`` for areal densities, the reference cell's ``Lz`` for volumetric
    ones. ``run_or_dir`` is a :class:`ModeRun` as read (not yet projected), a run directory holding
    ``modes.npz``, or that file. An archive without a reference cell is refused."""
    if isinstance(run_or_dir, ModeRun):
        if run_or_dir.depth is not None:
            raise ValueError(
                f"run {run_or_dir.tag!r} is already projected (depth={run_or_dir.depth!r}); pass the "
                f"run as read, whose reference cell holds Lz")
        if run_or_dir.reference_box is None:
            raise _no_cell(f"run {run_or_dir.tag!r}")
        return _depth_of(np.asarray(run_or_dir.reference_box, dtype=np.float64), areal)
    from aipf.pipeline.modes import REFERENCE_BOX_KEY

    path = Path(run_or_dir)
    if path.is_dir():
        path = path / "modes.npz"
    with np.load(path) as payload:
        if REFERENCE_BOX_KEY not in payload:
            raise _no_cell(str(path))
        cell = np.array(payload[REFERENCE_BOX_KEY], dtype=np.float64)
    if cell.shape != (3,):
        raise ValueError(
            f"{path}: {REFERENCE_BOX_KEY} has shape {tuple(cell.shape)}, which is not one cell of "
            f"three edge lengths")
    return _depth_of(cell, areal)


def common_lz(cells: Sequence[Tuple[str, np.ndarray]]) -> Optional[float]:
    """The one ``Lz`` of the reference cells ``(where, cell (3,))`` a projected run trains on, or ``None``
    for none; ``ValueError`` naming them when they differ. A trained two-dimensional model has one
    ``depth`` (and, under areal densities, one slab thickness its densities are per unit area of), so
    archives of several ``Lz_ref`` are refused rather than mixed: state one cell for all of them."""
    values = {}
    for where, cell in cells:
        values.setdefault(float(np.asarray(cell, dtype=np.float64)[2]), []).append(str(where))
    if len(values) > 1:
        shown = "; ".join(f"Lz_ref={lz!r}: {', '.join(names[:3])}"
                          + (f" and {len(names) - 3} more" if len(names) > 3 else "")
                          for lz, names in sorted(values.items()))
        raise ValueError(
            f"the projected runs are read in reference cells of {len(values)} different Lz ({shown}): "
            f"a two-dimensional model has one depth, so write every archive in one stated cell, "
            f"modes_from_dump(reference_box=(Lx, Ly, Lz))")
    return next(iter(values), None)


def archive_cells(root, pattern: str, *, file_name: str, key: str,
                  exclude_tags: Sequence[str] = ()) -> List[Tuple[Path, Optional[np.ndarray]]]:
    """``(archive, reference cell or None)`` for every run :func:`aipf.train.dataset.read_mode_runs`
    would read under ``root`` (the same glob, exclusions and missing files skipped), read from the
    archive's index and its ``key`` alone: the check a projection makes before anything is written."""
    excluded = set(exclude_tags)
    out = []
    for directory in sorted(Path(root).glob(pattern)):
        path = directory / file_name
        if directory.name in excluded or not path.is_file():
            continue
        with np.load(path) as payload:
            cell = (np.array(payload[key], dtype=np.float64) if key in payload else None)
        out.append((path, cell))
    return out
