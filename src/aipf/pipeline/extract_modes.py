"""A trajectory to the exact Fourier amplitudes a model is trained on.

    rho_c(n, t) = sum over the atoms of channel c of exp(-2 pi i n . s_i(t)),   s_i = (r_i - lo) / L(t)

``n`` is an integer triple, conserved as the box breathes: :func:`wavevectors` gives ``k = 2 pi n / L``.
Raw per-channel amplitudes in the caller's channel order, layout ``(n_frames, n_channels, n_modes)``.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

import numpy as np
import torch

# Imported by name: the package's one statement of a channel declaration and a device.
from .kde import DEVICES, _as_atom_types, _resolve_device

TWO_PI = 2.0 * np.pi

#: Box the mode labels are chosen on: ``"time_mean"`` (barostatted run) or ``"first_frame"`` (fixed box).
#: The two give mode sets of different size.
REFERENCE_BOXES = ("time_mean", "first_frame")

#: Row order of the mode list: ``"lexicographic"`` (enclosing cube, first axis slowest)
#: or ``"shell"`` (ascending wavenumber, stable within a shell).
ORDERINGS = ("lexicographic", "shell")

#: How the atom sum is evaluated: ``"dense"`` (whole phase matrix, on a device) or ``"separable"``
#: (per-axis factors, half cube plus conjugate symmetry, processor).
ROUTES = ("dense", "separable")

#: Width of the stored integer labels, as the archive carries them.
_LABEL_DTYPE = np.int16


def _edge_lengths(box) -> np.ndarray:
    """One box as ``(3,)`` positive float64 edges, or a refusal saying why."""
    lengths = np.asarray(box, dtype=np.float64)
    if lengths.shape != (3,):
        raise ValueError(
            f"box has shape {tuple(lengths.shape)}, which is not three edge "
            f"lengths"
        )
    if not np.all(lengths > 0.0):
        raise ValueError(f"box has a non-positive edge: {lengths.tolist()}")
    return lengths


def _labels(nvec) -> np.ndarray:
    """A mode list as ``(n_modes, 3)`` float64, or a refusal saying why."""
    labels = np.asarray(nvec)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError(
            f"nvec has shape {tuple(labels.shape)}, which is not a list of "
            f"integer triples"
        )
    return labels.astype(np.float64)


def reference_box(box_lengths, *, rule: str) -> np.ndarray:
    """The ``(3,)`` box a whole timeline's labels are chosen on.

    ``box_lengths`` are the KEPT frames' ``(n_frames, 3)`` edges. ``rule`` is one of :data:`REFERENCE_BOXES`.
    """
    if rule not in REFERENCE_BOXES:
        raise ValueError(f"rule={rule!r} is not one of {REFERENCE_BOXES}")
    lengths = np.asarray(box_lengths, dtype=np.float64)
    if lengths.ndim != 2 or lengths.shape[1] != 3:
        raise ValueError(
            f"box_lengths has shape {tuple(lengths.shape)}, which is not a "
            f"timeline of three edge lengths per frame"
        )
    if lengths.shape[0] < 1:
        raise ValueError(
            "box_lengths has no frames: there is no box to choose the mode "
            "labels on, and an empty average is a silent nan"
        )
    if not np.all(lengths > 0.0):
        raise ValueError(
            f"box_lengths has a non-positive edge: {lengths.min():.6g}"
        )
    if rule == "time_mean":
        return lengths.mean(axis=0)
    return lengths[0].copy()


def mode_set(box, *, k_cut: float, ordering: str) -> np.ndarray:
    """``(n_modes, 3)`` integer labels with ``|2 pi n / L| <= k_cut`` on the reference box ``L``.

    ``k_cut`` is in the reciprocal of the coordinates' unit. ``ordering`` is one of :data:`ORDERINGS`.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering={ordering!r} is not one of {ORDERINGS}")
    k_cut = float(k_cut)
    if not k_cut > 0.0:
        raise ValueError(
            f"k_cut={k_cut!r} is not positive: it is the largest wavenumber "
            f"kept, and at or below zero the only mode is the uniform one"
        )
    lengths = _edge_lengths(box)
    reach = np.floor(k_cut * lengths / TWO_PI).astype(int)
    axes = [np.arange(-far, far + 1) for far in reach]
    labels = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    k = TWO_PI * labels / lengths
    inside = (k ** 2).sum(1) <= k_cut ** 2
    labels = labels[inside]
    if ordering == "shell":
        labels = labels[np.argsort(np.linalg.norm(k[inside], axis=1),
                                   kind="stable")]
    return labels.astype(_LABEL_DTYPE)


def wavevectors(nvec, box_lengths) -> np.ndarray:
    """``k = 2 pi n / L``, ``(n_modes, 3)`` for one box or ``(n_frames, n_modes, 3)`` for a timeline."""
    labels = _labels(nvec)
    lengths = np.asarray(box_lengths, dtype=np.float64)
    if lengths.ndim == 1:
        return TWO_PI * labels / _edge_lengths(lengths)
    if lengths.ndim != 2 or lengths.shape[1] != 3:
        raise ValueError(
            f"box_lengths has shape {tuple(lengths.shape)}, which is neither "
            f"three edge lengths nor a timeline of them"
        )
    return TWO_PI * labels[None, :, :] / lengths[:, None, :]


def mode_amplitudes(positions, types, box_bounds, nvec, *,
                    atom_types: Sequence[int], route: str,
                    device: str | None = None) -> np.ndarray:
    """One frame's exact ``(n_channels, n_modes)`` complex128 amplitudes.

    ``box_bounds`` ``(3, 2)`` is this frame's. ``device`` is needed by ``"dense"``, refused otherwise.
    """
    if route not in ROUTES:
        raise ValueError(f"route={route!r} is not one of {ROUTES}")
    if route == "dense":
        if device is None:
            raise ValueError(
                f"route='dense' needs device: it builds the phase matrix with "
                f"the array library and the answer is not identical on every "
                f"one, so the device is stated rather than picked. Name one of "
                f"{DEVICES}"
            )
        resolved = _resolve_device(device)
    elif device is not None:
        raise ValueError(
            f"route='separable' has no device to honour: the contraction runs "
            f"on the processor. Accepting device={device!r} and ignoring it "
            f"would report an accelerator that never ran"
        )
    channels = _as_atom_types(atom_types)
    pos, typ, lo, lengths = _frame_arrays(positions, types, box_bounds)
    labels = _labels(nvec)
    scaled = (pos - lo) / lengths
    if route == "dense":
        return _dense(scaled, typ, labels, channels, resolved)
    return _separable(scaled, typ, _separable_plan(labels), channels)


def _frame_arrays(positions, types, box_bounds):
    """One frame's arrays as float64 coordinates, types, origin and edges."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"positions has shape {tuple(pos.shape)}, which is not one "
            f"coordinate triple per atom"
        )
    typ = np.asarray(types)
    if typ.shape != (pos.shape[0],):
        raise ValueError(
            f"types has shape {tuple(typ.shape)} against {pos.shape[0]} "
            f"atoms: there is one dump type per atom"
        )
    bounds = np.asarray(box_bounds, dtype=np.float64)
    if bounds.shape != (3, 2):
        raise ValueError(
            f"box_bounds has shape {tuple(bounds.shape)}, not (3, 2): it is "
            f"a lower and an upper bound per axis"
        )
    lengths = bounds[:, 1] - bounds[:, 0]
    if not np.all(lengths > 0.0):
        raise ValueError(
            f"box_bounds has a non-positive edge: {lengths.tolist()}"
        )
    return pos, typ, bounds[:, 0], lengths


def _dense(scaled, typ, labels, channels, resolved) -> np.ndarray:
    """Build the whole phase matrix and reduce it, one channel at a time."""
    scaled_t = torch.tensor(scaled, dtype=torch.float64, device=resolved)
    labels_t = torch.tensor(labels, dtype=torch.float64, device=resolved)
    typ_t = torch.tensor(np.asarray(typ), device=resolved)
    out = np.empty((len(channels), labels.shape[0]), np.complex128)
    for c, dump_type in enumerate(channels):
        mine = scaled_t[typ_t == dump_type]
        phase = TWO_PI * (labels_t @ mine.T)
        out[c] = torch.exp(-1j * phase).sum(1).cpu().numpy()
    return out


def _separable_plan(labels):
    """The enclosing integer cube: half-extent, wanted-label mask, permutation to the caller's row order."""
    labels = np.asarray(labels).astype(np.int64)
    reach = np.abs(labels).max(0)
    shape = 2 * reach + 1
    wanted = np.zeros(tuple(shape), bool)
    wanted[tuple((labels + reach).T)] = True
    scan = ((labels + reach) * np.array([shape[1] * shape[2], shape[2], 1])
            ).sum(1)
    return reach, wanted, np.argsort(scan)


def _separable(scaled, typ, plan, channels) -> np.ndarray:
    """Factor the phase per axis, contract, and mirror the conjugate half (exact for real coordinates)."""
    reach, wanted, order = plan
    shape = tuple(2 * far + 1 for far in reach)
    axis_x = np.arange(-reach[0], reach[0] + 1, dtype=float)
    axis_y = np.arange(-reach[1], reach[1] + 1, dtype=float)
    axis_z = np.arange(0, reach[2] + 1, dtype=float)
    out = np.empty((len(channels), int(wanted.sum())), np.complex128)
    for c, dump_type in enumerate(channels):
        mine = scaled[typ == dump_type]
        e_x = np.exp(-2j * np.pi * np.outer(axis_x, mine[:, 0]))
        e_y = np.exp(-2j * np.pi * np.outer(axis_y, mine[:, 1]))
        e_z = np.exp(-2j * np.pi * np.outer(axis_z, mine[:, 2]))
        planes = (e_x[:, None, :] * e_y[None, :, :]).reshape(
            shape[0] * shape[1], -1)
        half = (planes @ e_z.T).reshape(shape[0], shape[1], reach[2] + 1)
        cube = np.empty(shape, np.complex128)
        cube[:, :, reach[2]:] = half
        cube[:, :, :reach[2]] = np.conj(half[::-1, ::-1, reach[2]:0:-1])
        out[c, order] = cube[wanted]
    return out


def mode_series(frames: Iterable, nvec, *, atom_types: Sequence[int],
                route: str, device: str | None = None,
                dtype=np.complex64) -> tuple[np.ndarray, np.ndarray]:
    """Extract every frame, each in its own box, against one fixed mode list.

    Returns ``(n_frames, n_channels, n_modes)`` amplitudes in the complex ``dtype`` and int64 steps.
    Sums accumulate in double precision and are rounded once.
    """
    store = np.dtype(dtype)
    if store.kind != "c":
        raise ValueError(
            f"dtype={store} is not complex: an amplitude has a phase, and "
            f"storing it as a real would keep half of one"
        )
    rows = []
    steps = []
    for frame in frames:
        rows.append(mode_amplitudes(frame.positions, frame.types,
                                    frame.box_bounds, nvec,
                                    atom_types=atom_types, route=route,
                                    device=device).astype(store))
        steps.append(int(frame.timestep))
    if not rows:
        raise ValueError(
            "frames has no frames: there is no timeline to extract, and an "
            "empty series would only fail later with the mode list forgotten"
        )
    return np.stack(rows), np.array(steps, dtype=np.int64)


def frames_to_skip(timesteps, *, equilibration_steps: int | None,
                   production_steps: int | None) -> int:
    """How many leading frames are the run being prepared, not the state.

    Zero if either count is ``None`` or ``max(timesteps) <= production_steps``,
    else the number of frames with step ``< equilibration_steps``.
    """
    for name, value in (("equilibration_steps", equilibration_steps),
                        ("production_steps", production_steps)):
        if value is not None and int(value) < 0:
            raise ValueError(f"{name}={value} is negative: it is a step count")
    steps = np.asarray(timesteps)
    if steps.ndim != 1:
        raise ValueError(
            f"timesteps has shape {tuple(steps.shape)}: there is one timestep "
            f"per frame"
        )
    if equilibration_steps is None or production_steps is None:
        return 0
    if steps.size == 0:
        return 0
    if int(steps.max()) <= int(production_steps):
        return 0
    return int((steps < int(equilibration_steps)).sum())
