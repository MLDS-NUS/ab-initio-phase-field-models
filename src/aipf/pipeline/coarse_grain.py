"""A trajectory to the field pairs ``(rho, d rho / dt)`` a dynamical model is trained on.

Stages: read a dump, coarse-grain a timeline, Savitzky-Golay smooth and differentiate it, choose the
emitted frames. Stores raw per-channel number densities ``(n_frames, n_species, Gx, Gy, Gz)``.
A ratio is :func:`composition_field`.
"""
from __future__ import annotations


from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .kde import density_field

#: Emitted-frame spread over the kept tail: ``"log"`` (dense at the start, at most the requested count)
#: or ``"uniform"``.
SPACINGS = ("log", "uniform")

#: Savitzky-Golay edge treatment: ``"interp"`` (polynomial fit to the end windows)
#: or ``"nearest"`` (repeat the end samples). They differ within half a window of each end.
EDGE_MODES = ("interp", "nearest")

#: Regularised composition: ``"offset"`` ``num / (total + reg)``, ``"floored"`` ``num / max(total, reg)``,
#: ``"filled"`` ``num / total`` where ``total > reg`` and ``fill`` elsewhere. No default form or constant.
COMPOSITION_FORMS = ("offset", "floored", "filled")

#: Header lines of a LAMMPS text dump frame, ``ITEM: TIMESTEP`` to ``ITEM: ATOMS`` inclusive.
_HEADER_LINES = 9


@dataclass(frozen=True)
class Frame:
    """One dump frame: ``timestep`` (integrator step, not a time), ``box_bounds`` ``(3, 2)`` for THIS frame,
    ``positions`` ``(n_atoms, 3)`` float64 sorted by atom id, ``types`` ``(n_atoms,)`` int64."""

    timestep: int
    box_bounds: np.ndarray
    positions: np.ndarray
    types: np.ndarray


def read_dump(path, *, stride: int = 1) -> Iterator[Frame]:
    """Yield :class:`Frame` objects from a LAMMPS text dump, one in ``stride``.

    Atoms sorted by id, columns chosen by name, a truncated trailing frame ends the iteration.
    """
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride={stride!r} is below one: it counts frames")
    index = -1
    with open(path) as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            index += 1
            header = [handle.readline() for _ in range(_HEADER_LINES - 1)]
            if not all(header):
                return
            timestep = int(header[0])
            n_atoms = int(header[2])
            body = [handle.readline() for _ in range(n_atoms)]
            if not all(body):
                return
            if index % stride:
                continue
            columns = header[7].split()[2:]
            frame = _parse_frame(timestep, header[4:7], columns, body)
            if frame is None:
                return
            yield frame


def read_timesteps(path) -> np.ndarray:
    """Every frame's integrator step, ``(n_frames,)`` int64, WITHOUT parsing an atom.

    Stops at a short frame exactly as :func:`read_dump` does.
    """
    steps = []
    with open(path) as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            header = [handle.readline() for _ in range(_HEADER_LINES - 1)]
            if not all(header):
                break
            n_atoms = int(header[2])
            body = [handle.readline() for _ in range(n_atoms)]
            if not all(body):
                break
            steps.append(int(header[0]))
    return np.array(steps, dtype=np.int64)


#: Why the last bytes of a chained dump were not read as a frame: ``"nul_padding"`` (only NUL bytes, as
#: a file system leaves after a killed writer) or ``"truncated_frame"`` (a frame cut short).
TAIL_REASONS = ("nul_padding", "truncated_frame")

#: Longest line the chain reader takes, in bytes; a dump line is far shorter, a NUL run is not a line.
_LINE_LIMIT = 1 << 16

#: The line a frame starts with, as the chain reader searches for it in raw bytes.
_TIMESTEP = b"ITEM: TIMESTEP"


def read_dump_chain(paths, *, report: list) -> Iterator[Frame]:
    """Yield the frames of a restart chain of LAMMPS text dumps, in file order, each step once.

    A frame whose timestep is at or below the last one kept is a restart's overlap and is dropped. A file
    may end in NUL padding or a cut frame (:data:`TAIL_REASONS`): its complete frames are kept, the tail is
    skipped and the next file read. Anything else that is not a frame, with a frame after it, raises.
    :func:`read_dump` is left as it is: this reader holds every line to a newline and the frame layout.

    ``report`` gets one dict per file as it is opened, final once the iteration ends: ``source``,
    ``frames_kept``, ``frames_overlap``, ``tail_bytes`` and ``tail`` (``None`` or a reason).
    """
    last = None
    for path in paths:
        entry = {"source": str(path), "frames_kept": 0, "frames_overlap": 0,
                 "tail_bytes": 0, "tail": None}
        report.append(entry)
        for frame in _complete_frames(path, entry):
            if last is not None and frame.timestep <= last:
                entry["frames_overlap"] += 1
                continue
            last = frame.timestep
            entry["frames_kept"] += 1
            yield frame


def _complete_frames(path, entry: dict) -> Iterator[Frame]:
    """One file's complete frames; where they stop, :func:`_classify_tail` fills ``entry`` or raises."""
    with open(path, "rb") as handle:
        while True:
            start = handle.tell()
            first = handle.readline(_LINE_LIMIT)
            if not first:
                return
            frame = _strict_frame(first, handle)
            if frame is None:
                _classify_tail(path, handle, start, entry)
                return
            yield frame


def _strict_frame(first: bytes, handle) -> Frame | None:
    """The frame starting at line ``first``, or ``None`` if the bytes are not one whole frame."""
    if not first.startswith(_TIMESTEP) or not _whole(first):
        return None
    header = [handle.readline(_LINE_LIMIT) for _ in range(_HEADER_LINES - 1)]
    if not all(_whole(line) for line in header) \
            or not header[1].startswith(b"ITEM: NUMBER OF ATOMS") \
            or not header[3].startswith(b"ITEM: BOX BOUNDS") \
            or not header[7].startswith(b"ITEM: ATOMS"):
        return None
    try:
        timestep, n_atoms = int(header[0]), int(header[2])
        for line in header[4:7]:
            [float(v) for v in line.split()[:2]]
    except ValueError:
        return None
    body = [handle.readline(_LINE_LIMIT) for _ in range(n_atoms)]
    if not all(_whole(line) for line in body):
        return None
    return _parse_frame(timestep, [line.decode() for line in header[4:7]],
                        header[7].decode().split()[2:],
                        [line.decode() for line in body])


def _whole(line: bytes) -> bool:
    """Whether a raw line is a written line: ended by a newline, without a NUL byte."""
    return line.endswith(b"\n") and b"\0" not in line


def _classify_tail(path, handle, start: int, entry: dict) -> None:
    """Name the bytes from ``start`` to the end as a tail, or raise if a frame header follows them."""
    size = handle.seek(0, 2)
    handle.seek(start)
    only_nul = True
    carry = b""
    position = start
    while True:
        chunk = handle.read(1 << 20)
        if not chunk:
            break
        if only_nul and chunk.strip(b"\0"):
            only_nul = False
        window = carry + chunk
        # the frame that failed starts at ``start``; only a header after it is a frame that follows
        found = window.find(_TIMESTEP, 1 if position == start else 0)
        if found >= 0:
            at = position - len(carry) + found
            raise ValueError(
                f"{path}: the bytes from offset {start} are not a complete "
                f"frame, and a frame header follows at offset {at}. A cut "
                f"or padded tail is skipped, corruption inside a file is "
                f"not: the frames after it would join the chain with a gap "
                f"nobody sees")
        carry = window[-(len(_TIMESTEP) - 1):]
        position += len(chunk)
    entry["tail_bytes"] = size - start
    entry["tail"] = "nul_padding" if only_nul else "truncated_frame"


def _parse_frame(timestep, bound_lines, columns, body) -> Frame | None:
    """One frame's header and atom rows into a :class:`Frame`, or ``None`` for a truncated row.
    """
    bounds = np.array([[float(v) for v in line.split()[:2]]
                       for line in bound_lines])
    try:
        wanted = [columns.index(name) for name in ("id", "type", "x", "y", "z")]
    except ValueError as exc:
        raise ValueError(
            f"dump columns {columns} lack one of id, type, x, y, z, so there "
            f"is no way to tell which field is which"
        ) from exc

    block = None
    if _all_numeric(body[0]):
        try:
            flat = np.fromstring("".join(body), dtype=np.float64, sep=" ")
        except ValueError:
            flat = None
        if flat is not None and flat.size == len(body) * len(columns):
            block = flat.reshape(len(body), len(columns))[:, wanted]
    if block is None:
        rows = [line.split() for line in body]
        if any(len(row) < len(columns) for row in rows):
            return None
        try:
            block = np.array([[row[j] for j in wanted] for row in rows],
                             dtype=np.float64)
        except ValueError:
            return None

    block = block[np.argsort(block[:, 0])]
    return Frame(
        timestep=timestep,
        box_bounds=bounds,
        positions=np.ascontiguousarray(block[:, 2:5]),
        types=block[:, 1].astype(np.int64),
    )


def _all_numeric(line: str) -> bool:
    """Whether every whitespace-separated field of a line parses as a float."""
    try:
        [float(field) for field in line.split()]
    except ValueError:
        return False
    return True


def grid_for_spacing(box_bounds, *, spacing: float,
                     min_points: int) -> tuple[int, int, int]:
    """Per-axis grid ``max(min_points, round(L / spacing))`` on a ``(3, 2)`` box, in coordinate units."""
    spacing = float(spacing)
    if not spacing > 0.0:
        raise ValueError(f"spacing={spacing!r} is not positive")
    min_points = int(min_points)
    if min_points < 1:
        raise ValueError(f"min_points={min_points!r} is below one point")
    bounds = np.asarray(box_bounds, dtype=np.float64)
    if bounds.shape != (3, 2):
        raise ValueError(
            f"box_bounds has shape {tuple(bounds.shape)}, not (3, 2)"
        )
    lengths = bounds[:, 1] - bounds[:, 0]
    if not np.all(lengths > 0.0):
        raise ValueError(
            f"box_bounds has a non-positive edge: {lengths.tolist()}"
        )
    return tuple(max(min_points, int(round(float(length) / spacing)))
                 for length in lengths)


def density_series(frames: Iterable[Frame], grid, *, sigma: float,
                   atom_types: Sequence[int], method: str,
                   device: str = "auto",
                   dtype: torch.dtype = torch.float32,
                   cutoff_in_sigma: float | None = None) -> np.ndarray:
    """Coarse-grain every frame, each in its own box, onto one fixed grid.

    Returns ``(n_frames, n_species, Gx, Gy, Gz)`` raw number density. Other arguments go to ``density_field``.
    """
    fields = []
    for frame in frames:
        field = density_field(
            frame.positions, frame.types, frame.box_bounds, grid,
            sigma=sigma, atom_types=atom_types, method=method,
            device=device, dtype=dtype, cutoff_in_sigma=cutoff_in_sigma,
        )
        fields.append(field.cpu().numpy())
    if not fields:
        raise ValueError(
            "frames is empty: there is no timeline to coarse-grain, and an "
            "empty series would only fail later with the grid forgotten"
        )
    return np.stack(fields)


def smooth_in_time(series, *, window: int, polyorder: int, dt: float,
                   edge_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay smooth a timeline along axis 0 and differentiate it. Returns ``(smoothed, derivative)``.

    ``window`` odd and at most the frame count, ``polyorder < window``, ``dt`` the frame interval.
    ``edge_mode`` is one of :data:`EDGE_MODES`.
    """
    if edge_mode not in EDGE_MODES:
        raise ValueError(f"edge_mode={edge_mode!r} is not one of {EDGE_MODES}")
    values = np.asarray(series)
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError(
            f"series has shape {tuple(values.shape)}: axis 0 is the frame "
            f"axis and there has to be one"
        )
    window = int(window)
    polyorder = int(polyorder)
    if window % 2 == 0:
        raise ValueError(
            f"window={window} is even: a centred fit needs an odd number of "
            f"frames, or the sample it reports at is not the one it is at"
        )
    if window > values.shape[0]:
        raise ValueError(
            f"window={window} over a timeline of {values.shape[0]} frames: "
            f"the window has to fit. Narrowing it silently would apply a "
            f"filter the caller did not ask for"
        )
    if polyorder >= window:
        raise ValueError(
            f"polyorder={polyorder} in a window of {window}: the fit would "
            f"be exact and the filter would do nothing"
        )
    dt = float(dt)
    if not dt > 0.0:
        raise ValueError(
            f"dt={dt!r} is not positive: it is the time between frames and "
            f"it scales the derivative"
        )
    from scipy.signal import savgol_filter

    smoothed = savgol_filter(values, window, polyorder, deriv=0, axis=0,
                             mode=edge_mode)
    derivative = savgol_filter(values, window, polyorder, deriv=1, delta=dt,
                               axis=0, mode=edge_mode)
    return smoothed, derivative


def emit_indices(n_frames: int, *, n_melt: int, emit_n: int,
                 spacing: str) -> np.ndarray:
    """Strictly increasing frame indices in ``[n_melt, n_frames)``, at most ``emit_n``, spread by ``spacing``.
    """
    if spacing not in SPACINGS:
        raise ValueError(f"spacing={spacing!r} is not one of {SPACINGS}")
    n_frames = int(n_frames)
    n_melt = int(n_melt)
    emit_n = int(emit_n)
    if n_frames < 0:
        raise ValueError(f"n_frames={n_frames} is negative")
    if n_melt < 0:
        raise ValueError(
            f"n_melt={n_melt} is negative: it is a count of frames dropped "
            f"from the front"
        )
    if n_melt > n_frames:
        raise ValueError(
            f"n_melt={n_melt} over a timeline of {n_frames} frames: more "
            f"would be discarded than exist, and the result would be an "
            f"empty emission that looks like a short run"
        )
    if emit_n < 1:
        raise ValueError(
            f"emit_n={emit_n} is below one: emitting nothing is not a "
            f"sampling choice, it is a mistake with no error"
        )

    available = n_frames - n_melt
    if emit_n >= available:
        return np.arange(n_melt, n_frames)
    if spacing == "uniform":
        offsets = np.linspace(0, available - 1, emit_n)
    else:
        offsets = np.geomspace(1, available, emit_n) - 1.0
    return np.unique(np.round(offsets).astype(int)) + n_melt


def composition_field(density, *, numerator: int, form: str,
                      regulariser: float, fill: float | None = None):
    """One channel's share of the total over every channel, regularised by ``form``.

    ``density`` is ``(..., n_species, Gx, Gy, Gz)``. ``fill`` is required by ``"filled"`` only.
    """
    if form not in COMPOSITION_FORMS:
        raise ValueError(f"form={form!r} is not one of {COMPOSITION_FORMS}")
    regulariser = float(regulariser)
    if not regulariser > 0.0:
        raise ValueError(
            f"regulariser={regulariser!r} is not positive: it exists to keep "
            f"the division finite and cannot do that at zero or below"
        )
    if form == "filled":
        if fill is None:
            raise ValueError(
                "form='filled' needs fill: it is the value reported where "
                "there is no density, and there is no neutral choice for it"
            )
        fill = float(fill)
    elif fill is not None:
        raise ValueError(
            f"form={form!r} has no fill to honour: it divides everywhere and "
            f"never reaches for a replacement value. Accepting fill and "
            f"ignoring it would report a floor that was never applied"
        )

    if getattr(density, "ndim", 0) < 4:
        raise ValueError(
            f"density has shape {tuple(np.shape(density))}: the channel axis "
            f"is the fourth from the end, so at least four axes are needed"
        )
    n_species = density.shape[-4]
    numerator = int(numerator)
    if not -n_species <= numerator < n_species:
        raise ValueError(
            f"numerator={numerator} is not a channel of a field with "
            f"{n_species} of them"
        )

    top = density[..., numerator, :, :, :]
    if isinstance(density, torch.Tensor):
        total = density.sum(dim=-4)
        floored = torch.clamp(total, min=regulariser)
        replacement = torch.full_like(top, fill) if fill is not None else None
        where = torch.where
    else:
        total = density.sum(axis=-4)
        floored = np.maximum(total, regulariser)
        replacement = fill
        where = np.where
    if form == "offset":
        return top / (total + regulariser)
    if form == "floored":
        return top / floored
    return where(total > regulariser, top / floored, replacement)


def box_lengths(frames: Iterable[Frame]) -> np.ndarray:
    """Per-frame edge lengths of a timeline's boxes, ``(n_frames, 3)``."""
    lengths = [frame.box_bounds[:, 1] - frame.box_bounds[:, 0]
               for frame in frames]
    if not lengths:
        raise ValueError("frames is empty: there are no boxes to report")
    return np.stack(lengths)
