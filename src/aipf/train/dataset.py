"""An archived mode timeline to the drift batches a training step consumes.

Reads what :mod:`aipf.pipeline.extract_modes` wrote, cuts time windows, forms each window's measured
drift target, and scatters it on the rfft half spectrum in density convention ``rho_hat = rho_k / V``.
A sample's keys (``rho_hat_states``, ``lam``, ``target_hat``, ``boxes``, ``T``, ``sample_weight``,
``run_index``) collate directly into the ``"drift"`` group of ``LitModule.compute_losses``."""
from __future__ import annotations

import dataclasses
import json
import logging
import re
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import savgol_coeffs
from torch.utils.data import Dataset

#: How a window's measured time derivative is formed: ``"weak"`` (difference of adjacent window
#: means, see :func:`weak_target`), ``"weak_mid"`` (the same target, the model states at the
#: frame midpoints ``0.5 (rho[c+m] + rho[c+m-1])``) or ``"savgol"`` (Savitzky-Golay taps).
ESTIMATORS = ("weak", "weak_mid", "savgol")

#: The estimators that read the weak-form window ``[c-w+1, c+w]`` and its triangular weights.
_WEAK_FORMS = ("weak", "weak_mid")


@dataclass(frozen=True)
class ArchiveKeys:
    """The key names one archive uses, and which axis holds the channel; every field required but
    ``reference_box`` and ``mmap_file``.

    ``amplitudes_channel_axis``: ``1`` for ``(frames, channels, modes)``, ``2`` for
    ``(frames, modes, channels)``.
    ``quality_file`` and ``quality_key`` are declared together or both ``None``.
    ``reference_box``: the key of an optional reference cell ``(3,)``, read when a run carries it
    (:attr:`ModeRun.reference_box`); ``None`` reads none.
    ``mmap_file``: the file name of an optional memory-mapped copy of the amplitudes beside the archive
    (:data:`aipf.pipeline.modes.MMAP_FILE`), opened in place of the archive's amplitudes when a run has
    one (:func:`read_mode_runs`); ``None`` reads every run's amplitudes into memory."""

    file_name: str
    amplitudes: str
    amplitudes_channel_axis: int
    labels: str
    box: str
    temperature: str
    frame_interval: str
    composition: Tuple[str, ...]
    composition_fallback: Optional[str]
    quality_file: Optional[str]
    quality_key: Optional[str]
    # out of repr and equality, so a declaration without it prints and compares as it always did
    reference_box: Optional[str] = field(default=None, repr=False, compare=False)
    mmap_file: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "composition", tuple(self.composition))
        if self.amplitudes_channel_axis not in (1, 2):
            raise ValueError(
                f"amplitudes_channel_axis={self.amplitudes_channel_axis!r} "
                f"is neither 1 (frames, channels, modes) nor 2 (frames, "
                f"modes, channels): those are the two layouts on disk")
        if (self.quality_file is None) != (self.quality_key is None):
            missing = ("quality_key" if self.quality_key is None
                       else "quality_file")
            raise ValueError(
                f"quality_file={self.quality_file!r} and "
                f"quality_key={self.quality_key!r}: declare both or neither, "
                f"{missing} is missing. Half a declaration would read no "
                f"mask and train on frames the archive marked unusable")

    def replace(self, **changes) -> "ArchiveKeys":
        """A copy with some fields changed."""
        return dataclasses.replace(self, **changes)


class _MappedFrames(NamedTuple):
    """How a pickled :class:`ModeRun` re-opens its memory-mapped amplitudes: the file, the byte offset
    of its first frame, the dtype and the shape."""

    filename: str
    offset: int
    dtype: np.dtype
    shape: Tuple[int, ...]

    def open(self) -> np.memmap:
        return np.memmap(self.filename, dtype=self.dtype, mode="r", offset=self.offset,
                         shape=self.shape)


def _mapping_of(amplitudes) -> Optional[_MappedFrames]:
    """Where a C-contiguous memory-mapped array (or a contiguous view of one) lies in its file, or
    ``None`` for anything else."""
    if (not isinstance(amplitudes, np.memmap) or amplitudes.filename is None
            or not amplitudes.flags.c_contiguous or amplitudes.size == 0):
        return None
    root = amplitudes
    while isinstance(root.base, np.memmap):
        root = root.base
    # the file offset of the mapping np.load opened, plus how far into it this view starts
    start = (amplitudes.__array_interface__["data"][0]
             - root.__array_interface__["data"][0])
    return _MappedFrames(str(amplitudes.filename), int(root.offset) + int(start),
                         amplitudes.dtype, tuple(amplitudes.shape))


@dataclass(frozen=True)
class ModeRun:
    """One run's whole timeline; ``amplitudes`` is ``(n_frames, n_channels, n_modes)`` complex, in memory
    or a read-only :class:`numpy.memmap` of the run's sidecar (:attr:`ArchiveKeys.mmap_file`).

    ``boxes`` is ``(n_frames, 3)`` or ``(3,)``; ``composition`` follows the declared key order.
    ``reference_box`` ``(3,)``, when the archive records one, is the cell every window's ``k`` and ``V``
    are read in (:attr:`cell`); ``None`` reads them in the frames' own boxes.
    ``depth`` is set only on a run projected to two dimensions (:func:`aipf.train.projection.project_kz0`):
    its labels are ``(n_modes, 2)``, its boxes and cell ``(Lx, Ly)``, and a window's volume is the cell's
    area times ``depth``. ``None`` (every run as read) is a three-dimensional run."""

    tag: str
    amplitudes: np.ndarray
    labels: np.ndarray
    boxes: np.ndarray
    temperature: float
    frame_interval: float
    composition: Tuple[float, ...]
    reference_box: Optional[np.ndarray] = None
    depth: Optional[float] = None

    def __getstate__(self):
        # a memory-mapped run pickles as where its frames lie in the file, not as the frames, so a
        # loader worker started by spawn maps the file as a forked one inherits the mapping
        state = dict(self.__dict__)
        mapping = _mapping_of(self.amplitudes)
        if mapping is not None:
            state["amplitudes"] = mapping
        return state

    def __setstate__(self, state):
        if isinstance(state.get("amplitudes"), _MappedFrames):
            state = {**state, "amplitudes": state["amplitudes"].open()}
        self.__dict__.update(state)

    @property
    def cell(self) -> np.ndarray:
        """The boxes ``k`` is read in: the reference cell ``(3,)`` when recorded, else ``boxes``."""
        return self.boxes if self.reference_box is None else self.reference_box

    @property
    def n_frames(self) -> int:
        return int(self.amplitudes.shape[0])

    @property
    def n_species(self) -> int:
        return int(self.amplitudes.shape[1])

    def box_mean(self, start: int, stop: int) -> np.ndarray:
        """The box a window ``[start, stop)`` is read in; a fixed-box run returns its box unaveraged, a
        run with a reference cell returns the cell."""
        if stop <= start:
            raise ValueError(
                f"window [{start}, {stop}) is empty: there is no box to "
                f"average over")
        if self.reference_box is not None:
            return self.reference_box
        if self.boxes.ndim == 1:
            return self.boxes
        return self.boxes[start:stop].mean(axis=0)


def _valid_frame_limit(directory: Path, keys: ArchiveKeys) -> Optional[int]:
    """Last usable frame from the quality sidecar (a prefix range ``"0..HI"``), or ``None``;
    anything else raises."""
    if keys.quality_file is None:
        return None
    path = directory / keys.quality_file
    if not path.is_file():
        return None
    entry = json.loads(path.read_text()).get(keys.quality_key)
    match = (re.match(r"(\d+)\.\.(\d+)", str(entry))
             if entry is not None else None)
    if match is None:
        raise ValueError(
            f"{path}: cannot read {keys.quality_key}={entry!r}, expected a "
            f"range spelled 'LO..HI'")
    lo, hi = int(match.group(1)), int(match.group(2))
    if lo != 0 or hi < lo:
        raise ValueError(
            f"{path}: {keys.quality_key} is {lo}..{hi}, which is not a "
            f"prefix of the timeline. Only '0..HI' is supported: a mask that "
            f"starts partway through would renumber every frame after it")
    return hi


def _stored_layout(payload, key: str, path: Path) -> Tuple[Tuple[int, ...], np.dtype]:
    """The shape and dtype of the array ``key`` of an open ``.npz``, read off its ``.npy`` header alone:
    nothing of the array itself is decompressed."""
    try:
        member = payload.zip.open(f"{key}.npy")
    except KeyError:
        raise KeyError(
            f"{path}: holds no {key!r}, the amplitudes a memory-mapped copy is checked "
            f"against") from None
    with member:
        version = np.lib.format.read_magic(member)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError(
                f"{path}: {key} is stored in .npy format {version}, whose header this reader "
                f"does not parse")
        shape, _, dtype = readers[version](member)
    return tuple(shape), dtype


def _record_problems(path: Path, payload, keys: ArchiveKeys) -> list:
    """Why the sidecar ``path``'s record does not vouch for the archive array as it is now; empty when it
    does."""
    from aipf.pipeline.modes import archive_fingerprint, mmap_record_path

    record = mmap_record_path(path)
    try:
        recorded = json.loads(record.read_text())
    except FileNotFoundError:
        return [f"there is no {record.name}, the record of the archive array it was copied from "
                f"(a writer that was interrupted, or not this package's)"]
    except (OSError, ValueError) as unreadable:
        return [f"{record.name} is not a readable record ({unreadable})"]
    current = archive_fingerprint(payload.zip, keys.amplitudes)
    if not isinstance(recorded, dict) or \
            {name: recorded.get(name) for name in current} != current:
        shown = ({name: recorded.get(name) for name in current}
                 if isinstance(recorded, dict) else recorded)
        return [f"{record.name} records it was copied from {shown}, and the archive's "
                f"{keys.amplitudes} is now {current}"]
    return []


#: Open files kept free beyond one per mapped run, for the loader workers' pipes and everything else.
_SPARE_OPEN_FILES = 256

#: The sidecars :func:`mapped_amplitudes` has mapped that are still held (a view keeps its mapping
#: alive), each with one open file descriptor.
_MAPPINGS: "weakref.WeakValueDictionary[int, np.memmap]" = weakref.WeakValueDictionary()


def _open_files() -> int:
    """The open file descriptors this module's mappings hold."""
    return len(_MAPPINGS)


def _reserve_open_files(n_mapped: int, where: str) -> None:
    """Make room for ``n_mapped`` more open files: a memory-mapped run keeps one open file descriptor for
    as long as it is held. The room needed is those already held, ``n_mapped`` and
    :data:`_SPARE_OPEN_FILES`; the soft ``RLIMIT_NOFILE`` is raised toward the hard one when it is lower,
    and a hard limit that is lower is refused before anything is mapped."""
    try:
        import resource
    except ImportError:          # no rlimits on this platform: nothing to check
        return
    need = _open_files() + int(n_mapped) + _SPARE_OPEN_FILES
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or need <= soft:
        return
    if hard != resource.RLIM_INFINITY and need > hard:
        raise OSError(
            f"{where}: mapping {n_mapped} run(s) keeps one open file each, {need} with the runs mapped "
            f"already and {_SPARE_OPEN_FILES} spare, and the hard limit on open files is {hard}: "
            f"raise it (ulimit -Hn), map fewer runs, or delete their sidecars to read them into memory")
    resource.setrlimit(resource.RLIMIT_NOFILE, (need, hard))
    logging.info("%s: raised the soft limit on open files from %d to %d for %d mapped run(s)",
                 where, soft, need, n_mapped)


def mapped_amplitudes(directory, keys: ArchiveKeys, payload) -> Optional[np.memmap]:
    """A run's amplitudes ``(n_frames, n_channels, n_modes)`` mapped read-only from its sidecar
    ``<directory>/<keys.mmap_file>``, or ``None`` when ``keys`` declares none or the run has none.

    ``payload`` is the run's open archive. The sidecar is refused, naming the run, unless its shape is
    the archive's amplitudes' in the channel-first layout (frames, channels and modes, read off the
    stored header, and the modes and frames again off the labels and a per-frame box), its dtype is
    theirs, it is in C order, and its record (:func:`aipf.pipeline.modes.mmap_record_path`) names the
    archive array as it is now (the zip member's CRC-32 and size,
    :func:`aipf.pipeline.modes.archive_fingerprint`): a stale sidecar raises instead of training. The
    archive's amplitudes are never decompressed."""
    if keys.mmap_file is None:
        return None
    directory = Path(directory)
    path = directory / keys.mmap_file
    if not path.is_file():
        return None
    archive = directory / keys.file_name
    shape, dtype = _stored_layout(payload, keys.amplitudes, archive)
    if len(shape) != 3:
        raise ValueError(
            f"{archive}: {keys.amplitudes} has shape {shape}, which is not a timeline of "
            f"(channels, modes)")
    expected = shape if keys.amplitudes_channel_axis == 1 else (shape[0], shape[2], shape[1])
    try:
        mapped = np.load(path, mmap_mode="r")
        if not isinstance(mapped, np.memmap):
            raise ValueError("it holds no single array")
    except (OSError, ValueError, EOFError) as unreadable:
        raise ValueError(
            f"run {directory.name!r}: {path} is not a readable .npy ({unreadable}): rewrite it "
            f"(aipf modes-sidecar) or delete it to read the archive itself") from unreadable
    labels = payload[keys.labels].shape
    boxes = payload[keys.box].shape
    problems = []
    if mapped.shape != expected:
        problems.append(f"shape {tuple(mapped.shape)}, the archive's amplitudes {expected} "
                        f"(frames, channels, modes)")
    if mapped.dtype != dtype:
        problems.append(f"dtype {mapped.dtype}, the archive's {dtype}")
    if not mapped.flags.c_contiguous:
        problems.append("Fortran order, where a window's frames are one contiguous range only in "
                        "C order")
    if len(labels) < 1 or labels[0] != expected[2]:
        problems.append(f"{expected[2]} modes, {keys.labels} {labels}")
    if len(boxes) == 2 and boxes[0] != expected[0]:
        problems.append(f"{expected[0]} frames, {keys.box} {boxes}")
    problems += _record_problems(path, payload, keys)
    if problems:
        raise ValueError(
            f"run {directory.name!r}: {path} does not match {archive}: "
            + "; ".join(problems)
            + ". It is stale or another archive's: rewrite it (aipf modes-sidecar) or delete it "
              "to read the archive itself")
    _MAPPINGS[id(mapped)] = mapped
    return mapped


def _composition(payload: Mapping[str, np.ndarray], keys: ArchiveKeys,
                 tag: str) -> Tuple[float, ...]:
    """A run's composition label, from the declared keys or the fallback."""
    if not keys.composition:
        return ()
    if all(k in payload for k in keys.composition):
        return tuple(float(payload[k]) for k in keys.composition)
    if keys.composition_fallback is not None and \
            keys.composition_fallback in payload:
        value = float(payload[keys.composition_fallback])
        return (value,) * len(keys.composition)
    raise KeyError(
        f"{tag}: none of the declared composition keys "
        f"{list(keys.composition)} are present and the fallback "
        f"{keys.composition_fallback!r} is not either")


def read_mode_runs(root, pattern: str, *, keys: ArchiveKeys,
                   exclude_tags: Sequence[str] = ()) -> list[ModeRun]:
    """Every run under ``root`` whose directory matches ``pattern``, loaded, in sorted tag order.

    ``exclude_tags`` are skipped before the file is opened; a matching directory with no amplitude
    file is skipped. A run with the sidecar ``keys.mmap_file`` maps its amplitudes from it
    (:func:`mapped_amplitudes`) and never reads them from the archive; every other run reads them into
    memory, so a tree may mix the two."""
    excluded = set(exclude_tags)
    runs: list[ModeRun] = []
    if keys.mmap_file is not None:
        _reserve_open_files(sum((d / keys.mmap_file).is_file() and (d / keys.file_name).is_file()
                                for d in Path(root).glob(pattern) if d.name not in excluded),
                            f"{root}/{pattern}")
    for directory in sorted(Path(root).glob(pattern)):
        path = directory / keys.file_name
        if directory.name in excluded or not path.is_file():
            continue
        limit = _valid_frame_limit(directory, keys)
        with np.load(path) as payload:
            amplitudes = mapped_amplitudes(directory, keys, payload)
            if amplitudes is None:
                stored = payload[keys.amplitudes]
                if stored.ndim != 3:
                    raise ValueError(
                        f"{path}: {keys.amplitudes} has shape "
                        f"{tuple(stored.shape)}, which is not a timeline of "
                        f"(channels, modes)")
                if keys.amplitudes_channel_axis == 2:
                    stored = np.moveaxis(stored, 2, 1)
                amplitudes = np.ascontiguousarray(stored)
            boxes = np.array(payload[keys.box])
            reference = None
            if keys.reference_box is not None and \
                    keys.reference_box in payload:
                reference = np.array(payload[keys.reference_box],
                                     dtype=np.float64)
                if reference.shape != (3,):
                    raise ValueError(
                        f"{path}: {keys.reference_box} has shape "
                        f"{tuple(reference.shape)}, which is not one cell of "
                        f"three edge lengths")
            if limit is not None:
                total = amplitudes.shape[0]
                amplitudes = amplitudes[:limit + 1]
                if boxes.ndim == 2:
                    boxes = boxes[:limit + 1]
                logging.warning(
                    "%s: the quality sidecar keeps %d of %d frames (0..%d)",
                    directory.name, amplitudes.shape[0], total, limit)
            runs.append(ModeRun(
                tag=directory.name,
                amplitudes=amplitudes,
                labels=np.array(payload[keys.labels]),
                boxes=boxes,
                temperature=float(payload[keys.temperature]),
                frame_interval=float(payload[keys.frame_interval]),
                composition=_composition(payload, keys, directory.name),
                reference_box=reference))
    if not runs:
        raise FileNotFoundError(
            f"no run matching {pattern!r} under {root} holds a "
            f"{keys.file_name!r}")
    if keys.mmap_file is not None:
        eager = [run.tag for run in runs if not isinstance(run.amplitudes, np.memmap)]
        if eager:
            logging.warning(
                "%s/%s: %d of %d run(s) have no %s and were read into memory: %s%s",
                root, pattern, len(eager), len(runs), keys.mmap_file, ", ".join(eager[:5]),
                f" and {len(eager) - 5} more" if len(eager) > 5 else "")
        else:
            logging.info("%s/%s: all %d run(s) memory-mapped from %s", root, pattern, len(runs),
                         keys.mmap_file)
    return runs


def band_keep(labels, boxes, k_max: Optional[float]) -> Optional[np.ndarray]:
    """The modes with ``|k| <= k_max`` in at least one box a run visits, as a boolean mask over
    ``labels`` ``(n_modes, 3)``; ``None`` (keep every mode) when ``k_max`` is ``None``.

    ``boxes`` is ``(n_frames, 3)`` or ``(3,)``. Each mode's ``|k|`` is taken with the longest edge
    the run reaches on each axis, which is the smallest ``|k|`` that mode ever has, so a mode inside
    the band at any frame is kept for the whole run. A stored mode set cut at a larger extraction
    radius can reach past the training grid on a long box; this band is what is scattered then."""
    if k_max is None:
        return None
    edges = np.asarray(boxes, dtype=np.float64)
    edges = edges.max(axis=0) if edges.ndim > 1 else edges
    k = np.linalg.norm(2.0 * np.pi * np.asarray(labels, dtype=np.float64) / edges, axis=1)
    return k <= float(k_max)


def scatter_modes(amplitudes, labels, volume: float, grid) -> np.ndarray:
    """``(..., n_channels, n_modes)`` amplitudes -> ``(..., n_channels, Gx, Gy, Gz//2+1)`` complex64
    half spectrum.

    Labels with third index ``>= 0`` only, first two axes wrapped ``n mod G``, divided by ``volume``
    in float64. A two-axis ``grid`` ``(Gx, Gy)`` takes ``(n_modes, 2)`` labels and scatters onto
    ``(..., n_channels, Gx, Gy//2+1)`` by the same rule, its half axis ``y`` (:func:`_scatter_modes_2d`)."""
    if len(grid) == 2:
        return _scatter_modes_2d(amplitudes, labels, volume, grid)
    Gx, Gy, Gz = (int(g) for g in grid)
    Gzr = Gz // 2 + 1
    amplitudes = np.asarray(amplitudes)
    labels = np.asarray(labels)
    keep = labels[:, 2] >= 0
    kept = labels[keep]
    if kept.shape[0] == 0:
        raise ValueError(
            "no label has a non-negative third index: the half spectrum "
            "would be empty")
    if (np.abs(kept[:, 0]).max() >= Gx // 2
            or np.abs(kept[:, 1]).max() >= Gy // 2
            or kept[:, 2].max() >= Gzr):
        raise ValueError(
            f"the mode set reaches beyond the grid {(Gx, Gy, Gz)}: raise the "
            f"grid or lower the extraction cutoff")
    ix = np.mod(kept[:, 0], Gx)
    iy = np.mod(kept[:, 1], Gy)
    iz = kept[:, 2]
    lead = amplitudes.shape[:-2]
    out = np.zeros(lead + (amplitudes.shape[-2], Gx, Gy, Gzr),
                   dtype=np.complex64)
    out[..., :, ix, iy, iz] = (amplitudes[..., keep] / volume).astype(
        np.complex64)
    return out


def _scatter_modes_2d(amplitudes, labels, volume: float, grid) -> np.ndarray:
    """:func:`scatter_modes` on a two-axis grid: labels ``(n_modes, 2)`` with second index ``>= 0`` only
    (the rfft half axis is ``y``), the first wrapped ``n mod Gx``, divided by ``volume`` in float64."""
    Gx, Gy = (int(g) for g in grid)
    Gyr = Gy // 2 + 1
    amplitudes = np.asarray(amplitudes)
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(
            f"labels of shape {tuple(labels.shape)} on the two-axis grid {(Gx, Gy)}: a "
            f"two-dimensional half spectrum takes (n_x, n_y) labels, which "
            f"aipf.train.projection.project_kz0 makes from a three-dimensional archive")
    keep = labels[:, 1] >= 0
    kept = labels[keep]
    if kept.shape[0] == 0:
        raise ValueError(
            "no label has a non-negative second index: the half spectrum "
            "would be empty")
    if (np.abs(kept[:, 0]).max() >= Gx // 2
            or kept[:, 1].max() >= Gyr):
        raise ValueError(
            f"the mode set reaches beyond the grid {(Gx, Gy)}: raise the "
            f"grid or lower the extraction cutoff")
    ix = np.mod(kept[:, 0], Gx)
    iy = kept[:, 1]
    lead = amplitudes.shape[:-2]
    out = np.zeros(lead + (amplitudes.shape[-2], Gx, Gyr),
                   dtype=np.complex64)
    out[..., :, ix, iy] = (amplitudes[..., keep] / volume).astype(
        np.complex64)
    return out


def weak_target(amplitudes, centre: int, half_width: int,
                frame_interval: float):
    """``(mean over (c, c+w] - mean over (c-w, c]) / (w * dt)``: the weak-form time derivative at
    ``centre``."""
    right = amplitudes[centre + 1: centre + half_width + 1].mean(axis=0)
    left = amplitudes[centre - half_width + 1: centre + 1].mean(axis=0)
    return (right - left) / (half_width * frame_interval)


def triangular_weights(half_width: int, n_states: int):
    """``(offsets, weights)`` of the weak-form combination: offset ``m`` in ``[2-w, w]`` weighs
    ``w-m+1`` (``m>=1``) or ``w+m-1``.

    Normalised to one; ``n_states`` evenly spaced offsets, or the whole support."""
    if n_states >= 2 * half_width - 1:
        offsets = np.arange(2 - half_width, half_width + 1)
    else:
        offsets = np.unique(np.round(np.linspace(
            2 - half_width, half_width, n_states)).astype(int))
    weights = np.where(offsets >= 1, half_width - offsets + 1,
                       half_width + offsets - 1).astype(np.float64)
    return offsets, weights / weights.sum()


def savgol_taps(window: int, poly: int, frame_interval: float):
    """Savitzky-Golay value and derivative taps ``(c0, c1)`` for a chronological window (``use="dot"``)."""
    c0 = savgol_coeffs(window, poly, deriv=0, use="dot")
    c1 = savgol_coeffs(window, poly, deriv=1, delta=frame_interval,
                       use="dot")
    return c0, c1


@dataclass(frozen=True)
class WindowSettings:
    """How a run's timeline is cut into windows, and onto what ``grid`` ``(Gx, Gy, Gz)`` (or ``(Gx, Gy)``
    for runs projected to two dimensions); every field is required except ``band_k_max``, which is
    optional and defaults to ``None``.

    * ``estimator``: one of :data:`ESTIMATORS`.
    * ``half_width``, ``n_states``, ``stride``: the window's half width, its states and the frames
      between two centres, each at least 1.
    * ``grid``: the real-space grid the modes are scattered onto.
    * ``savgol_window`` (odd) and ``savgol_poly``: only for ``"savgol"``, ``None`` otherwise.
    * ``band_k_max`` (optional): the radius of the mode band scattered onto the grid, a positive
      number (see :func:`band_keep`); ``None`` scatters every stored mode."""

    estimator: str
    half_width: int
    n_states: int
    stride: int
    grid: Tuple[int, int, int]
    savgol_window: Optional[int]
    savgol_poly: Optional[int]
    band_k_max: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", tuple(int(g) for g in self.grid))
        if self.band_k_max is not None and not float(self.band_k_max) > 0:
            raise ValueError(
                f"band_k_max={self.band_k_max!r} is not a positive radius; None keeps every mode")
        if self.estimator not in ESTIMATORS:
            raise ValueError(
                f"estimator={self.estimator!r} is not one of {ESTIMATORS}")
        for name in ("half_width", "n_states", "stride"):
            if int(getattr(self, name)) < 1:
                raise ValueError(
                    f"{name}={getattr(self, name)!r} must be at least 1")
        if len(self.grid) not in (2, 3) or any(g < 1 for g in self.grid):
            raise ValueError(
                f"grid={self.grid!r} is not three positive axis lengths "
                f"(or two, for runs projected to two dimensions)")
        if self.estimator == "savgol":
            for name in ("savgol_window", "savgol_poly"):
                if getattr(self, name) is None:
                    raise ValueError(
                        f"estimator='savgol' needs {name}: its tap shape is "
                        f"a declared choice with no value in this package")
            if self.savgol_window % 2 == 0:
                raise ValueError(
                    f"savgol_window={self.savgol_window} is even: a centred "
                    f"tap window has an odd number of frames")
            if self.savgol_poly >= self.savgol_window:
                raise ValueError(
                    f"savgol_poly={self.savgol_poly} needs a window longer "
                    f"than itself, got savgol_window={self.savgol_window}")
        else:
            for name in ("savgol_window", "savgol_poly"):
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"estimator={self.estimator!r} has no use for "
                        f"{name}={getattr(self, name)!r}: accepting it would "
                        f"record a tap shape that never ran")

    @property
    def half_taps(self) -> int:
        """Half the tap window of the ``"savgol"`` estimator."""
        return int(self.savgol_window) // 2

    def centres(self, n_frames: int) -> range:
        """Window centres a run of ``n_frames`` supports: weak reads ``[c-w+1, c+w]``, taps ``[c-h, c+h]``."""
        if self.estimator in _WEAK_FORMS:
            lo, hi = self.half_width - 1, n_frames - self.half_width
        else:
            lo, hi = self.half_taps, n_frames - self.half_taps - 1
        return range(lo, hi, self.stride)

    def window_bounds(self, centre: int) -> Tuple[int, int]:
        """``[start, stop)`` of the frames the estimator reads."""
        if self.estimator in _WEAK_FORMS:
            return centre - self.half_width + 1, centre + self.half_width + 1
        return centre - self.half_taps, centre + self.half_taps + 1


def _check_axes(runs: Sequence[ModeRun], grid) -> None:
    """Refuse a run whose labels have another number of axes than ``grid``: a three-dimensional run on a
    two-axis grid, or a projected one on a three-axis grid."""
    for run in runs:
        axes = int(np.asarray(run.labels).shape[-1])
        if axes != len(grid):
            raise ValueError(
                f"run {run.tag!r} carries {axes}-axis labels and the grid {tuple(grid)} has "
                f"{len(grid)} axes: a two-axis grid reads runs projected to two dimensions "
                f"(aipf.train.projection.project_kz0), a three-axis grid the archive as read")


class ModeWindowDataset(Dataset):
    """Windows of a set of runs (one channel count) as the tensors a drift step consumes.

    ``run_weights`` is ``(len(runs),)`` per-run weights, or ``None`` for unweighted."""

    def __init__(self, runs: Sequence[ModeRun], settings: WindowSettings, *,
                 run_weights: Optional[np.ndarray] = None) -> None:
        runs = list(runs)
        if not runs:
            raise ValueError("runs is empty: there are no windows to cut")
        counts = {r.n_species for r in runs}
        if len(counts) != 1:
            raise ValueError(
                f"the runs carry {sorted(counts)} channels: one dataset is "
                f"one channel count, because one batch of it is one model's "
                f"input")
        _check_axes(runs, settings.grid)
        self.runs = runs
        self.settings = settings
        if run_weights is None:
            self.run_weights = np.ones(len(runs), np.float32)
        else:
            weights = np.asarray(run_weights, np.float32)
            if weights.shape != (len(runs),):
                raise ValueError(
                    f"run_weights has shape {tuple(weights.shape)} against "
                    f"{len(runs)} runs")
            self.run_weights = weights
        if settings.estimator in _WEAK_FORMS:
            self.offsets, self.lam = triangular_weights(settings.half_width,
                                                        settings.n_states)
        else:
            self.offsets, self.lam = np.array([0]), np.array([1.0])
        self._taps: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        #: Per run, the modes scattered (:func:`band_keep`), or ``None`` for all of them.
        self.keep = [band_keep(r.labels, r.cell, settings.band_k_max) for r in runs]
        self.samples: list[Tuple[int, int]] = [
            (index, centre)
            for index, run in enumerate(runs)
            for centre in settings.centres(run.n_frames)]

    @property
    def n_species(self) -> int:
        return self.runs[0].n_species

    def taps(self, frame_interval: float):
        """The tap pair for one frame interval, cached per interval (taps carry ``1 / dt``)."""
        key = float(frame_interval)
        if key not in self._taps:
            self._taps[key] = savgol_taps(self.settings.savgol_window,
                                          self.settings.savgol_poly, key)
        return self._taps[key]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        run_index, centre = self.samples[index]
        run = self.runs[run_index]
        settings = self.settings
        start, stop = settings.window_bounds(centre)
        box = run.box_mean(start, stop)
        volume = float(box.prod())
        if run.depth is not None:
            # a projected run: the cell's area times the depth along the axis it was averaged over
            volume = volume * float(run.depth)
        if settings.estimator == "weak":
            target = weak_target(run.amplitudes, centre, settings.half_width,
                                 run.frame_interval)
            states = run.amplitudes[centre + self.offsets]
        elif settings.estimator == "weak_mid":
            # the earliest offset is 2 - w, so the frame before it is c + 1 - w: inside the window
            target = weak_target(run.amplitudes, centre, settings.half_width,
                                 run.frame_interval)
            states = 0.5 * (run.amplitudes[centre + self.offsets]
                            + run.amplitudes[centre + self.offsets - 1])
        else:
            c0, c1 = self.taps(run.frame_interval)
            window = run.amplitudes[start:stop]
            target = np.tensordot(c1, window, axes=(0, 0))
            states = np.tensordot(c0, window, axes=(0, 0))[None]
        labels = run.labels
        keep = self.keep[run_index]
        if keep is not None:
            target, states, labels = target[..., keep], states[..., keep], labels[keep]
        return {
            "target_hat": torch.from_numpy(
                scatter_modes(target, labels, volume, settings.grid)),
            "rho_hat_states": torch.from_numpy(
                scatter_modes(states, labels, volume, settings.grid)),
            "lam": torch.tensor(self.lam, dtype=torch.float32),
            "boxes": torch.tensor(box, dtype=torch.float32),
            "T": torch.tensor(run.temperature, dtype=torch.float32),
            "sample_weight": torch.tensor(self.run_weights[run_index]),
            "run_index": torch.tensor(run_index, dtype=torch.long),
        }
