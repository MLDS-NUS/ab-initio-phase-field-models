"""The memory-mapped sidecar of mode archives already on disk.

:func:`convert_runs` writes, beside every archive a source directory and a glob select, the sidecar
``keys.mmap_file`` that :func:`aipf.train.dataset.read_mode_runs` maps in place of the archive's
amplitudes, and its record (:func:`aipf.pipeline.modes.write_mmap_sidecar`: written under a temporary
name, checked bit for bit against the archive, then renamed into place). Each archive is read once, and
only when a sidecar is written or ``verify`` asks for its frames; the archive itself is never changed.
Runs over disjoint globs can run in parallel, and a run interrupted or repeated picks up where it
stopped. A run that cannot be converted is reported and the others go on."""
from __future__ import annotations

import os
import socket
import zipfile
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Tuple

import numpy as np

from .dataset import ArchiveKeys, mapped_amplitudes

#: What :func:`convert_runs` did to one run: wrote a sidecar where there was none, found a sidecar that
#: matches its archive and left it, rewrote one that did not, or could not convert the run.
WRITTEN, SKIPPED, REPLACED, FAILED = "written", "skipped", "replaced", "failed"

#: What a run that cannot be converted raises: an archive that is not a zip or is cut short, one with no
#: amplitudes, one the reader cannot parse, or a file system error while writing.
_RUN_FAILURES = (zipfile.BadZipFile, KeyError, ValueError, OSError, EOFError)


class Conversion(NamedTuple):
    """One run as :func:`convert_runs` left it: its directory, the status, why (a replaced or failed
    run; else ``None``), the temporary files of killed writers left in it, and those removed."""

    directory: Path
    status: str
    reason: Optional[str]
    partials: Tuple[Path, ...]
    removed: Tuple[Path, ...]


def convert_run(directory, keys: ArchiveKeys, *,
                verify: bool = False) -> Tuple[str, Optional[str]]:
    """``(status, reason)`` for one run directory holding ``keys.file_name``: :data:`WRITTEN`,
    :data:`SKIPPED` or :data:`REPLACED`, with the reason a sidecar was replaced, else ``None``.

    An existing sidecar is kept when the reader accepts it (:func:`aipf.train.dataset.mapped_amplitudes`:
    shape, dtype and order against the archive's header, and its record against the archive array's
    CRC-32 and size), without reading the archive's amplitudes; ``verify`` also compares its frames with
    the archive's, bit for bit. One the reader refuses, or whose frames differ, is rewritten from the
    archive, which is the record of the run. Anything else raises."""
    from aipf.pipeline.modes import archive_fingerprint, mmap_sidecar_matches, write_mmap_sidecar

    if keys.mmap_file is None:
        raise ValueError(
            "keys.mmap_file is None: these archive keys name no sidecar to write")
    directory = Path(directory)
    sidecar = directory / keys.mmap_file
    reason = None
    with np.load(directory / keys.file_name) as payload:
        stored = None
        if sidecar.is_file():
            try:
                mapped = mapped_amplitudes(directory, keys, payload)
            except ValueError as stale:
                reason = str(stale)
            else:
                if not verify:
                    return SKIPPED, None
                stored = payload[keys.amplitudes]
                same = mmap_sidecar_matches(mapped, stored,
                                            channel_axis=keys.amplitudes_channel_axis)
                del mapped
                if same:
                    return SKIPPED, None
                reason = (f"run {directory.name!r}: {sidecar} has the archive's shape and dtype "
                          f"and other frames")
        if stored is None:
            stored = payload[keys.amplitudes]
        fingerprint = archive_fingerprint(payload.zip, keys.amplitudes)
    write_mmap_sidecar(sidecar, stored, fingerprint=fingerprint,
                       channel_axis=keys.amplitudes_channel_axis)
    return (REPLACED, reason) if reason is not None else (WRITTEN, None)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:      # another user's process
        return True
    return True


def leftover_partials(directory, keys: ArchiveKeys, *,
                      remove: bool = False) -> Tuple[Tuple[Path, ...], Tuple[Path, ...]]:
    """``(left, removed)``: the temporary files of the sidecar and its record in ``directory``
    (:data:`aipf.pipeline.modes.PARTIAL_SUFFIX`), as a killed writer leaves them. ``remove`` deletes those
    whose name records the machine running this call and a process that no longer exists; any other is
    left, since its writer may still be running (or runs elsewhere)."""
    from aipf.pipeline.modes import PARTIAL_SUFFIX, mmap_record_path, partial_writer

    directory = Path(directory)
    names = (keys.mmap_file, mmap_record_path(keys.mmap_file).name)
    left, removed = [], []
    for name in names:
        for path in sorted(directory.glob(f".{name}.*{PARTIAL_SUFFIX}")):
            writer = partial_writer(path, name)
            if remove and writer is not None and writer[0] == socket.gethostname() \
                    and writer[1] != os.getpid() and not _alive(writer[1]):
                path.unlink(missing_ok=True)
                removed.append(path)
            else:
                left.append(path)
    return tuple(left), tuple(removed)


def convert_runs(root, pattern: str, *, keys: ArchiveKeys, verify: bool = False,
                 remove_partials: bool = False) -> Iterator[Conversion]:
    """A :class:`Conversion` for every directory under ``root`` matching ``pattern`` that holds
    ``keys.file_name``, in sorted order, each converted by :func:`convert_run` as it is yielded. A run
    that raises one of the failures of a damaged archive or a failed write is :data:`FAILED`, with the
    error as its reason, and the next run is converted. Leftover temporary files are reported per run
    and, with ``remove_partials``, removed as :func:`leftover_partials` does. A glob that matches no
    archive raises ``FileNotFoundError`` before anything is written."""
    runs = [d for d in sorted(Path(root).glob(pattern)) if (d / keys.file_name).is_file()]
    if not runs:
        raise FileNotFoundError(
            f"no run matching {pattern!r} under {root} holds a {keys.file_name!r}")
    for directory in runs:
        left, removed = leftover_partials(directory, keys, remove=remove_partials)
        try:
            status, reason = convert_run(directory, keys, verify=verify)
        except _RUN_FAILURES as failure:
            status, reason = FAILED, f"{type(failure).__name__}: {failure}"
        yield Conversion(directory, status, reason, left, removed)
