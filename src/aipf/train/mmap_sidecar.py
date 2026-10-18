"""The memory-mapped sidecar of mode archives already on disk.

:func:`convert_runs` writes, beside every archive a tree and a glob select, the sidecar
``keys.mmap_file`` that :func:`aipf.train.dataset.read_mode_runs` maps in place of the archive's
amplitudes (:func:`aipf.pipeline.modes.write_mmap_sidecar`: written under a temporary name, checked
bit for bit against the archive, then renamed into place). Each archive is read once, and only when a
sidecar is written or ``verify`` asks for its frames; the archive itself is never changed. Runs over
disjoint globs can run in parallel, and a run interrupted or repeated picks up where it stopped."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from .dataset import ArchiveKeys, mapped_amplitudes

#: What :func:`convert_runs` did to one run: wrote a sidecar where there was none, found a sidecar that
#: matches its archive and left it, or rewrote one that did not.
WRITTEN, SKIPPED, REPLACED = "written", "skipped", "replaced"


def convert_run(directory, keys: ArchiveKeys, *,
                verify: bool = False) -> Tuple[str, Optional[str]]:
    """``(status, reason)`` for one run directory holding ``keys.file_name``: :data:`WRITTEN`,
    :data:`SKIPPED` or :data:`REPLACED`, with the reason a sidecar was replaced, else ``None``.

    An existing sidecar is kept when the reader accepts it (:func:`aipf.train.dataset.mapped_amplitudes`:
    shape, dtype and order against the archive's header), without reading the archive's amplitudes;
    ``verify`` also compares its frames with the archive's, bit for bit. One the reader refuses, or
    whose frames differ, is rewritten from the archive, which is the record of the run."""
    from aipf.pipeline.modes import mmap_sidecar_matches, write_mmap_sidecar

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
    write_mmap_sidecar(sidecar, stored, channel_axis=keys.amplitudes_channel_axis)
    return (REPLACED, reason) if reason is not None else (WRITTEN, None)


def convert_runs(root, pattern: str, *, keys: ArchiveKeys,
                 verify: bool = False) -> Iterator[Tuple[Path, str, Optional[str]]]:
    """``(run directory, status, reason)`` for every directory under ``root`` matching ``pattern`` that
    holds ``keys.file_name``, in sorted order, each converted by :func:`convert_run` as it is yielded.
    A glob that matches no archive raises ``FileNotFoundError`` before anything is written."""
    runs = [d for d in sorted(Path(root).glob(pattern)) if (d / keys.file_name).is_file()]
    if not runs:
        raise FileNotFoundError(
            f"no run matching {pattern!r} under {root} holds a {keys.file_name!r}")
    for directory in runs:
        status, reason = convert_run(directory, keys, verify=verify)
        yield directory, status, reason
