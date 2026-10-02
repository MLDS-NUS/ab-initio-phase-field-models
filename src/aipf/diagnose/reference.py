"""Reference datasets, each declared for one system and checked by :func:`for_system` before use.

The data itself lives in ``experiments/<system>/``; this module holds none."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReferenceDataset:
    """One reference dataset (``kind`` e.g. ``"binodal"``, opaque ``data``) declared for ``system_name``."""

    system_name: str
    kind: str
    data: Any


def for_system(reference: ReferenceDataset, system_name: str) -> ReferenceDataset:
    """Return ``reference`` if it was declared for ``system_name``; raise ``ValueError`` otherwise."""
    if reference.system_name != system_name:
        raise ValueError(
            f"reference dataset {reference.kind!r} was declared for system "
            f"{reference.system_name!r}, not {system_name!r}: using it here "
            f"would silently score the wrong system. Pass a "
            f"ReferenceDataset whose system_name matches, or none at all."
        )
    return reference
