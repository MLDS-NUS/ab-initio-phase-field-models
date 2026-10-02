"""Anchor row gating: a System-declared table, with no core default."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _table_length(table: Mapping[str, object]) -> int:
    for value in table.values():
        return len(value)  # type: ignore[arg-type]
    raise ValueError("gate_rows: table has no columns to measure a row count from")


def gate_rows(table: Mapping[str, object], *,
              declared_gate: str | Sequence[str] | None,
              known_gate_columns: Sequence[str] = ()) -> np.ndarray:
    """Boolean mask of anchor-eligible rows; ``declared_gate`` is ``None``, one column, or a priority list.

    Raises ValueError when no gate is declared but a ``known_gate_columns`` entry is present,
    KeyError when no declared column is present."""
    if declared_gate is None:
        present = [name for name in known_gate_columns if name in table]
        if present:
            raise ValueError(
                f"no gate declared, but the table has gate column(s) "
                f"{present}: declare which one applies (or that none of "
                f"them should gate this table)")
        return np.ones(_table_length(table), dtype=bool)

    names = (declared_gate,) if isinstance(declared_gate, str) else tuple(declared_gate)
    for name in names:
        if name in table:
            return np.asarray(table[name], dtype=bool)
    raise KeyError(
        f"declared gate {names!r} matches no column in the table; "
        f"available columns: {sorted(table)}")
