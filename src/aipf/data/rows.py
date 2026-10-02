"""One reader for the measured ``(x, T, n)`` tables, shared by two tiers.

Column names are the caller's: this module reads rows and does not know what the axes are called.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Sequence, Tuple


def measured_rows(path: Path, *, x_column: str, T_column: str,
                  n_columns: Sequence[str], status_column: str,
                  status_ok: str) -> Iterator[Tuple[float, float, float]]:
    """``(x, T, n)`` for each row whose status column equals ``status_ok`` (an absent column counts as ok).

    ``n_columns`` are alternatives: the first present and non-empty one wins, a row with none raises.
    """
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row.get(status_column, status_ok) != status_ok:
                continue
            value = None
            for name in n_columns:
                if row.get(name) not in (None, ""):
                    value = float(row[name])
                    break
            if value is None:
                raise KeyError(
                    f"{path}: no density column among {tuple(n_columns)}")
            yield float(row[x_column]), float(row[T_column]), value
