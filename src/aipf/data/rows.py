"""One reader for the measured ``(x, T, n)`` tables, shared by two tiers.

Column names are the caller's: this module reads rows and does not know what the axes are called.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple, Union

#: How far a row's channel fractions may sum from one (absolute), in a composition read as a vector.
FRACTION_SUM_TOL = 1e-6


def measured_rows(path: Path, *, x_column: Optional[str] = None,
                  x_columns: Optional[Sequence[str]] = None, T_column: str,
                  n_columns: Sequence[str], status_column: str,
                  status_ok: str
                  ) -> Iterator[Tuple[Union[float, Tuple[float, ...]], float, float]]:
    """``(x, T, n)`` for each row whose status column equals ``status_ok`` (an absent column counts as ok).

    ``x_column`` names one fraction and ``x`` is a float; ``x_columns`` names one fraction per channel,
    in channel order, and ``x`` is a tuple, refused on a row whose fractions do not sum to one within
    :data:`FRACTION_SUM_TOL`. Exactly one of the two is given.
    ``n_columns`` are alternatives: the first present and non-empty one wins, a row with none raises.
    """
    if (x_column is None) == (x_columns is None):
        raise TypeError(
            f"measured_rows takes exactly one of x_column (one fraction) or "
            f"x_columns (one per channel); got x_column={x_column!r}, "
            f"x_columns={x_columns!r}")
    if x_columns is not None:
        x_columns = tuple(x_columns)
    with open(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
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
            if x_columns is None:
                yield float(row[x_column]), float(row[T_column]), value
                continue
            x = tuple(float(row[name]) for name in x_columns)
            if not abs(sum(x) - 1.0) <= FRACTION_SUM_TOL:
                raise ValueError(
                    f"{path}, line {reader.line_num}: the fractions "
                    f"{dict(zip(x_columns, x))} sum to {sum(x)!r}, not to one "
                    f"within {FRACTION_SUM_TOL:g}; a composition that is not "
                    f"a point of the simplex has no densities to map to")
            yield x, float(row[T_column]), value
