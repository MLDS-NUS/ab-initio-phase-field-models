"""The ``UNDECLARED`` marker: a knob that changes which physics a rollout solves is declared, never defaulted.
Its refusal names the admissible values. A knob whose omission cannot change a result keeps a default."""
from __future__ import annotations


class _Undeclared:
    """The marker type; one instance, compared by identity, never equality."""

    __slots__ = ()

    def __repr__(self) -> str:                      # pragma: no cover
        return "<undeclared>"


#: Sole instance of :class:`_Undeclared`; checked as ``is UNDECLARED``.
UNDECLARED = _Undeclared()
