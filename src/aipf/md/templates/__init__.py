"""Prebuilt decks (``builtin``) and the filling machinery (``deck``).

    >>> from aipf.md.templates import fill
    >>> from aipf.md.templates.builtin import builtin   # not re-exported: it would shadow the submodule
    >>> text = fill(builtin("cube", "NPT"), point, system, caller_values)   # doctest: +SKIP
"""
from aipf.md.templates.builtin import DECKS, available, for_point
from aipf.md.templates.deck import (
    Deck,
    DeckError,
    check,
    fill,
    from_file,
    from_text,
    placeholders,
    render,
    values_from,
)

__all__ = [
    "DECKS",
    "Deck",
    "DeckError",
    "available",
    "check",
    "fill",
    "for_point",
    "from_file",
    "from_text",
    "placeholders",
    "render",
    "values_from",
]
