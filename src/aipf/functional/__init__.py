"""The free-energy ladder: the model protocol, and every rung and form."""
from .base import MODEL_REGISTRY  # noqa: F401
# ``build`` is deliberately NOT re-exported. Binding the name on this
# package would replace the SUBMODULE attribute of the same name, and
# which of the two a caller gets would then depend on whether the
# submodule had been imported yet. It is reached as
# ``from aipf.functional.build import build``, the convention every
# package of this project with such a clash follows.
from .build import rung_kwargs  # noqa: F401
