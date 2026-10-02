"""One registry, not one per module.

The parallel fan-out that built the rungs told each agent to touch only its own
files, so each created its own module-level ``ModelRegistry``. The result was
three disjoint registries and no single place to resolve a model name -- which
is the one thing a registry exists to provide, and which ``train`` and the
rung-3 assembly both need.
"""
import pytest


def test_every_rung_registers_into_one_shared_registry():
    import aipf.functional.landau  # noqa: F401  (import registers)
    import aipf.functional.square_gradient  # noqa: F401
    import aipf.functional.neural_operator  # noqa: F401
    from aipf.functional import MODEL_REGISTRY

    names = set(MODEL_REGISTRY.names)
    expected = {"fh", "landau", "square_gradient", "neural_operator"}
    missing = expected - names
    assert not missing, (
        f"{sorted(missing)} are registered in a module-local registry that "
        f"nothing else can see. MODEL_REGISTRY has {sorted(names)}.")


def test_the_module_level_names_are_the_same_object():
    """A module keeping its own instance would satisfy the test above by
    accident if it also registered into the shared one. Pin identity."""
    import aipf.functional.landau as landau
    import aipf.functional.square_gradient as square_gradient
    import aipf.functional.neural_operator as neural_operator
    from aipf.functional import MODEL_REGISTRY

    for module in (landau, square_gradient, neural_operator):
        local = getattr(module, "MODEL_REGISTRY", None)
        if local is not None:
            assert local is MODEL_REGISTRY, (
                f"{module.__name__} keeps its own registry instance")


def test_an_unknown_name_raises():
    from aipf.functional import MODEL_REGISTRY
    with pytest.raises(KeyError):
        MODEL_REGISTRY.build("no_such_rung")
