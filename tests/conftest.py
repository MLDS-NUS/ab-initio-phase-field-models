"""Puts ``tests/`` on the import path, so every test can ``import declared_roots``.

Also fails any test that leaves torch's process-wide deterministic-algorithms flag on: every later
CUDA matmul in the same process then refuses. See docs/reference/training.md, "Devices and determinism"."""
import sys
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _deterministic_flag() -> bool:
    torch = sys.modules.get("torch")  # never imported: the flag is at its default, off
    return bool(torch is not None and torch.are_deterministic_algorithms_enabled())


@pytest.fixture(autouse=True)
def _no_leaked_deterministic_flag(request):
    # module- and session-scoped fixtures are set up before this one, so "on at setup" is their leak
    on_at_setup = _deterministic_flag()
    yield
    if _deterministic_flag():
        sys.modules["torch"].use_deterministic_algorithms(False)  # charge this test, not every later one
        source = "a fixture it uses" if on_at_setup else "the test"
        pytest.fail(f"{request.node.nodeid}: {source} left torch.use_deterministic_algorithms(True) on "
                    "for the rest of the process", pytrace=False)
