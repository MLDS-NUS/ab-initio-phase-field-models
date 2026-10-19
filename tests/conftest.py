"""Puts ``tests/`` on the import path, so every test can ``import declared_roots``.

Also fails any test that leaves torch's process-wide deterministic-algorithms flag on: every later
CUDA matmul in the same process then refuses. See docs/reference/training.md, "Devices and determinism".
And fails any test that leaves torch's intra-op thread count changed: every later test then runs on
that count, and one that compares against a child process (which starts at the default) disagrees in
the last bits."""
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


def _thread_count():
    torch = sys.modules.get("torch")  # never imported: nothing can have changed it
    return None if torch is None else torch.get_num_threads()


@pytest.fixture(autouse=True)
def _no_leaked_thread_count(request):
    before = _thread_count()
    yield
    after = _thread_count()
    if before is not None and after != before:
        sys.modules["torch"].set_num_threads(before)  # charge this test, not every later one
        pytest.fail(f"{request.node.nodeid}: left torch.set_num_threads({after}) on for the rest of "
                    f"the process (it was {before})", pytrace=False)


@pytest.fixture(scope="module", autouse=True)
def _no_leaked_thread_count_by_a_module(request):
    # the function-scoped guard reads its "before" after module fixtures are set up, so it cannot see theirs
    before = _thread_count()
    yield
    after = _thread_count()
    if before is not None and after != before:
        sys.modules["torch"].set_num_threads(before)  # charge this module, not every later one
        pytest.fail(f"{request.node.nodeid}: a module fixture left torch.set_num_threads({after}) on "
                    f"for the rest of the process (it was {before})", pytrace=False)
