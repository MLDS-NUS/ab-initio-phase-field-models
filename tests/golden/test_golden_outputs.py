"""Every case in ``cases.py`` computes, bit for bit, what the published source computed.

The rest of the suite checks properties; these check the numbers themselves, so a change meant to
leave the arithmetic alone (a refactor of the spectral operators, the solvers or the dataset) can
show that it did. Each case is recomputed on the CPU in float32 with one thread and compared key by
key with the output stored under ``tests/golden/<torch version>-<capability>/``, which
``make_goldens.py`` wrote from the published source. A failure names the case, the first key that
differs and by how much.

The stored outputs belong to one torch build on one instruction set; on another the tests skip,
naming the pair and the command that writes them (which must be run on the published source, see its
docstring).
"""
from __future__ import annotations

import functools

import pytest
import torch

from golden import cases

pytestmark = pytest.mark.env

FIXTURES = cases.fixture_dir()

#: Every case, as ``(group, name)``, in definition order.
CASES = [(group, name) for group, table in cases.GROUPS.items() for name in table]


@functools.lru_cache(maxsize=None)
def _stored(group: str) -> dict:
    return torch.load(FIXTURES / f"{group}.pt", weights_only=True)


def _stored_or_skip(group: str) -> dict:
    if not (FIXTURES / f"{group}.pt").is_file():
        pytest.skip(f"no golden outputs for {cases.fixture_key()} (torch {torch.__version__}, "
                    f"{torch.backends.cpu.get_cpu_capability()}): write them on the published "
                    f"source with `python tests/golden/make_goldens.py --write`")
    return _stored(group)


@pytest.fixture(scope="module", autouse=True)
def _as_written():
    """One intra-op thread and float32 by default, as the outputs were written: a threaded reduction
    may sum in another order, and a default dtype left changed by another module changes every input."""
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_num_threads(threads)
    torch.set_default_dtype(dtype)


@pytest.fixture(scope="module")
def stored_models():
    """Every case's models carry the stored parameters, not whatever their builders draw today."""
    cases.use_model_states(cases.model_states(_stored_or_skip("models")))
    yield
    cases.use_model_states({})


@pytest.mark.parametrize("group,name", CASES, ids=[f"{g}/{n}" for g, n in CASES])
def test_each_case_computes_its_stored_output_bit_for_bit(stored_models, group, name):
    want = _stored_or_skip(group)["cases"].get(name)
    assert want is not None, (f"{group}/{name} has no stored output: a new case is written with "
                              f"make_goldens.py --write on the published source")
    got = cases.GROUPS[group][name]()
    problem = cases.compare(want, got)
    assert problem is None, f"{group}/{name}: {problem}"


@pytest.mark.parametrize("group", list(cases.GROUPS))
def test_every_stored_case_is_still_defined(group):
    """A case deleted from ``cases.py`` would otherwise stop being compared without a word."""
    stored = set(_stored_or_skip(group)["cases"])
    assert stored <= set(cases.GROUPS[group]), sorted(stored - set(cases.GROUPS[group]))


@pytest.mark.parametrize("group", list(cases.GROUPS))
def test_the_stored_outputs_name_the_torch_they_were_written_with(group):
    meta = _stored_or_skip(group)["meta"]
    assert meta["torch"] == torch.__version__
    assert meta["cpu_capability"] == torch.backends.cpu.get_cpu_capability()
    assert meta["group"] == group and len(meta["commit"]) == 40 and len(meta["src_tree"]) == 40
