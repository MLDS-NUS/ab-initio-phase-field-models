"""The kernel hinge says why it exists, where a user would read it before setting its weight to 0.

Without the ``kappa k^2`` floor nothing stops a finite-k instability, and the floor with
``W_hat(k_max) ~ 0`` is what pins the attractive ``W_hat(0)`` that carries the Fe-B miscibility gap."""
from pathlib import Path

import torch

import aipf.losses.extras as extras
import aipf.train.kernel_hinge as kernel_hinge
from aipf.system import load
from aipf.train.config import TrainConfig
from aipf.train.fit import _config_from_system
from aipf.train.kernel_hinge import KernelHinge

DOCS = Path(__file__).resolve().parents[2] / "docs" / "reference" / "training.md"


def _flat(text):
    return " ".join(text.split())


def test_the_module_docstrings_say_why():
    for module in (kernel_hinge, extras):
        text = _flat(module.__doc__)
        assert "k = 0" in text and "kappa k_max^2" in text and "miscibility gap" in text
    assert "loses the gap" in _flat(kernel_hinge.__doc__)


def test_the_reference_row_says_why():
    row = next(line for line in DOCS.read_text().splitlines() if line.startswith("| `L_W` |"))
    assert "instability can only start at k = 0" in row
    assert "-kappa k_max^2" in row and "-18" in row and "miscibility gap" in row


def test_the_feb_numbers_in_the_text_are_the_declared_ones():
    hinge = KernelHinge.from_config(TrainConfig(**_config_from_system(load("feb"))))
    assert hinge.kappa * hinge.k_max ** 2 == 18.0


def test_provenance_and_eigmin_are_documented():
    assert KernelHinge.provenance.__doc__
    assert "Symmetry is assumed" in _flat(extras.eigmin_sym2.__doc__)


def test_eigmin_reads_the_upper_entry_only():
    sym = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    upper_only = sym.clone()
    upper_only[1, 0] = 99.0
    assert extras.eigmin_sym2(upper_only) == extras.eigmin_sym2(sym)
    assert torch.isclose(extras.eigmin_sym2(sym), torch.linalg.eigvalsh(sym)[0])
