"""The hinge penalties as the published model declares them, and the three systems' trust domains.

The comparisons with the published model's own training code are migration evidence and not part
of this suite; the penalties themselves are described in docs/reference/training.md.
"""

import pytest

from aipf.system import load
from aipf.train.config import TrainConfig
from aipf.train.fit import _config_from_system
from aipf.train.penalties import penalties_from_system

import declared_roots


@pytest.fixture(scope="module")
def penalties():
    """The path table is built from the equation-of-state tables under the raw root."""
    declared_roots.raw_or_skip("hhe")
    system = load("hhe")
    return penalties_from_system(system, TrainConfig(**_config_from_system(system)))


def test_the_other_two_domains_are_declared_and_train_nothing(tmp_path):
    feb, lj = load("feb"), load("lj")
    assert feb.trust_domain.inner + feb.trust_domain.outer + \
        feb.trust_domain.T_range == (0.108, 0.0659, 0.165, 0.110, 1200.0, 2600.0)
    assert feb.constants["trust_domain"] == feb.trust_domain.inner + feb.trust_domain.outer
    assert feb.constants["trust_T_range"] == feb.trust_domain.T_range
    assert penalties_from_system(feb, TrainConfig(**_config_from_system(feb))) is None
    assert lj.trust_domain is None


@pytest.mark.env
def test_the_declared_penalties_are_the_published_models_two_at_one_seed(penalties):
    assert penalties.terms() == ("L_conv", "L_Gamma")
    assert penalties.seed == load("hhe").defaults["training"]["penalty_seed"]
    assert (penalties.conv.n, penalties.gamma.n_paths) == (1024, 16)
