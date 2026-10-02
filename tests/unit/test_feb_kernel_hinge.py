"""The kernel hinge ``L_W`` against the published iron-boron model, and the three declarations.

The published model was trained with ``lambda_wpsd = 0.3`` (``wpsd_kappa`` 2, ``wpsd_k_max`` 3,
``wpsd_n_k`` 32, margin 0). At its final weights the hinge is 0.00166, ``W_hat(0)`` has the two
eigenvalues -17.46 and -17.41, and 7 of the 32 wavenumbers are still active. The value pinned below
is this package's, in float64, on the tracked checkpoint (``System.resolve_checkpoint``); the closed
form and ``eigvalsh`` agree on it to 1e-14, so the pin is the formula's value and not one
eigensolver's. In float32, the dtype the run trained in, the same hinge differs by 1.7e-7 (1e-4
relative), so a float32 value is held to 1e-3 relative.
"""
import pytest
import torch

from aipf.diagnose.run import load_model
from aipf.losses.names import WEIGHT_MAP_A
from aipf.system import load
from aipf.train.ckpt_compat import (IMPLICIT_WEIGHTS_A, NONLOSS_KEY_MAP_LIGHTNING,
                                    hparams_to_train_config, read_hparams_checkpoint)
from aipf.train.config import TrainConfig
from aipf.train.fit import _config_from_system
from aipf.train.kernel_hinge import SHAPE_KEYS, KernelHinge

import declared_roots

#: The published model's hinge, float64, this package's kernel path.
PUBLISHED_HINGE = 0.0016601643657847664
#: Its ``W_hat(0)`` eigenvalues, ascending.
PUBLISHED_W0_EIGENVALUES = (-17.45891701760267, -17.407747479473127)


@pytest.fixture(scope="module")
def feb():
    return load("feb")


@pytest.fixture(scope="module")
def published(feb):
    declared_roots.published_or_skip("feb")
    return load_model(feb, feb.resolve_checkpoint())


@pytest.fixture(scope="module")
def hinge(feb):
    return KernelHinge.from_config(TrainConfig(**_config_from_system(feb)))


def test_the_declaration_trains_the_hinge_with_the_published_weight_and_shape(feb, hinge):
    cfg = TrainConfig(**_config_from_system(feb))
    assert cfg.lambda_W == 0.3
    assert hinge == KernelHinge(kappa=2.0, k_max=3.0, n_k=32, margin=0.0)


def test_the_declared_config_and_the_published_checkpoints_agree_on_the_hinge(feb):
    """The declaration loader and ``ckpt_compat`` put the hinge in the same fields."""
    declared_roots.published_or_skip("feb")
    declared = TrainConfig(**_config_from_system(feb))
    hparams, _ = read_hparams_checkpoint(feb.resolve_checkpoint())
    saved = hparams_to_train_config(hparams, weight_map=WEIGHT_MAP_A,
                                     key_map=NONLOSS_KEY_MAP_LIGHTNING,
                                     implicit_weights=IMPLICIT_WEIGHTS_A)
    assert saved.lambda_W == declared.lambda_W == 0.3
    for key in SHAPE_KEYS:
        assert saved.extra_experiment_config[key] == declared.extra_experiment_config[key], key


@pytest.mark.parametrize("name", ["hhe", "lj"])
def test_a_system_that_declares_no_hinge_trains_none(name):
    fields = _config_from_system(load(name))
    assert "lambda_W" not in fields and "extra_experiment_config" not in fields
    assert KernelHinge.from_config(TrainConfig(**fields)) is None


def test_the_hinge_at_the_published_weights_is_the_published_value(published, hinge):
    with torch.no_grad():
        value = float(hinge.loss(published))
    assert value == pytest.approx(PUBLISHED_HINGE, rel=1e-9)
    assert round(value, 4) == 0.0017


def test_the_closed_form_and_eigvalsh_agree_on_the_published_kernel(published, hinge):
    k = hinge.wavenumbers(dtype=torch.float64)
    with torch.no_grad():
        W = published.kernel.w_hat(torch.cat([k.new_zeros(1), k]))
        D = W[1:] - W[0]
        from aipf.losses.extras import eigmin_sym2
        assert torch.allclose(eigmin_sym2(D), torch.linalg.eigvalsh(D)[:, 0], rtol=0, atol=1e-12)
        eig0 = torch.linalg.eigvalsh(W[0]).tolist()
    assert eig0 == pytest.approx(PUBLISHED_W0_EIGENVALUES, abs=1e-9)
    assert int((2.0 * k * k - eigmin_sym2(D) > 0).sum()) == 7


def test_the_hinge_in_float32_is_the_published_value_to_its_precision(feb, hinge):
    declared_roots.published_or_skip("feb")
    model = load_model(feb, feb.resolve_checkpoint()).float()
    with torch.no_grad():
        assert float(hinge.loss(model)) == pytest.approx(PUBLISHED_HINGE, rel=1e-3)


def test_the_hinge_trains_the_kernel_and_only_the_kernel(feb, hinge):
    """At the published weights the hinge is active, so its gradient reaches the radial nets."""
    declared_roots.published_or_skip("feb")
    model = load_model(feb, feb.resolve_checkpoint()).float().train()
    hinge.loss(model).backward()
    kernel = {id(p) for p in model.kernel.parameters()}
    reached = [name for name, p in model.named_parameters()
               if p.grad is not None and bool(p.grad.abs().sum() > 0)]
    assert reached
    assert all(id(dict(model.named_parameters())[name]) in kernel for name in reached), reached
