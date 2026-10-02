"""``L_W``, the kernel hinge: its closed-form eigenvalue, its declaration, and the driver training it.

The toy system and its mode tree are ``test_train_fit``'s. The comparison against the published
iron-boron model is ``tests/unit/test_feb_kernel_hinge.py``.
"""
import dataclasses
import json
import math

import pytest
import torch

from aipf.losses.extras import eigmin_sym2, l_w
from aipf.losses.names import WEIGHT_MAP_A
from aipf.train.ckpt_compat import NONLOSS_KEY_MAP_LIGHTNING
from aipf.train.config import TrainConfig
from aipf.train.fit import _config_from_system, _terms, fit
from aipf.train.kernel_hinge import (HINGE_TERM, SHAPE_KEYS, WEIGHT_KEY, KernelHinge,
                                     declared_fields)

from test_train_fit import _demo_system_with_modes

#: The hinge a toy declaration carries, in the declared spelling.
_DECLARED = {"lambda_wpsd": 0.3, "wpsd_kappa": 2.0, "wpsd_k_max": 3.0, "wpsd_n_k": 32,
             "wpsd_margin": 0.0}


def _rotation(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


# ---------------------------------------------------------------------------
# the closed-form smallest eigenvalue
# ---------------------------------------------------------------------------

def test_eigmin_sym2_is_the_smaller_eigenvalue_of_a_rotated_diagonal():
    e1 = torch.tensor([-3.0, 0.5, 2.0, 7.0], dtype=torch.float64)
    e2 = torch.tensor([1.0, 0.25, 2.0, -1.0], dtype=torch.float64)
    D = torch.stack([_rotation(t) @ torch.diag(torch.stack([a, b])) @ _rotation(t).T
                     for t, a, b in zip((0.3, 1.1, 2.0, -0.7), e1, e2)])
    assert torch.allclose(eigmin_sym2(D), torch.minimum(e1, e2), rtol=0, atol=1e-12)
    assert torch.allclose(eigmin_sym2(D), torch.linalg.eigvalsh(D)[:, 0], rtol=0, atol=1e-12)


def test_eigmin_sym2_has_a_finite_gradient_at_an_exactly_degenerate_matrix():
    D = (2.0 * torch.eye(2, dtype=torch.float64)).requires_grad_(True)
    value = eigmin_sym2(D)
    value.backward()
    assert float(value) == pytest.approx(2.0, abs=1e-12)
    assert torch.isfinite(D.grad).all()


def test_eigmin_sym2_refuses_a_matrix_that_is_not_two_by_two():
    with pytest.raises(ValueError, match="2, 2"):
        eigmin_sym2(torch.eye(3))


# ---------------------------------------------------------------------------
# the hinge on a kernel with known eigenvalues
# ---------------------------------------------------------------------------

class _KnownKernel(torch.nn.Module):
    """``W(k) = W0 + R diag(e1(k), e2(k)) R^T`` with ``e1 = a k^2``, ``e2 = b k^2 - c``: known
    eigenvalues of ``W(k) - W(0)`` at every ``k > 0`` (``e2`` at ``k = 0`` is never read by ``D``)."""

    def __init__(self, a, b, c, theta, w0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
        self.a, self.b, self.c = a, b, c
        self.R = _rotation(theta)
        self.w0 = w0

    def eigenvalues(self, k):
        return self.a * k * k, torch.where(k > 0, self.b * k * k - self.c, torch.zeros_like(k))

    def w_hat(self, k):
        e1, e2 = self.eigenvalues(k)
        diag = torch.diag_embed(torch.stack([e1, e2], dim=-1))
        return self.scale * (self.w0 + self.R @ diag @ self.R.T)


class _KnownModel(torch.nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel


def test_the_hinge_on_a_kernel_with_known_eigenvalues_is_the_formula():
    kernel = _KnownKernel(a=5.0, b=1.0, c=0.5, theta=0.4,
                          w0=torch.tensor([[-17.0, 0.3], [0.3, -16.0]], dtype=torch.float64))
    hinge = KernelHinge(kappa=2.0, k_max=3.0, n_k=32, margin=0.1)
    k = torch.linspace(3.0 / 32, 3.0, 32, dtype=torch.float64)
    e1, e2 = kernel.eigenvalues(k)
    expected = torch.clamp(0.1 + 2.0 * k * k - torch.minimum(e1, e2), min=0.0).pow(2).mean()
    value = hinge.loss(_KnownModel(kernel))
    assert float(expected) > 1.0       # the toy kernel violates the bound: the hinge is active
    assert float(value) == pytest.approx(float(expected), rel=1e-12)


def test_the_hinge_is_zero_on_a_kernel_that_meets_the_bound_everywhere():
    kernel = _KnownKernel(a=5.0, b=3.0, c=0.0, theta=1.2,
                          w0=torch.zeros(2, 2, dtype=torch.float64))
    hinge = KernelHinge(kappa=2.0, k_max=3.0, n_k=32, margin=0.0)
    assert float(hinge.loss(_KnownModel(kernel))) == 0.0


def test_the_wavenumbers_are_the_declared_line():
    k = KernelHinge(kappa=2.0, k_max=3.0, n_k=32, margin=0.0).wavenumbers(dtype=torch.float64)
    assert k.shape == (32,)
    assert float(k[0]) == pytest.approx(3.0 / 32) and float(k[-1]) == 3.0


def test_a_model_without_a_pair_kernel_is_refused():
    with pytest.raises(NotImplementedError, match="pair kernel"):
        KernelHinge.check(torch.nn.Linear(2, 2))


# ---------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------

def test_the_declared_spelling_is_the_saved_one_and_maps_where_a_saved_checkpoint_does():
    """``declared_fields`` and ``ckpt_compat`` agree on both halves, so a declared run and its
    published checkpoint read to one ``TrainConfig`` for the hinge."""
    assert WEIGHT_KEY == WEIGHT_MAP_A.to_code(HINGE_TERM) == "lambda_wpsd"
    for key in SHAPE_KEYS:
        assert NONLOSS_KEY_MAP_LIGHTNING[key] == "extra_experiment_config", key
    fields = declared_fields(_DECLARED)
    assert fields == {"lambda_W": 0.3,
                      "extra_experiment_config": {k: _DECLARED[k] for k in SHAPE_KEYS}}


def test_a_declaration_without_the_hinge_gives_no_field():
    assert declared_fields({"lambda_M": 0.5, "lr": 1e-3}) == {}


def test_the_weight_declared_under_both_names_is_refused():
    with pytest.raises(ValueError, match="twice"):
        declared_fields({"lambda_W": 0.3, "lambda_wpsd": 0.3})


def test_a_weight_without_its_shape_is_refused_naming_the_missing_keys():
    cfg = TrainConfig(lambda_W=0.3, extra_experiment_config={"wpsd_kappa": 2.0})
    with pytest.raises(KeyError, match="wpsd_k_max"):
        KernelHinge.from_config(cfg)


def test_a_zero_weight_trains_no_hinge_whatever_shape_is_declared():
    cfg = TrainConfig(lambda_W=0.0,
                      extra_experiment_config={k: _DECLARED[k] for k in SHAPE_KEYS})
    assert KernelHinge.from_config(cfg) is None


def test_the_hinge_is_classified_by_the_term_accounting():
    """A non-zero ``lambda_W`` the driver did not feed is reported, exactly like an anchor's."""
    cfg = TrainConfig(lambda_W=0.3)
    assert _terms(cfg, ()) == (["L_dyn"], {"L_W": 0.3})
    assert _terms(cfg, (), (), (HINGE_TERM,)) == (["L_dyn", "L_W"], {})
    assert _terms(TrainConfig(), (), (), (HINGE_TERM,)) == (["L_dyn"], {})


# ---------------------------------------------------------------------------
# the driver trains it
# ---------------------------------------------------------------------------

def _hinged_demo(tmp_path):
    system, sources = _demo_system_with_modes(tmp_path)
    return dataclasses.replace(system, defaults={**system.defaults, **_DECLARED}), sources


def test_the_declaration_reaches_the_run_config(tmp_path):
    system, _ = _hinged_demo(tmp_path)
    cfg = TrainConfig(**_config_from_system(system))
    assert cfg.lambda_W == 0.3
    assert dict(cfg.extra_experiment_config) == {k: _DECLARED[k] for k in SHAPE_KEYS}
    assert KernelHinge.from_config(cfg) == KernelHinge(kappa=2.0, k_max=3.0, n_k=32, margin=0.0)


def test_a_declared_hinge_is_trained_logged_and_recorded(tmp_path):
    import yaml

    system, sources = _hinged_demo(tmp_path)
    run = fit(system, run_name="hinged", sources=sources, steps=2, seed=0,
              resume_optimizer=False, root=tmp_path / "data", log_every_step=True)
    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["terms_trained"] == ["L_dyn", "L_W"]
    assert man["declared_weights_without_data"] == {}
    assert man["kernel_hinge"] == {"term": "L_W", "kappa": 2.0, "k_max": 3.0, "n_k": 32,
                                   "margin": 0.0}
    steps = json.loads((run / "steps.json").read_text())
    assert all(math.isfinite(t["L_W"]) and t["L_W"] >= 0.0 for t in steps["terms"])
    hparams = yaml.safe_load((run / "hparams.yaml").read_text())
    assert hparams["lambda_W"] == 0.3
    assert hparams["extra_experiment_config"]["wpsd_kappa"] == 2.0
    assert not (run / "UNTRAINED_TERMS.txt").exists()


def test_the_hinge_adds_its_weighted_value_to_the_step_total(tmp_path):
    """Same seed, same first batch: the step-0 total moves by exactly ``lambda_W * L_W``."""
    system, sources = _hinged_demo(tmp_path)
    on = fit(system, run_name="on", sources=sources, steps=1, seed=3,
             resume_optimizer=False, root=tmp_path / "data", log_every_step=True)
    off = fit(system, run_name="off", sources=sources, steps=1, seed=3,
              resume_optimizer=False, root=tmp_path / "data", log_every_step=True,
              config_overrides={"lambda_W": 0.0})
    a = json.loads((on / "steps.json").read_text())
    b = json.loads((off / "steps.json").read_text())
    hinge = a["terms"][0]["L_W"]
    assert hinge > 0.0               # an untrained toy kernel violates the bound
    assert "L_W" not in b["terms"][0]
    assert a["terms"][0]["L_dyn"] == b["terms"][0]["L_dyn"]
    assert a["loss"][0] - b["loss"][0] == pytest.approx(0.3 * hinge, rel=1e-5)
    assert json.loads((off / "MANIFEST.json").read_text())["terms_trained"] == ["L_dyn"]
