"""``L_W`` on the number-conserving subspace, with a saturated floor: the reduced block, its basis, the
saturation, the declaration and the checkpoint round trip.

The full-matrix hinge and its driver are ``test_kernel_hinge.py``; nothing here changes them.
"""
import dataclasses
import json

import numpy as np
import pytest
import torch

from aipf.losses.extras import l_w, number_conserving_basis
from aipf.train.ckpt_compat import (IMPLICIT_WEIGHTS_A, NONLOSS_KEY_MAP_LIGHTNING,
                                    hparams_to_train_config, load_new_checkpoint,
                                    save_new_checkpoint)
from aipf.losses.names import WEIGHT_MAP_A
from aipf.train.config import TrainConfig
from aipf.train.fit import _config_from_system, fit
from aipf.train.kernel_hinge import OPTIONAL_KEYS, SHAPE_KEYS, KernelHinge, declared_fields

from test_kernel_hinge import _DECLARED, _hinged_demo
from test_train_fit import _demo_system_with_modes

_SUBSPACE = {"wpsd_subspace": "number_conserving", "wpsd_k_sat": 1.5}


def _random_symmetric(n, batch, seed):
    a = np.random.default_rng(seed).standard_normal((batch, n, n))
    return 0.5 * (a + np.swapaxes(a, 1, 2))


def _numpy_reduced_eigmin(D):
    """``lambda_min`` on ``1^perp`` by an independent route: the unit eigenvectors of the projector
    ``I - 1 1^T / n`` as the basis (whatever orientation numpy returns)."""
    n = D.shape[-1]
    w, v = np.linalg.eigh(np.eye(n) - np.ones((n, n)) / n)
    V = v[:, w > 0.5]
    return np.linalg.eigvalsh(np.einsum("ia,bij,jc->bac", V, D, V))[:, 0]


def _floor(k, kappa, k_sat):
    return kappa * np.minimum(k, k_sat) ** 2


# ---------------------------------------------------------------------------
# the basis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_basis_is_orthonormal_and_spans_the_number_conserving_subspace(n):
    B = number_conserving_basis(n)
    assert B.shape == (n, n - 1)
    assert torch.allclose(B.T @ B, torch.eye(n - 1, dtype=torch.float64), atol=1e-14)
    assert torch.allclose(torch.ones(n, dtype=torch.float64) @ B,
                          torch.zeros(n - 1, dtype=torch.float64), atol=1e-14)


def test_the_two_and_three_channel_bases_are_the_documented_columns():
    s2, s6 = 2.0 ** -0.5, 6.0 ** -0.5
    assert torch.allclose(number_conserving_basis(2),
                          torch.tensor([[s2], [-s2]], dtype=torch.float64), atol=1e-15)
    assert torch.allclose(number_conserving_basis(3),
                          torch.tensor([[s2, s6], [-s2, s6], [0.0, -2.0 * s6]],
                                       dtype=torch.float64), atol=1e-15)


def test_one_channel_is_refused_by_name():
    with pytest.raises(ValueError, match="number_conserving.*1 channel"):
        number_conserving_basis(1)


# ---------------------------------------------------------------------------
# the reduced block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3])
def test_the_reduced_block_eigenvalue_is_numpys_on_random_symmetric_kernels(n):
    """``l_w`` at ``kappa = 0`` and a large margin is ``mean((margin - lambda_min)^2)``, one ``k`` at a
    time: the reduced ``lambda_min`` read back out of it against numpy's, basis chosen independently."""
    D = _random_symmetric(n, 64, seed=n)
    expected = _numpy_reduced_eigmin(D)
    B = number_conserving_basis(n)
    margin = 50.0
    got = np.array([margin - float(l_w(torch.from_numpy(D[i:i + 1]), torch.ones(1, dtype=torch.float64),
                                       kappa=0.0, margin=margin, basis=B)) ** 0.5
                    for i in range(D.shape[0])])
    assert np.allclose(got, expected, rtol=0.0, atol=1e-10)


def test_the_eigenvalue_does_not_depend_on_the_basis():
    """``B Q`` for a random rotation ``Q`` of ``1^perp`` gives the same loss as ``B``."""
    D = torch.from_numpy(_random_symmetric(3, 16, seed=7))
    k = torch.linspace(0.1, 2.0, 16, dtype=torch.float64)
    B = number_conserving_basis(3)
    t = 0.83
    Q = torch.tensor([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=torch.float64)
    a = l_w(D, k, kappa=0.5, margin=3.0, basis=B)
    b = l_w(D, k, kappa=0.5, margin=3.0, basis=B @ Q)
    assert float(a) > 0.0 and float(a) == pytest.approx(float(b), rel=1e-12)


def test_the_projector_has_a_spurious_zero_eigenvalue_and_the_reduced_block_does_not():
    """A kernel stiff on ``1^perp`` (eigenvalues 2 and 3) and soft along ``1`` (-40, the liquid's
    total-number direction): the full matrix reads -40, the projector ``P D P`` reads 0 along ``1``
    whatever ``D`` is, the reduced block reads 2."""
    B = number_conserving_basis(3)
    one = torch.ones(3, 1, dtype=torch.float64) / 3.0 ** 0.5
    D = B @ torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64)) @ B.T - 40.0 * one @ one.T
    P = B @ B.T
    full = torch.linalg.eigvalsh(D)
    projected = torch.linalg.eigvalsh(P @ D @ P)
    assert float(full[0]) == pytest.approx(-40.0, abs=1e-12)
    assert float(projected[0]) == pytest.approx(0.0, abs=1e-12)
    w, v = torch.linalg.eigh(P @ D @ P)
    assert float(torch.abs(v[:, 0] @ one[:, 0])) == pytest.approx(1.0, abs=1e-12)
    k = torch.tensor([1.0], dtype=torch.float64)
    # floor 1: met on the subspace (2 >= 1), missed by the projector's zero (loss 1), and by the full matrix
    assert float(l_w(D[None], k, kappa=1.0, basis=B)) == 0.0
    assert float(l_w((P @ D @ P)[None], k, kappa=1.0)) == pytest.approx(1.0, abs=1e-12)
    assert float(l_w(D[None], k, kappa=1.0)) == pytest.approx(41.0 ** 2, rel=1e-12)


# ---------------------------------------------------------------------------
# the saturated floor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("basis", [None, "number_conserving"])
def test_the_floor_is_kappa_k_squared_below_k_sat_and_constant_from_it(basis):
    """``D = 0``: each wavenumber's loss is the floor squared, ``(kappa min(k, k_sat)^2)^2``."""
    k = torch.linspace(0.25, 4.0, 16, dtype=torch.float64)
    B = None if basis is None else number_conserving_basis(3)
    per_k = np.array([float(l_w(torch.zeros(1, 3, 3, dtype=torch.float64), k[i:i + 1],
                                kappa=0.25, basis=B, k_sat=1.5)) ** 0.5 for i in range(16)])
    assert np.allclose(per_k, _floor(k.numpy(), 0.25, 1.5), rtol=1e-12, atol=0.0)
    above = per_k[k.numpy() >= 1.5]
    assert above.size >= 2 and np.all(above == above[0])
    # without k_sat the floor keeps rising
    unsat = float(l_w(torch.zeros(1, 3, 3, dtype=torch.float64), k[-1:], kappa=0.25, basis=B))
    assert unsat ** 0.5 == pytest.approx(0.25 * 16.0, rel=1e-12)


# ---------------------------------------------------------------------------
# the hinge on a three-channel kernel with known eigenvalues
# ---------------------------------------------------------------------------

class _NumberSoftKernel(torch.nn.Module):
    """``W(k) = W0 + B diag(e1, e2) B^T + s(k) 1 1^T / 3``: ``e1 = a k^2``, ``e2 = b min(k, 1.5)^2``,
    ``s = -30 k^2`` (soft along the total number). The scale parameter carries the gradient."""

    n_species = 3

    def __init__(self, a, b):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
        self.a, self.b = a, b
        self.B = number_conserving_basis(3)

    def w_hat(self, k):
        e1, e2 = self.a * k * k, self.b * torch.clamp(k, max=1.5) ** 2
        sub = self.B @ torch.diag_embed(torch.stack([e1, e2], dim=-1)) @ self.B.T
        one = torch.ones(3, 3, dtype=torch.float64) / 3.0
        return self.scale * (sub - 30.0 * (k * k)[..., None, None] * one) - 17.0 * torch.eye(
            3, dtype=torch.float64)


class _Model(torch.nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel


def _subspace_hinge(**kw):
    return KernelHinge(kappa=0.25, k_max=3.96, n_k=128, margin=0.0,
                       subspace="number_conserving", k_sat=1.5, **kw)


def test_the_subspace_hinge_is_zero_where_the_full_matrix_one_is_not():
    model = _Model(_NumberSoftKernel(a=0.5, b=0.3))
    assert float(_subspace_hinge().loss(model)) == 0.0
    full = KernelHinge(kappa=0.25, k_max=3.96, n_k=128, margin=0.0)
    assert float(full.loss(model)) > 1.0


def test_a_violated_subspace_hinge_is_the_formula_with_a_finite_gradient():
    model = _Model(_NumberSoftKernel(a=0.5, b=0.1))     # e2 = 0.1 min(k, 1.5)^2 < the floor
    hinge = _subspace_hinge()
    k = hinge.wavenumbers(dtype=torch.float64).numpy()
    e = np.minimum(0.5 * k * k, 0.1 * np.minimum(k, 1.5) ** 2)
    expected = np.mean(np.maximum(_floor(k, 0.25, 1.5) - e, 0.0) ** 2)
    loss = hinge.loss(model)
    assert float(loss) > 0.0
    assert float(loss) == pytest.approx(expected, rel=1e-10)
    loss.backward()
    grad = model.kernel.scale.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad) != 0.0


def test_a_one_channel_kernel_is_refused_before_training():
    class _One(torch.nn.Module):
        n_species = 1

    model = _Model(_One())
    with pytest.raises(ValueError, match="number_conserving"):
        _subspace_hinge().check_subspace(model)
    KernelHinge(kappa=0.25, k_max=3.96, n_k=8, margin=0.0).check_subspace(model)   # no subspace: nothing


# ---------------------------------------------------------------------------
# the declaration, the record and the checkpoint
# ---------------------------------------------------------------------------

def test_the_optional_keys_are_registered_where_the_shape_keys_are():
    for key in OPTIONAL_KEYS:
        assert NONLOSS_KEY_MAP_LIGHTNING[key] == "extra_experiment_config", key
    fields = declared_fields({**_DECLARED, **_SUBSPACE})
    assert fields["extra_experiment_config"] == {
        **{k: _DECLARED[k] for k in SHAPE_KEYS}, **_SUBSPACE}


def test_a_hinge_without_the_optional_keys_is_the_one_it_always_was():
    plain = KernelHinge(kappa=2.0, k_max=3.0, n_k=32, margin=0.0, weight=0.3)
    assert plain.subspace is None and plain.k_sat is None
    assert set(plain.provenance()) == {"term", "kappa", "k_max", "n_k", "margin", "lambda_W"}
    cfg = TrainConfig(lambda_W=0.3, extra_experiment_config={k: _DECLARED[k] for k in SHAPE_KEYS})
    assert KernelHinge.from_config(cfg) == plain
    assert KernelHinge.from_config(cfg) != dataclasses.replace(plain, k_sat=1.5)


def test_the_optional_keys_reach_the_hinge_and_its_record():
    cfg = TrainConfig(lambda_W=0.3, extra_experiment_config={
        **{k: _DECLARED[k] for k in SHAPE_KEYS}, **_SUBSPACE})
    hinge = KernelHinge.from_config(cfg)
    assert hinge == KernelHinge(kappa=2.0, k_max=3.0, n_k=32, margin=0.0,
                                subspace="number_conserving", k_sat=1.5)
    assert hinge.provenance() == {"term": "L_W", "kappa": 2.0, "k_max": 3.0, "n_k": 32,
                                  "margin": 0.0, "lambda_W": 0.3,
                                  "subspace": "number_conserving", "k_sat": 1.5}


@pytest.mark.parametrize("bad, match", [({"wpsd_subspace": "concentration"}, "wpsd_subspace"),
                                        ({"wpsd_subspace": True}, "wpsd_subspace"),
                                        ({"wpsd_k_sat": 0.0}, "wpsd_k_sat"),
                                        ({"wpsd_k_sat": True}, "wpsd_k_sat"),
                                        ({"wpsd_k_sat": float("inf")}, "wpsd_k_sat")])
def test_an_unknown_subspace_and_a_non_positive_saturation_are_refused(bad, match):
    cfg = TrainConfig(lambda_W=0.3, extra_experiment_config={
        **{k: _DECLARED[k] for k in SHAPE_KEYS}, **bad})
    with pytest.raises(ValueError, match=match):
        KernelHinge.from_config(cfg)


def test_saved_hparams_with_the_optional_keys_round_trip_to_the_same_hinge(tmp_path):
    """A flat Lightning-hparams dict carrying the keys reads back (no leftover-key refusal) to the
    declared hinge, and this package's own checkpoint round-trips the config."""
    hparams = {**_DECLARED, **_SUBSPACE, "lr": 5e-4}
    cfg = hparams_to_train_config(hparams, weight_map=WEIGHT_MAP_A,
                                  key_map=NONLOSS_KEY_MAP_LIGHTNING,
                                  implicit_weights=IMPLICIT_WEIGHTS_A)
    declared = TrainConfig(**declared_fields(hparams), lr=5e-4, lambda_dyn=1.0)
    assert cfg == declared
    assert KernelHinge.from_config(cfg) == KernelHinge.from_config(declared)
    path = tmp_path / "x.ckpt"
    save_new_checkpoint(path, torch.nn.Linear(1, 1), cfg)
    _, back = load_new_checkpoint(path)
    assert back == cfg
    assert KernelHinge.from_config(back).k_sat == 1.5


def test_a_declared_subspace_hinge_is_trained_and_recorded(tmp_path):
    import yaml

    system, sources = _hinged_demo(tmp_path)
    system = dataclasses.replace(system, defaults={**system.defaults, **_SUBSPACE})
    assert KernelHinge.from_config(TrainConfig(**_config_from_system(system))).k_sat == 1.5
    run = fit(system, run_name="sub", sources=sources, steps=2, seed=0,
              resume_optimizer=False, root=tmp_path / "data", log_every_step=True)
    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["terms_trained"] == ["L_dyn", "L_W"]
    assert man["kernel_hinge"]["subspace"] == "number_conserving"
    assert man["kernel_hinge"]["k_sat"] == 1.5
    steps = json.loads((run / "steps.json").read_text())
    assert all(np.isfinite(t["L_W"]) and t["L_W"] >= 0.0 for t in steps["terms"])
    extra = yaml.safe_load((run / "hparams.yaml").read_text())["extra_experiment_config"]
    assert extra["wpsd_subspace"] == "number_conserving" and extra["wpsd_k_sat"] == 1.5


def test_a_run_that_declares_no_optional_key_writes_none(tmp_path):
    import yaml

    system, sources = _hinged_demo(tmp_path)
    run = fit(system, run_name="plain", sources=sources, steps=1, seed=0,
              resume_optimizer=False, root=tmp_path / "data")
    hparams = yaml.safe_load((run / "hparams.yaml").read_text())
    assert set(hparams["extra_experiment_config"]) == set(SHAPE_KEYS)
    text = (run / "MANIFEST.json").read_text() + (run / "hparams.yaml").read_text()
    assert "wpsd_subspace" not in text and "k_sat" not in text and "subspace" not in text


def test_the_demo_system_without_a_hinge_has_no_hinge_key(tmp_path):
    system, _ = _demo_system_with_modes(tmp_path)
    extra = TrainConfig(**_config_from_system(system)).extra_experiment_config
    assert not set(extra) & set(OPTIONAL_KEYS)


def test_a_float64_basis_reduces_a_float32_kernel_in_the_kernels_dtype():
    D = torch.from_numpy(_random_symmetric(3, 4, seed=3)).float()
    k = torch.linspace(0.5, 2.0, 4)
    loss = l_w(D, k, kappa=0.5, margin=3.0, basis=number_conserving_basis(3))
    assert loss.dtype == torch.float32 and float(loss) > 0.0
    exact = l_w(D.double(), k.double(), kappa=0.5, margin=3.0, basis=number_conserving_basis(3))
    assert float(loss) == pytest.approx(float(exact), rel=1e-5)
