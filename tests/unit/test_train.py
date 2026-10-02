"""Tests for `aipf.train.config` and `aipf.train.lit_module` (the loss-name
map and the training step).

`_ToyModel` below is not a rung -- it exists only so this file can exercise
a model that actually satisfies `FreeEnergyModel` (the same role
`tests/unit/test_model_protocol.py`'s `_ToyModelB` plays there), with one
trainable parameter so gradient-level reproducibility is actually
checkable, not merely tensor-value reproducibility.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from aipf.spectral import OpsCache
from aipf.train.ckpt_compat import load_new_checkpoint, save_new_checkpoint
from aipf.train.config import TrainConfig
from aipf.train.lit_module import LitModule
from aipf.train.pressure import pressure_from_model
from aipf.train.sampling import sample_uniform

GRID = (6, 5, 8)  # even last axis, matches aipf.spectral's own tests


class _ToyModel(nn.Module):
    """`mu = scale * rho`, `M = 1`, `f = 0.5 * scale * rho^2` -- the shared
    div(M grad mu) skeleton every rung follows, with one trainable
    parameter per species so a gradient exists to check.
    """

    def __init__(self, grid, n_species):
        super().__init__()
        self._cache = OpsCache(grid, n_species, nyquist_mask=True)
        self.ops = self._cache.ops
        self.n_species = n_species
        self.scale = nn.Parameter(torch.full((n_species,), 1.3))

    def _scale(self):
        return self.scale.view(1, -1, 1, 1, 1)

    def chemical_potential(self, rho, boxes, T):
        del boxes, T
        return self._scale() * rho

    def bulk_free_energy_density(self, rho, T):
        del T
        return 0.5 * (self._scale() * rho * rho).sum(dim=1, keepdim=True)

    def mobility(self, rho, T):
        del T
        return torch.ones_like(rho)

    def forward(self, rho_hat, boxes, T):
        ops = self._cache.ops_for(rho_hat.shape, rho_hat.device)
        N = ops.grid[0] * ops.grid[1] * ops.grid[2]
        rho = ops.irfft(rho_hat * N)
        mu = self.chemical_potential(rho, boxes, T)
        M = self.mobility(rho, T)
        kx, ky, kz = ops.k_axes(boxes)
        mu_hat = ops.rfft(mu) / N
        gx, gy, gz = ops.grad_hat(mu_hat * N, kx, ky, kz)
        grad_mu = torch.stack(
            [ops.irfft(gx), ops.irfft(gy), ops.irfft(gz)], dim=2)
        J = M.unsqueeze(2) * grad_mu
        Jx_hat = ops.rfft(J[:, :, 0]) / N
        Jy_hat = ops.rfft(J[:, :, 1]) / N
        Jz_hat = ops.rfft(J[:, :, 2]) / N
        return ops.div_hat(Jx_hat, Jy_hat, Jz_hat, kx, ky, kz)


def _config(**overrides):
    base = dict(sigma=2.0, k_max=2.0, alpha_loss=0.0, h_inv_eps=1e-6)
    base.update(overrides)
    return TrainConfig(**base)


def _boxes(batch, generator):
    return 8.0 + 4.0 * torch.rand(batch, 3, generator=generator)


def _rho_hat(batch, n_species, grid, ops, generator):
    rho = 0.3 + 0.1 * torch.rand(batch, n_species, *grid, generator=generator)
    return ops.rfft(rho) / (grid[0] * grid[1] * grid[2])


def _drift_batch(model, n_species, generator, *, n_s=2, B=3):
    ops = model.ops
    boxes = _boxes(B, generator)
    states = torch.stack(
        [_rho_hat(B, n_species, GRID, ops, generator) for _ in range(n_s)],
        dim=1)
    target = _rho_hat(B, n_species, GRID, ops, generator)
    lam = torch.rand(B, n_s, generator=generator)
    T = 1.0 + torch.rand(B, generator=generator)
    return {"rho_hat_states": states, "lam": lam, "target_hat": target,
            "boxes": boxes, "T": T}


# ---------------------------------------------------------------------------
# TrainConfig: the schema
# ---------------------------------------------------------------------------

def test_unknown_field_raises_at_construction():
    """No `**kwargs` catch-all: an unrecognised field raises straight from
    the dataclass, not from custom validation code."""
    with pytest.raises(TypeError):
        TrainConfig(this_is_not_a_field=1.0)


def test_from_dict_unknown_key_raises_naming_it():
    with pytest.raises(ValueError, match="unknown_field_xyz"):
        TrainConfig.from_dict({"sigma": 2.0, "unknown_field_xyz": 1.0})


def test_from_dict_round_trips_as_dict():
    cfg = _config(lambda_M=0.1, m_table="foo.npz")
    again = TrainConfig.from_dict(cfg.as_dict())
    assert again == cfg


@pytest.mark.parametrize("field_name,bad", [
    ("stat_metric", "not_a_metric"),
    ("bulk_residual", "not_a_residual"),
    ("conv_penalty", "not_a_penalty"),
])
def test_declared_string_fields_reject_an_unknown_value(field_name, bad):
    with pytest.raises(ValueError):
        TrainConfig(**{field_name: bad})


def test_mutable_default_fields_are_not_shared_between_instances():
    a = TrainConfig(model_kwargs={"x": 1})
    b = TrainConfig()
    assert b.model_kwargs == {}
    assert a.model_kwargs == {"x": 1}


def test_a_mapping_field_does_not_alias_the_callers_own_dict():
    """`TrainConfig` copies every mapping field on construction: mutating
    the CALLER's dict after building a config must not change the config's
    own value -- an aliased reference would let one caller's later edit
    silently corrupt an already-built, supposedly-frozen config."""
    caller_dict = {"x": 1}
    cfg = TrainConfig(model_kwargs=caller_dict)
    caller_dict["x"] = 999
    caller_dict["y"] = "new"
    assert cfg.model_kwargs == {"x": 1}


def test_sample_uniform_rejects_negative_n():
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        sample_uniform(-1, [0.0], [1.0], generator=g)


# ---------------------------------------------------------------------------
# LitModule.drift_loss / compute_losses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_drift_loss_is_finite_and_nonnegative(n_species):
    model = _ToyModel(GRID, n_species)
    lit = LitModule(model, model._cache, _config())
    g = torch.Generator().manual_seed(0)
    batch = _drift_batch(model, n_species, g)
    loss = lit.drift_loss(batch["rho_hat_states"], batch["lam"],
                           batch["target_hat"], batch["boxes"], batch["T"])
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_drift_loss_needs_sigma():
    model = _ToyModel(GRID, 1)
    lit = LitModule(model, model._cache, TrainConfig(k_max=2.0))
    g = torch.Generator().manual_seed(0)
    batch = _drift_batch(model, 1, g)
    with pytest.raises(ValueError, match="sigma"):
        lit.drift_loss(batch["rho_hat_states"], batch["lam"],
                        batch["target_hat"], batch["boxes"], batch["T"])


def test_resolved_k_max_falls_back_to_four_over_sigma():
    model = _ToyModel(GRID, 1)
    lit = LitModule(model, model._cache, TrainConfig(sigma=2.0))
    assert lit.resolved_k_max() == pytest.approx(2.0)


def test_resolved_k_max_prefers_an_explicit_value_over_four_over_sigma():
    """sigma=2.0 alone would resolve to 4/2=2.0 -- an explicit k_max that
    DIFFERS from that value pins that the explicit one wins, not merely
    that some number came back."""
    model = _ToyModel(GRID, 1)
    lit = LitModule(model, model._cache, TrainConfig(sigma=2.0, k_max=5.0))
    assert lit.resolved_k_max() == pytest.approx(5.0)


def test_compute_losses_totals_the_weighted_parts():
    n = 2
    model = _ToyModel(GRID, n)
    # lambda_dyn deliberately != 1.0: if the total silently dropped its
    # weighting (adding the raw part instead), 1.0 would hide it -- this
    # would not.
    config = _config(lambda_dyn=0.6, lambda_M=0.5)
    lit = LitModule(model, model._cache, config)
    g = torch.Generator().manual_seed(0)
    batch = {"drift": _drift_batch(model, n, g)}
    M_pred = torch.eye(n).expand(4, n, n).clone()
    M_target = torch.eye(n).expand(4, n, n) * 1.1
    batch["anchor_M"] = {"M_pred": M_pred, "M_target": M_target}
    total, parts = lit.compute_losses(batch)
    assert set(parts) == {"L_dyn", "L_M"}
    expected = config.lambda_dyn * parts["L_dyn"] + config.lambda_M * parts["L_M"]
    assert torch.allclose(total, expected)
    assert not torch.allclose(total, parts["L_dyn"] + parts["L_M"])  # not unweighted


def test_compute_losses_skips_a_group_whose_weight_is_zero():
    """A declared-but-zero-weight
    anchor contributes nothing -- and here it is not even computed."""
    n = 2
    model = _ToyModel(GRID, n)
    config = _config(lambda_dyn=1.0, lambda_M=0.0)  # off
    lit = LitModule(model, model._cache, config)
    g = torch.Generator().manual_seed(0)
    batch = {"drift": _drift_batch(model, n, g)}
    batch["anchor_M"] = {"M_pred": torch.zeros(1, n, n),
                          "M_target": torch.full((1, n, n), float("nan"))}
    total, parts = lit.compute_losses(batch)
    assert "L_M" not in parts
    assert torch.isfinite(total)


def test_compute_losses_bulk_and_pressure_and_static_wire_through():
    n = 2
    model = _ToyModel(GRID, n)
    config = _config(lambda_S=1.0, lambda_bulk=1.0, lambda_P=1.0)
    lit = LitModule(model, model._cache, config)
    H = torch.eye(n).expand(3, 2, n, n).clone()
    target = torch.eye(n).expand(3, 2, n, n) * 1.2
    mask = torch.ones(3, 2, dtype=torch.bool)
    zvec = torch.ones(3, n) / n
    batch = {
        "anchor_S": {"H": H, "target": target, "mask": mask},
        "anchor_bulk": {"H0": torch.eye(n).expand(3, n, n).clone(),
                         "kBT": torch.ones(3), "zvec": zvec,
                         "rho_tot": torch.ones(3), "target": torch.ones(3) * 0.5},
        "anchor_P": {"P_model": torch.tensor([1.0, 1.1, 0.9]),
                     "P_target": torch.tensor([1.0, 1.0, 1.0])},
    }
    total, parts = lit.compute_losses(batch)
    assert set(parts) == {"L_S", "L_bulk", "L_P"}
    for v in parts.values():
        assert torch.isfinite(v)


# ---------------------------------------------------------------------------
# Save / load round trip: bit for bit
# ---------------------------------------------------------------------------

def test_save_load_round_trip_reproduces_every_parameter_bit_for_bit(tmp_path):
    model = _ToyModel(GRID, 2)
    config = _config(lambda_M=0.3, m_table="anchor.npz")
    path = tmp_path / "ckpt.pt"
    save_new_checkpoint(path, model, config)

    state_dict, loaded_config = load_new_checkpoint(path)
    assert loaded_config == config

    fresh = _ToyModel(GRID, 2)
    with torch.no_grad():
        fresh.scale.fill_(-99.0)  # make sure the round trip, not luck, matches
    fresh.load_state_dict(state_dict)

    saved = model.state_dict()
    got = fresh.state_dict()
    assert set(saved) == set(got)
    for key in saved:
        assert torch.equal(saved[key], got[key]), key


# ---------------------------------------------------------------------------
# Generator reproducibility: new runs, not old ones
# ---------------------------------------------------------------------------

def test_sample_uniform_needs_an_explicit_generator():
    import inspect
    sig = inspect.signature(sample_uniform)
    assert sig.parameters["generator"].default is inspect.Parameter.empty


def test_sample_uniform_same_seed_gives_identical_draws():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = sample_uniform(100, [0.0, 0.0], [1.0, 2.0], generator=g1)
    b = sample_uniform(100, [0.0, 0.0], [1.0, 2.0], generator=g2)
    assert torch.equal(a, b)


def test_two_runs_with_the_same_explicit_generator_give_identical_first_step_gradients():
    """The property the brief asks for directly: build two independently-
    constructed but architecturally identical models (a `copy.deepcopy`, so
    initial weights match exactly), draw each run's batch from its OWN
    fresh `torch.Generator` seeded alike, and confirm every gradient
    matches bit for bit after one backward pass. This is what "pass an
    explicit generator" buys for a NEW run -- see the module docstrings in
    aipf.train.sampling and aipf.train.ckpt_compat for what it does NOT
    buy (replaying an OLD, `generator=None` run).
    """
    n = 2
    model_a = _ToyModel(GRID, n)
    model_b = copy.deepcopy(model_a)
    config = _config()
    lit_a = LitModule(model_a, model_a._cache, config)
    lit_b = LitModule(model_b, model_b._cache, config)

    g_a = torch.Generator().manual_seed(2026)
    g_b = torch.Generator().manual_seed(2026)
    batch_a = {"drift": _drift_batch(model_a, n, g_a)}
    batch_b = {"drift": _drift_batch(model_b, n, g_b)}

    total_a, _ = lit_a.compute_losses(batch_a)
    total_b, _ = lit_b.compute_losses(batch_b)
    total_a.backward()
    total_b.backward()

    assert torch.equal(model_a.scale.grad, model_b.scale.grad)
    assert model_a.scale.grad.abs().sum().item() > 0.0  # not a vacuous zero


def test_sample_uniform_respects_its_bounds():
    g = torch.Generator().manual_seed(3)
    pts = sample_uniform(2000, [-1.0, 5.0], [1.0, 6.0], generator=g)
    assert pts.shape == (2000, 2)
    assert pts[:, 0].min() >= -1.0 and pts[:, 0].max() <= 1.0
    assert pts[:, 1].min() >= 5.0 and pts[:, 1].max() <= 6.0
    # and actually spans the range, not just satisfies the bound trivially
    assert pts[:, 0].max() - pts[:, 0].min() > 1.5


def test_pressure_from_model_matches_the_euler_relation_by_hand():
    """`P = mu.rho - f` for the toy model's own `mu = scale*rho`,
    `f = 0.5*scale*rho^2`: `P = scale*rho^2 - 0.5*scale*rho^2 =
    0.5*scale*rho^2`, checked against a hand value, not merely "finite"."""
    model = _ToyModel(GRID, 2)
    with torch.no_grad():
        model.scale.copy_(torch.tensor([2.0, 3.0]))
    rho = torch.tensor([[0.5, 0.25], [1.0, 0.1]])
    boxes = torch.full((2, 3), 10.0)
    T = torch.ones(2)
    P = pressure_from_model(model, rho, boxes, T)
    expected = 0.5 * (torch.tensor([2.0, 3.0]) * rho * rho).sum(dim=-1)
    assert torch.allclose(P, expected)


def test_drift_loss_is_near_zero_when_the_target_is_the_models_own_prediction():
    """A hand-consistency check on `drift_loss`'s own residual construction:
    build a single-state batch (`n_s=1`, `lam=1`) whose target is exactly
    `sigma_filter(states) -> model.forward`, so the residual this loss is
    built from is exactly zero before any band-masking -- catches a sign or
    filter-application bug that "finite and nonnegative" cannot."""
    n = 2
    model = _ToyModel(GRID, n)
    lit = LitModule(model, model._cache, _config())
    g = torch.Generator().manual_seed(7)
    ops = model.ops
    boxes = _boxes(2, g)
    state = _rho_hat(2, n, GRID, ops, g).unsqueeze(1)          # (B,1,n,...)
    T = torch.ones(2)
    filt = ops.sigma_filter(boxes, 2.0)
    filtered = state.squeeze(1) * filt
    with torch.no_grad():
        pred_hat = model(filtered, boxes, T)
        # drift_loss's own residual is `target_hat*filt - rhs_hat`; choosing
        # `target_hat = pred_hat / filt` makes that residual exactly zero
        # (filt = exp(-k^2 sigma^2/2) > 0 everywhere, so this is safe).
        target_hat = pred_hat / filt
    lam = torch.ones(2, 1)
    loss = lit.drift_loss(state, lam, target_hat, boxes, T)
    assert loss.item() < 1e-8


def test_two_runs_with_independently_seeded_generators_generally_disagree():
    """The contrast case: two DIFFERENT seeds draw different batches, so
    (generically) different gradients -- confirming the match above is
    because the seeds agree, not because this drift loss is insensitive to
    its input.
    """
    n = 2
    model_a = _ToyModel(GRID, n)
    model_b = copy.deepcopy(model_a)
    config = _config()
    lit_a = LitModule(model_a, model_a._cache, config)
    lit_b = LitModule(model_b, model_b._cache, config)

    g_a = torch.Generator().manual_seed(1)
    g_b = torch.Generator().manual_seed(2)
    batch_a = {"drift": _drift_batch(model_a, n, g_a)}
    batch_b = {"drift": _drift_batch(model_b, n, g_b)}

    total_a, _ = lit_a.compute_losses(batch_a)
    total_b, _ = lit_b.compute_losses(batch_b)
    total_a.backward()
    total_b.backward()

    assert not torch.equal(model_a.scale.grad, model_b.scale.grad)
