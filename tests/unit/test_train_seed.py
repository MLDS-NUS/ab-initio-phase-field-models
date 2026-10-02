"""The training seed: what reads it, and what declaring one actually buys.

`TrainConfig.seed` existed and nothing in `src/aipf/` read it, so a rung's
parameters came from the global torch stream at construction and two runs
from one config started from different weights. These tests are the
measurement that this is closed for a NEW run: build twice from one config
and compare every parameter with `torch.equal`, draw the penalty probes'
points twice and compare them, take one optimizer step twice and compare
the gradients and the post-step weights. Nothing here claims anything
about an ARCHIVED run -- no published checkpoint records the global seed its
weights were drawn under, nor how many draws preceded them, so those are
not replayable and no test here pretends otherwise.

`_ToyModel` is not a rung: it exists so these tests can exercise a model
that satisfies `FreeEnergyModel` and whose parameters are DRAWN at
construction (the rungs that ship untrained initialise to zeros, so they
could not detect an unseeded build at all). The one real rung this file
also builds is `nonlocal_kernel`, the trained one, whose MLP heads do draw.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from aipf.functional.nonlocal_kernel import NonlocalKernel
from aipf.spectral import OpsCache
from aipf.train.config import SEED_STREAMS, TrainConfig, derive_seed
from aipf.system import TrustDomain
from aipf.train.lit_module import LitModule, seeded_rng
from aipf.train.penalties import ConvexityProbe, Penalties

GRID = (6, 5, 8)  # even last axis, matching aipf.spectral's own tests


class _ToyModel(nn.Module):
    """`mu = a*rho + b*rho^2`, `M = 1`, with BOTH parameters drawn from the
    global torch stream at construction, so an unseeded build is visible.
    """

    def __init__(self, grid, n_species):
        super().__init__()
        self._cache = OpsCache(grid, n_species, nyquist_mask=True)
        self.ops = self._cache.ops
        self.n_species = n_species
        self.a = nn.Parameter(0.8 + torch.rand(n_species))
        self.b = nn.Parameter(0.1 * torch.rand(n_species))

    def _v(self, p):
        return p.view(1, -1, 1, 1, 1)

    def chemical_potential(self, rho, boxes, T):
        del boxes, T
        return self._v(self.a) * rho + self._v(self.b) * rho * rho

    def bulk_free_energy_density(self, rho, T):
        del T
        f = 0.5 * self._v(self.a) * rho * rho + self._v(self.b) * rho ** 3 / 3
        return f.sum(dim=1, keepdim=True)

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
        return ops.div_hat(ops.rfft(J[:, :, 0]) / N, ops.rfft(J[:, :, 1]) / N,
                           ops.rfft(J[:, :, 2]) / N, kx, ky, kz)


def _config(**overrides):
    base = dict(sigma=2.0, k_max=2.0, seed=2026)
    base.update(overrides)
    return TrainConfig(**base)


def _rung_kwargs(n_species):
    """Small, fast, and every value an arbitrary test fixture -- this
    package ships no default for any of them because each is one system's
    own number (the same fixture `tests/unit/test_nonlocal_kernel.py` uses).
    """
    return dict(
        kB=1.0, rho_ref=[0.5] * n_species, kBT_ref=1.0, h_u=4, h_g=4,
        R_cut=3.0, kernel_n_quad=17, kernel_n_k_table=17,
        kernel_k_table_max=8.0, kernel_hidden=4,
        mobility_prefactor="mole_fraction", mobility_shape="mlp_rho",
        mobility_t_form="none", mobility_hidden=4, ideal_form="gas",
        nyquist_mask=True)


def _drift_batch(model, n_species, generator, *, n_s=2, B=3):
    ops = model.ops
    N = GRID[0] * GRID[1] * GRID[2]

    def rho_hat():
        rho = 0.3 + 0.1 * torch.rand(
            B, n_species, *GRID, generator=generator)
        return ops.rfft(rho) / N

    return {"rho_hat_states": torch.stack([rho_hat() for _ in range(n_s)],
                                          dim=1),
            "lam": torch.rand(B, n_s, generator=generator),
            "target_hat": rho_hat(),
            "boxes": 8.0 + 4.0 * torch.rand(B, 3, generator=generator),
            "T": 1.0 + torch.rand(B, generator=generator)}


def _params(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _assert_same_parameters(a, b):
    """Parameter by parameter, by name, with no tolerance."""
    assert sorted(a) == sorted(b)
    assert a, "a model with no parameters proves nothing"
    for name in sorted(a):
        assert a[name].shape == b[name].shape, name
        assert torch.equal(a[name], b[name]), name


def _update(h, tensors):
    for name in sorted(tensors):
        h.update(name.encode("ascii"))
        h.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())


def subprocess_digest(n_species=2, seed=4242):
    """The measurement the cross-process test repeats in a second
    interpreter: build a real rung under the declared seed, draw the
    penalty points, take one whole optimizer step, and digest the model
    before it, the points, the total, every gradient and the model after
    it. Not a test -- called by name from the child process, so both
    processes run THIS code, not two copies of it.
    """
    cfg = TrainConfig(sigma=2.0, k_max=2.0, seed=seed, conv_samples=7,
                      lambda_conv=0.5, lr=10.0, penalty_seed=seed)
    lit = LitModule.seeded(
        lambda: NonlocalKernel(GRID, n_species, **_rung_kwargs(n_species)),
        OpsCache(GRID, n_species, nyquist_mask=True), cfg)
    h = hashlib.sha256()
    _update(h, lit.model.state_dict())
    lit._trainer = _FakeTrainer()
    opt = lit.configure_optimizers()["optimizer"]
    batch = {"drift": _drift_batch(
        lit.model, n_species,
        torch.Generator().manual_seed(cfg.seed_for("data")))}
    penalties = _penalties(cfg, n_species)
    points, _ = penalties.draw()
    conv, _ = penalties.losses(lit.model, (points, None))
    total, _parts = lit.compute_losses(batch, conv=conv)
    total.backward()
    h.update(points[0].detach().numpy().tobytes())
    h.update(total.detach().numpy().tobytes())
    _update(h, {k: v.grad for k, v in lit.model.named_parameters()
                if v.grad is not None})
    opt.step()
    _update(h, lit.model.state_dict())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the seed streams and their derivation (aipf.train.config)
# ---------------------------------------------------------------------------

def test_the_declared_streams_are_the_two_a_run_draws_from():
    """The penalty points have their own declared seed (``penalty_seed``), not a child stream."""
    assert SEED_STREAMS == ("model", "data")


def test_derive_seed_is_a_function_of_the_seed_and_the_stream():
    assert derive_seed(7, "model") == derive_seed(7, "model")


@pytest.mark.parametrize("a,b", [("model", "data")])
def test_two_streams_of_one_seed_are_different_streams(a, b):
    """Otherwise the model's weights and the batch order would be one
    stream, and drawing either would move the other."""
    assert derive_seed(11, a) != derive_seed(11, b)


def test_two_seeds_give_different_stream_seeds():
    assert derive_seed(1, "model") != derive_seed(2, "model")


def test_adjacent_seeds_do_not_share_a_stream():
    """`seed + 1` as a derivation would make run 1's `data` stream run
    2's `model` stream. Measured over a range, not at one pair."""
    for seed in range(50):
        got = {derive_seed(s, st) for s in (seed, seed + 1)
               for st in SEED_STREAMS}
        assert len(got) == 2 * len(SEED_STREAMS)


def test_a_derived_seed_is_one_torch_accepts():
    for seed in (-5, 0, 1, 2 ** 40):
        for stream in SEED_STREAMS:
            child = derive_seed(seed, stream)
            assert 0 <= child < 2 ** 63
            torch.Generator().manual_seed(child)  # raises if out of range


def test_the_derivation_itself_is_pinned_to_its_values():
    """A golden pin, not a restatement: "the same config gives the same
    model" has to hold across VERSIONS of this package too, and every
    plausible edit to the derivation (a different digest, a different
    slice, a different byte order, a dropped separator) keeps every
    property above while silently moving every declared run's weights.
    Changing these numbers is a decision about every seeded run there has
    ever been, so it should have to be made on purpose.
    """
    assert [derive_seed(316, s) for s in SEED_STREAMS] == [
        2712470501721526283, 4943008123427671747]


def test_an_unknown_stream_raises_naming_the_declared_ones():
    with pytest.raises(ValueError, match="modle"):
        derive_seed(3, "modle")
    with pytest.raises(ValueError, match="data"):
        derive_seed(3, "modle")
    with pytest.raises(ValueError, match="sampling"):
        derive_seed(3, "sampling")


@pytest.mark.parametrize("bad", [1.5, "7", True, None])
def test_derive_seed_rejects_a_non_integer_seed(bad):
    """`True` included deliberately: it is an `int` to Python, and a
    boolean reaching a seed is a mistake, never a choice."""
    with pytest.raises(TypeError):
        derive_seed(bad, "model")


def test_the_derived_seed_is_stable_across_processes():
    """The property `hash()` does NOT have: Python randomises the hash of a
    `str` per process, so a `hash`-based derivation would replay a run with
    different numbers in a second interpreter and never say so.
    """
    out = subprocess.run(
        [sys.executable, "-c",
         "from aipf.train.config import derive_seed, SEED_STREAMS;"
         "print([derive_seed(316, s) for s in SEED_STREAMS])"],
        capture_output=True, text=True, check=True, env=_child_env())
    assert out.stdout.strip() == str(
        [derive_seed(316, s) for s in SEED_STREAMS])


def _child_env():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return env


# ---------------------------------------------------------------------------
# TrainConfig.seed: the caller's, with no default
# ---------------------------------------------------------------------------

def test_the_seed_field_has_no_default_value():
    """A seed is the caller's. A numeric default would make two runs that
    declared nothing agree by accident, and make an undeclared run look
    reproducible."""
    field, = [f for f in dataclasses.fields(TrainConfig) if f.name == "seed"]
    assert field.default is None
    assert TrainConfig().seed is None


def test_seed_for_raises_when_the_run_declared_no_seed():
    with pytest.raises(ValueError, match="no seed"):
        TrainConfig().seed_for("model")


def test_seed_for_is_the_streams_child_seed():
    cfg = _config(seed=99)
    for stream in SEED_STREAMS:
        assert cfg.seed_for(stream) == derive_seed(99, stream)


def test_seed_for_rejects_an_unknown_stream():
    with pytest.raises(ValueError, match="unknown seed stream"):
        _config(seed=99).seed_for("weights")


@pytest.mark.parametrize("bad", [1.5, "316", True])
def test_a_non_integer_seed_raises_at_construction(bad):
    """A seed read out of a config file as a string or a float derives
    different stream seeds from the same written number, so it is refused
    where it enters, not where it is used."""
    with pytest.raises(TypeError, match="seed"):
        TrainConfig(seed=bad)


def test_the_seed_round_trips_through_the_config_dict():
    cfg = _config(seed=-7)
    assert TrainConfig.from_dict(cfg.as_dict()).seed == -7


# ---------------------------------------------------------------------------
# seeded_rng: the global stream, forked and seeded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_two_builds_under_one_seed_give_identical_parameters(n_species):
    with seeded_rng(1234):
        a = _params(_ToyModel(GRID, n_species))
    with seeded_rng(1234):
        b = _params(_ToyModel(GRID, n_species))
    _assert_same_parameters(a, b)


@pytest.mark.parametrize("n_species", [1, 2])
def test_two_builds_under_different_seeds_differ(n_species):
    """Non-vacuity: the comparison above would also pass on a model whose
    parameters never varied at all."""
    with seeded_rng(1234):
        a = _params(_ToyModel(GRID, n_species))
    with seeded_rng(5678):
        b = _params(_ToyModel(GRID, n_species))
    assert not torch.equal(a["a"], b["a"])


def test_seeded_rng_restores_the_stream_it_forked():
    """Seeding a model must not silently reseed everything built after it."""
    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    with seeded_rng(999):
        _ToyModel(GRID, 2)
    assert torch.equal(torch.get_rng_state(), before)


def test_the_stream_outside_the_block_is_untouched_by_what_it_draws():
    torch.manual_seed(3)
    expected = torch.rand(4)
    torch.manual_seed(3)
    with seeded_rng(999):
        torch.rand(100)
    assert torch.equal(torch.rand(4), expected)


def test_seeding_a_cpu_model_does_not_initialise_an_accelerator():
    """`fork_rng`'s own default reads and restores every device's state,
    which initialises the accelerator. Building a small model on the CPU
    must not pay that, and must not turn the device on behind the caller.

    Measured in a FRESH interpreter: in this one, an earlier test has
    already called `seeded_rng`, so "unchanged since the last call" would
    be satisfied by a version that initialised the device on its first
    call and every call after. The check is only meaningful where a device
    is visible, which the child reports alongside the answer.
    """
    out = subprocess.run(
        [sys.executable, "-c",
         "import torch\n"
         "from aipf.train.lit_module import seeded_rng\n"
         "assert not torch.cuda.is_initialized()\n"
         "with seeded_rng(4):\n"
         "    torch.nn.Linear(4, 4)\n"
         "print(torch.cuda.device_count(), torch.cuda.is_initialized())"],
        capture_output=True, text=True, check=True, env=_child_env())
    devices, initialised = out.stdout.split()
    assert initialised == "False", out.stdout
    assert int(devices) >= 0


def test_seeding_does_not_queue_a_reseed_of_the_accelerators_stream():
    """`torch.manual_seed` also seeds every accelerator, and on a machine
    where none is initialised that seeding is QUEUED rather than applied --
    so it would outlive the fork, which restores only what it forked, and
    land on the first accelerator use after it. The queue is the only trace
    it leaves here, and reading it is the only way to see the leak without
    initialising a device. `torch.cuda._lazy_seed_tracker` is private: if a
    future torch drops it this test fails loudly, which is the right
    outcome for a claim about what this package does NOT touch.
    """
    before = list(torch.cuda._lazy_seed_tracker.get_calls())
    with seeded_rng(21):
        _ToyModel(GRID, 1)
    # the queue holds one slot per kind, so a second seeding REPLACES an
    # entry rather than lengthening the list: compare the entries
    assert torch.cuda._lazy_seed_tracker.get_calls() == before


# ---------------------------------------------------------------------------
# LitModule.seeded: the model, from the declared seed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_two_modules_seeded_from_one_config_hold_identical_models(n_species):
    cfg = _config(seed=77)
    build = lambda: _ToyModel(GRID, n_species)  # noqa: E731
    a = LitModule.seeded(build, OpsCache(GRID, n_species, nyquist_mask=True), cfg)
    b = LitModule.seeded(build, OpsCache(GRID, n_species, nyquist_mask=True), cfg)
    _assert_same_parameters(_params(a.model), _params(b.model))


@pytest.mark.parametrize("n_species", [1, 2])
def test_a_real_rungs_weights_are_a_function_of_the_declared_seed(n_species):
    """The claim is about the rungs that actually train, not only about a
    toy: rung 3's heads draw at construction, and two builds from one
    config agree parameter by parameter while two seeds disagree."""
    kw = _rung_kwargs(n_species)
    build = lambda: NonlocalKernel(GRID, n_species, **kw)  # noqa: E731
    ops = OpsCache(GRID, n_species, nyquist_mask=True)
    a = LitModule.seeded(build, ops, _config(seed=5))
    b = LitModule.seeded(build, ops, _config(seed=5))
    c = LitModule.seeded(build, ops, _config(seed=6))
    pa, pb, pc = _params(a.model), _params(b.model), _params(c.model)
    _assert_same_parameters(pa, pb)
    assert any(not torch.equal(pa[k], pc[k]) for k in pa)


def test_the_model_stream_is_the_one_the_weights_come_from():
    """Pins WHICH stream builds the model: swapping it for another would
    still be reproducible and would still be wrong, because the data
    stream would then be spent on weights."""
    cfg = _config(seed=31)
    with seeded_rng(cfg.seed_for("model")):
        expected = _params(_ToyModel(GRID, 2))
    lit = LitModule.seeded(lambda: _ToyModel(GRID, 2), OpsCache(GRID, 2, nyquist_mask=True), cfg)
    _assert_same_parameters(_params(lit.model), expected)


def test_seeded_refuses_a_config_that_declares_no_seed():
    with pytest.raises(ValueError, match="no seed"):
        LitModule.seeded(lambda: _ToyModel(GRID, 1), OpsCache(GRID, 1, nyquist_mask=True),
                         TrainConfig(sigma=2.0))


def test_seeded_refuses_a_model_that_is_not_a_builder():
    """Handing it an already-built model would build nothing under the
    seed and silently keep whatever weights that model already had."""
    model = _ToyModel(GRID, 1)
    with pytest.raises(TypeError, match="build_model"):
        LitModule.seeded(model, OpsCache(GRID, 1, nyquist_mask=True), _config())


def test_a_seeded_build_does_not_leak_into_the_global_stream():
    torch.manual_seed(12)
    before = torch.get_rng_state().clone()
    LitModule.seeded(lambda: _ToyModel(GRID, 2), OpsCache(GRID, 2, nyquist_mask=True), _config())
    assert torch.equal(torch.get_rng_state(), before)


# ---------------------------------------------------------------------------
# the penalty points: aipf.train.penalties, on the declared penalty_seed
# ---------------------------------------------------------------------------

def _penalties(cfg, n_species, *, margin=100.0):
    """``L_conv``'s probe on a toy domain, seeded by the config's ``penalty_seed``; the margin
    opens the hinge on every point so a gradient flows through it."""
    domain = TrustDomain(inner=(0.2,) * n_species, outer=(1.0,) * n_species,
                         T_range=(1.0, 2.0))
    probe = ConvexityProbe(domain=domain, n=int(cfg.conv_samples),
                           T_measure="uniform", form="hinge", margin=margin)
    return Penalties(cfg.penalty_seed, probe, None)


def test_an_already_built_pair_is_weighted_and_a_zero_weight_drops_it():
    loss = torch.tensor(0.25)
    lit = LitModule(_ToyModel(GRID, 1), OpsCache(GRID, 1, nyquist_mask=True),
                    _config(lambda_conv=2.0))
    total, parts = lit.compute_losses({}, conv=(loss, {"viol_frac": 0.0}))
    assert torch.equal(parts["L_conv"], loss)
    assert float(total) == pytest.approx(0.5)
    off = LitModule(_ToyModel(GRID, 1), OpsCache(GRID, 1, nyquist_mask=True),
                    _config(lambda_conv=0.0))
    _total, parts = off.compute_losses({}, conv=(loss, {"viol_frac": 0.0}))
    assert "L_conv" not in parts


def test_the_module_draws_no_points_of_its_own():
    """One route to penalty points: :mod:`aipf.train.penalties`, never the module."""
    lit = LitModule(_ToyModel(GRID, 1), OpsCache(GRID, 1, nyquist_mask=True),
                    _config())
    assert not hasattr(lit, "draw_points") and not hasattr(lit, "generator")


@pytest.mark.parametrize("n_species", [1, 2])
def test_a_convexity_penalty_over_the_drawn_points_is_reproducible(n_species):
    """The intended use end to end: the Hessian of the model's own bulk free energy at the
    declared stream's points, hinged, twice from one config."""
    cfg = _config(seed=3, lambda_conv=1.0, conv_samples=4, penalty_seed=5)
    out = []
    for _ in range(2):
        lit = LitModule.seeded(lambda: _ToyModel(GRID, n_species),
                               OpsCache(GRID, n_species, nyquist_mask=True), cfg)
        conv, _ = _penalties(cfg, n_species, margin=3.0).losses(lit.model)
        total, parts = lit.compute_losses({}, conv=conv)
        total.backward()
        out.append((total.detach().clone(),
                    lit.model.a.grad.detach().clone()))
    assert float(out[0][0]) > 0.0  # the hinge actually bites
    assert torch.equal(out[0][0], out[1][0])
    assert torch.equal(out[0][1], out[1][1])


# ---------------------------------------------------------------------------
# the headline: one config, two runs, the same first step
# ---------------------------------------------------------------------------

def _one_run(n_species, cfg):
    """Build from the config alone, draw the penalty points, take ONE
    optimizer step with the module's OWN configured optimizer.
    """
    lit = LitModule.seeded(lambda: _ToyModel(GRID, n_species),
                           OpsCache(GRID, n_species, nyquist_mask=True), cfg)
    initial = _params(lit.model)
    batch = {"drift": _drift_batch(
        lit.model, n_species,
        torch.Generator().manual_seed(cfg.seed_for("data")))}
    lit._trainer = _FakeTrainer()
    opt = lit.configure_optimizers()["optimizer"]
    penalties = _penalties(cfg, n_species)
    points, _ = penalties.draw()
    conv, _ = penalties.losses(lit.model, (points, None))
    total, parts = lit.compute_losses(batch, conv=conv)
    total.backward()
    grads = {n: p.grad.detach().clone() for n, p in
             lit.model.named_parameters()}
    opt.step()
    return {"initial": initial, "points": points[0], "total":
            total.detach().clone(), "grads": grads,
            "stepped": _params(lit.model)}


class _FakeTrainer:
    """`configure_optimizers` reads two numbers off the trainer. Supplying
    them directly keeps this a test of the optimizer and schedule this
    module builds, not of Lightning's loop."""
    estimated_stepping_batches = 100
    max_epochs = 10


@pytest.mark.parametrize("n_species", [1, 2])
def test_two_runs_from_one_config_agree_on_model_draws_and_first_step(
        n_species):
    # `lr` is a fixture, not a measurement: `configure_optimizers` starts
    # its warm-up at a factor of 1e-6, so at the schema's own default lr a
    # first step moves an O(1) float32 parameter by less than its last
    # place -- the step would be identical across runs and invisible, and
    # an invisible "same" proves nothing.
    cfg = _config(seed=2026, lambda_conv=0.5, conv_samples=8, lr=10.0,
                  penalty_seed=2026)
    a = _one_run(n_species, cfg)
    b = _one_run(n_species, cfg)

    _assert_same_parameters(a["initial"], b["initial"])
    assert torch.equal(a["points"], b["points"])
    assert torch.equal(a["total"], b["total"])
    assert sorted(a["grads"]) == sorted(b["grads"])
    for name in sorted(a["grads"]):
        assert torch.equal(a["grads"][name], b["grads"][name]), name
        assert a["grads"][name].abs().sum() > 0.0, name  # not vacuously zero
    _assert_same_parameters(a["stepped"], b["stepped"])
    # and the step MOVED every TRAINABLE parameter, so "same after" is not
    # "unchanged" (the state dict also carries the spectral buffers, which
    # no optimizer touches and which are not what this is checking)
    for name in a["grads"]:
        assert not torch.equal(a["initial"][name], a["stepped"][name]), name


@pytest.mark.parametrize("n_species", [1, 2])
def test_two_runs_at_different_seeds_disagree_on_all_three(n_species):
    a = _one_run(n_species, _config(seed=2026, lambda_conv=0.5,
                                    conv_samples=8, penalty_seed=2026))
    b = _one_run(n_species, _config(seed=2027, lambda_conv=0.5,
                                    conv_samples=8, penalty_seed=2027))
    assert not torch.equal(a["initial"]["a"], b["initial"]["a"])
    assert not torch.equal(a["points"], b["points"])
    assert not torch.equal(a["total"], b["total"])


def test_the_whole_first_step_is_the_same_in_another_process():
    """One process is not reproducibility. The child runs THIS file's own
    `subprocess_digest` -- a real rung built under the seed, its penalty
    points, its total, every gradient and the weights after one optimizer
    step -- so both sides run one implementation."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {here!r});"
         "import test_train_seed as t; print(t.subprocess_digest())"],
        capture_output=True, text=True, check=True, env=_child_env())
    assert out.stdout.strip() == subprocess_digest()


def test_the_cross_process_digest_is_not_a_constant():
    """Non-vacuity for the test above: a digest that ignored its inputs
    would match across processes too."""
    assert subprocess_digest(seed=4242) != subprocess_digest(seed=4243)
    assert subprocess_digest(n_species=1) != subprocess_digest(n_species=2)
