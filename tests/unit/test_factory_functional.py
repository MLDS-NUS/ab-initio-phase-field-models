"""A model this package does not define, declared by ``Functional(factory=...)``: built, trained, saved,
reloaded and rolled out through the same doors as a rung, and refused by name where it breaks the contract.

The model is ``tests/toy_factory_model.py``: a square gradient written as the pair kernel
``W(k) = kappa k^2`` on a polynomial local free energy, with a constant full mobility. Its system file
is written into a temporary experiments folder and imports it, as a system built on another package
does. The rungs' own behaviour is pinned bit for bit by ``tests/golden``; what is pinned here is that
the extension point adds a path beside them and changes none of theirs."""
import dataclasses
import hashlib
import json
import logging

import numpy as np
import pytest
import torch

import toy_factory_model as toy
from aipf.functional.build import build, rung_kwargs
from aipf.pipeline.modes import ModesRecord
from aipf.rollout.imex import rollout_imex
from aipf.solve import UNDECLARED, rollout_deterministic
from aipf.spectral import SpectralOps
from aipf.system import FUNCTIONAL_FORMS, Checkpoint, Functional, load
from aipf.train.datamodule import SourceSpec
from aipf.train.fit import _factory_record, _load_weights_into, fit

BOX = torch.tensor([8.0, 8.0, 8.0])
T = 1.0
IMEX = dict(kB=1.0, state_proj="floor", state_clamp=1e-3, mass_restore="shift", clamp_rho=1e-3)


@pytest.fixture(autouse=True)
def _demo_scratch_is_its_own(monkeypatch):
    """The demo declares its own raw root; a user's ``AIPF_RAW`` must not replace it."""
    monkeypatch.delenv("AIPF_RAW", raising=False)


def _system():
    return toy.demo_system("unused")


def _field(low=None):
    """One field on the toy's grid, ``rho_hat`` ``(1, 2, 4, 4, 3)``; ``low`` puts channel 0 of one cell there."""
    g = torch.Generator().manual_seed(3)
    rho = torch.stack([0.40 + 0.05 * torch.randn(toy.GRID, generator=g),
                       0.30 + 0.05 * torch.randn(toy.GRID, generator=g)]).unsqueeze(0)
    if low is not None:
        rho[0, 0, 1, 2, 3] = low
    return SpectralOps(toy.GRID, 2, nyquist_mask=True).rfft(rho) / rho[0, 0].numel()


def _pair(scale=1.0):
    """The toy without the stabiliser hook, and the same weights with it at ``scale``."""
    torch.manual_seed(0)
    plain = build(_system())
    with torch.no_grad():
        for p in plain.parameters():
            p.add_(0.3 * torch.randn_like(p))
    hooked = build(_system(), variant="stabilized", stabilizer_scale=scale)
    hooked.load_state_dict(plain.state_dict(), strict=True)
    return plain, hooked


# ---------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["lj", "hhe", "feb"])
def test_a_declared_functional_reads_and_compares_as_it_did_before_the_factory_field(name):
    system = load(name)
    for f in [system.functional] + [v.functional for v in system.variants.values()]:
        assert repr(f) == (f"Functional(form={f.form!r}, local={f.local!r}, "
                           f"kernel={f.kernel!r}, kwargs={f.kwargs!r})")
        assert f.factory is None
        with_factory = Functional(form=f.form, local=f.local, kernel=f.kernel,
                                  kwargs={**f.kwargs, "grid": (4, 4, 4), "nyquist_mask": True},
                                  factory=toy.build_toy)
        assert repr(with_factory).startswith("Functional(form=") and "factory" not in repr(with_factory)
        same = Functional(form=f.form, local=f.local, kernel=f.kernel, kwargs=f.kwargs)
        assert same == f


def test_the_factory_is_left_out_of_equality():
    assert toy.toy_functional(toy.build_toy) == toy.toy_functional(toy.build_stabilized_toy)


def test_a_factory_functional_names_any_form_and_the_forms_stay_four():
    assert FUNCTIONAL_FORMS == ("landau", "square_gradient", "nonlocal_kernel", "neural_operator")
    assert toy.toy_functional().form not in FUNCTIONAL_FORMS
    with pytest.raises(ValueError, match="unknown functional form"):
        Functional(form="toy_square_gradient", local="polynomial", kernel=None, kwargs={})


def test_a_factory_functional_declares_its_grid_and_nyquist_convention():
    with pytest.raises(ValueError, match="nyquist_mask"):
        Functional(form="toy", local="toy", kernel=None, kwargs={"grid": (4, 4, 4)},
                   factory=toy.build_toy)
    with pytest.raises(TypeError, match="not a callable"):
        Functional(form="toy", local="toy", kernel=None,
                   kwargs={"grid": (4, 4, 4), "nyquist_mask": True}, factory="toy:build")


def test_a_variant_carries_its_own_factory():
    system = _system()
    assert system.variant("stabilized").functional.factory is toy.build_stabilized_toy
    assert type(build(system, variant="stabilized")) is toy.StabilizedToyFactoryModel
    assert type(build(system)) is toy.ToyFactoryModel


def test_build_calls_the_factory_with_the_system_and_the_overrides():
    seen = []

    def factory(system, **overrides):
        seen.append((system.name, overrides))
        return toy.build_toy(system, **overrides)

    system = dataclasses.replace(_system(), functional=toy.toy_functional(factory))
    model = build(system, kappa=0.25)
    assert seen == [("demo", {"kappa": 0.25})]
    assert model.build_overrides == {"kappa": 0.25}
    assert torch.allclose(torch.nn.functional.softplus(model.kernel.kappa_raw),
                          torch.tensor([0.25, 0.25]))


def test_rung_kwargs_refuses_a_factory_functional_by_name():
    with pytest.raises(ValueError, match="toy_factory_model:build_toy.*build\\(\\)"):
        rung_kwargs(_system())


# ---------------------------------------------------------------------------
# what a factory may return
# ---------------------------------------------------------------------------

class _NoBulk(toy.ToyFactoryModel):
    bulk_free_energy_density = None


def _without(attribute):
    def factory(system, **overrides):
        model = toy.build_toy(system, **overrides)
        delattr(model, attribute)
        return model
    return factory


def _no_parameters(system, **overrides):
    model = toy.build_toy(system, **overrides)
    for name in [n for n, _ in model.named_parameters()]:
        module_name, _, leaf = name.rpartition(".")
        owner = model.get_submodule(module_name) if module_name else model
        delattr(owner, leaf)
        setattr(owner, leaf, torch.zeros(1))
    return model


@pytest.mark.parametrize("factory,kind,match", [
    (lambda system: {"model": "not one"}, TypeError, "returned a dict, not a torch.nn.Module"),
    (lambda system: _NoBulk(toy.GRID, 2, nyquist_mask=True, kappa=0.5, kB=1.0),
     TypeError, "missing \\['bulk_free_energy_density'\\]"),
    (_no_parameters, ValueError, "with no parameters"),
    (_without("_cache"), TypeError, "_cache is not an aipf.spectral.OpsCache"),
    (_without("ops"), TypeError, "ops is not an aipf.spectral.SpectralOps"),
    (lambda system: toy.build_toy(system, nyquist_mask=False), ValueError,
     "nyquist_mask=False and the declaration says True"),
])
def test_a_factory_that_breaks_the_contract_is_refused_by_name(factory, kind, match):
    system = dataclasses.replace(_system(), functional=toy.toy_functional(factory))
    with pytest.raises(kind, match=match):
        build(system)


# ---------------------------------------------------------------------------
# trained, saved, reloaded and rolled out
# ---------------------------------------------------------------------------

def _write_modes(directory, rng) -> None:
    """One toy run's ``modes.npz``, through the writer a real one uses (as ``test_train_fit.py``)."""
    labels = np.array([(nx, ny, nz) for nx in (-1, 0, 1) for ny in (-1, 0, 1)
                       for nz in (0, 1, 2)], dtype=np.int64)
    shape = (24, labels.shape[0], 2)
    amplitudes = (rng.standard_normal(shape)
                  + 1j * rng.standard_normal(shape)).astype(np.complex64)
    ModesRecord(amplitudes=amplitudes, labels=labels, boxes=np.full((24, 3), 8.0),
                T=1000.0, dt_frame=0.02,
                provenance={"composition": {"x_B": 0.5}, "cache_dir": str(directory)}
                ).save(directory)


@pytest.fixture
def declared_demo(tmp_path, monkeypatch):
    """``load("demo")`` from a system file that imports its factory, and one source of modes."""
    folder = tmp_path / "experiments" / "demo"
    folder.mkdir(parents=True)
    (folder / "system.py").write_text(
        "from toy_factory_model import demo_system\n"
        f"SYSTEM = demo_system({str(tmp_path)!r})\n")
    monkeypatch.setenv("AIPF_EXPERIMENTS", str(tmp_path / "experiments"))
    tree = tmp_path / "fields" / "modes_demo"
    _write_modes(tree / "run_a", np.random.default_rng(0))
    sources = (SourceSpec(name="demo_src", root=str(tree), pattern="run_*",
                          grid=toy.GRID, loss_weight=1.0, exclude_tags=()),)
    return load("demo"), sources


def test_a_factory_model_trains_saves_reloads_strictly_and_rolls_out(declared_demo, tmp_path):
    """Reloaded as the driver reloads a starting point: rebuilt from the declaration, then the tagged
    checkpoint's ``model_state_dict`` loaded strictly. The checkpoint stores no class."""
    system, sources = declared_demo
    assert system.functional.factory.__module__ == "toy_factory_model"
    run = fit(system, run_name="smoke", sources=sources, steps=2, seed=0,
              resume_optimizer=False, root=tmp_path / "data", device="cpu")
    manifest = json.loads((run / "MANIFEST.json").read_text())
    assert manifest["model_factory"] == "toy_factory_model:build_toy"
    assert manifest["global_step"] == 2

    saved = torch.load(run / "final.ckpt", map_location="cpu", weights_only=False)["model_state_dict"]
    ckpt = Checkpoint(md5=hashlib.md5((run / "final.ckpt").read_bytes()).hexdigest(),
                      path="data/demo/ckpt/smoke/final.ckpt")
    model = build(system)
    fresh = {k: v.clone() for k, v in model.state_dict().items()}
    _load_weights_into(model, ckpt, system.paths.raw, system)
    state = model.state_dict()
    assert type(model) is toy.ToyFactoryModel and set(state) == set(saved)
    assert all(torch.equal(state[k], saved[k]) for k in state)
    assert any(not torch.equal(state[k], fresh[k]) for k in state), "two steps changed nothing"
    with pytest.raises(RuntimeError, match="Missing key"):
        build(system).load_state_dict({k: v for k, v in saved.items() if k != "kernel.kappa_raw"})

    # the same checkpoint as a starting point with its optimizer resumed: built by the factory twice more
    again = fit(system, run_name="again", sources=sources, steps=1, seed=0,
                resume_optimizer=True, init_from=ckpt, root=tmp_path / "data", device="cpu")
    assert json.loads((again / "MANIFEST.json").read_text())["resume_optimizer"] is True

    h0 = _field()
    imex = rollout_imex(model, h0, BOX, T, 1e-2, 4, m_stab="mean", **IMEX)
    explicit = rollout_deterministic(model, h0, BOX.unsqueeze(0), torch.tensor([T]), 1e-3, 4,
                                     method="heun", state_proj="floor", floor=1e-3,
                                     clamp_rho=1e-3)
    for traj, saved_states in ((imex, 5), (explicit[:, 0], 5)):
        assert traj.shape == (saved_states, 2, 4, 4, 3)
        assert torch.isfinite(traj.real).all()
        assert torch.allclose(traj[:, :, 0, 0, 0].real, h0[0, :, 0, 0, 0].real.expand(5, 2))


def test_the_manifest_of_a_rung_has_no_factory_entry():
    for name in ("lj", "hhe", "feb"):
        assert _factory_record(load(name)) == {}


def test_diagnose_refuses_a_factory_model_before_writing_anything(tmp_path):
    from aipf.diagnose.run import check_diagnosable, run

    with pytest.raises(ValueError, match="toy_factory_model:build_toy"):
        check_diagnosable(_system())
    with pytest.raises(ValueError, match="diagnosed by its own code"):
        run(_system(), tmp_path / "missing.ckpt", stages=("kappa",), out=tmp_path / "out")
    assert not (tmp_path / "out").exists()
    check_diagnosable(load("lj"))


# ---------------------------------------------------------------------------
# the semi-implicit scheme's stabiliser hook
# ---------------------------------------------------------------------------

def test_the_hook_sets_M_s_once_from_the_initial_field(caplog):
    plain, hooked = _pair(scale=4.0)
    h0 = _field()
    calls = []
    hook = hooked.stabilizer_mobility

    def spy(rho, T_t):
        calls.append((rho.clone(), T_t.clone()))
        return hook(rho, T_t)

    hooked.stabilizer_mobility = spy
    with caplog.at_level(logging.INFO, logger="aipf.rollout.imex"):
        got = rollout_imex(hooked, h0, BOX, T, 1e-2, 4, **IMEX)
    assert len(calls) == 1
    rho, T_t = calls[0]
    assert rho.shape == (1, 2, *toy.GRID) and T_t.shape == (1,) and float(T_t) == T
    N = rho[0, 0].numel()
    assert torch.equal(rho, SpectralOps(toy.GRID, 2, nyquist_mask=True).irfft(h0 * N))
    assert "M_s from StabilizedToyFactoryModel.stabilizer_mobility" in caplog.text
    mean = rollout_imex(plain, h0, BOX, T, 1e-2, 4, m_stab="mean", **IMEX)
    assert not torch.equal(got, mean)


def test_a_hook_returning_the_mean_mobility_is_the_mean_scheme_bit_for_bit():
    """The toy's mobility is constant, so ``M`` at the mean density is ``M`` anywhere."""
    plain, hooked = _pair(scale=1.0)
    h0 = _field()
    assert torch.equal(rollout_imex(hooked, h0, BOX, T, 1e-2, 4, **IMEX),
                       rollout_imex(plain, h0, BOX, T, 1e-2, 4, m_stab="mean", **IMEX))


@pytest.mark.parametrize("m_stab", ["mean", "max"])
def test_without_the_hook_m_stab_decides_as_before(caplog, m_stab):
    plain, _ = _pair()
    h0 = _field()
    with caplog.at_level(logging.INFO, logger="aipf.rollout.imex"):
        want = rollout_imex(plain, h0, BOX, T, 1e-2, 4, m_stab=m_stab, **IMEX)
    assert f"M_s from m_stab='{m_stab}'" in caplog.text
    plain.stabilizer_mobility = None
    assert torch.equal(rollout_imex(plain, h0, BOX, T, 1e-2, 4, m_stab=m_stab, **IMEX), want)
    with pytest.raises(ValueError, match="m_stab must be declared"):
        rollout_imex(plain, h0, BOX, T, 1e-2, 1, m_stab=UNDECLARED, **IMEX)


@pytest.mark.parametrize("m_stab", ["mean", "max"])
def test_a_declared_m_stab_beside_the_hook_is_refused(m_stab):
    _, hooked = _pair()
    with pytest.raises(ValueError, match="has a stabilizer_mobility, which sets M_s itself"):
        rollout_imex(hooked, _field(), BOX, T, 1e-2, 1, m_stab=m_stab, **IMEX)


@pytest.mark.parametrize("returned,match", [
    (torch.eye(3), "returned \\(3, 3\\), not an \\(2, 2\\) tensor"),
    (torch.ones(1, 2, 2), "returned \\(1, 2, 2\\)"),
    ([[1.0, 0.0], [0.0, 1.0]], "returned list"),
    (torch.tensor([[1.0, 0.5], [0.0, 1.0]]), "non-symmetric"),
    (torch.tensor([[1.0, 2.0], [2.0, 1.0]]), "eigenvalue -"),
    (torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]]), "non-finite"),
])
def test_a_hook_returning_no_admissible_M_s_is_refused(returned, match):
    plain, _ = _pair()
    plain.stabilizer_mobility = lambda rho, T_t: returned
    with pytest.raises(ValueError, match=match):
        rollout_imex(plain, _field(), BOX, T, 1e-2, 1, **IMEX)


def test_a_hook_within_the_tolerance_is_taken():
    plain, _ = _pair()
    M = torch.tensor([[2.0, 0.5], [0.5 + 1e-7, 1.0]])
    plain.stabilizer_mobility = lambda rho, T_t: M
    assert torch.isfinite(rollout_imex(plain, _field(), BOX, T, 1e-2, 2, **IMEX).real).all()
    plain.stabilizer_mobility = lambda rho, T_t: torch.zeros(2, 2)
    assert torch.isfinite(rollout_imex(plain, _field(), BOX, T, 1e-2, 2, **IMEX).real).all()


# ---------------------------------------------------------------------------
# clamp_rho on a model that follows the contract
# ---------------------------------------------------------------------------

def _imex(model, h0, clamp_rho):
    return rollout_imex(model, h0, BOX, T, 1e-2, 3, m_stab="mean",
                        **{**IMEX, "clamp_rho": clamp_rho})


def _explicit(model, h0, clamp_rho):
    return rollout_deterministic(model, h0, BOX.unsqueeze(0), torch.tensor([T]), 1e-3, 3,
                                 method="euler", state_proj="floor", floor=-1.0,
                                 clamp_rho=clamp_rho)


@pytest.mark.parametrize("roll", [_imex, _explicit])
def test_clamp_rho_takes_effect_on_a_contract_following_model(roll):
    plain, _ = _pair()
    h0 = _field(low=0.02)
    unguarded = roll(plain, h0, None)
    assert torch.equal(roll(plain, h0, 0.01), unguarded)       # below every density: no cell moves
    assert not torch.equal(roll(plain, h0, 0.1), unguarded)     # above one cell: its mu is read at 0.1
    assert "mu_pointwise" not in plain.f_local.__dict__ and "mobility" not in plain.__dict__


def test_the_clamp_reaches_the_model_through_its_pointwise_mu():
    plain, _ = _pair()
    seen = []
    mu = plain.f_local.mu_pointwise

    def spy(rho, kBT):
        seen.append(float(rho.min()))
        return mu(rho, kBT)

    plain.f_local.mu_pointwise = spy
    _imex(plain, _field(low=0.02), 0.1)
    assert seen and min(seen) >= 0.1
