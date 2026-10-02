"""Iron-boron: the published model builds from its declaration, strictly, and computes the same function.

The structure mirrors ``test_hhe_build_matches_published.py``. A strict load
proves names and shapes only. This published model carries three quantities in
NON-persistent buffers -- the excess nets' input scale, the mobility net's
input scale and the reference energy -- so ``strict=True`` passes on a model
that computes another function; the two input scales alone were measured at a
factor of 10.4 in the mobility. The acceptance is therefore a ``torch.equal``
forward pass against the bridge from the checkpoint's own saved settings, and
two numbers measured on the published model's own training code.

The training code's constructor OPENED ``model_kwargs["m_table"]`` to seed the
mobility's last bias (``m_init_from_table``), which the state dict then
overwrites. This package's build opens no file.
"""
import dataclasses

import pytest
import torch

from aipf.functional.build import build as build_functional
from aipf.functional.build import rung_kwargs
from aipf.mobility import build as build_mobility
from aipf.system import Functional, Mobility, load
from aipf.train.checkpoint_formats import lightning_hparams_rung_kwargs, load_lightning_hparams_into

import declared_roots

#: The system's declared value, restated so a silent change moves one side only.
_K_B = 8.617333262e-5

#: The table directory the published model's ``m_table`` paths sit in.
_DERIVED = "training_derived"


@pytest.fixture(scope="module")
def published_model():
    declared_roots.published_or_skip("feb")
    return torch.load(load("feb").resolve_checkpoint(), map_location="cpu",
                      weights_only=False)


def _bridged(published_model):
    from aipf.functional.nonlocal_kernel import NonlocalKernel

    state = published_model["state_dict"]
    kwargs = lightning_hparams_rung_kwargs(
        published_model["hyper_parameters"]["model_kwargs"], _K_B, state)
    return load_lightning_hparams_into(NonlocalKernel(**kwargs), state).eval()


def _declared_model(published_model):
    return load_lightning_hparams_into(build_functional(load("feb")),
                              published_model["state_dict"]).eval()


def test_the_declared_kb_is_the_one_the_bridge_is_given():
    assert load("feb").constants["kB"] == _K_B


def test_build_with_no_overrides_loads_the_published_model_strictly(published_model):
    model = build_functional(load("feb"))
    load_lightning_hparams_into(model, published_model["state_dict"])   # raises on mismatch
    assert model.build_overrides == {}


def test_the_strict_load_leaves_no_key_on_either_side(published_model):
    """Every saved tensor has a home and every other key is a grid-derived buffer."""
    from aipf.train.checkpoint_formats import lightning_hparams_model_state_dict

    model = build_functional(load("feb"))
    incoming = set(lightning_hparams_model_state_dict(published_model["state_dict"]))
    own = set(model.state_dict())
    assert incoming <= own
    derived = own - incoming
    assert derived == {k for k in own if k.startswith("ops.")} | {
        "kernel.kappa_r_quad"}
    loaded = load_lightning_hparams_into(model, published_model["state_dict"])
    assert torch.equal(loaded.kernel.kappa_r_quad,
                       loaded.kernel.evaluator.r_quad)


def test_the_declaration_translates_to_the_bridges_kwargs_key_for_key(
        published_model):
    declared = rung_kwargs(load("feb"))
    bridged = lightning_hparams_rung_kwargs(
        published_model["hyper_parameters"]["model_kwargs"], _K_B,
        published_model["state_dict"])

    def _norm(d):
        return {k: (tuple(v) if isinstance(v, (list, tuple)) else v)
                for k, v in d.items()}

    assert _norm(declared) == _norm(bridged)


def test_the_built_model_gives_the_bridge_value_at_a_pinned_state(
        published_model):
    torch.set_num_threads(1)
    declared = _declared_model(published_model)
    bridged = _bridged(published_model)
    rho = torch.tensor([[0.12, 0.075]])
    T = torch.tensor([1800.0])
    with torch.no_grad():
        f_d = declared.f_local.f_pointwise(rho, _K_B * T)
        f_b = bridged.f_local.f_pointwise(rho, _K_B * T)
        m_d = declared._mobility(rho, T)
        m_b = bridged._mobility(rho, T)
    assert torch.equal(f_d, f_b), (f_d, f_b)
    assert torch.equal(m_d, m_b), (m_d, m_b)


def test_two_numbers_measured_on_the_training_code(published_model):
    """``f_pointwise`` 0.263353 and ``M[0,0]`` 0.06647 at (0.12, 0.075), 1800 K."""
    model = _declared_model(published_model)
    rho = torch.tensor([[0.12, 0.075]])
    T = torch.tensor([1800.0])
    with torch.no_grad():
        f = float(model.f_local.f_pointwise(rho, _K_B * T))
        m = float(model._mobility(rho, T)[0, 0, 0])
    assert f == pytest.approx(0.263353, abs=5e-7)
    assert m == pytest.approx(0.06647, abs=5e-6)


def test_the_standalone_mobility_matches_the_one_inside_the_rung(published_model):
    torch.set_num_threads(1)
    system = load("feb")
    inside = _declared_model(published_model)._mobility
    alone = build_mobility(system).eval()
    alone.load_state_dict(inside.state_dict(), strict=True)
    rho = torch.tensor([[0.06, 0.05], [0.11, 0.02]])
    T = torch.tensor([1400.0, 2400.0])
    with torch.no_grad():
        assert torch.equal(alone(rho, T), inside(rho, T))


def test_the_build_opens_no_file(monkeypatch, tmp_path):
    """The training code's constructor read the mobility table; this package's build reads nothing."""
    monkeypatch.setenv("AIPF_RAW_FEB", str(tmp_path / "absent"))
    assert build_functional(load("feb")).build_overrides == {}
    declared = load("feb")
    assert "m_table" not in declared.functional.kwargs
    assert "m_table" not in declared.mobility.kwargs


def test_every_declared_model_value_has_a_source_and_matches_it(published_model):
    """Each declared knob is a saved ``model_kwargs`` value or what the bridge derives from one."""
    mk = published_model["hyper_parameters"]["model_kwargs"]
    bridged = lightning_hparams_rung_kwargs(mk, _K_B, published_model["state_dict"])
    renamed = {"h_joint": "h_u_joint", "joint_depth": "n_hidden_joint",
               "local_input_scale": "fexc_input_scale"}

    def _same(a, b):
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return tuple(a) == tuple(b)
        return a == b

    system = load("feb")
    declared = dict(system.functional.kwargs)
    declared.update(system.mobility.kwargs)
    sourced = {"saved": [], "bridged": []}
    for key, value in sorted(declared.items()):
        if key == "tbasis_ortho_window":            # saved in K, declared in eV
            assert _same(value, bridged[key])
            assert _same(value, tuple(_K_B * t for t in mk[key]))
            sourced["bridged"].append(key)
        elif key in mk:
            assert _same(value, mk[key]), key
            sourced["saved"].append(key)
        elif key in bridged:
            assert _same(value, bridged[key]), key
            sourced["bridged"].append(key)
            if key in renamed:
                assert _same(value, mk[renamed[key]]), key
        else:
            raise AssertionError(f"declared {key}={value!r} has no source")
    assert sourced["saved"] and sourced["bridged"], sourced
    assert system.functional.kwargs["f_exc_form"] == mk["f_exc_form"]
    assert system.mobility.form == mk["m_form"]
    assert system.mobility.T_form == mk["m_T_form"]


# --------------------------------------------------------------------------
# Declared, never defaulted: deleting any load-bearing line must refuse
# --------------------------------------------------------------------------

_MUST_REFUSE = (
    "grid", "R_cut", "rho_ref", "h_g", "h_w", "h_m",
    "fexc_T_ref", "T_ref", "rho_eps", "activation", "g_form",
    "enable_TlnT", "enable_T2", "gauge_fix", "tbasis_ortho",
    "f_exc_form", "ideal_form", "h_g_hat", "h_g_tilde",
    "tbasis_ortho_window", "tbasis_ortho_points",
    "kernel_n_quad", "kernel_n_k_table", "kernel_k_table_max",
    "nyquist_mask",
    "h_joint", "joint_depth", "local_input_scale",
    "mobility_prefactor", "mobility_input_ref",
    "mobility_activation_energy_init",
)

_HARMLESS_TO_OMIT = {
    "disable_u": "a switch of the training code with no constructor argument; absent and "
                 "at its off value mean the same model",
    "disable_g": "as disable_u",
    "arrhenius_shared_Ea": "as disable_u, on the mobility's side",
    "h_u": "the split form's energy-net width (mk[\"h_u\"]); the joint form builds no such "
           "net, so the door asks for it only under f_exc_form='split'",
}


def _declared(system):
    merged = dict(system.functional.kwargs)
    merged.update(system.mobility.kwargs)
    return merged


def _without(system, key):
    f = {k: v for k, v in system.functional.kwargs.items() if k != key}
    m = {k: v for k, v in system.mobility.kwargs.items() if k != key}
    return dataclasses.replace(
        system,
        functional=Functional(form=system.functional.form,
                              local=system.functional.local,
                              kernel=system.functional.kernel, kwargs=f),
        mobility=Mobility(form=system.mobility.form,
                          T_form=system.mobility.T_form, kwargs=m))


def test_every_declared_key_is_classified_one_way_or_the_other():
    assert set(_MUST_REFUSE) | set(_HARMLESS_TO_OMIT) == set(
        _declared(load("feb")))


@pytest.mark.parametrize("key", _MUST_REFUSE)
def test_deleting_a_declared_value_is_refused_by_name(key):
    with pytest.raises(ValueError, match=key):
        build_functional(_without(load("feb"), key))


@pytest.mark.parametrize("key", sorted(_HARMLESS_TO_OMIT))
def test_a_key_that_carries_nothing_may_be_omitted(key):
    build_functional(_without(load("feb"), key))
