"""Binary Lennard-Jones: the published model builds from its declaration and loads strictly.

The published model is a k-modes checkpoint, so it loads through ``load_kmodes_into``. Its forward
pass was compared with the published model's own training code, which computes the chemical
potential's pair term in real space where this package does it in k space; the residue is about
1e-6 of the drift. That comparison is migration evidence and not part of this suite. Here: the strict
load, both layouts of the checkpoint, and the declaration's own refusals.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from aipf.functional.build import build as build_functional
from aipf.system import Functional, Mobility, load
from aipf.train.checkpoint_formats import kmodes_model_state_dict, load_kmodes_into

import declared_roots

#: The system's declared value, restated so a silent change moves one side only.
_K_B = 1.0

#: The published model's grid and box, hparams["grid"] and hparams["box"].
_GRID = (20, 20, 80)
_BOX = (9.524406, 9.524406, 38.097625)
_N = _GRID[0] * _GRID[1] * _GRID[2]

@pytest.fixture(scope="module")
def published_model():
    declared_roots.published_or_skip("lj")
    return torch.load(load("lj").resolve_checkpoint(), map_location="cpu",
                      weights_only=False)


@pytest.fixture(scope="module")
def declared(published_model):
    return load_kmodes_into(build_functional(load("lj")), published_model).eval()


# --------------------------------------------------------------------------
# the load
# --------------------------------------------------------------------------

def test_the_declared_kb_is_reduced_units():
    assert load("lj").constants["kB"] == _K_B


def test_build_with_no_overrides_loads_the_published_model_strictly(published_model):
    model = build_functional(load("lj"))
    load_kmodes_into(model, published_model)   # strict=True inside; raises on mismatch
    assert model.build_overrides == {}


def test_both_layouts_of_the_checkpoint_load_to_the_same_tensors(published_model):
    a = load_kmodes_into(build_functional(load("lj")), published_model)
    b = load_kmodes_into(build_functional(load("lj")),
                           {"state_dict": published_model["state_dict"]})
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def test_the_strict_load_leaves_no_key_on_either_side(published_model):
    """Every saved tensor has a home and every other key is a grid-derived operator buffer."""
    model = build_functional(load("lj"))
    incoming = set(kmodes_model_state_dict(published_model))
    own = set(model.state_dict())
    assert incoming <= own
    taylor = {"f_local.u_net.exps", "f_local.u_net.rho_ref"}  # the declared powers and centre
    assert own - incoming == {k for k in own if k.startswith("ops.")} | taylor
    loaded = load_kmodes_into(model, published_model)
    assert loaded.f_local.u_net.exps.view(-1).tolist() == [2, 4, 6, 8]
    assert loaded.f_local.u_net.rho_ref.tolist() == [0.5]
    # no quadrature is declared, so no kappa buffer is built (a lattice sum needs none)
    assert "kernel.kappa_r_quad" not in own


def test_the_build_opens_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("AIPF_RAW_LJ", str(tmp_path / "absent"))
    assert build_functional(load("lj")).build_overrides == {}


# --------------------------------------------------------------------------
# the mobility restates the saved initialisation
# --------------------------------------------------------------------------

def test_the_mobility_declaration_restates_the_saved_initialisation(published_model):
    import math
    mk = published_model["hyper_parameters"]["model_kwargs"]
    mob = load("lj").mobility
    assert mk["m_form"] == "gamma" and mk["m_T_form"] == mob.T_form == "arrhenius"
    assert mob.form == "lattice_scalar"
    assert mob.kwargs["shape_init"] == math.log(mk["gamma"])
    assert mob.kwargs["mobility_activation_energy_init"] == math.log(math.exp(mk["Ea_init"]) - 1.0)
    assert mob.kwargs["mobility_prefactor"] == "mole_fraction"


# --------------------------------------------------------------------------
# declared, never defaulted: deleting any load-bearing line must refuse
# --------------------------------------------------------------------------

_MUST_REFUSE = (
    "grid", "R_cut", "rho_ref", "h_g", "h_w", "T_ref", "rho_eps",
    "activation", "g_form", "enable_TlnT", "enable_T2", "h_g_hat", "h_g_tilde",
    "gauge_fix", "f_exc_form", "ideal_form", "local_input_scale", "u_degree",
    "u_parity", "u_variable", "g_symmetry", "icnn_output_bias", "kernel_argument",
    "nyquist_mask", "mobility_prefactor", "shape_init", "mobility_activation_energy_init",
)

#: Deleting these refuses too, naming what the form they fall back to needs instead.
_REFUSED_NAMING_THE_FALLBACK = {
    "kernel_evaluator": "kernel_n_quad",  # absent = the k-table evaluator, which integrates
}

_HARMLESS_TO_OMIT = {
    "disable_g": "a switch of the training code with no constructor argument; absent and at its off value "
                 "mean the same model",
}


def _declared_keys(system):
    return set(system.functional.kwargs) | set(system.mobility.kwargs)


def _without(system, key):
    f = {k: v for k, v in system.functional.kwargs.items() if k != key}
    m = {k: v for k, v in system.mobility.kwargs.items() if k != key}
    return dataclasses.replace(
        system,
        functional=Functional(form=system.functional.form, local=system.functional.local,
                              kernel=system.functional.kernel, kwargs=f),
        mobility=Mobility(form=system.mobility.form, T_form=system.mobility.T_form, kwargs=m))


def test_every_declared_key_is_classified_one_way_or_another():
    assert (set(_MUST_REFUSE) | set(_REFUSED_NAMING_THE_FALLBACK)
            | set(_HARMLESS_TO_OMIT)) == _declared_keys(load("lj"))


@pytest.mark.parametrize("key", _MUST_REFUSE)
def test_deleting_a_declared_value_is_refused_by_name(key):
    with pytest.raises(ValueError, match=key):
        build_functional(_without(load("lj"), key))


@pytest.mark.parametrize("key", sorted(_REFUSED_NAMING_THE_FALLBACK))
def test_deleting_the_evaluator_is_refused_naming_what_the_fallback_needs(key):
    with pytest.raises(ValueError, match=_REFUSED_NAMING_THE_FALLBACK[key]):
        build_functional(_without(load("lj"), key))


@pytest.mark.parametrize("key", sorted(_HARMLESS_TO_OMIT))
def test_a_key_that_carries_nothing_may_be_omitted(key):
    build_functional(_without(load("lj"), key))


def test_no_quadrature_and_no_mobility_width_is_declared_because_nothing_reads_them():
    """The lattice sum integrates nothing and the scalar mobility has no net."""
    declared = _declared_keys(load("lj"))
    assert "kernel_n_quad" not in declared and "h_m" not in declared
    model = build_functional(load("lj"))
    with pytest.raises(ValueError, match="kernel_n_quad"):
        model.kernel.kappa_eff()


#: Undeclared because the Taylor energy with no T-basis head reads neither, with a variant that does.
_UNREAD = {"h_u": {"local": "mlp"},
           "fexc_T_ref": {"kwargs": {"enable_TlnT": True, "h_g_hat": 4}}}


@pytest.mark.parametrize("key", sorted(_UNREAD))
def test_a_value_nothing_reads_is_not_declared_and_a_form_that_reads_it_asks_for_it(key):
    system = load("lj")
    assert key not in _declared_keys(system)
    build_functional(system)
    change = _UNREAD[key]
    functional = dataclasses.replace(
        system.functional, local=change.get("local", system.functional.local),
        kwargs={**system.functional.kwargs, **change.get("kwargs", {})})
    with pytest.raises(ValueError, match=f"declare it as '{key}'" if key == "fexc_T_ref" else key):
        build_functional(dataclasses.replace(system, functional=functional))
