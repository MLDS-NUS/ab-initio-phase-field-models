"""The published model builds from its declaration: strict load AND a forward-pass match.

A strict load proves names and shapes. Two of this model's own derived
quantities -- ``kBT_ref`` and the T-basis orthogonalisation coefficients --
are NON-persistent buffers (``aipf/functional/local_forms.py:444-453``), so
``strict=True`` passes on a model whose reference energy is wrong by any
factor at all, and the sibling system's map measured the same shape costing
a factor of 10.4 in the mobility. Only a forward pass proves the model is
the same model. The acceptance is both, which is why the second test below
compares ``torch.equal`` and not a tolerance.

The comparison is against ``aipf.train.checkpoint_formats.lightning_hparams_rung_kwargs``
-- the measured bridge from the checkpoint's own saved ``model_kwargs`` to
this package's rung-3 constructor. The declaration must agree with it
key for key; where the two disagree the declaration or the translation is
wrong, never this test.
"""
import pytest
import torch

from aipf.functional.build import build as build_functional
from aipf.mobility import build as build_mobility
from aipf.system import load
from aipf.train.checkpoint_formats import lightning_hparams_rung_kwargs, load_lightning_hparams_into

import declared_roots

#: The bridge takes ``kB`` as an argument because core holds no system
#: constant. This is the system's own declared value, restated here so a
#: silent change to the declaration cannot silently move both sides.
_K_B = 8.617333262e-5


@pytest.fixture(scope="module")
def published_model():
    declared_roots.published_or_skip("hhe")
    return torch.load(load("hhe").resolve_checkpoint(), map_location="cpu",
                      weights_only=False)


def _bridged(published_model):
    from aipf.functional.nonlocal_kernel import NonlocalKernel

    state = published_model["state_dict"]
    kwargs = lightning_hparams_rung_kwargs(
        published_model["hyper_parameters"]["model_kwargs"], _K_B, state)
    return load_lightning_hparams_into(NonlocalKernel(**kwargs), state).eval()


def test_the_declared_kb_is_the_one_the_bridge_is_given():
    assert load("hhe").constants["kB"] == _K_B


def test_build_with_no_overrides_loads_the_published_model_strictly(published_model):
    model = build_functional(load("hhe"))
    load_lightning_hparams_into(model, published_model["state_dict"])   # raises on mismatch
    assert model.build_overrides == {}


def test_the_declaration_translates_to_the_bridges_kwargs_key_for_key(published_model):
    """The map, checked as a map rather than only through its effect.

    A forward pass over two channels exercises most of these, but not all:
    a wrong ``kernel_k_table_max`` moves no number this file's pinned state
    reads. Comparing the dicts catches those too, and names the key.
    """
    from aipf.functional.build import rung_kwargs

    declared = rung_kwargs(load("hhe"))
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
    state = published_model["state_dict"]
    declared = load_lightning_hparams_into(build_functional(load("hhe")), state).eval()
    bridged = _bridged(published_model)

    rho = torch.tensor([[0.42, 0.25]])
    T = torch.tensor([7000.0])
    with torch.no_grad():
        f_d = declared.f_local.f_pointwise(rho, _K_B * T)
        f_b = bridged.f_local.f_pointwise(rho, _K_B * T)
        m_d = declared._mobility(rho, T)
        m_b = bridged._mobility(rho, T)
    assert torch.equal(f_d, f_b), (f_d, f_b)
    assert torch.equal(m_d, m_b), (m_d, m_b)


def test_the_pinned_state_is_not_a_state_where_everything_is_zero():
    """A guard on the guard: two equal zeros would pass the test above."""
    declared = build_functional(load("hhe"))
    rho = torch.tensor([[0.42, 0.25]])
    T = torch.tensor([7000.0])
    with torch.no_grad():
        f = declared.f_local.f_pointwise(rho, _K_B * T)
        m = declared._mobility(rho, T)
    assert torch.isfinite(f).all() and float(f.abs().max()) > 0.0
    assert torch.isfinite(m).all() and float(m.abs().max()) > 0.0


def test_the_standalone_mobility_matches_the_one_inside_the_rung(published_model):
    """``aipf.mobility.build`` builds the rung's mobility, not another one.

    Loaded from the same checkpoint tensors and evaluated at the same
    state, so this compares the built OBJECT, not two random inits.
    """
    torch.set_num_threads(1)
    system = load("hhe")
    inside = load_lightning_hparams_into(build_functional(system),
                                published_model["state_dict"])._mobility.eval()
    alone = build_mobility(system).eval()
    alone.load_state_dict(inside.state_dict(), strict=True)

    rho = torch.tensor([[0.42, 0.25]])
    T = torch.tensor([7000.0])
    with torch.no_grad():
        assert torch.equal(alone(rho, T), inside(rho, T))


# --------------------------------------------------------------------------
# Declared, never defaulted: deleting any load-bearing line must refuse
#
# `R_cut` always refused, because the rung's constructor has no default for
# it. Every parameter that HAS a default was a hole of the opposite kind:
# drop `h_m` from the declaration and the model was built at the signature's
# `mobility_hidden=32` and would train, silently, as a model nobody
# declared. These tests delete each declared key IN A SYSTEM BUILT FOR THE
# TEST -- the experiment file is never edited -- and require a refusal that
# names it.
# --------------------------------------------------------------------------

#: Declared keys whose absence must be refused, and the text the refusal has
#: to contain. For a key the constructor spells differently the message
#: names the constructor argument AND this spelling, so matching on the
#: declared name is enough either way.
_MUST_REFUSE = (
    "grid", "R_cut", "rho_ref", "h_u", "h_g", "h_w", "h_m",
    "fexc_T_ref", "T_ref", "rho_eps", "activation", "g_form",
    "enable_TlnT", "enable_T2", "gauge_fix", "tbasis_ortho",
    "f_exc_form", "ideal_form", "h_g_hat", "h_g_tilde",
    "tbasis_ortho_window", "tbasis_ortho_points",
    "kernel_n_quad", "kernel_n_k_table", "kernel_k_table_max",
    "nyquist_mask",
    "mobility_prefactor", "local_input_scale", "mobility_input_ref",
)

#: Declared keys whose absence is genuinely harmless, each with the reason.
#: They are listed rather than left out, so that the partition test below
#: fails when a key is added to the declaration and classified nowhere --
#: which is how a load-bearing key would otherwise slip into this side by
#: being forgotten.
_HARMLESS_TO_OMIT = {
    "u_form": "a redundant copy of Functional.local, which is the "
              "authority; build already refuses the two disagreeing",
    "disable_u": "a switch of the training code with no constructor argument; absent and "
                 "at its off value mean the same model, and its on value "
                 "is refused",
    "disable_g": "as disable_u",
    "arrhenius_shared_Ea": "as disable_u, on the mobility's side",
}


def _declared(system):
    merged = dict(system.functional.kwargs)
    merged.update(system.mobility.kwargs)
    return merged


def _without(system, key):
    """The same system with one declared key deleted. Never touches disk."""
    import dataclasses

    from aipf.system import Functional, Mobility

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
    """The partition, asserted as one. A key added to the declaration and
    listed in neither table fails here rather than going unchecked."""
    declared = set(_declared(load("hhe")))
    assert set(_MUST_REFUSE) | set(_HARMLESS_TO_OMIT) == declared


@pytest.mark.parametrize("key", _MUST_REFUSE)
def test_deleting_a_declared_value_is_refused_by_name(key):
    with pytest.raises(ValueError, match=key):
        build_functional(_without(load("hhe"), key))


@pytest.mark.parametrize("key", sorted(_HARMLESS_TO_OMIT))
def test_a_key_that_carries_nothing_may_be_omitted(key):
    """The other side of the partition, so it is a claim and not a gap."""
    build_functional(_without(load("hhe"), key))


def test_the_complete_declaration_still_builds():
    """The inverse of every row above: nothing here refuses a good system."""
    model = build_functional(load("hhe"))
    assert model.build_overrides == {}


def test_a_system_that_declares_only_form_names_is_refused_not_defaulted():
    """A system that declares its forms and an EMPTY kwargs, as the one-field
    system once did, is refused: a builder cannot tell a value
    nobody read from a value read as absent. All three systems now declare
    themselves, so the empty declaration is made here from this one."""
    import dataclasses

    from aipf.system import Functional
    hhe = load("hhe")
    empty = dataclasses.replace(hhe, functional=Functional(
        form=hhe.functional.form, local=hhe.functional.local,
        kernel=hhe.functional.kernel, kwargs={}))
    with pytest.raises(ValueError, match="declaration supplies no value"):
        build_functional(empty)
