"""``aipf.functional.build`` and ``aipf.mobility.build``: the front door.

These tests use a toy system that names nothing real, so
that what they pin is the TRANSLATION -- declared names in, constructor
names out -- and not any one system's numbers. The published model's own numbers
are pinned in ``tests/unit/test_hhe_build_matches_published.py``,
which is the only place a forward pass can be compared against a measured
model.

The toy is built the way every other test module in this directory builds
one: a local ``_demo``-shaped factory with the four fields ``System``
requires, because there is no shared fixture and inventing one here would
make a third convention.
"""
import dataclasses

import pytest
import torch

from aipf.functional.build import build, rung_kwargs
from aipf.paths import Paths
from aipf.system import AnchorRules, Functional, Mobility, System

#: The toy's Boltzmann constant. A round 1.0, because the toy works in
#: reduced units and a system's ``kB`` is that system's own unit
#: convention -- exactly why it is a declared constant and not a core
#: default.
_TOY_KB = 1.0


def _toy_system(_mobility_T_form="none", **functional_kwargs) -> System:
    """A two-channel nonlocal-kernel system; names nothing real."""
    kwargs = dict(
        grid=(4, 4, 4), R_cut=2.0, rho_ref=(0.3, 0.3),
        h_u=4, h_g=4, h_w=4, h_m=4,
        fexc_T_ref=1.0, rho_eps=1e-5, activation="gelu",
        g_form="mlp", ideal_form="gas", f_exc_form="split", nyquist_mask=True,
        gauge_fix=False, enable_TlnT=False, enable_T2=False,
        tbasis_ortho=False,
        kernel_n_quad=16, kernel_n_k_table=17, kernel_k_table_max=8.0,
        # Declared although this toy's variants do not read them, because
        # `build` refuses a parameter the declaration is silent about
        # rather than taking the constructor's default for it. That refusal
        # is the point of `test_every_declared_value_of_the_published_model_is_
        # required` below, and a toy exempt from it would be a toy testing
        # something the real door does not do.
        h_g_hat=4, h_g_tilde=4, T_ref=1000.0, local_input_scale=False,
        icnn_output_bias=True,
    )
    kwargs.update(functional_kwargs)
    return System(
        name="demo", n_species=2, species=("A", "B"),
        masses={"A": 1.0, "B": 2.0}, atom_types={"A": 1, "B": 2},
        table_keys={}, paths=Paths(system="demo"),
        anchor_rules=AnchorRules({}), constants={"kB": _TOY_KB}, defaults={},
        functional=Functional(form="nonlocal_kernel", local="mlp",
                              kernel="radial_mlp", kwargs=kwargs),
        mobility=Mobility(form="mlp_scaled", T_form=_mobility_T_form,
                          kwargs=dict(mobility_prefactor="mole_fraction",
                                      mobility_input_ref=None)),
    )


def _toy_arrhenius_system(**functional_kwargs) -> System:
    """The same toy with a temperature factor on the mobility.

    The reference temperature the mobility carries is only STORED when the
    factor is on (``aipf/mobility.py:220-228``), so this is the only shape
    in which a ``T_ref`` override can be observed on the built model rather
    than only in the constructor arguments.
    """
    kwargs = dict(T_ref=1000.0,
                  mobility_activation_energy_init=(0.1, 0.1))
    kwargs.update(functional_kwargs)
    return _toy_system("arrhenius", **kwargs)


def _without(system: System, *keys: str) -> System:
    """The same system with those functional kwargs deleted."""
    kwargs = {k: v for k, v in system.functional.kwargs.items()
              if k not in keys}
    return dataclasses.replace(
        system, functional=Functional(form=system.functional.form,
                                      local=system.functional.local,
                                      kernel=system.functional.kernel,
                                      kwargs=kwargs))


# --------------------------------------------------------------------------
# What build returns
# --------------------------------------------------------------------------
def test_build_returns_a_module_of_the_declared_form():
    model = build(_toy_system())
    assert isinstance(model, torch.nn.Module)
    assert type(model).__name__ == "NonlocalKernel"


def test_the_declared_forms_reach_the_constructor_under_its_own_names():
    """``local`` and the mobility's two axes are not decoration."""
    model = build(_toy_system())
    assert model.f_local.u_form == "mlp"
    assert model._mobility.prefactor_form == "mole_fraction"
    assert model._mobility.shape_form == "mlp_rho"
    assert model._mobility.t_form == "none"


def test_the_declared_kb_scales_the_reference_energy():
    """``kBT_ref`` is ``kB * fexc_T_ref``, from the system's own ``kB``."""
    model = build(_toy_system(fexc_T_ref=3.0))
    assert float(model.f_local.kBT_ref) == pytest.approx(_TOY_KB * 3.0)
    assert model.kB == _TOY_KB


# --------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------
def test_an_override_wins_over_the_declaration_and_is_recorded():
    model = build(_toy_system(), h_u=8)
    # The width is read off the built net, not off a stored attribute:
    # `FLocal` keeps no `h_u`, so a recorded-but-unused override would pass
    # an attribute check and fail this one.
    assert model.f_local.u_net.net.net[0].out_features == 8
    assert model.build_overrides == {"h_u": 8}


def test_no_override_records_an_empty_dict_not_a_missing_attribute():
    assert build(_toy_system()).build_overrides == {}


def test_an_unknown_override_is_refused_not_ignored():
    with pytest.raises(TypeError, match="h_uu"):
        build(_toy_system(), h_uu=8)


def test_an_override_may_be_spelled_the_declared_way_too():
    """``h_w`` is the declared name; ``kernel_hidden`` the constructor's.

    Both reach the same argument, because an override that had to be
    spelled the constructor's way would make a caller read the source --
    the thing this front door exists to avoid.
    """
    assert build(_toy_system(), h_w=6).kernel.radial_set.nets[0][0].out_features == 6
    assert build(_toy_system(),
                 kernel_hidden=6).kernel.radial_set.nets[0][0].out_features == 6


# --------------------------------------------------------------------------
# build supplies nothing of its own
# --------------------------------------------------------------------------
def test_build_carries_no_default_of_its_own():
    """Every kwarg reaching the constructor came from the declaration or an
    override. Delete one from the declaration and build must refuse.

    ``R_cut`` is the easy case -- the constructor has no default for it, so
    it refused even before the door checked. It is asserted as a
    ``ValueError`` here because the door now refuses FIRST, by name, rather
    than letting Python's own missing-argument ``TypeError`` come out of a
    constructor the caller never called.
    """
    with pytest.raises(ValueError, match="R_cut"):
        build(_without(_toy_system(), "R_cut"))


def test_a_missing_kernel_quadrature_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="kernel_n_quad"):
        build(_without(_toy_system(), "kernel_n_quad"))


def test_a_parameter_the_constructor_has_a_default_for_is_refused_too():
    """The hard case, and the one that was open.

    ``mobility_hidden`` defaults to 32 on the rung's constructor, so a
    declaration that lost its ``h_m`` line used to build a 32-wide mobility
    and train it without a word. The width the constructor would have
    chosen is asserted as well, so this cannot pass by the value happening
    to agree.
    """
    import inspect

    from aipf.functional.nonlocal_kernel import NonlocalKernel

    default = inspect.signature(
        NonlocalKernel.__init__).parameters["mobility_hidden"].default
    assert default == 32 and default != _toy_system().functional.kwargs["h_m"]
    with pytest.raises(ValueError, match="h_m"):
        build(_without(_toy_system(), "h_m"))


def test_the_t_basis_switch_may_not_be_left_implicit():
    """A window with no switch would have the window silently dropped."""
    with pytest.raises(ValueError, match="tbasis_ortho"):
        build(_without(_toy_system(tbasis_ortho=True,
                                   tbasis_ortho_window=(1.0, 2.0),
                                   tbasis_ortho_points=13), "tbasis_ortho"))


def test_a_system_that_declares_no_functional_is_refused_by_name():
    system = dataclasses.replace(_toy_system(), functional=None)
    with pytest.raises(ValueError, match="functional"):
        build(system)


def test_a_system_that_declares_no_mobility_is_refused_by_name():
    system = dataclasses.replace(_toy_system(), mobility=None)
    with pytest.raises(ValueError, match="mobility"):
        build(system)


def test_a_system_with_no_kb_is_refused_rather_than_given_one():
    system = dataclasses.replace(_toy_system(), constants={})
    with pytest.raises(KeyError, match="kB"):
        build(system)


# --------------------------------------------------------------------------
# Keys the constructor does not accept
# --------------------------------------------------------------------------
def test_a_flag_the_constructor_has_no_argument_for_is_dropped_with_a_reason():
    """``disable_u``/``disable_g`` are switches of the training code this package has no
    argument for. At their off value they are dropped; at their on value
    they describe a model this package does not build, so they raise."""
    assert build(_toy_system(disable_u=False, disable_g=False)) is not None
    with pytest.raises(NotImplementedError, match="disable_u"):
        build(_toy_system(disable_u=True))
    with pytest.raises(NotImplementedError, match="disable_g"):
        build(_toy_system(disable_g=True))


def test_the_t_basis_window_is_required_when_the_orthogonalisation_is_on():
    with pytest.raises(ValueError, match="tbasis_ortho_window"):
        build(_toy_system(tbasis_ortho=True))


def test_the_t_basis_window_is_dropped_when_the_orthogonalisation_is_off():
    """Declaring a window beside ``tbasis_ortho=False`` must not switch it
    on behind the declaration's back."""
    model = build(_toy_system(tbasis_ortho=False,
                              tbasis_ortho_window=(1.0, 2.0),
                              tbasis_ortho_points=13))
    assert model.f_local.tbasis_ortho is False


def test_the_declared_local_form_and_a_kwargs_copy_of_it_must_agree():
    with pytest.raises(ValueError, match="u_form"):
        build(_toy_system(u_form="icnn"))


def test_a_kernel_form_this_rung_does_not_build_is_refused():
    system = _toy_system()
    other = dataclasses.replace(
        system, functional=Functional(form="nonlocal_kernel", local="mlp",
                                      kernel="rasterised",
                                      kwargs=system.functional.kwargs))
    with pytest.raises(NotImplementedError, match="rasterised"):
        build(other)


def test_a_form_with_no_measured_translation_is_refused_not_splatted():
    system = _toy_system()
    other = dataclasses.replace(
        system, functional=Functional(form="square_gradient", local="mlp",
                                      kernel=None, kwargs={"kappa_init": 1.0}))
    with pytest.raises(NotImplementedError, match="square_gradient"):
        build(other)


# --------------------------------------------------------------------------
# The standalone mobility
# --------------------------------------------------------------------------
def test_the_standalone_mobility_is_the_one_the_rung_builds():
    from aipf.mobility import Mobility as MobilityModule
    from aipf.mobility import build as build_mobility

    system = _toy_system()
    alone = build_mobility(system)
    inside = build(system)._mobility
    assert isinstance(alone, MobilityModule)
    assert (alone.prefactor_form, alone.shape_form, alone.t_form) == (
        inside.prefactor_form, inside.shape_form, inside.t_form)
    assert alone.n_species == inside.n_species
    assert {k: tuple(v.shape) for k, v in alone.state_dict().items()} == {
        k: tuple(v.shape) for k, v in inside.state_dict().items()}


def test_the_standalone_mobility_takes_overrides_too():
    from aipf.mobility import build as build_mobility

    m = build_mobility(_toy_system(), prefactor="partial_density")
    assert m.prefactor_form == "partial_density"
    assert m.build_overrides == {"prefactor": "partial_density"}


def test_the_standalone_mobility_refuses_an_unknown_override():
    from aipf.mobility import build as build_mobility

    with pytest.raises(TypeError, match="prefctor"):
        build_mobility(_toy_system(), prefctor="partial_density")


def test_the_standalone_mobility_refuses_a_system_that_declares_none():
    from aipf.mobility import build as build_mobility

    with pytest.raises(ValueError, match="mobility"):
        build_mobility(dataclasses.replace(_toy_system(), mobility=None))


# --------------------------------------------------------------------------
# Every declared-spelling override reaches the argument it maps to
#
# The rule the module's docstring states is that an override may be spelled
# the declaration's way or the constructor's way and both reach the same
# argument. Two of the declared spellings did not: `h_m` and `T_ref` are
# owned by the MOBILITY, and the translation read them off the declaration a
# second time, after the override had been applied to a copy -- so the
# override was dropped while `build_overrides` went on reporting it as
# applied. A silent no-op that reports success is worse than a refusal,
# which is why the table below is exhaustive over the declared spellings
# rather than a test per key somebody has to remember to add.
# --------------------------------------------------------------------------

#: One row per declared-spelling override: what to pass, and every
#: constructor argument it must reach, with the value it must arrive as.
_OVERRIDE_REACHES = [
    ({"h_w": 6}, {"kernel_hidden": 6}),
    ({"h_m": 6}, {"mobility_hidden": 6}),
    ({"g_form": "icnn"}, {"g_exc_form": "icnn"}),
    ({"T_ref": 1234.0}, {"mobility_t_ref": 1234.0}),
    ({"fexc_T_ref": 3.0}, {"kBT_ref": _TOY_KB * 3.0}),
    ({"activation": "tanh"}, {"local_activation": "tanh",
                              "kernel_activation": "tanh",
                              "mobility_activation": "tanh"}),
]

#: The declared spellings with no constructor argument of their own, and
#: where each is covered instead. Listed so that the completeness check
#: below cannot be satisfied by quietly forgetting one.
_COVERED_ELSEWHERE = {
    "tbasis_ortho": "switches a window on or off; two tests above",
    "disable_u": "no constructor argument at all; refused at its on value",
    "disable_g": "no constructor argument at all; refused at its on value",
}


@pytest.mark.parametrize("override,expected", _OVERRIDE_REACHES)
def test_a_declared_spelling_override_reaches_the_constructor_argument(
        override, expected):
    kwargs = rung_kwargs(_toy_system(), **override)
    for key, value in expected.items():
        assert kwargs[key] == value, (key, kwargs.get(key), value)


def test_the_override_table_covers_every_declared_spelling():
    """A declared-spelling override added later with no row here fails.

    This is the test whose absence let the mobility defect through: nothing
    said the set of override spellings and the set of checked spellings had
    to be the same set.
    """
    from aipf.functional.build import _DECLARED_OVERRIDES

    covered = {k for override, _ in _OVERRIDE_REACHES for k in override}
    assert covered | set(_COVERED_ELSEWHERE) == set(_DECLARED_OVERRIDES)


def test_a_mobility_owned_override_changes_the_built_mobility():
    """Not the kwargs dict: the MODEL. `h_m` is the mobility's width."""
    declared = build(_toy_system())
    overridden = build(_toy_system(), h_m=6)
    assert declared._mobility.m_net[0].out_features == 4
    assert overridden._mobility.m_net[0].out_features == 6
    assert overridden.build_overrides == {"h_m": 6}


def test_a_reference_temperature_override_changes_the_built_mobility():
    declared = build(_toy_arrhenius_system())
    overridden = build(_toy_arrhenius_system(), T_ref=1234.0)
    assert declared._mobility.t_ref == 1000.0
    assert overridden._mobility.t_ref == 1234.0
    assert overridden.build_overrides == {"T_ref": 1234.0}


def test_both_spellings_of_the_mobility_width_agree():
    """The declaration's `h_m` and the constructor's `mobility_hidden`.

    The defect was exactly a disagreement between these two, with only the
    second one working, so they are compared rather than each checked alone.
    """
    a = build(_toy_system(), h_m=6)._mobility
    b = build(_toy_system(), mobility_hidden=6)._mobility
    assert a.m_net[0].out_features == b.m_net[0].out_features == 6


def test_no_declared_spelling_override_is_a_silent_no_op():
    """The inverse of the table: nothing reported as applied did nothing.

    For every row, the constructor arguments with the override differ from
    the ones without it. A key that reported itself in `build_overrides`
    while changing no argument at all is what this catches.
    """
    base = rung_kwargs(_toy_system())
    for override, _ in _OVERRIDE_REACHES:
        changed = rung_kwargs(_toy_system(), **override)
        assert changed != base, override


# --------------------------------------------------------------------------
# The signature partition itself
# --------------------------------------------------------------------------
def test_the_two_named_sets_only_name_real_constructor_parameters():
    """A stale name in either set is a silent exemption.

    Rename a constructor parameter and leave the old spelling in
    ``_DECLARED_OPTIONAL`` and the new one becomes required without anybody
    noticing, which is fine -- but the old one goes on exempting nothing
    while reading as though it exempts something. That is the kind of dead
    entry a reader trusts, so it fails here.
    """
    import inspect

    from aipf.functional.build import (_BUILDER_SUPPLIED, _DECLARED_OPTIONAL,
                                       _REQUIRED_WHEN)
    from aipf.functional.nonlocal_kernel import NonlocalKernel

    parameters = set(
        inspect.signature(NonlocalKernel.__init__).parameters) - {"self"}
    assert _BUILDER_SUPPLIED <= parameters, _BUILDER_SUPPLIED - parameters
    assert set(_DECLARED_OPTIONAL) <= parameters, (
        set(_DECLARED_OPTIONAL) - parameters)
    assert set(_REQUIRED_WHEN) <= set(_DECLARED_OPTIONAL)


def test_no_parameter_is_both_supplied_and_optional():
    from aipf.functional.build import _BUILDER_SUPPLIED, _DECLARED_OPTIONAL

    assert not _BUILDER_SUPPLIED & set(_DECLARED_OPTIONAL)


def test_an_optional_parameter_becomes_required_once_its_form_is_chosen():
    """``u_degree`` is dead weight under ``local="mlp"`` and load-bearing
    under ``local="taylor"``, where omitting it takes the signature's 4."""
    import dataclasses

    from aipf.system import Functional

    system = _toy_system()
    taylor = dataclasses.replace(
        system, functional=Functional(form="nonlocal_kernel", local="taylor",
                                      kernel="radial_mlp",
                                      kwargs=system.functional.kwargs))
    with pytest.raises(ValueError, match="u_degree"):
        build(taylor)


def test_the_input_scaling_is_declared_never_defaulted():
    """A non-persistent buffer: a strict load cannot tell the two apart, so the door must."""
    with pytest.raises(ValueError, match="local_input_scale"):
        build(_without(_toy_system(), "local_input_scale"))
    system = _toy_system()
    silent = dataclasses.replace(system, mobility=Mobility(
        form="mlp_scaled", T_form="none",
        kwargs=dict(mobility_prefactor="mole_fraction")))
    with pytest.raises(ValueError, match="mobility_input_ref"):
        build(silent)


def test_the_declared_input_scaling_reaches_both_nets():
    system = _toy_system(local_input_scale=True)
    scaled = dataclasses.replace(system, mobility=Mobility(
        form="mlp_scaled", T_form="none",
        kwargs=dict(mobility_prefactor="mole_fraction",
                    mobility_input_ref=(0.3, 0.3))))
    model = build(scaled)
    assert model.f_local.input_scale is True
    assert torch.equal(model._mobility.in_ref, torch.tensor([0.3, 0.3]))
    assert build(_toy_system())._mobility.in_ref is None


def test_the_joint_form_requires_its_width_and_depth():
    joint = _toy_system(f_exc_form="joint", g_form="icnn")
    with pytest.raises(ValueError, match="h_joint"):
        build(joint)
    with pytest.raises(ValueError, match="joint_depth"):
        build(_toy_system(f_exc_form="joint", g_form="icnn", h_joint=4))
    model = build(_toy_system(f_exc_form="joint", g_form="icnn", h_joint=4,
                              joint_depth=3))
    linears = [m for m in model.f_local.u_joint_net.net
               if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4


def test_every_functional_form_has_a_module_of_its_own_name():
    import importlib
    from aipf.system import FUNCTIONAL_FORMS
    for form in FUNCTIONAL_FORMS:
        importlib.import_module(f"aipf.functional.{form}")


def test_the_form_to_module_dict_names_every_form_once():
    from aipf.functional.build import _FORM_MODULES
    from aipf.system import FUNCTIONAL_FORMS
    assert tuple(_FORM_MODULES) == FUNCTIONAL_FORMS
    assert all(m == f"aipf.functional.{f}" for f, m in _FORM_MODULES.items())
