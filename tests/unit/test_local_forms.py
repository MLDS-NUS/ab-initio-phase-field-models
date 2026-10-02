"""Tests for aipf.functional.local_forms: the rung-3 local f_loc registry.

Three training runs' published models genuinely disagree on this piece, and the
module under test exists so each is one declared configuration away rather
than a separate hand-rolled class: a split energy/entropy with a convex
energy term and a restricted (even-degree) polynomial; a split energy/
entropy with a plain MLP energy term, a non-convex MLP entropy term and the
extra T-basis heads on; and the split abandoned for one joint net. These
tests check the properties every one of those configurations must hold, at
``n_species`` 1 AND 2, not any one published model's literal fitted numbers.
"""
from __future__ import annotations

import itertools
import math

import pytest
import torch

from aipf.functional.local_forms import (
    F_EXC_FORMS,
    G_EXC_FORMS,
    IDEAL_FORMS,
    U_FORMS,
    U_PARITIES,
    FLocal,
    GeneralMLP,
    ICNN,
    JointEnergy,
    MLPEnergy,
    TaylorEnergy,
    _taylor_exponents,
)

N_SPECIES = (1, 2)


def _rho_ref(n_species):
    # Distinct, non-degenerate reference densities per channel, well away
    # from a species-symmetric special case that could hide a swapped axis.
    return [0.3, 0.45][:n_species]


def _sample_rho(n_species, n=6, low=0.1, high=0.7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, n_species, generator=g) * (high - low) + low


# ── the ideal term, analytic, present in every configuration ──────────────


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_f_pointwise_is_exactly_ideal_plus_u_plus_kBT_g(n_species):
    """f_pointwise assembles f_id + u(rho) + kBT*g_exc(rho) in this order
    and no other terms, in split mode with the T-basis heads off. u_net and
    g_net are exercised independently (rather than isolated by a zero-init
    coincidence) so this also pins the assembly, not just the ideal term."""
    model = FLocal(n_species, _rho_ref(n_species), kBT_ref=1.0, h_u=4, h_g=4,
                    u_form="taylor", u_degree=2, ideal_form="gas")
    rho = _sample_rho(n_species)
    kBT = torch.full((rho.shape[0],), 1.3)
    f = model.f_pointwise(rho, kBT)
    f_id = kBT * (rho * rho.log() - rho).sum(-1)
    expected = f_id + model.u_net(rho) + kBT * model.g_net(rho)
    assert torch.allclose(f, expected, atol=1e-6)


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_ideal_term_is_exactly_zero_when_u_and_g_are_zero_init(n_species):
    """With a zero-degree-2-coefficient Taylor u AND a zero-init general-MLP
    g_exc, f_exc vanishes identically and f_pointwise IS the ideal term."""
    model = FLocal(n_species, _rho_ref(n_species), kBT_ref=1.0, h_u=4, h_g=4,
                    u_form="taylor", u_degree=2, g_exc_form="mlp", ideal_form="gas")
    with torch.no_grad():
        model.g_net.net[-1].weight.zero_()
    rho = _sample_rho(n_species)
    kBT = torch.full((rho.shape[0],), 1.3)
    f = model.f_pointwise(rho, kBT)
    expected = kBT * (rho * rho.log() - rho).sum(-1)
    assert torch.allclose(f, expected, atol=1e-6)
    mu = model.mu_pointwise(rho, kBT)
    expected_mu = kBT.unsqueeze(-1) * rho.log()
    assert torch.allclose(mu, expected_mu, atol=1e-6)


# ── ideal_form: the two closed forms it declares, and why one is required ──
#
# `ideal_form="gas"` is `f_id = kBT * sum_i [rho_i ln(rho_i) - rho_i]`, the
# form the module carried before this axis existed, defined on a partial
# DENSITY (any rho_i >= 0, no upper bound). `ideal_form="lattice"` is
# `f_id = kBT * sum_i [rho_i ln(rho_i) + (1 - rho_i) ln(1 - rho_i)]`, defined
# on a mole/site FRACTION strictly inside (0, 1) per species -- both ends
# are singular, unlike the gas form's single singularity at 0. One real
# published model's published checkpoint is fit against the lattice form; loading
# it into the gas form silently reproduces a different physical model.


def _zero_excess_model(n_species, ideal_form):
    """A model whose only live term is the ideal one: zero-init Taylor u
    (coefficients start at exactly zero) plus a zero-init general-MLP
    g_exc, so `f_pointwise` reduces to `_f_id` exactly."""
    model = FLocal(n_species, _rho_ref(n_species), kBT_ref=1.0, h_u=4, h_g=4,
                    u_form="taylor", u_degree=2, g_exc_form="mlp",
                    ideal_form=ideal_form)
    with torch.no_grad():
        model.g_net.net[-1].weight.zero_()
    return model


def test_gas_ideal_term_matches_a_hand_computed_value_one_species():
    model = _zero_excess_model(1, "gas")
    rho = torch.tensor([[0.2]])
    kBT = torch.tensor([2.0])
    f = model.f_pointwise(rho, kBT)
    expected = 2.0 * (0.2 * math.log(0.2) - 0.2)
    assert torch.allclose(f, torch.tensor([expected]), atol=1e-6)


def test_lattice_ideal_term_matches_a_hand_computed_value_one_species():
    model = _zero_excess_model(1, "lattice")
    rho = torch.tensor([[0.2]])
    kBT = torch.tensor([2.0])
    f = model.f_pointwise(rho, kBT)
    expected = 2.0 * (0.2 * math.log(0.2) + 0.8 * math.log(0.8))
    assert torch.allclose(f, torch.tensor([expected]), atol=1e-6)


def test_lattice_ideal_term_matches_a_hand_computed_value_two_species():
    model = _zero_excess_model(2, "lattice")
    rho = torch.tensor([[0.2, 0.6]])
    kBT = torch.tensor([1.5])
    f = model.f_pointwise(rho, kBT)
    term1 = 0.2 * math.log(0.2) + 0.8 * math.log(0.8)
    term2 = 0.6 * math.log(0.6) + 0.4 * math.log(0.4)
    expected = 1.5 * (term1 + term2)
    assert torch.allclose(f, torch.tensor([expected]), atol=1e-6)


def test_gas_ideal_term_matches_a_hand_computed_value_two_species():
    model = _zero_excess_model(2, "gas")
    rho = torch.tensor([[0.2, 0.6]])
    kBT = torch.tensor([1.5])
    f = model.f_pointwise(rho, kBT)
    term1 = 0.2 * math.log(0.2) - 0.2
    term2 = 0.6 * math.log(0.6) - 0.6
    expected = 1.5 * (term1 + term2)
    assert torch.allclose(f, torch.tensor([expected]), atol=1e-6)


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_gas_and_lattice_ideal_terms_are_genuinely_different_closed_forms(
    n_species,
):
    """These are not a reparametrisation of one another: pin the fact that
    they differ, at a scale loosely bracketing the ~0.67 max-absolute
    difference measured between this pair of forms over a real
    checkpoint's own density range, rather than fitting an exact digit a
    future refactor could accidentally preserve while breaking the
    substance of the check."""
    rho_ref = _rho_ref(n_species)
    torch.manual_seed(0)
    gas = _zero_excess_model(n_species, "gas")
    torch.manual_seed(0)
    lattice = _zero_excess_model(n_species, "lattice")
    # A density range that reaches close to 1 per channel, same shape the
    # real checkpoint's trust domain reaches -- the two forms agree closely
    # near the middle of (0, 1) and diverge sharply near the upper edge.
    rho = torch.tensor([[0.9] * n_species, [0.5] * n_species,
                         [0.2] * n_species])
    kBT = torch.full((rho.shape[0],), 1.0)
    f_gas = gas.f_pointwise(rho, kBT)
    f_lattice = lattice.f_pointwise(rho, kBT)
    max_abs_diff = (f_gas - f_lattice).abs().max().item()
    assert 0.3 < max_abs_diff < 2.0, max_abs_diff
    # A test that passed for either form alone would be worthless here --
    # the point is that they disagree, not that either is internally
    # consistent.
    assert not torch.allclose(f_gas, f_lattice, atol=1e-3)


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_lattice_ideal_term_is_symmetric_under_rho_to_one_minus_rho(n_species):
    """g_id(rho) = rho ln(rho) + (1 - rho) ln(1 - rho) is symmetric under
    rho -> 1 - rho by construction -- swapping the two additive terms
    reproduces the same sum -- unlike a LEARNED net over rho (g_exc, u),
    which needs an explicit post-hoc average over rho and 1 - rho to gain
    the same symmetry (a pre-existing, separate gap in this module's ICNN/
    GeneralMLP, not addressed here). No extra flag or averaging step is
    added for the ideal term itself: this test pins the symmetry directly
    against `f_pointwise` with the excess terms zeroed, exercising the real
    dispatch path rather than a hand-reimplemented formula."""
    model = _zero_excess_model(n_species, "lattice")
    rho = _sample_rho(n_species)
    kBT = torch.full((rho.shape[0],), 1.3)
    f = model.f_pointwise(rho, kBT)
    f_mirror = model.f_pointwise(1.0 - rho, kBT)
    assert torch.allclose(f, f_mirror, atol=1e-6)


def test_gas_ideal_term_is_not_symmetric_under_rho_to_one_minus_rho():
    """Contrast case: the gas form has no such symmetry (it is not even
    defined the same way outside (0, 1)), so the lattice test above is
    pinning a real, form-specific property, not an accident of the test
    harness."""
    model = _zero_excess_model(1, "gas")
    rho = torch.tensor([[0.2]])
    kBT = torch.tensor([1.3])
    f = model.f_pointwise(rho, kBT)
    f_mirror = model.f_pointwise(1.0 - rho, kBT)
    assert not torch.allclose(f, f_mirror, atol=1e-3)


def test_ideal_form_lattice_clamps_at_both_ends():
    """The lattice form has TWO singularities (rho == 0 and rho == 1),
    unlike the gas form's one (rho == 0 only): a rho at or beyond either
    edge must not produce inf/NaN, in EITHER `_f_id` (via `f_pointwise`)
    or `_mu_id` (via `mu_pointwise`) -- two independent clamp sites, not
    one shared helper, so each is checked directly."""
    model = _zero_excess_model(1, "lattice")
    rho = torch.tensor([[0.0], [1.0], [-0.3], [1.4]])
    kBT = torch.full((rho.shape[0],), 1.0)
    f = model.f_pointwise(rho, kBT)
    assert torch.isfinite(f).all(), f
    mu = model.mu_pointwise(rho, kBT)
    assert torch.isfinite(mu).all(), mu


# ── mu is the gradient of f_pointwise, for every axis combination ─────────


def _configs():
    """Every f_exc_form / u_form / g_exc_form combination this registry
    declares as valid, plus the T-basis heads on and off."""
    for f_exc_form in F_EXC_FORMS:
        if f_exc_form == "joint":
            yield dict(f_exc_form="joint")
            continue
        for u_form in U_FORMS:
            for g_exc_form in G_EXC_FORMS:
                yield dict(f_exc_form="split", u_form=u_form,
                            g_exc_form=g_exc_form)
    yield dict(f_exc_form="split", u_form="mlp", g_exc_form="icnn",
               enable_TlnT=True)
    yield dict(f_exc_form="split", u_form="mlp", g_exc_form="mlp",
               enable_T2=True)
    yield dict(f_exc_form="split", u_form="mlp", g_exc_form="mlp",
               enable_TlnT=True, enable_T2=True)


def _build(n_species, **overrides):
    kwargs = dict(n_species=n_species, rho_ref=_rho_ref(n_species),
                  kBT_ref=1.0, h_u=6, h_g=6, ideal_form="gas")
    if overrides.get("f_exc_form") == "joint":
        kwargs["h_joint"] = 6
    if overrides.get("enable_TlnT"):
        kwargs["h_g_hat"] = 5
    if overrides.get("enable_T2"):
        kwargs["h_g_tilde"] = 5
    kwargs.update(overrides)
    torch.manual_seed(0)
    return FLocal(**kwargs)


@pytest.mark.parametrize("n_species", N_SPECIES)
@pytest.mark.parametrize("ideal_form", IDEAL_FORMS)
@pytest.mark.parametrize("config", list(_configs()),
                         ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()))
def test_mu_is_the_gradient_of_f_pointwise(n_species, ideal_form, config):
    """mu = d(f_pointwise)/d(rho) against autograd, across every f_exc/u/
    g_exc/T-basis combination this registry declares AND both ideal_form
    values -- the ideal term's own gradient (`_mu_id`) has to agree with
    autograd through `_f_id` too, not just the excess-term pieces."""
    model = _build(n_species, ideal_form=ideal_form, **config)
    rho = _sample_rho(n_species).requires_grad_(True)
    kBT = torch.full((rho.shape[0],), 1.4)
    f = model.f_pointwise(rho, kBT)
    (autograd_mu,) = torch.autograd.grad(f.sum(), rho, create_graph=False)
    mu = model.mu_pointwise(rho.detach(), kBT)
    assert torch.allclose(mu, autograd_mu, atol=1e-5), (ideal_form, config)


# ── the gauge (Trap 1) ─────────────────────────────────────────────────


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_mlp_energy_gauge_fix_vanishes_at_rho_ref(n_species):
    rho_ref = _rho_ref(n_species)
    torch.manual_seed(3)
    net = MLPEnergy(n_species, 8, rho_ref, gauge_fix=True)
    ref = torch.tensor(rho_ref).unsqueeze(0)
    u_ref = net(ref)
    assert torch.allclose(u_ref, torch.zeros_like(u_ref), atol=1e-6)
    ref_var = ref.clone().requires_grad_(True)
    (grad_ref,) = torch.autograd.grad(net(ref_var).sum(), ref_var)
    assert torch.allclose(grad_ref, torch.zeros_like(grad_ref), atol=1e-5)


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_mlp_energy_without_gauge_fix_is_generally_nonzero_at_rho_ref(n_species):
    """The gauge is a declared choice, not an automatic property of the raw
    net: dropping it changes what the learned u means (module docstring)."""
    rho_ref = _rho_ref(n_species)
    torch.manual_seed(3)
    net = MLPEnergy(n_species, 8, rho_ref, gauge_fix=False)
    ref = torch.tensor(rho_ref).unsqueeze(0)
    assert not torch.allclose(net(ref), torch.zeros(1), atol=1e-6)


@pytest.mark.parametrize("n_species", N_SPECIES)
@pytest.mark.parametrize("parity", U_PARITIES)
def test_taylor_energy_vanishes_at_rho_ref_by_construction(n_species, parity):
    rho_ref = _rho_ref(n_species)
    net = TaylorEnergy(n_species, rho_ref, degree=4, parity=parity,
                       variable="ratio")
    with torch.no_grad():
        net.coefs.normal_()  # nonzero coefficients -- the vanishing is
                              # structural, not an artifact of zero init
    ref = torch.tensor(rho_ref).unsqueeze(0).requires_grad_(True)
    u_ref = net(ref)
    assert torch.allclose(u_ref, torch.zeros_like(u_ref), atol=1e-6)
    (grad_ref,) = torch.autograd.grad(u_ref.sum(), ref)
    assert torch.allclose(grad_ref, torch.zeros_like(grad_ref), atol=1e-5)


def test_gauge_choice_changes_the_bulk_value_away_from_rho_ref():
    """The gauge interacts with L_bulk (brief, Trap 1): gauged and
    un-gauged energies generally disagree everywhere except at rho_ref."""
    rho_ref = [0.3]
    torch.manual_seed(5)
    gauged = MLPEnergy(1, 8, rho_ref, gauge_fix=True)
    torch.manual_seed(5)
    ungauged = MLPEnergy(1, 8, rho_ref, gauge_fix=False)
    rho = torch.tensor([[0.55]])
    assert not torch.allclose(gauged(rho), ungauged(rho), atol=1e-6)


# ── analytic Taylor value, against a hand-computed polynomial ─────────────


def test_taylor_energy_matches_a_hand_set_polynomial_one_species():
    rho_ref = [0.4]
    net = TaylorEnergy(1, rho_ref, degree=4, parity="full", variable="ratio")
    with torch.no_grad():
        net.coefs.zero_()
        # exponents are generated in the order (2,), (3,), (4,)
        assert net.exps.tolist() == [[2], [3], [4]]
        net.coefs[:] = torch.tensor([2.0, -1.0, 0.5])
    rho = torch.tensor([[0.6]])
    z = 0.6 / 0.4 - 1.0
    expected = 2.0 * z**2 - 1.0 * z**3 + 0.5 * z**4
    assert torch.allclose(net(rho), torch.tensor([expected]), atol=1e-6)


def test_taylor_energy_even_parity_drops_odd_exponents():
    exps_even = _taylor_exponents(1, 6, "even")
    exps_full = _taylor_exponents(1, 6, "full")
    assert exps_even == [(2,), (4,), (6,)]
    assert exps_full == [(2,), (3,), (4,), (5,), (6,)]


def test_taylor_energy_two_species_exponents_are_the_full_redlich_kister_set():
    exps = _taylor_exponents(2, 4, "full")
    expected = sorted((i, tot - i) for tot in range(2, 5) for i in range(tot + 1))
    assert sorted(exps) == expected


# ── ICNN convexity, sampled ────────────────────────────────────────────


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_icnn_is_convex_on_a_sampled_grid(n_species):
    torch.manual_seed(7)
    net = ICNN(n_species, 12, output_bias=True)
    points = _sample_rho(n_species, n=12, low=0.1, high=0.6, seed=11)
    for i in range(points.shape[0]):
        x = points[i : i + 1].clone().requires_grad_(True)
        hess = torch.autograd.functional.hessian(
            lambda z: net(z).sum(), x
        ).reshape(n_species, n_species)
        eigenvalues = torch.linalg.eigvalsh(hess)
        assert torch.all(eigenvalues >= -1e-4), (i, eigenvalues)


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_icnn_is_convex_by_the_midpoint_definition_under_adversarial_weights(
    n_species,
):
    """Jensen's inequality directly, f(t x1 + (1-t) x2) <= t f(x1) + (1-t)
    f(x2), rather than a local Hessian sample: at the default (small,
    near -2-biased) init the raw hidden-to-hidden weight is ALREADY so
    negative that dropping the non-negativity constraint produces no
    measurable curvature at ordinary density magnitudes (a real gap this
    test closes over a Hessian-only check). Scaling the pre-activations up
    and setting the hidden-to-hidden weight to a signed, larger-magnitude
    matrix makes a dropped non-negativity constraint fail by a wide,
    unambiguous margin, while the constrained net stays exactly convex
    regardless of what raw values that weight is set to -- softplus always
    makes it non-negative before it is used."""
    torch.manual_seed(21)
    net = ICNN(n_species, 12, output_bias=True)
    with torch.no_grad():
        net.W0.weight.mul_(5.0)
        net.W0.bias.zero_()
        net.W1.weight.mul_(5.0)
        net.W1.bias.zero_()
        net.A1_raw.copy_(3.0 * torch.randn(12, 12))
    g = torch.Generator().manual_seed(31)
    for _ in range(50):
        x1 = torch.randn(1, n_species, generator=g) * 2
        x2 = torch.randn(1, n_species, generator=g) * 2
        t = torch.rand(1, generator=g).item()
        xm = t * x1 + (1 - t) * x2
        with torch.no_grad():
            lhs = net(xm).item()
            rhs = t * net(x1).item() + (1 - t) * net(x2).item()
        assert rhs - lhs >= -1e-4, (lhs, rhs)


# ── enable_TlnT / enable_T2: zero-init no-op, then a real contribution ────


@pytest.mark.parametrize("n_species", N_SPECIES)
@pytest.mark.parametrize("flag,hidden_kw", [("enable_TlnT", "h_g_hat"),
                                            ("enable_T2", "h_g_tilde")])
def test_t_basis_head_is_an_exact_no_op_at_init(n_species, flag, hidden_kw):
    rho_ref = _rho_ref(n_species)
    torch.manual_seed(9)
    off = FLocal(n_species, rho_ref, kBT_ref=1.0, h_u=6, h_g=6, ideal_form="gas")
    torch.manual_seed(9)
    on = FLocal(n_species, rho_ref, kBT_ref=1.0, h_u=6, h_g=6,
                **{flag: True, hidden_kw: 5}, ideal_form="gas")
    rho = _sample_rho(n_species)
    kBT = torch.full((rho.shape[0],), 1.5)
    assert torch.equal(off.f_pointwise(rho, kBT), on.f_pointwise(rho, kBT))
    assert torch.equal(off.mu_pointwise(rho, kBT), on.mu_pointwise(rho, kBT))


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_t_basis_head_becomes_a_real_contribution_once_trained(n_species):
    """The zero-init property is about INIT, not a permanently inert head:
    once the last-layer weight moves, the term is live."""
    rho_ref = _rho_ref(n_species)
    torch.manual_seed(9)
    model = FLocal(n_species, rho_ref, kBT_ref=1.0, h_u=6, h_g=6,
                    enable_TlnT=True, h_g_hat=5, ideal_form="gas")
    rho = _sample_rho(n_species)
    kBT = torch.full((rho.shape[0],), 1.5)
    before = model.f_pointwise(rho, kBT).clone()
    with torch.no_grad():
        model.g_hat_net.net[-1].weight.add_(0.7)
    after = model.f_pointwise(rho, kBT)
    assert not torch.allclose(before, after)


# ── construction order (Trap 2): the identical-weights guarantee ──────────


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_shared_submodules_are_identical_whether_or_not_a_variant_is_on(n_species):
    """u_net and g_net must draw the SAME weights at a fixed seed whether
    or not the T-basis heads are requested: the heads are built LAST, so
    the flag only ever appends draws, never inserts them earlier."""
    rho_ref = _rho_ref(n_species)
    torch.manual_seed(42)
    off = FLocal(n_species, rho_ref, kBT_ref=1.0, h_u=6, h_g=6, ideal_form="gas")
    torch.manual_seed(42)
    on = FLocal(n_species, rho_ref, kBT_ref=1.0, h_u=6, h_g=6,
                enable_TlnT=True, h_g_hat=5, enable_T2=True, h_g_tilde=5, ideal_form="gas")
    for name in ("u_net", "g_net"):
        sd_off = getattr(off, name).state_dict()
        sd_on = getattr(on, name).state_dict()
        assert sd_off.keys() == sd_on.keys()
        for key in sd_off:
            assert torch.equal(sd_off[key], sd_on[key]), (name, key)


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_deferred_t_heads_do_not_disturb_a_caller_s_later_submodules(n_species):
    """`defer_T_heads=True` moves the head draws past whatever the CALLER
    builds next, so a caller's own later submodule also stays identical
    across the flag."""
    rho_ref = _rho_ref(n_species)

    def build(enable):
        kwargs = dict(enable_TlnT=True, h_g_hat=5) if enable else {}
        model = FLocal(n_species, rho_ref, kBT_ref=1.0, h_u=6, h_g=6,
                        defer_T_heads=True, **kwargs, ideal_form="gas")
        later = torch.nn.Linear(n_species, 3)  # stands in for a caller's
                                                # kernel/mobility submodule
        model.build_T_heads()
        return model, later

    torch.manual_seed(123)
    off_model, off_later = build(False)
    torch.manual_seed(123)
    on_model, on_later = build(True)
    for key in off_later.state_dict():
        assert torch.equal(off_later.state_dict()[key],
                            on_later.state_dict()[key])


def test_build_t_heads_refuses_a_second_call():
    model = FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4,
                    enable_TlnT=True, h_g_hat=4, defer_T_heads=True, ideal_form="gas")
    model.build_T_heads()
    with pytest.raises(RuntimeError):
        model.build_T_heads()


def test_evaluating_before_build_t_heads_raises():
    model = FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4,
                    enable_TlnT=True, h_g_hat=4, defer_T_heads=True, ideal_form="gas")
    rho = _sample_rho(1)
    kBT = torch.full((rho.shape[0],), 1.0)
    with pytest.raises(RuntimeError):
        model.f_pointwise(rho, kBT)
    with pytest.raises(RuntimeError):
        model.mu_pointwise(rho, kBT)


# ── non-persistent buffers (Trap 3) ────────────────────────────────────


def test_kBT_ref_buffer_is_non_persistent():
    """state_dict keys stay identical across every variant flag: a
    persistent kBT_ref would break checkpoint loading the moment a
    trained system changed its reference temperature."""
    model = FLocal(1, [0.3], kBT_ref=987.0, h_u=4, h_g=4, ideal_form="gas")
    assert "kBT_ref" not in model.state_dict()
    # still a real, readable buffer -- just not saved/restored with the model
    assert torch.isclose(model.kBT_ref, torch.tensor(987.0, dtype=torch.float64))


def test_state_dict_keys_are_identical_across_a_kBT_ref_change():
    a = FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4, ideal_form="gas")
    b = FLocal(1, [0.3], kBT_ref=999.0, h_u=4, h_g=4, ideal_form="gas")
    assert a.state_dict().keys() == b.state_dict().keys()


def test_state_dict_keys_are_identical_regardless_of_t_head_flags():
    """The other half of the state_dict-stability guarantee: turning a
    T-basis head off does not remove a rho_ref/coefficient-style buffer
    that the on-configuration would otherwise carry as persistent state."""
    off = FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4, ideal_form="gas")
    on = FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4, enable_TlnT=True,
                h_g_hat=4, ideal_form="gas")
    shared = off.state_dict().keys()
    assert shared <= on.state_dict().keys()


# ── declared forms: unknown names raise ────────────────────────────────


@pytest.mark.parametrize("field,bad", [
    ("f_exc_form", "nonlocal"),
    ("u_form", "rbf"),
    ("g_exc_form", "gaussian"),
    ("u_parity", "odd"),
    ("activation", "relu6"),
    ("ideal_form", "poisson"),
])
def test_an_unknown_declared_form_raises(field, bad):
    kwargs = dict(n_species=1, rho_ref=[0.3], kBT_ref=1.0, h_u=4, h_g=4,
                  ideal_form="gas")
    kwargs[field] = bad
    with pytest.raises(ValueError):
        FLocal(**kwargs)


def test_ideal_form_has_no_default_and_is_required():
    """Unlike the other declared axes (all defaulted), `ideal_form` has no
    default: a caller that does not name it explicitly gets a TypeError
    from Python's own required-keyword mechanism, not a silently-picked
    physics choice."""
    with pytest.raises(TypeError):
        FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4)


def test_joint_form_forbids_g_exc_form_mlp():
    with pytest.raises(ValueError):
        FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4, h_joint=4,
               f_exc_form="joint", g_exc_form="mlp", ideal_form="gas")


def test_joint_form_forbids_enable_TlnT():
    with pytest.raises(ValueError):
        FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4, h_joint=4,
               f_exc_form="joint", enable_TlnT=True, h_g_hat=4, ideal_form="gas")


def test_joint_form_forbids_enable_T2():
    with pytest.raises(ValueError):
        FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4, h_joint=4,
               f_exc_form="joint", enable_T2=True, h_g_tilde=4, ideal_form="gas")


def test_joint_form_without_h_joint_raises():
    with pytest.raises(ValueError):
        FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4, f_exc_form="joint", ideal_form="gas")


def test_enable_TlnT_without_h_g_hat_raises():
    with pytest.raises(ValueError):
        FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4, enable_TlnT=True, ideal_form="gas")


def test_enable_T2_without_h_g_tilde_raises():
    with pytest.raises(ValueError):
        FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4, enable_T2=True, ideal_form="gas")


def test_rho_ref_length_mismatch_raises():
    with pytest.raises(ValueError):
        FLocal(2, [0.3], kBT_ref=1.0, h_u=4, h_g=4, ideal_form="gas")


def test_rho_ref_length_mismatch_raises_even_when_no_submodule_reads_rho_ref():
    """u_form='icnn' (and g_exc_form='icnn') builds only ICNN submodules,
    which never touch rho_ref at all -- unlike the default u_form='mlp'
    path, where MLPEnergy's own constructor happens to repeat this same
    check. FLocal's own validation is the only thing that can catch a
    wrong-length rho_ref for this configuration."""
    with pytest.raises(ValueError):
        FLocal(2, [0.3], kBT_ref=1.0, h_u=4, h_g=4, u_form="icnn",
               g_exc_form="icnn", ideal_form="gas")


def test_rho_ref_length_mismatch_raises_for_the_joint_form_too():
    """f_exc_form='joint' builds only JointEnergy, which also never reads
    rho_ref."""
    with pytest.raises(ValueError):
        FLocal(2, [0.3], kBT_ref=1.0, h_u=4, h_g=4, h_joint=4,
               f_exc_form="joint", ideal_form="gas")


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_taylor_energy_hessian_at_rho_ref_has_no_nan(n_species):
    """Monomials are built from CUMULATIVE powers, not torch.pow(z, k) per
    term (module docstring): naive per-term `z ** k` differentiates through
    an internal log(z), which is -inf at z == 0 (rho == rho_ref) and 0 *
    -inf -> NaN in the second derivative -- exactly the point a
    static-structure loss's Hessian is evaluated hottest."""
    rho_ref = _rho_ref(n_species)
    net = TaylorEnergy(n_species, rho_ref, degree=5, parity="full",
                       variable="ratio")
    with torch.no_grad():
        net.coefs.normal_()
    ref = torch.tensor(rho_ref).unsqueeze(0)
    hess = torch.autograd.functional.hessian(lambda z: net(z).sum(), ref)
    assert torch.isfinite(hess).all(), hess


def test_taylor_energy_rejects_degree_below_two():
    with pytest.raises(ValueError):
        TaylorEnergy(1, [0.3], degree=1, variable="ratio")


def test_unknown_activation_raises_even_when_no_net_would_otherwise_use_it():
    """u_form='icnn' and g_exc_form='icnn' with the T-basis heads off build
    NO GeneralMLP submodule at all, so nothing downstream ever calls
    `_make_activation` on this config's `activation` string. The only thing
    that can catch a bad name here is FLocal's own validation-first check."""
    with pytest.raises(ValueError):
        FLocal(1, [0.3], kBT_ref=1.0, h_u=4, h_g=4, u_form="icnn",
               g_exc_form="icnn", activation="not-a-real-activation", ideal_form="gas")


# ── n_species generality: the taylor exponent count is the closed form ────


@pytest.mark.parametrize("n_species,degree", [(1, 6), (2, 6), (3, 5)])
def test_taylor_full_exponent_count_matches_the_combinatorial_closed_form(
    n_species, degree
):
    """Number of compositions of `total` into n_species parts is
    C(total + n_species - 1, n_species - 1); sum over total 2..degree."""
    exps = _taylor_exponents(n_species, degree, "full")
    expected = sum(math.comb(total + n_species - 1, n_species - 1)
                   for total in range(2, degree + 1))
    assert len(exps) == expected


def test_taylor_exponents_are_unique():
    exps = _taylor_exponents(2, 6, "full")
    assert len(exps) == len(set(exps))


# ── joint form: shape and species-broadcast sanity ─────────────────────


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_joint_energy_output_shape(n_species):
    torch.manual_seed(2)
    net = JointEnergy(n_species, 6)
    rho = _sample_rho(n_species)
    t_ratio = torch.full((rho.shape[0],), 0.8)
    out = net(rho, t_ratio)
    assert out.shape == (rho.shape[0],)


# ── GeneralMLP zero_init produces an exact no-op net ───────────────────


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_general_mlp_zero_init_outputs_exactly_zero(n_species):
    torch.manual_seed(4)
    net = GeneralMLP(n_species, 8, zero_init=True)
    rho = _sample_rho(n_species)
    out = net(rho)
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("n_species", N_SPECIES)
def test_general_mlp_zero_init_still_has_a_live_gradient(n_species):
    torch.manual_seed(4)
    net = GeneralMLP(n_species, 8, zero_init=True)
    rho = _sample_rho(n_species).requires_grad_(True)
    net(rho).sum().backward()
    last_weight_grad = net.net[-1].weight.grad
    assert last_weight_grad is not None
    assert torch.any(last_weight_grad != 0)


# --------------------------------------------------------------------------
# T-basis orthogonalisation (the first system's `FLocal`, where
# the published checkpoint's own `model_kwargs` record `tbasis_ortho=True`).
#
# The prefactors kBT*ln(kBT/kBT_ref) and kBT*(kBT/kBT_ref) sit close to the
# span of {1, kBT}, which the other heads already carry, so the heads trade
# against each other rather than splitting the physics. Subtracting the
# least-squares projection over a declared window kills the trade direction
# while leaving the overall span alone. The window is system physics, so
# core ships no value for it: `None` is off, and that is the only default
# core is allowed to have here.
# --------------------------------------------------------------------------

def _ortho_model(n_species, **kw):
    return FLocal(n_species, _rho_ref(n_species), kBT_ref=0.5, h_u=4, h_g=4,
                  ideal_form="gas", g_exc_form="mlp", enable_TlnT=True,
                  enable_T2=True, h_g_hat=4, h_g_tilde=4, **kw)


@pytest.mark.parametrize("n_species", (1, 2))
def test_no_window_leaves_both_prefactors_untouched(n_species):
    """The default is genuinely off, not a window with a neutral value."""
    m = _ortho_model(n_species)
    assert m.tbasis_ortho is False
    kBT = torch.linspace(0.3, 0.9, 7, dtype=torch.float64)
    assert torch.equal(m._c_TlnT(kBT), kBT * torch.log(kBT / m.kBT_ref))
    assert torch.equal(m._c_T2(kBT), kBT * (kBT / m.kBT_ref))


@pytest.mark.parametrize("n_species", (1, 2))
def test_window_drives_both_prefactors_to_near_zero_in_window(n_species):
    """The whole point of the construction: ~0 inside the declared window."""
    lo, hi = 0.6, 0.9
    plain = _ortho_model(n_species)
    ortho = _ortho_model(n_species, tbasis_ortho_window=(lo, hi),
                         tbasis_ortho_points=13)
    assert ortho.tbasis_ortho is True
    kBT = torch.linspace(lo, hi, 25, dtype=torch.float64)
    for name in ("_c_TlnT", "_c_T2"):
        before = getattr(plain, name)(kBT).abs().max()
        after = getattr(ortho, name)(kBT).abs().max()
        assert after < 0.05 * before, (name, float(before), float(after))


@pytest.mark.parametrize("n_species", (1, 2))
def test_orthogonalised_prefactor_still_spans_the_same_functions(n_species):
    """Same span, different representative: the subtracted piece is exactly
    affine in kBT, so the difference is reproduced by {1, kBT} alone."""
    ortho = _ortho_model(n_species, tbasis_ortho_window=(0.6, 0.9),
                         tbasis_ortho_points=13)
    plain = _ortho_model(n_species)
    kBT = torch.linspace(0.2, 1.4, 31, dtype=torch.float64)
    for name, tag in (("_c_TlnT", "tlnt"), ("_c_T2", "t2")):
        diff = getattr(plain, name)(kBT) - getattr(ortho, name)(kBT)
        a = getattr(ortho, f"_ortho_{tag}_a")
        b = getattr(ortho, f"_ortho_{tag}_b")
        assert torch.allclose(diff, a + b * kBT, atol=1e-12)


@pytest.mark.parametrize("n_species", (1, 2))
def test_the_ortho_constants_are_not_saved_into_the_state_dict(n_species):
    """Derived config, not learned state: declaring a window must not change
    a single state_dict key, or checkpoints stop loading across the flag."""
    plain = _ortho_model(n_species)
    ortho = _ortho_model(n_species, tbasis_ortho_window=(0.6, 0.9),
                         tbasis_ortho_points=13)
    assert sorted(plain.state_dict()) == sorted(ortho.state_dict())
    ortho.load_state_dict(plain.state_dict(), strict=True)


@pytest.mark.parametrize("n_species", (1, 2))
def test_a_window_without_its_sample_count_raises(n_species):
    with pytest.raises(ValueError, match="tbasis_ortho_points"):
        _ortho_model(n_species, tbasis_ortho_window=(0.6, 0.9))


@pytest.mark.parametrize("n_species", (1, 2))
def test_a_sample_count_without_a_window_raises(n_species):
    with pytest.raises(ValueError, match="nothing to fit"):
        _ortho_model(n_species, tbasis_ortho_points=13)


@pytest.mark.parametrize("bad", ((0.9, 0.6), (0.7, 0.7)))
def test_a_window_that_is_not_increasing_raises(bad):
    with pytest.raises(ValueError, match="hi > lo"):
        _ortho_model(2, tbasis_ortho_window=bad, tbasis_ortho_points=13)


@pytest.mark.parametrize("bad", (0, 1))
def test_fewer_than_two_samples_cannot_fit_two_coefficients(bad):
    with pytest.raises(ValueError, match=">= 2"):
        _ortho_model(2, tbasis_ortho_window=(0.6, 0.9),
                     tbasis_ortho_points=bad)


def test_the_ortho_constants_are_reproducible_across_builds():
    """Two models built from one configuration must agree in these buffers.

    `torch.linalg.lstsq` left unnamed is not reproducible run to run --
    measured, 200 identical repeats of this fit inside ONE process give two
    distinct answers, modal share 0.93, scattering up to 16 ULP. These
    constants are drawn at BUILD time, so that scatter makes two models
    built from the same checkpoint differ in exactly these four buffers.

    It was first measured as an ENVIRONMENT delta -- 60 entries of one --
    before being traced here, which is why the driver is named and why this
    test asserts equality across repeated builds rather than against a
    stored value.
    """
    window, points = (0.6, 0.9), 13
    first = _ortho_model(2, tbasis_ortho_window=window,
                         tbasis_ortho_points=points)
    for _ in range(12):
        again = _ortho_model(2, tbasis_ortho_window=window,
                             tbasis_ortho_points=points)
        for tag in ("tlnt", "t2"):
            for part in ("a", "b"):
                name = f"_ortho_{tag}_{part}"
                assert torch.equal(getattr(first, name),
                                   getattr(again, name)), name


def test_the_named_driver_matches_the_one_the_measured_source_used():
    """The pin names a driver; the source named none, and got a third one.

    The measured source solves this fit with `numpy.linalg.lstsq(...,
    rcond=None)`, which is LAPACK **gelsd**. This package pins **gels**,
    because gels is the driver whose answer the unnamed torch call actually
    returns here -- so the two are not the same routine and nothing until
    now compared them. The end-to-end parity gate cannot: it runs in
    float32, where every driver's answer collapses to the same constant,
    which is why a mutation from `gels` to `gelsd` survives that gate and
    is an EQUIVALENT mutant rather than an uncaught one.

    Measured on the published 13-point fit: the two answers
    differ by 1.1e-16 in `a` and 4.4e-16 in `b`, i.e. one or two ULP, and
    are bit-identical once cast to float32. This asserts that, so that a
    later change of driver has to stay inside it.
    """
    import numpy as np

    window, points = (0.6, 0.9), 13
    model = _ortho_model(2, tbasis_ortho_window=window,
                         tbasis_ortho_points=points)
    kbt = torch.linspace(window[0], window[1], points, dtype=torch.float64)
    ref = model.kBT_ref.to(dtype=torch.float64)
    design = torch.stack([torch.ones_like(kbt), kbt], dim=1).numpy()
    curves = {"tlnt": kbt * torch.log(kbt / ref),
              "t2": kbt * (kbt / ref)}
    for name, curve in curves.items():
        source = np.linalg.lstsq(design, curve.numpy(), rcond=None)[0]
        for part, value in zip(("a", "b"), source):
            pinned = float(getattr(model, f"_ortho_{name}_{part}"))
            assert abs(pinned - float(value)) < 1e-14, (
                f"{name}.{part}: the pinned driver's constant {pinned!r} is "
                f"further than round-off from the measured source's "
                f"{float(value)!r}")
            assert np.float32(pinned) == np.float32(value), (
                f"{name}.{part}: the two drivers no longer agree in "
                f"float32, which is the width the parity gate runs at")


# ---------------------------------------------------------------------------
# the joint head's depth is a declaration, not a hardcoded two
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", (1, 2))
@pytest.mark.parametrize("depth,n_linear", [(1, 2), (2, 3), (4, 5)])
def test_the_joint_net_is_as_deep_as_it_is_declared(n_species, depth,
                                                     n_linear):
    """A real published model needs a depth this class used to hardcode.

    The second system's published model is `f_exc_form="joint"` with FOUR
    hidden layers -- `u_joint_net.net` carries Linear layers at 0, 2, 4, 6
    and 8, nine tensors -- while this class built two, five tensors. A
    `strict=True` load of that published model was impossible, which is a blocker,
    not a stylistic gap.

    It was not the ONLY blocker: the same checkpoint records
    `fexc_input_scale: True` and `m_input_scale: True`, so a net of the
    right shape still computed a different function until `FLocal`'s
    `input_scale` and `Mobility`'s `input_ref` existed (the tests below).
    """
    net = JointEnergy(n_species, hidden=8, depth=depth)
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == n_linear
    assert linears[0].in_features == n_species + 1
    assert linears[-1].out_features == 1
    assert linears[-1].bias is None
    for mid in linears[1:-1]:
        assert mid.in_features == mid.out_features == 8
    # it still runs
    rho = torch.rand(5, n_species, dtype=torch.float64)
    out = net.double()(rho, torch.full((5,), 0.7, dtype=torch.float64))
    assert out.shape == (5,)


def test_the_default_depth_is_what_every_earlier_model_was_built_with():
    """Two, so no checkpoint written before this parameter existed changes
    shape. A system needing another depth declares it."""
    import inspect
    assert inspect.signature(JointEnergy).parameters["depth"].default == 2
    assert len([m for m in JointEnergy(2, hidden=4).net
                if isinstance(m, torch.nn.Linear)]) == 3


def test_a_joint_net_with_no_hidden_layer_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        JointEnergy(2, hidden=4, depth=0)


# ---------------------------------------------------------------------------
# input scaling: z = rho / rho_ref - 1 into every excess net, never the ideal term
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overrides", [
    dict(f_exc_form="joint", h_joint=6),
    dict(f_exc_form="split", u_form="mlp", g_exc_form="icnn"),
    dict(f_exc_form="split", u_form="icnn", g_exc_form="mlp",
         enable_TlnT=True, h_g_hat=5),
])
def test_input_scale_feeds_the_excess_nets_the_scaled_density(overrides):
    """Same weights, scaled input: the scaled model at ``rho`` is the plain one at ``z``,
    up to the ideal term, which keeps ``rho``."""
    rho_ref = [0.05, 0.04]
    kwargs = dict(n_species=2, rho_ref=rho_ref, kBT_ref=1.0, h_u=6, h_g=6,
                  ideal_form="gas", **overrides)
    torch.manual_seed(0)
    plain = FLocal(**kwargs).double()
    scaled = FLocal(**kwargs, input_scale=True).double()
    scaled.load_state_dict(plain.state_dict(), strict=True)   # a non-persistent buffer
    rho = torch.tensor([[0.06, 0.05], [0.03, 0.02]], dtype=torch.float64)
    kBT = torch.tensor([0.9, 1.3], dtype=torch.float64)
    z = rho / torch.tensor(rho_ref, dtype=torch.float32).double() - 1.0
    excess = (plain.f_pointwise(z, kBT) - plain._f_id(z, kBT))
    assert torch.allclose(scaled.f_pointwise(rho, kBT),
                          scaled._f_id(rho, kBT) + excess, rtol=1e-12)
    # mu is the gradient of that f, the chain rule's 1/rho_ref included
    leaf = rho.clone().requires_grad_(True)
    grad = torch.autograd.grad(scaled.f_pointwise(leaf, kBT).sum(), leaf)[0]
    assert torch.allclose(scaled.mu_pointwise(rho, kBT), grad, rtol=1e-10)


def test_input_scale_with_a_taylor_energy_is_refused():
    with pytest.raises(ValueError, match="taylor"):
        FLocal(2, [0.3, 0.3], kBT_ref=1.0, h_u=4, h_g=4, ideal_form="gas",
               u_form="taylor", u_degree=4, input_scale=True)


def test_input_scale_with_a_gauge_fixed_energy_is_refused():
    with pytest.raises(NotImplementedError, match="gauge_fix"):
        FLocal(2, [0.3, 0.3], kBT_ref=1.0, h_u=4, h_g=4, ideal_form="gas",
               gauge_fix=True, input_scale=True)
