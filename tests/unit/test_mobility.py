"""Tests for `aipf.mobility`: the mobility form registry.

Every property test runs at `n_species = 1` **and** `n_species = 2`, per the
project's global constraint that the single-species case is one of the
three systems, not a degenerate afterthought.
"""
from __future__ import annotations

import pytest
import torch

from aipf.mobility import (
    PREFACTOR_FORMS,
    SHAPE_FORMS,
    T_FORMS,
    Mobility,
    cholesky_raw_to_matrix,
    mobility_prefactor,
)


def _rho(n_species: int, batch: int = 7, lo: float = 0.05, hi: float = 0.95,
         seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(batch, n_species, generator=g)


def _T(batch: int = 7, lo: float = 1.0, hi: float = 3.0, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(batch, generator=g)


def _constant_model(n_species: int, prefactor: str = "partial_density",
                     raw=None) -> Mobility:
    n_chol = n_species * (n_species + 1) // 2
    if raw is None:
        g = torch.Generator().manual_seed(42)
        raw = torch.randn(n_chol, generator=g)
    return Mobility(n_species, prefactor=prefactor, shape="constant",
                     t_form="none", shape_init=raw)


# ---------------------------------------------------------------------------
# Cholesky construction: symmetric PSD for arbitrary parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_cholesky_raw_to_matrix_is_symmetric_for_arbitrary_raw(n_species):
    g = torch.Generator().manual_seed(3)
    n_chol = n_species * (n_species + 1) // 2
    raw = torch.randn(11, n_chol, generator=g) * 5.0  # arbitrary, incl. large
    M = cholesky_raw_to_matrix(raw, n_species)
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-6)


@pytest.mark.parametrize("n_species", [1, 2])
def test_cholesky_raw_to_matrix_is_psd_for_arbitrary_raw(n_species):
    g = torch.Generator().manual_seed(4)
    n_chol = n_species * (n_species + 1) // 2
    raw = torch.randn(23, n_chol, generator=g) * 5.0
    M = cholesky_raw_to_matrix(raw, n_species)
    eigvals = torch.linalg.eigvalsh(M)
    assert torch.all(eigvals >= -1e-5)


def test_cholesky_off_diagonal_is_free_sign():
    """l21 (the cross term) must be reachable with either sign: the sign of
    the measured cross mobility is DATA, and the registry must not constrain
    it (spec: neither system's measured cross term is a form choice)."""
    raw_pos = torch.tensor([0.0, 3.0, 0.0])   # l21 = +3
    raw_neg = torch.tensor([0.0, -3.0, 0.0])  # l21 = -3
    M_pos = cholesky_raw_to_matrix(raw_pos, 2)
    M_neg = cholesky_raw_to_matrix(raw_neg, 2)
    assert M_pos[0, 1] > 0
    assert M_neg[0, 1] < 0
    # Both remain valid (symmetric, PSD) regardless of the cross term's sign.
    for M in (M_pos, M_neg):
        assert torch.allclose(M, M.T, atol=1e-6)
        assert torch.all(torch.linalg.eigvalsh(M) >= -1e-6)


def test_cholesky_raw_to_matrix_rejects_wrong_parameter_count():
    with pytest.raises(ValueError):
        cholesky_raw_to_matrix(torch.zeros(2), n_species=2)  # needs 3


# ---------------------------------------------------------------------------
# prefactor axis
# ---------------------------------------------------------------------------

def test_one_species_mole_fraction_prefactor_reduces_to_rho_times_one_minus_rho():
    rho = torch.tensor([[0.0], [0.2], [0.5], [0.9], [1.0]])
    g = mobility_prefactor(rho, "mole_fraction")
    expected = rho * (1.0 - rho)
    assert torch.allclose(g, expected, atol=1e-7)


@pytest.mark.parametrize("n_species", [1, 2])
def test_partial_density_prefactor_is_rho_itself(n_species):
    rho = _rho(n_species)
    g = mobility_prefactor(rho, "partial_density")
    assert torch.allclose(g, rho.clamp(min=0.0), atol=1e-7)


def test_unknown_prefactor_form_raises():
    with pytest.raises(ValueError):
        mobility_prefactor(torch.ones(3, 2), "not_a_real_form")


@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("form", PREFACTOR_FORMS)
def test_prefactor_vanishes_as_a_species_vanishes(n_species, form):
    """The prefactor's defining property: zero density in one channel must
    zero every matrix entry that channel touches."""
    model = _constant_model(n_species, prefactor=form)
    T = _T(batch=5)
    rho = _rho(n_species, batch=5)
    for i in range(n_species):
        rho_i = rho.clone()
        rho_i[:, i] = 0.0
        M = model(rho_i, T)
        assert torch.allclose(M[:, i, :], torch.zeros_like(M[:, i, :]), atol=1e-6)
        assert torch.allclose(M[:, :, i], torch.zeros_like(M[:, :, i]), atol=1e-6)


# ---------------------------------------------------------------------------
# The assembled model: symmetric PSD at both species counts, every shape form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("prefactor", PREFACTOR_FORMS)
@pytest.mark.parametrize("shape", ["constant", "mlp_rho", "mlp_rho_t"])
def test_mobility_is_symmetric_psd_for_arbitrary_parameters(n_species, prefactor, shape):
    n_chol = n_species * (n_species + 1) // 2
    kwargs = dict(prefactor=prefactor, shape=shape, t_form="none")
    if shape == "constant":
        g = torch.Generator().manual_seed(9)
        kwargs["shape_init"] = torch.randn(n_chol, generator=g) * 4.0
    model = Mobility(n_species, **kwargs)
    # Arbitrary parameter values: perturb every trainable parameter randomly.
    g = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 3.0)

    rho = _rho(n_species, batch=13)
    T = _T(batch=13)
    M = model(rho, T)

    assert M.shape == (13, n_species, n_species)
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-5)
    eigvals = torch.linalg.eigvalsh(M)
    assert torch.all(eigvals >= -1e-4)


@pytest.mark.parametrize("n_species", [1, 2])
def test_mobility_fixed_shape_is_not_trainable(n_species):
    n_chol = n_species * (n_species + 1) // 2
    raw = torch.zeros(n_chol)
    model = Mobility(n_species, prefactor="partial_density", shape="fixed",
                      t_form="none", shape_init=raw)
    assert list(model.parameters()) == []


@pytest.mark.parametrize("n_species", [1, 2])
def test_mobility_with_arrhenius_t_form_is_symmetric_psd(n_species):
    model = Mobility(
        n_species, prefactor="partial_density", shape="constant", t_form="arrhenius",
        shape_init=torch.zeros(n_species * (n_species + 1) // 2),
        t_ref=2.0, activation_energy_init=[0.1] * n_species, kB=1.0,
    )
    rho = _rho(n_species, batch=9)
    T = _T(batch=9, lo=0.5, hi=5.0)
    M = model(rho, T)
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-5)
    assert torch.all(torch.linalg.eigvalsh(M) >= -1e-5)


def test_mobility_one_species_constant_partial_density_reduces_to_rho_scaled_by_m_tilde():
    """At n_species = 1, partial_density gives M = rho * M_tilde exactly."""
    raw = torch.tensor([0.3])  # softplus(0.3) ** 2 == M_tilde scalar
    model = Mobility(1, prefactor="partial_density", shape="constant",
                      t_form="none", shape_init=raw)
    rho = _rho(1, batch=6)
    T = _T(batch=6)
    M = model(rho, T)
    m_tilde = cholesky_raw_to_matrix(raw, 1)  # (1, 1)
    expected = rho.unsqueeze(-1) * m_tilde
    assert torch.allclose(M, expected, atol=1e-6)


def test_mobility_one_species_constant_mole_fraction_reduces_to_rho_one_minus_rho_times_m_tilde():
    raw = torch.tensor([0.3])
    model = Mobility(1, prefactor="mole_fraction", shape="constant",
                      t_form="none", shape_init=raw)
    rho = _rho(1, batch=6)
    T = _T(batch=6)
    M = model(rho, T)
    m_tilde = cholesky_raw_to_matrix(raw, 1)
    expected = (rho * (1.0 - rho)).unsqueeze(-1) * m_tilde
    assert torch.allclose(M, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Unknown form names raise, at construction, for every axis
# ---------------------------------------------------------------------------

def test_unknown_prefactor_raises_at_construction():
    with pytest.raises(ValueError):
        Mobility(2, prefactor="not_a_real_form", shape="mlp_rho", t_form="none")


def test_unknown_shape_raises_at_construction():
    with pytest.raises(ValueError):
        Mobility(2, prefactor="partial_density", shape="not_a_real_form", t_form="none")


def test_unknown_t_form_raises_at_construction():
    with pytest.raises(ValueError):
        Mobility(2, prefactor="partial_density", shape="mlp_rho", t_form="not_a_real_form")


def test_shape_that_consumes_t_directly_rejects_a_nontrivial_t_form():
    with pytest.raises(ValueError):
        Mobility(2, prefactor="partial_density", shape="mlp_rho_t", t_form="arrhenius",
                  t_ref=2.0, activation_energy_init=[0.1, 0.1])


# ---------------------------------------------------------------------------
# No default carries a measured system's numbers
# ---------------------------------------------------------------------------

def test_constant_shape_requires_an_explicit_init():
    with pytest.raises(ValueError):
        Mobility(2, prefactor="partial_density", shape="constant", t_form="none")


def test_fixed_shape_requires_an_explicit_init():
    with pytest.raises(ValueError):
        Mobility(2, prefactor="partial_density", shape="fixed", t_form="none")


def test_arrhenius_requires_an_explicit_t_ref_and_activation_energy():
    with pytest.raises(ValueError):
        Mobility(2, prefactor="partial_density", shape="constant", t_form="arrhenius",
                  shape_init=torch.zeros(3))  # no t_ref, no activation_energy_init


def test_arrhenius_requires_the_barriers_energy_unit():
    """The barrier is an energy, so ``1/T`` alone would read it in kelvin."""
    with pytest.raises(ValueError, match="kB"):
        Mobility(2, prefactor="partial_density", shape="constant",
                 t_form="arrhenius", shape_init=torch.zeros(3), t_ref=2.0,
                 activation_energy_init=[0.1, 0.1])


def test_the_arrhenius_factor_reads_its_barrier_in_the_declared_unit():
    """``M(T) / M(T_ref) = exp(-E (1/(kB T) - 1/(kB T_ref)))`` per channel pair, E = softplus(raw)."""
    kB, t_ref, T = 8.617333262e-5, 1800.0, 1300.0
    raw = [0.2, -0.4]
    model = Mobility(2, prefactor="partial_density", shape="constant",
                     t_form="arrhenius",
                     shape_init=torch.tensor([0.3, 0.2, 0.1]),
                     t_ref=t_ref, activation_energy_init=raw, kB=kB)
    rho = torch.tensor([[0.06, 0.05]], dtype=torch.float64)
    model = model.double()
    ratio = (model(rho, torch.tensor([T], dtype=torch.float64))
             / model(rho, torch.tensor([t_ref], dtype=torch.float64)))[0]
    E = torch.nn.functional.softplus(torch.tensor(raw, dtype=torch.float64))
    c = torch.exp(-E * (1.0 / (kB * T) - 1.0 / (kB * t_ref)))
    expected = torch.sqrt(c)[:, None] * torch.sqrt(c)[None, :]
    assert torch.allclose(ratio, expected, rtol=1e-12)


def test_the_input_reference_scales_only_the_nets_input():
    """``z = rho / input_ref - 1`` into the net; the ``sqrt(g)`` sandwich keeps ``rho``."""
    torch.manual_seed(0)
    plain = Mobility(2, prefactor="partial_density", shape="mlp_rho",
                     t_form="none", hidden=4)
    scaled = Mobility(2, prefactor="partial_density", shape="mlp_rho",
                      t_form="none", hidden=4, input_ref=(0.05, 0.04))
    scaled.load_state_dict(plain.state_dict(), strict=True)  # non-persistent
    rho = torch.tensor([[0.06, 0.05], [0.03, 0.02]])
    T = torch.ones(2)
    z = rho / torch.tensor([0.05, 0.04]) - 1.0
    s = torch.sqrt(rho)
    expected = (cholesky_raw_to_matrix(plain.m_net(z), 2)
                * (s.unsqueeze(-1) * s.unsqueeze(-2)))
    assert torch.equal(scaled(rho, T), expected)
    assert not torch.equal(scaled(rho, T), plain(rho, T))


def test_an_input_reference_without_a_net_is_refused():
    with pytest.raises(ValueError, match="input_ref"):
        Mobility(2, prefactor="partial_density", shape="constant",
                 t_form="none", shape_init=torch.zeros(3),
                 input_ref=(0.05, 0.05))


def test_forms_are_declared_tuples_not_open_ended():
    """Guards against a future edit that silently widens a bare string into
    an accepted form without updating the validated tuple."""
    assert PREFACTOR_FORMS == ("mole_fraction", "partial_density")
    assert SHAPE_FORMS == ("fixed", "constant", "mlp_rho", "mlp_rho_t",
                           "lattice_scalar")
    assert T_FORMS == ("none", "arrhenius")
