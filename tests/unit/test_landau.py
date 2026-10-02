"""Tests for aipf.functional.landau: closed-form basis expansions.

Rung 1 ships implemented and tested but UNTRAINED: these
tests check the analytic free energy at known points, that the analytic
``mu = df/drho`` this module hand-derives matches an independent autograd
differentiation of its own ``bulk_free_energy_density``, the ``landau``
symmetry ``f(psi) == f(-psi)``, and that the model protocol check
(``check_protocol`` / registration) passes, tested at ``n_species`` 1 and 2.
"""
from __future__ import annotations

import math

import pytest
import torch

from aipf.functional.base import FreeEnergyModel, check_protocol
from aipf.functional.landau import MODEL_REGISTRY, FloryHuggins, Landau, _RedlichKister

GRID = (4, 4, 4)


def _T(batch, value=1.0):
    return torch.full((batch,), float(value), dtype=torch.float64)


def _boxes(batch):
    return torch.full((batch, 3), 10.0, dtype=torch.float64)


def _rho(batch, n_species, low=0.2, high=0.6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (low + (high - low) * torch.rand(
        batch, n_species, 1, 1, 1, generator=g, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Landau: analytic value at known points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_landau_bulk_free_energy_at_a_known_point(n_species):
    """rho = 0.5 -> psi = 0 -> f = 0 exactly, at any a0/b/Tc/T."""
    model = Landau(GRID, n_species, a0=2.0, b=3.0, T_c=1.5, gamma=1.0, nyquist_mask=True).double()
    rho = torch.full((2, n_species, 1, 1, 1), 0.5, dtype=torch.float64)
    T = _T(2, 4.0)
    f = model.bulk_free_energy_density(rho, T)
    assert f.shape == (2, 1, 1, 1, 1)
    assert torch.allclose(f, torch.zeros_like(f), atol=1e-12)


@pytest.mark.parametrize("n_species", [1, 2])
def test_landau_bulk_free_energy_matches_hand_computed_value(n_species):
    """rho = 1.0 -> psi = 1 -> f_s = (1/8)a0(T-Tc) + b/16, summed over species."""
    a0, b, T_c, T_val = 2.0, 3.0, 1.5, 4.0
    model = Landau(GRID, n_species, a0=a0, b=b, T_c=T_c, gamma=1.0, nyquist_mask=True).double()
    rho = torch.ones((1, n_species, 1, 1, 1), dtype=torch.float64)
    T = _T(1, T_val)
    f = model.bulk_free_energy_density(rho, T)
    per_species = 0.125 * a0 * (T_val - T_c) + b / 16.0
    expected = n_species * per_species
    assert torch.allclose(f, torch.full_like(f, expected))


# ---------------------------------------------------------------------------
# FH: analytic value at known points
# ---------------------------------------------------------------------------

def test_fh_one_species_bulk_free_energy_matches_hand_computed_value():
    w, T_val = 2.0, 1.3
    model = FloryHuggins(GRID, 1, gamma=1.0, w=w, nyquist_mask=True).double()
    rho = torch.full((1, 1, 1, 1, 1), 0.25, dtype=torch.float64)
    T = _T(1, T_val)
    f = model.bulk_free_energy_density(rho, T)
    expected = (w * 0.25 * 0.75
                + T_val * (0.25 * math.log(0.25) + 0.75 * math.log(0.75)))
    assert torch.allclose(f, torch.full_like(f, expected))


def test_fh_two_species_enthalpy_matches_hand_computed_polynomial():
    """degree=2, single nonzero coefficient on (1,1): u = c * z0 * z1."""
    model = FloryHuggins(GRID, 2, gamma=1.0, rho_ref=(0.4, 0.3), degree=2, nyquist_mask=True).double()
    with torch.no_grad():
        model.rk.coefs.zero_()
        idx = (model.rk.exps == torch.tensor([1, 1])).all(dim=-1).nonzero()[0, 0]
        model.rk.coefs[idx] = 5.0
    rho = torch.tensor([[0.6, 0.45]], dtype=torch.float64).view(1, 2, 1, 1, 1)
    T = _T(1, 0.0)  # kill the entropy term so only the enthalpy shows
    f = model.bulk_free_energy_density(rho, T)
    z0 = 0.6 / 0.4 - 1.0
    z1 = 0.45 / 0.3 - 1.0
    expected = 5.0 * z0 * z1
    assert torch.allclose(f, torch.full_like(f, expected), atol=1e-10)


def test_fh_two_species_reduces_the_identity_monomial_to_a_constant():
    """The (0,0) exponent tuple is always present: its coefficient is a bare
    constant offset, independent of rho."""
    model = FloryHuggins(GRID, 2, gamma=1.0, rho_ref=(0.4, 0.3), degree=2, nyquist_mask=True).double()
    with torch.no_grad():
        model.rk.coefs.zero_()
        model.rk.coefs[0] = 7.0  # exps[0] == (0, 0) by construction
    assert tuple(model.rk.exps[0].tolist()) == (0, 0)
    T = _T(1, 0.0)
    for rho_val in (0.1, 0.9):
        rho = torch.full((1, 2, 1, 1, 1), rho_val, dtype=torch.float64)
        f = model.bulk_free_energy_density(rho, T)
        assert torch.allclose(f, torch.full_like(f, 7.0), atol=1e-10)


# ---------------------------------------------------------------------------
# mu = df/drho, checked against autograd -- landau and fh, n_species 1 and 2
# ---------------------------------------------------------------------------

def _assert_mu_matches_autograd(model, rho, boxes, T):
    rho = rho.clone().requires_grad_(True)
    f = model.bulk_free_energy_density(rho, T)
    (grad,) = torch.autograd.grad(f.sum(), rho)
    mu = model.chemical_potential(rho.detach(), boxes, T)
    assert torch.allclose(mu, grad, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("n_species", [1, 2])
def test_landau_mu_matches_autograd(n_species):
    model = Landau(GRID, n_species, a0=1.7, b=2.3, T_c=1.1, gamma=1.0, nyquist_mask=True).double()
    rho = _rho(3, n_species, seed=10)
    _assert_mu_matches_autograd(model, rho, _boxes(3), _T(3, 2.0))


def test_fh_one_species_mu_matches_autograd():
    model = FloryHuggins(GRID, 1, gamma=1.0, w=1.4, nyquist_mask=True).double()
    rho = _rho(3, 1, seed=11)
    _assert_mu_matches_autograd(model, rho, _boxes(3), _T(3, 1.6))


def test_fh_two_species_mu_matches_autograd():
    model = FloryHuggins(GRID, 2, gamma=1.0, rho_ref=(0.4, 0.3), degree=4, nyquist_mask=True).double()
    torch.manual_seed(0)
    model.rk.coefs.data = torch.randn_like(model.rk.coefs.data)
    rho = _rho(3, 2, seed=12)
    _assert_mu_matches_autograd(model, rho, _boxes(3), _T(3, 1.9))


def test_fh_two_species_mu_matches_autograd_at_higher_degree():
    """degree=5 exercises an odd top degree and a longer exponent list."""
    model = FloryHuggins(GRID, 2, gamma=1.0, rho_ref=(0.5, 0.5), degree=5, nyquist_mask=True).double()
    torch.manual_seed(1)
    model.rk.coefs.data = torch.randn_like(model.rk.coefs.data)
    rho = _rho(2, 2, seed=13)
    _assert_mu_matches_autograd(model, rho, _boxes(2), _T(2, 2.2))


def test_redlich_kister_three_species_mu_matches_autograd():
    """The exponent-tuple machinery is not special-cased to n_species == 2."""
    model = FloryHuggins(
        GRID, 3, gamma=1.0, rho_ref=(0.3, 0.4, 0.5), degree=3, nyquist_mask=True).double()
    torch.manual_seed(2)
    model.rk.coefs.data = torch.randn_like(model.rk.coefs.data)
    rho = _rho(2, 3, low=0.1, high=0.7, seed=14)
    _assert_mu_matches_autograd(model, rho, _boxes(2), _T(2, 1.4))


# ---------------------------------------------------------------------------
# landau symmetry: f(psi) == f(-psi), i.e. f(rho) == f(1 - rho)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_landau_symmetry_f_of_psi_equals_f_of_minus_psi(n_species):
    model = Landau(GRID, n_species, a0=1.3, b=0.7, T_c=2.0, gamma=1.0, nyquist_mask=True).double()
    rho = _rho(4, n_species, low=0.05, high=0.95, seed=20)
    T = _T(4, 3.3)
    f_pos = model.bulk_free_energy_density(rho, T)
    f_neg = model.bulk_free_energy_density(1.0 - rho, T)
    assert torch.allclose(f_pos, f_neg, atol=1e-10)


# ---------------------------------------------------------------------------
# the model protocol check passes
# ---------------------------------------------------------------------------

def test_landau_passes_check_protocol():
    check_protocol(Landau)


def test_fh_passes_check_protocol():
    check_protocol(FloryHuggins)


def test_landau_and_fh_are_registered_and_conform_at_runtime():
    assert "landau" in MODEL_REGISTRY
    assert "fh" in MODEL_REGISTRY
    # Containment, not equality: the registry is shared across all rungs,
    # so its exact contents depend on which modules the test session has
    # imported -- an ordering dependency, not a property of this rung.
    assert {"fh", "landau"} <= set(MODEL_REGISTRY.names)
    landau = MODEL_REGISTRY.build(
        "landau", grid=GRID, n_species=1, a0=1.0, b=1.0, T_c=1.0, gamma=1.0,
        nyquist_mask=True)
    fh = MODEL_REGISTRY.build(
        "fh", grid=GRID, n_species=1, gamma=1.0, w=1.0, nyquist_mask=True)
    assert isinstance(landau, FreeEnergyModel)
    assert isinstance(fh, FreeEnergyModel)


def test_building_an_unregistered_form_raises():
    with pytest.raises(KeyError):
        MODEL_REGISTRY.build("nonexistent")


# ---------------------------------------------------------------------------
# forward: k=0 mass conservation, k-space in and out (protocol property)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_landau_forward_conserves_mass_at_k0(n_species):
    model = Landau((6, 5, 8), n_species, a0=1.0, b=1.0, T_c=1.4, gamma=1.0, nyquist_mask=True)
    boxes = torch.tensor([[10.0, 9.0, 8.0]] * 2)
    T = torch.full((2,), 1.6)
    torch.manual_seed(5)
    rho = 0.3 + 0.1 * torch.rand(2, n_species, 6, 5, 8)
    rho_hat = model.ops.rfft(rho) / (6 * 5 * 8)
    drho_hat_dt = model.forward(rho_hat, boxes, T)
    assert drho_hat_dt.is_complex()
    assert drho_hat_dt.shape == rho_hat.shape
    k0 = drho_hat_dt[:, :, 0, 0, 0]
    assert torch.allclose(k0.real, torch.zeros_like(k0.real), atol=1e-10)
    assert torch.allclose(k0.imag, torch.zeros_like(k0.imag), atol=1e-10)


@pytest.mark.parametrize("n_species", [1, 2])
def test_fh_forward_conserves_mass_at_k0(n_species):
    kwargs = (dict(w=1.0) if n_species == 1
              else dict(rho_ref=tuple(0.4 for _ in range(n_species))))
    model = FloryHuggins((6, 5, 8), n_species, gamma=1.0, **kwargs, nyquist_mask=True)
    boxes = torch.tensor([[10.0, 9.0, 8.0]] * 2)
    T = torch.full((2,), 1.6)
    torch.manual_seed(6)
    rho = 0.3 + 0.1 * torch.rand(2, n_species, 6, 5, 8)
    rho_hat = model.ops.rfft(rho) / (6 * 5 * 8)
    drho_hat_dt = model.forward(rho_hat, boxes, T)
    k0 = drho_hat_dt[:, :, 0, 0, 0]
    assert torch.allclose(k0.real, torch.zeros_like(k0.real), atol=1e-8)
    assert torch.allclose(k0.imag, torch.zeros_like(k0.imag), atol=1e-8)


# ---------------------------------------------------------------------------
# mobility: the value, not just its shape -- forward's mass conservation at
# k=0 holds for ANY mobility (it is a property of the divergence form, not
# of M), so it cannot by itself tell a correct mobility from a wrong scale
# or a dropped degeneracy factor. These pin the value directly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_landau_mobility_matches_hand_computed_value(n_species):
    model = Landau(GRID, n_species, a0=1.0, b=1.0, T_c=1.0, gamma=2.5, nyquist_mask=True).double()
    rho = torch.full((1, n_species, 1, 1, 1), 0.3, dtype=torch.float64)
    M = model.mobility(rho, _T(1))
    expected = 2.5 * 0.3 * 0.7
    assert torch.allclose(M, torch.full_like(M, expected))


@pytest.mark.parametrize("n_species", [1, 2])
def test_fh_mobility_matches_hand_computed_value(n_species):
    kwargs = (dict(w=1.0) if n_species == 1
              else dict(rho_ref=tuple(0.4 for _ in range(n_species))))
    model = FloryHuggins(GRID, n_species, gamma=3.0, **kwargs, nyquist_mask=True).double()
    rho = torch.full((1, n_species, 1, 1, 1), 0.2, dtype=torch.float64)
    M = model.mobility(rho, _T(1))
    expected = 3.0 * 0.2 * 0.8
    assert torch.allclose(M, torch.full_like(M, expected))


# ---------------------------------------------------------------------------
# forward against an independent linearized-dispersion calculation.
#
# k=0 mass conservation holds for ANY divergence form regardless of sign or
# an overall scale error (0 * anything is still 0), so it cannot catch a
# dropped "-" on the flux or a dropped "/N" spectral normalisation -- both
# real lines in `_BasisExpansionRung.forward`. This test can: for a small
# single-mode perturbation rho0 + amplitude*cos(k0 x) about a uniform
# background, d(rho)/dt = div(M grad mu) linearizes to
# -M(rho0) * mu'(rho0) * k0^2 * amplitude * cos(k0 x) to leading order in
# amplitude. mu'(rho0) and M(rho0) are obtained independently -- a finite
# difference of `chemical_potential` (itself already checked against
# autograd above) and a direct call to `mobility` (checked against a hand
# value above) -- so this exercises the actual N-scaling/grad_hat/div_hat
# pipeline against a calculation that never goes through `forward` itself.
# ---------------------------------------------------------------------------

def _assert_forward_matches_linearized_dispersion(model, grid, rho0, T_val,
                                                   amplitude=1e-4, mode=2):
    Gx, Gy, Gz = grid
    Lx = 10.0
    boxes = torch.tensor([[Lx, 8.0, 8.0]], dtype=torch.float64)
    T = torch.full((1,), T_val, dtype=torch.float64)
    k0 = 2.0 * math.pi * mode / Lx
    x = torch.arange(Gx, dtype=torch.float64) * (Lx / Gx)
    rho_real = rho0 + amplitude * torch.cos(k0 * x)
    rho = rho_real.view(1, 1, Gx, 1, 1).expand(
        1, model.n_species, Gx, Gy, Gz).clone()
    N = Gx * Gy * Gz
    rho_hat = model.ops.rfft(rho) / N
    drho_hat_dt = model.forward(rho_hat, boxes, T)
    drho_dt = model.ops.irfft(drho_hat_dt * N)

    delta = 1e-4
    rho_bg = torch.full((1, model.n_species, 1, 1, 1), rho0, dtype=torch.float64)
    rho_plus = torch.full((1, model.n_species, 1, 1, 1), rho0 + delta, dtype=torch.float64)
    rho_minus = torch.full((1, model.n_species, 1, 1, 1), rho0 - delta, dtype=torch.float64)
    mu_plus = model.chemical_potential(rho_plus, boxes[:1], T)
    mu_minus = model.chemical_potential(rho_minus, boxes[:1], T)
    mu_prime = ((mu_plus - mu_minus) / (2.0 * delta)).flatten()[0]
    M_bg = model.mobility(rho_bg, T).flatten()[0]

    expected_1d = -M_bg * mu_prime * (k0 ** 2) * amplitude * torch.cos(k0 * x)
    expected = expected_1d.view(1, 1, Gx, 1, 1).expand(
        1, model.n_species, Gx, Gy, Gz)
    assert torch.allclose(drho_dt, expected, atol=1e-6)


def test_landau_forward_matches_linearized_dispersion():
    model = Landau((16, 4, 4), 1, a0=1.3, b=0.9, T_c=1.1, gamma=2.0, nyquist_mask=True).double()
    _assert_forward_matches_linearized_dispersion(model, (16, 4, 4), 0.4, 1.7)


def test_fh_forward_matches_linearized_dispersion():
    model = FloryHuggins((16, 4, 4), 1, gamma=1.5, w=1.2, nyquist_mask=True).double()
    _assert_forward_matches_linearized_dispersion(model, (16, 4, 4), 0.35, 1.4)


# ---------------------------------------------------------------------------
# no default hides a physical constant
# ---------------------------------------------------------------------------

def test_fh_one_species_without_w_raises():
    with pytest.raises(ValueError, match="w"):
        FloryHuggins(GRID, 1, gamma=1.0, nyquist_mask=True)


def test_fh_multi_species_without_rho_ref_raises():
    with pytest.raises(ValueError, match="rho_ref"):
        FloryHuggins(GRID, 2, gamma=1.0, nyquist_mask=True)


def test_redlich_kister_rejects_a_mismatched_rho_ref_length():
    with pytest.raises(ValueError, match="rho_ref"):
        _RedlichKister(2, (0.4,), degree=4)


def test_redlich_kister_rejects_a_degree_below_two():
    with pytest.raises(ValueError, match="degree"):
        _RedlichKister(2, (0.4, 0.3), degree=1)


def test_redlich_kister_two_species_matches_taylor_u_exponent_order():
    """Reproduces the published Taylor form's own exponent list construction
    (degree=4) directly, so a change to _compositions' enumeration order
    would be caught here even though nothing else pins the order."""
    rk = _RedlichKister(2, (0.35, 0.33), degree=4)
    want = [(0, 0)] + [(i, tot - i) for tot in range(2, 5) for i in range(tot + 1)]
    assert [tuple(row.tolist()) for row in rk.exps] == want
