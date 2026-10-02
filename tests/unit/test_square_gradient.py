"""Tests for aipf.functional.square_gradient: the trainable square-gradient rung.

The three required bites: `mu` contains `-kappa
laplacian(rho)`, checked against a known sinusoid whose Laplacian is
analytic; `kappa` stays symmetric positive-definite for arbitrary
parameter values; the `n_species = 1` case reduces to a scalar. A fourth,
"the protocol check passes", is also required. Everything else here
(mass conservation, the gradient-energy diagnostic, the
`laplacian`-vs-`div_hat(grad_hat(...))` agreement on band-limited data)
exists because reading the finished module top to bottom turned up more
rules than those three, and a rule with no test is a rule a mutation can
silently break.
"""
from __future__ import annotations

import math

import numpy as np

import pytest
import torch

from aipf.functional.base import FreeEnergyModel, ModelRegistry, check_protocol
from aipf.functional.square_gradient import SquareGradient, _cholesky_to_matrix
from aipf.spectral import OpsCache

GRID = (8, 6, 8)  # even on every axis; Gzr = 5


#: The untrained polynomial form at a unit starting kappa (both declared: neither has a default).
POLY = dict(local="quadratic_quartic", kappa_init=1.0)

def _boxes(batch: int, grid=GRID):
    """Boxes whose length on each axis equals THAT axis's grid count --
    unit lattice spacing, so `k = 2*pi*n/L` reduces to `2*pi*n/G` and a
    single-mode sinusoid with an integer number of periods is exactly
    band-limited (no aliasing) regardless of which mode is chosen. Per-axis,
    not a single scalar broadcast: the grid here is not cubic (`GRID =
    (8, 6, 8)`), so a box that used `grid[0]` for every axis would silently
    mismatch the y axis.
    """
    L = torch.tensor(grid, dtype=torch.float32)
    return L.unsqueeze(0).expand(batch, 3).clone()


def _coords(grid):
    """`(x, y, z)`, each shape `grid`, unit lattice spacing."""
    Gx, Gy, Gz = grid
    x = torch.arange(Gx, dtype=torch.float32).view(Gx, 1, 1)
    y = torch.arange(Gy, dtype=torch.float32).view(1, Gy, 1)
    z = torch.arange(Gz, dtype=torch.float32).view(1, 1, Gz)
    return x.expand(grid), y.expand(grid), z.expand(grid)


def _zero_f_loc(model: SquareGradient) -> None:
    """Zero the local-free-energy parameters so `chemical_potential`
    isolates the gradient term -- the thing this task is responsible for.
    """
    with torch.no_grad():
        model._A_raw.zero_()
        # softplus(raw) == 0 only in the limit as raw -> -inf; -50 is finite
        # (no NaN) and softplus(-50) is small enough (~2e-22) that the
        # quartic term is negligible next to the gradient term's O(1)
        # magnitude, which is all this helper needs.
        model._b_raw.fill_(-50.0)


# ---------------------------------------------------------------------------
# mu contains -kappa @ laplacian(rho), checked against an analytic sinusoid
# ---------------------------------------------------------------------------

def test_mu_gradient_term_matches_analytic_laplacian_single_species():
    model = SquareGradient(GRID, n_species=1, nyquist_mask=True, **POLY)
    _zero_f_loc(model)
    with torch.no_grad():
        model._kappa_chol_raw.copy_(torch.tensor([0.3]))
    kappa = model.kappa.item()

    Gx = GRID[0]
    m = 2  # mode strictly below Nyquist (Gx/2 == 4)
    kx = 2.0 * math.pi * m / Gx
    A = 0.37
    x, _, _ = _coords(GRID)
    wave = A * torch.sin(kx * x)
    rho = (1.0 + wave).unsqueeze(0).unsqueeze(0)  # (1, 1, Gx, Gy, Gz)
    boxes = _boxes(1)

    mu = model.chemical_potential(rho, boxes, T=torch.tensor([1.0]))

    analytic_laplacian = -(kx ** 2) * wave  # d^2/dx^2 sin(kx x) = -kx^2 sin
    expected = -kappa * analytic_laplacian.unsqueeze(0).unsqueeze(0)
    torch.testing.assert_close(mu, expected, atol=1e-4, rtol=1e-4)


def test_mu_gradient_term_matches_analytic_laplacian_two_species_with_coupling():
    """Two species, each carrying a sinusoid along a DIFFERENT axis, and an
    off-diagonal kappa -- so a bug that drops the cross term (treats kappa
    as diagonal) is caught, not just a bug in the diagonal entries.
    """
    model = SquareGradient(GRID, n_species=2, nyquist_mask=True, **POLY)
    _zero_f_loc(model)
    with torch.no_grad():
        # raw = [L00_raw, L10, L11_raw]; nonzero L10 gives a nonzero M12.
        model._kappa_chol_raw.copy_(torch.tensor([0.2, 0.15, 0.1]))
    kappa = model.kappa  # (2, 2)

    Gx, Gy, _ = GRID
    m1, m2 = 1, 2
    kx = 2.0 * math.pi * m1 / Gx
    ky = 2.0 * math.pi * m2 / Gy
    A1, A2 = 0.4, 0.25
    x, y, _ = _coords(GRID)
    ripple1 = A1 * torch.sin(kx * x)
    ripple2 = A2 * torch.cos(ky * y)
    rho1 = 1.0 + ripple1
    rho2 = 0.8 + ripple2
    rho = torch.stack([rho1, rho2], dim=0).unsqueeze(0)  # (1, 2, Gx, Gy, Gz)
    boxes = _boxes(1)

    mu = model.chemical_potential(rho, boxes, T=torch.tensor([1.0]))

    lap1 = -(kx ** 2) * ripple1
    lap2 = -(ky ** 2) * ripple2
    lap = torch.stack([lap1, lap2], dim=0)  # (2, Gx, Gy, Gz)
    expected = -torch.einsum("ij,jxyz->ixyz", kappa, lap).unsqueeze(0)
    torch.testing.assert_close(mu, expected, atol=1e-4, rtol=1e-4)


def test_gradient_term_agrees_with_div_grad_composition_on_band_limited_data():
    """The docstring's claim: on band-limited data, `laplacian` and
    `div_hat(grad_hat(...))` agree tightly. This does not re-derive the
    9.1e-01 Nyquist-divergence measurement (that lives in
    `tests/unit/test_spectral.py`); it pins that THIS module's chosen
    operator is consistent with the alternative where both are valid, which
    is the regime training always runs in.
    """
    model = SquareGradient(GRID, n_species=2, nyquist_mask=True, **POLY)
    _zero_f_loc(model)
    torch.manual_seed(0)
    raw_rho = 0.5 + 0.05 * torch.randn(1, 2, *GRID)
    boxes = _boxes(1)
    ops = model.ops
    kx, ky, kz = ops.k_axes(boxes)

    # Band-limit the field FIRST (zero every axis's Nyquist-plane content;
    # GRID is even on all three axes, so all three planes exist), so both
    # routes below operate on exactly the same, genuinely Nyquist-free real
    # field -- comparing `laplacian` on the full field against
    # `div_hat(grad_hat(...))` on a truncated one would measure the
    # Nyquist-content DIFFERENCE already measured (9.1e-01), not
    # confirm the claimed agreement.
    rho_hat = ops.rfft(raw_rho)
    mx = ops.MX_ODD.view(-1) == 0.0
    my = ops.MY_ODD.view(-1) == 0.0
    mz = ops.MZ_ODD.view(-1) == 0.0
    if mx.any():
        rho_hat[:, :, mx, :, :] = 0.0
    if my.any():
        rho_hat[:, :, :, my, :] = 0.0
    if mz.any():
        rho_hat[:, :, :, :, mz] = 0.0
    rho = ops.irfft(rho_hat)

    mu_laplacian = model.chemical_potential(rho, boxes, T=torch.tensor([1.0]))

    rho_hat2 = ops.rfft(rho)  # re-transform the now band-limited real field
    gx_hat, gy_hat, gz_hat = ops.grad_hat(rho_hat2, kx, ky, kz)
    div_grad_hat = ops.div_hat(gx_hat, gy_hat, gz_hat, kx, ky, kz)
    lap_via_div_grad = ops.irfft(div_grad_hat)
    kappa = model.kappa
    mu_div_grad = -torch.einsum("ij,bjxyz->bixyz", kappa, lap_via_div_grad)

    torch.testing.assert_close(mu_laplacian, mu_div_grad, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# kappa: symmetric positive-definite by construction
# ---------------------------------------------------------------------------

def test_chemical_potential_gradient_term_uses_laplacian_not_div_grad_at_nyquist():
    """At a Nyquist-containing field, `laplacian` (even-order, unmasked) and
    `div_hat(grad_hat(...))` (odd-order, masked) disagree substantially
    (measured 9.1e-01 on such data) -- so this pins WHICH one
    `chemical_potential` actually uses, not merely that some Laplacian-
    shaped quantity was computed (the earlier band-limited test cannot tell
    the two apart, by construction).
    """
    model = SquareGradient(GRID, n_species=1, nyquist_mask=True, **POLY)
    _zero_f_loc(model)
    with torch.no_grad():
        model._kappa_chol_raw.copy_(torch.tensor([0.6]))
    kappa = model.kappa.item()

    torch.manual_seed(3)
    rho = 0.5 + 0.1 * torch.randn(1, 1, *GRID)  # full spectrum, incl. Nyquist
    boxes = _boxes(1)
    ops = model.ops
    kx, ky, kz = ops.k_axes(boxes)
    k2 = ops.k2(boxes)
    rho_hat = ops.rfft(rho)

    lap_via_laplacian = ops.irfft(ops.laplacian(rho_hat, k2))
    gx_hat, gy_hat, gz_hat = ops.grad_hat(rho_hat, kx, ky, kz)
    lap_via_div_grad = ops.irfft(
        ops.div_hat(gx_hat, gy_hat, gz_hat, kx, ky, kz))
    # Sanity: the two routes really do disagree on this data -- otherwise
    # the assertion below would pass under either implementation and prove
    # nothing about which one `chemical_potential` uses.
    assert not torch.allclose(lap_via_laplacian, lap_via_div_grad, atol=1e-3)

    mu = model.chemical_potential(rho, boxes, T=torch.tensor([1.0]))
    expected = -kappa * lap_via_laplacian
    torch.testing.assert_close(mu, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n_species", [1, 2, 3])
def test_kappa_is_symmetric_positive_definite_for_random_parameters(n_species):
    """`L`'s diagonal is always strictly positive (`softplus` never returns
    exactly 0), so `det(L)` -- the product of a triangular matrix's diagonal
    -- is always > 0 and `L` is always invertible, hence `L @ L.T` is always
    positive DEFINITE, in exact arithmetic, for every value of `raw`. Built
    and checked in float64: an extreme raw draw can make the diagonal tiny
    (`softplus` of a very negative number) next to a much larger
    off-diagonal, which is well-conditioned mathematically but ill-
    conditioned numerically -- float32 `eigvalsh` on such a matrix can
    return a small NEGATIVE eigenvalue purely from round-off (measured:
    -5e-07 on one float32 draw at n_species=2), which would make this test
    flag a construction that is not actually broken.
    """
    torch.manual_seed(1234)
    for _ in range(20):
        model = SquareGradient(GRID, n_species=n_species, nyquist_mask=True, **POLY).double()
        with torch.no_grad():
            model._kappa_chol_raw.copy_(
                5.0 * torch.randn_like(model._kappa_chol_raw))
        K = model._kappa_matrix()
        torch.testing.assert_close(K, K.transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(K)
        assert torch.all(eigvals > 0), eigvals


@pytest.mark.parametrize("n_species", [1, 2, 3])
def test_kappa_diagonal_is_built_from_softplus_not_the_raw_value_directly(n_species):
    """Pins the exact transform on the diagonal, not just its downstream PD
    consequence. `L @ L.T` is positive semi-definite for ANY real `L`
    regardless of the sign of its diagonal -- so a mutation that makes the
    diagonal free-sign (identity instead of softplus) does NOT reliably fail
    a random-parameter PD check (it only breaks at the measure-zero point
    where a raw diagonal entry is exactly 0, which `torch.randn` will not
    hit). This test targets that gap directly: at `raw = 0`, `kappa`'s
    diagonal must equal `softplus(0)**2 == log(2)**2`, not `0**2 == 0`.
    """
    model = SquareGradient(GRID, n_species=n_species, nyquist_mask=True, **POLY)
    with torch.no_grad():
        model._kappa_chol_raw.zero_()
    K = model._kappa_matrix()
    expected_diag = math.log(2.0) ** 2
    for i in range(n_species):
        assert K[i, i].item() == pytest.approx(expected_diag, rel=1e-5)


def test_kappa_diagonal_never_reaches_exactly_zero_even_at_extreme_raw():
    """Complements the `raw = 0` pin: a very negative raw diagonal still
    gives a strictly positive (if tiny) `kappa` diagonal entry, because
    `softplus` never returns exactly 0 for a finite input. A free-sign
    diagonal would instead pass a very negative raw straight through,
    producing a large NEGATIVE `kappa` diagonal entry.
    """
    model = SquareGradient(GRID, n_species=1, nyquist_mask=True, **POLY)
    with torch.no_grad():
        model._kappa_chol_raw.copy_(torch.tensor([-20.0]))
    K = model._kappa_matrix()
    assert K[0, 0].item() > 0.0


def test_cholesky_helper_rejects_the_wrong_parameter_count():
    with pytest.raises(ValueError):
        _cholesky_to_matrix(torch.zeros(2), n=2)  # needs 3 for n=2


# ---------------------------------------------------------------------------
# n_species == 1 reduces to a scalar
# ---------------------------------------------------------------------------

def test_kappa_is_a_bare_scalar_at_n_species_one():
    model = SquareGradient(GRID, n_species=1, nyquist_mask=True, **POLY)
    with torch.no_grad():
        model._kappa_chol_raw.copy_(torch.tensor([0.42]))
    kappa = model.kappa
    assert kappa.dim() == 0
    assert kappa.item() == pytest.approx(model._kappa_matrix()[0, 0].item())


def test_kappa_is_a_matrix_at_n_species_two():
    model = SquareGradient(GRID, n_species=2, nyquist_mask=True, **POLY)
    kappa = model.kappa
    assert kappa.dim() == 2
    assert kappa.shape == (2, 2)


# ---------------------------------------------------------------------------
# the protocol check passes
# ---------------------------------------------------------------------------

def test_square_gradient_satisfies_the_free_energy_model_protocol_at_registration():
    registry = ModelRegistry()
    registered = registry.register("square_gradient", SquareGradient)
    assert registered is SquareGradient
    assert "square_gradient" in registry


def test_square_gradient_check_protocol_does_not_raise():
    check_protocol(SquareGradient)  # would raise TypeError otherwise


def test_square_gradient_instance_passes_runtime_checkable_isinstance():
    model = SquareGradient(GRID, n_species=1, nyquist_mask=True, **POLY)
    assert isinstance(model, FreeEnergyModel)


def test_square_gradient_module_registers_itself_under_its_own_name():
    from aipf.functional.square_gradient import MODEL_REGISTRY
    assert "square_gradient" in MODEL_REGISTRY
    built = MODEL_REGISTRY.build("square_gradient", grid=GRID, n_species=2,
                                 nyquist_mask=True, **POLY)
    assert isinstance(built, SquareGradient)


# ---------------------------------------------------------------------------
# n_species < 1 is rejected
# ---------------------------------------------------------------------------

def test_n_species_below_one_raises():
    with pytest.raises(ValueError, match="n_species"):
        SquareGradient(GRID, n_species=0, nyquist_mask=True, **POLY)


# ---------------------------------------------------------------------------
# mobility: positive by construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_mobility_stays_positive_for_extreme_raw_parameters(n_species):
    model = SquareGradient(GRID, n_species=n_species, nyquist_mask=True, **POLY)
    with torch.no_grad():
        model._m_raw.copy_(torch.full((n_species,), -100.0))
    rho = 0.5 * torch.ones(1, n_species, *GRID)
    M = model.mobility(rho, T=torch.tensor([1.0]))
    assert torch.all(M > 0.0)


def test_mobility_broadcasts_one_value_per_species_over_the_grid():
    model = SquareGradient(GRID, n_species=2, nyquist_mask=True, **POLY)
    with torch.no_grad():
        model._m_raw.copy_(torch.tensor([2.0, -2.0]))
    rho = torch.rand(1, 2, *GRID)
    M = model.mobility(rho, T=torch.tensor([1.0]))
    assert M.shape == rho.shape
    # Constant per species (independent of position), and the two species
    # differ (so a bug that broadcasts species-0's value to every channel
    # would be caught).
    assert torch.allclose(M[0, 0], M[0, 0, 0, 0, 0] * torch.ones(GRID))
    assert torch.allclose(M[0, 1], M[0, 1, 0, 0, 0] * torch.ones(GRID))
    assert not torch.allclose(M[0, 0, 0, 0, 0], M[0, 1, 0, 0, 0])


# ---------------------------------------------------------------------------
# forward: k-space in and out, mass conserved
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_forward_is_k_space_in_and_out(n_species):
    model = SquareGradient(GRID, n_species=n_species, nyquist_mask=True, **POLY)
    boxes = _boxes(3)
    T = torch.full((3,), 1.5)
    torch.manual_seed(0)
    N = GRID[0] * GRID[1] * GRID[2]
    rho = 0.3 + 0.1 * torch.rand(3, n_species, *GRID)
    rho_hat = model.ops.rfft(rho) / N

    drho_hat_dt = model.forward(rho_hat, boxes, T)

    assert drho_hat_dt.is_complex()
    assert drho_hat_dt.shape == rho_hat.shape
    assert torch.isfinite(drho_hat_dt.real).all()
    assert torch.isfinite(drho_hat_dt.imag).all()


@pytest.mark.parametrize("n_species", [1, 2])
def test_forward_is_built_from_this_rungs_own_chemical_potential_and_mobility(
        n_species):
    """`forward` must not bypass its own `chemical_potential`/`mobility`
    (e.g. quietly falling back to `mu = rho`, which would still pass the
    shape and mass-conservation checks above, since neither depends on what
    `mu` physically is). Recomputes the same div(M grad mu) skeleton
    independently, calling `model.chemical_potential`/`model.mobility`
    explicitly, and requires an exact match with `model.forward`'s output.
    """
    model = SquareGradient(GRID, n_species=n_species, nyquist_mask=True, **POLY)
    with torch.no_grad():
        model._kappa_chol_raw.copy_(
            torch.arange(model._kappa_chol_raw.numel(), dtype=torch.float32)
            * 0.1 + 0.5)
    boxes = _boxes(2)
    T = torch.full((2,), 1.7)
    torch.manual_seed(2)
    N = GRID[0] * GRID[1] * GRID[2]
    rho = 0.4 + 0.1 * torch.rand(2, n_species, *GRID)
    rho_hat = model.ops.rfft(rho) / N

    drho_hat_dt = model.forward(rho_hat, boxes, T)

    ops = model.ops
    mu = model.chemical_potential(rho, boxes, T)
    M = model.mobility(rho, T)
    kx, ky, kz = ops.k_axes(boxes)
    mu_hat = ops.rfft(mu) / N
    gx, gy, gz = ops.grad_hat(mu_hat * N, kx, ky, kz)
    grad_mu = torch.stack(
        [ops.irfft(gx), ops.irfft(gy), ops.irfft(gz)], dim=2)
    J = M.unsqueeze(2) * grad_mu   # +div(M grad mu); see test_model_protocol
    Jx_hat = ops.rfft(J[:, :, 0]) / N
    Jy_hat = ops.rfft(J[:, :, 1]) / N
    Jz_hat = ops.rfft(J[:, :, 2]) / N
    expected = ops.div_hat(Jx_hat, Jy_hat, Jz_hat, kx, ky, kz)

    torch.testing.assert_close(drho_hat_dt, expected)


@pytest.mark.parametrize("n_species", [1, 2])
def test_mass_is_conserved_by_forward(n_species):
    model = SquareGradient(GRID, n_species=n_species, nyquist_mask=True, **POLY)
    boxes = _boxes(4)
    T = torch.full((4,), 2.0)
    torch.manual_seed(0)
    N = GRID[0] * GRID[1] * GRID[2]
    rho = 0.3 + 0.1 * torch.rand(4, n_species, *GRID)
    rho_hat = model.ops.rfft(rho) / N

    drho_hat_dt = model.forward(rho_hat, boxes, T)

    k0 = drho_hat_dt[:, :, 0, 0, 0]
    assert torch.all(k0.real == 0.0)
    assert torch.all(k0.imag == 0.0)


# ---------------------------------------------------------------------------
# bulk_free_energy_density excludes the gradient term
# ---------------------------------------------------------------------------

def test_bulk_free_energy_density_is_independent_of_kappa():
    """The gradient term acts on the field as a whole; `bulk_free_energy_
    density` must not see it (the model protocol's contract).
    """
    model = SquareGradient(GRID, n_species=1, nyquist_mask=True, **POLY)
    torch.manual_seed(0)
    rho = 0.5 + 0.1 * torch.randn(1, 1, *GRID)
    T = torch.tensor([1.0])

    before = model.bulk_free_energy_density(rho, T).clone()
    with torch.no_grad():
        model._kappa_chol_raw.mul_(37.0)
    after = model.bulk_free_energy_density(rho, T)

    torch.testing.assert_close(before, after)


def test_bulk_free_energy_density_has_a_species_axis_of_size_one():
    model = SquareGradient(GRID, n_species=2, nyquist_mask=True, **POLY)
    rho = 0.5 * torch.ones(3, 2, *GRID)
    f = model.bulk_free_energy_density(rho, T=torch.full((3,), 1.0))
    assert f.shape == (3, 1, *GRID)


# ---------------------------------------------------------------------------
# gradient_energy_density: the (1/2) factor
# ---------------------------------------------------------------------------

def test_gradient_energy_density_mean_matches_the_one_half_factor():
    """For a single Fourier mode `rho = c + A sin(kx x)`, `mean(grad(rho)^2)
    == A^2 kx^2 / 2` (mean of cos^2 over a full period), so the mean of
    `(1/2) kappa |grad(rho)|^2` is `kappa A^2 kx^2 / 4` -- an exact, closed
    form independent of the pointwise field, which is what makes this test
    able to catch a dropped or doubled 1/2 factor rather than only a
    qualitatively-right shape.
    """
    model = SquareGradient(GRID, n_species=1, nyquist_mask=True, **POLY)
    with torch.no_grad():
        model._kappa_chol_raw.copy_(torch.tensor([0.6]))
    kappa = model.kappa.item()

    Gx = GRID[0]
    m = 2
    kx = 2.0 * math.pi * m / Gx
    A = 0.33
    x, _, _ = _coords(GRID)
    rho = (0.5 + A * torch.sin(kx * x)).unsqueeze(0).unsqueeze(0)
    boxes = _boxes(1)

    energy = model.gradient_energy_density(rho, boxes)
    mean_energy = energy.mean().item()
    expected = kappa * (A ** 2) * (kx ** 2) / 4.0
    assert mean_energy == pytest.approx(expected, rel=1e-3)


def test_gradient_energy_density_matches_the_pointwise_analytic_formula():
    """Pointwise, not only on average. `(1/2) kappa (d rho/dx)^2` for a
    single-axis sinusoid is `(1/2) kappa A^2 kx^2 cos^2(kx x)` pointwise --
    which, by a periodic-box integration-by-parts identity, shares its MEAN
    with `-(1/2) kappa rho laplacian(rho)`, a different pointwise quantity
    built from `laplacian` instead of an actual `grad(rho)` dot product.
    The mean-only test above cannot tell these apart; this one, checking
    every grid point, can -- it is the test that actually catches "use
    laplacian where grad.grad is meant" inside this method.
    """
    model = SquareGradient(GRID, n_species=1, nyquist_mask=True, **POLY)
    with torch.no_grad():
        model._kappa_chol_raw.copy_(torch.tensor([0.6]))
    kappa = model.kappa.item()

    Gx = GRID[0]
    m = 2
    kx = 2.0 * math.pi * m / Gx
    A = 0.33
    x, _, _ = _coords(GRID)
    rho = (0.5 + A * torch.sin(kx * x)).unsqueeze(0).unsqueeze(0)
    boxes = _boxes(1)

    energy = model.gradient_energy_density(rho, boxes)
    expected = 0.5 * kappa * (A ** 2) * (kx ** 2) * torch.cos(kx * x) ** 2
    torch.testing.assert_close(energy[0, 0], expected, atol=1e-4, rtol=1e-4)


def test_gradient_energy_density_is_symmetric_under_kappa_transpose_relabelling():
    """A sanity check that the cross terms are included at all (n_species =
    2): swapping which off-diagonal raw parameter carries the coupling
    should not change the total energy for a symmetric field pair, since
    kappa is symmetric by construction.
    """
    model = SquareGradient(GRID, n_species=2, nyquist_mask=True, **POLY)
    with torch.no_grad():
        model._kappa_chol_raw.copy_(torch.tensor([0.3, 0.2, 0.25]))
    torch.manual_seed(0)
    rho = 0.5 + 0.05 * torch.randn(1, 2, *GRID)
    boxes = _boxes(1)

    energy = model.gradient_energy_density(rho, boxes)
    kappa = model.kappa
    assert torch.allclose(kappa, kappa.T)
    assert torch.isfinite(energy).all()


# -- the declared local form -----------------------------------------------------------------

_CLOSED = dict(nyquist_mask=True, kappa_init=0.5, rho_eps=1e-4, kB=1.0,
               mobility_prefactor="mole_fraction", mobility_shape="lattice_scalar",
               mobility_t_form="arrhenius", mobility_shape_init=-2.3,
               mobility_t_ref=1.65, mobility_activation_energy_init=-0.3)


def test_the_local_form_is_declared_and_named_when_wrong():
    with pytest.raises(TypeError, match="local"):
        SquareGradient(GRID, n_species=1, nyquist_mask=True, kappa_init=1.0)
    with pytest.raises(ValueError, match="quadratic_quartic"):
        SquareGradient(GRID, n_species=1, nyquist_mask=True, kappa_init=1.0, local="mlp")


@pytest.mark.parametrize("local,constants", [("flory_huggins", {"w": 3.0}),
                                             ("landau", {"a0": 1.0, "b": 1.0, "T_c": 1.5})])
def test_a_closed_form_asks_for_its_constants_and_refuses_the_others(local, constants):
    SquareGradient(GRID, 1, local=local, **constants, **_CLOSED)
    for name in constants:
        with pytest.raises(ValueError, match=name):
            SquareGradient(GRID, 1, local=local,
                           **{k: v for k, v in constants.items() if k != name}, **_CLOSED)
    other = {"w": 3.0} if local == "landau" else {"T_c": 1.5}
    with pytest.raises(ValueError, match="reads no"):
        SquareGradient(GRID, 1, local=local, **constants, **other, **_CLOSED)
    with pytest.raises(ValueError, match="one-field"):
        SquareGradient(GRID, 2, local=local, **constants, **_CLOSED)
    with pytest.raises(ValueError, match="rho_eps"):
        SquareGradient(GRID, 1, local=local, **constants, **{**_CLOSED, "rho_eps": None})


def test_the_polynomial_reads_no_mobility_argument():
    with pytest.raises(ValueError, match="mobility_prefactor"):
        SquareGradient(GRID, 1, nyquist_mask=True, mobility_prefactor="mole_fraction", **POLY)


def test_a_closed_forms_curvature_at_a_wavevector_is_kappa_k_squared():
    model = SquareGradient(GRID, 1, local="flory_huggins", w=3.0, **_CLOSED).double()
    k = torch.tensor([[0.0, 0.5, 2.0]], dtype=torch.float64)
    out = model.curvature_at_wavevector(k)
    assert out.shape == (1, 3, 1, 1)
    assert torch.allclose(out[..., 0, 0], 0.5 * k * k)


def test_the_closed_forms_read_offs_are_their_formulas():
    fh = SquareGradient(GRID, 1, local="flory_huggins", w=3.0, **_CLOSED).double()
    la = SquareGradient(GRID, 1, local="landau", a0=2.0, b=1.0, T_c=1.4, **_CLOSED).double()
    T = torch.tensor([1.2], dtype=torch.float64)
    assert fh.Tc_curvature() == 1.5 and la.Tc_curvature() == pytest.approx(1.4)
    assert float(fh.bulk_curvature(T)) == pytest.approx(4 * 1.2 - 6.0)
    assert float(la.bulk_curvature(T)) == pytest.approx(2.0 * (1.2 - 1.4))
    phi = np.linspace(0.3, 0.7, 5)
    f = fh.bulk_free_energy_curve(phi, 1.2).detach().numpy()
    want = 3.0 * phi * (1 - phi) + 1.2 * (phi * np.log(phi) + (1 - phi) * np.log1p(-phi))
    assert np.allclose(f, want, rtol=0, atol=1e-12)
    with pytest.raises(NotImplementedError, match="quadratic_quartic"):
        SquareGradient(GRID, 1, nyquist_mask=True, **POLY).Tc_curvature()
