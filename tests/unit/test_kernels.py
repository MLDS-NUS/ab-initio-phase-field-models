"""Tests for aipf.functional.kernels: rung 3's pair-kernel form registry.

Properties, not implementation, following the pattern of
``tests/unit/test_spectral.py``: every test that can be phrased as "the
result computed one way equals the result computed an independent way" is,
rather than pinning literal buffer contents.

``n_species`` is exercised at 1 (the third system's one-scalar case) and 2
(the first two systems' 00/01/11 case) throughout, per the plan's global
constraint that nothing here is a degenerate afterthought at either count.
"""
from __future__ import annotations

import math

import pytest
import torch

from aipf.functional.kernels import (
    AnalyticRadialTransform,
    PairKernel,
    QuinticEnvelope,
    RadialKernelSet,
    radial_fourier_transform,
)
from aipf.spectral import SpectralOps

torch.manual_seed(0)


def _build_kernel(n_species, R_cut=3.0, n_quad=64, n_k_table=129, k_table_max=8.0,
                   tail_sigma=None, hidden=8, activation="gelu"):
    envelope = QuinticEnvelope(R_cut, tail_sigma)
    radial_set = RadialKernelSet(n_species, envelope, hidden=hidden, activation=activation)
    evaluator = AnalyticRadialTransform(R_cut=R_cut, n_quad=n_quad,
                                         n_k_table=n_k_table, k_table_max=k_table_max)
    return PairKernel(radial_set, evaluator, n_quad=n_quad)


# ---------------------------------------------------------------------------
# radial_fourier_transform: the analytic radial quadrature, on its own
# ---------------------------------------------------------------------------

def test_gaussian_transform_matches_its_analytic_transform():
    """Ŵ(k) = 4*pi*int r^2 W(r) sinc(kr) dr for W(r) = A*exp(-r^2/(2*sigma^2))
    has a closed form: the 3D Fourier transform of an isotropic Gaussian is
    A*(2*pi*sigma^2)^1.5 * exp(-k^2*sigma^2/2). No envelope, no MLP, no
    interpolation table -- this pins the quadrature primitive alone.
    """
    A, sigma = 2.0, 0.5
    r = torch.linspace(0.0, 8.0 * sigma, 4000, dtype=torch.float64)
    w_vals = A * torch.exp(-r * r / (2.0 * sigma * sigma))
    k = torch.tensor([0.0, 0.3, 1.0, 2.5, 4.0], dtype=torch.float64)

    got = radial_fourier_transform(w_vals, r, k)
    want = A * (2.0 * math.pi * sigma * sigma) ** 1.5 * torch.exp(-k * k * sigma * sigma / 2.0)

    assert torch.allclose(got, want, rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------------------
# symmetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_w_hat_is_symmetric_for_arbitrary_parameters(n_species):
    kernel = _build_kernel(n_species)
    # perturb every parameter away from its (already random) init, so this
    # is not merely re-checking the init distribution's own symmetry.
    with torch.no_grad():
        for p in kernel.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    k = torch.rand(5, 3, 7) * 6.0
    w_hat = kernel.w_hat(k)
    assert w_hat.shape == (5, 3, 7, n_species, n_species)
    assert torch.allclose(w_hat, w_hat.transpose(-1, -2), atol=1e-6)


@pytest.mark.parametrize("n_species", [1, 2])
def test_matrix_of_r_is_symmetric(n_species):
    envelope = QuinticEnvelope(R_cut=3.0)
    radial_set = RadialKernelSet(n_species, envelope, hidden=8)
    r = torch.rand(4, 5) * 3.0
    m = radial_set.matrix_of_r(r)
    assert m.shape == (4, 5, n_species, n_species)
    assert torch.allclose(m, m.transpose(-1, -2))


# ---------------------------------------------------------------------------
# grid independence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_w_hat_is_grid_independent(n_species):
    """Ŵ(k) depends only on the physical |k|, never on the grid it came from.

    Two different grid shapes, with box lengths chosen so that a specific
    integer mode of each lands on the exact same physical |k| (to float
    precision), must agree exactly -- this is the property that makes a
    kernel trained on one grid shape valid on another.
    """
    kernel = _build_kernel(n_species)
    with torch.no_grad():
        for p in kernel.parameters():
            p.add_(torch.randn_like(p) * 0.3)

    grid_a = (8, 6, 10)
    grid_b = (5, 5, 12)
    ops_a = SpectralOps(grid_a, n_species, nyquist_mask=True)
    ops_b = SpectralOps(grid_b, n_species, nyquist_mask=True)

    # box_a chosen arbitrarily; box_b's z-length chosen so mode (1,0,0) of
    # grid_a and mode (0,1,0) of grid_b land on the same |k|.
    box_a = torch.tensor([[9.0, 11.0, 7.0]])
    k_a = ops_a.k_axes(box_a)
    kmag_a = torch.sqrt(k_a[0] ** 2 + k_a[1] ** 2 + k_a[2] ** 2)[0, 0, 1, 0, 0]

    L_b_y = 2.0 * math.pi / kmag_a.item()
    box_b = torch.tensor([[4.0, L_b_y, 13.0]])
    k_b = ops_b.k_axes(box_b)
    kmag_b = torch.sqrt(k_b[0] ** 2 + k_b[1] ** 2 + k_b[2] ** 2)[0, 0, 0, 1, 0]

    assert torch.allclose(kmag_a, kmag_b, atol=1e-5)

    w_a = kernel.w_hat(kmag_a)
    w_b = kernel.w_hat(kmag_b)
    assert torch.allclose(w_a, w_b, atol=1e-6)


# ---------------------------------------------------------------------------
# kappa_eff vs finite difference of w_hat at k = 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_kappa_eff_matches_finite_difference_of_w_hat_at_zero(n_species):
    """kappa_eff = -(2*pi/3) * int r^4 W(r) dr.

    Taylor-expanding sinc(kr) = 1 - (kr)^2/6 + O(k^4) in the radial
    transform gives W_hat(k) = W_hat(0) + kappa_eff*k^2 + O(k^4) near k=0,
    an EVEN function of k, so its exact second derivative at k=0 is
    2*kappa_eff. The central finite-difference estimate of that second
    derivative, 2*(W_hat(h) - W_hat(0))/h**2 (using W_hat(-h) = W_hat(h)),
    must therefore match 2*kappa_eff, not kappa_eff itself. Loosely pinned
    (order of magnitude of h and rtol), so the test states the
    relationship, not a fitted constant.
    """
    kernel = _build_kernel(n_species, n_k_table=2049, k_table_max=8.0, n_quad=256)
    with torch.no_grad():
        for p in kernel.parameters():
            p.add_(torch.randn_like(p) * 0.3)

    kappa = kernel.kappa_eff()
    assert kappa.shape == (n_species, n_species)

    h = 0.05
    w0 = kernel.w_hat(torch.tensor(0.0))
    wh = kernel.w_hat(torch.tensor(h))
    fd_second_deriv = 2.0 * (wh - w0) / (h * h)

    assert torch.allclose(fd_second_deriv, 2.0 * kappa, rtol=5e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# envelope
# ---------------------------------------------------------------------------

def test_envelope_is_exactly_zero_beyond_r_cut():
    env = QuinticEnvelope(R_cut=2.0)
    r = torch.tensor([2.0001, 3.0, 100.0])
    assert torch.all(env(r) == 0.0)


def test_envelope_is_one_at_the_origin():
    env = QuinticEnvelope(R_cut=2.0)
    assert env(torch.tensor(0.0)).item() == pytest.approx(1.0)


def test_envelope_gaussian_tail_decays_beyond_the_untailed_case():
    r = torch.tensor([1.0])
    plain = QuinticEnvelope(R_cut=5.0)
    tailed = QuinticEnvelope(R_cut=5.0, tail_sigma=0.5)
    assert tailed(r).item() < plain(r).item()


@pytest.mark.parametrize("n_species", [1, 2])
def test_w_of_r_is_exactly_zero_beyond_r_cut(n_species):
    envelope = QuinticEnvelope(R_cut=2.0)
    radial_set = RadialKernelSet(n_species, envelope, hidden=8)
    r = torch.tensor([2.0001, 5.0, 20.0])
    w = radial_set.w_of_r(r)
    assert torch.all(w == 0.0)


# ---------------------------------------------------------------------------
# System fields, not module defaults
# ---------------------------------------------------------------------------

def test_no_constructor_defaults_a_system_field():
    """R_cut, the quadrature sizes and the k-table maximum are System
    fields; this module must ship no default for any of them.

    Introspects the actual signatures rather than trying every constructor
    with a missing argument, so a default added later to any of these
    parameters -- not just the ones exercised elsewhere in this file, which
    always pass every argument explicitly -- is caught here.
    """
    import inspect

    no_default_params = {
        QuinticEnvelope: {"R_cut"},
        AnalyticRadialTransform: {"R_cut", "n_quad", "n_k_table", "k_table_max"},
        PairKernel: {"n_quad"},
    }
    for cls, names in no_default_params.items():
        sig = inspect.signature(cls.__init__)
        for name in names:
            param = sig.parameters[name]
            assert param.default is inspect.Parameter.empty, (
                f"{cls.__name__}.__init__'s {name!r} has a default "
                f"({param.default!r}); it is a System field and must be "
                f"required")


def test_n_species_below_one_raises():
    envelope = QuinticEnvelope(R_cut=2.0)
    with pytest.raises(ValueError, match="n_species"):
        RadialKernelSet(0, envelope)


def test_unknown_activation_raises():
    envelope = QuinticEnvelope(R_cut=2.0)
    with pytest.raises(ValueError, match="activation"):
        RadialKernelSet(2, envelope, activation="relu")


@pytest.mark.parametrize("n_species,n_pairs", [(1, 1), (2, 3), (3, 6)])
def test_pair_count_is_n_times_n_plus_one_over_two(n_species, n_pairs):
    envelope = QuinticEnvelope(R_cut=2.0)
    radial_set = RadialKernelSet(n_species, envelope, hidden=4)
    assert len(radial_set.pairs) == n_pairs
    assert len(radial_set.nets) == n_pairs


# ---------------------------------------------------------------------------
# the pluggable w_hat evaluator seam
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_pair_kernel_delegates_w_hat_to_its_evaluator(n_species):
    """A drop-in evaluator satisfying the one-method interface is used
    as-is, proving the seam: a route living entirely outside this module
    (a rasterise-and-FFT route) can be swapped in without
    PairKernel or RadialKernelSet changing at all.
    """
    class _ConstantEvaluator:
        def __init__(self, value):
            self.value = value

        def w_hat(self, radial_set, k):
            shape = tuple(k.shape) + (radial_set.n_species, radial_set.n_species)
            return self.value.expand(shape)

    envelope = QuinticEnvelope(R_cut=2.0)
    radial_set = RadialKernelSet(n_species, envelope, hidden=4)
    sentinel = torch.eye(n_species) * 7.0
    kernel = PairKernel(radial_set, _ConstantEvaluator(sentinel), n_quad=32)

    k = torch.rand(3, 4)
    got = kernel.w_hat(k)
    assert torch.allclose(got, sentinel.expand(3, 4, n_species, n_species))


# ---------------------------------------------------------------------------
# mutation-list coverage: assembly symmetry from an asymmetric pair matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [2, 3])
def test_assembled_matrix_is_symmetric_even_when_pair_values_are_distinct(n_species):
    """Every off-diagonal pair value is distinct (no accidental symmetry from
    equal radial functions), so a broken assembly that filled (i, j) but not
    (j, i) -- or vice-versa -- is caught even by chance.
    """
    kernel = _build_kernel(n_species, hidden=4)
    with torch.no_grad():
        for i, net in enumerate(kernel.radial_set.nets):
            for p in net.parameters():
                p.fill_(0.0)
            net[-1].bias.fill_(float(i + 1))  # distinct nonzero constant per pair
    k = torch.tensor([0.5, 1.0, 2.0])
    w_hat = kernel.w_hat(k)
    assert torch.allclose(w_hat, w_hat.transpose(-1, -2))
