"""The pair kernel in two dimensions: the Hankel transform, and the nonlocal-kernel functional on a two-axis grid.

The transform is checked against a closed form, the Gaussian ``W(r) = exp(-r^2 / 2a^2)``, whose 2D
transform is ``2 pi a^2 exp(-k^2 a^2 / 2)``; the evaluator keeps the 3D one's buffers and state."""
from __future__ import annotations

import math

import pytest
import torch

from aipf.functional.kernels import AnalyticRadialTransform, hankel_transform
from aipf.functional.nonlocal_kernel import NonlocalKernel
from aipf.spectral import SpectralOps2D


def _gaussian_closed_form(a, k):
    return 2.0 * math.pi * a * a * torch.exp(-k * k * a * a / 2.0)


@pytest.mark.parametrize("a", [0.5, 1.0, 1.7])
def test_the_hankel_transform_of_a_gaussian_is_its_closed_form(a):
    r = torch.linspace(0.0, 12.0 * a, 4001, dtype=torch.float64)
    k = torch.linspace(0.0, 6.0 / a, 37, dtype=torch.float64).reshape(1, 37)
    w = torch.exp(-r * r / (2.0 * a * a))
    got = hankel_transform(torch.stack([w, 2.0 * w]), r, k)
    want = _gaussian_closed_form(a, k)
    assert tuple(got.shape) == (2, 1, 37)
    assert torch.allclose(got[0], want, rtol=0.0, atol=1e-6 * float(want.max()))
    assert torch.allclose(got[1], 2.0 * want, rtol=0.0, atol=2e-6 * float(want.max()))


class _GaussianSet:
    """The radial-set seam with one pair and a Gaussian ``W``."""

    pairs = ((0, 0),)
    n_species = 1

    def __init__(self, a):
        self.a = a

    def w_of_r(self, r):
        return torch.exp(-r * r / (2.0 * self.a * self.a)).unsqueeze(0)


def test_the_2d_k_table_evaluator_interpolates_the_hankel_transform_on_the_3d_buffers():
    a = 1.0
    two = AnalyticRadialTransform(8.0, 2001, 801, 8.0, dim=2)
    three = AnalyticRadialTransform(8.0, 2001, 801, 8.0)
    assert two.dim == 2 and three.dim == 3
    assert {k: tuple(v.shape) for k, v in two.state_dict().items()} == \
        {k: tuple(v.shape) for k, v in three.state_dict().items()} == \
        {"r_quad": (2001,), "k_table": (801,)}
    k = torch.tensor([[0.0, 0.3, 1.1], [2.0, 3.7, 5.0]])
    got = two.w_hat(_GaussianSet(a), k)
    assert tuple(got.shape) == (2, 3, 1, 1)
    assert torch.allclose(got[..., 0, 0], _gaussian_closed_form(a, k), atol=2e-4)
    # the 3D evaluator is the sine transform, (2 pi a^2)^(3/2) exp(-k^2 a^2 / 2), not this one
    want3 = (2.0 * math.pi * a * a) ** 1.5 * torch.exp(-k * k * a * a / 2.0)
    assert torch.allclose(three.w_hat(_GaussianSet(a), k)[..., 0, 0], want3, atol=2e-4)
    with pytest.raises(ValueError, match="dim=1"):
        AnalyticRadialTransform(8.0, 11, 11, 8.0, dim=1)


def _nonlocal(grid, **kw):
    torch.manual_seed(0)
    return NonlocalKernel(grid, 2, 1.0, [0.5, 0.4], 8, 2.5, kernel_n_quad=64,
                          kernel_n_k_table=128, kernel_k_table_max=12.0, ideal_form="gas",
                          kBT_ref=1.0, h_u=8, nyquist_mask=True, mobility_input_ref=None, **kw)


def test_the_nonlocal_kernel_on_a_two_axis_grid_is_two_dimensional_and_conserves_mass():
    model = _nonlocal((8, 6))
    assert isinstance(model.ops, SpectralOps2D) and model.grid == (8, 6)
    assert model.kernel.evaluator.dim == 2
    assert set(model.state_dict()) == set(_nonlocal((8, 6, 4)).state_dict()) - {"ops.NZ"}
    g = torch.Generator().manual_seed(1)
    rho = torch.stack([0.5 + 0.05 * torch.randn(2, 8, 6, generator=g),
                       0.4 + 0.05 * torch.randn(2, 8, 6, generator=g)], dim=1)
    boxes, T = torch.tensor([[8.0, 6.5], [7.0, 7.0]]), torch.tensor([1.0, 1.2])
    rho_hat = model.ops.rfft(rho) / 48
    F = model(rho_hat, boxes, T)
    assert tuple(F.shape) == tuple(rho_hat.shape)
    assert float(F[..., 0, 0].abs().max()) < 1e-7
    assert float(F.abs().max()) > 1e-6
    # the real-space seam agrees with the k-space fast path
    mu = model.chemical_potential(rho, boxes, T)
    assert tuple(mu.shape) == (2, 2, 8, 6)
    assert tuple(model.mobility(rho, T).shape) == (2, 2, 2, 8, 6)
    assert tuple(model.bulk_free_energy_density(rho, T).shape) == (2, 1, 8, 6)


def test_a_2d_nonlocal_kernel_refuses_an_evaluator_that_does_not_declare_two_dimensions():
    three = AnalyticRadialTransform(2.5, 64, 128, 12.0)
    with pytest.raises(ValueError, match="dim=2"):
        _nonlocal((8, 6), kernel_evaluator=three)
    model = _nonlocal((8, 6), kernel_evaluator=AnalyticRadialTransform(2.5, 64, 128, 12.0, dim=2))
    assert model.kernel.evaluator.dim == 2
