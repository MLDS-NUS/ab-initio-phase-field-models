"""The two-dimensional operator set: ``SpectralOps2D``, the cache and ``Field`` on a declared two-axis grid.

The number of axes is declared by the grid's length, never read off a tensor; the 2D set is held to the
same properties as the 3D one (round trip, multiplicity, Nyquist), and to the 3D one itself: on a field
that does not vary along z, the k_z = 0 plane of every 3D operator is the 2D operator's result. The 3D
class is pinned bit for bit by ``tests/golden``; what is checked here is that its ``ndim`` is a class
attribute and not state."""
from __future__ import annotations

import math

import pytest
import torch

from aipf.fields import Field
from aipf.spectral import (OPS_CLASSES, OpsCache, SpectralOps, SpectralOps2D, k_squared,
                           make_ops)

EVEN = (6, 8)
ODD = (6, 7)
BOXES = torch.tensor([[7.0, 9.5], [5.5, 6.0]])


def _field(grid, batch=2, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n, *grid, generator=g)


# ---------------------------------------------------------------------------
# the operator set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid", [EVEN, ODD])
def test_the_2d_half_spectrum_is_halved_along_y_and_inverts_to_the_field(grid):
    ops = SpectralOps2D(grid, 2, nyquist_mask=True)
    f = _field(grid)
    f_hat = ops.rfft(f)
    assert tuple(f_hat.shape) == (2, 2, grid[0], grid[1] // 2 + 1)
    assert torch.allclose(ops.irfft(f_hat), f, atol=1e-5)


@pytest.mark.parametrize("grid", [EVEN, ODD])
def test_the_2d_multiplicity_reproduces_the_full_spectrum_sum(grid):
    """Parseval through the half spectrum, both parities of the half axis."""
    ops = SpectralOps2D(grid, 2, nyquist_mask=True)
    f = _field(grid, seed=1).double()
    full = (torch.fft.fftn(f, dim=(-2, -1)).abs() ** 2).sum(dim=(-2, -1))
    half = (ops.MULT.double() * ops.rfft(f).abs() ** 2).sum(dim=(-2, -1))
    assert torch.allclose(half, full, rtol=1e-10)
    assert torch.allclose(full, math.prod(grid) * (f ** 2).sum(dim=(-2, -1)), rtol=1e-10)


def test_the_2d_buffers_mean_what_the_3d_ones_mean():
    ops = SpectralOps2D(EVEN, 1, nyquist_mask=True)
    assert ops.ndim == 2 and ops.grid == EVEN and ops.Gyr == EVEN[1] // 2 + 1
    state = ops.state_dict()
    assert list(state) == ["NX", "NY", "MULT"]
    assert tuple(state["NX"].shape) == (EVEN[0], 1)
    assert tuple(state["NY"].shape) == (1, ops.Gyr)
    assert tuple(state["MULT"].shape) == (1, 1, 1, ops.Gyr)
    assert {name for name, _ in ops.named_buffers()} == {"NX", "NY", "MULT", "MX_ODD", "MY_ODD"}
    # the Nyquist wavenumber of each even axis is zeroed in the odd-order operators only
    assert float(ops.MX_ODD[EVEN[0] // 2, 0]) == 0.0 and float(ops.MY_ODD[0, -1]) == 0.0
    plain = SpectralOps2D(EVEN, 1, nyquist_mask=False)
    assert bool((plain.MX_ODD == 1).all()) and bool((plain.MY_ODD == 1).all())
    assert float(SpectralOps2D(ODD, 1, nyquist_mask=True).MY_ODD[0, -1]) == 1.0


def test_the_2d_set_reads_two_box_lengths_and_refuses_three():
    ops = SpectralOps2D(EVEN, 1, nyquist_mask=True)
    kx, ky = ops.k_axes(BOXES)
    assert tuple(kx.shape) == tuple(ky.shape) == (2, 1, EVEN[0], ops.Gyr)
    assert torch.allclose(ky[1, 0, 0, 1], torch.tensor(2 * math.pi / 6.0))
    with pytest.raises(ValueError, match=r"\(B, 2\)"):
        ops.k_axes(torch.ones(2, 3))


@pytest.mark.parametrize("nyquist_mask", [True, False])
@pytest.mark.parametrize("grid", [EVEN, ODD])
def test_every_3d_operator_on_a_z_invariant_field_is_the_2d_operator_on_its_kz0_plane(grid, nyquist_mask):
    Gz = 4
    ops2 = SpectralOps2D(grid, 2, nyquist_mask=nyquist_mask)
    ops3 = SpectralOps((*grid, Gz), 2, nyquist_mask=nyquist_mask)
    boxes3 = torch.cat([BOXES, torch.tensor([[3.0], [4.5]])], dim=1)
    f = _field(grid, seed=2)
    f2 = ops2.rfft(f) / math.prod(grid)
    f3 = ops3.rfft(f.unsqueeze(-1).expand(*f.shape, Gz).contiguous()) / (math.prod(grid) * Gz)
    Gyr = grid[1] // 2 + 1

    def plane(x3):
        return x3[..., :Gyr, 0]

    assert torch.allclose(plane(f3), f2, atol=1e-6)
    assert torch.allclose(f3[..., 1:], torch.zeros_like(f3[..., 1:]), atol=1e-6)
    ks2, ks3 = ops2.k_axes(BOXES), ops3.k_axes(boxes3)
    # fftfreq(G) * G is an integer only to the last bit, so the wavevectors agree to float precision;
    # and the 3D y axis is a full one, whose Nyquist wavenumber is -G/2 where the 2D half axis has +G/2
    assert torch.allclose(plane(ks3[0]), ks2[0], rtol=1e-6, atol=0.0)
    assert torch.allclose(plane(ks3[1]).abs(), ks2[1], rtol=1e-6, atol=0.0)
    assert torch.allclose(plane(ops3.k2(boxes3)), ops2.k2(BOXES), rtol=1e-6, atol=0.0)
    assert torch.allclose(plane(k_squared(ks3)), k_squared(ks2), rtol=1e-6, atol=0.0)
    assert torch.allclose(plane(ops3.sigma_filter(boxes3, 1.3)), ops2.sigma_filter(BOXES, 1.3),
                          rtol=1e-5, atol=0.0)
    assert torch.equal(plane(ops3.band_mask(boxes3, 2.0)), ops2.band_mask(BOXES, 2.0))
    assert torch.allclose(plane(ops3.laplacian(f3, ops3.k2(boxes3))),
                          ops2.laplacian(f2, ops2.k2(BOXES)), atol=1e-5)

    # the odd-order operators: equal on the plane but for that sign, which the y component carries
    # on the y Nyquist column (zeroed there anyway under nyquist_mask=True)
    sign = torch.ones(Gyr)
    if grid[1] % 2 == 0:
        sign[-1] = -1.0
    g2 = ops2.grad_hat(f2, *ks2)
    g3 = ops3.grad_hat(f3, *ks3)
    assert torch.allclose(plane(g3[0]), g2[0], atol=1e-6)
    assert torch.allclose(plane(g3[1]) * sign, g2[1], atol=1e-6)
    assert torch.allclose(g3[2], torch.zeros_like(g3[2]), atol=1e-6)
    assert torch.allclose(plane(ops3.div_hat(*g3, *ks3)), ops2.div_hat(*g2, *ks2), atol=1e-5)


# ---------------------------------------------------------------------------
# picking and caching by the declared grid
# ---------------------------------------------------------------------------

def test_make_ops_picks_the_class_by_the_grids_length():
    assert type(make_ops((4, 4, 4), 1, nyquist_mask=True)) is SpectralOps
    assert type(make_ops((4, 4), 1, nyquist_mask=True)) is SpectralOps2D
    assert set(OPS_CLASSES) == {SpectralOps, SpectralOps2D}
    with pytest.raises(ValueError, match="axes"):
        make_ops((4,), 1, nyquist_mask=True)


def test_a_2d_cache_reads_two_trailing_axes_never_the_species_axis():
    """A (B, n, Gx, Gyr) shape read as three axes would take n for Gx: the cache reads the declared two."""
    cache = OpsCache((8, 6), 2, nyquist_mask=True)
    assert cache.ndim == 2 and cache.grid == (8, 6)
    assert cache.ops_for((3, 2, 8, 4)) is cache.ops
    other = cache.ops_for((3, 2, 6, 5))
    assert isinstance(other, SpectralOps2D) and other.grid == (6, 8)
    assert cache.ops_for_grid((6, 8)) is other
    with pytest.raises(ValueError, match="two positive integers"):
        cache.ops_for_grid((6, 8, 4))


def test_a_2d_cache_keeps_its_declared_odd_grid():
    cache = OpsCache((8, 7), 1, nyquist_mask=True)
    assert cache.ops_for((1, 1, 8, 4)) is cache.ops
    assert cache.ops_for((1, 1, 8, 4)).grid == (8, 7)


def test_the_3d_operator_set_keeps_its_state_and_declares_ndim_as_a_class_attribute():
    assert SpectralOps.__dict__["ndim"] == 3 and SpectralOps2D.__dict__["ndim"] == 2
    ops = SpectralOps((6, 10, 7), 2, nyquist_mask=True)
    assert "ndim" not in ops.__dict__
    state = ops.state_dict()
    assert {k: tuple(v.shape) for k, v in state.items()} == {
        "NX": (6, 1, 1), "NY": (1, 10, 1), "NZ": (1, 1, 4), "MULT": (1, 1, 1, 1, 4)}
    assert OpsCache((4, 4, 4), 1, nyquist_mask=True).ndim == 3


# ---------------------------------------------------------------------------
# the field container
# ---------------------------------------------------------------------------

def test_a_field_on_a_two_axis_grid_is_two_dimensional():
    ops = SpectralOps2D(EVEN, 2, nyquist_mask=True)
    rho_hat = ops.rfft(_field(EVEN)) / math.prod(EVEN)
    field = Field(rho_hat, BOXES, EVEN)
    assert field.ndim == 2 and field.n_species == 2
    with pytest.raises(ValueError, match=r"\(B, 2\)"):
        Field(rho_hat, torch.ones(2, 3), EVEN)
    with pytest.raises(ValueError, match="trailing shape"):
        Field(rho_hat, BOXES, (6, 10))
    with pytest.raises(ValueError, match="Gx, Gyr"):
        Field(rho_hat.unsqueeze(-1), BOXES, EVEN)
