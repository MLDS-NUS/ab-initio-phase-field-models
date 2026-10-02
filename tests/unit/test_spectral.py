"""Tests for aipf.spectral: the k-space operator set.

Properties, not implementation. Every test that can be phrased as "the
result computed one way equals the result computed an independent way" is,
rather than pinning a specific buffer's literal contents -- see the
docstring of test_multiplicity_reproduces_the_full_spectrum_sum for the
concrete failure that shape of test avoids.
"""
from __future__ import annotations

import math

import pytest
import torch

from aipf.spectral import OpsCache, SpectralOps

# Two grid shapes, chosen to force both parities of Gz through the same
# tests: (6, 5, 8) has an even last axis (an explicit Nyquist plane exists),
# (6, 5, 7) has an odd one (it does not). The first system reaches spectral
# with an even Gz for both of its own grid shapes, but the multiplicity rule
# is parity-dependent, so a test suite that never exercises the odd case
# cannot catch a Nyquist-branch bug that only shows up there.
EVEN_GRID = (6, 5, 8)
ODD_GRID = (6, 5, 7)


def _boxes(batch: int, grid, device="cpu", low=8.0, high=12.0):
    """Random per-sample boxes, one full period per grid axis."""
    g = torch.Generator().manual_seed(0)
    return (low + (high - low) * torch.rand(batch, 3, generator=g)).to(device)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_construction_records_grid_and_n_species(n_species):
    ops = SpectralOps(EVEN_GRID, n_species, nyquist_mask=True)
    assert ops.grid == EVEN_GRID
    assert ops.n_species == n_species
    assert ops.Gzr == EVEN_GRID[2] // 2 + 1


def test_n_species_below_one_raises():
    with pytest.raises(ValueError, match="n_species"):
        SpectralOps(EVEN_GRID, 0, nyquist_mask=True)


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("grid", [EVEN_GRID, ODD_GRID])
def test_irfft_of_rfft_is_identity(n_species, grid):
    """Round trip, because every model does irfft -> pointwise -> rfft."""
    ops = SpectralOps(grid, n_species, nyquist_mask=True)
    torch.manual_seed(0)
    f = torch.randn(3, n_species, *grid)
    f_hat = ops.rfft(f)
    back = ops.irfft(f_hat)
    assert torch.allclose(back, f, atol=1e-5)


# ---------------------------------------------------------------------------
# multiplicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid", [EVEN_GRID, ODD_GRID])
def test_multiplicity_reproduces_the_full_spectrum_sum(grid):
    """sum(|full fft|^2) == sum(mult * |rfft|^2).

    This is the property the multiplicity weight exists for. Asserting the
    buffer's literal values instead would pass with the Nyquist plane wrong
    on an odd grid, which is exactly the case that differs between the
    conventions a published model may have been trained under.
    """
    ops = SpectralOps(grid, n_species=1, nyquist_mask=True)
    torch.manual_seed(1)
    f = torch.randn(2, 1, *grid, dtype=torch.float64)

    full = torch.fft.fftn(f, dim=(-3, -2, -1))
    full_sum = (full.abs() ** 2).sum()

    half = ops.rfft(f)
    half_sum = (ops.MULT.to(torch.float64) * (half.abs() ** 2)).sum()

    assert torch.allclose(full_sum, half_sum, rtol=1e-10)


def test_multiplicity_is_one_on_kz_zero_and_two_elsewhere_on_an_odd_grid():
    ops = SpectralOps(ODD_GRID, n_species=1, nyquist_mask=True)
    mult = ops.MULT.squeeze()
    assert mult[0].item() == pytest.approx(1.0)
    assert torch.all(mult[1:] == 2.0)


def test_multiplicity_is_one_on_kz_zero_and_nyquist_on_an_even_grid():
    ops = SpectralOps(EVEN_GRID, n_species=1, nyquist_mask=True)
    mult = ops.MULT.squeeze()
    assert mult[0].item() == pytest.approx(1.0)
    assert mult[-1].item() == pytest.approx(1.0)
    assert torch.all(mult[1:-1] == 2.0)


# ---------------------------------------------------------------------------
# band mask vs. a brute-force k-shell
# ---------------------------------------------------------------------------

def _brute_force_band_count(grid, box, k_max) -> int:
    """Independent recomputation of |k| via the textbook integer-mode rule.

    Loops over every rfft-grid index and converts it to a signed mode number
    the way `fftfreq` does (0..N/2-1, -N/2..-1 on the two full axes; 0..Gz/2
    on the half axis), then forms k = 2*pi*n/L directly -- a second code
    path from SpectralOps.k_axes, so this test cannot pass merely because
    both paths share the same bug.
    """
    Gx, Gy, Gz = grid
    Gzr = Gz // 2 + 1
    Lx, Ly, Lz = box
    count = 0
    for ix in range(Gx):
        nx = ix if ix <= (Gx - 1) // 2 else ix - Gx
        kx = TWO_PI * nx / Lx
        for iy in range(Gy):
            ny = iy if iy <= (Gy - 1) // 2 else iy - Gy
            ky = TWO_PI * ny / Ly
            for iz in range(Gzr):
                kz = TWO_PI * iz / Lz
                k2 = kx * kx + ky * ky + kz * kz
                if 0 < k2 <= k_max * k_max:
                    count += 1
    return count


TWO_PI = 2.0 * math.pi


@pytest.mark.parametrize("n_species", [1, 2])
def test_band_mask_counts_match_a_brute_force_k_shell(n_species):
    """Count modes with |k| <= k_max directly and compare."""
    ops = SpectralOps(ODD_GRID, n_species, nyquist_mask=True)
    box = (9.3, 10.7, 8.1)
    boxes = torch.tensor([box], dtype=torch.float64)
    k_max = 1.2

    mask = ops.band_mask(boxes, k_max)
    # band_mask broadcasts over the channel axis; count on one channel slice.
    got = int(mask[0, 0].sum().item())
    want = _brute_force_band_count(ODD_GRID, box, k_max)
    assert got == want
    assert got > 0, "test is vacuous if nothing falls inside the shell"


def test_band_mask_boundary_is_inclusive(monkeypatch):
    """|k| == k_max is INSIDE the admissible band: <=, not <.

    A random box makes exact boundary equality a measure-zero event, so
    test_band_mask_counts_match_a_brute_force_k_shell above cannot exercise
    this. Isolates band_mask's own comparison from the FFT frequency
    arithmetic by substituting k2 with values chosen to be exactly
    representable in floating point, so the boundary case is exact rather
    than a coin flip against rounding.
    """
    ops = SpectralOps((4, 4, 4), n_species=1, nyquist_mask=True)
    fixed_k2 = torch.tensor([0.0, 1.0, 4.0, 9.0])
    monkeypatch.setattr(ops, "k2", lambda boxes: fixed_k2)
    mask = ops.band_mask(torch.zeros(1, 3), k_max=2.0)  # k_max^2 == 4.0 exactly
    assert not bool(mask[0])   # k2 == 0: excluded regardless (the DC mode)
    assert bool(mask[1])       # k2 == 1.0 < 4.0: inside
    assert bool(mask[2])       # k2 == 4.0 == k_max^2: INSIDE, boundary inclusive
    assert not bool(mask[3])   # k2 == 9.0 > 4.0: outside


# ---------------------------------------------------------------------------
# per-sample boxes
# ---------------------------------------------------------------------------

def test_k_axes_are_per_sample():
    """Two different boxes in one batch must give two different k grids.

    A single-box implementation passes every test built on a uniform batch,
    and the first system trains on two grid shapes -- and per-sample,
    breathing NPT boxes -- at once.
    """
    ops = SpectralOps(EVEN_GRID, n_species=1, nyquist_mask=True)
    boxes = torch.tensor([[10.0, 10.0, 10.0], [20.0, 10.0, 10.0]])
    kx, ky, kz = ops.k_axes(boxes)
    # Different box[0] between the two samples must show up as different kx
    # somewhere (any nonzero-mode-x location), while ky/kz, whose boxes
    # agree, must match exactly.
    assert not torch.allclose(kx[0], kx[1])
    assert torch.allclose(ky[0], ky[1])
    assert torch.allclose(kz[0], kz[1])


def test_k2_uniform_batch_still_works():
    """A single-box implementation passes this one; it is not sufficient
    alone, but it must still pass."""
    ops = SpectralOps(EVEN_GRID, n_species=2, nyquist_mask=True)
    boxes = torch.tensor([[10.0, 10.0, 10.0]] * 4)
    k2 = ops.k2(boxes)
    assert k2.shape[0] == 4
    assert torch.allclose(k2[0], k2[-1])


# ---------------------------------------------------------------------------
# ops_for / OpsCache
# ---------------------------------------------------------------------------

def test_ops_cache_returns_distinct_ops_for_distinct_shapes():
    """The cache is keyed by shape. One entry per run would silently reuse a
    slab's k grid for a cube."""
    slab_grid = (16, 16, 64)
    cube_grid = (24, 24, 24)
    cache = OpsCache(slab_grid, n_species=2, nyquist_mask=True)

    slab_shape = (4, 2, 16, 16, slab_grid[2] // 2 + 1)
    cube_shape = (4, 2, 24, 24, cube_grid[2] // 2 + 1)

    slab_ops = cache.ops_for(slab_shape)
    cube_ops = cache.ops_for(cube_shape)

    assert slab_ops.grid == slab_grid
    assert cube_ops.grid == cube_grid
    assert slab_ops is not cube_ops
    # The seed grid's ops is the registered one -- identity, not just
    # equality, matters here because that is what carries the submodule
    # registration a caller relies on for .to()/state_dict().
    assert slab_ops is cache.ops


def test_ops_cache_hits_on_repeated_shape():
    cache = OpsCache((16, 16, 64), n_species=2, nyquist_mask=True)
    cube_shape = (4, 2, 24, 24, 13)
    first = cache.ops_for(cube_shape)
    second = cache.ops_for(cube_shape)
    assert first is second


def test_ops_cache_both_shapes_reachable_in_one_step():
    """Mirrors the actual failure mode: slabs (16,16,33) and cubes
    (24,24,13) rfft shapes reached from the SAME cache in the SAME step."""
    cache = OpsCache((16, 16, 64), n_species=2, nyquist_mask=True)
    slab_ops = cache.ops_for((4, 2, 16, 16, 33))
    cube_ops = cache.ops_for((4, 2, 24, 24, 13))
    assert slab_ops.grid == (16, 16, 64)
    assert cube_ops.grid == (24, 24, 24)
    # Both still independently retrievable afterward, i.e. neither entry
    # evicted the other.
    assert cache.ops_for((4, 2, 16, 16, 33)) is slab_ops
    assert cache.ops_for((4, 2, 24, 24, 13)) is cube_ops


# ---------------------------------------------------------------------------
# grad_hat / div_hat -- odd-order Nyquist masking
# ---------------------------------------------------------------------------

def test_grad_hat_zeros_the_derivative_at_a_nyquist_plane():
    """For a real field on an even axis, i*k of the (real) Nyquist
    coefficient is imaginary -- an inadmissible, non-Hermitian mode -- so the
    odd-order derivative must zero it. A non-Nyquist row must NOT be zeroed,
    so this also catches a mask applied uniformly rather than per axis."""
    grid = (6, 5, 8)  # Gx even (x-Nyquist at index 3), Gz even (z-Nyquist last)
    ops = SpectralOps(grid, n_species=1, nyquist_mask=True)
    boxes = torch.tensor([[10.0, 9.0, 8.0]], dtype=torch.float64)
    kx, ky, kz = ops.k_axes(boxes)
    torch.manual_seed(3)
    f_hat = torch.randn(1, 1, grid[0], grid[1], ops.Gzr, dtype=torch.complex128)
    fx_hat, fy_hat, fz_hat = ops.grad_hat(f_hat, kx, ky, kz)
    nyq_x = grid[0] // 2
    nyq_z = ops.Gzr - 1
    assert torch.all(fx_hat[:, :, nyq_x, :, :] == 0)
    assert torch.all(fz_hat[:, :, :, :, nyq_z] == 0)
    # Index 1 (mode n=1, k != 0, not the Nyquist row) must NOT be zeroed;
    # index 0 (k=0, the DC mode) is a poor choice for this check since it is
    # trivially zero from k alone, independent of the mask.
    assert not torch.all(fx_hat[:, :, 1, :, :] == 0)
    assert not torch.all(fz_hat[:, :, :, :, 1] == 0)


# ---------------------------------------------------------------------------
# buffer persistence -- checkpoint-compat contract
# ---------------------------------------------------------------------------

def test_odd_masks_are_not_persistent():
    """MX_ODD/MY_ODD/MZ_ODD must stay out of state_dict() so state_dict keys
    are stable across variant flags and old checkpoints keep loading."""
    ops = SpectralOps(EVEN_GRID, n_species=1, nyquist_mask=True)
    sd_keys = set(ops.state_dict().keys())
    for name in ("MX_ODD", "MY_ODD", "MZ_ODD"):
        assert name not in sd_keys, f"{name} leaked into state_dict()"
    buffer_names = {n for n, _ in ops.named_buffers()}
    for name in ("MX_ODD", "MY_ODD", "MZ_ODD"):
        assert name in buffer_names, f"{name} missing as a buffer entirely"


def test_index_and_multiplicity_buffers_are_persistent():
    ops = SpectralOps(EVEN_GRID, n_species=1, nyquist_mask=True)
    sd_keys = set(ops.state_dict().keys())
    for name in ("NX", "NY", "NZ", "MULT"):
        assert name in sd_keys, f"{name} dropped from state_dict()"


# ---------------------------------------------------------------------------
# laplacian
# ---------------------------------------------------------------------------

def test_laplacian_matches_analytic_second_derivative_of_a_plane_wave():
    """f(x) = sin(k0 x) has laplacian -k0^2 f(x); check the k-space operator
    reproduces that after a round trip through real space."""
    grid = (16, 6, 6)
    ops = SpectralOps(grid, n_species=1, nyquist_mask=True)
    Lx = 10.0
    boxes = torch.tensor([[Lx, 6.0, 6.0]], dtype=torch.float64)
    n = 2  # an admissible integer mode, well inside the grid
    k0 = TWO_PI * n / Lx
    # Periodic sample points for a box of length Lx on Gx points: spacing
    # Lx / Gx, NOT Lx / (Gx - 1) -- linspace's inclusive endpoint duplicates
    # the periodic image of x=0 at x=Lx and is not what rfftn assumes.
    x = torch.arange(grid[0], dtype=torch.float64) * (Lx / grid[0])
    f = torch.sin(k0 * x).view(1, 1, grid[0], 1, 1).expand(1, 1, *grid).clone()

    k2 = ops.k2(boxes)
    f_hat = ops.rfft(f)
    lap_hat = ops.laplacian(f_hat, k2)
    lap = ops.irfft(lap_hat)

    expected = -(k0 ** 2) * f
    assert torch.allclose(lap, expected, atol=1e-6)


def test_laplacian_is_minus_k2_times_field_pointwise():
    ops = SpectralOps(EVEN_GRID, n_species=2, nyquist_mask=True)
    boxes = torch.tensor([[10.0, 11.0, 9.0]], dtype=torch.float64)
    k2 = ops.k2(boxes)
    torch.manual_seed(2)
    f_hat = torch.randn(1, 2, *EVEN_GRID[:2], ops.Gzr,
                        dtype=torch.complex128)
    got = ops.laplacian(f_hat, k2)
    assert torch.allclose(got, -k2 * f_hat)


# The Nyquist mask is declared --------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, 1, 0, "yes"])
def test_a_nyquist_mask_that_is_not_a_bool_is_refused_naming_both(value):
    from aipf.spectral import check_nyquist_mask
    with pytest.raises(ValueError, match=r"\(True, False\)"):
        check_nyquist_mask(value)


def test_an_operator_set_without_a_nyquist_declaration_is_refused():
    with pytest.raises(TypeError, match="nyquist_mask"):
        OpsCache(EVEN_GRID, 1)


def test_without_the_mask_grad_and_div_keep_the_nyquist_wavenumber():
    masked = OpsCache(EVEN_GRID, 1, nyquist_mask=True).ops
    plain = OpsCache(EVEN_GRID, 1, nyquist_mask=False).ops
    boxes = torch.tensor([[2.0, 3.0, 4.0]])
    kx, ky, kz = plain.k_axes(boxes)
    f_hat = torch.ones(1, 1, *EVEN_GRID[:2], EVEN_GRID[2] // 2 + 1, dtype=torch.complex64)
    gx, _, gz = plain.grad_hat(f_hat, kx, ky, kz)
    assert torch.equal(gx, 1j * kx * f_hat) and torch.equal(gz, 1j * kz * f_hat)
    assert gz[..., -1].abs().max() > 0
    mx, _, mz = masked.grad_hat(f_hat, kx, ky, kz)
    assert mz[..., -1].abs().max() == 0
    div = plain.div_hat(f_hat, f_hat, f_hat, kx, ky, kz)
    assert torch.equal(div, 1j * (kx * f_hat + ky * f_hat + kz * f_hat))


def test_every_grid_an_operator_set_grows_to_carries_its_declaration():
    cache = OpsCache(EVEN_GRID, 1, nyquist_mask=False)
    other = cache.ops_for_grid((4, 6, 8))
    assert other.nyquist_mask is False
    assert all(bool((getattr(other, m) == 1).all()) for m in ("MX_ODD", "MY_ODD", "MZ_ODD"))
    assert cache.ops_for_grid((4, 6, 8), "cpu").nyquist_mask is False


def test_a_bare_operator_set_without_a_nyquist_declaration_is_refused():
    with pytest.raises(TypeError, match="nyquist_mask"):
        SpectralOps(EVEN_GRID, 1)
    with pytest.raises(ValueError, match=r"\(True, False\)"):
        SpectralOps(EVEN_GRID, 1, nyquist_mask=None)
