"""Tests for aipf.pipeline.kde: atom positions to a smooth density field.

Properties, not stored buffers, following ``tests/unit/test_kernels.py``:
wherever a result computed one way can be checked against the same result
computed an independent way -- the defining sum, the analytic image sum, the
atom count the field has to integrate to -- it is, rather than pinning
literal numbers.

``n_species`` is exercised at 1 and 2 throughout, per the plan's global
constraint. The two counts are not the same code path here in one respect
that matters: at 1 there is a single channel loop iteration and every
cross-channel rule (ordering, permutation, the contrast in the cumulant) is
a one-element statement, which is exactly where a degenerate implementation
hides.

The archived fields themselves are rebuilt end to end in
``test_pipeline_coarse_grain.py`` (marked ``env``: the H/He raw root).
"""
from __future__ import annotations

import math

import pytest
import torch

from aipf.pipeline.kde import (
    DEVICES,
    METHODS,
    binder_cumulant,
    channel_atom_types,
    density_field,
)

torch.manual_seed(0)

BOX = ((0.0, 6.0), (0.0, 6.0), (0.0, 6.0))
CUT = 5.0  # widths; the value the archived fields were produced with


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _frame(counts, types, box=BOX, seed=0):
    """A random frame: ``counts[c]`` atoms of dump type ``types[c]``."""
    gen = torch.Generator().manual_seed(seed)
    lo = torch.tensor([b[0] for b in box], dtype=torch.float64)
    length = torch.tensor([b[1] - b[0] for b in box], dtype=torch.float64)
    pos, typ = [], []
    for dump_type, n in zip(types, counts):
        pos.append(lo + length * torch.rand((n, 3), generator=gen,
                                            dtype=torch.float64))
        typ.append(torch.full((n,), int(dump_type), dtype=torch.long))
    return torch.cat(pos), torch.cat(typ)


def _cell_volume(box, grid):
    out = 1.0
    for (lo, hi), g in zip(box, grid):
        out *= (hi - lo) / g
    return out


def _integral(field, box, grid):
    """Per-channel integral of the field over the box: the atom count."""
    return field.sum(dim=(1, 2, 3)).to(torch.float64) * _cell_volume(box, grid)


def _periodic_gaussian(point, centre, box, sigma, n_images=4):
    """sum over periodic images of the normalised Gaussian, in float64."""
    total = 0.0
    norm = 1.0 / (2.0 * math.pi * sigma * sigma) ** 1.5
    lengths = [hi - lo for lo, hi in box]
    for ix in range(-n_images, n_images + 1):
        for iy in range(-n_images, n_images + 1):
            for iz in range(-n_images, n_images + 1):
                shift = (ix * lengths[0], iy * lengths[1], iz * lengths[2])
                d2 = sum((point[a] - centre[a] - shift[a]) ** 2
                         for a in range(3))
                total += norm * math.exp(-d2 / (2.0 * sigma * sigma))
    return total


ONE = ((1,), (8,))       # (atom_types, counts) for a one-channel frame
TWO = ((1, 2), (8, 5))   # ... and a two-channel one
BOTH = [pytest.param(ONE, id="n_species=1"), pytest.param(TWO, id="n_species=2")]


# ---------------------------------------------------------------------------
# shape, channels and the declared type mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", BOTH)
@pytest.mark.parametrize("method", METHODS)
def test_output_is_channel_first_with_one_channel_per_declared_type(case, method):
    types, counts = case
    pos, typ = _frame(counts, types)
    extra = {"cutoff_in_sigma": CUT} if method == "direct" else {}
    out = density_field(pos, typ, BOX, 8, sigma=1.0, atom_types=types,
                        method=method, device="cpu", **extra)
    assert out.shape == (len(types), 8, 8, 8)


@pytest.mark.parametrize("case", BOTH)
@pytest.mark.parametrize("method", METHODS)
def test_anisotropic_grid_is_honoured_per_axis(case, method):
    types, counts = case
    box = ((0.0, 4.0), (0.0, 4.0), (0.0, 12.0))
    pos, typ = _frame(counts, types, box=box)
    extra = {"cutoff_in_sigma": CUT} if method == "direct" else {}
    out = density_field(pos, typ, box, (4, 5, 12), sigma=1.0, atom_types=types,
                        method=method, device="cpu", **extra)
    assert out.shape == (len(types), 4, 5, 12)


@pytest.mark.parametrize("case", BOTH)
@pytest.mark.parametrize("method", METHODS)
def test_channel_order_follows_the_declared_type_order(case, method):
    """The channels are the caller's, in the caller's order.

    At one channel this says the single channel is the declared type and not
    whatever type happens to come first in the dump.
    """
    types, counts = case
    pos, typ = _frame(counts, types)
    extra = {"cutoff_in_sigma": CUT} if method == "direct" else {}
    kw = dict(sigma=1.0, method=method, device="cpu",
              dtype=torch.float64, **extra)
    straight = density_field(pos, typ, BOX, 8, atom_types=types, **kw)
    reversed_types = tuple(reversed(types))
    flipped = density_field(pos, typ, BOX, 8, atom_types=reversed_types, **kw)
    assert torch.equal(flipped, straight.flip(0))


@pytest.mark.parametrize("method", METHODS)
def test_a_dump_type_that_is_no_channel_is_ignored(method):
    """A dump may carry a type that is not a density channel.

    The one-channel systems in this repository read exactly such a dump, so
    "every atom in the file" and "every atom of my channels" are different
    sets, and taking the first for the second doubles a density.
    """
    pos, typ = _frame((8, 5), (1, 2))
    extra = {"cutoff_in_sigma": CUT} if method == "direct" else {}
    kw = dict(sigma=1.0, method=method, device="cpu",
              dtype=torch.float64, **extra)
    one = density_field(pos, typ, BOX, 8, atom_types=(1,), **kw)
    two = density_field(pos, typ, BOX, 8, atom_types=(1, 2), **kw)
    assert torch.equal(one[0], two[0])


@pytest.mark.parametrize("case", BOTH)
@pytest.mark.parametrize("method", METHODS)
def test_a_channel_with_no_atoms_is_zero_not_absent(case, method):
    types, counts = case
    pos, typ = _frame(counts, types)
    absent = tuple(types) + (99,)
    extra = {"cutoff_in_sigma": CUT} if method == "direct" else {}
    out = density_field(pos, typ, BOX, 8, sigma=1.0, atom_types=absent,
                        method=method, device="cpu", **extra)
    assert out.shape[0] == len(types) + 1
    assert torch.equal(out[-1], torch.zeros_like(out[-1]))


# ---------------------------------------------------------------------------
# the physics: what the field is
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", BOTH)
def test_mesh_field_integrates_to_the_atom_count(case):
    """The Gaussian is normalised and the mesh method loses nothing.

    Deposition conserves the count exactly and the k=0 mode of both the
    Gaussian and the shape factor is exactly one, so this is an equality,
    not an approximation.
    """
    types, counts = case
    pos, typ = _frame(counts, types)
    out = density_field(pos, typ, BOX, 12, sigma=0.9, atom_types=types,
                        method="mesh", device="cpu", dtype=torch.float64)
    got = _integral(out, BOX, (12, 12, 12))
    want = torch.tensor(counts, dtype=torch.float64)
    assert torch.allclose(got, want, rtol=1e-10)


@pytest.mark.parametrize("case", BOTH)
def test_direct_field_integrates_to_the_atom_count_up_to_its_truncation(case):
    """The direct method drops the tail past the cutoff, so it under-counts.

    The miss is what the cutoff is chosen to make negligible, and this pins
    that it is negligible rather than assuming it. The mass of an isotropic
    three-dimensional Gaussian beyond five widths is 1.54e-5 of it; measured
    here, with the box wide enough that the nearest image is the only one
    inside the cutoff, 1.56e-5.
    """
    types, counts = case
    pos, typ = _frame(counts, types)
    out = density_field(pos, typ, BOX, 24, sigma=0.4, atom_types=types,
                        method="direct", device="cpu", dtype=torch.float64,
                        cutoff_in_sigma=CUT)
    got = _integral(out, BOX, (24, 24, 24))
    want = torch.tensor(counts, dtype=torch.float64)
    assert torch.allclose(got, want, rtol=5e-5)
    assert bool((got < want).all())


def test_a_single_atom_reproduces_the_periodic_gaussian():
    """The defining object, checked against an explicit image sum.

    One atom, a grid that resolves the kernel with eight points per width,
    and the analytic sum over periodic images evaluated at three nodes.
    Measured agreement there is 5e-4 relative; the residual is the aliasing
    the cloud-in-cell deconvolution cannot remove, and it grows to about two
    per cent at two points per width, which is why a caller choosing a grid
    is choosing an accuracy.
    """
    box = ((0.0, 6.0), (0.0, 6.0), (0.0, 6.0))
    sigma = 1.0
    centre = (1.7, 3.1, 4.4)
    pos = torch.tensor([centre], dtype=torch.float64)
    typ = torch.tensor([1])
    grid = (48, 48, 48)
    out = density_field(pos, typ, box, grid, sigma=sigma, atom_types=(1,),
                        method="mesh", device="cpu", dtype=torch.float64)
    step = [(hi - lo) / g for (lo, hi), g in zip(box, grid)]
    for idx in [(0, 0, 0), (16, 24, 6), (47, 47, 47)]:
        point = [box[a][0] + idx[a] * step[a] for a in range(3)]
        want = _periodic_gaussian(point, centre, box, sigma)
        got = float(out[0][idx])
        assert got == pytest.approx(want, rel=2e-3, abs=1e-12)


def test_the_two_methods_agree_where_both_are_valid():
    """Independent implementations of the same integral.

    Valid here means the box is wide enough that no second image reaches
    inside the cutoff, so the direct sum's two approximations are both
    switched off and only the mesh method's aliasing is left. Measured
    largest difference is 3.4e-3 of the field's peak at four points per
    width, falling to 2.4e-3 at five.
    """
    pos, typ = _frame((30, 20), (1, 2))
    kw = dict(sigma=0.5, atom_types=(1, 2), device="cpu", dtype=torch.float64)
    mesh = density_field(pos, typ, BOX, 48, method="mesh", **kw)
    direct = density_field(pos, typ, BOX, 48, method="direct",
                           cutoff_in_sigma=CUT, block_size=8192, **kw)
    scale = float(mesh.abs().max())
    assert float((mesh - direct).abs().max()) < 6e-3 * scale


@pytest.mark.parametrize("case", BOTH)
@pytest.mark.parametrize("method", METHODS)
def test_shifting_every_atom_by_one_cell_rolls_the_field(case, method):
    """Translation covariance on the grid, exactly.

    A whole-cell shift is representable, so this is an equality and not a
    tolerance; it fails the moment an axis is mixed up or a stride is wrong.
    """
    types, counts = case
    pos, typ = _frame(counts, types)
    grid = (8, 10, 12)
    step = torch.tensor([(hi - lo) / g for (lo, hi), g in zip(BOX, grid)],
                        dtype=torch.float64)
    shift = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64) * step
    extra = {"cutoff_in_sigma": CUT} if method == "direct" else {}
    kw = dict(sigma=1.0, atom_types=types, method=method, device="cpu",
              dtype=torch.float64, **extra)
    base = density_field(pos, typ, BOX, grid, **kw)
    moved = density_field(pos + shift, typ, BOX, grid, **kw)
    rolled = torch.roll(base, shifts=(1, 2, 3), dims=(1, 2, 3))
    assert torch.allclose(moved, rolled, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("case", BOTH)
@pytest.mark.parametrize("method", METHODS)
def test_an_atom_outside_the_box_is_the_same_atom(case, method):
    """A dump need not be wrapped, and an unwrapped atom is not a new atom."""
    types, counts = case
    pos, typ = _frame(counts, types)
    lengths = torch.tensor([hi - lo for lo, hi in BOX], dtype=torch.float64)
    outside = pos.clone()
    outside[0] += lengths
    outside[-1] -= 2.0 * lengths
    extra = {"cutoff_in_sigma": CUT} if method == "direct" else {}
    kw = dict(sigma=1.0, atom_types=types, method=method, device="cpu",
              dtype=torch.float64, **extra)
    base = density_field(pos, typ, BOX, 8, **kw)
    wrapped = density_field(outside, typ, BOX, 8, **kw)
    assert torch.allclose(wrapped, base, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("case", BOTH)
@pytest.mark.parametrize("method", METHODS)
def test_the_inputs_are_not_modified(case, method):
    types, counts = case
    pos, typ = _frame(counts, types)
    pos_before, typ_before = pos.clone(), typ.clone()
    extra = {"cutoff_in_sigma": CUT} if method == "direct" else {}
    density_field(pos, typ, BOX, 8, sigma=1.0, atom_types=types,
                  method=method, device="cpu", **extra)
    assert torch.equal(pos, pos_before)
    assert torch.equal(typ, typ_before)


def test_block_size_is_memory_only_and_cannot_move_the_answer():
    pos, typ = _frame((30, 20), (1, 2))
    kw = dict(sigma=1.0, atom_types=(1, 2), method="direct", device="cpu",
              dtype=torch.float64, cutoff_in_sigma=CUT)
    small = density_field(pos, typ, BOX, 8, block_size=7, **kw)
    large = density_field(pos, typ, BOX, 8, block_size=100000, **kw)
    assert torch.equal(small, large)


@pytest.mark.parametrize("case", BOTH)
def test_the_working_dtype_is_the_callers(case):
    types, counts = case
    pos, typ = _frame(counts, types)
    kw = dict(sigma=1.0, atom_types=types, method="mesh", device="cpu")
    assert density_field(pos, typ, BOX, 8, **kw).dtype is torch.float32
    assert density_field(pos, typ, BOX, 8, dtype=torch.float64,
                         **kw).dtype is torch.float64


@pytest.mark.parametrize("case", BOTH)
def test_a_wider_kernel_is_a_flatter_field(case):
    """sigma is a caller's knob with a direction, not decoration."""
    types, counts = case
    pos, typ = _frame(counts, types)
    kw = dict(atom_types=types, method="mesh", device="cpu",
              dtype=torch.float64)
    sharp = density_field(pos, typ, BOX, 16, sigma=0.5, **kw)
    smooth = density_field(pos, typ, BOX, 16, sigma=1.5, **kw)
    assert float(smooth.std()) < float(sharp.std())


def test_numpy_input_is_accepted():
    """Trajectory readers hand out arrays, not tensors."""
    np = pytest.importorskip("numpy")
    pos, typ = _frame((8, 5), (1, 2))
    kw = dict(sigma=1.0, atom_types=(1, 2), method="mesh", device="cpu",
              dtype=torch.float64)
    from_torch = density_field(pos, typ, BOX, 8, **kw)
    from_numpy = density_field(pos.numpy(), typ.numpy(),
                               np.array(BOX, dtype=float), 8, **kw)
    assert torch.equal(from_numpy, from_torch)


def test_device_auto_runs_wherever_it_is():
    pos, typ = _frame((8, 5), (1, 2))
    out = density_field(pos, typ, BOX, 8, sigma=1.0, atom_types=(1, 2),
                        method="mesh", device="auto")
    assert out.shape == (2, 8, 8, 8)


def test_each_axis_uses_its_own_grid_spacing():
    """An anisotropic grid is three kernels, not one applied three times.

    The wavenumbers of an axis are set by that axis' spacing. Sharing one
    axis' spacing across all three leaves the shape of the output and every
    symmetry of it intact -- it is still a convolution with still a
    Gaussian -- and only the VALUES move, by 49 per cent at the nodes
    nearest the atom here, against 0.3 per cent for the correct kernel.
    """
    box = ((0.0, 6.0), (0.0, 6.0), (0.0, 12.0))
    grid = (48, 48, 48)  # spacings 0.125, 0.125, 0.25
    sigma = 1.0
    centre = (1.7, 3.1, 4.4)
    pos = torch.tensor([centre], dtype=torch.float64)
    typ = torch.tensor([1])
    out = density_field(pos, typ, box, grid, sigma=sigma, atom_types=(1,),
                        method="mesh", device="cpu", dtype=torch.float64)
    step = [(hi - lo) / g for (lo, hi), g in zip(box, grid)]
    for idx in [(14, 25, 18), (13, 24, 17), (14, 25, 14), (14, 25, 22),
                (10, 25, 18), (18, 25, 18)]:
        point = [box[a][0] + idx[a] * step[a] for a in range(3)]
        want = _periodic_gaussian(point, centre, box, sigma)
        assert float(out[0][idx]) == pytest.approx(want, rel=1e-2)


@pytest.mark.parametrize("case", BOTH)
@pytest.mark.parametrize("method", METHODS)
def test_moving_the_box_and_the_atoms_together_changes_nothing(case, method):
    """The origin is part of the box, not an assumption that it is zero.

    Every archived box in this project starts at zero on every axis, so
    nothing else here would notice a grid built from the spacing alone.
    """
    types, counts = case
    pos, typ = _frame(counts, types)
    shift = torch.tensor([1.5, -2.5, 0.75], dtype=torch.float64)
    moved_box = tuple((lo + float(shift[a]), hi + float(shift[a]))
                      for a, (lo, hi) in enumerate(BOX))
    extra_kw = {"cutoff_in_sigma": CUT} if method == "direct" else {}
    kw = dict(sigma=1.0, atom_types=types, method=method, device="cpu",
              dtype=torch.float64, **extra_kw)
    here = density_field(pos, typ, BOX, 8, **kw)
    there = density_field(pos + shift, typ, moved_box, 8, **kw)
    assert torch.allclose(there, here, rtol=1e-12, atol=1e-14)


def test_an_atom_a_hair_below_the_origin_is_the_atom_at_the_origin():
    """The wrapped coordinate that rounds up to exactly one grid length.

    At single precision an atom 1e-07 below the origin wraps to exactly
    G, whose floor is G and not G-1, and the cell index that comes out has
    to be folded back to zero. Without that fold the deposition indexes off
    the end of its own buffer.
    """
    typ = torch.tensor([1])
    kw = dict(sigma=1.0, atom_types=(1,), method="mesh", device="cpu")
    below = density_field(torch.tensor([[-1e-7, 0.5, 0.5]],
                                       dtype=torch.float64), typ, BOX, 8, **kw)
    at = density_field(torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float64),
                       typ, BOX, 8, **kw)
    assert torch.equal(below, at)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def _ok(**overrides):
    pos, typ = _frame((8, 5), (1, 2))
    kw = dict(positions=pos, types=typ, box_bounds=BOX, grid=8, sigma=1.0,
              atom_types=(1, 2), method="mesh", device="cpu")
    kw.update(overrides)
    return density_field(**kw)


def test_an_unknown_method_names_the_known_ones():
    with pytest.raises(ValueError, match="mesh"):
        _ok(method="fft")


def test_an_unknown_device_names_the_known_ones():
    with pytest.raises(ValueError, match="auto"):
        _ok(device="gpu")


def test_an_unavailable_accelerator_refuses_rather_than_falling_back(monkeypatch):
    """Falling back silently changes the arithmetic under the caller.

    A silent fallback would make a run that asked for the accelerator and got
    the processor indistinguishable from one that never asked. Measured on an archived frame, the two
    devices' fields differ by 4.0e-07 of the peak, so the difference is
    real and small, which is the worst combination to leave unannounced.

    Availability is patched rather than read, so the rule is checked on a
    host that has an accelerator as well as on one that does not. Read off
    the host, this test skipped exactly where it was most needed and the
    mutation that deletes the refusal survived the sweep.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        _ok(device="cuda")


def test_auto_takes_the_processor_when_no_accelerator_is_visible(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _ok(device="auto").device.type == "cpu"


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_kernel_width_that_is_not_positive_is_refused(bad):
    with pytest.raises(ValueError, match="sigma"):
        _ok(sigma=bad)


def test_a_grid_that_is_not_three_axes_is_refused():
    with pytest.raises(ValueError, match="grid"):
        _ok(grid=(8, 8))


def test_a_grid_axis_below_one_is_refused():
    with pytest.raises(ValueError, match="grid"):
        _ok(grid=(8, 0, 8))


def test_positions_that_are_not_three_dimensional_are_refused():
    with pytest.raises(ValueError, match="positions"):
        _ok(positions=torch.zeros((5, 2)), types=torch.ones(5, dtype=torch.long))


def test_a_type_per_atom_is_required():
    with pytest.raises(ValueError, match="types"):
        _ok(types=torch.ones(3, dtype=torch.long))


def test_no_channels_at_all_is_refused():
    with pytest.raises(ValueError, match="atom_types"):
        _ok(atom_types=())


def test_a_repeated_dump_type_is_refused():
    """Two channels fed by one type is a silently doubled density."""
    with pytest.raises(ValueError, match="atom_types"):
        _ok(atom_types=(1, 1))


def test_a_box_that_is_not_three_by_two_is_refused():
    with pytest.raises(ValueError, match="box_bounds"):
        _ok(box_bounds=((0.0, 6.0), (0.0, 6.0)))


def test_a_box_edge_that_is_not_positive_is_refused():
    with pytest.raises(ValueError, match="box_bounds"):
        _ok(box_bounds=((0.0, 6.0), (4.0, 4.0), (0.0, 6.0)))


def test_the_direct_method_requires_its_cutoff():
    with pytest.raises(ValueError, match="cutoff_in_sigma"):
        _ok(method="direct")


def test_the_mesh_method_refuses_a_cutoff_it_cannot_honour():
    """Accepting and ignoring a parameter is how a wrong number goes green.

    A mesh function that kept a batch size in its signature and deleted it
    on the first line would be exactly that.
    """
    with pytest.raises(ValueError, match="cutoff_in_sigma"):
        _ok(cutoff_in_sigma=CUT)


@pytest.mark.parametrize("bad", [0.0, -2.0])
def test_a_cutoff_that_is_not_positive_is_refused(bad):
    with pytest.raises(ValueError, match="cutoff_in_sigma"):
        _ok(method="direct", cutoff_in_sigma=bad)


def test_a_block_size_below_one_is_refused():
    with pytest.raises(ValueError, match="block_size"):
        _ok(method="direct", cutoff_in_sigma=CUT, block_size=0)


# ---------------------------------------------------------------------------
# the cumulant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_a_uniform_field_is_fully_ordered(n_species):
    """A constant contrast field has <psi^4> = <psi^2>^2, so U = 2/3."""
    field = torch.full((n_species, 4, 4, 4), 0.25, dtype=torch.float64)
    contrast = (1.0,) if n_species == 1 else (1.0, 0.0)
    assert binder_cumulant(field, contrast=contrast) == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize("n_species", [1, 2])
def test_an_identically_zero_contrast_reports_zero(n_species):
    field = torch.zeros((n_species, 4, 4, 4), dtype=torch.float64)
    contrast = (1.0,) * n_species
    assert binder_cumulant(field, contrast=contrast) == 0.0


@pytest.mark.parametrize("n_species", [1, 2])
def test_a_gaussian_contrast_field_is_unordered(n_species):
    """U -> 0 for a Gaussian field: <psi^4> = 3 <psi^2>^2 exactly."""
    gen = torch.Generator().manual_seed(3)
    field = torch.randn((n_species, 40, 40, 40), generator=gen,
                        dtype=torch.float64)
    contrast = (1.0,) if n_species == 1 else (1.0, -1.0)
    assert abs(binder_cumulant(field, contrast=contrast)) < 0.02


def test_the_contrast_is_the_callers_and_a_sign_flip_changes_nothing():
    gen = torch.Generator().manual_seed(4)
    field = torch.randn((2, 8, 8, 8), generator=gen, dtype=torch.float64)
    up = binder_cumulant(field, contrast=(1.0, -1.0))
    down = binder_cumulant(field, contrast=(-1.0, 1.0))
    assert up == pytest.approx(down, rel=1e-12)


def test_the_accumulation_precision_is_not_the_fields_precision():
    """A fourth moment is accumulated in double whatever the field is.

    The same field, once as it is and once widened, has to give exactly the
    same number: if the moments were accumulated at the field's own
    precision the two would differ, measured, by 3e-06 relative on a
    48-cubed field and 7e-06 on one with an offset.
    """
    gen = torch.Generator().manual_seed(1)
    wide = 10.0 + torch.randn((2, 48, 48, 48), generator=gen,
                              dtype=torch.float64)
    narrow = wide.to(torch.float32)
    # Weights that are not exactly representable, on purpose. At plus and
    # minus one the contrast is formed by exact multiplications whatever the
    # precision, and this test passed while the widening it exists to pin was
    # deleted -- the accumulator alone was enough for those two weights.
    contrast = (0.3, -0.7)
    assert (binder_cumulant(narrow, contrast=contrast)
            == binder_cumulant(narrow.to(torch.float64), contrast=contrast))


def test_the_contrast_must_have_one_weight_per_channel():
    field = torch.zeros((2, 4, 4, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="contrast"):
        binder_cumulant(field, contrast=(1.0,))


def test_the_cumulant_wants_a_channel_first_field():
    field = torch.zeros((4, 4, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="density"):
        binder_cumulant(field, contrast=(1.0,))


def test_a_demixed_field_is_more_ordered_than_a_mixed_one():
    gen = torch.Generator().manual_seed(5)
    noise = 0.05 * torch.randn((2, 16, 16, 16), generator=gen,
                               dtype=torch.float64)
    mixed = torch.full((2, 16, 16, 16), 0.5, dtype=torch.float64) + noise
    demixed = mixed.clone()
    demixed[0, :8] += 0.5
    demixed[1, :8] -= 0.5
    demixed[0, 8:] -= 0.5
    demixed[1, 8:] += 0.5
    assert (binder_cumulant(demixed, contrast=(1.0, -1.0))
            > binder_cumulant(mixed, contrast=(1.0, -1.0)))


# ---------------------------------------------------------------------------
# the channel/type mapping a System declares
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_channel_atom_types_reads_the_declaration_in_channel_order(n_species):
    from aipf.paths import Paths
    from aipf.system import AnchorRules, System

    species = ("A", "B")[:n_species]
    # Deliberately NOT channel index plus one, and deliberately carrying a
    # type that is no channel: both are legitimate and both are what a
    # positional assumption gets wrong.
    system = System(
        name="demo", n_species=n_species, species=species,
        masses={s: 1.0 for s in species},
        atom_types={"A": 7, "B": 3, "C": 9},
        table_keys={}, paths=Paths(system="demo", raw_default="/absent/demo"),
        anchor_rules=AnchorRules({}), constants={}, defaults={},
    )
    assert channel_atom_types(system) == (7, 3)[:n_species]


def test_channel_atom_types_refuses_an_undeclared_mapping():
    from aipf.paths import Paths
    from aipf.system import AnchorRules, System

    system = System(
        name="demo", n_species=1, species=("A",), masses={}, atom_types={},
        table_keys={}, paths=Paths(system="demo", raw_default="/absent/demo"),
        anchor_rules=AnchorRules({}), constants={}, defaults={},
    )
    with pytest.raises(ValueError, match="atom_types"):
        channel_atom_types(system)


def test_the_declared_names_are_exported():
    assert METHODS == ("mesh", "direct")
    assert DEVICES == ("auto", "cpu", "cuda")
