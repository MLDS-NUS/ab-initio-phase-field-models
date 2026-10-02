"""Tests for aipf.pipeline.coarse_grain: a trajectory to training pairs.

Properties wherever a result can be produced a second, independent way --
a polynomial the Savitzky-Golay fit has to reproduce exactly, a dump whose
atom order is shuffled and must not matter, a composition recomputed by
hand -- rather than pinned literals, following ``tests/unit/test_kernels.py``
and ``tests/unit/test_pipeline_kde.py``.

``n_species`` is exercised at 1 and at 2 throughout, per the plan's global
constraint. It matters here in a specific way: the composition's denominator
sums the channels of the field it is handed, so the one-channel case is the
one where a numerator equal to the denominator has to still behave, and it is
also the real case for a system that declares one density channel against a
dump carrying two types.

The archive tests at the bottom are the acceptance criterion. They read an
archived trajectory and an archived output from the H/He raw root, are marked
``env`` and skip, naming the root, when it is absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import declared_roots
from aipf.pipeline.coarse_grain import (
    COMPOSITION_FORMS,
    EDGE_MODES,
    SPACINGS,
    Frame,
    box_lengths,
    composition_field,
    density_series,
    emit_indices,
    grid_for_spacing,
    read_dump,
    smooth_in_time,
)
from aipf.pipeline.kde import density_field

BOX = ((0.0, 6.0), (0.0, 6.0), (0.0, 6.0))

ONE = ((1,), (7,))        # (atom_types, counts) for a one-channel frame
TWO = ((1, 2), (7, 5))    # ... and a two-channel one
BOTH = [pytest.param(ONE, id="n_species=1"), pytest.param(TWO, id="n_species=2")]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _atoms(counts, types, box=BOX, seed=0):
    """Random positions and dump types: ``counts[c]`` atoms of ``types[c]``."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in box])
    length = np.array([b[1] - b[0] for b in box])
    pos, typ = [], []
    for dump_type, n in zip(types, counts):
        pos.append(lo + length * rng.random((n, 3)))
        typ.append(np.full(n, int(dump_type), dtype=np.int64))
    return np.concatenate(pos), np.concatenate(typ)


def _frame(counts, types, box=BOX, seed=0, timestep=0):
    pos, typ = _atoms(counts, types, box, seed)
    bounds = np.array([[b[0], b[1]] for b in box], dtype=np.float64)
    return Frame(timestep=timestep, box_bounds=bounds, positions=pos,
                 types=typ)


def _write_dump(path, frames, *, columns=("id", "type", "x", "y", "z"),
                element=False, shuffle=False, truncate_atoms=None,
                truncate_header=False, short_last_row=False):
    """A LAMMPS text dump, written from Frame objects.

    ``columns`` names the numeric columns in the order they are written, so a
    reordered or extended header can be exercised. ``element`` inserts a text
    column, which is what forces the row-by-row parse.
    """
    rng = np.random.default_rng(7)
    lines = []
    for f_i, frame in enumerate(frames):
        n = frame.positions.shape[0]
        lines += ["ITEM: TIMESTEP", str(frame.timestep),
                  "ITEM: NUMBER OF ATOMS", str(n),
                  "ITEM: BOX BOUNDS pp pp pp"]
        for axis in range(3):
            lines.append(f"{float(frame.box_bounds[axis, 0])!r} "
                         f"{float(frame.box_bounds[axis, 1])!r}")
        header = list(columns)
        if element:
            header = header[:2] + ["element"] + header[2:]
        lines.append("ITEM: ATOMS " + " ".join(header))
        order = np.arange(n)
        if shuffle:
            order = rng.permutation(n)
        rows = []
        for i in order:
            values = {
                "id": repr(float(i + 1)),
                "type": repr(float(frame.types[i])),
                "x": repr(float(frame.positions[i, 0])),
                "y": repr(float(frame.positions[i, 1])),
                "z": repr(float(frame.positions[i, 2])),
                "vx": repr(float(i)),
            }
            row = [values[c] for c in columns]
            if element:
                row = row[:2] + ["Xx"] + row[2:]
            rows.append(" ".join(row))
        if truncate_atoms is not None and f_i == len(frames) - 1:
            rows = rows[:truncate_atoms]
        if short_last_row and f_i == len(frames) - 1:
            rows[-1] = " ".join(rows[-1].split()[:-1])
        lines += rows
    text = "\n".join(lines) + "\n"
    if truncate_header:
        text = text[: text.rindex("ITEM: TIMESTEP") + 40]
    Path(path).write_text(text)
    return path


def _roundtrip(tmp_path, frames, **kwargs):
    path = _write_dump(tmp_path / "dump.lammpstrj", frames, **kwargs)
    return list(read_dump(path))


# ---------------------------------------------------------------------------
# the named choices
# ---------------------------------------------------------------------------

def test_the_named_choices_are_the_ones_the_archive_contains():
    """Choices are named in a module-level tuple, per the package convention."""
    assert SPACINGS == ("log", "uniform")
    assert EDGE_MODES == ("interp", "nearest")
    assert COMPOSITION_FORMS == ("offset", "floored", "filled")


# ---------------------------------------------------------------------------
# read_dump
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", BOTH)
def test_a_dump_round_trips_through_the_reader(tmp_path, spec):
    types, counts = spec
    frames = [_frame(counts, types, timestep=200 * i, seed=i) for i in range(3)]
    got = _roundtrip(tmp_path, frames)
    assert [f.timestep for f in got] == [0, 200, 400]
    for want, have in zip(frames, got):
        assert np.array_equal(have.box_bounds, want.box_bounds)
        assert np.array_equal(have.positions, want.positions)
        assert np.array_equal(have.types, want.types)
        assert have.types.dtype == np.int64
        assert have.positions.dtype == np.float64


def test_atoms_come_back_ordered_by_identifier(tmp_path):
    """A shuffled dump reads as the sorted one, or the field depends on the
    order the run happened to write its atoms in."""
    frames = [_frame((7, 5), (1, 2))]
    ordered = _roundtrip(tmp_path, frames)
    shuffled = _roundtrip(tmp_path, frames, shuffle=True)
    assert np.array_equal(shuffled[0].positions, ordered[0].positions)
    assert np.array_equal(shuffled[0].types, ordered[0].types)


def test_the_stride_keeps_one_frame_in_n(tmp_path):
    frames = [_frame((4,), (1,), timestep=10 * i, seed=i) for i in range(7)]
    path = _write_dump(tmp_path / "d.lammpstrj", frames)
    assert [f.timestep for f in read_dump(path, stride=3)] == [0, 30, 60]
    assert [f.timestep for f in read_dump(path, stride=1)] == \
        [10 * i for i in range(7)]


def test_a_stride_below_one_is_refused(tmp_path):
    path = _write_dump(tmp_path / "d.lammpstrj", [_frame((4,), (1,))])
    with pytest.raises(ValueError, match="stride"):
        list(read_dump(path, stride=0))


def test_columns_are_selected_by_name_not_by_position(tmp_path):
    """A dump whose columns are reordered and extended still reads right."""
    frames = [_frame((7, 5), (1, 2))]
    plain = _roundtrip(tmp_path, frames)
    odd = _roundtrip(tmp_path, frames,
                     columns=("id", "x", "vx", "y", "type", "z"))
    assert np.array_equal(odd[0].positions, plain[0].positions)
    assert np.array_equal(odd[0].types, plain[0].types)


def test_a_dump_missing_a_needed_column_says_which_it_has(tmp_path):
    frames = [_frame((4,), (1,))]
    path = _write_dump(tmp_path / "d.lammpstrj", frames,
                       columns=("id", "x", "y", "z"))
    with pytest.raises(ValueError, match="type"):
        list(read_dump(path))


def test_a_text_column_falls_back_and_reads_the_same_values(tmp_path):
    """The block parse cannot take a text column; the row parse can, and the
    two have to agree on every number."""
    frames = [_frame((7, 5), (1, 2))]
    numeric = _roundtrip(tmp_path, frames)
    with_text = _roundtrip(tmp_path, frames, element=True)
    assert np.array_equal(with_text[0].positions, numeric[0].positions)
    assert np.array_equal(with_text[0].types, numeric[0].types)


@pytest.mark.parametrize("element", [False, True])
def test_a_truncated_trailing_frame_is_dropped_not_raised(tmp_path, element):
    """A run killed at the wall clock leaves a partial last frame. Both parse
    paths have to survive it, and the complete frames before it stay."""
    frames = [_frame((7, 5), (1, 2), timestep=100 * i, seed=i)
              for i in range(3)]
    path = _write_dump(tmp_path / "d.lammpstrj", frames, element=element,
                       truncate_atoms=4)
    got = list(read_dump(path))
    assert [f.timestep for f in got] == [0, 100]


@pytest.mark.parametrize("element", [False, True])
def test_a_frame_cut_in_the_middle_of_a_row_is_dropped_too(tmp_path, element):
    """The row is present but short -- a dump caught mid-write. The block
    parse cannot see it, so the count check hands over to the row parse, which
    can."""
    frames = [_frame((7, 5), (1, 2), timestep=100 * i, seed=i)
              for i in range(3)]
    path = _write_dump(tmp_path / "d.lammpstrj", frames, element=element,
                       short_last_row=True)
    got = list(read_dump(path))
    assert [f.timestep for f in got] == [0, 100]


@pytest.mark.parametrize("element", [False, True])
def test_a_field_that_is_not_a_number_ends_the_iteration(tmp_path, element):
    """A row cut in the middle of an exponent has all its fields and one of
    them is not a number. Both parse paths hand back rather than raise, the
    same as for a short row: the implementations this replaces raise here,
    against the promise one of them makes in its own docstring."""
    frames = [_frame((7, 5), (1, 2), timestep=100 * i, seed=i)
              for i in range(3)]
    path = _write_dump(tmp_path / "d.lammpstrj", frames, element=element)
    Path(path).write_text(Path(path).read_text().rstrip("\n") + "e\n")
    assert [f.timestep for f in read_dump(path)] == [0, 100]


def test_a_frame_that_cannot_be_parsed_ends_the_iteration(tmp_path):
    """Not only a trailing one. A frame that will not parse ends the read even
    when complete frames follow it, because carrying on would leave a hole in
    the frame sequence, and every stage after this one takes the interval
    between consecutive frames to be the one constant it was handed."""
    frames = [_frame((7, 5), (1, 2), timestep=100 * i, seed=i)
              for i in range(4)]
    path = _write_dump(tmp_path / "d.lammpstrj", frames)
    lines = Path(path).read_text().splitlines()
    marks = [i for i, line in enumerate(lines)
             if line.startswith("ITEM: TIMESTEP")]
    row = marks[2] + 9 + 3
    lines[row] = " ".join(lines[row].split()[:-1])
    Path(path).write_text("\n".join(lines) + "\n")
    assert [f.timestep for f in read_dump(path)] == [0, 100]


def test_an_item_line_that_is_not_the_timestep_marker_starts_nothing(tmp_path):
    """A dump written with unit or time metadata carries other ``ITEM:`` lines
    of its own. Only the timestep marker begins a frame."""
    frames = [_frame((7, 5), (1, 2), timestep=100 * i, seed=i)
              for i in range(2)]
    path = _write_dump(tmp_path / "d.lammpstrj", frames)
    Path(path).write_text("ITEM: UNITS\nlj\n" + Path(path).read_text())
    got = list(read_dump(path))
    assert [f.timestep for f in got] == [0, 100]
    assert np.array_equal(got[0].positions, frames[0].positions)


def test_a_truncated_header_ends_the_iteration(tmp_path):
    frames = [_frame((4,), (1,), timestep=100 * i, seed=i) for i in range(3)]
    path = _write_dump(tmp_path / "d.lammpstrj", frames, truncate_header=True)
    assert [f.timestep for f in read_dump(path)] == [0, 100]


def test_a_triclinic_bounds_line_still_gives_three_by_two(tmp_path):
    """A tilted box writes a third number per bounds line. Only the pair is
    the axis' extent."""
    frames = [_frame((4,), (1,))]
    path = _write_dump(tmp_path / "d.lammpstrj", frames)
    text = Path(path).read_text().replace("ITEM: BOX BOUNDS pp pp pp",
                                          "ITEM: BOX BOUNDS xy xz yz pp pp pp")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("ITEM: BOX BOUNDS"):
            for j in range(1, 4):
                lines[i + j] = lines[i + j] + " 0.25"
    Path(path).write_text("\n".join(lines) + "\n")
    got = list(read_dump(path))
    assert got[0].box_bounds.shape == (3, 2)
    assert np.array_equal(got[0].box_bounds, frames[0].box_bounds)


def test_an_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "empty.lammpstrj"
    path.write_text("")
    assert list(read_dump(path)) == []


# ---------------------------------------------------------------------------
# grid_for_spacing
# ---------------------------------------------------------------------------

def test_the_grid_is_the_nearest_integer_resolution_per_axis():
    bounds = np.array([[0.0, 12.0], [0.0, 11.9], [1.0, 41.0]])
    assert grid_for_spacing(bounds, spacing=0.5, min_points=8) == (24, 24, 80)


def test_a_short_axis_is_held_at_the_floor():
    bounds = np.array([[0.0, 1.0], [0.0, 12.0], [0.0, 12.0]])
    assert grid_for_spacing(bounds, spacing=0.5, min_points=8) == (8, 24, 24)
    assert grid_for_spacing(bounds, spacing=0.5, min_points=2) == (2, 24, 24)


@pytest.mark.parametrize("kwargs, match", [
    ({"spacing": 0.0, "min_points": 8}, "spacing"),
    ({"spacing": -1.0, "min_points": 8}, "spacing"),
    ({"spacing": 0.5, "min_points": 0}, "min_points"),
])
def test_a_grid_request_that_cannot_be_honoured_is_refused(kwargs, match):
    bounds = np.array([[0.0, 12.0], [0.0, 12.0], [0.0, 12.0]])
    with pytest.raises(ValueError, match=match):
        grid_for_spacing(bounds, **kwargs)


def test_a_grid_from_a_broken_box_is_refused():
    with pytest.raises(ValueError, match=r"\(3, 2\)"):
        grid_for_spacing(np.zeros((2, 2)), spacing=0.5, min_points=8)
    with pytest.raises(ValueError, match="edge"):
        grid_for_spacing(np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
                         spacing=0.5, min_points=8)


# ---------------------------------------------------------------------------
# density_series
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", BOTH)
def test_a_series_is_frame_first_then_channel(spec):
    types, counts = spec
    frames = [_frame(counts, types, seed=i) for i in range(4)]
    series = density_series(frames, (8, 8, 8), sigma=1.0, atom_types=types,
                            method="mesh", device="cpu")
    assert series.shape == (4, len(types), 8, 8, 8)
    assert series.dtype == np.float32


@pytest.mark.parametrize("spec", BOTH)
def test_a_series_frame_equals_the_single_frame_kernel(spec):
    """The series is the kernel applied per frame and nothing else: no
    normalisation, no ratio, no per-frame rescaling."""
    types, counts = spec
    frames = [_frame(counts, types, seed=i) for i in range(3)]
    series = density_series(frames, (8, 8, 8), sigma=1.0, atom_types=types,
                            method="mesh", device="cpu")
    for i, frame in enumerate(frames):
        one = density_field(frame.positions, frame.types, frame.box_bounds,
                            (8, 8, 8), sigma=1.0, atom_types=types,
                            method="mesh", device="cpu")
        assert np.array_equal(series[i], one.numpy())


def test_each_frame_is_coarse_grained_in_its_own_box():
    """A barostat changes the box frame by frame. Using frame zero's box for
    the whole timeline would put the same atom at a different fraction of the
    cell, which is a drift the derivative would then report as dynamics."""
    pos, typ = _atoms((7, 5), (1, 2))
    wide = np.array([[0.0, 6.0], [0.0, 6.0], [0.0, 7.0]])
    narrow = np.array([[0.0, 6.0], [0.0, 6.0], [0.0, 6.0]])
    frames = [Frame(0, narrow, pos, typ), Frame(1, wide, pos, typ)]
    series = density_series(frames, (8, 8, 8), sigma=1.0, atom_types=(1, 2),
                            method="mesh", device="cpu")
    assert not np.array_equal(series[0], series[1])
    alone = density_field(pos, typ, wide, (8, 8, 8), sigma=1.0,
                          atom_types=(1, 2), method="mesh", device="cpu")
    assert np.array_equal(series[1], alone.numpy())


def test_the_grid_does_not_follow_the_box():
    """The index grid is fixed for the timeline even though the box is not."""
    pos, typ = _atoms((7, 5), (1, 2))
    frames = [Frame(i, np.array([[0.0, 6.0], [0.0, 6.0], [0.0, 6.0 + i]]),
                    pos, typ) for i in range(3)]
    series = density_series(frames, (8, 8, 8), sigma=1.0, atom_types=(1, 2),
                            method="mesh", device="cpu")
    assert series.shape == (3, 2, 8, 8, 8)


def test_an_empty_timeline_is_refused():
    with pytest.raises(ValueError, match="empty"):
        density_series([], (8, 8, 8), sigma=1.0, atom_types=(1,),
                       method="mesh", device="cpu")


def test_the_series_passes_the_method_and_its_cutoff_through():
    frames = [_frame((7,), (1,))]
    direct = density_series(frames, (6, 6, 6), sigma=1.0, atom_types=(1,),
                            method="direct", device="cpu", cutoff_in_sigma=5.0)
    mesh = density_series(frames, (6, 6, 6), sigma=1.0, atom_types=(1,),
                          method="mesh", device="cpu")
    assert direct.shape == mesh.shape
    assert not np.array_equal(direct, mesh)
    with pytest.raises(ValueError, match="cutoff_in_sigma"):
        density_series(frames, (6, 6, 6), sigma=1.0, atom_types=(1,),
                       method="direct", device="cpu")


def test_the_series_honours_the_working_precision():
    frames = [_frame((7,), (1,))]
    wide = density_series(frames, (6, 6, 6), sigma=1.0, atom_types=(1,),
                          method="mesh", device="cpu", dtype=torch.float64)
    assert wide.dtype == np.float64


# ---------------------------------------------------------------------------
# smooth_in_time
# ---------------------------------------------------------------------------

def _polynomial_timeline(n_frames, dt, coefficients, shape=(2, 3)):
    """A timeline whose every element is the same polynomial in time, scaled."""
    t = np.arange(n_frames) * dt
    curve = sum(c * t ** p for p, c in enumerate(coefficients))
    scale = np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(shape)
    return curve[:, None, None] * scale[None]


@pytest.mark.parametrize("edge_mode, edge", [("interp", 0), ("nearest", 5)])
def test_a_polynomial_is_smoothed_to_itself_and_differentiated_exactly(
        edge_mode, edge):
    """A local cubic fit reproduces a cubic, so the filter is the identity on
    one and its derivative is the analytic one. That pins the window, the
    polynomial order and the frame spacing all at once.

    The two edge treatments differ in reach: fitting the end window is exact
    on a polynomial everywhere, while repeating the end sample is exact only
    where the window has not run off, so that case is checked inside.
    """
    dt, coefficients = 0.02, (0.5, -1.25, 3.0, 0.75)
    values = _polynomial_timeline(41, dt, coefficients)
    smoothed, derivative = smooth_in_time(values, window=11, polyorder=3,
                                          dt=dt, edge_mode=edge_mode)
    t = np.arange(41) * dt
    slope = sum(p * c * t ** (p - 1) for p, c in enumerate(coefficients) if p)
    scale = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    inside = slice(edge, 41 - edge) if edge else slice(None)
    assert np.allclose(smoothed[inside], values[inside], atol=1e-10)
    assert np.allclose(derivative[inside],
                       (slope[:, None, None] * scale[None])[inside],
                       atol=1e-9)


def test_the_frame_spacing_only_scales_the_derivative():
    values = _polynomial_timeline(31, 1.0, (0.0, 1.0, 0.5))
    _, one = smooth_in_time(values, window=9, polyorder=2, dt=1.0,
                            edge_mode="interp")
    _, tenth = smooth_in_time(values, window=9, polyorder=2, dt=0.1,
                              edge_mode="interp")
    assert np.allclose(tenth, 10.0 * one)


def test_the_filter_runs_along_frames_and_nothing_else():
    """Every other axis is carried through independently: a timeline of one
    voxel filtered alone equals that voxel of the whole field filtered."""
    rng = np.random.default_rng(3)
    values = rng.normal(size=(25, 2, 4, 4, 4))
    whole, whole_d = smooth_in_time(values, window=7, polyorder=2, dt=0.5,
                                    edge_mode="interp")
    one, one_d = smooth_in_time(values[:, 1, 2, 3, 0], window=7, polyorder=2,
                                dt=0.5, edge_mode="interp")
    assert np.array_equal(whole[:, 1, 2, 3, 0], one)
    assert np.array_equal(whole_d[:, 1, 2, 3, 0], one_d)


def test_the_two_edge_modes_agree_inside_and_differ_at_the_ends():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(41, 3)) + np.arange(41)[:, None]
    interp, _ = smooth_in_time(values, window=11, polyorder=3, dt=1.0,
                               edge_mode="interp")
    nearest, _ = smooth_in_time(values, window=11, polyorder=3, dt=1.0,
                                edge_mode="nearest")
    assert np.allclose(interp[5:-5], nearest[5:-5])
    assert not np.allclose(interp[:5], nearest[:5])
    assert not np.allclose(interp[-5:], nearest[-5:])


def test_a_window_out_of_the_interior_gives_the_same_bits_as_the_whole():
    """The measured property the archive check relies on: away from the ends
    the filter is a plain convolution, so one sample can be filtered from its
    own window alone."""
    rng = np.random.default_rng(11)
    values = rng.normal(size=(81, 2, 3)).astype(np.float32)
    whole, whole_d = smooth_in_time(values, window=31, polyorder=3, dt=0.02,
                                    edge_mode="interp")
    centre = 40
    window = values[centre - 15: centre + 16]
    part, part_d = smooth_in_time(window, window=31, polyorder=3, dt=0.02,
                                  edge_mode="interp")
    assert np.array_equal(part[15], whole[centre])
    assert np.array_equal(part_d[15], whole_d[centre])


def test_the_filter_keeps_the_timeline_precision():
    values = _polynomial_timeline(21, 0.1, (1.0, 2.0)).astype(np.float32)
    smoothed, derivative = smooth_in_time(values, window=7, polyorder=2,
                                          dt=0.1, edge_mode="interp")
    assert smoothed.dtype == np.float32
    assert derivative.dtype == np.float32


def test_an_unknown_edge_mode_names_the_known_ones():
    values = _polynomial_timeline(21, 0.1, (1.0,))
    with pytest.raises(ValueError, match="mirror"):
        smooth_in_time(values, window=7, polyorder=2, dt=0.1,
                       edge_mode="mirror")
    try:
        smooth_in_time(values, window=7, polyorder=2, dt=0.1,
                       edge_mode="mirror")
    except ValueError as exc:
        for name in EDGE_MODES:
            assert name in str(exc)


@pytest.mark.parametrize("kwargs, match", [
    ({"window": 10, "polyorder": 3, "dt": 0.1}, "even"),
    ({"window": 41, "polyorder": 3, "dt": 0.1}, "fit"),
    # The message, not just the type: the filtering library refuses the same
    # two cases with a message of its own, so matching "polyorder" alone
    # passes whether this module checked or not.
    ({"window": 7, "polyorder": 7, "dt": 0.1}, "the fit would be exact"),
    ({"window": 7, "polyorder": 9, "dt": 0.1}, "the fit would be exact"),
    ({"window": 7, "polyorder": 2, "dt": 0.0}, "dt"),
    ({"window": 7, "polyorder": 2, "dt": -0.1}, "dt"),
])
def test_a_filter_that_cannot_be_honoured_is_refused(kwargs, match):
    """A window longer than the timeline is refused, not silently narrowed."""
    values = _polynomial_timeline(21, 0.1, (1.0,))
    with pytest.raises(ValueError, match=match):
        smooth_in_time(values, edge_mode="interp", **kwargs)


def test_an_empty_timeline_cannot_be_filtered():
    with pytest.raises(ValueError, match="frame axis"):
        smooth_in_time(np.zeros((0, 3)), window=3, polyorder=1, dt=0.1,
                       edge_mode="interp")


# ---------------------------------------------------------------------------
# emit_indices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spacing", SPACINGS)
def test_emitted_frames_lie_in_the_tail_and_increase(spacing):
    idx = emit_indices(2001, n_melt=25, emit_n=400, spacing=spacing)
    assert idx.ndim == 1
    assert idx.dtype.kind == "i"
    assert idx.min() >= 25
    assert idx.max() <= 2000
    assert np.all(np.diff(idx) > 0)
    assert len(idx) <= 400


def test_log_spacing_is_dense_at_the_start_of_the_tail():
    idx = emit_indices(2001, n_melt=25, emit_n=400, spacing="log")
    assert idx[0] == 25
    assert idx[1] - idx[0] < idx[-1] - idx[-2]


def test_uniform_spacing_is_evenly_spread():
    idx = emit_indices(1000, n_melt=0, emit_n=10, spacing="uniform")
    assert len(idx) == 10
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1


@pytest.mark.parametrize("spacing", SPACINGS)
def test_both_spacings_reach_the_last_frame(spacing):
    """The tail's last frame is emitted, so a run's end state is in the set."""
    idx = emit_indices(500, n_melt=10, emit_n=50, spacing=spacing)
    assert idx[-1] == 499


@pytest.mark.parametrize("spacing", SPACINGS)
def test_asking_for_more_than_the_tail_holds_returns_all_of_it(spacing):
    idx = emit_indices(30, n_melt=25, emit_n=400, spacing=spacing)
    assert np.array_equal(idx, np.arange(25, 30))


@pytest.mark.parametrize("spacing", SPACINGS)
def test_asking_for_exactly_the_tail_returns_all_of_it(spacing):
    """The boundary. A request equal to the tail length is the whole tail and
    not a spread that happens to want the same count: rounding a spread onto
    that many integers collapses some of them and would emit fewer frames
    than a request one larger."""
    idx = emit_indices(30, n_melt=25, emit_n=5, spacing=spacing)
    assert np.array_equal(idx, np.arange(25, 30))


def test_discarding_the_whole_timeline_emits_nothing():
    assert emit_indices(30, n_melt=30, emit_n=4, spacing="log").size == 0


@pytest.mark.parametrize("kwargs, match", [
    ({"n_melt": -1, "emit_n": 4, "spacing": "log"}, "n_melt"),
    ({"n_melt": 31, "emit_n": 4, "spacing": "log"}, "n_melt"),
    ({"n_melt": 5, "emit_n": 0, "spacing": "log"}, "emit_n"),
    ({"n_melt": 5, "emit_n": -3, "spacing": "log"}, "emit_n"),
    ({"n_melt": 5, "emit_n": 4, "spacing": "geometric"}, "spacing"),
])
def test_an_impossible_emission_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        emit_indices(30, **kwargs)


def test_a_negative_timeline_length_is_refused():
    with pytest.raises(ValueError, match="n_frames"):
        emit_indices(-1, n_melt=0, emit_n=4, spacing="log")


def test_an_unknown_spacing_names_the_known_ones():
    try:
        emit_indices(30, n_melt=5, emit_n=4, spacing="geometric")
    except ValueError as exc:
        for name in SPACINGS:
            assert name in str(exc)


# ---------------------------------------------------------------------------
# composition_field
# ---------------------------------------------------------------------------

def _field(n_species, seed=1, shape=(4, 4, 4)):
    rng = np.random.default_rng(seed)
    return rng.random((n_species, *shape)).astype(np.float32)


@pytest.mark.parametrize("spec", BOTH)
def test_the_offset_form_is_the_share_of_the_total(spec):
    types, _ = spec
    field = _field(len(types))
    got = composition_field(field, numerator=0, form="offset",
                            regulariser=1e-6)
    want = field[0] / (field.sum(axis=0) + 1e-6)
    assert np.array_equal(got, want)
    assert got.shape == field.shape[1:]


@pytest.mark.parametrize("spec", BOTH)
def test_the_floored_form_divides_by_the_floored_total(spec):
    types, _ = spec
    field = _field(len(types))
    got = composition_field(field, numerator=0, form="floored",
                            regulariser=1e-12)
    assert np.array_equal(got, field[0] / np.maximum(field.sum(axis=0), 1e-12))


@pytest.mark.parametrize("spec", BOTH)
def test_the_filled_form_reports_the_fill_where_there_is_nothing(spec):
    types, _ = spec
    field = _field(len(types))
    field[:, 0, 0, 0] = 0.0
    got = composition_field(field, numerator=0, form="filled",
                            regulariser=1e-8, fill=0.5)
    assert got[0, 0, 0] == pytest.approx(0.5)
    assert got[1, 1, 1] == pytest.approx(
        field[0, 1, 1, 1] / field[:, 1, 1, 1].sum(), rel=1e-6)


def test_the_three_forms_agree_on_a_full_voxel_and_differ_on_an_empty_one():
    """The regularisation is invisible where there is density and is the whole
    answer where there is none, which is why it cannot be a hidden constant."""
    field = np.zeros((2, 2, 1, 1), dtype=np.float64)
    field[:, 0] = [[[1.0]], [[3.0]]]
    forms = {
        "offset": composition_field(field, numerator=0, form="offset",
                                    regulariser=1e-6),
        "floored": composition_field(field, numerator=0, form="floored",
                                     regulariser=1e-6),
        "filled": composition_field(field, numerator=0, form="filled",
                                    regulariser=1e-6, fill=0.5),
    }
    for name, value in forms.items():
        assert value[0, 0, 0] == pytest.approx(0.25, rel=1e-5), name
    assert forms["offset"][1, 0, 0] == 0.0
    assert forms["floored"][1, 0, 0] == 0.0
    assert forms["filled"][1, 0, 0] == 0.5


def test_a_negative_ring_separates_the_floored_form_from_the_filled_one():
    """A coarse-grained field rings slightly negative beside a sharp edge, and
    there the floored form divides a non-zero numerator by the floor while the
    filled one reports its declared value. Measured, not asserted in the
    abstract."""
    field = np.zeros((2, 1, 1, 1), dtype=np.float64)
    field[0, 0, 0, 0] = -1e-9
    floored = composition_field(field, numerator=0, form="floored",
                                regulariser=1e-12)
    filled = composition_field(field, numerator=0, form="filled",
                               regulariser=1e-12, fill=0.5)
    assert floored[0, 0, 0] == pytest.approx(-1000.0)
    assert filled[0, 0, 0] == 0.5


@pytest.mark.parametrize("spec", BOTH)
def test_the_composition_works_on_a_timeline_as_well_as_a_frame(spec):
    types, counts = spec
    frames = [_frame(counts, types, seed=i) for i in range(3)]
    series = density_series(frames, (6, 6, 6), sigma=1.0, atom_types=types,
                            method="mesh", device="cpu")
    whole = composition_field(series, numerator=0, form="offset",
                              regulariser=1e-6)
    assert whole.shape == (3, 6, 6, 6)
    for i in range(3):
        assert np.array_equal(
            whole[i], composition_field(series[i], numerator=0, form="offset",
                                        regulariser=1e-6))


@pytest.mark.parametrize("form, extra", [("offset", {}), ("floored", {}),
                                         ("filled", {"fill": 0.5})])
def test_the_composition_gives_the_same_numbers_on_either_array_type(form,
                                                                     extra):
    field = _field(2)
    numpy_value = composition_field(field, numerator=0, form=form,
                                    regulariser=1e-6, **extra)
    torch_value = composition_field(torch.from_numpy(field), numerator=0,
                                    form=form, regulariser=1e-6, **extra)
    assert torch_value.dtype == torch.float32
    assert np.array_equal(torch_value.numpy(), numpy_value)


def test_the_numerator_names_the_channel():
    field = _field(2)
    first = composition_field(field, numerator=0, form="offset",
                              regulariser=1e-6)
    second = composition_field(field, numerator=1, form="offset",
                               regulariser=1e-6)
    assert not np.array_equal(first, second)
    assert np.allclose(first + second, 1.0, atol=1e-5)
    assert np.array_equal(
        second, composition_field(field, numerator=-1, form="offset",
                                  regulariser=1e-6))


@pytest.mark.parametrize("kwargs, match", [
    ({"numerator": 0, "form": "share", "regulariser": 1e-6}, "form"),
    ({"numerator": 0, "form": "offset", "regulariser": 0.0}, "regulariser"),
    ({"numerator": 0, "form": "offset", "regulariser": -1e-6}, "regulariser"),
    ({"numerator": 0, "form": "filled", "regulariser": 1e-6}, "fill"),
    ({"numerator": 0, "form": "offset", "regulariser": 1e-6, "fill": 0.5},
     "fill"),
    ({"numerator": 0, "form": "floored", "regulariser": 1e-6, "fill": 0.5},
     "fill"),
    ({"numerator": 2, "form": "offset", "regulariser": 1e-6}, "numerator"),
    ({"numerator": -3, "form": "offset", "regulariser": 1e-6}, "numerator"),
])
def test_a_composition_that_cannot_be_honoured_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        composition_field(_field(2), **kwargs)


def test_the_numerator_is_checked_against_the_channel_axis_of_a_timeline():
    """On a timeline the leading axis is frames, so a bound taken from it
    would admit a numerator that is not a channel at all -- and admit it
    exactly when there are more frames than channels, which is always."""
    timeline = _field(2)[None].repeat(3, axis=0)
    assert timeline.shape == (3, 2, 4, 4, 4)
    with pytest.raises(ValueError, match="numerator"):
        composition_field(timeline, numerator=2, form="offset",
                          regulariser=1e-6)


def test_a_field_without_a_channel_axis_is_refused():
    with pytest.raises(ValueError, match="channel axis"):
        composition_field(np.zeros((4, 4, 4)), numerator=0, form="offset",
                          regulariser=1e-6)


def test_an_unknown_composition_form_names_the_known_ones():
    try:
        composition_field(_field(2), numerator=0, form="share",
                          regulariser=1e-6)
    except ValueError as exc:
        for name in COMPOSITION_FORMS:
            assert name in str(exc)


# ---------------------------------------------------------------------------
# box_lengths
# ---------------------------------------------------------------------------

def test_the_box_lengths_are_per_frame():
    frames = [Frame(i, np.array([[0.0, 6.0], [1.0, 7.5], [0.0, 6.0 + i]]),
                    np.zeros((1, 3)), np.ones(1, dtype=np.int64))
              for i in range(3)]
    assert np.array_equal(
        box_lengths(frames),
        np.array([[6.0, 6.5, 6.0], [6.0, 6.5, 7.0], [6.0, 6.5, 8.0]]))


def test_an_empty_timeline_has_no_boxes():
    with pytest.raises(ValueError, match="empty"):
        box_lengths([])


# ---------------------------------------------------------------------------
# against the archive (env: the H/He raw root)
# ---------------------------------------------------------------------------

_ARCHIVED_TRAJECTORY = ("slab_data_800GPa", "slab_xl0.30_xr0.70_T07000", "traj.lammpstrj")
_ARCHIVED_FIELDS = ("fields", "_pilot", "sigma_2.0", "cg.npz")

#: The settings the archived fields above were produced with, read out of the
#: run's own stdout log beside them.
_ARCHIVED = dict(grid=(60, 60, 200), sigma=2.0, dt=0.02, n_melt=25,
                 emit_n=400, spacing="log", window=31, polyorder=3,
                 atom_types=(1, 2))


def _first_archived_frame(path, member):
    """Frame zero of one member of a stored archive, without the rest of it.

    The two field members are 1.4 gigabytes each and the check below wants
    one frame of each, so they are decompressed only as far as that frame.
    """
    import zipfile
    from numpy.lib import format as npy

    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        version = npy.read_magic(handle)
        shape, fortran, dtype = (npy.read_array_header_1_0(handle)
                                 if version == (1, 0)
                                 else npy.read_array_header_2_0(handle))
        assert not fortran, member
        count = int(np.prod(shape[1:]))
        return np.frombuffer(handle.read(count * dtype.itemsize),
                             dtype=dtype).reshape(shape[1:])


def _needs(parts):
    return str(declared_roots.raw_or_skip("hhe", *parts))


@pytest.mark.env
def test_the_emitted_frames_reproduce_the_archived_ones_exactly():
    """The emission rule is bit-for-bit against every one of the archived
    run's 243 emitted timesteps, at the archived dump interval."""
    stored = np.load(_needs(_ARCHIVED_FIELDS))
    idx = emit_indices(2001, n_melt=int(stored["n_melt"]),
                       emit_n=_ARCHIVED["emit_n"],
                       spacing=str(stored["emit_spacing"]))
    assert len(idx) == len(stored["timesteps"])
    assert np.array_equal(idx * 200, stored["timesteps"])


@pytest.mark.env
def test_the_archived_fields_are_reproduced_to_the_precision_they_were_made_in():
    """The pipeline, end to end, against the archived fields.

    Everything that is not floating-point arithmetic reproduces exactly: the
    emitted frames, the timestep of each, and the box of each. The fields
    themselves do not, and the reason is measured rather than assumed. The
    archived run coarse-grained in single precision on an accelerator; the
    mesh method's transform pair at that precision is 5.6 units in the last
    place away from the same arithmetic in double, and no two accelerators or
    library versions round it the same way.

    Measured deviations from the archive, on the first emitted frame:
    accelerator 1 unit in the last place on the field and 5.5 on its
    derivative; processor 2 and 14.5. The bounds below are four and thirty-two,
    so a real regression -- a shifted window, a dropped frame, the wrong box --
    moves far outside them while round-off does not.
    """
    trajectory = _needs(_ARCHIVED_TRAJECTORY)
    fields = _needs(_ARCHIVED_FIELDS)
    stored = np.load(fields)
    idx = emit_indices(2001, n_melt=_ARCHIVED["n_melt"],
                       emit_n=_ARCHIVED["emit_n"], spacing=_ARCHIVED["spacing"])
    half = _ARCHIVED["window"] // 2
    centre = int(idx[0])
    assert centre - half >= 0, "the first emitted frame has to be interior"

    frames = []
    for i, frame in enumerate(read_dump(trajectory)):
        if centre - half <= i <= centre + half:
            frames.append(frame)
        if i >= centre + half:
            break

    lengths = box_lengths(frames)
    assert np.array_equal(lengths[half], stored["box"][0])

    series = density_series(frames, _ARCHIVED["grid"],
                            sigma=_ARCHIVED["sigma"],
                            atom_types=_ARCHIVED["atom_types"],
                            method="mesh", device="auto")
    smoothed, derivative = smooth_in_time(
        series, window=_ARCHIVED["window"], polyorder=_ARCHIVED["polyorder"],
        dt=_ARCHIVED["dt"], edge_mode="interp")

    # The archive is channel last; one exact transpose at the file boundary.
    want_field = np.moveaxis(
        _first_archived_frame(fields, "rho.npy"), -1, 0)
    want_derivative = np.moveaxis(
        _first_archived_frame(fields, "drho_dt.npy"), -1, 0)
    assert smoothed[half].shape == want_field.shape

    for name, got, want, budget in (
            ("field", smoothed[half], want_field, 4),
            ("derivative", derivative[half], want_derivative, 32)):
        peak = float(np.abs(want).max())
        deviation = float(np.abs(got.astype(np.float64)
                                 - want.astype(np.float64)).max())
        allowed = budget * float(np.spacing(np.float32(peak)))
        assert deviation <= allowed, (
            f"{name}: {deviation:.3e} exceeds {budget} units in the last "
            f"place of {peak:.4f} ({allowed:.3e})")
