"""Tests for the position columns of aipf.pipeline.coarse_grain's dump readers.

``x y z`` is read as it always was. ``xu yu zu`` (unwrapped) is read as is, and ``xs ys zs`` /
``xsu ysu zsu`` (scaled) are converted with the frame's bounds. The properties are checked against
a second reading of the same frames: the same numbers under another name read bit for bit alike,
and atoms moved by whole box vectors give the wrapped frame's amplitudes and fields to rounding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aipf.pipeline import coarse_grain
from aipf.pipeline.coarse_grain import POSITION_COLUMNS, read_dump, read_dump_chain
from aipf.pipeline.extract_modes import mode_amplitudes, mode_set
from aipf.pipeline.kde import density_field
from aipf.pipeline.modes import modes_from_dump

PARAMS = dict(sigma=1.0, k_cut=2.0, atom_types=(1, 2), fields="per_type",
              ordering="lexicographic", route="dense", T=1.0, dt_frame=0.02,
              skip_frames=0)

N_ATOMS = 24


def _box(step):
    """A box that changes from frame to frame and does not start at the origin."""
    return np.array([[-1.5, 8.5 + 0.01 * step], [0.25, 9.75], [2.0, 11.0 - 0.02 * step]])


def _atoms(step):
    rng = np.random.default_rng(step)
    box = _box(step)
    pos = box[:, 0] + (box[:, 1] - box[:, 0]) * rng.random((N_ATOMS, 3))
    types = np.where(np.arange(N_ATOMS) % 3, 1, 2)
    return box, pos, types


def _write(path, steps, *, names=("x", "y", "z"), values=None, extra=None, triclinic=False):
    """A dump of :func:`_atoms`' frames with the positions written under ``names``.

    ``values(box, pos, rng)`` gives the numbers written there (default the positions). ``extra``
    adds a second triple ``(names, values)``.
    """
    lines = []
    for step in steps:
        box, pos, types = _atoms(step)
        rng = np.random.default_rng(1000 + step)
        written = pos if values is None else values(box, pos, rng)
        columns = ["id", "type", *names]
        blocks = [written]
        if extra is not None:
            columns += list(extra[0])
            blocks.append(extra[1](box, pos, rng))
        lines += ["ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", str(N_ATOMS),
                  "ITEM: BOX BOUNDS xy xz yz pp pp pp" if triclinic else "ITEM: BOX BOUNDS pp pp pp"]
        lines += [f"{float(lo)!r} {float(hi)!r}" + (" 0.0" if triclinic else "") for lo, hi in box]
        lines.append("ITEM: ATOMS " + " ".join(columns))
        order = np.random.default_rng(step + 7).permutation(N_ATOMS)
        for i in order:
            row = [float(i + 1), float(types[i])]
            for block in blocks:
                row += [float(v) for v in block[i]]
            lines.append(" ".join(repr(v) for v in row))
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def _shifted(box, pos, rng):
    """The atoms moved by whole box vectors, up to five boxes each way: true unwrapped positions."""
    return pos + rng.integers(-5, 6, size=pos.shape) * (box[:, 1] - box[:, 0])


def _scaled(box, pos, rng):
    return (pos - box[:, 0]) / (box[:, 1] - box[:, 0])


def _frames_equal(one, other):
    assert len(one) == len(other)
    for a, b in zip(one, other):
        assert a.timestep == b.timestep
        assert np.array_equal(a.box_bounds, b.box_bounds)
        assert np.array_equal(a.positions, b.positions)
        assert np.array_equal(a.types, b.types)


STEPS = [0, 100, 200, 300]


# ---------------------------------------------------------------------------
# the same numbers under another name
# ---------------------------------------------------------------------------

def test_unwrapped_columns_holding_the_same_numbers_read_bit_for_bit_alike(tmp_path):
    plain = _write(tmp_path / "x.dump", STEPS)
    unwrapped = _write(tmp_path / "xu.dump", STEPS, names=("xu", "yu", "zu"))
    _frames_equal(list(read_dump(unwrapped)), list(read_dump(plain)))
    one = modes_from_dump(plain, **PARAMS)
    other = modes_from_dump(unwrapped, **PARAMS)
    assert np.array_equal(one.amplitudes, other.amplitudes)
    assert np.array_equal(one.labels, other.labels)
    assert np.array_equal(one.boxes, other.boxes)


def test_a_two_file_chain_of_unwrapped_columns_reads_as_the_plain_chain(tmp_path):
    plain = [_write(tmp_path / "x0.dump", [0, 100, 200]), _write(tmp_path / "x1.dump", [200, 300])]
    unwrapped = [_write(tmp_path / "u0.dump", [0, 100, 200], names=("xu", "yu", "zu")),
                 _write(tmp_path / "u1.dump", [200, 300], names=("xu", "yu", "zu"))]
    plain_report, unwrapped_report = [], []
    _frames_equal(list(read_dump_chain(unwrapped, report=unwrapped_report)),
                  list(read_dump_chain(plain, report=plain_report)))
    strip = lambda report: [{k: v for k, v in e.items() if k != "source"} for e in report]
    assert strip(unwrapped_report) == strip(plain_report)
    one = modes_from_dump(plain, **PARAMS)
    other = modes_from_dump(unwrapped, **PARAMS)
    assert np.array_equal(one.amplitudes, other.amplitudes)


def test_plain_columns_win_over_unwrapped_ones_in_the_same_dump(tmp_path):
    plain = _write(tmp_path / "x.dump", STEPS)
    both = _write(tmp_path / "both.dump", STEPS, extra=(("xu", "yu", "zu"), _shifted))
    _frames_equal(list(read_dump(both)), list(read_dump(plain)))
    report = []
    _frames_equal(list(read_dump_chain([both], report=report)), list(read_dump(plain)))


def test_unwrapped_columns_win_over_scaled_ones(tmp_path):
    unwrapped = _write(tmp_path / "xu.dump", STEPS, names=("xu", "yu", "zu"))
    both = _write(tmp_path / "both.dump", STEPS, names=("xs", "ys", "zs"), values=_scaled,
                  extra=(("xu", "yu", "zu"), lambda box, pos, rng: pos))
    _frames_equal(list(read_dump(both)), list(read_dump(unwrapped)))


# ---------------------------------------------------------------------------
# true unwrapped and scaled positions
# ---------------------------------------------------------------------------

def test_atoms_moved_by_box_vectors_give_the_wrapped_amplitudes_to_rounding(tmp_path):
    """Mode amplitudes are complex128; the phase of an atom five boxes out is formed from an ``s``
    about six times larger, so agreement is to rounding (1e-12 relative to the largest amplitude)."""
    plain = _write(tmp_path / "x.dump", STEPS)
    unwrapped = _write(tmp_path / "xu.dump", STEPS, names=("xu", "yu", "zu"), values=_shifted)
    moved = list(read_dump(unwrapped))
    assert max(np.abs(f.positions).max() for f in moved) > 30.0
    for wrapped, far in zip(read_dump(plain), moved):
        nvec = mode_set(wrapped.box_bounds[:, 1] - wrapped.box_bounds[:, 0], k_cut=2.0,
                        ordering="lexicographic")
        a = mode_amplitudes(wrapped.positions, wrapped.types, wrapped.box_bounds, nvec,
                            atom_types=(1, 2), route="separable")
        b = mode_amplitudes(far.positions, far.types, far.box_bounds, nvec,
                            atom_types=(1, 2), route="separable")
        assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(a))
    one = modes_from_dump(plain, **PARAMS)
    other = modes_from_dump(unwrapped, **PARAMS)
    assert np.max(np.abs(one.amplitudes - other.amplitudes)) \
        <= 1e-12 * np.max(np.abs(one.amplitudes))


@pytest.mark.parametrize("method", ["mesh", "direct"])
def test_atoms_moved_by_box_vectors_give_the_wrapped_fields_in_float64(tmp_path, method):
    plain = _write(tmp_path / "x.dump", STEPS)
    unwrapped = _write(tmp_path / "xu.dump", STEPS, names=("xu", "yu", "zu"), values=_shifted)
    for wrapped, far in zip(read_dump(plain), read_dump(unwrapped)):
        fields = [density_field(f.positions, f.types, f.box_bounds, (8, 8, 8), sigma=1.0,
                                atom_types=(1, 2), method=method, device="cpu",
                                dtype=torch.float64,
                                cutoff_in_sigma=4.0 if method == "direct" else None)
                  for f in (wrapped, far)]
        scale = float(fields[0].abs().max())
        assert float((fields[0] - fields[1]).abs().max()) <= 1e-12 * scale


def test_the_float32_mesh_error_grows_with_the_unwrap_distance_within_its_stated_bound(tmp_path):
    """Documented, not acceptance: in float32 the mesh KDE's error on an atom ``|r - lo|`` from the
    origin is a few ``2^-24 |r - lo|`` in position, so the field error stays within a small multiple
    of the float32 error of the wrapped frame times ``m + 1`` for ``m`` boxes of unwrapping."""
    wrapped = next(read_dump(_write(tmp_path / "x.dump", [0])))
    far = next(read_dump(_write(tmp_path / "xu.dump", [0], names=("xu", "yu", "zu"),
                                values=_shifted)))

    def field(frame, dtype):
        return density_field(frame.positions, frame.types, frame.box_bounds, (8, 8, 8),
                             sigma=1.0, atom_types=(1, 2), method="mesh", device="cpu",
                             dtype=dtype).double()

    exact = field(wrapped, torch.float64)
    scale = float(exact.abs().max())
    wrapped_error = float((field(wrapped, torch.float32) - exact).abs().max()) / scale
    far_error = float((field(far, torch.float32) - exact).abs().max()) / scale
    assert wrapped_error < 1e-5
    assert far_error <= 4 * (5 + 1) * max(wrapped_error, 2.0 ** -24)


@pytest.mark.parametrize("names", POSITION_COLUMNS[2:], ids=lambda t: t[0])
def test_scaled_columns_give_the_cartesian_positions_to_rounding(tmp_path, names):
    plain = _write(tmp_path / "x.dump", STEPS)
    scaled = _write(tmp_path / "xs.dump", STEPS, names=names, values=_scaled)
    for a, b in zip(read_dump(plain), read_dump(scaled)):
        length = a.box_bounds[:, 1] - a.box_bounds[:, 0]
        assert np.array_equal(a.box_bounds, b.box_bounds)
        assert np.array_equal(a.types, b.types)
        assert np.all(np.abs(a.positions - b.positions) <= 1e-14 * np.abs(a.box_bounds).max()
                      + 4e-16 * length * 8)


def test_scaled_columns_through_the_chain_reader(tmp_path):
    plain = _write(tmp_path / "x.dump", STEPS)
    scaled = _write(tmp_path / "xs.dump", STEPS, names=("xs", "ys", "zs"), values=_scaled)
    chained = list(read_dump_chain([scaled], report=[]))
    for a, b in zip(read_dump(plain), chained):
        assert np.allclose(a.positions, b.positions, rtol=0.0, atol=1e-13)


def test_scaled_columns_in_a_triclinic_box_are_refused(tmp_path):
    path = _write(tmp_path / "tri.dump", [0], names=("xs", "ys", "zs"), values=_scaled,
                  triclinic=True)
    with pytest.raises(ValueError, match="triclinic"):
        list(read_dump(path))
    with pytest.raises(ValueError, match="triclinic"):
        list(read_dump_chain([path], report=[]))


def test_cartesian_columns_in_a_triclinic_box_read_as_before(tmp_path):
    """Triclinic boxes are out of scope: a Cartesian triple there is read exactly as before."""
    path = _write(tmp_path / "tri.dump", [0], triclinic=True)
    frame = next(read_dump(path))
    assert np.array_equal(frame.positions, next(read_dump(_write(tmp_path / "x.dump", [0]))).positions)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("names", [("x", "yu", "z"), ("xu", "yu", "z"), ("xs", "ys", "zsu")],
                         ids=["x-yu-z", "xu-yu-z", "xs-ys-zsu"])
def test_a_mixed_triple_is_refused_by_name(tmp_path, names):
    path = _write(tmp_path / "mixed.dump", [0], names=names)
    with pytest.raises(ValueError, match="incomplete position triple") as caught:
        list(read_dump(path))
    for triple in POSITION_COLUMNS:
        assert " ".join(triple) in str(caught.value)


def test_a_dump_without_positions_lists_the_accepted_triples(tmp_path):
    path = _write(tmp_path / "none.dump", [0], names=("vx", "vy", "vz"))
    with pytest.raises(ValueError, match="position triple") as caught:
        list(read_dump(path))
    for triple in POSITION_COLUMNS:
        assert " ".join(triple) in str(caught.value)


def test_the_preference_order_is_the_documented_one():
    assert POSITION_COLUMNS == (("x", "y", "z"), ("xu", "yu", "zu"), ("xs", "ys", "zs"),
                                ("xsu", "ysu", "zsu"))
    assert coarse_grain._position_triple(["id", "type", "x", "y", "z", "xu"]) == ("x", "y", "z")
