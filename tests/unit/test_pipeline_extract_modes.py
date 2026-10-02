"""Tests for aipf.pipeline.extract_modes: a trajectory to Fourier amplitudes.

The object this module computes is exact -- a finite sum of unit-modulus
terms, with no grid, no kernel and no truncation -- so almost everything here
is a property checked against a second, independent construction rather than
a pinned literal: a brute enumeration of the integer triples inside the
cutoff, a hand-written sum over atoms, an analytically known single-atom
amplitude, a permutation that has to be invisible.

``n_species`` is exercised at 1 and at 2 throughout, per the plan's global
constraint. It matters here because the stored channel set is the CALLER's
declaration and need not be the system's density channels: one archived
artefact stores an amplitude for a dump type that carries no density channel
at all, and the one-channel case is the one where that distinction shows.

The archived artefacts themselves are rebuilt by ``aipf modes`` in the
system tests (``test_hhe_modes_roundtrip.py``, ``test_lj_modes_roundtrip.py``).
"""
from __future__ import annotations


import numpy as np
import pytest


from aipf.pipeline.extract_modes import (
    ORDERINGS,
    REFERENCE_BOXES,
    ROUTES,
    frames_to_skip,
    mode_amplitudes,
    mode_series,
    mode_set,
    reference_box,
    wavevectors,
)

TWO_PI = 2.0 * np.pi

CUBE = (6.0, 6.0, 6.0)
OBLONG = (4.0, 6.0, 11.0)


def _brute_mode_set(box, k_cut):
    """Every integer triple inside the cutoff sphere, found by enumeration."""
    box = np.asarray(box, float)
    reach = [int(np.floor(k_cut * length / TWO_PI)) + 2 for length in box]
    kept = []
    for nx in range(-reach[0], reach[0] + 1):
        for ny in range(-reach[1], reach[1] + 1):
            for nz in range(-reach[2], reach[2] + 1):
                n = np.array([nx, ny, nz], float)
                if ((TWO_PI * n / box) ** 2).sum() <= k_cut ** 2:
                    kept.append((nx, ny, nz))
    return np.array(sorted(kept), dtype=np.int64)


# ---------------------------------------------------------------------------
# the declared choices
# ---------------------------------------------------------------------------

def test_the_declared_choices_hold_exactly_the_archived_names():
    assert ORDERINGS == ("lexicographic", "shell")
    assert REFERENCE_BOXES == ("time_mean", "first_frame")
    assert ROUTES == ("dense", "separable")


# ---------------------------------------------------------------------------
# reference_box
# ---------------------------------------------------------------------------

def test_the_time_mean_reference_box_is_the_mean_over_frames():
    lengths = np.array([[4.0, 6.0, 8.0], [6.0, 8.0, 10.0], [8.0, 10.0, 12.0]])
    got = reference_box(lengths, rule="time_mean")
    assert np.array_equal(got, lengths.mean(axis=0))


def test_the_first_frame_reference_box_is_the_first_row():
    lengths = np.array([[4.0, 6.0, 8.0], [6.0, 8.0, 10.0]])
    got = reference_box(lengths, rule="first_frame")
    assert np.array_equal(got, lengths[0])


def test_the_reference_box_does_not_alias_the_timeline_it_came_from():
    lengths = np.array([[4.0, 6.0, 8.0], [6.0, 8.0, 10.0]])
    got = reference_box(lengths, rule="first_frame")
    got[0] = 99.0
    assert lengths[0, 0] == 4.0


def test_an_unknown_reference_rule_names_the_known_ones():
    lengths = np.array([[4.0, 6.0, 8.0]])
    with pytest.raises(ValueError, match="time_mean"):
        reference_box(lengths, rule="median")


@pytest.mark.parametrize("bad, match", [
    (np.zeros((3,)), "timeline of three edge lengths"),
    (np.zeros((2, 4)), "timeline of three edge lengths"),
    (np.zeros((0, 3)), "no frames"),
    (np.array([[4.0, 0.0, 8.0]]), "non-positive edge"),
    (np.array([[4.0, -1.0, 8.0]]), "non-positive edge"),
])
def test_a_reference_box_that_is_not_a_timeline_of_edges_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        reference_box(bad, rule="time_mean")


# ---------------------------------------------------------------------------
# mode_set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("box", [CUBE, OBLONG])
@pytest.mark.parametrize("ordering", ORDERINGS)
def test_the_mode_set_is_exactly_the_integer_triples_inside_the_cutoff(box, ordering):
    got = mode_set(box, k_cut=3.0, ordering=ordering)
    want = _brute_mode_set(box, 3.0)
    assert np.array_equal(np.array(sorted(map(tuple, got.astype(np.int64)))), want)


def test_a_cutoff_below_the_first_mode_leaves_only_the_origin():
    got = mode_set(CUBE, k_cut=TWO_PI / 6.0 - 1e-9, ordering="lexicographic")
    assert np.array_equal(got, np.zeros((1, 3), dtype=np.int16))


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_the_mode_set_is_closed_under_negation(ordering):
    got = mode_set(OBLONG, k_cut=2.0, ordering=ordering).astype(np.int64)
    have = {tuple(row) for row in got}
    assert {tuple(-np.array(row)) for row in have} == have


def test_the_lexicographic_order_runs_over_the_enclosing_integer_cube():
    got = mode_set(OBLONG, k_cut=2.0, ordering="lexicographic").astype(np.int64)
    ranked = got @ np.array([1_000_000, 1_000, 1])
    assert np.all(np.diff(ranked) > 0)


def test_the_lexicographic_order_pairs_a_mode_with_its_negation_end_to_end():
    got = mode_set(OBLONG, k_cut=2.0, ordering="lexicographic").astype(np.int64)
    assert np.array_equal(got, -got[::-1])


def test_the_shell_order_starts_at_the_origin_and_never_decreases_in_wavenumber():
    got = mode_set(OBLONG, k_cut=2.0, ordering="shell")
    assert np.array_equal(got[0], np.zeros(3, dtype=np.int16))
    mag = np.linalg.norm(wavevectors(got, OBLONG), axis=-1)
    assert np.all(np.diff(mag) >= 0.0)


def test_the_two_orderings_are_one_set_under_a_permutation():
    lex = mode_set(OBLONG, k_cut=2.0, ordering="lexicographic")
    shell = mode_set(OBLONG, k_cut=2.0, ordering="shell")
    mag = np.linalg.norm(wavevectors(lex, OBLONG), axis=-1)
    assert np.array_equal(lex[np.argsort(mag, kind="stable")], shell)


def test_each_axis_uses_its_own_box_length():
    got = mode_set(OBLONG, k_cut=4.0, ordering="lexicographic").astype(np.int64)
    reach = np.abs(got).max(axis=0)
    assert np.array_equal(reach, np.floor(4.0 * np.array(OBLONG) / TWO_PI).astype(int))
    assert len(set(reach.tolist())) == 3


def test_a_label_exactly_on_the_cutoff_is_kept():
    """The comparison is inclusive, and that is what the tolerance stands in for.

    On a box of edge 2 pi the wavevector of a label IS the label, so the six
    labels at distance one sit exactly on a cutoff of one -- ``k ** 2`` is
    1.0 to the last bit. Inclusive keeps seven labels there and strict keeps
    one. This is the single point where the third tree's ``|k| <= k_cut +
    1e-12`` could differ from the exact comparison on squares, and it does
    not: both keep the shell.
    """
    box = (TWO_PI, TWO_PI, TWO_PI)
    got = mode_set(box, k_cut=1.0, ordering="lexicographic")
    assert got.shape[0] == 7
    assert np.array_equal(np.array(sorted(map(tuple, got.astype(np.int64)))),
                          _brute_mode_set(box, 1.0))
    k = TWO_PI * np.array([1.0, 0.0, 0.0]) / np.array(box)
    assert float((k ** 2).sum()) == 1.0


def test_a_label_list_that_is_not_double_is_widened_before_the_arithmetic():
    """Mutation: dropping the widening in the shared label check.

    A label array of any integer width promotes exactly, so the widening
    looks free. It is not: a single-precision label array stays single
    through a multiplication by a plain number, and the wavevectors then move
    by 4.4e-08 -- small, and wrong, and silent.
    """
    nvec = mode_set(OBLONG, k_cut=2.0, ordering="lexicographic")
    wide = wavevectors(nvec.astype(np.float64), OBLONG)
    for dtype in (np.int16, np.int32, np.int64, np.float32, np.float64):
        assert np.array_equal(wavevectors(nvec.astype(dtype), OBLONG), wide)


def test_the_mode_labels_are_stored_as_the_archived_integer_width():
    assert mode_set(CUBE, k_cut=3.0, ordering="lexicographic").dtype == np.int16


def test_an_unknown_ordering_names_the_known_ones():
    with pytest.raises(ValueError, match="lexicographic"):
        mode_set(CUBE, k_cut=3.0, ordering="by_energy")


@pytest.mark.parametrize("kwargs, match", [
    (dict(box=CUBE, k_cut=0.0), "is not positive"),
    (dict(box=CUBE, k_cut=-1.0), "is not positive"),
    (dict(box=(6.0, 6.0), k_cut=3.0), "three edge lengths"),
    (dict(box=(6.0, 0.0, 6.0), k_cut=3.0), "non-positive edge"),
])
def test_a_mode_set_that_cannot_be_built_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mode_set(kwargs["box"], k_cut=kwargs["k_cut"], ordering="lexicographic")


# ---------------------------------------------------------------------------
# wavevectors
# ---------------------------------------------------------------------------

def test_a_wavevector_is_the_integer_label_over_the_box_of_its_own_frame():
    nvec = mode_set(OBLONG, k_cut=2.0, ordering="lexicographic")
    got = wavevectors(nvec, OBLONG)
    assert np.array_equal(got, TWO_PI * nvec.astype(np.float64) / np.array(OBLONG))


def test_a_breathing_box_gives_one_wavevector_per_frame_for_each_label():
    nvec = mode_set(OBLONG, k_cut=2.0, ordering="lexicographic")
    lengths = np.array([OBLONG, tuple(1.01 * length for length in OBLONG)])
    got = wavevectors(nvec, lengths)
    assert got.shape == (2, nvec.shape[0], 3)
    assert np.array_equal(got[0], wavevectors(nvec, lengths[0]))
    assert not np.array_equal(got[0], got[1])


@pytest.mark.parametrize("bad, match", [
    (np.zeros(2), "three edge lengths"),
    (np.array([1.0, 0.0, 2.0]), "non-positive edge"),
    (np.zeros((2, 2)), "three edge lengths"),
    (np.zeros((2, 3, 3)), "three edge lengths"),
])
def test_a_wavevector_box_that_is_not_three_edges_is_refused(bad, match):
    nvec = mode_set(CUBE, k_cut=2.0, ordering="lexicographic")
    with pytest.raises(ValueError, match=match):
        wavevectors(nvec, bad)


def test_a_mode_label_array_that_is_not_triples_is_refused():
    with pytest.raises(ValueError, match="integer triples"):
        wavevectors(np.zeros((5, 2), dtype=np.int16), CUBE)


# ---------------------------------------------------------------------------
# mode_amplitudes
# ---------------------------------------------------------------------------

BOUNDS = np.array([[0.0, 4.0], [0.0, 6.0], [0.0, 11.0]])
ONE = ((1,), (7,))        # (atom_types, counts) for a one-channel frame
TWO = ((1, 2), (7, 5))    # ... and a two-channel one
BOTH = [pytest.param(ONE, id="n_species=1"), pytest.param(TWO, id="n_species=2")]
ON_ROUTE = [pytest.param(r, id=r) for r in ROUTES]


def _atoms(counts, types, bounds=BOUNDS, seed=0):
    """Random positions and dump types: ``counts[c]`` atoms of ``types[c]``."""
    rng = np.random.default_rng(seed)
    lo = bounds[:, 0]
    length = bounds[:, 1] - bounds[:, 0]
    pos, typ = [], []
    for dump_type, n in zip(types, counts):
        pos.append(lo + length * rng.random((n, 3)))
        typ.append(np.full(n, int(dump_type), dtype=np.int64))
    return np.concatenate(pos), np.concatenate(typ)


def _on(route):
    """The device keyword each route takes."""
    return {"device": "cpu"} if route == "dense" else {}


def _amplitudes(spec, nvec, route, bounds=BOUNDS, seed=0, extra=()):
    types, counts = spec
    pos, typ = _atoms(counts, types, bounds, seed)
    if extra:
        more_pos, more_typ = _atoms(extra[1], extra[0], bounds, seed + 100)
        pos = np.concatenate([pos, more_pos])
        typ = np.concatenate([typ, more_typ])
    return mode_amplitudes(pos, typ, bounds, nvec, atom_types=types,
                           route=route, **_on(route))


NVEC = mode_set((4.0, 6.0, 11.0), k_cut=2.0, ordering="lexicographic")


@pytest.mark.parametrize("spec", BOTH)
@pytest.mark.parametrize("route", ON_ROUTE)
def test_a_frame_of_amplitudes_is_channel_first_and_double_precision(spec, route):
    got = _amplitudes(spec, NVEC, route)
    assert got.shape == (len(spec[0]), NVEC.shape[0])
    assert got.dtype == np.complex128


@pytest.mark.parametrize("spec", BOTH)
@pytest.mark.parametrize("route", ON_ROUTE)
def test_the_uniform_mode_counts_the_atoms_of_its_channel_exactly(spec, route):
    got = _amplitudes(spec, NVEC, route)
    origin = int(np.flatnonzero((NVEC == 0).all(axis=1))[0])
    assert np.array_equal(got[:, origin], np.array(spec[1], dtype=np.complex128))


@pytest.mark.parametrize("route", ON_ROUTE)
def test_an_atom_at_the_box_origin_contributes_one_to_every_mode(route):
    pos = np.zeros((1, 3))
    typ = np.array([1], dtype=np.int64)
    got = mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1,),
                          route=route, **_on(route))
    assert np.array_equal(got, np.ones((1, NVEC.shape[0]), dtype=np.complex128))


@pytest.mark.parametrize("route", ON_ROUTE)
def test_one_atom_carries_the_defining_phase(route):
    pos = np.array([[1.25, 2.5, 7.75]])
    typ = np.array([1], dtype=np.int64)
    got = mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1,),
                          route=route, **_on(route))
    s = (pos[0] - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    want = np.exp(-2j * np.pi * (NVEC.astype(np.float64) * s).sum(axis=1))
    assert np.abs(got[0] - want).max() < 1e-13


@pytest.mark.parametrize("spec", BOTH)
@pytest.mark.parametrize("route", ON_ROUTE)
def test_the_amplitude_of_a_negated_label_is_the_conjugate(spec, route):
    got = _amplitudes(spec, NVEC, route)
    labels = NVEC.astype(np.int64)
    index = {tuple(row): i for i, row in enumerate(labels)}
    mirror = np.array([index[tuple(-row)] for row in labels])
    assert np.abs(got - np.conj(got[:, mirror])).max() < 1e-12 * sum(spec[1])


@pytest.mark.parametrize("route", ON_ROUTE)
def test_the_channel_order_is_the_declared_order_of_dump_types(route):
    pos, typ = _atoms((7, 5), (1, 2))
    forward = mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1, 2),
                              route=route, **_on(route))
    reverse = mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(2, 1),
                              route=route, **_on(route))
    assert np.array_equal(forward, reverse[::-1])


@pytest.mark.parametrize("spec", BOTH)
@pytest.mark.parametrize("route", ON_ROUTE)
def test_a_dump_type_that_is_not_declared_contributes_to_nothing(spec, route):
    plain = _amplitudes(spec, NVEC, route)
    with_extra = _amplitudes(spec, NVEC, route, extra=((9,), (11,)))
    assert np.array_equal(plain, with_extra)


@pytest.mark.parametrize("route", ON_ROUTE)
def test_a_declared_channel_with_no_atoms_is_exactly_zero(route):
    pos, typ = _atoms((7,), (1,))
    got = mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1, 2),
                          route=route, **_on(route))
    assert np.array_equal(got[1], np.zeros(NVEC.shape[0], dtype=np.complex128))


@pytest.mark.parametrize("route", ON_ROUTE)
def test_one_declared_channel_reads_only_its_own_dump_type(route):
    pos, typ = _atoms((7, 5), (1, 2))
    pair = mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1, 2),
                           route=route, **_on(route))
    alone = mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1,),
                            route=route, **_on(route))
    assert np.array_equal(alone, pair[:1])


@pytest.mark.parametrize("spec", BOTH)
@pytest.mark.parametrize("route", ON_ROUTE)
def test_permuting_the_mode_list_permutes_the_amplitudes(spec, route):
    shell = mode_set((4.0, 6.0, 11.0), k_cut=2.0, ordering="shell")
    lex_amp = _amplitudes(spec, NVEC, route)
    shell_amp = _amplitudes(spec, shell, route)
    order = np.argsort(np.linalg.norm(wavevectors(NVEC, (4.0, 6.0, 11.0)),
                                      axis=-1), kind="stable")
    assert np.array_equal(shell_amp, lex_amp[:, order])


@pytest.mark.parametrize("keep", [
    pytest.param(np.arange(0, NVEC.shape[0], 3), id="every-third"),
    pytest.param(np.flatnonzero(NVEC[:, 2] == 0), id="flat-on-one-axis"),
])
def test_the_dense_route_gives_a_subset_of_the_modes_bit_for_bit(keep):
    full = _amplitudes(TWO, NVEC, "dense")
    part = _amplitudes(TWO, NVEC[keep], "dense")
    assert np.array_equal(part, full[:, keep])


@pytest.mark.parametrize("keep", [
    pytest.param(np.arange(0, NVEC.shape[0], 3), id="every-third"),
    pytest.param(np.flatnonzero(NVEC[:, 2] == 0), id="flat-on-one-axis"),
])
def test_the_separable_route_gives_a_subset_of_the_modes_to_round_off(keep):
    """The separable route's answer depends on the cube it is asked for.

    The contraction is over the whole enclosing integer cube, so asking for
    fewer labels changes the shape of the contraction and therefore the order
    the atom sum is accumulated in. Measured on this frame the rows move by
    5.6e-16 and 6.3e-16 on a sum of twelve unit-modulus terms, which is
    round-off and nothing else -- but it is NOT zero, so an archived artefact
    is reproduced bit for bit by this route only when the mode list asked for
    is exactly the archived one. The dense route has no such dependence and
    the test above pins that.
    """
    full = _amplitudes(TWO, NVEC, "separable")
    part = _amplitudes(TWO, NVEC[keep], "separable")
    assert not np.array_equal(part, full[:, keep])
    assert np.abs(part - full[:, keep]).max() < 1e-14


@pytest.mark.parametrize("route", ON_ROUTE)
def test_a_mode_list_that_is_not_closed_under_negation_still_reads_out(route):
    """Mutation: bounding the separable cube by the largest label rather than
    the largest absolute one. Every symmetric list makes the two the same
    number, and a mode ball is symmetric, so only a list that is not -- here
    the labels whose first index is negative, whose largest is -1 and whose
    largest absolute is 1 -- tells them apart. The docstring of the plan
    claims any subset reads out correctly, and this is the subset that
    claim is about."""
    keep = np.flatnonzero(NVEC[:, 0] < 0)
    assert NVEC[keep].max(axis=0)[0] != np.abs(NVEC[keep]).max(axis=0)[0]
    full = _amplitudes(TWO, NVEC, route)
    part = _amplitudes(TWO, NVEC[keep], route)
    assert np.abs(part - full[:, keep]).max() < 1e-14


def test_the_two_routes_agree_to_double_precision_on_the_same_frame():
    dense = _amplitudes(TWO, NVEC, "dense")
    separable = _amplitudes(TWO, NVEC, "separable")
    assert np.abs(dense - separable).max() < 1e-12 * sum(TWO[1])


@pytest.mark.parametrize("route", ON_ROUTE)
def test_an_unwrapped_coordinate_is_the_same_atom(route):
    pos, typ = _atoms((7, 5), (1, 2))
    shifted = pos + (BOUNDS[:, 1] - BOUNDS[:, 0])
    here = mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1, 2),
                           route=route, **_on(route))
    there = mode_amplitudes(shifted, typ, BOUNDS, NVEC, atom_types=(1, 2),
                            route=route, **_on(route))
    assert np.abs(here - there).max() < 1e-10 * sum(TWO[1])


def test_an_unknown_route_names_the_known_ones():
    pos, typ = _atoms((7,), (1,))
    with pytest.raises(ValueError, match="separable"):
        mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1,), route="fft")


def test_the_dense_route_needs_a_device_rather_than_choosing_one():
    pos, typ = _atoms((7,), (1,))
    with pytest.raises(ValueError, match="needs device"):
        mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1,), route="dense")


def test_the_separable_route_refuses_a_device_it_cannot_honour():
    pos, typ = _atoms((7,), (1,))
    with pytest.raises(ValueError, match="no device to honour"):
        mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1,),
                        route="separable", device="cuda")


def test_an_unknown_device_names_the_known_ones():
    pos, typ = _atoms((7,), (1,))
    with pytest.raises(ValueError, match="device="):
        mode_amplitudes(pos, typ, BOUNDS, NVEC, atom_types=(1,),
                        route="dense", device="tpu")


@pytest.mark.parametrize("route", ON_ROUTE)
@pytest.mark.parametrize("kwargs, match", [
    (dict(atom_types=()), "atom_types is empty"),
    (dict(atom_types=(1, 1)), "repeats"),
])
def test_a_channel_declaration_that_cannot_be_honoured_is_refused(route, kwargs,
                                                                  match):
    pos, typ = _atoms((7,), (1,))
    with pytest.raises(ValueError, match=match):
        mode_amplitudes(pos, typ, BOUNDS, NVEC, route=route, **_on(route),
                        **kwargs)


@pytest.mark.parametrize("route", ON_ROUTE)
def test_positions_that_are_not_triples_are_refused(route):
    with pytest.raises(ValueError, match="positions"):
        mode_amplitudes(np.zeros((7, 2)), np.ones(7, dtype=np.int64), BOUNDS,
                        NVEC, atom_types=(1,), route=route, **_on(route))


@pytest.mark.parametrize("route", ON_ROUTE)
def test_one_dump_type_per_atom_is_required(route):
    with pytest.raises(ValueError, match="types"):
        mode_amplitudes(np.zeros((7, 3)), np.ones(5, dtype=np.int64), BOUNDS,
                        NVEC, atom_types=(1,), route=route, **_on(route))


@pytest.mark.parametrize("route", ON_ROUTE)
@pytest.mark.parametrize("bad, match", [
    (np.zeros((3, 3)), r"not \(3, 2\)"),
    (np.zeros((2, 2)), r"not \(3, 2\)"),
    (np.array([[0.0, 4.0], [0.0, 0.0], [0.0, 11.0]]), "non-positive edge"),
])
def test_a_box_that_is_not_three_pairs_of_bounds_is_refused(route, bad, match):
    pos, typ = _atoms((7,), (1,))
    with pytest.raises(ValueError, match=match):
        mode_amplitudes(pos, typ, bad, NVEC, atom_types=(1,), route=route,
                        **_on(route))


@pytest.mark.parametrize("route", ON_ROUTE)
def test_a_mode_list_that_is_not_triples_is_refused(route):
    pos, typ = _atoms((7,), (1,))
    with pytest.raises(ValueError, match="integer triples"):
        mode_amplitudes(pos, typ, BOUNDS, np.zeros((5, 2), dtype=np.int16),
                        atom_types=(1,), route=route, **_on(route))


# ---------------------------------------------------------------------------
# frames_to_skip
# ---------------------------------------------------------------------------

def test_a_dump_that_started_inside_equilibration_drops_its_leading_frames():
    steps = np.array([0, 100, 200, 300, 400, 500], dtype=np.int64)
    got = frames_to_skip(steps, equilibration_steps=300, production_steps=400)
    assert got == 3


def test_a_dump_that_ends_at_the_production_count_drops_nothing():
    steps = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    got = frames_to_skip(steps, equilibration_steps=300, production_steps=400)
    assert got == 0


@pytest.mark.parametrize("kwargs", [
    dict(equilibration_steps=None, production_steps=400),
    dict(equilibration_steps=300, production_steps=None),
    dict(equilibration_steps=None, production_steps=None),
])
def test_a_run_that_does_not_record_both_counts_drops_nothing(kwargs):
    steps = np.array([0, 100, 200, 300, 400, 500], dtype=np.int64)
    assert frames_to_skip(steps, **kwargs) == 0


def test_a_recorded_zero_is_a_count_and_not_a_missing_value():
    """A truth test on both counts would read a recorded zero as absent.

    A guard written ``if not (n_melt and n_prod)`` cannot tell a run that
    recorded no production from one that recorded none at all. Here ``None`` says absent and a number says what it says.
    """
    steps = np.array([0, 100, 200, 300], dtype=np.int64)
    assert frames_to_skip(steps, equilibration_steps=200,
                          production_steps=0) == 2
    assert frames_to_skip(steps, equilibration_steps=200,
                          production_steps=None) == 0


def test_the_skip_counts_early_frames_rather_than_taking_a_prefix():
    steps = np.array([500, 0, 400, 100], dtype=np.int64)
    assert frames_to_skip(steps, equilibration_steps=300,
                          production_steps=400) == 2


def test_an_empty_timeline_has_no_frames_to_skip():
    assert frames_to_skip(np.zeros(0, dtype=np.int64), equilibration_steps=300,
                          production_steps=400) == 0


@pytest.mark.parametrize("kwargs, match", [
    (dict(equilibration_steps=-1, production_steps=400), "negative"),
    (dict(equilibration_steps=300, production_steps=-1), "negative"),
])
def test_a_negative_step_count_is_refused(kwargs, match):
    steps = np.array([0, 100], dtype=np.int64)
    with pytest.raises(ValueError, match=match):
        frames_to_skip(steps, **kwargs)


def test_a_timeline_of_timesteps_has_one_axis():
    with pytest.raises(ValueError, match="one timestep per frame"):
        frames_to_skip(np.zeros((2, 2), dtype=np.int64),
                       equilibration_steps=300, production_steps=400)


# ---------------------------------------------------------------------------
# mode_series
# ---------------------------------------------------------------------------

def _frames(spec, n_frames=4, bounds=BOUNDS, breathe=0.0):
    """A short timeline of frames, with an optional breathing box."""
    from aipf.pipeline.coarse_grain import Frame

    types, counts = spec
    out = []
    for f in range(n_frames):
        scale = 1.0 + breathe * f
        here = bounds * np.array([1.0, scale])
        pos, typ = _atoms(counts, types, here, seed=f)
        out.append(Frame(timestep=100 * f, box_bounds=here, positions=pos,
                         types=typ))
    return out


@pytest.mark.parametrize("spec", BOTH)
@pytest.mark.parametrize("route", ON_ROUTE)
def test_a_timeline_is_frames_then_channels_then_modes(spec, route):
    rho_k, timesteps = mode_series(_frames(spec), NVEC, atom_types=spec[0],
                                   route=route, dtype=np.complex64,
                                   **_on(route))
    assert rho_k.shape == (4, len(spec[0]), NVEC.shape[0])
    assert rho_k.dtype == np.complex64
    assert np.array_equal(timesteps, np.array([0, 100, 200, 300]))
    assert timesteps.dtype == np.int64


@pytest.mark.parametrize("spec", BOTH)
@pytest.mark.parametrize("route", ON_ROUTE)
def test_every_frame_of_a_timeline_is_that_frame_on_its_own(spec, route):
    frames = _frames(spec, breathe=0.02)
    rho_k, _ = mode_series(frames, NVEC, atom_types=spec[0], route=route,
                           dtype=np.complex128, **_on(route))
    for f, frame in enumerate(frames):
        alone = mode_amplitudes(frame.positions, frame.types, frame.box_bounds,
                                NVEC, atom_types=spec[0], route=route,
                                **_on(route))
        assert np.array_equal(rho_k[f], alone)


@pytest.mark.parametrize("route", ON_ROUTE)
def test_each_frame_is_read_in_its_own_breathing_box(route):
    still = mode_series(_frames(TWO), NVEC, atom_types=(1, 2), route=route,
                        dtype=np.complex128, **_on(route))[0]
    moving = mode_series(_frames(TWO, breathe=0.02), NVEC, atom_types=(1, 2),
                         route=route, dtype=np.complex128, **_on(route))[0]
    assert np.array_equal(still[0], moving[0])
    assert not np.array_equal(still[1], moving[1])


@pytest.mark.parametrize("route", ON_ROUTE)
def test_the_stored_width_is_a_rounding_of_the_double_precision_sum(route):
    wide = mode_series(_frames(TWO), NVEC, atom_types=(1, 2), route=route,
                       dtype=np.complex128, **_on(route))[0]
    narrow = mode_series(_frames(TWO), NVEC, atom_types=(1, 2), route=route,
                         dtype=np.complex64, **_on(route))[0]
    assert np.array_equal(narrow, wide.astype(np.complex64))


def test_a_timeline_may_be_a_generator_read_once():
    frames = _frames(TWO)
    rho_k, timesteps = mode_series(iter(frames), NVEC, atom_types=(1, 2),
                                   route="separable", dtype=np.complex64)
    assert rho_k.shape[0] == len(frames)


def test_a_timeline_with_no_frames_is_refused():
    with pytest.raises(ValueError, match="no frames"):
        mode_series([], NVEC, atom_types=(1, 2), route="separable",
                    dtype=np.complex64)


def test_a_stored_width_that_is_not_complex_is_refused():
    with pytest.raises(ValueError, match="complex"):
        mode_series(_frames(TWO), NVEC, atom_types=(1, 2), route="separable",
                    dtype=np.float32)


def test_a_timeline_passes_the_route_and_the_device_through():
    with pytest.raises(ValueError, match="no device to honour"):
        mode_series(_frames(TWO), NVEC, atom_types=(1, 2), route="separable",
                    device="cpu", dtype=np.complex64)
