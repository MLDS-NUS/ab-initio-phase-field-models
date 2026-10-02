"""Tests for :mod:`aipf.pipeline.qc`.

Every check is exercised at one channel and at two, every threshold is passed
in rather than assumed, and every refusal is checked by the words it says
rather than by its type alone -- a neighbouring refusal in the same function
often raises the same class.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aipf.pipeline import qc


# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------

def test_the_verdicts_are_the_three_the_trees_agree_on():
    assert qc.VERDICTS == ("clean", "flagged", "unusable")


def test_every_flag_name_is_unique_and_lower_case():
    assert len(set(qc.FLAGS)) == len(qc.FLAGS)
    assert all(name == name.lower() and " " not in name for name in qc.FLAGS)


def test_no_flag_or_verdict_names_a_species_or_an_observable():
    forbidden = ("temp", "volume", "vol", "energy", "box", "pressure")
    for name in qc.FLAGS + qc.VERDICTS:
        assert not any(word in name for word in forbidden), name


def test_a_run_with_no_flags_is_clean():
    assert qc.verdict((), fatal=()) == "clean"


def test_a_run_with_a_flag_nobody_called_fatal_is_flagged():
    assert qc.verdict(("short_of_expected",), fatal=()) == "flagged"


def test_a_run_with_a_declared_fatal_flag_is_unusable():
    assert qc.verdict(("short_of_expected",), fatal=("short_of_expected",)) == "unusable"


def test_the_same_flag_is_fatal_for_one_caller_and_not_another():
    flags = ("masked_tail",)
    assert qc.verdict(flags, fatal=("masked_tail",)) == "unusable"
    assert qc.verdict(flags, fatal=("no_frames",)) == "flagged"


def test_a_fatal_name_that_is_no_flag_is_refused():
    with pytest.raises(ValueError, match="fatal"):
        qc.verdict((), fatal=("crystallised",))


def test_a_flag_outside_the_vocabulary_is_refused():
    with pytest.raises(ValueError, match="flags"):
        qc.verdict(("crystallised",), fatal=())


def test_a_caller_may_widen_the_vocabulary_with_its_own_checks():
    wider = qc.FLAGS + ("crystallised",)
    assert qc.verdict(("crystallised",), fatal=("crystallised",),
                      vocabulary=wider) == "unusable"
    assert qc.verdict(("crystallised",), fatal=(), vocabulary=wider) == "flagged"


def test_a_widened_vocabulary_still_refuses_a_name_outside_it():
    with pytest.raises(ValueError, match="fatal"):
        qc.verdict((), fatal=("arrested",),
                   vocabulary=qc.FLAGS + ("crystallised",))


# ---------------------------------------------------------------------------
# expected_frames
# ---------------------------------------------------------------------------

def test_expected_frames_counts_the_frame_at_the_first_step():
    assert qc.expected_frames(1000, dump_every_steps=100) == 11


def test_expected_frames_floors_a_run_that_does_not_divide():
    assert qc.expected_frames(1050, dump_every_steps=100) == 11


def test_expected_frames_of_a_run_that_dumps_only_its_first_step():
    assert qc.expected_frames(0, dump_every_steps=100) == 1


def test_expected_frames_refuses_a_negative_run():
    with pytest.raises(ValueError, match="production_steps"):
        qc.expected_frames(-1, dump_every_steps=100)


def test_expected_frames_refuses_a_cadence_of_zero():
    with pytest.raises(ValueError, match="dump_every_steps"):
        qc.expected_frames(1000, dump_every_steps=0)


# ---------------------------------------------------------------------------
# frame_inventory
# ---------------------------------------------------------------------------

def _steps(n, every=100, start=0):
    return np.arange(n, dtype=np.int64) * every + start


def test_an_even_timeline_is_complete_and_carries_no_flag():
    inv = qc.frame_inventory(_steps(11))
    assert inv.n_frames == 11
    assert inv.first_step == 0 and inv.last_step == 1000
    assert inv.step_interval == 100
    assert inv.flags == ()


def test_an_empty_timeline_is_named_rather_than_left_to_look_like_a_short_one():
    inv = qc.frame_inventory(np.array([], dtype=np.int64))
    assert inv.n_frames == 0
    assert "no_frames" in inv.flags
    assert inv.first_step is None and inv.step_interval is None


def test_an_empty_timeline_against_an_expectation_is_zero_complete():
    inv = qc.frame_inventory(np.array([], dtype=np.int64), expected=11,
                             tolerated_fraction=0.95)
    assert inv.completeness == 0.0
    assert "no_frames" in inv.flags and "short_of_expected" in inv.flags


def test_a_single_frame_has_no_cadence_to_report():
    inv = qc.frame_inventory(np.array([7], dtype=np.int64))
    assert inv.n_frames == 1 and inv.step_interval is None
    assert inv.flags == ()


def test_a_gap_in_the_middle_is_uneven_cadence():
    steps = np.array([0, 100, 300, 400], dtype=np.int64)
    inv = qc.frame_inventory(steps)
    assert "uneven_cadence" in inv.flags
    assert inv.step_interval is None


def test_a_repeated_step_is_named():
    steps = np.array([0, 100, 100, 200], dtype=np.int64)
    inv = qc.frame_inventory(steps)
    assert "repeated_step" in inv.flags


def test_a_step_that_goes_backwards_is_named():
    steps = np.array([0, 200, 100, 300], dtype=np.int64)
    inv = qc.frame_inventory(steps)
    assert "out_of_order" in inv.flags


def test_a_cadence_that_is_not_the_declared_one_is_named():
    inv = qc.frame_inventory(_steps(5, every=50), dump_every_steps=100)
    assert "cadence_mismatch" in inv.flags
    assert inv.step_interval == 50


def test_the_declared_cadence_is_not_claimed_against_an_uneven_timeline():
    steps = np.array([0, 100, 300], dtype=np.int64)
    inv = qc.frame_inventory(steps, dump_every_steps=100)
    assert "uneven_cadence" in inv.flags
    assert "cadence_mismatch" not in inv.flags


def test_a_timeline_matching_its_declared_cadence_carries_no_flag():
    inv = qc.frame_inventory(_steps(5, every=100), dump_every_steps=100)
    assert inv.flags == ()


def test_a_short_dump_is_flagged_against_the_tolerated_fraction():
    inv = qc.frame_inventory(_steps(9), expected=11, tolerated_fraction=0.95)
    assert "short_of_expected" in inv.flags
    assert inv.completeness == pytest.approx(9 / 11)


def test_a_dump_inside_the_tolerated_fraction_is_not_short():
    inv = qc.frame_inventory(_steps(20), expected=20, tolerated_fraction=0.95)
    assert inv.flags == ()
    inv = qc.frame_inventory(_steps(19), expected=20, tolerated_fraction=0.95)
    assert inv.flags == ()


def test_the_short_test_is_inclusive_at_the_tolerated_fraction():
    # 19 of 20 is exactly 0.95: the boundary run is kept, as a test written
    # `n < frac * expect` keeps it.
    inv = qc.frame_inventory(_steps(19), expected=20, tolerated_fraction=0.95)
    assert "short_of_expected" not in inv.flags


def test_an_expectation_without_a_tolerance_is_refused():
    with pytest.raises(ValueError, match="tolerated_fraction"):
        qc.frame_inventory(_steps(5), expected=11)


def test_a_tolerance_without_an_expectation_is_refused():
    with pytest.raises(ValueError, match="expected"):
        qc.frame_inventory(_steps(5), tolerated_fraction=0.95)


def test_a_non_positive_expectation_is_refused():
    with pytest.raises(ValueError, match="expected"):
        qc.frame_inventory(_steps(5), expected=0, tolerated_fraction=0.95)


def test_a_non_positive_declared_cadence_is_refused():
    with pytest.raises(ValueError, match="dump_every_steps"):
        qc.frame_inventory(_steps(5), dump_every_steps=0)


def test_a_two_dimensional_timeline_is_refused():
    with pytest.raises(ValueError, match="timesteps"):
        qc.frame_inventory(np.zeros((3, 2), dtype=np.int64))


def test_a_non_integer_timeline_is_refused():
    with pytest.raises(ValueError, match="integer"):
        qc.frame_inventory(np.array([0.0, 0.5, 1.0]))


def test_the_inventory_reports_the_steps_it_was_given_not_a_count():
    inv = qc.frame_inventory(_steps(4, every=20, start=6000))
    assert inv.first_step == 6000 and inv.last_step == 6060


# ---------------------------------------------------------------------------
# channel_occupancy
# ---------------------------------------------------------------------------

def test_one_channel_is_the_ordinary_case():
    types = np.array([1, 1, 1, 1], dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1,))
    assert occ.counts == (4,)
    assert occ.shares == (1.0,)
    assert occ.flags == ()


def test_two_channels_are_counted_in_the_declared_order():
    types = np.array([2, 2, 2, 1], dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2))
    assert occ.counts == (1, 3)
    occ = qc.channel_occupancy(types, atom_types=(2, 1))
    assert occ.counts == (3, 1)


def test_a_declared_channel_with_no_atoms_is_named_not_left_to_be_a_zero_field():
    types = np.array([1, 1, 1], dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2))
    assert occ.counts == (3, 0)
    assert occ.empty == (1,)
    assert "empty_channel" in occ.flags


def test_every_declared_channel_empty_is_still_reported_as_shares_of_zero():
    types = np.array([7, 7], dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2))
    assert occ.counts == (0, 0)
    assert occ.shares == (0.0, 0.0)
    assert occ.empty == (0, 1)
    assert occ.n_unassigned == 2


def test_an_atom_of_no_declared_channel_is_counted_separately():
    types = np.array([1, 2, 3, 3], dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2))
    assert occ.counts == (1, 1)
    assert occ.n_unassigned == 2
    assert occ.shares == (0.5, 0.5)


def test_an_empty_frame_has_no_atoms_anywhere():
    occ = qc.channel_occupancy(np.array([], dtype=np.int64), atom_types=(1, 2))
    assert occ.counts == (0, 0) and occ.n_unassigned == 0
    assert "empty_channel" in occ.flags


def test_a_minority_channel_is_named_by_its_share():
    types = np.array([1] * 3 + [2] * 97, dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2),
                               minority_fraction=0.06, minority_atoms=400)
    assert occ.minority == (0,)
    assert "minority_channel" in occ.flags


def test_the_minority_share_test_is_inclusive_at_the_threshold():
    # 5 of 100 is exactly 0.05. A strict `< 0.05` makes the guard
    # dead code at the extreme composition it is written for; the inclusive
    # form is the measured fix.
    types = np.array([1] * 5 + [2] * 95, dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2),
                               minority_fraction=0.05, minority_atoms=1)
    assert occ.minority == (0,)


def test_the_minority_atom_count_fires_where_the_share_does_not():
    types = np.array([1] * 300 + [2] * 700, dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2),
                               minority_fraction=0.06, minority_atoms=400)
    assert occ.minority == (0,)


def test_the_minority_atom_test_is_strict_at_the_threshold():
    types = np.array([1] * 400 + [2] * 600, dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2),
                               minority_fraction=0.01, minority_atoms=400)
    assert occ.minority == ()


def test_a_majority_channel_is_never_a_minority_however_small_the_box():
    types = np.array([1] * 2, dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1,),
                               minority_fraction=0.9, minority_atoms=400)
    assert occ.minority == ()


def test_an_equal_split_of_two_channels_is_no_minority():
    types = np.array([1] * 50 + [2] * 50, dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2),
                               minority_fraction=0.6, minority_atoms=1)
    assert occ.minority == ()


def test_no_minority_thresholds_means_no_minority_test():
    types = np.array([1] + [2] * 999, dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2))
    assert occ.minority == ()
    assert "minority_channel" not in occ.flags


def test_one_minority_threshold_without_the_other_is_refused():
    types = np.array([1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="minority_atoms"):
        qc.channel_occupancy(types, atom_types=(1, 2), minority_fraction=0.06)
    with pytest.raises(ValueError, match="minority_fraction"):
        qc.channel_occupancy(types, atom_types=(1, 2), minority_atoms=400)


def test_a_repeated_dump_type_is_refused_as_it_is_in_the_kernel():
    with pytest.raises(ValueError, match="repeats"):
        qc.channel_occupancy(np.array([1], dtype=np.int64), atom_types=(1, 1))


def test_no_declared_channel_is_refused():
    with pytest.raises(ValueError, match="atom_types"):
        qc.channel_occupancy(np.array([1], dtype=np.int64), atom_types=())


def test_a_two_dimensional_type_array_is_refused():
    with pytest.raises(ValueError, match="types"):
        qc.channel_occupancy(np.zeros((2, 2), dtype=np.int64), atom_types=(1,))


# ---------------------------------------------------------------------------
# observable_report
# ---------------------------------------------------------------------------

def test_a_steady_series_is_on_target_and_not_drifting():
    values = np.full(40, 1200.0)
    rep = qc.observable_report(values, target=1200.0, tolerance=1.0,
                               drift_max=0.01)
    assert rep.flags == ()
    assert rep.mean == 1200.0 and rep.drift == 0.0


def test_a_series_off_its_target_is_named():
    values = np.full(40, 1210.0)
    rep = qc.observable_report(values, target=1200.0, tolerance=5.0)
    assert "off_target" in rep.flags


def test_the_target_test_is_inclusive_at_the_tolerance():
    values = np.full(40, 1205.0)
    rep = qc.observable_report(values, target=1200.0, tolerance=5.0)
    assert rep.flags == ()


def test_a_series_whose_halves_differ_is_drifting():
    values = np.concatenate([np.full(20, 100.0), np.full(20, 105.0)])
    rep = qc.observable_report(values, drift_max=0.01)
    assert "drifting" in rep.flags
    assert rep.drift == pytest.approx(5.0 / 102.5)


def test_the_drift_is_scaled_by_the_size_of_the_series_not_its_sign():
    # Dividing by `max(mean, 1e-30)` would be a division by 1e-30
    # for any observable whose mean is negative -- an energy, for instance --
    # and would flag it whatever it does. The scale is the magnitude.
    values = np.concatenate([np.full(20, -100.0), np.full(20, -105.0)])
    rep = qc.observable_report(values, drift_max=0.1)
    assert rep.drift == pytest.approx(5.0 / 102.5)
    assert rep.flags == ()


def test_a_non_finite_entry_is_named_and_the_rest_still_reported():
    values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
    rep = qc.observable_report(values)
    assert "not_finite" in rep.flags
    assert rep.n_not_finite == 1
    assert rep.mean == pytest.approx(3.0)


def test_a_series_of_nothing_finite_reports_no_statistics_rather_than_a_number():
    values = np.array([np.nan, np.nan, np.nan, np.nan])
    rep = qc.observable_report(values, target=0.0, tolerance=1.0)
    assert "not_finite" in rep.flags
    assert math.isnan(rep.mean)
    assert "off_target" not in rep.flags


def test_the_drift_ignores_a_non_finite_entry_rather_than_returning_one():
    values = np.concatenate([np.full(20, 100.0), np.full(20, 105.0)])
    values[3] = np.inf
    rep = qc.observable_report(values, drift_max=1.0)
    assert np.isfinite(rep.drift)


def test_a_target_without_a_tolerance_is_refused():
    with pytest.raises(ValueError, match="tolerance"):
        qc.observable_report(np.zeros(10), target=0.0)


def test_a_tolerance_without_a_target_is_refused():
    with pytest.raises(ValueError, match="target"):
        qc.observable_report(np.zeros(10), tolerance=1.0)


def test_a_negative_tolerance_is_refused():
    with pytest.raises(ValueError, match="tolerance"):
        qc.observable_report(np.zeros(10), target=0.0, tolerance=-1.0)


def test_a_negative_drift_limit_is_refused():
    with pytest.raises(ValueError, match="drift_max"):
        qc.observable_report(np.zeros(10), drift_max=-0.1)


def test_a_drift_over_one_sample_is_refused_rather_than_invented():
    with pytest.raises(ValueError, match="halves"):
        qc.observable_report(np.zeros(1), drift_max=0.1)


def test_an_empty_series_is_refused():
    with pytest.raises(ValueError, match="values"):
        qc.observable_report(np.zeros(0))


def test_a_two_dimensional_series_is_refused():
    with pytest.raises(ValueError, match="values"):
        qc.observable_report(np.zeros((3, 2)))


def test_a_complex_series_is_refused_because_a_target_has_no_meaning_on_it():
    with pytest.raises(ValueError, match="real"):
        qc.observable_report(np.zeros(4, dtype=np.complex128))


def test_the_standard_deviation_of_one_sample_is_not_a_number():
    rep = qc.observable_report(np.array([3.0]))
    assert rep.mean == 3.0 and math.isnan(rep.std)


# ---------------------------------------------------------------------------
# block_means
# ---------------------------------------------------------------------------

def test_block_means_average_each_span_and_report_its_end():
    times = np.arange(20) * 0.1
    values = np.arange(20, dtype=float)
    ends, means, per_block = qc.block_means(times, values, block_span=1.0)
    assert per_block == 10
    assert ends.tolist() == pytest.approx([1.0, 2.0])
    assert means.tolist() == pytest.approx([4.5, 14.5])


def test_a_ragged_final_block_is_dropped_when_it_is_under_half_full():
    times = np.arange(24) * 0.1
    values = np.ones(24)
    ends, means, per_block = qc.block_means(times, values, block_span=1.0)
    assert len(means) == 2
    assert per_block == 10


def test_a_ragged_final_block_is_kept_when_it_is_at_least_half_full():
    times = np.arange(25) * 0.1
    values = np.ones(25)
    ends, means, _ = qc.block_means(times, values, block_span=1.0)
    assert len(means) == 3


def test_a_gap_in_the_times_drops_the_empty_block_rather_than_averaging_nothing():
    times = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 2.2])
    values = np.ones(6)
    ends, means, _ = qc.block_means(times, values, block_span=1.0)
    assert len(means) == 2
    assert ends.tolist() == pytest.approx([1.0, 3.0])


def test_the_blocks_are_measured_from_the_first_sample_not_from_zero():
    times = np.arange(20) * 0.1 + 50.0
    values = np.arange(20, dtype=float)
    ends, means, _ = qc.block_means(times, values, block_span=1.0)
    assert ends.tolist() == pytest.approx([51.0, 52.0])


def test_block_means_refuse_a_non_positive_span():
    with pytest.raises(ValueError, match="block_span"):
        qc.block_means(np.arange(4.0), np.arange(4.0), block_span=0.0)


def test_block_means_refuse_times_that_go_backwards():
    times = np.array([0.0, 1.0, 0.5, 2.0])
    with pytest.raises(ValueError, match="decreases"):
        qc.block_means(times, np.ones(4), block_span=1.0)


def test_block_means_refuse_mismatched_lengths():
    with pytest.raises(ValueError, match="values"):
        qc.block_means(np.arange(4.0), np.ones(3), block_span=1.0)


def test_block_means_refuse_an_empty_series():
    with pytest.raises(ValueError, match="times"):
        qc.block_means(np.zeros(0), np.zeros(0), block_span=1.0)


# ---------------------------------------------------------------------------
# settle_time
# ---------------------------------------------------------------------------

def _settling(n_blocks=20, per_block=10, transient=3, span=1.0):
    """A series that starts off its level and settles onto it."""
    rng = np.random.default_rng(0)
    n = n_blocks * per_block
    times = np.arange(n) * (span / per_block)
    values = 100.0 + rng.normal(0.0, 0.01, n)
    values[: transient * per_block] += 10.0
    return times, values


def test_a_transient_at_the_start_is_measured_to_where_it_ends():
    times, values = _settling(transient=3)
    out = qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                         persist=2, min_blocks=6)
    assert out.settle == pytest.approx(3.0)
    assert out.flags == ()


def test_a_series_that_never_leaves_its_band_settles_at_once():
    times, values = _settling(transient=0)
    out = qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                         persist=2, min_blocks=6)
    assert out.settle == 0.0


def test_a_lone_block_in_mid_series_is_not_a_settle_time():
    times, values = _settling(transient=0)
    values[3 * 10: 4 * 10] += 10.0            # one block, away from the start
    out = qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                         persist=2, min_blocks=6)
    assert out.settle == 0.0
    assert out.strict == pytest.approx(4.0)


def test_a_lone_block_at_the_start_is_a_settle_time():
    times, values = _settling(transient=1)
    out = qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                         persist=2, min_blocks=6)
    assert out.settle == pytest.approx(1.0)


def test_the_strict_value_reports_the_last_excursion_wherever_it_is():
    times, values = _settling(transient=2)
    values[7 * 10: 8 * 10] += 10.0
    out = qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                         persist=2, min_blocks=6)
    assert out.settle == pytest.approx(2.0)
    assert out.strict == pytest.approx(8.0)


def test_a_wider_band_settles_no_later_than_a_narrower_one():
    times, values = _settling(transient=3)
    narrow = qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                            persist=2, min_blocks=6)
    wide = qc.settle_time(times, values, block_span=1.0, tol_sigma=3000.0,
                          persist=2, min_blocks=6)
    assert wide.settle <= narrow.settle


def test_a_longer_persistence_ignores_a_short_transient_that_does_not_start_at_zero():
    times, values = _settling(transient=0)
    values[20:40] += 10.0                     # blocks 2 and 3
    two = qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                         persist=2, min_blocks=6)
    three = qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                           persist=3, min_blocks=6)
    assert two.settle == pytest.approx(4.0)
    assert three.settle == 0.0


def test_the_reference_band_comes_from_the_last_half_not_the_whole_series():
    # With the whole series as the reference the transient inflates the
    # scatter and swallows itself.
    times, values = _settling(transient=4)
    out = qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                         persist=2, min_blocks=6)
    assert out.settle == pytest.approx(4.0)
    assert out.reference == pytest.approx(100.0, abs=0.05)
    assert out.scatter < 1.0


def test_a_series_too_short_to_block_says_so_rather_than_returning_zero():
    times = np.arange(30) * 0.1
    out = qc.settle_time(times, np.ones(30), block_span=1.0, tol_sigma=3.0,
                         persist=2, min_blocks=6)
    assert out.settle is None and out.strict is None
    assert "unsettled" in out.flags
    assert out.n_blocks == 3


def test_a_perfectly_constant_series_settles_rather_than_dividing_by_zero():
    times = np.arange(200) * 0.1
    out = qc.settle_time(times, np.full(200, 7.0), block_span=1.0,
                         tol_sigma=3.0, persist=2, min_blocks=6)
    assert out.settle == 0.0
    assert out.scatter > 0.0


def test_settle_time_refuses_a_non_positive_band_width():
    times, values = _settling()
    with pytest.raises(ValueError, match="tol_sigma"):
        qc.settle_time(times, values, block_span=1.0, tol_sigma=0.0,
                       persist=2, min_blocks=6)


def test_settle_time_refuses_a_persistence_below_one():
    times, values = _settling()
    with pytest.raises(ValueError, match="persist"):
        qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                       persist=0, min_blocks=6)


def test_settle_time_refuses_a_minimum_block_count_below_two():
    times, values = _settling()
    with pytest.raises(ValueError, match="min_blocks"):
        qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                       persist=2, min_blocks=1)


def test_settle_time_refuses_a_series_with_a_non_finite_entry():
    times, values = _settling()
    values[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                       persist=2, min_blocks=6)


# ---------------------------------------------------------------------------
# burn_in_frames
# ---------------------------------------------------------------------------

def test_a_measured_settle_longer_than_the_floor_sets_the_burn_in():
    assert qc.burn_in_frames(31.0, floor=30.0, sample_interval=0.1) == 310


def test_a_floor_longer_than_the_measured_settle_sets_the_burn_in():
    assert qc.burn_in_frames(11.0, floor=30.0, sample_interval=0.1) == 300


def test_the_settle_is_rounded_up_and_the_floor_is_not():
    assert qc.burn_in_frames(30.4, floor=5.25, sample_interval=0.1) == 310
    # 5.25 over 0.1 is 52.5 frames, and the rounding is the one the archived
    # overlay used: to even, not away from zero.
    assert qc.burn_in_frames(1.0, floor=5.25, sample_interval=0.1) == 52


def test_an_unmeasured_settle_falls_back_to_the_floor_alone():
    assert qc.burn_in_frames(None, floor=30.0, sample_interval=0.1) == 300


def test_a_burn_in_is_a_whole_number_of_frames():
    out = qc.burn_in_frames(7.0, floor=0.0, sample_interval=0.3)
    assert isinstance(out, int) and out == 23


def test_burn_in_frames_refuses_a_negative_settle():
    with pytest.raises(ValueError, match="settle"):
        qc.burn_in_frames(-1.0, floor=0.0, sample_interval=0.1)


def test_burn_in_frames_refuses_a_negative_floor():
    with pytest.raises(ValueError, match="floor"):
        qc.burn_in_frames(1.0, floor=-1.0, sample_interval=0.1)


def test_burn_in_frames_refuses_a_non_positive_interval():
    with pytest.raises(ValueError, match="sample_interval"):
        qc.burn_in_frames(1.0, floor=0.0, sample_interval=0.0)


# ---------------------------------------------------------------------------
# usable_window
# ---------------------------------------------------------------------------

def test_a_clean_run_keeps_every_frame():
    win = qc.usable_window(500, burn_in=0, stops=(), minimum=1)
    assert (win.start, win.stop, win.n_kept) == (0, 500, 500)
    assert win.flags == ()


def test_a_burn_in_moves_the_start_and_nothing_else():
    win = qc.usable_window(500, burn_in=300, stops=(), minimum=1)
    assert (win.start, win.stop, win.n_kept) == (300, 500, 200)
    assert win.flags == ()


def test_the_earliest_detector_wins():
    win = qc.usable_window(1501, burn_in=0, stops=(373, 372, None), minimum=1)
    assert win.stop == 372
    assert "masked_tail" in win.flags


def test_a_detector_that_found_nothing_does_not_shorten_the_window():
    win = qc.usable_window(100, burn_in=0, stops=(None, None), minimum=1)
    assert win.stop == 100 and win.flags == ()


def test_a_window_below_the_declared_minimum_is_named():
    win = qc.usable_window(500, burn_in=0, stops=(200,), minimum=300)
    assert "short_window" in win.flags
    assert win.n_kept == 200


def test_the_minimum_is_inclusive():
    win = qc.usable_window(500, burn_in=0, stops=(300,), minimum=300)
    assert "short_window" not in win.flags


def test_the_minimum_counts_the_frames_left_after_the_burn_in():
    win = qc.usable_window(500, burn_in=300, stops=(), minimum=300)
    assert win.n_kept == 200 and "short_window" in win.flags


def test_a_burn_in_past_the_end_leaves_nothing_and_is_never_silently_moved_back():
    win = qc.usable_window(500, burn_in=600, stops=(), minimum=1)
    assert win.n_kept == 0
    assert "burn_in_over_run" in win.flags
    assert win.start == 600


def test_a_burn_in_past_a_masked_tail_is_the_same_failure():
    win = qc.usable_window(500, burn_in=400, stops=(300,), minimum=1)
    assert win.n_kept == 0
    assert "burn_in_over_run" in win.flags
    assert "masked_tail" in win.flags


def test_a_timeline_with_no_frames_keeps_none():
    win = qc.usable_window(0, burn_in=0, stops=(), minimum=1)
    assert win.n_kept == 0 and "short_window" in win.flags


def test_a_stop_beyond_the_timeline_cannot_lengthen_it():
    win = qc.usable_window(100, burn_in=0, stops=(400,), minimum=1)
    assert win.stop == 100 and "masked_tail" not in win.flags


def test_usable_window_refuses_a_negative_frame_count():
    with pytest.raises(ValueError, match="n_frames"):
        qc.usable_window(-1, burn_in=0, stops=(), minimum=1)


def test_usable_window_refuses_a_negative_burn_in():
    with pytest.raises(ValueError, match="burn_in"):
        qc.usable_window(10, burn_in=-1, stops=(), minimum=1)


def test_usable_window_refuses_a_negative_stop():
    with pytest.raises(ValueError, match="stops"):
        qc.usable_window(10, burn_in=0, stops=(-1,), minimum=1)


def test_usable_window_refuses_a_negative_minimum():
    with pytest.raises(ValueError, match="minimum"):
        qc.usable_window(10, burn_in=0, stops=(), minimum=-1)


# ---------------------------------------------------------------------------
# displacement_series
# ---------------------------------------------------------------------------

def _frame(step, positions, box=(10.0, 10.0, 10.0), types=None):
    """One :class:`aipf.pipeline.coarse_grain.Frame`, the reader's own type."""
    from aipf.pipeline.coarse_grain import Frame
    positions = np.asarray(positions, dtype=np.float64)
    bounds = np.array([[0.0, box[0]], [0.0, box[1]], [0.0, box[2]]])
    if types is None:
        types = np.ones(len(positions), dtype=np.int64)
    return Frame(timestep=step, box_bounds=bounds, positions=positions,
                 types=np.asarray(types, dtype=np.int64))


def test_a_still_run_has_no_displacement():
    frames = [_frame(i * 10, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
              for i in range(4)]
    out = qc.displacement_series(frames)
    assert out.shape == (3,)
    assert np.array_equal(out, np.zeros(3))


def test_a_uniform_drift_is_measured_per_step():
    frames = [_frame(i * 10, [[i * 0.5, 0.0, 0.0]]) for i in range(5)]
    out = qc.displacement_series(frames)
    assert out == pytest.approx(np.full(4, 0.5))


def test_an_atom_crossing_the_boundary_is_the_short_way_round():
    frames = [_frame(0, [[9.9, 0.0, 0.0]]), _frame(10, [[0.1, 0.0, 0.0]])]
    out = qc.displacement_series(frames)
    assert out == pytest.approx([0.2])


def test_the_displacement_is_averaged_over_the_atoms_of_the_frame():
    frames = [_frame(0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
              _frame(10, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])]
    out = qc.displacement_series(frames)
    assert out == pytest.approx([0.5])


def test_the_displacement_uses_the_later_frame_own_box():
    small = _frame(0, [[0.0, 0.0, 0.0]], box=(10.0, 10.0, 10.0))
    grown = _frame(10, [[6.0, 0.0, 0.0]], box=(10.0, 10.0, 10.0))
    assert qc.displacement_series([small, grown]) == pytest.approx([4.0])
    grown_big = _frame(10, [[6.0, 0.0, 0.0]], box=(100.0, 10.0, 10.0))
    assert qc.displacement_series([small, grown_big]) == pytest.approx([6.0])


def test_a_single_frame_has_no_step_to_measure():
    out = qc.displacement_series([_frame(0, [[0.0, 0.0, 0.0]])])
    assert out.shape == (0,)


def test_displacement_series_refuses_an_empty_timeline():
    with pytest.raises(ValueError, match="frames"):
        qc.displacement_series([])


def test_displacement_series_refuses_a_frame_that_lost_an_atom():
    frames = [_frame(0, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
              _frame(10, [[0.0, 0.0, 0.0]])]
    with pytest.raises(ValueError, match="atoms"):
        qc.displacement_series(frames)


# ---------------------------------------------------------------------------
# frozen_stop
# ---------------------------------------------------------------------------

def test_a_run_that_keeps_moving_is_never_stopped():
    disp = np.full(100, 0.5)
    assert qc.frozen_stop(disp, window=11, threshold=0.05) is None


def test_a_run_that_stops_moving_is_stopped_where_the_median_falls():
    disp = np.concatenate([np.full(50, 0.5), np.zeros(50)])
    stop = qc.frozen_stop(disp, window=11, threshold=0.05)
    # the median of eleven first falls once six of them are still, at step 55,
    # and the frame after that step is the first one no longer usable
    assert stop == 56


def test_a_single_still_step_does_not_stop_a_moving_run():
    disp = np.full(100, 0.5)
    disp[40] = 0.0
    assert qc.frozen_stop(disp, window=11, threshold=0.05) is None


def test_the_window_is_full_before_the_median_is_believed():
    # A ragged trailing median would make the first sample a median of ONE,
    # and a single still first step would freeze the whole run.
    disp = np.concatenate([np.zeros(3), np.full(100, 0.5)])
    assert qc.frozen_stop(disp, window=11, threshold=0.05) is None


def test_a_wider_window_needs_a_longer_stillness():
    disp = np.concatenate([np.full(50, 0.5), np.zeros(5), np.full(50, 0.5)])
    assert qc.frozen_stop(disp, window=11, threshold=0.05) is None
    assert qc.frozen_stop(disp, window=5, threshold=0.05) == 53


def test_a_timeline_too_short_for_the_window_says_nothing():
    assert qc.frozen_stop(np.zeros(5), window=11, threshold=0.05) is None


def test_the_threshold_is_strict_so_a_step_exactly_on_it_is_still_motion():
    disp = np.full(40, 0.05)
    assert qc.frozen_stop(disp, window=11, threshold=0.05) is None


def test_frozen_stop_refuses_an_even_window():
    with pytest.raises(ValueError, match="window"):
        qc.frozen_stop(np.zeros(40), window=10, threshold=0.05)


def test_frozen_stop_refuses_a_non_positive_threshold():
    with pytest.raises(ValueError, match="threshold"):
        qc.frozen_stop(np.zeros(40), window=11, threshold=0.0)


def test_frozen_stop_refuses_a_two_dimensional_series():
    with pytest.raises(ValueError, match="displacements"):
        qc.frozen_stop(np.zeros((4, 4)), window=3, threshold=0.05)


# ---------------------------------------------------------------------------
# excursion_stop
# ---------------------------------------------------------------------------

def _breathing(n=1000, rms=0.001, seed=0):
    rng = np.random.default_rng(seed)
    return 10.0 * (1.0 + rng.normal(0.0, rms, n))


def test_a_run_that_stays_near_its_own_level_is_never_stopped():
    values = _breathing()
    assert qc.excursion_stop(values, head=300, tolerance_floor=0.02,
                             tolerance_sigma=8.0, margin=30) is None


def test_an_excursion_stops_the_run_a_margin_before_it():
    values = _breathing()
    values[500:] *= 1.3
    stop = qc.excursion_stop(values, head=300, tolerance_floor=0.02,
                             tolerance_sigma=8.0, margin=30)
    assert stop == 470


def test_the_margin_never_takes_the_stop_below_the_first_frame():
    values = _breathing()
    values[350:] *= 1.3
    stop = qc.excursion_stop(values, head=300, tolerance_floor=0.02,
                             tolerance_sigma=8.0, margin=400)
    assert stop == 0


def test_the_tolerance_adapts_to_a_run_that_breathes_harder():
    calm = _breathing(rms=0.0005, seed=1)
    wild = _breathing(rms=0.02, seed=1)
    for values in (calm, wild):
        values[600:] *= 1.05
    assert qc.excursion_stop(calm, head=300, tolerance_floor=0.02,
                             tolerance_sigma=8.0, margin=0) == 600
    assert qc.excursion_stop(wild, head=300, tolerance_floor=0.02,
                             tolerance_sigma=8.0, margin=0) is None


def test_the_floor_holds_when_the_run_is_quieter_than_it():
    values = _breathing(rms=1e-9, seed=2)
    values[600:] *= 1.01
    assert qc.excursion_stop(values, head=300, tolerance_floor=0.02,
                             tolerance_sigma=8.0, margin=0) is None
    assert qc.excursion_stop(values, head=300, tolerance_floor=0.005,
                             tolerance_sigma=8.0, margin=0) == 600


def test_a_declared_reference_replaces_the_run_own_level():
    values = np.full(100, 1200.0)
    values[50:] = 2500.0
    assert qc.excursion_stop(values, head=30, tolerance_floor=1.0,
                             tolerance_sigma=0.0, margin=0,
                             reference=1200.0) == 50
    assert qc.excursion_stop(values, head=30, tolerance_floor=1.5,
                             tolerance_sigma=0.0, margin=0,
                             reference=1200.0) is None


def test_the_excursion_test_is_strict_at_the_tolerance():
    # 8.25 against 8.0 is a relative deviation of exactly 0.03125, with no
    # rounding anywhere, so the comparison at the boundary is the one thing
    # this measures.
    values = np.full(100, 8.0)
    values[50:] = 8.25
    assert qc.excursion_stop(values, head=30, tolerance_floor=0.03125,
                             tolerance_sigma=0.0, margin=0) is None
    assert qc.excursion_stop(values, head=30, tolerance_floor=0.03124,
                             tolerance_sigma=0.0, margin=0) == 50


def test_a_head_longer_than_the_run_uses_what_there_is():
    values = _breathing(n=50)
    assert qc.excursion_stop(values, head=300, tolerance_floor=0.02,
                             tolerance_sigma=8.0, margin=0) is None


def test_excursion_stop_refuses_a_head_below_two():
    with pytest.raises(ValueError, match="head"):
        qc.excursion_stop(np.ones(10), head=1, tolerance_floor=0.02,
                          tolerance_sigma=8.0, margin=0)


def test_excursion_stop_refuses_a_negative_margin():
    with pytest.raises(ValueError, match="margin"):
        qc.excursion_stop(np.ones(10), head=5, tolerance_floor=0.02,
                          tolerance_sigma=8.0, margin=-1)


def test_excursion_stop_refuses_a_negative_tolerance_floor():
    with pytest.raises(ValueError, match="tolerance_floor"):
        qc.excursion_stop(np.ones(10), head=5, tolerance_floor=-0.01,
                          tolerance_sigma=8.0, margin=0)


def test_excursion_stop_refuses_a_negative_scatter_multiple():
    with pytest.raises(ValueError, match="tolerance_sigma"):
        qc.excursion_stop(np.ones(10), head=5, tolerance_floor=0.02,
                          tolerance_sigma=-1.0, margin=0)


def test_excursion_stop_refuses_a_reference_that_is_not_positive():
    with pytest.raises(ValueError, match="reference"):
        qc.excursion_stop(np.ones(10), head=5, tolerance_floor=0.02,
                          tolerance_sigma=0.0, margin=0, reference=0.0)


def test_excursion_stop_refuses_a_run_whose_own_level_is_not_positive():
    values = np.zeros(10)
    with pytest.raises(ValueError, match="reference"):
        qc.excursion_stop(values, head=5, tolerance_floor=0.02,
                          tolerance_sigma=8.0, margin=0)


def test_excursion_stop_refuses_a_two_dimensional_series():
    with pytest.raises(ValueError, match="values"):
        qc.excursion_stop(np.ones((4, 4)), head=2, tolerance_floor=0.02,
                          tolerance_sigma=8.0, margin=0)


# ---------------------------------------------------------------------------
# uniformity_deviation
# ---------------------------------------------------------------------------

def test_a_flat_field_deviates_from_its_mean_by_nothing():
    field = np.full((1, 4, 4, 4), 0.5)
    assert qc.uniformity_deviation(field, contrast=(1.0,)) == 0.0


def test_the_deviation_is_relative_to_the_mean():
    field = np.ones((1, 2, 2, 2))
    field[0, 0, 0, 0] = 1.8
    psi_mean = field.mean()
    assert qc.uniformity_deviation(field, contrast=(1.0,)) == pytest.approx(
        (1.8 - psi_mean) / psi_mean)


def test_two_channels_are_combined_by_the_declared_weights():
    field = np.stack([np.full((2, 2, 2), 0.3), np.full((2, 2, 2), 0.5)])
    field[0, 0, 0, 0] = 0.4
    total = qc.uniformity_deviation(field, contrast=(1.0, 1.0))
    only_first = qc.uniformity_deviation(field, contrast=(1.0, 0.0))
    assert total > 0.0 and only_first > total


def test_the_largest_deviation_wins_whichever_side_it_is_on():
    low = np.ones((1, 8))
    low[0, 3] = 0.2
    high = np.ones((1, 8))
    high[0, 3] = 1.8
    assert qc.uniformity_deviation(low, contrast=(1.0,)) == pytest.approx(
        0.7 / 0.9)
    assert qc.uniformity_deviation(high, contrast=(1.0,)) == pytest.approx(
        0.7 / 1.1)


def test_uniformity_refuses_a_combination_with_no_scale_to_be_relative_to():
    field = np.stack([np.ones((2, 2)), np.ones((2, 2))])
    with pytest.raises(ValueError, match="mean"):
        qc.uniformity_deviation(field, contrast=(1.0, -1.0))


def test_uniformity_refuses_a_weight_per_channel_mismatch():
    field = np.ones((2, 4, 4))
    with pytest.raises(ValueError, match="contrast"):
        qc.uniformity_deviation(field, contrast=(1.0,))


def test_uniformity_refuses_a_field_with_no_channel_axis():
    with pytest.raises(ValueError, match="field"):
        qc.uniformity_deviation(np.ones(4), contrast=(1.0,))


def test_uniformity_refuses_a_field_that_is_not_finite():
    field = np.ones((1, 4))
    field[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        qc.uniformity_deviation(field, contrast=(1.0,))


def test_uniformity_is_computed_in_double_whatever_the_field_carries():
    rng = np.random.default_rng(0)
    field = (1.0 + 1e-3 * rng.normal(size=(1, 64, 64))).astype(np.float32)
    single = qc.uniformity_deviation(field, contrast=(1.0,))
    double = qc.uniformity_deviation(field.astype(np.float64), contrast=(1.0,))
    assert single == double


# ---------------------------------------------------------------------------
# mode_power_ratio
# ---------------------------------------------------------------------------

def _modes(n_frames=8, n_channels=2, n_modes=5, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n_frames, n_channels, n_modes))
            + 1j * rng.normal(size=(n_frames, n_channels, n_modes)))


def test_the_ratio_of_a_combination_to_itself_is_one():
    rho_k = _modes()
    assert qc.mode_power_ratio(rho_k, numerator=(1.0, -1.0),
                               denominator=(1.0, -1.0)) == 1.0


def test_one_channel_is_an_ordinary_case_for_the_ratio():
    rho_k = _modes(n_channels=1)
    assert qc.mode_power_ratio(rho_k, numerator=(1.0,),
                               denominator=(1.0,)) == 1.0
    assert qc.mode_power_ratio(rho_k, numerator=(2.0,),
                               denominator=(1.0,)) == pytest.approx(4.0)


def test_the_ratio_is_a_mode_mean_of_time_means_not_a_time_mean_of_ratios():
    rho_k = np.zeros((2, 1, 2), dtype=np.complex128)
    rho_k[0, 0, 0] = 2.0        # mode 0: powers 4 and 0
    rho_k[1, 0, 1] = 4.0        # mode 1: powers 0 and 16
    num = qc.mode_power_ratio(rho_k, numerator=(1.0,), denominator=(0.5,))
    assert num == pytest.approx(4.0)


def test_a_contrast_scaling_cancels_exactly():
    rho_k = _modes()
    half = qc.mode_power_ratio(rho_k, numerator=(1.0, 1.0),
                               denominator=(0.5, -0.5)) / 4.0
    whole = qc.mode_power_ratio(rho_k, numerator=(1.0, 1.0),
                                denominator=(1.0, -1.0))
    assert half == whole


def test_a_mode_with_no_power_in_the_denominator_is_refused():
    rho_k = _modes()
    rho_k[:, :, 2] = 0.0
    with pytest.raises(ValueError, match="denominator"):
        qc.mode_power_ratio(rho_k, numerator=(1.0, 1.0),
                            denominator=(1.0, -1.0))


def test_mode_power_ratio_refuses_a_weight_per_channel_mismatch():
    rho_k = _modes()
    with pytest.raises(ValueError, match="numerator"):
        qc.mode_power_ratio(rho_k, numerator=(1.0,), denominator=(1.0, -1.0))
    with pytest.raises(ValueError, match="denominator"):
        qc.mode_power_ratio(rho_k, numerator=(1.0, 1.0), denominator=(1.0,))


def test_mode_power_ratio_refuses_an_array_that_is_not_a_mode_timeline():
    with pytest.raises(ValueError, match="rho_k"):
        qc.mode_power_ratio(np.zeros((2, 2), dtype=np.complex128),
                            numerator=(1.0,), denominator=(1.0,))


def test_mode_power_ratio_refuses_a_timeline_that_is_not_finite():
    rho_k = _modes()
    rho_k[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        qc.mode_power_ratio(rho_k, numerator=(1.0, 1.0),
                            denominator=(1.0, -1.0))


def test_mode_power_ratio_accumulates_in_double_whatever_the_store_carries():
    rho_k = _modes(n_frames=64, n_modes=64)
    single = qc.mode_power_ratio(rho_k.astype(np.complex64),
                                 numerator=(1.0, 1.0), denominator=(1.0, -1.0))
    double = qc.mode_power_ratio(rho_k.astype(np.complex64).astype(np.complex128),
                                 numerator=(1.0, 1.0), denominator=(1.0, -1.0))
    assert single == double


def test_a_real_mode_timeline_is_refused_because_an_amplitude_has_a_phase():
    with pytest.raises(ValueError, match="complex"):
        qc.mode_power_ratio(np.ones((2, 1, 2)), numerator=(1.0,),
                            denominator=(1.0,))


# ---------------------------------------------------------------------------
# band_growth
# ---------------------------------------------------------------------------

def test_the_partitions_are_the_two_the_trees_use():
    assert qc.PARTITIONS == ("fixed", "spanning")


def test_a_flat_band_does_not_grow():
    out = qc.band_growth(np.ones(100), n_blocks=5, partition="fixed",
                         n_independent=40)
    assert out.gain == 1.0
    assert out.z_floor == 0.0


def test_a_growing_band_reports_the_ratio_of_its_last_block_to_its_first():
    series = np.concatenate([np.full(20, 1.0), np.full(20, 1.0),
                             np.full(20, 1.0), np.full(20, 1.0),
                             np.full(20, 2.0)])
    out = qc.band_growth(series, n_blocks=5, partition="fixed",
                         n_independent=40)
    assert out.gain == pytest.approx(2.0)
    assert out.block_means == pytest.approx([1.0, 1.0, 1.0, 1.0, 2.0])


def test_the_fixed_partition_drops_the_remainder_at_the_end():
    series = np.arange(12, dtype=float)
    out = qc.band_growth(series, n_blocks=5, partition="fixed",
                         n_independent=4)
    # blocks of two, samples 10 and 11 dropped
    assert out.block_means == pytest.approx([0.5, 2.5, 4.5, 6.5, 8.5])


def test_the_spanning_partition_uses_every_sample():
    series = np.arange(12, dtype=float)
    out = qc.band_growth(series, n_blocks=5, partition="spanning",
                         n_independent=4)
    assert out.block_means == pytest.approx([0.5, 2.5, 5.0, 7.5, 10.0])


def test_the_two_partitions_agree_when_the_blocks_divide_evenly():
    series = np.arange(20, dtype=float)
    fixed = qc.band_growth(series, n_blocks=5, partition="fixed",
                           n_independent=4)
    spanning = qc.band_growth(series, n_blocks=5, partition="spanning",
                              n_independent=4)
    assert fixed.block_means == pytest.approx(spanning.block_means)


def test_the_error_bar_no_averaging_can_shrink_is_the_mode_count_alone():
    out = qc.band_growth(np.ones(100), n_blocks=5, partition="fixed",
                         n_independent=40)
    assert out.sigma_floor == pytest.approx(math.sqrt(2.0 / 40.0))


def test_the_floor_z_measures_the_excess_over_one_in_those_units():
    series = np.concatenate([np.full(50, 1.0), np.full(50, 1.5)])
    out = qc.band_growth(series, n_blocks=2, partition="fixed",
                         n_independent=8)
    assert out.gain == pytest.approx(1.5)
    assert out.z_floor == pytest.approx(0.5 / math.sqrt(2.0 / 8.0))


def test_an_unknown_partition_is_refused_and_the_known_ones_named():
    with pytest.raises(ValueError, match="fixed"):
        qc.band_growth(np.ones(20), n_blocks=5, partition="thirds",
                       n_independent=4)


def test_a_band_whose_first_block_is_not_positive_has_no_gain():
    series = np.concatenate([np.zeros(10), np.ones(40)])
    with pytest.raises(ValueError, match="first block"):
        qc.band_growth(series, n_blocks=5, partition="fixed", n_independent=4)


def test_band_growth_refuses_fewer_samples_than_blocks():
    with pytest.raises(ValueError, match="n_blocks"):
        qc.band_growth(np.ones(4), n_blocks=5, partition="fixed",
                       n_independent=4)


def test_band_growth_refuses_a_block_count_below_two():
    with pytest.raises(ValueError, match="n_blocks"):
        qc.band_growth(np.ones(40), n_blocks=1, partition="fixed",
                       n_independent=4)


def test_band_growth_refuses_a_non_positive_independent_count():
    with pytest.raises(ValueError, match="n_independent"):
        qc.band_growth(np.ones(40), n_blocks=5, partition="fixed",
                       n_independent=0)


def test_band_growth_refuses_a_series_that_is_not_finite():
    series = np.ones(40)
    series[7] = np.inf
    with pytest.raises(ValueError, match="finite"):
        qc.band_growth(series, n_blocks=5, partition="fixed", n_independent=4)


def test_band_growth_refuses_a_two_dimensional_series():
    with pytest.raises(ValueError, match="series"):
        qc.band_growth(np.ones((10, 2)), n_blocks=5, partition="fixed",
                       n_independent=4)


# ---------------------------------------------------------------------------
# the package exports and the whole shape
# ---------------------------------------------------------------------------

def test_every_public_name_is_exported_from_the_package():
    import aipf.pipeline as pipeline
    for name in ("FLAGS", "PARTITIONS", "VERDICTS", "band_growth",
                 "block_means", "burn_in_frames", "channel_occupancy",
                 "displacement_series", "excursion_stop", "expected_frames",
                 "frame_inventory", "frozen_stop", "mode_power_ratio",
                 "observable_report", "settle_time", "uniformity_deviation",
                 "usable_window", "verdict"):
        assert getattr(pipeline, name) is getattr(qc, name)


def test_the_module_names_no_species_and_holds_no_absolute_path():
    import inspect
    import re
    source = inspect.getsource(qc)
    for word in ("hydrogen", "helium", "iron", "boron", "argon"):
        assert word not in source.lower(), word
    # The two site roots, as a pattern rather than as path literals.
    assert not re.search(r"/(home|scratch)/", source.lower())


def test_a_whole_run_composes_into_one_verdict_at_one_channel():
    steps = _steps(500, every=100)
    inv = qc.frame_inventory(steps, expected=500, tolerated_fraction=0.95,
                             dump_every_steps=100)
    occ = qc.channel_occupancy(np.ones(64, dtype=np.int64), atom_types=(1,))
    win = qc.usable_window(inv.n_frames, burn_in=50, stops=(), minimum=100)
    flags = inv.flags + occ.flags + win.flags
    assert qc.verdict(flags, fatal=("no_frames", "short_window")) == "clean"


def test_a_whole_run_composes_into_one_verdict_at_two_channels():
    steps = np.array([0, 100, 300], dtype=np.int64)
    inv = qc.frame_inventory(steps, expected=500, tolerated_fraction=0.95,
                             dump_every_steps=100)
    occ = qc.channel_occupancy(np.ones(8, dtype=np.int64), atom_types=(1, 2))
    win = qc.usable_window(inv.n_frames, burn_in=0, stops=(2,), minimum=100)
    flags = inv.flags + occ.flags + win.flags
    assert set(flags) == {"uneven_cadence", "short_of_expected",
                          "empty_channel", "masked_tail", "short_window"}
    assert qc.verdict(flags, fatal=("empty_channel",)) == "unusable"
    assert qc.verdict(flags, fatal=()) == "flagged"


# ---------------------------------------------------------------------------
# the gaps the mutation sweep found
# ---------------------------------------------------------------------------

def test_a_tolerated_fraction_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match="tolerated_fraction"):
        qc.frame_inventory(_steps(5), expected=5, tolerated_fraction=1.5)


def test_a_minority_share_outside_zero_to_one_is_refused():
    types = np.array([1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="minority_fraction"):
        qc.channel_occupancy(types, atom_types=(1, 2), minority_fraction=1.5,
                             minority_atoms=400)


def test_a_negative_minority_atom_count_is_refused():
    types = np.array([1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="minority_atoms"):
        qc.channel_occupancy(types, atom_types=(1, 2), minority_fraction=0.06,
                             minority_atoms=-1)


def test_the_scatter_of_a_series_is_the_sample_one():
    rep = qc.observable_report(np.array([1.0, 2.0, 3.0, 4.0]))
    assert rep.mean == 2.5
    assert rep.std == pytest.approx(math.sqrt(5.0 / 3.0))


def test_a_drift_exactly_at_its_limit_is_not_drifting():
    # halves at 1 and 2: the difference is 1 against a mean of 1.5
    values = np.array([1.0, 1.0, 2.0, 2.0])
    rep = qc.observable_report(values, drift_max=1.0 / 1.5)
    assert rep.drift == pytest.approx(1.0 / 1.5)
    assert rep.flags == ()


def test_the_band_is_built_from_the_tail_mean_and_its_sample_scatter():
    # ten one-sample blocks; the tail is 1, 1, 1, 1, 6, whose mean is 2 and
    # whose MEDIAN is 1, and whose sample scatter is sqrt(5) against a
    # population scatter of 2
    times = np.arange(10, dtype=np.float64)
    values = np.array([2.0] * 5 + [1.0, 1.0, 1.0, 1.0, 6.0])
    out = qc.settle_time(times, values, block_span=1.0, tol_sigma=3.0,
                         persist=2, min_blocks=6)
    assert out.n_blocks == 10
    assert out.reference == 2.0
    assert out.scatter == pytest.approx(math.sqrt(5.0))


def test_a_block_exactly_on_the_band_edge_is_inside_it():
    # nine one-sample blocks; the tail 3, 3, 1, 1, 2 has mean 2 and sample
    # scatter exactly 1, both to the last bit, so a first block at 1 sits
    # exactly one band width away and one at 0.9 sits outside
    times = np.arange(9, dtype=np.float64)
    tail = [3.0, 3.0, 1.0, 1.0, 2.0]
    edge = qc.settle_time(times, np.array([1.0, 2.0, 2.0, 2.0] + tail),
                          block_span=1.0, tol_sigma=1.0, persist=2,
                          min_blocks=6)
    assert edge.reference == 2.0 and edge.scatter == 1.0
    assert edge.settle == 0.0
    over = qc.settle_time(times, np.array([0.9, 2.0, 2.0, 2.0] + tail),
                          block_span=1.0, tol_sigma=1.0, persist=2,
                          min_blocks=6)
    assert over.settle == pytest.approx(1.0)


def test_a_settle_time_is_measured_from_the_first_sample():
    times, values = _settling(transient=3)
    shifted = qc.settle_time(times + 50.0, values, block_span=1.0,
                             tol_sigma=3.0, persist=2, min_blocks=6)
    assert shifted.settle == pytest.approx(3.0)


def test_a_burn_in_that_exactly_consumes_the_window_has_not_over_run_it():
    win = qc.usable_window(500, burn_in=500, stops=(), minimum=0)
    assert win.n_kept == 0
    assert win.flags == ()


def test_the_displacement_is_measured_in_a_box_that_need_not_start_at_zero():
    frames = [_frame(0, [[19.9, 0.0, 0.0]]), _frame(10, [[10.1, 0.0, 0.0]])]
    for frame in frames:
        frame.box_bounds[0] = [10.0, 20.0]
    assert qc.displacement_series(frames) == pytest.approx([0.2])


def test_two_atoms_moving_apart_have_not_moved_nowhere():
    frames = [_frame(0, [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
              _frame(10, [[6.0, 0.0, 0.0], [4.0, 0.0, 0.0]])]
    assert qc.displacement_series(frames) == pytest.approx([1.0])


def test_the_level_is_the_head_median_so_one_spike_does_not_move_it():
    values = np.full(200, 10.0)
    values[7] = 1000.0
    values[100:] = 10.5
    assert qc.excursion_stop(values, head=50, tolerance_floor=0.02,
                             tolerance_sigma=0.0, margin=0) == 7


def test_the_scatter_is_relative_so_a_large_level_is_not_a_large_tolerance():
    rng = np.random.default_rng(3)
    values = 1000.0 * (1.0 + rng.normal(0.0, 0.001, 400))
    values[300:] *= 1.02
    assert qc.excursion_stop(values, head=200, tolerance_floor=0.001,
                             tolerance_sigma=8.0, margin=0) == 300


def test_the_ratio_averages_over_modes_that_do_not_share_a_ratio():
    rho_k = np.zeros((1, 1, 2), dtype=np.complex128)
    rho_k[0, 0, 0] = 1.0        # power 1
    rho_k[0, 0, 1] = 3.0        # power 9
    # numerator (1,) gives powers 1 and 9; denominator (1/3,) gives 1/9 and 1
    ratio = qc.mode_power_ratio(rho_k, numerator=(1.0,),
                                denominator=(1.0 / 3.0,))
    assert ratio == pytest.approx((9.0 + 9.0) / 2.0)
    # a denominator that scales the two modes differently is what separates
    # a mean of ratios from a ratio of means
    rho_k2 = np.zeros((1, 2, 2), dtype=np.complex128)
    rho_k2[0, 0, 0] = 1.0
    rho_k2[0, 1, 0] = 1.0
    rho_k2[0, 0, 1] = 3.0
    rho_k2[0, 1, 1] = math.sqrt(3.0)
    out = qc.mode_power_ratio(rho_k2, numerator=(1.0, 0.0),
                              denominator=(0.0, 1.0))
    assert out == pytest.approx((1.0 + 3.0) / 2.0)


# ---------------------------------------------------------------------------
# the gaps the second round of the sweep found
# ---------------------------------------------------------------------------

def test_an_empty_timeline_of_any_type_is_still_just_empty():
    # an empty list of steps arrives as float64, and a run that wrote nothing
    # is a run with no frames rather than a timeline of the wrong type
    inv = qc.frame_inventory(np.asarray([]))
    assert inv.n_frames == 0 and "no_frames" in inv.flags


def test_a_channel_holding_one_atom_is_not_an_empty_one():
    types = np.array([1, 2, 2, 2], dtype=np.int64)
    occ = qc.channel_occupancy(types, atom_types=(1, 2))
    assert occ.counts == (1, 3)
    assert occ.empty == () and occ.flags == ()


def test_a_single_sample_reports_no_scatter_and_does_not_complain():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rep = qc.observable_report(np.array([3.0]))
    assert math.isnan(rep.std)


def test_the_usual_block_size_is_taken_over_the_blocks_that_have_samples():
    # two whole blocks with nothing in them, between two blocks of three
    times = np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2])
    ends, means, per_block = qc.block_means(times, np.ones(6), block_span=1.0)
    assert per_block == 3
    assert len(means) == 2


def test_nothing_outside_the_band_is_a_strict_value_of_zero_as_well():
    times = np.arange(9, dtype=np.float64)
    values = np.array([2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0])
    out = qc.settle_time(times, values, block_span=1.0, tol_sigma=1.0,
                         persist=2, min_blocks=6)
    assert out.settle == 0.0 and out.strict == 0.0


def test_a_burn_in_is_rounded_rather_than_truncated():
    # two units over an interval of 0.3 is 6.67 frames: truncating would hand
    # two thirds of a frame of preparation back to the data
    assert qc.burn_in_frames(None, floor=2.0, sample_interval=0.3) == 7


def test_uniformity_weighs_the_channels_in_double_precision():
    rng = np.random.default_rng(0)
    field = (1.0 + 1e-3 * rng.normal(size=(2, 64, 64))).astype(np.float32)
    weights = (0.1, 0.9)          # neither is exact in single precision
    assert (qc.uniformity_deviation(field, contrast=weights)
            == qc.uniformity_deviation(field.astype(np.float64),
                                       contrast=weights))


def test_the_power_ratio_weighs_the_channels_in_double_precision():
    rho_k = _modes(n_frames=64, n_modes=64).astype(np.complex64)
    num, den = (0.1, 0.9), (0.3, -0.7)
    assert (qc.mode_power_ratio(rho_k, numerator=num, denominator=den)
            == qc.mode_power_ratio(rho_k.astype(np.complex128),
                                   numerator=num, denominator=den))
