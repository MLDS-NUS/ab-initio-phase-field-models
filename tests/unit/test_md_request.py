"""The state-point vocabulary, and the evidence that it spans what ran.

The table in ``ARCHIVED`` is the point of this file. Each row is a distinct
campaign shape found by reading the archived manifest generators, the archived
manifests they wrote, and the job files that submitted them. Each row names the
file it came from. A row asserts that the shape survives a round trip through
``StatePoint`` unchanged, and where a frame count was measured on disk it
asserts that too.

Times are given in the unit each archive works in, and step counts are
converted to times here rather than in the package, because a request that
counted steps would be a request that could not be reproduced at a different
timestep.
"""
import dataclasses

import pytest

from aipf.data.meta import SCHEMA_VERSION, validate
from aipf.md.request import (
    DUMP_FROM,
    StatePoint,
    campaign_from_records,
    campaign_to_records,
    normalise_ensemble,
    normalise_geometry,
)


def _point(**kwargs) -> StatePoint:
    """A valid request, overridable one field at a time."""
    base = dict(geometry="cube", ensemble="NPT", T=7000.0, x=0.5,
                dt_ps=2e-4, equil_ps=1.5, prod_ps=30.0,
                dump_every_ps=0.02, dump_from="prod", seed=1, P=800.0,
                n_atoms=3456)
    base.update(kwargs)
    return StatePoint(**base)


# --------------------------------------------------------------------------
# The archive. Each entry: a label naming the campaign it describes, the request, and
# the frame count measured on disk where one was measured.
# --------------------------------------------------------------------------

ARCHIVED: list[tuple[str, StatePoint, int | None]] = [
    # -- first heavy tree, its production campaign of cubes and slabs
    ("A1 uniform cube, barostat on all axes: n_equil 7500 and n_prod 150000 "
     "steps at dt 2e-4 ps, dumped every 100 steps",
     StatePoint(geometry="cube", ensemble="npt_iso", T=7000.0, x=0.5,
                dt_ps=2e-4, equil_ps=7500 * 2e-4, prod_ps=150000 * 2e-4,
                dump_every_ps=100 * 2e-4, dump_from="prod", seed=1,
                P=800.0, n_atoms=3456),
     1501),
    ("A2 two-sided slab, barostat on one axis: n_equil 5000 and n_prod 400000 "
     "steps at dt 1e-4 ps, dumped every 200 steps, atom count an outcome",
     StatePoint(geometry="slab", ensemble="npt_z", T=2000.0, x=(0.0, 1.0),
                dt_ps=1e-4, equil_ps=5000 * 1e-4, prod_ps=400000 * 1e-4,
                dump_every_ps=200 * 1e-4, dump_from="prod", seed=1,
                P=800.0),
     2001),
    # -- first heavy tree, the per-pressure anchor campaign
    ("A3 the same cube on another pressure manifold",
     StatePoint(geometry="cube", ensemble="npt_iso", T=5500.0, x=0.05,
                dt_ps=2e-4, equil_ps=7500 * 2e-4, prod_ps=150000 * 2e-4,
                dump_every_ps=100 * 2e-4, dump_from="prod", seed=1,
                P=200.0, n_atoms=3456),
     None),
    # -- first heavy tree, the per-pressure, per-seed slab campaign that dumps its equilibration
    ("A4 a cube that keeps its equilibration frames on purpose, because the "
     "early growth window lives in them",
     StatePoint(geometry="cube", ensemble="npt_iso", T=4000.0, x=0.3,
                dt_ps=2e-4, equil_ps=7500 * 2e-4, prod_ps=150000 * 2e-4,
                dump_every_ps=100 * 2e-4, dump_from="equil", seed=1,
                P=400.0, n_atoms=3456),
     None),
    # -- first heavy tree, the He-rich corner campaign
    ("A5 a corner-filling cube, identical in shape and carrying only a "
     "campaign label this vocabulary does not keep",
     StatePoint(geometry="cube", ensemble="npt_iso", T=5000.0, x=0.9,
                dt_ps=2e-4, equil_ps=7500 * 2e-4, prod_ps=150000 * 2e-4,
                dump_every_ps=100 * 2e-4, dump_from="prod", seed=1,
                P=800.0, n_atoms=3456),
     None),
    # -- first heavy tree, a batch job that dumps every single step, with a seed of its own
    ("A6 the dense-dump one-off: cadence equal to the timestep",
     StatePoint(geometry="cube", ensemble="npt_iso", T=7000.0, x=0.5,
                dt_ps=2e-4, equil_ps=7500 * 2e-4, prod_ps=250000 * 2e-4,
                dump_every_ps=2e-4, dump_from="prod", seed=11,
                P=800.0, n_atoms=3456),
     None),
    # -- first heavy tree, the equation-of-state campaign and its point deck: no trajectory
    ("A7 an equation-of-state point: 5000 equilibration and 5000 averaging "
     "steps at dt 2e-4 ps, no trajectory written at all",
     StatePoint(geometry="eos", ensemble="npt_iso", T=2000.0, x=0.1,
                dt_ps=2e-4, equil_ps=5000 * 2e-4, prod_ps=5000 * 2e-4,
                dump_every_ps=None, dump_from="none", seed=42,
                P=800.0, n_atoms=512),
     None),
    # -- first heavy tree, the S(0) measurement runs: a third dump cadence, 250 steps
    ("A8 the structure-factor spot check: 50000 equilibration and 200000 "
     "production steps, dumped every 250",
     StatePoint(geometry="cube", ensemble="npt_iso", T=7000.0, x=0.08,
                dt_ps=2e-4, equil_ps=50000 * 2e-4, prod_ps=200000 * 2e-4,
                dump_every_ps=250 * 2e-4, dump_from="prod", seed=42,
                P=800.0, n_atoms=3456),
     None),
    # -- first heavy tree, a short pilot at a new pressure
    ("A9 a pressure pilot: 5000 production steps only",
     StatePoint(geometry="cube", ensemble="npt_iso", T=9500.0, x=0.95,
                dt_ps=2e-4, equil_ps=7500 * 2e-4, prod_ps=5000 * 2e-4,
                dump_every_ps=100 * 2e-4, dump_from="prod", seed=1,
                P=400.0, n_atoms=3456),
     None),
    # -- second heavy tree, the first pass of its cube campaign, one manifest per shard
    ("B1 the grid cube: 10 ps melt and 50 ps production at dt 1e-3 ps, "
     "dumped every 0.1 ps, at zero pressure",
     StatePoint(geometry="cube", ensemble="npt_iso", T=1200.0, x=0.1,
                dt_ps=1e-3, equil_ps=10.0, prod_ps=50.0,
                dump_every_ps=0.1, dump_from="prod", seed=1,
                P=0.0, n_atoms=3456),
     501),
    ("B2 the timestep pilot: the same physical times at dt 2e-3 ps",
     StatePoint(geometry="cube", ensemble="npt_iso", T=1800.0, x=0.8,
                dt_ps=2e-3, equil_ps=10.0, prod_ps=50.0,
                dump_every_ps=0.1, dump_from="prod", seed=1,
                P=0.0, n_atoms=3456),
     None),
    ("B3 a cold row on a pressurised manifold, given twice the exposure",
     StatePoint(geometry="cube", ensemble="npt_iso", T=1400.0, x=0.5,
                dt_ps=1e-3, equil_ps=10.0, prod_ps=100.0,
                dump_every_ps=0.1, dump_from="prod", seed=1,
                P=5.0, n_atoms=3456),
     None),
    # -- second heavy tree, the second pass of its cube campaign
    ("B4 a continuation: no melt, resumed from a numbered restart of its "
     "parent",
     StatePoint(geometry="cube", ensemble="npt_iso", T=1500.0, x=0.7,
                dt_ps=1e-3, equil_ps=0.0, prod_ps=150.0,
                dump_every_ps=0.1, dump_from="prod", seed=1,
                P=0.0, n_atoms=3456,
                parent="cube_x0.70_T1500_P0_s1"),
     None),
    ("B5 a fresh replica seed, run to the full extended length",
     StatePoint(geometry="cube", ensemble="npt_iso", T=1500.0, x=0.7,
                dt_ps=1e-3, equil_ps=10.0, prod_ps=200.0,
                dump_every_ps=0.1, dump_from="prod", seed=2,
                P=0.0, n_atoms=3456),
     None),
    ("B6 an early row at the denser cadence the campaign started with, "
     "before it was aligned to the other tree by step count",
     StatePoint(geometry="cube", ensemble="npt_iso", T=2600.0, x=0.95,
                dt_ps=1e-3, equil_ps=10.0, prod_ps=50.0,
                dump_every_ps=0.02, dump_from="prod", seed=1,
                P=0.0, n_atoms=3456),
     None),
    # -- second heavy tree, the equation-of-state worker on a new grid
    ("B7 an equation-of-state point on the second tree's pressure grid",
     StatePoint(geometry="eos", ensemble="npt_iso", T=1900.0, x=0.0,
                dt_ps=2e-4, equil_ps=1.0, prod_ps=1.0,
                dump_every_ps=None, dump_from="none", seed=1,
                P=10.0, n_atoms=512),
     None),
    # -- third tree, configs/md_pipeline.yaml + scripts/01_md_pipeline.py
    ("C1 the melt-first snapshot generator: an inertial thermostat at "
     "dt 5e-3, 20000 melt and 1e7 production steps, dumped every 10000; "
     "no pressure is controlled anywhere in this tree",
     StatePoint(geometry="slab_meltfirst", ensemble="langevin", T=1.0,
                x=0.5, dt_ps=5e-3, equil_ps=20000 * 5e-3,
                prod_ps=10000000 * 5e-3, dump_every_ps=10000 * 5e-3,
                dump_from="prod", seed=42, n_atoms=3456),
     None),
    ("C2 the long overdamped leg, started from a frame of the run above and "
     "therefore never re-equilibrated: 1e8 steps at dt 2e-4, dumped every "
     "5000",
     StatePoint(geometry="slab_overdamped", ensemble="brownian", T=1.2,
                x=0.5, dt_ps=2e-4, equil_ps=0.0, prod_ps=100000000 * 2e-4,
                dump_every_ps=5000 * 2e-4, dump_from="prod", seed=43,
                n_atoms=3456, parent="slab_x0.50_T1_s42"),
     20001),
    ("C4 the uniform bulk reference above the critical point, fixed box, "
     "overdamped: 2.5e7 steps at dt 2e-4",
     StatePoint(geometry="homogeneous_brownian", ensemble="brownian",
                T=1.5, x=0.5, dt_ps=2e-4, equil_ps=250000 * 1e-3,
                prod_ps=25000000 * 2e-4, dump_every_ps=5000 * 2e-4,
                dump_from="prod", seed=1, n_atoms=4000),
     5001),
    ("C5 the in-dome uniform cube that feeds the dynamical term, on its own "
     "tree and its own seed stream",
     StatePoint(geometry="spinodal_cube", ensemble="brownian", T=1.1,
                x=0.5, dt_ps=2e-4, equil_ps=200000 * 1e-3,
                prod_ps=50000000 * 2e-4, dump_every_ps=5000 * 2e-4,
                dump_from="prod", seed=7000000, n_atoms=4000),
     10001),
    ("C6 an external-potential equilibrium run: an inertial thermostat at "
     "dt 5e-3, 500000 equilibration and 1500000 production steps, dumped "
     "every 20000",
     StatePoint(geometry="vext", ensemble="langevin", T=1.2, x=0.5,
                dt_ps=5e-3, equil_ps=500000 * 5e-3, prod_ps=1500000 * 5e-3,
                dump_every_ps=20000 * 5e-3, dump_from="prod", seed=12345,
                n_atoms=55296),
     None),
]

ARCHIVED_IDS = [row[0].split(" ", 1)[0] for row in ARCHIVED]


@pytest.mark.parametrize("label,point,n_frames",
                         ARCHIVED, ids=ARCHIVED_IDS)
def test_archived_campaign_shape_round_trips(label, point, n_frames):
    """Every distinct archived campaign shape survives a round trip."""
    assert StatePoint.from_record(point.to_record()) == point, label


@pytest.mark.parametrize("label,point,n_frames",
                         [row for row in ARCHIVED if row[2] is not None],
                         ids=[row[0].split(" ", 1)[0]
                              for row in ARCHIVED if row[2] is not None])
def test_archived_frame_count_is_reproduced(label, point, n_frames):
    """Where a trajectory was counted on disk, the request predicts it."""
    assert point.n_frames == n_frames, label


def test_the_whole_archive_is_one_campaign_with_distinct_tags():
    """Every shape above is a separate request, so the list is a campaign."""
    campaign = [row[1] for row in ARCHIVED]
    records = campaign_to_records(campaign)
    assert len(records) == len(ARCHIVED)
    assert campaign_from_records(records) == campaign


def test_every_archived_shape_yields_admissible_metadata():
    """A request's share of a metadata record passes the record's own check.

    The keys a request cannot know are supplied here, which is exactly the
    split the module claims: everything below comes from the run or the
    system, and nothing above does.
    """
    for label, point, _ in ARCHIVED:
        record = {
            "schema_version": SCHEMA_VERSION,
            "system": "archived",
            "composition": {"species": ["a", "b"], "x": {"b": 0.5}},
            "box": {"L": [None, None, None],
                    "varying": ["z"] if point.ensemble == "NPT_z"
                    else (["x", "y", "z"] if point.ensemble == "NPT" else [])},
            "engine": "archived",
            "potential": None,
            "status": "ok",
        }
        record.update(point.to_meta_fields())
        # A run with no trajectory has no frames, which the record's own
        # positivity rule rejects; that pairing is asserted separately.
        if record["n_frames"] == 0:
            record["n_frames"] = 1
        if record["dump_every_ps"] is None:
            record["dump_every_ps"] = point.dt_ps
        assert validate(record) == [], f"{label}: {validate(record)}"


# --------------------------------------------------------------------------
# Shapes the archive holds that this vocabulary does NOT express. Each of
# these is a reported finding, pinned here so that it cannot be quietly
# "fixed" by adding a field without the test changing too.
# --------------------------------------------------------------------------

def test_a_two_temperature_protocol_is_not_expressible():
    """A quench has a mixing temperature and a target, and this has one T.

    The archived quench equilibrates far above the critical point and then
    switches the thermostat. Only the target survives here, because the other
    temperature is the one that mixes THAT material and is therefore a
    declared property of the system, not of the request.
    """
    quench = StatePoint(geometry="quench", ensemble="brownian", T=1.1, x=0.5,
                        dt_ps=2e-4, equil_ps=200000 * 1e-3,
                        prod_ps=50000000 * 2e-4, dump_every_ps=5000 * 2e-4,
                        dump_from="prod", seed=5000000, n_atoms=6912)
    record = quench.to_record()
    assert quench.T == 1.1
    assert record["T"] == 1.1
    assert set(record) >= {"T", "dt_ps", "prod_ps", "equil_ps"}
    assert "T_high" not in record
    assert not any("melt" in k or "high" in k for k in record)


def test_a_second_equilibration_timestep_is_not_expressible():
    """One archived tree equilibrates at a different timestep than it runs.

    Expressing the equilibration as a TIME rather than as a step count is what
    keeps that run reproducible here: the length survives, the second timestep
    does not, and it belongs to the deck that performs the equilibration.
    """
    point = StatePoint(geometry="cube", ensemble="brownian", T=1.5, x=0.5,
                       dt_ps=2e-4, equil_ps=250000 * 1e-3, prod_ps=5000.0,
                       dump_every_ps=1.0, dump_from="prod", seed=1,
                       n_atoms=4000)
    record = point.to_record()
    assert point.equil_ps == pytest.approx(250.0)
    assert record["equil_ps"] == pytest.approx(250.0)
    assert record["dt_ps"] == 2e-4
    assert "dt_equil" not in record
    # The step count of the equilibration is NOT recoverable, and must not be
    # guessed from the production timestep.
    assert point.n_steps != 250000 + 5000.0 / 2e-4


def test_a_campaign_record_refuses_keys_it_does_not_model():
    """Archived manifests carry bookkeeping this vocabulary does not keep.

    Dropping them silently would let a campaign look round-tripped when its
    provenance had been thrown away, so an unknown key is refused by name.
    """
    record = _point().to_record()
    record["kind"] = "grid"
    record["engine"] = "one-of-two-potentials"
    record["n_EOS_atoms_per_A3"] = 0.082947
    with pytest.raises(ValueError) as excinfo:
        StatePoint.from_record(record)
    message = str(excinfo.value)
    assert "kind" in message and "engine" in message


# --------------------------------------------------------------------------
# The vocabulary itself.
# --------------------------------------------------------------------------

def test_archived_ensemble_spellings_normalise():
    assert normalise_ensemble("npt_iso") == "NPT"
    assert normalise_ensemble("npt") == "NPT"
    assert normalise_ensemble("npt_z") == "NPT_z"
    assert normalise_ensemble("nvt") == "NVT"
    assert normalise_ensemble("langevin") == "NVT"
    assert normalise_ensemble("brownian") == "langevin_overdamped"
    assert normalise_ensemble("overdamped") == "langevin_overdamped"
    assert normalise_ensemble("NPT_z") == "NPT_z"


def test_archived_geometry_spellings_normalise():
    assert normalise_geometry("slab_meltfirst") == "slab"
    assert normalise_geometry("slab_overdamped") == "slab"
    assert normalise_geometry("homogeneous_brownian") == "cube"
    assert normalise_geometry("spinodal_cube") == "cube"
    assert normalise_geometry("eos_grid") == "eos"
    assert normalise_geometry("cube") == "cube"


def test_surrounding_whitespace_is_not_a_new_ensemble():
    assert normalise_ensemble("  npt_z  ") == "NPT_z"
    assert normalise_geometry(" slab ") == "slab"


def test_an_unknown_ensemble_is_refused_and_both_sides_are_named():
    with pytest.raises(ValueError) as excinfo:
        _point(ensemble="nph")
    message = str(excinfo.value)
    assert "nph" in message
    assert "NPT_z" in message


def test_an_unknown_geometry_is_refused():
    with pytest.raises(ValueError) as excinfo:
        _point(geometry="ellipsoid")
    assert "ellipsoid" in str(excinfo.value)


def test_a_non_string_ensemble_is_a_type_error():
    with pytest.raises(TypeError):
        _point(ensemble=3)


def test_a_non_string_geometry_is_a_type_error():
    with pytest.raises(TypeError):
        _point(geometry=None)


def test_the_request_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _point().dt_ps = 1e-3


# -- the timestep, which is the field that corrupts silently -----------------

def test_the_timestep_has_no_default():
    with pytest.raises(TypeError):
        StatePoint(geometry="cube", ensemble="NPT", T=7000.0, x=0.5,
                   equil_ps=1.5, prod_ps=30.0, dump_every_ps=0.02,
                   dump_from="prod", seed=1, P=800.0)


@pytest.mark.parametrize("bad", [0.0, -1e-4])
def test_a_non_positive_timestep_is_refused(bad):
    with pytest.raises(ValueError):
        _point(dt_ps=bad)


def test_a_boolean_timestep_is_not_a_number():
    with pytest.raises(TypeError):
        _point(dt_ps=True)


def test_an_infinite_timestep_is_refused():
    with pytest.raises(ValueError):
        _point(dt_ps=float("inf"))


def test_a_not_a_number_timestep_is_refused():
    with pytest.raises(ValueError):
        _point(dt_ps=float("nan"))


def test_every_archived_timestep_is_accepted():
    """The archive spans a factor of fifty and the module rules out none."""
    for dt in (1e-4, 2e-4, 1e-3, 2e-3, 5e-3):
        assert _point(dt_ps=dt, dump_every_ps=max(0.02, dt)).dt_ps == dt


# -- lengths -----------------------------------------------------------------

def test_a_zero_equilibration_is_legal_because_a_continuation_has_none():
    assert _point(equil_ps=0.0).equil_ps == 0.0


def test_a_negative_equilibration_is_refused():
    with pytest.raises(ValueError):
        _point(equil_ps=-1.0)


def test_a_zero_production_is_refused():
    with pytest.raises(ValueError) as excinfo:
        _point(prod_ps=0.0)
    assert "prod_ps" in str(excinfo.value)


def test_a_zero_production_is_refused_even_where_nothing_is_dumped():
    """The shape with no cadence, which no other rule can catch for it."""
    with pytest.raises(ValueError) as excinfo:
        _point(prod_ps=0.0, dump_every_ps=None, dump_from="none")
    assert "prod_ps" in str(excinfo.value)


def test_total_time_is_both_stages():
    assert _point(equil_ps=1.5, prod_ps=30.0).total_ps == pytest.approx(31.5)


def test_step_count_uses_the_production_timestep_over_the_whole_run():
    assert _point(equil_ps=1.5, prod_ps=30.0, dt_ps=2e-4).n_steps == 157500


def test_a_time_that_binary_cannot_hold_exactly_still_counts_whole_steps():
    """0.7 divided by 0.1 is 6.999999999999999 in binary floating point.

    Every campaign length and cadence in the archive is written as a decimal,
    and truncating the quotient would drop the last step and the last frame of
    any run whose numbers land on the wrong side of that. None of the archived
    combinations happens to, which is exactly why this is pinned on a
    constructed one rather than left to luck.
    """
    point = _point(equil_ps=0.0, prod_ps=0.7, dt_ps=0.1, dump_every_ps=0.1)
    assert point.n_steps == 7
    assert point.n_frames == 8


# -- the dump ----------------------------------------------------------------

def test_no_cadence_means_no_trajectory_and_the_two_must_agree():
    point = _point(dump_every_ps=None, dump_from="none")
    assert point.n_frames == 0
    assert point.dump_every_ps is None
    assert _point(dump_every_ps=0.02, dump_from="prod").n_frames > 0


def test_a_cadence_with_no_dump_stage_is_refused():
    with pytest.raises(ValueError):
        _point(dump_every_ps=0.02, dump_from="none")


def test_a_dump_stage_with_no_cadence_is_refused():
    with pytest.raises(ValueError):
        _point(dump_every_ps=None, dump_from="prod")


def test_an_unknown_dump_stage_is_refused():
    with pytest.raises(ValueError):
        _point(dump_from="production")


def test_the_dump_stages_are_the_three_the_archive_uses():
    assert DUMP_FROM == {"prod", "equil", "none"}


def test_a_cadence_equal_to_the_timestep_is_legal():
    """One archived run dumped every single step, on purpose."""
    point = _point(dt_ps=2e-4, dump_every_ps=2e-4, prod_ps=50.0)
    assert point.dump_every_ps == 2e-4
    assert point.n_frames == 250001


def test_a_cadence_shorter_than_the_timestep_is_refused():
    with pytest.raises(ValueError):
        _point(dt_ps=2e-4, dump_every_ps=1e-4)


def test_a_cadence_longer_than_production_is_refused():
    with pytest.raises(ValueError):
        _point(prod_ps=30.0, dump_every_ps=31.0)


def test_a_cadence_equal_to_production_keeps_two_frames():
    assert _point(prod_ps=30.0, dump_every_ps=30.0).n_frames == 2


def test_keeping_the_equilibration_frames_lengthens_the_trajectory():
    kept = _point(dump_from="equil").n_frames
    dropped = _point(dump_from="prod").n_frames
    assert kept > dropped
    assert kept - dropped == round(1.5 / 0.02)


# -- pressure ----------------------------------------------------------------

def test_a_pressure_controlled_ensemble_must_name_a_pressure():
    for ensemble in ("NPT", "NPT_z"):
        with pytest.raises(ValueError) as excinfo:
            _point(ensemble=ensemble, P=None)
        assert "pressure" in str(excinfo.value)


def test_zero_pressure_is_a_target_and_not_an_absence():
    """One archived campaign ran its whole first pass at zero pressure."""
    point = _point(ensemble="NPT", P=0.0)
    assert point.P == 0.0
    assert point.P is not None
    assert "P0" in point.tag
    assert point.to_meta_fields()["P_GPa"] == 0.0


def test_a_fixed_box_run_may_have_no_pressure():
    """The reduced-unit archive controls no pressure anywhere."""
    point = _point(ensemble="brownian", P=None)
    assert point.P is None
    assert point.ensemble == "langevin_overdamped"
    assert StatePoint.from_record(point.to_record()) == point


def test_a_negative_pressure_is_refused():
    with pytest.raises(ValueError):
        _point(P=-1.0)


def test_every_archived_pressure_is_accepted():
    for P in (0.0, 5.0, 10.0, 15.0, 200.0, 400.0, 600.0, 800.0):
        assert _point(P=P).P == P


# -- composition -------------------------------------------------------------

def test_a_uniform_build_takes_one_fraction():
    assert _point(x=0.35).x == 0.35


def test_a_build_made_of_regions_takes_one_fraction_per_region():
    assert _point(geometry="slab", ensemble="NPT_z", x=[0.05, 0.95]).x == (0.05, 0.95)


def test_both_pure_limits_are_legal():
    """Archived slabs were built pure against pure, and isotherms run to both
    ends."""
    assert _point(geometry="slab", ensemble="NPT_z", x=(0.0, 1.0)).x == (0.0, 1.0)
    assert _point(x=0.0).x == 0.0
    assert _point(x=1.0).x == 1.0


def test_a_composition_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        _point(x=1.5)
    with pytest.raises(ValueError):
        _point(x=-0.1)
    with pytest.raises(ValueError):
        _point(x=(0.5, 1.5))


def test_a_one_element_sequence_is_refused_as_a_region_list():
    with pytest.raises(ValueError):
        _point(x=[0.5])


def test_a_uniform_slab_is_legal_because_one_archive_builds_it_that_way():
    assert _point(geometry="slab_meltfirst", ensemble="langevin",
                  P=None, x=0.5).x == 0.5


# -- atom count --------------------------------------------------------------

def test_the_atom_count_may_be_absent_because_a_slab_reports_its_own():
    assert _point(n_atoms=None).n_atoms is None


def test_a_non_positive_atom_count_is_refused():
    with pytest.raises(ValueError):
        _point(n_atoms=0)


def test_a_boolean_atom_count_is_not_an_integer():
    with pytest.raises(TypeError):
        _point(n_atoms=True)


def test_a_fractional_atom_count_is_refused():
    with pytest.raises(TypeError):
        _point(n_atoms=3456.0)


# -- seed --------------------------------------------------------------------

def test_the_seed_is_required():
    with pytest.raises(TypeError):
        StatePoint(geometry="cube", ensemble="NPT", T=7000.0, x=0.5,
                   dt_ps=2e-4, equil_ps=1.5, prod_ps=30.0,
                   dump_every_ps=0.02, dump_from="prod", P=800.0)


def test_a_boolean_seed_is_not_an_integer():
    with pytest.raises(TypeError):
        _point(seed=True)


def test_a_floating_point_seed_is_refused():
    with pytest.raises(TypeError):
        _point(seed=1.0)


def test_every_archived_seed_is_accepted():
    for seed in (1, 2, 3, 11, 42, 43, 1001, 12345, 498459, 5000000, 7000000):
        assert _point(seed=seed).seed == seed


# -- temperature -------------------------------------------------------------

def test_a_non_positive_temperature_is_refused():
    with pytest.raises(ValueError):
        _point(T=0.0)


def test_both_archives_temperature_scales_are_accepted():
    """One tree works in kelvin, another in the units of its own potential."""
    assert _point(T=12000.0).T == 12000.0
    assert _point(T=1.05).T == 1.05


# -- continuation ------------------------------------------------------------

def test_a_continuation_names_its_parent_and_does_not_re_equilibrate():
    parent = _point()
    child = parent.continuation(prod_ps=150.0)
    assert child.parent == parent.tag
    assert child.equil_ps == 0.0
    assert child.prod_ps == 150.0


def test_a_continuation_that_re_equilibrates_is_refused():
    with pytest.raises(ValueError) as excinfo:
        _point(parent="cube_x0.50_T7000_P800_s1", equil_ps=1.5)
    assert "equil_ps=0" in str(excinfo.value)


def test_an_empty_parent_is_not_a_parent():
    with pytest.raises(ValueError):
        _point(parent="   ", equil_ps=0.0)


def test_a_parent_is_stripped_of_surrounding_whitespace():
    assert _point(parent=" a_tag ", equil_ps=0.0).parent == "a_tag"


def test_a_continuation_is_a_distinct_request_from_its_parent():
    parent = _point()
    child = parent.continuation(prod_ps=150.0)
    assert child.tag != parent.tag
    campaign_to_records([parent, child])


# -- the tag, and campaigns --------------------------------------------------

def test_the_tag_is_derived_and_not_stored():
    point = _point()
    record = point.to_record()
    record["tag"] = "something-else"
    assert StatePoint.from_record(record).tag == point.tag
    assert point.tag.startswith("cube_")
    assert point.tag != "something-else"


def test_the_tag_separates_points_that_differ_only_in_composition():
    assert _point(x=0.5).tag != _point(x=0.55).tag


def test_a_two_decimal_composition_column_gives_distinct_tags():
    """One archived campaign filled a corner at two-decimal spacing.

    Its five compositions sit at the same temperature, pressure and seed, so
    the composition is the only thing separating their directories.
    """
    column = [0.90, 0.92, 0.94, 0.96, 0.98]
    tags = {_point(x=x).tag for x in column}
    assert len(tags) == len(column)


def test_the_tag_separates_points_that_differ_only_in_pressure():
    assert _point(P=200.0).tag != _point(P=400.0).tag


def test_the_tag_separates_points_that_differ_only_in_seed():
    assert _point(seed=1).tag != _point(seed=2).tag


def test_the_tag_separates_points_that_differ_only_in_temperature():
    assert _point(T=7000.0).tag != _point(T=8000.0).tag


def test_a_point_without_a_pressure_says_so_in_its_tag():
    unpressurised = _point(ensemble="brownian", P=None).tag
    pressurised = _point(ensemble="NPT", P=800.0).tag
    assert "P800" in pressurised
    assert "P" not in unpressurised
    assert unpressurised != pressurised


def test_the_tag_of_a_region_build_names_every_region():
    tag = _point(geometry="slab", ensemble="NPT_z", x=(0.05, 0.95)).tag
    assert "0.05" in tag and "0.95" in tag


def test_a_campaign_with_a_repeated_tag_is_refused():
    point = _point()
    with pytest.raises(ValueError) as excinfo:
        campaign_to_records([point, point])
    assert point.tag in str(excinfo.value)


def test_the_clash_report_counts_distinct_tags_not_occurrences():
    point = _point()
    with pytest.raises(ValueError) as excinfo:
        campaign_to_records([point, point, point])
    assert "1 repeated" in str(excinfo.value)


def test_a_repeated_tag_is_refused_on_the_way_in_too():
    records = [_point().to_record(), _point().to_record()]
    with pytest.raises(ValueError):
        campaign_from_records(records)


def test_an_empty_campaign_is_a_campaign():
    assert campaign_to_records([]) == []
    assert campaign_from_records([]) == []
    assert len(campaign_to_records([_point()])) == 1


def test_a_record_missing_a_required_key_is_refused_by_name():
    record = _point().to_record()
    del record["dt_ps"]
    with pytest.raises(ValueError) as excinfo:
        StatePoint.from_record(record)
    assert "dt_ps" in str(excinfo.value)


def test_a_record_may_omit_the_optional_keys():
    record = _point(ensemble="brownian", P=None).to_record()
    for key in ("P", "n_atoms", "parent"):
        del record[key]
    rebuilt = StatePoint.from_record(record)
    assert rebuilt.P is None and rebuilt.n_atoms is None


# -- the metadata bridge -----------------------------------------------------

def test_the_metadata_fields_are_only_the_ones_a_request_determines():
    fields = _point().to_meta_fields()
    assert set(fields) == {
        "tag", "geometry", "ensemble", "T_K", "P_GPa", "n_atoms",
        "dt_ps", "dump_every_ps", "n_frames", "seed"}
    for absent in ("system", "composition", "potential", "status", "box"):
        assert absent not in fields


def test_the_metadata_carries_the_requested_atom_count_or_its_absence():
    assert _point(n_atoms=3456).to_meta_fields()["n_atoms"] == 3456
    assert _point(n_atoms=512).to_meta_fields()["n_atoms"] == 512
    assert _point(n_atoms=None).to_meta_fields()["n_atoms"] is None


def test_the_metadata_fields_use_the_record_side_names():
    fields = _point(T=7000.0, P=800.0).to_meta_fields()
    assert fields["T_K"] == 7000.0
    assert fields["P_GPa"] == 800.0


def test_a_request_with_no_pressure_carries_a_null_pressure_forward():
    assert _point(ensemble="brownian", P=None).to_meta_fields()["P_GPa"] is None


@pytest.mark.parametrize("label,point,n_frames",
                         [row for row in ARCHIVED if row[2] is not None],
                         ids=[row[0].split(" ", 1)[0]
                              for row in ARCHIVED if row[2] is not None])
def test_the_metadata_frame_count_is_the_measured_one(label, point, n_frames):
    """The bridge carries the derived count, not a placeholder."""
    assert point.to_meta_fields()["n_frames"] == n_frames == point.n_frames


def test_a_request_that_writes_nothing_carries_a_zero_frame_count():
    fields = _point(dump_every_ps=None, dump_from="none").to_meta_fields()
    assert fields["n_frames"] == 0
    assert fields["dump_every_ps"] is None
