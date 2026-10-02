"""The decks, and the evidence that each one ran.

A deck nobody ran is not a template. Every built-in deck below names the
archived file it was derived from and the published dataset that file
produced, and the tests assert the command sequence the rendered deck emits
against the stage order read out of that file. Stage order is the load-bearing
part: one archived worker's own docstring says so, because declaring a dump
before rather than after the timestep is reset changes which frames the
dataset contains.

Step counts are the second kind of evidence. The archived campaigns were
written in STEPS and this package works in TIME, so every row of
``ARCHIVED_STEPS`` converts one back into the other and asserts the package
recovers the step count the campaign actually ran.

Nothing here reads the protected trees at test time. The numbers are quoted,
each beside the file it was read from, the same way the request module's
evidence table is.
"""
import re

import pytest

from aipf.md.request import StatePoint
from aipf.md.templates import (
    DECKS,
    Deck,
    DeckError,
    available,
    check,
    fill,
    for_point,
    from_file,
    from_text,
    placeholders,
    render,
    values_from,
)
from aipf.md.templates.builtin import builtin
from aipf.paths import Paths
from aipf.system import AnchorRules, PreparationRules, System


# ---------------------------------------------------------------------------
# Neutral fixtures. No system is named here either: the decks are general and
# so is their evidence.
# ---------------------------------------------------------------------------


def _system(**overrides) -> System:
    """A two-channel system with masses and types, one field replaced."""
    kwargs = dict(
        name="demo", n_species=2, species=("one", "two"),
        masses={"one": 1.008, "two": 4.0026},
        atom_types={"one": 1, "two": 2},
        table_keys={}, paths=Paths(system="demo"),
        anchor_rules=AnchorRules({}), constants={}, defaults={},
    )
    kwargs.update(overrides)
    return System(**kwargs)


def _point(**overrides) -> StatePoint:
    """A barostatted cube request, one field replaced."""
    base = dict(geometry="cube", ensemble="NPT", T=7000.0, x=0.5,
                dt_ps=2e-4, equil_ps=1.5, prod_ps=30.0,
                dump_every_ps=0.02, dump_from="prod", seed=1, P=800.0,
                n_atoms=3456)
    base.update(overrides)
    return StatePoint(**base)


#: Values a request and a system cannot determine. Every one of these is a
#: knob of the engine or a location on disk, and the point of the list is that
#: it is short and that each entry is genuinely unknowable here.
CALLER = {
    "UNITS": "metal",
    "CONFIGURATION": "read_data ic.data",
    "MASSES": "mass 1 1.008\nmass 2 4.0026",
    "PAIR": "pair_style test\npair_coeff * * one two",
    "SKIN": "2.0",
    "NEIGH_MODIFY": "every 1 delay 0 check yes",
    "RELAX": "",
    "TDAMP": "0.01",
    "PDAMP": "0.1",
    "DAMP": "0.2",
    "GAMMA": "2.0",
    "N_THERMO": "200",
    "P": "8000000",
    "TRAJECTORY": "traj.lammpstrj",
    "THERMO_FILE": "thermo.dat",
    "WRITE_END": "write_data final.data",
    "EXTERNAL": "fix bias all addforce 0.0 0.0 0.0",
}


def _commands(text: str) -> list[str]:
    """The LAMMPS commands a deck emits, comments and blank lines removed.

    Continuation lines are joined, so a command written over two lines counts
    once. This is what the stage-order assertions compare.
    """
    joined, buffer = [], ""
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.endswith("&"):
            buffer += stripped[:-1].strip() + " "
            continue
        joined.append(re.sub(r"\s+", " ", buffer + stripped))
        buffer = ""
    assert not buffer, "a continuation line was never closed"
    return joined


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------


def test_placeholders_are_found_in_order_of_first_appearance():
    text = "a @ONE@\nb @TWO@ @ONE@\nc @THREE_2@"
    assert placeholders(text) == ("ONE", "TWO", "THREE_2")


def test_a_lammps_variable_is_not_a_placeholder():
    """Every archived deck is full of ``${var}``. Reading one as a placeholder
    would make the module refuse decks it is supposed to leave alone."""
    assert placeholders("run ${N_prod}\nvelocity all create ${T} ${seed}") == ()


def test_a_bare_at_sign_is_not_a_placeholder():
    """One archived deck prints ``steps @ gamma=...``. A marker that fired on a
    lone ``@`` would refuse that deck."""
    assert placeholders("print '--- 1e8 steps @ gamma=2.0 ---'") == ()


def test_a_lower_case_token_is_not_a_placeholder():
    assert placeholders("@name@ @Name@") == ()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_every_occurrence_of_a_placeholder_is_filled():
    """The pressure appears twice in a barostatted deck, once per stage. A
    substitution that stopped at the first would run the second stage at a
    pressure nobody asked for."""
    out = render(from_text("fix a iso @P@ @P@\nfix b iso @P@ @P@"), {"P": "7"})
    assert out == "fix a iso 7 7\nfix b iso 7 7"


def test_an_unfilled_placeholder_is_refused_by_name():
    with pytest.raises(DeckError) as exc:
        render(from_text("timestep @DT@"), {"T": "1"})
    assert "DT" in str(exc.value)


def test_a_none_value_is_refused_rather_than_written():
    """A request with no pressure has ``P is None``. Writing ``iso None None``
    would be a deck LAMMPS rejects at parse time, hours after submission."""
    with pytest.raises(DeckError) as exc:
        render(from_text("fix a iso @P@ @P@"), {"P": None})
    assert "P" in str(exc.value)


def test_a_line_that_is_only_a_placeholder_disappears_when_it_is_empty():
    """A deck that optionally relaxes its configuration has one line for it.
    Leaving a blank line behind is harmless; leaving the marker is not."""
    out = render(from_text("units lj\n@RELAX@\ntimestep 1"),
                 {"RELAX": ""})
    assert out == "units lj\ntimestep 1"
    # Whitespace is nothing too, and leaves no stray line behind either.
    out = render(from_text("units lj\n@RELAX@\ntimestep 1"),
                 {"RELAX": "   "})
    assert out == "units lj\ntimestep 1"


def test_an_empty_value_inside_a_line_is_refused():
    """``write_data`` with nothing after it is a parse error. A deck drops a
    whole command or keeps it; it never emits half of one."""
    with pytest.raises(DeckError) as exc:
        render(from_text("write_data @FINAL@"), {"FINAL": ""})
    assert "FINAL" in str(exc.value)


def test_a_value_may_carry_several_lines():
    out = render(from_text("@PAIR@\ntimestep 1"),
                 {"PAIR": "pair_style x\npair_coeff * *"})
    assert out.splitlines() == ["pair_style x", "pair_coeff * *", "timestep 1"]


def test_a_value_may_itself_carry_a_placeholder():
    """The dump block is generated, and the file it writes to is the caller's.
    One of the two has to be able to contain the other."""
    out = render(from_text("@DUMP@"),
                 {"DUMP": "dump d all custom @N@ @FILE@ id",
                  "N": "100", "FILE": "traj.lammpstrj"})
    assert out == "dump d all custom 100 traj.lammpstrj id"


def test_a_self_referential_value_is_refused_rather_than_looped():
    with pytest.raises(DeckError) as exc:
        render(from_text("@A@"), {"A": "x @A@"})
    assert "A" in str(exc.value)


def test_a_rendered_deck_carries_no_placeholder_left():
    out = fill(builtin("cube", "NPT"), _point(), _system(), CALLER)
    assert placeholders(out) == ()


def test_values_the_deck_does_not_use_are_allowed():
    """One derived mapping serves every deck, and no deck uses all of it: the
    equation-of-state deck writes no trajectory and so never asks for a dump
    cadence. Refusing the spare keys would mean a mapping per deck."""
    out = render(from_text("timestep @DT@"), {"DT": "1", "SPARE": "2"})
    assert out == "timestep 1"


def test_a_blank_line_in_a_deck_survives():
    assert render(from_text("a\n\nb"), {}) == "a\n\nb"


# ---------------------------------------------------------------------------
# What a request determines. ARCHIVED_STEPS is the evidence table.
# ---------------------------------------------------------------------------

#: label, request, equilibration timestep override, expected step counts.
#: Every row names the file its numbers were read out of, and every number is
#: the step count that campaign actually ran.
ARCHIVED_STEPS: list[tuple[str, StatePoint, float | None, dict[str, int]]] = [
    ("first heavy tree, Data/slab_data/gen_manifest.py CUBE_RUN: "
     "n_equil 7500, n_prod 150000, dump 100, dt 2e-4",
     StatePoint(geometry="cube", ensemble="npt_iso", T=7000.0, x=0.5,
                dt_ps=2e-4, equil_ps=7500 * 2e-4, prod_ps=150000 * 2e-4,
                dump_every_ps=100 * 2e-4, dump_from="prod", seed=1,
                P=800.0, n_atoms=3456),
     None, {"N_EQUIL": 7500, "N_PROD": 150000, "N_DUMP": 100}),
    ("first heavy tree, Data/slab_data/gen_manifest.py SLAB_RUN: "
     "n_equil 5000, n_prod 400000, dump 200, dt 1e-4",
     StatePoint(geometry="slab", ensemble="npt_z", T=2000.0, x=(0.0, 1.0),
                dt_ps=1e-4, equil_ps=5000 * 1e-4, prod_ps=400000 * 1e-4,
                dump_every_ps=200 * 1e-4, dump_from="prod", seed=1, P=800.0),
     None, {"N_EQUIL": 5000, "N_PROD": 400000, "N_DUMP": 200}),
    ("first heavy tree, Data/eos/launch_eos.py: N_EQUIL 5000, N_AVG 5000 at "
     "the worker's hardcoded timestep 2e-4, and no trajectory at all",
     StatePoint(geometry="eos", ensemble="npt_iso", T=5000.0, x=0.5,
                dt_ps=2e-4, equil_ps=5000 * 2e-4, prod_ps=5000 * 2e-4,
                dump_every_ps=None, dump_from="none", seed=42, P=800.0,
                n_atoms=512),
     None, {"N_EQUIL": 5000, "N_PROD": 5000}),
    ("second heavy tree, Data/cube/run_cube.py defaults: melt 10 ps and "
     "production 50 ps at dt 1e-3, dumped every 0.1 ps",
     StatePoint(geometry="cube", ensemble="npt_iso", T=1800.0, x=0.8,
                dt_ps=1e-3, equil_ps=10.0, prod_ps=50.0,
                dump_every_ps=0.1, dump_from="prod", seed=1, P=0.0,
                n_atoms=3456),
     None, {"N_EQUIL": 10000, "N_PROD": 50000, "N_DUMP": 100}),
    ("reduced-unit tree, configs/md_pipeline.yaml quench block: N_equil "
     "200000 at dt_equil 1e-3, N_prod 5e7 at dt_sim 2e-4, dump_every 5000",
     StatePoint(geometry="quench", ensemble="brownian", T=1.10, x=0.5,
                dt_ps=2e-4, equil_ps=200000 * 1e-3,
                prod_ps=50000000 * 2e-4,
                dump_every_ps=5000 * 2e-4, dump_from="prod", seed=5000000),
     1e-3, {"N_EQUIL": 200000, "N_PROD": 50000000, "N_DUMP": 5000}),
    ("reduced-unit tree, md_pipeline.yaml homogeneous_brownian block: "
     "N_equil 250000 at dt_equil 1e-3, N_prod 2.5e7, dump_every 5000",
     StatePoint(geometry="homogeneous_brownian", ensemble="brownian",
                T=1.60, x=0.5, dt_ps=2e-4, equil_ps=250000 * 1e-3,
                prod_ps=25000000 * 2e-4, dump_every_ps=5000 * 2e-4,
                dump_from="prod", seed=42),
     1e-3, {"N_EQUIL": 250000, "N_PROD": 25000000, "N_DUMP": 5000}),
    ("reduced-unit tree, md_pipeline.yaml slab_overdamped block: no "
     "equilibration at all, N_prod 1e8 at dt 2e-4, dump_every 5000",
     StatePoint(geometry="slab_overdamped", ensemble="brownian", T=1.00,
                x=0.5, dt_ps=2e-4, equil_ps=0.0, prod_ps=100000000 * 2e-4,
                dump_every_ps=5000 * 2e-4, dump_from="prod", seed=43,
                parent="slab_meltfirst_frame5"),
     None, {"N_EQUIL": 0, "N_PROD": 100000000, "N_DUMP": 5000}),
    ("reduced-unit tree, md_pipeline.yaml vext.profiles.checkerboard: "
     "N_equil 500000, N_prod 1500000, dump_freq 20000, dt 5e-3",
     StatePoint(geometry="vext", ensemble="langevin", T=1.30, x=0.5,
                dt_ps=5e-3, equil_ps=500000 * 5e-3,
                prod_ps=1500000 * 5e-3, dump_every_ps=20000 * 5e-3,
                dump_from="prod", seed=12345),
     None, {"N_EQUIL": 500000, "N_PROD": 1500000, "N_DUMP": 20000}),
]


@pytest.mark.parametrize("label,point,dt_equil,expected",
                         [(r[0], r[1], r[2], r[3]) for r in ARCHIVED_STEPS],
                         ids=[r[0].split(",")[0] + f"-{i}"
                              for i, r in enumerate(ARCHIVED_STEPS)])
def test_the_archived_step_counts_come_back_exactly(label, point, dt_equil,
                                                    expected):
    values = values_from(point, _system(), dt_equil_ps=dt_equil)
    for key, want in expected.items():
        assert values[key] == str(want), label


def test_a_run_that_writes_no_trajectory_gets_no_dump_cadence():
    values = values_from(
        _point(dump_every_ps=None, dump_from="none", geometry="eos"),
        _system())
    assert "N_DUMP" not in values


def test_the_equilibration_timestep_defaults_to_the_production_one():
    """Both heavy trees integrate one timestep from end to end. The second
    timestep is the reduced-unit tree's, where the equilibration is inertial
    and the production is not."""
    point = _point(dt_ps=2e-4, equil_ps=1.5)
    assert values_from(point, _system())["N_EQUIL"] == "7500"
    assert values_from(point, _system(), dt_equil_ps=1e-3)["N_EQUIL"] == "1500"


def test_the_equilibration_timestep_is_reported_alongside_the_production_one():
    values = values_from(_point(), _system(), dt_equil_ps=1e-3)
    assert values["DT_EQUIL"] == "0.001"
    assert values["DT"] == "0.0002"


def test_step_counts_round_rather_than_truncate():
    """Measured, and the reason this is not a matter of taste: 0.7 / 0.1 is
    6.999999999999999 in binary floating point. Truncating would run one step
    short of what was asked for, silently, on a length nothing forbids."""
    point = _point(dt_ps=0.1, equil_ps=0.7, prod_ps=0.7, dump_every_ps=0.7)
    values = values_from(point, _system())
    assert values["N_EQUIL"] == "7"
    assert values["N_PROD"] == "7"


def test_a_timestep_is_written_without_losing_precision():
    """``%g`` turns a large pressure into 1.23457e+06, which is a different
    number. Every value this module writes round-trips."""
    assert values_from(_point(dt_ps=1e-4), _system())["DT"] == "0.0001"
    assert values_from(_point(dt_ps=5e-3), _system())["DT"] == "0.005"


def test_the_seed_and_the_temperature_come_from_the_request():
    values = values_from(_point(T=1234.5, seed=498459), _system())
    assert values["SEED"] == "498459"
    assert values["T"] == "1234.5"


def test_no_pressure_is_derived_from_the_request():
    """The one placeholder a request cannot fill. Both heavy trees converted
    the pressure in the LAUNCHER before the deck ever saw it -- one wrote
    ``P_BAR = 800.0 * 1e4`` in its manifest generator and the other a
    ``p_bar()`` helper in its worker -- and nothing in a system declares which
    unit its pressure is in. Filling ``@P@`` from the request would run a
    barostat at 800 bar instead of 800 GPa, a factor of ten thousand with
    nothing raised anywhere."""
    assert "P" not in values_from(_point(P=800.0), _system())


# ---------------------------------------------------------------------------
# Preparation. Three forms, each measured.
# ---------------------------------------------------------------------------


def test_an_offset_preparation_ramps_from_above_the_target_down_to_it():
    """``Data/slab_data/run_slab_prod.py``: ``T_RAMP = T_K + 2000.0``, and the
    equilibration fix ramps ``T_RAMP -> T_K``."""
    system = _system(preparation=PreparationRules(ramp_offset=2000.0))
    values = values_from(_point(T=7000.0), system)
    assert values["T_PREP"] == "9000.0"
    assert values["T_PREP_END"] == "7000.0"


def test_an_absolute_preparation_holds_at_the_melting_temperature():
    """``Data/cube/run_cube.py``: ``fix melt all npt temp 2600.0 2600.0``,
    held, not ramped, and the production fix goes straight to the target."""
    system = _system(preparation=PreparationRules(melt_T=2600.0))
    values = values_from(_point(T=1800.0), system)
    assert values["T_PREP"] == "2600.0"
    assert values["T_PREP_END"] == "2600.0"


def test_a_system_that_declares_neither_prepares_at_its_own_target():
    """Not a fallback. ``in.homogeneous_brownian.lammps`` equilibrates AT the
    target on purpose -- its own comment says the run is above the critical
    point, so nothing demixes during stage one -- and the slab protocol in the
    first heavy tree does the same, because its configuration arrives liquid."""
    values = values_from(_point(T=1.6), _system())
    assert values["T_PREP"] == "1.6"
    assert values["T_PREP_END"] == "1.6"


def test_a_system_declaring_both_forms_cannot_be_built_at_all():
    """The two are alternatives, not a pair, and nothing here can choose
    between them.

    This task originally asserted that `values_from` refuses such a system,
    which it does. The refusal has since moved EARLIER, into
    `PreparationRules` itself, after this task found a shipped experiment
    declaring both that nothing caught: a conflict that only a deck-filling
    consumer rejects is discovered after the queue wait rather than at
    import. The consumer guard is kept as depth and is now unreachable by
    construction, which is why this asserts the constructor and not the
    consumer.
    """
    with pytest.raises(ValueError) as exc:
        PreparationRules(melt_T=2600.0, ramp_offset=2000.0)
    assert "2600" in str(exc.value) and "2000" in str(exc.value)


def test_the_consumers_own_refusal_still_works_when_it_is_reached():
    """The depth guard, exercised -- because "kept as depth" and "covered by
    nothing" are not the same sentence.

    Moving the refusal into `PreparationRules.__post_init__` made the
    consumer's branch unreachable through the constructor, and the test
    above moved with it, which left the consumer's guard as the only line
    in this module that nothing runs. A frozen dataclass is not an
    immutable object: `object.__setattr__` reaches straight past
    `__post_init__`, which is exactly how a rules object could arrive at
    the consumer in the both-forms state the constructor refuses -- and it
    is the only way left, which is the point of keeping the guard.

    The consumer's message must still name BOTH values, because a caller
    who sees it has no other way to learn which two declarations collided.
    """
    rules = PreparationRules(melt_T=2600.0)
    object.__setattr__(rules, "ramp_offset", 2000.0)
    assert rules.melt_T is not None and rules.ramp_offset is not None
    with pytest.raises(DeckError) as exc:
        values_from(_point(T=7000.0), _system(preparation=rules))
    assert "2600" in str(exc.value) and "2000" in str(exc.value)


def test_a_preparation_that_lands_at_or_below_zero_is_refused():
    system = _system(preparation=PreparationRules(ramp_offset=-8000.0))
    with pytest.raises(DeckError) as exc:
        values_from(_point(T=7000.0), system)
    assert "-1000" in str(exc.value)


# ---------------------------------------------------------------------------
# Masses
# ---------------------------------------------------------------------------


def test_masses_are_written_in_numeric_type_order():
    """The order is the trajectory's, not the declaration's: a system may list
    its channels in one order and assign dump types in another."""
    system = _system(species=("one", "two"),
                     masses={"one": 1.008, "two": 4.0026},
                     atom_types={"one": 2, "two": 1})
    lines = values_from(_point(), system)["MASSES"].splitlines()
    assert lines == ["mass            1 4.0026", "mass            2 1.008"]


def test_an_atom_type_with_no_declared_mass_is_refused():
    """A dump may carry a type that is not a density channel -- a tracer, a
    wall -- and the declaration allows it. LAMMPS still needs a mass for every
    type, so a type without one is a run that dies at setup."""
    system = _system(atom_types={"one": 1, "two": 2, "wall": 3})
    with pytest.raises(DeckError) as exc:
        values_from(_point(), system)
    assert "3" in str(exc.value) and "wall" in str(exc.value)


def test_a_system_that_declares_no_masses_leaves_the_block_to_the_caller():
    """A configuration read from a data file carries its own masses. The
    block is then absent rather than wrong, and a deck that needs it says so
    by name."""
    system = _system(masses={}, atom_types={})
    assert "MASSES" not in values_from(_point(), system)


# ---------------------------------------------------------------------------
# Which stage the dump starts in
# ---------------------------------------------------------------------------


def test_a_production_dump_fills_the_production_slot_and_empties_the_other():
    values = values_from(_point(dump_from="prod"), _system())
    assert values["DUMP_EQUIL"] == ""
    assert "dump " in values["DUMP_PROD"]
    # The clock reset travels with the block, because the block moves; its
    # exact form is the deck's, because the archived decks disagree on it.
    assert "@DUMP_RESET@" in values["DUMP_PROD"]


def test_an_equilibration_dump_fills_the_other_slot():
    """``Data/slab_data/gen_phaseC.py`` asked for this on purpose: the early
    linear growth window lives in the frames the melt would otherwise throw
    away."""
    values = values_from(_point(dump_from="equil"), _system())
    assert values["DUMP_PROD"] == ""
    assert "dump " in values["DUMP_EQUIL"]


def test_a_run_with_no_trajectory_empties_both_slots():
    values = values_from(_point(dump_every_ps=None, dump_from="none",
                                geometry="eos"), _system())
    assert values["DUMP_EQUIL"] == "" and values["DUMP_PROD"] == ""


# ---------------------------------------------------------------------------
# check: the four ways a deck and a request can disagree
# ---------------------------------------------------------------------------


def test_a_deck_with_a_preparation_stage_refuses_a_continuation():
    """Every archived continuation sets its equilibration to zero, because the
    configuration it resumes is already equilibrated. Running the melt again
    would destroy exactly the structure the leg exists to grow."""
    deck = builtin("slab", "langevin_overdamped")
    with pytest.raises(DeckError):
        check(builtin("cube", "NPT"), _point(equil_ps=0.0, parent="p"))
    check(deck, _point(geometry="slab", ensemble="brownian", equil_ps=0.0,
                       parent="p", P=None))


def test_a_deck_with_no_preparation_stage_refuses_an_equilibration():
    with pytest.raises(DeckError) as exc:
        check(builtin("slab", "langevin_overdamped"),
              _point(geometry="slab", ensemble="brownian", equil_ps=1.0,
                     P=None))
    assert "equil" in str(exc.value).lower()


def test_a_deck_that_writes_a_trajectory_refuses_a_request_with_no_cadence():
    with pytest.raises(DeckError):
        check(builtin("cube", "NPT"),
              _point(dump_every_ps=None, dump_from="none"))


def test_the_equation_of_state_deck_refuses_a_request_that_wants_frames():
    """It writes averages, not a trajectory, and it has no dump to fill."""
    with pytest.raises(DeckError) as exc:
        check(builtin("eos", "NPT"), _point(geometry="eos"))
    assert "trajectory" in str(exc.value).lower()


def test_a_deck_without_an_equilibration_dump_slot_refuses_that_request():
    """No reduced-unit campaign dumped its equilibration, and those decks have
    nowhere to put one: their equilibration runs at a different timestep, so
    the frames would not be on the production cadence."""
    point = _point(geometry="cube", ensemble="brownian", dump_from="equil",
                   P=None)
    with pytest.raises(DeckError) as exc:
        check(builtin("cube", "langevin_overdamped"), point)
    assert "equil" in str(exc.value).lower()


def test_a_geometry_that_disagrees_with_the_deck_is_refused():
    with pytest.raises(DeckError) as exc:
        check(builtin("cube", "NPT"), _point(geometry="eos"))
    assert "eos" in str(exc.value)


def test_an_ensemble_that_disagrees_with_the_deck_is_refused():
    with pytest.raises(DeckError) as exc:
        check(builtin("cube", "NPT"),
              _point(geometry="cube", ensemble="NVE", P=None))
    assert "NVE" in str(exc.value)


def test_a_user_deck_declares_no_geometry_and_so_checks_neither():
    """A deck someone else wrote is theirs. The module fills its placeholders
    and does not second-guess what it is for."""
    check(from_text("timestep @DT@"), _point(geometry="eos",
                                             dump_every_ps=None,
                                             dump_from="none"))


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_every_built_in_deck_names_its_source_and_what_it_produced():
    assert len(DECKS) >= 7, "an empty registry would pass every sweep below"
    for key, deck in DECKS.items():
        assert deck.source, key
        assert deck.produced, key


def test_every_built_in_deck_renders_from_a_request_and_a_system():
    """The whole registry, filled end to end. A deck that cannot be filled is
    a deck nobody can run."""
    assert len(DECKS) >= 7, "an empty registry would pass this vacuously"
    for (geometry, ensemble), deck in DECKS.items():
        point = _point(
            geometry=geometry, ensemble=ensemble,
            P=800.0 if ensemble in ("NPT", "NPT_z") else None,
            equil_ps=1.5 if deck.prepares else 0.0,
            dump_every_ps=0.02 if deck.writes_trajectory else None,
            dump_from="prod" if deck.writes_trajectory else "none",
        )
        out = fill(deck, point, _system(), CALLER, dt_equil_ps=1e-3)
        assert placeholders(out) == ()
        assert _commands(out), (geometry, ensemble)


def test_the_registry_is_reachable_by_the_archived_spellings():
    assert builtin("cube", "npt_iso") is builtin("cube", "NPT")
    assert builtin("spinodal_cube", "brownian") is builtin(
        "cube", "langevin_overdamped")


def test_an_unknown_pairing_is_refused_and_lists_what_exists():
    with pytest.raises(DeckError) as exc:
        builtin("ball", "NVE")
    assert "ball" in str(exc.value)
    assert "cube" in str(exc.value)


def test_a_quench_and_a_cube_resolve_to_one_deck():
    """The reduced-unit tree ran ONE file, ``in.quench.lammps``, for both its
    coarsening-validation tree and its in-dome training tree. Two names, one
    deck, and the registry says so rather than carrying a second copy."""
    quench, cube = builtin("quench", "brownian"), builtin("cube", "brownian")
    assert quench.text == cube.text
    assert quench.source == cube.source
    assert quench.produced != cube.produced


def test_available_lists_every_registered_pairing():
    assert set(available()) == set(DECKS)
    assert available() == sorted(available())


def test_a_deck_is_chosen_from_the_request_alone():
    assert for_point(_point()) is builtin("cube", "NPT")


def test_no_built_in_deck_asks_for_a_dump_with_a_time_line():
    """``dump_modify ... time yes`` puts an extra ``ITEM: TIME`` block in every
    frame header. Every reader in this package assumes the nine-line header
    that every archived dump has, so a deck emitting one would produce
    trajectories nothing here can read."""
    assert len(DECKS) >= 7, "an empty registry would pass this vacuously"
    for key, deck in DECKS.items():
        assert "time yes" not in deck.text, key


def test_no_built_in_deck_carries_a_pair_style():
    """The interaction is the system's, and one archived tree runs two
    different engines over the same state points. A deck that named one would
    be a deck that can only run half the archive."""
    assert len(DECKS) >= 7, "an empty registry would pass this vacuously"
    for key, deck in DECKS.items():
        assert "pair_style" not in deck.text, key
        assert "PAIR" in deck.placeholders, key


def test_no_built_in_deck_fixes_a_timestep_or_a_temperature_of_its_own():
    """Every number a deck writes comes from a request, a system or a caller.
    A literal here would be a default that silently outlives the campaign that
    justified it."""
    assert len(DECKS) >= 7, "an empty registry would pass this vacuously"
    for key, deck in DECKS.items():
        for command in _commands(deck.text):
            head = command.split()[0]
            if head in ("timestep", "velocity", "run", "thermo"):
                assert "@" in command, (key, command)


# ---------------------------------------------------------------------------
# Stage order, deck by deck, against the archived worker it came from
# ---------------------------------------------------------------------------


#: A system whose preparation is an offset, so that the temperature the run
#: starts at and the one it produces at are DIFFERENT numbers. With a system
#: that declares no preparation they are equal, and a deck that used the wrong
#: one in either place would render identically.
RAMPING = PreparationRules(ramp_offset=2000.0)


def _rendered(geometry, ensemble, dt_equil_ps=None, system=None,
              **point_kwargs) -> list[str]:
    deck = builtin(geometry, ensemble)
    base = dict(geometry=geometry, ensemble=ensemble,
                P=800.0 if ensemble in ("NPT", "NPT_z") else None)
    base.update(point_kwargs)
    return _commands(fill(deck, _point(**base),
                          system if system is not None else _system(),
                          CALLER, dt_equil_ps=dt_equil_ps))


def test_the_barostatted_cube_keeps_the_archived_stage_order():
    """``Data/slab_data/run_slab_prod.py``, ENSEMBLE=npt_iso: velocities at the
    preparation temperature, one equilibration fix that ends at the target,
    the recentring fix, then the dump declared only after the clock is reset,
    then production. Its own docstring calls the order load-bearing."""
    got = _rendered("cube", "NPT", system=_system(preparation=RAMPING))
    assert got == [
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map yes",
        "read_data ic.data",
        "mass 1 1.008",
        "mass 2 4.0026",
        "pair_style test",
        "pair_coeff * * one two",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
        "timestep 0.0002",
        "thermo 200",
        "thermo_style custom step time temp pe press vol lx ly lz",
        "thermo_modify flush yes",
        "velocity all create 9000.0 1 mom yes rot yes dist gaussian",
        "fix prep all npt temp 9000.0 7000.0 0.01 iso 8000000 8000000 0.1",
        "fix recentre all recenter NULL NULL INIT",
        "run 7500",
        "unfix prep",
        "reset_timestep 0",
        "dump traj all custom 100 traj.lammpstrj id type x y z",
        "dump_modify traj sort id flush yes",
        "fix prod all npt temp 7000.0 7000.0 0.01 iso 8000000 8000000 0.1",
        "run 150000",
        "unfix prod",
        "undump traj",
        "write_data final.data",
    ]


def test_the_barostatted_cube_moves_its_dump_when_the_melt_is_kept():
    """``dump_from="equil"``: the dump block moves ahead of the equilibration
    fix and the clock is reset once, before it, so the trajectory covers the
    whole run."""
    got = _rendered("cube", "NPT", dump_from="equil",
                    system=_system(preparation=RAMPING))
    assert got.index("dump traj all custom 100 traj.lammpstrj id type x y z") \
        < got.index("run 7500")
    assert got.count("reset_timestep 0") == 1


def test_the_slab_deck_barostats_one_axis_and_does_not_melt():
    """``run_slab_prod.py`` ties the over-temperature ramp to the ISOTROPIC
    barostat, not to the geometry: ``RAMP = ENSEMBLE == "npt_iso"``, with the
    comment that the ramp is only needed to melt a lattice configuration. The
    one-axis path starts from a liquid, so it settles at its own target."""
    system = _system(preparation=PreparationRules(ramp_offset=2000.0))
    deck = builtin("slab", "NPT_z")
    got = _commands(fill(deck, _point(geometry="slab", ensemble="NPT_z",
                                      x=(0.0, 1.0), T=2000.0, n_atoms=None),
                         system, CALLER))
    assert "fix prep all npt temp 2000.0 2000.0 0.01 z 8000000 8000000 0.1" \
        in got
    assert not any("4000" in line for line in got)


def test_the_overdamped_cube_switches_integrator_between_its_two_stages():
    """``in.quench.lammps`` and ``in.homogeneous_brownian.lammps`` differ only
    in the stage-one temperature; everything below is the same file. Stage one
    is inertial at its own timestep, stage two is overdamped at another, and
    the clock is reset between them."""
    got = _rendered("cube", "langevin_overdamped", dt_equil_ps=1e-3,
                    T=1.10, dt_ps=2e-4, equil_ps=200.0, prod_ps=10000.0,
                    dump_every_ps=1.0)
    assert got == [
        "units metal",
        "atom_style atomic",
        "dimension 3",
        "boundary p p p",
        "read_data ic.data",
        "mass 1 1.008",
        "mass 2 4.0026",
        "pair_style test",
        "pair_coeff * * one two",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
        "thermo 200",
        "thermo_style custom step temp pe ke etotal press",
        "timestep 0.001",
        "velocity all create 1.1 1 dist gaussian",
        "fix prep_nve all nve",
        "fix prep all langevin 1.1 1.1 0.2 1",
        "run 200000",
        "unfix prep",
        "unfix prep_nve",
        "reset_timestep 0",
        "timestep 0.0002",
        "fix prod all brownian 1.1 1 gamma_t 2.0",
        "fix recentre all recenter INIT INIT INIT",
        "dump traj all custom 5000 traj.lammpstrj id type x y z",
        "dump_modify traj sort id first yes",
        "run 50000000",
        "write_data final.data",
    ]


def test_the_overdamped_cube_takes_its_melt_from_the_system():
    """With an absolute melt declared, stage one is the quench protocol: hot
    inertial equilibration well above the critical point, then an instant
    switch to the target."""
    system = _system(preparation=PreparationRules(melt_T=2.0))
    deck = builtin("cube", "langevin_overdamped")
    got = _commands(fill(deck, _point(geometry="cube", ensemble="brownian",
                                      T=1.10, P=None), system, CALLER,
                         dt_equil_ps=1e-3))
    assert "fix prep all langevin 2.0 2.0 0.2 1" in got
    assert "fix prod all brownian 1.1 1 gamma_t 2.0" in got


def test_the_continued_slab_deck_has_no_preparation_at_all():
    """``in.slab_overdamped.lammps``: reads a configuration and runs. Its own
    header says there is no relaxation stage on purpose, because the smoothing
    window downstream discards more than the transient lasts."""
    got = _rendered("slab", "langevin_overdamped", equil_ps=0.0,
                    parent="ic_frame5", x=0.5)
    assert not any(line.startswith("velocity") for line in got)
    assert [line for line in got if line.startswith("run ")] == ["run 150000"]


def test_the_external_potential_deck_keeps_its_thermostat_across_both_runs():
    """``in.vext_checkerboard.lammps`` equilibrates and produces under the same
    fixes; only the dump appears between the two runs. Unfixing in between
    would restart the thermostat's history halfway through."""
    got = _rendered("vext", "NVT", T=1.30, dt_ps=5e-3)
    runs = [i for i, line in enumerate(got) if line.startswith("run ")]
    assert len(runs) == 2
    between = got[runs[0] + 1:runs[1]]
    assert not any(line.startswith("unfix") for line in between)
    assert any(line.startswith("dump ") for line in between)


def test_the_equation_of_state_deck_averages_instead_of_dumping():
    got = _rendered("eos", "NPT", dump_every_ps=None,
                    dump_from="none", n_atoms=512)
    assert not any(line.startswith("dump ") for line in got)
    assert any(line.startswith("fix sample all ave/time") for line in got)


# ---------------------------------------------------------------------------
# A deck the user brought
# ---------------------------------------------------------------------------


def test_a_deck_from_a_file_is_filled_and_nothing_else():
    deck = from_text("units lj\ntimestep @DT@\nrun @N_PROD@")
    out = fill(deck, _point(), _system())
    assert out == "units lj\ntimestep 0.0002\nrun 150000"


def test_a_deck_with_no_placeholders_comes_back_unchanged():
    text = "units lj\nrun 100"
    assert fill(from_text(text), _point(), _system()) == text


def test_a_user_deck_reports_the_placeholders_it_uses(tmp_path):
    path = tmp_path / "in.mine.lammps"
    path.write_text("timestep @DT@\nrun @N_PROD@\ndump d all custom @N_DUMP@ f")
    deck = from_file(path)
    assert deck.placeholders == ("DT", "N_PROD", "N_DUMP")
    assert deck.source == str(path)


def test_a_user_deck_is_read_as_a_preparation_deck_when_it_asks_for_one():
    """Inferred from what it uses, not declared. A deck asking for an
    equilibration length is a deck with an equilibration stage."""
    assert from_text("run @N_EQUIL@").prepares
    assert not from_text("run @N_PROD@").prepares
    with pytest.raises(DeckError):
        check(from_text("run @N_EQUIL@"), _point(equil_ps=0.0, parent="p"))


def test_a_missing_deck_file_is_refused_naming_the_path(tmp_path):
    with pytest.raises(DeckError) as exc:
        from_file(tmp_path / "nowhere.lammps")
    assert "nowhere.lammps" in str(exc.value)


def test_a_directory_is_not_a_deck(tmp_path):
    with pytest.raises(DeckError):
        from_file(tmp_path)


def test_a_deck_is_frozen():
    import dataclasses

    deck = from_text("run 1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        deck.text = "run 2"


def test_the_package_exports_the_deck_vocabulary():
    import aipf.md as md

    for name in ("Deck", "DeckError", "builtin_deck", "fill_deck"):
        assert hasattr(md, name), name
        assert name in md.__all__, name


def test_a_deck_is_a_deck_whatever_it_came_from():
    assert isinstance(builtin("cube", "NPT"), Deck)
    assert isinstance(from_text("run 1"), Deck)


# ---------------------------------------------------------------------------
# Added after the mutation sweep, each for a mutant nothing above could kill
# ---------------------------------------------------------------------------


def test_a_placeholder_needs_both_of_its_delimiters():
    """Mutation: making the opening marker optional. A deck or a caller value
    may legitimately carry an upper-case word followed by an at sign -- a host
    in a path is the obvious one -- and reading that as a hole would refuse a
    deck for a word it was never asked about."""
    assert placeholders("@ONE@ TWO@") == ("ONE",)
    assert placeholders("@ONE ONE@") == ()


def test_a_deck_melts_exactly_when_it_asks_for_a_preparation_temperature():
    """Mutation: reading the melt flag off the wrong placeholder. The measured
    rule is that the over-temperature preparation belongs to the isotropic
    barostat and not to the geometry, so the one-axis deck settles at its own
    target and says so by using the target's placeholder."""
    assert builtin("cube", "NPT").melts
    assert builtin("eos", "NPT").melts
    assert not builtin("slab", "NPT_z").melts
    assert not builtin("vext", "NVT").melts
    assert not from_text("velocity all create @T@ @SEED@").melts


def test_a_deck_that_writes_its_own_dump_line_still_writes_a_trajectory():
    """Mutation: recognising only the generated slots. A deck someone brought
    writes its own dump command and asks only for the cadence, and it is just
    as much a deck that needs one."""
    deck = from_text("dump d all custom @N_DUMP@ traj.lammpstrj id type x y z")
    assert deck.writes_trajectory
    with pytest.raises(DeckError):
        check(deck, _point(dump_every_ps=None, dump_from="none"))


def test_something_that_is_not_text_is_not_a_deck():
    """Mutation: dropping the type check. Bytes survive as far as the join,
    and the message there names neither the deck nor what was wrong with it."""
    with pytest.raises(DeckError) as exc:
        from_text(b"units lj")
    assert "bytes" in str(exc.value)


def test_a_refusal_names_the_deck_it_came_from(tmp_path):
    """Mutation: dropping the file's own name. A campaign fills many decks and
    a message that says only "deck" leaves the reader to find which."""
    path = tmp_path / "in.mine.lammps"
    path.write_text("timestep @DT@\nrun @NOWHERE@")
    with pytest.raises(DeckError) as exc:
        fill(from_file(path), _point(), _system())
    assert "deck 'in.mine.lammps'" in str(exc.value)
    with pytest.raises(DeckError) as anonymous:
        render(from_text("run @NOWHERE@"), {})
    assert "deck 'deck'" in str(anonymous.value)


def test_a_missing_deck_and_a_directory_get_different_sentences(tmp_path):
    """Mutation: letting one check answer for both. Pointing at a folder and
    pointing at a file that was never written are different bugs."""
    with pytest.raises(DeckError) as missing:
        from_file(tmp_path / "nowhere.lammps")
    with pytest.raises(DeckError) as folder:
        from_file(tmp_path)
    assert "no deck at" in str(missing.value)
    assert "not a file" in str(folder.value)
    assert str(missing.value) != str(folder.value)


def test_a_given_value_beats_the_decks_own_default():
    """Mutation: reversing the precedence. The deck's defaults are the floor,
    so a campaign that has to change one does it in the call rather than by
    editing a deck."""
    deck = Deck(name="d", text="@A@", source="<test>", defaults={"A": "floor"})
    assert render(deck, {}) == "floor"
    assert render(deck, {"A": "given"}) == "given"


def test_a_preparation_that_lands_exactly_at_zero_is_refused():
    """Mutation: a strict inequality. Absolute zero is not a temperature the
    engine can draw velocities at, and it is the boundary case an offset that
    cancels the target lands on exactly."""
    system = _system(preparation=PreparationRules(ramp_offset=-7000.0))
    with pytest.raises(DeckError) as exc:
        values_from(_point(T=7000.0), system)
    assert "0.0" in str(exc.value)


def test_an_equilibration_timestep_that_is_not_positive_is_refused():
    """Mutation: dropping the check. Zero divides, and the resulting message
    names neither the argument nor the deck."""
    with pytest.raises(DeckError) as exc:
        values_from(_point(), _system(), dt_equil_ps=0.0)
    assert "dt_equil_ps" in str(exc.value)


def test_a_system_that_declares_masses_but_no_numeric_types_is_refused():
    """Found by the mutation sweep, not by design. The two declarations are
    checked separately where they are declared, so this combination is legal
    there; here it means a deck with no mass command at all, which the engine
    rejects at setup."""
    system = _system(masses={"one": 1.0, "two": 2.0}, atom_types={})
    with pytest.raises(DeckError) as exc:
        values_from(_point(), system)
    assert "numeric types" in str(exc.value)


def test_filling_a_deck_refuses_what_checking_it_would():
    """Mutation: dropping the check from the filling. Everything a campaign
    actually calls goes through this one function."""
    with pytest.raises(DeckError):
        fill(builtin("cube", "NPT"), _point(equil_ps=0.0, parent="p"),
             _system(), CALLER)


def test_a_given_value_beats_the_derived_one():
    """Mutation: reversing the precedence in the filling. The archived runs
    used a velocity seed hardcoded in their worker rather than the one their
    manifest carried, so reproducing one means overriding a derived value."""
    out = fill(from_text("velocity all create @T@ @SEED@"), _point(),
               _system(), {"SEED": "498459"})
    assert out == "velocity all create 7000.0 498459"


def test_the_external_field_deck_leaves_the_field_to_the_caller():
    """Mutation: dropping the field. Its shape, amplitude and periodicity were
    recorded nowhere but in the deck that applied them, so they are neither
    the request's nor the system's -- but the deck has to have somewhere to
    put them."""
    deck = builtin("vext", "NVT")
    assert "EXTERNAL" in deck.placeholders
    out = fill(deck, _point(geometry="vext", ensemble="NVT", P=None),
               _system(), CALLER)
    assert CALLER["EXTERNAL"] in out


@pytest.mark.parametrize("geometry,ensemble", [
    ("cube", "NPT"), ("slab", "NPT_z"), ("eos", "NPT"),
    ("slab", "langevin_overdamped"), ("vext", "NVT"),
])
def test_the_deck_a_request_gets_is_the_one_it_asked_for(geometry, ensemble):
    """Mutation: answering with one deck whatever was asked."""
    point = _point(
        geometry=geometry, ensemble=ensemble,
        P=800.0 if ensemble in ("NPT", "NPT_z") else None,
        equil_ps=1.5 if ensemble != "langevin_overdamped" else 0.0,
        dump_every_ps=None if geometry == "eos" else 0.02,
        dump_from="none" if geometry == "eos" else "prod",
    )
    assert for_point(point) is builtin(geometry, ensemble)
