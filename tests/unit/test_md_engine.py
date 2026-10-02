"""Tests for `aipf.md.engine`: the LAMMPS deck, the pair style and the run interface.

Three tiers, on purpose.

The default tier is everything that can be decided without the engine: the
route vocabulary, the pair-style spellings, the accelerator combination, the
environment, and the whole of the input-file route driven against a real
subprocess that is not the engine.

A small fake stands in for the engine's Python bindings in exactly one place:
the in-process route's CALL ORDER, which is the one load-bearing thing a
successful real run cannot demonstrate (a wrong order simply fails). It is
not a substitute for running the real thing.

The ``env`` tier runs the real engine, in the environment its build was
linked against, on both routes. Those tests are marked and skipped where the
engine is absent, because the package's own environment carries no bindings
at all -- the engine's build installs them into one environment and that is
not this one.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import declared_roots
from aipf.md import engine
from aipf.md.engine import (
    BROKEN_STYLES,
    KOKKOS_PACKAGE,
    REMOVED_FROM_ENVIRONMENT,
    ROUTES,
    UNIFIED_TOKEN,
    EngineError,
    Launch,
    Result,
    check_deck,
    clean_environment,
    kokkos_args,
    normalise_route,
    pair_commands,
    plan,
    run,
    strip_environment,
    unavailable_reason,
)
from aipf.md.potentials import Potential

# ---------------------------------------------------------------------------
# The real engine, where it lives on this machine: two site facts
# (aipf.site), the simulator's own interpreter ``AIPF_LAMMPS_PYTHON`` and its
# machine-learning build ``AIPF_LAMMPS``. The archived model and starting
# configuration are under the hydrogen-helium raw root. Every one of these is
# read only.
# ---------------------------------------------------------------------------

_ENGINE_PYTHON = declared_roots.site_fact("lammps_python")
_ENGINE_LIB = _ENGINE_PYTHON.parent.parent / "lib"
_ENGINE_BINARY = declared_roots.site_fact("lammps")
_REAL_MODEL = declared_roots.raw(
    "hhe", "builds/mliap/hhe_train_ist_swa-mliap_lammps_cueq_fp32.pt")
#: The first initial configuration of one archived equation-of-state point.
_REAL_CONFIG = next(iter(sorted(declared_roots.raw("hhe", "eos_800GPa", "x0.40").glob("init_x0.40_*.data"))),
                    declared_roots.raw("hhe", "eos_800GPa", "x0.40", "init.data"))
_PACKAGE_SRC = Path(__file__).resolve().parents[2] / "src"

_requires_engine_env = pytest.mark.skipif(
    not _ENGINE_PYTHON.is_file(),
    reason=f"the engine's own environment is not at {_ENGINE_PYTHON} (AIPF_LAMMPS_PYTHON or [site] lammps_python)")
def _engine_binary_or_skip():
    """The declared LAMMPS, when it is an ML-IAP build with the accelerated pair style; else a skip naming it."""
    declared_roots.lammps_or_skip(declared_roots.MLIAP_KK)


def _requires_engine_binary(test):
    """Decorator: :func:`_engine_binary_or_skip` at test time."""
    import functools

    @functools.wraps(test)
    def run(*args, **kwargs):
        _engine_binary_or_skip()
        return test(*args, **kwargs)
    return run
_requires_real_model = pytest.mark.skipif(
    not (_REAL_MODEL.is_file() and _REAL_CONFIG.is_file()),
    reason=f"the archived model {_REAL_MODEL} or starting configuration {_REAL_CONFIG} is not on "
           f"disk (the hhe raw root: AIPF_RAW_HHE, AIPF_RAW or [paths.raw] in aipf.toml)")


def _engine_environment() -> dict:
    """The environment the engine's bindings need, loader order first.

    Measured on this machine: with the session's inherited search path the
    bindings' shared library resolves a system library from a compiler
    toolchain and fails to load with an undefined symbol. Putting the
    environment's own directory first fixes it, and that is the same "loader
    order, not node type" rule the package's environment tests already pin.
    """
    env = dict(os.environ)
    inherited = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(_ENGINE_LIB / "python3.12" / "site-packages" / "nvidia"
             / "cuda_runtime" / "lib"), str(_ENGINE_LIB), inherited])
    env["PYTHONPATH"] = str(_PACKAGE_SRC)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTEST_CURRENT_TEST", None)
    return env


def _potential(species=("A", "B"), path="/nowhere/model.pt", model=None):
    """A resolved-potential record, without resolving one.

    Element names are placeholders. The module never reads them as anything
    but labels, and naming real ones here would say something about a system
    that this test file has no business knowing.
    """
    return Potential(path=Path(path), sha256="0" * 64, engine="mace-mliap",
                     species=tuple(species), dtype="float32", cutoff=3.0,
                     fallback_kernels=(), model=model)


# ---------------------------------------------------------------------------
# The route vocabulary
# ---------------------------------------------------------------------------


def test_both_measured_routes_are_in_the_vocabulary():
    """Two routes, because two complete campaigns ran, one on each."""
    assert ROUTES == {"in_process", "input_file"}


@pytest.mark.parametrize("name", ["in_process", "input_file"])
def test_a_canonical_route_name_passes_through(name):
    assert normalise_route(name) == name


@pytest.mark.parametrize("given,wanted", [
    ("in-process", "in_process"),
    ("in process", "in_process"),
    ("python", "in_process"),
    ("library", "in_process"),
    ("input-file", "input_file"),
    ("deck", "input_file"),
    ("subprocess", "input_file"),
    ("  in_process  ", "in_process"),
])
def test_every_accepted_spelling_of_a_route(given, wanted):
    assert normalise_route(given) == wanted


def test_an_unknown_route_is_refused_by_name_and_lists_what_exists():
    with pytest.raises(EngineError) as caught:
        normalise_route("mpi")
    message = str(caught.value)
    assert "'mpi'" in message
    assert "in_process" in message and "input_file" in message


def test_the_refusal_also_lists_the_spellings_it_accepts():
    """Two lists, and a caller who wrote one of the accepted spellings wrong
    needs the second one, not the first.

    The name asked about must share no substring with an accepted spelling:
    asking about "pythonic" and looking for "python" finds it inside the
    quoted name the refusal already carries, and proves nothing.
    """
    with pytest.raises(EngineError) as caught:
        normalise_route("mpi")
    assert "python" in str(caught.value)


def test_a_space_becomes_the_separator_and_does_not_vanish():
    """The replacement joins two words. It does not delete a character, so a
    name that is already separated and then also spaced is a typo."""
    with pytest.raises(EngineError):
        normalise_route("in_ process")


def test_a_route_that_is_not_a_string_is_refused_by_type():
    with pytest.raises(EngineError) as caught:
        normalise_route(2)
    assert "int" in str(caught.value)


# ---------------------------------------------------------------------------
# The environment that must not be inherited
# ---------------------------------------------------------------------------


def test_the_allocator_setting_is_the_variable_that_is_removed():
    assert REMOVED_FROM_ENVIRONMENT == ("PYTORCH_CUDA_ALLOC_CONF",)


def test_cleaning_removes_the_variable_and_says_what_it_removed():
    """Stripped is not enough: a run that behaves differently from the shell
    it was launched from must be able to say which setting was dropped."""
    given = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
             "CUDA_VISIBLE_DEVICES": "0"}
    kept, removed = clean_environment(given)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in kept
    assert kept == {"CUDA_VISIBLE_DEVICES": "0"}
    assert removed == (("PYTORCH_CUDA_ALLOC_CONF",
                        "expandable_segments:True"),)


def test_cleaning_does_not_touch_the_environment_it_was_given():
    given = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    clean_environment(given)
    assert given == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def test_cleaning_an_environment_that_never_set_it_removes_nothing():
    kept, removed = clean_environment({"HOME": "/somewhere"})
    assert kept == {"HOME": "/somewhere"}
    assert removed == ()


def test_the_variable_is_removed_whatever_its_value():
    """By name, not by value. The setting is a list and a partial edit is a
    worse answer than the library's own default."""
    _, removed = clean_environment({"PYTORCH_CUDA_ALLOC_CONF":
                                    "max_split_size_mb:128"})
    assert removed == (("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128"),)


def test_stripping_a_live_environment_removes_it_in_place():
    """The in-process route has no child to hand a clean environment to."""
    target = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
              "PATH": "/usr/bin"}
    removed = strip_environment(target)
    assert target == {"PATH": "/usr/bin"}
    assert removed == (("PYTORCH_CUDA_ALLOC_CONF",
                        "expandable_segments:True"),)


def test_stripping_defaults_to_this_process(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    removed = strip_environment()
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert removed == (("PYTORCH_CUDA_ALLOC_CONF",
                        "expandable_segments:True"),)


def test_stripping_an_environment_without_it_is_a_no_op():
    target = {"PATH": "/usr/bin"}
    assert strip_environment(target) == ()
    assert target == {"PATH": "/usr/bin"}


def test_a_variable_nobody_removes_has_no_reason_to_give():
    """An answer invented for a name this module never removes would read as
    though it did."""
    assert engine.removal_reason("PATH") == ""


def test_every_removed_variable_carries_a_reason():
    assert REMOVED_FROM_ENVIRONMENT, "nothing is removed at all"
    for name in REMOVED_FROM_ENVIRONMENT:
        assert engine.removal_reason(name), name


# ---------------------------------------------------------------------------
# The accelerator arguments
# ---------------------------------------------------------------------------


def test_the_accelerator_package_settings_are_the_one_pair_that_runs():
    assert KOKKOS_PACKAGE == ("newton", "on", "neigh", "half")


def test_the_arguments_enable_the_accelerator_and_apply_its_suffix():
    """Two separate facts, and measured: each without the other fails."""
    assert kokkos_args() == ("-k", "on", "g", "1", "-sf", "kk",
                             "-pk", "kokkos", "newton", "on", "neigh", "half")


def test_the_device_count_reaches_the_arguments():
    assert kokkos_args(gpus=2)[:4] == ("-k", "on", "g", "2")


@pytest.mark.parametrize("bad", [0, -1])
def test_fewer_than_one_device_is_refused(bad):
    with pytest.raises(EngineError) as caught:
        kokkos_args(gpus=bad)
    assert str(bad) in str(caught.value)


@pytest.mark.parametrize("bad", [1.0, "1", True, None])
def test_a_device_count_that_is_not_a_whole_number_is_refused(bad):
    with pytest.raises(EngineError):
        kokkos_args(gpus=bad)


def test_host_threads_put_the_accelerator_on_the_host():
    """A host-only build: the same suffix and package line, threads instead of devices."""
    assert kokkos_args(gpus=None, threads=2) == ("-k", "on", "t", "2", "-sf", "kk",
                                                 "-pk", "kokkos", "newton", "on", "neigh", "half")
    with pytest.raises(EngineError):
        kokkos_args(gpus=None, threads=0)


def test_a_model_launch_on_host_threads_is_refused_and_a_pair_launch_is_not():
    """The model route runs on a device only; host threads serve the pair-potential decks."""
    with pytest.raises(EngineError, match="needs a device"):
        plan("in_process", gpus=None, threads=1)
    launch = plan("in_process", gpus=None, threads=1, uses_model=False)
    assert launch.accelerated and ("-k", "on", "t", "1") == launch.args[2:6]


# ---------------------------------------------------------------------------
# The pair style, which is where the two routes differ
# ---------------------------------------------------------------------------


def test_in_process_the_style_carries_the_literal_token_and_no_path():
    """Measured spelling one. The interface already holds the model."""
    text = pair_commands(_potential(), route="in_process")
    assert text.splitlines()[0].split() == [
        "pair_style", "mliap/kk", "unified", "EXISTS"]


def test_from_an_input_file_the_style_carries_the_path_and_a_flag():
    """Measured spelling two, and the flag is a boolean, which is exactly why
    the composite in the design notes is a parse error."""
    text = pair_commands(_potential(path="/models/m.pt"), route="input_file")
    assert text.splitlines()[0].split() == [
        "pair_style", "mliap", "unified", "/models/m.pt", "0"]


def test_the_ghost_neighbour_flag_can_be_asked_for():
    text = pair_commands(_potential(path="/models/m.pt"), route="input_file",
                         ghost_neighbours=True)
    assert text.splitlines()[0].split()[-1] == "1"


def test_the_ghost_neighbour_flag_is_refused_where_there_is_no_path():
    with pytest.raises(EngineError) as caught:
        pair_commands(_potential(), route="in_process",
                      ghost_neighbours=True)
    assert "in-process" in str(caught.value)


def test_the_suffix_defaults_to_whichever_the_route_archived():
    """In process the style is spelled accelerated, from an input file it is
    not and the command line applies the suffix. Both are the same style."""
    inside = pair_commands(_potential(), route="in_process")
    outside = pair_commands(_potential(), route="input_file")
    assert inside.startswith("pair_style      mliap/kk")
    assert outside.startswith("pair_style      mliap ")


def test_the_suffix_can_be_asked_for_or_refused_explicitly():
    plain = pair_commands(_potential(), route="in_process",
                          accelerated=False)
    assert plain.startswith("pair_style      mliap unified")
    suffixed = pair_commands(_potential(path="/models/m.pt"),
                             route="input_file", accelerated=True)
    assert suffixed.startswith("pair_style      mliap/kk unified /models/m.pt")


def test_the_coefficients_are_written_in_the_models_own_order():
    text = pair_commands(_potential(species=("A", "B", "C")),
                         route="in_process")
    assert text.splitlines()[1].split() == [
        "pair_coeff", "*", "*", "A", "B", "C"]


def test_the_other_order_is_a_different_line():
    """The same set in the other order is the wrong physics, silently, which
    is why the record's order is the one written."""
    one = pair_commands(_potential(species=("A", "B")), route="in_process")
    other = pair_commands(_potential(species=("B", "A")), route="in_process")
    assert one != other


def test_a_potential_that_declares_no_elements_is_refused():
    with pytest.raises(EngineError) as caught:
        pair_commands(_potential(species=()), route="in_process")
    assert "no elements" in str(caught.value)


def test_pair_commands_needs_a_resolved_potential():
    with pytest.raises(EngineError) as caught:
        pair_commands("/models/model.pt", route="in_process")
    assert "str" in str(caught.value)


def test_pair_commands_refuses_an_unknown_route():
    with pytest.raises(EngineError):
        pair_commands(_potential(), route="mpi")


# ---------------------------------------------------------------------------
# Decks that cannot run, checked before the queue wait
# ---------------------------------------------------------------------------


def test_the_broken_accelerated_style_is_named_and_refused():
    assert "mace/kk" in BROKEN_STYLES
    with pytest.raises(EngineError) as caught:
        check_deck("pair_style mace/kk\npair_coeff * *",
                   route="in_process")
    message = str(caught.value)
    assert "mace/kk" in message
    assert "newton" in message


def test_the_composite_spelling_from_the_design_notes_is_refused():
    """A path AND the token. It is in this project's own spec and in a
    reproduction guide, and the engine rejects it: the argument after the
    path is a boolean flag and the token is not one."""
    with pytest.raises(EngineError) as caught:
        check_deck(f"pair_style mliap/kk unified /models/m.pt {UNIFIED_TOKEN}",
                   route="in_process")
    message = str(caught.value)
    assert UNIFIED_TOKEN in message
    assert "either working spelling" in message


def test_the_composite_is_refused_on_the_other_route_too():
    with pytest.raises(EngineError):
        check_deck(f"pair_style mliap unified /models/m.pt {UNIFIED_TOKEN}",
                   route="input_file")


def test_the_token_is_refused_where_nothing_could_have_loaded_a_model():
    with pytest.raises(EngineError) as caught:
        check_deck(f"pair_style mliap unified {UNIFIED_TOKEN}",
                   route="input_file")
    assert "loaded a model" in str(caught.value)


def test_a_model_path_is_refused_on_the_in_process_route():
    """The engine ACCEPTS this line and the deck's model wins, measured. So
    the record would name a file the run never used."""
    with pytest.raises(EngineError) as caught:
        check_deck("pair_style mliap/kk unified /models/other.pt 0",
                   route="in_process")
    message = str(caught.value)
    assert "would NOT fail" in message
    assert "discarded" in message or "never used" in message


def test_each_working_spelling_passes_on_its_own_route():
    check_deck(f"pair_style mliap/kk unified {UNIFIED_TOKEN}\n"
               f"pair_coeff * * A B", route="in_process")
    check_deck("pair_style mliap unified /models/m.pt 0\n"
               "pair_coeff * * A B", route="input_file")


def test_a_deck_with_no_machine_learned_style_is_left_alone():
    """A whole tree's campaign runs a plain pair potential and this check has
    nothing to say about it."""
    check_deck("pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0",
               route="input_file")


def test_a_deck_that_never_names_a_pair_style_is_left_alone():
    check_deck("units metal\nrun 100", route="in_process")


def test_a_line_that_merely_mentions_the_style_is_not_a_style_line():
    """Only the first word decides. A comment or a variable holding the same
    words is not a command."""
    check_deck("# pair_style mliap unified /models/m.pt 0\n"
               f"pair_style mliap/kk unified {UNIFIED_TOKEN}",
               route="in_process")


def test_a_commented_line_written_without_a_space_is_still_a_comment():
    """Only a line whose FIRST word is the command is a command."""
    check_deck("#pair_style mace/kk\n"
               f"pair_style mliap/kk unified {UNIFIED_TOKEN}",
               route="in_process")


def test_the_composite_refusal_quotes_the_line_it_found():
    with pytest.raises(EngineError) as caught:
        check_deck("pair_style      mliap/kk unified /models/m.pt EXISTS",
                   route="in_process")
    assert "'pair_style      mliap/kk unified /models/m.pt EXISTS'" in \
        str(caught.value)


def test_a_style_line_with_nothing_after_it_is_not_this_checks_business():
    check_deck("pair_style", route="in_process")


def test_a_unified_style_with_no_arguments_is_left_to_the_engine():
    check_deck("pair_style mliap/kk unified", route="in_process")


def test_a_non_unified_machine_learned_style_is_left_alone():
    check_deck("pair_style mliap model linear descriptor sna",
               route="in_process")


def test_a_deck_that_is_not_text_is_refused_by_type():
    with pytest.raises(EngineError) as caught:
        check_deck(["pair_style mliap/kk unified EXISTS"], route="in_process")
    assert "list" in str(caught.value)


def test_check_deck_refuses_an_unknown_route():
    with pytest.raises(EngineError):
        check_deck("units metal", route="mpi")


# ---------------------------------------------------------------------------
# One resolved launch
# ---------------------------------------------------------------------------


def _model_launch(**kw):
    kw.setdefault("route", "in_process")
    kw.setdefault("args", kokkos_args())
    kw.setdefault("uses_model", True)
    return Launch(**kw)


def test_a_launch_normalises_its_own_route():
    assert _model_launch(route="python").route == "in_process"


def test_a_launch_keeps_its_arguments_as_text():
    launch = Launch(route="in_process", args=["-screen", "none"],
                    uses_model=False)
    assert launch.args == ("-screen", "none")


def test_an_argument_that_is_not_text_becomes_text():
    """A device count read out of a configuration file arrives as a number,
    and the engine is handed a list of strings."""
    launch = Launch(route="in_process", uses_model=False,
                    args=("-k", "on", "g", 2))
    assert launch.args == ("-k", "on", "g", "2")


@pytest.mark.parametrize("newton,neigh", [
    ("off", "half"), ("on", "full"), ("off", "full")])
def test_every_other_accelerator_corner_is_refused(newton, neigh):
    """Measured against the real engine: each of the three fails, and each
    fails somewhere different."""
    with pytest.raises(EngineError) as caught:
        Launch(route="in_process", uses_model=False,
               args=("-k", "on", "-sf", "kk", "-pk", "kokkos",
                     "newton", newton, "neigh", neigh))
    assert "newton on neigh half" in str(caught.value)


def test_the_refusal_says_what_that_corner_actually_does():
    with pytest.raises(EngineError) as caught:
        Launch(route="in_process", uses_model=False,
               args=("-k", "on", "-sf", "kk", "-pk", "kokkos",
                     "newton", "on", "neigh", "full"))
    assert "package line itself" in str(caught.value)


def test_an_accelerator_package_line_missing_a_setting_is_refused():
    with pytest.raises(EngineError) as caught:
        Launch(route="in_process", uses_model=False,
               args=("-k", "on", "-sf", "kk", "-pk", "kokkos",
                     "newton", "on"))
    assert "neigh" in str(caught.value)


def test_a_repeated_setting_takes_its_last_value():
    """Measured against the real engine: `newton on ... newton off` fails
    with the complaint newton off earns, and the other order runs. Reading
    the first value would pass a line the engine then refuses, after the
    queue wait."""
    with pytest.raises(EngineError):
        Launch(route="in_process", uses_model=False,
               args=("-k", "on", "-sf", "kk", "-pk", "kokkos",
                     "newton", "on", "neigh", "half", "newton", "off"))
    Launch(route="in_process", uses_model=False,
           args=("-k", "on", "-sf", "kk", "-pk", "kokkos",
                 "newton", "off", "neigh", "half", "newton", "on"))


def test_the_settings_are_read_as_pairs_from_the_start_of_the_line():
    """A keyword sitting at an odd offset is not one. Reading overlapping
    pairs would find `newton on` inside a line whose first keyword has no
    value at all, and pass a line the engine rejects outright."""
    with pytest.raises(EngineError) as caught:
        Launch(route="in_process", uses_model=False,
               args=("-k", "on", "-sf", "kk", "-pk", "kokkos",
                     "binsize", "newton", "on", "neigh", "half"))
    assert "sets no" in str(caught.value)


def test_a_missing_setting_is_a_different_sentence_from_a_wrong_one():
    """A line that never mentions a setting has a different repair from one
    that sets it to the other value."""
    with pytest.raises(EngineError) as caught:
        Launch(route="in_process", uses_model=False,
               args=("-k", "on", "-sf", "kk", "-pk", "kokkos",
                     "neigh", "half"))
    assert "sets no 'newton'" in str(caught.value)


def test_a_package_line_for_something_else_is_not_this_checks_business():
    Launch(route="in_process", uses_model=False,
           args=("-pk", "gpu", "newton", "off"))


def test_a_bare_package_flag_with_nothing_after_it_is_left_alone():
    Launch(route="in_process", uses_model=False, args=("-pk",))


def test_a_model_run_without_the_suffix_is_refused():
    with pytest.raises(EngineError) as caught:
        _model_launch(args=("-k", "on", "g", "1", "-pk", "kokkos",
                            "newton", "on", "neigh", "half"))
    assert "accelerator suffix" in str(caught.value)


def test_a_model_run_with_the_suffix_but_no_accelerator_is_refused():
    """Measured: asking for the suffix without enabling it stops at the first
    style the deck names."""
    with pytest.raises(EngineError):
        _model_launch(args=("-sf", "kk"))


def test_the_accelerator_has_to_be_turned_on_rather_than_named():
    with pytest.raises(EngineError):
        _model_launch(args=("-k", "off", "-sf", "kk"))


def test_another_suffix_is_not_this_accelerators():
    with pytest.raises(EngineError):
        _model_launch(args=("-k", "on", "-sf", "gpu"))


def test_a_run_with_no_model_needs_no_accelerator_at_all():
    """A whole tree ran a plain pair potential under a parallel launcher."""
    launch = Launch(route="input_file", args=("-in", "in.lammps"),
                    deck_path="in.lammps", uses_model=False)
    assert launch.accelerated is False


def test_the_input_file_route_needs_a_deck_file():
    with pytest.raises(EngineError) as caught:
        Launch(route="input_file", args=(), uses_model=False)
    assert "deck" in str(caught.value)


def test_the_in_process_route_refuses_a_deck_file():
    with pytest.raises(EngineError) as caught:
        Launch(route="in_process", args=(), uses_model=False,
               deck_path="in.lammps")
    assert "in.lammps" in str(caught.value)


def test_a_launch_builds_the_whole_command():
    launch = Launch(route="input_file", args=("-log", "lmp.log", "-in",
                                              "in.lammps"),
                    deck_path="in.lammps", uses_model=False)
    assert launch.argv("/opt/lmp") == ("/opt/lmp", "-log", "lmp.log",
                                       "-in", "in.lammps")


def test_a_launcher_goes_in_front_of_the_executable():
    """One tree's whole campaign ran under a parallel launcher, whose name
    and arguments are the machine's and not this package's."""
    launch = Launch(route="input_file", args=("-in", "in.lammps"),
                    deck_path="in.lammps", uses_model=False)
    assert launch.argv("/opt/lmp", ["mpirun", "-np", "4"])[:3] == (
        "mpirun", "-np", "4")


def test_the_command_is_one_quoted_line():
    launch = Launch(route="input_file", args=("-in", "a deck.lammps"),
                    deck_path="a deck.lammps", uses_model=False)
    assert "'a deck.lammps'" in launch.command("/opt/lmp")


def test_the_in_process_route_has_no_command_line_to_build():
    with pytest.raises(EngineError) as caught:
        _model_launch().argv("/opt/lmp")
    assert "this interpreter" in str(caught.value)


# ---------------------------------------------------------------------------
# Planning a launch
# ---------------------------------------------------------------------------


def test_a_planned_in_process_launch_is_the_archived_command_line():
    launch = plan("in_process", log="run.log")
    assert launch.args == ("-screen", "none", "-log", "run.log",
                           "-k", "on", "g", "1", "-sf", "kk",
                           "-pk", "kokkos", "newton", "on", "neigh", "half")


def test_a_planned_input_file_launch_ends_with_its_deck():
    launch = plan("input_file", log="lmp.log", deck_path="in.lammps")
    assert launch.args[-2:] == ("-in", "in.lammps")
    assert launch.deck_path == "in.lammps"


def test_the_screen_is_silenced_by_default_and_can_be_asked_for():
    assert "-screen" in plan("in_process").args
    assert "-screen" not in plan("in_process", screen=None).args


def test_a_launch_with_no_log_names_none():
    assert "-log" not in plan("in_process", screen=None).args


def test_command_line_variables_are_passed_in_the_engines_own_form():
    """A whole tree's decks are written against these, with their own
    defaults, and overriding them is how that tree parameterised a run."""
    launch = plan("input_file", deck_path="in.lammps", screen=None, gpus=None,
                  uses_model=False, variables={"T": 1.6, "seed": 42})
    assert launch.args == ("-var", "T", "1.6", "-var", "seed", "42",
                           "-in", "in.lammps")


def test_a_variable_with_no_value_is_refused():
    """A command-line variable with nothing after it eats the next argument
    instead, and the run starts on a deck it was not given."""
    with pytest.raises(EngineError) as caught:
        plan("input_file", deck_path="in.lammps", gpus=None,
             uses_model=False, variables={"T": None})
    assert "'T'" in str(caught.value)


@pytest.mark.parametrize("bad", ["", 3])
def test_a_variable_that_is_not_named_by_a_string_is_refused(bad):
    with pytest.raises(EngineError):
        plan("input_file", deck_path="in.lammps", gpus=None,
             uses_model=False, variables={bad: 1})


def test_a_build_with_no_accelerator_is_a_launch_with_none():
    launch = plan("input_file", deck_path="in.lammps", screen=None,
                  gpus=None, uses_model=False)
    assert launch.args == ("-in", "in.lammps")


def test_a_model_run_with_no_device_is_refused():
    """There is no unaccelerated path for a model: its own code calls an
    exchange that only the accelerated coupling defines."""
    with pytest.raises(EngineError) as caught:
        plan("in_process", gpus=None)
    assert "unaccelerated" in str(caught.value)


def test_planning_the_input_file_route_without_a_deck_is_refused():
    with pytest.raises(EngineError) as caught:
        plan("input_file")
    assert "deck" in str(caught.value)


def test_planning_the_in_process_route_with_a_deck_is_refused():
    with pytest.raises(EngineError) as caught:
        plan("in_process", deck_path="in.lammps")
    assert "in.lammps" in str(caught.value)


def test_a_planned_child_environment_loses_the_variable_and_records_it():
    launch = plan("input_file", deck_path="in.lammps",
                  env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                       "PATH": "/usr/bin"})
    assert launch.env == {"PATH": "/usr/bin"}
    assert launch.removed == (("PYTORCH_CUDA_ALLOC_CONF",
                               "expandable_segments:True"),)


def test_a_planned_child_environment_defaults_to_this_process(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("AIPF_TASK4_MARKER", "kept")
    launch = plan("input_file", deck_path="in.lammps")
    assert "PYTORCH_CUDA_ALLOC_CONF" not in launch.env
    assert launch.env["AIPF_TASK4_MARKER"] == "kept"


def test_the_in_process_route_has_no_child_to_hand_an_environment_to():
    with pytest.raises(EngineError) as caught:
        plan("in_process", env={"PATH": "/usr/bin"})
    assert "strip_environment" in str(caught.value)


def test_an_in_process_launch_carries_an_empty_environment():
    launch = plan("in_process")
    assert launch.env == {} and launch.removed == ()


# ---------------------------------------------------------------------------
# What came back
# ---------------------------------------------------------------------------


def test_a_finished_run_is_ok_and_an_unfinished_one_is_not():
    assert Result(status="ok", route="in_process", seconds=1.0).ok
    assert not Result(status="error: boom", route="in_process",
                      seconds=1.0).ok


def test_the_result_fills_only_the_metadata_the_run_could_know():
    result = Result(status="ok", route="in_process", seconds=1.0,
                    n_atoms=4560)
    assert result.to_meta_fields() == {"status": "ok", "n_atoms": 4560}


def test_an_atom_count_nobody_counted_is_absent_rather_than_faked():
    """One archived geometry writes no trajectory and reports no count."""
    result = Result(status="ok", route="input_file", seconds=1.0)
    assert result.to_meta_fields() == {"status": "ok"}


def test_a_run_that_counted_zero_reports_zero():
    """Nobody counted and counted nothing are different answers, the same way
    a pressure of zero is a target rather than an absence."""
    result = Result(status="ok", route="in_process", seconds=1.0, n_atoms=0)
    assert result.to_meta_fields() == {"status": "ok", "n_atoms": 0}


def test_a_failure_carries_its_own_reason_into_the_record():
    result = Result(status="error: exit 1", route="input_file", seconds=1.0)
    assert result.to_meta_fields()["status"] == "error: exit 1"


# ---------------------------------------------------------------------------
# Whether this interpreter, on this machine, can run at all
# ---------------------------------------------------------------------------


def test_the_input_file_route_needs_an_executable():
    assert unavailable_reason("input_file") == (
        "no engine executable was named")


def test_an_executable_that_is_not_there_is_named(tmp_path):
    reason = unavailable_reason("input_file", binary=tmp_path / "lmp")
    assert "lmp" in reason and "no engine executable" in reason


def test_an_executable_that_cannot_be_executed_is_a_different_sentence(
        tmp_path):
    binary = tmp_path / "lmp"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o644)
    assert "not executable" in unavailable_reason("input_file",
                                                  binary=binary)


def test_an_executable_that_is_there_is_available(tmp_path):
    binary = tmp_path / "lmp"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    assert unavailable_reason("input_file", binary=binary) is None


def test_an_executable_on_the_search_path_is_found():
    assert unavailable_reason("input_file", binary="sh") is None


def test_the_in_process_route_reports_on_this_interpreter():
    """This package's own environment carries no bindings, by design: the
    engine's build installs them into the one environment it was linked
    against. The answer is a sentence or None, and either is the truth about
    THIS interpreter rather than about the machine."""
    import importlib.util

    reason = unavailable_reason("in_process")
    if importlib.util.find_spec("lammps") is None:
        assert reason is not None and "bindings" in reason
    else:
        assert reason is None


def test_unavailable_reason_refuses_an_unknown_route():
    with pytest.raises(EngineError):
        unavailable_reason("mpi")


# ---------------------------------------------------------------------------
# Running: the refusals that happen before anything starts
# ---------------------------------------------------------------------------


def test_run_takes_a_resolved_launch():
    with pytest.raises(EngineError) as caught:
        run(("-in", "in.lammps"), "units metal")
    assert "tuple" in str(caught.value)


def test_a_launch_that_says_it_runs_a_model_needs_one():
    with pytest.raises(EngineError) as caught:
        run(_model_launch(), f"pair_style mliap/kk unified {UNIFIED_TOKEN}")
    assert "none was given" in str(caught.value)


def test_a_model_handed_to_a_launch_that_loads_none_is_refused():
    """Nothing would load it, and the deck's own interactions would run
    instead, silently."""
    launch = Launch(route="input_file", args=("-in", "in.lammps"),
                    deck_path="in.lammps", uses_model=False)
    with pytest.raises(EngineError) as caught:
        run(launch, "pair_style lj/cut 2.5", potential=_potential())
    assert "silently" in str(caught.value)


def test_a_deck_that_cannot_run_is_refused_before_anything_starts():
    with pytest.raises(EngineError) as caught:
        run(_model_launch(), "pair_style mace/kk", potential=_potential())
    assert "mace/kk" in str(caught.value)


def test_the_input_file_route_needs_an_executable_to_run():
    launch = Launch(route="input_file", args=("-in", "in.lammps"),
                    deck_path="in.lammps", uses_model=False)
    with pytest.raises(EngineError) as caught:
        run(launch, "units metal")
    assert "none was named" in str(caught.value)


def test_running_an_executable_that_is_not_there_is_refused(tmp_path):
    launch = Launch(route="input_file", args=("-in", "in.lammps"),
                    deck_path="in.lammps", uses_model=False)
    with pytest.raises(EngineError) as caught:
        run(launch, "units metal", binary=tmp_path / "lmp",
            workdir=tmp_path)
    assert "cannot run here" in str(caught.value)


# ---------------------------------------------------------------------------
# Running the input-file route, against a real subprocess
# ---------------------------------------------------------------------------


def _recorder(tmp_path, body="exit 0"):
    """An executable that records how it was called. Not an engine."""
    binary = tmp_path / "fake_lmp"
    binary.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{tmp_path}/argv.txt"\n'
        f'env > "{tmp_path}/env.txt"\n'
        f'pwd > "{tmp_path}/cwd.txt"\n'
        f"{body}\n")
    binary.chmod(0o755)
    return binary


def test_the_deck_is_written_where_the_launch_says(tmp_path):
    launch = plan("input_file", deck_path="in.lammps", screen=None,
                  gpus=None, uses_model=False)
    run(launch, "units metal\nrun 0\n", binary=_recorder(tmp_path),
        workdir=tmp_path)
    assert (tmp_path / "in.lammps").read_text() == "units metal\nrun 0\n"


def test_the_deck_is_written_verbatim(tmp_path):
    """A contract, not an accident: one archived worker draws its initial
    velocities from a literal seed written into its own source, so a driver
    that substituted a seed of its own would produce a different trajectory
    and report success."""
    deck = "velocity all create 7000.0 498459 mom yes rot yes dist gaussian\n"
    launch = plan("input_file", deck_path="in.lammps", screen=None,
                  gpus=None, uses_model=False)
    run(launch, deck, binary=_recorder(tmp_path), workdir=tmp_path)
    assert (tmp_path / "in.lammps").read_text() == deck


def test_a_deck_in_a_directory_that_does_not_exist_yet_is_made(tmp_path):
    launch = plan("input_file", deck_path="runs/one/in.lammps", screen=None,
                  gpus=None, uses_model=False)
    run(launch, "units metal\n", binary=_recorder(tmp_path),
        workdir=tmp_path)
    assert (tmp_path / "runs" / "one" / "in.lammps").is_file()


def test_the_executable_is_called_with_the_launchs_arguments(tmp_path):
    launch = plan("input_file", deck_path="in.lammps", log="lmp.log",
                  gpus=None, uses_model=False)
    run(launch, "units metal\n", binary=_recorder(tmp_path),
        workdir=tmp_path)
    given = (tmp_path / "argv.txt").read_text().split()
    assert given == ["-screen", "none", "-log", "lmp.log",
                     "-in", "in.lammps"]


def test_the_child_runs_in_the_working_directory(tmp_path):
    where = tmp_path / "run"
    where.mkdir()
    launch = plan("input_file", deck_path="in.lammps", screen=None,
                  gpus=None, uses_model=False)
    run(launch, "units metal\n", binary=_recorder(tmp_path), workdir=where)
    assert Path((tmp_path / "cwd.txt").read_text().strip()).resolve() == \
        where.resolve()


def test_the_child_does_not_inherit_the_variable_that_breaks_the_run(
        tmp_path, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    launch = plan("input_file", deck_path="in.lammps", screen=None,
                  gpus=None, uses_model=False)
    run(launch, "units metal\n", binary=_recorder(tmp_path),
        workdir=tmp_path)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in (tmp_path / "env.txt").read_text()


def test_a_finished_child_is_recorded_as_ok(tmp_path):
    launch = plan("input_file", deck_path="in.lammps", screen=None,
                  gpus=None, uses_model=False)
    result = run(launch, "units metal\n", binary=_recorder(tmp_path),
                 workdir=tmp_path)
    assert result.ok and result.returncode == 0
    assert result.route == "input_file"
    assert result.seconds > 0.0


def test_a_failing_child_is_recorded_rather_than_raised(tmp_path):
    """A campaign of hundreds must not stop at the first point whose box
    blows up, which is what every archived worker does."""
    binary = _recorder(tmp_path, body='echo "bad box" >&2\nexit 3')
    launch = plan("input_file", deck_path="in.lammps", screen=None,
                  gpus=None, uses_model=False)
    result = run(launch, "units metal\n", binary=binary, workdir=tmp_path)
    assert not result.ok
    assert result.status == "error: exit 3"
    assert result.returncode == 3
    assert "bad box" in result.detail


def test_a_finished_child_carries_no_failure_detail(tmp_path):
    binary = _recorder(tmp_path, body='echo "a warning" >&2\nexit 0')
    launch = plan("input_file", deck_path="in.lammps", screen=None,
                  gpus=None, uses_model=False)
    assert run(launch, "units metal\n", binary=binary,
               workdir=tmp_path).detail is None


def test_a_child_that_overruns_is_recorded_with_its_limit(tmp_path):
    binary = _recorder(tmp_path, body="sleep 30")
    launch = plan("input_file", deck_path="in.lammps", screen=None,
                  gpus=None, uses_model=False)
    result = run(launch, "units metal\n", binary=binary, workdir=tmp_path,
                 timeout_s=0.5)
    assert not result.ok and "0.5" in result.status


def test_the_result_carries_the_log_and_the_command(tmp_path):
    binary = _recorder(tmp_path)
    launch = plan("input_file", deck_path="in.lammps", log="lmp.log",
                  gpus=None, uses_model=False)
    result = run(launch, "units metal\n", binary=binary, workdir=tmp_path)
    assert result.log == "lmp.log"
    assert str(binary) in result.command and "-in in.lammps" in result.command


def test_a_launcher_reaches_the_command_that_is_run(tmp_path):
    binary = _recorder(tmp_path)
    launch = plan("input_file", deck_path="in.lammps", screen=None,
                  gpus=None, uses_model=False)
    result = run(launch, "units metal\n", binary="/bin/sh",
                 launcher=[str(binary)], workdir=tmp_path)
    assert result.ok
    assert (tmp_path / "argv.txt").read_text().split()[0] == "/bin/sh"


def test_an_absolute_deck_path_is_used_as_given(tmp_path):
    where = tmp_path / "elsewhere"
    launch = plan("input_file", deck_path=where / "in.lammps", screen=None,
                  gpus=None, uses_model=False)
    run(launch, "units metal\n", binary=_recorder(tmp_path),
        workdir=tmp_path)
    assert (where / "in.lammps").is_file()


# ---------------------------------------------------------------------------
# The in-process route's call order
#
# A wrong order fails, so a successful real run cannot show that the order is
# right. This is the one place a stand-in is used, and it stands in for the
# bindings only.
# ---------------------------------------------------------------------------


class _FakeEngine:
    def __init__(self, calls, cmdargs):
        self.calls = calls
        self.cmdargs = cmdargs
        calls.append(("construct", tuple(cmdargs)))
        calls.append(("cwd", Path.cwd()))

    def commands_string(self, text):
        self.calls.append(("commands", text))

    def get_natoms(self):
        return 4560

    def close(self):
        self.calls.append(("close", None))


def _install_fake_bindings(monkeypatch, calls, engine_factory=None):
    import types

    factory = engine_factory or (
        lambda cmdargs: _FakeEngine(calls, cmdargs))
    module = types.ModuleType("lammps")
    module.lammps = lambda cmdargs: factory(cmdargs)
    mliap = types.ModuleType("lammps.mliap")
    mliap.activate_mliappy_kokkos = lambda lmp: calls.append(
        ("activate", None))
    mliap.load_unified_kokkos = lambda model: calls.append(("load", model))
    module.mliap = mliap
    monkeypatch.setitem(sys.modules, "lammps", module)
    monkeypatch.setitem(sys.modules, "lammps.mliap", mliap)
    monkeypatch.setattr(engine, "unavailable_reason",
                        lambda route, **kw: None)
    return calls


def test_the_model_is_in_the_interface_before_any_command_is_parsed(
        monkeypatch):
    """The whole point of the in-process route. The pair style is parsed
    while the deck runs, and by then the interface must already hold what the
    style says exists."""
    calls = _install_fake_bindings(monkeypatch, [])
    deck = f"pair_style mliap/kk unified {UNIFIED_TOKEN}\n"
    result = run(_model_launch(), deck,
                 potential=_potential(model=object()))
    assert [name for name, _ in calls] == [
        "construct", "cwd", "activate", "load", "commands", "close"]
    assert result.ok


def test_the_deck_reaches_the_engine_verbatim(monkeypatch):
    """Nothing here rewrites a value in a deck, including a seed."""
    calls = _install_fake_bindings(monkeypatch, [])
    deck = ("velocity all create 7000.0 498459 mom yes rot yes dist "
            f"gaussian\npair_style mliap/kk unified {UNIFIED_TOKEN}\n")
    run(_model_launch(), deck, potential=_potential(model=object()))
    assert [text for name, text in calls if name == "commands"] == [deck]


def test_the_planned_arguments_reach_the_engines_constructor(monkeypatch):
    calls = _install_fake_bindings(monkeypatch, [])
    launch = plan("in_process", log="run.log")
    run(launch, f"pair_style mliap/kk unified {UNIFIED_TOKEN}\n",
        potential=_potential(model=object()))
    assert calls[0] == ("construct", launch.args)


def test_the_atom_count_comes_back_from_the_run(monkeypatch):
    """A slab is built to a box and a density and REPORTS how many atoms that
    came to, where its sibling asks for a count and gets it."""
    _install_fake_bindings(monkeypatch, [])
    result = run(_model_launch(), f"pair_style mliap/kk unified "
                                  f"{UNIFIED_TOKEN}\n",
                 potential=_potential(model=object()))
    assert result.n_atoms == 4560


def test_a_run_with_no_model_activates_no_interface(monkeypatch):
    calls = _install_fake_bindings(monkeypatch, [])
    run(Launch(route="in_process", args=kokkos_args(), uses_model=False),
        "pair_style lj/cut 2.5\n")
    assert [name for name, _ in calls] == [
        "construct", "cwd", "commands", "close"]


def test_a_failure_inside_the_run_is_recorded_with_a_traceback(monkeypatch):
    calls = []

    class Exploding(_FakeEngine):
        def commands_string(self, text):
            raise RuntimeError("bad box")

    _install_fake_bindings(monkeypatch, calls,
                           lambda cmdargs: Exploding(calls, cmdargs))
    result = run(_model_launch(), f"pair_style mliap/kk unified "
                                  f"{UNIFIED_TOKEN}\n",
                 potential=_potential(model=object()))
    assert result.status == "error: bad box"
    assert "RuntimeError" in result.detail
    assert ("close", None) in calls


def test_a_loader_failure_is_not_reported_as_a_missing_install(monkeypatch):
    """Measured on this machine: with the session's inherited search path the
    bindings' shared library resolves a system library from a compiler
    toolchain and fails with an undefined symbol. Sending someone to
    reinstall what is already there is the wrong answer."""
    def explode(cmdargs):
        raise OSError("undefined symbol: __pgi_uacc_downloads")

    _install_fake_bindings(monkeypatch, [], explode)
    with pytest.raises(EngineError) as caught:
        run(_model_launch(), f"pair_style mliap/kk unified {UNIFIED_TOKEN}\n",
            potential=_potential(model=object()))
    message = str(caught.value)
    assert "loader order" in message
    assert "undefined symbol" in message


def test_a_potential_with_no_loaded_model_is_refused(monkeypatch):
    """The load is the expensive step and doing it twice can answer
    differently, so this module never does it."""
    _install_fake_bindings(monkeypatch, [])
    with pytest.raises(EngineError) as caught:
        run(_model_launch(), f"pair_style mliap/kk unified {UNIFIED_TOKEN}\n",
            potential=_potential(model=None))
    assert "Resolve it first" in str(caught.value)


def test_the_in_process_run_happens_in_the_working_directory(monkeypatch,
                                                            tmp_path):
    """The engine writes its own files beside itself -- a trajectory, a log,
    an end state -- and a farmed campaign gives every point its own place to
    put them."""
    calls = _install_fake_bindings(monkeypatch, [])
    run(_model_launch(), f"pair_style mliap/kk unified {UNIFIED_TOKEN}\n",
        potential=_potential(model=object()), workdir=tmp_path)
    where = dict(calls)["cwd"]
    assert where.resolve() == tmp_path.resolve()


def test_the_in_process_run_returns_to_where_it_started(monkeypatch,
                                                        tmp_path):
    _install_fake_bindings(monkeypatch, [])
    here = Path.cwd()
    run(_model_launch(), f"pair_style mliap/kk unified {UNIFIED_TOKEN}\n",
        potential=_potential(model=object()), workdir=tmp_path)
    assert Path.cwd() == here


def test_the_in_process_run_returns_even_when_it_fails(monkeypatch, tmp_path):
    def explode(cmdargs):
        raise RuntimeError("no device")

    _install_fake_bindings(monkeypatch, [], explode)
    here = Path.cwd()
    run(_model_launch(), f"pair_style mliap/kk unified {UNIFIED_TOKEN}\n",
        potential=_potential(model=object()), workdir=tmp_path)
    assert Path.cwd() == here


def test_the_in_process_route_refuses_when_the_bindings_are_absent(
        monkeypatch):
    monkeypatch.setattr(engine, "unavailable_reason",
                        lambda route, **kw: "no bindings here")
    with pytest.raises(EngineError) as caught:
        run(_model_launch(), f"pair_style mliap/kk unified {UNIFIED_TOKEN}\n",
            potential=_potential(model=object()))
    assert "no bindings here" in str(caught.value)


# ---------------------------------------------------------------------------
# Leaving by the back door
# ---------------------------------------------------------------------------


def test_leaving_flushes_what_was_written_and_reports_success():
    """The accelerator's teardown aborts on this build. The abort is
    cosmetic, but it replaces the exit status with a signal, and a launcher
    reading that status calls a finished run a crash. This exit does not
    flush, so the flush is here: the point of leaving early is to keep the
    output, not to lose it."""
    program = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        from aipf.md.engine import exit_cleanly
        sys.stdout.write("the run finished")
        exit_cleanly(0)
    """) % str(Path(engine.__file__).resolve().parents[3])
    done = subprocess.run([sys.executable, "-c", program],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0
    assert done.stdout == "the run finished"


def test_leaving_carries_a_status_of_its_own():
    program = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        from aipf.md.engine import exit_cleanly
        exit_cleanly(7)
    """) % str(Path(engine.__file__).resolve().parents[3])
    done = subprocess.run([sys.executable, "-c", program],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 7


def test_leaving_runs_nothing_that_was_registered_to_run_at_exit():
    """That is the point: the teardown that aborts is registered."""
    program = textwrap.dedent("""
        import atexit, sys
        sys.path.insert(0, %r)
        from aipf.md.engine import exit_cleanly
        atexit.register(lambda: sys.stdout.write("teardown"))
        sys.stdout.write("done")
        exit_cleanly(0)
    """) % str(Path(engine.__file__).resolve().parents[3])
    done = subprocess.run([sys.executable, "-c", program],
                          capture_output=True, text=True, timeout=120)
    assert done.stdout == "done"


# ---------------------------------------------------------------------------
# The real engine
# ---------------------------------------------------------------------------

_REAL_DRIVER = '''
import os, sys
from aipf.md import engine
from aipf.md.potentials import resolve

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
removed = engine.strip_environment()
assert removed and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ

route, out = sys.argv[1], sys.argv[2]
model, config, binary = sys.argv[3], sys.argv[4], sys.argv[5]
potential = resolve(model, engine="mace")

if route == "in_process":
    launch = engine.plan("in_process", log=os.path.join(out, "run.log"))
    pair = engine.pair_commands(potential, route="in_process")
else:
    launch = engine.plan("input_file", log="lmp.log", deck_path="in.lammps")
    pair = engine.pair_commands(potential, route="input_file")

deck = """
units           metal
boundary        p p p
atom_style      atomic
atom_modify     map yes
read_data       {config}
{pair}
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
timestep        0.0002
thermo          10
thermo_style    custom step time temp pe press vol
thermo_modify   flush yes
velocity        all create 7000.0 498459 mom yes rot yes dist gaussian
fix             prod all npt temp 7000.0 7000.0 0.01 iso 8000000 8000000 0.1
run             10
unfix           prod
""".format(config=config, pair=pair)

if route == "in_process":
    result = engine.run(launch, deck, potential=potential, workdir=out)
else:
    result = engine.run(launch, deck, potential=potential, binary=binary,
                        workdir=out, timeout_s=900)
print("STATUS", result.status)
print("ATOMS", result.n_atoms)
print("ROUTE", result.route)
sys.stdout.flush()
engine.exit_cleanly(0)
'''


def _drive_the_real_engine(route, out):
    done = subprocess.run(
        [str(_ENGINE_PYTHON), "-c", _REAL_DRIVER, route, str(out),
         str(_REAL_MODEL), str(_REAL_CONFIG), str(_ENGINE_BINARY)],
        capture_output=True, text=True, env=_engine_environment(),
        cwd=str(out), timeout=1800)
    return done


@pytest.mark.env
@_requires_engine_env
@_requires_real_model
def test_the_real_engine_runs_in_process(tmp_path):
    """The archived spelling, against the real engine, for real steps.

    Marked ``env``: it needs a device and the engine's own environment,
    neither of which the package's suite has. Measured on an A100: 512 atoms,
    about 26 s including the model load.
    """
    done = _drive_the_real_engine("in_process", tmp_path)
    assert "STATUS ok" in done.stdout, done.stdout + done.stderr
    assert "ATOMS 512" in done.stdout
    log = (tmp_path / "run.log").read_text()
    assert "pair_style      mliap/kk unified EXISTS" in log
    assert "Loop time" in log


@pytest.mark.env
@_requires_engine_env
@_requires_engine_binary
@_requires_real_model
def test_the_real_engine_runs_from_an_input_file(tmp_path):
    """The route the guide says cannot work, against the real engine.

    A whole tree's bulk and equation-of-state campaign ran this way.
    """
    done = _drive_the_real_engine("input_file", tmp_path)
    assert "STATUS ok" in done.stdout, done.stdout + done.stderr
    deck = (tmp_path / "in.lammps").read_text()
    assert f"unified {_REAL_MODEL} 0" in deck
    log = (tmp_path / "lmp.log").read_text()
    assert "Loop time" in log


@pytest.mark.env
@_requires_engine_binary
@_requires_real_model
def test_the_composite_spelling_is_a_parse_error_in_the_real_engine(
        tmp_path):
    """The spelling this project's own design notes carry. Measured: the
    argument after the path is a boolean flag, and the token is not one."""
    deck = tmp_path / "in.composite"
    deck.write_text(
        f"units metal\nboundary p p p\natom_style atomic\n"
        f"atom_modify map yes\nread_data {_REAL_CONFIG}\n"
        f"pair_style mliap/kk unified {_REAL_MODEL} {UNIFIED_TOKEN}\n"
        f"pair_coeff * * A B\nrun 0\n")
    subprocess.run(
        [str(_ENGINE_BINARY), "-screen", "none", "-log", str(tmp_path /
                                                             "log"),
         *kokkos_args(), "-in", str(deck)],
        capture_output=True, text=True, env=_engine_environment(),
        cwd=str(tmp_path), timeout=900)
    assert "ERROR" in (tmp_path / "log").read_text()
    assert UNIFIED_TOKEN in (tmp_path / "log").read_text()


@pytest.mark.env
@_requires_engine_binary
@_requires_real_model
@pytest.mark.parametrize("newton,neigh", [
    ("off", "half"), ("on", "full")])
def test_every_other_accelerator_corner_fails_in_the_real_engine(
        tmp_path, newton, neigh):
    """The constant is measured, not chosen."""
    deck = tmp_path / "in.corner"
    deck.write_text(
        f"units metal\nboundary p p p\natom_style atomic\n"
        f"atom_modify map yes\nread_data {_REAL_CONFIG}\n"
        f"pair_style mliap unified {_REAL_MODEL} 0\n"
        f"pair_coeff * * A B\nrun 0\n")
    log = tmp_path / f"log.{newton}.{neigh}"
    subprocess.run(
        [str(_ENGINE_BINARY), "-screen", "none", "-log", str(log),
         "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos",
         "newton", newton, "neigh", neigh, "-in", str(deck)],
        capture_output=True, text=True, env=_engine_environment(),
        cwd=str(tmp_path), timeout=900)
    assert "ERROR" in log.read_text()


@pytest.mark.env
@_requires_engine_env
def test_the_bindings_need_the_environments_own_libraries_first():
    """Loader order, not node type. With the session's inherited search path
    the shared library fails to load with an undefined symbol; with the
    environment's own directory first it loads."""
    program = ("import lammps; "
               "lammps.lammps(cmdargs=['-screen','none','-log','none'])"
               ".close(); print('OK')")
    inherited = dict(os.environ)
    inherited.pop("PYTHONPATH", None)
    first = subprocess.run([str(_ENGINE_PYTHON), "-c", program],
                           capture_output=True, text=True, env=inherited,
                           timeout=600)
    fixed = subprocess.run([str(_ENGINE_PYTHON), "-c", program],
                           capture_output=True, text=True,
                           env=_engine_environment(), timeout=600)
    assert "OK" in fixed.stdout, fixed.stderr
    if first.returncode != 0:
        assert "undefined symbol" in first.stderr or \
            "cannot open shared object" in first.stderr
