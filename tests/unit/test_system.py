import dataclasses
import pathlib
import textwrap

import pytest

from aipf.paths import Paths
from aipf.system import (FUNCTIONAL_FORMS, AnchorRules, Functional,
                         Mobility, PreparationRules, System, load)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _demo() -> System:
    return System(
        name="demo",
        n_species=2,
        species=("A", "B"),
        masses={"A": 1.0, "B": 2.0},
        atom_types={"A": 1, "B": 2},
        table_keys={"rho": ("rho_A", "rho_B"), "x": "x_B", "x_channel": 1},
        paths=Paths(system="demo", raw_default="/absent/demo"),
        anchor_rules=AnchorRules(T_min_by_pressure={200: 6100, 800: 8973}),
        constants={"P_GPa": 800.0},
        defaults={"sigma": 2.0},
    )


def test_channel_of():
    s = _demo()
    assert s.channel_of("A") == 0
    assert s.channel_of("B") == 1


def test_species_count_must_match():
    with pytest.raises(ValueError, match="n_species"):
        System(name="bad", n_species=3, species=("A", "B"), masses={},
               atom_types={}, table_keys={},
               paths=Paths(system="bad"), anchor_rules=AnchorRules({}),
               constants={}, defaults={})


def test_t_min_lookup_and_miss():
    s = _demo()
    assert s.t_min(800) == 8973
    assert s.t_min(400) is None


def test_load_from_a_path(tmp_path):
    d = tmp_path / "demo"
    d.mkdir()
    (d / "system.py").write_text(textwrap.dedent("""
        from aipf.paths import Paths
        from aipf.system import AnchorRules, System

        SYSTEM = System(
            name="demo", n_species=1, species=("A",), masses={"A": 1.0},
            atom_types={"A": 1}, table_keys={}, paths=Paths(system="demo"),
            anchor_rules=AnchorRules({}), constants={}, defaults={},
        )
    """))
    s = load(str(d))
    assert s.name == "demo"
    assert s.n_species == 1


def test_load_by_name_uses_the_experiments_directory(tmp_path, monkeypatch):
    exp = tmp_path / "experiments" / "demo"
    exp.mkdir(parents=True)
    (exp / "system.py").write_text(textwrap.dedent("""
        from aipf.paths import Paths
        from aipf.system import AnchorRules, System

        SYSTEM = System(
            name="demo", n_species=1, species=("A",), masses={"A": 1.0},
            atom_types={"A": 1}, table_keys={}, paths=Paths(system="demo"),
            anchor_rules=AnchorRules({}), constants={}, defaults={},
        )
    """))
    monkeypatch.setenv("AIPF_EXPERIMENTS", str(tmp_path / "experiments"))
    assert load("demo").name == "demo"


def test_load_unknown_name_lists_what_is_available(tmp_path, monkeypatch):
    (tmp_path / "experiments" / "demo").mkdir(parents=True)
    (tmp_path / "experiments" / "demo" / "system.py").write_text("SYSTEM = None")
    monkeypatch.setenv("AIPF_EXPERIMENTS", str(tmp_path / "experiments"))
    with pytest.raises(FileNotFoundError, match="demo"):
        load("missing")


# --------------------------------------------------------------------------
# What else construction has to catch.
#
# A system file writes these by hand, from values read out of its training
# code, and the index, the pipeline and training index into ``species`` with
# things declared beside it. Each test below names the mutation it catches.
# --------------------------------------------------------------------------
def _system(**overrides) -> System:
    """A minimal valid system, with one field replaced."""
    kwargs = dict(
        name="demo", n_species=2, species=("A", "B"),
        masses={"A": 1.0, "B": 2.0}, atom_types={"A": 1, "B": 2},
        table_keys={}, paths=Paths(system="demo"),
        anchor_rules=AnchorRules({}), constants={}, defaults={},
    )
    kwargs.update(overrides)
    return System(**kwargs)


def _write_system(folder, name="demo", extra=""):
    """An experiment folder that defines one loadable SYSTEM."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "system.py").write_text(textwrap.dedent(f"""
        from aipf.paths import Paths
        from aipf.system import AnchorRules, System

        SYSTEM = System(
            name={name!r}, n_species=1, species=("A",), masses={{"A": 1.0}},
            atom_types={{"A": 1}}, table_keys={{}}, paths=Paths(system={name!r}),
            anchor_rules=AnchorRules({{}}), constants={{{extra}}}, defaults={{}},
        )
    """))
    return folder


def test_a_repeated_species_is_rejected():
    """Mutation: dropping the uniqueness check.

    ``channel_of`` answers with the first index, so the second copy's channel
    becomes unreachable and every row of it is silently labelled with the
    first. The count check above passes such a system happily.
    """
    with pytest.raises(ValueError, match="repeats"):
        _system(species=("A", "A"), masses={"A": 1.0}, atom_types={"A": 1})


def test_a_system_with_no_channels_is_rejected():
    """Mutation: dropping the ``n_species < 1`` check.

    ``n_species=0`` with ``species=()`` satisfies the count check, and every
    field tensor built from it then has a zero-length channel axis: a model
    that trains on nothing, with no error anywhere.
    """
    with pytest.raises(ValueError, match="n_species"):
        _system(n_species=0, species=(), masses={}, atom_types={})


def test_masses_must_cover_exactly_the_species():
    """Mutation: dropping the ``masses`` check. A typo in a hand-copied key
    becomes a KeyError wherever the mass is looked up, far from the file that
    has to be corrected."""
    with pytest.raises(ValueError, match="masses"):
        _system(masses={"A": 1.0})                      # channel B has none
    with pytest.raises(ValueError, match="masses"):
        _system(masses={"A": 1.0, "B": 2.0, "C": 3.0})  # C is not a channel


def test_atom_types_must_cover_every_channel():
    """Mutation: dropping the ``atom_types`` check. This mapping decodes the
    numeric type column of a trajectory dump, so a channel with no type stops
    the parse of every frame."""
    with pytest.raises(ValueError, match="atom_types"):
        _system(atom_types={"A": 1})


def test_atom_types_may_carry_a_type_that_is_no_channel():
    """Mutation: checking ``atom_types`` for equality, as ``masses`` is checked.

    A dump type that carries no density channel of its own is legitimate, a
    dilute tracer or a wall, and the next planned system has one. Equality here
    would make that experiment undeclarable, and a test is harder to relax than
    a line of code, so the permission is pinned rather than left to be
    rediscovered.
    """
    s = _system(atom_types={"A": 1, "B": 2, "Tracer": 3})
    assert s.atom_types["Tracer"] == 3


def test_the_message_names_both_what_is_missing_and_what_is_unknown():
    """Mutation: truncating the message after ``species=...``.

    This test previously asserted ``"'B'" in message and "'Bb'" in message``,
    and the head of the message, "masses={...} does not match species=('A',
    'B')", already contains both, so a message cut off at that point left the
    suite green and the clause this test is named after was pinned by nothing.
    A review found it by doing exactly that. It now asserts the words.

    ``masses`` rather than ``atom_types`` because only the exact check has an
    "unknown" half to report.
    """
    with pytest.raises(ValueError) as excinfo:
        _system(masses={"A": 1.0, "Bb": 2.0})
    message = str(excinfo.value)
    assert "missing ['B']" in message, message
    assert "unknown ['Bb']" in message, message


def test_a_mapping_that_is_None_is_not_an_undeclared_mapping():
    """Mutation: dropping the ``None`` guard.

    The exemption for an undeclared mapping is written ``if not mapping``, and
    ``None`` satisfies it, so a field left as None sails through construction
    and is then indexed as a dictionary somewhere downstream. The error has to
    name the field, since both look identical from the traceback.
    """
    with pytest.raises(TypeError, match="masses"):
        _system(masses=None)
    with pytest.raises(TypeError, match="atom_types"):
        _system(atom_types=None)


def test_mappings_keyed_by_species_may_be_left_undeclared():
    """Guards against over-validation, not under-validation.

    The brief's own failing-construction test passes ``masses={}`` and
    ``atom_types={}``, and the data index's fixture leaves ``table_keys`` empty. Empty
    means "not declared" and must stay constructible, or a correct check turns
    into a broken interface.
    """
    s = _system(masses={}, atom_types={}, table_keys={})
    assert s.masses == {} and s.atom_types == {}


def test_rho_column_names_are_one_per_channel():
    """Mutation: dropping the ``table_keys['rho']`` length check. These column
    names are read in channel order, so a short list quietly maps the wrong
    column onto the last channel."""
    with pytest.raises(ValueError, match="rho"):
        _system(table_keys={"rho": ("rho_A",)})


def test_x_channel_must_index_into_the_species():
    """Mutation: dropping the ``x_channel`` range check.

    The data index writes ``system.species[system.table_keys.get("x_channel", 0)]``
    into the metadata of every state point. Out of range raises an IndexError
    halfway through building the index, and a negative index does not raise at
    all: it labels the composition with the last channel instead.
    """
    with pytest.raises(ValueError, match="x_channel"):
        _system(table_keys={"x_channel": 2})
    with pytest.raises(ValueError, match="x_channel"):
        _system(table_keys={"x_channel": -1})


def test_x_channel_must_not_be_a_species_name():
    """Mutation: checking the range but not the type.

    A species name where an index belongs is the natural typo, and the range
    check alone answers it with a TypeError from the comparison rather than
    with a message that says which key is wrong.
    """
    with pytest.raises(ValueError, match="x_channel"):
        _system(table_keys={"x_channel": "B"})


def test_x_channel_must_not_be_a_bool():
    """Mutation: ``isinstance(channel, int)`` on its own.

    ``isinstance(True, int)`` is True and ``species[True]`` is ``species[1]``,
    so a bool passes both the type check and the range check and then silently
    selects the second channel. Split from the test above because a single test
    stops at its first ``raises`` block and would never reach this case.
    """
    with pytest.raises(ValueError, match="x_channel"):
        _system(table_keys={"x_channel": True})


def test_the_composition_column_must_be_a_column_name():
    """Mutation: dropping the ``table_keys['x']`` check.

    The data index reads the composition with ``src.get(table_keys["x"])``. A value
    that is not a column name does not raise there, it simply never matches, and
    the composition is then recovered from the run tag or recorded as unknown.
    ``x_channel`` had two checks and this one had none.
    """
    with pytest.raises(ValueError, match=r"table_keys\['x'\]"):
        _system(table_keys={"x": 1})


def test_a_non_numeric_cut_table_is_rejected_at_construction():
    """Mutation: dropping the ``T_min_by_pressure`` check.

    Unchecked, the first ``t_min`` lookup raises "could not convert string to
    float" from inside the loop, naming neither the system nor the field. A system
    file writes real cut tables into this, so it is caught where it is written.
    """
    with pytest.raises(ValueError, match="T_min_by_pressure"):
        _system(anchor_rules=AnchorRules({"low": 6100}))


def test_the_cut_table_is_checked_on_both_sides_of_each_entry():
    """Mutation: validating the keys and not the values.

    A temperature is as arithmetic as a pressure, and a bad value survives the
    lookup to fail in whatever does the comparing.
    """
    with pytest.raises(ValueError, match="T_min_by_pressure"):
        _system(anchor_rules=AnchorRules({200: "cold"}))


def test_a_cut_table_spelled_as_strings_is_still_accepted():
    """The other half: JSON gives back string keys, and those are numeric."""
    s = _system(anchor_rules=AnchorRules({"200": "6100"}))
    assert s.t_min(200) == 6100


def test_a_valid_x_channel_is_accepted():
    """The other half of the two tests above: the check must pass what a system
    file actually writes, including channel 0, which is falsy."""
    assert _system(table_keys={"x_channel": 0}).table_keys["x_channel"] == 0
    assert _system(table_keys={"x_channel": 1}).table_keys["x_channel"] == 1


def test_channel_of_an_unknown_species_names_the_alternatives():
    """Mutation: leaving ``tuple.index`` to report it.

    Its message is "tuple.index(x): x not in tuple", which names neither the
    system nor the species it does have.
    """
    with pytest.raises(ValueError) as excinfo:
        _demo().channel_of("C")
    message = str(excinfo.value)
    assert "demo" in message and "A" in message and "B" in message


def test_t_min_matches_a_cut_table_whose_keys_are_not_numbers():
    """Mutation: replacing the loop with ``T_min_by_pressure.get(pressure)``.

    This test was first written to assert that a float query matches a
    whole-number key, and the mutation run showed it could not fail: Python
    hashes 800.0 and 800 alike, so a plain lookup answers that correctly
    already. What the loop is actually for is a table whose keys are spelled
    some other way, and every cut table that has been through JSON comes back
    with string keys, because JSON object keys are always strings.
    """
    s = _system(anchor_rules=AnchorRules({"800": 8973}))
    assert s.t_min(800) == 8973
    assert s.t_min(800.0) == 8973
    assert s.t_min(400) is None


def test_t_min_answers_with_a_number_not_the_stored_object():
    """Mutation: returning ``value`` unconverted. The tables are consumed
    arithmetically, and one of the three sources writes its cuts as strings."""
    s = _system(anchor_rules=AnchorRules({800: 8973}))
    assert isinstance(s.t_min(800), float)


def test_system_is_frozen():
    """Mutation: dropping ``frozen=True``.

    Tasks 9 to 11 hold this object and read published defaults off it. Nothing
    downstream may rewrite the settings that define a reproduction.
    """
    import dataclasses
    s = _demo()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.n_species = 5             # type: ignore[misc]


def test_anchor_rules_is_frozen():
    """Mutation: dropping ``frozen=True`` from AnchorRules.

    ``System`` is frozen and its own test says why. The rules object it holds
    was left unpinned, and it is the one that carries the published cut table,
    so a caller could rewrite the anchor set of a run in flight.
    """
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        AnchorRules({}).T_min_by_pressure = {0: 1}   # type: ignore[misc]


def test_the_first_positional_field_of_anchor_rules_is_the_cut_table():
    """Mutation: swapping the two fields.

    ``AnchorRules({})`` is written positionally in this file's own fixtures.
    No experiment file does -- all three pass ``T_min_by_pressure=`` by
    keyword. (An earlier version of this docstring claimed two of the three
    were positional. They are not, and the claim was reviewed and kept once
    before being caught.)

    The argument does not rest on that anyway, and deleting ``qc_gate``
    STRENGTHENED it. Before, a reorder would have assigned a dict to a bool
    and something downstream would have complained. Now BOTH remaining fields
    are cut tables of the same type, so a reordering raises nothing: it would
    file the anchor cut as the kernel-shape floor and leave the anchor cut
    empty, silently.
    """
    rules = AnchorRules({200: 6100})
    assert rules.T_min_by_pressure == {200: 6100}
    assert rules.shape_T_min_by_pressure == {}


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def test_a_bare_name_is_not_shadowed_by_a_directory_in_the_working_directory(
        tmp_path, monkeypatch):
    """Mutation: deciding the path branch with ``Path(name_or_path).is_dir()``.

    A short name is then answered by whatever directory of that name happens to
    sit in the directory the job started in, silently in preference to the
    system that was asked for.
    """
    _write_system(tmp_path / "demo", extra="'which': 'cwd'")
    _write_system(tmp_path / "experiments" / "demo", extra="'which': 'declared'")
    monkeypatch.setenv("AIPF_EXPERIMENTS", str(tmp_path / "experiments"))
    monkeypatch.chdir(tmp_path)
    assert load("demo").constants["which"] == "declared"


def test_a_relative_path_with_a_separator_is_still_a_path(tmp_path, monkeypatch):
    """The spec spells the explicit form ``./experiments/<name>``, so the path
    branch must not be reachable only by absolute paths."""
    _write_system(tmp_path / "experiments" / "demo")
    monkeypatch.delenv("AIPF_EXPERIMENTS", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load("./experiments/demo").name == "demo"


def test_experiments_root_falls_back_to_the_repository_not_the_cwd(
        tmp_path, monkeypatch):
    """Mutation: ``Path.cwd() / "experiments"``.

    That shortcut has already been removed twice from this package, from
    ``aipf.guard`` and from ``aipf.paths``. Here it would make ``load("demo")``
    answer differently depending on where a batch job started. Running from
    ``tmp_path`` is what distinguishes the two.
    """
    from aipf.system import experiments_root
    monkeypatch.delenv("AIPF_EXPERIMENTS", raising=False)
    monkeypatch.chdir(tmp_path)
    assert experiments_root() == REPO / "experiments"


def test_experiments_root_raises_rather_than_guessing_outside_a_checkout(
        monkeypatch):
    """Mutation: falling back to the working directory when there is no
    repository. The error has to name the way out, which is the variable."""
    import aipf.paths as paths_module
    from aipf.system import experiments_root
    monkeypatch.delenv("AIPF_EXPERIMENTS", raising=False)
    monkeypatch.setattr(paths_module, "_REPO_MARKER", "no-such-marker-file.toml")
    with pytest.raises(RuntimeError, match="AIPF_EXPERIMENTS"):
        experiments_root()


def test_an_empty_experiments_variable_counts_as_unset(tmp_path, monkeypatch):
    """``VAR=`` is how a variable reads when a job script never filled it.

    Honouring it resolves to ``Path("")``, the working directory, and every
    lookup then reports that no system exists rather than failing.
    """
    from aipf.system import experiments_root
    monkeypatch.setenv("AIPF_EXPERIMENTS", "")
    monkeypatch.chdir(tmp_path)
    assert experiments_root() == REPO / "experiments"


def test_available_lists_only_folders_that_define_a_system(tmp_path, monkeypatch):
    """Mutation: listing every entry of the experiments directory.

    The brief's own test has exactly one folder and it is a valid one, so a
    listing that ignored the file would pass it while advertising a stray
    directory, a README or a __pycache__ as a system.
    """
    from aipf.system import available
    _write_system(tmp_path / "experiments" / "demo")
    (tmp_path / "experiments" / "notasystem").mkdir()
    (tmp_path / "experiments" / "README.md").write_text("prose")
    monkeypatch.setenv("AIPF_EXPERIMENTS", str(tmp_path / "experiments"))
    assert available() == ["demo"]


def test_available_answers_in_sorted_order(tmp_path, monkeypatch):
    """Mutation: returning the directory listing unsorted.

    ``available`` promises sorted and feeds the list a reader sees when a system
    is not found, so the order is part of the interface. The test does not lean
    on the filesystem to prove it. A directory listing comes back in whatever
    order the filesystem chose, which on a common file system is a hash order that is
    almost never alphabetical and is guaranteed to be nothing in particular, so
    the root is wrapped to hand its entries back in reverse. The mutation then
    fails here and on any other filesystem too.
    """
    import aipf.system as system_module
    root = tmp_path / "experiments"
    for name in ("delta", "alpha", "charlie", "bravo"):
        _write_system(root / name, name=name)

    class _ReversedRoot:
        """The experiments directory, listing itself in the worst order."""

        def is_dir(self):
            return True

        def iterdir(self):
            return reversed(sorted(root.iterdir()))

    monkeypatch.setattr(system_module, "experiments_root",
                        lambda: _ReversedRoot())
    assert system_module.available() == ["alpha", "bravo", "charlie", "delta"]


def test_the_failure_names_a_route_out_when_there_is_nothing_to_list(
        tmp_path, monkeypatch):
    """Mutation: reporting a bare "none".

    With no experiments directory at all, the list is empty and the reader
    learns nothing. The message has to say how to point the package at one.
    """
    monkeypatch.setenv("AIPF_EXPERIMENTS", str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="AIPF_EXPERIMENTS"):
        load("demo")


def test_an_unresolvable_experiments_root_does_not_mask_a_bad_path(
        tmp_path, monkeypatch):
    """Mutation: building the "available" list without guarding it.

    Loading by an explicit path must fail with FileNotFoundError about that
    path, not with the RuntimeError raised while assembling the advice.
    """
    import aipf.paths as paths_module
    monkeypatch.delenv("AIPF_EXPERIMENTS", raising=False)
    monkeypatch.setattr(paths_module, "_REPO_MARKER", "no-such-marker-file.toml")
    with pytest.raises(FileNotFoundError):
        load(str(tmp_path / "no" / "such" / "folder"))


def test_two_systems_loaded_in_turn_stay_distinct(tmp_path):
    """Mutation: caching or reusing one module object across loads.

    Both files are named ``system.py``; only the folder tells them apart.
    """
    a = _write_system(tmp_path / "one", name="one")
    b = _write_system(tmp_path / "two", name="two")
    assert load(str(a)).name == "one"
    assert load(str(b)).name == "two"
    assert load(str(a)).name == "one"


def test_load_reports_a_file_that_defines_no_system(tmp_path, monkeypatch):
    """The brief's own fixture writes ``SYSTEM = None``, and nothing loads it.
    Mutation: returning the missing attribute instead of raising."""
    d = tmp_path / "demo"
    d.mkdir()
    (d / "system.py").write_text("SYSTEM = None\n")
    with pytest.raises(AttributeError, match="SYSTEM"):
        load(str(d))


def test_load_rejects_a_system_that_is_not_a_System(tmp_path):
    """Mutation: returning whatever the module called SYSTEM.

    A dictionary or a config object would be returned happily and fail later,
    on the first attribute access, with no mention of the file at fault.
    """
    d = tmp_path / "demo"
    d.mkdir()
    (d / "system.py").write_text("SYSTEM = {'name': 'demo'}\n")
    with pytest.raises(TypeError, match="not a System"):
        load(str(d))


# ---------------------------------------------------------------------------
# PreparationRules: the hot temperature a run starts from
#
# A molecular-dynamics run rarely starts at its target temperature -- it is
# melted or mixed hot first. That hot temperature is the one that melts THIS
# material, so it is declared per system. Measured in the archive at three
# sites, and unrepresentable before this field existed (found while checking
# the state-point vocabulary against 1487 archived manifest rows).
# ---------------------------------------------------------------------------

def test_preparation_defaults_to_declaring_nothing():
    """Core ships no melting temperature, so the default must be empty --
    not a number that happens to suit one system."""
    rules = PreparationRules()
    assert rules.melt_T is None
    assert rules.ramp_offset is None


def test_a_system_that_declares_no_preparation_still_builds():
    """`preparation` is additive: a system with no melt step is not obliged
    to say so in a field it does not use."""
    assert _demo().preparation == PreparationRules()


@pytest.mark.parametrize("field_name,value", [("melt_T", 2600.0),
                                              ("ramp_offset", 2000.0)])
def test_preparation_is_frozen_like_every_other_declaration(field_name, value):
    rules = PreparationRules(**{field_name: value})
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(rules, field_name, 0.0)


def test_an_absolute_melt_and_an_offset_are_different_declarations():
    """They are alternatives, not a pair. A fixed hot temperature does not
    depend on the target; an offset does. Conflating them would silently
    turn one system's preparation into another's."""
    absolute = PreparationRules(melt_T=2600.0)
    offset = PreparationRules(ramp_offset=2000.0)
    assert absolute != offset
    assert absolute.ramp_offset is None
    assert offset.melt_T is None


def test_declaring_both_preparation_forms_is_refused_at_construction():
    """Documented alternatives are not enforced alternatives.

    The first version of this class only said the two forms were
    alternatives. A system was landed the same day declaring both, and
    nothing caught it until a deck-filling consumer refused to choose --
    a failure after the queue wait instead of at import. The contract is
    enforced here now, where it is cheap.
    """
    with pytest.raises(ValueError, match="alternatives"):
        PreparationRules(melt_T=2600.0, ramp_offset=2000.0)


@pytest.mark.parametrize("kwargs", [
    {}, {"melt_T": 2600.0}, {"ramp_offset": 2000.0},
])
def test_every_admissible_preparation_still_builds(kwargs):
    """The refusal must bite only on the conflict, not on the three
    legitimate declarations."""
    assert PreparationRules(**kwargs) is not None


def test_no_shipped_experiment_declares_both_forms():
    """The defect this refusal exists for was shipped in an experiment
    file, not in a test fixture, so it is the experiments that are
    checked. A system whose preparation cannot be resolved fills no deck
    at all, which is discovered late and far from its cause.
    """
    import importlib.util
    root = pathlib.Path(__file__).resolve().parents[2] / "experiments"
    seen = 0
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        path = folder / "system.py"
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            f"_exp_{folder.name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)          # raises if both declared
        prep = module.SYSTEM.preparation
        assert not (prep.melt_T is not None and prep.ramp_offset is not None)
        seen += 1
    assert seen >= 3, f"scanned only {seen} experiments, expected every one"


# --------------------------------------------------------------------------
# The model a system trains, declared on the system (spec 3.1)
# --------------------------------------------------------------------------


def test_functional_forms_are_the_four_physics_names_and_no_integers():
    assert FUNCTIONAL_FORMS == ("landau", "square_gradient",
                                "nonlocal_kernel", "neural_operator")
    for f in FUNCTIONAL_FORMS:
        assert not any(ch.isdigit() for ch in f), f


def test_functional_refuses_an_unknown_form():
    with pytest.raises(ValueError, match="landau.*neural_operator"):
        Functional(form="nonlocal", local="mlp", kernel=None, kwargs={})


def test_functional_requires_a_kernel_only_for_nonlocal_kernel():
    Functional(form="square_gradient", local="mlp", kernel=None, kwargs={})
    with pytest.raises(ValueError, match="kernel"):
        Functional(form="nonlocal_kernel", local="mlp", kernel=None, kwargs={})
    with pytest.raises(ValueError, match="kernel"):
        Functional(form="landau", local="basis", kernel="radial_mlp", kwargs={})


def test_functional_and_mobility_are_frozen():
    f = Functional(form="landau", local="basis", kernel=None, kwargs={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.form = "square_gradient"   # type: ignore[misc]
    m = Mobility(form="constant", T_form="none", kwargs={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.form = "gamma"             # type: ignore[misc]


# ---------------------------------------------------------------------------
# Variants: other declared models of one system
# ---------------------------------------------------------------------------

def _variant_parts():
    from aipf.system import Checkpoint, Functional, Mobility, Variant
    f = Functional(form="square_gradient", local="landau", kernel=None, kwargs={"a0": 1.0})
    m = Mobility(form="lattice_scalar", T_form="none", kwargs={})
    ck = Checkpoint(path="runs/v/final.ckpt", md5="0" * 32)
    return Variant, f, m, ck


def test_a_variant_replaces_the_model_and_the_named_defaults_only():
    import dataclasses
    from aipf.system import load
    Variant, f, m, ck = _variant_parts()
    base = load("lj")
    system = dataclasses.replace(base, variants={"v": Variant(
        functional=f, mobility=m, checkpoint=ck, defaults={"rung": 2})})
    derived = system.variant("v")
    assert derived.functional is f and derived.mobility is m and derived.checkpoint is ck
    assert derived.defaults["rung"] == 2 and derived.defaults["sigma"] == base.defaults["sigma"]
    assert derived.constants == base.constants and derived.paths is base.paths
    assert derived.variants == {} and derived.name == base.name
    assert derived.variant_name == "v" and base.variant_name is None
    assert system.defaults["rung"] == base.defaults["rung"]


def test_an_undeclared_variant_is_refused_naming_the_declared_ones():
    from aipf.system import load
    with pytest.raises(KeyError, match="fh"):
        load("lj").variant("nope")


def test_a_variant_is_type_checked():
    Variant, f, m, ck = _variant_parts()
    with pytest.raises(TypeError, match="functional"):
        Variant(functional=m, mobility=m, checkpoint=ck, defaults={})
    with pytest.raises(TypeError, match="checkpoint"):
        Variant(functional=f, mobility=m, checkpoint="x", defaults={})
    import dataclasses
    from aipf.system import load
    with pytest.raises(TypeError, match="Variant"):
        dataclasses.replace(load("lj"), variants={"v": f})


def test_the_production_declaration_is_what_load_returns():
    from aipf.system import load
    lj = load("lj")
    assert lj.functional.form == "nonlocal_kernel" and set(lj.variants) == {"fh", "landau"}
    assert load("hhe").variants == {} and load("feb").variants == {}
