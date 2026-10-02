"""Resolving a potential, and refusing one that cannot run.

Two tiers. The default tier drives the module with FAKE loaded objects that
carry the same attributes a real converted model carries, so that every
refusal is reachable without paying the environment's import cost. The
``env`` tier at the bottom runs the same module against the real model files
on the scratch trees, which is what proves the fakes are shaped like the real
thing. Neither tier alone is worth much.
"""
import hashlib
import pathlib
import sys
import types

import pytest

from aipf.md.potentials import (
    E3NN_DISTRIBUTION,
    ENGINES,
    MLIAP_ATTRIBUTES,
    REQUIRED_E3NN,
    Potential,
    PotentialError,
    check_e3nn,
    describe,
    fallback_kernels,
    installed_version,
    normalise_engine,
    resolve,
    sha256_of,
)

# --------------------------------------------------------------------------
# Fakes shaped like a real converted model
# --------------------------------------------------------------------------


class _Leaf:
    """One dispatcher node: the method asked for, and the one built.

    The real library records the REQUESTED method on the dispatcher and
    stores the implementation it actually constructed beside it, so the two
    can disagree. That disagreement is the whole subject of this module, and
    it is why the fake has both attributes rather than one.
    """

    def __init__(self, requested, built_name):
        self.method = requested
        self.m = type(built_name, (), {})()


class _Tree:
    """An object with a module walk, the way a torch module has one."""

    def __init__(self, leaves):
        self._leaves = leaves

    def named_modules(self):
        yield "", self
        yield from self._leaves.items()


def _model(*, species=("Aa", "Bb"), dtype="float32", cutoff=1.5, leaves=None):
    """A loaded object carrying the attributes the run interface reads."""
    obj = types.SimpleNamespace(
        element_types=list(species),
        num_species=len(species),
        rcutfac=cutoff,
        ndescriptors=1,
        nparams=1,
        dtype=dtype,
        model=_Tree(leaves or {}),
    )
    return obj


def _loader_for(obj):
    """A loader that ignores the path and answers with one object."""
    def load(path):
        return obj
    return load


@pytest.fixture
def model_file(tmp_path):
    """A file that exists. Its contents are never parsed by the fake loader."""
    path = tmp_path / "potential.pt"
    path.write_bytes(b"not really a model, and nothing here parses it")
    return path


# --------------------------------------------------------------------------
# The engine name
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["mace", "mliap", "mace-mliap",
                                      "mace_mliap", "  mace  "])
def test_archived_engine_spellings_normalise(spelling):
    """The archive and the metadata schema spell the same engine differently.

    One archived manifest column says ``mace``; the metadata schema's own
    example says ``mace-mliap``. Both name the one engine this module drives.
    """
    assert normalise_engine(spelling) == "mace-mliap"


def test_an_engine_this_module_cannot_drive_is_refused_by_name():
    """A cross-check arm that ran on another engine must say so.

    One archived campaign carries an engine column with a second value, and
    four of its runs differ from their siblings in nothing but that. Dropping
    those silently would hand them this module's engine and record a
    provenance that is simply false, so the refusal names the engine it was
    given and says what is different about it.
    """
    with pytest.raises(PotentialError) as excinfo:
        normalise_engine("pet")
    message = str(excinfo.value)
    assert "pet" in message
    # Not merely "unsupported": the reason is what tells a reader whether to
    # convert the model, install something, or use another driver.
    assert "metatomic" in message


def test_an_unknown_engine_names_both_the_supported_and_the_refused():
    with pytest.raises(PotentialError) as excinfo:
        normalise_engine("nosuchengine")
    message = str(excinfo.value)
    assert "nosuchengine" in message
    assert "mace-mliap" in message
    assert "pet" in message


def test_no_engine_is_both_driven_and_refused():
    """The refusal is checked first, so an engine in both lists is refused
    today and accepted the moment anyone reorders the two checks. Asserting
    the lists are disjoint is what makes the ordering safe to change."""
    from aipf.md.potentials import _ENGINES_REFUSED

    assert ENGINES.isdisjoint(_ENGINES_REFUSED)


def test_an_engine_must_be_a_string():
    with pytest.raises(TypeError):
        normalise_engine(None)


def test_a_refusal_is_a_value_error():
    """The request vocabulary refuses with a plain ValueError. A launcher
    that wraps a whole campaign in one handler catches both or neither."""
    assert issubclass(PotentialError, ValueError)


# --------------------------------------------------------------------------
# The checksum
# --------------------------------------------------------------------------


def test_sha256_matches_the_bytes(tmp_path):
    path = tmp_path / "f.bin"
    payload = b"abcdefghijklmnopqrstuvwxyz"
    path.write_bytes(payload)
    assert sha256_of(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_reads_a_file_larger_than_one_chunk(tmp_path):
    """The real models measured on disk are 1 to 106 MB.

    A chunked reader that drops everything after the first chunk still
    produces a plausible-looking hex digest, so the size has to cross the
    chunk boundary for this to mean anything.
    """
    path = tmp_path / "big.bin"
    payload = bytes(range(256)) * 20_000          # about 5 MB
    path.write_bytes(payload)
    assert len(payload) > 1 << 20
    assert sha256_of(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_of_a_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        sha256_of(tmp_path / "absent.pt")


# --------------------------------------------------------------------------
# The live library version
# --------------------------------------------------------------------------


def test_installed_version_prefers_the_module_that_is_actually_imported(
        monkeypatch):
    """The brief's rule: verify the LIVE version, not the pin.

    An already-imported module IS the live version. A distribution lookup can
    disagree with it -- two installations on one path, or a user-site copy
    ahead of the environment's -- and the one that disagrees is the one that
    will be asked to unpickle the model.
    """
    fake = types.ModuleType("pretendlib")
    fake.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "pretendlib", fake)
    assert installed_version("pretendlib") == "9.9.9"


def test_installed_version_falls_back_to_distribution_metadata(monkeypatch):
    """A module that is imported but says nothing about its version.

    The imported module is replaced by one that carries no ``__version__``,
    which is the only way to reach the distribution branch for a name that is
    both importable and installed.
    """
    import importlib.metadata

    monkeypatch.setitem(sys.modules, "pytest", types.SimpleNamespace())
    assert installed_version("pytest") == importlib.metadata.version("pytest")


def test_a_version_that_is_not_a_name_is_not_believed(monkeypatch):
    """``__version__`` is conventional, not guaranteed, and a module that
    carries something else under that name has not answered the question."""
    fake = types.ModuleType("pretendlib_without_a_version")
    fake.__version__ = 7
    monkeypatch.setitem(sys.modules, "pretendlib_without_a_version", fake)
    assert installed_version("pretendlib_without_a_version") is None


def test_an_empty_version_has_not_answered_the_question(monkeypatch):
    """Same argument as a version that is not a name: the module was asked
    and said nothing, so the distribution is asked instead."""
    import importlib.metadata

    fake = types.ModuleType("pytest")
    fake.__version__ = ""
    monkeypatch.setitem(sys.modules, "pytest", fake)
    assert installed_version("pytest") == importlib.metadata.version("pytest")


def test_the_live_module_wins_when_the_two_sources_disagree(monkeypatch):
    """The case the docstring claims to handle, and the only one where the
    order of the two lookups is observable: both sources answer, and they
    answer differently. The imported copy is the one that reads the model."""
    import importlib.metadata

    installed = importlib.metadata.version("pytest")
    fake = types.ModuleType("pytest")
    fake.__version__ = "9.9.9"
    assert installed != "9.9.9"
    monkeypatch.setitem(sys.modules, "pytest", fake)
    assert installed_version("pytest") == "9.9.9"


def test_installed_version_is_none_when_nothing_is_installed():
    assert installed_version("no-such-distribution-anywhere") is None


def test_check_e3nn_is_quiet_when_the_version_matches(monkeypatch):
    monkeypatch.setattr("aipf.md.potentials.installed_version",
                        lambda name: "1.2.3")
    assert check_e3nn(required="1.2.3") is None


def test_check_e3nn_names_both_versions_when_they_differ(monkeypatch):
    """0.5.9 cannot unpickle the model, and fails with an error that names
    neither library nor version. The message has to supply both."""
    monkeypatch.setattr("aipf.md.potentials.installed_version",
                        lambda name: "0.5.9")
    problem = check_e3nn(required="0.5.1")
    assert problem is not None
    assert "0.5.9" in problem and "0.5.1" in problem


def test_check_e3nn_reports_an_absent_library_differently(monkeypatch):
    """Absent and wrong are different repairs: install versus downgrade."""
    monkeypatch.setattr("aipf.md.potentials.installed_version",
                        lambda name: None)
    problem = check_e3nn(required="0.5.1")
    assert problem is not None
    assert "not installed" in problem


def test_the_live_environment_carries_the_required_version():
    """Not the pin in the environment file -- the version that is installed.

    A pin is a request. This asserts the answer, in the interpreter the suite
    is running in, which is the one that would load a model.
    """
    live = installed_version(E3NN_DISTRIBUTION)
    if live is None:
        pytest.skip("e3nn is not installed in this interpreter")
    assert live == REQUIRED_E3NN


# --------------------------------------------------------------------------
# The kernel that got baked in
# --------------------------------------------------------------------------


def test_a_downgraded_kernel_is_reported_by_module_name():
    """Measured: converting with the accelerated library unreachable leaves
    the requested method recorded and the fallback implementation built.

    The dispatcher stores the method it was ASKED for before deciding what it
    can build, so the disagreement survives into the saved file and is
    readable afterwards. That is the only reason this is detectable at all.
    """
    tree = _Tree({"interactions.0.conv_tp.f":
                  _Leaf("uniform_1d", "SegmentedPolynomialNaive")})
    assert fallback_kernels(tree) == ("interactions.0.conv_tp.f",)


def test_a_kernel_that_asked_for_the_fallback_is_not_reported():
    """Most of the dispatcher nodes in a real model ask for the fallback.

    Measured on the real files: 12 of 16 in one converted model and 20 of 24
    in another. Reporting those would mean every correctly converted model
    looks broken, and a check that always fires is not a check.
    """
    tree = _Tree({"products.0.symmetric_contractions":
                  _Leaf("naive", "SegmentedPolynomialNaive")})
    assert fallback_kernels(tree) == ()


def test_only_the_disagreeing_nodes_are_named():
    tree = _Tree({
        "a": _Leaf("naive", "SegmentedPolynomialNaive"),
        "b": _Leaf("uniform_1d", "SegmentedPolynomialFromUniform1dJit"),
        "c": _Leaf("uniform_1d", "SegmentedPolynomialNaive"),
        "d": _Leaf("fused_tp", "SegmentedPolynomialNaive"),
    })
    assert fallback_kernels(tree) == ("c", "d")


def test_a_model_with_no_dispatcher_nodes_reports_none_found():
    """An empty tuple, not None: the walk happened and found nothing."""
    assert fallback_kernels(_Tree({})) == ()


def test_a_tree_that_cannot_be_walked_reports_that_it_was_not_checked():
    """None means "not checked", which is different from "nothing found".

    Recording an empty list for an object that could not be inspected would
    put "no problem" into the metadata of a run nobody looked at.
    """
    assert fallback_kernels(object()) is None


def test_a_node_missing_either_half_of_the_pair_is_skipped():
    """A torch module tree holds hundreds of nodes that are not dispatchers.

    Only a node that records BOTH a requested method and a built
    implementation can disagree with itself.

    The half with a method and no implementation is skipped twice over: the
    guard catches it, and so would the class-name test below it, because the
    name of the type of ``None`` does not end the way the fallback's does.
    The guard is kept for what it says, and a mutation sweep reports it as an
    equivalent mutant rather than an untested one.
    """
    half = types.SimpleNamespace(method="uniform_1d")
    other = types.SimpleNamespace(m=type("SegmentedPolynomialNaive", (), {})())
    assert fallback_kernels(_Tree({"a": half, "b": other})) == ()


def test_a_non_string_method_is_skipped():
    """``method`` is a common attribute name. A node that happens to carry a
    callable of that name is not a dispatcher and must not be read as one."""
    node = types.SimpleNamespace(method=lambda: None,
                                 m=type("SegmentedPolynomialNaive", (), {})())
    assert fallback_kernels(_Tree({"a": node})) == ()


# --------------------------------------------------------------------------
# What a usable model looks like
# --------------------------------------------------------------------------


def test_describe_reads_the_run_interface_attributes():
    obj = _model(species=("Aa", "Bb"), dtype="float64", cutoff=3.0)
    described = describe(obj)
    assert described["species"] == ("Aa", "Bb")
    assert described["dtype"] == "float64"
    assert described["cutoff"] == 3.0
    assert described["fallback_kernels"] == ()


def test_a_dtype_that_arrives_qualified_is_recorded_plainly():
    """Measured: the real models report their precision as the library's own
    qualified name. The record is read by people and by other tools, and the
    library's spelling of it is not part of the provenance."""
    obj = _model()
    obj.dtype = "torch.float64"
    assert describe(obj)["dtype"] == "float64"


def test_a_model_that_does_not_declare_a_dtype_records_nothing():
    """Null, not the string "None": the second is indistinguishable from a
    model that really did declare a precision called that."""
    obj = _model()
    del obj.dtype
    assert describe(obj)["dtype"] is None


def test_the_cutoff_is_recorded_as_a_number():
    """Whatever the model stores it as. A record is written as JSON, and a
    cutoff that arrives as a library object rather than a number fails there
    instead of here."""
    obj = _model()
    obj.rcutfac = 3
    cutoff = describe(obj)["cutoff"]
    assert cutoff == 3.0 and isinstance(cutoff, float)


def test_a_model_that_is_not_in_the_run_format_is_refused_by_its_type():
    """The raw trained model and the converted one are both loadable files.

    Only the converted one can be handed to the run interface. Handing it the
    raw one fails inside the engine, after the job has started, so the
    difference is refused here and the message names the type that was found.
    """
    class SomeTrainedModel:
        some_other_attribute = 1

    with pytest.raises(PotentialError) as excinfo:
        describe(SomeTrainedModel())
    message = str(excinfo.value)
    assert "SomeTrainedModel" in message
    # Written out, not read back from the module: a test that prints whatever
    # the constant currently says cannot notice the constant shrinking.
    assert ("['element_types', 'num_species', 'rcutfac', 'ndescriptors', "
            "'nparams']") in message
    assert list(MLIAP_ATTRIBUTES) == ["element_types", "num_species",
                                      "rcutfac", "ndescriptors", "nparams"]


def test_every_missing_interface_attribute_is_named_at_once():
    """One refusal per run, not one per attribute."""
    nearly = _model()
    del nearly.ndescriptors
    del nearly.nparams
    with pytest.raises(PotentialError) as excinfo:
        describe(nearly)
    message = str(excinfo.value)
    # The list of what is MISSING, distinct from the list of what is read.
    # Both appear, and with only two attributes gone they differ, which is
    # the only arrangement in which either can be checked for on its own.
    assert "missing ['ndescriptors', 'nparams']" in message
    assert ("reads ['element_types', 'num_species', 'rcutfac', "
            "'ndescriptors', 'nparams']") in message


@pytest.mark.parametrize("attribute", ["element_types", "num_species",
                                       "rcutfac", "ndescriptors", "nparams"])
def test_a_model_missing_any_one_interface_attribute_is_refused(attribute):
    """Refused, not crashed. Drop one name from the contract and the model
    sails past the check and fails later on an attribute error from inside
    whatever reads it -- which for the element list is the engine, mid-run."""
    obj = _model()
    delattr(obj, attribute)
    with pytest.raises(PotentialError):
        describe(obj)


def test_an_unordered_collection_is_not_an_element_list():
    """A set of the right names passes every other check here and has no
    order at all, and the order is the only thing this field is for."""
    obj = _model(species=("Aa", "Bb"))
    obj.element_types = {"Aa", "Bb"}
    with pytest.raises(PotentialError):
        describe(obj)


def test_an_empty_element_list_is_refused():
    """The element list sets the pair coefficient order. An empty one means
    the run would be given no order at all."""
    with pytest.raises(PotentialError):
        describe(_model(species=()))


def test_an_element_list_that_is_not_names_is_refused():
    with pytest.raises(PotentialError):
        describe(_model(species=(1, 2)))


def test_one_string_is_not_an_element_list():
    """A string IS a sequence of strings, character by character.

    A model whose element list came back as one name rather than a list of
    them would pass every shape test written the obvious way, and be read as
    having one element per letter.
    """
    obj = _model(species=("Aa", "Bb"))
    obj.element_types = "AaBb"      # one string, not a list of names
    obj.num_species = 4             # which would be read as four elements
    with pytest.raises(PotentialError):
        describe(obj)


def test_a_declared_count_that_disagrees_with_the_list_is_refused():
    """Both are read by the run interface, from two different places, and a
    disagreement is silent: the count sizes an array and the list names the
    coefficients."""
    obj = _model(species=("Aa", "Bb"))
    obj.num_species = 3
    with pytest.raises(PotentialError) as excinfo:
        describe(obj)
    assert "3" in str(excinfo.value)


# --------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------


def test_resolve_records_the_path_and_the_checksum(model_file):
    got = resolve(model_file, engine="mace", loader=_loader_for(_model()),
                  require_e3nn=None)
    assert got.path == model_file.resolve()
    # Hashed here rather than through the module, so that a module-wide
    # constant digest cannot satisfy both sides of the comparison.
    assert got.sha256 == hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert got.engine == "mace-mliap"
    assert got.species == ("Aa", "Bb")


def test_resolve_follows_a_symlink_to_the_bytes_it_hashed(tmp_path):
    """The data layout is built from symlinks that can be repointed later.

    A record naming the link rather than the target would carry a checksum of
    bytes the named path no longer has.
    """
    real = tmp_path / "real.pt"
    real.write_bytes(b"the actual model bytes")
    link = tmp_path / "link.pt"
    link.symlink_to(real)
    got = resolve(link, engine="mace", loader=_loader_for(_model()),
                  require_e3nn=None)
    assert got.path == real.resolve()
    assert got.sha256 == hashlib.sha256(real.read_bytes()).hexdigest()


def test_a_path_that_does_not_exist_is_refused_without_loading(tmp_path):
    """The refusal a campaign needs most, and the cheapest one to make."""
    calls = []

    def loader(path):
        calls.append(path)
        raise AssertionError("must not be reached")

    with pytest.raises(PotentialError) as excinfo:
        resolve(tmp_path / "absent.pt", engine="mace", loader=loader,
                require_e3nn=None)
    # Nothing there, which is a wrong path. Distinct from the message for
    # something that IS there and is not a file, which is a different repair.
    assert str(excinfo.value) == f"no potential at {tmp_path / 'absent.pt'}"
    assert calls == []


def test_a_directory_is_refused(tmp_path):
    """A path that exists and is not a file. The campaign pointed at the
    folder rather than the model in it, which is a different mistake from a
    path with nothing at it, so it gets a different sentence."""
    with pytest.raises(PotentialError) as excinfo:
        resolve(tmp_path, engine="mace", loader=_loader_for(_model()),
                require_e3nn=None)
    assert str(excinfo.value) == f"{tmp_path} is not a file"


def test_the_engine_is_refused_before_the_path_is_touched(tmp_path):
    """Naming an engine this module cannot drive is a mistake about the
    campaign, not about the file, so it does not depend on the file."""
    with pytest.raises(PotentialError) as excinfo:
        resolve(tmp_path / "absent.pt", engine="pet",
                loader=_loader_for(_model()), require_e3nn=None)
    assert "pet" in str(excinfo.value)


def test_a_wrong_library_version_refuses_before_the_load(model_file,
                                                         monkeypatch):
    """The load is the expensive step: 104 to 120 s of cold imports measured
    on this filesystem. The version check costs milliseconds and answers the
    same question, so it goes first."""
    monkeypatch.setattr("aipf.md.potentials.installed_version",
                        lambda name: "0.5.9")
    calls = []

    def loader(path):
        calls.append(path)
        return _model()

    with pytest.raises(PotentialError) as excinfo:
        resolve(model_file, engine="mace", loader=loader)
    assert "0.5.9" in str(excinfo.value)
    assert calls == []


def test_the_version_check_can_be_waived(model_file, monkeypatch):
    """A caller driving a model that does not go through that library at all
    says so explicitly, rather than the module guessing."""
    monkeypatch.setattr("aipf.md.potentials.installed_version",
                        lambda name: "0.5.9")
    got = resolve(model_file, engine="mace", loader=_loader_for(_model()),
                  require_e3nn=None)
    assert got.sha256


def test_a_load_failure_is_refused_with_the_original_cause(model_file):
    def loader(path):
        raise RuntimeError("Unknown type name 'something.NeighborListOptions'")

    with pytest.raises(PotentialError) as excinfo:
        resolve(model_file, engine="mace", loader=loader, require_e3nn=None)
    assert str(model_file.resolve()) in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_a_load_failure_mentions_the_library_version_that_was_checked(
        model_file, monkeypatch):
    """The measured failure of the wrong version is an unpacking error deep
    inside the pickle machinery, naming nothing useful. When a load fails,
    the version that was in force is the first thing a reader needs."""
    monkeypatch.setattr("aipf.md.potentials.installed_version",
                        lambda name: REQUIRED_E3NN)

    def loader(path):
        raise ValueError("too many values to unpack (expected 2)")

    with pytest.raises(PotentialError) as excinfo:
        resolve(model_file, engine="mace", loader=loader)
    assert REQUIRED_E3NN in str(excinfo.value)


def test_a_downgraded_model_warns_and_is_recorded(model_file):
    """It runs, and it is roughly three times slower. A refusal would block a
    usable model; silence would lose the one fact that explains the runtime."""
    obj = _model(leaves={"conv_tp.f":
                         _Leaf("uniform_1d", "SegmentedPolynomialNaive")})
    with pytest.warns(UserWarning, match="conv_tp.f"):
        got = resolve(model_file, engine="mace", loader=_loader_for(obj),
                      require_e3nn=None)
    assert got.fallback_kernels == ("conv_tp.f",)


def test_a_downgraded_model_can_be_refused_instead(model_file):
    obj = _model(leaves={"conv_tp.f":
                         _Leaf("uniform_1d", "SegmentedPolynomialNaive")})
    with pytest.raises(PotentialError) as excinfo:
        resolve(model_file, engine="mace", loader=_loader_for(obj),
                require_e3nn=None, refuse_fallback_kernels=True)
    assert "conv_tp.f" in str(excinfo.value)


def test_a_clean_model_does_not_warn(model_file, recwarn):
    obj = _model(leaves={"conv_tp.f":
                         _Leaf("uniform_1d",
                               "SegmentedPolynomialFromUniform1dJit")})
    got = resolve(model_file, engine="mace", loader=_loader_for(obj),
                  require_e3nn=None)
    assert got.fallback_kernels == ()
    assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []


def test_the_declared_element_order_is_checked_when_given(model_file):
    got = resolve(model_file, engine="mace",
                  loader=_loader_for(_model(species=("Aa", "Bb"))),
                  require_e3nn=None, species=("Aa", "Bb"))
    assert got.species == ("Aa", "Bb")


def test_an_empty_declared_order_is_a_refusal_not_a_skip(model_file):
    """A caller that declares no elements has declared a disagreement.

    Tested against None rather than for truth, the same rule the request
    vocabulary applies to a pressure of zero: an empty declaration is a
    statement, and treating it as "nothing was declared" turns a refusal into
    silence.
    """
    with pytest.raises(PotentialError):
        resolve(model_file, engine="mace",
                loader=_loader_for(_model(species=("Aa", "Bb"))),
                require_e3nn=None, species=())


def test_a_declared_order_may_be_a_list(model_file):
    """A caller's declaration usually arrives from JSON, where it is a list.
    Comparing the two as given would make every such declaration a mismatch."""
    got = resolve(model_file, engine="mace",
                  loader=_loader_for(_model(species=("Aa", "Bb"))),
                  require_e3nn=None, species=["Aa", "Bb"])
    assert got.species == ("Aa", "Bb")


def test_the_mismatch_names_both_orders(model_file):
    """Disjoint names on the two sides, so that dropping either side from
    the message is visible. With a permutation it is not: the same names
    appear whichever side is printed."""
    with pytest.raises(PotentialError) as excinfo:
        resolve(model_file, engine="mace",
                loader=_loader_for(_model(species=("Aa", "Bb"))),
                require_e3nn=None, species=("Cc", "Dd"))
    message = str(excinfo.value)
    assert "['Aa', 'Bb']" in message
    assert "['Cc', 'Dd']" in message


def test_the_element_order_matters_not_just_the_membership(model_file):
    """The pair coefficients are written in the model's order and the dump's
    numeric types are read in the caller's. The same two names in the other
    order is the same set and the wrong physics, with nothing raised."""
    with pytest.raises(PotentialError) as excinfo:
        resolve(model_file, engine="mace",
                loader=_loader_for(_model(species=("Aa", "Bb"))),
                require_e3nn=None, species=("Bb", "Aa"))
    message = str(excinfo.value)
    assert "Aa" in message and "Bb" in message


def test_the_model_object_is_carried_for_whoever_runs_it(model_file):
    """The engine needs the loaded object. Loading it twice would pay the
    import cost twice and could answer differently the second time."""
    obj = _model()
    got = resolve(model_file, engine="mace", loader=_loader_for(obj),
                  require_e3nn=None)
    assert got.model is obj


# --------------------------------------------------------------------------
# The metadata contract
# --------------------------------------------------------------------------


def test_meta_fields_carry_the_engine_and_the_potential(model_file):
    got = resolve(model_file, engine="mace", loader=_loader_for(_model()),
                  require_e3nn=None)
    fields = got.to_meta_fields()
    assert fields["engine"] == "mace-mliap"
    assert fields["potential"]["path"] == str(model_file.resolve())
    assert fields["potential"]["sha256"] == hashlib.sha256(
        model_file.read_bytes()).hexdigest()
    # Plain lists, not tuples: a record is compared against one read back
    # from JSON, where a tuple cannot survive.
    assert fields["potential"]["species"] == ["Aa", "Bb"]
    assert isinstance(fields["potential"]["species"], list)


def test_meta_fields_are_plain_data(model_file):
    """A record is written to JSON. A tuple or a Path in it is a later
    failure in whatever serialises the run."""
    import json

    got = resolve(model_file, engine="mace", loader=_loader_for(_model()),
                  require_e3nn=None)
    json.dumps(got.to_meta_fields())


def test_meta_fields_distinguish_no_problem_from_not_checked(model_file):
    """An empty list and a null are different answers and both are recorded.

    A run whose model could not be inspected must not be indistinguishable
    from one that was inspected and found clean.
    """
    clean = resolve(model_file, engine="mace", loader=_loader_for(_model()),
                    require_e3nn=None)
    assert clean.to_meta_fields()["potential"]["fallback_kernels"] == []

    opaque = _model()
    opaque.model = object()
    unchecked = resolve(model_file, engine="mace", loader=_loader_for(opaque),
                        require_e3nn=None)
    assert unchecked.to_meta_fields()["potential"]["fallback_kernels"] is None


def test_meta_fields_complete_a_record_that_validates():
    """The request knows nine metadata keys and this module knows two more.

    Composing them is the only way to see that the two halves fit, and the
    engine key in particular is one the request deliberately does not fill.
    """
    from aipf.data.meta import SCHEMA_VERSION, validate
    from aipf.md.request import StatePoint

    point = StatePoint(geometry="cube", ensemble="NPT", T=2000.0, x=0.5,
                       dt_ps=2e-4, equil_ps=1.5, prod_ps=30.0,
                       dump_every_ps=0.02, dump_from="prod", seed=1,
                       P=800.0, n_atoms=3456)
    potential = Potential(path=pathlib.Path("relative.pt"), sha256="0" * 64,
                          engine="mace-mliap", species=("Aa", "Bb"),
                          dtype="float32", cutoff=1.5, fallback_kernels=())
    record = {"schema_version": SCHEMA_VERSION, "system": "demo",
              "composition": {"species": ["Aa", "Bb"], "x": {"Bb": 0.5}},
              "box": {"L": [None, None, None], "varying": ["x", "y", "z"]},
              "status": "ok"}
    record.update(point.to_meta_fields())
    record.update(potential.to_meta_fields())
    assert validate(record) == []


def test_two_potentials_with_the_same_provenance_are_equal():
    """The loaded object is not provenance. Two resolutions of one file are
    the same record even though the objects are two different objects."""
    common = dict(path=pathlib.Path("a.pt"), sha256="1" * 64,
                  engine="mace-mliap", species=("Aa",), dtype="float32",
                  cutoff=1.5, fallback_kernels=())
    assert Potential(**common, model=object()) == Potential(**common,
                                                            model=object())


def test_a_potential_cannot_be_edited_after_it_is_resolved():
    potential = Potential(path=pathlib.Path("a.pt"), sha256="1" * 64,
                          engine="mace-mliap", species=("Aa",),
                          dtype="float32", cutoff=1.5, fallback_kernels=())
    with pytest.raises(Exception):
        potential.sha256 = "2" * 64


# --------------------------------------------------------------------------
# Against the real files
# --------------------------------------------------------------------------


def _scratch_models(pattern):
    """Real model files under the systems' raw roots, if they are declared and mounted.

    Found through the declared raw roots rather than written down here, so that this file holds
    no site path either.
    """
    from aipf.paths import MissingLocation
    from aipf.system import available, load

    found = []
    for name in available():
        try:
            root = load(name).paths.raw()
        except MissingLocation:
            continue
        if root.is_dir():
            found += sorted(root.glob(pattern))
    return found


@pytest.mark.env
def test_every_converted_model_on_disk_resolves():
    """Measured: eight converted files across two raw roots.

    This is the claim the fakes cannot make. If the run interface's attribute
    names ever change, every default-tier test here keeps passing and this
    one stops.
    """
    files = _scratch_models("models/*mliap*.pt") + \
        _scratch_models("builds/mliap/*.pt")
    if not files:
        pytest.skip("no converted models under any declared raw root (AIPF_RAW_<SYSTEM>, AIPF_RAW or [paths.raw])")
    for path in files:
        got = resolve(path, engine="mace")
        assert got.sha256 and len(got.sha256) == 64
        assert len(got.species) == got.model.num_species
        assert got.cutoff > 0
        assert got.dtype in ("float32", "float64"), (path, got.dtype)


@pytest.mark.env
def test_no_production_model_on_disk_carries_a_downgraded_kernel():
    """The check that would have caught a wasted campaign.

    Measured: the accelerated implementation is requested at four dispatcher
    nodes in each converted model on disk, and all four were built.
    """
    files = _scratch_models("builds/mliap/*.pt")
    if not files:
        pytest.skip("no converted models under any declared raw root (AIPF_RAW_<SYSTEM>, AIPF_RAW or [paths.raw])")
    for path in files:
        got = resolve(path, engine="mace")
        assert got.fallback_kernels == (), (path, got.fallback_kernels)


@pytest.mark.env
def test_an_unconverted_model_on_disk_is_refused():
    """The raw trained file loads perfectly well and cannot be run."""
    files = _scratch_models("models/*.model")
    files = [f for f in files if "mliap" not in f.name]
    if not files:
        pytest.skip("no raw models under any declared raw root (AIPF_RAW_<SYSTEM>, AIPF_RAW or [paths.raw])")
    for path in files:
        with pytest.raises(PotentialError):
            resolve(path, engine="mace")


@pytest.mark.env
def test_the_accelerated_kernel_is_actually_requested_on_disk():
    """Without this, a clean report above could mean nothing was ever asked.

    ``fallback_kernels`` reports only nodes that asked for something better
    than the reference implementation and did not get it. If no real model
    ever asked, an empty answer would be vacuously true forever and the check
    would be worthless. Measured: four nodes per converted model
    ask, in both converted models under the raw roots.
    """
    files = _scratch_models("builds/mliap/*cueq*.pt")
    if not files:
        pytest.skip("no accelerated models under any declared raw root (AIPF_RAW_<SYSTEM>, AIPF_RAW or [paths.raw])")
    asked = 0
    for path in files:
        got = resolve(path, engine="mace")
        for _, node in got.model.model.named_modules():
            method = getattr(node, "method", None)
            if isinstance(method, str) and method != "naive":
                asked += 1
    assert asked > 0


@pytest.mark.env
def test_every_saved_model_that_is_not_the_run_format_is_refused():
    """Measured: three such files under the raw roots, two ways.

    Two are saved for another engine and cannot be read by this interpreter
    at all: the load raises on an extension type name only that engine's
    runtime registers, and the error names neither the engine nor the fix.
    That is why an engine is refused BY NAME before any load is attempted.

    The third is a traced archive for the other pair style. It loads
    perfectly cleanly and carries none of the run interface, which is the
    more dangerous shape of the two: nothing about reading it suggests a
    problem. Both are refused, and the counts are asserted so that neither
    route can quietly disappear.
    """
    files = [p for p in _scratch_models("models/*.pt")
             if "mliap" not in p.name]
    if not files:
        pytest.skip("no such models under any declared raw root (AIPF_RAW_<SYSTEM>, AIPF_RAW or [paths.raw])")
    unreadable, readable_but_unusable = 0, 0
    for path in files:
        with pytest.raises(PotentialError):
            resolve(path, engine="mace")
        try:
            from aipf.md.potentials import _torch_load
            _torch_load(path)
        except Exception:
            unreadable += 1
        else:
            readable_but_unusable += 1
    assert unreadable > 0 and readable_but_unusable > 0, (
        unreadable, readable_but_unusable)
