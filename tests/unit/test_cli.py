"""The three production verbs on the command line, and what they refuse.

``train``, ``diagnose`` and ``modes`` each drive something that costs hours
of node time, so the property under test here is not that they run -- the
three functions have their own suites for that -- but that NOTHING they
need is defaulted. Every production value is an ``argparse`` flag with
``required=True``, the ``--help`` text names each one, and omitting one is
a usage error rather than a run with a value nobody chose.

Nothing in this file starts a real run. The paths exercised are ``--help``,
the refusals, and three in-process dispatch checks whose callees are
replaced, so the suite stays seconds rather than hours.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types

import pytest


def _aipf(*argv: str) -> subprocess.CompletedProcess:
    """The console entry point, in a child process, as an operator runs it."""
    return subprocess.run([sys.executable, "-m", "aipf.cli.main", *argv],
                          capture_output=True, text=True)


#: Every flag each verb refuses to run without, by verb. This table is the
#: requirement: a flag that leaves it has acquired a default somewhere.
REQUIRED = {
    "train": ("--system", "--run", "--seed", "--steps", "--epochs",
              "--source", "--resume-optimizer", "--anchors"),
    "diagnose": ("--system", "--ckpt", "--stage"),
    "modes": ("--system", "--tag", "--sigma", "--k-cut"),
}

#: A complete invocation of each verb, from which one flag at a time is
#: removed below. None of these is ever parsed to completion.
COMPLETE = {
    "train": ["train", "--system", "demo", "--run", "r1", "--seed", "0",
              "--steps", "10", "--source", "a=sub:*.npz:32,32,32",
              "--resume-optimizer", "no", "--anchors", "declared"],
    "diagnose": ["diagnose", "--system", "demo", "--ckpt", "x",
                 "--stage", "kappa"],
    "modes": ["modes", "--system", "demo", "--tag", "t1",
              "--sigma", "2.0", "--k-cut", "2.0"],
}


# ---------------------------------------------------------------------------
# the verbs exist and their help names what they need
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("verb", sorted(REQUIRED))
def test_verb_help_exits_zero_and_names_every_required_flag(verb):
    r = _aipf(verb, "--help")
    assert r.returncode == 0, r.stderr
    missing = [f for f in REQUIRED[verb] if f not in r.stdout]
    assert not missing, f"{verb} --help does not name {missing}:\n{r.stdout}"


def test_root_help_lists_the_three_verbs():
    r = _aipf("--help")
    assert r.returncode == 0, r.stderr
    for verb in sorted(REQUIRED):
        assert verb in r.stdout


def test_help_text_names_no_system():
    """Global constraint: the package's prose names no system or species.

    ``--help`` is prose that ships inside ``src/aipf``, and it is the one
    piece of it a reader sees without opening a file.
    """
    import re
    banned = re.compile(r"\b(hydrogen[ -]helium|iron[ -]boron|"
                        r"lennard[ -]jones|hhe|feb)\b", re.I)
    for verb in sorted(REQUIRED):
        r = _aipf(verb, "--help")
        assert not banned.search(r.stdout), f"{verb} --help: {r.stdout}"


# ---------------------------------------------------------------------------
# every required flag is refused when absent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("verb,flag", [(v, f) for v in sorted(REQUIRED)
                                       for f in REQUIRED[v]])
def test_missing_required_flag_is_a_usage_error_naming_it(verb, flag):
    """Drop one flag and nothing else: exit 2, with the flag named."""
    argv = list(COMPLETE[verb])
    if flag in argv:
        i = argv.index(flag)
        del argv[i:i + 2]
    else:
        # `--epochs` is the other half of the run-length pair: the complete
        # invocation carries `--steps`, so the case that proves `--epochs`
        # is not defaulted is the one where NEITHER is given.
        assert flag == "--epochs"
        i = argv.index("--steps")
        del argv[i:i + 2]
    r = _aipf(*argv)
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert flag in r.stderr, r.stderr


def test_train_refuses_both_a_step_count_and_an_epoch_count():
    r = _aipf(*COMPLETE["train"], "--epochs", "3")
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "--epochs" in r.stderr and "--steps" in r.stderr


def test_train_refuses_a_source_that_is_not_spelled_out_in_full():
    """A source names a tree, a glob and a grid. Two of the three is not a
    source with a default third; it is an unparsable request."""
    argv = [a if a != "a=sub:*.npz:32,32,32" else "a=sub:*.npz"
            for a in COMPLETE["train"]]
    r = _aipf(*argv)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "--source" in r.stderr


def test_diagnose_refuses_a_stage_it_does_not_have():
    r = _aipf("diagnose", "--system", "demo", "--ckpt", "x",
              "--stage", "bogus")
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    from aipf.diagnose.run import STAGES
    for stage in STAGES:
        assert stage in r.stderr, r.stderr


def test_diagnose_stage_choices_are_read_from_the_driver():
    """The admissible stages are the driver's tuple, not a retyped copy: a
    stage added there must appear on the command line without an edit here.
    """
    from aipf.cli.main import build_parser
    from aipf.diagnose.run import STAGES

    parser = build_parser()
    (verbs,) = [a for a in parser._actions if a.choices and "diagnose" in a.choices]
    diagnose = verbs.choices["diagnose"]
    (stage,) = [a for a in diagnose._actions if a.dest == "stage"]
    assert tuple(stage.choices) == tuple(STAGES)


# ---------------------------------------------------------------------------
# what each verb does with the values once they are all present
# ---------------------------------------------------------------------------
class _Paths:
    def __init__(self, root):
        self._root = root

    def raw(self):
        return self._root

    def data_root(self):
        return self._root / "farm"


class _System:
    """Enough of a System for the three ``run`` bodies, and no more.

    Its archive is laid out on construction: one run directory under each
    tree the in-process ``--source`` rows name, and a farm manifest that
    indexes the state point ``t1``, so a verb reaches its driver only when
    what it was asked for exists.
    """

    def __init__(self, root, defaults=None, checkpoint=None):
        self.name = "demo"
        self.paths = _Paths(root)
        self.defaults = dict(defaults or {})
        self.defaults.setdefault("training", {"source_root": {
            "tier": "raw", "path": "fields"}})
        self.checkpoint = checkpoint
        for tree in ("sub/T1000", "other/mT1000"):
            (root / "fields" / tree).mkdir(parents=True, exist_ok=True)
        self.paths.data_root().mkdir(parents=True, exist_ok=True)
        (self.paths.data_root() / "manifest.json").write_text(json.dumps(
            {"state_points": [{"tag": "t1", "farm_dir": "t1_part0"}]}))

    def resolve_checkpoint(self):
        return self.checkpoint.resolve(self.name, self.paths.raw)


class _Resolvable:
    """A declared checkpoint that some place holds: the CLI's pre-check passes and ``fit`` receives this object."""

    def __init__(self, path):
        self.path = path

    def resolve(self, *args, **kwargs):
        return self.path


class _Unresolvable:
    """A declared checkpoint no place holds, refused the way ``Checkpoint.resolve`` refuses."""

    def resolve(self, *args, **kwargs):
        raise FileNotFoundError("system 'demo': no location holds the declared checkpoint")


#: The one ``--source`` row the in-process train checks pass: its glob
#: selects the run directory ``_System`` lays out.
_SOURCE = "a=sub:T*:32,32,32"


def test_train_threads_the_declared_values_through_to_the_driver(
        monkeypatch, tmp_path, capsys):
    import importlib

    import aipf.system as system_mod
    from aipf.cli.main import main

    # The driver module, which is where `fit` lives and the only place it
    # is reachable from: no package of this project binds a name one of its
    # own submodules has.
    fit_mod = importlib.import_module("aipf.train.fit")
    system = _System(tmp_path, {"source_loss_weights": {"a": 0.5}})
    monkeypatch.setattr(system_mod, "load", lambda name: system)

    seen = {}

    def fake_fit(sys_, **kw):
        seen.update(kw)
        seen["system"] = sys_
        return tmp_path / "ckpt" / "r1"

    monkeypatch.setattr(fit_mod, "fit", fake_fit)

    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "7",
                 "--steps", "10", "--resume-optimizer", "no",
                 "--anchors", "declared",
                 "--source", _SOURCE,
                 "--source", "b=other:mT*:16,16,16"]) == 0

    assert seen["system"] is system
    assert seen["run_name"] == "r1"
    assert seen["seed"] == 7
    assert seen["steps"] == 10 and seen["epochs"] is None
    assert seen["init_from"] is None
    assert seen["resume_optimizer"] is False
    names = [s.name for s in seen["sources"]]
    assert names == ["a", "b"]           # in the order they were declared
    assert seen["sources"][0].pattern == "T*"
    assert seen["sources"][0].grid == (32, 32, 32)
    assert seen["sources"][0].root == str(tmp_path / "fields" / "sub")
    assert seen["sources"][0].loss_weight == 0.5    # the system's own weight
    assert seen["sources"][1].grid == (16, 16, 16)
    assert str(tmp_path / "ckpt" / "r1") in capsys.readouterr().out


def test_train_starts_from_the_declared_checkpoint_only_when_asked(
        monkeypatch, tmp_path):
    import importlib

    import aipf.system as system_mod
    from aipf.cli.main import main

    fit_mod = importlib.import_module("aipf.train.fit")
    marker = _Resolvable(tmp_path / "declared.ckpt")
    system = _System(tmp_path, checkpoint=marker)
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    seen = {}
    monkeypatch.setattr(fit_mod, "fit",
                        lambda sys_, **kw: seen.update(kw) or tmp_path)

    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "7",
                 "--epochs", "2", "--source", _SOURCE,
                 "--anchors", "declared",
                 "--resume-optimizer", "yes", "--init-from-published"]) == 0
    assert seen["init_from"] is marker
    assert seen["resume_optimizer"] is True
    assert seen["epochs"] == 2 and seen["steps"] is None


def test_train_refuses_an_unreachable_published_checkpoint_before_the_driver(
        monkeypatch, tmp_path, capsys):
    """``--init-from-published`` whose file no place holds is exit 2 and no run directory."""
    import importlib

    import aipf.system as system_mod
    from aipf.cli.main import main

    fit_mod = importlib.import_module("aipf.train.fit")
    system = _System(tmp_path, checkpoint=_Unresolvable())
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    monkeypatch.setattr(fit_mod, "fit", lambda *a, **kw: pytest.fail("fit was called"))
    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "7",
                 "--steps", "1", "--source", _SOURCE, "--init-from-published",
                 "--anchors", "none", "--resume-optimizer", "no"]) == 2
    assert "--init-from-published: the checkpoint system 'demo' declares is refused" in capsys.readouterr().err
    assert not (tmp_path / "ckpt" / "r1").exists()


@pytest.mark.parametrize("word,expected", [("yes", True), ("no", False)])
def test_train_passes_the_resume_choice_through_as_the_bool_it_said(
        monkeypatch, tmp_path, word, expected):
    """The flag is a CHOICE, and what reaches the driver is its bool.

    A fresh optimizer on warm weights takes its first steps at the linear
    warm-up floor, where no float32 parameter moves; a resumed one starts
    at the rate the saved run had reached. They are two experiments, so the
    driver takes a bool with no default and this asserts the word the
    operator typed is the bool that arrives -- driven through the REAL
    parser, with only the driver replaced.
    """
    import importlib

    import aipf.system as system_mod
    from aipf.cli.main import main

    fit_mod = importlib.import_module("aipf.train.fit")
    marker = _Resolvable(tmp_path / "declared.ckpt")
    system = _System(tmp_path, checkpoint=marker)
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    seen = {}
    monkeypatch.setattr(fit_mod, "fit",
                        lambda sys_, **kw: seen.update(kw) or tmp_path)

    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "7",
                 "--steps", "10", "--source", _SOURCE,
                 "--init-from-published",
                 "--anchors", "declared",
                 "--resume-optimizer", word]) == 0
    assert seen["resume_optimizer"] is expected
    assert isinstance(seen["resume_optimizer"], bool)


@pytest.mark.parametrize("word", ["declared", "none"])
def test_train_passes_the_anchor_choice_through_as_the_driver_spells_it(
        monkeypatch, tmp_path, word):
    """``--anchors`` reaches ``fit`` as one of its own two answers.

    ``declared`` arrives as ``None``, which is the driver's word for "the
    declaration answers" -- the same treatment ``split_mode`` and
    ``loader_order`` get. ``none`` arrives as the SENTINEL, because a
    drift-only run on a system that declares tables is an ablation and
    ``None`` already means "nobody said". Driven through the real parser,
    with only the driver replaced.
    """
    import importlib

    import aipf.system as system_mod
    from aipf.cli.main import main
    from aipf.train.anchors import NO_ANCHORS

    fit_mod = importlib.import_module("aipf.train.fit")
    monkeypatch.setattr(system_mod, "load", lambda name: _System(tmp_path))
    seen = {}
    monkeypatch.setattr(fit_mod, "fit",
                        lambda sys_, **kw: seen.update(kw) or tmp_path)

    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "7",
                 "--steps", "10", "--source", _SOURCE,
                 "--resume-optimizer", "no", "--anchors", word]) == 0
    assert seen["anchors"] is (None if word == "declared" else NO_ANCHORS)


def test_train_refuses_a_resume_with_nothing_to_resume_from(
        monkeypatch, tmp_path, capsys):
    """``--resume-optimizer yes`` without ``--init-from-published``.

    The two flags contradict each other: there is no saved optimizer to
    continue without a checkpoint to read one from. Refused here by name so
    the operator is told which flag to change, and refused again inside
    ``fit`` (``test_train_fit.py``), so removing either guard leaves the
    other.
    """
    import importlib

    import aipf.system as system_mod
    from aipf.cli.main import main

    fit_mod = importlib.import_module("aipf.train.fit")
    monkeypatch.setattr(system_mod, "load", lambda name: _System(tmp_path))
    monkeypatch.setattr(fit_mod, "fit", lambda *a, **k: pytest.fail(
        "a run must not start when it was told to continue an optimizer "
        "that no checkpoint carries"))

    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "7",
                 "--steps", "10", "--source", _SOURCE,
                 "--anchors", "declared",
                 "--resume-optimizer", "yes"]) == 2
    err = capsys.readouterr().err
    assert "--resume-optimizer" in err and "--init-from-published" in err


def test_train_refuses_to_start_from_a_checkpoint_the_system_does_not_declare(
        monkeypatch, tmp_path, capsys):
    import importlib

    import aipf.system as system_mod
    from aipf.cli.main import main

    fit_mod = importlib.import_module("aipf.train.fit")
    monkeypatch.setattr(system_mod, "load",
                        lambda name: _System(tmp_path, checkpoint=None))
    monkeypatch.setattr(fit_mod, "fit", lambda *a, **k: pytest.fail(
        "a run must not start when the weights it was told to start from "
        "do not exist"))

    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "7",
                 "--steps", "10", "--source", _SOURCE,
                 "--anchors", "declared",
                 "--resume-optimizer", "no", "--init-from-published"]) == 2
    assert "checkpoint" in capsys.readouterr().err


def test_diagnose_forwards_the_stages_and_the_declared_block(
        monkeypatch, tmp_path, capsys):
    import importlib

    from aipf.cli.main import main
    import aipf.system as system_mod

    run_mod = importlib.import_module("aipf.diagnose.run")
    declared = {"tc_method": "dome_apex", "poly_degree": 4}
    system = _System(tmp_path, {"diagnose": declared})
    monkeypatch.setattr(system_mod, "load", lambda name: system)

    seen = {}

    def fake_run(sys_, ckpt, **kw):
        seen["ckpt"] = ckpt
        seen.update(kw)
        return tmp_path / "out" / "abcdef012345"

    monkeypatch.setattr(run_mod, "run", fake_run)

    assert main(["diagnose", "--system", "demo", "--ckpt", "some.ckpt",
                 "--stage", "dome", "--stage", "tc",
                 "--out", str(tmp_path / "out"),
                 "--pressure", "200", "--pressure", "400",
                 "--T-grid", "2000,2200,100"]) == 0

    assert str(seen["ckpt"]) == "some.ckpt"
    assert tuple(seen["stages"]) == ("dome", "tc")
    assert str(seen["out"]) == str(tmp_path / "out")
    assert tuple(seen["pressures"]) == (200.0, 400.0)
    assert [float(t) for t in seen["T_grid"]] == [2000.0, 2100.0, 2200.0]
    assert seen["tc_method"] == "dome_apex" and seen["poly_degree"] == 4
    assert "abcdef012345" in capsys.readouterr().out


def test_diagnose_asks_for_no_stage_the_operator_did_not_name(
        monkeypatch, tmp_path):
    import importlib

    from aipf.cli.main import main
    import aipf.system as system_mod

    run_mod = importlib.import_module("aipf.diagnose.run")
    monkeypatch.setattr(system_mod, "load", lambda name: _System(tmp_path))
    seen = {}
    monkeypatch.setattr(run_mod, "run",
                        lambda s, c, **kw: seen.update(kw) or tmp_path)

    assert main(["diagnose", "--system", "demo", "--ckpt", "c",
                 "--stage", "kappa", "--out", str(tmp_path)]) == 0
    assert tuple(seen["stages"]) == ("kappa",)
    # Nothing was invented for the question the phase diagram asks.
    assert tuple(seen["pressures"]) == ()
    assert seen["T_grid"] is None


def test_modes_prints_where_the_archive_landed(monkeypatch, tmp_path, capsys):
    import importlib

    from aipf.cli.main import main
    import aipf.system as system_mod

    modes_mod = importlib.import_module("aipf.pipeline.modes")
    monkeypatch.setattr(system_mod, "load", lambda name: _System(tmp_path))
    entry = tmp_path / "modes" / "t1" / "deadbeef"
    seen = {}

    def fake_modes(system, tag, **kw):
        seen["tag"] = tag
        seen.update(kw)
        return types.SimpleNamespace(provenance={"cache_dir": str(entry)})

    monkeypatch.setattr(modes_mod, "modes", fake_modes)

    assert main(["modes", "--system", "demo", "--tag", "t1",
                 "--sigma", "2.0", "--k-cut", "1.5"]) == 0
    assert seen["tag"] == "t1"
    assert seen["sigma"] == 2.0 and seen["k_cut"] == 1.5
    # Neither named choice is decided here when the operator named neither.
    assert "ordering" not in seen and "route" not in seen
    assert str(entry) in capsys.readouterr().out


def test_modes_passes_a_named_choice_through_when_it_is_given(
        monkeypatch, tmp_path):
    import importlib

    from aipf.cli.main import main
    import aipf.system as system_mod

    modes_mod = importlib.import_module("aipf.pipeline.modes")
    monkeypatch.setattr(system_mod, "load", lambda name: _System(tmp_path))
    seen = {}

    def fake_modes(system, tag, **kw):
        seen.update(kw)
        return types.SimpleNamespace(provenance={"cache_dir": str(tmp_path)})

    monkeypatch.setattr(modes_mod, "modes", fake_modes)

    assert main(["modes", "--system", "demo", "--tag", "t1",
                 "--sigma", "2.0", "--k-cut", "1.5",
                 "--ordering", "shell", "--route", "separable"]) == 0
    assert seen["ordering"] == "shell" and seen["route"] == "separable"


# ---------------------------------------------------------------------------
# what is checked at the door, before any driver is imported
# ---------------------------------------------------------------------------
def _declared_published(root, *, digest_of=b"the published model's bytes"):
    """A real :class:`~aipf.system.Checkpoint` over a file under ``root``.

    ``digest_of`` is what the declared md5 is taken of: the file's own
    bytes by default, anything else to declare a digest the file fails.
    """
    from aipf.system import Checkpoint

    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "best.ckpt").write_bytes(b"the published model's bytes")
    return Checkpoint(path="runs/best.ckpt",
                      md5=hashlib.md5(digest_of).hexdigest())


def _diagnose_driver_must_not_run(monkeypatch):
    import importlib

    run_mod = importlib.import_module("aipf.diagnose.run")
    monkeypatch.setattr(run_mod, "run", lambda *a, **k: pytest.fail(
        "the diagnosis must not start on a checkpoint the door refused"))


def test_diagnose_published_is_the_declared_file_once_its_digest_is_verified(
        monkeypatch, tmp_path):
    import importlib

    from aipf.cli.main import main
    import aipf.system as system_mod

    run_mod = importlib.import_module("aipf.diagnose.run")
    system = _System(tmp_path, checkpoint=_declared_published(tmp_path))
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    seen = {}
    monkeypatch.setattr(run_mod, "run",
                        lambda s, c, **kw: seen.update(ckpt=c) or tmp_path)

    assert main(["diagnose", "--system", "demo", "--ckpt", "published",
                 "--stage", "kappa", "--out", str(tmp_path)]) == 0
    assert seen["ckpt"] == tmp_path / "runs" / "best.ckpt"


@pytest.mark.parametrize("fault", ["undeclared", "wrong digest", "missing"])
def test_diagnose_published_refusal_exits_2_naming_the_system(
        monkeypatch, tmp_path, capsys, fault):
    """No declaration, a digest the file fails, a declared file that is
    not there: each exits 2 with the system named, and the driver never
    runs."""
    from aipf.cli.main import main
    import aipf.system as system_mod
    from aipf.system import Checkpoint

    checkpoint = {
        "undeclared": lambda: None,
        "wrong digest": lambda: _declared_published(
            tmp_path, digest_of=b"some other file"),
        "missing": lambda: Checkpoint(path="runs/absent.ckpt",
                                      md5="0" * 32),
    }[fault]()
    system = _System(tmp_path, checkpoint=checkpoint)
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    _diagnose_driver_must_not_run(monkeypatch)

    assert main(["diagnose", "--system", "demo", "--ckpt", "published",
                 "--stage", "kappa", "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "--ckpt published" in err and "'demo'" in err, err


def test_train_refuses_a_source_glob_that_selects_nothing(
        monkeypatch, tmp_path, capsys):
    """A glob with no run directory under its tree exits 2 before ``fit``
    runs, naming the source, its TREE and the archive root; a glob that
    matches only FILES selects no run either."""
    import importlib

    import aipf.system as system_mod
    from aipf.cli.main import main

    fit_mod = importlib.import_module("aipf.train.fit")
    system = _System(tmp_path)
    (tmp_path / "fields" / "sub" / "T2000.npz").write_bytes(b"")
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    monkeypatch.setattr(fit_mod, "fit", lambda *a, **k: pytest.fail(
        "a run must not start on a source that selects no run"))

    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "7",
                 "--steps", "10", "--source", _SOURCE,
                 "--source", "b=sub:X*:16,16,16",
                 "--source", "c=sub:*.npz:16,16,16",
                 "--source", "d=nowhere:T*:16,16,16",
                 "--anchors", "declared", "--resume-optimizer", "no"]) == 2
    err = capsys.readouterr().err
    root = tmp_path / "fields"
    assert "--source a" not in err, err
    for name, tree in (("b", "sub"), ("c", "sub"), ("d", "nowhere")):
        assert f"--source {name}:" in err, err
        assert str(root / tree) in err, err
    assert f"the archive root is {root}" in err, err


def test_the_door_checks_the_tree_the_driver_reads(monkeypatch, tmp_path):
    """The root the refusal names is the one ``_specs_from`` builds."""
    from aipf.cli.train_cmd import unmatched_sources
    from aipf.train.fit import _specs_from

    system = _System(tmp_path)
    (spec,) = _specs_from(system, [("a", "gone", "T*", (1, 1, 1))], ["a"])
    (refusal,) = unmatched_sources(system, [("a", "gone", "T*", (1, 1, 1))])
    assert f"under {spec.root} " in refusal


@pytest.mark.parametrize("manifest", [True, False])
def test_modes_refuses_a_tag_the_archive_does_not_index(
        monkeypatch, tmp_path, capsys, manifest):
    import importlib

    from aipf.cli.main import main
    import aipf.system as system_mod

    modes_mod = importlib.import_module("aipf.pipeline.modes")
    system = _System(tmp_path)
    if not manifest:
        (system.paths.data_root() / "manifest.json").unlink()
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    monkeypatch.setattr(modes_mod, "modes", lambda *a, **k: pytest.fail(
        "no extraction may start for a state point the archive lacks"))

    assert main(["modes", "--system", "demo", "--tag", "t9",
                 "--sigma", "2.0", "--k-cut", "1.5"]) == 2
    err = capsys.readouterr().err
    assert "--tag t9" in err, err
    assert f"the archive root is {system.paths.data_root()}" in err, err


# ---------------------------------------------------------------------------
# what building the parser is allowed to cost
# ---------------------------------------------------------------------------
#: Import roots no command line may pay for before it has parsed anything.
_DEEP_LEARNING = ("torch", "lightning", "numpy", "scipy", "matplotlib", "yaml")


def test_building_the_command_line_imports_no_deep_learning_stack():
    """The parser is built on every invocation, `--help` and `--version` too.

    `--stage`'s choices have to be known when the parser is built, and
    reading them off the diagnosis DRIVER pulled torch and lightning in
    through that module -- measured at 4.4 s cumulative, on every
    subcommand, including ones that diagnose nothing. The stage names live
    in a stdlib-only module for that reason, and this is the check whose
    absence let the regression land.
    """
    code = (
        "import sys\n"
        "from aipf.cli.main import build_parser\n"
        "build_parser()\n"
        f"roots = set({_DEEP_LEARNING!r})\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} & roots))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", (
        f"building the parser imported {r.stdout.strip()}")


def test_the_stage_names_are_one_tuple_the_driver_and_the_parser_share():
    from aipf.diagnose.run import STAGES as from_driver
    from aipf.diagnose.stages import STAGES as from_names

    assert from_driver is from_names


def test_a_source_naming_an_absolute_tree_is_refused(tmp_path):
    import argparse

    from aipf.cli.train_cmd import source_row
    from aipf.train.fit import _specs_from
    with pytest.raises(argparse.ArgumentTypeError, match="absolute"):
        source_row(f"a={tmp_path}:T*:1,1,1")
    with pytest.raises(ValueError, match="absolute"):
        _specs_from(_System(tmp_path), [("a", str(tmp_path), "T*", (1, 1, 1))],
                    ["a"])


@pytest.mark.parametrize("declared, match", [
    ({"tier": "home", "path": "fields"}, "not one of"),
    ({"tier": "raw"}, "exactly the keys"),
    ({"tier": "raw", "path": "/fields"}, "absolute"),
])
def test_the_source_root_is_a_checked_declaration(tmp_path, declared, match):
    from aipf.train.fit import source_root
    system = _System(tmp_path, {"training": {"source_root": declared}})
    with pytest.raises(ValueError, match=match):
        source_root(system)


def test_the_farm_tier_is_the_systems_data_root(tmp_path):
    from aipf.train.fit import source_root
    system = _System(tmp_path, {"training": {"source_root": {
        "tier": "farm", "path": "modes"}}})
    assert source_root(system) == system.paths.data_root() / "modes"
    bare = _System(tmp_path, {"training": {}})
    with pytest.raises(KeyError, match="source_root"):
        source_root(bare)


# ---------------------------------------------------------------------------
# --variant: a declared model beside the production one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verb", [
    ["diagnose", "--system", "lj", "--ckpt", "published", "--stage", "kappa"],
    ["train", "--system", "lj", "--run", "r", "--seed", "0", "--steps", "1",
     "--source", "slab=x:*:20,20,80", "--resume-optimizer", "no", "--anchors", "none"],
    ["rollout", "slab", "--system", "lj", "--ckpt", "published", "--run", "r", "--seeds",
     "--t-end", "1", "--dt", "1", "--save-ps", "1", "--device", "cpu", "--out", "data"],
])
def test_an_undeclared_variant_is_refused_by_every_verb(verb, capsys):
    from aipf.cli.main import main
    assert main(verb + ["--variant", "nope"]) == 2
    assert "declares no variant 'nope'" in capsys.readouterr().err


def test_rollout_hands_the_driver_the_variant_system(monkeypatch, tmp_path):
    import aipf.cli.rollout_cmd as rollout_cmd
    import aipf.rollout.slab as slab_mod
    from aipf.cli.main import main
    seen = {}
    # the variant declares no rollout block; what is under test is which system reaches the driver
    monkeypatch.setattr(rollout_cmd, "refusal", lambda system, driver, ckpt: None)
    monkeypatch.setattr(slab_mod, "slab", lambda system, ckpt, **kw: seen.update(system=system) or tmp_path)
    assert main(["rollout", "slab", "--system", "lj", "--variant", "landau", "--ckpt", "published",
                 "--run", "r", "--seeds", "--t-end", "1", "--dt", "1", "--save-ps", "1",
                 "--device", "cpu", "--out", "data"]) == 0
    assert seen["system"].functional.local == "landau"
    assert seen["system"].variant_name == "landau"
