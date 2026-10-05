"""The published checkpoints travel with the repository.

Each system's declared checkpoint is committed as a real file at ``data/<system>/ckpt/published/final.ckpt``,
and a declared variant's at ``data/<system>/ckpt/published/<variant>/final.ckpt``: no run, wave, seed or
epoch name in the path. The rest of ``data/`` stays ignored. With the raw root pointed at an empty
directory, ``System.resolve_checkpoint`` finds the tracked file, the evaluation pipeline's loader reads it
strictly, and ``aipf diagnose --ckpt published`` resolves it. The resolution order and its refusals are
exercised on a toy checkpoint with the published directory moved to ``tmp_path``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import subprocess
from pathlib import Path

import pytest

from aipf.system import TRACKED_FILE_NAME, Checkpoint, load

import declared_roots

REPO = Path(__file__).resolve().parents[2]

#: The systems whose paper figures load a checkpoint.
SYSTEMS = ("hhe", "feb", "lj")

#: Every tracked checkpoint, as the repository spells it.
TRACKED = {
    ("hhe", None): "data/hhe/ckpt/published/final.ckpt",
    ("feb", None): "data/feb/ckpt/published/final.ckpt",
    ("lj", None): "data/lj/ckpt/published/final.ckpt",
    ("lj", "fh"): "data/lj/ckpt/published/fh/final.ckpt",
    ("lj", "landau"): "data/lj/ckpt/published/landau/final.ckpt",
}
KEYS = sorted(TRACKED, key=str)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _tracked(name: str, variant: str | None = None) -> Path:
    return REPO / TRACKED[(name, variant)]


def _in_git() -> bool:
    return (REPO / ".git").exists()


def _ignored(rel: str) -> bool:
    """``git check-ignore`` against the rules alone, whether or not the path is in the index."""
    done = subprocess.run(["git", "-C", str(REPO), "check-ignore", "--no-index", "-q", rel])
    return done.returncode == 0


@pytest.fixture
def empty_raw(monkeypatch, tmp_path):
    """Every raw root is an empty directory, as on a fresh clone; the data root is the repository's."""
    root = tmp_path / "empty_raw"
    root.mkdir()
    for name in SYSTEMS:
        monkeypatch.delenv(f"AIPF_RAW_{name.upper()}", raising=False)
    monkeypatch.delenv("AIPF_DATA", raising=False)
    monkeypatch.setenv("AIPF_RAW", str(root))
    return root


def test_the_file_name_is_the_package_convention():
    assert TRACKED_FILE_NAME == "final.ckpt"


@pytest.mark.parametrize("name", SYSTEMS)
def test_the_published_directory_is_under_the_data_root(name, monkeypatch):
    monkeypatch.delenv("AIPF_DATA", raising=False)
    assert load(name).paths.published() == REPO / "data" / name / "ckpt" / "published"


@pytest.mark.parametrize("name,variant", KEYS)
def test_each_tracked_file_carries_the_declared_digest(name, variant, monkeypatch):
    declared_roots.published_or_skip(name, variant)
    monkeypatch.delenv("AIPF_DATA", raising=False)
    system = load(name) if variant is None else load(name).variant(variant)
    path = Checkpoint.tracked_under(system.paths.published(), variant)
    assert path == _tracked(name, variant)
    assert path.is_file() and not path.is_symlink(), f"{path} is not a real file"
    assert _md5(path) == system.checkpoint.md5


@pytest.mark.parametrize("name,variant", KEYS)
def test_each_tracked_file_is_in_git_and_not_ignored(name, variant):
    declared_roots.published_or_skip(name, variant)
    if not _in_git():
        pytest.skip("not a git checkout, so there is no index or ignore rule to ask")
    rel = TRACKED[(name, variant)]
    assert not _ignored(rel), f"{rel} is ignored by git"
    listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel],
                            capture_output=True)
    assert listed.returncode == 0, f"{rel} is not tracked"


@pytest.mark.parametrize("rel", [
    "data/hhe/manifest.json", "data/hhe/md/x", "data/lj/modes", "data/tools/x", "data/x.txt",
    "data/hhe/ckpt/smoke/final.ckpt", "data/feb/ckpt/some_run/MANIFEST.json",
    "data/lj/ckpt/sample/final.ckpt", "data/feb/samples/x",
])
def test_the_rest_of_the_data_farm_stays_ignored(rel):
    if not _in_git():
        pytest.skip("not a git checkout, so there is no ignore rule to ask")
    assert _ignored(rel), f"{rel} is not ignored"


def _sample_files() -> set:
    """The training sample's files as its manifests list them, the manifests included."""
    import json
    found = set()
    for manifest in (REPO / "data").glob("*/sample/MANIFEST.json"):
        base = manifest.parent.relative_to(REPO)
        found.add(str(base / manifest.name))
        found.update(str(base / e["path"]) for e in json.loads(manifest.read_text())["files"])
    return found


def test_the_only_tracked_files_under_data_are_the_published_checkpoints_and_the_sample():
    if not _in_git():
        pytest.skip("not a git checkout")
    listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "data"],
                            capture_output=True, text=True, check=True).stdout.split()
    # a checkout may leave the checkpoints out; beside them it tracks only the files the training
    # sample's manifests list
    allowed = set(TRACKED.values()) | _sample_files()
    assert set(listed) <= allowed, sorted(set(listed) - allowed)


def test_no_tracked_path_names_a_run():
    for rel in TRACKED.values():
        for word in ("wave", "seed", "epoch", "_s1", "base_", "prod_", "feb_w", "checkpoints"):
            assert word not in rel, (rel, word)


@pytest.mark.parametrize("name", SYSTEMS)
def test_an_empty_raw_root_resolves_to_the_tracked_file(name, empty_raw):
    """A system declares its checkpoint by digest alone; every reader lands on the tracked file."""
    declared_roots.published_or_skip(name)
    system = load(name)
    assert system.checkpoint.path is None
    assert system.checkpoint_file() == _tracked(name)
    assert system.resolve_checkpoint() == _tracked(name)
    assert system.verify_checkpoint() == _tracked(name)


@pytest.mark.parametrize("name", SYSTEMS)
def test_the_evaluation_loader_reads_the_tracked_file_strictly(name, empty_raw):
    declared_roots.published_or_skip(name)
    pytest.importorskip("torch")
    from aipf.diagnose.run import load_model

    system = load(name)
    model = load_model(system, system.resolve_checkpoint())
    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.parametrize("name", SYSTEMS)
def test_diagnose_published_resolves_with_an_empty_raw_root(
        name, empty_raw, monkeypatch, tmp_path):
    declared_roots.published_or_skip(name)
    from aipf.cli.main import main

    run_mod = importlib.import_module("aipf.diagnose.run")
    seen = {}
    monkeypatch.setattr(run_mod, "run",
                        lambda s, c, **kw: seen.update(ckpt=c) or tmp_path)
    assert main(["diagnose", "--system", name, "--ckpt", "published",
                 "--stage", "kappa", "--out", str(tmp_path / "out")]) == 0
    assert seen["ckpt"] == _tracked(name)


def _system(name: str, variant: str | None):
    return load(name) if variant is None else load(name).variant(variant)


@pytest.mark.parametrize("name,variant", KEYS)
def test_the_three_readers_of_the_declared_checkpoint_agree(name, variant, empty_raw):
    """``checkpoint_file``, ``verify_checkpoint`` and ``resolve_checkpoint`` locate one file, the tracked one."""
    declared_roots.published_or_skip(name, variant)
    system = _system(name, variant)
    want = Checkpoint.tracked_under(system.paths.published(), variant)
    assert want == _tracked(name, variant)
    assert system.checkpoint_file() == system.verify_checkpoint() == system.resolve_checkpoint() == want


def test_a_rebuild_leaves_the_tracked_checkpoints_alone(empty_raw, tmp_path):
    """The farm links no checkpoint: a rebuild writes the md tier only."""
    for key in KEYS:
        declared_roots.published_or_skip(*key)
    import json

    from aipf.data import index

    before = {key: _md5(_tracked(*key)) for key in KEYS}
    manifest = tmp_path / "data" / "lj" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"system": "lj", "state_points": []}))
    assert index.rebuild_from_manifest(manifest) == 0
    assert not (tmp_path / "data" / "lj" / "ckpt").exists()
    assert {key: _md5(_tracked(*key)) for key in KEYS} == before


def test_an_unknown_config_schema_tag_is_refused_by_name(tmp_path):
    pytest.importorskip("torch")
    import torch

    from aipf.diagnose.run import load_model
    from aipf.train.fit import _load_weights_into

    path = tmp_path / "tagged.ckpt"
    torch.save({"config_schema": "someone.else.Config.v9", "model_state_dict": {}}, str(path))
    system = load("lj")
    with pytest.raises(ValueError, match="someone.else.Config.v9"):
        load_model(system, path)
    with pytest.raises(ValueError, match="someone.else.Config.v9"):
        _load_weights_into(None, Checkpoint(md5="0" * 32), tmp_path, path=path)


# ---------------------------------------------------------------------------
# The declared variants' checkpoints travel the same way
# ---------------------------------------------------------------------------

#: (system, variant) for every variant a system declares with a checkpoint.
VARIANTS = tuple((name, v) for name in SYSTEMS for v, spec in load(name).variants.items()
                 if spec.checkpoint is not None)


def test_the_baselines_are_the_declared_variants_with_a_checkpoint():
    assert set(VARIANTS) == {k for k in TRACKED if k[1] is not None}


@pytest.mark.parametrize("name,variant", VARIANTS)
def test_a_variant_loads_strictly_from_the_tracked_file(name, variant, empty_raw):
    declared_roots.published_or_skip(name, variant)
    pytest.importorskip("torch")
    from aipf.diagnose.run import load_model

    system = load(name).variant(variant)
    assert system.variant_name == variant
    path = system.resolve_checkpoint()
    assert path == _tracked(name, variant)
    model = load_model(system, path)
    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.parametrize("name,variant", VARIANTS)
def test_diagnose_variant_published_resolves_the_variants_file(
        name, variant, empty_raw, monkeypatch, tmp_path):
    declared_roots.published_or_skip(name, variant)
    from aipf.cli.main import main

    run_mod = importlib.import_module("aipf.diagnose.run")
    seen = {}
    monkeypatch.setattr(run_mod, "run",
                        lambda s, c, **kw: seen.update(system=s, ckpt=c) or tmp_path)
    assert main(["diagnose", "--system", name, "--variant", variant, "--ckpt", "published",
                 "--stage", "kappa", "--out", str(tmp_path / "out")]) == 0
    assert seen["ckpt"] == _tracked(name, variant)
    assert seen["system"].functional == load(name).variants[variant].functional


# ---------------------------------------------------------------------------
# The resolution order, on a toy checkpoint
# ---------------------------------------------------------------------------

BODY = b"the declared bytes"
DECLARED = Checkpoint(path="runs/r1/best.ckpt", md5=hashlib.md5(BODY).hexdigest())


@pytest.fixture
def roots(tmp_path):
    """(published directory of system ``toy``, raw root), the raw root created and empty."""
    published = tmp_path / "data" / "toy" / "ckpt" / "published"
    raw = tmp_path / "raw"
    raw.mkdir()
    return published, raw


def _write(path: Path, body: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def test_the_tracked_file_wins_over_the_raw_copy(roots):
    published, raw = roots
    _write(raw / DECLARED.path, BODY)
    want = _write(published / "final.ckpt", BODY)
    assert DECLARED.resolve("toy", raw, published) == want


def test_a_variant_is_one_directory_below(roots):
    published, raw = roots
    _write(published / "final.ckpt", b"the production model")
    want = _write(published / "v" / "final.ckpt", BODY)
    assert DECLARED.resolve("toy", raw, lambda: published, "v") == want


def test_other_bytes_in_the_tracked_file_fall_through_to_the_raw_root(roots):
    published, raw = roots
    _write(published / "final.ckpt", b"something else")
    want = _write(raw / DECLARED.path, BODY)
    assert DECLARED.resolve("toy", lambda: raw, published) == want


def test_nowhere_is_a_missing_file_refusal_naming_the_system(roots):
    published, raw = roots
    with pytest.raises(FileNotFoundError, match="system 'toy'.*missing or renamed"):
        DECLARED.resolve("toy", raw, published)


def test_other_bytes_everywhere_is_a_digest_refusal_naming_both(roots):
    published, raw = roots
    a = _write(published / "final.ckpt", b"one")
    b = _write(raw / DECLARED.path, b"two")
    with pytest.raises(ValueError, match="system 'toy'") as info:
        DECLARED.resolve("toy", raw, published)
    assert str(a) in str(info.value) and str(b) in str(info.value)


def test_an_unconfigured_raw_root_is_one_more_place_that_does_not_hold_it(roots):
    published, _ = roots

    def unset():
        raise RuntimeError("no raw location for system 'toy'")
    with pytest.raises(FileNotFoundError, match="no raw location"):
        DECLARED.resolve("toy", unset, published)


def test_without_a_published_directory_only_the_raw_root_is_looked_in(roots):
    published, raw = roots
    _write(published / "final.ckpt", BODY)
    with pytest.raises(FileNotFoundError):
        DECLARED.resolve("toy", raw)


@pytest.mark.parametrize("variant", ["", ".", "..", "a/b"])
def test_a_variant_that_is_not_one_directory_name_is_refused(variant):
    with pytest.raises(ValueError, match="one directory name"):
        Checkpoint.tracked_under(Path("published"), variant)


def test_a_checkpoint_declares_no_run():
    assert [f.name for f in dataclasses.fields(Checkpoint)] == ["md5", "path"]


@pytest.mark.parametrize("name", SYSTEMS)
def test_the_package_readers_of_the_declared_checkpoint_find_the_tracked_file(name, empty_raw):
    declared_roots.published_or_skip(name)
    pytest.importorskip("torch")
    from aipf.diagnose.run import resolve_checkpoint as diagnose_resolve
    from aipf.rollout.spinodal import resolve_checkpoint as rollout_resolve

    system = load(name)
    assert diagnose_resolve(system, system.checkpoint) == _tracked(name)
    assert rollout_resolve(system, "published") == _tracked(name)


def test_a_training_run_may_not_be_named_published(tmp_path):
    pytest.importorskip("torch")
    from aipf.train.fit import fit

    with pytest.raises(ValueError, match="published checkpoints"):
        fit(load("lj"), run_name="published", sources=[object()], seed=0,
            resume_optimizer=False, steps=1, root=tmp_path)
