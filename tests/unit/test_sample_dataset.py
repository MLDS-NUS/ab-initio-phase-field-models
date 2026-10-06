"""The training sample bundled with each system, ``data/<system>/sample``, and ``aipf train --source sample``.

The sample exists so that the training pipeline runs from a clean checkout with no raw data root. It is
a few mode runs cut to one training window (the Lennard-Jones runs to the windows that fill one batch)
and to the modes with ``max |n_i| <= 2``, beside the anchor tables the declaration reads. The fast tests
check the files against their manifest and the declaration; the slow one trains one epoch per system.
"""
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from aipf import paths
from aipf.system import load

SYSTEMS = ("feb", "hhe", "lj")
MAX_ABS_N = 2

#: A string that names a file: an absolute or home path, or a relative one with a directory part.
PATH_LIKE = re.compile(r"(^|[\s\"'=(:])(/|~/|\.\.?/)|[A-Za-z_][\w.-]*/[\w.-]+")


#: Systems whose sample is trained with ``--anchors none`` (the drift term and the kernel hinge).
DRIFT_ONLY_SAMPLE = ("feb",)

def _manifest(name):
    return json.loads((paths.sample_dir(name) / "MANIFEST.json").read_text())


def _mode_files(name):
    return [e for e in _manifest(name)["files"] if e["path"].endswith("modes.npz")]


@pytest.mark.parametrize("name", SYSTEMS)
def test_every_file_is_in_the_manifest_with_its_md5(name):
    root = paths.sample_dir(name)
    listed = {e["path"]: e for e in _manifest(name)["files"]}
    on_disk = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"MANIFEST.json"}
    assert on_disk == set(listed)
    for rel, entry in listed.items():
        data = (root / rel).read_bytes()
        assert len(data) == entry["bytes"], rel
        assert hashlib.md5(data).hexdigest() == entry["md5"], rel
        assert len(data) < 2_000_000, rel


@pytest.mark.parametrize("name", SYSTEMS)
def test_the_manifest_names_no_absolute_path(name):
    text = (paths.sample_dir(name) / "MANIFEST.json").read_text()
    assert not re.search(r"\"/|\"~|[A-Za-z]:\\\\", text)


@pytest.mark.parametrize("name", SYSTEMS)
def test_the_declaration_names_the_sources_the_manifest_lists(name):
    declared = load(name).defaults["training"]["sample"]
    listed = _manifest(name)["sources"]
    assert {n: {**r, "grid": list(r["grid"])} for n, r in declared.items()} == listed
    for entry in _mode_files(name):
        row = declared[entry["source"]]
        assert entry["path"].startswith(row["root"] + "/")
        assert Path(entry["path"]).parent.match(row["pattern"])


@pytest.mark.parametrize("name", SYSTEMS)
def test_each_run_holds_the_windows_its_manifest_says(name):
    """One window, unless the declared loader drops a partial batch the runs would not fill."""
    from aipf.train.fit import _window_from_system
    system = load(name)
    training = system.defaults["training"]
    w, stride = training["half_width"], training["stride"]
    window = _window_from_system(system)
    runs = _mode_files(name)
    for entry in runs:
        n = entry["windows"]
        if not training["drop_last"]:
            assert n == 1, entry["path"]
        else:
            assert n == math.ceil(training["batch_size"] / len(runs)), entry["path"]
        with np.load(paths.sample_dir(name) / entry["path"]) as z:
            frames = z["rho_k"].shape[0]
            assert frames == entry["frames"] == 2 * w + 1 + (n - 1) * stride
            assert z["box"].shape[0] == frames if z["box"].ndim == 2 else z["box"].shape == (3,)
            if "timesteps" in z.files:
                assert z["timesteps"].shape == (frames,)
        assert len(window.centres(frames)) == n, entry["path"]


@pytest.mark.parametrize("name", SYSTEMS)
def test_the_modes_are_within_the_cut(name):
    for entry in _mode_files(name):
        with np.load(paths.sample_dir(name) / entry["path"]) as z:
            nvec, rho_k = z["nvec"], z["rho_k"]
        assert np.abs(nvec).max() <= MAX_ABS_N
        assert (nvec[:, 2] >= 0).all()
        assert rho_k.shape[1] == nvec.shape[0] == entry["modes"]
        assert np.isfinite(rho_k).all()


@pytest.mark.parametrize("name", SYSTEMS)
def test_no_string_array_holds_a_path(name):
    for path in paths.sample_dir(name).rglob("*.npz"):
        with zipfile.ZipFile(path) as archive, np.load(path) as z:
            assert all(m.endswith(".npy") for m in archive.namelist())
            for key in z.files:
                array = z[key]
                if array.dtype.kind in "US":
                    bad = [v for v in np.atleast_1d(array).ravel().tolist() if PATH_LIKE.search(str(v))]
                    assert not bad, (path.name, key, bad[:3])


def test_the_sample_does_not_move_with_the_data_root(monkeypatch, tmp_path):
    """Like the published checkpoints, the sample is a tracked file of the checkout."""
    before = paths.sample_dir("feb")
    monkeypatch.setenv("AIPF_DATA", str(tmp_path))
    assert paths.sample_dir("feb") == before == paths.repo_root() / "data" / "feb" / "sample"


@pytest.mark.parametrize("name", SYSTEMS)
def test_the_sample_system_reads_no_raw_root(name, monkeypatch, tmp_path):
    """With the raw root pointed at an empty directory, every source and table still resolves."""
    from aipf.cli.train_cmd import unmatched_sources
    from aipf.train.anchors import _resolve
    from aipf.train.fit import declared_source_table, sample_system, source_root
    monkeypatch.setenv(paths.raw_env_name(name), str(tmp_path))
    system = sample_system(load(name))
    assert system.paths.raw() == paths.sample_dir(name)
    assert source_root(system) == paths.sample_dir(name) / "."
    table = declared_source_table(system)
    assert [row[0] for row in table] == list(load(name).defaults["training"]["sample"])
    assert unmatched_sources(system, table) == []
    tables = system.defaults["training"]["tables"]
    roles = ("eos_csvs",) if name in DRIFT_ONLY_SAMPLE else ("m_table", "s_table", "eos_csvs")
    for role in roles:
        if role in tables:
            declared = tables[role] if isinstance(tables[role], dict) else {0.0: tables[role]}
            for _, found in _resolve(system.paths.raw, declared, role, paths.tracked_tables(name)):
                assert tmp_path not in found.parents


def test_source_sample_stands_alone(capsys):
    from aipf.cli.main import main
    code = main(["train", "--system", "lj", "--run", "x", "--seed", "0", "--epochs", "1",
                 "--source", "sample", "--source", "a=.:b:1,1,1",
                 "--resume-optimizer", "no", "--anchors", "declared"])
    assert code == 2
    assert "--source sample stands alone" in capsys.readouterr().err


@pytest.mark.slow
@pytest.mark.parametrize("name", SYSTEMS)
def test_one_epoch_on_the_sample_trains_every_declared_term(name, monkeypatch, tmp_path):
    """``aipf train --source sample`` from the declaration alone, no raw root: every step finite and no
    declared term left without data."""
    from aipf.cli.main import main
    monkeypatch.setenv("AIPF_DATA", str(tmp_path))
    monkeypatch.setenv(paths.raw_env_name(name), str(tmp_path / "no-raw-root"))
    anchors = "none" if name in DRIFT_ONLY_SAMPLE else "declared"
    code = main(["train", "--system", name, "--run", "sample", "--seed", "0", "--epochs", "1",
                 "--source", "sample", "--resume-optimizer", "no", "--anchors", anchors,
                 "--device", "cpu", "--log-every-step"])
    assert code == 0
    run_dir = tmp_path / name / "ckpt" / "sample"
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    assert manifest["global_step"] >= 1
    if anchors == "declared":
        assert manifest["declared_weights_without_data"] == {}
        assert not (run_dir / "UNTRAINED_TERMS.txt").exists()
    else:
        assert "L_dyn" in manifest["terms_trained"] and "L_W" in manifest["terms_trained"]
    assert "L_dyn" in manifest["terms_trained"]
    assert [s["name"] for s in manifest["sources"]] == list(load(name).defaults["training"]["sample"])
    steps = json.loads((run_dir / "steps.json").read_text())
    for loss, terms in zip(steps["loss"], steps["terms"]):
        assert math.isfinite(loss)
        assert all(math.isfinite(v) for v in terms.values()), terms
        assert set(manifest["terms_trained"]) <= set(terms)
