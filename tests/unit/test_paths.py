"""Path resolution: environment, then ``aipf.toml``, then a default, for every location.

The ``repo`` fixture is a checkout of its own in ``tmp_path``: the repository root is found from the
package's own location, never from the working directory, so the fixture points ``repo_root`` there.
"""
import os
import textwrap
from pathlib import Path

import pytest

from aipf import paths
from aipf.paths import Paths

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "data" / "hhe" / "ckpt" / "published").mkdir(parents=True)
    (tmp_path / "data" / "hhe" / "ckpt" / "published" / "final.ckpt").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    for k in list(os.environ):
        if k.startswith("AIPF_"):
            monkeypatch.delenv(k)
    paths.config.cache_clear()
    yield tmp_path
    paths.config.cache_clear()


def test_published_resolves_without_config(repo):
    assert paths.published_checkpoint("hhe") == repo / "data/hhe/ckpt/published/final.ckpt"
    assert paths.data_root() == repo / "data"


def test_env_beats_toml_beats_default(repo, monkeypatch):
    (repo / "aipf.toml").write_text(textwrap.dedent("""
        [paths]
        data = "farm"
        [paths.raw]
        hhe = "/raw/hhe"
    """))
    paths.config.cache_clear()
    assert paths.data_root() == repo / "farm"
    assert paths.raw_root("hhe") == Path("/raw/hhe")
    monkeypatch.setenv("AIPF_DATA", str(repo / "env-farm"))
    monkeypatch.setenv("AIPF_RAW_HHE", "/env/hhe")
    assert paths.data_root() == repo / "env-farm"
    assert paths.raw_root("hhe") == Path("/env/hhe")
    monkeypatch.delenv("AIPF_RAW_HHE"); monkeypatch.setenv("AIPF_RAW", "/env/raw")
    assert paths.raw_root("hhe") == Path("/env/raw/hhe")


def test_missing_raw_root_names_the_key_and_the_file(repo):
    with pytest.raises(paths.MissingLocation) as e:
        paths.raw_root("feb")
    msg = str(e.value)
    assert "AIPF_RAW_FEB" in msg and "[paths.raw]" in msg and "feb" in msg and "aipf.toml" in msg
    assert e.value.env_name == "AIPF_RAW_FEB" and e.value.toml_path == repo / "aipf.toml"


def test_raw_parent_in_the_toml_is_joined_with_the_system(repo):
    (repo / "aipf.toml").write_text('[paths]\nraw_parent = "/big/disk"\n')
    paths.config.cache_clear()
    assert paths.raw_root("lj") == Path("/big/disk/lj")


def test_a_relative_toml_path_is_relative_to_the_repository_root(repo, tmp_path_factory, monkeypatch):
    (repo / "aipf.toml").write_text('[paths]\nexperiments = "exp"\n[paths.raw]\nlj = "raw/lj"\n')
    paths.config.cache_clear()
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    assert paths.raw_root("lj") == repo / "raw" / "lj"
    assert paths.experiments_root() == repo / "exp"


def test_empty_values_count_as_unset(repo, monkeypatch):
    (repo / "aipf.toml").write_text('[paths]\ndata = ""\n[paths.raw]\nhhe = "/raw/hhe"\n')
    paths.config.cache_clear()
    monkeypatch.setenv("AIPF_DATA", "")
    monkeypatch.setenv("AIPF_RAW_HHE", "")
    monkeypatch.setenv("AIPF_RAW", "")
    assert paths.data_root() == repo / "data"
    assert paths.raw_root("hhe") == Path("/raw/hhe")


def test_farm_tiers_and_variants(repo, monkeypatch):
    assert paths.farm("lj", "modes") == repo / "data" / "lj" / "modes"
    assert paths.published_checkpoint("lj", "fh") == repo / "data/lj/ckpt/published/fh/final.ckpt"
    monkeypatch.setenv("AIPF_DATA", str(repo / "elsewhere"))
    assert paths.farm("lj", "md") == repo / "elsewhere" / "lj" / "md"
    # the published checkpoints are tracked files of the checkout: AIPF_DATA does not move them
    assert paths.published_checkpoint("hhe") == repo / "data/hhe/ckpt/published/final.ckpt"
    assert Paths("hhe").published() == repo / "data/hhe/ckpt/published"


def test_experiments_root_env_then_toml_then_default(repo, monkeypatch):
    assert paths.experiments_root() == repo / "experiments"
    (repo / "aipf.toml").write_text('[paths]\nexperiments = "/toml/exp"\n')
    paths.config.cache_clear()
    assert paths.experiments_root() == Path("/toml/exp")
    monkeypatch.setenv("AIPF_EXPERIMENTS", "/env/exp")
    assert paths.experiments_root() == Path("/env/exp")


def test_paths_raw_declared_then_default_then_refused(repo, monkeypatch):
    p = Paths(system="demo", raw_default="/default/demo")
    assert p.raw() == Path("/default/demo")
    monkeypatch.setenv("AIPF_RAW", "/env/raw")
    assert p.raw() == Path("/env/raw/demo")
    assert p.sub("md", "run1") == Path("/env/raw/demo/md/run1")
    monkeypatch.setenv("AIPF_RAW_DEMO", "/env/demo")
    assert p.raw() == Path("/env/demo")
    assert p.data_root() == repo / "data" / "demo"
    monkeypatch.delenv("AIPF_RAW_DEMO"); monkeypatch.delenv("AIPF_RAW")
    with pytest.raises(paths.MissingLocation, match="AIPF_RAW_DEMO"):
        Paths(system="demo").raw()


def test_resolution_is_read_at_call_time(repo, monkeypatch):
    p = Paths(system="demo")
    monkeypatch.setenv("AIPF_RAW_DEMO", "/first")
    assert p.raw() == Path("/first")
    monkeypatch.setenv("AIPF_RAW_DEMO", "/second")
    assert p.raw() == Path("/second")


def test_paths_is_frozen_and_has_no_scratch():
    p = Paths(system="demo")
    with pytest.raises(Exception):
        p.system = "other"
    assert p.raw_default is None
    assert not hasattr(p, "scratch")


def test_the_repository_root_is_not_the_working_directory(monkeypatch, tmp_path):
    """A checkout of its own as the working directory does not move the farm."""
    monkeypatch.delenv("AIPF_DATA", raising=False)
    monkeypatch.setattr(paths, "config", lambda: {})  # not the checkout's aipf.toml either
    (tmp_path / "pyproject.toml").write_text("")
    monkeypatch.chdir(tmp_path)
    assert paths.repo_root() == REPO
    assert Paths(system="demo").data_root() == REPO / "data" / "demo"


def test_outside_a_checkout_the_data_root_is_refused_and_aipf_data_is_the_way_out(monkeypatch, tmp_path):
    monkeypatch.delenv("AIPF_DATA", raising=False)
    monkeypatch.delenv("AIPF_RAW_DEMO", raising=False)
    monkeypatch.setattr(paths, "_REPO_MARKER", "no-such-marker-file.toml")
    paths.config.cache_clear()
    with pytest.raises(RuntimeError) as excinfo:
        paths.data_root()
    assert "no-such-marker-file.toml" in str(excinfo.value) and "AIPF_DATA" in str(excinfo.value)
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "d"))
    assert Paths(system="demo").data_root() == tmp_path / "d" / "demo"
    monkeypatch.setenv("AIPF_RAW_DEMO", str(tmp_path / "s"))
    assert Paths(system="demo").raw() == tmp_path / "s"
    paths.config.cache_clear()

