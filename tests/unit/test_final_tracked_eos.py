"""The equation-of-state tables the Fe-B and H/He diagnosis stages read are tracked under
``experiments/<system>/eos/`` and read from there before the raw root, so ``phase_diagram``,
``stability_map`` and ``dome`` run in a fresh clone (they used to need the raw root for 23 KB of csv)."""
import hashlib
from pathlib import Path

import pytest

import aipf.paths as paths_mod
from aipf.diagnose.run import manifolds
from aipf.paths import Paths, tracked_eos
from aipf.system import load

#: The tracked bytes: the raw root's files at release (Fe-B 0 GPa is the post-apex grid, the same
#: bytes as eos_0GPa/eos_n_x_T_v2.csv that the training block names).
MD5 = {
    ("feb", "eos_0GPa/eos_n_x_T.csv"): "d3acf7781056e169d7cb9e720e04503a",
    ("feb", "eos_5GPa/eos_n_x_T.csv"): "1b3f00c224e2d7df662ca01996081aff",
    ("feb", "eos_10GPa/eos_n_x_T.csv"): "ff7210576c68766ee02a1db7c74338ec",
    ("hhe", "eos_200GPa/eos_n_x_T.csv"): "3eb6f15bcabac6cfbeb0d1ceb9c5007e",
    ("hhe", "eos_400GPa/eos_n_x_T.csv"): "31a70eedac7370635f008dc67596693d",
    ("hhe", "eos_600GPa/eos_n_x_T.csv"): "43285b768486a5089143ac3ccec14c02",
    ("hhe", "eos_800GPa/eos_n_x_T.csv"): "d800f3d58b82577a13d0abcab5a2d3f8",
}


def test_every_declared_diagnosis_table_is_tracked_with_its_bytes():
    for name in ("feb", "hhe"):
        declared = load(name).defaults["diagnose"]["eos_csvs"]
        for relative in declared.values():
            path = tracked_eos(name) / relative
            assert path.is_file(), path
            assert hashlib.md5(path.read_bytes()).hexdigest() == MD5[(name, relative)]
    assert len(MD5) == 7


@pytest.mark.parametrize("name", ["feb", "hhe"])
def test_the_diagnosis_reads_them_without_a_raw_root(name, monkeypatch):
    def refuse(self):
        raise AssertionError("the raw root was asked for")

    monkeypatch.setattr(Paths, "raw", refuse)
    system = load(name)
    loaded = manifolds(system, system.defaults["diagnose"])
    assert sorted(loaded) == sorted(float(p) for p in system.defaults["diagnose"]["eos_csvs"])


def test_a_table_not_tracked_is_read_from_the_raw_root(tmp_path, monkeypatch):
    system = load("feb")
    declared = system.defaults["diagnose"]
    raw = tmp_path / "raw"
    for relative in declared["eos_csvs"].values():
        (raw / relative).parent.mkdir(parents=True, exist_ok=True)
        (raw / relative).write_bytes((tracked_eos("feb") / relative).read_bytes())
    monkeypatch.setattr(paths_mod, "tracked_eos", lambda name: tmp_path / "nothing")
    monkeypatch.setattr(Paths, "raw", lambda self: raw)
    assert sorted(manifolds(system, declared)) == [0.0, 5.0, 10.0]
