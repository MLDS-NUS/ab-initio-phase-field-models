"""Site facts: environment, then ``[site]`` in ``aipf.toml``, one refusal listing every missing fact."""
import pytest

from aipf import paths, site


@pytest.fixture
def own_checkout(monkeypatch, tmp_path):
    """``tmp_path`` as the repository root: it is found from the package, not the working directory."""
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    paths.config.cache_clear()
    yield
    paths.config.cache_clear()


def test_site_lists_every_missing_fact(own_checkout, monkeypatch, tmp_path):
    for k in ("AIPF_LAMMPS", "AIPF_LATEXMK", "AIPF_MACE_POTENTIAL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path); (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    with pytest.raises(site.MissingSiteFact) as e:
        site.Site.load().require("lammps", "mace_potential")
    assert "AIPF_LAMMPS" in str(e.value) and "AIPF_MACE_POTENTIAL" in str(e.value) and "[site]" in str(e.value)


def test_site_reads_env_then_toml(own_checkout, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path); (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "aipf.toml").write_text('[site]\nlammps = "/toml/lmp"\n')
    monkeypatch.delenv("AIPF_LAMMPS", raising=False)
    from aipf import paths; paths.config.cache_clear()
    assert str(site.Site.load().lammps) == "/toml/lmp"
    monkeypatch.setenv("AIPF_LAMMPS", "/env/lmp")
    assert str(site.Site.load().lammps) == "/env/lmp"


def test_require_returns_the_site_and_rejects_unknown_names(own_checkout, monkeypatch):
    for env in site.FACTS.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("AIPF_CUDA_LIB", "/opt/cuda/lib64")
    loaded = site.Site.load()
    assert loaded.require("cuda_lib") is loaded and loaded.lammps_python is None
    with pytest.raises(ValueError, match="mace_potential"):
        loaded.require("mace")


def test_every_fact_has_a_variable_and_a_field():
    import dataclasses
    assert [f.name for f in dataclasses.fields(site.Site)] == list(site.FACTS)
    assert all(env == "AIPF_" + name.upper() for name, env in site.FACTS.items())


def test_md_commands_without_a_lammps_refuse_naming_the_fact(monkeypatch, capsys, tmp_path):
    """``--lammps`` defaults to the site's binary; with none declared, ``run`` exits 2 naming how to
    declare it and ``doctor`` reports it as a broken check naming the same."""
    from aipf import paths
    from aipf.cli.main import main
    monkeypatch.delenv("AIPF_LAMMPS", raising=False)
    monkeypatch.setattr(paths, "config", lambda: {})  # no aipf.toml either
    assert main(["md", "run", "--system", "lj", "--template", "cube-overdamped",
                 "--out", str(tmp_path / "run"), "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "AIPF_LAMMPS" in err and "[site]" in err and not (tmp_path / "run").exists()
    import aipf.md.doctor as doctor
    monkeypatch.setattr(doctor, "_site_child", lambda *a, **k: (None, "not asked", "", False))
    assert main(["md", "doctor", "--device", "cpu"]) == 1  # a named broken check, not a refusal
    out = capsys.readouterr().out
    assert "mliap_style" in out and "AIPF_LAMMPS" in out and "Traceback" not in out


def test_md_run_takes_the_site_binary(monkeypatch, tmp_path):
    """With ``AIPF_LAMMPS`` declared and no ``--lammps``, the run is handed the site's binary."""
    import aipf.md.run as md_run
    from aipf.cli.main import main
    seen = {}
    monkeypatch.setenv("AIPF_LAMMPS", "/opt/site/lmp")
    monkeypatch.setattr(md_run, "run", lambda system, **kw: seen.update(kw) or (_ for _ in ()).throw(ValueError("stop")))
    assert main(["md", "run", "--system", "lj", "--template", "cube-overdamped",
                 "--out", str(tmp_path / "run"), "--dry-run"]) == 2
    assert seen["lammps"] == "/opt/site/lmp"


def test_the_batch_facts_are_values_read_in_their_types(own_checkout, monkeypatch, tmp_path):
    """``pbs_*`` are site facts too, read as text, number or count rather than as paths."""
    (tmp_path / "aipf.toml").write_text('[site]\npbs_queue = "ai"\npbs_project = 12345678\n'
                                        'pbs_walltime_h = 24\npbs_gpus = 1\n')
    for env in ("AIPF_PBS_QUEUE", "AIPF_PBS_PROJECT", "AIPF_PBS_WALLTIME_H", "AIPF_PBS_GPUS",
                "AIPF_PBS_NCPUS", "AIPF_PBS_MEM"):
        monkeypatch.delenv(env, raising=False)
    loaded = site.Site.load()
    assert (loaded.pbs_queue, loaded.pbs_project, loaded.pbs_walltime_h, loaded.pbs_gpus,
            loaded.pbs_ncpus) == ("ai", "12345678", 24.0, 1, None)
    monkeypatch.setenv("AIPF_PBS_WALLTIME_H", "0.5")
    assert site.Site.load().pbs_walltime_h == 0.5
    monkeypatch.setenv("AIPF_PBS_GPUS", "one")
    with pytest.raises(site.MissingSiteFact, match="pbs_gpus = 'one' is not a int"):
        site.Site.load()
    with pytest.raises(site.MissingSiteFact, match='pbs_ncpus = 16 under'):
        site.Site().require("pbs_ncpus")
