"""``aipf md run`` on a deck of a system that declares a machine-learned potential (the ML-IAP route).

The potential is the site's (``Site.mace_potential``), checked against the md5 the system declares;
the deck runs in this interpreter on a GPU only, or is submitted with ``--pbs``. Every refusal is
exit 2 and leaves nothing behind. The loading and the engine are stood in for here; the real run is
``tests/unit/test_md_run_mliap_real.py``."""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

import aipf.md.run as md_run
from aipf.cli.main import main
from aipf.md.doctor import deck_styles, styles_of
from aipf.md.engine import Result
from aipf.md.potentials import Potential, PotentialError, md5_of, resolve
from aipf.system import load

_SHORT = ["--set", "n_atoms=2", "--set", "equil_ps=0.01", "--set", "prod_ps=0.01",
          "--set", "dump_every_ps=0.001"]


def _feb_run(out, *extra):
    return ["md", "run", "--system", "feb", "--template", "cube-npt", *_SHORT,
            "--lammps", "lmp", "--out", str(out), *extra]


@pytest.fixture
def stand_ins(monkeypatch, tmp_path):
    """A potential that is not loaded, a binary that lists every style, a GPU that is visible,
    a run that is not made, and the site's batch facts; records what reached them."""
    seen = {}
    potential = Potential(path=tmp_path / "model-mliap.pt", sha256="0" * 64, engine="mace-mliap",
                          species=("B", "Fe"), dtype="float32", cutoff=3.0, fallback_kernels=(),
                          model=object())

    def resolve_potential(system, declared):
        seen["declared"] = dict(declared)
        return potential, [n for n, _ in sorted(system.atom_types.items(), key=lambda i: i[1])]

    def run_deck(launch, text, **kw):
        seen.update(launch=launch, text=text, **kw)
        return Result(status="ok", route=launch.route, seconds=0.1, n_atoms=2)

    monkeypatch.setattr(md_run, "_resolve_potential", resolve_potential)
    monkeypatch.setattr(md_run, "deck_styles", lambda binary, text: deck_styles(
        binary, text, runner=lambda argv, timeout: type("R", (), dict(
            returncode=0, stdout=" ".join(styles_of(text)), stderr=""))()))
    monkeypatch.setattr(md_run, "gpu_visible", lambda: (True, "1 device(s): a test GPU"))
    monkeypatch.setattr(md_run, "run_deck", run_deck)
    monkeypatch.setattr("aipf.cli.md_cmd._leave", lambda code: seen.setdefault("left", code))
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    from aipf import paths
    monkeypatch.setattr(paths, "config", lambda: {})  # the site is the variables below, no aipf.toml
    for name, value in {"AIPF_PBS_QUEUE": "gpuq", "AIPF_PBS_PROJECT": "p123", "AIPF_PBS_GPUS": "1",
                        "AIPF_PBS_WALLTIME_H": "2"}.items():
        monkeypatch.setenv(name, value)
    for name in ("AIPF_PBS_NCPUS", "AIPF_PBS_MEM"):
        monkeypatch.delenv(name, raising=False)
    seen["potential"] = potential
    return seen


def test_the_systems_declare_their_potentials_by_md5():
    assert md_run.declared_potential(load("feb")) == {"md5": "f14252d355d95ed5932de858c7840be0"}
    assert md_run.declared_potential(load("hhe")) == {"md5": "2ed3c8297221761a915e212408e803b4"}
    assert md_run.declared_potential(load("lj")) is None


def test_mliap_run_refuses_cpu_naming_the_route(tmp_path, stand_ins, capsys):
    assert main(_feb_run(tmp_path / "out", "--device", "cpu")) == 2
    err = capsys.readouterr().err
    assert "ML-IAP route" in err and "--device cuda" in err
    assert not (tmp_path / "out").exists() and "declared" not in stand_ins


def test_mliap_run_refuses_cuda_when_absent(tmp_path, stand_ins, monkeypatch, capsys):
    monkeypatch.setattr(md_run, "gpu_visible", lambda: (False, "nvidia-smi lists no device"))
    assert main(_feb_run(tmp_path / "out", "--device", "cuda")) == 2
    err = capsys.readouterr().err
    assert "--device cuda and no GPU is visible here (nvidia-smi lists no device)" in err
    assert not (tmp_path / "out").exists() and "declared" not in stand_ins


def test_mliap_run_auto_without_a_gpu_is_refused_too(tmp_path, stand_ins, monkeypatch):
    monkeypatch.setattr(md_run, "gpu_visible", lambda: (False, "none"))
    assert main(_feb_run(tmp_path / "out")) == 2
    assert not (tmp_path / "out").exists()


def test_mliap_run_drives_the_engine_in_process_with_the_declared_potential(tmp_path, stand_ins, capsys):
    out = tmp_path / "out"
    assert main(_feb_run(out, "--device", "cuda")) == 0
    assert stand_ins["declared"] == {"md5": "f14252d355d95ed5932de858c7840be0"}
    assert stand_ins["workdir"] == out
    launch = stand_ins["launch"]
    assert launch.route == "in_process" and launch.uses_model and ("-k", "on", "g", "1") == launch.args[4:8]
    text = (out / "in.lammps").read_text()
    assert text == stand_ins["text"]
    assert "pair_style      mliap/kk unified EXISTS\npair_coeff      * * B Fe" in text
    assert "variable        n_atoms equal 2" in text and "set             type 2 type/ratio 1 0.5 1" in text
    assert "fix             prod all npt temp 1800.0 1800.0 0.1 iso 0.0 0.0 1.0" in text
    record = json.loads((out / "run.json").read_text())
    assert record["route"] == "in_process" and record["device"] == "cuda" and record["campaign"] == "production"
    assert record["result"]["status"] == "ok" and record["potential"]["species"] == ["B", "Fe"]
    assert stand_ins["left"] == 0                      # left by exit_cleanly, after the output


def test_mliap_dry_run_writes_deck_and_job(tmp_path, stand_ins, monkeypatch, capsys):
    monkeypatch.setattr("aipf.md.scheduler.PbsBackend.submit",
                        lambda self, job: pytest.fail("a dry run submitted"))
    monkeypatch.setattr(md_run, "gpu_visible", lambda: pytest.fail("a submission asked for this node's GPU"))
    out = tmp_path / "out"
    assert main(_feb_run(out, "--device", "cuda", "--pbs", "--dry-run", "--walltime-h", "0.25")) == 0
    assert (out / "in.lammps").is_file() and (out / "job.pbs").is_file()
    assert "run_deck" not in stand_ins and "launch" not in stand_ins and not (out / "job.json").exists()
    text = (out / "job.pbs").read_text()
    for line in ("#PBS -q gpuq", "#PBS -P p123", "#PBS -l select=1:ngpus=1", "#PBS -l walltime=00:15:00",
                 f"export CONDA_PREFIX={sys.prefix}"):
        assert line in text.splitlines(), line
    command = shlex.split(text.strip().splitlines()[-1])
    assert command == ["aipf", "md", "run", "--system", "feb", "--template", "cube-npt",
                       "--set", "n_atoms=2", "--set", "equil_ps=0.01", "--set", "prod_ps=0.01",
                       "--set", "dump_every_ps=0.001", "--lammps", "lmp", "--out", str(out),
                       "--device", "cuda"]
    assert str(out / "job.pbs") in capsys.readouterr().out


def test_mliap_pbs_refuses_cpu_before_writing(tmp_path, stand_ins):
    assert main(_feb_run(tmp_path / "out", "--device", "cpu", "--pbs", "--dry-run")) == 2
    assert not (tmp_path / "out").exists()


def test_mliap_pbs_submits_and_prints_the_job_id(tmp_path, stand_ins, monkeypatch, capsys):
    from aipf.md.scheduler import Handle
    submitted = {}

    def submit(self, job):
        submitted.update(job=job, select=self.resources["select"])
        self.write(job)
        return Handle(backend="pbs", job_id="7.srv", name=job.name, workdir=job.workdir,
                      stdout=job.stdout, stderr=job.stderr, submitted_at=0.0)
    monkeypatch.setattr("aipf.md.scheduler.PbsBackend.submit", submit)
    out = tmp_path / "out"
    assert main(_feb_run(out, "--device", "cuda", "--pbs")) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "7.srv"
    assert json.loads((out / "job.json").read_text())["job_id"] == "7.srv"
    job = submitted["job"]
    # the job runs the binary and the potential this submission resolved
    assert job.env == {"AIPF_LAMMPS": "lmp", "AIPF_MACE_POTENTIAL_FEB": str(stand_ins["potential"].path)}
    assert json.loads((out / "run.json").read_text())["job"] == {"file": "job.pbs", "id": "7.srv"}


def test_pbs_without_the_site_queue_refuses_naming_it(tmp_path, stand_ins, monkeypatch, capsys):
    monkeypatch.delenv("AIPF_PBS_QUEUE")
    assert main(_feb_run(tmp_path / "out", "--device", "cuda", "--pbs")) == 2
    assert "AIPF_PBS_QUEUE" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_walltime_without_pbs_is_refused(tmp_path, stand_ins):
    assert main(_feb_run(tmp_path / "out", "--walltime-h", "1")) == 2


def test_declared_values_that_write_the_model_style_are_refused(tmp_path, stand_ins, capsys):
    assert main(_feb_run(tmp_path / "out", "--set", "PAIR=pair_style mliap unified x.pt 0")) == 2
    assert "drop PAIR" in capsys.readouterr().err


def test_a_potential_of_another_md5_is_refused_before_it_is_loaded(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"not the declared file")
    with pytest.raises(PotentialError, match="different file"):
        resolve(model, engine="mace-mliap", md5="f14252d355d95ed5932de858c7840be0",
                loader=lambda path: pytest.fail("loaded before the md5 was checked"))
    assert md5_of(model) == __import__("hashlib").md5(b"not the declared file").hexdigest()


def test_the_potential_must_know_every_typed_species(tmp_path, monkeypatch):
    class Model:
        element_types, num_species, rcutfac, ndescriptors, nparams = ["Fe"], 1, 3.0, 1, 1
    monkeypatch.setenv("AIPF_MACE_POTENTIAL_FEB", str(tmp_path / "m.pt"))
    (tmp_path / "m.pt").write_bytes(b"x")
    monkeypatch.setattr("aipf.md.potentials._torch_load", lambda path: Model())
    monkeypatch.setattr("aipf.md.potentials.check_e3nn", lambda required=None: None)
    with pytest.raises(PotentialError, match=r"\['B'\] would have no interactions"):
        md_run._resolve_potential(load("feb"), {"md5": md5_of(tmp_path / "m.pt")})


def test_a_pair_potential_deck_is_unchanged_by_the_device(tmp_path, monkeypatch):
    """The LJ deck names its own pair style and runs the binary on the host."""
    monkeypatch.setattr(md_run, "deck_styles", lambda binary, text: deck_styles(
        binary, text, runner=lambda argv, timeout: type("R", (), dict(
            returncode=0, stdout=" ".join(styles_of(text)), stderr=""))()))
    monkeypatch.setattr(md_run, "gpu_visible", lambda: pytest.fail("a pair deck asked for a GPU"))
    assert main(["md", "run", "--system", "lj", "--template", "cube-overdamped", "--lammps", "lmp",
                 "--out", str(tmp_path / "r"), "--dry-run", "--device", "cpu"]) == 0
    assert json.loads((tmp_path / "r" / "run.json").read_text())["route"] == "input_file"


def test_a_potential_declared_without_an_md5_is_refused():
    with pytest.raises(PotentialError, match="without an md5"):
        md_run._resolve_potential(load("feb"), {})


def test_the_md5_refusal_names_where_the_systems_file_is_declared(tmp_path, monkeypatch):
    (tmp_path / "m.pt").write_bytes(b"another model")
    monkeypatch.setenv("AIPF_MACE_POTENTIAL_FEB", str(tmp_path / "m.pt"))
    with pytest.raises(PotentialError) as refused:
        md_run._resolve_potential(load("feb"), {"md5": "f14252d355d95ed5932de858c7840be0"})
    said = str(refused.value)
    assert "AIPF_MACE_POTENTIAL_FEB" in said and "[site.mace_potential]" in said


def test_each_system_has_its_own_potential_and_the_bare_one_serves_the_rest(tmp_path, monkeypatch):
    from aipf import paths
    from aipf.site import MissingSiteFact, Site
    (tmp_path / "aipf.toml").write_text('[site.mace_potential]\nfeb = "/toml/feb.pt"\n')
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    paths.config.cache_clear()
    try:
        for name in ("AIPF_MACE_POTENTIAL", "AIPF_MACE_POTENTIAL_FEB", "AIPF_MACE_POTENTIAL_HHE"):
            monkeypatch.delenv(name, raising=False)
        site = Site.load()
        assert site.mace_potential is None                  # the table is not the bare fact
        assert str(site.for_system("mace_potential", "feb")) == "/toml/feb.pt"
        with pytest.raises(MissingSiteFact, match=r"AIPF_MACE_POTENTIAL_HHE.*\[site.mace_potential\]"):
            site.for_system("mace_potential", "hhe")
        monkeypatch.setenv("AIPF_MACE_POTENTIAL", "/env/any.pt")
        assert str(Site.load().for_system("mace_potential", "hhe")) == "/env/any.pt"
        monkeypatch.setenv("AIPF_MACE_POTENTIAL_FEB", "/env/feb.pt")
        assert str(Site.load().for_system("mace_potential", "feb")) == "/env/feb.pt"
    finally:
        paths.config.cache_clear()
