"""One command as one batch job: :class:`aipf.md.scheduler.PbsSpec`, :func:`batch_job`, and the job
file :class:`PbsBackend` writes and submits (fake farm commands; nothing reaches a server)."""
from __future__ import annotations

import json
import shlex
import sys

import pytest

from aipf.md.scheduler import (WALLTIME_ENV, PbsBackend, PbsSpec, SchedulerError, activation,
                               batch_job, job_name)
from aipf.site import MissingSiteFact, Site

_FAKE = '''
import json, pathlib, sys
plan = json.loads(pathlib.Path(__file__ + ".plan").read_text())
with open(plan["log"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
sys.stdout.write(plan["out"])
'''

_QUEUE_OK = "Queue: gpuq\n    enabled = True\n    started = True\n"


def _fake(tmp_path, name, out):
    """An executable that records its arguments, one JSON list per call, and prints ``out``."""
    path, log = tmp_path / name, tmp_path / f"{name}.log"
    path.write_text(f"#!{sys.executable}\n{_FAKE}")
    path.chmod(0o755)
    (tmp_path / f"{name}.plan").write_text(json.dumps({"log": str(log), "out": out}))
    log.write_text("")
    return str(path), log


def _calls(log):
    return [json.loads(line) for line in log.read_text().splitlines() if line]


SPEC = PbsSpec(queue="gpuq", project="p123", gpus=1, walltime_h=0.25)


def test_the_spec_asks_for_one_chunk_with_its_gpus_and_optional_cpus_and_memory():
    assert SPEC.select == "1:ngpus=1"
    assert PbsSpec("q", "p", 2, 1.0, ncpus=16, mem="110gb").select == "1:ngpus=2:ncpus=16:mem=110gb"


@pytest.mark.parametrize("kwargs", [dict(queue=" "), dict(project=""), dict(gpus=-1), dict(gpus=1.5),
                                    dict(walltime_h=0.0), dict(walltime_h=float("inf")), dict(ncpus=0)])
def test_a_spec_that_names_nothing_usable_is_refused(kwargs):
    fields = dict(queue="q", project="p", gpus=1, walltime_h=1.0) | kwargs
    with pytest.raises(SchedulerError):
        PbsSpec(**fields)


def test_the_spec_is_the_sites_and_the_line_wins():
    site = Site(pbs_queue="ai", pbs_project="p1", pbs_gpus=1, pbs_walltime_h=24.0, pbs_mem="8gb")
    assert PbsSpec.from_site(site) == PbsSpec("ai", "p1", 1, 24.0, mem="8gb")
    assert PbsSpec.from_site(site, walltime_h=0.5).walltime_h == 0.5
    with pytest.raises(MissingSiteFact) as refused:
        PbsSpec.from_site(Site(pbs_queue="ai"))
    assert set(refused.value.missing) == {"pbs_project", "pbs_walltime_h", "pbs_gpus"}
    assert "AIPF_PBS_PROJECT" in str(refused.value)
    assert PbsSpec.from_site(Site(pbs_queue="ai", pbs_project="p"), walltime_h=1, gpus=1).gpus == 1


def test_the_activation_needs_no_conda_on_the_node(tmp_path):
    lines = activation(tmp_path / "env").splitlines()
    assert f"export CONDA_PREFIX={tmp_path / 'env'}" in lines
    assert 'export PATH="$CONDA_PREFIX/bin:$PATH"' in lines
    assert any("etc/conda/activate.d/\"*.sh" in line for line in lines)
    assert lines[-1] == "set -u"                      # the job file's own strictness is restored
    assert f"export CONDA_PREFIX={sys.prefix}" in activation().splitlines()


def test_a_job_name_is_one_the_server_takes():
    assert job_name("md-feb cuda/run") == "md-feb_cuda_run"
    assert len(job_name("x" * 40)) == 15 and job_name("...") == "aipf"


def _job(tmp_path, command=("aipf", "md", "run", "--out", "/o", "--set", "x=1 2")):
    return batch_job(SPEC, name="md-x", command=command, workdir=tmp_path / "out",
                     env={"AIPF_LAMMPS": "/site/lmp"}, prefix=tmp_path / "env")


def test_scheduler_pbs_backend_builds_qsub_command(tmp_path):
    qsub, qsub_log = _fake(tmp_path, "qsub", "42.srv\n")
    qstat, qstat_log = _fake(tmp_path, "qstat", _QUEUE_OK)
    job = _job(tmp_path)
    handle = SPEC.backend(qsub=qsub, qstat=qstat).submit(job)
    assert handle.job_id == "42.srv" and job.script_path == tmp_path / "out" / "job.pbs"
    [call] = _calls(qsub_log)
    assert call == ["-v", f"AIPF_LAMMPS=/site/lmp,{WALLTIME_ENV}=900.0", str(job.script_path)]
    assert _calls(qstat_log) == [["-Qf", "gpuq"]]     # the queue is asked about before submitting
    lines = job.script_path.read_text().splitlines()
    assert lines[0].startswith("#!")
    assert lines[1:8] == ["#PBS -N md-x", "#PBS -q gpuq", "#PBS -P p123",
                          "#PBS -l select=1:ngpus=1", "#PBS -l walltime=00:15:00",
                          f"#PBS -o {tmp_path / 'out' / 'job.log'}", "#PBS -j oe"]
    assert lines[-1] == shlex.join(["aipf", "md", "run", "--out", "/o", "--set", "x=1 2"])
    assert f"cd {tmp_path / 'out'}" in lines
    assert lines.index(f"cd {tmp_path / 'out'}") < lines.index(f"export CONDA_PREFIX={tmp_path / 'env'}")


def test_writing_a_job_file_asks_the_server_nothing(tmp_path):
    qsub, qsub_log = _fake(tmp_path, "qsub", "42.srv\n")
    qstat, qstat_log = _fake(tmp_path, "qstat", _QUEUE_OK)
    job = _job(tmp_path)
    back = SPEC.backend(qsub=qsub, qstat=qstat)
    path = back.write(job)
    assert path.read_text() == back.job_text(job)
    assert _calls(qsub_log) == [] and _calls(qstat_log) == []


def test_the_job_file_name_is_a_plain_name(tmp_path):
    with pytest.raises(SchedulerError, match="plain file name"):
        batch_job(SPEC, name="j", command=["true"], workdir=tmp_path, script_name="../job.pbs")


def test_the_written_and_the_submitted_job_files_are_the_same(tmp_path):
    """The dry run's file is the file a submission sends."""
    qsub, _ = _fake(tmp_path, "qsub", "1.srv\n")
    qstat, _ = _fake(tmp_path, "qstat", _QUEUE_OK)
    job = _job(tmp_path)
    back = PbsBackend(queue="gpuq", project="p123", resources={"select": SPEC.select},
                      qsub=qsub, qstat=qstat)
    written = back.write(job).read_text()
    back.submit(job)
    assert job.script_path.read_text() == written
