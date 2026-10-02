"""``fit(device=..., deterministic=...)`` and ``aipf train --device/--deterministic/--pbs``.

A device is chosen, never assumed: ``auto`` trains on cuda when torch sees one, ``cuda`` without one
is refused before the run directory exists, and determinism is asked for or left alone."""
from __future__ import annotations

import json
import os

import pytest
import torch

import test_train_fit as toy
from aipf.train.fit import DeviceUnavailable, fit


def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_train_refuses_cuda_when_absent(tmp_path, monkeypatch):
    _no_cuda(monkeypatch)
    system, sources = toy._demo_system_with_modes(tmp_path)
    with pytest.raises(DeviceUnavailable, match=r"device cuda requested, torch.cuda.is_available\(\) is False"):
        fit(system, run_name="r", sources=sources, steps=1, seed=0, resume_optimizer=False,
            root=tmp_path / "data", device="cuda")
    assert not (tmp_path / "data" / "demo" / "ckpt" / "r").exists()


def test_train_device_auto_picks_cpu_without_cuda(tmp_path, monkeypatch):
    _no_cuda(monkeypatch)
    system, sources = toy._demo_system_with_modes(tmp_path)
    run = fit(system, run_name="r", sources=sources, steps=1, seed=0, resume_optimizer=False,
              root=tmp_path / "data")
    manifest = json.loads((run / "MANIFEST.json").read_text())
    assert manifest["device"] == "cpu" and manifest["deterministic"] is False


def test_an_unknown_device_is_refused(tmp_path):
    system, sources = toy._demo_system_with_modes(tmp_path)
    with pytest.raises(ValueError, match="device is one of"):
        fit(system, run_name="r", sources=sources, steps=1, seed=0, resume_optimizer=False,
            root=tmp_path / "data", device="gpu")


def _switches_inside(monkeypatch):
    """What the torch switches read while the Trainer trains, recorded by a stand-in ``fit``."""
    import lightning as L
    seen = {}

    def record(self, *args, **kwargs):
        seen.update(flag=torch.are_deterministic_algorithms_enabled(),
                    benchmark=torch.backends.cudnn.benchmark,
                    cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    monkeypatch.setattr(L.Trainer, "fit", record)
    monkeypatch.setattr(L.Trainer, "save_checkpoint", lambda self, path: open(path, "wb").close())
    return seen


def test_fit_restores_flags_with_deterministic_off(tmp_path, monkeypatch):
    """``deterministic=False`` touches no switch: not during the run, not after it."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    before = toy._torch_switches()
    seen = _switches_inside(monkeypatch)
    system, sources = toy._demo_system_with_modes(tmp_path)
    fit(system, run_name="r", sources=sources, steps=1, seed=0, resume_optimizer=False,
        root=tmp_path / "data", device="cpu")
    assert seen == {"flag": False, "benchmark": True, "cublas": None}
    assert toy._torch_switches() == before


def test_deterministic_sets_the_switches_for_the_run_only(tmp_path, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = toy._torch_switches()
    seen = _switches_inside(monkeypatch)
    system, sources = toy._demo_system_with_modes(tmp_path)
    fit(system, run_name="r", sources=sources, steps=1, seed=0, resume_optimizer=False,
        root=tmp_path / "data", device="cpu", deterministic=True)
    assert seen == {"flag": True, "benchmark": False, "cublas": ":4096:8"}
    assert toy._torch_switches() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device visible to torch")
def test_cuda_trains_with_the_anchor_tables_on_the_device(tmp_path):
    """Two steps on the device with the toy's measured tables attached, a pressure manifold among them:
    the tables follow the model to the device (``on_fit_start``), and the mobility, structure, bulk and
    pressure terms are all evaluated there."""
    from aipf.train.anchors import AnchorTables
    from test_train_anchors import _COLUMNS, _EOS_COLUMNS, _write_tables

    system, sources = toy._demo_system_with_modes(tmp_path)
    _write_tables(tmp_path / "tier")
    tables = AnchorTables.load(
        system, key="pressure", m_table={1.0: "tier/m_table.npz"},
        s_table={1.0: "tier/s_table.npz"}, eos_csvs={1.0: "tier/manifold.csv"},
        pressure_unit=0.5, columns=dict(_COLUMNS), eos_columns=dict(_EOS_COLUMNS),
        row_weight=None, pressure_floor=None, w0={"route": "evaluator"})
    assert tables.pressure_rho.shape[0] > 0 and tables.pressure_rho.device.type == "cpu"
    run = fit(system, run_name="r", sources=sources, steps=2, seed=0, resume_optimizer=False,
              anchors=tables, root=tmp_path / "data", device="cuda", log_every_step=True,
              config_overrides={"lambda_M": 0.5, "lambda_S": 0.01, "lambda_bulk": 0.01,
                                "lambda_P": 5.0, "stat_metric": "rel_frob"})
    manifest = json.loads((run / "MANIFEST.json").read_text())
    assert manifest["device"] == "cuda"
    assert manifest["terms_trained"] == ["L_dyn", "L_M", "L_S", "L_bulk", "L_P"]
    steps = json.loads((run / "steps.json").read_text())
    assert len(steps["loss"]) == 2
    for terms in steps["terms"]:
        assert {"L_M", "L_S", "L_bulk", "L_P"} <= set(terms), terms


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def _cli_system(tmp_path, monkeypatch):
    """``aipf train --system demo`` reaches the toy, its data root in ``tmp_path``."""
    import dataclasses
    system, _ = toy._demo_system_with_modes(tmp_path)
    training = dict(system.defaults.get("training") or {})
    training["source_root"] = {"tier": "raw", "path": "fields"}
    system = dataclasses.replace(system, defaults={**system.defaults, "training": training})
    monkeypatch.setattr("aipf.system.load", lambda name: system)
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "data"))
    from aipf import paths
    paths.config.cache_clear()
    return system


_ARGS = ["train", "--system", "demo", "--run", "r1", "--seed", "0", "--steps", "1",
         "--source", "demo_src=modes_demo:run_*:4,4,4",
         "--resume-optimizer", "no", "--anchors", "none"]


def test_cli_refuses_cuda_when_absent_before_the_run_directory(tmp_path, monkeypatch, capsys):
    from aipf.cli.main import main
    _cli_system(tmp_path, monkeypatch)
    _no_cuda(monkeypatch)
    assert main(_ARGS + ["--device", "cuda"]) == 2
    assert "torch.cuda.is_available() is False" in capsys.readouterr().err
    assert not (tmp_path / "data" / "demo" / "ckpt" / "r1").exists()


def test_cli_pbs_dry_run_writes_the_job_and_submits_nothing(tmp_path, monkeypatch):
    from aipf import paths
    from aipf.cli.main import main
    _cli_system(tmp_path, monkeypatch)
    monkeypatch.setattr(paths, "config", lambda: {})  # the site is the variables below, no aipf.toml
    for name, value in {"AIPF_PBS_QUEUE": "gpuq", "AIPF_PBS_PROJECT": "p123",
                        "AIPF_PBS_GPUS": "1", "AIPF_PBS_WALLTIME_H": "12"}.items():
        monkeypatch.setenv(name, value)
    for name in ("AIPF_PBS_NCPUS", "AIPF_PBS_MEM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("aipf.md.scheduler.PbsBackend.submit",
                        lambda self, job: pytest.fail("a dry run submitted"))
    assert main(_ARGS + ["--device", "cuda", "--deterministic", "--pbs", "--dry-run"]) == 0
    job = tmp_path / "data" / "demo" / "ckpt" / "r1" / "job.pbs"
    text = job.read_text()
    assert "#PBS -q gpuq" in text and "#PBS -P p123" in text
    assert "#PBS -l select=1:ngpus=1\n" in text and "#PBS -l walltime=12:00:00" in text
    command = text.strip().splitlines()[-1]
    assert command.startswith("aipf train --system demo --run r1 --seed 0 --steps 1 ")
    assert command.endswith("--resume-optimizer no --anchors none --device cuda --deterministic")
    assert "--pbs" not in command and "--dry-run" not in command
    assert not (job.parent / "final.ckpt").exists()


def test_cli_pbs_without_a_queue_refuses_naming_it(tmp_path, monkeypatch, capsys):
    from aipf import paths
    from aipf.cli.main import main
    _cli_system(tmp_path, monkeypatch)
    monkeypatch.setattr(paths, "config", lambda: {})
    for name in ("AIPF_PBS_QUEUE", "AIPF_PBS_PROJECT", "AIPF_PBS_GPUS", "AIPF_PBS_WALLTIME_H"):
        monkeypatch.delenv(name, raising=False)
    assert main(_ARGS + ["--pbs", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "AIPF_PBS_QUEUE" in err and "pbs_project" in err


def _pbs_site(monkeypatch):
    from aipf import paths
    monkeypatch.setattr(paths, "config", lambda: {})
    for name, value in {"AIPF_PBS_QUEUE": "gpuq", "AIPF_PBS_PROJECT": "p123",
                        "AIPF_PBS_GPUS": "1", "AIPF_PBS_WALLTIME_H": "12"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("aipf.md.scheduler.PbsBackend.submit",
                        lambda self, job: pytest.fail("a refused run was submitted"))


def test_cli_pbs_refuses_the_reserved_run_name_before_writing(tmp_path, monkeypatch, capsys):
    """``--run published --pbs`` would put a job file into the published checkpoints' directory."""
    from aipf.cli.main import main
    _cli_system(tmp_path, monkeypatch)
    _pbs_site(monkeypatch)
    args = [a if a != "r1" else "published" for a in _ARGS]
    assert main(args + ["--pbs", "--dry-run"]) == 2
    assert "is the directory holding the published checkpoints" in capsys.readouterr().err
    assert not (tmp_path / "data" / "demo" / "ckpt").exists()


def test_cli_pbs_refuses_starting_weights_nothing_holds(tmp_path, monkeypatch, capsys):
    """``--init-from-published`` is found and digest-checked before a job is written, as ``fit`` does."""
    import dataclasses

    from aipf.cli.main import main
    from aipf.system import Checkpoint
    system = _cli_system(tmp_path, monkeypatch)
    absent = dataclasses.replace(system, checkpoint=Checkpoint(path="runs/none.ckpt", md5="0" * 32))
    monkeypatch.setattr("aipf.system.load", lambda name: absent)
    _pbs_site(monkeypatch)
    assert main(_ARGS + ["--init-from-published", "--pbs", "--dry-run"]) == 2
    assert "--init-from-published" in capsys.readouterr().err
    assert not (tmp_path / "data" / "demo" / "ckpt").exists()
