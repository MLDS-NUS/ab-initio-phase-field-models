"""``--resume-optimizer yes`` on a checkpoint whose saved optimizer does not fit is a named refusal.

The published ``lj`` checkpoint was written by a trainer that held the local net's linear skip
(``g_net.w2.weight``) frozen and listed its parameters in another order: its one optimizer group
has 15 tensors, a fresh optimizer here has 16. Torch's own load fails with a bare size error, so
the refusal is made before anything is written and names what differs."""
import pytest

from aipf.cli.main import main
from aipf.system import load
from aipf.train.fit import (OptimizerLayoutMismatch, check_optimizer_resumable, fit,
                            pre_run_checks)


def _published(system):
    return check_optimizer_resumable(system, system.checkpoint,
                                     pre_run_checks(system, "probe", system.checkpoint))


def test_lj_published_optimizer_is_refused_by_name():
    with pytest.raises(OptimizerLayoutMismatch) as refused:
        _published(load("lj"))
    text = str(refused.value)
    assert "saved [15]" in text and "fresh optimizer [16]" in text
    assert "f_local.g_net.w2.weight" in text
    assert "--resume-optimizer no" in text


@pytest.mark.parametrize("variant", ["fh", "landau"])
def test_lj_baseline_optimizers_fit(variant):
    _published(load("lj").variant(variant))


def test_hhe_published_optimizer_fits():
    _published(load("hhe"))


@pytest.mark.parametrize("batch", [[], ["--pbs", "--dry-run"]])
def test_cli_exits_2_before_writing(tmp_path, capsys, monkeypatch, batch):
    monkeypatch.setenv("AIPF_DATA", str(tmp_path))
    (tmp_path / "lj" / "modes" / "cube_probe").mkdir(parents=True)
    code = main(["train", "--system", "lj", "--run", "probe", "--seed", "0", "--steps", "1",
                 "--source", "cube=.:cube_probe:16,16,16", "--init-from-published",
                 "--resume-optimizer", "yes", "--anchors", "none", "--device", "cpu", *batch])
    assert code == 2
    err = capsys.readouterr().err
    assert "--resume-optimizer yes:" in err and "--resume-optimizer no" in err
    assert "f_local.g_net.w2.weight" in err
    assert not (tmp_path / "lj" / "ckpt" / "probe").exists()


def test_fit_refuses_before_the_run_directory_exists(tmp_path):
    system = load("lj")
    with pytest.raises(OptimizerLayoutMismatch):
        fit(system, run_name="probe", sources=[object()], seed=0, steps=1,
            init_from=system.checkpoint, resume_optimizer=True, root=tmp_path, device="cpu")
    assert not any(tmp_path.rglob("probe"))
