"""``--resume-optimizer yes`` on a checkpoint whose saved optimizer does not fit is a named refusal.

The published ``lj`` checkpoint was written by a trainer that held the local net's linear skip
(``g_net.w2.weight``) frozen and listed its parameters in another order: its one optimizer group
has 15 tensors, a fresh optimizer here has 16. Torch's own load fails with a bare size error, so
the refusal is made before anything is written and names what differs.

An optimizer state lists positions, not names, so two tensors of one shape in another order fit by
shape. The order is checked by the parameter names a checkpoint written here records; one that records
none (every published one) is refused when a group holds two tensors of one shape."""
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


@pytest.mark.parametrize("name,variant", [("hhe", None), ("feb", None), ("lj", "fh"),
                                          ("lj", "landau")])
def test_a_published_optimizer_that_records_no_names_is_refused_naming_the_shared_shapes(
        name, variant):
    """These saved states fit tensor by tensor in count and shape, but they record no parameter
    names, and in every group two tensors or more share a shape, so their order is unproved: refused,
    naming the tensors, with the way to start from the weights."""
    system = load(name) if variant is None else load(name).variant(variant)
    with pytest.raises(OptimizerLayoutMismatch) as refused:
        _published(system)
    text = str(refused.value)
    assert "records no parameter names" in text and "share a shape with another" in text
    assert "--resume-optimizer no" in text


def _parameters(*shapes):
    import torch
    return [torch.nn.Parameter(torch.zeros(shape)) for shape in shapes]


def _state(params):
    """A saved optimizer state for ``params`` in that order: one group, a moment per position."""
    import torch
    return {"param_groups": [{"params": list(range(len(params)))}],
            "state": {i: {"exp_avg": torch.zeros(p.shape)} for i, p in enumerate(params)}}


def test_two_same_shaped_parameters_swapped_are_refused_by_name():
    """Shapes pass (both are (3,)); the recorded names show the two moments sit in each other's place."""
    from aipf.train.fit import _optimizer_layout_refusal
    a, b, c = _parameters((3,), (3,), (2, 3))
    names = {id(a): "net.a", id(b): "net.b", id(c): "net.c"}
    groups = [{"params": [a, b, c]}]
    assert _optimizer_layout_refusal(groups, _state([a, b, c]), names, "here",
                                     [["net.a", "net.b", "net.c"]]) is None
    text = _optimizer_layout_refusal(groups, _state([b, a, c]), names, "here",
                                     [["net.b", "net.a", "net.c"]])
    assert text is not None
    assert "2 position(s) where the saved optimizer holds another parameter" in text
    assert "first at position 0 (saved net.b, this model net.a (3,))" in text


def test_without_names_shared_shapes_are_refused_and_distinct_shapes_pass():
    from aipf.train.fit import _optimizer_layout_refusal
    a, b, c = _parameters((3,), (3,), (2, 3))
    names = {id(a): "net.a", id(b): "net.b", id(c): "net.c"}
    text = _optimizer_layout_refusal([{"params": [a, b, c]}], _state([a, b, c]), names, "here")
    assert text is not None and "2 of this model's 3 tensors share a shape" in text
    assert "(net.a, net.b)" in text
    assert _optimizer_layout_refusal([{"params": [a, c]}], _state([a, c]), names, "here") is None


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
