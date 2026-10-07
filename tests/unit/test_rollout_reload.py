"""``aipf.rollout.spinodal.load_model``: the loader ``aipf rollout`` and the slab driver share.

A file ``fit`` wrote carries this package's ``config_schema`` tag, its own ``model_state_dict`` and
Lightning's ``state_dict`` (the same tensors under a ``model.`` prefix). The loader once took the
Lightning one first, so a fit checkpoint fell through to the Lightning-hparams bridge and raised. A tagged
file now loads from ``model_state_dict``; every other file is pinned here to the path it always took,
against a copy of the loader as it was.
"""
import pytest
import torch

import declared_roots
from aipf.functional.build import build
from aipf.rollout.spinodal import load_model
from aipf.system import load
from aipf.train.checkpoint_formats import load_lightning_hparams_into
from aipf.train.fit import fit
from test_train_fit import _demo_system_with_modes

SYSTEMS = ("hhe", "feb", "lj")


def _load_as_before(system, path):
    """The loader as public 1ba9974 wrote it: ``state_dict`` first, whatever the file's tag."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["state_dict"] if "state_dict" in saved else \
        saved["model_state_dict"]
    model = build(system)
    if set(state) <= set(model.state_dict()):
        merged = dict(model.state_dict())
        merged.update(state)
        model.load_state_dict(merged, strict=True)
    else:
        load_lightning_hparams_into(model, state)
    return model.eval()


def _outcome(loader, system, path):
    """``("ok", state_dict)``, or ``("raised", type, message)``."""
    try:
        return ("ok", loader(system, path).state_dict())
    except Exception as exc:  # the outcome itself is what is compared
        return ("raised", type(exc), str(exc))


def _assert_same(before, now):
    assert before[0] == now[0], (before[:1], now[:1])
    if before[0] == "raised":
        assert before[1:] == now[1:]
        return
    assert list(before[1]) == list(now[1])
    for key in before[1]:
        assert torch.equal(before[1][key], now[1][key]), key


def _fit_checkpoint(tmp_path):
    system, sources = _demo_system_with_modes(tmp_path)
    run = fit(system, run_name="reload", sources=sources, steps=2, seed=0,
              resume_optimizer=False, root=tmp_path / "data")
    return system, run / "final.ckpt"


def test_the_rollout_loader_reads_a_checkpoint_fit_wrote(tmp_path):
    system, path = _fit_checkpoint(tmp_path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert "state_dict" in saved and "model_state_dict" in saved

    with pytest.raises(KeyError):          # the loader as it was
        _load_as_before(system, path)
    state = load_model(system, path).state_dict()

    assert set(state) == set(saved["model_state_dict"])
    for key, value in saved["model_state_dict"].items():
        assert torch.equal(state[key], value), key


def test_a_fit_checkpoint_reloads_to_the_drift_it_was_saved_with(tmp_path):
    system, path = _fit_checkpoint(tmp_path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    reference = build(system)
    reference.load_state_dict(saved["model_state_dict"], strict=True)
    loaded = load_model(system, path)

    grid = tuple(system.functional.kwargs["grid"])
    gen = torch.Generator().manual_seed(0)
    rho = 0.5 + 0.05 * torch.randn(1, system.n_species, *grid, generator=gen)
    rho_hat = torch.fft.rfftn(rho, dim=(-3, -2, -1)) / (grid[0] * grid[1] * grid[2])
    boxes = torch.full((1, 3), 8.0)
    T = torch.tensor([1.0])
    with torch.no_grad():
        assert torch.equal(loaded(rho_hat, boxes, T), reference.eval()(rho_hat, boxes, T))


@pytest.mark.parametrize("name", SYSTEMS)
def test_each_published_checkpoint_takes_the_path_it_always_took(name):
    declared_roots.published_or_skip(name)
    system = load(name)
    path = system.resolve_checkpoint()
    _assert_same(_outcome(_load_as_before, system, path),
                 _outcome(load_model, system, path))


def test_a_tag_this_package_does_not_know_takes_the_path_it_always_took(tmp_path):
    system, path = _fit_checkpoint(tmp_path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved["config_schema"] = "apfm.train.TrainConfig.v1"     # the tag before the rename
    renamed = tmp_path / "old_tag.ckpt"
    torch.save(saved, renamed)

    before = _outcome(_load_as_before, system, renamed)
    assert before[0] == "raised"
    _assert_same(before, _outcome(load_model, system, renamed))


def test_an_untagged_file_with_only_model_state_dict_loads_as_before(tmp_path):
    system, path = _fit_checkpoint(tmp_path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    bare = tmp_path / "bare.ckpt"
    torch.save({"model_state_dict": saved["model_state_dict"]}, bare)

    before = _outcome(_load_as_before, system, bare)
    assert before[0] == "ok"
    _assert_same(before, _outcome(load_model, system, bare))
