"""``aipf.train.fit``: the driver, on a toy system and a toy mode tree.

Nothing here names a real system or uses one's numbers:
the system below is a two-channel ``"demo"`` in reduced units and its mode
archive is four-cubed noise written by the same
:class:`aipf.pipeline.modes.ModesRecord` that writes a real one, so the
shapes are right by construction rather than by a hand-copied spelling.
What these tests pin is the DRIVER -- the run directory it writes, the file
it signs, the refusals it makes, and the checkpointing it turns off. Its
comparison against a recorded training run of the published model is
migration evidence and not part of this suite.

Local helpers rather than fixtures: this directory has no ``conftest.py``
and each file carries its own ``_demo``-shaped builder.
"""
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from aipf.paths import Paths
from aipf.pipeline.modes import ModesRecord
from aipf.system import AnchorRules, Checkpoint, Functional, Mobility, System
from aipf.train.fit import fit
from aipf.train.datamodule import RunWeighting, SourceSpec


@pytest.fixture(autouse=True)
def _demo_scratch_is_its_own(monkeypatch):
    """A demo system here declares its own scratch root; a user's ``AIPF_RAW``, which
    ``Paths.scratch`` prefers to a declared default, must not replace it."""
    monkeypatch.delenv("AIPF_RAW", raising=False)

#: The toy's grid and its mode ball. ``scatter_modes`` places a label only
#: when ``|n_x| < G_x/2``, ``|n_y| < G_y/2`` and ``0 <= n_z <= G_z/2``, so on
#: a four-cubed grid the whole admissible set is this product -- which makes
#: the archive's mode count a fact about the grid rather than a number
#: somebody chose.
_GRID = (4, 4, 4)
_FRAMES = 24

#: The toy's Boltzmann constant: a round 1.0, because the toy works in
#: reduced units and a system's ``kB`` is that system's own convention.
_TOY_KB = 1.0


def _toy_functional() -> Functional:
    return Functional(
        form="nonlocal_kernel", local="mlp", kernel="radial_mlp",
        kwargs=dict(
            grid=_GRID, R_cut=2.0, rho_ref=(0.3, 0.3),
            h_u=4, h_g=4, h_w=4, h_m=4,
            fexc_T_ref=1.0, rho_eps=1e-5, activation="gelu",
            g_form="mlp", ideal_form="gas", f_exc_form="split", nyquist_mask=True,
            gauge_fix=False, enable_TlnT=False, enable_T2=False,
            tbasis_ortho=False, kernel_n_quad=16, kernel_n_k_table=17,
            kernel_k_table_max=8.0, h_g_hat=4, h_g_tilde=4, T_ref=1000.0,
            local_input_scale=False))


def _toy_defaults() -> dict:
    """Every knob the driver reads, at a toy value.

    Split in two exactly as a real system's is: the names that are already
    :class:`~aipf.train.config.TrainConfig` fields sit at the top level, and
    ``"training"`` carries the window, weighting, split and loader shapes,
    which are not.
    """
    return {
        "estimator": "weak",
        "sigma": 1.0,
        # The shell anchor's band bound. Declared here for the same reason
        # every other knob is: the anchor tier reads it off the system and
        # ships no value of its own.
        "k_fit_stat": 1.5,
        "k_max": 2.0,
        "alpha_loss": 0.0,
        "h_inv_eps": 1e-6,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "warmup_epochs": 1,
        "anneal_epochs": 1,
        "source_loss_weights": {"demo_src": 1.0},
        "training": {
            "half_width": 2,
            "n_states": 3,
            "stride": 1,
            "savgol_window": None,
            "savgol_poly": None,
            # Uniform, so the three band knobs are refused rather than
            # unused -- which is what makes this the branch a real system's
            # `inverse_band_power` does not cover.
            "run_weighting": "uniform",
            "run_weight_probe_every": 10,
            "val_split": "random",
            "val_labels": (),
            "val_fraction": 0.1,
            "split_seed": 0,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "order": "shuffled",
            "lambda_dyn": 1.0,
            "bulk_residual": "relative_inverse",
            "eta_min": 1e-6,
            "grad_clip": 1.0,
        },
    }


def _write_modes(directory, rng) -> None:
    """One toy run's ``modes.npz``, through the writer a real one uses."""
    labels = np.array([(nx, ny, nz)
                       for nx in (-1, 0, 1)
                       for ny in (-1, 0, 1)
                       for nz in (0, 1, 2)], dtype=np.int64)
    shape = (_FRAMES, labels.shape[0], 2)
    amplitudes = (rng.standard_normal(shape)
                  + 1j * rng.standard_normal(shape)).astype(np.complex64)
    boxes = np.full((_FRAMES, 3), 8.0, dtype=np.float64)
    ModesRecord(amplitudes=amplitudes, labels=labels, boxes=boxes,
                T=1000.0, dt_frame=0.02,
                provenance={"composition": {"x_B": 0.5},
                             "cache_dir": str(directory)}).save(directory)


def _demo_system_with_modes(tmp_path, *, n_runs: int = 1):
    """A ``"demo"`` system, and one source spec pointing at its mode tree.

    ``n_runs`` is the number of run directories written under that tree, and
    it is not cosmetic. ``split_runs`` returns every run as training when a
    source holds fewer than two, whatever split is declared, so a ONE-run
    tree can never produce a validation set and a test built on it cannot
    reach the branch of :func:`fit` where a validation loader exists. Two
    runs is the smallest tree that can be split.
    """
    tree = tmp_path / "fields" / "modes_demo"
    for i in range(n_runs):
        _write_modes(tree / f"run_{chr(ord('a') + i)}",
                     np.random.default_rng(i))
    system = System(
        name="demo", n_species=2, species=("A", "B"),
        masses={"A": 1.0, "B": 2.0}, atom_types={"A": 1, "B": 2},
        table_keys={"rho": ("rho_A", "rho_B"), "x": "x_B", "x_channel": 1},
        paths=Paths(system="demo", raw_default=str(tmp_path)),
        anchor_rules=AnchorRules({}), constants={"kB": _TOY_KB},
        defaults=_toy_defaults(), functional=_toy_functional(),
        mobility=Mobility(form="mlp_scaled", T_form="none",
                          kwargs=dict(mobility_prefactor="mole_fraction",
                                      mobility_input_ref=None)))
    sources = (SourceSpec(name="demo_src", root=str(tree), pattern="run_*",
                          grid=_GRID, loss_weight=1.0, exclude_tags=()),)
    return system, sources


# ---------------------------------------------------------------------------
# what the run directory holds
# ---------------------------------------------------------------------------

def test_fit_writes_final_ckpt_and_records_its_md5(tmp_path):
    sysm, sources = _demo_system_with_modes(tmp_path)
    run = fit(sysm, run_name="smoke", sources=sources, steps=2, seed=0,
              resume_optimizer=False,
              root=tmp_path / "data")

    assert run == tmp_path / "data" / "demo" / "ckpt" / "smoke"
    assert (run / "final.ckpt").is_file()
    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["final_md5"] == hashlib.md5(
        (run / "final.ckpt").read_bytes()).hexdigest()
    assert man["steps"] == 2 and man["seed"] == 0
    assert man["system"] == "demo" and man["run"] == "smoke"
    assert man["sources"][0]["name"] == "demo_src"
    assert (run / "hparams.yaml").is_file()


def test_fit_signs_the_file_it_wrote_and_not_a_neighbour(tmp_path):
    """The digest is of ``final.ckpt``'s bytes, so a swap is detectable."""
    sysm, sources = _demo_system_with_modes(tmp_path)
    run = fit(sysm, run_name="smoke", sources=sources, steps=2, seed=0,
              resume_optimizer=False,
              root=tmp_path / "data")
    man = json.loads((run / "MANIFEST.json").read_text())
    (run / "final.ckpt").write_bytes(b"not the trained weights")
    assert man["final_md5"] != hashlib.md5(
        (run / "final.ckpt").read_bytes()).hexdigest()


def _torch_switches():
    return (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def test_fit_puts_back_the_process_wide_switches_its_trainer_sets(tmp_path,
                                                                  monkeypatch):
    """``deterministic=True`` sets the flag, ``cudnn.benchmark`` and the cuBLAS variable for the
    whole process; ``fit`` puts all of them back. ``benchmark`` starts at the value the Trainer overwrites, so a reset to
    defaults would not pass."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    found = _torch_switches()
    sysm, sources = _demo_system_with_modes(tmp_path)
    fit(sysm, run_name="smoke", sources=sources, steps=2, seed=0,
        resume_optimizer=False, root=tmp_path / "data", deterministic=True,
        device="cpu")
    assert _torch_switches() == found


def test_fit_puts_them_back_when_training_raises(tmp_path, monkeypatch):
    import lightning as L

    def _raise(*args, **kwargs):
        assert torch.are_deterministic_algorithms_enabled()  # the Trainer did set it
        raise RuntimeError("training failed")

    monkeypatch.setattr(L.Trainer, "fit", _raise)
    found = _torch_switches()
    sysm, sources = _demo_system_with_modes(tmp_path)
    with pytest.raises(RuntimeError, match="training failed"):
        fit(sysm, run_name="smoke", sources=sources, steps=2, seed=0,
            resume_optimizer=False, root=tmp_path / "data", deterministic=True,
            device="cpu")
    assert _torch_switches() == found


def test_fit_records_the_declared_weights_it_could_not_train(tmp_path):
    """A weight declared for a term with no data is recorded, not dropped.

    The toy declares none of the anchor weights, so the record is empty
    here; the assertion is that the two keys EXIST, because a manifest that
    carried the list only when it was non-empty would read as "nothing was
    skipped" on a run that never looked.
    """
    sysm, sources = _demo_system_with_modes(tmp_path)
    run = fit(sysm, run_name="smoke", sources=sources, steps=2, seed=0,
              resume_optimizer=False,
              root=tmp_path / "data")
    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["terms_trained"] == ["L_dyn"]
    assert man["declared_weights_without_data"] == {}


# ---------------------------------------------------------------------------
# checkpointing, and the file this driver refuses to depend on
# ---------------------------------------------------------------------------

def test_fit_turns_lightning_checkpointing_off_entirely(tmp_path,
                                                         monkeypatch):
    """No ``ModelCheckpoint`` is registered, so no linked file is written.

    Lightning's ``save_last`` links to the last MONITORED save, which in one
    measured run directory is epoch 0. This driver
    does not patch that link, it declines to have one -- and the assertion
    is on the ``Trainer`` that really ran, not on the text of the module,
    because a source-text search passes for a file that imports its
    callbacks from somewhere else.
    """
    import lightning

    seen = {}
    real_trainer = lightning.Trainer

    class _Spy(real_trainer):
        def __init__(self, **kwargs):
            seen["kwargs"] = dict(kwargs)
            super().__init__(**kwargs)
            seen["trainer"] = self

    # The driver looks `Trainer` up on the `lightning` module at call
    # time, so patching it there is patching the one the driver builds.
    monkeypatch.setattr(lightning, "Trainer", _Spy)
    sysm, sources = _demo_system_with_modes(tmp_path)
    run = fit(sysm, run_name="smoke", sources=sources, steps=2, seed=0,
              resume_optimizer=False,
              root=tmp_path / "data")

    assert seen["kwargs"]["enable_checkpointing"] is False
    assert seen["trainer"].checkpoint_callbacks == []
    written = sorted(p.name for p in run.iterdir())
    assert "last.ckpt" not in written, written
    assert "final.ckpt" in written


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_fit_refuses_both_steps_and_epochs_or_neither(tmp_path):
    sysm, sources = _demo_system_with_modes(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        fit(sysm, run_name="x", sources=sources, seed=0,
            resume_optimizer=False, root=tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        fit(sysm, run_name="x", sources=sources, steps=1, epochs=1, seed=0,
            resume_optimizer=False,
            root=tmp_path)


def test_fit_refuses_a_system_that_declares_no_training_block(tmp_path):
    sysm, sources = _demo_system_with_modes(tmp_path)
    stripped = {k: v for k, v in sysm.defaults.items() if k != "training"}
    bare = sysm.__class__(**{**sysm.__dict__, "defaults": stripped})
    with pytest.raises(KeyError, match="training"):
        fit(bare, run_name="x", sources=sources, steps=1, seed=0,
            resume_optimizer=False,
            root=tmp_path / "data")


def test_fit_refuses_an_unrecognised_config_override(tmp_path):
    sysm, sources = _demo_system_with_modes(tmp_path)
    with pytest.raises(TypeError):
        fit(sysm, run_name="x", sources=sources, steps=1, seed=0,
            resume_optimizer=False,
            root=tmp_path / "data",
            config_overrides={"lambda_not_a_field": 1.0})


def test_fit_refuses_an_empty_source_list(tmp_path):
    sysm, _ = _demo_system_with_modes(tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        fit(sysm, run_name="x", sources=(), steps=1, seed=0,
            resume_optimizer=False,
            root=tmp_path / "data")


# ---------------------------------------------------------------------------
# the seed
# ---------------------------------------------------------------------------

def test_two_runs_at_one_seed_write_the_same_checkpoint(tmp_path):
    """The declared seed fixes the model, the draws and the batch order."""
    sysm, sources = _demo_system_with_modes(tmp_path)
    digests = []
    for name in ("a", "b"):
        run = fit(sysm, run_name=name, sources=sources, steps=2, seed=7,
                  resume_optimizer=False,
                  root=tmp_path / "data")
        digests.append(json.loads(
            (run / "MANIFEST.json").read_text())["final_md5"])
    assert digests[0] == digests[1]


def test_a_different_seed_is_a_different_run(tmp_path):
    sysm, sources = _demo_system_with_modes(tmp_path)
    first = fit(sysm, run_name="a", sources=sources, steps=2, seed=7,
                resume_optimizer=False,
                root=tmp_path / "data")
    second = fit(sysm, run_name="b", sources=sources, steps=2, seed=8,
                 resume_optimizer=False,
                 root=tmp_path / "data")
    assert (json.loads((first / "MANIFEST.json").read_text())["final_md5"]
            != json.loads((second / "MANIFEST.json").read_text())["final_md5"])


# ---------------------------------------------------------------------------
# init_from
# ---------------------------------------------------------------------------

def test_fit_starts_from_a_checkpoint_it_wrote_and_verifies_its_digest(
        tmp_path):
    """``init_from`` reads the driver's own round trip, digest first.

    The OTHER branch of the loader -- a Lightning checkpoint that carries its
    model's settings as hyper-parameters, through ``load_lightning_hparams_into`` --
    is exercised against the published bytes in ``test_published.py``.
    """
    sysm, sources = _demo_system_with_modes(tmp_path)
    first = fit(sysm, run_name="cold", sources=sources, steps=1, seed=0,
                resume_optimizer=False,
                root=tmp_path / "data")
    written = first / "final.ckpt"
    declared = Checkpoint(
        path=str(written.relative_to(tmp_path)),
        md5=hashlib.md5(written.read_bytes()).hexdigest())

    run = fit(sysm, run_name="warm", sources=sources, steps=1, seed=0,
              resume_optimizer=False,
              root=tmp_path / "data", init_from=declared)
    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["init_from"]["md5"] == declared.md5
    # Warm and cold start from different weights, so they cannot agree.
    assert man["final_md5"] != json.loads(
        (first / "MANIFEST.json").read_text())["final_md5"]

    written.write_bytes(b"something else")
    with pytest.raises(ValueError, match="md5"):
        fit(sysm, run_name="warm2", sources=sources, steps=1, seed=0,
            resume_optimizer=False,
            root=tmp_path / "data", init_from=declared)


def test_fit_refuses_a_checkpoint_with_no_recognised_weights(tmp_path):
    sysm, sources = _demo_system_with_modes(tmp_path)
    empty = tmp_path / "runs" / "empty.ckpt"
    empty.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 0}, empty)
    declared = Checkpoint(path="runs/empty.ckpt",
                          md5=hashlib.md5(empty.read_bytes()).hexdigest())
    with pytest.raises(KeyError, match="state_dict"):
        fit(sysm, run_name="x", sources=sources, steps=1, seed=0,
            resume_optimizer=False,
            root=tmp_path / "data", init_from=declared)


# ---------------------------------------------------------------------------
# the per-step series
# ---------------------------------------------------------------------------

def test_log_every_step_writes_one_loss_per_optimizer_step(tmp_path):
    sysm, sources = _demo_system_with_modes(tmp_path)
    run = fit(sysm, run_name="series", sources=sources, steps=3, seed=1,
              resume_optimizer=False,
              root=tmp_path / "data", log_every_step=True)
    series = json.loads((run / "steps.json").read_text())["loss"]
    assert len(series) == 3
    assert all(np.isfinite(series))


def test_no_series_is_written_unless_it_was_asked_for(tmp_path):
    sysm, sources = _demo_system_with_modes(tmp_path)
    run = fit(sysm, run_name="quiet", sources=sources, steps=1, seed=1,
              resume_optimizer=False,
              root=tmp_path / "data")
    assert not (run / "steps.json").exists()


# ---------------------------------------------------------------------------
# the two comparison overrides: no split, and a deterministic batch stream
# ---------------------------------------------------------------------------

def test_the_declared_split_and_order_are_what_a_run_takes_by_default(
        tmp_path):
    sysm, sources = _demo_system_with_modes(tmp_path)
    run = fit(sysm, run_name="declared", sources=sources, steps=1, seed=1,
              resume_optimizer=False, root=tmp_path / "data")
    manifest = json.loads((run / "MANIFEST.json").read_text())
    assert manifest["split_mode"] == "random"
    assert manifest["loader_order"] == "shuffled"


def test_a_run_can_take_the_recorded_stream_instead_of_the_declared_one(
        tmp_path):
    """The pair a step-by-step comparison needs: no split, and the batch stream an
    existing step-by-step record was taken under. Neither is a property of
    a system, so neither is declared -- and the run directory says which
    one it took rather than leaving a reader to assume the declaration."""
    sysm, sources = _demo_system_with_modes(tmp_path)
    run = fit(sysm, run_name="matched", sources=sources, steps=2, seed=1,
              resume_optimizer=False, split_mode="none",
              loader_order="index_stepping", root=tmp_path / "data",
              log_every_step=True)
    manifest = json.loads((run / "MANIFEST.json").read_text())
    assert manifest["split_mode"] == "none"
    assert manifest["loader_order"] == "index_stepping"
    series = json.loads((run / "steps.json").read_text())["loss"]
    assert len(series) == 2 and all(np.isfinite(series))


# ---------------------------------------------------------------------------
# the validation loop: the branch a one-run tree can never reach
# ---------------------------------------------------------------------------

def _trainer_spy(monkeypatch):
    """Record the keyword arguments `fit` builds its Trainer with.

    The Trainer instance is kept too, because `callback_metrics` survives
    `fit` and is the evidence that the validation loop RAN rather than that
    a loader was handed over: a metric logged only from `validation_step`
    cannot appear there unless that method was called.
    """
    import lightning as L

    seen: dict = {}
    original = L.Trainer.__init__

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        seen["trainer"] = self
        original(self, *args, **kwargs)

    monkeypatch.setattr(L.Trainer, "__init__", spy)
    return seen


def _validation_step_counter(monkeypatch):
    """How many times the driver's own `validation_step` was entered."""
    from aipf.train.fit import _MultiSourceDriver

    calls: list = []
    original = _MultiSourceDriver.validation_step

    def counted(self, batch, batch_idx, dataloader_idx=0):
        calls.append(batch_idx)
        return original(self, batch, batch_idx, dataloader_idx)

    monkeypatch.setattr(_MultiSourceDriver, "validation_step", counted)
    return calls


def test_a_source_with_a_validation_split_is_actually_validated(tmp_path,
                                                                 monkeypatch):
    """Two runs, the declared random split, one held out -- and the
    validation loop runs.

    Asserted on three things that cannot be produced without it: the
    Trainer was NOT given `limit_val_batches=0`, `validation_step` was
    entered, and `val/loss` -- logged nowhere else -- is in the Trainer's
    own `callback_metrics` when the run is over. One epoch rather than a
    few steps, because end-of-epoch is when the loop runs.
    """
    sysm, sources = _demo_system_with_modes(tmp_path, n_runs=2)
    seen = _trainer_spy(monkeypatch)
    calls = _validation_step_counter(monkeypatch)
    run = fit(sysm, run_name="validated", sources=sources, epochs=1, seed=0,
              resume_optimizer=False, split_mode="random",
              root=tmp_path / "data")

    assert "limit_val_batches" not in seen, (
        "a run whose source HAS a validation set must not have its "
        "validation loop turned off")
    assert calls, "validation_step was never entered"
    assert "val/loss" in seen["trainer"].callback_metrics
    assert json.loads((run / "MANIFEST.json").read_text())["split_mode"] \
        == "random"


def test_a_run_with_no_split_never_validates_and_says_so_to_the_trainer(
        tmp_path, monkeypatch):
    """The same two-run tree with `split_mode="none"`: no validation set,
    so the loop is turned off explicitly rather than left to be inferred
    from a `None` the Trainer also accepts for "ask the module"."""
    sysm, sources = _demo_system_with_modes(tmp_path, n_runs=2)
    seen = _trainer_spy(monkeypatch)
    calls = _validation_step_counter(monkeypatch)
    run = fit(sysm, run_name="unvalidated", sources=sources, epochs=1,
              seed=0, resume_optimizer=False, split_mode="none",
              root=tmp_path / "data")

    assert seen["limit_val_batches"] == 0
    assert calls == [], f"validation_step was entered {len(calls)} time(s)"
    assert "val/loss" not in seen["trainer"].callback_metrics
    assert json.loads((run / "MANIFEST.json").read_text())["split_mode"] \
        == "none"


def test_the_two_runs_differ_so_the_split_is_a_real_holdout(tmp_path):
    """The fixture's own precondition: two run directories with different
    contents, one of which the declared split holds out."""
    from aipf.train.datamodule import ModeDataModule
    from aipf.train.fit import (_archive_keys, _loader_from_system,
                                _split_from_system, _window_from_system)

    sysm, sources = _demo_system_with_modes(tmp_path, n_runs=2)
    dm = ModeDataModule(
        sources=sources, keys=_archive_keys(sysm),
        window=_window_from_system(sysm), split=_split_from_system(sysm),
        weighting=RunWeighting(mode="uniform", sigma=None, k_max=None,
                               eps=None, probe_every=None),
        loader=_loader_from_system(sysm))
    dm.setup()
    source = dm.sources[0]
    assert len(source.train_runs) == 1 and len(source.val_runs) == 1
    assert source.val_dataset is not None
    assert not np.array_equal(source.train_runs[0].amplitudes,
                              source.val_runs[0].amplitudes)


# ---------------------------------------------------------------------------
# the anchor tier, the optimizer resume, and the standing record of a gap
# ---------------------------------------------------------------------------

def test_a_declared_weight_with_no_data_leaves_a_file_and_a_warning(tmp_path):
    """Three places, and the FILE is the one a run directory carries.

    The gap used to be a bare ``logging.warning`` on an unconfigured
    logger: untagged stderr, gone the moment a job's output was rotated,
    and invisible to anyone reading the run afterwards. It is now a
    standing artifact.
    """
    sysm, sources = _demo_system_with_modes(tmp_path)
    run = fit(sysm, run_name="gap", sources=sources, steps=1, seed=0,
              resume_optimizer=False, root=tmp_path / "data",
              config_overrides={"lambda_M": 0.5, "lambda_conv": 1.0})

    text = (run / "UNTRAINED_TERMS.txt").read_text()
    assert "L_M" in text and "L_conv" in text

    import yaml
    hparams = yaml.safe_load((run / "hparams.yaml").read_text())
    assert hparams["warnings"]["declared_weights_without_data"] == [
        "L_M", "L_conv"]

    man = json.loads((run / "MANIFEST.json").read_text())
    assert set(man["declared_weights_without_data"]) == {"L_M", "L_conv"}


def test_a_run_with_nothing_untrained_writes_no_such_file(tmp_path):
    """Its presence has to mean something, so absence is asserted too."""
    sysm, sources = _demo_system_with_modes(tmp_path)
    run = fit(sysm, run_name="clean", sources=sources, steps=1, seed=0,
              resume_optimizer=False, root=tmp_path / "data")
    assert not (run / "UNTRAINED_TERMS.txt").exists()

    import yaml
    assert "warnings" not in yaml.safe_load(
        (run / "hparams.yaml").read_text())


def test_the_anchor_tier_is_recorded_and_its_terms_counted_as_trained(
        tmp_path):
    """With tables, the four anchors move from "no data" to "trained"."""
    from aipf.train.anchors import AnchorTables
    from test_train_anchors import _COLUMNS, _EOS_COLUMNS, _write_tables

    sysm, sources = _demo_system_with_modes(tmp_path)
    _write_tables(tmp_path / "tier")
    tables = AnchorTables.load(
        sysm, key="pressure", m_table={1.0: "tier/m_table.npz"},
        s_table={1.0: "tier/s_table.npz"},
        eos_csvs={1.0: "tier/manifold.csv"}, pressure_unit=0.5,
        columns=dict(_COLUMNS), eos_columns=dict(_EOS_COLUMNS),
        row_weight=None, pressure_floor=None, w0={"route": "evaluator"})
    run = fit(sysm, run_name="anchored", sources=sources, steps=1, seed=0,
              resume_optimizer=False, anchors=tables,
              root=tmp_path / "data",
              config_overrides={"lambda_M": 0.5, "lambda_S": 0.01,
                                "lambda_bulk": 0.01, "lambda_P": 5.0,
                                "stat_metric": "rel_frob"})
    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["terms_trained"] == ["L_dyn", "L_M", "L_S", "L_bulk", "L_P"]
    assert man["declared_weights_without_data"] == {}
    assert man["anchor_tables"]["rows"]["anchor_M"] == 3
    assert len(man["anchor_tables"]["provenance"]["m_table"][0][1]) == 64
    assert not (run / "UNTRAINED_TERMS.txt").exists()


#: The four anchor weights the toy trains its tier at, and the statistic
#: the shell term is measured by. Shared by the two tests below so that the
#: declared run and the ablation differ in ONE argument.
_ANCHOR_WEIGHTS = {"lambda_M": 0.5, "lambda_S": 0.01, "lambda_bulk": 0.01,
                   "lambda_P": 5.0, "stat_metric": "rel_frob"}


def _declaring_tables(tmp_path):
    """The toy, with its own measured tables written and DECLARED."""
    from test_train_anchors import _COLUMNS, _EOS_COLUMNS, _write_tables

    sysm, sources = _demo_system_with_modes(tmp_path)
    _write_tables(tmp_path / "tier")
    sysm.defaults["training"]["tables"] = dict(
        key="pressure", m_table={1.0: "tier/m_table.npz"},
        s_table={1.0: "tier/s_table.npz"},
        eos_csvs={1.0: "tier/manifold.csv"}, pressure_unit=0.5,
        columns=dict(_COLUMNS), eos_columns=dict(_EOS_COLUMNS),
        row_weight=None, pressure_floor=None, w0={"route": "evaluator"})
    return sysm, sources


def test_the_declared_tables_are_trained_when_the_caller_names_no_anchors(
        tmp_path):
    """A declaration nobody has to repeat at the call.

    This is the defect the whole-branch review found: ``anchors=None`` used
    to mean "drift alone" whatever the system declared, so every run taken
    through the command line trained one of the five terms its own
    declaration names. ``None`` now means what it means for ``split_mode``
    and ``loader_order`` -- the declaration answers.
    """
    sysm, sources = _declaring_tables(tmp_path)
    run = fit(sysm, run_name="declared_tier", sources=sources, steps=1,
              seed=0, resume_optimizer=False, root=tmp_path / "data",
              config_overrides=dict(_ANCHOR_WEIGHTS))

    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["terms_trained"] == ["L_dyn", "L_M", "L_S", "L_bulk", "L_P"]
    assert man["declared_weights_without_data"] == {}
    assert man["anchor_tables"]["rows"]["anchor_M"] == 3
    assert not (run / "UNTRAINED_TERMS.txt").exists()


def test_the_no_anchors_sentinel_is_an_ablation_the_run_directory_states(
        tmp_path):
    """``NO_ANCHORS`` drops the tier EVEN THOUGH the system declares one.

    And says so where the run is read: the four declared weights had no
    data, which is the standing report a gap between a declaration and a
    run gets. It is a sentinel and not ``None`` so that this run cannot be
    confused with one whose caller simply said nothing.
    """
    from aipf.train.anchors import NO_ANCHORS

    sysm, sources = _declaring_tables(tmp_path)
    run = fit(sysm, run_name="ablation", sources=sources, steps=1, seed=0,
              resume_optimizer=False, anchors=NO_ANCHORS,
              root=tmp_path / "data",
              config_overrides=dict(_ANCHOR_WEIGHTS))

    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["terms_trained"] == ["L_dyn"]
    assert set(man["declared_weights_without_data"]) == {
        "L_M", "L_S", "L_bulk", "L_P"}
    assert man["anchor_tables"] is None
    text = (run / "UNTRAINED_TERMS.txt").read_text()
    for name in ("L_M", "L_S", "L_bulk", "L_P"):
        assert name in text


def test_the_anchor_tier_changes_the_loss_it_is_added_to(tmp_path):
    """Not a bookkeeping change: the number the run minimises moves."""
    from aipf.train.anchors import AnchorTables
    from test_train_anchors import _COLUMNS, _EOS_COLUMNS, _write_tables

    sysm, sources = _demo_system_with_modes(tmp_path)
    _write_tables(tmp_path / "tier")
    tables = AnchorTables.load(
        sysm, key="pressure", m_table={1.0: "tier/m_table.npz"},
        s_table={1.0: "tier/s_table.npz"},
        eos_csvs={1.0: "tier/manifold.csv"}, pressure_unit=0.5,
        columns=dict(_COLUMNS), eos_columns=dict(_EOS_COLUMNS),
        row_weight=None, pressure_floor=None, w0={"route": "evaluator"})
    weights = {"lambda_M": 0.5, "lambda_S": 0.01, "lambda_bulk": 0.01,
               "lambda_P": 5.0, "stat_metric": "rel_frob"}
    drift = fit(sysm, run_name="drift", sources=sources, steps=1, seed=3,
                resume_optimizer=False, root=tmp_path / "data",
                log_every_step=True, config_overrides=weights)
    both = fit(sysm, run_name="both", sources=sources, steps=1, seed=3,
               resume_optimizer=False, anchors=tables,
               root=tmp_path / "data", log_every_step=True,
               config_overrides=weights)
    a = json.loads((drift / "steps.json").read_text())["loss"][0]
    b = json.loads((both / "steps.json").read_text())["loss"][0]
    assert b > a


def test_resume_optimizer_without_a_checkpoint_is_refused(tmp_path):
    sysm, sources = _demo_system_with_modes(tmp_path)
    with pytest.raises(ValueError, match="init_from"):
        fit(sysm, run_name="x", sources=sources, steps=1, seed=0,
            resume_optimizer=True, root=tmp_path / "data")


def test_resume_optimizer_continues_the_saved_moments_and_rate(tmp_path):
    """A resumed run is not a fresh run on the same weights.

    The fresh schedule starts in its linear warm-up, where the rate is
    ``lr * 1e-6`` and a float32 step moves nothing; the resumed one starts
    at the rate the saved run had reached. Both are run from the SAME
    weights, so the difference measured here is the optimizer's alone.
    """
    sysm, sources = _demo_system_with_modes(tmp_path)
    first = fit(sysm, run_name="seed_run", sources=sources, steps=3, seed=0,
                resume_optimizer=False, root=tmp_path / "data")
    saved = Checkpoint(path=str(
        (first / "final.ckpt").relative_to(tmp_path)),
        md5=hashlib.md5((first / "final.ckpt").read_bytes()).hexdigest())

    state = torch.load(first / "final.ckpt", map_location="cpu",
                       weights_only=False)
    assert state["optimizer_states"] and state["lr_schedulers"]

    cold = fit(sysm, run_name="cold_opt", sources=sources, steps=1, seed=0,
               resume_optimizer=False, init_from=saved,
               root=tmp_path / "data")
    warm = fit(sysm, run_name="warm_opt", sources=sources, steps=1, seed=0,
               resume_optimizer=True, init_from=saved,
               root=tmp_path / "data")
    assert json.loads((warm / "MANIFEST.json").read_text())["resume_optimizer"]
    cold_md5 = json.loads((cold / "MANIFEST.json").read_text())["final_md5"]
    warm_md5 = json.loads((warm / "MANIFEST.json").read_text())["final_md5"]
    assert cold_md5 != warm_md5


def test_the_optimizer_builds_the_two_declared_decay_groups(tmp_path):
    """Two groups, base and heads, at the two declared decays.

    One group of 39 is a different optimizer from 34 + 5: it decays the
    temperature heads at a rate nobody declared for them, and it cannot be
    handed a saved state built the other way, because the state is matched
    to parameters by position ACROSS the groups.
    """
    from aipf.train.config import TrainConfig
    from aipf.train.lit_module import LitModule
    from aipf.functional.build import build

    sysm, _sources = _demo_system_with_modes(tmp_path)
    # The heads exist only when the temperature basis is orthogonalised, so
    # the toy is built WITH them here -- a model that has none gets one
    # group, which the assertion at the end of this test also covers.
    model = build(sysm, enable_TlnT=True, enable_T2=True)
    lit = LitModule(model, model._cache, TrainConfig(
        weight_decay=0.25, wd_ghat=0.75, sigma=1.0, k_max=2.0, seed=0))
    groups = lit.parameter_groups()
    assert [g["weight_decay"] for g in groups] == [0.25, 0.75]
    heads = {id(p) for name in lit.DECAY_GROUP_HEADS
             for p in getattr(model.f_local, name).parameters()}
    assert {id(p) for p in groups[1]["params"]} == heads
    assert not ({id(p) for p in groups[0]["params"]} & heads)
    assert (len(groups[0]["params"]) + len(groups[1]["params"])
            == len(list(model.parameters())))


def test_an_anchor_term_at_zero_weight_is_neither_trained_nor_missing(
        tmp_path):
    """The third state, which a two-list classification has to carry.

    A term whose weight is exactly zero is not trained -- ``compute_losses``
    skips it outright rather than computing it and multiplying by nothing --
    and it is not a gap either, because nothing was declared for it. It
    belongs in neither list, and the driver's own completeness check must
    not read that as a term it forgot.
    """
    from aipf.train.anchors import AnchorTables
    from test_train_anchors import _COLUMNS, _EOS_COLUMNS, _write_tables

    sysm, sources = _demo_system_with_modes(tmp_path)
    _write_tables(tmp_path / "tier")
    tables = AnchorTables.load(
        sysm, key="pressure", m_table={1.0: "tier/m_table.npz"},
        s_table={1.0: "tier/s_table.npz"},
        eos_csvs={1.0: "tier/manifold.csv"}, pressure_unit=0.5,
        columns=dict(_COLUMNS), eos_columns=dict(_EOS_COLUMNS),
        row_weight=None, pressure_floor=None, w0={"route": "evaluator"})
    run = fit(sysm, run_name="partial", sources=sources, steps=1, seed=0,
              resume_optimizer=False, anchors=tables,
              root=tmp_path / "data",
              config_overrides={"lambda_M": 0.5, "lambda_S": 0.0,
                                "lambda_bulk": 0.0, "lambda_P": 0.0})
    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["terms_trained"] == ["L_dyn", "L_M"]
    assert man["declared_weights_without_data"] == {}


def test_a_model_with_the_heads_and_no_declared_decay_is_refused(tmp_path):
    """``wd_ghat`` is a declaration, and an undeclared one is not zero.

    Both measured systems that carry these heads declare 0.01, so a numeric
    default would be a production decay nobody chose applied silently to
    five tensors -- and folding them into the base group instead is not
    what a system WITHOUT the heads gets either. So: refuse, by name.
    """
    from aipf.functional.build import build
    from aipf.train.config import TrainConfig
    from aipf.train.lit_module import LitModule

    sysm, _sources = _demo_system_with_modes(tmp_path)
    model = build(sysm, enable_TlnT=True, enable_T2=True)
    config = TrainConfig(sigma=1.0, k_max=2.0, seed=0)
    assert config.wd_ghat is None, "undeclared has to MEAN undeclared"
    lit = LitModule(model, model._cache, config)
    with pytest.raises(ValueError, match="wd_ghat"):
        lit.parameter_groups()


def test_a_model_without_the_heads_needs_no_decay_for_them(tmp_path):
    """The other branch: no heads, so the value is inert and stays absent.

    One group, at the declared base decay, and nothing raises -- the same
    optimizer this path built before `wd_ghat` existed at all.
    """
    from aipf.functional.build import build
    from aipf.train.config import TrainConfig
    from aipf.train.lit_module import LitModule

    sysm, _sources = _demo_system_with_modes(tmp_path)
    model = build(sysm)          # the toy declares neither T head
    assert getattr(model.f_local, "g_hat_net", None) is None
    config = TrainConfig(sigma=1.0, k_max=2.0, weight_decay=0.125, seed=0)
    groups = LitModule(model, model._cache, config).parameter_groups()
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0.125
    assert len(groups[0]["params"]) == len(list(model.parameters()))


def test_a_k_modes_checkpoint_loads_through_the_kmodes_branch(tmp_path):
    """A synthetic k-modes checkpoint, written in its own key spelling, loads strictly."""
    import hashlib

    from aipf.functional.build import build
    from aipf.system import Checkpoint
    from aipf.train.fit import _is_kmodes, _load_weights_into
    from aipf.train.checkpoint_formats import KMODES_RENAMES
    from test_train_anchors import _one_field

    system = _one_field(tmp_path)
    source = build(system)
    kmodes = {}
    for key, value in source.state_dict().items():
        for old, new in KMODES_RENAMES:
            if key == new or (new.endswith(".") and key.startswith(new)):
                kmodes["model." + old + key[len(new):]] = value + 0.25
                break
    assert kmodes, "the toy model carries none of the k-modes tensors"
    saved = {"state_dict": kmodes}
    path = tmp_path / "kmodes.ckpt"
    torch.save(saved, path)
    declared = Checkpoint(path="kmodes.ckpt",
                          md5=hashlib.md5(path.read_bytes()).hexdigest())
    assert _is_kmodes(saved)
    target = build(system)
    _load_weights_into(target, declared, tmp_path)
    loaded = target.state_dict()
    for key, value in kmodes.items():
        bare = key[len("model."):]
        for old, new in KMODES_RENAMES:
            if bare == old or (old.endswith(".") and bare.startswith(old)):
                assert torch.equal(loaded[new + bare[len(old):]], value), key
                break


# ---------------------------------------------------------------------------
# declared sources: `--source declared` and per-source exclusions
# ---------------------------------------------------------------------------

def _with_declared_sources(system, sources):
    """The demo system with ``defaults["training"]`` carrying a source root and ``sources``."""
    import dataclasses
    defaults = dict(system.defaults)
    defaults["training"] = dict(defaults["training"],
                                source_root={"tier": "raw", "path": "fields"},
                                sources=sources)
    return dataclasses.replace(system, defaults=defaults)


def test_source_declared_is_refused_on_a_system_that_declares_no_sources(
        monkeypatch, tmp_path, capsys):
    import importlib

    import aipf.system as system_mod
    from aipf.cli.main import main

    system, _ = _demo_system_with_modes(tmp_path)
    assert "sources" not in system.defaults["training"]
    fit_mod = importlib.import_module("aipf.train.fit")
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    monkeypatch.setattr(fit_mod, "fit", lambda *a, **k: pytest.fail(
        "a run must not start without the sources it was asked for"))

    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "0",
                 "--steps", "1", "--source", "declared",
                 "--anchors", "none", "--resume-optimizer", "no"]) == 2
    err = capsys.readouterr().err
    assert "--source declared" in err, err
    assert "defaults['training']['sources']" in err, err
    assert not (tmp_path / "data").exists()


def test_source_declared_stands_alone(monkeypatch, tmp_path, capsys):
    import aipf.system as system_mod
    from aipf.cli.main import main

    system, _ = _demo_system_with_modes(tmp_path)
    system = _with_declared_sources(system, {"demo_src": {
        "root": "modes_demo", "pattern": "run_*", "grid": list(_GRID)}})
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "0",
                 "--steps", "1", "--source", "declared",
                 "--source", "demo_src=modes_demo:run_*:4,4,4",
                 "--anchors", "none", "--resume-optimizer", "no"]) == 2
    assert "stands alone" in capsys.readouterr().err


def test_declared_exclude_tags_are_applied(tmp_path):
    from aipf.train.dataset import read_mode_runs
    from aipf.train.fit import _archive_keys, _specs_from, declared_source_table

    system, _ = _demo_system_with_modes(tmp_path, n_runs=3)
    system = _with_declared_sources(system, {"demo_src": {
        "root": "modes_demo", "pattern": "run_*", "grid": list(_GRID),
        "exclude_tags": ["run_b"]}})
    table = declared_source_table(system)
    assert table == (("demo_src", "modes_demo", "run_*", _GRID, ("run_b",)),)
    (spec,) = _specs_from(system, table, ["demo_src"])
    assert spec.exclude_tags == ("run_b",)
    assert spec.root == str(tmp_path / "fields" / "modes_demo")
    runs = read_mode_runs(spec.root, spec.pattern, keys=_archive_keys(system),
                          exclude_tags=spec.exclude_tags)
    assert [r.tag for r in runs] == ["run_a", "run_c"]
    # an explicit four-field row still excludes nothing
    (plain,) = _specs_from(system, [table[0][:4]], ["demo_src"])
    assert plain.exclude_tags == ()


def test_the_batch_job_reruns_source_declared_as_spelled():
    from aipf.cli.main import build_parser
    from aipf.cli.train_cmd import _job_command

    args = build_parser().parse_args([
        "train", "--system", "demo", "--run", "r1", "--seed", "0", "--epochs", "2",
        "--source", "declared", "--anchors", "declared", "--resume-optimizer", "no"])
    command = _job_command(args)
    i = command.index("--source")
    assert command[i:i + 2] == ["--source", "declared"]
    assert command.count("--source") == 1



def test_a_declared_band_trains_a_stored_ball_that_overflows_the_grid(tmp_path):
    """Labels to |n| = 2 on the 4-cubed toy grid: refused without a band, trained with one.

    On the edge-8 box the band 1.45 keeps (1, 1, 1) (|k| = 1.36) and drops (2, 0, 0) (1.57)."""
    import dataclasses

    labels = np.array([(nx, ny, nz) for nx in range(-2, 3) for ny in range(-2, 3)
                       for nz in range(0, 3)], dtype=np.int64)
    tree = tmp_path / "fields" / "modes_wide"
    rng = np.random.default_rng(0)
    for name in ("run_a", "run_b"):
        shape = (_FRAMES, labels.shape[0], 2)
        amplitudes = (rng.standard_normal(shape)
                      + 1j * rng.standard_normal(shape)).astype(np.complex64)
        ModesRecord(amplitudes=amplitudes, labels=labels,
                    boxes=np.full((_FRAMES, 3), 8.0), T=1000.0, dt_frame=0.02,
                    provenance={"composition": {"x_B": 0.5},
                                "cache_dir": str(tree / name)}).save(tree / name)
    system, _ = _demo_system_with_modes(tmp_path)
    sources = (SourceSpec(name="demo_src", root=str(tree), pattern="run_*",
                          grid=_GRID, loss_weight=1.0, exclude_tags=()),)

    with pytest.raises(ValueError, match="beyond the grid"):
        fit(system, run_name="no_band", sources=sources, steps=1, seed=0,
            resume_optimizer=False, root=tmp_path / "data", device="cpu")

    defaults = dict(system.defaults)
    defaults["training"] = dict(defaults["training"], band_k_max=1.45)
    banded = dataclasses.replace(system, defaults=defaults)
    run = fit(banded, run_name="band", sources=sources, steps=1, seed=0,
              resume_optimizer=False, root=tmp_path / "data", device="cpu",
              log_every_step=True)
    steps = json.loads((run / "steps.json").read_text())
    assert len(steps["loss"]) == 1 and np.isfinite(steps["loss"][0])


def test_source_declared_whose_runs_are_all_excluded_is_refused_before_a_run(
        monkeypatch, tmp_path, capsys):
    """The CLI's check honours the exclusions, so a source with nothing left fails before any load."""
    import importlib

    import aipf.system as system_mod
    from aipf.cli.main import main

    system, _ = _demo_system_with_modes(tmp_path, n_runs=2)
    system = _with_declared_sources(system, {"demo_src": {
        "root": "modes_demo", "pattern": "run_*", "grid": list(_GRID),
        "exclude_tags": ["run_a", "run_b"]}})
    fit_mod = importlib.import_module("aipf.train.fit")
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    monkeypatch.setattr(fit_mod, "fit", lambda *a, **k: pytest.fail(
        "a run must not start on a source whose every run is excluded"))
    assert main(["train", "--system", "demo", "--run", "r1", "--seed", "0",
                 "--steps", "1", "--source", "declared",
                 "--anchors", "none", "--resume-optimizer", "no"]) == 2
    err = capsys.readouterr().err
    assert "--source demo_src:" in err and "2 excluded run(s)" in err, err


def test_steps_json_exists_only_once_a_step_trained_and_is_replaced_whole(tmp_path):
    from types import SimpleNamespace

    from aipf.train.fit import _StepLogger

    path = tmp_path / "steps.json"
    logger = _StepLogger(path)
    logger.on_exception(None, None, RuntimeError("before the first step"))
    logger.on_train_end(None, None)
    assert not path.exists()
    module = SimpleNamespace(last_step_terms={"L_dyn": 1.0})
    for value in (3.0, 2.0, 1.0):
        logger.on_train_batch_end(None, module, {"loss": torch.tensor(value)}, None, 0)
    assert json.loads(path.read_text())["loss"] == [3.0]  # the first step, then every EVERY
    logger.on_train_end(None, None)
    assert json.loads(path.read_text()) == {"loss": [3.0, 2.0, 1.0],
                                            "terms": [{"L_dyn": 1.0}] * 3}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["steps.json"]
