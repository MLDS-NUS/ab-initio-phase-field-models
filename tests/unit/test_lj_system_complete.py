"""The one-field published model pinned, and everything it needs declared beside it.

The structure mirrors ``test_feb_system_complete.py``. The CHECKPOINT tests pin the tracked file by
digest and refuse every shape a silent substitution can take. The AGREEMENT tests open the pinned
checkpoint and account for every key of its saved ``hyper_parameters`` in both directions. The mode
tier and a four-step training smoke read the Lennard-Jones raw root; they are marked ``env`` and skip,
naming it, when it is absent.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from aipf.system import Checkpoint, load

import declared_roots

#: A checkpoint path relative to a root, for the declaration's own refusals below.
PUBLISHED_PATH = "runs/published/final.ckpt"

#: Digest of the tracked copy: the published model's tensors and settings, without the run's
#: Lightning bookkeeping.
PUBLISHED_MD5 = "ef9da0b40952f7381f0ebaba876fdd5c"


def _lj():
    return load("lj")


def _md5(path: pathlib.Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scratch_or_skip(system):
    """The system's raw root, or a skip naming it."""
    return declared_roots.raw_or_skip(system.name)


def _published(system):
    """The published model's bytes, from the tracked copy the repository carries."""
    declared_roots.published_or_skip("lj")
    torch = pytest.importorskip("torch")
    return torch.load(system.resolve_checkpoint(), map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------
# the published model, declared
# --------------------------------------------------------------------------

def test_the_system_declares_the_published_model_checkpoint():
    s = _lj()
    assert s.checkpoint.path is None     # the tracked copy: declared by its digest alone
    assert s.checkpoint.md5 == PUBLISHED_MD5


def test_the_checkpoint_is_the_tracked_copy_wherever_the_raw_root_is(monkeypatch, tmp_path):
    from aipf.paths import published_checkpoint

    s = _lj()
    assert s.checkpoint_file() == published_checkpoint("lj")
    monkeypatch.setenv("AIPF_RAW_LJ", str(tmp_path))
    assert s.checkpoint_file() == published_checkpoint("lj")


def test_a_missing_file_is_refused_and_other_bytes_are_refused(tmp_path):
    ckpt = Checkpoint(path=PUBLISHED_PATH, md5=PUBLISHED_MD5)
    with pytest.raises(FileNotFoundError, match="missing or renamed"):
        ckpt.verify_under(tmp_path)
    target = tmp_path / PUBLISHED_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the published model")
    with pytest.raises(ValueError) as info:
        ckpt.verify_under(tmp_path)
    assert PUBLISHED_MD5 in str(info.value) and _md5(target) in str(info.value)


def test_the_published_model_is_the_final_epoch_the_config_asked_for():
    state = _published(_lj())
    assert state["epoch"] == 24 and state["hyper_parameters"]["anneal_epochs"] == 25
    assert _lj().defaults["epochs"] == 25


# --------------------------------------------------------------------------
# every saved hyperparameter, both directions
# --------------------------------------------------------------------------

#: defaults key -> (where in hparams, key there).
_IN_CHECKPOINT = {
    "sigma": ("top", "sigma"), "k_max": ("top", "k_max"),
    "alpha_loss": ("top", "alpha_loss"), "grid": ("top", "grid"), "box": ("top", "box"),
    "lr": ("top", "lr"), "weight_decay": ("top", "weight_decay"),
    "warmup_epochs": ("top", "warmup_epochs"), "anneal_epochs": ("top", "anneal_epochs"),
    "eta_min": ("top", "eta_min"), "h_inv_eps": ("top", "h_inv_eps"),
    "lambda_drift": ("top", "lambda_drift"), "lambda_M": ("top", "lambda_M"),
    "lambda_S": ("top", "lambda_S"), "m_table": ("top", "m_table"),
    "s_table": ("top", "s_table"),
    "R_cut": ("model", "R_cut"), "f_form": ("model", "f_form"),
    "taylor_max_order": ("model", "taylor_max_order"),
    "taylor_min_order": ("model", "taylor_min_order"),
    "gauge_fix_e": ("model", "gauge_fix_e"), "g_exc_hidden": ("model", "g_exc_hidden"),
    "m_form": ("model", "m_form"), "gamma": ("model", "gamma"),
    "m_T_form": ("model", "m_T_form"), "Ea_init": ("model", "Ea_init"),
    "T_ref": ("model", "T_ref"),
}

#: defaults keys the checkpoint does not carry, and where each comes from instead.
_NOT_IN_CHECKPOINT = {
    "rung": "the model class, from the published run's config",
    "k_cut": "baked into every modes.npz",
    "kernel_form": "this package's name for the radial kernel net",
    "epochs": "the Trainer's max_epochs, from the published run's config",
    "training": "the data split, window and loader: the run's hparams.yaml",
    "diagnose": "the published phase-diagram script's defaults",
    "md": "the archived engine deck",
    "mode_fields": "the published model's reader's phi",
    "estimator": "the drift estimator, from the published run's config",
}

#: Saved keys settled somewhere other than ``defaults``.
_SETTLED_ELSEWHERE = {
    ("top", "model_type"): "Functional.form and defaults['rung']",
    ("top", "model_kwargs"): "the container of the ('model', ...) keys",
    ("model", "activation"): "functional.kwargs['activation']",
    ("model", "disable_g_exc"): "functional.kwargs['disable_g']",
}


def test_every_declared_default_is_classified():
    assert set(_lj().defaults) == set(_IN_CHECKPOINT) | set(_NOT_IN_CHECKPOINT)


def test_every_saved_hyperparameter_is_accounted_for():
    hp = _published(_lj())["hyper_parameters"]
    saved = {("top", k) for k in hp} | {("model", k) for k in hp["model_kwargs"]}
    assert saved == set(_IN_CHECKPOINT.values()) | set(_SETTLED_ELSEWHERE)
    assert hp == _published(_lj())["config"]


def test_the_declarations_agree_with_the_saved_hyperparameters():
    s = _lj()
    hp = _published(s)["hyper_parameters"]
    for name, (where, key) in sorted(_IN_CHECKPOINT.items()):
        saved = (hp if where == "top" else hp["model_kwargs"])[key]
        if isinstance(saved, list):
            saved = tuple(saved)
        assert s.defaults[name] == saved, f"{name} -> {where}[{key}]"
    assert hp["model_type"] == "nonlocal_kernel_es" and s.defaults["rung"] == 3
    assert s.functional.kwargs["activation"] == hp["model_kwargs"]["activation"]
    assert s.functional.kwargs["disable_g"] == hp["model_kwargs"]["disable_g_exc"]
    tables = s.defaults["training"]["tables"]
    assert tables["key"] == "temperature"
    for role in ("m_table", "s_table"):
        # A path under the declared code tree; bare and relative where no tree is declared.
        assert tables[role] == hp[role] or tables[role].endswith("/" + hp[role]), role


# --------------------------------------------------------------------------
# what the checkpoint does not carry: the run's own hparams.yaml and the config
# --------------------------------------------------------------------------

#: training key -> the key under ``data:`` of the run's hparams.yaml and its config.
_FROM_DATA = {"modes_root": "modes_root", "train_temperatures": "train_temperatures",
              "val_seeds": "val_seeds", "test_temperatures": "test_temperatures",
              "half_width": "w", "n_states": "n_s", "stride": "stride",
              "estimator": "estimator", "batch_size": "batch_size"}


def test_the_test_temperature_is_held_out_of_training():
    block = _lj().defaults["training"]
    assert 1.20 not in block["train_temperatures"]
    assert block["test_temperatures"] == (1.20,) and block["val_seeds"] == (45,)


# --------------------------------------------------------------------------
# the engine deck and the diagnosis script the declarations quote
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# what is not this system's
# --------------------------------------------------------------------------

def test_no_engine_campaign_batch_or_shape_floor_is_declared():
    s = _lj()
    assert s.md_settings == {}
    assert s.anchor_rules.shape_T_min_by_pressure == {}


def test_no_published_knob_was_carried_over_from_another_system():
    lj, hhe, feb = _lj(), load("hhe"), load("feb")
    assert lj.checkpoint.md5 not in (hhe.checkpoint.md5, feb.checkpoint.md5)
    assert lj.functional.kwargs["nyquist_mask"] is False
    assert hhe.functional.kwargs["nyquist_mask"] is True
    assert feb.functional.kwargs["nyquist_mask"] is True
    for name in ("sigma", "R_cut", "T_ref", "lambda_M"):
        assert lj.defaults[name] != hhe.defaults[name], name


# --------------------------------------------------------------------------
# the mode tier and the training smoke (docs/reference/data.md)
# --------------------------------------------------------------------------

@pytest.mark.env
def test_the_archived_modes_carry_two_channels_and_the_system_declares_their_combination():
    """Two stored channels, one declared field: phi = 0.5 (rho_A - rho_B), zero mode 0.5."""
    np = pytest.importorskip("numpy")
    from aipf.pipeline.modes import mode_fields
    s = _lj()
    root = _scratch_or_skip(s) / s.defaults["training"]["modes_root"]
    temperatures = sorted(p.name for p in root.glob("T_*"))
    assert len(temperatures) == 13, temperatures
    with np.load(root / "T_1.30" / "seed_42" / "modes.npz") as z:
        assert z["rho_k"].shape[-1] == 2
    assert s.atom_types == {"A": 1}
    assert mode_fields(s) == ({"name": "A", "weights": {1: 0.5, 2: -0.5},
                               "mean": 0.5},)


#: The smoke's one source: three train runs of one train temperature (seed 45 was the published run's validation).
SMOKE_GLOB = "slab_overdamped_MD_xA_0.50_T_1.15_seed_4[234]"

#: The published model's saved loss weights, by this package's term: its saved L_S is the k -> 0 anchor.
PUBLISHED_TERMS = {"L_dyn": "lambda_drift", "L_M": "lambda_M", "L_bulk": "lambda_S"}


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """``aipf train --system lj ... --anchors declared --resume-optimizer no``, four steps, from the published model."""
    import os

    from aipf.cli.main import main
    from aipf.data import index

    s = _lj()
    _scratch_or_skip(s)
    tier = index.modes_dir(s, "")
    if not list(tier.glob(SMOKE_GLOB)):
        pytest.skip(f"the mode tier is not built under {tier}; see docs/reference/data.md")
    assert s.defaults["training"]["source_root"] == {"tier": "farm", "path": "modes"}
    # the run writes under its own data root, whose modes tier is the built one
    root = tmp_path_factory.mktemp("data")
    (root / "lj").mkdir()
    (root / "lj" / "modes").symlink_to(tier.resolve(), target_is_directory=True)
    old = os.environ.get("AIPF_DATA")
    os.environ["AIPF_DATA"] = str(root)
    try:
        code = main(["train", "--system", "lj", "--run", "c2c_four_steps",
                     "--seed", "0", "--steps", "4",
                     "--source", f"slab=.:{SMOKE_GLOB}:20,20,80",
                     "--init-from-published", "--resume-optimizer", "no",
                     "--anchors", "declared", "--log-every-step"])
    finally:
        if old is None:
            os.environ.pop("AIPF_DATA", None)
        else:
            os.environ["AIPF_DATA"] = old
    assert code == 0
    return root / "lj" / "ckpt" / "c2c_four_steps"


@pytest.mark.env
def test_four_training_steps_write_a_checkpoint_and_its_manifest(smoke_run):
    """``aipf train --system lj ... --anchors declared --resume-optimizer no``, four steps."""
    assert (smoke_run / "final.ckpt").is_file()
    manifest = json.loads((smoke_run / "MANIFEST.json").read_text())
    assert manifest["system"] == "lj" and manifest["global_step"] == 4
    assert manifest["resume_optimizer"] is False
    assert manifest["init_from"]["md5"] == PUBLISHED_MD5
    assert manifest["split_mode"] == "none" and manifest["loader_order"] == "shuffled"


@pytest.mark.env
def test_every_term_the_published_model_trained_is_trained(smoke_run):
    record = _published(_lj())["hyper_parameters"]
    trained_by_published = sorted(t for t, k in PUBLISHED_TERMS.items() if record[k] != 0.0)
    manifest = json.loads((smoke_run / "MANIFEST.json").read_text())
    assert sorted(manifest["terms_trained"]) == trained_by_published == ["L_M", "L_bulk", "L_dyn"]
    assert manifest["declared_weights_without_data"] == {}
    assert manifest["anchor_tables"]["rows"] == {"anchor_M": 4, "anchor_S": 0,
                                                 "anchor_bulk": 4, "anchor_P": 0}
    assert not (smoke_run / "UNTRAINED_TERMS.txt").exists()


@pytest.mark.env
def test_every_step_is_finite(smoke_run):
    import math
    steps = json.loads((smoke_run / "steps.json").read_text())
    assert len(steps["loss"]) == 4
    for loss, terms in zip(steps["loss"], steps["terms"]):
        assert math.isfinite(loss)
        assert set(terms) >= {"L_dyn", "L_M", "L_bulk"}
        assert all(math.isfinite(v) for v in terms.values()), terms
