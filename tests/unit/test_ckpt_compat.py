"""Tests for `aipf.train.ckpt_compat` and `aipf.train.checkpoint_formats`: the layer
that reads a published checkpoint's training settings at all.

The two formats are the Lightning-hparams one (the H/He and Fe-B checkpoints) and the
k-modes one (the Lennard-Jones checkpoint). Synthetic dictionaries in each format's own
key spelling come first; the tracked published checkpoints themselves are read at the
bottom of this file.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aipf.losses.names import WEIGHT_MAP_A, WEIGHT_MAP_B
from aipf.train import checkpoint_formats as lw
from aipf.train.ckpt_compat import (
    IMPLICIT_WEIGHTS_A,
    IMPLICIT_WEIGHTS_B,
    NONLOSS_KEY_MAP_LIGHTNING,
    NONLOSS_KEY_MAP_KMODES,
    hparams_to_train_config,
    load_new_checkpoint,
    read_hparams_checkpoint,
    save_new_checkpoint,
)
from aipf.train.config import TrainConfig

import declared_roots

# ---------------------------------------------------------------------------
# Small synthetic hparams dicts in each format, no real files needed. Values
# are made up but key SPELLING is measured from the published checkpoints.
# ---------------------------------------------------------------------------

def _lightning_hparams(**overrides):
    d = {
        "model_type": "nonlocal_kernel_2s", "model_kwargs": {"h_u": 32},
        "lr": 5e-4, "weight_decay": 0.0, "wd_ghat": 0.0,
        "warmup_epochs": 5, "anneal_epochs": 100, "eta_min": 1e-6,
        "alpha_loss": 0.0, "h_inv_eps": 1e-6, "sigma": 2.0, "k_max": 2.0,
        "lambda_M": 0.1, "m_table": "m.npz",
        "lambda_S": 0.05, "s_table": "s.npz",
        "k_fit_stat": 1.5, "stat_metric": "rel_frob",
        "lambda_bulk": 0.02, "lambda_P": 0.0,
        "eos_csv": None, "eos_csvs": None, "eos_pressures": None,
        "mask_dome": False, "lambda_conv": 0.0, "conv_penalty": "hinge",
        "conv_margin": 0.05, "conv_samples": 1024, "lambda_gamma": 0.0,
        "gamma_pt_samples": 16, "anchor_T_min_K": {200: 6100},
        "source_loss_weights": None,
    }
    d.update(overrides)
    return d


def _kmodes_hparams(**overrides):
    d = {
        "model_type": "nonlocal_kernel_es", "model_kwargs": {"gamma": 0.1},
        "lr": 5e-4, "weight_decay": 0.0, "warmup_epochs": 5,
        "anneal_epochs": 25, "eta_min": 1e-6, "alpha_loss": 0.0,
        "h_inv_eps": 1e-6, "sigma": 1.5, "k_max": None,
        "lambda_drift": 1.0, "lambda_M": 0.01, "m_table": "m.npz",
        "lambda_S": 0.01, "s_table": "s.npz",
        "grid": (20, 20, 80), "box": (9.5, 9.5, 38.1),
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# The lambda_S collision, resolved per format
# ---------------------------------------------------------------------------

def test_lambda_s_resolves_to_l_s_in_the_lightning_format():
    cfg = hparams_to_train_config(
        _lightning_hparams(), weight_map=WEIGHT_MAP_A,
        key_map=NONLOSS_KEY_MAP_LIGHTNING, implicit_weights=IMPLICIT_WEIGHTS_A)
    assert cfg.lambda_S == pytest.approx(0.05)
    assert cfg.lambda_bulk == pytest.approx(0.02)  # a SEPARATE key in the Lightning-hparams format


def test_lambda_s_resolves_to_l_bulk_in_the_kmodes_format():
    cfg = hparams_to_train_config(
        _kmodes_hparams(), weight_map=WEIGHT_MAP_B,
        key_map=NONLOSS_KEY_MAP_KMODES, implicit_weights=IMPLICIT_WEIGHTS_B)
    # the k-modes format's checkpoint key "lambda_S" means L_bulk, NOT L_S.
    assert cfg.lambda_bulk == pytest.approx(0.01)
    assert cfg.lambda_S == 0.0  # the schema default: this format has no L_S


def test_lightning_l_dyn_weight_is_implicit_not_a_checkpoint_key():
    """Measured (see aipf.losses.names' corrected WEIGHT_MAP_A docstring):
    the Lightning-hparams format has NO `lambda_drift` key at all; L_dyn's weight is hardcoded
    at 1 in that format's own training_step."""
    hparams = _lightning_hparams()
    assert "lambda_drift" not in hparams
    cfg = hparams_to_train_config(
        hparams, weight_map=WEIGHT_MAP_A, key_map=NONLOSS_KEY_MAP_LIGHTNING,
        implicit_weights=IMPLICIT_WEIGHTS_A)
    assert cfg.lambda_dyn == 1.0


def test_lightning_implicit_weight_is_actually_threaded_through_not_a_coincidence():
    """TrainConfig's OWN schema default for lambda_dyn also happens to be
    1.0, so the test above alone cannot tell "the implicit-weights
    mechanism worked" apart from "it silently did nothing and the schema
    default carried it". Using a deliberately non-default implicit weight
    closes that gap."""
    cfg = hparams_to_train_config(
        _lightning_hparams(), weight_map=WEIGHT_MAP_A,
        key_map=NONLOSS_KEY_MAP_LIGHTNING,
        implicit_weights={"L_dyn": 0.37})
    assert cfg.lambda_dyn == pytest.approx(0.37)


def test_kmodes_l_dyn_weight_is_a_real_checkpoint_key():
    cfg = hparams_to_train_config(
        _kmodes_hparams(lambda_drift=0.7), weight_map=WEIGHT_MAP_B,
        key_map=NONLOSS_KEY_MAP_KMODES, implicit_weights=IMPLICIT_WEIGHTS_B)
    assert cfg.lambda_dyn == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# An unmapped key raises, naming itself; no format declared raises too
# ---------------------------------------------------------------------------

def test_unmapped_key_raises_naming_itself():
    with pytest.raises(ValueError, match="totally_unrecognised_knob"):
        hparams_to_train_config(
            _lightning_hparams(totally_unrecognised_knob=1.0),
            weight_map=WEIGHT_MAP_A, key_map=NONLOSS_KEY_MAP_LIGHTNING,
            implicit_weights=IMPLICIT_WEIGHTS_A)


def test_no_declared_format_raises_rather_than_guessing():
    """`weight_map`/`key_map` are required, with no default -- a checkpoint
    that does not say which format it is in cannot be mapped at all."""
    with pytest.raises(TypeError):
        hparams_to_train_config(_kmodes_hparams())


def test_lightning_extra_experiment_specific_knobs_are_kept_verbatim():
    cfg = hparams_to_train_config(
        _lightning_hparams(conv_domain=[0.1, 0.2, 0.3, 0.4]),
        weight_map=WEIGHT_MAP_A, key_map=NONLOSS_KEY_MAP_LIGHTNING,
        implicit_weights=IMPLICIT_WEIGHTS_A)
    assert cfg.extra_experiment_config["conv_domain"] == [0.1, 0.2, 0.3, 0.4]


def test_lightning_wd_ghat_reaches_the_field_the_optimizer_reads():
    """``wd_ghat`` is a ``TrainConfig`` FIELD and not a verbatim extra.

    It used to be routed into ``extra_experiment_config``, where nothing
    reads it: ``lit_module`` builds the kernel head's parameter group from
    ``config.wd_ghat``, so a published checkpoint that declared the number
    came back with the field at ``None`` -- either a silently dropped
    weight decay or, on a model with the heads, a refusal naming a key the
    checkpoint had in fact carried.
    """
    cfg = hparams_to_train_config(
        _lightning_hparams(wd_ghat=0.01),
        weight_map=WEIGHT_MAP_A, key_map=NONLOSS_KEY_MAP_LIGHTNING,
        implicit_weights=IMPLICIT_WEIGHTS_A)
    assert cfg.wd_ghat == 0.01
    assert "wd_ghat" not in cfg.extra_experiment_config


# ---------------------------------------------------------------------------
# New-format checkpoint round trip: bit for bit
# ---------------------------------------------------------------------------

def test_new_checkpoint_round_trip_bit_for_bit(tmp_path):
    model = nn.Linear(3, 3)
    config = TrainConfig(sigma=1.0, lambda_M=0.4)
    path = tmp_path / "ckpt.pt"
    save_new_checkpoint(path, model, config)
    state_dict, loaded_config = load_new_checkpoint(path)
    assert loaded_config == config
    fresh = nn.Linear(3, 3)
    fresh.load_state_dict(state_dict)
    for key in model.state_dict():
        assert torch.equal(model.state_dict()[key], fresh.state_dict()[key])


def test_load_new_checkpoint_refuses_a_foreign_file(tmp_path):
    path = tmp_path / "foreign.ckpt"
    torch.save({"config": {"lr": 1e-3}, "model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="not a checkpoint aipf.train wrote"):
        load_new_checkpoint(path)


def test_read_hparams_checkpoint_refuses_an_unrelated_file(tmp_path):
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"foo": 1}, path)
    with pytest.raises(ValueError):
        read_hparams_checkpoint(path)


# ---------------------------------------------------------------------------
# The tracked k-modes checkpoint, the Lennard-Jones published model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_checkpoint():
    declared_roots.published_or_skip("lj")
    from aipf.system import load
    return read_hparams_checkpoint(load("lj").resolve_checkpoint())


def test_real_checkpoint_config_maps_onto_train_config(real_checkpoint):
    hparams, _state_dict = real_checkpoint
    cfg = hparams_to_train_config(
        hparams, weight_map=WEIGHT_MAP_B,
        key_map=NONLOSS_KEY_MAP_KMODES, implicit_weights=IMPLICIT_WEIGHTS_B)
    assert cfg.lr == pytest.approx(5e-4)
    assert cfg.sigma == pytest.approx(1.5)
    assert cfg.k_max is None
    assert cfg.lambda_dyn == pytest.approx(1.0)     # real lambda_drift key
    assert cfg.lambda_M == pytest.approx(0.01)
    assert cfg.lambda_bulk == pytest.approx(0.01)   # lambda_S -> L_bulk here
    assert cfg.lambda_S == 0.0                       # this format has no L_S
    assert cfg.grid == (20, 20, 80)
    assert cfg.model_type == "nonlocal_kernel_es"


def test_taylor_conversion_refuses_a_non_half_rho_ref():
    with pytest.raises(ValueError, match="rho_ref"):
        lw.taylor_coefs_from_psi_centered(
            torch.zeros(4), [2, 4, 6, 8], rho_ref=0.3)


# ---------------------------------------------------------------------------
# Every hparams key each published model carries, round-tripped
# ---------------------------------------------------------------------------
# A published model's ``hyper_parameters`` go through its format's maps into a
# TrainConfig, through this package's own checkpoint format and back, and every
# key is then read back off the config by the same maps: same value, same type,
# all the way down. Each is the system's declared, digest-verified, tracked file.

_FORMATS = {
    "lightning": (WEIGHT_MAP_A, NONLOSS_KEY_MAP_LIGHTNING, IMPLICIT_WEIGHTS_A),
    "kmodes": (WEIGHT_MAP_B, NONLOSS_KEY_MAP_KMODES, IMPLICIT_WEIGHTS_B),
}

#: published model -> the format it was written in.
_PUBLISHED_FORMAT = {"hhe": "lightning", "feb": "lightning", "lj": "kmodes"}


def _published_file(system: str) -> Path:
    declared_roots.published_or_skip(system)
    from aipf.system import load
    return load(system).verify_checkpoint()


def _same(got, want, where: str) -> None:
    """Equal value AND type, recursively: ``(1, 2)`` is not ``[1, 2]`` and ``1`` is not ``1.0``."""
    assert type(got) is type(want), f"{where}: {type(got)} vs {type(want)}"
    if isinstance(want, dict):
        assert list(got) == list(want), f"{where}: keys {list(got)} vs {list(want)}"
        for key in want:
            _same(got[key], want[key], f"{where}[{key!r}]")
    elif isinstance(want, (list, tuple)):
        assert len(got) == len(want), f"{where}: length"
        for i, (g, w) in enumerate(zip(got, want)):
            _same(g, w, f"{where}[{i}]")
    else:
        assert got == want, f"{where}: {got!r} vs {want!r}"


def _read_back(cfg: TrainConfig, key: str, weight_map, key_map):
    """``hparams[key]`` as the declared maps put it into ``cfg``."""
    from aipf.train.ckpt_compat import _LAMBDA_FIELD

    if key in weight_map.code_symbols:
        return getattr(cfg, _LAMBDA_FIELD[weight_map.to_canonical(key)])
    target = key_map[key]
    if target == "extra_experiment_config":
        return cfg.extra_experiment_config[key]
    return getattr(cfg, target)


@pytest.mark.parametrize("system", sorted(_PUBLISHED_FORMAT))
def test_every_published_hparams_key_round_trips(system, tmp_path):
    import dataclasses

    from aipf.train.ckpt_compat import _LAMBDA_FIELD

    saved = torch.load(str(_published_file(system)), map_location="cpu",
                       weights_only=False)
    hparams = saved["hyper_parameters"]
    weight_map, key_map, implicit = _FORMATS[_PUBLISHED_FORMAT[system]]

    cfg = hparams_to_train_config(
        hparams, weight_map=weight_map, key_map=key_map,
        implicit_weights=implicit)
    path = tmp_path / "round_trip.ckpt"
    save_new_checkpoint(path, nn.Linear(1, 1), cfg)
    _state, reread = load_new_checkpoint(path)
    assert reread == cfg

    for key, value in hparams.items():
        _same(_read_back(reread, key, weight_map, key_map), value,
              f"{system}: {key}")

    # and nothing arrived that no key (or the format's implicit weight) put there
    placed = {"extra_experiment_config"}
    placed |= {_LAMBDA_FIELD[weight_map.to_canonical(k)]
               for k in hparams if k in weight_map.code_symbols}
    placed |= {key_map[k] for k in hparams if k not in weight_map.code_symbols}
    placed |= {_LAMBDA_FIELD[c] for c in implicit}
    default = TrainConfig()
    invented = {f.name for f in dataclasses.fields(TrainConfig)
                if f.name not in placed
                and getattr(reread, f.name) != getattr(default, f.name)}
    assert not invented, f"{system}: fields no hparams key set: {invented}"
