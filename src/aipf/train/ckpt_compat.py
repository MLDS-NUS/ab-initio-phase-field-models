"""Checkpoint compatibility: saved flat ``hparams`` dicts to :class:`TrainConfig`,
and this package's own checkpoint format.

Every unmapped key raises, naming itself. The layout (A: Lightning hparams, B: k-modes) is always declared,
never guessed: ``lambda_S`` means ``L_S`` in layout A and ``L_bulk`` in layout B."""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple, Union

import torch

from aipf.losses.names import LossNameMap
from aipf.train.config import TrainConfig

#: canonical loss name -> the TrainConfig field its weight lives in.
_LAMBDA_FIELD: Mapping[str, str] = MappingProxyType({
    "L_dyn": "lambda_dyn", "L_M": "lambda_M", "L_S": "lambda_S",
    "L_bulk": "lambda_bulk", "L_P": "lambda_P", "L_conv": "lambda_conv",
    "L_Gamma": "lambda_gamma", "L_W": "lambda_W",
})

#: Layout A hardcodes L_dyn's weight at 1 instead of saving it.
IMPLICIT_WEIGHTS_A: Mapping[str, float] = MappingProxyType({"L_dyn": 1.0})
#: Layout B saves every weight it has; nothing is implicit.
IMPLICIT_WEIGHTS_B: Mapping[str, float] = MappingProxyType({})

#: Non-loss-weight constructor keys, layout A.
NONLOSS_KEY_MAP_LIGHTNING: Mapping[str, str] = MappingProxyType({
    "model_type": "model_type", "model_kwargs": "model_kwargs",
    "lr": "lr", "weight_decay": "weight_decay",
    "warmup_epochs": "warmup_epochs", "anneal_epochs": "anneal_epochs",
    "eta_min": "eta_min", "alpha_loss": "alpha_loss",
    "h_inv_eps": "h_inv_eps", "sigma": "sigma", "k_max": "k_max",
    "m_table": "m_table", "s_table": "s_table",
    "k_fit_stat": "k_fit_stat", "stat_metric": "stat_metric",
    "eos_csv": "eos_csv", "eos_csvs": "eos_csvs",
    "eos_pressures": "eos_pressures", "mask_dome": "mask_dome",
    "conv_penalty": "conv_penalty", "conv_margin": "conv_margin",
    "conv_samples": "conv_samples", "gamma_pt_samples": "gamma_pt_samples",
    "anchor_T_min_K": "anchor_T_min_K",
    "source_loss_weights": "source_loss_weights",
    # a field, not an extra: lit_module reads config.wd_ghat for the kernel head's group
    "wd_ghat": "wd_ghat",
    # experiment-specific, routed unchanged into extra_experiment_config:
    "wpsd_margin": "extra_experiment_config",
    "wpsd_k_max": "extra_experiment_config",
    "wpsd_n_k": "extra_experiment_config",
    "wpsd_kappa": "extra_experiment_config",
    "m_column": "extra_experiment_config",
    "p_ref_gpa": "extra_experiment_config",
    "eos_overlay": "extra_experiment_config",
    "anchor_row_weight": "extra_experiment_config",
    "anchor_gate": "extra_experiment_config",
    "grad_log_every": "extra_experiment_config",
    "resid_k_bins": "extra_experiment_config",
    "drift_scale": "extra_experiment_config",
    "conv_domain": "extra_experiment_config",
})

#: Layout B: a much smaller constructor.
NONLOSS_KEY_MAP_KMODES: Mapping[str, str] = MappingProxyType({
    "model_type": "model_type", "model_kwargs": "model_kwargs",
    "lr": "lr", "weight_decay": "weight_decay",
    "warmup_epochs": "warmup_epochs", "anneal_epochs": "anneal_epochs",
    "eta_min": "eta_min", "alpha_loss": "alpha_loss",
    "h_inv_eps": "h_inv_eps", "sigma": "sigma", "k_max": "k_max",
    "m_table": "m_table", "s_table": "s_table",
    "grid": "grid", "box": "box",
})


def _pop_loss_weights(hparams: Dict[str, Any], weight_map: LossNameMap, *,
                       implicit_weights: Mapping[str, float]) -> Dict[str, Any]:
    """Pop the keys ``weight_map`` declares from ``hparams`` (a copy) and return
    ``{TrainConfig field: value}``.

    An absent key falls back to ``implicit_weights``, else to the schema default."""
    out: Dict[str, Any] = {}
    for canonical, field_name in _LAMBDA_FIELD.items():
        if canonical in weight_map.canonical_names:
            code = weight_map.to_code(canonical)
            if code in hparams:
                out[field_name] = float(hparams.pop(code))
        elif canonical in implicit_weights:
            out[field_name] = float(implicit_weights[canonical])
    return out


def hparams_to_train_config(
    hparams: Mapping[str, Any], *,
    weight_map: LossNameMap,
    key_map: Mapping[str, str],
    implicit_weights: Mapping[str, float] = MappingProxyType({}),
) -> TrainConfig:
    """A saved flat ``hparams`` dict -> :class:`TrainConfig`; ``weight_map`` and ``key_map`` are required.

    Loss weights first, then ``key_map`` renames (or ``extra_experiment_config``); a leftover key raises."""
    remaining = dict(hparams)
    fields: Dict[str, Any] = _pop_loss_weights(
        remaining, weight_map, implicit_weights=implicit_weights)

    unmapped = sorted(k for k in remaining if k not in key_map)
    if unmapped:
        raise ValueError(
            f"hparams key(s) with no declared mapping for this "
            f"layout: {unmapped}. Add an explicit entry to the relevant "
            f"NONLOSS_KEY_MAP_* (or to the loss-weight map, if this "
            f"is actually a loss weight) rather than dropping it -- an "
            f"unmapped key here means the run used a knob this "
            f"compatibility layer does not yet account for, and silently "
            f"ignoring it would train or diagnose a different model than "
            f"the one that was saved.")

    extra: Dict[str, Any] = dict(fields.pop("extra_experiment_config", {}) or {})
    for key, value in remaining.items():
        target = key_map[key]
        if target == "extra_experiment_config":
            extra[key] = value
        else:
            fields[target] = value
    if extra:
        fields["extra_experiment_config"] = extra
    return TrainConfig(**fields)


def read_hparams_checkpoint(path: Union[str, Path]) -> Tuple[dict, Dict[str, torch.Tensor]]:
    """``(hparams, model_state_dict)`` from a saved ``.ckpt``: its ``config`` and
    ``model_state_dict`` entries, unrenamed."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "config" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError(
            f"{path}: missing 'config' or 'model_state_dict'; this does "
            f"not look like a checkpoint written by the on_save_checkpoint "
            f"hook of a LightningModule (keys "
            f"present: {sorted(ckpt)})")
    return dict(ckpt["config"]), dict(ckpt["model_state_dict"])


#: Schema tag of a new checkpoint's config, so a reader never guesses the layout from key spelling.
NEW_CONFIG_SCHEMA_TAG = "aipf.train.TrainConfig.v1"
#: Tags a reader accepts.
CONFIG_SCHEMA_TAGS = (NEW_CONFIG_SCHEMA_TAG,)


def save_new_checkpoint(path: Union[str, Path], model: torch.nn.Module,
                         config: TrainConfig) -> None:
    """Write a checkpoint in this package's format; ``config`` round-trips via
    ``TrainConfig.as_dict``/``from_dict``."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.as_dict(),
        "config_schema": NEW_CONFIG_SCHEMA_TAG,
    }, str(path))


def load_new_checkpoint(path: Union[str, Path]) -> Tuple[Dict[str, torch.Tensor], TrainConfig]:
    """``(model_state_dict, TrainConfig)`` from a checkpoint tagged :data:`NEW_CONFIG_SCHEMA_TAG`;
    anything else raises."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if ckpt.get("config_schema") != NEW_CONFIG_SCHEMA_TAG:
        raise ValueError(
            f"{path}: config_schema={ckpt.get('config_schema')!r}, not "
            f"{NEW_CONFIG_SCHEMA_TAG!r}. This is not a checkpoint "
            f"aipf.train wrote itself -- for a saved hparams checkpoint, read it "
            f"with read_hparams_checkpoint() and translate its config with "
            f"hparams_to_train_config(), declaring which layout "
            f"produced it.")
    return dict(ckpt["model_state_dict"]), TrainConfig.from_dict(ckpt["config"])
