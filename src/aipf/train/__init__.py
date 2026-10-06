"""``aipf.train``: config schema, LitModule, checkpoint compatibility, and the ``fit`` driver."""
from __future__ import annotations

from aipf.train.ckpt_compat import (
    IMPLICIT_WEIGHTS_A,
    IMPLICIT_WEIGHTS_B,
    NONLOSS_KEY_MAP_LIGHTNING,
    NONLOSS_KEY_MAP_KMODES,
    NEW_CONFIG_SCHEMA_TAG,
    hparams_to_train_config,
    load_new_checkpoint,
    read_hparams_checkpoint,
    save_new_checkpoint,
)
from aipf.train.anchors import NO_ANCHORS, AnchorTables
from aipf.train.config import TrainConfig
from aipf.train.lit_module import LitModule
from aipf.train.pressure import pressure_from_model
from aipf.train.projection import PROJECTIONS, project_kz0, projection_depth
from aipf.train.sampling import sample_uniform

__all__ = [
    "AnchorTables",
    "NO_ANCHORS",
    "TrainConfig",
    "LitModule",
    "sample_uniform",
    "pressure_from_model",
    "hparams_to_train_config",
    "read_hparams_checkpoint",
    "save_new_checkpoint",
    "load_new_checkpoint",
    "NONLOSS_KEY_MAP_LIGHTNING",
    "NONLOSS_KEY_MAP_KMODES",
    "IMPLICIT_WEIGHTS_A",
    "IMPLICIT_WEIGHTS_B",
    "NEW_CONFIG_SCHEMA_TAG",
    "PROJECTIONS",
    "project_kz0",
    "projection_depth",
]
