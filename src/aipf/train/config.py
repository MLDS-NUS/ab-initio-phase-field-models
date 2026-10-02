"""The training-run config schema: a strict, flat, frozen dataclass, and the run's seed streams.

Loss weights default to ``0.0`` (off) except ``lambda_dyn = 1.0``; system-varying fields have no default."""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

_STAT_METRICS: Tuple[str, ...] = ("rel_frob", "s_metric")
_BULK_RESIDUALS: Tuple[str, ...] = ("relative_inverse", "sigma_chi2")
_CONV_PENALTIES: Tuple[str, ...] = ("hinge", "hinge0", "softplus")

#: Named random streams of one run, each seeded by :func:`derive_seed`: ``model``, ``data``.
SEED_STREAMS: Tuple[str, ...] = ("model", "data")


def derive_seed(seed: int, stream: str) -> int:
    """``sha256("<seed>:<stream>")`` truncated to 63 bits; raises on an unknown ``stream`` or a
    non-int ``seed``."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            f"seed must be an int, got {seed!r} ({type(seed).__name__})")
    if stream not in SEED_STREAMS:
        raise ValueError(
            f"unknown seed stream {stream!r}; the declared streams are "
            f"{SEED_STREAMS}")
    digest = hashlib.sha256(f"{seed}:{stream}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _frozen_dict(d: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """A read-only view, so two ``TrainConfig``s never share a mutable dict."""
    return dict(d) if d else {}


@dataclass(frozen=True)
class TrainConfig:
    """A training run's hyperparameters, canonical loss names throughout; an unrecognised key raises
    ``TypeError``."""

    # -- model construction: opaque passthrough
    model_type: Optional[str] = None
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    # -- optimizer / LR schedule
    lr: float = 5e-4
    weight_decay: float = 0.0
    # the temperature heads' own decay; None = UNDECLARED, raised on only when the model has the heads
    wd_ghat: Optional[float] = None
    warmup_epochs: int = 5
    anneal_epochs: int = 100
    eta_min: float = 1e-6

    # -- L_dyn (weak-form drift residual) shape; `sigma`/`k_max` are per-system, no default
    alpha_loss: float = 0.0
    h_inv_eps: float = 1e-6
    sigma: Optional[float] = None
    k_max: Optional[float] = None

    # -- canonical loss weights (the seven canonical losses, plus L_W); 0.0 = off
    lambda_dyn: float = 1.0
    lambda_M: float = 0.0
    lambda_S: float = 0.0
    lambda_bulk: float = 0.0
    lambda_P: float = 0.0
    lambda_conv: float = 0.0
    lambda_gamma: float = 0.0
    lambda_W: float = 0.0

    # -- anchor-table references, resolved by a System, never interpreted here
    m_table: Optional[Any] = None
    s_table: Optional[Any] = None
    eos_csv: Optional[str] = None
    eos_csvs: Optional[Any] = None
    eos_pressures: Optional[Any] = None
    mask_dome: bool = False
    k_fit_stat: Optional[float] = None

    # -- L_S / L_bulk residual form (admissible values: _STAT_METRICS, _BULK_RESIDUALS)
    stat_metric: str = "rel_frob"
    bulk_residual: str = "relative_inverse"

    # -- L_conv / L_Gamma penalty shape (conv_penalty in _CONV_PENALTIES)
    conv_penalty: str = "hinge"
    conv_margin: float = 0.0
    conv_samples: int = 0
    gamma_pt_samples: int = 0
    # the probe points' own stream; None = undeclared, raised on only when a penalty is trained
    penalty_seed: Optional[int] = None

    # -- the anchor temperature cut this run used, recorded
    anchor_T_min_K: Mapping[Any, Any] = field(default_factory=dict)

    # -- multi-source per-step loss weighting
    source_loss_weights: Mapping[str, float] = field(default_factory=dict)

    # -- a fixed grid/box for a run whose ensemble never varies it
    grid: Optional[Sequence[int]] = None
    box: Optional[Sequence[float]] = None

    # -- the one declared seed; None = undeclared, and every consumer raises on it
    seed: Optional[int] = None

    # -- experiment-specific knobs under their saved key, routed by ckpt_compat
    extra_experiment_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_kwargs", _frozen_dict(self.model_kwargs))
        object.__setattr__(self, "anchor_T_min_K", _frozen_dict(self.anchor_T_min_K))
        object.__setattr__(self, "source_loss_weights",
                            _frozen_dict(self.source_loss_weights))
        object.__setattr__(self, "extra_experiment_config",
                            _frozen_dict(self.extra_experiment_config))
        if self.stat_metric not in _STAT_METRICS:
            raise ValueError(
                f"stat_metric={self.stat_metric!r} not in {_STAT_METRICS}")
        if self.bulk_residual not in _BULK_RESIDUALS:
            raise ValueError(
                f"bulk_residual={self.bulk_residual!r} not in {_BULK_RESIDUALS}")
        if self.conv_penalty not in _CONV_PENALTIES:
            raise ValueError(
                f"conv_penalty={self.conv_penalty!r} not in {_CONV_PENALTIES}")
        for name in ("seed", "penalty_seed"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, int)):
                raise TypeError(
                    f"{name} must be an int or None, got {value!r} "
                    f"({type(value).__name__}); a seed read out of a config "
                    f"file as a string or a float would derive different "
                    f"stream seeds from the same written number")

    def seed_for(self, stream: str) -> int:
        """This run's seed for one named ``stream`` (:data:`SEED_STREAMS`); raises when no seed was
        declared."""
        if self.seed is None:
            raise ValueError(
                f"this run declares no seed, so no {stream!r} seed can be "
                f"derived: TrainConfig.seed is None. A seed is the caller's "
                f"and this schema has no default for it -- declare one to "
                f"get a reproducible model, draws and first step.")
        return derive_seed(self.seed, stream)

    def as_dict(self) -> dict:
        """The flat dict a checkpoint's ``config`` entry stores."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "TrainConfig":
        """Build from this schema's own field names; a saved flat ``hparams`` dict goes through
        ``aipf.train.ckpt_compat.hparams_to_train_config`` first."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"TrainConfig.from_dict: unknown key(s) {unknown}; this "
                f"schema has no field by that name. If this dict came from "
                f"a Lightning-hparams checkpoint's flat hparams, map it through "
                f"aipf.train.ckpt_compat.hparams_to_train_config "
                f"first -- that function raises, naming itself, on any key "
                f"it does not recognise either, rather than silently "
                f"dropping it here.")
        return cls(**mapping)
