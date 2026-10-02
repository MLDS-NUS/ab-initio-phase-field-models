"""The training driver: a declared system in, a signed run directory
``<data root>/<sys>/ckpt/<run_name>/`` out.

Writes ``final.ckpt`` itself (Lightning checkpointing off), ``hparams.yaml`` and ``MANIFEST.json``.
A declared weight without data is recorded in the manifest, in ``hparams.yaml`` ``warnings``
and in ``UNTRAINED_TERMS.txt``."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import (Any, Dict, Mapping, Optional, Sequence, Tuple,
                    Union)

import lightning as L
import torch
import yaml

from aipf.functional.build import build as build_functional
from aipf.system import Checkpoint, System

from .ckpt_compat import CONFIG_SCHEMA_TAGS, NEW_CONFIG_SCHEMA_TAG
from .anchors import NO_ANCHORS, AnchorTables, _NoAnchors
from .config import TrainConfig
from .datamodule import (LoaderSettings, ModeDataModule, RunWeighting,
                          SourceSpec, SplitSettings)
from .dataset import ArchiveKeys, WindowSettings
from .checkpoint_formats import (kmodes_model_state_dict, load_kmodes_into,
                                 load_lightning_hparams_into)
from .kernel_hinge import HINGE_TERM, KernelHinge, declared_fields as _hinge_fields
from .lit_module import LitModule, seeded_rng
from .penalties import PENALTY_TERMS, penalties_from_system

__all__ = ["DEVICES", "DeviceUnavailable", "ReservedRunName", "fit", "pre_run_checks",
           "resolve_device"]

#: Where a run trains: ``auto`` is ``cuda`` when torch sees one, else ``cpu``.
DEVICES = ("auto", "cpu", "cuda")


class DeviceUnavailable(RuntimeError):
    """The device asked for is not there; raised before the run directory exists."""


def resolve_device(device: str) -> str:
    """``cpu`` or ``cuda`` for ``device``, or :class:`DeviceUnavailable` when ``cuda`` is asked for and absent."""
    if device not in DEVICES:
        raise ValueError(f"device is one of {DEVICES}, got {device!r}")
    available = bool(torch.cuda.is_available())
    if device == "cuda" and not available:
        raise DeviceUnavailable("device cuda requested, torch.cuda.is_available() is False")
    if device == "auto":
        return "cuda" if available else "cpu"
    return device

_LOG = logging.getLogger(__name__)

#: The :attr:`System.defaults` key this module reads.
_TRAINING_KEY = "training"

#: The entry of that block naming the anchor tables (the keywords of ``AnchorTables.load``).
_TABLES_KEY = "tables"

#: Archive-format file names, shared by every tree this package reads.
_MODES_FILE = "modes.npz"
_QUALITY_FILE, _QUALITY_KEY = "DATA_QUALITY_WARNING.json", "valid_frames"

#: The channel axis of the archived ``(n_frames, n_modes, n_channels)`` amplitudes.
_ARCHIVE_CHANNEL_AXIS = 2


# -- reading the declaration --
def _training(system: System) -> Mapping[str, Any]:
    """``system.defaults["training"]``, or a refusal that names the key."""
    block = system.defaults.get(_TRAINING_KEY)
    if not isinstance(block, Mapping):
        raise KeyError(
            f"system {system.name!r} declares no defaults[{_TRAINING_KEY!r}]: "
            f"the window shape, the run weighting, the split and the loader "
            f"shape are this system's choices and this driver ships no "
            f"value for any of them")
    return block


def _declared(block: Mapping[str, Any], name: str, where: str) -> Any:
    """One declared value, or a refusal that names it and where it belongs."""
    if name not in block:
        raise KeyError(
            f"{where} needs defaults[{_TRAINING_KEY!r}][{name!r}] and the "
            f"system declares none. Every field of {where} is required: it "
            f"has no default, because one system's value is another's bug")
    return block[name]


def _config_from_system(system: System) -> Dict[str, Any]:
    """Every :class:`TrainConfig` field ``system.defaults`` declares; the ``"training"`` block wins
    over the top level. The kernel hinge's keys, spelt as a saved checkpoint spells them, map as
    :mod:`aipf.train.ckpt_compat` maps that checkpoint (:func:`aipf.train.kernel_hinge.declared_fields`)."""
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    out = {k: v for k, v in system.defaults.items() if k in fields}
    out.update({k: v for k, v in _training(system).items() if k in fields})
    if "source_loss_weights" in out:
        out["source_loss_weights"] = dict(out["source_loss_weights"])
    hinge = _hinge_fields({**system.defaults, **_training(system)})
    if "extra_experiment_config" in hinge:
        hinge["extra_experiment_config"] = {**dict(out.get("extra_experiment_config") or {}),
                                            **hinge["extra_experiment_config"]}
    out.update(hinge)
    return out


def _window_from_system(system: System) -> WindowSettings:
    """The window shape; the functional's grid is a placeholder ``ModeDataModule`` replaces per source."""
    block = _training(system)
    where = "WindowSettings"
    if system.functional is None or "grid" not in system.functional.kwargs:
        raise KeyError(
            f"{where} needs a grid and system {system.name!r} declares no "
            f"functional grid to stand in for the per-source ones")
    return WindowSettings(
        estimator=str(system.defaults["estimator"]),
        half_width=int(_declared(block, "half_width", where)),
        n_states=int(_declared(block, "n_states", where)),
        stride=int(_declared(block, "stride", where)),
        grid=tuple(system.functional.kwargs["grid"]),
        savgol_window=_declared(block, "savgol_window", where),
        savgol_poly=_declared(block, "savgol_poly", where),
        # optional: a system that declares no band scatters every stored mode
        band_k_max=block.get("band_k_max"))


def _weighting_from_system(system: System, k_max: float) -> RunWeighting:
    """The run weighting over the drift term's own band; ``k_max`` arrives resolved
    (:meth:`LitModule.resolved_k_max`)."""
    block = _training(system)
    where = "RunWeighting"
    mode = str(_declared(block, "run_weighting", where))
    if mode == "uniform":
        # The band knobs are refused by this mode, not merely unused.
        return RunWeighting(mode=mode, sigma=None, k_max=None, eps=None,
                            probe_every=None)
    return RunWeighting(
        mode=mode,
        sigma=float(system.defaults["sigma"]),
        k_max=float(k_max),
        eps=float(system.defaults["h_inv_eps"]),
        probe_every=int(_declared(block, "run_weight_probe_every", where)))


def _split_from_system(system: System,
                       mode: Optional[str] = None) -> SplitSettings:
    """The declared split; ``mode`` (:data:`aipf.train.datamodule.SPLITS`) overrides it, ``None`` =
    the declaration."""
    block = _training(system)
    where = "SplitSettings"
    return SplitSettings(
        mode=str(_declared(block, "val_split", where)
                 if mode is None else mode),
        val_labels=tuple(_declared(block, "val_labels", where)),
        val_fraction=float(_declared(block, "val_fraction", where)),
        seed=int(_declared(block, "split_seed", where)))


def _loader_from_system(system: System,
                        order: Optional[str] = None) -> LoaderSettings:
    """The declared loader shape; ``order`` (:data:`aipf.train.datamodule.ORDERS`) overrides it,
    ``None`` = the declaration."""
    block = _training(system)
    where = "LoaderSettings"
    return LoaderSettings(
        batch_size=int(_declared(block, "batch_size", where)),
        num_workers=int(_declared(block, "num_workers", where)),
        pin_memory=bool(_declared(block, "pin_memory", where)),
        drop_last=bool(_declared(block, "drop_last", where)),
        order=str(_declared(block, "order", where)
                  if order is None else order))


def _anchors_from_system(system: System, anchors):
    """The anchor tier this run trains, or ``None`` for drift alone: ``None`` loads the declared tables,
    :data:`~aipf.train.anchors.NO_ANCHORS` loads none, an ``AnchorTables`` is used as given."""
    if anchors is NO_ANCHORS:
        return None
    if anchors is not None:
        return anchors
    tables = _training(system).get(_TABLES_KEY)
    if tables is None:
        return None
    return AnchorTables.load(system, **tables)


def _archive_keys(system: System) -> ArchiveKeys:
    """How this system's mode archive spells its quantities; the names come from
    :mod:`aipf.pipeline.modes`, the writer."""
    from aipf.pipeline.modes import ARCHIVE_KEYS, SIDE_KEYS

    return ArchiveKeys(
        file_name=_MODES_FILE,
        amplitudes=ARCHIVE_KEYS["amplitudes"],
        amplitudes_channel_axis=_ARCHIVE_CHANNEL_AXIS,
        labels=ARCHIVE_KEYS["labels"],
        box=ARCHIVE_KEYS["box"],
        temperature=ARCHIVE_KEYS["temperature"],
        frame_interval=ARCHIVE_KEYS["frame_interval"],
        composition=SIDE_KEYS,
        composition_fallback=system.table_keys["x"],
        quality_file=_QUALITY_FILE,
        quality_key=_QUALITY_KEY)


#: The trees a declared ``source_root`` can sit in: ``"raw"`` (``system.paths.raw()``) or
#: ``"farm"`` (the system's data farm, ``system.paths.data_root()``, where ``aipf modes`` writes).
SOURCE_TIERS: Tuple[str, ...] = ("raw", "farm")


def source_root(system: System) -> Path:
    """The directory every ``--source`` SUBDIR is relative to: ``defaults["training"]["source_root"]``,
    ``{"tier": one of SOURCE_TIERS, "path": relative}``."""
    spec = _declared(_training(system), "source_root", "the training sources")
    if not isinstance(spec, Mapping) or set(spec) != {"tier", "path"}:
        raise ValueError(
            f"source_root={spec!r} is not a mapping with exactly the keys "
            f"'tier' and 'path'")
    if spec["tier"] not in SOURCE_TIERS:
        raise ValueError(
            f"source_root tier={spec['tier']!r} is not one of {SOURCE_TIERS}")
    if Path(spec["path"]).is_absolute():
        raise ValueError(
            f"source_root path={spec['path']!r} is absolute; it is relative "
            f"to the declared tier")
    base = (system.paths.raw() if spec["tier"] == "raw"
            else system.paths.data_root())
    return base / spec["path"]


def declared_source_table(system: System) -> Tuple[Tuple[str, str, str, Tuple[int, ...],
                                                         Tuple[str, ...]], ...]:
    """``defaults["training"]["sources"]`` as ``(name, tree, pattern, grid, exclude_tags)`` rows in
    declared order, or a refusal that names the key."""
    declared = _training(system).get("sources")
    if not declared:
        raise KeyError(
            f"system {system.name!r} declares no defaults[{_TRAINING_KEY!r}]['sources']: "
            f"there is no declared set of training sources")
    rows = []
    for name, entry in declared.items():
        missing = sorted({"root", "pattern", "grid"} - set(entry))
        if missing:
            raise ValueError(
                f"defaults[{_TRAINING_KEY!r}]['sources'][{name!r}] declares no {missing}")
        rows.append((str(name), str(entry["root"]), str(entry["pattern"]),
                     tuple(int(g) for g in entry["grid"]),
                     tuple(str(t) for t in entry.get("exclude_tags", ()))))
    return tuple(rows)


def _specs_from(system: System,
                table: Sequence[Tuple],
                names: Sequence[str]) -> Tuple[SourceSpec, ...]:
    """:class:`SourceSpec` per named source off a ``(name, tree, pattern, grid[, exclude_tags])``
    table; the root is :func:`source_root` (an absolute tree is refused), the weight from
    ``source_loss_weights``."""
    known = {row[0]: row for row in table}
    missing = [n for n in names if n not in known]
    if missing:
        raise KeyError(
            f"no entry for source(s) {missing} in the source table; it "
            f"names {sorted(known)}")
    absolute = [row[0] for row in table if Path(row[1]).is_absolute()]
    if absolute:
        raise ValueError(
            f"source(s) {absolute} name an absolute tree; a source's tree is "
            f"relative to the declared source root {source_root(system)}")
    root = source_root(system)
    weights = system.defaults.get("source_loss_weights", {})
    return tuple(
        SourceSpec(name=name, root=str(root / known[name][1]),
                   pattern=known[name][2], grid=tuple(known[name][3]),
                   loss_weight=float(weights.get(name, 1.0)),
                   exclude_tags=tuple(known[name][4]) if len(known[name]) > 4 else ())
        for name in names)


def pre_run_checks(system: System, run_name: str,
                   init_from: Optional[Checkpoint]) -> Optional[Path]:
    """The refusals made before a run directory exists (a reserved run name; the declared starting
    weights found and digest-checked); returns the weights' path, ``None`` without ``init_from``.
    ``aipf train --pbs`` makes them before it writes or submits a job."""
    from aipf.paths import PUBLISHED_DIRNAME, TRACKED_FILE_NAME
    if run_name == PUBLISHED_DIRNAME:
        raise ReservedRunName(
            f"run_name={run_name!r} is the directory holding the published "
            f"checkpoints, and a run writes its {TRACKED_FILE_NAME} there: name the run "
            f"anything else")
    if init_from is None:
        return None
    if init_from == system.checkpoint:            # the system's own: the tracked copy first
        return system.resolve_checkpoint()
    return _resolve_weights(init_from, system.paths.raw, system)


class ReservedRunName(ValueError):
    """A run name that would write into the published checkpoints' directory."""


def _tables_on(tables, device):
    """The anchor tables with every tensor on ``device`` (the same object when they already are)."""
    moved = {f.name: value.to(device) for f in dataclasses.fields(tables)
             if isinstance(value := getattr(tables, f.name), torch.Tensor) and value.device != device}
    return dataclasses.replace(tables, **moved) if moved else tables


# -- the Lightning pieces this driver adds --
class _MultiSourceDriver(LitModule):
    """``LitModule`` fed a ``{source: batch}`` dict, each source's drift weighted by ``source_loss_weights``."""

    #: The loaded anchor tables and their energy unit, or ``None`` (drift only); set by
    #: :meth:`attach_anchors`.
    _anchors = None
    _anchor_kB = None

    #: The declared penalty probes (:class:`~aipf.train.penalties.Penalties`), or ``None``.
    _penalties = None

    #: The kernel hinge (:class:`~aipf.train.kernel_hinge.KernelHinge`), or ``None``.
    _hinge = None

    def attach_anchors(self, tables, kB: float) -> None:
        """Add the measured anchors to every training step of this run."""
        self._anchors = tables
        self._anchor_kB = float(kB)

    def attach_penalties(self, penalties) -> None:
        """Add the two hinge penalties, drawn afresh at every training step."""
        self._penalties = penalties

    def attach_hinge(self, hinge) -> None:
        """Add ``lambda_W`` times the kernel hinge to every training step."""
        self._hinge = hinge

    def on_fit_start(self) -> None:
        """Put the anchor tables on the device the Trainer put the model on."""
        super().on_fit_start()
        if self._anchors is not None:
            self._anchors = _tables_on(self._anchors, self.device)

    def _sources_of(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        if "rho_hat_states" in batch:
            return {"": batch}
        return dict(batch)

    def _drift_total(self, batch: Mapping[str, Any]):
        weights = self.config.source_loss_weights
        device = next(self.model.parameters()).device
        total = torch.zeros((), device=device)
        parts: Dict[str, torch.Tensor] = {}
        for name, sub in self._sources_of(batch).items():
            w = float(weights.get(name, 1.0))
            if w == 0.0:
                # a zero-weighted source contributes no gradient, so it is not evaluated
                continue
            one, _ = self.compute_losses({"drift": sub})
            total = total + w * one
            if name:
                parts[name] = one.detach()
        return total, parts

    def on_save_checkpoint(self, checkpoint) -> None:
        """Stamp the schema tag :func:`_load_weights_into` branches on, beside the base class's model state."""
        super().on_save_checkpoint(checkpoint)
        checkpoint["config_schema"] = NEW_CONFIG_SCHEMA_TAG

    def training_step(self, batch, batch_idx):
        total, parts = self._drift_total(batch)
        # this step's terms by canonical name, for `_StepLogger`
        step_terms = {"L_dyn": float(total.detach())}
        if self._anchors is not None or self._penalties is not None:
            # once per step, not per source: the anchors and the probes are grid-independent
            conv, gamma = (self._penalties.losses(self.model)
                           if self._penalties is not None else (None, None))
            anchors, anchor_parts = self.compute_losses(
                ({} if self._anchors is None
                 else self._anchors.batch(self.model, self._anchor_kB)),
                conv=conv, gamma=gamma)
            total = total + anchors
            for name, value in anchor_parts.items():
                self.log(f"train/{name}", value.detach(), on_step=False,
                         on_epoch=True)
                step_terms[name] = float(value.detach())
        if self._hinge is not None:
            # once per step: the hinge reads the kernel alone
            hinge = self._hinge.loss(self.model)
            total = total + self.config.lambda_W * hinge
            self.log(f"train/{HINGE_TERM}", hinge.detach(), on_step=False,
                     on_epoch=True)
            step_terms[HINGE_TERM] = float(hinge.detach())
        self.train_loss(total.detach())
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True,
                 prog_bar=False)
        for name, value in parts.items():
            self.log(f"train/drift_{name}", value, on_step=False,
                     on_epoch=True)
            step_terms[f"L_dyn/{name}"] = float(value)
        self.last_step_terms = step_terms
        return total

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        total, _ = self._drift_total(batch)
        self.val_loss(total.detach())
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=False)


#: Written process-wide by ``L.Trainer(deterministic=True)``; torch reads it once, at its first cuBLAS call.
_CUBLAS_VAR = "CUBLAS_WORKSPACE_CONFIG"


@contextlib.contextmanager
def _torch_flags_restored():
    """Put back, on exit, the torch switches and the variable a Trainer sets for the whole process."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cublas = os.environ.get(_CUBLAS_VAR)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        if cublas is None:
            os.environ.pop(_CUBLAS_VAR, None)
        else:
            os.environ[_CUBLAS_VAR] = cublas


class _StepLogger(L.Callback):
    """Every optimizer step's loss, written to ``steps.json`` on the first step, every :attr:`EVERY`
    steps and when training ends or fails (the file is rewritten whole, so a write per step would grow
    with the run). Each write goes to a temporary file that then replaces ``steps.json``, so the file
    is always whole; it exists only once a step has trained. A run killed by a signal (a walltime,
    for instance) keeps the steps up to the last write."""

    #: Steps between two writes of the file.
    EVERY = 50

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.losses: list = []
        self.terms: list = []

    def _write(self) -> None:
        if not self.losses:
            return
        partial = self.path.with_name(self.path.name + ".tmp")
        partial.write_text(json.dumps({"loss": self.losses,
                                       "terms": self.terms}, indent=1))
        os.replace(partial, self.path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                            batch_idx) -> None:
        value = outputs["loss"] if isinstance(outputs, Mapping) else outputs
        self.losses.append(float(value))
        self.terms.append(getattr(pl_module, "last_step_terms", None))
        if len(self.losses) == 1 or len(self.losses) % self.EVERY == 0:
            self._write()

    def on_train_end(self, trainer, pl_module) -> None:
        self._write()

    def on_exception(self, trainer, pl_module, exception) -> None:
        self._write()


# -- the driver --
def fit(system: System, *, run_name: str, sources: Sequence[SourceSpec],
        seed: int, resume_optimizer: bool,
        steps: Optional[int] = None, epochs: Optional[int] = None,
        window: Optional[WindowSettings] = None,
        weighting: Optional[RunWeighting] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        split_mode: Optional[str] = None,
        loader_order: Optional[str] = None,
        init_from: Optional[Checkpoint] = None,
        anchors: Union[AnchorTables, _NoAnchors, None] = None,
        root: Optional[Path] = None,
        log_every_step: bool = False,
        device: str = "auto",
        deterministic: bool = False) -> Path:
    """Train ``system``'s declared functional and return the run directory.

    ``seed`` is the one declared number every stream derives from; pass exactly one of ``steps``/``epochs``.
    ``device`` is ``auto`` (``cuda`` when torch sees one), ``cpu`` or ``cuda``; an absent ``cuda`` is
    refused before anything is written. ``deterministic=True`` asks torch for deterministic kernels
    (and the cuBLAS workspace setting) for the run, and puts the process-wide switches back after it;
    ``False`` touches none of them.
    ``resume_optimizer`` is required: ``True`` continues ``init_from``'s optimizer and schedule,
    ``False`` builds both fresh.
    ``split_mode``, ``loader_order``, ``anchors``: ``None`` = the declaration answers;
    ``NO_ANCHORS`` = drift-only ablation.

    ``system``: the declared system whose functional is trained.
    ``run_name``: the run directory's name under the farm's ``ckpt/``; ``published`` is refused.
    ``sources``: the :class:`SourceSpec` rows to train on, at least one (:func:`_specs_from` builds them).
    ``seed``: the one declared seed; the model and data streams derive from it.
    ``resume_optimizer``: continue ``init_from``'s optimizer and schedule, or build both fresh.
    ``steps``, ``epochs``: the run's length, exactly one of the two.
    ``window``: how a timeline is cut into windows; ``None`` reads ``defaults["training"]``.
    ``weighting``: the per-run drift weighting; ``None`` reads the declaration.
    ``config_overrides``: :class:`TrainConfig` fields applied over the declared ones.
    ``split_mode``: one of :data:`aipf.train.datamodule.SPLITS`, or ``None``.
    ``loader_order``: one of :data:`aipf.train.datamodule.ORDERS`, or ``None``.
    ``init_from``: the :class:`~aipf.system.Checkpoint` the weights start from, or ``None`` (fresh).
    ``anchors``: the anchor tables, :data:`NO_ANCHORS`, or ``None``.
    ``root``: the farm root the run directory is written under; ``None`` is the system's data root.
    ``log_every_step``: write every step's loss to ``steps.json``.
    ``device``: one of :data:`DEVICES`.
    ``deterministic``: ask torch for deterministic kernels for this run."""
    if (steps is None) == (epochs is None):
        raise ValueError(
            f"pass exactly one of steps= or epochs=, got steps={steps!r} and "
            f"epochs={epochs!r}. A run's length is the caller's declaration "
            f"and this driver has no default for it")
    if not sources:
        raise ValueError("sources is empty: a run trains on at least one")
    accelerator = resolve_device(device)

    from aipf.data import index
    from aipf.paths import TRACKED_FILE_NAME

    init_path = pre_run_checks(system, run_name, init_from)
    run_dir = index.ckpt_dir(system, root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(**{**_config_from_system(system), "seed": int(seed),
                         **dict(config_overrides or {})})

    # the model is built under the run's own model stream, forked from the global one and restored
    with seeded_rng(cfg.seed_for("model")):
        model = build_functional(system)
    saved = (None if init_from is None
             else _load_weights_into(model, init_from,
                                     system.paths.raw, system, path=init_path))

    lit = _MultiSourceDriver(model, model._cache, cfg)
    if resume_optimizer:
        if saved is None:
            raise ValueError(
                "resume_optimizer=True needs init_from: there is no saved "
                "optimizer to continue without a checkpoint to read it from")
        lit.resume_optimizer_from(*_optimizer_state_of(saved, init_from))
    anchors = _anchors_from_system(system, anchors)
    if anchors is not None:
        lit.attach_anchors(anchors, float(_declared_constant(system, "kB")))
    penalties = penalties_from_system(system, cfg)
    if penalties is not None:
        lit.attach_penalties(penalties)
    hinge = KernelHinge.from_config(cfg)
    if hinge is not None:
        hinge.check(model)
        lit.attach_hinge(hinge)
    window = window or _window_from_system(system)
    weighting = weighting or _weighting_from_system(system,
                                                     lit.resolved_k_max())
    dm = ModeDataModule(sources=tuple(sources), keys=_archive_keys(system),
                        window=window,
                        split=_split_from_system(system, split_mode),
                        weighting=weighting,
                        loader=_loader_from_system(system, loader_order))
    dm.setup()
    data_rng = torch.Generator().manual_seed(cfg.seed_for("data"))
    train_loader = dm.train_dataloader(generator=data_rng)
    val_loader = dm.val_dataloader(generator=data_rng)

    callbacks = [_StepLogger(run_dir / "steps.json")] if log_every_step else []
    final = run_dir / TRACKED_FILE_NAME
    with _torch_flags_restored():
        trainer = L.Trainer(
            max_steps=steps if steps is not None else -1,
            max_epochs=epochs if epochs is not None else -1,
            default_root_dir=str(run_dir),
            # off: nothing here reads the file Lightning would link
            enable_checkpointing=False,
            logger=False, enable_progress_bar=False,
            accelerator=accelerator, devices=1,
            # opt-in: None leaves torch's switches as they are
            deterministic=True if deterministic else None,
            gradient_clip_val=_training(system).get("grad_clip"),
            # no validation split: Lightning is told there is no validation loop
            **({} if val_loader is not None else {"limit_val_batches": 0}),
            callbacks=callbacks)
        trainer.fit(lit, train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
        # Written HERE, by this call, and never by `save_last`.
        trainer.save_checkpoint(str(final))
    trained, silent = _terms(cfg, () if anchors is None else anchors.terms(),
                             () if penalties is None else penalties.terms(),
                             () if hinge is None else (HINGE_TERM,))
    written = _plain(cfg.as_dict())
    if silent:
        # the declaration records on its face that it was not fully honoured
        written["warnings"] = {"declared_weights_without_data":
                               sorted(silent)}
    (run_dir / "hparams.yaml").write_text(
        yaml.safe_dump(written, sort_keys=True))

    if silent:
        _LOG.warning(
            "run %s: %s declared with a non-zero weight and not trained; "
            "see UNTRAINED_TERMS.txt in the run directory",
            run_name, sorted(silent))
        (run_dir / "UNTRAINED_TERMS.txt").write_text(
            "These loss terms carry a non-zero weight in this run's own\n"
            "declaration and were NOT trained: this driver had no data to\n"
            "feed them. The run is what it is; this file is so that says so\n"
            "where the run directory is read, and not only where its stderr\n"
            "went.\n\n"
            + "".join(f"{name}\tweight {weight!r}\n"
                      for name, weight in sorted(silent.items()))
            + f"\nrun: {run_name}\nsystem: {system.name}\n")
    (run_dir / "MANIFEST.json").write_text(json.dumps({
        "system": system.name,
        "run": run_name,
        "seed": int(seed),
        "steps": steps,
        "epochs": epochs,
        "global_step": int(trainer.global_step),
        "final_md5": hashlib.md5(final.read_bytes()).hexdigest(),
        "init_from": (None if init_from is None
                      else dataclasses.asdict(init_from)),
        "sources": [dataclasses.asdict(s) for s in sources],
        "terms_trained": trained,
        "declared_weights_without_data": silent,
        "resume_optimizer": bool(resume_optimizer),
        "device": accelerator,
        "deterministic": bool(deterministic),
        # what the run walked, not what the system declares
        "split_mode": dm.split.mode,
        "loader_order": dm.loader.order,
        "anchor_tables": (None if anchors is None
                          else {"rows": anchors.rows(),
                                "provenance": _plain(anchors.provenance)}),
        "penalties": (None if penalties is None
                      else _plain(penalties.provenance())),
        "kernel_hinge": None if hinge is None else hinge.provenance(),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1, sort_keys=True))
    return run_dir


def _resolve_weights(declared: Checkpoint, raw, system: System | None = None) -> Path:
    """The declared file, digest-checked: ``Checkpoint.resolve`` for ``system`` (its published copy first),
    else under ``raw``."""
    return (declared.verify_under(raw) if system is None
            else declared.resolve(system.name, raw, system.paths.published,
                                  system.variant_name))


def _load_weights_into(model, declared: Checkpoint, raw,
                       system: System | None = None, path: Path | None = None) -> None:
    """Verify ``declared`` (``Checkpoint.resolve`` for ``system``, its published copy first; else under
    ``raw``), then branch on the schema tag: this package's ``config_schema`` loads ``model_state_dict``
    strictly, a k-modes checkpoint through ``load_kmodes_into``, any other ``state_dict`` through
    ``load_lightning_hparams_into``; an unknown ``config_schema`` tag is refused by name."""
    if path is None:
        path = _resolve_weights(declared, raw, system)
    state = torch.load(path, map_location="cpu", weights_only=False)
    tag = state.get("config_schema")
    if tag is not None and tag not in CONFIG_SCHEMA_TAGS:
        raise ValueError(
            f"{path} declares config_schema={tag!r}, which this driver does not read; it reads "
            f"{list(CONFIG_SCHEMA_TAGS)} and the untagged Lightning-hparams and k-modes layouts")
    if tag in CONFIG_SCHEMA_TAGS:
        model.load_state_dict(state["model_state_dict"])
        return state
    if _is_kmodes(state):
        load_kmodes_into(model, state)
        return state
    if "state_dict" in state:
        load_lightning_hparams_into(model, state["state_dict"])
        return state
    raise KeyError(
        f"{path} carries neither {NEW_CONFIG_SCHEMA_TAG!r} under "
        f"'config_schema' (a checkpoint this driver wrote) nor a "
        f"'state_dict' (a Lightning-hparams checkpoint), "
        f"so there is no named way to read its weights")


def _is_kmodes(state: Mapping[str, Any]) -> bool:
    """A k-modes checkpoint: every model tensor has a k-modes rename rule."""
    if "state_dict" not in state and "model_state_dict" not in state:
        return False
    try:
        kmodes_model_state_dict(state)
    except (KeyError, TypeError):
        return False
    return True


#: The four terms an ``AnchorTables`` feeds, in :data:`aipf.losses.names.CANONICAL_LOSSES` order.
ANCHOR_TERMS: Tuple[str, ...] = ("L_M", "L_S", "L_bulk", "L_P")


def _declared_constant(system: System, name: str) -> Any:
    """One of this system's constants, or a refusal that names it."""
    if name not in system.constants:
        raise KeyError(
            f"system {system.name!r} declares no constants[{name!r}], which "
            f"the anchor tier needs to express a measured target in the "
            f"energy unit its model works in")
    return system.constants[name]


def _optimizer_state_of(state: Mapping[str, Any], declared: Checkpoint):
    """``(optimizer_states[0], lr_schedulers[0])`` of a loaded checkpoint; refuses one carrying neither."""
    for key in ("optimizer_states", "lr_schedulers"):
        entries = state.get(key)
        if not entries:
            raise KeyError(
                f"{declared.path or 'the published checkpoint'} carries no {key!r}, so there is no saved "
                f"optimizer to resume. Pass resume_optimizer=False to start "
                f"a fresh one from these weights")
    return state["optimizer_states"][0], state["lr_schedulers"][0]


def _terms(cfg: TrainConfig, anchored: Sequence[str],
           penalised: Sequence[str] = (),
           hinged: Sequence[str] = ()) -> Tuple[list, dict]:
    """``(trained, declared weights without data)``; an anchor, penalty or hinge term counts as trained
    only at non-zero weight; ``anchored``, ``penalised`` and ``hinged`` name the terms the tables, the
    probes and the kernel hinge fed."""
    from aipf.losses.names import CANONICAL_LOSSES

    trained = ["L_dyn"]
    weights = {
        "L_M": cfg.lambda_M, "L_S": cfg.lambda_S, "L_bulk": cfg.lambda_bulk,
        "L_P": cfg.lambda_P, "L_conv": cfg.lambda_conv,
        "L_Gamma": cfg.lambda_gamma,
        HINGE_TERM: cfg.lambda_W,
    }
    fed = set(anchored) | set(penalised) | set(hinged)
    # every canonical term must be classified here, zero-weighted ones included
    unclassified = set(CANONICAL_LOSSES) - set(trained) - set(weights)
    if unclassified:
        raise NotImplementedError(
            f"canonical loss term(s) {sorted(unclassified)} are neither "
            f"trained by this driver nor listed as lacking data: classify "
            f"them here rather than letting a run claim it trained them")
    trained += [name for name in ANCHOR_TERMS + PENALTY_TERMS + (HINGE_TERM,)
                if name in fed and float(weights[name]) != 0.0]
    return (trained,
            {name: float(w) for name, w in weights.items()
             if float(w) != 0.0 and name not in fed})


def _plain(value: Any) -> Any:
    """A nested structure YAML can write: mappings, lists, scalars."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
