"""Sources, the split between them, per-run weights, and the loaders.

A source is a tree, a glob, a grid and a per-step loss weight; sources stream in lockstep, one batch
each per step, keyed by name. Both loader methods require ``generator``, and each source loader gets
its own child generator, so the batch order is a pure function of the seed. Nothing here is defaulted."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import logging

import numpy as np
import torch
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader, Sampler

from aipf.train.dataset import (ArchiveKeys, ModeRun, ModeWindowDataset, band_keep,
                                WindowSettings, read_mode_runs, scatter_modes,
                                weak_target)

TWO_PI = 2.0 * np.pi

#: Train/validation split, once per source: ``"labels"`` (hold out declared composition labels),
#: ``"random"`` (seeded permutation of sorted tags, at least one run held out), ``"none"`` (no split).
SPLITS = ("labels", "random", "none")

#: Training order: ``"shuffled"`` (fresh permutation per epoch from the caller's generator) or
#: ``"index_stepping"`` (see :class:`IndexSteppingSampler`; reproduces a recorded trajectory).
ORDERS = ("shuffled", "index_stepping")

#: Run weighting: ``"uniform"`` (all one) or ``"inverse_band_power"`` (``1 / (s + median(s))``,
#: ``s`` = the run's in-band filtered ``1/k^2``-weighted mean square target, renormalised to mean one).
RUN_WEIGHTINGS = ("uniform", "inverse_band_power")


@dataclass(frozen=True)
class SourceSpec:
    """One training source: where to read, what to match, on what grid."""

    name: str
    root: str
    pattern: str
    grid: Tuple[int, int, int]
    loss_weight: float
    exclude_tags: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", tuple(int(g) for g in self.grid))
        object.__setattr__(self, "exclude_tags", tuple(self.exclude_tags))
        if not self.name:
            raise ValueError("name is empty: a source is keyed by its name")


@dataclass(frozen=True)
class SplitSettings:
    """How the runs of one source are divided into train and validation."""

    mode: str
    val_labels: Tuple[Tuple[float, ...], ...]
    val_fraction: float
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val_labels",
                           tuple(tuple(float(v) for v in label)
                                 for label in self.val_labels))
        if self.mode not in SPLITS:
            raise ValueError(f"mode={self.mode!r} is not one of {SPLITS}")
        if not 0.0 <= float(self.val_fraction) <= 1.0:
            raise ValueError(
                f"val_fraction={self.val_fraction!r} is not in [0, 1]")


@dataclass(frozen=True)
class RunWeighting:
    """How much of the drift loss each run carries; the band knobs belong to ``"inverse_band_power"`` only."""

    mode: str
    sigma: Optional[float]
    k_max: Optional[float]
    eps: Optional[float]
    probe_every: Optional[int]

    _BAND_FIELDS = ("sigma", "k_max", "eps", "probe_every")

    def __post_init__(self) -> None:
        if self.mode not in RUN_WEIGHTINGS:
            raise ValueError(
                f"mode={self.mode!r} is not one of {RUN_WEIGHTINGS}")
        for name in self._BAND_FIELDS:
            value = getattr(self, name)
            if self.mode == "inverse_band_power" and value is None:
                raise ValueError(
                    f"mode='inverse_band_power' needs {name}: the band it "
                    f"measures over is a declared choice with no value in "
                    f"this package")
            if self.mode == "uniform" and value is not None:
                raise ValueError(
                    f"mode='uniform' has no use for {name}={value!r}: "
                    f"accepting it would record a band that was never "
                    f"measured")
        if self.mode == "inverse_band_power" and int(self.probe_every) < 1:
            raise ValueError(
                f"probe_every={self.probe_every!r} must be at least 1")


@dataclass(frozen=True)
class LoaderSettings:
    """The loader shape; every field is the caller's. ``order`` is one of ``'shuffled'``,
    ``'index_stepping'``."""

    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    order: str

    def __post_init__(self) -> None:
        if int(self.batch_size) < 1:
            raise ValueError(
                f"batch_size={self.batch_size!r} must be at least 1")
        if int(self.num_workers) < 0:
            raise ValueError(
                f"num_workers={self.num_workers!r} cannot be negative")
        if self.order not in ORDERS:
            raise ValueError(
                f"order must be one of 'shuffled', 'index_stepping'; got "
                f"{self.order!r}")


def split_runs(runs: Sequence[ModeRun],
               settings: SplitSettings) -> Tuple[list, list]:
    """One source's runs as ``(train, val)``; deterministic, never reads the global RNG.

    ``"none"`` returns the runs in read order, unsorted."""
    if settings.mode == "labels":
        if any(not run.composition for run in runs):
            raise ValueError(
                "mode='labels' holds out runs by their composition label and "
                "at least one run carries none: declare the composition keys "
                "on the archive, or split by 'random'")
        wanted = set(settings.val_labels)
        val = [r for r in runs if r.composition in wanted]
        train = [r for r in runs if r.composition not in wanted]
        return train, val
    if settings.mode == "none":
        return list(runs), []
    if len(runs) < 2:
        return list(runs), []
    ordered = sorted(runs, key=lambda r: r.tag)
    order = np.random.default_rng(settings.seed).permutation(len(ordered))
    n_val = max(1, int(round(float(settings.val_fraction) * len(ordered))))
    held = set(order[:n_val].tolist())
    train = [r for i, r in enumerate(ordered) if i not in held]
    val = [r for i, r in enumerate(ordered) if i in held]
    return train, val


def band_multiplicity(n_half: int, n_full: int) -> np.ndarray:
    """Full-transform modes per half-spectrum index: 2, except 1 on the zero and (even axis) Nyquist
    planes."""
    mult = np.full(int(n_half), 2.0)
    mult[0] = 1.0
    if int(n_full) % 2 == 0:
        mult[-1] = 1.0
    return mult


def _band_mean_square(target_hat: np.ndarray, box: np.ndarray, n_full: int,
                      sigma: float, k_max: float, eps: float) -> float:
    """In-band mean of ``|target_hat * exp(-k^2 sigma^2 / 2)|^2 * mult / (k^2 + eps)`` for one target."""
    Gx, Gy, Gzr = target_hat.shape[-3:]
    nx = np.fft.fftfreq(Gx) * Gx
    ny = np.fft.fftfreq(Gy) * Gy
    nz = np.arange(Gzr)
    kx = TWO_PI * nx[:, None, None] / box[0]
    ky = TWO_PI * ny[None, :, None] / box[1]
    kz = TWO_PI * nz[None, None, :] / box[2]
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    band = (k2 > 0) & (k2 <= k_max ** 2)
    mult = band_multiplicity(Gzr, n_full)[None, None, :]
    filt = np.exp(-k2 * sigma ** 2 / 2.0)
    value = np.abs(target_hat * filt) ** 2 * mult / (k2 + eps)
    return value[..., band].sum() / band.sum()


def run_weights(runs: Sequence[ModeRun], weighting: RunWeighting,
                window: WindowSettings) -> np.ndarray:
    """A weight per run, in run order, averaging one over the runs that have a window; a windowless
    run gets 0."""
    if weighting.mode == "uniform":
        return np.ones(len(runs), np.float32)
    statistics = []
    for run in runs:
        box = run.boxes if run.boxes.ndim == 1 else run.boxes.mean(axis=0)
        volume = float(box.prod())
        keep = band_keep(run.labels, run.boxes, window.band_k_max)
        labels = run.labels if keep is None else run.labels[keep]
        values = []
        for centre in range(window.half_width - 1,
                            run.n_frames - window.half_width,
                            int(weighting.probe_every)):
            target = weak_target(run.amplitudes, centre, window.half_width,
                                 run.frame_interval)
            target = scatter_modes(target if keep is None else target[..., keep],
                                   labels, volume, window.grid)
            values.append(_band_mean_square(
                target, box, window.grid[2], weighting.sigma,
                weighting.k_max, weighting.eps))
        statistics.append(np.mean(values) if values else np.nan)
    statistics = np.asarray(statistics)
    ok = np.isfinite(statistics)
    if not ok.any():
        raise RuntimeError(
            "no run has a single window the estimator can cut: there is "
            "nothing to weight by")
    if not ok.all():
        logging.warning(
            "run weighting: %d run(s) have no window and get weight 0: %s",
            int((~ok).sum()),
            [r.tag for r, good in zip(runs, ok) if not good])
    floor = np.median(statistics[ok])
    weights = np.zeros(len(runs))
    weights[ok] = 1.0 / (statistics[ok] + floor)
    weights[ok] /= weights[ok].mean()
    return weights.astype(np.float32)


class IndexSteppingSampler(Sampler):
    """Batches of the ``"index_stepping"`` order; reads no RNG.

    Batch ``n`` is ``[(i * stride + n) % n_samples for i in range(batch_size)]``,
    ``stride = max(1, n_samples // batch_size)``."""

    def __init__(self, n_samples: int, batch_size: int) -> None:
        if int(n_samples) < 1:
            raise ValueError(
                f"n_samples={n_samples!r}: an empty dataset has no batches")
        if int(batch_size) < 1:
            raise ValueError(
                f"batch_size={batch_size!r} must be at least 1")
        self.n_samples = int(n_samples)
        self.batch_size = int(batch_size)
        self.stride = max(1, self.n_samples // self.batch_size)

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self):
        for n in range(self.n_samples):
            yield [(i * self.stride + n) % self.n_samples
                   for i in range(self.batch_size)]


class SourceState:
    """One source after :meth:`ModeDataModule.setup`: its runs and datasets."""

    def __init__(self, spec: SourceSpec, window: WindowSettings,
                 train_runs: list, val_runs: list,
                 train_dataset: ModeWindowDataset,
                 val_dataset: Optional[ModeWindowDataset]) -> None:
        self.spec = spec
        self.name = spec.name
        self.window = window
        self.train_runs = train_runs
        self.val_runs = val_runs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset


class ModeDataModule:
    """Several :class:`SourceSpec` sources (distinct names), split, weighted, and served as batches.

    ``window.grid`` is replaced per source; only TRAIN runs are weighted."""

    def __init__(self, *, sources: Sequence[SourceSpec], keys: ArchiveKeys,
                 window: WindowSettings, split: SplitSettings,
                 weighting: RunWeighting, loader: LoaderSettings) -> None:
        sources = tuple(sources)
        if not sources:
            raise ValueError("sources is empty: a run has at least one")
        names = [s.name for s in sources]
        if len(set(names)) != len(names):
            raise ValueError(
                f"source name(s) repeat in {names}: a batch is keyed by "
                f"name, so two sources sharing one would hide each other")
        self.specs = sources
        self.keys = keys
        self.window = window
        self.split = split
        self.weighting = weighting
        self.loader = loader
        #: Available before :meth:`setup`: the consuming module is built first.
        self.source_loss_weights: Dict[str, float] = {
            s.name: float(s.loss_weight) for s in sources}
        self.source_names: Tuple[str, ...] = tuple(names)
        self.sources: Optional[list] = None

    def setup(self) -> None:
        """Read every source, split it, weight it and build its datasets."""
        built = []
        for spec in self.specs:
            runs = read_mode_runs(spec.root, spec.pattern, keys=self.keys,
                                  exclude_tags=spec.exclude_tags)
            if spec.exclude_tags:
                logging.info("source %s: %d run(s) named for exclusion, "
                             "%d read", spec.name, len(spec.exclude_tags),
                             len(runs))
            train_runs, val_runs = split_runs(runs, self.split)
            if not train_runs:
                raise RuntimeError(
                    f"source {spec.name!r}: the train split is empty. Every "
                    f"run matching {spec.pattern!r} was held out or excluded")
            window = WindowSettings(
                estimator=self.window.estimator,
                half_width=self.window.half_width,
                n_states=self.window.n_states,
                stride=self.window.stride,
                grid=spec.grid,
                savgol_window=self.window.savgol_window,
                savgol_poly=self.window.savgol_poly,
                band_k_max=self.window.band_k_max)
            weights = run_weights(train_runs, self.weighting, window)
            train_dataset = ModeWindowDataset(train_runs, window,
                                              run_weights=weights)
            val_dataset = (ModeWindowDataset(val_runs, window)
                           if val_runs else None)
            logging.info(
                "source %s: %d train / %d validation runs, %d train windows",
                spec.name, len(train_runs), len(val_runs), len(train_dataset))
            built.append(SourceState(spec, window, train_runs, val_runs,
                                     train_dataset, val_dataset))
        self.sources = built

    def _ready(self) -> list:
        if self.sources is None:
            raise RuntimeError(
                "setup() has not run: the sources have not been read yet")
        return self.sources

    @staticmethod
    def _child_generators(n: int, generator: torch.Generator) -> list:
        """One generator per loader, each seeded in turn from the caller's."""
        seeds = torch.randint(0, 2 ** 62, (n,), generator=generator,
                              dtype=torch.int64).tolist()
        return [torch.Generator().manual_seed(seed) for seed in seeds]

    def _loader(self, dataset: ModeWindowDataset, *, shuffle: bool,
                generator: torch.Generator) -> DataLoader:
        """One source's loader; ``shuffle`` marks the training loader, the only one
        ``LoaderSettings.order`` governs."""
        shared = dict(num_workers=self.loader.num_workers,
                      pin_memory=self.loader.pin_memory,
                      generator=generator)
        if shuffle and self.loader.order == "index_stepping":
            return DataLoader(
                dataset,
                batch_sampler=IndexSteppingSampler(len(dataset),
                                                   self.loader.batch_size),
                **shared)
        return DataLoader(dataset, batch_size=self.loader.batch_size,
                          shuffle=shuffle,
                          drop_last=self.loader.drop_last, **shared)

    def train_dataloader(self, *, generator: torch.Generator):
        """One batch per source per step, keyed by name (``max_size_cycle``); ``generator`` is required."""
        sources = self._ready()
        children = self._child_generators(len(sources), generator)
        loaders = {s.name: self._loader(s.train_dataset, shuffle=True,
                                        generator=child)
                   for s, child in zip(sources, children)}
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self, *, generator: torch.Generator):
        """One validation loader per source that has a split, or ``None``; ``generator`` is required."""
        sources = [s for s in self._ready() if s.val_dataset is not None]
        children = self._child_generators(len(sources), generator)
        loaders = [self._loader(s.val_dataset, shuffle=False,
                                generator=child)
                   for s, child in zip(sources, children)]
        return loaders or None
