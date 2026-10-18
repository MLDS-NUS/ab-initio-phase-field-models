"""The memory-mapped sidecar of a mode archive: the writer, the converter and the second reader.

What is pinned (docs/reference/training.md, "Memory-mapped amplitudes"):

* every batch a loader serves, every key of every sample, is bit-identical between the reader that
  holds the amplitudes in memory and the one that maps them from ``rho_k.npy``, with 0 and 4 loader
  workers, on synthetic three-channel runs (a breathing box, a reference cell, a quality cut) and on
  the Fe-B sample archives; a mapped run also pickles as its file, not its frames;
* ``setup()`` of several mapped runs allocates no amplitudes: its traced Python allocation peak and its
  resident-set growth stay below one run's ``rho_k``, where the eager reader's peak is all of them;
* a run without the sidecar reads as before (with a warning when the sidecar is declared), a source may
  mix the two, an undeclared sidecar is ignored, and a sidecar of the wrong shape, frame count, channel
  count, dtype or order, an empty one, one without its record and one copied from another ``rho_k`` of
  the same shape are refused naming the run; the soft limit on open files is raised for the mappings;
* the converter writes the sidecar as ``moveaxis(rho_k, 2, 1)`` bit for bit, skips a valid one,
  rewrites a stale one, reports a damaged archive and goes on, lists (and on request removes) the
  temporary files of killed writers, and never touches ``modes.npz``;
* ``ModesRecord.save`` without the flag writes the archive it always wrote, byte for byte (pinned by its
  sha256), and with it the same archive, the sidecar and its record.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import torch

import aipf
from aipf.pipeline import modes as modes_mod
from aipf.pipeline.modes import (MMAP_FILE, MMAP_RECORD, ModesRecord, archive_fingerprint,
                                 write_mmap_sidecar)
from aipf.train.datamodule import (LoaderSettings, ModeDataModule, RunWeighting, SourceSpec,
                                   SplitSettings)
from aipf.train.dataset import ArchiveKeys, WindowSettings, read_mode_runs
from aipf.train.mmap_sidecar import (FAILED, REPLACED, SKIPPED, WRITTEN, convert_run, convert_runs,
                                     leftover_partials)

#: The three-channel archive these tests write, as ``aipf.train.fit`` declares the package's own.
KEYS = ArchiveKeys(
    file_name="modes.npz", amplitudes="rho_k", amplitudes_channel_axis=2, labels="nvec",
    box="box", temperature="T_K", frame_interval="dt_frame_ps",
    composition=("x_a", "x_b", "x_c"), composition_fallback=None,
    quality_file="DATA_QUALITY_WARNING.json", quality_key="valid_frames",
    reference_box="reference_box")
MAPPED = KEYS.replace(mmap_file=MMAP_FILE)

GRID = (8, 8, 8)

#: The tracked Fe-B sample: two runs at 0 GPa, 201 frames of 75 modes in two channels.
FEB_SAMPLE = (Path(aipf.__file__).resolve().parents[2]
              / "data" / "feb" / "sample" / "modes" / "modes_0GPa_v2")


def _labels(reach: int = 2) -> np.ndarray:
    axes = [np.arange(-reach, reach + 1)] * 3
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.int16)


def _record(*, n_frames: int, seed: int, n_channels: int = 3, reach: int = 2,
            reference=None, breathing: bool = True) -> ModesRecord:
    rng = np.random.default_rng(seed)
    labels = _labels(reach)
    shape = (n_frames, labels.shape[0], n_channels)
    amplitudes = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    boxes = np.full((n_frames, 3), 9.0)
    if breathing:
        boxes = boxes + 0.01 * rng.standard_normal((n_frames, 3))
    fractions = rng.dirichlet(np.ones(n_channels))
    label = {f"x_{c}": float(x) for c, x in zip("abc", fractions)}
    return ModesRecord(amplitudes=amplitudes, labels=labels, boxes=boxes, T=1500.0 + 10 * seed,
                       dt_frame=0.02 + 0.001 * seed, provenance={"composition": label},
                       reference_box=reference)


def _three_channel_tree(root: Path, *, sidecar: bool = True) -> Path:
    """Four three-channel runs: breathing boxes, one in a reference cell, one cut by the quality
    sidecar to its first 45 frames."""
    root.mkdir(parents=True, exist_ok=True)
    _record(n_frames=60, seed=0).save(root / "run_a", mmap_sidecar=sidecar)
    _record(n_frames=72, seed=1, reference=np.array([9.0, 9.1, 9.2])).save(
        root / "run_b", mmap_sidecar=sidecar)
    _record(n_frames=60, seed=2).save(root / "run_c", mmap_sidecar=sidecar)
    (root / "run_c" / KEYS.quality_file).write_text(json.dumps({KEYS.quality_key: "0..44"}))
    _record(n_frames=66, seed=3, breathing=False).save(root / "run_d", mmap_sidecar=sidecar)
    return root


def _window(estimator: str, grid=GRID, **overrides) -> WindowSettings:
    kwargs = dict(estimator=estimator, half_width=5, n_states=3, stride=4, grid=grid,
                  savgol_window=None, savgol_poly=None)
    if estimator == "savgol":
        kwargs.update(savgol_window=11, savgol_poly=3)
    kwargs.update(overrides)
    return WindowSettings(**kwargs)


def _module(root: Path, pattern: str, keys: ArchiveKeys, window: WindowSettings, *,
            num_workers: int, weighting: str = "inverse_band_power") -> ModeDataModule:
    band = (dict(sigma=1.0, k_max=1.5, eps=1e-6, probe_every=3)
            if weighting == "inverse_band_power" else
            dict(sigma=None, k_max=None, eps=None, probe_every=None))
    return ModeDataModule(
        sources=(SourceSpec(name="s", root=str(root), pattern=pattern, grid=window.grid,
                            loss_weight=1.0, exclude_tags=()),),
        keys=keys, window=window,
        split=SplitSettings(mode="random", val_labels=(), val_fraction=0.25, seed=316),
        weighting=RunWeighting(mode=weighting, **band),
        loader=LoaderSettings(batch_size=3, num_workers=num_workers, pin_memory=False,
                              drop_last=False, order="shuffled"))


def _served(dm: ModeDataModule) -> list:
    """Every batch of one training epoch and of every validation loader, in order."""
    batches = [batch["s"] for batch, _i, _l
               in dm.train_dataloader(generator=torch.Generator().manual_seed(11))]
    for loader in dm.val_dataloader(generator=torch.Generator().manual_seed(12)) or ():
        batches += list(loader)
    return batches


def _assert_bit_identical(eager: list, mapped: list) -> None:
    assert len(eager) == len(mapped) > 0
    for a, b in zip(eager, mapped):
        assert sorted(a) == sorted(b)
        for key in a:
            assert a[key].dtype == b[key].dtype, key
            assert _same_bits(a[key].numpy(), b[key].numpy()), key


def _same_bits(a, b) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and \
        np.array_equal(a.view(np.uint8), b.view(np.uint8))


# ---------------------------------------------------------------------------
# 1. bit-identical batches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_workers", [0, 4])
@pytest.mark.parametrize("estimator,band", [("weak", None), ("weak_mid", 2.2), ("savgol", None)])
def test_the_mapped_reader_serves_the_eager_readers_batches_bit_for_bit(tmp_path, estimator,
                                                                        band, num_workers):
    tree = _three_channel_tree(tmp_path / "three")
    window = _window(estimator, band_k_max=band)
    eager = _module(tree, "run_*", KEYS, window, num_workers=num_workers)
    mapped = _module(tree, "run_*", MAPPED, window, num_workers=num_workers)
    eager.setup()
    mapped.setup()
    runs = mapped.sources[0].train_runs + mapped.sources[0].val_runs
    assert all(isinstance(r.amplitudes, np.memmap) for r in runs)
    assert not any(isinstance(r.amplitudes, np.memmap)
                   for r in eager.sources[0].train_runs + eager.sources[0].val_runs)
    cut = next(r for r in runs if r.tag == "run_c")
    assert cut.n_frames == 45 and isinstance(cut.amplitudes, np.memmap)
    assert np.array_equal(eager.sources[0].train_dataset.run_weights,
                          mapped.sources[0].train_dataset.run_weights)
    _assert_bit_identical(_served(eager), _served(mapped))


@pytest.mark.skipif(not FEB_SAMPLE.is_dir(), reason="the tracked Fe-B sample is not in this checkout")
@pytest.mark.parametrize("num_workers", [0, 4])
def test_the_feb_sample_serves_the_same_batches_mapped_or_in_memory(tmp_path, num_workers):
    """The Fe-B sample as its system declares its keys: converted, then read both ways, at its own
    weak_mid estimator on a window short enough to cut many centres from its 201 frames."""
    from aipf.system import load
    from aipf.train.fit import _archive_keys

    tree = tmp_path / "modes_0GPa_v2"
    shutil.copytree(FEB_SAMPLE, tree)
    keys = _archive_keys(load("feb"))
    assert keys.mmap_file is None          # Fe-B declares no sidecar: it reads as it always did
    statuses = [done.status for done in convert_runs(tree, "cube_*",
                                                     keys=keys.replace(mmap_file=MMAP_FILE))]
    assert statuses == [WRITTEN, WRITTEN]
    window = _window("weak_mid", grid=(32, 32, 32), half_width=10, n_states=5, stride=9)
    eager = _module(tree, "cube_*", keys, window, num_workers=num_workers)
    mapped = _module(tree, "cube_*", keys.replace(mmap_file=MMAP_FILE), window,
                     num_workers=num_workers)
    eager.setup()
    mapped.setup()
    assert all(isinstance(r.amplitudes, np.memmap) for r in mapped.sources[0].train_runs)
    _assert_bit_identical(_served(eager), _served(mapped))


def test_a_mapped_run_pickles_as_its_file_and_an_eager_one_as_it_always_did(tmp_path):
    """A loader worker started by spawn unpickles the dataset: a mapped run carries where its frames
    lie (the prefix cut included), not the frames, and reopens the same view."""
    tree = _three_channel_tree(tmp_path / "three")
    eager = {r.tag: r for r in read_mode_runs(tree, "run_*", keys=KEYS)}
    for run in read_mode_runs(tree, "run_*", keys=MAPPED):
        blob = pickle.dumps(run)
        assert len(blob) < run.amplitudes.nbytes / 4
        back = pickle.loads(blob)
        assert isinstance(back.amplitudes, np.memmap)
        assert _same_bits(back.amplitudes, eager[run.tag].amplitudes)
        assert back.tag == run.tag and _same_bits(back.boxes, run.boxes)
        same = eager[run.tag]
        assert pickle.dumps(pickle.loads(pickle.dumps(same))) == pickle.dumps(same)
        assert _same_bits(pickle.loads(pickle.dumps(same)).amplitudes, same.amplitudes)


def test_a_spawned_loader_serves_the_mapped_runs(tmp_path):
    tree = _three_channel_tree(tmp_path / "three")
    window = _window("weak")
    eager = _module(tree, "run_*", KEYS, window, num_workers=0)
    mapped = _module(tree, "run_*", MAPPED, window, num_workers=0)
    eager.setup()
    mapped.setup()
    dataset = mapped.sources[0].train_dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=2,
                                         multiprocessing_context="spawn")
    reference = torch.utils.data.DataLoader(eager.sources[0].train_dataset, batch_size=3,
                                            shuffle=False)
    _assert_bit_identical(list(reference), list(loader))


# ---------------------------------------------------------------------------
# 2. memory: setup holds metadata, not amplitudes
# ---------------------------------------------------------------------------

def _resident_bytes() -> int:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


@pytest.mark.skipif(not Path("/proc/self/statm").is_file(), reason="reads /proc/self/statm")
def test_setup_of_mapped_runs_allocates_less_than_one_runs_amplitudes(tmp_path):
    """Four runs of 16 MB of ``rho_k`` each (500 frames, 2,000 modes, two channels). The bound: the
    traced allocation peak of ``setup()`` and its resident-set growth are each below ONE run's
    ``rho_k``. The eager reader's peak, the control, is above the whole tree's."""
    tree = tmp_path / "big"
    rng = np.random.default_rng(5)
    labels = rng.integers(-3, 4, size=(2000, 3)).astype(np.int16)
    n_frames, n_channels = 500, 2
    for i in range(4):
        shape = (n_frames, labels.shape[0], n_channels)
        amplitudes = (rng.standard_normal(shape, dtype=np.float32)
                      + 1j * rng.standard_normal(shape, dtype=np.float32)).astype(np.complex64)
        run = tree / f"run_{i}"
        run.mkdir(parents=True)
        np.savez(run / "modes.npz", rho_k=amplitudes, nvec=labels,
                 box=np.full((n_frames, 3), 20.0), T_K=1500.0, dt_frame_ps=0.1,
                 x_a=0.5, x_b=0.5, x_c=0.0)
        write_mmap_sidecar(run / MMAP_FILE, amplitudes,
                           fingerprint=archive_fingerprint(run / "modes.npz"))
        del amplitudes
    one_run = n_frames * labels.shape[0] * n_channels * 8
    assert one_run == 16_000_000
    window = _window("weak", grid=(8, 8, 8))

    def measured(keys):
        dm = _module(tree, "run_*", keys, window, num_workers=0, weighting="uniform")
        before = _resident_bytes()
        tracemalloc.start()
        try:
            dm.setup()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak, _resident_bytes() - before, dm

    peak, grown, dm = measured(MAPPED)
    assert all(isinstance(r.amplitudes, np.memmap) for r in dm.sources[0].train_runs)
    assert peak < one_run, f"setup traced a peak of {peak} bytes, one run's rho_k is {one_run}"
    assert grown < one_run, f"setup grew the resident set by {grown} bytes, one run's rho_k is {one_run}"
    del dm
    eager_peak, _, _ = measured(KEYS)
    assert eager_peak > 3 * one_run


# ---------------------------------------------------------------------------
# 3. fallbacks, mixed trees and refusals
# ---------------------------------------------------------------------------

def test_a_declared_sidecar_that_is_absent_reads_the_run_into_memory(tmp_path):
    tree = _three_channel_tree(tmp_path / "three", sidecar=False)
    eager = read_mode_runs(tree, "run_*", keys=KEYS)
    mapped = read_mode_runs(tree, "run_*", keys=MAPPED)
    for a, b in zip(eager, mapped):
        assert type(b.amplitudes) is np.ndarray and _same_bits(a.amplitudes, b.amplitudes)


def test_an_undeclared_sidecar_is_never_opened(tmp_path):
    tree = _three_channel_tree(tmp_path / "three")
    (tree / "run_a" / MMAP_FILE).write_bytes(b"not an array")
    runs = read_mode_runs(tree, "run_*", keys=KEYS)
    assert all(type(r.amplitudes) is np.ndarray for r in runs)


def test_a_tree_mixing_mapped_and_plain_runs_reads_each_its_own_way(tmp_path):
    tree = _three_channel_tree(tmp_path / "three")
    (tree / "run_b" / MMAP_FILE).unlink()
    (tree / "run_d" / MMAP_FILE).unlink()
    eager = {r.tag: r for r in read_mode_runs(tree, "run_*", keys=KEYS)}
    mapped = {r.tag: r for r in read_mode_runs(tree, "run_*", keys=MAPPED)}
    assert {t for t, r in mapped.items() if isinstance(r.amplitudes, np.memmap)} == {"run_a", "run_c"}
    for tag, run in mapped.items():
        assert _same_bits(run.amplitudes, eager[tag].amplitudes)
        assert _same_bits(run.boxes, eager[tag].boxes) and run.composition == eager[tag].composition
        assert (run.reference_box is None) == (eager[tag].reference_box is None)


def _stale(directory: Path, array) -> None:
    np.save(directory / MMAP_FILE, array)


@pytest.mark.parametrize("what,change", [
    ("frames", lambda a: a[:-1]),
    ("channels", lambda a: a[:, :2]),
    ("modes", lambda a: a[..., :-1]),
    ("dtype", lambda a: a.astype(np.complex128)),
    ("order", lambda a: np.asfortranarray(a)),
    ("layout", lambda a: np.ascontiguousarray(np.moveaxis(a, 1, 2))),
])
def test_a_sidecar_that_disagrees_with_its_archive_is_refused_naming_the_run(tmp_path, what, change):
    tree = _three_channel_tree(tmp_path / "three")
    good = np.load(tree / "run_b" / MMAP_FILE)
    _stale(tree / "run_b", change(good))
    with pytest.raises(ValueError, match="run 'run_b'.*does not match"):
        read_mode_runs(tree, "run_*", keys=MAPPED)


def test_an_unreadable_sidecar_is_refused_naming_the_run(tmp_path):
    tree = _three_channel_tree(tmp_path / "three")
    path = tree / "run_c" / MMAP_FILE
    path.write_bytes(path.read_bytes()[:-64])
    with pytest.raises(ValueError, match="run 'run_c'.*not a readable"):
        read_mode_runs(tree, "run_*", keys=MAPPED)


def test_the_sidecar_is_checked_without_decompressing_the_archives_amplitudes(tmp_path, monkeypatch):
    """The archive's ``rho_k`` is never read as an array when its sidecar is mapped: only its header."""
    tree = _three_channel_tree(tmp_path / "three")
    real = np.lib.npyio.NpzFile.__getitem__

    def getitem(self, key):
        if key == KEYS.amplitudes:
            raise AssertionError("the mapped reader read the archive's amplitudes")
        return real(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", getitem)
    assert len(read_mode_runs(tree, "run_*", keys=MAPPED)) == 4


def test_a_sidecar_beside_an_archive_without_amplitudes_is_refused(tmp_path):
    tree = _three_channel_tree(tmp_path / "three")
    with np.load(tree / "run_a" / "modes.npz") as payload:
        kept = {k: payload[k] for k in payload.files if k != "rho_k"}
    np.savez(tree / "run_a" / "modes.npz", **kept)
    with pytest.raises(KeyError, match="run_a.*holds no 'rho_k'"):
        read_mode_runs(tree, "run_*", keys=MAPPED)


def test_the_sidecar_name_is_out_of_the_keys_repr_and_equality():
    assert MAPPED == KEYS and repr(MAPPED) == repr(KEYS)
    assert KEYS.mmap_file is None and MAPPED.mmap_file == MMAP_FILE


def test_a_system_opts_in_by_declaring_the_sidecar_in_its_training_block():
    from dataclasses import replace

    from aipf.system import load
    from aipf.train.fit import _archive_keys, _training_archive_keys

    system = load("feb")
    assert _training_archive_keys(system).mmap_file is None
    training = dict(system.defaults["training"])
    declared = replace(system, defaults={**system.defaults,
                                         "training": {**training, "mmap_sidecar": True}})
    assert _training_archive_keys(declared).mmap_file == MMAP_FILE
    assert _archive_keys(declared).mmap_file is None
    assert _training_archive_keys(declared) == _archive_keys(system)
    wrong = replace(system, defaults={**system.defaults,
                                      "training": {**training, "mmap_sidecar": "yes"}})
    with pytest.raises(ValueError, match="mmap_sidecar"):
        _training_archive_keys(wrong)


def test_an_empty_sidecar_is_refused_naming_the_run_and_rewritten(tmp_path):
    tree = _three_channel_tree(tmp_path / "three")
    good = np.load(tree / "run_d" / MMAP_FILE)
    (tree / "run_d" / MMAP_FILE).write_bytes(b"")
    with pytest.raises(ValueError, match="run 'run_d'.*not a readable"):
        read_mode_runs(tree, "run_*", keys=MAPPED)
    assert convert_run(tree / "run_d", MAPPED)[0] == REPLACED
    assert _same_bits(np.load(tree / "run_d" / MMAP_FILE), good)


def _rewrite_amplitudes(directory: Path, factor: float) -> None:
    """The archive's ``rho_k`` replaced by another of the same shape and dtype, everything else kept."""
    with np.load(directory / "modes.npz") as payload:
        kept = {k: payload[k] for k in payload.files}
    kept["rho_k"] = (kept["rho_k"] * factor).astype(kept["rho_k"].dtype)
    np.savez_compressed(directory / "modes.npz", **kept)


def test_a_sidecar_copied_from_another_rho_k_of_the_same_shape_is_refused_and_replaced(tmp_path):
    tree = _three_channel_tree(tmp_path / "three")
    _rewrite_amplitudes(tree / "run_b", 2.0)
    with pytest.raises(ValueError, match=f"run 'run_b'.*{MMAP_RECORD} records it was copied from"):
        read_mode_runs(tree, "run_*", keys=MAPPED)
    status, reason = convert_run(tree / "run_b", MAPPED)
    assert status == REPLACED and "records it was copied from" in reason
    eager = {r.tag: r for r in read_mode_runs(tree, "run_*", keys=KEYS)}
    for run in read_mode_runs(tree, "run_*", keys=MAPPED):
        assert isinstance(run.amplitudes, np.memmap)
        assert _same_bits(run.amplitudes, eager[run.tag].amplitudes)


@pytest.mark.parametrize("record", [None, b"{not json", b"[1, 2]"])
def test_a_sidecar_without_a_readable_record_is_refused_and_rewritten(tmp_path, record):
    tree = _three_channel_tree(tmp_path / "three")
    path = tree / "run_a" / MMAP_RECORD
    path.unlink() if record is None else path.write_bytes(record)
    with pytest.raises(ValueError, match=f"run 'run_a'.*{MMAP_RECORD}"):
        read_mode_runs(tree, "run_*", keys=MAPPED)
    assert convert_run(tree / "run_a", MAPPED)[0] == REPLACED
    assert len(read_mode_runs(tree, "run_*", keys=MAPPED)) == 4


def test_a_declared_sidecar_missing_from_some_runs_is_a_warning_naming_them(tmp_path, caplog):
    tree = _three_channel_tree(tmp_path / "three")
    (tree / "run_c" / MMAP_FILE).unlink()
    with caplog.at_level("WARNING"):
        read_mode_runs(tree, "run_*", keys=MAPPED)
    warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("1 of 4 run(s) have no rho_k.npy" in m and "run_c" in m for m in warned), warned
    caplog.clear()
    with caplog.at_level("WARNING"):
        read_mode_runs(tree, "run_*", keys=KEYS)
    assert not [r for r in caplog.records if "rho_k.npy" in r.getMessage()]


def test_the_soft_limit_on_open_files_is_raised_for_the_mappings_or_refused(monkeypatch):
    import resource

    from aipf.train import dataset

    calls = []
    monkeypatch.setattr(dataset, "_open_files", lambda: 50)
    monkeypatch.setattr(resource, "getrlimit", lambda which: (300, 400))
    monkeypatch.setattr(resource, "setrlimit", lambda which, limits: calls.append(limits))
    dataset._reserve_open_files(10, "here")
    assert calls == [(50 + 10 + dataset._SPARE_OPEN_FILES, 400)]
    calls.clear()
    dataset._reserve_open_files(300 - 50 - dataset._SPARE_OPEN_FILES, "here")
    assert calls == []                     # exactly at the soft limit: nothing to raise
    with pytest.raises(OSError, match="here: mapping 1000 run.*hard limit on open files is 400"):
        dataset._reserve_open_files(1000, "here")


def test_the_open_files_counted_are_the_mappings_still_held(tmp_path):
    import gc

    from aipf.train import dataset

    tree = _three_channel_tree(tmp_path / "three")
    before = dataset._open_files()
    runs = read_mode_runs(tree, "run_*", keys=MAPPED)
    assert dataset._open_files() == before + 4
    del runs
    gc.collect()
    assert dataset._open_files() == before


def test_the_reader_reserves_one_open_file_per_mapped_run(tmp_path, monkeypatch):
    from aipf.train import dataset

    tree = _three_channel_tree(tmp_path / "three")
    (tree / "run_c" / MMAP_FILE).unlink()
    reserved = []
    monkeypatch.setattr(dataset, "_reserve_open_files", lambda n, where: reserved.append(n))
    read_mode_runs(tree, "run_*", keys=MAPPED, exclude_tags=("run_d",))
    read_mode_runs(tree, "run_*", keys=KEYS)
    assert reserved == [2]


# ---------------------------------------------------------------------------
# 4. the converter
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FEB_SAMPLE.is_dir(), reason="the tracked Fe-B sample is not in this checkout")
def test_the_converter_writes_the_moved_amplitudes_bit_for_bit_and_then_skips(tmp_path):
    tree = tmp_path / "modes_0GPa_v2"
    shutil.copytree(FEB_SAMPLE, tree)
    archives = {p: p.read_bytes() for p in tree.glob("*/modes.npz")}
    keys = MAPPED.replace(composition=("x_left", "x_right"), composition_fallback="x_B")
    first = list(convert_runs(tree, "cube_*", keys=keys))
    assert [done.status for done in first] == [WRITTEN, WRITTEN]
    stamps = {}
    for directory in (done.directory for done in first):
        with np.load(directory / "modes.npz") as payload:
            moved = np.moveaxis(payload["rho_k"], 2, 1)
        sidecar = np.load(directory / MMAP_FILE)
        assert sidecar.flags.c_contiguous and _same_bits(sidecar, moved)
        stamps[directory] = (directory / MMAP_FILE).stat().st_mtime_ns
    assert [d.status for d in convert_runs(tree, "cube_*", keys=keys)] == [SKIPPED, SKIPPED]
    assert [d.status for d in convert_runs(tree, "cube_*", keys=keys, verify=True)] == \
        [SKIPPED, SKIPPED]
    assert {d: (d / MMAP_FILE).stat().st_mtime_ns for d in stamps} == stamps
    assert {p: p.read_bytes() for p in tree.glob("*/modes.npz")} == archives
    assert not list(tree.rglob("*.partial"))


def test_the_converter_rewrites_a_stale_sidecar_and_verify_finds_other_frames(tmp_path):
    tree = _three_channel_tree(tmp_path / "three")
    good = np.load(tree / "run_a" / MMAP_FILE)
    _stale(tree / "run_a", good[:-2])
    status, reason = convert_run(tree / "run_a", MAPPED)
    assert status == REPLACED and "run 'run_a'" in reason and "frames" in reason
    assert _same_bits(np.load(tree / "run_a" / MMAP_FILE), good)
    _stale(tree / "run_a", good[::-1])                  # the archive's shape and dtype, other frames
    assert convert_run(tree / "run_a", MAPPED) == (SKIPPED, None)
    status, reason = convert_run(tree / "run_a", MAPPED, verify=True)
    assert status == REPLACED and "other frames" in reason
    assert _same_bits(np.load(tree / "run_a" / MMAP_FILE), good)


def test_the_converter_refuses_a_glob_that_matches_no_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="no run matching"):
        list(convert_runs(tmp_path, "nothing_*", keys=MAPPED))


def test_a_failed_write_leaves_neither_a_sidecar_nor_a_partial_file(tmp_path, monkeypatch):
    tree = _three_channel_tree(tmp_path / "three", sidecar=False)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(modes_mod, "mmap_sidecar_matches", interrupted)
    with pytest.raises(KeyboardInterrupt):
        convert_run(tree / "run_a", MAPPED)
    assert sorted(p.name for p in (tree / "run_a").iterdir()) == ["modes.npz", "provenance.json"]


def test_a_sidecar_that_does_not_read_back_is_never_renamed_into_place(tmp_path, monkeypatch):
    tree = _three_channel_tree(tmp_path / "three", sidecar=False)
    monkeypatch.setattr(modes_mod, "mmap_sidecar_matches", lambda *a, **k: False)
    with pytest.raises(OSError, match="differs from the amplitudes"):
        convert_run(tree / "run_a", MAPPED)
    assert sorted(p.name for p in (tree / "run_a").iterdir()) == ["modes.npz", "provenance.json"]


def test_the_sidecar_is_the_file_numpy_save_writes(tmp_path, monkeypatch):
    """Chunked writing (forced here to one frame per chunk) gives exactly ``np.save``'s file."""
    record = _record(n_frames=7, seed=4)
    monkeypatch.setattr(modes_mod, "_SIDECAR_CHUNK_BYTES", 1)
    write_mmap_sidecar(tmp_path / "chunked.npy", record.amplitudes,
                       fingerprint={"member": "rho_k.npy", "crc32": 0, "file_size": 0})
    np.save(tmp_path / "saved.npy", np.ascontiguousarray(np.moveaxis(record.amplitudes, 2, 1)))
    assert (tmp_path / "chunked.npy").read_bytes() == (tmp_path / "saved.npy").read_bytes()


@pytest.mark.skipif(not FEB_SAMPLE.is_dir(), reason="the tracked Fe-B sample is not in this checkout")
def test_aipf_modes_sidecar_converts_a_source_tree(tmp_path, monkeypatch, capsys):
    from aipf.cli.main import main

    monkeypatch.setenv("AIPF_RAW_FEB", str(tmp_path / "raw"))
    tree = tmp_path / "raw" / "fields" / "modes" / "modes_0GPa_v2"
    shutil.copytree(FEB_SAMPLE, tree)
    argv = ["modes-sidecar", "--system", "feb", "--tree", "modes/modes_0GPa_v2",
            "--pattern", "cube_*"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "written cube_x0.10_T1400_s1" in out and "2 written, 0 replaced, 0 skipped" in out
    assert len(list(tree.glob(f"*/{MMAP_FILE}"))) == 2
    assert main(argv + ["--verify"]) == 0
    assert "0 written, 0 replaced, 2 skipped, 0 failed" in capsys.readouterr().out
    damaged = tree / "cube_x0.75_T1400_s1"
    (damaged / MMAP_FILE).unlink()
    archive = (damaged / "modes.npz").read_bytes()
    (damaged / "modes.npz").write_bytes(archive[: len(archive) // 2])
    (damaged / f".{MMAP_FILE}.another-node.1.0a1b2c3d.partial").write_bytes(b"x")
    assert main(argv) == 1
    out = capsys.readouterr().out
    assert "failed cube_x0.75_T1400_s1: " in out
    assert "partial cube_x0.75_T1400_s1/.rho_k.npy.another-node.1.0a1b2c3d.partial" in out
    assert "0 written, 0 replaced, 1 skipped, 1 failed" in out and "1 partial file(s) left" in out
    assert main(["modes-sidecar", "--system", "feb", "--tree", "modes/modes_0GPa_v2",
                 "--pattern", "nothing_*"]) == 2
    assert "no run matching" in capsys.readouterr().err
    assert main(["modes-sidecar", "--system", "feb", "--tree", str(tree),
                 "--pattern", "cube_*"]) == 2
    assert "is absolute" in capsys.readouterr().err


def test_the_converter_reports_a_damaged_archive_and_converts_the_others(tmp_path):
    tree = _three_channel_tree(tmp_path / "three", sidecar=False)
    archive = tree / "run_b" / "modes.npz"
    archive.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
    with np.load(tree / "run_c" / "modes.npz") as payload:
        kept = {k: payload[k] for k in payload.files if k != "rho_k"}
    np.savez(tree / "run_c" / "modes.npz", **kept)
    done = {d.directory.name: d for d in convert_runs(tree, "run_*", keys=MAPPED)}
    assert {t: d.status for t, d in done.items()} == {
        "run_a": WRITTEN, "run_b": FAILED, "run_c": FAILED, "run_d": WRITTEN}
    assert done["run_b"].reason.startswith("BadZipFile") and done["run_c"].reason.startswith("KeyError")
    assert not (tree / "run_b" / MMAP_FILE).exists() and not (tree / "run_c" / MMAP_FILE).exists()


def _dead_pid() -> int:
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_leftover_partials_are_listed_and_only_a_dead_local_writers_removed(tmp_path):
    import socket

    tree = _three_channel_tree(tmp_path / "three")
    run = tree / "run_a"
    host = socket.gethostname()
    dead = run / f".{MMAP_FILE}.{host}.{_dead_pid()}.0a1b2c3d.partial"
    alive = run / f".{MMAP_RECORD}.{host}.{os.getppid()}.0a1b2c3d.partial"
    elsewhere = run / f".{MMAP_FILE}.another-node.example.{_dead_pid()}.0a1b2c3d.partial"
    unnamed = run / f".{MMAP_FILE}.junk.partial"
    for path in (dead, alive, elsewhere, unnamed):
        path.write_bytes(b"x")
    left, removed = leftover_partials(run, MAPPED)
    assert set(left) == {dead, alive, elsewhere, unnamed} and removed == ()
    done = list(convert_runs(tree, "run_*", keys=MAPPED, remove_partials=True))
    assert done[0].removed == (dead,) and set(done[0].partials) == {alive, elsewhere, unnamed}
    assert not dead.exists() and alive.exists() and elsewhere.exists()


# ---------------------------------------------------------------------------
# 5. the writer's default is unchanged
# ---------------------------------------------------------------------------

#: The time every zip member is stamped with in the writer tests: ``savez`` stamps each member with the
#: local time it is written.
_FIXED_TIME = time.struct_time((2000, 1, 1, 0, 0, 0, 5, 1, 0))

#: The ``modes.npz`` of :func:`_pinned_record`, as the writer before the sidecar existed wrote it at
#: :data:`_FIXED_TIME`: the sha256 of each member's name, compression and bytes in order, and of the
#: whole file under the zlib it was measured with (deflate output can change with the zlib build).
_PINNED_MEMBERS = "1ef018d7bdd5b063912d7c48980cfbd3c5aa2895c3829b2b337e153cfb0b5c84"
_PINNED_FILE = ("1.3.1", "f5b706f7e185c5971283690c4006bf375c4319668509455a4fac91c41e8c0b56")


def _pinned_record() -> ModesRecord:
    rng = np.random.default_rng(6)
    labels = _labels()
    shape = (30, labels.shape[0], 3)
    amplitudes = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    boxes = np.full((30, 3), 9.0) + 0.01 * rng.standard_normal((30, 3))
    return ModesRecord(amplitudes=amplitudes, labels=labels, boxes=boxes, T=1500.0, dt_frame=0.02,
                       provenance={"composition": {"x_a": 0.2, "x_b": 0.3, "x_c": 0.5}},
                       reference_box=np.array([9.0, 9.0, 9.5]))


@pytest.mark.parametrize("flag", [False, True])
def test_the_archive_save_writes_is_pinned(tmp_path, monkeypatch, flag):
    import hashlib
    import zipfile
    import zlib

    monkeypatch.setattr(time, "localtime", lambda *a: _FIXED_TIME)
    path = _pinned_record().save(tmp_path / "run", mmap_sidecar=flag)
    members = hashlib.sha256()
    with zipfile.ZipFile(path) as opened:
        for info in opened.infolist():
            members.update(info.filename.encode())
            members.update(bytes([info.compress_type]))
            members.update(opened.read(info))
    assert members.hexdigest() == _PINNED_MEMBERS
    if zlib.ZLIB_RUNTIME_VERSION == _PINNED_FILE[0]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == _PINNED_FILE[1]


def test_save_writes_the_same_archive_with_and_without_the_sidecar(tmp_path, monkeypatch):
    """Byte for byte, with the zip members' timestamps held fixed (``savez`` stamps each member with
    the time it is written)."""
    monkeypatch.setattr(time, "localtime", lambda *a: _FIXED_TIME)
    record = _record(n_frames=30, seed=6, reference=np.array([9.0, 9.0, 9.5]))
    plain = record.save(tmp_path / "plain")
    flagged = record.save(tmp_path / "flagged", mmap_sidecar=True)
    assert plain.read_bytes() == flagged.read_bytes()
    assert (tmp_path / "plain" / "provenance.json").read_bytes() == \
        (tmp_path / "flagged" / "provenance.json").read_bytes()
    assert sorted(p.name for p in (tmp_path / "plain").iterdir()) == ["modes.npz", "provenance.json"]
    assert sorted(p.name for p in (tmp_path / "flagged").iterdir()) == \
        ["modes.npz", "provenance.json", MMAP_RECORD, MMAP_FILE]
    with np.load(flagged) as payload:
        assert _same_bits(np.load(tmp_path / "flagged" / MMAP_FILE),
                          np.moveaxis(payload["rho_k"], 2, 1))


def test_saving_over_an_archive_removes_the_sidecar_of_the_one_it_replaces(tmp_path):
    first = _record(n_frames=30, seed=7)
    first.save(tmp_path / "run", mmap_sidecar=True)
    _record(n_frames=30, seed=8).save(tmp_path / "run")
    assert not (tmp_path / "run" / MMAP_FILE).exists()
    assert not (tmp_path / "run" / MMAP_RECORD).exists()
    assert len(read_mode_runs(tmp_path, "run", keys=MAPPED)) == 1
