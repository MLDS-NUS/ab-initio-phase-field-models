"""Iron-boron's declared training sources are the published training partition.

``aipf train --source declared`` reads ``defaults["training"]["sources"]``. Per source this
checks the runs it selects and the batches one epoch draws from them, through the driver's
own window, split and loader settings. Only the array headers are read, so this takes seconds.
Marked ``env``: the runs are under the Fe-B raw root.
"""
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aipf.system import load

pytestmark = pytest.mark.env

#: Runs per source, and batches per epoch: the loader cycles to the longest source (stable_5),
#: so an epoch is 216 steps and the published 60 epochs are 12960.
_RUNS = {"noneq_0": 22, "stable_0": 103, "noneq_5": 16, "stable_5": 107,
         "noneq_10": 11, "stable_10": 97}
_BATCHES = {"noneq_0": 209, "stable_0": 172, "noneq_5": 127, "stable_5": 216,
            "noneq_10": 127, "stable_10": 173}


def _frames(path: Path) -> int:
    """The timeline length of one ``modes.npz``, off its amplitude header alone."""
    with zipfile.ZipFile(path) as archive, archive.open("rho_k.npy") as f:
        version = np.lib.format.read_magic(f)
        read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                else np.lib.format.read_array_header_2_0)
        return int(read(f)[0][0])


@pytest.fixture(scope="module")
def selected():
    import declared_roots
    from aipf.train.fit import _specs_from, declared_source_table, source_root

    system = load("feb")
    declared_roots.raw_or_skip("feb", "fields")
    table = declared_source_table(system)
    specs = _specs_from(system, table, [row[0] for row in table])
    keys_file = "modes.npz"
    out = {}
    for spec in specs:
        excluded = set(spec.exclude_tags)
        out[spec.name] = [d for d in sorted(Path(spec.root).glob(spec.pattern))
                          if d.name not in excluded and (d / keys_file).is_file()]
    if not all(out.values()):
        pytest.skip(f"the Fe-B mode trees are not all under {source_root(system)}")
    return system, specs, out


def test_the_declared_sources_are_the_six_published_ones_in_order(selected):
    system, specs, _ = selected
    assert [s.name for s in specs] == list(_RUNS)
    assert {s.loss_weight for s in specs} == {15000.0}
    assert sum(len(s.exclude_tags) for s in specs) == 572
    assert {s.grid for s in specs} == {(32, 32, 32)}


def test_each_source_selects_the_published_runs(selected):
    _, _, runs = selected
    assert {name: len(found) for name, found in runs.items()} == _RUNS


def test_one_epoch_is_216_batches(selected):
    from aipf.train.datamodule import split_runs
    from aipf.train.fit import _loader_from_system, _split_from_system, _window_from_system

    system, _, runs = selected
    window = _window_from_system(system)
    split = _split_from_system(system, None)
    batch = _loader_from_system(system, None).batch_size
    batches = {}
    for name, found in runs.items():
        train, _ = split_runs([SimpleNamespace(tag=d.name, path=d) for d in found], split)
        windows = sum(len(window.centres(_frames(r.path / "modes.npz"))) for r in train)
        batches[name] = math.ceil(windows / batch)
    assert batches == _BATCHES
    assert 60 * max(batches.values()) == 12960


def test_the_declared_band_puts_every_run_on_the_grid(selected):
    """The stored ball (k_cut 3.0) reaches past 32^3 on 51 of the 356 runs, the long low-x_B boxes;
    the declared band |k| <= 2.0 keeps every run's scattered modes on the grid. The grid check is
    ``scatter_modes`` itself, on zero amplitudes."""
    from aipf.train.dataset import band_keep, scatter_modes
    from aipf.train.fit import _window_from_system

    system, specs, runs = selected
    band = _window_from_system(system).band_k_max
    assert band == system.defaults["k_max"] == 2.0
    grid = specs[0].grid

    def fits(labels) -> bool:
        try:
            scatter_modes(np.zeros((2, len(labels)), np.complex64), labels, 1.0, grid)
        except ValueError as refused:
            assert "beyond the grid" in str(refused)
            return False
        return True

    past_grid, kept_past_grid = [], []
    for found in runs.values():
        for d in found:
            with np.load(d / "modes.npz") as payload:
                labels, boxes = payload["nvec"], payload["box"]
            if not fits(labels):
                past_grid.append(d.name)
            if not fits(labels[band_keep(labels, boxes, band)]):
                kept_past_grid.append(d.name)
    assert len(past_grid) == 51
    assert kept_past_grid == []
