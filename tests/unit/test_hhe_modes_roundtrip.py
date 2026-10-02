"""The H/He mode tier: a cache ``aipf modes`` writes is a training source.

``aipf modes`` was compared with the archived modes files of two runs (a slab that dropped nothing,
a cube whose archive dropped its melt frames), judged against the published extractor's own
run-to-run deviation; that comparison is migration evidence and not part of this suite. Here the
cache it writes for the same two state points is read back by the training reader, through its own
declaration. Marked ``env`` and ``slow``: the state points are extracted from the H/He raw root.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aipf.pipeline.modes import modes
from aipf.system import load

import declared_roots


#: The state points, by farm directory: the run that dropped nothing and the
#: run that dropped a melt.
TAGS = ("200GPa/slab_xl0.05_xr0.50_T02000", "200GPa/cube_x0.05_T02000")

#: What the training reader gets back for each, through its own declaration.
#: The slab carries a composition per side; the cube carries one, and the
#: reader's fallback maps it onto both positions.
EXPECTED_COMPOSITION = {TAGS[0]: (0.05, 0.5), TAGS[1]: (0.05, 0.05)}


@pytest.mark.slow
@pytest.mark.env
@pytest.mark.parametrize("tag", TAGS)
def test_the_cached_archive_is_the_one_a_training_source_reads(tag):
    """The cache the front door writes is trainable, by the reader's own keys.

    Reached through ``provenance["cache_dir"]``, which is how a later stage
    points a training source at it. A cache that could only be read by
    knowing where it was put would not be one.
    """
    from aipf.train.dataset import ArchiveKeys, read_mode_runs

    declared_roots.raw_or_skip("hhe")
    system = load("hhe")
    rec = modes(system, tag, sigma=system.defaults["sigma"],
                k_cut=system.defaults["k_cut"])
    entry = Path(rec.provenance["cache_dir"])
    keys = ArchiveKeys(file_name="modes.npz", amplitudes="rho_k",
                       amplitudes_channel_axis=2, labels="nvec", box="box",
                       temperature="T_K", frame_interval="dt_frame_ps",
                       composition=("x_left", "x_right"),
                       composition_fallback=system.table_keys["x"],
                       quality_file=None, quality_key=None)
    runs = read_mode_runs(entry.parent, entry.name, keys=keys)
    assert len(runs) == 1
    run = runs[0]
    assert run.composition == EXPECTED_COMPOSITION[tag]
    assert run.amplitudes.shape == (rec.amplitudes.shape[0], 2,
                                    rec.amplitudes.shape[1])
    assert run.temperature == 2000.0
