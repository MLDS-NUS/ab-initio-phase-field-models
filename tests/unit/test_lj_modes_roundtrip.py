"""The Lennard-Jones mode tier against the archived modes: one run, field for field.

``aipf modes --system lj`` stores ONE channel, the declared combination of the two per-type sums
(``defaults["mode_fields"]``), extracted from the archived slab dump. The archived modes file
stores the two sums, and the published model's reader formed phi = 0.5 (rho_A - rho_B) / RHO_BAR
from them, then set the zero mode to 0.5 after dividing by V. The comparison is that reader's phi
against the tier's stored field. A missing tier entry is extracted here (about 80 s on a GPU);
the raw root is only read. Marked ``env``: everything here needs the Lennard-Jones raw root.
"""
from __future__ import annotations

import numpy as np
import pytest

from aipf.system import load

import declared_roots

pytestmark = pytest.mark.env

#: The run compared, by farm directory, and the archived modes file under the raw root.
TAG = "slab_overdamped_MD_xA_0.50_T_1.15_seed_43"
ARCHIVED = ("Data", "slab_overdamped", "modes", "T_1.15", "seed_43", "modes.npz")

#: The extraction as the archive was written: modes ordered by wavenumber.
ORDERING = "shell"


def _archived_phi(rho_k):
    """The published model's reader, in the archive's own complex64."""
    return np.ascontiguousarray(0.5 * (rho_k[..., 0] - rho_k[..., 1]) / 1.0)


@pytest.fixture(scope="module")
def pair():
    from aipf.pipeline.modes import modes
    system = load("lj")
    archive = declared_roots.raw_or_skip("lj", *ARCHIVED)
    rec = modes(system, TAG, sigma=system.defaults["sigma"],
                k_cut=system.defaults["k_cut"], ordering=ORDERING)
    with np.load(archive) as z:
        archived = {k: z[k] for k in ("rho_k", "nvec", "box", "T", "dt_frame",
                                      "k_cut", "n_A", "n_B")}
    return rec, archived


@pytest.mark.slow
def test_the_tier_stores_one_field_on_the_archived_mode_set(pair):
    rec, archived = pair
    assert rec.amplitudes.shape == archived["rho_k"].shape[:2] + (1,)
    assert rec.amplitudes.dtype == np.complex64
    assert np.array_equal(rec.labels, archived["nvec"])
    assert np.array_equal(rec.boxes, np.tile(archived["box"], (rec.boxes.shape[0], 1)))
    assert rec.T == float(archived["T"]) and rec.dt_frame == float(archived["dt_frame"])
    assert float(archived["k_cut"]) == rec.provenance["k_cut"]
    assert rec.provenance["skip_frames"] == 0


@pytest.mark.slow
def test_every_nonzero_mode_of_the_field_is_the_archive_readers_bit_for_bit(pair):
    rec, archived = pair
    zero = (rec.labels == 0).all(axis=1)
    ours = rec.amplitudes[:, ~zero, 0]
    theirs = _archived_phi(archived["rho_k"])[:, ~zero]
    assert np.array_equal(ours, theirs), int((ours != theirs).sum())


@pytest.mark.slow
def test_the_zero_mode_is_the_declared_mean_in_density_convention(pair):
    """The archive's reader sets 0.5 after dividing by V; the tier stores 0.5 V, which divides back exactly."""
    rec, archived = pair
    zero = np.flatnonzero((rec.labels == 0).all(axis=1))
    volume = float(np.prod(archived["box"]))
    stored = rec.amplitudes[:, zero, 0]
    assert np.all(stored == np.complex64(0.5 * volume))
    assert np.all((stored / volume).astype(np.complex64) == np.complex64(0.5))
    # an additive offset would not have been the archive's state: N_A != N_B in this run
    assert int(archived["n_A"]) != int(archived["n_B"])


@pytest.mark.slow
def test_the_tier_is_read_by_the_training_reader_as_one_channel(pair):
    from pathlib import Path

    from aipf.train.dataset import read_mode_runs
    from aipf.train.fit import _archive_keys
    rec, _ = pair
    entry = Path(rec.provenance["cache_dir"])
    runs = read_mode_runs(entry.parent, entry.name, keys=_archive_keys(load("lj")))
    assert len(runs) == 1 and runs[0].n_species == 1
    assert runs[0].composition == (0.5, 0.5)
    assert runs[0].temperature == 1.15


def test_the_tier_holds_every_archived_run():
    """34 runs over 13 temperatures, one entry each, as the raw root holds them."""
    from aipf.data import index
    system = load("lj")
    root = declared_roots.raw_or_skip("lj", "Data", "slab_overdamped", "modes")
    archived = sorted(root.glob("T_*/seed_*/modes.npz"))
    if not archived:
        pytest.skip(f"no archived modes under {root}")
    tier = index.modes_dir(system, "")
    built = sorted(tier.glob("slab_overdamped_MD_xA_0.50_T_*/modes.npz"))
    expected = {f"slab_overdamped_MD_xA_0.50_{p.parent.parent.name}_{p.parent.name}" for p in archived}
    assert {p.parent.name for p in built} == expected
    assert len(built) == len(archived) == 34
    assert len({p.parent.name.split("_T_")[1].split("_")[0] for p in built}) == 13
