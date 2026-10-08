"""The reference cell of an NPT run: the labels, ``k`` and ``V`` read in one stated box.

A barostatted box shrinks as the run heals. The amplitudes are sums in each frame's own scaled
coordinates, so an affine change of the box leaves them as they are; what the cell decides is the
mode set, every ``k`` and the volume a density is divided by. The default rule stays what the
archive was written under, and an archive without the new key reads exactly as before.

The dumps are written here, in ``tmp_path``; the shape is ``test_pipeline_modes.py``'s.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aipf.pipeline import extract_modes
from aipf.pipeline.kde import _resolve_device
from aipf.pipeline.modes import (REFERENCE_BOX, REFERENCE_BOX_KEY, ModesRecord,
                                 combine_fields, modes_from_dump,
                                 recorded_reference)
from aipf.train.dataset import (ArchiveKeys, ModeRun, ModeWindowDataset,
                                WindowSettings, read_mode_runs)

#: A stated cell with no short binary spelling, so a rounded copy anywhere would show.
STATED_CELL = (50.315273, 50.315273, 50.315273)

#: The sha256 of ``_write_dump(path, _shrinking())`` and of its default call's ``rho_k`` bytes.
DUMP_SHA256 = "3c24f95cb9ed34a7dfed2753fb7ef968707f3e98f66bbe03378e64461969dd3c"
RHO_K_SHA256 = "c1e60a6095cefa339a120710459573d161998089ba93b8c117fe53d64cce98b1"

#: A cell for the ten-Angstrom toy dumps below.
CELL = (10.1, 10.1, 10.1)

PARAMS = dict(sigma=1.0, k_cut=2.0, atom_types=(1, 2), fields="per_type",
              ordering="lexicographic", route="dense", T=1.0, dt_frame=0.02,
              skip_frames=0)

#: The archive keys ``aipf.train.fit`` declares for what ``modes_from_dump`` writes.
KEYS = ArchiveKeys(file_name="modes.npz", amplitudes="rho_k",
                   amplitudes_channel_axis=2, labels="nvec", box="box",
                   temperature="T_K", frame_interval="dt_frame_ps",
                   composition=(), composition_fallback=None,
                   quality_file=None, quality_key=None,
                   reference_box=REFERENCE_BOX_KEY)

#: A window the toy timelines hold several of, on a grid that holds ``|n| <= 3``.
WINDOW = WindowSettings(estimator="weak", half_width=2, n_states=2, stride=1,
                        grid=(8, 8, 8), savgol_window=None, savgol_poly=None)


def _write_dump(path, edges, *, n_atoms=8, seed=0):
    """One frame per row of ``edges``, the same scaled coordinates in every frame (an affine box)."""
    rng = np.random.default_rng(seed)
    scaled = rng.random((n_atoms, 3))
    types = np.where(np.arange(n_atoms) % 2, 2, 1)
    lines = []
    for frame, edge in enumerate(np.asarray(edges, dtype=np.float64)):
        pos = scaled * edge
        lines += ["ITEM: TIMESTEP", str(frame * 100),
                  "ITEM: NUMBER OF ATOMS", str(n_atoms),
                  "ITEM: BOX BOUNDS pp pp pp"]
        lines += [f"0.0 {float(edge[axis])!r}" for axis in range(3)]
        lines.append("ITEM: ATOMS id type x y z")
        for i in range(n_atoms):
            lines.append(" ".join(repr(float(v)) for v in
                                  (i + 1, types[i], *pos[i])))
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def _shrinking(n_frames=10, start=10.0, stop=9.76):
    return np.repeat(np.linspace(start, stop, n_frames)[:, None], 3, axis=1)


def _samples(runs):
    dataset = ModeWindowDataset(runs, WINDOW)
    return [dataset[i] for i in range(len(dataset))]


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

def test_a_stated_cell_is_returned_exactly_whatever_the_frames():
    frames = _shrinking()
    cell = extract_modes.reference_box(frames, rule=STATED_CELL)
    assert cell.dtype == np.float64
    assert cell.tolist() == list(STATED_CELL)
    assert extract_modes.reference_box(frames, rule=np.array(STATED_CELL)
                                       ).tolist() == list(STATED_CELL)


@pytest.mark.parametrize("cell", [(10.0, 10.0), (10.0, -1.0, 10.0),
                                  (10.0, 0.0, 10.0), (10.0, np.nan, 10.0),
                                  (10.0, np.inf, 10.0), ("a", "b", "c"),
                                  ((1.0, 2.0, 3.0),) * 2])
def test_a_stated_cell_that_is_not_three_positive_finite_edges_is_refused(cell):
    with pytest.raises(ValueError, match="positive finite"):
        extract_modes.reference_box(_shrinking(), rule=cell)
    with pytest.raises(ValueError, match="positive finite"):
        modes_from_dump("never-read.dump", reference_box=cell, **PARAMS)


def test_an_unknown_rule_name_is_still_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="not one of"):
        modes_from_dump(_write_dump(tmp_path / "a.dump", _shrinking()),
                        reference_box="last_frame", **PARAMS)


# ---------------------------------------------------------------------------
# the default is the archive it always was
# ---------------------------------------------------------------------------

def test_the_default_identity_is_the_one_written_before_the_cell_existed(
        tmp_path):
    """Spelled out key by key, so a new entry in a default call's identity is a failure here."""
    dump = _write_dump(tmp_path / "a.dump", _shrinking())
    rec = modes_from_dump(dump, cache_dir=tmp_path / "cache", **PARAMS)
    expected = {
        "schema": 1, "source": str(dump),
        "source_sha256": hashlib.sha256(dump.read_bytes()).hexdigest(),
        "sigma": 1.0, "k_cut": 2.0, "atom_types": [1, 2],
        "ordering": "lexicographic", "route": "dense",
        "reference_box": "time_mean", "device": "auto",
        "device_resolved": str(_resolve_device("auto")), "T": 1.0,
        "dt_frame": 0.02, "skip_frames": 0, "composition": {}}
    key = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()
                         ).hexdigest()[:16]
    assert rec.provenance == dict(expected, cache_dir=str(tmp_path / "cache" / key))
    assert rec.reference_box is None
    with np.load(tmp_path / "cache" / key / "modes.npz") as payload:
        assert sorted(payload.files) == ["T_K", "box", "dt_frame_ps", "nvec",
                                         "rho_k"]
        rho_k = payload["rho_k"]
    # the dump and its amplitudes as the code before the cell wrote them, on the processor and on a GPU
    assert expected["source_sha256"] == DUMP_SHA256
    assert hashlib.sha256(rho_k.tobytes()).hexdigest() == RHO_K_SHA256


def test_naming_the_default_rule_is_the_default_call(tmp_path):
    dump = _write_dump(tmp_path / "a.dump", _shrinking())
    a = modes_from_dump(dump, cache_dir=tmp_path / "c", **PARAMS)
    b = modes_from_dump(dump, cache_dir=tmp_path / "c",
                        reference_box=REFERENCE_BOX, **PARAMS)
    assert a.provenance == b.provenance
    assert len(list((tmp_path / "c").iterdir())) == 1


# ---------------------------------------------------------------------------
# another rule: recorded, written, and read
# ---------------------------------------------------------------------------

def test_a_stated_cell_is_recorded_and_written_and_the_boxes_stay_the_frames(
        tmp_path):
    edges = _shrinking()
    dump = _write_dump(tmp_path / "a.dump", edges)
    rec = modes_from_dump(dump, entry_dir=tmp_path / "e",
                          reference_box=STATED_CELL, **PARAMS)
    assert rec.provenance["reference_box"] == list(STATED_CELL)
    assert np.array_equal(rec.boxes, edges)
    assert np.array_equal(rec.labels, extract_modes.mode_set(
        np.array(STATED_CELL), k_cut=2.0, ordering="lexicographic"))
    with np.load(tmp_path / "e" / "modes.npz") as payload:
        assert payload[REFERENCE_BOX_KEY].dtype == np.float64
        assert payload[REFERENCE_BOX_KEY].tolist() == list(STATED_CELL)
        assert np.array_equal(payload["box"], edges)
    stored = json.loads((tmp_path / "e" / "provenance.json").read_text())
    assert stored["reference_box"] == list(STATED_CELL)
    assert recorded_reference(tmp_path / "e").tolist() == list(STATED_CELL)


def test_a_stated_cell_is_its_own_cache_entry_and_hits_on_the_second_call(
        tmp_path):
    dump = _write_dump(tmp_path / "a.dump", _shrinking())
    default = modes_from_dump(dump, cache_dir=tmp_path / "c", **PARAMS)
    first = modes_from_dump(dump, cache_dir=tmp_path / "c",
                            reference_box=CELL, **PARAMS)
    again = modes_from_dump(dump, cache_dir=tmp_path / "c",
                            reference_box=CELL, **PARAMS)
    nudged = modes_from_dump(dump, cache_dir=tmp_path / "c",
                             reference_box=(10.1, 10.1, 10.100000000000001),
                             **PARAMS)
    assert default.provenance["cache_dir"] != first.provenance["cache_dir"]
    assert again.provenance == first.provenance
    assert again.reference_box.tolist() == list(CELL)
    assert nudged.provenance["cache_dir"] != first.provenance["cache_dir"]


def test_the_first_frame_rule_writes_the_first_frames_box_as_the_cell(tmp_path):
    edges = _shrinking()
    rec = modes_from_dump(_write_dump(tmp_path / "a.dump", edges),
                          entry_dir=tmp_path / "e",
                          reference_box="first_frame", **PARAMS)
    assert rec.provenance["reference_box"] == "first_frame"
    assert np.array_equal(rec.reference_box, edges[0])
    assert np.array_equal(recorded_reference(tmp_path / "e"), edges[0])


def test_an_archive_without_the_key_records_the_default_rule(tmp_path):
    modes_from_dump(_write_dump(tmp_path / "a.dump", _shrinking()),
                    entry_dir=tmp_path / "e", **PARAMS)
    assert recorded_reference(tmp_path / "e") == REFERENCE_BOX


def test_the_record_round_trips_with_and_without_a_cell(tmp_path):
    dump = _write_dump(tmp_path / "a.dump", _shrinking())
    for name, rule in (("plain", REFERENCE_BOX), ("cell", STATED_CELL)):
        rec = modes_from_dump(dump, entry_dir=tmp_path / name,
                              reference_box=rule, **PARAMS)
        back = ModesRecord.load(tmp_path / name)
        assert back.provenance == rec.provenance
        if rec.reference_box is None:
            assert back.reference_box is None
        else:
            assert np.array_equal(back.reference_box, rec.reference_box)


def test_a_declared_mean_is_the_mean_times_the_cells_volume():
    labels = np.array([[0, 0, 0], [1, 0, 0]])
    boxes = _shrinking(n_frames=4)
    amplitudes = np.ones((4, 2, 2), np.complex64)
    fields = ({"name": "A", "weights": {1: 0.5, 2: -0.5}, "mean": 0.5},)
    own = combine_fields(amplitudes, labels, boxes, fields, (1, 2))
    fixed = combine_fields(amplitudes, labels, boxes, fields, (1, 2),
                           reference_box=np.array(CELL))
    assert np.array_equal(own[:, 0, 0], (0.5 * boxes.prod(axis=1)
                                         ).astype(np.complex64))
    assert np.all(fixed[:, 0, 0] == np.complex64(0.5 * np.prod(CELL)))
    assert np.array_equal(own[:, 1], fixed[:, 1])


# ---------------------------------------------------------------------------
# two seeds, one cell
# ---------------------------------------------------------------------------

def test_two_runs_whose_first_frames_differ_share_labels_k_and_v_in_a_stated_cell(
        tmp_path):
    """The first frames sit either side of a shell: under ``first_frame`` the mode sets differ."""
    a_edges, b_edges = _shrinking(start=10.0), _shrinking(start=10.3)
    kw = dict(PARAMS, k_cut=2.5)
    for name, edges, seed in (("a", a_edges, 0), ("b", b_edges, 1)):
        dump = _write_dump(tmp_path / f"{name}.dump", edges, seed=seed)
        modes_from_dump(dump, entry_dir=tmp_path / "cell" / name,
                        reference_box=CELL, **kw)
        modes_from_dump(dump, entry_dir=tmp_path / "first" / name,
                        reference_box="first_frame", **kw)
    first = read_mode_runs(tmp_path / "first", "*", keys=KEYS)
    assert first[0].labels.shape != first[1].labels.shape

    a, b = read_mode_runs(tmp_path / "cell", "*", keys=KEYS)
    assert np.array_equal(a.labels, b.labels)
    wide = WindowSettings(**{**WINDOW.__dict__, "grid": (10, 10, 10)})
    for one, other in zip(_samples_with(a, wide), _samples_with(b, wide)):
        assert torch.equal(one["boxes"], other["boxes"])
        assert torch.equal(one["boxes"], torch.tensor(CELL, dtype=torch.float32))
    assert float(a.box_mean(0, 3).prod()) == float(b.box_mean(5, 9).prod()) \
        == float(np.prod(np.array(CELL)))


# ---------------------------------------------------------------------------
# the dataset, the run weights, the rollouts
# ---------------------------------------------------------------------------

def test_the_mean_density_is_constant_over_a_shrinking_run_in_a_stated_cell(
        tmp_path):
    """``N / V_ref`` in every window; in the frames' own boxes it drifts with the volume."""
    dump = _write_dump(tmp_path / "a.dump", _shrinking())
    modes_from_dump(dump, entry_dir=tmp_path / "cell" / "a",
                    reference_box=CELL, **PARAMS)
    modes_from_dump(dump, entry_dir=tmp_path / "own" / "a", **PARAMS)
    n_per_type = 4

    cell = [s["rho_hat_states"][..., 0, 0, 0] for s in
            _samples(read_mode_runs(tmp_path / "cell", "*", keys=KEYS))]
    expected = torch.tensor(n_per_type / np.prod(CELL), dtype=torch.float32)
    for density in cell:
        assert torch.allclose(density.real, expected.expand_as(density.real),
                              rtol=1e-6, atol=0)
        assert torch.equal(density, cell[0])

    own = [s["rho_hat_states"][0, 0, 0, 0, 0].real for s in
           _samples(read_mode_runs(tmp_path / "own", "*", keys=KEYS))]
    assert own[-1] > own[0] * 1.03


def test_an_archive_without_the_key_reads_exactly_as_before(tmp_path):
    """Declaring the key changes nothing for a run that does not carry it."""
    modes_from_dump(_write_dump(tmp_path / "a.dump", _shrinking()),
                    entry_dir=tmp_path / "own" / "a", **PARAMS)
    before = read_mode_runs(tmp_path / "own", "*",
                            keys=KEYS.replace(reference_box=None))
    after = read_mode_runs(tmp_path / "own", "*", keys=KEYS)
    assert after[0].reference_box is None
    banded = WindowSettings(**{**WINDOW.__dict__, "band_k_max": 1.3})
    for settings in (WINDOW, banded):
        old = ModeWindowDataset(before, settings)
        new = ModeWindowDataset(after, settings)
        assert len(old) == len(new)
        for i in range(len(old)):
            for key, value in old[i].items():
                assert torch.equal(value, new[i][key]), key


def test_a_run_in_a_cell_reads_as_a_fixed_box_run_in_that_cell():
    """The window box, the band and the run weight all come from the cell."""
    from aipf.train.datamodule import RunWeighting, run_weights

    rng = np.random.default_rng(3)
    labels = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"),
                      -1).reshape(-1, 3).astype(np.int16)
    amplitudes = (rng.standard_normal((12, 2, len(labels)))
                  + 1j * rng.standard_normal((12, 2, len(labels)))
                  ).astype(np.complex64)
    common = dict(tag="r", amplitudes=amplitudes, labels=labels,
                  temperature=1.0, frame_interval=0.02, composition=())
    cell = np.array([9.0, 9.5, 10.0])
    moving = ModeRun(boxes=_shrinking(12), reference_box=cell, **common)
    fixed = ModeRun(boxes=cell, **common)
    settings = WindowSettings(**{**WINDOW.__dict__, "band_k_max": 1.3})
    for one, other in zip(_samples_with(moving, settings),
                          _samples_with(fixed, settings)):
        for key in one:
            assert torch.equal(one[key], other[key]), key
    weighting = RunWeighting(mode="inverse_band_power", sigma=1.0, k_max=1.3,
                             eps=1e-3, probe_every=2)
    other = ModeRun(boxes=_shrinking(12), **{**common, "tag": "s",
                                             "amplitudes": amplitudes[::-1].copy()})
    assert np.array_equal(run_weights([moving, other], weighting, settings),
                          run_weights([fixed, other], weighting, settings))


def _samples_with(run, settings):
    dataset = ModeWindowDataset([run], settings)
    return [dataset[i] for i in range(len(dataset))]


def test_the_declared_key_leaves_the_declarations_repr_and_equality_as_they_were():
    plain = KEYS.replace(reference_box=None)
    assert "reference_box" not in repr(KEYS)
    assert repr(KEYS) == repr(plain) and KEYS == plain


def test_training_declares_the_key_the_writer_writes():
    from aipf.system import load
    from aipf.train.fit import _archive_keys

    assert _archive_keys(load("lj")).reference_box == REFERENCE_BOX_KEY


def _rollout_system(raw: Path):
    return SimpleNamespace(paths=SimpleNamespace(raw=lambda: raw), n_species=2,
                           functional=SimpleNamespace(kwargs={"nyquist_mask": True}),
                           defaults={"sigma": 1.0})


_DECL = {"archive": dict(file_name="modes.npz", amplitudes="rho_k",
                         amplitudes_channel_axis=2, labels="nvec", box="box",
                         temperature="T_K", frame_interval="dt_frame_ps",
                         quality_file=None, quality_key=None),
         "spinodal": dict(modes_tree="tree", grid=(8, 8, 8), field_stride=1)}


def test_the_rollout_reads_a_cell_run_in_its_cell_and_an_old_run_as_before(
        tmp_path):
    from aipf.rollout.spinodal import read_run
    from aipf.spectral import SpectralOps
    from aipf.train.dataset import scatter_modes

    dump = _write_dump(tmp_path / "a.dump", _shrinking())
    modes_from_dump(dump, entry_dir=tmp_path / "tree" / "cell",
                    reference_box=CELL, **PARAMS)
    modes_from_dump(dump, entry_dir=tmp_path / "tree" / "own", **PARAMS)
    system = _rollout_system(tmp_path)

    cell = read_run(system, _DECL, "spinodal", "cell")
    assert np.array_equal(cell.boxes, np.repeat(np.array([CELL]), 10, axis=0))
    zero = cell.rho_hat[:, :, 0, 0, 0]
    assert torch.equal(zero, zero[:1].expand_as(zero))

    own = read_run(system, _DECL, "spinodal", "own")
    with np.load(tmp_path / "tree" / "own" / "modes.npz") as z:
        boxes = z["box"]
        rho_hat = torch.tensor(scatter_modes(
            np.moveaxis(z["rho_k"], -1, -2), z["nvec"],
            boxes.prod(axis=1)[:, None, None], (8, 8, 8)))
    ops = SpectralOps((8, 8, 8), 2, nyquist_mask=True)
    rho_hat = rho_hat * ops.sigma_filter(
        torch.tensor(boxes, dtype=torch.float32), 1.0).to(rho_hat.dtype)
    assert np.array_equal(own.boxes, boxes)
    assert torch.equal(own.rho_hat, rho_hat)


def test_the_mobility_reads_the_rule_the_archive_records(tmp_path):
    from aipf.pipeline.anchors import run_mobility

    edges = _shrinking()
    dump = _write_dump(tmp_path / "a.dump", edges)
    kw = dict(temperature=1.0, thermal_energy=1.0, lags=[1, 2, 3],
              frame_interval=0.02, band=(0.0, 9.0), min_lag=0.0,
              extrapolation_k_max=9.0)
    for name, rule, volume in (("cell", CELL, np.prod(CELL)),
                               ("own", REFERENCE_BOX,
                                np.prod(edges.mean(axis=0)))):
        rec = modes_from_dump(dump, entry_dir=tmp_path / name,
                              reference_box=rule, **PARAMS)
        out = run_mobility(np.moveaxis(rec.amplitudes, 2, 1), rec.boxes,
                           rec.labels,
                           reference_rule=recorded_reference(tmp_path / name),
                           **kw)
        assert out.volume == pytest.approx(volume, rel=1e-12)
