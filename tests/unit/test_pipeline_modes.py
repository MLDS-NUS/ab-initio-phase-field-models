"""The coarse-graining stage's front door: provenance, a cache keyed on it,
and an archive a training reader can actually open.

``tiny_dump`` is local to this file. There is no ``tests/unit/conftest.py``,
and the dump's shape is copied from ``test_pipeline_coarse_grain.py``, which
is the file that pins what ``read_dump`` parses.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from aipf.pipeline.modes import (ARCHIVE_KEYS, SIDE_KEYS, ModesRecord,
                                 composition_label, modes_from_dump)

#: Ten angstrom edges, so ``k_cut`` 2 and 3 give mode sets of different size:
#: ``floor(k_cut L / 2 pi)`` is 3 and 4.
BOX = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])


def _write_dump(path, *, n_frames=3, n_atoms=8, box=BOX):
    rng = np.random.default_rng(0)
    lo = box[:, 0]
    length = box[:, 1] - box[:, 0]
    lines = []
    for frame in range(n_frames):
        pos = lo + length * rng.random((n_atoms, 3))
        types = np.where(np.arange(n_atoms) % 2, 2, 1)
        lines += ["ITEM: TIMESTEP", str(frame * 100),
                  "ITEM: NUMBER OF ATOMS", str(n_atoms),
                  "ITEM: BOX BOUNDS pp pp pp"]
        for axis in range(3):
            lines.append(f"{float(box[axis, 0])!r} {float(box[axis, 1])!r}")
        lines.append("ITEM: ATOMS id type x y z")
        for i in range(n_atoms):
            lines.append(" ".join(repr(float(v)) for v in
                                  (i + 1, types[i], *pos[i])))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def tiny_dump(tmp_path):
    return _write_dump(tmp_path / "traj.lammpstrj")


PARAMS = dict(sigma=1.0, k_cut=2.0, atom_types=(1, 2), fields="per_type",
              ordering="lexicographic", route="dense", T=1.0, dt_frame=0.02,
              skip_frames=0)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def test_record_carries_provenance_to_its_input(tiny_dump):
    rec = modes_from_dump(tiny_dump, **PARAMS)
    p = rec.provenance
    assert p["source"] == str(tiny_dump)
    assert len(p["source_sha256"]) == 64
    assert p["k_cut"] == 2.0 and p["sigma"] == 1.0
    assert p["ordering"] == "lexicographic" and p["route"] == "dense"


def test_provenance_carries_the_cache_directory_it_was_written_to(tiny_dump,
                                                                  tmp_path):
    """A later stage points a training source at the cache by reading this."""
    cache = tmp_path / "cache"
    rec = modes_from_dump(tiny_dump, cache_dir=cache, **PARAMS)
    assert rec.provenance["cache_dir"] is not None
    written = Path(rec.provenance["cache_dir"])
    assert (written / "modes.npz").is_file()
    assert written.parent == cache

    loose = modes_from_dump(tiny_dump, **PARAMS)
    assert loose.provenance["cache_dir"] is None


def test_the_digest_is_of_the_bytes_and_not_of_the_name(tiny_dump, tmp_path):
    """A rerun that replaced a dump under the same name is a new record."""
    first = modes_from_dump(tiny_dump, **PARAMS).provenance["source_sha256"]
    _write_dump(tiny_dump, n_frames=4)
    assert modes_from_dump(tiny_dump, **PARAMS).provenance["source_sha256"] \
        != first


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------

def test_a_second_call_with_the_same_input_is_a_cache_hit(tiny_dump, tmp_path,
                                                          monkeypatch):
    import aipf.pipeline.modes as m

    calls = []
    original = m._extract
    monkeypatch.setattr(
        m, "_extract",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    kw = dict(PARAMS, cache_dir=tmp_path)
    a = modes_from_dump(tiny_dump, **kw)
    b = modes_from_dump(tiny_dump, **kw)
    assert len(calls) == 1
    assert np.array_equal(a.amplitudes, b.amplitudes)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.boxes, b.boxes)
    assert a.provenance == b.provenance


def test_a_changed_parameter_is_a_new_record_not_a_stale_hit(tiny_dump,
                                                             tmp_path):
    kw = dict(PARAMS, cache_dir=tmp_path)
    kw.pop("k_cut")
    a = modes_from_dump(tiny_dump, k_cut=2.0, **kw)
    b = modes_from_dump(tiny_dump, k_cut=3.0, **kw)
    assert a.labels.shape[0] < b.labels.shape[0]
    assert a.provenance["cache_dir"] != b.provenance["cache_dir"]


def test_an_empty_cache_directory_is_a_miss_and_not_a_hit(tiny_dump, tmp_path):
    """The directory is created on the first WRITE.

    A path function that made the directory would leave an empty one claiming
    a cache it does not hold, and a reader who trusted the name would take
    the miss for a hit.
    """
    import aipf.pipeline.modes as m

    identity = dict(m.modes_from_dump(tiny_dump, **PARAMS).provenance)
    identity.pop("cache_dir")
    (tmp_path / m._cache_key(identity)).mkdir()
    rec = modes_from_dump(tiny_dump, cache_dir=tmp_path, **PARAMS)
    assert rec.amplitudes.shape[0] == 3


def test_parameters_have_no_default():
    with pytest.raises(TypeError):
        modes_from_dump("x.dump")   # type: ignore[call-arg]


def test_the_leading_frames_to_drop_have_no_default_either(tiny_dump):
    """A silent zero on a run that dumped its melt is a different timeline
    under the same name, so the count is stated and never assumed."""
    without = {k: v for k, v in PARAMS.items() if k != "skip_frames"}
    with pytest.raises(TypeError, match="skip_frames"):
        modes_from_dump(tiny_dump, **without)   # type: ignore[call-arg]


def test_a_negative_count_of_leading_frames_is_refused(tiny_dump):
    with pytest.raises(ValueError, match="negative"):
        modes_from_dump(tiny_dump, **dict(PARAMS, skip_frames=-1))


def test_dropped_frames_leave_the_timeline_and_the_reference_box(tiny_dump,
                                                                 tmp_path):
    """The drop happens before the box is averaged, so a dropped transient
    never chooses the mode set either."""
    from aipf.pipeline import coarse_grain, extract_modes

    kept = modes_from_dump(tiny_dump, **dict(PARAMS, skip_frames=1))
    assert kept.amplitudes.shape[0] == 2
    assert kept.boxes.shape == (2, 3)

    frames = list(coarse_grain.read_dump(tiny_dump))[1:]
    boxes = coarse_grain.box_lengths(frames)
    expected = extract_modes.mode_set(
        extract_modes.reference_box(boxes, rule="time_mean"),
        k_cut=PARAMS["k_cut"], ordering=PARAMS["ordering"])
    assert np.array_equal(kept.labels, expected)
    assert np.array_equal(kept.boxes, boxes)


def test_a_different_drop_is_a_different_cache_entry(tiny_dump, tmp_path):
    a = modes_from_dump(tiny_dump, cache_dir=tmp_path, **PARAMS)
    b = modes_from_dump(tiny_dump, cache_dir=tmp_path,
                        **dict(PARAMS, skip_frames=1))
    assert a.provenance["skip_frames"] == 0
    assert b.provenance["skip_frames"] == 1
    assert a.provenance["cache_dir"] != b.provenance["cache_dir"]


def test_dropping_every_frame_is_refused_rather_than_returned_empty(tiny_dump):
    with pytest.raises(ValueError, match="no complete frame"):
        modes_from_dump(tiny_dump, **dict(PARAMS, skip_frames=3))


# ---------------------------------------------------------------------------
# which frames are the run being prepared
# ---------------------------------------------------------------------------

def _record(dump, **extra):
    return {"tag": "demo", "farm_dir": "P800/demo", "trajectory": str(dump),
            "meta": {"T_K": 1.0, "extra": dict(extra)}}


def test_a_dump_that_ran_past_production_drops_its_preparation(tiny_dump):
    """Steps 0, 100, 200; production 150, preparation 150. The dump ran past
    the production count, so the frames below the preparation count are it."""
    from aipf.pipeline.modes import frames_prepared

    assert frames_prepared(_record(tiny_dump, n_equil=150, n_prod=150),
                           tiny_dump) == 2


def test_a_dump_that_is_production_only_drops_nothing(tiny_dump):
    from aipf.pipeline.modes import frames_prepared

    assert frames_prepared(_record(tiny_dump, n_equil=150, n_prod=10_000),
                           tiny_dump) == 0


def test_a_run_that_recorded_no_step_counts_is_refused_not_assumed(tiny_dump):
    """``frames_to_skip`` answers zero for absent counts, which is right for a
    function given a timeline and wrong for a front door."""
    from aipf.pipeline.modes import frames_prepared

    with pytest.raises(KeyError, match="n_prod"):
        frames_prepared(_record(tiny_dump, n_equil=150), tiny_dump)
    with pytest.raises(KeyError, match="n_equil"):
        frames_prepared(_record(tiny_dump), tiny_dump)


def test_the_steps_are_read_without_parsing_an_atom(tiny_dump):
    from aipf.pipeline import coarse_grain

    steps = coarse_grain.read_timesteps(tiny_dump)
    assert steps.tolist() == [0, 100, 200]
    assert steps.tolist() == [f.timestep
                              for f in coarse_grain.read_dump(tiny_dump)]


def test_a_truncated_dump_stops_the_step_read_where_it_stops_the_frame_read(
        tmp_path):
    from aipf.pipeline import coarse_grain

    path = _write_dump(tmp_path / "cut.lammpstrj", n_frames=3)
    text = path.read_text()
    path.write_text(text[:text.rindex("ITEM: TIMESTEP") + 60])
    assert coarse_grain.read_timesteps(path).tolist() == [0, 100]
    assert [f.timestep for f in coarse_grain.read_dump(path)] == [0, 100]


# ---------------------------------------------------------------------------
# the archive
# ---------------------------------------------------------------------------

def test_the_record_holds_the_archive_layout_channel_last(tiny_dump):
    """One exact moveaxis, applied at the file boundary and only there."""
    from aipf.pipeline import coarse_grain, extract_modes

    rec = modes_from_dump(tiny_dump, **PARAMS)
    assert rec.amplitudes.shape == (3, rec.labels.shape[0], 2)
    assert rec.amplitudes.dtype == np.complex64

    frames = list(coarse_grain.read_dump(tiny_dump))
    channel_first, _ = extract_modes.mode_series(
        frames, rec.labels, atom_types=(1, 2), route="dense",
        device="auto", dtype=np.complex64)
    assert np.array_equal(rec.amplitudes, np.moveaxis(channel_first, 1, 2))


def test_save_and_load_return_the_same_record(tiny_dump, tmp_path):
    rec = modes_from_dump(tiny_dump, **PARAMS)
    written = ModesRecord(rec.amplitudes, rec.labels, rec.boxes, rec.T,
                          rec.dt_frame, dict(rec.provenance,
                                             composition={"x": 0.25}))
    archive = written.save(tmp_path / "entry")
    assert archive.is_file()
    back = ModesRecord.load(tmp_path / "entry")
    assert np.array_equal(back.amplitudes, written.amplitudes)
    assert np.array_equal(back.labels, written.labels)
    assert np.array_equal(back.boxes, written.boxes)
    assert back.T == written.T and back.dt_frame == written.dt_frame
    assert back.provenance == written.provenance
    assert json.loads((tmp_path / "entry" / "provenance.json").read_text())


def test_a_composition_label_may_not_take_one_of_the_archives_own_keys(
        tiny_dump, tmp_path):
    """Refused, rather than overwriting the array it collides with.

    ``savez`` takes the last value for a repeated key, so a composition
    column spelled ``rho_k`` would replace the amplitudes with a scalar and
    the file would still load.
    """
    rec = modes_from_dump(tiny_dump, **PARAMS)
    written = ModesRecord(rec.amplitudes, rec.labels, rec.boxes, rec.T,
                          rec.dt_frame,
                          dict(rec.provenance,
                               composition={ARCHIVE_KEYS["amplitudes"]: 0.25}))
    with pytest.raises(ValueError, match="collide"):
        written.save(tmp_path / "clash")


def test_the_archive_is_labelled_with_the_composition_it_was_given(tiny_dump,
                                                                   tmp_path):
    """The archive carries the composition it was given: one with no label is trainable by nothing.

    Read back through the training reader's own declaration rather than by
    key name, which is the thing that has to keep working.
    """
    from aipf.train.dataset import ArchiveKeys, read_mode_runs

    keys = ArchiveKeys(file_name="modes.npz", amplitudes="rho_k",
                       amplitudes_channel_axis=2, labels="nvec", box="box",
                       temperature="T_K", frame_interval="dt_frame_ps",
                       composition=("x_left", "x_right"),
                       composition_fallback="x_one", quality_file=None,
                       quality_key=None)

    uniform = modes_from_dump(tiny_dump, composition={"x_one": 0.05},
                              cache_dir=tmp_path / "uniform", **PARAMS)
    two_sided = modes_from_dump(tiny_dump,
                                composition=dict(zip(SIDE_KEYS, (0.05, 0.5))),
                                cache_dir=tmp_path / "sided", **PARAMS)

    runs = read_mode_runs(tmp_path / "uniform", "*", keys=keys)
    assert [r.composition for r in runs] == [(0.05, 0.05)]
    assert runs[0].amplitudes.shape == (3, 2, uniform.labels.shape[0])

    runs = read_mode_runs(tmp_path / "sided", "*", keys=keys)
    assert [r.composition for r in runs] == [(0.05, 0.5)]
    assert two_sided.provenance["composition"] == {"x_left": 0.05,
                                                   "x_right": 0.5}


# ---------------------------------------------------------------------------
# the composition label
# ---------------------------------------------------------------------------

def test_a_uniform_composition_is_one_key_and_a_two_sided_one_is_two():
    uniform = {"species": ["A", "B"], "kind": "uniform", "x": {"B": 0.05}}
    assert composition_label(uniform, x_key="x_minor") == {"x_minor": 0.05}

    sided = {"species": ["A", "B"], "kind": "two_slab", "x": {"B": [0.05, 0.5]}}
    assert composition_label(sided, x_key="x_minor") == dict(zip(SIDE_KEYS,
                                                             (0.05, 0.5)))


def test_a_run_nobody_recorded_a_composition_for_is_labelled_with_nothing():
    """Absent is absent. A missing label written as zero would be a run at zero."""
    assert composition_label({}, x_key="x_minor") == {}
    assert composition_label({"kind": "uniform", "x": {"B": None}},
                             x_key="x_minor") == {}


def test_a_composition_the_archive_cannot_spell_is_refused():
    three = {"kind": "other", "x": {"B": [0.1, 0.2, 0.3]}}
    with pytest.raises(ValueError, match="sides"):
        composition_label(three, x_key="x_minor")


# ---------------------------------------------------------------------------
# the declared field composition
# ---------------------------------------------------------------------------

#: One stored field from two dump types: half their difference, its zero mode a declared mean.
HALF_DIFFERENCE = ({"name": "A", "weights": {1: 0.5, 2: -0.5}, "mean": 0.5},)


def test_the_per_type_form_is_a_declaration_and_has_no_default(tiny_dump):
    from aipf.pipeline.modes import PER_TYPE, check_fields
    assert check_fields("per_type", species=("A", "B")) == PER_TYPE
    without = {k: v for k, v in PARAMS.items() if k != "fields"}
    with pytest.raises(TypeError, match="fields"):
        modes_from_dump(tiny_dump, **without)   # type: ignore[call-arg]


@pytest.mark.parametrize("fields, match", [
    ("per_species", "not a declared form"),
    (HALF_DIFFERENCE * 2, "one stored channel per species"),
    (({"name": "B", "weights": {1: 1.0}, "mean": None},), "species order"),
    (({"name": "A", "weights": {1: 1.0}},), "exactly the keys"),
    (({"name": "A", "weights": {1: 1.0}, "mean": None, "offset": 0.5},),
     "exactly the keys"),
    (({"name": "A", "weights": {}, "mean": None},), "at least one"),
    (({"name": "A", "weights": {"1": 1.0}, "mean": None},), "integer"),
    (({"name": "A", "weights": {True: 1.0}, "mean": None},), "integer"),
    (({"name": "A", "weights": {1: float("nan")}, "mean": None},),
     "non-finite"),
])
def test_a_combination_that_is_not_well_formed_is_refused(fields, match):
    from aipf.pipeline.modes import check_fields
    with pytest.raises(ValueError, match=match):
        check_fields(fields, species=("A",))


def test_an_undeclared_composition_is_refused_naming_both_forms():
    import dataclasses

    from aipf.pipeline.modes import mode_fields
    from aipf.system import load
    system = load("hhe")
    bare = dataclasses.replace(system, defaults={
        k: v for k, v in system.defaults.items() if k != "mode_fields"})
    with pytest.raises(KeyError, match="per_type.*name.*weights.*mean"):
        mode_fields(bare)


def test_every_system_declares_its_composition():
    from aipf.pipeline.modes import PER_TYPE, mode_fields
    from aipf.system import load
    assert mode_fields(load("hhe")) == PER_TYPE
    assert mode_fields(load("feb")) == PER_TYPE
    assert mode_fields(load("lj")) == HALF_DIFFERENCE


def test_combine_weighs_in_the_stored_precision_and_sets_the_mean():
    from aipf.pipeline.modes import check_fields, combine_fields
    rng = np.random.default_rng(1)
    per_type = (rng.normal(size=(4, 5, 2))
                + 1j * rng.normal(size=(4, 5, 2))).astype(np.complex64)
    labels = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    boxes = np.tile([2.0, 3.0, 4.0], (4, 1))
    fields = check_fields(HALF_DIFFERENCE, species=("A",))
    out = combine_fields(per_type, labels, boxes, fields, (1, 2))
    assert out.shape == (4, 5, 1) and out.dtype == np.complex64
    expected = 0.5 * (per_type[..., 0] - per_type[..., 1])
    rest = np.array([0, 2, 3, 4])
    assert np.array_equal(out[:, rest, 0], expected[:, rest])
    assert np.array_equal(out[:, 1, 0], np.full(4, 0.5 * 24.0, np.complex64))


def test_a_mean_of_none_keeps_the_combination_s_own_zero_mode():
    from aipf.pipeline.modes import check_fields, combine_fields
    per_type = np.ones((2, 1, 2), np.complex64) * np.array([3, 5], np.complex64)
    fields = check_fields(({"name": "A", "weights": {2: 1.0, 1: 1.0},
                            "mean": None},), species=("A",))
    out = combine_fields(per_type, np.zeros((1, 3), int), np.ones((2, 3)),
                         fields, (1, 2))
    assert np.array_equal(out[..., 0], np.full((2, 1), 8.0, np.complex64))


def test_a_weight_on_a_type_not_extracted_is_refused():
    from aipf.pipeline.modes import check_fields, combine_fields
    fields = check_fields(HALF_DIFFERENCE, species=("A",))
    with pytest.raises(ValueError, match="not extracted"):
        combine_fields(np.zeros((1, 1, 1), np.complex64),
                       np.zeros((1, 3), int), np.ones((1, 3)), fields, (1,))


def test_a_combination_is_stored_recorded_and_cached(tiny_dump, tmp_path):
    from aipf.pipeline.modes import check_fields
    fields = check_fields(HALF_DIFFERENCE, species=("A",))
    per_type = modes_from_dump(tiny_dump, **PARAMS)
    kw = dict(PARAMS, fields=fields, cache_dir=tmp_path)
    one = modes_from_dump(tiny_dump, **kw)
    assert one.amplitudes.shape == per_type.amplitudes.shape[:2] + (1,)
    zero = (one.labels == 0).all(axis=1)
    assert np.array_equal(
        one.amplitudes[:, ~zero, 0],
        0.5 * per_type.amplitudes[:, ~zero, 0]
        - 0.5 * per_type.amplitudes[:, ~zero, 1])
    assert np.allclose(one.amplitudes[:, zero, 0], 0.5 * 1000.0)
    assert one.provenance["fields"] == [
        {"name": "A", "weights": [[1, 0.5], [2, -0.5]], "mean": 0.5}]
    assert "fields" not in per_type.provenance
    again = modes_from_dump(tiny_dump, **kw)
    assert again.provenance == one.provenance
    assert np.array_equal(again.amplitudes, one.amplitudes)
    assert one.provenance["cache_dir"] != modes_from_dump(
        tiny_dump, **dict(PARAMS, cache_dir=tmp_path)).provenance["cache_dir"]


def test_an_unchecked_combination_is_refused(tiny_dump):
    with pytest.raises(TypeError, match="check_fields"):
        modes_from_dump(tiny_dump, **dict(PARAMS, fields=list(HALF_DIFFERENCE)))


# ---------------------------------------------------------------------------
# the farm's layout: one file per run, where the archive keeps it
# ---------------------------------------------------------------------------

def test_an_entry_directory_holds_the_archive_itself(tiny_dump, tmp_path):
    entry = tmp_path / "tree" / "cube_x0.50_T1_s1"
    rec = modes_from_dump(tiny_dump, entry_dir=entry, **PARAMS)
    assert (entry / "modes.npz").is_file() and (entry / "provenance.json").is_file()
    assert rec.provenance["cache_dir"] == str(entry)
    again = modes_from_dump(tiny_dump, entry_dir=entry, **PARAMS)
    np.testing.assert_array_equal(again.amplitudes, rec.amplitudes)


def test_other_inputs_replace_the_entry_rather_than_hit_it(tiny_dump, tmp_path):
    entry = tmp_path / "cube_x0.50_T1_s1"
    small = modes_from_dump(tiny_dump, entry_dir=entry, **PARAMS)
    large = modes_from_dump(tiny_dump, entry_dir=entry, **dict(PARAMS, k_cut=3.0))
    assert large.labels.shape[0] > small.labels.shape[0]
    assert ModesRecord.load(entry).provenance["k_cut"] == 3.0


def test_a_moved_entry_reports_where_it_is_now(tiny_dump, tmp_path):
    modes_from_dump(tiny_dump, entry_dir=tmp_path / "a", **PARAMS)
    (tmp_path / "a").rename(tmp_path / "b")
    assert modes_from_dump(tiny_dump, entry_dir=tmp_path / "b",
                           **PARAMS).provenance["cache_dir"] == str(tmp_path / "b")


def test_both_layouts_at_once_are_refused(tiny_dump, tmp_path):
    with pytest.raises(ValueError, match="Pass one"):
        modes_from_dump(tiny_dump, cache_dir=tmp_path, entry_dir=tmp_path / "e", **PARAMS)


def test_the_farm_writes_each_run_where_the_archive_keeps_it(tiny_dump, tmp_path, monkeypatch):
    """``modes(system, tag)`` writes ``modes/<partition>/<tag>/modes.npz``, so the training reader finds the run
    with the archive's own pattern and files it under its real tag (which ``exclude_tags`` names)."""
    from aipf.data import index
    from aipf.pipeline import modes as m
    from aipf.train.dataset import read_mode_runs
    from aipf.train.fit import _archive_keys
    from aipf.system import load

    feb = load("feb")
    record = {"farm_dir": "0GPa/cube_x0.50_T1800_P0_s1", "trajectory": str(tiny_dump),
              "meta": {"T_K": 1800.0, "dump_every_ps": 0.1, "extra": {"n_equil": 0, "n_prod": 200},
                       "composition": {"x": {"B": 0.5}}}}
    monkeypatch.setattr(index, "record_for_tag", lambda system, tag: record)
    monkeypatch.setattr(index, "modes_dir",
                        lambda system, farm_dir: tmp_path / "modes" / farm_dir)
    rec = m.modes(feb, record["farm_dir"], sigma=1.0, k_cut=2.0)
    entry = tmp_path / "modes" / "0GPa" / "cube_x0.50_T1800_P0_s1"
    assert rec.provenance["cache_dir"] == str(entry)
    assert sorted(p.name for p in entry.iterdir()) == ["modes.npz", "provenance.json"]
    [run] = read_mode_runs(tmp_path / "modes" / "0GPa", "cube_*", keys=_archive_keys(feb))
    assert run.tag == "cube_x0.50_T1800_P0_s1"
    assert read_mode_runs(tmp_path / "modes" / "0GPa", "cube_*", keys=_archive_keys(feb),
                          exclude_tags=("other",))[0].tag == run.tag
    with pytest.raises(FileNotFoundError):
        read_mode_runs(tmp_path / "modes" / "0GPa", "cube_*", keys=_archive_keys(feb),
                       exclude_tags=(run.tag,))
