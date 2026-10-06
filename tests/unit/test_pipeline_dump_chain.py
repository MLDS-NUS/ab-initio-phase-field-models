"""A run written as a restart chain of dumps, read as one timeline without a joined copy.

Each restart rewrites the steps since its restart file, so a step can be in two files; the last
writer wins. A gap at a junction, a file out of order and a step that goes back inside a file raise.
A killed job leaves its file ending in NUL padding or in a cut frame: the complete frames are
kept and the next file read. Corruption with a frame after it is not a tail and raises. A single path
is read by ``read_dump`` exactly as before.

The dumps are written here, in ``tmp_path``, frame by frame so files can share one.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from aipf.pipeline import coarse_grain
from aipf.pipeline.modes import modes_from_dump

PARAMS = dict(sigma=1.0, k_cut=2.0, atom_types=(1, 2), fields="per_type",
              ordering="lexicographic", route="dense", T=1.0, dt_frame=0.02,
              skip_frames=0)

#: A cell for the stated-cell rule, which streams the chain rather than holding it.
CELL = (10.1, 10.1, 10.1)


def _frame(step, *, n_atoms=8, seed=None):
    """One frame's text: its own positions and its own box, so no two frames are alike."""
    rng = np.random.default_rng(step if seed is None else seed)
    edge = 10.0 - 0.001 * step / 100
    pos = edge * rng.random((n_atoms, 3))
    types = np.where(np.arange(n_atoms) % 2, 2, 1)
    lines = ["ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS",
             str(n_atoms), "ITEM: BOX BOUNDS pp pp pp"]
    lines += [f"0.0 {edge!r}"] * 3
    lines.append("ITEM: ATOMS id type x y z")
    lines += [" ".join(repr(float(v)) for v in (i + 1, types[i], *pos[i]))
              for i in range(n_atoms)]
    return "\n".join(lines) + "\n"


def _write(path, steps, *, tail=b""):
    Path(path).write_bytes("".join(_frame(s) for s in steps).encode() + tail)
    return Path(path)


def _chain(tmp_path, *tails):
    """Steps 0-300 then a restart at 300 that rewrites step 300 and runs to 500. ``_frame`` writes one
    step alike wherever it is, so the copy kept makes no difference here."""
    a = _write(tmp_path / "chunk_0.dump", [0, 100, 200, 300],
               tail=tails[0] if tails else b"")
    b = _write(tmp_path / "chunk_1.dump", [300, 400, 500])
    return a, b


def _same(one, other):
    assert np.array_equal(one.amplitudes, other.amplitudes)
    assert np.array_equal(one.labels, other.labels)
    assert np.array_equal(one.boxes, other.boxes)


def _steps(paths):
    report = []
    return [f.timestep for f in coarse_grain.read_dump_chain(paths, report=report)], report


# ---------------------------------------------------------------------------
# the overlap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["time_mean", "first_frame", CELL])
def test_a_chain_with_an_overlap_is_the_single_dump_of_its_distinct_steps(
        tmp_path, rule):
    """Bit for bit; the stated cell and the first frame stream the chain, the time mean holds it."""
    a, b = _chain(tmp_path)
    joined = _write(tmp_path / "joined.dump", [0, 100, 200, 300, 400, 500])
    chained = modes_from_dump([a, b], reference_box=rule, **PARAMS)
    single = modes_from_dump(joined, reference_box=rule, **PARAMS)
    _same(chained, single)
    if rule != "time_mean":
        assert np.array_equal(chained.reference_box, single.reference_box)
    steps, _ = _steps([a, b])
    assert steps == [f.timestep for f in coarse_grain.read_dump(joined)]


def test_the_later_copy_of_an_overlapping_step_is_the_one_kept(tmp_path):
    """The last writer wins: the restart's copy of the step it restarted from."""
    a = _write(tmp_path / "a.dump", [0, 100])
    b = tmp_path / "b.dump"
    b.write_text(_frame(100, seed=7) + _frame(200))
    report = []
    frames = list(coarse_grain.read_dump_chain([a, b], report=report))
    later = next(f for f in coarse_grain.read_dump(b) if f.timestep == 100)
    assert [f.timestep for f in frames] == [0, 100, 200]
    assert np.array_equal(frames[1].positions, later.positions)
    assert [(e["frames_kept"], e["frames_superseded"]) for e in report] == \
        [(1, 1), (2, 0)]


def test_a_restart_from_further_back_replaces_every_step_it_rewrites(tmp_path):
    """The earlier file's frames from the restart on are a branch nothing continues."""
    a = _write(tmp_path / "a.dump", [0, 100, 200, 300])
    b = tmp_path / "b.dump"
    b.write_text("".join(_frame(s, seed=s + 7) for s in (100, 200, 300, 400)))
    report = []
    frames = list(coarse_grain.read_dump_chain([a, b], report=report))
    assert [f.timestep for f in frames] == [0, 100, 200, 300, 400]
    rewritten = list(coarse_grain.read_dump(b))
    for kept, written in zip(frames[1:], rewritten):
        assert np.array_equal(kept.positions, written.positions)
    assert report[0]["frames_superseded"] == 3
    assert (report[0]["first_step"], report[0]["last_step"]) == (0, 0)
    assert (report[1]["first_step"], report[1]["last_step"]) == (100, 400)


def test_a_restart_from_the_same_first_step_replaces_the_whole_file(tmp_path):
    a = _write(tmp_path / "a.dump", [0, 100, 200])
    b = _write(tmp_path / "b.dump", [0, 100, 200, 300])
    steps, report = _steps([a, b])
    assert steps == [0, 100, 200, 300]
    assert report[0]["frames_kept"] == 0 and report[0]["frames_superseded"] == 3
    assert report[0]["first_step"] is None and report[0]["last_step"] is None


# ---------------------------------------------------------------------------
# order and gaps
# ---------------------------------------------------------------------------

def test_a_chain_out_of_order_is_refused_naming_both_files(tmp_path):
    early = _write(tmp_path / "chunk_2.dump", [0, 100])
    late = _write(tmp_path / "chunk_10.dump", [200, 300])
    with pytest.raises(ValueError, match="out of order.*chunk_10 before"
                       ) as refused:
        _steps(sorted([early, late]))
    assert "chunk_2.dump" in str(refused.value)
    assert "chunk_10.dump" in str(refused.value)
    assert _steps([early, late])[0] == [0, 100, 200, 300]


def test_a_gap_at_a_junction_is_refused_naming_files_steps_and_stride(tmp_path):
    a = _write(tmp_path / "a.dump", [0, 100, 200])
    b = _write(tmp_path / "b.dump", [400, 500])
    with pytest.raises(ValueError, match="from step 200 .*a.dump.* to step "
                       "400 .*b.dump.*stride is 100"):
        _steps([a, b])


def test_a_gap_behind_a_one_frame_file_is_judged_on_the_next_files_stride(
        tmp_path):
    one = _write(tmp_path / "a.dump", [0])
    with pytest.raises(ValueError, match="stride is 100"):
        _steps([one, _write(tmp_path / "b.dump", [200, 300])])
    assert _steps([one, _write(tmp_path / "c.dump", [100, 200])])[0] == \
        [0, 100, 200]


def test_a_junction_between_two_one_frame_files_is_accepted(tmp_path):
    a = _write(tmp_path / "a.dump", [0])
    b = _write(tmp_path / "b.dump", [500])
    assert _steps([a, b])[0] == [0, 500]


def test_junctions_behind_one_frame_files_wait_for_the_first_stride_shown(
        tmp_path):
    """0 to 700 is checked when the third file shows its stride of 100, and refused."""
    files = [_write(tmp_path / f"{i}.dump", steps) for i, steps in
             enumerate(([0], [700], [800, 900]))]
    with pytest.raises(ValueError, match="from step 0 .* to step 700 .*stride is 100"):
        _steps(files)
    files = [_write(tmp_path / f"ok{i}.dump", steps) for i, steps in
             enumerate(([0], [100], [200, 300]))]
    assert _steps(files)[0] == [0, 100, 200, 300]


def test_junctions_after_the_last_stride_shown_are_checked_against_it(tmp_path):
    files = [_write(tmp_path / f"{i}.dump", steps) for i, steps in
             enumerate(([0, 100], [200], [700]))]
    with pytest.raises(ValueError, match="from step 200 .* to step 700 .*stride is 100"):
        _steps(files)


def test_a_file_holding_only_a_cut_frame_supersedes_nothing(tmp_path):
    a = _write(tmp_path / "a.dump", [0, 100, 200, 300])
    cut = _frame(200)
    b = tmp_path / "b.dump"
    b.write_text(cut[:len(cut) // 2])
    c = _write(tmp_path / "c.dump", [400, 500])
    steps, report = _steps([a, b, c])
    assert steps == [0, 100, 200, 300, 400, 500]
    assert report[0]["frames_superseded"] == 0
    assert report[1]["frames_kept"] == 0 and report[1]["tail"] == "truncated_frame"


def test_a_superseded_branch_leaves_no_gap(tmp_path):
    """The stride is read on the frames kept: the junction is 0 to 100, not 300 to 100."""
    a = _write(tmp_path / "a.dump", [0, 100, 200, 300])
    b = _write(tmp_path / "b.dump", [100, 200])
    assert _steps([a, b])[0] == [0, 100, 200]


@pytest.mark.parametrize("steps", [[0, 100, 50, 200], [0, 100, 100, 200]])
def test_a_step_that_goes_back_inside_a_file_is_refused(tmp_path, steps):
    a = _write(tmp_path / "a.dump", steps)
    with pytest.raises(ValueError, match="inside the file"):
        _steps([a])


def test_the_provenance_names_every_file_its_digest_and_what_was_kept(tmp_path):
    import hashlib

    a, b = _chain(tmp_path)
    rec = modes_from_dump([a, b], entry_dir=tmp_path / "e", **PARAMS)
    p = rec.provenance
    assert p["source"] == [str(a), str(b)]
    assert p["source_sha256"] == [hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in (a, b)]
    assert p["chain"] == [
        {"source": str(a), "first_step": 0, "last_step": 200,
         "frames_kept": 3, "frames_superseded": 1, "tail_bytes": 0,
         "tail": None},
        {"source": str(b), "first_step": 300, "last_step": 500,
         "frames_kept": 3, "frames_superseded": 0, "tail_bytes": 0,
         "tail": None}]
    assert json.loads((tmp_path / "e" / "provenance.json").read_text()) == p


def test_a_chain_is_a_cache_hit_that_keeps_its_report(tmp_path, monkeypatch):
    import aipf.pipeline.modes as m

    a, b = _chain(tmp_path, b"\0" * 64)
    first = modes_from_dump([a, b], cache_dir=tmp_path / "c", **PARAMS)
    monkeypatch.setattr(m, "_extract", lambda *a, **k: pytest.fail("re-read"))
    again = modes_from_dump([a, b], cache_dir=tmp_path / "c", **PARAMS)
    assert again.provenance == first.provenance
    assert again.provenance["chain"][0]["tail"] == "nul_padding"
    _same(again, first)


def test_the_frames_skipped_are_counted_on_the_joined_timeline(tmp_path):
    a, b = _chain(tmp_path)
    joined = _write(tmp_path / "joined.dump", [0, 100, 200, 300, 400, 500])
    for skip in (2, 5):
        _same(modes_from_dump([a, b], **dict(PARAMS, skip_frames=skip)),
              modes_from_dump(joined, **dict(PARAMS, skip_frames=skip)))
    with pytest.raises(ValueError, match="no complete frame"):
        modes_from_dump([a, b], **dict(PARAMS, skip_frames=6))
    with pytest.raises(ValueError, match="no complete frame"):
        modes_from_dump([a, b], reference_box=CELL,
                        **dict(PARAMS, skip_frames=6))


def test_an_empty_chain_is_refused(tmp_path):
    with pytest.raises(ValueError, match="empty chain"):
        modes_from_dump([], **PARAMS)


# ---------------------------------------------------------------------------
# the tails
# ---------------------------------------------------------------------------

def test_a_nul_padded_file_keeps_its_frames_and_the_next_file_is_read(tmp_path):
    padding = b"\0" * 300_000
    a, b = _chain(tmp_path, padding)
    (tmp_path / "clean").mkdir()
    clean_a, clean_b = _chain(tmp_path / "clean")
    rec = modes_from_dump([a, b], **PARAMS)
    _same(rec, modes_from_dump([clean_a, clean_b], **PARAMS))
    assert rec.provenance["chain"][0] == {
        "source": str(a), "first_step": 0, "last_step": 200,
        "frames_kept": 3, "frames_superseded": 1,
        "tail_bytes": len(padding), "tail": "nul_padding"}


def test_a_padding_longer_than_the_scan_chunk_is_still_padding(tmp_path):
    padding = b"\0" * (3 << 20)
    a, b = _chain(tmp_path, padding)
    steps, report = _steps([a, b])
    assert steps == [0, 100, 200, 300, 400, 500]
    assert report[0]["tail"] == "nul_padding"
    assert report[0]["tail_bytes"] == len(padding)


@pytest.mark.parametrize("cut", [10, 60, 200, -1])
def test_a_cut_last_frame_ends_its_file_and_the_next_file_is_read(tmp_path, cut):
    """Cut inside the header, inside the rows, and one byte short of its last newline."""
    a, b = _chain(tmp_path)
    text = a.read_bytes()
    start = text.rindex(b"ITEM: TIMESTEP")
    a.write_bytes(text[:start + cut] if cut > 0 else text[:cut])
    steps, report = _steps([a, b])
    assert steps == [0, 100, 200, 300, 400, 500]
    assert report[0]["frames_kept"] == 3 and report[0]["frames_superseded"] == 0
    assert report[0]["tail"] == "truncated_frame"
    assert report[0]["tail_bytes"] == len(a.read_bytes()) - start


def test_a_cut_frame_followed_by_padding_is_a_cut_frame(tmp_path):
    a, b = _chain(tmp_path)
    text = a.read_bytes()
    start = text.rindex(b"ITEM: TIMESTEP")
    a.write_bytes(text[:start + 150] + b"\0" * 5000)
    steps, report = _steps([a, b])
    assert steps == [0, 100, 200, 300, 400, 500]
    assert report[0]["tail"] == "truncated_frame"
    assert report[0]["tail_bytes"] == 150 + 5000


def test_a_tail_in_the_last_file_is_reported_too(tmp_path):
    a, b = _chain(tmp_path)
    b.write_bytes(b.read_bytes() + b"\0" * 10)
    _, report = _steps([a, b])
    assert report[1]["tail"] == "nul_padding" and report[1]["tail_bytes"] == 10


# ---------------------------------------------------------------------------
# corruption that is not a tail
# ---------------------------------------------------------------------------

def _corrupt(path, insert: bytes, *, before_step: int):
    text = path.read_bytes()
    at = text.index(b"ITEM: TIMESTEP\n%d\n" % before_step)
    path.write_bytes(text[:at] + insert + text[at:])


@pytest.mark.parametrize("insert", [b"garbage\n", b"\0" * 4096,
                                    b"ITEM: TIMESTEP\n150\nITEM: NUMBER"])
def test_corruption_with_a_frame_after_it_raises(tmp_path, insert):
    """A stray line, a NUL hole, a frame cut and then followed by whole ones."""
    a, b = _chain(tmp_path)
    _corrupt(a, insert, before_step=200)
    with pytest.raises(ValueError, match="frame header follows"):
        _steps([a, b])
    with pytest.raises(ValueError, match="frame header follows"):
        modes_from_dump([a, b], **PARAMS)


def test_a_frame_header_split_across_the_scan_chunks_is_still_found(tmp_path):
    """The next ``ITEM: TIMESTEP`` starts five bytes before the 1 MiB chunk boundary of the scan."""
    a, b = _chain(tmp_path)
    text = a.read_bytes()
    start = text.index(b"ITEM: TIMESTEP\n200\n")
    junk = b"x" * ((1 << 20) - 5)
    a.write_bytes(text[:start] + junk + text[start:])
    with pytest.raises(ValueError, match=f"follows at offset {start + len(junk)}"):
        _steps([a, b])


def test_a_cut_frame_that_declares_millions_of_atoms_stops_at_its_last_row(
        tmp_path):
    """The rows are read until one is missing, not as many as the header declares."""
    import time

    a, b = _chain(tmp_path)
    text = a.read_bytes()
    start = text.rindex(b"ITEM: TIMESTEP")
    a.write_bytes(text[:start] + text[start:].replace(
        b"ITEM: NUMBER OF ATOMS\n8\n", b"ITEM: NUMBER OF ATOMS\n50000000\n"))
    began = time.perf_counter()
    steps, report = _steps([a, b])
    assert time.perf_counter() - began < 10.0
    assert steps == [0, 100, 200, 300, 400, 500]
    assert report[0]["tail"] == "truncated_frame"


def test_a_frame_with_fewer_rows_than_it_declares_raises(tmp_path):
    a, b = _chain(tmp_path)
    text = a.read_bytes().decode()
    a.write_text(text.replace("ITEM: NUMBER OF ATOMS\n8\n", "ITEM: NUMBER OF ATOMS\n9\n", 1))
    with pytest.raises(ValueError, match="frame header follows"):
        _steps([a, b])


# ---------------------------------------------------------------------------
# a single path is what it was
# ---------------------------------------------------------------------------

def test_a_single_path_keeps_its_provenance_and_reader(tmp_path, monkeypatch):
    """A string source, one digest, no chain report, and ``read_dump``'s own frames."""
    a = _write(tmp_path / "a.dump", [0, 100, 200], tail=b"\0" * 64)
    rec = modes_from_dump(a, **PARAMS)
    assert rec.provenance["source"] == str(a)
    assert isinstance(rec.provenance["source_sha256"], str)
    assert "chain" not in rec.provenance

    monkeypatch.setattr(coarse_grain, "read_dump_chain",
                        lambda *a, **k: pytest.fail("chain reader on one path"))
    assert np.array_equal(modes_from_dump(a, **PARAMS).amplitudes,
                          rec.amplitudes)


def test_a_one_file_chain_is_a_chain_and_not_the_single_path(tmp_path):
    a = _write(tmp_path / "a.dump", [0, 100, 200])
    one = modes_from_dump([a], cache_dir=tmp_path / "c", **PARAMS)
    single = modes_from_dump(a, cache_dir=tmp_path / "c", **PARAMS)
    _same(one, single)
    assert one.provenance["source"] == [str(a)]
    assert one.provenance["cache_dir"] != single.provenance["cache_dir"]


def test_read_dump_is_unchanged_on_a_tail_the_chain_reports(tmp_path):
    """The single reader stops quietly at a padded or cut tail, as it always has."""
    a = _write(tmp_path / "a.dump", [0, 100, 200], tail=b"\0" * 64)
    assert [f.timestep for f in coarse_grain.read_dump(a)] == [0, 100, 200]
    text = a.read_bytes()
    a.write_bytes(text[:text.rindex(b"ITEM: TIMESTEP") + 60])
    assert [f.timestep for f in coarse_grain.read_dump(a)] == [0, 100]
