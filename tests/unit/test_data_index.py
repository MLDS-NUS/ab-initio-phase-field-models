import json

import pytest

from aipf.data import index
from aipf.data.meta import SCHEMA_VERSION
from aipf.paths import Paths
from aipf.system import AnchorRules, System


@pytest.fixture(autouse=True)
def _demo_scratch_is_its_own(monkeypatch):
    """A demo system here declares its own scratch root; a user's ``AIPF_RAW``, which
    ``Paths.scratch`` prefers to a declared default, must not replace it."""
    monkeypatch.delenv("AIPF_RAW", raising=False)


def _fake_scratch(tmp_path):
    """A miniature stand-in for a real scratch tree."""
    root = tmp_path / "scratch"
    for tag, T in (("cube_x0.05_T02000", 2000), ("cube_x0.50_T03000", 3000)):
        d = root / "cube_data_800GPa" / tag
        d.mkdir(parents=True)
        (d / "traj.lammpstrj").write_text("frames")
        (d / f"{tag}_meta.json").write_text(json.dumps({
            "tag": tag, "T_K": T, "P_GPa": 800.0, "x_He": 0.05,
            "n_atoms": 3456, "dt_ps": 0.001, "dump_every_ps": 0.02,
            "n_frames": 2001, "seed": 1, "status": "ok",
        }))
    return root


def _system(tmp_path, monkeypatch):
    root = _fake_scratch(tmp_path)
    monkeypatch.setenv("AIPF_RAW_DEMO", str(root))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "data"))
    return System(
        name="demo", n_species=2, species=("H", "He"),
        masses={"H": 1.008, "He": 4.0026}, atom_types={"H": 1, "He": 2},
        table_keys={"rho": ("rho_H", "rho_He"), "x": "x_He", "x_channel": 1},
        paths=Paths(system="demo"), anchor_rules=AnchorRules({}),
        constants={"P_GPa": 800.0},
        defaults={},
    )


def test_build_creates_symlinks_and_never_copies(tmp_path, monkeypatch):
    s = _system(tmp_path, monkeypatch)
    manifest = index.build(s)

    assert len(manifest["state_points"]) == 2
    # The farm is keyed by pressure, not by tag alone, so even a record with
    # no colliding sibling lands under its pressure directory.
    link = (s.paths.data_root() / "md" / "800GPa" / "cube_x0.05_T02000"
            / "traj.lammpstrj")
    assert link.is_symlink()
    assert link.resolve().read_text() == "frames"


def test_build_writes_valid_metadata(tmp_path, monkeypatch):
    s = _system(tmp_path, monkeypatch)
    index.build(s)
    meta = json.loads(
        (s.paths.data_root() / "md" / "800GPa" / "cube_x0.05_T02000"
         / "meta.json").read_text())
    assert meta["schema_version"] == SCHEMA_VERSION
    # no run.json and no declared cube deck: the table's cube row (every archived cube ran fix npt)
    assert meta["ensemble"] == "NPT"
    assert meta["geometry"] == "cube"
    assert meta["box"]["varying"] == ["x", "y", "z"]          # fix npt ... iso moves every axis


def test_run_json_wins_and_a_declared_deck_comes_second(tmp_path, monkeypatch):
    """The ensemble's order: a run's ``run.json`` request, then the system's declared md deck of the
    tag's geometry (for a bare geometry prefix), then the table."""
    import dataclasses
    from dataclasses import asdict

    from aipf.md.request import StatePoint

    s = _system(tmp_path, monkeypatch)
    s = dataclasses.replace(s, defaults={"md": {"cube-nvt": {
        "point": {"geometry": "cube", "ensemble": "NVT"}}}})
    point = StatePoint(geometry="cube", ensemble="langevin_overdamped", T=2500.0, x=0.25,
                       dt_ps=0.001, equil_ps=1.0, prod_ps=10.0, dump_every_ps=0.02,
                       dump_from="prod", seed=7, n_atoms=None, P=None)
    run = tmp_path / "scratch" / "cube_data_800GPa" / "cube_from_md_run"
    run.mkdir(parents=True)
    (run / "traj.lammpstrj").write_text("frames")
    (run / "run.json").write_text(json.dumps({
        "system": "demo", "template": "cube-nvt", "campaign": None, "point": asdict(point),
        "values": {}, "dry_run": False, "result": {"status": "ok"}}))
    manifest = index.build(s)
    ensembles = {r["tag"]: r["meta"]["ensemble"] for r in manifest["state_points"] if "meta" in r}
    assert ensembles[point.tag] == "langevin_overdamped"          # run.json's request
    assert ensembles["cube_x0.05_T02000"] == "NVT"                 # the declared deck, over the table
    assert ensembles["cube_x0.50_T03000"] == "NVT"


@pytest.mark.parametrize("ensemble,axes", [("NPT", ["x", "y", "z"]), ("NPT_z", ["z"]),
                                           ("NVT", []), ("langevin_overdamped", [])])
def test_a_run_json_records_the_same_moving_axes_as_the_table(tmp_path, monkeypatch, ensemble,
                                                               axes):
    """The two routes into a record, a run's ``run.json`` and an archived run's table row, write the
    same ``box.varying`` for the same ensemble: the axes that move (data.md), every one under NPT."""
    from dataclasses import asdict

    from aipf.data.meta import moving_axes
    from aipf.md.request import PRESSURE_CONTROLLED, StatePoint

    s = _system(tmp_path, monkeypatch)
    point = StatePoint(geometry="cube", ensemble=ensemble, T=2500.0, x=0.25, dt_ps=0.001,
                       equil_ps=1.0, prod_ps=10.0, dump_every_ps=0.02, dump_from="prod", seed=7,
                       n_atoms=None, P=800.0 if ensemble in PRESSURE_CONTROLLED else None)
    run = tmp_path / "scratch" / "cube_data_800GPa" / "cube_from_md_run"
    run.mkdir(parents=True)
    (run / "traj.lammpstrj").write_text("frames")
    (run / "run.json").write_text(json.dumps({
        "system": "demo", "template": "cube", "campaign": None, "point": asdict(point),
        "values": {}, "dry_run": False, "result": {"status": "ok"}}))
    records = {r["tag"]: r["meta"] for r in index.build(s)["state_points"] if "meta" in r}
    assert records[point.tag]["ensemble"] == ensemble
    assert records[point.tag]["box"]["varying"] == axes == moving_axes(ensemble)
    table_npt = records["cube_x0.05_T02000"]                   # the table's cube row: NPT
    assert table_npt["box"]["varying"] == moving_axes("NPT") == ["x", "y", "z"]


def test_slab_metadata_records_the_breathing_axis(tmp_path, monkeypatch):
    """A slab is run under NPT_z, so its box moves along z frame to frame.

    Recording ``varying: []`` for a slab, the same as a constant-volume cube,
    would be silently indistinguishable from one, and downstream code that
    branches on ``box.varying`` (a per-frame box and a moving k grid) would
    then run the wrong path on data that actually needs it.
    """
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    tag = "slab_xl0.00_xr0.90_T02000"
    d = scratch / "slab_data_800GPa" / tag
    d.mkdir(parents=True)
    (d / "traj.lammpstrj").write_text("frames")
    (d / f"{tag}_meta.json").write_text(json.dumps({
        "tag": tag, "T_K": 2000, "P_GPa": 800.0,
        "n_atoms": 4650, "dt_ps": 0.0001, "dump_freq": 200,
        "n_prod": 400000, "status": "ok",
    }))

    index.build(s)
    meta = json.loads(
        (s.paths.data_root() / "md" / "800GPa" / tag / "meta.json").read_text())
    assert meta["ensemble"] == "NPT_z"
    assert meta["box"]["varying"] == ["z"]


def test_dump_cadence_derived_from_steps_and_timestep():
    """Hydrogen helium records dump_freq in STEPS, not a cadence in ps.

    The two geometries reach the same 0.02 ps by different routes, so a
    hardcoded factor would be right for one and wrong for the other.
    """
    from aipf.data.index import _dump_every_ps
    assert _dump_every_ps({"dump_every_ps": 0.1}) == 0.1
    assert _dump_every_ps({"dt_ps": 0.0002, "dump_freq": 100}) == pytest.approx(0.02)
    assert _dump_every_ps({"dt_ps": 0.0001, "dump_freq": 200}) == pytest.approx(0.02)
    with pytest.raises(KeyError):
        _dump_every_ps({"dt_ps": 0.0002})


def test_frame_count_derived_when_absent():
    from aipf.data.index import _n_frames
    assert _n_frames({"n_frames": 2001}) == 2001
    assert _n_frames({"n_prod": 150000, "dump_freq": 100}) == 1500
    assert _n_frames({"n_prod": 400000, "dump_freq": 200}) == 2000


def test_two_slab_composition_is_a_pair(tmp_path, monkeypatch):
    """A direct-coexistence slab has two compositions and no single x."""
    s = _system(tmp_path, monkeypatch)
    from aipf.data.index import _composition
    c = _composition(s, "slab_xl0.00_xr0.90_T02000", {})
    assert c["kind"] == "two_slab"
    assert c["x"]["He"] == [0.0, 0.9]

    u = _composition(s, "cube_x0.05_T02000", {})
    assert u["kind"] == "uniform"
    assert u["x"]["He"] == 0.05


def test_two_slab_composition_requires_both_sides(tmp_path, monkeypatch):
    """One side alone is not a coexistence pair.

    A tag with only ``xl`` is not a two-slab record: it has nothing to pair it
    with, so falling back to the uniform reading (which then finds no ``x`` at
    all) is correct, and silently treating a lone ``xl`` as a pair would
    record a composition that was never run.
    """
    s = _system(tmp_path, monkeypatch)
    from aipf.data.index import _composition
    c = _composition(s, "slab_xl0.00_T02000", {})
    assert c["kind"] == "uniform"


def test_composition_prefers_explicit_metadata_over_the_tag_guess(
        tmp_path, monkeypatch):
    """The tag regex is a fallback, not the source of truth.

    A composition recorded in the metadata itself must win over a value
    parsed back out of the directory name, since the metadata is what the run
    actually used and the tag is only ever a label for humans.
    """
    s = _system(tmp_path, monkeypatch)
    from aipf.data.index import _composition
    c = _composition(s, "cube_x0.05_T02000", {"x_He": 0.37})
    assert c["x"]["He"] == 0.37


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    s = _system(tmp_path, monkeypatch)
    manifest = index.build(s, dry_run=True)
    assert len(manifest["state_points"]) == 2
    assert not s.paths.data_root().exists()


def test_rebuild_from_manifest_restores_the_farm(tmp_path, monkeypatch):
    s = _system(tmp_path, monkeypatch)
    index.build(s)
    path = s.paths.data_root() / "manifest.json"

    import shutil
    shutil.rmtree(s.paths.data_root() / "md")
    assert index.rebuild_from_manifest(path) == 2
    assert (s.paths.data_root() / "md" / "800GPa" / "cube_x0.05_T02000"
            / "traj.lammpstrj").is_symlink()


def test_a_real_file_in_the_farm_is_refused_not_deleted(tmp_path, monkeypatch):
    """The farm holds links. A real file means data was copied in.

    Unlinking it could destroy the only copy of something, which is the one
    catastrophic failure available to this task, so it must refuse rather than
    refresh. Observe the RuntimeError, and observe the bytes survive.
    """
    s = _system(tmp_path, monkeypatch)
    index.build(s)
    victim = (s.paths.data_root() / "md" / "800GPa" / "cube_x0.05_T02000"
              / "traj.lammpstrj")
    victim.unlink()
    victim.write_text("irreplaceable")

    with pytest.raises(RuntimeError, match="not a symlink"):
        index.build(s)
    assert victim.read_text() == "irreplaceable"


def test_a_tag_reused_at_a_different_pressure_does_not_overwrite_the_first(
        tmp_path, monkeypatch):
    """A run tag is not always unique.

    One source tree runs the same composition and temperature again at a
    different pressure, under the identical tag string. The farm directory
    is keyed by pressure first, ``<P>GPa/<tag>``, so the two pressures land
    in distinct, human-readable directories with no counter involved: the
    variable that actually distinguishes the two records -- the pressure --
    is exactly what is used to tell them apart.
    """
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    tag = "cube_x0.20_T06000"
    for p_gpa in (200.0, 400.0):
        d = scratch / f"cube_data_{int(p_gpa)}GPa" / tag
        d.mkdir(parents=True)
        (d / "traj.lammpstrj").write_text(f"frames at {p_gpa} GPa")
        (d / f"{tag}_meta.json").write_text(json.dumps({
            "tag": tag, "T_K": 2000, "P_GPa": p_gpa,
            "n_atoms": 3456, "dt_ps": 0.0002, "dump_freq": 100,
            "n_prod": 150000, "status": "ok",
        }))

    manifest = index.build(s)
    records = [r for r in manifest["state_points"] if r["tag"] == tag]
    assert len(records) == 2
    farm_dirs = {r["farm_dir"] for r in records}
    assert farm_dirs == {f"200GPa/{tag}", f"400GPa/{tag}"}

    pressures_on_disk = set()
    for farm_dir in farm_dirs:
        meta = json.loads(
            (s.paths.data_root() / "md" / farm_dir / "meta.json").read_text())
        pressures_on_disk.add(meta["P_GPa"])
        link = s.paths.data_root() / "md" / farm_dir / "traj.lammpstrj"
        assert link.resolve().read_text() == f"frames at {meta['P_GPa']} GPa"
    assert pressures_on_disk == {200.0, 400.0}


def test_rebuild_preserves_the_pressure_keyed_farm_dirs(tmp_path, monkeypatch):
    """A rebuild must use each record's own ``farm_dir``, not reconstruct it.

    ``rebuild_from_manifest`` does not know how to derive ``<P>GPa/<tag>`` on
    its own -- and should not have to, since the manifest already carries the
    authoritative value. Reconstructing it independently would be a second
    place that has to agree with ``build``'s layout rule forever.
    """
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    tag = "cube_x0.30_T07000"
    for p_gpa in (600.0, 800.0):
        d = scratch / f"cube_data_{int(p_gpa)}GPa" / tag
        d.mkdir(parents=True)
        (d / "traj.lammpstrj").write_text(f"frames at {p_gpa} GPa")
        (d / f"{tag}_meta.json").write_text(json.dumps({
            "tag": tag, "T_K": 7000, "P_GPa": p_gpa,
            "n_atoms": 3456, "dt_ps": 0.0002, "dump_freq": 100,
            "n_prod": 150000, "status": "ok",
        }))

    index.build(s)
    path = s.paths.data_root() / "manifest.json"
    import shutil
    shutil.rmtree(s.paths.data_root() / "md")
    index.rebuild_from_manifest(path)

    manifest = index.read_manifest(path)
    farm_dirs = {r["farm_dir"] for r in manifest["state_points"]
                 if r["tag"] == tag}
    assert farm_dirs == {f"600GPa/{tag}", f"800GPa/{tag}"}
    for farm_dir in farm_dirs:
        assert (s.paths.data_root() / "md" / farm_dir
                / "traj.lammpstrj").is_symlink()


def test_build_is_order_independent_across_which_pressure_is_discovered_first(
        tmp_path, monkeypatch):
    """The disambiguated directory must not depend on filesystem walk order.

    ``build`` walks ``scratch.rglob("*")`` in sorted path order, so which of
    two colliding-tag records is discovered "first" depends on how their
    parent directories happen to sort. A counter keyed by discovery order
    would then give the two records different names depending on that
    accident of naming -- exactly the defect this round exists to remove.
    Two source trees are built with their parent directories named so the
    two records sort in opposite relative order, and every ``farm_dir`` must
    come out identical regardless, because it is derived only from each
    record's own pressure and tag, never from the order it was seen in.
    """
    tag = "cube_x0.45_T08000"

    def _make(tmp_subdir, first_parent, second_parent, first_p, second_p):
        root = tmp_subdir / "scratch"
        for parent, p_gpa in ((first_parent, first_p), (second_parent, second_p)):
            d = root / parent / tag
            d.mkdir(parents=True)
            (d / "traj.lammpstrj").write_text(f"frames at {p_gpa} GPa")
            (d / f"{tag}_meta.json").write_text(json.dumps({
                "tag": tag, "T_K": 8000, "P_GPa": p_gpa,
                "n_atoms": 3456, "dt_ps": 0.0002, "dump_freq": 100,
                "n_prod": 150000, "status": "ok",
            }))
        return root

    # Build A: the 200 GPa parent directory name sorts before the 800 GPa
    # one, so 200 GPa is discovered first.
    root_a = _make(tmp_path / "a", "aaa_200GPa", "zzz_800GPa", 200.0, 800.0)
    monkeypatch.setenv("AIPF_RAW_DEMO", str(root_a))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "a" / "data"))
    s_a = System(
        name="demo", n_species=2, species=("H", "He"),
        masses={"H": 1.008, "He": 4.0026}, atom_types={"H": 1, "He": 2},
        table_keys={"rho": ("rho_H", "rho_He"), "x": "x_He", "x_channel": 1},
        paths=Paths(system="demo"), anchor_rules=AnchorRules({}),
        constants={"P_GPa": 800.0}, defaults={},
    )
    manifest_a = index.build(s_a)

    # Build B: the SAME two pressures, but with parent directory names
    # reversed so 800 GPa now sorts, and is discovered, first.
    root_b = _make(tmp_path / "b", "aaa_800GPa", "zzz_200GPa", 800.0, 200.0)
    monkeypatch.setenv("AIPF_RAW_DEMO", str(root_b))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "b" / "data"))
    s_b = System(
        name="demo", n_species=2, species=("H", "He"),
        masses={"H": 1.008, "He": 4.0026}, atom_types={"H": 1, "He": 2},
        table_keys={"rho": ("rho_H", "rho_He"), "x": "x_He", "x_channel": 1},
        paths=Paths(system="demo"), anchor_rules=AnchorRules({}),
        constants={"P_GPa": 800.0}, defaults={},
    )
    manifest_b = index.build(s_b)

    # Both builds must place the 200 GPa record and the 800 GPa record under
    # exactly the same farm_dir, no matter which one was discovered first.
    records_a = {r["meta"]["P_GPa"]: r["farm_dir"]
                 for r in manifest_a["state_points"] if r["tag"] == tag}
    records_b = {r["meta"]["P_GPa"]: r["farm_dir"]
                 for r in manifest_b["state_points"] if r["tag"] == tag}
    assert records_a == records_b == {
        200.0: f"200GPa/{tag}", 800.0: f"800GPa/{tag}"}


def test_a_genuine_collision_at_the_same_pressure_raises_naming_both_sources(
        tmp_path, monkeypatch):
    """Two different source directories that both normalise to the same
    pressure and tag are a real ambiguity in the data, not something to
    paper over with an invented name. This must halt the build loudly and
    say which two directories are in conflict.
    """
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    tag = "cube_x0.15_T05000"
    first = scratch / "cube_data_800GPa" / tag
    second = scratch / "cube_data_800GPa_rerun" / tag
    for d in (first, second):
        d.mkdir(parents=True)
        (d / "traj.lammpstrj").write_text("frames")
        (d / f"{tag}_meta.json").write_text(json.dumps({
            "tag": tag, "T_K": 5000, "P_GPa": 800.0,
            "n_atoms": 3456, "dt_ps": 0.0002, "dump_freq": 100,
            "n_prod": 150000, "status": "ok",
        }))

    with pytest.raises(RuntimeError) as exc_info:
        index.build(s)
    message = str(exc_info.value)
    assert str(first) in message
    assert str(second) in message


def test_a_null_pressure_cannot_be_placed_and_is_a_loud_skip(
        tmp_path, monkeypatch):
    """A record with no recorded pressure has nowhere deterministic to live
    under the pressure-keyed layout. Rather than default it into some
    directory that would look like a real pressure, it is recorded as a
    skip, with a reason that names the actual problem.
    """
    # A system where pressure is not a controlled variable at all -- exactly
    # the situation `System` itself allows via `constants={"P_GPa": None}` --
    # so the metadata's own absence of P_GPa is not overridden by a fallback.
    root = _fake_scratch(tmp_path)
    monkeypatch.setenv("AIPF_RAW_DEMO", str(root))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "data"))
    s = System(
        name="demo", n_species=2, species=("H", "He"),
        masses={"H": 1.008, "He": 4.0026}, atom_types={"H": 1, "He": 2},
        table_keys={"rho": ("rho_H", "rho_He"), "x": "x_He", "x_channel": 1},
        paths=Paths(system="demo"), anchor_rules=AnchorRules({}),
        constants={"P_GPa": None}, defaults={},
    )
    scratch = s.paths.raw()
    tag = "cube_x0.25_T09000"
    d = scratch / "cube_data_unknownGPa" / tag
    d.mkdir(parents=True)
    (d / "traj.lammpstrj").write_text("frames")
    (d / f"{tag}_meta.json").write_text(json.dumps({
        "tag": tag, "T_K": 9000,   # P_GPa deliberately absent
        "n_atoms": 3456, "dt_ps": 0.0002, "dump_freq": 100,
        "n_prod": 150000, "status": "ok",
    }))

    manifest = index.build(s)
    skipped = {r["tag"]: r["skipped"] for r in manifest["state_points"]
               if "skipped" in r}
    assert tag in skipped
    assert "P_GPa" in skipped[tag]
    assert not (s.paths.data_root() / "md").exists() or not any(
        (s.paths.data_root() / "md").rglob(tag))


def test_pressure_dir_is_exact_for_the_real_trees_integer_pressures():
    from aipf.data.index import _pressure_dir
    assert _pressure_dir(200.0) == "200GPa"
    assert _pressure_dir(400.0) == "400GPa"
    assert _pressure_dir(600.0) == "600GPa"
    assert _pressure_dir(800.0) == "800GPa"

    # A large integer-valued pressure must still be the exact integer, not
    # `:g`'s six-significant-figure default, which would switch to
    # exponential notation here (`1.23457e+06GPa`) and silently mangle the
    # name. This is the one input where the two branches actually disagree,
    # so it is what proves the integer branch is really being taken.
    assert _pressure_dir(1234567.0) == "1234567GPa"


def test_pressure_dir_non_integer_fallback():
    """No real tree runs at a fractional pressure today, but the fallback
    branch still has to produce something reasonable rather than silently
    truncating to the wrong integer."""
    from aipf.data.index import _pressure_dir
    assert _pressure_dir(162.5) == "162.5GPa"


# --------------------------------------------------------------------------
# Skips are a required output, not a footnote. A state point silently absent
# from the farm is indistinguishable from one that never existed, so a
# failure to normalise a state point's metadata must be recorded, with its
# reason, and counted in the manifest -- never just dropped.
# --------------------------------------------------------------------------

def test_a_state_point_missing_a_required_field_is_recorded_as_a_skip(
        tmp_path, monkeypatch):
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    broken = scratch / "cube_data_800GPa" / "cube_x0.90_T04000"
    broken.mkdir(parents=True)
    (broken / "traj.lammpstrj").write_text("frames")
    (broken / "cube_x0.90_T04000_meta.json").write_text(json.dumps({
        "tag": "cube_x0.90_T04000", "T_K": 4000, "P_GPa": 800.0,
        "dt_ps": 0.001, "dump_every_ps": 0.02, "n_frames": 2001,
        "seed": 1, "status": "ok",
        # n_atoms deliberately missing, as happens on a job that was
        # killed before it could finish writing its own metadata.
    }))

    manifest = index.build(s)
    skipped = [r for r in manifest["state_points"] if "skipped" in r]
    assert len(skipped) == 1
    assert skipped[0]["tag"] == "cube_x0.90_T04000"
    assert "n_atoms" in skipped[0]["skipped"]
    # The link is never created for a state point that failed to normalise.
    assert not (s.paths.data_root() / "md" / "cube_x0.90_T04000").exists()


def test_an_unclassifiable_tag_is_recorded_as_a_skip_not_guessed(
        tmp_path, monkeypatch):
    """A directory name outside the known geometry prefixes must not be
    silently matched by loosening the rule. It is reported, by name."""
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    pilot = scratch / "pilots_A2" / "pilot_x0.95_T07500"
    pilot.mkdir(parents=True)
    (pilot / "traj.lammpstrj").write_text("frames")

    manifest = index.build(s)
    skipped = {r["tag"]: r["skipped"] for r in manifest["state_points"]
               if "skipped" in r}
    assert "pilot_x0.95_T07500" in skipped
    assert "classif" in skipped["pilot_x0.95_T07500"]


def test_manifest_carries_explicit_skip_and_index_counts(tmp_path, monkeypatch):
    """The counts belong in the manifest, not only recoverable by scanning
    ``state_points`` by hand, so a later reader sees at a glance how many
    state points were indexed and how many were not, and why."""
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    pilot = scratch / "pilots_A2" / "pilot_x0.95_T07500"
    pilot.mkdir(parents=True)
    (pilot / "traj.lammpstrj").write_text("frames")

    manifest = index.build(s)
    assert manifest["n_indexed"] == 2
    assert manifest["n_skipped"] == 1
    assert manifest["n_indexed"] + manifest["n_skipped"] == len(
        manifest["state_points"])


def test_a_normalised_record_that_fails_validation_is_a_skip(tmp_path, monkeypatch):
    """Passing ``_normalise`` is not enough. The result still has to validate.

    A non-positive temperature normalises cleanly -- it is a well formed
    number -- but ``aipf.data.meta.validate`` rejects it, and a record that
    fails that check must not reach the farm as if it were clean data.
    """
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    tag = "cube_x0.10_T00000"
    d = scratch / "cube_data_800GPa" / tag
    d.mkdir(parents=True)
    (d / "traj.lammpstrj").write_text("frames")
    (d / f"{tag}_meta.json").write_text(json.dumps({
        "tag": tag, "T_K": 0, "P_GPa": 800.0,   # T_K must be positive
        "n_atoms": 3456, "dt_ps": 0.0002, "dump_freq": 100,
        "n_prod": 150000, "status": "ok",
    }))

    manifest = index.build(s)
    skipped = {r["tag"]: r["skipped"] for r in manifest["state_points"]
               if "skipped" in r}
    assert tag in skipped
    assert "T_K" in skipped[tag]
    assert not (s.paths.data_root() / "md" / tag).exists()


def test_zero_gpa_is_a_real_pressure_not_a_falsy_skip(tmp_path, monkeypatch):
    """0.0 GPa is a real, recorded pressure, not the same as no pressure.

    ``meta["P_GPa"] is None`` is the correct null check, and every fixture in
    this file happens to use 800.0, so nothing here pins the zero case. A
    later refactor to the more natural-looking ``if not meta["P_GPa"]`` would
    treat 0.0 as falsy and silently skip every 0 GPa record -- exactly the
    pressure at which one real system's authoritative field trees sit.
    """
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    tag = "cube_x0.10_T02000"
    d = scratch / "cube_data_0GPa" / tag
    d.mkdir(parents=True)
    (d / "traj.lammpstrj").write_text("frames")
    (d / f"{tag}_meta.json").write_text(json.dumps({
        "tag": tag, "T_K": 2000, "P_GPa": 0.0,
        "n_atoms": 3456, "dt_ps": 0.0002, "dump_freq": 100,
        "n_prod": 150000, "status": "ok",
    }))

    manifest = index.build(s)
    records = [r for r in manifest["state_points"] if r["tag"] == tag]
    assert len(records) == 1
    assert "skipped" not in records[0]
    assert records[0]["farm_dir"] == f"0GPa/{tag}"
    link = s.paths.data_root() / "md" / "0GPa" / tag / "traj.lammpstrj"
    assert link.is_symlink()
    assert link.resolve().read_text() == "frames"


def test_a_system_with_no_declared_partition_places_records_under_the_tag_alone(
        tmp_path, monkeypatch):
    """A system that declares ``farm_partition=()`` is not consulted on
    pressure at all, even when a record happens to carry one.

    This is the fix for a system where a field like pressure is genuinely not
    a controlled variable: hardcoding pressure as THE partition key inside
    this module would make every one of that system's state points hit the
    null-pressure branch and be skipped, silently emptying its whole farm.
    Declaring no partition fields means the layout is plain ``md/<tag>`` and
    the never-controlled field is never even read.
    """
    root = _fake_scratch(tmp_path)
    monkeypatch.setenv("AIPF_RAW_DEMO", str(root))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "data"))
    s = System(
        name="demo", n_species=2, species=("H", "He"),
        masses={"H": 1.008, "He": 4.0026}, atom_types={"H": 1, "He": 2},
        table_keys={"rho": ("rho_H", "rho_He"), "x": "x_He", "x_channel": 1},
        paths=Paths(system="demo"), anchor_rules=AnchorRules({}),
        constants={"P_GPa": None}, defaults={},
        farm_partition=(),
    )
    manifest = index.build(s)
    ok = [r for r in manifest["state_points"] if "skipped" not in r]
    assert len(ok) == 2
    for r in ok:
        assert r["farm_dir"] == r["tag"]
        assert "/" not in r["farm_dir"]
    link = (s.paths.data_root() / "md" / "cube_x0.05_T02000"
            / "traj.lammpstrj")
    assert link.is_symlink()


def test_farm_partition_with_two_fields_renders_in_declaration_order(
        tmp_path, monkeypatch):
    """A multi-field partition is joined in the order it was declared, not
    reversed or alphabetised. No real system declares more than one field
    today, so nothing else in this file distinguishes declaration order from
    its reverse; this is the test that does.
    """
    root = _fake_scratch(tmp_path)
    monkeypatch.setenv("AIPF_RAW_DEMO", str(root))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "data"))
    s = System(
        name="demo", n_species=2, species=("H", "He"),
        masses={"H": 1.008, "He": 4.0026}, atom_types={"H": 1, "He": 2},
        table_keys={"rho": ("rho_H", "rho_He"), "x": "x_He", "x_channel": 1},
        paths=Paths(system="demo"), anchor_rules=AnchorRules({}),
        constants={"P_GPa": 800.0}, defaults={},
        farm_partition=("P_GPa", "T_K"),
    )
    manifest = index.build(s)
    records = {r["tag"]: r for r in manifest["state_points"]
               if "skipped" not in r}
    assert records["cube_x0.05_T02000"]["farm_dir"] == \
        "800GPa/2000.0/cube_x0.05_T02000"


def test_farm_partition_defaults_to_pressure_for_an_undeclared_system(tmp_path,
                                                                        monkeypatch):
    """Every existing ``System(...)`` call in this file omits ``farm_partition``
    on purpose, so the default has to be the pressure-keyed layout this whole
    file already asserts against, or every other test here would break.
    """
    s = _system(tmp_path, monkeypatch)
    assert s.farm_partition == ("P_GPa",)


def test_md_exclude_dirs_is_a_loud_skip_not_a_silent_omission(tmp_path, monkeypatch):
    """A system may declare part of its tree as not production data.

    Found against a real tree: a diagnostic quality-control workspace reused
    a production tag at the identical pressure for its own decimated re-dump,
    which otherwise raises the genuine-collision error on every real build.
    Declaring the top-level directory in ``constants["md_exclude_dirs"]``
    keeps that directory out of the candidate pool entirely, but it must
    still show up in the manifest with its own reason -- excluding it is not
    the same failure mode as never having found it.
    """
    root = _fake_scratch(tmp_path)
    diagnostic_tag = "cube_x0.05_T02000"          # same tag as a real record
    d = root / "qc" / "cadence_test" / "dumps" / diagnostic_tag
    d.mkdir(parents=True)
    (d / "traj.lammpstrj").write_text("decimated frames")
    (d / f"{diagnostic_tag}_meta.json").write_text(json.dumps({
        "tag": diagnostic_tag, "T_K": 2000, "P_GPa": 800.0,
        "n_atoms": 3456, "dt_ps": 0.001, "dump_every_ps": 0.02,
        "n_frames": 501, "seed": 1, "status": "ok",
    }))
    monkeypatch.setenv("AIPF_RAW_DEMO", str(root))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "data"))
    s = System(
        name="demo", n_species=2, species=("H", "He"),
        masses={"H": 1.008, "He": 4.0026}, atom_types={"H": 1, "He": 2},
        table_keys={"rho": ("rho_H", "rho_He"), "x": "x_He", "x_channel": 1},
        paths=Paths(system="demo"), anchor_rules=AnchorRules({}),
        constants={"P_GPa": 800.0, "md_exclude_dirs": ("qc",)},
        defaults={},
    )

    manifest = index.build(s)         # would raise a collision if not excluded
    excluded = [r for r in manifest["state_points"]
                if r["tag"] == diagnostic_tag and "skipped" in r]
    assert len(excluded) == 1
    assert "'qc'" in excluded[0]["skipped"]
    assert "md_exclude_dirs" in excluded[0]["skipped"]
    # the real record under the same tag is unaffected
    kept = [r for r in manifest["state_points"]
            if r["tag"] == diagnostic_tag and "skipped" not in r]
    assert len(kept) == 1


def test_rebuild_from_manifest_never_rebuilds_a_skipped_record(
        tmp_path, monkeypatch):
    """A skip in the manifest carries no trajectory path to rebuild from.

    Rebuilding must count and act only on the indexed records, or it either
    crashes on a skip's missing keys or, worse, manufactures a farm entry for
    a state point that was never actually indexed.
    """
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    pilot = scratch / "pilots_A2" / "pilot_x0.95_T07500"
    pilot.mkdir(parents=True)
    (pilot / "traj.lammpstrj").write_text("frames")

    index.build(s)
    path = s.paths.data_root() / "manifest.json"
    n = index.rebuild_from_manifest(path)
    assert n == 2
    assert not (s.paths.data_root() / "md" / "pilot_x0.95_T07500").exists()


# --------------------------------------------------------------------------
# Nothing may be invisible. A trajectory-shaped file that matches no known
# name used to vanish entirely -- neither indexed nor skipped, appearing in
# no manifest at all. Every one now lands in exactly one of: indexed,
# skipped (both via `state_points`, one recognised file each), or
# `unrecognised`, with its filename.
# --------------------------------------------------------------------------

def test_traj_names_default_matches_the_first_two_systems():
    """No system.py declares traj_names, so the default is what build() used
    to hardcode as a module constant: the two systems that never declare it
    are unaffected by its existence."""
    s = System(
        name="demo", n_species=1, species=("A",), masses={"A": 1.0},
        atom_types={"A": 1}, table_keys={}, paths=Paths(system="demo"),
        anchor_rules=AnchorRules({}), constants={}, defaults={},
    )
    assert s.traj_names == ("traj.lammpstrj", "traj.dump", "dump.lammpstrj")
    assert s.tag_from_path is False
    assert s.tag_path_root is None


def test_a_system_declaring_its_own_traj_names_is_used_instead(
        tmp_path, monkeypatch):
    """A tree that spells its trajectory files differently is still found,
    without the indexer guessing at a filename convention it cannot know."""
    root = tmp_path / "scratch"
    d = root / "cube_data_800GPa" / "cube_x0.10_T02000"
    d.mkdir(parents=True)
    (d / "run.custom_trj").write_text("frames")   # NOT one of the defaults
    (d / "cube_x0.10_T02000_meta.json").write_text(json.dumps({
        "tag": "cube_x0.10_T02000", "T_K": 2000, "P_GPa": 800.0,
        "n_atoms": 3456, "dt_ps": 0.0002, "dump_freq": 100,
        "n_prod": 150000, "status": "ok",
    }))
    monkeypatch.setenv("AIPF_RAW_DEMO", str(root))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "data"))
    s = System(
        name="demo", n_species=2, species=("H", "He"),
        masses={"H": 1.008, "He": 4.0026}, atom_types={"H": 1, "He": 2},
        table_keys={"rho": ("rho_H", "rho_He"), "x": "x_He", "x_channel": 1},
        paths=Paths(system="demo"), anchor_rules=AnchorRules({}),
        constants={"P_GPa": 800.0}, defaults={},
        traj_names=("run.custom_trj",),
    )
    manifest = index.build(s, dry_run=True)
    ok = [r for r in manifest["state_points"] if "skipped" not in r]
    assert len(ok) == 1
    assert manifest["n_unrecognised"] == 0


def test_unrecognised_trajectory_shaped_file_beside_a_recognised_one(
        tmp_path, monkeypatch):
    """The real-tree shape this rule exists for: one directory holding both
    the recognised trajectory (indexed normally) and an extra file that
    merely looks like one -- a downsampled copy under a different name, on
    the real tree. The extra file must be visible, not silently dropped, and
    must not be double-indexed as a second candidate."""
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    tag = "cube_x0.05_T02000"
    d = scratch / "cube_data_800GPa" / tag
    extra = d / "traj_ds1000.lammpstrj"
    extra.write_text("a downsampled extra copy, not a run of its own")

    manifest = index.build(s, dry_run=True)
    ok = [r for r in manifest["state_points"] if "skipped" not in r]
    assert len(ok) == 2                      # the two real _fake_scratch records
    assert manifest["n_unrecognised"] == 1
    assert manifest["unrecognised"] == [{"file": str(extra)}]


def test_a_directory_with_only_an_unrecognised_trajectory_shaped_file(
        tmp_path, monkeypatch):
    """A directory that holds NO recognised trajectory name at all must
    still not be silently absent: it contributes no state point (there is
    nothing to index or skip), but its file is recorded, by name."""
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    stray = scratch / "builds" / "validate" / "forces.dump"
    stray.parent.mkdir(parents=True)
    stray.write_text("not a trajectory this system recognises")

    manifest = index.build(s, dry_run=True)
    assert manifest["n_unrecognised"] == 1
    assert manifest["unrecognised"] == [{"file": str(stray)}]
    # It contributes no state_points entry: it is not a candidate directory.
    assert all("forces" not in r["tag"] for r in manifest["state_points"])


def test_indexed_plus_skipped_plus_unrecognised_equals_files_found(
        tmp_path, monkeypatch):
    """The invariant this whole rule exists to guarantee, checked end to end
    against a tree that mixes all three outcomes: two indexed records (from
    ``_fake_scratch``), one unrecognised file beside a recognised one, and
    one unrecognised file with no recognised sibling at all."""
    s = _system(tmp_path, monkeypatch)
    scratch = s.paths.raw()
    (scratch / "cube_data_800GPa" / "cube_x0.05_T02000"
     / "extra.lammpstrj").write_text("beside a recognised trajectory")
    lone = scratch / "misc" / "forces.dump"
    lone.parent.mkdir(parents=True)
    lone.write_text("no recognised sibling")

    manifest = index.build(s, dry_run=True)
    n_found = sum(
        1 for p in scratch.rglob("*")
        if p.is_file() and p.suffix in (".lammpstrj", ".dump"))
    assert n_found == 4    # 2 recognised (the fixture) + 2 unrecognised
    assert (manifest["n_indexed"] + manifest["n_skipped"]
            + manifest["n_unrecognised"]) == n_found


def test_build_raises_if_its_own_accounting_does_not_add_up(
        tmp_path, monkeypatch):
    """The invariant this whole rule exists to guarantee is checked by
    ``build`` itself, not only by a test with fixture data that happens to
    satisfy it by construction. Forcing ``_find`` to return a path that is
    not actually one of the trajectory-shaped files on disk breaks the
    invariant directly: the real file is then counted as unrecognised (it is
    not the object ``_find`` "picked"), while the fake path is still treated
    as a found trajectory, so accounted (2) no longer equals files found (1),
    and ``build`` must refuse to return a manifest it cannot account for."""
    s = _system(tmp_path, monkeypatch)
    monkeypatch.setattr(
        index, "_find",
        lambda directory, names: directory / "not_actually_on_disk.lammpstrj")
    with pytest.raises(RuntimeError, match="accounting mismatch"):
        index.build(s, dry_run=True)


def test_quench_ensemble_is_overdamped_not_nvt():
    """Verified against a real quench tree's own log.lammps: the dump that is
    actually read starts only after the velocity-Langevin melt stage is
    unfixed, once the script has switched to the overdamped Brownian fix for
    its production stage, so the ensemble that describes the recorded frames
    is Brownian, not NVT."""
    from aipf.data.index import _classify
    assert _classify("quench_MD_xA_0.50_T_1.30_seed_2") == (
        "quench", "langevin_overdamped")


# --------------------------------------------------------------------------
# Tag reconstruction from the path, for a tree whose run directory is a
# purely structural name -- "MD", "seed_<n>" -- with the real identity one
# or more levels further up.
# --------------------------------------------------------------------------

def test_reconstruct_tag_uses_the_leaf_name_by_default(tmp_path, monkeypatch):
    from aipf.data.index import _reconstruct_tag
    s = _system(tmp_path, monkeypatch)
    run_dir = s.paths.raw() / "cube_data_800GPa" / "cube_x0.05_T02000"
    assert _reconstruct_tag(s, run_dir, s.paths.raw()) == "cube_x0.05_T02000"


def test_reconstruct_tag_flattens_the_path_when_declared(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    monkeypatch.setenv("AIPF_RAW_DEMO", str(root))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "data"))
    s = System(
        name="demo", n_species=1, species=("A",), masses={"A": 1.0},
        atom_types={"A": 1}, table_keys={}, paths=Paths(system="demo"),
        anchor_rules=AnchorRules({}), constants={}, defaults={},
        tag_from_path=True, tag_path_root="Data",
    )
    from aipf.data.index import _reconstruct_tag
    run_dir = root / "Data" / "slab_overdamped" / "MD" / "xA_0.50" / "T_1.15" / "seed_43"
    assert _reconstruct_tag(s, run_dir, root) == \
        "slab_overdamped_MD_xA_0.50_T_1.15_seed_43"


def test_reconstruct_tag_path_root_only_drops_a_matching_prefix(
        tmp_path, monkeypatch):
    """A candidate outside the declared root has nothing of that name to
    drop, so it must not be truncated as though it did -- this is the real
    tree's own ``nucl_T1.10/...`` campaign, which sits directly under
    scratch, never under ``Data/``."""
    root = tmp_path / "scratch"
    monkeypatch.setenv("AIPF_RAW_DEMO", str(root))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "data"))
    s = System(
        name="demo", n_species=1, species=("A",), masses={"A": 1.0},
        atom_types={"A": 1}, table_keys={}, paths=Paths(system="demo"),
        anchor_rules=AnchorRules({}), constants={}, defaults={},
        tag_from_path=True, tag_path_root="Data",
    )
    from aipf.data.index import _reconstruct_tag
    run_dir = root / "nucl_T1.10" / "parents" / "s42"
    assert _reconstruct_tag(s, run_dir, root) == "nucl_T1.10_parents_s42"


# --------------------------------------------------------------------------
# Metadata recovered from a LAMMPS log, for a tree that writes no meta.json.
# --------------------------------------------------------------------------

_LOG_TWO_STAGE = """\
variable    T_high      index 2.0
variable    T_target    index 1.10
variable    dt_sim      index 0.0002

units       lj

timestep    ${dt_equil}
timestep    0.001

fix         lang_high all langevin ${T_high} ${T_high} ${damp} ${seed}
fix         lang_high all langevin 2.0 ${T_high} ${damp} ${seed}
fix         lang_high all langevin 2.0 2.0 ${damp} ${seed}
fix         lang_high all langevin 2.0 2.0 0.2 ${seed}
fix         lang_high all langevin 2.0 2.0 0.2 6300003

run         ${N_equil}
run         200000

Created 6912 atoms

unfix       lang_high

timestep    ${dt_sim}
timestep    0.0002

fix         bd  all brownian ${T_target} ${seed} gamma_t ${gamma_t}
fix         bd  all brownian 1.3 ${seed} gamma_t ${gamma_t}
fix         bd  all brownian 1.3 6300003 gamma_t ${gamma_t}
fix         bd  all brownian 1.3 6300003 gamma_t 2.0

dump        quench all custom ${dump_every} ${save_dir}/dump.quench.lammpstrj id type x y z
dump        quench all custom 5000 ${save_dir}/dump.quench.lammpstrj id type x y z
dump        quench all custom 5000 /abs/path/dump.quench.lammpstrj id type x y z

run         ${N_prod}
run         50000000
"""


def test_meta_from_log_prefers_the_production_stage_over_the_melt_stage(
        tmp_path):
    """The two-stage shape verified on the real tree: a high-temperature
    Langevin melt (T=2.0) followed by the overdamped-Brownian production
    stage the trajectory actually dumps (T=1.3). The melt temperature must
    not win just because it is textually declared first."""
    from aipf.data.index import _meta_from_log
    log = tmp_path / "log.lammps"
    log.write_text(_LOG_TWO_STAGE)
    src = _meta_from_log(log)
    assert src["T_K"] == 1.3
    assert src["dt_ps"] == 0.0002          # the production timestep, not 0.001
    assert src["n_prod"] == 50000000       # the production run, not the melt
    assert src["dump_freq"] == 5000
    assert src["n_atoms"] == 6912
    assert src["seed"] == 6300003
    assert src["engine"] == "lammps"


def test_meta_from_log_single_stage_brownian_only(tmp_path):
    from aipf.data.index import _meta_from_log
    log = tmp_path / "log.lammps"
    log.write_text(
        "timestep    ${dt_sim}\n"
        "timestep    0.0002\n"
        "fix         bd  all brownian ${T_target} ${seed} gamma_t ${gamma_t}\n"
        "fix         bd  all brownian 1.15 ${seed} gamma_t ${gamma_t}\n"
        "fix         bd  all brownian 1.15 43 gamma_t ${gamma_t}\n"
        "fix         bd  all brownian 1.15 43 gamma_t 2.0\n"
        "  reading atoms ...\n"
        "  3456 atoms\n"
        "dump        d1 all custom ${dump_every} ${out}/dump.lammpstrj id type x y z\n"
        "dump        d1 all custom 5000 ${out}/dump.lammpstrj id type x y z\n"
        "dump        d1 all custom 5000 /abs/dump.lammpstrj id type x y z\n"
        "run         ${N_prod}\n"
        "run         100000000\n"
    )
    src = _meta_from_log(log)
    assert src["T_K"] == 1.15
    assert src["n_atoms"] == 3456
    assert src["dt_ps"] == 0.0002
    assert src["seed"] == 43
    assert src["n_prod"] == 100000000
    assert src["n_equil"] == 0             # one run: no preparation stage


def test_meta_from_log_counts_every_run_before_the_last_as_preparation(
        tmp_path):
    from aipf.data.index import _meta_from_log
    log = tmp_path / "log.lammps"
    log.write_text(_LOG_TWO_STAGE)
    src = _meta_from_log(log)
    runs = [int(line.split()[1]) for line in _LOG_TWO_STAGE.splitlines()
            if line.startswith("run") and line.split()[1].isdigit()]
    assert len(runs) >= 2
    assert src["n_equil"] == sum(runs[:-1])
    assert src["n_prod"] == runs[-1]


def test_meta_from_log_dump_cadence_takes_the_last_dump_block(tmp_path):
    """Two dump blocks -- an equilibration dump at one cadence, replaced by a
    production dump at another -- must resolve to the LAST one, the same
    last-wins rule as every other pattern this function reads."""
    from aipf.data.index import _meta_from_log
    log = tmp_path / "log.lammps"
    log.write_text(
        "dump        eq all custom ${eq_every} ${d}/dump.eq.lammpstrj id type x y z\n"
        "dump        eq all custom 20000 ${d}/dump.eq.lammpstrj id type x y z\n"
        "dump        eq all custom 20000 /abs/dump.eq.lammpstrj id type x y z\n"
        "undump      eq\n"
        "dump        prod all custom ${dump_every} ${d}/dump.lammpstrj id type x y z\n"
        "dump        prod all custom 5000 ${d}/dump.lammpstrj id type x y z\n"
        "dump        prod all custom 5000 /abs/dump.lammpstrj id type x y z\n"
    )
    src = _meta_from_log(log)
    assert src["dump_freq"] == 5000


def test_meta_from_log_langevin_uses_the_thermostat_target_not_the_start(
        tmp_path):
    """``fix ID group-ID langevin Tstart Tend damp seed`` -- a ramped stage's
    TARGET (Tend) is what the run actually holds at for the rest of the
    script, not the value it started from."""
    from aipf.data.index import _meta_from_log
    log = tmp_path / "log.lammps"
    log.write_text(
        "fix     lang all langevin ${Ts} ${Te} ${damp} ${seed}\n"
        "fix     lang all langevin 3.0 ${Te} ${damp} ${seed}\n"
        "fix     lang all langevin 3.0 1.4 ${damp} ${seed}\n"
        "fix     lang all langevin 3.0 1.4 0.2 ${seed}\n"
        "fix     lang all langevin 3.0 1.4 0.2 99\n"
    )
    src = _meta_from_log(log)
    assert src["T_K"] == 1.4


def test_meta_from_log_atom_count_takes_the_textually_last_occurrence(
        tmp_path):
    """"Created N atoms" (create_atoms) and "N atoms" under "reading atoms
    ..." (read_data) are two different LAMMPS code paths, and a script may
    use one to build an initial configuration and the other, later, to read
    a snapshot of a different size -- the later one is the one that actually
    ran for the rest of the script."""
    from aipf.data.index import _meta_from_log
    log = tmp_path / "log.lammps"
    log.write_text(
        "create_atoms 1 box\n"
        "Created 4000 atoms\n"
        "read_data   restart.data\n"
        "  reading atoms ...\n"
        "  6912 atoms\n"
    )
    src = _meta_from_log(log)
    assert src["n_atoms"] == 6912


def test_meta_from_log_returns_only_what_it_finds(tmp_path):
    """Nothing about this function invents a value. A log carrying none of
    the recognised patterns comes back with only the ``engine`` key, so the
    existing required-field check reports the real missing key rather than
    this function guessing one."""
    from aipf.data.index import _meta_from_log
    log = tmp_path / "log.lammps"
    log.write_text("LAMMPS (10 Dec 2025 - Development)\n")
    src = _meta_from_log(log)
    assert src == {"engine": "lammps"}


def test_meta_from_log_tolerates_non_ascii_bytes(tmp_path):
    """LAMMPS itself warns about and 'tries to continue' past non-ASCII bytes
    in a comment; a strict-decode here would crash on a real log this module
    must still be able to read."""
    from aipf.data.index import _meta_from_log
    log = tmp_path / "log.lammps"
    log.write_bytes(
        b"# Pair interactions \xe2\x80\x94 Das et al. 2006\n"
        b"timestep    0.0002\n")
    src = _meta_from_log(log)   # must not raise
    assert src["dt_ps"] == 0.0002


def test_source_meta_prefers_meta_json_over_the_log(tmp_path):
    d = tmp_path
    (d / "log.lammps").write_text(
        "fix bd all brownian 9.99 1 gamma_t 2.0\n"
        "timestep 0.0002\n")
    (d / "meta.json").write_text(json.dumps({"T_K": 1.15, "n_atoms": 3456}))
    from aipf.data.index import _source_meta
    src = _source_meta(d, "irrelevant_tag")
    assert src["T_K"] == 1.15          # the hand-written value, not the log's


def test_source_meta_falls_back_to_the_log_when_no_meta_json(tmp_path):
    d = tmp_path
    (d / "log.lammps").write_text(
        "fix bd all brownian 1.15 43 gamma_t 2.0\n"
        "timestep 0.0002\n")
    from aipf.data.index import _source_meta
    src = _source_meta(d, "irrelevant_tag")
    assert src["T_K"] == 1.15
    assert src["engine"] == "lammps"


def test_source_meta_is_empty_when_neither_exists(tmp_path):
    from aipf.data.index import _source_meta
    assert _source_meta(tmp_path, "irrelevant_tag") == {}


# --- the farm's tiers -----------------------------------------------------
#
# Local helpers rather than fixtures: this directory has no ``conftest.py``
# and each file carries its own ``_demo``-style builder (see
# ``tests/unit/test_system.py``).


def _demo_manifest(tmp_path):
    """A two-record manifest written where the farm keeps one.

    The location matters to what is under test: ``rebuild_from_manifest``
    writes the md tier beside the manifest, so the path has to be the real
    ``<farm>/<sys>/manifest.json`` shape and not just any file.
    """
    src = tmp_path / "scratch" / "runs"
    src.mkdir(parents=True)
    records = []
    for tag, T in (("cube_x0.05_T02000", 2000), ("cube_x0.50_T03000", 3000)):
        traj = src / f"{tag}.lammpstrj"
        traj.write_text("frames")
        records.append({
            "tag": tag,
            "farm_dir": f"P800/{tag}",
            "trajectory": str(traj),
            "meta": {"tag": tag, "T_K": T, "P_GPa": 800.0},
        })
    out = tmp_path / "data" / "demo"
    out.mkdir(parents=True)
    path = out / "manifest.json"
    path.write_text(json.dumps({"system": "demo", "state_points": records}))
    return path


def test_modes_dir_mirrors_the_md_partition():
    from test_system import _demo
    d = index.modes_dir(_demo(), "P800/cube_x0.05_T02000")
    assert d.parts[-4:] == ("demo", "modes", "P800", "cube_x0.05_T02000")


def test_the_tiers_are_siblings_under_one_system_directory():
    from test_system import _demo
    system = _demo()
    ckpt = index.ckpt_dir(system)
    modes = index.modes_dir(system, "P800/tag")
    diagnose = index.diagnose_dir(system)
    assert ckpt.name == "ckpt" and diagnose.name == "diagnose"
    assert ckpt.parent == diagnose.parent == modes.parent.parent.parent
    assert ckpt.parent.name == "demo"


def test_tier_paths_follow_the_data_root_the_system_resolves(tmp_path, monkeypatch):
    """With no explicit root the new tiers land where ``build`` writes, which
    ``AIPF_DATA`` can move. Recomputing the repository root would ignore it."""
    from test_system import _demo
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "elsewhere"))
    system = _demo()
    assert index.ckpt_dir(system) == tmp_path / "elsewhere" / "demo" / "ckpt"
    assert index.diagnose_dir(system) == tmp_path / "elsewhere" / "demo" / "diagnose"
    assert (index.modes_dir(system, "P800/tag")
            == tmp_path / "elsewhere" / "demo" / "modes" / "P800" / "tag")


def test_rebuild_replays_the_md_tier_and_nothing_else(tmp_path):
    """The published checkpoints are tracked files of the checkout: the farm links none."""
    n = index.rebuild_from_manifest(_demo_manifest(tmp_path))
    assert n == 2
    farm = tmp_path / "data" / "demo"
    assert (farm / "md" / "P800" / "cube_x0.05_T02000" / "traj.lammpstrj").is_symlink()
    assert not (farm / "ckpt").exists()


# ---------------------------------------------------------------------------
# looking a state point up by the name it is filed under
# ---------------------------------------------------------------------------

def _demo_lookup(tmp_path):
    """The two-record manifest, plus a skipped entry and a tag collision."""
    from test_system import _demo
    path = _demo_manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["state_points"] += [
        {"tag": "cube_x0.05_T02000", "farm_dir": "P200/cube_x0.05_T02000",
         "trajectory": "elsewhere", "meta": {"T_K": 2000.0, "P_GPa": 200.0}},
        {"tag": "cube_broken_T09000", "skipped": "no metadata"},
    ]
    path.write_text(json.dumps(payload))
    return _demo(), tmp_path / "data"


def test_a_state_point_is_found_by_its_farm_directory(tmp_path):
    system, root = _demo_lookup(tmp_path)
    record = index.record_for_tag(system, "P800/cube_x0.50_T03000", root=root)
    assert record["tag"] == "cube_x0.50_T03000"
    assert record["meta"]["T_K"] == 3000


def test_a_state_point_is_found_by_its_bare_tag_when_that_is_unambiguous(
        tmp_path):
    system, root = _demo_lookup(tmp_path)
    assert index.record_for_tag(system, "cube_x0.50_T03000",
                                root=root)["farm_dir"] == "P800/cube_x0.50_T03000"


def test_a_tag_two_partitions_share_is_refused_not_answered(tmp_path):
    """The same run tag exists at more than one pressure in the real farm.

    Answering with whichever came first would hand back a different state
    point than the caller meant, and nothing in the record would say so.
    """
    system, root = _demo_lookup(tmp_path)
    with pytest.raises(KeyError, match="names 2 state points"):
        index.record_for_tag(system, "cube_x0.05_T02000", root=root)
    assert index.record_for_tag(system, "P800/cube_x0.05_T02000",
                                root=root)["trajectory"] != "elsewhere"


def test_an_unknown_tag_names_the_nearest_ones_that_do_exist(tmp_path):
    system, root = _demo_lookup(tmp_path)
    with pytest.raises(KeyError, match="cube_x0.50_T03000"):
        index.record_for_tag(system, "P800/cube_x0.50_T03001", root=root)


def test_a_skipped_state_point_is_never_returned_by_a_lookup(tmp_path):
    """It carries no trajectory, so returning it would hand back a record
    whose one useful key does not exist."""
    system, root = _demo_lookup(tmp_path)
    with pytest.raises(KeyError):
        index.record_for_tag(system, "cube_broken_T09000", root=root)


def test_a_lookup_without_a_manifest_says_to_build_the_index(tmp_path):
    from test_system import _demo
    with pytest.raises(FileNotFoundError, match="no manifest"):
        index.record_for_tag(_demo(), "P800/anything", root=tmp_path / "data")


def test_a_pattern_takes_the_first_match_in_manifest_order(tmp_path):
    system, root = _demo_lookup(tmp_path)
    assert index.record_for_pattern(system, "P800/cube_x0.05*",
                                    root=root)["farm_dir"] == "P800/cube_x0.05_T02000"
    assert index.record_for_pattern(system, "*T03000",
                                    root=root)["tag"] == "cube_x0.50_T03000"


def test_a_pattern_nothing_matches_is_refused(tmp_path):
    system, root = _demo_lookup(tmp_path)
    with pytest.raises(KeyError, match="matching"):
        index.record_for_pattern(system, "P900/*", root=root)


@pytest.mark.parametrize("tag, x", [
    ("cube_x0.05_T02000", 0.05),
    ("slab_overdamped_MD_xA_0.50_T_1.15_seed_43", 0.5),
    ("slab_xl0.05_T02000", None),
])
def test_a_uniform_composition_is_read_off_the_tag_with_or_without_its_species(
        tag, x):
    from aipf.data.index import _composition
    from aipf.system import load
    system = load("lj")
    assert _composition(system, tag, {})["x"] == {"A": x}
