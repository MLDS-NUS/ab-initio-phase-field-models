"""The seams a replay of the workflow guide found: one test group per item that is code.

The published checkpoints' own seams are in ``test_published_checkpoints.py``. Nothing here runs
LAMMPS or trains.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from aipf.cli.main import main
from aipf.data import index
from aipf.paths import PUBLISHED_DIRNAME, TRACKED_FILE_NAME, Paths
from aipf.system import AnchorRules, Checkpoint, System, load

import declared_roots

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_generic_scratch(monkeypatch):
    monkeypatch.delenv("AIPF_RAW", raising=False)


# ---------------------------------------------------------------------------
# item 2: the tracked files are the checkout's, whatever AIPF_DATA says
# ---------------------------------------------------------------------------

def test_published_resolves_from_the_checkout_when_the_farm_is_elsewhere(monkeypatch, tmp_path):
    declared_roots.published_or_skip("lj")
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "farm"))
    system = load("lj")
    assert system.paths.data_root() == tmp_path / "farm" / "lj"
    assert system.paths.published() == REPO / "data" / "lj" / "ckpt" / PUBLISHED_DIRNAME
    assert system.resolve_checkpoint() == REPO / "data" / "lj" / "ckpt" / PUBLISHED_DIRNAME / TRACKED_FILE_NAME


def test_fit_refuses_unreachable_weights_before_it_makes_a_run_directory(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from aipf.train.fit import fit

    monkeypatch.setenv("AIPF_RAW_LJ", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    nowhere = Checkpoint(path="runs/r/final.ckpt", md5=hashlib.md5(b"no such bytes").hexdigest())
    with pytest.raises((FileNotFoundError, ValueError), match="system 'lj'"):
        fit(load("lj"), run_name="r", sources=[object()], seed=0, steps=1,
            resume_optimizer=False, init_from=nowhere, root=tmp_path / "farm")
    assert not (tmp_path / "farm" / "lj" / "ckpt" / "r").exists()


# ---------------------------------------------------------------------------
# items 3, 4, 13: the farm index
# ---------------------------------------------------------------------------

def _one_field(tmp_path, monkeypatch) -> System:
    monkeypatch.setenv("AIPF_RAW_TOY", str(tmp_path / "scratch"))
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "farm"))
    (tmp_path / "scratch").mkdir(exist_ok=True)
    return System(
        name="toy", n_species=1, species=("A",), masses={"A": 1.0}, atom_types={"A": 1},
        table_keys={"rho": ("rho_A",), "x": "x_A", "x_channel": 0},
        paths=Paths(system="toy"), anchor_rules=AnchorRules({}),
        constants={"kB": 1.0, "P_GPa": None}, farm_partition=(),
        traj_names=("dump.lammpstrj",), tag_from_path=True, defaults={})


_LOG = ("timestep        0.001\n"
        "Created 500 atoms\n"
        "run             1000\n"
        "timestep        0.0002\n"
        "fix             bd all brownian 1.6 42 gamma_t 2.0\n"
        "dump            traj all custom 100 dump.lammpstrj id type x y z\n"
        "dump_modify     traj sort id first yes\n"
        "run             50000\n")


def _md_run_dir(root: Path, name: str, *, n_atoms=500, dry_run=False, system="toy") -> Path:
    """A run directory as ``aipf md run`` leaves it: dump, log and ``run.json``."""
    from aipf.md.request import StatePoint

    point = StatePoint(geometry="cube", ensemble="langevin_overdamped", T=1.6, x=0.5, dt_ps=0.0002,
                       equil_ps=1.0, prod_ps=10.0, dump_every_ps=0.02, dump_from="prod", seed=42,
                       n_atoms=n_atoms)
    run = root / name
    run.mkdir(parents=True)
    (run / "dump.lammpstrj").write_text("frames")
    (run / "log.lammps").write_text(_LOG)
    (run / "run.json").write_text(json.dumps({
        "system": system, "template": "cube-overdamped", "campaign": None, "point": asdict(point),
        "values": {}, "deck": "in.lammps", "command": "lmp", "cwd": str(run), "dry_run": dry_run,
        "diagnosis": [], "result": None if dry_run else {"status": "ok"}}))
    return run


def test_the_index_reads_run_json_not_the_directory_name(tmp_path, monkeypatch):
    toy = _one_field(tmp_path, monkeypatch)
    run = _md_run_dir(tmp_path / "scratch", "anything-at-all")
    manifest = index.build(toy)
    [record] = manifest["state_points"]
    meta = record["meta"]
    assert record["tag"] == meta["tag"] == "cube_x0.50_T1.6_s42"
    assert (meta["geometry"], meta["ensemble"]) == ("cube", "langevin_overdamped")
    assert meta["composition"] == {"species": ["A"], "kind": "uniform", "x": {"A": 0.5}}
    assert (meta["T_K"], meta["n_atoms"], meta["seed"], meta["status"]) == (1.6, 500, 42, "ok")
    assert (meta["dt_ps"], meta["dump_every_ps"], meta["n_frames"]) == (0.0002, 0.02, 501)
    assert (meta["extra"]["n_equil"], meta["extra"]["n_prod"]) == (1000, 50000)
    assert record["trajectory"] == str(run / "dump.lammpstrj")


@pytest.mark.parametrize("box_line,edge", [
    ("Created orthogonal box = (0 0 0) to (32.039007 32.039007 32.039007)", 32.039007),
    ("Created orthogonal box = (0 0 0) to (9.5 9.5 38.1)", None),
    ("", None),
])
def test_the_run_record_meta_carries_the_box_edge_and_the_potential(tmp_path, monkeypatch, box_line, edge):
    """``box.L`` is the first box the log reports when it is a cube; ``potential`` is ``run.json``'s own."""
    toy = _one_field(tmp_path, monkeypatch)
    run = _md_run_dir(tmp_path / "scratch", "r")
    (run / "log.lammps").write_text(box_line + "\n" + _LOG)
    record = json.loads((run / "run.json").read_text())
    record["potential"] = {"path": "model.pt", "sha256": "0" * 64, "species": ["A"]}
    (run / "run.json").write_text(json.dumps(record))
    [state] = index.build(toy)["state_points"]
    assert state["meta"]["box"]["L"] == edge
    assert state["meta"]["potential"] == record["potential"]


def test_the_atom_count_is_the_one_left_after_delete_atoms(tmp_path):
    """A configuration that fills a lattice and deletes down to the request is counted after the deletion."""
    log = tmp_path / "log.lammps"
    log.write_text("Created 4096 atoms\nDeleted 640 atoms, new total = 3456\n" + _LOG.replace("Created 500 atoms\n", ""))
    assert index._meta_from_log(log)["n_atoms"] == 3456


@pytest.mark.parametrize("kwargs,reason", [
    ({"dry_run": True}, "dry run"),
    ({"n_atoms": 4000}, "n_atoms=4000"),
    ({"system": "other"}, "system 'other'"),
])
def test_a_run_json_the_index_cannot_trust_is_a_named_skip(tmp_path, monkeypatch, kwargs, reason):
    toy = _one_field(tmp_path, monkeypatch)
    _md_run_dir(tmp_path / "scratch", "r", **kwargs)
    [record] = index.build(toy)["state_points"]
    assert reason in record["skipped"]


def test_the_run_record_name_and_keys_are_what_md_run_writes(tmp_path):
    from aipf.md.run import RECORD_FILE, run

    lj = load("lj")
    declared = lj.defaults["md"]["cube-overdamped"]
    run(lj, template="cube-overdamped", point=declared["point"], values=declared["values"],
        out=tmp_path / "r", lammps=tmp_path / "no-binary", dry_run=True,
        dt_equil_ps=declared["dt_equil_ps"])
    assert index._RUN_RECORD_FILENAME == RECORD_FILE
    assert index._run_record(tmp_path / "r") is not None


def test_a_file_called_run_json_that_is_not_a_run_record_is_ignored(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"something": "else"}))
    assert index._run_record(tmp_path) is None


def test_dump_modify_first_yes_counts_the_frame_at_step_zero(tmp_path):
    log = tmp_path / "log.lammps"
    log.write_text(_LOG)
    assert index._meta_from_log(log)["n_frames"] == 50000 // 100 + 1
    log.write_text(_LOG.replace(" first yes", ""))
    assert "n_frames" not in index._meta_from_log(log)


def test_build_refuses_to_replace_a_manifest_of_another_scratch_root(tmp_path, monkeypatch, capsys):
    toy = _one_field(tmp_path, monkeypatch)
    manifest = tmp_path / "farm" / "toy" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"scratch": "/elsewhere", "state_points": ["kept"]}))
    _md_run_dir(tmp_path / "scratch", "r")
    with pytest.raises(index.ForeignManifest, match="/elsewhere"):
        index.build(toy)
    assert json.loads(manifest.read_text())["state_points"] == ["kept"]
    assert not (tmp_path / "farm" / "toy" / "md").exists()
    assert index.build(toy, dry_run=True)["n_indexed"] == 1          # a dry run writes nothing
    monkeypatch.setattr("aipf.system.load", lambda name: toy)
    assert main(["data", "build", "--system", "toy"]) == 2
    assert "AIPF_DATA" in capsys.readouterr().err


def test_build_replaces_a_manifest_it_indexed_itself(tmp_path, monkeypatch):
    toy = _one_field(tmp_path, monkeypatch)
    _md_run_dir(tmp_path / "scratch", "r")
    index.build(toy)
    assert index.build(toy)["n_indexed"] == 1


# ---------------------------------------------------------------------------
# items 5, 7: aipf md run and aipf md doctor
# ---------------------------------------------------------------------------

def test_md_run_set_reaches_the_request(monkeypatch, tmp_path):
    import aipf.md.run as md_run

    seen = {}

    def fake(system, **kw):
        seen.update(kw)
        return md_run.RunRecord(deck_path=tmp_path, command="", result=None,
                                diagnosis=types.SimpleNamespace(can_run=True))
    monkeypatch.setattr(md_run, "run", fake)
    assert main(["md", "run", "--system", "lj", "--template", "cube-overdamped", "--lammps", "lmp",
                 "--out", str(tmp_path), "--set", "prod_ps=10", "--set", "n_atoms=500",
                 "--set", "dt_equil_ps=0.002", "--set", "SKIN=0.3", "--dry-run"]) == 0
    assert seen["point"]["prod_ps"] == 10 and seen["point"]["n_atoms"] == 500
    assert seen["dt_equil_ps"] == 0.002 and seen["values"]["SKIN"] == "0.3"
    assert seen["point"]["T"] == load("lj").defaults["md"]["cube-overdamped"]["point"]["T"]


def test_md_run_names_a_missing_request_field(tmp_path, capsys):
    declared = json.loads(json.dumps(load("lj").defaults["md"]["cube-overdamped"]))
    del declared["point"]["seed"]
    (tmp_path / "d.json").write_text(json.dumps(declared))
    assert main(["md", "run", "--system", "lj", "--template", "cube-overdamped", "--lammps", "lmp",
                 "--declared", str(tmp_path / "d.json"), "--out", str(tmp_path / "r"),
                 "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "['seed']" in err and "Traceback" not in err


def test_md_run_names_an_unfilled_deck_value(tmp_path, capsys):
    declared = json.loads(json.dumps(load("lj").defaults["md"]["cube-overdamped"]))
    del declared["values"]["PAIR"]
    (tmp_path / "d.json").write_text(json.dumps(declared))
    assert main(["md", "run", "--system", "lj", "--template", "cube-overdamped",
                 "--lammps", str(tmp_path / "no-binary"), "--declared", str(tmp_path / "d.json"),
                 "--out", str(tmp_path / "r"), "--dry-run"]) == 2
    assert "@PAIR@" in capsys.readouterr().err


def test_md_doctor_is_one_call_of_examine_site(monkeypatch, capsys):
    import aipf.md.doctor as doctor

    seen = {}

    class _Report:
        can_run = True

        def __str__(self):
            return "the report"
    monkeypatch.setattr(doctor, "examine_site", lambda site, **kw: seen.update(kw) or _Report())
    assert main(["md", "doctor", "--device", "cuda", "--deep"]) == 0
    assert seen == {"device": "cuda", "deep": True}
    assert capsys.readouterr().out.strip() == "the report"
    _Report.can_run = None
    assert main(["md", "doctor"]) == 1
    _Report.can_run = False
    assert main(["md", "doctor", "--device", "cpu"]) == 1


# ---------------------------------------------------------------------------
# items 8, 9, 10: refusals are exit 2; the declared T grid; one checkpoint keyword
# ---------------------------------------------------------------------------

_ROLLOUT = ["--run", "r", "--seeds", "--t-end", "1", "--dt", "1", "--save-ps", "1",
            "--device", "cpu", "--out", "data"]


@pytest.mark.parametrize("driver", ["spinodal", "slab"])
def test_rollout_on_a_system_without_a_rollout_block_is_exit_2(driver, capsys):
    assert main(["rollout", driver, "--system", "lj", "--ckpt", PUBLISHED_DIRNAME] + _ROLLOUT) == 2
    err = capsys.readouterr().err
    assert "declares no defaults['rollout']" in err and "Traceback" not in err


def test_rollout_reads_the_published_checkpoint_by_the_one_keyword(capsys):
    declared_roots.published_or_skip("hhe")
    from aipf.rollout.spinodal import resolve_checkpoint

    hhe = load("hhe")
    assert resolve_checkpoint(hhe, PUBLISHED_DIRNAME) == hhe.resolve_checkpoint()
    assert main(["rollout", "spinodal", "--system", "hhe", "--ckpt", "declared"] + _ROLLOUT) == 2
    assert "no checkpoint at declared" in capsys.readouterr().err


def test_the_one_field_grid_is_the_declared_one():
    from aipf.diagnose.run import declared_grid, request_T_grid

    declared = load("lj").defaults["diagnose"]
    own = declared_grid(declared["one_field"]["T_grid"])
    stages = ("one_field_phase_diagram",)
    grid, source = request_T_grid(declared, stages, (), None, override_declared=False)
    assert source == "declared" and np.array_equal(grid, own)
    grid, source = request_T_grid(declared, stages, (), np.linspace(0.5, 2.0, 40),
                                  override_declared=False)
    assert source == "declared"
    other = np.linspace(0.5, 2.0, 41)
    with pytest.raises(ValueError, match=r"defaults\['diagnose'\]\['one_field'\]\['T_grid'\].*--override-declared"):
        request_T_grid(declared, stages, (), other, override_declared=False)
    grid, source = request_T_grid(declared, stages, (), other, override_declared=True)
    assert source == "override" and np.array_equal(grid, other)


def test_diagnose_refuses_a_grid_that_is_not_the_declared_one(capsys):
    declared_roots.published_or_skip("lj")
    assert main(["diagnose", "--system", "lj", "--ckpt", PUBLISHED_DIRNAME,
                 "--stage", "one_field_phase_diagram", "--T-grid-n", "0.5,2.0,41"]) == 2
    err = capsys.readouterr().err
    assert "--override-declared" in err and "Traceback" not in err


def test_diagnose_passes_the_override_through(monkeypatch, tmp_path):
    declared_roots.published_or_skip("lj")
    import aipf.diagnose.run as run_mod

    seen = {}
    monkeypatch.setattr(run_mod, "run", lambda s, c, **kw: seen.update(kw) or tmp_path)
    assert main(["diagnose", "--system", "lj", "--ckpt", PUBLISHED_DIRNAME,
                 "--stage", "one_field_phase_diagram", "--T-grid-n", "0.5,2.0,41",
                 "--override-declared"]) == 0
    assert seen["override_declared"] is True and len(seen["T_grid"]) == 41
    assert main(["diagnose", "--system", "lj", "--ckpt", PUBLISHED_DIRNAME,
                 "--stage", "one_field_phase_diagram"]) == 0
    assert seen["T_grid"] is None and seen["override_declared"] is False


def test_diagnose_without_any_grid_is_exit_2_naming_the_key(monkeypatch, capsys):
    declared_roots.published_or_skip("lj")
    lj = load("lj")
    one_field = {k: v for k, v in lj.defaults["diagnose"]["one_field"].items() if k != "T_grid"}
    bare = dataclasses.replace(lj, defaults={**lj.defaults, "diagnose": {"one_field": one_field}})
    monkeypatch.setattr("aipf.system.load", lambda name: bare)
    assert main(["diagnose", "--system", "lj", "--ckpt", PUBLISHED_DIRNAME,
                 "--stage", "one_field_phase_diagram"]) == 2
    assert "defaults['diagnose']['one_field']['T_grid']" in capsys.readouterr().err


def test_diagnose_phase_diagram_without_a_pressure_is_exit_2(capsys):
    declared_roots.published_or_skip("hhe")
    assert main(["diagnose", "--system", "hhe", "--ckpt", PUBLISHED_DIRNAME,
                 "--stage", "phase_diagram", "--T-grid", "5000,6000,500"]) == 2
    assert "--pressure" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# item 14: one spelling of the published checkpoint
# ---------------------------------------------------------------------------

def test_the_published_directory_file_and_keyword_are_one_constant_each():
    import aipf.cli.diagnose_cmd as diagnose_cmd
    import aipf.cli.rollout_cmd as rollout_cmd
    import aipf.cli.train_cmd as train_cmd
    from aipf.cli.main import build_parser

    system = load("lj")
    assert system.paths.published().name == PUBLISHED_DIRNAME
    assert Checkpoint.tracked_under(system.paths.published()).name == TRACKED_FILE_NAME
    assert f"--init-from-{PUBLISHED_DIRNAME}" in build_parser()._subparsers._group_actions[0] \
        .choices["train"].format_usage()
    for module in (diagnose_cmd, rollout_cmd, train_cmd):
        assert module.PUBLISHED_DIRNAME is PUBLISHED_DIRNAME
    literal = [str(p.relative_to(REPO)) for p in (REPO / "src" / "aipf").rglob("*.py")
               if p.name != "paths.py" and ('= "published"' in p.read_text()
                                           or '"final.ckpt"' in p.read_text())]
    assert literal == []
