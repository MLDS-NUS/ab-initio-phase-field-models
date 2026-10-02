"""aipf.md.run and ``aipf md run``: a named deck filled, written, pre-flighted, and run or not.

The declaration is the reduced-unit tree's homogeneous overdamped run, every value quoted from its
archived homogeneous overdamped deck of that tree, declared by the
system under ``defaults["md"]["cube-overdamped"]`` (experiments/lj/system.py).
"""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from aipf.md.doctor import Verdict, deck_styles, styles_of
from aipf.md.templates import DeckError


def _lj_declared():
    """The reduced-unit tree's archived deck, as the system declares it (``defaults["md"]``)."""
    from aipf.system import load
    return load("lj").defaults["md"]["cube-overdamped"]


LJ_DECLARED = _lj_declared()


def _help(*styles: str):
    """A runner whose binary's help lists ``styles``."""
    def runner(argv, timeout):
        return SimpleNamespace(returncode=0, stdout=" ".join(styles), stderr="")
    return runner


def _lj():
    from aipf.system import load
    return load("lj")


def _run(tmp_path, *, dry_run=True, lammps="lmp", **over):
    from aipf.md.run import run
    kw = dict(template="cube-overdamped", point=LJ_DECLARED["point"],
              values=LJ_DECLARED["values"], out=tmp_path, lammps=lammps,
              dry_run=dry_run, dt_equil_ps=LJ_DECLARED["dt_equil_ps"])
    kw.update(over)
    return run(_lj(), **kw)


def test_the_styles_a_deck_names_are_read_in_order():
    text = ("atom_style atomic\npair_style lj/smooth/linear 2.5\n"
            "fix a all nve\nfix b all brownian 1 2 gamma_t 2\n# fix c all npt\n"
            "compute t all temp\nfix d all nve")
    assert styles_of(text) == ("atomic", "lj/smooth/linear", "nve", "brownian", "temp")


def test_a_binary_missing_a_style_is_broken_and_names_it():
    finding = deck_styles("lmp", "pair_style lj/cut 2.5\nfix a all brownian 1 2",
                          runner=_help("lj/cut", "nve"))
    assert finding.verdict is Verdict.BROKEN
    assert "brownian" in finding.detail and finding.remedy


def test_the_dry_run_writes_the_deck_the_template_parametrises(tmp_path, monkeypatch):
    import aipf.md.run as run_module
    monkeypatch.setattr(run_module, "deck_styles",
                        lambda binary, text: deck_styles(binary, text, runner=_help(
                            *styles_of(text))))
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    record = _run(tmp_path)
    text = record.deck_path.read_text()
    assert record.deck_path == tmp_path / "in.lammps"
    assert "@" not in text
    for line in ("units           lj", "pair_style      lj/smooth/linear 2.5",
                 "timestep        0.001", "run             250000",
                 "fix             prod all brownian 1.6 42 gamma_t 2.0",
                 "run             25000000", "mass            2 1.0",
                 "fix             prep all langevin 2.0 2.0 0.2 42"):
        assert line in text, line
    assert record.diagnosis.can_run is True
    assert record.result is None
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["dry_run"] is True and saved["template"] == "cube-overdamped"
    assert record.command.endswith("-in in.lammps")


def test_an_undeclared_value_is_refused(tmp_path):
    values = dict(LJ_DECLARED["values"])
    values.pop("GAMMA")
    with pytest.raises(DeckError, match="GAMMA"):
        _run(tmp_path, values=values)


def test_an_unknown_template_lists_the_names(tmp_path):
    with pytest.raises(DeckError, match="cube-overdamped"):
        _run(tmp_path, template="nope")


def test_a_system_with_engine_settings_needs_its_campaign(tmp_path):
    from aipf.md.run import run
    from aipf.system import load
    hhe = load("hhe")
    point = {"geometry": "cube", "ensemble": "NPT", "T": 7000.0, "x": 0.5,
             "dt_ps": 2e-4, "equil_ps": 1.5, "prod_ps": 30.0,
             "dump_every_ps": 0.02, "dump_from": "prod", "seed": 1, "P": 800.0,
             "n_atoms": 3456}
    with pytest.raises(ValueError, match="campaign"):
        run(hhe, template="cube-npt", point=point, values={}, out=tmp_path,
            lammps="lmp", dry_run=True)


def test_a_run_is_refused_when_the_pre_flight_fails(tmp_path, monkeypatch):
    import aipf.md.run as run_module
    monkeypatch.setattr(run_module, "deck_styles",
                        lambda binary, text: deck_styles(binary, text, runner=_help()))
    with pytest.raises(RuntimeError, match="pre-flight"):
        _run(tmp_path, dry_run=False)


def test_the_command_line_dry_run(tmp_path, monkeypatch):
    """``aipf md run --system lj --template cube-overdamped --dry-run``."""
    import aipf.md.run as run_module
    from aipf.cli.main import main
    monkeypatch.setattr(run_module, "deck_styles",
                        lambda binary, text: deck_styles(binary, text, runner=_help(
                            *styles_of(text))))
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    declared = tmp_path / "declared.json"
    declared.write_text(json.dumps(LJ_DECLARED))
    code = main(["md", "run", "--system", "lj", "--template", "cube-overdamped",
                 "--declared", str(declared), "--lammps", "lmp",
                 "--out", str(tmp_path / "run"), "--dry-run"])
    assert code == 0
    assert (tmp_path / "run" / "in.lammps").is_file()


@pytest.mark.env
def test_the_dry_run_passes_the_doctor_on_the_trees_own_binary(tmp_path, monkeypatch):
    """Against the site's LAMMPS binary (``AIPF_LAMMPS``), one that has the reduced-unit styles."""
    from aipf.site import Site
    binary = Site.load().lammps
    if binary is None or not binary.is_file():
        pytest.skip("AIPF_LAMMPS (a LAMMPS binary with the brownian style) not set")
    binary = str(binary)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    record = _run(tmp_path, lammps=binary)
    assert record.diagnosis.find("deck_styles").verdict is Verdict.OK, record.diagnosis
    assert record.diagnosis.can_run is True
    # the engine itself parses the deck up to its first run command
    head = record.deck_path.read_text().split("\nrun ")[0]
    probe = tmp_path / "in.parse"
    probe.write_text(head.split("velocity")[0])
    done = subprocess.run([binary, "-in", str(probe), "-log", "none", "-screen", "none"],
                          cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


def test_the_command_line_dry_run_reads_the_systems_own_declaration(tmp_path, monkeypatch):
    """``aipf md run --system lj --template cube-overdamped --dry-run``, no ``--declared``."""
    import aipf.md.run as run_module
    from aipf.cli.main import main
    monkeypatch.setattr(run_module, "deck_styles",
                        lambda binary, text: deck_styles(binary, text, runner=_help(
                            *styles_of(text))))
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    code = main(["md", "run", "--system", "lj", "--template", "cube-overdamped",
                 "--lammps", "lmp", "--out", str(tmp_path / "run"), "--dry-run"])
    assert code == 0
    text = (tmp_path / "run" / "in.lammps").read_text()
    assert "fix             prod all brownian 1.6 42 gamma_t 2.0" in text
    assert "mass            2 1.0" in text


def test_a_template_the_system_declares_nothing_for_needs_declared(tmp_path, capsys):
    from aipf.cli.main import main
    assert main(["md", "run", "--system", "lj", "--template", "cube-npt",
                 "--lammps", "lmp", "--out", str(tmp_path / "run"), "--dry-run"]) == 2
    assert "defaults['md']['cube-npt']" in capsys.readouterr().err
