"""``aipf md doctor`` checks the MACE potential each system runs: ``Site.for_system`` reads
``[site.mace_potential] <system>`` (and ``AIPF_MACE_POTENTIAL_<SYSTEM>``) before the bare fact.

It used to read only the bare ``site.mace_potential``, so a site that declares its potentials per
system (as two two-species systems need) was reported as having none."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from aipf import paths
from aipf.md import doctor
from aipf.md.doctor import Verdict, examine_site, system_potential, systems_with_potential
from aipf.site import Site

FEB = Path("/opt/site/feb-mliap.pt")
BARE = Path("/opt/site/bare.pt")


@pytest.fixture
def per_system(monkeypatch):
    for name in ("AIPF_MACE_POTENTIAL", "AIPF_MACE_POTENTIAL_FEB", "AIPF_MACE_POTENTIAL_HHE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(paths, "config",
                        lambda: {"site": {"mace_potential": {"feb": str(FEB)}}})


class _Child:
    def __init__(self):
        self.asked, self.all = None, []

    def __call__(self, command, *, timeout, env):
        self.asked = json.loads(command[-1])
        self.all.append(self.asked["potential"])
        facts = {"module": {"ok": True, "origin": "/x", "version": 1, "packages": [],
                            "kokkos_api": [], "mliap_styles": ["mliap"]},
                 "potential": None, "ten_steps": None}
        return types.SimpleNamespace(returncode=0, stdout=doctor._MARK + json.dumps(facts),
                                     stderr="")


def _tools(command, *, timeout, env=None):
    return types.SimpleNamespace(returncode=0, stdout="mliap" if command[-1] == "-h" else "",
                                 stderr="")


def _examine(site, **kw):
    child = _Child()
    d = examine_site(site, runner=child, tool_runner=_tools, which=lambda name: None,
                     env={"PATH": "/usr/bin"}, python="/opt/env/bin/python", systems=[],
                     device="cpu", **kw)
    return d, child


def test_the_systems_that_declare_a_potential():
    assert systems_with_potential() == ["feb", "hhe"]


def test_a_per_system_entry_wins_over_the_bare_fact(per_system):
    assert system_potential(Site(mace_potential=BARE), "feb") == (FEB, "")
    _, child = _examine(Site(mace_potential=BARE), potential_systems=["feb"])
    assert child.asked["potential"] == str(FEB)
    _, child = _examine(Site(mace_potential=BARE))  # no system: the bare fact, as before
    assert child.asked["potential"] == str(BARE)


def test_the_bare_fact_answers_for_a_system_without_its_own_entry(per_system):
    assert system_potential(Site(mace_potential=BARE), "hhe") == (BARE, "")


def test_a_system_without_one_is_named(per_system):
    path, why = system_potential(Site(), "hhe")
    assert path is None and "AIPF_MACE_POTENTIAL_HHE" in why
    d, child = _examine(Site(), potential_systems=["hhe"])
    assert child.asked["potential"] is None
    assert d.find("potential_loads").verdict is Verdict.NOT_APPLICABLE
    assert "AIPF_MACE_POTENTIAL_HHE" in d.find("potential_loads").detail


def test_a_system_that_runs_no_mliap_deck(per_system):
    d, child = _examine(Site(mace_potential=BARE), potential_systems=["lj"])
    assert child.asked["potential"] is None
    assert "declares no ML-IAP potential" in d.find("potential_loads").detail


def test_two_files_are_two_named_pairs_one_file_is_one_pair(per_system, monkeypatch):
    d, child = _examine(Site(mace_potential=BARE), potential_systems=["feb", "hhe"])
    assert child.all == [str(FEB), str(BARE)]
    names = [f.check for f in d.findings]
    assert names[:6] == ["python_module", "mliap_style", "potential_loads:feb", "ten_steps:feb",
                         "potential_loads:hhe", "ten_steps:hhe"]
    monkeypatch.setenv("AIPF_MACE_POTENTIAL_HHE", str(FEB))
    d, child = _examine(Site(mace_potential=BARE), potential_systems=["feb", "hhe"])
    assert child.all == [str(FEB)]
    assert [f.check for f in d.findings][:4] == ["python_module", "mliap_style",
                                                 "potential_loads", "ten_steps"]
    assert d.find("potential_loads").detail.endswith("[feb, hhe]")


def test_the_command_checks_every_declaring_system(monkeypatch, capsys):
    from aipf.cli.main import main
    seen = []

    def fake(site, **kw):
        seen.append(kw["potential_systems"])
        return doctor.Diagnosis(())

    monkeypatch.setattr(doctor, "examine_site", fake)
    assert main(["md", "doctor", "--device", "cpu"]) == 0
    assert main(["md", "doctor", "--device", "cpu", "--system", "hhe"]) == 0
    assert seen == [["feb", "hhe"], ["hhe"]]
