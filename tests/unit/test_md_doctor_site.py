"""``aipf md doctor``: the declared site, end to end, with every outcome a named finding.

The interpreter, the binary and the device are fakes here; ``tests/unit/test_md_doctor_real.py``
runs the same examination against the built environment.
"""
from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import pytest

from aipf.md import doctor
from aipf.md.doctor import Verdict, deck_families, examine_site, loader_path, styles_of
from aipf.site import Site

LMP = Path("/opt/site/lmp")
MODEL = Path("/opt/site/model.pt")
GPU = "GPU 0: A Test Device (UUID: GPU-0)"


def _help(*, mliap=True, backend="CUDA OpenMP Serial"):
    styles = sorted({s for _, text in deck_families() for s in styles_of(text)})
    if mliap:
        styles += ["mliap", "mliap/kk"]
    return ("Large-scale Atomic/Molecular Massively Parallel Simulator - 10 Dec 2025\n"
            f"KOKKOS package API: {backend}\n" + " ".join(styles) + "\n")


def _facts(**over):
    facts = {
        "executable": "/opt/env/bin/python",
        "module": {"ok": True, "origin": "/opt/env/lib/lammps/__init__.py", "version": 20251210,
                   "packages": ["KOKKOS", "ML-IAP", "PYTHON"], "kokkos_api": ["cuda", "openmp"],
                   "mliap_styles": ["mliap", "mliap/kk"]},
        "torch": {"version": "2.6.0", "cuda": True, "device_name": "A Test Device"},
        "potential": {"ok": True, "path": str(MODEL), "species": ["B", "Fe"], "dtype": "float32",
                      "cutoff": 3.0, "fallback_kernels": [], "e3nn": "0.5.1"},
        "ten_steps": {"ok": True, "status": "ok", "n_atoms": 2, "seconds": 1.5, "args": [],
                      "last_row": ["10", "290.1", "-12.3", "-12.2"], "detail": ""},
    }
    facts.update(over)
    return facts


class _Child:
    """The child interpreter: records what it was asked, answers with ``facts`` (or raises)."""

    def __init__(self, facts=None, raises=None, stdout=None):
        self.facts, self.raises, self.stdout, self.asked = facts, raises, stdout, None

    def __call__(self, command, *, timeout, env):
        self.asked = json.loads(command[-1])
        self.env = env
        if self.raises is not None:
            raise self.raises
        out = self.stdout if self.stdout is not None else doctor._MARK + json.dumps(self.facts)
        return types.SimpleNamespace(returncode=0 if self.stdout is None else -6,
                                     stdout="LAMMPS chatter\n" + out + "\n", stderr="the tail")


def _tools(help_text, gpus=(GPU,)):
    def run(command, *, timeout, env=None):
        if command[-1] == "-h":
            return types.SimpleNamespace(returncode=0, stdout=help_text, stderr="")
        return types.SimpleNamespace(returncode=0, stdout="\n".join(gpus), stderr="")
    return run


def _examine(site, child, help_text=None, gpus=(GPU,), **kw):
    return examine_site(site, runner=child, tool_runner=_tools(help_text or _help(), gpus),
                        which=lambda name: f"/usr/bin/{name}", env={"PATH": "/usr/bin"},
                        python="/opt/env/bin/python", systems=None, **kw)


SITE = Site(lammps=LMP, mace_potential=MODEL)


def test_a_healthy_site_on_a_device_is_ok_everywhere():
    child = _Child(_facts())
    d = _examine(SITE, child, device="cuda")
    names = [f.check for f in d.findings]
    assert names[:5] == ["python_module", "mliap_style", "potential_loads", "ten_steps",
                         "kokkos_device"]
    assert all(n.startswith("deck_styles:") for n in names[5:])
    assert {n.split(":", 1)[1] for n in names[5:]} >= {"cube-overdamped", "lj/cube-overdamped"}
    assert d.can_run is True, str(d)
    assert child.asked == {"potential": str(MODEL), "run": True, "device": "cuda"}
    assert "290.1" in d.find("ten_steps").detail


def test_auto_takes_the_device_when_one_is_visible_and_the_host_otherwise():
    child = _Child(_facts())
    _examine(SITE, child, device="auto")
    assert child.asked["device"] == "cuda"
    _examine(SITE, child, device="auto", gpus=())
    assert child.asked["device"] == "cpu"


def test_a_binary_without_mliap_is_a_named_broken_check():
    d = _examine(SITE, _Child(_facts()), help_text=_help(mliap=False), device="cuda")
    found = d.find("mliap_style")
    assert found.verdict is Verdict.BROKEN and "lists no mliap" in found.detail
    assert d.can_run is False


def test_an_undeclared_binary_and_potential_name_their_variables():
    child = _Child(_facts(potential=None, ten_steps=None))
    d = _examine(Site(), child, device="cuda")
    assert d.find("mliap_style").verdict is Verdict.NOT_APPLICABLE
    assert "AIPF_LAMMPS" in d.find("mliap_style").remedy
    assert d.find("potential_loads").verdict is Verdict.NOT_APPLICABLE
    assert d.find("potential_loads").detail == doctor.NO_POTENTIAL
    assert d.find("ten_steps").verdict is Verdict.NOT_APPLICABLE
    assert child.asked["potential"] is None and child.asked["run"] is False
    assert d.find("deck_styles:cube-overdamped").verdict is Verdict.UNKNOWN
    assert d.can_run is None  # the pair decks need a binary, and none is declared


def test_a_pair_only_site_on_cpu_passes():
    """No potential declared, a binary without mliap: every route the site declared is ok."""
    child = _Child(_facts(potential=None, ten_steps=None))
    d = _examine(Site(lammps=LMP), child, help_text=_help(mliap=False), device="cpu")
    assert d.find("mliap_style").verdict is Verdict.NOT_APPLICABLE
    assert d.find("potential_loads").verdict is Verdict.NOT_APPLICABLE
    assert d.can_run is True, str(d)
    assert "n/a" in str(d)


def test_a_declared_potential_that_does_not_load_on_cpu_is_broken():
    facts = _facts(potential={"ok": False, "error": "PotentialError: not converted"}, ten_steps=None)
    d = _examine(SITE, _Child(facts), device="cpu")
    assert d.find("potential_loads").verdict is Verdict.BROKEN
    assert d.can_run is False


@pytest.mark.parametrize("backend, api", [("CUDA Serial", ["cuda", "serial"]),
                                          ("OpenMP Serial", ["openmp", "serial"])])
def test_on_cpu_the_model_is_never_started_and_ten_steps_is_not_applicable(backend, api):
    facts = _facts(ten_steps=None)
    facts["module"]["kokkos_api"] = api
    child = _Child(facts)
    d = _examine(SITE, child, help_text=_help(backend=backend), device="cpu")
    assert child.asked == {"potential": str(MODEL), "run": False, "device": "cpu"}
    found = d.find("ten_steps")
    assert found.verdict is Verdict.NOT_APPLICABLE
    assert found.detail == "ML-IAP runs on cuda only, use --device cuda"
    assert "[n/a          ] ten_steps: ML-IAP runs on cuda only, use --device cuda" in str(d)
    assert "kokkos_device" not in [f.check for f in d.findings]
    assert d.can_run is True, str(d)


def test_cuda_without_a_device_is_broken_and_nothing_runs():
    child = _Child(_facts(ten_steps=None))
    d = _examine(SITE, child, gpus=(), device="cuda")
    assert child.asked["run"] is False
    assert d.find("kokkos_device").verdict is Verdict.BROKEN
    assert d.find("ten_steps").verdict is Verdict.UNKNOWN


def test_a_child_that_times_out_is_not_inspected():
    d = _examine(SITE, _Child(raises=subprocess.TimeoutExpired("python", 1)), device="cuda")
    assert d.find("python_module").verdict is Verdict.UNKNOWN
    assert d.find("ten_steps").verdict is Verdict.UNKNOWN
    assert d.can_run is None  # nothing broken was seen, and nothing was inspected: exit 1
    assert "within" in d.find("python_module").detail


def test_a_child_that_dies_is_a_broken_run_with_its_tail():
    d = _examine(SITE, _Child(stdout="Segmentation fault"), device="cuda")
    found = d.find("ten_steps")
    assert found.verdict is Verdict.BROKEN and found.raw == "the tail"


def test_a_model_that_does_not_load_is_broken():
    facts = _facts(potential={"ok": False, "error": "PotentialError: not converted"}, ten_steps=None)
    d = _examine(SITE, _Child(facts), device="cuda")
    assert d.find("potential_loads").verdict is Verdict.BROKEN
    assert "mliap" in d.find("potential_loads").remedy
    assert d.find("ten_steps").verdict is Verdict.UNKNOWN


def test_the_child_environment_drops_the_allocator_and_leads_with_its_own_libraries(tmp_path):
    prefix = tmp_path / "env"
    site = prefix / "lib" / "python3.12" / "site-packages" / "nvidia"
    for name in ("cublas", "cuda_nvrtc"):
        (site / name / "lib").mkdir(parents=True)
    (prefix / "bin").mkdir()
    env = doctor.site_environment(prefix / "bin" / "python", cuda_lib="/site/cuda/lib64",
                                  env={"LD_LIBRARY_PATH": "/site/compilers/lib:/site/cuda/lib64",
                                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    parts = env["LD_LIBRARY_PATH"].split(":")
    assert parts[:5] == [str(site / "cublas" / "lib"), str(site / "cuda_nvrtc" / "lib"),
                         str(prefix / "lib"), "/site/cuda/lib64", "/site/compilers/lib"]
    assert env["PYTHONNOUSERSITE"] == "1"


def test_loader_path_never_leaves_an_empty_entry(tmp_path):
    assert "::" not in loader_path(tmp_path, "") and not loader_path(tmp_path, "").endswith(":")


def test_the_cli_reads_the_site(monkeypatch, capsys):
    from aipf.cli.main import main
    seen = {}

    class _Report:
        can_run = True

        def __str__(self):
            return "the report"
    monkeypatch.setattr(doctor, "examine_site", lambda site, **kw: seen.update(kw, site=site) or _Report())
    monkeypatch.setenv("AIPF_LAMMPS", "/opt/site/lmp")
    assert main(["md", "doctor", "--device", "cpu"]) == 0
    assert seen["device"] == "cpu" and seen["deep"] is False
    assert seen["site"].lammps == Path("/opt/site/lmp")
    assert capsys.readouterr().out.strip() == "the report"
    _Report.can_run = None
    assert main(["md", "doctor"]) == 1
    assert seen["device"] == "auto"
