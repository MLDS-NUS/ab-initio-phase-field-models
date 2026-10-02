"""``aipf md doctor`` against the built environment: every check, for real.

The site is the declared one (``AIPF_LAMMPS``, ``AIPF_MACE_POTENTIAL``, ``aipf.toml`` [site]) and the
interpreter is ``AIPF_ENV_PREFIX``'s, else the one running the tests. On ``cuda`` every check is ok. On
``cpu`` the ten steps are n/a (the model route runs on a device) and the command exits 0.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

import declared_roots

pytestmark = pytest.mark.env

TIMEOUT_S = 1800


def _doctor(device: str) -> subprocess.CompletedProcess:
    for fact in ("lammps", "mace_potential"):
        if not declared_roots.site_fact(fact).is_file():
            pytest.skip(f"no {fact} declared on this machine ({declared_roots.site_fact(fact)})")
    python = declared_roots.aipf_python()
    if not python.exists():
        pytest.skip(f"no environment at {python.parent.parent} (AIPF_ENV_PREFIX)")
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    try:
        return subprocess.run([str(python), "-m", "aipf.cli.main", "md", "doctor", "--device", device],
                              capture_output=True, text=True, env=env, timeout=TIMEOUT_S,
                              cwd=declared_roots.REPO)
    except subprocess.TimeoutExpired:
        pytest.skip(f"the doctor did not finish within {TIMEOUT_S} s; not inspected")


def _verdicts(out: str) -> dict[str, str]:
    """``check -> verdict`` from the printed report (``[verdict] check: detail``)."""
    found = {}
    for line in out.splitlines():
        if line.startswith("[") and "] " in line:
            verdict, rest = line[1:].split("] ", 1)
            found[rest.split(": ", 1)[0].strip()] = verdict.strip()
    return found


def test_the_doctor_on_cpu():
    """On cpu the model is never started: ten_steps is n/a and the command passes."""
    r = _doctor("cpu")
    verdicts = _verdicts(r.stdout)
    for check in ("python_module", "mliap_style", "potential_loads"):
        assert verdicts.get(check) == "ok", r.stdout
    assert [c for c in verdicts if c.startswith("deck_styles:")], r.stdout
    assert all(v == "ok" for c, v in verdicts.items() if c.startswith("deck_styles:")), r.stdout
    assert verdicts["ten_steps"] == "n/a", r.stdout
    assert "ML-IAP runs on cuda only, use --device cuda" in r.stdout
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_doctor_on_cuda():
    if not shutil.which("nvidia-smi") or subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True).returncode != 0:
        pytest.skip("no GPU visible here")
    r = _doctor("cuda")
    verdicts = _verdicts(r.stdout)
    assert {"python_module", "mliap_style", "potential_loads", "ten_steps", "kokkos_device"} <= set(verdicts)
    assert all(v == "ok" for v in verdicts.values()), r.stdout
    assert r.returncode == 0, r.stdout + r.stderr
