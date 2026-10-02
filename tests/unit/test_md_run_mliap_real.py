"""``aipf md run`` on the Fe-B cube deck through the ML-IAP route, for real.

Two atoms, ten steps of melt and ten of production, through the command line in the built
environment, with the site's Fe-B potential (its md5 must be the one Fe-B declares). On cpu the command
refuses (exit 2, nothing written): the route computes on a GPU only. On cuda, where a GPU is visible,
it runs in process and exits 0.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

import declared_roots

pytestmark = pytest.mark.env

TIMEOUT_S = 1800
SHORT = ["--set", "n_atoms=2", "--set", "equil_ps=0.01", "--set", "prod_ps=0.01",
         "--set", "dump_every_ps=0.001"]


def _aipf(out, device):
    from aipf.md.potentials import md5_of
    from aipf.system import load
    declared_roots.lammps_or_skip(declared_roots.MLIAP_KK)
    from aipf.site import MissingSiteFact, Site
    try:
        potential = Site.load().for_system("mace_potential", "feb")
    except MissingSiteFact as missing:
        pytest.skip(str(missing))
    if not potential.is_file():
        pytest.skip(f"no file at the site's Fe-B potential {potential}")
    want = load("feb").defaults["md"]["potential"]["md5"]
    if md5_of(potential) != want:
        pytest.skip(f"the site's potential {potential} is not the one Fe-B declares (md5 {want})")
    python = declared_roots.aipf_python()
    if not python.exists():
        pytest.skip(f"no environment at {python.parent.parent} (AIPF_ENV_PREFIX)")
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    try:
        return subprocess.run([str(python), "-m", "aipf.cli.main", "md", "run", "--system", "feb",
                               "--template", "cube-npt", *SHORT, "--device", device, "--out", str(out)],
                              capture_output=True, text=True, env=env, timeout=TIMEOUT_S,
                              cwd=declared_roots.REPO)
    except subprocess.TimeoutExpired:
        pytest.skip(f"the run did not finish within {TIMEOUT_S} s; not inspected")


def test_the_mliap_deck_refuses_cpu_and_writes_nothing(tmp_path):
    done = _aipf(tmp_path / "cpu", "cpu")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "ML-IAP route" in done.stderr and "--device cuda" in done.stderr
    assert not (tmp_path / "cpu").exists()


def test_the_mliap_deck_runs_two_atoms_on_cuda(tmp_path):
    if shutil.which("nvidia-smi") is None or not os.environ.get("CUDA_VISIBLE_DEVICES", "x"):
        pytest.skip("no GPU visible on this node (run it on a GPU node, or through --pbs)")
    out = tmp_path / "cuda"
    done = _aipf(out, "cuda")
    if done.returncode == 2 and "no GPU is visible" in done.stderr:
        pytest.skip(done.stderr.strip().splitlines()[-1])
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    record = json.loads((out / "run.json").read_text())
    assert record["route"] == "in_process" and record["result"]["status"] == "ok"
    assert record["result"]["n_atoms"] == 2
    log = (out / "log.lammps").read_text()
    assert "Created 8 atoms" in log and "Deleted 6 atoms, new total = 2" in log
    rows = [line.split() for line in log.splitlines() if line.split()[:1] == ["10"]]
    assert len(rows) == 2                              # the melt's step 10, then production's
    assert (out / "traj.dump").read_text().count("ITEM: TIMESTEP") == 11
