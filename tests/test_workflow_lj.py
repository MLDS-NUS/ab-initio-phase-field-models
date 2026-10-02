"""The workflow guide's reduced-unit chain, once, on a tiny box (docs/guides/workflow.md sections 1-5).

``aipf md run`` (real LAMMPS, the site's ``AIPF_LAMMPS``) -> ``aipf data build`` -> ``aipf modes`` -> ``aipf train``
(two steps, fresh weights, then two from the published weights) -> ``aipf diagnose --stage
one_field_phase_diagram`` on that run's final.ckpt and on the published one, through the command line in
one process. ``AIPF_RAW_LJ`` and ``AIPF_DATA`` point under ``tmp_path``, so every stage reads only what
the previous one wrote there, and each assertion reads that from the record the stage itself wrote. The
published checkpoint is read from the checkout's tracked file with the farm elsewhere, with no link into
the farm. No claim about the numbers: they exist, they are finite where a value is defined, and they name
their inputs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from aipf.cli.main import main
from aipf.system import load

#: The guide's tiny box: 5^3 FCC cells (500 atoms). The run is shortened on the command line: 1 ps of
#: preparation, 10 ps of production dumped every 0.02 ps (501 frames, enough for the declared window).
CELLS, N_ATOMS = 5, 500
SHORT = ("equil_ps=1", "prod_ps=10", "dump_every_ps=0.02")
TAG = "cube_x0.50_T1.6_s42"
GRID = "16,16,16"
N_T = 40  # the declared one-field grid, linspace(0.5, 2.0, 40)

#: The declared loader forks four workers inside pytest's threads; Python warns, the batches are the same.
_FORK = "ignore:This process .* is multi-threaded, use of fork\\(\\) may lead to deadlocks:DeprecationWarning"


def _tiny_box(lj) -> dict:
    """The system's own ``defaults["md"]["cube-overdamped"]`` with the box made 5^3 cells."""
    declared = copy.deepcopy(lj.defaults["md"]["cube-overdamped"])
    declared["point"]["n_atoms"] = N_ATOMS
    full = "region          box block 0 10 0 10 0 10"
    assert full in declared["values"]["CONFIGURATION"]
    declared["values"]["CONFIGURATION"] = declared["values"]["CONFIGURATION"].replace(
        full, f"region          box block 0 {CELLS} 0 {CELLS} 0 {CELLS}")
    return declared


@pytest.mark.slow
@pytest.mark.env
@pytest.mark.filterwarnings(_FORK)
def test_md_then_index_then_modes_then_train_then_diagnose(tmp_path, monkeypatch, capsys):
    from aipf.site import Site
    binary = Site.load().lammps
    if binary is None or not binary.is_file():
        pytest.skip("AIPF_LAMMPS (a LAMMPS binary with the brownian style) not set")
    lj = load("lj")
    from aipf.paths import tracked_tables
    tables = lj.defaults["training"]["tables"]
    for key in ("m_table", "s_table"):
        if not (tracked_tables("lj") / tables[key]).is_file():
            pytest.skip(f"the declared anchor table {key} is not tracked: {tables[key]}")

    raw, farm = tmp_path / "lj-md", tmp_path / "data"
    monkeypatch.setenv("AIPF_RAW_LJ", str(raw))
    monkeypatch.setenv("AIPF_DATA", str(farm))
    monkeypatch.delenv("AIPF_RAW", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    # 1. MD: the declared deck on a tiny box, shortened by --set, into a directory of any name.
    declared = tmp_path / "tiny.json"
    declared.write_text(json.dumps(_tiny_box(lj)))
    run_dir = raw / "tiny"
    argv = ["md", "run", "--system", "lj", "--template", "cube-overdamped", "--declared", str(declared),
            "--out", str(run_dir)]  # the binary is the site's: AIPF_LAMMPS
    for pair in SHORT:
        argv += ["--set", pair]
    assert main(argv) == 0
    dump = run_dir / "dump.lammpstrj"
    assert dump.is_file() and (run_dir / "log.lammps").is_file()
    record = json.loads((run_dir / "run.json").read_text())
    assert record["result"]["status"] == "ok"
    assert (record["point"]["prod_ps"], record["point"]["equil_ps"]) == (10.0, 1.0)

    # 2. The farm: exactly this run, indexed from its run.json and its log.
    assert main(["data", "build", "--system", "lj"]) == 0
    manifest = json.loads((farm / "lj" / "manifest.json").read_text())
    assert manifest["raw"] == str(raw)
    assert [r["tag"] for r in manifest["state_points"]] == [TAG]
    record = manifest["state_points"][0]
    assert record["trajectory"] == str(dump)
    meta = record["meta"]
    assert (meta["n_atoms"], meta["T_K"], meta["n_frames"]) == (N_ATOMS, 1.6, 501)
    assert (meta["geometry"], meta["ensemble"]) == ("cube", "langevin_overdamped")
    assert os.readlink(farm / "lj" / "md" / TAG / "traj.lammpstrj") == str(dump)

    # 3. Modes, at the declared width and cutoff, of the dump stage 1 wrote.
    capsys.readouterr()
    assert main(["modes", "--system", "lj", "--tag", TAG, "--sigma", str(lj.defaults["sigma"]),
                 "--k-cut", str(lj.defaults["k_cut"])]) == 0
    cache = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert cache == farm / "lj" / "modes" / TAG
    provenance = json.loads((cache / "provenance.json").read_text())
    assert provenance["source"] == str(dump)
    assert provenance["source_sha256"] == hashlib.sha256(dump.read_bytes()).hexdigest()
    with np.load(cache / "modes.npz") as archive:
        assert np.isfinite(archive["rho_k"]).all() and archive["rho_k"].shape[0] == 501

    # 4. Two steps from fresh weights on that archive, the declared anchors attached.
    assert main(["train", "--system", "lj", "--run", "guide", "--seed", "0", "--steps", "2",
                 "--source", f"cube=.:{TAG}:{GRID}", "--resume-optimizer", "no",
                 "--anchors", "declared", "--log-every-step", "--device", "cpu"]) == 0
    trained = farm / "lj" / "ckpt" / "guide"
    final = trained / "final.ckpt"
    run = json.loads((trained / "MANIFEST.json").read_text())
    assert run["final_md5"] == hashlib.md5(final.read_bytes()).hexdigest()
    assert run["global_step"] == 2 and run["init_from"] is None
    assert [(s["root"], s["pattern"]) for s in run["sources"]] == [
        (str(farm / "lj" / "modes"), TAG)]
    assert sorted(run["terms_trained"]) == ["L_M", "L_bulk", "L_dyn"]
    steps = json.loads((trained / "steps.json").read_text())
    assert len(steps["loss"]) == 2 and all(math.isfinite(v) for v in steps["loss"])

    # 4b. The same from the published weights: the checkout's tracked file, the farm elsewhere.
    assert not (farm / "lj" / "ckpt" / "published").exists()
    assert main(["train", "--system", "lj", "--run", "from-published", "--seed", "0", "--steps", "2",
                 "--source", f"cube=.:{TAG}:{GRID}", "--init-from-published",
                 "--resume-optimizer", "no", "--anchors", "declared"]) == 0
    warm = json.loads((farm / "lj" / "ckpt" / "from-published" / "MANIFEST.json").read_text())
    assert warm["init_from"] == {"path": lj.checkpoint.path, "md5": lj.checkpoint.md5}

    # 5. The one-field phase diagram of the checkpoint stage 4 signed, on the declared grid.
    capsys.readouterr()
    assert main(["diagnose", "--system", "lj", "--ckpt", str(final),
                 "--stage", "one_field_phase_diagram"]) == 0
    out = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert out == farm / "lj" / "diagnose" / run["final_md5"][:12]
    diagnosed = json.loads((out / "MANIFEST.json").read_text())
    assert diagnosed["md5"] == run["final_md5"] and diagnosed["checkpoint"] == str(final)
    assert diagnosed["T_grid_source"] == "declared" and len(diagnosed["T_grid"]) == N_T
    with np.load(out / "one_field_phase_diagram.npz") as diagram:
        assert len(diagram["T"]) == N_T and np.isfinite(diagram["T"]).all()
        assert np.isfinite(diagram["Tc_exact"]) and np.isfinite(diagram["mu_w0"])
        for key in ("binodal_L", "binodal_R", "spinodal_L", "spinodal_R"):
            values = diagram[key][np.isfinite(diagram[key])]    # NaN: no gap at that T
            assert ((values > 0) & (values < 1)).all(), key

    # 5b. The published checkpoint, by the keyword, with no link into the farm.
    capsys.readouterr()
    assert main(["diagnose", "--system", "lj", "--ckpt", "published",
                 "--stage", "one_field_phase_diagram"]) == 0
    out = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert out == farm / "lj" / "diagnose" / lj.checkpoint.md5[:12]
    with np.load(out / "one_field_phase_diagram.npz") as diagram:
        assert np.isfinite(diagram["binodal_L"]).any()
