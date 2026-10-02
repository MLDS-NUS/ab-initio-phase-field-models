"""``aipf rollout`` off the H/He declaration: the slab and spinodal drivers run from the CLI.

The published model's own slab and spinodal runs are reproduced bit for bit (the deterministic slab
over 200 steps, the noisy one on a shared seed and over its first published picosecond on CUDA);
those comparisons need the published run files and are not part of this suite. Here: five steps of
each driver from the command line, every flag given and the rest declared. Marked ``env``: the
initial state is a state point's modes under the H/He raw root.
"""

import numpy as np
import pytest
import torch

from aipf.system import load

import declared_roots

RUN = "slab_xl0.30_xr0.70_T05000"


@pytest.fixture(scope="module")
def system():
    s = load("hhe")
    try:
        s.verify_checkpoint()
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"published model not found: {exc}")
    return s


def _pinned():
    torch.set_num_threads(1)


@pytest.mark.env
@pytest.mark.parametrize("driver,run,seeds", [
    ("slab", RUN, []), ("spinodal", "cube_x0.50_T07000", ["1"])])
def test_the_cli_runs_off_the_system_declaration(system, tmp_path, capsys,
                                                 driver, run, seeds):
    """``aipf rollout <driver> --system hhe``: five steps, every flag given, the rest declared."""
    import json
    from aipf.cli.main import main

    tree = load("hhe").defaults["rollout"][driver]["modes_tree"]
    declared_roots.raw_or_skip("hhe", *tree.split("/"), run)
    _pinned()
    assert main(["rollout", driver, "--system", "hhe", "--ckpt", "published",
                 "--run", run, "--seeds", *seeds, "--t-end", "0.0005",
                 "--dt", "1e-4", "--save-ps", "1e-4", "--device", "cpu",
                 "--out", str(tmp_path)]) == 0
    out = next(tmp_path.iterdir())
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["noise"]["m_stab"] == "max"
    npz = np.load(out / manifest["outputs"][0])
    assert len(npz["t_model"]) == 6
    assert np.isfinite(npz["Phi_model"]).all()
