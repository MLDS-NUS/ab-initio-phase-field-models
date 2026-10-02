"""The rain column, ``experiments/hhe/column.py``: what it refuses, its command line, its noise.

The column driver was compared bit for bit with the published rain-zone chains' own driver
(deterministic and noisy, walled and mirror, chunked); that comparison is migration evidence and not
part of this suite. Its noisy convention is the published chains' ``m_stab="mean"``, declared as
``defaults["column"]["noise"]``. Here: the refusals, five steps from the command line each way, and
tier 1 of the FDT gate through the column's own fields (``slow``, on a GPU).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aipf.system import load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: experiments/ is a package
from experiments.hhe import column as col  # noqa: E402

import declared_roots

STEPS = 200
COMMON = dict(x_he=0.5, rho_tot=0.55, grid=(16, 16, 64), dx=0.70, ic_noise=0.005,
              seed=3, dt=1e-4, save_ps=0.001, field_stride=2)


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


def test_what_the_column_refuses(tmp_path):
    system = load("hhe")
    kw = dict(COMMON, geometry="walled", T=7000.0, bond=1.0, wall_amp=4.0, wall_width=8.0,
              t_profile=None, noisy=False, t_end=0.001, allow_t_edge=False,
              resume_from=None, chunk=None, device="cpu", out=tmp_path)
    for over, match in (({"geometry": "ball"}, "geometry must be one of"),
                        ({"geometry": "mirror"}, "no wall"),
                        ({"wall_amp": None}, "needs wall_amp"),
                        ({"t_profile": ("step", 6000, 9000, 0.75, 4)}, "mirror geometry only"),
                        ({"t_profile": ("linear", 5200, 9000)}, "within 500 K"),
                        ({"rho_tot": 0.2}, "outside the trust domain")):
        with pytest.raises(ValueError, match=match):
            col.column(system, "published", **{**kw, **over})
    bad = {k: v for k, v in system.defaults["column"].items() if k != "rain_rule"}
    with pytest.raises(ValueError, match="missing \\['rain_rule'\\]"):
        col.column(system, "published", **kw, declaration=bad)


def test_the_cli_runs_off_the_system_declaration(system, tmp_path):
    """``python -m experiments.hhe.column``: five deterministic steps of a small mirror column, CPU."""
    import json
    _pinned()
    assert col.main(["--ckpt", "published", "--geometry", "mirror", "--T", "7000", "--x-he", "0.5",
                     "--rho-tot", "0.55", "--dx", "0.7", "--bond", "1", "--wall-width", "4",
                     "--ic-noise", "0.005", "--t-end", "0.0005", "--dt", "1e-4",
                     "--save-ps", "1e-4", "--grid", "8,8,32", "--seed", "1",
                     "--field-stride", "1", "--noisy", "no", "--device", "cpu",
                     "--out", str(tmp_path)]) == 0
    (out,) = tmp_path.iterdir()
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["driver"] == "column" and manifest["noise"]["m_stab"] == "max"
    npz = np.load(out / manifest["outputs"][0])
    assert len(npz["t"]) == 6 and np.isfinite(npz["Phi"]).all()


def test_the_noisy_manifest_records_the_column_noise(system, tmp_path):
    """Five noisy steps: ``MANIFEST.json`` records ``defaults["column"]["noise"]``, gaussian at
    ``m_stab="mean"``, not the system's ``"max"``."""
    import json
    _pinned()
    assert col.main(["--ckpt", "published", "--geometry", "mirror", "--T", "7000", "--x-he", "0.5",
                     "--rho-tot", "0.55", "--dx", "0.7", "--bond", "1", "--wall-width", "4",
                     "--ic-noise", "0.005", "--t-end", "0.0005", "--dt", "1e-4",
                     "--save-ps", "1e-4", "--grid", "8,8,32", "--seed", "1",
                     "--field-stride", "1", "--noisy", "yes", "--device", "cpu",
                     "--out", str(tmp_path)]) == 0
    (out,) = tmp_path.iterdir()
    noise = json.loads((out / "MANIFEST.json").read_text())["noise"]
    assert (noise["noise_mode"], noise["m_stab"]) == ("gaussian", "mean")
    assert noise["sigma_noise"] == float(system.defaults["sigma"])


@pytest.mark.slow
@pytest.mark.env
def test_the_column_noise_passes_the_fdt_gate_tier1(system):
    """Tier 1 (``eps`` 0.01, band ``k <= 1.5``, window [0.9, 1.1]) at 10000 K on the column's grid and
    state, through the column's own fields: the mirror potential at ``Bo = 0`` (zero) and a uniform
    temperature profile, at the column's noisy convention ``defaults["column"]["noise"]``. The gate needs a
    homogeneous stationary state, so gravity itself is not tested."""
    import dataclasses
    from aipf.rollout.fdt import run_one
    from aipf.rollout.observables import tier1_ok
    from aipf.rollout.spinodal import load_model

    declared_roots.gpu_or_skip("20 ps of noisy rollout")
    T, grid = 10000.0, (16, 16, 64)
    box = np.array([g * 0.70 for g in grid])
    c = system.defaults["column"]
    kB = float(system.constants["kB"])
    v, _, _ = col.mirror_vext(grid, box, T, bond=0.0, smooth_width=8.0, kB=kB, masses=c["masses"])
    kbt, _ = col.t_profile_mirror(grid, box, T, T, smooth_width=8.0, t_range=c["T_window"], kB=kB)
    assert not v.any() and bool((kbt == kbt[0, 0, 0]).all())
    res = run_one(dataclasses.replace(system, noise=c["noise"]), load_model(system, system.verify_checkpoint()).to("cuda"),
                  {"solver": c["solver"]}, box=box, rho_bar=(0.275, 0.275), grid=grid, T=T,
                  eps=0.01, noise_eval="ito", t_end=20.0, dt=1e-4, save_dt=0.05,
                  burn_in=0.3, k_band=1.5, seed=0, device="cuda", v_ext=v.to("cuda"),
                  kbt_field=kbt.to("cuda"))
    print("\ncolumn tier 1:", res["band_mean_tr"], res["band_mean_cc"], res["rho_min"],
          res["out_of_domain_fraction"])
    assert res["rho_min"] > 1e-3 and res["out_of_domain_fraction"] <= 1e-4
    assert tier1_ok(res, 1.5, (0.9, 1.1))
