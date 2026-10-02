"""The published models: what a user checks first after installing.

Every check here reads what a checkout carries, the tracked checkpoints under
``data/<system>/ckpt/published/`` and the figure data under ``figures/<fig>/figdata/``; one also
needs the Fe-B equation-of-state tables under its raw root, is marked ``env`` and skips, naming the
root, where it is absent. A checkout without the tracked checkpoints skips every check here,
naming the checkpoint (``declared_roots.published_or_skip``):

* each published checkpoint (H/He, Fe-B, Lennard-Jones and its two baselines) carries its declared
  digest and loads strictly into the model its system declares;
* ``aipf diagnose --system feb --ckpt published --stage stability_map`` reproduces the published
  ``lambda_min`` exactly and ``Gamma`` to 1e-13, on one row of Fig. 4's map;
* the Lennard-Jones model's critical temperature is the published 1.4341588 to 1e-6, and the two
  baselines' are the ones their published phase diagrams carry;
* ``aipf diagnose --system hhe|feb --ckpt published --stage kappa``, the README's quick start,
  gives the published models' ``kappa_eff``.

The whole published figure data are recomputed by ``tests/test_figdata_from_package.py`` (``-m slow``).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from aipf.diagnose.run import load_model
from aipf.paths import repo_root
from aipf.system import load

import declared_roots

FIGDATA = repo_root() / "figures"

#: (system, variant) of every published checkpoint.
PUBLISHED = [("hhe", None), ("feb", None), ("lj", None), ("lj", "fh"), ("lj", "landau")]

#: The published Lennard-Jones critical temperature (Fig. 3, ``Tc_exact``).
LJ_TC = 1.4341588

#: Gamma = x(1-x)/S_cc(0) is a linear solve on Hessians that agree with the published ones bit for
#: bit; its last bits depend on the LAPACK build (measured up to 1.1e-14 over the whole map).
FEB_GAMMA_BOUND = 1e-13

#: The row of Fig. 4's map recomputed here, and its pressure.
FEB_T, FEB_P = 1150.0, 10


def _system(name, variant):
    system = load(name)
    return system.variant(variant) if variant else system


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.mark.parametrize("name,variant", PUBLISHED,
                         ids=[f"{n}/{v}" if v else n for n, v in PUBLISHED])
def test_each_published_checkpoint_carries_its_digest_and_loads_strictly(name, variant):
    declared_roots.published_or_skip(name, variant)
    system = _system(name, variant)
    path = system.verify_checkpoint()
    assert hashlib.md5(path.read_bytes()).hexdigest() == system.checkpoint.md5
    model = load_model(system, path)          # strict: a missing or unexpected tensor raises
    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.env
def test_feb_stability_map_reproduces_one_published_row(tmp_path):
    """The stage solves the model's own isobar on the equation-of-state manifold, which it reads from
    ``eos_<P>GPa/eos_n_x_T.csv`` under the Fe-B raw root."""
    declared_roots.published_or_skip("feb")
    from aipf.cli.main import main

    declared_roots.raw_or_skip("feb", f"eos_{FEB_P}GPa", "eos_n_x_T.csv")
    assert main(["diagnose", "--system", "feb", "--ckpt", "published", "--stage", "stability_map",
                 "--T-grid", f"{FEB_T},{FEB_T},25", "--pressure", str(FEB_P),
                 "--out", str(tmp_path)]) == 0
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    got = dict(np.load(run_dir / "stability_map.npz"))
    want = dict(np.load(FIGDATA / "fig4_feb" / "figdata" / "gamma_map.npz"))
    row = int(np.flatnonzero(want["T"] == FEB_T)[0])
    x_B = 1.0 - got["x_int"]
    order = np.argsort(x_B)
    assert np.array_equal(x_B[order], want["x_B"])
    lam = got[f"lam_P{FEB_P}"][0, order]
    assert np.array_equal(lam, want[f"lambda_min_P{FEB_P}"][row], equal_nan=True)
    gamma, published = got[f"Gamma_P{FEB_P}"][0, order], want[f"Gamma_P{FEB_P}"][row]
    assert np.array_equal(np.isnan(gamma), np.isnan(published))
    finite = ~np.isnan(published)
    assert finite.any()
    assert float(np.max(np.abs(gamma[finite] - published[finite]))) <= FEB_GAMMA_BOUND


def test_the_lennard_jones_critical_temperature_is_the_published_one():
    """The one-field stage's ``Tc_exact``, on the CPU: the published value was written on a GPU in
    float32, about three float32 steps away (measured 3.6e-7)."""
    declared_roots.published_or_skip("lj")
    from aipf.diagnose.one_field import tc_curvature

    system = load("lj")
    model = load_model(system, system.resolve_checkpoint())
    spec = system.defaults["diagnose"]["one_field"]
    w0 = model.kernel.w_hat_zero_radial(**dict(spec["tc_w0"]))[0, 0]
    tc = float(tc_curvature(model, w0))
    assert tc == pytest.approx(LJ_TC, abs=1e-6)
    published = np.load(FIGDATA / "fig3_lj" / "figdata" / "phase_diagram_aipf.npz")["Tc_exact"]
    assert round(float(published), 7) == LJ_TC


@pytest.mark.parametrize("variant,rounded", [("fh", 1.3243), ("landau", 1.4342)])
def test_each_baselines_critical_temperature_is_its_published_one(variant, rounded):
    declared_roots.published_or_skip("lj", variant)
    system = load("lj").variant(variant)
    model = load_model(system, system.resolve_checkpoint())
    tc = float(model.Tc_curvature())
    published = float(np.load(FIGDATA / "fig3_lj" / "figdata" / f"phase_diagram_{variant}.npz")["Tc_exact"])
    assert tc == published
    assert round(tc, 4) == rounded


#: ``kappa_eff`` of each published model, the README's quick start (``--stage kappa``). No figure
#: carries it, so the values are the stage's own on the tracked checkpoints, measured on the CPU
#: (identical at one and at eight threads); the bound is float32 round-off of the kernel nets.
KAPPA_EFF = {
    "hhe": [[-9.668334218083645, -17.320989865018092], [-17.320989865018092, -24.600600001317442]],
    "feb": [[0.7325506520176491, -0.005287475021429955], [-0.005287475021429955, 0.6716741795713056]],
}


@pytest.mark.parametrize("name", sorted(KAPPA_EFF))
def test_the_quick_start_kappa_stage_gives_the_published_kappa(name, tmp_path):
    """``aipf diagnose --system <name> --ckpt published --stage kappa``, as the README spells it."""
    declared_roots.published_or_skip(name)
    from aipf.cli.main import main

    assert main(["diagnose", "--system", name, "--ckpt", "published", "--stage", "kappa",
                 "--out", str(tmp_path)]) == 0
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert run_dir.name == load(name).checkpoint.md5[:12]
    kappa = np.array(json.loads((run_dir / "kappa.json").read_text())["kappa_eff"])
    np.testing.assert_allclose(kappa, KAPPA_EFF[name], rtol=1e-6, atol=0)
    assert np.array_equal(kappa, kappa.T)
