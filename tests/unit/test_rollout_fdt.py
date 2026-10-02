"""``aipf.rollout.fdt.run_one`` end to end on the CPU: a few noisy steps on an 8^3 grid with a randomly
initialised model of the declared H/He form. Checks the result's shape and determinism, not physics
(the published-checkpoint gate is tests/unit/test_hhe_rollout_spinodal.py, on CUDA)."""
import numpy as np
import pytest
import torch

from aipf.rollout.fdt import run_one
from aipf.rollout.spinodal import declared
from aipf.system import load

DEVICE = "cpu"
GRID = (8, 8, 8)
BOX = np.array([7.0, 7.0, 7.0])
RHO_BAR = np.array([0.40, 0.30])
K_BAND = 1.5


@pytest.fixture(scope="module")
def setup():
    from aipf.functional.build import build
    system = load("hhe")
    torch.manual_seed(0)
    model = build(system).eval().to(DEVICE)
    return system, model, declared(system, "spinodal")


def _run(setup, seed=0, noise_eval="ito"):
    system, model, decl = setup
    return run_one(system, model, decl, box=BOX, rho_bar=RHO_BAR, grid=GRID,
                   T=9000.0, eps=0.01, noise_eval=noise_eval, t_end=6e-4,
                   dt=1e-4, save_dt=1e-4, burn_in=0.5, k_band=K_BAND,
                   seed=seed, device=DEVICE)


def test_one_gate_run_returns_every_field_finite(setup):
    res = _run(setup)
    assert set(res) == {"T", "eps", "noise_eval", "k_mode", "w_mode",
                        "r_tr_mode", "r_cc_mode", "n_frames", "band_mean_tr",
                        "band_mean_cc", "rho_min", "out_of_domain_fraction"}
    assert res["n_frames"] >= 1
    k = res["k_mode"]
    assert len(k) > 0 and (k > 0).all() and (k <= K_BAND).all()
    assert res["r_tr_mode"].shape == k.shape == res["w_mode"].shape
    for key in ("r_tr_mode", "r_cc_mode"):
        assert np.isfinite(res[key]).all(), key
    for key in ("band_mean_tr", "band_mean_cc", "rho_min"):
        assert np.isfinite(res[key]), key
    assert 0.0 <= res["out_of_domain_fraction"] <= 1.0


def test_the_seed_fixes_the_run_and_another_seed_moves_it(setup):
    a, b, c = _run(setup, seed=4), _run(setup, seed=4), _run(setup, seed=5)
    assert np.array_equal(a["r_tr_mode"], b["r_tr_mode"])
    assert not np.array_equal(a["r_tr_mode"], c["r_tr_mode"])
