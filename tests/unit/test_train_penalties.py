"""The two hinge penalties: the declared trust domain, the probe stream, and ``fit`` training them.

A toy two-channel system only; the published model's declared penalties are in
``tests/unit/test_hhe_penalties.py``.
"""
import csv
import dataclasses
import json

import numpy as np
import pytest
import torch

import test_train_fit as toy
from aipf.losses.convexity import gamma_closed_form, gamma_on_path
from aipf.solve.trust_domain import in_domain
from aipf.system import System, TrustDomain
from aipf.train.config import TrainConfig
from aipf.train.fit import _config_from_system, fit
from aipf.train.penalties import (ConvexityProbe, GammaPaths, Penalties,
                                  penalties_from_system)
from aipf.train.sampling import sample_trust_domain


@pytest.fixture(autouse=True)
def _demo_scratch_is_its_own(monkeypatch):
    """A demo system here declares its own scratch root; a user's ``AIPF_RAW``, which
    ``Paths.scratch`` prefers to a declared default, must not replace it."""
    monkeypatch.delenv("AIPF_RAW", raising=False)

_DOMAIN = TrustDomain(inner=(0.2, 0.15), outer=(0.9, 0.6), T_range=(500.0, 1500.0))


def _eos_csv(path):
    """A hole-free (x, T) grid of a smooth n(x, T), in the declared column names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["x_B", "T_K", "n", "status"])
        for x in np.linspace(0.0, 1.0, 11):
            for T in (500.0, 1000.0, 1500.0):
                w.writerow([f"{x:.2f}", T, 0.5 + 0.1 * x - 1e-5 * T, "ok"])


def _with_penalties(system: System, tmp_path, *, seed=7, gamma=True) -> System:
    _eos_csv(tmp_path / "eos" / "eos.csv")
    training = dict(system.defaults["training"], penalty_seed=seed,
                    conv_T_measure="log_uniform")
    if gamma:
        training["gamma_paths"] = {
            "eos_csvs": {0.0: "eos/eos.csv"},
            "eos_columns": {"x_column": "x_B", "T_column": "T_K",
                            "n_columns": ("n",), "status_column": "status",
                            "status_ok": "ok"},
            "x_grid": (0.1, 0.9, 0.05), "poly_degree": 2,
            "T_measure": "uniform", "q_floor": 1e-6}
    defaults = dict(system.defaults, training=training)
    return dataclasses.replace(system, trust_domain=_DOMAIN, defaults=defaults)


def test_the_declared_class_is_the_one_the_solver_tests_against():
    from aipf.solve import TrustDomain as solver_class
    assert solver_class is TrustDomain


def test_a_system_refuses_a_domain_of_the_wrong_width(tmp_path):
    system, _ = toy._demo_system_with_modes(tmp_path)
    with pytest.raises(ValueError, match="2 density channels"):
        dataclasses.replace(system, trust_domain=TrustDomain(inner=(0.1,),
                                                             outer=(0.9,)))
    with pytest.raises(TypeError, match="not a TrustDomain"):
        dataclasses.replace(system, trust_domain=(0.1, 0.1, 0.9, 0.9))


@pytest.mark.parametrize("measure", ["uniform", "log_uniform"])
def test_the_draw_lands_inside_the_domain_and_its_temperature_window(measure):
    rho, T = sample_trust_domain(500, _DOMAIN, T_measure=measure,
                                 generator=torch.Generator().manual_seed(0))
    assert rho.shape == (500, 2) and T.shape == (500,)
    assert bool(in_domain(rho, _DOMAIN, T).all())
    again = sample_trust_domain(500, _DOMAIN, T_measure=measure,
                                generator=torch.Generator().manual_seed(0))
    assert torch.equal(rho, again[0]) and torch.equal(T, again[1])


def test_the_draw_refuses_a_domain_without_temperatures_and_an_unknown_measure():
    bare = TrustDomain(inner=(0.2, 0.15), outer=(0.9, 0.6))
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="T_range"):
        sample_trust_domain(4, bare, T_measure="uniform", generator=g)
    with pytest.raises(ValueError, match="T_measure"):
        sample_trust_domain(4, _DOMAIN, T_measure="normal", generator=g)


@pytest.mark.parametrize("x_channel", [0, 1])
def test_gamma_on_a_path_is_the_general_closed_form(x_channel):
    g = torch.Generator().manual_seed(1)
    A = torch.randn(64, 2, 2, generator=g, dtype=torch.float64)
    H = A @ A.transpose(1, 2) + torch.eye(2, dtype=torch.float64)
    x = torch.rand(64, generator=g, dtype=torch.float64)
    n = 0.5 + torch.rand(64, generator=g, dtype=torch.float64)
    kBT = 0.1 + torch.rand(64, generator=g, dtype=torch.float64)
    v = torch.stack([1.0 - x, x] if x_channel == 1 else [x, 1.0 - x], dim=-1)
    general = gamma_closed_form(H, n, x * (1.0 - x), v, kBT)
    ours = gamma_on_path(H, n, x, kBT, x_channel=x_channel, eps=1e-6)
    assert torch.allclose(ours, general, rtol=1e-12, atol=0.0)


def test_no_domain_and_no_paths_is_no_penalty(tmp_path):
    system, _ = toy._demo_system_with_modes(tmp_path)
    cfg = TrainConfig(**_config_from_system(system), lambda_conv=1.0,
                      lambda_gamma=1.0, conv_samples=8, gamma_pt_samples=2)
    assert penalties_from_system(system, cfg) is None


def test_a_convexity_probe_of_no_points_is_refused():
    with pytest.raises(ValueError, match="conv_samples=0"):
        ConvexityProbe(domain=_DOMAIN, n=0, T_measure="uniform", form="hinge",
                       margin=0.0)


def test_gamma_paths_of_no_paths_are_refused(tmp_path):
    system, _ = toy._demo_system_with_modes(tmp_path)
    with pytest.raises(ValueError, match="gamma_pt_samples=0"):
        GammaPaths.load(system, n_paths=0, eos_csvs={}, eos_columns={},
                        x_grid=(0.1, 0.9, 0.05), poly_degree=2,
                        T_measure="uniform", q_floor=1e-6)


def test_a_trained_penalty_without_a_declared_seed_is_refused(tmp_path):
    system, _ = toy._demo_system_with_modes(tmp_path)
    system = _with_penalties(system, tmp_path, seed=None)
    cfg = TrainConfig(**_config_from_system(system), lambda_conv=1.0,
                      conv_samples=8)
    with pytest.raises(KeyError, match="penalty_seed"):
        penalties_from_system(system, cfg)


def test_one_seed_one_stream_conv_first(tmp_path):
    system, _ = toy._demo_system_with_modes(tmp_path)
    system = _with_penalties(system, tmp_path)
    cfg = TrainConfig(**_config_from_system(system), lambda_conv=1.0,
                      lambda_gamma=1.0, conv_samples=8, gamma_pt_samples=3)
    a, b = penalties_from_system(system, cfg), penalties_from_system(system, cfg)
    assert isinstance(a, Penalties) and a.seed == 7
    assert a.terms() == ("L_conv", "L_Gamma")
    for _ in range(3):
        (ra, Ta), pa = a.draw()
        (rb, Tb), pb = b.draw()
        assert torch.equal(ra, rb) and torch.equal(Ta, Tb)
        assert torch.equal(pa.n_path, pb.n_path)
    g = torch.Generator().manual_seed(7)
    rho, T = a.conv.draw(g)
    path = a.gamma.draw(g)
    (rc, Tc), pc = penalties_from_system(system, cfg).draw()
    assert torch.equal(rho, rc) and torch.equal(path.T, pc.T)


def test_fit_trains_both_penalties_when_they_are_declared(tmp_path):
    system, sources = toy._demo_system_with_modes(tmp_path)
    system = _with_penalties(system, tmp_path)
    run = fit(system, run_name="pen", sources=sources, steps=2, seed=0,
              resume_optimizer=False, root=tmp_path / "data",
              config_overrides={"lambda_conv": 1.0, "lambda_gamma": 1.0,
                                "conv_samples": 8, "gamma_pt_samples": 3},
              log_every_step=True)
    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["terms_trained"] == ["L_dyn", "L_conv", "L_Gamma"]
    assert man["declared_weights_without_data"] == {}
    assert man["penalties"]["penalty_seed"] == 7
    assert man["penalties"]["terms"] == ["L_conv", "L_Gamma"]
    assert not (run / "UNTRAINED_TERMS.txt").exists()
    terms = json.loads((run / "steps.json").read_text())["terms"]
    assert all({"L_conv", "L_Gamma"} <= set(t) for t in terms)


def test_a_declared_domain_does_not_open_a_zero_weighted_penalty(tmp_path):
    system, sources = toy._demo_system_with_modes(tmp_path)
    system = _with_penalties(system, tmp_path, gamma=False)
    run = fit(system, run_name="off", sources=sources, steps=1, seed=0,
              resume_optimizer=False, root=tmp_path / "data")
    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["terms_trained"] == ["L_dyn"] and man["penalties"] is None
