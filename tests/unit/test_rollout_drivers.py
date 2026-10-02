"""Every rollout knob is declared, the observables measure what
they say, and the semi-implicit scheme conserves mass exactly on a toy of the declared form."""
import dataclasses

import numpy as np
import pytest
import torch

from aipf.rollout.imex import rollout_imex
from aipf.rollout.observables import (conc_profile, decomp_metrics,
                                      extract_plateaus, interface_width,
                                      tier2_ok)
from aipf.rollout.spinodal import declared, trust_domain
from aipf.solve import UNDECLARED
from aipf.system import load

_DECL = {
    "solver": dict(clamp_rho=1e-3, state_proj="domain", state_clamp=1e-3,
                   mass_restore="shift", noise_eval="ito", noise_scale=1.0,
                   predictor_floor=1e-4),
    "archive": dict(file_name="m.npz", amplitudes="a",
                    amplitudes_channel_axis=2, labels="n", box="b",
                    temperature="T", frame_interval="dt",
                    quality_file=None, quality_key=None),
    "slab": dict(modes_tree="x", grid=(8, 8, 8), field_stride=1,
                 profile_axis=2),
}


def test_a_system_without_a_rollout_declaration_is_refused_naming_the_keys():
    with pytest.raises(ValueError, match="declares no defaults\\['rollout'\\]"):
        declared(load("lj"), "slab")


def test_a_missing_or_unread_key_is_refused():
    bad = {**_DECL, "solver": {k: v for k, v in _DECL["solver"].items()
                               if k != "clamp_rho"}}
    with pytest.raises(ValueError, match="missing \\['clamp_rho'\\]"):
        declared(load("hhe"), "slab", bad)
    extra = {**_DECL, "slab": {**_DECL["slab"], "k_max": 2.0}}
    with pytest.raises(ValueError, match="not read \\['k_max'\\]"):
        declared(load("hhe"), "slab", extra)
    assert declared(load("hhe"), "slab", _DECL)["slab"]["profile_axis"] == 2


@pytest.fixture(scope="module")
def toy():
    from aipf.functional.build import build
    torch.manual_seed(0)
    return build(load("hhe")).eval()


def _field(grid=(8, 8, 8)):
    from aipf.spectral import SpectralOps
    g = torch.Generator().manual_seed(3)
    rho = torch.stack([0.40 + 0.01 * torch.randn(grid, generator=g),
                       0.30 + 0.01 * torch.randn(grid, generator=g)]).unsqueeze(0)
    ops = SpectralOps(grid, 2, nyquist_mask=True)
    return ops.rfft(rho) / (grid[0] * grid[1] * grid[2])


def _kw(**over):
    kw = dict(kB=8.617333262e-5, m_stab="max", state_proj="domain",
              state_clamp=1e-3, domain=trust_domain(load("hhe")),
              mass_restore="shift", clamp_rho=1e-3)
    kw.update(over)
    return kw


@pytest.mark.parametrize("knob,value,match", [
    ("m_stab", UNDECLARED, "m_stab must be declared"),
    ("m_stab", "median", "m_stab must be one of"),
    ("mass_restore", "uniform", "mass_restore must be one of"),
    ("state_proj", "box", "state_proj must be 'floor' or 'domain'"),
    ("state_clamp", UNDECLARED, "state_clamp must be declared"),
    ("clamp_rho", UNDECLARED, "clamp_rho must be declared"),
])
def test_every_knob_is_declared(toy, knob, value, match):
    with pytest.raises(ValueError, match=match):
        rollout_imex(toy, _field(), torch.tensor([7., 7., 7.]), 7000.0, 1e-4,
                     1, **_kw(**{knob: value}))


def test_a_functional_without_a_pair_kernel_is_refused():
    class Bare(torch.nn.Module):
        pass
    with pytest.raises(TypeError, match="kernel.w_hat"):
        rollout_imex(Bare(), _field(), torch.tensor([7., 7., 7.]), 7000.0,
                     1e-4, 1, **_kw())


@pytest.mark.parametrize("noisy", [False, True])
def test_mass_is_exact_and_the_state_real(toy, noisy):
    h0 = _field()
    noise = None if not noisy else dict(
        kBT_noise=0.6, noise_scale=1.0, noise_mode="gaussian",
        sigma_noise=2.0, noise_eval="ito", predictor_floor=1e-4)
    traj = rollout_imex(toy, h0, torch.tensor([7., 7.5, 8.]), 7000.0, 1e-4, 5,
                        noise=noise, generator=torch.Generator().manual_seed(1),
                        **_kw())
    assert torch.equal(traj[:, :, 0, 0, 0].real,
                       h0[0, :, 0, 0, 0].real.expand(6, 2))
    assert float(traj[:, :, 0, 0, 0].imag.abs().max()) == 0.0
    assert torch.isfinite(traj.real).all()


def test_zero_noise_amplitude_is_the_deterministic_scheme(toy):
    kw = _kw()
    a = rollout_imex(toy, _field(), torch.tensor([7., 7., 7.]), 7000.0, 1e-4,
                     3, **kw)
    b = rollout_imex(toy, _field(), torch.tensor([7., 7., 7.]), 7000.0, 1e-4,
                     3, noise=dict(kBT_noise=0.0, noise_scale=1.0,
                                   noise_mode="gaussian", sigma_noise=2.0,
                                   noise_eval="ito", predictor_floor=1e-4),
                     **kw)
    assert torch.equal(a, b)


def test_a_single_cosine_has_its_length_at_the_box():
    G, L = 16, 20.0
    x = np.arange(G) / G
    c = 0.5 + 0.1 * np.cos(2 * np.pi * x)[:, None, None] * np.ones((G, G, G))
    rho = np.stack([1 - c, c])[None]
    Phi, Lm, Sk = decomp_metrics(rho, np.array([L]), 1)
    assert Lm[0] == pytest.approx(L)
    assert Sk[0, 0] > 0 and np.allclose(Sk[0, 1:], 0, atol=1e-18)


def test_the_slab_profile_plateaus_and_width():
    G = 64
    z = np.arange(G)
    c = np.where((z >= 16) & (z < 48), 0.9, 0.1).astype(float)
    rho = np.stack([1 - c, c])[None, :, None, None, :] * np.ones((1, 2, 4, 4, G))
    prof = conc_profile(rho, 1, 2)
    assert extract_plateaus(prof[0]) == pytest.approx((0.1, 0.9))
    assert interface_width(prof[0], 0.5) == pytest.approx(0.8 * 0.5)
    flat = np.full(G, 0.3)
    assert np.isnan(interface_width(flat, 0.5))


def test_tier2_refuses_collapsed_conventions():
    res = {"k_mode": np.array([0.5, 1.0]), "w_mode": np.array([1.0, 2.0]),
           "r_tr_mode": np.array([1.0, 1.02]), "r_cc_mode": np.array([1.0, 0.98])}
    same = {m: dict(res) for m in ("ito", "midpoint", "kinetic")}
    ok, detail = tier2_ok(same, ("ito", "midpoint", "kinetic"), 1.5, 0.05,
                          1e-4, 0.5)
    assert not ok and not detail["control_ok"]
    ok, detail = tier2_ok({"ito": res}, ("ito", "midpoint", "kinetic"), 1.5,
                          0.05, 1e-4, 0.5)
    assert not ok and detail["missing"] == ["midpoint", "kinetic"]


def test_the_cli_requires_every_flag():
    from aipf.cli.main import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["rollout", "slab", "--system", "hhe"])
    args = build_parser().parse_args(
        ["rollout", "slab", "--system", "hhe", "--ckpt", "published", "--run",
         "r", "--seeds", "--t-end", "1", "--dt", "1e-4", "--save-ps", "0.1",
         "--device", "cpu", "--out", "data"])
    assert args.seeds == [] and args.driver == "slab"


def test_a_system_without_a_trust_domain_is_refused_by_name():
    system = dataclasses.replace(load("hhe"), trust_domain=None)
    with pytest.raises(ValueError, match="declares no trust_domain"):
        trust_domain(system)
    assert trust_domain(load("hhe")).inner == load("hhe").trust_domain.inner
