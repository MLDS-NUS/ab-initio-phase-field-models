"""The semi-implicit scheme's external fields. A static ``v_ext`` is added
to the pointwise ``mu`` and a per-cell ``kbt_field`` replaces its temperature, on full-grid calls only;
both are validated before anything is installed and removed afterwards."""
import pytest
import torch

from aipf.rollout.imex import pointwise_fields, rollout_imex
from aipf.rollout.spinodal import trust_domain
from aipf.spectral import SpectralOps
from aipf.system import load

GRID = (8, 8, 8)
BOX = torch.tensor([7.0, 7.0, 7.0])
KB = 8.617333262e-5


@pytest.fixture(scope="module")
def toy():
    from aipf.functional.build import build
    torch.manual_seed(0)
    return build(load("hhe")).eval()


def _field():
    g = torch.Generator().manual_seed(3)
    rho = torch.stack([0.40 + 0.01 * torch.randn(GRID, generator=g),
                       0.30 + 0.01 * torch.randn(GRID, generator=g)]).unsqueeze(0)
    return SpectralOps(GRID, 2, nyquist_mask=True).rfft(rho) / 512


def _roll(model, T=9000.0, steps=5, noise=None, generator=None, **fields):
    return rollout_imex(model, _field(), BOX, T, 1e-4, steps, kB=KB, m_stab="max",
                        state_proj="domain", state_clamp=1e-3,
                        domain=trust_domain(load("hhe")), mass_restore="shift",
                        clamp_rho=1e-3, noise=noise, generator=generator, **fields)


def _noise(T):
    return {"kBT_noise": KB * T, "noise_scale": 1.0, "noise_mode": "gaussian",
            "sigma_noise": 2.0, "noise_eval": "ito", "predictor_floor": 1e-4}


@pytest.mark.parametrize("fields,match", [
    ({"v_ext": torch.zeros(2, 8, 8, 4)}, "v_ext must have shape"),
    ({"kbt_field": torch.ones(2, 8, 8, 8)}, "kbt_field must have shape"),
    ({"kbt_field": torch.zeros(8, 8, 8)}, "finite and positive"),
    ({"kbt_field": torch.ones(8, 8, 8), "v_ext": torch.zeros(3, 8, 8, 8)},
     "v_ext must have shape"),
])
def test_a_wrong_field_is_refused_and_nothing_stays_installed(toy, fields, match):
    before = dict(toy.f_local.__dict__)
    with pytest.raises(ValueError, match=match):
        _roll(toy, **fields)
    assert "mu_pointwise" not in toy.f_local.__dict__
    assert set(toy.f_local.__dict__) == set(before)


def test_a_zero_potential_is_the_scheme_without_one(toy):
    assert torch.equal(_roll(toy), _roll(toy, v_ext=torch.zeros(2, *GRID)))
    assert "mu_pointwise" not in toy.f_local.__dict__


def test_a_uniform_temperature_field_is_the_scalar_temperature(toy):
    """Equal to float32 round-off of ``kB T`` (the two float32 renderings of ``kBT`` need not coincide)."""
    T = 9000.0
    a = _roll(toy, T)
    b = _roll(toy, T, kbt_field=torch.full(GRID, KB * T))
    assert torch.allclose(a, b, rtol=0.0, atol=1e-7)


def test_the_potential_and_temperature_act_only_on_full_grid_calls(toy):
    V = torch.randn(2, *GRID)
    kbt = torch.full(GRID, KB * 7000.0)
    probe = torch.tensor([[0.4, 0.3]])
    k1 = torch.tensor([KB * 9000.0])
    rho = SpectralOps(GRID, 2, nyquist_mask=True).irfft(_field() * 512)
    cells = rho.permute(0, 2, 3, 4, 1).reshape(-1, 2)
    full_T = torch.full((512,), KB * 9000.0)
    mu0_probe = toy.f_local.mu_pointwise(probe, k1)
    mu0_cells = toy.f_local.mu_pointwise(cells, kbt.reshape(-1))
    with pointwise_fields(toy, _field(), kbt, V):
        assert torch.equal(toy.f_local.mu_pointwise(probe, k1), mu0_probe)
        got = toy.f_local.mu_pointwise(cells, full_T)
    assert torch.equal(got, mu0_cells + V.permute(1, 2, 3, 0).reshape(-1, 2))


def test_a_species_collects_where_its_potential_is_low(toy):
    """``cos(2 pi z / L)`` on the second species only: after a few steps it is denser at ``z = L/2``."""
    z = torch.arange(8) * (7.0 / 8)
    V = torch.zeros(2, *GRID)
    V[1] = (0.05 * torch.cos(2 * torch.pi * z / 7.0)).view(1, 1, 8)
    out = _roll(toy, 12000.0, steps=20, v_ext=V)
    rho = SpectralOps(GRID, 2, nyquist_mask=True).irfft(out[-1:] * 512)[0]
    ref = SpectralOps(GRID, 2, nyquist_mask=True).irfft(_roll(toy, 12000.0, steps=20)[-1:] * 512)[0]
    shift = (rho - ref)[1].mean(dim=(0, 1))
    assert shift[4] > 0 > shift[0]


def test_noise_under_a_temperature_field_keeps_the_mass(toy):
    kbt = torch.full(GRID, KB * 9000.0)
    kbt[:, :, 4:] = KB * 11000.0
    g = torch.Generator().manual_seed(1)
    out = _roll(toy, 10000.0, steps=5, noise=_noise(10000.0), generator=g,
                kbt_field=kbt, v_ext=torch.zeros(2, *GRID))
    assert torch.allclose(out[-1, :, 0, 0, 0], out[0, :, 0, 0, 0], rtol=0, atol=1e-6)
    assert torch.isfinite(out.real).all()
