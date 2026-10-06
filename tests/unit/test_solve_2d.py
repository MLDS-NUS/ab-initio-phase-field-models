"""Both solvers on a declared two-dimensional model: mass, reality, the z-invariant 3D twin, and S(k).

The model is ``tests/toy_ndim_model.py``, built through the factory door on a ``(Gx, Gy)`` grid and,
for the twin, on ``(Gx, Gy, Gz)`` with the same weights. A field that does not vary along z stays so
under the 3D dynamics, and every z slice of its 3D rollout is the 2D rollout: that checks the 2D
solvers against the 3D ones, which ``tests/golden`` pins bit for bit. It does not check the physics of a
z average (the k_z != 0 modes of a real 3D field couple into k_z = 0; docs/reference/functional.md, "Two
dimensions"). Noise is left out of the twin, since the two draw different numbers by construction; its
amplitude is checked instead against the stationary spectrum of a linear model, ``S(k) = kBT / H(k)``,
which reads ``depth`` through the cell volume ``dV = dA * depth``."""
from __future__ import annotations

import dataclasses
import math

import pytest
import torch

import toy_factory_model as toy
from aipf.functional.build import build
from aipf.rollout.imex import rollout_imex
from aipf.solve import (hermitianize, project_state, rollout_deterministic, rollout_sde,
                        step_sde_euler_maruyama)
from aipf.spectral import SpectralOps2D
from aipf.system import TrustDomain
from toy_ndim_model import NdimToyModel, build_ndim_toy

GRID2 = (8, 6)
GZ = 4
GRID3 = (*GRID2, GZ)
BOX3 = torch.tensor([8.0, 7.0, 3.0])
BOX2 = BOX3[:2]
T = torch.tensor([1.0])
EXPLICIT = dict(state_proj="floor", floor=0.0, clamp_rho=None)
IMEX = dict(kB=1.0, state_proj="floor", state_clamp=0.0, clamp_rho=None)
NOISE = dict(kBT_noise=0.02, noise_scale=1.0, noise_mode="none", sigma_noise=None,
             noise_eval="ito", predictor_floor=1e-3)


@pytest.fixture(autouse=True)
def _one_thread():
    """Fields of a few dozen cells: a thread pool only adds its overhead, and on a shared machine a lot of it."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _system(grid):
    base = toy.demo_system("unused")
    return dataclasses.replace(base, functional=toy.toy_functional(build_ndim_toy, grid=grid),
                               variants={})


@pytest.fixture(scope="module")
def twins():
    """The toy on the 2D grid and on the 3D one, with the same (perturbed) weights."""
    torch.manual_seed(0)
    m3 = build(_system(GRID3))
    with torch.no_grad():
        for p in m3.parameters():
            p.add_(0.2 * torch.randn_like(p))
    m2 = build(_system(GRID2))
    weights = {k: v for k, v in m3.state_dict().items() if not k.startswith("ops.")}
    m2.load_state_dict({**m2.state_dict(), **weights})
    return m2, m3


def _rho2(seed=1, batch=1):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([0.50 + 0.05 * torch.randn(batch, *GRID2, generator=g),
                        0.40 + 0.05 * torch.randn(batch, *GRID2, generator=g)], dim=1)


def _hats(rho2, m2, m3):
    rho3 = rho2.unsqueeze(-1).expand(*rho2.shape, GZ).contiguous()
    return (m2.ops.rfft(rho2) / math.prod(GRID2), m3.ops.rfft(rho3) / math.prod(GRID3))


def _real(ops, traj):
    return ops.irfft(traj * math.prod(ops.grid))


def _assert_mass_exact_and_real(traj, ops):
    """``k = 0`` bit for bit at every save, and every state the half spectrum of a real field."""
    dc = traj[..., 0, 0]
    assert torch.equal(dc, dc[:1].expand_as(dc))
    assert torch.equal(dc.imag, torch.zeros_like(dc.imag))
    N = math.prod(ops.grid)
    assert torch.allclose(ops.rfft(ops.irfft(traj * N)) / N, traj, atol=1e-6)


# ---------------------------------------------------------------------------
# the factory door on a two-axis grid
# ---------------------------------------------------------------------------

def test_a_factory_declared_on_a_two_axis_grid_builds_a_two_dimensional_model(twins):
    m2, m3 = twins
    assert isinstance(m2.ops, SpectralOps2D) and m2.ops.ndim == 2 and m2._cache.grid == GRID2
    assert m3.ops.ndim == 3
    assert set(m2.state_dict()) == {k for k in m3.state_dict() if k != "ops.NZ"}


# ---------------------------------------------------------------------------
# mass and reality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["euler", "heun"])
def test_the_explicit_2d_rollout_keeps_mass_exactly_and_the_state_real(twins, method):
    m2, m3 = twins
    h2, _ = _hats(_rho2(batch=2), m2, m3)
    traj = rollout_deterministic(m2, h2, BOX2.expand(2, 2), T.expand(2), 1e-3, 12,
                                 method=method, **EXPLICIT)
    assert tuple(traj.shape) == (13, 2, 2, GRID2[0], GRID2[1] // 2 + 1)
    assert float((traj[-1] - traj[0]).abs().max()) > 1e-5
    _assert_mass_exact_and_real(traj, m2.ops)


def test_the_2d_sde_keeps_mass_exactly_and_the_state_real(twins):
    m2, m3 = twins
    h2, _ = _hats(_rho2(batch=2), m2, m3)
    gen = torch.Generator().manual_seed(5)
    traj = rollout_sde(m2, h2, BOX2.expand(2, 2), T.expand(2), 1e-3, 12, 0.02,
                       noise_mode="gaussian", sigma_noise=1.0, generator=gen, depth=3.0,
                       kappa_roll=0.1, **EXPLICIT)
    _assert_mass_exact_and_real(traj, m2.ops)


@pytest.mark.parametrize("noise_eval", [None, "ito", "midpoint"])
@pytest.mark.parametrize("mass_restore", ["shift", "headroom"])
def test_the_2d_semi_implicit_scheme_keeps_mass_exactly_and_the_state_real(twins, noise_eval,
                                                                           mass_restore):
    m2, m3 = twins
    h2, _ = _hats(_rho2(), m2, m3)
    noise = None if noise_eval is None else {**NOISE, "noise_eval": noise_eval}
    traj = rollout_imex(m2, h2, BOX2, 1.0, 1e-2, 10, m_stab="max", noise=noise,
                        generator=torch.Generator().manual_seed(2), mass_restore=mass_restore,
                        depth=None if noise is None else 3.0,
                        **{**IMEX, "state_clamp": 0.3})
    assert tuple(traj.shape) == (11, 2, GRID2[0], GRID2[1] // 2 + 1)
    _assert_mass_exact_and_real(traj.unsqueeze(1), m2.ops)


@pytest.mark.parametrize("state_proj, kw", [
    ("floor", dict(floor=0.42)),
    ("box", dict(lo=0.42, hi=0.58)),
    ("domain", dict(domain=TrustDomain(inner=(0.5, 0.4), outer=(1.0, 0.6)), lo=0.05)),
])
def test_the_2d_state_projection_keeps_mass_exactly_and_lands_in_the_set(state_proj, kw):
    ops = SpectralOps2D(GRID2, 2, nyquist_mask=True)
    rho = _rho2(seed=4, batch=2)
    h = ops.rfft(rho) / math.prod(GRID2)
    out = project_state(hermitianize(h, ops), ops, kw.pop("floor", None), state_proj=state_proj,
                        **kw)
    assert torch.equal(out[..., 0, 0], h[..., 0, 0].real.to(h.dtype))
    real = _real(ops, out)
    if state_proj in ("floor", "box"):
        # channel 0 (mean 0.50) is lifted to the floor; channel 1 (mean 0.40) cannot be, and drains
        assert float(real[:, 0].min()) >= 0.42 - 1e-5
        flat = real[:, 1].flatten(1)
        assert float((flat.max(1).values - flat.min(1).values).max()) < 1e-5
    if state_proj == "box":
        assert float(real[:, 0].max()) <= 0.58 + 1e-3     # the mass restore lifts uniformly


# ---------------------------------------------------------------------------
# the z-invariant 3D twin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["euler", "heun"])
def test_a_2d_explicit_rollout_is_its_z_invariant_3d_twin_sliced_in_z(twins, method):
    m2, m3 = twins
    h2, h3 = _hats(_rho2(batch=2), m2, m3)
    g = torch.Generator().manual_seed(9)
    v2 = 0.05 * torch.randn(2, *GRID2, generator=g)
    v3 = v2.unsqueeze(-1).expand(2, *GRID3).contiguous()
    T2 = T.expand(2)
    common = dict(method=method, kappa_roll=0.05, **EXPLICIT)
    t2 = rollout_deterministic(m2, h2, BOX2.expand(2, 2), T2, 2e-3, 15, v_ext=v2, **common)
    t3 = rollout_deterministic(m3, h3, BOX3.expand(2, 3), T2, 2e-3, 15, v_ext=v3, **common)
    r2, r3 = _real(m2.ops, t2), _real(m3.ops, t3)
    assert torch.allclose(r3, r3[..., :1].expand_as(r3), atol=1e-6)
    assert float((r2 - r2[0]).abs().max()) > 1e-4
    assert torch.allclose(r3[..., 0], r2, atol=2e-6)


@pytest.mark.parametrize("m_stab", ["mean", "max"])
def test_a_2d_semi_implicit_rollout_is_its_z_invariant_3d_twin_sliced_in_z(twins, m_stab):
    m2, m3 = twins
    h2, h3 = _hats(_rho2(), m2, m3)
    kbt2 = 1.0 + 0.1 * torch.rand(*GRID2, generator=torch.Generator().manual_seed(3))
    kbt3 = kbt2.unsqueeze(-1).expand(*GRID3).contiguous()
    t2 = rollout_imex(m2, h2, BOX2, 1.0, 2e-2, 10, m_stab=m_stab, kbt_field=kbt2,
                      mass_restore="headroom", **IMEX)
    t3 = rollout_imex(m3, h3, BOX3, 1.0, 2e-2, 10, m_stab=m_stab, kbt_field=kbt3,
                      mass_restore="headroom", **IMEX)
    r2, r3 = _real(m2.ops, t2), _real(m3.ops, t3)
    assert torch.allclose(r3, r3[..., :1].expand_as(r3), atol=1e-6)
    assert float((r2 - r2[0]).abs().max()) > 1e-4
    assert torch.allclose(r3[..., 0], r2, atol=2e-6)


def test_the_stabiliser_hook_is_handed_the_2d_real_field(twins):
    m2, _ = twins
    seen = []

    class Hooked(NdimToyModel):
        def stabilizer_mobility(self, rho, T):
            seen.append(tuple(rho.shape))
            return self.mobility(rho, T)[0, :, :, 0, 0]

    hooked = Hooked(GRID2, 2, nyquist_mask=True, kappa=0.5, kB=1.0)
    hooked.load_state_dict(m2.state_dict())
    h2, _ = _hats(_rho2(), m2, twins[1])
    a = rollout_imex(hooked, h2, BOX2, 1.0, 1e-2, 4, mass_restore="shift", **IMEX)
    b = rollout_imex(m2, h2, BOX2, 1.0, 1e-2, 4, m_stab="mean", mass_restore="shift", **IMEX)
    assert seen == [(1, 2, *GRID2)]
    assert torch.allclose(a, b, atol=1e-7)


# ---------------------------------------------------------------------------
# the noise amplitude: S(k) of a linear model
# ---------------------------------------------------------------------------

DEPTH = 2.5
KBT = 0.05


def _linear(grid):
    model = NdimToyModel(grid, 1, nyquist_mask=True, kappa=1.0, kB=1.0, linear_a=1.0)
    M = float(model.mobility_matrix()[0, 0])
    return model, M


def _ratio(states, ops, box, M, dt, scheme):
    """Mean over ``0 < |k|^2 <= 4`` of ``V <|rho_hat|^2> / (kBT / H(k))``, with the scheme's own exact
    stationary factor for a linear mode of rate ``lam`` (implicit: ``1 / (1 + lam dt / 2)``, Euler-Maruyama:
    ``1 / (1 - lam dt / 2)``)."""
    k2 = ops.k2(box.view(1, 2))[0, 0]
    H = 1.0 + k2
    lam = M * k2 * H
    factor = 1.0 + lam * dt / 2.0 if scheme == "implicit" else 1.0 - lam * dt / 2.0
    V = float(box.prod()) * DEPTH
    S = V * (states.abs() ** 2).mean(0).reshape(k2.shape)
    band = (k2 > 0) & (k2 <= 4.0)
    return float((S / (KBT / H / factor))[band].mean())


def test_the_2d_semi_implicit_noise_gives_s_of_k_equal_kbt_over_h_at_the_declared_depth():
    grid, box = (12, 12), torch.tensor([12.0, 12.0])
    model, M = _linear(grid)
    h0 = torch.zeros(1, 1, grid[0], grid[1] // 2 + 1, dtype=torch.complex64)
    h0[..., 0, 0] = 1.0
    dt = 0.05
    noise = {**NOISE, "kBT_noise": KBT, "predictor_floor": 0.0}
    traj = rollout_imex(model, h0, box, 1.0, dt, 3000, m_stab="mean", noise=noise,
                        generator=torch.Generator().manual_seed(3), save_every=5, depth=DEPTH,
                        mass_restore="shift", **IMEX)
    assert abs(_ratio(traj[60:], model.ops, box, M, dt, "implicit") - 1.0) < 0.08


def test_the_2d_sde_noise_gives_s_of_k_equal_kbt_over_h_at_the_declared_depth():
    grid, box, B = (8, 8), torch.tensor([8.0, 8.0]), 8
    model, M = _linear(grid)
    h0 = torch.zeros(B, 1, grid[0], grid[1] // 2 + 1, dtype=torch.complex64)
    h0[..., 0, 0] = 1.0
    dt = 0.004
    traj = rollout_sde(model, h0, box.expand(B, 2), torch.ones(B), dt, 4000, KBT,
                       noise_mode="none", sigma_noise=None,
                       generator=torch.Generator().manual_seed(3), save_every=20, depth=DEPTH,
                       **EXPLICIT)
    states = traj[20:].transpose(0, 1).reshape(-1, 1, grid[0], grid[1] // 2 + 1)
    assert abs(_ratio(states, model.ops, box, M, dt, "euler_maruyama") - 1.0) < 0.08


# ---------------------------------------------------------------------------
# depth: declared for 2D noise, absent in 3D
# ---------------------------------------------------------------------------

def test_2d_noise_without_a_depth_is_refused_and_a_deterministic_2d_rollout_needs_none(twins):
    m2, m3 = twins
    h2, _ = _hats(_rho2(), m2, m3)
    with pytest.raises(ValueError, match="depth=1.0 for areal"):
        rollout_sde(m2, h2, BOX2.view(1, 2), T, 1e-3, 2, 0.02, noise_mode="none",
                    sigma_noise=None, **EXPLICIT)
    with pytest.raises(ValueError, match="depth=1.0 for areal"):
        step_sde_euler_maruyama(m2, h2, BOX2.view(1, 2), T, 1e-3, 0.02, noise_mode="none",
                                sigma_noise=None)
    with pytest.raises(ValueError, match="depth=1.0 for areal"):
        rollout_imex(m2, h2, BOX2, 1.0, 1e-2, 2, m_stab="mean", noise=NOISE,
                     mass_restore="shift", **IMEX)
    with pytest.raises(ValueError, match="finite and positive"):
        rollout_sde(m2, h2, BOX2.view(1, 2), T, 1e-3, 2, 0.02, noise_mode="none",
                    sigma_noise=None, depth=0.0, **EXPLICIT)
    # kBT_noise = 0 is the deterministic step, and a deterministic 2D rollout reads no depth
    rollout_sde(m2, h2, BOX2.view(1, 2), T, 1e-3, 2, 0.0, noise_mode="none", sigma_noise=None,
                **EXPLICIT)
    rollout_imex(m2, h2, BOX2, 1.0, 1e-2, 2, m_stab="mean", mass_restore="shift", **IMEX)


def test_a_depth_declared_for_a_3d_model_is_refused(twins):
    _, m3 = twins
    h3 = m3.ops.rfft(0.5 + torch.zeros(1, 2, *GRID3)) / math.prod(GRID3)
    with pytest.raises(ValueError, match="three-dimensional model"):
        rollout_sde(m3, h3, BOX3.view(1, 3), T, 1e-3, 2, 0.02, noise_mode="none",
                    sigma_noise=None, depth=1.0, **EXPLICIT)
    with pytest.raises(ValueError, match="three-dimensional model"):
        step_sde_euler_maruyama(m3, h3, BOX3.view(1, 3), T, 1e-3, 0.02, noise_mode="none",
                                sigma_noise=None, depth=1.0)
    with pytest.raises(ValueError, match="three-dimensional model"):
        rollout_imex(m3, h3, BOX3, 1.0, 1e-2, 2, m_stab="mean", noise=NOISE, depth=1.0,
                     mass_restore="shift", **IMEX)
