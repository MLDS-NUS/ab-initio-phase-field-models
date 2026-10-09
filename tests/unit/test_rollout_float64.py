"""The semi-implicit scheme in float64: selected by the dtype of the state and the model, end to end.

A ``complex128`` state with a float64 model (``model.double()``) runs every operator, field, noise draw and
projection in float64 / complex128; the wavenumbers come from exact integer mode indices. Checked here
against an independent float64 numpy implementation of the documented update, against its own z-invariant
3D twin, and by an audit of every tensor a torch function returns during the rollout. A mixed pair is
refused. The float32 path is pinned bit for bit by ``tests/golden``, which this file does not touch.

The model is ``tests/toy_ndim_model.py``: ``W(k) = kappa k^2``, a polynomial local free energy and a
constant, full mobility."""
from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten

import toy_factory_model as toy
from aipf.functional.build import build
from aipf.rollout import imex as imex_mod
from aipf.rollout.imex import rollout_imex
from aipf.spectral import (OpsCache, SpectralOps, SpectralOps2D, exact_mode_indices,
                           make_ops)
from aipf.system import TrustDomain
from toy_ndim_model import NdimToyModel, build_ndim_toy

GRID2 = (8, 6)
GZ = 4
GRID3 = (*GRID2, GZ)
BOX3 = torch.tensor([8.0, 7.0, 3.0], dtype=torch.float64)
BOX2 = BOX3[:2]
T = 1.0
KB = 1.0
DET = dict(kB=KB, state_proj="floor", state_clamp=0.0, clamp_rho=None, mass_restore="shift")
NOISE = dict(kBT_noise=0.02, noise_scale=1.0, noise_mode="gaussian", sigma_noise=1.0,
             noise_eval="midpoint", predictor_floor=1e-3)
#: The dtypes a float64 run must never produce.
SINGLE = (torch.float32, torch.complex64, torch.float16, torch.bfloat16)


@pytest.fixture(autouse=True)
def _one_thread():
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
    """The toy on the 2D grid and on the 3D one, the same perturbed weights, both cast to float64."""
    torch.manual_seed(0)
    m3 = build(_system(GRID3))
    with torch.no_grad():
        for p in m3.parameters():
            p.add_(0.2 * torch.randn_like(p))
    m2 = build(_system(GRID2))
    weights = {k: v for k, v in m3.state_dict().items() if not k.startswith("ops.")}
    m2.load_state_dict({**m2.state_dict(), **weights})
    return m2.double().eval(), m3.double().eval()


def _rho2(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([0.50 + 0.05 * torch.randn(*GRID2, generator=g, dtype=torch.float64),
                        0.40 + 0.05 * torch.randn(*GRID2, generator=g, dtype=torch.float64)]
                       ).unsqueeze(0)


def _hat2(rho2):
    return torch.fft.rfftn(rho2, dim=(-2, -1)) / math.prod(GRID2)


def _mode_numbers(G):
    """``0, 1, .., -2, -1``: the full axis's integer mode numbers, built from integers."""
    return torch.tensor(np.r_[0:(G + 1) // 2, -(G // 2):0], dtype=torch.float64)


def _real(ops, traj):
    return ops.irfft(traj * math.prod(ops.grid))


# ---------------------------------------------------------------------------
# the independent reference: the documented update, written out in numpy
# ---------------------------------------------------------------------------

def _softplus(x):
    return np.log1p(np.exp(x))


def _toy_parameters(model):
    """``a, b, chi, kappa`` and the constant mobility ``M = L L^T + 0.1 I``, read off the parameters."""
    p = {k: v.detach().numpy() for k, v in model.named_parameters()}
    n = model.n_species
    L = np.zeros((n, n))
    L[np.tril_indices(n)] = p["mobility_raw"]
    L[np.diag_indices(n)] = _softplus(np.diag(L))
    return dict(a=_softplus(p["f_local.a_raw"]), b=_softplus(p["f_local.b_raw"]),
                chi=float(p["f_local.chi"]), kappa=_softplus(p["kernel.kappa_raw"]),
                M=L @ L.T + 0.1 * np.eye(n))


def _numpy_imex(model, rho0, box, dt, n_steps):
    """``rho^{n+1} = A^-1 (rho^n + dt F(rho^n) + dt k^2 M_s H rho^n)`` per mode, then the real-field
    projection of the half spectrum (k = 0 restored); ``A = I + dt k^2 M_s H(k)``,
    ``H = Hess f_loc(rho_bar) + kappa k^2``, ``M_s = M`` (constant), ``F = div(M grad mu)`` with the Nyquist
    wavenumber zeroed in grad and div. ``rho0`` ``(n, Gx, Gy)`` real; returns the half spectra
    ``(n_steps + 1, n, Gx, Gy // 2 + 1)`` in the state's convention (``rfft / N``)."""
    q = _toy_parameters(model)
    n, (Gx, Gy) = rho0.shape[0], rho0.shape[1:]
    N = Gx * Gy
    kBT = KB * T
    nx = np.r_[0:(Gx + 1) // 2, -(Gx // 2):0].astype(np.float64)
    ny = np.arange(Gy // 2 + 1, dtype=np.float64)
    kx = (2 * np.pi * nx / box[0])[:, None] * np.ones((1, Gy // 2 + 1))
    ky = np.ones((Gx, 1)) * (2 * np.pi * ny / box[1])[None, :]
    mx = np.where(np.arange(Gx) == Gx // 2, 0.0, 1.0)[:, None] if Gx % 2 == 0 else 1.0
    my = np.where(np.arange(Gy // 2 + 1) == Gy // 2, 0.0, 1.0)[None, :] if Gy % 2 == 0 else 1.0
    k2 = kx ** 2 + ky ** 2
    a, b, chi, kappa, M = q["a"], q["b"], q["chi"], q["kappa"], q["M"]

    def rfft(f):
        return np.fft.rfft2(f, axes=(-2, -1))

    def irfft(f):
        return np.fft.irfft2(f, s=(Gx, Gy), axes=(-2, -1))

    def drift(rho_hat):
        rho = irfft(rho_hat * N)
        mu = (a[:, None, None] + kBT) * rho + b[:, None, None] * rho ** 3
        mu = mu + chi * rho[::-1]
        mu = mu + irfft(kappa[:, None, None] * k2 * rfft(rho))
        mu_hat = rfft(mu)
        gx, gy = irfft(1j * kx * mx * mu_hat), irfft(1j * ky * my * mu_hat)
        Jx, Jy = np.einsum("ij,jxy->ixy", M, gx), np.einsum("ij,jxy->ixy", M, gy)
        return 1j * (kx * mx * rfft(Jx) / N + ky * my * rfft(Jy) / N)

    rho_hat = rfft(rho0) / N
    rho_bar = rho_hat[:, 0, 0].real
    Hb = np.diag(a + 3 * b * rho_bar ** 2 + kBT) + chi * (1 - np.eye(n))
    H = kappa[None, None, :, None] * np.eye(n)[None, None] * k2[:, :, None, None] + Hb
    dtL = dt * k2[:, :, None, None] * np.einsum("ij,xyjk->xyik", M, H)
    A_inv = np.linalg.inv(np.eye(n) + dtL)
    out = [rho_hat]
    for _ in range(n_steps):
        rh = np.moveaxis(rho_hat, 0, -1)[..., None]
        Fh = np.moveaxis(drift(rho_hat), 0, -1)[..., None]
        new = (A_inv @ (rh + dt * Fh + dtL @ rh))[..., 0]
        new = np.moveaxis(new, -1, 0)
        dc = new[:, 0, 0].copy()
        new = rfft(irfft(new * N)) / N
        new[:, 0, 0] = dc.real
        rho_hat = new
        out.append(rho_hat)
    return np.stack(out)


@pytest.mark.parametrize("steps", [1, 6])
def test_a_float64_step_is_the_independent_float64_numpy_update_to_1e_12(twins, steps):
    m2, _ = twins
    rho0 = _rho2()
    traj = rollout_imex(m2, _hat2(rho0), BOX2, T, 0.05, steps, m_stab="mean", **DET)
    assert traj.dtype == torch.complex128
    want = _numpy_imex(m2, rho0[0].numpy(), BOX2.numpy(), 0.05, steps)
    got = traj.numpy()
    scale = np.abs(want).max()
    assert np.abs(got[-1] - got[0]).max() > 1e-3 * scale          # it moved
    assert np.abs(got - want).max() <= 1e-12 * scale


def test_the_float32_scheme_is_only_float32_close_to_the_same_reference(twins):
    """The reference is tight enough to tell the precisions apart: the float32 run of the same model
    misses it by many orders more than the float64 one."""
    m2, _ = twins
    m32 = NdimToyModel(GRID2, 2, nyquist_mask=True, kappa=0.5, kB=1.0)
    m32.load_state_dict({**m32.state_dict(), **{k: v.float() for k, v in m2.state_dict().items()
                                                if not k.startswith("ops.")}})
    m32.eval()
    ref64 = NdimToyModel(GRID2, 2, nyquist_mask=True, kappa=0.5, kB=1.0)
    ref64.load_state_dict(m32.state_dict())
    ref64.double().eval()
    rho0 = _rho2()
    want = _numpy_imex(ref64, rho0[0].numpy(), BOX2.numpy(), 0.05, 6)
    scale = np.abs(want).max()
    got32 = rollout_imex(m32, _hat2(rho0).to(torch.complex64), BOX2.float(), T, 0.05, 6,
                         m_stab="mean", **DET)
    got64 = rollout_imex(ref64, _hat2(rho0), BOX2, T, 0.05, 6, m_stab="mean", **DET)
    err32 = np.abs(got32.numpy() - want).max() / scale
    err64 = np.abs(got64.numpy() - want).max() / scale
    assert got32.dtype == torch.complex64
    assert err64 <= 1e-12 and err32 > 1e-8


# ---------------------------------------------------------------------------
# the z-invariant 3D twin, in float64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m_stab", ["mean", "max"])
def test_a_float64_2d_rollout_is_its_z_invariant_float64_3d_twin(twins, m_stab):
    m2, m3 = twins
    assert m2.ops.nyquist_mask and m3.ops.nyquist_mask
    rho2 = _rho2()
    rho3 = rho2.unsqueeze(-1).expand(*rho2.shape, GZ).contiguous()
    h2 = _hat2(rho2)
    h3 = torch.fft.rfftn(rho3, dim=(-3, -2, -1)) / math.prod(GRID3)
    g = torch.Generator().manual_seed(3)
    kbt2 = 1.0 + 0.1 * torch.rand(*GRID2, generator=g, dtype=torch.float64)
    kbt3 = kbt2.unsqueeze(-1).expand(*GRID3).contiguous()
    kw = dict(DET, mass_restore="headroom")
    t2 = rollout_imex(m2, h2, BOX2, T, 2e-2, 10, m_stab=m_stab, kbt_field=kbt2, **kw)
    t3 = rollout_imex(m3, h3, BOX3, T, 2e-2, 10, m_stab=m_stab, kbt_field=kbt3, **kw)
    assert t2.dtype == t3.dtype == torch.complex128
    r2, r3 = _real(m2.ops, t2), _real(m3.ops, t3)
    assert float((r2 - r2[0]).abs().max()) > 1e-4
    assert torch.allclose(r3, r3[..., :1].expand_as(r3), rtol=0, atol=1e-13)
    assert torch.allclose(r3.mean(-1), r2, rtol=0, atol=1e-13)
    assert torch.allclose(r3[..., 0], r2, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# the dtype audit
# ---------------------------------------------------------------------------

class _Audit(TorchFunctionMode):
    """Every tensor any torch function returns (factories, methods, operators); and the complex operands
    of every matrix product (the scheme's ``A_inv @``, ``MH @`` and the predictor's pair)."""

    def __init__(self):
        super().__init__()
        self.single = []
        self.matmul = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = getattr(func, "__name__", repr(func))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor) and t.dtype in SINGLE:
                self.single.append((name, t.dtype, tuple(t.shape)))
        if name in ("__matmul__", "matmul"):
            self.matmul.update(a.dtype for a in args
                               if isinstance(a, torch.Tensor) and a.is_complex())
        return out


def _spy(monkeypatch, name, seen):
    real = getattr(imex_mod, name)

    def wrapped(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.setdefault(name, []).append((args, kwargs, out))
        return out

    monkeypatch.setattr(imex_mod, name, wrapped)


@pytest.mark.parametrize("noise_eval, m_stab, mass_restore, state_proj", [
    ("midpoint", "max", "headroom", "floor"),
    ("kinetic", "mean", "shift", "domain"),
    ("ito", "max", "shift", "floor"),
])
def test_no_single_precision_tensor_is_made_anywhere_on_the_float64_path(
        twins, monkeypatch, noise_eval, m_stab, mass_restore, state_proj):
    m2, _ = twins
    seen = {}
    for name in ("_inverse", "local_hessian", "draw_w", "zeta_from_w", "noise_div_hat",
                 "build_noise_filter", "max_norm_mobility", "hermitianize", "_project",
                 "make_ops"):
        _spy(monkeypatch, name, seen)
    proj = (dict(state_proj="floor", state_clamp=0.3) if state_proj == "floor" else
            dict(state_proj="domain", state_clamp=1e-3,
                 domain=TrustDomain(inner=(0.3, 0.2), outer=(1.5, 1.5))))
    kbt = 1.0 + 0.05 * torch.rand(*GRID2, generator=torch.Generator().manual_seed(4),
                                  dtype=torch.float64)
    v_ext = 0.01 * torch.randn(2, *GRID2, generator=torch.Generator().manual_seed(5),
                               dtype=torch.float64)
    audit = _Audit()
    with audit:
        traj = rollout_imex(m2, _hat2(_rho2()), BOX2, T, 1e-2, 3, kB=KB, m_stab=m_stab,
                            clamp_rho=1e-3, mass_restore=mass_restore,
                            noise={**NOISE, "noise_eval": noise_eval},
                            generator=torch.Generator().manual_seed(2), depth=3.0,
                            kbt_field=kbt, v_ext=v_ext, **proj)
    assert audit.single == []
    assert audit.matmul == {torch.complex128}                  # A_inv, MH, the predictor's pair
    assert traj.dtype == torch.complex128
    (ops,) = [out for _, _, out in seen["make_ops"]]
    assert {b.dtype for b in ops.buffers()} == {torch.float64}
    for name in ("NX", "NY"):
        index = getattr(ops, name)
        assert torch.equal(index, torch.round(index))
    assert torch.equal(ops.NX.flatten(), _mode_numbers(GRID2[0]))
    for args, _, out in seen["_inverse"]:
        assert args[0].dtype == out.dtype == torch.float64     # I + dt k^2 M_s H, before the cast
    (hb,) = [out for _, _, out in seen["local_hessian"]]
    assert hb.dtype == torch.float64
    for (_, _, _, _, dtype), _, w in seen["draw_w"]:
        assert dtype == w.dtype == torch.float64
    for (ops_, zeta, kx, ky, kz, filt), _, out in seen["noise_div_hat"]:
        assert zeta.dtype == kx.dtype == ky.dtype == filt.dtype == torch.float64 and kz is None
        assert out.dtype == torch.complex128
    for _, _, out in seen["build_noise_filter"]:
        assert out.dtype == torch.float64
    for _, _, out in seen.get("max_norm_mobility", []):
        assert out.dtype == torch.float64
    for name in ("hermitianize", "_project"):
        assert {out.dtype for _, _, out in seen[name]} == {torch.complex128}


def test_the_stabiliser_hook_and_the_fields_are_handed_float64(twins):
    m2, _ = twins
    got = []

    class Hooked(NdimToyModel):
        def stabilizer_mobility(self, rho, T_t):
            got.append((rho.dtype, T_t.dtype))
            return self.mobility(rho, T_t)[0, :, :, 0, 0]

    hooked = Hooked(GRID2, 2, nyquist_mask=True, kappa=0.5, kB=1.0).double()
    hooked.load_state_dict(m2.state_dict())
    kbt = torch.full(GRID2, 1.0, dtype=torch.float32)          # a float32 field is cast, not kept
    seen = {}
    with imex_mod.pointwise_fields(hooked, _hat2(_rho2()), kbt, None, torch.float64) as kbt_t:
        seen["kbt"] = kbt_t.dtype
    assert seen["kbt"] == torch.float64
    box32, h0 = BOX2.float(), _hat2(_rho2())                   # a float32 box is cast too
    audit = _Audit()
    with audit:
        rollout_imex(hooked, h0, box32, T, 1e-2, 2, kbt_field=kbt, **DET)
    assert got == [(torch.float64, torch.float64)]
    assert audit.single == []


def test_float64_mode_indices_are_exact_integers_and_float32_ones_are_unchanged():
    G = 100
    ops32 = SpectralOps((G, 6, 4), 1, nyquist_mask=True)
    ops64 = SpectralOps((G, 6, 4), 1, nyquist_mask=True, dtype=torch.float64)
    assert torch.equal(ops32.NX.flatten(), torch.fft.fftfreq(G) * G)       # bits it always had
    assert not torch.equal(ops32.NX, torch.round(ops32.NX))                # ...which are not integers
    exact = _mode_numbers(G)
    assert torch.equal(ops64.NX.flatten(), exact)
    cast = copy.deepcopy(ops32).double()                                   # a cast is a plain cast
    assert torch.equal(cast.NX, ops32.NX.double())
    assert {b.dtype for b in ops64.buffers()} == {torch.float64}
    sd32, sd64 = ops32.state_dict(), ops64.state_dict()
    assert list(sd32) == list(sd64) == ["NX", "NY", "NZ", "MULT"]
    assert all(sd32[k].shape == sd64[k].shape for k in sd32)
    assert all(v.dtype == torch.float32 for v in sd32.values())
    for k, v in sd64.items():
        assert torch.equal(v, sd32[k].double().round() if k != "MULT" else sd32[k].double())
    o2 = SpectralOps2D((G, 7), 1, nyquist_mask=False, dtype=torch.float64)
    assert torch.equal(o2.NX.flatten(), exact) and o2.NY.dtype == torch.float64
    with pytest.raises(ValueError, match="float32 \\(dtype=None"):
        make_ops((4, 4), 1, nyquist_mask=True, dtype=torch.float16)


def test_a_float64_rollout_reads_exact_indices_and_hands_the_model_back_unchanged():
    """``model.double()`` keeps the float32 ``fftfreq(G) * G`` error in ``ops.NX``; the rollout reads the
    rounded modes on every grid of the model's cache, then every buffer and the cache are put back."""
    G = (100, 6)
    m = NdimToyModel(G, 2, nyquist_mask=True, kappa=0.5, kB=1.0).double().eval()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    nx_before, table_before = m.ops.NX, dict(m._cache._cache)
    assert not torch.equal(m.ops.NX, m.ops.NX.round())
    seen = []
    forward = m.forward

    def spy(rho_hat, boxes, T_t):
        ops = m._cache.ops_for(rho_hat.shape, rho_hat.device)
        seen.append((ops is m.ops, ops.NX.dtype, torch.equal(ops.NX.flatten(), _mode_numbers(G[0]))))
        return forward(rho_hat, boxes, T_t)

    m.forward = spy
    g = torch.Generator().manual_seed(1)
    rho = 0.5 + 0.02 * torch.randn(1, 2, *G, generator=g, dtype=torch.float64)
    h = torch.fft.rfftn(rho, dim=(-2, -1)) / math.prod(G)
    rollout_imex(m, h, torch.tensor([50.0, 3.0], dtype=torch.float64), T, 1e-3, 2,
                 m_stab="mean", **DET)
    assert seen == [(True, torch.float64, True)] * 2
    del m.forward
    assert m.ops.NX is nx_before and m._cache._cache == table_before
    after = m.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_inside_the_exact_context_a_float64_cache_builds_exact_float64_sets_on_every_grid():
    cache = OpsCache((6, 6, 6), 1, nyquist_mask=True)
    holder = torch.nn.Module()
    holder.ops = cache.ops
    holder._cache = cache
    other32 = cache.ops_for_grid((10, 6, 6))
    holder.double()
    assert cache.ops_for_grid((10, 6, 6)) is other32 and other32.NX.dtype == torch.float32  # as before
    with exact_mode_indices(holder):
        other64 = cache.ops_for_grid((10, 6, 6))
        assert other64.NX.dtype == torch.float64 and other64 is not other32
        assert torch.equal(other64.NX.flatten(), _mode_numbers(10))
        assert {b.dtype for b in other64.buffers()} == {torch.float64}
        assert cache.ops_for_grid((10, 6, 6)) is other64
        assert cache.ops_for_grid((6, 6, 6)) is cache.ops
    assert cache.ops_for_grid((10, 6, 6)) is other32
    with exact_mode_indices(NdimToyModel(GRID2, 2, nyquist_mask=True, kappa=0.5, kB=1.0)) as none:
        assert none is None                                    # a float32 model: nothing to do


# ---------------------------------------------------------------------------
# mixed pairs
# ---------------------------------------------------------------------------

def test_a_float64_state_with_a_float32_model_is_refused_naming_both_and_the_cast():
    m32 = NdimToyModel(GRID2, 2, nyquist_mask=True, kappa=0.5, kB=1.0)
    with pytest.raises(ValueError) as refused:
        rollout_imex(m32, _hat2(_rho2()), BOX2, T, 1e-2, 1, m_stab="mean", **DET)
    text = str(refused.value)
    assert "complex128" in text and "float32" in text and "model.double()" in text
    assert "rho_hat.to(torch.complex64)" in text


def test_a_float32_state_with_a_float64_model_is_refused_naming_both_and_the_cast(twins):
    m2, _ = twins
    with pytest.raises(ValueError) as refused:
        rollout_imex(m2, _hat2(_rho2()).to(torch.complex64), BOX2.float(), T, 1e-2, 1,
                     m_stab="mean", **DET)
    text = str(refused.value)
    assert "complex64" in text and "float64" in text and "rho_hat.to(torch.complex128)" in text


def test_a_model_with_float64_operators_but_float32_weights_is_mixed_too():
    m = NdimToyModel(GRID2, 2, nyquist_mask=True, kappa=0.5, kB=1.0)
    m.ops.double()
    for state in (_hat2(_rho2()), _hat2(_rho2()).to(torch.complex64)):
        with pytest.raises(ValueError, match="float32 and float64"):
            rollout_imex(m, state, BOX2, T, 1e-2, 1, m_stab="mean", **DET)


# ---------------------------------------------------------------------------
# noise in float64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("noise_eval", ["ito", "midpoint", "kinetic"])
def test_a_noisy_float64_run_keeps_mass_exactly_the_state_real_and_complex128(twins, noise_eval):
    m2, _ = twins
    h0 = _hat2(_rho2())
    traj = rollout_imex(m2, h0, BOX2, T, 1e-2, 8, m_stab="max",
                        noise={**NOISE, "noise_eval": noise_eval},
                        generator=torch.Generator().manual_seed(7), depth=3.0,
                        **{**DET, "state_proj": "floor", "state_clamp": 0.3})
    assert traj.dtype == torch.complex128
    dc = traj[:, :, 0, 0]
    assert torch.equal(dc, h0[0, :, 0, 0].real.to(dc.dtype).expand_as(dc))
    N = math.prod(GRID2)
    assert torch.allclose(torch.fft.rfftn(_real(m2.ops, traj), dim=(-2, -1)) / N, traj,
                          rtol=0, atol=1e-14)
    assert float((traj[-1] - traj[0]).abs().max()) > 1e-4


def test_float64_noise_is_drawn_in_float64_a_stream_of_its_own(twins, monkeypatch):
    """Documented: a float64 noisy run draws float64 normals, which are not the float32 run's numbers
    for the same seed."""
    m2, _ = twins
    seen = {}
    _spy(monkeypatch, "draw_w", seen)
    rollout_imex(m2, _hat2(_rho2()), BOX2, T, 1e-2, 1, m_stab="max", noise=dict(NOISE),
                 generator=torch.Generator().manual_seed(9), depth=3.0, **DET)
    ((_, _, w64),) = seen["draw_w"]
    w32 = torch.randn(tuple(w64.shape), generator=torch.Generator().manual_seed(9),
                      dtype=torch.float32)
    assert w64.dtype == torch.float64
    assert not torch.allclose(w64, w32.double(), atol=1e-3)


# ---------------------------------------------------------------------------
# the exact-index context puts everything back
# ---------------------------------------------------------------------------

def _two_float64_sets():
    a = NdimToyModel((100, 6), 2, nyquist_mask=True, kappa=0.5, kB=1.0).double()
    b = NdimToyModel((100, 6, 4), 2, nyquist_mask=True, kappa=0.5, kB=1.0).double()
    holder = torch.nn.Module()
    holder.a, holder.b = a, b
    return holder, a, b


def test_an_exception_partway_through_the_swap_restores_every_buffer_and_table(monkeypatch):
    holder, a, b = _two_float64_sets()
    before = [(m, name, m.ops._buffers[name]) for m in (a, b) for name in ("NX", "NY")]
    tables = [(m._cache, m._cache._cache) for m in (a, b)]
    real_round, calls = torch.round, []

    def flaky(t, *args, **kwargs):
        calls.append(1)
        if len(calls) == 4:                    # after a's NX and NY and b's NX: fails on b's NY
            raise RuntimeError("injected")
        return real_round(t, *args, **kwargs)

    monkeypatch.setattr(torch, "round", flaky)
    with pytest.raises(RuntimeError, match="injected"):
        with exact_mode_indices(holder):
            pass
    monkeypatch.setattr(torch, "round", real_round)
    assert len(calls) == 4
    for m, name, t in before:
        assert m.ops._buffers[name] is t
    for cache, table in tables:
        assert cache._cache is table and cache._exact is False


def test_an_exception_inside_the_call_restores_every_buffer_and_table():
    holder, a, b = _two_float64_sets()
    before = [(m, name, m.ops._buffers[name]) for m in (a, b) for name in ("NX", "NY", "NZ")
              if name in m.ops._buffers]
    with pytest.raises(KeyError):
        with exact_mode_indices(holder):
            assert torch.equal(a.ops.NX, a.ops.NX.round())
            raise KeyError("inside")
    for m, name, t in before:
        assert m.ops._buffers[name] is t
    assert not a._cache._exact and not torch.equal(a.ops.NX, a.ops.NX.round())


def test_a_set_moved_inside_the_context_gets_its_buffers_back_where_it_now_is():
    """``ops_for_grid`` may move the seed (``.to(device)``) inside the call; on exit the original buffers
    are put back on the device the set is on then. On a CPU-only machine the move is the meta device."""
    m = NdimToyModel((100, 6), 2, nyquist_mask=True, kappa=0.5, kB=1.0).double()
    nx = m.ops.NX
    with exact_mode_indices(m):
        m.ops._buffers["NX"] = m.ops._buffers["NX"].to("meta")
    assert m.ops.NX.device.type == "meta" and m.ops.NX.dtype == torch.float64
    assert tuple(m.ops.NX.shape) == tuple(nx.shape)
