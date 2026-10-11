"""``m_stab="bound"``, ``refresh_every`` and ``stabiliser_log`` of the semi-implicit scheme, all opt-in.

``"bound"`` freezes ``M_s = bound_margin * max(r, 1) * M(rho_bar)``, ``r`` the largest generalised eigenvalue of
any cell's mobility against ``M(rho_bar)``; ``refresh_every`` re-freezes ``M_s`` from the current state. Checked
here: ``r`` against a brute-force generalised eigenproblem, a singular ``M(rho_bar)``, a refreshed run against
the same run chained one call per refresh (bit for bit, every source, deterministic and noisy, both
precisions), a stiff composition-dependent mobility where ``"mean"`` blows up at the grid scale and ``"bound"``
does not, the log, the refusals and the float64 dtype audit. The existing paths are pinned by ``tests/golden``.

The models are ``tests/toy_ndim_model.py``'s two-dimensional toy with its constant mobility replaced by a
composition-dependent full ``(2, 2)`` one."""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten

from aipf.rollout import bound_ratio
from aipf.rollout import imex as imex_mod
from aipf.rollout.imex import generalised_ratios, rollout_imex
from aipf.system import Noise
from toy_ndim_model import NdimToyModel

GRID = (16, 12)
BOX = torch.tensor([16.0, 12.0], dtype=torch.float64)
T = 1.0
DET = dict(kB=1.0, state_proj="floor", state_clamp=0.0, clamp_rho=None, mass_restore="shift")
NOISE = dict(kBT_noise=0.02, noise_scale=1.0, noise_mode="gaussian", sigma_noise=1.0,
             noise_eval="midpoint", predictor_floor=1e-3)
COUPLING = ((1.0, 0.3), (0.3, 1.0))


@pytest.fixture(autouse=True)
def _one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class CompositionMobilityToy(NdimToyModel):
    """The toy with a composition-dependent mobility, a linear local part ``f = a rho^2 / 2`` and
    ``W = kappa k^2``.

    ``kind="scaled"``: ``M = D^1/2 B D^1/2``, ``D = diag(d0 + d1 rho^2)``, ``B`` a fixed full coupling.
    ``kind="peaked"``: ``M = exp(-(rho_0 - c0)^2 / w) B``, largest where ``rho_0 = c0``.
    ``kind="singular"``: ``M = v v^T + (rho_0 - c0)^2 I``, singular where ``rho_0 = c0``."""

    def __init__(self, kind="scaled", *, c0=0.0, a=0.2, kappa=0.1, grid=GRID):
        super().__init__(grid, 2, nyquist_mask=True, kappa=kappa, kB=1.0, linear_a=a)
        self.kind = kind
        self.register_buffer("d0", torch.tensor([0.05, 0.05]))
        self.register_buffer("d1", torch.tensor([4.0, 1.0]))
        self.register_buffer("coupling", torch.tensor(COUPLING))
        self.register_buffer("c0", torch.tensor(float(c0)))
        self.register_buffer("v", torch.tensor([0.6, 0.8]))

    def mobility(self, rho, T):
        nd = self.ndim
        cell = (1,) * nd
        B = self.coupling.to(rho.dtype).view(1, 2, 2, *cell)
        if self.kind == "scaled":
            d = self.d0.view(1, -1, *cell) + self.d1.view(1, -1, *cell) * rho ** 2
            s = d.sqrt()
            return s[:, :, None] * B * s[:, None, :]
        dev = (rho[:, 0] - self.c0) ** 2                                   # (B, *grid)
        if self.kind == "peaked":
            return torch.exp(-dev / 0.05)[:, None, None] * B
        vv = (self.v[:, None] * self.v[None, :]).to(rho.dtype).view(1, 2, 2, *cell)
        eye = torch.eye(2, dtype=rho.dtype).view(1, 2, 2, *cell)
        return vv + dev[:, None, None] * eye


class HookedToy(CompositionMobilityToy):
    """A ``stabilizer_mobility`` that reads the field: 1.5 times its cells' mean mobility."""

    def stabilizer_mobility(self, rho, T_t):
        return 1.5 * self.mobility(rho, T_t).mean(dim=(-2, -1))[0]


def _model(cls=CompositionMobilityToy, fp64=False, **kw):
    m = cls(**kw).eval()
    return m.double() if fp64 else m


def _field(seed=0, blob=0.6, noise=1e-2, grid=GRID):
    """Species 0 a Gaussian blob (``blob`` high) on 0.3, species 1 flat at 0.5, plus a grid-scale
    perturbation."""
    g = torch.Generator().manual_seed(seed)
    X, Y = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in grid), indexing="ij")
    r2 = (X - grid[0] / 2) ** 2 + (Y - grid[1] / 2) ** 2
    rho = torch.stack([0.3 + blob * torch.exp(-r2 / (2 * 3.0 ** 2)), 0.5 + 0 * X])
    return rho + noise * torch.randn(2, *grid, generator=g, dtype=torch.float64)


def _hat(rho, fp64=False):
    h = torch.fft.rfftn(rho.unsqueeze(0), dim=(-2, -1)) / math.prod(rho.shape[-2:])
    return h if fp64 else h.to(torch.complex64)


def _box(fp64):
    return BOX if fp64 else BOX.float()


def _real(model, traj):
    return model.ops.irfft(traj.to(torch.complex128) * math.prod(model.ops.grid))


def _spy_operators(monkeypatch):
    """Every ``M_s`` the scheme builds its operators from, in order."""
    seen = []
    real = imex_mod._operators

    def spy(M_s, *args):
        seen.append(M_s.clone())
        return real(M_s, *args)

    monkeypatch.setattr(imex_mod, "_operators", spy)
    return seen


# ---------------------------------------------------------------------------
# r
# ---------------------------------------------------------------------------

def _brute(M_bar, cells):
    """``scipy.linalg.eigh(M_cell, M_bar)``'s largest eigenvalue per cell, both symmetrised first as the scheme
    does (a float32 ``D^1/2 B D^1/2`` is symmetric only to rounding, and ``eigh`` reads one triangle)."""
    Mb = np.asarray(M_bar, dtype=np.float64)
    Mb = 0.5 * (Mb + Mb.T)
    return np.array([scipy.linalg.eigh(0.5 * (c + c.T), Mb, eigvals_only=True).max()
                     for c in np.asarray(cells, dtype=np.float64)])


def test_the_ratio_of_random_matrices_is_the_brute_force_generalised_eigenvalue_to_1e_10():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(64, 3, 3, generator=g, dtype=torch.float64)
    cells = A @ A.transpose(-1, -2) + 0.05 * torch.eye(3, dtype=torch.float64)
    B = torch.randn(3, 3, generator=g, dtype=torch.float64)
    M_bar = B @ B.T + 0.1 * torch.eye(3, dtype=torch.float64)
    got = generalised_ratios(M_bar, cells).numpy()
    want = _brute(M_bar.numpy(), cells.numpy())
    assert np.abs(got - want).max() <= 1e-10 * np.abs(want).max()
    got32 = generalised_ratios(M_bar.float(), cells.float())                # inputs float32, ratio float64
    assert got32.dtype == torch.float64
    want32 = _brute(M_bar.float().double().numpy(), cells.float().double().numpy())
    assert np.abs(got32.numpy() - want32).max() <= 1e-10 * np.abs(want32).max()


@pytest.mark.parametrize("fp64", [False, True])
def test_r_of_a_field_is_the_brute_force_maximum_over_its_cells_to_1e_10(fp64):
    m = _model(fp64=fp64)
    h = _hat(_field(), fp64)
    got = bound_ratio(m, h, T)
    rho = m.ops.irfft(h * math.prod(GRID))
    M_bar = m.mobility(h[0, :, 0, 0].real.view(1, 2, 1, 1), torch.tensor([T]))[0, :, :, 0, 0]
    cells = m.mobility(rho, torch.tensor([T]))[0].permute(2, 3, 0, 1).reshape(-1, 2, 2)
    want = _brute(M_bar.double().numpy(), cells.double().numpy())
    assert np.abs(got["ratios"].flatten().numpy() - want).max() <= 1e-10 * want.max()
    assert abs(got["r_raw"] - want.max()) <= 1e-10 * want.max()
    assert got["r_raw"] > 2.0                                     # the blob: past "mean"'s stiff limit
    assert got["r_used"] == got["r_raw"]
    assert np.unravel_index(int(want.argmax()), GRID) == got["cell"]
    assert torch.equal(got["M_bar"], M_bar) and got["M_bar"].dtype == M_bar.dtype
    assert got["floored"] is False and 1.0 < got["cond"] < 10.0


def test_a_uniform_field_gives_r_one_and_m_s_equal_to_the_mean_one():
    m = _model(fp64=True)
    rho = torch.stack([torch.full(GRID, 0.4), torch.full(GRID, 0.5)]).double()
    got = bound_ratio(m, _hat(rho, True), T)
    assert abs(got["r_raw"] - 1.0) <= 1e-12 and got["r_used"] == max(got["r_raw"], 1.0)
    assert torch.equal(got["M_s"], (got["r_used"] * got["M_bar"]))


@pytest.mark.parametrize("fp64", [False, True])
def test_a_field_whose_cells_are_all_below_m_bar_makes_bound_the_mean_run_bit_for_bit(fp64):
    """``r < 1`` is clamped to 1, so ``"bound"`` is never less damped than ``"mean"``; here it is ``"mean"``."""
    rho = _field(noise=0.0, blob=0.3)
    h = _hat(rho, fp64)
    m = _model(kind="peaked", c0=float(h[0, 0, 0, 0].real), fp64=fp64)
    got = bound_ratio(m, h, T)
    assert got["r_raw"] < 1.0 and got["r_used"] == 1.0
    assert torch.equal(got["M_s"], got["M_bar"])
    log = []
    mean = rollout_imex(m, h, _box(fp64), T, 0.5, 6, m_stab="mean", **DET)
    bound = rollout_imex(m, h, _box(fp64), T, 0.5, 6, m_stab="bound", refresh_every=2,
                         stabiliser_log=log, **DET)
    assert torch.equal(mean, bound)
    assert [e["r_used"] for e in log] == [1.0] * 3 and all(e["r_raw"] < 1.0 for e in log)


def test_a_singular_m_bar_gives_a_finite_r_and_an_m_s_that_bounds_every_cell_in_its_null_direction(
        monkeypatch):
    """``M(rho_bar) = v v^T`` is singular; the cells have mobility along its null direction ``u``. The floored
    ``M(rho_bar)`` builds ``M_s``, so ``M(rho(x)) <= M_s / margin`` holds along ``u`` too, which no multiple of
    the singular ``M(rho_bar)`` could."""
    rho = _field(noise=1e-2, blob=0.2)
    h = _hat(rho, True)
    m = _model(kind="singular", c0=float(h[0, 0, 0, 0].real), fp64=True)
    got = bound_ratio(m, h, T)
    assert math.isfinite(got["r_raw"]) and got["r_raw"] > 1e6
    assert got["floored"] is True and got["cond"] > 1e12
    margin = 1.5
    seen = _spy_operators(monkeypatch)
    log = []
    traj = rollout_imex(m, h, BOX, T, 0.1, 2, m_stab="bound", bound_margin=margin,
                        stabiliser_log=log, **DET)
    assert torch.isfinite(traj).all()
    (M_s,) = seen
    assert log[0]["floored"] is True and log[0]["r_raw"] == got["r_raw"]
    assert torch.allclose(M_s, margin * got["M_s"], rtol=1e-15, atol=0)
    cells = m.mobility(m.ops.irfft(h * math.prod(GRID)), torch.tensor([T], dtype=torch.float64))
    cells = cells[0].permute(2, 3, 0, 1).reshape(-1, 2, 2)
    gap = torch.linalg.eigvalsh(M_s / margin - cells)
    assert float(gap.min()) >= -1e-14 * float(torch.linalg.eigvalsh(M_s)[-1])
    u = torch.tensor([0.8, -0.6], dtype=torch.float64)                   # the null direction of v v^T
    along = u @ (M_s / margin) @ u - torch.einsum("i,cij,j->c", u, cells, u)
    assert float(along.min()) >= -1e-12 * float(cells.max()) and float(cells[:, 0, 0].max()) > 0.0
    plain = got["r_used"] * got["M_bar"]                                  # without the floor: fails along u
    assert float(torch.linalg.eigvalsh(plain - cells).min()) < 0.0


# ---------------------------------------------------------------------------
# refresh_every = N is the run chained one call per refresh
# ---------------------------------------------------------------------------

def _chained(model, h, box, dt, n_steps, every, save_every, generator=None, log=None, **kw):
    """``ceil(n_steps / every)`` calls of ``every`` steps, each from the last state of the previous one; the
    states a single call with ``save_every`` would save."""
    keep = {s for s in range(1, n_steps + 1) if s % save_every == 0 or s == n_steps}
    out, state, step = [h[0].cpu().clone()], h, 0
    while step < n_steps:
        c = min(every, n_steps - step)
        part = [] if log is not None else None
        traj = rollout_imex(model, state, box, T, dt, c, save_every=1, generator=generator,
                            stabiliser_log=part, **kw)
        if log is not None:
            log += [{**e, "step": e["step"] + step} for e in part]
        out += [traj[i] for i in range(1, c + 1) if step + i in keep]
        state = traj[-1:].to(h.device)
        step += c
    return torch.stack(out)


SOURCES = [("mean", CompositionMobilityToy), ("max", CompositionMobilityToy),
           ("bound", CompositionMobilityToy), ("hook", HookedToy)]


@pytest.mark.parametrize("fp64", [False, True], ids=["fp32", "fp64"])
@pytest.mark.parametrize("noisy", [False, True], ids=["deterministic", "noisy"])
@pytest.mark.parametrize("source, cls", SOURCES, ids=[s for s, _ in SOURCES])
def test_a_refreshed_run_is_the_run_chained_one_call_per_refresh_bit_for_bit(source, cls, noisy, fp64):
    m = _model(cls, fp64=fp64)
    h = _hat(_field(), fp64)
    kw = dict(DET)
    if source != "hook":
        kw["m_stab"] = source
    if noisy:
        kw.update(noise=dict(NOISE), depth=1.0)
    n_steps, every, save_every = 11, 4, 3
    log_one, log_chain = [], []
    one = rollout_imex(m, h, _box(fp64), T, 0.5, n_steps, save_every=save_every,
                       refresh_every=every, stabiliser_log=log_one,
                       generator=torch.Generator().manual_seed(5), **kw)
    chain = _chained(m, h, _box(fp64), 0.5, n_steps, every, save_every,
                     generator=torch.Generator().manual_seed(5), log=log_chain, **kw)
    assert one.dtype == (torch.complex128 if fp64 else torch.complex64)
    assert one.shape[0] == 1 + len([s for s in range(1, 12) if s % 3 == 0 or s == 11])
    assert torch.equal(one, chain)
    assert log_one == log_chain and [e["step"] for e in log_one] == [0, 4, 8]
    assert {e["source"] for e in log_one} == {source}
    frozen = rollout_imex(m, h, _box(fp64), T, 0.5, n_steps, save_every=save_every,
                          generator=torch.Generator().manual_seed(5), **kw)
    # "mean" reads only rho_bar, which the scheme conserves: its refresh rebuilds the same operators
    assert torch.equal(one, frozen) == (source == "mean")


def test_a_refreshed_run_with_projection_fields_and_headroom_is_the_chained_run_bit_for_bit():
    """The per-call options a chained call rebuilds (``kbt_field``, ``v_ext``, the mobility floor, a projection
    that acts) are rebuilt identically; the projection keeps ``k = 0``, so ``rho_bar`` is the same."""
    m = _model(fp64=True)
    h = _hat(_field(), True)
    g = torch.Generator().manual_seed(4)
    kw = dict(kB=1.0, m_stab="bound", bound_margin=1.25, clamp_rho=1e-3, mass_restore="headroom",
              state_proj="floor", state_clamp=0.33, noise={**NOISE, "noise_eval": "kinetic"}, depth=2.0,
              kbt_field=1.0 + 0.05 * torch.rand(*GRID, generator=g, dtype=torch.float64),
              v_ext=0.01 * torch.randn(2, *GRID, generator=g, dtype=torch.float64))
    one = rollout_imex(m, h, BOX, T, 0.5, 7, refresh_every=2,
                       generator=torch.Generator().manual_seed(1), **kw)
    chain = _chained(m, h, BOX, 0.5, 7, 2, 1, generator=torch.Generator().manual_seed(1), **kw)
    assert torch.equal(one, chain)


@pytest.mark.parametrize("m_stab, cls", [("mean", CompositionMobilityToy), ("max", CompositionMobilityToy),
                                         (None, HookedToy)])
def test_without_refresh_or_log_the_existing_sources_freeze_once_and_a_log_changes_nothing(
        monkeypatch, m_stab, cls):
    m = _model(cls)
    h = _hat(_field())
    kw = dict(DET, noise=dict(NOISE), depth=1.0)
    if m_stab is not None:
        kw["m_stab"] = m_stab
    seen = _spy_operators(monkeypatch)
    calls = []
    hessian = imex_mod.local_hessian
    monkeypatch.setattr(imex_mod, "local_hessian", lambda *a: calls.append(1) or hessian(*a))
    plain = rollout_imex(m, h, BOX.float(), T, 0.5, 6, generator=torch.Generator().manual_seed(2), **kw)
    assert len(seen) == 1 and len(calls) == 1
    log = []
    logged = rollout_imex(m, h, BOX.float(), T, 0.5, 6, generator=torch.Generator().manual_seed(2),
                          stabiliser_log=log, **kw)
    never = rollout_imex(m, h, BOX.float(), T, 0.5, 6, generator=torch.Generator().manual_seed(2),
                         refresh_every=6, **kw)
    assert torch.equal(plain, logged) and torch.equal(plain, never)
    assert log == [{"step": 0, "source": m_stab or "hook"}]


# ---------------------------------------------------------------------------
# the stiff limit
# ---------------------------------------------------------------------------

def _stiff_case():
    grid = (32, 32)
    m = _model(grid=grid)
    rho = _field(seed=0, blob=0.6, noise=1e-3, grid=grid)
    return m, _hat(rho), torch.tensor([32.0, 32.0])


def test_the_stiff_case_has_r_above_two_at_the_blob():
    m, h, _ = _stiff_case()
    got = bound_ratio(m, h, T)
    assert got["r_raw"] > 4.0 and got["cell"] == (16, 16)


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
def test_mean_blows_up_at_the_grid_scale_where_the_mobility_ratio_passes_two(dt):
    m, h, box = _stiff_case()
    f = _real(m, rollout_imex(m, h, box, T, dt, 60, m_stab="mean", **DET))
    assert not bool(torch.isfinite(f[-1]).all()) or float(f[-1].abs().max()) > 1e3


@pytest.mark.parametrize("refresh_every", [None, 1, 5])
@pytest.mark.parametrize("dt", [0.05, 0.5, 1.0, 2.0, 5.0])
def test_bound_stays_bounded_in_the_same_case_at_every_dt(dt, refresh_every):
    m, h, box = _stiff_case()
    log = []
    f = _real(m, rollout_imex(m, h, box, T, dt, 60, m_stab="bound", refresh_every=refresh_every,
                              stabiliser_log=log, **DET))
    assert bool(torch.isfinite(f).all())
    assert float(f.abs().max()) <= float(f[0].abs().max()) + 1e-6
    hf = torch.fft.rfftn(f[-1], dim=(-2, -1)).abs()
    assert float(hf[:, 12:20, 12:].max()) < 1e-2 * float(hf[:, 0, 0].min())   # no grid-scale growth
    assert all(e["ratio_after"] <= 1.0 + 1e-6 for e in log)


# ---------------------------------------------------------------------------
# the log
# ---------------------------------------------------------------------------

def test_the_bound_log_records_r_the_cell_the_ratios_and_the_ratio_after():
    m = _model(fp64=True)
    h = _hat(_field(), True)
    log = []
    rollout_imex(m, h, BOX, T, 0.5, 10, m_stab="bound", bound_margin=1.25, refresh_every=3,
                 stabiliser_log=log, **DET)
    assert [e["step"] for e in log] == [0, 3, 6, 9]
    keys = {"step", "source", "r_raw", "r_used", "margin", "cell", "ratio_min", "ratio_mean",
            "ratio_max", "cond", "floored", "ratio_after"}
    assert all(set(e) == keys and e["source"] == "bound" and e["margin"] == 1.25 for e in log)
    first, ref = log[0], bound_ratio(m, h, T)
    assert first["r_raw"] == ref["r_raw"] == first["ratio_max"] and first["cell"] == ref["cell"]
    assert first["ratio_min"] == float(ref["ratios"].min())
    assert first["ratio_mean"] == float(ref["ratios"].mean())
    assert first["cond"] == ref["cond"] and first["floored"] is False
    for e in log:
        assert e["ratio_min"] <= e["ratio_mean"] <= e["ratio_max"] == e["r_raw"]
        assert e["r_used"] == max(e["r_raw"], 1.0)
        assert abs(e["ratio_after"] - e["r_raw"] / (1.25 * e["r_used"])) <= 1e-12
    assert len({e["r_raw"] for e in log}) == 4                            # the field moved, r with it


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m_stab", ["mean", "max"])
def test_a_bound_margin_beside_another_m_stab_is_refused(m_stab):
    with pytest.raises(ValueError, match="bound_margin=1.5 is a setting of m_stab='bound' only"):
        rollout_imex(_model(), _hat(_field()), BOX.float(), T, 0.1, 1, m_stab=m_stab,
                     bound_margin=1.5, **DET)


@pytest.mark.parametrize("margin", [0.99, 0.0, -1.0, float("nan"), float("inf"), True, "2"])
def test_a_bound_margin_below_one_or_not_finite_is_refused(margin):
    with pytest.raises(ValueError, match="bound_margin must be a finite number >= 1"):
        rollout_imex(_model(), _hat(_field()), BOX.float(), T, 0.1, 1, m_stab="bound",
                     bound_margin=margin, **DET)


@pytest.mark.parametrize("every", [0, -2, 1.5, 2.0, True, "3"])
def test_a_refresh_every_that_is_not_a_positive_int_is_refused(every):
    with pytest.raises(ValueError, match="refresh_every must be None"):
        rollout_imex(_model(), _hat(_field()), BOX.float(), T, 0.1, 1, m_stab="mean",
                     refresh_every=every, **DET)


def test_bound_beside_a_stabiliser_hook_is_refused_and_so_is_a_bound_margin():
    with pytest.raises(ValueError, match="stabilizer_mobility, which sets M_s itself"):
        rollout_imex(_model(HookedToy), _hat(_field()), BOX.float(), T, 0.1, 1, m_stab="bound", **DET)
    with pytest.raises(ValueError, match="HookedToy.stabilizer_mobility sets M_s"):
        rollout_imex(_model(HookedToy), _hat(_field()), BOX.float(), T, 0.1, 1, bound_margin=1.0, **DET)


def test_a_log_that_is_not_a_list_is_refused():
    with pytest.raises(TypeError, match="stabiliser_log must be None or a list"):
        rollout_imex(_model(), _hat(_field()), BOX.float(), T, 0.1, 1, m_stab="mean",
                     stabiliser_log={}, **DET)


def test_a_non_finite_state_or_mobility_is_refused_naming_which():
    rho = _field()
    rho[0, 3, 4] = float("nan")
    with pytest.raises(ValueError, match="M\\(rho_bar\\).*is non-finite"):           # the mean is NaN too
        rollout_imex(_model(), _hat(rho), BOX.float(), T, 0.1, 1, m_stab="bound", **DET)
    m = _model()
    mobility = m.mobility

    def holed(r, T_t):
        M = mobility(r, T_t).clone()
        if M.shape[-1] > 1:
            M[0, 0, 0, 3, 4] = float("inf")
        return M

    m.mobility = holed
    with pytest.raises(ValueError, match="non-finite at 1 of 192 cells"):
        rollout_imex(m, _hat(_field()), BOX.float(), T, 0.1, 1, m_stab="bound", **DET)
    with pytest.raises(ValueError, match="non-finite at 1 of 192 cells"):
        bound_ratio(m, _hat(_field()), T)
    m.mobility = lambda r, T_t: 0.0 * mobility(r, T_t)
    with pytest.raises(ValueError, match="no positive eigenvalue"):
        rollout_imex(m, _hat(_field()), BOX.float(), T, 0.1, 1, m_stab="bound", **DET)


def test_bound_is_a_rollout_option_and_not_a_noise_declaration():
    with pytest.raises(ValueError, match="Noise.m_stab='bound' is not one of"):
        Noise(mode="none", m_stab="bound")


# ---------------------------------------------------------------------------
# precision
# ---------------------------------------------------------------------------

#: The dtypes a float64 run must never produce.
SINGLE = (torch.float32, torch.complex64, torch.float16, torch.bfloat16)


class _Audit(TorchFunctionMode):
    """Every tensor any torch function returns, and the complex operands of every matrix product."""

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


def test_no_single_precision_tensor_is_made_by_a_refreshed_float64_bound_run(monkeypatch):
    m = _model(fp64=True)
    seen = _spy_operators(monkeypatch)
    g = torch.Generator().manual_seed(4)
    kbt = 1.0 + 0.05 * torch.rand(*GRID, generator=g, dtype=torch.float64)
    audit = _Audit()
    with audit:
        traj = rollout_imex(m, _hat(_field(), True), BOX, T, 0.5, 5, kB=1.0, m_stab="bound",
                            bound_margin=1.25, refresh_every=2, stabiliser_log=[], clamp_rho=1e-3,
                            mass_restore="headroom", state_proj="floor", state_clamp=0.3,
                            noise=dict(NOISE), depth=2.0, generator=torch.Generator().manual_seed(2),
                            kbt_field=kbt)
    assert audit.single == []
    assert audit.matmul == {torch.complex128}
    assert traj.dtype == torch.complex128
    assert len(seen) == 3 and {M.dtype for M in seen} == {torch.float64}


def test_a_float32_bound_run_freezes_a_float32_m_s(monkeypatch):
    m = _model()
    seen = _spy_operators(monkeypatch)
    h = _hat(_field())
    traj = rollout_imex(m, h, BOX.float(), T, 0.5, 3, m_stab="bound", refresh_every=1, **DET)
    assert traj.dtype == torch.complex64
    assert len(seen) == 3 and {M.dtype for M in seen} == {torch.float32}
    ref = bound_ratio(m, h, T)
    assert torch.equal(seen[0], ref["M_s"])
    assert torch.equal(seen[0], (ref["r_used"] * ref["M_bar"].double()).float())
