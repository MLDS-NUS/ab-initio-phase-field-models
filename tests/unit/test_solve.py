"""Tests for aipf.solve: integrators, the projection stack, the trust-domain
predicate, and the guards.

``_ToyModelB`` below is the same shape of fixture
``tests/unit/test_model_protocol.py`` and ``tests/unit/test_nonlocal_kernel.py`` use:
not a physical model, but the genuine ``div(M grad mu)`` control flow a real
rung follows, with the simplest possible physics (``mu = rho``, ``M`` a
constant multiple of the identity) so that a Fourier mode's decay rate is
known in closed form -- ``-M0 * k^2`` -- and a wrong sign or a wrong scale
factor anywhere in an integrator is directly checkable, not merely
"finite and the right shape".

Rebuilt here rather than imported from ``aipf.functional.nonlocal_kernel`` or
``aipf.functional.local_forms`` on purpose: a toy fixture that depends only on
``aipf.spectral`` keeps this file's tests meaningful whatever state the
functional forms are in.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from aipf.spectral import SpectralOps
from aipf.solve import (
    UNDECLARED,
    TrustDomain,
    in_domain,
    hermitianize,
    project_state,
    assert_finite,
    clamped_inputs,
    step_euler,
    step_heun,
    step_sde_euler_maruyama,
    rollout_deterministic,
    rollout_sde,
)
from aipf.solve.guards import kappa_roll_correction
from aipf.solve.projection import check_state_projection, restore_mass
from aipf.solve.trust_domain import project_trust_domain

GRID = (8, 6, 8)  # even every axis; Gzr = 5


def _roll_kw(**over):
    """The state projection and input guard a rollout must now DECLARE,
    at the values that reproduce what this file asserted before they
    became declarations.

    ``state_proj="floor"`` with ``floor=0.0`` is the old signature
    default and a genuine no-op, and ``clamp_rho=None`` installs no guard
    at all -- so every existing assertion in this file is about exactly
    the arithmetic it was about before, and the only thing that changed is
    that it is now said out loud. The production values are a different
    pair entirely, which is the point of the change; they are exercised by
    their own tests below and by the parity file.
    """
    kw = {"state_proj": "floor", "floor": 0.0, "clamp_rho": None}
    kw.update(over)
    return kw


class _ToyModelB(nn.Module):
    """``mu = rho`` (convex ``f = rho^2/2``), ``M = M0 * I``: the shared
    ``div(M grad mu)`` skeleton, generic in ``n_species``.
    """

    def __init__(self, grid, n_species, M0: float = 0.7):
        super().__init__()
        self.ops = SpectralOps(grid, n_species, nyquist_mask=True)
        self.n_species = n_species
        self.M0 = M0

    def chemical_potential(self, rho, boxes, T):
        del boxes, T
        return rho

    def bulk_free_energy_density(self, rho, T):
        del T
        return 0.5 * (rho ** 2).sum(dim=1, keepdim=True)

    def mobility(self, rho, T):
        del T
        n = self.n_species
        eye = torch.eye(n, dtype=rho.dtype, device=rho.device)
        eye = eye.view(1, n, n, *([1] * (rho.dim() - 2)))
        return self.M0 * eye.expand(rho.shape[0], n, n, *rho.shape[2:])

    def forward(self, rho_hat, boxes, T):
        ops = self.ops
        N = ops.grid[0] * ops.grid[1] * ops.grid[2]
        rho = ops.irfft(rho_hat * N)
        mu = self.chemical_potential(rho, boxes, T)
        M = self.mobility(rho, T)
        kx, ky, kz = ops.k_axes(boxes)
        mu_hat = ops.rfft(mu) / N
        gx, gy, gz = ops.grad_hat(mu_hat * N, kx, ky, kz)
        grad_mu = torch.stack(
            [ops.irfft(gx), ops.irfft(gy), ops.irfft(gz)], dim=2)
        Jx = torch.einsum("bij...,bj...->bi...", M, grad_mu[:, :, 0])
        Jy = torch.einsum("bij...,bj...->bi...", M, grad_mu[:, :, 1])
        Jz = torch.einsum("bij...,bj...->bi...", M, grad_mu[:, :, 2])
        Jx_hat = ops.rfft(Jx) / N
        Jy_hat = ops.rfft(Jy) / N
        Jz_hat = ops.rfft(Jz) / N
        return ops.div_hat(Jx_hat, Jy_hat, Jz_hat, kx, ky, kz)


def _boxes(batch, grid=GRID):
    L = torch.tensor(grid, dtype=torch.float32)
    return L.unsqueeze(0).expand(batch, 3).clone()


def _T(batch, value=1.5):
    return torch.full((batch,), float(value))


def _rho_hat(batch, n_species, grid, ops, seed=3):
    g = torch.Generator().manual_seed(seed)
    rho = 0.3 + 0.1 * torch.rand(batch, n_species, *grid, generator=g)
    N = grid[0] * grid[1] * grid[2]
    return ops.rfft(rho) / N


# ---------------------------------------------------------------------------
# mass conservation, exact, for every integrator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_step_euler_conserves_mass_exactly(n_species):
    model = _ToyModelB(GRID, n_species)
    boxes = _boxes(3)
    T = _T(3)
    rho_hat = _rho_hat(3, n_species, GRID, model.ops)
    dc0 = rho_hat[:, :, 0, 0, 0].clone()

    out = step_euler(model, rho_hat, boxes, T, dt=1e-3)

    assert torch.equal(out[:, :, 0, 0, 0], dc0)


@pytest.mark.parametrize("n_species", [1, 2])
def test_step_heun_conserves_mass_exactly(n_species):
    model = _ToyModelB(GRID, n_species)
    boxes = _boxes(3)
    T = _T(3)
    rho_hat = _rho_hat(3, n_species, GRID, model.ops)
    dc0 = rho_hat[:, :, 0, 0, 0].clone()

    out = step_heun(model, rho_hat, boxes, T, dt=1e-3)

    assert torch.equal(out[:, :, 0, 0, 0], dc0)


@pytest.mark.parametrize("n_species", [1, 2])
def test_step_sde_conserves_mass_exactly(n_species):
    """The noise contributes nothing to mass either: its spectral
    divergence is multiplied by ``k``, exactly zero at ``k=0`` (and
    additionally zeroed by hand -- see the module docstring)."""
    model = _ToyModelB(GRID, n_species)
    boxes = _boxes(2)
    T = _T(2)
    rho_hat = _rho_hat(2, n_species, GRID, model.ops)
    dc0 = rho_hat[:, :, 0, 0, 0].clone()
    gen = torch.Generator().manual_seed(7)

    out = step_sde_euler_maruyama(model, rho_hat, boxes, T, dt=1e-3,
                                  kBT_noise=0.05, noise_mode="none",
                                  sigma_noise=None, generator=gen)

    assert torch.equal(out[:, :, 0, 0, 0], dc0)


@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("method", ["euler", "heun"])
def test_rollout_deterministic_conserves_mass_exactly_every_saved_frame(
        n_species, method):
    model = _ToyModelB(GRID, n_species)
    boxes = _boxes(1)
    T = _T(1)
    rho_hat0 = _rho_hat(1, n_species, GRID, model.ops)
    dc0 = rho_hat0[:, :, 0, 0, 0].clone()

    traj = rollout_deterministic(model, rho_hat0, boxes, T, dt=1e-3,
                                 n_steps=5, method=method, save_every=1,
                                 **_roll_kw())

    for frame in traj:
        assert torch.equal(frame[:, :, 0, 0, 0], dc0)


@pytest.mark.parametrize("n_species", [1, 2])
def test_rollout_sde_conserves_mass_exactly_every_saved_frame(n_species):
    model = _ToyModelB(GRID, n_species)
    boxes = _boxes(1)
    T = _T(1)
    rho_hat0 = _rho_hat(1, n_species, GRID, model.ops)
    dc0 = rho_hat0[:, :, 0, 0, 0].clone()
    gen = torch.Generator().manual_seed(11)

    traj = rollout_sde(model, rho_hat0, boxes, T, dt=1e-3, n_steps=5,
                       kBT_noise=0.02, noise_mode="none", sigma_noise=None,
                       save_every=1, generator=gen, **_roll_kw())

    for frame in traj:
        assert torch.equal(frame[:, :, 0, 0, 0], dc0)


def test_sde_with_zero_kBT_noise_is_bit_for_bit_the_deterministic_step():
    model = _ToyModelB(GRID, 2)
    boxes = _boxes(2)
    T = _T(2)
    rho_hat = _rho_hat(2, 2, GRID, model.ops)

    a = step_sde_euler_maruyama(model, rho_hat, boxes, T, dt=1e-3,
                                kBT_noise=0.0, noise_mode="none",
                                sigma_noise=None)
    b = step_euler(model, rho_hat, boxes, T, dt=1e-3)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# state projection: exact k=0 mass, no band mask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_project_state_preserves_k0_mass_exactly(n_species):
    ops = SpectralOps(GRID, n_species, nyquist_mask=True)
    g = torch.Generator().manual_seed(4)
    # A bulk comfortably above the floor with a MINORITY of cells driven
    # below it -- the regime the alternation is actually built for (see
    # aipf.solve.projection's module docstring). A mean AT or below the
    # floor is a documented separate limit (the two convex sets become
    # disjoint) and is exercised on its own below, not here.
    rho = 0.5 + 0.02 * torch.randn(2, n_species, *GRID, generator=g)
    floor = 0.05
    rho[:, :, 0, 0, 0] = -0.3  # a few cells well below the floor
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat = ops.rfft(rho) / N
    dc0 = rho_hat[:, :, 0, 0, 0].clone()

    assert not bool((rho >= floor).all()), "fixture must actually violate the floor"

    out = project_state(rho_hat, ops, floor=floor, state_proj="floor")

    assert torch.equal(out[:, :, 0, 0, 0], dc0)
    rho_out = ops.irfft(out * N)
    assert float(rho_out.mean()) == pytest.approx(float(rho.mean()), abs=1e-5)
    # The defining property of the projection itself, not merely its side
    # effect on mass: cells must actually move ABOVE the floor. A clamp
    # applied in the wrong direction (e.g. `clamp(max=floor)`) would still
    # conserve mass via the same k=0 restore and would still change the
    # field, but would push cells the wrong way.
    below_before = float((rho < floor).to(torch.float32).mean())
    below_after = float((rho_out < floor - 1e-4).to(torch.float32).mean())
    assert below_before > 0.0, "fixture must have a violating fraction"
    assert below_after < below_before


def test_project_state_preserves_the_spread_of_already_admissible_cells():
    """The defining, direction-sensitive property `clamp(min=floor)` has
    and `clamp(max=floor)` does not: two cells that both already sit ABOVE
    the floor, at genuinely different values, must stay at (up to a single
    shared uniform mass-restore shift) their ORIGINAL DIFFERENCE from each
    other. `clamp(max=floor)` would instead collapse BOTH of them onto the
    floor -- an admissible cell is `> floor`, hence `> max=floor`, so a
    wrong-direction clamp treats it exactly like a violator -- destroying
    their spread entirely, a change the aggregate "did more cells become
    admissible" checks above cannot see (with enough of the field driven
    to the floor, the compensating mean-restore shift can coincidentally
    land every cell back above the floor regardless of which direction the
    clamp ran).

    The tolerance is RELATIVE, and that is a real change: the mass restore
    is no longer a single shared uniform shift. It spends the correction in
    proportion to each cell's headroom above the floor, so two cells at
    different heights give up different amounts and the gap shrinks by that
    proportion -- here by ~0.2%, which is the restore working, not the
    clamp running backwards. A `clamp(max=floor)` would leave a gap of
    zero, three orders away from either number, which is what the second
    assertion pins.
    """
    ops = SpectralOps(GRID, 1, nyquist_mask=True)
    rho = torch.full((1, 1, *GRID), 0.5)
    floor = 0.05
    rho[0, 0, 0, 0, 0] = 0.3   # admissible, distinct value A
    rho[0, 0, 1, 0, 0] = 0.8   # admissible, distinct value B
    rho[0, 0, 2, 0, 0] = -0.3  # the one violator, forces a real projection
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat = ops.rfft(rho) / N

    out = project_state(rho_hat, ops, floor, state_proj="floor")
    rho_out = ops.irfft(out * N)

    gap_before = float(rho[0, 0, 1, 0, 0] - rho[0, 0, 0, 0, 0])
    gap_after = float(rho_out[0, 0, 1, 0, 0] - rho_out[0, 0, 0, 0, 0])
    assert gap_after == pytest.approx(gap_before, rel=1e-2)
    assert gap_after > 0.5 * gap_before, (
        f"the spread collapsed from {gap_before} to {gap_after}: that is "
        f"the wrong-direction clamp this test exists to catch, not the "
        f"restore's proportional correction")


def test_project_state_is_a_pure_no_op_at_floor_zero():
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    rho_hat = _rho_hat(2, 2, GRID, ops)
    out = project_state(rho_hat, ops, floor=0.0, state_proj="floor")
    assert out is rho_hat


def test_project_state_at_floor_zero_is_a_no_op_even_when_the_field_is_negative():
    """`floor <= 0` means "no projection requested at all", not merely
    "no projection needed": a field that already violates `floor=0.0` (has
    negative cells) must still come back bit-for-bit untouched, never
    round-tripped through rfft/irfft. Without the dedicated early return,
    the loop's own `(rho_r >= floor).all()` early-exit would coincidentally
    also return the identical object for an admissible field, which is why
    `test_project_state_is_a_pure_no_op_at_floor_zero` alone cannot tell
    the two code paths apart -- this fixture is deliberately INADMISSIBLE
    at floor=0.0.
    """
    ops = SpectralOps(GRID, 1, nyquist_mask=True)
    g = torch.Generator().manual_seed(9)
    rho = 0.2 * torch.randn(1, 1, *GRID, generator=g)  # has negative cells
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat = ops.rfft(rho) / N
    assert not bool((rho >= 0.0).all()), "fixture must violate floor=0.0"

    out = project_state(rho_hat, ops, floor=0.0, state_proj="floor")
    assert out is rho_hat


def test_project_state_is_a_no_op_on_an_already_admissible_field():
    ops = SpectralOps(GRID, 1, nyquist_mask=True)
    rho = torch.full((1, 1, *GRID), 0.5)
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat = ops.rfft(rho) / N
    out = project_state(rho_hat, ops, floor=0.1, state_proj="floor")
    assert torch.equal(out, rho_hat)


# ---------------------------------------------------------------------------
# Nyquist repair
# ---------------------------------------------------------------------------

def test_hermitianize_repairs_an_injected_non_hermitian_nyquist_row():
    """A mode that must be REAL for a real-space field (a "self-paired"
    (kx, ky) index at the k_z Nyquist plane, where Gx and Gy are both even)
    is given a large imaginary part by hand. `hermitianize` must remove it.
    """
    ops = SpectralOps(GRID, 1, nyquist_mask=True)
    g = torch.Generator().manual_seed(5)
    rho = torch.rand(1, 1, *GRID, generator=g)
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat = ops.rfft(rho) / N

    Gx, Gy, Gz = GRID
    ix, iy, iz = Gx // 2, 0, ops.Gzr - 1  # self-paired index, Nyquist plane
    corrupted = rho_hat.clone()
    corrupted[:, :, ix, iy, iz] += 5.0j

    assert corrupted[:, :, ix, iy, iz].imag.abs().item() > 1.0, \
        "fixture must actually be non-Hermitian before repair"

    repaired = hermitianize(corrupted, ops)

    assert repaired[:, :, ix, iy, iz].imag.abs().item() < 1e-4, (
        "hermitianize must zero the imaginary part of a mode that has to "
        "be real for a real-space field")
    # Idempotent: a field that is already the rfft of a real field is a
    # fixed point.
    twice = hermitianize(repaired, ops)
    assert torch.allclose(twice, repaired, atol=1e-6)


def test_hermitianize_preserves_k0_exactly():
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    rho_hat = _rho_hat(1, 2, GRID, ops)
    corrupted = rho_hat.clone()
    corrupted[:, :, 3, 0, 0] += 2.0j  # unrelated mode, off-boundary
    dc0_real = rho_hat[:, :, 0, 0, 0].real.clone()

    out = hermitianize(corrupted, ops)
    # hermitianize restores k=0 bitwise from its own real part, so the
    # output's k=0 is real(dc0) exactly, with the imaginary part zeroed --
    # not merely close to dc0, which itself may carry FFT round-off in its
    # imaginary part.
    assert torch.equal(out[:, :, 0, 0, 0].real, dc0_real)
    assert torch.all(out[:, :, 0, 0, 0].imag == 0.0)


def test_hermitianize_zeros_an_injected_imaginary_part_at_k0_itself():
    """The mode `test_hermitianize_preserves_k0_exactly` corrupts is NOT
    k=0 itself, so it cannot catch a hermitianize that forgets to drop the
    IMAGINARY part of k=0 (a real field's mass must be real). This is that
    dedicated check: corrupt k=0's own imaginary part directly.
    """
    ops = SpectralOps(GRID, 1, nyquist_mask=True)
    rho_hat = _rho_hat(1, 1, GRID, ops)
    corrupted = rho_hat.clone()
    corrupted[:, :, 0, 0, 0] += 3.0j
    assert corrupted[:, :, 0, 0, 0].imag.abs().item() > 1.0

    out = hermitianize(corrupted, ops)
    assert torch.all(out[:, :, 0, 0, 0].imag == 0.0)


def test_removing_hermitianize_lets_a_corrupted_nyquist_row_survive_a_step():
    """Structural regression for the carried-forward Hermitian-phantom
    failure: without the per-step repair, injected non-Hermitian content
    at a self-paired mode is NOT cleaned by anything else in the pipeline
    (`irfft` alone discards it on the way OUT of a given step, but nothing
    stops the NEXT step's `rfft` of a real, but now-perturbed, field from
    reintroducing asymmetric content if the corruption is left standing).
    This test exercises `hermitianize` directly (rather than diffing a full
    rollout with and without it, which is the mutation-table's job, see the
    task report) to keep the assertion tight to the one function this
    module is responsible for.
    """
    ops = SpectralOps(GRID, 1, nyquist_mask=True)
    rho = torch.full((1, 1, *GRID), 0.5)
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat = ops.rfft(rho) / N
    Gx, Gy, Gz = GRID
    ix, iy, iz = Gx // 2, Gy // 2, ops.Gzr - 1
    rho_hat[:, :, ix, iy, iz] += 10.0j

    # Un-repaired: the real-space round trip silently drops the corruption
    # ON THE WAY OUT, but the corrupted half-spectrum itself is what a
    # step's arithmetic (grad_hat, mobility contraction, div_hat) actually
    # operates on -- so the corrupted tensor itself must still show the
    # phantom content.
    assert rho_hat[:, :, ix, iy, iz].imag.abs().item() > 5.0

    repaired = hermitianize(rho_hat, ops)
    assert repaired[:, :, ix, iy, iz].imag.abs().item() < 1e-4


# ---------------------------------------------------------------------------
# dispersion: a convex free energy must decay, for every deterministic
# integrator (structural tests do not check physics -- see rung 3's own
# version of this test and the task report for the historical defect)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step_fn", [step_euler, step_heun])
def test_a_convex_free_energy_decays_under_every_integrator(step_fn):
    grid = (8, 8, 8)
    length = 4.0
    model = _ToyModelB(grid, 1, M0=0.7)
    boxes = torch.tensor([[length, length, length]])
    T = torch.tensor([1.0])
    axis = torch.arange(grid[0], dtype=torch.float32) * (length / grid[0])
    rho = 0.5 + 0.01 * torch.cos(2 * torch.pi * axis / length).view(1, 1, -1, 1, 1)
    rho = rho.expand(1, 1, *grid).contiguous()
    N = grid[0] * grid[1] * grid[2]
    rho_hat = model.ops.rfft(rho) / N

    drho_hat_dt = model.forward(rho_hat, boxes, T)
    amplitude = rho_hat[0, 0, 1, 0, 0]
    rate = (drho_hat_dt[0, 0, 1, 0, 0] / amplitude).real
    assert rate < 0

    k_mag = 2 * torch.pi * 1 / length
    expected_rate = -0.7 * k_mag ** 2
    assert rate.item() == pytest.approx(expected_rate, rel=1e-4)

    # And a finite-dt step must shrink the mode's amplitude, for BOTH
    # integrators under test, using a dt small enough that neither scheme's
    # own truncation error flips the sign.
    dt = 1e-4
    out = step_fn(model, rho_hat, boxes, T, dt)
    before = amplitude.abs()
    after = out[0, 0, 1, 0, 0].abs()
    assert after < before


# ---------------------------------------------------------------------------
# kappa_roll: distinct dial, zero is a no-op, nonzero changes the flux
# without breaking k=0 conservation
# ---------------------------------------------------------------------------

def test_kappa_roll_zero_is_a_pure_no_op():
    model = _ToyModelB(GRID, 1)
    boxes = _boxes(1)
    T = _T(1)
    rho_hat = _rho_hat(1, 1, GRID, model.ops)

    with_zero = step_euler(model, rho_hat, boxes, T, dt=1e-3, kappa_roll=0.0)
    plain = step_euler(model, rho_hat, boxes, T, dt=1e-3)
    assert torch.equal(with_zero, plain)


def test_kappa_roll_nonzero_changes_the_flux_but_not_k0():
    model = _ToyModelB(GRID, 1)
    boxes = _boxes(1)
    T = _T(1)
    rho_hat = _rho_hat(1, 1, GRID, model.ops)

    plain = step_euler(model, rho_hat, boxes, T, dt=1e-3)
    stabilised = step_euler(model, rho_hat, boxes, T, dt=1e-3, kappa_roll=2.0)

    assert not torch.allclose(plain, stabilised)
    dc0 = rho_hat[:, :, 0, 0, 0]
    assert torch.equal(stabilised[:, :, 0, 0, 0], dc0)


def test_kappa_roll_correction_is_itself_exactly_zero_at_k0():
    model = _ToyModelB(GRID, 2)
    boxes = _boxes(2)
    T = _T(2)
    rho_hat = _rho_hat(2, 2, GRID, model.ops)
    ops = model.ops
    N = ops.grid[0] * ops.grid[1] * ops.grid[2]
    rho = ops.irfft(rho_hat * N)
    kx, ky, kz = ops.k_axes(boxes)

    correction = kappa_roll_correction(model, rho, rho_hat, T, ops, kx, ky,
                                       kz, kappa_roll=1.3)
    k0 = correction[:, :, 0, 0, 0]
    assert torch.all(k0.real == 0.0)
    assert torch.all(k0.imag == 0.0)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_assert_finite_raises_on_nan():
    x = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(FloatingPointError):
        assert_finite(x, "test")


def test_assert_finite_raises_on_inf():
    x = torch.tensor([1.0, float("inf")])
    with pytest.raises(FloatingPointError):
        assert_finite(x, "test")


def test_assert_finite_passes_a_clean_tensor():
    x = torch.tensor([1.0, 2.0, -3.0])
    assert_finite(x, "test")  # must not raise


def test_assert_finite_checks_complex_imaginary_part_too():
    """A NaN confined to the IMAGINARY part, with every real part finite,
    must still raise -- the real-only check `torch.isfinite(x.real)` alone
    is blind to this, since it never inspects `x.imag`."""
    x = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([1.0, float("nan")]))
    assert torch.isfinite(x.real).all(), "fixture: real part must be clean"
    assert not torch.isfinite(x.imag).all(), "fixture must actually corrupt imag"
    with pytest.raises(FloatingPointError):
        assert_finite(x, "test")


def test_clamped_inputs_clamps_and_restores():
    model = _ToyModelB(GRID, 1)
    seen = {}

    def _spy_mu(rho, boxes, T):
        seen["rho_min"] = float(rho.min())
        return rho
    model.chemical_potential = _spy_mu

    rho = torch.tensor([[-1.0, 0.2, 5.0]]).view(1, 1, 3)
    with clamped_inputs(model, floor=0.0):
        model.chemical_potential(rho, None, None)
    assert seen["rho_min"] == pytest.approx(0.0)

    # restored afterwards: calling it again with the same negative input
    # must see the UNCLAMPED value, i.e. the wrapper is gone.
    model.chemical_potential(rho, None, None)
    assert seen["rho_min"] == pytest.approx(-1.0)


def test_clamped_inputs_restores_on_exception():
    model = _ToyModelB(GRID, 1)
    with pytest.raises(RuntimeError):
        with clamped_inputs(model, floor=0.0):
            raise RuntimeError("boom")
    # Restored to the unwrapped class method: a negative input passes
    # through untouched (mu = rho, no clamp).
    rho = torch.tensor([[-1.0]]).view(1, 1, 1)
    out = model.chemical_potential(rho, None, None)
    assert float(out) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# trust domain: no default, generic geometry, n_species = 1 and 2
# ---------------------------------------------------------------------------

def test_in_domain_with_no_declared_corners_raises():
    rho = torch.tensor([[0.5, 0.3]])
    with pytest.raises(ValueError, match="no default"):
        in_domain(rho, None)


@pytest.mark.parametrize("n_species", [1, 2])
def test_trust_domain_rejects_a_non_positive_intercept(n_species):
    inner = tuple([0.1] * n_species)
    outer_bad = tuple([0.0] + [0.5] * (n_species - 1))
    with pytest.raises(ValueError):
        TrustDomain(inner=inner, outer=outer_bad)


def test_in_domain_at_n_species_1_is_an_interval():
    domain = TrustDomain(inner=(0.2,), outer=(0.8,))
    rho = torch.tensor([[0.1], [0.2], [0.5], [0.8], [0.9]])
    mask = in_domain(rho, domain)
    assert mask.tolist() == [False, True, True, True, False]


def test_in_domain_at_n_species_2_matches_the_generic_trapezoid_formula():
    domain = TrustDomain(inner=(0.4, 0.3), outer=(1.0, 0.7))
    # A point on the inner edge exactly, and one comfortably inside/outside.
    on_inner_edge = torch.tensor([0.4, 0.0])   # 0.4/0.4 + 0/0.3 = 1.0
    comfortably_in = torch.tensor([0.5, 0.2])
    outside_low = torch.tensor([0.05, 0.02])   # below the inner edge
    outside_high = torch.tensor([2.0, 2.0])    # beyond the outer edge
    negative = torch.tensor([-0.1, 0.2])

    rho = torch.stack(
        [on_inner_edge, comfortably_in, outside_low, outside_high, negative])
    mask = in_domain(rho, domain)
    assert mask.tolist() == [True, True, False, False, False]


def test_in_domain_requires_T_only_when_the_domain_declares_a_T_range():
    density_only = TrustDomain(inner=(0.2,), outer=(0.8,))
    rho = torch.tensor([[0.5]])
    # No T needed at all; passing none must not raise.
    assert bool(in_domain(rho, density_only)[0])

    gated = TrustDomain(inner=(0.2,), outer=(0.8,), T_range=(100.0, 200.0))
    with pytest.raises(ValueError, match="T_range"):
        in_domain(rho, gated)

    assert bool(in_domain(rho, gated, T=torch.tensor([150.0]))[0])
    assert not bool(in_domain(rho, gated, T=torch.tensor([500.0]))[0])


def test_trust_domain_shape_mismatch_raises():
    domain = TrustDomain(inner=(0.2, 0.1), outer=(0.8, 0.5))
    rho = torch.tensor([[0.5]])  # 1 channel, domain declares 2
    with pytest.raises(ValueError):
        in_domain(rho, domain)


def test_project_trust_domain_leaves_interior_points_untouched():
    domain = TrustDomain(inner=(0.4, 0.3), outer=(1.0, 0.7))
    rho = torch.tensor([[0.5, 0.2]])
    out = project_trust_domain(rho, domain)
    assert torch.equal(out, rho)


def test_project_trust_domain_moves_exterior_points_onto_the_boundary():
    domain = TrustDomain(inner=(0.4, 0.3), outer=(1.0, 0.7))
    rho = torch.tensor([[0.0, 0.0]])  # violates the inner edge
    out = project_trust_domain(rho, domain)
    assert not torch.equal(out, rho)
    inner_sum = out[0, 0] / domain.inner[0] + out[0, 1] / domain.inner[1]
    assert float(inner_sum) == pytest.approx(1.0, abs=1e-4)


def test_in_domain_nonnegativity_is_checked_independently_of_the_edges():
    """A point with a NEGATIVE channel that still happens to satisfy both
    edge sums (a generous domain makes this easy to construct) must still
    be rejected -- nonnegativity is its own, separate condition, not
    something the two edge inequalities already imply for every domain.
    """
    domain = TrustDomain(inner=(0.1, 0.1), outer=(10.0, 10.0))
    rho = torch.tensor([[-0.05, 0.3]])
    inner_sum = float(rho[0, 0] / domain.inner[0] + rho[0, 1] / domain.inner[1])
    outer_sum = float(rho[0, 0] / domain.outer[0] + rho[0, 1] / domain.outer[1])
    assert inner_sum >= 1.0 and outer_sum <= 1.0, (
        "fixture must satisfy both edges, isolating nonnegativity")
    assert not bool(in_domain(rho, domain)[0])


# ---------------------------------------------------------------------------
# _cell_volume: the SDE noise amplitude's per-sample dV, pinned directly
# since the mass-conservation tests above cannot see a wrong SCALE (only a
# wrong k=0 row), and a magnitude bug here is invisible to every test that
# only checks decay direction or exact conservation.
# ---------------------------------------------------------------------------

def test_cell_volume_matches_box_over_grid_per_axis():
    from aipf.solve.integrators import _cell_volume

    boxes = torch.tensor([[8.0, 6.0, 4.0], [10.0, 10.0, 10.0]])
    grid = (8, 6, 8)
    dV = _cell_volume(boxes, grid)
    expected = torch.tensor([
        (8.0 / 8) * (6.0 / 6) * (4.0 / 8),
        (10.0 / 8) * (10.0 / 6) * (10.0 / 8),
    ])
    assert torch.allclose(dV, expected)


# ---------------------------------------------------------------------------
# kappa_roll_correction: sign and magnitude, not merely "changes the
# output" -- a sign flip is otherwise invisible (still zero at k=0, still
# different from the un-stabilised flux either way).
# ---------------------------------------------------------------------------

def test_kappa_roll_correction_is_stabilising_not_destabilising():
    """Isolate the correction alone (no model drift) on a single Fourier
    mode. Linearised, the extra flux this dial adds is
    ``div(M grad(-kappa_roll * lap(rho)))``, which for a constant
    ``M = M0 * I`` and one Fourier mode gives the closed-form rate
    ``-M0 * kappa_roll * k^4`` -- MORE negative (more stabilising, not
    less) than the plain ``-M0 * k^2`` rate the bulk term alone gives. A
    sign error here would make ``kappa_roll`` destabilise exactly the
    high-k fronts it exists to protect, invisibly to every other test in
    this file (still zero at k=0, still numerically different from the
    unstabilised flux either way).
    """
    grid = (8, 8, 8)
    length = 4.0
    model = _ToyModelB(grid, 1, M0=0.7)
    boxes = torch.tensor([[length, length, length]])
    T = torch.tensor([1.0])
    axis = torch.arange(grid[0], dtype=torch.float32) * (length / grid[0])
    rho = 0.5 + 0.01 * torch.cos(2 * torch.pi * axis / length).view(1, 1, -1, 1, 1)
    rho = rho.expand(1, 1, *grid).contiguous()
    N = grid[0] * grid[1] * grid[2]
    rho_hat = model.ops.rfft(rho) / N
    ops = model.ops
    kx, ky, kz = ops.k_axes(boxes)

    kappa_roll = 0.3
    correction = kappa_roll_correction(model, rho, rho_hat, T, ops, kx, ky,
                                       kz, kappa_roll)

    amplitude = rho_hat[0, 0, 1, 0, 0]
    rate = (correction[0, 0, 1, 0, 0] / amplitude).real
    k_mag = 2 * torch.pi * 1 / length
    expected_rate = -0.7 * kappa_roll * k_mag ** 4
    assert rate.item() == pytest.approx(expected_rate, rel=1e-3)


# ---------------------------------------------------------------------------
# step_heun: pinned against the exact update of a LINEAR ODE, so that
# collapsing to Euler (using the first stage twice) is caught even though
# both schemes still decay for a small enough dt.
# ---------------------------------------------------------------------------

def test_step_heun_matches_the_exact_linear_update_to_second_order():
    grid = (8, 8, 8)
    length = 4.0
    model = _ToyModelB(grid, 1, M0=0.7)
    boxes = torch.tensor([[length, length, length]])
    T = torch.tensor([1.0])
    axis = torch.arange(grid[0], dtype=torch.float32) * (length / grid[0])
    rho = 0.5 + 0.01 * torch.cos(2 * torch.pi * axis / length).view(1, 1, -1, 1, 1)
    rho = rho.expand(1, 1, *grid).contiguous()
    N = grid[0] * grid[1] * grid[2]
    rho_hat = model.ops.rfft(rho) / N

    k_mag = 2 * torch.pi * 1 / length
    a = 0.7 * k_mag ** 2  # exact linear rate: d(rho_k)/dt = -a * rho_k
    dt = 0.05  # large enough that Euler and Heun visibly disagree

    amplitude = rho_hat[0, 0, 1, 0, 0]
    euler_out = step_euler(model, rho_hat, boxes, T, dt)
    heun_out = step_heun(model, rho_hat, boxes, T, dt)
    euler_ratio = (euler_out[0, 0, 1, 0, 0] / amplitude).real
    heun_ratio = (heun_out[0, 0, 1, 0, 0] / amplitude).real

    assert euler_ratio.item() == pytest.approx(1.0 - dt * a, rel=1e-4)
    assert heun_ratio.item() == pytest.approx(
        1.0 - dt * a + 0.5 * (dt * a) ** 2, rel=1e-4)
    # The two schemes must genuinely disagree at this dt -- otherwise this
    # test cannot distinguish Heun from a second Euler call.
    assert abs(heun_ratio.item() - euler_ratio.item()) > 1e-4


# ---------------------------------------------------------------------------
# rollout_deterministic: method dispatch is real, and an unknown method
# raises before any model.forward call
# ---------------------------------------------------------------------------

def test_rollout_deterministic_heun_and_euler_methods_genuinely_differ():
    grid = (8, 8, 8)
    length = 4.0
    model = _ToyModelB(grid, 1, M0=0.7)
    boxes = torch.tensor([[length, length, length]])
    T = torch.tensor([1.0])
    axis = torch.arange(grid[0], dtype=torch.float32) * (length / grid[0])
    rho = 0.5 + 0.01 * torch.cos(2 * torch.pi * axis / length).view(1, 1, -1, 1, 1)
    rho = rho.expand(1, 1, *grid).contiguous()
    N = grid[0] * grid[1] * grid[2]
    rho_hat0 = model.ops.rfft(rho) / N

    euler_traj = rollout_deterministic(model, rho_hat0, boxes, T, dt=0.05,
                                       n_steps=3, method="euler",
                                       **_roll_kw())
    heun_traj = rollout_deterministic(model, rho_hat0, boxes, T, dt=0.05,
                                      n_steps=3, method="heun", **_roll_kw())
    assert not torch.allclose(euler_traj[-1], heun_traj[-1])


def test_rollout_deterministic_rejects_an_unknown_method():
    model = _ToyModelB(GRID, 1)
    boxes = _boxes(1)
    T = _T(1)
    rho_hat0 = _rho_hat(1, 1, GRID, model.ops)
    with pytest.raises(ValueError, match="method"):
        rollout_deterministic(model, rho_hat0, boxes, T, dt=1e-3, n_steps=2,
                              method="nonexistent", **_roll_kw())


# ---------------------------------------------------------------------------
# the noise colour, declared rather than defaulted
# ---------------------------------------------------------------------------

def _sde_kw(**over):
    """The minimum a caller must now say out loud, plus overrides."""
    kw = {"noise_mode": "none", "sigma_noise": None}
    kw.update(over)
    return kw


def test_omitting_noise_mode_raises_and_the_message_names_both_values():
    """The point of the whole keyword: a caller who did not know there was
    a choice is told there is one, and told what the two answers are.

    A bare required keyword-only argument would raise too, but Python's
    own message names only the missing NAME. The defect this closes was
    not a typo, it was four drivers whose authors never learned the choice
    existed, so the message has to carry the answer set.
    """
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat = _rho_hat(1, 2, GRID, model.ops)

    with pytest.raises(ValueError) as exc:
        step_sde_euler_maruyama(model, rho_hat, boxes, T, dt=1e-3,
                                kBT_noise=0.05)
    assert "gaussian" in str(exc.value) and "none" in str(exc.value)

    with pytest.raises(ValueError, match="sigma_noise must be declared"):
        step_sde_euler_maruyama(model, rho_hat, boxes, T, dt=1e-3,
                                kBT_noise=0.05, noise_mode="none")

    with pytest.raises(ValueError) as exc:
        rollout_sde(model, rho_hat, boxes, T, dt=1e-3, n_steps=1,
                    kBT_noise=0.05, **_roll_kw())
    assert "gaussian" in str(exc.value) and "none" in str(exc.value)


@pytest.mark.parametrize("mode,sigma,match", [
    ("Gaussian", 2.0, "must be one of"),          # case is not a spelling
    ("hard", 2.0, "must be one of"),
    ("gaussian", None, "requires a positive sigma_noise"),
    ("gaussian", 0.0, "finite and positive"),
    ("gaussian", -1.0, "finite and positive"),
    ("none", 2.0, "takes sigma_noise=None"),
])
def test_an_ill_formed_noise_declaration_is_refused(mode, sigma, match):
    """Half-declared is not declared: a width with no filter to apply it
    to, and a filter with no width, are both caller bugs."""
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat = _rho_hat(1, 2, GRID, model.ops)
    with pytest.raises(ValueError, match=match):
        step_sde_euler_maruyama(model, rho_hat, boxes, T, dt=1e-3,
                                kBT_noise=0.05, noise_mode=mode,
                                sigma_noise=sigma)


def test_the_declaration_is_checked_before_the_deterministic_delegation():
    """``kBT_noise = 0`` does not excuse an undeclared filter.

    Otherwise a driver could be written, tested with the noise off, and
    only fail the day someone turns it on -- which is the shape of the
    failure this argument exists to prevent, not a case to be lenient
    about.
    """
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat = _rho_hat(1, 2, GRID, model.ops)
    with pytest.raises(ValueError, match="noise_mode must be declared"):
        step_sde_euler_maruyama(model, rho_hat, boxes, T, dt=1e-3,
                                kBT_noise=0.0)


def test_kBT_noise_of_none_is_refused_rather_than_silently_deterministic():
    """``None`` was the OTHER package's default and meant "use kT".

    ``if not kBT_noise:`` is true for it, so a call ported with its
    arguments intact ran with no noise at all, no error raised, and a
    trajectory that looked like a solution. Nothing about ``0.0`` changes:
    that is the documented deterministic delegation and stays bit-for-bit.
    """
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat = _rho_hat(1, 2, GRID, model.ops)

    with pytest.raises(ValueError, match="kBT_noise=None is refused"):
        step_sde_euler_maruyama(model, rho_hat, boxes, T, dt=1e-3,
                                kBT_noise=None, **_sde_kw())
    with pytest.raises(ValueError, match="kBT_noise=None is refused"):
        rollout_sde(model, rho_hat, boxes, T, dt=1e-3, n_steps=1,
                    kBT_noise=None, **_sde_kw(), **_roll_kw())
    with pytest.raises(ValueError, match="non-negative"):
        step_sde_euler_maruyama(model, rho_hat, boxes, T, dt=1e-3,
                                kBT_noise=-1.0, **_sde_kw())

    zero = step_sde_euler_maruyama(model, rho_hat, boxes, T, dt=1e-3,
                                   kBT_noise=0.0, **_sde_kw())
    assert torch.equal(zero, step_euler(model, rho_hat, boxes, T, dt=1e-3))


def test_the_declaration_is_checked_eagerly_but_the_filter_is_not_built():
    """Eager VALIDATION, lazy CONSTRUCTION -- and they are different
    things.

    The check has to be eager: that is the property the test above pins,
    and it is what stops a driver tested with the noise off from carrying
    an undeclared colour. Building the tensor does not have to be, and
    building it eagerly meant a deterministic rollout driven through this
    entry point allocated and threw away a half-grid exponential on every
    step. Counted here rather than argued: ``sigma_filter`` is the only
    thing that builds it.
    """
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat = _rho_hat(1, 2, GRID, model.ops)
    calls = {"n": 0}
    real = model.ops.sigma_filter

    def counted(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    model.ops.sigma_filter = counted
    try:
        step_sde_euler_maruyama(model, rho_hat, boxes, T, 1e-3, 0.0,
                                noise_mode="gaussian", sigma_noise=2.0,
                                ops=model.ops)
        assert calls["n"] == 0, "the zero-noise path built a filter"
        step_sde_euler_maruyama(model, rho_hat, boxes, T, 1e-3, 0.05,
                                noise_mode="gaussian", sigma_noise=2.0,
                                ops=model.ops,
                                generator=torch.Generator().manual_seed(1))
        assert calls["n"] == 1, calls
        # and the eager check still fires on that same zero-noise path
        with pytest.raises(ValueError, match="noise_mode must be declared"):
            step_sde_euler_maruyama(model, rho_hat, boxes, T, 1e-3, 0.0,
                                    ops=model.ops)
        assert calls["n"] == 1, calls
    finally:
        model.ops.sigma_filter = real


def test_the_gaussian_filter_is_the_coarse_graining_filter_itself():
    """``build_noise_filter`` does not re-type ``exp(-k^2 sigma^2/2)``; it
    returns the ops' own filter, and ``"none"`` returns ``None`` so the
    white path executes no arithmetic at all."""
    from aipf.solve import build_noise_filter

    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    boxes = _boxes(2)
    assert build_noise_filter(ops, boxes, "none", None) is None
    G = build_noise_filter(ops, boxes, "gaussian", 2.0)
    assert torch.equal(G, ops.sigma_filter(boxes, 2.0))
    assert torch.equal(G, torch.exp(-ops.k2(boxes) * (2.0 * 2.0 / 2.0)))
    assert torch.equal(G[..., 0, 0, 0], torch.ones_like(G[..., 0, 0, 0]))


class _ZeroDrift(_ToyModelB):
    """``_ToyModelB`` with the deterministic flux switched off.

    Used wherever a claim is about the NOISE increment alone and has to be
    bit-for-bit: with a zero drift AND a uniform initial state, every
    ``k != 0`` entry of the step is ``0 + dt * noise`` exactly, so the
    increment can be read straight off the result instead of being
    recovered by a subtraction that would round.
    """

    def forward(self, rho_hat, boxes, T):
        del boxes, T
        return torch.zeros_like(rho_hat)


def _uniform_state(n_species, grid, value=0.4):
    rho_hat = torch.zeros(1, n_species, grid[0], grid[1], grid[2] // 2 + 1,
                          dtype=torch.complex64)
    rho_hat[..., 0, 0, 0] = value
    return rho_hat


def test_the_colour_filter_multiplies_the_noise_and_leaves_the_drift_alone():
    """The filter's effect, isolated bit-for-bit rather than statistically.

    Same seed, same everything, two modes: the noise increment of the
    coloured step is EXACTLY ``G(k)`` times the white one. This is the
    assertion that a filter applied to the wrong tensor -- the state, the
    drift, the whole increment -- cannot pass.
    """
    model = _ZeroDrift(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat = _uniform_state(2, GRID)
    # dt = 1 so that the step's own `dt *` is the exact identity and the
    # only arithmetic separating the two results is the filter multiply.
    # At any other dt the same claim holds only up to ONE reassociated
    # multiply, `G*(dt*n)` against `dt*(G*n)`, which is a statement about
    # float multiplication rather than about this integrator.
    dt, kBT = 1.0, 0.05

    white = step_sde_euler_maruyama(
        model, rho_hat, boxes, T, dt, kBT, noise_mode="none",
        sigma_noise=None, generator=torch.Generator().manual_seed(5))
    coloured = step_sde_euler_maruyama(
        model, rho_hat, boxes, T, dt, kBT, noise_mode="gaussian",
        sigma_noise=2.0, generator=torch.Generator().manual_seed(5))

    G = model.ops.sigma_filter(boxes, 2.0)
    assert torch.equal(coloured - rho_hat, G * (white - rho_hat))
    assert not torch.equal(coloured, white)
    # and it is a LARGE effect, not a tweak: the highest mode on this grid
    # is suppressed by orders of magnitude.
    assert float(G.min()) < 1e-6


@pytest.mark.parametrize("mode,sigma", [("none", None), ("gaussian", 2.0)])
def test_the_coloured_noise_still_conserves_mass_exactly(mode, sigma):
    """``G(k)`` multiplies a divergence that is already exactly zero at
    ``k = 0``, and ``G(0) = 1`` could not disturb it anyway."""
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(2), _T(2)
    rho_hat = _rho_hat(2, 2, GRID, model.ops)
    dc0 = rho_hat[:, :, 0, 0, 0].clone()
    gen = torch.Generator().manual_seed(17)
    for _ in range(8):
        out = step_sde_euler_maruyama(model, rho_hat, boxes, T, dt=1e-3,
                                      kBT_noise=0.05, noise_mode=mode,
                                      sigma_noise=sigma, generator=gen)
        assert torch.equal(out[:, :, 0, 0, 0], dc0)


# ---------------------------------------------------------------------------
# the external drive: v_ext and T_field
# ---------------------------------------------------------------------------

class _ToyModelBT(_ToyModelB):
    """``_ToyModelB`` with a temperature-dependent local term,
    ``f_loc = rho^2/2 + c T rho^3/3``, so ``mu_loc = rho + c T rho^2`` and
    the per-cell temperature shift has a closed form to check against."""

    def __init__(self, grid, n_species, c: float = 0.25, **kw):
        super().__init__(grid, n_species, **kw)
        self.c = c

    def bulk_free_energy_density(self, rho, T):
        t = T.view(-1, *([1] * (rho.dim() - 1)))
        return (0.5 * (rho ** 2) + self.c * t * (rho ** 3) / 3.0).sum(
            dim=1, keepdim=True)

    def chemical_potential(self, rho, boxes, T):
        del boxes
        t = T.view(-1, *([1] * (rho.dim() - 1)))
        return rho + self.c * t * rho ** 2


def test_both_external_drives_off_is_bit_for_bit_the_undriven_step():
    """``None`` means no arithmetic, not a multiply by one."""
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat = _rho_hat(1, 2, GRID, model.ops)
    a = step_euler(model, rho_hat, boxes, T, 1e-3)
    b = step_euler(model, rho_hat, boxes, T, 1e-3, v_ext=None, T_field=None)
    assert torch.equal(a, b)


def test_v_ext_enters_as_the_flux_a_potential_on_mu_would_give():
    """One Fourier mode, where the answer is known in closed form.

    For ``V(r) = A cos(k_z z)`` on species 0 and a constant isotropic
    mobility, ``div[M grad V] = -M0 k_z^2 V``, so the extra spectral flux
    must be ``-M0 k_z^2 V_hat`` mode for mode. A sign error, a missing
    mobility contraction or a gradient taken in the wrong convention all
    fail this; "it changed the answer" would not.
    """
    from aipf.solve.integrators import _external_flux_hat

    model = _ToyModelB(GRID, 2, M0=0.7)
    boxes, T = _boxes(1), _T(1)
    ops, N = model.ops, GRID[0] * GRID[1] * GRID[2]
    Lz = float(boxes[0, 2])
    z = torch.arange(GRID[2], dtype=torch.float32) * (Lz / GRID[2])
    kz = 2.0 * torch.pi / Lz
    V = torch.zeros(1, 2, *GRID)
    V[0, 0] = 0.3 * torch.cos(kz * z).view(1, 1, -1)

    rho = 0.4 + torch.zeros(1, 2, *GRID)
    kx_, ky_, kz_ = ops.k_axes(boxes)
    got = _external_flux_hat(model, rho, T, ops, kx_, ky_, kz_, V)
    want = -0.7 * (kz ** 2) * (ops.rfft(V) / N)
    assert torch.allclose(got, want, atol=1e-6), (got.abs().max(),
                                                   (got - want).abs().max())
    assert torch.equal(got[..., 0, 0, 0], torch.zeros_like(got[..., 0, 0, 0]))


def test_the_external_flux_is_ADDED_to_the_drift_with_the_sign_it_has():
    """The composition, not just the term.

    The test above checks ``_external_flux_hat`` by calling it directly,
    which cannot see how ``_rhs`` puts it together with ``model.forward``.
    A mutation that SUBTRACTS the external flux instead of adding it
    survived the whole core suite when this test was not here, and was
    caught only by a comparison against the package this one supersedes --
    i.e. by a test that cannot run without that package present. This
    closes it here, where nothing external is needed: with the drift
    switched off, the whole step IS the external flux, and its closed form
    is known.
    """
    model = _ZeroDrift(GRID, 2, M0=0.7)
    boxes, T = _boxes(1), _T(1)
    rho_hat = _uniform_state(2, GRID)
    ops, N = model.ops, GRID[0] * GRID[1] * GRID[2]
    Lz = float(boxes[0, 2])
    z = torch.arange(GRID[2], dtype=torch.float32) * (Lz / GRID[2])
    kz = 2.0 * torch.pi / Lz
    V = torch.zeros(1, 2, *GRID)
    V[0, 0] = 0.3 * torch.cos(kz * z).view(1, 1, -1)

    out = step_euler(model, rho_hat, boxes, T, 1.0, v_ext=V)
    want = -0.7 * (kz ** 2) * (ops.rfft(V) / N)
    assert torch.allclose(out - rho_hat, want, atol=1e-6), (
        (out - rho_hat).abs().max(), want.abs().max())
    # sign, stated separately so a sign flip cannot hide inside a
    # tolerance: the flux must OPPOSE the potential's own shape.
    got = (out - rho_hat)[0, 0]
    ref = (ops.rfft(V) / N)[0, 0]
    overlap = float((got * ref.conj()).real.sum())
    assert overlap < 0, overlap


def test_a_driven_rollout_still_conserves_mass_exactly():
    model = _ToyModelBT(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat0 = _rho_hat(1, 2, GRID, model.ops)
    dc0 = rho_hat0[:, :, 0, 0, 0].clone()
    g = torch.Generator().manual_seed(2)
    V = 0.05 * torch.randn(2, *GRID, generator=g)
    Tf = 1.5 + 0.2 * torch.rand(GRID, generator=g)

    traj = rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 4,
                                 v_ext=V, T_field=Tf, save_every=1,
                                 **_roll_kw())
    for frame in traj:
        assert torch.equal(frame[:, :, 0, 0, 0], dc0)

    traj = rollout_sde(model, rho_hat0, boxes, T, 1e-3, 4, kBT_noise=0.02,
                       noise_mode="gaussian", sigma_noise=2.0, v_ext=V,
                       T_field=Tf, save_every=1, **_roll_kw(),
                       generator=torch.Generator().manual_seed(3))
    for frame in traj:
        assert torch.equal(frame[:, :, 0, 0, 0], dc0)


def test_the_temperature_shift_is_the_closed_form_it_claims():
    """``mu_loc(rho, T(r)) - mu_loc(rho, T)``, against the fixture's
    analytic ``c (T(r) - T) rho^2``."""
    from aipf.solve.integrators import _local_mu_shift

    model = _ToyModelBT(GRID, 2, c=0.25)
    T = _T(1, 1.5)
    g = torch.Generator().manual_seed(8)
    rho = 0.3 + 0.1 * torch.rand(1, 2, *GRID, generator=g)
    Tf = 1.0 + torch.rand(GRID, generator=g)

    got = _local_mu_shift(model, rho, T, Tf)
    want = 0.25 * (Tf.view(1, 1, *GRID) - 1.5) * rho ** 2
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_a_uniform_temperature_field_is_the_scalar_path_bit_for_bit():
    """Not "to 1e-9": the same numbers.

    The shift is a DIFFERENCE of two evaluations made in one call, so a
    uniform field equal to ``T`` makes the two halves bit-identical inputs
    and the difference an exact zero -- drift and noise alike. That is the
    property a driver relies on when it runs a constant-temperature case
    through the same code path as a profiled one.
    """
    model = _ToyModelBT(GRID, 2)
    boxes, T = _boxes(1), _T(1, 1.5)
    rho_hat = _rho_hat(1, 2, GRID, model.ops)
    uniform = torch.full(GRID, 1.5)

    assert torch.equal(
        step_euler(model, rho_hat, boxes, T, 1e-3, T_field=uniform),
        step_euler(model, rho_hat, boxes, T, 1e-3))
    a = step_sde_euler_maruyama(
        model, rho_hat, boxes, T, 1e-3, 0.05, noise_mode="gaussian",
        sigma_noise=2.0, T_field=uniform,
        generator=torch.Generator().manual_seed(4))
    b = step_sde_euler_maruyama(
        model, rho_hat, boxes, T, 1e-3, 0.05, noise_mode="gaussian",
        sigma_noise=2.0, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a, b)


def test_a_hotter_field_injects_more_noise_and_a_colder_one_less():
    """Local-equilibrium FDT, in its weakest checkable form: the noise
    increment scales as ``sqrt(T(r)/T)`` per cell, so a field uniformly
    ``4x`` the scalar temperature doubles it EXACTLY on the same draw.

    ``_ZeroDrift`` inherits a temperature-INDEPENDENT local form, so the
    drift is untouched by ``T_field`` here and the whole difference is the
    noise -- which is the quantity the claim is about.
    """
    model = _ZeroDrift(GRID, 2)
    boxes, T = _boxes(1), _T(1, 1.5)
    rho_hat = _uniform_state(2, GRID)

    def increment(T_field):
        out = step_sde_euler_maruyama(
            model, rho_hat, boxes, T, 1e-3, 0.05, noise_mode="none",
            sigma_noise=None, T_field=T_field,
            generator=torch.Generator().manual_seed(6))
        return out - rho_hat

    hot = increment(torch.full(GRID, 4 * 1.5))
    one = increment(torch.full(GRID, 1.5))
    cold = increment(torch.full(GRID, 1.5 / 4))
    assert torch.equal(hot, 2.0 * one)
    assert torch.equal(cold, 0.5 * one)


@pytest.mark.parametrize("bad,match", [
    (torch.zeros(GRID), "finite and strictly positive"),
    (torch.full(GRID, float("nan")), "finite and strictly positive"),
    (torch.ones((3, 3, 3)), "must have shape"),
])
def test_an_ill_formed_temperature_field_is_refused(bad, match):
    model = _ToyModelBT(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat = _rho_hat(1, 2, GRID, model.ops)
    with pytest.raises(ValueError, match=match):
        step_euler(model, rho_hat, boxes, T, 1e-3, T_field=bad)


def test_an_ill_formed_external_potential_is_refused():
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat = _rho_hat(1, 2, GRID, model.ops)
    with pytest.raises(ValueError, match="must have shape"):
        step_euler(model, rho_hat, boxes, T, 1e-3,
                   v_ext=torch.zeros(1, *GRID))
    with pytest.raises(ValueError, match="finite"):
        step_euler(model, rho_hat, boxes, T, 1e-3,
                   v_ext=torch.full((2, *GRID), float("inf")))


# ---------------------------------------------------------------------------
# the state projection and the input guard are DECLARED, never defaulted
#
# The defect these close is the one aipf.solve.declare states: a production
# driver set the knob from its own argparse default and passed it; the port
# carried the function and not the argument, so it ran the function's own
# SIGNATURE default instead, which is a different value. Nothing errored.
# ---------------------------------------------------------------------------

def _domain():
    """A two-species trapezoid with round numbers.

    Deliberately NOT any measured system's corners: `src/aipf/` carries
    none by construction, and a package test that reached
    for a real system's would be re-introducing exactly the literal that
    module's docstring records an accident about.
    """
    return TrustDomain(inner=(0.4, 0.25), outer=(1.0, 0.6))


#: The positive lower bound `state_proj="domain"` requires. It has no
#: default in the package, deliberately (the trapezoid's two axis edges are
#: admissible points with a literal 0.0 in a channel), so every domain-mode
#: call in this file has to name one, and they all name this.
_LO = 1e-3


def test_project_state_refuses_an_undeclared_target_and_names_both():
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    rho_hat = _rho_hat(1, 2, GRID, ops)
    with pytest.raises(ValueError) as exc:
        project_state(rho_hat, ops, 1e-3)
    assert "floor" in str(exc.value) and "domain" in str(exc.value)


def test_project_state_refuses_an_undeclared_floor():
    """The floor is the second half of the declaration, and it is the half
    that was actually wrong: every production driver passed 1e-3 where the
    function's own signature said `None` -> 0.0.
    """
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    rho_hat = _rho_hat(1, 2, GRID, ops)
    with pytest.raises(ValueError, match="floor must be declared"):
        project_state(rho_hat, ops, UNDECLARED, state_proj="floor")


def test_project_state_rejects_an_unknown_target():
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    rho_hat = _rho_hat(1, 2, GRID, ops)
    with pytest.raises(ValueError, match="must be one of"):
        project_state(rho_hat, ops, 1e-3, state_proj="trapezoid")


def test_project_state_in_domain_mode_requires_a_declared_domain():
    """`None` raises rather than falling back to a module constant, for the
    same reason `in_domain` does: core carries no corners, so a fallback
    could only describe one system.
    """
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    rho_hat = _rho_hat(1, 2, GRID, ops)
    with pytest.raises(ValueError, match="requires a declared TrustDomain"):
        project_state(rho_hat, ops, 1e-3, state_proj="domain", lo=_LO)


def test_project_state_in_domain_mode_moves_cells_onto_the_trapezoid():
    """The target every production driver passed, doing the thing the floor
    cannot: enforcing an UPPER bound as well as non-negativity.

    The violating cell is placed OUTSIDE the outer edge, where a floor
    projection at any floor is a no-op -- so this cannot pass by accident
    on a mode that ignored `state_proj` and clamped instead.
    """
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    domain = _domain()
    rho = torch.zeros(1, 2, *GRID)
    # The bulk sits strictly INSIDE the outer edge (0.5/1.0 + 0.2/0.6 =
    # 0.83), not on it. A bulk exactly ON the edge would be pushed outside
    # by the exact-mass restore's uniform shift -- which is a real property
    # of this alternation (the helper this replaces records the same thing:
    # out-of-domain fraction 0.0020 -> 0.0000 -> 1.0000 across one restore)
    # and would make the measurement below about the fixture rather than
    # about the projection.
    rho[:, 0] = 0.5
    rho[:, 1] = 0.2
    rho[0, 0, 0, 0, 0] = 1.6        # far outside the outer edge
    rho[0, 1, 0, 0, 0] = 0.9
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat = ops.rfft(rho) / N
    dc0 = rho_hat[:, :, 0, 0, 0].clone()

    cells = rho.permute(0, 2, 3, 4, 1).reshape(-1, 2)
    assert not bool(in_domain(cells, domain).all()), (
        "fixture must actually leave the trapezoid")
    # the violator is outside the OUTER edge, which no floor can reach
    assert bool((cells >= 0.0).all()), (
        "fixture must violate only the upper edge, so a floor projection "
        "at any floor is a no-op on it")

    out = project_state(rho_hat, ops, 0.0, state_proj="domain",
                        domain=domain, lo=_LO)
    rho_out = ops.irfft(out * N)
    cells_out = rho_out.permute(0, 2, 3, 4, 1).reshape(-1, 2)

    assert torch.equal(out[:, :, 0, 0, 0], dc0), "mass must stay exact"
    frac_before = float((~in_domain(cells, domain)).to(torch.float32).mean())
    frac_after = float((~in_domain(cells_out, domain)).to(torch.float32).mean())
    assert frac_before > 0.0
    # NOT `frac_after < frac_before`. The out-of-domain FRACTION is not
    # monotone under this alternation and the helper this replaces says so
    # in as many words: the exact-mass restore is a uniform real-space
    # shift, so it pushes every cell -- including the one the projection
    # just placed exactly on the boundary -- back out by the same amount,
    # and a cell can stay nominally outside while its violation collapses
    # by four orders of magnitude. Depth is the quantity that measures
    # whether the projection ran.
    assert frac_after <= frac_before

    def _depth(c):
        """How far past the OUTER edge the worst cell is, in the edge's own
        normalised units. The fraction alone cannot say whether a cell was
        moved to the boundary or merely nudged."""
        outer = torch.tensor(domain.outer)
        return float(((c / outer).sum(-1) - 1.0).clamp(min=0.0).max())

    assert _depth(cells) > 1.0, "fixture must be deeply outside"
    assert _depth(cells_out) < 1e-3

    # and the floor branch at the same arguments does nothing at all
    floor_out = project_state(rho_hat, ops, 0.0, state_proj="floor")
    assert floor_out is rho_hat


def test_project_state_in_domain_mode_ignores_the_floor():
    """Carried forward verbatim from the helper this replaces: in domain
    mode the trapezoid IS the admissible set and `floor` is not consulted.

    That is not an oversight to be fixed later -- all four production
    drivers pass `state_clamp=1e-3` AND `state_proj="domain"`, so a port
    that quietly added the floor to the trapezoid would be solving a
    different admissible set from the published runs. Two very different
    floors must give bit-identical answers.
    """
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    domain = _domain()
    rho = torch.zeros(1, 2, *GRID)
    rho[:, 0] = 0.5
    rho[:, 1] = 0.3
    rho[0, 0, 0, 0, 0] = -0.4       # below any floor AND outside the trapezoid
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat = ops.rfft(rho) / N

    a = project_state(rho_hat, ops, 0.0, state_proj="domain", domain=domain,
                      lo=_LO)
    b = project_state(rho_hat, ops, 0.5, state_proj="domain", domain=domain,
                      lo=_LO)
    assert torch.equal(a, b)


def test_project_state_in_domain_mode_is_a_no_op_on_an_admissible_field():
    """The early exit, which is the branch a production rollout spends all
    its time in -- and the one a mutation that deleted it would otherwise
    survive, since a violating fixture never reaches it.
    """
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    domain = _domain()
    rho = torch.zeros(1, 2, *GRID)
    rho[:, 0] = 0.5
    rho[:, 1] = 0.3
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat = ops.rfft(rho) / N
    cells = rho.permute(0, 2, 3, 4, 1).reshape(-1, 2)
    assert bool(in_domain(cells, domain).all()), "fixture must be admissible"

    out = project_state(rho_hat, ops, 0.0, state_proj="domain",
                        domain=domain, lo=_LO)
    assert torch.equal(out, rho_hat)


def test_check_state_projection_decides_everything_before_any_projection():
    """The eager half, split out so a rollout can refuse before its time
    loop -- at zero model evaluations, the same contract the noise colour
    has. Whatever `project_state` would raise, this raises.
    """
    with pytest.raises(ValueError, match="state_proj must be declared"):
        check_state_projection(UNDECLARED, 1e-3, None)
    with pytest.raises(ValueError, match="floor must be declared"):
        check_state_projection("floor", UNDECLARED, None)
    with pytest.raises(ValueError, match="requires a declared TrustDomain"):
        check_state_projection("domain", 0.0, None, lo=_LO)
    with pytest.raises(ValueError, match="TrustDomain"):
        check_state_projection("domain", 0.0, (0.4, 0.25, 1.0, 0.6), lo=_LO)
    check_state_projection("floor", 0.0, None)          # must not raise
    check_state_projection("domain", 0.0, _domain(), lo=_LO)   # must not raise


@pytest.mark.parametrize("missing", ["state_proj", "floor", "clamp_rho"])
def test_both_rollouts_refuse_an_undeclared_production_knob(missing):
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat0 = _rho_hat(1, 2, GRID, model.ops)
    kw = _roll_kw()
    kw.pop(missing)
    with pytest.raises(ValueError, match="must be declared"):
        rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 1, **kw)
    with pytest.raises(ValueError, match="must be declared"):
        rollout_sde(model, rho_hat0, boxes, T, 1e-3, 1, kBT_noise=0.0,
                    **_sde_kw(), **kw)


def test_both_rollouts_refuse_a_declaration_before_the_first_model_call():
    """Eagerly, at zero forwards -- the property that makes the refusal
    cheap enough to be unconditional. A model whose `forward` raises tells
    us whether the check ran first.
    """
    class _Exploding(_ToyModelB):
        def forward(self, rho_hat, boxes, T):
            raise AssertionError("a model forward ran before the check")

    model = _Exploding(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat0 = _rho_hat(1, 2, GRID, model.ops)
    with pytest.raises(ValueError, match="must be declared"):
        rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 5,
                              **_roll_kw(state_proj=UNDECLARED))
    with pytest.raises(ValueError, match="must be declared"):
        rollout_sde(model, rho_hat0, boxes, T, 1e-3, 5, kBT_noise=0.02,
                    **_sde_kw(), **_roll_kw(clamp_rho=UNDECLARED))


def test_clamp_rho_true_is_refused_because_its_value_is_rung_private():
    """`True` once meant "the rung's own rho_eps" --
    a rung-private attribute a protocol-only core cannot read, so the
    value would depend on which rung was loaded."""
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat0 = _rho_hat(1, 2, GRID, model.ops)
    with pytest.raises(ValueError, match="not carried"):
        rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 1,
                              **_roll_kw(clamp_rho=True))


@pytest.mark.parametrize("value", [False, 0, 0.0, -0.0])
def test_every_old_spelling_of_no_guard_is_refused_in_both_readings(value):
    """`False`, `0` and `0.0` are the three spellings of "no guard" in the
    published model's own training code -- its test is `if clamp_rho:` and its
    drivers' own `--clamp-rho` help says "0 disables". Read as this
    package's arithmetic the same value is a REAL clamp at zero, which is
    what stops a learned network being evaluated at a negative density.

    Two readings, opposite behaviours, and neither can be chosen for a
    caller. A ported driver line that meant "off" would otherwise silently
    acquire a guard -- and on a rung that watches its real-space seam, a
    guard is a measurable change to the drift even where the floor cannot
    fire. So all three raise, and the message names both readings and
    where each of them is spelled instead.

    `-0.0` is in the list because `-0.0 == 0.0` is True and `-0.0 is False`
    is not: the refusal has to come from the VALUE, not from the identity
    check that catches `False`.
    """
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat0 = _rho_hat(1, 2, GRID, model.ops)
    with pytest.raises(ValueError) as exc:
        rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 1,
                              **_roll_kw(clamp_rho=value))
    message = str(exc.value)
    assert "None" in message and "clamped_inputs" in message

    # and the escape hatch the message names has to exist AND compose the
    # way the message says, or the refusal removes a capability instead of
    # an ambiguity. Two things, because the message promises both.
    with clamped_inputs(model, 0.0):
        rho = torch.tensor([[-1.0, 0.5]]).view(1, 2, 1, 1, 1)
        assert float(model.chemical_potential(rho, None, None).min()) == 0.0

    # the composition the message spells out: wrapping the ROLLOUT gives a
    # guarded-at-zero rollout, and it is bit-for-bit the wrapper this
    # argument would have installed had it been allowed to take 0.0.
    seen = {"guarded": []}

    class _Watcher(_ToyModelB):
        def mobility(self, rho, T):
            seen["guarded"].append(float(rho.min()))
            return super().mobility(rho, T)

    watcher = _Watcher(GRID, 2)
    g = torch.Generator().manual_seed(23)
    negative = 0.05 + 0.2 * torch.randn(1, 2, *GRID, generator=g)
    assert float(negative.min()) < 0.0, "fixture must go negative"
    N = GRID[0] * GRID[1] * GRID[2]
    h0 = watcher.ops.rfft(negative) / N
    with clamped_inputs(watcher, 0.0):
        rollout_deterministic(watcher, h0, boxes, T, 1e-3, 2,
                              **_roll_kw(clamp_rho=None))
    assert seen["guarded"], "the model was never called"
    assert min(seen["guarded"]) >= 0.0, (
        "wrapping the rollout in clamped_inputs did not clamp its inputs: "
        "the refusal message above then points at a capability that is not "
        "there")


def test_clamp_rho_of_none_installs_no_wrapper_at_all():
    """`None` is bit-for-bit the unguarded rollout, and it gets there by
    installing nothing -- not by installing a wrapper that clamps at a
    harmless value.

    The distinction is load-bearing rather than stylistic: the wrapper is
    an instance attribute on `chemical_potential`, and a rung whose
    `forward` takes a faster k-space route when that seam is untouched
    changes route on the wrapper's mere PRESENCE (measured at 3.0e-05
    relative on the published model, with a floor that could not fire -- see
    `aipf.solve.guards.clamped_inputs`). So a "harmless" wrapper is a
    measurable change to the drift.

    Checked by asking the model, DURING the loop, whether either of the
    two names the guard installs is an instance attribute -- which is the
    same question the rung asks, and a stronger one than comparing
    trajectories, since two wrappers that clamp at a non-binding floor
    would give identical numbers on this toy.
    """
    seen = {"guarded": []}

    class _Watcher(_ToyModelB):
        def mobility(self, rho, T):
            seen["guarded"].append(
                "chemical_potential" in self.__dict__
                or "mobility" in self.__dict__)
            return super().mobility(rho, T)

    model = _Watcher(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat0 = _rho_hat(1, 2, GRID, model.ops)

    a = rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 2,
                              **_roll_kw(clamp_rho=None))
    assert seen["guarded"], "the model was never called"
    assert not any(seen["guarded"]), (
        "clamp_rho=None installed a wrapper: the rollout is no longer "
        "bit-for-bit the unguarded one for a rung that watches that seam")
    assert "chemical_potential" not in model.__dict__
    assert "mobility" not in model.__dict__

    # and with a guard asked for, the same probe says so -- otherwise the
    # negative above could be true of a probe that never sees anything.
    # The floor is a hair below the fixture's own minimum, so it cannot
    # bind: the guard is then arithmetically inert and the two
    # trajectories agree, which is exactly why the probe above, and not a
    # trajectory comparison, is what pins the difference. (0.0 would be
    # the natural inert floor and is refused -- see
    # `test_every_old_spelling_of_no_guard_is_refused_in_both_readings`.)
    N = GRID[0] * GRID[1] * GRID[2]
    inert = float(model.ops.irfft(rho_hat0 * N).min()) / 2.0
    assert inert > 0.0, "the fixture must be strictly positive"
    seen["guarded"].clear()
    b = rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 2,
                              **_roll_kw(clamp_rho=inert))
    assert all(seen["guarded"]), "the guard was asked for and not installed"
    assert "chemical_potential" not in model.__dict__, "and then removed"
    assert torch.equal(a, b)


def test_clamp_rho_evaluates_the_model_at_the_clamped_density(monkeypatch):
    """What the guard is FOR: the learned pointwise methods see a floored
    density, so they are never asked to extrapolate below it.

    The state is untouched -- the guard changes what the model is asked,
    never what the caller reads back -- which is why mass stays exact.
    """
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    # a field with genuinely negative cells, so the floor bites
    g = torch.Generator().manual_seed(17)
    rho = 0.05 + 0.2 * torch.randn(1, 2, *GRID, generator=g)
    assert float(rho.min()) < 0.0, "fixture must go negative"
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat0 = model.ops.rfft(rho) / N
    dc0 = rho_hat0[:, :, 0, 0, 0].clone()

    floor = 1e-3
    seen = {"mu": [], "mob": []}
    mu0, mob0 = model.chemical_potential, model.mobility

    def _mu(r, bx, t):
        seen["mu"].append(float(r.min()))
        return mu0(r, bx, t)

    def _mob(r, t):
        seen["mob"].append(float(r.min()))
        return mob0(r, t)

    model.chemical_potential = _mu
    model.mobility = _mob
    try:
        traj = rollout_deterministic(
            model, rho_hat0, boxes, T, 1e-3, 3,
            **_roll_kw(clamp_rho=floor))
    finally:
        del model.chemical_potential
        del model.mobility

    assert seen["mu"] and seen["mob"], "the model was never called"
    assert min(seen["mu"]) >= floor - 1e-12
    assert min(seen["mob"]) >= floor - 1e-12
    for frame in traj:
        assert torch.equal(frame[:, :, 0, 0, 0], dc0), (
            "an INPUT guard must leave the state's mass untouched")


def test_the_guard_is_removed_when_a_rollout_raises():
    """`clamped_inputs` restores in a `finally:`, and the rollout must not
    lose that by catching around it: a failed run leaves the caller's model
    exactly as it found it.
    """
    class _Exploding(_ToyModelB):
        def mobility(self, rho, T):
            raise RuntimeError("boom")

    model = _Exploding(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho_hat0 = _rho_hat(1, 2, GRID, model.ops)
    before = model.chemical_potential
    with pytest.raises(RuntimeError, match="boom"):
        rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 2,
                              **_roll_kw(clamp_rho=1e-3))
    assert "chemical_potential" not in model.__dict__
    assert model.chemical_potential == before


def test_the_two_projection_targets_give_genuinely_different_rollouts():
    """Non-vacuity: `state_proj` is read, not accepted and discarded.

    A mutation that ignored the argument and always clamped would pass
    every test above that exercises one mode at a time.
    """
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    domain = _domain()
    rho = torch.zeros(1, 2, *GRID)
    rho[:, 0] = 0.9
    rho[:, 1] = 0.5                 # bulk just inside the outer edge
    rho[0, 0, 0, 0, 0] = 1.5        # one cell far outside it
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat0 = model.ops.rfft(rho) / N

    a = rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 3,
                              **_roll_kw(state_proj="floor", floor=1e-3))
    b = rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 3,
                              **_roll_kw(state_proj="domain", floor=1e-3,
                                         domain=domain, lo=_LO))
    assert not torch.equal(a, b)


def test_the_noisy_rollout_projects_after_the_noise_not_before_it():
    """The composition order the published step performs, and the one that
    matters under noise: hermitianise then project, both AFTER the noisy
    update, every step.

    Asserted as an exact identity rather than as a property of the
    returned field, because the alternation does not always ATTAIN the
    floor (three passes of a two-set Dykstra alternation leave a residual,
    which the helper this replaces measures and documents), so "every
    frame is admissible" would be a claim about convergence rather than
    about order. One step, built by hand in each of the two orders: the
    rollout is bit-for-bit one of them and measurably far from the other.
    """
    model = _ToyModelB(GRID, 2)
    boxes, T = _boxes(1), _T(1)
    rho = torch.full((1, 2, *GRID), 0.02)
    N = GRID[0] * GRID[1] * GRID[2]
    rho_hat0 = model.ops.rfft(rho) / N
    floor = 1e-3

    def _seed():
        return torch.Generator().manual_seed(5)

    got = rollout_sde(model, rho_hat0, boxes, T, 1e-3, 1, kBT_noise=5.0,
                      **_sde_kw(), save_every=1, generator=_seed(),
                      **_roll_kw(state_proj="floor", floor=floor))[-1]

    stepped = step_sde_euler_maruyama(
        model, rho_hat0, boxes, T, 1e-3, 5.0, **_sde_kw(),
        generator=_seed())
    after = project_state(hermitianize(stepped, model.ops), model.ops,
                          floor, state_proj="floor")
    before = step_sde_euler_maruyama(
        model,
        project_state(hermitianize(rho_hat0, model.ops), model.ops, floor,
                      state_proj="floor"),
        boxes, T, 1e-3, 5.0, **_sde_kw(), generator=_seed())

    assert torch.equal(got, after), (
        "the rollout is not project-after-step")
    # and the wrong order is not merely a different rounding of the right
    # one: the fixture's noise really does drive cells below the floor, so
    # the two orders land measurably apart.
    raw = model.ops.irfft(stepped * N)
    assert float(raw.min()) < floor, (
        "the fixture's noise must actually drive cells below the floor, "
        "or neither order has anything to do")
    gap = float((after - before).abs().max() / after.abs().max())
    assert gap > 1e-3, gap


# ---------------------------------------------------------------------------
# restore_mass, the two-sided box, and the domain mode's required lower bound
# ---------------------------------------------------------------------------

def _toy_trust_domain():
    """A second toy trapezoid, distinct from `_domain()` so that the tests
    below cannot depend on one set of corners.

    Not any measured system's numbers, for the reason `_domain()` gives:
    `src/aipf/` carries none by construction, and a core test that reached
    for a real system's would re-introduce the literal
    `aipf.solve.trust_domain` records an accident about.
    """
    return TrustDomain(inner=(0.4, 0.3), outer=(1.0, 0.7))


def _toy_state_pushed_to_an_axis_edge(lo=1e-3):
    """`(ops, rho_hat)` for a field whose bulk is comfortably inside
    `_toy_trust_domain()` and whose single violating cell projects onto an
    AXIS edge of it.

    This is the case the `lo` declaration exists for, and it is not the
    generic out-of-domain cell. The trapezoid's vertices are
    `(0.4, 0), (1.0, 0), (0, 0.7), (0, 0.3)`, so its channel-0 = 0 edge is
    the segment from `(0, 0.7)` to `(0, 0.3)`. A cell reaches that edge
    only when its PARTNER channel lies inside that intercept band: here
    channel 0 is driven to -0.02 while channel 1 sits at 0.5, and the
    nearest boundary point is `(0.0, 0.5)` -- a literal zero, in any
    floating-point format, because both endpoints of the edge carry one.
    A cell depleted in BOTH channels would be lifted away from zero by the
    inner edge instead, which is why this fixture is asymmetric.
    """
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    N = GRID[0] * GRID[1] * GRID[2]
    axis = torch.arange(GRID[0], dtype=torch.float32)
    wave = torch.cos(2 * torch.pi * axis / GRID[0]).view(1, 1, -1, 1, 1)
    rho = torch.zeros(1, 2, *GRID)
    rho[:, 0] = 0.4
    rho[:, 1] = 0.2
    rho = rho + 0.01 * wave
    rho[0, 0, 0, 0, 0] = -0.02      # projects onto the channel-0 = 0 edge
    rho[0, 1, 0, 0, 0] = 0.5        # partner inside the intercept band
    assert rho.min() < lo, "the fixture must have something to project"
    return ops, ops.rfft(rho) / N


def test_restore_mass_never_pushes_a_cell_below_lo_when_removing_mass():
    torch.manual_seed(0)
    rho = torch.rand(1, 2, 4, 4, 4) * 0.3 + 0.01
    rho[0, 0, 0, 0, 0] = 1e-3                     # a cell exactly on the edge
    lo = 1e-3
    target = rho.mean(dim=(-3, -2, -1)) - 0.02    # ask to REMOVE mass
    out = restore_mass(rho, target, lo)
    assert torch.allclose(out.mean(dim=(-3, -2, -1)), target, atol=1e-7)
    assert out.min() >= lo - 1e-7, out.min()


def test_restore_mass_adds_uniformly_when_adding():
    rho = torch.full((1, 2, 2, 2, 2), 0.2)
    target = rho.mean(dim=(-3, -2, -1)) + 0.05
    out = restore_mass(rho, target, lo=1e-3)
    assert torch.allclose(out, torch.full_like(rho, 0.25))


def test_restore_mass_is_the_uniform_shift_the_k0_write_is_not():
    """Non-vacuity for the pair above: on a field with an edge cell the two
    restores give DIFFERENT answers, and the difference is exactly the
    defect -- the uniform shift takes the edge cell below `lo`.

    Without this, `restore_mass` could be the plain shift under another
    name and both tests above would still pass on their own fixtures.
    """
    lo = 1e-3
    rho = torch.full((1, 1, 4, 4, 4), 0.3)
    rho[0, 0, 0, 0, 0] = lo
    d = -0.02
    target = rho.mean(dim=(-3, -2, -1)) + d
    uniform = rho + d
    out = restore_mass(rho, target, lo)
    assert float(uniform.min()) < 0.0, (
        "the uniform shift must actually take the edge cell negative, or "
        "there is no defect here to be fixed")
    assert float(out.min()) >= lo - 1e-7
    assert float(out[0, 0, 0, 0, 0]) == pytest.approx(lo, abs=1e-7), (
        "the cell on the edge gave up mass it did not have")
    assert torch.allclose(out.mean(dim=(-3, -2, -1)), target, atol=1e-7)


def test_restore_mass_clamps_rather_than_undershoots_an_impossible_target():
    """`target_mean < lo`: no mass-conserving field satisfies the bound, so
    the removal factor is clamped to a full drain instead of being allowed
    to overshoot into negative densities. The caller's exact k=0 write is
    what keeps mass right in that degenerate case, as before.
    """
    lo = 1e-3
    rho = torch.full((1, 1, 4, 4, 4), 0.3)
    target = torch.full((1, 1), 0.5 * lo)
    out = restore_mass(rho, target, lo)
    assert float(out.min()) >= lo - 1e-7, out.min()
    assert float(out.max()) <= lo + 1e-7, out.max()


def test_domain_projection_without_lo_is_refused_naming_the_reason():
    dom = _toy_trust_domain()
    with pytest.raises(ValueError, match="edges.*axes.*lo="):
        check_state_projection("domain", UNDECLARED, dom, lo=UNDECLARED)


def test_domain_projection_refuses_a_non_positive_lo():
    """Zero is not a declaration of a lower bound here, it is the bound the
    trapezoid already has -- and the one the axis edges attain.
    """
    dom = _toy_trust_domain()
    with pytest.raises(ValueError, match="lo="):
        check_state_projection("domain", UNDECLARED, dom, lo=0.0)


def test_box_requires_both_bounds():
    with pytest.raises(ValueError, match="hi="):
        check_state_projection("box", 1e-3, None, hi=UNDECLARED)


def test_box_requires_lo_as_well():
    with pytest.raises(ValueError, match="lo="):
        check_state_projection("box", 1e-3, None, hi=1.0)


def test_box_projection_enforces_both_bounds_and_keeps_mass_exact():
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    N = GRID[0] * GRID[1] * GRID[2]
    rho = torch.zeros(1, 2, *GRID)
    rho[:, 0] = 0.5
    rho[:, 1] = 0.3
    rho[0, 0, 0, 0, 0] = -0.4       # below lo
    rho[0, 1, 1, 1, 1] = 1.9        # above hi
    rho_hat = ops.rfft(rho) / N
    dc0 = rho_hat[:, :, 0, 0, 0].clone()

    lo, hi = 1e-3, 1.2
    out = project_state(rho_hat, ops, UNDECLARED, state_proj="box",
                        lo=lo, hi=hi)
    rho_out = ops.irfft(out * N)
    assert torch.equal(out[:, :, 0, 0, 0], dc0), "mass must stay exact"
    assert float(rho_out.min()) >= lo - 1e-7, rho_out.min()
    assert float(rho_out.max()) <= hi + 1e-5, rho_out.max()


def test_box_projection_takes_one_bound_per_channel():
    """The bounds are per-channel quantities in every experiment that has
    measured them, so a caller with two numbers is not made to take the
    tighter and lose the other.
    """
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    N = GRID[0] * GRID[1] * GRID[2]
    rho = torch.zeros(1, 2, *GRID)
    rho[:, 0] = 0.5
    rho[:, 1] = 0.3
    rho[0, 0, 0, 0, 0] = 1.9
    rho[0, 1, 1, 1, 1] = 0.9
    rho_hat = ops.rfft(rho) / N

    out = project_state(rho_hat, ops, UNDECLARED, state_proj="box",
                        lo=[1e-3, 1e-3], hi=[1.2, 0.6])
    rho_out = ops.irfft(out * N)
    assert float(rho_out[0, 0].max()) <= 1.2 + 1e-5
    assert float(rho_out[0, 1].max()) <= 0.6 + 1e-5, (
        "the second channel's own upper bound was not applied: a scalar "
        "hi taken from the first channel would pass everything here")


def test_box_projection_is_a_no_op_on_an_already_admissible_field():
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    rho_hat = _rho_hat(1, 2, GRID, ops)
    N = GRID[0] * GRID[1] * GRID[2]
    rho = ops.irfft(rho_hat * N)
    lo = float(rho.min()) - 0.1
    hi = float(rho.max()) + 0.1
    out = project_state(rho_hat, ops, UNDECLARED, state_proj="box",
                        lo=lo, hi=hi)
    assert torch.equal(out, rho_hat)


def test_domain_with_lo_leaves_no_cell_below_lo_after_the_restore():
    ops, rho_hat = _toy_state_pushed_to_an_axis_edge()
    dom = _toy_trust_domain(); lo = 1e-3
    out = project_state(rho_hat, ops, UNDECLARED, state_proj="domain",
                        domain=dom, lo=lo)
    # `* N`: this file's spectra are `rfft(rho) / N`, so the inverse needs
    # the factor back before its numbers are densities. (The brief for this
    # task wrote `ops.irfft(out)`, which compares `rho / N` -- 384x smaller
    # here -- against `lo`, and can only fail.)
    rho = ops.irfft(out * (GRID[0] * GRID[1] * GRID[2]))
    assert rho.min() >= lo - 1e-7, rho.min()


def test_the_axis_edge_fixture_really_does_reach_the_axis():
    """Non-vacuity for the test above: the trapezoid projection ALONE puts
    a literal zero in a channel on this fixture, so "no cell below lo" is a
    statement about the mode and not about a fixture that never went there.
    """
    ops, rho_hat = _toy_state_pushed_to_an_axis_edge()
    dom = _toy_trust_domain()
    N = GRID[0] * GRID[1] * GRID[2]
    rho = ops.irfft(rho_hat * N)
    cells = rho.permute(0, 2, 3, 4, 1).reshape(-1, 2)
    projected = project_trust_domain(cells, dom)
    assert float(projected.min()) == 0.0, (
        f"the fixture's violating cell came back at "
        f"{float(projected.min())}, not on an axis edge")


def test_domain_mass_stays_exact_through_the_bounded_restore():
    """The k=0 write is still LAST and still exact: the restore changes
    WHERE the correction is spent, not whether mass is conserved.
    """
    ops, rho_hat = _toy_state_pushed_to_an_axis_edge()
    dom = _toy_trust_domain()
    dc0 = rho_hat[:, :, 0, 0, 0].clone()
    out = project_state(rho_hat, ops, UNDECLARED, state_proj="domain",
                        domain=dom, lo=1e-3)
    assert torch.equal(out[:, :, 0, 0, 0], dc0)
