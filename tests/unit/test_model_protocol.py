"""Tests for aipf.fields and aipf.functional.base: the model protocol.

The model boundary is k-space in, k-space out:
`forward(rho_hat, boxes, T) -> drho_hat_dt`, never a real-space
tensor at either end. The tests below check that property directly, check
the one structural invariant every rung must honour (mass conservation at
k=0, exactly, not approximately), and check that an incomplete model is
rejected where it is declared -- at registration -- rather than the first
time training calls the method that is missing.

`_ToyModelB` is not a rung. It exists only so this file can exercise a
model that actually satisfies `FreeEnergyModel`, built on the already-tested
k-space operators from `aipf.spectral`, with the simplest possible
physics: mu = rho, M = 1. The point under test is the protocol's shape and
its mass-conservation property, not any particular free energy.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from aipf.spectral import OpsCache

GRID = (6, 5, 8)  # even last axis; matches the grid aipf.spectral tests use


class _ToyModelB(nn.Module):
    """The shared div(M grad mu) skeleton every rung's forward follows.

    Not a physical model -- mu and M are deliberately trivial -- but the
    control flow (irfft rho_hat, evaluate mu and M pointwise, rfft mu back,
    take the k-space gradient, form the flux, rfft it, take the k-space
    divergence) is exactly the shape a real rung's forward takes, and it is
    this shape, not any particular f_loc or kernel term, that makes mass
    conservation at k=0 exact.
    """

    def __init__(self, grid, n_species):
        super().__init__()
        self._cache = OpsCache(grid, n_species, nyquist_mask=True)
        self.ops = self._cache.ops
        self.n_species = n_species

    def chemical_potential(self, rho, boxes, T):
        del boxes, T
        return rho

    def bulk_free_energy_density(self, rho, T):
        del T
        return 0.5 * (rho ** 2).sum(dim=1, keepdim=True)

    def mobility(self, rho, T):
        del T
        return torch.ones_like(rho)

    def forward(self, rho_hat, boxes, T):
        ops = self._cache.ops_for(rho_hat.shape, rho_hat.device)
        N = ops.grid[0] * ops.grid[1] * ops.grid[2]
        rho = ops.irfft(rho_hat * N)
        mu = self.chemical_potential(rho, boxes, T)
        M = self.mobility(rho, T)
        kx, ky, kz = ops.k_axes(boxes)
        mu_hat = ops.rfft(mu) / N
        gx, gy, gz = ops.grad_hat(mu_hat * N, kx, ky, kz)
        grad_mu = torch.stack(
            [ops.irfft(gx), ops.irfft(gy), ops.irfft(gz)], dim=2)
        # Continuity: d(rho)/dt = -div(J) with the Model B flux J = -M grad(mu),
        # so d(rho)/dt = +div(M grad(mu)). Both measured sources agree: one
        # writes `J = einsum(M, grad_mu)` then returns `div_hat(J)`, the other
        # `div_of_scalar_gradient(M, mu)`. Neither carries a minus.
        # Getting this backwards is ANTI-DIFFUSION -- every mode grows and the
        # free energy runs uphill -- and the k=0 test below cannot see it,
        # because the divergence of anything is zero at k=0 either way.
        J = M.unsqueeze(2) * grad_mu
        Jx_hat = ops.rfft(J[:, :, 0]) / N
        Jy_hat = ops.rfft(J[:, :, 1]) / N
        Jz_hat = ops.rfft(J[:, :, 2]) / N
        return ops.div_hat(Jx_hat, Jy_hat, Jz_hat, kx, ky, kz)


class _MissingMobility(nn.Module):
    """A model that implements everything except `mobility`."""

    def forward(self, rho_hat, boxes, T):
        return rho_hat

    def chemical_potential(self, rho, boxes, T):
        return rho

    def bulk_free_energy_density(self, rho, T):
        return rho.sum(dim=1, keepdim=True)


def _boxes(batch, low=8.0, high=12.0):
    g = torch.Generator().manual_seed(0)
    return low + (high - low) * torch.rand(batch, 3, generator=g)


def _rho_hat(batch, n_species, grid, ops):
    g = torch.Generator().manual_seed(1)
    rho = 0.3 + 0.1 * torch.rand(batch, n_species, *grid, generator=g)
    return ops.rfft(rho) / (grid[0] * grid[1] * grid[2])


# ---------------------------------------------------------------------------
# forward is k-space in and out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_forward_is_k_space_in_and_out(n_species):
    model = _ToyModelB(GRID, n_species)
    boxes = _boxes(3)
    T = torch.full((3,), 1.5)
    rho_hat = _rho_hat(3, n_species, GRID, model.ops)

    assert rho_hat.is_complex(), "input to forward must already be k-space"

    drho_hat_dt = model.forward(rho_hat, boxes, T)

    assert drho_hat_dt.is_complex(), "forward must return k-space, not real space"
    assert drho_hat_dt.shape == rho_hat.shape
    assert torch.isfinite(drho_hat_dt.real).all()
    assert torch.isfinite(drho_hat_dt.imag).all()


# ---------------------------------------------------------------------------
# mass conservation: drho_hat/dt at k=0 is exactly zero
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_mass_is_conserved_by_forward(n_species):
    """This is the check that protects every fan-out rung from a wrong
    divergence or a mishandled k=0 row: k=0 mass conservation is a structural
    property of a Model B flux (div of anything, evaluated in k-space, is
    multiplied by k, which is exactly zero at the k=0 mode), not something
    that should require a floating-point tolerance to observe.
    """
    model = _ToyModelB(GRID, n_species)
    boxes = _boxes(4)
    T = torch.full((4,), 2.0)
    rho_hat = _rho_hat(4, n_species, GRID, model.ops)

    drho_hat_dt = model.forward(rho_hat, boxes, T)

    k0 = drho_hat_dt[:, :, 0, 0, 0]
    assert torch.all(k0.real == 0.0)
    assert torch.all(k0.imag == 0.0)


@pytest.mark.parametrize("n_species", [1, 2])
def test_mass_is_conserved_across_different_boxes_in_one_batch(n_species):
    """The k=0 row is exactly zero per sample, not only on average across a
    batch that happens to share one box -- boxes are per-sample.
    """
    model = _ToyModelB(GRID, n_species)
    boxes = torch.tensor([[8.0, 8.0, 8.0], [11.3, 9.7, 10.1]])
    T = torch.tensor([1.5, 3.0])
    rho_hat = _rho_hat(2, n_species, GRID, model.ops)

    drho_hat_dt = model.forward(rho_hat, boxes, T)
    k0 = drho_hat_dt[:, :, 0, 0, 0]
    assert torch.all(k0 == 0)


# ---------------------------------------------------------------------------
# registration rejects an incomplete model
# ---------------------------------------------------------------------------

def test_a_model_missing_a_protocol_method_is_rejected_at_registration():
    from aipf.functional.base import ModelRegistry

    registry = ModelRegistry()
    with pytest.raises(TypeError, match="mobility"):
        registry.register("missing_mobility", _MissingMobility)
    assert "missing_mobility" not in registry


def test_a_complete_model_registers_and_builds_by_name():
    from aipf.functional.base import ModelRegistry

    registry = ModelRegistry()
    registered = registry.register("toy", _ToyModelB)
    assert registered is _ToyModelB
    assert "toy" in registry

    model = registry.build("toy", grid=GRID, n_species=2)
    assert isinstance(model, _ToyModelB)


def test_building_an_unregistered_name_raises():
    from aipf.functional.base import ModelRegistry

    registry = ModelRegistry()
    with pytest.raises(KeyError):
        registry.build("nonexistent")


def test_check_protocol_names_every_missing_method():
    from aipf.functional.base import check_protocol

    class _EmptyModel:
        pass

    with pytest.raises(TypeError) as excinfo:
        check_protocol(_EmptyModel)
    message = str(excinfo.value)
    for name in ("forward", "chemical_potential",
                 "bulk_free_energy_density", "mobility"):
        assert name in message


def test_check_protocol_rejects_a_non_callable_attribute():
    """A class attribute that merely shares a protocol method's name, but is
    not callable, is not an implementation of it.
    """
    from aipf.functional.base import check_protocol

    class _MobilityIsNotAMethod:
        forward = None
        chemical_potential = None
        bulk_free_energy_density = None
        mobility = None  # a class attribute, not a method

    with pytest.raises(TypeError, match="mobility"):
        check_protocol(_MobilityIsNotAMethod)


def test_registry_names_are_sorted():
    from aipf.functional.base import ModelRegistry

    registry = ModelRegistry()
    registry.register("zzz", _ToyModelB)
    registry.register("aaa", _ToyModelB)
    assert registry.names == ("aaa", "zzz")


def test_a_conforming_model_passes_runtime_checkable_isinstance():
    """FreeEnergyModel is declared `@runtime_checkable` so a caller can also
    check an instance directly, not only a class at registration time.
    """
    from aipf.functional.base import FreeEnergyModel

    model = _ToyModelB(GRID, 1)
    assert isinstance(model, FreeEnergyModel)
    assert not isinstance(_MissingMobility(), FreeEnergyModel)


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_field_carries_rho_hat_boxes_and_grid(n_species):
    from aipf.fields import Field

    ops = OpsCache(GRID, n_species, nyquist_mask=True).ops
    boxes = _boxes(3)
    rho_hat = _rho_hat(3, n_species, GRID, ops)

    f = Field(rho_hat=rho_hat, boxes=boxes, grid=GRID)

    assert f.rho_hat is rho_hat
    assert f.boxes is boxes
    assert f.grid == GRID
    assert f.n_species == n_species
    assert f.batch_size == 3


def test_field_rejects_a_rho_hat_missing_the_species_axis():
    from aipf.fields import Field

    ops = OpsCache(GRID, 1, nyquist_mask=True).ops
    boxes = _boxes(2)
    # (B, Gx, Gy, Gzr): 4D, no species axis.
    rho_hat = _rho_hat(2, 1, GRID, ops)[:, 0]

    with pytest.raises(ValueError):
        Field(rho_hat=rho_hat, boxes=boxes, grid=GRID)


def test_field_rejects_boxes_with_the_wrong_number_of_axes():
    from aipf.fields import Field

    ops = OpsCache(GRID, 1, nyquist_mask=True).ops
    rho_hat = _rho_hat(2, 1, GRID, ops)
    # (B, 2) instead of (B, 3): missing an axis.
    boxes = _boxes(2)[:, :2]

    with pytest.raises(ValueError):
        Field(rho_hat=rho_hat, boxes=boxes, grid=GRID)


def test_field_rejects_a_grid_that_does_not_match_rho_hat():
    from aipf.fields import Field

    ops = OpsCache(GRID, 1, nyquist_mask=True).ops
    boxes = _boxes(2)
    rho_hat = _rho_hat(2, 1, GRID, ops)

    with pytest.raises(ValueError):
        Field(rho_hat=rho_hat, boxes=boxes, grid=(6, 5, 6))


def test_field_rejects_a_real_valued_rho_hat():
    """Real-valued but with the correct half-spectrum SHAPE, so this isolates
    the dtype check from the trailing-shape check (a real tensor shaped like
    the full real-space grid would be rejected by the shape check instead,
    and would not exercise the dtype branch at all).
    """
    from aipf.fields import Field

    Gx, Gy, Gz = GRID
    boxes = _boxes(2)
    real_field = torch.rand(2, 1, Gx, Gy, Gz // 2 + 1)

    with pytest.raises(ValueError):
        Field(rho_hat=real_field, boxes=boxes, grid=GRID)


def test_field_rejects_mismatched_batch_between_rho_hat_and_boxes():
    from aipf.fields import Field

    ops = OpsCache(GRID, 1, nyquist_mask=True).ops
    rho_hat = _rho_hat(3, 1, GRID, ops)
    boxes = _boxes(2)

    with pytest.raises(ValueError):
        Field(rho_hat=rho_hat, boxes=boxes, grid=GRID)


def test_field_to_moves_tensors_and_keeps_grid():
    from aipf.fields import Field

    ops = OpsCache(GRID, 1, nyquist_mask=True).ops
    rho_hat = _rho_hat(2, 1, GRID, ops)
    boxes = _boxes(2)
    f = Field(rho_hat=rho_hat, boxes=boxes, grid=GRID)

    moved = f.to(device="cpu")
    assert moved.boxes.device == torch.device("cpu")
    assert moved.rho_hat.device == torch.device("cpu")
    assert moved.grid == GRID
    assert moved is not f


def test_field_to_forwards_its_arguments_to_both_tensors():
    """`to` forwards the same call to `rho_hat` and to `boxes`; this isolates
    each tensor moving independently by checking a dtype change lands on
    both, so a `to` that forgets one tensor (silently leaving it as the
    original object) is caught even though this environment has no second
    device to move to.
    """
    from aipf.fields import Field

    ops = OpsCache(GRID, 1, nyquist_mask=True).ops
    rho_hat = _rho_hat(2, 1, GRID, ops)
    boxes = _boxes(2)
    f = Field(rho_hat=rho_hat, boxes=boxes, grid=GRID)

    moved = f.to(torch.complex128)
    assert moved.rho_hat.dtype == torch.complex128
    assert moved.boxes.dtype == torch.complex128


# ---------------------------------------------------------------------------
# the sign of the flux: a convex free energy must DECAY
# ---------------------------------------------------------------------------

def test_a_convex_free_energy_decays_rather_than_grows():
    """The guard the k=0 mass test cannot provide.

    Mass conservation holds for BOTH signs of the flux, because the divergence
    of anything is exactly zero at k=0. So a model that runs free energy uphill
    passes every structural check while being physically inverted.

    Linearised, for a convex ``f`` and constant ``M`` a single Fourier mode
    obeys ``d(rho_k)/dt = -M k^2 f'' rho_k``, which is strictly negative. This
    test pins the sign, and it is the one that catches an inverted flux.

    It was added after a real defect: this file's own reference skeleton
    returned ``div(-M grad mu)``, one rung copied it, and 618 tests passed with
    a model whose every mode grew at +5.1 per unit time.
    """
    grid = (8, 8, 8)
    length = 4.0
    model = _ToyModelB(grid, n_species=1)
    ops = model.ops
    boxes = torch.tensor([[length, length, length]])

    axis = torch.arange(grid[0], dtype=torch.float32) * (length / grid[0])
    rho = 0.5 + 0.01 * torch.cos(2 * torch.pi * axis / length).view(1, 1, -1, 1, 1)
    rho = rho.expand(1, 1, *grid).contiguous()
    rho_hat = ops.rfft(rho) / (grid[0] * grid[1] * grid[2])

    drho_hat_dt = model.forward(rho_hat, boxes, torch.tensor([1.0]))

    amplitude = rho_hat[0, 0, 1, 0, 0]
    rate = (drho_hat_dt[0, 0, 1, 0, 0] / amplitude).real
    assert rate < 0, (
        f"a convex free energy must decay, got d(ln rho_k)/dt = {rate:+.4e}. "
        f"A positive rate is anti-diffusion: the flux sign is inverted.")
