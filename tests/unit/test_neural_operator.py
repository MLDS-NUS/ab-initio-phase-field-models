"""Tests for aipf.functional.neural_operator: the equivariant neural operator.

The rung is implemented and equivariance-tested, not trained. The deliverable this file exists to carry is the equivariance test
itself -- a symmetry claim, not an accuracy claim -- plus species
permutation equivariance, exact mass conservation at k=0, and the model
protocol check every rung owes the rest of the package.

**Rotation, precisely.** A continuous rotation does not preserve a discrete
periodic grid, so "rotate" here means an element of the grid's own discrete
symmetry group: relabelling which of the three (equal-length, since the
grid used below is a cube) spatial axes is which. This is only a genuine
physical symmetry when the box is isotropic too (`Lx == Ly == Lz`), the same
way a physical rotation is only a symmetry of a box that is itself cubic --
so every equivariance test below uses an isotropic box. Point reflections
(axis flips) are deliberately left out: the correct discrete flip on a
periodic grid is index `i -> (-i) mod N`, which is `roll(flip(x), 1)`, not
a bare `torch.flip`, and getting that index convention wrong would make a
test that looks like it is checking reflection equivariance actually check
something else. Axis permutation and integer translation (`torch.roll`) are
unambiguous, exact discrete symmetries and are what is exercised here; see
the task report for this scope decision.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from aipf.functional.base import FreeEnergyModel, ModelRegistry, check_protocol
from aipf.functional.neural_operator import NeuralOperator, MODEL_REGISTRY

GRID = (6, 6, 6)  # cubic, even: lets axis permutation preserve shape exactly


def _boxes_isotropic(batch: int, L: float = 9.0) -> torch.Tensor:
    """Per-sample cubic boxes, all axes equal -- required for an
    axis-permutation of the grid to be a genuine physical symmetry (see
    module docstring)."""
    return torch.full((batch, 3), float(L))


def _boxes_anisotropic(batch: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    return 8.0 + 4.0 * torch.rand(batch, 3, generator=g)


def _rho(batch: int, n_species: int, grid=GRID, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 0.3 + 0.1 * torch.rand(batch, n_species, *grid, generator=g)


def _T(batch: int, value: float = 1.7) -> torch.Tensor:
    return torch.full((batch,), float(value))


def _model(n_species: int, grid=GRID, n_layers: int = 2, hidden: int = 4,
           seed: int = 0) -> NeuralOperator:
    torch.manual_seed(seed)
    return NeuralOperator(grid, n_species, n_layers=n_layers, hidden=hidden, nyquist_mask=True)


def _grid_transform(x: torch.Tensor, perm, shifts) -> torch.Tensor:
    """Apply an axis permutation, then a per-axis integer roll, to the three
    trailing spatial dims of a `(B, n_species, Gx, Gy, Gz)` real tensor.
    """
    x = x.permute(0, 1, 2 + perm[0], 2 + perm[1], 2 + perm[2])
    return torch.roll(x, shifts=shifts, dims=(-3, -2, -1))


# ---------------------------------------------------------------------------
# The equivariance test IS the deliverable.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_chemical_potential_is_equivariant_to_rotation_and_translation(
        n_species, perm):
    """`mu(g.rho) == g.mu(rho)` for `g` a random grid-symmetry rotation
    (an axis permutation) composed with a random integer translation, for
    every one of the six axis permutations -- this is the equivariance
    claim rung 4 exists to make, checked directly on the method whose
    output is the layer stack's output field.
    """
    model = _model(n_species)
    B = 2
    boxes = _boxes_isotropic(B)
    T = _T(B)
    rho = _rho(B, n_species)
    shifts = (2, 5, 1)

    mu = model.chemical_potential(rho, boxes, T)

    rho_t = _grid_transform(rho, perm, shifts)
    mu_t = model.chemical_potential(rho_t, boxes, T)
    mu_expected = _grid_transform(mu, perm, shifts)

    torch.testing.assert_close(mu_t, mu_expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("n_species", [1, 2])
def test_forward_is_equivariant_to_rotation_and_translation(n_species):
    """The same claim end to end: `forward` (k-space in, k-space out,
    routed through the mobility contraction and the divergence) transforms
    the same way `chemical_potential` alone does.
    """
    model = _model(n_species)
    B = 2
    boxes = _boxes_isotropic(B)
    T = _T(B)
    rho = _rho(B, n_species)
    rho_hat = model.ops.rfft(rho)
    perm, shifts = (2, 0, 1), (1, 4, 2)

    drho_dt = model.ops.irfft(model.forward(rho_hat, boxes, T))

    rho_t = _grid_transform(rho, perm, shifts)
    rho_hat_t = model.ops.rfft(rho_t)
    drho_dt_t = model.ops.irfft(model.forward(rho_hat_t, boxes, T))
    expected = _grid_transform(drho_dt, perm, shifts)

    torch.testing.assert_close(drho_dt_t, expected, atol=1e-5, rtol=1e-4)


def test_equivariance_requires_an_isotropic_box():
    """Negative control on the test's own setup, not on the model: with an
    ANISOTROPIC box, axis-permuting the grid is not a physical symmetry of
    that box, so agreement is not expected. This pins down why every
    equivariance test above insists on `_boxes_isotropic`, rather than that
    requirement being an unexplained, easy-to-drop detail.
    """
    model = _model(2)
    B = 2
    boxes = _boxes_anisotropic(B)
    T = _T(B)
    rho = _rho(B, 2)
    perm, shifts = (1, 2, 0), (1, 1, 1)

    mu = model.chemical_potential(rho, boxes, T)
    rho_t = _grid_transform(rho, perm, shifts)
    mu_t = model.chemical_potential(rho_t, boxes, T)
    mu_expected = _grid_transform(mu, perm, shifts)

    # An isotropic-box run agrees to ~1e-8 (floating-point roundoff only);
    # an anisotropic box measurably breaks the axis-permutation symmetry,
    # three to four orders of magnitude above that floor.
    assert (mu_t - mu_expected).abs().max().item() > 5e-5


def test_equivariance_check_catches_a_position_dependent_bias():
    """Watch the equivariance check fail first, kept as a permanent,
    self-contained regression test rather than only a manual step: a layer
    built exactly like `NeuralOperator`'s stack, except its bias is a
    function of grid position instead of a single scalar, must FAIL the
    same rotation+translation check above. This is the mutation the task
    brief names explicitly, and it is not applied to the production module
    at all -- it is a small local stand-in, so this test cannot pass
    vacuously by accident of import caching.
    """
    import torch.nn as nn
    from aipf.spectral import OpsCache

    class _BrokenLayer(nn.Module):
        """Same radial gain and channel mix as the real layer; the bias
        is `arange(Gz) * 0.1`, i.e. it distinguishes grid points from one
        another, which nothing in an equivariant layer may do.
        """

        def __init__(self, hidden=4):
            super().__init__()
            self._radial = nn.Sequential(
                nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.mix_diag = nn.Parameter(torch.tensor(1.0))
            self.mix_off = nn.Parameter(torch.tensor(0.0))

        def forward(self, v_hat, k_mag, T, ops):
            t = T.reshape(-1, 1, 1, 1, 1).expand_as(k_mag)
            gain = self._radial(torch.stack((k_mag, t), dim=-1))
            gain = gain.squeeze(-1).to(v_hat.dtype)
            z = ops.irfft(gain * v_hat)
            total = z.sum(dim=1, keepdim=True)
            mixed = self.mix_diag * z + self.mix_off * (total - z)
            pos_bias = torch.arange(z.shape[-1], dtype=z.dtype) * 0.1
            out = torch.tanh(mixed + pos_bias)
            return ops.rfft(out)

    torch.manual_seed(0)
    cache = OpsCache(GRID, 2, nyquist_mask=True)
    layer = _BrokenLayer()
    B = 2
    boxes = _boxes_isotropic(B)
    T = _T(B)
    rho = _rho(B, 2)
    perm, shifts = (2, 0, 1), (1, 4, 2)

    def mu_of(rho_in):
        k_mag = torch.sqrt(torch.clamp(cache.ops.k2(boxes), min=0.0))
        v_hat = layer(cache.ops.rfft(rho_in), k_mag, T, cache.ops)
        return cache.ops.irfft(v_hat)

    mu = mu_of(rho)
    rho_t = _grid_transform(rho, perm, shifts)
    mu_t = mu_of(rho_t)
    mu_expected = _grid_transform(mu, perm, shifts)

    err = (mu_t - mu_expected).abs().max().item()
    assert err > 1e-3, (
        "the position-dependent-bias layer should NOT pass the "
        f"equivariance check, but its error was only {err}"
    )


# ---------------------------------------------------------------------------
# Permutation equivariance over species, n_species = 2
# ---------------------------------------------------------------------------

def test_chemical_potential_is_equivariant_to_species_permutation():
    model = _model(2)
    B = 2
    boxes = _boxes_isotropic(B)
    T = _T(B)
    rho = _rho(B, 2)

    mu = model.chemical_potential(rho, boxes, T)
    mu_swapped = model.chemical_potential(rho[:, [1, 0]], boxes, T)

    torch.testing.assert_close(mu_swapped, mu[:, [1, 0]], atol=1e-6, rtol=1e-5)


def test_mobility_is_equivariant_to_species_permutation():
    model = _model(2)
    B = 2
    T = _T(B)
    rho = _rho(B, 2)

    M = model.mobility(rho, T)
    M_swapped = model.mobility(rho[:, [1, 0]], T)

    expected = M[:, [1, 0]][:, :, [1, 0]]
    torch.testing.assert_close(M_swapped, expected, atol=1e-7, rtol=1e-6)


def test_forward_is_equivariant_to_species_permutation():
    model = _model(2)
    B = 2
    boxes = _boxes_isotropic(B)
    T = _T(B)
    rho = _rho(B, 2)
    rho_hat = model.ops.rfft(rho)
    rho_hat_swapped = model.ops.rfft(rho[:, [1, 0]])

    out = model.forward(rho_hat, boxes, T)
    out_swapped = model.forward(rho_hat_swapped, boxes, T)

    torch.testing.assert_close(out_swapped, out[:, [1, 0]], atol=1e-6, rtol=1e-5)


def test_species_permutation_at_one_species_is_the_identity():
    """At `n_species = 1` there is nothing to permute; the mix's
    off-diagonal term must have no effect on the output regardless of its
    value, which is the concrete, testable form that degenerate case takes.
    """
    model = _model(1)
    for layer in model.layers:
        layer.mix_off.data.fill_(37.0)
    B = 2
    boxes = _boxes_isotropic(B)
    T = _T(B)
    rho = _rho(B, 1)

    mu = model.chemical_potential(rho, boxes, T)
    for layer in model.layers:
        layer.mix_off.data.fill_(-11.0)
    mu_again = model.chemical_potential(rho, boxes, T)

    torch.testing.assert_close(mu, mu_again, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Mass conservation: drho_hat/dt at k=0 is exactly zero, every rung's rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_mass_is_conserved_exactly_at_k0(n_species):
    model = _model(n_species)
    B = 3
    boxes = _boxes_anisotropic(B)
    T = _T(B, 2.1)
    rho = _rho(B, n_species, seed=5)
    rho_hat = model.ops.rfft(rho)

    drho_hat_dt = model.forward(rho_hat, boxes, T)

    k0 = drho_hat_dt[:, :, 0, 0, 0]
    assert torch.all(k0.real == 0.0)
    assert torch.all(k0.imag == 0.0)


@pytest.mark.parametrize("n_species", [1, 2])
def test_mass_is_conserved_across_different_boxes_in_one_batch(n_species):
    """Boxes are per-sample; a k=0 check that happens to pass only
    because a whole batch shares one box would miss a `boxes[0]`-for-the-
    batch bug.
    """
    model = _model(n_species)
    boxes = torch.tensor([[8.0, 8.0, 8.0], [11.3, 9.7, 10.1]])
    T = torch.tensor([1.5, 3.0])
    rho = _rho(2, n_species, seed=6)
    rho_hat = model.ops.rfft(rho)

    drho_hat_dt = model.forward(rho_hat, boxes, T)
    k0 = drho_hat_dt[:, :, 0, 0, 0]
    assert torch.all(k0 == 0)


# ---------------------------------------------------------------------------
# The sign of the flux: a convex free energy must DECAY, not grow.
#
# Real defect, found and fixed during this task (see the task report): the
# `_ToyModelB` reference skeleton this rung's `forward` was built against
# (tests/unit/test_model_protocol.py) returned `div_hat(J)` with
# `J = -M grad(mu)`, i.e. `d(rho)/dt = -div(M grad(mu))` -- anti-diffusion.
# Both measured source trees return `div(M grad(mu))` with no leading minus.
# Mass conservation at k=0 cannot catch this: the divergence of anything is
# exactly zero at k=0 regardless of the flux's overall sign. This is the
# check that does catch it, ported to this rung's own architecture.
# ---------------------------------------------------------------------------

def test_a_convex_free_energy_decays_rather_than_grows():
    """Rung 4 has no explicit scalar free energy of its own to linearise
    (`bulk_free_energy_density` is honestly zero; `chemical_potential` is
    the direct, generally nonlinear output of the layer stack, not an
    autograd derivative of anything -- see the module docstring). The
    controlled case this test builds is the same one
    `test_model_protocol.py::test_a_convex_free_energy_decays_rather_than_grows`
    checks against `_ToyModelB`: `mu(rho) = rho` is exactly `f'(rho)` for
    the convex `f(rho) = rho^2 / 2`, and a constant, positive `M` -- so
    `chemical_potential` and `mobility` are monkeypatched to that pair on an
    otherwise-real `NeuralOperator`, isolating the sign of `forward`'s
    flux/divergence assembly from the (untrained, not under test here)
    layer stack and mobility gate.

    Linearised, for convex `f` and constant `M`, a single Fourier mode obeys
    `d(rho_k)/dt = -M k^2 f'' rho_k`, strictly negative for `f'' = 1 > 0`
    here -- growth (`d(ln rho_k)/dt > 0`) means the flux sign is inverted.
    """
    grid = (8, 8, 8)
    length = 4.0
    model = NeuralOperator(grid, 1, n_layers=1, hidden=2, nyquist_mask=True)
    model._mu_field = lambda rho, boxes, T, ops: rho
    model.mobility = lambda rho, T: 0.7 * torch.ones(
        rho.shape[0], 1, 1, *rho.shape[2:])

    boxes = torch.tensor([[length, length, length]])
    T = torch.tensor([1.0])
    axis = torch.arange(grid[0], dtype=torch.float32) * (length / grid[0])
    rho = 0.5 + 0.01 * torch.cos(2 * torch.pi * axis / length).view(1, 1, -1, 1, 1)
    rho = rho.expand(1, 1, *grid).contiguous()
    rho_hat = model.ops.rfft(rho)

    drho_hat_dt = model.forward(rho_hat, boxes, T)

    amplitude = rho_hat[0, 0, 1, 0, 0]
    rate = (drho_hat_dt[0, 0, 1, 0, 0] / amplitude).real
    assert rate < 0, (
        f"a convex free energy must decay, got d(ln rho_k)/dt = {rate:+.4e}. "
        f"A positive rate is anti-diffusion: the flux sign is inverted.")


# ---------------------------------------------------------------------------
# The model protocol
# ---------------------------------------------------------------------------

def test_check_protocol_accepts_the_equivariant_operator():
    check_protocol(NeuralOperator)  # must not raise


def test_every_layer_in_the_stack_affects_the_output():
    """The layer loop in `_mu_field` must actually thread through every
    layer, not silently drop a prefix or suffix of `self.layers` -- a bug
    equivariance and permutation tests cannot see, since a subset of
    equivariant layers is itself still equivariant. Perturbing each
    layer's bias in turn and requiring the output to move is the direct,
    structural check for that.
    """
    model = _model(2, n_layers=3, hidden=4)
    B = 2
    boxes = _boxes_isotropic(B)
    T = _T(B)
    rho = _rho(B, 2)

    base = model.chemical_potential(rho, boxes, T).clone()
    for layer in model.layers:
        original = layer.bias.data.clone()
        layer.bias.data.add_(5.0)
        moved = model.chemical_potential(rho, boxes, T)
        layer.bias.data.copy_(original)
        assert (moved - base).abs().max().item() > 1e-4


def test_equivariant_operator_passes_runtime_checkable_isinstance():
    model = _model(2)
    assert isinstance(model, FreeEnergyModel)


def test_module_level_registry_holds_the_form_name():
    assert "neural_operator" in MODEL_REGISTRY
    # Containment, not equality: the registry is shared across all rungs,
    # so its exact contents depend on which modules the test session has
    # imported -- an ordering dependency, not a property of this rung.
    assert "neural_operator" in set(MODEL_REGISTRY.names)
def test_a_fresh_registry_registers_and_builds_the_model():
    registry = ModelRegistry()
    registry.register("neural_operator", NeuralOperator)
    model = registry.build("neural_operator", grid=GRID, n_species=2,
                           nyquist_mask=True)
    assert isinstance(model, NeuralOperator)


# ---------------------------------------------------------------------------
# Shapes, dtypes, and the honest-zero local term
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_forward_is_k_space_in_and_out(n_species):
    model = _model(n_species)
    B = 2
    boxes = _boxes_anisotropic(B)
    T = _T(B)
    rho = _rho(B, n_species)
    rho_hat = model.ops.rfft(rho)

    assert rho_hat.is_complex()
    drho_hat_dt = model.forward(rho_hat, boxes, T)

    assert drho_hat_dt.is_complex()
    assert drho_hat_dt.shape == rho_hat.shape
    assert torch.isfinite(drho_hat_dt.real).all()
    assert torch.isfinite(drho_hat_dt.imag).all()


@pytest.mark.parametrize("n_species", [1, 2])
def test_chemical_potential_matches_rho_shape(n_species):
    model = _model(n_species)
    B = 2
    boxes = _boxes_anisotropic(B)
    T = _T(B)
    rho = _rho(B, n_species)

    mu = model.chemical_potential(rho, boxes, T)
    assert mu.shape == rho.shape
    assert torch.isfinite(mu).all()


@pytest.mark.parametrize("n_species", [1, 2])
def test_bulk_free_energy_density_is_exactly_zero(n_species):
    """Rung 4 has no term evaluable from a single grid point (module
    docstring): every layer needs the spatial radial convolution. Zero is
    the honest value, not a placeholder that happens to be zero.
    """
    model = _model(n_species)
    B = 2
    T = _T(B)
    rho = _rho(B, n_species)

    f = model.bulk_free_energy_density(rho, T)
    assert f.shape == (B, 1, *GRID)
    assert torch.all(f == 0.0)


@pytest.mark.parametrize("n_species", [1, 2, 3])
def test_mobility_is_symmetric_positive_semidefinite_for_arbitrary_parameters(
        n_species):
    """`m_diag`/`m_off` are constant-initialised (`0.0`), not seed-dependent,
    so a sweep that only varies the construction seed never actually
    exercises them -- confirmed by mutation-testing this exact test (see
    the task report): dropping the PSD-preserving bound on `m_off` survived
    a seed-only sweep because `torch.tanh(0.0) == 0.0 == the unmutated
    default`. The parameter values are therefore swept explicitly here,
    including large-magnitude and negative ones, rather than left to the
    model's default initialisation.
    """
    B = 2
    T = _T(B)
    rho = _rho(B, n_species, grid=GRID, seed=9)
    model = _model(n_species, n_layers=1, hidden=3)

    for m_diag, m_off in [(0.0, 0.0), (2.0, 5.0), (2.0, -5.0),
                           (-3.0, 0.7), (0.01, 100.0), (0.01, -100.0)]:
        model.m_diag.data.fill_(m_diag)
        model.m_off.data.fill_(m_off)
        M = model.mobility(rho, T)
        assert M.shape == (B, n_species, n_species) + GRID
        Mpt = M[0, :, :, 2, 3, 1]
        torch.testing.assert_close(Mpt, Mpt.transpose(0, 1), atol=1e-6, rtol=1e-6)
        eig = torch.linalg.eigvalsh(Mpt.detach())
        assert torch.all(eig >= -1e-6), (m_diag, m_off, eig)


def test_mobility_at_one_species_reduces_to_a_nonnegative_scalar():
    model = _model(1)
    B = 2
    T = _T(B)
    rho = _rho(B, 1)
    M = model.mobility(rho, T)
    assert M.shape == (B, 1, 1, *GRID)
    assert torch.all(M >= 0.0)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

def test_n_species_must_be_at_least_one():
    with pytest.raises(ValueError):
        NeuralOperator(GRID, 0, nyquist_mask=True)


def test_n_layers_must_be_at_least_one():
    with pytest.raises(ValueError):
        NeuralOperator(GRID, 2, n_layers=0, nyquist_mask=True)


# ---------------------------------------------------------------------------
# Grid generality: forward resolves ops for the shape it is actually called
# with, not only the shape the module was constructed at (aipf.spectral's
# OpsCache.ops_for, exercised through this rung rather than re-testing
# OpsCache itself).
# ---------------------------------------------------------------------------

def test_forward_works_at_a_grid_other_than_construction():
    model = _model(2, grid=GRID)
    other_grid = (8, 8, 8)
    B = 2
    boxes = _boxes_isotropic(B, L=10.0)
    T = _T(B)
    rho = _rho(B, 2, grid=other_grid, seed=12)
    from aipf.spectral import SpectralOps
    other_ops = SpectralOps(other_grid, 2, nyquist_mask=True)
    rho_hat = other_ops.rfft(rho)

    drho_hat_dt = model.forward(rho_hat, boxes, T)

    assert drho_hat_dt.shape == rho_hat.shape
    k0 = drho_hat_dt[:, :, 0, 0, 0]
    assert torch.all(k0 == 0)
