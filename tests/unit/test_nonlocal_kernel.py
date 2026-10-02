"""Tests for aipf.functional.nonlocal_kernel: the trained rung's assembly.

Rung 3 is the trained rung -- the model the published checkpoints
hold. This file exists to pin the three things that are the ways this assembly, specifically, can break without any test
noticing:

1. A silent channel-convention transpose across the three different shapes
   `PairKernel.w_hat`, `Mobility.forward` and the rung protocol each use.
   Caught only by a fixture whose per-species values are DISTINGUISHABLE --
   a symmetric fixture cannot detect a channel swap (see
   `test_mobility_does_not_mix_species` and
   `test_kernel_term_does_not_mix_species`).
2. An inverted flux sign. Mass conservation at k=0 holds for BOTH signs of
   the flux (the divergence of anything is zero at k=0 regardless of sign),
   so a dedicated dispersion test is the only thing that catches this (see
   `test_a_convex_free_energy_decays_rather_than_grows`).
3. A construction order that draws the T-basis heads before the kernel or
   mobility, which would make a flags-on model's kernel/mobility weights
   silently diverge from a flags-off model's at the same seed -- and would
   make every existing trained checkpoint irreproducible (see
   `test_kernel_and_mobility_weights_are_identical_whether_or_not_t_heads_are_on`).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from aipf.functional.base import check_protocol
from aipf.functional.nonlocal_kernel import MODEL_REGISTRY, NonlocalKernel

GRID = (6, 5, 8)  # even last axis; Gzr = 5


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _boxes(batch: int, grid=GRID) -> torch.Tensor:
    L = torch.tensor(grid, dtype=torch.float32)
    return L.unsqueeze(0).expand(batch, 3).clone()


def _T(batch: int, value: float = 1.7) -> torch.Tensor:
    return torch.full((batch,), float(value))


def _model(n_species: int, grid=GRID, seed: int = 0,
           **overrides) -> NonlocalKernel:
    """A small, fast-to-construct model. Every argument below is one this
    package ships no default for (System fields); the values here are
    arbitrary test fixtures, not measurements of any real system.
    """
    torch.manual_seed(seed)
    kwargs = dict(
        kB=1.0,
        rho_ref=[0.5] * n_species,
        kBT_ref=1.0,
        h_u=4,
        h_g=4,
        R_cut=3.0,
        kernel_n_quad=17,
        kernel_n_k_table=17,
        kernel_k_table_max=8.0,
        kernel_hidden=4,
        mobility_prefactor="mole_fraction",
        mobility_shape="mlp_rho",
        mobility_t_form="none",
        mobility_hidden=4,
        # Required, not defaulted: see local_forms.IDEAL_FORMS. The first
        # system, whose parity this rung must reproduce, is the ideal-gas one.
        ideal_form="gas",
        nyquist_mask=True,
    )
    kwargs.update(overrides)
    return NonlocalKernel(grid, n_species, **kwargs)


def _rho_hat(batch: int, n_species: int, grid, ops, seed: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rho = 0.3 + 0.1 * torch.rand(batch, n_species, *grid, generator=g)
    N = grid[0] * grid[1] * grid[2]
    return ops.rfft(rho) / N


# ---------------------------------------------------------------------------
# the model protocol, at n_species 1 and 2
# ---------------------------------------------------------------------------

def test_check_protocol_accepts_the_rung():
    check_protocol(NonlocalKernel)  # must not raise


def test_registered_into_the_shared_registry():
    assert "nonlocal_kernel" in MODEL_REGISTRY


@pytest.mark.parametrize("n_species", [1, 2])
def test_forward_is_complex_finite_and_shape_preserving(n_species):
    model = _model(n_species)
    boxes = _boxes(3)
    T = _T(3)
    rho_hat = _rho_hat(3, n_species, GRID, model.ops)

    assert rho_hat.is_complex()
    drho_hat_dt = model.forward(rho_hat, boxes, T)

    assert drho_hat_dt.is_complex()
    assert drho_hat_dt.shape == rho_hat.shape
    assert torch.isfinite(drho_hat_dt.real).all()
    assert torch.isfinite(drho_hat_dt.imag).all()


def test_forward_passes_the_correctly_scaled_rho_downstream():
    """`forward`'s own state convention is `rho = irfft(rho_hat * N_GRID)`
    (the "mode grid" convention the ported source and every other rung in
    this package share), not the bare `irfft(rho_hat)` a standard-normalised
    FFT pair would give. None of the other tests in this file exercise this
    specific line: the distinguishable-species tests call
    `chemical_potential`/`mobility` directly with a hand-built `rho`, never
    through `forward`'s own conversion, and the dispersion test's rate is
    scale-invariant (an overall dilation of `rho` cancels in the ratio
    `drho_hat_dt / rho_hat`) -- so a missing `* N_GRID` here would pass
    every other test in this file. Confirmed by mutation testing: this is a
    real coverage gap this test was added to close, not a redundant check.
    """
    model = _model(2)
    boxes = _boxes(2)
    T = _T(2)
    rho_hat = _rho_hat(2, 2, GRID, model.ops)
    N = GRID[0] * GRID[1] * GRID[2]
    expected_rho = model.ops.irfft(rho_hat * N)

    seen = {}
    real_chemical_potential = model.chemical_potential

    def spy(rho, boxes, T):
        seen["rho"] = rho.clone()
        return real_chemical_potential(rho, boxes, T)

    model.chemical_potential = spy
    model.forward(rho_hat, boxes, T)

    assert torch.allclose(seen["rho"], expected_rho, atol=1e-5)


# ---------------------------------------------------------------------------
# structural property 1: exact k=0 mass conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_mass_is_conserved_at_k0(n_species):
    """This is the assembly most able to break this: a wrong divergence,
    a mishandled k=0 row, or a mis-scaled N_GRID factor anywhere in the
    round trip would all be visible here -- except a flipped flux sign,
    which this test structurally CANNOT catch (see
    `test_a_convex_free_energy_decays_rather_than_grows` below).
    """
    model = _model(n_species)
    boxes = _boxes(4)
    T = _T(4, value=2.0)
    rho_hat = _rho_hat(4, n_species, GRID, model.ops)

    drho_hat_dt = model.forward(rho_hat, boxes, T)

    k0 = drho_hat_dt[:, :, 0, 0, 0]
    assert torch.all(k0.real == 0.0)
    assert torch.all(k0.imag == 0.0)


# ---------------------------------------------------------------------------
# structural property 2: a real field's forward output is a valid
# half-spectrum (round-trips through a real-space transform unchanged)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_forward_output_round_trips_through_real_space(n_species):
    """A Nyquist-handling inconsistency (the class of bug that once put 98%
    of a rollout's spectral energy into one mode) shows up here: taking the
    output out to real space and back must reproduce it exactly, since a
    genuine half-spectrum of a real field carries no information a real
    round trip could lose.
    """
    model = _model(n_species)
    boxes = _boxes(2)
    T = _T(2)
    rho_hat = _rho_hat(2, n_species, GRID, model.ops)

    drho_hat_dt = model.forward(rho_hat, boxes, T)
    real = torch.fft.irfftn(drho_hat_dt, s=model.grid, dim=(-3, -2, -1))
    back = torch.fft.rfftn(real, dim=(-3, -2, -1))
    assert torch.allclose(back, drho_hat_dt, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# the channel-convention seam: distinguishable-value tests
# ---------------------------------------------------------------------------

def test_mobility_does_not_mix_species():
    """`Mobility.forward` is channel-LAST (`rho: (..., n) -> M: (..., n,
    n)`); the protocol's own `mobility` is channel-SECOND (`(B, n, n, Gx,
    Gy, Gz)`). A permute that lands one species' row/column on another
    species' index would NOT raise -- both are valid `(2, 2)`-shaped
    outputs -- so this fixture uses two species with DISTINGUISHABLE
    Cholesky diagonals (`shape="fixed"`, off-diagonal held at exactly zero)
    and two spatially DISTINCT density fields (species 0 varies along x,
    species 1 along y, different offsets and amplitudes), and checks each
    diagonal entry against an independently, directly indexed ground truth
    -- not against a second call through the model's own reshape path.
    """
    n = 2
    # raw Cholesky params (L_00, L_10, L_11); L_10 = 0 => no cross term, so
    # M is exactly diagonal and each diagonal entry is an uncoupled,
    # directly-checkable function of exactly one species' density.
    raw = torch.tensor([2.0, 0.0, -1.0])
    model = _model(n, mobility_shape="fixed", mobility_prefactor="partial_density",
                   mobility_shape_init=raw)

    B, Gx, Gy, Gz = 1, *GRID
    x = torch.arange(Gx, dtype=torch.float32).view(1, 1, Gx, 1, 1) / Gx
    y = torch.arange(Gy, dtype=torch.float32).view(1, 1, 1, Gy, 1) / Gy
    rho0 = (0.5 + 0.3 * torch.cos(2 * torch.pi * x)).expand(B, 1, Gx, Gy, Gz)
    rho1 = (0.2 + 0.1 * torch.cos(2 * torch.pi * y)).expand(B, 1, Gx, Gy, Gz)
    rho = torch.cat([rho0, rho1], dim=1)
    T = _T(B)

    M = model.mobility(rho, T)  # (B, n, n, Gx, Gy, Gz)
    assert M.shape == (B, n, n, Gx, Gy, Gz)

    c00 = F.softplus(torch.tensor(2.0)) ** 2
    c11 = F.softplus(torch.tensor(-1.0)) ** 2
    expected_M00 = c00 * rho[:, 0]
    expected_M11 = c11 * rho[:, 1]

    assert torch.allclose(M[:, 0, 0], expected_M00, atol=1e-5)
    assert torch.allclose(M[:, 1, 1], expected_M11, atol=1e-5)
    assert torch.allclose(M[:, 0, 1], torch.zeros_like(M[:, 0, 1]), atol=1e-6)
    assert torch.allclose(M[:, 1, 0], torch.zeros_like(M[:, 1, 0]), atol=1e-6)

    # The two species are genuinely distinguishable at this fixture -- a
    # channel swap could not pass by accident.
    assert not torch.allclose(expected_M00, expected_M11, atol=1e-2)
    assert not torch.allclose(rho[:, 0], rho[:, 1], atol=1e-2)


def test_kernel_term_does_not_mix_species():
    """`PairKernel.w_hat(k)` returns `(*k.shape, n, n)`; `chemical_potential`
    reconciles it against `rho_hat`'s channel-SECOND convention with an
    explicit `permute` + `einsum`. This test zeros the local free-energy
    term entirely (so `chemical_potential`'s output IS the kernel term) and
    the kernel's cross-species (0, 1) radial net (so `W_hat` is exactly
    diagonal and each species' kernel contribution is a directly, and
    INDEPENDENTLY, checkable scalar multiplication -- computed here via a
    different indexing path than `chemical_potential`'s own `einsum`, so an
    index swap inside it cannot pass by coincidentally computing the same
    thing twice).
    """
    n = 2
    model = _model(n)
    model.f_local.mu_pointwise = lambda rho, kBT: torch.zeros_like(rho)

    pairs = model.kernel.radial_set.pairs
    cross_idx = pairs.index((0, 1))
    with torch.no_grad():
        for p in model.kernel.radial_set.nets[cross_idx].parameters():
            p.zero_()

    B, Gx, Gy, Gz = 1, *GRID
    x = torch.arange(Gx, dtype=torch.float32).view(1, 1, Gx, 1, 1) / Gx
    y = torch.arange(Gy, dtype=torch.float32).view(1, 1, 1, Gy, 1) / Gy
    rho0 = (0.5 + 0.3 * torch.cos(2 * torch.pi * x)).expand(B, 1, Gx, Gy, Gz)
    rho1 = (0.2 + 0.1 * torch.cos(2 * torch.pi * y)).expand(B, 1, Gx, Gy, Gz)
    rho = torch.cat([rho0, rho1], dim=1).contiguous()
    boxes = _boxes(B)
    T = _T(B)

    mu = model.chemical_potential(rho, boxes, T)

    ops = model.ops
    N = ops.grid[0] * ops.grid[1] * ops.grid[2]
    kx, ky, kz = ops.k_axes(boxes)
    kmag = torch.sqrt(kx * kx + ky * ky + kz * kz)[:, 0]
    Wm = model.kernel.w_hat(kmag)  # (B, Gx, Gy, Gzr, 2, 2)
    assert torch.allclose(Wm[..., 0, 1], torch.zeros_like(Wm[..., 0, 1]), atol=1e-6)
    assert not torch.allclose(Wm[..., 0, 0], Wm[..., 1, 1], atol=1e-3), (
        "the two diagonal kernel entries must be distinguishable for this "
        "fixture to be able to catch a channel swap")

    rho_hat_mode = ops.rfft(rho) / N
    # Ground truth via plain scalar indexing, NOT the model's own
    # permute+einsum: species i's kernel contribution to mu_hat is exactly
    # W_hat_ii(k) * rho_hat_i(k), since the cross term is zero.
    expected_mu_hat_0 = Wm[..., 0, 0] * rho_hat_mode[:, 0]
    expected_mu_hat_1 = Wm[..., 1, 1] * rho_hat_mode[:, 1]

    mu_hat = ops.rfft(mu) / N
    assert torch.allclose(mu_hat[:, 0], expected_mu_hat_0, atol=1e-5)
    assert torch.allclose(mu_hat[:, 1], expected_mu_hat_1, atol=1e-5)


# ---------------------------------------------------------------------------
# the sign of the flux: a convex free energy must DECAY
# ---------------------------------------------------------------------------

def test_a_convex_free_energy_decays_rather_than_grows():
    """Mass conservation at k=0 holds for BOTH signs of the flux, so it
    cannot catch an inverted one. `chemical_potential` and `mobility` are
    monkeypatched to `mu(rho) = rho` (exactly `f'(rho)` for the convex
    `f(rho) = rho^2/2`) and a constant, positive `M`, isolating the sign of
    `forward`'s own flux/divergence assembly from the (untrained) local
    form, kernel and mobility nets. Linearised, a single Fourier mode obeys
    `d(rho_k)/dt = -M k^2 f'' rho_k`, strictly negative here -- growth means
    the flux sign is inverted.

    This exact defect happened once already in this phase: a reference
    skeleton returned `div(-M grad mu)`, a rung copied it, and 618 tests
    passed with a model whose every mode grew at +5.1 per unit time,
    because every structural check (shape, dtype, finiteness, k=0 mass) is
    blind to this sign.
    """
    grid = (8, 8, 8)
    length = 4.0
    model = _model(1, grid=grid, R_cut=1.5)
    model.chemical_potential = lambda rho, boxes, T: rho
    model.mobility = lambda rho, T: 0.7 * torch.ones(
        rho.shape[0], 1, 1, *rho.shape[2:])

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
    assert rate < 0, (
        f"a convex free energy must decay, got d(ln rho_k)/dt = {rate:+.4e}. "
        f"A positive rate is anti-diffusion: the flux sign is inverted.")

    # The sign alone does not pin the MAGNITUDE, and the magnitude is what
    # exposes a mis-scaled (or missing) N_GRID factor anywhere in forward's
    # own mu_hat/grad_hat/flux chain -- a mistake the sign check, the k=0
    # mass test and the round-trip test all pass right through, since each
    # of those factors cancels in a ratio or preserves self-consistency
    # without preserving the correct absolute scale. For mu(rho) = rho and
    # constant M, the continuous, linearised Model B rate for a single
    # Fourier mode is exactly `-M * k^2`, and this mode is low enough
    # (k = pi/2, one full period across an 8-point axis) to be free of
    # discretisation error at float32 precision.
    k_mag = 2 * torch.pi * 1 / length
    expected_rate = -0.7 * k_mag ** 2
    assert rate.item() == pytest.approx(expected_rate, rel=1e-4), (
        f"got rate {rate.item():+.6e}, expected exactly -M*k^2 = "
        f"{expected_rate:+.6e}: a scaling factor is off somewhere in "
        f"forward's mu_hat/grad_hat/flux chain, not merely its sign.")


# ---------------------------------------------------------------------------
# construction order: kernel and mobility weights must not depend on
# whether the T-basis heads are on
# ---------------------------------------------------------------------------

def test_kernel_and_mobility_weights_are_identical_whether_or_not_t_heads_are_on():
    """`FLocal` is built with `defer_T_heads=True`, the kernel and mobility
    are built next, and `build_T_heads()` is called LAST -- so a model
    built with `enable_TlnT`/`enable_T2` on draws the SAME kernel and
    mobility weights, at a fixed seed, as one built with both off; the
    T-basis heads only ever append draws to the end of the RNG stream. If
    the construction order were wrong -- the heads built any earlier -- a
    flags-on model's kernel/mobility weights would silently diverge from a
    flags-off model's, which would make every checkpoint trained under one
    configuration irreproducible under the other.
    """
    seed = 7
    off = _model(2, seed=seed)
    on = _model(2, seed=seed, enable_TlnT=True, enable_T2=True,
                h_g_hat=4, h_g_tilde=4)

    off_kernel = off.kernel.radial_set.state_dict()
    on_kernel = on.kernel.radial_set.state_dict()
    assert off_kernel.keys() == on_kernel.keys()
    for key in off_kernel:
        assert torch.equal(off_kernel[key], on_kernel[key]), key

    off_mob = off._mobility.state_dict()
    on_mob = on._mobility.state_dict()
    assert off_mob.keys() == on_mob.keys()
    for key in off_mob:
        assert torch.equal(off_mob[key], on_mob[key]), key

    off_local = off.f_local.u_net.state_dict()
    on_local = on.f_local.u_net.state_dict()
    for key in off_local:
        assert torch.equal(off_local[key], on_local[key]), key


# ---------------------------------------------------------------------------
# units: kB converts T for the local form, but Mobility sees T literally
# ---------------------------------------------------------------------------

def test_kB_converts_T_for_the_local_form_but_mobility_sees_T_literally():
    """`FLocal` takes `kBT`, never bare `T`; `Mobility` takes the
    literal `T` and, under `t_form="arrhenius"`, converts it itself with the
    `kB` this rung hands it. The spy below sees both arguments.
    """
    kB = 3.0
    model = _model(1, kB=kB)

    seen = {}
    real_mu_pointwise = model.f_local.mu_pointwise
    real_mobility = model._mobility.forward

    def spy_mu_pointwise(rho, kBT):
        seen["kBT"] = kBT.clone()
        return real_mu_pointwise(rho, kBT)

    def spy_mobility(rho, T):
        seen["T"] = T.clone()
        return real_mobility(rho, T)

    model.f_local.mu_pointwise = spy_mu_pointwise
    model._mobility.forward = spy_mobility

    B, Gx, Gy, Gz = 1, *GRID
    rho = 0.4 * torch.ones(B, 1, Gx, Gy, Gz)
    boxes = _boxes(B)
    T = torch.tensor([5.0])

    model.chemical_potential(rho, boxes, T)
    model.mobility(rho, T)

    assert torch.allclose(seen["kBT"], torch.full_like(seen["kBT"], kB * 5.0))
    assert torch.allclose(seen["T"], torch.full_like(seen["T"], 5.0))


# ---------------------------------------------------------------------------
# the variant tripwire
# ---------------------------------------------------------------------------

def test_an_unrecognised_variant_kwarg_raises():
    with pytest.raises(TypeError):
        _model(2, not_a_real_variant_kwarg=True)


def test_an_unknown_local_form_variant_still_raises_from_flocal():
    with pytest.raises(ValueError):
        _model(2, u_form="not_a_real_form")


def test_an_unknown_mobility_form_still_raises_from_mobility():
    with pytest.raises(ValueError):
        _model(2, mobility_shape="not_a_real_form")


def test_ideal_form_is_required_and_not_defaulted():
    """Choosing the ideal term is choosing a system's physics.

    ``gas`` and ``lattice`` are different closed forms read on different
    quantities -- a partial density versus a site fraction -- and they diverge
    by order 0.67 on the same input. A default here would silently give one
    system the other's thermodynamics, which is how a published checkpoint
    stopped reproducing its own numbers.
    """
    import inspect
    from aipf.functional.nonlocal_kernel import NonlocalKernel

    parameter = inspect.signature(NonlocalKernel.__init__).parameters["ideal_form"]
    assert parameter.default is inspect.Parameter.empty, (
        "ideal_form must be required; a default picks physics for the caller")
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("n_species", (1, 2))
def test_the_tbasis_ortho_window_reaches_the_local_form(n_species):
    """The assembly forwards the window rather than swallowing it.

    `FLocal` owns the orthogonalisation; this rung only has to hand the two
    declared fields over untouched. A silently dropped kwarg would leave a
    model that trains happily against the wrong basis, which is exactly the
    failure this assembly's other pass-through tests exist to catch.
    """
    off = _model(n_species, enable_TlnT=True, h_g_hat=4)
    on = _model(n_species, enable_TlnT=True, h_g_hat=4,
                tbasis_ortho_window=(0.6, 0.9), tbasis_ortho_points=13)
    assert off.f_local.tbasis_ortho is False
    assert on.f_local.tbasis_ortho is True

    kBT = torch.linspace(0.6, 0.9, 11, dtype=torch.float64)
    assert on.f_local._c_TlnT(kBT).abs().max() < \
        0.05 * off.f_local._c_TlnT(kBT).abs().max()

    # and it stays a pure config change: same keys, loadable across the flag
    assert sorted(off.state_dict()) == sorted(on.state_dict())


@pytest.mark.parametrize("n_species", (1, 2))
def test_a_half_declared_tbasis_ortho_is_refused_by_the_assembly(n_species):
    """The validation lives in FLocal; this checks the rung does not paper
    over it with a default of its own."""
    with pytest.raises(ValueError, match="tbasis_ortho_points"):
        _model(n_species, enable_TlnT=True, h_g_hat=4,
               tbasis_ortho_window=(0.6, 0.9))
    with pytest.raises(ValueError, match="nothing to fit"):
        _model(n_species, enable_TlnT=True, h_g_hat=4,
               tbasis_ortho_points=13)


# ---------------------------------------------------------------------------
# the k-space fast path, and the seam it must not silently bypass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", (1, 2))
def test_the_fast_path_and_the_real_space_seam_agree(n_species):
    """`forward` builds `mu_hat` directly instead of inverting a spectrum
    and transforming it straight back. That is an optimisation, so it must
    change nothing: the two routes are compared here on the same model and
    the same input, and agree to float32 round-off.

    The gap this closes was measured against the published model's own
    training code: routing through the real-space seam put this rung 8.6e-05
    (relative) away from it in float32 and 2.2e-13 in float64 -- round-off from
    three redundant transforms, not a formula difference.

    Going direct brought float32 to ~7e-06 at the time, inside the spec's
    1e-5 parity tolerance. That residual no longer exists and this
    docstring is not the place to look it up: `8e8f611` fixed the
    Gram-Schmidt association a day later and the whole rung-3 forward is
    now bit-for-bit identical in float32 on both grids at every
    temperature (see the code-parity report). The tolerances below are the
    two-route agreement's own, not that measurement's.
    """
    model = _model(n_species, grid=(8, 8, 8))
    boxes = torch.tensor([[4.0, 4.0, 4.0]])
    T = torch.tensor([1.0])
    rho = 0.4 + 0.02 * torch.rand(1, n_species, 8, 8, 8)
    N = 8 * 8 * 8
    rho_hat = model.ops.rfft(rho) / N

    fast = model(rho_hat, boxes, T)
    assert not model._chemical_potential_is_overridden()

    # force the real-space route with a seam that is deliberately identical
    cls_cp = NonlocalKernel.chemical_potential
    model.chemical_potential = lambda r, b, t: cls_cp(model, r, b, t)
    assert model._chemical_potential_is_overridden()
    slow = model(rho_hat, boxes, T)

    assert torch.allclose(fast, slow, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("n_species", (1, 2))
def test_an_overridden_chemical_potential_still_reaches_forward(n_species):
    """The fast path must not make the documented override point dead code.

    A model whose `chemical_potential` is replaced -- by a subclass, or by a
    test monkeypatching an already-constructed instance -- must see that
    replacement honoured in `forward`. Taking the k-space route
    unconditionally would silently ignore it, which is the same class of
    defect as a swallowed unknown kwarg: it runs, it looks fine, and it
    computes the wrong model.
    """
    model = _model(n_species, grid=(8, 8, 8))
    boxes = torch.tensor([[4.0, 4.0, 4.0]])
    T = torch.tensor([1.0])
    rho = 0.4 + 0.02 * torch.rand(1, n_species, 8, 8, 8)
    N = 8 * 8 * 8
    rho_hat = model.ops.rfft(rho) / N

    before = model(rho_hat, boxes, T)
    model.chemical_potential = lambda r, b, t: 3.0 * r
    after = model(rho_hat, boxes, T)

    assert not torch.allclose(before, after), (
        "replacing chemical_potential left forward's output unchanged, so "
        "the override was ignored and forward is not built from the seam "
        "it documents")


@pytest.mark.parametrize("n_species", (1, 2))
def test_forward_actually_takes_the_k_space_path_when_nothing_overrides(
        n_species):
    """Agreement between the two routes does not prove the fast one runs.

    Without this, leaving `forward` permanently on the real-space seam
    passes every other test in this file -- the results are identical, only
    slower. Confirmed by mutation testing: forcing the slow branch survives
    all of them.

    What is watched here is `_mu_hat`, not `chemical_potential`. The seam
    itself can no longer be spied on: patching it, on the class or on the
    instance, IS an override, and the guard now says so, which is the whole
    of the fix this test was rewritten for. (Its earlier version patched
    the class attribute and asserted the guard reported `False` -- true
    only because the guard compared the patched attribute against itself,
    so the test was asserting the defect.) `_mu_hat` is not an override
    point, and the two routes reach it differently in a way that is not a
    matter of taste: the fast branch hands it the CALLER's spectrum, while
    the seam calls it with no spectrum at all and lets it re-derive one.
    So "was `_mu_hat` called exactly once, with the caller's `rho_hat`"
    distinguishes the branches exactly, and fails under a forced seam.
    """
    import unittest.mock as mock

    model = _model(n_species, grid=(8, 8, 8))
    boxes = torch.tensor([[4.0, 4.0, 4.0]])
    T = torch.tensor([1.0])
    rho = 0.4 + 0.02 * torch.rand(1, n_species, 8, 8, 8)
    rho_hat = model.ops.rfft(rho) / (8 * 8 * 8)

    assert model._chemical_potential_is_overridden() is False

    seen = []
    real_mu_hat = NonlocalKernel._mu_hat

    def recording(self, rho, boxes, T, rho_hat=None):
        seen.append(rho_hat)
        return real_mu_hat(self, rho, boxes, T, rho_hat)

    with mock.patch.object(NonlocalKernel, "_mu_hat", recording):
        model(rho_hat, boxes, T)

    assert len(seen) == 1, (
        f"_mu_hat ran {len(seen)} times for one forward; the k-space path "
        f"builds mu_hat exactly once")
    assert seen[0] is not None and torch.equal(seen[0], rho_hat), (
        "forward built mu_hat without the caller's spectrum, so it went "
        "through the real-space chemical_potential seam even though "
        "nothing overrides it, and the k-space fast path is dead code")


@pytest.mark.parametrize("n_species", (1, 2))
def test_the_fast_path_costs_three_transforms_less(n_species):
    """What the k-space route actually saves, counted rather than timed.

    The comment in `forward` used to claim "measured 1.46x on the whole
    step". That number is in no report, was asserted by nothing, and does
    not reproduce: re-measured over five grid/batch
    configurations, the two arms interleaved trial by trial, the ratio came
    out anywhere from 0.65 to 3.07 on a node carrying three times its core
    count. A wall-clock ratio is not a property of this code on a shared
    machine, so what is pinned here is the thing that IS one: the number of
    Fourier transforms a forward call makes.

    Eight against eleven. The seam costs one `irfft` to leave k-space, one
    `rfft` to come back, and one more `rfft` inside `_mu_hat`, which has to
    re-derive a spectrum it was not given. Three transforms out of eleven
    is roughly a tenth of a step once the pointwise work, the kernel
    einsum and the mobility are counted, which is the size of effect the
    timings above are consistent with and the 1.46x was not.
    """
    model = _model(n_species, grid=(8, 8, 8))
    boxes = torch.tensor([[4.0, 4.0, 4.0]])
    T = torch.tensor([1.0])
    rho = 0.4 + 0.02 * torch.rand(1, n_species, 8, 8, 8)
    rho_hat = model.ops.rfft(rho) / (8 * 8 * 8)
    cls_cp = NonlocalKernel.chemical_potential

    def count(slow):
        if slow:
            model.chemical_potential = lambda r, b, t: cls_cp(model, r, b, t)
        else:
            model.__dict__.pop("chemical_potential", None)
        assert model._chemical_potential_is_overridden() is slow
        seen = {"rfft": 0, "irfft": 0}
        real = {name: getattr(model.ops, name) for name in seen}

        def wrap(name):
            def wrapped(*a, **kw):
                seen[name] += 1
                return real[name](*a, **kw)
            return wrapped
        for name in seen:
            setattr(model.ops, name, wrap(name))
        try:
            with torch.no_grad():
                model(rho_hat, boxes, T)
        finally:
            for name, fn in real.items():
                setattr(model.ops, name, fn)
        return sum(seen.values())

    fast, slow = count(False), count(True)
    assert (fast, slow) == (8, 11), (
        f"the k-space path makes {fast} transforms per forward and the "
        f"real-space seam {slow}; the saving this branch exists for is "
        f"three (an irfft out, an rfft back, and the rfft the seam forces "
        f"_mu_hat to re-derive)")


@pytest.mark.parametrize("n_species", (1, 2))
def test_a_recomputed_spectrum_would_change_nothing_observable(n_species):
    """Passing the caller's `rho_hat` is not a change of meaning.

    `irfft` keeps only the Hermitian part of its input, so the caller's
    spectrum and one re-derived from `rho` genuinely differ -- and they
    differ across the WHOLE kz = 0 and kz = G_z/2 planes, not only at the
    eight self-conjugate corners. The earlier fixture tried exactly four
    corner-ish indices, which are the four where the reason then given
    (k = 0 killed by the gradient, Nyquist masked) happens to apply; it
    would have passed just as well at (1, 0, 0), where that reason does
    not. The reason that does hold everywhere is that every consumer of
    `mu_hat` returns to real space through `irfft`, which discards the
    antisymmetric component wherever it sits, so the list below now
    includes ordinary paired modes in both planes and one
    (4, 1, 0) where the per-axis Nyquist mask zeroes only the x-component
    and the y-derivative survives.

    Pinned here because the first version of this test asserted the
    OPPOSITE and passed on 5e-08 of round-off -- so that nobody, including
    a later reading of this file, "fixes" a discrepancy that is not one.
    """
    model = _model(n_species, grid=(8, 8, 8))
    boxes = torch.tensor([[4.0, 4.0, 4.0]])
    T = torch.tensor([1.0])
    N = 8 * 8 * 8
    rho = 0.4 + 0.02 * torch.rand(1, n_species, 8, 8, 8)
    rho_hat = model.ops.rfft(rho) / N

    for idx in ((0, 0, 0), (0, 0, 4), (4, 0, 0), (4, 4, 4),
                (1, 0, 0), (0, 1, 0), (1, 1, 0), (4, 1, 0), (2, 3, 0),
                (1, 0, 4), (1, 2, 4), (0, 3, 4), (3, 3, 4), (2, 0, 4)):
        dirty = rho_hat.clone()
        dirty[..., idx[0], idx[1], idx[2]] += 0.05j
        projected = model.ops.rfft(model.ops.irfft(dirty * N)) / N
        lost = (dirty - projected).abs().max()
        assert lost > 1e-3, (idx, float(lost))       # the fixture IS dirty
        moved = (model(dirty, boxes, T)
                 - model(projected, boxes, T)).abs().max()
        assert moved < 1e-6, (
            f"non-Hermitian content at {idx} moved forward by {moved:.3e}; "
            f"it is supposed to be discarded downstream, so passing the "
            f"caller's spectrum would no longer be a pure optimisation")


@pytest.mark.parametrize("n_species", (1, 2))
def test_a_class_level_patch_of_the_seam_is_honoured_too(n_species):
    """The override guard must see a patch of the CLASS attribute.

    The instance case is covered above. This is the other half, and it is
    the commoner idiom: `mock.patch.object(NonlocalKernel,
    "chemical_potential", ...)`, which every mocking library reaches for
    first. A guard that compares `type(self).chemical_potential` against
    `NonlocalKernel.chemical_potential` compares the patched attribute
    with ITSELF -- both names resolve to the same patched object -- so it
    reports "not overridden" and `forward` silently computes the unpatched
    model. That is exactly the swallowed-override defect the fast path's
    guard exists to prevent, and before the guard captured the function at
    import it was measured here at max diff 0.0.
    """
    import unittest.mock as mock

    model = _model(n_species, grid=(8, 8, 8))
    boxes = torch.tensor([[4.0, 4.0, 4.0]])
    T = torch.tensor([1.0])
    rho = 0.4 + 0.02 * torch.rand(1, n_species, 8, 8, 8)
    rho_hat = model.ops.rfft(rho) / (8 * 8 * 8)

    before = model(rho_hat, boxes, T)
    with mock.patch.object(NonlocalKernel, "chemical_potential",
                           lambda self, r, b, t: 3.0 * r):
        assert model._chemical_potential_is_overridden() is True
        after = model(rho_hat, boxes, T)

    assert not torch.allclose(before, after), (
        "a class-level patch of chemical_potential left forward's output "
        "unchanged, so the override guard compared the patched attribute "
        "against itself and the fast path ignored it")
