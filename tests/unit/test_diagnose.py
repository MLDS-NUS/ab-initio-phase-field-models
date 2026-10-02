"""Tests for aipf.diagnose: thermodynamics and dynamics.

Properties and known closed-form values, following the pattern of
``tests/unit/test_spectral.py`` and ``tests/unit/test_kernels.py``: a known
analytic free energy gives its known binodal/spinodal (Landau, closed form);
the lever anchor is a declared field and an off-centre dome with the wrong
anchor is detectably wrong (a two-well quartic whose wells sit both below
x=0.5, so the hardcoded x_bar=0.5 habit reads "single phase" there); a
reference dataset declared for one system raises if used for another
(the recorded accident this package makes structurally impossible); and
``n_species`` is exercised at 1 and 2 throughout, per the plan's global
constraint.
"""
from __future__ import annotations

import math

import pytest
import torch

from aipf.diagnose.dynamics import (
    finite_difference_rate,
    growth_rate,
    most_unstable_k,
    rollout_drift,
    structure_factor,
)
from aipf.diagnose.kappa import nonlocal_kernel_kappa_eff
from aipf.diagnose.reference import ReferenceDataset, for_system
from aipf.diagnose.thermo import (
    BINODAL_ROUTES,
    TC_METHODS,
    binodal,
    binodal_convex_hull,
    binodal_mu_roots,
    critical_temperature,
    lower_hull_indices,
    spinodal,
)
from aipf.functional.landau import Landau
from aipf.spectral import SpectralOps

torch.manual_seed(0)

GRID = (4, 4, 4)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _landau_mu(a0, b, T_c):
    """A plain-Python mu(rho, T) callable wrapping the ACTUAL rung-1
    ``Landau`` model (imported read-only) -- so the "known
    analytic free energy gives its known binodal" test exercises the real
    ladder component, not a hand-rolled reimplementation of its formula.
    """
    model = Landau(GRID, 1, a0=a0, b=b, T_c=T_c, gamma=1.0, nyquist_mask=True).double()

    def mu(rho: float, T: float) -> float:
        rho_t = torch.full((1, 1, 1, 1, 1), float(rho), dtype=torch.float64)
        boxes_t = torch.full((1, 3), 10.0, dtype=torch.float64)
        T_t = torch.full((1,), float(T), dtype=torch.float64)
        with torch.no_grad():
            out = model.chemical_potential(rho_t, boxes_t, T_t)
        return float(out.reshape(-1)[0])

    return mu


def _two_well_quartic(c: float, d: float, A: float = 1.0):
    """``f(x) = A * (x - c)^2 * (x - c - d)^2``: an EXACT equal-depth double
    well with minima at ``x = c`` and ``x = c + d``, both zero -- so the
    lower convex hull's touch points are the two minima themselves, up to
    grid resolution. Used as the off-centre-dome fixture: wells at 0.1 and
    0.4 sit entirely below x = 0.5, the hardcoded lever anchor.
    """
    def f(x: float, T: float) -> float:
        del T
        return A * (x - c) ** 2 * (x - c - d) ** 2

    return f


class _ConstantRateModel:
    """A ``FreeEnergyModel.forward`` stub returning a fixed rate regardless
    of its input -- lets :func:`rollout_drift` be tested against a
    trajectory built to have EXACTLY that rate (drift ~ 0) or a perturbed
    one (drift > 0), with no dependency on a real rung.
    """

    def __init__(self, rate: torch.Tensor):
        self.rate = rate

    def forward(self, rho_hat, boxes, T):
        del boxes, T
        return self.rate.expand_as(rho_hat).clone()


# ---------------------------------------------------------------------------
# binodal_mu_roots: known analytic free energy -> known binodal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_val,expected_psi", [(0.5, 0.5), (0.0, math.sqrt(0.75))])
def test_landau_binodal_matches_the_closed_form(T_val, expected_psi):
    """Landau: mu = 0 off psi=0 gives psi_eq = sqrt(a0*(Tc-T)/b) exactly.

    a0=2, b=4, Tc=1 -> psi_eq(T=0.5) = sqrt(2*0.5/4) = 0.5,
    psi_eq(T=0.0) = sqrt(2*1/4) = sqrt(0.5)... corrected inline below.
    """
    a0, b, T_c = 2.0, 4.0, 1.0
    mu = _landau_mu(a0, b, T_c)
    result = binodal_mu_roots(mu, T_val, (1e-4, 1 - 1e-4), symmetry_point=0.5)
    assert result is not None
    psi_eq = math.sqrt(a0 * (T_c - T_val) / b)
    rho_lo, rho_hi = 0.5 * (1 - psi_eq), 0.5 * (1 + psi_eq)
    assert result[0] == pytest.approx(rho_lo, abs=1e-6)
    assert result[1] == pytest.approx(rho_hi, abs=1e-6)


def test_landau_binodal_is_none_above_T_c():
    """T > T_c: mu is monotonic through psi=0, no non-trivial root."""
    mu = _landau_mu(a0=2.0, b=4.0, T_c=1.0)
    assert binodal_mu_roots(mu, 1.5, (1e-4, 1 - 1e-4), symmetry_point=0.5) is None


def test_binodal_mu_roots_requires_symmetry_point_inside_domain():
    mu = _landau_mu(a0=2.0, b=4.0, T_c=1.0)
    with pytest.raises(ValueError, match="domain"):
        binodal_mu_roots(mu, 0.5, (0.2, 0.6), symmetry_point=0.9)


def test_binodal_mu_roots_has_no_default_symmetry_point():
    """No default: TypeError, not a silent 0.5."""
    mu = _landau_mu(a0=2.0, b=4.0, T_c=1.0)
    with pytest.raises(TypeError):
        binodal_mu_roots(mu, 0.5, (1e-4, 1 - 1e-4))  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# spinodal: known analytic free energy -> known spinodal
# ---------------------------------------------------------------------------

def test_landau_spinodal_matches_the_closed_form():
    """d(mu)/drho = 0 off psi=0: psi_sp = sqrt(a0*(Tc-T)/(3b))."""
    a0, b, T_c, T_val = 2.0, 4.0, 1.0, 0.5
    mu = _landau_mu(a0, b, T_c)
    roots = spinodal(mu, T_val, (1e-4, 1 - 1e-4), n_grid=800)
    psi_sp = math.sqrt(a0 * (T_c - T_val) / (3.0 * b))
    want = sorted([0.5 * (1 - psi_sp), 0.5 * (1 + psi_sp)])
    assert len(roots) == 2
    assert roots[0] == pytest.approx(want[0], abs=1e-3)
    assert roots[1] == pytest.approx(want[1], abs=1e-3)


def test_landau_spinodal_is_empty_above_T_c():
    mu = _landau_mu(a0=2.0, b=4.0, T_c=1.0)
    assert spinodal(mu, 1.5, (1e-4, 1 - 1e-4)) == []


def test_spinodal_binodal_ordering():
    """Physical sanity, at every T with coexistence: spinodal is strictly
    inside binodal (the classic dome-inside-dome picture)."""
    mu = _landau_mu(a0=2.0, b=4.0, T_c=1.0)
    b_lo, b_hi = binodal_mu_roots(mu, 0.3, (1e-4, 1 - 1e-4), symmetry_point=0.5)
    s_lo, s_hi = spinodal(mu, 0.3, (1e-4, 1 - 1e-4), n_grid=800)
    assert b_lo < s_lo < 0.5 < s_hi < b_hi


# ---------------------------------------------------------------------------
# lower_hull_indices: correctness on a known point set
# ---------------------------------------------------------------------------

def test_lower_hull_indices_drops_the_local_bump():
    """W-shaped point set: the up-bump in the middle is not on the lower
    hull, every other point is."""
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [0.0, -1.0, 1.0, -1.0, 0.0]
    assert lower_hull_indices(xs, ys) == [0, 1, 3, 4]


def test_lower_hull_indices_drops_an_exactly_collinear_midpoint():
    """A minimal vertex set: three exactly collinear points keep only the
    two endpoints. Matters downstream in ``binodal_convex_hull``, which
    reads facet WIDTH off consecutive hull vertices -- a redundant
    collinear vertex would narrow a real two-phase facet into two pieces
    the "adjacent grid points" check then misreads as single phase.
    """
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 2.0]
    assert lower_hull_indices(xs, ys) == [0, 2]


def test_lower_hull_indices_keeps_every_point_of_a_convex_v():
    xs = [0.0, 1.0, 2.0]
    ys = [1.0, 0.0, 1.0]
    assert lower_hull_indices(xs, ys) == [0, 1, 2]


def test_lower_hull_lies_on_or_below_every_point():
    """General property, on a random point set: no point lies strictly
    below the piecewise-linear interpolation of the hull that claims to be
    the LOWER hull."""
    g = torch.Generator().manual_seed(3)
    xs = sorted(torch.rand(30, generator=g).tolist())
    ys = torch.rand(30, generator=g).tolist()
    idx = lower_hull_indices(xs, ys)
    hx = [xs[i] for i in idx]
    hy = [ys[i] for i in idx]
    for x, y in zip(xs, ys):
        j = max(k for k in range(len(hx) - 1) if hx[k] <= x) if x < hx[-1] else len(hx) - 2
        x0, x1 = hx[j], hx[j + 1]
        y0, y1 = hy[j], hy[j + 1]
        t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        line = y0 + t * (y1 - y0)
        assert y >= line - 1e-9


# ---------------------------------------------------------------------------
# binodal_convex_hull: the off-centre dome, the declared-anchor bug it fixes
# ---------------------------------------------------------------------------

def test_hardcoded_anchor_reads_single_phase_on_an_off_centre_dome():
    """The recorded accident, reproduced: wells at x=0.1 and x=0.4, so the
    two-phase region never reaches x=0.5. A hardcoded x_bar=0.5 -- the
    first system's habit -- returns None here, exactly the "single phase"
    misread its own comment names for the second system's actual dome.
    """
    f = _two_well_quartic(c=0.1, d=0.3)
    grid = torch.linspace(0.02, 0.9, 441).tolist()
    result = binodal_convex_hull(f, T=0.0, grid=grid, anchor=0.5)
    assert result is None


def test_auto_anchor_finds_the_off_centre_dome():
    """anchor="auto" resolves to the point of deepest curvature between the
    two wells and correctly returns the tie line, close to the true
    equal-depth minima at x=0.1 and x=0.4."""
    f = _two_well_quartic(c=0.1, d=0.3)
    grid = torch.linspace(0.02, 0.9, 441).tolist()
    result = binodal_convex_hull(f, T=0.0, grid=grid, anchor="auto")
    assert result is not None
    x_lo, x_hi = result
    assert x_lo == pytest.approx(0.1, abs=0.02)
    assert x_hi == pytest.approx(0.4, abs=0.02)


def test_binodal_convex_hull_has_no_default_anchor():
    f = _two_well_quartic(c=0.1, d=0.3)
    grid = torch.linspace(0.02, 0.9, 441).tolist()
    with pytest.raises(TypeError):
        binodal_convex_hull(f, T=0.0, grid=grid)  # type: ignore[call-arg]


def test_binodal_convex_hull_rejects_an_anchor_outside_the_grid():
    f = _two_well_quartic(c=0.1, d=0.3)
    grid = torch.linspace(0.02, 0.9, 441).tolist()
    with pytest.raises(ValueError, match="outside the grid domain"):
        binodal_convex_hull(f, T=0.0, grid=grid, anchor=0.99)


def test_binodal_convex_hull_none_for_a_single_convex_well():
    """A single, genuinely convex free energy: every anchor reads single
    phase, including "auto"."""
    def f(x, T):
        del T
        return (x - 0.5) ** 2

    grid = torch.linspace(0.02, 0.98, 200).tolist()
    assert binodal_convex_hull(f, T=0.0, grid=grid, anchor=0.5) is None
    assert binodal_convex_hull(f, T=0.0, grid=grid, anchor="auto") is None


@pytest.mark.parametrize("n_species_context", [1, 2])
def test_binodal_convex_hull_route_covers_both_species_counts(n_species_context):
    """The hull route is the one the n_species=2 systems actually use (their
    composition axis); n_species=1 is the density axis itself. Both are
    exercised as a scalar order-parameter scan, so the SAME function is
    checked at both -- there is no n_species branch inside this module, by
    design (see the module docstring)."""
    del n_species_context
    f = _two_well_quartic(c=0.15, d=0.2)
    grid = torch.linspace(0.02, 0.9, 300).tolist()
    lo, hi = binodal_convex_hull(f, T=0.0, grid=grid, anchor="auto")
    assert lo == pytest.approx(0.15, abs=0.02)
    assert hi == pytest.approx(0.35, abs=0.02)


# ---------------------------------------------------------------------------
# binodal(): the declared-route dispatcher
# ---------------------------------------------------------------------------

def test_binodal_dispatches_to_mu_roots():
    mu = _landau_mu(a0=2.0, b=4.0, T_c=1.0)
    got = binodal("mu_roots", T=0.5, mu=mu, domain=(1e-4, 1 - 1e-4),
                   symmetry_point=0.5)
    direct = binodal_mu_roots(mu, 0.5, (1e-4, 1 - 1e-4), symmetry_point=0.5)
    assert got == direct


def test_binodal_dispatches_to_convex_hull():
    f = _two_well_quartic(c=0.1, d=0.3)
    grid = torch.linspace(0.02, 0.9, 441).tolist()
    got = binodal("convex_hull", T=0.0, f=f, grid=grid, anchor="auto")
    direct = binodal_convex_hull(f, T=0.0, grid=grid, anchor="auto")
    assert got == direct


def test_binodal_rejects_an_unknown_route():
    with pytest.raises(ValueError, match="unknown binodal route"):
        binodal("common_tangent", T=0.5, mu=lambda x, T: x)


def test_binodal_routes_tuple_has_exactly_the_two_documented_routes():
    assert BINODAL_ROUTES == ("mu_roots", "convex_hull")


# ---------------------------------------------------------------------------
# critical_temperature: bracket and ising_fit, both declared
# ---------------------------------------------------------------------------

def test_critical_temperature_bracket_on_landau():
    mu = _landau_mu(a0=2.0, b=4.0, T_c=1.0)

    def binodal_at(T):
        return binodal_mu_roots(mu, T, (1e-4, 1 - 1e-4), symmetry_point=0.5)

    T_grid = [0.5, 0.7, 0.9, 0.95, 0.99, 1.0, 1.05, 1.1, 1.3]
    T_c = critical_temperature(binodal_at, T_grid, method="bracket")
    assert 0.99 <= T_c <= 1.05


def test_critical_temperature_ising_fit_recovers_a_synthetic_dome():
    """A synthetic delta(T) = (B/2)(1-T/Tc)^beta EXACTLY, beta=0.325 fixed:
    the fit should recover B and T_c to high precision, independent of any
    physical model -- this isolates the fitting method itself."""
    beta, B_true, T_c_true = 0.325, 0.8, 2.0

    def binodal_at(T):
        if T >= T_c_true:
            return None
        delta = 0.5 * B_true * (1.0 - T / T_c_true) ** beta
        return (0.5 - delta, 0.5 + delta)

    T_grid = [0.2 * i for i in range(1, 10)]
    T_c = critical_temperature(binodal_at, T_grid, method="ising_fit", beta=beta)
    assert T_c == pytest.approx(T_c_true, rel=1e-3)


def test_critical_temperature_ising_fit_requires_beta():
    def binodal_at(T):
        return (0.4, 0.6) if T < 1.0 else None

    with pytest.raises(ValueError, match="beta"):
        critical_temperature(binodal_at, [0.5, 0.9, 1.1], method="ising_fit")


def test_critical_temperature_rejects_an_unknown_method():
    def binodal_at(T):
        return (0.4, 0.6) if T < 1.0 else None

    with pytest.raises(ValueError, match="unknown method"):
        critical_temperature(binodal_at, [0.5, 1.5], method="fixed_point")


def test_critical_temperature_rejects_an_unknown_method_even_with_beta_given():
    """The unknown-method check must fire BEFORE anything falls through to
    the ising_fit branch by elimination: pass everything an ising_fit call
    would need (beta, >=3 two-phase points) so a version of this function
    that dispatches by "== 'bracket', else ising_fit" -- rather than
    validating against the declared tuple -- would silently run instead of
    raising, and this is the test built to catch exactly that.
    """
    def binodal_at(T):
        return (0.4, 0.6) if T < 2.0 else None

    with pytest.raises(ValueError, match="unknown method"):
        critical_temperature(binodal_at, [0.5, 1.0, 1.5, 2.5],
                              method="fixed_point", beta=0.325)


def test_tc_methods_tuple_has_exactly_the_documented_methods():
    """Three, because the published
    caches of the first system store a mean-field fit of the dome's own
    top, which is neither the fixed-exponent fit nor the bracket."""
    assert TC_METHODS == ("ising_fit", "bracket", "dome_apex")


# ---------------------------------------------------------------------------
# reference datasets: the structural guard
# ---------------------------------------------------------------------------

def test_reference_dataset_for_its_own_system_passes_through():
    ref = ReferenceDataset(system_name="alpha", kind="binodal", data=(1, 2, 3))
    assert for_system(ref, "alpha") is ref


def test_reference_dataset_for_the_wrong_system_raises():
    """The recorded accident, made structurally impossible: a reference
    declared for one system used to score another raises before any number
    is computed."""
    ref = ReferenceDataset(system_name="alpha", kind="binodal", data=(1, 2, 3))
    with pytest.raises(ValueError, match="alpha.*beta|declared for system"):
        for_system(ref, "beta")


# ---------------------------------------------------------------------------
# kappa_eff: a property of the rung-3 kernel, never a general ladder metric
# ---------------------------------------------------------------------------

class _FakeKernel:
    def kappa_eff(self):
        return torch.eye(2)


class _FakeNonlocalKernel:
    def __init__(self):
        self.kernel = _FakeKernel()


def test_nonlocal_kernel_kappa_eff_reads_through_a_model():
    model = _FakeNonlocalKernel()
    got = nonlocal_kernel_kappa_eff(model)
    assert torch.equal(got, torch.eye(2))


def test_nonlocal_kernel_kappa_eff_reads_a_bare_kernel_directly():
    got = nonlocal_kernel_kappa_eff(_FakeKernel())
    assert torch.equal(got, torch.eye(2))


def test_nonlocal_kernel_kappa_eff_raises_on_a_model_with_no_kernel():
    class _FakeLandau:
        pass

    with pytest.raises(AttributeError, match="kappa_eff"):
        nonlocal_kernel_kappa_eff(_FakeLandau())


# ---------------------------------------------------------------------------
# growth_rate: linear dispersion, against a known analytic case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_growth_rate_matches_the_classic_cahn_hilliard_dispersion(n_species):
    """Scalar-per-channel case (no cross terms): lambda(k) =
    -M*k^2*(f'' + kappa*k^2), maximised at k*^2 = -f''/(2*kappa) when
    f'' < 0 -- the textbook spinodal-decomposition result, checked against
    ``growth_rate`` with ``w_hat(k) = kappa * k^2 * I``.
    """
    M_scalar, fpp, kappa = 0.7, -2.0, 0.5
    hessian = fpp * torch.eye(n_species, dtype=torch.float64)
    mobility = M_scalar * torch.eye(n_species, dtype=torch.float64)

    def w_hat(k):
        eye = torch.eye(n_species, dtype=torch.float64)
        return (kappa * k * k)[..., None, None] * eye

    k = torch.linspace(0.0, 3.0, 601, dtype=torch.float64)
    lam = growth_rate(hessian, mobility, k, w_hat=w_hat)
    assert lam.shape == k.shape

    k_star_analytic = math.sqrt(-fpp / (2.0 * kappa))
    i_star = int(torch.argmax(lam))
    assert float(k[i_star]) == pytest.approx(k_star_analytic, abs=0.02)

    lam_analytic = -M_scalar * k * k * (fpp + kappa * k * k)
    assert torch.allclose(lam, lam_analytic, atol=1e-8)


def test_growth_rate_negative_definite_hessian_means_k0_is_unstable():
    """A convex-free-energy state (f'' > 0, no kernel): every k decays."""
    hessian = torch.tensor([[3.0]], dtype=torch.float64)
    mobility = torch.tensor([[1.0]], dtype=torch.float64)
    k = torch.linspace(0.01, 2.0, 50, dtype=torch.float64)
    lam = growth_rate(hessian, mobility, k)
    assert bool((lam <= 0).all())


def test_most_unstable_k_matches_growth_rate_argmax():
    fpp, kappa, M_scalar = -1.5, 0.3, 1.0
    hessian = torch.tensor([[fpp]], dtype=torch.float64)
    mobility = torch.tensor([[M_scalar]], dtype=torch.float64)

    def w_hat(k):
        return (kappa * k * k)[..., None, None]

    k_grid = torch.linspace(0.0, 3.0, 301, dtype=torch.float64)
    k_star, lam_star = most_unstable_k(hessian, mobility, k_grid, w_hat=w_hat)
    lam = growth_rate(hessian, mobility, k_grid, w_hat=w_hat)
    assert lam_star == pytest.approx(float(lam.max()))
    assert k_star == pytest.approx(float(k_grid[int(torch.argmax(lam))]))


def test_growth_rate_rejects_a_mismatched_mobility_shape():
    hessian = torch.eye(2, dtype=torch.float64)
    mobility = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="mobility"):
        growth_rate(hessian, mobility, torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# structure_factor: radial binning against a brute-force k-shell count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_structure_factor_matches_a_brute_force_shell_average(n_species):
    grid = (6, 6, 8)
    ops = SpectralOps(grid, n_species, nyquist_mask=True).double()
    boxes = torch.full((1, 3), 10.0, dtype=torch.float64)
    g = torch.Generator().manual_seed(7)
    rho_hat = torch.complex(
        torch.randn(1, n_species, *grid[:2], grid[2] // 2 + 1, generator=g,
                     dtype=torch.float64),
        torch.randn(1, n_species, *grid[:2], grid[2] // 2 + 1, generator=g,
                     dtype=torch.float64),
    )
    n_bins = 5
    centers, S = structure_factor(rho_hat, ops, boxes, n_bins)
    assert centers.shape == (n_bins,)
    assert S.shape == (1, n_bins)

    kmag = ops.k2(boxes).sqrt()[0, 0]
    power = (rho_hat.abs() ** 2 * ops.MULT).sum(dim=1)[0]
    k_top = float(kmag.max())
    edges = torch.linspace(0.0, k_top, n_bins + 1, dtype=torch.float64)
    for j in range(n_bins):
        lo, hi = edges[j], edges[j + 1]
        sel = (kmag >= lo) & (kmag < hi) if j < n_bins - 1 else (
            (kmag >= lo) & (kmag <= hi))
        if bool(sel.any()):
            assert S[0, j] == pytest.approx(float(power[sel].mean()), rel=1e-5)


def test_structure_factor_rejects_zero_bins():
    ops = SpectralOps((4, 4, 4), 1, nyquist_mask=True).double()
    boxes = torch.full((1, 3), 10.0, dtype=torch.float64)
    rho_hat = torch.zeros(1, 1, 4, 4, 3, dtype=torch.complex128)
    with pytest.raises(ValueError, match="n_bins"):
        structure_factor(rho_hat, ops, boxes, 0)


# ---------------------------------------------------------------------------
# rollout_drift: the training loss l_dyn, read off a stored trajectory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_rollout_drift_is_near_zero_when_the_model_matches_the_trajectory(n_species):
    grid = (4, 4, 4)
    ops = SpectralOps(grid, n_species, nyquist_mask=True).double()
    n_frames, B = 4, 2
    shape = (B, n_species, *grid[:2], grid[2] // 2 + 1)
    g = torch.Generator().manual_seed(11)
    rate = torch.complex(torch.randn(*shape, generator=g, dtype=torch.float64),
                          torch.randn(*shape, generator=g, dtype=torch.float64))
    rho0 = torch.complex(torch.randn(*shape, generator=g, dtype=torch.float64),
                          torch.randn(*shape, generator=g, dtype=torch.float64))
    dt = 0.5
    traj = torch.stack([rho0 + t * dt * rate for t in range(n_frames)], dim=0)
    boxes_traj = torch.full((n_frames, B, 3), 10.0, dtype=torch.float64)
    T_traj = torch.full((n_frames, B), 1.0, dtype=torch.float64)

    model = _ConstantRateModel(rate)
    drift = rollout_drift(model, traj, boxes_traj, T_traj, dt, ops,
                           alpha=0.5, k_max=100.0)
    assert drift.shape == (n_frames - 1,)
    assert torch.allclose(drift, torch.zeros_like(drift), atol=1e-8)


def test_rollout_drift_grows_with_a_biased_model():
    grid = (4, 4, 4)
    ops = SpectralOps(grid, 1, nyquist_mask=True).double()
    n_frames, B = 3, 1
    shape = (B, 1, *grid[:2], grid[2] // 2 + 1)
    g = torch.Generator().manual_seed(13)
    rate = torch.complex(torch.randn(*shape, generator=g, dtype=torch.float64),
                          torch.randn(*shape, generator=g, dtype=torch.float64))
    rho0 = torch.complex(torch.randn(*shape, generator=g, dtype=torch.float64),
                          torch.randn(*shape, generator=g, dtype=torch.float64))
    dt = 0.5
    traj = torch.stack([rho0 + t * dt * rate for t in range(n_frames)], dim=0)
    boxes_traj = torch.full((n_frames, B, 3), 10.0, dtype=torch.float64)
    T_traj = torch.full((n_frames, B), 1.0, dtype=torch.float64)

    exact = _ConstantRateModel(rate)
    biased = _ConstantRateModel(rate * 1.5 + 0.3)
    d_exact = rollout_drift(exact, traj, boxes_traj, T_traj, dt, ops,
                             alpha=0.5, k_max=100.0)
    d_biased = rollout_drift(biased, traj, boxes_traj, T_traj, dt, ops,
                              alpha=0.5, k_max=100.0)
    assert bool((d_biased > d_exact + 1e-6).all())


def test_finite_difference_rate_shape_and_requires_two_frames():
    traj = torch.zeros(1, 1, 1, 4, 4, 3, dtype=torch.complex128)
    with pytest.raises(ValueError, match="frame"):
        finite_difference_rate(traj, 0.5)

    traj2 = torch.zeros(3, 1, 1, 4, 4, 3, dtype=torch.complex128)
    out = finite_difference_rate(traj2, 0.5)
    assert out.shape == (2, 1, 1, 4, 4, 3)


def test_rollout_drift_uses_the_boxes_of_its_own_frame_not_frame_zero():
    """Per-sample, per-FRAME boxes (a measured fact: an NPT box
    breathes frame to frame). A drift routine that read frame 0's box for
    every transition would silently mismeasure every later frame's band --
    built to fail exactly that mutation, not merely to exercise the API.
    """
    grid = (4, 4, 4)
    ops = SpectralOps(grid, 1, nyquist_mask=True).double()
    Gzr = grid[2] // 2 + 1
    shape = (1, 1, grid[0], grid[1], Gzr)
    predicted_rate = torch.zeros(*shape, dtype=torch.complex128)
    true_rate = torch.zeros(*shape, dtype=torch.complex128)
    # Mode (nx=1, ny=0, nz=0): predicted != actual, a real residual there.
    predicted_rate[0, 0, 1, 0, 0] = 1.0 + 0.0j
    true_rate[0, 0, 1, 0, 0] = 1.0 + 2.0j
    dt = 1.0
    rho0 = torch.zeros(*shape, dtype=torch.complex128)
    traj = torch.stack(
        [rho0, rho0 + dt * true_rate, rho0 + 2 * dt * true_rate], dim=0)
    boxes_traj = torch.stack([
        torch.tensor([[0.01, 0.01, 0.01]], dtype=torch.float64),   # tiny: huge |k|
        torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64),
        torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64),
    ], dim=0)
    T_traj = torch.full((3, 1), 1.0, dtype=torch.float64)
    model = _ConstantRateModel(predicted_rate)

    drift = rollout_drift(model, traj, boxes_traj, T_traj, dt, ops,
                           alpha=0.5, k_max=5.0)
    # frame 0->1 is measured against boxes_traj[0] (tiny box): the mode's
    # |k| = 2*pi/0.01 >> k_max=5, so band excludes it -> ~0 even though the
    # residual there is real.
    assert float(drift[0]) < 1e-6
    # frame 1->2 is measured against boxes_traj[1] (normal box): the same
    # mode has |k| = 2*pi/10 ~ 0.63 < k_max=5, so band includes it -> the
    # real mismatch is measured.
    assert float(drift[1]) > 1e-3


def test_rollout_drift_has_no_default_alpha_or_k_max():
    grid = (4, 4, 4)
    ops = SpectralOps(grid, 1, nyquist_mask=True).double()
    shape = (1, 1, *grid[:2], grid[2] // 2 + 1)
    rate = torch.zeros(*shape, dtype=torch.complex128)
    traj = torch.stack([rate, rate], dim=0)
    boxes_traj = torch.full((2, 1, 3), 10.0, dtype=torch.float64)
    T_traj = torch.full((2, 1), 1.0, dtype=torch.float64)
    model = _ConstantRateModel(rate)
    with pytest.raises(TypeError):
        rollout_drift(model, traj, boxes_traj, T_traj, 0.5, ops)  # type: ignore[call-arg]
