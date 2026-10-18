"""Three channels through the model code: the pair kernel, the mobility and the kernel hinge.

Every other model test runs at one or two channels. The declaration here is Fe-B-shaped
(``experiments/feb/system.py``: ideal gas, a joint excess network, radial pair kernels on a k table,
``mlp_scaled`` mobility with the ``partial_density`` sandwich and one Arrhenius barrier per channel)
at three channels, with small widths and grid so it runs on a CPU in seconds. It names nothing real.

A state with one channel identically zero is a binary state of the three-channel model; it has to
stay finite, and its absent channel has to drop out of the mobility exactly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aipf.functional.build import build
from aipf.losses.extras import eigmin_sym2, l_w
from aipf.mobility import build as build_mobility
from aipf.mobility import cholesky_raw_to_matrix
from aipf.paths import Paths
from aipf.system import AnchorRules, Functional, Mobility, System
from aipf.train.kernel_hinge import KernelHinge

_K_B = 8.617333262e-5  # eV/K

_RHO_REF = (0.03, 0.03, 0.03)

#: The species order is the channel order.
SPECIES = ("Fe", "Cu", "Ni")

#: Barriers per channel, eV.
_BARRIERS = (0.38, 0.50, 0.44)


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def ternary_system(functional_kwargs=None, **system_fields) -> System:
    """The Fe-B functional form at three channels; ``functional_kwargs`` replace declared kwargs."""
    fields = dict(
        name="ternary_toy", n_species=3, species=SPECIES,
        masses={"Fe": 55.845, "Cu": 63.546, "Ni": 58.693},
        atom_types={"Fe": 1, "Cu": 2, "Ni": 3},
        table_keys={"rho": ("rho_Fe", "rho_Cu", "rho_Ni"), "x": "x_Fe"},
        paths=Paths(system="ternary_toy"),
        anchor_rules=AnchorRules({}),
        constants={"kB": _K_B},
        defaults={},
        functional=Functional(
            form="nonlocal_kernel", local="mlp", kernel="radial_mlp",
            kwargs=dict(
                grid=(6, 6, 6), R_cut=4.5, rho_ref=_RHO_REF,
                h_u=4, h_g=4, h_w=6, h_m=6,
                fexc_T_ref=1800.0, T_ref=1800.0, rho_eps=1e-5, activation="gelu",
                g_form="icnn", enable_TlnT=False, enable_T2=False, tbasis_ortho=True,
                gauge_fix=False, disable_u=False, disable_g=False,
                f_exc_form="joint", h_joint=8, joint_depth=2, local_input_scale=True,
                ideal_form="gas", h_g_hat=4, h_g_tilde=4,
                tbasis_ortho_window=(_K_B * 1500.0, _K_B * 2600.0), tbasis_ortho_points=13,
                kernel_n_quad=64, kernel_n_k_table=65, kernel_k_table_max=8.0,
                nyquist_mask=True,
            ) | dict(functional_kwargs or {}),
        ),
        mobility=Mobility(
            form="mlp_scaled", T_form="arrhenius",
            kwargs=dict(
                arrhenius_shared_Ea=False,
                mobility_prefactor="partial_density",
                mobility_input_ref=_RHO_REF,
                mobility_activation_energy_init=tuple(_inv_softplus(e) for e in _BARRIERS),
            ),
        ),
    )
    fields.update(system_fields)
    return System(**fields)


def _model(seed=0):
    torch.manual_seed(seed)
    return build(ternary_system())


def _state(batch=2, grid=(6, 6, 6), absent=None, seed=0):
    """Real-space densities about ``_RHO_REF`` with 20 % fluctuations; channel ``absent`` set to 0."""
    g = torch.Generator().manual_seed(seed)
    rho = torch.tensor(_RHO_REF).view(1, 3, 1, 1, 1) \
        * (1.0 + 0.2 * (2.0 * torch.rand(batch, 3, *grid, generator=g) - 1.0))
    if absent is not None:
        rho[:, absent] = 0.0
    return rho


def _boxes(batch=2):
    return torch.full((batch, 3), 12.0)


def _T(batch=2):
    return torch.tensor([1500.0, 2100.0])[:batch]


def _finite_grads(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    for grad in grads:
        assert torch.isfinite(grad).all()
    return grads


# ---------------------------------------------------------------------------
# 1. NonlocalKernel at three channels
# ---------------------------------------------------------------------------

def test_the_three_channel_declaration_builds():
    model = _model()
    assert model.n_species == 3
    assert model.kernel.n_species == 3
    assert len(model.kernel.radial_set.nets) == 6


@pytest.mark.parametrize("absent", [None, 0, 1, 2], ids=["ternary", "no-Fe", "no-Cu", "no-Ni"])
def test_forward_is_finite_with_finite_gradients(absent):
    model = _model()
    rho = _state(absent=absent)
    ops = model._ops_for_real(rho)
    N = math.prod(ops.grid)
    rho_hat = ops.rfft(rho) / N
    out = model(rho_hat, _boxes(), _T())
    assert out.shape == rho_hat.shape
    assert torch.isfinite(torch.view_as_real(out)).all()
    torch.view_as_real(out).pow(2).sum().backward()
    _finite_grads(model)
    if absent is not None:
        # nothing flows into or out of an absent channel: its M row and column vanish
        assert torch.equal(out[:, absent], torch.zeros_like(out[:, absent]))


@pytest.mark.parametrize("absent", [None, 1], ids=["ternary", "no-Cu"])
def test_chemical_potential_and_bulk_free_energy_have_the_expected_shapes(absent):
    model = _model()
    rho = _state(absent=absent)
    mu = model.chemical_potential(rho, _boxes(), _T())
    f = model.bulk_free_energy_density(rho, _T())
    assert mu.shape == rho.shape
    assert f.shape == (2, 1, 6, 6, 6)
    assert torch.isfinite(mu).all() and torch.isfinite(f).all()
    present = [c for c in range(3) if c != absent]
    (mu[:, present].sum() + f.sum()).backward()
    _finite_grads(model)


def test_the_pair_kernel_is_symmetric_with_the_six_pair_nets_in_their_slots():
    """``W_hat(k)[i, j]`` is the sine transform of pair net ``(min(i, j), max(i, j))``, recomputed
    in float64 numpy on the evaluator's quadrature at k-table nodes, where the interpolation is exact."""
    model = _model()
    kernel = model.kernel
    evaluator = kernel.evaluator
    table = evaluator.k_table
    k = table[torch.tensor([0, 3, 10, 40, 64])]
    W = kernel.w_hat(k).detach()
    assert W.shape == (5, 3, 3)
    assert torch.equal(W, W.transpose(-1, -2))

    # the slot of each unordered pair, written out rather than read from the code
    slot = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}
    r = evaluator.r_quad.double().numpy()
    with torch.no_grad():
        w = kernel.radial_set.w_of_r(evaluator.r_quad).double().numpy()     # (6, Q)
    kk = k.double().numpy()
    sinc = np.sinc(np.outer(kk, r) / np.pi)                                 # sin(kr)/(kr)
    f = r ** 2 * w[:, None, :] * sinc[None]                                # (6, 5, Q)
    expected = 4.0 * np.pi * 0.5 * np.sum((f[..., 1:] + f[..., :-1]) * np.diff(r), axis=-1)
    scale = np.abs(expected).max()
    for i in range(3):
        for j in range(3):
            got = W[:, i, j].double().numpy()
            assert np.allclose(got, expected[slot[(min(i, j), max(i, j))]],
                               rtol=0.0, atol=1e-5 * scale), (i, j)
    # and the six slots are six different functions
    assert len({tuple(np.round(row / scale, 6)) for row in expected}) == 6


# ---------------------------------------------------------------------------
# 2. Mobility at three channels
# ---------------------------------------------------------------------------

def _mobility(seed=0):
    torch.manual_seed(seed)
    return build_mobility(ternary_system())


def _uniform_rows(n=64, absent=None, seed=3):
    g = torch.Generator().manual_seed(seed)
    rho = torch.tensor(_RHO_REF) * (0.2 + 1.6 * torch.rand(n, 3, generator=g))
    if absent is not None:
        rho[:, absent] = 0.0
    T = 1200.0 + 1400.0 * torch.rand(n, generator=g)
    return rho, T


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mobility_is_symmetric_positive_semidefinite(seed):
    M = _mobility(seed)(*_uniform_rows(seed=seed + 10)).detach()
    assert M.shape == (64, 3, 3)
    assert torch.equal(M, M.transpose(-1, -2))
    lam = np.linalg.eigvalsh(M.double().numpy())
    assert np.all(lam[:, 0] >= -1e-6 * lam[:, -1])


@pytest.mark.parametrize("absent", [0, 1, 2])
def test_an_absent_channel_has_a_zero_mobility_row_and_column(absent):
    mobility = _mobility()
    rho, T = _uniform_rows(absent=absent)
    M = mobility(rho, T)
    assert torch.isfinite(M).all()
    assert torch.equal(M[:, absent, :], torch.zeros_like(M[:, absent, :]))
    assert torch.equal(M[:, :, absent], torch.zeros_like(M[:, :, absent]))
    present = [c for c in range(3) if c != absent]
    block = M[:, present][:, :, present].detach().double().numpy()
    assert np.all(np.linalg.eigvalsh(block)[:, 0] > 0.0)
    M.sum().backward()
    for p in mobility.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_the_mobility_is_the_square_root_sandwich_of_its_shape():
    """``M_ab = M_tilde_ab sqrt(rho_a rho_b) sqrt(c_a c_b)`` at three channels, recomputed from the
    shape network's Cholesky factor."""
    mobility = _mobility()
    rho, T = _uniform_rows(n=16)
    with torch.no_grad():
        M = mobility(rho, T).double()
        M_tilde = cholesky_raw_to_matrix(mobility._m_tilde_raw(rho, T), 3).double()
        E = torch.nn.functional.softplus(mobility._activation_energy_raw).double()
    c = torch.exp(-E * (1.0 / (_K_B * T.double()[:, None]) - 1.0 / (_K_B * 1800.0)))
    s = torch.sqrt(rho.double() * c)
    assert torch.allclose(M, M_tilde * s[:, :, None] * s[:, None, :], rtol=1e-5, atol=0.0)


def test_the_arrhenius_factor_has_one_barrier_per_channel():
    """``M_ab(T) = M_ab(T_ref) sqrt(c_a c_b)``, ``c_a = exp(-E_a (1/kT - 1/kT_ref))``, with the three
    declared barriers."""
    mobility = _mobility()
    E = torch.nn.functional.softplus(mobility._activation_energy_raw).detach()
    assert E.shape == (3,)
    assert torch.allclose(E.double(), torch.tensor(_BARRIERS, dtype=torch.float64), atol=1e-6)
    rho, _ = _uniform_rows(n=5)
    T_ref = torch.full((5,), 1800.0)
    T = torch.full((5,), 1400.0)
    with torch.no_grad():
        ratio = (mobility(rho, T) / mobility(rho, T_ref)).double()
    c = torch.exp(-torch.tensor(_BARRIERS, dtype=torch.float64)
                  * (1.0 / (_K_B * 1400.0) - 1.0 / (_K_B * 1800.0)))
    expected = torch.sqrt(c[:, None] * c[None, :])
    assert torch.allclose(ratio, expected.expand_as(ratio), rtol=1e-4)


# ---------------------------------------------------------------------------
# 5. The kernel hinge through eigvalsh
# ---------------------------------------------------------------------------

def test_the_hinge_reads_the_smallest_eigenvalue_numpy_finds():
    model = _model()
    hinge = KernelHinge(kappa=0.5, k_max=3.0, n_k=12, margin=0.0)
    k = hinge.wavenumbers(dtype=torch.float32)
    with torch.no_grad():
        W = model.kernel.w_hat(torch.cat([k.new_zeros(1), k]))
    D = (W[1:] - W[0]).double().numpy()
    lam_min = np.linalg.eigvalsh(D)[:, 0]
    expected = np.mean(np.maximum(0.5 * k.double().numpy() ** 2 - lam_min, 0.0) ** 2)
    got = float(hinge.loss(model))
    assert got == pytest.approx(expected, rel=1e-4, abs=1e-12)


def _scaled_pair_kernel(model, scale):
    """Scale every pair net's output layer, so ``W`` and ``W_hat(k) - W_hat(0)`` scale with it."""
    with torch.no_grad():
        for net in model.kernel.radial_set.nets:
            net[-1].weight.mul_(scale)
            net[-1].bias.mul_(scale)


def test_a_violated_hinge_is_positive_with_a_finite_gradient():
    """``W = 0`` (every pair net's output layer zeroed) violates the hinge whatever the seed:
    ``lambda_min = 0 < kappa k^2``. The output layers still get a gradient, the hidden activations."""
    model = _model()
    hinge = KernelHinge(kappa=0.5, k_max=3.0, n_k=12, margin=0.0)
    _scaled_pair_kernel(model, 0.0)
    loss = hinge.loss(model)
    assert float(loss) > 0.0
    loss.backward()
    grads = _finite_grads(model)
    assert any(float(g.abs().max()) > 0.0 for g in grads)


def test_the_seeded_kernel_violates_the_hinge_where_numpy_says_it_does():
    model = _model()
    hinge = KernelHinge(kappa=0.5, k_max=3.0, n_k=12, margin=0.0)
    loss = hinge.loss(model)
    k = hinge.wavenumbers(dtype=torch.float32)
    W = model.kernel.w_hat(torch.cat([torch.zeros(1), k])).detach()
    lam = np.linalg.eigvalsh((W[1:] - W[0]).double().numpy())
    assert np.any(lam[:, 0] < hinge.kappa * k.double().numpy() ** 2)
    assert float(loss) > 0.0
    loss.backward()
    _finite_grads(model)


def test_the_hinge_is_zero_on_a_three_channel_kernel_that_satisfies_it():
    """``l_w`` at three channels goes through ``eigvalsh``: a stiff diagonal ``D`` plus a small
    symmetric coupling, with ``lambda_min`` above ``kappa k^2`` everywhere, gives exactly zero, and
    lowering one eigenvalue below the floor gives a positive loss with a finite gradient."""
    k = torch.linspace(0.25, 3.0, 12, dtype=torch.float64)
    coupling = torch.tensor([[0.0, 0.1, -0.05], [0.1, 0.0, 0.08], [-0.05, 0.08, 0.0]],
                            dtype=torch.float64)
    D = (k ** 2)[:, None, None] * (torch.eye(3, dtype=torch.float64) + coupling)
    assert float(l_w(D, k, kappa=0.5)) == 0.0
    D = D.clone().requires_grad_(True)
    shifted = D - 0.9 * (k ** 2)[:, None, None] * torch.eye(3, dtype=torch.float64)
    loss = l_w(shifted, k, kappa=0.5)
    assert float(loss) > 0.0
    loss.backward()
    assert torch.isfinite(D.grad).all() and float(D.grad.abs().max()) > 0.0


def test_two_channels_still_take_the_closed_form():
    """At two channels ``l_w`` takes ``eigmin_sym2``, whose value differs from ``eigvalsh`` only in
    rounding; it is the closed form that is pinned here, bit for bit."""
    g = torch.Generator().manual_seed(5)
    A = torch.randn(9, 2, 2, generator=g)
    D = A + A.transpose(-1, -2)
    k = torch.linspace(0.1, 2.0, 9)
    margin_k = 0.3 * k * k
    expected = torch.nn.functional.relu(-eigmin_sym2(D) + margin_k).pow(2).mean()
    assert torch.equal(l_w(D, k, kappa=0.3), expected)
