"""The one-field lattice forms: bias-free ICNN, mirror symmetry, difference Taylor basis, lattice-sum
kernel, centred kernel argument, lattice_scalar mobility, the k-modes loader and the door's rules.
"""
from __future__ import annotations

import math

import pytest
import torch

from aipf.functional.build import rung_kwargs
from aipf.functional.kernels import LatticeSumTransform, QuinticEnvelope, RadialKernelSet
from aipf.functional.local_forms import FLocal, ICNN, TaylorEnergy
from aipf.functional.nonlocal_kernel import NonlocalKernel
from aipf.mobility import Mobility
from aipf.paths import Paths
from aipf.system import AnchorRules, Functional, Mobility as MobilityDecl, System
from aipf.train.checkpoint_formats import (KMODES_LIT_BUFFERS, kmodes_model_state_dict,
                                       load_kmodes_into)

GRID = (4, 4, 8)
BOXES = torch.tensor([[2.0, 2.0, 4.0]], dtype=torch.float64)


def _rung(**over):
    kw = dict(rho_ref=[0.5], kBT_ref=1.0, h_u=1, h_g=6, R_cut=1.0,
              kernel_n_quad=64, kernel_evaluator="lattice_sum",
              mobility_prefactor="mole_fraction", mobility_shape="lattice_scalar",
              mobility_t_form="arrhenius", mobility_shape_init=math.log(0.1),
              mobility_t_ref=1.5, mobility_activation_energy_init=0.3,
              ideal_form="lattice", u_form="taylor", u_degree=8, u_parity="even",
              u_variable="difference", g_exc_form="icnn", g_symmetry="mirror",
              icnn_output_bias=False, kernel_argument="difference", rho_eps=1e-4,
              nyquist_mask=False)
    kw.update(over)
    torch.manual_seed(0)
    return NonlocalKernel(GRID, 1, 1.0, **kw).double().eval()


def _field(seed=1):
    g = torch.Generator().manual_seed(seed)
    return 0.35 + 0.3 * torch.rand((1, 1, *GRID), generator=g, dtype=torch.float64)


def test_a_bias_free_icnn_has_no_bias_key():
    assert "w2.bias" not in ICNN(1, 4, output_bias=False).state_dict()
    assert "w2.bias" in ICNN(1, 4, output_bias=True).state_dict()


def test_mirror_symmetry_is_refused_outside_the_lattice_form():
    with pytest.raises(ValueError, match="mirror"):
        FLocal(1, [0.5], kBT_ref=1.0, h_u=4, h_g=4, ideal_form="gas",
               g_symmetry="mirror")


def test_the_mirrored_entropy_is_symmetric_about_one_half():
    f = _rung().f_local
    x = torch.linspace(0.05, 0.95, 19, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        assert torch.allclose(f._g()(x), f._g()(1.0 - x), atol=1e-14)


def test_the_difference_basis_is_a_polynomial_in_rho_minus_rho_ref():
    net = TaylorEnergy(2, [0.3, 0.4], degree=4, parity="full", variable="difference")
    with torch.no_grad():
        net.coefs.copy_(torch.arange(1, net.coefs.numel() + 1, dtype=torch.float32))
    rho = torch.tensor([[0.5, 0.1], [0.3, 0.4]])
    psi = rho - torch.tensor([0.3, 0.4])
    want = sum(c * psi[:, 0] ** e[0] * psi[:, 1] ** e[1]
               for c, e in zip(net.coefs.tolist(), net.exps.tolist()))
    assert torch.allclose(net(rho), want)
    assert float(net(torch.tensor([[0.3, 0.4]]))) == 0.0


def test_an_unknown_taylor_variable_is_refused():
    with pytest.raises(ValueError, match="variable"):
        TaylorEnergy(1, [0.5], degree=4, variable="log")


def test_the_lattice_sum_needs_the_grid_and_boxes():
    rs = RadialKernelSet(1, QuinticEnvelope(1.0), hidden=4)
    with pytest.raises(NotImplementedError, match="lattice_sum.*not supported"):
        LatticeSumTransform().w_hat(rs, torch.zeros(3))


def test_the_lattice_sum_at_k0_is_the_node_sum_times_the_cell_volume():
    m = _rung()
    with torch.no_grad():
        w_hat = m.kernel.w_hat(torch.zeros(1, 4, 4, 5), grid=GRID, boxes=BOXES)
        sp = m.kernel.evaluator.spacing(GRID, BOXES[0], torch.float64)
        W = m.kernel.evaluator.w_grid(m.kernel.radial_set, GRID, sp)[0]
    assert torch.allclose(w_hat[0, 0, 0, 0, 0, 0].real, W.sum() * 0.5 * 0.5 * 0.5)


def test_the_centred_kernel_leaves_the_drift_and_shifts_mu_by_w0_rho_ref():
    a, b = _rung(), _rung(kernel_argument="density")
    b.load_state_dict(a.state_dict())
    rho, T = _field(), torch.tensor([1.2], dtype=torch.float64)
    N = 4 * 4 * 8
    with torch.no_grad():
        rho_hat = a.ops.rfft(rho) / N
        assert torch.allclose(a(rho_hat, BOXES, T), b(rho_hat, BOXES, T), atol=1e-12)
        w0 = a.kernel.w_hat(torch.zeros(1, 4, 4, 5), grid=GRID, boxes=BOXES)[0, 0, 0, 0, 0, 0].real
        shift = b.chemical_potential(rho, BOXES, T) - a.chemical_potential(rho, BOXES, T)
    assert torch.allclose(shift, torch.full_like(shift, float(w0) * 0.5), atol=1e-12)


def test_the_uniform_free_energy_carries_the_centred_quadratic():
    m = _rung()
    x = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float64)
    w0 = torch.tensor([[2.0]], dtype=torch.float64)
    T = torch.full((3,), 1.3, dtype=torch.float64)
    with torch.no_grad():
        got = m.uniform_free_energy(x, T, w0) - m.f_local.f_pointwise(x, T)
    assert torch.allclose(got, (x[:, 0] - 0.5) ** 2)


def test_the_lattice_scalar_mobility_is_the_closed_form():
    mob = Mobility(1, "mole_fraction", "lattice_scalar", "arrhenius",
                   shape_init=math.log(0.1), t_ref=1.5, activation_energy_init=0.3,
                   kB=1.0, rho_eps=1e-4)
    assert {k: tuple(v.shape) for k, v in mob.state_dict().items()} == {
        "_log_gamma": (), "_Ea_raw": ()}
    rho, T = torch.tensor([[0.3]]), torch.tensor([1.2])
    ea = torch.nn.functional.softplus(torch.tensor(0.3))
    want = 0.3 * 0.7 * 0.1 * torch.exp(-ea * (1 / 1.2 - 1 / 1.5))
    assert torch.allclose(mob(rho, T)[0, 0, 0], want)


@pytest.mark.parametrize("bad", [dict(n_species=2), dict(prefactor="partial_density")])
def test_the_lattice_scalar_mobility_is_one_field_mole_fraction(bad):
    kw = dict(n_species=1, prefactor="mole_fraction", shape="lattice_scalar",
              t_form="none", shape_init=0.0, rho_eps=1e-4)
    kw.update(bad)
    with pytest.raises(ValueError, match="lattice_scalar"):
        Mobility(**kw)


def test_a_clamp_is_refused_on_a_shape_that_has_none():
    with pytest.raises(ValueError, match="rho_eps"):
        Mobility(1, "mole_fraction", "mlp_rho", "none", rho_eps=1e-4)


def _kmodes_layout(model) -> dict:
    """A synthetic k-modes model state dict with this model's values, in the k-modes names."""
    sd = model.state_dict()
    out = {"_log_gamma": sd["_mobility._log_gamma"], "_Ea_es_raw": sd["_mobility._Ea_raw"],
           "taylor_coefs": sd["f_local.u_net.coefs"]}
    for k, v in sd.items():
        if k.startswith("kernel.radial_set.nets.0."):
            out["w_net." + k[len("kernel.radial_set.nets.0."):]] = v
        if k.startswith("f_local.g_net."):
            out["g_exc_net." + k[len("f_local.g_net."):]] = v
    return {k: v.clone() + 0.25 for k, v in out.items()}


@pytest.mark.parametrize("lightning", [False, True])
def test_the_kmodes_layouts_round_trip_strictly(lightning):
    kmodes = _kmodes_layout(_rung())
    saved = ({"state_dict": {**{f"model.{k}": v for k, v in kmodes.items()},
                             **{b: torch.zeros(2) for b in KMODES_LIT_BUFFERS}}}
             if lightning else {"model_state_dict": kmodes})
    torch.manual_seed(7)
    loaded = load_kmodes_into(_rung(), saved)
    renamed = kmodes_model_state_dict(saved)
    assert len(renamed) == len(kmodes)
    for key, value in renamed.items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_an_unknown_kmodes_key_is_refused():
    with pytest.raises(KeyError, match="no k-modes rename rule"):
        kmodes_model_state_dict({"model_state_dict": {"m_net.0.weight": torch.zeros(1)}})
    with pytest.raises(KeyError, match="LitModule"):
        kmodes_model_state_dict({"state_dict": {"optimizer_step": torch.zeros(1)}})


def _declared(**over) -> System:
    kw = dict(grid=GRID, R_cut=1.0, rho_ref=(0.5,), h_u=1, h_g=6, h_w=4, h_m=4,
              fexc_T_ref=1.0, rho_eps=1e-4, activation="gelu", g_form="icnn",
              ideal_form="lattice", f_exc_form="split", gauge_fix=False,
              enable_TlnT=False, enable_T2=False, tbasis_ortho=False,
              kernel_n_quad=64, kernel_evaluator="lattice_sum", u_degree=8,
              u_parity="even", u_variable="difference", g_symmetry="mirror",
              icnn_output_bias=False, kernel_argument="difference",
              h_g_hat=4, h_g_tilde=4, T_ref=1.5, local_input_scale=False,
              mobility_shape_init=0.0, mobility_activation_energy_init=0.3,
              nyquist_mask=False)
    kw.update(over)
    kw = {k: v for k, v in kw.items() if v is not _DROP}
    return System(
        name="demo", n_species=1, species=("A",), masses={"A": 1.0},
        atom_types={"A": 1}, table_keys={}, paths=Paths(system="demo"),
        anchor_rules=AnchorRules({}), constants={"kB": 1.0}, defaults={},
        functional=Functional(form="nonlocal_kernel", local="taylor",
                              kernel="radial_mlp", kwargs=kw),
        mobility=MobilityDecl(form="lattice_scalar", T_form="arrhenius",
                              kwargs=dict(mobility_prefactor="mole_fraction")))


_DROP = object()


def test_the_declared_lattice_system_builds_and_hands_its_clamp_to_the_mobility():
    kwargs = rung_kwargs(_declared())
    assert "mobility_rho_eps" not in kwargs
    m = NonlocalKernel(**kwargs)
    assert m._mobility.rho_eps == 1e-4 and "w2.bias" not in m.f_local.g_net.state_dict()


@pytest.mark.parametrize("key", ["u_variable", "g_symmetry", "kernel_argument",
                                 "icnn_output_bias", "nyquist_mask"])
def test_the_door_refuses_a_lattice_declaration_silent_about_a_new_knob(key):
    with pytest.raises(ValueError, match=key):
        rung_kwargs(_declared(**{key: _DROP}))


# The door asks for a quadrature and a mobility width only where a built form reads them ----------

def test_a_lattice_sum_kernel_and_a_scalar_mobility_build_without_quadrature_or_net_width():
    kwargs = rung_kwargs(_declared(kernel_n_quad=_DROP, h_m=_DROP))
    assert "kernel_n_quad" not in kwargs and "mobility_hidden" not in kwargs
    m = NonlocalKernel(**kwargs)
    assert m.kernel.kappa_r_quad is None
    assert "kernel.kappa_r_quad" not in m.state_dict()
    with pytest.raises(ValueError, match="kernel_n_quad"):
        m.kernel.kappa_eff()


def test_the_k_table_evaluator_still_needs_its_quadrature_by_name():
    decl = _declared(kernel_n_quad=_DROP, kernel_evaluator=_DROP,
                     kernel_n_k_table=9, kernel_k_table_max=4.0)
    with pytest.raises(ValueError, match="kernel_n_quad"):
        rung_kwargs(decl)


def test_a_mobility_net_still_needs_its_width_by_name():
    import dataclasses
    decl = _declared(h_m=_DROP, mobility_shape_init=_DROP)
    decl = dataclasses.replace(decl, mobility=MobilityDecl(
        form="mlp", T_form="none",
        kwargs=dict(mobility_prefactor="mole_fraction", mobility_input_ref=None)))
    with pytest.raises(ValueError, match="mobility_hidden.*'h_m'"):
        rung_kwargs(decl)


# W_hat(0) without geometry, and the density-coupled bulk sum, refuse by name ----------------------

def test_the_closed_form_pressure_refuses_the_lattice_sum_by_name():
    from aipf.diagnose.run import model_pressure
    with pytest.raises(NotImplementedError, match="closed-form pressure.*lattice_sum"):
        model_pressure(_rung(), [[0.4]], 1.2, route="closed_form")


def test_the_bulk_hessian_refuses_the_lattice_sum_by_name():
    from aipf.diagnose.run import bulk_hessian
    with pytest.raises(NotImplementedError, match="bulk Hessian.*lattice_sum"):
        bulk_hessian(_rung(), [[0.4]], 1.2, torch.float64)


def test_the_density_coupled_bulk_sum_refuses_the_centred_kernel_by_name():
    from aipf.diagnose.run import bulk_free_energy
    with pytest.raises(NotImplementedError, match="kernel_argument='difference'"):
        bulk_free_energy(_rung(), [[0.4]], 1.2)
