"""The one-field fixed-density route and the count spelling of a declared grid."""
from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from aipf.diagnose import one_field as of
from aipf.diagnose.run import _states, declared_grid, grid, grid_n
from aipf.diagnose.stages import STAGES
from test_lattice_forms import _rung


def _regular(phi, T, w=3.0):
    r = np.clip(phi, 1e-6, 1 - 1e-6)
    return w * (1 - 2 * phi) + T * (np.log(r) - np.log1p(-r))


PHI = np.linspace(1e-3, 1 - 1e-3, 2001)


def test_the_count_spelling_is_linspace_exactly():
    assert np.array_equal(grid_n((0.5, 2.0, 40)), np.linspace(0.5, 2.0, 40))
    assert np.array_equal(declared_grid({"lo": 0.5, "hi": 2.0, "n": 40}),
                          np.linspace(0.5, 2.0, 40))
    assert np.array_equal(declared_grid((0.1, 0.5, 0.1)), grid((0.1, 0.5, 0.1)))
    with pytest.raises(ValueError, match="count"):
        grid_n((0.0, 1.0, 2.5))
    with pytest.raises(ValueError, match="lo, hi"):
        declared_grid({"lo": 0.0, "hi": 1.0})


def test_the_command_line_takes_the_count_spelling():
    from aipf.cli.diagnose_cmd import _t_grid_n
    assert np.array_equal(_t_grid_n("0.5,2.0,40"), np.linspace(0.5, 2.0, 40))
    with pytest.raises(argparse.ArgumentTypeError):
        _t_grid_n("0.5,2.0")


def test_the_one_field_stage_is_a_stage_and_the_isobar_names_it():
    assert "one_field_phase_diagram" in STAGES
    with pytest.raises(NotImplementedError, match="one_field_phase_diagram"):
        _states(np.zeros(3), np.ones(3), 1)


def test_a_regular_solution_gives_its_analytic_spinodal_and_symmetric_binodal():
    T, w = 1.2, 3.0          # spinodal: phi (1 - phi) = T / (2 w)
    lo, hi = of.spinodal_at(_regular, T, PHI)
    want = 0.5 - np.sqrt(0.25 - T / (2 * w))
    assert abs(lo - want) < 1e-6 and abs(hi - (1 - want)) < 1e-6
    left, right = of.binodal_at(_regular, T, PHI, min_width=0.02, centre=0.5,
                                bracket_eps=1e-6)
    assert abs(left + right - 1.0) < 1e-9 and left < lo
    assert of.binodal_at(_regular, 1.6, PHI, min_width=0.02, centre=0.5,
                         bracket_eps=1e-6) is None


def test_the_dome_fit_recovers_its_own_power_law():
    Ts = np.linspace(0.5, 1.3, 12)
    half = 0.5 * 1.7 * (1 - Ts / 1.45) ** 0.325
    fit = of.fit_dome(Ts, 0.5 - half, 0.5 + half, T_max=1.4, beta=0.325, B0=1.5,
                      Tc0_floor=1.42, Tc0_offset=0.05, B_bounds=(0.1, 5.0),
                      Tc_bounds_offset=(1e-3, 2.0), maxfev=10000)
    assert abs(fit["T_c"] - 1.45) < 1e-6 and abs(fit["B"] - 1.7) < 1e-6


def test_the_curvature_tc_zeroes_the_uniform_curvature():
    m = _rung()
    w0 = m.kernel.w_hat_zero_radial(1.0, 200)[0, 0]
    tc = of.tc_curvature(m, w0)
    mu = of.uniform_mu(m, clip=1e-6, w0=float(w0))
    h = 1e-4
    slope = (mu(np.array([0.5 + h]), tc) - mu(np.array([0.5 - h]), tc)) / (2 * h)
    assert abs(float(slope[0])) < 1e-5


def test_the_one_field_route_refuses_a_gas_model():
    from aipf.functional.nonlocal_kernel import NonlocalKernel
    m = NonlocalKernel((4, 4, 4), 1, 1.0, rho_ref=[0.5], kBT_ref=1.0, h_u=4, h_g=4,
                       R_cut=1.0, kernel_n_quad=16, kernel_n_k_table=17,
                       kernel_k_table_max=4.0, ideal_form="gas", g_exc_form="mlp",
                       mobility_shape="mlp_rho", nyquist_mask=True)
    with pytest.raises(NotImplementedError, match="lattice"):
        of.uniform_mu(m, clip=1e-6, w0=0.0)


def test_the_route_writes_the_published_keys():
    spec = dict(phi_clip=1e-6, min_width=0.02, bracket_eps=1e-6,
                mu_w0={"r_max": 1.5, "n_points": 64},
                tc_w0={"r_max": 1.0, "n_points": 64}, ising=None)
    out = of.phase_diagram(_rung(), np.linspace(0.5, 2.0, 4), PHI, spec)
    assert {"T", "binodal_L", "binodal_R", "spinodal_L", "spinodal_R", "Tc_exact",
            "ising_T_max", "ising_B", "ising_T_c", "ising_beta"} <= set(out)


# -- the "bulk_minima" read-off (scripts/04b_phase_diagram_es.py bulk_readoffs) --

def _closed_fh(w=3.0):
    from aipf.functional.square_gradient import SquareGradient
    return SquareGradient((4, 4, 4), 1, local="flory_huggins", w=w, nyquist_mask=True,
                          kappa_init=0.5, rho_eps=1e-4, kB=1.0, mobility_prefactor="mole_fraction",
                          mobility_shape="lattice_scalar", mobility_t_form="none",
                          mobility_shape_init=0.0).double()


def test_the_bulk_minima_readoff_finds_the_regular_solutions_dome():
    phi = np.linspace(1e-4, 1 - 1e-4, 2001)
    out = of.bulk_minima(_closed_fh(), np.array([1.2, 1.6]), phi, edge_trim=5, centre=0.5)
    assert set(out) == {"Tc", "T_grid", "binodal_lo", "binodal_hi", "spinodal_lo", "spinodal_hi"}
    assert out["Tc"] == 1.5
    want = 0.5 - np.sqrt(0.25 - 1.2 / 6.0)          # phi (1 - phi) = T / (2 w)
    assert abs(out["spinodal_lo"][0] - want) < 2e-3
    assert out["binodal_lo"][0] < out["spinodal_lo"][0]
    assert out["binodal_hi"][0] == 1.0 - out["binodal_lo"][0]
    assert np.isnan(out["binodal_lo"][1]) and np.isnan(out["spinodal_hi"][1])


def test_the_bulk_minima_readoff_refuses_a_model_without_a_closed_form():
    with pytest.raises(NotImplementedError, match="bulk_free_energy_curve"):
        of.bulk_minima(_rung(), np.array([1.0]), PHI, edge_trim=5, centre=0.5)
    with pytest.raises(ValueError, match="edge_trim"):
        of.bulk_minima(_closed_fh(), np.array([1.0]), PHI, edge_trim=0, centre=0.5)


def test_the_one_field_stage_refuses_an_undeclared_readoff(tmp_path):
    from aipf.diagnose.run import _one_field_stage

    class _S:
        n_species = 1
    block = {"dtype": "float64", "phi_grid": (0.1, 0.9, 0.1), "edge_trim": 1, "centre": 0.5}
    with pytest.raises(KeyError, match="readoff"):
        _one_field_stage(_closed_fh(), _S(), [1.0], {"one_field": block}, tmp_path)
    with pytest.raises(ValueError, match="bulk_minima"):
        _one_field_stage(_closed_fh(), _S(), [1.0], {"one_field": {**block, "readoff": "x"}}, tmp_path)
    with pytest.raises(KeyError, match="centre"):
        _one_field_stage(_closed_fh(), _S(), [1.0], {"one_field": {
            k: v for k, v in {**block, "readoff": "bulk_minima"}.items() if k != "centre"}}, tmp_path)
