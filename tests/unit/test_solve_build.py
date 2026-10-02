"""Tests for ``aipf.solve.build``: the solver's front door.

The contract under test is that EVERY declaration is validated by
``build``, before a model is loaded and before step one of a long run, and
that ``Solver.run`` then takes only what changes from run to run. So most
of this file is refusals, and each one asserts the message names the knob
the caller left out.

The fixtures are borrowed rather than rebuilt, following the convention
``tests/unit/test_data_index.py`` already uses for ``_demo``: there is no
``conftest.py`` in this directory, and a third toy model would be a third
thing to keep in step with the integrators.
"""
from __future__ import annotations

import pytest
import torch

from aipf.solve import TrustDomain, rollout_deterministic, rollout_sde
from aipf.solve.build import build
from aipf.solve.build import INTEGRATORS, Solver

from test_solve import GRID, _ToyModelB, _boxes, _rho_hat, _T   # noqa: E402
from test_system import _demo                                   # noqa: E402


# ---------------------------------------------------------------------------
# local helpers
# ---------------------------------------------------------------------------

def _sys():
    return _demo()


def _domain(n_species: int = 2) -> TrustDomain:
    """A trapezoid wide enough to contain the toy model's densities.

    ``_demo()`` carries no trust domain -- the package holds no system's
    corners -- so a caller declaring ``state_proj="domain"``
    builds one, which is exactly what a real experiment does.
    """
    return TrustDomain(inner=(0.05,) * n_species, outer=(4.0,) * n_species)


def _toy(n_species: int = 2, batch: int = 2, grid=GRID):
    """``model, rho_hat0, boxes, T, ops`` for a rollout of ``n_steps``."""
    model = _ToyModelB(grid, n_species)
    rho_hat0 = _rho_hat(batch, n_species, grid, model.ops)
    return model, rho_hat0, _boxes(batch, grid), _T(batch), model.ops


def _toy_below_floor(n_species: int = 2, batch: int = 2, grid=GRID):
    """The same, with a state whose real field dips well below any floor,
    so that a projection that never ran is visible in the assertion.
    """
    model = _ToyModelB(grid, n_species)
    g = torch.Generator().manual_seed(11)
    rho = 0.2 + 0.5 * torch.randn(batch, n_species, *grid, generator=g)
    N = grid[0] * grid[1] * grid[2]
    rho_hat0 = model.ops.rfft(rho) / N
    assert float(rho.min()) < 0.0          # the fixture is the point
    return model, rho_hat0, _boxes(batch, grid), _T(batch), model.ops


# ---------------------------------------------------------------------------
# no defaults: omitting a declaration is an error that names it
# ---------------------------------------------------------------------------

def test_build_has_no_default_dt_or_integrator():
    with pytest.raises(TypeError):
        build(_sys(), state_proj="floor", lo=1e-3)    # dt, integrator missing


def test_build_has_no_default_state_proj():
    with pytest.raises(TypeError):
        build(_sys(), integrator="euler", dt=1e-3, lo=1e-3, clamp_rho=None)


def test_build_refuses_an_unknown_integrator_and_names_the_admissible_ones():
    with pytest.raises(ValueError, match="integrator") as e:
        build(_sys(), integrator="rk4", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None)
    for name in INTEGRATORS:
        assert name in str(e.value)


@pytest.mark.parametrize("dt", [0.0, -1e-3, float("nan"), float("inf")])
def test_build_refuses_a_dt_that_is_not_a_positive_finite_number(dt):
    with pytest.raises(ValueError, match="dt"):
        build(_sys(), integrator="euler", dt=dt, state_proj="floor",
              lo=1e-3, clamp_rho=None)


def test_build_refuses_a_missing_clamp_rho():
    with pytest.raises(ValueError, match="clamp_rho"):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3)


def test_build_refuses_a_clamp_rho_of_zero_before_any_rollout():
    # `False`/`0`/`0.0` are refused by the integrators' own validator; the
    # point here is that `build` runs it NOW rather than on step one.
    with pytest.raises(ValueError):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=0.0)


# ---------------------------------------------------------------------------
# the three state-projection modes, all spelling their lower bound `lo`
# ---------------------------------------------------------------------------

def test_build_refuses_domain_without_lo():
    with pytest.raises(ValueError, match="lo="):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="domain",
              domain=_domain(), clamp_rho=None)


def test_build_refuses_domain_without_a_domain():
    with pytest.raises(ValueError, match="TrustDomain"):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="domain",
              lo=1e-3, clamp_rho=None)


def test_build_refuses_floor_without_lo():
    with pytest.raises(ValueError, match="lo="):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              clamp_rho=None)


def test_build_refuses_box_without_hi():
    with pytest.raises(ValueError, match="hi="):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="box",
              lo=1e-3, clamp_rho=None)


def test_build_refuses_the_floor_spelling_of_the_bound():
    # One spelling at this door: `lo`. `floor` is the ROLLOUT's name for
    # the same number, and accepting both would let a caller declare two.
    with pytest.raises(ValueError, match="lo="):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              floor=1e-3, clamp_rho=None)


def test_build_refuses_hi_outside_box_mode():
    with pytest.raises(ValueError, match="hi="):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, hi=2.0, clamp_rho=None)


def test_build_refuses_a_domain_outside_domain_mode():
    with pytest.raises(ValueError, match="domain="):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, domain=_domain(), clamp_rho=None)


def test_build_refuses_a_domain_of_the_wrong_species_count():
    # The accident aipf.solve.trust_domain exists to close: one system's
    # trapezoid declared for another. `build` has the System in hand, so
    # it can say so here rather than let the shapes meet in a rollout.
    with pytest.raises(ValueError, match="n_species"):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="domain",
              lo=1e-3, domain=_domain(3), clamp_rho=None)


def test_build_refuses_an_undeclared_state_proj_value():
    with pytest.raises(ValueError, match="state_proj"):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="trapezoid",
              lo=1e-3, clamp_rho=None)


# ---------------------------------------------------------------------------
# the noise colour, declared at the same door
# ---------------------------------------------------------------------------

def test_build_refuses_sde_without_noise_declaration():
    with pytest.raises(ValueError, match="noise_mode"):
        build(_sys(), integrator="sde", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None, kBT_noise=1.0)


def test_build_refuses_sde_without_sigma_noise():
    with pytest.raises(ValueError, match="sigma_noise"):
        build(_sys(), integrator="sde", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None, kBT_noise=1.0, noise_mode="gaussian")


def test_build_refuses_sde_without_kBT_noise():
    with pytest.raises(ValueError, match="kBT_noise"):
        build(_sys(), integrator="sde", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None, noise_mode="none", sigma_noise=None)


def test_build_refuses_a_noise_declaration_on_a_deterministic_integrator():
    with pytest.raises(ValueError, match="noise_mode"):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None, noise_mode="none", sigma_noise=None)


def test_build_refuses_a_keyword_no_rollout_accepts():
    with pytest.raises(ValueError, match="kapa_roll"):
        build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None, kapa_roll=0.5)


# ---------------------------------------------------------------------------
# what a built solver does
# ---------------------------------------------------------------------------

def test_a_built_solver_runs_and_keeps_the_floor():
    model, rho_hat0, boxes, T, ops = _toy_below_floor()
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None)
    traj = s.run(model, rho_hat0, boxes, T, n_steps=5)
    assert traj.shape[0] == 6
    N = ops.grid[0] * ops.grid[1] * ops.grid[2]
    assert float(ops.irfft(traj[-1] * N).min()) >= 1e-3 - 1e-7


def test_build_returns_a_frozen_solver_carrying_the_declaration():
    s = build(_sys(), integrator="heun", dt=2e-3, state_proj="floor",
              lo=1e-3, clamp_rho=0.25, kappa_roll=0.5)
    assert isinstance(s, Solver)
    assert (s.integrator, s.dt, s.state_proj) == ("heun", 2e-3, "floor")
    assert s.declared["lo"] == 1e-3 and s.declared["kappa_roll"] == 0.5
    with pytest.raises(Exception):
        s.dt = 1e-3                      # frozen: a declaration is not a knob
    with pytest.raises(TypeError):
        # One level down, and the level `frozen=True` does not reach on its
        # own: a plain dict here would have let the declaration be edited
        # after it was validated, which is the whole point of the freeze.
        s.declared["state_proj"] = "box"


def test_floor_mode_maps_lo_to_the_rollouts_floor():
    model, rho_hat0, boxes, T, _ = _toy_below_floor()
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=2e-3, clamp_rho=None)
    got = s.run(model, rho_hat0, boxes, T, n_steps=4)
    want = rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 4,
                                 method="euler", state_proj="floor",
                                 floor=2e-3, clamp_rho=None)
    assert torch.equal(got, want)


def test_box_mode_passes_lo_and_hi_through():
    model, rho_hat0, boxes, T, _ = _toy_below_floor()
    s = build(_sys(), integrator="heun", dt=1e-3, state_proj="box",
              lo=1e-3, hi=0.9, clamp_rho=None)
    got = s.run(model, rho_hat0, boxes, T, n_steps=3)
    want = rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 3,
                                 method="heun", state_proj="box",
                                 lo=1e-3, hi=0.9, clamp_rho=None)
    assert torch.equal(got, want)


def test_domain_mode_passes_lo_and_the_trapezoid_through():
    model, rho_hat0, boxes, T, _ = _toy()
    dom = _domain()
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="domain",
              lo=1e-3, domain=dom, clamp_rho=None)
    got = s.run(model, rho_hat0, boxes, T, n_steps=3)
    want = rollout_deterministic(model, rho_hat0, boxes, T, 1e-3, 3,
                                 method="euler", state_proj="domain",
                                 lo=1e-3, domain=dom, clamp_rho=None)
    assert torch.equal(got, want)


def test_the_sde_integrator_forwards_its_noise_declaration():
    model, rho_hat0, boxes, T, _ = _toy()
    s = build(_sys(), integrator="sde", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None, kBT_noise=1e-4,
              noise_mode="gaussian", sigma_noise=2.0)
    got = s.run(model, rho_hat0, boxes, T, n_steps=3,
                generator=torch.Generator().manual_seed(7))
    want = rollout_sde(model, rho_hat0, boxes, T, 1e-3, 3, 1e-4,
                       noise_mode="gaussian", sigma_noise=2.0,
                       state_proj="floor", floor=1e-3, clamp_rho=None,
                       generator=torch.Generator().manual_seed(7))
    assert torch.equal(got, want)


def test_a_per_run_keyword_overrides_a_declaration_for_that_run_only():
    model, rho_hat0, boxes, T, _ = _toy()
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None, save_every=1)
    assert s.run(model, rho_hat0, boxes, T, n_steps=4).shape[0] == 5
    assert s.run(model, rho_hat0, boxes, T, n_steps=4,
                 save_every=2).shape[0] == 3
    # and the declaration is unchanged for the next run
    assert s.run(model, rho_hat0, boxes, T, n_steps=4).shape[0] == 5


def test_a_per_run_keyword_no_rollout_accepts_is_refused():
    model, rho_hat0, boxes, T, _ = _toy()
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None)
    with pytest.raises(ValueError, match="kapa_roll"):
        s.run(model, rho_hat0, boxes, T, n_steps=1, kapa_roll=0.5)


# ---------------------------------------------------------------------------
# a per-run keyword is a declaration too: `run` re-runs `build`'s checks
# on the MERGED keywords, not only the membership half
# ---------------------------------------------------------------------------

def test_a_per_run_bound_outside_its_mode_is_refused_like_a_declaration():
    # `hi=` at build time is refused on a floor solver, so `hi=` at run
    # time has to be as well -- otherwise the rollout takes it, ignores
    # it, and the caller believes an upper bound was applied.
    model, rho_hat0, boxes, T, _ = _toy()
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None)
    with pytest.raises(ValueError, match="hi="):
        s.run(model, rho_hat0, boxes, T, n_steps=1, hi=2.0)


def test_a_per_run_domain_outside_domain_mode_is_refused():
    model, rho_hat0, boxes, T, _ = _toy()
    s = build(_sys(), integrator="heun", dt=1e-3, state_proj="box",
              lo=1e-3, hi=2.0, clamp_rho=None)
    with pytest.raises(ValueError, match="domain="):
        s.run(model, rho_hat0, boxes, T, n_steps=1, domain=_domain())


def test_a_per_run_domain_of_the_wrong_species_count_is_refused():
    model, rho_hat0, boxes, T, _ = _toy()
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="domain",
              lo=1e-3, domain=_domain(), clamp_rho=None)
    with pytest.raises(ValueError, match="n_species"):
        s.run(model, rho_hat0, boxes, T, n_steps=1, domain=_domain(3))


def test_a_per_run_clamp_rho_of_zero_is_refused():
    model, rho_hat0, boxes, T, _ = _toy()
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None)
    with pytest.raises(ValueError):
        s.run(model, rho_hat0, boxes, T, n_steps=1, clamp_rho=0.0)


def test_a_legitimate_per_run_override_still_runs():
    # Re-validating is not meant to make `run` refuse overrides.
    model, rho_hat0, boxes, T, _ = _toy()
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None)
    assert s.run(model, rho_hat0, boxes, T, n_steps=6,
                 save_every=3).shape[0] == 3
    assert s.run(model, rho_hat0, boxes, T, n_steps=2,
                 kappa_roll=0.0, check_finite=False).shape[0] == 3
    # a tighter floor for one run only, and then the declared one again
    tight = s.run(model, rho_hat0, boxes, T, n_steps=2, lo=0.35)
    assert not torch.equal(tight,
                           s.run(model, rho_hat0, boxes, T, n_steps=2))


def test_a_per_run_generator_on_an_sde_solver_still_runs():
    model, rho_hat0, boxes, T, _ = _toy()
    s = build(_sys(), integrator="sde", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None, kBT_noise=1e-4,
              noise_mode="none", sigma_noise=None)
    a = s.run(model, rho_hat0, boxes, T, n_steps=2,
              generator=torch.Generator().manual_seed(5))
    b = s.run(model, rho_hat0, boxes, T, n_steps=2,
              generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def test_run_refuses_a_state_whose_species_count_is_not_the_systems():
    model, rho_hat0, boxes, T, _ = _toy(n_species=1)
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None)          # _demo() has n_species=2
    with pytest.raises(ValueError, match="n_species"):
        s.run(model, rho_hat0, boxes, T, n_steps=1)


def test_the_same_solver_runs_twice_and_gives_the_same_trajectory():
    model, rho_hat0, boxes, T, _ = _toy()
    s = build(_sys(), integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None)
    a = s.run(model, rho_hat0, boxes, T, n_steps=3)
    b = s.run(model, rho_hat0, boxes, T, n_steps=3)
    assert torch.equal(a, b)


def test_build_refuses_a_system_that_is_not_a_System():
    with pytest.raises(TypeError, match="System"):
        build(None, integrator="euler", dt=1e-3, state_proj="floor",
              lo=1e-3, clamp_rho=None)
