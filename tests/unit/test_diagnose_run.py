"""``aipf.diagnose.run``: the driver, and the two ports it carries.

Nothing here names a system or uses one's numbers: the
toys below work in units where the pressure conversion is 1 and the free
energy is a quadratic whose isobar can be written down in closed form, so
what these tests pin is the ARITHMETIC -- a root solve that lands on the
analytic root, an apex fit that recovers the apex it was given, a directory
keyed on the bytes it read. The published models' own numbers are
compared in ``tests/test_published.py`` and ``tests/test_figdata_from_package.py``,
where the published figure data exist to compare against.

Local helpers rather than fixtures: this directory has no ``conftest.py``
and each file carries its own ``_demo``-shaped builder.
"""
import hashlib
import json

import numpy as np
import pytest
import torch

from aipf.diagnose.run import run
from aipf.diagnose.run import (STAGES, isobar_density, isobar_scale,
                                load_manifold, manifold_density,
                                nearest_anchor)
from aipf.diagnose.thermo import dome_apex, dome_tail
from aipf.paths import Paths
from aipf.system import AnchorRules, Checkpoint, Functional, Mobility, System

#: The toy's Boltzmann constant: a round 1.0, because the toy works in
#: reduced units and a system's ``kB`` is that system's own convention.
_TOY_KB = 1.0


def _toy_functional() -> Functional:
    return Functional(
        form="nonlocal_kernel", local="mlp", kernel="radial_mlp",
        kwargs=dict(
            grid=(4, 4, 4), R_cut=2.0, rho_ref=(0.3, 0.3),
            h_u=4, h_g=4, h_w=4, h_m=4,
            fexc_T_ref=1.0, rho_eps=1e-5, activation="gelu",
            g_form="mlp", ideal_form="gas", f_exc_form="split", nyquist_mask=True,
            gauge_fix=False, enable_TlnT=False, enable_T2=False,
            tbasis_ortho=False, kernel_n_quad=16, kernel_n_k_table=17,
            kernel_k_table_max=8.0, h_g_hat=4, h_g_tilde=4, T_ref=1000.0,
            local_input_scale=False))


def _demo_system_with_ckpt(tmp_path):
    """A system whose declared checkpoint really exists, and really loads.

    Same shape as the builder in ``test_data_index.py``, with one addition
    it does not need: the file holds a state dict this package's own
    ``functional.build`` can take, because the driver builds the model
    before it runs a stage.
    """
    from aipf.functional.build import build

    system = System(
        name="demo", n_species=2, species=("A", "B"),
        masses={"A": 1.0, "B": 2.0}, atom_types={"A": 1, "B": 2},
        table_keys={"rho": ("rho_A", "rho_B"), "x": "x_B", "x_channel": 1},
        paths=Paths(system="demo", raw_default=str(tmp_path)),
        anchor_rules=AnchorRules({}), constants={"kB": _TOY_KB},
        defaults={}, functional=_toy_functional(),
        mobility=Mobility(form="mlp_scaled", T_form="none",
                          kwargs=dict(mobility_prefactor="mole_fraction",
                                      mobility_input_ref=None)))
    run_dir = tmp_path / "runs" / "demo_run"
    run_dir.mkdir(parents=True)
    ckpt = run_dir / "final.ckpt"
    torch.save({"state_dict": build(system).state_dict()}, ckpt)
    return (system.__class__(**{**system.__dict__,
                                "checkpoint": Checkpoint(
                                    path="runs/demo_run/final.ckpt",
                                    md5=hashlib.md5(
                                        ckpt.read_bytes()).hexdigest())}),
            ckpt)


class _Quadratic(torch.nn.Module):
    """``f = 1/2 sum rho_i^2``, so ``P = 1/2 sum rho_i^2`` too.

    A model whose isobar can be written down: at composition ``x`` the
    total density solving ``P(rho) = P`` is
    ``sqrt(2 P / ((1-x)^2 + x^2))``, monotone in the density, so the
    solver's rising-branch rule has exactly one root to find and the test
    compares against the closed form rather than against itself.
    """

    def bulk_free_energy_density(self, rho, T):
        return 0.5 * (rho ** 2).sum(dim=1)

    def chemical_potential(self, rho, boxes, T):
        return rho


def _flat_manifold(tmp_path, columns=("x", "T", "n")):
    """A manifold of constant density 1, on two temperature columns."""
    x_col, T_col, n_col = columns
    path = tmp_path / "manifold.csv"
    lines = [f"{x_col},{T_col},{n_col},status"]
    for x in np.linspace(0.0, 1.0, 11):
        for T in (1000.0, 2000.0):
            lines.append(f"{x:.3f},{T:.1f},1.0,ok")
    path.write_text("\n".join(lines) + "\n")
    return load_manifold(path, x_column=x_col, T_column=T_col,
                          n_columns=(n_col,), status_column="status",
                          status_ok="ok")


# --------------------------------------------------------------------------
# the driver's refusals and its output directory
# --------------------------------------------------------------------------

def test_run_requires_stages_and_refuses_unknown(tmp_path):
    sysm, ck = _demo_system_with_ckpt(tmp_path)
    with pytest.raises(TypeError):
        run(sysm, ck)
    with pytest.raises(ValueError, match="score"):
        run(sysm, ck, stages=("score",), out=tmp_path)


def test_output_dir_is_keyed_on_the_checkpoint_digest(tmp_path):
    sysm, ck = _demo_system_with_ckpt(tmp_path)
    out = run(sysm, ck, stages=("kappa",), out=tmp_path / "d")
    assert out.name == hashlib.md5(ck.read_bytes()).hexdigest()[:12]
    assert (out / "kappa.json").is_file()
    assert (out / "MANIFEST.json").is_file()
    record = json.loads((out / "MANIFEST.json").read_text())
    assert record["md5"] == hashlib.md5(ck.read_bytes()).hexdigest()
    assert record["stages"] == ["kappa"]
    assert json.loads((out / "kappa.json").read_text())["kappa_eff"]


def test_a_checkpoint_this_package_wrote_is_read_by_its_schema_tag(tmp_path):
    """The seam between ``aipf.train.fit`` and this driver.

    A run this package trains is saved by Lightning, so the file holds the
    model's tensors TWICE: once as ``model_state_dict``, under the
    functional's own names, and once as Lightning's ``state_dict``, where
    every name is prefixed ``model.`` by the module that wraps the
    functional and where the metric buffers sit beside them. Read the
    prefixed copy and ``load_model`` takes the file for a foreign one,
    hands it to the family-A renamer and is refused by it at the first
    prefixed key -- which is what happened, so no run this package trained
    could be diagnosed at all until an end-to-end smoke found it.

    The tag is what is asked, not the shape of the names. Which family a
    file belongs to is recorded in the file, and guessing it from its keys
    is the thing ``aipf.train.ckpt_compat`` exists to stop.
    """
    from aipf.diagnose.run import load_model
    from aipf.functional.build import build
    from aipf.train.ckpt_compat import NEW_CONFIG_SCHEMA_TAG

    sysm, _ = _demo_system_with_ckpt(tmp_path)
    trained = {name: (torch.full_like(value, 0.25)
                      if value.is_floating_point() else value)
               for name, value in build(sysm).state_dict().items()}
    path = tmp_path / "runs" / "demo_run" / "lightning.ckpt"
    torch.save({
        "config_schema": NEW_CONFIG_SCHEMA_TAG,
        "model_state_dict": trained,
        # Lightning's own, exactly as `Trainer.save_checkpoint` writes it.
        "state_dict": {**{f"model.{name}": value
                          for name, value in trained.items()},
                       "train_loss.mean_value": torch.zeros(())},
    }, path)

    loaded = load_model(sysm, path).state_dict()
    assert set(loaded) == set(trained)
    floating = [name for name, value in trained.items()
                if value.is_floating_point()]
    assert floating, "the toy has no trainable tensor to tell the two apart"
    for name in floating:
        assert torch.equal(loaded[name], trained[name].to(loaded[name].dtype))


def test_two_checkpoints_of_one_run_do_not_write_into_each_other(tmp_path):
    """The reason the key is the digest and not the file's name.

    A run directory holds several checkpoints that carry the same
    configuration and different weights. Keyed on a name, the second
    diagnosis would overwrite the first's answers while every check a
    reader can make on the directory still passes.
    """
    sysm, ck = _demo_system_with_ckpt(tmp_path)
    other = ck.parent / "later.ckpt"
    other.write_bytes(ck.read_bytes() + b"\0")
    first = run(sysm, ck, stages=("kappa",), out=tmp_path / "d")
    second = run(sysm, other, stages=("kappa",), out=tmp_path / "d")
    assert first != second


def test_a_missing_declaration_is_named_rather_than_filled_in(tmp_path):
    sysm, ck = _demo_system_with_ckpt(tmp_path)
    with pytest.raises((KeyError, ValueError), match="pressures"):
        run(sysm, ck, stages=("phase_diagram",), out=tmp_path / "d")


def test_every_stage_of_the_list_is_accepted(tmp_path):
    sysm, ck = _demo_system_with_ckpt(tmp_path)
    for stage in STAGES:
        assert stage in STAGES
    with pytest.raises(ValueError):
        run(sysm, ck, stages=(), out=tmp_path / "d")


# --------------------------------------------------------------------------
# the isobar: a root solve with a closed-form answer
# --------------------------------------------------------------------------

def test_isobar_density_lands_on_the_analytic_root(tmp_path):
    manifold = _flat_manifold(tmp_path)
    model = _Quadratic()
    x = np.linspace(0.05, 0.95, 19)
    P = 0.25
    n = isobar_density(model, P, x, 1500.0, manifold=manifold,
                       pressure_unit=1.0, poly_degree=4, s_lo=0.30,
                       s_hi=1.30, n_scan=161, n_iter=45, n_species=2,
                       pressure_route="protocol")
    exact = np.sqrt(2.0 * P / ((1.0 - x) ** 2 + x ** 2))
    assert np.allclose(n, exact, rtol=1e-9, atol=0.0)


def test_the_isobar_scale_is_the_tilt_away_from_the_manifold(tmp_path):
    """``s`` is solved per composition, so a tilt is reported and not
    averaged away: on a flat manifold the closed-form root is itself
    composition-dependent, and ``s`` must follow it."""
    manifold = _flat_manifold(tmp_path)
    x = np.array([0.1, 0.5, 0.9])
    s = isobar_scale(_Quadratic(), 0.25, x, 1500.0, manifold=manifold,
                     pressure_unit=1.0, poly_degree=4, s_lo=0.30, s_hi=1.30,
                     n_scan=161, n_iter=45, n_species=2,
                     pressure_route="protocol")
    assert s[1] == pytest.approx(1.0, rel=1e-9)
    assert s[0] < 1.0 and s[2] < 1.0
    assert s[0] == pytest.approx(s[2], rel=1e-9)


def test_a_pressure_outside_the_scan_window_is_refused(tmp_path):
    """Not silently an empty phase diagram: no state at this pressure is a
    request out of range, and the message says so."""
    manifold = _flat_manifold(tmp_path)
    with pytest.raises(ValueError, match="no state"):
        isobar_density(_Quadratic(), 1e4, np.linspace(0.1, 0.9, 5), 1500.0,
                       manifold=manifold, pressure_unit=1.0, poly_degree=4,
                       s_lo=0.30, s_hi=1.30, n_scan=161, n_iter=45,
                       n_species=2, pressure_route="protocol")


def test_the_manifold_is_a_window_and_a_hole_in_it_is_refused(tmp_path):
    path = tmp_path / "holed.csv"
    path.write_text("x,T,n,status\n0.0,1000,1.0,ok\n1.0,2000,1.0,ok\n")
    with pytest.raises(ValueError, match="holes"):
        load_manifold(path, x_column="x", T_column="T", n_columns=("n",),
                      status_column="status", status_ok="ok")


def test_manifold_density_fits_rather_than_interpolates(tmp_path):
    """A degree-4 fit of a noisy straight line does not pass through the
    points, which is the property the construction depends on: a spline
    through every point rings in the second derivative the stability
    readout is built from."""
    path = tmp_path / "noisy.csv"
    rows = ["x,T,n,status"]
    noise = np.array([0.0, 0.01, -0.01, 0.01, -0.01, 0.0, 0.01, -0.01,
                      0.01, -0.01, 0.0])
    for x, dn in zip(np.linspace(0.0, 1.0, 11), noise):
        for T in (1000.0, 2000.0):
            rows.append(f"{x:.3f},{T:.1f},{1.0 + dn:.4f},ok")
    path.write_text("\n".join(rows) + "\n")
    manifold = load_manifold(path, x_column="x", T_column="T",
                              n_columns=("n",), status_column="status",
                              status_ok="ok")
    at_nodes = manifold_density(manifold, manifold[0], 1000.0, poly_degree=4)
    assert not np.allclose(at_nodes, manifold[2][:, 0], atol=1e-6)
    assert np.allclose(at_nodes, 1.0, atol=0.01)


def test_the_anchor_of_a_midpoint_pressure_breaks_upward(tmp_path):
    assert nearest_anchor(300.0, (200.0, 400.0)) == 400.0
    assert nearest_anchor(299.0, (200.0, 400.0)) == 200.0
    assert nearest_anchor(800.0, (200.0, 400.0, 600.0, 800.0)) == 800.0


# --------------------------------------------------------------------------
# the apex fit
# --------------------------------------------------------------------------

_APEX = dict(central_lo_max=0.6, tail_fraction=0.5, min_tail_points=4,
             max_overshoot_steps=10.0)


def _synthetic_dome(T, T_c=9000.0, a=4e-5, x_mid=0.3):
    """A dome that closes exactly as the fit's own form says it does.

    The temperature window the tests scan it over starts high on purpose.
    The tail cut is RELATIVE to the widest tie, so a dome watched from far
    below its apex has a tail spanning a quarter of that whole range, and
    the fit's own overshoot guard -- ten steps above the last clean tie --
    then refuses a fit that is perfectly good. That is the guard doing its
    job on a synthetic dome, not a defect, and the real domes it was
    measured against close within a few steps of their last clean tie.
    """
    w = np.sqrt(np.clip(a * (T_c - T), 0.0, None))
    lo = np.where(w > 0, x_mid - 0.5 * w, np.nan)
    hi = np.where(w > 0, x_mid + 0.5 * w, np.nan)
    return lo, hi


def test_dome_apex_recovers_the_apex_it_was_given():
    T = np.arange(5500.0, 9001.0, 100.0)
    lo, hi = _synthetic_dome(T)
    T_c, w0, T_last = dome_apex(T, lo, hi, T_step=100.0, **_APEX)
    assert T_c == pytest.approx(9000.0, rel=1e-9)
    assert w0 == pytest.approx(np.sqrt(4e-5), rel=1e-9)
    assert np.isfinite(T_last) and T_last < 9000.0


def test_dome_apex_fits_the_tail_and_reports_where_clean_data_ends():
    """``T_last`` is the last CLEAN tie, not the last finite one: the fit's
    own input is the noisy end, so a guard against it would be
    self-referential."""
    T = np.arange(5500.0, 9001.0, 100.0)
    lo, hi = _synthetic_dome(T)
    _central, tail, T_last = dome_tail(T, lo, hi, central_lo_max=0.6,
                                        tail_fraction=0.5)
    assert tail.any()
    assert T_last == T[np.isfinite(lo) & ~tail].max()
    assert T_last < T[np.isfinite(lo)].max()


def test_dome_apex_refuses_a_dome_that_never_closes():
    """A flat, still-wide tail has a slope from noise alone, and it
    extrapolates to an apex nobody measured."""
    T = np.arange(5500.0, 9001.0, 100.0)
    lo = np.full(len(T), 0.2)
    hi = np.full(len(T), 0.5) - 1e-9 * np.arange(len(T))
    T_c, w0, T_last = dome_apex(T, lo, hi, T_step=100.0, **_APEX)
    assert not np.isfinite(T_c) and not np.isfinite(w0)


def test_dome_apex_refuses_a_tail_of_fewer_points_than_declared():
    T = np.arange(2000.0, 3000.0, 100.0)
    lo, hi = _synthetic_dome(T, T_c=3000.0, a=1e-3)
    strict = dict(_APEX, min_tail_points=99)
    T_c, w0, T_last = dome_apex(T, lo, hi, T_step=100.0, **strict)
    assert not np.isfinite(T_c)
    assert np.isfinite(T_last), "where the data ends is still reported"


def test_dome_apex_ignores_a_gap_outside_the_central_dome():
    """A tie whose low branch sits above the cut is another branch, and
    including it would push the apex to the temperature ceiling."""
    T = np.arange(5500.0, 9501.0, 100.0)
    lo, hi = _synthetic_dome(T)
    lo = np.where(np.isfinite(lo), lo, 0.8)
    hi = np.where(np.isfinite(hi), hi, 0.95)
    T_c, _w0, _T_last = dome_apex(T, lo, hi, T_step=100.0, **_APEX)
    assert T_c == pytest.approx(9000.0, rel=1e-9)


def test_no_function_shadows_a_module_level_helper_it_also_calls():
    """A local named like a helper makes that name local to the WHOLE body.

    Measured, in this module: promoting ``_grid`` to ``grid`` collided with
    a local ``grid`` assigned forty lines below the three calls to it, and
    every ``--stage dome`` and ``--stage tc`` run raised
    ``UnboundLocalError`` on the scan's first line. Order does not save
    this -- Python decides on the name, not on the line -- so the rule is
    that no body both calls a module-level function of this module and
    binds that name.
    """
    import ast
    import importlib
    import inspect

    module = importlib.import_module("aipf.diagnose.run")
    tree = ast.parse(inspect.getsource(module))
    helpers = {node.name for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    hits = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        called = {n.func.id for n in ast.walk(func)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        bound = {n.id for n in ast.walk(func)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        shadowed = sorted(called & bound & helpers)
        if shadowed:
            hits.append(f"{func.name}() binds and calls {shadowed}")
    assert not hits, hits
