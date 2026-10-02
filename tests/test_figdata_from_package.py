"""The figure data computed from the published models, recomputed by the package.

Lennard-Jones:

* ``figures/fig3_lj/figdata/phase_diagram_aipf.npz``: ``aipf diagnose --system lj --ckpt published
  --stage one_field_phase_diagram`` on the declared temperature grid.
* ``phase_diagram_{fh,landau}.npz``: the published baseline models built by the package, ``mu`` by
  autograd of their closed-form bulk free energy, and the stage's own root finders
  (``aipf.diagnose.one_field.binodal_at`` / ``spinodal_at``) on the composition grid the figure
  was read off (the Landau binodal leaves [0, 1] below T ~ 1.17, so its grid runs from -1 to 2).
* ``figures/ed6_lj_free_energy/figdata/free_energy.npz``: the stage's float32 ``mu``
  (``aipf.diagnose.one_field.uniform_mu``) of the published model, integrated to ``f`` and
  differentiated to ``f''`` on the figure's grid, with the same binodal and spinodal read-off.

Iron-boron:

* ``figures/fig4_feb/figdata/gamma_map.npz``: ``aipf diagnose --system feb --ckpt published
  --stage stability_map`` at 0, 5 and 10 GPa, on every tenth row of the map's temperature grid.
* ``figures/ed3_feb_dgmix/figdata/dG_mix.npz``: ``aipf.diagnose.run.isobar_density`` and
  ``tangent_free_energy`` (g = (f + P)/n on the model's own isobar), less the chord between the
  ends.
* Not tied: ``gamma_isotherms.npz`` and ``gamma_parity.npz`` (Gamma along fixed-temperature
  lines and at the MD states, computed once with the figure) and ``md_phase_class.npz`` (MD data).

Hydrogen-helium:

* ``figures/ed5_hhe_dgmix_gamma/figdata/gamma.npz``: the ``stability_map`` stage on the figure's
  grid (97 compositions, 2000-12000 K, float64 Hessians); the system declares no stability map,
  so the test declares the figure's.
* ``figures/ed5_hhe_dgmix_gamma/figdata/dG_mix.npz``: as for iron-boron.
* ``figures/fig5_hhe/figdata/domes.npz`` and ``isopleth.npz``: the ``phase_diagram`` stage at
  400 and 800 GPa on the slices' temperature grid, and the isopleth read off it at
  x_He = 0.089. The published slices are a two-dimensional hull over a patch of density scales
  with a matrix spinodal; the stage is the one-dimensional construction on the same solved
  isobar, so they agree to a few composition steps, not to round-off.
* Not tied: ``two_phase_surface.npz`` (the dome scan, smoothed into a mesh), the schematic,
  the planet profiles and the literature curves (published data), and Extended Data Fig. 4's
  ``benchmark.json`` (a timing benchmark). Extended Data Figs. 7 and 8 are tied in
  tests/unit/test_md_analysis.py.

The iron-boron and hydrogen-helium checks solve the model's own isobar on the equation-of-state
manifold tracked under ``experiments/<system>/eos/``, so they need no raw root; every check skips,
naming the checkpoint, on a checkout without the published checkpoints. Every comparison runs on the CPU, one
thread. NaN masks must be equal; each bound is at most
three times the largest deviation measured, with its cause.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from scipy.interpolate import CubicSpline

from aipf.diagnose import one_field
from aipf.diagnose import run as run_module
from aipf.diagnose.run import load_model
from aipf.paths import repo_root
from aipf.system import load

import declared_roots

FIGDATA = repo_root() / "figures"
PD_KEYS = ("binodal_L", "binodal_R", "spinodal_L", "spinodal_R")

#: |figdata - package| per key and model, and why it is not zero.
PD_BOUND = {
    # the published arrays were written on a GPU in float32
    "aipf": {"T": 0.0, "binodal_L": 0.0, "binodal_R": 1e-15, "spinodal_L": 1e-16,
             "spinodal_R": 2e-14, "Tc_exact": 1e-6},
    # mu differentiated by hand in the published arrays, by autograd here: float64 rounding
    # (measured 1.3e-15 binodal, 2.4e-13 spinodal)
    "fh": {"T": 0.0, "binodal_L": 1e-14, "binodal_R": 1e-14, "spinodal_L": 1e-12,
           "spinodal_R": 1e-12, "Tc_exact": 0.0},
    # the published arrays took a0 = exp(log a0) and b = exp(log b) in float32, the package
    # in float64 (measured 1.6e-8 binodal, 9.2e-9 spinodal)
    "landau": {"T": 0.0, "binodal_L": 3e-8, "binodal_R": 3e-8, "spinodal_L": 2e-8,
               "spinodal_R": 2e-8, "Tc_exact": 0.0},
}
#: The composition grids the baseline diagrams were read off.
PD_GRID = {"fh": np.linspace(1e-3, 1 - 1e-3, 2001), "landau": np.linspace(-1.0, 2.0, 6001)}

#: |figdata - package| per key of free_energy.npz: the figure data were evaluated in float32 on
#: a GPU, the package in float32 on the CPU (where it equals the original code bit for bit).
#: f'' is a finite difference of mu, so the float32 noise is largest there and in the spinodal
#: it locates (measured: f 1.3e-7, f'' 9.8e-4, binodal 3.2e-7 / 1.1e-7, spinodal 2.3e-4 / 9.8e-6).
FE_BOUND = {"f": 3e-7, "curvature": 2e-3, "binodal_x": 1e-6, "binodal_f": 3e-7,
            "spinodal_x": 5e-4, "spinodal_f": 2e-5}

pytestmark = pytest.mark.slow


def _deviation(a, b, key):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, key
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{key}: NaN masks differ"
    m = ~np.isnan(b)
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_aipf_phase_diagram_is_the_diagnose_stage(tmp_path):
    declared_roots.published_or_skip("lj")
    from aipf.cli.main import main
    spec = load("lj").defaults["diagnose"]["one_field"]["T_grid"]
    code = main(["diagnose", "--system", "lj", "--ckpt", "published",
                 "--stage", "one_field_phase_diagram",
                 "--T-grid-n", f"{spec['lo']!r},{spec['hi']!r},{spec['n']}",
                 "--out", str(tmp_path)])
    assert code == 0
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    got = dict(np.load(run_dir / "one_field_phase_diagram.npz"))
    want = dict(np.load(FIGDATA / "fig3_lj" / "figdata" / "phase_diagram_aipf.npz"))
    for key, tol in PD_BOUND["aipf"].items():
        assert _deviation(got[key], want[key], key) <= tol, key


def _closed_form_mu(model):
    def mu(phi, T):
        p = torch.tensor(phi, dtype=torch.float64, requires_grad=True)
        f = model.bulk_free_energy_curve(p, float(T))
        return torch.autograd.grad(f.sum(), p)[0].numpy()
    return mu


@pytest.mark.parametrize("variant", ["fh", "landau"])
def test_baseline_phase_diagram_is_the_package_model(variant):
    declared_roots.published_or_skip("lj", variant)
    system = load("lj").variant(variant)
    model = load_model(system, system.resolve_checkpoint())
    mu = _closed_form_mu(model)
    spec = load("lj").defaults["diagnose"]["one_field"]
    want = dict(np.load(FIGDATA / "fig3_lj" / "figdata" / f"phase_diagram_{variant}.npz"))
    T = want["T"]
    grid = spec["T_grid"]
    assert np.array_equal(T, np.linspace(grid["lo"], grid["hi"], grid["n"]))
    rows = {k: [] for k in PD_KEYS}
    for t in T:
        b = one_field.binodal_at(mu, t, PD_GRID[variant], min_width=spec["min_width"],
                                 centre=0.5, bracket_eps=spec["bracket_eps"])
        s = one_field.spinodal_at(mu, t, PD_GRID[variant])
        for k, v in zip(PD_KEYS, (b or (np.nan,) * 2) + (s or (np.nan,) * 2)):
            rows[k].append(v)
    got = {"T": T, **rows, "Tc_exact": model.Tc_curvature()}
    for key, tol in PD_BOUND[variant].items():
        assert _deviation(got[key], want[key], key) <= tol, key


def test_free_energy_is_the_stage_mu():
    declared_roots.published_or_skip("lj")
    system = load("lj")
    spec = system.defaults["diagnose"]["one_field"]
    model = copy.deepcopy(load_model(system, system.resolve_checkpoint())).to(torch.float32)
    w0 = float(model.kernel.w_hat_zero_radial(**dict(spec["mu_w0"]))[0, 0].item())
    mu = one_field.uniform_mu(model, clip=spec["phi_clip"], w0=w0)
    want = dict(np.load(FIGDATA / "ed6_lj_free_energy" / "figdata" / "free_energy.npz"))
    x = want["x_A"]
    i = int(np.argmin(np.abs(x - 0.5)))
    got = {k: [] for k in FE_BOUND}
    for T in want["T"]:
        m = mu(x, T)
        f = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(x))])
        slope = (f[i + 1] - f[i - 1]) / (x[i + 1] - x[i - 1])
        f = f - f[i] - slope * (x - 0.5)          # tangent at x = 1/2 removed
        b = one_field.binodal_at(mu, T, x, min_width=spec["min_width"], centre=0.5,
                                 bracket_eps=spec["bracket_eps"])
        s = one_field.spinodal_at(mu, T, x)
        spline = CubicSpline(x, f)
        bx = np.array(b) if b else np.full(2, np.nan)
        sx = np.array(s) if s else np.full(2, np.nan)
        got["f"].append(f)
        got["curvature"].append(np.gradient(m, x))
        got["binodal_x"].append(bx)
        got["binodal_f"].append(spline(bx) if b else bx)
        got["spinodal_x"].append(sx)
        got["spinodal_f"].append(spline(sx) if s else sx)
    for key, tol in FE_BOUND.items():
        assert _deviation(np.array(got[key]), want[key], key) <= tol, key


# -- iron-boron and hydrogen-helium -------------------------------------------------------------

#: Gamma = x(1-x)/S_cc(0) is a linear solve on Hessians the package and the published file agree
#: on bit for bit (lambda_min is exact); the solve's last bits depend on the LAPACK build. Measured
#: on these rows: 4.4e-16 with pip's numpy (scipy-openblas), 7.1e-15 at P = 10 GPa with conda-forge's
#: numpy (its OpenBLAS through the reference LAPACK interface), which the aipf environment carries;
#: 1.1e-14 over the whole published map.
FEB_GAMMA_BOUND = 1e-14
#: Every tenth row of the published map's grid (1150-2650 K in steps of 25 K).
FEB_T_GRID = "1150,2650,250"
#: g = (f + P)/n summed in another order than the figure's evaluation: float64 rounding on
#: values of order 0.1-1 eV (measured 8.9e-16 Fe-B, 1.1e-14 H/He, eV per atom).
DGMIX_BOUND = {"feb": 2.5e-15, "hhe": 3e-14}
#: as FEB_GAMMA_BOUND, float64 Hessians (measured 1.3e-14)
HHE_GAMMA_BOUND = 3e-14
#: the published two-dimensional hull and matrix spinodal against the stage's one-dimensional
#: construction on the same isobar (x grid 0.005); measured over 400 and 800 GPa: binodal ends
#: 0.0187 / 0.0052, spinodal ends 0.0021 / 0.0019, T_c 18.5 K at 400 GPa, isopleth 100 K (one
#: row of the 100 K grid) on the binodal and 13.3 K on the spinodal.
DOME_BOUND = {"binodal_lo": 0.03, "binodal_hi": 0.01, "spinodal_lo": 0.004,
              "spinodal_hi": 0.004}
TC_BOUND = 40.0
ISOPLETH_BOUND = {"binodal": 200.0, "spinodal": 30.0}
X_PROTOSOLAR = 0.089


def test_feb_gamma_map_is_the_stability_map_stage(tmp_path):
    declared_roots.published_or_skip("feb")
    from aipf.cli.main import main
    argv = ["diagnose", "--system", "feb", "--ckpt", "published", "--stage", "stability_map",
            "--T-grid", FEB_T_GRID, "--out", str(tmp_path)]
    for P in (0, 5, 10):
        argv += ["--pressure", str(P)]
    assert main(argv) == 0
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    got = dict(np.load(run_dir / "stability_map.npz"))
    want = dict(np.load(FIGDATA / "fig4_feb" / "figdata" / "gamma_map.npz"))
    x_B = 1.0 - got["x_int"]
    order = np.argsort(x_B)
    assert np.array_equal(x_B[order], want["x_B"])
    rows = [int(np.flatnonzero(want["T"] == T)[0]) for T in got["T"]]
    assert len(rows) == 7
    for P in (0, 5, 10):
        lam = got[f"lam_P{P}"][:, order]
        assert np.array_equal(lam, want[f"lambda_min_P{P}"][rows], equal_nan=True), P
        dev = _deviation(got[f"Gamma_P{P}"][:, order], want[f"Gamma_P{P}"][rows], P)
        assert dev <= FEB_GAMMA_BOUND, (P, dev)


def _dgmix_deviation(name, figdir, x_key, pressures, x):
    system = load(name)
    declared = system.defaults["diagnose"]
    model = load_model(system, system.resolve_checkpoint())
    manifolds = run_module.manifolds(system, declared)
    unit = float(declared["pressure_unit"])
    want = dict(np.load(FIGDATA / figdir / "figdata" / "dG_mix.npz"))
    worst, n = 0.0, 0
    for P in pressures:
        for T in want[f"T_P{P}"]:
            dens = run_module.isobar_density(
                model, float(P), x, float(T), manifold=manifolds[float(P)], pressure_unit=unit,
                poly_degree=int(declared["poly_degree"]), n_species=system.n_species,
                **dict(declared["isobar"]))
            m = np.isfinite(dens)
            xx, nn = x[m], dens[m]
            g = run_module.tangent_free_energy(model, xx, nn, float(T), float(P),
                                               pressure_unit=unit, n_species=system.n_species)
            dG = g - (g[0] + (g[-1] - g[0]) * (xx - xx[0]) / (xx[-1] - xx[0]))
            assert np.array_equal(xx, want[f"{x_key}_P{P}_T{T:.0f}"]), (P, T)
            worst = max(worst, _deviation(dG, want[f"dG_mix_P{P}_T{T:.0f}"], (P, T)))
            n += 1
    return worst, n


def test_feb_dgmix_is_the_isobar_free_energy():
    declared_roots.published_or_skip("feb")
    worst, n = _dgmix_deviation("feb", "ed3_feb_dgmix", "x_Fe", (0, 5, 10),
                                np.linspace(1e-3, 1 - 1e-3, 601))
    assert n == 24 and worst <= DGMIX_BOUND["feb"], worst


def test_hhe_dgmix_is_the_isobar_free_energy():
    declared_roots.published_or_skip("hhe")
    worst, n = _dgmix_deviation("hhe", "ed5_hhe_dgmix_gamma", "x_He", (200, 400, 600, 800),
                                np.linspace(0.002, 0.998, 601))
    assert n == 44 and worst <= DGMIX_BOUND["hhe"], worst


def test_hhe_gamma_is_the_stability_map_stage(tmp_path):
    declared_roots.published_or_skip("hhe")
    system = load("hhe")
    declared = dict(system.defaults["diagnose"])
    declared["stability_map"] = {"x_points": (0.02, 0.98, 97), "min_points": 1,
                                 "hessian_dtype": "float64"}
    model = load_model(system, system.resolve_checkpoint())
    pressures = (200, 400, 600, 800)
    run_module._stability_map(model, system, [float(P) for P in pressures],
                              np.arange(2000.0, 12001.0, 1000.0), declared, tmp_path)
    got = dict(np.load(tmp_path / "stability_map.npz"))
    want = dict(np.load(FIGDATA / "ed5_hhe_dgmix_gamma" / "figdata" / "gamma.npz"))
    assert np.array_equal(got["x_int"], want["x_He"])
    worst = 0.0
    for P in pressures:
        assert np.array_equal(got["T"], want[f"T_P{P}"]), P
        for i, T in enumerate(got["T"]):
            worst = max(worst, _deviation(got[f"Gamma_P{P}"][i], want[f"Gamma_P{P}_T{T:.0f}"],
                                          (P, T)))
    assert 0.0 < worst <= HHE_GAMMA_BOUND, worst


def _isopleth(x_of_T, T, x_target):
    m = np.isfinite(x_of_T) & np.isfinite(T)
    xs, Ts = x_of_T[m], T[m]
    o = np.argsort(xs)
    return float(np.interp(x_target, xs[o], Ts[o]))


def test_hhe_domes_and_isopleth_are_the_phase_diagram_stage(tmp_path):
    declared_roots.published_or_skip("hhe")
    from aipf.diagnose.run import run
    system = load("hhe")
    domes = dict(np.load(FIGDATA / "fig5_hhe" / "figdata" / "domes.npz"))
    iso = dict(np.load(FIGDATA / "fig5_hhe" / "figdata" / "isopleth.npz"))
    out = run(system, system.checkpoint, stages=("phase_diagram",), out=tmp_path,
              pressures=(400.0, 800.0), T_grid=domes["T_P400"], **system.defaults["diagnose"])
    for P in (400, 800):
        got = dict(np.load(out / f"P{P}GPa" / "phase_diagram.npz"))
        assert np.array_equal(got["T"], domes[f"T_P{P}"]), P
        for key, stage in (("binodal_lo", "bin_lo"), ("binodal_hi", "bin_hi"),
                           ("spinodal_lo", "spmat_lo"), ("spinodal_hi", "spmat_hi")):
            dev = _deviation(got[stage], domes[f"{key}_P{P}"], (P, key))
            assert dev <= DOME_BOUND[key], (P, key, dev)
        for which, stage in (("binodal", "bin_lo"), ("spinodal", "spmat_lo")):
            i = int(np.flatnonzero(iso[f"{which}_P"] == P)[0])
            dev = abs(_isopleth(got[stage], got["T"], X_PROTOSOLAR) - iso[f"{which}_T"][i])
            assert dev <= ISOPLETH_BOUND[which], (P, which, dev)
    tc = float(np.load(out / "P400GPa" / "phase_diagram.npz")["Tc_fit"])
    assert abs(tc - float(domes["Tc_P400"])) <= TC_BOUND, tc
    # at 800 GPa the dome is still open at the top of its fitting window: the stage's apex guard
    # declines the fit, where the published slice fell back to a looser mean-field regression
    assert not np.isfinite(np.load(out / "P800GPa" / "phase_diagram.npz")["Tc_fit"])
    assert np.isfinite(domes["Tc_P800"])
