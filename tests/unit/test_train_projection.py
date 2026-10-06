"""A two-dimensional model trained on the ``k_z = 0`` plane of three-dimensional mode archives.

The archives are three-dimensional, as every one this package writes: labels ``(n_x, n_y, n_z)``, atom
sums ``rho_k`` and a reference cell. ``aipf.train.projection.project_kz0`` keeps the ``n_z == 0``
labels, and a window divides by ``A_ref * depth``: ``depth = Lz_ref`` gives the z mean of the density
(volumetric), ``depth = 1`` the density integrated along z (areal). What is pinned here: the projected
half spectrum is the 3D one's ``k_z = 0`` plane (times ``Lz_ref`` when areal), its real-space field is
the z mean of the 3D field, a 2D factory model trained through it recovers the parameters that made
its data, and every inconsistent combination is refused before a run directory exists. The 3D path
is pinned bit for bit by ``tests/golden``; nothing here changes it.

The toy is ``tests/toy_ndim_model.py`` with a linear local part, ``f = a rho^2 / 2``, the pair kernel
``kappa k^2`` and a constant full mobility, so its drift is ``-k^2 M (a + kappa k^2) rho_hat`` and ``M``
and ``kappa`` are identified by the drift alone. The 2D model is an effective model: a real 3D
trajectory's ``k_z != 0`` modes couple into ``k_z = 0`` (docs/reference/functional.md, "Two
dimensions"); the data here are z-invariant, so that coupling is absent by construction."""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import toy_factory_model as toy
from aipf.functional.build import build
from aipf.pipeline.modes import ModesRecord, modes_from_dump
from aipf.solve import rollout_deterministic
from aipf.train.anchors import NO_ANCHORS
from aipf.train.datamodule import (LoaderSettings, ModeDataModule, RunWeighting, SourceSpec,
                                   SplitSettings, run_weights)
from aipf.train.dataset import ModeWindowDataset, WindowSettings, read_mode_runs, scatter_modes
from aipf.train.fit import DimensionMismatch, _archive_keys, check_dimensions, fit
from aipf.train.projection import PROJECTIONS, project_kz0, projection_depth
from toy_ndim_model import build_ndim_toy

#: A stated cell with a short z edge, so a slab of a bulk run; no edge is a round binary number.
CELL = (10.3, 9.1, 4.2)
#: ``|2 pi n / L| <= 2`` on :data:`CELL`: ``|n_x| <= 3``, ``|n_y| <= 2``, ``|n_z| <= 1``.
K_CUT = 2.0
GRID3 = (8, 6, 4)
GRID2 = GRID3[:2]
PARAMS = dict(sigma=1.0, k_cut=K_CUT, atom_types=(1, 2), fields="per_type",
              ordering="lexicographic", route="separable", T=1.0, dt_frame=0.02,
              skip_frames=0, composition={"x_B": 0.5})
KEYS = _archive_keys(toy.demo_system("unused"))


@pytest.fixture(autouse=True)
def _demo_scratch_is_its_own(monkeypatch):
    """The demo declares its own raw root; a user's ``AIPF_RAW`` must not replace it."""
    monkeypatch.delenv("AIPF_RAW", raising=False)


@contextlib.contextmanager
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture(autouse=True)
def _one_thread():
    """Fields of a few dozen cells: a thread pool only adds its overhead, and on a shared machine a lot of it."""
    with _single_thread():
        yield


def _write_dump(path, *, n_frames=8, n_atoms=24, seed=0):
    """A barostatted toy run: atoms that move, a box that shrinks a little, two dump types."""
    rng = np.random.default_rng(seed)
    types = np.where(np.arange(n_atoms) % 3, 1, 2)
    lines = []
    for frame in range(n_frames):
        edge = np.asarray(CELL) * (1.0 + 0.01 * (n_frames - frame) / n_frames)
        pos = rng.random((n_atoms, 3)) * edge
        lines += ["ITEM: TIMESTEP", str(frame * 100), "ITEM: NUMBER OF ATOMS", str(n_atoms),
                  "ITEM: BOX BOUNDS pp pp pp"]
        lines += [f"0.0 {float(edge[axis])!r}" for axis in range(3)]
        lines.append("ITEM: ATOMS id type x y z")
        lines += [" ".join(repr(float(v)) for v in (i + 1, types[i], *pos[i]))
                  for i in range(n_atoms)]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def _archive(tmp_path, *, reference_box=CELL, tag="run_a", seed=0):
    """``<tree>/<tag>/modes.npz`` through the writer a real archive goes through; returns the tree."""
    tree = tmp_path / "fields" / "modes_demo"
    dump = _write_dump(tmp_path / f"{tag}.dump", seed=seed)
    modes_from_dump(dump, **PARAMS, entry_dir=tree / tag, reference_box=reference_box)
    return tree


def _read(tree):
    return read_mode_runs(tree, "run_*", keys=KEYS)


def _window(grid, estimator="weak"):
    return WindowSettings(estimator=estimator, half_width=2, n_states=3, stride=1, grid=grid,
                          savgol_window=None, savgol_poly=None)


# ---------------------------------------------------------------------------
# the projection is the k_z = 0 plane
# ---------------------------------------------------------------------------

def test_the_projected_run_keeps_the_kz0_labels_their_amplitudes_and_the_cells_xy(tmp_path):
    (run,) = _read(_archive(tmp_path))
    flat = project_kz0(run, areal=False)
    plane = run.labels[:, 2] == 0
    assert plane.sum() < run.labels.shape[0], "the archive holds k_z != 0 modes to drop"
    assert np.array_equal(flat.labels, run.labels[plane][:, :2])
    assert np.array_equal(flat.amplitudes, run.amplitudes[..., plane])
    assert flat.amplitudes.dtype == run.amplitudes.dtype
    assert np.array_equal(flat.reference_box, np.asarray(CELL)[:2])
    assert np.array_equal(flat.boxes, run.boxes[:, :2])
    assert flat.depth == CELL[2] and project_kz0(run, areal=True).depth == 1.0
    assert run.depth is None, "the run as read is left as it was"


@pytest.mark.parametrize("estimator", ["weak", "weak_mid"])
def test_a_volumetric_window_is_the_kz0_plane_of_the_3d_window_bit_for_bit(tmp_path, estimator):
    """``rho_hat = rho_k / V_ref``: the 2D half spectrum is the 3D one's ``iz = 0`` plane, cut to the
    half ``y`` axis, in every window, target and states alike."""
    (run,) = _read(_archive(tmp_path))
    three = ModeWindowDataset([run], _window(GRID3, estimator))
    two = ModeWindowDataset([project_kz0(run, areal=False)], _window(GRID2, estimator))
    Gyr = GRID2[1] // 2 + 1
    assert len(two) == len(three) > 1
    for i in range(len(three)):
        a, b = three[i], two[i]
        for key in ("target_hat", "rho_hat_states"):
            assert torch.equal(b[key], a[key][..., :Gyr, 0]), key
        assert torch.equal(b["boxes"], torch.tensor(CELL[:2], dtype=torch.float32))
        assert all(torch.equal(a[k], b[k]) for k in ("lam", "T", "sample_weight", "run_index"))


def test_an_areal_window_is_lz_times_the_kz0_plane(tmp_path):
    """``rho_hat = rho_k / (Lx_ref Ly_ref) = Lz_ref * rho_k / V_ref``, to float precision."""
    (run,) = _read(_archive(tmp_path))
    three = ModeWindowDataset([run], _window(GRID3))
    areal = ModeWindowDataset([project_kz0(run, areal=True)], _window(GRID2))
    Gyr = GRID2[1] // 2 + 1
    for i in range(len(three)):
        for key in ("target_hat", "rho_hat_states"):
            want = three[i][key][..., :Gyr, 0].to(torch.complex128) * CELL[2]
            got = areal[i][key].to(torch.complex128)
            assert torch.allclose(got, want, rtol=1e-6, atol=0), key
            assert got.abs().max() > 0


def test_the_2d_scatter_keeps_the_half_y_axis_and_folds_negative_n_x():
    """Labels with ``n_y < 0`` are the conjugate half, left out; ``n_x`` wraps ``mod Gx``."""
    labels = np.array([(0, 0), (1, 0), (-1, 0), (2, 1), (-2, 1), (1, -1), (-3, 2)])
    amps = (np.arange(len(labels)) + 1j).astype(np.complex64)[None]
    out = scatter_modes(amps, labels, 2.0, (8, 6))
    assert out.shape == (1, 8, 4) and out.dtype == np.complex64
    for (nx, ny), value in zip(labels, amps[0]):
        if ny >= 0:
            assert out[0, nx % 8, ny] == value / 2.0
    assert np.count_nonzero(out) == 6
    with pytest.raises(ValueError, match="beyond the grid"):
        scatter_modes(amps, labels, 1.0, (6, 6))
    with pytest.raises(ValueError, match=r"\(n_x, n_y\) labels"):
        scatter_modes(np.ones((1, 2)), np.zeros((2, 3), int), 1.0, (8, 6))


def test_the_z_mean_of_the_3d_field_is_the_projected_2d_field(tmp_path):
    """The real-space field of the projected modes is the z average of the 3D field the same archive
    reconstructs on its grid (volumetric), and ``Lz_ref`` times it (areal), to float precision."""
    (run,) = _read(_archive(tmp_path))
    V = math.prod(CELL)
    A = CELL[0] * CELL[1]
    half3 = scatter_modes(run.amplitudes, run.labels, V, GRID3).astype(np.complex128)
    field3 = np.fft.irfftn(half3 * math.prod(GRID3), s=GRID3, axes=(-3, -2, -1))
    for areal, scale in ((False, 1.0), (True, CELL[2])):
        flat = project_kz0(run, areal=areal)
        half2 = scatter_modes(flat.amplitudes, flat.labels, A * flat.depth, GRID2)
        field2 = np.fft.irfftn(half2.astype(np.complex128) * math.prod(GRID2), s=GRID2,
                               axes=(-2, -1))
        want = scale * field3.mean(axis=-1)
        assert np.abs(field2 - want).max() <= 1e-6 * np.abs(want).max()
    assert np.ptp(field3.mean(axis=-1)) > 0 and np.ptp(field3 - field3.mean(-1, keepdims=True)) > 0


def test_the_projected_run_weights_its_band_on_the_2d_grid(tmp_path):
    """``inverse_band_power`` reads the projected run's own ``k`` and volume: two runs whose targets
    differ by a factor have weights that are not one."""
    tree = _archive(tmp_path)
    _archive(tmp_path, tag="run_b", seed=1)
    runs = [project_kz0(r, areal=True) for r in _read(tree)]
    weighting = RunWeighting(mode="inverse_band_power", sigma=1.0, k_max=1.5, eps=1e-6,
                             probe_every=1)
    weights = run_weights(runs, weighting, _window(GRID2))
    assert weights.shape == (2,) and np.isfinite(weights).all()
    assert abs(weights.mean() - 1.0) < 1e-6 and not np.allclose(weights, 1.0)


def test_projection_depth_is_one_for_areal_and_lz_for_volumetric(tmp_path):
    tree = _archive(tmp_path)
    (run,) = _read(tree)
    for where in (run, tree / "run_a", tree / "run_a" / "modes.npz", str(tree / "run_a")):
        assert projection_depth(where, areal=True) == 1.0
        assert projection_depth(where, areal=False) == CELL[2]
    with pytest.raises(ValueError, match="already projected"):
        projection_depth(project_kz0(run, areal=False), areal=False)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_an_archive_without_a_reference_cell_is_refused_naming_the_writer(tmp_path):
    tree = _archive(tmp_path, reference_box="time_mean")
    (run,) = _read(tree)
    assert run.reference_box is None
    for call in (lambda: project_kz0(run, areal=True),
                 lambda: projection_depth(run, areal=False),
                 lambda: projection_depth(tree / "run_a", areal=True)):
        with pytest.raises(ValueError, match=r"modes_from_dump\(reference_box=\.\.\.\)"):
            call()
    dm = _datamodule(tree, "kz0-areal", GRID2)
    with pytest.raises(ValueError, match="source 'demo_src'.*no reference cell"):
        dm.setup()


def test_a_run_and_a_grid_of_another_number_of_axes_are_refused(tmp_path):
    (run,) = _read(_archive(tmp_path))
    with pytest.raises(ValueError, match="3-axis labels.*2 axes"):
        ModeWindowDataset([run], _window(GRID2))
    with pytest.raises(ValueError, match="2-axis labels.*3 axes"):
        ModeWindowDataset([project_kz0(run, areal=True)], _window(GRID3))
    with pytest.raises(ValueError, match="not a three-dimensional run"):
        project_kz0(project_kz0(run, areal=True), areal=True)


def _datamodule(tree, projection, grid):
    return ModeDataModule(
        sources=(SourceSpec("demo_src", str(tree), "run_*", grid, 1.0, ()),), keys=KEYS,
        window=_window(grid), split=SplitSettings("none", (), 0.0, 0),
        weighting=RunWeighting("uniform", None, None, None, None),
        loader=LoaderSettings(2, 0, False, False, "shuffled"), projection=projection)


@pytest.mark.parametrize("projection, grid, match", [
    ("kz0-areal", GRID3, "two-axis grid"),
    (None, GRID2, "needs projection="),
    ("kz0", GRID2, "is not one of"),
])
def test_the_datamodule_refuses_a_projection_its_grids_contradict(tmp_path, projection, grid,
                                                                  match):
    with pytest.raises(ValueError, match=match):
        _datamodule(tmp_path, projection, grid)


def test_the_datamodule_refuses_a_projection_with_no_reference_cell_key(tmp_path):
    with pytest.raises(ValueError, match="reference cell"):
        ModeDataModule(
            sources=(SourceSpec("s", str(tmp_path), "run_*", GRID2, 1.0, ()),),
            keys=KEYS.replace(reference_box=None), window=_window(GRID2),
            split=SplitSettings("none", (), 0.0, 0),
            weighting=RunWeighting("uniform", None, None, None, None),
            loader=LoaderSettings(2, 0, False, False, "shuffled"), projection="kz0-areal")


# ---------------------------------------------------------------------------
# training a 2D factory model
# ---------------------------------------------------------------------------

#: The toy's fixed local curvature ``a`` (``f = a rho^2 / 2``), and the parameters that make its data.
CURVATURE = 1.0
TRUE_KAPPA = (0.3, 0.6)
TRUE_MOBILITY_RAW = (0.4, 0.3, -0.2)
#: The training cell: the 2D toy's grid, and the slab its z-invariant 3D twin is written in.
TRAIN_CELL = (8.3, 6.2, 2.7)
TRAIN_GRID = (8, 6)
#: Labels of the written archive: the x, y cube the grid holds, and three z planes.
TRAIN_LABELS = np.array([(nx, ny, nz) for nx in range(-3, 4) for ny in range(-2, 3)
                         for nz in (-1, 0, 1)], dtype=np.int64)
FRAME_DT, SUBSTEPS, N_FRAMES = 0.005, 5, 60


def _linear_system(raw_root, grid=TRAIN_GRID, **training):
    """The demo system with the linear 2D toy on ``grid``, trained on the drift term alone."""
    base = toy.demo_system(str(raw_root))
    defaults = {**base.defaults, "estimator": "weak_mid", "k_max": 2.0, "sigma": 0.5,
                "training": {**base.defaults["training"], "batch_size": 16, **training}}
    functional = toy.toy_functional(functools.partial(build_ndim_toy, linear_a=CURVATURE),
                                    grid=grid, kappa=0.5)
    return dataclasses.replace(base, functional=functional, variants={}, defaults=defaults)


def _truth(system):
    torch.manual_seed(0)
    model = build(system)
    with torch.no_grad():
        model.kernel.kappa_raw.copy_(torch.log(torch.expm1(torch.tensor(TRUE_KAPPA))))
        model.mobility_raw.copy_(torch.tensor(TRUE_MOBILITY_RAW))
    return model


def _write_z_invariant_archive(tree, model, *, n_runs=4):
    """2D trajectories of ``model`` (the explicit Heun integrator), written as the atom sums of the 3D
    field that is the same at every z, in :data:`TRAIN_CELL`: ``rho_k = V_ref c(n_x, n_y)`` on the
    ``n_z = 0`` plane, ``c`` the 2D field's Fourier coefficient, and noise on ``n_z = +-1`` that the
    projection must drop."""
    rng = np.random.default_rng(7)
    g = torch.Generator().manual_seed(11)
    B, (Gx, Gy) = n_runs, TRAIN_GRID
    rho = torch.stack([0.5 + 0.05 * torch.randn(B, Gx, Gy, generator=g),
                       0.4 + 0.05 * torch.randn(B, Gx, Gy, generator=g)], dim=1)
    h0 = model.ops.rfft(rho) / (Gx * Gy)
    boxes = torch.tensor(TRAIN_CELL[:2]).expand(B, 2)
    with torch.no_grad():
        traj = rollout_deterministic(model, h0, boxes, torch.ones(B), FRAME_DT / SUBSTEPS,
                                     SUBSTEPS * (N_FRAMES - 1), method="heun", state_proj="floor",
                                     floor=0.0, clamp_rho=None, save_every=SUBSTEPS)
    assert traj.shape[0] == N_FRAMES
    field = model.ops.irfft(traj * (Gx * Gy)).double()               # (frames, B, n, Gx, Gy)
    coeff = (torch.fft.fftn(field, dim=(-2, -1)) / (Gx * Gy)).numpy()
    V = math.prod(TRAIN_CELL)
    nx, ny, nz = TRAIN_LABELS.T
    plane = nz == 0
    for b in range(B):
        amps = np.zeros((N_FRAMES, len(TRAIN_LABELS), 2), np.complex128)
        amps[:, plane] = np.moveaxis(V * coeff[:, b][..., nx[plane] % Gx, ny[plane] % Gy], 1, 2)
        amps[:, ~plane] = 5.0 * (rng.standard_normal((N_FRAMES, (~plane).sum(), 2))
                                 + 1j * rng.standard_normal((N_FRAMES, (~plane).sum(), 2)))
        boxes3 = np.asarray(TRAIN_CELL) * (1.0 + 0.002 * rng.standard_normal((N_FRAMES, 3)))
        ModesRecord(amplitudes=amps.astype(np.complex64), labels=TRAIN_LABELS, boxes=boxes3,
                    T=1.0, dt_frame=FRAME_DT,
                    provenance={"composition": {"x_B": 0.5}, "cache_dir": str(tree)},
                    reference_box=np.asarray(TRAIN_CELL)).save(tree / f"run_{b}")


@pytest.fixture(scope="module")
def linear_data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("projection")
    system = _linear_system(tmp)
    tree = tmp / "fields" / "modes_demo"
    with _single_thread():          # set up before the function-scoped fixture that does this
        _write_z_invariant_archive(tree, _truth(system))
    sources = (SourceSpec("demo_src", str(tree), "run_*", TRAIN_GRID, 1.0, ()),)
    return tmp, system, sources


def _trained(run_dir, system):
    model = build(system)
    state = torch.load(run_dir / "final.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model


def _fit(tmp, system, sources, projection, run_name, steps, **kw):
    return fit(system, run_name=run_name, sources=sources, steps=steps, seed=0,
               resume_optimizer=False, root=tmp / "data", device="cpu", anchors=NO_ANCHORS,
               split_mode="none", projection=projection,
               config_overrides={"lr": 0.05, "warmup_epochs": 0, "anneal_epochs": 1}, **kw)


def test_the_true_model_leaves_no_drift_residual_on_its_own_projected_windows(linear_data):
    """The LitModule on a two-dimensional operator set: the model that made the data fits its
    windows to the estimator's error, a model with other parameters does not."""
    from aipf.train.config import TrainConfig
    from aipf.train.lit_module import LitModule

    _, system, sources = linear_data
    dm = ModeDataModule(
        sources=sources, keys=KEYS, window=_window(TRAIN_GRID, "weak_mid"),
        split=SplitSettings("none", (), 0.0, 0),
        weighting=RunWeighting("uniform", None, None, None, None),
        loader=LoaderSettings(32, 0, False, False, "shuffled"), projection="kz0-volumetric")
    dm.setup()
    dataset = dm.sources[0].train_dataset
    batch = torch.utils.data.default_collate([dataset[i] for i in range(0, len(dataset), 7)])
    cfg = TrainConfig(sigma=0.5, k_max=2.0, alpha_loss=0.0, h_inv_eps=1e-6)
    losses = []
    for model in (_truth(system), build(system)):
        lit = LitModule(model, model.ops, cfg)
        with torch.no_grad():
            losses.append(float(lit.compute_losses({"drift": batch})[0]))
    assert losses[0] < 1e-4 * losses[1], losses


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_a_2d_factory_model_trains_through_the_projection_and_records_it(linear_data, projection):
    tmp, system, sources = linear_data
    run = _fit(tmp, system, sources, projection, f"smoke_{projection}", 3)
    manifest = json.loads((run / "MANIFEST.json").read_text())
    assert manifest["projection"] == projection and manifest["global_step"] == 3
    assert manifest["terms_trained"] == ["L_dyn"]
    model = _trained(run, system)
    assert model.ops.ndim == 2 and model.ops.grid == TRAIN_GRID


def test_a_2d_factory_model_recovers_the_mobility_and_kappa_that_made_its_data(linear_data):
    """Areal densities, as a slab of a bulk run is trained; the drift of a linear model is the same in
    either measure, so the parameters are the ones that made the data."""
    tmp, system, sources = linear_data
    truth = _truth(system)
    model = _trained(_fit(tmp, system, sources, "kz0-areal", "recover", 400), system)
    with torch.no_grad():
        kappa = torch.nn.functional.softplus(model.kernel.kappa_raw)
        M, M_true = model.mobility_matrix(), truth.mobility_matrix()
    assert torch.allclose(kappa, torch.tensor(TRUE_KAPPA), rtol=0.02), kappa
    assert torch.allclose(M, M_true, atol=0.01 * float(M_true.abs().max())), (M, M_true)
    start = build(system)
    assert not torch.allclose(start.mobility_matrix(), M_true, atol=0.1), "it started elsewhere"


@pytest.mark.parametrize("projection, grid, anchors, training, match", [
    (None, TRAIN_GRID, NO_ANCHORS, {}, "trains on its k_z = 0 plane"),
    ("kz0-areal", (8, 6, 4), NO_ANCHORS, {}, "another number of axes"),
    ("kz0-areal", TRAIN_GRID, None, {"tables": {"anchors": "anchors.csv"}},
     "anchor tables are three-dimensional only"),
])
def test_fit_refuses_a_2d_model_on_an_inconsistent_run_before_writing(tmp_path, projection, grid,
                                                                      anchors, training, match):
    system = _linear_system(tmp_path, **training)
    sources = (SourceSpec("demo_src", str(tmp_path), "run_*", grid, 1.0, ()),)
    with pytest.raises(DimensionMismatch, match=match):
        fit(system, run_name="r", sources=sources, steps=1, seed=0, resume_optimizer=False,
            root=tmp_path / "data", device="cpu", anchors=anchors, projection=projection)
    assert not (tmp_path / "data").exists()


def test_fit_refuses_a_projection_for_a_3d_model_before_writing(tmp_path):
    system = dataclasses.replace(toy.demo_system(str(tmp_path)), variants={})
    sources = (SourceSpec("demo_src", str(tmp_path), "run_*", GRID2, 1.0, ()),)
    with pytest.raises(DimensionMismatch, match="three-axis grid"):
        fit(system, run_name="r", sources=sources, steps=1, seed=0, resume_optimizer=False,
            root=tmp_path / "data", device="cpu", anchors=NO_ANCHORS, projection="kz0-areal")
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("override, match", [
    ({"lambda_W": 1.0}, "kernel hinge"),
])
def test_a_2d_model_is_refused_a_three_dimensional_only_term(tmp_path, override, match):
    system = _linear_system(tmp_path)
    sources = (SourceSpec("demo_src", str(tmp_path), "run_*", TRAIN_GRID, 1.0, ()),)
    with pytest.raises(DimensionMismatch, match=match):
        check_dimensions(system, sources, projection="kz0-areal", anchors=NO_ANCHORS,
                         config_overrides=override)
    check_dimensions(system, sources, projection="kz0-areal", anchors=NO_ANCHORS)


# ---------------------------------------------------------------------------
# aipf train --projection
# ---------------------------------------------------------------------------

def _cli(monkeypatch, system, *extra):
    import aipf.system as system_mod
    from aipf.cli.main import main

    training = {**system.defaults["training"], "source_root": {"tier": "raw", "path": "fields"}}
    system = dataclasses.replace(system, defaults={**system.defaults, "training": training})
    monkeypatch.setattr(system_mod, "load", lambda name: system)
    return main(["train", "--system", "demo", "--run", "cli", "--seed", "0", "--steps", "1",
                 "--resume-optimizer", "no", "--device", "cpu", *extra])


def test_aipf_train_projection_trains_a_2d_model_and_spells_its_grid_gx_gy(tmp_path, monkeypatch,
                                                                         capsys):
    from aipf.cli import train_cmd
    from aipf.data import index

    assert train_cmd.PROJECTIONS == PROJECTIONS
    system = _linear_system(tmp_path)
    _write_z_invariant_archive(tmp_path / "fields" / "modes_demo", _truth(system), n_runs=2)
    monkeypatch.setenv("AIPF_DATA", str(tmp_path / "data"))
    assert _cli(monkeypatch, system, "--source", "demo_src=modes_demo:run_*:8,6",
                "--anchors", "none", "--projection", "kz0-areal") == 0, capsys.readouterr().err
    run = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert json.loads((run / "MANIFEST.json").read_text())["projection"] == "kz0-areal"
    assert run.parent == index.ckpt_dir(system)


@pytest.mark.parametrize("extra, match", [
    (("--source", "s=modes_demo:run_*:8,6", "--anchors", "none"), "pass --projection"),
    (("--source", "s=modes_demo:run_*:8,6,4", "--anchors", "none", "--projection", "kz0-areal"),
     "spells 3 grid lengths"),
    (("--source", "s=modes_demo:run_*:8,6", "--anchors", "declared", "--projection",
      "kz0-volumetric"), "--anchors none"),
])
def test_aipf_train_refuses_a_projection_its_flags_contradict(tmp_path, monkeypatch, capsys,
                                                              extra, match):
    import importlib

    fit_mod = importlib.import_module("aipf.train.fit")
    monkeypatch.setattr(fit_mod, "fit", lambda *a, **k: pytest.fail("refused before fit"))
    assert _cli(monkeypatch, _linear_system(tmp_path), *extra) == 2
    assert match in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


def test_aipf_train_without_a_projection_is_the_command_it_was(monkeypatch):
    """The batch job of a three-dimensional run spells no --projection; one that has it, does."""
    import argparse

    from aipf.cli.train_cmd import _job_command

    args = argparse.Namespace(system="demo", variant=None, run_name="r", seed=0, steps=1,
                              epochs=None, source=[("s", "m", "run_*", (4, 4, 4))],
                              init_from_published=False, resume_optimizer="no", anchors="none",
                              log_every_step=False, device="cpu", deterministic=False,
                              projection=None)
    assert "--projection" not in _job_command(args)
    args.projection, args.source = "kz0-areal", [("s", "m", "run_*", (8, 6))]
    command = _job_command(args)
    assert command[command.index("--projection") + 1] == "kz0-areal"
    assert "s=m:run_*:8,6" in command
