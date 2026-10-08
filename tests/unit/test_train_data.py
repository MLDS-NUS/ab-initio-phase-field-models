"""Tests for `aipf.train.dataset` and `aipf.train.datamodule`.

Three kinds of check live here and they are not interchangeable.

**Unit checks** build a synthetic archive in `tmp_path` and exercise every
declared choice at `n_species` 1 and 2. They are hermetic and run anywhere.

**Reproducibility checks** are about the batch SEQUENCE rather than a
batch's contents: the same explicit generator must give the same order, a
different one must not, the worker count must not move it, and the global
torch RNG must not reach it at all. That last one is the defect
`aipf.train.sampling` was written against, and this file measures it on a
real loader rather than describing it.

**Parity checks** compare this pipeline against the published one, on the
published archive, with `numpy.array_equal` / `torch.equal` and no
tolerance. They are skipped where that tree or that archive is not
reachable, so the suite stays green off the machine that holds them, and
they are the evidence that this port is a port.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from aipf.spectral import SpectralOps
from aipf.train.config import TrainConfig
from aipf.train.dataset import (
    ESTIMATORS,
    ArchiveKeys,
    ModeRun,
    ModeWindowDataset,
    WindowSettings,
    band_keep,
    read_mode_runs,
    savgol_taps,
    scatter_modes,
    triangular_weights,
    weak_target,
)
from aipf.train.datamodule import (
    ORDERS,
    RUN_WEIGHTINGS,
    SPLITS,
    IndexSteppingSampler,
    LoaderSettings,
    ModeDataModule,
    RunWeighting,
    SourceSpec,
    SplitSettings,
    run_weights,
    split_runs,
)
from aipf.train.lit_module import LitModule

# ---------------------------------------------------------------------------
# the synthetic archive
# ---------------------------------------------------------------------------

#: The key spelling of the archive these tests write. Nothing in the package
#: knows these names: they are this fixture's declaration, exactly as a real
#: system's are its own.
KEYS = ArchiveKeys(
    file_name="modes.npz",
    amplitudes="rho_k",
    amplitudes_channel_axis=2,
    labels="nvec",
    box="box",
    temperature="T_K",
    frame_interval="dt_frame_ps",
    composition=("x_left", "x_right"),
    composition_fallback="x_mean",
    quality_file="DATA_QUALITY_WARNING.json",
    quality_key="valid_frames",
)

#: Two grids, so a multi-source run is exercised the way the published one
#: runs: one loader per source, each at its own grid, in one step.
GRID_A = (8, 8, 8)
GRID_B = (6, 6, 4)

#: The label cube both grids can hold: `|n| <= 2` needs `Gx // 2 > 2` and
#: `Gz // 2 + 1 > 2`, which is true of both.
REACH = 2


def _labels() -> np.ndarray:
    axes = [np.arange(-REACH, REACH + 1)] * 3
    return np.stack(np.meshgrid(*axes, indexing="ij"),
                    -1).reshape(-1, 3).astype(np.int16)


def _write_run(directory: Path, *, n_frames: int, n_channels: int,
               temperature: float, frame_interval: float,
               composition=None, fallback=None, fixed_box=False,
               seed: int = 0, channel_axis: int = 2) -> np.ndarray:
    """One synthetic `modes.npz`. Returns the amplitudes as written."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = _labels()
    n_modes = labels.shape[0]
    amp = (rng.standard_normal((n_frames, n_modes, n_channels))
           + 1j * rng.standard_normal((n_frames, n_modes, n_channels))
           ).astype(np.complex64)
    stored = amp if channel_axis == 2 else np.moveaxis(amp, 1, 2)
    if fixed_box:
        box = np.array([9.0, 9.5, 10.0])
    else:
        box = 9.0 + 0.01 * rng.standard_normal((n_frames, 3))
    payload = dict(rho_k=stored, nvec=labels, box=box,
                   T_K=float(temperature), dt_frame_ps=float(frame_interval))
    if composition is not None:
        payload["x_left"], payload["x_right"] = (float(composition[0]),
                                                 float(composition[1]))
    if fallback is not None:
        payload["x_mean"] = float(fallback)
    np.savez(directory / "modes.npz", **payload)
    return amp


def _archive(root: Path, *, n_channels: int = 2, n_runs: int = 4,
             n_frames: int = 60, prefix: str = "run") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_runs):
        _write_run(root / f"{prefix}_{i:02d}", n_frames=n_frames,
                   n_channels=n_channels, temperature=1000.0 + 100 * i,
                   frame_interval=0.02, composition=(0.1 * i, 1.0 - 0.1 * i),
                   seed=i)
    return root


def _window(estimator: str = "weak", grid=GRID_A, **overrides) -> WindowSettings:
    kwargs = dict(estimator=estimator, half_width=5, n_states=3, stride=7,
                  grid=grid, savgol_window=None, savgol_poly=None)
    if estimator == "savgol":
        kwargs.update(savgol_window=11, savgol_poly=3)
    kwargs.update(overrides)
    return WindowSettings(**kwargs)


def _loader_settings(**overrides) -> LoaderSettings:
    kwargs = dict(batch_size=4, num_workers=0, pin_memory=False,
                  drop_last=False, order="shuffled")
    kwargs.update(overrides)
    return LoaderSettings(**kwargs)


def _split(**overrides) -> SplitSettings:
    kwargs = dict(mode="random", val_labels=(), val_fraction=0.25, seed=316)
    kwargs.update(overrides)
    return SplitSettings(**kwargs)


def _weighting(**overrides) -> RunWeighting:
    kwargs = dict(mode="uniform", sigma=None, k_max=None, eps=None,
                  probe_every=None)
    kwargs.update(overrides)
    return RunWeighting(**kwargs)


# ---------------------------------------------------------------------------
# ArchiveKeys: every spelling is the caller's, and a half-declared sidecar
# is refused rather than silently disabled
# ---------------------------------------------------------------------------

def test_archive_keys_has_no_default_for_any_field():
    with pytest.raises(TypeError):
        ArchiveKeys()


def test_archive_keys_refuses_a_sidecar_file_without_its_key():
    with pytest.raises(ValueError, match="quality_key"):
        ArchiveKeys(file_name="modes.npz", amplitudes="rho_k",
                    amplitudes_channel_axis=2, labels="nvec", box="box",
                    temperature="T_K", frame_interval="dt",
                    composition=(), composition_fallback=None,
                    quality_file="WARN.json", quality_key=None)


def test_archive_keys_refuses_a_sidecar_key_without_its_file():
    with pytest.raises(ValueError, match="quality_file"):
        ArchiveKeys(file_name="modes.npz", amplitudes="rho_k",
                    amplitudes_channel_axis=2, labels="nvec", box="box",
                    temperature="T_K", frame_interval="dt",
                    composition=(), composition_fallback=None,
                    quality_file=None, quality_key="valid_frames")


@pytest.mark.parametrize("axis", [0, 3, -1])
def test_archive_keys_refuses_an_impossible_channel_axis(axis):
    with pytest.raises(ValueError, match="amplitudes_channel_axis"):
        ArchiveKeys(file_name="modes.npz", amplitudes="rho_k",
                    amplitudes_channel_axis=axis, labels="nvec", box="box",
                    temperature="T_K", frame_interval="dt",
                    composition=(), composition_fallback=None,
                    quality_file=None, quality_key=None)


# ---------------------------------------------------------------------------
# read_mode_runs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_read_mode_runs_returns_channel_first_amplitudes(tmp_path, n_species):
    written = _write_run(tmp_path / "run_00", n_frames=12,
                         n_channels=n_species, temperature=1234.0,
                         frame_interval=0.02, composition=(0.2, 0.8))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    assert len(runs) == 1
    r = runs[0]
    assert r.n_species == n_species
    assert r.amplitudes.shape == (12, n_species, written.shape[1])
    assert np.array_equal(r.amplitudes, np.moveaxis(written, 1, 2))
    assert r.temperature == 1234.0 and r.frame_interval == 0.02
    assert r.composition == (0.2, 0.8)
    assert r.tag == "run_00"


def test_read_mode_runs_honours_a_channel_first_archive(tmp_path):
    written = _write_run(tmp_path / "run_00", n_frames=8, n_channels=2,
                         temperature=1.0, frame_interval=0.5,
                         composition=(0.0, 1.0), channel_axis=1)
    keys = KEYS.replace(amplitudes_channel_axis=1)
    runs = read_mode_runs(tmp_path, "run_*", keys=keys)
    assert np.array_equal(runs[0].amplitudes, np.moveaxis(written, 1, 2))


def test_read_mode_runs_is_sorted_by_path(tmp_path):
    for name in ("run_c", "run_a", "run_b"):
        _write_run(tmp_path / name, n_frames=8, n_channels=2,
                   temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    assert [r.tag for r in runs] == ["run_a", "run_b", "run_c"]


def test_read_mode_runs_falls_back_to_the_single_composition_key(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=8, n_channels=2,
               temperature=1.0, frame_interval=0.5, fallback=0.5)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    assert runs[0].composition == (0.5, 0.5)


def test_read_mode_runs_refuses_a_run_with_no_composition_at_all(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=8, n_channels=2,
               temperature=1.0, frame_interval=0.5)
    with pytest.raises(KeyError, match="x_left"):
        read_mode_runs(tmp_path, "run_*", keys=KEYS)


def test_read_mode_runs_accepts_no_composition_declaration(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=8, n_channels=2,
               temperature=1.0, frame_interval=0.5)
    keys = KEYS.replace(composition=(), composition_fallback=None)
    runs = read_mode_runs(tmp_path, "run_*", keys=keys)
    assert runs[0].composition == ()


def test_read_mode_runs_skips_a_directory_with_no_amplitude_file(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=8, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    (tmp_path / "run_99").mkdir()
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    assert [r.tag for r in runs] == ["run_00"]


def test_read_mode_runs_refuses_an_empty_match(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing_*"):
        read_mode_runs(tmp_path, "nothing_*", keys=KEYS)


def test_read_mode_runs_truncates_at_the_quality_sidecar(tmp_path, caplog):
    written = _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
                         temperature=1.0, frame_interval=0.5,
                         composition=(0.0, 1.0))
    (tmp_path / "run_00" / "DATA_QUALITY_WARNING.json").write_text(
        json.dumps({"valid_frames": "0..17 (timesteps 0..3400)"}))
    with caplog.at_level("WARNING"):
        runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    assert runs[0].n_frames == 18
    assert np.array_equal(runs[0].amplitudes,
                          np.moveaxis(written, 1, 2)[:18])
    assert runs[0].boxes.shape[0] == 18
    assert "run_00" in caplog.text and "18" in caplog.text


@pytest.mark.parametrize("spec", ["all of them", "3..10", None])
def test_read_mode_runs_refuses_a_malformed_quality_sidecar(tmp_path, spec):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    payload = {} if spec is None else {"valid_frames": spec}
    (tmp_path / "run_00" / "DATA_QUALITY_WARNING.json").write_text(
        json.dumps(payload))
    with pytest.raises(ValueError, match="valid_frames"):
        read_mode_runs(tmp_path, "run_*", keys=KEYS)


def test_read_mode_runs_ignores_a_sidecar_that_was_not_declared(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    (tmp_path / "run_00" / "DATA_QUALITY_WARNING.json").write_text(
        json.dumps({"valid_frames": "0..17"}))
    keys = KEYS.replace(quality_file=None, quality_key=None)
    runs = read_mode_runs(tmp_path, "run_*", keys=keys)
    assert runs[0].n_frames == 40


# ---------------------------------------------------------------------------
# the box a window is read in
# ---------------------------------------------------------------------------

def test_a_fixed_box_run_reports_that_box_exactly(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=20, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0),
               fixed_box=True)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    r = runs[0]
    assert r.boxes.shape == (3,)
    got = r.box_mean(3, 11)
    assert np.array_equal(got, np.array([9.0, 9.5, 10.0]))


def test_a_breathing_box_run_averages_the_window(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=20, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    r = read_mode_runs(tmp_path, "run_*", keys=KEYS)[0]
    assert np.array_equal(r.box_mean(3, 11), r.boxes[3:11].mean(axis=0))


def test_box_mean_refuses_an_empty_window(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=20, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    r = read_mode_runs(tmp_path, "run_*", keys=KEYS)[0]
    with pytest.raises(ValueError, match="empty"):
        r.box_mean(5, 5)


# ---------------------------------------------------------------------------
# scatter_modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_scatter_modes_places_every_non_negative_label(n_species):
    labels = _labels()
    amp = np.arange(n_species * labels.shape[0], dtype=np.float32).reshape(
        n_species, labels.shape[0]).astype(np.complex64)
    out = scatter_modes(amp, labels, 2.0, GRID_A)
    Gx, Gy, Gz = GRID_A
    assert out.shape == (n_species, Gx, Gy, Gz // 2 + 1)
    assert out.dtype == np.complex64
    for m, n in enumerate(labels):
        if n[2] < 0:
            continue
        got = out[:, n[0] % Gx, n[1] % Gy, n[2]]
        assert np.array_equal(got, (amp[:, m] / 2.0).astype(np.complex64))


def test_scatter_modes_leaves_unfilled_modes_at_zero():
    labels = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int16)
    amp = np.ones((2, 2), dtype=np.complex64)
    out = scatter_modes(amp, labels, 1.0, GRID_A)
    assert int(np.count_nonzero(out)) == 2 * 2


def test_scatter_modes_keeps_a_leading_state_axis():
    labels = _labels()
    amp = np.ones((3, 2, labels.shape[0]), dtype=np.complex64)
    out = scatter_modes(amp, labels, 1.0, GRID_A)
    assert out.shape == (3, 2, 8, 8, 5)


def test_scatter_modes_divides_by_the_volume_in_double_then_rounds_once():
    labels = np.array([[0, 0, 0]], dtype=np.int16)
    amp = np.array([[np.complex64(1.0 + 0j)]], dtype=np.complex64)
    volume = 3.0
    out = scatter_modes(amp, labels, volume, GRID_A)
    expected = (np.complex128(1.0 + 0j) / volume).astype(np.complex64)
    assert out[0, 0, 0, 0] == expected


@pytest.mark.parametrize("label", [[4, 0, 0], [0, 4, 0], [0, 0, 5]])
def test_scatter_modes_refuses_a_label_the_grid_cannot_hold(label):
    labels = np.array([label], dtype=np.int16)
    amp = np.ones((1, 1), dtype=np.complex64)
    with pytest.raises(ValueError, match="grid"):
        scatter_modes(amp, labels, 1.0, GRID_A)


# ---------------------------------------------------------------------------
# the estimators
# ---------------------------------------------------------------------------

def test_weak_target_is_the_difference_of_adjacent_window_means():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((30, 2, 5))
         + 1j * rng.standard_normal((30, 2, 5))).astype(np.complex64)
    got = weak_target(a, 12, 4, 0.25)
    expected = (a[13:17].mean(axis=0) - a[9:13].mean(axis=0)) / (4 * 0.25)
    assert np.array_equal(got, expected)


def test_weak_target_of_a_linear_ramp_is_its_slope():
    slope = 0.75
    a = (slope * np.arange(40, dtype=np.float64))[:, None, None] * np.ones(
        (1, 1, 3))
    got = weak_target(a.astype(np.complex128), 20, 6, 1.0)
    assert np.allclose(got, slope)


def test_triangular_weights_are_the_subsampled_triangle():
    off, lam = triangular_weights(25, 5)
    assert np.array_equal(off, np.array([-23, -11, 1, 13, 25]))
    assert np.allclose(lam * 53.0, np.array([1.0, 13.0, 25.0, 13.0, 1.0]))
    assert lam.sum() == pytest.approx(1.0)


def test_triangular_weights_saturate_at_the_full_support():
    off, lam = triangular_weights(4, 99)
    assert np.array_equal(off, np.arange(-2, 5))
    assert lam.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("half_width,n_states", [(2, 1), (3, 2), (5, 3),
                                                 (25, 5), (25, 49), (4, 99)])
def test_triangular_weights_are_distinct_and_inside_the_support(half_width,
                                                                n_states):
    off, lam = triangular_weights(half_width, n_states)
    assert list(off) == sorted(set(off.tolist()))
    assert off[0] >= 2 - half_width and off[-1] <= half_width
    assert len(off) == len(lam)
    assert lam.sum() == pytest.approx(1.0)
    assert np.all(lam > 0)


def test_savgol_taps_are_symmetric_and_antisymmetric():
    c0, c1 = savgol_taps(11, 3, 0.5)
    assert np.allclose(c0, c0[::-1])
    assert np.allclose(c1, -c1[::-1])


def test_savgol_taps_differentiate_a_cubic_exactly():
    dt = 0.25
    t = (np.arange(11) - 5) * dt
    series = 1.0 - 2.0 * t + 0.5 * t ** 2 + 0.3 * t ** 3
    c0, c1 = savgol_taps(11, 3, dt)
    assert c0 @ series == pytest.approx(1.0)
    assert c1 @ series == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# WindowSettings
# ---------------------------------------------------------------------------

def test_window_settings_refuses_an_unregistered_estimator():
    with pytest.raises(ValueError, match="estimator"):
        _window(estimator="finite_difference")


def test_estimator_registry_names_the_three_measured_forms():
    assert ESTIMATORS == ("weak", "weak_mid", "savgol")


def test_window_settings_needs_savgol_knobs_for_the_savgol_estimator():
    with pytest.raises(ValueError, match="savgol_window"):
        WindowSettings(estimator="savgol", half_width=5, n_states=1, stride=3,
                       grid=GRID_A, savgol_window=None, savgol_poly=3)


def test_window_settings_refuses_savgol_knobs_on_the_weak_estimator():
    with pytest.raises(ValueError, match="savgol_window"):
        _window(estimator="weak", savgol_window=11)


def test_window_settings_refuses_an_even_savgol_window():
    with pytest.raises(ValueError, match="odd"):
        _window(estimator="savgol", savgol_window=10)


def test_window_settings_refuses_a_polynomial_the_window_cannot_carry():
    with pytest.raises(ValueError, match="savgol_poly"):
        _window(estimator="savgol", savgol_window=5, savgol_poly=5)


@pytest.mark.parametrize("field,bad", [("half_width", 0), ("n_states", 0),
                                       ("stride", 0)])
def test_window_settings_refuses_a_non_positive_count(field, bad):
    with pytest.raises(ValueError, match=field):
        _window(**{field: bad})


def test_window_settings_refuses_a_grid_that_is_neither_three_axes_nor_two():
    for grid in ((8,), (8, 8, 8, 8), (8, 0, 8)):
        with pytest.raises(ValueError, match="grid"):
            _window(grid=grid)
    # two axes: the grid of runs projected to two dimensions (test_train_projection.py)
    assert _window(grid=(8, 8)).grid == (8, 8)


# ---------------------------------------------------------------------------
# ModeWindowDataset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_dataset_sample_is_exactly_the_drift_group_the_lit_module_takes(
        tmp_path, n_species):
    _archive(tmp_path, n_channels=n_species, n_runs=2, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    ds = ModeWindowDataset(runs, _window())
    s = ds[0]
    assert set(s) == {"rho_hat_states", "target_hat", "lam", "boxes", "T",
                      "sample_weight", "run_index"}
    Gzr = GRID_A[2] // 2 + 1
    assert s["target_hat"].shape == (n_species, 8, 8, Gzr)
    assert s["rho_hat_states"].shape == (3, n_species, 8, 8, Gzr)
    assert s["target_hat"].dtype == torch.complex64
    assert s["lam"].shape == (3,) and s["lam"].dtype == torch.float32
    assert s["boxes"].shape == (3,) and s["boxes"].dtype == torch.float32
    assert s["T"].shape == () and s["sample_weight"].shape == ()


def test_dataset_centre_range_for_the_weak_estimator(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    ds = ModeWindowDataset(runs, _window(half_width=5, stride=7))
    assert [c for _, c in ds.samples] == list(range(4, 35, 7))
    assert len(ds) == len(ds.samples)


def test_weak_mid_cuts_the_weak_windows_and_states_at_frame_midpoints(tmp_path):
    """Same centres, same target, same weights; the states are ``(a[c+m] + a[c+m-1]) / 2``."""
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.02, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    weak = ModeWindowDataset(runs, _window(half_width=5, n_states=3, stride=7))
    mid = ModeWindowDataset(runs, _window(estimator="weak_mid", half_width=5,
                                          n_states=3, stride=7))
    assert mid.samples == weak.samples
    run = runs[0]
    centre = mid.samples[2][1]
    offsets, lam = triangular_weights(5, 3)
    box = run.boxes[centre - 4:centre + 6].mean(axis=0)
    states = 0.5 * (run.amplitudes[centre + offsets]
                    + run.amplitudes[centre + offsets - 1])
    expected = scatter_modes(states, run.labels, float(box.prod()), GRID_A)
    assert np.array_equal(mid[2]["rho_hat_states"].numpy(), expected)
    assert np.array_equal(mid[2]["target_hat"].numpy(),
                          weak[2]["target_hat"].numpy())
    assert np.array_equal(mid[2]["lam"].numpy(), weak[2]["lam"].numpy())
    assert not np.array_equal(mid[2]["rho_hat_states"].numpy(),
                              weak[2]["rho_hat_states"].numpy())


def test_dataset_centre_range_for_the_savgol_estimator(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    ds = ModeWindowDataset(runs, _window(estimator="savgol",
                                         savgol_window=11, stride=7))
    assert [c for _, c in ds.samples] == list(range(5, 34, 7))


def test_dataset_target_matches_the_estimator_called_directly(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.02, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    ds = ModeWindowDataset(runs, _window(half_width=5, stride=7))
    run = runs[0]
    centre = ds.samples[3][1]
    box = run.boxes[centre - 4:centre + 6].mean(axis=0)
    expected = scatter_modes(
        weak_target(run.amplitudes, centre, 5, run.frame_interval),
        run.labels, float(box.prod()), GRID_A)
    assert np.array_equal(ds[3]["target_hat"].numpy(), expected)


def test_dataset_states_are_the_subsampled_offsets(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.02, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    ds = ModeWindowDataset(runs, _window(half_width=5, n_states=3, stride=7))
    run = runs[0]
    centre = ds.samples[2][1]
    offsets, lam = triangular_weights(5, 3)
    box = run.boxes[centre - 4:centre + 6].mean(axis=0)
    expected = scatter_modes(run.amplitudes[centre + offsets], run.labels,
                             float(box.prod()), GRID_A)
    assert np.array_equal(ds[2]["rho_hat_states"].numpy(), expected)
    assert np.allclose(ds[2]["lam"].numpy(), lam.astype(np.float32))


@pytest.mark.parametrize("n_species", [1, 2])
def test_the_savgol_sample_carries_exactly_one_state(tmp_path, n_species):
    """The savgol estimator has one state, and it still carries the state
    axis. Asserted on the SHAPE, because a missing leading axis broadcasts
    against a correctly shaped tensor rather than failing a comparison."""
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=n_species,
               temperature=1.0, frame_interval=0.02, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    ds = ModeWindowDataset(runs, _window(estimator="savgol",
                                         savgol_window=11, stride=7))
    sample = ds[0]
    Gzr = GRID_A[2] // 2 + 1
    assert sample["rho_hat_states"].shape == (1, n_species, 8, 8, Gzr)
    assert sample["target_hat"].shape == (n_species, 8, 8, Gzr)
    assert sample["lam"].shape == (1,)
    assert float(sample["lam"][0]) == 1.0
    batch = default_collate([ds[0], ds[1]])
    assert batch["rho_hat_states"].shape == (2, 1, n_species, 8, 8, Gzr)


def test_dataset_savgol_target_uses_each_runs_own_frame_interval(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.02, composition=(0.0, 1.0),
               seed=1)
    _write_run(tmp_path / "run_01", n_frames=40, n_channels=2,
               temperature=2.0, frame_interval=0.05, composition=(0.2, 0.8),
               seed=2)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    ds = ModeWindowDataset(runs, _window(estimator="savgol",
                                         savgol_window=11, stride=7))
    for index, (run_index, centre) in enumerate(ds.samples):
        run = runs[run_index]
        _c0, c1 = savgol_taps(11, 3, run.frame_interval)
        window = run.amplitudes[centre - 5:centre + 6]
        box = run.boxes[centre - 5:centre + 6].mean(axis=0)
        expected = scatter_modes(np.tensordot(c1, window, axes=(0, 0)),
                                 run.labels, float(box.prod()), GRID_A)
        assert np.array_equal(ds[index]["target_hat"].numpy(), expected)


def test_dataset_run_weights_reach_the_sample(tmp_path):
    _archive(tmp_path, n_runs=2, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    ds = ModeWindowDataset(runs, _window(),
                           run_weights=np.array([2.5, 0.5], np.float32))
    by_run = {int(ds[i]["run_index"]): float(ds[i]["sample_weight"])
              for i in range(len(ds))}
    assert by_run == {0: 2.5, 1: 0.5}


def test_dataset_refuses_run_weights_of_the_wrong_length(tmp_path):
    _archive(tmp_path, n_runs=2, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    with pytest.raises(ValueError, match="run_weights"):
        ModeWindowDataset(runs, _window(),
                          run_weights=np.array([1.0], np.float32))


def test_dataset_refuses_runs_of_different_channel_counts(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    _write_run(tmp_path / "run_01", n_frames=40, n_channels=1,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    with pytest.raises(ValueError, match="channel"):
        ModeWindowDataset(runs, _window())


def test_dataset_refuses_an_empty_run_list():
    """`match` is the specific message: with no runs the channel-count check
    below also raises a ValueError mentioning "runs", so a loose pattern
    passes whether or not this guard exists at all."""
    with pytest.raises(ValueError, match="runs is empty"):
        ModeWindowDataset([], _window())


def test_a_run_too_short_for_one_window_contributes_no_samples(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    _write_run(tmp_path / "run_01", n_frames=6, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.1, 0.9))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    ds = ModeWindowDataset(runs, _window(half_width=5, stride=7))
    assert {ri for ri, _ in ds.samples} == {0}


# ---------------------------------------------------------------------------
# split_runs
# ---------------------------------------------------------------------------

def test_split_registry_names_every_mode_it_serves():
    assert SPLITS == ("labels", "random", "none")


def test_label_split_holds_out_the_declared_labels(tmp_path):
    _archive(tmp_path, n_runs=4, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    settings = _split(mode="labels", val_labels=((0.1, 0.9),))
    train, val = split_runs(runs, settings)
    assert [r.tag for r in val] == ["run_01"]
    assert len(train) == 3


def test_label_split_can_hold_out_nothing(tmp_path):
    _archive(tmp_path, n_runs=4, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    train, val = split_runs(runs, _split(mode="labels",
                                         val_labels=((9.0, 9.0),)))
    assert len(train) == 4 and val == []


def test_random_split_is_a_function_of_its_seed_alone(tmp_path):
    _archive(tmp_path, n_runs=8, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    a = split_runs(runs, _split(seed=316, val_fraction=0.25))
    b = split_runs(list(reversed(runs)), _split(seed=316, val_fraction=0.25))
    assert [r.tag for r in a[1]] == [r.tag for r in b[1]]
    c = split_runs(runs, _split(seed=317, val_fraction=0.25))
    assert [r.tag for r in a[1]] != [r.tag for r in c[1]]


def test_random_split_keeps_at_least_one_validation_run(tmp_path):
    _archive(tmp_path, n_runs=4, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    _train, val = split_runs(runs, _split(val_fraction=0.0))
    assert len(val) == 1


def test_random_split_of_a_single_run_validates_nothing(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    train, val = split_runs(runs, _split())
    assert len(train) == 1 and val == []


def test_random_split_rounds_the_fraction_to_the_nearest_run(tmp_path):
    _archive(tmp_path, n_runs=10, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    _train, val = split_runs(runs, _split(val_fraction=0.26))
    assert len(val) == 3


def test_split_refuses_an_unregistered_mode(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        SplitSettings(mode="kfold", val_labels=(), val_fraction=0.1, seed=0)


def test_label_split_refuses_runs_with_no_composition(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.5)
    keys = KEYS.replace(composition=(), composition_fallback=None)
    runs = read_mode_runs(tmp_path, "run_*", keys=keys)
    with pytest.raises(ValueError, match="composition"):
        split_runs(runs, _split(mode="labels", val_labels=((0.0, 1.0),)))


# ---------------------------------------------------------------------------
# run_weights
# ---------------------------------------------------------------------------

def test_run_weighting_registry_names_both_measured_modes():
    assert RUN_WEIGHTINGS == ("uniform", "inverse_band_power")


def test_uniform_weighting_is_all_ones(tmp_path):
    _archive(tmp_path, n_runs=3, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    w = run_weights(runs, _weighting(), _window())
    assert np.array_equal(w, np.ones(3, np.float32))


def test_inverse_band_power_weights_average_to_one(tmp_path):
    _archive(tmp_path, n_runs=4, n_frames=60)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    w = run_weights(runs, _weighting(mode="inverse_band_power", sigma=1.0,
                                     k_max=2.0, eps=1e-6, probe_every=10),
                    _window())
    assert w.dtype == np.float32
    assert float(w.mean()) == pytest.approx(1.0, rel=1e-5)
    assert np.all(w > 0)


def test_inverse_band_power_gives_a_noisier_run_less_weight(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=60, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0),
               seed=3)
    _write_run(tmp_path / "run_01", n_frames=60, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.1, 0.9),
               seed=4)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    loud = ModeRun(tag=runs[1].tag, amplitudes=runs[1].amplitudes * 10.0,
                   labels=runs[1].labels, boxes=runs[1].boxes,
                   temperature=runs[1].temperature,
                   frame_interval=runs[1].frame_interval,
                   composition=runs[1].composition)
    w = run_weights([runs[0], loud],
                    _weighting(mode="inverse_band_power", sigma=1.0,
                               k_max=2.0, eps=1e-6, probe_every=10),
                    _window())
    assert w[0] > w[1]


def test_inverse_band_power_gives_a_run_with_no_window_zero_weight(
        tmp_path, caplog):
    _archive(tmp_path, n_runs=2, n_frames=60)
    _write_run(tmp_path / "run_09", n_frames=6, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.4, 0.6))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    with caplog.at_level("WARNING"):
        w = run_weights(runs, _weighting(mode="inverse_band_power", sigma=1.0,
                                         k_max=2.0, eps=1e-6, probe_every=10),
                        _window())
    assert w[-1] == 0.0 and np.all(w[:-1] > 0)
    assert "run_09" in caplog.text


def test_inverse_band_power_refuses_when_no_run_has_a_window(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=6, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    with pytest.raises(RuntimeError, match="no run"):
        run_weights(runs, _weighting(mode="inverse_band_power", sigma=1.0,
                                     k_max=2.0, eps=1e-6, probe_every=10),
                    _window())


@pytest.mark.parametrize("field", ["sigma", "k_max", "eps", "probe_every"])
def test_inverse_band_power_needs_every_one_of_its_knobs(field):
    kwargs = dict(mode="inverse_band_power", sigma=1.0, k_max=2.0, eps=1e-6,
                  probe_every=10)
    kwargs[field] = None
    with pytest.raises(ValueError, match=field):
        RunWeighting(**kwargs)


def test_uniform_weighting_refuses_knobs_it_does_not_use():
    with pytest.raises(ValueError, match="sigma"):
        _weighting(sigma=1.0)


def test_run_weighting_refuses_an_unregistered_mode():
    with pytest.raises(ValueError, match="mode"):
        _weighting(mode="by_temperature")


def test_band_multiplicity_matches_the_spectral_operators(tmp_path):
    """The run weighting counts each half-spectrum mode with the same
    multiplicity `SpectralOps.MULT` carries, so one statement of the rfft
    folding rule governs both the weighting and the loss."""
    from aipf.train.datamodule import band_multiplicity
    for grid in (GRID_A, GRID_B, (4, 4, 6), (4, 4, 7)):
        ops = SpectralOps(grid, 1, nyquist_mask=True)
        got = band_multiplicity(grid[2] // 2 + 1, grid[2])
        assert np.array_equal(got, ops.MULT.numpy().reshape(-1))


# ---------------------------------------------------------------------------
# ModeDataModule
# ---------------------------------------------------------------------------

def _sources(root: Path):
    return (SourceSpec(name="a", root=str(root / "a"), pattern="run_*",
                       grid=GRID_A, loss_weight=1.0, exclude_tags=()),
            SourceSpec(name="b", root=str(root / "b"), pattern="run_*",
                       grid=GRID_B, loss_weight=0.5, exclude_tags=()))


def _two_source_module(root: Path, n_channels: int = 2, **overrides):
    _archive(root / "a", n_channels=n_channels, n_runs=4, n_frames=60)
    _archive(root / "b", n_channels=n_channels, n_runs=4, n_frames=60)
    kwargs = dict(sources=_sources(root), keys=KEYS, window=_window(),
                  split=_split(), weighting=_weighting(),
                  loader=_loader_settings())
    kwargs.update(overrides)
    return ModeDataModule(**kwargs)


def test_datamodule_exposes_source_weights_before_setup(tmp_path):
    dm = _two_source_module(tmp_path)
    assert dm.source_loss_weights == {"a": 1.0, "b": 0.5}
    assert dm.source_names == ("a", "b")


def test_datamodule_builds_one_train_dataset_per_source(tmp_path):
    dm = _two_source_module(tmp_path)
    dm.setup()
    assert [s.name for s in dm.sources] == ["a", "b"]
    assert dm.sources[0].window.grid == GRID_A
    assert dm.sources[1].window.grid == GRID_B


def test_datamodule_gives_every_source_its_own_grid_in_one_step(tmp_path):
    dm = _two_source_module(tmp_path)
    dm.setup()
    g = torch.Generator().manual_seed(0)
    batch, _idx, _loader = next(iter(dm.train_dataloader(generator=g)))
    assert set(batch) == {"a", "b"}
    assert batch["a"]["target_hat"].shape[-3:] == (8, 8, 5)
    assert batch["b"]["target_hat"].shape[-3:] == (6, 6, 3)


def test_datamodule_excludes_a_named_tag(tmp_path):
    _archive(tmp_path / "a", n_runs=4, n_frames=60)
    src = SourceSpec(name="a", root=str(tmp_path / "a"), pattern="run_*",
                     grid=GRID_A, loss_weight=1.0,
                     exclude_tags=("run_00", "run_01"))
    dm = ModeDataModule(sources=(src,), keys=KEYS, window=_window(),
                        split=_split(mode="labels", val_labels=()),
                        weighting=_weighting(), loader=_loader_settings())
    dm.setup()
    assert {r.tag for r in dm.sources[0].train_runs} == {"run_02", "run_03"}


def test_an_excluded_run_is_never_opened(tmp_path):
    """Exclusion is a reason not to READ a run, not a filter after the fact:
    this module holds a whole timeline per run in memory, so a named run that
    is still read costs its own size for nothing. Measured by naming a run
    whose file cannot be read at all."""
    _archive(tmp_path / "a", n_runs=2, n_frames=60)
    broken = tmp_path / "a" / "run_99"
    broken.mkdir()
    (broken / "modes.npz").write_bytes(b"not an npz file")
    src = SourceSpec(name="a", root=str(tmp_path / "a"), pattern="run_*",
                     grid=GRID_A, loss_weight=1.0, exclude_tags=("run_99",))
    dm = ModeDataModule(sources=(src,), keys=KEYS, window=_window(),
                        split=_split(mode="labels", val_labels=()),
                        weighting=_weighting(), loader=_loader_settings())
    dm.setup()
    assert {r.tag for r in dm.sources[0].train_runs} == {"run_00", "run_01"}
    with pytest.raises(Exception):
        read_mode_runs(tmp_path / "a", "run_99", keys=KEYS)


def test_datamodule_refuses_a_source_with_an_empty_train_split(tmp_path):
    _archive(tmp_path / "a", n_runs=2, n_frames=60)
    src = SourceSpec(name="a", root=str(tmp_path / "a"), pattern="run_*",
                     grid=GRID_A, loss_weight=1.0, exclude_tags=())
    dm = ModeDataModule(sources=(src,), keys=KEYS, window=_window(),
                        split=_split(mode="labels",
                                     val_labels=((0.0, 1.0), (0.1, 0.9))),
                        weighting=_weighting(), loader=_loader_settings())
    with pytest.raises(RuntimeError, match="train split"):
        dm.setup()


def test_datamodule_refuses_duplicate_source_names(tmp_path):
    src = SourceSpec(name="a", root=str(tmp_path), pattern="run_*",
                     grid=GRID_A, loss_weight=1.0, exclude_tags=())
    with pytest.raises(ValueError, match="name"):
        ModeDataModule(sources=(src, src), keys=KEYS, window=_window(),
                       split=_split(), weighting=_weighting(),
                       loader=_loader_settings())


def test_datamodule_refuses_no_sources_at_all():
    with pytest.raises(ValueError, match="sources"):
        ModeDataModule(sources=(), keys=KEYS, window=_window(),
                       split=_split(), weighting=_weighting(),
                       loader=_loader_settings())


def test_datamodule_validation_loaders_are_one_per_source(tmp_path):
    dm = _two_source_module(tmp_path)
    dm.setup()
    g = torch.Generator().manual_seed(0)
    loaders = dm.val_dataloader(generator=g)
    assert len(loaders) == 2


def test_datamodule_validation_is_none_when_nothing_is_held_out(tmp_path):
    _archive(tmp_path / "a", n_runs=4, n_frames=60)
    src = SourceSpec(name="a", root=str(tmp_path / "a"), pattern="run_*",
                     grid=GRID_A, loss_weight=1.0, exclude_tags=())
    dm = ModeDataModule(sources=(src,), keys=KEYS, window=_window(),
                        split=_split(mode="labels", val_labels=()),
                        weighting=_weighting(), loader=_loader_settings())
    dm.setup()
    g = torch.Generator().manual_seed(0)
    assert dm.val_dataloader(generator=g) is None


def test_datamodule_dataloaders_need_setup_first(tmp_path):
    dm = _two_source_module(tmp_path)
    g = torch.Generator().manual_seed(0)
    with pytest.raises(RuntimeError, match="setup"):
        dm.train_dataloader(generator=g)


def test_validation_datasets_carry_no_run_weighting(tmp_path):
    dm = _two_source_module(
        tmp_path, weighting=_weighting(mode="inverse_band_power", sigma=1.0,
                                       k_max=2.0, eps=1e-6, probe_every=10))
    dm.setup()
    val = dm.sources[0].val_dataset
    assert val is not None
    assert np.array_equal(val.run_weights,
                          np.ones(len(val.runs), np.float32))


# ---------------------------------------------------------------------------
# reproducibility: the batch sequence is declared, not inherited
# ---------------------------------------------------------------------------

def test_the_same_generator_seed_gives_the_same_batch_sequence(tmp_path):
    dm = _two_source_module(tmp_path)
    dm.setup()
    a = [b["a"]["run_index"].tolist() for b, _i, _l
         in dm.train_dataloader(generator=torch.Generator().manual_seed(7))]
    b = [b["a"]["run_index"].tolist() for b, _i, _l
         in dm.train_dataloader(generator=torch.Generator().manual_seed(7))]
    assert a == b


def test_a_different_generator_seed_gives_a_different_batch_sequence(tmp_path):
    dm = _two_source_module(tmp_path)
    dm.setup()
    a = [b["a"]["run_index"].tolist() for b, _i, _l
         in dm.train_dataloader(generator=torch.Generator().manual_seed(7))]
    b = [b["a"]["run_index"].tolist() for b, _i, _l
         in dm.train_dataloader(generator=torch.Generator().manual_seed(8))]
    assert a != b


def test_the_global_torch_rng_does_not_reach_the_batch_sequence(tmp_path):
    """The defect `aipf.train.sampling` names, measured on a real loader:
    with `shuffle=True` and no generator a DataLoader seeds its sampler from
    the GLOBAL stream, so the order moves when anything else draws. With this
    module's required generator it does not."""
    dm = _two_source_module(tmp_path)
    dm.setup()
    torch.manual_seed(0)
    a = [b["a"]["run_index"].tolist() for b, _i, _l
         in dm.train_dataloader(generator=torch.Generator().manual_seed(7))]
    torch.manual_seed(1234)
    torch.rand(17)
    b = [b["a"]["run_index"].tolist() for b, _i, _l
         in dm.train_dataloader(generator=torch.Generator().manual_seed(7))]
    assert a == b


def test_the_worker_count_does_not_move_the_batch_sequence(tmp_path):
    dm0 = _two_source_module(tmp_path, loader=_loader_settings(num_workers=0))
    dm0.setup()
    a = [b["a"]["run_index"].tolist() for b, _i, _l
         in dm0.train_dataloader(generator=torch.Generator().manual_seed(7))]
    dm2 = _two_source_module(tmp_path, loader=_loader_settings(num_workers=2))
    dm2.setup()
    b = [b["a"]["run_index"].tolist() for b, _i, _l
         in dm2.train_dataloader(generator=torch.Generator().manual_seed(7))]
    assert a == b


def test_the_worker_count_does_not_move_the_batch_contents(tmp_path):
    dm0 = _two_source_module(tmp_path, loader=_loader_settings(num_workers=0))
    dm0.setup()
    first0 = next(iter(dm0.train_dataloader(
        generator=torch.Generator().manual_seed(7))))[0]
    dm2 = _two_source_module(tmp_path, loader=_loader_settings(num_workers=2))
    dm2.setup()
    first2 = next(iter(dm2.train_dataloader(
        generator=torch.Generator().manual_seed(7))))[0]
    for name in ("a", "b"):
        for key in ("target_hat", "rho_hat_states", "boxes", "T",
                    "sample_weight", "lam"):
            assert torch.equal(first0[name][key], first2[name][key]), (
                f"{name}/{key}")


def test_train_dataloader_requires_an_explicit_generator(tmp_path):
    dm = _two_source_module(tmp_path)
    dm.setup()
    with pytest.raises(TypeError):
        dm.train_dataloader()


def test_val_dataloader_requires_an_explicit_generator(tmp_path):
    dm = _two_source_module(tmp_path)
    dm.setup()
    with pytest.raises(TypeError):
        dm.val_dataloader()


def test_setup_is_a_pure_function_of_its_declarations(tmp_path):
    a = _two_source_module(tmp_path)
    a.setup()
    b = _two_source_module(tmp_path)
    b.setup()
    for sa, sb in zip(a.sources, b.sources):
        assert [r.tag for r in sa.train_runs] == [r.tag for r in sb.train_runs]
        assert [r.tag for r in sa.val_runs] == [r.tag for r in sb.val_runs]
        assert np.array_equal(sa.train_dataset.run_weights,
                              sb.train_dataset.run_weights)
        assert sa.train_dataset.samples == sb.train_dataset.samples


# ---------------------------------------------------------------------------
# the two declared knobs a recorded stream is matched with: a split of
# "none" and the "index_stepping" order
# ---------------------------------------------------------------------------

def _index_stepping_module(root: Path, **overrides):
    """The file's own two-source toy, declared the way a comparison run is.

    No split -- every run trains, nothing validates -- and the training
    loader walking the record driver's own cut rather than a permutation.
    """
    kwargs = dict(split=_split(mode="none"),
                  loader=_loader_settings(order="index_stepping"))
    kwargs.update(overrides)
    return _two_source_module(root, **kwargs)


def test_order_registry_names_both_orders():
    assert ORDERS == ("shuffled", "index_stepping")


def test_split_mode_none_keeps_every_run_for_training(tmp_path):
    _archive(tmp_path, n_runs=4, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    train, val = split_runs(runs, _split(mode="none"))
    assert [r.tag for r in train] == [r.tag for r in runs]
    assert val == []


def test_split_mode_none_holds_nothing_back_even_at_a_fraction(tmp_path):
    """The reason `"none"` is its own mode and not `val_fraction=0.0`: the
    random split deliberately rounds a zero fraction UP to one run, so there
    is no fraction at which it validates nothing."""
    _archive(tmp_path, n_runs=4, n_frames=40)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    assert split_runs(runs, _split(val_fraction=0.0))[1] != []
    assert split_runs(runs, _split(mode="none", val_fraction=0.5))[1] == []


def test_a_source_with_no_split_has_no_validation_dataset(tmp_path):
    dm = _index_stepping_module(tmp_path)
    dm.setup()
    assert all(s.val_dataset is None for s in dm.sources)
    assert dm.val_dataloader(generator=torch.Generator().manual_seed(0)) is None


def test_loader_order_is_declared_not_defaulted():
    with pytest.raises(TypeError):
        LoaderSettings(batch_size=4, num_workers=0, pin_memory=False,
                       drop_last=False)
    with pytest.raises(ValueError, match="shuffled.*index_stepping"):
        LoaderSettings(batch_size=4, num_workers=0, pin_memory=False,
                       drop_last=False, order="sorted")


def test_the_index_stepping_sampler_is_the_drivers_cut(tmp_path):
    """The arithmetic on its own, against the step-by-step record driver's line:
    `stride = max(1, len // batch)`, indices `(i * stride + n) % len`."""
    for n_samples, batch in ((292, 16), (10, 4), (3, 4), (7, 1)):
        sampler = IndexSteppingSampler(n_samples, batch)
        stride = max(1, n_samples // batch)
        got = list(sampler)
        assert len(got) == n_samples == len(sampler)
        for n, indices in enumerate(got):
            assert indices == [(i * stride + n) % n_samples
                               for i in range(batch)]


def test_index_stepping_yields_the_drivers_windows(tmp_path):
    # batch n holds windows (i * stride + n) % len for i in range(batch_size),
    # stride = max(1, len // batch_size) -- the record driver's cut, verbatim
    dm = _index_stepping_module(tmp_path)
    dm.setup()
    loader = dm.train_dataloader(generator=torch.Generator().manual_seed(0))
    dataset = dm.sources[0].train_dataset
    stride = max(1, len(dataset) // dm.loader.batch_size)
    seen = 0
    for n, (batch, _idx, _loader) in zip(range(3), loader):
        expected = default_collate(
            [dataset[(i * stride + n) % len(dataset)]
             for i in range(dm.loader.batch_size)])
        for key in ("target_hat", "rho_hat_states", "run_index"):
            torch.testing.assert_close(batch["a"][key], expected[key])
        seen += 1
    assert seen == 3


def test_index_stepping_ignores_the_generator(tmp_path):
    dm = _index_stepping_module(tmp_path)
    dm.setup()
    a = next(iter(dm.train_dataloader(
        generator=torch.Generator().manual_seed(1))))[0]
    b = next(iter(dm.train_dataloader(
        generator=torch.Generator().manual_seed(2))))[0]
    for name in ("a", "b"):
        torch.testing.assert_close(a[name]["target_hat"],
                                   b[name]["target_hat"])
        assert a[name]["run_index"].tolist() == b[name]["run_index"].tolist()


def test_index_stepping_does_not_move_with_the_worker_count(tmp_path):
    a = _index_stepping_module(
        tmp_path, loader=_loader_settings(order="index_stepping",
                                          num_workers=0))
    a.setup()
    b = _index_stepping_module(
        tmp_path, loader=_loader_settings(order="index_stepping",
                                          num_workers=2))
    b.setup()
    first_a = next(iter(a.train_dataloader(
        generator=torch.Generator().manual_seed(7))))[0]
    first_b = next(iter(b.train_dataloader(
        generator=torch.Generator().manual_seed(7))))[0]
    torch.testing.assert_close(first_a["a"]["target_hat"],
                               first_b["a"]["target_hat"])


def test_index_stepping_never_reaches_a_validation_loader(tmp_path):
    """Declared on the module, the order governs the TRAINING stream only:
    a validation number is a mean over a whole split, which the walk would
    reorder and nothing else."""
    dm = _two_source_module(
        tmp_path, loader=_loader_settings(order="index_stepping"))
    dm.setup()
    loaders = dm.val_dataloader(generator=torch.Generator().manual_seed(0))
    assert loaders and all(loader.batch_size == dm.loader.batch_size
                           for loader in loaders)
    assert all(not isinstance(loader.batch_sampler, IndexSteppingSampler)
               for loader in loaders)


# ---------------------------------------------------------------------------
# one real training step, end to end
# ---------------------------------------------------------------------------

def _rung(grid, n_species: int, seed: int = 0):
    from aipf.functional.nonlocal_kernel import NonlocalKernel
    torch.manual_seed(seed)
    return NonlocalKernel(
        grid, n_species, kB=1.0, rho_ref=[0.5] * n_species, kBT_ref=1.0,
        h_u=4, h_g=4, R_cut=3.0, kernel_n_quad=17, kernel_n_k_table=17,
        kernel_k_table_max=8.0, kernel_hidden=4,
        mobility_prefactor="mole_fraction", mobility_shape="mlp_rho",
        mobility_t_form="none", mobility_hidden=4, ideal_form="gas",
        nyquist_mask=True)


@pytest.mark.parametrize("n_species", [1, 2])
def test_one_training_step_runs_end_to_end(tmp_path, n_species):
    """A batch off this pipeline is the `drift` group `LitModule` takes, with
    no adapter: dataset -> loader -> `compute_losses` -> backward -> step."""
    from aipf.spectral import OpsCache
    dm = _two_source_module(tmp_path, n_channels=n_species)
    dm.setup()
    model = _rung(GRID_A, n_species)
    model._cache.ops_for_grid(GRID_B)
    lit = LitModule(model, model._cache,
                    TrainConfig(sigma=1.0, k_max=2.0, lambda_dyn=1.0))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    batch, _i, _l = next(iter(dm.train_dataloader(
        generator=torch.Generator().manual_seed(3))))
    total = torch.zeros(())
    for name, source_batch in batch.items():
        loss, parts = lit.compute_losses({"drift": source_batch})
        assert set(parts) == {"L_dyn"}
        assert torch.isfinite(loss) and loss.item() >= 0.0
        total = total + dm.source_loss_weights[name] * loss
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()
    assert any(not torch.equal(a, b)
               for a, b in zip(before, list(model.parameters())))
    del OpsCache


def test_two_identically_seeded_runs_give_the_same_first_step_gradient(
        tmp_path):
    dm = _two_source_module(tmp_path)
    dm.setup()

    def first_gradient():
        model = _rung(GRID_A, 2, seed=11)
        model._cache.ops_for_grid(GRID_B)
        lit = LitModule(model, model._cache,
                        TrainConfig(sigma=1.0, k_max=2.0))
        batch, _i, _l = next(iter(dm.train_dataloader(
            generator=torch.Generator().manual_seed(5))))
        total = torch.zeros(())
        for name, sb in batch.items():
            loss, _ = lit.compute_losses({"drift": sb})
            total = total + dm.source_loss_weights[name] * loss
        total.backward()
        return [p.grad.detach().clone() for p in model.parameters()
                if p.grad is not None]

    a, b = first_gradient(), first_gradient()
    assert len(a) == len(b) and a
    assert all(torch.equal(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# pinned values: the split and the weighting, replayed
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the declarations refuse what they cannot mean
# ---------------------------------------------------------------------------

def test_read_mode_runs_refuses_amplitudes_that_are_not_a_timeline(tmp_path):
    (tmp_path / "run_00").mkdir(parents=True)
    np.savez(tmp_path / "run_00" / "modes.npz",
             rho_k=np.zeros((4, 3), np.complex64),
             nvec=np.zeros((3, 3), np.int16), box=np.ones((4, 3)),
             T_K=1.0, dt_frame_ps=0.5, x_left=0.0, x_right=1.0)
    with pytest.raises(ValueError, match="rho_k"):
        read_mode_runs(tmp_path, "run_*", keys=KEYS)


def test_read_mode_runs_falls_back_when_the_label_is_only_half_stored(
        tmp_path):
    """A run carrying one of two declared keys is not half-labelled: the
    fallback stands in for the WHOLE label, because a label read from two
    different conventions at once is not a label."""
    _write_run(tmp_path / "run_00", n_frames=8, n_channels=2,
               temperature=1.0, frame_interval=0.5, fallback=0.5)
    payload = dict(np.load(tmp_path / "run_00" / "modes.npz"))
    payload["x_left"] = np.float64(0.25)
    np.savez(tmp_path / "run_00" / "modes.npz", **payload)
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    assert runs[0].composition == (0.5, 0.5)


def test_source_spec_refuses_an_empty_name(tmp_path):
    with pytest.raises(ValueError, match="name"):
        SourceSpec(name="", root=str(tmp_path), pattern="run_*", grid=GRID_A,
                   loss_weight=1.0, exclude_tags=())


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_split_settings_refuses_a_fraction_outside_the_unit_interval(
        fraction):
    with pytest.raises(ValueError, match="val_fraction"):
        _split(val_fraction=fraction)


def test_run_weighting_refuses_a_non_positive_probe_spacing():
    with pytest.raises(ValueError, match="probe_every"):
        RunWeighting(mode="inverse_band_power", sigma=1.0, k_max=2.0,
                     eps=1e-6, probe_every=0)


def test_loader_settings_refuses_a_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        _loader_settings(batch_size=0)


def test_loader_settings_refuses_a_negative_worker_count():
    with pytest.raises(ValueError, match="num_workers"):
        _loader_settings(num_workers=-1)


# ---------------------------------------------------------------------------
# the window bounds, at the stride that can see them
# ---------------------------------------------------------------------------

def test_dataset_centre_range_at_unit_stride(tmp_path):
    """Stride one, so the last centre is the estimator's real reach rather
    than wherever the stride happened to stop."""
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=1.0, frame_interval=0.5, composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    weak = ModeWindowDataset(runs, _window(half_width=5, stride=1))
    assert [c for _, c in weak.samples] == list(range(4, 35))
    # the last window must still fit: it reads frames [30, 40)
    assert weak.settings.window_bounds(34) == (30, 40)
    savgol = ModeWindowDataset(runs, _window(estimator="savgol",
                                             savgol_window=11, stride=1))
    assert [c for _, c in savgol.samples] == list(range(5, 34))
    assert savgol.settings.window_bounds(33) == (28, 39)


def test_dataset_sample_carries_the_state_point_and_a_unit_weight(tmp_path):
    _write_run(tmp_path / "run_00", n_frames=40, n_channels=2,
               temperature=4321.0, frame_interval=0.5,
               composition=(0.0, 1.0))
    runs = read_mode_runs(tmp_path, "run_*", keys=KEYS)
    sample = ModeWindowDataset(runs, _window())[0]
    assert float(sample["T"]) == 4321.0
    assert float(sample["sample_weight"]) == 1.0
    assert int(sample["run_index"]) == 0


# ---------------------------------------------------------------------------
# the band statistic, against an independent statement of the same band
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the loaders' shape
# ---------------------------------------------------------------------------

def _uneven_module(root: Path, **overrides):
    _archive(root / "a", n_runs=4, n_frames=60)
    _archive(root / "b", n_runs=2, n_frames=30)
    kwargs = dict(sources=_sources(root), keys=KEYS, window=_window(),
                  split=_split(mode="labels", val_labels=()),
                  weighting=_weighting(), loader=_loader_settings())
    kwargs.update(overrides)
    return ModeDataModule(**kwargs)


def test_the_train_loader_runs_to_the_longest_source(tmp_path):
    """Every step carries every source: the short ones are recycled until
    the longest is exhausted, so no source silently ends an epoch early."""
    dm = _uneven_module(tmp_path)
    dm.setup()
    lengths = {s.name: len(s.train_dataset) for s in dm.sources}
    assert lengths["a"] > lengths["b"]
    steps = 0
    for batch, _i, _l in dm.train_dataloader(
            generator=torch.Generator().manual_seed(1)):
        assert set(batch) == {"a", "b"}
        steps += 1
    expected = -(-lengths["a"] // dm.loader.batch_size)
    assert steps == expected


def test_the_validation_loader_walks_its_dataset_in_order(tmp_path):
    """Compared on the window box, not on `T`: a validation split is often a
    single run, every sample of which carries the SAME temperature, so an
    order check on `T` passes whether the loader shuffles or not. Each
    window has its own box mean, so the box sequence is the order."""
    dm = _two_source_module(tmp_path)
    dm.setup()
    loader = dm.val_dataloader(generator=torch.Generator().manual_seed(4))[0]
    dataset = dm.sources[0].val_dataset
    expected = torch.stack([dataset[i]["boxes"] for i in range(len(dataset))])
    assert len(torch.unique(expected, dim=0)) == len(dataset), (
        "this fixture needs a distinct box per window")
    seen = torch.cat([batch["boxes"] for batch in loader])
    assert torch.equal(seen, expected)


def test_drop_last_shortens_the_epoch(tmp_path):
    keep = _two_source_module(tmp_path,
                              loader=_loader_settings(batch_size=5,
                                                      drop_last=False))
    keep.setup()
    drop = _two_source_module(tmp_path,
                              loader=_loader_settings(batch_size=5,
                                                      drop_last=True))
    drop.setup()
    n = len(keep.sources[0].train_dataset)
    assert n % 5 != 0, "this fixture must not divide evenly"
    kept = sum(1 for _ in keep.train_dataloader(
        generator=torch.Generator().manual_seed(1)))
    dropped = sum(1 for _ in drop.train_dataloader(
        generator=torch.Generator().manual_seed(1)))
    assert dropped == kept - 1


def test_the_drift_band_excludes_the_zero_mode():
    """Why a residue on the zero mode cannot reach a training step: the drift
    loss masks to `0 < |k| <= k_max`, so the zero mode is outside the band the
    residual is ever summed over."""
    ops = SpectralOps(GRID_A, 2, nyquist_mask=True)
    boxes = torch.tensor([[9.0, 9.5, 10.0]])
    band = ops.band_mask(boxes, 2.0)
    assert not bool(band[0, 0, 0, 0, 0])
    assert bool(band.any())


# ---------------------------------------------------------------------------
# the scattered mode band
# ---------------------------------------------------------------------------

def _cube(reach: int) -> np.ndarray:
    axes = [np.arange(-reach, reach + 1)] * 3
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def test_band_keep_keeps_the_lattice_points_inside_the_radius():
    """On a fixed cubic box of edge 8 the band 2 pi sqrt(2) / 8 holds exactly the 19 labels with
    n^2 <= 2 (the origin, 6 faces, 12 edges)."""
    labels = _cube(3)
    radius = 2 * np.pi * np.sqrt(2.0) / 8.0 * (1 + 1e-9)
    keep = band_keep(labels, np.array([8.0, 8.0, 8.0]), radius)
    assert keep.shape == (labels.shape[0],) and keep.dtype == bool
    assert int(keep.sum()) == 19
    assert set(map(tuple, labels[keep])) == {
        tuple(n) for n in labels if int((n ** 2).sum()) <= 2}
    assert band_keep(labels, np.array([8.0, 8.0, 8.0]), None) is None


def test_band_keep_measures_each_mode_in_the_longest_box_the_run_reaches():
    """A mode inside the band at any frame is kept: x reaches 8.5, y never leaves 8."""
    boxes = np.array([[8.0, 8.0, 8.0], [8.5, 8.0, 8.0]])
    radius = 2 * np.pi * 3 / 8.25
    keep = band_keep(np.array([[3, 0, 0], [0, 3, 0]]), boxes, radius)
    assert keep.tolist() == [True, False]


def test_a_stored_ball_past_the_grid_is_scattered_inside_its_band(tmp_path):
    """|n| = 2 overflows a 4-cubed grid; the band |k| <= 1.3 on the ~9 A boxes keeps the 27 labels
    with every |n_i| <= 1, which fit."""
    (run,) = read_mode_runs(_archive(tmp_path / "a", n_runs=1), "run_*", keys=KEYS)
    with pytest.raises(ValueError, match="beyond the grid"):
        ModeWindowDataset([run], _window(grid=(4, 4, 4)))[0]
    dataset = ModeWindowDataset([run], _window(grid=(4, 4, 4), band_k_max=1.3))
    (keep,) = dataset.keep
    assert int(keep.sum()) == 27
    assert np.abs(run.labels[keep]).max() == 1
    item = dataset[0]
    assert item["target_hat"].shape[-3:] == (4, 4, 3)
    assert item["rho_hat_states"].shape[-3:] == (4, 4, 3)
    # without a band nothing is dropped and the grid that holds the ball is unchanged
    assert ModeWindowDataset([run], _window()).keep == [None]


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_window_settings_refuses_a_band_that_is_not_a_positive_radius(radius):
    with pytest.raises(ValueError, match="band_k_max"):
        _window(band_k_max=radius)
