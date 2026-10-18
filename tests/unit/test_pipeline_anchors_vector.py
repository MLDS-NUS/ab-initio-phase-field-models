"""``aipf.pipeline.anchors`` beyond two channels: the concentration block, its gate, and absent channels.

The vector forms judge the per-frame ``m x m`` concentration matrix through its largest eigenvalue
relative to the ideal mixture, ``lambda = lambda_max(S_id^-1 B)``. At a binary row (two species
present) that is the scalar statistic divided by ``x (1 - x)``, so the binary-row checks below hold
the vector path to the scalar one; the scalar path itself is ``test_pipeline_anchors.py``'s.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aipf.pipeline.anchors import (
    AnchorTable,
    anchor_table,
    concentration_level,
    concentration_matrix_series,
    concentration_series,
    concentration_weights,
    condensing,
    ideal_concentration_matrix,
    kernel_shape,
    quarter_growth,
    single_phase,
    time_fifths,
)

_GATE = dict(growth_tol=0.2, condensing_ratio=2.5,
             condensing_ideal_factor=3.0, level_factor=10.0)


# ---------------------------------------------------------------------------
# the weights and the series
# ---------------------------------------------------------------------------
def test_the_weights_are_e_i_minus_f_i_times_the_present_channels():
    w = concentration_weights((0.2, 0.3, 0.5))
    assert np.allclose(w, [[0.8, -0.2, -0.2], [-0.3, 0.7, -0.3]], rtol=0, atol=1e-15)
    w = concentration_weights((0.4, 0.0, 0.6))          # channel 1 absent: no weight on it
    assert np.allclose(w, [[0.6, 0.0, -0.4]], rtol=0, atol=1e-15)
    w = concentration_weights((0.0, 0.7, 0.3, 0.0))
    assert np.allclose(w, [[0.0, 0.3, -0.7, 0.0]], rtol=0, atol=1e-15)


def test_each_mode_vanishes_on_a_uniform_mixture():
    f = np.array([0.2, 0.3, 0.5])
    assert np.allclose(concentration_weights(f) @ f, 0.0, atol=1e-15)


def test_a_pure_composition_has_no_mode():
    assert concentration_weights((0.0, 0.0, 1.0)).shape == (0, 3)


def test_two_channels_keep_the_crossed_fractions():
    assert np.array_equal(concentration_weights((0.3, 0.7)), np.array([0.7, -0.3]))


def test_a_matrix_of_weights_gives_one_series_per_row_bit_for_bit():
    rng = np.random.default_rng(1)
    rho = rng.standard_normal((6, 3, 5)) + 1j * rng.standard_normal((6, 3, 5))
    w = concentration_weights((0.2, 0.3, 0.5))
    out = concentration_series(rho, weights=w)
    assert out.shape == (6, 2, 5)
    for i in range(2):
        assert np.array_equal(out[:, i], concentration_series(rho, weights=w[i]))
    assert concentration_series(rho, weights=np.zeros((0, 3))).shape == (6, 0, 5)


def _ideal_amplitudes(counts, n_frames, n_modes, seed):
    """Raw ``sum_j exp(-i phi_j)`` per channel with uncorrelated uniform phases."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_frames, len(counts), n_modes), np.complex128)
    for a, n in enumerate(counts):
        if n:
            out[:, a] = np.exp(-2j * np.pi * rng.random((n_frames, n, n_modes))).sum(1)
    return out


def test_an_ideal_mixture_screens_at_the_ideal_block():
    counts = (40, 60, 100)
    N = sum(counts)
    x = np.array(counts) / N
    rho = _ideal_amplitudes(counts, 400, 8, seed=2)
    k = np.linspace(0.1, 1.2, 8)                        # two modes beyond k_qc
    S = concentration_matrix_series(rho, k, composition=x, n_atoms=N, k_qc=0.9)
    assert S.shape == (400, 2, 2)
    assert np.allclose(S.mean(0), ideal_concentration_matrix(x), atol=0.02)
    assert concentration_level(S.mean(0), x) == pytest.approx(1.0, abs=0.1)


def test_the_screen_is_the_written_out_sum_over_the_kept_modes():
    rng = np.random.default_rng(3)
    rho = rng.standard_normal((4, 3, 6)) + 1j * rng.standard_normal((4, 3, 6))
    k = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
    x = np.array([0.25, 0.25, 0.5])
    c = np.einsum("in,fnm->fim", concentration_weights(x), rho)[:, :, 1:4]
    want = np.einsum("fim,fjm->fij", c, c.conj()).real / (12.0 * 3)
    got = concentration_matrix_series(rho, k, composition=x, n_atoms=12, k_qc=0.9)
    assert np.allclose(got, want, rtol=1e-12)


def test_the_ideal_block_is_diag_x_minus_x_x_transpose():
    x = np.array([0.2, 0.3, 0.5])
    assert np.allclose(ideal_concentration_matrix(x),
                       np.diag([0.2, 0.3]) - np.outer([0.2, 0.3], [0.2, 0.3]))
    Z = concentration_weights(x)
    assert np.allclose(ideal_concentration_matrix(x), Z @ np.diag(x) @ Z.T)
    assert np.allclose(ideal_concentration_matrix((0.4, 0.0, 0.6)), [[0.24]])


def test_the_level_is_the_largest_eigenvalue_against_the_ideal_block():
    x = np.array([0.2, 0.3, 0.5])
    s_id = ideal_concentration_matrix(x)
    assert concentration_level(s_id, x) == pytest.approx(1.0)
    assert concentration_level(4.0 * s_id, x) == pytest.approx(4.0)
    want = np.max(np.linalg.eigvals(np.linalg.solve(s_id, np.diag([0.5, 0.1]))).real)
    assert concentration_level(np.diag([0.5, 0.1]), x) == pytest.approx(want)
    # a binary row: the scalar statistic over x (1 - x)
    assert concentration_level([[0.48]], (0.4, 0.0, 0.6)) == pytest.approx(0.48 / 0.24)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
_BINARY_ROW = (0.4, 0.0, 0.6)


def _as_matrix(values):
    return np.asarray(values, dtype=np.float64)[:, None, None]


@pytest.mark.parametrize("fifths", [
    [1e-6, 1e-5, 1e-5, 1e-5, 1e-5], [100.0] * 5, [1.0, 2.0, 3.0, 4.0, 100.0],
    [0.24, 0.3, 0.5, 0.7, 0.9], [0.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.2, 2.0, 2.2, 2.3],
    [2.0, 2.2, 2.0, 2.2, 2.5],  # level 2.4 is the tie, equal in exact arithmetic only
])
def test_a_binary_row_is_judged_as_the_scalar_gate_judges_it(fifths):
    x = _BINARY_ROW[0]
    assert condensing(_as_matrix(fifths), _BINARY_ROW, ratio=2.5, ideal_factor=3.0) \
        == condensing(np.asarray(fifths), x, ratio=2.5, ideal_factor=3.0)
    for growth in (0.06, 0.9):
        for T, t_min in ((1000.0, None), (9000.0, 8000.0), (7000.0, 8000.0)):
            assert single_phase(growth, _as_matrix(fifths), _BINARY_ROW, T,
                                t_min=t_min, **_GATE) \
                == single_phase(growth, np.asarray(fifths), x, T, t_min=t_min, **_GATE)


def test_one_condensing_direction_is_not_diluted_by_the_other():
    x = np.array([0.2, 0.3, 0.5])
    s_id = ideal_concentration_matrix(x)
    vals, vecs = np.linalg.eigh(s_id)
    root = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    def block(lam):                                     # eigenvalues (lam, 1) against S_id
        return root @ np.diag([lam, 1.0]) @ root
    fifths = np.stack([block(1.0), block(2.0), block(4.0), block(8.0), block(12.0)])
    assert concentration_level(fifths[-1], x) == pytest.approx(12.0)
    assert condensing(fifths, x, ratio=2.5, ideal_factor=3.0)
    # level 12 > 10: not single-phase although the trace average (12 + 1) / 2 is below it
    assert not single_phase(0.06, fifths, x, 1000.0, t_min=None, **_GATE)
    assert single_phase(0.06, np.stack([block(1.5)] * 5), x, 1000.0, t_min=None, **_GATE)


def test_the_matrix_fifths_and_quarters_average_the_frames_alone():
    rng = np.random.default_rng(4)
    series = rng.random((20, 2, 2))
    fifths = time_fifths(series, axis=0)
    assert fifths.shape == (5, 2, 2)
    assert np.allclose(fifths[1], series[4:8].mean(0))
    x = np.array([0.2, 0.3, 0.5])
    want = (concentration_level(series[-5:].mean(0), x)
            / concentration_level(series[:5].mean(0), x) - 1.0)
    assert quarter_growth(series, x) == pytest.approx(want)
    s_id = ideal_concentration_matrix(x)
    doubling = np.concatenate([np.tile(s_id, (10, 1, 1)), np.tile(2 * s_id, (10, 1, 1))])
    assert quarter_growth(doubling, x) == pytest.approx(1.0)


def test_a_binary_rows_growth_is_the_scalar_growth():
    s = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0])
    assert quarter_growth(_as_matrix(s), _BINARY_ROW) == pytest.approx(quarter_growth(s))


@pytest.mark.parametrize("call", [
    lambda c: condensing(np.ones((5, 1, 1)), c, ratio=2.5, ideal_factor=3.0),
    lambda c: single_phase(0.06, np.ones((5, 1, 1)), c, 1000.0, t_min=None, **_GATE),
    lambda c: quarter_growth(np.ones((8, 1, 1)), c),
])
def test_a_pure_composition_is_refused_by_name(call):
    with pytest.raises(ValueError, match="pure composition"):
        call((0.0, 1.0, 0.0))


def test_the_vector_gate_wants_five_matrices():
    with pytest.raises(ValueError, match="not five"):
        condensing(np.ones(5), (0.2, 0.3, 0.5), ratio=2.5, ideal_factor=3.0)
    with pytest.raises(ValueError, match="two\\s+channels pass the scalar"):
        condensing(np.ones((5, 1, 1)), (0.4, 0.6), ratio=2.5, ideal_factor=3.0)


# ---------------------------------------------------------------------------
# absent channels in the table
# ---------------------------------------------------------------------------
def _stub(temperature, icpt, k=(1.0, 2.0, 3.0)):
    n = icpt.shape[1]
    return type("Stub", (), dict(
        shell_wavenumbers=np.asarray(k, float), shell_intercepts=icpt,
        temperature=float(temperature), M_window=np.eye(n), M_zero_wavenumber=np.eye(n)))()


def _icpt(values, absent=()):
    out = np.zeros((3, 3, 3))
    out[:] = np.asarray(values, float)[:, None, None]
    for a in absent:
        out[:, a, :] = out[:, :, a] = 0.0                # an absent channel's amplitudes are zero
    return out


def test_a_pair_a_run_does_not_carry_is_nan_and_averaged_over_the_runs_that_do():
    full = _stub(2000.0, _icpt([4.0, 3.0, 1.0]))
    other = _stub(2000.0, _icpt([4.0, 2.0, 1.0]))
    binary = _stub(2000.0, _icpt([4.0, 2.0, 1.0], absent=(1,)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alone = kernel_shape([binary], t_min=0.0, shape_set="above_temperature")
        mixed = kernel_shape([full, binary], t_min=0.0, shape_set="above_temperature")
        both = kernel_shape([full, other], t_min=0.0, shape_set="above_temperature")
    assert np.isnan(alone.kernel[:, 1, :]).all() and np.isnan(alone.kernel[:, :, 1]).all()
    assert np.isfinite(alone.kernel[:, 0, 2]).all()
    # pair (0, 1): only the full run carries it; pair (0, 2): both runs, as an ordinary mean
    full_only = kernel_shape([full], t_min=0.0, shape_set="above_temperature")
    assert np.array_equal(mixed.kernel[:, 0, 1], full_only.kernel[:, 0, 1])
    assert np.array_equal(mixed.spread[:, 0, 1], np.zeros(3))
    assert np.array_equal(mixed.kernel[:, 0, 2], both.kernel[:, 0, 2])
    assert mixed.n_runs == 2


def test_an_absent_channels_density_scaled_entries_are_nan_without_a_warning():
    runs = [_stub(1000.0, _icpt([4.0, 3.0, 1.0])) for _ in range(2)]
    shape = kernel_shape(runs, t_min=0.0, shape_set="above_temperature")
    table = anchor_table(runs, pressure=0.0,
                         composition=np.array([[0.5, 0.0, 0.5], [0.2, 0.3, 0.5]]),
                         densities=np.array([[0.04, 0.0, 0.04], [0.02, 0.03, 0.05]]),
                         growth=np.zeros(2), single_phase=np.ones(2, bool), shape=shape,
                         shape_k_max=9.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        diagonal, cross = table.density_scaled("window")
    assert np.isnan(diagonal[0, 1]) and np.isnan(cross[0, [0, 2]]).all()
    assert diagonal[0, 0] == pytest.approx(1.0 / 0.04) and cross[0, 1] == 0.0
    assert np.isfinite(diagonal[1]).all() and np.isfinite(cross[1]).all()


def test_a_table_takes_one_composition_vector_per_run():
    runs = [_stub(1000.0, _icpt([4.0, 3.0, 1.0])) for _ in range(2)]
    shape = kernel_shape(runs, t_min=0.0, shape_set="above_temperature")
    kwargs = dict(pressure=0.0, densities=np.ones((2, 3)), growth=np.zeros(2),
                  single_phase=np.ones(2, bool), shape=shape, shape_k_max=9.0)
    table = anchor_table(runs, composition=np.full((2, 3), 1.0 / 3.0), **kwargs)
    assert isinstance(table, AnchorTable) and table.composition.shape == (2, 3)
    with pytest.raises(ValueError, match="composition has shape"):
        anchor_table(runs, composition=np.full((2, 2), 0.5), **kwargs)


@pytest.mark.parametrize("bad", [(0.2, 0.3, 0.6), (0.6, -0.1, 0.5), (0.5, float("nan"), 0.5)])
def test_a_composition_off_the_simplex_is_refused(bad):
    with pytest.raises(ValueError, match="not a point of the simplex"):
        concentration_weights(bad)
    with pytest.raises(ValueError, match="not a point of the simplex"):
        ideal_concentration_matrix(bad)


def test_a_nan_shape_on_an_active_pair_is_reported(caplog):
    binary = _stub(1000.0, _icpt([4.0, 3.0, 1.0], absent=(1,)))
    shape = kernel_shape([binary], t_min=0.0, shape_set="above_temperature")
    full = _stub(1000.0, _icpt([4.0, 3.0, 1.0]))
    with caplog.at_level("WARNING", logger="aipf.pipeline.anchors"):
        anchor_table([binary, full], pressure=0.0,
                     composition=np.array([[0.5, 0.0, 0.5], [0.2, 0.3, 0.5]]),
                     densities=np.array([[0.04, 0.0, 0.04], [0.02, 0.03, 0.05]]),
                     growth=np.zeros(2), single_phase=np.ones(2, bool), shape=shape,
                     shape_k_max=9.0)
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1 and "row 1" in messages[0], messages
