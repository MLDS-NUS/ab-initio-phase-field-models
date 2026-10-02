"""The measured anchor tables, and the archived ones they reproduce.

Two kinds of test. Most build small arrays whose answer is known by hand and
pin one property each. A handful reach the archived campaigns and rebuild
whole tables, because the claim this module exists to support is that an
archived table is reproducible bit for bit and no synthetic case can carry
that claim.

The archive tests read only. They are marked ``env`` and skip, naming the raw
root, when the campaign is not on this machine.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

import declared_roots
from aipf.pipeline.anchors import (
    SCALINGS,
    SHAPE_SETS,
    AnchorTable,
    KernelShape,
    anchor_rows,
    anchor_table,
    band_average,
    concentration_series,
    concentration_weights,
    condensing,
    extrapolate_to_zero,
    kernel_shape,
    lag_intercept,
    mobility_spectrum,
    quarter_growth,
    run_mobility,
    shape_amplitude,
    shell_index,
    shell_intercepts,
    single_phase,
    time_fifths,
)
from aipf.system import AnchorRules, System
from aipf.paths import Paths

# --------------------------------------------------------------------------
# The archived campaign this module is measured against. Read only, and named
# here once so that a test never spells a location of its own.
# --------------------------------------------------------------------------

#: The first campaign's field tree. Its four pressures are the named targets.
_FIELDS = str(declared_roots.raw("hhe", "fields"))

#: Pressures the campaign covers, in GPa.
_PRESSURES = (200, 400, 600, 800)

#: The Boltzmann constant in the unit the
#: campaign's tables are written in. A system constant, so it lives in a test
#: rather than in the package.
_KB = 8.617333262e-5

#: The measurement settings, each from the line that settles it.
_LAGS = tuple(range(1, 11))      # the measurement's own default lags
_BAND = (0.0, 1.0)               # K_FIT_LO, K_FIT_HI
_MIN_LAG = 0.06                  # FIT_MIN_LAG_PS
_EXTRAP_K_MAX = 1.5              # K_PLOT_HI
_SHAPE_K_MAX = 1.6               # K_M0_FIT_HI

#: The shape set's temperature floor per pressure, which no artefact records
#: (an inline comment on the published pipeline's command line).
_SHAPE_T_MIN = {200: 7500.0, 400: 9500.0, 600: 9500.0, 800: 10000.0}

#: experiments/hhe/system.py, anchor_rules.T_min_by_pressure. The critical
#: temperature per pressure, which waives the gate's level test.
_T_C = {200: 6100.0, 400: 7800.0, 600: 8800.0, 800: 8973.0}

#: analysis/phase_gate.py GROWTH_TOL and the two evidence factors, plus the
#: level factor from the commit that wrote these tables (55ab66a).
_GATE = dict(growth_tol=0.2, condensing_ratio=2.5,
             condensing_ideal_factor=3.0, level_factor=10.0)

#: The nine rows whose single-phase verdict was overridden by hand after the
#: tables were aggregated, recorded as (pressure, composition, temperature),
#: replayed in place.
_HAND_OVERRIDE = (
    (200, 0.05, 4000.0), (200, 0.10, 5000.0), (200, 0.80, 2000.0),
    (200, 0.90, 2000.0), (400, 0.80, 7000.0), (600, 0.20, 8000.0),
    (600, 0.80, 8000.0), (600, 0.90, 3000.0), (800, 0.20, 8000.0),
)

_have_archive = os.path.isdir(_FIELDS)


def archive(test):
    """An ``env`` test that skips, naming the raw root, when the archived campaign is absent."""
    return pytest.mark.env(pytest.mark.skipif(
        not _have_archive,
        reason=f"the archived campaign is not on this machine: {_FIELDS}"
               " (the system's raw root: AIPF_RAW_<SYSTEM>, AIPF_RAW or [paths.raw] in aipf.toml)")(test))


# --------------------------------------------------------------------------
# Small helpers for the hand-built cases
# --------------------------------------------------------------------------

def _straight(lags, intercept, slope, n_channels=2):
    """Matrices exactly linear in the lag, so the intercept is known."""
    out = np.zeros((len(lags), n_channels, n_channels))
    for a in range(n_channels):
        for b in range(n_channels):
            out[:, a, b] = intercept[a, b] + slope[a, b] * np.asarray(lags)
    return out


def _system(t_min_by_pressure, name="demo"):
    return System(name=name, n_species=2, species=("A", "Z"),
                  masses={"A": 1.0, "Z": 2.0}, atom_types={"A": 1, "Z": 2},
                  table_keys={}, paths=Paths(system=name),
                  anchor_rules=AnchorRules(
                      T_min_by_pressure=dict(t_min_by_pressure)),
                  constants={}, defaults={})


def _table(pressure, temperature, single_phase_flags):
    n = len(temperature)
    shape = KernelShape(wavenumbers=np.array([1.0]),
                        kernel=np.ones((1, 2, 2)), spread=np.zeros((1, 2, 2)),
                        n_runs=1, t_min=0.0)
    return AnchorTable(pressure=pressure, composition=np.full(n, 0.5),
                       temperature=np.asarray(temperature, float),
                       densities=np.ones((n, 2)),
                       M_window=np.ones((n, 2, 2)),
                       M_zero_wavenumber=np.full((n, 2, 2), 2.0),
                       M_shape=np.full((n, 2, 2), 3.0),
                       shape_residual=np.zeros((n, 2, 2)),
                       growth=np.zeros(n),
                       single_phase=np.asarray(single_phase_flags, bool),
                       shape=shape)


# ==========================================================================
# shell_index
# ==========================================================================

def test_the_fundamental_falls_in_the_first_shell():
    which, centres = shell_index(np.array([1.0, 1.0, 2.0, 3.0]))
    assert which[0] == 0 and which[1] == 0
    assert centres[0] == pytest.approx(1.0)


def test_a_shell_is_one_fundamental_wide():
    """Bins run from half a fundamental, so shell s holds |k| near (s+1) k_min."""
    which, _ = shell_index(np.array([1.0, 1.4, 1.6, 2.4, 2.6]))
    assert list(which) == [0, 0, 1, 1, 2]


def test_the_shell_wavenumber_is_its_members_mean_not_the_bin_centre():
    _, centres = shell_index(np.array([1.0, 1.6, 2.0]))
    assert centres[1] == pytest.approx(1.8)
    assert centres[1] != pytest.approx(2.0)


def test_an_empty_shell_keeps_its_index_and_is_not_a_number():
    _, centres = shell_index(np.array([1.0, 3.0]))
    assert centres.shape == (3,)
    assert np.isnan(centres[1])
    assert np.isfinite(centres[0]) and np.isfinite(centres[2])


def test_a_wavenumber_list_is_one_dimensional():
    with pytest.raises(ValueError, match="one wavenumber per mode"):
        shell_index(np.ones((3, 2)))


def test_an_empty_wavenumber_list_is_refused():
    with pytest.raises(ValueError, match="no fundamental"):
        shell_index(np.array([]))


def test_the_uniform_mode_has_to_be_removed_before_binning():
    with pytest.raises(ValueError, match="non-positive"):
        shell_index(np.array([0.0, 1.0]))


# ==========================================================================
# mobility_spectrum
# ==========================================================================

def _two_frame_case(n_channels=2):
    rng = np.random.default_rng(11)
    rho = (rng.standard_normal((4, n_channels, 3))
           + 1j * rng.standard_normal((4, n_channels, 3)))
    k = np.array([1.0, 1.0, 2.0])
    which, centres = shell_index(k)
    return rho, k, which, centres


def test_the_spectrum_is_the_defining_fluctuation_dissipation_ratio():
    rho, k, which, centres = _two_frame_case()
    out = mobility_spectrum(rho, k, which, len(centres), thermal_energy=0.5,
                            volume=7.0, lags=[1], frame_interval=0.25)
    d = np.moveaxis(rho, 1, 2)[1:] - np.moveaxis(rho, 1, 2)[:-1]
    cov = np.einsum("fma,fmb->mab", d, d.conj()).real / (4 - 1)
    per_mode = cov / (2 * 0.5 * k[:, None, None] ** 2 * 7.0 * 1 * 0.25)
    assert np.array_equal(out[0, 0], per_mode[:2].mean(0))
    assert np.array_equal(out[1, 0], per_mode[2:].mean(0))


def test_the_spectrum_is_generic_over_the_channel_count():
    for n_channels in (1, 2, 3):
        rho, k, which, centres = _two_frame_case(n_channels)
        out = mobility_spectrum(rho, k, which, len(centres),
                                thermal_energy=1.0, volume=1.0, lags=[1, 2],
                                frame_interval=1.0)
        assert out.shape == (2, 2, n_channels, n_channels)


def test_the_measured_matrix_is_symmetric():
    rho, k, which, centres = _two_frame_case()
    out = mobility_spectrum(rho, k, which, len(centres), thermal_energy=1.0,
                            volume=1.0, lags=[1], frame_interval=1.0)
    assert np.allclose(out[:, 0], np.swapaxes(out[:, 0], -1, -2))


def test_an_empty_shell_is_not_a_number_in_the_spectrum():
    rho = np.ones((4, 1, 2), complex)
    out = mobility_spectrum(rho, np.array([1.0, 3.0]), np.array([0, 2]), 3,
                            thermal_energy=1.0, volume=1.0, lags=[1],
                            frame_interval=1.0)
    assert np.isnan(out[1]).all()


def test_every_start_frame_contributes_an_increment():
    """Divided by the number of increments, which is F - lag, not F."""
    rho = np.zeros((10, 1, 1), complex)
    rho[:, 0, 0] = np.arange(10)
    out = mobility_spectrum(rho, np.array([1.0]), np.array([0]), 1,
                            thermal_energy=0.5, volume=1.0, lags=[3],
                            frame_interval=1.0)
    # every increment at lag 3 is exactly 3, and there are 10 - 3 of them
    expected = (7 * 9.0 / 7) / (2 * 0.5 * 1.0 * 1.0 * 3 * 1.0)
    assert out[0, 0, 0, 0] == pytest.approx(expected)


def test_the_spectrum_scales_with_the_thermal_energy_the_volume_and_the_lag():
    rho, k, which, centres = _two_frame_case()
    base = mobility_spectrum(rho, k, which, len(centres), thermal_energy=1.0,
                             volume=1.0, lags=[2], frame_interval=1.0)
    hotter = mobility_spectrum(rho, k, which, len(centres),
                               thermal_energy=2.0, volume=1.0, lags=[2],
                               frame_interval=1.0)
    bigger = mobility_spectrum(rho, k, which, len(centres),
                               thermal_energy=1.0, volume=4.0, lags=[2],
                               frame_interval=1.0)
    slower = mobility_spectrum(rho, k, which, len(centres),
                               thermal_energy=1.0, volume=1.0, lags=[2],
                               frame_interval=3.0)
    assert np.allclose(hotter, base / 2)
    assert np.allclose(bigger, base / 4)
    assert np.allclose(slower, base / 3)


def test_the_spectrum_does_not_depend_on_how_the_caller_laid_its_array_out():
    """A transposed view holds the same numbers and contracts differently."""
    rho, k, which, centres = _two_frame_case()
    view = np.moveaxis(np.ascontiguousarray(np.moveaxis(rho, 1, 2)), 2, 1)
    assert not view.flags["C_CONTIGUOUS"]
    a = mobility_spectrum(rho, k, which, len(centres), thermal_energy=1.0,
                          volume=1.0, lags=[1], frame_interval=1.0)
    b = mobility_spectrum(view, k, which, len(centres), thermal_energy=1.0,
                          volume=1.0, lags=[1], frame_interval=1.0)
    assert np.array_equal(a, b)


def test_amplitudes_are_a_timeline_of_channels_and_modes():
    with pytest.raises(ValueError, match="n_frames, n_channels, n_modes"):
        mobility_spectrum(np.ones((4, 2)), np.array([1.0]), np.array([0]), 1,
                          thermal_energy=1.0, volume=1.0, lags=[1],
                          frame_interval=1.0)


def test_one_wavenumber_per_stored_mode():
    with pytest.raises(ValueError, match="against 3 modes"):
        mobility_spectrum(np.ones((4, 1, 3), complex), np.array([1.0]),
                          np.array([0]), 1, thermal_energy=1.0, volume=1.0,
                          lags=[1], frame_interval=1.0)


def test_the_binning_covers_exactly_the_modes_given():
    with pytest.raises(ValueError, match="binning has to cover"):
        mobility_spectrum(np.ones((4, 1, 2), complex), np.array([1.0, 2.0]),
                          np.array([0]), 1, thermal_energy=1.0, volume=1.0,
                          lags=[1], frame_interval=1.0)


def test_an_empty_lag_list_is_refused():
    with pytest.raises(ValueError, match="lags is empty"):
        mobility_spectrum(np.ones((4, 1, 1), complex), np.array([1.0]),
                          np.array([0]), 1, thermal_energy=1.0, volume=1.0,
                          lags=[], frame_interval=1.0)


def test_a_lag_is_at_least_one_frame():
    with pytest.raises(ValueError, match="offset of at least one"):
        mobility_spectrum(np.ones((4, 1, 1), complex), np.array([1.0]),
                          np.array([0]), 1, thermal_energy=1.0, volume=1.0,
                          lags=[0], frame_interval=1.0)


def test_a_lag_longer_than_the_run_is_refused():
    with pytest.raises(ValueError, match="more than the 4 frames"):
        mobility_spectrum(np.ones((4, 1, 1), complex), np.array([1.0]),
                          np.array([0]), 1, thermal_energy=1.0, volume=1.0,
                          lags=[4], frame_interval=1.0)


# ==========================================================================
# band_average
# ==========================================================================

def test_the_band_average_takes_the_shells_inside_the_band():
    spectrum = np.arange(4 * 1 * 1 * 1, dtype=float).reshape(4, 1, 1, 1)
    centres = np.array([1.0, 2.0, 3.0, 4.0])
    out = band_average(spectrum, centres, k_min=2.0, k_max=3.0)
    assert out[0, 0, 0] == pytest.approx(1.5)


def test_the_band_is_inclusive_at_both_ends():
    spectrum = np.arange(3, dtype=float).reshape(3, 1, 1, 1)
    centres = np.array([1.0, 2.0, 3.0])
    assert band_average(spectrum, centres, k_min=1.0,
                        k_max=3.0)[0, 0, 0] == pytest.approx(1.0)
    assert band_average(spectrum, centres, k_min=1.5,
                        k_max=2.5)[0, 0, 0] == pytest.approx(1.0)


def test_an_empty_shell_does_not_poison_the_band_average():
    spectrum = np.array([1.0, np.nan, 3.0]).reshape(3, 1, 1, 1)
    centres = np.array([1.0, np.nan, 3.0])
    assert band_average(spectrum, centres, k_min=0.0,
                        k_max=9.0)[0, 0, 0] == pytest.approx(2.0)


def test_a_band_with_no_shell_in_it_is_refused():
    spectrum = np.ones((2, 1, 1, 1))
    with pytest.raises(ValueError, match="no shell lies in"):
        band_average(spectrum, np.array([1.0, 2.0]), k_min=5.0, k_max=6.0)


def test_the_band_average_checks_its_spectrum_shape():
    with pytest.raises(ValueError, match="n_shells, n_lags"):
        band_average(np.ones((2, 2)), np.array([1.0, 2.0]), k_min=0.0,
                     k_max=9.0)


def test_the_band_average_checks_it_has_one_centre_per_shell():
    with pytest.raises(ValueError, match="shells against"):
        band_average(np.ones((3, 1, 1, 1)), np.array([1.0, 2.0]), k_min=0.0,
                     k_max=9.0)


# ==========================================================================
# lag_intercept
# ==========================================================================

def test_a_straight_lag_curve_gives_its_own_intercept():
    lags = np.array([0.1, 0.2, 0.3, 0.4])
    icpt = np.array([[5.0, 1.0], [1.0, 3.0]])
    slope = np.array([[-2.0, -0.5], [-0.5, -1.0]])
    out, start = lag_intercept(lags, _straight(lags, icpt, slope),
                               min_lag=0.0)
    assert np.allclose(out, icpt)
    assert start == 0


def test_the_short_lags_are_dropped_before_the_branch_is_found():
    lags = np.array([0.01, 0.1, 0.2, 0.3])
    matrices = _straight(lags, np.eye(2) * 4.0, -np.eye(2))
    matrices[0] = 99.0                      # the jitter spike
    out, start = lag_intercept(lags, matrices, min_lag=0.05)
    assert start == 1
    assert np.allclose(out, np.eye(2) * 4.0)


def test_the_returned_start_indexes_the_original_lags():
    lags = np.array([0.01, 0.02, 0.1, 0.2, 0.3, 0.4])
    matrices = _straight(lags, np.eye(2), -np.eye(2))
    matrices[2] = 50.0                      # the trace maximum, after the cut
    _, start = lag_intercept(lags, matrices, min_lag=0.05)
    assert start == 2
    assert lags[start] == pytest.approx(0.1)


def test_the_branch_starts_at_the_maximum_of_the_trace():
    lags = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    matrices = _straight(lags, np.eye(2) * 2.0, -np.eye(2))
    matrices[1] += np.eye(2) * 100.0
    _, start = lag_intercept(lags, matrices, min_lag=0.0)
    assert start == 1


def test_the_branch_leaves_at_least_three_lags_to_fit():
    lags = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    matrices = _straight(lags, np.eye(2), -np.eye(2))
    matrices[-1] += np.eye(2) * 100.0       # maximum at the very end
    _, start = lag_intercept(lags, matrices, min_lag=0.0)
    assert start == 2


def test_a_window_too_short_after_the_cut_falls_back_to_every_lag():
    lags = np.array([0.01, 0.02, 0.03, 0.9])
    matrices = _straight(lags, np.eye(2) * 7.0, np.zeros((2, 2)))
    out, start = lag_intercept(lags, matrices, min_lag=0.5)
    assert start == 0
    assert np.allclose(out, np.eye(2) * 7.0)


def test_fewer_than_three_lags_is_not_a_measurement():
    with pytest.raises(ValueError, match="has no residual"):
        lag_intercept(np.array([0.1, 0.2]), np.ones((2, 2, 2)), min_lag=0.0)


def test_the_lag_times_are_one_dimensional():
    with pytest.raises(ValueError, match="one time per lag"):
        lag_intercept(np.ones((3, 1)), np.ones((3, 2, 2)), min_lag=0.0)


def test_there_is_one_matrix_per_lag():
    with pytest.raises(ValueError, match="against 3 lags"):
        lag_intercept(np.array([0.1, 0.2, 0.3]), np.ones((4, 2, 2)),
                      min_lag=0.0)


def test_a_mobility_matrix_is_square_in_its_channels():
    with pytest.raises(ValueError, match="not square"):
        lag_intercept(np.array([0.1, 0.2, 0.3]), np.ones((3, 2, 3)),
                      min_lag=0.0)


# ==========================================================================
# shell_intercepts
# ==========================================================================

def test_every_shell_is_fitted_over_the_whole_window():
    lags = np.array([0.1, 0.2, 0.3, 0.4])
    spectrum = np.zeros((2, 4, 2, 2))
    spectrum[0] = _straight(lags, np.eye(2) * 3.0, -np.eye(2))
    spectrum[1] = _straight(lags, np.eye(2) * 5.0, np.eye(2))
    out = shell_intercepts(lags, spectrum, min_lag=0.0)
    assert np.allclose(out[0], np.eye(2) * 3.0)
    assert np.allclose(out[1], np.eye(2) * 5.0)


def test_the_shell_fit_does_not_search_for_a_branch():
    """A hump at the start is fitted THROUGH, unlike the band average."""
    lags = np.array([0.1, 0.2, 0.3, 0.4])
    spectrum = np.zeros((1, 4, 1, 1))
    spectrum[0] = _straight(lags, np.ones((1, 1)), -np.ones((1, 1)),
                            n_channels=1)
    spectrum[0, 0] += 10.0
    out = shell_intercepts(lags, spectrum, min_lag=0.0)
    assert out[0, 0, 0] != pytest.approx(1.0)


def test_a_shell_that_is_not_measured_at_every_lag_stays_not_a_number():
    lags = np.array([0.1, 0.2, 0.3])
    spectrum = np.zeros((2, 3, 1, 1))
    spectrum[1, 1] = np.nan
    out = shell_intercepts(lags, spectrum, min_lag=0.0)
    assert np.isfinite(out[0, 0, 0]) and np.isnan(out[1, 0, 0])


def test_the_shell_fit_falls_back_when_the_window_is_too_short():
    lags = np.array([0.01, 0.02, 0.03, 0.9])
    spectrum = np.full((1, 4, 1, 1), 2.0)
    out = shell_intercepts(lags, spectrum, min_lag=0.5)
    assert out[0, 0, 0] == pytest.approx(2.0)


def test_the_shell_fit_checks_it_has_one_lag_per_stored_lag():
    with pytest.raises(ValueError, match="against 3 lags"):
        shell_intercepts(np.array([0.1, 0.2, 0.3]), np.ones((2, 4, 1, 1)),
                         min_lag=0.0)


# ==========================================================================
# extrapolate_to_zero
# ==========================================================================

def test_the_extrapolation_is_a_straight_line_in_the_squared_wavenumber():
    centres = np.array([1.0, 2.0, 3.0])
    icpt = np.zeros((3, 1, 1))
    icpt[:, 0, 0] = 7.0 - 0.5 * centres ** 2
    assert extrapolate_to_zero(centres, icpt,
                               k_max=9.0)[0, 0] == pytest.approx(7.0)


def test_only_shells_at_or_below_the_cutoff_are_extrapolated_from():
    centres = np.array([1.0, 2.0, 10.0])
    icpt = np.zeros((3, 1, 1))
    icpt[:, 0, 0] = np.array([6.5, 5.0, -1000.0])
    assert extrapolate_to_zero(centres, icpt,
                               k_max=5.0)[0, 0] == pytest.approx(7.0)


def test_which_shells_are_measured_is_judged_on_the_first_channel_pair():
    """The archived quirk: one element decides the shell set for all of them."""
    centres = np.array([1.0, 2.0, 3.0])
    icpt = np.zeros((3, 2, 2))
    icpt[:, 0, 0] = 7.0 - 0.5 * centres ** 2
    icpt[:, 1, 1] = 4.0 - 0.25 * centres ** 2
    icpt[2, 1, 1] = np.nan          # finite in (0,0), absent in (1,1)
    out = extrapolate_to_zero(centres, icpt, k_max=9.0)
    assert out[0, 0] == pytest.approx(7.0)
    assert np.isnan(out[1, 1])


def test_a_line_to_zero_wavenumber_needs_two_measured_shells():
    centres = np.array([1.0, 2.0])
    icpt = np.zeros((2, 1, 1))
    icpt[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="needs two"):
        extrapolate_to_zero(centres, icpt, k_max=9.0)


def test_the_extrapolation_checks_it_has_one_intercept_per_shell():
    with pytest.raises(ValueError, match="against 3 shells"):
        extrapolate_to_zero(np.array([1.0, 2.0, 3.0]), np.ones((2, 1, 1)),
                            k_max=9.0)


# ==========================================================================
# the concentration mode
# ==========================================================================

def test_the_concentration_weights_are_the_crossed_fractions():
    assert np.array_equal(concentration_weights((0.3, 0.7)),
                          np.array([0.7, -0.3]))


def test_the_concentration_mode_vanishes_on_a_uniform_mixture():
    """z . rho is zero when the channels sit at their own fractions."""
    w = concentration_weights((0.25, 0.75))
    assert float(np.dot(w, np.array([0.25, 0.75]))) == pytest.approx(0.0)


def test_one_channel_has_no_concentration_mode_to_return():
    with pytest.raises(ValueError, match="exactly two channels"):
        concentration_weights((1.0,))


def test_three_channels_have_no_single_concentration_mode():
    with pytest.raises(ValueError, match="Pass explicit weights"):
        concentration_weights((0.2, 0.3, 0.5))


def test_the_concentration_series_is_the_weighted_channel_sum():
    rho = np.arange(2 * 2 * 3).reshape(2, 2, 3) * (1 + 1j)
    out = concentration_series(rho, weights=np.array([2.0, -3.0]))
    assert np.array_equal(out, 2.0 * rho[:, 0, :] - 3.0 * rho[:, 1, :])


def test_the_concentration_series_accumulates_in_channel_order():
    """Bit for bit against the written-out sum, not merely close to it."""
    rng = np.random.default_rng(3)
    rho = (rng.standard_normal((5, 3, 4)) + 1j * rng.standard_normal((5, 3, 4)))
    w = np.array([0.25, -0.5, 0.125])
    expected = (w[0] * rho[:, 0, :] + w[1] * rho[:, 1, :]) + w[2] * rho[:, 2, :]
    assert np.array_equal(concentration_series(rho, weights=w), expected)


def test_a_single_channel_still_has_a_density_mode():
    rho = np.ones((3, 1, 2), complex)
    assert np.array_equal(concentration_series(rho, weights=np.array([2.0])),
                          2.0 * np.ones((3, 2), complex))


def test_the_concentration_series_wants_a_timeline_of_channels_and_modes():
    with pytest.raises(ValueError, match="n_frames, n_channels, n_modes"):
        concentration_series(np.ones((3, 2)), weights=np.array([1.0, 1.0]))


def test_there_is_one_weight_per_channel():
    with pytest.raises(ValueError, match="against 2 channels"):
        concentration_series(np.ones((3, 2, 4), complex),
                             weights=np.array([1.0]))


# ==========================================================================
# quarter_growth and time_fifths
# ==========================================================================

def test_a_stationary_run_has_no_growth():
    assert quarter_growth(np.ones(20)) == pytest.approx(0.0)


def test_a_run_that_doubles_grows_by_one():
    s = np.concatenate([np.ones(10), np.full(10, 2.0)])
    assert quarter_growth(s) == pytest.approx(1.0)


def test_the_quarter_is_an_integer_division_of_the_frame_count():
    s = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0])   # 7 frames, quarter 1
    assert quarter_growth(s) == pytest.approx(3.0)


def test_a_run_shorter_than_four_frames_has_no_quarter():
    with pytest.raises(ValueError, match="quarter of the run is empty"):
        quarter_growth(np.ones(3))


def test_the_growth_statistic_is_one_value_per_frame():
    with pytest.raises(ValueError, match="one value per frame"):
        quarter_growth(np.ones((8, 2)))


def test_the_fifths_are_the_means_of_five_index_slices():
    assert np.array_equal(time_fifths(np.arange(10.0)),
                          np.array([0.5, 2.5, 4.5, 6.5, 8.5]))


def test_the_fifths_of_an_uneven_run_split_on_the_index():
    out = time_fifths(np.arange(7.0))
    assert out[0] == pytest.approx(0.0)      # frames [0, 1)
    assert out[4] == pytest.approx(5.5)      # frames [5, 7)


def test_a_block_of_values_averages_over_everything_in_the_fifth():
    values = np.arange(20.0).reshape(10, 2)
    assert np.array_equal(time_fifths(values),
                          np.array([1.5, 5.5, 9.5, 13.5, 17.5]))


def test_a_run_shorter_than_five_frames_has_an_empty_fifth():
    with pytest.raises(ValueError, match="at least one fifth would be empty"):
        time_fifths(np.ones(4))


# ==========================================================================
# the single-phase gate
# ==========================================================================

def test_condensation_needs_both_a_rise_and_a_meaningful_level():
    rising_but_small = np.array([1e-6, 1e-5, 1e-5, 1e-5, 1e-5])
    assert not condensing(rising_but_small, 0.5, ratio=2.5, ideal_factor=3.0)
    high_but_flat = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    assert not condensing(high_but_flat, 0.5, ratio=2.5, ideal_factor=3.0)
    both = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert condensing(both, 0.5, ratio=2.5, ideal_factor=3.0)


def test_a_vanishing_first_fifth_does_not_divide_by_zero():
    assert condensing(np.array([0.0, 1.0, 1.0, 1.0, 1.0]), 0.5, ratio=2.5,
                      ideal_factor=3.0)


def test_the_evidence_is_five_numbers():
    with pytest.raises(ValueError, match="not five values"):
        condensing(np.ones(4), 0.5, ratio=2.5, ideal_factor=3.0)


def test_a_stationary_run_at_a_condensed_level_is_not_single_phase():
    """Stationarity alone would let demixed runs anchor."""
    fifths = np.full(5, 88.0 * 0.25)
    assert not single_phase(0.06, fifths, 0.5, 1000.0, t_min=None, **_GATE)


def test_a_stationary_run_at_a_near_ideal_level_is_single_phase():
    fifths = np.full(5, 2.0 * 0.25)
    assert single_phase(0.06, fifths, 0.5, 1000.0, t_min=None, **_GATE)


def test_above_the_critical_temperature_the_level_test_is_waived():
    fifths = np.full(5, 88.0 * 0.25)
    assert single_phase(0.06, fifths, 0.5, 9000.0, t_min=8973.0, **_GATE)
    assert not single_phase(0.06, fifths, 0.5, 8000.0, t_min=8973.0, **_GATE)


def test_the_waiver_is_inclusive_at_the_critical_temperature():
    fifths = np.full(5, 88.0 * 0.25)
    assert single_phase(0.06, fifths, 0.5, 8973.0, t_min=8973.0, **_GATE)


def test_without_a_declared_critical_temperature_nothing_is_waived():
    fifths = np.full(5, 88.0 * 0.25)
    assert not single_phase(0.06, fifths, 0.5, 1e9, t_min=None, **_GATE)


def test_a_growing_run_the_evidence_acquits_is_still_single_phase():
    fifths = np.array([1.0, 1.1, 1.0, 1.1, 1.2])      # rise too small
    assert single_phase(0.9, fifths, 0.5, 1000.0, t_min=None, **_GATE)


def test_a_growing_run_the_evidence_convicts_is_not_single_phase():
    fifths = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert not single_phase(0.9, fifths, 0.5, 1000.0, t_min=None, **_GATE)


def test_a_run_with_no_evidence_is_judged_on_its_growth_alone():
    assert single_phase(0.06, None, 0.5, 1000.0, t_min=None, **_GATE)
    assert not single_phase(0.9, None, 0.5, 1000.0, t_min=None, **_GATE)


# ==========================================================================
# run_mobility
# ==========================================================================

def _run_case(n_channels=2, n_frames=12, seed=5):
    rng = np.random.default_rng(seed)
    nvec = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0]])
    rho = (rng.standard_normal((n_frames, n_channels, len(nvec)))
           + 1j * rng.standard_normal((n_frames, n_channels, len(nvec))))
    box = np.full((n_frames, 3), 2 * np.pi)
    return rho, box, nvec


def test_a_run_measurement_drops_the_uniform_mode():
    rho, box, nvec = _run_case()
    out = run_mobility(rho, box, nvec, temperature=300.0, thermal_energy=1.0,
                       lags=[1, 2, 3], frame_interval=0.02, band=(0.0, 9.0),
                       min_lag=0.0, extrapolation_k_max=9.0,
                           reference_rule="time_mean")
    assert list(out.kept) == [False, True, True, True, True]
    assert out.shell_of_mode.shape == (4,)


def test_a_run_measurement_is_generic_over_the_channel_count():
    for n_channels in (1, 2, 3):
        rho, box, nvec = _run_case(n_channels)
        out = run_mobility(rho, box, nvec, temperature=1.0,
                           thermal_energy=1.0, lags=[1, 2, 3],
                           frame_interval=1.0, band=(0.0, 9.0), min_lag=0.0,
                           extrapolation_k_max=9.0,
                           reference_rule="time_mean")
        assert out.M_window.shape == (n_channels, n_channels)
        assert out.M_zero_wavenumber.shape == (n_channels, n_channels)
        assert out.spectrum.shape[2:] == (n_channels, n_channels)


def test_a_run_measurement_records_its_reference_volume_and_temperature():
    rho, box, nvec = _run_case()
    box[:, 0] = np.linspace(6.0, 8.0, len(box))
    out = run_mobility(rho, box, nvec, temperature=1234.0,
                       thermal_energy=1.0, lags=[1, 2, 3], frame_interval=1.0,
                       band=(0.0, 9.0), min_lag=0.0, extrapolation_k_max=9.0,
                           reference_rule="time_mean")
    assert out.temperature == 1234.0
    assert out.volume == pytest.approx(7.0 * (2 * np.pi) ** 2)


def test_the_band_reaches_the_fundamental_shell_even_when_it_rounds_low():
    """The lower edge is pulled back below the smallest wavenumber present."""
    rho, box, nvec = _run_case()
    out = run_mobility(rho, box, nvec, temperature=1.0, thermal_energy=1.0,
                       lags=[1, 2, 3], frame_interval=1.0, band=(1.0, 9.0),
                       min_lag=0.0, extrapolation_k_max=9.0,
                           reference_rule="time_mean")
    assert np.isfinite(out.M_window).all()


def test_the_reference_box_rule_is_the_callers_and_changes_the_answer():
    """A fixed-box campaign took the first frame, a barostatted one the mean."""
    rho, box, nvec = _run_case()
    box[:, 0] = np.linspace(6.0, 8.0, len(box))
    kwargs = dict(temperature=1.0, thermal_energy=1.0, lags=[1, 2, 3],
                  frame_interval=1.0, band=(0.0, 9.0), min_lag=0.0,
                  extrapolation_k_max=9.0)
    averaged = run_mobility(rho, box, nvec, reference_rule="time_mean",
                            **kwargs)
    first = run_mobility(rho, box, nvec, reference_rule="first_frame",
                         **kwargs)
    assert averaged.volume == pytest.approx(7.0 * (2 * np.pi) ** 2)
    assert first.volume == pytest.approx(6.0 * (2 * np.pi) ** 2)
    assert not np.array_equal(averaged.shell_wavenumbers,
                              first.shell_wavenumbers)


def test_an_unknown_reference_box_rule_is_refused():
    rho, box, nvec = _run_case()
    with pytest.raises(ValueError, match="not one of"):
        run_mobility(rho, box, nvec, temperature=1.0, thermal_energy=1.0,
                     lags=[1, 2, 3], frame_interval=1.0, band=(0.0, 9.0),
                     min_lag=0.0, extrapolation_k_max=9.0,
                     reference_rule="last_frame")


def test_a_run_measurement_wants_one_label_per_stored_mode():
    rho, box, nvec = _run_case()
    with pytest.raises(ValueError, match="labels against"):
        run_mobility(rho, box, nvec[:3], temperature=1.0, thermal_energy=1.0,
                     lags=[1], frame_interval=1.0, band=(0.0, 9.0),
                     min_lag=0.0, extrapolation_k_max=9.0,
                           reference_rule="time_mean")


def test_a_run_measurement_wants_a_timeline_of_channels_and_modes():
    _, box, nvec = _run_case()
    with pytest.raises(ValueError, match="n_frames, n_channels, n_modes"):
        run_mobility(np.ones((4, 5), complex), box, nvec, temperature=1.0,
                     thermal_energy=1.0, lags=[1], frame_interval=1.0,
                     band=(0.0, 9.0), min_lag=0.0, extrapolation_k_max=9.0,
                           reference_rule="time_mean")


def test_the_trace_diagnostic_is_the_sum_of_the_measured_diagonals():
    rho, box, nvec = _run_case()
    out = run_mobility(rho, box, nvec, temperature=1.0, thermal_energy=1.0,
                       lags=[1, 2, 3], frame_interval=1.0, band=(0.0, 9.0),
                       min_lag=0.0, extrapolation_k_max=9.0,
                           reference_rule="time_mean")
    expected = shell_intercepts(
        out.lags_ps,
        (out.spectrum[:, :, 0, 0] + out.spectrum[:, :, 1, 1])[:, :, None,
                                                              None],
        min_lag=0.0)[:, 0, 0]
    assert np.array_equal(out.trace_intercepts, expected, equal_nan=True)


# ==========================================================================
# kernel_shape and shape_amplitude
# ==========================================================================

def _shape_run(temperature, values, k=(1.0, 2.0, 3.0), n_channels=2):
    k = np.asarray(k, float)
    icpt = np.zeros((len(k), n_channels, n_channels))
    for a in range(n_channels):
        for b in range(n_channels):
            icpt[:, a, b] = values
    return _stub_run(temperature, k, icpt)


def _stub_run(temperature, k, icpt):
    n = icpt.shape[1]
    return type("Stub", (), dict(
        shell_wavenumbers=np.asarray(k, float), shell_intercepts=icpt,
        temperature=float(temperature), M_window=np.zeros((n, n)),
        M_zero_wavenumber=np.zeros((n, n))))()


def test_identical_runs_give_their_own_normalised_shape_with_no_spread():
    values = np.array([4.0, 3.0, 1.0])
    runs = [_shape_run(2000.0, values) for _ in range(3)]
    shape = kernel_shape(runs, t_min=1000.0, shape_set="above_temperature")
    y0 = np.polyfit(np.array([1.0, 2.0]) ** 2, values[:2], 1)[1]
    assert np.allclose(shape.kernel[:, 0, 0], values / y0)
    assert np.allclose(shape.spread, 0.0)
    assert shape.n_runs == 3 and shape.t_min == 1000.0


def test_only_runs_at_or_above_the_shape_temperature_enter_it():
    hot = _shape_run(2000.0, np.array([4.0, 3.0, 1.0]))
    cold = _shape_run(500.0, np.array([40.0, 30.0, 10.0]))
    shape = kernel_shape([cold, hot], t_min=1000.0,
                         shape_set="above_temperature")
    assert shape.n_runs == 1


def test_the_reference_axis_is_the_first_qualifying_run_and_short_runs_clamp():
    """The archived bias: a shorter run contributes a clamped constant."""
    long_run = _shape_run(2000.0, np.array([4.0, 3.0, 1.0]),
                          k=(1.0, 2.0, 3.0))
    short_run = _shape_run(2000.0, np.array([4.0, 3.0]), k=(1.0, 2.0))
    shape = kernel_shape([long_run, short_run], t_min=0.0,
                         shape_set="above_temperature")
    assert np.array_equal(shape.wavenumbers, np.array([1.0, 2.0, 3.0]))
    # at the top shell the short run contributes its own last value, so the
    # spread there is smaller than a real measurement would give
    assert shape.kernel.shape == (3, 2, 2)
    assert shape.spread[2, 0, 0] > 0.0


def test_a_shell_the_run_did_not_measure_is_left_out_of_its_shape():
    icpt = np.zeros((3, 2, 2))
    icpt[:, :, :] = np.array([4.0, 3.0, 1.0])[:, None, None]
    icpt[2, 0, 0] = np.nan
    run = _stub_run(2000.0, np.array([1.0, 2.0, 3.0]), icpt)
    shape = kernel_shape([run], t_min=0.0, shape_set="above_temperature")
    assert shape.wavenumbers.shape == (2,)


def test_an_unknown_shape_set_is_refused_and_the_known_ones_named():
    with pytest.raises(ValueError, match="not one of"):
        kernel_shape([_shape_run(1.0, np.array([1.0, 2.0, 3.0]))], t_min=0.0,
                     shape_set="hot")


def test_the_phase_restricted_shape_set_needs_the_verdicts():
    runs = [_shape_run(1.0, np.array([4.0, 3.0, 1.0]))]
    with pytest.raises(ValueError, match="needs single_phase_flags"):
        kernel_shape(runs, t_min=0.0,
                     shape_set="single_phase_above_temperature")


def test_the_phase_restricted_shape_set_drops_the_flagged_runs():
    good = _shape_run(2000.0, np.array([4.0, 3.0, 1.0]))
    bad = _shape_run(2000.0, np.array([40.0, 3.0, 1.0]))
    shape = kernel_shape([good, bad], t_min=0.0,
                         shape_set="single_phase_above_temperature",
                         single_phase_flags=[True, False])
    assert shape.n_runs == 1


def test_a_verdict_per_run_is_required_by_the_restricted_set():
    runs = [_shape_run(1.0, np.array([4.0, 3.0, 1.0]))] * 2
    with pytest.raises(ValueError, match="verdicts against"):
        kernel_shape(runs, t_min=0.0,
                     shape_set="single_phase_above_temperature",
                     single_phase_flags=[True])


def test_the_temperature_only_shape_set_refuses_verdicts_it_would_ignore():
    runs = [_shape_run(1.0, np.array([4.0, 3.0, 1.0]))]
    with pytest.raises(ValueError, match="cannot honour"):
        kernel_shape(runs, t_min=0.0, shape_set="above_temperature",
                     single_phase_flags=[True])


def test_a_shape_set_with_no_run_in_it_is_refused():
    runs = [_shape_run(100.0, np.array([4.0, 3.0, 1.0]))]
    with pytest.raises(ValueError, match="top temperature"):
        kernel_shape(runs, t_min=5000.0, shape_set="above_temperature")


def test_an_exact_multiple_of_the_shape_recovers_its_amplitude():
    values = np.array([4.0, 3.0, 1.0])
    shape = kernel_shape([_shape_run(1.0, values)], t_min=0.0,
                         shape_set="above_temperature")
    run = _shape_run(1.0, 5.0 * values)
    amplitude, residual = shape_amplitude(run, shape, k_max=9.0)
    y0 = np.polyfit(np.array([1.0, 2.0]) ** 2, values[:2], 1)[1]
    assert np.allclose(amplitude, 5.0 * y0)
    assert np.allclose(residual, 0.0)


def test_the_amplitude_fit_uses_only_shells_inside_its_cutoff():
    values = np.array([4.0, 3.0, 1.0])
    shape = kernel_shape([_shape_run(1.0, values)], t_min=0.0,
                         shape_set="above_temperature")
    spoiled = values.copy()
    spoiled[2] = -1000.0
    inside = shape_amplitude(_shape_run(1.0, spoiled), shape, k_max=2.5)[0]
    outside = shape_amplitude(_shape_run(1.0, spoiled), shape, k_max=9.0)[0]
    assert not np.allclose(inside, outside)


def test_an_amplitude_with_no_shell_inside_the_cutoff_is_refused():
    values = np.array([4.0, 3.0, 1.0])
    shape = kernel_shape([_shape_run(1.0, values)], t_min=0.0,
                         shape_set="above_temperature")
    with pytest.raises(ValueError, match="no measured shell"):
        shape_amplitude(_shape_run(1.0, values), shape, k_max=0.1)


# ==========================================================================
# the table object
# ==========================================================================

def _campaign(n_rows=3, n_channels=2):
    rng = np.random.default_rng(17)
    runs = []
    for i in range(n_rows):
        rho, box, nvec = _run_case(n_channels, seed=i + 1)
        runs.append(run_mobility(rho, box, nvec, temperature=1000.0 + 100 * i,
                                 thermal_energy=1.0, lags=[1, 2, 3, 4],
                                 frame_interval=1.0, band=(0.0, 9.0),
                                 min_lag=0.0, extrapolation_k_max=9.0,
                           reference_rule="time_mean"))
    shape = kernel_shape(runs, t_min=0.0, shape_set="above_temperature")
    return runs, shape, rng


def test_a_table_records_the_pressure_it_was_declared_at():
    runs, shape, _ = _campaign()
    table = anchor_table(runs, pressure=400.0, composition=np.full(3, 0.5),
                         densities=np.ones((3, 2)), growth=np.zeros(3),
                         single_phase=np.ones(3, bool), shape=shape,
                         shape_k_max=9.0)
    assert table.pressure == 400.0


def test_a_system_that_controls_no_pressure_declares_none():
    runs, shape, _ = _campaign()
    table = anchor_table(runs, pressure=None, composition=np.full(3, 0.5),
                         densities=np.ones((3, 2)), growth=np.zeros(3),
                         single_phase=np.ones(3, bool), shape=shape,
                         shape_k_max=9.0)
    assert table.pressure is None


def test_a_table_is_generic_over_the_channel_count():
    for n_channels in (1, 2, 3):
        runs, shape, _ = _campaign(n_channels=n_channels)
        table = anchor_table(runs, pressure=0.0, composition=np.full(3, 0.5),
                             densities=np.ones((3, n_channels)),
                             growth=np.zeros(3),
                             single_phase=np.ones(3, bool), shape=shape,
                             shape_k_max=9.0)
        assert table.M_shape.shape == (3, n_channels, n_channels)


def test_an_empty_campaign_has_no_table():
    _, shape, _ = _campaign()
    with pytest.raises(ValueError, match="no runs"):
        anchor_table([], pressure=0.0, composition=np.zeros(0),
                     densities=np.zeros((0, 2)), growth=np.zeros(0),
                     single_phase=np.zeros(0, bool), shape=shape,
                     shape_k_max=9.0)


def test_a_table_wants_one_composition_per_run():
    runs, shape, _ = _campaign()
    with pytest.raises(ValueError, match="composition has shape"):
        anchor_table(runs, pressure=0.0, composition=np.full(2, 0.5),
                     densities=np.ones((3, 2)), growth=np.zeros(3),
                     single_phase=np.ones(3, bool), shape=shape,
                     shape_k_max=9.0)


def test_a_table_wants_one_growth_statistic_per_run():
    runs, shape, _ = _campaign()
    with pytest.raises(ValueError, match="growth has shape"):
        anchor_table(runs, pressure=0.0, composition=np.full(3, 0.5),
                     densities=np.ones((3, 2)), growth=np.zeros(2),
                     single_phase=np.ones(3, bool), shape=shape,
                     shape_k_max=9.0)


def test_a_table_wants_one_verdict_per_run():
    runs, shape, _ = _campaign()
    with pytest.raises(ValueError, match="single_phase has shape"):
        anchor_table(runs, pressure=0.0, composition=np.full(3, 0.5),
                     densities=np.ones((3, 2)), growth=np.zeros(3),
                     single_phase=np.ones(2, bool), shape=shape,
                     shape_k_max=9.0)


def test_a_table_wants_a_density_per_channel_per_run():
    runs, shape, _ = _campaign()
    with pytest.raises(ValueError, match="densities has shape"):
        anchor_table(runs, pressure=0.0, composition=np.full(3, 0.5),
                     densities=np.ones((3, 3)), growth=np.zeros(3),
                     single_phase=np.ones(3, bool), shape=shape,
                     shape_k_max=9.0)


def test_the_table_keeps_the_row_order_it_was_given():
    runs, shape, _ = _campaign()
    table = anchor_table(runs[::-1], pressure=0.0, composition=np.full(3, 0.5),
                         densities=np.ones((3, 2)), growth=np.zeros(3),
                         single_phase=np.ones(3, bool), shape=shape,
                         shape_k_max=9.0)
    assert list(table.temperature) == [1200.0, 1100.0, 1000.0]


def test_the_stored_matrix_a_diagnostic_uses_is_a_named_choice():
    table = _table(0.0, [1.0], [True])
    assert np.array_equal(table.matrices("window"), np.ones((1, 2, 2)))
    assert np.array_equal(table.matrices("zero_wavenumber"),
                          np.full((1, 2, 2), 2.0))
    with pytest.raises(ValueError, match="not one of"):
        table.matrices("shape")


def test_the_density_scaled_columns_divide_by_the_right_densities():
    table = _table(0.0, [1.0], [True])
    table = AnchorTable(**{**vars(table),
                           "densities": np.array([[2.0, 8.0]])})
    diagonal, cross = table.density_scaled("window")
    assert diagonal[0, 0] == pytest.approx(0.5)
    assert diagonal[0, 1] == pytest.approx(0.125)
    assert cross[0, 0] == pytest.approx(1.0 / np.sqrt(16.0))


def test_a_single_channel_table_has_no_cross_columns_at_all():
    runs, shape, _ = _campaign(n_channels=1)
    table = anchor_table(runs, pressure=0.0, composition=np.full(3, 1.0),
                         densities=np.ones((3, 1)), growth=np.zeros(3),
                         single_phase=np.ones(3, bool), shape=shape,
                         shape_k_max=9.0)
    diagonal, cross = table.density_scaled("zero_wavenumber")
    assert diagonal.shape == (3, 1)
    assert cross.shape == (3, 0)


def test_the_cross_columns_run_over_the_upper_triangle_in_order():
    runs, shape, _ = _campaign(n_channels=3)
    table = anchor_table(runs, pressure=0.0, composition=np.full(3, 0.5),
                         densities=np.ones((3, 3)), growth=np.zeros(3),
                         single_phase=np.ones(3, bool), shape=shape,
                         shape_k_max=9.0)
    _, cross = table.density_scaled("window")
    assert cross.shape == (3, 3)
    assert np.allclose(cross[:, 0], table.M_window[:, 0, 1])
    assert np.allclose(cross[:, 1], table.M_window[:, 0, 2])
    assert np.allclose(cross[:, 2], table.M_window[:, 1, 2])


def test_the_einstein_column_is_the_mobility_times_the_thermal_energy():
    table = _table(0.0, [1.0], [True])
    table = AnchorTable(**{**vars(table),
                           "densities": np.array([[2.0, 4.0]])})
    out = table.einstein_diffusivity("window", np.array([10.0]))
    assert out[0, 0] == pytest.approx(5.0)
    assert out[0, 1] == pytest.approx(2.5)


# ==========================================================================
# anchor_rows: the pressure, and what replaces the path regex
# ==========================================================================

def test_only_single_phase_rows_above_the_floor_may_anchor():
    system = _system({800: 8973.0})
    table = _table(800.0, [8000.0, 9000.0, 9500.0], [True, True, False])
    assert list(anchor_rows(table, system)) == [False, True, False]


def test_the_temperature_floor_is_a_strict_inequality():
    system = _system({800: 8973.0})
    table = _table(800.0, [8973.0], [True])
    assert list(anchor_rows(table, system)) == [False]


def test_a_table_with_no_declared_pressure_is_refused_not_left_unfiltered():
    """What the path regex did silently, and the reason it is forbidden."""
    system = _system({800: 8973.0})
    table = _table(None, [1000.0], [True])
    with pytest.raises(ValueError, match="declares no pressure"):
        anchor_rows(table, system)


def test_a_pressure_the_cut_table_does_not_cover_is_refused():
    system = _system({200: 6100.0, 800: 8973.0})
    table = _table(500.0, [1000.0], [True])
    with pytest.raises(ValueError, match="not among them"):
        anchor_rows(table, system)


def test_a_system_that_declares_no_floor_at_all_applies_none():
    system = _system({})
    table = _table(None, [1.0, 2.0], [True, False])
    assert list(anchor_rows(table, system)) == [True, False]


def test_the_admissible_rows_do_not_alias_the_table():
    system = _system({})
    table = _table(None, [1.0], [True])
    out = anchor_rows(table, system)
    out[0] = False
    assert bool(table.single_phase[0]) is True


# ==========================================================================
# The archive
# ==========================================================================

def _load_run_results(pressure):
    """The archived per-run measurements, as stub runs the aggregate takes."""
    pattern = f"{_FIELDS}/fdt_{pressure}GPa/cube_*/fdt.npz"
    rows = []
    for path in sorted(glob.glob(pattern)):
        z = np.load(path)
        sibling = os.path.join(os.path.dirname(path), "static_s.npz")
        fifths = None
        if os.path.exists(sibling):
            s = np.load(sibling)
            if "scc_kmin_fifths" in s.files:
                fifths = s["scc_kmin_fifths"]
        rows.append(dict(
            run=_stub_run(float(z["T_K"]), z["k_shells"], z["M_k_icpt"]),
            M_window=z["M"], M_zero=z["M_k0"], x=float(z["x_He"]),
            T=float(z["T_K"]), growth=float(z["scc_growth"]),
            densities=np.array([float(z["rho_H"]), float(z["rho_He"])]),
            fifths=fifths))
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["x"], rows[i]["T"]))
    return [rows[i] for i in order]


def _rebuild(pressure):
    rows = _load_run_results(pressure)
    for row in rows:
        run = row["run"]
        run.M_window = row["M_window"]
        run.M_zero_wavenumber = row["M_zero"]
    flags = np.array([single_phase(r["growth"], r["fifths"], r["x"], r["T"],
                                   t_min=_T_C[pressure], **_GATE)
                      for r in rows], bool)
    shape = kernel_shape([r["run"] for r in rows],
                         t_min=_SHAPE_T_MIN[pressure],
                         shape_set="above_temperature")
    return anchor_table(
        [r["run"] for r in rows], pressure=float(pressure),
        composition=np.array([r["x"] for r in rows]),
        densities=np.stack([r["densities"] for r in rows]),
        growth=np.array([r["growth"] for r in rows]),
        single_phase=flags, shape=shape, shape_k_max=_SHAPE_K_MAX)


@archive
@pytest.mark.parametrize("pressure", _PRESSURES)
def test_the_published_table_is_the_rebuilt_one_plus_a_recorded_overlay(
        pressure):
    """Nine verdicts were forced by hand after the tables were built.

    They are DATA, not a rule: recorded as the rows they name, not re-derived.
    Anything else in the published table has to equal the rebuild.
    """
    table = _rebuild(pressure)
    published = np.load(f"{_FIELDS}/fdt_{pressure}GPa/M_table.npz")
    overridden = np.zeros(len(table.composition), bool)
    for p, x, t in _HAND_OVERRIDE:
        if p == pressure:
            overridden |= ((table.composition == x) & (table.temperature == t))
    expected = table.single_phase & ~overridden
    assert np.array_equal(expected, published["single_phase"])
    assert np.array_equal(table.M_shape, published["M_kappa0"])
    assert overridden.sum() == sum(1 for p, _, _ in _HAND_OVERRIDE
                                   if p == pressure)


@archive
def test_the_two_names_of_one_archived_table_are_the_same_bytes():
    """The published model reads a name that carries no pressure. It is a link.

    The published model's filter recovered the pressure from the directory name and
    applied NO floor when the name did not match, so the same bytes under
    this second name kept rows the first name cut. The bytes being identical
    is what makes that a defect of the filter rather than of the data.
    """
    linked = f"{_FIELDS}/fdt/M_table.npz"
    if not os.path.exists(linked):
        pytest.skip("the second name is not on this machine")
    a = np.load(linked)
    b = np.load(f"{_FIELDS}/fdt_800GPa/M_table.npz")
    assert set(a.files) == set(b.files)
    assert all(np.array_equal(a[k], b[k]) for k in a.files)


@archive
def test_the_declared_pressure_and_not_the_name_decides_the_anchor_rows():
    """One table, two systems: the rows follow the declaration."""
    table = _rebuild(800)
    strict = _system({800: 8973.0})
    lenient = _system({800: 0.0})
    assert int(anchor_rows(table, strict).sum()) \
        < int(anchor_rows(table, lenient).sum())
    assert int(anchor_rows(table, lenient).sum()) \
        == int(table.single_phase.sum())


@archive
def test_an_archived_run_measurement_is_rebuilt_from_its_stored_modes():
    """The per-run stage, on one run, against the file it wrote.

    Everything up to the least-squares fits is bit for bit, in every
    environment. The fits go through a library routine whose result depends
    on the linear-algebra build, and in the environment the archive was
    written in they are bit for bit too -- 79 of 79 runs of one pressure,
    whole. In a newer one they move by at most 8 units in the last place OF
    THE ARRAY'S PEAK, measured over 20 runs, which is the bound below with a
    factor of two in hand. Against an array's OWN magnitude the same
    deviation reads as up to 1492 units, because an intercept can sit near a
    cancellation, and that number says nothing about the fit; the peak is
    what a least-squares residual is bounded against.

    The next test pins the cause: the port's arithmetic is identical to the
    expression it replaces, in whichever environment it runs.
    """
    pressure, tag = 200, "cube_x0.05_T02000"
    modes = f"{_FIELDS}/modes_{pressure}GPa/{tag}/modes.npz"
    if not os.path.exists(modes):
        pytest.skip(f"the archived run is not on this machine: under {_FIELDS}"
                    " (the system's raw root: AIPF_RAW_<SYSTEM>, AIPF_RAW or [paths.raw] in aipf.toml)")
    z = np.load(modes)
    run = run_mobility(np.moveaxis(z["rho_k"], 1, 2), z["box"], z["nvec"],
                       temperature=float(z["T_K"]),
                       thermal_energy=_KB * float(z["T_K"]), lags=_LAGS,
                       frame_interval=float(z["dt_frame_ps"]), band=_BAND,
                       min_lag=_MIN_LAG, extrapolation_k_max=_EXTRAP_K_MAX,
                       reference_rule="time_mean")
    reference = np.load(f"{_FIELDS}/fdt_{pressure}GPa/{tag}/fdt.npz")
    for name, mine in (("k_shells", run.shell_wavenumbers),
                       ("M_k", run.spectrum), ("M_app", run.band),
                       ("lags_ps", run.lags_ps),
                       ("V_mean", np.float64(run.volume))):
        assert np.array_equal(mine, reference[name], equal_nan=True), name
    assert run.lag_start == int(reference["i0"])
    for name, mine in (("M", run.M_window), ("M_k0", run.M_zero_wavenumber),
                       ("M_k_icpt", run.shell_intercepts),
                       ("tr_icpt", run.trace_intercepts)):
        ref = np.asarray(reference[name], float)
        finite = np.isfinite(ref)
        gap = np.abs(np.asarray(mine, float)[finite] - ref[finite])
        step = np.spacing(np.abs(ref[finite]).max())
        assert gap.max() <= 16 * step, (name, float(gap.max() / step))


@archive
def test_the_least_squares_deviation_is_the_library_and_not_the_port():
    """The port's own arithmetic is identical to the code it replaces.

    Whatever the linear-algebra build does to the intercept, it does to both,
    so a deviation from the archive is the environment and not this module.
    """
    pressure, tag = 200, "cube_x0.05_T02000"
    path = f"{_FIELDS}/fdt_{pressure}GPa/{tag}/fdt.npz"
    if not os.path.exists(path):
        pytest.skip(f"the archived run is not on this machine: under {_FIELDS}"
                    " (the system's raw root: AIPF_RAW_<SYSTEM>, AIPF_RAW or [paths.raw] in aipf.toml)")
    z = np.load(path)
    lags_ps, spectrum = z["lags_ps"], z["M_k"]
    window = lags_ps >= _MIN_LAG
    reference = np.full(spectrum.shape[0:1] + spectrum.shape[2:], np.nan)
    for s in range(spectrum.shape[0]):
        if np.isfinite(spectrum[s][window]).all():
            for a in range(2):
                for b in range(2):
                    reference[s, a, b] = np.polyfit(
                        lags_ps[window], spectrum[s, window, a, b], 1)[1]
    mine = shell_intercepts(lags_ps, spectrum, min_lag=_MIN_LAG)
    assert np.array_equal(mine, reference, equal_nan=True)


@archive
def test_the_stationarity_evidence_is_rebuilt_from_the_stored_modes():
    """The fifths the gate reads, which live in a sibling artefact."""
    pressure, tag = 200, "cube_x0.05_T02000"
    modes = f"{_FIELDS}/modes_{pressure}GPa/{tag}/modes.npz"
    sibling = f"{_FIELDS}/fdt_{pressure}GPa/{tag}/static_s.npz"
    if not (os.path.exists(modes) and os.path.exists(sibling)):
        pytest.skip(f"the archived run is not on this machine: under {_FIELDS}"
                    " (the system's raw root: AIPF_RAW_<SYSTEM>, AIPF_RAW or [paths.raw] in aipf.toml)")
    z = np.load(modes)
    amplitudes = np.moveaxis(z["rho_k"], 1, 2)
    lengths = np.asarray(z["box"], float).mean(0)
    k = np.linalg.norm(2 * np.pi * np.asarray(z["nvec"], float) / lengths,
                       axis=1)
    kept = k > 1e-12
    total = float(z["n_H"]) + float(z["n_He"])
    measured = float(z["n_He"]) / total
    c = concentration_series(
        amplitudes[:, :, kept],
        weights=concentration_weights((1.0 - measured, measured)))
    fundamental = np.abs(k[kept] - k[kept].min()) < 1e-9
    volume = float(np.prod(lengths))
    mine = (time_fifths(np.abs(c[:, fundamental]) ** 2)
            / volume / (total / volume))
    assert np.array_equal(mine, np.load(sibling)["scc_kmin_fifths"])


@archive
def test_the_second_campaigns_tables_are_reached_and_their_shape_reproduced():
    """The other tree's equivalents: its shape and amplitudes, not its gate.

    Its single-phase verdicts come from a quality-control campaign, which is
    a different module's business. What IS this module's is the shape, the
    amplitudes and the residuals, and those are rebuilt from its stored
    per-run intercepts.
    """
    root = str(declared_roots.raw("feb", "fields"))
    if not os.path.isdir(root):
        pytest.skip("the second campaign is not on this machine")
    checked = 0
    for tree in ("fdt_0GPa_v2", "fdt_5GPa_v2", "fdt_10GPa_v2"):
        table_path = f"{root}/{tree}/M_table.npz"
        runs_dir = f"{root}/{tree}_runs"
        if not (os.path.exists(table_path) and os.path.isdir(runs_dir)):
            continue
        reference = np.load(table_path, allow_pickle=True)
        rows = []
        for path in sorted(glob.glob(f"{runs_dir}/*/fdt.npz")):
            z = np.load(path)
            rows.append((str(z["tag"]), float(z["x_B"]), float(z["T_K"]),
                         _stub_run(float(z["T_K"]), z["k_shells"],
                                   z["M_k_icpt"])))
        order = sorted(range(len(rows)), key=lambda i: (rows[i][1], rows[i][2]))
        rows = [rows[i] for i in order]
        assert [r[0] for r in rows] == [str(t) for t in reference["tag"]]
        shape = kernel_shape(
            [r[3] for r in rows], t_min=float(reference["kappa_T_min"]),
            shape_set="single_phase_above_temperature",
            single_phase_flags=list(reference["single_phase"]))
        assert shape.n_runs == int(reference["kappa_n_runs"])
        assert np.array_equal(shape.wavenumbers, reference["kappa_k"])
        assert np.array_equal(shape.kernel, reference["kappa"])
        assert np.array_equal(shape.spread, reference["kappa_sd"])
        fits = [shape_amplitude(r[3], shape, k_max=1.5) for r in rows]
        assert np.array_equal(np.stack([f[0] for f in fits]),
                              reference["M_kappa0"])
        assert np.array_equal(np.stack([f[1] for f in fits]),
                              reference["M_kappa0_resid"])
        checked += 1
    assert checked > 0


def test_the_named_choices_hold_exactly_the_archived_alternatives():
    assert SCALINGS == ("window", "zero_wavenumber")
    assert SHAPE_SETS == ("above_temperature",
                          "single_phase_above_temperature")
