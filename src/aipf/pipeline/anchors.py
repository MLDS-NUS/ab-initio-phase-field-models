"""Measured anchor tables: the mobility a model is fit against.

    M_ab(k; dt) = Re< d rho_a(k) d rho_b(k)* > / (2 kT k^2 V dt),   d rho = rho(t + dt) - rho(t)

The transport coefficient is the ``dt -> 0`` intercept of the diffusive branch, then either a straight
line in ``k^2`` to ``k = 0`` or an amplitude against a universal kernel shape. Both are stored.
The pressure is declared, never parsed from a path.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .extract_modes import reference_box, wavevectors

#: Stored matrix a density-scaled column is built from: ``"window"`` (:attr:`RunMobility.M_window`)
#: or ``"zero_wavenumber"`` (:attr:`RunMobility.M_zero_wavenumber`).
SCALINGS = ("window", "zero_wavenumber")

#: Runs that define the universal kernel shape: ``"above_temperature"``
#: or ``"single_phase_above_temperature"`` (also flagged single-phase).
SHAPE_SETS = ("above_temperature", "single_phase_above_temperature")


def shell_index(wavenumbers) -> tuple[np.ndarray, np.ndarray]:
    """Bin modes into shells of width ``k_min`` (the smallest wavenumber) centred on its multiples.

    Returns ``(which, centres)``: shell per mode, each shell's MEAN wavenumber (``nan`` if empty).
    """
    k = np.asarray(wavenumbers, dtype=np.float64)
    if k.ndim != 1:
        raise ValueError(
            f"wavenumbers has shape {tuple(k.shape)}, which is not one "
            f"wavenumber per mode")
    if k.size == 0:
        raise ValueError(
            "wavenumbers is empty: there is no fundamental to set the shell "
            "width, and an empty binning is a silent zero-shell spectrum")
    if not np.all(k > 0.0):
        raise ValueError(
            f"wavenumbers has a non-positive entry ({k.min():.6g}): the "
            f"uniform mode has to be removed before binning, because its "
            f"mobility is divided by its own vanishing wavenumber")
    k_min = k.min()
    edges = np.arange(0.5 * k_min, k.max() + k_min, k_min)
    which = np.digitize(k, edges) - 1
    n_shells = int(which.max()) + 1
    centres = np.array([k[which == s].mean() if (which == s).any() else np.nan
                        for s in range(n_shells)])
    return which, centres


def mobility_spectrum(amplitudes, wavenumbers, which, n_shells, *,
                      thermal_energy: float, volume: float,
                      lags: Sequence[int], frame_interval: float
                      ) -> np.ndarray:
    """The shell-resolved mobility ``(n_shells, n_lags, n_channels, n_channels)`` at every lag.

    ``amplitudes`` ``(n_frames, n_channels, n_modes)`` without the uniform mode, ``thermal_energy`` is kT
    in the caller's unit, lag ``l`` is ``l * frame_interval``. Every start frame is used.
    """
    rho = np.asarray(amplitudes)
    if rho.ndim != 3:
        raise ValueError(
            f"amplitudes has shape {tuple(rho.shape)}, which is not "
            f"(n_frames, n_channels, n_modes)")
    k = np.asarray(wavenumbers, dtype=np.float64)
    if k.ndim != 1 or k.shape[0] != rho.shape[2]:
        raise ValueError(
            f"wavenumbers has shape {tuple(k.shape)} against "
            f"{rho.shape[2]} modes")
    which = np.asarray(which)
    if which.shape != k.shape:
        raise ValueError(
            f"which has shape {tuple(which.shape)} against {tuple(k.shape)} "
            f"modes: the binning has to cover exactly the modes given")
    lags = [int(lag) for lag in lags]
    if not lags:
        raise ValueError(
            "lags is empty: the lag intercept is a fit over lags and there "
            "is nothing to fit")
    n_frames = rho.shape[0]
    for lag in lags:
        if lag < 1:
            raise ValueError(
                f"lag={lag} is not a frame offset of at least one")
        if lag >= n_frames:
            raise ValueError(
                f"lag={lag} needs more than the {n_frames} frames given")
    n_channels = rho.shape[1]
    n_shells = int(n_shells)
    # Channel last and contiguous, so the contraction's bits do not depend on the caller's layout.
    r = np.ascontiguousarray(np.moveaxis(rho, 1, 2), dtype=np.complex128)
    out = np.full((n_shells, len(lags), n_channels, n_channels), np.nan)
    masks = [which == s for s in range(n_shells)]
    for index, lag in enumerate(lags):
        d = r[lag:] - r[:-lag]
        cov = np.einsum("fma,fmb->mab", d, d.conj()).real / (n_frames - lag)
        per_mode = cov / (2 * thermal_energy * k[:, None, None] ** 2
                          * volume * lag * frame_interval)
        for s, mask in enumerate(masks):
            if mask.any():
                out[s, index] = per_mode[mask].mean(0)
    return out


def band_average(spectrum, centres, *, k_min: float, k_max: float
                 ) -> np.ndarray:
    """Average the spectrum over shells with ``k_min <= centre <= k_max``, ignoring empty shells.

    Returns ``(n_lags, n_channels, n_channels)``.
    """
    centres = np.asarray(centres, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if spectrum.ndim != 4:
        raise ValueError(
            f"spectrum has shape {tuple(spectrum.shape)}, which is not "
            f"(n_shells, n_lags, n_channels, n_channels)")
    if centres.shape != spectrum.shape[:1]:
        raise ValueError(
            f"centres has {centres.shape[0]} shells against "
            f"{spectrum.shape[0]} in the spectrum")
    band = ((centres >= k_min) & (centres <= k_max) & np.isfinite(centres))
    if not band.any():
        raise ValueError(
            f"no shell lies in [{k_min:.6g}, {k_max:.6g}]: the band average "
            f"would be entirely nan and every downstream fit silently so")
    return np.nanmean(spectrum[band], axis=0)


def lag_intercept(lags_ps, matrices, *, min_lag: float
                  ) -> tuple[np.ndarray, int]:
    """The zero-lag intercept of the diffusive branch: ``(M, start)``, ``start`` indexing the original lags.

    Lags below ``min_lag`` go first. The branch starts at the trace maximum, keeping at least three lags.
    """
    lags_ps = np.asarray(lags_ps, dtype=np.float64)
    matrices = np.asarray(matrices, dtype=np.float64)
    if lags_ps.ndim != 1:
        raise ValueError(
            f"lags_ps has shape {tuple(lags_ps.shape)}, which is not one "
            f"time per lag")
    if matrices.ndim != 3 or matrices.shape[0] != lags_ps.shape[0]:
        raise ValueError(
            f"matrices has shape {tuple(matrices.shape)} against "
            f"{lags_ps.shape[0]} lags")
    if matrices.shape[1] != matrices.shape[2]:
        raise ValueError(
            f"matrices is not square in its channels: "
            f"{tuple(matrices.shape)}")
    if lags_ps.size < 3:
        raise ValueError(
            f"{lags_ps.size} lags given: a straight line through fewer than "
            f"three points has no residual and its intercept is not a "
            f"measurement")
    used = lags_ps >= min_lag
    if used.sum() < 3:
        used = np.ones(len(lags_ps), bool)
    offset = int(np.argmax(used))
    lp, ma = lags_ps[used], matrices[used]
    trace = sum(ma[:, a, a] for a in range(ma.shape[1]))
    start = int(np.argmax(trace))
    start = max(0, min(start, len(lp) - 3))
    n_channels = matrices.shape[1]
    out = np.zeros((n_channels, n_channels))
    for a in range(n_channels):
        for b in range(n_channels):
            out[a, b] = np.polyfit(lp[start:], ma[start:, a, b], 1)[1]
    return out, offset + start


def shell_intercepts(lags_ps, spectrum, *, min_lag: float) -> np.ndarray:
    """Per-shell zero-lag intercepts ``(n_shells, n_channels, n_channels)`` over the whole lag window.

    ``nan`` for a shell not finite at every fitted lag.
    """
    lags_ps = np.asarray(lags_ps, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if spectrum.ndim != 4 or spectrum.shape[1] != lags_ps.shape[0]:
        raise ValueError(
            f"spectrum has shape {tuple(spectrum.shape)} against "
            f"{lags_ps.shape[0]} lags")
    window = lags_ps >= min_lag
    if window.sum() < 3:
        window = np.ones(len(lags_ps), bool)
    n_shells, _, n_channels, _ = spectrum.shape
    out = np.full((n_shells, n_channels, n_channels), np.nan)
    for s in range(n_shells):
        if np.isfinite(spectrum[s][window]).all():
            for a in range(n_channels):
                for b in range(n_channels):
                    out[s, a, b] = np.polyfit(
                        lags_ps[window], spectrum[s, window, a, b], 1)[1]
    return out


def extrapolate_to_zero(centres, intercepts, *, k_max: float) -> np.ndarray:
    """Extrapolate per-shell intercepts to ``k = 0`` by a straight line in ``k^2`` over shells ``<= k_max``.

    Shells are selected on the first channel pair alone.
    """
    centres = np.asarray(centres, dtype=np.float64)
    intercepts = np.asarray(intercepts, dtype=np.float64)
    if intercepts.ndim != 3 or intercepts.shape[0] != centres.shape[0]:
        raise ValueError(
            f"intercepts has shape {tuple(intercepts.shape)} against "
            f"{centres.shape[0]} shells")
    used = (np.isfinite(centres) & (centres <= k_max)
            & np.isfinite(intercepts[:, 0, 0]))
    if used.sum() < 2:
        raise ValueError(
            f"{int(used.sum())} shells at or below k_max={k_max:.6g} are "
            f"measured: a straight line to zero wavenumber needs two")
    n_channels = intercepts.shape[1]
    out = np.zeros((n_channels, n_channels))
    for a in range(n_channels):
        for b in range(n_channels):
            out[a, b] = np.polyfit(centres[used] ** 2,
                                   intercepts[used, a, b], 1)[1]
    return out


def concentration_weights(fractions) -> np.ndarray:
    """Concentration-mode weights ``z = (f_1, -f_0)`` for two channel fractions ``f``. Two channels only.
    """
    f = np.asarray(fractions, dtype=np.float64)
    if f.shape != (2,):
        raise ValueError(
            f"fractions has shape {tuple(f.shape)}: the concentration mode "
            f"is defined here for exactly two channels, and with more there "
            f"is no single one to return. Pass explicit weights instead")
    return np.array([f[1], -f[0]])


def concentration_series(amplitudes, *, weights) -> np.ndarray:
    """``(n_frames, n_modes)`` complex128 weighted channel sum, accumulated in channel order."""
    rho = np.asarray(amplitudes)
    if rho.ndim != 3:
        raise ValueError(
            f"amplitudes has shape {tuple(rho.shape)}, which is not "
            f"(n_frames, n_channels, n_modes)")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (rho.shape[1],):
        raise ValueError(
            f"weights has shape {tuple(w.shape)} against {rho.shape[1]} "
            f"channels")
    out = w[0] * np.asarray(rho[:, 0, :], np.complex128)
    for c in range(1, rho.shape[1]):
        out = out + w[c] * np.asarray(rho[:, c, :], np.complex128)
    return out


def quarter_growth(series) -> float:
    """``last/first - 1`` of the mean of the last and first quarter of a per-frame series."""
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(
            f"series has shape {tuple(s.shape)}, which is not one value per "
            f"frame")
    q = s.shape[0] // 4
    if q < 1:
        raise ValueError(
            f"{s.shape[0]} frames: a quarter of the run is empty and the "
            f"growth would be nan rather than a verdict")
    return float(s[-q:].mean() / s[:q].mean() - 1.0)


def time_fifths(values) -> np.ndarray:
    """``(5,)`` means over the index fifths ``[F*w//5, F*(w+1)//5)`` of the frame axis."""
    v = np.asarray(values)
    if v.ndim < 1:
        raise ValueError("values has no frame axis")
    n = v.shape[0]
    if n < 5:
        raise ValueError(
            f"{n} frames: at least one fifth would be empty and its mean a "
            f"nan that no comparison can read")
    return np.array([v[n * w // 5:n * (w + 1) // 5].mean() for w in range(5)])


def condensing(fifths, composition, *, ratio: float, ideal_factor: float
               ) -> bool:
    """Condensing evidence: last fifth ``> ratio * first`` AND ``> ideal_factor * x (1 - x)``."""
    f = np.asarray(fifths, dtype=np.float64)
    if f.shape != (5,):
        raise ValueError(
            f"fifths has shape {tuple(f.shape)}, which is not five values")
    x = float(composition)
    return bool(f[-1] > ratio * max(f[0], 1e-12)
                and f[-1] > ideal_factor * x * (1 - x))


def single_phase(growth, fifths, composition, temperature, *,
                 growth_tol: float, condensing_ratio: float,
                 condensing_ideal_factor: float, level_factor: float,
                 t_min: float | None) -> bool:
    """Whether a run may anchor: STATIONARY and AT AN ADMISSIBLE LEVEL.

    Stationary: ``growth <= growth_tol`` or :func:`condensing` false. Level: last fifth at most
    ``level_factor * x (1 - x)``, waived at ``temperature >= t_min``. ``fifths=None`` judges on growth alone.
    """
    x = float(composition)
    stationary = bool(growth <= growth_tol)
    if not stationary and fifths is not None:
        stationary = not condensing(fifths, x, ratio=condensing_ratio,
                                    ideal_factor=condensing_ideal_factor)
    if fifths is None:
        level_ok = True
    else:
        level_ok = bool(np.asarray(fifths, dtype=np.float64)[-1]
                        <= level_factor * x * (1 - x))
    waived = t_min is not None and float(temperature) >= float(t_min)
    return bool(stationary and (level_ok or waived))


@dataclass(frozen=True)
class RunMobility:
    """Everything one equilibrium run contributes to an anchor table (``C`` channels).

    ``spectrum`` ``(n_shells, n_lags, C, C)``, ``band`` ``(n_lags, C, C)``, ``M_window`` and
    ``M_zero_wavenumber`` ``(C, C)``, ``shell_intercepts`` ``(n_shells, C, C)``, ``trace_intercepts``
    ``(n_shells,)``. ``kept`` marks the stored modes used.
    """

    shell_wavenumbers: np.ndarray
    spectrum: np.ndarray
    band: np.ndarray
    lags_ps: np.ndarray
    M_window: np.ndarray
    lag_start: int
    shell_intercepts: np.ndarray
    trace_intercepts: np.ndarray
    M_zero_wavenumber: np.ndarray
    volume: float
    temperature: float
    kept: np.ndarray
    shell_of_mode: np.ndarray


def run_mobility(amplitudes, box_lengths, nvec, *, temperature: float,
                 thermal_energy: float, lags: Sequence[int],
                 frame_interval: float, band: tuple[float, float],
                 min_lag: float, extrapolation_k_max: float,
                 reference_rule) -> RunMobility:
    """Measure one equilibrium run, from its stored amplitudes to ``M(0)``. The uniform mode is dropped here.

    ``reference_rule`` must be the rule the mode labels were chosen under (:data:`REFERENCE_BOXES`), or
    the cell ``(Lx, Ly, Lz)`` the archive records; :func:`aipf.pipeline.modes.recorded_reference` reads
    either off an archive.
    """
    rho = np.asarray(amplitudes)
    if rho.ndim != 3:
        raise ValueError(
            f"amplitudes has shape {tuple(rho.shape)}, which is not "
            f"(n_frames, n_channels, n_modes)")
    lengths = reference_box(box_lengths, rule=reference_rule)
    volume = float(np.prod(lengths))
    k = np.linalg.norm(wavevectors(nvec, lengths), axis=1)
    if k.shape[0] != rho.shape[2]:
        raise ValueError(
            f"{k.shape[0]} labels against {rho.shape[2]} stored modes")
    kept = k > 1e-12
    k = k[kept]
    rho = rho[:, :, kept]
    which, centres = shell_index(k)
    spectrum = mobility_spectrum(rho, k, which, centres.shape[0],
                                 thermal_energy=thermal_energy,
                                 volume=volume, lags=lags,
                                 frame_interval=frame_interval)
    lags_ps = np.array(list(lags)) * frame_interval
    k_lo, k_hi = band
    banded = band_average(spectrum, centres, k_min=max(k_lo, k.min() * 0.99),
                          k_max=k_hi)
    window, start = lag_intercept(lags_ps, banded, min_lag=min_lag)
    intercepts = shell_intercepts(lags_ps, spectrum, min_lag=min_lag)
    n_channels = rho.shape[1]
    traces = np.full((spectrum.shape[0], len(lags_ps), 1, 1), np.nan)
    traces[:, :, 0, 0] = sum(spectrum[:, :, a, a] for a in range(n_channels))
    trace_intercepts = shell_intercepts(lags_ps, traces,
                                        min_lag=min_lag)[:, 0, 0]
    zero_k = extrapolate_to_zero(centres, intercepts,
                                 k_max=extrapolation_k_max)
    return RunMobility(shell_wavenumbers=centres, spectrum=spectrum,
                       band=banded, lags_ps=lags_ps, M_window=window,
                       lag_start=start, shell_intercepts=intercepts,
                       trace_intercepts=trace_intercepts,
                       M_zero_wavenumber=zero_k, volume=volume,
                       temperature=float(temperature), kept=kept,
                       shell_of_mode=which)


@dataclass(frozen=True)
class KernelShape:
    """The mobility-spectrum shape a campaign shares: ``kernel`` and ``spread`` ``(n_shells, C, C)``.

    ``wavenumbers`` are the first qualifying run's shells.
    """

    wavenumbers: np.ndarray
    kernel: np.ndarray
    spread: np.ndarray
    n_runs: int
    t_min: float


def kernel_shape(runs, *, t_min: float, shape_set: str,
                 single_phase_flags=None) -> KernelShape:
    """Measure the universal shape of ``M(k)`` over a campaign's runs at or above ``t_min``.

    Each run is divided by its own ``k = 0`` value. ``single_phase_flags`` is required by
    ``"single_phase_above_temperature"`` only.
    """
    if shape_set not in SHAPE_SETS:
        raise ValueError(f"shape_set={shape_set!r} is not one of {SHAPE_SETS}")
    if shape_set == "single_phase_above_temperature":
        if single_phase_flags is None:
            raise ValueError(
                "shape_set='single_phase_above_temperature' needs "
                "single_phase_flags: the verdicts are what it selects on")
        flags = [bool(f) for f in single_phase_flags]
        if len(flags) != len(runs):
            raise ValueError(
                f"{len(flags)} verdicts against {len(runs)} runs")
    else:
        if single_phase_flags is not None:
            raise ValueError(
                "shape_set='above_temperature' cannot honour "
                "single_phase_flags: it selects on temperature alone, and "
                "accepting a set of verdicts it then ignores would let a "
                "caller believe in a restriction that had no effect")
        flags = [True] * len(runs)
    reference = None
    shapes = []
    for run, keep in zip(runs, flags):
        if run.temperature < t_min or not keep:
            continue
        measured = (np.isfinite(run.shell_wavenumbers)
                    & np.isfinite(run.shell_intercepts[:, 0, 0]))
        k = run.shell_wavenumbers[measured]
        icpt = run.shell_intercepts[measured]
        if reference is None:
            reference = k
        n_channels = icpt.shape[1]
        shape = np.zeros((len(reference), n_channels, n_channels))
        for a in range(n_channels):
            for b in range(n_channels):
                y = icpt[:, a, b]
                y0 = np.polyfit(k[:2] ** 2, y[:2], 1)[1]
                shape[:, a, b] = np.interp(reference, k, y / y0)
        shapes.append(shape)
    if not shapes:
        raise ValueError(
            f"no run in the set is at or above t_min={t_min:.6g}: the shape "
            f"is an average over a campaign's hot runs and there are none, "
            f"so pass a t_min at or below this campaign's top temperature")
    return KernelShape(wavenumbers=reference, kernel=np.mean(shapes, 0),
                       spread=np.std(shapes, 0), n_runs=len(shapes),
                       t_min=float(t_min))


def shape_amplitude(run, shape, *, k_max: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares ``M0`` in ``M(k_s) = M0 * kernel(k_s)`` over shells ``<= k_max``: ``(M0, residual)``.

    ``residual`` is the rms fit residual over the rms data, per channel pair.
    """
    measured = (np.isfinite(run.shell_wavenumbers)
                & np.isfinite(run.shell_intercepts[:, 0, 0]))
    k = run.shell_wavenumbers[measured]
    icpt = run.shell_intercepts[measured]
    used = k <= k_max
    if not used.any():
        raise ValueError(
            f"no measured shell is at or below k_max={k_max:.6g}: the "
            f"amplitude would be a zero-divided-by-zero rather than a fit")
    n_channels = icpt.shape[1]
    out = np.zeros((n_channels, n_channels))
    residual = np.zeros((n_channels, n_channels))
    for a in range(n_channels):
        for b in range(n_channels):
            kernel = np.interp(k, shape.wavenumbers, shape.kernel[:, a, b])
            y = icpt[:, a, b]
            out[a, b] = (np.sum(kernel[used] * y[used])
                         / np.sum(kernel[used] ** 2))
            predicted = out[a, b] * kernel[used]
            scale = np.sqrt(np.mean(y[used] ** 2))
            residual[a, b] = (np.sqrt(np.mean((y[used] - predicted) ** 2))
                              / max(scale, 1e-30))
    return out, residual


@dataclass(frozen=True)
class AnchorTable:
    """A campaign's measured rows at one DECLARED pressure (``None`` when the system controls none).

    Per row: ``composition``, ``temperature``, ``densities`` ``(n_rows, C)``, ``M_window``,
    ``M_zero_wavenumber``, ``M_shape``, ``shape_residual`` ``(n_rows, C, C)``, ``growth``, ``single_phase``.
    """

    pressure: float | None
    composition: np.ndarray
    temperature: np.ndarray
    densities: np.ndarray
    M_window: np.ndarray
    M_zero_wavenumber: np.ndarray
    M_shape: np.ndarray
    shape_residual: np.ndarray
    growth: np.ndarray
    single_phase: np.ndarray
    shape: KernelShape

    def matrices(self, source: str) -> np.ndarray:
        """The stored matrices named by ``source``, one of :data:`SCALINGS`."""
        if source not in SCALINGS:
            raise ValueError(f"source={source!r} is not one of {SCALINGS}")
        return (self.M_window if source == "window"
                else self.M_zero_wavenumber)

    def density_scaled(self, source: str) -> tuple[np.ndarray, np.ndarray]:
        """Diagonal ``M_aa / rho_a`` and upper-triangle cross ``M_ab / sqrt(rho_a rho_b)`` mobilities.
        """
        m = self.matrices(source)
        n_channels = self.densities.shape[1]
        diagonal = np.stack([m[:, a, a] / self.densities[:, a]
                             for a in range(n_channels)], axis=1)
        pairs = [(a, b) for a in range(n_channels)
                 for b in range(a + 1, n_channels)]
        cross = np.stack(
            [m[:, a, b] / np.sqrt(self.densities[:, a] * self.densities[:, b])
             for a, b in pairs],
            axis=1) if pairs else np.zeros((m.shape[0], 0))
        return diagonal, cross

    def einstein_diffusivity(self, source: str, thermal_energy
                             ) -> np.ndarray:
        """Self-diffusivities ``D_a = M_aa kT / rho_a``, a cross-check against displacements."""
        diagonal, _ = self.density_scaled(source)
        return diagonal * np.asarray(thermal_energy, dtype=np.float64)[
            ..., None]


def anchor_table(runs, *, pressure: float | None, composition, densities,
                 growth, single_phase, shape: KernelShape,
                 shape_k_max: float) -> AnchorTable:
    """Assemble a campaign's runs into one table, in the caller's row order.

    ``growth`` and ``single_phase`` are passed in, and ``shape`` may come from another table.
    """
    runs = list(runs)
    if not runs:
        raise ValueError("no runs: an empty campaign has no table")
    composition = np.asarray(composition, dtype=np.float64)
    densities = np.asarray(densities, dtype=np.float64)
    flags = np.asarray(single_phase, dtype=bool)
    growth = np.asarray(growth, dtype=np.float64)
    n_rows = len(runs)
    for name, value, shape_wanted in (
            ("composition", composition, (n_rows,)),
            ("growth", growth, (n_rows,)),
            ("single_phase", flags, (n_rows,))):
        if value.shape != shape_wanted:
            raise ValueError(
                f"{name} has shape {tuple(value.shape)} against {n_rows} "
                f"runs")
    n_channels = runs[0].M_window.shape[0]
    if densities.shape != (n_rows, n_channels):
        raise ValueError(
            f"densities has shape {tuple(densities.shape)} against "
            f"{n_rows} runs of {n_channels} channels")
    fits = [shape_amplitude(run, shape, k_max=shape_k_max) for run in runs]
    return AnchorTable(
        pressure=None if pressure is None else float(pressure),
        composition=composition,
        temperature=np.array([run.temperature for run in runs]),
        densities=densities,
        M_window=np.stack([run.M_window for run in runs]),
        M_zero_wavenumber=np.stack([run.M_zero_wavenumber for run in runs]),
        M_shape=np.stack([f[0] for f in fits]),
        shape_residual=np.stack([f[1] for f in fits]),
        growth=growth,
        single_phase=flags,
        shape=shape)


def anchor_rows(table: AnchorTable, system) -> np.ndarray:
    """Rows that may anchor: single-phase and ``T > system.t_min(table.pressure)``, strictly.

    Raises on an undeclared pressure, and on one a non-empty cut table does not cover.
    """
    rules = system.anchor_rules
    keep = np.asarray(table.single_phase, dtype=bool).copy()
    if not rules.T_min_by_pressure:
        return keep
    if table.pressure is None:
        raise ValueError(
            f"system {system.name!r} declares a temperature floor per "
            f"pressure but this table declares no pressure, so there is no "
            f"floor to apply. A pressure is a property of the state points "
            f"measured, not of the directory they were written to")
    t_min = system.t_min(table.pressure)
    if t_min is None:
        raise ValueError(
            f"system {system.name!r} declares floors at pressures "
            f"{sorted(float(p) for p in rules.T_min_by_pressure)} and this "
            f"table is at {table.pressure:g}, which is not among them")
    return keep & (np.asarray(table.temperature, dtype=np.float64) > t_min)
