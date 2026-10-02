"""Whether the data the pipeline produced can be used, and from which frame.

Each check reports a number, proposes a first unusable frame, or raises a flag from :data:`FLAGS`.
:func:`usable_window` joins the burn-in and the detectors' stops, :func:`verdict` applies the
caller's fatal flags. No threshold is chosen here.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

#: The three things a run can be. ``"flagged"`` is a recorded fact, not a lesser ``"unusable"``.
VERDICTS = ("clean", "flagged", "unusable")

#: Every flag this module's checks can raise, a closed list. A wider vocabulary goes to :func:`verdict`.
FLAGS = (
    # the timeline the pipeline read
    "no_frames",
    "short_of_expected",
    "uneven_cadence",
    "cadence_mismatch",
    "out_of_order",
    "repeated_step",
    # the atoms in a frame
    "empty_channel",
    "minority_channel",
    # a recorded series
    "not_finite",
    "off_target",
    "drifting",
    "unsettled",
    # the window
    "burn_in_over_run",
    "masked_tail",
    "short_window",
)

#: How a timeline is cut into blocks: ``"fixed"`` (equal blocks, the remainder at the end dropped)
#: or ``"spanning"`` (split points ``len * w // n_blocks``). They agree when the count divides.
PARTITIONS = ("fixed", "spanning")


def verdict(flags: Sequence[str], *, fatal: Sequence[str],
            vocabulary: Sequence[str] = FLAGS) -> str:
    """One of :data:`VERDICTS` from a run's flags: ``"unusable"`` if any is in ``fatal``.

    Every name in ``flags`` and ``fatal`` must be in ``vocabulary`` (default :data:`FLAGS`).
    """
    known = tuple(str(name) for name in vocabulary)
    raised = tuple(str(name) for name in flags)
    deadly = tuple(str(name) for name in fatal)
    unknown = sorted({name for name in raised if name not in known})
    if unknown:
        raise ValueError(
            f"flags names {unknown}, which are not in the vocabulary "
            f"{known}. Pass a wider vocabulary to declare a check of your own"
        )
    unknown = sorted({name for name in deadly if name not in known})
    if unknown:
        raise ValueError(
            f"fatal names {unknown}, which are not in the vocabulary "
            f"{known}, so the rule could never fire and the run would pass "
            f"whatever it did"
        )
    if any(name in deadly for name in raised):
        return "unusable"
    return "flagged" if raised else "clean"


@dataclass(frozen=True)
class Inventory:
    """What a timeline of integrator steps holds: its ends, the one ``step_interval``, completeness, flags."""

    n_frames: int
    first_step: int | None
    last_step: int | None
    step_interval: int | None
    n_expected: int | None
    completeness: float | None
    flags: tuple[str, ...]


def expected_frames(production_steps: int, *, dump_every_steps: int) -> int:
    """``production_steps // dump_every_steps + 1``: the frames a run writes, the first step included."""
    production_steps = int(production_steps)
    dump_every_steps = int(dump_every_steps)
    if production_steps < 0:
        raise ValueError(
            f"production_steps={production_steps} is negative: it is a count "
            f"of integrator steps"
        )
    if dump_every_steps < 1:
        raise ValueError(
            f"dump_every_steps={dump_every_steps} is below one: a cadence of "
            f"zero steps writes no frames and divides by nothing"
        )
    return production_steps // dump_every_steps + 1


def frame_inventory(timesteps, *, expected: int | None = None,
                    tolerated_fraction: float | None = None,
                    dump_every_steps: int | None = None) -> Inventory:
    """What a dump's own step numbers say: ``"no_frames"``, ``"short_of_expected"``, the cadence flags.

    ``expected`` and ``tolerated_fraction`` go together.
    """
    if (expected is None) != (tolerated_fraction is None):
        raise ValueError(
            "expected and tolerated_fraction go together: a frame count with "
            "no tolerance states no rule, and a tolerated_fraction with no "
            "expected count has nothing to be a fraction of"
        )
    steps = np.asarray(timesteps)
    if steps.ndim != 1:
        raise ValueError(
            f"timesteps has shape {tuple(steps.shape)}: there is one "
            f"integrator step per frame"
        )
    if steps.size and steps.dtype.kind not in "iu":
        raise ValueError(
            f"timesteps has dtype {steps.dtype}, which is not an integer "
            f"type: an integrator step is a count of steps"
        )
    if expected is not None:
        expected = int(expected)
        tolerated_fraction = float(tolerated_fraction)
        if expected < 1:
            raise ValueError(
                f"expected={expected} is below one frame: a run that writes "
                f"nothing cannot be the yardstick for one that does"
            )
        if not 0.0 <= tolerated_fraction <= 1.0:
            raise ValueError(
                f"tolerated_fraction={tolerated_fraction!r} is outside "
                f"[0, 1]: it is the share of the expected frames that still "
                f"counts as a complete run"
            )
    if dump_every_steps is not None:
        dump_every_steps = int(dump_every_steps)
        if dump_every_steps < 1:
            raise ValueError(
                f"dump_every_steps={dump_every_steps} is below one: no "
                f"timeline can have that interval"
            )

    n_frames = int(steps.size)
    flags: list[str] = []
    if n_frames == 0:
        flags.append("no_frames")
    first = int(steps[0]) if n_frames else None
    last = int(steps[-1]) if n_frames else None

    interval: int | None = None
    if n_frames > 1:
        gaps = np.diff(steps.astype(np.int64))
        if int(np.unique(steps).size) != n_frames:
            flags.append("repeated_step")
        if bool((gaps < 0).any()):
            flags.append("out_of_order")
        unique_gaps = np.unique(gaps)
        if unique_gaps.size == 1:
            interval = int(unique_gaps[0])
        else:
            flags.append("uneven_cadence")
    if (dump_every_steps is not None and interval is not None
            and interval != dump_every_steps):
        flags.append("cadence_mismatch")

    completeness = None
    if expected is not None:
        completeness = n_frames / expected
        if n_frames < tolerated_fraction * expected:
            flags.append("short_of_expected")
    return Inventory(n_frames=n_frames, first_step=first, last_step=last,
                     step_interval=interval, n_expected=expected,
                     completeness=completeness, flags=tuple(flags))


@dataclass(frozen=True)
class Occupancy:
    """How a frame's atoms fall into the declared channels: counts, shares, unassigned, empty, minority."""

    counts: tuple[int, ...]
    shares: tuple[float, ...]
    n_unassigned: int
    empty: tuple[int, ...]
    minority: tuple[int, ...]
    flags: tuple[str, ...]


def channel_occupancy(types, *, atom_types: Sequence[int],
                      minority_fraction: float | None = None,
                      minority_atoms: int | None = None) -> Occupancy:
    """Whether a frame carries every channel it declares, and which channels are a minority.

    Minority: share ``< 0.5`` AND (share ``<= minority_fraction`` OR count ``< minority_atoms``).
    The two thresholds are given both or neither.
    """
    if (minority_fraction is None) != (minority_atoms is None):
        raise ValueError(
            "minority_fraction and minority_atoms go together: the share "
            "test alone misses a small channel in a large box, and the count "
            "test alone misses a small channel in a small one"
        )
    channels = tuple(int(t) for t in atom_types)
    if not channels:
        raise ValueError(
            "atom_types is empty: with no channel declared there is nothing "
            "to count atoms into"
        )
    if len(set(channels)) != len(channels):
        repeated = sorted({t for t in channels if channels.count(t) > 1})
        raise ValueError(
            f"atom_types={channels} repeats {repeated}: the same atoms would "
            f"be counted into more than one channel"
        )
    values = np.asarray(types)
    if values.ndim != 1:
        raise ValueError(
            f"types has shape {tuple(values.shape)}: there is one dump type "
            f"per atom"
        )
    if minority_fraction is not None:
        minority_fraction = float(minority_fraction)
        minority_atoms = int(minority_atoms)
        if not 0.0 <= minority_fraction <= 1.0:
            raise ValueError(
                f"minority_fraction={minority_fraction!r} is outside [0, 1]: "
                f"it is a share of the declared atoms"
            )
        if minority_atoms < 0:
            raise ValueError(
                f"minority_atoms={minority_atoms} is negative: it is a count "
                f"of atoms"
            )

    counts = tuple(int((values == dump_type).sum()) for dump_type in channels)
    declared = sum(counts)
    n_unassigned = int(values.size) - declared
    shares = tuple(count / declared if declared else 0.0 for count in counts)

    empty = tuple(c for c, count in enumerate(counts) if count == 0)
    minority: tuple[int, ...] = ()
    if minority_fraction is not None:
        minority = tuple(
            c for c, (count, share) in enumerate(zip(counts, shares))
            if share < 0.5 and (share <= minority_fraction
                                or count < minority_atoms)
        )
    flags: list[str] = []
    if empty:
        flags.append("empty_channel")
    if minority:
        flags.append("minority_channel")
    return Occupancy(counts=counts, shares=shares, n_unassigned=n_unassigned,
                     empty=empty, minority=minority, flags=tuple(flags))


@dataclass(frozen=True)
class Observable:
    """One recorded scalar: counts, finite ``mean`` and ``std`` (ddof 1), half-to-half ``drift``, flags."""

    n: int
    n_not_finite: int
    mean: float
    std: float
    drift: float | None
    flags: tuple[str, ...]


def observable_report(values, *, target: float | None = None,
                      tolerance: float | None = None,
                      drift_max: float | None = None) -> Observable:
    """Three checks on one recorded series: finite, on ``target`` within ``tolerance``, not drifting.

    ``drift = |mean(second half) - mean(first half)| / |mean|``, flagged above ``drift_max``.
    """
    if (target is None) != (tolerance is None):
        raise ValueError(
            "target and tolerance go together: a target with no tolerance is "
            "not a test, and a tolerance with no target has nothing to be "
            "measured from"
        )
    series = np.asarray(values)
    if series.ndim != 1 or series.size < 1:
        raise ValueError(
            f"values has shape {tuple(series.shape)}: it is one sample per "
            f"recorded point and there has to be at least one"
        )
    if series.dtype.kind not in "iufb":
        raise ValueError(
            f"values has dtype {series.dtype}, which is not real: a target "
            f"and a drift are statements about an ordered quantity"
        )
    series = series.astype(np.float64)
    if tolerance is not None:
        target = float(target)
        tolerance = float(tolerance)
        if tolerance < 0.0:
            raise ValueError(
                f"tolerance={tolerance!r} is negative: it is a distance from "
                f"the target"
            )
    if drift_max is not None:
        drift_max = float(drift_max)
        if drift_max < 0.0:
            raise ValueError(
                f"drift_max={drift_max!r} is negative: it is a fractional "
                f"difference between two means"
            )
        if series.size < 2:
            raise ValueError(
                f"drift_max was asked for over {series.size} sample: a drift "
                f"between halves needs a sample in each of the halves"
            )

    finite = np.isfinite(series)
    n_not_finite = int((~finite).sum())
    good = series[finite]
    flags: list[str] = []
    if n_not_finite:
        flags.append("not_finite")
    mean = float(good.mean()) if good.size else float("nan")
    std = float(good.std(ddof=1)) if good.size > 1 else float("nan")

    # A mean that is not a number compares false against everything, so a
    # series with nothing finite in it is never reported as off target.
    if tolerance is not None and abs(mean - target) > tolerance:
        flags.append("off_target")

    drift = None
    if drift_max is not None:
        half = series.size // 2
        first = series[:half][finite[:half]]
        second = series[half:][finite[half:]]
        if first.size and second.size and mean != 0.0:
            drift = float(abs(second.mean() - first.mean()) / abs(mean))
            if drift > drift_max:
                flags.append("drifting")
        else:
            drift = float("nan")
    return Observable(n=int(series.size), n_not_finite=n_not_finite,
                      mean=mean, std=std, drift=drift, flags=tuple(flags))


@dataclass(frozen=True)
class Settle:
    """When a series settled: ``settle`` (persistent blocks) and ``strict`` (any block), or ``None``.

    ``reference`` and ``scatter`` are the settled tail's mean and block-to-block std.
    """

    settle: float | None
    strict: float | None
    n_blocks: int
    reference: float
    scatter: float
    flags: tuple[str, ...]


def block_means(times, values, *, block_span: float):
    """Non-overlapping time-block averages. Returns ``(ends, means, per_block)``.

    A final block under half the usual sample count is dropped.
    """
    block_span = float(block_span)
    if not block_span > 0.0:
        raise ValueError(
            f"block_span={block_span!r} is not positive: it is the duration "
            f"of one block"
        )
    stamps = np.asarray(times, dtype=np.float64)
    series = np.asarray(values, dtype=np.float64)
    if stamps.ndim != 1 or stamps.size < 1:
        raise ValueError(
            f"times has shape {tuple(stamps.shape)}: it is one time per "
            f"sample and there has to be at least one"
        )
    if series.shape != stamps.shape:
        raise ValueError(
            f"values has shape {tuple(series.shape)} against "
            f"{tuple(stamps.shape)} times: one sample per time"
        )
    if bool((np.diff(stamps) < 0.0).any()):
        raise ValueError(
            "times decreases: blocks are laid out forward from the first "
            "sample, so a series that goes back in time has no block layout"
        )
    index = np.floor((stamps - stamps[0]) / block_span).astype(np.int64)
    n_blocks = int(index.max()) + 1
    counts = np.bincount(index, minlength=n_blocks)
    totals = np.bincount(index, series, minlength=n_blocks)
    kept = counts > 0
    per_block = float(np.median(counts[kept]))
    kept &= counts >= max(1.0, 0.5 * per_block)
    ends = stamps[0] + (np.arange(n_blocks) + 1) * block_span
    return ends[kept], totals[kept] / counts[kept], int(per_block)


def settle_time(times, values, *, block_span: float, tol_sigma: float,
                persist: int, min_blocks: int) -> Settle:
    """How long a series took to settle onto the level it ends at.

    Band: tail mean ``+- tol_sigma`` tail scatter. A block outside counts in a run of ``persist``, or first.
    Fewer than ``min_blocks`` blocks gives ``None`` and ``"unsettled"``.
    """
    tol_sigma = float(tol_sigma)
    persist = int(persist)
    min_blocks = int(min_blocks)
    if not tol_sigma > 0.0:
        raise ValueError(
            f"tol_sigma={tol_sigma!r} is not positive: a band of no width "
            f"puts every block outside it"
        )
    if persist < 1:
        raise ValueError(
            f"persist={persist} is below one: a stretch of no blocks is not "
            f"a stretch"
        )
    if min_blocks < 2:
        raise ValueError(
            f"min_blocks={min_blocks} is below two: the band comes from the "
            f"last half of the blocks, so there have to be blocks on both "
            f"sides of the half"
        )
    series = np.asarray(values, dtype=np.float64)
    if not np.isfinite(series).all():
        raise ValueError(
            "values holds an entry that is not finite: a band built from a "
            "mean and a scatter over such a series is not a band"
        )

    ends, means, _ = block_means(times, series, block_span=block_span)
    n_blocks = int(means.size)
    if n_blocks < min_blocks:
        return Settle(settle=None, strict=None, n_blocks=n_blocks,
                      reference=float("nan"), scatter=float("nan"),
                      flags=("unsettled",))

    tail = means[n_blocks // 2:]
    reference = float(tail.mean())
    scatter = float(tail.std(ddof=1)) if tail.size > 1 else 0.0
    if scatter == 0.0 or not math.isfinite(scatter):
        # A constant tail: the smallest band still relative to the quantity's own size.
        scatter = float(np.abs(tail).mean()) * 1e-12 + 1e-30

    start = float(np.asarray(times, dtype=np.float64)[0])
    outside = np.abs(means - reference) > tol_sigma * scatter
    strict = (float(ends[np.where(outside)[0][-1]] - start)
              if outside.any() else 0.0)

    kept = np.zeros(n_blocks, dtype=bool)
    i = 0
    while i < n_blocks:
        if not outside[i]:
            i += 1
            continue
        j = i
        while j < n_blocks and outside[j]:
            j += 1
        if (j - i) >= persist or i == 0:
            kept[i:j] = True
        i = j
    settle = float(ends[np.where(kept)[0][-1]] - start) if kept.any() else 0.0
    return Settle(settle=settle, strict=strict, n_blocks=n_blocks,
                  reference=reference, scatter=scatter, flags=())


def burn_in_frames(settle: float | None, *, floor: float,
                   sample_interval: float) -> int:
    """Frames to drop: ``max(floor, ceil(settle)) / sample_interval``, the floor alone for ``settle=None``.
    """
    floor = float(floor)
    sample_interval = float(sample_interval)
    if floor < 0.0:
        raise ValueError(f"floor={floor!r} is negative: it is a duration")
    if not sample_interval > 0.0:
        raise ValueError(
            f"sample_interval={sample_interval!r} is not positive: it is the "
            f"time between stored frames"
        )
    if settle is None:
        span = floor
    else:
        settle = float(settle)
        if settle < 0.0:
            raise ValueError(
                f"settle={settle!r} is negative: it is a duration from the "
                f"start of the run"
            )
        span = max(floor, math.ceil(settle))
    return int(round(span / sample_interval))


@dataclass(frozen=True)
class Window:
    """The half-open usable range ``[start, stop)`` of ``n_frames``, its ``n_kept`` frames, and flags."""

    start: int
    stop: int
    n_frames: int
    n_kept: int
    flags: tuple[str, ...]


def usable_window(n_frames: int, *, burn_in: int, stops: Iterable[int | None],
                  minimum: int) -> Window:
    """The burn-in and the EARLIEST proposed stop, made one range. A burn-in past the stop keeps nothing.

    ``stops`` are first unusable frames, ``None`` where a detector found nothing. ``minimum`` is inclusive.
    """
    n_frames = int(n_frames)
    burn_in = int(burn_in)
    minimum = int(minimum)
    if n_frames < 0:
        raise ValueError(f"n_frames={n_frames} is negative: it is a count")
    if burn_in < 0:
        raise ValueError(
            f"burn_in={burn_in} is negative: it is a count of frames dropped "
            f"from the front"
        )
    if minimum < 0:
        raise ValueError(
            f"minimum={minimum} is negative: it is a count of frames"
        )
    proposed = [int(stop) for stop in stops if stop is not None]
    if any(stop < 0 for stop in proposed):
        raise ValueError(
            f"stops holds {sorted(s for s in proposed if s < 0)}: a stop is "
            f"an index into the timeline"
        )

    stop = min([n_frames, *proposed]) if proposed else n_frames
    flags: list[str] = []
    if stop < n_frames:
        flags.append("masked_tail")
    n_kept = max(stop - burn_in, 0)
    if burn_in > stop:
        flags.append("burn_in_over_run")
    if n_kept < minimum:
        flags.append("short_window")
    return Window(start=burn_in, stop=stop, n_frames=n_frames, n_kept=n_kept,
                  flags=tuple(flags))


def displacement_series(frames: Iterable) -> np.ndarray:
    """``(n_frames - 1,)`` mean nearest-image displacement per atom between consecutive frames.

    The later frame's cell is used.
    """
    steps: list[float] = []
    previous = None
    count = 0
    for frame in frames:
        count += 1
        positions = np.asarray(frame.positions, dtype=np.float64)
        bounds = np.asarray(frame.box_bounds, dtype=np.float64)
        lengths = bounds[:, 1] - bounds[:, 0]
        if previous is not None:
            if positions.shape != previous.shape:
                raise ValueError(
                    f"a frame holds {positions.shape[0]} atoms against "
                    f"{previous.shape[0]} in the one before it: a "
                    f"displacement is per atom, so the atoms have to be the "
                    f"same ones in the same order"
                )
            delta = positions - previous
            delta -= np.round(delta / lengths) * lengths
            steps.append(float(np.linalg.norm(delta, axis=1).mean()))
        previous = positions
    if count == 0:
        raise ValueError(
            "frames is empty: there is no timeline to measure motion along"
        )
    return np.asarray(steps, dtype=np.float64)


def frozen_stop(displacements, *, window: int,
                threshold: float) -> int | None:
    """First unusable frame: where the rolling median of ``window`` steps first falls below ``threshold``.

    ``window`` odd and always full, ``threshold`` strict. ``None`` if the run never stops.
    """
    window = int(window)
    threshold = float(threshold)
    if window < 1 or window % 2 == 0:
        raise ValueError(
            f"window={window} is not a positive odd number of steps: an even "
            f"window has no middle sample and its median is a mean of two"
        )
    if not threshold > 0.0:
        raise ValueError(
            f"threshold={threshold!r} is not positive: motion is a distance "
            f"and at or below zero nothing is ever still"
        )
    steps = np.asarray(displacements, dtype=np.float64)
    if steps.ndim != 1:
        raise ValueError(
            f"displacements has shape {tuple(steps.shape)}: it is one number "
            f"per step between consecutive frames"
        )
    if steps.size < window:
        return None
    rolling = np.lib.stride_tricks.sliding_window_view(steps, window)
    still = np.where(np.median(rolling, axis=1) < threshold)[0]
    if not still.size:
        return None
    return int(still[0]) + window


def excursion_stop(values, *, head: int, tolerance_floor: float,
                   tolerance_sigma: float, margin: int,
                   reference: float | None = None) -> int | None:
    """First unusable frame of a per-frame quantity that left its own band, less ``margin``.

    Centre: median of the first ``head`` frames, or ``reference``. Half-width:
    ``max(tolerance_floor, tolerance_sigma * relative head scatter)``.
    """
    head = int(head)
    margin = int(margin)
    tolerance_floor = float(tolerance_floor)
    tolerance_sigma = float(tolerance_sigma)
    if head < 2:
        raise ValueError(
            f"head={head} is below two frames: a scatter needs two samples"
        )
    if margin < 0:
        raise ValueError(
            f"margin={margin} is negative: it is a count of frames cut "
            f"before the deviation"
        )
    if tolerance_floor < 0.0:
        raise ValueError(
            f"tolerance_floor={tolerance_floor!r} is negative: it is a "
            f"relative deviation"
        )
    if tolerance_sigma < 0.0:
        raise ValueError(
            f"tolerance_sigma={tolerance_sigma!r} is negative: it is a "
            f"multiple of the run's own scatter"
        )
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < 2:
        raise ValueError(
            f"values has shape {tuple(series.shape)}: it is one number per "
            f"frame and a band needs more than one"
        )
    lead = series[:min(head, series.size)]
    centre = float(np.median(lead)) if reference is None else float(reference)
    if not centre > 0.0:
        raise ValueError(
            f"reference={centre!r} is not positive: the deviation is taken "
            f"relative to it, and a level of zero has no relative deviation"
        )
    scatter = float(np.std(lead / centre - 1.0))
    tolerance = max(tolerance_floor, tolerance_sigma * scatter)
    outside = np.where(np.abs(series / centre - 1.0) > tolerance)[0]
    if not outside.size:
        return None
    return max(int(outside[0]) - margin, 0)


def uniformity_deviation(field, *, contrast: Sequence[float]) -> float:
    """``max |psi - <psi>| / <psi>`` for ``psi = sum_c contrast[c] field[c]``, accumulated in double.
    """
    values = np.asarray(field)
    if values.ndim < 2:
        raise ValueError(
            f"field has shape {tuple(values.shape)}: the channel axis is the "
            f"first one and there has to be something after it"
        )
    weights = tuple(float(w) for w in contrast)
    if len(weights) != values.shape[0]:
        raise ValueError(
            f"contrast has {len(weights)} weights for {values.shape[0]} "
            f"channels: the combination has to say something about each"
        )
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            "field holds an entry that is not finite: a deviation from a "
            "mean that is not a number is not a measurement"
        )
    psi = np.zeros(values.shape[1:], dtype=np.float64)
    for weight, channel in zip(weights, values):
        psi = psi + weight * channel
    mean = float(psi.mean())
    if not mean > 0.0:
        raise ValueError(
            f"the declared combination has mean {mean!r}: a RELATIVE "
            f"deviation needs a scale to be relative to, and a combination "
            f"that averages to zero or below gives none"
        )
    return float(np.abs(psi - mean).max() / mean)


def mode_power_ratio(rho_k, *, numerator: Sequence[float],
                     denominator: Sequence[float]) -> float:
    """Mode average of the time-averaged power of ``numerator`` over that of ``denominator``, per mode.

    A mode with no denominator power raises.
    """
    values = np.asarray(rho_k)
    if values.ndim != 3:
        raise ValueError(
            f"rho_k has shape {tuple(values.shape)}, not (n_frames, "
            f"n_channels, n_modes)"
        )
    if values.dtype.kind != "c":
        raise ValueError(
            f"rho_k has dtype {values.dtype}, which is not complex: an "
            f"amplitude has a phase and its power is not its square"
        )
    n_channels = values.shape[1]
    top = tuple(float(w) for w in numerator)
    bottom = tuple(float(w) for w in denominator)
    if len(top) != n_channels:
        raise ValueError(
            f"numerator has {len(top)} weights for {n_channels} channels"
        )
    if len(bottom) != n_channels:
        raise ValueError(
            f"denominator has {len(bottom)} weights for {n_channels} channels"
        )
    values = values.astype(np.complex128)
    if not np.isfinite(values).all():
        raise ValueError(
            "rho_k holds an amplitude that is not finite: a power taken from "
            "it is not a measurement"
        )

    def combine(weights):
        out = np.zeros(values.shape[::2], dtype=np.complex128)
        for weight, channel in zip(weights, np.moveaxis(values, 1, 0)):
            out = out + weight * channel
        return out

    lower = (np.abs(combine(bottom)) ** 2).mean(axis=0)
    empty = int((lower == 0.0).sum())
    if empty:
        raise ValueError(
            f"the denominator combination has no power at all in {empty} of "
            f"{lower.size} modes, so the ratio is not a number there. Select "
            f"the modes the combination lives on before asking for it"
        )
    upper = (np.abs(combine(top)) ** 2).mean(axis=0)
    return float((upper / lower).mean())


@dataclass(frozen=True)
class Growth:
    """A band's growth: block means, ``gain`` last over first, ``sigma_floor``,
    ``z_floor = (gain - 1) / sigma_floor``."""

    block_means: tuple[float, ...]
    gain: float
    sigma_floor: float
    z_floor: float


def band_growth(series, *, n_blocks: int, partition: str,
                n_independent: float) -> Growth:
    """A band-averaged timeline's growth over ``n_blocks`` blocks cut by ``partition``.

    ``sigma_floor = sqrt(2 / n_independent)``, what a frozen band can fake. Stored conjugate pairs count once.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition={partition!r} is not one of {PARTITIONS}")
    n_blocks = int(n_blocks)
    n_independent = float(n_independent)
    if n_blocks < 2:
        raise ValueError(
            f"n_blocks={n_blocks} is below two: a growth is a comparison "
            f"between a first block and a last one"
        )
    if not n_independent > 0.0:
        raise ValueError(
            f"n_independent={n_independent!r} is not positive: it is a count "
            f"of modes that vary on their own"
        )
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"series has shape {tuple(values.shape)}: it is one band average "
            f"per frame"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "series holds an entry that is not finite: a block mean over it "
            "is not a measurement"
        )
    n_samples = int(values.size)
    if n_samples < n_blocks:
        raise ValueError(
            f"n_blocks={n_blocks} over {n_samples} samples: every block has "
            f"to hold at least one sample"
        )
    if partition == "fixed":
        per_block = n_samples // n_blocks
        means = [float(values[i * per_block:(i + 1) * per_block].mean())
                 for i in range(n_blocks)]
    else:
        edges = [n_samples * w // n_blocks for w in range(n_blocks + 1)]
        means = [float(values[lo:hi].mean())
                 for lo, hi in zip(edges[:-1], edges[1:])]
    if not means[0] > 0.0:
        raise ValueError(
            f"the first block's mean is {means[0]!r}: a gain is a ratio to "
            f"it, and it has to be a positive quantity for that to be a "
            f"growth rather than a sign change"
        )
    gain = means[-1] / means[0]
    sigma_floor = math.sqrt(2.0 / n_independent)
    return Growth(block_means=tuple(means), gain=float(gain),
                  sigma_floor=float(sigma_floor),
                  z_floor=float((gain - 1.0) / sigma_floor))
