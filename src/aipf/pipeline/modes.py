"""A trajectory on disk to a mode archive on disk, with provenance and a cache.

Every input that decides the answer, the source's sha256 included, is the provenance; a cache entry is
reused only when it matches. The farm keeps one file per run where the archive does, ``<tree>/<tag>/``.
:class:`ModesRecord` holds the archive's channel-last layout ``(n_frames, n_modes, n_channels)``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from aipf.system import System

from . import coarse_grain, extract_modes
from .kde import _resolve_device

#: Record version, compared on read so an older layout is a miss.
SCHEMA = 1

#: The archive's own key spellings in ``modes.npz``.
ARCHIVE_KEYS = {"amplitudes": "rho_k", "labels": "nvec", "box": "box",
                "temperature": "T_K", "frame_interval": "dt_frame_ps"}

#: Keys of a two-sided composition label: a direct-coexistence run has no single composition.
SIDE_KEYS = ("x_left", "x_right")

#: The box the mode labels are chosen on, fixed to match the archive.
REFERENCE_BOX = "time_mean"

#: Metadata step counts of the preparation and production stages. A run with neither is refused.
STEP_COUNT_KEYS = ("n_equil", "n_prod")

#: Device of the atom sum, fixed to match the archive. The resolved device is recorded.
DEVICE = "auto"

#: The system declaration naming how the stored channels are formed from the per-type sums.
MODE_FIELDS_KEY = "mode_fields"

#: One stored channel per species, the sum over that species' dump type.
PER_TYPE = "per_type"

#: The keys of one declared combination: its channel ``name``, ``weights`` ``{dump type: w}``, ``mean``.
FIELD_KEYS = ("name", "weights", "mean")


@dataclass(frozen=True)
class ModesRecord:
    """One run's mode timeline in the archive's layout, and where it came from.

    ``amplitudes`` ``(n_frames, n_modes, n_channels)`` complex64, ``labels`` ``(n_modes, 3)`` int,
    ``boxes`` ``(n_frames, 3)``, ``T``, ``dt_frame``, ``provenance`` (with ``cache_dir``).
    """

    amplitudes: np.ndarray
    labels: np.ndarray
    boxes: np.ndarray
    T: float
    dt_frame: float
    provenance: Mapping[str, Any]

    def save(self, directory: Path) -> Path:
        """Write ``modes.npz`` with the composition label, and ``provenance.json``. Returns the archive."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            ARCHIVE_KEYS["amplitudes"]: self.amplitudes,
            ARCHIVE_KEYS["labels"]: self.labels,
            ARCHIVE_KEYS["box"]: self.boxes,
            ARCHIVE_KEYS["temperature"]: self.T,
            ARCHIVE_KEYS["frame_interval"]: self.dt_frame,
        }
        label = self.provenance.get("composition") or {}
        # A label key that collided with an archive key would overwrite that array.
        clash = sorted(set(label) & set(payload))
        if clash:
            raise ValueError(
                f"composition label key(s) {clash} collide with this "
                f"archive's own key(s) {sorted(ARCHIVE_KEYS.values())}: "
                f"writing them would overwrite the array they name. The "
                f"label comes from the system's declared composition "
                f"column, which is where the name has to change")
        payload.update(label)
        np.savez_compressed(directory / "modes.npz", **payload)
        (directory / "provenance.json").write_text(
            json.dumps(dict(self.provenance), indent=1, sort_keys=True))
        return directory / "modes.npz"

    @classmethod
    def load(cls, directory: Path) -> "ModesRecord":
        """Read back what :meth:`save` wrote."""
        directory = Path(directory)
        with np.load(directory / "modes.npz") as payload:
            amplitudes = payload[ARCHIVE_KEYS["amplitudes"]]
            labels = payload[ARCHIVE_KEYS["labels"]]
            boxes = payload[ARCHIVE_KEYS["box"]]
            temperature = float(payload[ARCHIVE_KEYS["temperature"]])
            interval = float(payload[ARCHIVE_KEYS["frame_interval"]])
        provenance = json.loads((directory / "provenance.json").read_text())
        return cls(amplitudes, labels, boxes, temperature, interval,
                   provenance)


def composition_label(composition: Mapping[str, Any], *,
                      x_key: str) -> dict[str, float]:
    """A state point's composition as label scalars: ``{x_key: x}``, or ``SIDE_KEYS`` when two-sided.

    Empty when the record carries no value.
    """
    values = (composition or {}).get("x") or {}
    if not values:
        return {}
    value = next(iter(values.values()))
    if value is None:
        return {}
    if np.ndim(value) == 0:
        return {x_key: float(value)}
    sides = [float(v) for v in value]
    if len(sides) != len(SIDE_KEYS):
        raise ValueError(
            f"composition {composition!r} has {len(sides)} sides, and the "
            f"archive labels a two-sided run with exactly {len(SIDE_KEYS)} "
            f"keys {SIDE_KEYS}. A label this cannot spell would be written as "
            f"nothing and the run would train as one nobody labelled")
    return dict(zip(SIDE_KEYS, sides))


def frames_prepared(record: Mapping[str, Any], dump) -> int:
    """Leading frames of a state point that are the run being prepared, via ``frames_to_skip``.

    The counts are ``record["meta"]["extra"][STEP_COUNT_KEYS]``. A record with neither raises ``KeyError``.
    """
    extra = (record.get("meta") or {}).get("extra") or {}
    missing = [k for k in STEP_COUNT_KEYS if extra.get(k) is None]
    if missing:
        raise KeyError(
            f"state point {record.get('farm_dir', record.get('tag'))!r} "
            f"records no {missing}: there is no way to tell the frames that "
            f"are the run being prepared from the frames that are the state, "
            f"and answering zero would extract a melt into the training data "
            f"of a run nobody checked. State the count instead")
    equilibration, production = (int(extra[k]) for k in STEP_COUNT_KEYS)
    return extract_modes.frames_to_skip(
        coarse_grain.read_timesteps(dump),
        equilibration_steps=equilibration, production_steps=production)


def check_fields(fields, *, species: Sequence[str]):
    """A ``mode_fields`` declaration checked: :data:`PER_TYPE`, or one combination per species in channel order.

    A combination is ``{"name": species, "weights": {dump type: w}, "mean": m or None}``.
    """
    admissible = (f"{PER_TYPE!r}, or one mapping per species with keys "
                  f"{FIELD_KEYS}")
    if isinstance(fields, str):
        if fields != PER_TYPE:
            raise ValueError(
                f"mode_fields={fields!r} is not a declared form. It is "
                f"{admissible}")
        return PER_TYPE
    fields = tuple(fields)
    if len(fields) != len(species):
        raise ValueError(
            f"mode_fields declares {len(fields)} combination(s) for "
            f"{len(species)} species {tuple(species)}: one stored channel "
            f"per species. It is {admissible}")
    checked = []
    for spec, name in zip(fields, species):
        if not isinstance(spec, Mapping) or set(spec) != set(FIELD_KEYS):
            raise ValueError(
                f"mode_fields entry {spec!r} is not a mapping with exactly "
                f"the keys {FIELD_KEYS}")
        if spec["name"] != name:
            raise ValueError(
                f"mode_fields entry {spec['name']!r} stands where channel "
                f"{name!r} is: the combinations follow species order")
        weights = dict(spec["weights"])
        if not weights or any(isinstance(t, bool) or not isinstance(t, int)
                              for t in weights):
            raise ValueError(
                f"mode_fields[{name!r}]['weights']={weights!r} must map at "
                f"least one integer dump type to its weight")
        weights = {t: float(w) for t, w in weights.items()}
        mean = spec["mean"]
        mean = None if mean is None else float(mean)
        if not all(np.isfinite(v) for v in
                   list(weights.values()) + ([] if mean is None else [mean])):
            raise ValueError(
                f"mode_fields[{name!r}] carries a non-finite weight or mean")
        checked.append({"name": name, "weights": weights, "mean": mean})
    return tuple(checked)


def mode_fields(system: System):
    """``system.defaults["mode_fields"]`` through :func:`check_fields`; an undeclared one is refused by name."""
    if MODE_FIELDS_KEY not in system.defaults:
        raise KeyError(
            f"system {system.name!r} declares no defaults[{MODE_FIELDS_KEY!r}]. "
            f"It is {PER_TYPE!r} (one channel per species) or one "
            f"combination per species with keys {FIELD_KEYS}; neither is "
            f"assumed")
    return check_fields(system.defaults[MODE_FIELDS_KEY],
                        species=system.species)


def extracted_types(fields, species_types: Sequence[int]) -> tuple[int, ...]:
    """The dump types summed: ``species_types`` for :data:`PER_TYPE`, else every weighted type, ascending."""
    if fields == PER_TYPE:
        return tuple(int(t) for t in species_types)
    return tuple(sorted({t for spec in fields for t in spec["weights"]}))


def combine_fields(amplitudes: np.ndarray, labels: np.ndarray,
                   boxes: np.ndarray, fields,
                   atom_types: Sequence[int]) -> np.ndarray:
    """``(n_frames, n_modes, n_types)`` per-type sums to ``(n_frames, n_modes, n_fields)``, same dtype.

    Weights are applied in the stored precision, in declaration order. A declared ``mean`` replaces
    the zero mode by ``mean * V`` of the frame's own box.

    ``amplitudes``: per-type sums, ``(n_frames, n_modes, n_types)`` complex.
    ``labels``: the integer mode labels, ``(n_modes, 3)``.
    ``boxes``: per-frame box edges, ``(n_frames, 3)``, in the coordinates' length unit.
    ``fields``: the checked combinations (:func:`check_fields`).
    ``atom_types``: the dump type of each column of ``amplitudes``.
    """
    column = {int(t): i for i, t in enumerate(atom_types)}
    real = np.empty(0, amplitudes.dtype).real.dtype
    zero = np.flatnonzero((np.asarray(labels) == 0).all(axis=1))
    volumes = np.prod(np.asarray(boxes, dtype=np.float64), axis=-1)
    out = np.empty(amplitudes.shape[:2] + (len(fields),), amplitudes.dtype)
    for c, spec in enumerate(fields):
        missing = sorted(set(spec["weights"]) - set(column))
        if missing:
            raise ValueError(
                f"field {spec['name']!r} weighs dump type(s) {missing} that "
                f"were not extracted (extracted: {sorted(column)})")
        total = np.zeros(amplitudes.shape[:2], amplitudes.dtype)
        for dump_type, weight in spec["weights"].items():
            total = total + real.type(weight) * amplitudes[:, :, column[dump_type]]
        if spec["mean"] is not None:
            total[:, zero] = (spec["mean"] * volumes)[:, None]
        out[..., c] = total
    return out


def _fields_identity(fields):
    """A combination declaration as JSON-stable lists, so a cached record compares equal after a read."""
    return [{"name": spec["name"],
             "weights": [[int(t), float(w)] for t, w in spec["weights"].items()],
             "mean": spec["mean"]} for spec in fields]


def _sha256(path: Path) -> str:
    """The digest of a file's bytes, read in chunks so a dump is not loaded."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(identity: Mapping[str, Any]) -> str:
    """The cache entry's name: 16 hex digits of the sha256 of the sorted-key identity."""
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def _extract(dump: Path, *, k_cut: float, atom_types: Sequence[int],
             ordering: str, route: str, device: Optional[str],
             skip_frames: int):
    """Drop ``skip_frames``, choose the mode set on the time-mean box of the rest, extract."""
    frames = list(coarse_grain.read_dump(dump))[skip_frames:]
    if not frames:
        raise ValueError(
            f"{dump} holds no complete frame after dropping {skip_frames}: "
            f"there is no timeline to extract")
    boxes = coarse_grain.box_lengths(frames)
    labels = extract_modes.mode_set(
        extract_modes.reference_box(boxes, rule=REFERENCE_BOX),
        k_cut=k_cut, ordering=ordering)
    amplitudes, _ = extract_modes.mode_series(
        frames, labels, atom_types=atom_types, route=route, device=device,
        dtype=np.complex64)
    # Channel first inside the package, channel last on disk.
    return np.moveaxis(amplitudes, 1, 2), labels, boxes


def modes_from_dump(dump, *, sigma: float, k_cut: float,
                    atom_types: Sequence[int], fields, ordering: str,
                    route: str, T: float, dt_frame: float, skip_frames: int,
                    composition: Optional[Mapping[str, float]] = None,
                    cache_dir: Optional[Path] = None,
                    entry_dir: Optional[Path] = None) -> ModesRecord:
    """One trajectory's modes, provenance-stamped and cached.

    ``atom_types`` are the dump types summed; ``fields`` is :data:`PER_TYPE` (store them) or checked
    combinations (:func:`check_fields`). ``sigma`` is recorded, not used. ``skip_frames`` has no default.
    ``composition`` is a label, never physics. ``cache_dir``: one entry per provenance, under its key;
    ``entry_dir``: exactly this directory, replaced when the provenance differs (the archive's
    ``<tree>/<tag>/modes.npz``). With neither nothing is cached; both is refused.

    ``dump``: the trajectory file (a LAMMPS text dump).
    ``sigma``: the coarse-graining width, in the coordinates' length unit (recorded).
    ``k_cut``: the mode cutoff ``|k| <= k_cut``, in the reciprocal length unit.
    ``atom_types``: the dump types summed, one per stored channel before combination.
    ``fields``: :data:`PER_TYPE` or :func:`check_fields`'s combinations.
    ``ordering``: the mode-list order, one of :data:`aipf.pipeline.extract_modes.ORDERINGS`.
    ``route``: the atom-sum evaluation, one of :data:`aipf.pipeline.extract_modes.ROUTES`.
    ``T``: the run's temperature, in the system's temperature unit (recorded).
    ``dt_frame``: the time between stored frames, ps.
    ``skip_frames``: leading frames dropped as preparation.
    ``composition``: label scalars stored beside the modes, or ``None``.
    ``cache_dir``, ``entry_dir``: the two cache layouts, at most one.
    """
    if cache_dir is not None and entry_dir is not None:
        raise ValueError(
            "cache_dir and entry_dir are two layouts of one cache: a keyed "
            "directory per provenance, or one fixed directory per run. Pass one")
    if fields != PER_TYPE and not isinstance(fields, tuple):
        raise TypeError(
            f"fields={fields!r} is neither {PER_TYPE!r} nor the tuple "
            f"check_fields returns: pass the declaration through it")
    dump = Path(dump)
    skip_frames = int(skip_frames)
    if skip_frames < 0:
        raise ValueError(
            f"skip_frames={skip_frames} is negative: it counts the leading "
            f"frames that are the run being prepared, and a negative count "
            f"would silently extract the whole dump")
    identity = {
        "schema": SCHEMA,
        "source": str(dump),
        "source_sha256": _sha256(dump),
        "sigma": float(sigma),
        "k_cut": float(k_cut),
        "atom_types": [int(t) for t in atom_types],
        "ordering": ordering,
        "route": route,
        "reference_box": REFERENCE_BOX,
        "device": DEVICE,
        "device_resolved": str(_resolve_device(DEVICE)),
        "T": float(T),
        "dt_frame": float(dt_frame),
        "skip_frames": int(skip_frames),
        "composition": dict(composition or {}),
    }
    if fields != PER_TYPE:
        identity["fields"] = _fields_identity(fields)
    if entry_dir is not None:
        entry = Path(entry_dir)
    else:
        entry = None if cache_dir is None else Path(cache_dir) / _cache_key(identity)
    provenance = dict(identity,
                      cache_dir=None if entry is None else str(entry))
    if entry is not None and (entry / "modes.npz").is_file():
        cached = ModesRecord.load(entry)
        if {k: v for k, v in cached.provenance.items()
                if k != "cache_dir"} == identity:
            # where it is now, which a moved or relinked tier changes
            return dataclasses.replace(cached, provenance=provenance)
    amplitudes, labels, boxes = _extract(
        dump, k_cut=k_cut, atom_types=atom_types, ordering=ordering,
        route=route, device=DEVICE if route == "dense" else None,
        skip_frames=int(skip_frames))
    if fields != PER_TYPE:
        amplitudes = combine_fields(amplitudes, labels, boxes, fields,
                                    atom_types)
    record = ModesRecord(amplitudes, labels, boxes, float(T), float(dt_frame),
                         provenance)
    if entry is not None:
        record.save(entry)
    return record


def modes(system: System, tag: str, *, sigma: float, k_cut: float,
          ordering: str = "lexicographic", route: str = "dense") -> ModesRecord:
    """The farm convenience: one state point's modes, by ``farm_dir`` or ``tag``.

    Written where the archive keeps a run's file, ``modes/<farm_dir>/modes.npz`` (a partitioned farm's
    ``<partition>`` is the archive's tree, ``<tag>`` its run directory), so a training source reads it
    with the archive's own pattern and run tags. Other inputs for the same run replace it.
    The channels are the system's declared :func:`mode_fields`. ``skip_frames`` comes from
    :func:`frames_prepared`. ``ordering`` and ``route`` default to the archive's algorithm.

    ``system``: the declared system whose farm holds the state point.
    ``tag``: the state point's ``farm_dir`` or run tag.
    ``sigma``: the coarse-graining width, in the coordinates' length unit.
    ``k_cut``: the mode cutoff, in the reciprocal length unit.
    ``ordering``: one of :data:`aipf.pipeline.extract_modes.ORDERINGS`.
    ``route``: one of :data:`aipf.pipeline.extract_modes.ROUTES`.
    """
    from aipf.data import index

    record = index.record_for_tag(system, tag)
    meta = record["meta"]
    dump = Path(record["trajectory"])
    fields = mode_fields(system)
    return modes_from_dump(
        dump,
        sigma=sigma, k_cut=k_cut,
        atom_types=extracted_types(
            fields, [system.atom_types[s] for s in system.species]),
        fields=fields, ordering=ordering, route=route,
        T=float(meta["T_K"]), dt_frame=float(meta["dump_every_ps"]),
        skip_frames=frames_prepared(record, dump),
        composition=composition_label(meta.get("composition") or {},
                                      x_key=system.table_keys["x"]),
        entry_dir=index.modes_dir(system, record["farm_dir"]))
