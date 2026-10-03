"""The organised view over data that stays where it is: a symlink farm, new metadata, a manifest.

Source trees are read only. Every trajectory-shaped file ends up indexed, skipped or unrecognised.
Trajectory tiers write ``meta.json`` and ``manifest.json``. The published checkpoints are tracked files of the
checkout; the farm links none.
"""
from __future__ import annotations

import difflib
import fnmatch
import json
import re
from collections.abc import Iterable
from pathlib import Path

from aipf.data.meta import SCHEMA_VERSION, moving_axes, validate
from aipf.paths import CKPT_DIRNAME, FIELDS_DIRNAME, raw_env_name
from aipf.system import System

#: Run-tag prefix -> (geometry, ensemble), matched by ``startswith`` in order: most specific first.
#: The last resort for the ensemble (:func:`_classify`): a run's ``run.json`` records it, and a bare
#: geometry prefix (``cube``, ``slab``, ...) takes the one the system's declared md deck of that
#: geometry integrates. ``cube`` is ``NPT``: every archived two-species cube ran ``fix npt ... iso``.
_GEOMETRY_ENSEMBLE: tuple[tuple[str, str, str], ...] = (
    ("slab_overdamped", "slab", "langevin_overdamped"),
    ("slab_meltfirst", "slab", "NVT"),
    ("homogeneous_brownian", "cube", "langevin_overdamped"),
    ("nucleation_seed", "cube", "langevin_overdamped"),
    ("spinodal_cube", "cube", "langevin_overdamped"),
    ("cube", "cube", "NPT"),
    ("slab", "slab", "NPT_z"),
    ("ball", "ball", "NVT"),
    ("column", "column", "NVT"),
    ("vext", "vext", "NVT"),
    # Only the overdamped Brownian production stage is dumped.
    ("quench", "quench", "langevin_overdamped"),
)

_THERMO_NAMES = ("thermo_prod.txt", "log.lammps")

#: The one log file also read as a metadata source: see ``_meta_from_log``.
_LOG_FILENAME = "log.lammps"

#: Suffixes that make a file trajectory-shaped for the accounting in ``build``.
_TRAJ_SHAPE_SUFFIXES = frozenset({".lammpstrj", ".dump"})

#: Frame count used when the source does not record one. Frames are never counted here.
_UNKNOWN_FRAMES = 1

# Metadata recovered from a LAMMPS log. Patterns match the final, fully substituted command line.
_LOG_TIMESTEP_RE = re.compile(r"^timestep\s+([0-9.eE+-]+)\s*$", re.MULTILINE)
_LOG_RUN_RE = re.compile(r"^run\s+(\d+)\s*$", re.MULTILINE)
#: ``fix <id> all brownian <T> <seed> gamma_t <gamma>``, overdamped Langevin.
_LOG_BROWNIAN_RE = re.compile(
    r"^fix\s+\S+\s+all\s+brownian\s+([0-9.eE+-]+)\s+(\d+)\s+gamma_t\s+"
    r"([0-9.eE+-]+)\s*$", re.MULTILINE)
#: ``fix <id> all langevin <Tstart> <Tend> <damp> <seed>``, a velocity-thermostatted stage.
_LOG_LANGEVIN_RE = re.compile(
    r"^fix\s+\S+\s+all\s+langevin\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
    r"([0-9.eE+-]+)\s+(\d+)\s*$", re.MULTILINE)
_LOG_DUMP_RE = re.compile(
    r"^dump\s+\S+\s+all\s+custom\s+(\d+)\s+\S+\s+id\s+type\s+x\s+y\s+z\s*$",
    re.MULTILINE)
#: ``dump_modify <id> ... first yes``: a frame at the first step of the run, whatever the cadence.
_LOG_DUMP_FIRST_RE = re.compile(r"^dump_modify\s+\S+\s.*\bfirst\s+yes\b", re.MULTILINE)
_LOG_CREATED_ATOMS_RE = re.compile(r"^Created\s+(\d+)\s+atoms", re.MULTILINE)
#: The ``N atoms`` line printed under "reading atoms ..." by ``read_data``.
_LOG_READ_ATOMS_RE = re.compile(r"^\s*(\d+)\s+atoms\s*$", re.MULTILINE)
#: ``delete_atoms`` reports the count it leaves: ``Deleted M atoms, new total = N``.
_LOG_DELETED_ATOMS_RE = re.compile(r"^Deleted\s+\d+\s+atoms, new total = (\d+)", re.MULTILINE)
#: ``[Created ]orthogonal box = (xlo ylo zlo) to (xhi yhi zhi)``, printed by ``create_box`` and ``read_data``.
_LOG_BOX_RE = re.compile(
    r"^\s*(?:Created\s+)?orthogonal box = \((\S+) (\S+) (\S+)\) to \((\S+) (\S+) (\S+)\)",
    re.MULTILINE)


def _initial_cube_edge(text: str) -> float | None:
    """The edge of the first box a LAMMPS log reports, when that box is a cube; ``None`` otherwise."""
    m = _LOG_BOX_RE.search(text)
    if m is None:
        return None
    lo = [float(v) for v in m.groups()[:3]]
    hi = [float(v) for v in m.groups()[3:]]
    edges = [b - a for a, b in zip(lo, hi)]
    if max(edges) - min(edges) > 1e-9 * max(edges):
        return None
    return edges[0]


def _meta_from_log(log_path: Path) -> dict:
    """Best-effort source metadata from a LAMMPS log, in ``meta.json`` key spelling.

    Keys not found stay absent, never guessed. The LAST match of each pattern wins.
    """
    # Replace, not raise, on non-ASCII bytes, as LAMMPS itself does.
    text = log_path.read_text(encoding="utf-8", errors="replace")

    src: dict = {"engine": "lammps"}

    dt = _LOG_TIMESTEP_RE.findall(text)
    if dt:
        src["dt_ps"] = float(dt[-1])

    runs = _LOG_RUN_RE.findall(text)
    if runs:
        src["n_prod"] = int(runs[-1])
        # every run before the last is preparation; a log with one run has none
        src["n_equil"] = sum(int(n) for n in runs[:-1])

    # The thermostat (brownian or langevin) that appears last in the file wins.
    t_candidates = []
    for m in _LOG_BROWNIAN_RE.finditer(text):
        t_candidates.append((m.end(), float(m.group(1)), int(m.group(2))))
    for m in _LOG_LANGEVIN_RE.finditer(text):
        # Tend, not Tstart: the target the run holds at.
        t_candidates.append((m.end(), float(m.group(2)), int(m.group(4))))
    if t_candidates:
        t_candidates.sort(key=lambda c: c[0])
        _, t_k, seed = t_candidates[-1]
        src["T_K"] = t_k
        src["seed"] = seed

    dumps = _LOG_DUMP_RE.findall(text)
    if dumps:
        src["dump_freq"] = int(dumps[-1])
        if runs and _LOG_DUMP_FIRST_RE.search(text):
            # the forced first frame plus one per full cadence of the production run
            src["n_frames"] = int(runs[-1]) // int(dumps[-1]) + 1

    # create_atoms, read_data and delete_atoms each report a count: the last one in the file wins.
    atom_candidates = [(m.end(), int(m.group(1)))
                        for m in _LOG_CREATED_ATOMS_RE.finditer(text)]
    atom_candidates += [(m.end(), int(m.group(1)))
                         for m in _LOG_READ_ATOMS_RE.finditer(text)]
    atom_candidates += [(m.end(), int(m.group(1)))
                         for m in _LOG_DELETED_ATOMS_RE.finditer(text)]
    if atom_candidates:
        atom_candidates.sort(key=lambda c: c[0])
        src["n_atoms"] = atom_candidates[-1][1]

    return src


def _pressure_dir(p_gpa: float) -> str:
    """Farm subdirectory name for a pressure: ``200GPa``, or ``{p:g}GPa`` when not an integer."""
    if float(p_gpa).is_integer():
        return f"{int(p_gpa)}GPa"
    return f"{p_gpa:g}GPa"


#: How a declared partition field renders as one directory segment. Default ``str``.
_PARTITION_FORMATTERS = {"P_GPa": _pressure_dir}


def _farm_partition_dir(system: System, meta: dict) -> tuple[str, str | None]:
    """Farm subdirectory for the fields ``system.farm_partition`` declares, joined by ``/``.

    Returns ``(directory, None)``, or ``("", field)`` naming a declared field whose value is null.
    """
    parts = []
    for field in system.farm_partition:
        value = meta.get(field)
        if value is None:
            return "", field
        formatter = _PARTITION_FORMATTERS.get(field, str)
        parts.append(formatter(value))
    return "/".join(parts), None


def _declared_deck_ensemble(system: System, geometry: str) -> str | None:
    """The ensemble ``system``'s declared md decks (``defaults["md"]``) of ``geometry`` integrate;
    ``None`` when it declares none. Decks of one geometry that disagree raise: there is no rule to
    pick one."""
    from aipf.md.request import normalise_ensemble, normalise_geometry

    found = {}
    for name, deck in (system.defaults.get("md") or {}).items():
        point = deck.get("point") if isinstance(deck, dict) else None
        if not isinstance(point, dict) or "geometry" not in point or "ensemble" not in point:
            continue
        if normalise_geometry(point["geometry"]) == geometry:
            found[name] = normalise_ensemble(point["ensemble"])
    if len(set(found.values())) > 1:
        raise ValueError(
            f"system {system.name!r} declares md decks of geometry {geometry!r} that integrate "
            f"different ensembles ({found}), so an archived run's ensemble cannot be read off them")
    return next(iter(found.values()), None)


def _classify(tag: str, system: System | None = None) -> tuple[str, str]:
    """``(geometry, ensemble)`` of an archived run without ``run.json``, from its tag's prefix.

    The ensemble: for a bare geometry prefix (the prefix is the geometry's own name), the one
    ``system``'s declared md deck of that geometry integrates, when it declares one; else the
    table's. A protocol prefix (``slab_meltfirst``, ``homogeneous_brownian``, ...) names its own."""
    for prefix, geometry, ensemble in _GEOMETRY_ENSEMBLE:
        if tag.startswith(prefix):
            if system is not None and prefix == geometry:
                ensemble = _declared_deck_ensemble(system, geometry) or ensemble
            return geometry, ensemble
    raise ValueError(
        f"cannot classify run tag {tag!r}. Add its prefix to _GEOMETRY_ENSEMBLE "
        f"rather than guessing at read time")


def _find(directory: Path, names: tuple[str, ...]) -> Path | None:
    for n in names:
        p = directory / n
        if p.is_file():
            return p
    return None


def _reconstruct_tag(system: System, run_dir: Path, raw: Path) -> str:
    """The run's tag: the leaf directory name, or (``system.tag_from_path``) the path below ``raw``.

    The path form drops ``system.tag_path_root`` and joins the remaining components by ``_``.
    """
    if not system.tag_from_path:
        return run_dir.name
    parts = run_dir.relative_to(raw).parts
    if system.tag_path_root and parts and parts[0] == system.tag_path_root:
        parts = parts[1:]
    return "_".join(parts)


def _source_meta(directory: Path, tag: str) -> dict:
    """Source metadata: ``<tag>_meta.json`` or ``meta.json``, else ``log.lammps``, else ``{}``."""
    for candidate in (directory / f"{tag}_meta.json", directory / "meta.json"):
        if candidate.is_file():
            return json.loads(candidate.read_text())
    log = directory / _LOG_FILENAME
    if log.is_file():
        return _meta_from_log(log)
    return {}


#: What ``aipf md run`` writes beside the deck (``aipf.md.run.RECORD_FILE``), and the keys that identify it.
_RUN_RECORD_FILENAME = "run.json"
_RUN_RECORD_KEYS = frozenset({"system", "template", "point", "dry_run", "result"})


def _run_record(directory: Path) -> dict | None:
    """The ``aipf md run`` record in ``directory``; ``None`` when there is none (or a file of that name is not one)."""
    path = directory / _RUN_RECORD_FILENAME
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except ValueError:
        return None
    return record if isinstance(record, dict) and _RUN_RECORD_KEYS <= set(record) else None


def _meta_from_run_record(system: System, run_dir: Path, record: dict) -> dict:
    """``meta.json`` for a run ``aipf md run`` wrote: the request in ``run.json`` (tag, geometry, ensemble, T,
    composition, timestep, cadence, seed) and the log for what the engine did. ``ValueError``: why not."""
    from aipf.md.request import StatePoint

    if record["system"] != system.name:
        raise ValueError(f"{_RUN_RECORD_FILENAME} is a run of system {record['system']!r}, "
                         f"not {system.name!r}")
    if record["dry_run"] or record["result"] is None:
        raise ValueError(f"{_RUN_RECORD_FILENAME} records a dry run: nothing was run")
    point = dict(record["point"])
    if isinstance(point.get("x"), list):
        point["x"] = tuple(point["x"])
    request = StatePoint(**point)
    log = run_dir / _LOG_FILENAME
    src = _meta_from_log(log) if log.is_file() else {"engine": "lammps"}
    box_edge = (_initial_cube_edge(log.read_text(encoding="utf-8", errors="replace"))
                if log.is_file() else None)
    measured = src.get("n_atoms")
    if request.n_atoms is not None and measured is not None and measured != request.n_atoms:
        raise ValueError(
            f"{_RUN_RECORD_FILENAME} requests n_atoms={request.n_atoms} and "
            f"{_LOG_FILENAME} counts {measured}: the deck's configuration makes the atoms")
    fields = request.to_meta_fields()
    if fields["n_atoms"] is None:
        fields["n_atoms"] = measured
    if fields["P_GPa"] is None:
        fields["P_GPa"] = system.constants.get("P_GPa")
    fields["n_frames"] = src.get("n_frames", fields["n_frames"])
    if isinstance(request.x, tuple):
        composition = {"species": list(system.species), "kind": "regions",
                       "x": {system.species[system.table_keys.get("x_channel", 0)]: list(request.x)}}
    else:
        composition = {"species": list(system.species), "kind": "uniform",
                       "x": {system.species[system.table_keys.get("x_channel", 0)]: request.x}}
    status = (record["result"] or {}).get("status", "unknown")
    return {
        "schema_version": SCHEMA_VERSION,
        "system": system.name,
        "tag": fields["tag"],
        "geometry": fields["geometry"],
        "ensemble": fields["ensemble"],
        "T_K": float(fields["T_K"]),
        "P_GPa": fields["P_GPa"],
        "composition": composition,
        "box": {"L": box_edge, "varying": moving_axes(fields["ensemble"])},
        "n_atoms": None if fields["n_atoms"] is None else int(fields["n_atoms"]),
        "dt_ps": float(fields["dt_ps"]),
        "dump_every_ps": fields["dump_every_ps"],
        "n_frames": int(fields["n_frames"]),
        "engine": src.get("engine", "lammps"),
        "potential": record.get("potential"),
        "seed": int(fields["seed"]),
        "status": status,
        "extra": {**{k: v for k, v in src.items() if k not in {"T_K", "n_atoms", "dt_ps", "seed",
                                                                 "n_frames", "engine"}},
                  "template": record["template"], "campaign": record.get("campaign"),
                  "run_record": str(run_dir / _RUN_RECORD_FILENAME)},
    }


def _dump_every_ps(src: dict) -> float:
    """Picoseconds between frames: ``dump_every_ps``, or ``dump_freq * dt_ps``."""
    if "dump_every_ps" in src:
        return float(src["dump_every_ps"])
    if "dump_freq" in src and "dt_ps" in src:
        return float(src["dump_freq"]) * float(src["dt_ps"])
    raise KeyError("no dump cadence: need dump_every_ps, or dump_freq with dt_ps")


def _n_frames(src: dict) -> int:
    """Frame count, derived when the source tree does not record one."""
    if src.get("n_frames"):
        return int(src["n_frames"])
    if src.get("n_prod") and src.get("dump_freq"):
        return int(src["n_prod"]) // int(src["dump_freq"])
    return _UNKNOWN_FRAMES


def _composition(system: System, tag: str, src: dict) -> dict:
    """Composition, uniform or two-slab (``_xl<a>_xr<b>`` in the tag)."""
    labelled = system.species[system.table_keys.get("x_channel", 0)]

    pair = dict(re.findall(r"_x([lr])([0-9.]+)", tag))
    if "l" in pair and "r" in pair:
        return {"species": list(system.species), "kind": "two_slab",
                "x": {labelled: [float(pair["l"]), float(pair["r"])]}}

    x_key = system.table_keys.get("x", "x")
    x_value = src.get(x_key)
    if x_value is None:
        # ``_x0.05``, or ``_xA_0.50`` when the fraction's species is spelled in the tag
        m = re.search(r"_x(?:[A-Za-z]+_)?([0-9.]+)", tag)
        x_value = float(m.group(1)) if m else None
    return {"species": list(system.species), "kind": "uniform",
            "x": {labelled: x_value}}


def _normalise(system: System, tag: str, src: dict) -> dict:
    geometry, ensemble = _classify(tag, system)

    return {
        "schema_version": SCHEMA_VERSION,
        "system": system.name,
        "tag": tag,
        "geometry": geometry,
        "ensemble": ensemble,
        "T_K": float(src["T_K"]),
        "P_GPa": src.get("P_GPa", system.constants.get("P_GPa")),
        "composition": _composition(system, tag, src),
        "box": {"L": src.get("box_L_A"), "varying": moving_axes(ensemble)},
        "n_atoms": int(src["n_atoms"]),
        "dt_ps": float(src["dt_ps"]),
        "dump_every_ps": _dump_every_ps(src),
        "n_frames": _n_frames(src),
        "engine": src.get("engine", "unknown"),
        "potential": {"path": src["model"]} if "model" in src else None,
        "seed": int(src.get("seed", src.get("seed_used", 0))),
        "status": src.get("status", "unknown"),
        "extra": {k: v for k, v in src.items()
                  if k not in {"tag", "T_K", "P_GPa", "n_atoms", "dt_ps",
                               "dump_every_ps", "n_frames", "seed", "status"}},
    }


def _link(target: Path, link: Path) -> None:
    """Point `link` at `target`, replacing a stale link. A regular file there raises."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(
            f"refusing to replace {link}: it is not a symlink. The farm holds "
            f"only links, so a real file here means data was copied in. "
            f"Inspect it and remove it by hand if it is safe to lose."
        )
    link.symlink_to(target)


class ForeignManifest(ValueError):
    """``build`` was asked to replace a manifest that indexes another raw root."""


#: The manifest key that records the raw root a build read. An older manifest spells it
#: ``"scratch"``, which is still read.
_RAW_KEY, _RAW_KEY_BEFORE = "raw", "scratch"


def manifest_raw_root(manifest: dict) -> str | None:
    """The raw root a manifest was built from, ``None`` when it does not say."""
    return manifest.get(_RAW_KEY, manifest.get(_RAW_KEY_BEFORE))


def _refuse_foreign_manifest(system: System, raw: Path, path: Path) -> None:
    """Refuse, before anything is written, to replace a manifest built from another raw root than ``raw``."""
    if not path.is_file():
        return
    try:
        indexed_from = manifest_raw_root(json.loads(path.read_text()))
    except ValueError:
        indexed_from = None
    if indexed_from is not None and Path(indexed_from) != Path(raw):
        raise ForeignManifest(
            f"{path} indexes the raw root {indexed_from}, and this build reads {raw} "
            f"({raw_env_name(system.name)}, AIPF_RAW or [paths.raw] in aipf.toml). Building "
            f"would replace that index. Point AIPF_DATA at a farm of its own for this tree, "
            f"or remove {path} if the old index is no longer wanted")


def build(system: System, dry_run: bool = False) -> dict:
    """Index one system's trajectories. Returns the manifest.

    Each trajectory-shaped file is indexed, skipped with a reason, or unrecognised, and the counts must sum.
    """
    raw = system.paths.raw()
    out = system.paths.data_root()
    if not dry_run:
        _refuse_foreign_manifest(system, raw, out / "manifest.json")

    # ``farm_dir`` is ``<partition>/<tag>`` because tags repeat across partitions. A collision raises.
    seen_farm_dirs: dict[str, str] = {}

    # Top-level raw subdirectories the system declares as not production data.
    md_exclude_dirs = set(system.constants.get("md_exclude_dirs", ()))

    state_points: list[dict] = []
    unrecognised: list[dict] = []
    n_traj_shaped = 0
    for run_dir in sorted(p for p in raw.rglob("*") if p.is_dir()):
        traj = _find(run_dir, system.traj_names)

        # Every trajectory-shaped file here (suffix or declared name) other than ``traj`` is unrecognised.
        traj_shaped_here = [f for f in run_dir.iterdir()
                             if f.is_file()
                             and (f.suffix in _TRAJ_SHAPE_SUFFIXES
                                  or f.name in system.traj_names)]
        n_traj_shaped += len(traj_shaped_here)
        for f in traj_shaped_here:
            if f != traj:
                unrecognised.append({"file": str(f)})

        if traj is None:
            continue
        run_record = _run_record(run_dir)
        tag = _reconstruct_tag(system, run_dir, raw)

        rel_parts = run_dir.relative_to(raw).parts
        if rel_parts and rel_parts[0] in md_exclude_dirs:
            state_points.append({
                "tag": tag,
                "skipped": f"excluded: under {rel_parts[0]!r}, declared in "
                           f"system.constants['md_exclude_dirs']",
            })
            continue

        try:
            if run_record is not None:
                meta = _meta_from_run_record(system, run_dir, run_record)
                tag = meta["tag"]
            else:
                meta = _normalise(system, tag, _source_meta(run_dir, tag))
        except (ValueError, KeyError, TypeError) as exc:
            state_points.append({"tag": tag, "skipped": str(exc)})
            continue

        problems = validate(meta)
        if problems:
            state_points.append({"tag": tag, "skipped": "; ".join(problems)})
            continue

        partition, null_field = _farm_partition_dir(system, meta)
        if null_field is not None:
            # A null declared partition field has no deterministic place in the farm.
            state_points.append({
                "tag": tag,
                "skipped": f"{null_field} is null: cannot place under the "
                           f"partition-keyed farm layout "
                           f"(md/<partition>/<tag>) declared by "
                           f"system.farm_partition",
            })
            continue

        farm_dir = f"{partition}/{tag}" if partition else tag
        if farm_dir in seen_farm_dirs:
            prior_source = seen_farm_dirs[farm_dir]
            raise RuntimeError(
                f"genuine farm-directory collision at {farm_dir!r}: "
                f"{prior_source!r} and {str(run_dir)!r} both normalise to "
                f"the same partition and tag. This is a real ambiguity in "
                f"the source data, not an ordering accident, so it is not "
                f"disambiguated automatically -- inspect both directories "
                f"by hand.")
        seen_farm_dirs[farm_dir] = str(run_dir)

        record = {"tag": tag, "farm_dir": farm_dir, "meta": meta,
                  "source_dir": str(run_dir), "trajectory": str(traj)}
        thermo = _find(run_dir, _THERMO_NAMES)
        if thermo:
            record["thermo"] = str(thermo)
        state_points.append(record)

        if not dry_run:
            dest = out / "md" / farm_dir
            _link(traj, dest / "traj.lammpstrj")
            if thermo:
                _link(thermo, dest / "thermo.txt")
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "meta.json").write_text(json.dumps(meta, indent=1))

    n_skipped = sum(1 for r in state_points if "skipped" in r)
    n_indexed = len(state_points) - n_skipped

    # Every trajectory-shaped file is indexed, skipped or unrecognised, exactly once.
    accounted = n_indexed + n_skipped + len(unrecognised)
    if accounted != n_traj_shaped:
        raise RuntimeError(
            f"accounting mismatch for {system.name!r}: found {n_traj_shaped} "
            f"trajectory-shaped file(s) under {raw}, but indexed "
            f"({n_indexed}) + skipped ({n_skipped}) + unrecognised "
            f"({len(unrecognised)}) = {accounted}. Every trajectory-shaped "
            f"file must land in exactly one of those three buckets.")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "system": system.name,
        "raw": str(raw),
        "state_points": state_points,
        "n_indexed": n_indexed,
        "n_skipped": n_skipped,
        "unrecognised": unrecognised,
        "n_unrecognised": len(unrecognised),
    }
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _indexed_records(system: System, root: Path | None = None) -> list[dict]:
    """Every indexed state point of one system's manifest, in manifest order. Skipped ones are dropped."""
    path = _farm_root(system, root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no manifest for system {system.name!r} at {path}. Build the "
            f"index first; a lookup cannot invent the trajectory a tag names")
    return [r for r in read_manifest(path)["state_points"] if "skipped" not in r]


def _nearby(wanted: str, known: Iterable[str], limit: int = 5) -> list[str]:
    """The closest known names to one that was not found."""
    return difflib.get_close_matches(wanted, sorted(known), n=limit, cutoff=0.3)


def record_for_tag(system: System, tag: str,
                   root: Path | None = None) -> dict:
    """The manifest record filed under ``tag``, matched on ``farm_dir`` first and ``tag`` second.

    An ambiguous bare tag raises. A miss raises ``KeyError`` naming the closest tags that exist.
    """
    records = _indexed_records(system, root)
    for record in records:
        if record.get("farm_dir") == tag:
            return record
    matches = [r for r in records if r.get("tag") == tag]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(
            f"tag {tag!r} names {len(matches)} state points of system "
            f"{system.name!r}: {[r['farm_dir'] for r in matches]}. Name the "
            f"one wanted by its farm directory, which carries the partition")
    known = {r.get("farm_dir", r["tag"]) for r in records} | \
        {r["tag"] for r in records}
    near = _nearby(tag, known)
    raise KeyError(
        f"no state point {tag!r} in the manifest for system "
        f"{system.name!r} ({len(records)} indexed). "
        + (f"Nearest: {near}" if near else "Nothing there is close to it"))


def record_for_pattern(system: System, pattern: str,
                       root: Path | None = None) -> dict:
    """The first indexed record, in manifest order, whose farm directory matches a glob."""
    records = _indexed_records(system, root)
    for record in records:
        if fnmatch.fnmatch(record.get("farm_dir", record["tag"]), pattern):
            return record
    raise KeyError(
        f"no state point of system {system.name!r} has a farm directory "
        f"matching {pattern!r} ({len(records)} indexed)")


def _farm_root(system: System, root: Path | None) -> Path:
    """This system's farm directory: ``root / system.name``, else ``system.paths.data_root()``."""
    if root is not None:
        return Path(root) / system.name
    return system.paths.data_root()


def ckpt_dir(system: System, root: Path | None = None) -> Path:
    """Where this system's checkpoints live in the farm: one directory per run, and ``published/``."""
    return _farm_root(system, root) / CKPT_DIRNAME


def modes_dir(system: System, farm_dir: str, root: Path | None = None) -> Path:
    """The coarse-grained cache for one state point, keyed by the ``md`` tier's ``farm_dir``."""
    return _farm_root(system, root) / "modes" / farm_dir


def diagnose_dir(system: System, root: Path | None = None) -> Path:
    """Where diagnostic output for this system's checkpoints is written."""
    return _farm_root(system, root) / "diagnose"


def rebuild_from_manifest(path: str | Path) -> int:
    """Recreate the md tier of the symlink farm from a manifest. Returns the number of state points
    replayed. The ``modes`` cache is never rebuilt.
    """
    path = Path(path)
    manifest = read_manifest(path)
    out = path.parent
    n = 0
    for record in manifest["state_points"]:
        if "skipped" in record:
            continue
        dest = out / "md" / record.get("farm_dir", record["tag"])
        _link(Path(record["trajectory"]), dest / "traj.lammpstrj")
        if "thermo" in record:
            _link(Path(record["thermo"]), dest / "thermo.txt")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "meta.json").write_text(json.dumps(record["meta"], indent=1))
        n += 1
    return n


def index_fields(system: System) -> dict:
    """Index the coarse-grained field trees under ``fields/``, each with its role.

    Roles come from the ``constants`` keys ``"modes_trees"``, ``"fdt_runs_trees"`` and ``"table_trees"``.
    Any other tree is ``"archived"``.
    """
    fields_dir = system.paths.raw() / FIELDS_DIRNAME
    roles = {}
    for role, key in (("modes", "modes_trees"),
                      ("fdt_runs", "fdt_runs_trees"),
                      ("tables", "table_trees")):
        for name in system.constants.get(key, ()):
            roles[name] = role

    trees = []
    if fields_dir.is_dir():
        for d in sorted(p for p in fields_dir.iterdir() if p.is_dir()):
            trees.append({
                "name": d.name,
                "path": str(d),
                "role": roles.get(d.name, "archived"),
                "entries": sum(1 for _ in d.iterdir()),
            })
    return {"system": system.name, "root": str(fields_dir), "trees": trees}
