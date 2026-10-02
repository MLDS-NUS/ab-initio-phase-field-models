"""The System object and how one is found.
A system is everything the general package cannot know. Each lives in the repository's
``experiments/<name>/system.py`` as one ``SYSTEM`` object; construction checks its declarations agree."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

# Shared with aipf.paths rather than duplicated.
from aipf import paths as _paths
from aipf.paths import PUBLISHED_DIRNAME, TRACKED_FILE_NAME, Paths

#: Environment variable that overrides where experiment folders are looked for.
_EXPERIMENTS_ENV = "AIPF_EXPERIMENTS"

#: The one file an experiment folder must contain.
_SYSTEM_FILENAME = "system.py"

#: Attribute the experiment file must define.
_SYSTEM_ATTR = "SYSTEM"

#: A digest as ``md5sum`` prints one.
_MD5_DIGEST = re.compile(r"[0-9a-f]{32}")

#: Read size for digesting a file.
_DIGEST_BLOCK = 1 << 20

def _lookup(table: Mapping, pressure_gpa: float) -> float | None:
    """One per-pressure temperature table, matched numerically by pressure."""
    for key, value in table.items():
        if float(key) == float(pressure_gpa):
            return float(value)
    return None


def _is_one_directory_name(name: object) -> bool:
    """``name`` is a string naming one directory: not empty, not ``.`` or ``..``, no separator."""
    return (isinstance(name, str) and name not in ("", ".", "..")
            and Path(name).name == name)


def _file_md5(path: Path) -> str:
    """Hex digest of a file's bytes."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_DIGEST_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AnchorRules:
    """The two per-pressure temperature tables (GPa to K), which are not the same numbers.
    ``T_min_by_pressure``: lowest admissible anchor temperature. ``shape_T_min_by_pressure``: lowest
    temperature entering the kernel-shape set (empty when undeclared)."""

    T_min_by_pressure: dict[float, float] = field(default_factory=dict)
    shape_T_min_by_pressure: dict[float, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MdSettings:
    """The engine knobs one campaign of one system ran with (one per campaign, not per system).
    ``units`` names the engine's unit system; ``skin``, ``T_damp``, ``P_damp`` are in its units;
    ``neigh_modify`` is verbatim; ``thermo_every >= 1`` steps; ``pressure_scale`` converts request pressure."""

    units: str
    skin: float
    neigh_modify: str
    thermo_every: int
    T_damp: float
    P_damp: float
    pressure_scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError(
                f"units={self.units!r} names no unit system, and every other "
                f"number declared here is in the units it fixes")
        if not isinstance(self.neigh_modify, str) \
                or not self.neigh_modify.strip():
            raise ValueError(
                f"neigh_modify={self.neigh_modify!r} is empty. It is written "
                f"into a deck as one command's arguments, and a command "
                f"missing its arguments is a parse error the engine reports "
                f"only once the job is running")
        for name in ("skin", "T_damp", "P_damp", "pressure_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not value > 0.0:
                raise ValueError(f"{name}={value!r} is not a positive number")
        if isinstance(self.thermo_every, bool) \
                or not isinstance(self.thermo_every, int) \
                or self.thermo_every < 1:
            raise ValueError(
                f"thermo_every={self.thermo_every!r} is not a whole number of "
                f"steps of at least one")

    def deck_pressure(self, pressure: float) -> float:
        """A request's pressure in the unit a deck writes it in."""
        if isinstance(pressure, bool) or not isinstance(pressure,
                                                        (int, float)):
            raise TypeError(
                f"pressure={pressure!r} is not a number, so there is nothing "
                f"to convert")
        return float(pressure) * self.pressure_scale


@dataclass(frozen=True)
class PreparationRules:
    """How a state point is prepared, in the unit of a request's temperature: an absolute
    ``melt_T`` or an additive ``ramp_offset``, alternatives (declaring both is refused)."""

    melt_T: float | None = None
    ramp_offset: float | None = None

    def __post_init__(self) -> None:
        if self.melt_T is not None and self.ramp_offset is not None:
            raise ValueError(
                f"preparation declares both melt_T={self.melt_T!r} and "
                f"ramp_offset={self.ramp_offset!r}; they are alternatives "
                f"(a fixed hot temperature does not depend on the target, "
                f"an offset does), so declaring both describes two "
                f"different preparations and no consumer can choose "
                f"between them")


@dataclass(frozen=True)
class Checkpoint:
    """A trained file identified by its bytes: ``md5``, 32 lowercase hex digits (``md5sum``).
    A system's own checkpoint declares the digest alone: its file is the tracked ``final.ckpt`` of the
    system's ``ckpt/published`` directory, or of its ``<variant>`` subdirectory
    (:func:`aipf.paths.published_checkpoint`). Any other file, a run's own checkpoint as a starting
    point, also gives ``path``, relative to the raw root."""

    md5: str
    path: str | None = None

    def __post_init__(self) -> None:
        if self.path is not None and not self.path:
            raise ValueError(
                "checkpoint path is empty: name the file, relative to the "
                "system's raw root, or declare none for the tracked copy")
        if self.path is not None and Path(self.path).is_absolute():
            raise ValueError(
                f"checkpoint path {self.path!r} is absolute. It is joined to "
                f"the system's raw root, which the environment can move, "
                f"so it is written relative to that root")
        if not _MD5_DIGEST.fullmatch(self.md5):
            raise ValueError(
                f"checkpoint md5={self.md5!r} is not a digest: 32 lowercase "
                f"hexadecimal digits, as `md5sum` prints them")

    def _verified(self, target: Path, root: str | Path) -> Path:
        """``target`` once its bytes are proved to be this file's."""
        if not target.is_file():
            raise FileNotFoundError(
                f"no checkpoint at {target}. The declared file is missing or "
                f"renamed, which is not the same as unreachable: it was "
                f"looked for under the root given, {root}")
        found = _file_md5(target)
        if found != self.md5:
            raise ValueError(
                f"{target} has md5 {found}, not the declared {self.md5}. The "
                f"file at this path is not the one the published numbers came "
                f"from")
        return target

    def verify_under(self, root: str | Path) -> Path:
        """The file under ``root`` once proved to be this one; ``FileNotFoundError`` if missing,
        ``ValueError`` if its bytes differ or no ``path`` is declared."""
        if self.path is None:
            raise ValueError(
                f"checkpoint md5 {self.md5} declares no path under a root: it is a "
                f"system's tracked copy, found by System.resolve_checkpoint")
        return self._verified(Path(root) / self.path, root)

    @staticmethod
    def tracked_under(published: str | Path, variant: str | None = None) -> Path:
        """Where the tracked copy sits in one system's ``published`` directory: ``[<variant>/]final.ckpt``."""
        if variant is None:
            return Path(published) / TRACKED_FILE_NAME
        if not _is_one_directory_name(variant):
            raise ValueError(
                f"variant={variant!r} is not one directory name: the tracked "
                f"copy sits at {PUBLISHED_DIRNAME}/<variant>/{TRACKED_FILE_NAME}")
        return Path(published) / variant / TRACKED_FILE_NAME

    def resolve(self, system: str,
                raw: Path | Callable[[], Path],
                published: Path | Callable[[], Path] | None = None,
                variant: str | None = None) -> Path:
        """The declared bytes: the tracked copy in ``published`` (the system's directory, or a callable
        returning it; ``None``: none) first, then, when a ``path`` is declared, the raw root; digest
        verified either way. Refuses naming ``system`` when neither holds them."""
        refused: list[Exception] = []
        if published is not None:
            try:
                root = published() if callable(published) else Path(published)
                return self._verified(self.tracked_under(root, variant), root)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                refused.append(exc)
        if self.path is not None or not refused:
            try:
                return self.verify_under(raw() if callable(raw)
                                         else raw)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                refused.append(exc)
        kind = (ValueError if any(isinstance(e, ValueError) for e in refused)
                else FileNotFoundError)
        where = self.path if self.path is not None else "(the tracked copy)"
        raise kind(
            f"system {system!r}: no location holds the declared checkpoint "
            f"{where} (md5 {self.md5}). "
            + " ".join(str(e) for e in refused)) from refused[-1]


@dataclass(frozen=True)
class TrustDomain:
    """``rho_i >= 0``, ``sum_i rho_i / inner_i >= 1``, ``sum_i rho_i / outer_i <= 1``: per-channel axis
    intercepts (strictly positive, channel order), plus an optional ``(T_min, T_max)``."""

    inner: tuple[float, ...]
    outer: tuple[float, ...]
    T_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        inner = tuple(float(v) for v in self.inner)
        outer = tuple(float(v) for v in self.outer)
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "outer", outer)
        if len(inner) == 0:
            raise ValueError("TrustDomain needs at least one species: "
                              "inner is empty")
        if len(inner) != len(outer):
            raise ValueError(
                f"inner has {len(inner)} entries but outer has "
                f"{len(outer)}: one intercept per channel, both edges")
        for label, vals in (("inner", inner), ("outer", outer)):
            if any(v <= 0.0 for v in vals):
                raise ValueError(
                    f"{label}={vals!r} must be strictly positive in every "
                    f"channel: an intercept of zero or less makes "
                    f"rho_i / {label}_i undefined or sign-reversed")
        if self.T_range is not None:
            lo, hi = (float(v) for v in self.T_range)
            object.__setattr__(self, "T_range", (lo, hi))
            if lo > hi:
                raise ValueError(f"T_range={(lo, hi)!r} has T_min > T_max")

    @property
    def n_species(self) -> int:
        return len(self.inner)


#: Admissible ``Noise.mode`` and ``Noise.m_stab`` values; ``aipf.solve.noise`` reads these.
NOISE_MODES = ("gaussian", "none")
M_STAB_MODES = ("mean", "max")


@dataclass(frozen=True)
class Noise:
    """How a system's conserved noise is injected. ``mode``: ``"gaussian"`` filters it by
    ``exp(-k^2 sigma^2 / 2)``, sigma the declared coarse-graining ``defaults["sigma"]``; ``"none"`` is white.
    ``m_stab``: the mobility a semi-implicit operator freezes at, ``"mean"`` or ``"max"``."""

    mode: str
    m_stab: str

    def __post_init__(self) -> None:
        if self.mode not in NOISE_MODES:
            raise ValueError(
                f"Noise.mode={self.mode!r} is not one of {list(NOISE_MODES)}: "
                f"'gaussian' filters the noise by exp(-k^2 sigma^2/2) with "
                f"the system's own coarse-graining sigma, 'none' leaves it "
                f"white")
        if self.m_stab not in M_STAB_MODES:
            raise ValueError(
                f"Noise.m_stab={self.m_stab!r} is not one of "
                f"{list(M_STAB_MODES)}: 'mean' freezes the implicit operator "
                f"at M(mean density), 'max' at the initial field's "
                f"largest-norm cell")


def _frozen_mapping(m: Mapping[str, Any]) -> Mapping[str, Any]:
    """A read-only copy of a declared mapping."""
    return MappingProxyType(dict(m))


#: The free-energy ladder by physics; the paper numbers these rungs 1..4 in this order.
FUNCTIONAL_FORMS = ("landau", "square_gradient", "nonlocal_kernel",
                    "neural_operator")


@dataclass(frozen=True)
class Functional:
    """Which free-energy functional a system trains: ``form`` (one of ``FUNCTIONAL_FORMS``), ``local``,
    ``kernel`` (required for ``nonlocal_kernel``, refused otherwise) and the published model's ``kwargs``."""

    form: str
    local: str
    kernel: str | None
    kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.form not in FUNCTIONAL_FORMS:
            raise ValueError(
                f"unknown functional form {self.form!r}; one of "
                f"{', '.join(FUNCTIONAL_FORMS)}")
        needs_kernel = self.form == "nonlocal_kernel"
        if needs_kernel and self.kernel is None:
            raise ValueError(
                f"form {self.form!r} needs a kernel form; pass kernel=")
        if not needs_kernel and self.kernel is not None:
            raise ValueError(
                f"form {self.form!r} has no pair kernel; kernel must be None")
        object.__setattr__(self, "kwargs", _frozen_mapping(self.kwargs))


@dataclass(frozen=True)
class Mobility:
    """Which mobility a system trains: the density ``form``, the ``T_form`` factor and ``kwargs``.
    Not the whole mobility: read its arguments with ``aipf.mobility.mobility_kwargs(system)``."""

    form: str
    T_form: str
    kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", _frozen_mapping(self.kwargs))


@dataclass(frozen=True)
class Variant:
    """A second model of one system: its own ``functional``, ``mobility`` and ``checkpoint``, and the entries of
    the system's ``defaults`` it replaces (``defaults``, top-level keys); constants, paths and data are shared."""

    functional: Functional
    mobility: Mobility
    checkpoint: Checkpoint | None
    defaults: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, kind in (("functional", Functional), ("mobility", Mobility)):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"Variant.{name} is a {type(getattr(self, name)).__name__}, "
                                f"not a {kind.__name__}")
        if self.checkpoint is not None and not isinstance(self.checkpoint, Checkpoint):
            raise TypeError(f"Variant.checkpoint is a {type(self.checkpoint).__name__}, "
                            f"not a Checkpoint or None")
        object.__setattr__(self, "defaults", _frozen_mapping(self.defaults))


@dataclass(frozen=True)
class System:
    """One physical system, and everything specific to it.
    ``name`` (e.g. ``"demo"``), ``n_species``, ``species`` in channel order, ``masses`` (exactly the species),
    ``atom_types`` (covers the species, may carry extra dump types), ``table_keys`` (``"rho"`` per channel,
    ``"x_channel"`` index, ``"x"`` column), ``paths``, ``anchor_rules``, ``constants``, ``defaults``.
    ``farm_partition``: metadata fields splitting the farm (``()`` = unsplit). ``traj_names``: recognised
    trajectory filenames by priority. ``tag_from_path``/``tag_path_root``: rebuild a run tag from the path.
    ``checkpoint``, ``functional``, ``mobility``, ``noise``, ``trust_domain``: ``None`` means
    undeclared, never defaulted.
    ``md_settings``: engine knobs keyed by campaign name. ``variants``: other declared models, by name
    (:meth:`variant`); empty when there are none. ``variant_name``: set by :meth:`variant` on the system it
    returns; ``None`` is the production model."""

    name: str
    n_species: int
    species: tuple[str, ...]
    masses: dict[str, float]
    atom_types: dict[str, int]
    table_keys: dict
    paths: Paths
    anchor_rules: AnchorRules
    constants: dict
    defaults: dict
    farm_partition: tuple[str, ...] = ("P_GPa",)
    traj_names: tuple[str, ...] = ("traj.lammpstrj", "traj.dump", "dump.lammpstrj")
    tag_from_path: bool = False
    tag_path_root: str | None = None
    preparation: PreparationRules = field(default_factory=PreparationRules)
    checkpoint: Checkpoint | None = None
    functional: Functional | None = None
    mobility: Mobility | None = None
    md_settings: dict[str, MdSettings] = field(default_factory=dict)
    noise: Noise | None = None
    trust_domain: TrustDomain | None = None
    variants: Mapping[str, Variant] = field(default_factory=dict)
    variant_name: str | None = None

    def __post_init__(self) -> None:
        # Checked from the most basic outwards.
        if self.n_species < 1:
            raise ValueError(
                f"n_species={self.n_species}: a system carries at least one "
                f"density channel"
            )
        if len(self.species) != self.n_species:
            raise ValueError(
                f"n_species={self.n_species} but species={self.species!r} "
                f"has {len(self.species)} entries"
            )
        # A repeated name would make a channel unreachable through ``channel_of``.
        if len(set(self.species)) != len(self.species):
            duplicates = sorted({s for s in self.species
                                 if self.species.count(s) > 1})
            raise ValueError(
                f"species={self.species!r} repeats {', '.join(duplicates)}: "
                f"channel_of would answer with the first index for each copy"
            )
        # masses must match the channels exactly; atom_types must cover them (a dump may carry extra types).
        self._check_keyed_by_species("masses", self.masses, exact=True)
        self._check_keyed_by_species("atom_types", self.atom_types, exact=False)
        self._check_table_keys()
        self._check_anchor_rules()
        self._check_trust_domain()
        self._check_variants()

    def _check_keyed_by_species(self, label: str,
                                mapping: Mapping[str, object] | None,
                                *, exact: bool) -> None:
        """A mapping keyed by species covers every channel (``exact``: nothing else); ``None`` is refused."""
        if mapping is None:
            raise TypeError(
                f"{label} is None, not a mapping. An undeclared mapping is "
                f"written as an empty one, and None only looks undeclared."
            )
        if not mapping:
            return
        problems = []
        missing = [s for s in self.species if s not in mapping]
        if missing:
            problems.append(f"missing {missing}")
        if exact:
            extra = sorted(set(mapping) - set(self.species))
            if extra:
                problems.append(f"unknown {extra}")
        if problems:
            relation = "match" if exact else "cover"
            raise ValueError(
                f"{label}={dict(mapping)!r} does not {relation} species="
                f"{self.species!r}: " + ", ".join(problems)
            )

    def _check_table_keys(self) -> None:
        """``x`` is a column name, ``rho`` has one name per channel, ``x_channel`` is a valid int index."""
        x_key = self.table_keys.get("x")
        if x_key is not None and not isinstance(x_key, str):
            raise ValueError(
                f"table_keys['x']={x_key!r} is not a column name. It is looked "
                f"up in a row of a table, so it has to be a string"
            )
        rho_keys = self.table_keys.get("rho")
        if rho_keys is not None and len(rho_keys) != self.n_species:
            raise ValueError(
                f"table_keys['rho']={rho_keys!r} has {len(rho_keys)} column "
                f"names but the system has {self.n_species} density channels"
            )
        channel = self.table_keys.get("x_channel")
        if channel is None:
            return
        if not isinstance(channel, int) or isinstance(channel, bool):
            raise ValueError(
                f"table_keys['x_channel']={channel!r} is not an integer "
                f"index into species={self.species!r}"
            )
        if not 0 <= channel < self.n_species:
            raise ValueError(
                f"table_keys['x_channel']={channel!r} does not index "
                f"species={self.species!r}, which has {self.n_species} entries"
            )

    def _check_anchor_rules(self) -> None:
        """Both cut tables are numeric, keys pressures in GPa and values temperatures in K."""
        for name in ("T_min_by_pressure", "shape_T_min_by_pressure"):
            for key, value in getattr(self.anchor_rules, name).items():
                try:
                    float(key), float(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"anchor_rules.{name} has a non-numeric entry "
                        f"{key!r}: {value!r}. Keys are pressures in GPa and "
                        f"values are temperatures in K"
                    ) from None

    def _check_trust_domain(self) -> None:
        """A declared domain is a ``TrustDomain`` with one intercept per channel."""
        domain = self.trust_domain
        if domain is None:
            return
        if not isinstance(domain, TrustDomain):
            raise TypeError(
                f"trust_domain is a {type(domain).__name__}, not a "
                f"TrustDomain")
        if domain.n_species != self.n_species:
            raise ValueError(
                f"trust_domain has {domain.n_species} intercepts per edge "
                f"but the system has {self.n_species} density channels")

    def _check_variants(self) -> None:
        """Each variant is a ``Variant`` under one directory name (its tracked copy's directory)."""
        for name, variant in self.variants.items():
            if not _is_one_directory_name(name):
                raise ValueError(
                    f"variant name {name!r} is not one directory name, and it "
                    f"names the directory of the variant's published checkpoint")
            if not isinstance(variant, Variant):
                raise TypeError(f"variants[{name!r}] is a {type(variant).__name__}, not a Variant")

    def variant(self, name: str) -> "System":
        """This system with variant ``name``'s functional, mobility, checkpoint and ``defaults`` entries;
        ``KeyError`` naming the declared variants. The result declares no variants of its own."""
        if name not in self.variants:
            raise KeyError(
                f"system {self.name!r} declares no variant {name!r}; it declares "
                f"{sorted(self.variants) or 'none'}")
        v = self.variants[name]
        return dataclasses.replace(
            self, functional=v.functional, mobility=v.mobility,
            checkpoint=v.checkpoint, defaults={**self.defaults, **v.defaults},
            variants={}, variant_name=name)

    def channel_of(self, species: str) -> int:
        """Index of ``species`` in the channel ordering."""
        try:
            return self.species.index(species)
        except ValueError:
            raise ValueError(
                f"{species!r} is not a species of system {self.name!r}: "
                f"known species are {', '.join(self.species)}"
            ) from None

    def _require_checkpoint(self) -> Checkpoint:
        """This system's checkpoint, or a refusal that says whose fault it is."""
        if self.checkpoint is None:
            raise ValueError(
                f"system {self.name!r} declares no checkpoint, so there is no "
                f"file its published numbers can be reproduced from"
            )
        return self.checkpoint

    def checkpoint_file(self) -> Path:
        """Where this system's pinned checkpoint is (not checked): the tracked copy in ``self.paths.published()``
        (:meth:`Checkpoint.tracked_under`), or a declared ``path`` under the raw root."""
        ck = self._require_checkpoint()
        if ck.path is None:
            return Checkpoint.tracked_under(self.paths.published(), self.variant_name)
        return self.paths.raw() / ck.path

    def verify_checkpoint(self) -> Path:
        """The pinned checkpoint at :meth:`checkpoint_file`, proved present and unchanged."""
        ck = self._require_checkpoint()
        if ck.path is None:
            target = self.checkpoint_file()
            return ck._verified(target, target.parent)
        return ck.verify_under(self.paths.raw())

    def resolve_checkpoint(self) -> Path:
        """The pinned checkpoint wherever it is held: the tracked copy first, then a declared ``path`` under
        the raw root."""
        return self._require_checkpoint().resolve(self.name, self.paths.raw,
                                                  self.paths.published,
                                                  self.variant_name)

    def md(self, campaign: str) -> MdSettings:
        """The engine knobs of one named campaign; ``KeyError`` naming what is declared."""
        if not self.md_settings:
            raise KeyError(
                f"system {self.name!r} declares no engine settings, so there "
                f"is nothing to run a campaign with")
        if campaign not in self.md_settings:
            raise KeyError(
                f"system {self.name!r} declares no campaign {campaign!r}; it "
                f"declares {sorted(self.md_settings)}")
        return self.md_settings[campaign]

    def t_min(self, pressure_gpa: float) -> float | None:
        """Lowest admissible anchor temperature at this pressure, or None."""
        return _lookup(self.anchor_rules.T_min_by_pressure, pressure_gpa)

    def shape_t_min(self, pressure_gpa: float) -> float | None:
        """Lowest temperature in the kernel-shape set at this pressure, or None (a different table)."""
        return _lookup(self.anchor_rules.shape_T_min_by_pressure,
                       pressure_gpa)


def experiments_root() -> Path:
    """Where experiment folders live: ``AIPF_EXPERIMENTS``, else ``[paths] experiments`` in ``aipf.toml``,
    else the repository's ``experiments/`` (:func:`aipf.paths.experiments_root`)."""
    return _paths.experiments_root()


def available() -> list[str]:
    """Names of the experiment folders that define a system, sorted; empty when none can be located."""
    try:
        root = experiments_root()
    except RuntimeError:
        return []
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if (p / _SYSTEM_FILENAME).is_file())


def _looks_like_a_path(name_or_path: str) -> bool:
    """Whether the argument names a location (contains a separator) rather than a system."""
    separators = [os.sep] + ([os.altsep] if os.altsep else [])
    return any(sep in name_or_path for sep in separators)


def load(name_or_path: str) -> System:
    """Load a System by short name or by path to its folder."""
    folder = (Path(name_or_path) if _looks_like_a_path(name_or_path)
              else experiments_root() / name_or_path)
    module_path = folder / _SYSTEM_FILENAME
    if not module_path.is_file():
        names = available()
        known = (", ".join(names) if names else
                 f"none (set {_EXPERIMENTS_ENV}, or [paths] experiments in aipf.toml, "
                 f"to say where they live)")
        raise FileNotFoundError(
            f"no {_SYSTEM_FILENAME} at {module_path}. Available systems: "
            f"{known}"
        )
    spec = importlib.util.spec_from_file_location(
        f"aipf_experiment_{folder.name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    system = getattr(module, _SYSTEM_ATTR, None)
    if system is None:
        raise AttributeError(f"{module_path} defines no {_SYSTEM_ATTR} object")
    if not isinstance(system, System):
        raise TypeError(
            f"{module_path} defines {_SYSTEM_ATTR} as "
            f"{type(system).__name__}, not a System"
        )
    return system
