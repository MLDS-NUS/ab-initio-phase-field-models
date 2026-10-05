"""Every location the package reads or writes, resolved in one place.

Each value is looked up at call time in this order: the environment variable ``AIPF_<KEY>``, then
``aipf.toml`` at the repository root, then a default. A relative path in ``aipf.toml`` is taken
relative to the repository root; an empty value counts as unset; ``~`` is not expanded.

=====================  ========================  ===========================  =========================
location               environment               ``aipf.toml``                default
=====================  ========================  ===========================  =========================
data farm              ``AIPF_DATA``             ``[paths] data``             ``<repo>/data``
raw root of a system   ``AIPF_RAW_<SYSTEM>``,    ``[paths.raw] <system>``,    none: refused, naming both
                       ``AIPF_RAW``/<system>     ``[paths] raw_parent``/...
experiment folders     ``AIPF_EXPERIMENTS``      ``[paths] experiments``      ``<repo>/experiments``
=====================  ========================  ===========================  =========================

Every environment variable, ``AIPF_RAW`` included, wins over every ``aipf.toml`` entry. The raw root (a system's
MD archives) is only read; the farm ``<data>/<system>/{md,modes,ckpt,diagnose}`` is written; the published
checkpoints and the training sample ``data/<system>/sample`` are tracked files ``AIPF_DATA`` does not move. Site facts resolve the same way in :mod:`aipf.site`."""
from __future__ import annotations

import functools
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

#: Marker that identifies the repository root.
_REPO_MARKER = "pyproject.toml"

#: The configuration file at the repository root (gitignored; ``aipf.toml.example`` is tracked).
CONFIG_FILE = "aipf.toml"

_DATA_DIRNAME = "data"
_EXPERIMENTS_DIRNAME = "experiments"

#: Raw-root subdirectory every system's coarse-grained field trees sit under.
FIELDS_DIRNAME = "fields"

CKPT_DIRNAME = "ckpt"

#: The published checkpoints' directory and the one CLI keyword for them.
PUBLISHED_DIRNAME = "published"
TRACKED_FILE_NAME = "final.ckpt"

class ConfigError(RuntimeError):
    """``aipf.toml`` exists and is not valid TOML; the message names the file."""


class MissingLocation(RuntimeError):
    """A location nothing declares; the message names the variable and the ``aipf.toml`` key."""

    def __init__(self, key: str, env_name: str, toml_key: str, toml_path: Path, detail: str = ""):
        self.key, self.env_name, self.toml_key, self.toml_path = key, env_name, toml_key, toml_path
        super().__init__(detail or f"no {key}: set {env_name}, or {toml_key} in {toml_path}")


def repo_root() -> Path:
    """The repository root, searched upwards from this module for ``pyproject.toml``; raises if absent."""
    for directory in Path(__file__).resolve().parents:
        if (directory / _REPO_MARKER).is_file():
            return directory
    raise RuntimeError(
        f"cannot locate the repository root: no {_REPO_MARKER} above "
        f"{Path(__file__).resolve()}. Set AIPF_DATA to say where the organised "
        f"index should live.")


def _repo_root() -> Path:
    return repo_root()


def config_path() -> Path:
    """Where ``aipf.toml`` is read from: the repository root."""
    return repo_root() / CONFIG_FILE


@functools.lru_cache(maxsize=1)
def config() -> dict:
    """The parsed ``aipf.toml`` at the repository root, ``{}`` when absent (cached; ``config.cache_clear()``)."""
    try:
        p = config_path()
    except RuntimeError:
        return {}
    if not p.is_file():
        return {}
    try:
        return tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p} is not valid TOML: {exc}") from exc


def _env(name: str) -> str | None:
    return os.environ.get(name) or None


def _toml(table: str, key: str) -> str | None:
    node = config()
    for part in table.split("."):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    value = node.get(key) if isinstance(node, dict) else None
    # a table under that key (``[site.mace_potential]``, one entry per system) is not this value
    return str(value) if value not in (None, "") and not isinstance(value, dict) else None


def _from_toml(value: str) -> Path:
    """A path written in ``aipf.toml``, a relative one taken from the repository root."""
    p = Path(value)
    return p if p.is_absolute() else repo_root() / p


def _lookup(env: str, table: str, key: str) -> Path | None:
    """``env`` if set, else ``key`` of ``[table]`` in ``aipf.toml``, else ``None``."""
    value = _env(env)
    if value:
        return Path(value)
    value = _toml(table, key)
    return _from_toml(value) if value else None


def _lookup_value(env: str, table: str, key: str) -> str | None:
    """``env`` if set, else ``key`` of ``[table]`` in ``aipf.toml``, as text (not a path), else ``None``."""
    return _env(env) or _toml(table, key)


def _where_config() -> Path:
    try:
        return config_path()
    except RuntimeError:
        return Path(CONFIG_FILE)


def data_root() -> Path:
    """The data farm: ``AIPF_DATA``, else ``[paths] data``, else ``<repo>/data``."""
    found = _lookup("AIPF_DATA", "paths", "data")
    return found if found is not None else repo_root() / _DATA_DIRNAME


def farm(system: str, tier: str) -> Path:
    """One tier (``md``, ``modes``, ``ckpt``, ``diagnose``) of a system's data farm."""
    return data_root() / system / tier


def raw_env_name(system: str) -> str:
    """The variable that names one system's raw root."""
    return f"AIPF_RAW_{system.upper()}"


def _declared_raw_root(system: str) -> Path | None:
    """The raw root the environment or ``aipf.toml`` declares, environment first; ``None`` if neither does."""
    specific = _env(raw_env_name(system))
    if specific:
        return Path(specific)
    parent = _env("AIPF_RAW")
    if parent:
        return Path(parent) / system
    value = _toml("paths.raw", system)
    if value:
        return _from_toml(value)
    value = _toml("paths", "raw_parent")
    if value:
        return _from_toml(value) / system
    return None


def _missing_raw(system: str) -> MissingLocation:
    where, env = _where_config(), raw_env_name(system)
    return MissingLocation(
        f"raw data root for system {system!r}", env, f"[paths.raw] {system}", where,
        f"no raw data root for system {system!r}: set {env} (or AIPF_RAW, a parent holding one "
        f"directory per system), or put {system} = \"/path\" under [paths.raw] in {where}")


def raw_root(system: str) -> Path:
    """Where a system's MD archives live (read only); raises :class:`MissingLocation` when undeclared."""
    found = _declared_raw_root(system)
    if found is None:
        raise _missing_raw(system)
    return found


def published_dir(system: str) -> Path:
    """The tracked published-checkpoint directory of a system, in the checkout."""
    return repo_root() / _DATA_DIRNAME / system / CKPT_DIRNAME / PUBLISHED_DIRNAME


def published_checkpoint(system: str, variant: str | None = None) -> Path:
    """The tracked published checkpoint, ``final.ckpt`` in ``published_dir(system)`` or its ``variant`` subdirectory."""
    base = published_dir(system)
    return (base / variant if variant else base) / TRACKED_FILE_NAME


#: The directory of a system's bundled training sample, beside ``ckpt/`` under ``data/<system>/``.
SAMPLE_DIRNAME = "sample"


def sample_dir(system: str) -> Path:
    """A system's bundled training sample, tracked at ``data/<system>/sample`` in the checkout; like the
    published checkpoints, ``AIPF_DATA`` does not move it. ``aipf train --source sample`` reads it."""
    return repo_root() / _DATA_DIRNAME / system / SAMPLE_DIRNAME


#: The subdirectory of a system's experiment folder that holds its tracked measured tables.
TABLES_DIRNAME = "anchors"


def tracked_tables(system: str) -> Path:
    """A system's tracked measured tables, ``<experiments>/<system>/anchors``; a declared relative table
    path is looked for here before the raw root."""
    return experiments_root() / system / TABLES_DIRNAME


#: A system's tracked equation-of-state tables, under its experiment folder.
EOS_DIRNAME = "eos"


def tracked_eos(system: str) -> Path:
    """A system's tracked equation-of-state tables, ``<experiments>/<system>/eos``; a declared relative
    manifold path is looked for here before the raw root (``aipf diagnose``)."""
    return experiments_root() / system / EOS_DIRNAME


def experiments_root() -> Path:
    """Where experiment folders live: ``AIPF_EXPERIMENTS``, else ``[paths] experiments``, else ``<repo>/experiments``."""
    try:
        found = _lookup("AIPF_EXPERIMENTS", "paths", "experiments")
        return found if found is not None else repo_root() / _EXPERIMENTS_DIRNAME
    except ConfigError:
        raise
    except RuntimeError as exc:
        raise RuntimeError(
            f"cannot locate the {_EXPERIMENTS_DIRNAME} directory: there is no "
            f"repository root above this package. Set AIPF_EXPERIMENTS to "
            f"the directory that holds the experiment folders.") from exc


@dataclass(frozen=True)
class Paths:
    """Where one system's data lives; ``raw_default=None`` means the environment or ``aipf.toml`` must supply it."""

    system: str
    raw_default: str | None = None

    def raw(self) -> Path:
        """The raw root: declared (environment, then ``aipf.toml``), else ``raw_default``, else refused."""
        found = _declared_raw_root(self.system)
        if found is not None:
            return found
        if self.raw_default:
            return Path(self.raw_default)
        raise _missing_raw(self.system)

    def data_root(self) -> Path:
        """This system's farm directory, ``data_root() / system``."""
        return data_root() / self.system

    def published(self) -> Path:
        """The published checkpoints tracked in the checkout's data directory; ``AIPF_DATA`` never moves them."""
        return published_dir(self.system)

    def sample(self) -> Path:
        """The bundled training sample tracked in the checkout's data directory (:func:`sample_dir`)."""
        return sample_dir(self.system)

    def sub(self, *parts: str) -> Path:
        """A path under the raw root."""
        return self.raw().joinpath(*parts)


@dataclass(frozen=True)
class SamplePaths(Paths):
    """A system's locations with its raw root replaced by its bundled training sample (:func:`sample_dir`).

    The sample holds every file the declared training reads from the raw root, at the same relative path
    (the mode runs under ``modes/``), so a training run that reads through this object needs no raw root."""

    def raw(self) -> Path:
        """The bundled sample, whatever the environment or ``aipf.toml`` declares as the raw root."""
        return sample_dir(self.system)
