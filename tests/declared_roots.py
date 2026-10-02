"""What a test needs from this machine, and the skip that names it when the machine lacks it.

A test that needs more than a fresh checkout is marked ``env`` and calls one of three skips:

* :func:`raw_or_skip`: a path under a system's raw data root (the archived MD trajectories and
  run outputs), :func:`aipf.paths.raw_root`: ``AIPF_RAW_<SYSTEM>``, else ``AIPF_RAW``/<system>,
  else ``[paths.raw]`` in ``aipf.toml``;
* :func:`lammps_or_skip`: the site's LAMMPS binary (``AIPF_LAMMPS`` or ``[site] lammps``), with
  every style a deck needs;
* :func:`gpu_or_skip`: a CUDA device visible to torch.

A checkout may also come without the tracked published checkpoints
(``data/<system>/ckpt/published/[<variant>/]final.ckpt``); a test that reads one calls
:func:`published_or_skip` and skips, naming it, where it is absent.

:func:`raw` and :func:`site_fact` are the same lookups without the skip, for a ``skipif`` written
at import: undeclared, each gives a placeholder that exists nowhere and names its variable. No
absolute path is written down in ``tests/`` (``tests/unit/test_no_absolute_paths.py``); the
variables are listed in ``docs/guides/environment.md``.
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def raw(system: str, *parts: str) -> Path:
    """A path under the system's raw root, or a placeholder that exists nowhere when it is undeclared."""
    from aipf.paths import MissingLocation, raw_env_name
    from aipf.system import load
    try:
        root = load(system).paths.raw()
    except MissingLocation:
        return Path(f"<{raw_env_name(system)} unset>").joinpath(*parts)
    return root.joinpath(*parts)


def raw_or_skip(system: str, *parts: str) -> Path:
    """Inside a test: the path under the raw root, or a skip naming what is missing."""
    path = raw(system, *parts)
    if not path.exists():
        pytest.skip(f"no raw data at {path} (the {system} raw root: "
                    f"AIPF_RAW_{system.upper()}, AIPF_RAW or [paths.raw] in aipf.toml)")
    return path


def published_or_skip(system: str, variant: str | None = None) -> Path:
    """Inside a test: the tracked published checkpoint of ``system`` (or its ``variant``), or a skip
    saying that this checkout does not carry it."""
    from aipf.paths import published_checkpoint
    path = published_checkpoint(system, variant)
    if not path.is_file():
        name = f"{system}/{variant}" if variant else system
        pytest.skip(f"the published {name} checkpoint is not part of this checkout ({path})")
    return path


def site_fact(name: str) -> Path:
    """One :class:`aipf.site.Site` fact, or a placeholder that exists nowhere and names its variable."""
    from aipf.site import FACTS, Site
    value = getattr(Site.load(), name)
    return value if value is not None else Path(f"<{FACTS[name]} unset>")


#: The deck line of the ML-IAP route with the accelerated pair style.
MLIAP_KK = "pair_style mliap/kk unified model.pt 0"


def lammps_or_skip(*deck_lines: str) -> Path:
    """The site's LAMMPS binary when its help lists every style ``deck_lines`` name; else a skip naming
    the binary and the missing styles. The package declares one LAMMPS: a build without a style is a
    fact of this machine, not a failure."""
    from aipf.md.doctor import Verdict, deck_styles
    binary = site_fact("lammps")
    if not binary.is_file():
        pytest.skip(f"no LAMMPS binary at {binary} (AIPF_LAMMPS or [site] lammps in aipf.toml)")
    if deck_lines:
        found = deck_styles(binary, "\n".join(deck_lines) + "\n")
        if found.verdict is not Verdict.OK:
            pytest.skip(f"the declared LAMMPS lacks a style these tests need: {found.detail}")
    return binary


def gpu_or_skip(why: str = "") -> None:
    """Inside a test: return when torch sees a CUDA device, else a skip that says so (and ``why``)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible to torch" + (f": {why}" if why else ""))


@functools.lru_cache(maxsize=1)
def aipf_prefix() -> Path:
    """This package's environment: ``AIPF_ENV_PREFIX``, else the interpreter running the tests."""
    return Path(os.environ.get("AIPF_ENV_PREFIX") or sys.prefix)


def aipf_python() -> Path:
    return aipf_prefix() / "bin" / "python"
