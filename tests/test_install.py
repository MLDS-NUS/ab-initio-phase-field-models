"""What an install gives: the package imports, the command line answers quickly, the extras import.

``aipf --help`` is timed because it is the first thing a user runs and it must not import torch; the
optional dependency groups of ``pyproject.toml`` are imported where they are installed and skipped,
by name, where they are not.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import subprocess
import sys
import re

import pytest

from aipf.paths import repo_root

#: The ``aipf --help`` budget, in seconds, best of five, timed inside the process from the import of
#: the command line to its return, so interpreter start-up on a loaded node does not count.
HELP_BUDGET_S = 0.5

#: Each optional dependency's import name, where it is not the distribution's own.
IMPORT_NAME = {"torch-ema": "torch_ema", "gitpython": "git", "cuequivariance-torch": "cuequivariance_torch"}


def _extras() -> dict[str, list[str]]:
    """The optional dependency groups of ``pyproject.toml``, by distribution name."""
    tomllib = pytest.importorskip("tomllib", reason="reading pyproject.toml needs Python 3.11")
    with open(repo_root() / "pyproject.toml", "rb") as handle:
        groups = tomllib.load(handle)["project"]["optional-dependencies"]
    names = {}
    for group, requirements in groups.items():
        names[group] = [re.split(r"[<>=!~\[; ]", r, maxsplit=1)[0] for r in requirements]
    return names


def test_the_package_imports_and_has_a_version():
    import aipf
    assert isinstance(aipf.__version__, str) and aipf.__version__


def test_the_command_line_help_exits_zero():
    r = subprocess.run([sys.executable, "-m", "aipf.cli.main", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "aipf" in r.stdout


def test_the_command_line_help_is_quick_and_imports_no_torch():
    probe = ("import sys, time; t = time.perf_counter(); from aipf.cli.main import main\n"
             "try:\n    main(['--help'])\nexcept SystemExit:\n    pass\n"
             "print(time.perf_counter() - t, 'torch' in sys.modules, file=sys.stderr)")
    best = float("inf")
    for _ in range(5):
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        seconds, torch_loaded = r.stderr.split()[-2:]
        assert torch_loaded == "False", "aipf --help imported torch"
        best = min(best, float(seconds))
    assert best < HELP_BUDGET_S, f"aipf --help took {best:.2f} s, budget {HELP_BUDGET_S} s"


def test_main_returns_zero_in_process():
    from aipf.cli.main import main
    assert main(["--version"]) == 0
    assert main([]) == 0


def test_the_console_script_is_installed():
    """Running the module with -m exercises the __main__ guard, not the installed console script."""
    exe = shutil.which("aipf")
    if exe is None:
        pytest.skip("the aipf console script is not on PATH: the package is not installed")
    r = subprocess.run([exe, "--version"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()


@pytest.mark.parametrize("extra", ["md", "figures", "dev"])
def test_each_extra_imports_where_it_is_installed(extra):
    missing, imported = [], []
    for dist in _extras()[extra]:
        try:
            importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            missing.append(dist)
            continue
        importlib.import_module(IMPORT_NAME.get(dist, dist.replace("-", "_")))
        imported.append(dist)
    if not imported:
        pytest.skip(f"the [{extra}] extra is not installed (missing: {', '.join(missing)})")
