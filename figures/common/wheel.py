"""The pip build of matplotlib 3.10.8 (freetype 2.6.1), which some of the paper's figures were drawn with.

The pip wheel bundles its own freetype, 2.6.1, and the conda build links 2.14.1; the two place and
rasterise glyphs differently, so a figure is redrawn pixel for pixel only under its own build
(``BUILD`` in its ``draw.py``, else ``figures/common/BUILD.txt``). The pip build is kept apart from
the environment, in the data farm (``tools/matplotlib-3.10.8-wheel``), and goes first on
``PYTHONPATH`` for the figures that need it.

``python figures/common/wheel.py`` installs it (network needed). The hash pins the CPython 3.12,
x86-64 Linux wheel; another interpreter or platform is refused by pip's hash check.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

VERSION = "3.10.8"
FREETYPE = "2.6.1"
BUILD = f"matplotlib {VERSION}, freetype {FREETYPE}"
SHA256 = "3ab4aabc72de4ff77b3ec33a6d78a68227bf1123465887f9905ba79184a1cc04"
REQUIREMENT = f"matplotlib=={VERSION} --hash=sha256:{SHA256}"


def wheel_dir() -> Path:
    """Where the pip build lives: ``<data farm>/tools/matplotlib-3.10.8-wheel``."""
    from aipf.paths import data_root
    return data_root() / "tools" / f"matplotlib-{VERSION}-wheel"


def present(target: Path | None = None) -> bool:
    return ((target or wheel_dir()) / "matplotlib" / "__init__.py").is_file()


def install_command(target: Path | None = None, python: str = sys.executable,
                    requirements: str = "matplotlib-wheel.txt") -> str:
    """The two shell lines that install the pip build by hash into ``target``."""
    target = target or wheel_dir()
    return (f"printf '{REQUIREMENT}\\n' > {requirements}\n"
            f"{python} -m pip install --no-deps --only-binary :all: --require-hashes "
            f"--target {target} -r {requirements}")


def ensure(target: Path | None = None, python: str = sys.executable) -> Path:
    """The pip build's directory, installed first if it is not there (pip, network)."""
    target = target or wheel_dir()
    if present(target):
        return target
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "requirements.txt"
        req.write_text(REQUIREMENT + "\n")
        subprocess.run([python, "-m", "pip", "install", "--no-deps", "--only-binary", ":all:",
                        "--require-hashes", "--target", str(target), "-r", str(req)],
                       check=True)
    if not present(target):
        raise RuntimeError(f"pip finished but {target} holds no matplotlib")
    return target


if __name__ == "__main__":
    print(ensure(Path(sys.argv[1]) if len(sys.argv) > 1 else None))
