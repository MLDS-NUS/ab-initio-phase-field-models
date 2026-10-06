"""Write the golden outputs that ``test_golden_outputs.py`` compares against, bit for bit.

    python tests/golden/make_goldens.py --write [--groups spectral imex ...]

Run it only on source identical to the published baseline: a golden output records what the package
computes, so one written from changed source records the change and every later comparison
passes against it. The script refuses unless ``src/`` has no uncommitted change, and records in each
file's ``meta`` the commit it ran on, the git tree of ``src/`` at that commit, and the torch and numpy
versions. Without ``--write`` it refuses and writes nothing.

The outputs are float32 on the CPU, one thread, and depend on the torch build and the instruction set
torch dispatches to (``torch.backends.cpu.get_cpu_capability()``), so they live in one directory per
pair, ``tests/golden/<torch version>-<capability>/`` (:func:`cases.fixture_dir`); on another pair the
tests skip, naming this command. Each group is one file, ``<group>.pt``; every case is computed twice
and must agree with itself bit for bit before it is written. ``models.pt`` is written first: every
other case runs its models with those parameters.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from golden import cases  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def _meta(group: str) -> dict:
    return {"group": group, "commit": _git("rev-parse", "HEAD"),
            "src_tree": _git("rev-parse", "HEAD:src"), "torch": str(torch.__version__),
            "numpy": np.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "generator": "tests/golden/make_goldens.py"}


def _twice(group: str, name: str, fn) -> dict:
    """``fn()``, computed twice and refused unless the two agree bit for bit."""
    first, second = fn(), fn()
    problem = cases.compare(first, second)
    if problem is not None:
        raise SystemExit(f"{group}/{name} is not deterministic on this machine: {problem}")
    return first


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="write the fixture files (refused without it)")
    parser.add_argument("--groups", nargs="*", default=None, metavar="GROUP",
                        help=f"only these groups, of {sorted(cases.GROUPS)}")
    args = parser.parse_args(argv)
    if not args.write:
        parser.error("refusing to write golden outputs without --write; read this script's "
                     "docstring first")
    dirty = _git("status", "--porcelain", "--", "src")
    if dirty:
        raise SystemExit(f"src/ has uncommitted changes, so it is not the source a golden output "
                         f"may record:\n{dirty}")
    groups = list(cases.GROUPS) if args.groups is None else args.groups
    unknown = sorted(set(groups) - set(cases.GROUPS))
    if unknown:
        parser.error(f"no such group {unknown}; have {sorted(cases.GROUPS)}")

    torch.set_num_threads(1)
    out_dir = cases.fixture_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # The models first: every other case loads its parameters from what is stored.
    models_path = out_dir / "models.pt"
    if "models" in groups or not models_path.is_file():
        table = {name: _twice("models", name, fn) for name, fn in cases.GROUPS["models"].items()}
        torch.save({"meta": _meta("models"), "cases": table}, models_path)
        print(f"wrote {models_path.relative_to(REPO)} ({len(table)} cases)")
    cases.use_model_states(cases.model_states(torch.load(models_path, weights_only=True)))

    for group in groups:
        if group == "models":
            continue
        table = {name: _twice(group, name, fn) for name, fn in cases.GROUPS[group].items()}
        path = out_dir / f"{group}.pt"
        torch.save({"meta": _meta(group), "cases": table}, path)
        print(f"wrote {path.relative_to(REPO)} ({len(table)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
