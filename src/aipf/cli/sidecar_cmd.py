"""``aipf modes-sidecar`` -- write the memory-mapped sidecar of a training source's mode archives.

The runs are a training source's: SUBDIR relative to the system's declared source root, and a glob over
its run directories. Every archive it selects gains ``rho_k.npy`` and its record ``rho_k.json`` beside
it, checked bit for bit; a valid pair already there is left as it is, so the command can be repeated,
and run as a batch array over disjoint globs."""
from __future__ import annotations

import argparse
import sys


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "modes-sidecar", help="write the memory-mapped copy of a source's mode archives",
        description="Write rho_k.npy, the memory-mapped copy of the amplitudes that training maps "
                    "in place of reading them into memory, and its record rho_k.json beside every "
                    "modes.npz a glob selects, and check it against the archive bit for bit. A "
                    "valid sidecar already there is skipped; a stale one is rewritten; a run that "
                    "cannot be converted is reported and the others go on. modes.npz is never "
                    "changed.")
    p.add_argument("--system", required=True,
                   help="the system whose archives these are, by name or by folder")
    p.add_argument("--tree", required=True,
                   help="the subdirectory of the system's declared source root "
                        "(defaults['training']['source_root']) the runs are in, as a --source "
                        "SUBDIR of aipf train")
    p.add_argument("--pattern", required=True,
                   help="the glob over the run directories under SUBDIR, as a --source PATTERN")
    p.add_argument("--verify", action="store_true",
                   help="also compare the frames of every existing sidecar with its archive "
                        "(which reads the archive) and rewrite one that differs; without it an "
                        "existing sidecar is checked by shape, dtype, order and its record of the "
                        "archive array's CRC-32 and size")
    p.add_argument("--remove-partials", action="store_true", dest="remove_partials",
                   help="delete the temporary files of killed writers whose name records the "
                        "machine this command runs on and a process that no longer exists; "
                        "without it they are only listed")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from aipf.pipeline.modes import MMAP_FILE
    from aipf.system import load
    from aipf.train.fit import _archive_keys, source_root
    from aipf.train.mmap_sidecar import FAILED, REPLACED, SKIPPED, WRITTEN, convert_runs

    if Path(args.tree).is_absolute():
        print(f"aipf modes-sidecar: --tree {args.tree!r} is absolute; it is relative to the "
              f"system's declared source root (defaults['training']['source_root'])",
              file=sys.stderr)
        return 2
    system = load(args.system)
    try:
        root = source_root(system) / args.tree
        keys = _archive_keys(system).replace(mmap_file=MMAP_FILE)
    except (KeyError, ValueError) as refused:
        print(f"aipf modes-sidecar: {refused.args[0]}", file=sys.stderr)
        return 2
    counts = {WRITTEN: 0, REPLACED: 0, SKIPPED: 0, FAILED: 0}
    left, removed = [], []
    try:
        for done in convert_runs(root, args.pattern, keys=keys, verify=args.verify,
                                 remove_partials=args.remove_partials):
            counts[done.status] += 1
            left += done.partials
            removed += done.removed
            where = done.directory.relative_to(root)
            print(f"{done.status} {where}" + (f": {done.reason}" if done.reason else ""),
                  flush=True)
    except FileNotFoundError as refused:
        print(f"aipf modes-sidecar: {refused}", file=sys.stderr)
        return 2
    for path in removed:
        print(f"removed partial {path.relative_to(root)}")
    for path in left:
        print(f"partial {path.relative_to(root)}")
    print(f"{counts[WRITTEN]} written, {counts[REPLACED]} replaced, {counts[SKIPPED]} skipped, "
          f"{counts[FAILED]} failed under {root}; {len(left)} partial file(s) left, "
          f"{len(removed)} removed")
    return 1 if counts[FAILED] else 0
