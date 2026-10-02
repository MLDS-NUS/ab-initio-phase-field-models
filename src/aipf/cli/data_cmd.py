"""``aipf data`` -- build and rebuild the organised data index."""
from __future__ import annotations

import argparse
import sys


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("data", help="organise the data index")
    sub = p.add_subparsers(dest="data_command", required=True)

    b = sub.add_parser("build", help="index a system's trajectories")
    b.add_argument("--system", required=True)
    b.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("rebuild", help="recreate the md tier of the symlink farm from its manifest")
    r.add_argument("--system", required=True)

    p.set_defaults(func=run)


def _raw_root_keys(system: str) -> str:
    """Where a raw root is declared: the variables and the ``aipf.toml`` key, in the order they win."""
    from aipf.paths import config_path, raw_env_name
    return (f"{raw_env_name(system)}, AIPF_RAW, or {system} under [paths.raw] in {config_path()}")


def run(args: argparse.Namespace) -> int:
    if args.data_command == "build":
        from aipf.data import index
        from aipf.paths import MissingLocation
        from aipf.system import load
        system = load(args.system)
        try:
            raw = system.paths.raw()
        except MissingLocation as refused:
            print(f"aipf data build: {refused}", file=sys.stderr)
            return 2
        if not raw.is_dir():
            print(f"aipf data build: the raw root of system {system.name!r} is {raw}, which is not "
                  f"a directory; correct {_raw_root_keys(system.name)}", file=sys.stderr)
            return 2
        try:
            manifest = index.build(system, dry_run=args.dry_run)
        except index.ForeignManifest as refused:
            print(f"aipf data build: {refused}", file=sys.stderr)
            return 2
        ok = [r for r in manifest["state_points"] if "skipped" not in r]
        skipped = [r for r in manifest["state_points"] if "skipped" in r]
        print(f"indexed {len(ok)} state points, skipped {len(skipped)}")
        for record in skipped:
            print(f"  SKIP {record['tag']}: {record['skipped']}")
        return 0

    if args.data_command == "rebuild":
        from aipf.data import index
        from aipf.system import load
        system = load(args.system)
        n = index.rebuild_from_manifest(system.paths.data_root() / "manifest.json")
        print(f"rebuilt {n} state points")
        return 0

    raise AssertionError(f"unhandled data command {args.data_command!r}")
