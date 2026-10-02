"""Draw one figure in a process of its own: ``python render.py <draw.py> <out dir> [<build>]``.

With ``<build>`` ("matplotlib X, freetype Y") the interpreter's matplotlib must be that build, checked
before anything is drawn (exit status 3 and both builds named otherwise). Prints one JSON line,
``{"build": ..., "files": [...]}``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

WRONG_BUILD = 3


def running_build() -> str:
    import matplotlib
    import matplotlib.ft2font as ft2font
    return f"matplotlib {matplotlib.__version__}, freetype {ft2font.__freetype_version__}"


def main(argv: list[str]) -> int:
    draw_py, out = Path(argv[1]), Path(argv[2])
    want = argv[3] if len(argv) > 3 else None
    import matplotlib
    matplotlib.use("Agg")
    have = running_build()
    if want is not None and " ".join(want.split()) != have:
        print(f"{draw_py.parent.name} is drawn with {want}; this process has {have} "
              f"(matplotlib from {Path(matplotlib.__file__).parent})", file=sys.stderr)
        return WRONG_BUILD
    spec = importlib.util.spec_from_file_location(f"figure_{draw_py.parent.name}", draw_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    files = mod.draw(out)
    print(json.dumps({"build": have, "files": [str(p) for p in files]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
