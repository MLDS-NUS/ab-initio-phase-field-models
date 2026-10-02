"""Every script-drawn figure of the paper, redrawn from its committed data and
compared with the paper's file.

Each ``figures/<fig>/draw.py`` reads only ``figures/<fig>/figdata/`` and
exposes ``draw(out) -> list[Path]`` and ``REFERENCE``, which maps every file
name it writes to the paper's copy under ``figures/reference/``. Both are
rasterised at 200 dpi (PDFs through PyMuPDF, PNGs read directly, alpha
included) and no pixel may differ, except for a file listed in the figure's
``BOUND_PX``, whose ``CAUSE`` says why it cannot match exactly.

Text is laid out and rasterised by freetype, so a matplotlib build other than
the one a figure was drawn with moves glyphs by a pixel here and there. The
build is ``figures/common/BUILD.txt`` unless the figure's ``draw.py`` names
its own ``BUILD``. Each figure is drawn in a process of its own
(``figures/common/render.py``); a figure drawn with the pip build of
matplotlib (``figures/common/wheel.py``) gets that build first on its
``PYTHONPATH``, and is skipped, naming the command that installs it, when it
is not installed. A process whose matplotlib is not the figure's build fails and
names both.

A file in the figure's ``OPTIONAL`` (name -> the site fact it needs) may be
left out by ``draw``; it is a test case of its own, compared when drawn and
skipped, naming the fact, when not.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FIGURES = Path(__file__).resolve().parents[1] / "figures"
REFERENCE = FIGURES / "reference"
DRAWS = sorted(FIGURES.glob("*/draw.py"))
COMMON = FIGURES / "common"
DPI = 200

_BUILD = re.compile(r"matplotlib\s+(\S+),\s*freetype\s+(\S+)")


def parse_build(text: str) -> tuple[str, str]:
    """("3.10.8", "2.14.1") from "matplotlib 3.10.8, freetype 2.14.1"."""
    m = _BUILD.fullmatch(text.strip())
    if not m:
        raise ValueError(f"not a build line: {text!r}")
    return m.group(1), m.group(2)


def running_build() -> str:
    import matplotlib
    import matplotlib.ft2font as ft2font
    return f"matplotlib {matplotlib.__version__}, freetype {ft2font.__freetype_version__}"


def wanted_build(mod) -> str:
    return getattr(mod, "BUILD", None) or (FIGURES / "common" / "BUILD.txt").read_text().strip()


def load(draw_py: Path, name: str | None = None):
    spec = importlib.util.spec_from_file_location(name or f"figure_{draw_py.parent.name}", draw_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def child_env(fig: str, want: str) -> dict:
    """The drawing process's environment: the pip build first on PYTHONPATH when the figure's
    build is the pip one, and never otherwise. AIPF_FIGURES_RENDER_3D is dropped, so the test
    draws from the stored layers and never repaints or rewrites them."""
    wheel = load(COMMON / "wheel.py", "figures_common_wheel")
    target = wheel.wheel_dir()
    env = dict(os.environ)
    env.pop("AIPF_FIGURES_RENDER_3D", None)
    path = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
            if p and Path(p).resolve() != target.resolve()]
    if parse_build(want) == parse_build(wheel.BUILD):
        if not wheel.present(target):
            pytest.skip(f"{fig} is drawn with {want}, the pip build of matplotlib, which is not "
                        f"installed at {target}. Install it with "
                        f"{sys.executable} {COMMON / 'wheel.py'} "
                        f"(or by hand: {wheel.install_command(target)})")
        path.insert(0, str(target))
    env["PYTHONPATH"] = os.pathsep.join(path)
    env["MPLBACKEND"] = "Agg"
    return env


def raster(path: Path) -> list[np.ndarray]:
    """RGBA pages of a PNG or a PDF, as int arrays."""
    if path.suffix == ".png":
        from PIL import Image
        return [np.asarray(Image.open(path).convert("RGBA")).astype(int)]
    import fitz
    pages = []
    for page in fitz.open(path):
        pm = page.get_pixmap(dpi=DPI, alpha=True)
        pages.append(np.frombuffer(pm.samples, np.uint8)
                     .reshape(pm.h, pm.w, pm.n).astype(int))
    return pages


def differing_pixels(a: Path, b: Path) -> int:
    """Pixels that differ in any channel; a page or shape mismatch is a failure."""
    pa, pb = raster(a), raster(b)
    assert len(pa) == len(pb), f"{a.name}: {len(pa)} pages, the paper's has {len(pb)}"
    n = 0
    for x, y in zip(pa, pb):
        assert x.shape == y.shape, f"{a.name}: {x.shape} against the paper's {y.shape}"
        n += int((np.abs(x - y).max(-1) > 0).sum())
    return n


def test_build_lines_parse():
    assert parse_build("matplotlib 3.10.8, freetype 2.14.1") == ("3.10.8", "2.14.1")
    assert parse_build(running_build())
    with pytest.raises(ValueError):
        parse_build("matplotlib 3.10.8")


def test_the_pip_build_is_pinned_by_hash(tmp_path):
    wheel = load(COMMON / "wheel.py", "figures_common_wheel")
    assert parse_build(wheel.BUILD) == ("3.10.8", "2.6.1")
    cmd = wheel.install_command(tmp_path / "w", python="python")
    assert f"--hash=sha256:{wheel.SHA256}" in cmd and "--require-hashes" in cmd
    assert f"--target {tmp_path / 'w'}" in cmd and "--no-deps" in cmd
    assert not wheel.present(tmp_path / "w")


@pytest.mark.parametrize("draw_py", DRAWS, ids=lambda p: p.parent.name)
def test_draw_contract(draw_py):
    """One-line docstring, draw(), REFERENCE onto existing files, and a CAUSE
    for every bound."""
    mod = load(draw_py)
    doc = (mod.__doc__ or "").strip()
    assert doc and "\n" not in doc, "draw.py needs a one-line docstring naming the figure"
    assert callable(getattr(mod, "draw", None))
    assert mod.REFERENCE, "REFERENCE maps nothing"
    missing = [r for r in mod.REFERENCE.values() if not (REFERENCE / r).is_file()]
    assert not missing, f"not in figures/reference/: {missing}"
    bound = getattr(mod, "BOUND_PX", {})
    cause = getattr(mod, "CAUSE", {})
    assert set(bound) <= set(mod.REFERENCE)
    assert all(cause.get(name) for name in bound), "every bounded file states its cause"
    optional = getattr(mod, "OPTIONAL", {})
    assert set(optional) <= set(mod.REFERENCE) and all(optional.values())
    parse_build(wanted_build(mod))
    assert (draw_py.parent / "figdata").is_dir()
    assert sorted(p.name for p in draw_py.parent.iterdir()
                  if p.name != "__pycache__") == ["draw.py", "figdata"]


_RENDERED: dict = {}


def render(draw_py: Path, tmp_path_factory):
    """``(module, {name: path})`` of one figure drawn in its own process, once per session."""
    if draw_py in _RENDERED:
        return _RENDERED[draw_py]
    fig = draw_py.parent.name
    mod = load(draw_py)
    if any(r.endswith(".pdf") for r in mod.REFERENCE.values()):
        pytest.importorskip("fitz", reason="PyMuPDF rasterises the PDFs")
    want = "matplotlib {}, freetype {}".format(*parse_build(wanted_build(mod)))
    out = tmp_path_factory.mktemp(fig)
    run = subprocess.run([sys.executable, str(COMMON / "render.py"), str(draw_py),
                          str(out), want], env=child_env(fig, want),
                         capture_output=True, text=True)
    if run.returncode == 3:
        pytest.fail(run.stderr.strip(), pytrace=False)
    assert run.returncode == 0, f"{fig}: drawing failed\n{run.stderr[-3000:]}"
    outs = [Path(p) for p in json.loads(run.stdout.strip().splitlines()[-1])["files"]]
    names = {p.name for p in outs}
    required = set(mod.REFERENCE) - set(getattr(mod, "OPTIONAL", {}))
    assert required <= names <= set(mod.REFERENCE), \
        f"drawn {sorted(names)}, the paper's {sorted(mod.REFERENCE)}"
    _RENDERED[draw_py] = (mod, {p.name: p for p in outs})
    return _RENDERED[draw_py]


def compare(mod, path: Path, record_property) -> None:
    """``path`` against the paper's copy, within the file's bound."""
    n = differing_pixels(path, REFERENCE / mod.REFERENCE[path.name])
    allowed = getattr(mod, "BOUND_PX", {}).get(path.name, 0)
    record_property(path.name, n)
    print(f"{path.name}: {n} px differ" + (f" (bound {allowed}: {mod.CAUSE[path.name]})"
                                           if allowed else ""))
    assert n <= allowed, f"{path.name}: {n} px differ, {allowed} allowed"


@pytest.mark.slow
@pytest.mark.parametrize("draw_py", DRAWS, ids=lambda p: p.parent.name)
def test_figure_matches_the_paper(draw_py, tmp_path_factory, record_property):
    """Every file the figure always draws; its ``OPTIONAL`` files are cases of their own."""
    mod, outs = render(draw_py, tmp_path_factory)
    over = []
    for name in sorted(set(mod.REFERENCE) - set(getattr(mod, "OPTIONAL", {}))):
        try:
            compare(mod, outs[name], record_property)
        except AssertionError as exc:
            over.append(str(exc))
    assert not over, "; ".join(over)


OPTIONAL_CASES = [(p, name) for p in DRAWS for name in sorted(getattr(load(p), "OPTIONAL", {}))]


@pytest.mark.slow
@pytest.mark.parametrize("draw_py,name", OPTIONAL_CASES,
                         ids=[f"{p.parent.name}-{n}" for p, n in OPTIONAL_CASES])
def test_optional_file_matches_the_paper(draw_py, name, tmp_path_factory, record_property):
    """Compared when the site fact it needs is declared, a visible skip naming the fact otherwise."""
    mod, outs = render(draw_py, tmp_path_factory)
    if name not in outs:
        pytest.skip(f"{name}: {mod.OPTIONAL[name]}")
    compare(mod, outs[name], record_property)


@pytest.mark.slow
@pytest.mark.parametrize("draw_py", sorted({p for p, _ in OPTIONAL_CASES}),
                         ids=lambda p: p.parent.name)
def test_optional_files_are_left_out_without_their_site_fact(draw_py, tmp_path, monkeypatch):
    """With no site fact declared, ``draw`` writes every file but its ``OPTIONAL`` ones."""
    import aipf.site
    monkeypatch.setattr(aipf.site.Site, "load", classmethod(lambda cls: cls()))
    mod = load(draw_py)
    names = {p.name for p in mod.draw(tmp_path)}
    assert names == set(mod.REFERENCE) - set(mod.OPTIONAL)
    assert not any((tmp_path / name).exists() for name in mod.OPTIONAL)
