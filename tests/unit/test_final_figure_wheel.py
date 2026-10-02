"""The coarsening figure names its repaint command with ``figures/common/wheel.py`` even when the PyPI
``wheel`` package is already imported in the process (a bare ``import wheel`` returned that one, and the
refusal message died with ``AttributeError: wheel_dir``); the notebook reads the paint switch through the
same ``box3d.paint_requested`` that ``draw.py`` calls."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

FIGURES = Path(__file__).resolve().parents[2] / "figures"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rerun_survives_the_pypi_wheel_in_sys_modules(monkeypatch):
    pytest.importorskip("matplotlib")
    # whatever ``wheel`` the process imported first: here a stand-in for the PyPI package, which has
    # no wheel_dir (the collision this guards against)
    pypi = types.ModuleType("wheel")
    monkeypatch.setitem(sys.modules, "wheel", pypi)
    monkeypatch.syspath_prepend(str(FIGURES / "common"))
    draw = _load(FIGURES / "sm_lj_coarsening" / "draw.py", "final_sm_lj_coarsening_draw")
    ours = _load(FIGURES / "common" / "wheel.py", "final_common_wheel")
    command = draw.rerun()
    assert f"PYTHONPATH={ours.wheel_dir()} " in command
    assert sys.modules["wheel"] is pypi  # the PyPI package is left where it was


def test_the_notebook_reads_the_switch_the_way_draw_py_does():
    nb = json.loads((FIGURES / "paper_figures.ipynb").read_text())
    source = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    start = source.index("def boxes_mode(")
    body = source[start:source.index("\ndef ", start + 1)]
    assert "box3d.paint_requested(os.environ)" in body
    assert '"1", "true", "yes"' not in body
