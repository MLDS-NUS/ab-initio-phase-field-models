"""The ``phase_diagram`` stage runs one binodal route, ``convex_hull``; any other is refused by name.

``thermo.BINODAL_ROUTES`` also lists ``mu_roots``, but the stage passes convex-hull keywords, so a
declared ``mu_roots`` used to die with a ``TypeError`` deep in the temperature loop."""
import dataclasses

import pytest

from aipf.diagnose.run import PHASE_DIAGRAM_BINODAL_ROUTES, check_declared
from aipf.diagnose.run import run as diagnose
from aipf.system import load


def test_only_convex_hull_is_admitted():
    assert PHASE_DIAGRAM_BINODAL_ROUTES == ("convex_hull",)
    check_declared({"binodal_route": "convex_hull"}, ("phase_diagram",))
    check_declared({"binodal_route": "mu_roots"}, ("stability_map",))  # not asked: not checked
    with pytest.raises(ValueError, match="binodal_route='mu_roots' is not a route the "
                                         "phase_diagram stage runs"):
        check_declared({"binodal_route": "mu_roots"}, ("phase_diagram",))


def test_the_shipped_declarations_pass():
    for name in ("feb", "hhe"):
        check_declared(load(name).defaults["diagnose"], ("phase_diagram",))


def test_run_refuses_before_writing(tmp_path):
    system = load("feb")
    declared = {**system.defaults["diagnose"], "binodal_route": "mu_roots"}
    with pytest.raises(ValueError, match="mu_roots"):
        diagnose(system, system.checkpoint, stages=("phase_diagram",), out=tmp_path,
                 pressures=(0.0,), T_grid=[1500.0, 1600.0], **declared)
    assert not any(tmp_path.iterdir())


def test_cli_exits_2(tmp_path, monkeypatch, capsys):
    import aipf.system as system_mod
    from aipf.cli.main import main

    feb = load("feb")
    defaults = {**feb.defaults,
                "diagnose": {**feb.defaults["diagnose"], "binodal_route": "mu_roots"}}
    monkeypatch.setattr(system_mod, "load",
                        lambda name: dataclasses.replace(feb, defaults=defaults))
    code = main(["diagnose", "--system", "feb", "--ckpt", "published", "--stage",
                 "phase_diagram", "--pressure", "0", "--T-grid", "1500,1600,100",
                 "--out", str(tmp_path)])
    assert code == 2
    assert "is not a route the phase_diagram stage runs" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
