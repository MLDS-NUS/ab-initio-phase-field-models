"""The archive index's ensemble label: ``run.json`` first, then (for a bare geometry prefix) the
system's declared md deck of that geometry, then the table.

Every archived Fe-B and H/He cube ran ``fix npt ... iso``; the table's ``cube`` row said ``NVT``."""
import dataclasses

import pytest

from aipf.data.index import _classify
from aipf.system import load


def test_feb_cube_reads_npt_from_its_deck():
    feb = load("feb")
    assert feb.defaults["md"]["cube-npt"]["point"]["ensemble"] == "NPT"
    assert _classify("cube_x0.50_T1800_s1", feb) == ("cube", "NPT")


def test_hhe_cube_reads_npt_from_the_table():
    hhe = load("hhe")
    assert "cube" not in {deck.get("point", {}).get("geometry")
                          for deck in hhe.defaults["md"].values() if isinstance(deck, dict)}
    assert _classify("cube_x0.05_T02000", hhe) == ("cube", "NPT")
    assert _classify("slab_x0.10_T05000", hhe) == ("slab", "NPT_z")


def test_lj_keeps_its_protocol_prefixes():
    lj = load("lj")
    for tag, want in (("homogeneous_brownian_x0.5", ("cube", "langevin_overdamped")),
                      ("quench_MD_xA_0.50_T_1.30_seed_2", ("quench", "langevin_overdamped")),
                      ("vext_x0.5", ("vext", "NVT"))):
        assert _classify(tag, lj) == want


def test_a_deck_wins_over_the_table():
    feb = load("feb")
    deck = feb.defaults["md"]["cube-npt"]
    nvt = {**deck, "point": {**deck["point"], "ensemble": "NVT"}}
    only = dataclasses.replace(feb, defaults={**feb.defaults, "md": {"cube-nvt": nvt}})
    assert _classify("cube_x0.50_T1800_s1", only) == ("cube", "NVT")
    assert _classify("cube_x0.50_T1800_s1") == ("cube", "NPT")


def test_disagreeing_decks_are_refused():
    feb = load("feb")
    other = {**feb.defaults["md"]["cube-npt"],
             "point": {**feb.defaults["md"]["cube-npt"]["point"], "ensemble": "NVT"}}
    two = dataclasses.replace(feb, defaults={**feb.defaults,
                                             "md": {**feb.defaults["md"], "cube-nvt": other}})
    with pytest.raises(ValueError, match="different ensembles"):
        _classify("cube_x0.50_T1800_s1", two)


def test_train_and_diagnose_read_no_manifest_field():
    """The label cannot move a number: no module of training or diagnosis reads the index's
    ``manifest.json`` or ``meta.json``, or an ``"ensemble"`` key."""
    import re
    from pathlib import Path

    import aipf

    pattern = re.compile(r"manifest\.json|meta\.json|[\"']ensemble[\"']|supports_pressure_anchor")
    root = Path(aipf.__file__).parent
    hits = [f"{path.relative_to(root)}:{n}"
            for package in ("train", "diagnose")
            for path in sorted((root / package).rglob("*.py"))
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if pattern.search(line)]
    assert hits == []
