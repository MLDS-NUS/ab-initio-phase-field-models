"""The anchor rules, pinned against the CODE rather than its documentation.

Two independent figure sessions found that the sets the training code actually
reads differ from the sets the documented cut rules imply, for both two-density
systems. This file records which is which, so that "correcting" a value back to
its documentation fails rather than passes.
"""
from aipf.system import load


def test_hhe_anchor_cut_matches_its_own_documentation():
    """Hydrogen helium is the clean case, and it is worth pinning as such."""
    hhe = load("hhe")
    assert hhe.anchor_rules.T_min_by_pressure == {
        200: 6100, 400: 7800, 600: 8800, 800: 8973}


def test_feb_documented_rule_and_actual_rule_disagree():
    """Fe-B is the mismatch, and the code is authoritative.

    Config header at :145-146 claims {0: 1799}. The key at :948 is {}, with the
    comment "no T floor (gate decides)". Anchors are gated by an eligibility
    column the config names at :939 instead of by a temperature cut.
    """
    feb = load("feb")
    assert feb.anchor_rules.T_min_by_pressure == {}, (
        "Fe-B has no temperature cut. If this fails because someone restored "
        "{0: 1799} from the config header, the header is wrong and :948 is right.")


def test_lj_has_no_temperature_cut():
    """Lennard-Jones has one bulk anchor and no finite-k shell term."""
    lj = load("lj")
    assert lj.anchor_rules.T_min_by_pressure == {}
