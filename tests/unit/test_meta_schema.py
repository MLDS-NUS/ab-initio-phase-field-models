import pytest

from aipf.data.meta import ENSEMBLES, SCHEMA_VERSION, supports_pressure_anchor, validate


def _valid() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "system": "hhe",
        "tag": "slab_xl0.00_xr0.90_T02000",
        "geometry": "slab",
        "ensemble": "NPT_z",
        "T_K": 2000.0,
        "P_GPa": 800.0,
        "composition": {"species": ["H", "He"], "x": {"He": 0.9}},
        "box": {"L": [16.0, 16.0, 64.0], "varying": ["z"]},
        "n_atoms": 3456,
        "dt_ps": 0.001,
        "dump_every_ps": 0.02,
        "n_frames": 2001,
        "engine": "mace-mliap",
        "potential": {"path": "/x/model.pt", "sha256": "0" * 64},
        "seed": 1,
        "status": "ok",
    }


def test_valid_metadata_has_no_problems():
    assert validate(_valid()) == []


def test_missing_ensemble_is_a_problem():
    d = _valid()
    del d["ensemble"]
    assert any("ensemble" in p for p in validate(d))


def test_unknown_ensemble_is_rejected():
    d = _valid()
    d["ensemble"] = "NPH"
    problems = validate(d)
    assert any("NPH" in p for p in problems)


def test_known_ensembles():
    assert ENSEMBLES == frozenset(
        {"NVE", "NVT", "NPT", "NPT_z", "langevin_overdamped"})


def test_varying_box_must_be_consistent_with_the_ensemble():
    d = _valid()
    d["ensemble"] = "NVT"          # NVT cannot have a breathing box
    assert any("varying" in p for p in validate(d))


def test_pressure_anchor_requires_a_controlled_pressure():
    npt = _valid()
    assert supports_pressure_anchor(npt) is True

    nvt = _valid()
    nvt["ensemble"] = "NVT"
    nvt["box"]["varying"] = []
    assert supports_pressure_anchor(nvt) is False

    lj = _valid()
    lj["ensemble"] = "langevin_overdamped"
    lj["box"]["varying"] = []
    assert supports_pressure_anchor(lj) is False


@pytest.mark.parametrize("key", [
    "schema_version", "system", "tag", "geometry", "ensemble", "T_K",
    "composition", "box", "n_atoms", "dt_ps", "dump_every_ps", "n_frames",
    "engine", "seed", "status",
])
def test_every_required_key_is_actually_required(key):
    """The loop is one rule, but thirteen keys rode on one test."""
    d = _valid()
    del d[key]
    assert any(key in p for p in validate(d)), (key, validate(d))


def test_schema_version_must_match():
    d = _valid()
    d["schema_version"] = SCHEMA_VERSION + 1
    assert any("schema_version" in p for p in validate(d))


def test_unknown_geometry_is_rejected():
    d = _valid()
    d["geometry"] = "torus"
    assert any("torus" in p for p in validate(d))


@pytest.mark.parametrize("key", ["n_frames", "dt_ps", "dump_every_ps", "T_K"])
@pytest.mark.parametrize("bad", [0, -1, "2001", None, True])
def test_numeric_fields_reject_non_positive_and_non_numeric(key, bad):
    """`validate` RETURNS problems; it must not raise on a bad type.

    A bulk scan over thousands of state points cannot die on one record, which
    is the same argument the module makes for supports_pressure_anchor.
    """
    d = _valid()
    d[key] = bad
    problems = validate(d)          # must not raise
    assert any(key in p for p in problems), (key, bad, problems)


def test_a_malformed_box_is_reported_not_raised():
    d = _valid()
    d["box"] = None
    assert any("box" in p for p in validate(d))


def test_pressure_anchor_needs_a_pressure_not_just_the_ensemble():
    """A pressure-controlled record with no pressure must not grant the anchor.

    P_GPa is not a required key, so such a record validates cleanly. Granting
    the anchor there is the silent meaningless computation this module exists
    to prevent.
    """
    d = _valid()
    d["P_GPa"] = None
    assert validate(d) == []
    assert supports_pressure_anchor(d) is False

    del d["P_GPa"]
    assert supports_pressure_anchor(d) is False
