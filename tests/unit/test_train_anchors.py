"""``aipf.train.anchors``: the measured tables, on toy files.

Nothing here names a real system, uses one of
its numbers, or reads one of its files: the three tables below are written
by this module with the shapes and the row filters the real ones carry,
and their COLUMN NAMES are deliberately not the real ones. That last point
is the test: the loader holds no column name of its own, so a system whose
files spell the density columns some other way loads with no change to the
package. A test that reused the measured spelling could not tell the
difference between a declaration being read and a literal being matched.

The comparisons against real, measured tables are in the system tests
(``test_pipeline_anchors.py`` and each system's ``test_*_system_complete.py``) --
that is where a measurement exists to compare against, and this file stays
free of one on purpose.
"""
import numpy as np
import pytest
import torch

from aipf.functional.build import build
from aipf.paths import Paths
from aipf.system import AnchorRules, Functional, Mobility, System
from aipf.train.anchors import AnchorTables


@pytest.fixture(autouse=True)
def _demo_scratch_is_its_own(monkeypatch):
    """A demo system here declares its own scratch root; a user's ``AIPF_RAW``, which
    ``Paths.scratch`` prefers to a declared default, must not replace it."""
    monkeypatch.delenv("AIPF_RAW", raising=False)

_GRID = (4, 4, 4)

#: The toy's own column spellings. Not the measured files' -- see the
#: module docstring.
_COLUMNS = {
    "phase": "usable",
    "temperature": "temperature",
    "densities": ("density_a", "density_b"),
    "composition": "fraction",
    "mobility": "transport",
    "shells": "wavenumbers",
    "structure": "structure_factor",
    "bulk_target": "zero_wavenumber",
    "bulk_sigma": "zero_wavenumber_sigma",
    "bulk_ok": "extrapolation_ok",
}
_EOS_COLUMNS = {
    "x_column": "fraction", "T_column": "temperature",
    "n_columns": ("total_density",),
    "status_column": "state", "status_ok": "fine",
}


# ---------------------------------------------------------------------------
# the toy system and its three tables
# ---------------------------------------------------------------------------
def _toy_functional() -> Functional:
    return Functional(
        form="nonlocal_kernel", local="mlp", kernel="radial_mlp",
        kwargs=dict(
            grid=_GRID, R_cut=2.0, rho_ref=(0.3, 0.3),
            h_u=4, h_g=4, h_w=4, h_m=4,
            fexc_T_ref=1.0, rho_eps=1e-5, activation="gelu",
            g_form="mlp", ideal_form="gas", f_exc_form="split", nyquist_mask=True,
            gauge_fix=False, enable_TlnT=False, enable_T2=False,
            tbasis_ortho=False, kernel_n_quad=16, kernel_n_k_table=17,
            kernel_k_table_max=8.0, h_g_hat=4, h_g_tilde=4, T_ref=1000.0,
            local_input_scale=False))


def _demo(root, *, floors=None) -> System:
    return System(
        name="demo", n_species=2, species=("A", "B"),
        masses={"A": 1.0, "B": 2.0}, atom_types={"A": 1, "B": 2},
        table_keys={"rho": ("rho_A", "rho_B"), "x": "x_B", "x_channel": 1},
        paths=Paths(system="demo", raw_default=str(root)),
        anchor_rules=AnchorRules(dict(floors or {})),
        constants={"kB": 1.0},
        defaults={"k_fit_stat": 1.5},
        functional=_toy_functional(),
        mobility=Mobility(form="mlp_scaled", T_form="none",
                          kwargs=dict(mobility_prefactor="mole_fraction",
                                      mobility_input_ref=None)))


def _write_tables(root, *, rows=3, shells=4, with_ok=True):
    """One control value's pair of tables and one manifold, under ``root``.

    Row 0 is DROPPED by every filter the loader applies -- it is not
    single-phase -- so a test that finds three rows loaded from four
    written has measured the filter and not the writer.
    """
    root.mkdir(parents=True, exist_ok=True)
    n = rows + 1
    usable = np.array([False] + [True] * rows)
    temperature = np.linspace(1000.0, 4000.0, n)
    fraction = np.linspace(0.2, 0.8, n)
    total = np.linspace(0.5, 0.9, n)
    density_a = (1.0 - fraction) * total
    density_b = fraction * total
    transport = np.stack([np.eye(2) * (0.5 + i) for i in range(n)])
    wavenumbers = np.tile(np.linspace(0.4, 1.9, shells), (n, 1))
    structure = np.stack([np.stack([np.eye(2) * (0.8 + 0.1 * s)
                                    for s in range(shells)])
                          for _ in range(n)])
    table = dict(usable=usable, temperature=temperature, fraction=fraction,
                 density_a=density_a, density_b=density_b,
                 transport=transport, wavenumbers=wavenumbers,
                 structure_factor=structure,
                 zero_wavenumber=np.full(n, 0.2),
                 zero_wavenumber_sigma=np.full(n, 0.01))
    if with_ok:
        table["extrapolation_ok"] = np.ones(n, dtype=bool)
    np.savez(root / "m_table.npz", **table)
    np.savez(root / "s_table.npz", **table)
    lines = ["fraction,temperature,total_density,state"]
    for i in range(n):
        lines.append(f"{fraction[i]},{temperature[i]},{total[i]},"
                     f"{'fine' if i else 'broken'}")
    (root / "manifold.csv").write_text("\n".join(lines) + "\n")


def _declared(sub="tier"):
    return dict(key="pressure", m_table={100.0: f"{sub}/m_table.npz"},
                s_table={100.0: f"{sub}/s_table.npz"},
                eos_csvs={100.0: f"{sub}/manifold.csv"},
                pressure_unit=0.5, columns=dict(_COLUMNS),
                eos_columns=dict(_EOS_COLUMNS), row_weight=None,
                pressure_floor=None, w0={"route": "evaluator"})


@pytest.fixture
def tiny_tables(tmp_path):
    _write_tables(tmp_path / "tier")
    return _demo(tmp_path), _declared()


@pytest.fixture
def toy_model(tmp_path):
    model = build(_demo(tmp_path))
    model.train()
    return model


# ---------------------------------------------------------------------------
# what load refuses
# ---------------------------------------------------------------------------
def test_load_requires_every_table_and_refuses_a_missing_file(tmp_path):
    system = _demo(tmp_path)
    with pytest.raises(TypeError):
        AnchorTables.load(system)                      # nothing declared
    with pytest.raises(FileNotFoundError, match="m_table"):
        AnchorTables.load(system, **{**_declared(), "m_table":
                                     {100.0: "nowhere/m_table.npz"}})


def test_load_refuses_an_undeclared_column(tiny_tables):
    system, declared = tiny_tables
    columns = dict(declared["columns"])
    columns.pop("structure")
    with pytest.raises(KeyError, match="structure"):
        AnchorTables.load(system, **{**declared, "columns": columns})


def test_load_refuses_an_empty_declaration(tiny_tables):
    system, declared = tiny_tables
    with pytest.raises(ValueError, match="eos_csvs"):
        AnchorTables.load(system, **{**declared, "eos_csvs": {}})


def test_load_needs_the_systems_own_band_bound_and_energy_unit(tmp_path):
    """Both come off the DECLARATION and neither has a package fallback."""
    _write_tables(tmp_path / "tier")
    bare = _demo(tmp_path)
    no_band = _replace_defaults(bare, {})
    with pytest.raises(KeyError, match="k_fit_stat"):
        AnchorTables.load(no_band, **_declared())
    no_unit = _replace_constants(bare, {})
    with pytest.raises(KeyError, match="kB"):
        AnchorTables.load(no_unit, **_declared())


def _replace_defaults(system, defaults):
    import dataclasses
    return dataclasses.replace(system, defaults=defaults)


def _replace_constants(system, constants):
    import dataclasses
    return dataclasses.replace(system, constants=constants)


# ---------------------------------------------------------------------------
# what load produces
# ---------------------------------------------------------------------------
def test_the_declared_row_filter_is_applied(tiny_tables):
    """Four rows written, one not usable, three loaded into every anchor."""
    tables = AnchorTables.load(tiny_tables[0], **tiny_tables[1])
    assert tables.rows() == {"anchor_M": 3, "anchor_S": 3,
                             "anchor_bulk": 3, "anchor_P": 3}


def test_the_temperature_floor_is_keyed_on_the_declared_control_value(
        tmp_path):
    """The floor comes off the KEY, never off the path.

    The table below sits in a directory whose name says nothing, and its
    control value is 100.0 because the declaration says so; the floor
    declared at 100.0 is what cuts its rows. A loader that recovered the
    value from the path -- which is what the source tree did -- would find
    nothing to match and would apply no cut at all.
    """
    _write_tables(tmp_path / "tier")
    unfiltered = AnchorTables.load(_demo(tmp_path), **_declared())
    filtered = AnchorTables.load(_demo(tmp_path, floors={100.0: 2500.0}),
                                 **_declared())
    assert unfiltered.rows()["anchor_M"] == 3
    assert filtered.rows()["anchor_M"] == 2
    assert float(filtered.mobility_T.min()) > 2500.0
    # The manifold is NOT an anchor table and the floor never reaches it.
    assert filtered.rows()["anchor_P"] == unfiltered.rows()["anchor_P"]


def test_the_declared_pressure_unit_scales_the_manifold_target(tiny_tables):
    system, declared = tiny_tables
    half = AnchorTables.load(system, **declared)
    double = AnchorTables.load(system, **{**declared, "pressure_unit": 1.0})
    assert torch.allclose(2.0 * half.pressure_target, double.pressure_target)


def test_the_shell_band_keeps_only_the_declared_wavenumbers(tiny_tables):
    """``k_fit_stat`` bounds the band and ``k = 0`` is never in it."""
    system, declared = tiny_tables
    tables = AnchorTables.load(system, **declared)
    kept = tables.shell_k[tables.shell_mask]
    # `_write_tables` writes `linspace(0.4, 1.9, 4)` -- 0.4, 0.9, 1.4, 1.9 --
    # on each of its three usable rows, and the declared bound is 1.5, so the
    # kept band is the first three of the four, three times over. Written out
    # rather than recomputed from the writer: a bound that stopped being
    # applied would still agree with a recomputed expectation.
    in_band = torch.tensor([0.4, 0.9, 1.4], dtype=kept.dtype)
    assert kept.shape == (3 * 3,)
    assert torch.allclose(kept.reshape(3, 3), in_band.expand(3, 3), atol=1e-6)
    assert float(kept.max()) <= system.defaults["k_fit_stat"]
    assert tables.shell_k.shape[1] == 3, (
        "four shells were written and one sits above the declared bound, so "
        "a shell axis of any other width means the bound was not applied")


def test_a_table_without_the_flag_column_falls_back_and_says_so(tmp_path,
                                                                caplog):
    _write_tables(tmp_path / "tier", with_ok=False)
    with caplog.at_level("WARNING"):
        tables = AnchorTables.load(_demo(tmp_path), **_declared())
    assert tables.rows()["anchor_bulk"] == 3
    assert any("extrapolation_ok" in r.getMessage()
               for r in caplog.records)


def test_several_tables_concatenate_in_declaration_order(tmp_path):
    """Two control values, and the rows arrive in the order declared."""
    _write_tables(tmp_path / "hot", rows=3)
    _write_tables(tmp_path / "cold", rows=2)
    declared = dict(
        key="pressure",
        m_table={100.0: "hot/m_table.npz", 50.0: "cold/m_table.npz"},
        s_table={100.0: "hot/s_table.npz", 50.0: "cold/s_table.npz"},
        eos_csvs={100.0: "hot/manifold.csv", 50.0: "cold/manifold.csv"},
        pressure_unit=0.5, columns=dict(_COLUMNS),
        eos_columns=dict(_EOS_COLUMNS), row_weight=None, pressure_floor=None,
        w0={"route": "evaluator"})
    tables = AnchorTables.load(_demo(tmp_path), **declared)
    assert tables.rows()["anchor_M"] == 5
    one = AnchorTables.load(_demo(tmp_path),
                            **{**declared,
                               "m_table": {100.0: "hot/m_table.npz"},
                               "s_table": {100.0: "hot/s_table.npz"},
                               "eos_csvs": {100.0: "hot/manifold.csv"}})
    assert torch.equal(tables.mobility_target[:3], one.mobility_target)


def test_provenance_names_each_table_and_its_digest(tiny_tables):
    tables = AnchorTables.load(*tiny_tables[:1], **tiny_tables[1])
    for role in ("m_table", "s_table", "eos_csvs"):
        assert role in tables.provenance
        for path, digest in tables.provenance[role]:
            assert path.endswith((".npz", ".csv"))
            assert len(digest) == 64 and int(digest, 16) >= 0


# ---------------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------------
def test_batch_carries_exactly_the_four_entries_compute_losses_reads(
        tiny_tables, toy_model):
    tables = AnchorTables.load(tiny_tables[0], **tiny_tables[1])
    b = tables.batch(toy_model, kB=1.0)
    assert set(b) == {"anchor_M", "anchor_S", "anchor_bulk", "anchor_P"}
    assert set(b["anchor_M"]) >= {"M_pred", "M_target"}
    assert set(b["anchor_S"]) >= {"H", "target", "mask"}
    assert set(b["anchor_bulk"]) >= {"H0", "kBT", "zvec", "rho_tot", "target"}
    assert "P_model" in b["anchor_P"]


def test_the_batch_is_what_compute_losses_consumes(tiny_tables, toy_model):
    """Not a shape assertion: the four terms are really computed from it."""
    from aipf.train.config import TrainConfig
    from aipf.train.lit_module import LitModule

    tables = AnchorTables.load(tiny_tables[0], **tiny_tables[1])
    config = TrainConfig(lambda_M=0.5, lambda_S=0.01, lambda_bulk=0.01,
                         lambda_P=5.0, sigma=1.0, k_max=2.0,
                         stat_metric="rel_frob",
                         bulk_residual="relative_inverse", seed=0)
    lit = LitModule(toy_model, toy_model._cache, config)
    total, parts = lit.compute_losses(tables.batch(toy_model, kB=1.0))
    assert sorted(parts) == ["L_M", "L_P", "L_S", "L_bulk"]
    assert torch.isfinite(total) and total.requires_grad
    total.backward()
    reached = [n for n, p in toy_model.named_parameters()
               if p.grad is not None and float(p.grad.abs().sum()) > 0.0]
    assert reached, "no parameter was reached by the four anchors"


def test_targets_are_loaded_once_and_the_model_side_is_recomputed(
        tiny_tables, toy_model):
    tables = AnchorTables.load(tiny_tables[0], **tiny_tables[1])
    first = tables.batch(toy_model, kB=1.0)
    frozen = {key: first[key][name].detach().clone()
              for key, name in (("anchor_M", "M_target"),
                                ("anchor_S", "target"),
                                ("anchor_bulk", "target"),
                                ("anchor_P", "P_target"))}
    moved = first["anchor_M"]["M_pred"].detach().clone()
    with torch.no_grad():
        for p in toy_model.parameters():
            p.add_(0.1)
    second = tables.batch(toy_model, kB=1.0)
    for key, name in (("anchor_M", "M_target"), ("anchor_S", "target"),
                      ("anchor_bulk", "target"), ("anchor_P", "P_target")):
        assert torch.equal(frozen[key], second[key][name])
    assert not torch.equal(moved, second["anchor_M"]["M_pred"])


def test_the_energy_unit_reaches_the_bulk_anchor(tiny_tables, toy_model):
    tables = AnchorTables.load(tiny_tables[0], **tiny_tables[1])
    one = tables.batch(toy_model, kB=1.0)["anchor_bulk"]["kBT"]
    two = tables.batch(toy_model, kB=2.0)["anchor_bulk"]["kBT"]
    assert torch.allclose(2.0 * one, two)


def test_a_model_without_a_field_wide_term_contributes_a_zero(toy_model):
    """``w_hat`` is the one non-protocol reach, and its absence is a zero.

    A rung whose free energy is purely local has no such submodule, and the
    curvature it compares against the measured one is then the local
    Hessian alone -- which is the correct limit, not a degenerate case.
    Exercised on the resolver itself, because deleting a real rung's kernel
    breaks its own forward pass long before the anchor tier sees it.
    """
    from aipf.train.anchors import _w_hat_of

    class _Local:
        n_species = 2

    k = torch.linspace(0.1, 1.0, 5).reshape(5, 1)
    zero = _w_hat_of(_Local())(k)
    assert zero.shape == (5, 1, 2, 2)
    assert float(zero.abs().max()) == 0.0
    # The resolver returns the model's OWN `w_hat`, so the value-level
    # expectation is that method's own answer. "Something non-zero" would
    # pass for any other tensor of the right shape.
    real = _w_hat_of(toy_model)(k)
    assert torch.equal(real, toy_model.kernel.w_hat(k))
    assert real.shape == zero.shape
    assert float(real.abs().max()) > 0.0


def test_a_field_wide_term_this_module_cannot_read_is_refused_not_zeroed():
    """A square-gradient rung carries a real curvature at every ``k``.

    It keeps that term as a coefficient, not as a pair kernel, and there is
    no protocol method to read it from here. Returning zero would drop a
    physical contribution out of the static anchors' target comparison and
    say so nowhere, which is the one failure a static anchor cannot show --
    so the resolver refuses and names the method that would fix it.
    """
    from aipf.train.anchors import _w_hat_of

    class _SquareGradient:
        n_species = 2
        kappa = object()          # present and not None: that is the test

    with pytest.raises(NotImplementedError,
                       match="curvature_at_wavevector"):
        _w_hat_of(_SquareGradient())


# ---------------------------------------------------------------------------
# the declared variants: a gate per role, row weights, a pressure floor,
# and a composition column that labels channel 0
# ---------------------------------------------------------------------------
def _with_gate(root, name="strict", drop=1):
    """Add a second boolean column to the structure table that drops one more row."""
    with np.load(root / "s_table.npz") as z:
        table = dict(z)
    gate = table["usable"].copy()
    gate[drop] = False
    table[name] = gate
    table["zero_wavenumber_err"] = np.linspace(0.001, 0.02, len(gate))
    np.savez(root / "s_table.npz", **table)


def test_a_gate_per_role_filters_each_table_by_its_own_column(tmp_path):
    _write_tables(tmp_path / "tier")
    _with_gate(tmp_path / "tier")
    columns = {**_COLUMNS, "phase": {"mobility": "usable",
                                     "structure": "strict"}}
    tables = AnchorTables.load(_demo(tmp_path),
                               **{**_declared(), "columns": columns})
    assert tables.rows()["anchor_M"] == 3
    assert tables.rows()["anchor_S"] == 2
    assert tables.rows()["anchor_bulk"] == 2


def test_a_gate_mapping_must_name_both_roles(tmp_path):
    _write_tables(tmp_path / "tier")
    columns = {**_COLUMNS, "phase": {"mobility": "usable"}}
    with pytest.raises(KeyError, match="roles"):
        AnchorTables.load(_demo(tmp_path),
                          **{**_declared(), "columns": columns})


def test_declared_row_weights_are_relative_error_weights_of_mean_one(tmp_path):
    """``1 / max(|err / value|, floor)^2``, normalised; ``None`` weighs nothing."""
    _write_tables(tmp_path / "tier")
    _with_gate(tmp_path / "tier")
    spec = {"value": "zero_wavenumber", "error": "zero_wavenumber_err",
            "floor": 0.02}
    weighted = AnchorTables.load(_demo(tmp_path),
                                 **{**_declared(), "row_weight": spec})
    plain = AnchorTables.load(_demo(tmp_path), **_declared())
    assert plain.shell_weight is None and plain.bulk_weight is None
    with np.load(tmp_path / "tier" / "s_table.npz") as z:
        keep = z["usable"]
        rel = np.maximum(np.abs(z["zero_wavenumber_err"][keep]
                                / z["zero_wavenumber"][keep]), 0.02)
    w = 1.0 / rel ** 2
    expected = torch.from_numpy((w / w.mean()).astype(np.float32))
    assert torch.equal(weighted.shell_weight, expected)
    assert torch.equal(weighted.bulk_weight, expected)
    assert weighted.batch(build(_demo(tmp_path)), 1.0)["anchor_S"][
        "row_weight"] is weighted.shell_weight


def test_a_pressure_floor_is_declared_in_the_keys_unit(tiny_tables):
    system, declared = tiny_tables
    tables = AnchorTables.load(system, **{**declared, "pressure_floor": 4.0})
    assert tables.pressure_floor == 4.0 * 0.5
    assert AnchorTables.load(system, **declared).pressure_floor is None


def test_a_composition_that_labels_channel_zero_is_read_as_such(tmp_path):
    """``x_channel`` 0: the manifold row is ``(x n, (1 - x) n)`` and the bulk projector uses ``1 - x``."""
    import dataclasses

    _write_tables(tmp_path / "tier")
    one = _demo(tmp_path)
    zero = dataclasses.replace(one, table_keys={**one.table_keys,
                                                "x_channel": 0})
    a = AnchorTables.load(one, **_declared())
    b = AnchorTables.load(zero, **_declared())
    assert torch.equal(a.pressure_rho.flip(-1), b.pressure_rho)
    assert torch.allclose(a.bulk_zvec, -b.bulk_zvec.flip(-1), atol=1e-7)


# ---------------------------------------------------------------------------
# the declared key, and the temperature-keyed form
# ---------------------------------------------------------------------------
def test_the_coordinate_the_tables_are_keyed_by_is_declared(tiny_tables):
    system, declared = tiny_tables
    with pytest.raises(ValueError, match="pressure.*temperature"):
        AnchorTables.load(system, **{**declared, "key": "volume"})
    rest = {k: v for k, v in declared.items() if k != "key"}
    with pytest.raises(TypeError, match="key"):
        AnchorTables.load(system, **rest)


def test_pressure_keyed_tables_feed_all_four_terms(tiny_tables):
    tables = AnchorTables.load(tiny_tables[0], **tiny_tables[1])
    assert tables.terms() == ("L_M", "L_S", "L_bulk", "L_P")


#: A one-field system's temperature-keyed tables: toy column names, three samples per temperature.
_T_COLUMNS = {"temperature": "temp", "mobility": "m_mean", "records": "samples",
              "record_temperature": "temp", "structure_zero": "s_zero",
              "temperatures": "temps"}
_T_ROWS = (1.5, 1.7)
_SAMPLES = {1.5: (2.0, 2.2, 2.4), 1.7: (1.0, 1.1, 1.3)}


def _write_temperature_tables(root):
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "m.npz", temp=np.array(_T_ROWS), m_mean=np.array([0.02, 0.03]))
    records = np.array([{"temp": T, "s_zero": v} for T in _T_ROWS
                        for v in _SAMPLES[T]], dtype=object)
    np.savez(root / "s.npz", samples=records, temps=np.array(_T_ROWS))
    return dict(key="temperature", m_table=str(root / "m.npz"),
                s_table=str(root / "s.npz"), state=(0.5,), rho_total=1.0,
                columns=dict(_T_COLUMNS), w0={"route": "lattice"})


#: A toy one-field lattice functional: a Taylor energy and a lattice-sum kernel, no real system's numbers.
_ONE_FIELD_KWARGS = dict(
    grid=(4, 4, 8), R_cut=1.2, rho_ref=(0.5,), h_u=4, h_g=4, h_w=4,
    fexc_T_ref=1.0, T_ref=1.0, rho_eps=1e-4, activation="gelu", g_form="icnn",
    enable_TlnT=False, enable_T2=False, h_g_hat=None, h_g_tilde=None,
    gauge_fix=False, disable_g=False, f_exc_form="split", ideal_form="lattice",
    local_input_scale=False, u_degree=4, u_parity="even",
    u_variable="difference", g_symmetry="mirror", icnn_output_bias=False,
    kernel_argument="difference", kernel_evaluator="lattice_sum",
    nyquist_mask=False)


def _one_field(tmp_path, **defaults) -> System:
    return System(
        name="demo1", n_species=1, species=("A",), masses={"A": 1.0},
        atom_types={"A": 1}, table_keys={"rho": ("rho_A",), "x": "x_A",
                                         "x_channel": 0},
        paths=Paths(system="demo1", raw_default=str(tmp_path)),
        anchor_rules=AnchorRules({}), constants={"kB": 1.0},
        defaults={"box": (2.0, 2.0, 4.0), **defaults},
        functional=Functional(form="nonlocal_kernel", local="taylor",
                              kernel="radial_mlp",
                              kwargs=dict(_ONE_FIELD_KWARGS)),
        mobility=Mobility(form="lattice_scalar", T_form="arrhenius",
                          kwargs=dict(mobility_prefactor="mole_fraction",
                                      shape_init=-2.0,
                                      mobility_activation_energy_init=0.1)))


def test_temperature_keyed_rows_are_the_mean_and_its_inverse_standard_error(
        tmp_path):
    declared = _write_temperature_tables(tmp_path / "t")
    tables = AnchorTables.load(_one_field(tmp_path), **declared)
    assert tables.rows() == {"anchor_M": 2, "anchor_S": 0, "anchor_bulk": 2,
                             "anchor_P": 0}
    assert tables.terms() == ("L_M", "L_bulk")
    for i, T in enumerate(_T_ROWS):
        s = np.array(_SAMPLES[T])
        sem = s.std(ddof=1) / np.sqrt(s.size)
        assert tables.bulk_target[i].item() == pytest.approx(s.mean(), rel=1e-6)
        assert tables.bulk_sigma[i].item() == pytest.approx(
            sem / s.mean() ** 2, rel=1e-6)
    assert torch.equal(tables.bulk_rho, torch.full((2, 1), 0.5))
    assert torch.equal(tables.mobility_target.reshape(-1),
                       torch.tensor([0.02, 0.03]))
    assert torch.equal(tables.bulk_zvec, torch.ones(2, 1))


def test_temperature_keyed_tables_refuse_more_than_one_channel(tmp_path):
    declared = _write_temperature_tables(tmp_path / "t")
    _write_tables(tmp_path / "tier")
    with pytest.raises(NotImplementedError, match="one channel"):
        AnchorTables.load(_demo(tmp_path), **{**declared, "state": (0.5, 0.5)})


def test_temperature_keyed_tables_refuse_an_undeclared_column(tmp_path):
    declared = _write_temperature_tables(tmp_path / "t")
    columns = dict(declared["columns"])
    columns.pop("structure_zero")
    with pytest.raises(KeyError, match="structure_zero"):
        AnchorTables.load(_one_field(tmp_path), **{**declared,
                                                  "columns": columns})


def test_a_lattice_sum_kernel_is_given_the_declared_grid_and_box_at_k_zero(
        tmp_path):
    """``W_hat(0) = dV sum_ij W(r_ij)`` on the declared grid and box, the zero entry of the lattice sum."""
    system = _one_field(tmp_path)
    declared = _write_temperature_tables(tmp_path / "t")
    tables = AnchorTables.load(system, **declared)
    model = build(system)
    batch = tables.batch(model, 1.0)
    assert set(batch) == {"anchor_M", "anchor_bulk"}
    grid = tuple(system.functional.kwargs["grid"])
    box = torch.tensor([system.defaults["box"]], dtype=torch.float64)
    whole = model.kernel.w_hat(torch.zeros(1, 1), grid=grid, boxes=box)
    from aipf.train.anchors import _curvature
    H0 = _curvature(model, tables.bulk_rho, tables.bulk_T)
    expected = H0 + whole[:, 0, 0, 0].real
    assert torch.allclose(batch["anchor_bulk"]["H0"], expected)
    evaluator = model.kernel.evaluator
    sp = evaluator.spacing(grid, box[0], torch.float32)
    W = evaluator.w_grid(model.kernel.radial_set, grid, sp)
    direct = (sp[0] * sp[1] * sp[2]) * W.sum()
    assert whole[0, 0, 0, 0].real.reshape(()).item() == pytest.approx(
        direct.item(), rel=1e-5)


def test_without_a_declared_box_the_lattice_sum_still_refuses_by_name(
        tmp_path):
    system = _one_field(tmp_path)
    no_box = _replace_defaults(system, {k: v for k, v in system.defaults.items()
                                        if k != "box"})
    tables = AnchorTables.load(no_box, **_write_temperature_tables(tmp_path / "t"))
    assert tables.geometry is None
    with pytest.raises(NotImplementedError, match="lattice_sum"):
        tables.batch(build(no_box), 1.0)


@pytest.mark.parametrize("w0, match", [
    ({}, "not one of"),
    ({"route": "quadrature"}, "not one of"),
    ({"route": "radial", "r_max": 1.0}, "exactly"),
    ({"route": "lattice", "n_points": 10}, "exactly"),
])
def test_the_zero_wavenumber_route_is_a_checked_declaration(tmp_path, w0, match):
    declared = _write_temperature_tables(tmp_path / "t")
    with pytest.raises(ValueError, match=match):
        AnchorTables.load(_one_field(tmp_path), **{**declared, "w0": w0})


def test_the_radial_route_is_the_kernels_quadrature_and_keeps_its_graph(tmp_path):
    system = _one_field(tmp_path)
    declared = {**_write_temperature_tables(tmp_path / "t"),
                "w0": {"route": "radial", "r_max": 1.2, "n_points": 50}}
    tables = AnchorTables.load(system, **declared)
    model = build(system)
    H0 = tables.batch(model, 1.0)["anchor_bulk"]["H0"]
    from aipf.train.anchors import _curvature
    expected = (_curvature(model, tables.bulk_rho, tables.bulk_T)
                + model.kernel.w_hat_zero_radial(1.2, 50))
    assert torch.allclose(H0, expected)
    H0.sum().backward()
    grads = [p.grad for p in model.kernel.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_the_lattice_route_is_refused_for_a_kernel_that_is_not_a_lattice_sum(
        tiny_tables, toy_model):
    system, declared = tiny_tables
    tables = AnchorTables.load(system, **{**declared, "w0": {"route": "lattice"}})
    with pytest.raises(ValueError, match="lattice sum"):
        tables.batch(toy_model, 1.0)


def test_the_lj_anchor_tables_load_with_no_raw_root_declared(monkeypatch):
    """The reduced-unit system's tables are tracked beside its declaration: a fresh checkout trains
    with no raw root declared, and the raw root is never asked for."""
    from aipf import paths
    from aipf.system import load
    from aipf.train.fit import _anchors_from_system
    for name in ("AIPF_RAW_LJ", "AIPF_RAW"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(paths, "config", lambda: {})
    lj = load("lj")
    with pytest.raises(paths.MissingLocation):
        lj.paths.raw()
    tables = _anchors_from_system(lj, None)
    assert tables is not None and len(tables.mobility_T) > 0
    read = {p for entries in tables.provenance.values() for p, _digest in entries}
    assert read and all(p.startswith(str(paths.tracked_tables("lj"))) for p in read), read
