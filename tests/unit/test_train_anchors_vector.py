"""``aipf.train.anchors`` beyond two channels: the vector declaration form, on toy files.

Three channels, and the three kinds of row a ternary campaign holds: a ternary row, two binary rows
(each with a different channel absent) and a pure row, plus one row the gate drops. The tables use
the toy's own column spellings, as in ``test_train_anchors.py``, except the manifold header, which
is the one a ternary campaign writes (the reader is checked against it).

What is pinned: which rows each anchor keeps (a pure row feeds ``L_P`` alone), the projector columns
``z_i = e_i - x_i a``, the active-block shell targets ``k_B T S_PP^-1`` with no non-positive-definite
warning for an absent species, the masked residuals against the same residual on the extracted
block, finite losses and gradients at an exactly zero density, the refusals of the declaration and
of a row that contradicts itself, and a toy ternary that recovers a known ``H(0)`` through
``L_bulk`` alone.
"""
import logging

import numpy as np
import pytest
import torch

from aipf.data.rows import measured_rows
from aipf.functional.build import build
from aipf.losses.mobility import l_m
from aipf.losses.static import l_bulk, l_s
from aipf.paths import Paths
from aipf.system import AnchorRules, Functional, Mobility, System
from aipf.train.anchors import AnchorTables


@pytest.fixture(autouse=True)
def _demo_scratch_is_its_own(monkeypatch):
    monkeypatch.delenv("AIPF_RAW", raising=False)


_KB = 1.0

#: Per file row: the fractions and the temperature. Row 0 is ternary but not usable (the gate drops
#: it); 1 ternary; 2 and 3 binary with channel 1 and channel 0 absent; 4 pure channel 2.
_X = np.array([[0.3, 0.3, 0.4],
               [0.2, 0.3, 0.5],
               [0.4, 0.0, 0.6],
               [0.0, 0.7, 0.3],
               [0.0, 0.0, 1.0]])
_T = np.array([1.0, 1.2, 1.4, 1.6, 1.8])
_TOTAL = np.array([0.50, 0.55, 0.60, 0.65, 0.70])
_SHELLS = np.array([0.4, 0.8, 1.2])

_COLUMNS = {
    "phase": "usable",
    "temperature": "temperature",
    "densities": ("density_a", "density_b", "density_c"),
    "composition": ("fraction_a", "fraction_b", "fraction_c"),
    "active": "present",
    "mobility": "transport",
    "shells": "wavenumbers",
    "structure": "structure_factor",
    "bulk_target": "zero_wavenumber",
    "bulk_sigma": None,
    "bulk_ok": None,
}
#: The manifold header a ternary campaign writes; extra columns are ignored.
_EOS_HEADER = ("composition_id,x_Fe,x_Cu,x_Ni,T_K,P_GPa,n_atoms_per_A3,n_atoms_per_A3_sem,"
               "V_per_atom_A3,status")
_EOS_COLUMNS = {
    "x_columns": ("x_Fe", "x_Cu", "x_Ni"), "T_column": "T_K",
    "n_columns": ("n_atoms_per_A3",),
    "status_column": "status", "status_ok": "ok",
}


def _spd_block(m, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, m))
    return a @ a.T + m * np.eye(m)


def _structure(row, shell):
    """The measured ``S(k)`` of a row: positive definite on the row's active block, zero (row 2) or NaN
    (rows 3, 4) outside it, as the two conventions a writer may use."""
    present = np.nonzero(_X[row] > 0)[0]
    out = np.full((3, 3), 0.0 if row == 2 else np.nan)
    out[np.ix_(present, present)] = _spd_block(present.size, 10 * row + shell)
    return out


def _write(root, *, x=_X, rho=None, active=None, transport=None):
    """The F1-layout pair of tables and one manifold, under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    rho = x * _TOTAL[:, None] if rho is None else rho
    active = (x > 0) if active is None else active
    n_rows = x.shape[0]
    if transport is None:
        transport = np.full((n_rows, 3, 3), np.nan)
        for r in range(n_rows):
            p = np.nonzero(_X[r] > 0)[0]
            transport[r][np.ix_(p, p)] = 0.1 * _spd_block(p.size, 100 + r)
    structure = np.stack([np.stack([_structure(r, s) for s in range(len(_SHELLS))])
                          for r in range(n_rows)])
    target = np.full((n_rows, 2, 2), np.nan)
    for r in range(n_rows):
        m = int((_X[r] > 0).sum()) - 1
        target[r, :m, :m] = 0.05 * _spd_block(m, 200 + r) if m else np.nan
    table = dict(usable=np.array([False] + [True] * (n_rows - 1)),
                 temperature=_T, present=active,
                 density_a=rho[:, 0], density_b=rho[:, 1], density_c=rho[:, 2],
                 fraction_a=x[:, 0], fraction_b=x[:, 1], fraction_c=x[:, 2],
                 transport=transport,
                 wavenumbers=np.tile(_SHELLS, (n_rows, 1)),
                 structure_factor=structure, zero_wavenumber=target,
                 relative_error=np.array([0.1, 0.01, 0.04, 0.05, 0.02]))
    np.savez(root / "m_table.npz", **table)
    np.savez(root / "s_table.npz", **table)
    lines = [_EOS_HEADER]
    for r in range(n_rows):
        lines.append(f"c{r},{x[r, 0]},{x[r, 1]},{x[r, 2]},{_T[r]},0.0001,{_TOTAL[r]},"
                     f"1e-7,{1 / _TOTAL[r]},ok")
    lines.append(f"cbroken,0.5,0.5,0.0,2.0,0.0001,0.6,1e-7,1.6,failed")
    (root / "manifold.csv").write_text("\n".join(lines) + "\n")
    return table


def _toy_functional():
    return Functional(
        form="nonlocal_kernel", local="mlp", kernel="radial_mlp",
        kwargs=dict(
            grid=(4, 4, 4), R_cut=2.0, rho_ref=(0.2, 0.2, 0.2),
            h_u=4, h_g=4, h_w=4, h_m=4,
            fexc_T_ref=1.0, rho_eps=1e-5, activation="gelu",
            g_form="icnn", ideal_form="gas", f_exc_form="joint", h_joint=8,
            joint_depth=2, nyquist_mask=True,
            gauge_fix=False, enable_TlnT=False, enable_T2=False,
            tbasis_ortho=False, kernel_n_quad=16, kernel_n_k_table=17,
            kernel_k_table_max=8.0, h_g_hat=4, h_g_tilde=4, T_ref=1.0,
            local_input_scale=False))


def _ternary(root) -> System:
    return System(
        name="demo", n_species=3, species=("A", "B", "C"),
        masses={"A": 1.0, "B": 2.0, "C": 3.0}, atom_types={"A": 1, "B": 2, "C": 3},
        table_keys={"rho": ("rho_A", "rho_B", "rho_C"), "x": "x_A"},
        paths=Paths(system="demo", raw_default=str(root)),
        anchor_rules=AnchorRules({}),
        constants={"kB": _KB},
        defaults={"k_fit_stat": 1.5},
        functional=_toy_functional(),
        mobility=Mobility(form="mlp_scaled", T_form="none",
                          kwargs=dict(mobility_prefactor="partial_density",
                                      mobility_input_ref=None)))


def _declared(**over):
    out = dict(key="pressure", m_table={0.0: "tier/m_table.npz"},
               s_table={0.0: "tier/s_table.npz"},
               eos_csvs={0.0: "tier/manifold.csv"}, pressure_unit=1.0,
               columns=dict(_COLUMNS), eos_columns=dict(_EOS_COLUMNS),
               row_weight={"value": None, "error": "relative_error", "floor": 0.02},
               pressure_floor=1.0, w0={"route": "evaluator"})
    out.update(over)
    return out


@pytest.fixture
def ternary(tmp_path):
    _write(tmp_path / "tier")
    return _ternary(tmp_path), _declared()


@pytest.fixture
def loaded(ternary):
    return AnchorTables.load(ternary[0], **ternary[1])


# ---------------------------------------------------------------------------
# rows, projector, targets
# ---------------------------------------------------------------------------
def test_a_pure_row_feeds_the_pressure_anchor_alone(loaded):
    # file rows 1, 2, 3 (row 0 is gated out, row 4 is pure); every ok manifold row, pure ones too
    assert loaded.rows() == {"anchor_M": 3, "anchor_S": 3, "anchor_bulk": 3,
                             "anchor_P": 5}
    expected = torch.tensor(_X[1:4] > 0)
    for mask in (loaded.mobility_active, loaded.shell_active, loaded.bulk_active):
        assert torch.equal(mask, expected)
    assert torch.equal(loaded.bulk_zmask,
                       torch.tensor([[True, True], [True, False], [True, False]]))
    assert torch.allclose(loaded.pressure_rho.double(),
                          torch.tensor(_X * _TOTAL[:, None]), rtol=1e-6)


def test_the_projector_columns_are_e_i_minus_x_i_times_the_active_indicator(loaded):
    expected = torch.tensor([
        # ternary (0.2, 0.3, 0.5): independent species 0 and 1
        [[0.8, -0.3], [-0.2, 0.7], [-0.2, -0.3]],
        # (0.4, 0, 0.6): species 0 alone, nothing on the absent channel
        [[0.6, 0.0], [0.0, 0.0], [-0.4, 0.0]],
        # (0, 0.7, 0.3): species 1 alone
        [[0.0, 0.0], [0.3, 0.0], [-0.7, 0.0]],
    ], dtype=torch.float32)
    assert torch.allclose(loaded.bulk_zvec, expected, atol=1e-7)


def test_the_bulk_target_is_the_leading_block_padded_by_the_identity(ternary, loaded):
    table = np.load(ternary[0].paths.raw() / "tier/s_table.npz")
    target = table["zero_wavenumber"]
    assert torch.allclose(loaded.bulk_target[0].double(), torch.tensor(target[1]))
    for i, row in ((1, 2), (2, 3)):
        assert float(loaded.bulk_target[i, 0, 0]) == pytest.approx(target[row, 0, 0])
        assert torch.equal(loaded.bulk_target[i, 1], torch.tensor([0.0, 1.0]))
        assert torch.equal(loaded.bulk_target_rootinv[i, 1], torch.tensor([0.0, 1.0]))
    R = loaded.bulk_target_rootinv.double()
    assert torch.allclose(R @ loaded.bulk_target.double() @ R,
                          torch.eye(2, dtype=torch.float64).expand(3, 2, 2), atol=1e-5)


def test_an_absent_species_drops_no_shell_and_its_target_is_the_active_block_inverse(
        ternary, caplog):
    with caplog.at_level(logging.WARNING, logger="aipf.train.anchors"):
        tables = AnchorTables.load(ternary[0], **ternary[1])
    assert not caplog.records, [r.getMessage() for r in caplog.records]
    assert bool(tables.shell_mask.all())
    for i, row in enumerate((1, 2, 3)):
        present = np.nonzero(_X[row] > 0)[0]
        for shell in range(len(_SHELLS)):
            want = np.eye(3)
            S = _structure(row, shell)[np.ix_(present, present)]
            want[np.ix_(present, present)] = _KB * _T[row] * np.linalg.inv(S)
            got = tables.shell_target[i, shell].double()
            assert torch.allclose(got, torch.tensor(want), rtol=1e-6, atol=1e-7)
            R = tables.shell_target_rootinv[i, shell].double()
            assert torch.allclose(R @ got @ R, torch.eye(3, dtype=torch.float64),
                                  atol=1e-5)


def test_the_mobility_target_is_zero_outside_the_active_block(ternary, loaded):
    table = np.load(ternary[0].paths.raw() / "tier/m_table.npz")
    for i, row in enumerate((1, 2, 3)):
        pairs = np.outer(_X[row] > 0, _X[row] > 0)
        want = np.where(pairs, table["transport"][row], 0.0)
        assert torch.allclose(loaded.mobility_target[i].double(), torch.tensor(want))


def test_relative_errors_are_weighted_without_a_value_column(loaded):
    w = 1.0 / np.maximum(np.array([0.01, 0.04, 0.05]), 0.02) ** 2
    want = torch.tensor(w / w.mean(), dtype=torch.float32)
    assert torch.allclose(loaded.bulk_weight, want)
    assert torch.allclose(loaded.shell_weight, want)


# ---------------------------------------------------------------------------
# the masked residuals against the extracted block
# ---------------------------------------------------------------------------
def _spd_t(n, batch, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, n, n, generator=g, dtype=torch.float64)
    return a @ a.transpose(-1, -2) + n * torch.eye(n, dtype=torch.float64)


_ACTIVE = torch.tensor([[True, True, True], [True, False, True], [False, True, True]])


def test_the_masked_mobility_residual_is_the_active_blocks_own():
    pred, target = _spd_t(3, 3, 1), _spd_t(3, 3, 2)
    pred[1, 1, :] = pred[1, :, 1] = float("nan")       # an absent channel's entries are never read
    target[2, 0, :] = target[2, :, 0] = 0.0
    for r in range(3):
        idx = torch.nonzero(_ACTIVE[r]).flatten()
        block = l_m(pred[r][idx][:, idx][None], target[r][idx][:, idx][None])
        assert torch.allclose(l_m(pred[r:r + 1], target[r:r + 1], active=_ACTIVE[r:r + 1]),
                              block, rtol=1e-12)


@pytest.mark.parametrize("metric", ["rel_frob", "s_metric"])
def test_the_masked_shell_residual_is_the_active_blocks_own(metric):
    from aipf.train.anchors import _rootinv_on_blocks
    shells = 2
    H = _spd_t(3, 3 * shells, 3).reshape(3, shells, 3, 3)
    target = _spd_t(3, 3 * shells, 4).reshape(3, shells, 3, 3)
    eye = torch.eye(3, dtype=torch.float64)
    pairs = (_ACTIVE[:, :, None] & _ACTIVE[:, None, :])[:, None]
    target = torch.where(pairs, target, eye)          # the loader's identity padding
    R = _rootinv_on_blocks(target, _ACTIVE)
    mask = torch.ones(3, shells, dtype=torch.bool)
    for r in range(3):
        idx = torch.nonzero(_ACTIVE[r]).flatten()
        sub = (slice(r, r + 1), slice(None), idx[:, None], idx)
        block = l_s(H[sub], target[sub], mask[r:r + 1], metric=metric,
                    target_rootinv=R[sub])
        masked = l_s(H[r:r + 1], target[r:r + 1], mask[r:r + 1], metric=metric,
                     target_rootinv=R[r:r + 1], active=_ACTIVE[r:r + 1])
        assert torch.allclose(masked, block, rtol=1e-10), (r, metric)


def _block_batch(H0, x, active):
    """The loader's matrix batch for one row, by hand: ``Z`` columns and the padding."""
    n = active.numel()
    independent = torch.nonzero(active).flatten()[:-1]
    zvec = torch.zeros(1, n, n - 1, dtype=torch.float64)
    zmask = torch.zeros(1, n - 1, dtype=torch.bool)
    for j, i in enumerate(independent):
        zvec[0, :, j] = -x[i] * active.double()
        zvec[0, i, j] += 1.0
        zmask[0, j] = True
    return zvec, zmask


def test_the_masked_bulk_residual_is_the_active_blocks_own():
    x = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.0, 0.6], [0.0, 0.7, 0.3]],
                     dtype=torch.float64)
    H0 = _spd_t(3, 3, 5)
    H0[1, 1, :] = H0[1, :, 1] = 1e5                   # the ideal term of an absent channel
    kBT = torch.tensor([1.0, 1.3, 0.7], dtype=torch.float64)
    rho_tot = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
    from aipf.losses.static import sinv_rootinv
    for r in range(3):
        zvec, zmask = _block_batch(H0[r], x[r], _ACTIVE[r])
        m = int(zmask.sum())
        target = torch.eye(2, dtype=torch.float64)[None].clone()
        target[0, :m, :m] = 0.04 * _spd_t(m, 1, 6 + r)[0]
        R = torch.eye(2, dtype=torch.float64)[None].clone()
        R[0, :m, :m] = sinv_rootinv(target[0, :m, :m])
        masked = l_bulk(H0[r:r + 1], kBT[r:r + 1], zvec, rho_tot[r:r + 1], target,
                        residual="s_metric", target_rootinv=R,
                        active=_ACTIVE[r:r + 1], zmask=zmask)
        idx = torch.nonzero(_ACTIVE[r]).flatten()
        n_p = idx.numel()
        sub_z = zvec[:, idx][:, :, :n_p - 1]
        block = l_bulk(H0[r][idx][:, idx][None], kBT[r:r + 1], sub_z, rho_tot[r:r + 1],
                       target[:, :m, :m], residual="s_metric",
                       target_rootinv=R[:, :m, :m],
                       active=torch.ones(1, n_p, dtype=torch.bool),
                       zmask=torch.ones(1, n_p - 1, dtype=torch.bool))
        # and by hand: S_model = kBT Z_P^T H_PP^-1 Z_P / rho_tot
        S = kBT[r] / rho_tot[r] * sub_z[0].T @ torch.linalg.solve(
            H0[r][idx][:, idx], sub_z[0])
        by_hand = ((R[0, :m, :m] @ S @ R[0, :m, :m] - torch.eye(m, dtype=torch.float64))
                   .pow(2).sum())
        assert torch.allclose(masked, block, rtol=1e-10)
        assert torch.allclose(masked, by_hand, rtol=1e-8)


def test_the_block_residual_of_one_species_pair_is_the_squared_relative_error():
    """``m = 1``: ``||R S R - I||^2 = (rho - 1)^2`` with ``rho = S_model / S_target``."""
    H0 = torch.tensor([[[2.0, 0.3], [0.3, 1.5]]], dtype=torch.float64)
    x1 = 0.3
    zvec = torch.tensor([[[1.0 - x1], [-x1]]], dtype=torch.float64)
    kBT, rho_tot = torch.tensor([1.2], dtype=torch.float64), torch.tensor([0.8], dtype=torch.float64)
    model = float(kBT / rho_tot * zvec[0].T @ torch.linalg.solve(H0[0], zvec[0]))
    target = 0.7 * model
    got = l_bulk(H0, kBT, zvec, rho_tot,
                 torch.tensor([[[target]]], dtype=torch.float64), residual="s_metric",
                 target_rootinv=torch.tensor([[[target ** -0.5]]], dtype=torch.float64),
                 active=torch.ones(1, 2, dtype=torch.bool),
                 zmask=torch.ones(1, 1, dtype=torch.bool), jitter=0.0)
    assert float(got) == pytest.approx((model / target - 1.0) ** 2, rel=1e-12)


def test_a_matrix_batch_and_a_scalar_residual_do_not_mix(loaded):
    t = loaded
    H0 = torch.eye(3).expand(3, 3, 3)
    for residual in ("relative_inverse", "sigma_chi2"):
        with pytest.raises(ValueError, match="declare 's_metric'"):
            l_bulk(H0, t.bulk_T, t.bulk_zvec, t.bulk_rho_tot, t.bulk_target,
                   residual=residual, target_rootinv=t.bulk_target_rootinv,
                   active=t.bulk_active, zmask=t.bulk_zmask)
    with pytest.raises(ValueError, match="needs a matrix batch"):
        l_bulk(torch.eye(2).expand(4, 2, 2), torch.ones(4), torch.ones(4, 2),
               torch.ones(4), torch.ones(4), residual="s_metric")


# ---------------------------------------------------------------------------
# through the model
# ---------------------------------------------------------------------------
def test_losses_and_gradients_are_finite_at_an_exactly_zero_density(loaded, ternary):
    from aipf.train.config import TrainConfig
    from aipf.train.lit_module import LitModule

    assert float(loaded.bulk_rho[1, 1]) == 0.0 and float(loaded.shell_rho[2, 0]) == 0.0
    torch.manual_seed(0)
    model = build(ternary[0])
    model.train()
    batch = loaded.batch(model, kB=_KB)
    assert torch.equal(batch["anchor_S"]["active"], loaded.shell_active)
    assert set(batch["anchor_bulk"]) >= {"active", "zmask", "target_rootinv"}
    for metric in ("rel_frob", "s_metric"):
        config = TrainConfig(lambda_M=1.0, lambda_S=1.0, lambda_bulk=1.0,
                             lambda_P=1.0, sigma=1.0, k_max=2.0, stat_metric=metric,
                             bulk_residual="s_metric", seed=0)
        lit = LitModule(model, model._cache, config)
        model.zero_grad(set_to_none=True)
        total, parts = lit.compute_losses(loaded.batch(model, kB=_KB))
        assert sorted(parts) == ["L_M", "L_P", "L_S", "L_bulk"]
        assert all(bool(torch.isfinite(v)) for v in parts.values()), parts
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


def test_a_scalar_residual_on_vector_tables_is_refused_at_the_first_step(loaded, ternary):
    from aipf.train.config import TrainConfig
    from aipf.train.lit_module import LitModule

    model = build(ternary[0])
    config = TrainConfig(lambda_bulk=1.0, sigma=1.0, k_max=2.0,
                         bulk_residual="relative_inverse", seed=0)
    with pytest.raises(ValueError, match="declare 's_metric'"):
        LitModule(model, model._cache, config).compute_losses(loaded.batch(model, kB=_KB))


# ---------------------------------------------------------------------------
# the manifold reader
# ---------------------------------------------------------------------------
def test_the_manifold_reads_one_fraction_per_channel_off_the_campaign_header(tmp_path):
    _write(tmp_path)
    rows = list(measured_rows(tmp_path / "manifold.csv", **_EOS_COLUMNS))
    assert len(rows) == 5                               # the failed row is skipped
    assert rows[1] == ((0.2, 0.3, 0.5), 1.2, 0.55)
    assert rows[4][0] == (0.0, 0.0, 1.0)


def test_the_manifold_refuses_fractions_that_do_not_sum_to_one(tmp_path):
    x = _X.copy()
    x[3] = (0.0, 0.7, 0.31)
    _write(tmp_path, x=x)
    with pytest.raises(ValueError, match="line 5.*not to one"):
        list(measured_rows(tmp_path / "manifold.csv", **_EOS_COLUMNS))


def test_the_manifold_reader_takes_exactly_one_composition_form(tmp_path):
    _write(tmp_path)
    both = dict(_EOS_COLUMNS, x_column="x_Fe")
    neither = {k: v for k, v in _EOS_COLUMNS.items() if k != "x_columns"}
    for columns in (both, neither):
        with pytest.raises(TypeError, match="exactly one of x_column"):
            list(measured_rows(tmp_path / "manifold.csv", **columns))


# ---------------------------------------------------------------------------
# what the declaration refuses
# ---------------------------------------------------------------------------
def test_one_composition_name_beyond_two_channels_is_refused(ternary):
    system, declared = ternary
    columns = {k: v for k, v in _COLUMNS.items() if k != "active"}
    columns["composition"] = "fraction_a"
    eos = {k: v for k, v in _EOS_COLUMNS.items() if k != "x_columns"}
    eos["x_column"] = "x_Fe"
    with pytest.raises(ValueError, match="cannot say\\s+which channel"):
        AnchorTables.load(system, **{**declared, "columns": columns, "eos_columns": eos})


def test_the_two_forms_do_not_mix(ternary):
    system, declared = ternary
    scalar_eos = {k: v for k, v in _EOS_COLUMNS.items() if k != "x_columns"}
    scalar_eos["x_column"] = "x_Fe"
    with pytest.raises(ValueError, match="mix the two forms"):
        AnchorTables.load(system, **{**declared, "eos_columns": scalar_eos})
    one_name = dict(_COLUMNS, composition="fraction_a")
    with pytest.raises(ValueError, match="mix the two forms"):
        AnchorTables.load(system, **{**declared, "columns": one_name})
    both = dict(_EOS_COLUMNS, x_column="x_Fe")
    with pytest.raises(ValueError, match="not both"):
        AnchorTables.load(system, **{**declared, "eos_columns": both})


def test_the_vector_form_needs_the_active_column_and_refuses_the_scalar_bulk_test(ternary):
    system, declared = ternary
    columns = {k: v for k, v in _COLUMNS.items() if k != "active"}
    with pytest.raises(KeyError, match="active"):
        AnchorTables.load(system, **{**declared, "columns": columns})
    with pytest.raises(ValueError, match="scalar one with no meaning"):
        AnchorTables.load(system, **{**declared,
                                     "columns": dict(_COLUMNS, bulk_sigma="relative_error")})
    with pytest.raises(ValueError, match="has 2 names against 3 density channels"):
        AnchorTables.load(system, **{**declared, "eos_columns": dict(
            _EOS_COLUMNS, x_columns=("x_Fe", "x_Cu"))})


def test_the_vector_form_is_refused_at_two_channels(tmp_path):
    """Two channels keep one name and the scalar path, so there is one code path per count."""
    from test_train_anchors import _COLUMNS as BINARY, _EOS_COLUMNS as BINARY_EOS
    from test_train_anchors import _declared as binary_declared, _demo, _write_tables

    _write_tables(tmp_path / "tier")
    columns = dict(BINARY, composition=("fraction", "fraction"), active="usable")
    eos = {k: v for k, v in BINARY_EOS.items() if k != "x_column"}
    eos["x_columns"] = ("fraction", "fraction")
    with pytest.raises(ValueError, match="2 channels declare one composition name"):
        AnchorTables.load(_demo(tmp_path), **{**binary_declared(), "columns": columns,
                                              "eos_columns": eos})
    with pytest.raises(ValueError, match="read by the vector form only"):
        AnchorTables.load(_demo(tmp_path), **{**binary_declared(),
                                              "columns": dict(BINARY, active="usable")})


@pytest.mark.parametrize("what", ["x > 0", "rho > 0"])
def test_a_row_whose_mask_contradicts_its_fractions_or_densities_is_refused(tmp_path, what):
    rho = _X * _TOTAL[:, None]
    active = _X > 0
    if what == "x > 0":
        active = active.copy()
        active[3, 0] = True                             # claims an absent species is present
    else:
        rho = rho.copy()
        rho[2, 1] = 1e-3                                # a density where the fraction is zero
    _write(tmp_path / "tier", rho=rho, active=active)
    with pytest.raises(ValueError, match=f"m_table.npz: row {3 if what == 'x > 0' else 2} "
                                         f".*{what} is"):
        AnchorTables.load(_ternary(tmp_path), **_declared())


def test_a_non_finite_active_mobility_entry_is_refused_and_an_inactive_one_ignored(tmp_path):
    transport = np.full((5, 3, 3), 7.0)                 # whatever the absent entries hold
    for r in range(5):
        p = np.nonzero(_X[r] > 0)[0]
        transport[r][np.ix_(p, p)] = 0.1 * _spd_block(p.size, 100 + r)
    transport[2, 1, 1] = np.nan                         # absent channel: ignored
    _write(tmp_path / "tier", transport=transport)
    tables = AnchorTables.load(_ternary(tmp_path), **_declared())
    assert bool(torch.isfinite(tables.mobility_target).all())
    assert float(tables.mobility_target[1, 0, 1]) == 0.0
    transport[3, 1, 2] = np.inf                         # active block: refused
    _write(tmp_path / "tier", transport=transport)
    with pytest.raises(ValueError, match="row 3 has a non-finite entry"):
        AnchorTables.load(_ternary(tmp_path), **_declared())


# ---------------------------------------------------------------------------
# a toy ternary recovers a known H(0) through L_bulk alone
# ---------------------------------------------------------------------------
_H_STAR = np.array([[3.0, 0.5, -0.4],
                    [0.5, 2.0, 0.3],
                    [-0.4, 0.3, 4.0]])


def test_a_toy_ternary_recovers_a_known_zero_wavenumber_curvature(tmp_path):
    """Exact ``S_cc(0)`` targets from ``H_star`` on four ternary compositions (pairwise non-parallel, so
    ``H^-1`` is fixed on every direction) and three binary ones; ``H(0)`` a free SPD matrix, trained on
    ``L_bulk`` alone. Tolerance: ``1e-4`` relative on every row's ``S_cc(0)`` (Frobenius), ``1e-3`` on
    ``H(0)``."""
    x = np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2], [0.1, 0.7, 0.2], [0.34, 0.33, 0.33],
                  [0.5, 0.0, 0.5], [0.0, 0.4, 0.6], [0.7, 0.3, 0.0]])
    n_rows = x.shape[0]
    T = np.linspace(0.8, 1.6, n_rows)
    total = np.linspace(0.4, 0.7, n_rows)
    target = np.full((n_rows, 2, 2), np.nan)
    for r in range(n_rows):
        p = np.nonzero(x[r] > 0)[0]
        m = p.size - 1
        Z = np.stack([np.eye(3)[i] - x[r, i] * (x[r] > 0) for i in p[:-1]], axis=1)[p]
        target[r, :m, :m] = (_KB * T[r] / total[r]
                             * Z.T @ np.linalg.solve(_H_STAR[np.ix_(p, p)], Z))
    rho = x * total[:, None]
    table = dict(usable=np.ones(n_rows, bool), temperature=T, present=x > 0,
                 density_a=rho[:, 0], density_b=rho[:, 1], density_c=rho[:, 2],
                 fraction_a=x[:, 0], fraction_b=x[:, 1], fraction_c=x[:, 2],
                 transport=np.tile(np.eye(3), (n_rows, 1, 1)),
                 wavenumbers=np.tile(_SHELLS, (n_rows, 1)),
                 structure_factor=np.tile(np.eye(3), (n_rows, 3, 1, 1)),
                 zero_wavenumber=target, relative_error=np.full(n_rows, 0.05))
    (tmp_path / "tier").mkdir()
    np.savez(tmp_path / "tier/m_table.npz", **table)
    np.savez(tmp_path / "tier/s_table.npz", **table)
    (tmp_path / "tier/manifold.csv").write_text(
        _EOS_HEADER + "\n" + "".join(f"c,{a},{b},{c},1.0,0,0.5,0,2,ok\n" for a, b, c in x))
    t = AnchorTables.load(_ternary(tmp_path), **_declared())
    assert t.rows()["anchor_bulk"] == n_rows

    d = {name: getattr(t, name).double() for name in
         ("bulk_T", "bulk_zvec", "bulk_rho_tot", "bulk_target", "bulk_target_rootinv")}
    L = torch.nn.Parameter(torch.eye(3, dtype=torch.float64))

    def H0():
        lower = torch.tril(L)
        return (lower @ lower.T).expand(n_rows, 3, 3)

    def loss():
        return l_bulk(H0(), _KB * d["bulk_T"], d["bulk_zvec"], d["bulk_rho_tot"],
                      d["bulk_target"], residual="s_metric",
                      target_rootinv=d["bulk_target_rootinv"], active=t.bulk_active,
                      zmask=t.bulk_zmask, row_weight=t.bulk_weight)

    opt = torch.optim.LBFGS([L], lr=1.0, max_iter=500, tolerance_grad=1e-14,
                            tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        value = loss()
        value.backward()
        return value

    for _ in range(4):
        opt.step(closure)
    H = H0()[0].detach().numpy()
    for r in range(n_rows):
        p = np.nonzero(x[r] > 0)[0]
        m = p.size - 1
        Z = d["bulk_zvec"][r].numpy()[p][:, :m]
        S = _KB * T[r] / total[r] * Z.T @ np.linalg.solve(H[np.ix_(p, p)], Z)
        assert np.linalg.norm(S - target[r, :m, :m]) <= 1e-4 * np.linalg.norm(
            target[r, :m, :m]), r
    assert np.linalg.norm(H - _H_STAR) <= 1e-3 * np.linalg.norm(_H_STAR)
