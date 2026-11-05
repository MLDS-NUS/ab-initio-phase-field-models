"""``L_conv`` on the anchor tables' single-phase states (``conv_states="anchors"``): the active-block
eigenvalue on a wavenumber line, its ``k = 0`` slice, the refusals and the declaration.

The toy of the first part has a known curvature and a known ``W_hat(k)`` and its bulk entry is written
by hand, with ternary, binary and pure active patterns. The second part reads the ternary toy tables of
``test_three_species_training.py`` through ``AnchorTables.batch`` with the Fe-B-shaped model; the step
that trains it is in that file. The trust-domain ``L_conv`` is ``test_train_penalties.py``.
"""
import dataclasses

import numpy as np
import pytest
import torch

from aipf.functional.build import build
from aipf.losses.convexity import l_conv
from aipf.losses.names import WEIGHT_MAP_A
from aipf.train.anchors import NO_ANCHORS, AnchorTables
from aipf.train.ckpt_compat import (IMPLICIT_WEIGHTS_A, NONLOSS_KEY_MAP_LIGHTNING,
                                    hparams_to_train_config)
from aipf.train.config import TrainConfig
from aipf.train.datamodule import SourceSpec
from aipf.train.fit import _config_from_system, fit
from aipf.train.penalties import (STATE_KEYS, ConvexityProbe, Penalties, StateStability,
                                  conv_line, conv_states, declared_fields,
                                  penalties_from_system)

import test_three_species_training as t3

_KB = 8.617333262e-5
_ANCHORS = {"conv_states": "anchors", "conv_k": (8, 3.0)}


# ---------------------------------------------------------------------------
# a toy with a known curvature and kernel, its bulk entry written by hand
# ---------------------------------------------------------------------------

class _Kernel(torch.nn.Module):
    """``W(k) = M0 + k^2 M2``, symmetric, with a trainable scale."""

    n_species = 3

    def __init__(self, M0, M2):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
        self.M0, self.M2 = M0, M2

    def w_hat(self, k):
        return self.scale * (self.M0 + (k * k)[..., None, None] * self.M2)


class _Model(torch.nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel


#: Ternary, binary (Cu absent), pure Ni, binary (Ni absent), ternary.
_ACTIVE = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)


def _sym(rng, n=3, scale=1.0):
    a = rng.standard_normal((n, n))
    return scale * 0.5 * (a + a.T)


def _toy(seed=0, *, curvature_shift=0.0, absent=-1e3):
    """``(model, bulk, local, M0, M2)``: a random symmetric curvature per row, shifted by
    ``curvature_shift`` on the diagonal, and ``absent`` on an absent channel's diagonal (what the
    density clamp leaves there: anything, and here very negative)."""
    rng = np.random.default_rng(seed)
    M0, M2 = _sym(rng), _sym(rng, scale=0.3)
    local = np.stack([_sym(rng) + curvature_shift * np.eye(3) for _ in range(_ACTIVE.shape[0])])
    for r, row in enumerate(_ACTIVE.numpy()):
        for i in np.nonzero(~row)[0]:
            local[r, i, i] = absent
    model = _Model(_Kernel(torch.from_numpy(M0), torch.from_numpy(M2)))
    local_t = torch.from_numpy(local)
    bulk = {"curvature": local_t, "H0": local_t + model.kernel.w_hat(torch.zeros(1))[0]}
    return model, bulk, local, M0, M2


def _probe(n_k=8, k_max=3.0, margin=0.05, form="hinge"):
    return StateStability(n_k=n_k, k_max=k_max, form=form, margin=margin, active=_ACTIVE)


def _numpy_eigmin(local, M0, M2, k):
    """``(rows, n_k)``: numpy's ``lambda_min`` of each non-pure row's active block, at each ``k``."""
    out, rows = [], []
    for r, row in enumerate(_ACTIVE.numpy()):
        p = np.nonzero(row)[0]
        if p.size < 2:
            continue
        H = local[r][None] + M0[None] + (k * k)[:, None, None] * M2[None]
        out.append(np.linalg.eigvalsh(H[:, p[:, None], p])[:, 0])
        rows.append(r)
    return np.array(out), np.array(rows)


def test_the_active_block_eigenvalues_are_numpys_and_the_pure_row_is_left_out():
    model, bulk, local, M0, M2 = _toy()
    lam, rows, k = _probe().eigmin(model, bulk)
    expected, expected_rows = _numpy_eigmin(local, M0, M2, k.numpy())
    assert rows.tolist() == expected_rows.tolist() == [0, 1, 3, 4]
    assert lam.shape == (4, 8)
    assert np.allclose(lam.detach().numpy(), expected, rtol=0.0, atol=1e-12)


def test_the_line_starts_at_zero_and_reaches_k_max():
    k = _probe(n_k=32, k_max=3.96).wavenumbers(dtype=torch.float64)
    assert k.shape == (32,) and float(k[0]) == 0.0 and float(k[-1]) == pytest.approx(3.96)


def test_an_absent_channel_never_enters():
    """Whatever the absent channel's curvature row holds, the eigenvalues do not move; a full-matrix
    minimum would read the clamp's -1000."""
    model, bulk, *_ = _toy(absent=-1e3)
    lam, _, _ = _probe().eigmin(model, bulk)
    other, other_bulk, *_ = _toy(absent=+7.0)
    for r, row in enumerate(_ACTIVE.numpy()):          # absent rows and columns: random, not symmetric
        for i in np.nonzero(~row)[0]:
            noise = torch.from_numpy(np.random.default_rng(r).standard_normal(3))
            for key in ("curvature", "H0"):
                other_bulk[key][r, i, :] += noise
                other_bulk[key][r, :, i] -= 2.0 * noise
    lam_other, _, _ = _probe().eigmin(other, other_bulk)
    assert torch.equal(lam, lam_other)
    assert float(lam.min()) > -100.0
    full = torch.linalg.eigvalsh(0.5 * (bulk["H0"] + bulk["H0"].transpose(-1, -2)))[:, 0]
    assert float(full[1]) < -900.0


def test_the_zero_slice_is_the_bulk_anchors_own_h0_whatever_its_route():
    """A ``W_hat(0)`` route other than the evaluator puts another ``W(0)`` in ``H0``: the ``k = 0``
    slice reads it, the line beyond reads the evaluator."""
    model, bulk, local, M0, M2 = _toy()
    shifted = dict(bulk, H0=bulk["H0"] + 0.37 * torch.eye(3, dtype=torch.float64))
    lam, rows, k = _probe().eigmin(model, shifted)
    expected, _ = _numpy_eigmin(local, M0, M2, k.numpy())
    for j, r in enumerate(rows.tolist()):
        p = np.nonzero(_ACTIVE[r].numpy())[0]
        H0 = shifted["H0"][r].detach().numpy()[np.ix_(p, p)]
        assert float(lam[j, 0]) == pytest.approx(np.linalg.eigvalsh(H0)[0], abs=1e-12)
    assert np.allclose(lam[:, 1:].detach().numpy(), expected[:, 1:], atol=1e-12)


@pytest.mark.parametrize("form", ["hinge", "hinge0", "softplus"])
def test_the_forms_and_the_aux_are_l_convs(form):
    model, bulk, *_ = _toy()
    probe = _probe(form=form, margin=0.3)
    lam, _, _ = probe.eigmin(model, bulk)
    loss, aux = probe.loss(model, bulk)
    # l_conv on diagonal matrices whose smallest eigenvalue is lam: the same numbers
    diag = torch.diag_embed(torch.stack([lam, lam + 1.0], dim=-1))
    ref_loss, ref_aux = l_conv(diag, form=form, margin=0.3)
    assert float(loss) == pytest.approx(float(ref_loss), rel=1e-12)
    assert aux == pytest.approx(ref_aux)
    assert set(aux) == {"viol_frac", "lam_worst"}


def test_a_stable_toy_scores_zero_and_an_unstable_one_a_finite_positive_gradient():
    model, bulk, *_ = _toy(curvature_shift=50.0)
    loss, aux = _probe().loss(model, bulk)
    assert float(loss) == 0.0 and aux["viol_frac"] == 0.0

    model, bulk, *_ = _toy(curvature_shift=-1.0)
    loss, aux = _probe().loss(model, bulk)
    assert float(loss) > 0.0 and aux["viol_frac"] > 0.0 and aux["lam_worst"] < 0.0
    loss.backward()
    grad = model.kernel.scale.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad) != 0.0


def test_a_bulk_entry_without_its_curvature_or_of_other_rows_is_refused():
    model, bulk, *_ = _toy()
    with pytest.raises(KeyError, match="curvature=True"):
        _probe().eigmin(model, {"H0": bulk["H0"]})
    short = {key: value[:3] for key, value in bulk.items()}
    with pytest.raises(ValueError, match="rows"):
        _probe().eigmin(model, short)


def test_a_lattice_sum_kernel_is_refused():
    class _Lattice(torch.nn.Module):
        reads_geometry = True

    model = _Model(_Kernel(torch.eye(3), torch.eye(3)))
    model.kernel.evaluator = _Lattice()
    with pytest.raises(NotImplementedError, match="conv_states='trust_domain'"):
        StateStability.check(model)
    StateStability.check(_Model(_Kernel(torch.eye(3), torch.eye(3))))


@pytest.mark.parametrize("n_k, k_max, match", [(1, 3.0, "two wavenumbers"),
                                               (8, 0.0, "positive k_max")])
def test_a_line_too_short_is_refused(n_k, k_max, match):
    with pytest.raises(ValueError, match=match):
        _probe(n_k=n_k, k_max=k_max)


# ---------------------------------------------------------------------------
# through the anchor tables and the Fe-B-shaped model
# ---------------------------------------------------------------------------

@pytest.fixture
def ternary(tmp_path):
    t3._write_tables(tmp_path / "tier")
    system = t3._system(tmp_path)
    tables = AnchorTables.load(system, **system.defaults["training"]["tables"])
    torch.manual_seed(0)
    return system, tables, build(system)


def test_the_batch_hands_over_the_curvature_its_h0_is_built_from(ternary):
    _, tables, model = ternary
    plain = tables.batch(model, _KB)
    batch = tables.batch(model, _KB, curvature=True)
    assert "curvature" not in plain["anchor_bulk"]
    assert set(batch["anchor_bulk"]) == set(plain["anchor_bulk"]) | {"curvature"}
    for key in ("H0", "zvec", "target"):
        assert torch.equal(batch["anchor_bulk"][key], plain["anchor_bulk"][key])
    W0 = model.kernel.w_hat(torch.zeros(1))[0]
    assert torch.allclose(batch["anchor_bulk"]["H0"], batch["anchor_bulk"]["curvature"] + W0,
                          rtol=0.0, atol=0.0)


def test_on_the_ternary_tables_the_line_is_numpys_and_k_zero_is_the_h0_path(ternary):
    _, tables, model = ternary
    probe = StateStability.from_tables(tables, n_k=6, k_max=3.96, form="hinge", margin=0.05)
    bulk = tables.batch(model, _KB, curvature=True)["anchor_bulk"]
    lam, rows, k = probe.eigmin(model, bulk)
    # the loader dropped the pure row: a ternary row and a binary row (Cu absent)
    assert rows.tolist() == [0, 1] and torch.equal(probe.active, tables.bulk_active)
    H0 = bulk["H0"].detach().double().numpy()
    local = bulk["curvature"].detach().double().numpy()
    W = model.kernel.w_hat(k).detach().double().numpy()
    for j, r in enumerate(rows.tolist()):
        p = np.nonzero(tables.bulk_active[r].numpy())[0]
        sym0 = 0.5 * (H0[r] + H0[r].T)
        assert float(lam[j, 0]) == pytest.approx(np.linalg.eigvalsh(sym0[np.ix_(p, p)])[0],
                                                 rel=1e-5, abs=1e-6)
        for q in range(1, k.numel()):
            H = local[r] + W[q]
            H = 0.5 * (H + H.T)
            assert float(lam[j, q]) == pytest.approx(np.linalg.eigvalsh(H[np.ix_(p, p)])[0],
                                                     rel=1e-5, abs=1e-6)
    assert 1 not in np.nonzero(tables.bulk_active[1].numpy())[0]


def test_on_the_ternary_tables_the_loss_and_its_gradients_are_finite(ternary):
    _, tables, model = ternary
    probe = StateStability.from_tables(tables, n_k=8, k_max=3.96, form="hinge", margin=1e3)
    loss, aux = probe.loss(model, tables.batch(model, _KB, curvature=True)["anchor_bulk"])
    assert float(loss) > 0.0 and np.isfinite(aux["lam_worst"])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


def test_tables_without_a_bulk_row_are_refused(ternary):
    _, tables, _ = ternary
    empty = dataclasses.replace(tables, bulk_rho=tables.bulk_rho[:0],
                                bulk_active=tables.bulk_active[:0])
    with pytest.raises(ValueError, match="anchor_bulk rows"):
        StateStability.from_tables(empty, n_k=8, k_max=3.0, form="hinge", margin=0.05)


def test_two_channel_tables_read_the_positive_densities_as_active():
    tables = type("T", (), {"bulk_active": None,
                            "bulk_rho": torch.tensor([[0.3, 0.2], [0.0, 0.5]])})()
    probe = StateStability.from_tables(tables, n_k=4, k_max=2.0, form="hinge", margin=0.0)
    assert probe.active.tolist() == [[True, True], [False, True]]
    assert probe.provenance()["rows"] == 1


# ---------------------------------------------------------------------------
# the declaration, the refusals and the record
# ---------------------------------------------------------------------------

def _cfg(**extra):
    return TrainConfig(lambda_conv=1.0, conv_margin=0.05,
                       extra_experiment_config={**_ANCHORS, **extra})


def test_the_keys_map_where_a_saved_checkpoint_does_and_round_trip():
    for key in STATE_KEYS:
        assert NONLOSS_KEY_MAP_LIGHTNING[key] == "extra_experiment_config", key
    assert declared_fields({"lambda_M": 0.5}) == {}
    assert declared_fields({**_ANCHORS, "lr": 1e-3}) == {
        "extra_experiment_config": {"conv_states": "anchors", "conv_k": [8, 3.0]}}
    cfg = hparams_to_train_config({**_ANCHORS, "lambda_conv": 1.0, "conv_margin": 0.05},
                                  weight_map=WEIGHT_MAP_A, key_map=NONLOSS_KEY_MAP_LIGHTNING,
                                  implicit_weights=IMPLICIT_WEIGHTS_A)
    assert cfg == _cfg()
    assert conv_states(cfg) == "anchors"
    assert TrainConfig.from_dict(cfg.as_dict()) == cfg


def test_an_undeclared_source_is_the_trust_domain_and_other_values_are_refused():
    assert conv_states(TrainConfig()) == "trust_domain"
    with pytest.raises(ValueError, match="conv_states='somewhere'"):
        conv_states(TrainConfig(extra_experiment_config={"conv_states": "somewhere"}))
    with pytest.raises(ValueError, match="conv_k"):
        conv_states(TrainConfig(extra_experiment_config={"conv_k": (8, 3.0)}))


def test_the_anchors_source_without_tables_is_refused_by_name(ternary):
    system, tables, _ = ternary
    with pytest.raises(ValueError, match="loads no anchor tables"):
        penalties_from_system(system, _cfg())
    cfg = TrainConfig(lambda_conv=1.0, extra_experiment_config={"conv_states": "anchors"})
    with pytest.raises(KeyError, match="conv_k"):
        penalties_from_system(system, cfg, tables)


def test_the_anchors_source_needs_no_seed_and_draws_nothing(ternary):
    system, tables, _ = ternary
    penalties = penalties_from_system(system, _cfg(), tables)
    assert isinstance(penalties.conv, StateStability) and penalties.reads_anchors
    assert penalties.seed is None and penalties.terms() == ("L_conv",)
    assert penalties.draw() == (None, None)
    assert penalties.provenance() == {
        "penalty_seed": None, "terms": ["L_conv"], "gamma": None,
        "conv_states": {"states": "anchors", "n_k": 8, "k_max": 3.0, "rows": 2}}
    with pytest.raises(ValueError, match="anchor_bulk"):
        penalties.losses(build(system))


def test_a_drawing_probe_without_a_seed_is_still_refused():
    with pytest.raises(ValueError, match="penalty_seed"):
        Penalties(None, ConvexityProbe(domain=None, n=1, T_measure="uniform", form="hinge",
                                       margin=0.0), None)


def test_the_default_source_without_a_trust_domain_trains_nothing_as_before(ternary):
    system, tables, _ = ternary
    system = dataclasses.replace(system, trust_domain=None)
    assert penalties_from_system(system, TrainConfig(lambda_conv=1.0), tables) is None


def test_fit_refuses_the_anchors_source_with_no_anchors_before_the_run_directory(tmp_path):
    t3._write_tables(tmp_path / "tier")
    tree = tmp_path / "fields" / "modes_toy"
    t3._write_modes(tree / "run_a", np.random.default_rng(0))
    sources = (SourceSpec(name="toy", root=str(tree), pattern="run_*", grid=t3._GRID,
                          loss_weight=1.0, exclude_tags=()),)
    system = t3._system(tmp_path)
    system = dataclasses.replace(system, defaults={**system.defaults, **_ANCHORS,
                                                   "lambda_conv": 1.0})
    assert conv_states(TrainConfig(**_config_from_system(system))) == "anchors"
    with pytest.raises(ValueError, match="--anchors none"):
        fit(system, run_name="refused", sources=sources, steps=1, seed=0,
            resume_optimizer=False, root=tmp_path / "data", anchors=NO_ANCHORS)
    assert not (tmp_path / "data").exists()


# ---------------------------------------------------------------------------
# the declared line, two channels, L_Gamma beside it, two dimensions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [(32, 3.96), [32, 3.96], (32.0, "3.96")])
def test_the_line_is_kept_as_one_list_however_it_is_declared(value):
    assert declared_fields({"conv_k": value}) == {"extra_experiment_config": {"conv_k": [32, 3.96]}}
    assert conv_line(value) == (32, 3.96)


@pytest.mark.parametrize("value", [{"n_k": 32, "k_max": 3.96}, "32,3.96", 32, (32.5, 3.96),
                                   (32, 3.96, 1.0)])
def test_a_line_that_is_not_two_numbers_is_refused_naming_the_key(value):
    with pytest.raises(ValueError, match="conv_k"):
        conv_line(value)


def _two_channel(tmp_path):
    import test_train_anchors as ta
    import test_train_penalties as tp

    ta._write_tables(tmp_path / "tier")
    tp._eos_csv(tmp_path / "eos" / "eos.csv")
    system = ta._demo(tmp_path)
    training = {"penalty_seed": 7, "tables": ta._declared(), "gamma_paths": {
        "eos_csvs": {0.0: "eos/eos.csv"},
        "eos_columns": {"x_column": "x_B", "T_column": "T_K", "n_columns": ("n",),
                        "status_column": "status", "status_ok": "ok"},
        "x_grid": (0.1, 0.9, 0.05), "poly_degree": 2, "T_measure": "uniform", "q_floor": 1e-6}}
    system = dataclasses.replace(system, defaults={**system.defaults, "training": training})
    return system, AnchorTables.load(system, **ta._declared())


def test_two_channel_scalar_tables_read_the_full_block_through_the_batch(tmp_path):
    system, tables = _two_channel(tmp_path)
    assert tables.bulk_active is None and tables.rows()["anchor_bulk"] == 3
    torch.manual_seed(0)
    model = build(system)
    probe = StateStability.from_tables(tables, n_k=5, k_max=3.0, form="hinge", margin=1e6)
    bulk = tables.batch(model, 1.0, curvature=True)["anchor_bulk"]
    lam, rows, k = probe.eigmin(model, bulk)
    assert rows.tolist() == [0, 1, 2]
    local = bulk["curvature"].detach().double().numpy()
    W = model.kernel.w_hat(k).detach().double().numpy()
    H0 = bulk["H0"].detach().double().numpy()
    for r in range(3):
        want = [np.linalg.eigvalsh(0.5 * (H0[r] + H0[r].T))[0]]
        want += [np.linalg.eigvalsh(0.5 * ((local[r] + W[q]) + (local[r] + W[q]).T))[0]
                 for q in range(1, 5)]
        assert np.allclose(lam[r].detach().numpy(), want, rtol=1e-5, atol=1e-6)
    loss, _ = probe.loss(model, bulk)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert float(loss) > 0.0 and grads and all(bool(torch.isfinite(g).all()) for g in grads)


def test_beside_l_gamma_the_seed_is_gammas_and_its_draws_are_its_own(tmp_path):
    system, tables = _two_channel(tmp_path)
    both = TrainConfig(lambda_conv=1.0, lambda_gamma=1.0, gamma_pt_samples=3,
                       extra_experiment_config=_ANCHORS)
    with pytest.raises(KeyError, match=r"\['L_Gamma'\]"):
        penalties_from_system(system, both, tables)
    seeded = dataclasses.replace(both, penalty_seed=7)
    penalties = penalties_from_system(system, seeded, tables)
    assert penalties.terms() == ("L_conv", "L_Gamma") and penalties.reads_anchors
    alone = penalties_from_system(system, dataclasses.replace(seeded, lambda_conv=0.0), tables)
    c, g = penalties.draw()
    _, g_alone = alone.draw()
    assert c is None and torch.equal(g.n_path, g_alone.n_path) and torch.equal(g.T, g_alone.T)


def test_a_two_dimensional_model_is_told_it_trains_no_l_conv(tmp_path):
    from aipf.train.fit import _dimension_refusals

    system, _ = _two_channel(tmp_path)
    system = dataclasses.replace(system, trust_domain=None)
    lines = _dimension_refusals(system, 2, _cfg(), (), "mean", NO_ANCHORS)
    assert any("convexity" in line for line in lines), lines
    assert not any("convexity" in line for line in
                   _dimension_refusals(system, 2, TrainConfig(lambda_conv=1.0), (), "mean",
                                       NO_ANCHORS))


@pytest.mark.parametrize("value", [("abc", 3.0), (8, "x"), (None, 3.0), (float("nan"), 3.0),
                                   (float("inf"), 3.0), (8, float("inf")), (8, True), (True, 3.0)])
def test_a_line_with_an_entry_that_is_not_a_finite_number_is_refused_naming_the_key(value):
    with pytest.raises(ValueError, match="conv_k"):
        conv_line(value)


def test_a_line_declared_as_an_array_is_read():
    assert conv_line(np.array([32, 3.96])) == (32, 3.96)
    assert conv_line(torch.tensor([8.0, 3.0])) == (8, 3.0)
