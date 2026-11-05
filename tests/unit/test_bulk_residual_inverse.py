"""``bulk_residual = "s_metric_inverse"``: the concentration block scored in stiffness space.

``l_r = ||T S_model^-1 T - I||_F^2`` with ``T = S_target^1/2`` fixed at load and
``S_model^-1 = (rho_tot / kBT) (Z^T H_PP(0)^-1 Z)^-1``, the inverse taken as the Schur complement of
``A = Q^-1 H Q^-T`` on the basis ``Q = [Z  1_P]`` (:func:`aipf.losses.static.concentration_stiffness`),
so no ``H`` is inverted.

What is pinned: ``m = 1`` is ``relative_inverse`` row by row (no clamp), on positive and negative
``z^T H^-1 z``; ``m = 2`` is an independent numpy computation on positive-definite and indefinite
``H``; on a path ``H(t)`` whose concentration stiffness crosses zero the residual and its gradient
stay finite and continuous while ``s_metric`` diverges; a binary row of a ternary table reads its
active block alone (values and gradient); the loader's ``target_root``; the gradients through the
model; the refusals; the config, the early declaration check and the checkpoint round trip.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from aipf.functional.build import build
from aipf.losses.static import (_pairs, _principal_block, concentration_stiffness,
                                l_bulk, sinv_rootinv, spd_root)
from aipf.train.config import TrainConfig

from test_train_anchors_vector import (_ACTIVE, _KB, _block_batch, _demo_scratch_is_its_own,  # noqa: F401
                                       _spd_t, loaded, ternary)

_F64 = torch.float64


def _rows(x, active):
    """``(zvec, zmask)`` of a batch of rows, as the loader builds them (:func:`_block_batch`)."""
    parts = [_block_batch(None, x[r], active[r]) for r in range(x.shape[0])]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def _fractions(active, seed):
    g = torch.Generator().manual_seed(seed)
    x = (0.1 + torch.rand(active.shape, generator=g, dtype=_F64)) * active
    return x / x.sum(-1, keepdim=True)


def _sym(n, batch, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, n, n, generator=g, dtype=_F64)
    return a + a.transpose(-1, -2)


def _per_row(H0, kBT, zvec, rho_tot, target, **kw):
    return torch.stack([l_bulk(H0[r:r + 1], kBT[r:r + 1], zvec[r:r + 1], rho_tot[r:r + 1],
                               target[r:r + 1], **{k: v[r:r + 1] for k, v in kw.items()},
                               residual="s_metric_inverse")
                        for r in range(H0.shape[0])])


def _padded(blocks, m_rows, size):
    """``(P, size, size)``: each row's leading ``m x m`` block, the identity beyond."""
    out = torch.eye(size, dtype=_F64).repeat(len(blocks), 1, 1)
    for r, (block, m) in enumerate(zip(blocks, m_rows)):
        out[r, :m, :m] = block
    return out


# ---------------------------------------------------------------------------
# the Schur route
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spd", [True, False])
def test_the_schur_complement_is_the_inverse_of_z_h_inverse_z(spd):
    active = torch.tensor([[True, True, True]] * 4 + [[True, False, True], [False, True, True]])
    x = _fractions(active, 1)
    zvec, zmask = _rows(x, active)
    H = _spd_t(3, 6, 2) if spd else _sym(3, 6, 2)
    K = concentration_stiffness(_principal_block(H, _pairs(active)), zvec, active, zmask)
    for r in range(6):
        p = torch.nonzero(active[r]).flatten()
        m = p.numel() - 1
        Z = zvec[r][p][:, :m]
        want = torch.linalg.inv(Z.T @ torch.linalg.solve(H[r][p][:, p], Z))
        assert torch.allclose(K[r, :m, :m], want, rtol=1e-10, atol=1e-12), r
        assert torch.equal(K[r, m:, m:], torch.eye(2 - m, dtype=_F64))
        assert float(K[r, :m, m:].abs().sum()) == 0.0


def test_the_basis_is_invertible_whatever_the_fractions():
    """``1_P`` is outside ``span Z`` for any fractions, even ones that do not sum to one on ``P``, and
    ``det Q = +-1``."""
    active = torch.tensor([[True, True, True], [True, False, True]])
    x = torch.tensor([[0.0, 0.0, 1.0], [5.0, 0.0, -3.0]], dtype=_F64)    # degenerate on purpose
    zvec, zmask = _rows(x, active)
    H = _spd_t(3, 2, 3)
    K = concentration_stiffness(_principal_block(H, _pairs(active)), zvec, active, zmask)
    for r in range(2):
        p = torch.nonzero(active[r]).flatten()
        m = p.numel() - 1
        Z = zvec[r][p][:, :m]
        Q = torch.cat([Z, torch.ones(p.numel(), 1, dtype=_F64)], -1)
        assert abs(abs(float(torch.linalg.det(Q))) - 1.0) < 1e-12
        want = torch.linalg.inv(Z.T @ torch.linalg.solve(H[r][p][:, p], Z))
        assert torch.allclose(K[r, :m, :m], want, rtol=1e-10)


def test_a_zmask_that_does_not_match_the_active_mask_is_refused():
    active = torch.tensor([[True, False, True]])
    zvec, _ = _rows(_fractions(active, 0), active)
    with pytest.raises(ValueError, match="zmask marks m"):
        concentration_stiffness(torch.eye(3, dtype=_F64)[None], zvec, active,
                                torch.ones(1, 2, dtype=torch.bool))


# ---------------------------------------------------------------------------
# m = 1 and m = 2 against independent computations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sign", ["positive", "negative", "mixed"])
def test_one_species_pair_is_relative_inverse_row_by_row(sign):
    P = 12
    active = torch.ones(P, 2, dtype=torch.bool)
    x = _fractions(active, 4)
    zvec, zmask = _rows(x, active)
    H0 = _spd_t(2, P, 5) if sign == "positive" else _sym(2, P, 5)
    if sign == "negative":
        H0 = -_spd_t(2, P, 5)
    g = torch.Generator().manual_seed(6)
    kBT = 0.5 + torch.rand(P, generator=g, dtype=_F64)
    rho_tot = 0.5 + torch.rand(P, generator=g, dtype=_F64)
    target = 0.05 + torch.rand(P, generator=g, dtype=_F64)
    q = (zvec[:, :, 0] * torch.linalg.solve(H0, zvec[:, :, 0].unsqueeze(-1)).squeeze(-1)).sum(-1)
    if sign == "negative":
        assert bool((q < 0).all())
    if sign == "mixed":
        assert bool((q < 0).any()) and bool((q > 0).any())
    got = _per_row(H0, kBT, zvec, rho_tot, target[:, None, None],
                   target_root=target.sqrt()[:, None, None],
                   active=active, zmask=zmask)
    # the scalar batch: zvec = (x1, -(1 - x1)), the block's own column
    scalar = zvec[:, :, 0]
    assert torch.allclose(scalar, torch.stack([x[:, 1], -(1 - x[:, 1])], -1))
    want = torch.stack([l_bulk(H0[r:r + 1], kBT[r:r + 1], scalar[r:r + 1], rho_tot[r:r + 1],
                               target[r:r + 1], residual="relative_inverse", jitter=0.0)
                        for r in range(P)])
    assert torch.allclose(got, want, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("spd", [True, False])
def test_the_ternary_block_is_the_numpy_residual(spd):
    P = 10
    active = torch.ones(P, 3, dtype=torch.bool)
    x = _fractions(active, 7)
    zvec, zmask = _rows(x, active)
    if spd:
        H0 = _spd_t(3, P, 8)
    else:
        # indefinite, every eigenvalue at least 0.5 away from zero
        g = torch.Generator().manual_seed(8)
        V = torch.linalg.qr(torch.randn(P, 3, 3, generator=g, dtype=_F64))[0]
        lam = torch.tensor([-2.0, 0.7, 3.0], dtype=_F64) * (1 + torch.rand(P, 3, generator=g,
                                                                           dtype=_F64))
        H0 = V @ torch.diag_embed(lam) @ V.transpose(-1, -2)
    g = torch.Generator().manual_seed(9)
    kBT = 0.5 + torch.rand(P, generator=g, dtype=_F64)
    rho_tot = 0.5 + torch.rand(P, generator=g, dtype=_F64)
    target = 0.05 * _spd_t(2, P, 10)
    root = spd_root(target)
    got = _per_row(H0, kBT, zvec, rho_tot, target, target_root=root, active=active, zmask=zmask)
    for r in range(P):
        Z = zvec[r].numpy()
        S = float(kBT[r] / rho_tot[r]) * Z.T @ np.linalg.inv(H0[r].numpy()) @ Z
        w, v = np.linalg.eigh(target[r].numpy())
        half = v @ np.diag(np.sqrt(w)) @ v.T
        want = np.sum((half @ np.linalg.inv(S) @ half - np.eye(2)) ** 2)
        assert float(got[r]) == pytest.approx(want, rel=1e-9), r


def test_row_weights_average_the_rows_as_s_metric_does():
    P = 5
    active = torch.ones(P, 3, dtype=torch.bool)
    zvec, zmask = _rows(_fractions(active, 11), active)
    H0, kBT, rho = _spd_t(3, P, 12), torch.ones(P, dtype=_F64), torch.ones(P, dtype=_F64)
    target = 0.05 * _spd_t(2, P, 13)
    w = torch.tensor([1.0, 2.0, 0.5, 0.0, 3.0], dtype=_F64)
    per = _per_row(H0, kBT, zvec, rho, target, target_root=spd_root(target),
                   active=active, zmask=zmask)
    got = l_bulk(H0, kBT, zvec, rho, target, residual="s_metric_inverse",
                 target_root=spd_root(target), active=active, zmask=zmask, row_weight=w)
    assert float(got) == pytest.approx(float((w * per).sum() / w.sum()), rel=1e-12)
    plain = l_bulk(H0, kBT, zvec, rho, target, residual="s_metric_inverse",
                   target_root=spd_root(target), active=active, zmask=zmask)
    assert float(plain) == pytest.approx(float(per.mean()), rel=1e-12)


# ---------------------------------------------------------------------------
# through a concentration singularity
# ---------------------------------------------------------------------------
def _crossing_path(t):
    """``H(t) = Q A(t) Q^T`` on one ternary row: ``A_uu = 50``, and the concentration stiffness
    ``K(t) = A_ZZ - A_Zu A_uZ / A_uu`` has eigenvalues ``t`` and ``2`` (plus a rotation), so ``det H`` and
    the lowest concentration eigenvalue cross zero at ``t = 0``."""
    active = torch.ones(1, 3, dtype=torch.bool)
    x = torch.tensor([[0.2, 0.3, 0.5]], dtype=_F64)
    zvec, zmask = _rows(x, active)
    Q = torch.cat([zvec[0], torch.ones(3, 1, dtype=_F64)], -1)
    c, s = np.cos(0.4), np.sin(0.4)
    U = torch.tensor([[c, -s], [s, c]], dtype=_F64)
    K = U @ torch.diag(torch.stack([t, torch.ones((), dtype=_F64) * 2.0])) @ U.T
    b = torch.tensor([[3.0], [-1.0]], dtype=_F64)
    a_uu = torch.tensor([[50.0]], dtype=_F64)
    A = torch.cat([torch.cat([K + b @ b.T / a_uu, b], 1), torch.cat([b.T, a_uu], 1)], 0)
    return (Q @ A @ Q.T)[None], zvec, zmask, active


def test_the_residual_is_smooth_where_the_concentration_stiffness_crosses_zero():
    ts = torch.arange(-200, 201, dtype=_F64) * 1e-3              # t = 0 on the grid
    assert float(ts[200]) == 0.0
    target = torch.tensor([[[0.3, 0.05], [0.05, 0.6]]], dtype=_F64)
    one = torch.ones(1, dtype=_F64)
    values, grads, s_metric = [], [], []
    for t in ts:
        t = t.clone().requires_grad_(True)
        H, zvec, zmask, active = _crossing_path(t)
        H.retain_grad()
        loss = l_bulk(H, one, zvec, one, target, residual="s_metric_inverse",
                      target_root=spd_root(target), active=active, zmask=zmask)
        loss.backward()
        assert bool(torch.isfinite(H.grad).all())
        values.append(float(loss))
        grads.append(float(t.grad))
        with torch.no_grad():
            s_metric.append(float(l_bulk(H, one, zvec, one, target, residual="s_metric",
                                         target_rootinv=sinv_rootinv(target), active=active,
                                         zmask=zmask)))
    values, grads = np.array(values), np.array(grads)
    assert np.isfinite(values).all() and np.isfinite(grads).all()
    # continuous, with a continuous derivative: neighbouring steps differ by O(dt)
    dt = float(ts[1] - ts[0])
    assert np.abs(np.diff(values)).max() < 50 * dt
    assert np.abs(np.diff(grads)).max() < 50 * dt
    # the stiffness is linear in t, so the residual is a quadratic polynomial in t: exactly
    fit = np.polyfit(ts.numpy(), values, 2)
    assert np.abs(np.polyval(fit, ts.numpy()) - values).max() < 1e-9
    assert np.allclose(grads, np.polyval(np.polyder(fit), ts.numpy()), atol=1e-8)
    # where s_metric, in S space, has its barrier
    s_metric = np.array(s_metric)
    assert s_metric[200] > 1e8 or not np.isfinite(s_metric[200])
    assert s_metric[199] > 1e3 and s_metric[201] > 1e3
    assert s_metric[0] < 1e3 and s_metric[-1] < 1e3


# ---------------------------------------------------------------------------
# a binary row of a ternary table
# ---------------------------------------------------------------------------
def test_a_binary_row_reads_its_active_block_alone():
    x = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.0, 0.6], [0.0, 0.7, 0.3]], dtype=_F64)
    base = _sym(3, 3, 14)
    kBT = torch.tensor([1.0, 1.3, 0.7], dtype=_F64)
    rho_tot = torch.tensor([0.5, 0.6, 0.7], dtype=_F64)
    for r in range(3):
        H0 = base[r:r + 1].clone()
        idx = torch.nonzero(_ACTIVE[r]).flatten()
        absent = torch.nonzero(~_ACTIVE[r]).flatten()
        H0[0, absent, :] = 1e5                                   # whatever H holds there: unread
        H0[0, :, absent] = -3e4
        H0.requires_grad_(True)
        zvec, zmask = _block_batch(None, x[r], _ACTIVE[r])
        m = int(zmask.sum())
        target = torch.eye(2, dtype=_F64)[None].clone()
        target[0, :m, :m] = 0.04 * _spd_t(m, 1, 15 + r)[0]
        root = torch.eye(2, dtype=_F64)[None].clone()
        root[0, :m, :m] = spd_root(target[0, :m, :m])
        masked = l_bulk(H0, kBT[r:r + 1], zvec, rho_tot[r:r + 1], target,
                        residual="s_metric_inverse", target_root=root,
                        active=_ACTIVE[r:r + 1], zmask=zmask)
        masked.backward()
        n_p = idx.numel()
        block = l_bulk(base[r][idx][:, idx][None], kBT[r:r + 1], zvec[:, idx][:, :, :n_p - 1],
                       rho_tot[r:r + 1], target[:, :m, :m], residual="s_metric_inverse",
                       target_root=root[:, :m, :m],
                       active=torch.ones(1, n_p, dtype=torch.bool),
                       zmask=torch.ones(1, n_p - 1, dtype=torch.bool))
        assert torch.allclose(masked, block, rtol=1e-10), r
        assert bool(torch.isfinite(H0.grad).all())
        if absent.numel():
            assert float(H0.grad[0, absent].abs().sum() + H0.grad[0, :, absent].abs().sum()) == 0.0
        if m == 1:          # and the binary row is relative_inverse on its pair
            pair = zvec[0][idx][:, 0][None]
            want = l_bulk(base[r][idx][:, idx][None], kBT[r:r + 1], pair, rho_tot[r:r + 1],
                          target[:, 0, 0], residual="relative_inverse", jitter=0.0)
            assert torch.allclose(masked, want, rtol=1e-10), r


# ---------------------------------------------------------------------------
# the loader, the model, the refusals
# ---------------------------------------------------------------------------
def test_the_loader_carries_the_targets_square_root_on_its_block(loaded):
    T = loaded.bulk_target_root.double()
    S = loaded.bulk_target.double()
    R = loaded.bulk_target_rootinv.double()
    assert T.shape == S.shape == (3, 2, 2)
    assert torch.allclose(T @ T, S, rtol=1e-5, atol=1e-7)
    assert torch.allclose(T @ R, torch.eye(2, dtype=_F64).expand(3, 2, 2), atol=1e-5)
    for i in (1, 2):                                   # the binary rows: the identity beyond m = 1
        assert torch.equal(loaded.bulk_target_root[i, 1], torch.tensor([0.0, 1.0]))
        assert torch.equal(loaded.bulk_target_root[i, :, 1], torch.tensor([0.0, 1.0]))


def test_losses_and_gradients_through_the_model_are_finite(loaded, ternary):
    from aipf.train.lit_module import LitModule

    torch.manual_seed(0)
    model = build(ternary[0])
    model.train()
    batch = loaded.batch(model, kB=_KB)
    assert torch.equal(batch["anchor_bulk"]["target_root"], loaded.bulk_target_root)
    config = TrainConfig(lambda_bulk=1.0, sigma=1.0, k_max=2.0,
                         bulk_residual="s_metric_inverse", seed=0)
    lit = LitModule(model, model._cache, config)
    model.zero_grad(set_to_none=True)
    total, parts = lit.compute_losses(batch)
    assert bool(torch.isfinite(parts["L_bulk"])) and float(parts["L_bulk"]) > 0.0
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    # the term is l_bulk on the batch, with the table's row weights
    b = batch["anchor_bulk"]
    want = l_bulk(b["H0"].detach(), b["kBT"], b["zvec"], b["rho_tot"], b["target"],
                  residual="s_metric_inverse", target_root=b["target_root"], active=b["active"],
                  zmask=b["zmask"], row_weight=b["row_weight"])
    assert float(parts["L_bulk"]) == pytest.approx(float(want), rel=1e-6)


def test_the_inverse_form_refuses_what_s_metric_refuses(loaded):
    t = loaded
    H0 = torch.eye(3).expand(3, 3, 3)
    with pytest.raises(ValueError, match="needs a matrix batch"):
        l_bulk(torch.eye(2).expand(4, 2, 2), torch.ones(4), torch.ones(4, 2),
               torch.ones(4), torch.ones(4), residual="s_metric_inverse")
    with pytest.raises(ValueError, match=r"no \['target_root'\]"):
        l_bulk(H0, t.bulk_T, t.bulk_zvec, t.bulk_rho_tot, t.bulk_target,
               residual="s_metric_inverse", target_rootinv=t.bulk_target_rootinv,
               active=t.bulk_active, zmask=t.bulk_zmask)
    for extra in ({"sigma": torch.ones(3)}, {"clamp": 3.0}):
        with pytest.raises(ValueError, match="takes neither sigma nor clamp"):
            l_bulk(H0, t.bulk_T, t.bulk_zvec, t.bulk_rho_tot, t.bulk_target,
                   residual="s_metric_inverse", target_root=t.bulk_target_root,
                   active=t.bulk_active, zmask=t.bulk_zmask, **extra)
    for residual in ("relative_inverse", "sigma_chi2"):
        with pytest.raises(ValueError, match="declare 's_metric' or 's_metric_inverse'"):
            l_bulk(H0, t.bulk_T, t.bulk_zvec, t.bulk_rho_tot, t.bulk_target,
                   residual=residual, target_root=t.bulk_target_root,
                   active=t.bulk_active, zmask=t.bulk_zmask)


# ---------------------------------------------------------------------------
# the config, the early check, the checkpoint
# ---------------------------------------------------------------------------
def test_the_config_accepts_the_new_value_and_round_trips():
    cfg = TrainConfig(lambda_bulk=1.0, bulk_residual="s_metric_inverse", seed=0)
    assert TrainConfig.from_dict(cfg.as_dict()) == cfg
    with pytest.raises(ValueError, match="not in"):
        TrainConfig(bulk_residual="s_metric_inv")


def test_a_new_checkpoint_round_trips_the_new_value(tmp_path):
    from aipf.train.ckpt_compat import load_new_checkpoint, save_new_checkpoint

    cfg = TrainConfig(sigma=1.0, lambda_bulk=0.5, bulk_residual="s_metric_inverse", seed=0)
    save_new_checkpoint(tmp_path / "c.ckpt", torch.nn.Linear(2, 2), cfg)
    _state, again = load_new_checkpoint(tmp_path / "c.ckpt")
    assert again == cfg and again.bulk_residual == "s_metric_inverse"


def test_fit_accepts_either_block_form_on_vector_tables_and_neither_on_scalar(ternary):
    from aipf.train.fit import _check_bulk_residual

    system, declared = ternary
    system = dataclasses.replace(system, defaults={**system.defaults,
                                                   "training": {"tables": declared}})
    inverse = TrainConfig(lambda_bulk=1.0, bulk_residual="s_metric_inverse", seed=0)
    _check_bulk_residual(system, inverse, None)
    _check_bulk_residual(system, TrainConfig(lambda_bulk=1.0, bulk_residual="s_metric",
                                             seed=0), None)
    for scalar in ("relative_inverse", "sigma_chi2"):
        with pytest.raises(ValueError, match="vector-form anchor tables"):
            _check_bulk_residual(system, TrainConfig(lambda_bulk=1.0, bulk_residual=scalar,
                                                     seed=0), None)
    from test_train_anchors import _declared as binary_declared, _demo
    binary = dataclasses.replace(_demo(ternary[0].paths.raw()), defaults={
        "k_fit_stat": 1.5, "training": {"tables": binary_declared()}})
    with pytest.raises(ValueError, match="on scalar anchor tables"):
        _check_bulk_residual(binary, inverse, None)


def test_a_fit_trains_the_inverse_form_and_its_checkpoint_records_it(tmp_path):
    import json

    from test_three_species_training import _system, _write_modes, _write_tables, _GRID
    from aipf.train.datamodule import SourceSpec
    from aipf.train.fit import fit

    _write_tables(tmp_path / "tier")
    tree = tmp_path / "fields" / "modes_toy"
    _write_modes(tree / "run_a", np.random.default_rng(0))
    _write_modes(tree / "run_b", np.random.default_rng(1), absent=1)
    sources = (SourceSpec(name="toy", root=str(tree), pattern="run_*", grid=_GRID,
                          loss_weight=1.0, exclude_tags=()),)
    run = fit(_system(tmp_path), run_name="inverse", sources=sources, steps=2, seed=0,
              resume_optimizer=False, root=tmp_path / "data", log_every_step=True,
              config_overrides={"bulk_residual": "s_metric_inverse", "lr": 1e3})
    steps = json.loads((run / "steps.json").read_text())
    assert all(np.isfinite(step["L_bulk"]) and step["L_bulk"] > 0 for step in steps["terms"])
    ck = torch.load(run / "final.ckpt", map_location="cpu", weights_only=False)
    assert ck["config"]["bulk_residual"] == "s_metric_inverse"
    assert TrainConfig.from_dict(ck["config"]).bulk_residual == "s_metric_inverse"
