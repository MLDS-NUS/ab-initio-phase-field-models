"""Tests for the seven canonical losses plus the L_W extra.

Every loss here is a PURE function of already-computed tensors -- a
prediction, a target, and whatever bookkeeping (a band mask, a mode
multiplicity, a padding mask, a row weight) the reduction needs. None of them
take a model, sample a trust domain, load an anchor table, or read a
constant such as k_B or a reference pressure: those are System or caller
concerns (spec: "no system constants, no core defaults"). What is tested
here is the reduction itself -- that each loss is zero exactly at its own
target and strictly positive away from it, that the two `L_bulk` residual
forms are not interchangeable at equal magnitude, and that the anchor row
gate raises rather than silently defaulting.

Every matrix-shaped loss (`L_M`, `L_S`, `L_bulk`, `L_conv`, `L_W`) is tested
at `n_species = 1` and `n_species = 2`, using `torch.linalg` reductions
(`eigvalsh`, `solve`, `det`) that are n-general rather than the 2x2 closed
forms the source trees used, so nothing here is baked to two channels.
`L_dyn` is tested at both species counts too, since its reduction sums over
the species axis. `L_P` and `L_Gamma` operate on tensors that are already
reduced to one value per sample or per path point (a pressure, a Gamma
curve) with no species axis of their own, so a species-count parametrization
would be vacuous for them; this is noted rather than forced.
"""
from __future__ import annotations

import pytest
import torch

from aipf.losses.convexity import gamma_closed_form, l_conv, l_gamma
from aipf.losses.dyn import l_dyn
from aipf.losses.extras import l_w
from aipf.losses.gating import gate_rows
from aipf.losses.mobility import l_m
from aipf.losses.pressure import l_p_absolute, l_p_variance
from aipf.losses.static import l_bulk, l_s, sinv_rootinv

torch.manual_seed(0)


def _spd(n_species: int, batch: int = 5, extra_shape: tuple[int, ...] = ()) -> torch.Tensor:
    """A batch of random symmetric positive-definite n_species x n_species
    matrices, shaped (*extra_shape, batch, n_species, n_species)."""
    shape = (*extra_shape, batch, n_species, n_species)
    A = torch.randn(shape, dtype=torch.float64)
    A = A @ A.transpose(-1, -2)
    eye = torch.eye(n_species, dtype=torch.float64)
    return A + n_species * eye  # keep well away from singular


# ---------------------------------------------------------------------------
# L_dyn
# ---------------------------------------------------------------------------

def _dyn_ingredients(n_species, grid=(4, 3, 5), batch=3, alpha=0.3):
    Gx, Gy, Gz = grid
    Gzr = Gz // 2 + 1
    shape = (batch, n_species, Gx, Gy, Gzr)
    k2 = torch.rand(batch, 1, Gx, Gy, Gzr) + 0.1
    mult = torch.full((1, 1, 1, 1, Gzr), 2.0)
    mult[..., 0] = 1.0
    band = torch.ones(batch, 1, Gx, Gy, Gzr, dtype=torch.bool)
    band[:, :, 0, 0, 0] = False  # k=0 is never in-band
    return shape, k2, mult, band, alpha


@pytest.mark.parametrize("n_species", [1, 2])
def test_l_dyn_is_zero_when_residual_is_exactly_zero(n_species):
    shape, k2, mult, band, alpha = _dyn_ingredients(n_species)
    residual = torch.zeros(shape, dtype=torch.complex64)
    loss = l_dyn(residual, k2, mult, band, alpha=alpha)
    assert loss.item() == 0.0


@pytest.mark.parametrize("n_species", [1, 2])
def test_l_dyn_is_positive_for_a_nonzero_residual(n_species):
    shape, k2, mult, band, alpha = _dyn_ingredients(n_species)
    residual = torch.ones(shape, dtype=torch.complex64) * (0.1 + 0.1j)
    loss = l_dyn(residual, k2, mult, band, alpha=alpha)
    assert loss.item() > 0.0


def test_l_dyn_matches_the_ported_expression_by_hand():
    """`alpha*r2 + (1-alpha)*r2/(k2+eps)`, MULT-weighted, band-masked, summed
    and divided by n_band, then meaned over the batch -- worked by hand on a
    single mode so a coefficient error (dropped MULT, dropped 1/n_band) shows
    up as a wrong number rather than merely a wrong sign.
    """
    shape = (1, 1, 1, 1, 1)
    residual = torch.tensor([[[[[2.0 + 0.0j]]]]])
    k2 = torch.tensor([[[[[3.0]]]]])
    mult = torch.tensor([[[[[2.0]]]]])
    band = torch.ones(shape, dtype=torch.bool)
    alpha = 0.25
    eps = 1e-9
    got = l_dyn(residual, k2, mult, band, alpha=alpha, eps=eps).item()
    r2 = (2.0 ** 2) * 2.0  # |residual|^2 * mult
    expect = alpha * r2 + (1.0 - alpha) * r2 / (3.0 + eps)  # n_band == 1
    assert got == pytest.approx(expect, rel=1e-6)


def test_l_dyn_normalizes_by_n_band_not_by_the_full_mode_count():
    """`test_l_dyn_matches_the_ported_expression_by_hand` above uses a
    single-mode grid where `n_band == 1`, so dividing by `n_band` or not
    dividing at all gives the same number there -- it cannot catch a
    dropped `/n_band`. Two in-band modes, one with error and one without,
    do: with the division the loss is HALF of the one-mode value (the
    error is diluted across the band); without it, it would equal the
    one-mode value exactly.
    """
    shape = (1, 1, 1, 1, 2)
    residual = torch.tensor([[[[[2.0 + 0.0j, 0.0 + 0.0j]]]]])
    k2 = torch.tensor([[[[[3.0, 3.0]]]]])
    mult = torch.tensor([[[[[2.0, 2.0]]]]])
    band = torch.ones(shape, dtype=torch.bool)  # both modes in-band
    alpha, eps = 0.25, 1e-9
    got = l_dyn(residual, k2, mult, band, alpha=alpha, eps=eps).item()
    r2 = (2.0 ** 2) * 2.0  # the one nonzero mode's |r|^2 * mult
    per_mode_sum = alpha * r2 + (1.0 - alpha) * r2 / (3.0 + eps)  # the other mode contributes 0
    expect = per_mode_sum / 2.0  # n_band == 2
    assert got == pytest.approx(expect, rel=1e-6)


def test_l_dyn_eps_keeps_a_k_equal_zero_mode_finite():
    """`k2 = 0` is a real input this function must not blow up on (band
    masking normally excludes k=0, but nothing HERE enforces that, and a
    caller test double easily forgets to). Without `eps` under `k2`, an
    in-band k=0 mode with any residual divides by zero.
    """
    shape = (1, 1, 1, 1, 1)
    residual = torch.tensor([[[[[1.0 + 0.0j]]]]])
    k2 = torch.zeros(shape)
    mult = torch.ones(shape)
    band = torch.ones(shape, dtype=torch.bool)
    loss = l_dyn(residual, k2, mult, band, alpha=0.0, eps=1e-6)
    assert torch.isfinite(loss)


def test_l_dyn_weights_samples_by_sample_weight():
    shape, k2, mult, band, alpha = _dyn_ingredients(1, batch=2)
    residual = torch.zeros(shape, dtype=torch.complex64)
    residual[1] = 1.0 + 0.0j  # only the second sample has any error
    unweighted = l_dyn(residual, k2, mult, band, alpha=alpha)
    only_first = l_dyn(residual, k2, mult, band, alpha=alpha,
                       sample_weight=torch.tensor([1.0, 0.0]))
    assert unweighted.item() > 0.0
    assert only_first.item() == 0.0


# ---------------------------------------------------------------------------
# L_M
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_l_m_is_zero_at_its_own_target(n_species):
    M = _spd(n_species)
    assert l_m(M, M).item() == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("n_species", [1, 2])
def test_l_m_is_positive_away_from_target(n_species):
    M = _spd(n_species)
    off = M + 0.5
    assert l_m(off, M).item() > 0.0


def test_l_m_normalizes_by_the_target_not_the_prediction():
    """Pins WHICH side normalises the residual. `(pred - target)^2 /
    target^2` and `(pred - target)^2 / pred^2` are both zero at the target
    and both positive away from it, so neither of those two properties
    alone distinguishes them; a hand-computed value does.
    """
    M_target = torch.tensor([[[2.0]]])  # n_species = 1, one row
    M_pred = torch.tensor([[[4.0]]])
    got = l_m(M_pred, M_target).item()
    expect = (4.0 - 2.0) ** 2 / 2.0 ** 2  # normalised by the TARGET
    assert got == pytest.approx(expect, rel=1e-6)
    assert got != pytest.approx((4.0 - 2.0) ** 2 / 4.0 ** 2, rel=1e-6)


def test_l_m_row_weight_zeroes_out_a_rows_contribution():
    M_target = _spd(2, batch=2)
    M_pred = M_target.clone()
    M_pred[0] = M_pred[0] + 10.0  # a large miss on row 0
    w = torch.tensor([0.0, 1.0])
    assert l_m(M_pred, M_target, row_weight=w).item() == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# L_S
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("metric", ["rel_frob", "s_metric"])
def test_l_s_is_zero_at_its_own_target(n_species, metric):
    H = _spd(n_species, batch=4, extra_shape=(3,))  # (S, P, n, n) shell axis first
    mask = torch.ones(H.shape[:-2], dtype=torch.bool)
    loss = l_s(H, H, mask, metric=metric)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("metric", ["rel_frob", "s_metric"])
def test_l_s_is_positive_away_from_target(n_species, metric):
    H = _spd(n_species, batch=4, extra_shape=(3,))
    target = _spd(n_species, batch=4, extra_shape=(3,))
    mask = torch.ones(H.shape[:-2], dtype=torch.bool)
    loss = l_s(H, target, mask, metric=metric)
    assert loss.item() > 0.0


def test_l_s_mask_excludes_padded_shells():
    H = _spd(2, batch=2, extra_shape=(2,))
    target = H.clone()
    target[0, 0] = target[0, 0] + 100.0  # a huge miss, but on a padded shell
    mask = torch.ones(H.shape[:-2], dtype=torch.bool)
    mask[0, 0] = False
    loss = l_s(H, target, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_l_s_unknown_metric_raises():
    H = _spd(2, batch=2)
    mask = torch.ones(H.shape[:-2], dtype=torch.bool)
    with pytest.raises(ValueError):
        l_s(H, H, mask, metric="not_a_real_metric")


def test_l_s_rel_frob_normalizes_by_the_target_not_h():
    """Same shape of pin as `l_m`'s: zero-at-target and positive-away-from-
    target do not distinguish normalising by `H` from normalising by
    `target`, so this checks the exact value.
    """
    H = torch.tensor([[[[4.0]]]])       # (row=1, shell=1, 1, 1)
    target = torch.tensor([[[[2.0]]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    got = l_s(H, target, mask, metric="rel_frob").item()
    expect = (4.0 - 2.0) ** 2 / 2.0 ** 2
    assert got == pytest.approx(expect, rel=1e-6)


def test_l_s_row_weight_zeroes_out_a_rows_contribution():
    """`row_weight` broadcasts against mask's ROW axis (the second-to-last
    axis of H/target: shape (..., row, shell, n, n)), one weight per row,
    applied to every shell of that row.
    """
    target = _spd(2, batch=2, extra_shape=(1,)).movedim(0, 1)  # (row=2, shell=1, 2, 2)
    H = target.clone()
    H[0] = H[0] + 10.0  # a large miss on row 0 only
    mask = torch.ones(H.shape[:-2], dtype=torch.bool)  # (row=2, shell=1)
    row_weight = torch.tensor([0.0, 1.0])  # down-weight the row with the miss
    loss = l_s(H, target, mask, row_weight=row_weight)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_l_s_row_weight_is_load_bearing_when_not_applied():
    """The companion to the test above: WITHOUT the row_weight, the same
    miss must make the loss positive, so the previous test is pinning the
    weighting rather than something already zero regardless.
    """
    target = _spd(2, batch=2, extra_shape=(1,)).movedim(0, 1)
    H = target.clone()
    H[0] = H[0] + 10.0
    mask = torch.ones(H.shape[:-2], dtype=torch.bool)
    loss = l_s(H, target, mask)
    assert loss.item() > 0.0


@pytest.mark.parametrize("n_species", (1, 2, 3))
def test_sinv_rootinv_satisfies_its_defining_identity(n_species):
    """`sinv_rootinv` advertises being general in the matrix size, so the
    identity is checked above two species as well.

    Two is not enough to exercise it. `torch.linalg.eigh` happens to return
    a SYMMETRIC eigenvector matrix for 2x2 -- measured max|V - V^T| = 0 over
    200 draws -- so a transpose dropped from `V D V^T` is invisible there.
    At three and four it is 2.0 and 1.9, and the identity breaks by up to
    2.2. Found by mutation testing: the shipped code
    is correct, and nothing would have caught it becoming wrong.
    """
    S = _spd(n_species, batch=3)
    R = sinv_rootinv(S)
    eye = torch.eye(n_species, dtype=torch.float64).expand(3, n_species,
                                                            n_species)
    assert torch.allclose(R @ S @ R, eye, atol=1e-6)


# ---------------------------------------------------------------------------
# L_bulk -- two residual forms, declared, not inferred
# ---------------------------------------------------------------------------

def _bulk_ingredients(n_species, batch=6):
    H0 = _spd(n_species, batch=batch)
    kBT = torch.rand(batch, dtype=torch.float64) + 0.5
    zvec = torch.randn(batch, n_species, dtype=torch.float64)
    zvec[zvec.abs() < 0.1] = 0.5  # keep z away from an accidental exact zero
    rho_tot = torch.rand(batch, dtype=torch.float64) + 0.5
    eye = torch.eye(n_species, dtype=torch.float64)
    y = torch.linalg.solve(H0 + 1e-10 * eye, zvec.unsqueeze(-1)).squeeze(-1)
    q = (zvec * y).sum(-1)
    target = kBT * q / rho_tot  # the exact model-side value: zero residual
    return H0, kBT, zvec, rho_tot, target


@pytest.mark.parametrize("n_species", [1, 2])
def test_l_bulk_relative_inverse_is_zero_at_its_own_target(n_species):
    H0, kBT, zvec, rho_tot, target = _bulk_ingredients(n_species)
    loss = l_bulk(H0, kBT, zvec, rho_tot, target, residual="relative_inverse")
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("n_species", [1, 2])
def test_l_bulk_relative_inverse_is_positive_away_from_target(n_species):
    H0, kBT, zvec, rho_tot, target = _bulk_ingredients(n_species)
    loss = l_bulk(H0, kBT, zvec, rho_tot, target * 1.5, residual="relative_inverse")
    assert loss.item() > 0.0


@pytest.mark.parametrize("n_species", [1, 2])
def test_l_bulk_sigma_chi2_is_zero_at_its_own_target(n_species):
    H0, kBT, zvec, rho_tot, target = _bulk_ingredients(n_species)
    sigma = torch.full_like(target, 0.02)
    loss = l_bulk(H0, kBT, zvec, rho_tot, target, residual="sigma_chi2", sigma=sigma)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("n_species", [1, 2])
def test_l_bulk_sigma_chi2_is_positive_away_from_target(n_species):
    H0, kBT, zvec, rho_tot, target = _bulk_ingredients(n_species)
    sigma = torch.full_like(target, 0.02)
    loss = l_bulk(H0, kBT, zvec, rho_tot, target * 1.5, residual="sigma_chi2", sigma=sigma)
    assert loss.item() > 0.0


def test_l_bulk_sigma_chi2_requires_sigma():
    H0, kBT, zvec, rho_tot, target = _bulk_ingredients(2)
    with pytest.raises(ValueError):
        l_bulk(H0, kBT, zvec, rho_tot, target, residual="sigma_chi2")


def test_l_bulk_relative_inverse_matches_the_ported_expression_by_hand():
    """`resid = target/model - 1`, not `model/target - 1`: both vanish at
    the same point and are both positive away from it after squaring, so
    neither the zero-at-target nor the positive-away-from-target test above
    distinguishes the two orientations. This one does, on a scalar
    (n_species=1) example simple enough to compute by hand: H0=[[2]],
    kBT=1, zvec=[1], rho_tot=1 gives model = kBT*z^T H0^-1 z/rho_tot = 0.5.
    """
    H0 = torch.tensor([[[2.0]]], dtype=torch.float64)
    kBT = torch.tensor([1.0], dtype=torch.float64)
    zvec = torch.tensor([[1.0]], dtype=torch.float64)
    rho_tot = torch.tensor([1.0], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)
    got = l_bulk(H0, kBT, zvec, rho_tot, target,
                residual="relative_inverse", jitter=0.0).item()
    model = 0.5
    expect = (target.item() / model - 1.0) ** 2
    assert got == pytest.approx(expect, rel=1e-6)


def test_l_bulk_sigma_chi2_matches_the_ported_expression_by_hand():
    """Same shape of pin, for the `sigma_chi2` form: `resid = (model_inv -
    target_inv) / sigma`, model_inv = 1/model = 2.0 in the example above.
    """
    H0 = torch.tensor([[[2.0]]], dtype=torch.float64)
    kBT = torch.tensor([1.0], dtype=torch.float64)
    zvec = torch.tensor([[1.0]], dtype=torch.float64)
    rho_tot = torch.tensor([1.0], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)
    sigma = torch.tensor([0.1], dtype=torch.float64)
    got = l_bulk(H0, kBT, zvec, rho_tot, target, residual="sigma_chi2",
                sigma=sigma, jitter=0.0).item()
    model_inv, target_inv = 2.0, 1.0
    expect = ((model_inv - target_inv) / 0.1) ** 2
    assert got == pytest.approx(expect, rel=1e-6)


def test_l_bulk_relative_inverse_clamp_bounds_a_huge_residual():
    H0 = torch.tensor([[[1e8]]], dtype=torch.float64)  # near-zero model value
    kBT = torch.tensor([1.0], dtype=torch.float64)
    zvec = torch.tensor([[1.0]], dtype=torch.float64)
    rho_tot = torch.tensor([1.0], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)
    unclamped = l_bulk(H0, kBT, zvec, rho_tot, target,
                       residual="relative_inverse", jitter=0.0).item()
    clamped = l_bulk(H0, kBT, zvec, rho_tot, target,
                     residual="relative_inverse", jitter=0.0, clamp=50.0).item()
    assert unclamped > clamped
    assert clamped == pytest.approx(50.0 ** 2, rel=1e-6)


def test_l_bulk_unknown_residual_form_raises():
    H0, kBT, zvec, rho_tot, target = _bulk_ingredients(2)
    with pytest.raises(ValueError):
        l_bulk(H0, kBT, zvec, rho_tot, target, residual="not_a_real_form")


def test_l_bulk_residual_forms_differ_by_roughly_the_documented_order_of_magnitude():
    """The training code's own comment on the anchor loss: the sigma-weighted chi-square
    "would dominate the drift loss by ~10^3 under our lambda=0.01" relative
    to the relative-inverse form it was chosen over. This is the fact being
    stated, not a fitted constant: on the SAME (H0, kBT, zvec, rho_tot,
    target) input, moved the SAME small relative amount off target, the two
    forms' losses should differ by something on the order of a thousand for
    a realistic sigma -- pinned loosely (a couple of orders of magnitude
    either side), never to a specific ratio.

    `sigma` is scaled with the target's own inverse magnitude (a fixed
    relative uncertainty, `frac`), matching how the measured uncertainty is
    actually used in the source tree (a per-row SEM comparable in scale to
    the quantity it measures) -- an absolute constant `sigma` would instead
    let the ratio swing with the arbitrary magnitude of the random SPD
    matrices this test builds, which is not the effect being pinned.
    """
    H0, kBT, zvec, rho_tot, target = _bulk_ingredients(2, batch=20)
    off_target = target * 1.01  # a small, fixed relative miss
    frac = 0.02  # a realistic few-percent relative uncertainty
    sigma = frac / off_target
    rel_inv = l_bulk(H0, kBT, zvec, rho_tot, off_target,
                     residual="relative_inverse").item()
    chi2 = l_bulk(H0, kBT, zvec, rho_tot, off_target,
                 residual="sigma_chi2", sigma=sigma).item()
    assert rel_inv > 0.0
    ratio = chi2 / rel_inv
    assert 10.0 < ratio < 1.0e6, (
        f"ratio={ratio}: expected the sigma-weighted form to dominate by "
        f"something like 10^3, not merely be larger"
    )


# ---------------------------------------------------------------------------
# L_P
# ---------------------------------------------------------------------------

def test_l_p_absolute_is_zero_at_its_own_target():
    P_target = torch.tensor([1.0, 2.0, 3.0])
    assert l_p_absolute(P_target, P_target).item() == pytest.approx(0.0, abs=1e-10)


def test_l_p_absolute_is_positive_away_from_target():
    P_target = torch.tensor([1.0, 2.0, 3.0])
    assert l_p_absolute(P_target * 1.1, P_target).item() > 0.0


def test_l_p_absolute_matches_the_ported_expression_by_hand():
    """Pins that the residual is normalised by the TARGET (per-row, so a
    low- and a high-pressure manifold weigh in equally), not by the raw
    difference: both are zero at target and positive away from it, so
    neither property above distinguishes them.
    """
    P_model = torch.tensor([1.1, 20.0])
    P_target = torch.tensor([1.0, 10.0])
    got = l_p_absolute(P_model, P_target).item()
    expect = (((0.1 / 1.0) ** 2) + ((10.0 / 10.0) ** 2)) / 2.0
    assert got == pytest.approx(expect, rel=1e-6)


def test_l_p_absolute_with_a_floor_divides_by_max_abs_target_and_floor():
    """A zero target is a division by zero without the floor; above it the floor changes nothing."""
    P_target = torch.tensor([0.0, 0.5, -2.0])
    P_model = torch.tensor([0.1, 0.6, -2.5])
    got = l_p_absolute(P_model, P_target, floor=1.0).item()
    by_hand = ((0.1 / 1.0) ** 2 + (0.1 / 1.0) ** 2 + (0.5 / 2.0) ** 2) / 3.0
    assert got == pytest.approx(by_hand, rel=1e-6)
    above = torch.tensor([2.0, 3.0])
    assert torch.equal(l_p_absolute(above * 1.1, above, floor=1.0),
                       l_p_absolute(above * 1.1, above))


def test_l_p_variance_is_zero_when_every_group_is_already_flat():
    P = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0])
    group_id = torch.tensor([0, 0, 1, 1, 1])
    assert l_p_variance(P, group_id, p_ref=1.0).item() == pytest.approx(0.0, abs=1e-10)


def test_l_p_variance_is_positive_when_a_group_disagrees():
    P = torch.tensor([1.0, 3.0, 2.0, 2.0, 2.0])
    group_id = torch.tensor([0, 0, 1, 1, 1])
    assert l_p_variance(P, group_id, p_ref=1.0).item() > 0.0


def test_l_p_variance_normalizes_by_the_number_of_groups():
    """Two groups, only one of which disagrees, should give HALF the loss
    of one group alone with the same disagreement -- pins the `/n_groups`
    reduction rather than merely its presence (which `test_l_p_variance_is_
    positive_when_a_group_disagrees` above already shows).
    """
    P_one_group = torch.tensor([1.0, 3.0])
    one_group_id = torch.tensor([0, 0])
    one = l_p_variance(P_one_group, one_group_id, p_ref=1.0).item()

    P_two_groups = torch.tensor([1.0, 3.0, 5.0, 5.0])
    two_group_id = torch.tensor([0, 0, 1, 1])
    two = l_p_variance(P_two_groups, two_group_id, p_ref=1.0).item()

    assert two == pytest.approx(one / 2.0, rel=1e-6)


def test_l_p_variance_scales_with_p_ref_squared():
    P = torch.tensor([1.0, 3.0])
    group_id = torch.tensor([0, 0])
    a = l_p_variance(P, group_id, p_ref=1.0).item()
    b = l_p_variance(P, group_id, p_ref=2.0).item()
    assert a == pytest.approx(4.0 * b, rel=1e-6)


# ---------------------------------------------------------------------------
# L_conv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_l_conv_hinge_is_zero_when_lambda_min_meets_the_margin(n_species):
    margin = 0.05
    eye = torch.eye(n_species, dtype=torch.float64)
    H = (margin * eye).expand(7, n_species, n_species).clone()
    loss, aux = l_conv(H, form="hinge", margin=margin)
    assert loss.item() == pytest.approx(0.0, abs=1e-10)
    assert aux["viol_frac"] == 0.0


@pytest.mark.parametrize("n_species", [1, 2])
def test_l_conv_hinge_is_positive_below_the_margin(n_species):
    margin = 0.05
    eye = torch.eye(n_species, dtype=torch.float64)
    H = (-0.1 * eye).expand(7, n_species, n_species).clone()
    loss, aux = l_conv(H, form="hinge", margin=margin)
    assert loss.item() > 0.0
    assert aux["viol_frac"] == 1.0


def test_l_conv_hinge0_ignores_margin_and_only_penalizes_a_negative_eigenvalue():
    eye = torch.eye(2, dtype=torch.float64)
    H_ok = (0.001 * eye).expand(4, 2, 2).clone()   # positive, below a would-be margin
    H_bad = (-0.001 * eye).expand(4, 2, 2).clone()  # negative
    loss_ok, aux_ok = l_conv(H_ok, form="hinge0", margin=0.05)
    loss_bad, aux_bad = l_conv(H_bad, form="hinge0", margin=0.05)
    assert loss_ok.item() == pytest.approx(0.0, abs=1e-10)
    assert loss_bad.item() > 0.0
    assert aux_ok["viol_frac"] == 0.0
    assert aux_bad["viol_frac"] == 1.0


def test_l_conv_softplus_is_positive_below_the_margin_and_smaller_above_it():
    margin = 0.05
    eye = torch.eye(2, dtype=torch.float64)
    H_above = (0.1 * eye).expand(4, 2, 2).clone()
    H_below = (-0.1 * eye).expand(4, 2, 2).clone()
    loss_above, _ = l_conv(H_above, form="softplus", margin=margin)
    loss_below, _ = l_conv(H_below, form="softplus", margin=margin)
    assert 0.0 < loss_above.item() < loss_below.item()


def test_l_conv_aux_reports_the_worst_eigenvalue():
    eye = torch.eye(2, dtype=torch.float64)
    H = torch.stack([eye, -3.0 * eye])
    _, aux = l_conv(H, form="hinge0", margin=0.0)
    assert aux["lam_worst"] == pytest.approx(-3.0, rel=1e-6)


def test_l_conv_unknown_form_raises():
    H = _spd(2, batch=3)
    with pytest.raises(ValueError):
        l_conv(H, form="not_a_real_form")


# ---------------------------------------------------------------------------
# L_Gamma
# ---------------------------------------------------------------------------

def test_gamma_closed_form_is_finite_and_generalizes_to_one_species():
    H = _spd(1, batch=4)
    v = torch.ones(4, 1, dtype=torch.float64)
    kBT = torch.rand(4, dtype=torch.float64) + 0.5
    n_density = torch.rand(4, dtype=torch.float64) + 1.0
    prefactor = torch.rand(4, dtype=torch.float64) + 0.1
    gamma = gamma_closed_form(H, n_density, prefactor, v, kBT)
    assert torch.isfinite(gamma).all()


def test_gamma_closed_form_matches_the_binary_closed_form_by_hand():
    """The source trees' binary form: Gamma = n*x*(1-x)*det(H) /
    (kBT*[x^2 H11 + 2x(1-x) H01 + (1-x)^2 H00]), recovered here with
    prefactor = x(1-x) and v = (1-x, x). Worked on H = [[3,1],[1,2]],
    x=0.3, n=1, kBT=1.
    """
    H = torch.tensor([[[3.0, 1.0], [1.0, 2.0]]], dtype=torch.float64)
    x = 0.3
    v = torch.tensor([[1.0 - x, x]], dtype=torch.float64)
    kBT = torch.tensor([1.0], dtype=torch.float64)
    n_density = torch.tensor([1.0], dtype=torch.float64)
    prefactor = torch.tensor([x * (1.0 - x)], dtype=torch.float64)
    got = gamma_closed_form(H, n_density, prefactor, v, kBT).item()

    det = 3.0 * 2.0 - 1.0 * 1.0
    denom = x * x * 2.0 + 2.0 * x * (1.0 - x) * 1.0 + (1.0 - x) * (1.0 - x) * 3.0
    expect = 1.0 * x * (1.0 - x) * det / (1.0 * denom)
    assert got == pytest.approx(expect, rel=1e-6)


def test_l_gamma_is_zero_on_a_perfectly_convex_gamma_path():
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.float64)
    gamma = x ** 2  # d2/dx2 = 2 > 0 everywhere: no violation
    h = float(x[1] - x[0])
    loss, aux = l_gamma(gamma.unsqueeze(0), h=h)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert aux["viol_frac"] == 0.0


def test_l_gamma_stencil_reproduces_the_exact_second_derivative():
    """`test_l_gamma_is_zero_on_a_perfectly_convex_gamma_path` above only
    checks the SIGN of d2 (it must be >= 0 somewhere on a convex path), and
    a mistyped stencil coefficient can still land on a positive number by
    accident, passing that test without being correct. A pure quadratic has
    a known, CONSTANT exact second derivative (2.0 for `x**2`), and every
    interior point of the five-point stencil -- reported here through
    `aux["d2_worst"]`, the minimum over the interior -- must equal it
    exactly, not merely be positive.
    """
    x = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    gamma = x ** 2
    h = float(x[1] - x[0])
    _, aux = l_gamma(gamma.unsqueeze(0), h=h)
    assert aux["d2_worst"] == pytest.approx(2.0, rel=1e-8)


def test_l_gamma_is_positive_on_a_concave_stretch():
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.float64)
    gamma = -(x ** 2)  # d2/dx2 = -2 < 0 everywhere: a full violation
    h = float(x[1] - x[0])
    loss, aux = l_gamma(gamma.unsqueeze(0), h=h)
    assert loss.item() > 0.0
    assert aux["viol_frac"] == 1.0


# ---------------------------------------------------------------------------
# L_W (non-canonical extra)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species", [1, 2])
def test_l_w_is_zero_when_the_hinge_bound_is_exactly_met(n_species):
    kappa = 2.0
    k = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    eye = torch.eye(n_species, dtype=torch.float64)
    D = torch.einsum("k,ij->kij", kappa * k * k, eye)  # eigmin(D) == kappa*k^2 exactly
    loss = l_w(D, k, kappa=kappa)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("n_species", [1, 2])
def test_l_w_is_positive_when_the_bound_is_violated(n_species):
    kappa = 2.0
    k = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    eye = torch.eye(n_species, dtype=torch.float64)
    D = torch.einsum("k,ij->kij", 0.5 * kappa * k * k, eye)  # below the bound
    loss = l_w(D, k, kappa=kappa)
    assert loss.item() > 0.0


def test_l_w_kappa_actually_grows_the_margin_with_k():
    """A D that is fine at `kappa=0` (constant, positive eigmin) but
    violates the bound once `kappa` makes the margin grow with k: pins that
    `kappa * k^2` is load-bearing, not merely present in the signature.
    Without it, this D would score zero at every k tested here.
    """
    eigmin_const = 0.3
    k = torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64)
    D = torch.stack([eigmin_const * torch.eye(2, dtype=torch.float64)] * 3)
    zero_kappa = l_w(D, k, kappa=0.0)
    grown_margin = l_w(D, k, kappa=2.0)
    assert zero_kappa.item() == pytest.approx(0.0, abs=1e-10)
    assert grown_margin.item() > 0.0


# ---------------------------------------------------------------------------
# Anchor row gating: a System-declared table, no package default.
# ---------------------------------------------------------------------------

def test_gate_rows_declared_by_name_selects_the_named_column():
    table = {"single_phase": [True, False, True], "T_K": [1.0, 2.0, 3.0]}
    mask = gate_rows(table, declared_gate="single_phase")
    assert mask.tolist() == [True, False, True]


def test_gate_rows_declared_priority_list_uses_the_first_present_column():
    table = {"anchor_eligible": [True, True, False]}
    mask = gate_rows(table, declared_gate=("anchor_eligible_noxi", "anchor_eligible",
                                           "single_phase"))
    assert mask.tolist() == [True, True, False]


def test_gate_rows_declared_priority_list_prefers_the_first_over_a_later_one():
    """Two priority-list columns both present with DIFFERENT values: the
    single-column test above cannot tell 'uses the first' apart from 'uses
    whichever is present', since only one candidate existed there.
    """
    table = {
        "anchor_eligible_noxi": [True, False, True],
        "anchor_eligible": [False, False, False],
    }
    mask = gate_rows(table, declared_gate=("anchor_eligible_noxi",
                                           "anchor_eligible", "single_phase"))
    assert mask.tolist() == [True, False, True]


def test_gate_rows_declared_priority_list_raises_if_none_present():
    table = {"T_K": [1.0, 2.0]}
    with pytest.raises(KeyError):
        gate_rows(table, declared_gate=("anchor_eligible_noxi", "single_phase"))


def test_gate_rows_no_declared_gate_and_no_known_gate_columns_keeps_every_row():
    """The k-modes system's own tables: no gate at all, and this module is
    never told any gate-shaped column name to watch for, so nothing is being
    silently dropped -- there is nothing to drop.
    """
    table = {"T_K": [1.0, 2.0, 3.0]}
    mask = gate_rows(table, declared_gate=None, known_gate_columns=())
    assert mask.tolist() == [True, True, True]


def test_gate_rows_no_declared_gate_but_a_known_gate_column_is_present_raises():
    """The required behaviour: a table that HAS a gate column,
    against a caller that declared none, is an error -- not a silent
    'use everything'.
    """
    table = {"single_phase": [True, False], "T_K": [1.0, 2.0]}
    with pytest.raises(ValueError):
        gate_rows(table, declared_gate=None,
                 known_gate_columns=("single_phase", "anchor_eligible"))


def test_gate_rows_declared_gate_missing_from_the_table_raises():
    table = {"T_K": [1.0, 2.0]}
    with pytest.raises(KeyError):
        gate_rows(table, declared_gate="single_phase")


# ---------------------------------------------------------------------------
# l_dyn's two floors are two floors
# ---------------------------------------------------------------------------

def test_the_k2_floor_does_not_also_floor_the_weight_sum():
    """`eps` guards a zero wavevector; `weight_eps` guards an empty batch.

    They were one parameter here once, and the caller passes its `k2` floor
    of 1e-6 where the measured source floors the weight sum with 1e-12. The
    result was every `L_dyn` low by exactly `weight_eps/(sum(w)+weight_eps)`
    -- a uniform rescaling of the drift term and of every drift gradient,
    which does NOT shrink with precision and so survives a float64 control.
    It was found by an L1 training-step comparison for exactly that reason.

    This asserts the closed form rather than a tolerance: a deviation that
    is an exact identity should be checked as one.
    """
    torch.manual_seed(0)
    residual = torch.randn(4, 2, 6, 6, 4, dtype=torch.float64)
    k2 = torch.rand(1, 1, 6, 6, 4, dtype=torch.float64) + 0.5
    mult = torch.ones_like(k2)
    band = torch.ones_like(k2, dtype=torch.bool)
    w = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

    big = 1e-6                      # what the caller passes for the k2 floor
    exact = l_dyn(residual, k2, mult, band, alpha=0.0, eps=big,
                  weight_eps=0.0, sample_weight=w)
    floored = l_dyn(residual, k2, mult, band, alpha=0.0, eps=big,
                    weight_eps=big, sample_weight=w)

    # the floor's whole effect is the denominator, so the ratio is closed form
    predicted = float(w.sum()) / (float(w.sum()) + big)
    assert float(floored / exact) == pytest.approx(predicted, rel=1e-12)

    # and the k2 floor must not reach the denominator at all
    other_k2_floor = l_dyn(residual, k2, mult, band, alpha=0.0, eps=1e-3,
                           weight_eps=0.0, sample_weight=w)
    ratio = float(other_k2_floor / exact)
    assert ratio != 1.0, "eps must still affect the k2 division"
    assert float(l_dyn(residual, k2, mult, band, alpha=0.0, eps=1e-3,
                       weight_eps=big, sample_weight=w)
                 / other_k2_floor) == pytest.approx(predicted, rel=1e-12)


def test_the_weight_floor_defaults_to_the_measured_sources_value():
    """1e-12, which is what the published model's training step divides by."""
    import inspect
    assert inspect.signature(l_dyn).parameters["weight_eps"].default == 1e-12
