"""Tests for the two-way loss-name map.

The package's canonical loss names are the paper's seven: `L_dyn`, `L_M`,
`L_S`, `L_bulk`, `L_P`, `L_conv`, `L_Gamma`, plus `L_W` as a non-canonical,
system-specific extra. Three source codebases wrote checkpoints using their
own code symbols for these, and two of those codebases wrote near-identical
symbols while a third genuinely collides with them: both spell the
mobility-anchor weight the same way, but ONE codebase's `lambda_S` names the
k-resolved shell anchor (`L_S`) while ANOTHER codebase's `lambda_S` names the
k=0 bulk anchor (`L_bulk`) -- the same string, two different canonical
losses, depending on which codebase wrote the checkpoint.

A map built from the string alone, without knowing which codebase wrote it,
would resolve one of the two wrongly with nothing raising: the loss would
still fall during training, every number derived from it would be wrong, and
nothing would say so. So a single flat map cannot serve every checkpoint;
which map applies is a fact about the checkpoint's origin, declared by
whoever loads it, never guessed from a key's spelling.

These tests therefore pin TWO separate weight-key maps (one per codebase
family) and TWO separate log-key maps, each round-tripping in both directions
on its own, and pin the collision explicitly: the two families' maps answer
differently for the same code string.
"""
from __future__ import annotations

import pytest

from aipf.losses.names import (
    CANONICAL_LOSSES,
    NONCANONICAL_EXTRAS,
    LossNameMap,
    LOG_MAP_A,
    LOG_MAP_B,
    WEIGHT_MAP_A,
    WEIGHT_MAP_B,
)

ALL_WEIGHT_MAPS = (WEIGHT_MAP_A, WEIGHT_MAP_B)
ALL_LOG_MAPS = (LOG_MAP_A, LOG_MAP_B)
ALL_MAPS = ALL_WEIGHT_MAPS + ALL_LOG_MAPS


def test_canonical_list_is_exactly_the_papers_seven():
    assert CANONICAL_LOSSES == (
        "L_dyn", "L_M", "L_S", "L_bulk", "L_P", "L_conv", "L_Gamma",
    )


def test_l_w_is_not_canonical():
    assert "L_W" not in CANONICAL_LOSSES
    assert NONCANONICAL_EXTRAS == ("L_W",)


@pytest.mark.parametrize("loss_map", ALL_MAPS)
def test_every_declared_pair_round_trips_canonical_to_code_to_canonical(loss_map):
    for canonical in loss_map.canonical_names:
        code = loss_map.to_code(canonical)
        assert loss_map.to_canonical(code) == canonical


@pytest.mark.parametrize("loss_map", ALL_MAPS)
def test_every_declared_pair_round_trips_code_to_canonical_to_code(loss_map):
    for code in loss_map.code_symbols:
        canonical = loss_map.to_canonical(code)
        assert loss_map.to_code(canonical) == code


@pytest.mark.parametrize("loss_map", ALL_MAPS)
def test_an_unmapped_canonical_name_raises_not_none(loss_map):
    with pytest.raises(KeyError):
        loss_map.to_code("not_a_real_canonical_name")


@pytest.mark.parametrize("loss_map", ALL_MAPS)
def test_an_unmapped_code_symbol_raises_not_none(loss_map):
    with pytest.raises(KeyError):
        loss_map.to_canonical("not_a_real_code_symbol")


def test_weight_map_a_maps_the_shell_anchor_weight_to_l_s():
    """Family A's `lambda_S` names the k-resolved shell anchor."""
    assert WEIGHT_MAP_A.to_canonical("lambda_S") == "L_S"
    assert WEIGHT_MAP_A.to_code("L_S") == "lambda_S"


def test_weight_map_b_maps_its_bulk_anchor_symbol_to_l_bulk_not_l_s():
    """The trap the brief names by hand: family B's checkpoint weight key
    for its k=0 bulk anchor is the literal string `lambda_S` -- the SAME
    string family A uses for its k-resolved shell anchor's weight
    (`L_S`). A map built from the string alone, without knowing which
    family wrote the checkpoint, would wire family B's `lambda_S` into the
    k-resolved slot instead of `L_bulk`.

    The k-modes format's training code additionally spells the underlying
    function `loss_S`, which reads even more like "S" for
    "static" than the checkpoint key does -- the exact trap the brief
    names by hand -- but the checkpoint key, not the function name, is
    what the checkpoint-format layer actually has to resolve, so that is
    what this map is built from and what is pinned here.
    """
    assert WEIGHT_MAP_B.to_canonical("lambda_S") == "L_bulk"
    assert WEIGHT_MAP_B.to_code("L_bulk") == "lambda_S"


def test_weight_map_b_has_no_shell_anchor_at_all():
    """Measured: family B has no finite-wavevector static anchor."""
    assert "L_S" not in WEIGHT_MAP_B.canonical_names
    with pytest.raises(KeyError):
        WEIGHT_MAP_B.to_code("L_S")


def test_weight_map_b_has_no_pressure_convexity_or_gamma_terms():
    """Measured: family B ships only L_dyn, L_M, L_bulk in code."""
    for missing in ("L_P", "L_conv", "L_Gamma"):
        assert missing not in WEIGHT_MAP_B.canonical_names


def test_the_lambda_s_collision_resolves_differently_per_family():
    """The load-bearing fact this whole module exists to represent: the
    same literal checkpoint key, `lambda_S`, means two different canonical
    losses depending on which family's map is asked. Both maps recognise
    the key; they simply disagree about what it means, which is exactly
    why a single flat map cannot serve both checkpoints.
    """
    assert WEIGHT_MAP_A.to_canonical("lambda_S") == "L_S"
    assert WEIGHT_MAP_B.to_canonical("lambda_S") == "L_bulk"


def test_dyn_weight_key_is_lambda_drift_only_in_kmodes():
    """`lambda_drift` is a real, saved checkpoint hyper-parameter key in the
    k-modes format -- but NOT in the Lightning-hparams format, whose
    training code has no `lambda_drift` constructor keyword at all: it
    computes `loss = drift + lambda_M*anchor + ...` with the drift term's
    weight hardcoded at 1 (measured directly against both published
    models' training code). This corrects an earlier, wrong version
    of this test (and of `WEIGHT_MAP_A` itself), found while
    building the checkpoint-format layer that actually has to resolve this
    key against a real checkpoint.
    """
    assert "L_dyn" not in WEIGHT_MAP_A.canonical_names
    with pytest.raises(KeyError):
        WEIGHT_MAP_A.to_code("L_dyn")
    assert WEIGHT_MAP_B.to_code("L_dyn") == "lambda_drift"
    assert WEIGHT_MAP_B.to_canonical("lambda_drift") == "L_dyn"


def test_log_map_a_bulk_key_is_train_bulk_not_train_static():
    """The brief's own table: `L_bulk` <-> `train/bulk`, `L_S` <-> `train/static`
    (k-resolved) in family A. These are two DIFFERENT log keys for two
    DIFFERENT canonical losses in the same family -- unlike the weight-key
    collision above, there is no ambiguity within one family, only across
    families.
    """
    assert LOG_MAP_A.to_canonical("train/bulk") == "L_bulk"
    assert LOG_MAP_A.to_canonical("train/static") == "L_S"
    assert LOG_MAP_A.to_canonical("train/drift") == "L_dyn"
    assert LOG_MAP_A.to_canonical("train/anchor") == "L_M"


def test_log_map_b_bulk_key_is_train_anchor_s_not_train_bulk():
    """Measured: the k-modes format logs its `L_bulk` term as `train/anchor_S`,
    not `train/bulk` -- another place
    a name-shaped guess would go wrong.
    """
    assert LOG_MAP_B.to_canonical("train/anchor_S") == "L_bulk"
    assert LOG_MAP_B.to_canonical("train/anchor_M") == "L_M"
    assert LOG_MAP_B.to_canonical("train/drift") == "L_dyn"


def test_l_w_is_present_only_in_the_family_that_has_it():
    """Measured: `L_W` appears in exactly one family's
    supplement paragraph and code (`lambda_wpsd`, logged `train/wpsd`).
    """
    assert WEIGHT_MAP_A.to_code("L_W") == "lambda_wpsd"
    assert LOG_MAP_A.to_code("L_W") == "train/wpsd"
    with pytest.raises(KeyError):
        WEIGHT_MAP_B.to_code("L_W")


def test_constructing_a_map_rejects_an_unknown_canonical_name():
    with pytest.raises(ValueError):
        LossNameMap({"not_a_real_name": "some_code"}, label="bad")


def test_constructing_a_map_rejects_two_canonical_names_sharing_one_code_symbol():
    """A code symbol claimed by two canonical names is exactly the shape of
    the trap this module exists to prevent, just built directly instead of
    discovered by testing a finished map.
    """
    with pytest.raises(ValueError):
        LossNameMap({"L_dyn": "same_symbol", "L_M": "same_symbol"}, label="bad")


def test_map_label_is_recorded_and_distinct_between_families():
    assert WEIGHT_MAP_A.label != WEIGHT_MAP_B.label
    assert LOG_MAP_A.label != LOG_MAP_B.label
