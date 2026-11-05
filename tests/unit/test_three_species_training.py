"""Three channels through the training path: a ternary ``System``, its anchor tables and one step.

The declaration is the Fe-B-shaped one of ``test_three_species_model.py`` plus the vector form of
the anchor tables (``docs/reference/training.md``, Anchor tables): one fraction column per channel,
the ``active`` mask and a relative ``Scc0_err``. The tables and the manifold are written here in the
layout a ternary campaign writes (key names included), with a ternary row, a binary row with one
species absent and a pure row (the manifold with one column the reader ignores), and the mode
archive is a three-channel ``ModesRecord`` with one ternary run and one run whose Cu channel is
identically zero. ``fit`` reads ``modes.npz`` and its provenance; a ``meta.json`` is read only by
the data index, so none is written.

What is pinned: the declaration validates and builds; ``AnchorTables.load`` keeps the rows each
anchor should (a pure row feeds ``L_P`` alone), carries the active masks, warns about no shell of
the absent-species row and records every file it read with its digest; and one optimiser step with
``L_dyn + L_W + L_P + L_S + L_bulk + L_M`` (``bulk_residual = "s_metric"``) has every term finite,
changes the parameters, and writes a checkpoint that reloads strictly. With the number-conserving,
saturated kernel hinge and ``L_conv`` on the anchor states declared, one step trains both beside the
anchors and records them.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging

import numpy as np
import pytest
import torch

from aipf.functional.build import build
from aipf.mobility import build as build_mobility
from aipf.paths import Paths
from aipf.pipeline.modes import ModesRecord
from aipf.system import Checkpoint
from aipf.train.anchors import AnchorTables
from aipf.train.datamodule import SourceSpec
from aipf.train.config import TrainConfig
from aipf.train.fit import _config_from_system, _load_weights_into, fit
from aipf.train.lit_module import seeded_rng

from test_three_species_model import SPECIES, ternary_system


@pytest.fixture(autouse=True)
def _demo_scratch_is_its_own(monkeypatch):
    """The toy declares its own raw root; a user's ``AIPF_RAW`` or ``AIPF_RAW_TERNARY_TOY``, which
    ``Paths`` prefers to a declared default, must not replace it."""
    monkeypatch.delenv("AIPF_RAW", raising=False)
    monkeypatch.delenv("AIPF_RAW_TERNARY_TOY", raising=False)


_GRID = (4, 4, 4)
_FRAMES = 24
_BOX = 9.0

#: Per row: the fractions, the temperature and the total density. Row 0 is ternary, row 1 binary
#: with Cu absent, row 2 pure Ni.
_X = np.array([[0.4, 0.3, 0.3],
               [0.5, 0.0, 0.5],
               [0.0, 0.0, 1.0]])
_T = np.array([1500.0, 1700.0, 1900.0])
_TOTAL = np.array([0.085, 0.084, 0.087])
_SHELLS = np.array([0.6, 1.0, 1.4])

_COLUMNS = {
    "phase": "single_phase",
    "temperature": "T_K",
    "densities": ("rho_Fe", "rho_Cu", "rho_Ni"),
    "composition": ("x_Fe", "x_Cu", "x_Ni"),
    "active": "active",
    "mobility": "M_k0",
    "shells": "k_shells",
    "structure": "S_k",
    "bulk_target": "Scc0",
    "bulk_sigma": None,
    "bulk_ok": None,
}
_EOS_COLUMNS = {
    "x_columns": ("x_Fe", "x_Cu", "x_Ni"), "T_column": "T_K",
    "n_columns": ("n_atoms_per_A3",),
    "status_column": "status", "status_ok": "ok",
}
_ROW_WEIGHT = {"value": None, "error": "Scc0_err", "floor": 0.02}


def _spd(m, seed, scale=1.0):
    a = np.random.default_rng(seed).standard_normal((m, m))
    return scale * (a @ a.T + m * np.eye(m))


def _write_tables(root):
    """``m_table``, ``s_table`` and the manifold, in the layout a ternary campaign writes: NaN
    outside each row's active block, ``Scc0`` on the first ``|P| - 1`` present species."""
    root.mkdir(parents=True, exist_ok=True)
    n_rows = _X.shape[0]
    active = _X > 0
    rho = _X * _TOTAL[:, None]
    M = np.full((n_rows, 3, 3), np.nan)
    S = np.full((n_rows, len(_SHELLS), 3, 3), np.nan)
    Scc0 = np.full((n_rows, 2, 2), np.nan)
    for r in range(n_rows):
        p = np.nonzero(active[r])[0]
        M[r][np.ix_(p, p)] = _spd(p.size, 10 + r, 1e-3)
        for s in range(len(_SHELLS)):
            S[r, s][np.ix_(p, p)] = _spd(p.size, 20 + 3 * r + s, 0.1)
        m = p.size - 1
        if m:
            Scc0[r, :m, :m] = _spd(m, 40 + r, 0.05)
    common = dict(
        rho_Fe=rho[:, 0], rho_Cu=rho[:, 1], rho_Ni=rho[:, 2],
        x_Fe=_X[:, 0], x_Cu=_X[:, 1], x_Ni=_X[:, 2],
        active=active, T_K=_T, single_phase=np.ones(n_rows, dtype=bool))
    np.savez(root / "M_table.npz", M_k0=M, **common)
    np.savez(root / "S_table.npz", k_shells=np.tile(_SHELLS, (n_rows, 1)), S_k=S, Scc0=Scc0,
             Scc0_err=np.array([0.05, 0.08, np.nan]), **common)
    lines = ["x_Fe,x_Cu,x_Ni,T_K,n_atoms_per_A3,P_GPa,status"]
    lines += [f"{_X[r, 0]},{_X[r, 1]},{_X[r, 2]},{_T[r]},{_TOTAL[r]},0.0001,ok"
              for r in range(n_rows)]
    (root / "eos_n_x_T.csv").write_text("\n".join(lines) + "\n")


def _tables_block():
    return dict(
        key="pressure", m_table={0.0: "tier/M_table.npz"}, s_table={0.0: "tier/S_table.npz"},
        eos_csvs={0.0: "tier/eos_n_x_T.csv"}, pressure_unit=1.0 / 160.2176,
        columns=dict(_COLUMNS), eos_columns=dict(_EOS_COLUMNS),
        row_weight=dict(_ROW_WEIGHT), pressure_floor=0.5, w0={"route": "evaluator"})


def _defaults():
    """The driver's knobs at toy values, with the Fe-B anchor and hinge weights and ``s_metric``."""
    return {
        "estimator": "weak", "sigma": 2.0, "k_fit_stat": 1.5, "k_max": 2.0,
        "alpha_loss": 0.0, "h_inv_eps": 1e-6, "lr": 1e-3, "weight_decay": 0.0,
        "warmup_epochs": 1, "anneal_epochs": 1,
        "source_loss_weights": {"toy": 1.0},
        "lambda_M": 0.5, "lambda_S": 0.03, "lambda_bulk": 0.01, "lambda_P": 5.0,
        "lambda_conv": 0.0, "lambda_gamma": 0.0,
        "lambda_wpsd": 0.3, "wpsd_kappa": 2.0, "wpsd_margin": 0.0, "wpsd_k_max": 3.0,
        "wpsd_n_k": 8,
        "training": {
            "half_width": 2, "n_states": 3, "stride": 1,
            "savgol_window": None, "savgol_poly": None,
            "run_weighting": "uniform", "run_weight_probe_every": 10,
            "val_split": "random", "val_labels": (), "val_fraction": 0.1, "split_seed": 0,
            "batch_size": 2, "num_workers": 0, "pin_memory": False, "drop_last": False,
            "order": "shuffled", "lambda_dyn": 1.0, "bulk_residual": "s_metric",
            "eta_min": 1e-6, "grad_clip": 1.0,
            "tables": _tables_block(),
        },
    }


def _system(root):
    return ternary_system(
        functional_kwargs=dict(grid=_GRID),
        table_keys={"rho": ("rho_Fe", "rho_Cu", "rho_Ni"), "x": "x_Fe"},
        paths=Paths(system="ternary_toy", raw_default=str(root)),
        defaults=_defaults())


def _write_modes(directory, rng, *, absent=None):
    """One run's three-channel ``modes.npz``: about ``0.028`` atoms per cubic Angstrom per channel
    in a ``9^3`` box, Poisson-sized fluctuations, channel ``absent`` identically zero."""
    labels = np.array([(nx, ny, nz) for nx in (-1, 0, 1) for ny in (-1, 0, 1)
                       for nz in (0, 1, 2)], dtype=np.int64)
    n_atoms = 0.028 * _BOX ** 3
    shape = (_FRAMES, labels.shape[0], 3)
    amplitudes = np.sqrt(n_atoms / 2) * (rng.standard_normal(shape)
                                         + 1j * rng.standard_normal(shape))
    zero = np.flatnonzero((labels == 0).all(axis=1))
    amplitudes[:, zero] = n_atoms
    if absent is not None:
        amplitudes[..., absent] = 0.0
    ModesRecord(amplitudes=amplitudes.astype(np.complex64), labels=labels,
                boxes=np.full((_FRAMES, 3), _BOX), T=1600.0, dt_frame=0.1,
                provenance={"composition": {"x_Fe": 0.5 if absent else 1 / 3},
                            "cache_dir": str(directory)}).save(directory)


@pytest.fixture
def toy(tmp_path):
    _write_tables(tmp_path / "tier")
    tree = tmp_path / "fields" / "modes_toy"
    _write_modes(tree / "run_a", np.random.default_rng(0))
    _write_modes(tree / "run_b", np.random.default_rng(1), absent=1)
    sources = (SourceSpec(name="toy", root=str(tree), pattern="run_*", grid=_GRID,
                          loss_weight=1.0, exclude_tags=()),)
    return _system(tmp_path), sources


# ---------------------------------------------------------------------------
# 6. the ternary System
# ---------------------------------------------------------------------------

def test_the_ternary_declaration_with_its_tables_validates_and_builds(toy):
    system, _ = toy
    assert system.n_species == 3 and system.species == SPECIES
    model = build(system)
    assert model.n_species == 3
    assert build_mobility(system).n_species == 3
    tables = AnchorTables.load(system, **system.defaults["training"]["tables"])
    assert tables.bulk_active is not None


# ---------------------------------------------------------------------------
# 4. the anchor tables
# ---------------------------------------------------------------------------

def test_the_loader_keeps_the_rows_each_anchor_should_and_says_nothing(toy, caplog):
    system, _ = toy
    with caplog.at_level(logging.WARNING):
        tables = AnchorTables.load(system, **system.defaults["training"]["tables"])
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text
    # the pure row feeds the manifold alone
    assert tables.rows() == {"anchor_M": 2, "anchor_S": 2, "anchor_bulk": 2, "anchor_P": 3}
    assert tables.terms() == ("L_M", "L_S", "L_bulk", "L_P")

    present = torch.tensor(_X[:2] > 0)
    for mask in (tables.mobility_active, tables.shell_active, tables.bulk_active):
        assert torch.equal(mask, present)
    assert torch.equal(tables.bulk_zmask, torch.tensor([[True, True], [True, False]]))
    assert tables.bulk_zvec.shape == (2, 3, 2)
    assert tables.bulk_target.shape == tables.bulk_target_rootinv.shape == (2, 2, 2)
    assert tables.shell_target.shape == (2, len(_SHELLS), 3, 3)
    assert bool(tables.shell_mask.all())        # no shell of the binary row is dropped
    for tensor in (tables.mobility_target, tables.shell_target, tables.shell_target_rootinv,
                   tables.bulk_zvec, tables.bulk_target, tables.bulk_target_rootinv,
                   tables.pressure_target):
        assert torch.isfinite(tensor).all()
    # the binary row's absent channel is outside every target
    assert torch.equal(tables.mobility_target[1, 1], torch.zeros(3))
    assert torch.equal(tables.shell_target[1, :, 1], torch.eye(3)[1].expand(len(_SHELLS), 3))


def test_the_absent_species_rows_shell_target_is_the_active_blocks_inverse(toy, tmp_path):
    system, _ = toy
    tables = AnchorTables.load(system, **system.defaults["training"]["tables"])
    kB = system.constants["kB"]
    S = np.load(tmp_path / "tier" / "S_table.npz")["S_k"]
    p = [0, 2]
    for s in range(len(_SHELLS)):
        expected = kB * _T[1] * np.linalg.inv(S[1, s][np.ix_(p, p)])
        got = tables.shell_target[1, s].double().numpy()[np.ix_(p, p)]
        assert np.allclose(got, expected, rtol=1e-5)


def test_the_loader_records_every_file_it_read_with_its_digest(toy, tmp_path):
    system, _ = toy
    tables = AnchorTables.load(system, **system.defaults["training"]["tables"])
    root = tmp_path / "tier"
    expected = {"m_table": "M_table.npz", "s_table": "S_table.npz", "eos_csvs": "eos_n_x_T.csv"}
    for role, name in expected.items():
        ((path, digest),) = tables.provenance[role]
        assert path == str(root / name)
        assert digest == hashlib.sha256((root / name).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 3. one training step
# ---------------------------------------------------------------------------

def test_one_step_trains_every_term_finitely_and_reloads_strictly(toy, tmp_path):
    system, sources = toy
    # the first step runs at lr times the warm-up's start factor, 1e-6: lr = 1e3 makes it 1e-3
    run = fit(system, run_name="one_step", sources=sources, steps=1, seed=0,
              resume_optimizer=False, root=tmp_path / "data", log_every_step=True,
              config_overrides={"lr": 1e3})
    # the starting weights, built as fit builds them: under the run's own model stream
    cfg = TrainConfig(**{**_config_from_system(system), "seed": 0})
    with seeded_rng(cfg.seed_for("model")):
        before = build(system).state_dict()

    terms = {"L_dyn", "L_M", "L_S", "L_bulk", "L_P", "L_W"}
    man = json.loads((run / "MANIFEST.json").read_text())
    assert set(man["terms_trained"]) == terms
    assert man["declared_weights_without_data"] == {}
    assert man["anchor_tables"]["rows"] == {"anchor_M": 2, "anchor_S": 2, "anchor_bulk": 2,
                                            "anchor_P": 3}
    steps = json.loads((run / "steps.json").read_text())
    assert len(steps["loss"]) == 1 and np.isfinite(steps["loss"][0])
    (step,) = steps["terms"]
    assert terms <= set(step), step
    assert all(np.isfinite(step[name]) for name in terms), step
    # every term is in the total the step minimised, at its declared weight
    weights = _defaults()
    weighted = {"L_M": weights["lambda_M"], "L_S": weights["lambda_S"],
                "L_bulk": weights["lambda_bulk"], "L_P": weights["lambda_P"],
                "L_W": weights["lambda_wpsd"]}
    assert all(step[name] > 0.0 for name in weighted), step
    contributions = [w * step[name] for name, w in weighted.items()]
    expected = step["L_dyn"] + sum(contributions)
    # the terms span five decades: a tenth of the smallest one, so leaving any one out fails
    assert steps["loss"][0] == pytest.approx(expected, rel=0.0, abs=0.1 * min(contributions))

    path = run / "final.ckpt"
    declared = Checkpoint(path=str(path.relative_to(tmp_path)),
                          md5=hashlib.md5(path.read_bytes()).hexdigest())
    reloaded = build(system)
    _load_weights_into(reloaded, declared, tmp_path)
    after = reloaded.state_dict()
    assert set(after) == set(before)
    for value in after.values():
        if value.is_floating_point():
            assert torch.isfinite(value).all()
    moved = {k: float((after[k] - before[k]).abs().max()) for k in before
             if before[k].is_floating_point()}
    # an Adam step moves each parameter with a gradient by about the learning rate
    assert max(moved.values()) > 1e-4, moved
    assert sum(v > 1e-5 for v in moved.values()) >= len(moved) // 2, moved


def test_the_starting_weights_the_step_test_compares_against_are_the_runs_own(toy, tmp_path):
    """At ``lr = 0`` the step moves nothing, so the reconstructed start must equal the checkpoint:
    the "parameters change" check above is against the weights the run really started from."""
    system, sources = toy
    run = fit(system, run_name="frozen", sources=sources, steps=1, seed=0,
              resume_optimizer=False, root=tmp_path / "data",
              config_overrides={"lr": 0.0})
    cfg = TrainConfig(**{**_config_from_system(system), "seed": 0})
    with seeded_rng(cfg.seed_for("model")):
        before = build(system).state_dict()
    path = run / "final.ckpt"
    reloaded = build(system)
    _load_weights_into(reloaded, Checkpoint(path=str(path.relative_to(tmp_path)),
                                            md5=hashlib.md5(path.read_bytes()).hexdigest()),
                       tmp_path)
    after = reloaded.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)



def test_one_step_with_the_subspace_hinge_and_the_anchor_state_penalty(toy, tmp_path):
    """``wpsd_subspace``, ``wpsd_k_sat`` and ``conv_states="anchors"`` declared: one step trains
    ``L_W`` and ``L_conv`` beside the anchors, every term finite, the total the weighted sum, and the
    run records the new keys where it records the hinge and the penalties."""
    import yaml

    system, sources = toy
    declared = {"wpsd_subspace": "number_conserving", "wpsd_k_sat": 1.5,
                "lambda_conv": 1.0, "conv_penalty": "hinge", "conv_margin": 100.0,
                "conv_states": "anchors", "conv_k": (8, 3.96)}
    system = dataclasses.replace(system, defaults={**system.defaults, **declared})
    run = fit(system, run_name="f_both", sources=sources, steps=1, seed=0,
              resume_optimizer=False, root=tmp_path / "data", log_every_step=True,
              config_overrides={"lr": 1e3})
    terms = {"L_dyn", "L_M", "L_S", "L_bulk", "L_P", "L_W", "L_conv"}
    man = json.loads((run / "MANIFEST.json").read_text())
    assert set(man["terms_trained"]) == terms
    assert man["declared_weights_without_data"] == {}
    assert man["kernel_hinge"]["subspace"] == "number_conserving"
    assert man["kernel_hinge"]["k_sat"] == 1.5
    assert man["penalties"] == {"penalty_seed": None, "terms": ["L_conv"], "gamma": None,
                                "conv_states": {"states": "anchors", "n_k": 8, "k_max": 3.96,
                                                "rows": 2}}
    extra = yaml.safe_load((run / "hparams.yaml").read_text())["extra_experiment_config"]
    assert extra["wpsd_subspace"] == "number_conserving" and extra["conv_states"] == "anchors"
    assert extra["conv_k"] == [8, 3.96]

    steps = json.loads((run / "steps.json").read_text())
    (step,) = steps["terms"]
    assert terms <= set(step), step
    assert all(np.isfinite(step[name]) for name in terms), step
    weights = {**_defaults(), **declared}
    weighted = {"L_M": weights["lambda_M"], "L_S": weights["lambda_S"],
                "L_bulk": weights["lambda_bulk"], "L_P": weights["lambda_P"],
                "L_W": weights["lambda_wpsd"], "L_conv": weights["lambda_conv"]}
    # a margin of 100 is violated on every row, so the penalty is active
    assert step["L_conv"] > 0.0
    contributions = [w * step[name] for name, w in weighted.items() if step[name] > 0.0]
    expected = step["L_dyn"] + sum(w * step[name] for name, w in weighted.items())
    assert steps["loss"][0] == pytest.approx(expected, rel=0.0, abs=0.1 * min(contributions))
