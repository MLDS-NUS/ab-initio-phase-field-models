"""The published model pinned, and everything it needs declared beside it.

Four kinds of test live here.

The CHECKPOINT tests pin the tracked file by digest, and refuse every shape a
silent substitution can take: repointing the System, swapping the bytes, a file
missing under a root that should hold it. A checkout that leaves the published
checkpoints out altogether skips them, naming the file.

The DECLARATION tests pin what ``experiments/feb/system.py`` says. They are
change detectors in the sense ``test_system_constants.py`` already uses --
not an independent source for any value, because the provenance lives beside
each constant.

The AGREEMENT tests are not change detectors. They open the pinned
checkpoint and compare every declared knob against the ``hyper_parameters``
dict saved inside it, and they account for every key of that dict in both
directions, so that "declared but unchecked" and "saved but undeclared" are
lists somebody wrote rather than gaps.

The MEASURED tests read the Fe-B raw root: the kernel-shape floor stored in
this system's own anchor tables, the 1024 archived engine decks, and the
anchor tables the published model's buffers were built from. They are marked
``env`` and skip, naming the root, when it is absent.
"""
from __future__ import annotations

import hashlib
import pathlib
import re

import pytest

from aipf.system import Checkpoint, load

import declared_roots

# --------------------------------------------------------------------------
# The published model, spelled out
# --------------------------------------------------------------------------

#: A checkpoint path relative to a root, for the declaration's own refusals below.
PUBLISHED_PATH = "runs/published/final.ckpt"

#: Digest of the tracked published model's bytes, measured with ``md5sum``.
PUBLISHED_MD5 = "663fcd195951e9726148fee9d4e4a8b4"

#: Directories that hold CODE.
CODE_DIRS = ("src", "tests", "experiments")

#: Checkpoints of the ``periodic-epoch=<NNN>.ckpt`` shape that a file under
#: ``CODE_DIRS`` may name. Empty: the published model is the tracked
#: ``final.ckpt``, and an entry here would be a reviewed decision.
ALLOWED_OTHER_CHECKPOINTS: tuple[str, ...] = ()

_REPO = pathlib.Path(__file__).resolve().parents[2]

_PERIODIC_LITERAL = re.compile(r"periodic-epoch=\d+\.ckpt")


def _feb():
    return load("feb")


def _md5(path: pathlib.Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scratch_or_skip(system):
    """The system's raw root, or a skip naming it."""
    return declared_roots.raw_or_skip(system.name)


def _published_hparams(system):
    """The published model's saved hyper_parameters, from the tracked copy the repository carries."""
    declared_roots.published_or_skip("feb")
    torch = pytest.importorskip("torch")
    path = system.resolve_checkpoint()
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------
# The published model, declared
# --------------------------------------------------------------------------


def test_the_system_declares_the_published_model_checkpoint():
    """Repointing the System at another file fails here, not silently."""
    s = _feb()
    assert s.checkpoint is not None, (
        "the system declares no checkpoint: its published numbers then come "
        "from whichever file a caller happens to pick")
    assert s.checkpoint.path is None     # the tracked copy: declared by its digest alone
    assert s.checkpoint.md5 == PUBLISHED_MD5


def test_the_checkpoint_is_the_tracked_copy():
    """Declared by digest alone, so it resolves to the repository's tracked file."""
    from aipf.paths import published_checkpoint

    assert _feb().checkpoint_file() == published_checkpoint("feb")


def test_a_raw_root_override_does_not_move_the_tracked_checkpoint(monkeypatch,
                                                                  tmp_path):
    s = _feb()
    before = s.checkpoint_file()
    monkeypatch.setenv("AIPF_RAW_FEB", str(tmp_path))
    assert s.checkpoint_file() == before


def test_the_file_on_disk_still_carries_the_declared_digest():
    """Swapping the file's CONTENTS fails here. A checkout that leaves the published checkpoints out
    skips, naming the file."""
    declared_roots.published_or_skip("feb")
    s = _feb()
    path = s.verify_checkpoint()
    assert _md5(path) == PUBLISHED_MD5


def test_a_missing_file_under_a_present_root_is_refused_not_skipped(tmp_path):
    """Matched on the refusal's own wording, not on the filename.

    ``open`` raises the same exception type with the same filename in its
    message, so a filename match cannot tell "this code refused" from "this
    code dropped the check and the file system raised on its way past".
    """
    ckpt = Checkpoint(path=PUBLISHED_PATH, md5=PUBLISHED_MD5)
    with pytest.raises(FileNotFoundError, match="missing or renamed") as info:
        ckpt.verify_under(tmp_path)
    assert str(tmp_path / PUBLISHED_PATH) in str(info.value)


def test_a_present_file_with_other_bytes_is_refused_and_both_digests_named(
        tmp_path):
    target = tmp_path / PUBLISHED_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the published model")
    with pytest.raises(ValueError) as info:
        Checkpoint(path=PUBLISHED_PATH, md5=PUBLISHED_MD5).verify_under(tmp_path)
    message = str(info.value)
    assert PUBLISHED_MD5 in message
    assert _md5(target) in message


def test_no_code_file_names_a_different_periodic_checkpoint():
    """The declaration is the only opinion the code holds about this shape.

    A second copy of the path is a second thing to repoint, and repointing
    it would leave this file's own assertions green.
    """
    scanned, hits = 0, []
    for directory in CODE_DIRS:
        for path in sorted((_REPO / directory).rglob("*")):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in _PERIODIC_LITERAL.finditer(text):
                found = match.group(0)
                if found not in ALLOWED_OTHER_CHECKPOINTS:
                    line = text[:match.start()].count("\n") + 1
                    hits.append(f"{path}:{line}: {found}")
    assert scanned > 0, f"scanned no files under {_REPO}"
    assert not hits, hits


# --------------------------------------------------------------------------
# The run the published model ended
# --------------------------------------------------------------------------


def test_the_published_model_is_the_final_epoch_the_config_asked_for():
    """`max_epochs: 60` and a save every fifth epoch put the last save at 59."""
    s = _feb()
    state = _published_hparams(s)
    assert state["epoch"] == 59
    assert state["global_step"] == 12960
    assert s.defaults["max_epochs"] == 60


# --------------------------------------------------------------------------
# The kernel-shape temperature floor
# --------------------------------------------------------------------------


def test_the_kernel_shape_temperature_floor_is_declared_per_pressure():
    """The published pipeline's own cut, 2300 K in both places it is written.

    Without it ``kappa`` and ``M_kappa0`` cannot be rebuilt: the universal
    shape is averaged over the runs at or above this temperature and every
    other run's amplitude is a one-parameter fit against that average.
    """
    feb = load("feb")
    assert feb.anchor_rules.shape_T_min_by_pressure == {0: 2300, 5: 2300,
                                                        10: 2300}


@pytest.mark.env
def test_the_shape_floor_is_the_one_stored_in_this_systems_own_tables():
    """Measured, not transcribed: every aggregate table records its own.

    The other two-species system's four values survive only as prose, and
    that asymmetry is how the absence was noticed in the first place. Here
    the number is in the data, so the declaration is checked against it.
    """
    feb = load("feb")
    root = _scratch_or_skip(feb)
    numpy = pytest.importorskip("numpy")
    fields = root / "fields"
    stored = set()
    for tree in feb.constants["table_trees"]:
        table = fields / tree / "M_table.npz"
        if not table.is_file():
            continue
        with numpy.load(table, allow_pickle=True) as data:
            stored.add(float(data["kappa_T_min"]))
    assert stored, "no aggregate mobility table was readable"
    assert stored == set(float(v) for v
                         in feb.anchor_rules.shape_T_min_by_pressure.values())


def test_the_two_temperature_cuts_are_two_different_tables():
    """This system has no anchor temperature cut and does have a shape floor.

    They are different rules with the same shape, and the one way this
    declaration can be got wrong is by merging them: the anchor cut is empty
    here because eligibility is decided by a gate column instead.
    """
    feb = load("feb")
    assert feb.anchor_rules.T_min_by_pressure == {}
    assert set(feb.anchor_rules.shape_T_min_by_pressure) == {0, 5, 10}
    assert feb.t_min(0) is None
    assert feb.shape_t_min(0) == 2300.0


def test_the_shape_floor_is_looked_up_the_way_the_anchor_cut_is():
    feb = load("feb")
    assert feb.shape_t_min(0) == 2300.0
    assert feb.shape_t_min(0.0) == 2300.0
    assert isinstance(feb.shape_t_min(10), float)
    assert feb.shape_t_min(15) is None


def test_the_shape_floor_is_not_the_other_systems():
    """Its four values are 7500 to 10000 K, and this material's whole
    measured range tops out at 2600 K.

    Nothing in this system's own files could catch a carried-over number,
    which is why it is asserted against the other system's declaration.
    """
    feb, hhe = load("feb"), load("hhe")
    assert set(feb.anchor_rules.shape_T_min_by_pressure).isdisjoint(
        hhe.anchor_rules.shape_T_min_by_pressure)
    assert set(feb.anchor_rules.shape_T_min_by_pressure.values()).isdisjoint(
        set(hhe.anchor_rules.shape_T_min_by_pressure.values()))


# --------------------------------------------------------------------------
# The engine knobs
# --------------------------------------------------------------------------


def test_both_campaigns_the_published_model_reads_declare_their_engine_settings():
    """The published model's own hparams name two data families, so two campaigns.

    ``s_table``/``m_table`` are built from the cube trajectories and
    ``eos_csvs`` from the equation-of-state grid. They were run by two
    different workers, so two declarations, even though every value in them
    turns out to agree.
    """
    feb = load("feb")
    assert set(feb.md_settings) == {"production", "eos"}


def test_the_production_engine_settings_are_the_published_trajectories():
    """The worker that ran all 464 of them."""
    md = load("feb").md("production")
    assert md.units == "metal"
    assert md.skin == 2.0
    assert md.neigh_modify == "every 10 delay 0 check yes"
    assert md.thermo_every == 100
    assert md.T_damp == 0.1
    assert md.P_damp == 1.0


def test_this_systems_two_campaigns_agree_on_every_knob():
    """MEASURED, and the opposite of the other two-species system.

    There the equation-of-state grid ran at different damping times and a
    different cadence from the trajectories, deliberately and on the record.
    Here the two workers agree on every value, over all 1024 archived
    decks, so a test that merely asserted "the two differ" would be importing the
    other system's history as though it were a rule.
    """
    feb = load("feb")
    production, eos = feb.md("production"), feb.md("eos")
    assert production == eos


@pytest.mark.env
def test_the_engine_knobs_are_what_every_archived_deck_says():
    """Parsed out of the 1024 decks on disk, not read off the worker.

    A worker can be edited after a campaign; the decks it wrote cannot.
    """
    feb = load("feb")
    root = _scratch_or_skip(feb)
    fields = {
        "units": re.compile(r"^units\s+(\S+)", re.M),
        "neighbor": re.compile(r"^neighbor\s+(.+?)\s*$", re.M),
        "neigh_modify": re.compile(r"^neigh_modify\s+(.+?)\s*$", re.M),
        "npt": re.compile(r"^fix\s+\S+\s+all\s+npt\s+temp\s+\S+\s+\S+\s+"
                          r"(\S+)\s+iso\s+(\S+)\s+\S+\s+(\S+)", re.M),
    }
    seen = {name: set() for name in fields}
    decks = 0
    for pattern in ("cube_data_*GPa/*/in.lammps", "eos_*GPa/*/*/in.lammps"):
        for deck in root.glob(pattern):
            decks += 1
            text = deck.read_text(errors="replace")
            for name, regex in fields.items():
                seen[name].update(regex.findall(text))
    if decks == 0:
        pytest.skip("no archived engine deck under this root")
    md = feb.md("production")
    assert seen["units"] == {md.units}
    assert seen["neighbor"] == {f"{md.skin} bin"}
    assert seen["neigh_modify"] == {md.neigh_modify}
    assert {t for t, _, _ in seen["npt"]} == {str(md.T_damp)}
    assert {p for _, _, p in seen["npt"]} == {str(md.P_damp)}


def test_the_deck_pressure_is_bar_and_the_factor_is_ten_thousand():
    """A request carries GPa and the deck's `fix npt` wants bar."""
    md = load("feb").md("production")
    assert md.pressure_scale == 1e4
    assert md.deck_pressure(5.0) == 50_000.0
    # A second pressure, so the factor cannot be a coincidence of one value.
    assert md.deck_pressure(10.0) == 100_000.0


def test_the_zero_pressure_floor_is_the_workers_and_not_the_conversions():
    """A RECORDED GAP, with the number that shows it.

    This system's reference pressure is zero, and both workers write
    ``1.0`` bar there rather than ``0`` (every archived 0 GPa deck says
    ``iso 1 1``). ``deck_pressure`` is a pure multiplication and answers
    ``0.0``. Nothing in this package implements the floor; this test pins
    the discrepancy so that closing it is a deliberate change rather than a
    surprise, and it fails the day the conversion learns about it.
    """
    md = load("feb").md("production")
    assert md.deck_pressure(0.0) == 0.0
    assert load("feb").constants["P_GPa"] == 0.0


def test_every_campaign_converts_the_pressure_the_same_way():
    feb = load("feb")
    for name in feb.md_settings:
        assert feb.md(name).deck_pressure(10.0) == 100_000.0


def test_an_unknown_campaign_is_refused_and_the_known_ones_are_named():
    feb = load("feb")
    with pytest.raises(KeyError) as info:
        feb.md("slab")
    assert "production" in str(info.value) and "eos" in str(info.value)


# --------------------------------------------------------------------------
# The batch shape
# --------------------------------------------------------------------------


def test_no_batch_shape_is_declared():
    """Where a job is submitted is a site fact (``aipf.site``, ``pbs_*``), not the system's."""
    assert not hasattr(load("feb"), "batch")


# --------------------------------------------------------------------------
# The field trees and the tables inside them
# --------------------------------------------------------------------------


def test_the_tree_the_published_model_actually_reads_is_declared():
    """`index_fields` marks every tree no declared role names as archived.

    Before this declaration the one tree the published model's own ``s_table`` and
    ``m_table`` name -- ``fields/training_derived`` -- was reported as
    archived, while six trees it never opens were reported as tables.
    """
    feb = load("feb")
    assert "training_derived" in feb.constants["table_trees"]


@pytest.mark.env
def test_the_index_now_calls_the_published_models_tree_authoritative():
    """The declaration's whole point, measured on the real tree."""
    from aipf.data import index
    feb = load("feb")
    _scratch_or_skip(feb)
    roles = {t["name"]: t["role"] for t in index.index_fields(feb)["trees"]}
    assert roles.get("training_derived") == "tables"
    for name in feb.constants["modes_trees"]:
        assert roles.get(name) == "modes", name
    for name in feb.constants["fdt_runs_trees"]:
        assert roles.get(name) == "fdt_runs", name


def test_the_derived_tables_are_named_because_one_tree_holds_many():
    """Twenty-eight files in one directory, six of which this declaration
    names. The run reads a seventh, the loader burn-in overlay.

    ``M_table_v2_trainexcl.npz``, ``..._trainexcl_gstable.npz``,
    ``..._trainexcl_fixed.npz``, ``..._trainexcl_xle06.npz`` and
    ``..._trainexcl_xle07.npz`` are five different anchor sets side by side.
    Declaring the directory is not enough to find the run's own tables, which is
    the difference from the other two-species system, whose trees hold one
    ``M_table.npz`` each.
    """
    feb = load("feb")
    assert feb.constants["m_table_files"] == (
        "M_table_v2_trainexcl.npz", "M_table_5GPa_trainexcl.npz",
        "M_table_10GPa_trainexcl.npz")
    assert feb.constants["s_table_files"] == (
        "S_table_v2_trainexcl_gstable.npz",
        "S_table_5GPa_trainexcl_gstable.npz",
        "S_table_10GPa_trainexcl_gstable.npz")
    # The M tables are the plain exclusion set and the S tables the gated
    # one. Both spellings exist for both quantities in that directory.
    assert not any(name.endswith("_gstable.npz")
                   for name in feb.constants["m_table_files"])


def test_the_equation_of_state_files_do_not_share_one_name():
    """The 0 GPa grid is `_v2` and the other two are not.

    The apex cell was filled into a new file rather than over
    the live one, precisely so a pinned config could not change underneath
    a finished run. A reader who assumed one basename would read the
    pre-apex 0 GPa grid.
    """
    feb = load("feb")
    assert feb.constants["eos_trees"] == ("eos_0GPa", "eos_5GPa", "eos_10GPa")
    assert feb.constants["eos_csv_files"] == (
        "eos_n_x_T_v2.csv", "eos_n_x_T.csv", "eos_n_x_T.csv")
    assert len(set(feb.constants["eos_csv_files"])) == 2


def test_the_trees_are_declared_in_the_pressure_order_the_published_model_lists():
    feb = load("feb")
    pressures = feb.constants["eos_pressures_GPa"]
    assert pressures == (0.0, 5.0, 10.0)
    for key in ("modes_trees", "fdt_runs_trees", "eos_trees"):
        names = feb.constants[key]
        assert len(names) == len(pressures), key
        assert all(f"{int(p)}GPa" in name
                   for p, name in zip(pressures, names)), key


@pytest.mark.env
def test_the_files_on_disk_are_the_ones_named():
    """Named files, opened. A basename that no longer exists is a failure."""
    feb = load("feb")
    root = _scratch_or_skip(feb)
    derived = root / "fields" / "training_derived"
    if not derived.is_dir():
        pytest.skip(f"{derived} not present")
    for key in ("m_table_files", "s_table_files"):
        for name in feb.constants[key]:
            assert (derived / name).is_file(), name
    for tree, name in zip(feb.constants["eos_trees"],
                          feb.constants["eos_csv_files"]):
        assert (root / tree / name).is_file(), f"{tree}/{name}"


# --------------------------------------------------------------------------
# The published model's own knobs
# --------------------------------------------------------------------------


def test_the_excess_head_is_the_joint_network_and_its_shape_is_declared():
    """One network over all channels and the temperature, not two heads.

    ``h_u_joint`` and ``n_hidden_joint`` are the only two numbers that fix
    it, and without them ``f_form`` names a form with no size.
    """
    d = load("feb").defaults
    assert d["f_form"] == "joint_mlp"
    assert d["h_u_joint"] == 64
    assert d["n_hidden_joint"] == 4
    assert d["activation"] == "gelu"


def test_the_split_head_widths_are_declared_although_they_are_inert():
    """They are in the checkpoint and they build nothing in this run.

    The joint form excludes the split heads, so ``h_u`` and ``h_g`` size
    networks that are never constructed. Declared anyway, because half of
    ``model_kwargs`` declared is the state in which a reader believes the
    file is complete.
    """
    d = load("feb").defaults
    assert (d["h_u"], d["h_g"], d["h_w"], d["h_m"]) == (16, 32, 16, 16)


def test_the_input_scaling_and_its_reference_densities_are_declared():
    """z = rho/rho_ref - 1 into every density network.

    ``rho_ref`` is this material's own mid-isobar density and is two orders
    of magnitude below the other two-species system's. An inherited value
    would put every network input at a different place on its own curve.
    """
    d = load("feb").defaults
    assert d["rho_ref"] == (0.05, 0.05)
    assert d["fexc_input_scale"] is True
    assert d["m_input_scale"] is True
    assert d["rho_eps"] == 1e-5
    assert d["rho_ref"] != load("hhe").defaults["rho_ref"]


def test_the_mobility_temperature_factor_and_its_table_start_are_declared():
    """Arrhenius with two learned barriers, initialised from a measured table."""
    d = load("feb").defaults
    assert d["m_form"] == "mlp_scaled"
    assert d["m_T_form"] == "arrhenius"
    assert d["arrhenius_shared_Ea"] is False
    assert d["T_ref"] == 1800.0
    assert d["m_column"] == "M_k0"
    assert d["m_init_from_table"] is True


def test_the_two_reference_temperatures_are_both_declared_and_equal_here():
    """`fexc_T_ref` and `T_ref` are different knobs that happen to agree.

    In the other two-species system they are 8000 and 10000, and the
    confusion between the names has cost this project time. Here they are
    both 1800, which is the one arrangement in which a reader cannot tell
    from a single number which knob they are looking at.
    """
    d = load("feb").defaults
    assert d["fexc_T_ref"] == 1800.0
    assert d["T_ref"] == 1800.0


def test_the_seven_loss_weights_are_declared_including_the_kernel_floor():
    """Six the other system has, and a seventh it does not.

    ``lambda_wpsd`` weights a hinge on the kernel's own eigenvalue floor.
    The other two-species published model has no such term, and this package maps
    it to the non-canonical ``L_W``.
    """
    d = load("feb").defaults
    assert d["lambda_M"] == 0.5
    assert d["lambda_S"] == 0.03
    assert d["lambda_bulk"] == 0.01
    assert d["lambda_P"] == 5.0
    assert d["lambda_conv"] == 0.0
    assert d["lambda_gamma"] == 0.0
    assert d["lambda_wpsd"] == 0.3


def test_the_kernel_floors_own_shape_is_declared():
    """A weight without the band it acts on describes no penalty at all."""
    d = load("feb").defaults
    assert d["wpsd_kappa"] == 2.0
    assert d["wpsd_margin"] == 0.0
    assert d["wpsd_k_max"] == 3.0
    assert d["wpsd_n_k"] == 32


def test_the_two_penalties_this_run_switched_off_keep_their_shapes():
    """`lambda_conv` and `lambda_gamma` are zero and the shapes are declared.

    A later arm that turns either back on needs the hinge, the margin and
    the sample counts it was turned off with, and reading a zero weight as
    "the shape is unknown" is how a second arm ends up being a third one.
    """
    d = load("feb").defaults
    assert d["conv_penalty"] == "hinge"
    assert d["conv_margin"] == 0.05
    assert d["conv_samples"] == 1024
    assert d["gamma_pt_samples"] == 16


def test_the_anchor_gate_and_the_row_weighting_are_declared():
    """This system gates its anchors by an eligibility COLUMN, not by T.

    The column is built into the ``_gstable`` tables by
    scripts/build_gamma_stable_tables.py and named in the config. Without
    the name, the same tables answer a different question.
    """
    d = load("feb").defaults
    assert d["anchor_gate"] == "anchor_eligible_g2new_gamma1_fixed"
    assert d["anchor_row_weight"] == "scc0_err"
    assert d["p_ref_gpa"] == 10.0
    assert d["stat_metric"] == "rel_frob"
    assert d["k_fit_stat"] == 1.5


def test_the_optimiser_and_the_schedule_are_declared():
    """The cosine leg does NOT finish when training stops.

    ``anneal_epochs`` is 200 and ``max_epochs`` is 60, so the published model is
    taken a third of the way down the schedule. In the other two-species
    system the two are equal, and a reader who carried that arrangement over
    would anneal this run three times too fast.
    """
    d = load("feb").defaults
    assert d["lr"] == 5e-4
    assert d["weight_decay"] == 0.0
    assert d["wd_ghat"] == 0.01
    assert d["warmup_epochs"] == 5
    assert d["anneal_epochs"] == 200
    assert d["max_epochs"] == 60
    assert d["anneal_epochs"] != d["max_epochs"]
    assert d["grad_clip"] == 1.0


def test_every_source_carries_the_same_weight_and_the_split_is_by_phase():
    """Six sources: two phase classes at each of three pressures.

    The other two-species system splits its sources by GEOMETRY (slab and
    cube). This one has cubes only and splits by the frozen phase
    classification instead, so a name-shaped assumption carried across
    would look for sources that do not exist.
    """
    weights = load("feb").defaults["source_loss_weights"]
    assert len(weights) == 6
    assert set(weights.values()) == {15000.0}
    assert set(weights) == {f"{c}_{p}" for c in ("noneq", "stable")
                            for p in (0, 5, 10)}
    assert not any(name.startswith("slab") for name in weights)


def test_the_drift_window_and_the_loader_split_are_declared():
    """The knobs that decide WHICH windows exist and which runs validate.

    Nothing of this is in the checkpoint: it belongs to the data module, and
    a run rebuilt without it trains a different model on a different split
    and reports no disagreement anywhere.
    """
    d = load("feb").defaults
    assert d["estimator"] == "weak_mid"
    assert d["w"] == 100
    assert d["n_s"] == 5
    assert d["stride"] == 40
    assert d["burn_in_frames"] == 0
    assert d["run_weighting"] == "uniform"
    assert d["batch_size"] == 6
    assert d["num_workers"] == 4
    assert d["val_split"] == "random"
    assert d["val_frac"] == 0.1
    assert d["split_seed"] == 316
    assert d["source_pattern"] == "cube_*"
    assert d["grid"] == (32, 32, 32)


def test_the_drift_estimator_this_published_model_used_is_registered():
    """The gap this file used to record, now closed.

    ``weak_mid`` is the weak target with the model states at the frame
    midpoints ``0.5 * (rho[t+m] + rho[t+m-1])``; this system's own worker
    says the difference is NOT inert at its 0.1 ps frame spacing, where the
    plain states read a relaxing mode as growing.
    """
    from aipf.train.dataset import ESTIMATORS
    assert load("feb").defaults["estimator"] == "weak_mid"
    assert "weak_mid" in ESTIMATORS
    assert load("hhe").defaults["estimator"] in ESTIMATORS


def test_the_training_block_restates_the_config_spellings_it_translates():
    """Two spellings of one knob must never drift apart."""
    d = load("feb").defaults
    t = d["training"]
    assert (t["half_width"], t["n_states"], t["stride"]) == (
        d["w"], d["n_s"], d["stride"])
    assert t["band_k_max"] == d["k_max"]
    assert (t["val_split"], t["val_fraction"], t["split_seed"]) == (
        d["val_split"], d["val_frac"], d["split_seed"])
    assert (t["batch_size"], t["order"], t["run_weighting"],
            t["grad_clip"]) == (d["batch_size"], d["order"],
                                d["run_weighting"], d["grad_clip"])
    # The one deliberate difference, the same one the other system makes.
    assert (d["num_workers"], t["num_workers"]) == (4, 0)
    tables = t["tables"]
    names = tuple(v.rsplit("/", 1)[-1] for v in tables["m_table"].values())
    assert names == load("feb").constants["m_table_files"]
    names = tuple(v.rsplit("/", 1)[-1] for v in tables["s_table"].values())
    assert names == load("feb").constants["s_table_files"]
    assert tables["pressure_floor"] == d["p_ref_gpa"]
    assert tables["columns"]["phase"]["structure"] == d["anchor_gate"]
    assert tables["columns"]["mobility"] == d["m_column"]


@pytest.mark.env
def test_the_anchor_tables_rebuild_the_published_models_own_anchor_buffers():
    """All nineteen anchor tensors the published model saved, ``torch.equal``.

    The declaration's gate per role, row weights, pressure floor and
    channel-0 composition label are the four things the other system's
    tables never needed; a wrong one moves a row or a weight here.
    """
    torch = pytest.importorskip("torch")
    from aipf.train.anchors import AnchorTables

    declared_roots.raw_or_skip("feb")
    system = load("feb")
    state = _published_hparams(system)["state_dict"]
    try:
        t = AnchorTables.load(system, **system.defaults["training"]["tables"])
    except FileNotFoundError as exc:              # the measured tables live under the raw root
        pytest.skip(str(exc))
    pairs = {
        "anchor_rho": t.mobility_rho, "anchor_T": t.mobility_T,
        "anchor_M": t.mobility_target,
        "stat_rho": t.shell_rho, "stat_T": t.shell_T, "stat_k": t.shell_k,
        "stat_Sinv": t.shell_target, "stat_mask": t.shell_mask,
        "stat_w": t.shell_weight,
        "stat_Sinv_rootinv": t.shell_target_rootinv,
        "bulk_rho": t.bulk_rho, "bulk_T": t.bulk_T, "bulk_z": t.bulk_zvec,
        "bulk_rho_tot": t.bulk_rho_tot, "bulk_scc0": t.bulk_target,
        "bulk_w": t.bulk_weight,
        "press_rho": t.pressure_rho, "press_T": t.pressure_T,
        "press_tgt": t.pressure_target,
    }
    for key, rebuilt in pairs.items():
        assert torch.equal(state[key], rebuilt), key
    # every saved press_w is one: the trust overlay drops and scales nothing
    assert bool((state["press_w"] == 1.0).all())
    # the floor the published model divided by, in eV/A^3
    assert t.pressure_floor == 10.0 * (1.0 / 160.2176)


# --------------------------------------------------------------------------
# Against the checkpoint's own bytes
# --------------------------------------------------------------------------

#: Declared name to where it lives in the published model's saved
#: ``hyper_parameters``: ``"top"`` for the flat dict, ``"model"`` for the
#: nested ``model_kwargs``. The declared name is the saved key except where
#: this package has its own vocabulary, and those are on the next list.
_IN_CHECKPOINT = {
    "sigma": ("top", "sigma"),
    "k_max": ("top", "k_max"),
    "k_fit_stat": ("top", "k_fit_stat"),
    "alpha_loss": ("top", "alpha_loss"),
    "h_inv_eps": ("top", "h_inv_eps"),
    "drift_scale": ("top", "drift_scale"),
    "lr": ("top", "lr"),
    "weight_decay": ("top", "weight_decay"),
    "warmup_epochs": ("top", "warmup_epochs"),
    "anneal_epochs": ("top", "anneal_epochs"),
    "wd_ghat": ("top", "wd_ghat"),
    "lambda_M": ("top", "lambda_M"),
    "lambda_S": ("top", "lambda_S"),
    "lambda_bulk": ("top", "lambda_bulk"),
    "lambda_P": ("top", "lambda_P"),
    "lambda_conv": ("top", "lambda_conv"),
    "lambda_gamma": ("top", "lambda_gamma"),
    "lambda_wpsd": ("top", "lambda_wpsd"),
    "wpsd_kappa": ("top", "wpsd_kappa"),
    "wpsd_margin": ("top", "wpsd_margin"),
    "wpsd_k_max": ("top", "wpsd_k_max"),
    "wpsd_n_k": ("top", "wpsd_n_k"),
    "conv_penalty": ("top", "conv_penalty"),
    "conv_margin": ("top", "conv_margin"),
    "conv_samples": ("top", "conv_samples"),
    "gamma_pt_samples": ("top", "gamma_pt_samples"),
    "stat_metric": ("top", "stat_metric"),
    "p_ref_gpa": ("top", "p_ref_gpa"),
    "anchor_gate": ("top", "anchor_gate"),
    "anchor_row_weight": ("top", "anchor_row_weight"),
    "m_column": ("top", "m_column"),
    "source_loss_weights": ("top", "source_loss_weights"),
    "R_cut": ("model", "R_cut"),
    "fexc_T_ref": ("model", "fexc_T_ref"),
    "T_ref": ("model", "T_ref"),
    "tbasis_ortho": ("model", "tbasis_ortho"),
    "h_u": ("model", "h_u"),
    "h_g": ("model", "h_g"),
    "h_w": ("model", "h_w"),
    "h_m": ("model", "h_m"),
    "h_u_joint": ("model", "h_u_joint"),
    "n_hidden_joint": ("model", "n_hidden_joint"),
    "activation": ("model", "activation"),
    "rho_eps": ("model", "rho_eps"),
    "fexc_input_scale": ("model", "fexc_input_scale"),
    "m_input_scale": ("model", "m_input_scale"),
    "arrhenius_shared_Ea": ("model", "arrhenius_shared_Ea"),
    "m_form": ("model", "m_form"),
    "m_T_form": ("model", "m_T_form"),
    "m_init_from_table": ("model", "m_init_from_table"),
    "grid": ("model", "grid"),
}

#: Declared knobs the checkpoint does NOT carry, and why. Listed rather than
#: left out, so that "declared but unchecked" is a decision and not a gap.
_NOT_IN_CHECKPOINT = {
    "rung": "this package's vocabulary for the model form",
    "mode_fields": "how the archive's channels are formed, one per species: no run knob",
    "kernel_form": "this package's vocabulary",
    "f_form": "this package's name for f_exc_form; compared below",
    "rho_ref": "a list in the checkpoint, a tuple here; compared below",
    "k_cut": "baked into every modes.npz by the extraction, not a run knob",
    "max_epochs": "the trainer's stop, in the config and not in the hparams",
    "grad_clip": "a Trainer argument, popped before the module is built",
    "estimator": "a data-module knob; none of this block is saved",
    "w": "a data-module knob",
    "n_s": "a data-module knob",
    "stride": "a data-module knob",
    "burn_in_frames": "a data-module knob",
    "run_weighting": "a data-module knob, in this package's vocabulary",
    "batch_size": "a data-module knob",
    "num_workers": "a data-module knob",
    "order": "a data-module knob, and one no checkpoint of this run could "
             "carry: the training code's loader always shuffled "
             "(shuffle=True), so the declaration makes "
             "the published behaviour explicit rather than recording a "
             "choice that run was ever offered",
    "val_split": "a data-module knob",
    "val_frac": "a data-module knob",
    "split_seed": "a data-module knob",
    "source_pattern": "a data-module knob",
    "training": "aipf.train.fit's own vocabulary for the knobs above and the "
                "anchor tables; pinned in test_system_constants.py and "
                "compared with the config spellings below",
    "diagnose": "the diagnosis driver's knobs, read off the published evaluation "
                "modules; no checkpoint carries them",
    "md": "aipf md run's declarations: the potential the archive was generated with, by md5, and "
          "the declared decks; molecular dynamics, upstream of any training",
}

#: Saved keys this System does NOT declare under that name, and why. The
#: union of this and the values of ``_IN_CHECKPOINT`` is asserted to be
#: EXACTLY what the published model saved, in both halves of its hparams, so a key
#: nobody decided about fails rather than passing quietly.
_SETTLED_ELSEWHERE = {
    ("top", "model_kwargs"): "the nested half, accounted for in its own right",
    ("top", "model_type"): "rung, f_form and kernel_form say it in this "
                           "package's words",
    ("top", "eta_min"): "this package's own TrainConfig default; a second "
                        "copy would be a package default declared as physics",
    ("top", "mask_dome"): "off, and the other two-species system leaves the "
                          "same switch undeclared for the same reason",
    ("top", "anchor_T_min_K"): "empty, and declared on anchor_rules",
    ("top", "eos_pressures"): "declared in constants as eos_pressures_GPa",
    ("top", "eos_csvs"): "absolute paths; the trees and basenames are declared",
    ("top", "eos_csv"): "None; the multi-pressure list above is what ran",
    ("top", "eos_overlay"): "an absolute path to a one-key trust overlay that "
                            "drops no row of this run; see the declaration",
    ("top", "m_table"): "absolute paths; tree and basenames are declared",
    ("top", "s_table"): "absolute paths; tree and basenames are declared",
    ("top", "conv_domain"): "declared as trust_domain and trust_T_range",
    ("top", "grad_log_every"): "a logging cadence",
    ("top", "resid_k_bins"): "the validation residual's own binning, reported "
                             "and not trained against",
    ("model", "f_exc_form"): "declared as f_form in this package's vocabulary",
    ("model", "g_form"): "configured and never built under the joint form; "
                         "proved against the weights below",
    ("model", "enable_TlnT"): "false; the joint network carries the "
                              "temperature instead of a basis",
    ("model", "enable_T2"): "false, likewise",
    ("model", "tbasis_ortho_window"): "read only by the two temperature-basis "
                                      "heads, which this run does not build",
    ("model", "disable_u"): "off",
    ("model", "disable_g"): "off",
    ("model", "gauge_fix"): "off",
    ("model", "fixed_select"): "read only when m_form is 'fixed'",
    ("model", "m_table"): "an absolute path, and only the 0 GPa table: it is "
                          "the bias initialiser, not the L_M anchor set",
    ("model", "m_column"): "the same value as the top-level copy, compared "
                           "against it below",
    ("model", "rho_ref"): "a list here and a tuple in the declaration",
}


def test_every_declared_knob_is_accounted_for_against_the_checkpoint():
    """No declared knob is unclassified, in either direction."""
    declared = set(load("feb").defaults)
    assert declared == set(_IN_CHECKPOINT) | set(_NOT_IN_CHECKPOINT)


def test_every_saved_hyperparameter_is_accounted_for_too():
    """The other direction, which is where a silent gap actually lives.

    A knob the published model saved and nobody decided about is exactly the shape
    of "half-declared", and the only way to find one is to enumerate what
    the bytes carry rather than what the file says.
    """
    state = _published_hparams(load("feb"))
    hparams = state["hyper_parameters"]
    saved = {("top", key) for key in hparams}
    saved |= {("model", key) for key in hparams["model_kwargs"]}
    accounted = set(_IN_CHECKPOINT.values()) | set(_SETTLED_ELSEWHERE)
    assert saved == accounted


def test_the_declarations_agree_with_the_published_models_saved_hyperparameters():
    """The declaration, checked against the bytes the figures load.

    A config file on disk can be edited after a run. This dict cannot.
    """
    system = load("feb")
    hparams = _published_hparams(system)["hyper_parameters"]
    model = hparams["model_kwargs"]
    for name, (where, key) in sorted(_IN_CHECKPOINT.items()):
        saved = (hparams if where == "top" else model)[key]
        assert system.defaults[name] == saved, f"{name} -> {where}[{key}]"


def test_the_declarations_the_checkpoint_stores_differently_still_agree():
    """Three values the checkpoint carries in another shape or another word."""
    system = load("feb")
    hparams = _published_hparams(system)["hyper_parameters"]
    model = hparams["model_kwargs"]
    assert tuple(model["rho_ref"]) == system.defaults["rho_ref"]
    assert model["f_exc_form"] == "joint"
    assert system.defaults["f_form"] == "joint_mlp"
    assert model["m_column"] == hparams["m_column"]


def test_the_constants_agree_with_the_checkpoint_too():
    """The pressures, the empty cut table, the trust domain and the tables."""
    system = load("feb")
    hparams = _published_hparams(system)["hyper_parameters"]
    assert tuple(hparams["eos_pressures"]) == system.constants[
        "eos_pressures_GPa"]
    assert dict(hparams["anchor_T_min_K"]) == dict(
        system.anchor_rules.T_min_by_pressure)
    assert tuple(hparams["conv_domain"]) == (
        system.constants["trust_domain"] + system.constants["trust_T_range"])
    for key, saved in (("m_table_files", "m_table"),
                       ("s_table_files", "s_table")):
        names = tuple(p.rsplit("/", 1)[-1] for p in hparams[saved])
        assert names == system.constants[key], key
        trees = {p.rsplit("/", 2)[-2] for p in hparams[saved]}
        assert trees == {"training_derived"}, saved
    assert tuple(p.rsplit("/", 2)[-2] for p in hparams["eos_csvs"]) == \
        system.constants["eos_trees"]
    assert tuple(p.rsplit("/", 1)[-1] for p in hparams["eos_csvs"]) == \
        system.constants["eos_csv_files"]


def test_the_temperature_basis_is_absent_and_the_flags_say_so():
    """`T_basis` is deliberately undeclared here, and this is the evidence."""
    system = load("feb")
    model = _published_hparams(system)["hyper_parameters"]["model_kwargs"]
    assert model["enable_TlnT"] is False
    assert model["enable_T2"] is False
    assert "T_basis" not in system.defaults
    assert "T_basis" in load("hhe").defaults


def test_the_convex_head_is_configured_and_is_never_built():
    """`g_form: icnn` is INERT in this published model, and the weights prove it.

    The config carries it with its own in-file comment "unused under joint",
    the joint form excludes the split heads, and the saved ``state_dict``
    holds no tensor of either. What this published model's excess free energy
    actually is, is one joint network of three inputs -- the two densities
    and the temperature ratio -- with no bias on its last layer.
    """
    state = _published_hparams(load("feb"))
    model = state["hyper_parameters"]["model_kwargs"]
    keys = set(state["state_dict"])
    assert model["g_form"] == "icnn"
    assert model["f_exc_form"] == "joint"
    assert not [k for k in keys if k.startswith("model.f_local.g_net")]
    assert not [k for k in keys if k.startswith("model.f_local.u_net")]
    joint = sorted(k for k in keys if k.startswith("model.f_local.u_joint_net"))
    assert joint, "no joint head in the weights either"
    first = state["state_dict"]["model.f_local.u_joint_net.net.0.weight"]
    assert tuple(first.shape) == (model["h_u_joint"], 3)
    hidden = [k for k in joint if k.endswith(".weight")]
    assert len(hidden) == model["n_hidden_joint"] + 1
    assert "model.f_local.u_joint_net.net.8.bias" not in keys


def test_the_declared_joint_head_has_this_published_models_depth():
    """The gap this file used to record, closed: nine tensors, as saved."""
    torch = pytest.importorskip("torch")
    from aipf.functional.build import build
    head = build(load("feb")).f_local.u_joint_net
    state = _published_hparams(load("feb"))["state_dict"]
    saved = {k[len("model.f_local.u_joint_net."):]: v.shape
             for k, v in state.items()
             if k.startswith("model.f_local.u_joint_net.")}
    assert {k: v.shape for k, v in head.state_dict().items()} == saved
    assert len(saved) == 9


def test_no_convex_machinery_carries_a_weight_in_this_run():
    """The convexity penalty is switched off, so nothing hinges on it here."""
    d = load("feb").defaults
    assert d["lambda_conv"] == 0.0
    assert d["lambda_gamma"] == 0.0


# --------------------------------------------------------------------------
# What is not this system's
# --------------------------------------------------------------------------


def test_no_engine_knob_was_carried_over_from_the_other_system():
    """Two of the four engine values differ, and two genuinely agree.

    The neighbour rebuild rule is "every 10" here and "every 1" there, and
    the barostat damping is 1.0 against 0.1. The unit system and the
    neighbour skin are the same number in both, measured, and pinning them
    as equal is as much a claim as pinning the others as different.
    """
    feb, hhe = load("feb"), load("hhe")
    mine, theirs = feb.md("production"), hhe.md("production")
    assert mine.neigh_modify != theirs.neigh_modify
    assert mine.neigh_modify == "every 10 delay 0 check yes"
    assert mine.P_damp != theirs.P_damp
    assert mine.T_damp != theirs.T_damp
    assert mine.units == theirs.units
    assert mine.skin == theirs.skin == 2.0


def test_no_published_knob_was_carried_over_from_the_other_system():
    """The fork makes carry-over easy and it has happened twice already."""
    feb, hhe = load("feb"), load("hhe")
    for name in ("fexc_T_ref", "T_ref", "rho_ref", "lambda_S",
                 "anneal_epochs", "max_epochs", "estimator"):
        assert feb.defaults[name] != hhe.defaults[name], name
    assert feb.checkpoint.md5 != hhe.checkpoint.md5
    # And the ones that MUST agree, because they are the same
    # measurement or this package's own vocabulary.
    for name in ("sigma", "k_cut", "k_fit_stat", "R_cut", "rung",
                 "kernel_form", "m_form", "alpha_loss", "lr",
                 "weight_decay", "warmup_epochs", "wd_ghat", "stat_metric",
                 "h_inv_eps", "lambda_M", "lambda_bulk", "lambda_P"):
        assert feb.defaults[name] == hhe.defaults[name], name


def test_the_third_system_declares_none_of_this():
    """The single-species system declares no engine campaign; its published model is its own."""
    lj = load("lj")
    assert lj.md_settings == {}
    assert lj.anchor_rules.shape_T_min_by_pressure == {}
    assert lj.checkpoint.md5 != load("feb").checkpoint.md5
    with pytest.raises(KeyError):
        lj.md("production")


@pytest.mark.env
def test_no_slab_preparation_is_declared_from_a_deck_nobody_ran():
    """The correction of b22ac35, pinned so it cannot come back.

    This tree's slab worker is byte-identical to the other system's (md5
    7a78839baaa15ca87c43453a7e0eb76e on both) and its
    ramp offset is right there in the source. The DATA says otherwise: this
    system's slab directory under the raw root is empty and no slab run appears in
    any of its 93 manifests. A preparation declared from a deck nobody ran is
    the mistake the template rule exists to prevent.
    """
    feb = load("feb")
    assert feb.preparation.melt_T == 2600.0
    assert feb.preparation.ramp_offset is None
    root = declared_roots.raw_or_skip("feb")
    slab = root / "slab_data"
    if slab.is_dir():
        assert not any(slab.iterdir()), (
            f"{slab} is no longer empty; the preparation rule above was "
            f"decided on it being so")
