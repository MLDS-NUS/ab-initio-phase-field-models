"""Everything the published model needs, declared, and proved against its own bytes.

Two kinds of test live here and the split is deliberate.

The DECLARATION tests pin what ``experiments/hhe/system.py`` now says: the
kernel-shape temperature floor, the engine settings per campaign, the batch
shape, the field trees, and the published model's own model and training knobs. They
are change detectors in the sense ``test_system_constants.py`` already uses --
they are not an independent source for any value, because the provenance
lives beside each constant with the file and line that settles it.

The AGREEMENT test is not a change detector. It opens the pinned checkpoint
and compares every declared knob with the ``hyper_parameters`` dict saved
inside it, which is the strongest evidence there is: a config file on disk can
be edited after a run and a saved hparams dict cannot. It also accounts for
every declared knob that is NOT in that dict, with a reason, so that
"declared but unchecked" is a list somebody wrote rather than a gap.

The REFUSAL tests exercise the core declarations -- ``MdSettings`` and the
second temperature table on ``AnchorRules`` -- with
neutral systems. They sit here rather than in ``tests/unit/test_system.py``
because the fields they exercise exist for this system.

The tests that read the H/He raw root are marked ``env`` and skip, naming it, when it is absent.
"""

import pytest

from aipf.paths import Paths
from aipf.system import AnchorRules, MdSettings, System, load

import declared_roots

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _system(**overrides) -> System:
    """A minimal, species-neutral system, one field replaced."""
    kwargs = dict(
        name="demo", n_species=1, species=("one",), masses={}, atom_types={},
        table_keys={}, paths=Paths(system="demo"),
        anchor_rules=AnchorRules({}), constants={}, defaults={},
    )
    kwargs.update(overrides)
    return System(**kwargs)


def _md(**overrides) -> MdSettings:
    """A valid engine declaration, one field replaced."""
    kwargs = dict(units="metal", skin=2.0,
                  neigh_modify="every 1 delay 0 check yes",
                  thermo_every=200, T_damp=0.01, P_damp=0.1,
                  pressure_scale=1e4)
    kwargs.update(overrides)
    return MdSettings(**kwargs)


# --------------------------------------------------------------------------
# The kernel-shape temperature floor
# --------------------------------------------------------------------------


def test_the_kernel_shape_temperature_floor_is_declared_per_pressure():
    """Recorded only as an inline comment on the published pipeline's command line.

    Without it `kappa` and `M_kappa0` cannot be rebuilt: `universal_kappa`
    averages the runs at or above this temperature and every other run's
    amplitude is fitted against that average.
    """
    hhe = load("hhe")
    assert hhe.anchor_rules.shape_T_min_by_pressure == {
        200: 7500, 400: 9500, 600: 9500, 800: 10000}


def test_the_two_temperature_cuts_are_two_different_tables():
    """The one way this declaration can be got wrong is by merging it.

    Both are per-pressure temperature floors in kelvin, four pressures each,
    and they are not the same numbers: the anchor cut is the measured
    critical temperature and the shape floor is the top anchor temperature of
    each dataset, which is higher at every pressure.
    """
    hhe = load("hhe")
    anchor = hhe.anchor_rules.T_min_by_pressure
    shape = hhe.anchor_rules.shape_T_min_by_pressure
    assert set(anchor) == set(shape) == {200, 400, 600, 800}
    for pressure in sorted(anchor):
        assert shape[pressure] > anchor[pressure], pressure


def test_the_shape_floor_is_looked_up_the_way_the_anchor_cut_is():
    """Whole-number keys, a caller holding a float read out of a file."""
    hhe = load("hhe")
    assert hhe.shape_t_min(800) == 10000.0
    assert hhe.shape_t_min(800.0) == 10000.0
    assert isinstance(hhe.shape_t_min(200), float)
    assert hhe.shape_t_min(1000) is None


def test_a_system_that_declares_no_shape_floor_answers_none():
    assert _system().shape_t_min(800) is None


def test_the_shape_floor_is_checked_for_being_numeric_like_the_anchor_cut():
    """Mutation: checking only the first table.

    Unchecked, a non-numeric entry raises "could not convert string to float"
    from inside the lookup, naming neither the system nor the field.
    """
    with pytest.raises(ValueError, match="shape_T_min_by_pressure"):
        _system(anchor_rules=AnchorRules({}, shape_T_min_by_pressure={
            "hot": 10000}))
    with pytest.raises(ValueError, match="shape_T_min_by_pressure"):
        _system(anchor_rules=AnchorRules({}, shape_T_min_by_pressure={
            800: "hot"}))


def test_a_shape_table_that_has_been_through_json_still_works():
    """String keys and string values are numeric spellings, as for the cut."""
    s = _system(anchor_rules=AnchorRules(
        {}, shape_T_min_by_pressure={"800": "10000"}))
    assert s.shape_t_min(800) == 10000.0


# --------------------------------------------------------------------------
# The engine knobs
# --------------------------------------------------------------------------


def test_both_campaigns_the_published_model_reads_declare_their_engine_settings():
    """The published model's own hparams name two data families, so two campaigns.

    `s_table`/`m_table` come from the trajectory campaign and `eos_csvs` from
    the equation-of-state grid, and those two were run with different damping
    times. A single pair of numbers would be false for one of them.
    """
    hhe = load("hhe")
    assert set(hhe.md_settings) == {"production", "eos"}


def test_the_production_engine_settings_are_the_published_trajectories():
    """The worker that ran all 491 of them."""
    md = load("hhe").md("production")
    assert md.units == "metal"
    assert md.skin == 2.0
    assert md.neigh_modify == "every 1 delay 0 check yes"
    assert md.thermo_every == 200
    assert md.T_damp == 0.01
    assert md.P_damp == 0.1


def test_the_equation_of_state_grid_ran_at_different_damping_times():
    """Measured, not assumed: 0.02/0.2 and a faster thermo cadence.

    If this ever reads the same as the production pair, somebody has tidied
    away a difference the published pipeline recorded.
    """
    md = load("hhe").md("eos")
    assert md.T_damp == 0.02
    assert md.P_damp == 0.2
    assert md.thermo_every == 100


def test_the_campaigns_agree_on_everything_that_does_not_vary():
    """Three knobs are the system's and one is the campaign's.

    Unit system, neighbour skin and neighbour rebuild rule are identical in
    every hydrogen-helium worker and in all 1391 archived logs. The damping
    times and the thermo cadence are not.
    """
    hhe = load("hhe")
    campaigns = [hhe.md(name) for name in hhe.md_settings]
    assert len({m.units for m in campaigns}) == 1
    assert len({m.skin for m in campaigns}) == 1
    assert len({m.neigh_modify for m in campaigns}) == 1
    assert len({m.pressure_scale for m in campaigns}) == 1
    assert len({m.T_damp for m in campaigns}) == 2
    assert len({m.P_damp for m in campaigns}) == 2


def test_an_unknown_campaign_is_refused_and_the_known_ones_are_named():
    hhe = load("hhe")
    with pytest.raises(KeyError) as exc:
        hhe.md("slab")
    assert "production" in str(exc.value) and "eos" in str(exc.value)


def test_a_system_declaring_no_engine_settings_is_refused_not_guessed():
    with pytest.raises(KeyError, match="declares no engine settings"):
        _system().md("production")


# --------------------------------------------------------------------------
# The pressure unit, which nothing declared
# --------------------------------------------------------------------------


def test_the_deck_pressure_is_bar_and_the_factor_is_ten_thousand():
    """The gap that runs a barostat at 800 bar instead of 800 GPa.

    A request carries GPa and the deck's `fix npt` wants bar. Both archived
    heavy trees converted it in the launcher, so nothing in the package
    derives it and nothing raised when it was absent.
    """
    md = load("hhe").md("production")
    assert md.pressure_scale == 1e4
    assert md.deck_pressure(800.0) == 8_000_000.0
    # A second pressure, so the factor cannot be a coincidence of 800.
    assert md.deck_pressure(200.0) == 2_000_000.0


def test_every_campaign_converts_the_pressure_the_same_way():
    hhe = load("hhe")
    for name in hhe.md_settings:
        assert hhe.md(name).deck_pressure(800.0) == 8_000_000.0


def test_the_deck_pressure_of_a_pressure_that_is_not_a_number_is_refused():
    with pytest.raises(TypeError, match="pressure"):
        _md().deck_pressure("800")


# --------------------------------------------------------------------------
# The batch shape is a site fact
# --------------------------------------------------------------------------


def test_a_system_declares_no_batch_shape():
    """Where a job is submitted is a site fact (``aipf.site``, ``pbs_*``), so a System has no such field."""
    assert not hasattr(load("hhe"), "batch")


# --------------------------------------------------------------------------
# The engine declaration's own refusals
# --------------------------------------------------------------------------


def test_an_engine_declaration_with_no_unit_system_is_refused():
    with pytest.raises(ValueError, match="units"):
        _md(units="")


@pytest.mark.parametrize("field", ["skin", "T_damp", "P_damp",
                                   "pressure_scale"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_non_positive_engine_quantity_is_refused_by_name(field, value):
    with pytest.raises(ValueError, match=field):
        _md(**{field: value})


@pytest.mark.parametrize("value", [0, -1])
def test_a_thermo_cadence_below_one_step_is_refused(value):
    """Zero is not "never": the engine refuses it and the run dies at setup."""
    with pytest.raises(ValueError, match="thermo_every"):
        _md(thermo_every=value)


def test_an_engine_declaration_with_no_neighbour_rule_is_refused():
    with pytest.raises(ValueError, match="neigh_modify"):
        _md(neigh_modify="")


@pytest.mark.parametrize("field", ["units", "neigh_modify"])
def test_an_engine_string_of_pure_whitespace_is_refused(field):
    """A blank is not a declaration, and it reaches the deck as one."""
    with pytest.raises(ValueError, match=field):
        _md(**{field: "   "})


@pytest.mark.parametrize("field", ["skin", "T_damp", "P_damp",
                                   "pressure_scale", "thermo_every"])
def test_a_boolean_is_not_a_number_here(field):
    """``bool`` is a subclass of ``int``, so True is a damping time of 1.

    Every one of these is written into a deck as a number, and True renders
    as ``True``, which the engine reads as neither.
    """
    with pytest.raises(ValueError, match=field):
        _md(**{field: True})


def test_a_fractional_thermo_cadence_is_refused():
    """A cadence is a step count. 200.0 renders into the deck as `200.0`."""
    with pytest.raises(ValueError, match="thermo_every"):
        _md(thermo_every=200.0)


def test_a_boolean_pressure_is_not_converted():
    with pytest.raises(TypeError, match="pressure"):
        _md().deck_pressure(True)


# --------------------------------------------------------------------------
# The field trees, which a core function was already reading
# --------------------------------------------------------------------------


def test_the_field_trees_the_published_model_trains_on_are_declared():
    """`index_fields` marks every tree no declared role names as archived.

    Before this declaration that was every tree this system has, including
    the four the published model reads.
    """
    hhe = load("hhe")
    assert hhe.constants["modes_trees"] == (
        "modes_800GPa", "modes_200GPa", "modes_400GPa", "modes_600GPa")
    assert hhe.constants["table_trees"] == (
        "fdt_800GPa", "fdt_200GPa", "fdt_400GPa", "fdt_600GPa")
    assert hhe.constants["eos_trees"] == (
        "eos_800GPa", "eos_200GPa", "eos_400GPa", "eos_600GPa")


def test_the_trees_are_declared_in_the_order_the_published_model_lists_them():
    """800 first, then 200, 400, 600. Not ascending, and not a typo."""
    hhe = load("hhe")
    pressures = hhe.constants["eos_pressures_GPa"]
    assert pressures == (800.0, 200.0, 400.0, 600.0)
    assert pressures != tuple(sorted(pressures))
    for key in ("modes_trees", "table_trees", "eos_trees"):
        names = hhe.constants[key]
        assert len(names) == len(pressures), key
        assert all(f"{int(p)}GPa" in name
                   for p, name in zip(pressures, names)), key


def test_this_system_does_not_split_its_runs_from_its_tables():
    """The other two-species system declares three roles; this one has two.

    Its `fdt_<P>GPa` trees hold the per-run directories AND the aggregated
    tables together, so a `fdt_runs_trees` entry would name the same tree
    twice under two roles and the index would report the later one.
    """
    hhe = load("hhe")
    assert "fdt_runs_trees" not in hhe.constants


@pytest.mark.env
def test_the_index_now_calls_the_published_models_trees_authoritative():
    """The declaration's whole point, measured on the raw root's own trees."""
    from aipf.data import index
    declared_roots.raw_or_skip("hhe")
    hhe = load("hhe")
    roles = {t["name"]: t["role"] for t in index.index_fields(hhe)["trees"]}
    for name in hhe.constants["modes_trees"]:
        assert roles.get(name) == "modes", name
    for name in hhe.constants["table_trees"]:
        assert roles.get(name) == "tables", name


# --------------------------------------------------------------------------
# The published model's own knobs
# --------------------------------------------------------------------------


def test_the_model_construction_knobs_are_complete():
    """The four head widths, the reference densities and the basis flag.

    Half-declared was the worst state: every other `model_kwargs` entry the
    inventory names was already here.
    """
    d = load("hhe").defaults
    assert (d["h_u"], d["h_g"], d["h_w"], d["h_m"]) == (16, 32, 16, 16)
    assert d["rho_ref"] == (0.35, 0.33)
    assert d["tbasis_ortho"] is True


def test_the_two_reference_temperatures_are_both_declared():
    """`fexc_T_ref` and `T_ref` are different knobs on the same model.

    The second is inert here because the mobility carries no temperature
    factor, and it is declared anyway: the confusion between the two names
    has cost this project time before, and a reader who sees only one of
    them has no way to know which one they are looking at.
    """
    d = load("hhe").defaults
    assert d["fexc_T_ref"] == 8000.0
    assert d["T_ref"] == 10000.0
    assert d["m_T_form"] == "none"


def test_the_six_loss_weights_are_declared():
    d = load("hhe").defaults
    assert d["lambda_M"] == 0.5
    assert d["lambda_S"] == 0.01
    assert d["lambda_bulk"] == 0.01
    assert d["lambda_P"] == 5.0
    assert d["lambda_conv"] == 1.0
    assert d["lambda_gamma"] == 1.0


def test_the_schedule_and_the_two_decays_are_declared():
    """`wd_ghat` is the kernel head's own decay and is not `weight_decay`.

    The inventory lists the first and not the second, which is exactly how
    one gets read as the other.
    """
    d = load("hhe").defaults
    assert d["lr"] == 5e-4
    assert d["warmup_epochs"] == 5
    assert d["anneal_epochs"] == 30
    assert d["max_epochs"] == 30
    assert d["wd_ghat"] == 0.01
    assert d["weight_decay"] == 0.0


def test_the_penalty_shapes_are_declared():
    d = load("hhe").defaults
    assert d["conv_penalty"] == "hinge"
    assert d["conv_margin"] == 0.05
    assert d["conv_samples"] == 1024
    assert d["gamma_pt_samples"] == 16


def test_the_static_anchor_metric_the_config_never_sets_is_declared():
    """`stat_metric` is in the checkpoint and not in the YAML.

    It is the relative Frobenius the L_S / L_bulk argument names, so a
    reader who only has the config cannot tell which metric ran.
    """
    assert load("hhe").defaults["stat_metric"] == "rel_frob"


def test_the_regulariser_the_inventory_omits_is_declared():
    """The inventory writes `r2/k2`; the code writes `r2 / (k2 + eps)`.

    That was recorded as a finding. The value is declared here so the
    form can be rebuilt without rediscovering it.
    """
    assert load("hhe").defaults["h_inv_eps"] == 1e-6


def test_every_source_carries_the_same_weight():
    """Eight sources: two geometries at each of four pressures, all at 1.0."""
    weights = load("hhe").defaults["source_loss_weights"]
    assert len(weights) == 8
    assert set(weights.values()) == {1.0}
    assert set(weights) == {f"{g}_{int(p)}"
                            for g in ("slab", "cube")
                            for p in (800, 200, 400, 600)}


def test_the_preparation_offset_is_pinned():
    """Found by the mutation sweep, not by reading the assertions.

    This declaration predates this task and nothing reached it: changing
    2000.0 to 2001.0 left the whole suite green. It decides how far above
    its target a ramping run of this system starts, so a silent change to
    it changes every trajectory the ramping stream produces --
    `test_the_ramp_the_offset_describes_is_the_one_the_archive_ran` below
    says which stream that is, against the archive rather than against the
    worker's source.
    """
    preparation = load("hhe").preparation
    assert preparation.ramp_offset == 2000.0
    # The alternative form, and declaring both is refused at construction:
    # this system has no single temperature that melts it.
    assert preparation.melt_T is None


@pytest.mark.env
def test_the_ramp_the_offset_describes_is_the_one_the_archive_ran():
    """The offset is cited from the DATA, not from a branch of the worker.

    The worker is shared by both of this system's streams and ramps only
    under the isotropic barostat (`RAMP = ENSEMBLE == "npt_iso"`). Reading that file alone, and reading
    it under its own name, is how the first version of this declaration
    came to cite the slab stream for a ramp the slab stream never used --
    the same mistake `71b0297` fixed for the other system by asking the
    archive instead.

    So this asks the archive. Every manifest entry on scratch carries the
    ensemble its run was launched with, and the two streams split cleanly:
    the cube entries all ramp and the slab entries all do not. A future
    edit that moves this system to one barostat, or that re-points the
    declaration at the stream with no ramp, fails here.
    """
    import collections
    import json

    root = declared_roots.raw_or_skip("hhe", "manifests")
    hhe = load("hhe")
    seen = collections.Counter()
    for path in sorted(root.glob("*.json")):
        entries = json.loads(path.read_text())
        if not isinstance(entries, list):
            entries = entries.get("runs", [])
        for entry in entries:
            if isinstance(entry, dict) and "run" in entry:
                seen[(entry.get("kind"),
                      entry["run"].get("ensemble"))] += 1
    assert seen, f"no manifest entries under {root}"
    # `npt_iso` is the ramping branch; every other ensemble runs at target.
    ramping = {kind for (kind, ens) in seen if ens == "npt_iso"}
    flat = {kind for (kind, ens) in seen if ens != "npt_iso"}
    assert ramping == {"cube"}, (
        f"the ramp belongs to the cube stream and to no other; found "
        f"{sorted(ramping)} in {dict(seen)}")
    assert flat == {"slab"}, (
        f"only the slab stream runs without the ramp; found "
        f"{sorted(flat)} in {dict(seen)}")
    # ... and the declaration answers to that: an offset is declared if and
    # only if the archive holds runs that ramp. A ramp declared from a deck
    # nobody ran is the mistake the template rule exists to prevent, and a
    # ramp dropped while the runs still carry it is the same mistake
    # backwards.
    assert (hhe.preparation.ramp_offset is not None) == bool(ramping), (
        f"preparation.ramp_offset is {hhe.preparation.ramp_offset!r} but "
        f"the archive's ramping streams are {sorted(ramping)}")


def test_the_estimator_the_published_model_selected_is_declared():
    """The config's selector, not the function's name.

    `estimator: weak` chooses `weak_target`, which is that module's default;
    the two strings are the same choice seen from the two sides.
    """
    assert load("hhe").defaults["estimator"] == "weak"


# --------------------------------------------------------------------------
# Against the checkpoint's own bytes
# --------------------------------------------------------------------------

#: Declared name to where it lives in the checkpoint's saved
#: ``hyper_parameters``: ``"top"`` for the flat dict, ``"model"`` for the
#: nested ``model_kwargs``. The declared name is the saved key except where
#: this package has its own vocabulary, which is the third column.
_IN_CHECKPOINT = {
    "sigma": ("top", "sigma"),
    "k_fit_stat": ("top", "k_fit_stat"),
    "alpha_loss": ("top", "alpha_loss"),
    "h_inv_eps": ("top", "h_inv_eps"),
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
    "conv_penalty": ("top", "conv_penalty"),
    "conv_margin": ("top", "conv_margin"),
    "conv_samples": ("top", "conv_samples"),
    "gamma_pt_samples": ("top", "gamma_pt_samples"),
    "stat_metric": ("top", "stat_metric"),
    "R_cut": ("model", "R_cut"),
    "fexc_T_ref": ("model", "fexc_T_ref"),
    "T_ref": ("model", "T_ref"),
    "tbasis_ortho": ("model", "tbasis_ortho"),
    "h_u": ("model", "h_u"),
    "h_g": ("model", "h_g"),
    "h_w": ("model", "h_w"),
    "h_m": ("model", "h_m"),
    "m_form": ("model", "m_form"),
    "m_T_form": ("model", "m_T_form"),
    # This package's own name for the excess head's form.
    "f_form": ("model", "g_form"),
}

#: Declared knobs the checkpoint does NOT carry, and why. Listed rather than
#: left out, so that "declared but unchecked" is a decision and not a gap.
_NOT_IN_CHECKPOINT = {
    "k_cut": "baked into every modes.npz by the extraction, not a run knob",
    "mode_fields": "how the archive's channels are formed, one per species: no run knob",
    "k_max": "declared null, resolved at construction to 4 / sigma",
    "rung": "this package's vocabulary for the model form",
    "kernel_form": "this package's vocabulary",
    "T_basis": "the consequence of enable_TlnT and enable_T2, checked below",
    "max_epochs": "the trainer's stop, in the config and not in the hparams",
    "estimator": "a dataset knob, in the config and not in the hparams",
    "rho_ref": "a list in the checkpoint, a tuple here; compared below",
    "source_loss_weights": "a mapping; compared below",
    "diagnose": "the diagnosis driver's declared knobs: "
                "a command line of the eval battery, run AFTER training, so "
                "no training checkpoint carries any of it",
    "training": "the training driver's declared knobs: "
                "the window, run-weighting, split and loader shapes, which "
                "the published model's CONFIG carries and its hyper_parameters do "
                "not -- its LightningModule never saw them, the "
                "DataModule did. Their provenance is the config file, cited "
                "line by line where they are declared",
    "rollout": "the rollout drivers' declared knobs: solver, "
               "archive and grids of the published spinodal/slab drivers, run "
               "after training; cited line by line where declared",
    "column": "the rain-column driver's declared knobs: masses, "
              "T window, diagnostics and solver of the published column "
              "driver, run after training; cited line by line where declared",
    "md": "aipf md run's declarations: the potential the archive was generated with, by md5, and "
          "the declared decks; molecular dynamics, upstream of any training",
}


def _published_hparams(system):
    """The published model's saved hyper_parameters, from the tracked copy the repository carries."""
    declared_roots.published_or_skip("hhe")
    torch = pytest.importorskip("torch")
    path = system.resolve_checkpoint()
    return torch.load(path, map_location="cpu", weights_only=False)[
        "hyper_parameters"]


def test_every_declared_knob_is_accounted_for_against_the_checkpoint():
    """No declared knob is unclassified, in either direction.

    A knob added to `defaults` without a decision about where it is settled
    fails here, which is the point: the two lists above are the decision.
    """
    declared = set(load("hhe").defaults)
    assert declared == set(_IN_CHECKPOINT) | set(_NOT_IN_CHECKPOINT)


def test_the_declarations_agree_with_the_published_models_saved_hyperparameters():
    """The declaration, checked against the bytes the figures load.

    A config file on disk can be edited after a run. This dict cannot.
    """
    system = load("hhe")
    hparams = _published_hparams(system)
    model = hparams["model_kwargs"]
    for name, (where, key) in sorted(_IN_CHECKPOINT.items()):
        saved = (hparams if where == "top" else model)[key]
        assert system.defaults[name] == saved, f"{name} -> {where}[{key}]"


def test_the_declarations_the_checkpoint_stores_differently_still_agree():
    """Two values the checkpoint carries in another shape.

    A list against a tuple and a mapping whose identity is its contents.
    """
    system = load("hhe")
    hparams = _published_hparams(system)
    assert (tuple(hparams["model_kwargs"]["rho_ref"])
            == system.defaults["rho_ref"])
    assert dict(hparams["source_loss_weights"]) == dict(
        system.defaults["source_loss_weights"])


def test_the_temperature_basis_is_the_flags_the_checkpoint_carries():
    """`T_basis` is this package's name for a pair of flags."""
    system = load("hhe")
    model = _published_hparams(system)["model_kwargs"]
    assert model["enable_TlnT"] is True
    assert model["enable_T2"] is False
    assert system.defaults["T_basis"] == ("1", "T", "TlnT")


def test_the_drift_band_resolves_to_what_the_checkpoint_stored():
    """The config writes null; the checkpoint stores the resolved number."""
    system = load("hhe")
    hparams = _published_hparams(system)
    assert system.defaults["k_max"] is None
    assert hparams["k_max"] == 4.0 / system.defaults["sigma"]


def test_the_constants_agree_with_the_checkpoint_too():
    """The pressures, the cut table and the trees the published model reads."""
    system = load("hhe")
    hparams = _published_hparams(system)
    assert tuple(hparams["eos_pressures"]) == system.constants[
        "eos_pressures_GPa"]
    assert dict(hparams["anchor_T_min_K"]) == dict(
        system.anchor_rules.T_min_by_pressure)
    assert tuple(p.split("/")[-2] for p in hparams["s_table"]) == system.\
        constants["table_trees"]
    assert tuple(p.split("/")[-2] for p in hparams["eos_csvs"]) == system.\
        constants["eos_trees"]


# --------------------------------------------------------------------------
# What is not this system's
# --------------------------------------------------------------------------


def test_no_engine_knob_was_carried_over_from_another_system():
    """Two of the pairs the brief describes are cross-system, not internal.

    A neighbour skin of 0.5 belongs to the reduced-unit tree and a barostat
    damping of 1.0 to the other two-species tree. Declaring either here
    would be silently wrong, and neither could be caught by reading this
    system's own files.
    """
    hhe = load("hhe")
    for name in hhe.md_settings:
        assert hhe.md(name).skin != 0.5
        assert hhe.md(name).P_damp != 1.0


def test_the_single_species_system_declares_none_of_this():
    """The third system declares no engine campaign, and its absence is a loud one.

    Engine settings, batch shape and shape floor are undeclared there, and
    every accessor refuses rather than answering with this system's numbers.
    Its published model is its own, never this system's.

    The OTHER two-species system declares the same gaps closed, against its
    own measurements. What
    that leaves worth asserting is not that it declares nothing, but that
    nothing it declares is a copy of this system's -- see below.
    """
    other = load("lj")
    assert other.md_settings == {}
    assert other.checkpoint.md5 != load("hhe").checkpoint.md5
    assert other.anchor_rules.shape_T_min_by_pressure == {}
    with pytest.raises(KeyError):
        other.md("production")


def test_nothing_the_other_two_species_system_declares_is_a_copy_of_this():
    """The fork makes carry-over easy and it has happened twice.

    Every field below is one both systems now declare, measured separately
    from each tree, and every one of them differs. The neighbour skin and
    the unit system are deliberately NOT on this list: those genuinely agree
    and are pinned as equal in that system's own file.
    """
    feb = load("feb")
    assert feb.checkpoint.md5 != load("hhe").checkpoint.md5
    assert feb.md("production").neigh_modify != load("hhe").md(
        "production").neigh_modify
    assert feb.md("production").P_damp != load("hhe").md("production").P_damp
    assert set(feb.anchor_rules.shape_T_min_by_pressure).isdisjoint(
        load("hhe").anchor_rules.shape_T_min_by_pressure)


# --------------------------------------------------------------------------
# The model declaration, against the published model's own bytes (spec 3.1)
# --------------------------------------------------------------------------


#: The bridge's own kB, the value the published model's training code
#: uses. It scales ``kBT_ref`` and the T-basis window and changes no name or
#: flag, so nothing this test asserts depends on it -- it is passed because
#: the bridge requires it, not because a declared value is measured in it.
_K_B = 8.617333262e-5


