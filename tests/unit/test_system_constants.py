from pathlib import Path

import math

import pytest

from aipf.system import Noise, load


def test_hhe():
    s = load("hhe")
    assert s.n_species == 2
    assert s.species == ("H", "He")
    assert s.channel_of("H") == 0
    assert s.atom_types == {"H": 1, "He": 2}
    assert s.anchor_rules.T_min_by_pressure == {200: 6100, 400: 7800,
                                                600: 8800, 800: 8973}
    assert s.constants["P_GPa"] == 800.0
    assert s.constants["binodal_7000"] == (0.069, 0.925)
    assert s.constants["trust_domain"] == (0.424, 0.252, 1.036, 0.628)
    assert s.defaults["m_form"] == "mlp_scaled"
    assert s.defaults["m_T_form"] == "none"
    # k_cut and k_max are different knobs with different values. k_cut = 3.0 is
    # read from modes_800GPa/*/modes.npz; k_max is null in the published model's config
    # and resolves to 4 / sigma = 2.0, which is the inventory's reported value.
    assert s.defaults["sigma"] == 2.0
    assert s.defaults["k_cut"] == 3.0
    assert s.defaults["k_max"] is None
    assert 4.0 / s.defaults["sigma"] == 2.0


def test_feb():
    s = load("feb")
    assert s.n_species == 2
    assert s.species == ("B", "Fe")
    assert s.channel_of("B") == 0
    assert s.atom_types == {"B": 1, "Fe": 2}
    assert s.constants["P_GPa"] == 0.0
    # Fe-B's OWN trust band. The module default in its training code's convexity module
    # is still HHe's (0.424, 0.252, 1.036, 0.628); Fe-B supplies its own only
    # through the config key conv_domain. Running without that key silently
    # uses the hydrogen-helium trapezoid. Carrying the value here removes the
    # trap.
    assert s.constants["trust_domain"] == (0.108, 0.0659, 0.165, 0.110)
    assert s.constants["trust_T_range"] == (1200.0, 2600.0)


def test_lj():
    s = load("lj")
    assert s.n_species == 1
    assert s.species == ("A",)
    assert s.defaults["f_form"] == "taylor"
    assert s.defaults["taylor_max_order"] == 8
    assert s.defaults["taylor_min_order"] == 2
    assert s.defaults["R_cut"] == 3.0
    assert s.defaults["sigma"] == 1.5
    assert s.defaults["m_form"] == "gamma"
    assert s.defaults["m_T_form"] == "arrhenius"
    assert s.defaults["Ea_init"] == 0.55
    assert s.defaults["T_ref"] == 1.65
    assert s.anchor_rules.T_min_by_pressure == {}


@pytest.mark.parametrize("name", ["hhe", "feb", "lj"])
def test_no_system_declares_a_raw_root(name):
    """Where the MD archives live is a fact of the machine: AIPF_RAW_<SYSTEM>, AIPF_RAW or aipf.toml."""
    s = load(name)
    assert s.paths.raw_default is None


def test_the_corrected_values_are_pinned():
    """Every value this task moved away from its brief.

    Each was wrong in the brief and right in the source. Without these, a later
    reader "restoring" the documented value gets a green suite.
    """
    feb, lj = load("feb"), load("lj")
    # The published run's config. Fe-B's mobility carries an Arrhenius factor;
    # hydrogen helium's does not. "none" survives only in a stale header.
    assert feb.defaults["m_T_form"] == "arrhenius"
    assert feb.defaults["T_ref"] == 1800.0
    # The config's f_exc_form: joint. "mlp" is hydrogen helium's g_form.
    assert feb.defaults["f_form"] == "joint_mlp"
    # The structure-factor measurement and the npz meta. 15.874 is prose only.
    assert lj.constants["box_L"] == 15.874010519681994


def test_feb_anchor_rule_follows_the_code_not_its_header():
    """The one real documented-versus-actual disagreement in this project.

    The published run's config claims `anchor_T_min_K {0: 1799}` in a
    header, "Keeps T >= 1800 K". The key 800 lines below is `{}`. The code wins.
    """
    feb = load("feb")
    assert feb.anchor_rules.T_min_by_pressure == {}


def test_no_system_carries_another_systems_values():
    """The fork makes cross-system carry-over easy, and it has happened twice.

    Fe-B's trust domain defaulted to hydrogen helium's, and the brief gave Fe-B
    hydrogen helium's g_form. Pin the things a sibling could plausibly supply.
    """
    hhe, feb, lj = load("hhe"), load("feb"), load("lj")

    assert hhe.constants["trust_domain"] != feb.constants["trust_domain"]
    assert hhe.masses != feb.masses
    assert hhe.table_keys["x"] == "x_He"
    assert feb.table_keys["x"] == "x_B"
    assert lj.table_keys["x"] == "x_A"
    assert feb.constants["modes_trees"] == (
        "modes_0GPa_v2", "modes_5GPa_v2", "modes_10GPa_v2")
    assert lj.constants["ensemble"] == "langevin_overdamped"


def test_feb_can_reproduce_its_published_from_its_own_defaults():
    """Naming a system is enough to reproduce its run.

    Hydrogen helium's declaration was completed against its published model's
    saved hyper-parameters, which left a 27-name list of knobs Fe-B still
    lacked. Those 27 are closed at Fe-B's own values, with 31 more that
    hydrogen helium has no analogue for, so the difference now runs
    almost entirely the other way. Both directions are LISTS rather than
    counts, so closing or opening one is a deliberate edit here.
    """
    hhe, feb = load("hhe"), load("feb")
    # `T_basis` is the one knob hydrogen helium declares and Fe-B does not.
    # Its joint network replaces the temperature basis rather than carrying
    # one, which the checkpoint's own enable_TlnT / enable_T2 flags confirm.
    # `diagnose` and `training` used to be the second and third; Fe-B now
    # declares both, in the package's vocabulary, beside its own config
    # spellings at the top level (pinned in test_feb_system_complete.py).
    # `rollout`: Fe-B published no rollout, so it declares none.
    # `column`: the rain column is this system's experiment only.
    assert set(hhe.defaults) - set(feb.defaults) == {"T_basis", "rollout",
                                                      "column"}
    # And what Fe-B declares that hydrogen helium does not. Every name here
    # is either a knob of a model form that system does not use, a term it
    # does not have, or a data-module knob neither published model saves and only
    # this one has been measured for.
    assert set(feb.defaults) - set(hhe.defaults) == {
        # the joint excess head, and the input scaling every net gets
        "h_u_joint", "n_hidden_joint", "activation", "rho_eps",
        "fexc_input_scale", "m_input_scale",
        # the Arrhenius mobility's second barrier and its table start
        "arrhenius_shared_Ea", "m_column", "m_init_from_table",
        # the kernel-floor term this system has and the other does not
        "lambda_wpsd", "wpsd_kappa", "wpsd_margin", "wpsd_k_max", "wpsd_n_k",
        # the anchor gate, the row weighting and the pressure denominator
        "anchor_gate", "anchor_row_weight", "p_ref_gpa",
        # the drift term's own scale, and the one grid every source uses
        "drift_scale", "grid",
        # how a window is cut and which runs validate. `estimator` is NOT
        # here: both systems declare one, and they are different values --
        # this one's is a third form the package does not register.
        "w", "n_s", "stride", "burn_in_frames",
        "run_weighting", "source_pattern", "batch_size", "num_workers",
        "val_split", "val_frac", "split_seed",
        # The loader order is HERE rather than shared because the two
        # systems declare the same value in two places: this one at the top
        # level with its other loader knobs, the other under its `training`
        # block, which is the block this system does not have.
        "order",
        # the Trainer argument the other system leaves undeclared
        "grad_clip",
    }, ("a name here that is not in the list is a knob one of the two "
        "systems has just gained or lost")
    assert feb.defaults["sigma"] == 2.0            # the published run's config
    assert feb.defaults["k_cut"] == 3.0            # its 464 modes.npz
    assert feb.defaults["k_max"] == 2.0            # :861, written out, not null
    assert feb.defaults["k_fit_stat"] == 1.5       # :937
    assert feb.defaults["R_cut"] == 4.5            # :889
    assert feb.defaults["fexc_T_ref"] == 1800.0    # :912, not hhe's 8000.0
    assert feb.defaults["fexc_T_ref"] != hhe.defaults["fexc_T_ref"]


def test_lj_declares_its_cutoff_and_its_kernel():
    """The two knobs Lennard-Jones was missing.

    k_cut is not in its config at all: it is baked into the 49 modes.npz, the
    same way it is for the other two systems.
    """
    lj = load("lj")
    assert lj.defaults["k_cut"] == 3.0
    assert lj.defaults["kernel_form"] == "radial_mlp"
    assert lj.defaults["k_max"] is None


# The tests above pin the values that an argument was had about. These three
# pin EVERYTHING ELSE, because an independent mutation sweep over the literals
# in the three files, enumerated from their source rather than from this file,
# found 43 of 118 that no assertion reached: every atomic mass, every density
# column name, Fe-B's composition channel, the pressures, and most of both
# free-energy form names. A test written from the assertion list cannot find
# that, which is how the corrections shipped unprotected the first time.
#
# These are change detectors and nothing more. They are not an independent
# source for any value: the provenance lives beside each constant in
# experiments/<name>/system.py, with the file and line that settles it. Their
# job is to make moving a number a deliberate act, done in two places, rather
# than a silent one.


def test_hhe_every_declaration_is_pinned():
    s = load("hhe")
    assert s.masses == {"H": 1.008, "He": 4.0026}
    assert (s.trust_domain.inner, s.trust_domain.outer,
            s.trust_domain.T_range) == ((0.424, 0.252), (1.036, 0.628),
                                        (2000.0, 12000.0))
    assert s.table_keys == {"rho": ("rho_H", "rho_He"), "x": "x_He",
                            "x_channel": 1}
    assert s.constants == {
        # The system's own unit convention, which `aipf.functional.build`
        # reads to turn a declared reference temperature into an energy.
        # A package-level default here would carry one system's units
        # into another's model, which is why it is declared per system
        # and pinned per system.
        "kB": 8.617333262e-5,
        "P_GPa": 800.0,
        "binodal_7000": (0.069, 0.925),
        "trust_domain": (0.424, 0.252, 1.036, 0.628),
        "gate_tol": 0.05,
        "eos_pressures_GPa": (800.0, 200.0, 400.0, 600.0),
        "modes_trees": ("modes_800GPa", "modes_200GPa", "modes_400GPa",
                        "modes_600GPa"),
        "table_trees": ("fdt_800GPa", "fdt_200GPa", "fdt_400GPa",
                        "fdt_600GPa"),
        "eos_trees": ("eos_800GPa", "eos_200GPa", "eos_400GPa",
                      "eos_600GPa"),
    }
    # The rollout, column and md blocks are pinned on their own below.
    assert s.defaults["md"] == {"potential": {"md5": "2ed3c8297221761a915e212408e803b4"}}
    assert {k: v for k, v in s.defaults.items()
            if k not in ("rollout", "column", "md")} == {
        "sigma": 2.0, "k_cut": 3.0, "mode_fields": "per_type", "k_max": None,
        "k_fit_stat": 1.5, "R_cut": 4.5, "rung": 3, "f_form": "mlp", "kernel_form": "radial_mlp",
        "T_basis": ("1", "T", "TlnT"), "fexc_T_ref": 8000.0,
        "m_form": "mlp_scaled", "m_T_form": "none", "alpha_loss": 0.0,
        "tbasis_ortho": True, "h_u": 16, "h_g": 32, "h_w": 16, "h_m": 16,
        "rho_ref": (0.35, 0.33), "T_ref": 10000.0, "h_inv_eps": 1e-6,
        "source_loss_weights": {
            "slab_800": 1.0, "cube_800": 1.0, "slab_200": 1.0,
            "cube_200": 1.0, "slab_400": 1.0, "cube_400": 1.0,
            "slab_600": 1.0, "cube_600": 1.0,
        },
        "estimator": "weak", "lambda_M": 0.5, "lambda_S": 0.01,
        "lambda_bulk": 0.01, "lambda_P": 5.0, "lambda_conv": 1.0,
        "lambda_gamma": 1.0, "stat_metric": "rel_frob",
        "conv_penalty": "hinge", "conv_margin": 0.05, "conv_samples": 1024,
        "gamma_pt_samples": 16, "lr": 5e-4, "weight_decay": 0.0,
        "warmup_epochs": 5, "anneal_epochs": 30, "max_epochs": 30,
        "wd_ghat": 0.01,
        # The diagnosis driver's knobs. Every one of them
        # is a command-line default of the published eval battery or a
        # module constant of its dense dome scan; the driver itself has no
        # defaults, so these are what a diagnosis of this system runs with.
        "diagnose": {
            "eos_csvs": {200.0: "eos_200GPa/eos_n_x_T.csv",
                          400.0: "eos_400GPa/eos_n_x_T.csv",
                          600.0: "eos_600GPa/eos_n_x_T.csv",
                          800.0: "eos_800GPa/eos_n_x_T.csv"},
            "eos_columns": {"x_column": "x_He", "T_column": "T_K",
                             "n_columns": ("n", "n_atoms_per_A3"),
                             "status_column": "status", "status_ok": "ok"},
            "poly_degree": 4,
            "manifold_fit": "column",
            "pressure_unit": 1.0 / 160.2176,
            "x_grid": (0.02, 0.98, 0.005),
            "iso_x_grid": (0.05, 0.95, 0.05),
            "x_bar": 0.5,
            "isobar": {"s_lo": 0.30, "s_hi": 1.30, "n_scan": 161,
                        "n_iter": 45, "pressure_route": "protocol"},
            "min_path_points": 8,
            "binodal_route": "convex_hull",
            "min_tie_gap": 0.015,
            "central_lo_max": 0.6,
            "central_width_max": 0.45,
            "tc_method": "dome_apex",
            "apex": {"central_lo_max": 0.6, "tail_fraction": 0.5,
                      "min_tail_points": 4, "max_overshoot_steps": 10.0},
            "dome": {"pressures": (200.0, 800.0, 50.0),
                      "T_grid": (2000.0, 11000.0, 100.0),
                      "anchor_pressures": (200.0, 400.0, 600.0, 800.0),
                      "x_grid": (0.02, 0.98, 0.005),
                      "close_after": 2},
        },
        # The training driver's knobs. Every name here is
        # one the published model's CONFIG carries and its saved hyper_parameters
        # do not -- its LightningModule never saw a window shape or
        # a loader shape, its DataModule did -- plus the three TrainConfig
        # fields the config spells differently or not at all. The names
        # this dict does NOT repeat are the ones already above: they are
        # TrainConfig field names at the top level and the driver reads
        # them from there.
        "training": {
            "source_root": {"tier": "raw", "path": "fields"},
            "half_width": 25, "n_states": 5, "stride": 10,
            "savgol_window": None, "savgol_poly": None,
            "run_weighting": "inverse_band_power",
            "run_weight_probe_every": 100,
            "val_split": "random", "val_fraction": 0.1, "split_seed": 316,
            "val_labels": ((0.1, 0.9),),
            "batch_size": 16, "num_workers": 0, "pin_memory": False,
            "drop_last": False, "order": "shuffled",
            "lambda_dyn": 1.0, "bulk_residual": "relative_inverse",
            "eta_min": 1e-6, "grad_clip": 1.0,
            # The two hinge penalties' probes.
            "penalty_seed": 2, "conv_T_measure": "log_uniform",
            "gamma_paths": {
                "eos_csvs": {
                    800.0: "eos_800GPa/eos_n_x_T.csv",
                    200.0: "eos_200GPa/eos_n_x_T.csv",
                    400.0: "eos_400GPa/eos_n_x_T.csv",
                    600.0: "eos_600GPa/eos_n_x_T.csv",
                },
                "eos_columns": {
                    "x_column": "x_He", "T_column": "T_K",
                    "n_columns": ("n", "n_atoms_per_A3"),
                    "status_column": "status", "status_ok": "ok",
                },
                "x_grid": (0.11, 0.89, 0.02), "poly_degree": 4,
                "T_measure": "uniform", "q_floor": 1e-6,
            },
            # The three measured anchor tables, handed to
            # `aipf.train.anchors.AnchorTables.load` as `**tables`. Paths
            # are RELATIVE to this system's own root and keyed by the
            # pressure each was measured at -- the key is what selects the
            # table's temperature floor out of `anchor_rules`, so a path
            # read for its `fdt_<P>GPa` segment instead (what the training
            # code did) is the thing this shape exists to rule out. The
            # ORDER is the published model's own, 800 first, and it is
            # load bearing: the tables' rows are concatenated in it and
            # every anchor loss is a mean over the concatenation.
            "tables": {
                "key": "pressure", "w0": {"route": "evaluator"},
                "m_table": {
                    800.0: "fields/fdt_800GPa/M_table.npz",
                    200.0: "fields/fdt_200GPa/M_table.npz",
                    400.0: "fields/fdt_400GPa/M_table.npz",
                    600.0: "fields/fdt_600GPa/M_table.npz",
                },
                "s_table": {
                    800.0: "fields/fdt_800GPa/S_table.npz",
                    200.0: "fields/fdt_200GPa/S_table.npz",
                    400.0: "fields/fdt_400GPa/S_table.npz",
                    600.0: "fields/fdt_600GPa/S_table.npz",
                },
                # The same four manifolds `defaults["diagnose"]` reads,
                # from one shared declaration, in the published model's order.
                "eos_csvs": {
                    800.0: "eos_800GPa/eos_n_x_T.csv",
                    200.0: "eos_200GPa/eos_n_x_T.csv",
                    400.0: "eos_400GPa/eos_n_x_T.csv",
                    600.0: "eos_600GPa/eos_n_x_T.csv",
                },
                "pressure_unit": 1.0 / 160.2176,
                "eos_columns": {
                    "x_column": "x_He", "T_column": "T_K",
                    "n_columns": ("n", "n_atoms_per_A3"),
                    "status_column": "status", "status_ok": "ok",
                },
                # The two measured tables' own column names. A published FILE
                # FORMAT, not a package name: two of them are spelled with
                # a species in them and could not live under `src/aipf`.
                "columns": {
                    "phase": "single_phase", "temperature": "T_K",
                    "densities": ("rho_H", "rho_He"),
                    "composition": "x_He", "mobility": "M_kappa0",
                    "shells": "k_shells", "structure": "S_k",
                    "bulk_target": "Scc0", "bulk_sigma": "sigma_OZ",
                    "bulk_ok": "oz_ok",
                },
                "row_weight": None,
                "pressure_floor": None,
            },
        },
    }
    assert s.defaults["rollout"] == {
        "solver": {"clamp_rho": 1e-3, "state_proj": "domain",
                   "state_clamp": 1e-3, "mass_restore": "shift",
                   "noise_eval": "ito", "noise_scale": 1.0,
                   "predictor_floor": 1e-4},
        "archive": {"file_name": "modes.npz", "amplitudes": "rho_k",
                    "amplitudes_channel_axis": 2, "labels": "nvec",
                    "box": "box", "temperature": "T_K",
                    "frame_interval": "dt_frame_ps",
                    "quality_file": "DATA_QUALITY_WARNING.json",
                    "quality_key": "mask_end"},
        "spinodal": {"modes_tree": "fields/modes_800GPa",
                     "grid": (24, 24, 24), "field_stride": 10},
        "slab": {"modes_tree": "fields/modes_800GPa", "grid": (16, 16, 64),
                 "field_stride": 10, "profile_axis": 2},
    }
    assert s.defaults["column"] == {
        "noise": Noise(mode="gaussian", m_stab="mean"),
        "masses": (1.008, 4.003), "T_window": (5000.0, 12000.0),
        "T_margin": 500.0, "ramp_min_cells": 8.0, "quarantine_thresh": 0.5,
        "seed_chunk_stride": 7919, "rain_rule": "mean_ref_v2",
        "solver": {"clamp_rho": 1e-3, "state_proj": "domain",
                   "state_clamp": 1e-3, "mass_restore": "shift",
                   "noise_eval": "ito", "noise_scale": 1.0,
                   "predictor_floor": 1e-4},
    }
    assert s.paths.system == "hhe"
    assert s.paths.raw_default is None  # the raw root is a site fact
    # Pressure is a genuine control variable here, so the farm is split on it.
    assert s.farm_partition == ("P_GPa",)
    # Undeclared: this tree's trajectories are already named for the
    # package-wide default, and its run directories already ARE their tags.
    assert s.traj_names == ("traj.lammpstrj", "traj.dump", "dump.lammpstrj")
    assert s.tag_from_path is False
    assert s.tag_path_root is None
    # The checkpoint and anchor fields. This detector enumerates them too,
    # so that the two change detectors in this repository stay in step: the
    # values themselves are pinned, with their provenance, in
    # tests/unit/test_hhe_system_complete.py.
    assert s.checkpoint.path is None
    assert s.checkpoint.md5 == "ab44d9883f2a3dd27e43cd1c783907a9"
    assert s.anchor_rules.shape_T_min_by_pressure == {
        200: 7500, 400: 9500, 600: 9500, 800: 10000}
    assert set(s.md_settings) == {"production", "eos"}


def test_feb_every_declaration_is_pinned():
    s = load("feb")
    assert s.masses == {"B": 10.811, "Fe": 55.845}
    assert (s.trust_domain.inner, s.trust_domain.outer,
            s.trust_domain.T_range) == ((0.108, 0.0659), (0.165, 0.110),
                                        (1200.0, 2600.0))
    # x_B labels channel 0, the opposite of hydrogen helium's x_He on channel
    # 1. A 0 here silently mislabels every composition it writes, and 1 is a
    # legal index, so nothing else catches it.
    assert s.table_keys == {"rho": ("rho_B", "rho_Fe"), "x": "x_B",
                            "x_channel": 0}
    assert s.constants == {
        # The system's own unit convention, which `aipf.functional.build`
        # reads to turn a declared reference temperature into an energy.
        # A package-level default here would carry one system's units
        # into another's model, which is why it is declared per system
        # and pinned per system.
        "kB": 8.617333262e-5,
        "P_GPa": 0.0,
        # Renamed from `pressures_GPa`, to the name the other
        # two-species system already used for the same quantity. Nothing
        # read either spelling.
        "eos_pressures_GPa": (0.0, 5.0, 10.0),
        "trust_domain": (0.108, 0.0659, 0.165, 0.110),
        "trust_T_range": (1200.0, 2600.0),
        "modes_trees": ("modes_0GPa_v2", "modes_5GPa_v2", "modes_10GPa_v2"),
        "fdt_runs_trees": ("fdt_0GPa_v2_runs", "fdt_5GPa_v2_runs",
                           "fdt_10GPa_v2_runs"),
        # `training_derived` is the seventh entry and the only one the
        # published model itself opens; the six before it are what it is built from.
        "table_trees": ("fdt_0GPa_v2", "fdt_5GPa_v2", "fdt_10GPa_v2",
                        "static_0GPa_v2", "static_5GPa_v2", "static_10GPa_v2",
                        "training_derived"),
        "eos_trees": ("eos_0GPa", "eos_5GPa", "eos_10GPa"),
        "m_table_files": ("M_table_v2_trainexcl.npz",
                          "M_table_5GPa_trainexcl.npz",
                          "M_table_10GPa_trainexcl.npz"),
        "s_table_files": ("S_table_v2_trainexcl_gstable.npz",
                          "S_table_5GPa_trainexcl_gstable.npz",
                          "S_table_10GPa_trainexcl_gstable.npz"),
        "eos_csv_files": ("eos_n_x_T_v2.csv", "eos_n_x_T.csv",
                          "eos_n_x_T.csv"),
        "md_exclude_dirs": ("qc",),
    }
    # The md block: the potential by md5, and one archived point of the cube campaign.
    md = s.defaults["md"]
    assert md["potential"] == {"md5": "f14252d355d95ed5932de858c7840be0"}
    assert md["cube-npt"]["campaign"] == "production"
    assert md["cube-npt"]["point"] == {
        "geometry": "cube", "ensemble": "NPT", "T": 1800.0, "x": 0.5, "P": 0.0, "dt_ps": 0.001,
        "equil_ps": 10.0, "prod_ps": 50.0, "dump_every_ps": 0.1, "dump_from": "prod", "seed": 1,
        "n_atoms": 3456}
    assert sorted(md["cube-npt"]["values"]) == ["CONFIGURATION", "RELAX", "SAMPLE_PROD", "SAMPLE_PROD_END",
                                                "THERMO_COLUMNS", "TRAJECTORY", "WRITE_END"]
    assert "(v_n_atoms/0.105084)^(1.0/3.0)" in md["cube-npt"]["values"]["CONFIGURATION"]
    # The published training partition: six sources, each a glob minus its excluded runs
    # (the runs per source are pinned against the archive in test_feb_train_partition.py).
    sources = s.defaults["training"]["sources"]
    assert {n: (v["root"], v["pattern"], tuple(v["grid"]), len(v["exclude_tags"]))
            for n, v in sources.items()} == {
        "noneq_0": ("modes_0GPa_v2", "cube_*", (32, 32, 32), 134),
        "stable_0": ("modes_0GPa_v2", "cube_*", (32, 32, 32), 53),
        "noneq_5": ("modes_5GPa_v2", "cube_*", (32, 32, 32), 143),
        "stable_5": ("modes_5GPa_v2", "cube_*", (32, 32, 32), 52),
        "noneq_10": ("modes_10GPa_v2", "cube_*", (32, 32, 32), 138),
        "stable_10": ("modes_10GPa_v2", "cube_*", (32, 32, 32), 52)}
    assert list(sources) == list(s.defaults["source_loss_weights"])
    assert {k: ({kk: vv for kk, vv in v.items() if kk != "sources"} if k == "training" else v)
            for k, v in s.defaults.items() if k != "md"} == {
        "rung": 3, "sigma": 2.0, "k_cut": 3.0, "mode_fields": "per_type",
        "k_max": 2.0,
        "k_fit_stat": 1.5, "R_cut": 4.5, "grid": (32, 32, 32),
        "fexc_T_ref": 1800.0,
        "f_form": "joint_mlp", "h_u_joint": 64, "n_hidden_joint": 4,
        "activation": "gelu", "kernel_form": "radial_mlp",
        "h_u": 16, "h_g": 32, "h_w": 16, "h_m": 16,
        "rho_ref": (0.05, 0.05), "fexc_input_scale": True,
        "m_input_scale": True, "rho_eps": 1e-5, "tbasis_ortho": True,
        "m_form": "mlp_scaled", "m_T_form": "arrhenius",
        "arrhenius_shared_Ea": False, "T_ref": 1800.0,
        "m_column": "M_k0", "m_init_from_table": True,
        "alpha_loss": 0.0, "h_inv_eps": 1e-6, "drift_scale": 1.0,
        "source_loss_weights": {
            "noneq_0": 15000.0, "stable_0": 15000.0,
            "noneq_5": 15000.0, "stable_5": 15000.0,
            "noneq_10": 15000.0, "stable_10": 15000.0,
        },
        "estimator": "weak_mid", "w": 100, "n_s": 5, "stride": 40,
        "burn_in_frames": 0, "run_weighting": "uniform",
        "source_pattern": "cube_*", "batch_size": 6, "num_workers": 4,
        "order": "shuffled",
        "val_split": "random", "val_frac": 0.1, "split_seed": 316,
        "lambda_M": 0.5, "lambda_S": 0.03, "lambda_bulk": 0.01,
        "lambda_P": 5.0, "lambda_conv": 0.0, "lambda_gamma": 0.0,
        "lambda_wpsd": 0.3, "wpsd_kappa": 2.0, "wpsd_margin": 0.0,
        "wpsd_k_max": 3.0, "wpsd_n_k": 32,
        "stat_metric": "rel_frob", "anchor_row_weight": "scc0_err",
        "anchor_gate": "anchor_eligible_g2new_gamma1_fixed",
        "p_ref_gpa": 10.0,
        "conv_penalty": "hinge", "conv_margin": 0.05, "conv_samples": 1024,
        "gamma_pt_samples": 16,
        "lr": 5e-4, "weight_decay": 0.0, "wd_ghat": 0.01,
        "warmup_epochs": 5, "anneal_epochs": 200, "max_epochs": 60,
        "grad_clip": 1.0,
        # The two driver blocks, in the package's vocabulary.
        "diagnose": {
            "eos_csvs": {0.0: "eos_0GPa/eos_n_x_T.csv",
                          5.0: "eos_5GPa/eos_n_x_T.csv",
                          10.0: "eos_10GPa/eos_n_x_T.csv"},
            "eos_columns": {"x_column": "x_B", "T_column": "T_K",
                             "n_columns": ("n", "n_atoms_per_A3"),
                             "status_column": "status", "status_ok": "ok"},
            "manifold_fit": "per_row", "manifold_min_nodes": 5,
            "poly_degree": 4, "pressure_unit": 1.0 / 160.2176,
            "isobar": {"s_lo": 0.30, "s_hi": 1.30, "n_scan": 161,
                        "n_iter": 45, "pressure_route": "closed_form"},
            "x_grid": (0.05, 0.95, 0.005),
            "iso_x_grid": (0.05, 0.95, 0.05),
            "x_bar": "auto", "binodal_route": "convex_hull",
            "min_path_points": 8, "min_tie_gap": 0.015,
            "tc_method": "dome_apex",
            "apex": {"central_lo_max": 0.6, "tail_fraction": 0.5,
                      "min_tail_points": 4, "max_overshoot_steps": 10.0},
            "stability_map": {"x_points": (0.005, 0.995, 249),
                               "min_points": 10,
                               "hessian_dtype": "float32"},
        },
        "training": {
            "source_root": {"tier": "raw", "path": "fields"},
            "half_width": 100, "n_states": 5, "stride": 40, "band_k_max": 2.0,
            "savgol_window": None, "savgol_poly": None,
            "run_weighting": "uniform",
            "val_split": "random", "val_fraction": 0.1, "split_seed": 316,
            "val_labels": ((0.1, 0.9),),
            "batch_size": 6, "num_workers": 0, "pin_memory": False,
            "drop_last": False, "order": "shuffled",
            "lambda_dyn": 1.0, "bulk_residual": "relative_inverse",
            "eta_min": 1e-6, "grad_clip": 1.0,
            "tables": {
                "key": "pressure", "w0": {"route": "evaluator"},
                "m_table": {
                    0.0: "fields/training_derived/M_table_v2_trainexcl.npz",
                    5.0: "fields/training_derived/M_table_5GPa_trainexcl.npz",
                    10.0: "fields/training_derived/"
                          "M_table_10GPa_trainexcl.npz",
                },
                "s_table": {
                    0.0: "fields/training_derived/"
                         "S_table_v2_trainexcl_gstable.npz",
                    5.0: "fields/training_derived/"
                         "S_table_5GPa_trainexcl_gstable.npz",
                    10.0: "fields/training_derived/"
                          "S_table_10GPa_trainexcl_gstable.npz",
                },
                "eos_csvs": {0.0: "eos_0GPa/eos_n_x_T_v2.csv",
                             5.0: "eos_5GPa/eos_n_x_T.csv",
                             10.0: "eos_10GPa/eos_n_x_T.csv"},
                "pressure_unit": 1.0 / 160.2176,
                "eos_columns": {"x_column": "x_B", "T_column": "T_K",
                                "n_columns": ("n", "n_atoms_per_A3"),
                                "status_column": "status", "status_ok": "ok"},
                "columns": {
                    "phase": {"mobility": "single_phase",
                              "structure":
                                  "anchor_eligible_g2new_gamma1_fixed"},
                    "temperature": "T_K", "densities": ("rho_B", "rho_Fe"),
                    "composition": "x_B", "mobility": "M_k0",
                    "shells": "k_shells", "structure": "S_k",
                    "bulk_target": "Scc0", "bulk_sigma": None,
                    "bulk_ok": None,
                },
                "row_weight": {"value": "Scc0", "error": "Scc0_err",
                               "floor": 0.02},
                "pressure_floor": 10.0,
            },
        },
    }
    assert s.functional.form == "nonlocal_kernel"
    assert (s.functional.local, s.functional.kernel) == ("mlp", "radial_mlp")
    assert dict(s.functional.kwargs) == dict(
        grid=(32, 32, 32), R_cut=4.5, rho_ref=(0.05, 0.05),
        h_u=16, h_g=32, h_w=16, h_m=16, fexc_T_ref=1800.0, T_ref=1800.0,
        rho_eps=1e-5, activation="gelu", g_form="icnn", enable_TlnT=False,
        enable_T2=False, tbasis_ortho=True, gauge_fix=False, disable_u=False,
        disable_g=False, f_exc_form="joint", h_joint=64, joint_depth=4,
        local_input_scale=True, ideal_form="gas", h_g_hat=16, h_g_tilde=16,
        tbasis_ortho_window=(8.617333262e-5 * 1500.0,
                             8.617333262e-5 * 2600.0),
        tbasis_ortho_points=13, kernel_n_quad=512, kernel_n_k_table=513,
        kernel_k_table_max=8.0, nyquist_mask=True)
    assert (s.mobility.form, s.mobility.T_form) == ("mlp_scaled", "arrhenius")
    assert dict(s.mobility.kwargs) == dict(
        arrhenius_shared_Ea=False, mobility_prefactor="partial_density",
        mobility_input_ref=(0.05, 0.05),
        mobility_activation_energy_init=(math.log(math.expm1(0.38)),
                                         math.log(math.expm1(0.50))))
    assert s.paths.system == "feb"
    assert s.paths.raw_default is None  # the raw root is a site fact
    # Every EOS pressure (0, 5, 10 GPa) is a genuine control variable here too.
    assert s.farm_partition == ("P_GPa",)
    assert s.traj_names == ("traj.lammpstrj", "traj.dump", "dump.lammpstrj")
    assert s.tag_from_path is False
    assert s.tag_path_root is None
    # The checkpoint and anchor fields. This detector enumerates them too, so that
    # the two change detectors in this repository stay in step: the values
    # themselves are pinned, with their provenance, in
    # tests/unit/test_feb_system_complete.py.
    assert s.checkpoint.path is None
    assert s.checkpoint.md5 == "663fcd195951e9726148fee9d4e4a8b4"
    assert s.anchor_rules.shape_T_min_by_pressure == {0: 2300, 5: 2300,
                                                      10: 2300}
    assert set(s.md_settings) == {"production", "eos"}


def test_lj_every_declaration_is_pinned():
    s = load("lj")
    assert s.masses == {"A": 1.0}
    assert s.trust_domain is None  # declared: no penalty trained
    assert s.atom_types == {"A": 1}
    assert s.table_keys == {"rho": ("rho_A",), "x": "x_A", "x_channel": 0}
    assert s.constants == {
        # The system's own unit convention, which `aipf.functional.build`
        # reads to turn a declared reference temperature into an energy;
        # this system works in reduced units, so it is exactly 1.
        "kB": 1.0,
        "P_GPa": None,
        "box_L": 15.874010519681994,
        "n_atoms": 4000,
        "rho_total": 1.0,
        "x_A": 0.5,
        "ensemble": "langevin_overdamped",
    }
    md = s.defaults["md"]
    assert set(md) == {"cube-overdamped"}
    assert set(md["cube-overdamped"]) == {"point", "values", "dt_equil_ps"}
    assert md["cube-overdamped"]["point"]["T"] == 1.60
    assert md["cube-overdamped"]["values"]["GAMMA"] == "2.0"
    assert md["cube-overdamped"]["values"]["MASSES"] == (
        "mass            1 1.0\nmass            2 1.0")
    tables = s.defaults["training"]["tables"]
    # Under the declared code tree; bare and relative where the checkout declares none.
    assert ("/" + tables["m_table"]).endswith("/FDT_M/results/M_meso_homogeneous.npz")
    assert ("/" + tables["s_table"]).endswith(
        "/Tc_structure_factor/results/Scc_reciprocal_from_md.npz")
    training = {k: v for k, v in s.defaults["training"].items() if k != "tables"}
    assert {k: v for k, v in tables.items() if k not in ("m_table", "s_table")} == {
        "key": "temperature", "state": (0.5,), "rho_total": 1.0,
        "w0": {"route": "radial", "r_max": 3.0, "n_points": 1000},
        "columns": {"temperature": "T", "mobility": "M_mean",
                    "records": "records", "record_temperature": "T",
                    "structure_zero": "Scc0", "temperatures": "TS"}}
    assert training == {
        "source_root": {"tier": "farm", "path": "modes"},
        "modes_root": "Data/slab_overdamped/modes",
        "train_temperatures": (1.10, 1.15, 1.25, 1.30, 1.35, 1.40,
                               1.45, 1.50, 1.60, 1.70),
        "val_seeds": (45,), "test_temperatures": (1.20,),
        "half_width": 25, "n_states": 5, "stride": 20,
        "estimator": "weak", "batch_size": 16, "order": "shuffled",
        "drop_last": True, "grad_clip": 1.0, "eta_min": 1e-6,
        "lambda_dyn": 1.0, "lambda_bulk": 0.01, "lambda_S": 0.0,
        "bulk_residual": "sigma_chi2", "val_split": "none", "val_labels": (),
        "val_fraction": 0.0, "split_seed": 0,
        "run_weighting": "inverse_band_power", "run_weight_probe_every": 317,
        "savgol_window": None, "savgol_poly": None, "num_workers": 4,
        "pin_memory": True,
    }
    assert {k: v for k, v in s.defaults.items() if k not in ("md", "training")} == {
        "rung": 3, "sigma": 1.5, "k_cut": 3.0, "kernel_form": "radial_mlp",
        "mode_fields": ({"name": "A", "weights": {1: 0.5, 2: -0.5}, "mean": 0.5},),
        "estimator": "weak",
        "k_max": None, "R_cut": 3.0, "f_form": "taylor",
        "taylor_max_order": 8, "taylor_min_order": 2, "gauge_fix_e": False,
        "g_exc_hidden": 32, "m_form": "gamma", "gamma": 0.1,
        "m_T_form": "arrhenius", "Ea_init": 0.55, "T_ref": 1.65,
        "alpha_loss": 0.0, "epochs": 25,
        "grid": (20, 20, 80), "box": (9.524406, 9.524406, 38.097625),
        "lr": 5e-4, "weight_decay": 0.0, "warmup_epochs": 5,
        "anneal_epochs": 25, "eta_min": 1e-6, "h_inv_eps": 1e-6,
        "lambda_drift": 1.0, "lambda_M": 0.01, "lambda_S": 0.01,
        "m_table": "FDT_M/results/M_meso_homogeneous.npz",
        "s_table": "Tc_structure_factor/results/Scc_reciprocal_from_md.npz",
        "diagnose": {"one_field": {
            "readoff": "mu_roots",
            "T_grid": {"lo": 0.5, "hi": 2.0, "n": 40},
            "dtype": "float32",
            "phi_grid": {"lo": 1e-3, "hi": 1 - 1e-3, "n": 2001},
            "phi_clip": 1e-6, "min_width": 0.02, "bracket_eps": 1e-6,
            "mu_w0": {"r_max": 4.5, "n_points": 1024},
            "tc_w0": {"r_max": 3.0, "n_points": 1000},
            "ising": {"T_max": 1.40, "beta": 0.325, "B0": 1.5,
                      "Tc0_floor": 1.42, "Tc0_offset": 0.05,
                      "B_bounds": (0.1, 5.0),
                      "Tc_bounds_offset": (1e-3, 2.0), "maxfev": 10000},
        }},
    }
    assert s.checkpoint.path is None
    assert s.checkpoint.md5 == "ef9da0b40952f7381f0ebaba876fdd5c"
    assert (s.functional.form, s.functional.local, s.functional.kernel) == (
        "nonlocal_kernel", "taylor", "radial_mlp")
    assert dict(s.functional.kwargs) == dict(
        grid=(20, 20, 80), R_cut=3.0, rho_ref=(0.5,), h_g=32, h_w=16,
        T_ref=1.65, rho_eps=1e-4, activation="gelu",
        g_form="icnn", enable_TlnT=False, enable_T2=False, h_g_hat=None,
        h_g_tilde=None, gauge_fix=False, disable_g=False, f_exc_form="split",
        ideal_form="lattice", local_input_scale=False, u_degree=8,
        u_parity="even", u_variable="difference", g_symmetry="mirror",
        icnn_output_bias=False, kernel_argument="difference",
        kernel_evaluator="lattice_sum", nyquist_mask=False)
    assert (s.mobility.form, s.mobility.T_form) == ("lattice_scalar", "arrhenius")
    assert dict(s.mobility.kwargs) == dict(
        mobility_prefactor="mole_fraction", shape_init=math.log(0.1),
        mobility_activation_energy_init=math.log(math.exp(0.55) - 1.0))
    assert s.paths.system == "lj"
    assert s.paths.raw_default is None  # the raw root is a site fact
    # Pressure is not a controlled variable here (P_GPa is None above), so
    # the farm is not split on it at all.
    assert s.farm_partition == ()
    # This tree writes no meta.json and spells its trajectories under four
    # different names, only one of which is the package-wide default; its
    # run directory is a structural name ("MD", "seed_<n>") with the real tag
    # one or more levels up, under the "Data" container.
    assert s.traj_names == ("dump.lammpstrj", "dump.seed.lammpstrj",
                            "dump.quench.lammpstrj", "dump.parent.lammpstrj")
    assert s.tag_from_path is True
    assert s.tag_path_root == "Data"
