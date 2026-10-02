"""Iron-boron.

Every value reproduces the published model, whose tracked checkpoint carries its training
configuration as ``hyper_parameters`` (``hparams`` below; ``mk`` is ``hparams["model_kwargs"]``;
``state_dict`` is the published model's). The excess free energy is a JOINT network (no convex
head), and there is no anchor temperature cut but a 2300 K kernel-shape floor.
Energies in eV, lengths in Angstrom, temperatures in K, pressures in GPa.
"""
import json
import math
from pathlib import Path

from aipf.paths import Paths
from aipf.system import (AnchorRules, Checkpoint, Functional, MdSettings,
                         Mobility, PreparationRules, System, TrustDomain)

_K_B = 8.617333262e-5  # eV/K

# rho_B/0.108 + rho_Fe/0.0659 >= 1 and rho_B/0.165 + rho_Fe/0.110 <= 1, T in [1200, 2600] K:
# hparams["conv_domain"], the density trapezoid the training data cover.
_TRUST = TrustDomain(inner=(0.108, 0.0659), outer=(0.165, 0.110), T_range=(1200.0, 2600.0))

_GPA_IN_MODEL_PRESSURE = 1.0 / 160.2176  # eV/A^3 per GPa

_RHO_REF = (0.05, 0.05)  # mk["rho_ref"]

_EOS_COLUMNS = {  # the csv header; x_B labels channel 0
    "x_column": "x_B", "T_column": "T_K",
    "n_columns": ("n", "n_atoms_per_A3"),
    "status_column": "status", "status_ok": "ok",
}

_EOS_MANIFOLDS = {  # hparams["eos_pressures"] -> hparams["eos_csvs"], relative to the system root
    0.0: "eos_0GPa/eos_n_x_T_v2.csv",
    5.0: "eos_5GPa/eos_n_x_T.csv",
    10.0: "eos_10GPa/eos_n_x_T.csv",
}


# The barostatted cube of the 0 GPa campaign at one archived point, x_B = 0.5 and 1800 K, with the
# archive's timestep, melt, production length, dump cadence, seed and atom count.
_MD_CUBE_NPT = {
    "campaign": "production",  # md_settings below
    "point": {"geometry": "cube", "ensemble": "NPT",
              "T": 1800.0, "x": 0.5,
              "P": 0.0,  # GPa; the archive held 0 GPa at 1 bar, this deck at 0 bar
              "dt_ps": 0.001,
              "equil_ps": 10.0,  # the melt, held at melt_T (preparation below)
              "prod_ps": 50.0,
              "dump_every_ps": 0.1,  # 100 steps
              "dump_from": "prod",  # the melt is not dumped
              "seed": 1,
              "n_atoms": 3456},
    "values": {
        # The archive's start: a simple-cubic scaffold of ceil(N^(1/3))^3 sites at the point's
        # equation-of-state density (0.105084 atoms/A^3 at x_B = 0.5, 1800 K), N kept at random,
        # B on floor(x_B N) of them (the archive rounded), jittered uniformly by up to 0.05 A (the
        # archive drew Gaussian, sigma 0.05 A); the melt at 2600 K erases both differences.
        "CONFIGURATION": (
            "variable        n_atoms equal @N_ATOMS@\n"
            "variable        edge equal (v_n_atoms/0.105084)^(1.0/3.0)\n"
            "variable        sites equal ceil(v_n_atoms^(1.0/3.0))\n"
            "region          box block 0 ${edge} 0 ${edge} 0 ${edge} units box\n"
            "create_box      2 box\n"
            "lattice         sc $(v_edge/v_sites)\n"
            "create_atoms    2 box\n"
            "delete_atoms    random count $(v_sites^3-v_n_atoms) yes all NULL @SEED@\n"
            "set             type 2 type/ratio 1 @X@ @SEED@\n"
            "displace_atoms  all random 0.05 0.05 0.05 @SEED@ units box"),
        "RELAX": "",  # no minimisation in the archived deck
        "TRAJECTORY": "traj.dump",  # the archive's trajectory file name
        "WRITE_END": "write_data      final.data",
        # The archive's thermo line, and its production average written to thermo_prod.txt
        # (columns TimeStep v_vtemp v_vpress v_vvol v_vpe), which the farm links as thermo.txt
        "THERMO_COLUMNS": "step time temp pe press vol density",
        "SAMPLE_PROD": (
            "variable        vtemp equal temp\n"
            "variable        vpress equal press\n"
            "variable        vvol equal vol\n"
            "variable        vpe equal pe\n"
            "fix             samp all ave/time @N_THERMO@ 1 @N_THERMO@ v_vtemp v_vpress v_vvol v_vpe &\n"
            "                file thermo_prod.txt"),
        "SAMPLE_PROD_END": "unfix           samp",
    },
}


def _inv_softplus(y: float) -> float:
    """``softplus^-1``, as ``log(expm1(y))``."""
    return math.log(math.expm1(y))


SYSTEM = System(
    name="feb",
    n_species=2,
    species=("B", "Fe"),  # index 0 = B
    masses={"B": 10.811, "Fe": 55.845},  # amu
    atom_types={"B": 1, "Fe": 2},  # dump types
    table_keys={"rho": ("rho_B", "rho_Fe"), "x": "x_B", "x_channel": 0},  # x_B is channel 0
    paths=Paths(system="feb"),  # raw root: AIPF_RAW_FEB, AIPF_RAW or [paths.raw] in aipf.toml
    checkpoint=Checkpoint(md5="663fcd195951e9726148fee9d4e4a8b4"),  # epoch 59, step 12960
    # mk, minus m_table (the build opens no file); `local` is the split form's u_form, inert under the joint form.
    functional=Functional(
        form="nonlocal_kernel", local="mlp", kernel="radial_mlp",  # mk has no u_form: "mlp"
        kwargs=dict(
            grid=(32, 32, 32),  # mk["grid"]
            R_cut=4.5,  # mk["R_cut"]
            rho_ref=_RHO_REF,
            h_u=16, h_g=32, h_w=16, h_m=16,  # mk["h_u"], ["h_g"], ["h_w"], ["h_m"]
            fexc_T_ref=1800.0,  # mk["fexc_T_ref"]
            T_ref=1800.0,  # mk["T_ref"], the Arrhenius reference
            rho_eps=1e-5,  # mk["rho_eps"]
            activation="gelu",  # mk["activation"]
            g_form="icnn",  # mk["g_form"], never built under the joint form
            enable_TlnT=False, enable_T2=False,  # mk["enable_TlnT"], ["enable_T2"]
            tbasis_ortho=True,  # mk["tbasis_ortho"]; inert, no T-basis head
            gauge_fix=False, disable_u=False, disable_g=False,  # mk["gauge_fix"], ["disable_u"], ["disable_g"]
            f_exc_form="joint",  # mk["f_exc_form"]
            h_joint=64,  # mk["h_u_joint"]
            joint_depth=4,  # mk["n_hidden_joint"]
            local_input_scale=True,  # mk["fexc_input_scale"]
            ideal_form="gas",  # rho ln rho - rho
            h_g_hat=16, h_g_tilde=16,  # constructor defaults; mk carries neither
            tbasis_ortho_window=(_K_B * 1500.0, _K_B * 2600.0),  # mk["tbasis_ortho_window"], K -> energy
            tbasis_ortho_points=13,  # np.linspace(t_lo, t_hi, 13)
            kernel_n_quad=512, kernel_n_k_table=513,  # state_dict kernel.r_quad / kernel.k_lin sizes
            kernel_k_table_max=8.0,  # state_dict kernel.k_lin[-1]
            nyquist_mask=True,  # grad and div zero the Nyquist wavenumber (the odd-order masks)
        ),
    ),
    # mk["m_form"] / mk["m_T_form"]; the prefactor and the start of the barriers are the saved model's.
    mobility=Mobility(
        form="mlp_scaled", T_form="arrhenius",  # mk["m_form"], mk["m_T_form"]
        kwargs=dict(
            arrhenius_shared_Ea=False,  # mk["arrhenius_shared_Ea"]
            mobility_prefactor="partial_density",  # the sqrt(rho) sandwich
            mobility_input_ref=_RHO_REF,  # mk["m_input_scale"] is True: z = rho / mk["rho_ref"] - 1
            mobility_activation_energy_init=(  # one barrier per channel, eV
                _inv_softplus(0.38), _inv_softplus(0.50)),
        ),
    ),
    preparation=PreparationRules(
        melt_T=2600.0,  # K
        # No ramp_offset: this system has no slab data.
    ),
    # No published stochastic rollout, so no noise is declared.
    noise=None,
    # Declared; the published model trained both penalties at weight 0 (lambda_conv, lambda_gamma below),
    # so fit draws nothing on it.
    trust_domain=_TRUST,
    anchor_rules=AnchorRules(
        T_min_by_pressure={},  # hparams["anchor_T_min_K"]: {}; the anchor gate decides
        shape_T_min_by_pressure={0: 2300, 5: 2300, 10: 2300},  # M_table.npz kappa_T_min
    ),
    constants={
        "kB": 8.617333262e-5,  # eV/K
        "P_GPa": 0.0,  # reference pressure
        "eos_pressures_GPa": (0.0, 5.0, 10.0),  # hparams["eos_pressures"]
        "trust_domain": _TRUST.inner + _TRUST.outer,  # the rollout's reading of trust_domain above
        "trust_T_range": _TRUST.T_range,
        "modes_trees": ("modes_0GPa_v2", "modes_5GPa_v2", "modes_10GPa_v2"),
        "fdt_runs_trees": ("fdt_0GPa_v2_runs", "fdt_5GPa_v2_runs",
                           "fdt_10GPa_v2_runs"),
        "table_trees": ("fdt_0GPa_v2", "fdt_5GPa_v2", "fdt_10GPa_v2",
                        "static_0GPa_v2", "static_5GPa_v2", "static_10GPa_v2",
                        "training_derived"),
        "eos_trees": ("eos_0GPa", "eos_5GPa", "eos_10GPa"),
        "m_table_files": ("M_table_v2_trainexcl.npz",  # the plain exclusion set
                          "M_table_5GPa_trainexcl.npz",
                          "M_table_10GPa_trainexcl.npz"),
        "s_table_files": ("S_table_v2_trainexcl_gstable.npz",  # the gated set
                          "S_table_5GPa_trainexcl_gstable.npz",
                          "S_table_10GPa_trainexcl_gstable.npz"),
        "eos_csv_files": ("eos_n_x_T_v2.csv", "eos_n_x_T.csv",  # basenames differ
                          "eos_n_x_T.csv"),
        "md_exclude_dirs": ("qc",),  # qc/cadence_test/ duplicates production tags
    },
    md_settings={  # read off every archived deck; both campaigns identical
        "production": MdSettings(
            units="metal",
            skin=2.0,
            neigh_modify="every 10 delay 0 check yes",
            thermo_every=100,
            T_damp=0.1,
            P_damp=1.0,
            pressure_scale=1e4,  # bar per GPa
        ),
        "eos": MdSettings(
            units="metal",
            skin=2.0,
            neigh_modify="every 10 delay 0 check yes",
            thermo_every=100,
            T_damp=0.1,
            P_damp=1.0,
            pressure_scale=1e4,
        ),
    },
    farm_partition=("P_GPa",),  # md/<P>GPa/<tag>
    defaults={
        "rung": 3,
        "sigma": 2.0,             # Angstrom
        "k_cut": 3.0,             # modes.npz k_cut, inverse Angstrom
        "mode_fields": "per_type",  # modes.npz rho_k: one channel per species, (frames, modes, 2)
        "k_max": 2.0,
        "k_fit_stat": 1.5,
        "R_cut": 4.5,             # kernel range in Angstrom
        "grid": (32, 32, 32),
        "fexc_T_ref": 1800.0,
        "f_form": "joint_mlp",    # f_exc_form: joint
        "h_u_joint": 64,
        "n_hidden_joint": 4,
        "activation": "gelu",
        "kernel_form": "radial_mlp",
        "h_u": 16,
        "h_g": 32,
        "h_w": 16,
        "h_m": 16,
        "rho_ref": (0.05, 0.05),
        "fexc_input_scale": True,
        "m_input_scale": True,
        "rho_eps": 1e-5,          # the floor under the ideal term
        "tbasis_ortho": True,     # inert here
        # -- the mobility ---------------------------------------------------
        "m_form": "mlp_scaled",
        "m_T_form": "arrhenius",
        "arrhenius_shared_Ea": False,
        "T_ref": 1800.0,          # the Arrhenius reference temperature
        "m_column": "M_k0",
        "m_init_from_table": True,
        # -- the drift residual ---------------------------------------------
        "alpha_loss": 0.0,        # pure H^-1
        "h_inv_eps": 1e-6,
        "drift_scale": 1.0,       # hparams["drift_scale"]
        "source_loss_weights": {
            "noneq_0": 15000.0, "stable_0": 15000.0,
            "noneq_5": 15000.0, "stable_5": 15000.0,
            "noneq_10": 15000.0, "stable_10": 15000.0,
        },
        # -- how a window is cut, and which runs validate (not in the checkpoint)
        "estimator": "weak_mid",
        "w": 100,                 # half window in frames (10 ps)
        "n_s": 5,                 # states per window
        "stride": 40,             # frames between window centres
        "burn_in_frames": 0,
        "run_weighting": "uniform",
        "source_pattern": "cube_*",   # and the five siblings of each source
        "batch_size": 6,
        "num_workers": 4,
        "order": "shuffled",
        "val_split": "random",
        "val_frac": 0.1,
        "split_seed": 316,
        # -- the anchor terms -----------------------------------------------
        "lambda_M": 0.5,
        "lambda_S": 0.03,
        "lambda_bulk": 0.01,
        "lambda_P": 5.0,
        "lambda_conv": 0.0,
        "lambda_gamma": 0.0,
        "lambda_wpsd": 0.3,
        "wpsd_kappa": 2.0,
        "wpsd_margin": 0.0,       # hparams["wpsd_margin"]
        "wpsd_k_max": 3.0,        # hparams["wpsd_k_max"]
        "wpsd_n_k": 32,           # hparams["wpsd_n_k"]
        "stat_metric": "rel_frob",  # hparams["stat_metric"]
        "anchor_row_weight": "scc0_err",
        "anchor_gate": "anchor_eligible_g2new_gamma1_fixed",
        "p_ref_gpa": 10.0,
        "conv_penalty": "hinge",
        "conv_margin": 0.05,
        "conv_samples": 1024,
        "gamma_pt_samples": 16,
        # -- the optimiser --------------------------------------------------
        "lr": 5e-4,
        "weight_decay": 0.0,
        "wd_ghat": 0.01,
        "warmup_epochs": 5,
        "anneal_epochs": 200,
        "max_epochs": 60,
        "grad_clip": 1.0,         # Trainer argument, in no checkpoint
        # -- the diagnosis
        "diagnose": {
            # eos_<P>GPa/eos_n_x_T.csv, tracked under experiments/feb/eos/ (read before the raw
            # root). At 0 GPa this file is the grid measured after the apex refinement, byte for byte
            # the eos_0GPa/eos_n_x_T_v2.csv the training block reads: one table under two names
            "eos_csvs": {P: f"eos_{int(P)}GPa/eos_n_x_T.csv" for P in _EOS_MANIFOLDS},
            "eos_columns": dict(_EOS_COLUMNS),
            "manifold_fit": "per_row",  # the manifolds have holes
            "manifold_min_nodes": 5,  # measured compositions a temperature row needs
            "poly_degree": 4,
            "pressure_unit": _GPA_IN_MODEL_PRESSURE,
            "isobar": {"s_lo": 0.30, "s_hi": 1.30, "n_scan": 161,
                       "n_iter": 45,
                       "pressure_route": "closed_form"},
            "x_grid": (0.05, 0.95, 0.005),
            "iso_x_grid": (0.05, 0.95, 0.05),
            "x_bar": "auto",  # the most unstable interior point
            "binodal_route": "convex_hull",
            "min_path_points": 8,
            "min_tie_gap": 0.015,
            "tc_method": "dome_apex",
            "apex": {"central_lo_max": 0.6, "tail_fraction": 0.5,
                     "min_tail_points": 4, "max_overshoot_steps": 10.0},
            # the published stability map
            "stability_map": {
                "x_points": (0.005, 0.995, 249),  # np.linspace(0.005, 0.995, 249)
                "min_points": 10,  # an isobar with fewer points is skipped
                "hessian_dtype": "float32",  # H_tot(0) of the model in float32
            },
        },
        # -- aipf md run: the potential the archive was generated with, by md5 (its file, the ML-IAP
        # conversion, is a site fact: Site.for_system), and the one declared deck
        "md": {
            "potential": {"md5": "f14252d355d95ed5932de858c7840be0"},
            "cube-npt": _MD_CUBE_NPT,
        },
        # -- aipf.train.fit's own vocabulary for the loader and the anchors
        "training": {
            "source_root": {"tier": "raw", "path": "fields"},  # <raw>/fields/...
            # the published training partition: per source, its tree, glob, grid and excluded runs
            "sources": json.loads((Path(__file__).parent / "training_sources.json").read_text()),
            "half_width": 100,  # frames
            "n_states": 5,
            "stride": 40,
            # the scattered mode band, |k| <= k_max in any box the run visits; the stored
            # k_cut 3.0 ball reaches |n| = 17 on the longest boxes, past a 32^3 grid
            "band_k_max": 2.0,
            "savgol_window": None,  # weak_mid reads no taps
            "savgol_poly": None,
            "run_weighting": "uniform",
            "val_split": "random",
            "val_fraction": 0.1,
            "split_seed": 316,
            "val_labels": ((0.1, 0.9),),  # inert under a random split
            "batch_size": 6,
            "num_workers": 0,  # throughput only
            "pin_memory": False,
            "drop_last": False,  # DataLoader default
            "order": "shuffled",
            "lambda_dyn": 1.0,  # hparams["drift_scale"] 1.0; no other drift weight
            "bulk_residual": "relative_inverse",
            "eta_min": 1e-6,  # hparams["eta_min"]
            "grad_clip": 1.0,  # the Trainer's
            # AnchorTables.load(**tables): paths relative to the system root, published model order.
            "tables": {
                "key": "pressure",  # keys are pressures in GPa, as anchor_rules.T_min_by_pressure
                # H0 at k = 0 through the kernel's own k path
                "w0": {"route": "evaluator"},
                "m_table": {  # hparams["m_table"]
                    0.0: "fields/training_derived/M_table_v2_trainexcl.npz",
                    5.0: "fields/training_derived/M_table_5GPa_trainexcl.npz",
                    10.0: "fields/training_derived/M_table_10GPa_trainexcl.npz",
                },
                "s_table": {  # hparams["s_table"]
                    0.0: "fields/training_derived/S_table_v2_trainexcl_gstable.npz",
                    5.0: "fields/training_derived/S_table_5GPa_trainexcl_gstable.npz",
                    10.0: "fields/training_derived/S_table_10GPa_trainexcl_gstable.npz",
                },
                "eos_csvs": dict(_EOS_MANIFOLDS),
                "pressure_unit": _GPA_IN_MODEL_PRESSURE,
                "eos_columns": dict(_EOS_COLUMNS),
                "columns": {  # np.load(...).files of the tables above
                    # M rows: single_phase; S and bulk: hparams["anchor_gate"]
                    "phase": {"mobility": "single_phase",
                              "structure": "anchor_eligible_g2new_gamma1_fixed"},
                    "temperature": "T_K",
                    "densities": ("rho_B", "rho_Fe"),
                    "composition": "x_B",  # channel 0, table_keys["x_channel"]
                    "mobility": "M_k0",  # hparams["m_column"]
                    "shells": "k_shells",
                    "structure": "S_k",
                    "bulk_target": "Scc0",
                    "bulk_sigma": None,  # a gate column replaces the OZ test
                    "bulk_ok": None,
                },
                # hparams["anchor_row_weight"] = "scc0_err": 1 / max(|error/value|, floor)^2
                "row_weight": {"value": "Scc0", "error": "Scc0_err", "floor": 0.02},
                "pressure_floor": 10.0,  # hparams["p_ref_gpa"], GPa
            },
        },
    },
)
