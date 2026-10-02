"""Hydrogen-helium at high pressure.

Every value reproduces the published model, whose tracked checkpoint carries its training
configuration as ``hyper_parameters`` (``hparams`` below; ``mk`` is ``hparams["model_kwargs"]``).
Energies in eV, lengths in Angstrom, temperatures in K, pressures in GPa.
"""

from aipf.paths import Paths
from aipf.system import (AnchorRules, Checkpoint, Functional, MdSettings,
                         Mobility, Noise, PreparationRules, System,
                         TrustDomain)

_K_B = 8.617333262e-5  # eV/K

# rho_H/0.424 + rho_He/0.252 >= 1 and rho_H/1.036 + rho_He/0.628 <= 1, T in [2000, 12000] K:
# the density trapezoid the training data cover.
_TRUST = TrustDomain(inner=(0.424, 0.252), outer=(1.036, 0.628), T_range=(2000.0, 12000.0))

_GPA_IN_MODEL_PRESSURE = 1.0 / 160.2176  # eV/A^3 per GPa

_EOS_COLUMNS = {  # the measured equation-of-state tables' header
    "x_column": "x_He", "T_column": "T_K",
    "n_columns": ("n", "n_atoms_per_A3"),
    "status_column": "status", "status_ok": "ok",
}

_EOS_MANIFOLDS = {  # keyed by pressure, relative to the system root
    800.0: "eos_800GPa/eos_n_x_T.csv",
    200.0: "eos_200GPa/eos_n_x_T.csv",
    400.0: "eos_400GPa/eos_n_x_T.csv",
    600.0: "eos_600GPa/eos_n_x_T.csv",
}


SYSTEM = System(
    name="hhe",
    n_species=2,
    species=("H", "He"),  # channel 0 = H
    masses={"H": 1.008, "He": 4.0026},  # amu
    atom_types={"H": 1, "He": 2},  # dump types
    table_keys={"rho": ("rho_H", "rho_He"), "x": "x_He", "x_channel": 1},
    paths=Paths(system="hhe"),  # raw root: AIPF_RAW_HHE, AIPF_RAW or [paths.raw] in aipf.toml
    checkpoint=Checkpoint(md5="ab44d9883f2a3dd27e43cd1c783907a9"),  # the best-validation epoch
    # mk, minus m_table (the build opens no file); ideal_form, u_form and f_exc_form are not saved and take the
    # values the saved model was built with.
    functional=Functional(
        form="nonlocal_kernel", local="mlp", kernel="radial_mlp",
        kwargs=dict(
            grid=(16, 16, 64), R_cut=4.5, rho_ref=(0.35, 0.33),
            h_u=16, h_g=32, h_w=16, h_m=16,
            fexc_T_ref=8000.0, T_ref=10000.0, rho_eps=1e-5,
            activation="gelu", g_form="mlp",
            enable_TlnT=True, enable_T2=False, tbasis_ortho=True,
            gauge_fix=False, disable_u=False, disable_g=False,
            u_form="mlp", f_exc_form="split", ideal_form="gas",  # not saved
            h_g_hat=16, h_g_tilde=16,  # not saved; mandatory with TlnT
            tbasis_ortho_window=(_K_B * 9000.0, _K_B * 12000.0),  # 9000-12000 K as an energy
            tbasis_ortho_points=13,
            kernel_n_quad=512, kernel_n_k_table=513, kernel_k_table_max=8.0,  # read off the kernel buffers
            local_input_scale=False,  # no mk["fexc_input_scale"]: the nets read raw rho
            nyquist_mask=True,  # grad and div zero the Nyquist wavenumber (the odd-order masks)
        ),
    ),
    # mk["m_form"] / mk["m_T_form"]; the prefactor is the saved model's, NOT the constructor default.
    mobility=Mobility(
        form="mlp_scaled", T_form="none",
        kwargs=dict(arrhenius_shared_Ea=False,
                    mobility_prefactor="partial_density",  # the sqrt(rho) sandwich
                    mobility_input_ref=None),  # no mk["m_input_scale"]: the net reads raw rho
    ),
    preparation=PreparationRules(
        ramp_offset=2000.0,  # K above the target; the cube runs only
    ),
    # Conserved noise filtered at the coarse-graining sigma (2.0 A); every published rollout froze the
    # semi-implicit operator at the largest-norm cell.
    noise=Noise(mode="gaussian", m_stab="max"),
    trust_domain=_TRUST,
    anchor_rules=AnchorRules(
        T_min_by_pressure={200: 6100, 400: 7800, 600: 8800, 800: 8973},  # hparams["anchor_T_min_K"]; T > T_min
        # Kernel-shape floor, NOT the cut above; inclusive.
        shape_T_min_by_pressure={200: 7500, 400: 9500, 600: 9500,
                                 800: 10000},
    ),
    constants={
        "kB": _K_B,  # eV/K
        "P_GPa": 800.0,  # reference pressure
        "binodal_7000": (0.069, 0.925),  # the MD binodal at 7000 K, x_He
        "trust_domain": _TRUST.inner + _TRUST.outer,  # the rollout's reading of trust_domain above
        "gate_tol": 0.05,
        "eos_pressures_GPa": (800.0, 200.0, 400.0, 600.0),  # hparams["eos_pressures"]; do not sort
        # Field trees per role, in eos_pressures order; fdt_<P>GPa holds runs and tables.
        "modes_trees": ("modes_800GPa", "modes_200GPa", "modes_400GPa",
                        "modes_600GPa"),
        "table_trees": ("fdt_800GPa", "fdt_200GPa", "fdt_400GPa",
                        "fdt_600GPa"),
        "eos_trees": ("eos_800GPa", "eos_200GPa", "eos_400GPa",
                      "eos_600GPa"),
    },
    # Engine knobs per campaign: trajectory (production) and equation-of-state grid (eos).
    md_settings={
        "production": MdSettings(
            units="metal",
            skin=2.0,
            neigh_modify="every 1 delay 0 check yes",
            thermo_every=200,
            T_damp=0.01,
            P_damp=0.1,
            pressure_scale=1e4,  # bar per GPa
        ),
        "eos": MdSettings(
            units="metal",
            skin=2.0,
            neigh_modify="every 1 delay 0 check yes",
            thermo_every=100,
            T_damp=0.02,
            P_damp=0.2,
            pressure_scale=1e4,
        ),
    },
    farm_partition=("P_GPa",),  # EOS pressure is a controlled variable
    defaults={
        # -- aipf md run: the potential the archive was generated with, by md5 (its file, the cueq
        # fp32 ML-IAP conversion, is a site fact: Site.for_system)
        "md": {"potential": {"md5": "2ed3c8297221761a915e212408e803b4"}},
        # -- the rollouts: aipf.rollout's solver, archive and driver declarations, as the published
        # rollouts ran them
        "rollout": {
            "solver": {
                "clamp_rho": 1e-3,  # floor under the density mu and M are evaluated at
                "state_proj": "domain",  # project the state onto the trust trapezoid
                "state_clamp": 1e-3,
                "mass_restore": "shift",  # k=0 written back
                "noise_eval": "ito",
                "noise_scale": 1.0,
                "predictor_floor": 1e-4,
            },
            "archive": {  # the mode archive's key spellings
                "file_name": "modes.npz",
                "amplitudes": "rho_k",
                "amplitudes_channel_axis": 2,  # (..., modes, 2)
                "labels": "nvec",
                "box": "box",
                "temperature": "T_K",
                "frame_interval": "dt_frame_ps",
                "quality_file": "DATA_QUALITY_WARNING.json",
                "quality_key": "mask_end",
            },
            "spinodal": {
                "modes_tree": "fields/modes_800GPa",
                "grid": (24, 24, 24),
                "field_stride": 10,
            },
            "slab": {
                "modes_tree": "fields/modes_800GPa",
                "grid": (16, 16, 64),
                "field_stride": 10,
                "profile_axis": 2,  # the profile c(z), averaged over x and y
            },
        },
        # -- the rain column, experiments/hhe/column.py
        "column": {
            # The noisy half's convention, which every published rain-zone chain ran.
            "noise": Noise(mode="gaussian", m_stab="mean"),
            "masses": (1.008, 4.003),  # amu, the column's gravity masses (not masses["He"] 4.0026)
            "T_window": (5000.0, 12000.0),  # K, the trained temperature window
            "T_margin": 500.0,  # K
            "ramp_min_cells": 8.0,
            "quarantine_thresh": 0.5,
            "seed_chunk_stride": 7919,
            "rain_rule": "mean_ref_v2",
            "solver": {
                "clamp_rho": 1e-3,
                "state_proj": "domain",
                "state_clamp": 1e-3,
                "mass_restore": "shift",  # k=0 written back
                "noise_eval": "ito",
                "noise_scale": 1.0,
                "predictor_floor": 1e-4,
            },
        },
        # k_cut is baked into modes.npz; k_max is the drift band mask. Different knobs.
        "sigma": 2.0,        # coarse-graining width, Angstrom
        "k_cut": 3.0,        # mode-extraction cutoff, inverse Angstrom
        "mode_fields": "per_type",  # modes.npz rho_k: one channel per species
        "k_max": None,       # drift band; None -> 4 / sigma
        "k_fit_stat": 1.5,   # inverse Angstrom
        "R_cut": 4.5,        # kernel range, Angstrom
        "rung": 3,
        "f_form": "mlp",  # mk["g_form"]
        "kernel_form": "radial_mlp",
        "T_basis": ("1", "T", "TlnT"),
        "fexc_T_ref": 8000.0,
        "tbasis_ortho": True,
        "h_u": 16,
        "h_g": 32,
        "h_w": 16,
        "h_m": 16,
        "rho_ref": (0.35, 0.33),
        "m_form": "mlp_scaled",
        "m_T_form": "none",      # c(T) = 1
        "T_ref": 10000.0,  # inert at m_T_form="none"
        # -- the drift residual
        "alpha_loss": 0.0,       # pure H^-1
        "h_inv_eps": 1e-6,  # r2 / (k2 + eps)
        "source_loss_weights": {
            "slab_800": 1.0, "cube_800": 1.0,
            "slab_200": 1.0, "cube_200": 1.0,
            "slab_400": 1.0, "cube_400": 1.0,
            "slab_600": 1.0, "cube_600": 1.0,
        },
        "estimator": "weak",
        "lambda_M": 0.5,
        "lambda_S": 0.01,
        "lambda_bulk": 0.01,
        "lambda_P": 5.0,
        "lambda_conv": 1.0,
        "lambda_gamma": 1.0,
        "stat_metric": "rel_frob",  # hparams["stat_metric"]
        "conv_penalty": "hinge",
        "conv_margin": 0.05,
        "conv_samples": 1024,
        "gamma_pt_samples": 16,
        # -- the optimiser
        "lr": 5e-4,
        "weight_decay": 0.0,
        "warmup_epochs": 5,
        "anneal_epochs": 30,
        "max_epochs": 30,
        "wd_ghat": 0.01,  # kernel head decay, not weight_decay
        # -- the diagnosis
        "diagnose": {
            "eos_csvs": dict(_EOS_MANIFOLDS),
            "eos_columns": dict(_EOS_COLUMNS),
            "poly_degree": 4,  # a fit, not an interpolation
            "manifold_fit": "column",  # blend two T columns, then fit
            "pressure_unit": _GPA_IN_MODEL_PRESSURE,  # eV/A^3 per GPa
            "x_grid": (0.02, 0.98, 0.005),
            "iso_x_grid": (0.05, 0.95, 0.05),
            "x_bar": 0.5,
            "isobar": {"s_lo": 0.30, "s_hi": 1.30, "n_scan": 161,
                        "n_iter": 45,
                        "pressure_route": "protocol"},  # the route the dome was matched on
            "min_path_points": 8,
            "binodal_route": "convex_hull",  # lower convex hull of g(x)
            "min_tie_gap": 0.015,
            "central_lo_max": 0.6,
            "central_width_max": 0.45,
            "tc_method": "dome_apex",
            "apex": {"central_lo_max": 0.6, "tail_fraction": 0.5,
                      "min_tail_points": 4, "max_overshoot_steps": 10.0},
            "dome": {
                "pressures": (200.0, 800.0, 50.0),
                "T_grid": (2000.0, 11000.0, 100.0),
                "anchor_pressures": (200.0, 400.0, 600.0, 800.0),
                "x_grid": (0.02, 0.98, 0.005),
                "close_after": 2,
            },
        },
        # -- aipf.train.fit's own vocabulary for the loader and the anchors
        "training": {
            "source_root": {"tier": "raw", "path": "fields"},  # <raw>/fields/...
            "half_width": 25,      # frames
            "n_states": 5,
            "stride": 10,          # frames between window centres
            "savgol_window": None,  # the weak estimator reads no taps
            "savgol_poly": None,
            "run_weighting": "inverse_band_power",
            "run_weight_probe_every": 100,  # 10 * stride
            "val_split": "random",
            "val_fraction": 0.1,
            "split_seed": 316,
            "val_labels": ((0.1, 0.9),),  # inert under a random split
            "batch_size": 16,
            "num_workers": 0,  # throughput only
            "pin_memory": False,
            "drop_last": False,  # DataLoader default
            "order": "shuffled",
            "lambda_dyn": 1.0,  # the drift weight, not saved: 1
            "bulk_residual": "relative_inverse",
            "eta_min": 1e-6,  # hparams["eta_min"]
            "grad_clip": 1.0,  # the Trainer's
            # The two hinge penalties draw from one stream seeded here.
            "penalty_seed": 2,
            "conv_T_measure": "log_uniform",  # T ~ 1/T on trust_domain.T_range
            "gamma_paths": {
                "eos_csvs": dict(_EOS_MANIFOLDS),  # hparams["eos_csvs"]; order = manifold index
                "eos_columns": dict(_EOS_COLUMNS),
                "x_grid": (0.11, 0.89, 0.02),  # x_lo, x_hi, h
                "poly_degree": 4,
                "T_measure": "uniform",
                "q_floor": 1e-6,
            },
            # AnchorTables.load(**tables): relative paths keyed by pressure, published model order.
            "tables": {
                "key": "pressure",  # one fdt_<P>GPa file per pressure
                # H0 at k = 0 through the kernel's own k path
                "w0": {"route": "evaluator"},
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
                "eos_csvs": dict(_EOS_MANIFOLDS),  # the diagnosis's four manifolds
                "pressure_unit": _GPA_IN_MODEL_PRESSURE,
                "eos_columns": dict(_EOS_COLUMNS),
                "row_weight": None,  # every row the same
                "pressure_floor": None,  # the pure relative form
                "columns": {  # np.load(...).files of the tables above
                    "phase": "single_phase",
                    "temperature": "T_K",
                    "densities": ("rho_H", "rho_He"),
                    "composition": "x_He",
                    "mobility": "M_kappa0",
                    "shells": "k_shells",
                    "structure": "S_k",
                    "bulk_target": "Scc0",
                    "bulk_sigma": "sigma_OZ",
                    "bulk_ok": "oz_ok",
                },
            },
        },
    },
)
