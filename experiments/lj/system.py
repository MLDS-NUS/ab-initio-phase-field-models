"""Binary Lennard-Jones, the validation system, in reduced units.

One concentration field phi = (rho_A - rho_B) / (2 rho_bar V), shifted so the
equimolar state sits at 0.5; overdamped Langevin, fixed box, no MLIP.
Every value reproduces the published model, whose tracked checkpoint carries its training
configuration as ``hyper_parameters`` (``hparams``; ``mk`` = its ``model_kwargs``); a constructor
default the checkpoint does not save is marked as such.
"""
import math

from aipf.paths import Paths
from aipf.system import (AnchorRules, Checkpoint, Functional, Mobility,
                         Noise, PreparationRules, System, Variant)

_T_REF = 1.65  # mk["T_ref"], the Arrhenius reference

#: The published homogeneous overdamped deck, filled by `aipf md run`.
_MD_CUBE_OVERDAMPED = {
    "point": {"geometry": "cube", "ensemble": "langevin_overdamped",
              "T": 1.60,
              "x": 0.5,
              "dt_ps": 0.0002,  # the overdamped production step
              "equil_ps": 250.0,  # 250000 x 0.001
              "prod_ps": 5000.0,  # 25000000 x 2e-4
              "dump_every_ps": 1.0,  # 5000 x 2e-4
              "dump_from": "prod",  # dumps start with stage 2
              "seed": 42,
              "P": None,  # no barostat in the deck
              "n_atoms": 4000},  # 10^3 FCC cells x 4
    "dt_equil_ps": 0.001,
    "values": {
        "UNITS": "lj",
        "CONFIGURATION": ("lattice         fcc 1.0\n"
                          "region          box block 0 10 0 10 0 10\n"
                          "create_box      2 box\n"
                          "create_atoms    1 box\n"
                          "set             group all type/fraction 2 0.5 42"),
        "MASSES": "mass            1 1.0\nmass            2 1.0",  # type 2 is no species here
        "PAIR": ("pair_style      lj/smooth/linear 2.5\n"
                 "pair_coeff      1 1 1.0 1.0\n"
                 "pair_coeff      2 2 1.0 1.0\n"
                 "pair_coeff      1 2 0.5 1.0"),
        "SKIN": "0.5",
        "NEIGH_MODIFY": "every 5 delay 0 check yes",
        "RELAX": "minimize        1.0e-6 1.0e-8 1000 10000",
        "N_THERMO": "10000",
        "DAMP": "0.2",
        "GAMMA": "2.0",
        "TRAJECTORY": "dump.lammpstrj",
        "WRITE_END": "",  # the deck writes no end state
    },
}


_TRAINING = {
    # the tier `aipf modes` writes (index.modes_dir)
    "source_root": {"tier": "farm", "path": "modes"},
    "modes_root": "Data/slab_overdamped/modes",  # under the raw root
    # the bundled sample, data/lj/sample: four runs, one per temperature, each cut to the four
    # windows that fill one batch (drop_last drops a partial one) and to the modes with max |n_i| <= 2.
    # `aipf train --source sample` trains on it with no raw root; it runs the pipeline and carries no
    # physics. The root is relative to data/lj/sample.
    "sample": {"slab": {"root": "modes", "pattern": "slab_overdamped_*", "grid": (20, 20, 80)}},
    "train_temperatures": (1.10, 1.15, 1.25, 1.30, 1.35, 1.40,
                           1.45, 1.50, 1.60, 1.70),
    "val_seeds": (45,),
    "test_temperatures": (1.20,),
    "half_width": 25,  # frames
    "n_states": 5,
    "stride": 20,  # frames between window centres
    "estimator": "weak",
    "batch_size": 16,
    "order": "shuffled",
    "drop_last": True,
    "grad_clip": 1.0,  # the Trainer's gradient_clip_val
    "eta_min": 1e-6,  # hparams["eta_min"]
    "lambda_dyn": 1.0,  # hparams["lambda_drift"]
    # the saved "S" weight is H(0;T) against S_cc(0): this package's L_bulk
    "lambda_bulk": 0.01,  # hparams["lambda_S"]
    "lambda_S": 0.0,  # no finite-k shell term
    "bulk_residual": "sigma_chi2",  # ((H0/T - rho_bar/S)/sigma)^2
    # this package's spelling of the loader
    "val_split": "none",  # the sources are the train runs; the validation seed 45 is not trained on
    "val_labels": (), "val_fraction": 0.0, "split_seed": 0,  # unread under "none"
    "run_weighting": "inverse_band_power",  # 1/(m + median m), mean one
    "run_weight_probe_every": 317,  # the stride nearest 64 evenly spaced probes
    "savgol_window": None, "savgol_poly": None,  # the weak estimator has no taps
    "num_workers": 4,
    "pin_memory": True,
    # one row per temperature at phi = 0.5
    "tables": {
        "key": "temperature",
        "m_table": "FDT_M/results/M_meso_homogeneous.npz",  # hparams["m_table"]; tracked in anchors/, so no raw root is needed
        "s_table": "Tc_structure_factor/results/Scc_reciprocal_from_md.npz",  # hparams["s_table"]; tracked in anchors/
        "state": (0.5,),  # M and H(0;T) at phi = 1/2
        "rho_total": 1.0,  # rho_bar
        # H(0;T) adds W_tilde_0: 4 pi trapz(r^2 W) on 1000 points to R_cut
        "w0": {"route": "radial", "r_max": 3.0, "n_points": 1000},
        "columns": {"temperature": "T", "mobility": "M_mean",
                    "records": "records", "record_temperature": "T",
                    "structure_zero": "Scc0", "temperatures": "TS"},
    },
}


#: The rung-2 baselines' bulk anchor: H(0;T) = f''(1/2) with no kernel term (the closed form's
#: bulk_curvature); the square gradient's kappa k^2 is zero at k = 0.
_TRAINING_BASELINE = {**_TRAINING, "tables": {**_TRAINING["tables"], "w0": {"route": "evaluator"}}}

#: The baselines' phase diagram: the minima of the closed form's f and the sign of f'' on a grid.
_DIAGNOSE_BASELINE = {
    "one_field": {
        "readoff": "bulk_minima",  # argmin of f, f'' by np.gradient
        "T_grid": {"lo": 1.00, "hi": 1.60, "step": 0.02},  # np.arange(lo, hi + 1e-9, step)
        "dtype": "float64",
        "phi_grid": {"lo": 1e-4, "hi": 1 - 1e-4, "n": 2001},
        "edge_trim": 5,  # f''[5:-5]
        "centre": 0.5,  # phi < 0.5, b_hi = 1 - b_lo
    },
}


def _baseline(local: str, bulk: dict, md5: str) -> Variant:
    """One rung-2 baseline (Flory-Huggins or Landau square gradient), trained with the production
    training block. ``mk`` = its checkpoint's hparams["model_kwargs"]."""
    return Variant(
        functional=Functional(form="square_gradient", local=local, kernel=None, kwargs=dict(
            grid=(20, 20, 80),  # hparams["grid"]
            kappa=0.5,  # mk["kappa"]; _log_kappa = log(kappa)
            T_ref=_T_REF,  # mk["T_ref"]
            rho_eps=1e-4,  # constructor default; not saved
            gamma_fixed=None,  # mk["gamma_fixed"]: gamma is trained
            nyquist_mask=False,  # plain i k on every axis
            **bulk)),
        # M = gamma r (1 - r) c(T)
        mobility=Mobility(
            form="lattice_scalar", T_form="arrhenius",  # mk["m_T_form"]
            kwargs=dict(
                mobility_prefactor="mole_fraction",  # rho_s * (1 - rho_s)
                shape_init=math.log(0.1),  # mk["gamma"], stored as its log
                mobility_activation_energy_init=math.log(math.expm1(0.55)),  # mk["Ea_init"], softplus^-1
            ),
        ),
        checkpoint=Checkpoint(md5=md5),
        defaults={"rung": 2,  # the square-gradient rung
                  "training": _TRAINING_BASELINE, "diagnose": _DIAGNOSE_BASELINE},
    )


#: The two baselines the paper's Fig. 3 compares, seed s1 (epoch 24).
_VARIANTS = {
    "fh": _baseline("flory_huggins", dict(w=3.0),  # mk["w"]
                    "095301c85c8c6530abc9aa8b3d298c76"),
    "landau": _baseline("landau", dict(a0=1.0, b=1.0, T_c=1.5),  # mk["a0"], mk["b"], mk["T_c"]
                        "456db79a1a30d879f8ce4d98c768972f"),
}


SYSTEM = System(
    name="lj",
    n_species=1,
    species=("A",),
    masses={"A": 1.0},  # reduced units
    atom_types={"A": 1},  # B has no channel of its own: phi is A against B
    table_keys={"rho": ("rho_A",), "x": "x_A", "x_channel": 0},
    paths=Paths(system="lj"),  # raw root: AIPF_RAW_LJ, AIPF_RAW or [paths.raw] in aipf.toml
    checkpoint=Checkpoint(md5="ef9da0b40952f7381f0ebaba876fdd5c"),
    # The one-field lattice form.
    functional=Functional(
        form="nonlocal_kernel", local="taylor", kernel="radial_mlp",  # hparams["model_type"], mk["f_form"]
        kwargs=dict(
            grid=(20, 20, 80),  # hparams["grid"]
            R_cut=3.0,  # mk["R_cut"]
            rho_ref=(0.5,),  # psi = rho - 0.5, the kernel's centre
            h_g=32,  # mk["g_exc_hidden"]
            h_w=16,  # constructor default; not saved
            T_ref=_T_REF,  # no h_u, no fexc_T_ref: taylor energy, no T-basis head
            rho_eps=1e-4,  # constructor default; not saved
            activation="gelu",  # mk["activation"]
            g_form="icnn",  # constructor default
            enable_TlnT=False, enable_T2=False,  # constructor defaults
            h_g_hat=None, h_g_tilde=None,  # no temperature-basis head to size
            gauge_fix=False,  # mk["gauge_fix_e"]
            disable_g=False,  # mk["disable_g_exc"]
            f_exc_form="split",  # constructor default
            ideal_form="lattice",  # mu_id = log(rho) - log1p(-rho)
            local_input_scale=False,  # the nets read phi itself
            u_degree=8,  # mk["taylor_max_order"]
            u_parity="even",  # constructor default
            u_variable="difference",  # powers of rho - 0.5
            g_symmetry="mirror",  # constructor default: g_exc symmetrised
            icnn_output_bias=False,  # w2 = Linear(1, 1, bias=False)
            kernel_argument="difference",  # W convolves rho - 0.5
            kernel_evaluator="lattice_sum",  # W on the grid's nearest-image distances, dV rfftn
            nyquist_mask=False,  # plain i k on every axis
        ),
    ),
    # M = r (1 - r) exp(log_gamma) c(T)
    mobility=Mobility(
        form="lattice_scalar", T_form="arrhenius",  # mk["m_form"] "gamma"; mk["m_T_form"]
        kwargs=dict(
            mobility_prefactor="mole_fraction",  # rho_s * (1 - rho_s)
            shape_init=math.log(0.1),  # mk["gamma"], stored as its log
            mobility_activation_energy_init=math.log(math.exp(0.55) - 1.0),  # mk["Ea_init"], softplus^-1
        ),
    ),
    preparation=PreparationRules(
        melt_T=2.0,  # reduced units
    ),
    # Conserved noise filtered at the coarse-graining sigma (1.5); the published stochastic rollouts froze
    # the operator at the largest mobility over the domain.
    noise=Noise(mode="gaussian", m_stab="max"),
    # No trust domain: the loss is drift + M + S only, and the solver's [1e-4, 1 - 1e-4] is a clamp,
    # not a domain.
    trust_domain=None,
    anchor_rules=AnchorRules(T_min_by_pressure={}),  # no pressure anchor, no T cut
    variants=_VARIANTS,  # the Flory-Huggins and Landau baselines
    constants={
        "kB": 1.0,  # reduced units
        "P_GPa": None,              # not a controlled variable here
        "box_L": 15.874010519681994,   # units of sigma_AA
        "n_atoms": 4000,
        "rho_total": 1.0,
        "x_A": 0.5,  # Scc_reciprocal_from_md.npz meta
        "ensemble": "langevin_overdamped",
    },
    farm_partition=(),  # P not controlled: layout md/<tag>
    traj_names=("dump.lammpstrj", "dump.seed.lammpstrj",
                "dump.quench.lammpstrj", "dump.parent.lammpstrj"),
    tag_from_path=True,  # no meta.json
    tag_path_root="Data",
    defaults={
        "rung": 3,
        "sigma": 1.5,               # units of sigma_AA
        "k_cut": 3.0,               # units of 1 / sigma_AA; from modes.npz
        # phi = 0.5 (rho_A - rho_B) / rho_bar, zero mode 0.5; type 1 = A, type 2 = B
        "mode_fields": ({"name": "A", "weights": {1: 0.5, 2: -0.5}, "mean": 0.5},),
        "kernel_form": "radial_mlp",
        "k_max": None,  # None -> 4/sigma = 2.667
        "R_cut": 3.0,
        "f_form": "taylor",
        "taylor_max_order": 8,
        "taylor_min_order": 2,      # c_2 kept: the bulk curvature is data
        "gauge_fix_e": False,
        "g_exc_hidden": 32,
        "m_form": "gamma",
        "gamma": 0.1,               # init -> M(0.5) = 0.025, matching FDT
        "m_T_form": "arrhenius",
        "Ea_init": 0.55,
        "T_ref": 1.65,
        "alpha_loss": 0.0,          # pure H^-1
        "estimator": "weak",
        "epochs": 25,               # fixed early stop
        # -- the published model's saved values
        "grid": (20, 20, 80),       # hparams["grid"]
        "box": (9.524406, 9.524406, 38.097625),  # hparams["box"]
        "lr": 5e-4,                 # hparams["lr"]
        "weight_decay": 0.0,        # hparams["weight_decay"]
        "warmup_epochs": 5,         # hparams["warmup_epochs"]
        "anneal_epochs": 25,        # hparams["anneal_epochs"]
        "eta_min": 1e-6,            # hparams["eta_min"]
        "h_inv_eps": 1e-6,          # hparams["h_inv_eps"]
        "lambda_drift": 1.0,        # hparams["lambda_drift"]
        "lambda_M": 0.01,           # hparams["lambda_M"]
        "lambda_S": 0.01,           # hparams["lambda_S"]
        "m_table": "FDT_M/results/M_meso_homogeneous.npz",  # hparams["m_table"]
        "s_table": "Tc_structure_factor/results/Scc_reciprocal_from_md.npz",  # hparams["s_table"]
        # -- the training data and the loader: not in the checkpoint
        "training": _TRAINING,
        # -- the diagnosis, as the published phase diagram was read off
        "diagnose": {
            "one_field": {
                "readoff": "mu_roots",  # the roots of mu
                "T_grid": {"lo": 0.5, "hi": 2.0, "n": 40},  # the stage's grid, a flag must equal it
                "dtype": "float32",  # the model in float32
                "phi_grid": {"lo": 1e-3, "hi": 1 - 1e-3, "n": 2001},
                "phi_clip": 1e-6,
                "min_width": 0.02,
                "bracket_eps": 1e-6,
                "mu_w0": {"r_max": 4.5, "n_points": 1024},  # 1.5 R_cut on 1024 points
                "tc_w0": {"r_max": 3.0, "n_points": 1000},  # W_tilde_0 over R_cut
                "ising": {"T_max": 1.40, "beta": 0.325,
                          "B0": 1.5, "Tc0_floor": 1.42, "Tc0_offset": 0.05,
                          "B_bounds": (0.1, 5.0), "Tc_bounds_offset": (1e-3, 2.0),
                          "maxfev": 10000},
            },
        },
        # -- `aipf md run`, per built-in deck
        "md": {"cube-overdamped": _MD_CUBE_OVERDAMPED},
    },
)
