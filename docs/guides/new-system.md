# A new system

A system is one file, `experiments/<name>/system.py`, that defines `SYSTEM`, an
`aipf.system.System`. `aipf --system <name>` finds it under `experiments/` (or `AIPF_EXPERIMENTS`,
or `[paths] experiments` in `aipf.toml`); `--system <folder>` with a path also works. Nothing in a
declaration is defaulted on the system's behalf: a command that needs a key the system does not
declare refuses, naming the key and the block it belongs in. Start from the closest shipped system:
`experiments/lj/system.py` (one field, a pair-potential MD deck, two baseline variants) or
`experiments/feb/system.py` and `experiments/hhe/system.py` (two fields, a machine-learned
potential, anchor tables, a trust domain; H/He also rollouts and noise).

## A minimal system

This one is complete enough to train and diagnose on the modes that
[workflow.md](workflow.md) makes in its sections 1 to 3:

```python
import math

from aipf.paths import Paths
from aipf.system import AnchorRules, Functional, Mobility, Noise, System

TRAINING = {
    "source_root": {"tier": "farm", "path": "modes"},
    "half_width": 25, "n_states": 5, "stride": 20, "savgol_window": None, "savgol_poly": None,
    "run_weighting": "uniform",
    "val_split": "none", "val_labels": (), "val_fraction": 0.0, "split_seed": 0,
    "batch_size": 16, "num_workers": 0, "pin_memory": False, "drop_last": True, "order": "shuffled",
    "grad_clip": 1.0,
}

SYSTEM = System(
    name="toy",
    n_species=1,
    species=("A",),
    masses={"A": 1.0},
    atom_types={"A": 1},
    table_keys={"rho": ("rho_A",), "x": "x_A", "x_channel": 0},
    paths=Paths(system="toy"),            # raw root from AIPF_RAW_TOY or [paths.raw] toy
    anchor_rules=AnchorRules(T_min_by_pressure={}),
    functional=Functional(form="square_gradient", local="flory_huggins", kernel=None, kwargs=dict(
        grid=(16, 16, 16), kappa=0.5, T_ref=1.65, rho_eps=1e-4, gamma_fixed=None,
        nyquist_mask=False, w=3.0)),
    mobility=Mobility(form="lattice_scalar", T_form="arrhenius", kwargs=dict(
        mobility_prefactor="mole_fraction", shape_init=math.log(0.1),
        mobility_activation_energy_init=math.log(math.expm1(0.55)))),
    noise=Noise(mode="gaussian", m_stab="mean"),
    constants={"kB": 1.0, "P_GPa": None},
    farm_partition=(),
    defaults={"rung": 2, "sigma": 1.5, "k_cut": 3.0, "estimator": "weak", "h_inv_eps": 1e-6,
              "lr": 5e-4, "weight_decay": 0.0, "warmup_epochs": 5, "anneal_epochs": 25, "eta_min": 1e-6,
              "lambda_dyn": 1.0, "k_max": None, "alpha_loss": 0.0,
              "training": TRAINING,
              "diagnose": {"one_field": {"readoff": "bulk_minima", "T_grid": {"lo": 1.0, "hi": 1.6, "step": 0.02},
                                         "dtype": "float64", "phi_grid": {"lo": 1e-4, "hi": 1 - 1e-4, "n": 2001},
                                         "edge_trim": 5, "centre": 0.5}},
              "mode_fields": ({"name": "A", "weights": {1: 0.5, 2: -0.5}, "mean": 0.5},)},
)
```

With it in `$WORK/newsys/toy/system.py`, and `$WORK/toydata/toy/modes` a link to the modes tier
of the workflow's farm (`$WORK/data/lj/modes`):

```text
$ export AIPF_EXPERIMENTS=$WORK/newsys AIPF_DATA=$WORK/toydata
$ aipf train --system toy --run first --seed 0 --steps 2 \
    --source "cube=.:cube_x0.50_T1.6_s42:16,16,16" \
    --resume-optimizer no --anchors none
$WORK/toydata/toy/ckpt/first
$ aipf diagnose --system toy --ckpt $AIPF_DATA/toy/ckpt/first/final.ckpt \
    --stage one_field_phase_diagram
$WORK/toydata/toy/diagnose/<md5[:12]>
```

Two steps on the CPU take a few seconds. A new system also needs its raw root, `AIPF_RAW_<NAME>` or
`<name> = "..."` under `[paths.raw]` in `aipf.toml`, for `aipf data build` and for every source or
table declared under the raw tier.

## The fields of System

| field | admissible values |
|---|---|
| `name` | a string; the farm directory and the suffix of `AIPF_RAW_<NAME>` |
| `n_species` | an int >= 1 |
| `species` | `n_species` distinct names, in channel order |
| `masses` | `{species: mass}`, exactly the species |
| `atom_types` | `{species: dump type}`, covering the species (extra types allowed) |
| `table_keys` | `{"rho": (one column per channel), "x": "<composition column>", "x_channel": <channel index>}` |
| `paths` | `Paths(system=name)`, optionally `raw_default="..."` |
| `anchor_rules` | `AnchorRules(T_min_by_pressure={P: T}, shape_T_min_by_pressure={P: T})`, numbers, either may be empty |
| `constants` | a dict; `kB` (required for an Arrhenius mobility, anchors and rollouts), `P_GPa` (or `None`), optionally `md_exclude_dirs`, `modes_trees`, `fdt_runs_trees`, `table_trees` |
| `defaults` | a dict, the blocks below |
| `farm_partition` | `("P_GPa",)` (default) when pressure is a controlled variable, `()` otherwise |
| `traj_names` | trajectory file names, default `("traj.lammpstrj", "traj.dump", "dump.lammpstrj")` |
| `tag_from_path`, `tag_path_root` | `False`, `None` (default): a run's tag is its directory name |
| `preparation` | `PreparationRules(melt_T=...)` or `PreparationRules(ramp_offset=...)` or neither |
| `checkpoint` | `Checkpoint(md5="<32 hex>")` once a model is published at `data/<name>/ckpt/published/final.ckpt`; `None` before |
| `functional` | `Functional(form, local, kernel, kwargs)`: `square_gradient` with `local` `flory_huggins` or `landau`, or `nonlocal_kernel` with `local` `mlp`, `icnn` or `taylor` and `kernel="radial_mlp"` |
| `mobility` | `Mobility(form, T_form, kwargs)`: `form` one of `mlp_scaled`, `mlp`, `mlp_rho_t`, `constant`, `gamma`, `fixed`, `lattice_scalar`; `T_form` `none` or `arrhenius` |
| `md_settings` | `{campaign: MdSettings(units, skin, neigh_modify, thermo_every, T_damp, P_damp, pressure_scale)}`, or `{}` |
| `noise` | `Noise(mode="gaussian" or "none", m_stab="mean" or "max")`, or `None` |
| `trust_domain` | `TrustDomain(inner=(...), outer=(...), T_range=(T_min, T_max) or None)`, one positive intercept per channel, or `None` |
| `variants` | `{name: Variant(functional, mobility, checkpoint, defaults)}`, or `{}` |
| `variant_name` | never declared: `System.variant(name)` sets it on the system it returns; `None` is the production model |

The meaning of each is in [../reference/system.md](../reference/system.md); every constructor
argument of the functional and the mobility, with its admissible values, in
[../reference/functional.md](../reference/functional.md) and
[../reference/mobility.md](../reference/mobility.md).

## The blocks of defaults

| block | needed by | keys |
|---|---|---|
| top level | `aipf train`, `aipf rollout`, `aipf modes` | `sigma`, `estimator` (`weak`, `weak_mid`, `savgol`), `h_inv_eps`, `k_max`, `mode_fields`, the `TrainConfig` fields ([../reference/training.md](../reference/training.md#the-optimiser-and-the-loss-weights)) |
| `training` | `aipf train` | `source_root`, `half_width`, `n_states`, `stride`, `savgol_window`, `savgol_poly`, `run_weighting`, `val_split`, `val_labels`, `val_fraction`, `split_seed`, `batch_size`, `num_workers`, `pin_memory`, `drop_last`, `order`; optionally `band_k_max`, `sources`, `sample`, `tables`, `grad_clip`, `penalty_seed`, `conv_T_measure`, `gamma_paths` |
| `diagnose` | `aipf diagnose` | per stage, [../reference/diagnose.md](../reference/diagnose.md#the-declared-block) |
| `rollout` | `aipf rollout` | `solver`, `archive`, `spinodal`, `slab` ([../reference/rollout.md](../reference/rollout.md#the-declaration)) |
| `md` | `aipf md run` | `potential: {"md5": ...}` for a machine-learned potential, and one `{point, values, dt_equil_ps, campaign}` per deck ([../reference/md.md](../reference/md.md#the-declaration)) |

## What each command reads

| command | reads |
|---|---|
| `aipf md run` | `defaults["md"]`, `preparation`, `masses`, `atom_types`, `md_settings` (with `--campaign`) |
| `aipf data build` | `paths`, `traj_names`, `tag_from_path`, `tag_path_root`, `farm_partition`, `species`, `table_keys`, `constants` (`P_GPa`, `md_exclude_dirs`) |
| `aipf modes` | `paths`, `defaults["mode_fields"]`, `atom_types`, `species`, `table_keys["x"]` |
| `aipf train` | `functional`, `mobility`, `constants["kB"]`, the top-level `defaults` (`estimator`, `sigma`, `h_inv_eps`, `k_fit_stat`, `box`, `source_loss_weights`, the `TrainConfig` fields), `defaults["training"]` (its `tables` give the anchors), `anchor_rules`, `table_keys`, `trust_domain` (the convexity penalty), `checkpoint` and `variants` (with `--init-from-published`, `--variant`), `paths` |
| `aipf diagnose` | `defaults["diagnose"]`, `checkpoint`, `variants`, `functional`, `mobility`, `table_keys`, `constants`, `paths` (the measured manifolds) |
| `aipf rollout` | `defaults["rollout"]`, `defaults["sigma"]`, `noise`, `trust_domain`, `checkpoint`, `variants`, `functional`, `mobility`, `table_keys["x_channel"]`, `constants["kB"]`, `paths` |

The same table is in [../reference/system.md](../reference/system.md#what-reads-what).

## Where the model is defined

- `System.functional` names it: `form` (the rung), `local`, `kernel` and the constructor arguments
  in `kwargs`, in the declaration's own spelling.
- The rungs are `src/aipf/functional/landau.py` (`landau`, `fh`), `square_gradient.py`,
  `nonlocal_kernel.py` and `neural_operator.py`. Each registers its class in `MODEL_REGISTRY`
  (`src/aipf/functional/base.py`). The pieces they share are `local_forms.py` and `kernels.py`.
- `src/aipf/functional/build.py` turns a declaration into constructor arguments. Only
  `nonlocal_kernel` and `square_gradient` have a translation; a form without one is refused. Every
  constructor argument must be declared.
- The mobility is `System.mobility`, built by `src/aipf/mobility.py` (`MOBILITY_SHAPES` lists the
  forms).
- A new functional is a class with the `FreeEnergyModel` methods (`base.py`), registered, plus a
  translation in `build.py` ([../reference/functional.md](../reference/functional.md#adding-a-functional)).
- A model defined in another package is declared with `Functional(..., factory=make_model)`, and
  `build` calls `make_model(system)`. Import the factory from that package in `system.py`; the
  contract it meets is in
  [../reference/functional.md](../reference/functional.md#a-model-defined-elsewhere).

## Publishing a model

Copy a run's `final.ckpt` to `data/<name>/ckpt/published/final.ckpt` (a variant's to
`published/<variant>/final.ckpt`), declare `checkpoint=Checkpoint(md5="...")` with its `md5sum`, and
`--ckpt published` and `--init-from-published` read it, digest-checked, from then on.
