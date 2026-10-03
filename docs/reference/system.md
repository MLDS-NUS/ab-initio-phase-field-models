# The system declaration

A system is one file, `experiments/<name>/system.py`, that defines `SYSTEM`, an
`aipf.system.System`. Everything the general package cannot know about a material lives there:
species, units, where its data are, which functional and mobility it trains, how it trains, how it
is diagnosed and rolled out. Construction checks that the declarations agree with each other; a
command that needs a value the system does not declare refuses, naming the key.

## Finding a system

| function | does |
|---|---|
| `aipf.system.load(name_or_path)` | a name is looked up as `<experiments>/<name>/system.py`; an argument containing a path separator is a folder. Refuses with the declared names when there is no `system.py` (`UnknownSystem`, a `FileNotFoundError`, which every `aipf` command prints as one line and exit 2), when it defines no `SYSTEM`, or when `SYSTEM` is not a `System` |
| `aipf.system.available()` | the sorted names of the experiment folders that hold a `system.py` |
| `aipf.system.experiments_root()` | `AIPF_EXPERIMENTS`, else `[paths] experiments` in `aipf.toml`, else `<repo>/experiments` |

## System

`System` is a frozen dataclass. Fields without a default are required.

| field | type | meaning |
|---|---|---|
| `name` | `str` | the system's name; also its farm directory and the suffix of `AIPF_RAW_<NAME>` |
| `n_species` | `int` | density channels, at least 1 |
| `species` | `tuple[str, ...]` | channel order; `n_species` distinct names |
| `masses` | `dict[str, float]` | exactly the species, in the engine's mass unit |
| `atom_types` | `dict[str, int]` | dump type per species; must cover the species, may carry more types |
| `table_keys` | `dict` | `"rho"`: one column name per channel; `"x"`: the composition column (a string); `"x_channel"`: the channel whose fraction `x` is (an index into `species`) |
| `paths` | `aipf.paths.Paths` | `Paths(system=name, raw_default=None)`; the raw root comes from the environment or `aipf.toml` ([cli.md](cli.md#locations-aipfpaths)) |
| `anchor_rules` | `AnchorRules` | per-pressure temperature cuts of the anchor tables |
| `constants` | `dict` | unit conventions and fixed numbers, below |
| `defaults` | `dict` | every setting of every command, below |
| `farm_partition` | `tuple[str, ...]` | metadata fields that split the farm, `md/<partition>/<tag>`; default `("P_GPa",)`; `()` is unsplit |
| `traj_names` | `tuple[str, ...]` | trajectory file names a run directory is recognised by, in priority order; default `("traj.lammpstrj", "traj.dump", "dump.lammpstrj")` |
| `tag_from_path` | `bool` | `True`: a run's tag is its path below the raw root joined by `_`; default `False` (the leaf directory name) |
| `tag_path_root` | `str \| None` | with `tag_from_path`, a leading path component dropped from the tag |
| `preparation` | `PreparationRules` | how a state point is prepared before production |
| `checkpoint` | `Checkpoint \| None` | the published model's file, by digest |
| `functional` | `Functional \| None` | the free-energy functional ([functional.md](functional.md)) |
| `mobility` | `Mobility \| None` | the mobility ([mobility.md](mobility.md)) |
| `md_settings` | `dict[str, MdSettings]` | engine settings per campaign name |
| `noise` | `Noise \| None` | the conserved noise of a stochastic rollout |
| `trust_domain` | `TrustDomain \| None` | the density region the training data cover |
| `variants` | `Mapping[str, Variant]` | other models of the same system, by name |
| `variant_name` | `str \| None` | set by `variant()`; `None` is the production model |

`None` for `checkpoint`, `functional`, `mobility`, `noise` and `trust_domain` means undeclared, and
is never replaced by a default: the command that needs the value refuses.

Methods:

| method | returns |
|---|---|
| `variant(name)` | this system with the variant's functional, mobility, checkpoint and `defaults` entries; `KeyError` listing the declared ones |
| `channel_of(species)` | the channel index of a species |
| `md(campaign)` | the `MdSettings` of one campaign; `KeyError` naming the declared ones |
| `t_min(P)`, `shape_t_min(P)` | the two anchor cuts at pressure `P`, or `None` |
| `checkpoint_file()` | where the checkpoint is expected (not checked) |
| `verify_checkpoint()` | that file, its digest checked |
| `resolve_checkpoint()` | the checkpoint wherever it is held, digest checked |

## Checkpoint

    Checkpoint(md5, path=None)

`md5` is 32 lowercase hexadecimal digits, as `md5sum` prints them. A system's own checkpoint
declares the digest alone: its file is the tracked `final.ckpt` of `data/<system>/ckpt/published/`,
or of `published/<variant>/` for a variant (`aipf.paths.published_checkpoint(system, variant)`,
`Checkpoint.tracked_under(published, variant)`). Any other file, for instance a run's own
checkpoint used as a starting point, also gives `path`, relative to the raw root (an absolute or
empty path is refused).

`resolve(system, raw, published, variant)` tries the tracked copy first and then, when `path` is
declared, the raw root; the bytes are digest-checked either way. A missing file is
`FileNotFoundError`, a file whose digest differs is `ValueError`, and the message names the system
and every location tried. `verify_under(root)` checks a declared `path` under one root.

`aipf diagnose --ckpt published`, `aipf rollout --ckpt published` and
`aipf train --init-from-published` all go through `System.resolve_checkpoint`.

## Functional and Mobility

    Functional(form, local, kernel, kwargs)
    Mobility(form, T_form, kwargs)

`form` is one of `landau`, `square_gradient`, `nonlocal_kernel`, `neural_operator`. `kernel` is
required for `nonlocal_kernel` and must be `None` otherwise. `kwargs` are the constructor arguments
in the declaration's spelling; every argument the constructor takes must be declared. The
admissible values are listed in [functional.md](functional.md) and [mobility.md](mobility.md).

## Variant

    Variant(functional, mobility, checkpoint, defaults)

A second model of the same system, sharing its constants, paths and data. `checkpoint` is a
`Checkpoint` or `None`; `defaults` replaces top-level keys of the system's `defaults`. A variant's
name is one directory name: its tracked checkpoint sits at
`data/<system>/ckpt/published/<name>/final.ckpt`. The reduced-unit system declares two, `fh` and
`landau`, the Flory-Huggins and Landau square-gradient baselines.

## TrustDomain

    TrustDomain(inner, outer, T_range=None)

The density trapezoid `rho_i >= 0`, `sum_i rho_i / inner_i >= 1`, `sum_i rho_i / outer_i <= 1`, with
one strictly positive intercept per channel on each edge, in channel order, and an optional
`(T_min, T_max)`. Training draws the convexity penalty's probe points from it, and the rollout
projects its state onto it. A declared domain has one intercept per channel of the system.

## Noise

    Noise(mode, m_stab)

| field | admissible | meaning |
|---|---|---|
| `mode` | `"gaussian"`, `"none"` | `gaussian` filters the conserved noise by `exp(-k^2 sigma^2 / 2)` with the system's `defaults["sigma"]`; `none` leaves it white |
| `m_stab` | `"mean"`, `"max"` | the mobility the semi-implicit operator is frozen at: `M` at the mean density, or at the initial field's largest-norm cell |

## AnchorRules, PreparationRules, MdSettings

    AnchorRules(T_min_by_pressure={}, shape_T_min_by_pressure={})

Two tables from pressure (GPa) to temperature (K): the lowest admissible anchor temperature, and the
lowest temperature that enters the kernel-shape set. Keys and values must be numbers.

    PreparationRules(melt_T=None, ramp_offset=None)

How `aipf md run` prepares a state point: hold at an absolute `melt_T`, or at the target plus
`ramp_offset`, in the unit of the request's temperature. Declaring both is refused.

    MdSettings(units, skin, neigh_modify, thermo_every, T_damp, P_damp, pressure_scale)

The engine settings of one campaign: the LAMMPS unit system (`"metal"`, `"lj"`, ...), the neighbour
skin, the `neigh_modify` arguments (non-empty), thermo output every `thermo_every >= 1` steps, the
thermostat and barostat damping in the engine's time unit, and `pressure_scale`, the factor from a
request's pressure (GPa) to the deck's unit (`1e4` for bar). Every number must be positive.

## constants

| key | read by | meaning |
|---|---|---|
| `kB` | the Arrhenius mobility, the anchors, the penalties, the rollout noise | the Boltzmann constant in the system's energy unit per temperature unit (`8.617333262e-5` eV/K, or `1.0` reduced) |
| `P_GPa` | `aipf data build` | the pressure recorded for a run that records none; `None` when pressure is not a variable |
| `md_exclude_dirs` | `aipf data build` | top-level raw subdirectories whose runs are skipped |
| `modes_trees`, `fdt_runs_trees`, `table_trees` | `aipf.data.index.index_fields` | the role of each tree under `<raw>/fields/` |

Other keys are documentation of the system and are not read.

## defaults

| key | read by | meaning |
|---|---|---|
| `sigma` | training, rollouts, noise | the coarse-graining width, in the system's length unit |
| `k_cut` | nothing: a record | the mode-extraction cutoff the system's archives were made with; `aipf modes` takes it as the required `--k-cut` |
| `mode_fields` | `aipf modes` | `"per_type"` (one channel per species) or a tuple of combinations, [data.md](data.md#modes) |
| `estimator` | training | `"weak"`, `"weak_mid"` or `"savgol"`, [training.md](training.md#windows) |
| `h_inv_eps` | training | the regulariser of the drift residual's `1/(k^2 + eps)` |
| `k_max` | training | the drift band `k <= k_max`; `None` is `4 / sigma` |
| `k_fit_stat` | pressure-keyed anchors | the largest shell wavenumber of the structure-factor anchor |
| `box` | the lattice-sum kernel's `W(0)` | a fixed box, for a system whose ensemble never varies it |
| `source_loss_weights` | training | the drift weight of each named source (default 1.0) |
| `lr`, `weight_decay`, `wd_ghat`, `warmup_epochs`, `anneal_epochs`, `eta_min`, `alpha_loss`, `lambda_*`, ... | training | the `TrainConfig` fields, [training.md](training.md#the-optimiser-and-the-loss-weights) |
| `training` | `aipf train` | the loader, the windows, the split, the sources, the anchor tables, the penalties |
| `diagnose` | `aipf diagnose` | the thresholds and grids of every stage, [diagnose.md](diagnose.md) |
| `rollout` | `aipf rollout` | the solver, the archive keys and each driver's grid, [rollout.md](rollout.md) |
| `md` | `aipf md run` | `potential: {"md5": ...}` and one declaration per built-in deck, [md.md](md.md) |

Keys not listed here (`rung`, the published configuration's own names such as `f_form` or
`m_form`) record the published model and are not read. A `TrainConfig` field in
`defaults["training"]` wins over the same field at the top level.

## What reads what

| command | reads |
|---|---|
| `aipf md run` | `defaults["md"]`, `preparation`, `masses`, `atom_types`, `md_settings` (with `--campaign`) |
| `aipf data build` | `paths`, `traj_names`, `tag_from_path`, `tag_path_root`, `farm_partition`, `species`, `table_keys`, `constants` (`P_GPa`, `md_exclude_dirs`) |
| `aipf modes` | `paths`, `defaults["mode_fields"]`, `atom_types`, `species`, `table_keys["x"]` |
| `aipf train` | `functional`, `mobility`, `constants["kB"]`, the top-level `defaults` (`estimator`, `sigma`, `h_inv_eps`, `k_fit_stat`, `box`, `source_loss_weights`, the `TrainConfig` fields), `defaults["training"]` (its `tables` give the anchors), `anchor_rules`, `table_keys`, `trust_domain` (the convexity penalty), `checkpoint` and `variants` (with `--init-from-published`, `--variant`), `paths` |
| `aipf diagnose` | `defaults["diagnose"]`, `checkpoint`, `variants`, `functional`, `mobility`, `table_keys`, `constants`, `paths` (the measured manifolds) |
| `aipf rollout` | `defaults["rollout"]`, `defaults["sigma"]`, `noise`, `trust_domain`, `checkpoint`, `variants`, `functional`, `mobility`, `table_keys["x_channel"]`, `constants["kB"]`, `paths` |

[../guides/new-system.md](../guides/new-system.md) builds a system from these pieces.
