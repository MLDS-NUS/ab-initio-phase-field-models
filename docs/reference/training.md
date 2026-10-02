# Training

`aipf train` and `aipf.train.fit.fit` train the functional a system declares on windows of its mode
archives, with the measured anchor tables and the convexity penalties the system declares. The
command line names only the run ([cli.md](cli.md#aipf-train)); everything else is read from
`System.defaults` and `System.defaults["training"]`. A missing key is refused, naming it and the
block it belongs in.

    fit(system, *, run_name, sources, seed, resume_optimizer, steps=None, epochs=None,
        window=None, weighting=None, config_overrides=None, split_mode=None, loader_order=None,
        init_from=None, anchors=None, root=None, log_every_step=False, device="auto",
        deterministic=False) -> Path

`sources` are `SourceSpec` rows (`aipf.train.fit._specs_from(system, table, names)` builds them from
the command line's table). `anchors=None` loads the declared tables, `aipf.train.NO_ANCHORS` trains the
drift term alone. `None` for `window`, `weighting`, `split_mode` and `loader_order` means the
declaration answers. Exactly one of `steps` and `epochs`.

## The run directory

`<data>/<system>/ckpt/<run_name>/` (or under `root`):

| file | what |
|---|---|
| `final.ckpt` | the Lightning checkpoint after the last step |
| `hparams.yaml` | every `TrainConfig` field of the run; a `warnings` entry lists declared weights that had no data |
| `MANIFEST.json` | system, run, seed, length, `global_step`, `final_md5`, the starting checkpoint, the sources, `terms_trained`, `declared_weights_without_data`, device, determinism, the split and order walked, the anchor rows and the sha256 of every table read, the penalty provenance, the kernel hinge's shape |
| `steps.json` | with `--log-every-step`: `{"loss": [...], "terms": [...]}` per step, written after the first step, every 50 steps, at the end and on failure |
| `UNTRAINED_TERMS.txt` | only when a term carries a non-zero weight and no data fed it |
| `job.pbs`, `job.log` | with `--pbs` |

The name `published` is refused: that directory holds the tracked checkpoints.

## Sources

A source is `(name, tree, pattern, grid[, exclude_tags])`. Its tree is relative to the declared
`source_root`; the pattern is a glob over the run directories in it, each holding a `modes.npz`
([data.md](data.md#the-archive-layout)); the grid is the real-space grid its modes are scattered
onto; the excluded tags are dropped. Several sources train together: one batch of each per step
(Lightning's `max_size_cycle`), each source's drift term weighted by
`defaults["source_loss_weights"][name]` (1.0 when undeclared).

| key of `defaults["training"]` | admissible | meaning |
|---|---|---|
| `source_root` | `{"tier": "raw" or "farm", "path": relative}` | `raw`: under the system's raw root; `farm`: under `<data>/<system>/` (where `aipf modes` writes) |
| `sources` | `{name: {"root", "pattern", "grid", "exclude_tags"}}` | the declared set, `--source declared`; `exclude_tags` optional |

## Windows

A run's frames are cut into windows; each window gives the drift target of its centre.

| key | admissible | meaning |
|---|---|---|
| `defaults["estimator"]` | `weak`, `weak_mid`, `savgol` | the weak-form estimator over `[c - w + 1, c + w]`, its midpoint variant, or Savitzky-Golay taps |
| `half_width` | int >= 1 | `w`, the half width in frames |
| `n_states` | int >= 1 | states per window |
| `stride` | int >= 1 | frames between window centres |
| `savgol_window`, `savgol_poly` | odd int, int; `None` unless `savgol` | the tap shape |
| `band_k_max` | positive float, optional | keep only the modes with `k <= band_k_max` in at least one box the run visits, before scattering onto the grid. Absent: every stored mode, which must then fit the grid |

## Split, loader, weighting

| key | admissible | meaning |
|---|---|---|
| `val_split` | `labels`, `random`, `none` | hold out the runs whose composition label is in `val_labels`; a seeded permutation of the sorted tags (at least one run held out); no validation |
| `val_labels` | tuple of label tuples | read by `labels` |
| `val_fraction` | in `[0, 1]` | read by `random` |
| `split_seed` | int | the permutation's seed |
| `batch_size` | int >= 1 | |
| `num_workers` | int >= 0 | loader processes |
| `pin_memory`, `drop_last` | bool | |
| `order` | `shuffled`, `index_stepping` | a fresh permutation per epoch, or the stepping sampler that replays a recorded order |
| `run_weighting` | `uniform`, `inverse_band_power` | every run weight one, or `1 / (s + median s)`, `s` a run's band-limited `1/k^2`-weighted mean square target, renormalised to mean one |
| `run_weight_probe_every` | int >= 1 | with `inverse_band_power`: the frame stride of the probes |
| `grad_clip` | float or absent | Lightning's `gradient_clip_val` |

## The optimiser and the loss weights

`aipf.train.config.TrainConfig` is flat and frozen; an unknown field is refused. Its fields are
read from the top level of `defaults`, then from `defaults["training"]`, which wins.

| field | default | meaning |
|---|---|---|
| `lr`, `weight_decay` | `5e-4`, `0.0` | AdamW |
| `wd_ghat` | `None` | the temperature heads' own decay; refused as undeclared when the model has them |
| `warmup_epochs`, `anneal_epochs`, `eta_min` | `5`, `100`, `1e-6` | linear warm-up, then cosine annealing to `eta_min` |
| `alpha_loss`, `h_inv_eps` | `0.0`, `1e-6` | the drift residual's mix of `r^2` and `r^2 / (k^2 + eps)` |
| `sigma`, `k_max` | `None` | the coarse-graining width; the drift band, `None` is `4 / sigma` |
| `lambda_dyn` | `1.0` | `L_dyn` |
| `lambda_M`, `lambda_S`, `lambda_bulk`, `lambda_P`, `lambda_conv`, `lambda_gamma`, `lambda_W` | `0.0` | the other terms; `0.0` is off |
| `stat_metric` | `rel_frob` | or `s_metric` |
| `bulk_residual` | `relative_inverse` | or `sigma_chi2` |
| `conv_penalty`, `conv_margin`, `conv_samples`, `gamma_pt_samples` | `hinge`, `0.0`, `0`, `0` | `conv_penalty` one of `hinge`, `hinge0`, `softplus` |
| `penalty_seed` | `None` | the penalty probes' stream; required when a penalty is trained |
| `source_loss_weights` | `{}` | the drift weight per source name (1.0 when a name is absent) |
| `seed` | `None` | set from `--seed`; refused as undeclared when a stream is derived |
| `model_type`, `model_kwargs` | `None`, `{}` | a saved checkpoint's model name and constructor arguments; the model itself is built from `System.functional` |
| `m_table`, `s_table`, `eos_csv`, `eos_csvs`, `eos_pressures`, `mask_dome`, `anchor_T_min_K` | `None`, `None`, `None`, `None`, `None`, `False`, `{}` | a saved checkpoint's anchor settings; the run reads its anchors from `defaults["training"]["tables"]` and `anchor_rules` |
| `k_fit_stat`, `grid`, `box` | `None` | recorded; the anchors read `defaults["k_fit_stat"]`, the functional's `grid` and `defaults["box"]` directly |
| `extra_experiment_config` | `{}` | a saved checkpoint's keys with no field of their own, and the kernel hinge's shape (`wpsd_kappa`, `wpsd_k_max`, `wpsd_n_k`, `wpsd_margin`) |

The fields of the last four rows are written to `hparams.yaml` and are filled when a saved
checkpoint's flat `hparams` are translated (`aipf.train.ckpt_compat.hparams_to_train_config`); `fit`
does not read them, except the kernel hinge's four shape keys in `extra_experiment_config`.

`seed` comes from `--seed`. The model and data streams are `derive_seed(seed, "model")` and
`derive_seed(seed, "data")`, the first 63 bits of `sha256("<seed>:<stream>")`.

## Losses

| term | what |
|---|---|
| `L_dyn` | the weak-form drift residual, `sum(band (alpha r^2 + (1 - alpha) r^2 / (k^2 + eps))) / n_band`, a batch mean over the band `0 < k <= k_max` |
| `L_M` | the mobility anchor, `norm(M_pred - M_target)^2 / norm(M_target)^2` per row (Frobenius norms) |
| `L_S` | the structure-factor shells `0 < k <= k_fit_stat`: the model's `H(k)` against `kB T S(k)^-1` |
| `L_bulk` | `k = 0`: `S_cc(0) = kB T z^T H(0)^-1 z / rho_tot` against the measured value |
| `L_P` | the pressure `P = mu . rho - f` against the equation-of-state table, relative to `max(abs(P_target), pressure_floor)` |
| `L_conv` | the hinge `lambda_min(H) >= conv_margin` at points drawn in the trust domain |
| `L_Gamma` | the hinge `d^2 Gamma / dx^2 >= 0` along the measured equation-of-state paths |
| `L_W` | the kernel hinge `lambda_min(W(k) - W(0)) >= margin + kappa k^2` on `n_k` wavenumbers `linspace(k_max / n_k, k_max, n_k)`, scored `mean(relu(margin + kappa k^2 - lambda_min)^2)`; two channels take the closed-form smallest eigenvalue |

`L_W` needs no data and is computed once a step from the kernel alone. A system declares it the way
a saved checkpoint spells it: the weight `lambda_wpsd`, which the driver maps to `lambda_W`, and the
shape `wpsd_kappa`, `wpsd_k_max`, `wpsd_n_k` and `wpsd_margin`, which it keeps in
`extra_experiment_config`. A non-zero weight without all four shape keys is refused, and so is a
model without a pair kernel. `MANIFEST.json` names the terms a run trained.

## Anchor tables

`defaults["training"]["tables"]` holds the keywords of `AnchorTables.load`. A relative path is looked
for under `experiments/<system>/anchors/` first, then under the raw root. `key` picks the layout.

`key: "pressure"`, one file per pressure, rows that carry their densities (feeds `L_M`, `L_S`,
`L_bulk`, `L_P`):

| key | meaning |
|---|---|
| `m_table`, `s_table`, `eos_csvs` | `{pressure: relative path}` |
| `pressure_unit` | one pressure unit in the model's energy-per-volume unit |
| `columns` | `phase` (a name, or `{"mobility": ..., "structure": ...}`), `temperature`, `densities` (channel order), `composition`, `mobility`, `shells`, `structure`, `bulk_target`, `bulk_sigma`, `bulk_ok` |
| `eos_columns` | `x_column`, `T_column`, `n_columns` (alternatives), `status_column`, `status_ok` |
| `row_weight` | `None`, or `{"value", "error", "floor"}`: `1 / max(abs(error / value), floor)^2` |
| `pressure_floor` | `None` or the floor of `L_P`'s denominator, in the keys' unit |
| `w0` | `{"route": "evaluator"}`, `{"route": "lattice"}` or `{"route": "radial", "r_max", "n_points"}`: how `W(0)` enters the bulk anchor |

A row is kept when its `phase` column is true and its temperature is above
`anchor_rules.T_min_by_pressure` at that pressure (every temperature when none is declared). `defaults["k_fit_stat"]` and
`constants["kB"]` are read too.

`key: "temperature"`, one channel, one row per temperature at one declared state (feeds `L_M` and
`L_bulk`): `m_table`, `s_table` (paths), `state` (the composition), `rho_total`, `w0`, and `columns`
with `temperature`, `mobility`, `records`, `record_temperature`, `structure_zero`, `temperatures`.

## Penalties

`L_conv` is trained when `lambda_conv` is non-zero and the system declares a `trust_domain`; it needs
`defaults["training"]["conv_T_measure"]` (`uniform` or `log_uniform` over `T_range`) and
`conv_samples >= 1`. `L_Gamma` is trained when `lambda_gamma` is non-zero and
`defaults["training"]["gamma_paths"]` declares exactly `eos_csvs`, `eos_columns`, `x_grid`
(`(x_lo, x_hi, h)`), `poly_degree`, `T_measure`, `q_floor`. Both draw from `penalty_seed`.

## Starting weights

`--init-from-published` (`init_from=system.checkpoint`) loads the system's published weights,
digest-checked, into the declared model. `--resume-optimizer no` builds the optimizer and schedule
fresh, so the first steps sit in the linear warm-up. `--resume-optimizer yes` resumes the saved
optimizer and schedule; the published `lj` checkpoint's saved optimizer state has parameter groups that do not match a fresh optimizer's, so `yes` fails on it; start from it with `no`. A fresh run is built
under the model stream of `--seed`, and does not reproduce a published run's own initialisation.

## Devices and determinism

`device="auto"` is cuda when torch sees a GPU, else cpu; `cuda` without one raises
`DeviceUnavailable` before the run directory exists. `deterministic=True` asks torch for
deterministic kernels for the run and restores the process-wide switches after it; `False` touches
none.
