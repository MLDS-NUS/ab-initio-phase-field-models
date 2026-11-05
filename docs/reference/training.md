# Training

`aipf train` and `aipf.train.fit.fit` train the functional a system declares on windows of its mode
archives, with the measured anchor tables and the convexity penalties the system declares. The
command line names only the run ([cli.md](cli.md#aipf-train)); everything else is read from
`System.defaults` and `System.defaults["training"]`. A missing key is refused, naming it and the
block it belongs in.

    fit(system, *, run_name, sources, seed, resume_optimizer, steps=None, epochs=None,
        window=None, weighting=None, config_overrides=None, split_mode=None, loader_order=None,
        init_from=None, anchors=None, root=None, log_every_step=False, device="auto",
        deterministic=False, projection=None) -> Path

`sources` are `SourceSpec` rows (`aipf.train.fit._specs_from(system, table, names)` builds them from
the command line's table). `anchors=None` loads the declared tables, `aipf.train.NO_ANCHORS` trains the
drift term alone. `None` for `window`, `weighting`, `split_mode` and `loader_order` means the
declaration answers. Exactly one of `steps` and `epochs`. `projection` trains a two-dimensional model
on the `k_z = 0` plane of the archives ([Two dimensions](#two-dimensions)); `None` is the
three-dimensional run it always was.

## The run directory

`<data>/<system>/ckpt/<run_name>/` (or under `root`):

| file | what |
|---|---|
| `final.ckpt` | the Lightning checkpoint after the last step |
| `hparams.yaml` | every `TrainConfig` field of the run; a `warnings` entry lists declared weights that had no data |
| `MANIFEST.json` | system, run, seed, length, `global_step`, `final_md5`, the starting checkpoint, the sources, `terms_trained`, `declared_weights_without_data`, device, determinism, the split and order walked, the anchor rows and the sha256 of every table read, the penalty provenance, the kernel hinge's shape; `model_factory` for a functional built by a factory; `projection`, `projection_Lz_ref` and `projection_depth` for a two-dimensional run |
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
| `sample` | `{name: {"root", "pattern", "grid"}}` | the bundled sample, `--source sample`; each root relative to `data/<system>/sample` |
| `mmap_sidecar` | `True` or `False`, optional | `True`: map the amplitudes of every run that carries `rho_k.npy` instead of reading them into memory ([Memory-mapped amplitudes](#memory-mapped-amplitudes)). Absent or `False`: every run is read into memory, as before |

`--source sample` trains `aipf.train.fit.sample_system(system)`: the system with its raw root
replaced by `data/<system>/sample` (`aipf.paths.SamplePaths`), its source root that directory and
its declared sources the `sample` rows. Every other entry of the declaration is the system's own, and
the anchor and equation-of-state tables are read from the sample at the paths the declaration gives
relative to the raw root (a table tracked under `experiments/<system>/anchors/` first, as always).
The sample is tracked in the checkout like the published checkpoints, so `AIPF_DATA` does not move
it. It lets the pipeline run without the raw archive; a model trained on it carries no physics.

### Memory-mapped amplitudes

By default every run's amplitudes are read into memory when the sources are set up, so a run needs
RAM for its whole dataset. A run directory may also hold `rho_k.npy`, an uncompressed copy of the
amplitudes in the layout a run is held in, `(n_frames, n_channels, n_modes)` C order, and its record
`rho_k.json` ([data.md](data.md#the-memory-mapped-sidecar)). With
`defaults["training"]["mmap_sidecar"] = True` (`aipf.train.dataset.ArchiveKeys.mmap_file =
"rho_k.npy"`), `read_mode_runs` opens that file with `numpy.load(mmap_mode="r")` and never reads
`rho_k` from `modes.npz`: `ModeRun.amplitudes` is a read-only `numpy.memmap`, the quality cut a view
of it, and a window reads its frames from the file when it is cut. Setup holds the labels, the boxes
and the scalars; the frames are paged in as windows read them and stay in the page cache while there
is free RAM, and the kernel drops them again under memory pressure.

- A run without the file reads into memory, as before, so a source may mix the two. When the sidecar
  is declared and some runs have none, `read_mode_runs` logs a warning with their count and the first
  few tags (each of them costs its whole timeline in RAM).
- Before a run is mapped its `rho_k.npy` is checked against `modes.npz`, without decompressing
  `rho_k`: its shape is `rho_k`'s, read off the stored array's header, with the channel axis moved
  (frames, channels and modes, and again the modes against `nvec` and the frames against a per-frame
  `box`), its dtype is `rho_k`'s, it is in C order, and its record `rho_k.json` holds the CRC-32 and
  size of the `rho_k` member of `modes.npz` as it is now (read off the zip directory). Anything
  else, a file that is not a readable `.npy` (an empty one included), or a sidecar without its record,
  raises `ValueError` naming the run. A rewritten `rho_k` of the same shape is therefore refused; a
  sidecar overwritten by hand with other frames of the right shape is not, and
  `aipf modes-sidecar --verify` compares the frames.
- Every window, state, target, box, label and weight is bit-identical to the in-memory reader's,
  with any number of loader workers. Workers started by `fork` (Linux's default) share the mapping.
  Under `spawn` a mapped run pickles as its file, offset and shape and each worker opens the file
  again by its path, so its frames are never copied; a sidecar rewritten while such a run trains
  would be read by the workers started after it, so do not rewrite archives under a running job.
- Each mapped run keeps one open file descriptor (CPython's `mmap` holds a duplicate), so 1,625 runs
  hold 1,625 descriptors in the trainer and in every forked worker. Before it maps a source,
  `read_mode_runs` raises the soft `RLIMIT_NOFILE` to one per run mapped and still held, plus one per
  run it is about to map, plus 256 for everything else, when it is lower, and refuses with `OSError`
  when the hard limit is lower than that.
- Mapped pages that have been read count in the process's resident set (`VmRSS`, as file-backed
  `RssFile`) although they are cache the kernel can reclaim: a pass over every frame of a 300 MB
  sidecar adds about 300 MB to it. Shared pages are counted in every process that touched them, so
  with forked loader workers a scheduler that sums the resident sets of a job's processes (PBS can)
  may report several times the page cache actually used. Check one pilot job's reported memory
  before sizing a campaign by it.
- The `inverse_band_power` run weighting reads its probe windows at setup, which pages those frames
  in. A run read through `--projection` is held in memory: `project_kz0` copies the `n_z = 0` modes
  of the mapped run.

The sidecar is written by `ModesRecord.save(directory, mmap_sidecar=True)` for a new archive and by
`aipf modes-sidecar` for archives already on disk ([cli.md](cli.md#aipf-modes-sidecar)).

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
| `bulk_residual` | `relative_inverse` | or `sigma_chi2` (two channels); `s_metric` (the concentration block, three or more) |
| `conv_penalty`, `conv_margin`, `conv_samples`, `gamma_pt_samples` | `hinge`, `0.0`, `0`, `0` | `conv_penalty` one of `hinge`, `hinge0`, `softplus` |
| `penalty_seed` | `None` | the penalty probes' stream; required when a penalty is trained |
| `source_loss_weights` | `{}` | the drift weight per source name (1.0 when a name is absent) |
| `seed` | `None` | set from `--seed`; refused as undeclared when a stream is derived |
| `model_type`, `model_kwargs` | `None`, `{}` | a saved checkpoint's model name and constructor arguments; the model itself is built from `System.functional` |
| `m_table`, `s_table`, `eos_csv`, `eos_csvs`, `eos_pressures`, `mask_dome`, `anchor_T_min_K` | `None`, `None`, `None`, `None`, `None`, `False`, `{}` | a saved checkpoint's anchor settings; the run reads its anchors from `defaults["training"]["tables"]` and `anchor_rules` |
| `k_fit_stat`, `grid`, `box` | `None` | recorded; the anchors read `defaults["k_fit_stat"]`, the functional's `grid` and `defaults["box"]` directly |
| `extra_experiment_config` | `{}` | a saved checkpoint's keys with no field of their own, the kernel hinge's shape (`wpsd_kappa`, `wpsd_k_max`, `wpsd_n_k`, `wpsd_margin`, and when declared `wpsd_subspace`, `wpsd_k_sat`), and when declared `L_conv`'s source `conv_states`, `conv_k` |

The fields of the last four rows are written to `hparams.yaml` and are filled when a saved
checkpoint's flat `hparams` are translated (`aipf.train.ckpt_compat.hparams_to_train_config`); `fit`
does not read them, except the kernel hinge's shape keys and `L_conv`'s source keys in
`extra_experiment_config`.

`seed` comes from `--seed`. The model and data streams are `derive_seed(seed, "model")` and
`derive_seed(seed, "data")`, the first 63 bits of `sha256("<seed>:<stream>")`.

## Losses

| term | what |
|---|---|
| `L_dyn` | the weak-form drift residual, `sum(band (alpha r^2 + (1 - alpha) r^2 / (k^2 + eps))) / n_band`, a batch mean over the band `0 < k <= k_max` |
| `L_M` | the mobility anchor, `norm(M_pred - M_target)^2 / norm(M_target)^2` per row (Frobenius norms; over the row's active block, beyond two channels) |
| `L_S` | the structure-factor shells `0 < k <= k_fit_stat`: the model's `H(k)` against `kB T S(k)^-1` (on the row's active block `S_PP(k)`, beyond two channels) |
| `L_bulk` | `k = 0`: `S_cc(0) = kB T z^T H(0)^-1 z / rho_tot` against the measured value; beyond two channels the `m x m` block `kB T Z^T H_PP(0)^-1 Z / rho_tot`, scored `norm(R S_cc R - I)^2` with `R = S_target^-1/2` (`s_metric`) |
| `L_P` | the pressure `P = mu . rho - f` against the equation-of-state table, relative to `max(abs(P_target), pressure_floor)` |
| `L_conv` | the hinge `lambda_min(H) >= conv_margin` at points drawn in the trust domain; or, with `conv_states="anchors"`, `lambda_min(H(k)) >= conv_margin` on each single-phase anchor row's active block at every `k` of a declared line (see Penalties) |
| `L_Gamma` | the hinge `d^2 Gamma / dx^2 >= 0` along the measured equation-of-state paths |
| `L_W` | the kernel hinge `lambda_min(W(k) - W(0)) >= margin + kappa k^2` on `n_k` wavenumbers `linspace(k_max / n_k, k_max, n_k)`, scored `mean(relu(margin + kappa k^2 - lambda_min)^2)`; two channels take the closed-form smallest eigenvalue (of a symmetric kernel). Why: the floor gives the kernel a positive square-gradient stiffness at every k, so instability can only start at k = 0, and with `W(k_max) ~ 0` it pins `W(0) <= -kappa k_max^2` (about -18 for the Fe-B declaration), the attractive kernel that carries the miscibility gap. It is not an optional regulariser: weighed at 0, Fe-B loses its gap |

`L_W` needs no data and is computed once a step from the kernel alone. A system declares it the way
a saved checkpoint spells it: the weight `lambda_wpsd`, which the driver maps to `lambda_W`, and the
shape `wpsd_kappa`, `wpsd_k_max`, `wpsd_n_k` and `wpsd_margin`, which it keeps in
`extra_experiment_config`. A non-zero weight without all four shape keys is refused, and so is a
model without a pair kernel. `MANIFEST.json` names the terms a run trained, and its `kernel_hinge`
records the hinge's shape and the `lambda_W` it trained at.

Two optional keys, declared beside the four and kept with them, change what is hinged. Neither appears
in `hparams.yaml`, `MANIFEST.json` or the checkpoint of a run that does not declare it.

| key | meaning |
|---|---|
| `wpsd_subspace` | `None` (absent: the full `n x n` matrix) or `"number_conserving"`: `lambda_min` of the reduced block `B^T (W(k) - W(0)) B` |
| `wpsd_k_sat` | absent: the floor `kappa k^2`; a positive wavenumber: `kappa min(k, k_sat)^2`, constant from `k_sat` on |

`B` is an orthonormal basis of the number-conserving subspace `1^perp`, `n x (n - 1)`: the Helmert
columns `b_j = (1, ..., 1, -j, 0, ..., 0) / sqrt(j (j + 1))` (`j` ones), so `(1, -1)/sqrt(2)` at two
channels and `(1, -1, 0)/sqrt(2)`, `(1, 1, -2)/sqrt(6)` at three. `lambda_min` does not depend on that
choice: any other orthonormal basis is `B Q` with `Q` orthogonal, and `Q^T (B^T D B) Q` has the same
eigenvalues. Two channels take the scalar `b^T D b`, three the closed-form smallest eigenvalue of the
`2 x 2` block, more `eigvalsh`; one channel is refused (`1^perp` is `{0}`). The block is the reduced
one and not the projector `P D P` (`P = B B^T`), which keeps an `n x n` matrix with a spurious zero
eigenvalue along `1` and so reads `min(0, ...)`. Why a subspace: a liquid's number-number structure
factor rises toward its first peak, so `W(k) - W(0)` is negative along the total-number direction at
every `k` in measured data, and no `kappa >= 0` on the full matrix agrees with it; on `1^perp` the
same data admit a small positive `kappa`.

`wpsd_k_max` may reach the grid's Nyquist corner (it only needs to be positive). The radial
transform's `W(k)` (`AnalyticRadialTransform`) is interpolated on its table up to the functional's
`kernel_k_table_max` and clamps `|k|` there, so a hinge beyond that wavenumber reads one flat `W` and
gains nothing.

## Anchor tables

`defaults["training"]["tables"]` holds the keywords of `AnchorTables.load`. A relative path is looked
for under `experiments/<system>/anchors/` first, then under the raw root. `key` picks the layout.

`key: "pressure"`, one file per pressure, rows that carry their densities (feeds `L_M`, `L_S`,
`L_bulk`, `L_P`):

| key | meaning |
|---|---|
| `m_table`, `s_table`, `eos_csvs` | `{pressure: relative path}` |
| `pressure_unit` | one pressure unit in the model's energy-per-volume unit |
| `columns` | `phase` (a name, or `{"mobility": ..., "structure": ...}`), `temperature`, `densities` (channel order), `composition`, `mobility`, `shells`, `structure`, `bulk_target`, `bulk_sigma`, `bulk_ok`; the vector form adds `active` |
| `eos_columns` | `x_column`, `T_column`, `n_columns` (alternatives), `status_column`, `status_ok`; the vector form has `x_columns` in place of `x_column` |
| `row_weight` | `None`, or `{"value", "error", "floor"}`: `1 / max(abs(error / value), floor)^2`; `value: None` reads `error` as already relative, `1 / max(abs(error), floor)^2` |
| `pressure_floor` | `None` or the floor of `L_P`'s denominator, in the keys' unit |
| `w0` | `{"route": "evaluator"}`, `{"route": "lattice"}` or `{"route": "radial", "r_max", "n_points"}`: how `W(0)` enters the bulk anchor |

A row is kept when its `phase` column is true and its temperature is above
`anchor_rules.T_min_by_pressure` at that pressure (every temperature when none is declared). `defaults["k_fit_stat"]` and
`constants["kB"]` are read too. Weights are normalised to mean one, and the shells and the bulk anchor
take the same ones.

The composition is declared in one of two forms, and each count of channels has exactly one:

| channels | `columns["composition"]` | `eos_columns` | also |
|---|---|---|---|
| two | one name, the fraction of channel `table_keys["x_channel"]` | `x_column` | |
| three or more | a tuple, one fraction column per channel in channel order | `x_columns`, the same | `columns["active"]`, the `(rows, n)` bool mask of the species present in a row |

```python
columns["densities"]   = ("rho_Fe", "rho_Cu", "rho_Ni")   # one name per channel, as always
columns["composition"] = ("x_Fe", "x_Cu", "x_Ni")
columns["active"]      = "active"
eos_columns["x_columns"] = ("x_Fe", "x_Cu", "x_Ni")
row_weight = {"value": None, "error": "Scc0_err", "floor": 0.02}
```

One name at three or more channels is refused (it cannot say which channel it labels), the tuple at
two channels is refused, and so is a mix of the two (a tuple `composition` with `x_column`, or one
name with `x_columns`). The vector form does not read `table_keys["x_channel"]`. Its tables carry,
per row:

- `active`, equal on every row to `x > 0` and to `rho > 0`, and fractions summing to one within
  `1e-6`; a row that disagrees is refused, by file and row index;
- `M_k0` `(rows, n, n)`, `S_k` `(rows, shells, n, n)`, and the bulk target `(rows, n-1, n-1)`: `S_cc(0)`
  on the row's independent species, the first `m = |P| - 1` present species `P` in channel order, in its
  leading `m x m` block. Entries outside a row's active block are ignored, whatever they hold (NaN, say);
  a non-finite entry inside it drops the shell or the bulk row, and is refused in `M_k0`. A finite bulk
  block that is not positive definite is refused, naming the file and row: `S_cc(0)` is positive definite
  by construction, and its inverse square root `R` has no meaning otherwise.

A row with fewer than two species present feeds `L_P` alone. Each other row is compared on its active
principal block: `L_M` on `M_PP`, `L_S` on `H_PP(k)` against `kB T S_PP(k)^-1` (an absent species
drops no shell), and `L_bulk` on

    S_cc(0) = kB T Z^T H_PP(0)^-1 Z / rho_tot,   z_i = e_i - x_i a,  i the first m present species,

with `a` the row's active indicator, so `z_i` vanishes on an absent channel and is `(x_1, -(1 - x_1))`
at two channels. The block is scored `norm(R S_cc(0) R - I)^2` over its `m x m` entries, with
`R = S_target^-1/2` fixed at load; `bulk_residual` is then `s_metric`, and `bulk_sigma`, `bulk_ok` are
`None` (they are refused). Why the principal block and not the full `H^-1`: the true ideal term
`kB T / rho` of an absent channel is infinite, and in that limit the absent channel's block of `H^-1`
vanishes, leaving `(H_PP)^-1`. The model's own `H` cannot carry that limit: its ideal term clamps the
density at `rho_eps`, so on a channel that is exactly absent it carries no ideal stiffness at all. The
manifold reads
every ok row, pure ones too, as `rho = n (x_1, ..., x_n)`; fractions that do not sum to one within
`1e-6` are refused, naming the line.

`key: "temperature"`, one channel, one row per temperature at one declared state (feeds `L_M` and
`L_bulk`): `m_table`, `s_table` (paths), `state` (the composition), `rho_total`, `w0`, and `columns`
with `temperature`, `mobility`, `records`, `record_temperature`, `structure_zero`, `temperatures`.

## Penalties

`L_conv` is trained when `lambda_conv` is non-zero and the system declares a `trust_domain`; it needs
`defaults["training"]["conv_T_measure"]` (`uniform` or `log_uniform` over `T_range`) and
`conv_samples >= 1`. `L_Gamma` is trained when `lambda_gamma` is non-zero and
`defaults["training"]["gamma_paths"]` declares exactly `eos_csvs`, `eos_columns`, `x_grid`
(`(x_lo, x_hi, h)`), `poly_degree`, `T_measure`, `q_floor`. Both draw from `penalty_seed`.

### `L_conv` on the anchor states

Two optional keys, declared beside the loss weights and kept in `extra_experiment_config` (absent from
every record of a run that does not declare them), move `L_conv` from the trust domain to the states the
anchor tables measured single phase:

| key | meaning |
|---|---|
| `conv_states` | `"trust_domain"` (absent: the points drawn above) or `"anchors"` |
| `conv_k` | `(n_k, k_max)`, required with `"anchors"`: the line `linspace(0, k_max, n_k)`, `n_k >= 2` a whole number, kept as a list; refused with `"trust_domain"` |

With `"anchors"`, on every `anchor_bulk` row (the rows the phase gate kept; rows with fewer than two
species present are not bulk rows) and every `k` of the line, `k = 0` included,

    lambda_min[ d2 f_loc(rho, T) + W(k) ]_PP >= conv_margin,

scored by `conv_penalty` exactly as on the trust domain (`hinge`, `hinge0`, `softplus`, the same
`viol_frac` and `lam_worst`, `viol_frac` counting `lambda_min < 0`), the mean over rows and wavenumbers.
`PP` is the row's active principal block: rows are grouped by active pattern and each block is taken at
its own size, never padded, since an exactly absent channel carries no ideal stiffness in the model
(`rho_eps` clamps its density) and a full-matrix minimum would read that channel. Tables without an
`active` column (the two-channel form) count a channel active where its density is positive. The
condition holds for any equilibrium single-phase state (`S(k)` finite and positive at every `k`), so
it only removes a model that is unstable where the measurement says the liquid is not; it constrains
the total-number direction at finite `k`, which a number-conserving kernel hinge leaves free.

Nothing is evaluated beyond what the anchors already evaluate. The `k = 0` slice is the bulk anchor's
own `H(0)` (so its declared `w0` route); every other slice adds the `W(k)` that `L_S` reads, the
kernel's evaluator, to the same curvature, which `AnchorTables.batch(..., curvature=True)` hands over
once a step. Under the `radial` route the `k = 0` slice therefore carries the quadrature `W(0)` and the
next ones the evaluator's table, as `L_bulk` and `L_S` do. A kernel evaluated as a lattice sum, which
has no `W(k)` on a bare wavenumber, is refused, and so is a run that loads no anchor tables
(`anchors=NO_ANCHORS`, `aipf train --anchors none`, no `tables` declared, before the run directory
exists) or tables without a bulk row. No point is drawn, so `penalty_seed` is needed only for
`L_Gamma`; `conv_samples` and `conv_T_measure` are not read. As for the kernel hinge, `W(k)` is flat
beyond `kernel_k_table_max`, so a line past it repeats its last slice. `MANIFEST.json` records the line
and the row count under `penalties["conv_states"]`, and each step logs `train/L_conv_viol_frac` and
`train/L_conv_lam_worst` (in `steps.json`, `L_conv/viol_frac` and `L_conv/lam_worst`). A
two-dimensional model is refused as for the trust-domain `L_conv`. The trust-domain `L_conv` and
`L_Gamma` are as above.

## Starting weights

`--init-from-published` (`init_from=system.checkpoint`) loads the system's published weights,
digest-checked, into the declared model. `--resume-optimizer no` builds the optimizer and schedule
fresh, so the first steps sit in the linear warm-up. `--resume-optimizer yes` resumes the saved
optimizer and schedule when its parameter groups fit this model's fresh optimizer tensor by tensor,
and otherwise is refused by name (exit 2, `OptimizerLayoutMismatch` from `fit`) before the run
directory exists. A saved optimizer lists positions, not names, so the order is checked against the
parameter names a checkpoint written by `fit` records at each position (`optimizer_param_names`). A
checkpoint that records none is refused when two tensors of one group share a shape, since their
order cannot be checked. So `yes` can only succeed on a checkpoint that `fit` wrote. Every published
checkpoint records none and is refused, so start from one with `no`. The published `lj` checkpoint differs in count as well: the run that wrote it held the
local net's linear skip `g_net.w2.weight` frozen, so its one group has 15 tensors against 16 here.
A fresh run is built
under the model stream of `--seed`, and does not reproduce a published run's own initialisation.

## Two dimensions

A two-dimensional model (a factory model declared on a two-axis `grid`, [functional.md](functional.md#two-dimensions))
trains on three-dimensional archives through a projection: `fit(..., projection=...)`,
`aipf train --projection`. The archives stay as they are written; every run is read, then cut to its
`k_z = 0` plane by `aipf.train.projection.project_kz0` before it is split, weighted and windowed.

| `projection` | `rho_hat` of a window | `depth` |
|---|---|---|
| `None` | `rho_k / V` (three dimensions, as always) | none |
| `"kz0-volumetric"` | `rho_k / V_ref`: the z mean of the 3D density, per unit volume | `Lz_ref` |
| `"kz0-areal"` | `rho_k / (Lx_ref Ly_ref) = Lz_ref rho_k / V_ref`: the density integrated along z, per unit area | `1.0` |

The window's measure is `Lx_ref Ly_ref * depth`, the reference cell's ([data.md](data.md#the-k_z--0-projection)).
Every source grid is then `(Gx, Gy)`; the labels with `n_z == 0` are scattered onto the half spectrum
`(Gx, Gy//2+1)` with `n_y >= 0`, the rule the 3D scatter applies to `n_z`. The window boxes, `k`, the
band mask, the sigma filter and the run weights are read in the cell's `(Lx, Ly)`. The `LitModule`
reads the number of axes from its operator set (`ops.ndim`), never from a batch's shape.

A two-dimensional model trains the drift term alone. Refused before the run directory exists, as
`aipf.train.fit.DimensionMismatch` (exit 2 from `aipf train`):

- a projection for a model built on a three-axis grid, and a two-axis model without one;
- a source grid of another number of axes than the model's;
- the anchor tables (a declared `tables` entry under `anchors=None`, or an `AnchorTables`): pass
  `anchors=NO_ANCHORS`, `--anchors none`;
- `L_conv` or `L_Gamma` that the declaration would train, and a non-zero `lambda_W`.

- an archive the projection would read that records no reference cell: the projection needs one
  `Lz` for every frame, which the frames of an NPT run, each in its own box, do not share. Write the
  archive with `modes_from_dump(reference_box=...)` (a stated cell or `"first_frame"`);
- archives whose reference cells differ in `Lz`: a trained 2D model has one `depth`, and its areal
  densities are per unit area of one slab thickness. State one cell for every run of every source.

These are read off each archive's index and its `reference_box` alone.
`aipf.train.fit.check_dimensions(system, sources, projection=..., anchors=...)` makes the same
refusals without training (it builds the model); `aipf train --pbs` makes them before it submits,
with `--projection` or for a system whose functional declares a two-axis grid. A three-dimensional
`fit` whose source grids are not all three-axis is refused the same way (it would fail when its
sources are read).

A noisy rollout of the trained model declares `depth` ([functional.md](functional.md#two-dimensions)),
the same number the window was divided by. The run's `MANIFEST.json` records it beside the projection
(`projection`, `projection_Lz_ref`, `projection_depth`: `1.0` areal, `Lz_ref` volumetric), and
`aipf.train.projection_depth(archive_run_dir, areal=...)` reads it off an archive's run directory.
With another `depth` the noise is off by that ratio in variance.

The trained model is an effective one. The `k_z = 0` plane of a 3D trajectory is not a closed
dynamics: the nonlinear terms of the 3D dynamics couple the `k_z != 0` modes into it, so the drift the
2D model fits is the measured drift of the z mean, closure error included, and its parameters (the
2D `W(r)` among them) are not those of a 3D model. For a z-invariant 3D field the plane is closed, and
`tests/unit/test_train_projection.py` recovers the mobility and `kappa` of a known 2D model from such
data.

## Devices and determinism

`device="auto"` is cuda when torch sees a GPU, else cpu; `cuda` without one raises
`DeviceUnavailable` before the run directory exists. `deterministic=True` asks torch for
deterministic kernels for the run and restores the process-wide switches after it; `False` touches
none.
