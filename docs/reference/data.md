# Data

Three places hold a system's data:

| place | what | written by |
|---|---|---|
| the raw root (`AIPF_RAW_<SYSTEM>`, [cli.md](cli.md#locations-aipfpaths)) | the MD runs: trajectories, logs, run records; and, under `fields/`, the archived mode trees, measured tables and equation-of-state tables | MD, `aipf md run`; never written by any other command |
| the farm (`AIPF_DATA`, default `<repo>/data`) | `<data>/<system>/`: the index, the mode archives `aipf modes` extracts, training runs, diagnoses, rollouts | every other command |
| the checkout | `data/<system>/ckpt/published[/<variant>]/final.ckpt`, the published checkpoints; `data/<system>/sample/`, the training sample `aipf train --source sample` reads ([training.md](training.md#sources)); and `experiments/<system>/anchors/`, small measured tables | tracked files |

## The farm

| path under `<data>/<system>/` | what |
|---|---|
| `manifest.json` | every run found under the raw root, indexed or skipped with a reason |
| `md/<farm_dir>/traj.lammpstrj` | a link to the run's trajectory |
| `md/<farm_dir>/thermo.txt` | a link to `thermo_prod.txt`, else `log.lammps` |
| `md/<farm_dir>/meta.json` | the normalised metadata |
| `modes/<farm_dir>/modes.npz`, `provenance.json` | one run's mode archive, `aipf modes` |
| `ckpt/<run>/` | a training run ([training.md](training.md#the-run-directory)) |
| `diagnose/<md5[:12]>/` | one checkpoint's diagnosis |
| `rollout/<key>/` | rollouts, `aipf rollout --out data` |

`farm_dir` is `<partition>/<tag>`, the partition built from the system's `farm_partition` fields
(`P_GPa` renders as `800GPa`, or `{P:g}GPa` when not an integer), or the tag alone when the
system declares `farm_partition = ()`. The trajectories stay where the MD wrote them.

## aipf data build

    aipf.data.index.build(system, dry_run=False) -> manifest

Every directory under the raw root holding a file named in `system.traj_names` is a run. Its
metadata come from, in order:

1. `run.json` written by `aipf md run`: the request (tag, geometry, ensemble, T, composition,
   timestep, cadence, seed) and `log.lammps` for what the engine did. A dry run, another system's
   record, or a log whose atom count differs from the requested `n_atoms` is skipped with the reason.
2. `<tag>_meta.json` or `meta.json` in the run directory.
3. `log.lammps`: the last `timestep`, the `run` counts (the last is production, the others
   preparation: `n_prod`, `n_equil`), the thermostat's temperature and seed, the dump cadence, the
   atom count (the last of `create_atoms`, `read_data` and `delete_atoms`).

A run `aipf md run` wrote takes its ensemble from its `run.json` request. Without a run record, the
geometry comes from the tag's prefix, and so does the ensemble, except that a bare geometry prefix
(`cube`, `slab`, `ball`, `column`, `vext`) takes the ensemble of the system's declared md deck of
that geometry (`defaults["md"][<deck>]["point"]`) when it declares one:

| prefix | geometry | ensemble |
|---|---|---|
| `slab_overdamped` | slab | langevin_overdamped |
| `slab_meltfirst` | slab | NVT |
| `homogeneous_brownian`, `nucleation_seed`, `spinodal_cube` | cube | langevin_overdamped |
| `cube` | cube | NPT |
| `slab` | slab | NPT_z |
| `ball`, `column`, `vext` | ball, column, vext | NVT |
| `quench` | quench | langevin_overdamped |

Every archived cube of the two-species systems ran a barostat (`fix npt ... iso`), so `cube` reads
`NPT`: Fe-B's from its declared `cube-npt` deck, H/He's (no cube deck) from the table. The label is
metadata: `aipf train` and `aipf diagnose` read no manifest field. The tag is the run directory's name, or with `tag_from_path` its path below the raw root (without
`tag_path_root`) joined by `_`. Runs under a top-level directory named in
`constants["md_exclude_dirs"]` are skipped. A run that fails validation, or whose partition field is
null, is skipped with the reason; two runs with one `farm_dir` stop the build. Every
trajectory-shaped file (`.lammpstrj`, `.dump`, or a declared name) is indexed, skipped or listed as
unrecognised, and the three counts must add up.

A farm's `manifest.json` indexes one raw root. A build that reads another raw root into a farm whose
manifest indexes a different one is refused, and writes nothing (`ForeignManifest`); point
`AIPF_DATA` at another farm. `dry_run` writes nothing.

## meta.json

| key | meaning |
|---|---|
| `schema_version` | 1 |
| `system`, `tag` | |
| `geometry` | `cube`, `slab`, `ball`, `column`, `vext`, `eos`, `quench` |
| `ensemble` | `NVE`, `NVT`, `NPT`, `NPT_z`, `langevin_overdamped` |
| `T_K` | the temperature, in the system's unit |
| `P_GPa` | the pressure, else `constants["P_GPa"]` |
| `composition` | `{"species", "kind": "uniform", "regions" or "two_slab", "x": {species: value or [values]}}` |
| `box` | `{"L": the starting edge of a cubic box or null, "varying": the axes that move}`: `["z"]` under `NPT_z`, `["x", "y", "z"]` under `NPT`, none under the others (`aipf.data.meta.moving_axes`) |
| `n_atoms`, `dt_ps`, `dump_every_ps`, `n_frames` | numbers; the last three positive |
| `engine`, `potential`, `seed`, `status` | |
| `extra` | everything else the source recorded: `n_equil`, `n_prod`, the template, the campaign, the path of `run.json` |

`aipf.data.meta.validate(meta)` lists the problems of a record; empty is valid.

## The manifest

`manifest.json`: `schema_version`, `system`, `raw` (the raw root), `state_points` (per run: `tag`,
`farm_dir`, `meta`, `source_dir`, `trajectory`, `thermo`; or `tag` and `skipped`), `n_indexed`,
`n_skipped`, `unrecognised`, `n_unrecognised`.

`aipf data rebuild` (`index.rebuild_from_manifest(path)`) recreates `md/<farm_dir>/` (the two links
and `meta.json`) from it. The `modes` tier is never rebuilt.

`index.record_for_tag(system, tag)` finds one indexed run by tag or `farm_dir`;
`index.ckpt_dir(system)`, `index.modes_dir(system, farm_dir)` and `index.diagnose_dir(system)` are the
tiers. `index.index_fields(system)` lists the trees under `<raw>/fields/` with the role
`constants` gives them (`modes`, `fdt_runs`, `tables`, else `archived`).

## Modes

A mode archive holds, for every frame, the Fourier sums of each channel's atoms over the
wavevectors `k = 2 pi n / L` with `|k| <= k_cut`, `n` integer, on the time-mean box of the run.

    aipf.pipeline.modes.modes(system, tag, *, sigma, k_cut, ordering="lexicographic", route="dense")

`aipf modes` calls it. `tag` is a run tag or `farm_dir` of the manifest. It writes
`<data>/<system>/modes/<farm_dir>/modes.npz` and `provenance.json`, the layout an archive keeps,
`<tree>/<tag>/modes.npz`, so a training source reads the farm with the archive's own pattern and
tags. The channels are the system's `mode_fields`, the frames skipped are the preparation's
(`frames_prepared`, from `n_equil` and `n_prod` in `meta["extra"]`; a record with neither is
refused), `T` and the frame interval come from the metadata.

| parameter | meaning |
|---|---|
| `system` | the declared system whose farm holds the state point |
| `tag` | the state point's `farm_dir` or run tag |
| `sigma` | the coarse-graining width, in the coordinates' length unit; recorded, not applied |
| `k_cut` | the mode cutoff, in the reciprocal length unit |
| `ordering` | `lexicographic` (the enclosing cube, first axis slowest) or `shell` (ascending wavenumber) |
| `route` | `dense` (the whole phase matrix, on a GPU when one is visible) or `separable` (per-axis factors, on the CPU) |

    aipf.pipeline.modes.modes_from_dump(dump, *, sigma, k_cut, atom_types, fields, ordering, route,
                                        T, dt_frame, skip_frames, composition=None,
                                        cache_dir=None, entry_dir=None) -> ModesRecord

| parameter | meaning |
|---|---|
| `dump` | the trajectory, a LAMMPS text dump |
| `sigma`, `k_cut`, `ordering`, `route` | as `modes` |
| `atom_types` | the dump types summed, one per column before combination |
| `fields` | `"per_type"`, or the combinations `check_fields` returns |
| `T` | the run's temperature (recorded) |
| `dt_frame` | the time between stored frames, ps |
| `skip_frames` | leading frames dropped as preparation; no default, negative refused |
| `composition` | label scalars stored beside the modes, or `None` |
| `cache_dir` | keep one entry per provenance, under a key |
| `entry_dir` | keep exactly this directory, replaced when the provenance differs |

With neither `cache_dir` nor `entry_dir` nothing is written; both is refused. The provenance is
every input, the dump's sha256 included: the same inputs read the stored file back, other inputs (a
new `k_cut`, a rewritten dump) recompute and replace it.

`defaults["mode_fields"]` is `"per_type"` (one channel per species, the sum over its dump type) or
one combination per species in channel order, `{"name": species, "weights": {dump type: w},
"mean": m or None}`. A combination stores `sum_t w_t rho_t(k)`, and a `mean` replaces its zero mode
by `mean * V`. The reduced-unit system stores `phi = (rho_1 - rho_2) / 2` with `mean` 0.5:
`({"name": "A", "weights": {1: 0.5, 2: -0.5}, "mean": 0.5},)`.

## The archive layout

What training and the rollouts read, under a tree:

| file | key | shape, type | what |
|---|---|---|---|
| `<tag>/modes.npz` | `rho_k` | `(n_frames, n_modes, n_channels)` complex64 | the amplitudes |
| | `nvec` | `(n_modes, 3)` int16 | the integer labels `n` |
| | `box` | `(n_frames, 3)` float64 | the box edges per frame |
| | `T_K` | float64 | the temperature |
| | `dt_frame_ps` | float64 | the time between frames |
| | the system's `table_keys["x"]`, or `x_left` and `x_right` | float64 | the composition label |
| `<tag>/DATA_QUALITY_WARNING.json` | `valid_frames` (training), the declared `quality_key` (rollouts) | int | optional: the frames after this are not read |

Other keys in an archive are ignored.
