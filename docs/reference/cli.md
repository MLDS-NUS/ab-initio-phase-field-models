# Command line

`aipf` is one console script (`aipf.cli.main:main`). Every subcommand takes `--system`, a system
name (a folder under the experiments directory) or a path to a folder holding `system.py`. What a
command does not take as a flag comes from that system's declaration ([system.md](system.md)).

    aipf [--version] <command> ...

| command | does | reference |
|---|---|---|
| `aipf md run` | fills a built-in deck for one state point, pre-flights it, runs or submits it | [md.md](md.md) |
| `aipf md doctor` | checks that this site runs the molecular dynamics | [md.md](md.md#aipf-md-doctor) |
| `aipf data build` | indexes a system's raw MD tree into the data farm | [data.md](data.md) |
| `aipf data rebuild` | recreates the farm's `md` links from its manifest | [data.md](data.md#the-manifest) |
| `aipf modes` | extracts one state point's Fourier modes | [data.md](data.md#modes) |
| `aipf train` | trains the system's declared functional | [training.md](training.md) |
| `aipf diagnose` | runs diagnostic stages over one checkpoint | [diagnose.md](diagnose.md) |
| `aipf rollout` | integrates a trained functional from an archived run | [rollout.md](rollout.md) |

## Exit codes

| code | meaning |
|---|---|
| 0 | done; the last line printed is the output (a directory, a file, or `ok`) |
| 1 | `aipf md run`: the engine ran and failed, or the pre-flight of a `--dry-run` did not pass; `aipf md doctor`: a check the site needs is broken or could not be inspected |
| 2 | refused before anything ran: a usage error, an undeclared system, location or site fact, an `aipf.toml` that is not valid TOML, or a declaration that is missing a key. The reason is on stderr and names the key, flag or variable to set |

## Configuration

Locations and site facts are resolved at call time in one order: the environment variable, then
`aipf.toml` at the repository root (untracked; copy `aipf.toml.example`), then the default. A
relative path in `aipf.toml` is relative to the repository root, an empty value counts as unset and
`~` is not expanded. Every variable wins over every `aipf.toml` key.

### Locations (`aipf.paths`)

| variable | `aipf.toml` key | what | default |
|---|---|---|---|
| `AIPF_DATA` | `[paths] data` | the data farm, `<data>/<system>/{md,modes,ckpt,diagnose,rollout}` | `<repo>/data` |
| `AIPF_RAW_<SYSTEM>` | `[paths.raw] <system>` | one system's raw root: its MD archives, only read | none: a command that needs it refuses, naming both |
| `AIPF_RAW` | `[paths] raw_parent` | a directory holding one raw root per system, `<parent>/<system>` | none |
| `AIPF_EXPERIMENTS` | `[paths] experiments` | the folders `aipf --system NAME` looks in | `<repo>/experiments` |

The raw root of system `s` is the first of `AIPF_RAW_S`, `$AIPF_RAW/s`, `[paths.raw] s`,
`[paths] raw_parent/s`, then the system's own `Paths(raw_default=...)` (none of the shipped systems
declares one). The published checkpoints are tracked files,
`<repo>/data/<system>/ckpt/published[/<variant>]/final.ckpt`, and so is the training sample,
`<repo>/data/<system>/sample/`; `AIPF_DATA` moves neither.

### Site facts (`aipf.site`)

| fact (key under `[site]`) | variable | what |
|---|---|---|
| `lammps` | `AIPF_LAMMPS` | the LAMMPS binary `aipf md run` and `aipf md doctor` use |
| `lammps_python` | `AIPF_LAMMPS_PYTHON` | the interpreter LAMMPS's python module lives in, when it is not this one |
| `latexmk` | `AIPF_LATEXMK` | the `latexmk` that typesets figure labels |
| `cuda_lib` | `AIPF_CUDA_LIB` | a site CUDA library directory put on the loader path |
| `mace_potential` | `AIPF_MACE_POTENTIAL` | a MACE model converted for ML-IAP (a `.pt`), for every system |
| `pbs_queue` | `AIPF_PBS_QUEUE` | the batch queue of a `--pbs` job (text) |
| `pbs_project` | `AIPF_PBS_PROJECT` | the project code the batch server charges (text) |
| `pbs_gpus` | `AIPF_PBS_GPUS` | GPUs a job asks for (integer) |
| `pbs_walltime_h` | `AIPF_PBS_WALLTIME_H` | a job's wall-clock limit in hours (number); `--walltime-h` wins |
| `pbs_ncpus` | `AIPF_PBS_NCPUS` | CPU cores a job asks for (integer, optional) |
| `pbs_mem` | `AIPF_PBS_MEM` | memory a job asks for, in the server's spelling, e.g. `"110gb"` (optional) |

`mace_potential` may also be declared per system, and the per-system value wins:
`AIPF_MACE_POTENTIAL_<SYSTEM>`, or a table

    [site.mace_potential]
    feb = "/path/to/feb-model-mliap.pt"
    hhe = "/path/to/hhe-model-mliap.pt"

TOML holds one of the two spellings at a time: the plain `mace_potential` key or the table. A
missing fact is refused once, listing every missing fact with its variable and key. A value that
does not parse as its type (`pbs_gpus = "two"`) is refused the same way.

### Other variables

| variable | read by | what |
|---|---|---|
| `AIPF_ENV_PREFIX` | `env/build-env.sh`, the environment tests | where the conda environment is built; else a named environment `aipf` |
| `AIPF_CACHE_DIR` | `env/build-env.sh` | the parent of conda's and pip's caches |
| `AIPF_WALLTIME_S` | the job, set by the scheduler | the job's wall-time budget in seconds |

## aipf md run

    aipf md run --system S --template NAME --out DIR [--declared FILE] [--set KEY=VALUE]...
                [--lammps BIN] [--campaign NAME] [--device {auto,cpu,cuda}]
                [--pbs] [--walltime-h H] [--dry-run]

| flag | meaning |
|---|---|
| `--template` | a built-in deck by name: `cube-npt`, `slab-npt-z`, `eos-npt`, `cube-overdamped`, `quench-overdamped`, `slab-overdamped`, `vext-nvt` |
| `--out` | the run directory, created |
| `--declared FILE` | JSON with `point`, `values` and optionally `dt_equil_ps` and `campaign`; without it the system's `defaults["md"][TEMPLATE]` |
| `--set KEY=VALUE` | repeatable; a request field (`geometry`, `ensemble`, `T`, `x`, `P`, `dt_ps`, `equil_ps`, `prod_ps`, `dump_every_ps`, `dump_from`, `seed`, `n_atoms`, `parent`) or `dt_equil_ps`, read as JSON; any other key is a deck value |
| `--lammps` | the binary; default the site's `lammps` |
| `--campaign` | the system's engine-settings campaign (`MdSettings`), required when it declares any; default the declaration's `campaign` |
| `--device` | where a machine-learned-potential deck runs. That route runs on a GPU only: `cpu` is refused, `auto` and `cuda` need a visible GPU (or `--pbs`). A pair-potential deck runs on the host whatever this says. Default `auto` |
| `--pbs` | write `OUT/job.pbs`, which runs the same command without `--pbs` in this environment, and submit it; needs `pbs_queue`, `pbs_project`, `pbs_gpus`, `pbs_walltime_h` |
| `--walltime-h` | with `--pbs`: the job's limit in hours |
| `--dry-run` | write the deck and the pre-flight (and with `--pbs` the job file); run and submit nothing |

Prints the deck path, the pre-flight, the launch command, then the job file and job id (`--pbs`)
or the result (`ok` or the failure).

## aipf md doctor

    aipf md doctor [--device {auto,cpu,cuda}] [--deep]

Checks the declared site; `--deep` also imports the accelerated kernels. Exit 0 when every check
the site needs is `ok`, `degraded` or `n/a`. See [md.md](md.md#aipf-md-doctor).

## aipf data build, aipf data rebuild

    aipf data build --system S [--dry-run]
    aipf data rebuild --system S

`build` indexes the raw root into `<data>/<system>/` and prints `indexed N state points, skipped M`,
then one `SKIP <tag>: <reason>` line per skipped run. `--dry-run` writes nothing. A farm whose
`manifest.json` indexes another raw root is refused (exit 2) rather than replaced. `rebuild`
recreates the `md` links and `meta.json` files from `manifest.json` and prints
`rebuilt N state points`.

## aipf modes

    aipf modes --system S --tag TAG --sigma SIGMA --k-cut K [--ordering {lexicographic,shell}]
               [--route {dense,separable}]

`--tag` is a run tag or a farm directory (`<partition>/<tag>`) from the manifest; an unindexed tag
is refused naming the farm. `--sigma` is recorded with the archive, not applied. Prints the
directory holding `modes.npz`. See [data.md](data.md#modes).

## aipf train

    aipf train --system S --run NAME --seed N (--steps N | --epochs N)
               --source NAME=SUBDIR:PATTERN:GX,GY,GZ [--source ...] | --source declared | --source sample
               --resume-optimizer {yes,no} --anchors {declared,none}
               [--variant NAME] [--init-from-published] [--log-every-step]
               [--device {auto,cpu,cuda}] [--deterministic] [--projection {kz0-volumetric,kz0-areal}]
               [--pbs] [--walltime-h H] [--dry-run]

| flag | meaning |
|---|---|
| `--run` | the run directory's name under `<data>/<system>/ckpt/`; `published` is refused |
| `--seed` | the one seed the model, batch-order and draw streams derive from |
| `--steps`, `--epochs` | the run's length, exactly one |
| `--source` | a training source: SUBDIR is relative to the declared `source_root`, PATTERN a glob of run directories, the grid the real-space grid it is scattered onto. Repeatable. `declared` alone trains the system's `defaults["training"]["sources"]`, exclusions included. `sample` alone trains on the sample bundled in the checkout, `data/<system>/sample` (`defaults["training"]["sample"]`), with the anchor tables read from it and no raw root; it exercises the pipeline and gives a model with no physics in it ([../guides/workflow.md](../guides/workflow.md#7-training-on-the-bundled-sample)) |
| `--resume-optimizer` | `no` builds the optimizer and schedule fresh; `yes` resumes them from `--init-from-published`'s checkpoint when its parameter groups fit this model's optimizer in count, shape and the order of the parameter names it records, else exit 2 naming what differs. `yes` can only succeed on a checkpoint that `aipf train` wrote; the published checkpoints record no names, so start from them with `no`. Required |
| `--anchors` | `declared` trains the system's anchor tables; `none` the drift term alone. Required |
| `--variant` | a declared variant ([system.md](system.md#variant)) |
| `--init-from-published` | start from the system's (or variant's) published weights, digest-checked |
| `--log-every-step` | also write `steps.json`, the loss per step |
| `--device` | `auto` (cuda when torch sees a GPU), `cpu` or `cuda`; `cuda` without a GPU is refused before the run directory exists |
| `--deterministic` | ask torch for deterministic kernels for this run |
| `--projection` | train a two-dimensional model (a factory model on a two-axis grid) on the `k_z = 0` plane of the archives: `kz0-volumetric` divides by `V_ref`, `kz0-areal` by `Lx_ref Ly_ref` ([training.md](training.md#two-dimensions)). Every `--source` grid is then `GX,GY`, `--anchors none` is required, and the archives must record their reference cell. Off by default |
| `--pbs`, `--walltime-h`, `--dry-run` | as `aipf md run`; the job file is `job.pbs` in the run directory, and `--dry-run` needs `--pbs` |

Before anything is written the command refuses a source whose glob selects no run directory (after
its exclusions), `--resume-optimizer yes` without `--init-from-published` or with a saved optimizer
whose parameter groups do not fit this model's, a published checkpoint it cannot find or whose
digest differs, a `GX,GY` grid without `--projection` and a three-length one with it, `--projection`
with `--anchors declared`, and a model whose number of axes the projection or the grids contradict. Prints the run directory. See
[training.md](training.md).

## aipf diagnose

    aipf diagnose --system S --ckpt {FILE|published} --stage STAGE [--stage ...]
                  [--variant NAME] [--out DIR] [--pressure P]...
                  [--T-grid LO,HI,STEP | --T-grid-n LO,HI,N] [--override-declared]

`--stage` is one of `phase_diagram`, `dome`, `tc`, `kappa`, `stability_map`,
`one_field_phase_diagram`, repeatable, at least one. `--ckpt published` is the system's tracked
checkpoint, digest-verified. `--T-grid` is inclusive (`np.arange(LO, HI + 1e-9, STEP)`), `--T-grid-n`
is `np.linspace(LO, HI, N)`. The output is `<out>/<md5[:12]>/`, default `<data>/<system>/diagnose/`,
and the command prints it. See [diagnose.md](diagnose.md).

## aipf rollout

    aipf rollout {spinodal,slab} --system S --ckpt {FILE|published} --run RUN --seeds [N ...]
                 --t-end T --dt DT --save-ps DT_SAVE --device DEV --out {DIR|data} [--variant NAME]

Every flag is required. `--seeds` with no value is one deterministic rollout; each seed is one noisy
rollout. Times are in ps. `--device` is a torch device (`cpu`, `cuda`, `cuda:1`). `--out data` writes
under `<data>/<system>/rollout/`. Prints the output directory. See [rollout.md](rollout.md).

## --variant

`aipf train`, `aipf diagnose` and `aipf rollout` take `--variant NAME`, a model the system declares
beside its production one. Its functional, mobility, checkpoint and `defaults` entries replace the
production ones, so `--ckpt published` and `--init-from-published` read the variant's tracked file.
An undeclared name is refused, listing the declared ones. Example:

    aipf diagnose --system lj --variant fh --ckpt published --stage one_field_phase_diagram
