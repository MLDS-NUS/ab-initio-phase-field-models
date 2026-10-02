# Molecular dynamics

`aipf md run` makes one state point of a system: it fills a built-in LAMMPS deck from a request,
the system's declaration and the caller's values, writes it, checks the binary against every style
it names, and runs it here or submits it as a batch job. `aipf md doctor` checks that the site can.

## The request

`aipf.md.request.StatePoint` says what to simulate. Nothing is defaulted except `P`, `n_atoms` and
`parent`.

| field | meaning |
|---|---|
| `geometry` | `cube`, `slab`, `ball`, `column`, `vext`, `eos`, `quench` |
| `ensemble` | `NVE`, `NVT`, `NPT`, `NPT_z`, `langevin_overdamped` |
| `T` | the target temperature, in the system's unit |
| `x` | the fraction of channel `table_keys["x_channel"]`; a tuple, one per region, for a build made of regions |
| `dt_ps` | the production timestep, ps |
| `equil_ps`, `prod_ps` | the preparation and production lengths, ps |
| `dump_every_ps` | the trajectory cadence, ps; `None` exactly when `dump_from` is `none` |
| `dump_from` | `prod`, `equil` (keeps the preparation frames) or `none` |
| `seed` | the engine's seed |
| `P` | the target pressure in GPa; required by `NPT` and `NPT_z`, `None` otherwise |
| `n_atoms` | the atom count, when the configuration builds a given number |
| `parent` | the tag a continuation extends (then `equil_ps` is 0) |

Archived spellings are normalised (`npt_iso` to `NPT`, `brownian` to `langevin_overdamped`,
`spinodal_cube` to `cube`, ...). A request's tag is
`<geometry>_x<x>_T<T>[_P<P>]_s<seed>[_ext<prod_ps>]`, for instance `cube_x0.50_T1800_P0_s1`.

## The declaration

`aipf md run` reads `defaults["md"][TEMPLATE]` of the system, or the JSON file `--declared` names:

| key | meaning |
|---|---|
| `point` | the request's fields |
| `values` | the deck's values the request and the system do not determine |
| `dt_equil_ps` | the preparation timestep, when it differs from `dt_ps` (optional) |
| `campaign` | the system's `md_settings` campaign the run belongs to (optional; `--campaign` wins) |

`--set KEY=VALUE` wins over both: a request field or `dt_equil_ps` by name (read as JSON), any
other key a deck value (as text). A missing request field or an unknown key is refused.

`defaults["md"]["potential"] = {"md5": ...}` declares the machine-learned potential the system's
data were generated with. Its file is a site fact (`AIPF_MACE_POTENTIAL_<SYSTEM>`,
`[site.mace_potential] <system>`, or `AIPF_MACE_POTENTIAL` for every system), checked against the
md5 and against the system's typed species before anything is written.

## Decks

| name | geometry, ensemble | stages |
|---|---|---|
| `cube-npt` | `cube`, `NPT` | prepare at `T_PREP`, then barostatted production |
| `slab-npt-z` | `slab`, `NPT_z` | the same, the barostat on z only |
| `eos-npt` | `eos`, `NPT` | equation-of-state points, averages written to `THERMO_FILE` |
| `cube-overdamped` | `cube`, `langevin_overdamped` | Langevin preparation, then Brownian production |
| `quench-overdamped` | `quench`, `langevin_overdamped` | as `cube-overdamped`, from a hot configuration |
| `slab-overdamped` | `slab`, `langevin_overdamped` | Brownian production only |
| `vext-nvt` | `vext`, `NVT` | Langevin dynamics in an external field (`EXTERNAL`) |

A deck is text with `@NAME@` holes. Who fills them, the later winning:

| source | holes |
|---|---|
| the request and the system | `T`, `T_PREP`, `T_PREP_END` (from `preparation`: `melt_T` held, or a ramp from `T + ramp_offset` to `T`, or `T`), `SEED`, `DT`, `DT_EQUIL`, `N_EQUIL`, `N_PROD`, `N_DUMP`, `DUMP_EQUIL`, `DUMP_PROD` (the dump block in the stage `dump_from` names), `MASSES` (one line per type) |
| the campaign (`MdSettings`) | `UNITS`, `SKIN`, `NEIGH_MODIFY`, `N_THERMO`, `TDAMP`, `PDAMP`, and `P` when the request sets one |
| the request's counts | `N_ATOMS`, `X` |
| the declared potential | `PAIR` (`pair_style mliap/kk unified EXISTS`, the model handed to the coupling in process, and `pair_coeff * * <species in type order>`) |
| the declaration and `--set` | everything else: `CONFIGURATION`, `RELAX`, `TRAJECTORY`, `WRITE_END`, `PAIR` for a pair potential, `DAMP`, `GAMMA`, `EXTERNAL`, `THERMO_FILE`, `THERMO_COLUMNS`, `SAMPLE_PROD`, ... |
| the deck's own defaults | `DUMP_MODIFY`, `DUMP_RESET`, and in `cube-npt` `THERMO_COLUMNS`, `SAMPLE_PROD`, `SAMPLE_PROD_END` |

A hole nothing fills is refused, naming it. A request the deck does not describe (another geometry
or ensemble, a preparation stage of length zero, a trajectory without a cadence) is refused too.

## Routes

| route | when | runs |
|---|---|---|
| `input_file` | the values give `PAIR` (a pair potential) | the binary on `in.lammps`, a separate process, on the host |
| `in_process` | the system declares a potential and the values give no `PAIR` | LAMMPS's python module in this interpreter through the ML-IAP coupling, with Kokkos on one GPU |

The `in_process` route computes on a GPU only: `--device cpu` is refused, and `auto` or `cuda` with
no visible GPU is refused unless the run is submitted with `--pbs`. Both refusals are exit 2, with
nothing written.

## What a run writes

| file | what |
|---|---|
| `in.lammps` | the filled deck |
| `log.lammps` | the engine's log |
| `run.json` | the system, template, campaign, request, values, command, route, device, potential (path, sha256, species, dtype, cutoff), the pre-flight, the job, the result |
| the trajectory and the files the deck writes | `TRAJECTORY`, and for instance `final.data`, `thermo_prod.txt` |
| `job.pbs`, `job.log`, `job.json` | with `--pbs`: the job script, its output and the submission record |

The pre-flight (printed, and stored in `run.json`) is two checks: `allocator`
(`PYTORCH_CUDA_ALLOC_CONF` must not set `expandable_segments`) and `deck_styles` (the binary lists
every style the deck uses). A run that fails it is not started.

`aipf data build` reads `run.json`, so `--out` may be any directory under the system's raw root
([data.md](data.md)).

## Batch jobs

`--pbs` writes `job.pbs` in the run directory. It changes to that directory, activates the
environment the command was submitted from (with its activation hooks), exports the site facts the
submission resolved (`AIPF_LAMMPS`, `AIPF_MACE_POTENTIAL_<SYSTEM>`) and runs the same command
without `--pbs`. It asks for one chunk, `select=1:ngpus=<pbs_gpus>[:ncpus=<pbs_ncpus>][:mem=<pbs_mem>]`,
in queue `pbs_queue`, charged to `pbs_project`, for `pbs_walltime_h` hours (`--walltime-h` wins). The
command prints the job file and the job id. `aipf train --pbs` writes the same kind of job.

## aipf md doctor

    aipf md doctor [--device auto|cpu|cuda] [--deep] [--system NAME]

| check | asks |
|---|---|
| `python_module` | `import lammps` starts in the declared interpreter (`lammps_python`, else this one), and which packages it has |
| `mliap_style` | the module and the binary list the `mliap` pair style |
| `potential_loads` | the `mace_potential` each system runs loads: `Site.for_system` (`AIPF_MACE_POTENTIAL_<SYSTEM>`, `[site.mace_potential] <system>`, then the plain fact), for every system that declares an ML-IAP potential (`defaults["md"]["potential"]`) or for `--system NAME`. One file for all of them is one `potential_loads`/`ten_steps` pair; several files are one pair each, suffixed `:<system>` |
| `ten_steps` | two atoms of its elements run ten steps through the `in_process` route on cuda |
| `kokkos_device` | on cuda: Kokkos has a CUDA backend and a device is visible |
| `accelerator_kernels` | with `--deep`: the fast kernels import |
| `deck_styles:<deck>` | the binary lists every style of each built-in deck, and of each system's declared decks (`deck_styles:<system>/<deck>`) |

Each check is `ok`, `degraded` (the right numbers at extra cost), `broken` (with what to do),
`unknown` (not inspected, with why) or `n/a` (it does not apply to what the site declared or the
device asked for). On `cpu` the model is never started: `[n/a] ten_steps: ML-IAP runs on cuda only,
use --device cuda`. With no `mace_potential` declared for the systems checked (and none when no
system declares an ML-IAP potential), `potential_loads` and `ten_steps` are `n/a`, and a broken `python_module` or `mliap_style` is reported as `n/a`, so a site
with only pair-potential decks passes on a CPU. The command exits 1 only when a check the site needs is broken or unknown; `n/a` never fails.
The heavy checks run in one child process, so a crash is a finding, not a failure of the command.

## Python

`aipf.md.run.run(system, *, template, point, values, out, lammps=None, dry_run=False,
campaign=None, dt_equil_ps=None, device="auto", pbs=None, job_command=None)` is the command's
body and returns a `RunRecord` (`deck_path`, `command`, `diagnosis`, `result`, `route`, `job_path`,
`job_id`). `aipf.md.templates.DECKS` maps `(geometry, ensemble)` to each `Deck`; `fill(deck, point,
system, values)` renders one. `aipf.md.doctor.examine_site(site, device=..., deep=..., potential_systems=None)` is the doctor
(`potential_systems=None` checks the plain fact only).
