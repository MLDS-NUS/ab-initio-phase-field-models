# The workflow, from MD to a diagnosed model

This guide runs the whole chain once on the reduced-unit system `lj`: molecular dynamics,
the data index, Fourier modes, training, diagnosis. Every command below was run as written,
from the checkout root. Outputs are quoted from that run, trimmed, with each path shown as the
variable that held it (`$WORK`, `$AIPF_LAMMPS`). The same chain runs as a test,
`tests/test_workflow_lj.py`. Section 7 trains each system on the small sample bundled with the
repository, and section 8 runs the iron-boron system at full size.
Every command and its flags are in [../reference/cli.md](../reference/cli.md).

## 0. Install and site facts

    git clone https://github.com/mengyi-chen/ab-initio-phase-field-models.git
    cd ab-initio-phase-field-models
    bash env/build-env.sh          # or: pip install -e . into an environment you have
    pip install -e ".[md]"         # the MD stack; LAMMPS itself is a separate compile

```text
$ aipf --version
0.1.0
```

Three kinds of setting come from outside the package:

- **Site facts** have no default: interpreters, the LAMMPS build, the figure toolchain.
  `AIPF_LAMMPS` names the LAMMPS build this guide uses (it needs the `brownian` and
  `lj/smooth/linear` styles). Each is an `AIPF_*` variable or a key under `[site]` in
  `aipf.toml` (copy `aipf.toml.example`); the full list is in
  [../reference/cli.md](../reference/cli.md#configuration).
- **Raw roots**: each system's MD archives, read and never written, `AIPF_RAW_<SYSTEM>` or
  `[paths.raw] <system>` in `aipf.toml`. A command that needs one and finds none refuses,
  naming both.
- **Where data goes**: `AIPF_RAW_<SYSTEM>` is the raw directory `aipf data build` reads, and
  `AIPF_DATA` is the farm every command writes (default: the checkout's `data/`). The
  published checkpoints are not the farm: they are tracked in the checkout, under
  `data/<system>/ckpt/published/`, and read there whatever `AIPF_DATA` says.

This guide keeps its own MD tree and farm in a directory of its own, so nothing it writes
touches the checkout's `data/`:

    export WORK=<an empty directory of yours>
    export AIPF_RAW_LJ=$WORK/lj-md        # the new MD runs; aipf data build reads it
    export AIPF_LAMMPS=<your lmp>         # the LAMMPS build aipf md run uses
    export AIPF_DATA=$WORK/data           # the farm this session writes
    export OMP_NUM_THREADS=1              # bit-for-bit runs need one thread

Set both before `aipf data build`. Without `AIPF_DATA` the farm is the checkout's `data/`: in a
fresh clone `data/lj/` holds only `ckpt/published/`, so `aipf data build --system lj` writes
`data/lj/manifest.json` and `data/lj/md/` into the checkout. A farm indexes one raw root: once its
`manifest.json` indexes one, a build that reads another is refused and nothing is written:

```text
$ AIPF_RAW_LJ=<another raw root> aipf data build --system lj
aipf data build: $WORK/data/lj/manifest.json indexes the raw root $WORK/lj-md, and this build
reads <another raw root> (AIPF_RAW_LJ, AIPF_RAW or [paths.raw] in aipf.toml). Building would replace that index. ...
```

Every refusal here is exit code 2 with the reason, naming the key or setting. Tests: `pytest`
runs everything but the long runs, which `-m slow` selects; a check that needs a raw root, the
simulator or a GPU is marked `env` and skips, naming what is missing, where the machine lacks it.

## 1. MD data

A system declares the MD decks it has run under `defaults["md"][<template>]`: the state point
and every value the deck needs (`experiments/lj/system.py`, `_MD_CUBE_OVERDAMPED`, 4000 atoms,
5 × 10⁶ overdamped steps). `--declared FILE` replaces that declaration with a JSON file.
`--set KEY=VALUE` changes one value: a state-point field (`prod_ps`, `n_atoms`, ...) or
`dt_equil_ps` by name, a deck value otherwise. This guide shrinks the box in a JSON file and
shortens the run on the command line:

    python - <<'EOF'
    import json, os
    from aipf.system import load
    d = load("lj").defaults["md"]["cube-overdamped"]
    d["point"]["n_atoms"] = 500
    d["values"]["CONFIGURATION"] = d["values"]["CONFIGURATION"].replace(
        "0 10 0 10 0 10", "0 5 0 5 0 5")
    json.dump(d, open(os.path.join(os.environ["WORK"], "tiny.json"), "w"), indent=1)
    EOF

That is 5³ FCC cells (500 atoms), 1000 preparation steps at `melt_T = 2.0`, then 50000
overdamped steps at T = 1.6, dumped every 100. `--dry-run` writes the deck and checks the
binary against every style it names:

```text
$ aipf md run --system lj --template cube-overdamped --declared $WORK/tiny.json \
    --set equil_ps=1 --set prod_ps=10 --set dump_every_ps=0.02 \
    --out $AIPF_RAW_LJ/tiny --dry-run
$WORK/lj-md/tiny/in.lammps
[ok           ] allocator: PYTORCH_CUDA_ALLOC_CONF is not set
[ok           ] deck_styles: $AIPF_LAMMPS lists every style the deck uses: atomic, lj/smooth/linear, nve, langevin, brownian, recenter
2 ok, 0 degraded, 0 broken, 0 not inspected
$AIPF_LAMMPS -screen none -log log.lammps -in in.lammps
```

The same command without `--dry-run` runs it (24 s on one core) and prints `ok` last. The run
directory holds `in.lammps`, `log.lammps`, `dump.lammpstrj` and `run.json` (the request, the
pre-flight and the result). `aipf data build` reads `run.json`, so `--out` can be any
directory under the raw root: the request gives the tag (`cube_x0.50_T1.6_s42`), geometry,
ensemble and composition, and `log.lammps` what the engine did (a run whose log counts other
than the requested `n_atoms` is skipped with the reason).

**The H/He and Fe-B route** is the same command. These systems declare the machine-learned
potential their archive was generated with, by md5, under `defaults["md"]["potential"]`; the file
is a site fact, the ML-IAP conversion (a `.pt`), declared per system (`AIPF_MACE_POTENTIAL_FEB`, or
`feb = "..."` under `[site.mace_potential]`) or once for every system (`AIPF_MACE_POTENTIAL`).
`aipf md run` checks the file against the md5, writes the pair lines from it
(`pair_style mliap/kk unified EXISTS`) and runs the deck in this interpreter through the ML-IAP
coupling. That route computes on a GPU only: `--device cpu` is refused, and so is `--device cuda`
(or `auto`) on a node where no GPU is visible, both with exit 2 and nothing written. Fe-B declares
one archived point of its cube campaign (`defaults["md"]["cube-npt"]`, x_B = 0.5 at 1800 K, 0 GPa);
two atoms for ten steps of each stage take about 40 s on an A100, most of it loading the model:

```text
$ aipf md run --system feb --template cube-npt --set n_atoms=2 --set equil_ps=0.01 \
    --set prod_ps=0.01 --set dump_every_ps=0.001 --device cuda --out $WORK/feb_two_atoms
```

`--pbs` runs it on a batch node instead. It writes `OUT/job.pbs`, which activates this environment
and runs the same command without `--pbs`, submits it to the site's queue and prints the job id
(`job.json` records it). The queue, the project code, the GPUs and the wall-clock limit are site
facts (`[site] pbs_queue`, `pbs_project`, `pbs_gpus`, `pbs_walltime_h`, optionally `pbs_ncpus` and
`pbs_mem`; `--walltime-h` on the line wins). `--pbs --dry-run` writes the deck and the job file and
submits nothing. `aipf md doctor --device cuda` checks the machine first.

## 2. The farm

```text
$ aipf data build --system lj
indexed 1 state points, skipped 0
```

The farm is one directory per system under `$AIPF_DATA` (default `data/`):

| path | what it is |
|---|---|
| `lj/manifest.json` | file: every run found under the raw root, indexed or skipped with a reason |
| `lj/md/<tag>/traj.lammpstrj`, `thermo.txt` | symlinks to the trajectory and log in the raw tree |
| `lj/md/<tag>/meta.json` | file: the metadata, normalised (T, n_atoms, dt, dump cadence, composition) |
| `lj/modes/<tag>/` | files: `modes.npz` and `provenance.json`, written by `aipf modes` |
| `lj/ckpt/<run>/` | files: a training run's `final.ckpt`, `MANIFEST.json`, `hparams.yaml` |
| `lj/diagnose/<md5[:12]>/` | files: one checkpoint's diagnosis |
| `lj/rollout/<key>/` | files: rollouts (`aipf rollout --out data`) |

The published checkpoints are not in the farm (section 0). The trajectories stay where the MD
wrote them. `meta.json` of the run above records `"T_K": 1.6` (the system's own unit, reduced here),
`"n_atoms": 500`, `"dt_ps": 0.0002`, `"dump_every_ps": 0.02`, `"n_frames": 501` (the first
production step is dumped too) and, under `extra`, `"n_prod": 50000, "n_equil": 1000`.

The published trees are indexed the same way. `--dry-run` writes nothing:

```text
$ AIPF_RAW_LJ=<lj tree> aipf data build --system lj --dry-run
indexed 60 state points, skipped 232
  SKIP homogeneous_brownian_MD_xA_0.50_T_1.70_seed_0: 'no dump cadence: need dump_every_ps, or dump_freq with dt_ps'
  ...
```

`aipf data rebuild --system S` recreates the `md` links and `meta.json` files from
`manifest.json`. See [../reference/data.md](../reference/data.md).

## 3. Modes

```text
$ aipf modes --system lj --tag cube_x0.50_T1.6_s42 --sigma 1.5 --k-cut 3.0
$WORK/data/lj/modes/cube_x0.50_T1.6_s42
```

`--tag` is a tag or farm directory from the manifest. The command extracts the Fourier
amplitudes of every frame after the preparation (`skip_frames` from `n_equil` and `n_prod`)
for every wavevector with |k| ≤ `--k-cut`, on the time-mean box. `--sigma` is recorded with
the archive and not applied: the Gaussian filter `exp(-k²σ²/2)` enters at training and in the
rollouts, from `defaults["sigma"]`. Pass the system's declared values
(`defaults["sigma"]`, `defaults["k_cut"]`); both are required because the archive is
identified by them.

What is stored is the system's declared `defaults["mode_fields"]`. For `lj` it is one field,
φ = ½(ρ_type1 − ρ_type2) with the zero mode set to 0.5 (`{"name": "A", "weights": {1: 0.5,
2: -0.5}, "mean": 0.5}`); a two-species system declares `"per_type"`. Here `modes.npz` holds
`rho_k` of shape (501 frames, 251 modes, 1 field), `nvec`, `box`, `T_K`, `dt_frame_ps` and the
composition label `x_A`. The file sits where an archive keeps a run's file,
`modes/<farm_dir>/modes.npz`, so a training source names the run by its tag. `provenance.json`
records every input, including the dump's path and sha256: a re-run with the same inputs reads
the file back, and one with other inputs (another `--k-cut`, a rewritten dump) replaces it. See
[../reference/data.md](../reference/data.md#modes).

## 4. Train

```text
$ aipf train --system lj --run guide-smoke --seed 0 --steps 2 \
    --source "cube=.:cube_x0.50_T1.6_s42:16,16,16" \
    --resume-optimizer no --anchors declared --log-every-step
`Trainer.fit` stopped: `max_steps=2` reached.
$WORK/data/lj/ckpt/guide-smoke
```

**The command line** gives only what identifies the run: `--run` (the directory name under
`ckpt/`), `--seed`, `--steps` or `--epochs`, one `--source` per data set, `--resume-optimizer`
and `--anchors`, all required. A source is `NAME=SUBDIR:PATTERN:GX,GY,GZ`: SUBDIR and PATTERN
select mode directories under the system's declared source root (`lj`: `$AIPF_DATA/lj/modes`),
and the grid is the real-space grid that source is evaluated on. 16³ on this 7.9 σ box gives
a spacing of 0.50 σ, close to the published 20 × 20 × 80 grid on its 9.5 × 9.5 × 38.1 box.

**The system declares everything else**, in `defaults` and `defaults["training"]`: the model,
the window (half-width 25 frames, stride 20), batch size 16, the split, the run weighting,
the learning-rate schedule, the loss weights, and the anchor tables (the measured mobility
and S_cc(0) per temperature, tracked under `experiments/lj/anchors/`). `--anchors none` trains the drift
term alone.

The run directory holds `final.ckpt`, `MANIFEST.json` (the sources, the terms trained, the
checkpoint's md5), `hparams.yaml` and, with `--log-every-step`, `steps.json`. This run trained
`["L_dyn", "L_M", "L_bulk"]`. A fresh run's first steps sit in the linear warm-up, where the
learning rate is near zero; two steps prove the plumbing, not the model. The loader forks the
declared four workers; in a threaded process (a test) Python warns, and the batches are the same.

**The device** is `--device auto` by default: cuda when torch sees a GPU, else cpu. `--device
cuda` without one is refused before the run directory exists. `--deterministic` asks torch for
deterministic kernels for the run (slower; what a bit-for-bit comparison needs, and what the
N-step comparisons with the archived trainers pass together with `--device cpu`); without it no
torch switch is touched. `--pbs` (and `--walltime-h`, `--dry-run`) submits the run as a batch job,
as for `aipf md run`: the job file is `job.pbs` in the run directory.

`--init-from-published` starts from the published weights (the checkout's tracked file,
digest-checked) instead of fresh ones. `--resume-optimizer yes` resumes the saved optimizer and
schedule too, when the saved parameter groups fit this model's optimizer, in order: a checkpoint
written by `aipf train` records the parameter at each position, so `yes` can only succeed on a
checkpoint that `aipf train` wrote. The published checkpoints record no names, and each holds two
tensors or more of one shape whose order cannot be checked without them (the published `lj`
checkpoint differs in count as well, 15 tensors against 16), so `yes` is refused on them with exit 2
and a message naming what differs. Start from one with `--resume-optimizer no`, as below. A
checkpoint nothing holds is refused before any run directory exists.

```text
$ aipf train --system lj --run from-published --seed 0 --steps 2 \
    --source "cube=.:cube_x0.50_T1.6_s42:16,16,16" \
    --init-from-published --resume-optimizer no --anchors declared
$WORK/data/lj/ckpt/from-published
```

`--variant fh` or `--variant landau` trains a declared baseline. A fresh `lj` run does not
reproduce the published run's initialisation (the kernel's last layer scaled by 0.1, one weight
frozen), so a run meant to match the published one starts from `--init-from-published`. See
[../reference/training.md](../reference/training.md).

## 5. Diagnose

```text
$ aipf diagnose --system lj --ckpt $AIPF_DATA/lj/ckpt/guide-smoke/final.ckpt \
    --stage one_field_phase_diagram
$WORK/data/lj/diagnose/<md5[:12]>
```

The temperatures are the declared `defaults["diagnose"]["one_field"]["T_grid"]` (`lj`:
`linspace(0.5, 2.0, 40)`). `--T-grid LO,HI,STEP` or `--T-grid-n LO,HI,N` may repeat it; another
grid is refused unless `--override-declared` is also given:

```text
$ aipf diagnose --system lj --ckpt published --stage one_field_phase_diagram --T-grid-n 0.5,2.0,41
aipf diagnose: the temperature grid given (41 points, 0.5 to 2) is not the declared
defaults['diagnose']['one_field']['T_grid'] = {'lo': 0.5, 'hi': 2.0, 'n': 40} (40 points). ...
```

The output directory is named by the first 12 hex digits of the checkpoint's md5. It holds
`MANIFEST.json` (the checkpoint, its md5, the declared thresholds and the grid used) and one
file per stage. `one_field_phase_diagram.npz` holds `T`, `binodal_L/R`, `spinodal_L/R` (NaN
where there is no gap), `Tc_exact`, `mu_w0` and the Ising fit. After two steps from fresh
weights there is no gap at any T, as expected.

`--ckpt published` diagnoses the published checkpoint instead (the checkout's tracked file,
digest-checked):

```text
$ aipf diagnose --system lj --ckpt published --stage one_field_phase_diagram
$WORK/data/lj/diagnose/<md5[:12]>
```

This one has a gap at 22 of the 40 temperatures, `Tc_exact` 1.434 and an Ising-fit T_c of
1.507. `--variant fh --ckpt published` diagnoses the Flory-Huggins baseline on its own
declared grid, 1.0 to 1.6 in steps of 0.02 (`$WORK/data/lj/diagnose/<md5[:12]>`).

`--stage` is required and repeatable; the thresholds come from the system's
`defaults["diagnose"]`. Which stage fits which system:

| stage | systems | extra flags |
|---|---|---|
| `one_field_phase_diagram` | `lj` | none: the grid is declared (a flag must equal it, or `--override-declared`) |
| `kappa` | `hhe`, `feb` | none |
| `phase_diagram` | `hhe`, `feb` | `--pressure P` (repeatable) and a T grid |
| `stability_map` | `feb` | `--pressure P` (repeatable) and a T grid |
| `dome`, `tc` | `hhe` | none: the grids are declared |

```text
$ aipf diagnose --system hhe --ckpt published --stage kappa
$WORK/data/hhe/diagnose/<md5[:12]>
```

needs no site data and writes `kappa.json` (`kappa_eff`, a 2 × 2 matrix). See
[../reference/diagnose.md](../reference/diagnose.md).

## 6. Rollouts and figures

`aipf rollout spinodal` rolls the trained functional from frame 0 of an archived cube and
compares with that run; `aipf rollout slab` does the same from a two-phase slab. The solver,
the archive layout and the grid come from `defaults["rollout"]`, the noise from
`System.noise`, the density edges from `System.trust_domain`. Only `hhe` declares them;
`lj` and `feb` are refused (`aipf rollout spinodal: system 'lj' declares no
defaults['rollout']: the solver knobs [...]`, every missing key listed). A short deterministic
H/He run (no `--seeds` value: no noise):

```text
$ aipf rollout spinodal --system hhe --ckpt published --run cube_x0.50_T07000 --seeds \
    --t-end 0.2 --dt 1e-4 --save-ps 0.02 --device cuda --out data
$WORK/data/hhe/rollout/<key>
```

`--ckpt published` is the same keyword as in `aipf diagnose`. The run needs the H/He raw
tree (`fields/modes_800GPa`). The output holds
`spinodal_cube_x0.50_T07000_det.npz` and `MANIFEST.json`. See
[../reference/rollout.md](../reference/rollout.md).

The H/He rain column is `python -m experiments.hhe.column --ckpt published --geometry
{walled,mirror} ...` from the checkout root (every flag required; `--help` lists them), declared
in `defaults["column"]` ([../reference/rollout.md](../reference/rollout.md#the-hhe-rain-column)).
The paper figures are drawn from committed inputs; `figures/README.md` says how, and what each
needs.

## 7. Training on the bundled sample

Training reads Fourier-mode archives and measured anchor tables, and the archives the published
models were trained on are not part of the repository. So that the training step can still be run
from a clean checkout, each system carries a small sample under `data/<system>/sample/`, a few
megabytes in all:

- a few mode runs (one per declared source for `feb`; one cube per pressure and one slab for
  `hhe`; one per temperature at four temperatures for `lj`), each cut to its first training window and to the modes with
  max |n_i| <= 2. The `lj` runs keep the four windows that fill one batch, since its declared loader
  drops a partial batch;
- the anchor and equation-of-state tables the system's training declaration reads, at the same
  relative paths as under the raw root;
- `MANIFEST.json`, which lists every file with its frames, its mode cut and its md5.

```bash
aipf train --system feb --source sample --anchors declared --resume-optimizer no --seed 0 --epochs 1 --device cpu --run sample
aipf train --system hhe --source sample --anchors declared --resume-optimizer no --seed 0 --epochs 1 --device cpu --run sample
aipf train --system lj  --source sample --anchors declared --resume-optimizer no --seed 0 --epochs 1 --device cpu --run sample
```

`--source sample` takes the sources from `defaults["training"]["sample"]` and reads the anchor tables
from the sample wherever the declaration reads the raw root. Nothing else in the declaration
changes, so every loss term the system declares is trained, and `MANIFEST.json` in the run
directory lists them under `terms_trained`. No raw root and no `aipf.toml` are needed. Each command
runs on a CPU in minutes and prints its run directory, `data/<system>/ckpt/sample/` by default.
`aipf diagnose` reads the result like any other checkpoint:

```bash
aipf diagnose --system feb --ckpt data/feb/ckpt/sample/final.ckpt --stage kappa
aipf diagnose --system hhe --ckpt data/hhe/ckpt/sample/final.ckpt --stage kappa
aipf diagnose --system lj  --ckpt data/lj/ckpt/sample/final.ckpt  --stage one_field_phase_diagram
```

The sample is there so that the pipeline can be exercised from end to end. It is not a smaller
copy of the training data. A model trained on it has seen a window or two of a few runs in a narrow
band of modes, so it carries no physics: its phase diagram, its gradient-energy matrix and its
mobility mean nothing, and no number it gives should be compared with the published models.
Training those again needs the raw MD archive, available from the authors on request. Section 8
does it for iron-boron.

## 8. Fe-B: training the published model again

The iron-boron system `feb` runs the same chain at full size, with its machine-learned potential.
Its commands were run on one node with an A100 (40 GB) and through the batch queue, and the outputs
quoted are from those runs.
Besides the site facts of section 0 it needs the Fe-B potential (`AIPF_MACE_POTENTIAL_FEB`, its md5
checked against the one `experiments/feb/system.py` declares), the `pbs_*` facts for `--pbs`, and,
for training, the Fe-B raw root, the archived MD data the published model was trained on. That
archive is not part of the repository; it is available from the authors on request.

### 8.1 One new run

The system declares one point of its 0 GPa cube campaign under `defaults["md"]["cube-npt"]`:
x_B = 0.5, 1800 K, 0 GPa, 3456 atoms, a 1 fs step, a 10 ps melt at 2600 K
(`PreparationRules(melt_T=2600)`), 50 ps of production dumped every 0.1 ps. This run keeps every
value but shortens the melt to 1 ps and the production to 2 ps (`aipf md doctor --device cuda`
checks the node first):

```text
$ export WORK=<an empty directory of yours>
$ export AIPF_RAW_FEB=$WORK/feb-raw AIPF_DATA=$WORK/feb-data
$ aipf md run --system feb --template cube-npt --set equil_ps=1 --set prod_ps=2 \
    --device cuda --out $AIPF_RAW_FEB/run_x0.50_T1800_P0
$WORK/feb-raw/run_x0.50_T1800_P0/in.lammps
[ok           ] allocator: PYTORCH_CUDA_ALLOC_CONF is not set
[ok           ] deck_styles: $AIPF_LAMMPS lists every style the deck uses: atomic, mliap/kk, npt, recenter, ave/time
2 ok, 0 degraded, 0 broken, 0 not inspected
in process: lammps.lammps(cmdargs=['-screen', 'none', '-log', 'log.lammps', '-k', 'on', 'g', '1', '-sf', 'kk', '-pk', 'kokkos', 'newton', 'on', 'neigh', 'half'])
ok
```

It takes about ten minutes on one GPU. The directory holds `in.lammps`, `log.lammps`,
`run.json`, `traj.dump` (21 frames: step 0 of production and one every 100 steps), `thermo_prod.txt`
(the production averages of temperature, pressure, volume and energy) and `final.data`. Keep the farm
(`AIPF_DATA`) outside the raw root: links inside it would be indexed again by the next build.

The same command with `--pbs --walltime-h 0.5` writes `job.pbs` into the run directory, submits it
and prints the job file and the job id; the job writes the same `in.lammps` byte for byte. A job whose GPU is shared with another process can fail to allocate
device memory (`Kokkos ERROR: Cuda memory space failed to allocate`, in `job.log`); a resubmission
is the remedy.

### 8.2 Index and modes

```text
$ aipf data build --system feb
indexed 1 state points, skipped 0
$ aipf modes --system feb --tag 0GPa/cube_x0.50_T1800_P0_s1 --sigma 2.0 --k-cut 3.0
$WORK/feb-data/feb/modes/0GPa/cube_x0.50_T1800_P0_s1
```

The farm files the run under its pressure, `0GPa/`, with the tag the request gives it. Its
`meta.json` records `"ensemble": "NPT"`, `"T_K": 1800.0`, `"P_GPa": 0.0`, `"n_atoms": 3456`,
`"dt_ps": 0.001`, `"dump_every_ps": 0.1`, `"n_frames": 21` and, under `extra`, `"n_equil": 1000,
"n_prod": 2000`. `modes.npz` holds `rho_k` of shape (21 frames, 15515 modes, 2 channels), complex64,
every mode with |k| <= 3 1/A on the time-mean box (32.32 A after 3 ps; the archived run's 100 ps
average is 32.05 A, and 15155 modes). The keys, types and layout are those of the archived files the
published model was trained on, `fields/modes_<P>GPa_v2/<tag>/modes.npz`. `aipf modes` keeps every
production frame; the archived files drop the first 5 ps after the quench from the melt.

### 8.3 Training on the published partition

The published model was trained on six sources, a non-equilibrium and an equilibrium set at each of
0, 5 and 10 GPa. Each is every `cube_*` run of one archived tree minus a list of excluded runs (572 in
all, 356 runs kept), scattered onto a 32^3 grid, with a drift weight of 15000. They are declared in
`experiments/feb/training_sources.json` and selected by `--source declared`. The archived modes reach
|k| = 3 1/A, which on the longest boxes (33.5 to 36 A) is a label of 16 or 17, past a 32^3 grid; the
training declaration's `"band_k_max": 2.0` keeps the modes with |k| <= 2 1/A in any box the run
visits, and the run trains on those. The raw root must hold `fields/modes_{0,5,10}GPa_v2/`, the
measured tables under `fields/training_derived/` and the equation-of-state tables `eos_{0,5,10}GPa/`
(the archive of section 8's opening paragraph, available from the authors).

```text
$ export AIPF_RAW_FEB=<the Fe-B raw root> AIPF_DATA=$WORK/feb-data
$ aipf train --system feb --run retrain --seed 4 --epochs 60 --source declared \
    --anchors declared --resume-optimizer no --device cuda --log-every-step --pbs --walltime-h 24
$WORK/feb-data/feb/ckpt/retrain/job.pbs
<job id>
```

Without `--pbs` the same command trains in this process. With `--pbs --dry-run` it writes the job
file and submits nothing. Loading the 356 runs needs about 72 GB of host memory, so the job needs
`pbs_mem` of 80 GB or more. One step trains one batch of 6 windows from each source; the largest
source sets the epoch at 216 steps, so 60 epochs are 12960 steps, about two to three hours on one
A100. The run trains `L_dyn`, `L_M`, `L_S`, `L_bulk`, `L_P` and the kernel hinge `L_W` (the declared
`lambda_wpsd`), and `MANIFEST.json` lists them under `terms_trained`. The hinge is part of the recipe,
not an optional regulariser: it is what keeps the k = 0 kernel attractive (see `docs/reference/training.md`).

A fresh run starts from the initialisation `--seed` gives and draws its batches in its own order,
so it does not reproduce the published run bit for bit. Diagnose it with the same two stages as the
published model:

```text
$ aipf diagnose --system feb --ckpt $AIPF_DATA/feb/ckpt/retrain/final.ckpt \
    --stage stability_map --T-grid 1150,2650,25 --pressure 0 --pressure 5 --pressure 10
$ aipf diagnose --system feb --ckpt $AIPF_DATA/feb/ckpt/retrain/final.ckpt \
    --stage phase_diagram --T-grid 1200,2600,50 --pressure 0 --pressure 5 --pressure 10
```

The same two stages with `--ckpt published` diagnose the published model; the `stability_map` stage
reproduces the published Fe-B Gamma map in about 10 minutes on one CPU thread.

## 9. A new system

[new-system.md](new-system.md) builds a system from scratch: the fields of `System`, the blocks of
`defaults` each command reads, and a minimal declaration that trains and diagnoses on the modes of
section 3.
