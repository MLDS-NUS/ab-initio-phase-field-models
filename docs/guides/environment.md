# Environment

One conda environment holds everything: this package, MACE, and a LAMMPS built with ML-IAP and its
Python coupling. Three commands build and check it:

    bash env/build-env.sh -p "$AIPF_ENV_PREFIX" --with-lammps   # or -n aipf; --dry-run prints the plan
    conda activate "$AIPF_ENV_PREFIX"
    aipf md doctor

`env/build-env.sh` takes `-p PREFIX` (an environment at that path; `AIPF_ENV_PREFIX` when set) or
`-n NAME` (a named environment, `aipf` by default). With `AIPF_CACHE_DIR` set, conda's and pip's
caches go to `conda-pkgs/` and `pip-cache/` under it. Without `--with-lammps` it stops before LAMMPS,
and `env/build-lammps.sh` can be run on its own later.

## What the script installs, and why in that order

1. `env/environment.yml`: Python 3.12 and pip, nothing of any host, and from conda-forge the
   matplotlib and freetype the figures were drawn with (`figures/common/BUILD.txt`; pip's matplotlib
   wheels carry another freetype, and the pixel tests compare against those renders), PyMuPDF for
   those tests, and what executes the figures notebook. An existing environment is updated from it.
2. `torch==2.6.0`, then `pip install -e ".[md,figures,dev]"`. `pyproject.toml` is the one list of
   Python dependencies. torch is pinned in the script because the published models and the kernels
   below were run with it. For another CUDA build or a CPU-only one, point pip at the index pytorch.org
   gives for your platform with `PIP_EXTRA_INDEX_URL`.
3. `pip install --no-deps mace-torch==0.3.16`. mace-torch declares `e3nn==0.4.4`, and the published
   MACE potential deserialises only under `e3nn==0.5.1`, so no resolver can install the two together.
   `--no-deps` drops all of mace-torch's declared requirements, so the ones it needs at run time
   (torch-ema, matscipy, h5py, gitpython, lmdb, orjson) are pinned in the `[md]` extra with e3nn,
   ase and cuequivariance. Without torch-ema every `import mace.*` fails.
4. On Linux x86_64, the cuequivariance CUDA kernels and `nvidia-cublas-cu12==12.9.2.10`. The kernels
   call `cublasGemmGroupedBatchedEx`, which the cuBLAS torch 2.6.0 pins (12.4.5.8) does not export.
   This comes after the editable install, whose resolve would otherwise put 12.4.5.8 back, so
   `pip check` reports the override, deliberately.
5. The activation hook `etc/conda/activate.d/zz-aipf.sh`: `PYTHONNOUSERSITE=1`, so a stray
   `pip install --user` cannot shadow a pinned version, and the environment's own cuBLAS and nvrtc
   first on `LD_LIBRARY_PATH`. Without that order the kernels do not load, and MACE falls back to the
   reference implementation silently (the same numbers at about 2.5 times the cost). A script that
   calls `$PREFIX/bin/python` without activating does not read the hook. `aipf md doctor` and the
   job scripts set the same order themselves.

## LAMMPS

    bash env/build-lammps.sh --prefix "$AIPF_ENV_PREFIX" [--cuda|--no-cuda] [--arch AMPERE80] [--jobs N]

builds the pinned source (LAMMPS "10 Dec 2025", the develop commit the published molecular
dynamics ran with) with ML-IAP and its Python coupling, PYTHON, KOKKOS,
and the packages the built-in decks use (BROWNIAN, EXTRA-PAIR), serial and shared. It installs
`$PREFIX/bin/lmp`, `$PREFIX/lib/liblammps.so` and the `lammps` python module into the environment,
then prints the lines for `aipf.toml`. Kokkos uses CUDA when `nvcc` is on `PATH` (`--arch` is read from
the visible GPU, or given where none is visible), and OpenMP on the host with `--no-cuda`. A CUDA build
runs the pair-potential decks anywhere, GPU or not: only its Kokkos (`-k on`, the MACE route) needs a
GPU. The MACE route runs on a GPU only, because its ML-IAP coupling refuses host arrays (and, with that
refusal lifted, the model evaluation crashes). So `--no-cuda` is only for a site that cannot compile
CUDA, and such a build serves the pair-potential decks. The
libraries are found through an RPATH that names the environment's own `lib` first, so an OpenMP or
C++ runtime that a site module puts on `LD_LIBRARY_PATH` is not picked up instead. `--source DIR`
uses an existing source tree at that version, read only. The source and build tree are kept under
`$PREFIX/share/aipf-lammps`.

## The site section of aipf.toml

The programs and files on your machine are declared once, in `aipf.toml` at the repository root
(gitignored; copy `aipf.toml.example`) or as `AIPF_*` variables. A variable wins over the file,
and the file over the default:

    [site]
    lammps = "/path/to/env/bin/lmp"                 # AIPF_LAMMPS
    # lammps_python = "/path/to/python"             # AIPF_LAMMPS_PYTHON: only if not this interpreter
    # latexmk = "/path/to/latexmk"                  # AIPF_LATEXMK: typesets figure labels
    # cuda_lib = "/path/to/cuda/lib64"              # AIPF_CUDA_LIB: a site CUDA library directory
    # pbs_queue = "<queue>"                         # AIPF_PBS_QUEUE, and pbs_project, pbs_gpus,
    # pbs_walltime_h = 24                           # pbs_ncpus, pbs_mem: what a --pbs job asks for

    [site.mace_potential]                           # a MACE model converted for ML-IAP, per system
    feb = "/path/to/feb-model-mliap.pt"             # AIPF_MACE_POTENTIAL_FEB
    hhe = "/path/to/hhe-model-mliap.pt"             # AIPF_MACE_POTENTIAL_HHE

One potential for every system is the plain key `mace_potential = "..."` under `[site]`, or
`AIPF_MACE_POTENTIAL`; TOML holds either that key or the table, not both. `aipf md doctor` reads the
plain key. The full list, with the locations (`[paths]`, `[paths.raw]`), is in
[../reference/cli.md](../reference/cli.md#configuration).

## aipf md doctor

    aipf md doctor [--device auto|cpu|cuda] [--deep]

reads the site and checks, by running them: `import lammps` (`python_module`), the `mliap` pair style
in the module and the binary (`mliap_style`), the declared potential loads (`potential_loads`), two
atoms of its elements run ten steps on the device (`ten_steps`), a CUDA Kokkos backend and a visible
device (`kokkos_device`, on `cuda`), the fast kernels import (`accelerator_kernels`, with `--deep`),
and the binary lists every style of every deck family (`deck_styles:<family>`). Each check is ok,
degraded, broken (with what to do), not inspected (with why), or n/a: it does not apply to what the
site declared or the device asked for. Exit 1 only when a check breaks, or could not inspect, a route
the site needs; n/a never fails. `--device auto` takes the GPU when one is visible. On `cpu` the model
is never started: `[n/a] ten_steps: ML-IAP runs on cuda only, use --device cuda`. With no
`mace_potential` declared, `potential_loads` and `ten_steps` are n/a, and so is a binary or module
without the ML-IAP pieces, so a pair-potential-only site passes on a CPU.

## Exact versions

`env/environment-lock.yml` is `conda env export --no-builds` of an environment built this way. The
lines pip cannot resolve together (mace-torch, the kernels, their cuBLAS) and the two built here (the
`lammps` module, this package) are comments in it, installed afterwards as above:

    conda env create -n aipf-lock -f env/environment-lock.yml
    conda run -n aipf-lock pip install --no-deps -e .
    conda run -n aipf-lock pip install --no-deps mace-torch==0.3.16 \
        cuequivariance-ops-cu12==0.9.1 cuequivariance-ops-torch-cu12==0.9.1 \
        nvidia-cublas-cu12==12.9.2.10

then the activation hook (step 5 of `env/build-env.sh`, which writes it; re-running the whole script
would also move the unpinned packages off the lock) and `env/build-lammps.sh --prefix` that
environment.

The tests find the environment as `AIPF_ENV_PREFIX`, else the interpreter running them.
