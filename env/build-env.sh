#!/usr/bin/env bash
# env/build-env.sh -- build the aipf environment: Python, this package, MACE, optionally LAMMPS.
#
#   bash env/build-env.sh [-p PREFIX | -n NAME] [--with-lammps] [--dry-run]
#
#   -p PREFIX      build the environment at that path (conda env create -p).
#                  Default: $AIPF_ENV_PREFIX when set.
#   -n NAME        a named environment in your own conda. Default when neither -p nor
#                  AIPF_ENV_PREFIX is given: -n aipf.
#   --with-lammps  then build LAMMPS (ML-IAP, Python, Kokkos) into the same environment
#                  with env/build-lammps.sh (CUDA when nvcc is found; see its --help).
#   --dry-run      print what would be done and where, then stop; nothing is run.
#   AIPF_CACHE_DIR a directory: conda's package cache and pip's cache go under it
#                  (conda-pkgs/, pip-cache/). Unset: conda and pip keep their defaults.
#
# Steps, in this order (the order matters, see the comments at each):
#   1. conda env create from env/environment.yml (Python, pip, and the conda-forge matplotlib and
#      freetype the figures were drawn with); an existing environment is updated from it instead.
#   2. torch, pinned (TORCH below), then pip install -e ".[md,figures,dev]".
#   3. MACE: pip install --no-deps mace-torch (its runtime packages are pyproject's [md]).
#   4. On Linux x86_64: the cuequivariance CUDA kernels and the cuBLAS they need.
#   5. The activation hook etc/conda/activate.d/zz-aipf.sh (and its deactivate twin).
#   6. With --with-lammps: env/build-lammps.sh --prefix <the environment>.
# Exact versions of a known-good environment: env/environment-lock.yml.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SPEC="$HERE/environment.yml"

# torch is pinned here, not in pyproject.toml (which only sets a floor): the published
# models, e3nn 0.5.1 and the cuequivariance 0.9.1 kernels were run and measured with it.
# For another CUDA build or a CPU-only one, give pip the index pytorch.org names for your
# platform through pip's own variable, PIP_EXTRA_INDEX_URL.
TORCH="torch==2.6.0"
# mace-torch declares Requires-Dist: e3nn==0.4.4, and the published MACE potential only
# deserialises under e3nn==0.5.1, so no resolver can install the two together. mace-torch
# goes in WITHOUT its dependencies; the runtime packages it does need (torch-ema, matscipy,
# h5py, gitpython, lmdb, orjson, with e3nn 0.5.1 and ase) are pinned in pyproject.toml's [md]
# extra, which step 2 installs. Without torch-ema every `import mace.*` fails.
MACE="mace-torch==0.3.16"
# The kernels need cuBLAS >= 12.5 (cublasGemmGroupedBatchedEx); torch 2.6.0 pins 12.4.5.8,
# which does not export it. They go in AFTER the editable install, whose resolve would
# otherwise put 12.4.5.8 back.
KERNELS=("cuequivariance-ops-cu12==0.9.1" "cuequivariance-ops-torch-cu12==0.9.1")
CUBLAS="nvidia-cublas-cu12==12.9.2.10"

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

DRY_RUN=0
WITH_LAMMPS=0
PREFIX_ARG=""
NAME_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        -p|--prefix) PREFIX_ARG="${2:?-p needs a path}"; shift 2 ;;
        -n|--name) NAME_ARG="${2:?-n needs a name}"; shift 2 ;;
        --with-lammps) WITH_LAMMPS=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done
if [ -n "$PREFIX_ARG" ] && [ -n "$NAME_ARG" ]; then
    echo "give -p or -n, not both" >&2; exit 2
fi

if [ -n "$PREFIX_ARG" ]; then
    TARGET=(-p "$PREFIX_ARG"); WHERE="the prefix $PREFIX_ARG (-p)"
elif [ -n "$NAME_ARG" ]; then
    TARGET=(-n "$NAME_ARG"); WHERE="the named environment $NAME_ARG of the conda on PATH (-n)"
elif [ -n "${AIPF_ENV_PREFIX:-}" ]; then
    TARGET=(-p "$AIPF_ENV_PREFIX"); WHERE="the prefix $AIPF_ENV_PREFIX (AIPF_ENV_PREFIX)"
else
    TARGET=(-n aipf); WHERE="the named environment aipf of the conda on PATH (AIPF_ENV_PREFIX unset)"
fi

KERNEL_STEP="skipped (not Linux x86_64)"
if [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ]; then
    KERNEL_STEP="pip install ${KERNELS[*]}, then $CUBLAS"
fi

echo "environment: $WHERE"
echo "spec:        $SPEC"
echo "step 2:      pip install $TORCH; pip install -e \"$REPO[md,figures,dev]\""
echo "step 3:      pip install --no-deps $MACE"
echo "step 4:      $KERNEL_STEP"
echo "step 5:      etc/conda/activate.d/zz-aipf.sh (PYTHONNOUSERSITE, cuBLAS/nvrtc first on LD_LIBRARY_PATH)"
if [ "$WITH_LAMMPS" = 1 ]; then
    echo "step 6:      bash $HERE/build-lammps.sh --prefix <the environment>"
else
    echo "step 6:      skipped (no --with-lammps)"
fi
if [ -n "${AIPF_CACHE_DIR:-}" ]; then
    export CONDA_PKGS_DIRS="$AIPF_CACHE_DIR/conda-pkgs"
    export PIP_CACHE_DIR="$AIPF_CACHE_DIR/pip-cache"
    echo "caches:      $CONDA_PKGS_DIRS and $PIP_CACHE_DIR (AIPF_CACHE_DIR)"
else
    echo "caches:      conda's and pip's own defaults (AIPF_CACHE_DIR unset)"
fi
if [ "$DRY_RUN" = 1 ]; then
    echo "dry run: nothing built"
    exit 0
fi

if [ -n "${AIPF_CACHE_DIR:-}" ]; then
    mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
fi

# Keep the user site-packages out of this interpreter's sys.path, so a stray
# `pip install --user` cannot shadow a version this environment pins.
export PYTHONNOUSERSITE=1

# --- 1. the environment ---
prefix_of() {  # the environment's prefix: the path itself, or asked of conda for a name
    if [ "${TARGET[0]}" = -p ]; then (cd "${TARGET[1]}" && pwd)
    else conda run "${TARGET[@]}" python -c 'import sys; print(sys.prefix)'; fi
}
if [ "${TARGET[0]}" = -p ]; then
    exists=$([ -x "${TARGET[1]}/bin/python" ] && echo yes || echo no)
else
    exists=$(conda env list | awk '{print $1}' | grep -qx "${TARGET[1]}" && echo yes || echo no)
fi
if [ "$exists" = yes ]; then
    echo "== 1. $WHERE exists; updating it from the spec (to start over: conda env remove ${TARGET[*]})"
    # A matplotlib that pip installed would be overwritten file by file by conda's and leave two
    # records of itself; it goes first, and conda's takes its place.
    OLD="$(prefix_of)/bin/python"
    if "$OLD" -c "import importlib.metadata as m, sys; sys.exit((m.distribution('matplotlib').read_text('INSTALLER') or '').strip() != 'pip')" 2>/dev/null; then
        "$OLD" -m pip uninstall -y matplotlib
    fi
    conda env update "${TARGET[@]}" -f "$SPEC"
else
    echo "== 1. conda env create ${TARGET[*]}"
    conda env create "${TARGET[@]}" -f "$SPEC"
fi
PREFIX="$(prefix_of)"
PY="$PREFIX/bin/python"

# --- 2. torch, then this package with its extras ---
echo "== 2. $TORCH, then the package"
"$PY" -m pip install "$TORCH"
"$PY" -m pip install -e "$REPO[md,figures,dev]"

# --- 3. MACE, without its dependencies (see MACE above) ---
echo "== 3. $MACE --no-deps"
"$PY" -m pip install --no-deps "$MACE"

# --- 4. the CUDA kernels and their cuBLAS, after everything that resolves torch ---
if [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ]; then
    echo "== 4. ${KERNELS[*]} and $CUBLAS"
    "$PY" -m pip install "${KERNELS[@]}"
    "$PY" -m pip install --no-deps "$CUBLAS"
else
    echo "== 4. skipped: the cuequivariance CUDA kernels ship for Linux x86_64 only"
fi

# --- 5. the activation hook ---
# The kernels load only when the environment's own cuBLAS and nvrtc come before any other
# CUDA library directory on the loader path; without that, MACE catches the ImportError and
# silently falls back to the slower reference implementation. Paths are taken from
# CONDA_PREFIX at activate time, so the hook is relocatable. ${VAR:+:$VAR} keeps a trailing
# colon (read by glibc as the working directory) off the path. A stale md hook is removed.
echo "== 5. activation hook"
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
rm -f "$PREFIX/etc/conda/activate.d/zz-aipf-md.sh"
cat > "$PREFIX/etc/conda/activate.d/zz-aipf.sh" <<'ACTIVATE'
# Written by env/build-env.sh. Keep ~/.local off sys.path, so a stray `pip install --user`
# cannot shadow a pinned version; put this environment's cuBLAS and nvrtc first on the
# loader path, so the cuequivariance kernels load (MACE falls back silently otherwise).
export PYTHONNOUSERSITE=1
export _AIPF_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
for _aipf_site in "$CONDA_PREFIX"/lib/python3*/site-packages/nvidia; do
    for _aipf_lib in cuda_nvrtc cublas; do
        if [ -d "$_aipf_site/$_aipf_lib/lib" ]; then
            export LD_LIBRARY_PATH="$_aipf_site/$_aipf_lib/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
    done
done
unset _aipf_site _aipf_lib
ACTIVATE
cat > "$PREFIX/etc/conda/deactivate.d/zz-aipf.sh" <<'DEACTIVATE'
# Written by env/build-env.sh: undo activate.d/zz-aipf.sh.
if [ -n "${_AIPF_OLD_LD_LIBRARY_PATH+set}" ]; then
    if [ -n "$_AIPF_OLD_LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="$_AIPF_OLD_LD_LIBRARY_PATH"
    else
        unset LD_LIBRARY_PATH
    fi
    unset _AIPF_OLD_LD_LIBRARY_PATH
fi
unset PYTHONNOUSERSITE
DEACTIVATE

echo "built $PREFIX"
"$PY" -c "
import aipf, torch, e3nn, importlib.metadata as m
from mace.calculators import MACECalculator
print('aipf', aipf.__version__, '| torch', torch.__version__, '| e3nn', e3nn.__version__,
      '| mace-torch', m.version('mace-torch'), '| cuda available', torch.cuda.is_available())
"

# --- 6. LAMMPS ---
if [ "$WITH_LAMMPS" = 1 ]; then
    echo "== 6. LAMMPS"
    bash "$HERE/build-lammps.sh" --prefix "$PREFIX"
fi
