#!/usr/bin/env bash
# env/build-lammps.sh -- build LAMMPS into the aipf environment: one binary for every route.
#
#   bash env/build-lammps.sh [--prefix ENVPREFIX] [--cuda|--no-cuda] [--arch ARCH]
#                            [--source DIR] [--jobs N] [--dry-run]
#
#   --prefix ENVPREFIX  the environment to build against and install into (its bin/python
#                       is the interpreter LAMMPS is linked against). Default: $CONDA_PREFIX,
#                       else $AIPF_ENV_PREFIX.
#   --cuda | --no-cuda  Kokkos with the CUDA backend, or host-only (OpenMP). Default: --cuda
#                       when nvcc is on PATH. A CUDA build runs the pair-potential decks
#                       anywhere; only its Kokkos (-k on, the MACE route) needs a GPU. The MACE
#                       route runs on a GPU only, so --no-cuda is for a site that cannot
#                       compile CUDA, and it serves the pair-potential decks.
#   --arch ARCH         the Kokkos GPU architecture (AMPERE80, HOPPER90, ...). Default: read
#                       from nvidia-smi's compute capability; required where no GPU is visible.
#   --source DIR        an existing LAMMPS source tree at the pinned version; it is only read
#                       (a git checkout must be at the pinned commit). Default: fetch the
#                       pinned commit by its hash into the work directory (git).
#   --jobs N            parallel compile jobs. Default: the number of processors.
#   --dry-run           print the configuration and stop; nothing is downloaded or built.
#
# What is built: LAMMPS at the pinned commit with ML-IAP and its Python coupling
# (the MACE route), PYTHON, KOKKOS, and the packages the built-in decks use (BROWNIAN for
# fix brownian, EXTRA-PAIR for lj/smooth/linear), as shared libraries, serial (no MPI).
# It installs $PREFIX/bin/lmp, $PREFIX/lib/liblammps.so and the `lammps` python module into
# the environment's site-packages, then prints the [site] lines for aipf.toml. Pip's cache is
# $AIPF_CACHE_DIR/pip-cache when AIPF_CACHE_DIR is set.
# Libraries are found through an RPATH (the environment's lib first), not LD_LIBRARY_PATH, so
# an OpenMP or C++ runtime inherited from a site module cannot be picked up instead.
# Work directory (source and build tree): $PREFIX/share/aipf-lammps.
set -euo pipefail

# Pinned: the commit the published molecular dynamics ran with, not a stable release. It is the
# develop branch of 2025-12-19, whose version string is "10 Dec 2025" (its python module reports
# 20251210); the feature release of that name, tag patch_10Dec2025, is nine days older and is not
# the same code. It is fetched by its hash, which pins the content.
REF="a51f9ba0e719be544293987bb3cbd9939f1b01ee"
VERSION="10 Dec 2025"
REPO_URL="https://github.com/lammps/lammps"
# The ML-IAP Python coupling is cythonised at build time; its Kokkos (CUDA) form hands device
# arrays to Python through cupy.
CYTHON="cython==3.2.4"
CUPY="cupy-cuda12x==13.6.0"

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    echo "Pinned: LAMMPS $VERSION, commit $REF ($REPO_URL)."
}

PREFIX="${CONDA_PREFIX:-${AIPF_ENV_PREFIX:-}}"
CUDA=auto
ARCH=""
SOURCE=""
JOBS="$(nproc 2>/dev/null || echo 4)"
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="${2:?--prefix needs a path}"; shift 2 ;;
        --cuda) CUDA=yes; shift ;;
        --no-cuda) CUDA=no; shift ;;
        --arch) ARCH="${2:?--arch needs a Kokkos architecture name}"; shift 2 ;;
        --source) SOURCE="${2:?--source needs a directory}"; shift 2 ;;
        --jobs) JOBS="${2:?--jobs needs a number}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done

if [ -z "$PREFIX" ]; then
    echo "no environment: give --prefix, or activate the aipf environment" >&2; exit 2
fi
PY="$PREFIX/bin/python"
WORK="$PREFIX/share/aipf-lammps"
if [ -n "${AIPF_CACHE_DIR:-}" ] && [ -z "${PIP_CACHE_DIR:-}" ]; then
    export PIP_CACHE_DIR="$AIPF_CACHE_DIR/pip-cache"
fi

# Compiler settings exported by ANOTHER activated conda environment (its compilers package sets
# CC, CXX and *FLAGS, with that environment's include and library directories in them) would
# build against, and link an RPATH to, that environment. They are dropped; set them after
# activating this one, or not at all, to choose a compiler.
if [ -n "${CONDA_PREFIX:-}" ] && [ "$CONDA_PREFIX" != "$PREFIX" ]; then
    for var in CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS; do
        if [[ "${!var:-}" == *"$CONDA_PREFIX"* ]]; then
            echo "note: ignoring $var, which names another environment ($CONDA_PREFIX)" >&2
            unset "$var"
        fi
    done
fi
SRC="${SOURCE:-$WORK/lammps-$REF}"
BUILD="$WORK/build"

if [ "$CUDA" = auto ]; then
    CUDA=$(command -v nvcc >/dev/null 2>&1 && echo yes || echo no)
fi

# Kokkos names a GPU architecture by family and compute capability.
arch_of() {
    case "$1" in
        7.0) echo VOLTA70 ;; 7.2) echo VOLTA72 ;; 7.5) echo TURING75 ;;
        8.0) echo AMPERE80 ;; 8.6) echo AMPERE86 ;; 8.7) echo AMPERE87 ;; 8.9) echo ADA89 ;;
        9.0) echo HOPPER90 ;; 10.0) echo BLACKWELL100 ;; 12.0) echo BLACKWELL120 ;;
        *) echo "" ;;
    esac
}

CMAKE_ARGS=(
    -D CMAKE_BUILD_TYPE=Release
    -D CMAKE_INSTALL_PREFIX="$PREFIX"
    -D CMAKE_INSTALL_LIBDIR=lib
    -D BUILD_SHARED_LIBS=yes
    -D BUILD_MPI=no
    -D BUILD_OMP=yes
    -D BUILD_TOOLS=no
    -D PKG_ML-IAP=yes
    -D PKG_ML-SNAP=yes
    -D PKG_PYTHON=yes
    -D MLIAP_ENABLE_PYTHON=yes
    -D PKG_KOKKOS=yes
    -D PKG_BROWNIAN=yes
    -D PKG_EXTRA-PAIR=yes
    -D Kokkos_ENABLE_SERIAL=yes
    -D PYTHON_EXECUTABLE="$PY"
    -D Python_EXECUTABLE="$PY"
    -D Cythonize_EXECUTABLE="$PREFIX/bin/cythonize"
    -D CMAKE_BUILD_WITH_INSTALL_RPATH=yes
    -D CMAKE_SHARED_LINKER_FLAGS=-Wl,--disable-new-dtags
    -D CMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags
)
RPATH="$PREFIX/lib"
if [ "$CUDA" = yes ]; then
    NVCC="$(command -v nvcc || true)"
    if [ -z "$NVCC" ]; then
        echo "--cuda: no nvcc on PATH; load or install a CUDA toolkit first" >&2; exit 2
    fi
    if [ -z "$ARCH" ]; then
        CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
        ARCH="$(arch_of "$CAP")"
        if [ -z "$ARCH" ] && [ "$DRY_RUN" = 1 ]; then
            ARCH="<needs --arch>"
        elif [ -z "$ARCH" ]; then
            echo "--cuda: cannot tell the GPU architecture here (compute capability '${CAP:-none}');" \
                 "give --arch, e.g. --arch AMPERE80" >&2
            exit 2
        fi
    fi
    # The toolkit's runtime libraries, found by RPATH after the environment's own.
    RPATH="$RPATH;$(cd "$(dirname "$NVCC")/.." && pwd)/lib64"
    CMAKE_ARGS+=(
        -D Kokkos_ENABLE_CUDA=yes
        -D Kokkos_ENABLE_OPENMP=no
        -D "Kokkos_ARCH_$ARCH=yes"
        -D CMAKE_CXX_COMPILER="$SRC/lib/kokkos/bin/nvcc_wrapper"
    )
fi
if [ "$CUDA" != yes ]; then
    CMAKE_ARGS+=(-D Kokkos_ENABLE_OPENMP=yes)
fi
CMAKE_ARGS+=(-D CMAKE_INSTALL_RPATH="$RPATH")

echo "LAMMPS:      $VERSION, commit $REF ($REPO_URL)"
echo "environment: $PREFIX (python: $PY)"
echo "source:      $SRC${SOURCE:+ (given, read only)}"
echo "build:       $BUILD (-j $JOBS)"
if [ "$CUDA" = yes ]; then
    echo "kokkos:      CUDA ($ARCH), serial host, nvcc: $NVCC"
else
    echo "kokkos:      OpenMP (host only)"
fi
echo "rpath:       $RPATH"
if [ "$DRY_RUN" = 1 ]; then
    printf 'cmake -S %s/cmake -B %s' "$SRC" "$BUILD"; printf ' %q' "${CMAKE_ARGS[@]}"; echo
    echo "dry run: nothing built"
    exit 0
fi

if [ ! -x "$PY" ]; then
    echo "no interpreter at $PY; build the environment first (env/build-env.sh)" >&2; exit 2
fi
command -v cmake >/dev/null || { echo "cmake is required (3.16 or newer)" >&2; exit 2; }
# The bundled Kokkos (5.0) is C++20, which the host compiler must support (GCC 10 or newer).
if [ "$CUDA" = yes ]; then CXXVAR=NVCC_WRAPPER_DEFAULT_COMPILER; else CXXVAR=CXX; fi
HOSTCXX="${!CXXVAR:-g++}"
HOSTVER="$("$HOSTCXX" -dumpversion 2>/dev/null || echo 0)"
if "$HOSTCXX" --version 2>/dev/null | grep -qi 'gcc\|g++' && [ "${HOSTVER%%.*}" -lt 10 ]; then
    echo "$HOSTCXX is GCC $HOSTVER; Kokkos 5 needs C++20, GCC 10 or newer: put one first on PATH" \
         "(or name it with $CXXVAR)" >&2
    exit 2
fi
echo "host compiler: $HOSTCXX ($HOSTVER)"

# --- the source ---
if [ -z "$SOURCE" ] && [ ! -f "$SRC/cmake/CMakeLists.txt" ]; then
    command -v git >/dev/null || { echo "git is required to fetch the pinned commit" >&2; exit 2; }
    echo "== fetching $REF"
    mkdir -p "$SRC"
    git -C "$SRC" init -q
    git -C "$SRC" fetch -q --depth 1 "$REPO_URL" "$REF"
    git -C "$SRC" checkout -q FETCH_HEAD
fi
if [ -d "$SRC/.git" ]; then
    HEAD_REF="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)"
    if [ "$HEAD_REF" != "$REF" ]; then
        echo "$SRC is at commit ${HEAD_REF:-<none>}, not the pinned $REF; refusing" >&2
        exit 2
    fi
fi
if [ ! -f "$SRC/cmake/CMakeLists.txt" ]; then
    echo "no LAMMPS source at $SRC (expected cmake/CMakeLists.txt)" >&2; exit 2
fi
if ! grep -q "LAMMPS_VERSION \"$VERSION\"" "$SRC/src/version.h" 2>/dev/null; then
    echo "$SRC is not LAMMPS $VERSION ($(grep -m1 LAMMPS_VERSION "$SRC/src/version.h" 2>/dev/null))" >&2
    exit 2
fi

# --- build-time and runtime Python packages of the ML-IAP coupling ---
"$PY" -m pip install "$CYTHON"
if [ "$CUDA" = yes ]; then
    "$PY" -m pip install "$CUPY"
fi

# --- configure, build, install ---
echo "== configure"
cmake -S "$SRC/cmake" -B "$BUILD" "${CMAKE_ARGS[@]}"
echo "== build"
cmake --build "$BUILD" -j "$JOBS"
echo "== install"
cmake --install "$BUILD"
echo "== install the python module"
cmake --build "$BUILD" --target install-python

echo "== check"
"$PREFIX/bin/lmp" -h | sed -n '/^Large-scale/p;/KOKKOS package API/p;/^Installed packages/{n;n;p;}'
"$PY" -c "
import lammps
l = lammps.lammps(cmdargs=['-log', 'none', '-screen', 'none', '-nocite'])
print('python module: LAMMPS', l.version(), l.installed_packages)
print('mliap pair styles:', [s for s in l.available_styles('pair') if s.startswith('mliap')])
l.close()
"

echo
echo "Put these lines in aipf.toml (or set AIPF_LAMMPS and AIPF_MACE_POTENTIAL):"
echo
echo "[site]"
echo "lammps = \"$PREFIX/bin/lmp\""
echo "# mace_potential = \"/path/to/model-mliap.pt\"   # a MACE model converted for ML-IAP"
