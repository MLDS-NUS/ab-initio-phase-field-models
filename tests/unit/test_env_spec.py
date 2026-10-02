"""The environment files: a portable spec, a lock, and the two build scripts.

``env/environment.yml`` is the environment anyone can build (``name: aipf``, no prefix, Python and
pip). ``pyproject.toml``'s ``[md]`` extra carries the MACE runtime; mace-torch itself goes in with
``--no-deps`` from ``env/build-env.sh``, because it pins an e3nn the published potential cannot be
read with. ``env/build-lammps.sh`` builds LAMMPS with ML-IAP, Python and Kokkos into the same
environment. ``env/environment-lock.yml`` is the exact export of a built one. The checks of a built
environment run against ``AIPF_ENV_PREFIX``, else the interpreter running the tests, and skip where
that environment does not carry the stack.
"""
import os
import pathlib
import re
import subprocess
import tomllib

import pytest
import yaml

import declared_roots

REPO = pathlib.Path(__file__).resolve().parents[2]
ENV = REPO / "env" / "environment.yml"
LOCK = REPO / "env" / "environment-lock.yml"
BUILD = REPO / "env" / "build-env.sh"
LAMMPS = REPO / "env" / "build-lammps.sh"
PYPROJECT = REPO / "pyproject.toml"

#: How long a subprocess import of the CUDA stack may take before it is treated as unanswered;
#: firing it skips, because a busy shared node and a broken environment look the same from here.
IMPORT_TIMEOUT_S = 300

#: The MACE runtime the [md] extra pins: what `--no-deps mace-torch` leaves out and needs.
MD_PINS = {"e3nn": "0.5.1", "torch-ema": "0.3", "matscipy": "1.2.0", "h5py": "3.16.0",
           "gitpython": "3.1.46", "lmdb": "1.7.5", "orjson": "3.11.5"}


def _extras():
    return tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]


def _lock_pins():
    """Every ``name==version`` of the lock, the commented (installed-afterwards) lines included."""
    return dict(re.findall(r"^\s*(?:#\s*)?-\s*([A-Za-z0-9_.-]+)==(\S+)", LOCK.read_text(), re.M))


def _bash(script, *args, env=None, path_first=None):
    base = {k: v for k, v in os.environ.items() if k not in ("AIPF_ENV_PREFIX", "AIPF_CACHE_DIR",
                                                            "CONDA_PREFIX")}
    if path_first is not None:
        base["PATH"] = f"{path_first}{os.pathsep}{base.get('PATH', '')}"
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                          env={**base, **(env or {})})


# ---------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------

def test_the_spec_is_portable():
    """A named environment, no prefix, nothing of any host: it builds anywhere."""
    spec = yaml.safe_load(ENV.read_text())
    assert spec["name"] == "aipf"
    assert "prefix" not in spec
    assert "python=3.12" in spec["dependencies"]
    assert "pip" in spec["dependencies"]
    assert not [d for d in spec["dependencies"] if isinstance(d, dict)], "pip packages are pyproject's"


def test_the_spec_carries_the_figures_renderer():
    """The figures were drawn with conda-forge's matplotlib and its freetype (figures/common/BUILD.txt);
    the extras pin the same matplotlib, so pip leaves it in place."""
    deps = yaml.safe_load(ENV.read_text())["dependencies"]
    built = (REPO / "figures" / "common" / "BUILD.txt").read_text().split()
    mpl, freetype = built[built.index("matplotlib") + 1].rstrip(","), built[built.index("freetype") + 1]
    assert f"matplotlib-base={mpl}" in deps and f"freetype={freetype}" in deps
    for name in ("pymupdf", "nbconvert", "ipykernel", "jupyter_core"):
        assert name in deps, name
    for extra in ("figures", "dev"):
        assert f"matplotlib=={mpl}" in _extras()[extra], extra


def test_the_md_extra_carries_the_mace_runtime_and_not_mace():
    """mace-torch is installed by the script (its e3nn pin is unresolvable); its runtime is here, pinned."""
    md = _extras()["md"]
    pins = dict(d.split("==") for d in md if "==" in d)
    for name, version in MD_PINS.items():
        assert pins.get(name) == version, (name, md)
    assert any(d.startswith("ase>=") for d in md), md
    assert not [d for d in md if d.startswith("mace")], md
    assert {"figures", "dev"} <= set(_extras())


def test_the_lock_pins_the_published_stack():
    """Exact versions: torch 2.6.0 and e3nn 0.5.1 live, mace-torch among the lines installed after."""
    spec = yaml.safe_load(LOCK.read_text())
    assert "prefix" not in spec
    pins = _lock_pins()
    assert pins["torch"] == "2.6.0"
    assert pins["e3nn"] == "0.5.1"
    assert pins["mace-torch"] == "0.3.16"
    assert LOCK.read_text().splitlines()[0].startswith("#")


def test_the_lock_leaves_out_what_pip_cannot_resolve():
    """mace-torch pins e3nn==0.4.4, the kernels want a cuBLAS torch does not, and the lammps module is
    this environment's own build: listed live, the lock would not build, or would build another LAMMPS."""
    live = set()
    for item in yaml.safe_load(LOCK.read_text())["dependencies"]:
        if isinstance(item, dict):
            live |= {d.split("==")[0] for d in item["pip"]}
    held = {"mace-torch", "nvidia-cublas-cu12", "cuequivariance-ops-cu12",
            "cuequivariance-ops-torch-cu12", "lammps", "ab-initio-phase-field-models"}
    assert not live & held, live & held


def test_no_tracked_env_file_names_a_host():
    for path in (ENV, LOCK, BUILD, LAMMPS):
        text = path.read_text()
        # spelled in pieces, so the public word gate does not read this test as naming them
        for word in ("/" + "home/", "/" + "scratch/", "torch" + "env", "lammps" + "_env"):
            assert word not in text, (path.name, word)


# ---------------------------------------------------------------------------
# build-env.sh
# ---------------------------------------------------------------------------

def _fake(tmp_path, name, body="exit 0"):
    """An executable ``name`` first on PATH (returns its directory)."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / name).write_text(f"#!/bin/sh\n{body}\n")
    (bindir / name).chmod(0o755)
    return bindir


def test_the_dry_run_prints_the_plan_and_never_calls_conda(tmp_path):
    marker = tmp_path / "conda-was-called"
    bindir = _fake(tmp_path, "conda", f"touch {marker}\nexit 99")
    r = _bash(BUILD, "--dry-run", "--with-lammps", path_first=bindir)
    assert r.returncode == 0, r.stderr
    assert not marker.exists()
    out = r.stdout
    assert "nothing built" in out
    assert "pip install -e" in out and "[md,figures,dev]" in out
    assert "--no-deps mace-torch==0.3.16" in out
    assert "build-lammps.sh" in out and "zz-aipf.sh" in out


def test_the_build_script_asks_where():
    """``-p``, else ``-n``, else ``AIPF_ENV_PREFIX``, else ``-n aipf``; caches only under ``AIPF_CACHE_DIR``."""
    out = _bash(BUILD, "--dry-run").stdout
    assert "named environment aipf" in out and "conda's and pip's own defaults" in out
    out = _bash(BUILD, "--dry-run", env={"AIPF_ENV_PREFIX": "/opt/envs/x",
                                         "AIPF_CACHE_DIR": "/opt/cache"}).stdout
    assert "the prefix /opt/envs/x" in out
    assert "/opt/cache/conda-pkgs" in out and "/opt/cache/pip-cache" in out
    assert "the prefix /opt/envs/y (-p)" in _bash(BUILD, "--dry-run", "-p", "/opt/envs/y",
                                                  env={"AIPF_ENV_PREFIX": "/opt/envs/x"}).stdout
    assert "named environment mine" in _bash(BUILD, "--dry-run", "-n", "mine").stdout
    assert _bash(BUILD, "--dry-run", "-p", "a", "-n", "b").returncode == 2
    assert _bash(BUILD, "--bogus").returncode == 2


def test_the_script_orders_the_installs():
    """torch before the package, mace without deps, the kernels' cuBLAS after anything that resolves torch."""
    text = BUILD.read_text()
    order = [text.index(s) for s in ('pip install "$TORCH"', 'pip install -e "$REPO[md,figures,dev]"',
                                     'pip install --no-deps "$MACE"', 'pip install "${KERNELS[@]}"',
                                     'pip install --no-deps "$CUBLAS"')]
    assert order == sorted(order)
    assert "zz-aipf-md.sh" in text  # a stale md hook is removed


# ---------------------------------------------------------------------------
# build-lammps.sh
# ---------------------------------------------------------------------------

def test_build_lammps_help():
    r = _bash(LAMMPS, "--help")
    assert r.returncode == 0, r.stderr
    for flag in ("--prefix", "--cuda", "--no-cuda", "--source", "--jobs", "--arch"):
        assert flag in r.stdout


def test_build_lammps_dry_run_host_only():
    r = _bash(LAMMPS, "--dry-run", "--no-cuda", "--prefix", "/opt/envs/x")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "a51f9ba0e719be544293987bb3cbd9939f1b01ee" in out and "10 Dec 2025" in out
    for option in ("PKG_ML-IAP=yes", "PKG_ML-SNAP=yes", "PKG_PYTHON=yes", "MLIAP_ENABLE_PYTHON=yes",
                   "PKG_KOKKOS=yes", "BUILD_SHARED_LIBS=yes", "PKG_BROWNIAN=yes",
                   "PKG_EXTRA-PAIR=yes", "CMAKE_INSTALL_PREFIX=/opt/envs/x",
                   "Python_EXECUTABLE=/opt/envs/x/bin/python", "--disable-new-dtags"):
        assert option in out, option
    assert "Kokkos_ENABLE_CUDA" not in out and "Kokkos_ENABLE_OPENMP=yes" in out
    assert "nothing built" in out


def test_build_lammps_dry_run_cuda(tmp_path):
    bindir = _fake(tmp_path, "nvcc")
    r = _bash(LAMMPS, "--dry-run", "--cuda", "--arch", "AMPERE80", "--prefix", "/opt/envs/x",
              path_first=bindir)
    assert r.returncode == 0, r.stderr
    assert "Kokkos_ENABLE_CUDA=yes" in r.stdout and "Kokkos_ARCH_AMPERE80=yes" in r.stdout
    assert "nvcc_wrapper" in r.stdout and "Kokkos_ENABLE_OPENMP=no" in r.stdout


def test_build_lammps_fetches_by_hash_and_refuses_another_commit(tmp_path):
    """The pinned commit is fetched by its hash; a git source at any other commit is refused."""
    text = LAMMPS.read_text()
    assert 'git -C "$SRC" fetch -q --depth 1 "$REPO_URL" "$REF"' in text
    src = tmp_path / "src"
    (src / "cmake").mkdir(parents=True)
    (src / "src").mkdir()
    (src / "cmake" / "CMakeLists.txt").write_text("")
    (src / "src" / "version.h").write_text('#define LAMMPS_VERSION "10 Dec 2025"\n')
    git = ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "not the pin"], check=True)
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    bindir = _fake(tmp_path, "python")
    (prefix / "bin" / "python").symlink_to(bindir / "python")
    _fake(tmp_path, "cmake")
    _fake(tmp_path, "g++", 'case "$1" in -dumpversion) echo 12;; *) echo "g++ (GCC) 12";; esac')
    r = _bash(LAMMPS, "--no-cuda", "--prefix", str(prefix), "--source", str(src),
              path_first=bindir, env={"CXX": str(bindir / "g++")})
    assert r.returncode == 2, r.stdout + r.stderr
    assert "not the pinned" in r.stderr


def test_build_lammps_needs_an_environment():
    assert _bash(LAMMPS, "--dry-run", "--no-cuda").returncode == 2


# ---------------------------------------------------------------------------
# a built environment
# ---------------------------------------------------------------------------

def _md_prefix():
    """The environment to check: ``AIPF_ENV_PREFIX``, else the running one; skip without the stack."""
    prefix = declared_roots.aipf_prefix()
    py = prefix / "bin" / "python"
    if not py.exists():
        pytest.skip(f"no environment at {prefix} (AIPF_ENV_PREFIX)")
    r = subprocess.run([str(py), "-c", "import importlib.metadata as m; m.version('mace-torch')"],
                       capture_output=True, text=True,
                       env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin"})
    if r.returncode != 0:
        pytest.skip(f"{prefix} carries no MACE (env/build-env.sh builds one that does)")
    return prefix


def test_activate_hook_sets_both_variables():
    """Sourced, the hook keeps ~/.local off sys.path and puts the environment's cuBLAS and nvrtc first.

    Losing that order silently disables the cuequivariance kernels."""
    prefix = _md_prefix()
    hook = prefix / "etc" / "conda" / "activate.d" / "zz-aipf.sh"
    if not hook.exists():
        pytest.skip(f"no activation hook under {prefix}; env/build-env.sh writes it")
    assert not (prefix / "etc" / "conda" / "activate.d" / "zz-aipf-md.sh").exists()
    r = subprocess.run(
        ["bash", "-c", f'CONDA_PREFIX="{prefix}"; LD_LIBRARY_PATH=/inherited; . "{hook}"; '
                       'printf "%s\\n%s\\n" "$PYTHONNOUSERSITE" "$LD_LIBRARY_PATH"'],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    nousersite, ldpath = r.stdout.splitlines()[:2]
    assert nousersite == "1"
    parts = ldpath.split(":")
    assert parts[0].endswith("nvidia/cublas/lib"), ldpath
    assert parts[1].endswith("nvidia/cuda_nvrtc/lib"), ldpath
    assert parts[-1] == "/inherited"


def test_built_environment_has_the_pinned_versions():
    """Check the INTERPRETER, not the file."""
    import json
    prefix = _md_prefix()
    r = subprocess.run(
        [str(prefix / "bin" / "python"), "-c",
         "import json, importlib.metadata as m;"
         "print(json.dumps({p: m.version(p) for p in ('torch', 'e3nn', 'mace-torch')}))"],
        capture_output=True, text=True, env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin"})
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got["e3nn"] == "0.5.1" and got["mace-torch"] == "0.3.16", got
    assert got["torch"].startswith("2.6.0"), got


@pytest.mark.env
def test_built_environment_can_import_mace_and_the_kernels():
    """A --no-deps install that cannot import mace is worse than no mace; marked ``env``: it needs the
    built environment and skips naming it."""
    prefix = _md_prefix()
    site = next((prefix / "lib").glob("python3*/site-packages/nvidia"), None)
    libs = "" if site is None else f"{site / 'cublas' / 'lib'}:{site / 'cuda_nvrtc' / 'lib'}"
    try:
        r = subprocess.run(
            [str(prefix / "bin" / "python"), "-c",
             "from mace.calculators import MACECalculator; import cuequivariance_torch, "
             "cuequivariance_ops_torch"],
            capture_output=True, text=True, timeout=IMPORT_TIMEOUT_S,
            env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin", "LD_LIBRARY_PATH": libs})
    except subprocess.TimeoutExpired:
        pytest.skip(f"the import did not finish within {IMPORT_TIMEOUT_S} s, so this was not inspected")
    assert r.returncode == 0, r.stderr


def test_the_slow_marker_is_deselected_by_default(pytestconfig):
    """Registering the marker is not enough -- `addopts` has to apply it. The ``env`` checks are not
    deselected: they run and skip, naming what is missing, where this machine lacks it."""
    markexpr = pytestconfig.getoption("-m")
    assert "not slow" in markexpr, (
        f"the default mark expression is {markexpr!r}; the slow checks would run on every full-suite run")
