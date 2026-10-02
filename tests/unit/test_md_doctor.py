"""What the environment inspector must and must not claim.

Two rules run through the whole file.

First: a check that cannot tell "broken" from "not inspected" is worse than no
check. Every check here has three or four documented outcomes and one of them
is always ``UNKNOWN``, which says the question was not answered and why.

Second: a timeout is never a failure. A busy shared node and a broken
environment look identical from the outside, so a check that runs out of time
reports ``UNKNOWN``.

The ``env``-marked tests at the bottom run against the real interpreters,
binaries and model files on this machine, including deliberately broken
loader environments, because a checker verified only against a healthy
machine has not been verified at all.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

import declared_roots
from aipf.md import doctor
from aipf.md.doctor import Diagnosis, Finding, Probe, Verdict

# The environments this machine actually has, each through a declared root:
# this package's own from its environment spec, the simulator's interpreter
# and its machine-learning build from the site facts ``AIPF_LAMMPS_PYTHON``
# and ``AIPF_LAMMPS``, the build a user reaches by typing its name from the
# search path, and the production model under the hydrogen-helium raw root.
AIPF_PY = declared_roots.aipf_python()
LAMMPS_PY = declared_roots.site_fact("lammps_python")
MLIAP_LMP = declared_roots.site_fact("lammps")
PRODUCTION_MODEL = declared_roots.raw(
    "hhe", "builds/mliap/hhe_train_ist_swa-mliap_lammps_cueq_fp32.pt")
#: A directory that poisons the dynamic loader for both libraries this stack
#: loads: it holds an OpenMP runtime whose symbols the LAMMPS library cannot
#: resolve, and it does not hold the accelerated-kernel libraries at all. On
#: this machine it is the site compiler's library directory. Only this test
#: reads it, from ``AIPF_POISONED_LD``; unset, the path exists nowhere.
POISONED_LD = pathlib.Path(os.environ.get("AIPF_POISONED_LD") or "<AIPF_POISONED_LD unset>")
#: The site's own CUDA library directory, which ahead of the environment's own
#: two keeps the fast kernels from loading: the site fact ``AIPF_CUDA_LIB``.
SITE_CUDA_LIB = declared_roots.site_fact("cuda_lib")
#: A synthetic CUDA directory for the facts-record tests, which never touch disk.
SYNTHETIC_CUDA_LIB = "/somewhere/cuda/12.2.2/lib64"


# ---------------------------------------------------------------------------
# Fixtures: a facts record, and probes made of one
# ---------------------------------------------------------------------------


def facts(**overrides):
    """A healthy facts record, with whatever the caller wants changed."""
    record = {
        "executable": "/somewhere/bin/python",
        "python_version": "3.12.0",
        "versions": {
            "e3nn": "0.5.1",
            "torch": "2.6.0",
            "mace-torch": "0.3.16",
            "lammps": "2025.12.10",
            "nvidia-cublas-cu12": "12.9.2.10",
            "cuequivariance-torch": "0.9.1",
            "cuequivariance-ops-torch-cu12": "0.9.1",
        },
        "modules": {"torch": True, "lammps": True,
                    "cuequivariance_ops_torch": True},
        "accelerator_dirs": {"cublas": "/somewhere/nvidia/cublas/lib",
                             "cuda_nvrtc": "/somewhere/nvidia/cuda_nvrtc/lib"},
        "ld_library_path": ["/somewhere/nvidia/cublas/lib",
                            "/somewhere/nvidia/cuda_nvrtc/lib"],
        "lammps": {
            "origin": "/somewhere/site-packages/lammps/__init__.py",
            "shared_library": "/somewhere/site-packages/lammps/liblammps.so",
            "dlopen": "ok",
            "entry_points": ["activate_mliappy_kokkos", "load_unified_kokkos"],
        },
    }
    record.update(overrides)
    return record


def healthy(**overrides):
    return Probe(facts=facts(**overrides))


def unanswered(reason="the interpreter could not be run"):
    return Probe(facts=None, reason=reason)


def canned(answer):
    """A prober that hands back one prepared probe, and records the call."""
    def prober(python=None, **kwargs):
        prober.calls.append(python)
        return answer
    prober.calls = []
    return prober


class _Result:
    """What a finished subprocess looks like to the code under test."""

    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------


def test_unknown_is_a_separate_answer_from_every_kind_of_problem():
    """The distinction the whole module exists for."""
    assert Verdict.UNKNOWN not in (Verdict.OK, Verdict.BROKEN, Verdict.DEGRADED)
    assert len({v.value for v in Verdict}) == 5
    assert Verdict.NOT_APPLICABLE not in (Verdict.UNKNOWN, Verdict.BROKEN)


def test_a_finding_that_is_not_inspected_must_say_why():
    with pytest.raises(ValueError, match="why"):
        Finding("devices", Verdict.UNKNOWN, "")


def test_a_problem_must_carry_what_to_do_about_it():
    """"Reports what is missing" is the job; a bare verdict does not do it."""
    for verdict in (Verdict.BROKEN, Verdict.DEGRADED):
        with pytest.raises(ValueError, match="remedy"):
            Finding("devices", verdict, "no device", remedy="")


def test_a_finding_always_says_something():
    with pytest.raises(ValueError, match="detail"):
        Finding("devices", Verdict.OK, "")


def test_a_verdict_must_be_a_verdict():
    with pytest.raises(TypeError, match="Verdict"):
        Finding("devices", "ok", "one device")


def test_a_diagnosis_refuses_two_answers_for_one_check():
    """Two findings under one name means one of them is never read."""
    one = Finding("devices", Verdict.OK, "one device")
    with pytest.raises(ValueError, match="devices"):
        Diagnosis((one, one))


def test_can_run_is_three_valued():
    """The property is the module's whole argument, expressed as a value.

    ``False`` means something was inspected and will not work. ``True`` means
    everything was inspected and will. ``None`` means nobody knows, which is
    neither of those and must not be rounded to either.
    """
    ok = Finding("devices", Verdict.OK, "one device")
    slow = Finding("accelerator_libraries", Verdict.DEGRADED, "no kernels",
                   remedy="put them on the loader path")
    bad = Finding("lammps_module", Verdict.BROKEN, "absent",
                  remedy="build it")
    dunno = Finding("potential", Verdict.UNKNOWN, "no model named")

    assert Diagnosis((ok,)).can_run is True
    assert Diagnosis((ok, slow)).can_run is True
    assert Diagnosis((ok, dunno)).can_run is None
    assert Diagnosis((ok, bad)).can_run is False
    # A problem outranks an unanswered question: one certain failure is enough.
    assert Diagnosis((bad, dunno)).can_run is False


def test_a_diagnosis_sorts_its_findings_into_the_four_answers():
    ok = Finding("devices", Verdict.OK, "one device")
    slow = Finding("accelerator_libraries", Verdict.DEGRADED, "no kernels",
                   remedy="put them on the loader path")
    bad = Finding("lammps_module", Verdict.BROKEN, "absent", remedy="build it")
    dunno = Finding("potential", Verdict.UNKNOWN, "no model named")
    d = Diagnosis((ok, slow, bad, dunno))
    assert d.passed == (ok,)
    assert d.degraded == (slow,)
    assert d.broken == (bad,)
    assert d.not_inspected == (dunno,)


def test_the_printed_report_carries_every_answer_and_counts_them():
    d = Diagnosis((
        Finding("devices", Verdict.OK, "one device"),
        Finding("potential", Verdict.UNKNOWN, "no model named"),
    ))
    text = str(d)
    assert "devices" in text and "one device" in text
    assert "potential" in text and "no model named" in text
    assert "1 ok" in text and "1 not inspected" in text


def test_the_printed_report_shows_the_remedy():
    text = str(Diagnosis((
        Finding("devices", Verdict.BROKEN, "no device is visible",
                remedy="submit to a queue that provides one"),)))
    assert "submit to a queue that provides one" in text


def test_records_are_plain_data():
    """A diagnosis goes into a run record, which is written as JSON."""
    d = Diagnosis((Finding("devices", Verdict.OK, "one device", raw="GPU 0"),))
    records = d.to_records()
    assert json.loads(json.dumps(records)) == records
    assert records[0]["check"] == "devices"
    assert records[0]["verdict"] == "ok"


# ---------------------------------------------------------------------------
# The allocator setting
# ---------------------------------------------------------------------------


def test_the_allocator_option_that_breaks_the_run_is_refused():
    f = doctor.alloc_conf({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert f.verdict is Verdict.BROKEN
    assert "expandable_segments" in f.detail
    assert f.remedy


def test_the_allocator_refusal_says_the_setting_is_right_for_training():
    """Without that sentence the reader unsets it everywhere and loses it."""
    f = doctor.alloc_conf({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert "training" in (f.detail + f.remedy)


def test_the_allocator_option_is_found_among_others():
    f = doctor.alloc_conf(
        {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:true"})
    assert f.verdict is Verdict.BROKEN


def test_other_allocator_options_are_left_alone():
    f = doctor.alloc_conf({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"})
    assert f.verdict is Verdict.OK
    assert "max_split_size_mb:128" in f.detail


def test_expandable_segments_switched_off_is_not_the_defect():
    f = doctor.alloc_conf({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"})
    assert f.verdict is Verdict.OK


def test_an_unset_allocator_is_the_good_case():
    f = doctor.alloc_conf({})
    assert f.verdict is Verdict.OK


def test_the_allocator_check_is_never_unanswerable():
    """It reads a mapping that is always there; it has no third outcome."""
    for value in ("", "expandable_segments:True", "garbage", "a:b,c"):
        f = doctor.alloc_conf({"PYTORCH_CUDA_ALLOC_CONF": value})
        assert f.verdict in (Verdict.OK, Verdict.BROKEN)
    assert doctor.alloc_conf({}).verdict is Verdict.OK


def test_the_allocator_check_reads_the_live_environment_by_default(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    assert doctor.alloc_conf().verdict is Verdict.BROKEN


# ---------------------------------------------------------------------------
# The library that has to read the model
# ---------------------------------------------------------------------------


def test_the_required_deserialiser_version_passes():
    f = doctor.deserialiser(healthy())
    assert f.verdict is Verdict.OK
    assert "0.5.1" in f.detail


def test_another_deserialiser_version_is_refused_by_number():
    f = doctor.deserialiser(healthy(versions={"e3nn": "0.5.9"}))
    assert f.verdict is Verdict.BROKEN
    assert "0.5.9" in f.detail and "0.5.1" in f.detail


def test_an_absent_deserialiser_is_refused():
    f = doctor.deserialiser(healthy(versions={}))
    assert f.verdict is Verdict.BROKEN
    assert f.remedy


def test_the_deserialiser_version_is_the_one_the_module_measured():
    """The pin lives in one place. Restating it here would let it drift."""
    from aipf.md.potentials import REQUIRED_E3NN
    f = doctor.deserialiser(healthy(versions={"e3nn": REQUIRED_E3NN}))
    assert f.verdict is Verdict.OK


def test_an_engine_that_does_not_use_the_library_is_not_inspected():
    """The pin belongs to one engine, not to the module."""
    f = doctor.deserialiser(healthy(versions={}), required=None)
    assert f.verdict is Verdict.UNKNOWN
    assert "engine" in f.detail


def test_an_unanswered_probe_leaves_the_version_unknown():
    f = doctor.deserialiser(unanswered("no such interpreter"))
    assert f.verdict is Verdict.UNKNOWN
    assert "no such interpreter" in f.detail


def test_the_version_check_names_the_interpreter_it_asked():
    f = doctor.deserialiser(healthy(executable="/elsewhere/bin/python",
                                    versions={"e3nn": "0.5.9"}))
    assert "/elsewhere/bin/python" in f.detail


# ---------------------------------------------------------------------------
# Whether the fast kernels can be reached, cheaply
# ---------------------------------------------------------------------------


def test_installed_kernels_on_an_ordered_loader_path_pass():
    f = doctor.accelerator_libraries(healthy())
    assert f.verdict is Verdict.OK


def test_kernels_that_are_not_installed_are_a_downgrade_not_a_failure():
    f = doctor.accelerator_libraries(healthy(
        versions={"nvidia-cublas-cu12": "12.9.2.10"},
        modules={"cuequivariance_ops_torch": False}))
    assert f.verdict is Verdict.DEGRADED
    assert f.remedy


def test_an_old_math_library_disables_the_kernels_silently():
    """Measured: the older library exports the needed symbol zero times."""
    f = doctor.accelerator_libraries(
        healthy(versions={"nvidia-cublas-cu12": "12.4.5.8",
                          "cuequivariance-ops-torch-cu12": "0.9.1"}))
    assert f.verdict is Verdict.DEGRADED
    assert "12.4.5.8" in f.detail


def test_a_newer_math_library_passes():
    f = doctor.accelerator_libraries(
        healthy(versions={"nvidia-cublas-cu12": "13.0.0.0",
                          "cuequivariance-ops-torch-cu12": "0.9.1"}))
    assert f.verdict is Verdict.OK


def test_the_loader_path_missing_the_environments_own_directories_is_reported():
    """Loader order, not node type, is what decides this."""
    f = doctor.accelerator_libraries(healthy(ld_library_path=["/somewhere/cuda/lib64"]))
    assert f.verdict is Verdict.DEGRADED
    assert "cublas" in f.detail and "cuda_nvrtc" in f.detail


def test_one_of_the_two_directories_is_not_enough():
    f = doctor.accelerator_libraries(
        healthy(ld_library_path=["/somewhere/nvidia/cublas/lib"]))
    assert f.verdict is Verdict.DEGRADED
    assert "cuda_nvrtc" in f.detail


def test_another_cuda_directory_ahead_of_the_environments_own_is_named():
    f = doctor.accelerator_libraries(healthy(ld_library_path=[
        SYNTHETIC_CUDA_LIB,
        "/somewhere/nvidia/cublas/lib",
        "/somewhere/nvidia/cuda_nvrtc/lib"]))
    assert f.verdict is Verdict.DEGRADED
    assert SYNTHETIC_CUDA_LIB in f.detail


def test_a_non_cuda_directory_ahead_of_them_is_not_a_problem():
    f = doctor.accelerator_libraries(healthy(ld_library_path=[
        "/opt/pbs/lib",
        "/somewhere/nvidia/cublas/lib",
        "/somewhere/nvidia/cuda_nvrtc/lib"]))
    assert f.verdict is Verdict.OK


def test_the_downgrade_says_what_it_costs():
    """The number is the argument for caring: the model runs, and pays."""
    f = doctor.accelerator_libraries(healthy(ld_library_path=[]))
    assert "2.49" in f.detail or "2.5" in f.detail


def test_an_interpreter_with_no_accelerator_directories_at_all():
    f = doctor.accelerator_libraries(
        healthy(accelerator_dirs={"cublas": None, "cuda_nvrtc": None}))
    assert f.verdict is Verdict.DEGRADED


def test_an_unanswered_probe_leaves_the_kernels_unknown():
    f = doctor.accelerator_libraries(unanswered())
    assert f.verdict is Verdict.UNKNOWN


def test_the_cheap_kernel_check_says_what_would_confirm_it():
    """It predicts from the loader path. Only the import settles it."""
    f = doctor.accelerator_libraries(healthy(ld_library_path=[]))
    assert "import" in (f.detail + f.remedy)


# ---------------------------------------------------------------------------
# Whether the fast kernels actually load: the expensive confirmation
# ---------------------------------------------------------------------------


def test_the_deep_kernel_check_is_not_run_unless_asked():
    d = doctor.examine(python=sys.executable, prober=canned(unanswered()))
    f = d.find("accelerator_kernels")
    assert f.verdict is Verdict.UNKNOWN
    assert "deep" in f.detail


def test_an_import_that_fails_is_a_downgrade_with_the_error_kept():
    f = doctor.kernels(python=sys.executable,
                       runner=lambda *a, **k: _Result(1, "", "ImportError: libcue_ops.so"))
    assert f.verdict is Verdict.DEGRADED
    assert "libcue_ops.so" in (f.raw or "")


def test_an_import_that_succeeds_passes():
    f = doctor.kernels(python=sys.executable,
                       runner=lambda *a, **k: _Result(0, "", ""))
    assert f.verdict is Verdict.OK


def test_a_kernel_import_that_runs_out_of_time_is_not_a_failure():
    """A busy node and a broken environment look the same from out here."""
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="python", timeout=300.0)

    f = doctor.kernels(python=sys.executable, runner=timeout)
    assert f.verdict is Verdict.UNKNOWN
    assert "300" in f.detail


def test_an_interpreter_that_cannot_be_run_is_not_a_failure_either():
    def missing(*a, **k):
        raise OSError("No such file or directory")

    f = doctor.kernels(python="/no/such/python", runner=missing)
    assert f.verdict is Verdict.UNKNOWN


def test_the_kernel_timeout_is_far_above_the_measured_import():
    """The repository already carries a timeout set AT its own measurement.

    Measured for this module: the failing import answers in 4.1 s, the warm
    success in 10.2 s, and a cold one was measured elsewhere at 39-45 s on top
    of a 42-49 s library import. A limit at the measurement is a coin flip.
    """
    assert doctor.KERNEL_TIMEOUT_S >= 300.0


# ---------------------------------------------------------------------------
# The simulator, as a python module
# ---------------------------------------------------------------------------


def test_a_complete_simulator_module_passes():
    f = doctor.lammps_module(healthy())
    assert f.verdict is Verdict.OK
    assert "2025.12.10" in f.detail


def test_an_absent_simulator_module_is_a_refusal_that_says_what_builds_it():
    f = doctor.lammps_module(healthy(modules={"lammps": False}, lammps=None))
    assert f.verdict is Verdict.BROKEN
    assert "PKG_PYTHON" in f.remedy and "BUILD_SHARED_LIBS" in f.remedy


def test_the_absent_module_refusal_offers_another_interpreter():
    """The simulator commonly lives in a different environment entirely."""
    f = doctor.lammps_module(healthy(modules={"lammps": False}, lammps=None))
    assert "interpreter" in f.remedy


def test_a_shared_library_that_will_not_load_is_a_refusal_naming_the_cause():
    f = doctor.lammps_module(healthy(lammps={
        "origin": "/somewhere/lammps/__init__.py",
        "shared_library": "/somewhere/lammps/liblammps.so",
        "dlopen": "libgomp.so.1: undefined symbol: __pgi_uacc_downloads",
        "entry_points": ["activate_mliappy_kokkos", "load_unified_kokkos"]}))
    assert f.verdict is Verdict.BROKEN
    assert "libgomp.so.1" in f.detail
    assert "LD_LIBRARY_PATH" in f.remedy
    # Not the other remedy: the library is built and present, and rebuilding
    # it changes nothing about which copy of a third library the loader picks.
    assert "BUILD_SHARED_LIBS" not in f.remedy


def test_a_module_with_no_shared_library_anywhere_is_a_refusal():
    f = doctor.lammps_module(healthy(lammps={
        "origin": "/somewhere/lammps/__init__.py",
        "shared_library": None,
        "dlopen": "liblammps.so: cannot open shared object file",
        "entry_points": None}))
    assert f.verdict is Verdict.BROKEN
    assert "BUILD_SHARED_LIBS" in f.remedy


def test_a_build_without_the_kokkos_interface_is_a_refusal():
    """The non-kokkos entry points exist in every build and are not these."""
    f = doctor.lammps_module(healthy(lammps={
        "origin": "/somewhere/lammps/__init__.py",
        "shared_library": "/somewhere/lammps/liblammps.so",
        "dlopen": "ok",
        "entry_points": []}))
    assert f.verdict is Verdict.BROKEN
    assert "activate_mliappy_kokkos" in f.detail


def test_one_missing_kokkos_entry_point_is_still_a_refusal():
    f = doctor.lammps_module(healthy(lammps={
        "origin": "/somewhere/lammps/__init__.py",
        "shared_library": "/somewhere/lammps/liblammps.so",
        "dlopen": "ok",
        "entry_points": ["activate_mliappy_kokkos"]}))
    assert f.verdict is Verdict.BROKEN
    assert "load_unified_kokkos" in f.detail


def test_entry_points_that_were_never_looked_at_are_not_a_refusal():
    """``None`` is "not asked", ``[]`` is "asked and there are none"."""
    f = doctor.lammps_module(healthy(lammps={
        "origin": "/somewhere/lammps/__init__.py",
        "shared_library": "/somewhere/lammps/liblammps.so",
        "dlopen": "ok",
        "entry_points": None}))
    assert f.verdict is Verdict.UNKNOWN


def test_an_interface_that_cannot_be_imported_is_a_refusal_naming_the_error():
    """Distinct from "nobody looked": the interface was looked for and threw."""
    f = doctor.lammps_module(healthy(lammps={
        "origin": "/somewhere/lammps/__init__.py",
        "shared_library": "/somewhere/lammps/liblammps.so",
        "dlopen": "ok",
        "entry_points": None,
        "entry_points_error": "ModuleNotFoundError: No module named 'lammps.mliap'"}))
    assert f.verdict is Verdict.BROKEN
    assert "lammps.mliap" in f.detail


def test_an_unanswered_probe_leaves_the_simulator_module_unknown():
    f = doctor.lammps_module(unanswered())
    assert f.verdict is Verdict.UNKNOWN


# ---------------------------------------------------------------------------
# The simulator, as a build
# ---------------------------------------------------------------------------


HELP_WITH_MLIAP = """\
Large-scale Atomic/Molecular Massively Parallel Simulator - 10 Dec 2025
Installed packages:

KOKKOS ML-IAP ML-SNAP PYTHON

KOKKOS package API: CUDA Serial
* Pair styles:

lj/cut mliap mliap/kk mliap/lmp snap zbl
"""

HELP_WITHOUT_MLIAP = """\
Large-scale Atomic/Molecular Massively Parallel Simulator - 2 Aug 2023
Installed packages:

MANYBODY MOLECULE

* Pair styles:

lj/cut eam tersoff zbl
"""

HELP_CPU_ONLY = HELP_WITH_MLIAP.replace("KOKKOS package API: CUDA Serial",
                                        "KOKKOS package API: Serial")


def test_a_build_with_the_working_pair_style_passes():
    f = doctor.lammps_build("/somewhere/lmp",
                            runner=lambda *a, **k: _Result(0, HELP_WITH_MLIAP))
    assert f.verdict is Verdict.OK
    assert "mliap/kk" in f.detail
    # Which build it is, not just that it was one: two binaries on this
    # machine differ by two years and by every package that matters.
    assert "10 Dec 2025" in f.detail


def test_a_build_without_it_is_refused_and_says_which_build_is_wanted():
    f = doctor.lammps_build("/somewhere/lmp",
                            runner=lambda *a, **k: _Result(0, HELP_WITHOUT_MLIAP))
    assert f.verdict is Verdict.BROKEN
    assert "mliap/kk" in f.detail
    assert "ML-IAP" in f.remedy and "KOKKOS" in f.remedy


def test_a_kokkos_build_with_no_device_backend_is_refused():
    """The processor-only interface fails inside the run, not at startup."""
    f = doctor.lammps_build("/somewhere/lmp",
                            runner=lambda *a, **k: _Result(0, HELP_CPU_ONLY))
    assert f.verdict is Verdict.BROKEN
    assert "CUDA" in f.detail


def test_the_broken_upstream_pair_style_is_named_when_the_build_has_it():
    help_text = HELP_WITH_MLIAP.replace("mliap mliap/kk", "mace mace/kk mliap mliap/kk")
    f = doctor.lammps_build("/somewhere/lmp",
                            runner=lambda *a, **k: _Result(0, help_text))
    assert f.verdict is Verdict.OK
    assert "mace/kk" in f.detail


def test_a_pair_style_whose_name_merely_contains_the_wanted_one_is_not_it():
    help_text = HELP_WITH_MLIAP.replace("mliap mliap/kk mliap/lmp", "mliap/kkx")
    f = doctor.lammps_build("/somewhere/lmp",
                            runner=lambda *a, **k: _Result(0, help_text))
    assert f.verdict is Verdict.BROKEN


def test_a_binary_that_cannot_print_its_help_is_refused():
    f = doctor.lammps_build("/somewhere/lmp",
                            runner=lambda *a, **k: _Result(1, "", "boom"))
    assert f.verdict is Verdict.BROKEN
    assert "boom" in (f.raw or "")


def test_a_binary_that_runs_out_of_time_is_not_inspected():
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="lmp", timeout=30.0)

    f = doctor.lammps_build("/somewhere/lmp", runner=timeout)
    assert f.verdict is Verdict.UNKNOWN


def test_a_binary_that_is_not_there_is_not_inspected():
    def missing(*a, **k):
        raise OSError("No such file or directory")

    f = doctor.lammps_build("/no/such/lmp", runner=missing)
    assert f.verdict is Verdict.UNKNOWN


def test_no_binary_named_and_none_on_the_path_is_not_inspected():
    f = doctor.lammps_build(None, which=lambda name: None)
    assert f.verdict is Verdict.UNKNOWN
    assert "does not build" in f.detail


def test_a_binary_found_on_the_path_is_used_and_named():
    f = doctor.lammps_build(None, which=lambda name: "/found/lmp",
                            runner=lambda *a, **k: _Result(0, HELP_WITH_MLIAP))
    assert f.verdict is Verdict.OK
    assert "/found/lmp" in f.detail


# ---------------------------------------------------------------------------
# Devices and the batch client
# ---------------------------------------------------------------------------


def test_a_visible_device_passes_and_is_named():
    f = doctor.devices(
        runner=lambda *a, **k: _Result(0, "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-01)\n"),
        which=lambda name: "/usr/bin/nvidia-smi")
    assert f.verdict is Verdict.OK
    assert "A100" in f.detail


def test_two_visible_devices_are_counted():
    f = doctor.devices(
        runner=lambda *a, **k: _Result(0, "GPU 0: A (UUID: 1)\nGPU 1: B (UUID: 2)\n"),
        which=lambda name: "/usr/bin/nvidia-smi")
    assert "2" in f.detail


def test_no_visible_device_is_a_refusal_when_the_route_needs_one():
    f = doctor.devices(runner=lambda *a, **k: _Result(0, "\n"),
                       which=lambda name: "/usr/bin/nvidia-smi")
    assert f.verdict is Verdict.BROKEN
    assert f.remedy


def test_no_visible_device_is_only_a_note_when_the_run_does_not_need_one():
    """One archived campaign is processor-only on purpose."""
    f = doctor.devices(required=False, runner=lambda *a, **k: _Result(0, "\n"),
                       which=lambda name: "/usr/bin/nvidia-smi")
    assert f.verdict is Verdict.OK


def test_a_driver_that_cannot_be_reached_is_a_refusal_with_its_own_words():
    f = doctor.devices(
        runner=lambda *a, **k: _Result(
            9, "", "NVIDIA-SMI has failed because it couldn't communicate with "
                   "the NVIDIA driver."),
        which=lambda name: "/usr/bin/nvidia-smi")
    assert f.verdict is Verdict.BROKEN
    assert "driver" in (f.raw or "")


def test_no_device_inspector_at_all_is_not_inspected():
    f = doctor.devices(which=lambda name: None)
    assert f.verdict is Verdict.UNKNOWN


def test_a_device_query_that_runs_out_of_time_is_not_inspected():
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=30.0)

    f = doctor.devices(runner=timeout, which=lambda name: "/usr/bin/nvidia-smi")
    assert f.verdict is Verdict.UNKNOWN


def test_an_emptied_device_selection_is_reported_because_it_hides_the_device():
    f = doctor.devices(environ={"CUDA_VISIBLE_DEVICES": ""},
                       runner=lambda *a, **k: _Result(0, "GPU 0: A (UUID: 1)\n"),
                       which=lambda name: "/usr/bin/nvidia-smi")
    assert f.verdict is Verdict.BROKEN
    assert "CUDA_VISIBLE_DEVICES" in f.detail


def test_a_batch_client_that_is_there_passes():
    f = doctor.batch_client(which=lambda name: "/opt/pbs/bin/qsub")
    assert f.verdict is Verdict.OK
    assert "/opt/pbs/bin/qsub" in f.detail


def test_no_batch_client_is_only_a_note_until_a_campaign_wants_one():
    f = doctor.batch_client(which=lambda name: None)
    assert f.verdict is Verdict.OK
    assert "local" in f.detail


def test_no_batch_client_is_a_refusal_when_the_campaign_asked_to_submit():
    f = doctor.batch_client(required=True, which=lambda name: None)
    assert f.verdict is Verdict.BROKEN
    assert f.remedy


def test_the_doctor_names_no_queue():
    """Which queue a job belongs in is a site fact and a campaign's choice."""
    source = pathlib.Path(doctor.__file__).read_text()
    assert " -q " not in source


# ---------------------------------------------------------------------------
# A model file
# ---------------------------------------------------------------------------


class _FakePotential:
    def __init__(self, fallback):
        self.fallback_kernels = fallback
        self.path = pathlib.Path("/somewhere/model.pt")
        self.species = ("A", "B")
        self.dtype = "float32"


def test_no_model_named_is_not_inspected():
    f = doctor.potential(None, engine="mace")
    assert f.verdict is Verdict.UNKNOWN
    assert "model" in f.detail


def test_a_clean_model_passes():
    f = doctor.potential("/somewhere/model.pt", engine="mace",
                         resolver=lambda *a, **k: _FakePotential(()))
    assert f.verdict is Verdict.OK


def test_a_model_with_downgraded_kernels_is_reported_with_its_nodes():
    f = doctor.potential(
        "/somewhere/model.pt", engine="mace",
        resolver=lambda *a, **k: _FakePotential(("interactions.0.conv_tp.f",)))
    assert f.verdict is Verdict.DEGRADED
    assert "interactions.0.conv_tp.f" in f.detail
    assert f.remedy


def test_the_downgrade_names_what_it_costs_and_where_it_came_from():
    f = doctor.potential(
        "/somewhere/model.pt", engine="mace",
        resolver=lambda *a, **k: _FakePotential(("a",)))
    assert "2.49" in f.detail
    assert "convert" in (f.detail + f.remedy)


def test_a_model_that_could_not_be_walked_is_not_a_clean_bill_of_health():
    f = doctor.potential("/somewhere/model.pt", engine="mace",
                         resolver=lambda *a, **k: _FakePotential(None))
    assert f.verdict is Verdict.UNKNOWN


def test_a_model_the_stack_refuses_is_a_refusal_with_the_reason_kept():
    from aipf.md.potentials import PotentialError

    def refuse(*a, **k):
        raise PotentialError("there is no file at /somewhere/model.pt")

    f = doctor.potential("/somewhere/model.pt", engine="mace", resolver=refuse)
    assert f.verdict is Verdict.BROKEN
    assert "no file" in f.detail


def test_a_model_named_without_its_engine_is_a_caller_error():
    """The engine is never defaulted; assuming one writes a false record."""
    with pytest.raises(ValueError, match="engine"):
        doctor.examine(model="/somewhere/model.pt",
                       prober=canned(unanswered()))


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def test_the_probe_reads_this_interpreter_without_importing_anything_heavy():
    p = doctor.probe(sys.executable)
    assert p.facts is not None
    assert p.facts["executable"]
    # The probe reports which expensive libraries it ended up holding. Asking
    # the metadata costs milliseconds; importing these costs minutes, and a
    # doctor that imports them has stopped being a doctor.
    assert p.facts["heavy_imports"] == []


def test_the_probe_reports_what_is_installed_and_what_is_merely_present():
    p = doctor.probe(sys.executable)
    assert set(p.facts["versions"]) >= {"e3nn", "torch", "lammps"}
    assert set(p.facts["modules"]) >= {"torch", "lammps"}


def test_an_interpreter_that_is_not_there_answers_with_a_reason_not_a_crash():
    p = doctor.probe("/no/such/python")
    assert p.facts is None
    assert "/no/such/python" in p.reason


def test_a_probe_that_runs_out_of_time_answers_with_a_reason():
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="python", timeout=60.0)

    p = doctor.probe(sys.executable, runner=timeout)
    assert p.facts is None
    assert "60" in p.reason


def test_a_probe_whose_output_is_not_readable_answers_with_a_reason():
    p = doctor.probe(sys.executable,
                     runner=lambda *a, **k: _Result(0, "not json at all"))
    assert p.facts is None
    assert p.raw


def test_the_probe_does_not_write_bytecode_into_someone_elses_environment(
        monkeypatch):
    """It runs foreign interpreters, whose directories are not ours to touch.

    The variable is deleted first: this suite is usually run with it already
    set, and a test that passes because of what it inherited proves nothing.
    """
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    captured = {}

    def runner(cmd, **kwargs):
        captured.update(kwargs)
        return _Result(0, "{}")

    doctor.probe(sys.executable, runner=runner)
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_the_probe_inherits_the_environment_a_job_would_inherit(monkeypatch):
    """The loader path is the thing under test; scrubbing it hides the defect."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/somewhere/that/is/wrong")
    captured = {}

    def runner(cmd, **kwargs):
        captured.update(kwargs)
        return _Result(0, "{}")

    doctor.probe(sys.executable, runner=runner)
    assert captured["env"]["LD_LIBRARY_PATH"] == "/somewhere/that/is/wrong"


def test_the_probe_passes_on_an_environment_it_was_given():
    captured = {}

    def runner(cmd, **kwargs):
        captured.update(kwargs)
        return _Result(0, "{}")

    doctor.probe(sys.executable, env={"LD_LIBRARY_PATH": "/x", "PATH": "/bin"},
                 runner=runner)
    assert captured["env"]["LD_LIBRARY_PATH"] == "/x"


# ---------------------------------------------------------------------------
# The whole examination
# ---------------------------------------------------------------------------


def test_every_check_appears_in_the_result_even_when_it_was_not_run():
    """A check left out of the report reads as a check that passed."""
    d = doctor.examine(prober=canned(unanswered()))
    assert tuple(f.check for f in d.findings) == doctor.CHECKS


def test_a_completely_unanswerable_environment_reports_nothing_broken():
    """Nothing inspected means nothing found wrong. It does not mean healthy."""
    d = doctor.examine(prober=canned(unanswered()), which=lambda name: None,
                       environ={})
    assert d.broken == ()
    assert d.can_run is None


def test_the_examination_passes_its_own_probe_through_once():
    """Four checks read one probe. Probing four times is four subprocesses."""
    prober = canned(healthy())
    doctor.examine(python="/somewhere/bin/python", prober=prober,
                   which=lambda name: None, environ={})
    assert prober.calls == ["/somewhere/bin/python"]


def test_asking_for_the_deep_check_runs_it():
    d = doctor.examine(prober=canned(healthy()), deep=True,
                       which=lambda name: None, environ={},
                       kernel_runner=lambda *a, **k: _Result(0, "", ""))
    assert d.find("accelerator_kernels").verdict is Verdict.OK


def test_a_named_check_that_does_not_exist_is_a_caller_error():
    with pytest.raises(KeyError):
        doctor.examine(prober=canned(unanswered())).find("no_such_check")


def test_the_report_is_readable_as_one_block_of_text():
    text = str(doctor.examine(prober=canned(unanswered()),
                              which=lambda name: None, environ={}))
    for check in doctor.CHECKS:
        assert check in text


# ---------------------------------------------------------------------------
# Against the real machine, including deliberately broken loader environments
# ---------------------------------------------------------------------------


@pytest.mark.env
def test_the_simulators_own_environment_passes_every_module_check():
    if not LAMMPS_PY.exists():
        pytest.skip(f"the simulator's interpreter is not at {LAMMPS_PY} (AIPF_LAMMPS_PYTHON or [site] lammps_python)")
    clean = {"PATH": "/usr/bin:/bin"}
    f = doctor.lammps_module(doctor.probe(LAMMPS_PY, env=clean))
    assert f.verdict is Verdict.OK, f.detail


@pytest.mark.env
def test_a_poisoned_loader_path_breaks_the_simulator_library_and_is_caught():
    """A real failure, not a simulated one.

    With the site compiler directory ahead of the environment's own, loading
    the simulator's shared library fails on an unresolved symbol in another
    library entirely. This is the configuration a login shell on this machine
    hands a job by default.
    """
    if not LAMMPS_PY.exists():
        pytest.skip(f"the simulator's interpreter is not at {LAMMPS_PY} (AIPF_LAMMPS_PYTHON or [site] lammps_python)")
    if not POISONED_LD.is_dir():
        pytest.skip(f"no poisoning directory at {POISONED_LD} (AIPF_POISONED_LD)")
    poisoned = {"PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": str(POISONED_LD)}
    f = doctor.lammps_module(doctor.probe(LAMMPS_PY, env=poisoned))
    assert f.verdict is Verdict.BROKEN
    assert "LD_LIBRARY_PATH" in f.remedy


@pytest.mark.env
def test_the_real_binary_with_the_working_pair_style_passes():
    declared_roots.lammps_or_skip(declared_roots.MLIAP_KK)
    f = doctor.lammps_build(MLIAP_LMP)
    assert f.verdict is Verdict.OK, f.detail
    assert "mliap/kk" in f.detail


@pytest.mark.env
def test_the_fast_kernels_really_do_not_load_under_the_inherited_loader_path():
    """The hazard is loader order, not node type: measured on a device node.

    With the site CUDA directory first the import fails outright; with the
    environment's own two directories first it succeeds. A conversion run in
    the first configuration bakes the slow path into the model for good.
    """
    if not AIPF_PY.exists():
        pytest.skip(f"the package environment is not at {AIPF_PY} (AIPF_ENV_PREFIX)")
    if not SITE_CUDA_LIB.is_dir():
        pytest.skip(f"no site CUDA directory at {SITE_CUDA_LIB} (AIPF_CUDA_LIB or [site] cuda_lib)")
    poisoned = {"PATH": "/usr/bin:/bin",
                "LD_LIBRARY_PATH": str(SITE_CUDA_LIB)}
    f = doctor.kernels(python=AIPF_PY, env=poisoned)
    assert f.verdict is Verdict.DEGRADED
    assert "cue_ops" in (f.raw or "")


@pytest.mark.env
def test_the_fast_kernels_load_with_the_environments_own_libraries_first():
    if not AIPF_PY.exists():
        pytest.skip(f"the package environment is not at {AIPF_PY} (AIPF_ENV_PREFIX)")
    site = AIPF_PY.parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia"
    good = {"PATH": "/usr/bin:/bin",
            "LD_LIBRARY_PATH": f"{site / 'cublas' / 'lib'}:{site / 'cuda_nvrtc' / 'lib'}"}
    f = doctor.kernels(python=AIPF_PY, env=good)
    assert f.verdict is Verdict.OK


@pytest.mark.env
def test_the_cheap_check_predicts_what_the_expensive_one_measures():
    """The two answers are worth having only if they agree."""
    if not AIPF_PY.exists():
        pytest.skip(f"the package environment is not at {AIPF_PY} (AIPF_ENV_PREFIX)")
    if not SITE_CUDA_LIB.is_dir():
        pytest.skip(f"no site CUDA directory at {SITE_CUDA_LIB} (AIPF_CUDA_LIB or [site] cuda_lib)")
    poisoned = {"PATH": "/usr/bin:/bin",
                "LD_LIBRARY_PATH": str(SITE_CUDA_LIB)}
    cheap = doctor.accelerator_libraries(doctor.probe(AIPF_PY, env=poisoned))
    deep = doctor.kernels(python=AIPF_PY, env=poisoned)
    assert cheap.verdict is deep.verdict is Verdict.DEGRADED


@pytest.mark.env
def test_the_production_model_is_not_a_downgraded_one():
    if not PRODUCTION_MODEL.exists():
        pytest.skip(f"the production model is not at {PRODUCTION_MODEL} (under the hhe raw root: AIPF_RAW_HHE, AIPF_RAW or [paths.raw])")
    f = doctor.potential(PRODUCTION_MODEL, engine="mace")
    assert f.verdict is Verdict.OK, f.detail


@pytest.mark.env
def test_a_whole_examination_of_this_machine_answers_every_check():
    if not AIPF_PY.exists():
        pytest.skip(f"the package environment is not at {AIPF_PY} (AIPF_ENV_PREFIX)")
    d = doctor.examine(python=AIPF_PY, lammps_binary=MLIAP_LMP)
    assert tuple(f.check for f in d.findings) == doctor.CHECKS
    # Whatever this machine is today, no check may answer with nothing.
    for f in d.findings:
        assert f.detail


def test_the_package_exposes_the_examination_under_a_name_that_says_so():
    """A bare ``examine`` or ``probe`` says nothing at the package level."""
    import aipf.md as md

    assert md.examine_environment is doctor.examine
    assert md.probe_interpreter is doctor.probe
    assert set(md.__all__) >= {"Diagnosis", "Finding", "Probe", "Verdict",
                               "examine_environment", "probe_interpreter"}


# ---------------------------------------------------------------------------
# Tests the mutation sweep asked for
# ---------------------------------------------------------------------------


def test_the_probe_answers_for_everything_the_checks_read():
    """A fact no check reads is dead weight; a fact a check reads and the

    probe never collects turns that check into a false alarm about a library
    that is installed.
    """
    p = doctor.probe(sys.executable)
    assert set(p.facts["versions"]) >= {
        "e3nn", doctor.CUBLAS_DISTRIBUTION, doctor.KERNEL_DISTRIBUTION,
        doctor.SIMULATOR_MODULE}
    assert set(p.facts["modules"]) >= {doctor.KERNEL_MODULE,
                                       doctor.SIMULATOR_MODULE}


def test_the_expensive_libraries_the_probe_must_not_import_are_named():
    """Otherwise "it imported nothing expensive" is true of an empty list."""
    assert set(doctor._HEAVY) >= {"torch", doctor.KERNEL_MODULE}


def test_an_installed_version_is_a_version_and_an_absent_one_is_nothing():
    """The two are different answers, and an empty string is neither."""
    p = doctor.probe(sys.executable)
    for name, version in p.facts["versions"].items():
        assert version is None or (isinstance(version, str) and version), name


def test_the_probe_says_which_modules_are_there_not_which_are_not():
    p = doctor.probe(sys.executable)
    # torch is a hard dependency of this package, so this interpreter has it.
    assert p.facts["modules"]["torch"] is True


def test_an_interpreter_with_no_accelerator_libraries_reports_none(tmp_path):
    """A fresh interpreter with nothing installed, which is a real starting

    point. The directories must come back as absent rather than as paths that
    are not there, because the loader check then blames the loader path for a
    directory that was never built.
    """
    import venv

    prefix = tmp_path / "bare"
    venv.EnvBuilder(with_pip=False).create(prefix)
    p = doctor.probe(prefix / "bin" / "python")
    assert p.facts is not None, p.reason
    assert p.facts["accelerator_dirs"] == {"cublas": None, "cuda_nvrtc": None}
    assert p.facts["lammps"] is None


def test_empty_entries_in_the_loader_path_are_dropped():
    """An empty entry means the working directory to the loader, and it

    matches no directory this check looks for.
    """
    p = doctor.probe(sys.executable,
                     env={"PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": "/a::/b:"})
    assert p.facts["ld_library_path"] == ["/a", "/b"]


def test_a_simulator_module_with_neither_library_nor_interface(tmp_path):
    """Built from a real module on a real path, not from a fake record.

    A module that shadows the real one has no shared library and no interface
    submodule, which is what a half-installed or hand-copied tree looks like.
    """
    package = tmp_path / "lammps"
    package.mkdir()
    (package / "__init__.py").write_text("")
    p = doctor.probe(sys.executable,
                     env={**os.environ, "PYTHONPATH": str(tmp_path)})
    found = p.facts["lammps"]
    assert found["shared_library"] is None
    # Loading the library by name can still succeed: an environment with LAMMPS built into it
    # (env/build-lammps.sh) has liblammps.so on its interpreter's own library path. The verdict
    # does not rest on that: the interface the model is loaded through is missing.
    assert found["entry_points"] is None
    assert "lammps.mliap" in found["entry_points_error"]
    assert doctor.lammps_module(p).verdict is Verdict.BROKEN


def test_an_interpreter_that_exits_says_so_and_keeps_what_it_printed():
    """Measured: an interpreter older than the metadata library answers this

    way, and the reason it could not be asked is in what it printed.
    """
    p = doctor.probe("/bin/false")
    assert p.facts is None
    assert "1" in p.reason and "exited" in p.reason


def test_output_that_is_not_a_record_is_reported_with_the_reason_and_the_text():
    p = doctor.probe("/bin/echo")
    assert p.facts is None
    assert "/bin/echo" in p.reason
    assert "readable" in p.reason
    assert p.raw


def test_kept_output_is_the_end_of_it_and_is_bounded():
    """The cause of a failure is at the end, and a whole log is not a finding."""
    long = "\n".join(f"line {n}" for n in range(40))
    f = doctor.lammps_build("/somewhere/lmp",
                            runner=lambda *a, **k: _Result(1, "", long))
    assert "line 39" in f.raw
    assert "line 0" not in f.raw
    assert len(f.raw.splitlines()) <= 10


def test_the_allocator_option_is_matched_whatever_case_it_is_written_in():
    f = doctor.alloc_conf(
        {"PYTORCH_CUDA_ALLOC_CONF": "Expandable_Segments:TRUE"})
    assert f.verdict is Verdict.BROKEN


def test_an_allocator_setting_that_is_empty_reads_as_unset():
    f = doctor.alloc_conf({"PYTORCH_CUDA_ALLOC_CONF": ""})
    assert f.verdict is Verdict.OK
    assert "not set" in f.detail


def test_library_versions_compare_as_numbers_not_as_text():
    """A one-digit major sorts after a two-digit one as text and before it

    as a version, and the older library is the one that disables the kernels.
    """
    f = doctor.accelerator_libraries(
        healthy(versions={"nvidia-cublas-cu12": "9.10.0",
                          "cuequivariance-ops-torch-cu12": "0.9.1"}))
    assert f.verdict is Verdict.DEGRADED
    assert "9.10.0" in f.detail


def test_exactly_the_minimum_library_version_is_enough():
    """The bound is measured at this version, so it is included, not excluded."""
    f = doctor.accelerator_libraries(
        healthy(versions={"nvidia-cublas-cu12": "12.5",
                          "cuequivariance-ops-torch-cu12": "0.9.1"}))
    assert f.verdict is Verdict.OK


def test_a_missing_math_library_is_reported_rather_than_crashed_on():
    f = doctor.accelerator_libraries(healthy(versions={}))
    assert f.verdict is Verdict.DEGRADED
    assert "nvidia-cublas-cu12" in f.detail


def test_a_cuda_directory_ahead_is_recognised_whatever_case_it_is_spelled_in():
    f = doctor.accelerator_libraries(healthy(ld_library_path=[
        "/opt/NVIDIA/CUDA/lib64",
        "/somewhere/nvidia/cublas/lib",
        "/somewhere/nvidia/cuda_nvrtc/lib"]))
    assert f.verdict is Verdict.DEGRADED


def test_the_deep_check_asks_in_the_environment_it_was_given():
    """The loader path IS the question; asking in another one answers nothing."""
    captured = {}

    def runner(cmd, **kwargs):
        captured.update(kwargs)
        return _Result(0, "", "")

    doctor.kernels(sys.executable, env={"LD_LIBRARY_PATH": "/x", "PATH": "/bin"},
                   runner=runner)
    assert captured["env"]["LD_LIBRARY_PATH"] == "/x"


def test_each_check_looks_for_its_tool_under_the_name_users_type():
    """Written out, not taken from the constant: a test that reads the

    constant it is checking agrees with whatever the constant says, which is
    the one thing it must not do.
    """
    asked = []
    doctor.lammps_build(None, which=lambda name: asked.append(name),
                        runner=lambda *a, **k: _Result(0, HELP_WITH_MLIAP))
    doctor.devices(which=lambda name: asked.append(name))
    doctor.batch_client(which=lambda name: asked.append(name))
    assert asked == ["lmp", "nvidia-smi", "qsub"]


def test_the_shared_library_is_looked_for_under_the_name_it_is_built_with():
    """Same argument, for the one name that is never typed by a user."""
    assert doctor.SIMULATOR_LIBRARY == "liblammps.so"
    assert doctor.SIMULATOR_ENTRY_POINTS == ("activate_mliappy_kokkos",
                                             "load_unified_kokkos")


def test_a_style_list_that_separates_its_names_with_commas_is_read():
    help_text = HELP_WITH_MLIAP.replace("lj/cut mliap mliap/kk mliap/lmp snap zbl",
                                        "lj/cut, mliap, mliap/kk, snap, zbl")
    f = doctor.lammps_build("/somewhere/lmp",
                            runner=lambda *a, **k: _Result(0, help_text))
    assert f.verdict is Verdict.OK


def test_a_build_whose_help_does_not_say_its_backend_is_not_inspected():
    """Having the pair style and not knowing the backend is not a pass."""
    help_text = HELP_WITH_MLIAP.replace("KOKKOS package API: CUDA Serial", "")
    f = doctor.lammps_build("/somewhere/lmp",
                            runner=lambda *a, **k: _Result(0, help_text))
    assert f.verdict is Verdict.UNKNOWN


def test_a_build_with_no_banner_is_still_named():
    help_text = HELP_WITH_MLIAP.replace(
        "Large-scale Atomic/Molecular Massively Parallel Simulator - 10 Dec 2025", "")
    f = doctor.lammps_build("/somewhere/lmp",
                            runner=lambda *a, **k: _Result(0, help_text))
    assert "/somewhere/lmp" in f.detail


def test_a_device_inspector_that_cannot_be_run_is_not_inspected():
    def missing(*a, **k):
        raise OSError("No such file or directory")

    f = doctor.devices(runner=missing, which=lambda name: "/usr/bin/nvidia-smi")
    assert f.verdict is Verdict.UNKNOWN


def test_an_emptied_selection_is_not_a_refusal_when_no_device_is_needed():
    f = doctor.devices(required=False, environ={"CUDA_VISIBLE_DEVICES": ""},
                       runner=lambda *a, **k: _Result(0, "GPU 0: A (UUID: 1)\n"),
                       which=lambda name: "/usr/bin/nvidia-smi")
    assert f.verdict is Verdict.OK


def test_a_model_handed_straight_to_the_check_still_needs_its_engine():
    with pytest.raises(ValueError, match="engine"):
        doctor.potential("/somewhere/model.pt")


def test_a_model_that_needs_a_library_this_interpreter_lacks_is_not_inspected():
    def missing(*a, **k):
        raise ImportError("No module named 'torch'")

    f = doctor.potential("/somewhere/model.pt", engine="mace", resolver=missing)
    assert f.verdict is Verdict.UNKNOWN


def test_a_caller_error_costs_no_subprocess():
    """The refusal is repeated in the examination for this reason alone."""
    prober = canned(unanswered())
    with pytest.raises(ValueError, match="engine"):
        doctor.examine(model="/somewhere/model.pt", prober=prober)
    assert prober.calls == []


def test_the_examination_passes_on_what_the_campaign_said_it_needs():
    """The device lookup is pointed at a program that lists nothing, which is

    what a node with no device looks like to this check. Whether that is a
    refusal is the campaign's call, and the examination has to pass it on.
    """
    def tools(name):
        return "/bin/true" if name == "nvidia-smi" else None

    needs_one = doctor.examine(prober=canned(healthy()), environ={},
                               require_device=True, which=tools)
    assert needs_one.find("devices").verdict is Verdict.BROKEN

    d = doctor.examine(prober=canned(healthy()), environ={},
                       require_device=False, require_batch_client=True,
                       which=tools)
    assert d.find("devices").verdict is Verdict.OK
    assert d.find("batch_client").verdict is Verdict.BROKEN


def test_the_examination_passes_on_the_environment_it_was_given():
    d = doctor.examine(
        prober=canned(healthy()), which=lambda name: None,
        environ={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert d.find("allocator").verdict is Verdict.BROKEN


def test_the_examination_passes_on_the_binary_it_was_given():
    asked = []
    d = doctor.examine(prober=canned(healthy()), environ={},
                       which=lambda name: asked.append(name),
                       lammps_binary="/somewhere/lmp")
    assert "/somewhere/lmp" in d.find("lammps_build").detail
    # The binary was named, so nothing looked one up; the other two checks did.
    assert asked == ["nvidia-smi", "qsub"]


def test_the_examination_passes_on_the_lookup_it_was_given():
    """Without this the build check would quietly consult the real PATH and

    report on a binary the caller never named.
    """
    asked = []
    d = doctor.examine(prober=canned(healthy()), environ={},
                       which=lambda name: asked.append(name))
    assert asked == ["lmp", "nvidia-smi", "qsub"]
    assert d.find("lammps_build").verdict is Verdict.UNKNOWN


def test_the_examination_passes_on_an_engine_that_does_not_need_the_library():
    d = doctor.examine(prober=canned(healthy()), environ={},
                       which=lambda name: None, require_e3nn=None)
    assert d.find("deserialiser").verdict is Verdict.UNKNOWN


def test_the_examination_resolves_the_model_it_was_given():
    """A path with nothing at it is refused before anything is loaded."""
    d = doctor.examine(prober=canned(healthy()), environ={},
                       which=lambda name: None,
                       model="/no/such/model.pt", engine="mace")
    f = d.find("potential")
    assert f.verdict is Verdict.BROKEN
    assert "/no/such/model.pt" in f.detail


def test_the_report_follows_the_order_the_checks_are_declared_in():
    """The findings are built in that order; nothing re-sorts them afterwards."""
    d = doctor.examine(prober=canned(healthy()), environ={},
                       which=lambda name: None)
    assert tuple(f.check for f in d.findings) == doctor.CHECKS


def test_the_tool_timeout_is_far_above_what_a_build_takes_to_answer():
    """Measured on this machine: 0.06 s for one binary, 2.4 s for the other."""
    assert doctor.TOOL_TIMEOUT_S >= 30.0


def test_a_finding_must_name_a_check():
    with pytest.raises(ValueError, match="check"):
        Finding("", Verdict.OK, "something")


def test_a_diagnosis_refuses_anything_that_is_not_a_finding():
    with pytest.raises(TypeError, match="Finding"):
        Diagnosis(("devices: ok",))


def test_a_record_carries_everything_the_finding_carries():
    f = Finding("devices", Verdict.BROKEN, "no device", remedy="ask for one",
                raw="nothing")
    record = Diagnosis((f,)).to_records()[0]
    assert record == {"check": "devices", "verdict": "broken",
                      "detail": "no device", "remedy": "ask for one",
                      "raw": "nothing"}


