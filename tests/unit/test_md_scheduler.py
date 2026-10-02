"""Submitting a campaign, and refusing one that would be lost.

Two tiers, same as the potential resolver's. The local backend is exercised
for real: it starts processes, fills lanes, kills what overruns, and the tests
read the log files it wrote. The batch backend is driven against FAKE farm
commands -- small executables that log what they were asked and answer with
output copied from this farm's real answers -- because a test that submitted
to the real queue would cost an allocation and could not be run twice.

The canned answers below are transcriptions, not inventions. Every state
letter, every message and every exit status in this file was measured with
``qstat`` on the farm this package runs on.
"""
import json
import os
import pathlib
import stat
import sys
import time

import pytest

from aipf.md.request import StatePoint
from aipf.md.scheduler import (
    BACKENDS,
    Handle,
    Job,
    JobState,
    LocalBackend,
    PartialSubmission,
    PbsBackend,
    PBS_STATES,
    Report,
    SchedulerError,
    TERMINAL,
    WALLTIME_ENV,
    backend,
    jobs_for,
    register_backend,
)
from aipf.md.scheduler import _parse_attributes, _walltime

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _job(tmp_path, name="run1", script="echo hello", **kwargs):
    kwargs.setdefault("workdir", tmp_path / name)
    return Job(name=name, script=script, **kwargs)


def _local(tmp_path, lanes=1, **kwargs):
    return LocalBackend(lanes=lanes, **kwargs)


#: One fake farm command. It appends its arguments to a log and answers from a
#: plan, so a test can assert both what was asked and how many times.
_FAKE = '''
import json, pathlib, sys
plan = json.loads(pathlib.Path(__file__ + ".plan").read_text())
args = sys.argv[1:]
log = pathlib.Path(plan["log"])
with log.open("a") as fh:
    fh.write(" ".join(args) + "\\n")
call = sum(1 for _ in log.open())
for rule in plan["rules"]:
    match = rule.get("match")
    if match is not None and match not in " ".join(args):
        continue
    if rule.get("nth") is not None and rule["nth"] != call:
        continue
    sys.stdout.write(rule.get("out", ""))
    sys.stderr.write(rule.get("err", ""))
    sys.exit(rule.get("rc", 0))
sys.exit(0)
'''


def _fake_command(tmp_path, name, rules, log=None):
    """Write an executable that answers from ``rules`` and logs its calls."""
    path = tmp_path / name
    log = log if log is not None else tmp_path / f"{name}.log"
    path.write_text(f"#!{sys.executable}\n{_FAKE}")
    path.chmod(0o755)
    (tmp_path / f"{name}.plan").write_text(
        json.dumps({"log": str(log), "rules": rules}))
    log.write_text("")
    return str(path), log


#: What ``qstat -Qf`` prints for a queue that is taking work here.
_QUEUE_OK = ("Queue: ai\n"
             "    queue_type = Route\n"
             "    route_destinations = aiq1,aiq2,aiq3\n"
             "    enabled = True\n"
             "    started = True\n")


def _record(state, *, exit_status=None, queue="aiq1", job_id="1.srv"):
    """A ``qstat -x -f`` record, in the layout the real command prints."""
    lines = [f"Job Id: {job_id}", "    Job_Name = probe",
             f"    job_state = {state}", f"    queue = {queue}",
             "    server = srv"]
    if exit_status is not None:
        lines.append(f"    Exit_status = {exit_status}")
    return "\n".join(lines) + "\n"


def _pbs(tmp_path, *, qsub_rules=None, qstat_rules=None, qdel_rules=None,
         **kwargs):
    """A batch backend wired to fake farm commands, and their logs."""
    qsub, qsub_log = _fake_command(
        tmp_path, "qsub", qsub_rules if qsub_rules is not None
        else [{"out": "1.srv\n"}])
    qstat, qstat_log = _fake_command(
        tmp_path, "qstat", qstat_rules if qstat_rules is not None
        else [{"match": "-Qf", "out": _QUEUE_OK}])
    qdel, qdel_log = _fake_command(
        tmp_path, "qdel", qdel_rules if qdel_rules is not None else [{}])
    kwargs.setdefault("queue", "ai")
    kwargs.setdefault("project", "12345678")
    back = PbsBackend(qsub=qsub, qstat=qstat, qdel=qdel, **kwargs)
    return back, {"qsub": qsub_log, "qstat": qstat_log, "qdel": qdel_log}


def _calls(log):
    return [line for line in log.read_text().splitlines() if line]


def _alive(pid):
    """Whether a process id still exists. A reaped process counts as gone."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _until(condition, seconds=10.0):
    """Poll a condition, so that no test depends on a scheduling moment."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def _never(condition, seconds=1.0):
    """Watch for a while that something does NOT happen.

    A glance is not enough where the thing being watched for would be started
    by the very pump that ended the wait.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return False
        time.sleep(0.02)
    return True


# ==========================================================================
# Job: what can be asked for
# ==========================================================================


@pytest.mark.parametrize("name", ["hhe_anchors_200", "feb_cube0",
                                  "nucl_sweep_p1", "cube_x0.50_T1400_s1",
                                  "a", "eos_P.0"])
def test_a_job_accepts_every_archived_job_name_shape(tmp_path, name):
    """The names the archive actually used must all be legal.

    Taken from the archived job files and from a state point's own tag: two
    of them carry a dot, which a stricter rule would refuse, and the tag form
    is what a campaign names its jobs by. The derived file names are asserted
    too, so that a constructor which merely accepted the name and derived
    nothing from it would not pass.
    """
    job = _job(tmp_path, name=name)
    assert job.name == name
    assert job.script_path.name == f"{name}.sh"
    assert job.stdout.name == f"{name}.out"


@pytest.mark.parametrize("name", ["", "  ", "two words", "a/b", "-lead",
                                  ".hidden", "with\ttab", "x\ny"])
def test_a_job_name_that_is_not_a_file_name_is_refused(tmp_path, name):
    """A name is a file stem and a farm label at once, so it obeys both."""
    with pytest.raises(SchedulerError, match="not usable"):
        _job(tmp_path, name=name)


def test_a_job_name_that_is_not_a_string_is_refused(tmp_path):
    with pytest.raises(SchedulerError, match="not usable"):
        Job(name=3, script="echo", workdir=tmp_path)


@pytest.mark.parametrize("script", ["", "   \n\t "])
def test_a_job_with_no_script_is_refused(tmp_path, script):
    with pytest.raises(SchedulerError, match="empty script"):
        _job(tmp_path, script=script)


def test_a_script_that_is_not_text_is_refused(tmp_path):
    with pytest.raises(SchedulerError, match="empty script"):
        Job(name="x", script=None, workdir=tmp_path)


@pytest.mark.parametrize("key", ["1BAD", "with-dash", "", "a b", "a="])
def test_an_environment_name_a_shell_cannot_hold_is_refused(tmp_path, key):
    with pytest.raises(SchedulerError, match="not an environment variable"):
        _job(tmp_path, env={key: "1"})


@pytest.mark.parametrize("value", ["a\nb", "a\0b"])
def test_an_environment_value_that_cannot_be_transmitted_is_refused(
        tmp_path, value):
    """A value spanning lines does not survive being passed with a job."""
    with pytest.raises(SchedulerError, match="spans lines or holds a null"):
        _job(tmp_path, env={"A": value})


def test_a_non_string_environment_value_is_written_as_text(tmp_path):
    """A shard number is an integer to the caller and text to the farm."""
    assert _job(tmp_path, env={"SHARD": 3}).job_env["SHARD"] == "3"


def test_a_job_may_ask_for_a_variable_to_be_removed(tmp_path):
    """``None`` means unset, which is the only way to say it on both farms."""
    job = _job(tmp_path, env={"KEEP": "1", "DROP": None})
    assert job.job_env == {"KEEP": "1"}
    assert job.unset_env == ("DROP",)


def test_an_empty_value_is_passed_and_is_not_a_removal(tmp_path):
    """An archived job builds an empty argument string and passes it on."""
    job = _job(tmp_path, env={"RERUN_ARGS": ""})
    assert job.job_env["RERUN_ARGS"] == ""
    assert job.unset_env == ()


def test_the_removals_keep_the_order_they_were_given_in(tmp_path):
    job = _job(tmp_path, env={"B": None, "A": None, "C": "1"})
    assert job.unset_env == ("B", "A")


def test_a_job_that_sets_the_walltime_variable_itself_is_refused(tmp_path):
    """The archived campaign that stated its budget twice warns about it."""
    with pytest.raises(SchedulerError, match="told its budget twice"):
        _job(tmp_path, env={WALLTIME_ENV: "10"}, walltime_s=10.0)


def test_the_walltime_is_exported_to_the_job(tmp_path):
    job = _job(tmp_path, walltime_s=3600.0)
    assert job.job_env[WALLTIME_ENV] == repr(3600.0)


def test_a_job_with_no_walltime_exports_none(tmp_path):
    assert WALLTIME_ENV not in _job(tmp_path).job_env


@pytest.mark.parametrize("walltime", [0, -1.0, float("inf"), float("nan")])
def test_a_walltime_that_is_not_a_positive_duration_is_refused(
        tmp_path, walltime):
    with pytest.raises(SchedulerError, match="positive finite"):
        _job(tmp_path, walltime_s=walltime)


@pytest.mark.parametrize("walltime", ["3600", True, None.__class__])
def test_a_walltime_that_is_not_a_number_is_refused(tmp_path, walltime):
    with pytest.raises(SchedulerError, match="must be a number"):
        _job(tmp_path, walltime_s=walltime)


def test_the_two_streams_default_to_the_job_name_in_its_workdir(tmp_path):
    """The archived files name them exactly this way beside the job."""
    job = _job(tmp_path, name="anchors_200GPa")
    assert job.stdout == job.workdir / "anchors_200GPa.out"
    assert job.stderr == job.workdir / "anchors_200GPa.err"
    assert job.script_path == job.workdir / "anchors_200GPa.sh"
    assert not job.joins_streams


def test_naming_one_file_for_both_streams_is_how_joining_is_asked_for(tmp_path):
    one = tmp_path / "both.log"
    assert _job(tmp_path, stdout=one, stderr=one).joins_streams


def test_a_workdir_given_as_text_becomes_a_path(tmp_path):
    assert _job(tmp_path, workdir=str(tmp_path)).workdir == tmp_path


def test_a_stream_given_as_text_becomes_a_path(tmp_path):
    """The backends open these, so text here is a crash at submission."""
    job = _job(tmp_path, stdout=str(tmp_path / "a.out"),
               stderr=str(tmp_path / "a.err"))
    assert job.stdout == tmp_path / "a.out"
    assert job.stderr == tmp_path / "a.err"


# ==========================================================================
# Report: what may be claimed
# ==========================================================================


def test_a_finished_report_carries_a_zero_exit(tmp_path):
    report = Report(JobState.FINISHED, exit_code=0)
    assert report.is_terminal
    assert report.to_record() == {"state": "finished", "exit_code": 0,
                                  "detail": "", "raw": None}


def test_a_record_carries_the_exit_code_it_was_given():
    """Including the two that are not zero: a real code, and none at all."""
    assert Report(JobState.FAILED, exit_code=3,
                  detail="exited 3").to_record()["exit_code"] == 3
    assert Report(JobState.UNKNOWN, detail="purged").to_record() == {
        "state": "unknown", "exit_code": None, "detail": "purged",
        "raw": None}


def test_a_finished_report_with_a_non_zero_exit_is_refused():
    with pytest.raises(ValueError, match="reported as one"):
        Report(JobState.FINISHED, exit_code=1)


def test_a_finished_report_with_no_exit_at_all_is_refused():
    """An exit nobody read is not a finished job, it is an unknown one."""
    with pytest.raises(ValueError, match="nobody read"):
        Report(JobState.FINISHED)


def test_a_failed_report_that_exited_zero_is_refused():
    """Zero and 'not known' are different claims and neither is failure."""
    with pytest.raises(ValueError, match="cannot have exited zero"):
        Report(JobState.FAILED, exit_code=0, detail="x")


@pytest.mark.parametrize("state", [JobState.QUEUED, JobState.RUNNING,
                                   JobState.NOT_SUBMITTED])
def test_a_job_that_has_not_finished_cannot_carry_an_exit_code(state):
    with pytest.raises(ValueError, match="ran to completion"):
        Report(state, exit_code=0)


def test_an_unknown_report_cannot_carry_an_exit_code():
    with pytest.raises(ValueError, match="ran to completion"):
        Report(JobState.UNKNOWN, exit_code=1, detail="x")


@pytest.mark.parametrize("state", [JobState.UNKNOWN, JobState.FAILED])
def test_an_unexplained_unknown_or_failure_is_refused(state):
    with pytest.raises(ValueError, match="needs a reason"):
        Report(state)


def test_a_report_state_must_be_one_of_the_six():
    with pytest.raises(TypeError, match="must be a JobState"):
        Report("finished")


def test_only_finished_and_failed_are_terminal():
    assert TERMINAL == {JobState.FINISHED, JobState.FAILED}
    assert not Report(JobState.UNKNOWN, detail="x").is_terminal
    assert not Report(JobState.QUEUED).is_terminal
    assert not Report(JobState.NOT_SUBMITTED).is_terminal
    assert Report(JobState.FAILED, exit_code=2, detail="x").is_terminal


def test_the_six_states_are_distinct_values():
    assert len({s.value for s in JobState}) == 6
    assert JobState.NOT_SUBMITTED.value == "not_submitted"


# ==========================================================================
# Handle: what a submitted job is known by
# ==========================================================================


def test_a_handle_round_trips_through_a_record(tmp_path):
    """A batch job outlives the process that submitted it, so it is written."""
    handle = Handle(backend="pbs", job_id="1.srv", name="run1",
                    workdir=tmp_path, stdout=tmp_path / "a.out",
                    stderr=tmp_path / "a.err", submitted_at=1.5)
    again = Handle.from_record(handle.to_record())
    assert again == handle


def test_a_handle_given_text_paths_holds_paths(tmp_path):
    """A caller reads the logs off the handle, so these must be openable."""
    handle = Handle(backend="pbs", job_id="1.srv", name="n",
                    workdir=str(tmp_path), stdout=str(tmp_path / "o"),
                    stderr=str(tmp_path / "e"), submitted_at=0.0)
    assert handle.workdir == tmp_path
    assert handle.stdout == tmp_path / "o"
    assert handle.stderr == tmp_path / "e"


def test_a_handle_record_missing_a_field_is_refused(tmp_path):
    record = Handle(backend="local", job_id="1", name="n", workdir=tmp_path,
                    stdout=tmp_path / "o", stderr=tmp_path / "e",
                    submitted_at=0.0).to_record()
    del record["job_id"]
    with pytest.raises(SchedulerError, match=r"missing \['job_id'\]"):
        Handle.from_record(record)


@pytest.mark.parametrize("job_id", ["", "   "])
def test_a_handle_with_no_identifier_is_refused(tmp_path, job_id):
    with pytest.raises(SchedulerError, match="nothing can ever be asked"):
        Handle(backend="pbs", job_id=job_id, name="n", workdir=tmp_path,
               stdout=tmp_path / "o", stderr=tmp_path / "e", submitted_at=0.0)


# ==========================================================================
# A campaign is a list of state points
# ==========================================================================


def _point(**kwargs):
    base = dict(geometry="cube", ensemble="npt_iso", T=1400.0, x=0.5,
                dt_ps=1e-3, equil_ps=10.0, prod_ps=50.0, dump_every_ps=0.1,
                dump_from="prod", seed=1, P=0.0)
    base.update(kwargs)
    return StatePoint(**base)


def test_a_campaign_becomes_one_job_per_state_point(tmp_path):
    """The list of state points IS the manifest; nothing is invented here."""
    campaign = [_point(T=1400.0), _point(T=1500.0)]
    jobs = jobs_for(campaign, lambda p: f"echo {p.tag}", tmp_path,
                    walltime_s=3600.0)
    assert [j.name for j in jobs] == [p.tag for p in campaign]
    assert jobs[0].workdir == tmp_path / campaign[0].tag
    assert jobs[0].walltime_s == 3600.0
    assert campaign[1].tag in jobs[1].script


def test_a_campaign_may_give_each_point_its_own_environment(tmp_path):
    jobs = jobs_for([_point()], lambda p: "echo", tmp_path,
                    env_for=lambda p: {"SEED": str(p.seed)})
    assert jobs[0].job_env == {"SEED": "1"}


def test_two_state_points_with_one_tag_are_refused(tmp_path):
    """Two jobs of one name share a job file and two logs."""
    twice = [_point(), _point()]
    with pytest.raises(SchedulerError, match="repeated job name"):
        jobs_for(twice, lambda p: "echo", tmp_path)


def test_the_clash_count_counts_repeated_names_and_nothing_else(tmp_path):
    """Two of one name beside a third job is one clash, and names it."""
    jobs = [_job(tmp_path, name="a"), _job(tmp_path, name="a"),
            _job(tmp_path, name="b")]
    with pytest.raises(SchedulerError,
                       match=r"1 repeated job name\(s\): \['a'\]"):
        _local(tmp_path).check(jobs)


def test_three_copies_of_one_name_report_one_clash(tmp_path):
    """The count is of distinct names, not of extra copies."""
    jobs = [_job(tmp_path, name="a"), _job(tmp_path, name="a"),
            _job(tmp_path, name="a")]
    with pytest.raises(SchedulerError, match=r"1 repeated job name"):
        _local(tmp_path).check(jobs)


# ==========================================================================
# The job file
# ==========================================================================


def _written(tmp_path, **kwargs):
    job = _job(tmp_path, **kwargs)
    back = _local(tmp_path)
    back.check([job])
    from aipf.md.scheduler import _write_job_file
    return job, _write_job_file(job, back.shell).read_text()


def test_the_job_file_refuses_an_unset_variable_and_a_swallowed_status(
        tmp_path):
    """Both lines are archived, and the second names the cost of its absence."""
    _, text = _written(tmp_path)
    assert "set -u" in text
    assert "set -o pipefail" in text


def test_a_body_that_pipes_its_work_reports_the_work_not_the_pipe(tmp_path):
    """The archived note's own measurement, as behaviour rather than text."""
    back = _local(tmp_path)
    handle = back.submit(_job(tmp_path, script="exit 4 | cat"))
    report = back.wait([handle], timeout_s=30.0)[handle.job_id]
    assert report.state is JobState.FAILED
    assert report.exit_code == 4


def test_the_job_file_does_not_turn_on_exit_on_error(tmp_path):
    """No archived file uses it; turning it on would change a given body."""
    _, text = _written(tmp_path)
    assert "set -e\n" not in text


def test_the_job_file_removes_what_the_caller_asked_to_remove(tmp_path):
    _, text = _written(tmp_path, env={"A": None, "B": None})
    assert "unset A" in text and "unset B" in text


def test_the_job_file_moves_into_the_workdir(tmp_path):
    """A batch job starts in the submitting user's home directory."""
    job, text = _written(tmp_path)
    assert f"cd {job.workdir}" in text


def test_the_job_file_keeps_the_body_verbatim(tmp_path):
    body = "python run.py --x 0.5   # two  spaces\nexit $?"
    _, text = _written(tmp_path, script=body)
    assert text.endswith(body + "\n")


def test_the_job_file_starts_with_an_interpreter_that_exists(tmp_path):
    _, text = _written(tmp_path)
    first = text.splitlines()[0]
    assert first.startswith("#!")
    assert pathlib.Path(first[2:]).is_file()


def test_the_job_file_is_executable(tmp_path):
    job, _ = _written(tmp_path)
    assert job.script_path.stat().st_mode & stat.S_IXUSR


def test_an_interpreter_that_is_not_installed_is_refused(tmp_path):
    with pytest.raises(SchedulerError, match="no interpreter"):
        LocalBackend(lanes=1, shell="no-such-shell-anywhere")


def test_a_named_interpreter_is_used(tmp_path):
    """A caller may write job files for an interpreter of its own."""
    from aipf.md.scheduler import _write_job_file
    back = LocalBackend(lanes=1, shell=sys.executable)
    job = _job(tmp_path, script="pass")
    back.check([job])
    assert _write_job_file(job, back.shell).read_text().splitlines()[0] == (
        f"#!{sys.executable}")


def test_a_campaign_whose_directory_cannot_be_written_is_refused(tmp_path):
    """Refused for the whole campaign, before any of it goes."""
    closed = tmp_path / "closed"
    closed.mkdir()
    closed.chmod(0o500)
    try:
        with pytest.raises(SchedulerError, match="not writable|cannot make"):
            _local(tmp_path).check([_job(closed, name="x")])
    finally:
        closed.chmod(0o700)


def test_a_campaign_whose_logs_have_nowhere_to_go_is_refused(tmp_path):
    """The logs need not sit in the workdir, so both places are checked."""
    closed = tmp_path / "logs"
    closed.mkdir()
    closed.chmod(0o500)
    job = _job(tmp_path, stdout=closed / "x.out")
    try:
        with pytest.raises(SchedulerError, match="not writable"):
            _local(tmp_path).check([job])
    finally:
        closed.chmod(0o700)


def test_a_job_file_that_cannot_be_written_is_refused_by_name(tmp_path):
    """The path that would not take it is in the message."""
    from aipf.md.scheduler import _write_job_file
    job = _job(tmp_path)
    job.workdir.mkdir(parents=True)
    job.script_path.mkdir()
    with pytest.raises(SchedulerError, match="cannot write its job file"):
        _write_job_file(job, _local(tmp_path).shell)


# ==========================================================================
# The local backend, running real processes
# ==========================================================================


def test_lanes_must_be_a_positive_count(tmp_path):
    """No default: the archived launcher defaults it to four."""
    for bad in (0, -1, 1.5, True, "4"):
        with pytest.raises(SchedulerError, match="positive integer"):
            LocalBackend(lanes=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_poll_interval_is_refused(bad):
    with pytest.raises(SchedulerError, match="poll_interval_s"):
        LocalBackend(lanes=1, poll_interval_s=bad)


def test_a_negative_kill_grace_is_refused():
    with pytest.raises(SchedulerError, match="kill_grace_s"):
        LocalBackend(lanes=1, kill_grace_s=-1.0)


def test_no_grace_at_all_is_a_legal_choice():
    """A campaign that would rather not wait for a polite signal may say so."""
    assert LocalBackend(lanes=1, kill_grace_s=0.0).kill_grace_s == 0.0


def test_a_short_job_runs_end_to_end(tmp_path):
    """The whole point of this backend: it actually runs the work."""
    back = _local(tmp_path)
    job = _job(tmp_path, script="echo ran-here")
    handle = back.submit(job)
    reports = back.wait([handle], timeout_s=30.0)
    report = reports[handle.job_id]
    assert report.state is JobState.FINISHED
    assert report.exit_code == 0
    assert job.stdout.read_text().strip() == "ran-here"


def test_submitting_starts_the_work_without_anyone_asking_again(tmp_path):
    """Submit and walk away: nothing else in a campaign has to drive it."""
    back = _local(tmp_path)
    marker = tmp_path / "started"
    back.submit_all([_job(tmp_path, script=f"touch {marker}")])
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.exists()


def test_a_job_that_exits_non_zero_is_a_failure_with_its_code(tmp_path):
    back = _local(tmp_path)
    handle = back.submit(_job(tmp_path, script="exit 3"))
    report = back.wait([handle], timeout_s=30.0)[handle.job_id]
    assert report.state is JobState.FAILED
    assert report.exit_code == 3
    assert "exited 3" in report.detail


def test_the_body_runs_in_its_own_workdir(tmp_path):
    back = _local(tmp_path)
    job = _job(tmp_path, script="pwd")
    handle = back.submit(job)
    back.wait([handle], timeout_s=30.0)
    assert job.stdout.read_text().strip() == str(job.workdir)


def test_the_job_sees_the_variables_it_was_given(tmp_path):
    back = _local(tmp_path)
    job = _job(tmp_path, script="echo $SHARD $PREFIX", env={"SHARD": "0",
                                                            "PREFIX": "p5"})
    handle = back.submit(job)
    back.wait([handle], timeout_s=30.0)
    assert job.stdout.read_text().split() == ["0", "p5"]


def test_the_job_sees_its_own_wall_clock_budget(tmp_path):
    """So that work with a deadline reads it rather than being told twice."""
    back = _local(tmp_path)
    job = _job(tmp_path, script=f"echo ${WALLTIME_ENV}", walltime_s=120.0)
    handle = back.submit(job)
    back.wait([handle], timeout_s=30.0)
    assert float(job.stdout.read_text().strip()) == 120.0


def test_the_job_inherits_the_environment_it_was_launched_from(tmp_path,
                                                               monkeypatch):
    """A real run needs the module and library settings of the session that
    launched it, and a body using only builtins would not notice them go: an
    interpreter supplies its own search path when there is none."""
    monkeypatch.setenv("AN_INHERITED_SETTING", "carried")
    back = _local(tmp_path)
    job = _job(tmp_path, script='echo "[${AN_INHERITED_SETTING:-lost}]"')
    handle = back.submit(job)
    back.wait([handle], timeout_s=30.0)
    assert job.stdout.read_text().strip() == "[carried]"


def test_a_variable_the_caller_removed_does_not_reach_the_job(tmp_path,
                                                              monkeypatch):
    """The one constraint the run interface will need, and it is testable."""
    monkeypatch.setenv("A_BAD_SETTING", "on")
    back = _local(tmp_path)
    job = _job(tmp_path, script='echo "[${A_BAD_SETTING:-gone}]"',
               env={"A_BAD_SETTING": None})
    handle = back.submit(job)
    back.wait([handle], timeout_s=30.0)
    assert job.stdout.read_text().strip() == "[gone]"


def test_naming_one_file_for_both_streams_puts_both_in_it(tmp_path):
    one = tmp_path / "joined.log"
    back = _local(tmp_path)
    job = _job(tmp_path, script="echo out; echo err 1>&2", stdout=one,
               stderr=one)
    handle = back.submit(job)
    back.wait([handle], timeout_s=30.0)
    assert set(one.read_text().split()) == {"out", "err"}


def test_separate_streams_stay_separate(tmp_path):
    back = _local(tmp_path)
    job = _job(tmp_path, script="echo out; echo err 1>&2")
    handle = back.submit(job)
    back.wait([handle], timeout_s=30.0)
    assert job.stdout.read_text().strip() == "out"
    assert job.stderr.read_text().strip() == "err"


def test_only_as_many_jobs_run_at_once_as_there_are_lanes(tmp_path):
    """Two lanes, four jobs: the node is never oversubscribed."""
    back = _local(tmp_path, lanes=2)
    counter = tmp_path / "live"
    counter.write_text("")
    script = (f'echo start >> {counter}\n'
              f'sleep 0.4\n'
              f'echo stop >> {counter}\n')
    jobs = [_job(tmp_path, name=f"j{i}", script=script) for i in range(4)]
    handles = back.submit_all(jobs)
    peak = 0
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        live = sum(1 for h in handles
                   if back.state(h).state is JobState.RUNNING)
        peak = max(peak, live)
        if all(back.state(h).is_terminal for h in handles):
            break
        time.sleep(0.05)
    assert peak <= 2
    assert all(back.state(h).state is JobState.FINISHED for h in handles)


def test_a_job_waiting_for_a_lane_is_queued_and_not_running(tmp_path):
    back = _local(tmp_path, lanes=1)
    first = _job(tmp_path, name="first", script="sleep 0.6")
    second = _job(tmp_path, name="second", script="echo second")
    handles = back.submit_all([first, second])
    report = back.state(handles[1])
    assert report.state is JobState.QUEUED
    assert "lane" in report.detail
    back.wait(handles, timeout_s=30.0)


def test_a_job_that_overruns_its_limit_is_killed_with_its_children(tmp_path):
    """Killing the body alone leaves the work it launched holding the machine."""
    back = _local(tmp_path, lanes=1, kill_grace_s=0.5)
    pidfile = tmp_path / "child.pid"
    script = (f"sleep 120 &\n"
              f"echo $! > {pidfile}\n"
              f"wait\n")
    job = _job(tmp_path, script=script, walltime_s=0.5)
    handle = back.submit(job)
    report = back.wait([handle], timeout_s=60.0)[handle.job_id]
    assert report.state is JobState.FAILED
    assert "wall-clock limit" in report.detail
    child = int(pidfile.read_text().strip())
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except OSError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"the job's child {child} outlived the kill")


def test_a_job_whose_log_cannot_be_opened_is_a_failure_and_not_a_silence(
        tmp_path):
    """The archived launcher prints this case and the run vanishes from the
    report; here it is a failure with a reason."""
    back = _local(tmp_path)
    job = _job(tmp_path)
    job.workdir.mkdir(parents=True)
    job.stdout.mkdir()
    handle = back.submit(job)
    report = back.state(handle)
    assert report.state is JobState.FAILED
    assert "could not be opened" in report.detail


def test_a_job_that_cannot_be_started_at_all_is_a_failure_with_the_reason(
        tmp_path):
    """Not a line of log and then silence, which is what the archive does."""
    import shutil
    interpreter = tmp_path / "shell-copy"
    shutil.copy(_local(tmp_path).shell, interpreter)
    back = LocalBackend(lanes=1, shell=str(interpreter))
    handles = back.submit_all([_job(tmp_path, name="a", script="sleep 0.3"),
                               _job(tmp_path, name="b", script="echo b")])
    interpreter.unlink()
    reports = back.wait(handles, timeout_s=30.0)
    assert reports[handles[1].job_id].state is JobState.FAILED
    assert "could not be started" in reports[handles[1].job_id].detail


def test_a_job_whose_second_log_cannot_be_opened_is_a_failure(tmp_path):
    """Both streams are checked, not just the first."""
    back = _local(tmp_path)
    job = _job(tmp_path)
    job.workdir.mkdir(parents=True)
    job.stderr.mkdir()
    handle = back.submit(job)
    report = back.state(handle)
    assert report.state is JobState.FAILED
    assert str(job.stderr) in report.detail


def test_cancelling_a_waiting_job_says_it_never_started(tmp_path):
    back = _local(tmp_path, lanes=1)
    handles = back.submit_all([_job(tmp_path, name="a", script="sleep 0.6"),
                               _job(tmp_path, name="b", script="echo b")])
    back.cancel(handles[1])
    assert back.state(handles[1]).detail == "cancelled before it started"
    assert back.state(handles[1]).state is JobState.FAILED
    back.wait([handles[0]], timeout_s=30.0)


def test_a_cancelled_waiting_job_never_runs_at_all(tmp_path):
    """Reporting it cancelled while letting it start would be the worst of
    both: a job nobody expects, writing into the campaign's directories."""
    back = _local(tmp_path, lanes=1)
    marker = tmp_path / "should-not-exist"
    handles = back.submit_all(
        [_job(tmp_path, name="a", script="sleep 0.4"),
         _job(tmp_path, name="b", script=f"touch {marker}")])
    back.cancel(handles[1])
    back.wait(handles, timeout_s=30.0)
    assert _never(marker.exists), "the cancelled job ran anyway"


def test_a_job_killed_by_a_signal_says_which_one(tmp_path):
    back = _local(tmp_path)
    handle = back.submit(_job(tmp_path, script="kill -9 $$"))
    report = back.wait([handle], timeout_s=30.0)[handle.job_id]
    assert report.state is JobState.FAILED
    assert report.detail == "killed by signal 9"


def test_a_job_that_ignores_the_polite_signal_is_still_stopped(tmp_path):
    """Which is why there are two signals. The report proves nothing here: a
    kill that killed nothing reports exactly the same."""
    back = _local(tmp_path, lanes=1, kill_grace_s=0.3)
    pidfile = tmp_path / "stubborn.pid"
    job = _job(tmp_path,
               script=f"trap '' TERM\nsleep 60 &\necho $! > {pidfile}\n"
                      f"wait\n",
               walltime_s=0.4)
    handle = back.submit(job)
    report = back.wait([handle], timeout_s=60.0)[handle.job_id]
    assert report.state is JobState.FAILED
    assert "wall-clock limit" in report.detail
    child = int(pidfile.read_text().strip())
    assert _until(lambda: not _alive(child)), f"{child} ignored both signals"


def test_a_job_is_asked_to_stop_before_it_is_made_to(tmp_path):
    """The polite signal comes first and is given time, so a job that tidies
    up on it gets the chance. Only the second signal cannot be caught."""
    back = _local(tmp_path, lanes=1, kill_grace_s=5.0)
    caught = tmp_path / "caught"
    job = _job(tmp_path,
               script=f"trap 'echo yes > {caught}; exit 7' TERM\n"
                      f"sleep 60 &\nwait\n",
               walltime_s=0.4)
    handle = back.submit(job)
    report = back.wait([handle], timeout_s=60.0)[handle.job_id]
    assert report.state is JobState.FAILED
    assert caught.is_file(), "the job was never asked, only killed"


def test_cancelling_a_running_job_stops_it(tmp_path):
    back = _local(tmp_path, lanes=1, kill_grace_s=0.5)
    pidfile = tmp_path / "child.pid"
    job = _job(tmp_path,
               script=f"sleep 120 &\necho $! > {pidfile}\nwait\n")
    handle = back.submit(job)
    assert _until(lambda: pidfile.is_file() and pidfile.read_text().strip())
    assert back.state(handle).state is JobState.RUNNING
    back.cancel(handle)
    report = back.state(handle)
    assert report.state is JobState.FAILED
    assert "cancelled while running" in report.detail
    # A report saying FAILED is also what a cancel that signalled nothing
    # would produce, so the process is what is checked.
    child = int(pidfile.read_text().strip())
    assert _until(lambda: not _alive(child)), f"child {child} survived"


def test_cancelling_a_finished_job_changes_nothing(tmp_path):
    back = _local(tmp_path)
    handle = back.submit(_job(tmp_path, script="echo x"))
    back.wait([handle], timeout_s=30.0)
    back.cancel(handle)
    assert back.state(handle).state is JobState.FINISHED


def test_cancelling_a_handle_this_pool_never_issued_does_nothing(tmp_path):
    back = _local(tmp_path)
    back.cancel(Handle(backend="local", job_id="elsewhere", name="n",
                       workdir=tmp_path, stdout=tmp_path / "o",
                       stderr=tmp_path / "e", submitted_at=0.0))


def test_a_handle_from_another_process_is_unknown_and_says_why(tmp_path):
    """A local job is known only to the process that ran it."""
    back = _local(tmp_path)
    report = back.state(Handle(backend="local", job_id="local-1-1", name="n",
                               workdir=tmp_path, stdout=tmp_path / "o",
                               stderr=tmp_path / "e", submitted_at=0.0))
    assert report.state is JobState.UNKNOWN
    assert "did not start" in report.detail


def test_nothing_starts_when_the_campaign_is_refused(tmp_path):
    """The refusal is whole-campaign: no job file, no process."""
    back = _local(tmp_path, lanes=4)
    jobs = [_job(tmp_path, name="same", script="echo 1"),
            _job(tmp_path, name="same", script="echo 2")]
    with pytest.raises(SchedulerError, match="repeated job name"):
        back.submit_all(jobs)
    assert not list((tmp_path / "same").glob("*.sh"))


def test_a_campaign_root_that_does_not_exist_yet_is_made(tmp_path):
    """A campaign names where it wants to live; it does not require it."""
    back = _local(tmp_path)
    jobs = jobs_for([_point()], lambda p: "echo x", tmp_path / "new" / "root")
    handles = back.submit_all(jobs)
    assert back.wait(handles, timeout_s=30.0)[
        handles[0].job_id].state is JobState.FINISHED


def test_waiting_returns_only_when_every_handle_is_terminal(tmp_path):
    """One job finishing is not the campaign finishing."""
    back = _local(tmp_path, lanes=1)
    handles = back.submit_all([_job(tmp_path, name="a", script="true"),
                               _job(tmp_path, name="b", script="sleep 0.4")])
    reports = back.wait(handles, timeout_s=30.0)
    assert [reports[h.job_id].state for h in handles] == [
        JobState.FINISHED, JobState.FINISHED]


def test_waiting_gives_up_and_reports_what_it_knows(tmp_path):
    """A caller out of patience still gets states, not an exception."""
    back = _local(tmp_path, lanes=1)
    handle = back.submit(_job(tmp_path, script="sleep 30"))
    reports = back.wait([handle], timeout_s=0.3)
    assert reports[handle.job_id].state is JobState.RUNNING
    back.cancel(handle)


def test_a_long_campaign_does_not_leak_a_handle_per_job(tmp_path):
    """Each job opens two log files, and a campaign is hundreds of jobs."""
    fds = pathlib.Path("/proc/self/fd")
    if not fds.is_dir():
        pytest.skip("no open-descriptor listing on this machine")
    back = _local(tmp_path, lanes=4)
    before = len(list(fds.iterdir()))
    handles = back.submit_all(
        [_job(tmp_path, name=f"leak{i}", script="true") for i in range(40)])
    back.wait(handles, timeout_s=60.0)
    assert len(list(fds.iterdir())) - before < 10


def test_every_handle_of_a_campaign_is_distinct(tmp_path):
    back = _local(tmp_path, lanes=4)
    handles = back.submit_all(
        [_job(tmp_path, name=f"j{i}", script="true") for i in range(5)])
    assert len({h.job_id for h in handles}) == 5
    back.wait(handles, timeout_s=30.0)


# ==========================================================================
# The batch backend, against fake farm commands
# ==========================================================================


def test_a_backend_with_no_queue_is_refused():
    for bad in ("", "  ", None):
        with pytest.raises(SchedulerError, match="queue must be named"):
            PbsBackend(queue=bad, project="1")


def test_a_backend_with_no_project_is_refused():
    for bad in ("", "   ", None):
        with pytest.raises(SchedulerError, match="project code"):
            PbsBackend(queue="ai", project=bad)


def test_a_walltime_in_the_backend_resources_is_refused():
    """One limit, in one place: the archive states it twice and warns."""
    with pytest.raises(SchedulerError, match="disagree with itself"):
        PbsBackend(queue="ai", project="1",
                   resources={"walltime": "24:00:00"})


def test_negative_query_retries_are_refused():
    with pytest.raises(SchedulerError, match="query_retries"):
        PbsBackend(queue="ai", project="1", query_retries=-1)


def test_a_batch_job_with_no_wall_clock_limit_is_refused(tmp_path):
    """A queue applies a site default nobody here can name."""
    back, logs = _pbs(tmp_path)
    with pytest.raises(SchedulerError, match="site default"):
        back.check([_job(tmp_path)])
    assert _calls(logs["qsub"]) == []


def test_a_comma_in_a_passed_value_is_refused(tmp_path):
    """Measured: the submitter separates variables by comma and refuses."""
    back, logs = _pbs(tmp_path)
    job = _job(tmp_path, env={"RERUN": "0.90:1200,0.95:1700"},
               walltime_s=3600.0)
    with pytest.raises(SchedulerError, match="holds a comma"):
        back.check([job])
    assert _calls(logs["qsub"]) == []


def test_two_batch_jobs_of_one_name_are_refused_before_the_queue_is_asked(
        tmp_path):
    back, logs = _pbs(tmp_path)
    jobs = [_job(tmp_path, name="same", walltime_s=60.0),
            _job(tmp_path, name="same", walltime_s=60.0)]
    with pytest.raises(SchedulerError, match="repeated job name"):
        back.check(jobs)
    assert _calls(logs["qstat"]) == []


def test_a_queue_that_does_not_exist_refuses_the_campaign(tmp_path):
    """Measured message and exit status of the real command."""
    back, logs = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "err": "qstat: Unknown queue normal\n", "rc": 170}])
    with pytest.raises(SchedulerError, match="Unknown queue"):
        back.check([_job(tmp_path, walltime_s=3600.0)])
    assert _calls(logs["qsub"]) == []


def test_a_queue_that_exists_but_is_closed_refuses_the_campaign(tmp_path):
    """A queue that takes a campaign and never runs it is no better."""
    shut = ("Queue: gpu4\n    queue_type = Execution\n"
            "    enabled = False\n    started = False\n")
    back, logs = _pbs(tmp_path, qstat_rules=[{"match": "-Qf", "out": shut}])
    with pytest.raises(SchedulerError, match="not enabled"):
        back.check([_job(tmp_path, walltime_s=3600.0)])
    assert _calls(logs["qsub"]) == []


def test_a_queue_that_is_enabled_but_not_started_refuses_the_campaign(
        tmp_path):
    """Two separate flags on the real record, and either one stops the work."""
    halted = ("Queue: ai\n    enabled = True\n    started = False\n")
    back, logs = _pbs(tmp_path, qstat_rules=[{"match": "-Qf", "out": halted}])
    with pytest.raises(SchedulerError, match="not started"):
        back.check([_job(tmp_path, walltime_s=3600.0)])
    assert _calls(logs["qsub"]) == []


def test_the_queue_is_asked_about_once_and_not_once_per_job(tmp_path):
    back, logs = _pbs(tmp_path)
    jobs = [_job(tmp_path, name=f"j{i}", walltime_s=60.0) for i in range(4)]
    back.check(jobs)
    back.check(jobs)
    assert [c for c in _calls(logs["qstat"]) if "-Qf" in c] == ["-Qf ai"]


def test_the_directives_carry_the_queue_project_chunk_and_limit(tmp_path):
    """Every one of these is on the twenty archived job files."""
    back, _ = _pbs(tmp_path, resources={"select": "1:ngpus=4:mem=128GB"})
    job = _job(tmp_path, name="anchors_200GPa", walltime_s=24 * 3600.0)
    back.submit(job)
    text = job.script_path.read_text()
    assert "#PBS -N anchors_200GPa" in text
    assert "#PBS -q ai" in text
    assert "#PBS -P 12345678" in text
    assert "#PBS -l select=1:ngpus=4:mem=128GB" in text
    assert "#PBS -l walltime=24:00:00" in text
    assert f"#PBS -o {job.stdout}" in text
    assert f"#PBS -e {job.stderr}" in text
    assert "#PBS -j oe" not in text


def test_a_queue_given_with_whitespace_is_normalised(tmp_path):
    """A name read out of a configuration file arrives with its newline."""
    back, _ = _pbs(tmp_path, queue="  ai\n")
    assert back.queue == "ai"
    job = _job(tmp_path, walltime_s=60.0)
    back.submit(job)
    assert "#PBS -q ai\n" in job.script_path.read_text()


def test_the_directives_follow_the_interpreter_line(tmp_path):
    """A directive above the interpreter line is not a job file."""
    back, _ = _pbs(tmp_path)
    job = _job(tmp_path, walltime_s=60.0)
    back.submit(job)
    lines = job.script_path.read_text().splitlines()
    assert lines[0].startswith("#!")
    assert lines[1].startswith("#PBS ")


def test_one_file_for_both_streams_becomes_a_join_directive(tmp_path):
    """Seven of the twenty join their streams; eleven name two files."""
    one = tmp_path / "both.log"
    back, _ = _pbs(tmp_path)
    job = _job(tmp_path, walltime_s=60.0, stdout=one, stderr=one)
    back.submit(job)
    text = job.script_path.read_text()
    assert "#PBS -j oe" in text
    assert "#PBS -e " not in text


def test_no_directive_holds_a_variable(tmp_path):
    """Measured: a directive is not expanded. One archived file writes a
    shell variable into an output directive and no file of that name exists
    in the directory it was supposed to write to."""
    back, _ = _pbs(tmp_path, resources={"select": "1:ngpus=4"})
    job = _job(tmp_path, name="shard0", walltime_s=60.0,
               env={"SHARD": "0"})
    back.submit(job)
    for line in job.script_path.read_text().splitlines():
        if line.startswith("#PBS"):
            assert "$" not in line


def test_the_variables_are_passed_at_submission_and_not_in_a_directive(
        tmp_path):
    """Which is what every archived submitter does."""
    back, logs = _pbs(tmp_path)
    job = _job(tmp_path, walltime_s=60.0, env={"SHARD": "0", "PREFIX": "p5"})
    back.submit(job)
    call = _calls(logs["qsub"])[0]
    assert f"-v SHARD=0,PREFIX=p5,{WALLTIME_ENV}=60.0 " in call
    assert "#PBS -v" not in job.script_path.read_text()


def test_every_batch_job_passes_at_least_its_own_budget(tmp_path):
    """So the flag is never written with nothing after it: a batch job always
    carries a wall-clock limit, and the limit is one of its variables."""
    back, logs = _pbs(tmp_path)
    back.submit(_job(tmp_path, walltime_s=60.0))
    call = _calls(logs["qsub"])[0]
    assert f"-v {WALLTIME_ENV}=60.0" in call


def test_the_identifier_is_what_the_submitter_printed(tmp_path):
    back, _ = _pbs(tmp_path, qsub_rules=[{"out": "24048079.srv1\n"}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    assert handle.job_id == "24048079.srv1"
    assert handle.backend == "pbs"


def test_a_refused_submission_raises_with_what_the_farm_said(tmp_path):
    back, _ = _pbs(tmp_path, qsub_rules=[
        {"err": "qsub: Unknown queue\n", "rc": 1}])
    with pytest.raises(SchedulerError,
                       match="refused by the submitter.*Unknown queue"):
        back.submit(_job(tmp_path, walltime_s=60.0))


def test_an_accepted_job_with_no_identifier_is_a_loud_refusal(tmp_path):
    """It may be running, and a second submission would write over it."""
    back, _ = _pbs(tmp_path, qsub_rules=[{"out": "   \n", "rc": 0}])
    with pytest.raises(SchedulerError, match="write over its output"):
        back.submit(_job(tmp_path, walltime_s=60.0))


def test_a_campaign_that_fails_half_way_keeps_the_handles_it_has(tmp_path):
    """A running job nobody holds a handle to is the worst outcome here."""
    back, _ = _pbs(tmp_path, qsub_rules=[
        {"nth": 1, "out": "1.srv\n"},
        {"nth": 2, "err": "qsub: over the limit\n", "rc": 1}])
    jobs = [_job(tmp_path, name=f"j{i}", walltime_s=60.0) for i in range(3)]
    with pytest.raises(PartialSubmission) as caught:
        back.submit_all(jobs)
    assert [h.job_id for h in caught.value.handles] == ["1.srv"]
    assert "must be polled or cancelled" in str(caught.value)


def test_the_first_job_failing_is_an_ordinary_refusal(tmp_path):
    """Nothing went, so nothing is carried."""
    back, _ = _pbs(tmp_path, qsub_rules=[{"err": "no\n", "rc": 1}])
    with pytest.raises(SchedulerError) as caught:
        back.submit_all([_job(tmp_path, walltime_s=60.0)])
    assert not isinstance(caught.value, PartialSubmission)


def test_every_job_file_is_written_before_any_job_is_submitted(tmp_path):
    back, logs = _pbs(tmp_path)
    jobs = [_job(tmp_path, name=f"j{i}", walltime_s=60.0) for i in range(3)]
    back.submit_all(jobs)
    assert all(j.script_path.is_file() for j in jobs)
    assert len(_calls(logs["qsub"])) == 3


@pytest.mark.parametrize("letter,expected", sorted(
    {"Q": JobState.QUEUED, "H": JobState.QUEUED, "W": JobState.QUEUED,
     "T": JobState.QUEUED, "R": JobState.RUNNING, "S": JobState.RUNNING,
     "B": JobState.RUNNING, "E": JobState.RUNNING}.items()))
def test_each_state_letter_the_server_uses_is_read(tmp_path, letter, expected):
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": _record(letter)}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is expected
    assert letter in report.detail
    assert report.exit_code is None
    assert report.raw


def test_a_finished_job_that_exited_zero_is_finished(tmp_path):
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": _record("F", exit_status=0)}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.FINISHED
    assert report.exit_code == 0


@pytest.mark.parametrize("code,phrase", [(1, "exited 1"),
                                         (143, "signal 15"),
                                         (271, "signal 143"),
                                         (-3, "job-level errors")])
def test_a_finished_job_that_exited_otherwise_is_a_failure(tmp_path, code,
                                                           phrase):
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": _record("F", exit_status=code)}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.FAILED
    assert report.exit_code == code
    assert phrase in report.detail


def test_a_job_that_has_reported_an_exit_status_has_ended(tmp_path):
    """Measured on a real submission, and it is why the letter alone is not
    read: a job whose two log files could not be copied back to the host that
    submitted it ran, exited zero, and then sat in "exiting" for as long as it
    was watched, with Exit_status = 0 beside it the whole time. Reading the
    letter would report that finished job as running for ever."""
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": _record("E", exit_status=0)}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.FINISHED
    assert report.exit_code == 0


def test_an_exiting_job_that_has_reported_a_failure_has_failed(tmp_path):
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": _record("E", exit_status=271)}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.FAILED
    assert report.exit_code == 271


def test_an_exiting_job_with_no_exit_status_is_still_running(tmp_path):
    """The letter is read when there is nothing better to read."""
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": _record("E")}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    assert back.state(handle).state is JobState.RUNNING


def test_a_finished_job_with_no_exit_status_is_unknown(tmp_path):
    """It has not said which way it went, and neither will this."""
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": _record("F")}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.UNKNOWN
    assert "no exit status" in report.detail


def test_an_exit_status_that_is_not_a_number_is_unknown(tmp_path):
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": _record("F", exit_status="oops")}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    assert back.state(handle).state is JobState.UNKNOWN


def test_a_state_letter_this_module_does_not_know_is_unknown(tmp_path):
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": _record("Z")}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.UNKNOWN
    assert "does not know how to read" in report.detail


def test_a_job_the_server_has_forgotten_is_unknown_and_not_failed(tmp_path):
    """Measured: finished jobs are kept twelve hours here and then purged."""
    back, logs = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "err": "qstat: Unknown Job Id 1.srv\n", "rc": 153}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.UNKNOWN
    assert "purged" in report.detail
    assert report.exit_code is None
    assert len([c for c in _calls(logs["qstat"]) if "-Qf" not in c]) == 1


def test_a_server_that_cannot_be_reached_is_unknown_after_the_retries(
        tmp_path):
    """The query is a read and is retried; the submission never is."""
    back, logs = _pbs(tmp_path, query_retries=2, retry_pause_s=0.01,
                      qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "err": "qstat: cannot connect to server srv "
                               "(errno=15008)\n", "rc": 1}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.UNKNOWN
    assert "3 attempt" in report.detail
    assert "cannot connect" in report.detail
    assert len([c for c in _calls(logs["qstat"]) if "-Qf" not in c]) == 3


def test_asking_once_is_a_legal_choice(tmp_path):
    back, logs = _pbs(tmp_path, query_retries=0, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "err": "qstat: cannot connect\n", "rc": 1}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.UNKNOWN
    assert "1 attempt" in report.detail
    assert len([c for c in _calls(logs["qstat"]) if "-Qf" not in c]) == 1


def test_a_negative_retry_pause_is_refused():
    with pytest.raises(SchedulerError, match="retry_pause_s"):
        PbsBackend(queue="ai", project="1", retry_pause_s=-1.0)


def test_an_answer_that_holds_no_record_at_all_is_unknown(tmp_path):
    """Rather than a record with no state, which would read as a bad letter."""
    back, _ = _pbs(tmp_path, retry_pause_s=0.0, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": "", "rc": 0}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.UNKNOWN
    assert "could not be asked" in report.detail


def test_a_query_is_not_retried_when_the_job_is_simply_gone(tmp_path):
    back, logs = _pbs(tmp_path, query_retries=5, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "err": "qstat: Unknown Job Id 1.srv\n", "rc": 153}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    back.state(handle)
    assert len([c for c in _calls(logs["qstat"]) if "-Qf" not in c]) == 1


def test_a_moved_job_is_followed_to_the_server_that_runs_it(tmp_path):
    """Measured on a live job: the accepting server reports 'moved' forever,
    and the project convention submits to exactly such a routing queue."""
    back, logs = _pbs(tmp_path, qsub_rules=[{"out": "24048079.srv1\n"}],
                      qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "@srv2", "out": _record("R", queue="aiq1",
                                            job_id="24048079.srv1")},
        {"match": "-f", "out": _record("M", queue="ai@srv2",
                                       job_id="24048079.srv1")}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.RUNNING
    assert "moved to ai@srv2" in report.detail
    assert any("24048079.srv1@srv2" in c for c in _calls(logs["qstat"]))


def test_a_moved_job_that_has_finished_there_is_finished(tmp_path):
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "@srv2", "out": _record("F", exit_status=0,
                                            queue="aiq1")},
        {"match": "-f", "out": _record("M", queue="ai@srv2")}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.FINISHED
    assert report.exit_code == 0


def test_a_moved_job_whose_record_names_no_server_stays_unknown(tmp_path):
    back, logs = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "-f", "out": _record("M", queue="somewhere")}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.UNKNOWN
    assert "moved to somewhere" in report.detail
    assert not any("@" in c for c in _calls(logs["qstat"]) if "-Qf" not in c)


def test_a_moved_job_that_moved_again_stays_unknown(tmp_path):
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "@srv2", "out": _record("M", queue="ai@srv3")},
        {"match": "-f", "out": _record("M", queue="ai@srv2")}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.UNKNOWN
    assert "that server reports" in report.detail


def test_a_moved_job_the_other_server_has_forgotten_is_unknown(tmp_path):
    back, _ = _pbs(tmp_path, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"match": "@srv2", "err": "qstat: Unknown Job Id 1.srv\n",
         "rc": 153},
        {"match": "-f", "out": _record("M", queue="ai@srv2")}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    report = back.state(handle)
    assert report.state is JobState.UNKNOWN
    assert "purged" in report.detail


def test_cancelling_asks_the_server_to_delete_the_job(tmp_path):
    back, logs = _pbs(tmp_path)
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    back.cancel(handle)
    assert _calls(logs["qdel"]) == ["1.srv"]


def test_cancelling_a_job_the_server_has_forgotten_is_not_an_error(tmp_path):
    back, _ = _pbs(tmp_path, qdel_rules=[
        {"err": "qdel: Unknown Job Id 1.srv\n", "rc": 153}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    back.cancel(handle)


@pytest.mark.parametrize("said,code", [
    ("qdel: Request invalid for state of job 1.srv\n", 168),
    ("qdel: Job has finished 1.srv\n", 35)])
def test_cancelling_a_job_past_deleting_is_not_an_error(tmp_path, said, code):
    """Both measured on one real submission: while the job was exiting the
    delete command refused it, and once it had finished it refused it
    differently. Neither is a failed cancellation."""
    back, _ = _pbs(tmp_path, qdel_rules=[{"err": said, "rc": code}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    back.cancel(handle)


def test_a_cancellation_the_server_refuses_is_raised(tmp_path):
    back, _ = _pbs(tmp_path, qdel_rules=[
        {"err": "qdel: Unauthorized Request\n", "rc": 159}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    with pytest.raises(SchedulerError, match="cannot cancel"):
        back.cancel(handle)


def test_a_missing_farm_command_is_a_refusal_naming_it(tmp_path):
    back = PbsBackend(queue="ai", project="1", qstat="no-such-qstat-anywhere")
    with pytest.raises(SchedulerError, match="no 'no-such-qstat-anywhere'"):
        back.check([_job(tmp_path, walltime_s=60.0)])


def test_a_farm_command_that_never_answers_is_a_refusal(tmp_path):
    back, _ = _pbs(tmp_path, command_timeout_s=0.5,
                   qstat_rules=[{"match": "-Qf",
                                 "out": "import time\n"}])
    slow = tmp_path / "qstat"
    slow.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(5)\n")
    slow.chmod(0o755)
    with pytest.raises(SchedulerError, match="did not answer"):
        back.check([_job(tmp_path, walltime_s=60.0)])


def test_waiting_on_the_batch_backend_polls_until_it_is_terminal(tmp_path):
    back, logs = _pbs(tmp_path, poll_interval_s=0.01, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"nth": 2, "out": _record("Q")},
        {"nth": 3, "out": _record("R")},
        {"match": "-f", "out": _record("F", exit_status=0)}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    reports = back.wait([handle], timeout_s=30.0)
    assert reports[handle.job_id].state is JobState.FINISHED
    assert len([c for c in _calls(logs["qstat"]) if "-Qf" not in c]) == 3


def test_the_batch_backend_pauses_between_questions(tmp_path):
    """A server polled in a tight loop is a server that answers slowly."""
    back, _ = _pbs(tmp_path, poll_interval_s=0.3, qstat_rules=[
        {"match": "-Qf", "out": _QUEUE_OK},
        {"nth": 2, "out": _record("Q")},
        {"nth": 3, "out": _record("R")},
        {"match": "-f", "out": _record("F", exit_status=0)}])
    handle = back.submit(_job(tmp_path, walltime_s=60.0))
    start = time.monotonic()
    back.wait([handle], timeout_s=30.0)
    assert time.monotonic() - start >= 0.6


# ==========================================================================
# The record parser
# ==========================================================================


def test_a_wrapped_attribute_is_joined_and_not_truncated(tmp_path):
    """The real command wraps a long value onto a tab-indented line."""
    text = ("Job Id: 1.srv\n"
            "    Resource_List.select = 1:ngpus=4:ncpus=64:mem=220GB:\n"
            "\tplace=scatter\n"
            "    job_state = R\n")
    parsed = _parse_attributes(text)
    assert parsed["Resource_List.select"].endswith("place=scatter")
    assert parsed["job_state"] == "R"


def test_a_line_that_is_not_an_attribute_ends_a_wrapped_value():
    """And a tab-indented line is a continuation or nothing: read as a new
    attribute it would turn the tail of a wrapped value into a record of its
    own, named after whatever word the wrap happened to break on."""
    text = ("Job Id: 1.srv\n"
            "    A = one\n"
            "\tmore\n"
            "Job Id: 2.srv\n"
            "\tstray = 1\n")
    parsed = _parse_attributes(text)
    assert parsed["A"] == "onemore"
    assert "stray" not in parsed


def test_the_state_table_covers_every_letter_the_farm_reports():
    """Eleven letters, and only one of them maps to unknown."""
    assert set(PBS_STATES) == set("QHWTRSBEFXM")
    assert [k for k, v in PBS_STATES.items() if v is JobState.UNKNOWN] == ["M"]


# ==========================================================================
# The wall-clock limit
# ==========================================================================


@pytest.mark.parametrize("seconds,text", [
    (4 * 3600, "04:00:00"), (6 * 3600, "06:00:00"), (10 * 3600, "10:00:00"),
    (12 * 3600, "12:00:00"), (24 * 3600, "24:00:00"),
    (15 * 3600, "15:00:00")])
def test_every_archived_wall_clock_limit_is_written_the_way_it_was(seconds,
                                                                   text):
    """The five distinct limits the twenty job files state as directives, and
    the sixth, which one of them only ever gives on the command line for its
    second pass."""
    assert _walltime(seconds) == text


@pytest.mark.parametrize("seconds,text", [(1, "00:00:01"), (60, "00:01:00"),
                                          (3600, "01:00:00")])
def test_a_short_limit_is_written_with_every_field(seconds, text):
    assert _walltime(seconds) == text


def test_a_limit_is_rounded_up_and_never_down():
    """Asking for the floor asks for less time than the caller said."""
    assert _walltime(90.5) == "00:01:31"
    assert _walltime(0.1) == "00:00:01"


@pytest.mark.parametrize("bad", [0, -1, None])
def test_a_limit_that_is_not_a_duration_is_refused(bad):
    with pytest.raises(SchedulerError, match="positive number of seconds"):
        _walltime(bad)


# ==========================================================================
# The registry: a further backend is a plugin
# ==========================================================================


def test_both_backends_are_registered_under_their_names():
    assert BACKENDS["local"] is LocalBackend
    assert BACKENDS["pbs"] is PbsBackend


def test_a_backend_is_built_by_name(tmp_path):
    assert isinstance(backend("local", lanes=2), LocalBackend)


def test_an_unregistered_name_is_refused_with_the_ones_that_exist():
    with pytest.raises(SchedulerError, match=r"registered backends are"):
        backend("slurm")


def test_a_plugin_backend_is_reached_the_same_way():
    class _Plugin:
        name = "plugin"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def check(self, jobs):
            pass

        def submit_all(self, jobs):
            return []

        def state(self, handle):
            return Report(JobState.UNKNOWN, detail="a plugin")

        def cancel(self, handle):
            pass

    register_backend("plugin-under-test", _Plugin)
    try:
        made = backend("plugin-under-test", lanes=1)
        assert made.kwargs == {"lanes": 1}
    finally:
        del BACKENDS["plugin-under-test"]


def test_a_name_already_taken_is_refused_unless_replacement_is_asked_for():
    """A campaign sent to the wrong farm looks exactly like one sent right."""
    with pytest.raises(SchedulerError, match="already registered"):
        register_backend("local", LocalBackend)
    register_backend("local", LocalBackend, replace=True)
    assert BACKENDS["local"] is LocalBackend


def test_registering_something_that_is_not_callable_is_refused():
    with pytest.raises(SchedulerError, match="something callable"):
        register_backend("not-callable", object())


@pytest.mark.parametrize("name", ["", "   ", None])
def test_registering_without_a_name_is_refused(name):
    with pytest.raises(SchedulerError, match="needs a name"):
        register_backend(name, LocalBackend)


def test_half_a_backend_is_refused_when_it_is_built():
    """Half an interface fails in the middle of a campaign, not before it."""
    class _Half:
        def check(self, jobs):
            pass

    register_backend("half-under-test", _Half)
    try:
        with pytest.raises(SchedulerError, match="has no submit_all"):
            backend("half-under-test")
    finally:
        del BACKENDS["half-under-test"]


def test_both_built_in_backends_answer_the_whole_interface(tmp_path):
    for made in (LocalBackend(lanes=1),
                 PbsBackend(queue="ai", project="1")):
        for method in ("check", "submit_all", "submit", "state", "cancel",
                       "wait"):
            assert callable(getattr(made, method))
        assert isinstance(made.name, str)
