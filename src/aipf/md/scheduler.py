"""Submit a campaign of state points to a backend and report each job's state honestly.

A campaign is a list of :class:`~aipf.md.request.StatePoint`; this module turns it into
jobs, submits them (``local`` or ``pbs``; a further backend registers as a plugin) and
answers one question per job: which :class:`JobState` it is in. ``UNKNOWN`` always
carries a reason and never an exit code. A state query is retried; a submission never is.
"""
from __future__ import annotations

import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from aipf.md.request import StatePoint

__all__ = [
    "BACKENDS",
    "Backend",
    "Handle",
    "Job",
    "JobState",
    "PBS_STATES",
    "PartialSubmission",
    "PbsSpec",
    "Report",
    "SchedulerError",
    "TERMINAL",
    "WALLTIME_ENV",
    "LocalBackend",
    "PbsBackend",
    "activation",
    "backend",
    "batch_job",
    "job_name",
    "jobs_for",
    "register_backend",
]


class JobState(str, Enum):
    """What is known about one job; ``UNKNOWN`` means asked and could not tell."""

    NOT_SUBMITTED = "not_submitted"
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    UNKNOWN = "unknown"


#: States that will not change again (``UNKNOWN`` may be asked again).
TERMINAL = frozenset({JobState.FINISHED, JobState.FAILED})

#: The job's wall-clock limit is exported under this name, from one place only.
WALLTIME_ENV = "AIPF_WALLTIME_S"


class SchedulerError(RuntimeError):
    """A campaign that cannot be submitted, and why."""


class PartialSubmission(SchedulerError):
    """Some jobs went and then one did not; the submitted handles are on ``.handles``."""

    def __init__(self, handles: Sequence["Handle"], cause: BaseException):
        self.handles = tuple(handles)
        super().__init__(
            f"{len(self.handles)} job(s) were submitted and then one was "
            f"refused: {cause}. The submitted handles are on this exception "
            f"as .handles; they are running and must be polled or cancelled")


@dataclass(frozen=True)
class Report:
    """One job's state, and what the backend could back up.

    ``exit_code`` is ``None`` for "not known", never a stand-in for zero. ``detail`` is
    required for ``UNKNOWN`` and ``FAILED``. ``raw`` is what the farm said, verbatim.
    """

    state: JobState
    exit_code: int | None = None
    detail: str = ""
    raw: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, JobState):
            raise TypeError(f"state must be a JobState, got {self.state!r}")
        if self.state in (JobState.FINISHED, JobState.FAILED):
            if self.state is JobState.FINISHED and self.exit_code != 0:
                raise ValueError(
                    f"a finished job exited {self.exit_code!r}; a non-zero "
                    "exit is a failure and is reported as one, and an exit "
                    "nobody read is not a finished job but an unknown one")
            if self.state is JobState.FAILED and self.exit_code == 0:
                raise ValueError(
                    "a failed job cannot have exited zero; either it worked "
                    "or the code is not known, and 'not known' is None")
        elif self.exit_code is not None:
            raise ValueError(
                f"state {self.state.value!r} carries exit code "
                f"{self.exit_code!r}; only a job that ran to completion has "
                "one")
        if self.state in (JobState.UNKNOWN, JobState.FAILED) and not self.detail:
            raise ValueError(
                f"a {self.state.value!r} report needs a reason; without one a "
                "caller cannot tell a purged record from a dead job")

    @property
    def is_terminal(self) -> bool:
        """Whether this state can still change."""
        return self.state in TERMINAL

    def to_record(self) -> dict[str, Any]:
        """A plain dictionary, ready to be written beside a manifest."""
        return {"state": self.state.value, "exit_code": self.exit_code,
                "detail": self.detail, "raw": self.raw}


# -- What is submitted ----------

#: A job name doubles as a file name and as the farm's label for the job.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")

#: An environment variable name, as a shell will accept it.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class Job:
    """One unit of work: what to run, where, and under what limit.

    ``name`` labels the job and stems its job file and logs; ``workdir`` holds them and is
    where the body starts. An ``env`` value of ``None`` removes that variable. ``walltime_s``
    is in seconds (``None`` = none; the batch backend requires one). ``stdout == stderr``
    joins the streams. ``script_name`` is the job file's name in ``workdir`` (default
    ``<name>.sh``).
    """

    name: str
    script: str
    workdir: Path
    env: Mapping[str, str | None] = field(default_factory=dict)
    walltime_s: float | None = None
    stdout: Path | None = None
    stderr: Path | None = None
    script_name: str | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise SchedulerError(
                f"job name {self.name!r} is not usable: it labels the job on "
                "the farm and names its files, so it must start with a letter "
                "or digit and hold only letters, digits, dot, dash, plus and "
                "underscore")
        if not isinstance(self.script, str) or not self.script.strip():
            raise SchedulerError(f"job {self.name!r} has an empty script")
        set_(self, "workdir", Path(self.workdir))
        if self.script_name is not None and (
                not isinstance(self.script_name, str) or not _NAME.match(self.script_name)):
            raise SchedulerError(
                f"job {self.name!r}: script_name {self.script_name!r} is not a plain file "
                "name; the job file is written into the job's own workdir")

        env: dict[str, str | None] = {}
        for key, value in dict(self.env).items():
            if not isinstance(key, str) or not _ENV_NAME.match(key):
                raise SchedulerError(
                    f"job {self.name!r}: {key!r} is not an environment "
                    "variable name")
            if value is None:
                env[key] = None
                continue
            text = value if isinstance(value, str) else str(value)
            if "\n" in text or "\0" in text:
                raise SchedulerError(
                    f"job {self.name!r}: the value of {key} spans lines or "
                    "holds a null; neither survives being passed with a job")
            env[key] = text
        if WALLTIME_ENV in env:
            raise SchedulerError(
                f"job {self.name!r} sets {WALLTIME_ENV} itself. The wall-clock "
                "limit is requested in one place and exported from it; a job "
                "told its budget twice is a job whose two answers can "
                "disagree")
        set_(self, "env", env)

        if self.walltime_s is not None:
            if isinstance(self.walltime_s, bool) or not isinstance(
                    self.walltime_s, (int, float)):
                raise SchedulerError(
                    f"job {self.name!r}: walltime_s must be a number, got "
                    f"{self.walltime_s!r}")
            set_(self, "walltime_s", float(self.walltime_s))
            if not self.walltime_s > 0.0 or math.isinf(self.walltime_s):
                raise SchedulerError(
                    f"job {self.name!r}: walltime_s must be a positive finite "
                    f"number of seconds, got {self.walltime_s!r}")

        set_(self, "stdout",
             self.workdir / f"{self.name}.out" if self.stdout is None
             else Path(self.stdout))
        set_(self, "stderr",
             self.workdir / f"{self.name}.err" if self.stderr is None
             else Path(self.stderr))

    @property
    def script_path(self) -> Path:
        """Where the job file is written."""
        return self.workdir / (self.script_name or f"{self.name}.sh")

    @property
    def joins_streams(self) -> bool:
        """Whether the caller asked for one file rather than two."""
        return self.stdout == self.stderr

    @property
    def job_env(self) -> dict[str, str]:
        """The variables actually passed: removals taken out, the wall-clock limit added."""
        passed = {k: v for k, v in self.env.items() if v is not None}
        if self.walltime_s is not None:
            passed[WALLTIME_ENV] = repr(self.walltime_s)
        return passed

    @property
    def unset_env(self) -> tuple[str, ...]:
        """Variables the job must NOT inherit, in the order given."""
        return tuple(k for k, v in self.env.items() if v is None)


@dataclass(frozen=True)
class Handle:
    """What a submitted job is known by afterwards; enough to poll it from another process."""

    backend: str
    job_id: str
    name: str
    workdir: Path
    stdout: Path
    stderr: Path
    submitted_at: float

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        for name in ("workdir", "stdout", "stderr"):
            set_(self, name, Path(getattr(self, name)))
        if not str(self.job_id).strip():
            raise SchedulerError(
                f"job {self.name!r} was submitted and came back without an "
                "identifier, so nothing can ever be asked about it")

    def to_record(self) -> dict[str, Any]:
        """A plain dictionary, ready to be written beside a manifest."""
        return {"backend": self.backend, "job_id": self.job_id,
                "name": self.name, "workdir": str(self.workdir),
                "stdout": str(self.stdout), "stderr": str(self.stderr),
                "submitted_at": self.submitted_at}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Handle":
        """Rebuild a handle written by an earlier process."""
        missing = [k for k in ("backend", "job_id", "name", "workdir",
                               "stdout", "stderr", "submitted_at")
                   if k not in record]
        if missing:
            raise SchedulerError(f"handle record is missing {sorted(missing)}")
        return cls(backend=record["backend"], job_id=record["job_id"],
                   name=record["name"], workdir=Path(record["workdir"]),
                   stdout=Path(record["stdout"]),
                   stderr=Path(record["stderr"]),
                   submitted_at=float(record["submitted_at"]))


@runtime_checkable
class Backend(Protocol):
    """The one interface: a ``name`` and ``check``, ``submit_all``, ``state``, ``cancel``."""

    name: str

    def check(self, jobs: Sequence[Job]) -> None:
        """Refuse a campaign that cannot be submitted. Submits nothing."""

    def submit_all(self, jobs: Sequence[Job]) -> list[Handle]:
        """Check, write every job file, then submit."""

    def state(self, handle: Handle) -> Report:
        """What is known about one submitted job."""

    def cancel(self, handle: Handle) -> None:
        """Stop one submitted job."""


# -- Shared machinery ----------


def _check_distinct(jobs: Sequence[Job]) -> None:
    """Refuse a campaign whose jobs share a name, and so a job file and two logs."""
    seen: set[str] = set()
    clashes: list[str] = []
    for job in jobs:
        if job.name in seen and job.name not in clashes:
            clashes.append(job.name)
        seen.add(job.name)
    if clashes:
        raise SchedulerError(
            f"campaign has {len(clashes)} repeated job name(s): "
            f"{sorted(clashes)}; each would overwrite the other's job file "
            "and logs")


def _resolve_shell(shell: str | None) -> str:
    """The interpreter a job file is written for, resolved on this machine."""
    if shell is not None:
        found = shutil.which(shell)
        if found is None:
            raise SchedulerError(f"no interpreter {shell!r} on this machine")
        return found
    found = shutil.which("bash")
    if found is None:
        raise SchedulerError(
            "no bash on this machine and none was named; pass shell= with the "
            "interpreter the job files should be written for")
    return found


def _preamble(job: Job, shell: str) -> str:
    """The lines every job file starts with: shebang, ``set -u``, ``set -o pipefail``, one
    ``unset`` per removed variable, ``cd`` into the workdir. Deliberately no ``set -e``.
    """
    lines = [f"#!{shell}", "set -u", "set -o pipefail"]
    lines += [f"unset {name}" for name in job.unset_env]
    lines.append(f"cd {job.workdir}")
    return "\n".join(lines)


def _write_job_file(job: Job, shell: str) -> Path:
    """Write one job file, or refuse with the path that could not take it."""
    path = job.script_path
    body = job.script if job.script.endswith("\n") else job.script + "\n"
    try:
        path.write_text(_preamble(job, shell) + "\n" + body, encoding="utf-8")
        path.chmod(0o755)
    except OSError as exc:
        raise SchedulerError(
            f"job {job.name!r}: cannot write its job file at {path}: "
            f"{exc}") from exc
    return path


def _prepare_directories(jobs: Sequence[Job]) -> None:
    """Make every directory a job writes into, for the whole campaign before any submission."""
    for job in jobs:
        for directory in {job.workdir, job.stdout.parent, job.stderr.parent}:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise SchedulerError(
                    f"job {job.name!r}: cannot make {directory}: "
                    f"{exc}") from exc
            if not os.access(directory, os.W_OK):
                raise SchedulerError(
                    f"job {job.name!r}: {directory} is not writable, so its "
                    "job file and its logs have nowhere to go")


def _wait(get: Callable[[Handle], Report], handles: Sequence[Handle],
          timeout_s: float | None, interval_s: float) -> dict[str, Report]:
    """Poll until every handle is terminal or the timeout passes; return what is known."""
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while True:
        reports = {h.job_id: get(h) for h in handles}
        if all(r.is_terminal for r in reports.values()):
            return reports
        if deadline is not None and time.monotonic() >= deadline:
            return reports
        time.sleep(interval_s)


# -- The local backend ----------


@dataclass
class _Lane:
    """One running job: the process, its log, and when it must be dead by."""

    job: Job
    process: subprocess.Popen
    log: Any
    started_at: float
    deadline: float | None


class LocalBackend:
    """Runs the campaign on this machine, ``lanes`` jobs at a time (no default).

    A timed-out job is killed by process group; a job that could not start is ``FAILED``
    with a reason.
    """

    name = "local"

    def __init__(self, *, lanes: int, shell: str | None = None,
                 poll_interval_s: float = 0.05, kill_grace_s: float = 20.0):  # seconds between SIGTERM and SIGKILL
        if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes < 1:
            raise SchedulerError(
                f"lanes must be a positive integer, got {lanes!r}")
        if poll_interval_s <= 0.0:
            raise SchedulerError(
                f"poll_interval_s must be positive, got {poll_interval_s!r}")
        if kill_grace_s < 0.0:
            raise SchedulerError(
                f"kill_grace_s must not be negative, got {kill_grace_s!r}")
        self.lanes = lanes
        self.shell = _resolve_shell(shell)
        self.poll_interval_s = float(poll_interval_s)
        self.kill_grace_s = float(kill_grace_s)
        self._pending: list[tuple[Handle, Job, Path]] = []
        self._running: dict[str, _Lane] = {}
        self._done: dict[str, Report] = {}
        self._serial = 0

    # -- the interface ---------------------------------------------------

    def check(self, jobs: Sequence[Job]) -> None:
        """Refuse a campaign that cannot run here. Starts nothing."""
        jobs = list(jobs)
        _check_distinct(jobs)
        _prepare_directories(jobs)

    def submit_all(self, jobs: Sequence[Job]) -> list[Handle]:
        """Check the campaign, write every job file, then queue them all."""
        jobs = list(jobs)
        self.check(jobs)
        written = [_write_job_file(job, self.shell) for job in jobs]
        handles = []
        for job, path in zip(jobs, written):
            self._serial += 1
            handle = Handle(backend=self.name,
                            job_id=f"{self.name}-{os.getpid()}-{self._serial}",
                            name=job.name, workdir=job.workdir,
                            stdout=job.stdout, stderr=job.stderr,
                            submitted_at=time.time())
            self._pending.append((handle, job, path))
            handles.append(handle)
        self._pump()
        return handles

    def submit(self, job: Job) -> Handle:
        """One job is a campaign of one."""
        return self.submit_all([job])[0]

    def state(self, handle: Handle) -> Report:
        """What is known about one job of this pool's; a handle it did not issue is ``UNKNOWN``."""
        self._pump()
        if handle.job_id in self._done:
            return self._done[handle.job_id]
        if handle.job_id in self._running:
            lane = self._running[handle.job_id]
            return Report(JobState.RUNNING,
                          detail=f"started {time.time() - lane.started_at:.1f} "
                                 "s ago")
        if any(h.job_id == handle.job_id for h, _, _ in self._pending):
            return Report(JobState.QUEUED,
                          detail=f"waiting for one of {self.lanes} lane(s)")
        return Report(
            JobState.UNKNOWN,
            detail=f"this pool did not start {handle.job_id!r}; a local job is "
                   "known only to the process that ran it, so a handle from an "
                   "earlier process cannot be polled here")

    def cancel(self, handle: Handle) -> None:
        """Stop one job, whether it is waiting or already running."""
        self._pump()
        if handle.job_id in self._done:
            return
        for index, (waiting, _, _) in enumerate(self._pending):
            if waiting.job_id == handle.job_id:
                del self._pending[index]
                self._done[handle.job_id] = Report(
                    JobState.FAILED,
                    detail="cancelled before it started")
                return
        lane = self._running.get(handle.job_id)
        if lane is None:
            return
        code = self._kill(lane)
        self._finish(handle.job_id, lane, code, "cancelled while running")

    def wait(self, handles: Sequence[Handle], *, timeout_s: float | None = None,
             interval_s: float | None = None) -> dict[str, Report]:
        """Run the pool until every handle is finished, failed or gone."""
        return _wait(self.state, list(handles), timeout_s,
                     self.poll_interval_s if interval_s is None else interval_s)

    # -- the pool --------------------------------------------------------

    def _pump(self) -> None:
        """Reap what has ended, kill what has overrun, start what fits."""
        for job_id in list(self._running):
            lane = self._running[job_id]
            code = lane.process.poll()
            if code is None:
                if lane.deadline is not None and time.monotonic() > lane.deadline:
                    killed = self._kill(lane)
                    self._finish(
                        job_id, lane, killed,
                        f"exceeded its wall-clock limit of "
                        f"{lane.job.walltime_s} s and its process group was "
                        "killed")
                continue
            self._finish(job_id, lane, code, "")
        while self._pending and len(self._running) < self.lanes:
            handle, job, path = self._pending.pop(0)
            self._start(handle, job, path)

    def _start(self, handle: Handle, job: Job, path: Path) -> None:
        """Launch one job in its own session, or record why it did not."""
        env = os.environ.copy()
        for name in job.unset_env:
            env.pop(name, None)
        env.update(job.job_env)
        try:
            out = open(job.stdout, "w", encoding="utf-8")
        except OSError as exc:
            self._done[handle.job_id] = Report(
                JobState.FAILED,
                detail=f"its log at {job.stdout} could not be opened: {exc}")
            return
        err = out if job.joins_streams else None
        try:
            if err is None:
                err = open(job.stderr, "w", encoding="utf-8")
        except OSError as exc:
            out.close()
            self._done[handle.job_id] = Report(
                JobState.FAILED,
                detail=f"its log at {job.stderr} could not be opened: {exc}")
            return
        try:
            process = subprocess.Popen(
                [self.shell, str(path)], cwd=str(job.workdir), env=env,
                stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                start_new_session=True)
        except OSError as exc:
            out.close()
            if err is not out:
                err.close()
            self._done[handle.job_id] = Report(
                JobState.FAILED, detail=f"could not be started: {exc}")
            return
        deadline = (None if job.walltime_s is None
                    else time.monotonic() + job.walltime_s)
        self._running[handle.job_id] = _Lane(
            job=job, process=process, log=(out, err), started_at=time.time(),
            deadline=deadline)

    def _kill(self, lane: _Lane) -> int | None:
        """Signal the job's whole process group (SIGTERM, then SIGKILL) and make sure of it."""
        process = lane.process
        try:
            group = os.getpgid(process.pid)
        except OSError:
            group = None
        for sig, grace in ((signal.SIGTERM, self.kill_grace_s),
                           (signal.SIGKILL, 10.0)):
            if process.poll() is not None:
                break
            try:
                if group is not None:
                    os.killpg(group, sig)
                else:
                    process.send_signal(sig)
            except ProcessLookupError:
                # Already gone. Any other error is raised, never swallowed.
                break
            try:
                process.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        return process.poll()

    def _finish(self, job_id: str, lane: _Lane, code: int | None,
                detail: str) -> None:
        """Close the logs and record the one answer this job now has."""
        for stream in dict.fromkeys(lane.log):
            if stream is not None:
                stream.close()
        self._running.pop(job_id, None)
        if code == 0:
            # A job that exited 0 while being cancelled or killed still worked.
            self._done[job_id] = Report(JobState.FINISHED, exit_code=0,
                                        detail=detail)
            return
        if not detail:
            detail = (f"killed by signal {-code}" if code is not None and code < 0
                      else f"exited {code}")
        self._done[job_id] = Report(JobState.FAILED, exit_code=code,
                                    detail=detail)


# -- The batch backend ----------

#: Batch-server state letters. ``M`` (moved by a routing queue) is UNKNOWN until followed;
#: ``F`` is finished only once its exit status says which way.
PBS_STATES: dict[str, JobState] = {
    "Q": JobState.QUEUED,      # queued
    "H": JobState.QUEUED,      # held
    "W": JobState.QUEUED,      # waiting for its start time
    "T": JobState.QUEUED,      # in transit
    "R": JobState.RUNNING,
    "S": JobState.RUNNING,     # suspended, but it has started
    "B": JobState.RUNNING,     # an array job with subjobs begun
    "E": JobState.RUNNING,     # exiting; the exit status is not final yet
    "F": JobState.FINISHED,    # subject to the exit status
    "X": JobState.FINISHED,    # a finished subjob, likewise
    "M": JobState.UNKNOWN,     # moved to another server; ask that one
}

_ATTRIBUTE = re.compile(r"^\s{2,}([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*)$")
_UNKNOWN_JOB = re.compile(r"unknown job id", re.I)

#: Replies to a delete of a job already past deleting; neither is an error.
_PAST_DELETING = re.compile(
    r"request invalid for state of job|job has finished", re.I)


class PbsBackend:
    """Submits the campaign to a batch server, and asks it what happened.

    ``queue``, ``project`` and ``resources`` have no defaults. Values are rendered into
    directives (never a shell variable), a resource is never a passed variable, and a
    passed value holding a comma is refused.
    """

    name = "pbs"

    def __init__(self, *, queue: str, project: str,
                 resources: Mapping[str, str] | None = None,
                 shell: str | None = None, qsub: str = "qsub",
                 qstat: str = "qstat", qdel: str = "qdel",
                 command_timeout_s: float = 120.0, query_retries: int = 2,
                 retry_pause_s: float = 2.0, poll_interval_s: float = 30.0):
        if not isinstance(queue, str) or not queue.strip():
            raise SchedulerError("a queue must be named; this package has no "
                                 "default one and the right answer differs "
                                 "per machine and per project")
        if not isinstance(project, str) or not project.strip():
            raise SchedulerError("a project code must be given; every archived "
                                 "job file names one and a job without one is "
                                 "refused by the server, not by this module")
        if query_retries < 0:
            raise SchedulerError(
                f"query_retries must not be negative, got {query_retries!r}")
        if retry_pause_s < 0.0:
            raise SchedulerError(
                f"retry_pause_s must not be negative, got {retry_pause_s!r}")
        self.queue = queue.strip()
        self.project = project.strip()
        self.resources = dict(resources or {})
        for key in self.resources:
            if key == "walltime":
                raise SchedulerError(
                    "walltime belongs to the job, not to the backend's "
                    "resources: a limit stated twice is a limit that can "
                    "disagree with itself, which is what an archived campaign "
                    "warns about in a comment")
        self.shell = _resolve_shell(shell)
        self.qsub = qsub
        self.qstat = qstat
        self.qdel = qdel
        self.command_timeout_s = float(command_timeout_s)
        self.query_retries = int(query_retries)
        self.retry_pause_s = float(retry_pause_s)
        self.poll_interval_s = float(poll_interval_s)
        self._queue_checked = False

    # -- the interface ---------------------------------------------------

    def check(self, jobs: Sequence[Job]) -> None:
        """Refuse a campaign the server would refuse, before submitting any (queue asked once)."""
        jobs = list(jobs)
        _check_distinct(jobs)
        for job in jobs:
            if job.walltime_s is None:
                raise SchedulerError(
                    f"job {job.name!r} names no wall-clock limit. A batch queue "
                    "applies its own site default to a job that names none, "
                    "and nothing in this package knows what that default is "
                    "here, so the job would run under a limit nobody chose")
            for key, value in job.job_env.items():
                if "," in value:
                    raise SchedulerError(
                        f"job {job.name!r}: the value of {key} holds a comma. "
                        "The submitter separates passed variables by comma and "
                        "refuses the whole job rather than mis-splitting it")
        self._check_queue()
        _prepare_directories(jobs)

    def submit_all(self, jobs: Sequence[Job]) -> list[Handle]:
        """Check, write every job file, then submit one at a time. A later failure raises
        :class:`PartialSubmission` carrying the handles already obtained.
        """
        jobs = list(jobs)
        self.check(jobs)
        written = [self._write(job) for job in jobs]
        handles: list[Handle] = []
        for job, path in zip(jobs, written):
            try:
                handles.append(self._submit_written(job, path))
            except BaseException as exc:
                # Everything, an interruption included: the handles reach the jobs.
                if handles:
                    raise PartialSubmission(handles, exc) from exc
                raise
        return handles

    def submit(self, job: Job) -> Handle:
        """One job is a campaign of one."""
        return self.submit_all([job])[0]

    def job_text(self, job: Job) -> str:
        """The whole job file: interpreter line, directives, preamble, body."""
        body = job.script if job.script.endswith("\n") else job.script + "\n"
        preamble = _preamble(job, self.shell).split("\n")
        return "\n".join([preamble[0], *self._directives(job), *preamble[1:]]) + "\n" + body

    def write(self, job: Job) -> Path:
        """Write one job file exactly as it would be submitted, and submit nothing (a dry run).

        The server is not asked anything; the job's directories are made."""
        if job.walltime_s is None:
            raise SchedulerError(
                f"job {job.name!r} names no wall-clock limit, and a batch job file "
                "states one")
        _prepare_directories([job])
        return self._write(job)

    def _write(self, job: Job) -> Path:
        path = job.script_path
        try:
            path.write_text(self.job_text(job), encoding="utf-8")
            path.chmod(0o755)
        except OSError as exc:
            raise SchedulerError(
                f"job {job.name!r}: cannot write its job file at {path}: "
                f"{exc}") from exc
        return path

    def state(self, handle: Handle) -> Report:
        """Ask the server about one job, following it if a routing queue moved it."""
        record, refusal = self._qstat(handle.job_id)
        if record is None:
            return refusal
        mapped = PBS_STATES.get(record.get("job_state", ""))
        if mapped is JobState.UNKNOWN:
            moved = self._follow_move(handle, record)
            if moved is not None:
                return moved
        return self._read(record, mapped)

    def cancel(self, handle: Handle) -> None:
        """Ask the server to delete the job."""
        result = self._run([self.qdel, handle.job_id])
        if result.returncode == 0:
            return
        said = result.stdout + result.stderr
        # Forgotten, or past deleting: nothing left to do, and not a failure.
        if _UNKNOWN_JOB.search(said) or _PAST_DELETING.search(said):
            return
        raise SchedulerError(
            f"cannot cancel {handle.job_id}: "
            f"{(result.stderr or result.stdout).strip()}")

    def wait(self, handles: Sequence[Handle], *, timeout_s: float | None = None,
             interval_s: float | None = None) -> dict[str, Report]:
        """Poll the server until every handle is terminal, or time runs out."""
        return _wait(self.state, list(handles), timeout_s,
                     self.poll_interval_s if interval_s is None else interval_s)

    # -- pieces ----------------------------------------------------------

    def _check_queue(self) -> None:
        """Refuse a queue that does not exist, or is not enabled and started."""
        if self._queue_checked:
            return
        result = self._run([self.qstat, "-Qf", self.queue])
        text = result.stdout + result.stderr
        if result.returncode != 0:
            raise SchedulerError(
                f"queue {self.queue!r} cannot be submitted to: "
                f"{text.strip() or f'the queue command exited {result.returncode}'}")
        attributes = _parse_attributes(text)
        for flag in ("enabled", "started"):
            value = attributes.get(flag, "")
            if value.strip().lower() in ("false", "no"):
                raise SchedulerError(
                    f"queue {self.queue!r} exists but is not {flag}, so it "
                    "would take this campaign and never run it")
        self._queue_checked = True

    def _directives(self, job: Job) -> list[str]:
        """The job file's header, one resolved value per line (a directive is never expanded)."""
        lines = [f"#PBS -N {job.name}",
                 f"#PBS -q {self.queue}",
                 f"#PBS -P {self.project}"]
        for key, value in self.resources.items():
            lines.append(f"#PBS -l {key}={value}")
        lines.append(f"#PBS -l walltime={_walltime(job.walltime_s)}")
        lines.append(f"#PBS -o {job.stdout}")
        if job.joins_streams:
            lines.append("#PBS -j oe")
        else:
            lines.append(f"#PBS -e {job.stderr}")
        return lines

    def _submit_written(self, job: Job, path: Path) -> Handle:
        """Submit one written job file; its identifier is the submitter's stripped output."""
        # Never empty: the wall-clock limit is always among the passed variables.
        passed = ",".join(f"{k}={v}" for k, v in job.job_env.items())
        command = [self.qsub, "-v", passed, str(path)]
        result = self._run(command)
        if result.returncode != 0:
            raise SchedulerError(
                f"job {job.name!r} was refused by the submitter (exit "
                f"{result.returncode}): "
                f"{(result.stderr or result.stdout).strip()}")
        job_id = result.stdout.strip()
        if not job_id:
            raise SchedulerError(
                f"job {job.name!r} was accepted and no identifier came back "
                f"(the submitter said {result.stderr.strip()!r}). It may be "
                "running; find it before submitting the campaign again, "
                "because a second submission would write over its output")
        return Handle(backend=self.name, job_id=job_id, name=job.name,
                      workdir=job.workdir, stdout=job.stdout,
                      stderr=job.stderr, submitted_at=time.time())

    def _read(self, record: Mapping[str, str],
              mapped: JobState | None) -> Report:
        """Turn one parsed record into the one answer it supports."""
        state = record.get("job_state", "")
        raw = record.get("_raw")
        if mapped is None:
            return Report(
                JobState.UNKNOWN,
                detail=f"the server reports state {state!r}, which this module "
                       "does not know how to read",
                raw=raw)
        if mapped is JobState.UNKNOWN:
            return Report(
                JobState.UNKNOWN,
                detail=f"the job was moved to "
                       f"{record.get('queue') or 'another queue'} and this "
                       "server no longer tracks it",
                raw=raw)
        # An exit status is written when the job ENDS: a record with one has run, whatever
        # its letter.
        if mapped is JobState.FINISHED or "Exit_status" in record:
            return self._finished(record, state)
        return Report(mapped, detail=f"the server reports {state!r}", raw=raw)

    def _qstat(self, job_id: str) -> tuple[dict[str, str] | None, Report | None]:
        """One job's attributes, or the report saying why there are none. The query is
        retried; a purged record is ``UNKNOWN``, not ``FAILED``.
        """
        last = ""
        for attempt in range(self.query_retries + 1):
            result = self._run([self.qstat, "-x", "-f", job_id])
            text = result.stdout + result.stderr
            if result.returncode == 0 and "job id" in text.lower():
                parsed = _parse_attributes(text)
                parsed["_raw"] = text
                return parsed, None
            last = text.strip() or f"the query exited {result.returncode}"
            if _UNKNOWN_JOB.search(text):
                return None, Report(
                    JobState.UNKNOWN,
                    detail=f"the server has no record of {job_id}. A finished "
                           "job is kept for a limited window and then purged, "
                           "so this is not a failure and not a success",
                    raw=text)
            if attempt < self.query_retries:
                # Linear back-off.
                time.sleep(self.retry_pause_s * (attempt + 1))
        return None, Report(
            JobState.UNKNOWN,
            detail=f"the server could not be asked about {job_id} in "
                   f"{self.query_retries + 1} attempt(s): {last}",
            raw=last)

    def _follow_move(self, handle: Handle,
                     record: Mapping[str, str]) -> Report | None:
        """Ask the server the job was moved to, when its queue reads ``name@server``."""
        queue = record.get("queue", "")
        if "@" not in queue:
            return None
        server = queue.split("@", 1)[1].strip()
        if not server:
            return None
        moved, refusal = self._qstat(f"{handle.job_id}@{server}")
        if moved is None:
            return refusal
        state = moved.get("job_state", "")
        mapped = PBS_STATES.get(state)
        if mapped is None or mapped is JobState.UNKNOWN:
            return Report(
                JobState.UNKNOWN,
                detail=f"the job was moved to {queue} and that server reports "
                       f"state {state!r}",
                raw=moved.get("_raw"))
        if mapped is JobState.FINISHED:
            return self._finished(moved, state)
        return Report(mapped, detail=f"moved to {queue}, which reports {state!r}",
                      raw=moved.get("_raw"))

    def _finished(self, record: Mapping[str, str], state: str) -> Report:
        """Which way a finished job went; with no exit status it is ``UNKNOWN``."""
        raw = record.get("_raw")
        text = record.get("Exit_status")
        if text is None:
            return Report(
                JobState.UNKNOWN,
                detail=f"the server reports {state!r} and no exit status, so "
                       "whether the job worked is not recorded",
                raw=raw)
        try:
            code = int(text.strip())
        except ValueError:
            return Report(
                JobState.UNKNOWN,
                detail=f"the server reports an exit status of {text!r}, which "
                       "is not a number",
                raw=raw)
        if code == 0:
            return Report(JobState.FINISHED, exit_code=0,
                          detail=f"the server reports {state!r}", raw=raw)
        return Report(JobState.FAILED, exit_code=code,
                      detail=_why_failed(code), raw=raw)

    def _run(self, command: Sequence[str]) -> subprocess.CompletedProcess:
        """Run one farm command, and turn a missing one into a refusal."""
        try:
            return subprocess.run(list(command), capture_output=True, text=True,
                                  timeout=self.command_timeout_s)
        except FileNotFoundError as exc:
            raise SchedulerError(
                f"no {command[0]!r} on this machine, so this backend cannot be "
                "used here") from exc
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                list(command), returncode=124, stdout="",
                stderr=f"{command[0]} did not answer within "
                       f"{self.command_timeout_s} s")


def _parse_attributes(text: str) -> dict[str, str]:
    """The ``name = value`` lines of one record, tab-indented continuations joined."""
    attributes: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        if current is not None and line.startswith("\t"):
            attributes[current] += line.strip()
            continue
        match = _ATTRIBUTE.match(line)
        if match is None:
            current = None
            continue
        current = match.group(1)
        attributes[current] = match.group(2).strip()
    return attributes


def _why_failed(code: int) -> str:
    """What a non-zero exit status says, without claiming more than it does."""
    if code < 0:
        return (f"the server reports exit status {code}, which is one of its "
                "own job-level errors rather than the script's")
    if code > 128:
        return (f"the script was killed by signal {code - 128} "
                f"(exit status {code})")
    return f"the script exited {code}"


def _walltime(seconds: float | None) -> str:
    """Seconds as the submitter's ``HH:MM:SS``, rounded up."""
    if seconds is None or seconds <= 0.0:
        raise SchedulerError(
            f"a wall-clock limit must be a positive number of seconds, got "
            f"{seconds!r}")
    total = int(math.ceil(seconds))
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


# -- One command as one batch job ----------


@dataclass(frozen=True)
class PbsSpec:
    """Where and with what one batch job runs: the site's queue and project, its devices and limit.

    ``ncpus`` and ``mem`` are optional (the server's defaults otherwise); ``mem`` is in the
    server's own spelling, ``"110gb"``."""

    queue: str
    project: str
    gpus: int
    walltime_h: float
    ncpus: int | None = None
    mem: str | None = None

    def __post_init__(self) -> None:
        for name in ("queue", "project"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SchedulerError(f"a batch job needs a {name}, got {value!r}")
        for name in ("gpus", "ncpus"):
            value = getattr(self, name)
            if value is None and name == "ncpus":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "gpus" else 1):
                raise SchedulerError(f"{name} is a whole number, got {value!r}")
        if isinstance(self.walltime_h, bool) or not isinstance(self.walltime_h, (int, float)) \
                or not 0.0 < float(self.walltime_h) < math.inf:
            raise SchedulerError(f"walltime_h is a positive number of hours, got {self.walltime_h!r}")

    @classmethod
    def from_site(cls, site: Any, *, walltime_h: float | None = None,
                  gpus: int | None = None) -> "PbsSpec":
        """The spec the site declares (``pbs_*`` facts), with ``walltime_h``/``gpus`` given on the line winning.

        A fact neither declared nor given is refused by :meth:`aipf.site.Site.require`, naming it."""
        needed = ["pbs_queue", "pbs_project"]
        needed += [] if walltime_h is not None else ["pbs_walltime_h"]
        needed += [] if gpus is not None else ["pbs_gpus"]
        site.require(*needed)
        return cls(queue=site.pbs_queue, project=site.pbs_project,
                   gpus=int(site.pbs_gpus if gpus is None else gpus),
                   walltime_h=float(site.pbs_walltime_h if walltime_h is None else walltime_h),
                   ncpus=site.pbs_ncpus, mem=site.pbs_mem)

    @property
    def select(self) -> str:
        """The one chunk this job asks for, in the server's syntax."""
        chunk = f"1:ngpus={self.gpus}"
        if self.ncpus is not None:
            chunk += f":ncpus={self.ncpus}"
        if self.mem is not None:
            chunk += f":mem={self.mem}"
        return chunk

    def backend(self, **kwargs: Any) -> "PbsBackend":
        """The batch backend that submits to this queue and project, with this chunk."""
        return PbsBackend(queue=self.queue, project=self.project,
                          resources={"select": self.select}, **kwargs)


def activation(prefix: str | Path | None = None) -> str:
    """Shell lines that put a job into the environment at ``prefix`` (default: this interpreter's).

    No ``conda`` is needed on the compute node: the environment's ``bin`` goes first on ``PATH``
    and its own activation hooks are sourced, as ``conda activate`` would."""
    root = Path(sys.prefix if prefix is None else prefix)
    return "\n".join([
        "# the environment the job was submitted from",
        f"export CONDA_PREFIX={shlex.quote(str(root))}",
        'export PATH="$CONDA_PREFIX/bin:$PATH"',
        "set +u",
        'for hook in "$CONDA_PREFIX/etc/conda/activate.d/"*.sh; do [ -r "$hook" ] && . "$hook"; done',
        "set -u",
    ])


def batch_job(spec: PbsSpec, *, name: str, command: Sequence[str], workdir: Path | str,
              env: Mapping[str, str | None] | None = None,
              script_name: str = "job.pbs", log: str = "job.log",
              prefix: str | Path | None = None) -> Job:
    """One command as one batch job: ``workdir/script_name`` activates the environment and runs
    ``command``; both streams go to ``workdir/log``."""
    workdir = Path(workdir)
    script = activation(prefix) + "\n" + shlex.join([str(c) for c in command]) + "\n"
    return Job(name=name, script=script, workdir=workdir, env=dict(env or {}),
               walltime_s=float(spec.walltime_h) * 3600.0,
               stdout=workdir / log, stderr=workdir / log, script_name=script_name)


def job_name(text: str) -> str:
    """``text`` as a job name the server and the file system both take."""
    cleaned = re.sub(r"[^A-Za-z0-9._+-]", "_", text).lstrip("._+-")
    return cleaned[:15] or "aipf"


# -- A campaign, and the plugin registry ----------


def jobs_for(campaign: Sequence[StatePoint], script_for: Callable[[StatePoint], str],
             root: Path | str, *, walltime_s: float | None = None,
             env_for: Callable[[StatePoint], Mapping[str, str | None]] | None = None
             ) -> list[Job]:
    """One job per state point, named by the point's tag and placed at ``root / tag``.
    What to run is the caller's ``script_for``.
    """
    root = Path(root)
    jobs = []
    for point in campaign:
        env = dict(env_for(point)) if env_for is not None else {}
        jobs.append(Job(name=point.tag, script=script_for(point),
                        workdir=root / point.tag, env=env,
                        walltime_s=walltime_s))
    _check_distinct(jobs)
    return jobs


#: Backends by name; a further one joins through :func:`register_backend`.
BACKENDS: dict[str, Callable[..., Backend]] = {
    "local": LocalBackend,
    "pbs": PbsBackend,
}


def register_backend(name: str, factory: Callable[..., Backend], *,
                     replace: bool = False) -> None:
    """Add a backend under a name; a name already taken needs ``replace=True``."""
    if not isinstance(name, str) or not name.strip():
        raise SchedulerError("a backend needs a name")
    if name in BACKENDS and not replace:
        raise SchedulerError(
            f"a backend is already registered as {name!r}; pass replace=True "
            "to mean it, because a campaign sent to the wrong farm looks "
            "exactly like one sent to the right one")
    if not callable(factory):
        raise SchedulerError(
            f"backend {name!r} must be registered with something callable, "
            f"got {factory!r}")
    BACKENDS[name] = factory


def backend(name: str, **kwargs: Any) -> Backend:
    """Build a backend by name, or refuse naming the ones that exist."""
    if name not in BACKENDS:
        raise SchedulerError(
            f"no backend {name!r}; registered backends are "
            f"{sorted(BACKENDS)}")
    made = BACKENDS[name](**kwargs)
    for method in ("check", "submit_all", "state", "cancel"):
        if not callable(getattr(made, method, None)):
            raise SchedulerError(
                f"backend {name!r} produced {made!r}, which has no "
                f"{method}(); a backend is whatever answers the whole "
                "interface, and half of one fails in the middle of a campaign")
    return made
