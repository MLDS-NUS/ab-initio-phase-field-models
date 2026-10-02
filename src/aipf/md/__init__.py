"""Running molecular dynamics: what is asked for, and what runs it.

The request vocabulary is here from the start because it is the part every
other piece of this module depends on. A template fills a deck from a state
point, a scheduler farms a campaign of them, and a finished run's record is
that request plus everything only the run could know.
"""
from aipf.md.doctor import Diagnosis, Finding, Probe, Verdict
# Named for what they inspect at the package level, where a bare
# ``examine`` or ``probe`` would say nothing, and where ``probe`` would
# read as a verb with no object. Inside their own module the short names
# are unambiguous, the same way the potential resolver is named here.
from aipf.md.doctor import examine as examine_environment
from aipf.md.doctor import probe as probe_interpreter
from aipf.md.engine import (
    BROKEN_STYLES,
    KOKKOS_PACKAGE,
    REMOVED_FROM_ENVIRONMENT,
    ROUTES,
    EngineError,
    Launch,
    Result,
    exit_cleanly,
    kokkos_args,
    normalise_route,
    unavailable_reason,
)
# Named for what they act on at the package level, where a bare ``run``,
# ``plan``, ``check_deck`` or ``pair_commands`` would say nothing, and where
# ``check_deck`` would read as the templates module's own deck check rather
# than as the question of whether a deck can run at all. Inside their own
# module the short names are unambiguous, the same way the potential resolver
# and the deck helpers are named here.
from aipf.md.engine import check_deck as check_deck_runs
from aipf.md.engine import clean_environment as clean_run_environment
from aipf.md.engine import pair_commands as pair_commands_for
from aipf.md.engine import plan as plan_launch
from aipf.md.engine import run as run_deck
from aipf.md.engine import strip_environment as strip_run_environment
from aipf.md.potentials import (
    Potential,
    PotentialError,
    check_e3nn,
    fallback_kernels,
)
# Named for what it resolves at the package level, where a bare ``resolve``
# would say nothing. Inside its own module the short name is unambiguous.
from aipf.md.potentials import resolve as resolve_potential
from aipf.md.request import (
    DUMP_FROM,
    PRESSURE_CONTROLLED,
    StatePoint,
    campaign_from_records,
    campaign_to_records,
    normalise_ensemble,
    normalise_geometry,
)
from aipf.md.scheduler import (
    Handle,
    Job,
    JobState,
    LocalBackend,
    PbsBackend,
    Report,
    SchedulerError,
    register_backend,
)
# Named for what it selects and for what it is built from, at the package
# level, where a bare ``backend`` or ``jobs_for`` would say nothing. Inside
# their own module the short names are unambiguous, the same way the potential
# resolver and the deck helpers are named here.
from aipf.md.scheduler import backend as scheduler_backend
from aipf.md.scheduler import jobs_for as jobs_for_campaign
from aipf.md.templates import Deck, DeckError
# Named for what they act on at the package level, where a bare ``builtin`` or
# ``fill`` would say nothing. Inside their own module the short names are
# unambiguous, the same way the potential resolver is named here.
from aipf.md.templates.builtin import builtin as builtin_deck
from aipf.md.templates import fill as fill_deck

__all__ = [
    "BROKEN_STYLES",
    "DUMP_FROM",
    "KOKKOS_PACKAGE",
    "PRESSURE_CONTROLLED",
    "REMOVED_FROM_ENVIRONMENT",
    "ROUTES",
    "Deck",
    "DeckError",
    "Diagnosis",
    "EngineError",
    "Finding",
    "Handle",
    "Job",
    "JobState",
    "Launch",
    "LocalBackend",
    "PbsBackend",
    "Potential",
    "PotentialError",
    "Probe",
    "Report",
    "Result",
    "SchedulerError",
    "StatePoint",
    "Verdict",
    "builtin_deck",
    "campaign_from_records",
    "campaign_to_records",
    "check_deck_runs",
    "check_e3nn",
    "clean_run_environment",
    "examine_environment",
    "exit_cleanly",
    "fallback_kernels",
    "fill_deck",
    "jobs_for_campaign",
    "kokkos_args",
    "normalise_ensemble",
    "normalise_geometry",
    "normalise_route",
    "pair_commands_for",
    "plan_launch",
    "probe_interpreter",
    "register_backend",
    "resolve_potential",
    "run_deck",
    "scheduler_backend",
    "strip_run_environment",
    "unavailable_reason",
]
