"""Site facts: the programs, files and batch settings on this machine a command needs, resolved in one place.
Each is looked up at call time: the variable in ``FACTS``, then ``[site] <name>`` in ``aipf.toml``, else
``None``; ``Site.load().require(...)`` refuses once, listing every missing fact and how to set it."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from aipf import paths

FACTS = {  # fact (also its key under [site]) -> the variable that sets it
    "lammps": "AIPF_LAMMPS",  # the LAMMPS binary (ML-IAP, and its python module for MACE)
    "lammps_python": "AIPF_LAMMPS_PYTHON",  # LAMMPS's python module's interpreter, if not this one
    "latexmk": "AIPF_LATEXMK",  # the latexmk that typesets figure labels
    "cuda_lib": "AIPF_CUDA_LIB",  # a site CUDA library directory for the loader path
    "mace_potential": "AIPF_MACE_POTENTIAL",  # a MACE model file, converted for ML-IAP (per system: below)
    "pbs_queue": "AIPF_PBS_QUEUE",  # the batch queue a --pbs job is submitted to
    "pbs_project": "AIPF_PBS_PROJECT",  # the project code the batch server charges
    "pbs_walltime_h": "AIPF_PBS_WALLTIME_H",  # a --pbs job's wall-clock limit, hours
    "pbs_gpus": "AIPF_PBS_GPUS",  # GPUs a --pbs job asks for
    "pbs_ncpus": "AIPF_PBS_NCPUS",  # CPU cores a --pbs job asks for (optional; the server's default otherwise)
    "pbs_mem": "AIPF_PBS_MEM",  # memory a --pbs job asks for, in the server's spelling (optional)
}

#: Facts that are values rather than files, and the type each is read as.
VALUES = {"pbs_queue": str, "pbs_project": str, "pbs_walltime_h": float, "pbs_gpus": int,
          "pbs_ncpus": int, "pbs_mem": str}

#: What ``require`` shows as the value of a fact in its remedy line (a file otherwise).
_EXAMPLE = {"pbs_queue": '"<queue>"', "pbs_project": '"<project code>"', "pbs_walltime_h": "24",
            "pbs_gpus": "1", "pbs_ncpus": "16", "pbs_mem": '"110gb"'}
_A_FILE = '"/path"'


#: Facts that may also be declared per system: ``<VARIABLE>_<SYSTEM>`` or ``[site.<fact>] <system> = ...``,
#: either winning over the fact itself, as ``[paths.raw]`` does for raw roots.
PER_SYSTEM = ("mace_potential",)


def per_system_variable(name: str, system: str) -> str:
    """The variable that declares fact ``name`` for one system, ``<VARIABLE>_<SYSTEM>``."""
    return f"{FACTS[name]}_{system.upper()}"


class MissingSiteFact(RuntimeError):
    """One or more site facts a command needs are not declared; ``.missing`` names them."""

    def __init__(self, missing: tuple[str, ...], message: str):
        self.missing = missing
        super().__init__(message)


def _read(name: str):
    """One fact as declared now: a path, a typed value, or ``None``; a value that does not parse is refused."""
    if name not in VALUES:
        return paths._lookup(FACTS[name], "site", name)
    text = paths._lookup_value(FACTS[name], "site", name)
    if text is None:
        return None
    try:
        return VALUES[name](text.strip())
    except ValueError:
        raise MissingSiteFact((name,), f"site fact {name} = {text!r} is not a {VALUES[name].__name__}; "
                                       f"set {FACTS[name]}, or {name} = {_EXAMPLE[name]} under [site] "
                                       f"in {paths._where_config()}") from None


@dataclass(frozen=True)
class Site:
    lammps: Path | None = None
    lammps_python: Path | None = None
    latexmk: Path | None = None
    cuda_lib: Path | None = None
    mace_potential: Path | None = None
    pbs_queue: str | None = None
    pbs_project: str | None = None
    pbs_walltime_h: float | None = None
    pbs_gpus: int | None = None
    pbs_ncpus: int | None = None
    pbs_mem: str | None = None

    @classmethod
    def load(cls) -> "Site":
        """Every fact as declared now (environment, then ``[site]`` in ``aipf.toml``); undeclared ones are ``None``."""
        return cls(**{f.name: _read(f.name) for f in fields(cls)})

    def for_system(self, name: str, system: str):
        """Fact ``name`` for ``system``: ``<VARIABLE>_<SYSTEM>``, then ``[site.<name>] <system>``, then the
        fact itself; :class:`MissingSiteFact` naming all three when none is declared."""
        if name not in PER_SYSTEM:
            raise ValueError(f"site fact {name!r} is not declared per system; those are {PER_SYSTEM}")
        found = paths._lookup(per_system_variable(name, system), f"site.{name}", system)
        if found is None:
            found = getattr(self, name)
        if found is None:
            raise MissingSiteFact((name,), f"missing site fact {name} for system {system!r}: set "
                                           f"{per_system_variable(name, system)}, or {system} = "
                                           f"\"/path\" under [site.{name}] in {paths._where_config()}, "
                                           f"or {FACTS[name]} for every system")
        return found

    def require(self, *names: str) -> "Site":
        """This site, or :class:`MissingSiteFact` listing every one of ``names`` that is undeclared."""
        unknown = [n for n in names if n not in FACTS]
        if unknown:
            raise ValueError(f"unknown site fact(s) {unknown}; the facts are {sorted(FACTS)}")
        missing = tuple(n for n in names if getattr(self, n) is None)
        if missing:
            where = paths._where_config()
            lines = [f"  {n}: set {FACTS[n]}, or {n} = {_EXAMPLE.get(n, _A_FILE)} under [site] in {where}"
                     for n in missing]
            raise MissingSiteFact(missing, "missing site facts:\n" + "\n".join(lines))
        return self
