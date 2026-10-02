"""The decks that ran, keyed by what they are for.

A deck nobody ran is not a template. Every text below was derived from an
archived file that produced published data, and each one names that file and
that data in its ``source`` and ``produced``. Six texts, seven pairings: two
names in the frozen geometry vocabulary were run from a single archived file,
and the registry says so rather than carrying a second copy.

What the archive actually varies
--------------------------------
Reading the nine archived decks and workers side by side, the thing that
distinguishes them is NOT the geometry. It is the ensemble, the output
contract, and whether an external field acts. The geometry lives in the
initial configuration, and in both heavy-element trees that configuration is
built by a separate program and read from a file -- one worker serves a
one-axis barostat and an isotropic one with the same text, and decides which
by the ensemble, not by the shape of the box. So ``@CONFIGURATION@`` is a
hole, and the decks below differ where the archive differs.

Two decks that were deliberately NOT built
------------------------------------------
A Monte-Carlo species-exchange protocol: it ran, and its own notes call its
dynamics non-physical and admit it only for a phase-diagram check and as a
source of starting configurations. Its exchange move is not an ensemble the
frozen vocabulary can name, and the engine build its own pipeline
configuration points at does not carry the package that provides it.

An equation-of-state grid deck in the reduced-unit tree: nothing in that tree
invokes it, and no directory of its output exists. It is the rule's own
example.
"""
from __future__ import annotations

from aipf.md.request import StatePoint, normalise_ensemble, normalise_geometry
from aipf.md.templates.deck import Deck, DeckError

__all__ = ["DECKS", "available", "builtin", "for_point"]


# ---------------------------------------------------------------------------
# The heavy-element route: one barostatted box, one trajectory
# ---------------------------------------------------------------------------

#: Preparation, then production, with the trajectory opened at whichever of
#: the two the request asks for. The stage order is the archived worker's and
#: its own docstring calls it load-bearing: the clock is reset exactly once,
#: with the dump, so that a trajectory's step numbering means what the
#: campaign that reads it assumes.
#:
#: The two heavy-element trees ran this deck with different outputs beside the
#: trajectory, so those are holes the deck's defaults fill with the first
#: tree's: the thermo columns (the second tree printed ``density`` where the
#: first printed the box edges) and a production sampling block, which the
#: first tree did not have and the second wrote to its own file (a time
#: average of the thermo quantities, opened with production and closed after
#: it). A system whose archive is the second tree declares both.
_CUBE_NPT = """\
# Homogeneous box under an isotropic barostat.
#
# Stage order is load-bearing: the clock is reset once, together with the
# dump, so a trajectory written from production is numbered in production
# steps and one written from the preparation covers the whole run.
units           @UNITS@
boundary        p p p
atom_style      atomic
atom_modify     map yes
@CONFIGURATION@
@MASSES@
@PAIR@
neighbor        @SKIN@ bin
neigh_modify    @NEIGH_MODIFY@
@RELAX@
timestep        @DT@
thermo          @N_THERMO@
thermo_style    custom @THERMO_COLUMNS@
thermo_modify   flush yes
velocity        all create @T_PREP@ @SEED@ mom yes rot yes dist gaussian
@DUMP_EQUIL@
fix             prep all npt temp @T_PREP@ @T_PREP_END@ @TDAMP@ &
                iso @P@ @P@ @PDAMP@
fix             recentre all recenter NULL NULL INIT
run             @N_EQUIL@
unfix           prep
@DUMP_PROD@
@SAMPLE_PROD@
fix             prod all npt temp @T@ @T@ @TDAMP@ iso @P@ @P@ @PDAMP@
run             @N_PROD@
unfix           prod
@SAMPLE_PROD_END@
undump          traj
@WRITE_END@
"""

#: The same worker, its other ensemble. Two things change and both are the
#: archive's: the barostat acts on the normal axis alone, so the lateral box
#: is fixed and the normal pressure is the one controlled, and there is no
#: over-temperature preparation, because the worker ties that to the
#: ISOTROPIC barostat -- its comment says the ramp exists to melt a lattice,
#: and this path starts from a configuration that is already liquid.
_SLAB_NPT_Z = """\
# Interfacial box, barostat on the normal axis only.
#
# No over-temperature preparation: the archived worker ties its ramp to the
# isotropic barostat, not to the geometry, because the ramp exists to melt a
# lattice and this path starts from a liquid configuration.
units           @UNITS@
boundary        p p p
atom_style      atomic
atom_modify     map yes
@CONFIGURATION@
@MASSES@
@PAIR@
neighbor        @SKIN@ bin
neigh_modify    @NEIGH_MODIFY@
@RELAX@
timestep        @DT@
thermo          @N_THERMO@
thermo_style    custom step time temp pe press vol lx ly lz
thermo_modify   flush yes
velocity        all create @T@ @SEED@ mom yes rot yes dist gaussian
@DUMP_EQUIL@
fix             prep all npt temp @T@ @T@ @TDAMP@ z @P@ @P@ @PDAMP@
fix             recentre all recenter NULL NULL INIT
run             @N_EQUIL@
unfix           prep
@DUMP_PROD@
fix             prod all npt temp @T@ @T@ @TDAMP@ z @P@ @P@ @PDAMP@
run             @N_PROD@
unfix           prod
undump          traj
@WRITE_END@
"""

#: No trajectory at all. The point reports time-averaged thermodynamics and
#: one row, which is why this deck has no dump to fill and refuses a request
#: that asks for frames.
_EOS_NPT = """\
# Equation-of-state point: a box whose volume is averaged, not dumped.
#
# The averaging window is sampled on the thermo cadence, which is what sets
# how many samples the point reports.
units           @UNITS@
boundary        p p p
atom_style      atomic
atom_modify     map yes
@CONFIGURATION@
@MASSES@
@PAIR@
neighbor        @SKIN@ bin
neigh_modify    @NEIGH_MODIFY@
@RELAX@
timestep        @DT@
thermo          @N_THERMO@
thermo_style    custom step time temp pe press vol lx ly lz
thermo_modify   flush yes
velocity        all create @T_PREP@ @SEED@ mom yes rot yes dist gaussian
fix             prep all npt temp @T_PREP@ @T_PREP_END@ @TDAMP@ &
                iso @P@ @P@ @PDAMP@
run             @N_EQUIL@
unfix           prep
reset_timestep  0
variable        sample_temp equal temp
variable        sample_press equal press
variable        sample_vol equal vol
fix             sample all ave/time @N_THERMO@ 1 @N_THERMO@ &
                v_sample_temp v_sample_press v_sample_vol file @THERMO_FILE@
fix             prod all npt temp @T@ @T@ @TDAMP@ iso @P@ @P@ @PDAMP@
run             @N_PROD@
unfix           prod
unfix           sample
@WRITE_END@
"""


# ---------------------------------------------------------------------------
# The reduced-unit route: an inertial preparation, overdamped production
# ---------------------------------------------------------------------------

#: Two archived decks, one text. They differ in the stage-one temperature and
#: in nothing else that runs: one equilibrates hot and quenches, the other
#: equilibrates at its own target because it sits above its critical point.
#: That difference is exactly what a system's preparation declares, which is
#: why one deck covers both.
#:
#: Stage two has no inertia, so the timestep changes with it; the two are
#: integrated differently and the request's timestep is the production one.
_CUBE_OVERDAMPED = """\
# Homogeneous box: inertial preparation, then overdamped production.
#
# Stage one is a thermostatted inertial run at its own timestep, and its
# temperature is the system's preparation temperature -- hot for a quench,
# the target itself for a system that declares no melt. Stage two has no
# inertia and runs at the request's own timestep, and the clock is reset
# between them so the trajectory is numbered in production steps.
units           @UNITS@
atom_style      atomic
dimension       3
boundary        p p p
@CONFIGURATION@
@MASSES@
@PAIR@
neighbor        @SKIN@ bin
neigh_modify    @NEIGH_MODIFY@
@RELAX@
thermo          @N_THERMO@
thermo_style    custom step temp pe ke etotal press
timestep        @DT_EQUIL@
velocity        all create @T_PREP@ @SEED@ dist gaussian
fix             prep_nve all nve
fix             prep all langevin @T_PREP@ @T_PREP_END@ @DAMP@ @SEED@
run             @N_EQUIL@
unfix           prep
unfix           prep_nve
reset_timestep  0
timestep        @DT@
fix             prod all brownian @T@ @SEED@ gamma_t @GAMMA@
fix             recentre all recenter INIT INIT INIT
@DUMP_PROD@
run             @N_PROD@
@WRITE_END@
"""

#: A continuation. It reads a configuration taken from another run and starts
#: integrating, with no preparation of any kind: the archived deck's own
#: header says so on purpose, because the smoothing window that consumes the
#: trajectory discards more than the transient lasts.
_SLAB_OVERDAMPED = """\
# Overdamped production from a configuration another run left behind.
#
# No preparation stage at all, on purpose: the archived deck's header records
# that the downstream smoothing window pads away far more than the switch
# transient lasts, so equilibrating again would only cost trajectory.
units           @UNITS@
atom_style      atomic
dimension       3
boundary        p p p
@CONFIGURATION@
@MASSES@
@PAIR@
neighbor        @SKIN@ bin
neigh_modify    @NEIGH_MODIFY@
timestep        @DT@
fix             prod all brownian @T@ @SEED@ gamma_t @GAMMA@
fix             recentre all recenter INIT INIT INIT
thermo          @N_THERMO@
thermo_style    custom step temp pe
@DUMP_PROD@
run             @N_PROD@
@WRITE_END@
"""

#: An external field, held through both runs. The field itself is a hole:
#: its shape, its amplitude and its periodicity are declared nowhere but in
#: the deck that applies it, so they are neither the request's nor the
#: system's, and the module fills everything around them instead.
_VEXT_NVT = """\
# Equilibrium under an external field.
#
# The thermostat is declared once and held across both runs: only the dump
# appears between them, so that the equilibration and the production are one
# continuous trajectory of the same dynamics and the field never restarts.
# The field is the caller's -- nothing but a deck ever recorded its shape.
units           @UNITS@
atom_style      atomic
dimension       3
boundary        p p p
@CONFIGURATION@
@MASSES@
@PAIR@
neighbor        @SKIN@ bin
neigh_modify    @NEIGH_MODIFY@
@RELAX@
timestep        @DT@
thermo          @N_THERMO@
thermo_style    custom step temp pe etotal press
reset_timestep  0
velocity        all scale @T@
@EXTERNAL@
fix             prep_nve all nve
fix             prep all langevin @T@ @T@ @DAMP@ @SEED@
fix             recentre all recenter INIT INIT INIT
run             @N_EQUIL@
@DUMP_PROD@
run             @N_PROD@
@WRITE_END@
"""


# ---------------------------------------------------------------------------
# What each one came from, and what that produced
# ---------------------------------------------------------------------------

#: The two lines that open a trajectory in the archived heavy-element worker.
#: It flushes every frame, so a run killed by a wall clock leaves a
#: trajectory that can still be read to its last complete frame.
_HEAVY_DUMP = {
    "DUMP_RESET": "reset_timestep  0",
    "DUMP_MODIFY": "dump_modify     traj sort id flush yes",
}

#: The reduced-unit decks reset their own clock between stages, so the block
#: does not, and they force a frame at the first step of the run.
_REDUCED_DUMP = {
    "DUMP_RESET": "",
    "DUMP_MODIFY": "dump_modify     traj sort id first yes",
}

#: The external-field deck sorts and nothing else.
_FIELD_DUMP = {
    "DUMP_RESET": "",
    "DUMP_MODIFY": "dump_modify     traj sort id",
}

#: The first heavy-element tree's outputs beside the cube trajectory: the box
#: edges in the thermo line, and no sampling file.
_CUBE_NPT_OUTPUTS = {
    "THERMO_COLUMNS": "step time temp pe press vol lx ly lz",
    "SAMPLE_PROD": "",
    "SAMPLE_PROD_END": "",
}

_CUBE_NPT_DECK = Deck(
    name="cube-npt",
    text=_CUBE_NPT,
    source=("Data/slab_data/run_slab_prod.py (first heavy-element tree), "
            "ENSEMBLE=npt_iso, cross-read against Data/cube/run_cube.py "
            "(second heavy-element tree), whose melt stage is the same "
            "shape with a held rather than a ramped temperature"),
    produced=("the barostatted-cube trajectory sets on both raw data trees: "
              "cube_data and its four per-pressure siblings in the first, "
              "cube_data and its three in the second"),
    geometry="cube",
    ensemble="NPT",
    defaults={**_HEAVY_DUMP, **_CUBE_NPT_OUTPUTS},
)

_SLAB_NPT_Z_DECK = Deck(
    name="slab-npt-z",
    text=_SLAB_NPT_Z,
    source="Data/slab_data/run_slab_prod.py (first heavy-element tree), "
           "ENSEMBLE=npt_z",
    produced="the 88 interfacial runs in slab_data and its four "
             "per-pressure siblings on the first raw data tree",
    geometry="slab",
    ensemble="NPT_z",
    defaults=_HEAVY_DUMP,
)

_EOS_NPT_DECK = Deck(
    name="eos-npt",
    text=_EOS_NPT,
    source=("Data/eos/run_eos_point.py in both heavy-element trees: the "
            "preparation ramp and the stage order from the first, the "
            "sampling written to a file from the second, whose worker runs "
            "a deck rather than driving the engine in process and so cannot "
            "read its thermodynamics back one chunk at a time"),
    produced=("the equation-of-state grids on both raw data trees: four "
              "pressure directories in the first, five in the second"),
    geometry="eos",
    ensemble="NPT",
)

_CUBE_OVERDAMPED_DECK = Deck(
    name="cube-overdamped",
    text=_CUBE_OVERDAMPED,
    source=("Data/input/in.quench.lammps and "
            "Data/input/in.homogeneous_brownian.lammps (reduced-unit tree). "
            "Stripped of comments the two differ only in the stage-one "
            "temperature and in housekeeping, which is why one text serves "
            "both"),
    produced=("the in-dome training trees and the above-critical "
              "fluctuation trees: Data/spinodal_cube at three temperatures "
              "and Data/homogeneous_brownian at four"),
    geometry="cube",
    ensemble="langevin_overdamped",
    defaults=_REDUCED_DUMP,
)

#: The same file again, under the other name the frozen vocabulary gives it.
#: The reduced-unit tree ran one deck for its coarsening-validation tree and
#: its in-dome training tree, and kept them in separate directories with
#: separate seed streams so that no glob could pool them.
_QUENCH_DECK = Deck(
    name="quench-overdamped",
    text=_CUBE_OVERDAMPED,
    source=_CUBE_OVERDAMPED_DECK.source,
    produced=("Data/quench at five temperatures, the coarsening ground "
              "truth, deliberately kept out of training"),
    geometry="quench",
    ensemble="langevin_overdamped",
    defaults=_REDUCED_DUMP,
)

_SLAB_OVERDAMPED_DECK = Deck(
    name="slab-overdamped",
    text=_SLAB_OVERDAMPED,
    source="Data/input/in.slab_overdamped.lammps (reduced-unit tree)",
    produced=("Data/slab_overdamped, one long trajectory per temperature "
              "and seed -- the dynamics this package exists to learn"),
    geometry="slab",
    ensemble="langevin_overdamped",
    defaults=_REDUCED_DUMP,
)

_VEXT_NVT_DECK = Deck(
    name="vext-nvt",
    text=_VEXT_NVT,
    source=("Data/input/in.vext_checkerboard.lammps and "
            "Data/input/in.vext_hexagonal.lammps (reduced-unit tree), which "
            "differ only in the field they apply"),
    produced="the four external-field equilibrium trees, two profiles at "
             "two temperatures each",
    geometry="vext",
    ensemble="NVT",
    defaults=_FIELD_DUMP,
)


#: Every pairing that ran, and the deck it ran. Keyed by the canonical names,
#: so a lookup goes through the same normalisation a request does and the
#: archived spellings reach the same entry.
DECKS: dict[tuple[str, str], Deck] = {
    ("cube", "NPT"): _CUBE_NPT_DECK,
    ("slab", "NPT_z"): _SLAB_NPT_Z_DECK,
    ("eos", "NPT"): _EOS_NPT_DECK,
    ("cube", "langevin_overdamped"): _CUBE_OVERDAMPED_DECK,
    ("quench", "langevin_overdamped"): _QUENCH_DECK,
    ("slab", "langevin_overdamped"): _SLAB_OVERDAMPED_DECK,
    ("vext", "NVT"): _VEXT_NVT_DECK,
}


def available() -> list[tuple[str, str]]:
    """Every pairing that has a deck, sorted."""
    return sorted(DECKS)


def builtin(geometry: str, ensemble: str) -> Deck:
    """The deck for a geometry and an ensemble, or a refusal listing what ran.

    Both names go through the request module's normalisation first, so the
    spellings the archive actually wrote reach the same entry as the
    canonical ones. There is one vocabulary, not two.
    """
    key = (normalise_geometry(geometry), normalise_ensemble(ensemble))
    try:
        return DECKS[key]
    except KeyError:
        raise DeckError(
            f"no deck for geometry {geometry!r} and ensemble {ensemble!r} "
            f"(read as {key}). Decks exist for {available()}. A pairing that "
            f"is missing is one nothing in the archive ran; bring your own "
            f"deck for it rather than adopting a neighbour's") from None


def for_point(point: StatePoint) -> Deck:
    """The deck a request asks for, chosen from the request alone."""
    return builtin(point.geometry, point.ensemble)
