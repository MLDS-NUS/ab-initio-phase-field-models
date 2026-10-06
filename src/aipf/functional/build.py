"""The front door: build the free-energy functional a system declares.
A translation from the checkpoint's saved ``model_kwargs`` vocabulary to the rung constructor,
agreeing key for key with :func:`aipf.train.checkpoint_formats.lightning_hparams_rung_kwargs`. It never
guesses, swallows a key, or renames behind the caller. ``nonlocal_kernel`` and the closed local forms of
``square_gradient`` are translated; ``variant=`` builds a declared :class:`aipf.system.Variant`. A functional
declared with a ``factory`` is built by calling it, and the model it returns is checked against the contract
the drivers read (:func:`check_factory_model`)."""
from __future__ import annotations

import functools
import inspect
from typing import Any, Dict

import torch

from aipf.mobility import mobility_kwargs
from aipf.spectral import OPS_CLASSES, OpsCache

from .base import MODEL_REGISTRY, PROTOCOL_METHODS

#: Declared name -> constructor name, for keys whose spelling differs (pure renames).
_RENAMES = {
    "h_w": "kernel_hidden",      # the pair kernel's radial-net width
    "g_form": "g_exc_form",      # the excess-entropy net's form
}

#: Mobility-owned keys saved beside the rest; read by `mobility_kwargs`, not emitted here.
_MOBILITY_OWNED = ("h_m", "T_ref")

#: Declared flags with no constructor argument: the value at which each says nothing.
_INERT_ONLY = {
    "disable_u": (False, "switches the energy net off entirely; this "
                         "package builds no such variant"),
    "disable_g": (False, "switches the excess-entropy net off entirely; "
                         "this package builds no such variant"),
}

#: Overrides spelled the declaration's way, applied before the translation.
_DECLARED_OVERRIDES = frozenset(_RENAMES) | frozenset(_INERT_ONLY) | {
    "fexc_T_ref", "activation", "tbasis_ortho", *_MOBILITY_OWNED}

# Which constructor parameters the declaration must carry: this door never uses a signature default.

#: Parameters computed from the system object, not from `Functional.kwargs`.
_BUILDER_SUPPLIED = frozenset({
    "n_species",        # System.n_species
    "kB",               # System.constants["kB"]
    "u_form",           # Functional.local
    "mobility_shape",   # Mobility.form, through MOBILITY_SHAPES
    "mobility_t_form",  # Mobility.T_form
})

#: Parameters a declaration may omit, and why each is safe.
_DECLARED_OPTIONAL = {
    "u_degree": "read only by u_form='taylor'; required for that form by "
                "_REQUIRED_WHEN below",
    "u_parity": "read only by u_form='taylor'; required for that form by "
                "_REQUIRED_WHEN below",
    "h_joint": "read only by f_exc_form='joint'; required for that form by "
               "_REQUIRED_WHEN below",
    "joint_depth": "read only by f_exc_form='joint'; required for that form "
                   "by _REQUIRED_WHEN below",
    "mobility_input_ref": "read only by the MLP mobility shapes, and "
                          "mobility_kwargs refuses an MLP shape that does "
                          "not declare it (None for raw rho)",
    "kernel_tail_sigma": "the envelope's optional tail; None IS the shape, "
                         "not an unset value",
    "kernel_evaluator": "the escape hatch for an experiment supplying its "
                        "own evaluator; None means this rung builds its own",
    "mobility_shape_init": "required only by the 'fixed', 'constant' and "
                           "'lattice_scalar' shapes, and Mobility raises for "
                           "those when it is None",
    "mobility_activation_energy_init": "required only by t_form="
                                       "'arrhenius'; required for that "
                                       "form by _REQUIRED_WHEN below",
    "tbasis_ortho_window": "required only when the T-basis orthogonalisation "
                           "is on, which the `tbasis_ortho` switch above "
                           "already refuses to leave implicit",
    "tbasis_ortho_points": "as tbasis_ortho_window",
    "u_variable": "read only by u_form='taylor'; required for that form by "
                  "_REQUIRED_WHEN below",
    "g_symmetry": "admissible as 'mirror' only under ideal_form='lattice'; "
                  "required for that form by _REQUIRED_WHEN below",
    "kernel_argument": "required under ideal_form='lattice' by _REQUIRED_WHEN "
                       "below",
    "icnn_output_bias": "read only when an ICNN is built; required then by "
                        "_REQUIRED_IF below",
    "kernel_n_k_table": "read only by the k-table evaluator (kernel_evaluator "
                        "None), and NonlocalKernel raises for it when absent",
    "kernel_k_table_max": "as kernel_n_k_table",
    "kernel_n_quad": "read only by the k-table evaluator and by kappa_eff; required "
                     "for that evaluator by _REQUIRED_IF below, and kappa_eff refuses "
                     "without it",
    "mobility_hidden": "the width of a mobility net; required for the MLP shapes by "
                       "_REQUIRED_IF below, read by no other shape",
    "h_u": "the width of the split form's icnn or mlp energy net; required for those by "
           "_REQUIRED_IF below, read by no other form",
    "kBT_ref": "read only by the joint form, the T-basis heads and their orthogonalisation; "
               "required for those by _REQUIRED_IF below",
}

#: Optional parameters that become required once another argument selects their variant.
_REQUIRED_WHEN = {
    "u_degree": ("u_form", "taylor"),
    "u_parity": ("u_form", "taylor"),
    "h_joint": ("f_exc_form", "joint"),
    "joint_depth": ("f_exc_form", "joint"),
    "mobility_activation_energy_init": ("mobility_t_form", "arrhenius"),
    "u_variable": ("u_form", "taylor"),
    "g_symmetry": ("ideal_form", "lattice"),
    "kernel_argument": ("ideal_form", "lattice"),
}


def _builds_icnn(out: Dict[str, Any]) -> bool:
    """The split form with an ICNN energy or entropy net."""
    return out.get("f_exc_form") == "split" and "icnn" in (
        out.get("u_form"), out.get("g_exc_form"))


def _builds_k_table(out: Dict[str, Any]) -> bool:
    """The rung builds its own k-table evaluator, which integrates on ``kernel_n_quad`` points."""
    return out.get("kernel_evaluator") is None


def _builds_mobility_net(out: Dict[str, Any]) -> bool:
    """The mobility shape is a net of width ``mobility_hidden``."""
    return out.get("mobility_shape") in ("mlp_rho", "mlp_rho_t")


def _builds_energy_net(out: Dict[str, Any]) -> bool:
    """The split form with an ``icnn`` or ``mlp`` energy net of width ``h_u``."""
    return out.get("f_exc_form") == "split" and out.get("u_form") in ("icnn", "mlp")


def _reads_kbt_ref(out: Dict[str, Any]) -> bool:
    """A local form that divides ``kBT`` by ``kBT_ref``."""
    return (out.get("f_exc_form") == "joint" or bool(out.get("enable_TlnT"))
            or bool(out.get("enable_T2")) or out.get("tbasis_ortho_window") is not None)


#: Optional parameters that become required when a predicate on the arguments holds.
_REQUIRED_IF = {"icnn_output_bias": _builds_icnn,
                "kernel_n_quad": _builds_k_table,
                "mobility_hidden": _builds_mobility_net,
                "h_u": _builds_energy_net,
                "kBT_ref": _reads_kbt_ref}

#: Constructor parameter -> its declared spelling, for the refusal message.
_DECLARED_SPELLING = {
    "kernel_hidden": "h_w",
    "mobility_hidden": "h_m",
    "g_exc_form": "g_form",
    "mobility_t_ref": "T_ref",
    "kBT_ref": "fexc_T_ref",
    "local_activation": "activation",
    "kernel_activation": "activation",
    "mobility_activation": "activation",
}


#: The same four tables for ``square_gradient``'s closed local forms.
_SG_BUILDER_SUPPLIED = frozenset({
    "n_species",        # System.n_species
    "kB",               # System.constants["kB"], for the Arrhenius factor
    "local",            # Functional.local
    "mobility_shape",   # Mobility.form, through MOBILITY_SHAPES
    "mobility_t_form",  # Mobility.T_form
})
_SG_DECLARED_OPTIONAL = {
    "w": "read only by local='flory_huggins'; required for it by _SG_REQUIRED_WHEN",
    "a0": "read only by local='landau'; required for it by _SG_REQUIRED_WHEN",
    "b": "as a0",
    "T_c": "as a0",
    "mobility_shape_init": "required by every shape the closed forms compose, and Mobility "
                           "raises when it is None",
    "mobility_t_ref": "read only by t_form='arrhenius'; required for it by _SG_REQUIRED_WHEN",
    "mobility_activation_energy_init": "as mobility_t_ref",
}
_SG_REQUIRED_WHEN = {
    "w": ("local", "flory_huggins"),
    "a0": ("local", "landau"),
    "b": ("local", "landau"),
    "T_c": ("local", "landau"),
    "mobility_t_ref": ("mobility_t_form", "arrhenius"),
    "mobility_activation_energy_init": ("mobility_t_form", "arrhenius"),
}
_SG_DECLARED_SPELLING = {"kappa_init": "kappa", "mobility_t_ref": "T_ref"}

#: Form -> (builder-supplied, declared-optional, required-when, required-if, declared spelling).
_RULES = {
    "nonlocal_kernel": (_BUILDER_SUPPLIED, _DECLARED_OPTIONAL, _REQUIRED_WHEN, _REQUIRED_IF,
                        _DECLARED_SPELLING),
    "square_gradient": (_SG_BUILDER_SUPPLIED, _SG_DECLARED_OPTIONAL, _SG_REQUIRED_WHEN, {},
                        _SG_DECLARED_SPELLING),
}


def _check_every_parameter_is_declared(cls, out: Dict[str, Any], form: str) -> None:
    """Refuse a build that would fall through to a constructor default."""
    supplied, optional, required_when, required_if, spelling = _RULES[form]
    parameters = set(inspect.signature(cls.__init__).parameters) - {"self"}
    required = parameters - supplied - set(optional)
    missing = required - set(out)
    for parameter, (key, value) in required_when.items():
        if out.get(key) == value and parameter not in out:
            missing.add(parameter)
    for parameter, holds in required_if.items():
        if holds(out) and parameter not in out:
            missing.add(parameter)
    if not missing:
        return
    named = ", ".join(
        f"{p} (declare it as {spelling[p]!r})"
        if p in spelling else p
        for p in sorted(missing))
    raise ValueError(
        f"the declaration supplies no value for {named}. "
        f"{cls.__name__} either has a default for it, so building anyway "
        f"would make a model that differs from the declared one and says so "
        f"nowhere, or none at all. Every value a model is built from is "
        f"declared or overridden, never defaulted here")


#: Pair-kernel forms rung 3 builds; any other declared kernel is refused.
_KERNEL_FORMS = {"nonlocal_kernel": ("radial_mlp",)}


def rung_kwargs(system, variant: str | None = None, **overrides: Any) -> Dict[str, Any]:
    """The rung constructor's arguments, as ``system`` (or its declared ``variant``) declares them, with
    ``overrides`` applied."""
    if variant is not None:
        system = system.variant(variant)
    if system.functional is None:
        raise ValueError(
            f"system {system.name!r} declares no functional, so there is "
            f"nothing to build; what model a system trains is a fact about "
            f"that system and this package ships no default for it")
    _register_rungs()
    spec = system.functional
    if getattr(spec, "factory", None) is not None:
        raise ValueError(
            f"system {system.name!r} declares its functional by a factory, "
            f"{factory_name(spec.factory)}, which is called with the system; there are no "
            f"constructor arguments to translate. Use build()")
    if spec.form not in _TRANSLATIONS:
        raise NotImplementedError(
            f"no declaration-to-constructor translation is measured for "
            f"functional form {spec.form!r}; only "
            f"{sorted(_TRANSLATIONS)} has been built from a real "
            f"checkpoint and checked against it")

    spelled = _OVERRIDES_SPELLED_AS_DECLARED[spec.form]
    declared = dict(spec.kwargs)
    declared.update({k: v for k, v in overrides.items() if k in spelled})
    out = _TRANSLATIONS[spec.form](system, declared)

    cls = MODEL_REGISTRY[spec.form]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    rest = {k: v for k, v in overrides.items() if k not in spelled}
    unknown = sorted(set(rest) - accepted)
    if unknown:
        raise TypeError(
            f"{cls.__name__} accepts no argument named {unknown}, and "
            f"neither does the declaration; refusing rather than ignoring "
            f"it, because an ignored override builds the model you did not "
            f"ask for")
    out.update(rest)
    _check_every_parameter_is_declared(cls, out, spec.form)
    return out


def build(system, variant: str | None = None, **overrides: Any) -> torch.nn.Module:
    """Construct ``system.functional`` (or variant ``variant``'s) with ``overrides`` applied last, recorded as
    ``build_overrides``. A declared ``factory`` is called as ``factory(system, **overrides)`` instead, and
    what it returns is checked (:func:`check_factory_model`)."""
    if variant is not None:
        system = system.variant(variant)
    if getattr(system.functional, "factory", None) is not None:
        model = system.functional.factory(system, **overrides)
        check_factory_model(model, system)
        model.build_overrides = dict(overrides)
        return model
    kwargs = rung_kwargs(system, **overrides)
    model = MODEL_REGISTRY[system.functional.form](**kwargs)
    model.build_overrides = dict(overrides)
    return model


def factory_name(factory) -> str:
    """``module:qualname`` of a factory, as a run's manifest records it; a ``functools.partial`` names the
    function it wraps, a callable object its class."""
    while isinstance(factory, functools.partial):
        factory = factory.func
    module = getattr(factory, "__module__", None) or type(factory).__module__
    qualname = getattr(factory, "__qualname__", None) or type(factory).__qualname__
    return f"{module}:{qualname}"


def check_factory_model(model, system) -> None:
    """Refuse what ``system.functional.factory`` returned unless training, the checkpoint reload and the
    explicit solvers can use it: an ``nn.Module`` with the ``FreeEnergyModel`` methods, at least one
    parameter, ``_cache`` (an :class:`aipf.spectral.OpsCache`) and ``ops`` (its
    :class:`aipf.spectral.SpectralOps`, or :class:`aipf.spectral.SpectralOps2D` on a two-axis grid, the
    same object as ``_cache.ops``) on the declared ``grid`` and under the declared ``nyquist_mask``.
    The semi-implicit scheme's own needs are checked when it runs."""
    name = factory_name(system.functional.factory)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"the factory {name} returned a {type(model).__name__}, not a torch.nn.Module; "
            f"training wraps the model in a LightningModule and the checkpoint holds its "
            f"state_dict")
    missing = [m for m in PROTOCOL_METHODS if not callable(getattr(model, m, None))]
    if missing:
        raise TypeError(
            f"the factory {name} returned a {type(model).__name__}, which does not implement "
            f"FreeEnergyModel: missing {missing}")
    if next(model.parameters(), None) is None:
        raise ValueError(
            f"the factory {name} returned a {type(model).__name__} with no parameters: there "
            f"is nothing to train and nothing for a checkpoint to hold")
    if not isinstance(getattr(model, "_cache", None), OpsCache):
        raise TypeError(
            f"the factory {name} returned a {type(model).__name__} whose _cache is not an "
            f"aipf.spectral.OpsCache; training reads the spectral operators per grid from it")
    if not isinstance(getattr(model, "ops", None), OPS_CLASSES):
        raise TypeError(
            f"the factory {name} returned a {type(model).__name__} whose ops is not an "
            f"aipf.spectral.SpectralOps (the OpsCache's own, _cache.ops); the explicit solvers "
            f"read the Nyquist convention from it")
    if model.ops is not model._cache.ops:
        raise TypeError(
            f"the factory {name} returned a {type(model).__name__} whose ops is not its "
            f"_cache.ops; training reads the operators from the one and the solvers from the "
            f"other, so they are one object (self.ops = self._cache.ops)")
    grid = tuple(int(g) for g in system.functional.kwargs["grid"])
    if tuple(model._cache.grid) != grid:
        raise ValueError(
            f"the factory {name} built its operators on grid {tuple(model._cache.grid)} and the "
            f"declaration says {grid}; training reads the one and the model the other")
    declared = bool(system.functional.kwargs["nyquist_mask"])
    if bool(model.ops.nyquist_mask) != declared:
        raise ValueError(
            f"the factory {name} built ops with nyquist_mask={model.ops.nyquist_mask!r} and "
            f"the declaration says {declared!r}; the drivers read one and the model the "
            f"other")


#: Functional form -> the module that implements it and registers it (``aipf.system.FUNCTIONAL_FORMS``).
_FORM_MODULES = {
    "landau": "aipf.functional.landau",
    "square_gradient": "aipf.functional.square_gradient",
    "nonlocal_kernel": "aipf.functional.nonlocal_kernel",
    "neural_operator": "aipf.functional.neural_operator",
}


def _register_rungs() -> None:
    """Import every form's module so the shared registry holds them (on the call, not at import)."""
    import importlib

    for module in _FORM_MODULES.values():
        importlib.import_module(module)


def _nonlocal_kernel(system, declared: Dict[str, Any]) -> Dict[str, Any]:
    """Rung 3's constructor arguments from its declaration: renames, named conversions, named refusals."""
    spec = system.functional
    kernel_forms = _KERNEL_FORMS["nonlocal_kernel"]
    if spec.kernel not in kernel_forms:
        raise NotImplementedError(
            f"declared kernel form {spec.kernel!r} is not one this rung "
            f"builds; it builds {list(kernel_forms)}, and ignoring the name "
            f"would build that one under another")

    # Read the mobility's arguments first, from the declaration with overrides applied.
    mobility = mobility_kwargs(system, functional_kwargs=declared)

    out: Dict[str, Any] = {"n_species": int(system.n_species)}

    # kB is a system constant (unit convention), never a core value.
    try:
        out["kB"] = system.constants["kB"]
    except KeyError:
        raise KeyError(
            f"system {system.name!r} declares no constant 'kB', so a "
            f"declared reference temperature cannot be turned into the "
            f"reference energy this rung's local form takes; kB is that "
            f"system's own unit convention and core holds no value for it"
        ) from None

    # Declared reference temperature -> the constructor's reference energy.
    if "fexc_T_ref" in declared:
        out["kBT_ref"] = out["kB"] * declared.pop("fexc_T_ref")

    for declared_name, arg in _RENAMES.items():
        if declared_name in declared:
            out[arg] = declared.pop(declared_name)

    # One declared activation fans out to the local and kernel nets.
    if "activation" in declared:
        activation = declared.pop("activation")
        out["local_activation"] = activation
        out["kernel_activation"] = activation

    # `Functional.local` is the local form; a copy in kwargs must agree.
    if declared.get("u_form", spec.local) != spec.local:
        raise ValueError(
            f"the declaration disagrees with itself: local={spec.local!r} "
            f"but kwargs says u_form={declared['u_form']!r}")
    declared.pop("u_form", None)
    out["u_form"] = spec.local

    # `tbasis_ortho` decides whether the window is forwarded; ambiguity is refused.
    if "tbasis_ortho" not in declared and "tbasis_ortho_window" in declared:
        raise ValueError(
            "tbasis_ortho_window is declared but tbasis_ortho is not, so "
            "there is nothing that says whether the orthogonalisation is "
            "on; declare the switch rather than letting the window be "
            "dropped")
    if declared.pop("tbasis_ortho", False):
        absent = [k for k in ("tbasis_ortho_window", "tbasis_ortho_points")
                  if k not in declared]
        if absent:
            raise ValueError(
                f"tbasis_ortho is on but {', '.join(absent)} is not "
                f"declared; this package switches the orthogonalisation on "
                f"by the window's presence and fits it over the declared "
                f"points, so without them the model is built with the "
                f"orthogonalisation off or over somebody else's window, in "
                f"silence")
    else:
        declared.pop("tbasis_ortho_window", None)
        declared.pop("tbasis_ortho_points", None)

    for name, (inert, meaning) in _INERT_ONLY.items():
        if name in declared and declared.pop(name) != inert:
            raise NotImplementedError(
                f"{name} is declared at a value this package cannot build: "
                f"it {meaning}. Dropping it would build the variant the run "
                f"did not use")

    # Mobility-owned keys, already read into `mobility` above.
    for name in _MOBILITY_OWNED:
        declared.pop(name, None)

    # The mobility, under the rung's `mobility_` prefix.
    # `n_species`, `kB` and `rho_eps` are the rung's own, which it hands its mobility itself.
    for arg, value in mobility.items():
        if arg not in ("n_species", "kB", "rho_eps"):
            out[f"mobility_{arg}"] = value

    # Everything else keeps its own name; the constructor accepts or refuses it.
    out.update(declared)
    return out


#: The closed local forms rung 2 is built in, each measured against a real checkpoint.
_SG_LOCAL_FORMS = ("landau", "flory_huggins")

#: Declared name -> constructor name, rung 2.
_SG_RENAMES = {"kappa": "kappa_init"}

#: Declared flags with no constructor argument, rung 2: the value at which each says nothing.
_SG_INERT_ONLY = {
    "gamma_fixed": (None, "freezes the mobility scale as a buffer; this package trains it"),
}

#: Rung 2's overrides spelled the declaration's way.
_SG_DECLARED_OVERRIDES = frozenset(_SG_RENAMES) | frozenset(_SG_INERT_ONLY) | {*_MOBILITY_OWNED}


def _square_gradient(system, declared: Dict[str, Any]) -> Dict[str, Any]:
    """Rung 2's constructor arguments from a closed local form's declaration."""
    spec = system.functional
    if spec.local not in _SG_LOCAL_FORMS:
        raise NotImplementedError(
            f"no declaration-to-constructor translation is measured for "
            f"square_gradient with local={spec.local!r}; only {list(_SG_LOCAL_FORMS)} "
            f"have been built from a real checkpoint and checked against it")
    if "local" in declared and declared.pop("local") != spec.local:
        raise ValueError(f"the declaration disagrees with itself about local={spec.local!r}")

    mobility = mobility_kwargs(system, functional_kwargs=declared)
    out: Dict[str, Any] = {"n_species": int(system.n_species), "local": spec.local}
    if "kB" in mobility:
        out["kB"] = mobility["kB"]

    for declared_name, arg in _SG_RENAMES.items():
        if declared_name in declared:
            out[arg] = declared.pop(declared_name)
    for name, (inert, meaning) in _SG_INERT_ONLY.items():
        if name in declared and declared.pop(name) != inert:
            raise NotImplementedError(
                f"{name} is declared at a value this package cannot build: it {meaning}. "
                f"Dropping it would build the variant the run did not use")
    for name in _MOBILITY_OWNED:
        declared.pop(name, None)
    for arg, value in mobility.items():
        if arg not in ("n_species", "kB", "rho_eps"):
            out[f"mobility_{arg}"] = value
    out.update(declared)
    return out


#: Declared form -> its translation; a form with no entry is refused.
_TRANSLATIONS = {"nonlocal_kernel": _nonlocal_kernel, "square_gradient": _square_gradient}

#: Declared form -> the override keys spelled the declaration's way (applied before the translation).
_OVERRIDES_SPELLED_AS_DECLARED = {"nonlocal_kernel": _DECLARED_OVERRIDES,
                                  "square_gradient": _SG_DECLARED_OVERRIDES}
