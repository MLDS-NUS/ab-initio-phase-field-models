"""The mobility form registry: ``M(rho, T) = sqrt(g) M_tilde sqrt(g)``, times an optional T factor.
Three declared axes, ``n_species``-general:
``prefactor``: ``"mole_fraction"`` (``g_i = rho_i (1 - rho_i)``) or ``"partial_density"`` (``g_i = rho_i``);
``shape`` of ``M_tilde``: ``"fixed"``, ``"constant"``, ``"mlp_rho"``, ``"mlp_rho_t"``;
``t_form``: ``"none"`` or ``"arrhenius"``. ``M_tilde = L L^T``, softplus diagonal, free-sign off-diagonal.
The Arrhenius barrier is an energy: ``c = exp(-E (1/(kB T) - 1/(kB T_ref)))``, ``kB`` in its unit per kelvin.
``input_ref`` feeds an MLP shape ``z = rho/input_ref - 1``; the ``sqrt(g)`` sandwich always uses ``rho``.
``shape="lattice_scalar"`` (one field, ``prefactor="mole_fraction"``): ``M = r (1 - r) exp(log_gamma) c(T)``,
``r = clamp(rho, rho_eps, 1 - rho_eps)``, ``c = exp(-softplus(Ea_raw) (1/(kB T) - 1/(kB T_ref)))``; two scalars.
Measured values have no default."""
from __future__ import annotations

import inspect
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

#: The three declared axes, each validated against its own tuple.
PREFACTOR_FORMS = ("mole_fraction", "partial_density")
SHAPE_FORMS = ("fixed", "constant", "mlp_rho", "mlp_rho_t", "lattice_scalar")
T_FORMS = ("none", "arrhenius")

#: Shape forms that take `T` directly; they may not combine with a nontrivial `t_form`.
_SHAPE_CONSUMES_T = frozenset({"mlp_rho_t"})

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
}


def _n_lower_triangular(n_species: int) -> int:
    """Parameter count of an `n_species x n_species` Cholesky factor."""
    return n_species * (n_species + 1) // 2


def cholesky_raw_to_matrix(raw: torch.Tensor, n_species: int) -> torch.Tensor:
    """``raw (..., n(n+1)/2)``, row-major lower triangle, softplus diagonal -> PSD ``L @ L^T``.
    At ``n_species = 1`` this is ``softplus(raw)**2``."""
    expected = _n_lower_triangular(n_species)
    if raw.shape[-1] != expected:
        raise ValueError(
            f"expected {expected} Cholesky parameters for n_species="
            f"{n_species}, got {raw.shape[-1]}")

    zero = torch.zeros_like(raw[..., 0])
    entries = {}
    idx = 0
    for i in range(n_species):
        for j in range(i + 1):
            val = raw[..., idx]
            if i == j:
                val = F.softplus(val)
            entries[(i, j)] = val
            idx += 1

    rows = []
    for i in range(n_species):
        row = [entries[(i, j)] if j <= i else zero for j in range(n_species)]
        rows.append(torch.stack(row, dim=-1))
    L = torch.stack(rows, dim=-2)
    return L @ L.transpose(-1, -2)


def mobility_prefactor(rho: torch.Tensor, form: str) -> torch.Tensor:
    """The degeneracy vector ``g(rho)``, ``(..., n_species)``, with ``rho`` clamped at zero."""
    if form not in PREFACTOR_FORMS:
        raise ValueError(f"unknown mobility prefactor {form!r}; have {PREFACTOR_FORMS}")
    rho_c = rho.clamp(min=0.0)
    if form == "mole_fraction":
        return rho_c * (1.0 - rho_c).clamp(min=0.0)
    return rho_c  # "partial_density"


class Mobility(nn.Module):
    """``M(rho, T)`` from the three axes; ``rho`` ``(..., n_species)`` channel-last, ``T`` ``(...,)``.
    Returns ``(..., n_species, n_species)``, symmetric PSD for any parameter values."""

    def __init__(
        self,
        n_species: int,
        prefactor: str,
        shape: str,
        t_form: str,
        *,
        shape_init: Optional[torch.Tensor] = None,
        hidden: int = 32,
        activation: str = "gelu",
        t_ref: Optional[float] = None,
        activation_energy_init: Optional[Sequence[float]] = None,
        kB: Optional[float] = None,
        input_ref: Optional[Sequence[float]] = None,
        rho_eps: Optional[float] = None,
    ) -> None:
        super().__init__()
        if prefactor not in PREFACTOR_FORMS:
            raise ValueError(f"unknown mobility prefactor {prefactor!r}; have {PREFACTOR_FORMS}")
        if shape not in SHAPE_FORMS:
            raise ValueError(f"unknown mobility shape {shape!r}; have {SHAPE_FORMS}")
        if t_form not in T_FORMS:
            raise ValueError(f"unknown mobility T form {t_form!r}; have {T_FORMS}")
        if shape in _SHAPE_CONSUMES_T and t_form != "none":
            raise ValueError(
                f"shape={shape!r} already consumes T directly; t_form must "
                f"be 'none' to avoid applying a temperature dependence "
                f"twice, got t_form={t_form!r}")

        if input_ref is not None and shape not in ("mlp_rho", "mlp_rho_t"):
            raise ValueError(
                f"input_ref scales an MLP's density input and shape="
                f"{shape!r} has no MLP; accepting it would record a scaling "
                f"that never ran")
        if input_ref is not None and len(input_ref) != n_species:
            raise ValueError(
                f"input_ref={input_ref!r} has {len(input_ref)} entries, not "
                f"n_species={n_species}")

        if (rho_eps is not None) != (shape == "lattice_scalar"):
            raise ValueError(
                f"rho_eps is the clamp of shape='lattice_scalar' and only of "
                f"it; got rho_eps={rho_eps!r} with shape={shape!r}")
        if shape == "lattice_scalar":
            self._init_lattice_scalar(n_species, prefactor, t_form, shape_init,
                                      t_ref, activation_energy_init, kB, rho_eps)
            return

        self.n_species = int(n_species)
        self.prefactor_form = prefactor
        self.shape_form = shape
        self.t_form = t_form
        self._n_chol = _n_lower_triangular(self.n_species)

        if shape in ("fixed", "constant"):
            if shape_init is None:
                raise ValueError(
                    f"shape={shape!r} requires shape_init (the "
                    f"{self._n_chol} raw Cholesky parameters); core ships "
                    f"no default -- initialisation comes from the caller")
            init = torch.as_tensor(shape_init, dtype=torch.float32).clone()
            if init.shape[-1] != self._n_chol:
                raise ValueError(
                    f"shape_init must carry {self._n_chol} parameters for "
                    f"n_species={self.n_species}, got shape {tuple(init.shape)}")
            if shape == "fixed":
                self.register_buffer("_raw", init)
            else:
                self._raw = nn.Parameter(init)
        else:
            act = _ACTIVATIONS[activation]
            n_in = self.n_species + (1 if shape == "mlp_rho_t" else 0)
            self.m_net = nn.Sequential(
                nn.Linear(n_in, hidden), act(),
                nn.Linear(hidden, hidden), act(),
                nn.Linear(hidden, self._n_chol),
            )
        # Non-persistent, float32: the same state_dict with the scaling on or off.
        self.register_buffer(
            "in_ref",
            None if input_ref is None
            else torch.as_tensor(input_ref, dtype=torch.float32),
            persistent=False)

        if t_form == "arrhenius":
            if t_ref is None:
                raise ValueError(
                    "t_form='arrhenius' requires t_ref; core ships no "
                    "default reference temperature")
            if activation_energy_init is None:
                raise ValueError(
                    "t_form='arrhenius' requires activation_energy_init "
                    f"({self.n_species} values); core ships no default")
            if kB is None:
                raise ValueError(
                    "t_form='arrhenius' requires kB, the Boltzmann constant "
                    "in the barrier's energy unit per kelvin; it is a "
                    "system's unit convention and core ships no value")
            self.kB = float(kB)
            self.t_ref = float(t_ref)
            e_init = torch.as_tensor(activation_energy_init, dtype=torch.float32).clone()
            if e_init.shape[-1] != self.n_species:
                raise ValueError(
                    f"activation_energy_init must carry {self.n_species} "
                    f"values, got shape {tuple(e_init.shape)}")
            self._activation_energy_raw = nn.Parameter(e_init)

    def _init_lattice_scalar(self, n_species, prefactor, t_form, shape_init,
                             t_ref, activation_energy_init, kB, rho_eps) -> None:
        """Two scalar parameters: ``_log_gamma`` (raw ``shape_init``) and, for Arrhenius, ``_Ea_raw``."""
        if int(n_species) != 1 or prefactor != "mole_fraction":
            raise ValueError(
                "shape='lattice_scalar' is the one-field fraction form: "
                f"n_species=1 and prefactor='mole_fraction', got "
                f"{n_species!r} and {prefactor!r}")
        if shape_init is None:
            raise ValueError("shape='lattice_scalar' requires shape_init, the raw "
                             "log gamma; core ships no default")
        self.n_species, self.prefactor_form = 1, prefactor
        self.shape_form, self.t_form = "lattice_scalar", t_form
        self.rho_eps = float(rho_eps)
        self._log_gamma = nn.Parameter(
            torch.as_tensor(shape_init, dtype=torch.float32).clone().reshape(()))
        if t_form == "arrhenius":
            if t_ref is None or activation_energy_init is None or kB is None:
                raise ValueError(
                    "t_form='arrhenius' requires t_ref, activation_energy_init "
                    "(the raw barrier) and kB; core ships no default for any")
            self.kB, self.t_ref = float(kB), float(t_ref)
            self._Ea_raw = nn.Parameter(torch.as_tensor(
                activation_energy_init, dtype=torch.float32).clone().reshape(()))

    def _lattice_scalar(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        rho_s = rho.clamp(self.rho_eps, 1.0 - self.rho_eps)
        pre = rho_s * (1.0 - rho_s)                                   # (..., 1)
        fac = torch.exp(self._log_gamma).expand_as(T)
        if self.t_form == "arrhenius":
            fac = fac * torch.exp(-F.softplus(self._Ea_raw)
                                  * (1.0 / (self.kB * T)
                                     - 1.0 / (self.kB * self.t_ref)))
        return (pre * fac.unsqueeze(-1)).unsqueeze(-1)                # (..., 1, 1)

    def _m_tilde_raw(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        if self.shape_form in ("fixed", "constant"):
            return self._raw.expand(*rho.shape[:-1], self._n_chol)
        if self.in_ref is not None:
            rho = rho / self.in_ref.to(dtype=rho.dtype, device=rho.device) - 1.0
        if self.shape_form == "mlp_rho":
            return self.m_net(rho)
        T_col = T.unsqueeze(-1).expand(*rho.shape[:-1], 1)
        return self.m_net(torch.cat([rho, T_col], dim=-1))

    def forward(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        if self.shape_form == "lattice_scalar":
            return self._lattice_scalar(rho, T)
        raw = self._m_tilde_raw(rho, T)
        M_tilde = cholesky_raw_to_matrix(raw, self.n_species)

        g = mobility_prefactor(rho, self.prefactor_form)
        s = torch.sqrt(g.clamp(min=0.0))
        M = M_tilde * (s.unsqueeze(-1) * s.unsqueeze(-2))

        if self.t_form == "arrhenius":
            E = F.softplus(self._activation_energy_raw)
            inv = (1.0 / (self.kB * T.unsqueeze(-1))
                   - 1.0 / (self.kB * self.t_ref))
            c = torch.exp(-E * inv).clamp(min=0.0)
            a = torch.sqrt(c)
            M = M * (a.unsqueeze(-1) * a.unsqueeze(-2))
        return M


# ===========================================================================
# The front door: the mobility a System declares.
# ===========================================================================

#: Declared mobility form -> this module's ``shape`` axis; the prefactor is declared beside it.
MOBILITY_SHAPES = {
    "mlp_scaled": "mlp_rho",
    "mlp": "mlp_rho",
    "mlp_rho_t": "mlp_rho_t",
    "constant": "constant",
    "fixed": "fixed",
    "gamma": "constant",
    "lattice_scalar": "lattice_scalar",
}

#: Where each constructor argument is declared: the Mobility declaration or the model kwargs.
_FROM_MOBILITY = {"mobility_prefactor": "prefactor",
                  "mobility_input_ref": "input_ref",
                  "mobility_activation_energy_init": "activation_energy_init"}
_FROM_FUNCTIONAL = {"h_m": "hidden", "activation": "activation",
                    "T_ref": "t_ref"}


def mobility_kwargs(system, *, functional_kwargs=None, **overrides):
    """The :class:`Mobility` constructor arguments ``system`` declares; ``overrides`` use this class's names.
    ``functional_kwargs`` (default ``system.functional.kwargs``) carries already-overridden model kwargs."""
    if system.mobility is None:
        raise ValueError(
            f"system {system.name!r} declares no mobility, so there is "
            f"nothing to build; a mobility form is a fact about a system "
            f"and this package ships no default for it")
    spec = system.mobility
    if spec.form not in MOBILITY_SHAPES:
        raise NotImplementedError(
            f"declared mobility form {spec.form!r} has no measured shape in "
            f"this package; have {sorted(MOBILITY_SHAPES)}")

    out = {"n_species": int(system.n_species),
           "shape": MOBILITY_SHAPES[spec.form],
           "t_form": spec.T_form}

    declared = dict(spec.kwargs)
    for declared_name, arg in _FROM_MOBILITY.items():
        if declared_name in declared:
            out[arg] = declared.pop(declared_name)
    if functional_kwargs is None and system.functional is not None:
        functional_kwargs = system.functional.kwargs
    if functional_kwargs is not None:
        for declared_name, arg in _FROM_FUNCTIONAL.items():
            if declared_name in functional_kwargs:
                out[arg] = functional_kwargs[declared_name]

    # `arrhenius_shared_Ea=True` (one shared activation energy) is a model this package does not build.
    if declared.pop("arrhenius_shared_Ea", False):
        raise NotImplementedError(
            "arrhenius_shared_Ea=True describes one activation energy "
            "shared across species; this package gives each species its "
            "own, so the flag is refused rather than dropped")
    # Anything left reaches the constructor under its own name.
    out.update(declared)

    # An MLP's input scaling is a non-persistent buffer: absent from every checkpoint, so declared.
    if out["shape"] in ("mlp_rho", "mlp_rho_t") and "input_ref" not in out:
        raise ValueError(
            f"system {system.name!r} declares an MLP mobility and no "
            f"mobility_input_ref: whether the net is fed rho or "
            f"rho/rho_ref - 1 is stored in no checkpoint, so a strict load "
            f"succeeds either way and the wrong one is off by any factor. "
            f"Declare mobility_input_ref (None for raw rho)")
    # The lattice form clamps at the functional's own declared rho_eps.
    if out["shape"] == "lattice_scalar" and "rho_eps" not in out:
        if functional_kwargs is None or "rho_eps" not in functional_kwargs:
            raise ValueError(
                f"system {system.name!r} declares a lattice_scalar mobility "
                f"and no rho_eps to clamp its fraction at")
        out["rho_eps"] = functional_kwargs["rho_eps"]
    # The barrier is an energy, so the Arrhenius factor needs the system's own kB.
    if out["t_form"] == "arrhenius" and "kB" not in out:
        if "kB" not in system.constants:
            raise KeyError(
                f"system {system.name!r} declares an Arrhenius mobility and "
                f"no constant 'kB' to express its barrier in")
        out["kB"] = system.constants["kB"]

    unknown = sorted(set(overrides) - _mobility_arguments())
    if unknown:
        raise TypeError(
            f"Mobility accepts no argument named {unknown}; have "
            f"{sorted(_mobility_arguments())}")
    out.update(overrides)
    return out


def _mobility_arguments() -> frozenset:
    """Every keyword :class:`Mobility` accepts, read off its signature."""
    return frozenset(inspect.signature(Mobility.__init__).parameters) - {"self"}


def build(system, **overrides) -> Mobility:
    """The standalone mobility ``system`` declares, with ``overrides`` recorded as ``build_overrides``."""
    model = Mobility(**mobility_kwargs(system, **overrides))
    model.build_overrides = dict(overrides)
    return model
