"""Rung 3, the local free-energy registry: ``f_loc = f_id(rho, kBT) + f_exc(rho, kBT)``.
``ideal_form="gas"``: ``f_id = kBT * sum_i [rho_i ln rho_i - rho_i]``, ``rho_i >= 0`` a partial density.
``ideal_form="lattice"``: ``f_id = kBT * sum_i [rho_i ln rho_i + (1 - rho_i) ln(1 - rho_i)]``, in (0, 1).
``f_exc`` split: ``u(rho) + kBT g_exc(rho) [+ g_hat kBT ln(kBT/kBT_ref)] [+ g_tilde kBT (kBT/kBT_ref)]``;
joint: ``u_joint(rho, kBT/kBT_ref)``. ``kBT`` is in energy units; T-basis heads are built last, zero-init.
``input_scale=True`` feeds every excess net ``z = rho/rho_ref - 1`` instead of ``rho`` (the ideal term keeps ``rho``).
``g_symmetry="mirror"`` (lattice only): ``g_exc(rho) -> (g_exc(rho) + g_exc(1 - rho)) / 2``."""
from __future__ import annotations

import itertools
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATIONS = {"gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU,
                "softplus": nn.Softplus}

U_FORMS: Tuple[str, ...] = ("icnn", "mlp", "taylor")

G_EXC_FORMS: Tuple[str, ...] = ("icnn", "mlp")

F_EXC_FORMS: Tuple[str, ...] = ("split", "joint")

U_PARITIES: Tuple[str, ...] = ("full", "even")

IDEAL_FORMS: Tuple[str, ...] = ("gas", "lattice")

G_SYMMETRIES: Tuple[str, ...] = ("none", "mirror")

TAYLOR_VARIABLES: Tuple[str, ...] = ("ratio", "difference")


def _make_activation(name: str) -> nn.Module:
    name = str(name)
    if name not in _ACTIVATIONS:
        raise ValueError(f"activation={name!r} not in {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


def _check(value: str, allowed: Tuple[str, ...], label: str) -> str:
    value = str(value)
    if value not in allowed:
        raise ValueError(f"{label}={value!r} not in {allowed}")
    return value


def _grad_wrt_rho(fn, rho: torch.Tensor, create_graph: bool) -> torch.Tensor:
    """d(fn(rho).sum()) / d(rho), on a fresh detached leaf per call."""
    rho_var = rho.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        out = fn(rho_var)
        return torch.autograd.grad(out.sum(), rho_var,
                                    create_graph=create_graph)[0]


def _taylor_exponents(n_species: int, degree: int,
                       parity: str) -> list[tuple[int, ...]]:
    """Every multi-index (e_1, ..., e_{n_species}) with 2 <= sum(e) <= degree.
    ``parity="even"`` keeps even total degree only; ``"full"`` keeps all."""
    exponents: list[tuple[int, ...]] = []
    for total in range(2, degree + 1):
        if parity == "even" and total % 2 != 0:
            continue
        slots = total + n_species - 1
        for cuts in itertools.combinations(range(slots), n_species - 1):
            bounds = list(cuts) + [slots]
            parts, prev = [], -1
            for b in bounds:
                parts.append(b - prev - 1)
                prev = b
            exponents.append(tuple(parts))
    return exponents


class TaylorEnergy(nn.Module):
    """``u = sum_{2 <= |e| <= degree} c_e * prod_s z_s^{e_s}``; ``u = grad u = 0`` at ``rho_ref``.
    ``variable="ratio"``: ``z = rho / rho_ref - 1``, cumulative powers (``torch.pow`` has a NaN second
    derivative at 0). ``variable="difference"``: ``z = rho - rho_ref``, ``**`` powers, ``sum(c * z**e)``."""

    def __init__(self, n_species: int, rho_ref: Sequence[float],
                 degree: int, parity: str = "full", *, variable: str):
        super().__init__()
        parity = _check(parity, U_PARITIES, "parity")
        self.variable = _check(variable, TAYLOR_VARIABLES, "variable")
        if degree < 2:
            raise ValueError(f"degree={degree!r} must be >= 2")
        if len(rho_ref) != n_species:
            raise ValueError(
                f"rho_ref={rho_ref!r} has {len(rho_ref)} entries, not "
                f"n_species={n_species}")
        self.n_species = int(n_species)
        self.degree = int(degree)
        self.parity = parity
        exponents = _taylor_exponents(self.n_species, self.degree, parity)
        self.register_buffer("exps", torch.tensor(exponents, dtype=torch.long))
        self.register_buffer("rho_ref",
                              torch.as_tensor(rho_ref, dtype=torch.float32))
        self.coefs = nn.Parameter(torch.zeros(len(exponents)))

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        if self.variable == "difference":
            return self._forward_difference(rho)
        z = rho / self.rho_ref.to(rho.dtype) - 1.0                # (..., n_species)
        pows = [torch.ones_like(z)]
        for _ in range(self.degree):
            pows.append(pows[-1] * z)
        P = torch.stack(pows, dim=-2)                              # (..., degree+1, n_species)
        mono = torch.ones(z.shape[:-1] + (self.exps.shape[0],),
                           dtype=z.dtype, device=z.device)
        for s in range(self.n_species):
            mono = mono * P[..., self.exps[:, s], s]
        return mono @ self.coefs.to(rho.dtype)

    def _forward_difference(self, rho: torch.Tensor) -> torch.Tensor:
        psi = rho - self.rho_ref.to(rho.dtype)                     # (..., n_species)
        terms = []
        for e in self.exps.tolist():
            mono = psi[..., 0] ** int(e[0])
            for s in range(1, self.n_species):
                mono = mono * psi[..., s] ** int(e[s])
            terms.append(mono)
        return (torch.stack(terms, dim=-1) * self.coefs.to(rho.dtype)).sum(dim=-1)


class ICNN(nn.Module):
    """Input-convex net over every species: softplus activations, non-negative hidden-to-hidden weights.
    ``output_bias`` gives the linear skip ``w2`` a bias or not; it is part of the state dict."""

    def __init__(self, n_species: int, hidden: int, *, output_bias: bool):
        super().__init__()
        self.W0 = nn.Linear(n_species, hidden)
        self.A1_raw = nn.Parameter(torch.randn(hidden, hidden) * 0.1 - 2.0)
        self.W1 = nn.Linear(n_species, hidden)
        self.a2_raw = nn.Parameter(torch.randn(hidden) * 0.1 - 2.0)
        self.w2 = nn.Linear(n_species, 1, bias=bool(output_bias))

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        z1 = F.softplus(self.W0(rho))
        z2 = F.softplus(z1 @ F.softplus(self.A1_raw).T + self.W1(rho))
        return z2 @ F.softplus(self.a2_raw) + self.w2(rho).squeeze(-1)


class GeneralMLP(nn.Module):
    """A plain net over every species; bias-free last layer scaled by ``init_scale``.
    ``zero_init=True`` sets that weight to exactly zero (an exact no-op head at init)."""

    def __init__(self, n_species: int, hidden: int, activation: str = "gelu",
                 zero_init: bool = False, init_scale: float = 0.1):
        super().__init__()
        act = _make_activation
        self.net = nn.Sequential(
            nn.Linear(n_species, hidden), act(activation),
            nn.Linear(hidden, hidden), act(activation),
            nn.Linear(hidden, 1, bias=False),
        )
        with torch.no_grad():
            if zero_init:
                self.net[-1].weight.zero_()
            else:
                self.net[-1].weight.mul_(init_scale)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        return self.net(rho).squeeze(-1)


class MLPEnergy(nn.Module):
    """``u(rho)`` via a general MLP; with ``gauge_fix=True``
    ``u_gauged(rho) = u_raw(rho) - u_raw(rho_ref) - grad(u_raw)(rho_ref) . (rho - rho_ref)``."""

    def __init__(self, n_species: int, hidden: int, rho_ref: Sequence[float],
                 activation: str = "gelu", gauge_fix: bool = False):
        super().__init__()
        if len(rho_ref) != n_species:
            raise ValueError(
                f"rho_ref={rho_ref!r} has {len(rho_ref)} entries, not "
                f"n_species={n_species}")
        self.net = GeneralMLP(n_species, hidden, activation,
                               zero_init=False, init_scale=0.1)
        self.register_buffer("rho_ref",
                              torch.as_tensor(rho_ref, dtype=torch.float32))
        self.gauge_fix = bool(gauge_fix)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        if not self.gauge_fix:
            return self.net(rho)
        ref = (self.rho_ref.detach().clone().to(rho.dtype)
               .unsqueeze(0).requires_grad_(True))
        with torch.enable_grad():
            u_ref = self.net(ref)
            g_ref = torch.autograd.grad(u_ref.sum(), ref, create_graph=True)[0]
        u = self.net(rho)
        return u - u_ref.squeeze() - ((rho - ref) * g_ref).sum(-1)


class JointEnergy(nn.Module):
    """``u_joint(rho, kBT / kBT_ref)``: one net replacing ``u + kBT * g_exc`` and every T-basis head."""

    def __init__(self, n_species: int, hidden: int, activation: str = "gelu",
                 depth: int = 2):
        """``depth`` is the number of hidden Linear layers before the output (>= 1)."""
        super().__init__()
        if int(depth) < 1:
            raise ValueError(
                f"depth={depth!r} must be at least 1: a joint net with no "
                f"hidden layer is a linear map, not the form this class is")
        act = _make_activation
        layers: list[nn.Module] = [nn.Linear(n_species + 1, hidden),
                                   act(activation)]
        for _ in range(int(depth) - 1):
            layers += [nn.Linear(hidden, hidden), act(activation)]
        layers.append(nn.Linear(hidden, 1, bias=False))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)

    def forward(self, rho: torch.Tensor, t_ratio: torch.Tensor) -> torch.Tensor:
        """``rho`` (..., n_species), ``t_ratio`` (...,) -> (...,)."""
        t = t_ratio.to(dtype=rho.dtype, device=rho.device).unsqueeze(-1)
        return self.net(torch.cat([rho, t], dim=-1)).squeeze(-1)


class FLocal(nn.Module):
    """The assembled local free-energy density; see the module docstring."""

    def __init__(
        self,
        n_species: int,
        rho_ref: Sequence[float],
        kBT_ref: Optional[float],
        h_u: Optional[int],
        h_g: int,
        *,
        ideal_form: str,
        f_exc_form: str = "split",
        u_form: str = "mlp",
        u_degree: int = 4,
        u_parity: str = "full",
        u_variable: str = "ratio",
        gauge_fix: bool = False,
        g_exc_form: str = "icnn",
        g_symmetry: str = "none",
        icnn_output_bias: bool = True,
        enable_TlnT: bool = False,
        enable_T2: bool = False,
        tbasis_ortho_window: Optional[Sequence[float]] = None,
        tbasis_ortho_points: Optional[int] = None,
        h_g_hat: Optional[int] = None,
        h_g_tilde: Optional[int] = None,
        h_joint: Optional[int] = None,
        joint_depth: int = 2,
        activation: str = "gelu",
        rho_eps: float = 1e-6,
        defer_T_heads: bool = False,
        input_scale: bool = False,
    ):
        # Validate everything before anything is constructed.
        ideal_form = _check(ideal_form, IDEAL_FORMS, "ideal_form")
        f_exc_form = _check(f_exc_form, F_EXC_FORMS, "f_exc_form")
        u_form = _check(u_form, U_FORMS, "u_form")
        g_exc_form = _check(g_exc_form, G_EXC_FORMS, "g_exc_form")
        u_parity = _check(u_parity, U_PARITIES, "u_parity")
        u_variable = _check(u_variable, TAYLOR_VARIABLES, "u_variable")
        g_symmetry = _check(g_symmetry, G_SYMMETRIES, "g_symmetry")
        if g_symmetry == "mirror" and (ideal_form != "lattice" or input_scale
                                       or f_exc_form != "split"):
            raise ValueError(
                "g_symmetry='mirror' averages g_exc over rho and 1 - rho: it "
                "needs ideal_form='lattice' (a fraction in (0, 1)), the split "
                "form and unscaled input")
        _make_activation(activation)  # raises on an unknown name
        if n_species < 1:
            raise ValueError(f"n_species={n_species!r} must be >= 1")
        if len(rho_ref) != n_species:
            raise ValueError(
                f"rho_ref={rho_ref!r} has {len(rho_ref)} entries, not "
                f"n_species={n_species}")
        if f_exc_form == "joint" and (g_exc_form != "icnn" or enable_TlnT
                                       or enable_T2):
            raise ValueError(
                "f_exc_form='joint' replaces u(rho) + kBT*g_exc(rho) "
                "entirely; combining it with g_exc_form='mlp', "
                "enable_TlnT or enable_T2 is ill-defined")
        if f_exc_form == "joint" and h_joint is None:
            raise ValueError("h_joint is required when f_exc_form='joint'")
        if input_scale and f_exc_form == "split" and u_form == "taylor":
            raise ValueError(
                "input_scale=True with u_form='taylor': the Taylor energy is "
                "already a polynomial in z = rho/rho_ref - 1, so scaling its "
                "input again would expand it about a different point")
        if input_scale and f_exc_form == "split" and u_form == "mlp" \
                and gauge_fix:
            raise NotImplementedError(
                "input_scale=True with gauge_fix=True: the gauge subtracts "
                "the energy's affine part at rho_ref, and whether that "
                "reference is taken before or after the input scaling is a "
                "choice no measured model has made")
        if enable_TlnT and h_g_hat is None:
            raise ValueError("h_g_hat is required when enable_TlnT=True")
        if kBT_ref is None and (f_exc_form == "joint" or enable_TlnT or enable_T2
                                or tbasis_ortho_window is not None):
            raise ValueError(
                "kBT_ref is required by the joint form, the T-basis heads and "
                "their orthogonalisation, which all read kBT / kBT_ref")
        if h_u is None and f_exc_form == "split" and u_form in ("icnn", "mlp"):
            raise ValueError(f"h_u is required by u_form={u_form!r}: it is that net's width")
        if enable_T2 and h_g_tilde is None:
            raise ValueError("h_g_tilde is required when enable_T2=True")

        super().__init__()
        self.n_species = int(n_species)
        self.ideal_form = ideal_form
        self.f_exc_form = f_exc_form
        self.u_form = u_form
        self.g_exc_form = g_exc_form
        self.enable_TlnT = bool(enable_TlnT)
        self.enable_T2 = bool(enable_T2)
        self.rho_eps = float(rho_eps)
        self.g_symmetry = g_symmetry
        self._activation = str(activation)
        self._h_g_hat = h_g_hat
        self._h_g_tilde = h_g_tilde

        # Non-persistent: state_dict keys identical across variant flags.
        self.register_buffer(
            "kBT_ref", None if kBT_ref is None
            else torch.tensor(float(kBT_ref), dtype=torch.float64),
            persistent=False)

        self._build_tbasis_ortho(tbasis_ortho_window, tbasis_ortho_points)

        # Non-persistent, float32 (the nets' width): z = rho / in_ref - 1 into every excess net.
        self.input_scale = bool(input_scale)
        self.register_buffer(
            "in_ref",
            torch.as_tensor(rho_ref, dtype=torch.float32) if self.input_scale
            else None,
            persistent=False)

        # Energy then entropy, never interleaved with the T-basis heads.
        if f_exc_form == "joint":
            self.u_joint_net = JointEnergy(self.n_species, int(h_joint),
                                            activation, depth=joint_depth)
        else:
            if u_form == "icnn":
                self.u_net = ICNN(self.n_species, int(h_u),
                                  output_bias=icnn_output_bias)
            elif u_form == "mlp":
                self.u_net = MLPEnergy(self.n_species, int(h_u), rho_ref,
                                        activation, gauge_fix=gauge_fix)
            else:  # "taylor"
                self.u_net = TaylorEnergy(self.n_species, rho_ref,
                                           int(u_degree), u_parity,
                                           variable=u_variable)
            self.g_net = (ICNN(self.n_species, int(h_g),
                               output_bias=icnn_output_bias)
                          if g_exc_form == "icnn"
                          else GeneralMLP(self.n_species, int(h_g), activation,
                                          zero_init=False, init_scale=0.1))

        # T-basis heads last (or deferred past the caller's submodules): RNG stream unchanged.
        self._pending_T_heads = bool(defer_T_heads)
        if not self._pending_T_heads:
            self.build_T_heads()

    def build_T_heads(self) -> None:
        """Construct the T-basis heads (the second half of a deferred build)."""
        if getattr(self, "g_hat_net", None) is not None or \
                getattr(self, "g_tilde_net", None) is not None:
            raise RuntimeError("build_T_heads() called twice")
        if self.f_exc_form != "joint":
            if self.enable_TlnT:
                self.g_hat_net = GeneralMLP(self.n_species, int(self._h_g_hat),
                                             self._activation, zero_init=True)
            if self.enable_T2:
                self.g_tilde_net = GeneralMLP(self.n_species,
                                               int(self._h_g_tilde),
                                               self._activation, zero_init=True)
        self._pending_T_heads = False

    def _check_heads_built(self) -> None:
        if self._pending_T_heads:
            raise RuntimeError(
                "FLocal was built with defer_T_heads=True and "
                "build_T_heads() was never called: the T-basis heads are "
                "missing, so this model is not the form its config asks for.")

    def _net_in(self, rho: torch.Tensor) -> torch.Tensor:
        """An excess net's input: ``rho / in_ref - 1`` when ``input_scale``, else ``rho`` itself."""
        if self.in_ref is None:
            return rho
        return rho / self.in_ref.to(dtype=rho.dtype, device=rho.device) - 1.0

    def _scaled(self, net):
        """``net`` behind :meth:`_net_in`; the net itself when nothing is scaled (the same graph as before)."""
        if self.in_ref is None:
            return net
        return lambda r: net(self._net_in(r))

    def _g(self):
        """The excess-entropy net as ``g_symmetry`` declares it; the net itself under ``"none"``."""
        if self.g_symmetry == "none":
            return self.g_net
        return lambda r: 0.5 * (self.g_net(r) + self.g_net(1.0 - r))

    def du_drho(self, rho: torch.Tensor) -> torch.Tensor:
        """``d u / d rho`` of the split form's energy net, pointwise."""
        return _grad_wrt_rho(self._scaled(self.u_net), rho,
                             create_graph=self.training)

    def dg_drho(self, rho: torch.Tensor) -> torch.Tensor:
        """``d g_exc / d rho`` of the split form's entropy net (symmetry applied), pointwise."""
        return _grad_wrt_rho(self._scaled(self._g()), rho,
                             create_graph=self.training)

    def _kBT_ratio(self, kBT: torch.Tensor) -> torch.Tensor:
        return kBT / self.kBT_ref.to(dtype=kBT.dtype, device=kBT.device)

    def _build_tbasis_ortho(self, window: Optional[Sequence[float]],
                            points: Optional[int]) -> None:
        """Gram-Schmidt the T-basis prefactors against ``{1, kBT}`` over ``window`` (kBT units).
        ``window=None`` is off; ``points`` is required with a window."""
        self.tbasis_ortho = window is not None
        if window is None:
            if points is not None:
                raise ValueError(
                    "tbasis_ortho_points was given without "
                    "tbasis_ortho_window, so there is nothing to fit; pass "
                    "both to orthogonalise the T basis, or neither")
            return
        if points is None:
            raise ValueError(
                "tbasis_ortho_window requires tbasis_ortho_points (the "
                "sample count of the least-squares fit); core ships no "
                "default for it")
        lo, hi = (float(w) for w in window)
        if not hi > lo:
            raise ValueError(
                f"tbasis_ortho_window={window!r} must be (lo, hi) with "
                f"hi > lo")
        if int(points) < 2:
            raise ValueError(
                f"tbasis_ortho_points={points!r} must be >= 2 to fit two "
                f"coefficients")

        # float64: config constants solved once.
        kbt = torch.linspace(lo, hi, int(points), dtype=torch.float64)
        kbt_ref = self.kBT_ref.to(dtype=torch.float64)
        A = torch.stack([torch.ones_like(kbt), kbt], dim=1)
        curves = (("tlnt", kbt * torch.log(kbt / kbt_ref)),
                  ("t2", kbt * (kbt / kbt_ref)))
        for name, c in curves:
            # `driver` named: an unnamed lstsq is not reproducible run to run.
            coef = torch.linalg.lstsq(
                A, c.unsqueeze(-1), driver="gels").solution.squeeze(-1)
            self.register_buffer(f"_ortho_{name}_a", coef[0].clone(),
                                 persistent=False)
            self.register_buffer(f"_ortho_{name}_b", coef[1].clone(),
                                 persistent=False)

    def _ortho_subtract(self, name: str, c: torch.Tensor,
                        kBT: torch.Tensor) -> torch.Tensor:
        """``c - a - b*kBT``, in that association (the float32 rounding the published weights were fitted under)."""
        a = getattr(self, f"_ortho_{name}_a").to(dtype=kBT.dtype,
                                                 device=kBT.device)
        b = getattr(self, f"_ortho_{name}_b").to(dtype=kBT.dtype,
                                                 device=kBT.device)
        return c - a - b * kBT

    def _c_TlnT(self, kBT: torch.Tensor) -> torch.Tensor:
        c = kBT * torch.log(self._kBT_ratio(kBT))
        if self.tbasis_ortho:
            c = self._ortho_subtract("tlnt", c, kBT)
        return c

    def _c_T2(self, kBT: torch.Tensor) -> torch.Tensor:
        c = kBT * self._kBT_ratio(kBT)
        if self.tbasis_ortho:
            c = self._ortho_subtract("t2", c, kBT)
        return c

    def _joint_grad(self, rho: torch.Tensor, kBT: torch.Tensor) -> torch.Tensor:
        """d(u_joint)/d(rho) at fixed kBT (the temperature ratio is detached)."""
        t_ratio = self._kBT_ratio(kBT).detach()
        return _grad_wrt_rho(
            lambda r: self.u_joint_net(self._net_in(r), t_ratio), rho,
            create_graph=self.training)

    def _f_id(self, rho: torch.Tensor, kBT: torch.Tensor) -> torch.Tensor:
        """The analytic ideal term, dispatched on ``ideal_form``."""
        if self.ideal_form == "gas":
            rc = rho.clamp(min=self.rho_eps)
            return kBT * (rc * rc.log() - rc).sum(-1)
        # lattice: clamp both ends; ln(1 - rho) as log1p(-rho).
        rc = rho.clamp(min=self.rho_eps, max=1.0 - self.rho_eps)
        return kBT * (rc * torch.log(rc)
                      + (1.0 - rc) * torch.log1p(-rc)).sum(-1)

    def _mu_id(self, rho: torch.Tensor, kBT: torch.Tensor) -> torch.Tensor:
        """``d(_f_id)/d(rho)`` in closed form."""
        if self.ideal_form == "gas":
            rc = rho.clamp(min=self.rho_eps)
            return kBT.unsqueeze(-1) * rc.log()
        rc = rho.clamp(min=self.rho_eps, max=1.0 - self.rho_eps)
        return kBT.unsqueeze(-1) * (torch.log(rc) - torch.log1p(-rc))

    def f_pointwise(self, rho: torch.Tensor, kBT: torch.Tensor) -> torch.Tensor:
        """``rho`` (..., n_species), ``kBT`` (...,) -> ``f_loc`` (...,)."""
        self._check_heads_built()
        f = self._f_id(rho, kBT)
        if self.f_exc_form == "joint":
            return f + self.u_joint_net(self._net_in(rho),
                                        self._kBT_ratio(kBT))
        z = self._net_in(rho)
        f = f + self.u_net(z)
        f = f + kBT * self._g()(z)
        if self.enable_TlnT:
            f = f + self.g_hat_net(z) * self._c_TlnT(kBT)
        if self.enable_T2:
            f = f + self.g_tilde_net(z) * self._c_T2(kBT)
        return f

    def mu_pointwise(self, rho: torch.Tensor, kBT: torch.Tensor) -> torch.Tensor:
        """``rho`` (..., n_species), ``kBT`` (...,) -> ``mu`` (..., n_species)."""
        self._check_heads_built()
        mu = self._mu_id(rho, kBT)
        if self.f_exc_form == "joint":
            return mu + self._joint_grad(rho, kBT)
        mu = mu + self.du_drho(rho)
        mu = mu + kBT.unsqueeze(-1) * self.dg_drho(rho)
        if self.enable_TlnT:
            dg_hat = _grad_wrt_rho(self._scaled(self.g_hat_net), rho,
                                    create_graph=self.training)
            mu = mu + self._c_TlnT(kBT).unsqueeze(-1) * dg_hat
        if self.enable_T2:
            dg_tilde = _grad_wrt_rho(self._scaled(self.g_tilde_net), rho,
                                      create_graph=self.training)
            mu = mu + self._c_T2(kBT).unsqueeze(-1) * dg_tilde
        return mu
