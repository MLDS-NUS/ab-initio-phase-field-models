"""The golden cases: each one a function of nothing that returns ``{key: value}``, run by the generator
(``make_goldens.py``) and by the test (``test_golden_outputs.py``) alike.

A value is a tensor (compared bit for bit), or a string, number, tuple or ``None`` (compared with
``==``). Every input is made here from explicit seeds with torch and numpy primitives, never with the
code under test, so that a change to that code moves only its outputs. The models are built under a
fixed seed and then given the parameters stored in ``models.pt`` (:func:`use_model_states`): a change
to how a constructor draws its initial weights is reported once, by the ``models`` group, and does not
move every rollout. Everything runs on the CPU in float32 (the published precision), one thread.
"""
from __future__ import annotations

import math
import warnings
from collections import namedtuple
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict

import numpy as np
import torch

#: group -> case name -> the function computing it. One fixture file per group.
GROUPS: Dict[str, Dict[str, Callable[[], Dict[str, Any]]]] = {}

#: model name -> its builder (a module built under its own seed).
MODELS: Dict[str, Callable[[], torch.nn.Module]] = {}

#: model name -> the stored state dict every case's model is given; set by :func:`use_model_states`.
_STATES: Dict[str, Dict[str, torch.Tensor]] = {}

#: The Boltzmann constant of the eV-unit models (eV/K); the reduced-unit models use 1.0.
EV_KB = 8.617333262e-5


def case(group: str, name: str):
    """Register ``fn`` as case ``name`` of ``group``."""
    def register(fn):
        table = GROUPS.setdefault(group, {})
        if name in table:
            raise ValueError(f"case {group}/{name} is registered twice")
        table[name] = fn
        return fn
    return register


def model_builder(name: str):
    """Register a zero-argument model builder under ``name``."""
    def register(fn):
        MODELS[name] = fn
        return fn
    return register


def use_model_states(states: Dict[str, Dict[str, torch.Tensor]]) -> None:
    """Every :func:`model` from now on loads its parameters from ``states`` (strictly)."""
    _STATES.clear()
    _STATES.update(states)


def model_states(models_file: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """The state dicts in a loaded ``models.pt``, by model name."""
    prefix = "state/"
    return {name: {k[len(prefix):]: v for k, v in outputs.items()}
            for name, outputs in models_file["cases"].items()}


# ---------------------------------------------------------------------------
# where the outputs live, and how two of them are compared
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent


def fixture_key() -> str:
    """``torch-<version>-<capability>``: the build and the instruction set the bits depend on."""
    raw = f"torch-{torch.__version__}-{torch.backends.cpu.get_cpu_capability()}"
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in raw).lower()


def fixture_dir() -> Path:
    return HERE / fixture_key()


def _bits(t: torch.Tensor) -> torch.Tensor:
    """``t`` as integers of the same width, so that ``-0.0``, ``0.0`` and every NaN compare by bits."""
    if t.is_complex():
        t = torch.view_as_real(t)
    if t.is_floating_point():
        t = t.contiguous().view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])
    return t


def compare(expected: Dict[str, Any], actual: Dict[str, Any]) -> str | None:
    """``None`` when every key agrees bit for bit; otherwise the first disagreement, described."""
    missing = [k for k in expected if k not in actual]
    extra = [k for k in actual if k not in expected]
    if missing or extra:
        return f"keys differ: missing {missing}, new {extra}"
    for key, want in expected.items():
        got = actual[key]
        if isinstance(want, torch.Tensor) or isinstance(got, torch.Tensor):
            if not (isinstance(want, torch.Tensor) and isinstance(got, torch.Tensor)):
                return f"{key}: a {type(want).__name__} became a {type(got).__name__}"
            if want.dtype != got.dtype or want.shape != got.shape:
                return (f"{key}: {want.dtype}{tuple(want.shape)} became "
                        f"{got.dtype}{tuple(got.shape)}")
            a, b = _bits(want), _bits(got)
            if not torch.equal(a, b):
                wide = torch.complex128 if want.is_complex() else torch.float64
                diff = (want.to(wide) - got.to(wide)).abs()
                return (f"{key}: differs in {int((a != b).sum())} of {a.numel()} words, "
                        f"max abs diff {float(diff.max()):.6g}")
        elif want != got:
            return f"{key}: {want!r} became {got!r}"
    return None


def built(name: str) -> torch.nn.Module:
    """The model as its builder makes it, before any stored state is loaded."""
    return MODELS[name]()


def model(name: str) -> torch.nn.Module:
    """A fresh model ``name`` in eval mode, carrying the stored parameters when they are set."""
    m = built(name)
    if name in _STATES:
        m.load_state_dict(_STATES[name], strict=True)
    return m.eval()


def _raises(fn: Callable[[], Any]) -> Dict[str, Any]:
    """``fn()``'s outcome when it is expected to raise: the exception's type and message."""
    try:
        fn()
    except Exception as exc:                                    # noqa: BLE001 -- recorded, not handled
        return {"raises": f"{type(exc).__name__}: {exc}"}
    return {"raises": None}


def _np(a) -> torch.Tensor:
    """A numpy array as a tensor of the same dtype (a copy)."""
    return torch.from_numpy(np.array(a, copy=True))


# ---------------------------------------------------------------------------
# inputs, from torch and numpy primitives only
# ---------------------------------------------------------------------------

#: The rollout grid: even, odd and even axes, non-cubic (``Gzr = 3``).
GRID = (6, 5, 4)
#: Two boxes per batch, reduced units and Angstrom alike.
BOXES = torch.tensor([[6.0, 5.5, 4.5], [7.0, 6.25, 5.0]])
#: The mean density of each channel, by channel count.
BASE = {1: (0.45,), 2: (0.40, 0.30)}
def _domain():
    """A two-channel trapezoid that the fields here straddle, so the domain projection acts."""
    from aipf.system import TrustDomain
    return TrustDomain(inner=(0.5, 0.4), outer=(1.0, 0.6))


def _real_field(n: int, grid=GRID, batch: int = 2, *, seed: int, amp: float = 0.03):
    """``(batch, n, *grid)`` density around :data:`BASE`, a seeded Gaussian perturbation."""
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor(BASE[n]).view(1, n, 1, 1, 1)
    return base + amp * torch.randn(batch, n, *grid, generator=g)


def _hat(rho: torch.Tensor) -> torch.Tensor:
    """The state convention ``rfftn(rho) / N`` with torch's own FFT."""
    N = rho.shape[-3] * rho.shape[-2] * rho.shape[-1]
    return torch.fft.rfftn(rho, dim=(-3, -2, -1)) / N


# ---------------------------------------------------------------------------
# the models: one per built-in functional form and channel count, sized to run in milliseconds
# ---------------------------------------------------------------------------

def _seeded(seed: int, build: Callable[[], torch.nn.Module]) -> torch.nn.Module:
    with torch.random.fork_rng(devices=()):
        torch.manual_seed(seed)
        return build()


@model_builder("landau_n1")
def _landau_n1():
    from aipf.functional.landau import Landau
    return _seeded(1, lambda: Landau(GRID, 1, a0=1.0, b=1.0, T_c=1.5, gamma=1.0,
                                     nyquist_mask=True))


@model_builder("landau_n2")
def _landau_n2():
    from aipf.functional.landau import Landau
    return _seeded(2, lambda: Landau(GRID, 2, a0=1.3, b=0.7, T_c=1.4, gamma=0.8,
                                     nyquist_mask=False))


@model_builder("sg_poly_n2")
def _sg_poly_n2():
    from aipf.functional.square_gradient import SquareGradient
    m = _seeded(3, lambda: SquareGradient(GRID, 2, local="quadratic_quartic",
                                          nyquist_mask=True, kappa_init=0.7))
    with torch.no_grad():    # a non-trivial bulk, so the cross terms are read
        m._A_raw.copy_(torch.tensor([[-0.4, 0.1], [0.1, -0.3]]))
        m._b_raw.copy_(torch.tensor([0.2, -0.1]))
        m._m_raw.copy_(torch.tensor([0.3, -0.2]))
    return m


@model_builder("sg_landau_n1")
def _sg_landau_n1():
    from aipf.functional.square_gradient import SquareGradient
    return _seeded(4, lambda: SquareGradient(
        GRID, 1, local="landau", nyquist_mask=False, kappa_init=0.5, a0=1.0, b=1.0, T_c=1.5,
        rho_eps=1e-4, kB=1.0, mobility_prefactor="mole_fraction",
        mobility_shape="lattice_scalar", mobility_t_form="arrhenius",
        mobility_shape_init=math.log(0.1), mobility_t_ref=1.65,
        mobility_activation_energy_init=math.log(math.expm1(0.55))))


def _nk_gas(n: int, seed: int):
    from aipf.functional.nonlocal_kernel import NonlocalKernel
    return _seeded(seed, lambda: NonlocalKernel(
        GRID, n, EV_KB, rho_ref=(0.35, 0.33)[:n], h_g=4, R_cut=2.5,
        kernel_n_quad=33, kernel_n_k_table=17, kernel_k_table_max=8.0,
        mobility_prefactor="partial_density", mobility_shape="mlp_rho",
        mobility_t_form="none", ideal_form="gas", kBT_ref=EV_KB * 8000.0, h_u=4,
        nyquist_mask=True, f_exc_form="split", u_form="mlp", g_exc_form="mlp",
        enable_TlnT=True, h_g_hat=4, h_g_tilde=4,
        tbasis_ortho_window=(EV_KB * 9000.0, EV_KB * 12000.0), tbasis_ortho_points=5,
        rho_eps=1e-5, kernel_hidden=4, mobility_hidden=4, mobility_input_ref=None,
        mobility_t_ref=10000.0))


@model_builder("nk_gas_n1")
def _nk_gas_n1():
    return _nk_gas(1, 5)


@model_builder("nk_gas_n2")
def _nk_gas_n2():
    return _nk_gas(2, 6)


@model_builder("nk_joint_n2")
def _nk_joint_n2():
    from aipf.functional.nonlocal_kernel import NonlocalKernel
    return _seeded(7, lambda: NonlocalKernel(
        GRID, 2, EV_KB, rho_ref=(0.35, 0.33), h_g=4, R_cut=2.5,
        kernel_n_quad=33, kernel_n_k_table=17, kernel_k_table_max=8.0,
        mobility_prefactor="partial_density", mobility_shape="mlp_rho",
        mobility_t_form="arrhenius", ideal_form="gas", kBT_ref=EV_KB * 8000.0, h_u=4,
        nyquist_mask=True, f_exc_form="joint", h_joint=6, joint_depth=2, u_form="mlp",
        g_exc_form="icnn", local_input_scale=True, rho_eps=1e-5, kernel_hidden=4,
        mobility_hidden=4, mobility_input_ref=(0.35, 0.33), mobility_t_ref=9000.0,
        mobility_activation_energy_init=(-0.7, -0.4)))


@model_builder("nk_lattice_n1")
def _nk_lattice_n1():
    from aipf.functional.nonlocal_kernel import NonlocalKernel
    return _seeded(8, lambda: NonlocalKernel(
        GRID, 1, 1.0, rho_ref=(0.5,), h_g=4, R_cut=2.0, kernel_n_quad=17,
        kernel_evaluator="lattice_sum", mobility_prefactor="mole_fraction",
        mobility_shape="lattice_scalar", mobility_t_form="arrhenius",
        mobility_shape_init=math.log(0.1), mobility_t_ref=1.65,
        mobility_activation_energy_init=math.log(math.expm1(0.55)),
        ideal_form="lattice", u_form="taylor", u_degree=8, u_parity="even",
        u_variable="difference", g_exc_form="icnn", g_symmetry="mirror",
        icnn_output_bias=False, kernel_argument="difference", rho_eps=1e-4,
        kernel_hidden=4, nyquist_mask=False))


@model_builder("nop_n1")
def _nop_n1():
    from aipf.functional.neural_operator import NeuralOperator
    return _seeded(9, lambda: NeuralOperator(GRID, 1, n_layers=2, hidden=4, nyquist_mask=True))


@model_builder("nop_n2")
def _nop_n2():
    from aipf.functional.neural_operator import NeuralOperator
    return _seeded(10, lambda: NeuralOperator(GRID, 2, n_layers=2, hidden=4,
                                              nyquist_mask=False))


@model_builder("radial_n2")
def _radial_n2():
    from aipf.functional.kernels import QuinticEnvelope, RadialKernelSet
    return _seeded(11, lambda: RadialKernelSet(2, QuinticEnvelope(2.0), hidden=4))


@model_builder("radial_tail_n1")
def _radial_tail_n1():
    from aipf.functional.kernels import QuinticEnvelope, RadialKernelSet
    return _seeded(12, lambda: RadialKernelSet(1, QuinticEnvelope(2.0, tail_sigma=0.7),
                                               hidden=4, activation="tanh"))


#: Per model: its channel count, temperature, thermal energy of the noise, time step, and whether
#: its mobility is a full ``(B, n, n, ...)`` matrix (the noise, ``v_ext``, ``T_field`` and
#: ``kappa_roll`` read it as one; a diagonal ``(B, n, ...)`` mobility runs the plain drift only).
ROLL = {
    "landau_n1": dict(n=1, T=1.2, kBT=0.0, dt=2e-2, full_M=False),
    "landau_n2": dict(n=2, T=1.2, kBT=0.0, dt=2e-2, full_M=False),
    "sg_poly_n2": dict(n=2, T=1.0, kBT=0.0, dt=2e-3, full_M=False),
    "sg_landau_n1": dict(n=1, T=1.2, kBT=0.02, dt=2e-2, full_M=True),
    "nk_gas_n1": dict(n=1, T=9000.0, kBT=EV_KB * 9000.0, dt=1e-3, full_M=True),
    "nk_gas_n2": dict(n=2, T=9000.0, kBT=EV_KB * 9000.0, dt=1e-3, full_M=True),
    "nk_joint_n2": dict(n=2, T=9000.0, kBT=EV_KB * 9000.0, dt=1e-3, full_M=True),
    "nk_lattice_n1": dict(n=1, T=1.2, kBT=0.02, dt=2e-2, full_M=True),
    "nop_n1": dict(n=1, T=1.2, kBT=0.02, dt=2e-2, full_M=True),
    "nop_n2": dict(n=2, T=1.2, kBT=0.02, dt=2e-2, full_M=True),
}


# ---------------------------------------------------------------------------
# models: what each builder makes under its seed
# ---------------------------------------------------------------------------

def _model_case(name: str):
    def run():
        return {f"state/{k}": v.clone() for k, v in built(name).state_dict().items()}
    return run


for _name in list(MODELS):
    case("models", _name)(_model_case(_name))


# ---------------------------------------------------------------------------
# spectral: the operator set, the cache, and every operator on a fixed field
# ---------------------------------------------------------------------------

#: Three grids: cubic, non-cubic with an odd last axis, and a single cell.
SPECTRAL_GRIDS = ((8, 8, 8), (6, 10, 7), (1, 1, 1))
SPECTRAL_BOXES = torch.tensor([[7.0, 7.5, 8.0], [5.0, 9.0, 6.5]])


def _spectral_case(grid, nyquist_mask: bool):
    def run():
        from aipf.spectral import SpectralOps
        out: Dict[str, Any] = {}
        try:
            ops = SpectralOps(grid, 2, nyquist_mask=nyquist_mask)
        except Exception as exc:                                # noqa: BLE001
            return {"raises": f"{type(exc).__name__}: {exc}"}
        out["grid"] = tuple(ops.grid)
        out["Gzr"] = int(ops.Gzr)
        out["state_keys"] = tuple(ops.state_dict())
        for k, v in ops.state_dict().items():
            out[f"state/{k}"] = v.clone()
        for k, v in ops.named_buffers():
            out[f"buffer/{k}"] = v.clone()
        boxes = SPECTRAL_BOXES
        kx, ky, kz = ops.k_axes(boxes)
        # Stored as returned: an expanded view saves only the storage under it.
        out.update({"kx": kx, "ky": ky, "kz": kz})
        k2 = ops.k2(boxes)
        out["k2"] = k2
        out["band_mask"] = ops.band_mask(boxes, 2.0)
        out["sigma_filter"] = ops.sigma_filter(boxes, 1.3)
        g = torch.Generator().manual_seed(21)
        f = torch.randn(2, 2, *grid, generator=g)
        fh = ops.rfft(f)
        out["rfft"] = fh
        out["irfft"] = ops.irfft(fh)
        out["irfft_of_scaled"] = ops.irfft(fh * 0.37)
        gx, gy, gz = ops.grad_hat(fh, kx, ky, kz)
        out.update({"grad_x": gx, "grad_y": gy, "grad_z": gz})
        out["div"] = ops.div_hat(gx, gy, gz, kx, ky, kz)
        out["laplacian"] = ops.laplacian(fh, k2)
        return out
    return run


for _grid in SPECTRAL_GRIDS:
    for _mask in (True, False):
        case("spectral", f"ops/{'x'.join(map(str, _grid))}/nyquist_{_mask}")(
            _spectral_case(_grid, _mask))


def _cache_case(nyquist_mask: bool):
    def run():
        from aipf.spectral import OpsCache
        cache = OpsCache((8, 8, 8), 2, nyquist_mask=nyquist_mask)
        out: Dict[str, Any] = {"seed_grid": tuple(cache.ops.grid)}
        lookups = {
            "ops_for/8x8x5": lambda: cache.ops_for((3, 2, 8, 8, 5)),
            "ops_for/6x10x4": lambda: cache.ops_for((6, 10, 4)),
            "ops_for/1x1x1": lambda: cache.ops_for((1, 1, 1)),
            "ops_for_grid/6x10x7": lambda: cache.ops_for_grid((6, 10, 7)),
            "ops_for_grid/1x1x1": lambda: cache.ops_for_grid((1, 1, 1)),
            "ops_for_grid/8x8x8": lambda: cache.ops_for_grid((8, 8, 8)),
        }
        for key, look in lookups.items():
            try:
                ops = look()
            except Exception as exc:                            # noqa: BLE001
                out[f"{key}/raises"] = f"{type(exc).__name__}: {exc}"
                continue
            out[f"{key}/grid"] = tuple(ops.grid)
            out[f"{key}/is_seed"] = ops is cache.ops
            out[f"{key}/cached"] = look() is ops
            out[f"{key}/nyquist_mask"] = ops.nyquist_mask
            for k, v in ops.named_buffers():
                out[f"{key}/buffer/{k}"] = v.clone()
        out.update({f"bad_grid/{k}": v for k, v in
                    _raises(lambda: cache.ops_for_grid((0, 4, 4))).items()})
        return out
    return run


for _mask in (True, False):
    case("spectral", f"cache/nyquist_{_mask}")(_cache_case(_mask))


# ---------------------------------------------------------------------------
# kernels: the radial transform, its table, kappa_eff and W_hat(0)
# ---------------------------------------------------------------------------

@case("kernels", "radial_fourier_transform")
def _rft():
    from aipf.functional.kernels import radial_fourier_transform
    g = torch.Generator().manual_seed(31)
    r = torch.linspace(0.0, 2.0, 17)
    w = torch.randn(3, 17, generator=g)
    k = torch.linspace(0.0, 6.0, 12).reshape(3, 4)
    return {"w_hat": radial_fourier_transform(w, r, k),
            "w_hat_flat_k": radial_fourier_transform(w[0], r, torch.tensor([0.0, 0.5, 9.0]))}


def _analytic_case(radial: str, R_cut: float, n_quad: int, n_k: int, k_max: float):
    def run():
        from aipf.functional.kernels import (AnalyticRadialTransform, PairKernel,
                                             radial_fourier_transform)
        rs = model(radial)
        ev = AnalyticRadialTransform(R_cut, n_quad, n_k, k_max)
        pk = PairKernel(rs, ev, n_quad)
        out: Dict[str, Any] = {"r_quad": ev.r_quad.clone(), "k_table": ev.k_table.clone()}
        out["state_keys"] = tuple(ev.state_dict())
        with torch.no_grad():
            out["table"] = radial_fourier_transform(rs.w_of_r(ev.r_quad), ev.r_quad,
                                                    ev.k_table)
            k = torch.linspace(0.0, 1.3 * k_max, 24).reshape(2, 3, 4)   # past the table's end too
            out["w_hat"] = ev.w_hat(rs, k)
            out["pair_w_hat"] = pk.w_hat(k)
            out["w_of_r"] = rs.w_of_r(torch.linspace(0.0, 1.2 * R_cut, 9))
            out["matrix_of_r"] = rs.matrix_of_r(torch.linspace(0.0, R_cut, 5))
        out["kappa_eff"] = pk.kappa_eff()
        out["w_hat_zero_radial"] = pk.w_hat_zero_radial(R_cut, 65)
        q = pk.w_hat_zero_quadrature(R_cut, 33)
        out["w_hat_zero_quadrature"] = q.detach()
        grads = torch.autograd.grad(q.sum(), list(rs.parameters()))
        for (name, _), gr in zip(rs.named_parameters(), grads):
            out[f"w_hat_zero_grad/{name}"] = gr
        out["kappa_eff_without_quadrature"] = _raises(
            lambda: PairKernel(rs, ev, None).kappa_eff())["raises"]
        return out
    return run


case("kernels", "analytic/n2")(_analytic_case("radial_n2", 2.0, 33, 17, 8.0))
case("kernels", "analytic/tail_n1")(_analytic_case("radial_tail_n1", 2.0, 64, 9, 5.0))


@case("kernels", "lattice_sum/n2")
def _lattice_sum():
    from aipf.functional.kernels import LatticeSumTransform, PairKernel
    rs = model("radial_n2")
    ev = LatticeSumTransform()
    pk = PairKernel(rs, ev, 17)
    k = torch.zeros(2, 1)
    with torch.no_grad():
        out = {"w_hat": pk.w_hat(k, grid=(6, 5, 4), boxes=BOXES),
               "distances": ev.distances((6, 5, 4), (1.0, 1.1, 1.125), "cpu", torch.float32),
               "spacing": ev.spacing((6, 5, 4), BOXES[1], torch.float32)}
    out["bare_k"] = _raises(lambda: pk.w_hat(k))["raises"]
    return out


@case("kernels", "model_kernels")
def _model_kernels():
    out: Dict[str, Any] = {}
    for name in ("nk_gas_n2", "nk_joint_n2", "nk_gas_n1"):
        m = model(name)
        out[f"{name}/kappa_eff"] = m.kernel.kappa_eff()
        out[f"{name}/w_hat_zero_radial"] = m.kernel.w_hat_zero_radial(2.5, 101)
        with torch.no_grad():
            out[f"{name}/w_hat"] = m.kernel.w_hat(torch.linspace(0.0, 4.0, 7))
    return out


# ---------------------------------------------------------------------------
# projection: hermitianize, the state projections and the trust-domain projection
# ---------------------------------------------------------------------------

def _wide_hat(n: int, seed: int):
    """A field wide enough that every projection below acts (cells below the floors, outside the box)."""
    return _hat(_real_field(n, seed=seed, amp=0.12))


@case("projection", "hermitianize")
def _herm():
    from aipf.solve import hermitianize
    from aipf.spectral import SpectralOps
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    g = torch.Generator().manual_seed(41)
    raw = torch.randn(2, 2, 6, 5, 3, dtype=torch.complex64, generator=g)
    return {"out": hermitianize(raw, ops),
            "of_a_real_field": hermitianize(_wide_hat(2, 42), ops)}


def _projection_case(n: int, state_proj: str, **kw):
    def run():
        from aipf.solve import project_state
        from aipf.solve.declare import UNDECLARED
        from aipf.spectral import SpectralOps
        ops = SpectralOps(GRID, n, nyquist_mask=True)
        h = _wide_hat(n, 43 + n)
        rest = dict(kw)
        floor = rest.pop("floor", UNDECLARED)
        if state_proj == "domain":
            rest["domain"] = _domain()
        out = project_state(h, ops, floor, state_proj=state_proj, **rest)
        return {"out": out, "unchanged": out is h}
    return run


case("projection", "project_state/floor_n1")(_projection_case(1, "floor", floor=0.4))
case("projection", "project_state/floor_n2")(_projection_case(2, "floor", floor=0.3))
case("projection", "project_state/floor_zero")(_projection_case(2, "floor", floor=0.0))
case("projection", "project_state/box_scalar")(_projection_case(1, "box", lo=0.38, hi=0.52))
case("projection", "project_state/box_per_channel")(
    _projection_case(2, "box", lo=[0.33, 0.24], hi=[0.47, 0.36]))
case("projection", "project_state/domain")(_projection_case(2, "domain", lo=0.2))


def _shift_case(n: int, state_proj: str, floor: float):
    def run():
        from aipf.solve.projection import project_state_uniform_shift
        from aipf.spectral import SpectralOps
        ops = SpectralOps(GRID, n, nyquist_mask=True)
        h = _wide_hat(n, 47 + n)
        out = project_state_uniform_shift(
            h, ops, state_proj=state_proj, floor=floor,
            domain=_domain() if state_proj == "domain" else None)
        return {"out": out}
    return run


case("projection", "uniform_shift/floor")(_shift_case(1, "floor", 0.4))
case("projection", "uniform_shift/domain")(_shift_case(2, "domain", 1e-3))


@case("projection", "restore_mass_and_trust_domain")
def _restore():
    from aipf.solve import in_domain, project_trust_domain
    from aipf.solve.projection import restore_mass
    rho = _real_field(2, seed=51, amp=0.12)
    target = torch.tensor([[0.41, 0.29], [0.39, 0.31]])
    flat = rho.permute(0, 2, 3, 4, 1).reshape(-1, 2)
    return {"restore_scalar_lo": restore_mass(rho, target, torch.tensor(0.3)),
            "restore_drain": restore_mass(rho, torch.tensor([[0.1, 0.1], [0.1, 0.1]]),
                                          torch.tensor(0.25)),
            "in_domain": in_domain(flat, _domain()),
            "projected": project_trust_domain(flat, _domain())}


# ---------------------------------------------------------------------------
# solve.noise: the colour filter and the declarations read off a system
# ---------------------------------------------------------------------------

@case("noise", "solve_noise")
def _solve_noise():
    from aipf.solve import build_noise_filter, declared_noise
    from aipf.spectral import SpectralOps
    from aipf.system import load
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    out: Dict[str, Any] = {"gaussian": build_noise_filter(ops, BOXES, "gaussian", 1.5),
                           "none": build_noise_filter(ops, BOXES, "none", None)}
    for system in ("lj", "hhe"):
        for key, value in declared_noise(load(system)).items():
            out[f"{system}/{key}"] = value
    return out


@case("noise", "rollout_noise")
def _rollout_noise():
    from aipf.rollout.noise import (cell_mobilities, draw_w, max_norm_mobility,
                                    noise_div_hat, zeta_from_w)
    from aipf.solve import build_noise_filter
    from aipf.spectral import SpectralOps
    m = model("nk_gas_n2")
    ops = SpectralOps(GRID, 2, nyquist_mask=True)
    rho = _real_field(2, batch=1, seed=61)
    T = torch.tensor([9000.0])
    g = torch.Generator().manual_seed(62)
    with torch.no_grad():
        w = draw_w(GRID, 2, g, "cpu", torch.float32)
        zeta = zeta_from_w(m, rho, T, 0.8, w)
        kx, ky, kz = ops.k_axes(BOXES[:1])
        filt = build_noise_filter(ops, BOXES[:1], "gaussian", 1.5)
        return {"w": w, "generator_state": g.get_state(), "zeta": zeta,
                "div_filtered": noise_div_hat(ops, zeta, kx, ky, kz, filt),
                "div_white": noise_div_hat(ops, zeta, kx, ky, kz, None),
                "cell_mobilities": cell_mobilities(m, rho, T),
                "max_norm_mobility": max_norm_mobility(m, rho, T)}


# ---------------------------------------------------------------------------
# integrators: euler, heun, the two rollouts, on every form
# ---------------------------------------------------------------------------

def _drive(name: str, *, seed: int):
    """``v_ext`` ``(B, n, *grid)`` and ``T_field`` ``(B, *grid)`` for model ``name``, seeded."""
    spec = ROLL[name]
    g = torch.Generator().manual_seed(seed)
    v = 0.05 * torch.randn(2, spec["n"], *GRID, generator=g)
    tf = spec["T"] * (1.0 + 0.1 * torch.rand(2, *GRID, generator=g))
    if spec["T"] > 100.0:
        v = v * EV_KB * spec["T"]
    return v, tf


def _projection_kw(name: str, proj: str) -> Dict[str, Any]:
    n = ROLL[name]["n"]
    base = BASE[n]
    if proj == "floor":
        return dict(state_proj="floor", floor=min(base) - 0.02)
    if proj == "box":
        lo = [b - 0.04 for b in base]
        hi = [b + 0.04 for b in base]
        return dict(state_proj="box", lo=lo if n > 1 else lo[0], hi=hi if n > 1 else hi[0])
    if proj == "domain":
        return dict(state_proj="domain", domain=_domain(), lo=0.05)
    raise ValueError(proj)


def _rollout_case(name: str, kind: str, proj: str, *, steps: int = 3, save_every: int = 1,
                  clamp_rho=None, kappa_roll: float = 0.0, v_ext: bool = False,
                  T_field: bool = False, noise_mode: str = "gaussian", kBT=None, seed: int = 0):
    def run():
        from aipf.solve import rollout_deterministic, rollout_sde
        spec = ROLL[name]
        m = model(name)
        n = spec["n"]
        h0 = _hat(_real_field(n, seed=100 + seed))
        T = torch.full((2,), spec["T"])
        v, tf = _drive(name, seed=200 + seed)
        kw = dict(clamp_rho=clamp_rho, kappa_roll=kappa_roll, save_every=save_every,
                  v_ext=v if v_ext else None, T_field=tf if T_field else None,
                  **_projection_kw(name, proj))
        out: Dict[str, Any] = {}
        if kind in ("euler", "heun"):
            out["traj"] = rollout_deterministic(m, h0, BOXES, T, spec["dt"], steps,
                                                method=kind, **kw)
        else:
            g = torch.Generator().manual_seed(300 + seed)
            out["traj"] = rollout_sde(
                m, h0, BOXES, T, spec["dt"], steps,
                spec["kBT"] if kBT is None else kBT, noise_mode=noise_mode,
                sigma_noise=1.5 if noise_mode == "gaussian" else None, generator=g, **kw)
            out["generator_state"] = g.get_state()
        return out
    return run


def _step_case(name: str, *, seed: int):
    def run():
        from aipf.solve import step_euler, step_heun, step_sde_euler_maruyama
        from aipf.spectral import SpectralOps
        spec = ROLL[name]
        m = model(name)
        n = spec["n"]
        ops = SpectralOps(GRID, n, nyquist_mask=m.ops.nyquist_mask)
        h0 = _hat(_real_field(n, seed=400 + seed))
        T = torch.full((2,), spec["T"])
        out = {"euler": step_euler(m, h0, BOXES, T, spec["dt"]),
               "euler_ops": step_euler(m, h0, BOXES, T, spec["dt"], ops=ops),
               "heun": step_heun(m, h0, BOXES, T, spec["dt"], ops=ops),
               "forward": m(h0, BOXES, T)}
        if spec["full_M"]:
            g = torch.Generator().manual_seed(500 + seed)
            out["sde"] = step_sde_euler_maruyama(m, h0, BOXES, T, spec["dt"], spec["kBT"],
                                                 noise_mode="gaussian", sigma_noise=1.5,
                                                 ops=ops, generator=g)
            out["sde_generator_state"] = g.get_state()
        return {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
    return run


for _i, _name in enumerate(ROLL):
    case("integrators", f"{_name}/steps")(_step_case(_name, seed=_i))
    case("integrators", f"{_name}/euler_floor")(_rollout_case(_name, "euler", "floor", seed=_i))
    case("integrators", f"{_name}/heun_box_clamped")(
        _rollout_case(_name, "heun", "box", clamp_rho=1e-3, seed=_i))
    if ROLL[_name]["full_M"]:
        case("integrators", f"{_name}/sde_floor")(_rollout_case(_name, "sde", "floor", seed=_i))

# The drive, the stabiliser and the remaining projections, on the forms that read a full mobility.
_EXTRA = {
    "nk_gas_n2/euler_kappa_roll_v_ext_T_field": ("nk_gas_n2", "euler", "floor",
                                                  dict(kappa_roll=0.3, v_ext=True, T_field=True)),
    "nk_gas_n2/heun_v_ext_domain": ("nk_gas_n2", "heun", "domain", dict(v_ext=True)),
    "nk_gas_n2/sde_all_drives_domain": ("nk_gas_n2", "sde", "domain",
                                        dict(kappa_roll=0.3, v_ext=True, T_field=True,
                                             clamp_rho=1e-3)),
    "nk_gas_n2/sde_white": ("nk_gas_n2", "sde", "box", dict(noise_mode="none")),
    "nk_gas_n2/sde_zero_kbt": ("nk_gas_n2", "sde", "floor", dict(kBT=0.0)),
    "nk_gas_n2/euler_save_every_2": ("nk_gas_n2", "euler", "floor",
                                     dict(steps=5, save_every=2)),
    "nk_gas_n2/sde_save_every_2": ("nk_gas_n2", "sde", "floor", dict(steps=5, save_every=2)),
    "nk_joint_n2/sde_T_field": ("nk_joint_n2", "sde", "domain", dict(T_field=True)),
    "nk_lattice_n1/euler_kappa_roll_T_field": ("nk_lattice_n1", "euler", "box",
                                               dict(kappa_roll=0.5, T_field=True)),
    "nk_lattice_n1/sde_v_ext_clamped": ("nk_lattice_n1", "sde", "floor",
                                        dict(v_ext=True, clamp_rho=1e-3)),
    "sg_landau_n1/sde_v_ext": ("sg_landau_n1", "sde", "box", dict(v_ext=True)),
    "sg_landau_n1/heun_T_field_kappa_roll": ("sg_landau_n1", "heun", "floor",
                                             dict(T_field=True, kappa_roll=0.2)),
    "nop_n2/euler_domain_kappa_roll": ("nop_n2", "euler", "domain", dict(kappa_roll=0.2)),
    "nop_n2/sde_v_ext_white": ("nop_n2", "sde", "floor", dict(v_ext=True, noise_mode="none")),
    "nop_n1/heun_v_ext": ("nop_n1", "heun", "box", dict(v_ext=True)),
}
for _j, (_key, (_name, _kind, _proj, _kw)) in enumerate(_EXTRA.items()):
    case("integrators", _key)(_rollout_case(_name, _kind, _proj, seed=50 + _j, **_kw))


# ---------------------------------------------------------------------------
# imex: the semi-implicit scheme, deterministic and noisy
# ---------------------------------------------------------------------------

IMEX_BOX = torch.tensor([7.0, 7.5, 8.0])


def _imex_case(name: str, *, m_stab: str, state_proj: str = "floor", mass_restore: str = "shift",
               clamp_rho=1e-3, noise_eval=None, kbt_field: bool = False, v_ext: bool = False,
               noise_mode: str = "gaussian", steps: int = 3, save_every: int = 1, seed: int = 0):
    def run():
        from aipf.rollout.imex import rollout_imex
        m = model(name)
        n = ROLL[name]["n"]
        h0 = _hat(_real_field(n, batch=1, seed=600 + seed))
        T = 9000.0
        g = torch.Generator().manual_seed(700 + seed)
        fields: Dict[str, Any] = {}
        if kbt_field:
            fields["kbt_field"] = EV_KB * T * (1.0 + 0.1 * torch.rand(GRID, generator=g))
        if v_ext:
            fields["v_ext"] = 0.02 * torch.randn(n, *GRID, generator=g)
        noise = None
        if noise_eval is not None:
            noise = dict(kBT_noise=EV_KB * T, noise_scale=1.0, noise_mode=noise_mode,
                         sigma_noise=1.5 if noise_mode == "gaussian" else None,
                         noise_eval=noise_eval, predictor_floor=1e-4)
        if state_proj == "domain":
            proj = dict(state_proj="domain", state_clamp=1e-3, domain=_domain())
        else:
            proj = dict(state_proj="floor", state_clamp=min(BASE[n]) - 0.02)
        gen = torch.Generator().manual_seed(800 + seed)
        traj = rollout_imex(m, h0, IMEX_BOX, T, 1e-3, steps, kB=EV_KB, m_stab=m_stab,
                            mass_restore=mass_restore, clamp_rho=clamp_rho, noise=noise,
                            save_every=save_every, generator=gen, **proj, **fields)
        return {"traj": traj, "generator_state": gen.get_state()}
    return run


_IMEX: Dict[str, Dict[str, Any]] = {}
for _ms in ("mean", "max"):
    for _mr in ("shift", "headroom"):
        _IMEX[f"nk_gas_n2/det/{_ms}/{_mr}/floor"] = dict(name="nk_gas_n2", m_stab=_ms,
                                                          mass_restore=_mr)
        _IMEX[f"nk_gas_n2/det/{_ms}/{_mr}/domain"] = dict(name="nk_gas_n2", m_stab=_ms,
                                                           mass_restore=_mr, state_proj="domain")
        _IMEX[f"nk_gas_n1/det/{_ms}/{_mr}/floor"] = dict(name="nk_gas_n1", m_stab=_ms,
                                                          mass_restore=_mr)
    for _ev in ("ito", "midpoint", "kinetic"):
        for _kf in (False, True):
            _IMEX[f"nk_gas_n2/noisy/{_ev}/{_ms}/kbt_field_{_kf}"] = dict(
                name="nk_gas_n2", m_stab=_ms, noise_eval=_ev, kbt_field=_kf,
                state_proj="domain")
        _IMEX[f"nk_gas_n1/noisy/{_ev}/{_ms}"] = dict(name="nk_gas_n1", m_stab=_ms,
                                                     noise_eval=_ev)
_IMEX.update({
    "nk_gas_n2/det/unclamped": dict(name="nk_gas_n2", m_stab="max", clamp_rho=None),
    "nk_gas_n2/det/v_ext": dict(name="nk_gas_n2", m_stab="max", v_ext=True),
    "nk_gas_n2/det/v_ext_kbt_field": dict(name="nk_gas_n2", m_stab="mean", v_ext=True,
                                          kbt_field=True, state_proj="domain"),
    "nk_gas_n2/noisy/unclamped_white": dict(name="nk_gas_n2", m_stab="max", clamp_rho=None,
                                            noise_eval="midpoint", noise_mode="none"),
    "nk_gas_n2/noisy/v_ext_kbt_field_headroom": dict(
        name="nk_gas_n2", m_stab="max", noise_eval="kinetic", v_ext=True, kbt_field=True,
        mass_restore="headroom", state_proj="domain"),
    "nk_gas_n2/noisy/save_every_2": dict(name="nk_gas_n2", m_stab="max", noise_eval="ito",
                                         steps=5, save_every=2),
    "nk_gas_n1/noisy/v_ext_kbt_field_unclamped": dict(
        name="nk_gas_n1", m_stab="mean", noise_eval="ito", v_ext=True, kbt_field=True,
        clamp_rho=None),
    "nk_joint_n2/noisy/midpoint": dict(name="nk_joint_n2", m_stab="max", noise_eval="midpoint",
                                       state_proj="domain"),
})
for _j, (_key, _kw) in enumerate(_IMEX.items()):
    _kw = dict(_kw)
    case("imex", _key)(_imex_case(_kw.pop("name"), seed=_j, **_kw))


@case("imex", "local_hessian")
def _local_hessian():
    from aipf.rollout.imex import local_hessian
    return {"n2": local_hessian(model("nk_gas_n2"), torch.tensor([0.40, 0.30]), EV_KB * 9000.0),
            "n1": local_hessian(model("nk_gas_n1"), torch.tensor([0.45]), EV_KB * 9000.0)}


# ---------------------------------------------------------------------------
# pipeline: the mode set, the reference box and the exact amplitudes of a synthetic trajectory
# ---------------------------------------------------------------------------

Frame = namedtuple("Frame", "positions types box_bounds timestep")

#: The synthetic trajectory: two dump types, a breathing box, atoms on a seeded random walk.
N_FRAMES = 14
EDGES = np.array([7.0, 7.5, 8.0])


def synthetic_frames(seed: int = 71, n_atoms: int = 120):
    rng = np.random.default_rng(seed)
    types = np.where(np.arange(n_atoms) % 3 == 0, 2, 1)
    s = rng.random((n_atoms, 3))
    frames = []
    for t in range(N_FRAMES):
        lengths = EDGES * (1.0 + 0.01 * np.sin(0.7 * t + np.arange(3)))
        lo = np.array([-0.5, 0.25, 0.0]) + 0.01 * t
        s = np.mod(s + 0.02 * rng.standard_normal(s.shape), 1.0)
        bounds = np.stack([lo, lo + lengths], axis=1)
        frames.append(Frame(lo + s * lengths, types, bounds, 1000 * t))
    return frames


def _box_lengths(frames):
    return np.array([f.box_bounds[:, 1] - f.box_bounds[:, 0] for f in frames])


@case("pipeline", "extract_modes")
def _extract_modes():
    from aipf.pipeline.extract_modes import (frames_to_skip, mode_amplitudes, mode_series,
                                             mode_set, reference_box, wavevectors)
    frames = synthetic_frames()
    lengths = _box_lengths(frames)
    out: Dict[str, Any] = {}
    for rule in ("time_mean", "first_frame"):
        out[f"reference_box/{rule}"] = _np(reference_box(lengths, rule=rule))
    ref = reference_box(lengths, rule="time_mean")
    for ordering in ("lexicographic", "shell"):
        out[f"mode_set/{ordering}"] = _np(mode_set(ref, k_cut=2.5, ordering=ordering))
    labels = mode_set(ref, k_cut=2.5, ordering="shell")
    out["wavevectors/one_box"] = _np(wavevectors(labels, ref))
    out["wavevectors/timeline"] = _np(wavevectors(labels, lengths[:3]))
    f0 = frames[0]
    out["amplitudes/separable"] = _np(mode_amplitudes(
        f0.positions, f0.types, f0.box_bounds, labels, atom_types=(1, 2), route="separable"))
    out["amplitudes/dense_cpu"] = _np(mode_amplitudes(
        f0.positions, f0.types, f0.box_bounds, labels, atom_types=(2, 1), route="dense",
        device="cpu"))
    amps, steps = mode_series(frames[:4], labels, atom_types=(1, 2), route="separable")
    out["series/amplitudes"] = _np(amps)
    out["series/steps"] = _np(steps)
    out["frames_to_skip"] = frames_to_skip(np.arange(0, 14000, 1000), equilibration_steps=3000,
                                           production_steps=10000)
    return out


@case("pipeline", "combine_fields")
def _combine_fields():
    from aipf.pipeline.extract_modes import mode_series, mode_set, reference_box
    from aipf.pipeline.modes import check_fields, combine_fields
    frames = synthetic_frames()[:5]
    lengths = _box_lengths(frames)
    labels = mode_set(reference_box(lengths, rule="first_frame"), k_cut=2.0,
                      ordering="lexicographic")
    amps, _ = mode_series(frames, labels, atom_types=(1, 2), route="separable")
    per_type = np.moveaxis(amps, 1, 2)                       # (frames, modes, types)
    fields = check_fields(({"name": "A", "weights": {1: 0.5, 2: -0.5}, "mean": 0.5},
                           {"name": "B", "weights": {2: 1.0, 1: 0.25}, "mean": None}),
                          species=("A", "B"))
    return {"combined": _np(combine_fields(per_type, labels, lengths, fields, (1, 2))),
            "combined_c128": _np(combine_fields(per_type.astype(np.complex128), labels, lengths,
                                                fields, (1, 2)))}


# ---------------------------------------------------------------------------
# train: the archive a dataset reads, its windows, the run weights, the drift loss and one step
# ---------------------------------------------------------------------------

TRAIN_GRID = (6, 6, 6)


def write_archive(root: Path) -> None:
    """Three runs of the synthetic trajectory's exact amplitudes, as :mod:`aipf.pipeline` stores them:
    one with a breathing box, one with a fixed box, one cut short by its quality sidecar."""
    import json

    from aipf.pipeline.extract_modes import mode_series, mode_set, reference_box
    for i, (tag, fixed, quality) in enumerate((("run_a", False, None), ("run_b", True, None),
                                               ("run_c", False, "0..10"))):
        frames = synthetic_frames(seed=81 + i, n_atoms=90 + 15 * i)
        lengths = _box_lengths(frames)
        labels = mode_set(reference_box(lengths, rule="time_mean"), k_cut=2.5, ordering="shell")
        amps, _ = mode_series(frames, labels, atom_types=(1, 2), route="separable")
        d = root / tag
        d.mkdir()
        payload = dict(a=np.moveaxis(amps, 1, 2), n=labels,
                       b=lengths[0] if fixed else lengths, T=np.float64(9000.0 + 500 * i),
                       dt=np.float64(0.25), x_A=np.float64(0.6 + 0.05 * i),
                       x_B=np.float64(0.4 - 0.05 * i))
        np.savez(d / "m.npz", **payload)
        if quality is not None:
            (d / "quality.json").write_text(json.dumps({"frames": quality}))


def _keys():
    from aipf.train.dataset import ArchiveKeys
    return ArchiveKeys(file_name="m.npz", amplitudes="a", amplitudes_channel_axis=2, labels="n",
                       box="b", temperature="T", frame_interval="dt",
                       composition=("x_A", "x_B"), composition_fallback=None,
                       quality_file="quality.json", quality_key="frames")


def _settings(estimator: str, band_k_max=None):
    from aipf.train.dataset import WindowSettings
    savgol = estimator == "savgol"
    return WindowSettings(estimator=estimator, half_width=3, n_states=3, stride=2,
                          grid=TRAIN_GRID, savgol_window=5 if savgol else None,
                          savgol_poly=2 if savgol else None, band_k_max=band_k_max)


def _with_runs(fn):
    """``fn(runs)`` on the archive above, written to and read from a temporary directory."""
    from aipf.train.dataset import read_mode_runs
    with TemporaryDirectory() as tmp:
        write_archive(Path(tmp))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import logging
            logging.disable(logging.WARNING)
            try:
                runs = read_mode_runs(tmp, "run_*", keys=_keys())
            finally:
                logging.disable(logging.NOTSET)
    return fn(runs)


def _dataset_case(estimator: str, band_k_max=None, weighted: bool = False):
    def run():
        from aipf.train.dataset import ModeWindowDataset

        def body(runs):
            weights = np.array([0.5, 1.5, 1.0]) if weighted else None
            ds = ModeWindowDataset(runs, _settings(estimator, band_k_max), run_weights=weights)
            out: Dict[str, Any] = {"len": len(ds), "samples": tuple(ds.samples),
                                   "offsets": _np(ds.offsets), "lam": _np(ds.lam)}
            for r, run_ in enumerate(runs):
                out[f"run{r}/n_frames"] = run_.n_frames
                out[f"run{r}/composition"] = run_.composition
                out[f"run{r}/keep"] = None if ds.keep[r] is None else _np(ds.keep[r])
            # The first and the last window of every run.
            firsts = {}
            for i, (r, _) in enumerate(ds.samples):
                firsts.setdefault(r, []).append(i)
            for i in sorted({j for idx in firsts.values() for j in (idx[0], idx[-1])}):
                for k, v in ds[i].items():
                    out[f"item{i}/{k}"] = v
            return out
        return _with_runs(body)
    return run


case("train", "dataset/weak")(_dataset_case("weak"))
case("train", "dataset/weak_mid_band")(_dataset_case("weak_mid", band_k_max=1.6))
case("train", "dataset/savgol_weighted")(_dataset_case("savgol", weighted=True))


@case("train", "scatter_and_band")
def _scatter_and_band():
    from aipf.train.dataset import band_keep, scatter_modes, triangular_weights, weak_target
    from aipf.pipeline.extract_modes import mode_set
    labels = mode_set(EDGES, k_cut=2.5, ordering="shell")
    rng = np.random.default_rng(91)
    amps = (rng.standard_normal((4, 2, len(labels)))
            + 1j * rng.standard_normal((4, 2, len(labels)))).astype(np.complex64)
    boxes = np.array([EDGES, EDGES * 1.02, EDGES * 0.99])
    off, w = triangular_weights(4, 3)
    off_all, w_all = triangular_weights(3, 10)
    return {"scatter": _np(scatter_modes(amps, labels, float(EDGES.prod()), TRAIN_GRID)),
            "scatter_c128": _np(scatter_modes(amps.astype(np.complex128), labels, 336.0,
                                              (6, 6, 7))),
            "band_keep/timeline": _np(band_keep(labels, boxes, 1.6)),
            "band_keep/one_box": _np(band_keep(labels, EDGES, 2.0)),
            "band_keep/none": band_keep(labels, EDGES, None),
            "weak_target": _np(weak_target(amps, 1, 2, 0.25)),
            "triangular/offsets": _np(off), "triangular/weights": _np(w),
            "triangular_all/offsets": _np(off_all), "triangular_all/weights": _np(w_all),
            "too_small_grid": _raises(lambda: scatter_modes(amps, labels, 1.0, (4, 4, 4)))["raises"]}


@case("train", "run_weights")
def _run_weights():
    from aipf.train.datamodule import RunWeighting, run_weights

    def body(runs):
        out = {}
        for band in (None, 1.6):
            settings = _settings("weak", band)
            out[f"uniform/band_{band}"] = _np(run_weights(
                runs, RunWeighting("uniform", None, None, None, None), settings))
            out[f"inverse_band_power/band_{band}"] = _np(run_weights(
                runs, RunWeighting("inverse_band_power", 1.5, 2.0, 1e-6, 2), settings))
        return out
    return _with_runs(body)


@case("train", "l_dyn")
def _l_dyn():
    from aipf.losses import l_dyn
    g = torch.Generator().manual_seed(95)
    residual = torch.randn(3, 2, 6, 5, 3, dtype=torch.complex64, generator=g)
    k2 = 4.0 * torch.rand(3, 1, 6, 5, 3, generator=g)
    k2[:, :, 0, 0, 0] = 0.0
    mult = torch.tensor([1.0, 2.0, 1.0]).view(1, 1, 1, 1, 3)
    band = (k2 > 0) & (k2 <= 3.0)
    weight = torch.tensor([0.5, 1.0, 2.0])
    real = torch.randn(3, 2, 6, 5, 3, generator=g)
    return {"alpha_0.3": l_dyn(residual, k2, mult, band, alpha=0.3),
            "weighted": l_dyn(residual, k2, mult, band, alpha=0.0, sample_weight=weight),
            "real_residual": l_dyn(real, k2, mult, band, alpha=1.0, eps=1e-6)}


def _train_case(name: str):
    def run():
        from torch.utils.data import default_collate

        from aipf.spectral import OpsCache
        from aipf.train.config import TrainConfig
        from aipf.train.dataset import ModeWindowDataset
        from aipf.train.lit_module import LitModule

        def body(runs):
            ds = ModeWindowDataset(runs, _settings("weak"))
            return default_collate([ds[i] for i in (0, 3, len(ds) - 1)])
        batch = _with_runs(body)
        m = model(name).train()
        n = ROLL[name]["n"]
        if n == 1:   # one channel: the first species of the two-channel archive
            for key in ("rho_hat_states", "target_hat"):
                batch[key] = batch[key][..., :1, :, :, :]
        cfg = TrainConfig(sigma=1.5, k_max=2.2, alpha_loss=0.2, h_inv_eps=1e-6)
        lit = LitModule(m, OpsCache(TRAIN_GRID, n, nyquist_mask=m.ops.nyquist_mask), cfg)
        loss = lit.drift_loss(batch["rho_hat_states"], batch["lam"], batch["target_hat"],
                              batch["boxes"], batch["T"], sample_weight=batch["sample_weight"])
        params = [(k, p) for k, p in m.named_parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
        out: Dict[str, Any] = {"drift_loss": loss.detach()}
        for (k, _), gr in zip(params, grads):
            out[f"grad/{k}"] = gr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # self.log outside a Trainer warns and returns
            total = lit.training_step({"drift": batch}, 0)
        out["training_step"] = total.detach()
        return out
    return run


for _name in ("nk_gas_n2", "nk_lattice_n1", "nop_n2", "sg_landau_n1", "landau_n2"):
    case("train", f"drift_loss/{_name}")(_train_case(_name))
