"""A model this package does not define, declared through ``Functional(factory=...)``.

Imported by the ``system.py`` the factory tests write into a temporary experiments folder, which is
why it lives here, on the path ``conftest.py`` sets, and not in that file: ``aipf.system.load``
executes a system file outside ``sys.modules``, so the classes a factory returns come from an
importable module.

The functional is a square gradient written as a pair kernel, ``W(k) = kappa k^2`` (``kappa`` a
positive diagonal), on a polynomial local free energy
``f = sum_i [a_i rho_i^2 / 2 + b_i rho_i^4 / 4] + chi rho_0 rho_1 + kBT sum_i rho_i^2 / 2``,
with a constant, full, positive definite ``(n, n)`` mobility. It follows the contract of
docs/reference/functional.md: ``forward`` reads ``mu`` through ``self.chemical_potential``, which reads
the pointwise part through ``self.f_local.mu_pointwise``, and ``M`` through ``self.mobility``."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aipf.paths import Paths
from aipf.spectral import OpsCache
from aipf.system import AnchorRules, Functional, System, Variant

#: The toy's grid; four cubed, as the training tests' mode archives are.
GRID = (4, 4, 4)


class SquareGradientKernel(nn.Module):
    """``W(k) = kappa k^2`` per species, ``kappa = softplus(raw)``; reads no grid and no box."""

    def __init__(self, n_species: int, kappa: float):
        super().__init__()
        raw = torch.log(torch.expm1(torch.tensor(float(kappa))))
        self.kappa_raw = nn.Parameter(raw.repeat(n_species))

    def w_hat(self, k: torch.Tensor, **_geometry) -> torch.Tensor:
        """``(*k.shape, n, n)``."""
        return (k * k)[..., None, None] * torch.diag(F.softplus(self.kappa_raw)).to(k.dtype)


class PolynomialLocal(nn.Module):
    """The local free energy on flattened states: ``rho`` ``(P, n)``, ``kBT`` ``(P,)``."""

    def __init__(self, n_species: int):
        super().__init__()
        self.a_raw = nn.Parameter(torch.zeros(n_species))
        self.b_raw = nn.Parameter(torch.zeros(n_species))
        self.chi = nn.Parameter(torch.tensor(0.1))

    def f_pointwise(self, rho: torch.Tensor, kBT: torch.Tensor) -> torch.Tensor:
        a, b = F.softplus(self.a_raw), F.softplus(self.b_raw)
        f = (0.5 * a * rho ** 2 + 0.25 * b * rho ** 4).sum(-1)
        return f + self.chi * rho[:, 0] * rho[:, -1] + 0.5 * kBT * (rho ** 2).sum(-1)

    def mu_pointwise(self, rho: torch.Tensor, kBT: torch.Tensor) -> torch.Tensor:
        a, b = F.softplus(self.a_raw), F.softplus(self.b_raw)
        mu = a * rho + b * rho ** 3 + kBT.unsqueeze(-1) * rho
        cross = torch.zeros_like(mu)
        cross[:, 0] = self.chi * rho[:, -1]
        cross[:, -1] = cross[:, -1] + self.chi * rho[:, 0]
        return mu + cross


class ToyFactoryModel(nn.Module):
    """The four ``FreeEnergyModel`` methods, ``_cache``, ``ops``, ``kernel.w_hat`` and ``f_local``."""

    def __init__(self, grid, n_species: int, *, nyquist_mask: bool, kappa: float, kB: float):
        super().__init__()
        self.n_species = int(n_species)
        self.kB = float(kB)
        self._cache = OpsCache(grid, self.n_species, nyquist_mask=nyquist_mask)
        self.ops = self._cache.ops
        self.kernel = SquareGradientKernel(self.n_species, kappa)
        self.f_local = PolynomialLocal(self.n_species)
        n_lower = self.n_species * (self.n_species + 1) // 2
        self.mobility_raw = nn.Parameter(torch.zeros(n_lower))

    def mobility_matrix(self) -> torch.Tensor:
        """``L L^T + 0.1 I``, ``L`` lower triangular with a softplus diagonal."""
        n = self.n_species
        rows, cols = torch.tril_indices(n, n)
        L = torch.zeros(n, n, dtype=self.mobility_raw.dtype, device=self.mobility_raw.device)
        L = L.index_put((rows, cols), self.mobility_raw)
        L = L - torch.diag(torch.diagonal(L)) + torch.diag(F.softplus(torch.diagonal(L)))
        return L @ L.T + 0.1 * torch.eye(n, dtype=L.dtype, device=L.device)

    def _flat(self, rho: torch.Tensor) -> torch.Tensor:
        return rho.permute(0, 2, 3, 4, 1).reshape(-1, self.n_species)

    def _kbt_flat(self, T: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        B, grid = rho.shape[0], rho.shape[-3:]
        return (self.kB * T).to(rho.dtype).view(B, 1, 1, 1).expand(B, *grid).reshape(-1)

    def bulk_free_energy_density(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        f = self.f_local.f_pointwise(self._flat(rho), self._kbt_flat(T, rho))
        return f.view(rho.shape[0], 1, *rho.shape[-3:])

    def chemical_potential(self, rho: torch.Tensor, boxes: torch.Tensor,
                           T: torch.Tensor) -> torch.Tensor:
        B, grid = rho.shape[0], tuple(int(g) for g in rho.shape[-3:])
        ops = self._cache.ops_for_grid(grid, rho.device)
        mu = self.f_local.mu_pointwise(self._flat(rho), self._kbt_flat(T, rho))
        mu = mu.view(B, *grid, self.n_species).permute(0, 4, 1, 2, 3)
        kx, ky, kz = ops.k_axes(boxes)
        kmag = torch.sqrt(kx * kx + ky * ky + kz * kz)[:, 0]
        W = self.kernel.w_hat(kmag)
        rho_hat = ops.rfft(rho).permute(0, 2, 3, 4, 1)
        term = torch.einsum("b...ij,b...j->b...i", W.to(rho_hat.dtype), rho_hat)
        return mu + ops.irfft(term.permute(0, 4, 1, 2, 3))

    def mobility(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        n, B = self.n_species, rho.shape[0]
        M = self.mobility_matrix().to(rho.dtype)
        return M.view(1, n, n, 1, 1, 1).expand(B, n, n, *rho.shape[-3:])

    def forward(self, rho_hat: torch.Tensor, boxes: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        ops = self._cache.ops_for(rho_hat.shape, rho_hat.device)
        N = ops.grid[0] * ops.grid[1] * ops.grid[2]
        rho = ops.irfft(rho_hat * N)
        mu = self.chemical_potential(rho, boxes, T)
        M = self.mobility(rho, T)
        kx, ky, kz = ops.k_axes(boxes)
        gx, gy, gz = ops.grad_hat(ops.rfft(mu), kx, ky, kz)
        J = [torch.einsum("bijxyz,bjxyz->bixyz", M, ops.irfft(g)) for g in (gx, gy, gz)]
        return ops.div_hat(*[ops.rfft(j) / N for j in J], kx, ky, kz)


class StabilizedToyFactoryModel(ToyFactoryModel):
    """The same model, with the semi-implicit scheme's ``M_s`` hook: ``stabilizer_scale`` times ``M``."""

    def __init__(self, *args, stabilizer_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.stabilizer_scale = float(stabilizer_scale)

    def stabilizer_mobility(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return self.stabilizer_scale * self.mobility(rho, T)[0, :, :, 0, 0, 0]


def _constructor_kwargs(system: System) -> dict:
    kw = system.functional.kwargs
    return dict(nyquist_mask=kw["nyquist_mask"], kappa=kw["kappa"], kB=system.constants["kB"])


def build_toy(system: System, **overrides) -> ToyFactoryModel:
    """The factory: ``factory(system, **overrides)``."""
    return ToyFactoryModel(system.functional.kwargs["grid"], system.n_species,
                           **{**_constructor_kwargs(system), **overrides})


def build_stabilized_toy(system: System, **overrides) -> StabilizedToyFactoryModel:
    return StabilizedToyFactoryModel(system.functional.kwargs["grid"], system.n_species,
                                     **{**_constructor_kwargs(system), **overrides})


def toy_functional(factory=build_toy, **kwargs) -> Functional:
    return Functional(form="toy_square_gradient", local="polynomial", kernel=None,
                      kwargs={"grid": GRID, "nyquist_mask": True, "kappa": 0.5, **kwargs},
                      factory=factory)


def toy_defaults() -> dict:
    """Every knob the training driver reads, at a toy value (as ``tests/unit/test_train_fit.py``)."""
    return {
        "estimator": "weak", "sigma": 1.0, "k_fit_stat": 1.5, "k_max": 2.0,
        "alpha_loss": 0.0, "h_inv_eps": 1e-6, "lr": 1e-3, "weight_decay": 0.0,
        "warmup_epochs": 1, "anneal_epochs": 1, "source_loss_weights": {"demo_src": 1.0},
        "training": {
            "half_width": 2, "n_states": 3, "stride": 1, "savgol_window": None,
            "savgol_poly": None, "run_weighting": "uniform", "run_weight_probe_every": 10,
            "val_split": "random", "val_labels": (), "val_fraction": 0.1, "split_seed": 0,
            "batch_size": 2, "num_workers": 0, "pin_memory": False, "drop_last": False,
            "order": "shuffled", "lambda_dyn": 1.0, "bulk_residual": "relative_inverse",
            "eta_min": 1e-6, "grad_clip": 1.0,
        },
    }


def demo_system(raw_root: str) -> System:
    """A two-channel ``"demo"`` in reduced units whose functional, and whose ``stabilized`` variant's,
    is built by a factory."""
    return System(
        name="demo", n_species=2, species=("A", "B"),
        masses={"A": 1.0, "B": 2.0}, atom_types={"A": 1, "B": 2},
        table_keys={"rho": ("rho_A", "rho_B"), "x": "x_B", "x_channel": 1},
        paths=Paths(system="demo", raw_default=str(raw_root)),
        anchor_rules=AnchorRules({}), constants={"kB": 1.0},
        defaults=toy_defaults(), functional=toy_functional(),
        variants={"stabilized": Variant(
            functional=toy_functional(build_stabilized_toy), mobility=None,
            checkpoint=None, defaults={})})
