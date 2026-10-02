"""The free-energy model protocol every ladder rung implements: k-space in, k-space out."""
from __future__ import annotations

from typing import Dict, Protocol, Tuple, Type, runtime_checkable

import torch


@runtime_checkable
class FreeEnergyModel(Protocol):
    """The protocol every rung of the ladder implements."""

    def forward(self, rho_hat: torch.Tensor, boxes: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        """`drho_hat_dt` from `rho_hat` `(B, n_species, Gx, Gy, Gzr)`, `boxes` `(B, 3)`, `T` `(B,)`.
        The k=0 mode is exactly zero (mass conservation by construction)."""
        ...

    def chemical_potential(self, rho: torch.Tensor, boxes: torch.Tensor,
                            T: torch.Tensor) -> torch.Tensor:
        """`mu`, pointwise, real space; `rho` and the result are `(B, n_species, Gx, Gy, Gz)`."""
        ...

    def bulk_free_energy_density(self, rho: torch.Tensor,
                                  T: torch.Tensor) -> torch.Tensor:
        """The pointwise `f_loc(rho, T)`, real space, without gradient or kernel terms."""
        ...

    def mobility(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """`M(rho, T)`, real space, broadcastable against `(n_species, n_species)` per point."""
        ...


PROTOCOL_METHODS: Tuple[str, ...] = (
    "forward", "chemical_potential", "bulk_free_energy_density", "mobility",
)


def check_protocol(cls: type) -> None:
    """Raise `TypeError` if the class `cls` is missing a `FreeEnergyModel` method."""
    missing = [name for name in PROTOCOL_METHODS
               if not callable(getattr(cls, name, None))]
    if missing:
        raise TypeError(
            f"{cls.__name__!r} does not implement FreeEnergyModel: missing "
            f"{missing}")


class ModelRegistry:
    """Name to `FreeEnergyModel` class, validated at registration."""

    def __init__(self) -> None:
        self._models: Dict[str, Type] = {}

    def register(self, name: str, cls: type) -> type:
        """Validate `cls` against `FreeEnergyModel`, record it under `name`, return it."""
        check_protocol(cls)
        self._models[str(name)] = cls
        return cls

    def build(self, name: str, **kwargs):
        if name not in self._models:
            raise KeyError(f"unknown form {name!r}; have {sorted(self._models)}")
        return self._models[name](**kwargs)

    def __contains__(self, name: object) -> bool:
        return name in self._models

    def __getitem__(self, name: str) -> type:
        """The registered class itself; `KeyError` names what is registered."""
        if name not in self._models:
            raise KeyError(
                f"unknown form {name!r}; have {sorted(self._models)}")
        return self._models[name]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._models))


#: The one registry every rung registers into.
MODEL_REGISTRY = ModelRegistry()
