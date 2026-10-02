"""The field container every model, loss and solver driver shares.
A batched k-space density state with per-sample boxes; it carries data and performs no transform."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Field:
    """``rho_hat`` ``(B, n_species, Gx, Gy, Gzr)`` complex half-spectrum (``rfft`` / mode count),
    ``boxes`` ``(B, 3)`` per sample, ``grid`` ``(Gx, Gy, Gz)`` with ``Gzr = Gz // 2 + 1`` checked."""

    rho_hat: torch.Tensor
    boxes: torch.Tensor
    grid: Tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.rho_hat.dim() != 5:
            raise ValueError(
                "rho_hat must be (B, n_species, Gx, Gy, Gzr); got shape "
                f"{tuple(self.rho_hat.shape)}")
        if not self.rho_hat.is_complex():
            raise ValueError(
                "rho_hat must be the complex half-spectrum SpectralOps.rfft "
                f"produces; got dtype {self.rho_hat.dtype}")
        Gx, Gy, Gz = self.grid
        want = (int(Gx), int(Gy), Gz // 2 + 1)
        have = tuple(self.rho_hat.shape[-3:])
        if have != want:
            raise ValueError(
                f"grid {self.grid} implies rfft trailing shape {want}, but "
                f"rho_hat's trailing shape is {have}")
        if self.boxes.dim() != 2 or self.boxes.shape[1] != 3:
            raise ValueError(
                f"boxes must be (B, 3); got shape {tuple(self.boxes.shape)}")
        if self.boxes.shape[0] != self.rho_hat.shape[0]:
            raise ValueError(
                f"boxes batch {self.boxes.shape[0]} does not match "
                f"rho_hat's batch {self.rho_hat.shape[0]}")

    @property
    def batch_size(self) -> int:
        return self.rho_hat.shape[0]

    @property
    def n_species(self) -> int:
        return self.rho_hat.shape[1]

    @property
    def device(self) -> torch.device:
        return self.rho_hat.device

    def to(self, *args, **kwargs) -> "Field":
        """A new `Field` with `rho_hat` and `boxes` moved as ``torch.Tensor.to`` would; `grid` unchanged."""
        return Field(self.rho_hat.to(*args, **kwargs),
                     self.boxes.to(*args, **kwargs), self.grid)
