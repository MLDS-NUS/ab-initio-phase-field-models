"""The names of the diagnostic stages; stdlib only, so ``aipf --help`` stays light."""
from __future__ import annotations

from typing import Tuple

__all__ = ["STAGES"]

# The stages the driver knows; there is no default stage.
STAGES: Tuple[str, ...] = ("phase_diagram", "dome", "tc", "kappa",
                           "stability_map", "one_field_phase_diagram")
