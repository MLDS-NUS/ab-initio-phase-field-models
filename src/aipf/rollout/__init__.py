"""``aipf.rollout``: the two basic checks every system runs on a trained functional (spinodal
decomposition from a cube, coexistence from a slab), the semi-implicit scheme the published runs used,
and its conserved noise under the system's declared ``Noise``."""
from __future__ import annotations

from .imex import MASS_RESTORES, rollout_imex
from .noise import NOISE_EVAL_MODES
# The drivers share names with their modules and are not re-exported:
# ``aipf.rollout.spinodal.spinodal``, ``aipf.rollout.slab.slab``.

__all__ = ["MASS_RESTORES", "NOISE_EVAL_MODES", "rollout_imex"]
