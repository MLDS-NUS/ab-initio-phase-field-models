"""The seven canonical losses, plus the L_W extra.

Pure reductions over already-computed tensors: no model, no trust-domain sampling, no anchor
loading, no physical constant. Canonical names are the paper's (:mod:`aipf.losses.names`);
row gating and weighting are declared by the caller (:mod:`aipf.losses.gating`)."""
from __future__ import annotations

from aipf.losses.convexity import (gamma_closed_form, gamma_on_path, l_conv,
                                   l_gamma)
from aipf.losses.dyn import l_dyn
from aipf.losses.extras import eigmin_sym2, l_w
from aipf.losses.gating import gate_rows
from aipf.losses.mobility import l_m
from aipf.losses.names import (
    CANONICAL_LOSSES,
    LOG_MAP_A,
    LOG_MAP_B,
    NONCANONICAL_EXTRAS,
    LossNameMap,
    WEIGHT_MAP_A,
    WEIGHT_MAP_B,
)
from aipf.losses.pressure import l_p_absolute, l_p_variance
from aipf.losses.static import l_bulk, l_s, sinv_rootinv

__all__ = [
    "CANONICAL_LOSSES",
    "NONCANONICAL_EXTRAS",
    "LossNameMap",
    "WEIGHT_MAP_A",
    "WEIGHT_MAP_B",
    "LOG_MAP_A",
    "LOG_MAP_B",
    "gate_rows",
    "gamma_closed_form",
    "gamma_on_path",
    "l_bulk",
    "l_conv",
    "l_dyn",
    "l_gamma",
    "l_m",
    "l_p_absolute",
    "l_p_variance",
    "l_s",
    "l_w",
    "eigmin_sym2",
    "sinv_rootinv",
]
