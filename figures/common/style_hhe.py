"""The Nature type profile (6/7 pt) of the hydrogen-helium main-figure panels and
the panels drawn at that size (Fig. 5, Extended Data Fig. 4)."""
from __future__ import annotations

import matplotlib.pyplot as plt

import paperstyle as ps

# ── Nature profile: 183 mm full width, 6/7 pt type ───────────────────────
W_NATURE = 183.0 / 25.4
FS_BASE, FS_LABEL, FS_TICK = 6.0, 7.0, 6.0
FS_LEGEND, FS_LEGEND_SM, FS_ANNOT = 6.0, 6.0, 6.0
LW_AXES, LW_CURVE, TICK_LEN = 0.6, 1.0, 2.2
MS_MARKER, MEW_MARKER = 3.0, 0.5
M_TOP, M_BOTTOM, M_LEFT, M_RIGHT = 0.26, 0.32, 0.42, 0.10

RC_NATURE = dict(ps.RC)
RC_NATURE.update({
    "font.size": FS_BASE,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "xtick.major.size": TICK_LEN,
    "ytick.major.size": TICK_LEN,
    "xtick.major.width": LW_AXES,
    "ytick.major.width": LW_AXES,
    "axes.linewidth": LW_AXES,
    "lines.linewidth": LW_CURVE,
})


def use_nature():
    """Apply the Nature profile's rcParams."""
    plt.rcParams.update(RC_NATURE)
