"""Second style layer of the Fe-B figures: a warm off-white surface, muted
axes and the house grid, applied on top of paperstyle part way through a
figure, so that only the elements created after it carry it."""
from __future__ import annotations

import matplotlib.pyplot as plt

SURF, INK1, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e",
                                       "#898781", "#e1e0d9", "#c3c2b7")

RC = {
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "text.color": INK1,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
}


def use():
    """Apply the surface layer to the global rcParams."""
    plt.rcParams.update(RC)
