"""The Extended Data row: four panels and a temperature bar in 6.50 x 1.95 in,
8/9 pt, ink #15181c, and the clipped plasma map every temperature fan uses."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import paperstyle as ps

INK = "#15181c"
ROW = dict(W=6.50, H=1.95, left=0.56, gap=0.36, right=0.56, top=0.34,
           bottom=0.42)


def row_style():
    """The 8/9 pt profile on a white canvas with ink-coloured text and axes."""
    ps.use()
    plt.rcParams.update({"axes.grid": False, "figure.facecolor": "white",
                         "savefig.facecolor": "white",
                         "axes.facecolor": "white", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.edgecolor": INK})


def row_axes(fig, n=4):
    """n equal panels and the colour-bar axes right of them."""
    R = ROW
    pw = (R["W"] - R["left"] - R["right"] - (n - 1) * R["gap"]) / n
    ph = R["H"] - R["bottom"] - R["top"]
    axes = [fig.add_axes([(R["left"] + i * (pw + R["gap"])) / R["W"],
                          R["bottom"] / R["H"], pw / R["W"], ph / R["H"]])
            for i in range(n)]
    cax = fig.add_axes([(R["W"] - R["right"] + 0.12) / R["W"],
                        R["bottom"] / R["H"], 0.085 / R["W"], ph / R["H"]])
    return axes, cax


def row_cbar(fig, cax, cmap, norm, ticks):
    """Vertical temperature bar in 10^3 K."""
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"$T$ [$10^{3}$ K]", fontsize=ps.FS_LABEL, labelpad=2)
    cb.ax.tick_params(labelsize=ps.FS_TICK, length=ps.TICK_LEN,
                      width=ps.LW_AXES)
    cb.outline.set_linewidth(ps.LW_AXES)
    cb.set_ticks(list(ticks))
    cb.ax.yaxis.set_major_formatter(lambda v, _p: f"{v / 1000:.0f}")
    return cb


def plasma():
    """Plasma clipped at both ends, the temperature map of every row."""
    return LinearSegmentedColormap.from_list(
        "plasma_clip", plt.get_cmap("plasma")(np.linspace(0.06, 0.88, 256)))
