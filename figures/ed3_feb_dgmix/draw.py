"""Extended Data Fig. 3: iron-boron free energy of mixing at 0, 5 and 10 GPa."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import edrow  # noqa: E402
import paperstyle as ps  # noqa: E402

FIGDATA = HERE / "figdata"
REFERENCE = {"ed_feb_dgmix.pdf": "ed_feb_dgmix.pdf"}

PRESSURES = (0, 5, 10)
T_MIN, T_MAX = 1200.0, 2600.0
PW, PH = 1.075, 1.19          # plot area per panel, inches
LEFT, GAP, RIGHT = 0.56, 0.36, 0.74
TOP, BOTTOM = 0.34, 0.42


def xB_axis(ax, ticks):
    """Axis of a curve drawn in x_Fe, shown as x_B increasing to the right."""
    ax.set_xlim(1.0, 0.0)
    ticks = np.asarray(ticks, dtype=float)
    ax.set_xticks(1.0 - ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def row(out):
    edrow.row_style()
    d = np.load(FIGDATA / "dG_mix.npz")
    n = len(PRESSURES)
    W = LEFT + n * PW + (n - 1) * GAP + RIGHT
    H = BOTTOM + PH + TOP
    fig = plt.figure(figsize=(W, H))
    axes = [fig.add_axes([(LEFT + i * (PW + GAP)) / W, BOTTOM / H,
                          PW / W, PH / H]) for i in range(n)]
    cax = fig.add_axes([(W - RIGHT + 0.12) / W, BOTTOM / H, 0.085 / W, PH / H])
    cmap = edrow.plasma()
    norm = Normalize(T_MIN, T_MAX)

    g_lo = 0.0
    for ax, P in zip(axes, PRESSURES):
        ax.grid(False)
        for T in d[f"T_P{P}"]:
            dG = d[f"dG_mix_P{P}_T{T:.0f}"]
            ax.plot(d[f"x_Fe_P{P}_T{T:.0f}"], dG, "-", color=cmap(norm(T)),
                    lw=1.0, zorder=2)
            g_lo = min(g_lo, float(np.nanmin(dG)))
        ax.axhline(0.0, color="0.55", lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.set_title(f"{P:g} GPa", fontsize=ps.FS_ANNOT, pad=3)
        xB_axis(ax, ticks=[0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel(r"$x_\mathrm{B}$", labelpad=1)
        ps.open_axes(ax)
    for i, ax in enumerate(axes):
        ax.set_ylim(1.08 * g_lo, 0.06 * abs(g_lo))
        ax.set_yticks([-0.3, -0.2, -0.1, 0.0])
        if i:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"$\Delta G_{\mathrm{mix}}$ [eV/atom]", labelpad=3)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"$T$ [K]", fontsize=ps.FS_LABEL, labelpad=2)
    cb.ax.tick_params(labelsize=ps.FS_TICK, length=ps.TICK_LEN,
                      width=ps.LW_AXES)
    cb.outline.set_linewidth(ps.LW_AXES)
    cb.set_ticks([1200, 1600, 2000, 2400])
    cb.ax.yaxis.set_major_formatter(lambda v, _p: f"{v:,.0f}")
    return ps.save(fig, out / "ed_feb_dgmix.pdf")


def draw(out: Path) -> list[Path]:
    with plt.rc_context():
        matplotlib.rcdefaults()
        return [row(Path(out))]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
