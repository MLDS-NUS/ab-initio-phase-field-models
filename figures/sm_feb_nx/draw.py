"""Supplementary figure: iron-boron number density along the 0, 5 and 10 GPa isobars, MD against AIPF."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import edrow  # noqa: E402
import paperstyle as ps  # noqa: E402

FIGDATA = HERE / "figdata"
REFERENCE = {"ed_feb_nx.pdf": "ed_feb_nx.pdf"}

PRESSURES = (0, 5, 10)
W_FIG, H_FIG = 6.50, 1.95
M_LEFT, GAP, RIGHT_BAND = 0.56, 0.36, 0.56


def row(out):
    ps.use()
    plt.rcParams["axes.grid"] = False
    d = np.load(FIGDATA / "density_isobars.npz")
    temps = d["T"]
    cmap = edrow.plasma()
    norm = Normalize(temps.min(), temps.max())

    sub_w = (W_FIG - M_LEFT - RIGHT_BAND - 2 * GAP) / 3
    sub_h = H_FIG - ps.M_TOP - ps.M_BOTTOM
    fig = plt.figure(figsize=(W_FIG, H_FIG))
    axes = [fig.add_axes([(M_LEFT + i * (sub_w + GAP)) / W_FIG,
                          ps.M_BOTTOM / H_FIG, sub_w / W_FIG, sub_h / H_FIG])
            for i in range(3)]
    cax = fig.add_axes([(W_FIG - RIGHT_BAND + 0.12) / W_FIG,
                        ps.M_BOTTOM / H_FIG, 0.085 / W_FIG, sub_h / H_FIG])

    xB = d["x_B"]
    for ax, P in zip(axes, PRESSURES):
        ax.grid(False)
        for T in temps:
            ax.plot(xB, d[f"n_model_P{P}_T{T:.0f}"], lw=0.9,
                    color=cmap(norm(T)), zorder=2)
        xm, Tm, nm = d[f"x_B_md_P{P}"], d[f"T_md_P{P}"], d[f"n_md_P{P}"]
        for T in temps:
            s = Tm == T
            ax.plot(xm[s], nm[s], ls="none", marker="o", ms=2.6,
                    mfc=cmap(norm(T)), mec="none", zorder=3)
        ax.set_title(f"{P} GPa", fontsize=ps.FS_ANNOT, pad=3)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel(r"$x_{\mathrm{B}}$", labelpad=1)
        ps.open_axes(ax)
    axes[0].set_ylabel(r"$n$ [Å$^{-3}$]", labelpad=3)
    hh = [Line2D([], [], ls="none", marker="o", ms=2.6, mfc="0.4", mec="none"),
          Line2D([], [], color="0.4", lw=0.9)]
    axes[0].legend(hh, ["MD", "model"], loc="upper left",
                   fontsize=ps.FS_LEGEND_SM, frameon=False, handlelength=1.2,
                   handletextpad=0.4, labelspacing=0.2, borderaxespad=0.2)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"$T$ [K]", fontsize=ps.FS_LABEL, labelpad=2)
    cb.ax.tick_params(labelsize=ps.FS_TICK, length=ps.TICK_LEN,
                      width=ps.LW_AXES)
    cb.outline.set_linewidth(ps.LW_AXES)
    cb.set_ticks([1200, 1600, 2000, 2400])
    cb.ax.yaxis.set_major_formatter(lambda v, _p: f"{v:,.0f}")
    return ps.save(fig, out / "ed_feb_nx.pdf")


def draw(out: Path) -> list[Path]:
    with plt.rc_context():
        matplotlib.rcdefaults()
        return [row(Path(out))]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
