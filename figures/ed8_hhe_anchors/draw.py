"""Extended Data Fig. 8: AIPF against MD at the training anchors, thermodynamic factor and the three mobility components."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import edrow  # noqa: E402
import paperstyle as ps  # noqa: E402

FIGDATA = HERE / "figdata"
REFERENCE = {"fig_anchor_parity.png": "fig_anchor_parity.png"}

PRESSURES = (200, 400, 600, 800)
PMARK = {200: "o", 400: "s", 600: "^", 800: "D"}
COMPS = (r"$M_\mathrm{HH}$", r"$M_\mathrm{HeHe}$", r"$|M_\mathrm{HHe}|$")
M_UNIT = r"[$\mathrm{\AA}^{-1}\mathrm{eV}^{-1}\mathrm{ps}^{-1}$]"
WIDTH, HEIGHT = 6.50, 1.96
MS, GAP, RIGHT, ALPHA, TOP = 2.4, 0.40, 0.60, 0.75, 0.50


def parity(out):
    ps.use()
    d = np.load(FIGDATA / "anchor_parity.npz")
    cmap = edrow.plasma()
    T_all = np.concatenate([d["gamma_T"], d["M_T"]])
    nrm = mcolors.Normalize(np.floor(T_all.min() / 1000) * 1000,
                            np.ceil(T_all.max() / 1000) * 1000)

    Md, Mm, MP, MT = d["M_data"], d["M_model"], d["M_P"], d["M_T"]
    series = [(r"$\Gamma$" + "\n ", d["gamma_data"], d["gamma_model"],
               d["gamma_T"], d["gamma_P"])]
    for lab, (a, b) in zip(COMPS, ((0, 0), (1, 1), (0, 1))):
        series.append((lab + "\n" + M_UNIT, np.abs(Md[:, a, b]),
                       np.abs(Mm[:, a, b]), MT, MP))

    left = ps.M_LEFT
    ph = HEIGHT - ps.M_BOTTOM - TOP
    pw = (WIDTH - left - RIGHT - 3 * GAP) / 4.0
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    axes = []
    for k, (lab, xd, yd, T, P) in enumerate(series):
        ax = fig.add_axes([(left + k * (pw + GAP)) / WIDTH,
                           ps.M_BOTTOM / HEIGHT, pw / WIDTH, ph / HEIGHT])
        axes.append(ax)
        hi = 1.06 * max(xd.max(), yd.max())
        ax.plot([0.0, hi], [0.0, hi], ls=(0, (4, 2.5)), lw=ps.LW_AXES,
                color="0.35", zorder=1)
        for Pv in np.unique(P):
            s = P == Pv
            ax.scatter(xd[s], yd[s], c=T[s], cmap=cmap, norm=nrm,
                       marker=PMARK[int(Pv)], s=MS ** 2 * 2.2, linewidths=0.0,
                       edgecolors="none", alpha=ALPHA, zorder=3)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(lab, fontsize=ps.FS_ANNOT, pad=2.0, linespacing=1.15)
        ax.set_xlabel("MD")
        ax.grid(False)
        ax.tick_params(labelsize=ps.FS_TICK)
    axes[0].set_ylabel("AIPF")

    hp = [Line2D([], [], ls="none", marker=PMARK[P], ms=MS + 0.4, mfc="0.35",
                 mec="none") for P in PRESSURES]
    axes[0].legend(hp, [f"{P} GPa" for P in PRESSURES], loc="lower right",
                   fontsize=6.0, frameon=False, handletextpad=0.25,
                   borderaxespad=0.25, labelspacing=0.14)

    cax = fig.add_axes([(WIDTH - RIGHT + 0.10) / WIDTH, ps.M_BOTTOM / HEIGHT,
                        ps.CBAR_W / WIDTH, ph / HEIGHT])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cmap), cax=cax)
    cb.set_label(r"$T$ [$10^{3}$ K]", fontsize=ps.FS_LABEL)
    ticks = np.arange(nrm.vmin, nrm.vmax + 1, 2000)
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([f"{t/1000:.0f}" for t in ticks])
    cb.ax.tick_params(labelsize=ps.FS_TICK, length=ps.TICK_LEN,
                      width=ps.LW_AXES)
    cb.outline.set_linewidth(ps.LW_AXES)
    return ps.save(fig, out / "fig_anchor_parity.png")


def draw(out: Path) -> list[Path]:
    with plt.rc_context():
        matplotlib.rcdefaults()
        return [parity(Path(out))]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
