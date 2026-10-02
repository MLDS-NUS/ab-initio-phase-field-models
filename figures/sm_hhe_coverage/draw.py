"""Supplementary figure: hydrogen-helium training states at 200 to 800 GPa, drift only against drift and anchors."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import paperstyle as ps  # noqa: E402

FIGDATA = HERE / "figdata"
REFERENCE = {"ed_hhe_coverage.pdf": "ed_hhe_coverage.pdf"}

C_DRIFT = "#B5471F"     # rust, states that feed the drift loss only
C_ANCHOR = "#1F3A5F"    # navy, states that also feed the anchors
W, H = 6.50, 1.95
M_LEFT, M_RIGHT, GAP = 0.56, 0.14, 0.36


def row(out):
    ps.use()
    d = np.load(FIGDATA / "training_states.npz")
    P_list = [int(P) for P in d["pressures"]]
    n = len(P_list)
    pw = (W - M_LEFT - M_RIGHT - (n - 1) * GAP) / n
    ph = H - ps.M_TOP - ps.M_BOTTOM
    fig = plt.figure(figsize=(W, H))
    axes = [fig.add_axes([(M_LEFT + i * (pw + GAP)) / W, ps.M_BOTTOM / H,
                          pw / W, ph / H]) for i in range(n)]
    for k, (ax, P) in enumerate(zip(axes, P_list)):
        x, T, anchor = d[f"x_He_P{P}"], d[f"T_P{P}"], d[f"anchored_P{P}"]
        drift = ~anchor
        ax.plot(x[drift], T[drift], ls="none", marker="s", ms=2.5,
                mfc=C_DRIFT, mec="none", zorder=3)
        ax.plot(x[anchor], T[anchor], ls="none", marker="s", ms=2.5,
                mfc=C_ANCHOR, mec="none", zorder=4)
        ax.set_title(f"{P} GPa", fontsize=ps.FS_ANNOT, pad=3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel(r"$x_{\mathrm{He}}$", labelpad=1)
        ax.set_ylim(1500, 12800)
        ax.set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
        ax.yaxis.set_major_formatter(lambda v, _p: f"{v / 1000:.0f}")
        ps.open_axes(ax)
        if k == 0:
            ax.set_ylabel(r"$T$ [$10^{3}$ K]", labelpad=3)
        else:
            ax.tick_params(labelleft=False)
    keys = [(Line2D([], [], ls="none", marker="s", ms=2.5, mfc=C_DRIFT,
                    mec="none"), "drift only"),
            (Line2D([], [], ls="none", marker="s", ms=2.5, mfc=C_ANCHOR,
                    mec="none"), "drift and anchors")]
    h, lab = zip(*keys)
    leg = axes[0].legend(h, lab, loc="upper left", bbox_to_anchor=(-0.02, 1.02),
                         fontsize=ps.FS_LEGEND_SM, frameon=False,
                         handlelength=1.3, handletextpad=0.4,
                         labelspacing=0.25, borderaxespad=0.2)
    leg.set_zorder(6)
    return ps.save(fig, out / "ed_hhe_coverage.pdf")


def draw(out: Path) -> list[Path]:
    with plt.rc_context():
        matplotlib.rcdefaults()
        return [row(Path(out))]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
