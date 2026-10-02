"""Extended Data Fig. 7: hydrogen-helium number density along the 200 to 800 GPa isobars, MD against AIPF."""
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
import paperstyle as ps  # noqa: E402
import edrow  # noqa: E402

FIGDATA = HERE / "figdata"
REFERENCE = {"ed_hhe_nx.pdf": "ed_hhe_nx.pdf"}


def row(out):
    edrow.row_style()
    d = np.load(FIGDATA / "density_isobars.npz")
    x, temps, pressures = d["x_He"], d["T"], d["pressures"]
    cmap, norm = edrow.plasma(), Normalize(temps.min(), temps.max())
    fig = plt.figure(figsize=(edrow.ROW["W"], edrow.ROW["H"]))
    axes, cax = edrow.row_axes(fig)
    for ax, P in zip(axes, pressures):
        x_md, n_md = d[f"x_He_md_P{P:.0f}"], d[f"n_md_P{P:.0f}"]
        n_model = d[f"n_model_P{P:.0f}"]
        for i, T in enumerate(temps):
            ax.plot(x, n_model[i], lw=0.9, color=cmap(norm(T)), zorder=2)
        for i, T in enumerate(temps):
            ax.plot(x_md, n_md[i], ls="none", marker="o", ms=2.6,
                    mfc=cmap(norm(T)), mec="none", zorder=3)
        ax.set_title(f"{P:.0f} GPa", fontsize=ps.FS_ANNOT, pad=3)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel(r"$x_{\mathrm{He}}$", labelpad=1)
        ps.open_axes(ax)
    axes[0].set_ylabel(r"$n$ [Å$^{-3}$]", labelpad=3)
    hh = [Line2D([], [], ls="none", marker="o", ms=2.6, mfc="0.4", mec="none"),
          Line2D([], [], color="0.4", lw=0.9)]
    axes[0].legend(hh, ["MD", "AIPF"], loc="upper right",
                   fontsize=ps.FS_LEGEND_SM, frameon=False, handlelength=1.2,
                   handletextpad=0.4, labelspacing=0.2, borderaxespad=0.2)
    edrow.row_cbar(fig, cax, cmap, norm, [t for t in temps if t % 2000 == 0])
    return ps.save(fig, out / "ed_hhe_nx.pdf")


def draw(out: Path) -> list[Path]:
    with plt.rc_context():
        matplotlib.rcdefaults()
        return [row(Path(out))]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
