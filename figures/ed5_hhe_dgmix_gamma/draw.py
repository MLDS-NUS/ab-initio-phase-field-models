"""Extended Data Fig. 5: hydrogen-helium free energy of mixing (a) and thermodynamic factor (b), 200 to 800 GPa."""
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
import paperstyle as ps  # noqa: E402
import edrow  # noqa: E402

FIGDATA = HERE / "figdata"
REFERENCE = {"ed_hhe_dgmix.pdf": "ed_hhe_dgmix.pdf",
             "ed_hhe_gamma.pdf": "ed_hhe_gamma.pdf"}

PRESSURES = (200, 400, 600, 800)
T_LO, T_HI = 2000.0, 12000.0
T_TICKS = [2000.0, 4000.0, 6000.0, 8000.0, 10000.0, 12000.0]


def x_axis(ax, P):
    ax.set_title(f"{P:g} GPa", fontsize=ps.FS_ANNOT, pad=3)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel(r"$x_{\mathrm{He}}$", labelpad=1)
    ps.open_axes(ax)


def dgmix_row(out):
    edrow.row_style()
    d = np.load(FIGDATA / "dG_mix.npz")
    cmap, norm = edrow.plasma(), Normalize(T_LO, T_HI)
    fig = plt.figure(figsize=(edrow.ROW["W"], edrow.ROW["H"]))
    axes, cax = edrow.row_axes(fig)
    g_lo = 0.0
    for ax, P in zip(axes, PRESSURES):
        for T in d[f"T_P{P}"]:
            dG = d[f"dG_mix_P{P}_T{T:.0f}"]
            ax.plot(d[f"x_He_P{P}_T{T:.0f}"], dG, "-", color=cmap(norm(T)),
                    lw=1.0, zorder=2)
            g_lo = min(g_lo, float(np.nanmin(dG)))
        ax.axhline(0.0, color=ps.C_NEUTRAL, lw=0.6, ls=(0, (4, 3)), zorder=1)
        x_axis(ax, P)
    for i, ax in enumerate(axes):
        ax.set_ylim(1.08 * g_lo, max(0.12, -0.35 * g_lo))
        ax.set_yticks([-0.3, -0.2, -0.1, 0.0])
        if i:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"$\Delta G_{\mathrm{mix}}$ [eV/atom]", labelpad=3)
    edrow.row_cbar(fig, cax, cmap, norm, T_TICKS)
    return ps.save(fig, out / "ed_hhe_dgmix.pdf")


def gamma_row(out):
    edrow.row_style()
    d = np.load(FIGDATA / "gamma.npz")
    x = d["x_He"]
    cmap, norm = edrow.plasma(), Normalize(T_LO, T_HI)
    fig = plt.figure(figsize=(edrow.ROW["W"], edrow.ROW["H"]))
    axes, cax = edrow.row_axes(fig)
    gam_lo = np.inf
    for ax, P in zip(axes, PRESSURES):
        for T in d[f"T_P{P}"]:
            g = d[f"Gamma_P{P}_T{T:.0f}"]
            ax.plot(x, g, "-", color=cmap(norm(T)), lw=1.0, zorder=2)
            if np.isfinite(g).any():
                gam_lo = min(gam_lo, float(np.nanmin(g)))
        ax.axhline(1.0, color=ps.C_NEUTRAL, lw=0.6, ls=(0, (1.5, 2)), zorder=1)
        ax.axhline(0.0, color=edrow.INK, lw=0.7, zorder=1)
        x_axis(ax, P)
    for i, ax in enumerate(axes):
        ax.set_ylim(1.06 * gam_lo, 1.15)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        if i:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"$\Gamma = x(1-x)/S_{cc}(0)$", labelpad=3)
    axes[-1].text(0.97, 1.0, "ideal", transform=axes[-1].get_yaxis_transform(),
                  fontsize=ps.FS_LEGEND_SM, color=ps.C_NEUTRAL, ha="right",
                  va="bottom")
    edrow.row_cbar(fig, cax, cmap, norm, T_TICKS)
    return ps.save(fig, out / "ed_hhe_gamma.pdf")


def draw(out: Path) -> list[Path]:
    out = Path(out)
    with plt.rc_context():
        matplotlib.rcdefaults()
        return [dgmix_row(out), gamma_row(out)]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
