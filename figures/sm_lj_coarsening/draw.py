"""Supplementary figure: spinodal coarsening of binary Lennard-Jones at T = 1.10, 1.15, 1.25 and 1.30, AIPF boxes and L(t) against MD."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import box3d  # noqa: E402
import paperstyle as ps  # noqa: E402

FIGDATA = HERE / "figdata"
# The painted boxes ship pre-rendered at the PNG's resolution (figdata/painted_T*_*.png, see
# figures/common/box3d.py). AIPF_FIGURES_RENDER_3D=1 (or true, yes) paints them from the faces
# again (about 6 minutes) and rewrites those files.
DPI = 600
FACES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
TEMPERATURES = ("1.10", "1.15", "1.25", "1.30")
REFERENCE = {}
for _T in TEMPERATURES:
    REFERENCE[f"supp_fig2e_coarsening_T{_T}.png"] = f"supp_fig2e_coarsening_T{_T}.png"
    REFERENCE[f"supp_fig2f_growth_T{_T}.pdf"] = f"supp_fig2f_growth_T{_T}.pdf"

# The published files were drawn by the pip build of matplotlib, whose
# freetype rasterises and measures text differently from the conda build.
BUILD = "matplotlib 3.10.8, freetype 2.6.1"

K_SM = ps.W_SM / ps.W_FULL            # panels drawn at the supplement's width
ELEV, AZIM, RES = 22.0, -52.0, 200
LW_EDGE = 0.8 * (0.87 / 3.90)         # cube wireframe, scaled to the cube size
LW_SEED = ps.LW_CURVE * 0.75
C_MD = ps.shades(ps.C_MD, 3)          # seed 0 to 2, dark to light
C_MODEL = ps.shades(ps.C_MODEL, 3)
FS_KEYS = ps.FS_LEGEND * 0.88 * (0.9 * 1.25 * 0.9)


def layer_path(T, j):
    return FIGDATA / f"painted_T{T}_{j}.png"


def rerun() -> str:
    """The command that paints the boxes again and rewrites their layers, under this
    figure's build (the pip wheel first on the path)."""
    import wheel
    return (f"{box3d.RENDER_3D}=1 PYTHONPATH={wheel.wheel_dir()} "
            f"python figures/{HERE.name}/draw.py figures/out/{HERE.name}")


def boxes(T, out, stage):
    """The painted boxes at temperature ``T``; ``stage`` collects the layers when the
    boxes are painted."""
    paint = stage is not None
    d = np.load(FIGDATA / f"boxes_T{T}.npz")
    t, L = d["t"], float(d["L_box"])
    box = ((0.0, L),) * 3
    perm = (2, 0, 1)
    norm = Normalize(0.0, 1.0)
    w, h = ps.PANEL["e"][0] * K_SM, ps.PANEL["e"][1] * K_SM
    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(1, len(t), left=0.005, right=0.865, top=0.80,
                          bottom=0.02, wspace=0.04)
    sources = []
    for j in range(len(t)):
        ax = fig.add_subplot(gs[0, j], projection="3d")
        faces = [d[k][j] for k in FACES]
        sources.append(box3d.source_key(*faces, d["L_box"]))
        if paint:
            box3d.paint_faces(ax, faces, box, ps.RHO_CMAP, norm, RES, perm)
        else:
            box3d.add_layer(ax, layer_path(T, j), sources[j], rerun)
        box3d.draw_box_edges(ax, box, perm, lw=LW_EDGE)
        ax.set_box_aspect((L, L, L), zoom=1.05)
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_axis_off()
        lim = (-0.04 * L, L + 0.04 * L)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_zlim(*lim)
        ax.set_title(rf"$t = {t[j]:.0f}\,\tau$", fontsize=ps.FS_LABEL, y=1.06)

    sm = cm.ScalarMappable(cmap=ps.RHO_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.885, 0.14, 0.016, 0.60])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=ps.FS_TICK, length=ps.TICK_LEN * 0.6,
                      width=ps.LW_AXES * 0.5, direction="in")
    cb.outline.set_linewidth(ps.LW_AXES * 0.5)
    cb.set_label(r"$x_\mathrm{A}$", fontsize=ps.FS_LABEL, labelpad=2)
    cb.solids.set_rasterized(True)
    if paint:
        dpi, fig.dpi = fig.dpi, DPI
        fig.canvas.draw()
        for j, ax in enumerate(fig.axes[:len(t)]):
            box3d.save_layer(fig, ax, layer_path(T, j), sources[j], rerun, stage)
        fig.dpi = dpi
    for ax in fig.axes:
        for art in ax.collections:
            art.set_rasterized(True)
    return ps.save(fig, out / f"supp_fig2e_coarsening_T{T}.png", dpi=DPI)


def growth(T, out):
    g = np.load(FIGDATA / f"growth_T{T}.npz")
    w, h = ps.PANEL["f"][0] * K_SM, ps.PANEL["f"][1] * K_SM
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes(ps.axes_rect(w, h, left=0.46, bottom=0.40, top=0.10,
                                   right=ps.M_RIGHT))
    for fam, cols in (("md", C_MD), ("aipf", C_MODEL)):
        for i in range(3):
            ax.plot(g[f"{fam}{i}_t"], g[f"{fam}{i}_L"], "-", color=cols[i],
                    lw=LW_SEED)
    ax.set_xscale("symlog", linthresh=10.0, linscale=0.35)
    ax.set_yscale("log")
    ax.set_xlim(0, 1e4)
    ax.set_xticks([0, 1e1, 1e2, 1e3, 1e4])
    ax.set_yticks([1e1, 2e1])
    ax.set_yticklabels(["10", "20"])
    ax.set_ylim(7.4, 21.0)
    ax.minorticks_off()
    ax.set_xlabel(r"$t\ [\tau]$", fontsize=ps.FS_LABEL)
    ax.set_ylabel(r"$L(t)\ [\sigma]$", fontsize=ps.FS_LABEL, labelpad=2)
    ax.tick_params(labelsize=ps.FS_TICK, which="both")
    ax.grid(False)
    h_ = [plt.Line2D([], [], color=C_MD[1], lw=LW_SEED * 1.5),
          plt.Line2D([], [], color=C_MODEL[1], lw=LW_SEED * 1.5)]
    ax.legend(h_, ["MD (3 seeds)", "Model (3 seeds)"], fontsize=FS_KEYS,
              frameon=False, loc="upper left", handlelength=1.5,
              borderaxespad=0.4, labelspacing=0.3)
    return ps.save(fig, out / f"supp_fig2f_growth_T{T}.pdf")


def draw(out: Path) -> list[Path]:
    out = Path(out)
    ps.use()
    if not box3d.paint_requested(os.environ):
        return [p for T in TEMPERATURES for p in (boxes(T, out, None), growth(T, out))]
    with box3d.staged_layers(FIGDATA) as stage:
        return [p for T in TEMPERATURES for p in (boxes(T, out, stage), growth(T, out))]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
