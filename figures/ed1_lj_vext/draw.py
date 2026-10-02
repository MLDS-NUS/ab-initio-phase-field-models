"""Extended Data Fig. 1: equilibria under an external potential, MD against AIPF, Flory-Huggins and Landau."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import proj3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import box3d  # noqa: E402
import paperstyle as ps  # noqa: E402

FIGDATA = HERE / "figdata"
# The painted boxes ship pre-rendered at the PNG's resolution (figdata/painted_*.png, see
# figures/common/box3d.py). AIPF_FIGURES_RENDER_3D=1 (or true, yes) paints them from the fields
# again (about 6 minutes) and rewrites those files.
DPI = 600
REFERENCE = {
    "ed_lj_vext_hexagonal.png": "ed_lj_vext_hexagonal.png",
    "ed_lj_vext_checkerboard.png": "ed_lj_vext_checkerboard.png",
}

ROW_W, ROW_H = ps.W_SM, 1.50
# layout, inches
M_SIDE = 0.03            # outermost text to the page edge
PAD_TEXT = 0.06          # text to the next column's graphic
AX3D_BOTTOM, AX3D_TOP = 0.21, 0.10
VEXT_AX_W = 1.00
CUBE_AX_W = 0.92
CUBE_ZOOM = 1.10
PROF_W, PROF_BOTTOM, PROF_H = 1.04, 0.42, 0.80
PROF_YTOP = 1.95
LW_EDGE = 0.8 * (0.85 / 3.53)    # cube wireframe, scaled to the cube size
NZ = 80                          # the fields are uniform along z
TICK_W = ps.LW_AXES * 0.7
XMAX = 40.0                      # the profile runs to x = 40 sigma (periodic)

C_MD, C_MODEL = ps.C_MD, ps.C_MODEL
C_FH, C_LANDAU = "#7F43A4", "#2CA02C"
LS_FH = (0, (4.0, 1.4, 1.0, 1.4))
LS_LANDAU = (0, (1.0, 1.1))
COL_TITLES = (r"$V_{\rm ext}$", "MD", "AIPF", "FH", "Landau", "Density profile")


def surface(ax, X, Y, Z, cmap, norm, zlim, title):
    """V_ext as a height surface over the x-y plane."""
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(Z)), rcount=150, ccount=150,
                    shade=False, antialiased=False, linewidth=0, edgecolor="none")
    ax.set_title(title, fontsize=ps.FS_LABEL, pad=2)
    ax.set_xlabel(r"$x\ [\sigma]$", fontsize=ps.FS_LABEL, labelpad=-2)
    ax.set_ylabel(r"$y\ [\sigma]$", fontsize=ps.FS_LABEL, labelpad=-2)
    ax.set_zlim(*zlim)
    ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(3, prune=None))
    ax.set_xticks([0, 20, 40])
    ax.set_yticks([0, 20, 40])
    ax.set_box_aspect((1, 1, 0.55), zoom=1.0)
    ax.view_init(elev=32, azim=-58)
    ax.tick_params(labelsize=ps.FS_TICK, pad=1, direction="in",
                   length=ps.TICK_LEN, width=TICK_W)
    for a3 in (ax.xaxis, ax.yaxis, ax.zaxis):
        a3.line.set_linewidth(ps.LW_AXES)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((1, 1, 1, 0))
        pane.pane.set_edgecolor((0.85, 0.85, 0.85, 1))
    ax.grid(False)


def paint_box(ax, field3d, box, title, layer, source, paint, res=200):
    """x_A painted on the faces of the cube (or its pre-rendered ``layer``, painted
    from the arrays whose key is ``source``); the top face is the x-y plane."""
    perm = (0, 1, 2)
    if paint:
        box3d.paint_faces(ax, box3d.faces_of(field3d), box, ps.RHO_CMAP,
                          Normalize(0.0, 1.0), res, perm)
    else:
        box3d.add_layer(ax, layer, source, rerun)
    box3d.draw_box_edges(ax, box, perm, lw=LW_EDGE)
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = box
    ax.set_box_aspect((xhi - xlo, yhi - ylo, zhi - zlo), zoom=CUBE_ZOOM)
    ax.view_init(elev=26, azim=-58)
    ax.set_axis_off()

    def _pad(lo, hi, f=0.04):
        m = (hi - lo) * f
        return lo - m, hi + m
    ax.set_xlim(*_pad(xlo, xhi))
    ax.set_ylim(*_pad(ylo, yhi))
    ax.set_zlim(*_pad(zlo, zhi))
    ax.set_title(title, fontsize=ps.FS_LABEL, pad=2)


def profile(ax, x_phys, fields, ym):
    """x_A(x) along the row y = ym: MD, AIPF and the two baselines."""
    dx = x_phys[1] - x_phys[0]
    xe = np.arange(0.0, XMAX + 0.5 * dx, dx)
    wrap = (np.round(xe / dx).astype(int)) % len(x_phys)
    ax.plot(xe, fields["md"][wrap, ym], color=C_MD, ls="-", lw=ps.LW_CURVE,
            solid_capstyle="round", zorder=4, label="MD")
    ax.plot(xe, fields["aipf"][wrap, ym], color=C_MODEL, ls=(0, (5, 2.2)),
            lw=ps.LW_CURVE * 0.85, dash_capstyle="round", zorder=5, label="AIPF")
    ax.set_xlim(0.0, XMAX)
    ax.set_xticks([0, 20, 40])
    ax.set_ylim(-0.04, 1.62)
    ax.set_xlabel(r"$x\ [\sigma]$", fontsize=ps.FS_LABEL, labelpad=2)
    ax.set_ylabel(r"$x_\mathrm{A}$", fontsize=ps.FS_LABEL, labelpad=2)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=ps.FS_TICK, direction="in", length=ps.TICK_LEN,
                   width=TICK_W)
    for sp in ax.spines.values():
        sp.set_linewidth(ps.LW_AXES)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for lab, c, ls in (("FH", C_FH, LS_FH), ("Landau", C_LANDAU, LS_LANDAU)):
        ax.plot(xe, fields[lab.lower()][wrap, ym], color=c, ls=ls,
                lw=ps.LW_CURVE, dash_capstyle="round", zorder=3, label=lab)
    ax.set_ylim(ax.get_ylim()[0], PROF_YTOP)
    hs, labs = ax.get_legend_handles_labels()
    order = [labs.index(k) for k in ("MD", "AIPF", "FH", "Landau")]
    ax.legend([hs[i] for i in order], [labs[i] for i in order],
              fontsize=ps.FS_LEGEND_SM, frameon=False, loc="upper center",
              ncol=2, handlelength=1.4, handletextpad=0.35, labelspacing=0.2,
              columnspacing=0.8, borderaxespad=0.1)


# ── layout: measured from the rendered columns ───────────────────────────

def box_x_extent(ax, box, fig):
    """Horizontal extent (figure fraction) of the projected box."""
    xs = []
    for cx in box[0]:
        for cy in box[1]:
            for cz in box[2]:
                u, v, _ = proj3d.proj_transform(cx, cy, cz, ax.get_proj())
                xs.append(ax.transData.transform((u, v))[0])
    inv = fig.transFigure.inverted()
    return inv.transform((min(xs), 0))[0], inv.transform((max(xs), 0))[0]


def surface_x_extent(ax, fig):
    zlo, zhi = ax.get_zlim()
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    return box_x_extent(ax, [(xlo, xhi), (ylo, yhi), (zlo, zhi)], fig)


def _union_x(bbs, fig):
    bbs = [b for b in bbs if b is not None and b.width > 0]
    return min(b.x0 for b in bbs) / fig.dpi, max(b.x1 for b in bbs) / fig.dpi


def _text_x(ax, fig):
    r = fig.canvas.get_renderer()
    return _union_x([a.get_tightbbox(r) for a in (ax.xaxis, ax.yaxis, ax.zaxis)]
                    + [ax.title.get_window_extent(r)], fig)


def _shift(ax, dx_in, fig_w):
    p = ax.get_position()
    ax.set_position([p.x0 + dx_in / fig_w, p.y0, p.width, p.height])


def place_columns(fig, ax_v, cubes, ax_p, box):
    """Columns left to right: the two gaps that hold labels sized from the
    rendered text, the cube-to-cube gaps equal, the row centred."""
    w = fig.get_size_inches()[0]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    sv = [x * w for x in surface_x_extent(ax_v, fig)]
    tv = _text_x(ax_v, fig)
    sc = [[x * w for x in box_x_extent(ax, box, fig)] for ax in cubes]
    pp = ax_p.get_position()
    sp = (pp.x0 * w, pp.x1 * w)
    tp = _union_x([ax_p.get_tightbbox(r)], fig)
    g_v = (tv[1] - sv[1]) + PAD_TEXT
    g_p = (sp[0] - tp[0]) + PAD_TEXT
    left_hang = sv[0] - tv[0]
    right_hang = tp[1] - sp[1]
    widths = [sv[1] - sv[0]] + [b - a for a, b in sc] + [sp[1] - sp[0]]
    avail = w - 2 * M_SIDE - left_hang - right_hang
    g = (avail - sum(widths) - g_v - g_p) / (len(cubes) - 1)
    x = M_SIDE + left_hang
    starts = [sv[0]] + [a for a, _ in sc] + [sp[0]]
    gaps = [g_v] + [g] * (len(cubes) - 1) + [g_p]
    for k, ax in enumerate([ax_v] + cubes + [ax_p]):
        _shift(ax, x - starts[k], w)
        x += widths[k] + (gaps[k] if k < len(gaps) else 0.0)


def layer_path(prof, lab):
    return FIGDATA / f"painted_{prof}_{lab}.png"


def rerun() -> str:
    """The command that paints the boxes again and rewrites their layers."""
    return (f"{box3d.RENDER_3D}=1 python figures/{HERE.name}/draw.py "
            f"figures/out/{HERE.name}")


def row(prof, out, stage):
    """One potential's row; ``stage`` collects the layers when the boxes are painted."""
    paint = stage is not None
    d = np.load(FIGDATA / f"vext_{prof}_T1.20.npz")
    V, dx = d["V_ext"], float(d["dx"])
    fields = {k: d[f"x_A_{k}"] for k in ("md", "aipf", "fh", "landau")}
    Nx, Ny = fields["md"].shape
    X, Y = np.meshgrid(np.arange(Nx) * dx, np.arange(Ny) * dx, indexing="ij")
    box = [(0.0, Nx * dx)] * 3
    v_norm = ps.vext_norm(float(V.min()), float(V.max()))

    w, h = ROW_W, ROW_H
    fig = plt.figure(figsize=(w, h))
    y0, y1 = AX3D_BOTTOM / h, 1.0 - AX3D_TOP / h
    ax_v = fig.add_axes([0.0, y0, VEXT_AX_W / w, y1 - y0], projection="3d")
    surface(ax_v, X, Y, V, ps.v_cmap(), v_norm,
            (float(V.min()), float(V.max())), COL_TITLES[0])
    cubes, sources = [], {}
    for j, lab in enumerate(("md", "aipf", "fh", "landau"), start=1):
        ax = fig.add_axes([0.0, y0, CUBE_AX_W / w, y1 - y0], projection="3d")
        f3 = np.broadcast_to(fields[lab][:, :, None], (Nx, Ny, NZ))
        source = box3d.source_key(fields[lab], d["dx"], np.int64(NZ))
        sources[lab] = source
        paint_box(ax, f3, box, COL_TITLES[j], layer_path(prof, lab), source, paint)
        ax.patch.set_visible(False)
        cubes.append(ax)
    ax_p = fig.add_axes([0.0, PROF_BOTTOM / h, PROF_W / w, PROF_H / h])
    profile(ax_p, X[:, 0], fields, Ny // 2)

    place_columns(fig, ax_v, cubes, ax_p, box)
    fig.canvas.draw()
    t_top = cubes[0].title.get_window_extent(fig.canvas.get_renderer()).y1
    y_top = fig.transFigure.inverted().transform((0, t_top))[1]
    pp = ax_p.get_position()
    fig.text(pp.x0 + 0.5 * pp.width, y_top, COL_TITLES[5],
             fontsize=ps.FS_LABEL, ha="center", va="top")
    if paint:
        dpi, fig.dpi = fig.dpi, DPI
        fig.canvas.draw()
        for ax, lab in zip(cubes, ("md", "aipf", "fh", "landau")):
            box3d.save_layer(fig, ax, layer_path(prof, lab), sources[lab], rerun, stage)
        fig.dpi = dpi
    fig.canvas.draw()
    for ax in [ax_v] + cubes:
        for art in ax.collections:
            art.set_rasterized(True)
    return ps.save(fig, out / f"ed_lj_vext_{prof}.png", dpi=DPI)


def draw(out: Path) -> list[Path]:
    out = Path(out)
    ps.use()
    if not box3d.paint_requested(os.environ):
        return [row("hexagonal", out, None), row("checkerboard", out, None)]
    with box3d.staged_layers(FIGDATA) as stage:
        return [row("hexagonal", out, stage), row("checkerboard", out, stage)]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
