r"""Drawing style of the paper's figures: every panel at its final printed size.

A panel is written at the size it occupies in the manuscript, so LaTeX places
it 1:1 and point sizes here are point sizes on the page. Margins are in
inches, never in figure fractions, so that panels sharing a row line up, and
files are saved without a tight bounding box, which would crop the canvas and
rescale the type.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ── page geometry, inches ────────────────────────────────────────────────
W_FULL = 7.06      # figure* of the main text, 510 pt
W_COL = 3.40       # one column of the main text, 246 pt
W_SM = 6.50        # \linewidth of the Supplemental Material, 468 pt

# ── panel sizes, inches ──────────────────────────────────────────────────
PANEL = {
    "a": (2.05, 2.40),     # free energy
    "b": (2.45, 2.40),     # curvature, with the T colorbar
    "c": (2.35, 2.40),     # phase diagram
    "d": (7.06, 1.80),     # equilibrium under V_ext
    "e": (4.08, 1.25),     # coarsening, painted boxes
    "f": (2.81, 1.25),     # domain size L(t)
}

# ── margins, inches ──────────────────────────────────────────────────────
M_TOP = 0.34
M_BOTTOM = 0.42
M_LEFT = 0.56
M_RIGHT = 0.14
M_RIGHT_CBAR = 0.46   # right margin of a panel that carries a colorbar
CBAR_W = 0.085

# ── quantities shared across panels ──────────────────────────────────────
T_MIN, T_MAX, T_STEP = 1.0, 1.5, 0.1   # temperature family and its colorbar
T_C_MD = 1.423                          # critical temperature of the MD reference

# ── type, absolute point sizes ───────────────────────────────────────────
FS_BASE = 8.0
FS_LABEL = 9.0
FS_TICK = 8.0
FS_LEGEND = 8.0
FS_LEGEND_SM = 7.0
FS_ANNOT = 8.0

LW_AXES = 0.8
LW_CURVE = 1.2
TICK_LEN = 2.6
MS_MARKER = 3.6
MEW_MARKER = 0.6

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "it",
    "font.size": FS_BASE,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.major.size": TICK_LEN,
    "ytick.major.size": TICK_LEN,
    "xtick.major.width": LW_AXES,
    "ytick.major.width": LW_AXES,
    "axes.linewidth": LW_AXES,
    "lines.linewidth": LW_CURVE,
    "axes.unicode_minus": True,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.28,
    "legend.borderaxespad": 0.25,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# ── colour ───────────────────────────────────────────────────────────────
# Composition x_A: deep orange - white - navy, white at the 50/50 mixture.
RHO_CMAP = LinearSegmentedColormap.from_list(
    "deep_orange_navy", ["#B5471F", "#F5F5F5", "#1F3A5F"])
# Temperature family: six discrete steps, cold blue to hot red.
T_CMAP = ListedColormap(["#16324F", "#2E5E99", "#7FADDC",
                         "#E59A72", "#B5471F", "#7A2E11"], name="discrete_rb")
C_MD = "#2E5E99"
C_MODEL = "#B5471F"
C_GRID = "#D9D9D9"
C_NEUTRAL = "#8A8A8A"


def v_cmap():
    """Diverging map of the external potential (Crameri's vik)."""
    from cmcrameri import cm
    return cm.vik


def vext_norm(vmin, vmax):
    """Norm that puts the neutral colour of v_cmap() exactly on V = 0."""
    import matplotlib.colors as mc
    return mc.TwoSlopeNorm(vmin=min(vmin, -1e-9), vcenter=0.0,
                           vmax=max(vmax, 1e-9))


def shades(base, n=3, lo=0.55, hi=1.45):
    """n lightness variants of `base`, dark to light (one per seed)."""
    import colorsys
    import matplotlib.colors as mc
    h, l, sat = colorsys.rgb_to_hls(*mc.to_rgb(base))
    return [mc.to_hex(colorsys.hls_to_rgb(h, min(0.92, l * f), sat))
            for f in np.linspace(lo, hi, n)]


def use():
    """Apply the shared rcParams."""
    plt.rcParams.update(RC)


def axes_rect(fig_w, fig_h, left=M_LEFT, right=M_RIGHT,
              top=M_TOP, bottom=M_BOTTOM):
    """Figure-fraction rect [x0, y0, w, h] from margins given in inches."""
    return [left / fig_w, bottom / fig_h,
            1.0 - (left + right) / fig_w, 1.0 - (top + bottom) / fig_h]


def cbar_rect(fig_w, fig_h, rect, pad=0.06, frac_h=0.72):
    """Slim vertical colorbar rect just right of `rect`, sizes in inches."""
    x0, y0, w, h = rect
    return [x0 + w + pad / fig_w, y0 + 0.5 * (1 - frac_h) * h,
            CBAR_W / fig_w, frac_h * h]


def temperatures():
    """The temperatures every panel of the family colours its curves by."""
    return np.round(np.arange(T_MIN, T_MAX + 1e-9, T_STEP), 2)


def norm():
    return mpl.colors.Normalize(vmin=T_MIN, vmax=T_MAX)


def open_axes(ax, keep_top=False, keep_right=False):
    """Drop the top and right spines and their ticks."""
    if not keep_right:
        ax.spines["right"].set_visible(False)
    if not keep_top:
        ax.spines["top"].set_visible(False)
    ax.tick_params(top=False, right=False)


def _data_ceiling_under(ax, bb):
    """Highest plotted y inside the x-span of `bb` (data coordinates)."""
    top = -np.inf
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        if len(x) == 0:
            continue
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = (x >= bb.x0) & (x <= bb.x1) & np.isfinite(y)
        if m.any():
            top = max(top, float(y[m].max()))
    return top


def clear_artist(fig, ax, art, margin_pt=4.0, max_iter=12):
    """Raise the top of the y range until `art` (a legend or text anchored to
    the top of the axes) clears the curves under it by `margin_pt` points."""
    for _ in range(max_iter):
        fig.canvas.draw()
        bb = art.get_window_extent().transformed(ax.transData.inverted())
        ceiling = _data_ceiling_under(ax, bb)
        lo, hi = ax.get_ylim()
        if not np.isfinite(ceiling):
            return hi
        p0, p1 = ax.transData.inverted().transform(
            [(0.0, 0.0), (0.0, margin_pt * fig.dpi / 72.0)])
        margin = abs(p1[1] - p0[1])
        gap = bb.y0 - ceiling - margin
        if gap >= -1e-3 * (hi - lo):
            return hi
        ax.set_ylim(lo, hi - gap * 1.25)
    raise RuntimeError(f"artist still overlaps the data after {max_iter} lifts")


def save(fig, path, dpi=600):
    """Write the figure at exactly its figsize: PDF as vector, PNG at `dpi`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".png":
        fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0.0)
    else:
        fig.savefig(path, bbox_inches=None, pad_inches=0.0,
                    metadata={"CreationDate": None})
    plt.close(fig)
    return path
