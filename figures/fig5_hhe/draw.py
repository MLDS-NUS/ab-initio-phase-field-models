"""Fig. 5: hydrogen-helium phase diagram, helium rain in gas giants and the dynamics, panels and the assembled figure."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Wedge
from matplotlib.text import Text
from mpl_toolkits.mplot3d import proj3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import paperstyle as ps  # noqa: E402
import style_hhe as hs  # noqa: E402

FIGDATA = HERE / "figdata"
# panel letter in the paper -> file: a hhe_main_b.png, b hhe_main_c.pdf, c hhe_main_a.pdf,
# d hhe_main_lit.pdf, e hhe_main_d.pdf, f and g hhe_main_fg.pdf; the section titles below
# use the paper's letters
COMPOSITE = "fig_hhe_main.pdf"
SHIPPED = ("md_snapshot_T05000_slab.png", "md_snapshot_T10000_slab.png",
           "hhe_main_fg.pdf")
REFERENCE = {name: name for name in (
    "hhe_main_a.pdf", "hhe_main_b.png", "hhe_main_c.pdf", "hhe_main_d.pdf",
    "hhe_main_lit.pdf") + SHIPPED + (COMPOSITE,)}
# drawn only where the site declares latexmk (AIPF_LATEXMK, or [site] latexmk)
OPTIONAL = {COMPOSITE: "site fact latexmk is not declared (AIPF_LATEXMK or [site] latexmk in aipf.toml)"}

DPI = 600
H_ROW = 40.0 / 25.4          # first row of panels
H_ROW2 = 58.0 / 25.4         # second row
WA, WB, WC = 26.0 / 25.4, 48.0 / 25.4, 48.0 / 25.4
WD = 45.0 / 25.4
INK = "#15181c"

C_BINODAL = "#1F3A5F"        # navy
C_SPINODAL = "#B5471F"       # rust
C_REF = INK                  # MD references, open markers


def clear(fig):
    """Transparent canvas: the panels sit on the composite's block frames."""
    fig.patch.set_alpha(0.0)
    return fig


# ── c: schematic of helium rain in a giant-planet interior ───────────────
# one hue, lightness = depth: molecular / rain / metallic top / deep / core
C_MOL, C_RAIN, C_MET_HI, C_MET_LO, C_CORE = (
    "#EEF3FB", "#A9BEE8", "#5A72C4", "#2A3475", "#0D1136")
C_EDGE = C_CORE
C_DROP = "#FFFFFF"
C_INK_LO, C_INK_HI = "#12202C", "#FFFFFF"
R_TOP, R_RAIN, R_DIS, R_CORE = 0.92, 0.720, 0.545, 0.380   # not to scale
TH0, TH1 = 68.0, 112.0
CX, CY = 0.38, 0.03
LW_EDGE = 0.7


def _pol(r, deg):
    a = np.radians(deg)
    return CX + r * np.cos(a), CY + r * np.sin(a)


def _ring(ax, r_out, r_in, color, z=1):
    ax.add_patch(Wedge((CX, CY), r_out, TH0, TH1, width=r_out - r_in,
                       facecolor=color, edgecolor="none", zorder=z))


def _arc(ax, r, lw=LW_EDGE):
    t = np.radians(np.linspace(TH0, TH1, 200))
    ax.plot(CX + r * np.cos(t), CY + r * np.sin(t), ls="-", lw=lw,
            color=C_EDGE, zorder=6, solid_capstyle="round")


def _ramp(ax, r_hi, r_lo, c_hi, c_lo, n=48):
    """Overlapping thin rings painted outside-in: a gradient without seams."""
    edges = np.linspace(r_hi, r_lo, n + 1)
    over = 0.5 * (r_hi - r_lo) / n
    a = np.array(matplotlib.colors.to_rgb(c_hi))
    b = np.array(matplotlib.colors.to_rgb(c_lo))
    for i in range(n):
        f = (i + 0.5) / n
        _ring(ax, edges[i] + over, edges[i + 1], tuple(a + f * (b - a)))


def _label(ax, r, text, color=C_INK_LO):
    x, y = _pol(r, 90.0)
    ax.text(x, y, text, ha="center", va="center", fontsize=hs.FS_ANNOT,
            weight="bold", color=color, zorder=12)


def _droplet(ax, r, deg, size, alpha):
    """Teardrop with its tip outward, x = cos t, y = sin t sin^3(t/2)."""
    ang = np.radians(deg)
    er = np.array([np.cos(ang), np.sin(ang)])
    et = np.array([-np.sin(ang), np.cos(ang)])
    c = np.array(_pol(r, deg))
    t = np.linspace(0.0, 2.0 * np.pi, 120)
    X = np.cos(t)
    Y = np.sin(t) * np.sin(t / 2.0) ** 3
    k = size / 0.65
    pts = np.array([c + k * ((X[i] + 0.35) * er + Y[i] * et)
                    for i in range(len(t))])
    ax.add_patch(Polygon(pts, closed=True, facecolor=C_DROP, edgecolor="none",
                         alpha=alpha, zorder=9))


def panel_schematic(out):
    fig = plt.figure(figsize=(WA, H_ROW))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 0.76)
    ax.set_ylim(0.0, 0.99)
    ax.set_aspect("equal")
    ax.axis("off")
    _ring(ax, R_TOP, R_RAIN, C_MOL)
    _ring(ax, R_RAIN, R_DIS, C_RAIN)
    _ramp(ax, R_DIS, R_CORE, C_MET_HI, C_MET_LO)
    _ramp(ax, R_CORE, 0.0, C_MET_LO, C_CORE)
    _arc(ax, R_TOP)
    _arc(ax, R_RAIN, lw=0.9)
    _arc(ax, R_DIS, lw=0.9)
    for deg in (TH0, TH1):
        x, y = _pol(R_TOP, deg)
        ax.plot([CX, x], [CY, y], lw=LW_EDGE, color=C_EDGE, zorder=6)
    _label(ax, 0.835, "Molecular hydrogen")
    _label(ax, 0.640, "Helium rain")
    _label(ax, 0.440, "Metallic\nhydrogen", color=C_INK_HI)
    _label(ax, 0.255, "Dilute\ncore", color=C_INK_HI)
    d = np.load(FIGDATA / "schematic_droplets.npz")
    for r, deg, size, alpha in zip(d["r"], d["angle_deg"], d["size"], d["alpha"]):
        _droplet(ax, r, deg, size, alpha)
    clear(fig)
    return ps.save(fig, out / "hhe_main_a.pdf")


# ── a: the two-phase region over (x_He, P, T) ────────────────────────────
ROYAL = ["#dceafa", "#bcd8f4", "#93bfea", "#6aa2dc", "#3f7fca", "#2360a8",
         "#123f70"]
C_CRITICAL = "#8E2A1B"


def _rule(ax, p0, p1, lw=1.1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax.plot(*np.column_stack([p0, p1]), color="black", lw=lw,
            solid_capstyle="butt", zorder=10)


def _ticks(ax, base_pts, out_dir, labels, pad, size, ha, va, tick):
    """Tick stubs and labels along a hand-drawn axis, in data coordinates."""
    out_dir = np.asarray(out_dir, float)
    for pt, lab in zip(base_pts, labels):
        pt = np.asarray(pt, float)
        ax.plot(*np.column_stack([pt, pt + tick * out_dir]), color=INK,
                lw=0.9, zorder=10)
        q = pt + pad * out_dir
        ax.text(q[0], q[1], q[2], lab, fontsize=size, color=INK, ha=ha,
                va=va, zorder=11)


def _axis_label(ax, p0, p1, text, out_dir, pad, size):
    """Label centred on a hand-drawn axis and rotated along its projection."""
    M = ax.get_proj()
    a = ax.transData.transform(proj3d.proj_transform(*p0, M)[:2])
    b = ax.transData.transform(proj3d.proj_transform(*p1, M)[:2])
    ang = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
    while ang > 90.0:
        ang -= 180.0
    while ang <= -90.0:
        ang += 180.0
    if ang < -60.0:
        ang += 180.0
    q = 0.5 * (np.asarray(p0, float) + np.asarray(p1, float)) \
        + pad * np.asarray(out_dir, float)
    px, py, _ = proj3d.proj_transform(*q, M)
    t = Text(px, py, text, fontsize=size, color=INK, ha="center",
             va="center", rotation=ang, rotation_mode="anchor", zorder=11)
    t.set_transform(ax.transData)
    ax.add_artist(t)
    t.set_clip_on(False)
    t.set_clip_path(None)


def _render_surface(path, width, height):
    s = np.load(FIGDATA / "two_phase_surface.npz")
    X, Y, Z, ridge = s["x_He"], s["P"], s["T"], s["critical_line"]
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.08, 0.0, 0.86, 1.0], projection="3d",
                      computed_zorder=False)
    ax.view_init(elev=16.0, azim=-55.0)
    ax.set_axis_off()
    ax.set_box_aspect((1.5, 1.0, 0.95), zoom=1.16)
    T0, T_TOP, P0, P1 = 2000.0, 10400.0, 200.0, 800.0
    AX_X0 = -0.16
    ax.set_xlim(AX_X0 - 0.04, 1.34)
    ax.set_ylim(P0 - 10, 900.0)
    ax.set_zlim(T0, T_TOP + 600.0)
    ramp = LinearSegmentedColormap.from_list("royal", ROYAL)
    rgba = ramp(Normalize(P0, P1)(Y))
    rgba[..., 3] = 1.0
    ax.plot_surface(X, Y, Z, facecolors=rgba, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=4)
    okr = np.isfinite(ridge[:, 0])
    ax.plot(ridge[okr, 0], ridge[okr, 1], ridge[okr, 2], color=C_CRITICAL,
            lw=1.6, zorder=8)
    O = (AX_X0, P0, T0)
    P_CORNER = 1.06
    _rule(ax, O, (P_CORNER, P0, T0))
    _rule(ax, (P_CORNER, P0, T0), (P_CORNER, 900.0, T0))
    _rule(ax, O, (AX_X0, P0, T_TOP))
    _ticks(ax, [(v, P0, T0) for v in (0.0, 0.5, 1.0)], (0.0, -1.0, 0.0),
           ["0.0", "0.5", "1.0"], pad=105.0, size=hs.FS_TICK, ha="center",
           va="top", tick=16.0)
    _ticks(ax, [(P_CORNER, P, T0) for P in (200.0, 400.0, 600.0, 800.0)],
           (1.0, 0.0, 0.0), ["200", "400", "600", "800"], pad=0.075,
           size=hs.FS_TICK, ha="left", va="center", tick=0.028)
    _ticks(ax, [(AX_X0, P0, T) for T in (4000.0, 6000.0, 8000.0, 10000.0)],
           (-1.0, 0.0, 0.0), ["4", "6", "8", "10"], pad=0.075,
           size=hs.FS_TICK, ha="right", va="center", tick=0.028)
    _axis_label(ax, (AX_X0, P0, T0), (P_CORNER, P0, T0), r"$x_{\mathrm{He}}$",
                (0.0, -1.0, 0.0), 330.0, hs.FS_LABEL)
    _axis_label(ax, (P_CORNER, P0, T0), (P_CORNER, 900.0, T0), r"$P$ [GPa]",
                (1.0, 0.0, 0.0), 0.62, hs.FS_LABEL)
    _axis_label(ax, (AX_X0, P0, T0), (AX_X0, P0, T_TOP), r"$T$ [$10^{3}$ K]",
                (-1.0, 0.0, 0.0), 0.48, hs.FS_LABEL)
    ax.text(0.42, 290.0, T0 + 750.0, "two-phase\nregion",
            fontsize=hs.FS_ANNOT, color=INK, ha="center", va="center",
            zorder=11)
    k = int(np.where(okr)[0][-1])
    ax.text(ridge[k, 0], ridge[k, 1], ridge[k, 2] + 300,
            r"critical line $T_{\mathrm{c}}(P)$", fontsize=hs.FS_ANNOT,
            color=INK, ha="center", va="bottom", zorder=11)
    ps.save(fig, path, dpi=DPI)


def _crop_white(path, pad_px, thresh=250):
    """Trim the white border of a PNG in place; (w, h) of the result in px."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = (a < thresh).any(axis=2)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    y0 = max(int(rows[0]) - pad_px, 0)
    y1 = min(int(rows[-1]) + 1 + pad_px, a.shape[0])
    x0 = max(int(cols[0]) - pad_px, 0)
    x1 = min(int(cols[-1]) + 1 + pad_px, a.shape[1])
    im.crop((x0, y0, x1, y1)).save(path)
    return x1 - x0, y1 - y0


def _pad_to_height(path, height):
    """Centre the cropped panel vertically on a white canvas `height` tall."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    H_px = int(round(height * DPI))
    if im.height < H_px:
        canvas = Image.new("RGB", (im.width, H_px), (255, 255, 255))
        canvas.paste(im, (0, (H_px - im.height) // 2))
        canvas.save(path, dpi=(DPI, DPI))


def _white_to_alpha(path, thresh=250):
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im).copy()
    a[(a[..., :3] >= thresh).all(-1), 3] = 0
    Image.fromarray(a).save(path, dpi=im.info.get("dpi", (DPI, DPI)))


def panel_surface(out):
    """Rendered, cropped to its ink and rescaled until the ink is WB wide,
    then padded to the row height and made transparent outside the ink."""
    path = out / "hhe_main_b.png"
    w, h = 3.0, 2.1
    for _ in range(6):
        _render_surface(path, w, h)
        cw, ch = _crop_white(path, pad_px=3)
        s = WB / (cw / DPI)
        if ch / DPI * s > H_ROW:
            s = H_ROW / (ch / DPI)
        if abs(s - 1.0) < 0.006:
            break
        w, h = w * s, h * s
    else:
        _render_surface(path, w, h)
        _crop_white(path, pad_px=3)
    _pad_to_height(path, 30.0 / 25.4)
    _white_to_alpha(path)
    return path


# ── b: the T-x domes at 200, 400, 600 and 800 GPa ────────────────────────
CENTRAL_LO_MAX = 0.6         # a tie with bin_lo above this is a He-rich wing, not the dome
SMOOTH_ROWS = 1.5            # display smoothing of the drawn branches, in T-grid rows


def _cutover(T, lo, hi, x_step):
    """Rows drawn solid: finite, central-dome ties wider than the noise floor
    max(6 x-grid steps, 0.03); the rest is left to the dashed mean-field cap."""
    central = np.isfinite(lo) & np.isfinite(hi) & (lo < CENTRAL_LO_MAX)
    solid = central & (hi - lo > max(6.0 * float(x_step), 0.03))
    T_last = np.float64(T[solid].max()) if solid.any() else np.float64(np.nan)
    return solid, T_last


def _smooth(y, keep):
    """Gaussian-smooth one branch over the rows it is drawn on."""
    from scipy.ndimage import gaussian_filter1d
    y = np.asarray(y, float).copy()
    idx = np.where(np.asarray(keep) & np.isfinite(y))[0]
    if idx.size < 5:
        return y
    y[idx] = gaussian_filter1d(y[idx], sigma=SMOOTH_ROWS, mode="nearest")
    return y


def _cap_x(T, T_last, x_last, x_apex, Tc):
    """One side of the sqrt cap through (x_last, T_last) and the apex (x_apex, Tc)."""
    ratio = np.clip((Tc - np.asarray(T, float)) / (Tc - T_last), 0.0, 1.0)
    return x_apex + (x_last - x_apex) * np.sqrt(ratio)


def _envelope(T, lo, hi, keep, cap=None):
    """Closed (x, T) outline: up one branch, over the cap, down the other."""
    m = np.asarray(keep) & np.isfinite(lo) & np.isfinite(hi)
    if m.sum() < 2:
        return None
    Tk, lok, hik = T[m], lo[m], hi[m]
    o = np.argsort(Tk)
    Tk, lok, hik = Tk[o], lok[o], hik[o]
    if cap is not None:
        Tc_, cl, ch = cap
        return (np.concatenate([lok, cl, ch[::-1], hik[::-1]]),
                np.concatenate([Tk, Tc_, Tc_[::-1], Tk[::-1]]))
    return np.concatenate([lok, hik[::-1]]), np.concatenate([Tk, Tk[::-1]])


def _dome(ax, d, P, first):
    """One pressure's T-x binodal and spinodal: solid through the resolved
    ties, a dashed sqrt cap to the critical point, the spinodal held inside
    the binodal, two washes, and the MD binodal states."""
    k = f"_P{P}"
    T, x_step, Tc = d["T" + k], float(d["x_step" + k]), float(d["Tc" + k])
    solid, T_last = _cutover(T, d["binodal_lo" + k], d["binodal_hi" + k], x_step)
    bin_lo = _smooth(d["binodal_lo" + k], solid)
    bin_hi = _smooth(d["binodal_hi" + k], solid)
    i_last = int(np.where(solid)[0][-1])
    x_apex = 0.5 * (bin_lo[i_last] + bin_hi[i_last])
    ax.plot(np.r_[np.where(solid, bin_lo, np.nan), np.where(solid, bin_hi, np.nan)[::-1]],
            np.r_[T, T[::-1]], "-", color=C_BINODAL, lw=hs.LW_CURVE,
            solid_joinstyle="round", solid_capstyle="round", zorder=4)
    bin_cap = None
    if np.isfinite(Tc) and np.isfinite(T_last) and T_last < Tc:
        T_cap = np.linspace(float(T_last), Tc, 40)
        cl = _cap_x(T_cap, T_last, bin_lo[i_last], x_apex, Tc)
        ch = _cap_x(T_cap, T_last, bin_hi[i_last], x_apex, Tc)
        bin_cap = (T_cap, cl, ch)
        ax.plot(np.r_[cl, ch[::-1]], np.r_[T_cap, T_cap[::-1]], ":",
                color=C_BINODAL, lw=hs.LW_CURVE, zorder=4)
    ax.plot([x_apex], [Tc], "D", ms=hs.MS_MARKER, color=C_SPINODAL,
            mfc=C_SPINODAL, mec="white", mew=hs.MEW_MARKER, ls="none",
            zorder=7)

    # a spinodal end outside the binodal at the same T, or past its closure, is dropped
    sp_lo, sp_hi = d["spinodal_lo" + k], d["spinodal_hi" + k]
    fin_sp = np.isfinite(sp_lo) & np.isfinite(sp_hi)
    fin_bin = np.isfinite(bin_lo) & np.isfinite(bin_hi)
    beyond = (fin_sp & ~fin_bin) | (fin_sp & fin_bin & (
        (sp_lo < bin_lo - 1e-6) | (sp_hi > bin_hi + 1e-6)))
    sp_lo = np.where(beyond, np.nan, sp_lo)
    sp_hi = np.where(beyond, np.nan, sp_hi)
    sp_solid, sp_T_last = _cutover(T, sp_lo, sp_hi, x_step)
    sp_lo = _smooth(sp_lo, sp_solid)
    sp_hi = _smooth(sp_hi, sp_solid)
    ax.plot(np.r_[np.where(sp_solid, sp_lo, np.nan), np.where(sp_solid, sp_hi, np.nan)[::-1]],
            np.r_[T, T[::-1]], "--", color=C_SPINODAL, lw=hs.LW_CURVE,
            dash_capstyle="round", zorder=3)
    sp_cap = None
    if sp_solid.any() and np.isfinite(Tc) and np.isfinite(sp_T_last) and sp_T_last < Tc:
        i_sp = int(np.where(sp_solid)[0][-1])
        T_cap = np.linspace(float(sp_T_last), Tc, 40)
        cl = _cap_x(T_cap, sp_T_last, sp_lo[i_sp], x_apex, Tc)
        ch = _cap_x(T_cap, sp_T_last, sp_hi[i_sp], x_apex, Tc)
        if bin_cap is not None:
            bl = _cap_x(T_cap, T_last, bin_lo[i_last], x_apex, Tc)
            bh = _cap_x(T_cap, T_last, bin_hi[i_last], x_apex, Tc)
            cl, ch = np.maximum(cl, bl), np.minimum(ch, bh)
        sp_cap = (T_cap, cl, ch)
        ax.plot(np.r_[cl, ch[::-1]], np.r_[T_cap, T_cap[::-1]], ":",
                color=C_SPINODAL, lw=hs.LW_CURVE, zorder=3)

    if "md_binodal_x" + k in d:
        for xs, Ts in zip(d["md_binodal_x" + k], d["md_binodal_T" + k]):
            ax.plot(xs, Ts, "o", ms=2.2, mfc="none", mec=C_REF,
                    mew=hs.MEW_MARKER + 0.15, ls="none", zorder=6,
                    clip_on=False)
    env = _envelope(T, bin_lo, bin_hi, solid, bin_cap)
    ax.fill(*env, facecolor=C_BINODAL, alpha=0.075, edgecolor="none", zorder=1)
    env = _envelope(T, sp_lo, sp_hi, sp_solid, sp_cap)
    # knock the navy wash out inside the spinodal before tinting it rust
    ax.fill(*env, facecolor="white", alpha=0.85, edgecolor="none", zorder=2)
    ax.fill(*env, facecolor=C_SPINODAL, alpha=0.12, edgecolor="none", zorder=2)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$x_{\mathrm{He}}$", labelpad=1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    if not first:
        ax.tick_params(labelleft=False)
    ps.open_axes(ax)
    return x_apex


def panel_domes(out, pressures=(200, 400, 600, 800), ncol=2):
    d = dict(np.load(FIGDATA / "domes.npz"))
    nrow = int(np.ceil(len(pressures) / ncol))
    w, h = WC, H_ROW2
    m_left, m_right, gap_x, gap_y = 0.42, 0.05, 0.10, 0.20
    top_band, title_band, bot = hs.M_TOP, 0.15, 0.30
    col = (w - m_left - m_right - (ncol - 1) * gap_x) / ncol
    ax_h = (h - top_band - bot - nrow * title_band - (nrow - 1) * gap_y) / nrow
    fig = plt.figure(figsize=(w, h))
    for i, P in enumerate(pressures):
        r_, c_ = divmod(i, ncol)
        x0 = m_left + c_ * (col + gap_x)
        y0 = bot + (nrow - 1 - r_) * (ax_h + title_band + gap_y)
        ax = fig.add_axes([x0 / w, y0 / h, col / w, ax_h / h])
        x_apex = _dome(ax, d, P, first=(c_ == 0))
        ax.set_title(f"{P} GPa", fontsize=hs.FS_ANNOT, pad=2.0)
        Tc = float(d[f"Tc_P{P}"])
        if np.isfinite(Tc) and np.isfinite(x_apex):
            ax.annotate(f"{Tc:,.0f} K", xy=(x_apex, Tc), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=hs.FS_ANNOT, color=C_SPINODAL, zorder=9,
                        annotation_clip=False)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"] if r_ == nrow - 1 else ["", "", ""])
        ax.tick_params(axis="x", direction="in")
        ax.set_xlabel("")
        ax.set_ylim(1900, 11000)
        ax.set_yticks([2000, 4000, 6000, 8000, 10000])
        ax.yaxis.set_major_formatter(lambda v, _p: f"{v / 1000:.0f}")
        ax.set_ylabel(r"$T$ [$10^{3}$ K]" if c_ == 0 else "", labelpad=2)
        if c_:
            ax.tick_params(labelleft=False)
    fig.text((m_left + 0.5 * col) / w, 0.03 / h, r"$x_\mathrm{He}$",
             ha="center", va="bottom", fontsize=hs.FS_LABEL)
    items = [
        (Line2D([], [], color=C_BINODAL, lw=hs.LW_CURVE), "binodal"),
        (Line2D([], [], color=C_SPINODAL, lw=hs.LW_CURVE, ls="--"), "spinodal"),
        (Line2D([], [], color=C_SPINODAL, marker="D", ls="none", mec="white",
                mew=hs.MEW_MARKER, ms=hs.MS_MARKER), r"$T_\mathrm{c}$"),
        (Line2D([], [], color=C_REF, marker="o", ls="none", mfc="none",
                mew=hs.MEW_MARKER + 0.15, ms=2.2), "MD, 800 GPa"),
    ]
    hh, ll = zip(*items)
    leg = fig.legend(hh, ll, ncol=2, frameon=True, fontsize=hs.FS_LEGEND,
                     handlelength=1.6, handletextpad=0.4, columnspacing=1.0,
                     borderpad=0.3, labelspacing=0.2, borderaxespad=0.0,
                     loc="upper center",
                     bbox_to_anchor=((m_left + 0.5 * (w - m_left - m_right)) / w,
                                     1.0 - 0.015 / h))
    leg.get_frame().set_linewidth(hs.LW_AXES)
    leg.get_frame().set_edgecolor("0.75")
    clear(fig)
    return ps.save(fig, out / "hhe_main_c.pdf")


# ── d, e: the protosolar isopleth (x_He = 0.089) against P ───────────────
DASH_SPIN = (0, (2.6, 1.2))
LW_OURS, LW_REF = 1.6, 1.0
A_META, A_UNSTABLE, A_OUT_OF_DOMAIN = 0.075, 0.12, 0.30
P_VALID_MIN = 150.0          # below this hydrogen is molecular
P_MAX = 900.0                # past 900 GPa the isobar leaves the density domain
C_JUPITER, C_SATURN = "#7B2D8E", "#1F8A6D"
PROFILES = {"J": ("Jupiter_Nettelmann", "Jupiter_Militzer", "Jupiter_RR_2018",
                  "Jupiter_today_2024"),
            "S": ("Saturn_isentrope_Nettelmann", "Saturn_present_MF20",
                  "Saturn_today_2024")}
# the literature: legend name, data key, marker
LITERATURE = (("Schöttler 2018", "schottler", "X"),
              ("Morales 2013", "morales", "s"),
              ("Lorenzen 2011", "lorenzen", "^"))
LIT_COLOR = {"Schöttler 2018": "#CC79A7", "Morales 2013": "#009E73",
             "Lorenzen 2011": "#56B4E9", "Wang 2026": "#C9B037"}
WANG_NODES_P = (100.0, 150.0, 200.0, 400.0, 600.0, 800.0, 1000.0)


def _last_in_domain(P, inside):
    ok = P[inside.astype(bool)]
    return float(ok.max()) if ok.size else None


def _split(P, T, p_cut):
    """(trusted, extrapolated) halves sharing one point, so they join."""
    if p_cut is None or P.max() <= p_cut:
        return (P, T), (P[:0], T[:0])
    return (P[P <= p_cut], T[P <= p_cut]), (P[P >= p_cut], T[P >= p_cut])


def _isopleth(ax, ylo):
    """This work's binodal and spinodal, with the unstable and metastable
    washes; a curve leaving the density domain continues as a faint ghost."""
    iso = np.load(FIGDATA / "isopleth.npz")
    from scipy.ndimage import gaussian_filter1d
    curves = []
    for which in ("binodal", "spinodal"):
        P, T = iso[f"{which}_P"], iso[f"{which}_T"]
        # read off a 100 K grid, the curve steps; smoothed along P over one 25 GPa step
        step = float(np.median(np.diff(P)))
        curves.append((P, gaussian_filter1d(T, 25.0 / step, mode="nearest")))
    (bP, bT), (sP, sT) = curves
    m = (bP >= P_VALID_MIN) & (bP <= P_MAX)
    bP, bT = bP[m], bT[m]
    m = (sP >= P_VALID_MIN) & (sP <= P_MAX)
    sP, sT = sP[m], sT[m]
    p_bi = _last_in_domain(iso["domain_P"], iso["binodal_in_domain"])
    p_sp = _last_in_domain(iso["domain_P"], iso["spinodal_in_domain"])
    if len(sP):
        lo = max(bP.min(), sP.min())
        hi = min(bP.max(), sP.max(), p_sp if p_sp is not None else np.inf)
        g = np.linspace(lo, hi, 400)
        Tsp = np.interp(g, sP, sT)
        Tbi = np.interp(g, bP, bT)
        ax.fill_between(g, ylo, Tsp, color=C_SPINODAL, alpha=A_UNSTABLE,
                        lw=0, zorder=0)
        ax.fill_between(g, Tsp, Tbi, color=C_BINODAL, alpha=A_META, lw=0,
                        zorder=0)
    (P_in, T_in), (P_out, T_out) = _split(bP, bT, p_bi)
    lines = [ax.plot(P_in, T_in, "-", color=C_BINODAL, lw=LW_OURS, zorder=8,
                     solid_capstyle="round")[0]]
    if len(P_out):
        lines += ax.plot(P_out, T_out, "-", color=C_BINODAL, lw=LW_OURS,
                         zorder=8, solid_capstyle="round",
                         alpha=A_OUT_OF_DOMAIN)
    if len(sP):
        (P_in, T_in), (P_out, T_out) = _split(sP, sT, p_sp)
        lines += ax.plot(P_in, T_in, ls=DASH_SPIN, color=C_SPINODAL,
                         lw=LW_OURS, zorder=8)
        if len(P_out):
            lines += ax.plot(P_out, T_out, ls=DASH_SPIN, color=C_SPINODAL,
                             lw=LW_OURS, zorder=8, alpha=A_OUT_OF_DOMAIN)
    return lines


def _frame(ax, xlim, ylim):
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$P$ [GPa]")
    ax.set_ylabel(r"$T$ [$10^{3}$ K]")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}"))
    # ticks before the limit: tick locations widen the view interval
    ax.set_xticks([t for t in (200, 400, 600, 800, 1000)
                   if xlim[0] <= t <= xlim[1]])
    ax.set_xlim(*xlim)
    ax.grid(False)


def _planet_bands(ax, xlim):
    """Envelope of the published interior profiles of each planet."""
    prof = np.load(FIGDATA / "planet_profiles.npz")
    grid = np.linspace(*xlim, 500)
    for planet, c in (("J", C_JUPITER), ("S", C_SATURN)):
        stack = np.vstack([np.interp(grid, prof[f"{n}_P"], prof[f"{n}_T"],
                                     left=np.nan, right=np.nan)
                           for n in PROFILES[planet]])
        valid = np.isfinite(stack).all(0)
        lo, hi = np.nanmin(stack[:, valid], 0), np.nanmax(stack[:, valid], 0)
        ax.fill_between(grid[valid], lo, hi, color=c, alpha=0.16, lw=0,
                        zorder=2)
        ax.plot(grid[valid], lo, color=c, lw=0.5, alpha=0.7, zorder=2)
        ax.plot(grid[valid], hi, color=c, lw=0.5, alpha=0.7, zorder=2)


def _panel_axes(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes(ps.axes_rect(w, h, left=0.42, right=0.08,
                                   top=hs.M_TOP, bottom=0.32))
    return fig, ax


def panel_planets(out):
    fig, ax = _panel_axes(WD, H_ROW)
    xlim, ylim = (100.0, 1000.0), (2000.0, 14000.0)
    _isopleth(ax, ylim[0])
    _planet_bands(ax, xlim)
    _frame(ax, xlim, ylim)
    ax.set_xticks([200, 400, 600, 800])
    ax.set_yticks([2000, 6000, 10000, 14000])
    handles = [
        Line2D([], [], color=C_BINODAL, lw=LW_OURS),
        Line2D([], [], color=C_SPINODAL, lw=LW_OURS, ls=DASH_SPIN),
        Patch(facecolor=C_JUPITER, alpha=0.35, edgecolor=C_JUPITER, lw=0.5),
        Patch(facecolor=C_SATURN, alpha=0.35, edgecolor=C_SATURN, lw=0.5),
    ]
    ax.legend(handles, ["binodal", "spinodal", "Jupiter", "Saturn"],
              loc="upper left", bbox_to_anchor=(0.0, 1.02),
              fontsize=hs.FS_LEGEND, frameon=False, handlelength=1.4,
              handletextpad=0.4, labelspacing=0.2, borderaxespad=0.0)
    clear(fig)
    return ps.save(fig, out / "hhe_main_d.pdf")


def panel_literature(out):
    lit = np.load(FIGDATA / "literature.npz")
    fig, ax = _panel_axes(WD, H_ROW)
    xlim, ylim = (100.0, 1000.0), (2000.0, 11000.0)
    c = LIT_COLOR["Wang 2026"]
    wP, wT = lit["wang2026_spinodal_P"], lit["wang2026_spinodal_T"]
    # one curve, two artists: the line through the published contour, the
    # diamonds only at the pressures that were computed
    wang = ax.plot(wP, wT, color=c, ls=DASH_SPIN, lw=0.8, alpha=0.85,
                   zorder=3, mec=c, mfc=c, ms=1.8)[0]
    node = np.isin(wP, WANG_NODES_P)
    twin = ax.plot(wP[node], wT[node], ls="none", marker="D", ms=1.8, mec=c,
                   mfc=c, alpha=0.85, zorder=3, clip_on=False, color=c,
                   lw=0.8)[0]
    by = {"Wang 2026": (wang, twin)}
    for name, key, marker in LITERATURE:
        c = LIT_COLOR[name]
        ln = ax.plot(lit[f"{key}_binodal_P"], lit[f"{key}_binodal_T"],
                     marker=marker, ls="-", lw=0.8, ms=1.8, color=c, mec=c,
                     mfc=c, alpha=0.85, zorder=3)[0]
        by[name] = (ln, None)
    for ln in _isopleth(ax, ylim[0]):
        ln.set_linewidth(1.8)
    _frame(ax, xlim, ylim)

    def key(name):
        ln, tw = by[name]
        m = tw if tw is not None else ln
        return Line2D([], [], color=ln.get_color(), ls=ln.get_linestyle(),
                      lw=ln.get_linewidth(), marker=m.get_marker(),
                      ms=m.get_markersize(), mfc=m.get_markerfacecolor(),
                      mec=m.get_markeredgecolor())
    order = ["Schöttler 2018", "Morales 2013", "Lorenzen 2011", "Wang 2026"]
    ax.legend([key(k) for k in order], order, loc="upper right",
              bbox_to_anchor=(1.03, 1.12), ncol=1, fontsize=hs.FS_LEGEND,
              frameon=False, handlelength=1.4, handletextpad=0.35,
              labelspacing=0.2, borderaxespad=0.0)
    ax.set_xticks([200, 400, 600, 800])
    ax.set_yticks([2000, 6000, 10000])
    clear(fig)
    return ps.save(fig, out / "hhe_main_lit.pdf")


# ── the assembled figure ──────────────────────────────────────────────────
def latexmk():
    """The latexmk the site declares, or None."""
    from aipf.site import Site
    return Site.load().latexmk


def assemble(out, exe):
    env = dict(os.environ, PATH=f"{Path(exe).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    with tempfile.TemporaryDirectory() as build:
        run = subprocess.run(
            [str(exe), "-pdf", "-interaction=nonstopmode", "-halt-on-error",
             f"-outdir={build}", "-jobname=fig_hhe_main", str(FIGDATA / "assemble.tex")],
            cwd=out, env=env, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"latexmk failed:\n{run.stdout[-3000:]}\n{run.stderr[-2000:]}")
        return Path(shutil.copyfile(Path(build) / COMPOSITE, out / COMPOSITE))


def draw(out: Path) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        matplotlib.rcdefaults()
        hs.use_nature()
        files = [panel_surface(out), panel_domes(out), panel_schematic(out),
                 panel_literature(out), panel_planets(out)]
    files += [Path(shutil.copyfile(FIGDATA / name, out / name)) for name in SHIPPED]
    exe = latexmk()
    if exe is not None:
        files.append(assemble(out, exe))
    return files


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
