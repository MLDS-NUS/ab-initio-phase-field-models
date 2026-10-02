"""Fig. 3: binary Lennard-Jones phase diagram (a) and domain growth at T = 1.20 (b)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms as mtransforms
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import paperstyle as ps  # noqa: E402

FIGDATA = HERE / "figdata"
REFERENCE = {
    "fig3a_pd_baselines_1col.pdf": "fig3a_pd_baselines_1col.pdf",
    "fig3_growth_baselines_1col.pdf": "fig3_growth_baselines_1col.pdf",
}

# phase-diagram colours: binodal, spinodal, metastable band, unstable core, MD
PD_BINODAL = "#1F6FB2"
PD_SPIN = "#5B9BD5"
PD_META = "#CFE2F3"
PD_CORE = "#F4D3B8"
PD_CORE_TXT = "#BE6A2A"
PD_STAR_REF = "#1A1A1A"

# one frame for a and b: width, margins (inches), y-label distance (points)
FRAME = dict(width_pt=246.0, margins=dict(left=0.56, right=0.14, top=0.10,
                                          bottom=0.42), ylabel_pt=20.5)
# a: big AIPF panel left, Landau over Flory-Huggins as small panels right
A_HEIGHT_PT = 180.0
BIG_X1_PT, SMALL_X0_PT, SMALL_GAP_PT = 158.0, 176.0, 8.0
Y_TOP = 1.65
K_SMALL = 0.7
DISPLAY = {"FH": "Flory–Huggins"}

# b: per-family seed colours, seed 0 to 2 = dark to light
B_HEIGHT_PT = 110.0
RAMPS = {
    "MD": ("#0C5290", "#2F6FB8", "#578DDA"),
    "AIPF": ("#9A2D00", "#C24A17", "#E86A35"),
    "Landau": ("#107E18", "#2CA02C", "#54C14C"),
    "FH": ("#5F2584", "#7F43A4", "#A061C5"),
}
LW_SEED = ps.LW_CURVE * 0.75
BASE_Z = 1.9          # the baselines lie under MD and AIPF


def pin_ylabel(ax, dist_pt):
    """Put the y label's frame-side edge dist_pt points left of the frame."""
    ax.yaxis.set_label_coords(0.0, 0.5, transform=mtransforms.offset_copy(
        ax.transAxes, fig=ax.figure, x=-dist_pt, y=0.0, units="points"))


# ── a: phase diagram ─────────────────────────────────────────────────────

def smooth_dome(T, bL, bR, n=400, T_top=None):
    """Cubic-spline (T, bL / bR), closed at T_top on x = midpoint."""
    finite = np.isfinite(bL) & np.isfinite(bR)
    Tm, bLm, bRm = T[finite], bL[finite], bR[finite]
    if len(Tm) < 4:
        return Tm, bLm, bRm
    order = np.argsort(Tm)
    Tm, bLm, bRm = Tm[order], bLm[order], bRm[order]
    sL, sR = CubicSpline(Tm, bLm), CubicSpline(Tm, bRm)
    T_top = T_top if T_top is not None else Tm[-1]
    T_fine = np.linspace(Tm[0], T_top, n)
    bL_fine, bR_fine = sL(T_fine), sR(T_fine)
    bL_fine[-1] = 0.5 * (bL_fine[-1] + bR_fine[-1])
    bR_fine[-1] = bL_fine[-1]
    return T_fine, bL_fine, bR_fine


def draw_dome(ax, d, ref, T_min, T_max, spinodal_label=True, label_fs=19,
              tick_fs=15, spin_fs=13.5, lw_bin=2.6, lw_spin=1.7, ms_md=7.0,
              mew_md=1.5, ms_star=17, n_ref=14):
    """Binodal, spinodal, metastable band, unstable core, MD binodal and
    the two critical points on one (x_A, T) panel."""
    T, bL, bR = d["T"], d["binodal_L"], d["binodal_R"]
    sL, sR = d["spinodal_L"], d["spinodal_R"]
    Tc = float(T[np.isfinite(bL)].max()) if np.isfinite(bL).any() else float("nan")
    if np.isfinite(float(d["Tc_exact"])):
        Tc = float(d["Tc_exact"])

    Td, bLd, bRd = smooth_dome(T, bL, bR, n=700, T_top=Tc)
    Ts, sLd, sRd = smooth_dome(T, sL, sR, n=700, T_top=Tc)
    sLg = np.minimum(np.interp(Td, Ts, sLd), 0.5)
    sRg = np.maximum(np.interp(Td, Ts, sRd), 0.5)
    have_spin = (np.isfinite(sL) & np.isfinite(sR)).sum() >= 4

    if have_spin:
        ax.fill_betweenx(Td, sLg, sRg, color=PD_CORE, alpha=0.9, lw=0, zorder=1)
        ax.fill_betweenx(Td, bLd, sLg, color=PD_META, alpha=0.9, lw=0, zorder=1)
        ax.fill_betweenx(Td, sRg, bRd, color=PD_META, alpha=0.9, lw=0, zorder=1)
    else:
        ax.fill_betweenx(Td, bLd, bRd, color=PD_META, alpha=0.9, lw=0, zorder=1)
    ax.plot(bLd, Td, "-", color=PD_BINODAL, lw=lw_bin, zorder=5)
    ax.plot(bRd, Td, "-", color=PD_BINODAL, lw=lw_bin, zorder=5)
    if have_spin:
        ax.plot(sLd, Ts, "--", color=PD_SPIN, lw=lw_spin, dashes=(5, 3), zorder=4)
        ax.plot(sRd, Ts, "--", color=PD_SPIN, lw=lw_spin, dashes=(5, 3), zorder=4)

    keep = ref["T"] <= ps.T_C_MD
    Tr, rL, rR = ref["T"][keep], ref["x_L"][keep], ref["x_R"][keep]
    step = max(1, len(Tr) // n_ref)
    idx = np.unique(np.r_[np.arange(0, len(Tr), step), len(Tr) - 1])
    for r in (rL, rR):
        ax.plot(r[idx], Tr[idx], "o", ms=ms_md, mfc="white", mec=PD_STAR_REF,
                mew=mew_md, zorder=6)
    ax.plot(0.5, ps.T_C_MD, marker="*", color=PD_STAR_REF, ms=ms_star,
            mec="none", zorder=8)
    ax.plot(0.5, Tc, marker="*", color=PD_BINODAL, ms=ms_star, mec="none",
            zorder=9)
    if spinodal_label and have_spin:
        ax.text(0.5, T_min + 0.55 * (Tc - T_min), "spinodal", ha="center",
                va="center", fontsize=spin_fs, color=PD_CORE_TXT, zorder=7)

    ax.set_xlim(0, 1)
    ax.set_ylim(T_min, T_max)
    ax.set_xlabel(r"composition  $x_\mathrm{A}$", fontsize=label_fs)
    ax.set_ylabel(r"$\mathrm{T}\;\;[\,\varepsilon/k_{B}\,]$", fontsize=label_fs)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(labelsize=tick_fs)
    for spine in ax.spines.values():
        spine.set_zorder(10)
    return Tc


def small_panel(fig, rect_pt, d, ref, name, ylim, xlabel):
    """Small phase diagram of a baseline: no legend, its name top left."""
    W, H = fig.get_size_inches() * 72.0
    x0, y0, w, h = rect_pt
    fs, k = ps.FS_LEGEND_SM, K_SMALL
    ax = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    draw_dome(ax, d, ref, ylim[0], ylim[1], spinodal_label=False,
              label_fs=fs, tick_fs=fs, spin_fs=fs,
              lw_bin=ps.LW_CURVE * 1.3 * k, lw_spin=ps.LW_CURVE * k,
              ms_md=ps.MS_MARKER * 0.8 * k, mew_md=ps.MEW_MARKER * 1.3 * k,
              ms_star=ps.MS_MARKER * 2.2 * k, n_ref=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ps.open_axes(ax)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0.0", "0.5", "1.0"])
    ax.set_yticks([1.2, 1.4])
    ax.set_ylabel("")
    ax.tick_params(labelsize=fs, length=ps.TICK_LEN * k, pad=1.5)
    if xlabel:
        ax.set_xlabel(r"$x_\mathrm{A}$", fontsize=fs, labelpad=1.0)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    ax.text(0.04, 0.97, DISPLAY.get(name, name), transform=ax.transAxes,
            ha="left", va="top", fontsize=fs, color=PD_STAR_REF, zorder=10)


def panel_a(out):
    datas = {lab: dict(np.load(FIGDATA / f"phase_diagram_{lab.lower()}.npz"))
             for lab in ("AIPF", "Landau", "FH")}
    ref = dict(np.load(FIGDATA / "md_binodal.npz"))
    W, H = FRAME["width_pt"], A_HEIGHT_PT
    m = FRAME["margins"]
    w0, h0 = ps.PANEL["c"]

    d = datas["AIPF"]
    fig = plt.figure(figsize=(W / 72.0, H / 72.0))
    ax = fig.add_axes(ps.axes_rect(w0, h0))
    Tc = draw_dome(ax, d, ref, ps.T_MIN, 1.52, label_fs=ps.FS_LABEL,
                   tick_fs=ps.FS_TICK, spin_fs=ps.FS_ANNOT,
                   lw_bin=ps.LW_CURVE * 1.3, lw_spin=ps.LW_CURVE,
                   ms_md=ps.MS_MARKER * 0.8, mew_md=ps.MEW_MARKER * 1.3,
                   ms_star=ps.MS_MARKER * 2.2, n_ref=8)
    ax.set_ylabel(r"$T$", fontsize=ps.FS_LABEL)
    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ps.open_axes(ax)
    ax.set_yticks([1.1, 1.2, 1.3, 1.4])
    ax.set_position(ps.axes_rect(W / 72.0, H / 72.0,
                                 **dict(m, right=(W - BIG_X1_PT) / 72.0)))

    def blank():
        return Line2D([], [], ls="none", marker="none")

    hs = [blank(),
          Line2D([], [], color=PD_BINODAL, lw=ps.LW_CURVE * 1.2),
          Line2D([], [], ls="none", marker="*", color=PD_BINODAL,
                 ms=ps.MS_MARKER * 1.8, mec="none"),
          blank(),
          Line2D([], [], ls="none", marker="o", mfc="white", mec=PD_STAR_REF,
                 ms=ps.MS_MARKER * 0.8, mew=ps.MEW_MARKER * 1.3),
          Line2D([], [], ls="none", marker="*", color=PD_STAR_REF,
                 ms=ps.MS_MARKER * 1.8, mec="none")]
    labels = ["AIPF", "binodal", rf"$T_\mathrm{{c}} = {Tc:.3f}$",
              "MD", "binodal", rf"$T_\mathrm{{c}} = {ps.T_C_MD:.3f}$"]
    leg = ax.legend(hs, labels, loc="upper center", bbox_to_anchor=(0.54, 1.0),
                    ncol=2, fontsize=ps.FS_LEGEND_SM, handlelength=1.5,
                    handletextpad=0.4, labelspacing=0.16, columnspacing=1.0,
                    borderaxespad=0.0, borderpad=0.35, alignment="left",
                    frameon=True, framealpha=1.0, edgecolor=ps.C_GRID,
                    facecolor="white")
    leg.get_frame().set_linewidth(ps.LW_AXES * 0.6)
    pin_ylabel(ax, FRAME["ylabel_pt"])
    ps.clear_artist(fig, ax, leg)
    ax.set_yticks([1.2, 1.4, 1.6])
    if Y_TOP > ax.get_ylim()[1] + 1e-12:
        ax.set_ylim(ax.get_ylim()[0], Y_TOP)
        ax.set_yticks([1.2, 1.4, 1.6])

    ylim = ax.get_ylim()
    fy0, fy1 = m["bottom"] * 72.0, H - m["top"] * 72.0
    x0, x1 = SMALL_X0_PT, W - m["right"] * 72.0
    hs_pt = (fy1 - fy0 - SMALL_GAP_PT) / 2.0
    for name, y0, xl in (("Landau", fy1 - hs_pt, False), ("FH", fy0, True)):
        small_panel(fig, (x0, y0, x1 - x0, hs_pt), datas[name], ref, name,
                    ylim, xl)
    return ps.save(fig, out / "fig3a_pd_baselines_1col.pdf")


# ── b: domain growth ─────────────────────────────────────────────────────

def panel_b(out):
    g = np.load(FIGDATA / "growth_T1.20.npz")
    w, h = FRAME["width_pt"] / 72.0, B_HEIGHT_PT / 72.0
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes(ps.axes_rect(w, h, **FRAME["margins"]))
    for i in range(3):
        for fam in ("Landau", "FH"):
            k = fam.lower()
            ax.plot(g[f"{k}{i}_t"], g[f"{k}{i}_L"], "-", color=RAMPS[fam][i],
                    lw=LW_SEED, zorder=BASE_Z)
    for i in range(3):
        for fam in ("MD", "AIPF"):
            k = fam.lower()
            ax.plot(g[f"{k}{i}_t"], g[f"{k}{i}_L"], "-", color=RAMPS[fam][i],
                    lw=LW_SEED, zorder=2)

    ax.set_xscale("symlog", linthresh=10.0, linscale=0.35)
    ax.set_yscale("log")
    ax.set_xlim(0, 1e4)
    ax.set_xticks([0, 1e1, 1e2, 1e3, 1e4])
    ax.set_yticks([1e1, 2e1])
    ax.set_yticklabels(["10", "20"])
    ax.set_ylim(6.3, 21.0)
    ax.minorticks_off()
    ax.set_xlabel(r"$t\ [\tau]$", fontsize=ps.FS_LABEL)
    ax.set_ylabel(r"$L(t)\ [\sigma]$", fontsize=ps.FS_LABEL, labelpad=2)
    pin_ylabel(ax, FRAME["ylabel_pt"])
    ax.tick_params(labelsize=ps.FS_TICK, which="both")
    ax.grid(False)

    fams = ("MD", "AIPF", "Landau", "FH")
    hs = [tuple(Line2D([], [], color=c, lw=LW_SEED * 1.5) for c in RAMPS[f])
          for f in fams]
    ax.legend(hs, [DISPLAY.get(f, f) for f in fams],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
              fontsize=ps.FS_LEGEND_SM, frameon=False, loc="upper left",
              handlelength=2.4, handletextpad=0.4, labelspacing=0.1,
              borderaxespad=0.25)
    return ps.save(fig, out / "fig3_growth_baselines_1col.pdf")


def draw(out: Path) -> list[Path]:
    out = Path(out)
    ps.use()
    return [panel_a(out), panel_b(out)]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
