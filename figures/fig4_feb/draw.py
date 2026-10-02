"""Fig. 4: iron-boron thermodynamic factor maps, spinodals, isotherms and the MD parity."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (BoundaryNorm, LinearSegmentedColormap,
                               ListedColormap, Normalize)
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import paperstyle as ps  # noqa: E402
import style_feb  # noqa: E402

FIGDATA = HERE / "figdata"
REFERENCE = {"feb_composite.pdf": "feb_composite.pdf"}

PRESSURES = [0, 5, 10]
# pressure: single-hue luminance ramp, light = 0 GPa, dark = 10 GPa
P_COLOR = {0: "#fdbf6f", 5: "#e6550d", 10: "#5c1a00"}
G_TEMPS = [1500.0, 1800.0, 2100.0]
T_COLOR = {1500.0: "#0a2a52", 1800.0: "#3182bd", 2100.0: "#a8d0e8"}
ACC = "#1F3A5F"
WASH = "#f6d7cf"
W_FIG = 7.06
ROW1_H = 1.45          # row-1 panel height (T-x panels)
ROW2_H = 1.15          # row-2 plot height
ROW_GAP = 0.52
X_LABEL = r"$x_{\mathrm{B}}$"

CLASS_LABEL = {"stable": "MD stable", "unsure": "MD unsure",
               "spinodal": "MD spinodal"}
# MD classification markers on the maps
SMALL = {"stable": dict(marker="o", ms=1.7, mfc="0.45", mec="none", mew=0),
         "unsure": dict(marker="^", ms=2.1, mfc="#d0a83e", mec="none", mew=0),
         "spinodal": dict(marker="x", ms=3.4, mfc="none", mec="#a31515",
                          mew=1.0)}
# the same classes on the parity panel, filled
PARITY = {"unsure": dict(marker="^", ms=2.8, mfc="#d0a83e", mec="none"),
          "spinodal": dict(marker="X", ms=3.6, mfc="#a31515", mec="none")}


def gamma_law():
    """Diverging Gamma scale: continuous map, its discrete levels and norms."""
    VMIN, VMAX = -3.0, 12.0
    rdbu = plt.get_cmap("RdBu")
    f0 = (0.0 - VMIN) / (VMAX - VMIN)
    N = 1024
    n_neg = int(round(N * f0))
    cols = np.vstack([rdbu(np.linspace(0.03, 0.5, n_neg)),
                      rdbu(np.linspace(0.5, 0.99, N - n_neg))])
    cont = LinearSegmentedColormap.from_list("gl", cols)
    cnorm = Normalize(VMIN, VMAX)
    levels = np.r_[np.arange(-3.0, 0.0, 0.25), np.arange(0.0, 12.01, 1.0)]
    mids = 0.5 * (levels[:-1] + levels[1:])
    disc = ListedColormap(cont(cnorm(mids)))
    # saturate past the level range so a peak above VMAX is not a white hole
    disc.set_under(cont(0.0))
    disc.set_over(cont(1.0))
    return cont, cnorm, disc, BoundaryNorm(levels, disc.N), levels


def xB_axis(ax, ticks):
    """Axis of a curve drawn in x_Fe, shown as x_B increasing to the right."""
    ax.set_xlim(1.0, 0.0)
    ticks = np.asarray(ticks, dtype=float)
    ax.set_xticks(1.0 - ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def draw(out: Path) -> list[Path]:
    with plt.rc_context():
        matplotlib.rcdefaults()
        return composite(Path(out))


def composite(out):
    ps.use()
    plt.rcParams["axes.grid"] = False
    M_B, M_T = ps.M_BOTTOM, ps.M_TOP
    H_FIG = M_B + ROW2_H + ROW_GAP + ROW1_H + M_T
    fig = plt.figure(figsize=(W_FIG, H_FIG))
    y1 = (M_B + ROW2_H + ROW_GAP) / H_FIG
    cont, cnorm, disc, bnorm, levels = gamma_law()

    d = np.load(FIGDATA / "gamma_map.npz")
    T, xB = d["T"], d["x_B"]
    cls = np.load(FIGDATA / "md_phase_class.npz")

    # ── row 1: Gamma maps at 0, 5, 10 GPa, the colour bar, the spinodals ──
    M_L1 = 0.52
    CB_STRIP = 0.82
    GAP1 = 0.10
    sub1 = (W_FIG - M_L1 - CB_STRIP - 3 * GAP1 - 0.06) / 4
    xs1 = [M_L1 + i * (sub1 + GAP1) for i in range(3)]
    bar_x1 = xs1[2] + sub1 + 0.08
    xs1.append(bar_x1 + CB_STRIP)
    ax1 = [fig.add_axes([x / W_FIG, y1, sub1 / W_FIG, ROW1_H / H_FIG])
           for x in xs1]
    for i, P in enumerate(PRESSURES):
        ax = ax1[i]
        ax.grid(False)
        gam = d[f"Gamma_P{P}"].copy()
        for r in gam:                 # bridge a row's isolated holes for the fill
            m = np.isfinite(r)
            if 10 < m.sum() < len(r):
                r[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), r[m])
        ax.contourf(xB, T, gam, levels=levels, cmap=disc, norm=bnorm,
                    extend="both")
        ax.contour(xB, T, gam, levels=[0.0], colors="k",
                   linewidths=ps.LW_AXES, linestyles="--", zorder=3)
        kind_P = cls[f"class_P{P}"]
        for kind, st in SMALL.items():
            sel = kind_P == kind
            if sel.any():
                ax.plot(cls[f"x_B_P{P}"][sel], cls[f"T_P{P}"][sel],
                        ls="none", zorder=5, **st)
        ax.set_title(f"{P} GPa", fontsize=ps.FS_ANNOT, pad=3)
        if i == 0:
            ax.legend(handles=[
                Line2D([], [], color="k", lw=ps.LW_AXES, ls="--",
                       label="spinodal")] + [
                Line2D([], [], ls="none", label=CLASS_LABEL[k], **SMALL[k])
                for k in ("stable", "unsure", "spinodal")],
                frameon=True, framealpha=0.85, facecolor="white",
                edgecolor="none", fontsize=ps.FS_LEGEND_SM,
                loc="upper left", handlelength=1.2, borderaxespad=0.25,
                labelspacing=0.25)
    axO = ax1[3]
    axO.grid(False)
    for P in PRESSURES:
        axO.contour(xB, T, d[f"lambda_min_P{P}"], levels=[0.0],
                    colors=[P_COLOR[P]], linewidths=ps.LW_CURVE)
        axO.plot([], [], color=P_COLOR[P], lw=ps.LW_CURVE, label=f"{P} GPa")
    axO.plot([0.8], [1500.0], ls="none", marker="*", ms=6.0, mfc=ACC,
             mec="none", zorder=6, label="FeB$_4$, 1500 K")
    axO.legend(frameon=False, fontsize=ps.FS_LEGEND_SM, loc="upper left",
               handlelength=1.1, borderaxespad=0.25, labelspacing=0.3)
    for i, ax in enumerate(ax1):
        if i == 3:      # zoomed to the dome: the 0 and 5 GPa curves nearly coincide
            ax.set_xlim(0.55, 1.0)
            ax.set_ylim(1150, 2150)
            ax.set_xticks([0.6, 0.8, 1.0])
            ax.set_yticks(np.arange(1200, 2101, 200))
            ax.tick_params(labelleft=True)
            ax.set_xlabel(X_LABEL, fontsize=ps.FS_LABEL, labelpad=1)
            continue
        ax.set_xlim(0.0, 1.01)
        ax.set_ylim(1170, 2630)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["0" if i == 0 else "", "0.5", "1.0"])
        ax.set_yticks(np.arange(1200, 2601, 200))
        if i == 0:
            ax.set_xlabel(X_LABEL, fontsize=ps.FS_LABEL, labelpad=1)
            ax.set_ylabel(r"$T$ [K]", fontsize=ps.FS_LABEL, labelpad=2)
        else:
            ax.tick_params(labelleft=False)
    cax = fig.add_axes([bar_x1 / W_FIG, y1, 0.085 / W_FIG, ROW1_H / H_FIG])
    cax.grid(False)
    cb = fig.colorbar(ScalarMappable(norm=cnorm, cmap=cont), cax=cax,
                      ticks=[-3, 0, 3, 6, 9, 12], extend="both")
    cax.set_title(r"$\Gamma$", fontsize=ps.FS_LABEL, pad=5)
    cb.ax.tick_params(labelsize=ps.FS_TICK, length=ps.TICK_LEN,
                      width=ps.LW_AXES)
    cb.outline.set_linewidth(ps.LW_AXES)

    # ── row 2: Gamma isotherms under each map, parity under the spinodals ──
    y2 = M_B / H_FIG
    d4 = np.load(FIGDATA / "gamma_isotherms.npz")
    axD = [fig.add_axes([xs1[i] / W_FIG, y2, sub1 / W_FIG, ROW2_H / H_FIG])
           for i in range(3)]
    tri_dy = -(0.5 * 4.0 / 72.0) / ROW2_H
    for i, P in enumerate(PRESSURES):
        ax = axD[i]
        ax.grid(False)
        for Tk in G_TEMPS:
            c = T_COLOR[Tk]
            xx = d4[f"x_Fe_P{P}_T{Tk:.0f}"]
            gam = d4[f"Gamma_P{P}_T{Tk:.0f}"]
            ax.plot(xx, gam, "-", color=c, lw=ps.LW_CURVE,
                    label=f"{Tk:.0f} K", zorder=3)
            ax.plot([0.2], [np.interp(0.2, xx, gam)], marker="o", ms=3.4,
                    mfc=c, mec="none", ls="none", zorder=4)
        ax.axhline(0.0, color="0.35", lw=ps.LW_AXES, ls="--", zorder=1)
        ax.axhspan(-10.0, 0.0, color=WASH, alpha=0.55, lw=0, zorder=0)
        if i == 0:
            ax.text(0.03, 0.03, "AIPF spinodal", fontsize=ps.FS_LEGEND_SM,
                    color="#8a2f1d", ha="left", va="bottom",
                    transform=ax.transAxes)
            ax.plot([0.2], [tri_dy], transform=ax.get_xaxis_transform(),
                    marker="^", ms=4.0, mfc=ACC, mec="none", ls="none",
                    clip_on=False, zorder=6)
            ax.annotate("FeB$_4$", xy=(0.2, 0.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -9.5), textcoords="offset points",
                        ha="center", va="top", fontsize=ps.FS_ANNOT,
                        color=ACC, annotation_clip=False)
        ax.set_title(f"{P:g} GPa", fontsize=ps.FS_ANNOT, pad=2)
        xB_axis(ax, ticks=[0, 0.5, 1.0])
        ax.set_xticklabels(["0" if i == 0 else "", "0.5", "1"])
        ps.open_axes(ax)
        if i == 0:
            ax.set_ylabel(r"$\Gamma$", labelpad=2)
            ax.set_xlabel(X_LABEL, labelpad=1)
        else:
            ax.tick_params(labelleft=False)
    g_top = max(np.nanmax(d4[f"Gamma_P{P}_T{Tk:.0f}"])
                for P in PRESSURES for Tk in G_TEMPS)
    for ax in axD:
        ax.set_ylim(-2.0, 1.06 * g_top)
    axD[0].legend(frameon=True, framealpha=0.85, facecolor="white",
                  edgecolor="none", fontsize=ps.FS_LEGEND_SM,
                  loc="upper right", handlelength=1.0, borderaxespad=0.2,
                  labelspacing=0.25)

    # the parity panel, the letters and the page carry the surface layer
    style_feb.use()
    par = np.load(FIGDATA / "gamma_parity.npz")
    GD, GM, CL = par["Gamma_MD"], par["Gamma_AIPF"], par["class_MD"]
    axE = fig.add_axes([xs1[3] / W_FIG, y2, sub1 / W_FIG, ROW2_H / H_FIG])
    axE.grid(False)
    XL, YL = (-1.0, 3.2), (-2.2, 3.2)
    axE.axhspan(YL[0], 0.0, color=WASH, alpha=0.55, lw=0, zorder=0)
    axE.axhline(0.0, color="0.35", lw=ps.LW_AXES, ls="--", zorder=1)
    axE.axvline(0.0, color="0.35", lw=ps.LW_AXES, ls="--", zorder=1)
    for kind in ("unsure", "spinodal"):
        sel = CL == kind
        axE.plot(GD[sel], GM[sel], ls="none", zorder=3,
                 label=CLASS_LABEL[kind], **PARITY[kind])
    for sp in axE.spines.values():
        sp.set_color("k")
    axE.tick_params(colors="k")
    axE.set_xlim(XL)
    axE.set_ylim(YL)
    axE.set_xlabel(r"$\Gamma_\mathrm{MD}$", fontsize=ps.FS_LABEL,
                   labelpad=1, color="k")
    axE.set_ylabel(r"$\Gamma_\mathrm{AIPF}$", fontsize=ps.FS_LABEL,
                   labelpad=1, color="k")
    axE.legend(frameon=True, framealpha=0.85, facecolor="white",
               edgecolor="none", fontsize=ps.FS_LEGEND_SM, loc="upper left",
               handlelength=1.0, borderaxespad=0.2, labelspacing=0.25)

    for lab, (xf, yf) in {
            "a": (0.01, y1 + (ROW1_H + 0.10) / H_FIG),
            "b": ((xs1[3] - 0.30) / W_FIG, y1 + (ROW1_H + 0.10) / H_FIG),
            "c": (0.01, (M_B + ROW2_H + 0.12) / H_FIG),
            "d": ((xs1[3] - 0.30) / W_FIG, (M_B + ROW2_H + 0.12) / H_FIG)}.items():
        fig.text(xf, yf, lab, fontsize=10, fontweight="bold", ha="left",
                 va="bottom")
    return [ps.save(fig, out / "feb_composite.pdf")]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
