"""Extended Data Fig. 2: relaxation of an MD slab interface, MD against AIPF, Flory-Huggins and Landau."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import paperstyle as ps  # noqa: E402

FIGDATA = HERE / "figdata"
REFERENCE = {
    "ed_lj_slab_b.pdf": "ed_lj_slab_b.pdf",
    "ed_lj_slab_c.pdf": "ed_lj_slab_c.pdf",
    "ed_lj_slab_landau.pdf": "ed_lj_slab_landau.pdf",
    "ed_lj_slab_fh.pdf": "ed_lj_slab_fh.pdf",
    "ed_lj_slab_aipf.pdf": "ed_lj_slab_aipf.pdf",
}

# panel sizes and margins, inches
MAP_W = 2.90          # the box is 9.33 x 37.91 sigma, drawn with z horizontal
MAP_GAP = 0.05
M_LEFT, M_RIGHT = 0.50, 0.08
M_TOP_A, M_TOP, M_BOT = 0.26, 0.24, 0.40
COL_GAP = 0.16
M_RIGHT_A = 0.08
ROW1_H = 1.02
MAP_LEFT = 0.10
CBAR_PAD = 0.07
CBAR_TICKS = 0.30

T_LIN = 200.0         # linear stretch of the time axis, tau
T_END = 2.0e4
BAND_ALPHA = 0.38
C_GUIDE = "#1A1A1A"
C_T0, C_T1 = "tab:orange", "tab:blue"

MODELS = {"landau": ("Landau", "#2CA02C"), "fh": ("FH", "#7F43A4"),
          "aipf": ("AIPF", ps.C_MODEL)}
# Landau's binodal leaves [0, 1] below T ~ 1.17: those panels are greyed
GREY_T_MAX = {"landau": 1.15}
GREY_FACE, GREY_TEXT = "#ECECEC", "0.40"


def sizes():
    map_h = MAP_W * 9.3339 / 37.9071
    h = M_BOT + 2 * map_h + MAP_GAP + M_TOP
    w_b = MAP_LEFT + MAP_W + CBAR_PAD + ps.CBAR_W + CBAR_TICKS + 0.04
    return {"a": (ps.W_SM, M_BOT + ROW1_H + M_TOP_A), "b": (w_b, h),
            "c": (ps.W_SM - w_b - 0.14, h)}, map_h


# ── psi(t), seven temperatures, one model ────────────────────────────────

def psi_row(key, out):
    label, colour = MODELS[key]
    d = np.load(FIGDATA / f"psi_{key}.npz")
    Ts = [float(T) for T in d["T"]]
    sides = {T: [(d[f"T{T:.2f}_{s}_t"], d[f"T{T:.2f}_{s}_mean"],
                  d[f"T{T:.2f}_{s}_sd"]) for s in ("md", "model")] for T in Ts}
    w, h = sizes()[0]["a"]
    n_col = len(Ts)
    col_w = (w - M_LEFT - M_RIGHT_A - (n_col - 1) * COL_GAP) / n_col
    fig = plt.figure(figsize=(w, h))
    lo = min(float((mu - sd).min()) for T in Ts for _, mu, sd in sides[T])
    hi = max(float((mu + sd).max()) for T in Ts for _, mu, sd in sides[T])
    pad = 0.06 * (hi - lo)
    ylim = (lo - pad, hi + 2.2 * pad)

    axes = []
    for i, T in enumerate(Ts):
        ax = fig.add_axes([(M_LEFT + i * (col_w + COL_GAP)) / w, M_BOT / h,
                           col_w / w, ROW1_H / h])
        for (t, mu, sd), c in zip(sides[T], (ps.C_MD, colour)):
            ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=BAND_ALPHA, lw=0)
            ax.plot(t, mu, color=c, lw=ps.LW_CURVE)
        ax.set_xscale("symlog", linthresh=T_LIN, linscale=0.5)
        ax.set_xlim(0.0, T_END)
        ax.xaxis.set_major_locator(FixedLocator([0.0, 1e4, 2e4]))
        ax.set_xticklabels(["0", "", r"$2{\times}10^4$"])
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelleft=(i == 0), labelbottom=(i == 0))
        ax.set_title(rf"$T = {T:.2f}$", fontsize=ps.FS_ANNOT, pad=3.0)
        axes.append(ax)
    axes[0].set_ylabel(r"$\psi$", labelpad=1.5)
    fig.text((M_LEFT + 0.5 * col_w) / w, 0.035 / h,
             r"$t\ [\tau_{\mathrm{LJ}}]$", ha="center", va="bottom",
             fontsize=ps.FS_LABEL)

    lo, hi = axes[0].get_ylim()
    if (hi - lo) / 0.05 > 6:
        for ax in axes:
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
    for T, ax in zip(Ts, axes):
        if T <= GREY_T_MAX.get(key, -np.inf) + 1e-9:
            ax.set_facecolor(GREY_FACE)
            ax.text(0.5, 0.05, "clamp\nartefact", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=ps.FS_ANNOT * 0.8,
                    color=GREY_TEXT, linespacing=1.0)

    handles = [Line2D([0], [0], color=c, lw=ps.LW_CURVE) for c in (ps.C_MD, colour)]
    axes[0].legend(handles=handles, labels=["MD", label], loc="center left",
                   bbox_to_anchor=(0.0, 0.5), bbox_transform=axes[0].transAxes,
                   fontsize=ps.FS_LEGEND, borderpad=0.0,
                   borderaxespad=(ps.TICK_LEN + 1.0) / ps.FS_LEGEND)
    return ps.save(fig, out / f"ed_lj_slab_{key}.pdf")


# ── the MD slab at T = 1.20: field and profile ───────────────────────────

def t_end_label(t_end):
    return rf"$t={t_end / 1e4:g}\times10^{{4}}\,\tau$"


def field_panel(f, out):
    sz, map_h = sizes()
    w, h = sz["b"]
    fig = plt.figure(figsize=(w, h))
    ax_top = fig.add_axes([MAP_LEFT / w, (M_BOT + map_h + MAP_GAP) / h,
                           MAP_W / w, map_h / h])
    ax_bot = fig.add_axes([MAP_LEFT / w, M_BOT / h, MAP_W / w, map_h / h])
    z, x = f["z"], f["x"]
    for ax, key, label in ((ax_top, "field_t0", r"$t=0$"),
                           (ax_bot, "field_end", t_end_label(float(f["t_end"])))):
        ax.imshow(f[key], cmap=ps.RHO_CMAP, vmin=0.0, vmax=1.0, origin="lower",
                  extent=[z[0], z[-1], x[0], x[-1]], aspect="equal",
                  interpolation="bilinear", rasterized=True)
        ax.set_yticks([])
        ax.text(0.015, 0.90, label, transform=ax.transAxes, ha="left",
                va="top", fontsize=ps.FS_ANNOT, color="0.15")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(ps.LW_AXES)
    ax_top.set_xticks([])
    ax_bot.set_xlabel(r"$z\ [\sigma]$", labelpad=1.5)

    cax = fig.add_axes([(MAP_LEFT + MAP_W + CBAR_PAD) / w, M_BOT / h,
                        ps.CBAR_W / w, (2 * map_h + MAP_GAP) / h])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ps.RHO_CMAP,
                                            norm=plt.Normalize(0, 1)), cax=cax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cax.set_title(r"$x_{\mathrm{A}}$", fontsize=ps.FS_LABEL, pad=3.0)
    cb.outline.set_linewidth(ps.LW_AXES)
    cax.tick_params(width=ps.LW_AXES, length=ps.TICK_LEN)
    return ps.save(fig, out / "ed_lj_slab_b.pdf")


def profile_panel(f, out):
    w, h = sizes()[0]["c"]
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes(ps.axes_rect(w, h, left=M_LEFT, right=M_RIGHT,
                                   top=M_TOP, bottom=M_BOT))
    x_L, x_R = float(f["x_L"]), float(f["x_R"])
    z = f["z"]
    ax.plot(z, f["profile_t0"], color=C_T0, lw=ps.LW_CURVE, label=r"$t=0$")
    ax.plot(z, f["profile_end"], color=C_T1, lw=ps.LW_CURVE,
            label=t_end_label(float(f["t_end"])))
    for y in (x_L, x_R):
        ax.axhline(y, color=C_GUIDE, lw=ps.LW_AXES * 1.2, ls=(0, (1.4, 1.4)),
                   zorder=0)
    ax.axhline(0.5, color="0.6", lw=ps.LW_AXES, ls=(0, (4, 2)), zorder=0)
    z_arr = z[-1] * 0.86
    ax.annotate("", xy=(z_arr, x_R), xytext=(z_arr, 0.5),
                arrowprops=dict(arrowstyle="<->", lw=ps.LW_AXES * 1.2,
                                color=C_GUIDE, shrinkA=0, shrinkB=0))
    ax.text(z_arr - 0.9, 0.5 * (0.5 + x_R), r"$\psi$", ha="right",
            va="center", fontsize=ps.FS_LABEL, color=C_GUIDE)
    ax.set_xlim(z[0], z[-1])
    ax.set_ylim(-0.06, 1.10)
    ax.set_xlabel(r"$z\ [\sigma]$", labelpad=1.5)
    ax.set_ylabel(r"$x_{\mathrm{A}}$", labelpad=1.5)
    ax.set_yticks([x_L, 0.5, x_R])
    ax.set_yticklabels([r"$\rho_L$", "0.5", r"$\rho_R$"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.08),
              fontsize=ps.FS_LEGEND, borderaxespad=0.0, handlelength=1.4,
              handletextpad=0.4, labelspacing=0.25)
    return ps.save(fig, out / "ed_lj_slab_c.pdf")


def draw(out: Path) -> list[Path]:
    out = Path(out)
    ps.use()
    f = np.load(FIGDATA / "slab_md_T1.20.npz")
    return [field_panel(f, out), profile_panel(f, out),
            *(psi_row(k, out) for k in ("landau", "fh", "aipf"))]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
