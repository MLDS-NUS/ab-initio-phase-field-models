"""Extended Data Fig. 6: AIPF bulk free energy (a) and its curvature (b), binary Lennard-Jones, T = 1.0 to 1.5."""
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
REFERENCE = {
    "fig2a_free_energy.pdf": "fig2a_free_energy.pdf",
    "fig2b_curvature.pdf": "fig2b_curvature.pdf",
}
# The arrays are a float32 evaluation of the published model on another
# device than the published files', so the curves sit a fraction of a point
# away from the published ones (largest in b, where the curvature is a
# numerical derivative of the chemical potential).
BOUND_PX = {"fig2a_free_energy.pdf": 96, "fig2b_curvature.pdf": 6592}
CAUSE = {name: "float32 model evaluation on a different device from the "
               "published file's; the curves move by a fraction of a point"
         for name in REFERENCE}

C_UNSTABLE, C_UNSTABLE_TXT = "#F4D3B8", "#BE6A2A"
D2_YLIM = (-4.0, 4.0)
MS_BINODAL, MS_SPINODAL = ps.MS_MARKER, ps.MS_MARKER * 0.9


def _family():
    d = np.load(FIGDATA / "free_energy.npz")
    norm = ps.norm()
    return d, [ps.T_CMAP(norm(T)) for T in d["T"]], norm


def _finish(ax):
    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ps.open_axes(ax)


def _markers(ax, xs, ys, marker, ms, col):
    for r, y in zip(xs, ys):
        if np.isfinite(r):
            ax.plot(float(r), float(y), marker, color=col, ms=ms, mec="white",
                    mew=ps.MEW_MARKER, zorder=5)


def panel_a(out):
    d, colors, _ = _family()
    w, h = ps.PANEL["a"]
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes(ps.axes_rect(w, h))
    x = d["x_A"]
    for k, col in enumerate(colors):
        ax.plot(x, d["f"][k], color=col, lw=ps.LW_CURVE, zorder=3)
        _markers(ax, d["binodal_x"][k], d["binodal_f"][k], "o", MS_BINODAL, col)
        _markers(ax, d["spinodal_x"][k], d["spinodal_f"][k], "s", MS_SPINODAL, col)
    ax.axhline(0, color="0.6", lw=0.6, zorder=1)
    ax.set_ylabel(r"bulk free energy  $f$", fontsize=ps.FS_LABEL)
    ax.tick_params(labelsize=ps.FS_TICK)
    ax.set_xlabel(r"composition  $x_\mathrm{A}$", fontsize=ps.FS_LABEL)
    _finish(ax)
    handles = [Line2D([], [], ls="none", marker="o", color="0.35",
                      ms=MS_BINODAL, mec="white", mew=ps.MEW_MARKER),
               Line2D([], [], ls="none", marker="s", color="0.35",
                      ms=MS_SPINODAL, mec="white", mew=ps.MEW_MARKER)]
    leg = ax.legend(handles, ["binodal", "spinodal"], loc="upper center",
                    ncol=1, handletextpad=0.5, labelspacing=0.3,
                    borderaxespad=0.4)
    ps.clear_artist(fig, ax, leg)
    return ps.save(fig, out / "fig2a_free_energy.pdf")


def panel_b(out):
    d, colors, norm = _family()
    w, h = ps.PANEL["b"]
    fig = plt.figure(figsize=(w, h))
    rect = ps.axes_rect(w, h, right=ps.M_RIGHT_CBAR)
    ax = fig.add_axes(rect)
    x = d["x_A"]
    ax.axhspan(D2_YLIM[0], 0, color=C_UNSTABLE, alpha=0.9, lw=0, zorder=0)
    for k, col in enumerate(colors):
        ax.plot(x, d["curvature"][k], color=col, lw=ps.LW_CURVE, zorder=3)
        _markers(ax, d["spinodal_x"][k], np.zeros(2), "s", MS_SPINODAL, col)
    ax.axhline(0, color="k", lw=1.0, zorder=2)
    ax.set_ylim(*D2_YLIM)
    ax.set_xlabel(r"composition  $x_\mathrm{A}$", fontsize=ps.FS_LABEL)
    ax.set_ylabel(r"$\partial^2 f/\partial x_\mathrm{A}^2$", fontsize=ps.FS_LABEL)
    ax.tick_params(labelsize=ps.FS_TICK)
    ax.text(0.975, D2_YLIM[0] + 0.06 * (D2_YLIM[1] - D2_YLIM[0]), "unstable",
            ha="right", va="bottom", fontsize=ps.FS_LEGEND, color=C_UNSTABLE_TXT)
    _finish(ax)
    ax.set_yticks([-2, 0, 2])

    cax = fig.add_axes(ps.cbar_rect(w, h, rect))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=ps.T_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$T$", fontsize=ps.FS_LABEL, labelpad=2)
    cb.ax.tick_params(labelsize=ps.FS_TICK, length=ps.TICK_LEN, width=ps.LW_AXES)
    cb.outline.set_linewidth(ps.LW_AXES)
    cb.set_ticks(np.arange(ps.T_MIN, ps.T_MAX + 1e-9, 0.1))
    return ps.save(fig, out / "fig2b_curvature.pdf")


def draw(out: Path) -> list[Path]:
    out = Path(out)
    ps.use()
    return [panel_a(out), panel_b(out)]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
