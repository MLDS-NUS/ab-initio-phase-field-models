"""Extended Data Fig. 4: cost of AIPF against molecular dynamics with the ML potential, GPU hours per ns and peak GPU memory against system size."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import paperstyle as ps  # noqa: E402
import style_hhe as hs  # noqa: E402

FIGDATA = HERE / "figdata"
REFERENCE = {"ed_hhe_scaling.pdf": "ed_hhe_scaling.pdf"}

C_THIS = "#1F3A5F"          # AIPF, navy
C_REF = "#B5471F"           # MD with the ML potential, rust
C_GUIDE = "0.55"
GPU_GB = 40.0               # A100-SXM4-40GB, the card both methods ran on
LABELS = {"aipf": "AIPF", "md": "ML potential"}


def peak_gb(r):
    """Peak allocated GPU memory; a value under 0.1 GB is a run that reused a
    compiled kernel and skipped the compile transient, not a measurement of
    the state, and is left out."""
    v = r["peak_gpu_GB"]
    if v is not None and np.isfinite(v) and v >= 0.1:
        return float(v)
    return np.nan


def scaling(out, width_mm=89.0, height_in=1.75):
    hs.use_nature()
    data = json.loads((FIGDATA / "benchmark.json").read_text())
    w = width_mm / 25.4
    h = height_in
    fig = plt.figure(figsize=(w, h))
    left, gap, right, bottom, top = 0.50, 0.74, 0.06, 0.36, hs.M_TOP
    pw = (w - left - gap - right) / 2
    ph = h - bottom - top
    ax_a = fig.add_axes([left / w, bottom / h, pw / w, ph / h])
    ax_b = fig.add_axes([(left + pw + gap) / w, bottom / h, pw / w, ph / h])

    style = {
        "aipf": dict(color=C_THIS, marker="o", mfc=C_THIS, mec=C_THIS,
                     ms=hs.MS_MARKER, mew=hs.MEW_MARKER, lw=hs.LW_CURVE),
        "md": dict(color=C_REF, marker="o", mfc="white", mec=C_REF,
                   ms=hs.MS_MARKER, mew=hs.MEW_MARKER, lw=hs.LW_CURVE),
    }
    xs_all = []
    for key in ("md", "aipf"):
        recs = data[key]
        x = np.array([r["atoms"] for r in recs], float)
        y = np.array([r["hours_per_ns"] for r in recs], float)
        g = np.array([peak_gb(r) for r in recs], float)
        xs_all.append(x)
        ax_a.plot(x, y, **style[key], label=LABELS[key], zorder=3)
        ok = np.isfinite(g)
        ax_b.plot(x[ok], g[ok], **style[key], zorder=3)
    x_lo = min(x.min() for x in xs_all) / 1.8
    x_hi = max(x.max() for x in xs_all) * 1.8

    # slope-1 guide over the upper decade, anchored on the largest AIPF run
    xm = np.array([r["atoms"] for r in data["aipf"]], float)
    ym = np.array([r["hours_per_ns"] for r in data["aipf"]], float)
    xg = np.array([xm[-1] / 30.0, xm[-1] * 1.6])
    yg = ym[-1] * (xg / xm[-1]) * 0.30
    ax_a.plot(xg, yg, ls=":", lw=hs.LW_AXES, color=C_GUIDE, zorder=1)
    xl = np.sqrt(xg[0] * xg[1]) * 2.5
    ax_a.text(xl, ym[-1] * (xl / xm[-1]) * 0.30 * 0.22, r"$\propto N$",
              ha="center", va="top", fontsize=hs.FS_TICK, color=C_GUIDE)

    ax_b.axhline(GPU_GB, ls="--", lw=hs.LW_AXES, color=C_GUIDE, zorder=1)
    ax_b.text(x_lo * 1.4, GPU_GB * 1.18, "A100, 40 GB", fontsize=hs.FS_TICK,
              color=C_GUIDE, ha="left", va="bottom")

    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("number of atoms")
        ps.open_axes(ax)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10.0, numticks=8))
            axis.set_minor_locator(NullLocator())
    ax_a.set_ylabel("GPU hours per ns")
    ax_a.set_ylim(top=max(r["hours_per_ns"]
                          for r in data["md"] + data["aipf"]) * 12.0)
    ax_b.set_ylabel("peak GPU memory [GB]")
    ax_b.set_ylim(top=GPU_GB * 3.0)

    handles = [Line2D([], [], **style["aipf"]), Line2D([], [], **style["md"])]
    ax_a.legend(handles, [LABELS["aipf"], LABELS["md"]], loc="upper right",
                frameon=False, fontsize=hs.FS_LEGEND_SM, handlelength=1.6,
                borderaxespad=0.2, labelspacing=0.3)
    return ps.save(fig, out / "ed_hhe_scaling.pdf")


def draw(out: Path) -> list[Path]:
    with plt.rc_context():
        matplotlib.rcdefaults()
        return [scaling(Path(out))]


if __name__ == "__main__":
    print(*draw(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), sep="\n")
