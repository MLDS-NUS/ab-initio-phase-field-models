"""MD against the model, drawn the way the paper draws it: the n(x) isobar row and the anchor parity row.

Each figure has a compute half (the model at the declared states, from ``ckpt``) and a draw half
(arrays in, ``Figure`` out). The draw halves reproduce the published rows pixel for pixel.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from aipf.system import System

__all__ = ["ROW_STYLE", "anchor_parity", "anchor_rows", "draw_anchor_parity",
           "draw_eos_density", "eos_density", "isobar_rows", "save"]

#: The supplement's 8/9 pt profile: matplotlib rc for every row drawn here.
ROW_STYLE: dict[str, Any] = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "mathtext.default": "it",
    "font.size": 8.0, "axes.labelsize": 9.0, "axes.titlesize": 9.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": False, "ytick.right": False,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.2, "axes.unicode_minus": True,
    "legend.frameon": False, "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5, "legend.labelspacing": 0.28,
    "legend.borderaxespad": 0.25, "savefig.bbox": None,
    "savefig.pad_inches": 0.0, "figure.dpi": 150, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": False,
}
_FS_TICK, _FS_LABEL, _FS_ANNOT, _FS_LEGEND_SM = 8.0, 9.0, 8.0, 7.0
_LW_AXES, _TICK_LEN, _CBAR_W = 0.8, 2.6, 0.085
_M_LEFT, _M_BOTTOM, _M_TOP = 0.56, 0.42, 0.34
#: The Extended Data row: width, height, gap and right band, inches.
_ROW = dict(W=6.50, H=1.95, gap=0.36, right=0.56)
_T_CLIP = (0.06, 0.88)
_PRESSURE_MARKERS = ("o", "s", "^", "D", "v", "P")


def _t_cmap():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "plasma_clip", plt.get_cmap("plasma")(np.linspace(*_T_CLIP, 256)))


def _ink_rc(ink: str) -> dict:
    return {"figure.facecolor": "white", "savefig.facecolor": "white",
            "axes.facecolor": "white", "text.color": ink,
            "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink,
            "axes.edgecolor": ink}


def _open_axes(ax) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(top=False, right=False)


def _colorbar(fig, cax, cmap, norm, ticks, label, tick_label, labelpad):
    import matplotlib.pyplot as plt
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(label, fontsize=_FS_LABEL, labelpad=labelpad)
    cb.ax.tick_params(labelsize=_FS_TICK, length=_TICK_LEN, width=_LW_AXES)
    cb.outline.set_linewidth(_LW_AXES)
    cb.set_ticks(list(ticks))
    cb.ax.yaxis.set_major_formatter(lambda v, _p: tick_label(v))
    return cb


# -- the n(x) isobar row ------------------------------------------------------------------------

def draw_eos_density(rows: Mapping[str, Any], *, composition_label: str,
                     n_label: str, pressure_unit: str, model_label: str,
                     T_label: str, T_ticks: Sequence[float],
                     T_tick_label: Callable[[float], str], legend_loc: str,
                     ink: str):
    """One panel per pressure: model isobars (lines) and MD states (dots), coloured by T.

    ``rows`` is :func:`isobar_rows`' layout: ``x``, ``T``, ``pressures``; per pressure ``P``
    (``f"{P:g}"``): ``n_model_<P>`` ``(len(T), len(x))`` and flat ``x_md_<P>``, ``T_md_<P>``, ``n_md_<P>``."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    x, temps, pressures = (np.asarray(rows[k]) for k in ("x", "T", "pressures"))
    with plt.rc_context({**ROW_STYLE, **_ink_rc(ink)}):
        cmap, norm = _t_cmap(), Normalize(temps.min(), temps.max())
        W, H, n = _ROW["W"], _ROW["H"], len(pressures)
        pw = (W - _M_LEFT - _ROW["right"] - (n - 1) * _ROW["gap"]) / n
        ph = H - _M_BOTTOM - _M_TOP
        fig = plt.figure(figsize=(W, H))
        axes = [fig.add_axes([(_M_LEFT + i * (pw + _ROW["gap"])) / W,
                              _M_BOTTOM / H, pw / W, ph / H]) for i in range(n)]
        cax = fig.add_axes([(W - _ROW["right"] + 0.12) / W, _M_BOTTOM / H,
                            _CBAR_W / W, ph / H])
        for ax, P in zip(axes, pressures):
            key = f"{float(P):g}"
            n_mod = np.asarray(rows[f"n_model_{key}"])
            for i, T in enumerate(temps):
                ax.plot(x, n_mod[i], lw=0.9, color=cmap(norm(T)), zorder=2)
            xm, Tm, nm = (np.asarray(rows[f"{q}_md_{key}"]) for q in ("x", "T", "n"))
            for T in temps:
                s = Tm == T
                ax.plot(xm[s], nm[s], ls="none", marker="o", ms=2.6,
                        mfc=cmap(norm(T)), mec="none", zorder=3)
            ax.set_title(f"{float(P):g} {pressure_unit}", fontsize=_FS_ANNOT, pad=3)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xticklabels(["0", "0.5", "1"])
            ax.set_xlabel(composition_label, labelpad=1)
            _open_axes(ax)
        axes[0].set_ylabel(n_label, labelpad=3)
        hh = [Line2D([], [], ls="none", marker="o", ms=2.6, mfc="0.4", mec="none"),
              Line2D([], [], color="0.4", lw=0.9)]
        axes[0].legend(hh, ["MD", model_label], loc=legend_loc,
                       fontsize=_FS_LEGEND_SM, frameon=False, handlelength=1.2,
                       handletextpad=0.4, labelspacing=0.2, borderaxespad=0.2)
        _colorbar(fig, cax, cmap, norm, T_ticks, T_label, T_tick_label, 2)
    return fig


def _load(system: System, ckpt: Any):
    from aipf.diagnose.run import load_model, resolve_checkpoint
    return load_model(system, resolve_checkpoint(system, ckpt))


def _composition(system: System, x_channel_one: np.ndarray) -> np.ndarray:
    """The plotted fraction, of channel ``table_keys["x_channel"]``, from the isobar's channel-1 fraction."""
    return x_channel_one if int(system.table_keys["x_channel"]) == 1 \
        else 1.0 - x_channel_one


def isobar_rows(system: System, *, pressures: Sequence[float],
                T_grid: Sequence[float], ckpt: Any, x: Sequence[float]) -> dict:
    """The model's own isobars and the measured manifold at ``pressures`` x ``T_grid``.

    ``x`` is the channel-1 fraction the isobar walks; the manifolds, pressure unit and solver are the
    system's ``defaults["diagnose"]`` declarations."""
    from aipf.diagnose.run import isobar_density, manifolds

    declared = system.defaults["diagnose"]
    model = _load(system, ckpt)
    by_pressure = manifolds(system, declared)
    xs = np.asarray(x, dtype=np.float64)
    temps = np.asarray(T_grid, dtype=np.float64)
    order = np.argsort(_composition(system, xs))
    out: dict = {"x": _composition(system, xs)[order], "T": temps,
                 "pressures": np.asarray(pressures, dtype=np.float64)}
    for P in pressures:
        key, manifold = f"{float(P):g}", by_pressure[float(P)]
        n_mod = np.full((len(temps), len(xs)), np.nan)
        for i, T in enumerate(temps):
            try:
                n_mod[i] = isobar_density(
                    model, float(P), xs, float(T), manifold=manifold,
                    pressure_unit=float(declared["pressure_unit"]),
                    poly_degree=int(declared["poly_degree"]),
                    n_species=system.n_species, **dict(declared["isobar"]))
            except ValueError:
                pass
        out[f"n_model_{key}"] = n_mod[:, order]
        x_grid, T_nodes, grid = manifold
        md_x, md_T, md_n = [], [], []
        for T in temps:
            j = np.flatnonzero(np.isclose(T_nodes, T))
            if len(j):
                keep = np.isfinite(grid[:, j[0]])
                md_x.append(_composition(system, np.asarray(x_grid)[keep]))
                md_T.append(np.full(int(keep.sum()), T))
                md_n.append(grid[keep, j[0]])
        for name, parts in (("x", md_x), ("T", md_T), ("n", md_n)):
            out[f"{name}_md_{key}"] = np.concatenate(parts) if parts else np.empty(0)
    return out


def eos_density(system: System, *, pressures: Sequence[float],
                T_grid: Sequence[float], ckpt: Any, x: Sequence[float],
                **labels: Any):
    """:func:`isobar_rows` drawn by :func:`draw_eos_density`; ``labels`` are that function's keywords."""
    return draw_eos_density(isobar_rows(system, pressures=pressures, T_grid=T_grid,
                                        ckpt=ckpt, x=x), **labels)


# -- the anchor parity row ----------------------------------------------------------------------

def draw_anchor_parity(rows: Mapping[str, Any], *, species: Sequence[str],
                       M_unit: str, pressure_unit: str, model_label: str,
                       T_label: str, T_round: float, T_tick_step: float,
                       T_tick_label: Callable[[float], str],
                       width: float, height: float):
    """Gamma and the three mobility entries, MD (x) against the model (y), coloured by T, one marker per pressure.

    ``rows`` keys: ``gamma_{data,model,T,P}``, ``M_{data,model,T,P}`` (``M_*`` ``(N, 2, 2)``)."""
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.lines import Line2D

    a, b = species
    comps = (rf"$M_\mathrm{{{a}{a}}}$", rf"$M_\mathrm{{{b}{b}}}$",
             rf"$|M_\mathrm{{{a}{b}}}|$")
    pressures = [float(p) for p in np.unique(np.concatenate(
        [np.asarray(rows["gamma_P"]), np.asarray(rows["M_P"])]))]
    marks = dict(zip(pressures, _PRESSURE_MARKERS))
    with plt.rc_context(ROW_STYLE):
        cmap = _t_cmap()
        T_all = np.concatenate([rows["gamma_T"], rows["M_T"]])
        nrm = mcolors.Normalize(np.floor(T_all.min() / T_round) * T_round,
                                np.ceil(T_all.max() / T_round) * T_round)
        Md, Mm = np.asarray(rows["M_data"]), np.asarray(rows["M_model"])
        series = [(r"$\Gamma$" + "\n ", rows["gamma_data"], rows["gamma_model"],
                   rows["gamma_T"], rows["gamma_P"])]
        for lab, (i, j) in zip(comps, ((0, 0), (1, 1), (0, 1))):
            series.append((lab + "\n" + M_unit, np.abs(Md[:, i, j]),
                           np.abs(Mm[:, i, j]), rows["M_T"], rows["M_P"]))
        gap, right, top = 0.40, 0.60, 0.50
        ph = height - _M_BOTTOM - top
        pw = (width - _M_LEFT - right - 3 * gap) / 4.0
        fig = plt.figure(figsize=(width, height))
        axes = []
        for k, (lab, xd, yd, T, P) in enumerate(series):
            xd, yd, T, P = (np.asarray(v) for v in (xd, yd, T, P))
            ax = fig.add_axes([(_M_LEFT + k * (pw + gap)) / width,
                               _M_BOTTOM / height, pw / width, ph / height])
            axes.append(ax)
            hi = 1.06 * max(xd.max(), yd.max())
            ax.plot([0.0, hi], [0.0, hi], ls=(0, (4, 2.5)), lw=_LW_AXES,
                    color="0.35", zorder=1)
            for Pv in np.unique(P):
                s = P == Pv
                ax.scatter(xd[s], yd[s], c=T[s], cmap=cmap, norm=nrm,
                           marker=marks[float(Pv)], s=2.4 ** 2 * 2.2,
                           linewidths=0.0, edgecolors="none", alpha=0.75,
                           zorder=3)
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(lab, fontsize=_FS_ANNOT, pad=2.0, linespacing=1.15)
            ax.set_xlabel("MD")
            ax.grid(False)
            ax.tick_params(labelsize=_FS_TICK)
        axes[0].set_ylabel(model_label)
        hp = [Line2D([], [], ls="none", marker=marks[p], ms=2.8, mfc="0.35",
                     mec="none") for p in pressures]
        axes[0].legend(hp, [f"{p:g} {pressure_unit}" for p in pressures],
                       loc="lower right", fontsize=6.0, frameon=False,
                       handletextpad=0.25, borderaxespad=0.25, labelspacing=0.14)
        cax = fig.add_axes([(width - right + 0.10) / width, _M_BOTTOM / height,
                            _CBAR_W / width, ph / height])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cmap), cax=cax)
        cb.set_label(T_label, fontsize=_FS_LABEL)
        ticks = np.arange(nrm.vmin, nrm.vmax + 1, T_tick_step)
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels([T_tick_label(t) for t in ticks])
        cb.ax.tick_params(labelsize=_FS_TICK, length=_TICK_LEN, width=_LW_AXES)
        cb.outline.set_linewidth(_LW_AXES)
    return fig


def _double(tables):
    """The tables with every tensor field in float64, the dtype :func:`_load` gives the model."""
    return dataclasses.replace(tables, **{
        f.name: getattr(tables, f.name).double()
        for f in dataclasses.fields(tables)
        if isinstance(getattr(tables, f.name), torch.Tensor)})


def anchor_rows(system: System, *, ckpt: Any) -> dict:
    """Gamma and M, measured and modelled, at every row the system's declared anchor tables keep.

    One pressure at a time through :class:`aipf.train.anchors.AnchorTables`, so each row keeps its
    pressure; tables and filters are ``defaults["training"]["tables"]``."""
    from aipf.train.anchors import AnchorTables

    declared = dict(system.defaults["training"]["tables"])
    if declared.get("key") != "pressure":
        raise NotImplementedError(
            f"anchor_rows reads pressure-keyed tables only; system {system.name!r} declares "
            f"tables key={declared.get('key')!r}")
    kB = float(system.constants["kB"])
    model = _load(system, ckpt)
    parts: dict[str, list] = {k: [] for k in (
        "gamma_data", "gamma_model", "gamma_T", "gamma_P",
        "M_data", "M_model", "M_T", "M_P")}
    for P in declared["m_table"]:
        one = {**declared, **{k: {P: declared[k][P]}
                              for k in ("m_table", "s_table", "eos_csvs")}}
        tables = _double(AnchorTables.load(system, **one))
        batch = tables.batch(model, kB)
        bulk, mob = batch["anchor_bulk"], batch["anchor_M"]
        with torch.no_grad():
            z, H0 = bulk["zvec"], bulk["H0"].detach()
            q = (z * torch.linalg.solve(H0, z.unsqueeze(-1)).squeeze(-1)).sum(-1)
            x1 = z[:, 0]
            mix = (x1 * (1.0 - x1)).numpy()
            parts["gamma_data"].append(mix / bulk["target"].numpy())
            parts["gamma_model"].append(
                mix / (bulk["kBT"] * q / bulk["rho_tot"]).numpy())
            parts["M_model"].append(mob["M_pred"].detach().numpy())
        parts["gamma_T"].append(tables.bulk_T.numpy())
        parts["gamma_P"].append(np.full(len(mix), float(P)))
        parts["M_data"].append(tables.mobility_target.numpy())
        parts["M_T"].append(tables.mobility_T.numpy())
        parts["M_P"].append(np.full(len(tables.mobility_T), float(P)))
    return {k: np.concatenate(v) for k, v in parts.items()}


def anchor_parity(system: System, *, ckpt: Any, **labels: Any):
    """:func:`anchor_rows` drawn by :func:`draw_anchor_parity`; ``labels`` are that function's keywords."""
    return draw_anchor_parity(anchor_rows(system, ckpt=ckpt), **labels)


def save(fig, path, *, dpi: float) -> None:
    """Write ``fig`` at exactly its size (no tight box, no padding), under :data:`ROW_STYLE`."""
    import matplotlib.pyplot as plt
    with plt.rc_context(ROW_STYLE):
        fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0.0)
