"""Painted 3-D boxes: each face of a box coloured by the field on that face.

Painting a face draws one polygon per pixel of it, which takes minutes for a row of
boxes. A figure therefore ships its painted surfaces pre-rendered: ``save_layer``
writes the pixels the painted surfaces of one 3-D axes cover, at the figure's
output resolution, as an RGBA PNG, and ``add_layer`` draws that PNG back in their
place. The surfaces are opaque and drawn without antialiasing, so the composite is
the painted figure pixel for pixel; edges, titles and labels are still drawn by
matplotlib.

Besides its position on the canvas and its place in the axes' drawing order, the
PNG stores two keys: a digest of the arrays it was painted from, which
``add_layer`` checks against the figure's data, and the layout of its axes
(canvas size, axes position, view angles, limits and the projection, which holds
the zoom and the box aspect), which the layer checks whenever it is drawn at its
own resolution. A layer that no longer fits either refuses to draw and names the
command that paints it again.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import RendererAgg
from PIL import Image, PngImagePlugin
from scipy.ndimage import map_coordinates


def draw_face(ax, face, box, axis, level, u_lim, v_lim, cmap, norm, res=120,
              perm=(2, 0, 1), zorder=1):
    """Paint one axis-aligned face of `box` with `face`, the field on it
    (shape: the two in-plane grid sizes, in axis order), bilinearly
    interpolated onto a res x res grid."""
    axes_lo = [b[0] for b in box]
    axes_hi = [b[1] for b in box]
    a0, a1 = [a for a in (0, 1, 2) if a != axis]
    u = np.linspace(u_lim[0], u_lim[1], res)
    v = np.linspace(v_lim[0], v_lim[1], res)
    U, V = np.meshgrid(u, v)
    coords = [None, None, None]
    coords[axis] = np.full_like(U, level)
    coords[a0] = U
    coords[a1] = V
    n0, n1 = face.shape
    fu = (u - axes_lo[a0]) / (axes_hi[a0] - axes_lo[a0]) * (n0 - 1)
    fv = (v - axes_lo[a1]) / (axes_hi[a1] - axes_lo[a1]) * (n1 - 1)
    FU, FV = np.meshgrid(fu, fv)
    vals = map_coordinates(face, [FU, FV], order=1, mode="nearest")
    ax.plot_surface(coords[perm[0]], coords[perm[1]], coords[perm[2]],
                    facecolors=cmap(norm(vals)), rstride=1, cstride=1,
                    shade=False, antialiased=False, linewidth=0,
                    edgecolor="none", zorder=zorder)


def draw_box_edges(ax, box, perm, lw, color="0.25"):
    """The twelve edges of `box` as a thin wireframe."""
    corners = [(x, y, z) for x in box[0] for y in box[1] for z in box[2]]
    for i, c0 in enumerate(corners):
        for c1 in corners[i + 1:]:
            if sum(a != b for a, b in zip(c0, c1)) == 1:
                p0 = [c0[perm[0]], c0[perm[1]], c0[perm[2]]]
                p1 = [c1[perm[0]], c1[perm[1]], c1[perm[2]]]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        color=color, lw=lw, zorder=10)


def faces_of(field3d):
    """The six boundary faces of a 3-D field, in the order paint_faces draws
    them: (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)."""
    f = field3d
    return (f[0], f[-1], f[:, 0], f[:, -1], f[:, :, 0], f[:, :, -1])


def paint_faces(ax, faces, box, cmap, norm, res, perm):
    """Paint the six faces (see faces_of) of `box`."""
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = box
    spec = [(0, xlo, (ylo, yhi), (zlo, zhi)), (0, xhi, (ylo, yhi), (zlo, zhi)),
            (1, ylo, (xlo, xhi), (zlo, zhi)), (1, yhi, (xlo, xhi), (zlo, zhi)),
            (2, zlo, (xlo, xhi), (ylo, yhi)), (2, zhi, (xlo, xhi), (ylo, yhi))]
    for face, (axis, level, ulim, vlim) in zip(faces, spec):
        draw_face(ax, face, box, axis, level, ulim, vlim, cmap, norm,
                  res=res, perm=perm)


# ── pre-rendered painted surfaces ────────────────────────────────────────

RENDER_3D = "AIPF_FIGURES_RENDER_3D"
_YES = ("1", "true", "yes")
_FIGURES = Path(__file__).resolve().parents[1]


def paint_requested(environ=os.environ) -> bool:
    """Whether AIPF_FIGURES_RENDER_3D asks for the boxes to be painted from the fields
    ("1", "true" or "yes", in any case)."""
    return environ.get(RENDER_3D, "").strip().lower() in _YES


def source_key(*arrays) -> str:
    """sha256 of the arrays a layer is painted from (dtype, shape and values, in order)."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def layout_key(ax) -> dict:
    """Everything that places the painted faces of ``ax`` on the canvas, at the figure's dpi."""
    fig = ax.figure
    return {"dpi": float(fig.dpi),
            "canvas_px": [float(v) for v in fig.bbox.size],
            "position": [float(v) for v in ax.get_position().bounds],
            "view": [float(ax.elev), float(ax.azim), float(ax.roll)],
            "limits": [float(v) for lim in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
                       for v in lim],
            "projection": np.asarray(ax.get_proj(), float).ravel().tolist()}


def _changed(stored: dict, live: dict) -> list[str]:
    return [k for k in stored
            if k not in live or np.shape(stored[k]) != np.shape(live[k])
            or not np.allclose(stored[k], live[k], rtol=1e-9, atol=1e-9)]


def _shown(path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(_FIGURES.parent))
    except ValueError:
        return str(path)


class StaleLayer(RuntimeError):
    """A pre-rendered layer that does not belong to the figure being drawn."""


class Layer(Artist):
    """An RGBA raster at a fixed position of the figure canvas, drawn in its axes'
    drawing order. Drawn at its own dpi it is exact, and it first checks that its axes
    still have the layout it was painted in; at another dpi it is resampled, which is
    approximate and not checked."""

    def __init__(self, rgba, x0, y0, dpi, zorder, layout, path, rerun):
        super().__init__()
        self._rgba = np.array(rgba, dtype=np.uint8)   # writable, as the renderer wants
        self._x0, self._y0, self._dpi = x0, y0, dpi     # bottom-left corner, pixels
        self._layout, self._path, self._rerun = layout, path, rerun
        self.set_zorder(zorder)

    def draw(self, renderer):
        if not self.get_visible():
            return
        mag = renderer.get_image_magnification()
        pos = self.figure.dpi / self._dpi                   # canvas units per layer pixel
        size = pos * mag                                    # drawn pixels per layer pixel
        im = self._rgba
        if size == 1.0:
            changed = _changed(self._layout, layout_key(self.axes))
            if changed:
                raise StaleLayer(
                    f"{_shown(self._path)} was painted for another layout of this figure "
                    f"({', '.join(changed)} changed). Paint the boxes again, from the "
                    f"repository root, with\n  {self._rerun()}")
        else:
            h, w = im.shape[:2]
            im = np.array(Image.fromarray(im).resize(
                (max(1, round(w * size)), max(1, round(h * size))), Image.LANCZOS))
        gc = renderer.new_gc()
        # the renderer takes the bottom row first
        renderer.draw_image(gc, self._x0 * pos, self._y0 * pos, np.ascontiguousarray(im[::-1]))
        gc.restore()


@contextlib.contextmanager
def staged_layers(figdata):
    """Collect the layers one painting run writes (``save_layer(..., stage=...)``) under
    temporary names and move them into place together when the run succeeds, so a failed
    run leaves the stored set as it was. Temporary files that a killed run left in
    ``figdata`` are removed first."""
    for leftover in Path(figdata).glob(".painted_*.png.part"):
        leftover.unlink()
    stage: dict[Path, Path] = {}
    try:
        yield stage
    except BaseException:
        for tmp in stage.values():
            tmp.unlink(missing_ok=True)
        raise
    for path, tmp in stage.items():
        os.replace(tmp, path)


def save_layer(fig, ax, path, source, rerun, stage):
    """Draw the painted surfaces (the collections) of ``ax`` alone at the figure's dpi,
    write the pixels they cover as an RGBA PNG for ``path`` (staged in ``stage``, see
    ``staged_layers``), and replace the surfaces by that layer.

    ``source`` is the ``source_key`` of the arrays the surfaces were painted from and
    ``rerun()`` the command that paints them again. The figure must have been drawn at
    that dpi just before, so that the surfaces hold their projection and depth order."""
    dpi = fig.dpi
    renderer = RendererAgg(*fig.bbox.size, dpi)
    surfaces = sorted(ax.collections, key=lambda c: c.get_zorder())
    for coll in surfaces:
        coll.draw(renderer)
    rgba = np.asarray(renderer.buffer_rgba())
    h = rgba.shape[0]
    rows, cols = np.nonzero(rgba[..., 3])
    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1
    info = PngImagePlugin.PngInfo()
    info.add_text("x0", str(int(left)))
    info.add_text("y0", str(int(h - bottom)))
    info.add_text("zorder", repr(float(surfaces[-1].get_zorder())))
    info.add_text("layout", json.dumps(layout_key(ax)))
    info.add_text("source", source)
    path = Path(path)
    tmp = path.with_name(f".{path.name}.part")
    Image.fromarray(rgba[top:bottom, left:right].copy()).save(
        tmp, format="PNG", pnginfo=info, dpi=(round(dpi), round(dpi)), optimize=True)
    stage[path] = tmp
    for coll in surfaces:
        coll.remove()
    return add_layer(ax, path, source, rerun, read_from=tmp)


def add_layer(ax, path, source, rerun, read_from=None):
    """Draw the pre-rendered surfaces in ``path`` (see ``save_layer``) into ``ax``, after
    checking that they were painted from the arrays whose ``source_key`` is ``source``."""
    im = Image.open(read_from or path)
    im.load()
    if im.text.get("source") != source:
        raise StaleLayer(
            f"{_shown(path)} was painted from other data than the figure's figdata"
            + ("" if "source" in im.text else " (it names none)")
            + f". Paint the boxes again, from the repository root, with\n  {rerun()}")
    dpi = round(float(im.info["dpi"][0]))
    layer = Layer(np.asarray(im.convert("RGBA")), int(im.text["x0"]), int(im.text["y0"]),
                  dpi, float(im.text["zorder"]), json.loads(im.text["layout"]), path, rerun)
    ax.add_artist(layer)
    layer.set_clip_path(None)
    return layer
