"""aipf.md.analysis: the n(x) isobar row and the anchor parity row.

Fast tests draw synthetic rows. The slow ones hold the module to the paper: the draw halves,
fed the published npz files in figures/<fig>/figdata/, must reproduce the published rows pixel for
pixel (in the figure interpreter, matplotlib 3.10.8), and the compute halves, run on the H/He
published model, must reproduce the numbers in those npz files.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import declared_roots

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "figures"
REFERENCE = REPO / "figures" / "reference"

EOS_LABELS = dict(composition_label=r"$x_{\mathrm{b}}$", n_label="$n$",
                  pressure_unit="u", model_label="fit", T_label="$T$",
                  T_ticks=[1.0, 2.0], T_tick_label=lambda v: f"{v:g}",
                  legend_loc="upper right", ink="black")


def _synthetic_rows(pressures=(1.0, 2.0, 3.0)):
    x = np.linspace(0.0, 1.0, 11)
    T = np.array([1.0, 1.5, 2.0])
    rows = {"x": x, "T": T, "pressures": np.asarray(pressures)}
    for P in pressures:
        k = f"{P:g}"
        rows[f"n_model_{k}"] = np.stack([P + t - x for t in T])
        rows[f"x_md_{k}"] = np.tile(x[::5], len(T))
        rows[f"T_md_{k}"] = np.repeat(T, 3)
        rows[f"n_md_{k}"] = np.concatenate([P + t - x[::5] for t in T])
    return rows


def test_the_isobar_row_has_one_panel_per_pressure_and_a_colorbar():
    from aipf.md.analysis import draw_eos_density
    fig = draw_eos_density(_synthetic_rows(), **EOS_LABELS)
    panels, cbar = fig.axes[:-1], fig.axes[-1]
    assert [ax.get_title() for ax in panels] == ["1 u", "2 u", "3 u"]
    assert all(ax.get_xlabel() == EOS_LABELS["composition_label"] for ax in panels)
    assert [t.get_text() for t in panels[0].get_legend().get_texts()] == ["MD", "fit"]
    assert cbar.get_ylabel() == "$T$"
    assert len(panels[0].lines) == 3 + 3          # a line and a marker set per temperature


def test_every_label_is_the_callers():
    """Nothing about a system is in the module: leaving a label out is a TypeError."""
    from aipf.md.analysis import draw_eos_density
    labels = dict(EOS_LABELS)
    labels.pop("model_label")
    with pytest.raises(TypeError):
        draw_eos_density(_synthetic_rows(), **labels)


def test_the_parity_row_is_gamma_and_three_mobility_entries():
    from aipf.md.analysis import draw_anchor_parity
    rng = np.random.default_rng(0)
    M = rng.uniform(0.1, 1.0, (8, 2, 2))
    rows = {"gamma_data": rng.uniform(0.1, 1, 8), "gamma_model": rng.uniform(0.1, 1, 8),
            "gamma_T": np.linspace(1000, 3000, 8), "gamma_P": np.repeat([1.0, 2.0], 4),
            "M_data": M, "M_model": M * 1.1, "M_T": np.linspace(1000, 3000, 8),
            "M_P": np.repeat([1.0, 2.0], 4)}
    fig = draw_anchor_parity(rows, species=("a", "b"), M_unit="[u]", pressure_unit="u",
                             model_label="fit", T_label="$T$", T_round=1000.0,
                             T_tick_step=1000.0, T_tick_label=lambda t: f"{t:g}",
                             width=6.5, height=1.96)
    titles = [ax.get_title() for ax in fig.axes[:4]]
    assert titles[0].startswith(r"$\Gamma$")
    assert [t.split("\n")[0] for t in titles[1:]] == [
        r"$M_\mathrm{aa}$", r"$M_\mathrm{bb}$", r"$|M_\mathrm{ab}|$"]
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["1 u", "2 u"]


def test_the_plotted_fraction_follows_the_declared_channel():
    from aipf.md.analysis import _composition
    from aipf.system import load
    x = np.array([0.1, 0.7])
    assert np.array_equal(_composition(load("hhe"), x), x)            # x_channel 1
    assert np.allclose(_composition(load("feb"), x), 1.0 - x)         # x_channel 0


# ── against the paper ──────────────────────────────────────────────────────

#: The three published rows, drawn by the module from the published npz files.
_PORT = r'''
import sys
import numpy as np
from aipf.md.analysis import draw_anchor_parity, draw_eos_density, save
figures, out = sys.argv[1], sys.argv[2]
d = np.load(figures + "/ed7_hhe_nx/figdata/density_isobars.npz")
temps = d["T"]
rows = {"x": d["x_He"], "T": temps, "pressures": d["pressures"]}
for P in d["pressures"]:
    k, xg = f"{P:g}", d[f"x_He_md_P{P:.0f}"]
    rows[f"n_model_{k}"] = d[f"n_model_P{P:.0f}"]
    rows[f"x_md_{k}"] = np.tile(xg, len(temps))
    rows[f"T_md_{k}"] = np.repeat(temps, len(xg))
    rows[f"n_md_{k}"] = d[f"n_md_P{P:.0f}"].ravel()
save(draw_eos_density(rows, composition_label=r"$x_{\mathrm{He}}$",
     n_label=r"$n$ [Å$^{-3}$]", pressure_unit="GPa", model_label="AIPF",
     T_label=r"$T$ [$10^{3}$ K]", T_ticks=[t for t in d["T"] if t % 2000 == 0],
     T_tick_label=lambda v: f"{v / 1000:.0f}", legend_loc="upper right",
     ink="#15181c"), out + "/ed_hhe_nx.pdf", dpi=600)
d = np.load(figures + "/sm_feb_nx/figdata/density_isobars.npz")
T = d["T"]
rows = {"x": d["x_B"], "T": T, "pressures": np.array([0.0, 5.0, 10.0])}
for P in (0, 5, 10):
    rows[f"n_model_{P}"] = np.stack([d[f"n_model_P{P}_T{t:.0f}"] for t in T])
    for q, src in (("x", "x_B"), ("T", "T"), ("n", "n")):
        rows[f"{q}_md_{P}"] = d[f"{src}_md_P{P}"]
save(draw_eos_density(rows, composition_label=r"$x_{\mathrm{B}}$",
     n_label=r"$n$ [Å$^{-3}$]", pressure_unit="GPa", model_label="model",
     T_label=r"$T$ [K]", T_ticks=[1200, 1600, 2000, 2400],
     T_tick_label=lambda v: f"{v:,.0f}", legend_loc="upper left", ink="black"),
     out + "/ed_feb_nx.pdf", dpi=600)
d = dict(np.load(figures + "/ed8_hhe_anchors/figdata/anchor_parity.npz"))
save(draw_anchor_parity(d, species=("H", "He"),
     M_unit=r"[$\mathrm{\AA}^{-1}\mathrm{eV}^{-1}\mathrm{ps}^{-1}$]",
     pressure_unit="GPa", model_label="AIPF", T_label=r"$T$ [$10^{3}$ K]",
     T_round=1000.0, T_tick_step=2000.0, T_tick_label=lambda t: f"{t/1000:.0f}",
     width=6.50, height=1.96), out + "/fig_anchor_parity.png", dpi=600)
'''

_RASTER = r'''
import json, sys
import numpy as np
def raster(path):
    if path.endswith(".png"):
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGBA")).astype(int)
    import fitz
    pm = fitz.open(path)[0].get_pixmap(dpi=200, alpha=True)
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n).astype(int)
res = {}
for name in sys.argv[3:]:
    a, b = raster(sys.argv[1] + "/" + name), raster(sys.argv[2] + "/" + name)
    res[name] = (-1 if a.shape != b.shape else int((abs(a - b).max(-1) > 0).sum()))
print(json.dumps(res))
'''


def _interpreter_env(py: str) -> dict:
    prefix = subprocess.run([py, "-c", "import sys; print(sys.prefix)"], check=True,
                            capture_output=True, text=True).stdout.strip()
    return dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
                PYTHONPATH=str(REPO / "src"),
                LD_LIBRARY_PATH=f"{prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}")


#: The interpreter the published figures were drawn in: not a site fact, so not located here.
_FIGURE_PYTHON = None


@pytest.mark.slow
def test_the_draw_halves_reproduce_the_published_rows_pixel_for_pixel(tmp_path):
    py = _FIGURE_PYTHON
    if not py or not Path(py).exists():
        pytest.skip("needs the interpreter the published figures were drawn in (matplotlib 3.10.8), "
                    "which this repository does not locate; the figure tests check the drawings")
    env = _interpreter_env(py)
    subprocess.run([py, "-c", _PORT, str(FIGURES), str(tmp_path)], check=True, env=env)
    rpy = py
    names = ["ed_hhe_nx.pdf", "ed_feb_nx.pdf", "fig_anchor_parity.png"]
    out = subprocess.run([rpy, "-c", _RASTER, str(tmp_path), str(REFERENCE), *names],
                         check=True, capture_output=True, text=True,
                         env=_interpreter_env(rpy)).stdout
    assert json.loads(out.strip().splitlines()[-1]) == {n: 0 for n in names}


def _hhe_or_skip():
    """The H/He system, or a skip naming what is missing: its published checkpoint, or its raw root
    (the equation-of-state manifolds and the anchor tables are read there)."""
    from aipf.system import load
    declared_roots.published_or_skip("hhe")
    declared_roots.raw_or_skip("hhe")
    return load("hhe")


@pytest.mark.slow
@pytest.mark.env
def test_the_isobars_are_the_published_ones():
    """The model's own isobars and the MD rows, against the Extended Data Fig. 7 figdata."""
    from aipf.md.analysis import isobar_rows
    system = _hhe_or_skip()
    d = np.load(FIGURES / "ed7_hhe_nx" / "figdata" / "density_isobars.npz")
    T_grid = [4000.0, 10000.0]
    rows = isobar_rows(system, pressures=[200.0, 800.0], T_grid=T_grid,
                       ckpt=system.checkpoint, x=d["x_He"])
    for P in (200, 800):
        for i, T in enumerate(T_grid):
            j = int(np.flatnonzero(d["T"] == T)[0])
            got, want = rows[f"n_model_{P}"][i], d[f"n_model_P{P}"][j]
            assert np.isfinite(got).all()
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
            md = rows[f"T_md_{P}"] == T
            np.testing.assert_array_equal(rows[f"n_md_{P}"][md], d[f"n_md_P{P}"][j])
            np.testing.assert_array_equal(rows[f"x_md_{P}"][md], d[f"x_He_md_P{P}"])


def test_the_anchor_rows_refuse_temperature_keyed_tables_by_name():
    from aipf.md.analysis import anchor_rows
    from aipf.system import load
    with pytest.raises(NotImplementedError, match="pressure-keyed.*'lj'.*'temperature'"):
        anchor_rows(load("lj"), ckpt="unused")


@pytest.mark.slow
@pytest.mark.env
def test_the_anchor_rows_are_the_published_ones():
    """Gamma and M at the declared anchor rows, against figs/fig_anchor_parity.npz (98 rows, the
    tables are float32, hence the tolerances)."""
    from aipf.md.analysis import anchor_rows
    system = _hhe_or_skip()
    got = anchor_rows(system, ckpt=system.checkpoint)
    want = np.load(FIGURES / "ed8_hhe_anchors" / "figdata" / "anchor_parity.npz")
    for pre, rtol in (("gamma", 1e-5), ("M", 1e-4)):
        a = np.lexsort((got[f"{pre}_T"], got[f"{pre}_P"]))
        b = np.lexsort((want[f"{pre}_T"], want[f"{pre}_P"]))
        np.testing.assert_array_equal(got[f"{pre}_P"][a], want[f"{pre}_P"][b])
        np.testing.assert_allclose(got[f"{pre}_T"][a], want[f"{pre}_T"][b], rtol=1e-6)
        for q in ("data", "model"):
            np.testing.assert_allclose(got[f"{pre}_{q}"][a], want[f"{pre}_{q}"][b],
                                       rtol=rtol, atol=0)
