"""The noise convention is declared (``Noise(mode, m_stab)``) and its filter is
derived from the system's own coarse-graining ``sigma``, never a second knob."""

import pytest
import torch

from aipf.solve import (UNDECLARED, build_noise_filter, check_m_stab,
                        declared_noise)
from aipf.spectral import SpectralOps
from aipf.system import Noise, load

_GRID = (16, 16, 16)
_BOXES = torch.tensor([[13.0, 14.5, 17.25]])


def test_the_filter_is_exp_minus_k2_sigma2_over_2_on_the_grid():
    ops = SpectralOps(_GRID, 2, nyquist_mask=True)
    G = build_noise_filter(ops, _BOXES, "gaussian", 2.0)
    kx, ky, kz = ops.k_axes(_BOXES)
    k2 = kx * kx + ky * ky + kz * kz
    assert torch.equal(G, torch.exp(-k2 * (2.0 * 2.0 / 2.0)))
    assert float(G[..., 0, 0, 0].min()) == 1.0


def test_mode_none_gives_no_filter():
    assert build_noise_filter(SpectralOps(_GRID, 2, nyquist_mask=True), _BOXES, "none",
                              None) is None


def test_an_unknown_mode_is_refused_naming_both_admissible_modes():
    with pytest.raises(ValueError) as err:
        Noise(mode="gauss", m_stab="mean")
    assert "'gaussian'" in str(err.value) and "'none'" in str(err.value)


@pytest.mark.parametrize("bad", ["median", "MAX", None])
def test_an_unknown_m_stab_is_refused_naming_mean_and_max(bad):
    with pytest.raises(ValueError) as err:
        Noise(mode="gaussian", m_stab=bad)
    assert "'mean'" in str(err.value) and "'max'" in str(err.value)
    with pytest.raises(ValueError):
        check_m_stab(bad)


def test_an_undeclared_m_stab_is_refused():
    with pytest.raises(ValueError, match="m_stab must be declared"):
        check_m_stab(UNDECLARED)


def test_noise_has_no_width_of_its_own():
    assert set(Noise.__dataclass_fields__) == {"mode", "m_stab"}


@pytest.mark.parametrize("name", ["hhe", "lj"])
def test_the_width_is_the_systems_coarse_graining_sigma(name):
    system = load(name)
    got = declared_noise(system)
    assert got["sigma_noise"] == float(system.defaults["sigma"])
    assert got["noise_mode"] == "gaussian"


@pytest.mark.parametrize("name,noise", [
    ("hhe", Noise(mode="gaussian", m_stab="max")),
    ("lj", Noise(mode="gaussian", m_stab="max")),
    ("feb", None)])
def test_every_system_declares_what_its_published_driver_ran(name, noise):
    """Cited per system in its ``system.py``; Fe-B published no SPDE (docs/reference/rollout.md, "The declaration")."""
    assert load(name).noise == noise


def test_a_system_without_noise_is_refused_by_the_noisy_path():
    with pytest.raises(ValueError, match="declares no noise"):
        declared_noise(load("feb"))


def test_mode_none_derives_no_width_and_undeclared_noise_is_refused():
    import dataclasses
    system = load("hhe")
    white = dataclasses.replace(system, noise=Noise("none", "mean"))
    assert declared_noise(white) == {"noise_mode": "none",
                                     "sigma_noise": None, "m_stab": "mean"}
    with pytest.raises(ValueError, match="declares no noise"):
        declared_noise(dataclasses.replace(system, noise=None))
