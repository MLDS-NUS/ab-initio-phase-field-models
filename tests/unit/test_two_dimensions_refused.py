"""What is three-dimensional only says so, by name, when a two-dimensional model or grid reaches it.

Each route is written for a ``(Gx, Gy, Gz)`` grid and ``(B, 3)`` boxes: the other functional forms, the
lattice-sum kernel, the 3D moments ``W_hat(0)`` and ``kappa_eff``, the anchors, penalties and kernel
hinge, the diagnosis, the slab, spinodal and FDT drivers, their observables, the KDE deposit, and the
checkpoint layouts saved before two dimensions existed. None of them is reached by a 3D model any
differently than before (``tests/golden``)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

import toy_factory_model as toy
from aipf.diagnose import dynamics, one_field
from aipf.diagnose.run import load_model as diagnose_load_model
from aipf.functional.build import build
from aipf.functional.kernels import (AnalyticRadialTransform, LatticeSumTransform, PairKernel,
                                     QuinticEnvelope, RadialKernelSet)
from aipf.functional.landau import FloryHuggins, Landau
from aipf.functional.neural_operator import NeuralOperator
from aipf.functional.nonlocal_kernel import NonlocalKernel
from aipf.functional.square_gradient import SquareGradient
from aipf.pipeline import kde
from aipf.rollout import fdt, observables
from aipf.rollout.spinodal import load_model as spinodal_load_model
from aipf.spectral import SpectralOps2D
from aipf.train.anchors import AnchorTables
from aipf.train.checkpoint_formats import load_kmodes_into, load_lightning_hparams_into
from aipf.train.kernel_hinge import KernelHinge
from aipf.train.penalties import ConvexityProbe, GammaPaths
from toy_ndim_model import build_ndim_toy

GRID2 = (8, 6)
THREE_D_ONLY = "three-dimensional only"


def _system(grid):
    base = toy.demo_system("unused")
    return dataclasses.replace(base, functional=toy.toy_functional(build_ndim_toy, grid=grid),
                               variants={})


@pytest.fixture(scope="module")
def model2d():
    return build(_system(GRID2))


# ---------------------------------------------------------------------------
# the functional forms and the kernel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: Landau(GRID2, 1, 1.0, 1.0, 1.0, 1.0, nyquist_mask=True),
    lambda: FloryHuggins(GRID2, 2, 1.0, w=1.0, nyquist_mask=True),
    lambda: SquareGradient(GRID2, 1, local="landau", nyquist_mask=True, kappa_init=1.0),
    lambda: NeuralOperator(GRID2, 1, nyquist_mask=True),
], ids=["landau", "fh", "square_gradient", "neural_operator"])
def test_the_other_functional_forms_refuse_a_two_axis_grid(make):
    with pytest.raises(NotImplementedError, match=THREE_D_ONLY):
        make()


def _radial_set():
    torch.manual_seed(0)
    return RadialKernelSet(2, QuinticEnvelope(2.5), hidden=4)


def test_the_lattice_sum_refuses_two_dimensions():
    with pytest.raises(NotImplementedError, match="lattice_sum"):
        NonlocalKernel(GRID2, 2, 1.0, [0.5, 0.4], 8, 2.5, ideal_form="gas", h_u=8,
                       nyquist_mask=True, kernel_evaluator="lattice_sum")
    with pytest.raises(NotImplementedError, match="three-dimensional only"):
        LatticeSumTransform().w_hat(_radial_set(), torch.zeros(1, 8, 4), grid=GRID2,
                                    boxes=torch.ones(1, 2))


def test_the_3d_moments_of_w_refuse_a_two_dimensional_kernel():
    kernel = PairKernel(_radial_set(), AnalyticRadialTransform(2.5, 32, 32, 8.0, dim=2), n_quad=32)
    with pytest.raises(NotImplementedError, match="W_hat"):
        kernel.w_hat_zero_radial(2.5, 32)
    with pytest.raises(NotImplementedError, match="W_hat"):
        kernel.w_hat_zero_quadrature(2.5, 32)
    with pytest.raises(NotImplementedError, match="kappa_eff"):
        kernel.kappa_eff()
    three = PairKernel(_radial_set(), AnalyticRadialTransform(2.5, 32, 32, 8.0), n_quad=32)
    assert tuple(three.kappa_eff().shape) == (2, 2)


# ---------------------------------------------------------------------------
# training extras, diagnosis, drivers and observables
# ---------------------------------------------------------------------------

def test_the_training_extras_refuse_a_two_dimensional_model(model2d):
    with pytest.raises(NotImplementedError, match="anchor"):
        AnchorTables.batch(None, model2d, 1.0)
    with pytest.raises(NotImplementedError, match="convexity"):
        ConvexityProbe.loss(None, model2d, None)
    with pytest.raises(NotImplementedError, match="Gamma"):
        GammaPaths.loss(None, model2d, None)
    with pytest.raises(NotImplementedError, match="hinge"):
        KernelHinge.check(model2d)
    with pytest.raises(NotImplementedError, match="hinge"):
        KernelHinge.loss(None, model2d)


def test_the_diagnosis_refuses_a_two_dimensional_model(model2d, tmp_path):
    path = tmp_path / "toy2d.ckpt"
    torch.save({"state_dict": model2d.state_dict()}, path)
    with pytest.raises(NotImplementedError, match="diagnosis"):
        diagnose_load_model(_system(GRID2), path)
    with pytest.raises(NotImplementedError, match="one-field"):
        one_field._check_model(model2d)
    ops = SpectralOps2D(GRID2, 2, nyquist_mask=True)
    with pytest.raises(NotImplementedError, match="structure_factor"):
        dynamics.structure_factor(torch.zeros(1, 2, 8, 4, dtype=torch.complex64), ops,
                                  torch.ones(1, 2), 4)
    with pytest.raises(NotImplementedError, match="rollout_drift"):
        dynamics.rollout_drift(model2d, None, None, None, 1.0, ops, alpha=0.0, k_max=1.0)


def test_the_rollout_drivers_and_their_observables_refuse_two_dimensions(model2d, tmp_path):
    path = tmp_path / "toy2d.ckpt"
    torch.save({"state_dict": model2d.state_dict()}, path)
    with pytest.raises(NotImplementedError, match="spinodal and slab"):
        spinodal_load_model(_system(GRID2), path)
    with pytest.raises(NotImplementedError, match="FDT"):
        fdt.homogeneous_state([0.5, 0.4], GRID2, "cpu")
    rho = np.ones((3, 2, *GRID2))
    with pytest.raises(NotImplementedError, match="decomp_metrics"):
        observables.decomp_metrics(rho, 8.0, 1)
    with pytest.raises(NotImplementedError, match="conc_profile"):
        observables.conc_profile(rho, 1, 0)
    with pytest.raises(NotImplementedError, match="mode_wavevectors"):
        observables.mode_wavevectors((8.0, 6.0), GRID2)
    with pytest.raises(NotImplementedError, match="measure_S_modes"):
        observables.measure_S_modes(np.ones((3, 2, 8, 4), dtype=complex), 48.0)


def test_the_kde_deposit_refuses_a_two_entry_grid_as_it_always_refused_one():
    with pytest.raises(NotImplementedError, match="three-dimensional atom positions"):
        kde._as_grid((8, 8))
    with pytest.raises(ValueError, match="grid"):
        kde._as_grid((8, 8))


# ---------------------------------------------------------------------------
# the checkpoint layouts whose grid buffers are dropped on load
# ---------------------------------------------------------------------------

def test_a_lightning_hparams_checkpoint_of_another_dimension_is_refused(model2d):
    saved_3d = {"model.ops.NX": torch.zeros(8, 1, 1), "model.f_local.a_raw": torch.zeros(2)}
    with pytest.raises(ValueError, match="3-axis grid and the model is built on a 2-axis"):
        load_lightning_hparams_into(model2d, saved_3d)
    # no grid buffer saved: the layout predates two dimensions, so it is read as 3D
    with pytest.raises(ValueError, match="3-axis grid and the model is built on a 2-axis"):
        load_lightning_hparams_into(model2d, {"model.f_local.a_raw": torch.zeros(2)})
    model3d = build(_system((4, 4, 4)))
    with pytest.raises(ValueError, match="2-axis grid and the model is built on a 3-axis"):
        load_lightning_hparams_into(model3d, {"model.ops.NX": torch.zeros(4, 1)})


def test_a_kmodes_checkpoint_is_refused_by_a_two_dimensional_model(model2d):
    with pytest.raises(ValueError, match="k-modes checkpoint"):
        load_kmodes_into(model2d, {"model_state_dict": {}})
