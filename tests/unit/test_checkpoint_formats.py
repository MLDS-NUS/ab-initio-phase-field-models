"""The Lightning-hparams whole-model bridge in `aipf.train.checkpoint_formats`.

The H/He and Fe-B checkpoints carry their model's settings as Lightning
hyper-parameters, and their architecture corresponds to this package's rung 3
wholesale, so the bridge is a key renaming plus a config translation -- and that is worth a test file of its own because the
thing it must never do is succeed PARTIALLY. A dropped tensor leaves a
randomly initialised submodule inside a model that otherwise reports a
clean load, which is the same silent-wrong-model failure the checkpoint
format layer exists to refuse. The last test here loads the tracked
published checkpoint end to end.
"""
from __future__ import annotations

import math

import pytest
import torch

from aipf.functional.nonlocal_kernel import NonlocalKernel
from aipf.system import load
from aipf.train.checkpoint_formats import (
    lightning_hparams_kernel_grid,
    lightning_hparams_model_state_dict,
    lightning_hparams_rung_kwargs,
    load_lightning_hparams_into,
)

import declared_roots

K_B = 8.617333262e-5



def _model_kwargs(**overrides):
    """The published model's own model_kwargs, spelled as the checkpoint spells
    them. Values are measured; this dict is what a real one looks like."""
    d = {
        "R_cut": 4.5, "T_ref": 10000.0, "activation": "gelu",
        "arrhenius_shared_Ea": False, "disable_g": False, "disable_u": False,
        "enable_T2": False, "enable_TlnT": True, "fexc_T_ref": 8000.0,
        "g_form": "mlp", "gauge_fix": False, "grid": (16, 16, 64),
        "h_g": 32, "h_m": 16, "h_u": 16, "h_w": 16,
        "m_T_form": "none", "m_form": "mlp_scaled", "m_table": "/x/M.npz",
        "rho_eps": 1e-5, "rho_ref": [0.35, 0.33], "tbasis_ortho": True,
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# the config translation
# ---------------------------------------------------------------------------

def test_the_translation_reads_the_nested_model_kwargs_not_the_top_level():
    """The architecture flags live one level down, inside `model_kwargs`.

    Looking at the top level of `hyper_parameters` finds none of them and
    raises no error, so a bridge written against the top level silently
    builds a model of entirely default architecture. This is not
    hypothetical: reading the top level is how `tbasis_ortho` was first
    reported ABSENT from a checkpoint that in fact sets it True.
    """
    hparams = {"lr": 5e-4, "lambda_S": 1.0,
               "model_kwargs": _model_kwargs()}
    assert "tbasis_ortho" not in hparams          # the trap, made explicit
    out = lightning_hparams_rung_kwargs(hparams["model_kwargs"], K_B)
    assert out["tbasis_ortho_points"] == 13
    assert out["enable_TlnT"] is True


def test_the_ortho_window_is_converted_from_kelvin_into_kBT():
    """The saved window is in Kelvin; FLocal declares it in kBT units."""
    out = lightning_hparams_rung_kwargs(
        _model_kwargs(tbasis_ortho=True,
                      tbasis_ortho_window=(9000.0, 12000.0)), K_B)
    lo, hi = out["tbasis_ortho_window"]
    assert lo == pytest.approx(K_B * 9000.0)
    assert hi == pytest.approx(K_B * 12000.0)


def test_no_window_is_declared_when_the_run_did_not_use_one():
    out = lightning_hparams_rung_kwargs(_model_kwargs(tbasis_ortho=False), K_B)
    assert "tbasis_ortho_window" not in out
    assert "tbasis_ortho_points" not in out


def test_an_omitted_flag_takes_the_training_codes_default_not_this_packages():
    """The value a checkpoint trained with is its training code's constructor
    default, and the two disagree: there `g_form` defaults to 'icnn', so
    taking this package's own default would build another entropy net than
    the one the run trained."""
    mk = _model_kwargs()
    del mk["g_form"]
    assert lightning_hparams_rung_kwargs(mk, K_B)["g_exc_form"] == "icnn"


def test_an_unrecognised_architecture_flag_raises():
    """An ignored architecture flag builds a DIFFERENT model that loads and
    trains without complaint -- the exact failure this package refuses."""
    with pytest.raises(KeyError, match="no rule for"):
        lightning_hparams_rung_kwargs(_model_kwargs(some_new_switch=True), K_B)


def test_the_input_scalings_are_bridged_to_their_constructor_arguments():
    """The two flags a published joint-head model carries.

    They are not cosmetic: with them on, every excess net and the mobility
    are fed `z = rho/rho_ref - 1` rather than `rho`, and both land in
    NON-persistent buffers, so a strict load succeeds whichever way they are
    set. The bridge therefore carries them through by name, never drops them.
    """
    out = lightning_hparams_rung_kwargs(
        _model_kwargs(fexc_input_scale=True, m_input_scale=True), K_B)
    assert out["local_input_scale"] is True
    assert out["mobility_input_ref"] == out["rho_ref"]


def test_absent_input_scalings_are_the_raw_density():
    out = lightning_hparams_rung_kwargs(_model_kwargs(), K_B)
    assert out["local_input_scale"] is False
    assert out["mobility_input_ref"] is None


def test_the_joint_nets_width_and_depth_are_bridged_under_the_joint_form():
    out = lightning_hparams_rung_kwargs(
        _model_kwargs(f_exc_form="joint", g_form="icnn", enable_TlnT=False,
                      h_u_joint=8, n_hidden_joint=3), K_B)
    assert (out["h_joint"], out["joint_depth"]) == (8, 3)
    split = lightning_hparams_rung_kwargs(_model_kwargs(h_u_joint=8), K_B)
    assert "h_joint" not in split and "joint_depth" not in split


def test_an_arrhenius_mobility_starts_from_the_saved_barriers():
    out = lightning_hparams_rung_kwargs(_model_kwargs(m_T_form="arrhenius"), K_B)
    assert out["mobility_t_form"] == "arrhenius"
    assert out["mobility_activation_energy_init"] == [
        math.log(math.expm1(0.38)), math.log(math.expm1(0.50))]
    with pytest.raises(NotImplementedError, match="shared"):
        lightning_hparams_rung_kwargs(_model_kwargs(m_T_form="arrhenius",
                                           arrhenius_shared_Ea=True), K_B)


def test_the_arrhenius_barrier_tensor_is_renamed_not_dropped():
    from aipf.train.checkpoint_formats import lightning_hparams_model_state_dict

    renamed = lightning_hparams_model_state_dict({"mobility.E_raw": torch.zeros(2)})
    assert set(renamed) == {"_mobility._activation_energy_raw"}


def test_a_convex_temperature_head_is_refused_rather_than_dropped():
    """The one architecture flag this bridge used to drop on purpose.

    `ghat_form` chooses the SHAPE of the TlnT head in the training code's class:
    a plain head, or an input-convex one. This package builds the plain
    head only, so a checkpoint asking for the convex one asks for a model
    this package cannot build -- and dropping the flag builds the plain
    head silently, which is the swallowed-kwarg failure every other line
    of this bridge refuses.

    Unreachable today, which is why it was dropped: no measured published model
    sets it, and the training code's class refuses that head together with the
    T-basis orthogonalisation every measured published model does use. Refusing
    costs nothing while it stays unreachable and is the difference between
    a loud stop and a wrong model if it ever is not.
    """
    with pytest.raises(NotImplementedError, match="ghat_form"):
        lightning_hparams_rung_kwargs(_model_kwargs(ghat_form="icnn"), K_B)
    # the shape this package does build is consumed, not refused
    assert lightning_hparams_rung_kwargs(_model_kwargs(ghat_form="mlp"), K_B)
    assert lightning_hparams_rung_kwargs(_model_kwargs(ghat_form=None), K_B)


def test_an_unbridged_mobility_form_raises_rather_than_guessing():
    with pytest.raises(NotImplementedError, match="m_form"):
        lightning_hparams_rung_kwargs(_model_kwargs(m_form="constant"), K_B)


def test_kBT_ref_is_derived_from_fexc_T_ref_and_kB():
    out = lightning_hparams_rung_kwargs(_model_kwargs(fexc_T_ref=8000.0), K_B)
    assert out["kBT_ref"] == pytest.approx(K_B * 8000.0)


# ---------------------------------------------------------------------------
# the kernel grid, which is NOT in model_kwargs
# ---------------------------------------------------------------------------

def test_the_kernel_grid_is_read_back_from_the_stored_buffers():
    """The training code's kernel takes n_quad/k_table_max as constructor arguments
    and stores the resulting GRIDS as persistent buffers, so a checkpoint
    records the grids and not the arguments."""
    sd = {"model.kernel.r_quad": torch.linspace(0.0, 4.5, 512),
          "model.kernel.k_lin": torch.linspace(0.0, 8.0, 513)}
    got = lightning_hparams_kernel_grid(sd)
    assert got["kernel_n_quad"] == 512
    assert got["kernel_n_k_table"] == 513
    assert got["R_cut"] == pytest.approx(4.5)
    assert got["kernel_k_table_max"] == pytest.approx(8.0)


def test_a_checkpoint_that_disagrees_with_itself_about_R_cut_raises():
    sd = {"model.kernel.r_quad": torch.linspace(0.0, 3.0, 512),
          "model.kernel.k_lin": torch.linspace(0.0, 8.0, 513)}
    with pytest.raises(ValueError, match="disagrees with itself"):
        lightning_hparams_rung_kwargs(_model_kwargs(R_cut=4.5), K_B, sd)


def test_a_missing_kernel_buffer_raises_rather_than_being_guessed():
    with pytest.raises(KeyError, match="does not guess"):
        lightning_hparams_kernel_grid({"model.kernel.r_quad": torch.zeros(4)})


# ---------------------------------------------------------------------------
# the tensor renaming
# ---------------------------------------------------------------------------

def test_lit_module_anchor_tables_are_dropped_but_model_tensors_are_not():
    sd = {"model.f_local.u_net.net.0.weight": torch.zeros(2, 2),
          "bulk_rho": torch.zeros(98, 2),        # LitModule anchor table
          "gamma_T_nodes": torch.zeros(11)}
    out = lightning_hparams_model_state_dict(sd)
    assert set(out) == {"f_local.u_net.net.net.0.weight"}


def test_an_unknown_model_tensor_raises_rather_than_being_dropped():
    with pytest.raises(KeyError, match="no Lightning-hparams rename rule"):
        lightning_hparams_model_state_dict({"model.mystery_net.weight": torch.zeros(1)})


def test_a_bare_model_state_dict_must_account_for_every_key():
    """With no `model.` prefix there is nothing to tell weights from
    training state, so nothing may be dropped on a guess."""
    with pytest.raises(KeyError, match="no Lightning-hparams rename rule"):
        lightning_hparams_model_state_dict({"bulk_rho": torch.zeros(98, 2)})


# ---------------------------------------------------------------------------
# the tracked published checkpoint
# ---------------------------------------------------------------------------

def test_the_real_published_loads_into_nonlocal_kernel_strictly():
    """End to end on the first system's published model: build the rung
    from the checkpoint's own config, load its tensors, strict=True.

    `strict=True` is the whole point. Anything missing or unexpected is an
    error, so this cannot pass with a partially populated model.
    """
    declared_roots.published_or_skip("hhe")
    ckpt = torch.load(load("hhe").resolve_checkpoint(), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    kwargs = lightning_hparams_rung_kwargs(
        ckpt["hyper_parameters"]["model_kwargs"], K_B, sd)
    model = load_lightning_hparams_into(NonlocalKernel(**kwargs), sd)

    assert kwargs["tbasis_ortho_window"] is not None
    assert model.f_local.tbasis_ortho is True

    B, grid = 2, kwargs["grid"]
    N = grid[0] * grid[1] * grid[2]
    torch.manual_seed(0)
    rho = 0.3 + 0.05 * torch.rand(B, 2, *grid)
    rho_hat = torch.fft.rfftn(rho, dim=(-3, -2, -1)) / N
    boxes = torch.tensor([[12.0, 12.0, 48.0]] * B)
    T = torch.full((B,), 6000.0)

    out = model(rho_hat, boxes, T)
    assert out.shape == rho_hat.shape
    assert torch.isfinite(out.view(-1).real).all()
    # Model B conserves mass exactly: the divergence of anything is zero at
    # k = 0, so this is a structural identity, not a tolerance.
    assert out[..., 0, 0, 0].abs().max() == 0.0


