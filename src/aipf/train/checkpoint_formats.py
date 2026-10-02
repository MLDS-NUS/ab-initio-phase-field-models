"""Checkpoints saved in two other layouts, loaded onto this package's rung submodules.

The Lightning-hparams layout (a ``state_dict`` of ``model.*`` tensors with nested ``model_kwargs``) and the
k-modes layout (one-field rung; ``model_state_dict``, or ``state_dict["model.*"]``). A tensor is copied
where both sides share a parametrisation and reparametrised in closed form where they make a quantity
positive or centre it differently (stored floats differ, the physical quantity agrees)."""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

#: Sentinel for "this key has no default and its absence is an error".
_REQUIRED = object()


def softplus_inverse(y: torch.Tensor) -> torch.Tensor:
    """Inverse of ``softplus`` for ``y > 0``: ``log(exp(y) - 1)`` as ``y + log(-expm1(-y))``."""
    return y + torch.log(-torch.expm1(-y))


def kernel_radial_net_state_dict(state_dict: Dict[str, torch.Tensor],
                                  *, prefix: str = "w_net"
                                  ) -> Dict[str, torch.Tensor]:
    """Straight copy of one ``RadialKernelSet`` net from ``<prefix>.{0,2,4}.{weight,bias}``."""
    out = {}
    for i in (0, 2, 4):
        out[f"{i}.weight"] = state_dict[f"{prefix}.{i}.weight"].clone()
        out[f"{i}.bias"] = state_dict[f"{prefix}.{i}.bias"].clone()
    return out


def icnn_state_dict_from_bias_free(state_dict: Dict[str, torch.Tensor],
                                   *, prefix: str
                                   ) -> Dict[str, torch.Tensor]:
    """Straight copy of a saved bias-free one-dimensional ICNN (``<prefix>.{W0,A1_raw,W1,a2_raw,w2}``) into
    :class:`~aipf.functional.local_forms.ICNN`'s ``state_dict()``.

    ``w2.bias`` has no saved counterpart and is set to exactly zero."""
    p = prefix
    w2_weight = state_dict[f"{p}.w2.weight"]
    return {
        "W0.weight": state_dict[f"{p}.W0.weight"].clone(),
        "W0.bias": state_dict[f"{p}.W0.bias"].clone(),
        "A1_raw": state_dict[f"{p}.A1_raw"].clone(),
        "W1.weight": state_dict[f"{p}.W1.weight"].clone(),
        "W1.bias": state_dict[f"{p}.W1.bias"].clone(),
        "a2_raw": state_dict[f"{p}.a2_raw"].clone(),
        "w2.weight": w2_weight.clone(),
        "w2.bias": torch.zeros(w2_weight.shape[0], dtype=w2_weight.dtype),
    }


def taylor_coefs_from_psi_centered(coefs: torch.Tensor,
                                   exponents: Sequence[int], *,
                                   rho_ref: float) -> torch.Tensor:
    """Taylor coefs on ``psi = rho - 0.5`` -> ``TaylorEnergy`` coefs on ``z = rho/rho_ref - 1``.

    ``out[k] = coefs[k] / (2*rho_ref)**exponents[k]``, exact only at ``rho_ref == 0.5``;
    otherwise raises."""
    if float(rho_ref) != 0.5:
        raise ValueError(
            f"rho_ref={rho_ref!r}: this closed-form conversion is exact "
            f"only at rho_ref=0.5, where this package's z=rho/rho_ref-1 "
            f"and the saved additive psi=rho-0.5 are proportional "
            f"(z=2*psi). At any other rho_ref the two polynomials are "
            f"centred at different points and no per-power rescale "
            f"reproduces one from the other.")
    scale = torch.tensor([2.0 ** e for e in exponents],
                         dtype=coefs.dtype)
    return coefs / scale


def mobility_constant_raw_from_gamma(gamma: torch.Tensor) -> torch.Tensor:
    """Raw ``Mobility(shape="constant")`` parameter with ``softplus(raw)**2 == gamma`` (a saved
    ``exp(log_gamma)``).

    Equal as a physical scale, not as stored floats."""
    return softplus_inverse(gamma.clamp(min=1e-12).sqrt())


# The Lightning-hparams layout: the whole-model bridge (key renaming plus config translation).

# Saved prefix -> this package's prefix, longest-first.
LIGHTNING_HPARAMS_PREFIX_RENAMES: Tuple[Tuple[str, str], ...] = (
    ("mobility.E_raw", "_mobility._activation_energy_raw"),
    ("f_local.u_net.net.", "f_local.u_net.net.net."),
    ("kernel.nets.", "kernel.radial_set.nets."),
    ("kernel.r_quad", "kernel.evaluator.r_quad"),
    ("kernel.k_lin", "kernel.evaluator.k_table"),
    ("mobility.", "_mobility."),
)

# Derived grid state, not weights: rebuilt from `grid` and `boxes`; any other unmatched key is an error.
LIGHTNING_HPARAMS_DERIVED_PREFIXES: Tuple[str, ...] = ("ops.",)


def lightning_hparams_model_state_dict(
        state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rename a Lightning-hparams checkpoint's model tensors (bare or ``"model."``-prefixed) onto rung 3's keys.

    Derived buffers are left out (see :func:`load_lightning_hparams_into`); a key with no rule raises."""
    # Non-`model.` keys of a Lightning state_dict are LitModule anchor tables, dropped deliberately.
    lightning = any(k.startswith("model.") for k in state_dict)
    out: Dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        if lightning and not raw_key.startswith("model."):
            continue
        key = raw_key[len("model."):] if raw_key.startswith("model.") else raw_key
        if key.startswith(LIGHTNING_HPARAMS_DERIVED_PREFIXES):
            continue
        for old, new in LIGHTNING_HPARAMS_PREFIX_RENAMES:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        else:
            if not (key.startswith("f_local.")
                    or key.startswith("kernel.")):
                raise KeyError(
                    f"no Lightning-hparams rename rule for checkpoint key "
                    f"{raw_key!r}; refusing to drop it silently, because a "
                    f"dropped tensor leaves a randomly initialised "
                    f"submodule behind in a model that looks loaded")
        out[key] = value
    return out


def lightning_hparams_kernel_grid(
        state_dict: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """The kernel's quadrature grid read off the checkpoint's ``r_quad``/``k_lin`` buffers, never
    re-derived."""
    def get(name):
        for key in (f"model.kernel.{name}", f"kernel.{name}"):
            if key in state_dict:
                return state_dict[key]
        raise KeyError(
            f"checkpoint has no kernel.{name} buffer, so its quadrature "
            f"grid cannot be read back; this bridge does not guess it")

    r_quad, k_lin = get("r_quad"), get("k_lin")
    return {
        "kernel_n_quad": int(r_quad.numel()),
        "R_cut": float(r_quad[-1]),
        "kernel_n_k_table": int(k_lin.numel()),
        "kernel_k_table_max": float(k_lin[-1]),
    }


def lightning_hparams_rung_kwargs(
        model_kwargs: Mapping[str, Any], kB: float,
        state_dict: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Translate a Lightning-hparams checkpoint's nested ``model_kwargs`` (not its top level) into rung-3 kwargs.

    Unrecognised keys raise; a missing key takes a default only where the saving model's constructor had one."""
    mk = dict(model_kwargs)

    def take(name, *, default=_REQUIRED):
        if name in mk:
            return mk.pop(name)
        if default is _REQUIRED:
            raise KeyError(
                f"model_kwargs is missing {name!r}, which this "
                f"package has no default for")
        return default

    fexc_T_ref = float(take("fexc_T_ref"))
    out: Dict[str, Any] = {
        "grid": tuple(take("grid")),
        "kB": float(kB),
        "kBT_ref": float(kB) * fexc_T_ref,
        "ideal_form": "gas",
        # The saving model's spectral operators zero the Nyquist wavenumber in grad and div
        # (the odd-order masks).
        "nyquist_mask": True,
    }
    out["rho_ref"] = [float(r) for r in take("rho_ref")]
    out["n_species"] = len(out["rho_ref"])

    out["h_u"] = int(take("h_u"))
    out["h_g"] = int(take("h_g"))
    out["R_cut"] = float(take("R_cut"))
    out["kernel_hidden"] = int(take("h_w"))
    out["mobility_hidden"] = int(take("h_m"))
    out["rho_eps"] = float(take("rho_eps", default=1e-5))
    # Missing flags take the saving model's constructor defaults, not this package's.
    out["u_form"] = str(take("u_form", default="mlp"))
    out["g_exc_form"] = str(take("g_form", default="icnn"))
    out["gauge_fix"] = bool(take("gauge_fix", default=False))
    out["enable_TlnT"] = bool(take("enable_TlnT", default=False))
    out["enable_T2"] = bool(take("enable_T2", default=False))
    out["f_exc_form"] = str(take("f_exc_form", default="split"))
    # The joint net's width and depth, at the saving constructor's defaults (32, 2) when absent.
    h_joint = int(take("h_u_joint", default=32))
    joint_depth = int(take("n_hidden_joint", default=2))
    if out["f_exc_form"] == "joint":
        out["h_joint"] = h_joint
        out["joint_depth"] = joint_depth
    # z = rho/rho_ref - 1 into the excess nets and the mobility net; absent = raw rho (the saved default).
    out["local_input_scale"] = bool(take("fexc_input_scale", default=False))
    out["mobility_input_ref"] = (list(out["rho_ref"])
                                 if bool(take("m_input_scale", default=False))
                                 else None)

    activation = str(take("activation"))
    out["local_activation"] = activation
    out["kernel_activation"] = activation
    out["mobility_activation"] = activation

    out["h_g_hat"] = int(take("h_g_hat", default=16))
    out["h_g_tilde"] = int(take("h_g_tilde", default=16))

    # The saved window is in kelvin; converted to kBT_ref units. 13 points is pinned, not a tunable.
    if bool(take("tbasis_ortho", default=False)):
        lo, hi = take("tbasis_ortho_window", default=(9000.0, 12000.0))
        out["tbasis_ortho_window"] = (float(kB) * float(lo),
                                      float(kB) * float(hi))
        out["tbasis_ortho_points"] = 13
    else:
        mk.pop("tbasis_ortho_window", None)

    m_form = str(take("m_form"))
    if m_form != "mlp_scaled":
        raise NotImplementedError(
            f"m_form={m_form!r} is not bridged; only "
            f"'mlp_scaled' has been measured against a real checkpoint")
    out["mobility_shape"] = "mlp_rho"
    out["mobility_prefactor"] = "partial_density"
    out["mobility_t_form"] = {"none": "none",
                              "arrhenius": "arrhenius"}[
                                  str(take("m_T_form", default="none"))]
    out["mobility_t_ref"] = float(take("T_ref", default=10000.0))
    if out["mobility_t_form"] == "arrhenius":
        # the saving constructor's own start: softplus^-1 of 0.38 and 0.50 (energy units), per channel
        if bool(mk.get("arrhenius_shared_Ea", False)):
            raise NotImplementedError(
                "arrhenius_shared_Ea=True is one barrier shared by "
                "every channel; this package builds one per channel")
        out["mobility_activation_energy_init"] = [
            math.log(math.expm1(0.38)), math.log(math.expm1(0.50))]

    # A non-plain `ghat_form` is refused, not dropped: only the plain TlnT head is built here.
    ghat_form = take("ghat_form", default=None)
    if ghat_form is not None and str(ghat_form) != "mlp":
        raise NotImplementedError(
            f"ghat_form={ghat_form!r} is not bridged: this "
            f"package builds the plain temperature head only, and dropping "
            f"the flag would build that head where the run had another")

    # Consumed but not part of the architecture this package builds.
    # `m_column` / `m_init_from_table` only seed the mobility's last bias, which a load overwrites.
    for spent in ("m_table", "arrhenius_shared_Ea", "disable_u", "disable_g",
                  "u_poly_degree", "tail_sigma", "init_attract",
                  "m_bound", "fixed_M", "fixed_select", "m_column",
                  "m_init_from_table"):
        mk.pop(spent, None)
    if mk:
        raise KeyError(
            f"model_kwargs carries keys this bridge has no rule "
            f"for: {sorted(mk)}. Refusing to ignore them, because an "
            f"ignored architecture flag builds a DIFFERENT model that "
            f"loads and trains without complaint")

    if state_dict is not None:
        grid_kwargs = lightning_hparams_kernel_grid(state_dict)
        stored_R_cut = grid_kwargs.pop("R_cut")
        if abs(stored_R_cut - float(out["R_cut"])) > 1e-6:
            raise ValueError(
                f"checkpoint disagrees with itself: model_kwargs says "
                f"R_cut={out['R_cut']!r}, the stored r_quad buffer ends at "
                f"{stored_R_cut!r}")
        out.update(grid_kwargs)
    return out


def load_lightning_hparams_into(model, state_dict: Dict[str, torch.Tensor]):
    """Load a Lightning-hparams checkpoint into a built rung 3, derived buffers from the model,
    ``strict=True``; returns the model."""
    incoming = lightning_hparams_model_state_dict(state_dict)
    current = model.state_dict()
    unexpected = sorted(set(incoming) - set(current))
    if unexpected:
        raise KeyError(
            f"renamed checkpoint keys that this model does not have: "
            f"{unexpected}")
    merged = dict(current)
    merged.update(incoming)
    model.load_state_dict(merged, strict=True)
    return model


# The k-modes layout (one-field lattice rung; ``model_state_dict``, or ``state_dict["model.*"]``).

#: Saved key (exact, or prefix when it ends in ".") -> this package's key.
KMODES_RENAMES: Tuple[Tuple[str, str], ...] = (
    ("_log_gamma", "_mobility._log_gamma"),
    ("_Ea_es_raw", "_mobility._Ea_raw"),
    # rung 2's closed local forms: the barrier's saved name, and the bulk constants under their own
    ("_Ea_raw", "_mobility._Ea_raw"),
    ("_log_kappa", "_log_kappa"),
    ("w", "w"),
    ("_log_a0", "_log_a0"),
    ("_log_b", "_log_b"),
    ("T_c", "T_c"),
    ("taylor_coefs", "f_local.u_net.coefs"),
    ("w_net.", "kernel.radial_set.nets.0."),
    ("g_exc_net.", "f_local.g_net."),
)

#: LitModule buffers beside ``model.*``: k grids, band and filter, spacing, anchor tables. Dropped.
KMODES_LIT_BUFFERS: Tuple[str, ...] = (
    "k2", "mult", "band", "filt", "spacing", "sT", "s_inv", "s_sig", "mT",
    "m_val")


def kmodes_model_state_dict(saved: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """A k-modes checkpoint's model tensors renamed onto the one-field rung's keys; an unmatched key raises."""
    if "model_state_dict" in saved:
        raw = dict(saved["model_state_dict"])
    else:
        raw = {}
        for key, value in saved["state_dict"].items():
            if key.startswith("model."):
                raw[key[len("model."):]] = value
            elif key not in KMODES_LIT_BUFFERS:
                raise KeyError(f"k-modes checkpoint key {key!r} is neither "
                               f"a model tensor nor a LitModule buffer")
    out: Dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        for old, new in KMODES_RENAMES:
            if key == old or (old.endswith(".") and key.startswith(old)):
                out[new + key[len(old):]] = value
                break
        else:
            raise KeyError(f"no k-modes rename rule for checkpoint key {key!r}")
    return out


def load_kmodes_into(model, saved: Mapping[str, Any]):
    """Load a k-modes checkpoint into a built one-field rung (2 or 3), derived buffers from the model,
    ``strict=True``."""
    incoming = kmodes_model_state_dict(saved)
    current = model.state_dict()
    unexpected = sorted(set(incoming) - set(current))
    if unexpected:
        raise KeyError(f"renamed checkpoint keys that this model does not have: "
                       f"{unexpected}")
    merged = dict(current)
    merged.update(incoming)
    model.load_state_dict(merged, strict=True)
    return model
