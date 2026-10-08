"""The LitModule: the canonical losses combined by their canonical weights.

``L_dyn`` is built here from the model's ``forward`` and :mod:`aipf.spectral`; every anchor term takes
tensors the caller already built, and ``L_conv``/``L_Gamma`` take a ``(loss, aux)`` pair built by
:mod:`aipf.train.penalties`. Model weights come from a child stream of the one declared seed."""
from __future__ import annotations

import contextlib
import copy
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MeanMetric

from aipf.functional.base import FreeEnergyModel
from aipf.losses import l_bulk, l_dyn, l_m, l_p_absolute, l_p_variance, l_s
from aipf.spectral import OpsCache, SpectralOps, SpectralOps2D, ops_ndim
from aipf.train.config import TrainConfig

#: Canonical loss names logged and totalled, in this order.
_CANONICAL_ORDER: Tuple[str, ...] = (
    "L_dyn", "L_M", "L_S", "L_bulk", "L_P", "L_conv", "L_Gamma",
)


@contextlib.contextmanager
def seeded_rng(seed: int) -> Iterator[None]:
    """Run a block with the global torch CPU stream forked and seeded, restored on exit.

    Only the CPU stream: build on the CPU, then ``.to(device)``."""
    with torch.random.fork_rng(devices=()):
        torch.default_generator.manual_seed(int(seed))
        yield


#: ``conv``/``gamma`` argument: a built ``(loss, aux)`` pair.
ProbeArg = Tuple[torch.Tensor, dict]


#: Per number of spatial axes, the view that broadcasts the ``(B, n_s)`` state weights over
#: ``(B, n_s, n_species, *half spectrum)``.
_STATE_WEIGHT_VIEW = {3: (1, 1, 1, 1), 2: (1, 1, 1)}


class LitModule(L.LightningModule):
    """Wraps a :class:`~aipf.functional.base.FreeEnergyModel`, its ops (``OpsCache``, ``SpectralOps`` or
    ``SpectralOps2D``) and a :class:`TrainConfig`. The number of spatial axes is the ops' declared one
    (``ops.ndim``), never read off a batch's shape."""

    def __init__(self, model: FreeEnergyModel, ops, config: TrainConfig) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model must be an nn.Module implementing FreeEnergyModel, "
                f"got {type(model).__name__}")
        self.model = model
        if isinstance(ops, OpsCache):
            self._ops_cache: Optional[OpsCache] = ops
            self._ops: Optional[SpectralOps] = None
        elif isinstance(ops, (SpectralOps, SpectralOps2D)):
            self._ops_cache = None
            self._ops = ops
        else:
            raise TypeError(
                f"ops must be an OpsCache or a SpectralOps, got "
                f"{type(ops).__name__}")
        self.config = config
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    @classmethod
    def seeded(cls, build_model: Callable[[], nn.Module], ops,
               config: TrainConfig) -> "LitModule":
        """Build the model under the config seed's ``"model"`` stream (:func:`seeded_rng`) and wrap it.

        ``build_model`` is a zero-argument callable; no declared seed raises."""
        if isinstance(build_model, nn.Module) or not callable(build_model):
            raise TypeError(
                f"build_model must be a callable returning the model, got "
                f"{type(build_model).__name__}. An already-built model is "
                f"the one thing this refuses although it IS callable: "
                f"building it happened before the seed was applied, so its "
                f"weights are whatever the global stream held, and calling "
                f"it here would run its forward pass instead.")
        seed = config.seed_for("model")
        with seeded_rng(seed):
            model = build_model()
        return cls(model, ops, config)

    def _ops_for(self, rfft_shape, device) -> SpectralOps:
        if self._ops_cache is not None:
            return self._ops_cache.ops_for(rfft_shape, device)
        return self._ops

    def resolved_k_max(self) -> float:
        """``config.k_max`` if declared, else ``4 / sigma``."""
        if self.config.k_max is not None:
            return float(self.config.k_max)
        if self.config.sigma is None:
            raise ValueError(
                "k_max is not declared and sigma is None: the drift band "
                "cutoff (4/sigma) cannot be resolved")
        return 4.0 / float(self.config.sigma)

    def drift_loss(self, rho_hat_states: torch.Tensor, lam: torch.Tensor,
                    target_hat: torch.Tensor, boxes: torch.Tensor,
                    T: torch.Tensor, *,
                    sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The band-restricted weak-form drift residual, through :func:`aipf.losses.l_dyn`.

        Shapes: ``rho_hat_states`` ``(B, n_s, n_species, Gx, Gy, Gzr)``, ``lam`` ``(B, n_s)``,
        ``target_hat`` ``(B, n_species, Gx, Gy, Gzr)``, ``boxes`` ``(B, 3)``,
        ``T``/``sample_weight`` ``(B,)``; on a two-dimensional operator set ``(..., Gx, Gyr)`` and
        ``boxes`` ``(B, 2)``."""
        if self.config.sigma is None:
            raise ValueError("drift_loss needs config.sigma (no default)")
        B, n_s = rho_hat_states.shape[:2]
        ops = self._ops_for(rho_hat_states.shape[2:], rho_hat_states.device)
        filt = ops.sigma_filter(boxes, self.config.sigma)          # (B,1,...)
        states = rho_hat_states * filt.unsqueeze(1)
        flat = states.reshape(B * n_s, *states.shape[2:])
        boxes_f = boxes.repeat_interleave(n_s, dim=0)
        T_f = T.repeat_interleave(n_s, dim=0)
        pred_hat = self.model(flat, boxes_f, T_f)
        pred_hat = pred_hat.reshape(B, n_s, *pred_hat.shape[1:])
        lam_b = lam.to(pred_hat.real.dtype).view(B, n_s, *_STATE_WEIGHT_VIEW[ops_ndim(ops)])
        rhs_hat = (lam_b * pred_hat).sum(dim=1)
        residual = target_hat * filt - rhs_hat
        k2 = ops.k2(boxes)
        band = ops.band_mask(boxes, self.resolved_k_max())
        return l_dyn(residual, k2, ops.MULT, band,
                     alpha=self.config.alpha_loss, eps=self.config.h_inv_eps,
                     sample_weight=sample_weight)

    def mobility_loss(self, M_pred: torch.Tensor, M_target: torch.Tensor, *,
                       row_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return l_m(M_pred, M_target, row_weight=row_weight)

    def static_loss(self, H: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor, *,
                     target_rootinv: Optional[torch.Tensor] = None,
                     row_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return l_s(H, target, mask, metric=self.config.stat_metric,
                   target_rootinv=target_rootinv, row_weight=row_weight)

    def bulk_loss(self, H0: torch.Tensor, kBT: torch.Tensor,
                   zvec: torch.Tensor, rho_tot: torch.Tensor,
                   target: torch.Tensor, *,
                   sigma: Optional[torch.Tensor] = None,
                   row_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return l_bulk(H0, kBT, zvec, rho_tot, target,
                      residual=self.config.bulk_residual, sigma=sigma,
                      row_weight=row_weight)

    def pressure_loss(self, P_model: torch.Tensor, *,
                       P_target: Optional[torch.Tensor] = None,
                       group_id: Optional[torch.Tensor] = None,
                       p_ref: Optional[float] = None,
                       p_floor: Optional[float] = None) -> torch.Tensor:
        if P_target is not None:
            return l_p_absolute(P_model, P_target, floor=p_floor)
        if group_id is not None:
            if p_ref is None:
                raise ValueError(
                    "pressure_loss in variance mode (group_id given) needs "
                    "p_ref -- a system constant with no core default")
            return l_p_variance(P_model, group_id, p_ref=p_ref)
        raise ValueError(
            "pressure_loss needs either P_target (absolute mode) or "
            "group_id (variance mode)")

    def compute_losses(
        self, batch: Mapping[str, Any], *,
        conv: Optional[ProbeArg] = None,
        gamma: Optional[ProbeArg] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """The weighted total and every present part, by canonical name.

        ``batch`` groups: ``"drift"``, ``"anchor_M"``, ``"anchor_S"``, ``"anchor_bulk"``,
        ``"anchor_P"``; an absent
        group or a zero weight is never computed; ``conv``/``gamma`` are ``(loss, aux)`` pairs."""
        device = next(self.model.parameters()).device
        total = torch.zeros((), device=device)
        parts: Dict[str, torch.Tensor] = {}

        if "drift" in batch:
            d = batch["drift"]
            drift = self.drift_loss(
                d["rho_hat_states"], d["lam"], d["target_hat"], d["boxes"],
                d["T"], sample_weight=d.get("sample_weight"))
            parts["L_dyn"] = drift
            total = total + self.config.lambda_dyn * drift

        if "anchor_M" in batch and self.config.lambda_M:
            a = batch["anchor_M"]
            lM = self.mobility_loss(a["M_pred"], a["M_target"],
                                     row_weight=a.get("row_weight"))
            parts["L_M"] = lM
            total = total + self.config.lambda_M * lM

        if "anchor_S" in batch and self.config.lambda_S:
            a = batch["anchor_S"]
            lS = self.static_loss(
                a["H"], a["target"], a["mask"],
                target_rootinv=a.get("target_rootinv"),
                row_weight=a.get("row_weight"))
            parts["L_S"] = lS
            total = total + self.config.lambda_S * lS

        if "anchor_bulk" in batch and self.config.lambda_bulk:
            a = batch["anchor_bulk"]
            lB = self.bulk_loss(
                a["H0"], a["kBT"], a["zvec"], a["rho_tot"], a["target"],
                sigma=a.get("sigma"), row_weight=a.get("row_weight"))
            parts["L_bulk"] = lB
            total = total + self.config.lambda_bulk * lB

        if "anchor_P" in batch and self.config.lambda_P:
            a = batch["anchor_P"]
            lP = self.pressure_loss(
                a["P_model"], P_target=a.get("P_target"),
                group_id=a.get("group_id"), p_ref=a.get("p_ref"),
                p_floor=a.get("p_floor"))
            parts["L_P"] = lP
            total = total + self.config.lambda_P * lP

        if conv is not None and self.config.lambda_conv:
            lC, _aux = conv
            parts["L_conv"] = lC
            total = total + self.config.lambda_conv * lC

        if gamma is not None and self.config.lambda_gamma:
            lG, _aux = gamma
            parts["L_Gamma"] = lG
            total = total + self.config.lambda_gamma * lG

        return total, parts

    def training_step(self, batch: Mapping[str, Any], batch_idx: int,
                       *, conv=None, gamma=None) -> torch.Tensor:
        total, parts = self.compute_losses(batch, conv=conv, gamma=gamma)
        self.train_loss(total.detach())
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True,
                 prog_bar=True)
        for name in _CANONICAL_ORDER:
            if name in parts:
                self.log(f"train/{name}", parts[name].detach(),
                         on_step=False, on_epoch=True)
        return total

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        total, _parts = self.compute_losses(batch)
        self.val_loss(total.detach())
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)

    def on_train_epoch_start(self) -> None:
        self.train_loss.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()

    #: Local-form heads decayed at ``wd_ghat`` instead of ``weight_decay``; named, never discovered.
    DECAY_GROUP_HEADS: Tuple[str, ...] = ("g_hat_net", "g_tilde_net")

    #: ``(optimizer state, scheduler state)`` set by :meth:`resume_optimizer_from`; ``None`` builds fresh.
    _resume_states = None

    def parameter_groups(self):
        """Optimizer groups: base at ``weight_decay``, then :data:`DECAY_GROUP_HEADS` at ``wd_ghat``
        if present.

        A model with the heads and no declared ``wd_ghat`` raises."""
        local = getattr(self.model, "f_local", None)
        present, head = [], []
        for name in self.DECAY_GROUP_HEADS:
            net = getattr(local, name, None)
            if net is not None:
                present.append(name)
                head += [p for p in net.parameters() if p.requires_grad]
        head_ids = {id(p) for p in head}
        base = [p for p in self.model.parameters()
                if p.requires_grad and id(p) not in head_ids]
        groups = [{"params": base,
                   "weight_decay": float(self.config.weight_decay)}]
        if head:
            if self.config.wd_ghat is None:
                raise ValueError(
                    f"this model carries the separately decayed head(s) "
                    f"{present} and the config declares no 'wd_ghat', so "
                    f"there is no rate to decay them at. It is a "
                    f"declaration and not a default: every measured system "
                    f"that has these heads declares one, and what it "
                    f"declares is NOT the base 'weight_decay'")
            groups.append({"params": head,
                           "weight_decay": float(self.config.wd_ghat)})
        return groups

    def resume_optimizer_from(self, optimizer_state, scheduler_state) -> None:
        """Continue from a saved optimizer; applied in ``configure_optimizers``, schedule first,
        optimizer last.

        Both states are deep-copied."""
        self._resume_states = (copy.deepcopy(optimizer_state),
                               copy.deepcopy(scheduler_state))

    def configure_optimizers(self):
        opt = AdamW(self.parameter_groups(), lr=float(self.config.lr))
        total = self.trainer.estimated_stepping_batches
        max_epochs = max(self.trainer.max_epochs, 1)
        warm = max(self.config.warmup_epochs * total // max_epochs, 1)
        ann = max(self.config.anneal_epochs * total // max_epochs, 1)
        sched = SequentialLR(
            opt,
            [LinearLR(opt, 1e-6, 1.0, total_iters=warm),
             CosineAnnealingLR(opt, T_max=ann, eta_min=self.config.eta_min)],
            milestones=[warm])
        if self._resume_states is not None:
            optimizer_state, scheduler_state = self._resume_states
            sched.load_state_dict(scheduler_state)
            opt.load_state_dict(optimizer_state)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["model_state_dict"] = self.model.state_dict()
        checkpoint["config"] = self.config.as_dict()
