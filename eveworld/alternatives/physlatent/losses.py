from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .modules import cfg_get


class PhysicsAuxiliaryLoss:
    """Optional auxiliary losses for physics latent supervision.

    The first stage has no external pseudo labels, so all weights default to
    zero and this loss is a no-op. Once pseudo-label generation is available,
    batches can provide ``phys_state``, ``phys_goal``, ``phys_contact`` and
    ``phys_trajectory`` tensors without changing the trainer. Optional masks
    named ``<key>_mask`` skip low-confidence pseudo labels.
    """

    def __init__(self, weights: Any | None = None) -> None:
        self.state_weight = float(cfg_get(weights, 'state', 0.0))
        self.goal_weight = float(cfg_get(weights, 'goal', 0.0))
        self.contact_weight = float(cfg_get(weights, 'contact', cfg_get(weights, 'progress_stage', 0.0)))
        self.traj_weight = float(cfg_get(weights, 'trajectory', 0.0))
        self.phase_weight = float(cfg_get(weights, 'phase', 0.0))
        self.done_weight = float(cfg_get(weights, 'done', 0.0))
        self.goal_reached_weight = float(cfg_get(weights, 'goal_reached', 0.0))
        self.release_weight = float(cfg_get(weights, 'release', 0.0))
        self.object_motion_weight = float(cfg_get(weights, 'object_motion', 0.0))
        self.terminal_stable_weight = float(cfg_get(weights, 'terminal_stable', 0.0))
        self.quality_threshold = float(cfg_get(weights, 'quality_threshold', 0.6))

    @property
    def enabled(self) -> bool:
        return any(
            weight > 0
            for weight in (
                self.state_weight,
                self.goal_weight,
                self.contact_weight,
                self.traj_weight,
                self.phase_weight,
                self.done_weight,
                self.goal_reached_weight,
                self.release_weight,
                self.object_motion_weight,
                self.terminal_stable_weight,
            )
        )

    def __call__(
        self,
        aux_outputs: dict[str, torch.Tensor],
        batch_dict: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {}

        losses: dict[str, torch.Tensor] = {}

        self._maybe_add_mse(losses, 'state', aux_outputs, batch_dict, 'phys_state', 'phys_state_mask', self.state_weight)
        self._maybe_add_mse(losses, 'goal', aux_outputs, batch_dict, 'phys_goal', 'phys_goal_mask', self.goal_weight)
        self._maybe_add_mse(
            losses,
            'trajectory',
            aux_outputs,
            batch_dict,
            'phys_trajectory',
            'phys_trajectory_mask',
            self.traj_weight,
        )

        if self.contact_weight > 0 and 'phys_contact' in batch_dict and 'contact_logits' in aux_outputs:
            logits = aux_outputs['contact_logits'].float()
            target = batch_dict['phys_contact'].to(logits.device).long()
            if target.ndim == 1 and logits.ndim == 3:
                target = target[:, None].expand(-1, logits.shape[1])
            target = target.reshape(-1)
            logits = logits.reshape(-1, logits.shape[-1])
            mask = self._get_mask(batch_dict, 'phys_contact_mask', logits.device, logits.dtype)
            if mask is not None:
                mask = mask.reshape(-1)
                if mask.numel() != target.numel():
                    mask = None
            value = F.cross_entropy(logits, target, reduction='none')
            value = self._masked_mean(value, mask)
            if value is not None:
                losses['contact'] = value * self.contact_weight

        if self.phase_weight > 0 and 'phys_phase' in batch_dict and 'phase_logits' in aux_outputs:
            logits = aux_outputs['phase_logits'].float()
            target = batch_dict['phys_phase'].to(logits.device).long()
            mask = self._get_mask(batch_dict, 'phys_phase_mask', logits.device, logits.dtype)
            value = self._sequence_cross_entropy(logits, target, mask)
            losses['phase'] = value * self.phase_weight

        if self.done_weight > 0 and 'phys_done_bin' in batch_dict and 'done_logits' in aux_outputs:
            value = self._done_loss(aux_outputs['done_logits'].float(), batch_dict)
            losses['done'] = value * self.done_weight

        self._maybe_add_bce(
            losses,
            'goal_reached',
            aux_outputs,
            batch_dict,
            'goal_reached_logits',
            'phys_goal_reached',
            'phys_goal_reached_mask',
            self.goal_reached_weight,
        )
        self._maybe_add_bce(
            losses,
            'release',
            aux_outputs,
            batch_dict,
            'release_logits',
            'phys_release',
            'phys_release_mask',
            self.release_weight,
        )
        self._maybe_add_mse(
            losses,
            'object_motion',
            aux_outputs,
            batch_dict,
            'phys_object_motion',
            'phys_object_motion_mask',
            self.object_motion_weight,
        )
        if self.terminal_stable_weight > 0 and 'object_motion' in aux_outputs and 'phys_terminal_mask' in batch_dict:
            pred = aux_outputs['object_motion'].float()
            mask = self._terminal_mask(batch_dict, pred.device, pred.dtype)
            value = self._masked_mean(pred.square(), mask)
            if value is None:
                value = pred.sum() * 0.0
            losses['terminal_stable'] = value * self.terminal_stable_weight

        return losses

    def _done_loss(self, logits: torch.Tensor, batch_dict: dict[str, Any]) -> torch.Tensor | None:
        target = batch_dict['phys_done_bin'].to(logits.device).long()
        if target.ndim > 1:
            target = target.view(target.shape[0], -1)[:, 0]
        valid = (target >= 0) & (target < logits.shape[-1])
        if 'phys_done_mask' in batch_dict:
            done_mask = batch_dict['phys_done_mask'].to(logits.device).float().view(-1) > 0
            valid = valid & done_mask
        if 'phys_label_quality' in batch_dict:
            quality = batch_dict['phys_label_quality'].to(logits.device).float().view(-1)
            valid = valid & (quality >= self.quality_threshold)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        value = F.cross_entropy(logits[valid], target[valid], reduction='mean')
        return value

    @staticmethod
    def _sequence_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if target.ndim == 1 and logits.ndim == 3:
            target = target[:, None].expand(-1, logits.shape[1])
        target = target.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])
        if mask is not None:
            mask = mask.reshape(-1)
            if mask.numel() != target.numel():
                mask = None
        value = F.cross_entropy(logits, target, reduction='none')
        reduced = PhysicsAuxiliaryLoss._masked_mean(value, mask)
        if reduced is None:
            return logits.sum() * 0.0
        return reduced

    @staticmethod
    def _maybe_add_bce(
        losses: dict[str, torch.Tensor],
        name: str,
        aux_outputs: dict[str, torch.Tensor],
        batch_dict: dict[str, Any],
        output_key: str,
        batch_key: str,
        mask_key: str,
        weight: float,
    ) -> None:
        if weight <= 0 or batch_key not in batch_dict or output_key not in aux_outputs:
            return
        logits = aux_outputs[output_key].float()
        target = batch_dict[batch_key].to(device=logits.device, dtype=logits.dtype)
        if target.shape != logits.shape:
            target = target.view_as(logits)
        mask = PhysicsAuxiliaryLoss._get_mask(batch_dict, mask_key, logits.device, logits.dtype)
        value = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        value = PhysicsAuxiliaryLoss._masked_mean(value, mask)
        if value is None:
            value = logits.sum() * 0.0
        losses[name] = value * weight

    def _terminal_mask(
        self,
        batch_dict: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        mask = self._get_mask(batch_dict, 'phys_terminal_mask', device, dtype)
        if mask is None:
            return None
        if 'phys_label_quality' in batch_dict:
            quality = batch_dict['phys_label_quality'].to(device=device, dtype=dtype)
            while quality.ndim < mask.ndim:
                quality = quality.unsqueeze(-1)
            mask = mask * (quality >= self.quality_threshold).to(dtype)
        return mask

    @staticmethod
    def _maybe_add_mse(
        losses: dict[str, torch.Tensor],
        name: str,
        aux_outputs: dict[str, torch.Tensor],
        batch_dict: dict[str, Any],
        batch_key: str,
        mask_key: str,
        weight: float,
    ) -> None:
        if weight <= 0 or batch_key not in batch_dict or name not in aux_outputs:
            return
        pred = aux_outputs[name].float()
        target = batch_dict[batch_key].to(device=pred.device, dtype=pred.dtype)
        if target.shape != pred.shape:
            target = target.view_as(pred)
        value = (pred - target).square()
        while value.ndim > 2:
            value = value.mean(dim=-1)
        if value.ndim == 2 and pred.shape[-1] != 1 and name in {'state', 'goal'}:
            value = value.mean(dim=-1)
        mask = PhysicsAuxiliaryLoss._get_mask(batch_dict, mask_key, pred.device, pred.dtype)
        value = PhysicsAuxiliaryLoss._masked_mean(value, mask)
        if value is not None:
            losses[name] = value * weight

    @staticmethod
    def _get_mask(
        batch_dict: dict[str, Any],
        mask_key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if mask_key in batch_dict:
            return batch_dict[mask_key].to(device=device, dtype=dtype)
        if 'phys_valid_mask' in batch_dict and mask_key == 'phys_trajectory_mask':
            return batch_dict['phys_valid_mask'].to(device=device, dtype=dtype)
        return None

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return value.mean()
        mask = mask.to(device=value.device, dtype=value.dtype)
        while mask.ndim > value.ndim:
            mask = mask.squeeze(-1)
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        if mask.shape != value.shape:
            try:
                mask = mask.expand_as(value)
            except RuntimeError:
                return value.mean()
        denom = mask.sum()
        if denom <= 0:
            return None
        return (value * mask).sum() / denom
