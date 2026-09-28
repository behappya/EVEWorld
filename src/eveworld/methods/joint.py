"""The joint training objective: IGR restoration plus the weighted TIA term.

`docs/training.md` trains the released backbones with

    L = L_IGR + lambda_TIA * L_TIA

where `L_IGR` is the restoration term of `eq:pipeline`
(:func:`eveworld.methods.igr.loss.igr_loss`) and `L_TIA` is the transport-consistency term
computed on the post-transport features. Only the schedule and the noise gate of
`lambda_TIA` live here: on GigaWorld-0 the weight warms up linearly from 0 to 0.5 over the
first 20 steps, and the gate restricts the term to sampled noise levels `sigma in [0.2,
0.5]`, where the transport is informative but the input is not yet clean; FlowWAM keeps
`lambda_TIA = 0.1` with no gate. The released values are `method.warmup_steps`,
`method.sigma_low`, `method.sigma_high` and `method.gated`.

The TIA term is passed in by the trainer rather than computed here, so importing this
module does not pull in the transport stack.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

__all__ = ["LAMBDA_IGR", "JointConfig", "JointObjective", "combine", "warmup_weight"]

LAMBDA_IGR = 1.0
"""Fixed weight of the IGR restoration term, `L = L_IGR + lambda_TIA * L_TIA`."""


@dataclass
class JointConfig:
    """Weights, schedule and gate of the joint objective.

    Mirrors the `method` block of the training configurations, e.g.
    `configs/paper/gigaworld/dreamgen/eveworld.yaml`.

    Attributes:
        p_dup: Probability that a sample is count-edited rather than relocated, the
            `method.p_dup` of `alg:igr_construction`, `0.5` in the released runs.
        lambda_tia: Weight of the TIA term at full warm-up, `0.5` on GigaWorld-0 and `0.1`
            on FlowWAM.
        warmup_steps: Steps over which `lambda_tia` warms up linearly from 0, `20` on
            GigaWorld-0 and `0` for a constant weight.
        sigma_low: Lower end of the noise-gate window, inclusive.
        sigma_high: Upper end of the noise-gate window, inclusive.
        gated: Whether the noise gate is applied at all. `False` for FlowWAM and for the
            single-term arms.

    Raises:
        ValueError: If a probability, weight or window is out of range.
    """

    p_dup: float = 0.5
    lambda_tia: float = 0.5
    warmup_steps: int = 20
    sigma_low: float = 0.2
    sigma_high: float = 0.5
    gated: bool = True

    def __post_init__(self) -> None:
        self.p_dup = float(self.p_dup)
        self.lambda_tia = float(self.lambda_tia)
        self.warmup_steps = int(self.warmup_steps)
        self.sigma_low = float(self.sigma_low)
        self.sigma_high = float(self.sigma_high)
        self.gated = bool(self.gated)
        if not 0.0 <= self.p_dup <= 1.0:
            raise ValueError(f"p_dup must be a probability, got {self.p_dup}")
        if self.lambda_tia < 0.0:
            raise ValueError(f"lambda_tia must not be negative, got {self.lambda_tia}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must not be negative, got {self.warmup_steps}")
        if self.sigma_low > self.sigma_high:
            raise ValueError(f"sigma_low must not exceed sigma_high, got {self.sigma_low} and {self.sigma_high}")

    @classmethod
    def from_mapping(cls, mapping: Any) -> JointConfig:
        """Build the config from a `method` block.

        The block also carries the keys of the other terms (`kind`, `layer`, `window`,
        `temperature`, `gamma`), which belong to their own configurations and are ignored
        here.

        Args:
            mapping: Parsed `method` block of a training configuration.

        Returns:
            The joint configuration named by the block, defaulted per missing key.
        """
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in dict(mapping).items() if key in known})


def combine(
    igr_loss: torch.Tensor,
    tia_loss: torch.Tensor | None,
    *,
    lambda_tia: float,
    step: int,
    warmup_steps: int = 20,
    sigma: Any = None,
    sigma_low: float = 0.2,
    sigma_high: float = 0.5,
    gated: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine the two terms into `L = L_IGR + lambda_TIA * L_TIA`.

    The TIA weight is first warmed up and then passed through the noise gate: it is
    `lambda_tia` scaled by `min(1, step / warmup_steps)`, zeroed unless every sampled level
    lies within the window `[sigma_low, sigma_high]`.

    Args:
        igr_loss: Scalar restoration loss of `eq:pipeline`, with the fixed weight
            :data:`LAMBDA_IGR`.
        tia_loss: Scalar transport-consistency loss, or `None` for a run that trains the
            restoration term alone. `None` contributes a zero tensor, so the returned total
            keeps the dtype and device of `igr_loss`.
        lambda_tia: Weight of the TIA term at full warm-up, e.g. `0.5` on GigaWorld-0 and
            `0.1` on FlowWAM.
        step: Current optimisation step, counted from 0.
        warmup_steps: Steps over which `lambda_tia` warms up linearly from 0; at most `0`
            uses the full weight from the first step.
        sigma: Sampled noise level of the batch, a scalar or a tensor. `None` - no level in
            hand - leaves the gate open.
        sigma_low: Lower end of the noise-gate window, inclusive, `0.2` on GigaWorld-0.
        sigma_high: Upper end of the noise-gate window, inclusive, `0.5` on GigaWorld-0.
        gated: Whether to apply the gate. Pass `False` for a run without one, as on FlowWAM,
            and for a clip whose target could not be tracked, which is trained on the clean
            video with a unit weight map.

    Returns:
        The total loss and a statistics mapping with the keys `lambda_igr`, `lambda_tia`
        (the effective weight, zero while the gate is closed), `lambda_tia_raw` (the warm-up
        value before the gate), `step`, `gated` (whether the gate admitted the term) and
        `sigma` (the mean sampled level, or `None`).

    Raises:
        ValueError: If `sigma_low` exceeds `sigma_high` while the gate is enabled.
    """
    if gated and float(sigma_low) > float(sigma_high):
        raise ValueError(f"sigma_low must not exceed sigma_high, got {sigma_low} and {sigma_high}")
    raw_weight = warmup_weight(step, warmup_steps)
    admitted = _gate_admitted(sigma, sigma_low, sigma_high, gated)
    effective = raw_weight * lambda_tia if admitted else 0.0
    if tia_loss is None:
        tia_term = torch.zeros((), dtype=igr_loss.dtype, device=igr_loss.device)
    else:
        tia_term = tia_loss
    total = LAMBDA_IGR * igr_loss + effective * tia_term
    stats: dict[str, Any] = {
        "lambda_igr": float(LAMBDA_IGR),
        "lambda_tia": float(effective),
        "lambda_tia_raw": float(raw_weight) * float(lambda_tia),
        "step": int(step),
        "gated": admitted,
        "sigma": None if sigma is None else _mean_level(sigma),
    }
    return total, stats


def warmup_weight(step: int, warmup_steps: int) -> float:
    """Linear warm-up factor of the TIA weight, `min(1, step / warmup_steps)`.

    Args:
        step: Current optimisation step, counted from 0.
        warmup_steps: Steps the warm-up spans; at most `0` means no warm-up.

    Returns:
        A factor in `[0, 1]`, `1.0` once the warm-up is over.
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(step) / float(warmup_steps)))


class JointObjective(torch.nn.Module):
    """The joint objective with the schedule and the running statistics of a run.

    The module carries no parameters: the backbone is trained through the returned total
    loss, and the module only decides how much of the TIA term enters it and keeps the
    means a trainer logs.

    Args:
        config: Weights, schedule and gate to use; defaults to the released GigaWorld-0
            values.

    Attributes:
        config: The :class:`JointConfig` in use.
    """

    def __init__(self, config: JointConfig | None = None) -> None:
        super().__init__()
        self.config = JointConfig() if config is None else config
        self._running: dict[str, float] = {"igr": 0.0, "tia": 0.0, "total": 0.0, "steps": 0.0}

    def forward(
        self,
        igr: torch.Tensor,
        tia: torch.Tensor | None = None,
        step: int = 0,
        sigma: Any = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Combine one batch of the two terms and fold it into the running means.

        Args:
            igr: Scalar restoration loss of the batch.
            tia: Scalar transport-consistency loss of the batch, or `None` for a run that
                trains the restoration term alone.
            step: Current optimisation step, counted from 0.
            sigma: Sampled noise level of the batch, a scalar or a tensor.

        Returns:
            The total loss and the statistics of :func:`combine` extended with the batch
            means `igr`, `tia` and `total` and with `running`, the means since the last
            :meth:`reset_running_mean`.
        """
        total, stats = combine(
            igr,
            tia,
            lambda_tia=self.config.lambda_tia,
            step=step,
            warmup_steps=self.config.warmup_steps,
            sigma=sigma,
            sigma_low=self.config.sigma_low,
            sigma_high=self.config.sigma_high,
            gated=self.config.gated,
        )
        batch = {
            "igr": float(igr.detach().mean()),
            "tia": 0.0 if tia is None else float(tia.detach().mean()),
            "total": float(total.detach().mean()),
        }
        stats.update(batch)
        for key, value in batch.items():
            self._running[key] += value
        self._running["steps"] += 1.0
        stats["running"] = self.running_mean()
        return total, stats

    def running_mean(self) -> dict[str, float]:
        """Mean of the three terms over the batches since the last reset.

        Returns:
            Mapping with the keys `igr`, `tia`, `total` and `steps`, all zero before the
            first batch.
        """
        steps = self._running["steps"]
        if steps <= 0.0:
            return {"igr": 0.0, "tia": 0.0, "total": 0.0, "steps": 0}
        return {
            "igr": self._running["igr"] / steps,
            "tia": self._running["tia"] / steps,
            "total": self._running["total"] / steps,
            "steps": int(steps),
        }

    def reset_running_mean(self) -> None:
        """Forget the batches accumulated so far."""
        for key in self._running:
            self._running[key] = 0.0


def _gate_admitted(sigma: Any, sigma_low: float, sigma_high: float, gated: bool) -> bool:
    """Whether the sampled noise levels pass the gate of `sigma in [sigma_low, sigma_high]`."""
    if not gated or sigma is None:
        return True
    levels = torch.as_tensor(sigma, dtype=torch.float32)
    if levels.numel() == 0:
        return True
    return bool(((levels >= float(sigma_low)) & (levels <= float(sigma_high))).all())


def _mean_level(sigma: Any) -> float:
    """Mean of the sampled noise levels, for the statistics of a step."""
    levels = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma, dtype=torch.float32)
    return float(levels.detach().mean())
