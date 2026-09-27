"""Training objective of the TIA stage and its two schedules.

The contrastive term pulls every tracked token towards the token it corresponds to in the
neighbouring frame. The candidates are the same local window that
:mod:`eveworld.methods.tia.transport` uses for the transport, so the term only ever competes
within a neighbourhood and cannot be satisfied by a globally uniform attention. The schedules keep
the term out of the first training steps and out of the near-clean diffusion steps, where the paper
found it to be noise.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from eveworld.methods.tia.transport import _window_scores

__all__ = ["contrastive_loss", "lambda_schedule", "noise_gate", "windowed_scores"]


def windowed_scores(
    source: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
    window: int = 7,
    grid: Sequence[int] | None = None,
) -> torch.Tensor:
    """Correspondence logits between two token grids, as consumed by :func:`contrastive_loss`.

    The scores are the cosine similarities of :func:`eveworld.methods.tia.transport.transport_weights`
    *before* the softmax: entry ``(b, i, j)`` compares the normalised source token ``i`` with the
    normalised target token ``j``, and it is ``-inf`` whenever ``j`` lies outside the
    ``window x window`` neighbourhood of ``i``. Unlike the transport weights, which accumulate the
    softmax mass of a neighbour that the clamped border window repeats, the scores hold a single
    comparable logit per candidate token, so the term competes over the distinct tokens of the
    window - the candidate set of the paper's own probe formulation. They are deliberately returned
    unscaled, so ``windowed_scores(source, target, temperature) / temperature`` is what
    :func:`contrastive_loss` consumes: the temperature is applied once, by the loss. In the interior
    of the grid, where the window repeats no neighbour, ``softmax(windowed_scores(s, t, tau) / tau)``
    reproduces ``transport_weights(s, t, temperature=tau)`` exactly.

    Args:
        source: Float tensor ``(B, N, C)`` or ``(N, C)``; defines the output grid and the queries.
        target: Float tensor with the same shape, holding the candidate tokens.
        temperature: Softmax temperature ``tau`` that :func:`contrastive_loss` will apply (the paper
            uses 0.07); validated here so that a bad scale fails at the call site.
        window: Odd window size ``k``; ``k = 7`` is the paper's 7 x 7 neighbourhood.
        grid: Optional ``(height, width)`` of the token grid, used to validate the token count.

    Returns:
        torch.Tensor: ``(B, N, N)`` float32 logits (``(N, N)`` for unbatched inputs) with ``-inf``
        outside the local window of each source token.

    Raises:
        ValueError: If the inputs are mis-shaped, if ``window`` is not a positive odd number, if
        ``grid`` does not match the token count or if ``temperature`` is not a positive finite
        number.
    """
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be a positive finite number, got {temperature}")
    source = torch.as_tensor(source)
    target = torch.as_tensor(target)
    logits, indices, unbatched = _window_scores(source, target, window=window, grid=grid)
    batch, num_tokens = logits.shape[0], logits.shape[1]
    scores = logits.new_full((batch, num_tokens, num_tokens), float("-inf"))
    scores.scatter_(2, indices.unsqueeze(0).expand(batch, -1, -1), logits)
    return scores[0] if unbatched else scores


def contrastive_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """The paper's TIA term ``L_TIA = -1/|V| * sum_t log(exp(s_{t,u_t}/tau) / sum_v exp(s_{t,v}/tau))``.

    Args:
        scores: Float logits ``(T, M)`` (any shape with the candidates last); entries set to ``-inf``
            are outside the tracking window and can never be selected.
        targets: int64 positive indices with the shape of ``scores`` without its last dimension.
            Entries ``< 0`` (tokens that have no correspondence, e.g. occluded ones) are ignored.
        temperature: Softmax temperature ``tau`` (the paper uses 0.07).

    Returns:
        torch.Tensor: Scalar loss tensor. Entries whose target is negative are excluded from the
        mean; when every entry is excluded, or when there is no entry at all, the result is a zero
        tensor rather than ``NaN``.

    Raises:
        ValueError: If ``scores`` has fewer than two dimensions, if ``targets`` does not match the
        leading shape of ``scores`` or if ``temperature`` is not positive.
    """
    scores = torch.as_tensor(scores)
    targets = torch.as_tensor(targets)
    if scores.ndim < 2:
        raise ValueError(f"scores must be at least 2-D (..., M), got {tuple(scores.shape)}")
    expected = tuple(scores.shape[:-1])
    if tuple(targets.shape) != expected:
        raise ValueError(f"targets {tuple(targets.shape)} must match the leading shape {expected} of scores")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    targets = targets.to(torch.int64)
    valid = targets >= 0
    if scores.numel() == 0 or not bool(valid.any()):
        return scores.new_zeros(())
    targets = torch.where(valid, targets, torch.full_like(targets, -1))
    flat_scores = scores.reshape(-1, scores.shape[-1])
    flat_targets = targets.reshape(-1)
    return F.cross_entropy(flat_scores / temperature, flat_targets, ignore_index=-1)


def lambda_schedule(step: int, *, warmup: int = 20, target: float = 0.5) -> float:
    """Linear warm-up of the TIA weight: ``0`` at step 0 and ``target`` from ``warmup`` on.

    Args:
        step: Optimiser step; negative values are clamped to 0.
        warmup: Number of steps the ramp takes; a non-positive value jumps straight to ``target``.
        target: Weight reached at the end of the ramp (the paper uses 0.5).

    Returns:
        float: The weight of the TIA term at ``step``.
    """
    warmup = int(warmup)
    target = float(target)
    if warmup <= 0:
        return target
    progress = max(0.0, float(step)) / warmup
    return target * min(1.0, progress)


def noise_gate(
    sigma: float | torch.Tensor | np.ndarray,
    *,
    low: float = 0.2,
    high: float = 0.5,
) -> bool | torch.Tensor | np.ndarray:
    """Whether a diffusion noise level lies inside the band where the TIA term is applied.

    The paper found the term to be harmful for very low (near-clean) and very high (pure noise)
    sigmas, so it is only switched on for ``low <= sigma <= high``. Both bounds are inclusive.

    Args:
        sigma: Noise level, as a Python number, a 0-dim tensor or an array of levels.
        low: Lower bound of the band (the paper uses 0.2).
        high: Upper bound of the band (the paper uses 0.5).

    Returns:
        bool | torch.Tensor | np.ndarray: A ``bool`` for scalar input, otherwise a boolean tensor or
        array of the same shape.

    Raises:
        ValueError: If ``low`` is greater than ``high``.
    """
    low = float(low)
    high = float(high)
    if low > high:
        raise ValueError(f"noise gate bounds must satisfy low <= high, got low={low}, high={high}")
    if isinstance(sigma, torch.Tensor):
        gate = (sigma >= low) & (sigma <= high)
        return bool(gate) if sigma.ndim == 0 else gate
    if isinstance(sigma, np.ndarray):
        gate = (sigma >= low) & (sigma <= high)
        return bool(gate) if gate.ndim == 0 else gate
    return low <= float(sigma) <= high
