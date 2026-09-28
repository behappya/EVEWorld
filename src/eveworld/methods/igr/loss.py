"""IGR restoration loss and the EDM noise weighting that scales it.

The restoration term of `eq:pipeline` trains the backbone to rebuild the clean latent `z`
of the demonstration from the count-edited video built by `alg:igr_construction`:

    L_IGR = E[ lambda(sigma) * (1 / (C*T*H*W)) *
               || W_hat^(1/2) * (D_theta(z_tilde + sigma*eps, c, sigma) - z) ||^2_2 ]

`lambda(sigma)` is the EDM weighting `(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`
with `sigma_data = 0.5`, and `W_hat` is the per-cell weight map of `eq:igr_weight_raw`
normalized to unit mean by `eq:igr_weight_norm`. The map is built on the latent grid of
the backbone, so it is routed through
:func:`~eveworld.data.transforms.latent.resize_weight_map`, which keeps the unit mean, and
with it the loss scale, unchanged at every resolution.

Instance-Guided Restoration expresses the term as a mean over the latent cells, so the
unit mean of `W_hat` is the only scale convention the term assumes; `lambda(sigma)` alone
sets the magnitude of the gradient.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from eveworld.data.transforms.latent import latent_grid_size, resize_weight_map

__all__ = ["edm_weight", "igr_loss", "latent_grid_size", "resize_weight_map"]


def edm_weight(sigma: Any, sigma_data: float = 0.5) -> Any:
    """EDM noise weighting `lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`.

    Args:
        sigma: Noise level, a scalar or a tensor of levels such as the `(B,)` levels of a
            batch.
        sigma_data: Standard deviation of the training data, `0.5` in the protocol runs.

    Returns:
        The weighting: a `float` for a scalar `sigma`, a tensor of the same shape as
        `sigma` otherwise.

    Raises:
        ValueError: If `sigma_data` is not positive, or if `sigma` holds a non-finite or
            non-positive level, at which the weighting is infinite.
    """
    data = float(sigma_data)
    if data <= 0.0:
        raise ValueError(f"sigma_data must be positive, got {sigma_data}")
    if torch.is_tensor(sigma):
        minimum = float(sigma.min())
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError(f"sigma must be finite and positive for the EDM weighting, got a minimum of {minimum}")
        return (sigma**2 + data**2) / (sigma * data) ** 2
    level = float(sigma)
    if not math.isfinite(level) or level <= 0.0:
        raise ValueError(f"sigma must be finite and positive for the EDM weighting, got {sigma}")
    return (level**2 + data**2) / (level * data) ** 2


def igr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_map: Any,
    sigma: Any,
    *,
    sigma_data: float = 0.5,
    latent_size: tuple[int, int] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """IGR restoration loss on the latent grid, the `L_IGR` of `eq:pipeline`.

    The prediction is compared with the clean latent of the demonstration under the
    weight map, which is the square of the `W_hat^(1/2)` of the equation, and the mean
    over the latent cells is scaled by the EDM weighting of the sampled noise level.

    Args:
        pred: `(C, T, h, w)` or `(B, C, T, h, w)` denoised latent of the count-edited
            video.
        target: Clean latent of the demonstration, same shape as `pred`.
        weight_map: `(h, w)`, `(T, h, w)`, `(C, T, h, w)` or `(B, C, T, h, w)` map of
            `eq:igr_weight_raw`, `numpy` or `torch`. Lower-rank maps are broadcast over the
            leading dimensions of `pred`; a map whose grid differs from `pred`'s is
            resized with :func:`~eveworld.data.transforms.latent.resize_weight_map`, which
            leaves its unit mean, and so the loss scale, alone.
        sigma: Sampled noise level, a scalar or a `(B,)` tensor matched with `pred`.
        sigma_data: Standard deviation of the training data, `0.5` in the protocol runs.
        latent_size: `(height, width)` of the frames the map was built on, or the latent
            grid itself. Used to catch a map left at the pixel resolution of the clip
            instead of the latent one; `None` trusts the last two dimensions of
            `weight_map`.
        eps: Positive floor applied to `sigma`, which would otherwise weight the term
            infinitely.

    Returns:
        Scalar loss tensor on the device and in the dtype of `pred`, `lambda(sigma)` times
        the weighted mean squared error.

    Raises:
        ValueError: If `pred` and `target` differ in shape, `pred` is not a rank-4 or
            rank-5 latent, the weight map does not broadcast against it, `latent_size`
            names a grid that matches neither `pred` nor its pixel resolution, or `eps` is
            not positive.
    """
    prediction = pred if torch.is_tensor(pred) else torch.as_tensor(pred)
    clean = target if torch.is_tensor(target) else torch.as_tensor(target)
    if prediction.ndim not in (4, 5):
        raise ValueError(f"Expected a (C, T, h, w) or (B, C, T, h, w) latent, got {tuple(prediction.shape)}")
    if tuple(prediction.shape) != tuple(clean.shape):
        raise ValueError(f"pred and target must have the same shape, got {tuple(prediction.shape)} and " f"{tuple(clean.shape)}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    difference = prediction - clean
    weight = _broadcast_weight(_weight_tensor(weight_map, prediction, latent_size), tuple(prediction.shape))
    weighted = weight * difference * difference
    if weighted.ndim == 4:
        reduction = weighted.mean()
    else:
        reduction = weighted.flatten(1).mean(dim=1)
    levels = torch.as_tensor(sigma, dtype=torch.float32, device=prediction.device).clamp_min(eps)
    return (edm_weight(levels, sigma_data) * reduction).mean()


def _weight_tensor(weight_map: Any, pred: torch.Tensor, latent_size: tuple[int, int] | None) -> torch.Tensor:
    """Put `weight_map` on the latent grid and device of `pred` as a tensor."""
    grid = (int(pred.shape[-2]), int(pred.shape[-1]))
    _check_latent_size(latent_size, grid)
    resized = resize_weight_map(weight_map, grid)
    if torch.is_tensor(resized):
        return resized.to(dtype=pred.dtype, device=pred.device)
    return torch.as_tensor(resized, dtype=pred.dtype, device=pred.device)


def _check_latent_size(latent_size: tuple[int, int] | None, grid: tuple[int, int]) -> None:
    """Reject a `latent_size` naming neither the latent grid nor the frames behind it."""
    if latent_size is None:
        return
    try:
        declared = (int(latent_size[0]), int(latent_size[1]))
    except (TypeError, IndexError, KeyError) as error:
        raise ValueError(f"latent_size must be a (height, width) pair, got {latent_size!r}") from error
    if declared == grid:
        return
    if latent_grid_size(declared) == grid:
        return
    raise ValueError(
        f"latent_size {declared} resolves to a {latent_grid_size(declared)} grid, not the {grid} "
        "latent grid of the prediction; pass the latent grid, the frame size it came from, or None"
    )


def _broadcast_weight(weight: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Left-pad a weight map with singleton dimensions so it broadcasts against `shape`."""
    rank = len(shape)
    if weight.ndim < 2:
        raise ValueError(f"Expected at least a (H, W) weight map, got {tuple(weight.shape)}")
    if weight.ndim > rank:
        raise ValueError(
            f"A {weight.ndim}-D weight map cannot weight a {rank}-D latent of shape {shape}; pass the "
            "per-cell map, or its grid through latent_size"
        )
    offset = rank - weight.ndim
    for position in range(weight.ndim):
        map_dim = int(weight.shape[position])
        latent_dim = int(shape[offset + position])
        if map_dim not in (1, latent_dim):
            raise ValueError(
                f"Weight map shape {tuple(weight.shape)} does not broadcast against the latent {shape} "
                f"({map_dim} against {latent_dim} along the trailing dimensions); resize the map with "
                "latent_size"
            )
    return weight.reshape((1,) * offset + tuple(weight.shape))
