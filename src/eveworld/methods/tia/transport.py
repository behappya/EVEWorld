"""Local softmax window matching and the transport operator (paper Algorithm 3).

The transport is a **local softmax window matcher with mass normalisation**: for every source
token the similarity against its ``window x window`` neighbourhood in the target grid is divided
by the temperature and softmaxed over that neighbourhood, and the target values are splatted back
onto the source grid and divided by the accumulated mass. There is no Sinkhorn iteration anywhere
in this codebase and none is needed - the correspondence is a single softmax over a small local
window.
"""

from __future__ import annotations

from typing import Sequence

import torch

from eveworld.methods.tia.matcher import local_window_indices, normalize_features, token_grid_shape

__all__ = ["first_frame_unchanged", "transport", "transport_weights"]

_MASS_EPS = 1e-6
_WINDOW_CACHE: dict[tuple[int, int, int, str], torch.Tensor] = {}


def _window_indices(height: int, width: int, radius: int, device: torch.device) -> torch.Tensor:
    """Cached int64 ``(H * W, k * k)`` window indices for one grid and device."""
    key = (int(height), int(width), int(radius), str(device))
    indices = _WINDOW_CACHE.get(key)
    if indices is None:
        indices = torch.as_tensor(local_window_indices(height, width, radius), dtype=torch.int64, device=device)
        _WINDOW_CACHE[key] = indices
    return indices


def _window_scores(
    source: torch.Tensor,
    target: torch.Tensor,
    window: int = 7,
    grid: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Raw cosine logits of every source token against its window in the target grid.

    Args:
        source: Float tensor ``(B, N, C)`` or ``(N, C)``; defines the output grid and the queries.
        target: Float tensor ``(B, N, C)`` or ``(N, C)`` with the same token count; defines the
            candidate window of each query.
        window: Odd window size ``k``; ``k = 7`` is the paper's 7 x 7 neighbourhood.
        grid: Optional ``(height, width)`` of the token grid, used to validate the token count.

    Returns:
        tuple: ``(logits, indices, unbatched)`` with ``logits`` of shape ``(B, N, k * k)``, the
        int64 window ``indices`` of shape ``(N, k * k)`` used to gather them, and whether the
        inputs were unbatched.

    Raises:
        ValueError: If the inputs are not 2-D or 3-D, if their shapes disagree, if ``window`` is
        not a positive odd number or if ``grid`` does not match the token count.
    """
    unbatched = source.ndim == 2 and target.ndim == 2
    if unbatched:
        source = source.unsqueeze(0)
        target = target.unsqueeze(0)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError(f"expected (B, N, C) or (N, C) inputs, got {tuple(source.shape)} and {tuple(target.shape)}")
    if source.shape != target.shape:
        raise ValueError(f"source and target must share shape, got {tuple(source.shape)} and {tuple(target.shape)}")
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd size, got {window}")

    source = source.float()
    target = target.float()
    batch, num_tokens, channels = source.shape
    height, width = token_grid_shape(num_tokens, grid)
    indices = _window_indices(height, width, (window - 1) // 2, source.device)
    gathered = target.index_select(1, indices.reshape(-1)).reshape(batch, num_tokens, indices.shape[1], channels)
    logits = (gathered * normalize_features(source, dim=-1).unsqueeze(2)).sum(dim=-1)
    return logits, indices, unbatched


def transport_weights(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    window: int = 7,
    temperature: float = 0.07,
    grid: Sequence[int] | None = None,
) -> torch.Tensor:
    """Local softmax correspondence weights between two token grids.

    For every source token the weights are a softmax, at ``temperature``, over the similarities
    against its ``window x window`` neighbourhood in the target grid, and zero everywhere else.
    Neighbours clamped to the grid border land on the same target token and therefore accumulate.

    Args:
        source: Float tensor ``(B, N, C)`` or ``(N, C)`` of source features.
        target: Float tensor with the same shape, holding the candidate features.
        window: Odd window size ``k``; ``k = 7`` is the paper's 7 x 7 neighbourhood.
        temperature: Softmax temperature ``tau`` (the paper uses 0.07).
        grid: Optional ``(height, width)`` of the token grid.

    Returns:
        torch.Tensor: ``(B, N, N)`` weights (``(N, N)`` for unbatched inputs) in a floating dtype
        of at least float32 precision; each row sums to one over the local window.

    Raises:
        ValueError: If the inputs are mis-shaped or ``temperature`` is not positive.
    """
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    source = torch.as_tensor(source)
    target = torch.as_tensor(target)
    logits, indices, unbatched = _window_scores(source, target, window=window, grid=grid)
    weights = torch.softmax(logits / temperature, dim=-1)
    batch, num_tokens = weights.shape[0], weights.shape[1]
    out = torch.zeros(batch, num_tokens, num_tokens, dtype=weights.dtype, device=weights.device)
    out.scatter_add_(2, indices.unsqueeze(0).expand(batch, -1, -1), weights)
    return out[0] if unbatched else out


def transport(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    window: int = 7,
    temperature: float = 0.07,
    normalize_mass: bool = True,
    grid: Sequence[int] | None = None,
) -> torch.Tensor:
    """Splat target features onto the source grid with the local window weights.

    The values of the target tokens are carried along the correspondence weights of
    :func:`transport_weights` and, when ``normalize_mass`` is set, divided by the accumulated weight
    mass of each source token. With the softmax weights of this codebase the mass is already one, so
    the division only guards against unnormalised or masked weights.

    Args:
        source: Float tensor ``(B, N, C)`` or ``(N, C)`` of source features; defines the output grid.
        target: Float tensor with the same shape, holding the values that are transported.
        window: Odd window size ``k``; ``k = 7`` is the paper's 7 x 7 neighbourhood.
        temperature: Softmax temperature ``tau`` (the paper uses 0.07).
        normalize_mass: Divide each source token by the weight mass it accumulated.
        grid: Optional ``(height, width)`` of the token grid.

    Returns:
        torch.Tensor: ``(B, N, C)`` (``(N, C)`` for unbatched inputs) transported target features in
        the dtype of ``target``.

    Raises:
        ValueError: If the inputs are mis-shaped or ``temperature`` is not positive.
    """
    target = torch.as_tensor(target)
    weights = transport_weights(source, target, window=window, temperature=temperature, grid=grid)
    values = target.float()
    unbatched = values.ndim == 2
    if unbatched:
        values = values.unsqueeze(0)
    transported = torch.matmul(weights, values)
    if normalize_mass:
        mass = weights.sum(dim=-1, keepdim=True).clamp_min(_MASS_EPS)
        transported = transported / mass
    transported = transported.to(dtype=target.dtype)
    return transported[0] if unbatched else transported


def first_frame_unchanged(
    source: torch.Tensor,
    target: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Whether the temporal index 0 slice of ``out`` was left untouched.

    Frames are located on the shape convention of the caller: index 2 of a ``(B, C, T, H, W)``
    tensor, index 1 of a ``(B, T, N, C)`` tensor. A ``(B, T * N, C)`` tensor has no visible frame
    boundary and is treated as ``T == 1``, so the whole tensor is compared. The check passes when
    the first frame of ``out`` is bit-identical to that of ``source`` or of ``target``, i.e. it was
    copied from one of the two frames that entered the operator.

    Args:
        source: Reference tensor, ``(B, C, T, H, W)``, ``(B, T, N, C)`` or ``(B, T * N, C)``.
        target: Second reference tensor; compared when it has the same shape as ``source``.
        out: Operator output, with the same shape as ``source``.

    Returns:
        bool: ``True`` when the first frame was preserved.

    Raises:
        ValueError: If ``out`` does not match ``source`` or the tensors have an unsupported rank.
    """
    source = torch.as_tensor(source)
    target = torch.as_tensor(target)
    out = torch.as_tensor(out)
    if out.shape != source.shape:
        raise ValueError(f"out {tuple(out.shape)} must match source {tuple(source.shape)}")
    if source.ndim == 5:
        axis = 2
    elif source.ndim == 4:
        axis = 1
    elif source.ndim == 3:
        axis = None
    else:
        raise ValueError(f"expected 3-D, 4-D or 5-D tensors, got {source.ndim}-D")
    if axis is not None:
        if source.shape[axis] == 0:
            return False
        head = out.select(axis, 0)
        preserved = torch.equal(head, source.select(axis, 0))
        if target.shape == source.shape:
            preserved = preserved or torch.equal(head, target.select(axis, 0))
        return bool(preserved)
    preserved = torch.equal(out, source)
    if target.shape == source.shape:
        preserved = preserved or torch.equal(out, target)
    return bool(preserved)
