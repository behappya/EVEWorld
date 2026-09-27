"""Local-window correspondence on l2-normalised token grids.

Every token of a flattened grid is matched against its ``window x window`` neighbourhood in
another grid; neighbours that fall outside the grid are clamped to the border cell, which keeps
the index tensor shape static and makes the operator well defined at the borders. The geometry
defined here is shared by the transport operator, the adaptation adapter and the offline layer
probe.

Token grids are flattened in row-major order, so the cell ``(row, col)`` of a ``height x width``
grid maps to the flat token index ``row * width + col``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

__all__ = [
    "ProbeResult",
    "correlation_matrix",
    "local_window_indices",
    "normalize_features",
    "retrieve",
    "select_layer",
    "token_grid_shape",
]


def local_window_indices(height: int, width: int, radius: int = 3) -> np.ndarray:
    """Flat neighbourhood indices of every cell of a token grid.

    For each of the ``height * width`` cells this yields the flat indices of the ``k x k`` window
    centred on it, with ``k = 2 * radius + 1`` (the paper uses ``radius=3``, i.e. a 7 x 7 window).
    Neighbours outside the grid are clamped to the border cell, so corner and edge cells repeat
    their outermost neighbours instead of dropping them. The window is ordered row-major: the row
    offset varies slowest.

    Args:
        height: Grid height in tokens.
        width: Grid width in tokens.
        radius: Window radius in cells; ``0`` selects the cell itself.

    Returns:
        np.ndarray: ``(height * width, k * k)`` int64 indices into the flattened target grid.

    Raises:
        ValueError: If ``height`` or ``width`` is not positive or ``radius`` is negative.
    """
    height = int(height)
    width = int(width)
    radius = int(radius)
    if height <= 0 or width <= 0:
        raise ValueError(f"grid dimensions must be positive, got height={height}, width={width}")
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    rows = np.arange(height, dtype=np.int64)[:, None] + offsets[None, :]
    cols = np.arange(width, dtype=np.int64)[:, None] + offsets[None, :]
    rows = np.clip(rows, 0, height - 1)[:, None, :, None]
    cols = np.clip(cols, 0, width - 1)[None, :, None, :]
    indices = (rows * width + cols).reshape(height * width, -1)
    return indices.astype(np.int64, copy=False)


def token_grid_shape(num_tokens: int, grid: Sequence[int] | None = None) -> tuple[int, int]:
    """Resolve the ``(height, width)`` layout of a flat token grid.

    Args:
        num_tokens: Number of tokens in the flattened grid.
        grid: Optional explicit ``(height, width)``; validated against ``num_tokens`` when given.

    Returns:
        tuple[int, int]: ``(height, width)`` with ``height <= width``. Without an explicit grid the
        factorisation closest to square is used, which recovers the token grids of this codebase
        (300 -> 15 x 20, 720 -> 24 x 30, 1200 -> 30 x 40).

    Raises:
        ValueError: If ``num_tokens`` is not positive or ``grid`` does not match ``num_tokens``.
    """
    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if grid is not None:
        if len(grid) != 2:
            raise ValueError(f"grid must be a (height, width) pair, got {grid!r}")
        height, width = int(grid[0]), int(grid[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"grid dimensions must be positive, got {grid!r}")
        if height * width != num_tokens:
            raise ValueError(f"grid {grid!r} does not match {num_tokens} tokens")
        return height, width
    for height in range(math.isqrt(num_tokens), 0, -1):
        if num_tokens % height == 0:
            return height, num_tokens // height
    return 1, num_tokens


def normalize_features(features: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise features along ``dim`` (the paper's matching space).

    Args:
        features: Float tensor of shape ``(..., C)``.
        dim: Dimension holding the feature vector.
        eps: Lower bound on the norm, so all-zero rows stay finite.

    Returns:
        torch.Tensor: Tensor with the same shape as ``features`` and unit-norm vectors along
        ``dim``.
    """
    features = torch.as_tensor(features)
    norm = features.norm(p=2, dim=dim, keepdim=True)
    return features / norm.clamp_min(float(eps))


def correlation_matrix(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between every source and target token.

    Args:
        source: Float tensor ``(B, N_src, C)`` or ``(N_src, C)``.
        target: Float tensor ``(B, N_tgt, C)`` or ``(N_tgt, C)``.

    Returns:
        torch.Tensor: ``(B, N_src, N_tgt)`` cosine similarities, or ``(N_src, N_tgt)`` when both
        inputs are unbatched.

    Raises:
        ValueError: If the inputs are not 2-D or 3-D, if their batch sizes differ or if their
        feature dimensions differ.
    """
    source = torch.as_tensor(source)
    target = torch.as_tensor(target)
    unbatched = source.ndim == 2 and target.ndim == 2
    if unbatched:
        source = source.unsqueeze(0)
        target = target.unsqueeze(0)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError(f"expected (B, N, C) or (N, C) inputs, got {tuple(source.shape)} and {tuple(target.shape)}")
    if source.shape[0] != target.shape[0] or source.shape[-1] != target.shape[-1]:
        raise ValueError(f"source {tuple(source.shape)} and target {tuple(target.shape)} must share batch and channels")
    similarity = normalize_features(source) @ normalize_features(target).transpose(-1, -2)
    return similarity[0] if unbatched else similarity


@dataclass
class ProbeResult:
    """End-point error of one candidate block in the offline layer probe."""

    layer: int
    epe: float

    def to_dict(self) -> dict[str, int | float]:
        """Return the record as a JSON-serialisable mapping."""
        return {"layer": int(self.layer), "epe": float(self.epe)}


def select_layer(candidates: Sequence[int], epes: Sequence[float]) -> ProbeResult:
    """Pick the candidate block with the lowest end-point error.

    Args:
        candidates: Block indices, aligned with ``epes``.
        epes: Mean end-point error of each candidate in grid cells; non-finite entries (blocks
            that could not be scored) are skipped.

    Returns:
        ProbeResult: The winning block and its error.

    Raises:
        ValueError: If the sequences are empty, their lengths differ, or no candidate has a finite
        error.
    """
    candidates = list(candidates)
    epes = list(epes)
    if len(candidates) != len(epes):
        raise ValueError(f"candidates and epes must align, got {len(candidates)} and {len(epes)}")
    if not candidates:
        raise ValueError("no candidate layers to select from")
    best: tuple[int, float] | None = None
    for layer, epe in zip(candidates, epes):
        value = float(epe)
        if not math.isfinite(value):
            continue
        if best is None or value < best[1]:
            best = (int(layer), value)
    if best is None:
        raise ValueError("every candidate layer has a non-finite end-point error")
    return ProbeResult(layer=best[0], epe=best[1])


def retrieve(
    features: torch.Tensor,
    coords: torch.Tensor,
    query_coords: torch.Tensor,
    grid: Sequence[int] | None = None,
) -> torch.Tensor:
    """Nearest-neighbour retrieval on the raw token grid (used by the offline probe).

    Each query token is taken from frame ``t`` at the cell given by ``query_coords[t]`` and matched
    against **every** token of frame ``t + 1`` (no window restriction, no transport weights); the
    cell of the best match is returned. Query cells that are not finite, or that do not appear in
    ``coords``, fall back to the first token of the grid.

    Args:
        features: Float tensor ``(T, N, C)`` or ``(N, C)`` with T >= 2 frames of token features.
        coords: Cell coordinates of every token, ``(T, N, 2)`` or ``(N, 2)``, broadcast over
            frames, as ``(row, col)`` float or integer values. This defines the token layout that
            ``query_coords`` is resolved against.
        query_coords: Query cells, ``(T, 2)`` or ``(T, K, 2)``, taken from the corresponding frame.
        grid: Optional ``(height, width)`` used to validate ``coords``.

    Returns:
        torch.Tensor: ``(T - 1, 2)`` or ``(T - 1, K, 2)`` int64 cells of the retrieved tokens,
        where pair ``t`` queries frame ``t`` and matches in frame ``t + 1``.

    Raises:
        ValueError: If the inputs are not shaped as described above, if fewer than two frames are
        given, or if ``coords`` holds cells outside ``grid``.
    """
    features = torch.as_tensor(features)
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3:
        raise ValueError(f"features must be (T, N, C) or (N, C), got {tuple(features.shape)}")
    num_frames, num_tokens, _ = features.shape
    if num_frames < 2:
        raise ValueError(f"retrieve needs at least two frames, got {num_frames}")

    device = features.device
    cells = torch.as_tensor(coords, dtype=torch.float32, device=device)
    if cells.ndim == 2:
        cells = cells.unsqueeze(0)
    if cells.ndim != 3 or cells.shape[-1] != 2 or cells.shape[1] != num_tokens:
        raise ValueError(f"coords must be (T, {num_tokens}, 2) or ({num_tokens}, 2), got {tuple(cells.shape)}")
    if cells.shape[0] not in (1, num_frames):
        raise ValueError(f"coords has {cells.shape[0]} frames, expected 1 or {num_frames}")
    if grid is not None:
        height, width = token_grid_shape(num_tokens, grid)
        inside = (cells[..., 0] >= 0) & (cells[..., 0] <= height - 1)
        inside = inside & (cells[..., 1] >= 0) & (cells[..., 1] <= width - 1)
        if not bool(inside.all()):
            raise ValueError(f"coords contain cells outside the {height} x {width} grid")
    cells = cells.expand(num_frames, -1, -1)

    queries = torch.as_tensor(query_coords, dtype=torch.float32, device=device)
    single = queries.ndim == 2
    if single:
        queries = queries.unsqueeze(1)
    if queries.ndim != 3 or queries.shape[0] != num_frames or queries.shape[-1] != 2:
        raise ValueError(f"query_coords must be (T, 2) or (T, K, 2) with T={num_frames}")
    queries = torch.nan_to_num(queries, nan=0.0, posinf=0.0, neginf=0.0)

    matched = (cells.unsqueeze(2) == queries.unsqueeze(1)).all(dim=-1)
    token_index = matched.to(torch.float32).argmax(dim=1)

    normalized = normalize_features(features, dim=-1)
    query_tokens = normalized.gather(1, token_index.unsqueeze(-1).expand(-1, -1, normalized.shape[-1]))
    similarity = torch.einsum("tkc,tnc->tkn", query_tokens[:-1], normalized[1:])
    best = similarity.argmax(dim=-1)
    retrieved = cells[1:].gather(1, best.unsqueeze(-1).expand(-1, -1, 2))
    retrieved = retrieved.round().to(torch.int64)
    return retrieved[:, 0] if single else retrieved
