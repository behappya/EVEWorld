"""Spatial weight map of the IGR restoration term.

The restoration objective of `eq:pipeline` weights the reconstruction error of every
spatiotemporal latent location by the square root of a map `W_hat`. That map is built
from the annotation of the disturbed interaction region: locations inside the region
receive three times the reconstruction weight of ordinary ones,

    W(p) = 1 + 2 * M(p),                                            (`eq:igr_weight_raw`)

with `M` the indicator of the restoration region, and the raw map is then normalized to
unit mean over all of its locations,

    W_hat(p) = W(p) / mean(W),                                      (`eq:igr_weight_norm`)

so that concentrating weight on a small region does not rescale the global objective
(`alg:igr_construction`, step 6). Regions reach this module in whichever form the
sampler holds them: an `(H, W)` or `(T, H, W)` mask, as a single `(4,)` box or an
`(M, 4)` stack of boxes, all in `xyxy` pixel coordinates. Boolean arrays are always read
as masks, so a box array has to be numeric.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "BACKGROUND",
    "DISTURBED",
    "REGION_WEIGHTS",
    "boxes_to_mask",
    "build_weight_map",
    "clip_boxes",
    "normalize_unit_mean",
    "stamp_regions",
    "support_mask",
]

DISTURBED = 3.0
BACKGROUND = 1.0
REGION_WEIGHTS = {"interaction": DISTURBED, "background": BACKGROUND}


def normalize_unit_mean(weight_map: np.ndarray) -> np.ndarray:
    """Rescale a weight map to unit mean (`eq:igr_weight_norm`).

    Relative weighting is preserved: entries that stand in a `3:1` ratio still stand in
    a `3:1` ratio afterwards, only the global scale changes.

    Args:
        weight_map: `(H, W)` weights, or an array of any other shape; values must be
            finite and must not average out to zero.

    Returns:
        `float32` array of the same shape with mean `1`. An empty array is returned
        unchanged, since it holds no location to normalize over.

    Raises:
        ValueError: if the map holds non-finite values, or if its mean is not positive.
    """
    array = np.asarray(weight_map, dtype=np.float32)
    if array.size == 0:
        return array
    if not np.isfinite(array).all():
        raise ValueError("weight map holds non-finite values")
    mean = float(array.mean())
    if not mean > 0.0:
        raise ValueError(f"weight map must have a positive mean, got {mean}")
    return array / mean


def stamp_regions(shape: tuple[int, int], regions: Iterable[Any], *, value: float = 1.0) -> np.ndarray:
    """Rasterize `regions` onto a zero canvas, writing `value` under their union.

    Args:
        shape: `(H, W)` shape of the canvas; extra trailing axes are ignored, so
            `frames.shape[1:]` of a `(T, H, W, 3)` clip is accepted.
        regions: iterable of region specifications, each a mask or a box as described in
            the module docstring.
        value: value written under the union of `regions`.

    Returns:
        `(H, W)` `float32` canvas of zeros outside the regions.
    """
    canvas = np.zeros(_canvas_shape(shape), dtype=np.float32)
    canvas[support_mask(shape, regions)] = float(value)
    return canvas


def boxes_to_mask(shape: tuple[int, int], boxes: np.ndarray) -> np.ndarray:
    """Rasterize `xyxy` boxes into a boolean `(H, W)` mask.

    Boxes that leave the canvas are clipped rather than dropped, so a box straddling the
    border still marks the part of it that is inside the frame. Coordinates are rounded
    half up and read half open, `[x0, x1) x [y0, y1)`.

    Args:
        shape: `(H, W)` shape of the mask.
        boxes: `(4,)` box or `(M, 4)` boxes in `xyxy` pixel coordinates. Rows holding a
            non-finite value are skipped.

    Returns:
        `(H, W)` boolean mask, `True` where a box covers the cell.
    """
    canvas = _canvas_shape(shape)
    mask = np.zeros(canvas, dtype=bool)
    for box in _as_box_rows(boxes):
        mask |= _box_mask(canvas, box)
    return mask


def support_mask(shape: tuple[int, int], regions: Iterable[Any]) -> np.ndarray:
    """Union of `regions` as a boolean `(H, W)` mask.

    This is the support of the weight map: the cells the restoration loss emphasizes.

    Args:
        shape: `(H, W)` shape of the mask.
        regions: iterable of region specifications, each a mask or a box as described in
            the module docstring.

    Returns:
        `(H, W)` boolean mask, `True` under the union of `regions`.
    """
    canvas = _canvas_shape(shape)
    support = np.zeros(canvas, dtype=bool)
    for region in regions:
        support |= _region_mask(canvas, region)
    return support


def clip_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Clip `xyxy` boxes to the canvas without dropping the ones that leave it.

    Only the coordinates are clamped: a box that lands outside the canvas becomes
    degenerate rather than empty, which keeps its index in a stack of boxes stable.

    Args:
        boxes: `(4,)` box or `(M, 4)` boxes in `xyxy` pixel coordinates.
        shape: `(H, W)` canvas shape.

    Returns:
        `float32` array with the rank of the input, coordinates clamped to
        `[0, W] x [0, H]`. Rows holding a non-finite value are returned untouched, as
        there is nothing to clip them against.
    """
    height, width = _canvas_shape(shape)
    rows, single = _box_stack(boxes)
    clipped = np.array(rows, dtype=np.float32, copy=True)
    finite = np.isfinite(clipped).all(axis=1)
    clipped[finite, 0] = np.clip(clipped[finite, 0], 0.0, float(width))
    clipped[finite, 1] = np.clip(clipped[finite, 1], 0.0, float(height))
    clipped[finite, 2] = np.clip(clipped[finite, 2], 0.0, float(width))
    clipped[finite, 3] = np.clip(clipped[finite, 3], 0.0, float(height))
    return clipped[0] if single else clipped


def build_weight_map(
    shape: tuple[int, int],
    regions: Iterable[Any],
    *,
    disturbed: float = DISTURBED,
    background: float = BACKGROUND,
    normalize: bool = True,
) -> np.ndarray:
    """Weight map of the disturbed interaction region (`eq:igr_weight_raw`, `eq:igr_weight_norm`).

    Args:
        shape: `(H, W)` shape of the map.
        regions: iterable of region specifications marking the cells that carry the
            disturbed weight; each is a mask or a box as described in the module
            docstring. An empty iterable yields a flat map.
        disturbed: weight of a cell covered by `regions`.
        background: weight of every other cell.
        normalize: divide the map by its own mean, giving it unit mean as
            `eq:igr_weight_norm` prescribes. Turn this off to keep the raw levels.

    Returns:
        `(H, W)` `float32` weight map.
    """
    weight = np.full(_canvas_shape(shape), float(background), dtype=np.float32)
    weight[support_mask(shape, regions)] = float(disturbed)
    return normalize_unit_mean(weight) if normalize else weight


def _canvas_shape(shape: Sequence[int]) -> tuple[int, int]:
    """Read `(height, width)` out of a canvas shape, ignoring extra trailing axes."""
    dims = np.asarray(shape).reshape(-1)
    if dims.size < 2:
        raise ValueError(f"Expected at least a (H, W) shape, got {shape}")
    height, width = int(dims[0]), int(dims[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Canvas must be non-empty, got {(height, width)}")
    return height, width


def _box_stack(boxes: Any) -> tuple[np.ndarray, bool]:
    """View `boxes` as an `(M, 4)` array and report whether the input was a single box."""
    array = np.asarray(boxes)
    single = array.ndim == 1
    if single:
        if array.size != 4:
            raise ValueError(f"Expected a (4,) box, got {array.shape}")
        array = array.reshape(1, 4)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"Expected (4,) or (M, 4) boxes, got {array.shape}")
    return array, single


def _as_box_rows(boxes: Any) -> np.ndarray:
    """View `boxes` as a `(M, 4)` float array, tolerating an empty input."""
    try:
        rows, _ = _box_stack(boxes)
    except ValueError:
        if np.asarray(boxes).size == 0:
            return np.zeros((0, 4), dtype=np.float64)
        raise
    return rows


def _as_boxes(region: Any) -> np.ndarray | None:
    """View a region as `(M, 4)` float boxes, or `None` when it is not box shaped."""
    array = np.asarray(region)
    if array.dtype == bool:
        return None
    if array.ndim == 1 and array.size == 4:
        return array.reshape(1, 4).astype(np.float64)
    if array.ndim in (2, 3) and array.shape[-1] == 4:
        return array.reshape(-1, 4).astype(np.float64)
    return None


def _region_mask(canvas: tuple[int, int], region: Any) -> np.ndarray:
    """Rasterize one region, given either as boxes or as a mask."""
    boxes = _as_boxes(region)
    if boxes is None:
        return _mask_region(canvas, region)
    mask = np.zeros(canvas, dtype=bool)
    for box in boxes:
        mask |= _box_mask(canvas, box)
    return mask


def _mask_region(canvas: tuple[int, int], region: Any) -> np.ndarray:
    """View a region as a boolean `(H, W)` mask, OR-folding any leading axes."""
    array = np.asarray(region)
    if array.ndim < 2:
        raise ValueError(f"Expected a mask, a (4,) box or (M, 4) boxes, got {array.shape}")
    spatial = array if array.ndim == 2 else np.any(array != 0, axis=tuple(range(array.ndim - 2)))
    if spatial.shape != canvas:
        raise ValueError(f"Region mask {array.shape} does not cover the {canvas} canvas")
    return np.asarray(spatial != 0, dtype=bool)


def _box_mask(canvas: tuple[int, int], box: np.ndarray) -> np.ndarray:
    """Rasterize one `(4,)` box, rounded half up and clipped to the canvas."""
    height, width = canvas
    mask = np.zeros(canvas, dtype=bool)
    if not np.isfinite(box).all():
        return mask
    x0, y0, x1, y1 = (int(np.floor(float(value) + 0.5)) for value in box[:4])
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask
