"""Where the disturbed instance is placed, and how the patch lands there.

Instance-Guided Restoration disturbs a clip by re-instantiating the interaction the
instruction asks for at a location the world model can render (`alg:igr_construction`).
Two ingredients are needed for that: a *free* region of the frame, and a way to write
the instance patch into it.

Regions are searched on the stride lattice the video VAE uses, so a candidate is a
whole number of latent cells wide and the pasted instance never straddles a cell
boundary. A candidate is admissible only when it neither overlaps the instance it is
derived from nor collides with anything already in the scene, which is what keeps the
duplicate readable as a second object rather than a smear over the original one. The
patch itself is written in with a hard paste: only the evaluation protocol varies the
blend, so the training corruption stays exact.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "STRIDE",
    "admissible_regions",
    "blend_patch",
    "box_iou",
    "candidate_regions",
    "in_frame",
    "inscribed_boxes",
    "occupancy_ratio",
    "overlaps_any",
    "paste_box_region",
    "select_paste_region",
]

STRIDE = 8


def in_frame(box: np.ndarray, shape: Sequence[int]) -> bool:
    """Report whether a box lies inside the frame and has a non-zero extent.

    Args:
        box: `(4,)` box in `xyxy` pixel coordinates.
        shape: frame shape whose first two entries are `(height, width)`.

    Returns:
        `True` when every coordinate is finite and `0 <= x0 < x1 <= W`,
        `0 <= y0 < y1 <= H`.
    """
    array = _box(box)
    if array is None:
        return False
    height, width = _canvas(shape)
    x0, y0, x1, y1 = array
    return bool(0.0 <= x0 < x1 <= width and 0.0 <= y0 < y1 <= height)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boxes.

    Args:
        a: `(4,)` box in `xyxy` pixel coordinates.
        b: `(4,)` box in `xyxy` pixel coordinates.

    Returns:
        The IoU in `[0, 1]`, or `0.0` when either box is degenerate or holds a
        non-finite coordinate, so that an unusable box never blocks a candidate.
    """
    first, second = _box(a), _box(b)
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    union = _area(first) + _area(second) - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def overlaps_any(box: np.ndarray, boxes: Any, threshold: float = 0.0) -> bool:
    """Report whether a box overlaps any box of a collection.

    Args:
        box: `(4,)` box in `xyxy` pixel coordinates.
        boxes: `(4,)` box or `(M, 4)` boxes to test against; an empty collection
            overlaps nothing.
        threshold: an overlap counts only when the IoU is strictly greater than this
            value, so the default of `0.0` rejects any touching candidate.

    Returns:
        `True` when some box in `boxes` has an IoU above `threshold` with `box`.
    """
    for other in _box_rows(boxes):
        if box_iou(box, other) > float(threshold):
            return True
    return False


def occupancy_ratio(box: np.ndarray, occupancy_map: np.ndarray) -> float:
    """Mean occupancy of the cells a box covers.

    Args:
        box: `(4,)` box in the coordinate frame of `occupancy_map`.
        occupancy_map: `(H, W)` occupancy of the frame, or a `(T, H, W)` stack whose
            leading axes are averaged over. Expected to hold values in `[0, 1]` where
            `1` means fully occupied by an existing object.

    Returns:
        Mean occupancy under the box, in `[0, 1]`. `1.0` is returned when the box is
        unusable or covers no cell at all, so that an unknown region is treated as
        occupied rather than free.
    """
    array = np.asarray(occupancy_map, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError(f"Expected a (H, W) or (T, H, W) occupancy map, got {array.shape}")
    spatial = array if array.ndim == 2 else array.mean(axis=tuple(range(array.ndim - 2)))
    cells = _crop(spatial, box)
    if cells is None:
        return 1.0
    values = cells[np.isfinite(cells)]
    if values.size == 0:
        return 1.0
    return float(np.clip(values.mean(), 0.0, 1.0))


def inscribed_boxes(box: np.ndarray, size: tuple[int, int] | None = None, *, stride: int = STRIDE) -> np.ndarray:
    """Enumerate the windows of `size` that fit inside `box` on the stride lattice.

    Anchors start at the top-left corner of the container and advance by `stride`, and
    the flush bottom-right anchor is always added, so the far edge of the container is
    reachable even when the stride does not divide it.

    Args:
        box: `(4,)` container in `xyxy` pixel coordinates.
        size: `(height, width)` of the window in pixels; defaults to the container's own
            size, which yields the container itself.
        stride: lattice step in pixels, 8 for the latent grid of the video VAE.

    Returns:
        `(M, 4)` `float32` windows in `xyxy` order, sorted by `(y0, x0)`, or an empty
        `(0, 4)` array when the container is degenerate or smaller than the window.
    """
    container = _box(box)
    if container is None:
        return np.zeros((0, 4), dtype=np.float32)
    x0, y0, x1, y1 = container
    width, height = x1 - x0, y1 - y0
    if size is None:
        window_h, window_w = height, width
    else:
        dims = np.asarray(size, dtype=np.float64).reshape(-1)
        if dims.size < 2:
            raise ValueError(f"Expected a (height, width) window size, got {size}")
        window_h, window_w = float(dims[0]), float(dims[1])
    step = int(stride)
    if step <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if window_w <= 0.0 or window_h <= 0.0 or width < window_w or height < window_h:
        return np.zeros((0, 4), dtype=np.float32)
    columns = _anchors(x0, width, window_w, step)
    rows = _anchors(y0, height, window_h, step)
    grid_x, grid_y = np.meshgrid(columns, rows)
    corners = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    boxes = np.concatenate([corners, corners + np.asarray([window_w, window_h])], axis=1)
    return _unique_sorted(boxes)


def candidate_regions(
    target_box: np.ndarray,
    occupied_boxes: Any,
    shape: Sequence[int],
    *,
    stride: int = STRIDE,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Windows the size of `target_box` that are free of everything already placed.

    The search space is every window of the target's size that fits in the frame, on the
    stride lattice. A window is dropped when it overlaps `occupied_boxes` or the target
    itself, which is what stops the duplicate from landing on top of the instance it was
    cut from.

    Args:
        target_box: `(4,)` box of the instance that will be duplicated, in `xyxy` pixel
            coordinates. A non-finite box has no free region to be found.
        occupied_boxes: `(4,)` box or `(M, 4)` boxes that must not be covered, usually
            the trajectory of every tracked object.
        shape: frame shape whose first two entries are `(height, width)`.
        stride: lattice step in pixels.
        candidates: pre-enumerated `(M, 4)` windows to filter instead of searching the
            frame; useful when a caller already holds the lattice.

    Returns:
        `(M, 4)` `float32` admissible windows in enumeration order, possibly empty.
    """
    target = _box(target_box)
    if target is None:
        return np.zeros((0, 4), dtype=np.float32)
    height, width = _canvas(shape)
    if candidates is None:
        window = (target[3] - target[1], target[2] - target[0])
        frame = np.asarray([0.0, 0.0, float(width), float(height)], dtype=np.float64)
        candidates = inscribed_boxes(frame, size=window, stride=stride)
    rows = _box_rows(candidates)
    kept = [row for row in rows if box_iou(row, target) <= 0.0 and not overlaps_any(row, occupied_boxes, 0.0)]
    return _stack(kept)


def admissible_regions(
    target_box: np.ndarray,
    occupied_boxes: Any,
    shape: Sequence[int],
    *,
    occupancy_map: np.ndarray | None = None,
    stride: int = STRIDE,
    collision_threshold: float = 0.0,
    occupancy_threshold: float = 0.5,
) -> np.ndarray:
    """Free windows that also clear the occupancy map, the form `build_sample` samples.

    Args:
        target_box: `(4,)` box of the instance that will be relocated, in `xyxy` pixel
            coordinates.
        occupied_boxes: `(4,)` box or `(M, 4)` boxes that must not be covered.
        shape: frame shape whose first two entries are `(height, width)`.
        occupancy_map: optional `(H, W)` or `(T, H, W)` occupancy of the scene; windows
            occupied above `occupancy_threshold` are dropped.
        stride: lattice step in pixels.
        collision_threshold: an overlap with `occupied_boxes` counts only when its IoU
            is strictly greater than this value.
        occupancy_threshold: highest mean occupancy a window may have.

    Returns:
        `(M, 4)` `float32` admissible windows in enumeration order, possibly empty.
    """
    target = _box(target_box)
    if target is None:
        return np.zeros((0, 4), dtype=np.float32)
    height, width = _canvas(shape)
    window = (target[3] - target[1], target[2] - target[0])
    frame = np.asarray([0.0, 0.0, float(width), float(height)], dtype=np.float64)
    candidates = inscribed_boxes(frame, size=window, stride=stride)
    kept: list[np.ndarray] = []
    for row in candidates:
        if box_iou(row, target) > 0.0:
            continue
        if overlaps_any(row, occupied_boxes, collision_threshold):
            continue
        if occupancy_map is not None and occupancy_ratio(row, occupancy_map) > occupancy_threshold:
            continue
        kept.append(row)
    return _stack(kept)


def select_paste_region(
    target_box: np.ndarray,
    occupied_boxes: Any,
    shape: Sequence[int],
    *,
    occupancy_map: np.ndarray | None = None,
    stride: int = STRIDE,
    rng: Any = None,
    collision_threshold: float = 0.0,
    occupancy_threshold: float = 0.5,
) -> np.ndarray | None:
    """Draw one admissible paste region uniformly at random.

    Args:
        target_box: `(4,)` box of the instance that will be displaced.
        occupied_boxes: `(4,)` box or `(M, 4)` boxes that must not be covered.
        shape: frame shape whose first two entries are `(height, width)`.
        occupancy_map: optional occupancy map forwarded to :func:`admissible_regions`.
        stride: lattice step in pixels.
        rng: `numpy` generator, or a seed for one, or `None` for a fresh generator.
        collision_threshold: IoU above which a window counts as a collision.
        occupancy_threshold: highest mean occupancy a window may have.

    Returns:
        A `(4,)` `float32` box in `xyxy` pixel coordinates, or `None` when no region is
        admissible for this frame.
    """
    regions = admissible_regions(
        target_box,
        occupied_boxes,
        shape,
        occupancy_map=occupancy_map,
        stride=stride,
        collision_threshold=collision_threshold,
        occupancy_threshold=occupancy_threshold,
    )
    if regions.shape[0] == 0:
        return None
    index = int(_as_rng(rng).integers(regions.shape[0]))
    return regions[index].copy()


def blend_patch(
    frames: np.ndarray,
    region: np.ndarray,
    patch: np.ndarray,
    frame_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Write an instance patch into a region of a clip.

    The paste is hard: every covered pixel is replaced, with no alpha ramp, because only
    the evaluation protocol varies the blend. Frames outside `frame_range` and cells
    outside `region` are copied through untouched.

    Args:
        frames: `(T, H, W, 3)` clip of the source video.
        region: `(4,)` box in `xyxy` pixel coordinates, rounded half up and clipped to
            the frame.
        patch: `(h, w, 3)` patch pasted into every covered frame, or `(T', h, w, 3)`
            patches with one entry per covered frame.
        frame_range: half-open `(start, stop)` frame interval to paste into; defaults to
            the whole clip.

    Returns:
        A new `(T, H, W, 3)` clip of the same dtype with the patch written in. A
        degenerate or out-of-frame `region` leaves the clip unchanged.

    Raises:
        ValueError: if `patch` holds a number of frames other than one or the length of
            `frame_range`.
    """
    clip = np.array(frames, copy=True)
    if clip.ndim != 4:
        raise ValueError(f"Expected a (T, H, W, 3) clip, got {clip.shape}")
    start, stop = _frame_bounds(frame_range, clip.shape[0])
    stacked = np.asarray(patch)
    single = stacked.ndim == 3
    if not single and stacked.ndim != 4:
        raise ValueError(f"Expected a (h, w, 3) or (T, h, w, 3) patch, got {stacked.shape}")
    covered = stop - start
    if not single and stacked.shape[0] != covered:
        raise ValueError(
            f"Patch holds {stacked.shape[0]} frames but the range covers {covered}; " "pass a single (h, w, 3) patch to reuse it for every frame"
        )
    for offset, index in enumerate(range(start, stop)):
        source = stacked if single else stacked[offset]
        clip[index] = paste_box_region(clip[index], region, source)
    return clip


def paste_box_region(frame: np.ndarray, region: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """Paste one patch into one frame, the single-frame primitive of :func:`blend_patch`.

    Args:
        frame: `(H, W, 3)` frame.
        region: `(4,)` box in `xyxy` pixel coordinates; rounded half up and clipped to
            the frame.
        patch: `(h, w, 3)` patch, nearest-neighbour resized to the region. The resize is
            exact when the patch already has the region's size.

    Returns:
        A new `(H, W, 3)` frame of the same dtype. A region that is degenerate or lands
        entirely outside the frame leaves the frame unchanged.

    Raises:
        ValueError: if the frame or the patch is not a three-axis image, or if the
            patch does not carry the frame's channel count.
    """
    image = np.array(frame, copy=True)
    if image.ndim != 3:
        raise ValueError(f"Expected a (H, W, 3) frame, got {image.shape}")
    source = np.asarray(patch)
    if source.ndim != 3:
        raise ValueError(f"Expected a (h, w, 3) patch, got {source.shape}")
    if source.shape[-1] != image.shape[-1]:
        raise ValueError(f"Patch has {source.shape[-1]} channels but the frame has {image.shape[-1]}")
    window = _box_slices(region, image.shape[:2])
    if window is None:
        return image
    rows, columns = window
    size = (rows.stop - rows.start, columns.stop - columns.start)
    resized = _resize_rgb(source, size)
    image[rows, columns] = _cast_like(resized, image.dtype)
    return image


def _box(box: Any) -> np.ndarray | None:
    """View `box` as a finite `(4,)` float array, or `None` when it is unusable."""
    array = np.asarray(box, dtype=np.float64).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        return None
    return array


def _box_rows(boxes: Any) -> np.ndarray:
    """View `boxes` as an `(M, 4)` float array, tolerating an empty input."""
    array = np.asarray(boxes)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[-1] != 4:
        raise ValueError(f"Expected (4,) or (M, 4) boxes, got {array.shape}")
    return array.astype(np.float64)


def _stack(rows: Iterable[np.ndarray]) -> np.ndarray:
    """Stack `(4,)` rows into an `(M, 4)` `float32` array."""
    collected = [np.asarray(row, dtype=np.float32).reshape(4) for row in rows]
    if not collected:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(collected).astype(np.float32)


def _unique_sorted(boxes: np.ndarray) -> np.ndarray:
    """Drop duplicate rows and order the rest by `(y0, x0)`."""
    rounded = np.round(boxes, decimals=6)
    _, index = np.unique(rounded, axis=0, return_index=True)
    unique = boxes[np.sort(index)]
    order = np.lexsort((unique[:, 0], unique[:, 1]))
    return unique[order].astype(np.float32)


def _anchors(origin: float, extent: float, window: float, stride: int) -> np.ndarray:
    """Lattice offsets from `origin` plus the flush far anchor."""
    offsets = np.arange(0.0, extent - window + 1e-9, float(stride))
    flush = extent - window
    if offsets.size == 0 or abs(float(offsets[-1]) - flush) > 1e-6:
        offsets = np.append(offsets, flush)
    return origin + offsets


def _area(box: np.ndarray) -> float:
    """Area of a `(4,)` `xyxy` box, zero when it is degenerate."""
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _crop(array: np.ndarray, box: Any) -> np.ndarray | None:
    """Cells of `array` under `box`, or `None` when the box covers nothing."""
    window = _box_slices(box, array.shape[:2])
    if window is None:
        return None
    rows, columns = window
    return array[rows, columns]


def _box_slices(box: Any, shape: Sequence[int]) -> tuple[slice, slice] | None:
    """Half-open row and column slices of `box`, clipped to the frame."""
    array = _box(box)
    if array is None:
        return None
    height, width = _canvas(shape)
    x0, y0, x1, y1 = (int(np.floor(value + 0.5)) for value in array)
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return slice(y0, y1), slice(x0, x1)


def _canvas(shape: Sequence[int]) -> tuple[int, int]:
    """Read `(height, width)` out of a frame shape, ignoring trailing axes."""
    dims = np.asarray(shape).reshape(-1)
    if dims.size < 2:
        raise ValueError(f"Expected at least a (H, W) shape, got {shape}")
    height, width = int(dims[0]), int(dims[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Frame must be non-empty, got {(height, width)}")
    return height, width


def _frame_bounds(frame_range: Any, num_frames: int) -> tuple[int, int]:
    """Half-open `(start, stop)` of `frame_range`, clipped to the clip."""
    if frame_range is None:
        return 0, num_frames
    dims = np.asarray(frame_range).reshape(-1)
    if dims.size == 0:
        raise ValueError("frame_range must hold a start and a stop")
    start = int(dims[0])
    stop = int(dims[1]) if dims.size > 1 else num_frames
    start = max(0, min(start, num_frames))
    stop = max(start, min(stop, num_frames))
    return start, stop


def _as_rng(rng: Any) -> np.random.Generator:
    """Accept a generator, a seed or `None` and return a generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng()
    return np.random.default_rng(int(rng))


def _resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of an `(h, w, 3)` image to `(height, width)`."""
    height, width = int(size[0]), int(size[1])
    array = np.asarray(image)
    if array.shape[0] == height and array.shape[1] == width:
        return array
    try:
        import cv2

        return cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)
    except ImportError:
        pass
    try:
        from PIL import Image

        resized = Image.fromarray(np.ascontiguousarray(array)).resize((width, height), Image.NEAREST)
        return np.asarray(resized)
    except ImportError as exc:
        raise ImportError("Resizing a patch needs either 'opencv-python' or 'pillow' to be installed") from exc


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast `values` to `dtype`, rounding and clipping when that dtype is integral."""
    target = np.dtype(dtype)
    if target.kind in "iu":
        info = np.iinfo(target)
        return np.clip(np.rint(np.asarray(values, dtype=np.float64)), info.min, info.max).astype(target)
    return np.asarray(values, dtype=target)
