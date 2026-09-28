"""The Instance-Guided Restoration corruption and the sample it produces.

A sample has exactly two possible shapes, never an intermediate one. Either the
construction succeeds: the instance the instruction acts on is re-instantiated at a free
location, the clip is rewritten there, and the loss is told where the disturbance lives
through a weight map concentrated on the interaction region. Or it fails: the clip is
passed through untouched behind a flat unit map, and the sample says so through
:attr:`IGRSample.fallback`.

`build_sample` follows `alg:igr_construction`: the target is tracked, the most reliable
observation of it is chosen, the corruption branch is drawn, and only then is a free
region looked for. Drawing before searching keeps the realised duplication rate at or
below the proposal, since samples without an admissible region fall back to the clean
clip instead of being forced somewhere they do not belong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from eveworld.methods.igr.paste_region import STRIDE, _as_rng, _box_slices, blend_patch, paste_box_region, select_paste_region
from eveworld.methods.igr.trajectory import Track, interaction_box, most_reliable_frame
from eveworld.methods.igr.weight_map import BACKGROUND, build_weight_map, support_mask

if TYPE_CHECKING:
    from eveworld.data.parsers.instruction_parser import ParsedInstruction

__all__ = [
    "MARGIN",
    "MAX_MEDIAN",
    "MEDIAN_KERNEL",
    "P_DUPLICATE",
    "IGREvent",
    "IGRSample",
    "as_clean_sample",
    "build_sample",
    "insert_duplicate",
    "interaction_region",
    "relocate_instance",
]

MARGIN = 4
MEDIAN_KERNEL = 5
MAX_MEDIAN = 31
P_DUPLICATE = 0.5


@dataclass
class IGREvent:
    """One disturbance applied to a clip, the record the weight map is built from.

    Attributes:
        kind: which branch produced the disturbance, `"duplicate"` or `"relocate"`.
        frame: frame the instance patch was cut from, the most reliable observation of
            the track.
        source_box: `(4,)` `float32` box the patch was cut from, in `xyxy` pixel
            coordinates. It is the target's own box, so it coincides with `target_box`.
        paste_box: `(4,)` `float32` box the patch was written into, in `xyxy` pixel
            coordinates, rounded and clipped to the frame.
        target_box: `(4,)` `float32` box of the target instance in the clip.
        track_id: identifier of the track the disturbance was derived from.
    """

    kind: str
    frame: int
    source_box: np.ndarray
    paste_box: np.ndarray
    target_box: np.ndarray
    track_id: int

    def __post_init__(self) -> None:
        self.kind = str(self.kind)
        self.frame = int(self.frame)
        self.track_id = int(self.track_id)
        self.source_box = _event_box(self.source_box)
        self.paste_box = _event_box(self.paste_box)
        self.target_box = _event_box(self.target_box)


@dataclass
class IGRSample:
    """A training sample of the restoration objective.

    Attributes:
        frames: `(T, H, W, 3)` clip, disturbed or clean.
        weight_map: `(H, W)` `float32` map of `eq:igr_weight_raw` normalized to unit mean
            by `eq:igr_weight_norm`, on the grid handed to :func:`build_sample`.
        support: `(H, W)` boolean mask of the cells the disturbed region covers.
        events: disturbances applied to the clip; empty on a fallback sample.
        fallback: `True` when the clip is the clean original behind a flat unit map.
        metadata: bookkeeping for the logs, e.g. the target phrase, the reliable frame
            and the number of candidate regions that were admissible.
    """

    frames: np.ndarray
    weight_map: np.ndarray
    support: np.ndarray
    events: list[IGREvent] = field(default_factory=list)
    fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.events = list(self.events)
        self.metadata = dict(self.metadata)


def as_clean_sample(
    frames: np.ndarray,
    *,
    reason: str = "no_track",
    metadata: dict[str, Any] | None = None,
) -> IGRSample:
    """Build the fallback sample: the untouched clip behind a flat unit map.

    The clip is passed through by reference rather than copied, since nothing is written
    to it, and the gate of the joint objective stays enabled: the sample still
    contributes a restoration term, it just has no disturbed region to emphasize.

    Args:
        frames: `(T, H, W, 3)` clip.
        reason: why the construction fell back, e.g. `"no_track"`,
            `"no_reliable_observation"` or `"no_paste_region"`.
        metadata: extra bookkeeping merged over the default entries.

    Returns:
        An :class:`IGRSample` with `fallback` set, an empty support and a unit weight map.

    Raises:
        ValueError: if `frames` is not a `(T, H, W, ...)` clip.
    """
    clip = np.asarray(frames)
    if clip.ndim < 3:
        raise ValueError(f"Expected a (T, H, W, 3) clip, got {clip.shape}")
    shape = (int(clip.shape[1]), int(clip.shape[2]))
    details: dict[str, Any] = {"kind": "clean", "target": None, "reason": str(reason)}
    if metadata:
        details.update(metadata)
    return IGRSample(
        frames=frames,
        weight_map=np.full(shape, BACKGROUND, dtype=np.float32),
        support=np.zeros(shape, dtype=bool),
        events=[],
        fallback=True,
        metadata=details,
    )


def insert_duplicate(
    frames: np.ndarray,
    track: Track,
    region: np.ndarray,
    *,
    rng: Any = None,
    frame_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, IGREvent]:
    """Duplicate the tracked instance into a free region of the clip.

    The patch is cut from the most reliable observation of the track and pasted into
    `region`, so the clip shows the instructed interaction twice: where it was, and where
    the world model has to learn to render it.

    Args:
        frames: `(T, H, W, 3)` clip.
        track: track of the instance to duplicate.
        region: `(4,)` `xyxy` pixel box the duplicate is written into; rounded half up and
            clipped to the frame.
        rng: generator or seed, accepted for signature parity with
            :func:`relocate_instance`. The duplicate is sampled from the track alone, so
            no random number is drawn here and the result is deterministic.
        frame_range: half-open `(start, stop)` frame interval to paste into; defaults to
            the whole clip.

    Returns:
        The disturbed copy of `frames` and the :class:`IGREvent` that records it.

    Raises:
        ValueError: if `frames` is not a `(T, H, W, ...)` clip, if no observation of the
            track is reliable enough to cut a patch from, or if `region` is empty after
            rounding and clipping.
    """
    clip = _as_clip(frames)
    _as_rng(rng)
    source_frame = _reliable_frame(track, clip)
    patch = _crop_patch(clip[source_frame], track.boxes[source_frame])
    box = _paste_box(region, clip.shape[1:3])
    return blend_patch(clip, region, patch, frame_range), IGREvent(
        kind="duplicate",
        frame=source_frame,
        source_box=track.boxes[source_frame],
        paste_box=box,
        target_box=track.boxes[source_frame],
        track_id=track.track_id,
    )


def relocate_instance(
    frames: np.ndarray,
    track: Track,
    region: np.ndarray,
    *,
    frame_range: tuple[int, int] | None = None,
    kernel: int = MEDIAN_KERNEL,
) -> tuple[np.ndarray, IGREvent]:
    """Move the tracked instance out of its trajectory and into a free region.

    Every frame that shows the instance is repaired first: the pixels under its box are
    replaced by a median filtered crop of the same coordinates taken from a donor frame,
    the frame of the track whose instance sat furthest from where the patch is cut. That
    fill only ever touches a window around the box, so the cost stays at patch scale
    instead of filtering whole frames. The instance itself is then written into `region`.

    Args:
        frames: `(T, H, W, 3)` clip.
        track: track of the instance to relocate.
        region: `(4,)` `xyxy` pixel box the instance is moved to; rounded half up and
            clipped to the frame.
        frame_range: half-open `(start, stop)` frame interval the instance is pasted
            into; defaults to the whole clip. The repair of the original trajectory is
            not restricted to this range.
        kernel: side of the median window used for the repair, in pixels; clamped to
            `[1, MAX_MEDIAN]` and to the size of the region it filters.

    Returns:
        The disturbed copy of `frames` and the :class:`IGREvent` that records it.

    Raises:
        ValueError: if `frames` is not a `(T, H, W, ...)` clip, if no observation of the
            track is reliable enough to cut a patch from, or if `region` is empty after
            rounding and clipping.
    """
    clip = _as_clip(frames)
    source_frame = _reliable_frame(track, clip)
    source_box = track.boxes[source_frame]
    patch = _crop_patch(clip[source_frame], source_box)
    donor = _donor_frame(track, source_box, source_frame)
    repaired = np.array(clip, copy=True)
    for index, box in enumerate(track.boxes):
        if not np.isfinite(box).all():
            continue
        fill = _median_patch(repaired[donor], box, kernel)
        if fill is None:
            continue
        repaired[index] = paste_box_region(repaired[index], box, fill)
    box = _paste_box(region, clip.shape[1:3])
    return blend_patch(repaired, region, patch, frame_range), IGREvent(
        kind="relocate",
        frame=source_frame,
        source_box=source_box,
        paste_box=box,
        target_box=source_box,
        track_id=track.track_id,
    )


def interaction_region(track: Track, event: IGREvent, shape: Sequence[int]) -> np.ndarray:
    """Box around everything one disturbance touched.

    The region unions the instance at the disturbed frame, the box the patch was cut from
    and the box it was written into, then grows it by `MARGIN` so that the halo of the
    paste is weighted too. It is the region the weight map of `eq:igr_weight_raw` marks
    as disturbed.

    Args:
        track: track the event was derived from.
        event: disturbance to bound.
        shape: frame shape whose first two entries are `(height, width)`.

    Returns:
        `(4,)` `float32` box in `xyxy` pixel coordinates, clipped to the frame and
        guaranteed to have a non-zero extent. A fully unusable event yields the whole
        frame.
    """
    dims = np.asarray(shape).reshape(-1)
    if dims.size < 2:
        raise ValueError(f"Expected at least a (H, W) shape, got {shape}")
    height, width = int(dims[0]), int(dims[1])
    candidates = [
        interaction_box(track, event.frame),
        np.asarray(event.source_box, dtype=np.float64),
        np.asarray(event.target_box, dtype=np.float64),
        np.asarray(event.paste_box, dtype=np.float64),
    ]
    usable = [box for box in candidates if np.asarray(box).reshape(-1).size == 4 and np.isfinite(np.asarray(box)).all()]
    if not usable:
        return np.asarray([0.0, 0.0, float(width), float(height)], dtype=np.float32)
    stack = np.stack([np.asarray(box, dtype=np.float64).reshape(4) for box in usable])
    box = np.asarray(
        [
            stack[:, 0].min() - MARGIN,
            stack[:, 1].min() - MARGIN,
            stack[:, 2].max() + MARGIN,
            stack[:, 3].max() + MARGIN,
        ]
    )
    x0, y0, x1, y1 = (int(value) for value in np.rint(box))
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    if x1 <= x0:
        x0 = min(x0, width - 1)
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y0 = min(y0, height - 1)
        y1 = min(height, y0 + 1)
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def build_sample(
    frames: np.ndarray,
    track: Track,
    parsed: "ParsedInstruction",
    *,
    p_dup: float = P_DUPLICATE,
    rng: Any = None,
    shape: Sequence[int] | None = None,
    occupancy_map: np.ndarray | None = None,
    occupied_boxes: np.ndarray | None = None,
    stride: int = STRIDE,
) -> IGRSample:
    """Build one disturbed clip with its weight map (`alg:igr_construction`).

    The steps follow the algorithm in order: the most reliable observation of the track
    is chosen, the branch is drawn, a free region is searched, the disturbance is applied
    and the weight map is built over the union of the trajectory and the paste region.
    Any step that finds nothing usable returns a clean sample through
    :func:`as_clean_sample` rather than a half-built one.

    Args:
        frames: `(T, H, W, 3)` clip.
        track: track of the instance the instruction acts on.
        parsed: parsed instruction, used for the target phrase recorded in the metadata.
        p_dup: probability of drawing the duplication branch; the rest of the draws go to
            relocation.
        rng: generator, seed or `None` for a fresh generator.
        shape: `(H, W)` grid the weight map is built on; defaults to the frame size.
        occupancy_map: optional `(H, W)` or `(T, H, W)` occupancy of the scene, used to
            reject paste regions that are already taken.
        occupied_boxes: `(M, 4)` boxes the paste must avoid; defaults to the trajectory
            of the track itself.
        stride: lattice step in pixels for the region search.

    Returns:
        An :class:`IGRSample`; `fallback` is `True` when no disturbance was applied.

    Raises:
        ValueError: if `frames` is not a `(T, H, W, ...)` clip or `shape` is not a
            positive `(H, W)` pair.
    """
    clip = _as_clip(frames)
    grid = _grid(shape, clip)
    generator = _as_rng(rng)
    target = parsed.target or track.label
    details: dict[str, Any] = {"target": target, "track_id": track.track_id, "label": track.label}
    source_frame = most_reliable_frame(track, clip.shape[1:3])
    if source_frame is None:
        return as_clean_sample(clip, reason="no_reliable_observation", metadata=details)
    details["reliable_frame"] = source_frame
    duplicate = bool(generator.random() < float(p_dup))
    source_box = track.boxes[source_frame]
    obstacle = np.asarray(occupied_boxes) if occupied_boxes is not None else _trajectory_boxes(track)
    region = select_paste_region(
        source_box,
        obstacle,
        clip.shape[1:3],
        occupancy_map=occupancy_map,
        stride=stride,
        rng=generator,
    )
    if region is None:
        return as_clean_sample(clip, reason="no_paste_region", metadata=details)
    if duplicate:
        disturbed, event = insert_duplicate(clip, track, region)
    else:
        disturbed, event = relocate_instance(clip, track, region)
    details["candidate_count"] = int(np.asarray(_admissible(source_box, obstacle, clip.shape[1:3], occupancy_map, stride)).shape[0])
    details.update(
        {
            "kind": event.kind,
            "source_box": event.source_box,
            "paste_box": event.paste_box,
            "mask_area": _mask_area(track, source_frame, event.source_box),
        }
    )
    regions = [_trajectory_boxes(track), event.paste_box]
    return IGRSample(
        frames=disturbed,
        weight_map=build_weight_map(grid, regions, normalize=True),
        support=support_mask(grid, regions),
        events=[event],
        fallback=False,
        metadata=details,
    )


def _admissible(
    target_box: np.ndarray,
    occupied: np.ndarray,
    shape: Sequence[int],
    occupancy_map: np.ndarray | None,
    stride: int,
) -> np.ndarray:
    """Admissible regions for the metadata count of :func:`build_sample`."""
    from eveworld.methods.igr.paste_region import admissible_regions

    return admissible_regions(target_box, occupied, shape, occupancy_map=occupancy_map, stride=stride)


def _trajectory_boxes(track: Track) -> np.ndarray:
    """Distinct finite boxes of a track, the default obstacle set of a paste."""
    boxes = np.asarray(track.boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = boxes.reshape(-1, 4)
    finite = boxes[np.isfinite(boxes).all(axis=1)]
    if finite.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.unique(np.round(finite, decimals=6), axis=0).astype(np.float32)


def _mask_area(track: Track, frame: int, box: np.ndarray) -> float:
    """Area the disturbance covers: the instance mask when there is one, else its box."""
    if track.masks is not None:
        return float(np.count_nonzero(track.masks[frame]))
    array = np.asarray(box, dtype=np.float64).reshape(4)
    if not np.isfinite(array).all():
        return 0.0
    return float(max(0.0, array[2] - array[0]) * max(0.0, array[3] - array[1]))


def _reliable_frame(track: Track, clip: np.ndarray) -> int:
    """Frame the instance patch is cut from, or a loud error when there is none."""
    frame = most_reliable_frame(track, clip.shape[1:3])
    if frame is None:
        raise ValueError(
            f"Track {track.track_id} ({track.label!r}) has no reliable observation to cut the "
            f"instance patch from; every frame is below the area or score floor, outside the "
            f"{clip.shape[1:3]} frame, or not detected at all"
        )
    return frame


def _donor_frame(track: Track, source_box: np.ndarray, source_frame: int) -> int:
    """Frame whose instance sat furthest from `source_box`, the repair donor."""
    from eveworld.methods.igr.paste_region import box_iou

    best, best_overlap = source_frame, None
    for index, box in enumerate(track.boxes):
        if not np.isfinite(box).all():
            continue
        overlap = box_iou(box, source_box)
        if best_overlap is None or overlap < best_overlap:
            best, best_overlap = index, overlap
    return best


def _median_patch(frame: np.ndarray, box: np.ndarray, kernel: int) -> np.ndarray | None:
    """Median filtered pixels under `box`, taken from `frame` with a padded window."""
    window = _box_slices(box, frame.shape[:2])
    if window is None:
        return None
    rows, columns = window
    side = int(np.clip(int(kernel), 1, MAX_MEDIAN))
    pad = side // 2
    top = max(0, rows.start - pad)
    bottom = min(frame.shape[0], rows.stop + pad)
    left = max(0, columns.start - pad)
    right = min(frame.shape[1], columns.stop + pad)
    patch = np.asarray(frame[top:bottom, left:right], dtype=np.float32)
    if patch.ndim != 3 or patch.shape[0] == 0 or patch.shape[1] == 0:
        return None
    kernel_y = _odd(side, patch.shape[0])
    kernel_x = _odd(side, patch.shape[1])
    padded = np.pad(patch, ((kernel_y // 2, kernel_y // 2), (kernel_x // 2, kernel_x // 2), (0, 0)), mode="edge")
    view = sliding_window_view(padded, (kernel_y, kernel_x, 1), axis=(0, 1, 2))
    filtered = np.median(view, axis=(-3, -2, -1))
    return filtered[rows.start - top : rows.stop - top, columns.start - left : columns.stop - left].astype(frame.dtype)


def _odd(side: int, available: int) -> int:
    """Largest odd kernel no larger than `side` and no larger than `available`."""
    limit = max(1, int(available))
    value = min(int(side), limit)
    if value % 2 == 0:
        value = max(1, value - 1)
    return value


def _crop_patch(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Pixels under a box, the instance patch a disturbance re-uses."""
    window = _box_slices(box, frame.shape[:2])
    if window is None:
        raise ValueError(f"Cannot cut a patch from the degenerate box {np.asarray(box).tolist()}")
    rows, columns = window
    return np.array(frame[rows, columns], copy=True)


def _paste_box(region: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Round a region to whole pixels and clip it to the frame."""
    box = np.asarray(region, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.isfinite(box).all():
        raise ValueError(f"Expected a finite (4,) paste region, got {np.asarray(region).tolist()}")
    height, width = int(shape[0]), int(shape[1])
    x0, y0, x1, y1 = (int(np.floor(value + 0.5)) for value in box)
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Paste region {box.tolist()} is empty once rounded and clipped to {(height, width)}")
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def _grid(shape: Sequence[int] | None, clip: np.ndarray) -> tuple[int, int]:
    """Resolve the weight map grid, defaulting to the frames of the clip."""
    if shape is None:
        return int(clip.shape[1]), int(clip.shape[2])
    dims = np.asarray(shape).reshape(-1)
    if dims.size < 2:
        raise ValueError(f"Expected a (H, W) weight map shape, got {shape}")
    height, width = int(dims[0]), int(dims[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Weight map grid must be non-empty, got {(height, width)}")
    return height, width


def _as_clip(frames: np.ndarray) -> np.ndarray:
    """View `frames` as a `(T, H, W, 3)` clip."""
    clip = np.asarray(frames)
    if clip.ndim != 4:
        raise ValueError(f"Expected a (T, H, W, 3) clip, got {clip.shape}")
    return clip


def _event_box(box: Any) -> np.ndarray:
    """Coerce one event box to a `(4,)` `float32` array."""
    array = np.asarray(box, dtype=np.float32).reshape(-1)
    if array.size != 4:
        raise ValueError(f"Expected a (4,) event box, got {np.shape(box)}")
    return array
