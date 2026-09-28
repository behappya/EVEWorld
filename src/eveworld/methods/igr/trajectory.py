"""Per-object tracks that decide *which* interaction Instance-Guided Restoration repairs.

The construction of `alg:igr_construction` needs one instance, not a whole scene: the
object the instruction acts on, followed from the detection model across the clip. This
module turns the per-frame detections into those tracks, then answers the questions the
corruption step asks about them -- which frame is good enough to cut the patch from
(:func:`most_reliable_frame`), where the object sits over time
(:func:`trajectory_center`, :func:`displacement`) and what a box around the interaction
looks like (:func:`interaction_box`).

Association is delegated to the tracking backend, so the linking rule is the one the
rest of the pipeline uses. Frames in which an object was not detected are kept as gaps
rather than closing the track, since a short occlusion is not a new instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from eveworld.methods.igr.paste_region import in_frame

if TYPE_CHECKING:
    from eveworld.data.grounding.grounding_dino import Detection

__all__ = [
    "MIN_AREA",
    "MIN_SCORE",
    "Track",
    "build_tracks",
    "displacement",
    "interaction_box",
    "most_reliable_frame",
    "reliability",
    "trajectory_center",
]

MIN_AREA = 64
MIN_SCORE = 0.3


@dataclass
class Track:
    """One object followed through a clip.

    Attributes:
        track_id: Identifier of the track within the clip, assigned in order of first
            appearance.
        label: Object phrase the detections matched, e.g. `"red cube"`.
        boxes: `(T, 4)` `float32` boxes in `xyxy` pixel coordinates, one per frame of
            the clip. A frame in which the object was not detected holds `nan`.
        scores: `(T,)` `float32` detection confidence per frame; undetected frames hold
            `0.0`.
        masks: optional `(T, H, W)` boolean instance masks in the frame that was
            segmented, or `None` when the track carries boxes only.
    """

    track_id: int
    label: str
    boxes: np.ndarray
    scores: np.ndarray
    masks: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.track_id = int(self.track_id)
        self.label = str(self.label)
        boxes = np.asarray(self.boxes, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"Expected the (T, 4) boxes of a track, got {np.shape(self.boxes)}")
        self.boxes = boxes
        scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        if scores.size != boxes.shape[0]:
            raise ValueError(
                f"Track holds {boxes.shape[0]} boxes but {scores.size} scores; pass one score " "per frame, 0.0 for the frames without a detection"
            )
        self.scores = scores
        if self.masks is None:
            return
        masks = np.asarray(self.masks) != 0
        if masks.ndim < 3 or masks.shape[0] != boxes.shape[0]:
            raise ValueError(f"Expected (T, H, W) masks matching the {boxes.shape[0]} frames, got " f"{masks.shape}")
        self.masks = masks

    def __len__(self) -> int:
        """Number of frames the track spans, detected or not."""
        return int(self.boxes.shape[0])


@dataclass
class _OpenTrack:
    """A track still being extended frame by frame while `build_tracks` runs."""

    track_id: int
    label: str
    boxes: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    last_box: np.ndarray | None = None


def build_tracks(
    detections: Sequence[Sequence["Detection"]],
    labels: Iterable[str] | None = None,
    *,
    iou_threshold: float = 0.5,
) -> list[Track]:
    """Chain per-frame detections into tracks.

    Detections of one frame are matched to the tracks left open by the previous frames
    with the backend's IoU association, and every detection that matches no track starts
    a new one. A track that goes undetected in a frame keeps a gap there: it is still the
    same instance and is expected to reappear.

    Args:
        detections: one sequence of detections per frame, in clip order. Each detection
            exposes `box` as a `(4,)` `xyxy` box in pixels, `score` and `label`.
        labels: restrict tracking to these object phrases; `None` keeps every detection.
        iou_threshold: minimum IoU for a detection to continue a track.

    Returns:
        One :class:`Track` per tracked object, ordered by first appearance. Every track
        spans all frames of the clip, with `nan` boxes and `0.0` scores where the object
        was not detected. An empty `detections` yields an empty list.
    """
    frames = list(detections)
    if not frames:
        return []
    from eveworld.data.tracking.sam2_tracker import associate_boxes

    wanted = None if labels is None else {str(label) for label in labels}
    open_tracks: list[_OpenTrack] = []
    for index, frame in enumerate(frames):
        boxes, scores, names = _detections_of(frame, wanted)
        association = _associate(associate_boxes, open_tracks, boxes, iou_threshold)
        claimed: set[int] = set()
        for slot, track in enumerate(open_tracks):
            index_in_frame = association[slot] if slot < association.size else -1
            if index_in_frame >= 0 and index_in_frame not in claimed:
                claimed.add(index_in_frame)
                track.boxes.append(boxes[index_in_frame])
                track.scores.append(scores[index_in_frame])
                track.last_box = boxes[index_in_frame]
            else:
                track.boxes.append(_MISSING_BOX.copy())
                track.scores.append(0.0)
        for index_in_frame, box in enumerate(boxes):
            if index_in_frame in claimed:
                continue
            track = _OpenTrack(
                track_id=len(open_tracks),
                label=names[index_in_frame],
                boxes=[_MISSING_BOX.copy() for _ in range(index)],
                scores=[0.0] * index,
                last_box=box,
            )
            track.boxes.append(box)
            track.scores.append(scores[index_in_frame])
            open_tracks.append(track)
    return [
        Track(
            track_id=track.track_id,
            label=track.label,
            boxes=np.stack(track.boxes) if track.boxes else np.zeros((0, 4), dtype=np.float32),
            scores=np.asarray(track.scores, dtype=np.float32),
        )
        for track in open_tracks
    ]


def reliability(
    track: Track,
    shape: Sequence[int] | None = None,
    *,
    min_area: float = MIN_AREA,
    min_score: float = MIN_SCORE,
) -> np.ndarray:
    """Frames in which a track is observed well enough to be used.

    Args:
        track: track to test.
        shape: optional frame shape whose first two entries are `(height, width)`; when
            given, a frame counts as reliable only if its box lies inside the frame.
        min_area: smallest box area in pixels a frame may have.
        min_score: smallest detection confidence a frame may have.

    Returns:
        `(T,)` boolean mask, `True` in the frames the patch may be cut from.
    """
    boxes = track.boxes
    finite = np.isfinite(boxes).all(axis=1)
    widths = np.where(finite, boxes[:, 2] - boxes[:, 0], 0.0)
    heights = np.where(finite, boxes[:, 3] - boxes[:, 1], 0.0)
    areas = np.maximum(widths, 0.0) * np.maximum(heights, 0.0)
    usable = finite & (areas >= float(min_area))
    usable &= np.where(np.isfinite(track.scores), track.scores, 0.0) >= float(min_score)
    if shape is not None:
        inside = np.asarray([in_frame(box, shape) for box in boxes], dtype=bool)
        usable &= inside
    return np.asarray(usable, dtype=bool)


def trajectory_center(track: Track) -> np.ndarray:
    """Centre of the box in every frame.

    Args:
        track: track to project onto its centres.

    Returns:
        `(T, 2)` `float32` array of `(x, y)` centres; a row holds `nan` where the track
        was not detected.
    """
    boxes = track.boxes
    finite = np.isfinite(boxes).all(axis=1)
    centres = np.full((boxes.shape[0], 2), np.nan, dtype=np.float32)
    centres[finite] = 0.5 * (boxes[finite, :2] + boxes[finite, 2:])
    return centres


def displacement(track: Track, origin_len: int = 6) -> np.ndarray:
    """Distance of the instance from where it started, in pixels.

    The origin is the median centre over the first frames of the clip, which is more
    robust than the single first frame when a detection is noisy.

    Args:
        track: track to measure.
        origin_len: number of leading frames the origin is averaged over, clamped to the
            frames that hold a finite centre.

    Returns:
        `(T,)` `float32` distances; a row holds `nan` where the track was not detected.
        An all-`nan` track yields an all-`nan` result.
    """
    centres = trajectory_center(track)
    finite = np.isfinite(centres).all(axis=1)
    distances = np.full(centres.shape[0], np.nan, dtype=np.float32)
    if not finite.any():
        return distances
    count = max(1, min(int(origin_len), int(finite.sum())))
    origin = np.median(centres[finite][:count], axis=0)
    distances[finite] = np.linalg.norm(centres[finite] - origin, axis=1)
    return distances


def most_reliable_frame(
    track: Track,
    shape: Sequence[int] | None = None,
    *,
    min_area: float = MIN_AREA,
    min_score: float = MIN_SCORE,
) -> int | None:
    """Frame the instance patch should be cut from.

    Args:
        track: track to inspect.
        shape: optional frame shape forwarded to :func:`reliability`.
        min_area: smallest box area in pixels a frame may have.
        min_score: smallest detection confidence a frame may have.

    Returns:
        Index of the reliable frame with the highest detection confidence, the first one
        on a tie, or `None` when no frame of the track is reliable.
    """
    usable = reliability(track, shape, min_area=min_area, min_score=min_score)
    if not usable.any():
        return None
    scores = np.where(np.isfinite(track.scores), track.scores, 0.0)
    return int(np.argmax(np.where(usable, scores, -np.inf)))


def interaction_box(track: Track, frame: int, *, margin: float = 4.0) -> np.ndarray:
    """Box around the instance in one frame, grown by a margin.

    The box is only expanded, never clipped: a caller that needs it inside the frame
    clips it itself, and keeping the growth here makes the margin the single place the
    halo width is decided.

    Args:
        track: track to read the box from.
        frame: frame index, negative values counting from the end of the clip.
        margin: pixels the box is grown by on every side.

    Returns:
        `(4,)` `float32` box in `xyxy` pixel coordinates, or four `nan` when the frame is
        outside the clip or holds no detection.
    """
    boxes = track.boxes
    index = int(frame)
    if index < 0:
        index += boxes.shape[0]
    if index < 0 or index >= boxes.shape[0]:
        return _MISSING_BOX.copy()
    box = boxes[index]
    if not np.isfinite(box).all():
        return _MISSING_BOX.copy()
    grown = np.asarray(box, dtype=np.float64).copy()
    grown[0] -= float(margin)
    grown[1] -= float(margin)
    grown[2] += float(margin)
    grown[3] += float(margin)
    return grown.astype(np.float32)


def _detections_of(frame: Sequence["Detection"], wanted: set[str] | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Boxes, scores and labels of the usable detections of one frame."""
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    names: list[str] = []
    for detection in frame:
        label = str(getattr(detection, "label", "") or "")
        if wanted is not None and label not in wanted:
            continue
        box = np.asarray(getattr(detection, "box", None), dtype=np.float64).reshape(-1)
        if box.size != 4 or not np.isfinite(box).all():
            continue
        boxes.append(box.astype(np.float32))
        scores.append(float(getattr(detection, "score", 0.0) or 0.0))
        names.append(label)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), names
    return np.stack(boxes), np.asarray(scores, dtype=np.float32), names


def _associate(
    associate_boxes: Any,
    open_tracks: Sequence[_OpenTrack],
    boxes: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Match this frame's boxes to the open tracks, one entry per open track.

    The backend returns one association per *previous* box. Its convention is read back
    from the length of its output, and a shorter-than-expected answer is padded with
    `-1` so that a track the backend left out is simply treated as unmatched.
    """
    count = len(open_tracks)
    if count == 0:
        return np.zeros((0,), dtype=np.int64)
    previous = np.stack([track.last_box if track.last_box is not None else _MISSING_BOX for track in open_tracks])
    if boxes.shape[0] == 0:
        return np.full((count,), -1, dtype=np.int64)
    raw = np.asarray(associate_boxes(previous, boxes, iou_threshold)).reshape(-1)
    if raw.size == count:
        association = raw
    elif raw.size == boxes.shape[0]:
        association = np.full((count,), -1, dtype=np.int64)
        for index_in_frame, slot in enumerate(raw):
            slot = int(slot)
            if 0 <= slot < count and association[slot] < 0:
                association[slot] = index_in_frame
        return association
    else:
        raise ValueError(
            f"associate_boxes returned {raw.size} entries for {count} previous and "
            f"{boxes.shape[0]} current boxes; expected one entry per previous or per current box"
        )
    valid = (association >= 0) & (association < boxes.shape[0])
    return np.where(valid, association, -1).astype(np.int64)


_MISSING_BOX = np.full((4,), np.nan, dtype=np.float32)
