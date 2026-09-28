"""Instance counting for the MLR instrument.

The instrument counts how many instances of the instructed object are visible on
every sampled timestamp of a generated clip, and compares that count with the
number the annotation of the reference clip expects. Two detection passes are
needed per timestamp: the movable object with one prompt and the robot gripper
with another, because a gripper holding an object is the most common reason a
detector reports a spurious duplicate.

The second pass is what makes the counts usable. A detection that overlaps the
gripper box by more than :data:`DEFAULT_GRIPPER_OVERLAP_THRESHOLD` of its own
area is the held object seen again, and a detection whose centre is not strictly
farther than :data:`DEFAULT_MIN_CENTER_DISTANCE` pixels (L1) from a
better-scoring detection is the same instance reported twice; neither is counted
as a separate instance.

Grounding-DINO itself is reached through :class:`InstanceCounter`, which imports
``eveworld.data.grounding.grounding_dino`` lazily so that this module imports in
environments without the evaluation extras.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_BOX_AREA_RATIO",
    "DEFAULT_GRIPPER_OVERLAP_THRESHOLD",
    "DEFAULT_MIN_CENTER_DISTANCE",
    "DEFAULT_OBJECT_BOX_THRESHOLD",
    "DEFAULT_OBJECT_TOPK",
    "DEFAULT_ROBOT_BOX_THRESHOLD",
    "DEFAULT_ROBOT_TOPK",
    "DEFAULT_TEXT_THRESHOLD",
    "ROBOT_PROMPT",
    "VAGUE_CONTAINER_WORDS",
    "InstanceCounter",
    "box_area_ratio",
    "count_instances",
    "count_valid_instances",
    "expected_count",
    "gate_off_reasons",
]

ROBOT_PROMPT = "robot gripper"
DEFAULT_OBJECT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.15
DEFAULT_ROBOT_BOX_THRESHOLD = 0.15
DEFAULT_OBJECT_TOPK = 6
DEFAULT_ROBOT_TOPK = 3
DEFAULT_MIN_CENTER_DISTANCE = 60.0
DEFAULT_GRIPPER_OVERLAP_THRESHOLD = 0.35
DEFAULT_BOX_AREA_RATIO = 0.35
VAGUE_CONTAINER_WORDS = ("table", "desk", "surface", "floor")

_GATE_NO_B = "no_B_detected"
_GATE_VAGUE = "vague_container_word"
_GATE_TOO_LARGE = "B_box_too_large"


def _field(detection: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a detection mapping or object, falling back to a default."""
    if isinstance(detection, Mapping):
        return detection.get(name, default)
    return getattr(detection, name, default)


def _label(detection: Any) -> str | None:
    """Return the label of a detection, or ``None`` when it does not carry one."""
    value = _field(detection, "label")
    if value is None:
        value = _field(detection, "text")
    return None if value is None else str(value)


def _score(detection: Any) -> float | None:
    """Return the detection score, or ``None`` when it does not carry one."""
    value = _field(detection, "score")
    if value is None:
        value = _field(detection, "logit")
    return None if value is None else float(value)


def _box(detection: Any) -> tuple[float, float, float, float]:
    """Return the ``(x0, y0, x1, y1)`` box of one detection.

    Four layouts are accepted: the released :class:`~eveworld.data.grounding.grounding_dino.
    Detection` object with a ``box`` field, a mapping with a ``box`` key, the earlier
    ``(cx, cy, box, score)`` tuple whose third field is the corner box itself, and a bare
    ``(x0, y0, x1, y1)`` sequence.
    """
    box = _field(detection, "box")
    if box is None and isinstance(detection, Sequence) and not isinstance(detection, str):
        legacy = len(detection) == 4 and np.asarray(detection[2]).size == 4
        box = detection[2] if legacy else detection
    if box is None:
        raise ValueError(f"detection without a box: {detection!r}")
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size != 4:
        raise ValueError(f"boxes must have four corners, got {values.size}")
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _center(detection: Any, box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the centre of a detection, preferring an explicit centre field."""
    value = _field(detection, "center")
    if value is None and isinstance(detection, Sequence) and not isinstance(detection, str):
        if len(detection) == 4 and np.asarray(detection[2]).size == 4:
            value = (detection[0], detection[1])
    if value is not None:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if values.size == 2:
            return (float(values[0]), float(values[1]))
    if _field(detection, "cx") is not None and _field(detection, "cy") is not None:
        return (float(_field(detection, "cx")), float(_field(detection, "cy")))
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def _area(box: tuple[float, float, float, float]) -> float:
    """Area of a corner box, guarded against degenerate boxes."""
    return max(0.0, abs(box[2] - box[0])) * max(0.0, abs(box[3] - box[1]))


def _inter_area(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    """Area of the intersection of two corner boxes."""
    left_x0, left_x1 = min(left[0], left[2]), max(left[0], left[2])
    left_y0, left_y1 = min(left[1], left[3]), max(left[1], left[3])
    right_x0, right_x1 = min(right[0], right[2]), max(right[0], right[2])
    right_y0, right_y1 = min(right[1], right[3]), max(right[1], right[3])
    width = min(left_x1, right_x1) - max(left_x0, right_x0)
    height = min(left_y1, right_y1) - max(left_y0, right_y0)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return float(width) * float(height)


def count_instances(detections: Iterable[Any], label: str, threshold: float | None = None) -> int:
    """Count detections whose label matches ``label``.

    Matching is a case-insensitive substring test, so ``"cup"`` matches the
    Grounding-DINO label ``"coffee cup"`` and ``"ROBOT"`` matches
    ``"robot gripper"``. ``threshold`` is optional: when it is given, a
    detection is only counted if its score is at or above it, which is how the
    per-prompt box thresholds of the protocol are re-applied to an already
    detected frame. Detections without a label are ignored.

    Args:
        detections: iterable of detections, each a mapping or an object
            carrying ``label`` and, for thresholded counts, ``score``.
        label: label to match, case-insensitive, as a substring.
        threshold: optional minimum score.

    Returns:
        The number of matching detections.
    """
    needle = str(label).strip().lower()
    minimum = None if threshold is None else float(threshold)
    count = 0
    for detection in detections:
        text = _label(detection)
        if text is None or needle not in text.strip().lower():
            continue
        if minimum is not None:
            score = _score(detection)
            if score is None or score < minimum:
                continue
        count += 1
    return count


def count_valid_instances(
    obj_dets: Iterable[Any],
    gripper_dets: Iterable[Any],
    *,
    min_dist: float = DEFAULT_MIN_CENTER_DISTANCE,
    grip_overlap_thr: float = DEFAULT_GRIPPER_OVERLAP_THRESHOLD,
) -> int:
    """Count the independent instances among the object detections.

    The previous implementation's duplicate rule is reproduced here:

    * a detection is dropped when it overlaps a gripper box by more than
      ``grip_overlap_thr`` of *its own* area, which removes the object that the
      gripper is holding and that the detector reports a second time;
    * a detection is dropped when its centre is not strictly farther than
      ``min_dist`` pixels (L1 distance) from the centre of an already accepted,
      better-scoring detection, which removes the second box of a two-box
      duplicate.

    Detections are processed in descending score order; the accepted centres are
    the ones later detections are compared against. The overlap test is strict,
    so a detection exactly on ``grip_overlap_thr`` is kept; the distance test
    keeps only detections strictly beyond ``min_dist``, so a tie drops the later
    box.

    Args:
        obj_dets: object detections as ``(cx, cy, box, score)`` tuples,
            mappings with ``box``/``score``, or detection objects.
        gripper_dets: gripper detections in any of the same layouts; pass an
            empty sequence when the gripper was not detected.
        min_dist: minimum L1 centre distance in pixels between two instances.
        grip_overlap_thr: maximum gripper overlap, as a fraction of the object
            box area, for a detection to survive.

    Returns:
        The number of independent instances.
    """
    gripper_boxes = [_box(detection) for detection in gripper_dets]
    candidates: list[tuple[float, tuple[float, float], tuple[float, float, float, float]]] = []
    for detection in obj_dets:
        if _score(detection) is None and isinstance(detection, Sequence) and not isinstance(detection, str):
            score = float(detection[3]) if len(detection) == 4 else 0.0
        else:
            score = _score(detection)
        box = _box(detection)
        candidates.append((float(score), _center(detection, box), box))

    overlap_limit = float(grip_overlap_thr)
    distance = float(min_dist)
    kept: list[tuple[float, float]] = []
    for _score_value, center, box in sorted(candidates, key=lambda item: item[0], reverse=True):
        box_area = max(1.0, _area(box))
        if any(_inter_area(box, gripper) / box_area > overlap_limit for gripper in gripper_boxes):
            continue
        if all(abs(center[0] - other[0]) + abs(center[1] - other[1]) > distance for other in kept):
            kept.append(center)
    return len(kept)


def box_area_ratio(box: Sequence[float], frame_shape: Sequence[int] | None = None) -> float:
    """Area of ``box`` as a fraction of the frame.

    Args:
        box: ``(x0, y0, x1, y1)`` corners, either in pixels with ``frame_shape``
            given, or normalised to ``[0, 1]`` when it is not.
        frame_shape: optional ``(height, width)`` of the frame in pixels.

    Returns:
        The box area divided by the frame area, ``0.0`` for a degenerate box.
    """
    x0, y0, x1, y1 = (float(value) for value in np.asarray(box, dtype=np.float64).reshape(-1)[:4])
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    if frame_shape is None:
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.0:
            raise ValueError("pixel boxes need frame_shape to give an area ratio")
        return float(width * height)
    shape = np.asarray(frame_shape, dtype=np.float64).reshape(-1)
    if shape.size < 2:
        raise ValueError(f"frame_shape must be (height, width), got {frame_shape!r}")
    return float(width * height / max(1.0, float(shape[0]) * float(shape[1])))


def _cell_in(cell: Any, cells: Any) -> bool:
    """Whether a ``(row, col)`` cell appears in a list of cells."""
    if cell is None or cells is None:
        return False
    target = tuple(int(value) for value in np.asarray(cell, dtype=np.int64).reshape(-1)[:2])
    for candidate in cells:
        if candidate is None:
            continue
        other = tuple(int(value) for value in np.asarray(candidate, dtype=np.int64).reshape(-1)[:2])
        if other == target:
            return True
    return False


def _at(sequence: Sequence[Any], index: int) -> Any:
    """Index a sequence like Python does, returning ``None`` when out of range."""
    values = list(sequence)
    position = int(index)
    if position < 0:
        position += len(values)
    if position < 0 or position >= len(values):
        return None
    return values[position]


def expected_count(metadata: Mapping[str, Any], timestamp: int) -> int | None:
    """Number of target instances the annotation expects to be visible at ``timestamp``.

    The annotation of §8.3 describes one reference clip: ``inventory_cells``
    holds the cells of the target instances seen in the conditioning frame, and
    ``per_lat_frame[t]`` holds ``target_cell`` together with the container cells
    ``b_cells`` of that latent timestamp. The expected count is the inventory
    size whenever the annotation says the target is somewhere the count is
    meaningful, and ``None`` otherwise:

    * the metadata has no annotation, or no frame at that index -> ``None``;
    * ``target_cell`` is ``None`` (the annotator lost the target) -> ``None``;
    * the inventory is empty, so the clip asks for no instance -> ``None``;
    * ``target_cell`` lies inside ``b_cells``, i.e. the target has already been
      placed inside the container, where its visibility is not constrained by
      the reference -> ``None``.

    ``metadata`` is either the annotation itself or the mapping produced by
    :func:`eveworld.evaluation.mlr.merge.build_metadata`, whose
    ``expected_counts`` and ``instances`` fields are read as fallbacks.

    Args:
        metadata: annotation mapping, or a ``build_metadata`` result.
        timestamp: sampled latent index; negative values count from the end.

    Returns:
        The expected number of visible instances, or ``None``.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")

    per_frame = metadata.get("per_lat_frame")
    if per_frame is None:
        counts = metadata.get("expected_counts")
        if counts is not None:
            entry = _at(counts, timestamp)
            return None if entry is None else int(entry)
        instances = metadata.get("instances")
        if instances is not None:
            entry = _at(instances, timestamp)
            if entry is None:
                return None
            detections = entry.get("detections") if isinstance(entry, Mapping) else entry
            return None if detections is None else len(list(detections))
        return None

    frame = _at(per_frame, timestamp)
    if not isinstance(frame, Mapping):
        return None
    target_cell = frame.get("target_cell", frame.get("target"))
    if target_cell is None:
        return None
    inventory = metadata.get("inventory_cells")
    total = metadata.get("n_inventory")
    if total is None:
        total = len(inventory) if inventory is not None else 0
    if int(total) <= 0:
        return None
    if _cell_in(target_cell, frame.get("b_cells")):
        return None
    return int(total)


def _gate_box_ratio(entry: Any, frame_shape: Sequence[int] | None) -> float | None:
    """Area ratio of one container entry; ``None`` for a bare ``(row, col)`` cell."""
    if not isinstance(entry, Mapping) and not hasattr(entry, "box"):
        values = np.asarray(entry, dtype=np.float64).reshape(-1)
        if values.size == 2:
            return None
    return box_area_ratio(_box(entry), frame_shape)


def gate_off_reasons(
    record: Mapping[str, Any] | None = None,
    *,
    b_boxes: Sequence[Any] | None = None,
    container_word: str | None = None,
    frame_shape: Sequence[int] | None = None,
    box_area_ratio_threshold: float = DEFAULT_BOX_AREA_RATIO,
) -> list[str]:
    """Reasons the count gate has to be switched off for a clip.

    The gate decides whether the detector's counts carry information about the
    instructed object at all. The previous implementation switched it off with
    ``no_B_detected`` when the destination container was not found, with
    ``vague_container_word`` when the container name was one of
    :data:`VAGUE_CONTAINER_WORDS`, and with ``B_box_too_large`` when the
    container box covered more than ``box_area_ratio_threshold`` of the frame;
    the third reason is checked on every detected container box.

    ``record`` may be an annotation mapping carrying ``b_name`` and
    ``b_cells``/``b_boxes``; the keyword arguments override whatever the record
    holds.

    Args:
        record: optional annotation mapping (and the override source).
        b_boxes: container boxes as ``(x0, y0, x1, y1)``, pixel or normalised.
        container_word: the container name from the instruction, e.g. ``"bowl"``.
        frame_shape: ``(height, width)`` in pixels, needed for pixel boxes.
        box_area_ratio_threshold: area ratio above which the container box is
            considered too large.

    Returns:
        The list of gate reasons in check order, empty when the gate stays on.
    """
    boxes: Any = b_boxes
    word = container_word
    shape = frame_shape
    if isinstance(record, Mapping):
        if boxes is None:
            boxes = record.get("b_boxes")
        if boxes is None:
            boxes = record.get("b_det0")
        if boxes is None:
            boxes = record.get("b_cells")
        if word is None:
            word = record.get("b_name")
            if word is None:
                word = record.get("container_word")
        if shape is None:
            shape = record.get("frame_shape")
            if shape is None:
                height, width = record.get("H"), record.get("W")
                if height is not None and width is not None:
                    shape = (height, width)

    reasons: list[str] = []
    entries = [] if boxes is None else [entry for entry in boxes if entry is not None]
    if not entries:
        reasons.append(_GATE_NO_B)
    if word is not None and any(vague in str(word).lower() for vague in VAGUE_CONTAINER_WORDS):
        reasons.append(_GATE_VAGUE)
    for entry in entries:
        try:
            ratio = _gate_box_ratio(entry, shape)
        except (ValueError, TypeError):
            logger.debug("container box %r has no usable geometry", entry)
            continue
        if ratio is not None and ratio > float(box_area_ratio_threshold):
            reasons.append(_GATE_TOO_LARGE)
            break
    return reasons


class InstanceCounter:
    """Grounding-DINO wrapper that counts instructed instances per frame.

    The backend is imported on first use, inside the class, so an environment
    without the evaluation extras can still import the MLR package; constructing
    the counter never touches torch. ``detector`` may be passed explicitly, which
    is what the tests use to avoid the model download.

    Args:
        detector: optional backend, either a callable ``(frames, prompt)`` or an
            object with a ``detect(frames, prompt)`` method.
        config_path: Grounding-DINO config for the default backend.
        weights_path: Grounding-DINO checkpoint for the default backend.
        device: torch device for the default backend.
        box_threshold: object box threshold of the protocol.
        text_threshold: text threshold of the protocol.
        robot_prompt: prompt used for the gripper pass.
        robot_topk: number of gripper boxes kept per frame.
        object_topk: number of object boxes kept per frame.
        robot_box_threshold: box threshold of the gripper pass.
        min_dist: minimum centre distance passed to :func:`count_valid_instances`.
        grip_overlap_thr: gripper overlap fraction passed to
            :func:`count_valid_instances`.
    """

    def __init__(
        self,
        detector: Any | None = None,
        *,
        config_path: str | None = None,
        weights_path: str | None = None,
        device: str = "cuda",
        box_threshold: float = DEFAULT_OBJECT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        robot_prompt: str = ROBOT_PROMPT,
        robot_topk: int = DEFAULT_ROBOT_TOPK,
        object_topk: int = DEFAULT_OBJECT_TOPK,
        robot_box_threshold: float = DEFAULT_ROBOT_BOX_THRESHOLD,
        min_dist: float = DEFAULT_MIN_CENTER_DISTANCE,
        grip_overlap_thr: float = DEFAULT_GRIPPER_OVERLAP_THRESHOLD,
    ) -> None:
        self._detector = detector
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.robot_prompt = robot_prompt
        self.robot_topk = int(robot_topk)
        self.object_topk = int(object_topk)
        self.robot_box_threshold = float(robot_box_threshold)
        self.min_dist = float(min_dist)
        self.grip_overlap_thr = float(grip_overlap_thr)

    @property
    def detector(self) -> Any:
        """The detection backend, built on first access."""
        if self._detector is None:
            from eveworld.data.grounding.grounding_dino import GroundingDinoDetector

            if self.config_path is None and self.weights_path is None:
                self._detector = GroundingDinoDetector.from_env(
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device,
                )
            else:
                self._detector = GroundingDinoDetector(
                    config_path=self.config_path,
                    weights_path=self.weights_path,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device,
                )
        return self._detector

    def detect(
        self,
        frames: Any,
        prompt: str,
        *,
        topk: int | None = None,
        box_threshold: float | None = None,
    ) -> list[list[Any]]:
        """Detect ``prompt`` on one frame or on a batch of frames.

        ``frames`` is either a single ``(H, W, 3)`` image or a ``(T, H, W, 3)``
        clip; the result is one detection list per frame, each sorted by
        descending score and truncated to ``topk``, which defaults to
        :attr:`object_topk`. A ``box_threshold`` given here re-filters the
        detections of the backend.
        """
        if topk is None:
            topk = self.object_topk
        array = np.asarray(frames)
        single = array.ndim == 3
        batch = [array] if single else list(array)
        backend = self.detector
        if hasattr(backend, "detect"):
            results = backend.detect(batch, prompt)
        elif callable(backend):
            results = backend(batch, prompt)
        else:
            raise TypeError(f"unsupported detection backend: {type(backend).__name__}")
        if isinstance(results, np.ndarray):
            results = list(results)
        elif results and not isinstance(results[0], (list, tuple, np.ndarray)):
            results = [results]

        minimum = None if box_threshold is None else float(box_threshold)
        output: list[list[Any]] = []
        for detections in results:
            kept = list(detections)
            if minimum is not None:
                kept = [det for det in kept if (_score(det) or 0.0) >= minimum]
            if topk is not None and int(topk) > 0:
                kept = sorted(kept, key=lambda det: _score(det) or 0.0, reverse=True)[: int(topk)]
            output.append(kept)
        return output

    def count(
        self,
        frame: Any,
        prompt: str,
        *,
        topk: int | None = None,
        box_threshold: float | None = None,
    ) -> int:
        """Count the independent instances of ``prompt`` on a single frame."""
        object_topk = self.object_topk if topk is None else int(topk)
        obj_dets = self.detect(frame, prompt, topk=object_topk, box_threshold=box_threshold)[0]
        robot_dets = self.detect(frame, self.robot_prompt, topk=self.robot_topk, box_threshold=self.robot_box_threshold)[0]
        return count_valid_instances(
            obj_dets,
            robot_dets,
            min_dist=self.min_dist,
            grip_overlap_thr=self.grip_overlap_thr,
        )

    def counts(self, frames: Any, prompt: str, *, box_threshold: float | None = None) -> np.ndarray:
        """Count the independent instances of ``prompt`` on every frame.

        Args:
            frames: ``(T, H, W, 3)`` clip, or a single ``(H, W, 3)`` frame.
            prompt: object prompt, e.g. the target name of the instruction.
            box_threshold: optional override of the object box threshold.

        Returns:
            ``(T,)`` int64 counts, one per input frame.
        """
        obj_dets = self.detect(frames, prompt, box_threshold=box_threshold)
        robot_dets = self.detect(frames, self.robot_prompt, topk=self.robot_topk, box_threshold=self.robot_box_threshold)
        if len(robot_dets) == 1 and len(obj_dets) > 1:
            robot_dets = robot_dets * len(obj_dets)
        counts = [
            count_valid_instances(obj, robot, min_dist=self.min_dist, grip_overlap_thr=self.grip_overlap_thr)
            for obj, robot in zip(obj_dets, robot_dets)
        ]
        return np.asarray(counts, dtype=np.int64)
