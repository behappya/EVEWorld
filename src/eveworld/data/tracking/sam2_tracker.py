"""Instance tracking with SAM2.

An initial set of boxes is given on the first frame; SAM2 propagates a mask for each box through the
rest of the clip and this module turns those masks into per-frame boxes, visibility flags and object
scores. SAM2 is imported lazily because the box association helper is useful on its own.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["Sam2Tracker", "TrackResult", "associate_boxes"]

CHECKPOINT_ENV_VAR = "SAM2_CHECKPOINT"
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
JPEG_QUALITY = 95
# The papers do not fix a visibility threshold, so the mask area cut-off stays a module default.
MIN_VISIBLE_AREA = 16


@dataclass
class TrackResult:
    """Masks and boxes produced by :class:`Sam2Tracker`.

    Attributes:
        boxes: ``(T, N, 4)`` float32 pixel ``xyxy`` boxes, one per frame and object.
        masks: ``(T, N, H, W)`` bool masks at video resolution.
        visibility: ``(T, N)`` bool flags marking frames where the object is visible.
        scores: ``(T, N)`` float32 object confidences in ``[0, 1]``.
    """

    boxes: np.ndarray
    masks: np.ndarray
    visibility: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32)
        self.masks = np.asarray(self.masks, dtype=bool)
        self.visibility = np.asarray(self.visibility, dtype=bool)
        self.scores = np.asarray(self.scores, dtype=np.float32)

    @property
    def num_frames(self) -> int:
        """Number of frames covered by the tracking result."""
        return int(self.boxes.shape[0])

    @property
    def num_objects(self) -> int:
        """Number of tracked objects."""
        return int(self.boxes.shape[1]) if self.boxes.ndim > 1 else 0

    def to_dict(self) -> dict[str, Any]:
        """Return a metadata view of the result, leaving the arrays out.

        Returns:
            Dictionary with the frame and object counts, plus the visible frame count.
        """
        return {
            "num_frames": self.num_frames,
            "num_objects": self.num_objects,
            "visible": int(self.visibility.sum()),
        }


def associate_boxes(
    previous: np.ndarray, current: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """Match boxes of two consecutive frames by greedy IoU maximisation.

    Candidate pairs are visited from the highest IoU down, ties broken by box order, and every box is
    used at most once, which keeps the mapping deterministic.

    Args:
        previous: ``(N, 4)`` float array of pixel ``xyxy`` boxes from the earlier frame.
        current: ``(M, 4)`` float array of pixel ``xyxy`` boxes from the later frame.
        iou_threshold: Smallest IoU that still counts as a match.

    Returns:
        ``(N,)`` int64 array holding, for every box of ``previous``, the index of its match in
        ``current`` or ``-1`` when it stays unpaired.
    """
    previous_boxes = np.asarray(previous, dtype=np.float32).reshape(-1, 4)
    current_boxes = np.asarray(current, dtype=np.float32).reshape(-1, 4)
    matches = np.full(previous_boxes.shape[0], -1, dtype=np.int64)
    if previous_boxes.shape[0] == 0 or current_boxes.shape[0] == 0:
        return matches
    overlap = _iou_matrix(previous_boxes, current_boxes)
    candidates: list[tuple[float, int, int]] = []
    for row in range(previous_boxes.shape[0]):
        for column in range(current_boxes.shape[0]):
            value = float(overlap[row, column])
            if value >= float(iou_threshold):
                candidates.append((-value, row, column))
    candidates.sort()
    taken_rows: set[int] = set()
    taken_columns: set[int] = set()
    for _, row, column in candidates:
        if row in taken_rows or column in taken_columns:
            continue
        taken_rows.add(row)
        taken_columns.add(column)
        matches[row] = column
    return matches


class Sam2Tracker:
    """Track an initial set of boxes through a clip with the SAM2 video predictor.

    Args:
        checkpoint: Path to the SAM2 checkpoint; falls back to ``SAM2_CHECKPOINT``.
        model_cfg: Hydra config name resolved inside the ``sam2`` package.
        device: Torch device string; a CUDA request without a visible GPU falls back to CPU.

    Raises:
        RuntimeError: If no checkpoint can be resolved.
        ImportError: If the ``sam2`` package is not installed.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        model_cfg: str = DEFAULT_MODEL_CFG,
        device: str = "cuda",
    ) -> None:
        resolved = str(checkpoint) if checkpoint else os.environ.get(CHECKPOINT_ENV_VAR)
        if not resolved:
            raise RuntimeError(
                "A SAM2 checkpoint is required; pass it explicitly or set the "
                f"{CHECKPOINT_ENV_VAR} environment variable."
            )
        if not Path(resolved).is_file():
            raise FileNotFoundError(f"SAM2 checkpoint {resolved!r} is not an existing file")

        self.checkpoint = resolved
        self.model_cfg = str(model_cfg)
        self.device = _resolve_device(device)

        build_sam2_video_predictor = _load_backend()
        self.predictor = build_sam2_video_predictor(
            self.model_cfg,
            self.checkpoint,
            device=self.device,
            apply_postprocessing=False,
        )
        logger.info("SAM2 video predictor ready on %s (%s)", self.device, self.model_cfg)

    def track(self, frames: Any, boxes0: np.ndarray) -> TrackResult:
        """Propagate ``boxes0`` from the first frame through the whole clip.

        Args:
            frames: Frames as ``(T, H, W, 3)`` uint8 RGB, or ``(C, T, H, W)`` floats in ``[-1, 1]``.
            boxes0: ``(N, 4)`` float array of pixel ``xyxy`` boxes drawn on frame 0.

        Returns:
            A :class:`TrackResult` covering every frame of the clip.

        Raises:
            ValueError: If the clip is empty or ``boxes0`` holds no box.
        """
        video = _as_uint8_frames(frames)
        num_frames, height, width = (
            int(video.shape[0]),
            int(video.shape[1]),
            int(video.shape[2]),
        )
        if num_frames == 0:
            raise ValueError("cannot track an empty clip")
        initial = np.asarray(boxes0, dtype=np.float32).reshape(-1, 4)
        num_objects = int(initial.shape[0])
        if num_objects == 0:
            raise ValueError("boxes0 does not contain any box to track")

        boxes = np.repeat(initial[None], num_frames, axis=0).astype(np.float32)
        masks = np.zeros((num_frames, num_objects, height, width), dtype=bool)
        visibility = np.zeros((num_frames, num_objects), dtype=bool)
        scores = np.zeros((num_frames, num_objects), dtype=np.float32)

        with tempfile.TemporaryDirectory(prefix="eveworld-sam2-") as directory:
            _write_jpeg_frames(video, directory)
            state = self.predictor.init_state(
                directory,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            for obj_id, box in enumerate(initial, start=1):
                self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=int(obj_id),
                    box=box,
                )
            for frame_index, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                index = int(frame_index)
                if index < 0 or index >= num_frames:
                    continue
                for position, obj_id in enumerate(obj_ids):
                    slot = int(obj_id) - 1
                    if slot < 0 or slot >= num_objects:
                        continue
                    mask = _mask_from_logits(mask_logits[position], (height, width))
                    masks[index, slot] = mask
                    visibility[index, slot] = bool(int(mask.sum()) >= MIN_VISIBLE_AREA)
                    box = _box_from_mask(mask)
                    if box is None:
                        box = boxes[index - 1, slot] if index > 0 else initial[slot]
                    boxes[index, slot] = box
                    scores[index, slot] = _object_score(state, int(obj_id), index)
        return TrackResult(boxes=boxes, masks=masks, visibility=visibility, scores=scores)


def _iou_matrix(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the ``(N, M)`` pairwise IoU of two sets of ``xyxy`` boxes."""
    left = np.maximum(previous[:, None, 0], current[None, :, 0])
    top = np.maximum(previous[:, None, 1], current[None, :, 1])
    right = np.minimum(previous[:, None, 2], current[None, :, 2])
    bottom = np.minimum(previous[:, None, 3], current[None, :, 3])
    intersection = np.clip(right - left, 0.0, None) * np.clip(bottom - top, 0.0, None)
    previous_area = np.clip(previous[:, 2] - previous[:, 0], 0.0, None) * np.clip(
        previous[:, 3] - previous[:, 1], 0.0, None
    )
    current_area = np.clip(current[:, 2] - current[:, 0], 0.0, None) * np.clip(
        current[:, 3] - current[:, 1], 0.0, None
    )
    union = previous_area[:, None] + current_area[None, :] - intersection
    return np.where(union > 0.0, intersection / np.where(union > 0.0, union, 1.0), 0.0)


def _load_backend() -> Any:
    """Import and return ``build_sam2_video_predictor``.

    Raises:
        ImportError: If the ``sam2`` package is not importable.
    """
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise ImportError(
            "The 'sam2' package is required for instance tracking but is not installed. Install "
            "SAM2 from source (https://github.com/facebookresearch/sam2), download a checkpoint and "
            f"point the {CHECKPOINT_ENV_VAR} environment variable at it."
        ) from exc
    return build_sam2_video_predictor


def _resolve_device(requested: str) -> str:
    """Return ``requested`` unless it asks for CUDA while no GPU is visible.

    Args:
        requested: Device string handed to the tracker.

    Returns:
        A device string that torch can address on this machine.
    """
    device = str(requested)
    try:
        import torch
    except ImportError:
        logger.warning("torch is not installed; running SAM2 setup for %r", device)
        return device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("device %r was requested but CUDA is not available; using 'cpu'", device)
        return "cpu"
    return device


def _write_jpeg_frames(frames: np.ndarray, directory: str) -> None:
    """Write ``(T, H, W, 3)`` uint8 RGB frames into ``directory`` as sorted JPEG files.

    The SAM2 image-sequence loader sorts frames by the integer stem of their names, so zero padded
    indices keep the temporal order intact.
    """
    from PIL import Image

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(target / f"{index:05d}.jpg", quality=JPEG_QUALITY)


def _mask_from_logits(logits: Any, shape: tuple[int, int]) -> np.ndarray:
    """Threshold per-object mask logits into a bool mask of ``shape``.

    Args:
        logits: Logits of a single object; the trailing axes hold the mask resolution.
        shape: Target ``(height, width)`` the mask must take.

    Returns:
        ``(height, width)`` bool mask.

    Raises:
        ValueError: If the logits do not describe one two-dimensional mask.
    """
    height, width = int(shape[0]), int(shape[1])
    values = np.squeeze(np.asarray(_numpy(logits), dtype=np.float32))
    if values.ndim != 2:
        raise ValueError(f"expected a single 2D mask, got logits shaped {values.shape}")
    if values.shape != (height, width):
        values = _nearest_resize(values, height, width)
    return values > 0.0


def _nearest_resize(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resample a 2D array to ``(height, width)`` by picking the nearest source cell."""
    rows = np.floor((np.arange(height) + 0.5) * values.shape[0] / height).astype(np.int64)
    columns = np.floor((np.arange(width) + 0.5) * values.shape[1] / width).astype(np.int64)
    rows = np.clip(rows, 0, values.shape[0] - 1)
    columns = np.clip(columns, 0, values.shape[1] - 1)
    return np.asarray(values)[rows][:, columns]


def _box_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Return the tight pixel ``xyxy`` box of ``mask``, or ``None`` when it is empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    return np.asarray(
        [columns[0], rows[0], columns[-1], rows[-1]],
        dtype=np.float32,
    )


def _object_score(state: dict[str, Any], obj_id: int, frame_index: int) -> float:
    """Return the SAM2 object confidence for one frame as a probability.

    The score sits under the private per-object output dictionary of the inference state; when it
    cannot be read the track is trusted, i.e. ``1.0`` is returned.

    Args:
        state: Inference state returned by ``init_state``.
        obj_id: Object identifier used when the box was registered.
        frame_index: Frame the score is requested for.

    Returns:
        Confidence in ``[0, 1]``.
    """
    try:
        position = int(state["obj_id_to_idx"][int(obj_id)])
        outputs = state["output_dict_per_obj"][position]
        entry = outputs["cond_frame_outputs"].get(int(frame_index))
        if entry is None:
            entry = outputs["non_cond_frame_outputs"].get(int(frame_index))
        if entry is None:
            return 1.0
        logit = float(np.asarray(_numpy(entry["object_score_logits"]), dtype=np.float32).reshape(-1)[0])
    except (KeyError, IndexError, TypeError, ValueError):
        return 1.0
    return float(1.0 / (1.0 + float(np.exp(-np.clip(logit, -30.0, 30.0)))))


def _numpy(value: Any) -> np.ndarray:
    """Convert tensors, lists and scalars into numpy arrays."""
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _as_uint8_frames(frames: Any) -> np.ndarray:
    """Return frames as a contiguous ``(T, H, W, 3)`` uint8 RGB array.

    Args:
        frames: ``(T, H, W, 3)`` uint8 RGB, ``(H, W, 3)`` uint8 RGB or ``(C, T, H, W)`` floats.

    Returns:
        The frames in ``(T, H, W, 3)`` uint8 layout.

    Raises:
        ValueError: If the input rank or channel layout is not one of the supported forms.
    """
    array = _numpy(frames)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(
            f"expected frames shaped (T, H, W, 3), (H, W, 3) or (C, T, H, W), got {array.shape}"
        )
    if array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 3, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"expected three colour channels, got frames shaped {array.shape}")
    if array.dtype != np.uint8:
        array = _to_uint8(array)
    return np.ascontiguousarray(array)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Map float frames that are either in ``[-1, 1]`` or in ``[0, 255]`` onto uint8."""
    values = array.astype(np.float32)
    if values.size:
        if float(values.min()) < 0.0:
            values = (values + 1.0) / 2.0
        elif float(values.max()) > 1.0 + 1e-3:
            values = values / 255.0
    return np.clip(np.rint(values * 255.0), 0.0, 255.0).astype(np.uint8)
